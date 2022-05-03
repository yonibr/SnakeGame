# TODO:
#    - Make walls push snake (except if head hits wall head on) and food

import itertools
import numpy as np

from collections import Counter, defaultdict, deque as queue
from math import copysign
from typing import (
    Callable,
    Dict,
    Iterator,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union
)

import state


rng = np.random.default_rng()


def rand_pos(w: int, h: int, buffer: int=1) -> Tuple[int, int]:
    return tuple(rng.integers(buffer, [w - buffer, h - buffer], size=2))


class Direction(object):
    def __init__(self, offset_x: int, offset_y: int, name: str):
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.name = name

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, type(self)) and
            self.offset_x == other.offset_x and
            self.offset_y == other.offset_y
        )

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self):
        return hash((self.offset_x, self.offset_y))

    def is_opposite(self, other) -> bool:
        return (
            isinstance(other, type(self)) and
            self != other and
            abs(self.offset_x) == abs(other.offset_x) and
            abs(self.offset_y) == abs(other.offset_y)
        )

    @classmethod
    def left(cls) -> 'Direction':
        return Direction(-1, 0, 'left')

    @classmethod
    def right(cls) -> 'Direction':
        return Direction(1, 0, 'right')

    @classmethod
    def up(cls) -> 'Direction':
        return Direction(0, -1, 'up')

    @classmethod
    def down(cls) -> 'Direction':
        return Direction(0, 1, 'down')


class Node(object):
    def __init__(self, x: int, y: int, marker: str):
        self.x = x
        self.y = y
        self.marker = marker

    def __contains__(self, other) -> bool:
        if isinstance(other, Node):
            return (other.x, other.y) == (self.x, self.y)
        # Otherwise, assume tuple
        return other == (self.x, self.y)

    def __hash__(self) -> int:
        return hash(self.to_tuple())

    def __repr__(self) -> str:
        return f'Node{self.to_tuple()}'

    def __eq__(self, other) -> bool:
        if isinstance(other, Node):
            return self.to_tuple() == other.to_tuple()
        return False

    @property
    def pos(self) -> Tuple[int, int]:
        return self.x, self.y

    def to_tuple(self) -> Tuple[int, int, str]:
        return (self.x, self.y, self.marker)

    def move(self, direction: Direction) -> None:
        self.x += direction.offset_x
        self.y += direction.offset_y

    def copy(self):
        return Node(self.x, self.y, self.marker)


class NodeCollection(object):
    def __init__(self, positions: Iterable[Tuple[int, int]], markers: Sequence[str]):
        self.nodes = [Node(*pos, marker) for pos, marker in zip(positions, markers)]

    def __contains__(self, other) -> bool:
        return any(other in n for n in self)

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self) -> Iterator[Node]:
        return iter(self.nodes)

    def __getitem__(self, item: Union[int, slice]) -> Node:
        return self.nodes[item]


class Snake(NodeCollection):
    def __init__(self, start_x: int, start_y: int, start_length: int):
        self.head = Node(start_x, start_y, '@')
        self.nodes = [self.head]
        self.node_queue = []
        self.direction = Direction.right()

        # TODO: pathfinding to place snake not in a straight line if necessary
        for i in range(1, start_length):
            self.nodes.append(Node(start_x - i, start_y, '#'))

    def __len__(self) -> int:
        return super().__len__() + len(self.node_queue)

    # TO DO: maybe add option for food to not be added until the end of the tail reaches where the food was eaten
    def move(self) -> None:
        last_pos = self.head.x, self.head.y

        # Move head
        self.head.move(self.direction)

        # Add food to the end if needed
        if len(self.node_queue) > 0:
            new_node = self.node_queue.pop()
            new_node.marker = '#'
            self.nodes.append(new_node)

        # Move tail
        for node in self.nodes[1:]:
            curr_pos = node.x, node.y
            node.x, node.y = last_pos
            last_pos = curr_pos


class Wall(NodeCollection):
    def __init__(self, x: int, y: int, width: int, height: int):
        positions = itertools.product(range(x, x + width), range(y, y + height))
        markers = '█' * width * height
        super().__init__(positions, markers)

    def move(self, direction: Direction) -> None:
        for node in self:
            node.move(direction)


class Level(object):
    def __init__(
            self, walls: Mapping[str, Wall], board_width: int, board_height: int,
            tick_func: Optional[Callable[[Mapping[str, Wall], int], None]]=None):
        self.walls = walls
        self.board_width = board_width
        self.board_height = board_height
        self.tick_func = tick_func
        self.tick_counter = 0

        self.mask = self.compute_mask()

    def __len__(self) -> int:
        return np.sum(self.mask)

    def tick(self) -> None:
        self.tick_counter += 1
        if self.tick_func:
            self.tick_func(self.walls, self.tick_counter)

    def in_wall(self, to_check: Union[Node, Tuple[int, int]]) -> bool:
        if isinstance(to_check, Node):
            to_check = to_check.to_tuple()
        if 0 <= to_check[0] < len(self.mask[0]) and 0 <= to_check[1] < len(self.mask):
            return self.mask[to_check[1]][to_check[0]]
        return False

    def get_neighbors(self, x, y):
        is_valid = lambda nx, ny: 0 <= nx < self.board_width and 0 <= ny < self.board_height
        # Direction vectors
        d_row = [-1, 0, 1, 0]
        d_col = [0, 1, 0, -1]

        neighbors = []
        for i in range(4):
            adj_x = x + d_row[i]
            adj_y = y + d_col[i]
            if is_valid(adj_x, adj_y):
                neighbors.append((adj_x, adj_y))
        return neighbors

    def compute_mask(self, cells=None):
        if not cells:
            cells = [[False] * self.board_width for _ in range(self.board_height)]
            for node in self.wall_nodes:
                cells[node.y][node.x] = True

        cell_classes = [[None] * len(cells[0]) for _ in cells]
        to_check = set((x, y) for x in range(len(cells[0])) for y in range(len(cells)) if not cells[y][x])
        is_valid = lambda x, y: 0 <= x < len(cells[0]) and 0 <= y < len(cells) and (adjx, adjy) in to_check

        # Direction vectors
        d_row = [-1, 0, 1, 0]
        d_col = [0, 1, 0, -1]

        class_idx = 0
        class_counts = defaultdict(int)
        # BFS to find all areas
        while len(to_check) > 0:
            row, col = to_check.pop()

            q = queue()
            q.append((row, col))

            while len(q):
                x, y = q.popleft()
                cell_classes[y][x] = class_idx
                class_counts[class_idx] += 1

                # Go to the adjacent cells
                for adjx, adjy in self.get_neighbors(x, y):
                    if is_valid(adjx, adjy):
                        q.append((adjx, adjy))
                        to_check.remove((adjx, adjy))

            class_idx += 1

        # Find the biggest area 
        max_class = max(class_counts, key=class_counts.get)

        mask = [
            [
                cell_classes[y][x] != max_class for x in range(len(cells[0]))
            ] for y in range(len(cells))
        ]
        
        return mask

    def is_valid_level(self) -> bool:
        mask = self.mask
        width, height = self.board_width, self.board_height

        def get_cell_accessibility(m):
            accessibility = [[0] * width for _ in range(height)]
            for x, y in itertools.product(range(len(m[0])), range(len(m))):
                if not m[y][x]:
                    for nx, ny in self.get_neighbors(x, y):
                        accessibility[y][x] += not m[ny][nx]
            return accessibility

        cell_accessibility = get_cell_accessibility(mask)

        to_check2 = []
        to_check3 = []
        to_check4 = []
        fully_accessible = True
        for x, y in itertools.product(range(width), range(height)):
            if cell_accessibility[y][x] == 1:
                fully_accessible = False
                break
            elif cell_accessibility[y][x] == 2:
                to_check2.append((x, y))
            elif cell_accessibility[y][x] == 3:
                to_check3.append((x, y))
            elif cell_accessibility[y][x] == 4:
                to_check4.append((x, y))

        # First we want to check cells with accessibility of 2, then 3, then 4. This
        # order is because it is more likely that a blockage will occurr when the
        # accessibility is 2 vs 3 and 3 vs 4, and we want to check as few cells as possible.
        if fully_accessible:
            for x, y in itertools.chain(to_check2, to_check3, to_check4):
                modified_mask = [list(row) for row in mask]
                modified_mask[y][x] = True
                modified_mask = self.compute_mask(cells=modified_mask)
                if np.sum(modified_mask) - 1 > len(self):
                    fully_accessible = False
                    break

        return fully_accessible

    @property
    def wall_nodes(self) -> List[Node]:
        return [node for wall in self.walls.values() for node in wall]

    @classmethod
    def outer_walls(cls, board_width: int, board_height: int) -> Dict[str, Wall]:
        return {
            'outer_top': Wall(0, 0, board_width, 1),
            'outer_bottom': Wall(0, board_height - 1, board_width, 1),
            'outer_left': Wall(0, 1, 1, board_height - 2),
            'outer_right': Wall(board_width - 1, 1, 1, board_height - 2)
        }

    @classmethod
    def Basic(cls, board_width: int, board_height: int) -> 'Level':
        return Level(cls.outer_walls(board_width, board_height), board_width, board_height)

    @classmethod
    def MovingVertWall(cls, board_width: int, board_height: int) -> 'Level':
        max_center_offset = board_width // 4
        def tick_func(walls, tick_counter):
            wall = walls['moving_wall']
            wall_x = wall.nodes[0].x
            cycle_sign = copysign(1, tick_counter % (4 * max_center_offset) - max_center_offset * 2)
            if cycle_sign < 0:
                direction = Direction.left()
            else:
                direction = Direction.right()

            if abs(board_width // 2 - (wall_x + cycle_sign)) < max_center_offset:
                wall.move(direction)

        outer_walls = cls.outer_walls(board_width, board_height)
        moving_wall = Wall(board_width // 2, board_height // 3, 1, board_height // 3)

        return Level({**outer_walls, 'moving_wall': moving_wall}, board_width, board_height, tick_func=tick_func)

    @classmethod 
    def Viewfinder(cls, board_width: int, board_height: int) -> 'Level':
        frame_offset = min(round(board_width / 6), round(board_height / 6))
        frame_width = round(board_width / 5)
        frame_height = round(board_height / 5)
        target_width = target_height = round(min(board_width, board_height) / 10)
        bot_start_y = board_height - frame_offset - 1
        right_start_x = board_width - frame_offset - 1
        middle_x = board_width // 2
        middle_y = board_height // 2

        outer_walls = cls.outer_walls(board_width, board_height)
        frame_walls = {
            'frame_top_left_horz': Wall(frame_offset, frame_offset, frame_width, 1),
            'frame_top_left_vert': Wall(frame_offset, frame_offset + 1, 1, frame_height - 1),
            'frame_bot_left_horz': Wall(frame_offset, bot_start_y, frame_width, 1),
            'frame_bot_left_vert': Wall(frame_offset, bot_start_y - frame_height + 1, 1, frame_height - 1),
            'frame_top_right_horz': Wall(right_start_x - frame_width + 1, frame_offset, frame_width, 1),
            'frame_top_right_vert': Wall(right_start_x, frame_offset + 1, 1, frame_height - 1),
            'frame_bot_right_horz': Wall(right_start_x - frame_width + 1, bot_start_y, frame_width, 1),
            'frame_bot_right_vert': Wall(right_start_x, bot_start_y - frame_height + 1, 1, frame_height - 1)
        }
        target_walls = {
            'target_horz_left': Wall(middle_x - target_width, middle_y, target_width, 1),
            'target_horz_right': Wall(middle_x + 1, middle_y, target_width, 1),
            'target_vert_top': Wall(middle_x, middle_y - target_height, 1, target_height),
            'target_vert_bot': Wall(middle_x, middle_y + 1, 1, target_height),
            'target_center': Wall(middle_x, middle_y, 1, 1)
        }

        return Level({**outer_walls, **frame_walls, **target_walls}, board_width, board_height)

    @classmethod
    def Random(
            cls, board_width: int, board_height: int, width_divisor: int=6, height_divisor: int=6,
            volume_divisor: int=10) -> 'Level':
        max_width = round(board_width / width_divisor)
        max_height = round(board_height / height_divisor)

        outer_walls = cls.outer_walls(board_width, board_height)

        counter = 0
        while True:
            available_volume = round(board_width * board_height / volume_divisor)
            walls = {}
            wall_num = 0
            while available_volume > 0:
                width = rng.integers(max_width) + 1
                height = rng.integers(max_height) + 1
                if width * height <= available_volume:
                    start_x = rng.integers(max(1, board_width - 1 - width)) + 1
                    start_y = rng.integers(max(1, board_height - 1 - height)) + 1
                    walls[f'wall_{wall_num}'] = Wall(start_x, start_y, width, height)
                    available_volume -= width * height
                    wall_num += 1
            level = Level({**outer_walls, **walls}, board_width, board_height)
            if level.is_valid_level():
                break
            counter += 1
            if counter > 100000:
                raise Exception('Failed to create random level without inaccessible portions after 100000 tries.')

        return level

    @classmethod
    def Training(cls, board_width: int, board_height: int) -> 'Level':
        # level_type = rng.choice(['Basic', 'Viewfinder', 'Random'], p=[0.85, 0.15, 0.025])
        # level_type = rng.choice(['Basic', 'Viewfinder'], p=[2 / 3, 1 / 3])
        level_type = 'Basic'
        kw_params = dict()
        if level_type == 'Random':
            wd, hd, vd = rng.integers([5, 5, 9], [9, 9, 15], size=3)
            kw_params = {'width_divisor': wd, 'height_divisor': hd, 'volume_divisor': vd}
        return levels[level_type](board_width, board_height, **kw_params)

    @classmethod
    def Training2(cls, board_width: int, board_height: int) -> 'Level':
        probabilities = state.get_probabilities()
        level_type = rng.choice(['Basic', 'Viewfinder', 'Random'], p=probabilities)
        kw_params = dict()
        if level_type == 'Random':
            wd, hd, vd = rng.integers([5, 5, 9], [9, 9, 15], size=3)
            kw_params = {'width_divisor': wd, 'height_divisor': hd, 'volume_divisor': vd}
        return levels[level_type](board_width, board_height, **kw_params)

    # @classmethod
    # def AsteroidField(cls, board_width: int, board_height: int) -> 'Level':
    #     n_asteroids = round(sqrt(board_width * board_height))

    #     outer_walls = cls.outer_walls(board_width, board_height)
    #     # FIXME make not random
    #     walls = {
    #         f'wall_{i}': Wall(rng.integers(max(1, board_width - 1), rng.integers(max(1, board_height - 1, 1, 1)
    #     }

    #     def tick_func(walls, tick_counter):
    #         pass # TODO

    #     return Level({**outer_walls, **walls}, board_width, board_height, tick_func=tick_func(walls, tick_counter))

    # @classmethod
    # def Pinwheels(cls, board_width: int, board_height: int) -> 'Level':
    #     pinwheel_radius = max(min(4, round(board_width / 10)), 2)
    #     pinwheel_count = round(board_width * board_height / 25)
    #     spacing = 

    #     for i in range(pinwheel_count):
    #         x = 


levels = {
    'Basic': Level.Basic,
    'MovingVertWall': Level.MovingVertWall,
    'Viewfinder': Level.Viewfinder,
    'Random': Level.Random,
    'Training': Level.Training,
    'Training2': Level.Training2
}


class Game(object):
    def __init__(
            self, start_x: int, start_y: int, start_length: int, board_width: int,
            board_height: int, score_multiplier: int,
            level: Union[str, Callable[[int, int], Level]]):
        self.board_width = board_width
        self.board_height = board_height

        if start_x >= board_width:
            start_x = board_width // 2 - 1
        if start_y >= board_height:
            start_y = board_height // 2 - 1

        counter = 0
        self.snake = None
        while not self.snake and counter < 1000:
            self.create_level(level, board_width, board_height)
            self.place_snake(start_x, start_y, start_length)
            counter += 1
        if not self.snake:
            print('Failed to place snake. Please make sure the starting parameters are valid.')
            exit()

        self.place_food()
        self.game_over = False
        self.won = False
        self.start_length = start_length
        self.score_multiplier = score_multiplier

    def __len__(self) -> int:
        return self.board_width * self.board_height

    def __str__(self) -> str:
        board = self.get_board()

        sl = []

        for row in board:
            for cell in row:
                sl.append(cell.marker if cell else ' ')
            sl.append('\n')

        return ''.join(sl)

    def __repr__(self) -> str:
        return str(self)

    @property
    def score(self) -> int:
        return self.score_multiplier * (len(self.snake) - self.start_length)

    @property
    def all_nodes(self) -> List[Node]:
        return [*self.level.wall_nodes, *self.snake.nodes, self.food]

    def create_level(
            self, level: Union[str, Callable[[int, int], Level]], board_width: int,
            board_height: int) -> None:
        if level in levels:
            self.level = levels[level](board_width, board_height)
        else:
            self.level = level(board_width, board_height)

    def place_snake(self, start_x: int, start_y: int, start_length: int) -> None:
        get_all_pos = lambda x, y: [(new_x, y) for new_x in range(x - start_length - 1, x + 3)]

        possible_positions = [
            (x, y) for x, y in itertools.product(range(self.board_width), range(self.board_height))
            if not self.level.in_wall((x, y))
        ]
        np.random.shuffle(possible_positions)

        failed = False
        pos = start_x, start_y
        while any(self.level.in_wall((x, y)) for x, y in get_all_pos(*pos)):
            if len(possible_positions) == 0:
                failed = True
                break
            pos = possible_positions.pop()

        if not failed:
            self.snake = Snake(*pos, start_length)

    def place_food(self) -> None:
        pos = rand_pos(self.board_width, self.board_height)
        while pos in self.snake or self.level.in_wall(pos):
            pos = rand_pos(self.board_width, self.board_height)

        self.food = Node(*pos, '•')

    def handle_eating(self) -> None:
        # If food is in snake, it's because the snake ate it, since place food
        # won't place into the snake
        if self.food in self.snake:
            # old_food = self.food
            self.snake.node_queue.append(self.food)

            if len(self.snake) < len(self) - len(self.level):
                self.place_food()
            else: # The snake has completely filled up the board
                self.game_over = True
                self.won = True

    def handle_collisions(self, which) -> bool:
        did_collide = False
        if which == 'snake' or which == 'all':
            did_collide |= any(
                node is not self.snake.head and node in self.snake.head
                for node in self.snake
            )
        if which == 'walls' or which == 'all':
            did_collide |= self.level.in_wall(self.snake.head)
        
        self.game_over = did_collide
        return did_collide

    def get_board(self) -> List[List[Optional[Node]]]:
        grid = [[None] * self.board_width for _ in range(self.board_height)]
        all_nodes = self.all_nodes

        for node in all_nodes:
            x = min(len(grid[0]) - 1, max(node.x, 0))
            y = min(len(grid) - 1, max(node.y, 0))
            grid[y][x] = node

        return grid

    def tick(self) -> None:
        if not self.game_over:
            self.snake.move()
            self.handle_eating()
            if not self.handle_collisions('all'):
                self.level.tick()
                self.handle_collisions('walls')

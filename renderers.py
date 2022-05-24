# TODO:
#   - Print out high scores in CLRenderer
#   - Make curses renderer work even if curses.has_colors() is false
#   - Make it so scale factor decreases if window size would be to big for the screen
#   - Make font size depend on scale factor
#   - Fix PGRenderer2 not working at very low FPS
#   - Display names in high scores

import itertools
import moderngl
import moderngl_window as mglw
import numpy as np
import os
import platform
import pygame as pg
import pyglet
import subprocess
import time

from abc import ABC, abstractmethod
from collections import defaultdict
from enum import auto, Enum
from moderngl_window import geometry as geom
from moderngl_window.timers.clock import Timer
from pygame import gfxdraw, surfarray
from pygame.mixer import Sound
from pygame.time import Clock
from pyrr import Matrix44
from typing import Any, Dict, List, Optional, Sequence, Tuple

import state

from main import exit_game, handle_input
from opengl_renderer import (
    FontBook,
    get_viewport_dimensions,
    InstancedObject,
    ProgramRepository,
    Scene,
    TextRenderer,
    Transform3D
)
from snake import Direction, Game, Node, Snake
from snake_args import add_args
from themes import Theme, themes
from utils import (
    game_over_text,
    get_high_scores,
    parse_key,
    SetInterval,
    update_high_scores
)

if platform.system() != 'Windows':
    import curses

pg.mixer.pre_init()
pg.init()

os.environ['MODERNGL_WINDOW'] = 'pyglet'
os.environ['PYGLET_AUDIO'] = 'openal,pulse,xaudio2,directsound,silent'

PI = 3.141592654

config_defaults = {
    'show_grid': False
}


class HashableRect(pg.Rect):
    def __hash__(self):
        return hash((self.x, self.y, self.width, self.height))


class Renderer(ABC):
    def __init__(self):
        self.fps = 0

    @abstractmethod
    def initialize(self, game: Game, **kw_args: Any) -> None:
        pass

    @abstractmethod
    def render(self, game: Game) -> None:
        pass

    @abstractmethod
    def run(self, game: Game) -> None:
        pass

    @abstractmethod
    def game_over(self, text: str) -> Optional[Sequence[pg.Rect]]:
        pass

    @classmethod
    def default_config(cls, option_name: str) -> Any:
        return config_defaults.get(option_name)


class PGRenderer(Renderer):
    def __init__(self):
        super().__init__()
        self.eating_sound = None
        self.draw_game_over = None
        self.theme = None
        self.fonts = None
        self.font_type = None
        self.clock = None
        self.screen = None
        self.height = None
        self.width = None
        self.scale_factor = None

    def initialize(self, game: Game, **kw_args: Any) -> None:
        self.scale_factor = kw_args['scale_factor']
        if game:
            self.width = game.board_width * self.scale_factor
            self.height = game.board_height * self.scale_factor
        else:
            self.width = self.height = 1

        self.screen = pg.display.set_mode(size=(self.width, self.height), depth=32)

        pg.event.set_allowed([pg.QUIT, pg.KEYDOWN, pg.KEYUP])

        pg.display.set_caption(f'Snake — {state.level_name}')
        self.clock = Clock()

        self.font_type = pg.freetype.get_default_font()
        self.fonts = dict()

        self.theme = themes[kw_args['theme']]

        self.draw_game_over = False

        if kw_args['enable_sound']:
            self.eating_sound = Sound('resources/audio/eating_sound.wav')

    def render(self, game: Game) -> None:
        arr = self.game_to_array(game)

        # Scale the array
        array_img = arr.repeat(self.scale_factor, 1).repeat(self.scale_factor, 0)

        surfarray.blit_array(self.screen, array_img)

        if game.game_over:
            self.game_over(game)
        else:
            self.draw_score(game.score)

        self.draw_fps()

        self.draw_length(len(game.snake))

        self.draw_level_name()

        pg.display.flip()

    def run(self, game: Game) -> None:
        elapsed_time = frames = 0

        state.run = True
        
        while state.run:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    exit_game()
                    break
                if event.type in {pg.KEYDOWN, pg.KEYUP}:
                    handle_input(parse_key(event.key, 'pygame'), event.type == pg.KEYUP)

            self.render(game)

            elapsed_time += self.clock.tick(60)
            frames += 1
            if elapsed_time >= 1000:
                self.fps = frames * 1000 / elapsed_time
                frames = elapsed_time = 0

    def game_to_array(self, game: Game) -> np.ndarray:
        arr = np.tile(self.theme.background, (game.board_width, game.board_height, 1))

        for node in game.level.wall_nodes:
            arr[node.x][node.y] = self.theme.walls

        for node in game.snake:
            arr[node.x][node.y] = self.theme.snake

        arr[game.food.x][game.food.y] = self.theme.food

        return arr

    def get_font(self, font_size: int) -> pg.freetype.Font:
        if font_size not in self.fonts:
            self.fonts[font_size] = pg.freetype.SysFont(self.font_type, font_size)
        return self.fonts[font_size]

    def draw_text(
            self,
            text: str,
            pos: Tuple[int, int],
            which_point: str,
            font_size: int,
            color: Tuple[int, int, int],
            center_lines_vertically: bool=True,
            line_spacing: int=6) -> List[pg.Rect]:
        font = self.get_font(font_size)

        text_lines = text.split('\n')
        n_lines = len(text_lines)
        if center_lines_vertically:
            offsets = [x / 2 + .5 for x in range(-n_lines, n_lines, 2)]
        else:
            offsets = range(n_lines)

        dirty_rects = []

        for text_line, offset in zip(text_lines, offsets):
            text_surface, text_rect = font.render(text_line, color)
            new_y = pos[1] + int(offset * (font_size + line_spacing))
            setattr(text_rect, which_point, (pos[0], new_y))

            self.screen.blit(text_surface, text_rect)

            dirty_rects.append(text_rect)

        return dirty_rects

    def draw_fps(self) -> pg.Rect:
        return self.draw_text(
            f'FPS: {self.fps: 0.1f}', (6, 6), 'topleft', 22, self.theme.text
        )[0]

    def draw_score(self, score: int) -> pg.Rect:
        return self.draw_text(
            f'Score: {score}', (self.width / 2, 6), 'midtop', 22, self.theme.text
        )[0]

    def draw_length(self, length: int) -> pg.Rect:
        return self.draw_text(
            f'Length: {length}', (self.width - 6, 6), 'topright', 22, self.theme.text
        )[0]

    def draw_high_scores(self, top_n=5) -> pg.Rect:
        names, scores, lengths = get_high_scores(top_n=top_n)

        font_size = 36
        line_spacing = 6
        color = self.theme.text

        upper_bound = self.height // 2 - line_spacing
        lower_bound = self.height // 2 + (font_size + line_spacing) * (top_n + 1) + line_spacing
        left_bound = self.width // 2 - font_size * 5
        right_bound = self.width // 2 + font_size * 5
        hline_y = self.height // 2 + font_size + line_spacing
        scores_top = hline_y + line_spacing * 2

        self.draw_text(
            'Score', (self.width // 2 - font_size, self.height // 2), 'topright',
            font_size, color
        )
        
        self.draw_text(
            'Length', (self.width // 2 + font_size, self.height // 2), 'topleft',
            font_size, color
        )

        gfxdraw.vline(self.screen, self.width // 2, upper_bound, lower_bound, color)

        gfxdraw.hline(self.screen, left_bound, right_bound, hline_y, color)

        self.draw_text(
            '\n'.join(scores), (self.width // 2 - font_size * 2, scores_top), 'midtop',
            font_size, color, center_lines_vertically=False, line_spacing=line_spacing
        )

        self.draw_text(
            '\n'.join(lengths), (self.width // 2 + font_size * 2, scores_top), 'midtop',
            font_size, color, center_lines_vertically=False, line_spacing=line_spacing
        )
        
        return pg.Rect(
            left_bound, upper_bound, right_bound - left_bound, lower_bound - upper_bound
        )

    def draw_level_name(self) -> pg.Rect:
        rect = self.draw_text(
            f'Level: {state.level_name}', (self.width / 2, self.height - 2), 'midbottom', 22,
            self.theme.text
        )[0]

        return rect

    def game_over(self, game: Game) -> List[pg.Rect]:
        if self.draw_game_over:
            return [
                *self.draw_text(
                    game_over_text(game, update_high_scores(game)),
                    (self.width // 2, self.height // 5), 'center', 36, self.theme.text
                ),
                self.draw_high_scores()
            ]
        self.draw_game_over = True
        return []


class PGRenderer2(PGRenderer):
    def __init__(self):
        super().__init__()
        self.previous_wall_rects = None
        self.grid_lines = None
        self.drew_level_name = None
        self.food_was_in_wall = None
        self.wall_surface = None
        self.board_surface = None
        self.last_length_rect = None
        self.last_score_rect = None
        self.last_fps_rect = None
        self.past_head_rect = self.past_food_rect = pg.Rect(0, 0, 0, 0)
        self.past_snake_rects = set()
        self.first_render = True
        self.drew_game_over = False

    def initialize(self, game: Game, **kw_args: Any) -> None:
        super(PGRenderer2, self).initialize(game, **kw_args)
        self.last_fps_rect = pg.Rect(6, 6, 0, 0)
        self.last_score_rect = pg.Rect(self.width, 6, 0, 0)
        self.last_length_rect = pg.Rect(self.width / 2, 6, 0, 0)
        self.board_surface = pg.Surface((self.width, self.height))
        self.wall_surface = pg.Surface((self.width, self.height))
        self.wall_surface.set_colorkey((0, 0, 0))
        self.food_was_in_wall = False
        self.drew_level_name = False

        self.grid_lines = self.generate_gridlines() if kw_args['show_grid'] else None

        self.previous_wall_rects = set()

    def render(self, game: Game) -> None:
        dirty_rects = []

        # Get all snake piece rectangles
        snake_rects = set(
            self.get_node_rect(node) for node in game.snake
        )

        # Recompute the snake head. Unless I want to install or create an IndexedSet class,
        # recomputing the head rect is probably the best option
        snake_head_rect = self.get_node_rect(game.snake.head)

        # Add snake rectangles to known rectangles
        self.past_snake_rects.update(snake_rects)

        dirty_rects.extend(self.draw_walls(game))

        # If it's the first time render() is called
        first_render = self.first_render
        if self.first_render:
            self.draw_board()
            dirty_rects.append(
                self.fill_background(pg.Rect(0, 0, self.width, self.height))
            )

            # Draw the snake
            for rect in snake_rects:
                if rect == snake_head_rect:
                    self.draw_head(rect, game.snake)
                else:
                    pg.draw.rect(self.screen, self.theme.snake, rect)

            self.first_render = False

        # Find squares where the snake used to be and clear them
        to_remove = {rect for rect in self.past_snake_rects if rect not in snake_rects}
        for rect in to_remove:
            dirty_rects.append(self.fill_background(rect))
        self.past_snake_rects = self.past_snake_rects - to_remove
        
        food_rect = self.draw_food(game)
        if food_rect:
            dirty_rects.append(food_rect)
            if not first_render and self.eating_sound:
                self.eating_sound.play()

        # If the head has moved since the last render call, draw the new head
        if self.past_head_rect != snake_head_rect:
            # Cover previous head location
            if len(game.snake) > 1:
                pg.draw.rect(self.screen, self.theme.snake, self.past_head_rect)
            dirty_rects.append(self.past_head_rect)

            # If the starting length was 1 and we just ate food, we have to manually
            # add the previous head to the past rects or it will leave a square behind.
            if food_rect and len(game.snake) == 2:
                self.past_snake_rects.add(self.past_head_rect)

            dirty_rects.append(self.draw_head(snake_head_rect, game.snake))

        dirty_rects.extend(self.draw_all_text(game))

        # Update the dirty rectangles
        pg.display.update(dirty_rects)

    @classmethod
    def default_config(cls, option_name: str) -> Any:
        if option_name == 'show_grid':
            return True
        return super(PGRenderer2, cls).default_config(option_name)

    def draw_board(self) -> None:
        # Draw the background
        screen_rect = pg.Rect(0, 0, self.width, self.height)
        pg.draw.rect(self.board_surface, self.theme.background, screen_rect)

        # Draw grid lines if present
        if self.grid_lines:
            for start_pt, end_pt in self.grid_lines:
                pg.draw.line(self.board_surface, self.theme.grid, start_pt, end_pt)

    def draw_walls(self, game: Game) -> List[pg.Rect]:
        wall_rects = {self.get_node_rect(node) for node in game.level.wall_nodes}
        new_wall_rects = wall_rects - self.previous_wall_rects
        wall_rects_to_remove = self.previous_wall_rects - wall_rects
        self.previous_wall_rects = wall_rects

        # Draw new wall positions
        for rect in new_wall_rects:
            pg.draw.rect(self.wall_surface, self.theme.walls, rect)
            self.fill_background(rect)

        # Clear old wall positions
        for rect in wall_rects_to_remove:
            pg.draw.rect(self.wall_surface, (0, 0, 0), rect)
            self.fill_background(rect)

        return [*new_wall_rects, *wall_rects_to_remove]

    def fill_background(self, fill_area: pg.Rect) -> pg.Rect:
        self.screen.blit(self.board_surface, fill_area, area=fill_area)
        self.screen.blit(self.wall_surface, fill_area, area=fill_area)
        return fill_area

    def generate_gridlines(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        horz_lns = [
            (
                (0, offset), (self.width, offset)
            ) for offset in range(self.scale_factor, self.height, self.scale_factor)
        ]
        vert_lns = [
            (
                (offset, 0), (offset, self.height)
            ) for offset in range(self.scale_factor, self.width, self.scale_factor)
        ]

        return horz_lns + vert_lns

    def draw_head(self, head_rect: pg.Rect, snake: Snake) -> pg.Rect:
        direction = snake.direction

        # Draw the head with curved ends
        head_params = self.get_head_params(direction)
        pg.draw.rect(self.screen, self.theme.snake, head_rect, **head_params)

        # Calculate eye positions
        side_offset = self.scale_factor / 5
        front_offset = self.scale_factor / 2
        eye_radius = max(self.scale_factor // 12, 1)
        if direction.name == 'up':
            x1 = head_rect.x + side_offset
            y1 = head_rect.y + front_offset
            x2 = head_rect.x + head_rect.width - side_offset
            y2 = head_rect.y + front_offset
        elif direction.name == 'down':
            x1 = head_rect.x + side_offset
            y1 = head_rect.y + head_rect.height - front_offset
            x2 = head_rect.x + head_rect.width - side_offset
            y2 = head_rect.y + head_rect.height - front_offset
        elif direction.name == 'right':
            x1 = head_rect.x + head_rect.width - front_offset
            y1 = head_rect.y + side_offset
            x2 = head_rect.x + head_rect.width - front_offset
            y2 = head_rect.y + head_rect.height - side_offset
        else:
            x1 = head_rect.x + front_offset
            y1 = head_rect.y + side_offset
            x2 = head_rect.x + front_offset
            y2 = head_rect.y + head_rect.height - side_offset

        # Draw eyes
        eye_color = self.theme.eyes
        gfxdraw.aacircle(self.screen, int(x1), int(y1), eye_radius, eye_color)
        gfxdraw.filled_circle(self.screen, int(x1), int(y1), eye_radius, eye_color)
        gfxdraw.aacircle(self.screen, int(x2), int(y2), eye_radius, eye_color)
        gfxdraw.filled_circle(self.screen, int(x2), int(y2), eye_radius, eye_color)

        self.past_head_rect = head_rect

        return head_rect

    def draw_food(self, game: Game) -> Optional[pg.Rect]:
        food = game.food
        food_in_wall = game.level.in_wall(game.food)
        force_redraw = food_in_wall | self.food_was_in_wall
        self.food_was_in_wall = food_in_wall

        # Get the food rectangle
        food_rect = self.get_node_rect(food)

        # If the food was just eaten, draw the new food
        if self.past_food_rect != food_rect or force_redraw:
            r = food_rect.width / 2 - 2
            pg.draw.circle(self.screen, self.theme.food, food_rect.center, r)
            self.past_food_rect = food_rect

            return food_rect

    def draw_fps(self) -> pg.Rect:
        last_fps_rect = self.last_fps_rect

        # Clear previous text
        self.fill_background(last_fps_rect)

        text_rect = super(PGRenderer2, self).draw_fps()
        self.last_fps_rect = text_rect

        # Return dirty rect
        return last_fps_rect.union(text_rect)

    def draw_level_name(self) -> Optional[pg.Rect]:
        if not self.drew_level_name:
            self.drew_level_name = True
            return super(PGRenderer2, self).draw_level_name()

    def draw_all_text(self, game: Game) -> List[pg.Rect]:        
        # Either draw the score or the game over screen
        if game.game_over and not self.drew_game_over:
            # We want to skip one rendering cycle before drawing game over and the high
            # scores because saving/reading the high scores takes a bit
            if self.draw_game_over:
                dirty_rects = self.game_over(game)
                # Clear score text if game is over
                self.fill_background(self.last_score_rect)
                dirty_rects.append(self.last_score_rect)
                self.drew_game_over = True
            else:
                self.draw_game_over = True
                dirty_rects = []
        elif not self.drew_game_over:
            dirty_rects = [self.draw_score(game.score)]
        else:
            dirty_rects = []
            
        # Draw the frame rate
        dirty_rects.append(self.draw_fps())

        # Draw the snake length
        dirty_rects.append(self.draw_length(len(game.snake)))

        # Draw level name
        level_name_rect = self.draw_level_name()
        if level_name_rect:
            dirty_rects.append(level_name_rect)

        return dirty_rects

    def draw_score(self, score: int) -> pg.Rect:
        last_score_rect = self.last_score_rect

        # Clear previous text
        self.fill_background(last_score_rect)

        text_rect = super(PGRenderer2, self).draw_score(score)
        self.last_score_rect = text_rect

        # Return dirty rect
        return last_score_rect.union(text_rect)

    def draw_length(self, length: int) -> pg.Rect:
        last_length_rect = self.last_length_rect

        # Clear previous text
        self.fill_background(last_length_rect)

        text_rect = super(PGRenderer2, self).draw_length(length)
        self.last_length_rect = text_rect

        # Return dirty rect
        return last_length_rect.union(text_rect)

    def get_node_rect(self, node: Node, buffer: int=0) -> HashableRect:
        sf = self.scale_factor
        wh = sf + 2 * buffer
        return HashableRect(
            sf * node.x - buffer,
            sf * node.y - buffer,
            wh,
            wh
        )

    def get_head_params(self, direction: Direction) -> Dict:
        rounding_factor = 3
        if direction == Direction.up():
            params = {
                'border_top_left_radius': self.scale_factor // rounding_factor,
                'border_top_right_radius': self.scale_factor // rounding_factor
            }
        elif direction == Direction.down():
            params = {
                'border_bottom_left_radius': self.scale_factor // rounding_factor,
                'border_bottom_right_radius': self.scale_factor // rounding_factor
            }
        elif direction == Direction.left():
            params = {
                'border_top_left_radius': self.scale_factor // rounding_factor,
                'border_bottom_left_radius': self.scale_factor // rounding_factor
            }
        elif direction == Direction.right():
            params = {
                'border_top_right_radius': self.scale_factor // rounding_factor,
                'border_bottom_right_radius': self.scale_factor // rounding_factor
            }
        else:
            print('Invalid direction!')
            params = {}

        return params


class RelativeDirection(Enum):
    ABOVE = auto()
    BELOW = auto()
    LEFT_OF = auto()
    RIGHT_OF = auto()


connecting_polygon_params = {
    (RelativeDirection.LEFT_OF, RelativeDirection.ABOVE): {
        'start_angle': -PI / 2,
        'stop_angle': 0,
        'index1': 1,
        'index2': 0,
        'index3': 3,
        'inter_along_x_axis': False,
        'add_current_first': False,
        'swap_ss_indices': False,
        'swap_es_indices': True
    },
    (RelativeDirection.LEFT_OF, RelativeDirection.BELOW): {
        'start_angle': PI / 2,
        'stop_angle': 0,
        'index1': 0,
        'index2': 1,
        'index3': 2,
        'inter_along_x_axis': False,
        'add_current_first': False,
        'swap_ss_indices': True,
        'swap_es_indices': True
    },
    (RelativeDirection.ABOVE, RelativeDirection.RIGHT_OF): {
        'start_angle': 0,
        'stop_angle': PI / 2,
        'index1': 2,
        'index2': 1,
        'index3': 0,
        'inter_along_x_axis': True,
        'add_current_first': True,
        'swap_ss_indices': True,
        'swap_es_indices': True
    },
    (RelativeDirection.ABOVE, RelativeDirection.LEFT_OF): {
        'start_angle': PI,
        'stop_angle': PI / 2,
        'index1': 1,
        'index2': 2,
        'index3': 3,
        'inter_along_x_axis': True,
        'add_current_first': False,
        'swap_ss_indices': False,
        'swap_es_indices': True
    },
    (RelativeDirection.BELOW, RelativeDirection.LEFT_OF): {
        'start_angle': PI,
        'stop_angle': 3 * PI / 2,
        'index1': 0,
        'index2': 3,
        'index3': 2,
        'inter_along_x_axis': True,
        'add_current_first': False,
        'swap_ss_indices': False,
        'swap_es_indices': False
    },
    (RelativeDirection.BELOW, RelativeDirection.RIGHT_OF): {
        'start_angle': 0,
        'stop_angle': -PI / 2,
        'index1': 3,
        'index2': 0,
        'index3': 1,
        'inter_along_x_axis': True,
        'add_current_first': True,
        'swap_ss_indices': True,
        'swap_es_indices': False
    },
    (RelativeDirection.RIGHT_OF, RelativeDirection.ABOVE): {
        'start_angle': -PI / 2,
        'stop_angle': -PI,
        'index1': 2,
        'index2': 3,
        'index3': 0,
        'inter_along_x_axis': False,
        'add_current_first': True,
        'swap_ss_indices': False,
        'swap_es_indices': False
    },
    (RelativeDirection.RIGHT_OF, RelativeDirection.BELOW): {
        'start_angle': PI / 2,
        'stop_angle': PI,
        'index1': 3,
        'index2': 2,
        'index3': 1,
        'inter_along_x_axis': False,
        'add_current_first': True,
        'swap_ss_indices': True,
        'swap_es_indices': False
    }
}


class PGRenderer3(PGRenderer2):
    def render(self, game: Game) -> None:
        dirty_rects = []

        dirty_rects.extend(self.draw_walls(game))

        # If it's the first time render() is called
        first_render = self.first_render
        if self.first_render:
            self.draw_board()
            dirty_rects.append(
                self.fill_background(pg.Rect(0, 0, self.width, self.height))
            )

            self.first_render = False

        dirty_rects.extend(self.draw_snake(game.snake))

        food_rect = self.draw_food(game)
        if food_rect:
            dirty_rects.append(food_rect)
            if not first_render and self.eating_sound:
                self.eating_sound.play()

        dirty_rects.extend(self.draw_all_text(game))

        # Update the dirty rectangles
        pg.display.update(dirty_rects)

    def get_relative_direction(self, this_node: Node, relative_to: Node) -> RelativeDirection:
        if this_node.y < relative_to.y:
            return RelativeDirection.ABOVE
        if this_node.y > relative_to.y:
            return RelativeDirection.BELOW
        if this_node.x < relative_to.x:
            return RelativeDirection.LEFT_OF
        return RelativeDirection.RIGHT_OF

    def get_bend_poly(
            self,
            start_angle: float,
            end_angle: float,
            start_scale: int,
            end_scale: int,
            center_point: Tuple[int, int],
            connecting_point: Tuple[int, int],
            points: int=20) -> List[Tuple[int, int]]:
        r = np.linspace(start_scale, end_scale, points)
        t = np.linspace(start_angle, end_angle, points)
        x = r * np.cos(t) + center_point[0]
        y = r * np.sin(t) + center_point[1]

        return [*zip(x.tolist(), y.tolist()), connecting_point, center_point]

    def get_interp_poly(
            self,
            start_pt: Tuple[int, int],
            inter_pt: Tuple[int, int],
            end_pt: Tuple[int, int],
            center_point: Tuple[int, int],
            connecting_point: Tuple[int, int],
            points=20) -> List[Tuple[int, int]]:
        x, y = zip(start_pt, inter_pt, end_pt)
        p = np.polynomial.Polynomial.fit(x, y, deg=2)

        return [*zip(*p.linspace(points)), center_point, connecting_point]

    def fix_disjoint_polygons(
            self,
            current_polygon: List[Tuple[int, int]],
            previous_polygon: List[Tuple[int, int]],
            current_relative_direction: RelativeDirection,
            previous_relative_direction: RelativeDirection) -> List[Tuple[int, int]]:
        relative_directions = current_relative_direction, previous_relative_direction
        if relative_directions in connecting_polygon_params:
            # add parameters as function local variables
            d = connecting_polygon_params[relative_directions]
            start_angle = d['start_angle']
            stop_angle = d['stop_angle']
            index1 = d['index1']
            index2 = d['index2']
            index3 = d['index3']
            inter_along_x_axis = d['inter_along_x_axis']
            add_current_first = int(d['add_current_first'])
            swap_ss_indices = d['swap_ss_indices']
            swap_es_indices = d['swap_es_indices']

            shift_axis = int(inter_along_x_axis)
            other_axis = 1 - shift_axis

            # Shift necessary previous_polygon points
            shift = previous_polygon[index3][shift_axis] - current_polygon[index1][shift_axis]
            new_point = [current_polygon[index1][other_axis]]
            new_point.insert(shift_axis, previous_polygon[index2][shift_axis] - shift)
            previous_polygon[index2] = tuple(new_point)
            new_point[shift_axis] = previous_polygon[index3][shift_axis] - shift
            previous_polygon[index3] = tuple(new_point)

            # Compute start_scale
            indices = index1, index2
            swap_idx = int(swap_ss_indices)
            start_scale = (
                current_polygon[indices[swap_idx]][other_axis] -
                current_polygon[indices[1 - swap_idx]][other_axis]
            )

            # Compute end_scale
            indices = index3, index1
            swap_idx = int(swap_es_indices)
            end_scale = (
                previous_polygon[indices[swap_idx]][shift_axis] -
                previous_polygon[indices[1 - swap_idx]][shift_axis]
            )

            # Generate the connecting polygon
            if min(start_scale, end_scale) / max(start_scale, end_scale) > .6:
                connecting_polygon = self.get_bend_poly(
                    start_angle, stop_angle, start_scale, end_scale, current_polygon[index1],
                    previous_polygon[index1]
                )
            else:
                start_pt = current_polygon[index2]
                end_pt = previous_polygon[index1]

                inter_pt = (
                    previous_polygon[index2][shift_axis],
                    start_pt[other_axis] + (previous_polygon[index2][other_axis] - start_pt[other_axis]) / 2
                )
                inter_pt = inter_pt[shift_axis], inter_pt[other_axis]

                connecting_pts = current_polygon[index1], previous_polygon[index1]
                connecting_pts = connecting_pts[1 - add_current_first], connecting_pts[add_current_first]
                connecting_polygon = self.get_interp_poly(start_pt, inter_pt, end_pt, *connecting_pts)
        else:
            connecting_polygon = None

        return connecting_polygon

    def get_polygons(self, snake: Snake) -> List[List[Tuple[int, int]]]:
        decrease_factor = self.scale_factor / 2 / len(snake.nodes)

        get_unchanged = lambda c: c * self.scale_factor
        get_decreased = lambda c, d, s: c * self.scale_factor + s * decrease_factor * d

        polygons = []
        joining_polygons = []
        for i in range(len(snake.nodes) - 1):
            node = snake.nodes[i + 1]
            relative_direction = self.get_relative_direction(node, snake.nodes[i])

            if relative_direction == RelativeDirection.ABOVE:
                polygon = [
                    (get_decreased(node.x + 1, i + 1, -1), get_unchanged(node.y)),  # top right corner
                    (get_decreased(node.x + 1, i, -1), get_unchanged(node.y + 1)),  # bottom right corner
                    (get_decreased(node.x, i, 1), get_unchanged(node.y + 1)),       # bottom left corner
                    (get_decreased(node.x, i + 1, 1), get_unchanged(node.y))        # top left corner
                ]
            elif relative_direction == RelativeDirection.BELOW:
                polygon = [
                    (get_decreased(node.x + 1, i, -1), get_unchanged(node.y)),          # top right corner
                    (get_decreased(node.x + 1, i + 1, -1), get_unchanged(node.y + 1)),  # bottom right corner
                    (get_decreased(node.x, i + 1, 1), get_unchanged(node.y + 1)),       # bottom left corner
                    (get_decreased(node.x, i, 1), get_unchanged(node.y))                # top left corner
                ]
            elif relative_direction == RelativeDirection.LEFT_OF:
                polygon = [
                    (get_unchanged(node.x + 1), get_decreased(node.y, i, 1)),       # top right corner
                    (get_unchanged(node.x + 1), get_decreased(node.y + 1, i, -1)),  # bottom right corner
                    (get_unchanged(node.x), get_decreased(node.y + 1, i + 1, -1)),  # bottom left corner
                    (get_unchanged(node.x), get_decreased(node.y, i + 1, 1))        # top left corner
                ]
            else:
                polygon = [
                    (get_unchanged(node.x + 1), get_decreased(node.y, i + 1, 1)),       # top right corner
                    (get_unchanged(node.x + 1), get_decreased(node.y + 1, i + 1, -1)),  # bottom right corner
                    (get_unchanged(node.x), get_decreased(node.y + 1, i, -1)),          # bottom left corner
                    (get_unchanged(node.x), get_decreased(node.y, i, 1))                # top left corner
                ]

            if i >= 1:
                previous_relative_direction = self.get_relative_direction(
                    snake.nodes[i], snake.nodes[i - 1]
                )
                joining_polygon = self.fix_disjoint_polygons(
                    polygon, polygons[-1], relative_direction, previous_relative_direction
                )
                if joining_polygon:
                    joining_polygons.append(joining_polygon)

            polygons.append(polygon)

        # Add in joining polygons
        polygons.extend(joining_polygons)

        return polygons

    def draw_snake(self, snake: Snake) -> List[pg.Rect]:
        dirty_rects = []

        # Get all snake piece rectangles
        snake_rects = [self.get_node_rect(node, buffer=2) for node in snake]
        snake_rects_set = set(snake_rects)
        snake_head_rect = snake_rects[0]

        # Add snake rectangles to known rectangles
        self.past_snake_rects.update(snake_rects_set)

        # Find squares where the snake used to be and clear them
        to_remove = {rect for rect in self.past_snake_rects if rect not in snake_rects_set}
        dirty_rects.extend(self.fill_background(rect) for rect in to_remove)
        self.past_snake_rects -= to_remove

        # If there has been any changes since the last render:
        if len(to_remove) > 0 or self.past_head_rect != snake_head_rect or self.first_render:
            # Clear previous snake squares
            for rect in snake_rects[1:]:
                self.fill_background(rect)

            # Draw snake
            snake_color = self.theme.snake
            for polygon in self.get_polygons(snake):
                gfxdraw.aapolygon(self.screen, polygon, snake_color)
                gfxdraw.filled_polygon(self.screen, polygon, snake_color)
            self.draw_head(snake_head_rect, snake)

            dirty_rects.extend(snake_rects)

        return dirty_rects


class CLRenderer(Renderer):
    def __init__(self, *vargs):
        super().__init__()
        self.loop = None

    @staticmethod
    def clear_terminal():
        if platform.system() == 'Windows':
            subprocess.Popen('cls', shell=True).communicate()
        else:  # Linux and Mac
            print('\033c', end='')

    def handle_exit(self, *args: Any) -> None:
        if not state.run:
            self.loop.cancel()

    def initialize(self, game: Game, **kw_args: Any) -> None:
        state.run = True
        self.loop = SetInterval(kw_args['tick_time'], 0, self.render, self.handle_exit, game)

    def render(self, game: Game) -> None:
        self.clear_terminal()
        print(game)

        if game.game_over:
            self.game_over(game)

    def run(self, game: Game) -> None:
        self.render(game)

    def game_over(self, game: Game) -> Optional[Sequence[pg.Rect]]:
        print(game_over_text(game, update_high_scores(game)))
        self.loop.cancel()


if platform.system() != 'Windows':
    class CursesRenderer(Renderer):
        def __init__(self):
            super().__init__()
            self.color_pairs = None
            self.eye_color = None
            self.text_color = None
            self.food_color = None
            self.wall_color = None
            self.snake_color = None
            self.background_color = None
            self.debug_text = []
            self.height = None
            self.width = None
            self.draw_game_over = False
            self.drew_game_over = False
            self.previous_food = None
            self.previous_wall_nodes = set()
            self.previous_snake_nodes = set()
            self.elapsed_time = None
            self.frames = None
            self.prev_time = None
            self.previous_calc_time = None
            self.use_colors = None
            self.window = None
            self.stdscr = None

        def initialize(self, game: Game, **kw_args) -> None:
            self.stdscr = curses.initscr()
            self.window = curses.newwin(game.board_height, game.board_width, 0, 0)
            curses.start_color()
            curses.noecho()
            curses.cbreak()
            self.window.keypad(True)
            self.window.nodelay(True)
            curses.curs_set(False)

            self.use_colors = self.init_theme(themes[kw_args['theme']])
            if self.use_colors:
                self.window.bkgd(' ', curses.color_pair(4))

            self.previous_calc_time = self.prev_time = time.perf_counter()
            self.elapsed_time = self.frames = 0
            self.previous_snake_nodes = set()
            self.previous_wall_nodes = set()
            self.previous_food = None
            self.drew_game_over = False
            self.draw_game_over = False
            self.height = game.board_height
            self.width = game.board_width
            self.debug_text = []

        def render(self, game: Game) -> None:  # TODO make this work when colors don't work
            width = self.width

            wall_nodes = set(game.level.wall_nodes)
            new_nodes = wall_nodes - self.previous_wall_nodes
            nodes_to_remove = self.previous_wall_nodes - wall_nodes
            self.previous_wall_nodes = {node.copy() for node in wall_nodes}
            new_nodes = sorted(new_nodes, key=lambda node: (node.y, node.x))
            for node in itertools.chain(new_nodes, nodes_to_remove):
                self.draw_str(' ', node.x, node.y, self.text_color, game)

            snake_nodes = set(game.snake.nodes)
            new_nodes = snake_nodes - self.previous_snake_nodes
            nodes_to_remove = self.previous_snake_nodes - snake_nodes
            self.previous_snake_nodes = {node.copy() for node in snake_nodes}

            for node in nodes_to_remove:
                if node.marker == game.snake.head.marker and node in game.snake:
                    color = self.eye_color
                else:
                    color = self.text_color
                self.draw_str(' ', node.x, node.y, color, game)
            for node in new_nodes:
                s = ' '
                if node is game.snake.head:
                    s = ':' if game.snake.direction.offset_y == 0 else '܅'
                self.draw_str(s, node.x, node.y, self.eye_color, game, options=curses.A_BOLD)
     
            if game.food != self.previous_food:
                self.draw_str(game.food.marker, game.food.x, game.food.y, self.food_color, game)
                self.previous_food = game.food

            n_top_strings = 0
            if width > 12:
                length_str = f'Length: {len(game.snake)}'
                n_top_strings += 1
            if width > 20:
                score_str = f'Score: {game.score}'
                n_top_strings += 1
            if width > 31:
                fps_str = f'FPS: {self.fps:.0f}'
                n_top_strings += 1

            # TODO: draw the top wall only for the first render and then fill any missing
            # characters when drawing the top strings
            self.draw_str(' ' * width, 0, 0, self.text_color, game)

            if n_top_strings == 3:
                self.draw_str(fps_str, 1, 0, self.text_color, game)
                if not game.game_over:
                    start_x = (width - len(score_str)) // 2
                    self.draw_str(score_str, start_x, 0, self.text_color, game)
                start_x = width - len(length_str) - 1
                self.draw_str(length_str, start_x, 0, self.text_color, game)
            elif n_top_strings == 2:
                if not game.game_over:
                    self.draw_str(score_str, 1, 0, self.text_color, game)
                start_x = width - len(length_str) - 1
                self.draw_str(length_str, start_x, 0, self.text_color, game)
            elif n_top_strings == 1:
                start_x = width // 2 - len(length_str)
                self.draw_str(length_str, start_x, 0, self.text_color, game)

            if game.game_over and not self.drew_game_over:
                if self.draw_game_over:
                    self.game_over(game)
                    self.drew_game_over = True
                else:
                    self.draw_game_over = True

            self.window.refresh()

        def run(self, game: Game) -> None:
            state.run = True
            while state.run:
                self.process_input()
                self.render(game)
                elapsed_time = self.track_fps()

                # Note: it seems like this doesn't always sleep long enough (at least on the
                # Mac computer I'm testing on, python 3.6). I think it's due to the timer not
                # precisely measuring how much time has elapsed (overestimating). I get the
                # proper FPS if I require it to sleep for 16ms, but I don't want to
                # artificially lower FPS on slower computers. So, my options seem to be requiring
                # python 3.7 and use nanosecond precision or accept that the update rate may be
                # higher than 60fps. For now, I'll do the latter.
                curses.napms(max(0, 17 - int(elapsed_time * 1000)))

            self.handle_exit()

        def game_over(self, game: Game) -> Optional[Sequence[pg.Rect]]:
            game_over_str = game_over_text(game, update_high_scores(game))
            lines = game_over_str.split('\n')
            start_y = self.height // 5
            for line, y in zip(lines, range(start_y, start_y + len(lines))):
                start_x = (self.width + 2 - len(line)) // 2
                self.draw_str(line, start_x, y, self.text_color, game)

            self.draw_high_scores(game)

        def draw_high_scores(self, game: Game, top_n: int=5) -> None:
            names, scores, lengths = get_high_scores(top_n=top_n)

            start_y = self.height // 2

            self.draw_str(
                ' Score | Length ', self.width // 2 - 7, start_y, self.text_color, game,
                options=curses.A_UNDERLINE
            )
            for score, length, i in zip(scores, lengths, range(1, len(scores) + 1)):
                x = self.width // 2 - len(score) - 1
                self.draw_str(f'{score} | {length}', x, start_y + i, self.text_color, game)

        def draw_str(
                self, s: str,
                start_x: int,
                y: int,
                fg_color: int,
                game: Game,
                options: int=0) -> None:
            snake_points = {(node.x, node.y) for node in game.snake}
            wall_points = {(node.x, node.y) for node in game.level.wall_nodes}

            str_len = len(s)
            for x, i in zip(range(start_x, start_x + str_len), range(str_len)):
                if self.use_colors:
                    if (x, y) in snake_points:
                        bg_color = self.snake_color
                    elif (x, y) in wall_points:
                        bg_color = self.wall_color
                    else:
                        bg_color = self.background_color

                    color_pair = self.color_pairs[fg_color, bg_color]
                else:
                    color_pair = 0

                # Note: since I'm drawing individual wall nodes, I'm just going to assume the last cell
                # of the screen is the same character as the one to it's left
                if x == self.width - 1 and y == self.height - 1:
                    self.window.insch(y, x - 1, s[i], curses.color_pair(color_pair) | options)
                elif x == self.width - 2 and y == self.height - 1:
                    self.window.addstr(y, x, s[i], curses.color_pair(color_pair) | options)
                else:
                    self.window.addstr(y, x, s[i], curses.color_pair(color_pair) | options)

        def init_theme(self, theme: Theme) -> bool:
            to_curses_color = lambda r, g, b: (min(999, r * 4), min(999, g * 4), min(999, b * 4))
            use_colors = True
            if curses.can_change_color():
                curses.init_color(0, 0, 0, 0)
                curses.init_color(1, *to_curses_color(*theme.background))
                curses.init_color(2, *to_curses_color(*theme.snake))
                curses.init_color(3, *to_curses_color(*theme.walls))
                curses.init_color(4, *to_curses_color(*theme.food))
                curses.init_color(5, *to_curses_color(*theme.text))
                curses.init_color(6, *to_curses_color(*theme.eyes))

                self.background_color = 1
                self.snake_color = 2
                self.wall_color = 3
                self.food_color = 4
                self.text_color = 5
                self.eye_color = 6
            elif curses.has_colors():
                self.background_color = curses.COLOR_BLACK
                self.snake_color = curses.COLOR_GREEN
                self.wall_color = curses.COLOR_BLUE
                self.food_color = curses.COLOR_CYAN
                self.text_color = curses.COLOR_RED
                self.eye_color = curses.COLOR_YELLOW
            else:
                use_colors = False

            if use_colors:
                curses.init_pair(1, self.text_color, self.background_color)
                curses.init_pair(2, self.eye_color, self.snake_color)
                curses.init_pair(3, self.text_color, self.wall_color)
                curses.init_pair(4, self.food_color, self.background_color)
                curses.init_pair(5, self.text_color, self.background_color)
                curses.init_pair(6, self.text_color, self.snake_color)

                self.color_pairs = {
                    (self.text_color, self.background_color): 1,
                    (self.eye_color, self.snake_color): 2,
                    (self.text_color, self.wall_color): 3,
                    (self.food_color, self.background_color): 4,
                    (self.text_color, self.background_color): 5,
                    (self.text_color, self.snake_color): 6
                }

            return use_colors

        def process_input(self):
            key = self.window.getch()
            while key != -1:
                handle_input(parse_key(key, 'curses'), False, ignore_released=True)
                key = self.window.getch()

        def handle_exit(self, *args: Any) -> None:
            curses.nocbreak()
            self.stdscr.keypad(False)
            curses.echo()
            curses.endwin()

        def track_fps(self) -> float:
            curr_time = time.perf_counter()
            elapsed_time = curr_time - self.prev_time
            self.prev_time = curr_time
            self.elapsed_time += elapsed_time
            self.frames += 1
            if self.elapsed_time >= 1:
                self.fps = self.frames / self.elapsed_time
                self.elapsed_time = 0
                self.frames = 0

            return elapsed_time

        def debug(self, s: str) -> None:
            self.debug_text.append(s)
            x = self.width + 2
            y = len(self.debug_text) % self.height
            self.stdscr.addstr(y, x, s, curses.color_pair(3))
            self.stdscr.refresh()


class OpenGLRenderer(mglw.WindowConfig, Renderer):
    title = 'Snake'
    gl_version = (3, 3)
    window_size = (1920, 1080)
    aspect_ratio = None
    resizable = True
    vsync = True
    samples = 0

    resource_dir = os.path.normpath(os.path.join(__file__, '../resources'))

    def __init__(self, **kwargs):
        display = pyglet.canvas.Display()
        screen = display.get_default_screen()
        self.fullscreen_width = screen.width
        self.fullscreen_height = screen.height

        self.create_window()
        self.timer = Timer()
        super().__init__(ctx=self.wnd.ctx, wnd=self.wnd, timer=self.timer, **kwargs)
        Renderer.__init__(self)
        self.wnd.config = self
        self.wnd.fullscreen_key = self.wnd.keys.F
        self.wnd.swap_buffers()
        self.wnd.set_default_viewport()

        self.high_score_vert_line = None
        self.high_score_horiz_line = None
        self.font = None
        self.food_pos = None
        self.orthogonal_proj = None
        self.viewport_height = None
        self.viewport_width = None
        self.font_book = None
        self.scene = None
        self.background_color = None
        self.theme = None
        self.game = None
        self.frames = None
        self.score_renderer = None
        self.level_renderer = None
        self.fps_renderer = None
        self.length_renderer = None
        self.game_over_renderer = None
        self.high_scores_renderer = None
        self.elapsed_time = self.frames = 0
        self.draw_game_over = False
        self.eating_sound = None

        self.recreate_text_renderer = defaultdict(lambda: True)

        # Note: after we finish rendering the first frame, set state.run to true. We do it in the
        # renderer instead of in main because startup can take a while.
        self.should_set_state_run = True

    # We want to add the snake arguments to this class so it doesn't fail when parsing
    @classmethod
    def add_arguments(cls, parser):
        add_args(parser)

    def create_window(self):
        parser = mglw.create_parser()
        self.add_arguments(parser)
        values = mglw.parse_args(parser=parser)
        self.argv = values
        window_cls = mglw.get_local_window_cls(values.window)

        size = self.window_size
        size = int(size[0] * values.size_mult), int(size[1] * values.size_mult)

        # Resolve cursor
        show_cursor = values.cursor
        if show_cursor is None:
            show_cursor = self.cursor

        self.wnd = window_cls(
            title=self.title,
            size=size,
            fullscreen=False,
            resizable=values.resizable
            if values.resizable is not None
            else self.resizable,
            gl_version=self.gl_version,
            aspect_ratio=self.aspect_ratio,
            vsync=values.vsync if values.vsync is not None else self.vsync,
            samples=values.samples if values.samples is not None else self.samples,
            cursor=show_cursor if show_cursor is not None else True,
        )
        self.wnd.print_context_info()

        mglw.activate_context(window=self.wnd)

    def initialize(self, game: Game, **kwargs) -> None:
        kwargs.update(vars(self.argv))
        state.shader_program_repo = ProgramRepository()
        self.game = game
        self.food_pos = game.food.pos
        self.theme = themes[kwargs['theme']]
        self.background_color = tuple(x / 255 for x in self.theme.background)

        if kwargs['enable_sound']:
            self.eating_sound = pyglet.media.load(
                'resources/audio/eating_sound.wav', streaming=False
            )

        if self.fullscreen:
            aspect_ratio = self.fullscreen_width / self.fullscreen_height
            self.wnd.fullscreen = True
        else:
            aspect_ratio = self.window_size[0] / self.window_size[1]

        self.update_viewport()

        self.scene = Scene(
            self.game, self.theme, kwargs['taper_opengl'], aspect_ratio=aspect_ratio
        )
        self.font_book = FontBook()
        self.font = self.font_book['SFNSMono', 64]

    def key_event(self, key, action, modifiers):
        if action in {self.wnd.keys.ACTION_PRESS, self.wnd.keys.ACTION_RELEASE}:
            handle_input(parse_key(key, 'pyglet'), action == self.wnd.keys.ACTION_RELEASE)

        if action == self.wnd.keys.ACTION_RELEASE and key == self.wnd.keys.F:
            self.toggle_fullscreen()

    def update_viewport(self):
        self.viewport_width, self.viewport_height = get_viewport_dimensions()

        self.orthogonal_proj = Matrix44.orthogonal_projection(
            0,  # left
            self.viewport_width,  # right
            0,  # bottom
            self.viewport_height,  # top
            1.0,  # near
            -1.0,  # far
            dtype='f4'
        )

        self.recreate_text_renderer.clear()

    def toggle_fullscreen(self) -> None:
        if self.wnd.fullscreen:
            width, height = self.fullscreen_width, self.fullscreen_height
        else:
            width, height = self.window_size

        self.update_viewport()
        self.scene.resize(width / height)

    def render(self, run_time: float, frame_time: float):
        self.elapsed_time += frame_time
        self.frames += 1
        if self.elapsed_time >= 1:
            self.fps = self.frames / self.elapsed_time
            self.frames = self.elapsed_time = 0

        food_pos = self.game.food.pos
        if self.food_pos != food_pos:
            self.food_pos = food_pos
            if self.eating_sound:
                self.eating_sound.play()

        self.ctx.clear()
        self.scene.render(run_time, frame_time)
        self.ctx.enable(moderngl.BLEND)
        self.draw_all_text()

        if self.should_set_state_run:
            self.should_set_state_run = False
            state.run = True

    def game_over(self, game: Game) -> None:
        pass

    def run(self, game: Game) -> None:
        window = self.wnd

        self.timer.start()

        while not window.is_closing and (state.run or self.should_set_state_run):
            current_time, delta = self.timer.next_frame()

            window.use()
            window.render(current_time, delta)
            if not window.is_closing:
                window.swap_buffers()

        _, duration = self.timer.stop()
        self.wnd.destroy()
        if duration > 0:
            print(
                "Duration: {0:.2f}s @ {1:.2f} FPS".format(
                    duration, self.wnd.frames / duration
                )
            )

    def draw_fps(self) -> None:
        if self.recreate_text_renderer['fps']:
            self.fps_renderer = TextRenderer(
                self.font, f'FPS: {self.fps: 0.1f}', self.theme.text, 10, self.viewport_height - 10
            )
            self.recreate_text_renderer['fps'] = False
        else:
            self.fps_renderer.text = f'FPS: {self.fps: 0.1f}'
        self.fps_renderer.render(self.orthogonal_proj)

    def draw_score(self, score: int) -> None:
        if self.recreate_text_renderer['score']:
            self.score_renderer = TextRenderer(
                self.font, f'Score: {score}', self.theme.text, self.viewport_width / 2,
                self.viewport_height - 10, which_point='midtop'
            )
            self.recreate_text_renderer['score'] = False
        else:
            self.score_renderer.text = f'Score: {score}'
        self.score_renderer.render(self.orthogonal_proj)

    def draw_length(self, length: int) -> None:
        if self.recreate_text_renderer['length']:
            self.length_renderer = TextRenderer(
                self.font, f'Length: {length}', self.theme.text, self.viewport_width - 10,
                self.viewport_height - 10, which_point='topright'
            )
            self.recreate_text_renderer['length'] = False
        else:
            self.length_renderer.text = f'Length: {length}'
        self.length_renderer.render(self.orthogonal_proj)

    def draw_level_name(self) -> None:
        if self.recreate_text_renderer['level']:
            self.level_renderer = TextRenderer(
                self.font, f'Level: {state.level_name}', self.theme.text, self.viewport_width / 2,
                10, which_point='midbottom'
            )
            self.recreate_text_renderer['level'] = False
        self.level_renderer.render(self.orthogonal_proj)

    def draw_game_over_text(self):
        # We want to skip one rendering cycle before drawing game over and the high
        # scores because saving/reading the high scores takes a bit
        if self.recreate_text_renderer['game_over']:
            text = game_over_text(self.game, update_high_scores(self.game))
            self.game_over_renderer = TextRenderer(
                self.font, text, self.theme.text, self.viewport_width / 2,
                self.viewport_height * 0.7, which_point='midbottom'
            )
            self.recreate_text_renderer['game_over'] = False
        self.game_over_renderer.render(self.orthogonal_proj)

    def draw_high_scores(self, top_n=5) -> None:
        if self.recreate_text_renderer['high_scores']:
            names, scores, lengths = get_high_scores(top_n=top_n)
            scores = [' ' * (6 - len(s)) + s for s in scores]

            text = '\n'.join([
                ' Score  Length\n',
                '\n'.join(map('  '.join, zip(scores, lengths)))
            ])

            self.high_scores_renderer = TextRenderer(
                self.font, text, self.theme.text, self.viewport_width / 2,
                self.viewport_height * 0.4, which_point='midtop'
            )
            self.recreate_text_renderer['high_scores'] = False

            self.high_score_horiz_line = InstancedObject(
                1, geom.quad_2d, 'colored_quad', self.theme.text, [Transform3D()],
                vao_generator_kwargs={
                    'size': (self.font.char_width * 16, 6),
                    'pos': (
                        self.viewport_width / 2,
                        self.viewport_height * 0.4 - self.font.char_height * 1.5 + 3
                    )
                }
            )
            self.high_score_vert_line = InstancedObject(
                1, geom.quad_2d, 'colored_quad', self.theme.text, [Transform3D()],
                vao_generator_kwargs={
                    'size': (6, self.font.char_height * (top_n + 3)),
                    'pos': (
                        self.viewport_width / 2,
                        self.viewport_height * 0.4 - self.font.char_height * (top_n + 1.5) / 2
                    )
                }
            )

        self.high_score_horiz_line.render(write_uniforms={'Mvp': self.orthogonal_proj})
        self.high_score_vert_line.render(write_uniforms={'Mvp': self.orthogonal_proj})
        self.high_scores_renderer.render(self.orthogonal_proj)

    def draw_all_text(self):
        self.draw_fps()
        self.draw_length(len(self.game.snake))
        if self.game.game_over:
            if self.draw_game_over:
                self.draw_game_over_text()
                self.draw_high_scores()
            self.draw_game_over = True
        else:
            self.draw_score(self.game.score)
        self.draw_level_name()


renderers = {
    'PG': PGRenderer,
    'PG2': PGRenderer2,
    'PG3': PGRenderer3,
    'CL': CLRenderer,
    'OpenGL': OpenGLRenderer
}

if platform.system() != 'Windows':
    renderers['curses'] = CursesRenderer


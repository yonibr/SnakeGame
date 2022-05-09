# TODO:
#   - Exceptions for bad arguments

import argparse

from typing import Dict, List, Sequence, Tuple, Union

import state

from themes import themes
from utils import high_score_params

SPEED_OPTIONS = {
    1: 0.5,
    2: 0.25,
    3: 0.2,
    4: 0.175,
    5: 0.15,
    6: 0.125,
    7: 0.1,
    8: 0.075,
    9: 0.05,
    10: 0.025,
    11: 0.017
}

min_speed, max_speed = min(SPEED_OPTIONS.keys()), max(SPEED_OPTIONS.keys())

DEFAULT_SPEED = 5
DEFAULT_BOARD_WIDTH, DEFAULT_BOARD_HEIGHT = 32, 32
DEFAULT_SCALE_FACTOR = 30
DEFAULT_START_X = DEFAULT_START_Y = DEFAULT_BOARD_WIDTH // 2
DEFAULT_START_LEN = 3
MAX_KEY_QUEUE_DEPTH = 2
DEFAULT_RENDERER = 'PG3'
DEFAULT_LEVEL = 'Basic'

game_param_names = [
    'start_x', 'start_y', 'start_length', 'board_width', 'board_height', 'speed', 'level'
]


def add_args(parser: argparse.ArgumentParser) -> None:
    from renderers import renderers
    from snake import levels

    parser.add_argument(
        '--speed', default=DEFAULT_SPEED, type=int, choices=list(SPEED_OPTIONS.keys()),
        help=f'Integer between {min_speed} (slowest) and {max_speed} (fastest) representing' +
              ' how fast the snake moves.'
    )
    parser.add_argument(
        '-bw', '--board_width', default=DEFAULT_BOARD_WIDTH, type=int,
        help='The number of cells wide the board is.'
    )
    parser.add_argument(
        '-bh', '--board_height', default=DEFAULT_BOARD_HEIGHT, type=int,
        help='The number of cells tall the board is.'
    )
    parser.add_argument(
        '-sf', '--scale_factor', default=DEFAULT_SCALE_FACTOR, type=int,
        help='The number of times to repeat pixels when using pygame renderer.'
    )
    parser.add_argument(
        '-l', '--start_length', default=DEFAULT_START_LEN, type=int,
        help='The starting length of the snake.'
    )
    parser.add_argument(
        '-q', '--max_queue_depth', default=MAX_KEY_QUEUE_DEPTH, type=int,
        help='The maximum number of moves to queue.'
    )
    parser.add_argument(
        '-sx', '--start_x', type=int, default=DEFAULT_START_X,
        help='X coordinate of cell where the snake\'s head starts.'
    )
    parser.add_argument(
        '-sy', '--start_y', type=int, default=DEFAULT_START_Y,
        help='Y coordinate of cell where the snake\'s head starts.'
    )
    parser.add_argument(
        '--renderer', type=str, choices=renderers.keys(), default=DEFAULT_RENDERER,
        help='The renderer to use.'
    )
    parser.add_argument(
        '-t', '--theme', type=str, choices=themes.keys(), default='default'
    )
    parser.add_argument(
        '-es', '--enable_sound', help='Enables playing sounds in supported renderers.',
        action='store_true'
    )
    parser.add_argument('--level', default=DEFAULT_LEVEL, type=str, choices=levels.keys())

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '-g', '--show_grid', dest='show_grid', action='store_true',
        help='Show grid on supported renderers'
    )
    group.add_argument(
        '-hg', '--hide_grid', dest='show_grid', action='store_false',
        help='Don\'t show grid on supported renderers'
    )

    parser.set_defaults(show_grid=None)
    parser.set_defaults(enable_sound=False)


def parse_args() -> Tuple[List[int], Dict[str, Union[bool, str]], Dict[str, Union[float, int, str]]]:
    from renderers import config_defaults, renderers

    parser = argparse.ArgumentParser()

    add_args(parser)
    
    args = parser.parse_known_args()[0]

    for param in high_score_params:
        setattr(state, param, getattr(args, param))

    game_params = [getattr(args, param) for param in game_param_names]

    renderer_params = {
        k: v for k, v in vars(args).items() if k not in {
            *game_param_names, 'max_queue_depth', 'renderer'
        }
    }
    for param in config_defaults.keys():
        if renderer_params.get(param, None) is None:
            renderer_params[param] = renderers[args.renderer].default_config(param)

    other_params = {
        'tick_time': SPEED_OPTIONS[args.speed],
        'max_queue_depth': args.max_queue_depth,
        'renderer': args.renderer
    }

    return game_params, renderer_params, other_params

# TODO: maybe switch from parquet to feather

import operator
import pandas as pd
import platform
import threading
import time

from functools import reduce
from pygame.locals import K_UP, K_DOWN, K_LEFT, K_RIGHT, K_ESCAPE
from moderngl_window.context.pyglet import Keys as PglKeys
from pynput.keyboard import Key
from typing import Any, Callable, List, Optional, Tuple, Union

import state

from snake import Game

if platform.system() != 'Windows':
    import curses.ascii

high_score_params = ['start_length', 'board_width', 'board_height', 'speed', 'level']


# Modified from https://stackoverflow.com/a/48709380/2892775
class SetInterval(object):
    def __init__(
            self,
            interval: float,
            delay: float,
            action: Callable[..., Any],
            callback: Optional[Callable[[Any], Any]],
            *params: Any,
            wait_for_state_run: bool=True,
            **kwparams: Any):
        self.interval = interval
        self.delay = delay
        self.action = action
        self.callback = callback
        self.wait_for_state_run = wait_for_state_run
        self.stopEvent = threading.Event()
        self.params = params
        self.kwparams = kwparams
        thread = threading.Thread(target=self.__set_interval)
        thread.start()

    def __set_interval(self) -> None:
        while not state.run and self.wait_for_state_run:
            time.sleep(0.005)
        time.sleep(self.delay)
        next_time = time.time() + self.interval
        while not self.stopEvent.wait(next_time - time.time()):
            next_time += self.interval
            ret_val = self.action(*self.params, **self.kwparams)
            if self.callback:
                self.callback(ret_val)

    def cancel(self) -> None:
        self.stopEvent.set()


key_maps = {
    'pynput': {
        Key.up: 'up',
        Key.down: 'down',
        Key.left: 'left',
        Key.right: 'right',
        Key.esc: 'escape'
    },
    'pygame': {
        K_UP: 'up',
        K_DOWN: 'down',
        K_LEFT: 'left',
        K_RIGHT: 'right',
        K_ESCAPE: 'escape'
    },
    'pyglet': {
        PglKeys.UP: 'up',
        PglKeys.DOWN: 'down',
        PglKeys.LEFT: 'left',
        PglKeys.RIGHT: 'right',
        PglKeys.ESCAPE: 'escape'
    }
}

if platform.system() != 'Windows':
    key_maps['curses'] = {
        curses.KEY_UP: 'up',
        curses.KEY_DOWN: 'down',
        curses.KEY_LEFT: 'left',
        curses.KEY_RIGHT: 'right',
        curses.ascii.ESC: 'escape'
    }


def parse_key(key: Union[int, Key], input_library) -> Optional[str]:
    return key_maps[input_library].get(key)


def game_over_text(game: Game, got_high_score: bool) -> str:
    return '\n'.join(filter(None, [
        'Game Over!',
        'You Won!' if game.won else 'Better luck next time!',
        'New High Score!' if got_high_score else '',
        f'Score: {game.score}'
    ]))


def get_same_param_score_df(score_df: pd.DataFrame) -> pd.DataFrame:
    indices = reduce(
        operator.and_,
        (score_df.get(param) == getattr(state, param) for param in high_score_params)
    )
    return score_df[indices].reset_index(drop=True)


def update_high_scores(game: Game, check_top_n: int=5) -> bool:
    score = game.score
    length = len(game.snake)
    row = {
        'score': score,
        'length': length,
        **{param: getattr(state, param) for param in high_score_params}
    }

    try:
        score_df = pd.read_parquet('scores.parquet')
        score_df = score_df.append(row, ignore_index=True)
        score_df.sort_values(
            ['score', 'length'], ignore_index=True, inplace=True, ascending=False
        )

        same_param_score_df = get_same_param_score_df(score_df)

        lowest_high_score = same_param_score_df.iloc[
            min(check_top_n, len(same_param_score_df)) - 1
        ]
        high_score = (
            (lowest_high_score.score < score) or
            (
                lowest_high_score.score == score and lowest_high_score.length < length
            )
        )
    except FileNotFoundError:
        score_df = pd.DataFrame(data=row, index=[0])
        high_score = True

    score_df.to_parquet('scores.parquet')

    return high_score


def get_high_scores(top_n: int=5) -> Tuple[List[str], List[str]]:
    high_scores_df = get_same_param_score_df(pd.read_parquet(f'scores.parquet')).head(top_n)

    return (
        high_scores_df.score.astype(str).to_list(),
        high_scores_df.length.astype(str).to_list()
    )

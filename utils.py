import pandas as pd
import platform
import sqlite3 as sl
import threading
import time

from enum import Enum
from pygame.locals import K_UP, K_DOWN, K_LEFT, K_RIGHT, K_ESCAPE
from moderngl_window.context.pyglet import Keys as PglKeys
from pynput.keyboard import Key
from typing import Any, Callable, List, Optional, Tuple, Union

import state

from snake import Game

if platform.system() != 'Windows':
    import curses.ascii

high_score_params = ['start_length', 'board_width', 'board_height', 'speed', 'level']

db_conn = sl.connect('snake.db')

with db_conn:
    db_conn.execute('''
    create table if not exists scores (
        player_name text,
        start_length integer,
        board_width integer,
        board_height integer,
        speed integer,
        level text,
        score integer,
        length integer
    );
    ''')


class HorizontalTextAlignment(Enum):
    LEFT = 'left'
    CENTERED = 'centered'
    RIGHT = 'right'


class RectanglePoint(Enum):
    TOP_LEFT = 'topleft'
    MID_TOP = 'midtop'
    TOP_RIGHT = 'topright'
    MID_LEFT = 'midleft'
    CENTER = 'center'
    MID_RIGHT = 'midright'
    BOTTOM_LEFT = 'bottomleft'
    MID_BOTTOM = 'midbottom'
    BOTTOM_RIGHT = 'bottomright'


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


def update_high_scores(game: Game, check_top_n: int=5) -> bool:
    global db_conn
    try:
        name = state.player_name
        score = game.score
        data = [
            name,
            state.start_length,
            state.board_width,
            state.board_height,
            state.speed,
            state.level_name,
            score,
            len(game.snake)
        ]
        with db_conn:
            db_conn.execute('''
                insert into scores
                    (player_name, start_length, board_width, board_height, speed, level, score, length)
                     values(?, ?, ?, ?, ?, ?, ?, ?)
            ''', data)

            min_high_score_name, min_high_score = db_conn.execute(f'''
                select
                    player_name,
                    score
                from scores
                where
                    level = "{state.level_name}" and
                    start_length = {state.start_length} and
                    board_width = {state.board_width} and
                    board_height = {state.board_height} and
                    speed = {state.speed}
                order by score desc, rowid asc
                limit {check_top_n}
            ''').fetchall()[-1]

        return score > min_high_score or (score == min_high_score and name == min_high_score_name)
    # If db_conn was created on a different thread, we need to re-create it
    except sl.ProgrammingError:
        db_conn = sl.connect('snake.db')
        return update_high_scores(game, check_top_n=check_top_n)


def get_high_scores(
        top_n: int=5,
        as_dataframe: bool=False) -> Union[Tuple[List[str], List[str], List[str]], pd.DataFrame]:
    global db_conn
    try:
        high_scores_df = pd.read_sql(f'''
            select
                player_name,
                score,
                length
            from scores
            where
                level = "{state.level_name}" and
                start_length = {state.start_length} and
                board_width = {state.board_width} and
                board_height = {state.board_height} and
                speed = {state.speed}
            order by score desc, rowid asc
            limit {top_n}
        ''', db_conn)

        return high_scores_df if as_dataframe else (
            high_scores_df.player_name.to_list(),
            high_scores_df.score.astype(str).to_list(),
            high_scores_df.length.astype(str).to_list()
        )
    # If db_conn was created on a different thread, we need to re-create it
    except sl.ProgrammingError:
        db_conn = sl.connect('snake.db')
        return get_high_scores(top_n=top_n, as_dataframe=as_dataframe)

# TODO:
#   - Optionally allow speed increase if you press the same direction you're already travelling
#   - Don't initialize pygame when renderer isn't a pygame renderer
#   - Add value sanity checking to command line arguments
#   - In-game way to view high scores for games with different configurations (with scrolling or
#     pages, selecting filters for what configurations to show, etc.)
#   - Ability to start a new game without quitting
#   - Ability to configure without needing command line arguments
#   = Two-player version that's a combination of snake and tron (players eat the food to make
#     their tails longer and try to get the other player to crash into the wall or their tail)
#   - Allow specifying name for high scores
#   - OpenGL renderer
#   - Starting countdown

import time

from collections import deque
from pynput import keyboard
from threading import RLock, Thread

import state

from snake import Direction, Game
from snake_args import parse_args
from utils import parse_key, SetInterval

key_dir_map = {
    'down': Direction.down(),
    'up': Direction.up(),
    'left': Direction.left(),
    'right': Direction.right()
}


def is_valid_direction(direction: Direction) -> bool:
    if len(state.direction_queue) == 0:
        curr_direction = state.game.snake.direction
    else:
        curr_direction = state.direction_queue[0]

    return not direction.is_opposite(curr_direction) and direction != curr_direction


def handle_input(key: str, released: bool, ignore_released: bool=False) -> None:
    direction = key_dir_map.get(key)
    if (not released or ignore_released) and direction:
        with state.direction_queue_lock:
            if len(state.direction_queue) < state.max_key_queue_depth and is_valid_direction(direction):
                state.direction_queue.appendleft(direction)
    elif (released or ignore_released) and key == 'escape':
        exit_game()


def on_press(key):
    # Try to clear the keypress
    print('\b\b\b\b', end='\r')

    handle_input(parse_key(key, 'pynput'), False)

    return True


def on_release(key):
    # Try to clear the keypress
    print('\b\b\b\b\b', end='\r')
    print('    \b\b\b\b', end='\r')

    key = parse_key(key, 'pynput')
    handle_input(key, True)

    return key != 'escape'


def callback(game_over: bool) -> None:
    if game_over:
        state.interval.cancel()


def exit_game() -> None:
    if state.interval:
        state.interval.cancel()
    state.run = False


def loop(game: Game) -> bool:
    if state.run:
        if state.input_hook:  # TODO instead of adding to direction_queue, make input_hook have get_input
            state.input_hook.run()

        with state.direction_queue_lock:
            if len(state.direction_queue):
                game.snake.direction = state.direction_queue.pop()

        game.tick()

    return game.game_over


random_counter = 0


def rl_loop(env: 'Env', model) -> bool: # TODO provide type hint for model
    # action = env.action_space.sample()
    # obs, reward, done, info = env.step(action)

    global random_counter

    if random_counter < 250:
        action, _states = model.predict(env.get_observation(), deterministic=True)
    else:
        action = env.action_space.sample()
        random_counter = 0

    obs, reward, done, info = env.step(action)

    if reward <= 0:
        random_counter += 1
    else:
        random_counter = 0

    if done:
        env.reset()
        state.game = env.game
        state.run = False

    return done


def run_recurr(env: 'Env', model) -> None:
    global run_control_thread

    while run_control_thread:
        obs = env.reset()

        state.game = env.envs[0].env.game

        while not state.run:
            time.sleep(0.001)

        _state = None
        done = [False for _ in range(env.num_envs)]            
        while state.run:
            time.sleep(0.1)
            action, _state = model.predict(obs, state=_state, mask=done, deterministic=True)
            obs, reward, done, _ = env.step(action)
            if state.game.game_over:
                state.run = False


def main():
    # Import renderers.renderers here so when renderers.py imports exit_game
    # and handle_input, there's no exception
    from renderers import renderers

    # from computer_player import RandomPlayer
    # state.input_hook = RandomPlayer()

    game_params, renderer_params, other_params = parse_args()

    state.game = Game(*game_params)

    state.level_name = game_params[-1]
    state.max_key_queue_depth = other_params['max_queue_depth']
    state.direction_queue = deque()
    state.direction_queue_lock = RLock()

    renderer_name = other_params['renderer']

    tick_time = other_params['tick_time']

    if renderer_name != 'OpenGL':
        renderer = renderers[renderer_name]()

        renderer.initialize(state.game, tick_time=tick_time, **renderer_params)

        if renderer_name == 'CL':
            listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            listener.start()

        state.interval = SetInterval(tick_time, 2, loop, callback, state.game)
        state.run = True

        renderer.run(state.game)
    else:
        state.interval = SetInterval(tick_time, 5, loop, callback, state.game)
        # Note: OpenGL renderer needs to set state.run = True because it can take a while to start up
        renderers[renderer_name].start()


def rl_main():
    from renderers import renderers
    from reinforcement_learning import create_cnn_lstm_ppo2_model

    game_params, renderer_params, other_params = parse_args()

    state.level_name = game_params[-1] + ' — Reinforcement Learning'
    state.max_key_queue_depth = other_params['max_queue_depth']
    state.direction_queue = deque()
    state.direction_queue_lock = RLock()

    renderer_name = other_params['renderer']

    state.run = True

    renderer = renderers[renderer_name]()

    tick_time = other_params['tick_time']

    filename = f'{state.board_width}x{state.board_height}_ppo2_cnn_lstm.model'
    model, env = create_cnn_lstm_ppo2_model(
        filename, game_params, iters=state.rl_training_iters, gamma_start=0.989,
        gamma_stop=.991, taper_steps=0
    )

    global run_control_thread
    run_control_thread = True

    control_thread = Thread(target=run_recurr, args=(env, model))
    control_thread.start()

    while not state.game:
        pass

    renderer.initialize(state.game, tick_time=.017, **renderer_params)

    counter = 0
    while counter < 10000:
        if state.game.game_over:
            time.sleep(0.001)
        renderer.draw_game_over = False
        renderer.drew_game_over = False
        renderer.first_render = True
        state.run = True
        renderer.run(state.game)
        counter += 1

    run_control_thread = False

    control_thread.join()

    exit_game()


if __name__ == '__main__':
    import cProfile
    cProfile.run('main()', 'restats')
    # main()
    # rl_main()

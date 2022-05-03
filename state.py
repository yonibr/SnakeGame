# This is just a module used to share state across other modules. Probably bad design
# but whatevskies.
import numpy as np
from threading import Lock

level_name = None
direction_queue = None
run = None
max_key_queue_depth = None
game = None
direction_queue_lock = None
interval = None
start_length = None
board_width = None
board_height = None
speed = None
input_hook = None
shader_program_repo = None

############ Reinforcement Learning Training Variables ##############
model_snapshot_freq = 5
rl_training_iters = 120000000
episodes_to_change_level_over = rl_training_iters // 2500
episode_counter = 0
episode_counter_lock = Lock()

# Level probability interpolations
points = (
    0,
    episodes_to_change_level_over // 5,
    2 * episodes_to_change_level_over // 5,
    3 * episodes_to_change_level_over // 5,
    4 * episodes_to_change_level_over // 5
)
basic_vals = 1, 0.8, 0.15, 0.1, 0.05
viewfinder_vals = 0, 0.2, 0.8, 0.4, 0.15
random_vals = 0, 0, 0.05, 0.5, 0.8

level_probabilities = np.array(list(zip(
    np.interp(np.arange(0, episodes_to_change_level_over), points, basic_vals),
    np.interp(np.arange(0, episodes_to_change_level_over), points, viewfinder_vals),
    np.interp(np.arange(0, episodes_to_change_level_over), points, random_vals)
)))


def get_probabilities():
    with episode_counter_lock:
        global episode_counter, level_probabilities
        probabilities = level_probabilities[min(episode_counter, rl_training_iters - 1)]
        episode_counter += 1
        return probabilities

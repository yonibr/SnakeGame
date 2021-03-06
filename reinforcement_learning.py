import gym
import numpy as np
import tensorflow as tf

from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.schedules import Schedule
from stable_baselines.common.tf_layers import conv, linear, conv_to_fc
from typing import Any, Optional, Sequence, Tuple

from stable_baselines.common.vec_env import VecEnv

import state

from snake import Direction, Game

# Note: for moving walls, we'll want to have the observation space be normal 
# observation space * number of frames (using gym.Wrapper). See
# https://blog.paperspace.com/getting-started-with-openai-gym/

action_map = {
    0: Direction.up(),
    1: Direction.down(),
    2: Direction.left(),
    3: Direction.right()
}


rng = np.random.default_rng()


class SnakeEnv(gym.Env):
    def __init__(self, game_params, cnn_policy=False, randomize_start_length=True):
        self.randomize_start_length = randomize_start_length
        self.game_params = game_params
        self.max_start_length = min(game_params[3], game_params[4]) // 2
        self.update_start_length()
        self.game = Game(*game_params)
        self.food_pos = None
        self.foods_eaten = 0
        self.steps_since_food = 0
        self.observation_shape = self.game.board_width, self.game.board_height, 4

        self.action_space = gym.spaces.Discrete(4)

        high = np.ones(self.observation_shape)
        if cnn_policy:
            high *= 255

        self.observation_space = gym.spaces.Box(
            low=np.zeros(self.observation_shape),
            high=high,
            dtype=np.int16
        )
        self.max_dim = max(self.game.board_width, self.game.board_height)
        self.volume = self.food_multiplier = None
        self.update_vol_and_food_mult()
        self.end_length = -1
        self.volume_filled = -1
        self.did_end = False
        self.on_value = 255 if cnn_policy else 1

    def get_observation(self) -> np.ndarray:
        # 0 - wall
        # 1 - snake body/tail
        # 2 - food
        # 3 - snake head
        observation = np.zeros(self.observation_shape)

        observation[self.game.food.y, self.game.food.x, 2] = self.on_value

        for node in self.game.snake:
            if node is not self.game.snake.head:
                observation[node.y, node.x, 1] = self.on_value
            else:
                observation[node.y, node.x, 3] = self.on_value
        for node in self.game.level.wall_nodes:  # TODO figure out if I want to also set enclosed spaces to 1
            observation[node.y, node.x, 0] = self.on_value

        return observation

    @staticmethod
    def get_action_meanings():
        return {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right', 4: 'No Change'}

    def reset(self) -> np.ndarray:
        # self.end_length = len(self.game.snake)
        self.end_length = self.foods_eaten
        self.volume_filled = len(self.game.snake) / self.volume
        self.did_end = True
        self.update_start_length()
        self.game = Game(*self.game_params)
        self.food_pos = None
        self.foods_eaten = 0
        self.steps_since_food = 0
        self.update_vol_and_food_mult()

        return self.get_observation()

    def step(self, action: int):
        direction = action_map.get(action)

        if direction:
            self.game.snake.direction = direction

        head_pos = self.game.snake.head.pos
        food_pos = self.game.food.pos
        distance_to_food = abs(food_pos[0] - head_pos[0]) + abs(food_pos[1] - head_pos[1])

        self.game.tick()

        head_pos = self.game.snake.head.pos
        new_distance_to_food = abs(food_pos[0] - head_pos[0]) + abs(food_pos[1] - head_pos[1])

        self.steps_since_food += 1
        if self.game.won:
            reward = 10
        elif self.game.game_over:
            reward = -1
        elif self.game.food.pos != food_pos:
            self.foods_eaten += 1
            reward = self.food_multiplier
            self.steps_since_food = 0
        else:
            reward = (distance_to_food - new_distance_to_food) * 0.1 - 1 / (self.volume + 2)

        early_stop = min(self.max_dim * 4 * (self.foods_eaten + 1)**0.5, self.volume * 1.2) < self.steps_since_food

        return self.get_observation(), reward, self.game.game_over or early_stop, {}

    def update_start_length(self):
        if self.randomize_start_length:
            self.game_params[2] = rng.integers(1, self.max_start_length)

    def update_vol_and_food_mult(self):
        board_interior_size = (self.game.board_width - 2) * (self.game.board_height - 2)
        outer_wall_size = len(self.game) - board_interior_size
        inner_wall_size = len(self.game.level) - outer_wall_size
        self.volume = board_interior_size - inner_wall_size
        self.food_multiplier = board_interior_size / self.volume

    def render(self, mode: str='human'):
        if mode == 'human':
            print(self.game)
        else:
            raise NotImplementedError()


def modified_cnn(scaled_images, **kwargs):
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = conv_to_fc(layer_3)
    return activ(linear(layer_3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))


def modified_cnn2(scaled_images, **kwargs):
    activ = tf.nn.swish
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=7, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=5, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = conv_to_fc(layer_3)
    return activ(linear(layer_3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))


def modified_cnn3(scaled_images, **kwargs):
    activ = tf.nn.swish
    layer_1 = activ(conv(
        scaled_images, 'c1', n_filters=32, filter_size=5, stride=3, pad='SAME',
        init_scale=np.sqrt(2), **kwargs
    ))
    layer_2 = activ(conv(
        layer_1, 'c2', n_filters=64, filter_size=4, stride=2, pad='SAME',
        init_scale=np.sqrt(2), **kwargs
    ))
    layer_3 = activ(conv(
        layer_2, 'c3', n_filters=64, filter_size=3, stride=1, pad='SAME',
        init_scale=np.sqrt(2), **kwargs
    ))
    layer_3 = conv_to_fc(layer_3)
    return activ(linear(layer_3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))


def modified_cnn4(scaled_images, **kwargs):
    activ = tf.nn.swish
    layer_1 = activ(conv(
        scaled_images, 'c1', n_filters=32, filter_size=8, stride=4, pad='SAME',
        init_scale=np.sqrt(2), **kwargs
    ))
    layer_2 = activ(conv(
        layer_1, 'c2', n_filters=64, filter_size=4, stride=2, pad='SAME',
        init_scale=np.sqrt(2), **kwargs
    ))
    layer_3 = activ(conv(
        layer_2, 'c3', n_filters=64, filter_size=3, stride=1, pad='SAME',
        init_scale=np.sqrt(2), **kwargs
    ))
    layer_3 = conv_to_fc(layer_3)
    return activ(linear(layer_3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))


# class CustomCnnLnLstmPolicy(LstmPolicy):
#     """
#     Policy object that implements actor critic, using a layer normalized LSTMs with a CNN feature extraction
# 
#     :param sess: (TensorFlow session) The current TensorFlow session
#     :param ob_space: (Gym Space) The observation space of the environment
#     :param ac_space: (Gym Space) The action space of the environment
#     :param n_env: (int) The number of environments to run
#     :param n_steps: (int) The number of steps to run for each environment
#     :param n_batch: (int) The number of batch to run (n_envs * n_steps)
#     :param n_lstm: (int) The number of LSTM cells (for recurrent policies)
#     :param reuse: (bool) If the policy is reusable or not
#     :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
#     """
#     def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False, **_kwargs):
#         super(CnnLnLstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
#                                               layer_norm=True, feature_extraction="cnn", cnn_extractor=modified_cnn, **_kwargs)


# Note: This is a fixed version of the stable-baselines LinearSchedule.
# PPO2 expects value to take the elapsed fraction, not timesteps.
class LinearSchedule(Schedule):
    """
    Linear interpolation between initial_p and final_p over
    schedule_timesteps. After this many timesteps pass final_p is
    returned.
    :param schedule_timesteps: (int) Number of timesteps for which to linearly anneal initial_p to final_p
    :param initial_p: (float) initial output value
    :param final_p: (float) final output value
    """

    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, fraction):
        return self.initial_p + (1 - fraction) * (self.final_p - self.initial_p)


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=1):
        self.is_tb_set = False
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log additional tensor
        if not self.is_tb_set:
            with self.model.graph.as_default():
                # tf.summary.scalar('value_target', tf.reduce_mean(self.model.value_target))
                self.model.summary = tf.summary.merge_all()
            self.is_tb_set = True

        envs = []
        if isinstance(self.training_env, VecEnv):
            envs.extend(self.training_env.envs)
        elif isinstance(self.training_env, gym.Env):
            envs.append(self.training_env)

        for env in envs:
            env = env.env
            if env.did_end:
                summary = tf.Summary(value=[
                    tf.Summary.Value(tag='snake_length', simple_value=env.end_length),
                    tf.Summary.Value(tag='frac_volume_occupied', simple_value=env.volume_filled)
                ])
                self.locals['writer'].add_summary(summary, self.num_timesteps)
                env.did_end = False
                env.end_length = env.volume_filled = -1

        return True


#  gamma   half_life
# 0.96500  19.455574
# 0.97000  22.756573
# 0.97500  27.377851
# 0.98000  34.309618
# 0.98250  39.260817
# 0.98500  45.862365
# 0.98750  55.104474
# 0.99000  68.967563
# 0.99200  86.296360
# 0.99400  115.177609
# 0.99500  138.282573
# 0.99600  172.939990
# 0.99700  230.702313
# 0.99750  276.912154
# 0.99800  346.226901
# 0.99850  461.751460
# 0.99900  692.800549
# 0.99925  923.849624
def create_cnn_lstm_ppo2_model(
        save_location: str,
        game_params: Sequence[Any],
        iters: int=7500000,
        verbose: int=1,
        gamma_start: float=0.99,
        gamma_stop: Optional[float]=None,
        taper_steps: int=100,
        lr_start: int=7.5e-4,
        lr_stop: int=5e-4,
        start_steps=2000,
        end_steps=10000,
        step_incr_freq=5,
        tb_log=r'D:\snake_tb_logs') -> Tuple[PPO2, gym.Env]:
    from stable_baselines.common.policies import CnnLnLstmPolicy
    import gc

    taper_steps = max(taper_steps, 1)
    env = make_vec_env(SnakeEnv, n_envs=64, env_kwargs={'game_params': game_params, 'cnn_policy': True})

    if gamma_stop and iters > 1:
        gammas = np.linspace(gamma_start, gamma_stop, taper_steps)
    else:
        gammas = [gamma_start]

    n_steps = start_steps
    step_incr_val = 0
    if step_incr_freq > 0 and start_steps != end_steps:
        step_incr_val = step_incr_freq * (end_steps - start_steps) // taper_steps

    cb = TensorboardCallback()

    step_iters = iters // taper_steps
    lr_ranges = list(np.linspace(lr_start, lr_stop, taper_steps + 1))
    lr_schedule = LinearSchedule(step_iters, lr_ranges[1], initial_p=lr_ranges[0])
    counter = 0
    model = None
    for i in range(taper_steps):
        lr_schedule.initial_p = lr_ranges[i]
        lr_schedule.final_p = lr_ranges[i + 1]
        try:
            if gammas[i] == gamma_start or (step_incr_freq > 0 and counter % step_incr_freq == 0):
                if model:
                    del model
                    gc.collect()
                model = PPO2.load(
                    save_location, env=env, n_steps=n_steps, verbose=verbose, gamma=gammas[i],
                    learning_rate=lr_schedule.value, ent_coef=.025, cliprange=0.225,
                    nminibatches=32, tensorboard_log=tb_log, noptepochs=4
                )
                print('Loaded existing model from:', save_location)
                n_steps += step_incr_val
            else:
                model.gamma = gammas[i]
        except ValueError as e:
            print(e)
            print('Creating new model...')
            model = PPO2(
                CnnLnLstmPolicy, env, verbose=verbose, gamma=gammas[i], n_steps=1000, learning_rate=lr_schedule.value,
                cliprange=0.275, ent_coef=.025, noptepochs=4, tensorboard_log=tb_log, nminibatches=32,
                policy_kwargs={'cnn_extractor': modified_cnn4, 'n_lstm': 512}
            )

        if iters > 0:
            model.learn(total_timesteps=step_iters, callback=cb, reset_num_timesteps=False)
            model.save(save_location)
            counter += 1
            if state.model_snapshot_freq > 0 and counter % state.model_snapshot_freq == 0:
                model.save(f'D:/snake_model_snapshots/snapshot_{counter}.model')
            env.reset()
            print('Saved model to:', save_location)
            gc.collect()

    return model, env

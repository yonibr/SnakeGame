import gym
import numpy as np
import tensorflow as tf

from stable_baselines import ACER, DQN, PPO2
from stable_baselines.common import make_vec_env
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.tf_layers import conv, linear, conv_to_fc
from typing import Any, Optional, Sequence, Tuple

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

direction_state_map = {
    'up': 4,
    'down': 5,
    'left': 6,
    'right': 7
}

rng = np.random.default_rng()


class SnakeEnv(gym.Env):
    def __init__(self, game_params):
        self.game_params = game_params
        self.game = Game(*game_params)
        self.food_pos = None
        self.foods_eaten = 0
        self.steps_since_food = 0
        self.observation_shape = self.game.board_width, self.game.board_height, 4
        
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(
            low=np.zeros(self.observation_shape),
            high=np.ones(self.observation_shape),
            dtype=np.int16
        )

        self.do_nothing_threshold = max(self.game.board_width, self.game.board_height) * 5

        self.end_length = -1
        self.did_end = False

    def get_observation(self) -> np.ndarray:
        # 0 - wall
        # 1 - snake body/tail
        # 2 - food
        # 3 - snake head
        observation = np.zeros(self.observation_shape)

        observation[self.game.food.y, self.game.food.x, 2] = 1

        for node in self.game.snake:
            if node is not self.game.snake.head:
                observation[node.y, node.x, 1] = 1
            else:
                observation[node.y, node.x, 3] = 1
        for node in self.game.level.wall_nodes: # TODO figure out if I want to also set enclosed spaces to 1
            observation[node.y, node.x, 0] = 1

        return observation

    def get_action_meanings(self):
        return {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right', 4: 'No Change'}

    def reset(self) -> np.ndarray:
        self.end_length = len(self.game.snake)
        self.did_end = True
        self.game = Game(*self.game_params)
        self.food_pos = None
        self.foods_eaten = 0
        self.steps_since_food = 0

        return self.get_observation()

    def step(self, action):
        direction = action_map.get(action)

        if direction:
            self.game.snake.direction = direction

        self.game.tick()
        self.steps_since_food += 1
        if self.game.won:
            reward = 5000
        elif self.game.game_over:
            reward = min(-10 / (self.foods_eaten + 1), -1)
        elif self.game.food.pos != self.food_pos:
            reward = 0 if self.food_pos is None else 10
            self.foods_eaten += 1 if self.food_pos is not None else 0
            self.food_pos = self.game.food.pos
            self.steps_since_food = 0
        elif self.foods_eaten < 1:
            reward = 0.01 / self.steps_since_food**2.1
        elif self.foods_eaten < 3:
            reward = 0.0001 / self.steps_since_food**2.1
        elif self.foods_eaten < 5:
            reward = 0
        else:
            reward = -.0000004

        doing_nothing_penalty = self.steps_since_food / (self.foods_eaten + 1)
        if doing_nothing_penalty > self.do_nothing_threshold:
            reward -= 0.0005 * doing_nothing_penalty

        return self.get_observation(), reward, self.game.game_over, {}


class SnakeEnv2(gym.Env):
    def __init__(self, game_params):
        self.game_params = game_params
        self.game = Game(*game_params)
        self.food_pos = None
        self.foods_eaten = 0
        self.steps_since_food = 0
        self.observation_shape = self.game.board_width, self.game.board_height, 4
        
        self.action_space = gym.spaces.Discrete(4)

        high = np.ones(self.observation_shape)
        high[:, :, 1] = len(self.game) - len(self.game.level)

        self.observation_space = gym.spaces.Box(
            low=np.zeros(self.observation_shape),
            high=high,
            dtype=np.int16
        )
        self.do_nothing_threshold = max(self.game.board_width, self.game.board_height) * 5

        self.end_length = -1
        self.did_end = False

    def get_observation(self) -> np.ndarray:
        # 0 - wall
        # 1 - snake body/tail
        # 2 - food
        # 3 - snake head
        observation = np.zeros(self.observation_shape)

        observation[self.game.food.y, self.game.food.x, 2] = 1

        # TODO for snake tail feature, might want a number to indicate how many more turns a specific grid location will be filled for
        for node, fill_remaining in zip(self.game.snake, range(len(self.game.snake), 0, -1)):
            if node is not self.game.snake.head:
                observation[node.y, node.x, 1] = fill_remaining
            else:
                observation[node.y, node.x, 3] = 1
        for node in self.game.level.wall_nodes: # TODO figure out if I want to also set enclosed spaces to 1
            observation[node.y, node.x, 0] = 1


        return observation

    def get_action_meanings(self):
        return {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right', 4: 'No Change'}

    def reset(self) -> np.ndarray:
        self.end_length = len(self.game.snake)
        self.did_end = True
        self.game = Game(*self.game_params)
        self.food_pos = None
        self.foods_eaten = 0
        self.steps_since_food = 0

        return self.get_observation()

    def step(self, action):
        direction = action_map.get(action)

        if direction:
            self.game.snake.direction = direction

        self.game.tick()
        self.steps_since_food += 1
        if self.game.won:
            reward = 5000
        elif self.game.game_over:
            reward = min(-12 / (self.foods_eaten + 1)**0.5, -1.25)
        elif self.game.food.pos != self.food_pos:
            self.foods_eaten += 1 if self.food_pos is not None else 0
            reward = 10 * self.foods_eaten**.5
            self.food_pos = self.game.food.pos
            self.steps_since_food = 0
        elif self.foods_eaten < 1:
            reward = 0.01 / self.steps_since_food**2.1
        elif self.foods_eaten < 3:
            reward = 0.0001 / self.steps_since_food**2.1
        elif self.foods_eaten < 5:
            reward = 0
        else:
            reward = -.0000003

        doing_nothing_penalty = self.steps_since_food / (self.foods_eaten + 1)**.75
        if doing_nothing_penalty > self.do_nothing_threshold:
            reward -= 0.0005 * doing_nothing_penalty

        return self.get_observation(), reward, self.game.game_over, {}


class SnakeEnv3(gym.Env):
    def __init__(self, game_params, cnn_policy=False):
        self.game_params = game_params
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
        self.do_nothing_threshold = self.max_dim * 5
        self.end_length = -1
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
        for node in self.game.level.wall_nodes: # TODO figure out if I want to also set enclosed spaces to 1
            observation[node.y, node.x, 0] = self.on_value

        return observation

    @staticmethod
    def get_action_meanings():
        return {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right', 4: 'No Change'}

    def reset(self) -> np.ndarray:
        self.end_length = len(self.game.snake)
        self.did_end = True
        self.game = Game(*self.game_params)
        self.food_pos = None
        self.foods_eaten = 0
        self.steps_since_food = 0

        return self.get_observation()

    def step(self, action):
        direction = action_map.get(action)

        if direction:
            self.game.snake.direction = direction

        self.game.tick()
        self.steps_since_food += 1
        if self.game.won:
            reward = 10000
        elif self.game.game_over:
            # reward = min(-13 / (self.foods_eaten + 1)**0.5, -1.5)
            reward = -10 - self.foods_eaten**0.49
        elif self.game.food.pos != self.food_pos:
            self.foods_eaten += 1 if self.food_pos is not None else 0
            reward = 12 * self.foods_eaten**0.555
            self.food_pos = self.game.food.pos
            self.steps_since_food = 0
        elif self.foods_eaten < 1:
            reward = 0.01 / self.steps_since_food**2.1
        elif self.foods_eaten < 3:
            reward = 0.0001 / self.steps_since_food**2.1
        elif self.foods_eaten < 10:
            reward = 0.000025 / self.steps_since_food**2.1
        elif self.foods_eaten < 15 or self.steps_since_food < self.max_dim:
            reward = 0
        else:
            reward = -.0000002

        doing_nothing_penalty = self.steps_since_food / (self.foods_eaten + 1)**.75
        if doing_nothing_penalty > self.do_nothing_threshold:
            reward -= 0.0005 * doing_nothing_penalty

        return self.get_observation(), reward, self.game.game_over, {}


class SnakeEnv4(gym.Env):
    def __init__(self, game_params, cnn_policy=False):
        self.game_params = game_params
        self.game = Game(*game_params)
        self.food_pos = None
        self.foods_eaten = 0
        self.steps_since_food = 0
        self.observation_shape = self.game.board_width, self.game.board_height, 3
        
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
        self.do_nothing_threshold = self.max_dim * 5
        self.end_length = -1
        self.did_end = False
        self.on_value = 255 if cnn_policy else 1

    def get_observation(self) -> np.ndarray:
        # 0 - wall
        # 1 - snake body/tail
        # 2 - food
        # 3 - snake head
        observation = np.zeros(self.observation_shape)

        observation[self.game.food.y, self.game.food.x, 1] = self.on_value

        for node in self.game.snake:
            if node is not self.game.snake.head:
                observation[node.y, node.x, 0] = self.on_value
            else:
                observation[node.y, node.x, 2] = self.on_value
        for node in self.game.level.wall_nodes: # TODO figure out if I want to also set enclosed spaces to 1
            observation[node.y, node.x, 0] = self.on_value


        return observation

    def get_action_meanings(self):
        return {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right', 4: 'No Change'}

    def reset(self) -> np.ndarray:
        self.end_length = len(self.game.snake)
        self.did_end = True
        self.game = Game(*self.game_params)
        self.food_pos = None
        self.foods_eaten = 0
        self.steps_since_food = 0

        return self.get_observation()

    def step(self, action):
        direction = action_map.get(action)

        if direction:
            self.game.snake.direction = direction

        self.game.tick()
        self.steps_since_food += 1
        if self.game.won:
            reward = 10000
        elif self.game.game_over:
            # reward = min(-13 / (self.foods_eaten + 1)**0.5, -1.5)
            reward = -10 - self.foods_eaten**0.475
        elif self.game.food.pos != self.food_pos:
            self.foods_eaten += 1 if self.food_pos is not None else 0
            reward = 10 * self.foods_eaten**0.55
            self.food_pos = self.game.food.pos
            self.steps_since_food = 0
        elif self.foods_eaten < 1:
            reward = 0.01 / self.steps_since_food**2.1
        elif self.foods_eaten < 3:
            reward = 0.0001 / self.steps_since_food**2.1
        elif self.foods_eaten < 10:
            reward = 0.000025 / self.steps_since_food**2.1
        elif self.foods_eaten < 15 or self.steps_since_food < self.max_dim:
            reward = 0
        else:
            reward = -.000000075

        doing_nothing_penalty = self.steps_since_food / (self.foods_eaten + 1)**.75
        if doing_nothing_penalty > self.do_nothing_threshold:
            reward -= 0.0005 * doing_nothing_penalty

        return self.get_observation(), reward, self.game.game_over, {}


class SnakeEnv5(gym.Env):
    def __init__(self, game_params, cnn_policy=False, randomize_start_length=True):
        self.randomize_start_length = randomize_start_length
        self.game_params = game_params
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

    def get_action_meanings(self):
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

    def step(self, action):
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
            self.game_params[2] = rng.integers(1, 13)

    def update_vol_and_food_mult(self):
        board_interior_size = (self.game.board_width - 2) * (self.game.board_height - 2)
        outer_wall_size = len(self.game) - board_interior_size
        inner_wall_size = len(self.game.level) - outer_wall_size
        self.volume = board_interior_size - inner_wall_size
        self.food_multiplier = board_interior_size / self.volume

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
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=5, stride=3, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
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

        for env in self.training_env.envs:
            env = env.env
            if env.did_end:
                value = env.end_length
                summary = tf.Summary(value=[tf.Summary.Value(tag='snake_length', simple_value=value)])
                self.locals['writer'].add_summary(summary, self.num_timesteps)
                env.did_end = False
                env.end_length = -1

        return True


def create_dqn_model(save_location: str, game_params: Sequence[Any], iters: int=350000, verbose: int=1) -> Tuple[DQN, gym.Env]:
    from stable_baselines.deepq.policies import MlpPolicy

    env = SnakeEnv2(game_params)

    try:
        model = DQN.load(save_location, env=env)
        print('Loaded existing model from:', save_location)
    except ValueError as e:
        print('Creating new model...')
        model = DQN(MlpPolicy, env, verbose=verbose, buffer_size=100000, exploration_fraction=0.2, exploration_final_eps=0.03)

    model.learn(total_timesteps=iters)
    env.reset()
    model.save(save_location)
    print('Saved model to:', save_location)

    return model, env


def create_ppo2_model(
        save_location: str, game_params: Sequence[Any], iters: int=3500000,
        verbose: int=1) -> Tuple[PPO2, gym.Env]:
    from stable_baselines.common.policies import MlpPolicy

    env = make_vec_env(SnakeEnv2, n_envs=8, env_kwargs={'game_params': game_params})

    try:
        model = PPO2.load(save_location, env=env, n_steps=2500, cliprange=0.2)
        print('Loaded existing model from:', save_location)
    except ValueError as e:
        print('Creating new model...')
        model = PPO2(MlpPolicy, env, verbose=verbose, noptepochs=8, cliprange=0.22)

    if iters > 0:
        model.learn(total_timesteps=iters)
        model.save(save_location)
        print('Saved model to:', save_location)

    return model, SnakeEnv2(game_params)


def create_recurrent_ppo2_model(
        save_location: str, game_params: Sequence[Any], iters: int=3500000,
        verbose: int=1) -> Tuple[PPO2, gym.Env]:
    from stable_baselines.common.policies import MlpLnLstmPolicy

    env = make_vec_env(SnakeEnv3, n_envs=8, env_kwargs={'game_params': game_params})

    try:
        model = PPO2.load(
            save_location, env=env, n_steps=1250, verbose=verbose, noptepochs=8, gamma=0.995,
            tensorboard_log='./recur_ppo2_snake/'
        )
        print('Loaded existing model from:', save_location)
    except ValueError as e:
        print('Creating new model...')
        model = PPO2(
            MlpLnLstmPolicy, env, verbose=verbose, noptepochs=8, gamma=0.98, n_steps=500,
            tensorboard_log='./recur_ppo2_snake/'
        )

    if iters > 0:
        model.learn(total_timesteps=iters, callback=TensorboardCallback())
        model.save(save_location)
        env.reset()
        print('Saved model to:', save_location)

    return model, env

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
            save_location: str, game_params: Sequence[Any], iters: int=3500000,
            verbose: int=1, gamma_start: float=0.99, gamma_stop=Optional[float],
            taper_gamma_steps: int=100) -> Tuple[PPO2, gym.Env]:
    from stable_baselines.common.policies import CnnLnLstmPolicy

    env = make_vec_env(SnakeEnv3, n_envs=8, env_kwargs={'game_params': game_params, 'cnn_policy': True})


    if gamma_stop:
        gammas = np.linspace(gamma_start, gamma_stop, taper_gamma_steps)
    else:
        gammas = [gamma_start]

    gamma_step_iters = iters // taper_gamma_steps
    tb = TensorboardCallback()

    for gamma in gammas:
        try:
            if gamma == gamma_start:
                model = PPO2.load(
                    save_location, env=env, n_steps=500, verbose=verbose, gamma=gamma, cliprange=0.21,
                    lam=0.925, tensorboard_log='./recur_ppo2_snake/'
                )
                print('Loaded existing model from:', save_location)
            else:
                model.gamma = gamma
                model.n_steps += 5
        except ValueError as e:
            print('Creating new model...')
            model = PPO2(
                CnnLnLstmPolicy, env, verbose=verbose, gamma=gamma, n_steps=250,
                tensorboard_log='./recur_ppo2_snake/', policy_kwargs={'cnn_extractor': modified_cnn}
            )

        if iters > 0:
            model.learn(total_timesteps=gamma_step_iters, callback=tb, reset_num_timesteps=False)
            model.save(save_location)
            env.reset()
            print('Saved model to:', save_location)

    return model, env


def create_cnn_lstm_ppo2_model2(
            save_location: str, game_params: Sequence[Any], iters: int=3500000,
            verbose: int=1, gamma_start: float=0.99, gamma_stop=Optional[float],
            taper_gamma_steps: int=100) -> Tuple[PPO2, gym.Env]:
    from stable_baselines.common.policies import CnnLnLstmPolicy

    env = make_vec_env(SnakeEnv5, n_envs=64, env_kwargs={'game_params': game_params, 'cnn_policy': True})

    if gamma_stop:
        gammas = np.linspace(gamma_start, gamma_stop, taper_gamma_steps)
    else:
        gammas = [gamma_start]
    cb = TensorboardCallback()

    gamma_step_iters = iters // taper_gamma_steps
    for gamma in gammas:
        try:
            if gamma == gamma_start:
                model = PPO2.load(
                    save_location, env=env, n_steps=1250, verbose=verbose, gamma=gamma,
                    learning_rate=5e-4, ent_coef=.015, cliprange=0.235,
                    nminibatches=32, tensorboard_log=r'D:\snake_tb_logs', noptepochs=5
                )
                print('Loaded existing model from:', save_location)
            else:
                model.gamma = gamma
        except ValueError as _:
            print('Creating new model...')
            model = PPO2(
                CnnLnLstmPolicy, env, verbose=verbose, gamma=gamma, n_steps=750, learning_rate=7e-4, cliprange=0.25,
                ent_coef=.015, noptepochs=5, tensorboard_log=r'D:\snake_tb_logs', nminibatches=32,
                policy_kwargs={'cnn_extractor': modified_cnn2, 'n_lstm': 128}
            )

        if iters > 0:
            model.learn(total_timesteps=gamma_step_iters, callback=cb, reset_num_timesteps=False)
            model.save(save_location)
            env.reset()
            print('Saved model to:', save_location)

    return model, env


def create_cnn_lstm_ppo2_model3(
            save_location: str, game_params: Sequence[Any], iters: int=7500000,
            verbose: int=1, gamma_start: float=0.99, gamma_stop=Optional[float],
            taper_gamma_steps: int=100) -> Tuple[PPO2, gym.Env]:
    from stable_baselines.common.policies import CnnLnLstmPolicy
    from stable_baselines.common.schedules import LinearSchedule

    env = make_vec_env(SnakeEnv5, n_envs=64, env_kwargs={'game_params': game_params, 'cnn_policy': True})

    if gamma_stop:
        gammas = np.linspace(gamma_start, gamma_stop, taper_gamma_steps)
    else:
        gammas = [gamma_start]
    cb = TensorboardCallback()

    gamma_step_iters = iters // taper_gamma_steps
    learning_schedule = LinearSchedule(iters, 2e-4, initial_p=5e-4)
    for gamma in gammas:
        try:
            if gamma == gamma_start:
                model = PPO2.load(
                    save_location, env=env, n_steps=2000, verbose=verbose, gamma=gamma,
                    learning_rate=learning_schedule.value, ent_coef=.016, cliprange=0.225,
                    nminibatches=32, tensorboard_log=r'D:\snake_tb_logs', noptepochs=5
                )
                print('Loaded existing model from:', save_location)
            else:
                model.gamma = gamma
        except ValueError as _:
            print('Creating new model...')
            model = PPO2(
                CnnLnLstmPolicy, env, verbose=verbose, gamma=gamma, n_steps=1250, learning_rate=learning_schedule.value,
                cliprange=0.25, ent_coef=.015, noptepochs=5, tensorboard_log=r'D:\snake_tb_logs', nminibatches=32,
                policy_kwargs={'cnn_extractor': modified_cnn3, 'n_lstm': 256}
            )

        if iters > 0:
            model.learn(total_timesteps=gamma_step_iters, callback=cb, reset_num_timesteps=False)
            model.save(save_location)
            env.reset()
            print('Saved model to:', save_location)

    return model, env

# def create_cnn_lstm_ppo2_model(save_location: str, game_params: Sequence[Any], iters: int=3500000, verbose: int=1) -> Tuple[PPO2, gym.Env]:
#     from stable_baselines.common.policies import CnnLnLstmPolicy

#     env = make_vec_env(SnakeEnv3, n_envs=8, env_kwargs={'game_params': game_params, 'cnn_policy': True})

#     try:
#         model = PPO2.load(
#             save_location, env=env, n_steps=510, verbose=verbose, gamma=0.9785, cliprange=.208,
#             tensorboard_log='./recur_ppo2_snake/'
#         )
#         print('Loaded existing model from:', save_location)
#     except Exception as e:
#         print('Creating new model...')
#         model = PPO2(
#             CnnLnLstmPolicy, env, verbose=verbose, gamma=0.98, n_steps=500,
#             tensorboard_log='./recur_ppo2_snake/', policy_kwargs={'cnn_extractor': modified_cnn}
#         )

#     if iters > 0:
#         model.learn(total_timesteps=iters, callback=TensorboardCallback())
#         model.save(save_location)
#         env.reset()
#         print('Saved model to:', save_location)

#     return model, env


def create_acer_model(save_location: str, game_params: Sequence[Any], iters=2000000, verbose: int=1) -> Tuple[ACER, gym.Env]:
    from stable_baselines.common.policies import CnnPolicy, MlpPolicy

    env = make_vec_env(SnakeEnv2, n_envs=8, env_kwargs={'game_params': game_params})

    try:
        model = ACER.load(save_location)
        print('Loaded existing model from:', save_location)
    except ValueError as e:
        print('Creating new model...')
        model = ACER(MlpPolicy, env, verbose=verbose, n_steps=1000, buffer_size=50000)

    if iters > 0:
        model.learn(total_timesteps=iters)
        model.save(save_location)
        print('Saved model to:', save_location)

    return model, SnakeEnv2(game_params)


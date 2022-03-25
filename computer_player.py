import numpy as np

from main import handle_input

rng = np.random.default_rng()

# TODO make ComputerPlayer abstract base class


class RandomPlayer(object):
	def run(self):
		handle_input(rng.choice(['up', 'down', 'left', 'right', '']), False)


from collections import namedtuple
from typing import Tuple


Color = namedtuple('Color', ['r', 'g', 'b'])

class Theme(object):
    """docstring for Theme"""
    def __init__(
            self, snake: Color, eyes: Color, background: Color, walls: Color,
            text: Color, grid: Color, food: Color):
        self.snake = snake
        self.eyes = eyes
        self.background = background
        self.walls = walls
        self.text = text
        self.grid = grid
        self.food = food



themes = {
    'default': Theme(
        Color( 77, 140,  63),   # snake
        Color(250, 182,  10),   # eyes
        Color(  0,   0,   0),   # background
        Color(230, 230, 230),   # walls
        Color(197,  27,  64),   # text
        Color( 30,  30,  30),   # grid
        Color( 84,  48,   0)    # food
    ),
    'theme2': Theme(
        Color(101, 116,  58),   # snake
        Color(242, 100,  25),   # eyes
        Color( 36,  36,  46),   # background
        Color(187, 184, 178),   # walls
        Color(175,  59, 110),   # text
        Color( 26,  56,  54),   # grid
        Color(119,  54,  24)    # food
    ),
    'pastel': Theme(
        Color(218, 174, 234),   # snake
        Color(249, 249, 118),   # eyes
        Color(234, 133, 138),   # background
        Color(254, 192, 154),   # walls
        Color(127, 198, 164),   # text
        Color(195, 199, 213),   # grid
        Color(156, 210, 252)    # food
    ),
    'high_contrast': Theme(
        Color(135, 233,  17),   # snake
        Color(225,  24,  69),   # eyes
        Color(  0,   0,   0),   # background
        Color(  0,  87, 233),   # walls
        Color(255,   0, 189),   # text
        Color(137,  49, 239),   # grid
        Color(242, 202,  25)    # food
    ),
    'inverted': Theme(
        Color(178, 115,  92),   # snake
        Color(  5,  73, 245),   # eyes
        Color(255, 255, 255),   # background
        Color( 25,  25,  25),   # walls
        Color( 58, 228, 191),   # text
        Color(225, 225, 225),   # grid
        Color(171, 207, 255)    # food
    ),
    'bold': Theme(
        Color(128, 186,  90),   # snake
        Color(230, 131,  16),   # eyes
        Color(242, 183,   1),   # background
        Color( 57, 105, 172),   # walls
        Color(231,  63, 116),   # text
        Color( 17, 165, 121),   # grid
        Color(127,  60, 141)    # food
    ),
    'vivid': Theme(
        Color( 36, 121, 108),   # snake
        Color(153, 201,  69),   # eyes
        Color(218, 165,  27),   # background
        Color( 93, 105, 177),   # walls
        Color(204,  97, 176),   # text
        Color(229, 134,   6),   # grid
        Color( 82, 188, 163)    # food
    ),
    'antique': Theme(
        Color(104, 133,  92),   # snake
        Color(175, 100,  88),   # eyes
        Color(133,  92, 117),   # background
        Color( 82, 106, 131),   # walls
        Color(217, 175, 107),   # text
        Color( 98,  83, 119),   # grid
        Color(115, 111,  76)    # food
    )
}

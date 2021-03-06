# TODO:
#    - Shader to give snake scales?

import math
import moderngl
import moderngl_window as mglw
import numpy as np

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Mapping as MappingABC
from freetype import Face, FT_LOAD_FORCE_AUTOHINT, FT_LOAD_RENDER
from itertools import chain
from moderngl_window import geometry as geom
from moderngl_window.opengl.vao import VAO
from pyrr import Matrix33, Matrix44, Vector3
from threading import Lock
from typing import Any, Callable, Mapping, List, Optional, Sequence, Tuple

import state

from snake import Game
from themes import Color, Theme
from utils import HorizontalTextAlignment, RectanglePoint


def get_viewport_dimensions() -> Tuple[int, int]:
    viewport = mglw.ctx().fbo.viewport
    return viewport[2] - viewport[0], viewport[3] - viewport[1]


class ShaderProgram(object):
    def __init__(self, program: moderngl.Program, name: str):
        self.program = program
        self.name = name
        self.uniforms = {
            name: program[name] for name in program if isinstance(program[name], moderngl.Uniform)
        }
        print(self.uniforms)

    def __del__(self):
        self.program.release()

    def write_uniform(self, name: str, value: Any) -> None:
        if name in self.uniforms:
            self.uniforms[name].write(value)

    def set_uniform_value(self, name: str, value: Any) -> None:
        if name in self.uniforms:
            self.uniforms[name].value = value


class ProgramRepository(MappingABC):
    def __init__(self, res_dir: str='shaders'):
        self.programs = {}
        self.dir = res_dir

    def __contains__(self, other: str) -> bool:
        return other in self.programs

    def __iter__(self):
        return iter(self.programs)

    def __len__(self) -> int:
        return len(self.programs)

    def __getitem__(self, item: str) -> ShaderProgram:
        if item not in self:
            print(f'Loading program "{item}" with uniforms', end=' ')
            self.programs[item] = ShaderProgram(
                mglw.window().config.load_program(path=f'{self.dir}/{item}.glsl'), item
            )
        return self.programs[item]


class Transform3D(object):
    def __init__(
            self,
            translation: Optional[Vector3]=None,
            rotation: Optional[Vector3]=None,
            scale: Optional[Vector3]=None):
        if translation is None:
            translation = Vector3(dtype='f4')
        if rotation is None:
            rotation = Vector3(dtype='f4')
        if scale is None:
            scale = Vector3([1] * 3, dtype='f4')

        self.translation = translation
        self.rotation = rotation
        self.scale_factor = scale
        self._transformation_matrix = None

    @property
    def translation(self) -> Vector3:
        return self._translation

    @translation.setter
    def translation(self, value: Vector3) -> None:
        self._translation = value
        self.translation_matrix = Matrix44.from_translation(self._translation, dtype='f4')
        self._transformation_matrix = None

    @property
    def rotation(self) -> Vector3:
        return self._rotation

    @rotation.setter
    def rotation(self, value: Vector3) -> None:
        self._rotation = value
        self.rotation_matrix = Matrix44.from_eulers(self._rotation, dtype='f4')
        self._transformation_matrix = None

    @property
    def scale_factor(self) -> Vector3:
        return self._scale

    @scale_factor.setter
    def scale_factor(self, value: Vector3) -> None:
        self._scale = value
        self.scale_matrix = Matrix44.from_scale(self.scale_factor, dtype='f4')
        self._transformation_matrix = None

    @property
    def transformation_matrix(self) -> Matrix44:
        if self._transformation_matrix is None:
            self._transformation_matrix = self.translation_matrix * self.rotation_matrix * self.scale_matrix
        return self._transformation_matrix

    def copy(self) -> 'Transform3D':
        return Transform3D(
            translation=self.translation.copy(),
            rotation=self.rotation.copy(),
            scale=self.scale_factor.copy()
        )

    def translate(self, translation: Vector3) -> None:
        self.translation = self.translation + translation

    def rotate(self, rotation: Vector3) -> None:
        self.rotation = self.rotation + rotation

    def scale(self, scale: Vector3, multiply: bool=True) -> None:
        if multiply:
            self.scale_factor = self.scale_factor * scale
        else:
            self.scale_factor = self.scale_factor + scale


class Renderable(ABC):
    @abstractmethod
    def render(self, *args: Any, **kwargs: Any) -> None:
        pass


class Font(object):
    def __init__(self, font_name: str, size: int):
        # Load font  and check it is monotype
        face = Face(f'resources/fonts/{font_name}.ttf')
        face.set_char_size(size * 64)
        if not face.is_fixed_width:
            raise ValueError('Font is not monotype.')

        # Determine largest glyph size
        width, height, ascender, descender = 0, 0, 0, 0
        for c in range(32, 128):
            face.load_char(chr(c), FT_LOAD_RENDER | FT_LOAD_FORCE_AUTOHINT)
            bitmap = face.glyph.bitmap
            width = max(width, bitmap.width)
            ascender = max(ascender, face.glyph.bitmap_top)
            descender = max(descender, bitmap.rows-face.glyph.bitmap_top)
        height = ascender + descender

        self.char_width = width
        self.char_height = height

        # Generate texture data
        data = np.zeros((height * 6, width * 16), dtype='u1')
        for j in range(5, -1, -1):
            for i in range(16):
                face.load_char(chr(32 + j * 16 + i), FT_LOAD_RENDER | FT_LOAD_FORCE_AUTOHINT)
                bitmap = face.glyph.bitmap
                x = i * width + face.glyph.bitmap_left
                y = j * height + ascender - face.glyph.bitmap_top
                data[y:y + bitmap.rows, x:x + bitmap.width].flat = bitmap.buffer

        data = np.flip(data, axis=0).copy()
        self.texture_height, self.texture_width = data.shape

        ctx = mglw.ctx()
        self.texture = ctx.texture((self.texture_width, self.texture_height), 1, data=data)
        self.texture.build_mipmaps()

    def __del__(self):
        self.texture.release()


class FontBook(MappingABC):
    def __init__(self, res_dir: str='resources/fonts'):
        self.fonts = {}
        self.dir = res_dir

    def __contains__(self, other):
        return other in self.fonts

    def __iter__(self):
        return iter(self.fonts)

    def __len__(self):
        return len(self.fonts)

    def __getitem__(self, name_and_size: Tuple[str, int]) -> Font:
        assert (
            isinstance(name_and_size, tuple) and
            isinstance(name_and_size[0], str) and
            isinstance(name_and_size[1], int)
        )

        if name_and_size not in self:
            self.fonts[name_and_size] = Font(*name_and_size)

        return self.fonts[name_and_size]


class TextRenderer(Renderable):
    instance_number = 0

    def __init__(
            self,
            font: Font,
            text: str,
            color: Color,
            x: float,
            y: float,
            which_point: RectanglePoint=RectanglePoint.TOP_LEFT,
            line_spacing: int=0,
            horizontal_alignment: HorizontalTextAlignment=HorizontalTextAlignment.LEFT):
        self.program = state.shader_program_repo['font']
        self.font = font
        self.which_point = which_point
        self.color = color.r / 255, color.g / 255, color.b / 255
        self.x = x
        self.y = y
        self.line_spacing = line_spacing
        self.horizontal_alignment = horizontal_alignment

        self.position_buffer = None
        self.texture_coords_buffer = None
        self.num_renderable_chars = 0
        self.first_update = True
        self.vao = VAO(f'text_renderer_{TextRenderer.instance_number}')

        self._text = ''
        self.text = text

        TextRenderer.instance_number += 1

    def __del__(self):
        self.vao.release()

    @property
    def text(self) -> str:
        return self._text

    @text.setter
    def text(self, new_text: str) -> None:
        if self._text != new_text:
            self._text = new_text

            renderable_characters = self._text.replace('\t', ' ' * 4)
            num_renderable_chars = len(renderable_characters.replace('\n', ''))
            if num_renderable_chars != self.num_renderable_chars:
                self.num_renderable_chars = num_renderable_chars

                glo = self.program.program.glo
                if glo in self.vao.vaos:
                    del self.vao.vaos[glo]

                self.create_buffers()

            # Set width and height of text area
            self.rows = renderable_characters.split('\n')
            self.num_rows = len(self.rows)
            self.width = max(len(row) for row in self.rows) * self.font.char_width
            self.height = self.num_rows * self.font.char_height + (self.num_rows - 1) * self.line_spacing

            self.update_buffers()

    def create_buffers(self) -> None:
        ctx = mglw.ctx()
        length = self.num_renderable_chars

        if self.position_buffer:
            self.position_buffer.orphan(4 * 2 * length)
        else:
            self.position_buffer = ctx.buffer(reserve=4 * 2 * length)

        if self.texture_coords_buffer:
            self.texture_coords_buffer.orphan(4 * 2 * length)
        else:
            self.texture_coords_buffer = ctx.buffer(reserve=4 * 2 * length)

    def get_position_offset(self) -> Tuple[float, float]:
        which_point = self.which_point

        if which_point == RectanglePoint.TOP_LEFT:
            return self.x, self.y
        if which_point == RectanglePoint.MID_TOP:
            return self.x - self.width / 2, self.y
        if which_point == RectanglePoint.TOP_RIGHT:
            return self.x - self.width, self.y
        if which_point == RectanglePoint.MID_LEFT:
            return self.x, self.y + self.height / 2
        if which_point == RectanglePoint.CENTER:
            return self.x - self.width / 2, self.y + self.height / 2
        if which_point == RectanglePoint.MID_RIGHT:
            return self.x - self.width, self.y + self.height / 2
        if which_point == RectanglePoint.BOTTOM_LEFT:
            return self.x, self.y + self.height
        if which_point == RectanglePoint.MID_BOTTOM:
            return self.x - self.width / 2, self.y + self.height
        if which_point == RectanglePoint.BOTTOM_RIGHT:
            return self.x + self.width, self.y + self.height

        raise ValueError(''.join([
            f'Invalid TextRenderer position offset point: {which_point}.\n',
            'Valid options: topleft, midtop, topright, midleft, center, midright,',
            ' bottomleft, midbottom, bottomright'
        ]))

    def get_line_start_offsets(self) -> List[float]:
        char_width = self.font.char_width
        width_in_chars = self.width / char_width

        if self.horizontal_alignment == HorizontalTextAlignment.CENTERED:
            return [(width_in_chars - len(row)) / 2 * char_width for row in self.rows]
        if self.horizontal_alignment == HorizontalTextAlignment.RIGHT:
            return [(width_in_chars - len(row)) * char_width for row in self.rows]
        return [0.0] * self.num_rows

    def update_buffers(self) -> None:
        offsets, texture_coords = self.get_offsets_and_texture_coords()

        self.position_buffer.write(offsets)
        self.texture_coords_buffer.write(texture_coords)

        if self.first_update:
            self.first_update = False
            self.vao.buffer(self.position_buffer, '2f/v', ['in_position'])
            self.vao.buffer(self.texture_coords_buffer, '2f/v', ['in_texcoord_0'])

        self.vao.vertex_count = self.num_renderable_chars
        self.vao.get_buffer_by_name('in_position').vertices = self.num_renderable_chars
        self.vao.get_buffer_by_name('in_texcoord_0').vertices = self.num_renderable_chars

    def get_offsets_and_texture_coords(self) -> Tuple[np.ndarray, np.ndarray]:
        font = self.font
        width = font.char_width
        height = font.char_height
        dx = width / font.texture_width
        dy = height / font.texture_height

        length = 2 * self.num_renderable_chars
        start_x, start_y = self.get_position_offset()
        line_start_offsets = self.get_line_start_offsets()
        offsets = np.zeros(length, dtype='f4')
        texture_coords = np.zeros(length, dtype='f4')
        x_offset = line_start_offsets[0]
        y_offset = 0
        line_idx = 0
        idx = 0
        for char in self.text:
            i = ord(char)
            x, y = i % 16, 9 - i // 16 - 2

            if char == '\n':
                line_idx += 1
                x_offset = line_start_offsets[line_idx]
                y_offset += font.char_height + self.line_spacing
            elif char == '\t':
                x_offset += 4 * font.char_width
            elif 32 <= i < 128:
                offsets[idx] = start_x + x_offset
                offsets[idx + 1] = start_y - y_offset
                texture_coords[idx] = x * dx
                texture_coords[idx + 1] = y * dy
                x_offset += font.char_width
                idx += 2

        return offsets, texture_coords

    def render(self, proj: Matrix44) -> None:
        self.program.write_uniform('projection', proj)
        self.program.set_uniform_value('color', self.color)

        font = self.font
        self.program.set_uniform_value('charWidth', font.char_width)
        self.program.set_uniform_value('charHeight', font.char_height)
        self.program.set_uniform_value('textureWidth', font.texture_width)
        self.program.set_uniform_value('textureHeight', font.texture_height)

        self.font.texture.use(location=0)

        self.vao.render(self.program.program, mode=moderngl.POINTS)


class InstancedObject(Renderable):
    def __init__(
            self,
            instance_count: int,
            vao_generator: Callable[..., VAO],
            program_name: str,
            color: Optional[Color],
            transforms: List[Transform3D],
            texture: Optional[moderngl.Texture]=None,
            vao_generator_args: Optional[Sequence[any]]=None,
            vao_generator_kwargs: Optional[Mapping[str, any]]=None):
        self.program = state.shader_program_repo[program_name]
        self.vao_generator = vao_generator
        self.vao_generator_args = [] if vao_generator_args is None else vao_generator_args
        self.vao_generator_kwargs = {} if vao_generator_kwargs is None else vao_generator_kwargs
        self.instance_count = instance_count

        self.color = None
        if color:
            self.color = np.array([
                color.r / 255, color.g / 255, color.b / 255
            ], dtype='f4')

        self.texture = texture

        self.transforms = transforms

        self.vao = None
        self.model_buffer = None
        self.mvp_buffer = None
        self.normal_mat_buffer = None
        self.create_buffers()
        self.create_vao()

    def __del__(self):
        # Note: Since moderngl_window.opengl.vao.VAO defaults to releasing buffers, we don't need
        # to do so manually.
        self.vao.release()

    def create_vao(self) -> None:
        self.vao = self.vao_generator(*self.vao_generator_args, **self.vao_generator_kwargs)

        # This fixes a performance issue likely caused by a bug in the macOS OpenGL driver
        if not self.vao._index_buffer:
            self.vao.index_buffer(np.arange(self.vao.vertex_count, dtype=np.uint32))

        self.vao.buffer(self.mvp_buffer, '16f/i', ['mvp'])
        self.vao.buffer(self.normal_mat_buffer, '9f/i', ['normal_mat'])
        self.vao.buffer(self.model_buffer, '16f/i', ['model'])

    def create_buffers(self) -> None:
        ctx = mglw.ctx()
        length = self.instance_count

        if self.mvp_buffer:
            self.mvp_buffer.orphan(4 * 16 * length)
        else:
            self.mvp_buffer = ctx.buffer(reserve=4 * 16 * length)

        if self.model_buffer:
            self.model_buffer.orphan(4 * 16 * length)
        else:
            self.model_buffer = ctx.buffer(reserve=4 * 16 * length)

        if self.normal_mat_buffer:
            self.normal_mat_buffer.orphan(4 * 9 * length)
        else:
            self.normal_mat_buffer = ctx.buffer(reserve=4 * 9 * length)

    def get_instance_data(self, view_proj: Matrix44) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        transforms = [transform.transformation_matrix for transform in self.transforms]
        models = np.array(transforms, dtype='f4')
        mvps = np.array([view_proj * model for model in transforms], dtype='f4')
        normal_mats = np.array([Matrix33(model.inverse.T) for model in transforms], dtype='f4')

        return models, mvps, normal_mats

    def add_instance(self, transform: Transform3D) -> None:
        self.instance_count += 1
        self.create_buffers()
        self.transforms.append(transform)

    def add_instances(self, transforms: Sequence[Transform3D]) -> None:
        if len(transforms):
            self.instance_count += len(transforms)
            self.create_buffers()
            self.transforms.extend(transforms)

    def update(self, view_proj: Matrix44) -> None:
        models, mvps, normal_mats = self.get_instance_data(view_proj)
        self.model_buffer.write(models)
        self.mvp_buffer.write(mvps)
        self.normal_mat_buffer.write(normal_mats)

    def render(
            self,
            value_uniforms: Optional[Mapping[str, Any]]=None,
            write_uniforms: Optional[Mapping[str, Any]]=None,
            override_program: Optional[ShaderProgram]=None) -> None:
        program = override_program if override_program else self.program

        if self.color is not None:
            program.write_uniform('color', self.color)

        if value_uniforms:
            for name, value in value_uniforms.items():
                program.set_uniform_value(name, value)
        if write_uniforms:
            for name, value in write_uniforms.items():
                program.write_uniform(name, value)

        self.vao.render(program.program, instances=self.instance_count)


class Light(Renderable):
    def __init__(
            self,
            pos: Vector3,
            color: Color,
            board_width: int,
            board_height: int,
            radius: float=5.0,
            brightness=5.0,
            rendered_color: Color=Color(249, 215, 28)):
        self.pos = pos
        self.color = color.r / 255, color.g / 255, color.b / 255
        self.brightness = brightness

        # Set up view and projection matrices
        left = -1.5 * board_width - abs(pos[0])
        right = 1.5 * board_width + abs(pos[0])
        top = -1.5 * board_height - abs(pos[1])
        bottom = 1.5 * board_height + abs(pos[1])
        near = abs(self.pos[2]) / 6.0
        far = (self.pos[0]**2 + self.pos[1]**2 + self.pos[2]**2)**0.5 + (board_width**2 + board_height**2)**0.5
        self.proj = Matrix44.orthogonal_projection(
            left, right, top, bottom, near, far, dtype='f4'
        )
        self.view = Matrix44.look_at(
            pos, (0, 0, 0), (0, 1, 0), dtype='f4'
        )
        self.view_proj = self.proj * self.view

        # Buffer the light data used by various shaders
        ctx = mglw.ctx()
        self.pos_buffer = ctx.buffer(pos)
        self.color_buffer = ctx.buffer(Vector3(self.color, dtype='f4'))
        self.view_proj_buffer = ctx.buffer(self.view_proj)

        # Create a sphere to render as the light
        transforms = Transform3D(translation=pos)
        self.sphere = InstancedObject(
            1, geom.sphere, 'lighting', rendered_color, [transforms],
            vao_generator_kwargs={'radius': radius, 'sectors': 64, 'rings': 64}
        )

    def __del__(self):
        self.pos_buffer.release()
        self.color_buffer.release()
        self.view_proj_buffer.release()

    def update(self, view_proj: Matrix44, force_update: bool=False) -> None:
        if force_update:
            self.sphere.update(view_proj)

    def render(self) -> None:
        self.sphere.render(value_uniforms={
            'isLightSource': True, 'brightnessMult': self.brightness
        })


class SnakeRenderer(Renderable):
    def __init__(self, game: Game, theme: Theme):
        self.snake = game.snake
        self.head_location = self.snake.head.x, self.snake.head.y

        # Create snake
        transforms = [
            Transform3D(translation=Scene.grid_position(
                node.x, node.y, game.board_width, game.board_height)
            ) for node in self.snake
        ]
        self.snake_length = len(transforms)

        self.body = InstancedObject(
            len(transforms), geom.cube, 'lighting', theme.snake, transforms,
            vao_generator_kwargs={'size': (2.0, 2.0, 2.0)}
        )

        self.eyes = InstancedObject(
            2, geom.sphere, 'lighting', theme.eyes, self.get_eye_transforms(),
            vao_generator_kwargs={'radius': 0.25}
        )

    def get_eye_transforms(self) -> List[Transform3D]:
        direction = self.snake.direction.name

        left_transform = self.body.transforms[0].copy()
        right_transform = left_transform.copy()

        if direction == 'left':
            left_transform.translate(Vector3([-0.3, 0.85, -1.0], dtype='f4'))
            right_transform.translate(Vector3([-0.3, 0.85, 1.0], dtype='f4'))
        elif direction == 'right':
            left_transform.translate(Vector3([0.3, 0.85, -1.0], dtype='f4'))
            right_transform.translate(Vector3([0.3, 0.85, 1.0], dtype='f4'))
        elif direction == 'up':
            left_transform.translate(Vector3([-1.0, 0.85, -0.3], dtype='f4'))
            right_transform.translate(Vector3([1.0, 0.85, -0.3], dtype='f4'))
        elif direction == 'down':
            left_transform.translate(Vector3([-1.0, 0.85, 0.3], dtype='f4'))
            right_transform.translate(Vector3([1.0, 0.85, 0.3], dtype='f4'))

        return [left_transform, right_transform]

    def update(
            self,
            view_proj: Matrix44,
            board_width: int,
            board_height: int,
            force_update: bool=False) -> None:
        needs_update = force_update

        # If snake grew, create new buffers since there are more instances
        current_length = len(self.snake.nodes)
        if current_length > self.snake_length:
            self.snake_length = current_length
            needs_update = True

        # If snake moved, update MVP buffer
        head_location = self.snake.head.pos
        if head_location != self.head_location:
            self.head_location = head_location
            needs_update = True

        if needs_update:
            # Update snake positions and add new nodes if needed
            new_transforms = []
            for i, node in enumerate(self.snake):
                position = Scene.grid_position(node.x, node.y, board_width, board_height)
                if i < self.body.instance_count:
                    self.body.transforms[i].translation = position
                else:
                    new_transforms.append(Transform3D(translation=position))
            self.body.add_instances(new_transforms)
            self.body.update(view_proj)

            self.eyes.transforms = self.get_eye_transforms()
            self.eyes.update(view_proj)

    def render(self, override_program: Optional[ShaderProgram]=None) -> None:
        self.body.render(
            value_uniforms={'specularStrength': 0.1, 'isLightSource': False},
            override_program=override_program
        )
        self.eyes.render(
            value_uniforms={'specularStrength': 1.0, 'isLightSource': False},
            override_program=override_program
        )


class WallRenderer(Renderable):
    def __init__(self, game: Game, theme: Theme):
        self.walls = game.level.wall_nodes
        self.wall_locations = [wall.pos for wall in self.walls]

        transforms = [
            Transform3D(
                translation=Scene.grid_position(
                    x, y, game.board_width, game.board_height
                ),
                scale=Vector3([1.0, 1.5, 1.0], dtype='f4')
            ) for x, y in self.wall_locations
        ]

        self.instanced_cube = InstancedObject(
            len(self.walls), geom.cube, 'lighting', theme.walls, transforms,
            vao_generator_kwargs={'size': (2.0, 2.0, 2.0)}
        )

    def update(
            self,
            view_proj: Matrix44,
            board_width: int,
            board_height: int,
            force_update: bool=False) -> None:
        needs_update = force_update

        # If a wall moved, update the transform
        for i, wall in enumerate(self.walls):
            if wall.pos != self.wall_locations[i]:
                needs_update = True
                self.wall_locations[i] = wall.pos
                self.instanced_cube.transforms[i].translation = Scene.grid_position(
                    wall.x, wall.y, board_width, board_height
                )

        if needs_update:
            self.instanced_cube.update(view_proj)

    def render(self, override_program: Optional[ShaderProgram]=None) -> None:
        self.instanced_cube.render(
            value_uniforms={'specularStrength': 0.6, 'isLightSource': False},
            override_program=override_program
        )


class FoodRenderer(Renderable):
    def __init__(self, game: Game, theme: Theme):
        self.game = game
        self.food_pos = game.food.pos

        self.sphere = InstancedObject(
            1, geom.sphere, 'lighting', theme.food, [Transform3D()],
            vao_generator_kwargs={'radius': 0.9}
        )

    def update(
            self,
            view_proj: Matrix44,
            board_width: int,
            board_height: int,
            force_update: bool=False) -> None:
        food_pos = self.game.food.pos
        needs_update = self.food_pos != food_pos or force_update

        if needs_update:
            # Update food position
            self.food_pos = food_pos
            self.sphere.transforms[0].translation = Scene.grid_position(
                *food_pos, board_width, board_height
            )
            self.sphere.update(view_proj)

    def render(self, override_program: Optional[ShaderProgram]=None) -> None:
        self.sphere.render(
            value_uniforms={'specularStrength': 0.6, 'isLightSource': False},
            override_program=override_program
        )


class BoardRenderer(Renderable):
    def __init__(self, board_width: int, board_height: int, theme: Theme):
        # Create grid
        horizontal_scale = Vector3([board_width * 2, 0.1, 1], dtype='f4')
        vertical_scale = Vector3([0.1, board_height * 2, 1], dtype='f4')
        rotation = Vector3([math.pi / 2, 0, 0], dtype='f4')
        transforms = [
            *(
                Transform3D(
                    translation=Vector3([x, 0, 0], dtype='f4'),
                    scale=vertical_scale,
                    rotation=rotation
                ) for x in range(-board_width + 2, board_width, 2)
            ),
            *(
                Transform3D(
                    translation=Vector3([0, 0, y], dtype='f4'),
                    scale=horizontal_scale,
                    rotation=rotation
                ) for y in range(-board_height + 2, board_height, 2)
            )
        ]
        self.grid = InstancedObject(
            board_width + board_height - 2, geom.quad_2d, 'lighting', theme.grid, transforms
        )

        # Create floor
        transforms = Transform3D(
            translation=Vector3([0, -0.05, 0], dtype='f4'),
            scale=Vector3([board_width * 2, board_height * 2, 1], dtype='f4'),
            rotation=rotation
        )
        self.floor = InstancedObject(1, geom.quad_2d, 'lighting', theme.background, [transforms])

    def update(self, view_proj: Matrix44, force_update: bool=False) -> None:
        # Set the view projection matrix in the floor and the grid, which only needs to happen once
        # per time the projection changes
        if force_update:
            self.floor.update(view_proj)
            self.grid.update(view_proj)

    def render(self, override_program: Optional[ShaderProgram]=None) -> None:
        self.floor.render(
            value_uniforms={'specularStrength': 0.6, 'isLightSource': False},
            override_program=override_program
        )
        self.grid.render(
            value_uniforms={'specularStrength': 0.0, 'isLightSource': False},
            override_program=override_program
        )


class TaperedSnakeRenderer(Renderable):
    def __init__(self, game: Game, theme: Theme):
        self.snake = game.snake
        self.head_location = self.snake.head.pos
        self.snake_length = len(self.snake)
        self.color = theme.snake.r / 255, theme.snake.g / 255, theme.snake.b / 255

        # Because VAO instances are cached internally in mglw.vao, we need to force# it to
        # regenerate the instance when the buffers grow
        self.clear_vao_instance = defaultdict(lambda: True)

        self.start_radius = 0.85

        head_translation = Scene.grid_position(
            *self.head_location, game.board_width, game.board_height
        )
        self.transform = Transform3D(
            translation=head_translation
        )

        transforms = [
            Transform3D() for _ in range(self.snake_length - 1)
        ]
        self.curve_spheres = InstancedObject(
            self.snake_length - 1, geom.sphere, 'lighting', theme.snake, transforms,
            vao_generator_kwargs={'radius': 0.95}
        )

        self.first_update = True
        self.mvp_buffer = None
        self.model_buffer = None
        self.normal_mat_buffer = None
        self.body_buffer = None
        self.tail = None
        self.create_buffers()
        self.create_tail(game.board_width, game.board_height)
        self.tail.buffer(self.mvp_buffer, '16f/r', ['mvp'])
        self.tail.buffer(self.model_buffer, '16f/r', ['model'])
        self.tail.buffer(self.normal_mat_buffer, '9f/r', ['normal_mat'])

        self.body_program = state.shader_program_repo['snake_procedural']
        self.body_shadow_map_prog = state.shader_program_repo['snake_procedural_shadow_map_depth']

        self.head_transform = Transform3D(
            translation=head_translation,
            scale=Vector3([1, self.start_radius, self.start_radius], dtype='f4')
        )
        self.head = InstancedObject(
            1, geom.sphere, 'lighting', theme.snake, [self.head_transform],
            vao_generator_kwargs={'radius': 1.0}
        )

        self.eyes = InstancedObject(
            2, geom.sphere, 'lighting', theme.eyes, self.get_eye_transforms(),
            vao_generator_kwargs={'radius': 0.25}
        )

    def get_eye_transforms(self) -> List[Transform3D]:
        direction = self.snake.direction.name

        left_transform = self.transform.copy()
        right_transform = left_transform.copy()

        if direction == 'left':
            left_transform.translate(Vector3([-0.3, 0.5, -self.start_radius * 0.65], dtype='f4'))
            right_transform.translate(Vector3([-0.3, 0.5, self.start_radius * 0.65], dtype='f4'))
        elif direction == 'right':
            left_transform.translate(Vector3([0.3, 0.5, -self.start_radius * 0.65], dtype='f4'))
            right_transform.translate(Vector3([0.3, 0.5, self.start_radius * 0.65], dtype='f4'))
        elif direction == 'up':
            left_transform.translate(Vector3([-self.start_radius * 0.65, 0.5, -0.3], dtype='f4'))
            right_transform.translate(Vector3([self.start_radius * 0.65, 0.5, -0.3], dtype='f4'))
        elif direction == 'down':
            left_transform.translate(Vector3([-self.start_radius * 0.65, 0.5, 0.3], dtype='f4'))
            right_transform.translate(Vector3([self.start_radius * 0.65, 0.5, 0.3], dtype='f4'))

        return [left_transform, right_transform]

    def create_tail(self, board_width: int, board_height: int) -> None:
        snake_length = self.snake_length
        start_radius = self.start_radius
        min_radius = 0.2
        decrease = (start_radius - min_radius) / (snake_length - 1)

        positions = [
            Scene.grid_position(*node.pos, board_width, board_height)
            for node in self.snake
        ]

        points = np.fromiter(
            chain(*positions), dtype='f4'
        ).reshape(snake_length, 3)
        points -= points[0]
        radii = start_radius - np.arange(snake_length, dtype='f4') * decrease

        for i, transform in enumerate(self.curve_spheres.transforms):
            transform.scale_factor = Vector3([radii[i + 1]] * 3, dtype='f4')
            transform.translation = Vector3(positions[i + 1])

        body = np.concatenate([points, radii[:, np.newaxis]], axis=1).flatten()
        self.body_buffer.write(body)

        if self.first_update:
            self.tail = VAO('tail')
            self.tail.buffer(self.body_buffer, '4f', ['in_position'])
            self.first_update = False
        self.tail.vertex_count = body.shape[0]
        self.tail.get_buffer_by_name('in_position').vertices = body.shape[0]
        self.clear_vao_instance.clear()

    def update_head_scale(self) -> None:
        direction = self.snake.direction.name

        if direction == 'left' or direction == 'right':
            self.head_transform.scale_factor = Vector3([1, self.start_radius, self.start_radius], dtype='f4')
        else:
            self.head_transform.scale_factor = Vector3([self.start_radius, 1, self.start_radius], dtype='f4')

    def create_buffers(self) -> None:
        ctx = mglw.ctx()
        length = self.snake_length

        if self.mvp_buffer:
            self.mvp_buffer.orphan(4 * 16 * length)
        else:
            self.mvp_buffer = ctx.buffer(reserve=4 * 16 * length)

        if self.model_buffer:
            self.model_buffer.orphan(4 * 16 * length)
        else:
            self.model_buffer = ctx.buffer(reserve=4 * 16 * length)

        if self.normal_mat_buffer:
            self.normal_mat_buffer.orphan(4 * 9 * length)
        else:
            self.normal_mat_buffer = ctx.buffer(reserve=4 * 9 * length)

        if self.body_buffer:
            self.body_buffer.orphan(4 * 4 * length)
        else:
            self.body_buffer = ctx.buffer(reserve=4 * 4 * length)

    def update(
            self,
            view_proj: Matrix44,
            board_width: int,
            board_height: int,
            force_update: bool=False) -> None:
        needs_update = force_update

        # If snake grew, create new buffers since there are more instances
        current_length = len(self.snake.nodes)
        if current_length > self.snake_length:
            self.snake_length = current_length
            needs_update = True
            self.create_buffers()
            self.curve_spheres.add_instance(Transform3D())

        # If snake moved, update MVP buffer
        head_location = self.snake.head.pos
        if head_location != self.head_location:
            self.head_location = head_location
            needs_update = True

        if needs_update:
            new_translation = Scene.grid_position(*head_location, board_width, board_height)
            self.transform.translation = new_translation
            self.head_transform.translation = new_translation

            model = self.transform.transformation_matrix
            mvp = view_proj * model
            normal_mat = Matrix33(model.inverse.T, dtype='f4').copy()

            self.mvp_buffer.write(mvp)
            self.model_buffer.write(model)
            self.normal_mat_buffer.write(normal_mat)

            self.create_tail(board_width, board_height)
            self.curve_spheres.update(view_proj)

            self.update_head_scale()
            self.head.update(view_proj)

            self.eyes.transforms = self.get_eye_transforms()
            self.eyes.update(view_proj)

    def render(self, override_program: Optional[ShaderProgram]=None) -> None:
        body_program = override_program if override_program else self.body_program
        if body_program.name == 'shadow_map_depth':
            body_program = self.body_shadow_map_prog

        # If the snake grew, we need to clear the vao instance from self.tail because it caches the
        # buffers. If we don't clear it, the snake tail doesn't render longer than the start length
        if self.clear_vao_instance[body_program]:
            glo = body_program.program.glo
            if glo in self.tail.vaos:
                del self.tail.vaos[glo]
            self.clear_vao_instance[body_program] = False

        body_program.set_uniform_value('specularStrength', 0.1)
        body_program.set_uniform_value('color', self.color)

        self.tail.render(body_program.program, mode=moderngl.LINE_STRIP)

        self.curve_spheres.render(
            override_program=override_program, value_uniforms={
                'specularStrength': 0.1, 'isLightSource': False
            }
        )

        self.head.render(
            override_program=override_program, value_uniforms={
                'specularStrength': 0.1, 'isLightSource': False
            }
        )
        self.eyes.render(
            override_program=override_program, value_uniforms={
                'specularStrength': 1.0, 'isLightSource': False
            }
        )


class ShadowMap(Renderable):
    def __init__(self, width: int=4096, height: int=4096):
        ctx = mglw.ctx()
        self.width, self.height = width, height

        # Create the shadow map depth texture
        self.shadow_map = ctx.depth_texture((self.width, self.height))
        self.shadow_map.compare_func = ''
        self.shadow_map.repeat_x = False
        self.shadow_map.repeat_y = False
        self.shadow_map.filter = moderngl.LINEAR, moderngl.LINEAR

        # Create the shadow map framebuffer
        self.framebuffer = ctx.framebuffer(depth_attachment=self.shadow_map)

        self.prog = state.shader_program_repo['shadow_map_depth']

    def __del__(self):
        self.shadow_map.release()
        self.framebuffer.release()

    def render(self, renderables: Sequence[Renderable]) -> None:
        self.framebuffer.clear()
        self.framebuffer.use()

        for renderable in renderables:
            renderable.render(override_program=self.prog)


class HDRBloomRenderer(Renderable):
    def __init__(self):
        ctx = mglw.ctx()

        viewport_dimensions = get_viewport_dimensions()
        ping_pong_dimensions = viewport_dimensions[0] // 6, viewport_dimensions[1] // 6

        # Set up textures
        self.scene_texture = ctx.texture(viewport_dimensions, 4, dtype='f2')
        self.brightness_texture = ctx.texture(viewport_dimensions, 4, dtype='f2')
        self.hdr_depth_texture = ctx.depth_texture(viewport_dimensions)
        hdr_textures = [self.scene_texture, self.brightness_texture]
        self.ping_pong_textures = [
            ctx.texture(ping_pong_dimensions, 4, dtype='f2'),
            ctx.texture(ping_pong_dimensions, 4, dtype='f2')
        ]

        # Configure textures
        for texture in chain(self.ping_pong_textures, hdr_textures):
            texture.repeat_x = False
            texture.repeat_y = False
            texture.filter = moderngl.LINEAR, moderngl.LINEAR

        # Set up framebuffers
        self.hdr_framebuffer = ctx.framebuffer(
            depth_attachment=self.hdr_depth_texture, color_attachments=hdr_textures
        )
        self.ping_pong_framebuffers = [
            ctx.framebuffer(color_attachments=[self.ping_pong_textures[0]]),
            ctx.framebuffer(color_attachments=[self.ping_pong_textures[1]])
        ]

        # Set up programs
        self.blur_program = state.shader_program_repo['blur']
        prog_name = 'bloom_final_fxaa'
        prog = state.shader_program_repo[prog_name].program
        prog['scene'].value = 0
        prog['bloomBlur'].value = 1

        # Create quad which acts as a render target for the various textures
        self.quad = InstancedObject(
            1, geom.quad_2d, prog_name, None, [Transform3D()], vao_generator_kwargs={'size': (2.0, 2.0)}
        )

    def __del__(self):
        textures = [*self.ping_pong_textures, self.hdr_depth_texture, self.brightness_texture, self.scene_texture]
        framebuffers = [*self.ping_pong_framebuffers, self.hdr_framebuffer]
        for obj in chain(textures, framebuffers):
            obj.release()

    def blur(self) -> bool:
        horizontal = True
        first_iteration = True
        amount = 12

        # Repeatedly apply gaussian blur to the brightness texture, alternating between vertical
        # and horizontal
        for i in range(amount):
            self.ping_pong_framebuffers[horizontal].use()

            if first_iteration:
                self.brightness_texture.use()
                first_iteration = False
            else:
                self.ping_pong_textures[not horizontal].use()

            self.quad.render(override_program=self.blur_program, value_uniforms={'horizontal': horizontal})

            horizontal = not horizontal

        return not horizontal

    def render(self, renderables: Sequence[Renderable]) -> None:
        self.hdr_framebuffer.clear()
        self.hdr_framebuffer.use()

        # Render scene to HDR framebuffer with two render targets???scene and brightness
        for renderable in renderables:
            renderable.render()

        # Blur brightness texture
        horizontal = self.blur()

        # Render scene and bloom textures to the screen
        mglw.ctx().screen.use()
        self.scene_texture.use(location=0)
        self.ping_pong_textures[horizontal].use(location=1)
        self.quad.render()


class Scene(Renderable):
    def __init__(self, game: Game, theme: Theme, tapered_snake: bool, aspect_ratio: float=16 / 9):
        ctx = mglw.ctx()

        # Create a lock to avoid race conditions when the screen is resized in case the OpenGL
        # windowing library listens for key presses on a different thread than it renders from.
        # When this was written, pyglet didn't do that, but I might as well do this in case that
        # changes or if I switch to a different windowing library.
        self.resize_lock = Lock()

        self.theme = theme
        self.game = game
        self.board_width = game.board_width
        self.board_height = game.board_height

        max_dim = max(self.board_width / aspect_ratio, self.board_height)

        # Create camera and associated view and projection matrices
        self.camera_pos = 0.0, max_dim * 2.25, max_dim * 2.1
        self.camera_pos_buffer = ctx.buffer(Vector3(self.camera_pos, dtype='f4'))

        self.view = None
        self.proj = None
        self.view_proj = None
        self.update_view_proj(aspect_ratio)
        self.force_update = True

        # Create shadow map
        self.shadow_map = ShadowMap()

        # Create light
        light_pos = -max_dim * 0.9, max_dim * 1.1, -max_dim * 0.65
        self.light = Light(
            Vector3(light_pos, dtype='f4'), Color(255, 255, 255), self.board_width, self.board_height,
            radius=max_dim / 4
        )

        # Create board
        self.board_renderer = BoardRenderer(self.board_width, self.board_height, theme)

        # Create walls
        self.wall_renderer = WallRenderer(game, theme)

        # Create snake
        if tapered_snake:
            self.snake_renderer = TaperedSnakeRenderer(game, theme)
        else:
            self.snake_renderer = SnakeRenderer(game, theme)

        # Create food
        self.food_renderer = FoodRenderer(game, theme)

        prog = state.shader_program_repo['lighting'].program
        prog['LightPos'].binding = 1
        prog['LightColor'].binding = 2
        prog['ViewPos'].binding = 3
        prog['LightSpaceMatrix'].binding = 4

        prog = state.shader_program_repo['snake_procedural'].program
        prog['LightPos'].binding = 1
        prog['LightColor'].binding = 2
        prog['ViewPos'].binding = 3
        prog['LightSpaceMatrix'].binding = 4

        prog = state.shader_program_repo['shadow_map_depth'].program
        prog['LightSpaceMatrix'].binding = 4

        prog = state.shader_program_repo['snake_procedural_shadow_map_depth'].program
        prog['LightSpaceMatrix'].binding = 4

        self.scope = ctx.scope(
            None, moderngl.DEPTH_TEST, uniform_buffers=[
                (self.light.pos_buffer, 1),
                (self.light.color_buffer, 2),
                (self.camera_pos_buffer, 3),
                (self.light.view_proj_buffer, 4)
            ]
        )

        # Create HDR Bloom renderer
        self.hdr_bloom_renderer = HDRBloomRenderer()

        self.shadow_map_pass_renderables = [
            self.board_renderer, self.snake_renderer, self.wall_renderer, self.food_renderer
        ]

        self.hdr_bloom_pass_renderables = [*self.shadow_map_pass_renderables, self.light]

    def __del__(self):
        self.camera_pos_buffer.release()

    def update_view_proj(self, aspect_ratio: float) -> None:
        self.proj = Matrix44.perspective_projection(
            45.0, aspect_ratio, 0.1, self.camera_pos[2] * 2, dtype='f4'
        )
        self.view = Matrix44.look_at(
            self.camera_pos, (0, 0, 0), (0, 1, 0), dtype='f4'
        )
        self.view_proj = self.proj * self.view

    def resize(self, aspect_ratio: float) -> None:
        with self.resize_lock:
            self.update_view_proj(aspect_ratio)
            self.hdr_bloom_renderer = HDRBloomRenderer()
            self.force_update = True

    @staticmethod
    def grid_position(x: int, y: int, width: int, height: int) -> Vector3:
        return Vector3(
            [2 * x + 1 - width, 1, 2 * y + 1 - height], dtype='f4'
        )

    def update(self) -> None:
        with self.resize_lock:
            self.snake_renderer.update(
                self.view_proj, self.board_width, self.board_height, force_update=self.force_update
            )
            self.wall_renderer.update(
                self.view_proj, self.board_width, self.board_height, force_update=self.force_update
            )
            self.food_renderer.update(
                self.view_proj, self.board_width, self.board_height, force_update=self.force_update
            )
            self.board_renderer.update(self.view_proj, force_update=self.force_update)
            self.light.update(self.view_proj, force_update=self.force_update)
            self.force_update = False

    def render(self, time: float, frame_time: float) -> None:
        self.update()

        with self.scope:
            # Render shadow map
            self.shadow_map.render(self.shadow_map_pass_renderables)

            # Render scene
            self.shadow_map.shadow_map.use(location=0)
            self.hdr_bloom_renderer.render(self.hdr_bloom_pass_renderables)

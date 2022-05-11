# TODO:
#    - Shader to give snake scales?
#    - Procedurally generated tapered snake body
#    - When adding instances, grow buffers rather than just deleting and recreating;
#      see moderngl example growing_buffers.py
#    - destroy resources
#    - Sounds

import math
import moderngl
import moderngl_window as mglw
import numpy as np

from abc import ABC, abstractmethod
from collections.abc import Mapping as MappingABC
from freetype import Face, FT_LOAD_FORCE_AUTOHINT, FT_LOAD_RENDER
from itertools import chain
from moderngl_window import geometry as geom
from moderngl_window.opengl.vao import VAO
from pyrr import Matrix33, Matrix44, Vector3
from typing import Any, Callable, Mapping, List, Optional, Sequence, Tuple

import state

from snake import Game
from themes import Color, Theme


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

    def write_uniform(self, name: str, value: Any) -> None:
        if name in self.uniforms:
            self.uniforms[name].write(value)
        # else:
            # print(f'Trying to set uniform "{name}" which is not present in program {self.name}.')

    def set_uniform_value(self, name: str, value: Any) -> None:
        if name in self.uniforms:
            self.uniforms[name].value = value
        # else:
            # print(f'Trying to set uniform "{name}" which is not present in program {self.name}.')


class ProgramRepository(MappingABC):  # FIXME: safer path handling
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
        return self.translation_matrix * self.rotation_matrix * self.scale_matrix

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
        # self.texture.write(Z)
        self.texture.build_mipmaps()


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


# TODO:
#    - Horizontal alignment
#    - Controllable line spacing
class TextRenderer(Renderable):
    def __init__(self, font: Font, text: str, color: Color, x: float, y: float, which_point: str='topleft'):
        self.program = state.shader_program_repo['font']
        self.font = font
        self.which_point = which_point
        self.color = color.r / 255, color.g / 255, color.b / 255
        self.x = x
        self.y = y
        self.vaos = []
        self._text = self._previous_text = ''
        self.text = text

    @property
    def text(self) -> str:
        return self._text

    @text.setter
    def text(self, new_text: str) -> None:
        self._previous_text = self._text
        self._text = new_text
        rows = self.text.replace('\t', ' ' * 4).split('\n')
        self.width = max(len(row) for row in rows) * self.font.char_width
        self.height = len(rows) * self.font.char_height
        self.backup_vaos = []
        self.create_vaos_and_buffer_data()

    def previous_text_at_index(self, index: int):
        return self._previous_text[index] if index < len(self._previous_text) else ''

    def get_position_offset(self) -> Tuple[float, float]:
        which_point = self.which_point

        if which_point == 'topleft':
            return self.x, self.y
        if which_point == 'midtop':
            return self.x - self.width / 2, self.y
        if which_point == 'topright':
            return self.x - self.width, self.y
        if which_point == 'midleft':
            return self.x, self.y + self.height / 2
        if which_point == 'center':
            return self.x - self.width / 2, self.y + self.height / 2
        if which_point == 'midright':
            return self.x - self.width, self.y + self.height / 2
        if which_point == 'bottomleft':
            return self.x, self.y + self.height
        if which_point == 'midbottom':
            return self.x - self.width / 2, self.y + self.height
        if which_point == 'bottomright':
            return self.x + self.width, self.y + self.height

        raise ValueError(''.join([
            f'Invalid TextRenderer position offset point: {which_point}.\n',
            'Valid options: topleft, midtop, topright, midleft, center, midright,',
            ' bottomleft, midbottom, bottomright'
        ]))

    def create_vaos_and_buffer_data(self):
        while len(self.text) < len(self.vaos):
            self.backup_vaos.append(self.vaos.pop())

        offsets, texture_coords = self.get_offsets_and_texture_coords()

        special_char_counter = 0
        for i in range(len(self.text)):
            # If needed, try to pull a backup VAO into self.vaos
            if i >= len(self.vaos) and len(self.backup_vaos):
                self.vaos.append(self.backup_vaos.pop())

            if self.text[i] == '\n' or self.text[i] == '\t':
                special_char_counter += 1
            # If the character at index i changed, update the relevant VAO
            elif self.text[i] != self.previous_text_at_index(i):
                i = i - special_char_counter
                # If there is enough VAOs in self.vaos and a character has changed, update the buffers,
                # otherwise create a new VAO.
                if i < len(self.vaos):
                    self.vaos[i].get_buffer_by_name('in_offset').buffer.write(offsets[i])
                    self.vaos[i].get_buffer_by_name('in_texcoord_0').buffer.write(texture_coords[i])
                else:
                    vao = geom.quad_2d(
                        uvs=False, normals=False, size=(self.font.char_width, self.font.char_height),
                        pos=(self.font.char_width / 2, -self.font.char_height / 2)
                    )
                    vao.buffer(offsets[i], '2f/r', ['in_offset'])
                    vao.buffer(texture_coords[i], '2f/v', ['in_texcoord_0'])
                    self.vaos.append(vao)

    def get_offsets_and_texture_coords(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        font = self.font
        width = font.char_width
        height = font.char_height
        dx = width / font.texture_width
        dy = height / font.texture_height

        start_x, start_y = self.get_position_offset()
        texture_coords = []
        offsets = []
        x_offset = 0
        y_offset = 0
        for char in self.text:
            i = ord(char)
            x, y = i % 16, 9 - i // 16 - 2

            if char == '\n':
                x_offset = 0
                y_offset += font.char_height
            elif char == '\t':
                x_offset += 4 * font.char_width
            elif 32 <= i < 128:
                offsets.append(
                    np.array([start_x + x_offset, start_y - y_offset], dtype='f4')
                )
                texture_coords.append(np.array([
                     x      * dx, (y + 1) * dy,
                     x      * dx,  y      * dy,
                    (x + 1) * dx,  y      * dy,
                     x      * dx, (y + 1) * dy,
                    (x + 1) * dx,  y      * dy,
                    (x + 1) * dx, (y + 1) * dy
                ], dtype='f4'))
                x_offset += font.char_width

        return offsets, texture_coords

    def render(self, proj: Matrix44) -> None:
        self.program.write_uniform('projection', proj)
        self.program.set_uniform_value('text_color', self.color)

        self.font.texture.use(location=0)

        for vao in self.vaos:
            vao.render(self.program.program)


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
        self.instance_models = None
        self.instance_mvps = None
        self.instance_normal_mats = None
        self.instance_texture_coords = None
        self.create_vao_and_buffers()

    def create_vao_and_buffers(self) -> None:
        ctx = mglw.ctx()
        length = self.instance_count

        if self.vao:
            self.vao.release()

        self.vao = self.vao_generator(*self.vao_generator_args, **self.vao_generator_kwargs)

        # This fixes a performance issue likely caused by a bug in the macOS OpenGL driver
        if not self.vao._index_buffer:
            self.vao.index_buffer(np.arange(self.vao.vertex_count, dtype=np.uint32))

        self.instance_models = ctx.buffer(reserve=4 * 16 * length)
        self.instance_mvps = ctx.buffer(reserve=4 * 16 * length)
        self.instance_normal_mats = ctx.buffer(reserve=4 * 9 * length)

        if self.texture:
            self.instance_texture_coords = ctx.buffer(reserve=4 * 2 * length)
            self.vao.buffer(self.instance_texture_coords, '2f/i', ['in_texcoord'])

        self.vao.buffer(self.instance_mvps, '16f/i', ['mvp'])
        self.vao.buffer(self.instance_normal_mats, '9f/i', ['normal_mat'])
        self.vao.buffer(self.instance_models, '16f/i', ['model'])

    def get_instance_data(self, view_proj: Matrix44) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        transforms = [transform.transformation_matrix for transform in self.transforms]
        models = np.array(transforms, dtype='f4')
        mvps = np.array([view_proj * model for model in transforms], dtype='f4')
        normal_mats = np.array([Matrix33(model.inverse.T) for model in transforms], dtype='f4')

        return models, mvps, normal_mats

    def add_instance(self, transform: Transform3D) -> None:
        self.instance_count += 1
        self.create_vao_and_buffers()
        self.transforms.append(transform)

    def add_instances(self, transforms: Sequence[Transform3D]) -> None:
        if len(transforms):
            self.instance_count += len(transforms)
            self.create_vao_and_buffers()
            self.transforms.extend(transforms)

    def update(self, view_proj: Matrix44) -> None:
        models, mvps, normal_mats = self.get_instance_data(view_proj)
        self.instance_models.write(models)
        self.instance_mvps.write(mvps)
        self.instance_normal_mats.write(normal_mats)

    def render(
            self,
            value_uniforms: Optional[Mapping[str, Any]]=None,
            write_uniforms: Optional[Mapping[str, Any]]=None,
            override_program: Optional[ShaderProgram]=None) -> None:
        program = override_program if override_program else self.program

        if self.color is not None:
            program.write_uniform('in_color', self.color)

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
        ctx = mglw.ctx()
        self.pos = pos
        self.color = color.r / 255, color.g / 255, color.b / 255
        self.brightness = brightness
        self.pos_buffer = ctx.buffer(pos)
        self.color_buffer = ctx.buffer(Vector3(self.color, dtype='f4'))

        self.near = abs(self.pos[2]) / 4.0
        self.far = (self.pos[0]**2 + self.pos[1]**2 + self.pos[2]**2)**0.5 + (board_width**2 + board_height**2)**0.5

        left = -1.5 * board_width - abs(pos[0])
        right = 1.5 * board_width + abs(pos[0])
        top = -1.5 * board_height - abs(pos[1])
        bottom = 1.5 * board_height + abs(pos[1])
        self.proj = Matrix44.orthogonal_projection(
            left, right, top, bottom, self.near, self.far, dtype='f4'
        )

        self.view = Matrix44.look_at(
            pos, (0, 0, 0), (0, 1, 0), dtype='f4'
        )
        self.view_proj = self.proj * self.view
        self.view_proj_buffer = ctx.buffer(self.view_proj)

        transforms = Transform3D(translation=pos)
        self.sphere = InstancedObject(
            1, geom.sphere, 'lighting', rendered_color, [transforms],
            vao_generator_kwargs={'radius': radius, 'sectors': 64, 'rings': 64}
        )

        self.should_update = True

    def update(self, view_proj: Matrix44) -> None:
        if self.should_update:
            self.sphere.update(view_proj)
            self.should_update = False

    def render(self):
        self.sphere.render(value_uniforms={
            'is_light_source': True, 'brightness_mult': self.brightness
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

        self.instanced_cube = InstancedObject(
            len(transforms), geom.cube, 'lighting', theme.snake, transforms,
            vao_generator_kwargs={'size': (2.0, 2.0, 2.0)}
        )

        self.instanced_sphere = InstancedObject(
            2, geom.sphere, 'lighting', theme.eyes, self.get_eye_transforms(),
            vao_generator_kwargs={'radius': 0.25}
        )

        self.first_update = True

    def get_eye_transforms(self) -> List[Transform3D]:
        direction = self.snake.direction.name

        left_transform = self.instanced_cube.transforms[0].copy()
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
            board_height: int) -> None:
        needs_update = self.first_update

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
                if i < self.instanced_cube.instance_count:
                    self.instanced_cube.transforms[i].translation = position
                else:
                    new_transforms.append(Transform3D(translation=position))
            self.instanced_cube.add_instances(new_transforms)
            self.instanced_cube.update(view_proj)

            self.instanced_sphere.transforms = self.get_eye_transforms()
            self.instanced_sphere.update(view_proj)

            self.first_update = False

    def render(self, override_program: Optional[ShaderProgram]=None) -> None:
        self.instanced_cube.render(
            value_uniforms={'specularStrength': 0.1, 'is_light_source': False},
            override_program=override_program
        )
        self.instanced_sphere.render(
            value_uniforms={'specularStrength': 1.0, 'is_light_source': False},
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

        self.first_update = True

    def update(self, view_proj: Matrix44, board_width: int, board_height: int) -> None:
        needs_update = self.first_update

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

            self.first_update = False

    def render(self, override_program: Optional[ShaderProgram]=None) -> None:
        self.instanced_cube.render(
            value_uniforms={'specularStrength': 0.6, 'is_light_source': False},
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

        self.first_update = True

    def update(self, view_proj: Matrix44, board_width: int, board_height: int) -> None:
        food_pos = self.game.food.pos
        needs_update = self.food_pos != food_pos or self.first_update

        if needs_update:
            # Update food position
            self.food_pos = food_pos
            self.sphere.transforms[0].translation = Scene.grid_position(
                *food_pos, board_width, board_height
            )
            self.sphere.update(view_proj)

            self.first_update = False
            
    def render(self, override_program: Optional[ShaderProgram]=None) -> None:
        self.sphere.render(
            value_uniforms={'specularStrength': 0.6, 'is_light_source': False},
            override_program=override_program
        )


class BoardRenderer(Renderable):
    def __init__(self, board_width: int, board_height: int, theme: Theme):
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

        self.should_update = True

        transforms = Transform3D(
            translation=Vector3([0, -0.05, 0], dtype='f4'),
            scale=Vector3([board_width * 2, board_height * 2, 1], dtype='f4'),
            rotation=rotation
        )

        self.quad = InstancedObject(1, geom.quad_2d, 'lighting', theme.background, [transforms])

    def update(self, view_proj: Matrix44) -> None:
        if self.should_update:
            self.should_update = False
            self.quad.update(view_proj)
            self.grid.update(view_proj)

    def render(self, override_program: Optional[ShaderProgram]=None) -> None:
        self.quad.render(
            value_uniforms={'specularStrength': 0.6, 'is_light_source': False},
            override_program=override_program
        )
        self.grid.render(
            value_uniforms={'specularStrength': 0.0, 'is_light_source': False},
            override_program=override_program
        )


class ShadowMap(Renderable):
    def __init__(self, width: int=4096, height: int=4096):
        self.width, self.height = width, height
        self.shadow_map = mglw.ctx().depth_texture((self.width, self.height))
        self.shadow_map.compare_func = ''
        self.shadow_map.repeat_x = False
        self.shadow_map.repeat_y = False
        self.shadow_map.filter = moderngl.LINEAR, moderngl.LINEAR

        self.frame_buffer = mglw.ctx().framebuffer(depth_attachment=self.shadow_map)

    def render(self, renderables: Sequence[Renderable]):
        self.frame_buffer.clear()
        self.frame_buffer.use()

        prog = state.shader_program_repo['shadow_map_depth']

        for renderable in renderables:
            renderable.render(override_program=prog)


class HDRBloomRenderer(Renderable):  # TODO: Multisampling
    def __init__(self):
        viewport_dimensions = get_viewport_dimensions()
        ctx = mglw.ctx()

        self.scene_texture = ctx.texture(viewport_dimensions, 4, dtype='f2')
        self.brightness_texture = ctx.texture(viewport_dimensions, 4, dtype='f2')
        hdr_depth_texture = ctx.depth_texture(viewport_dimensions)
        hdr_textures = [self.scene_texture, self.brightness_texture]

        ping_pong_dimensions = viewport_dimensions[0] // 6, viewport_dimensions[1] // 6

        self.ping_pong_textures = [
            ctx.texture(ping_pong_dimensions, 4, dtype='f2'),
            ctx.texture(ping_pong_dimensions, 4, dtype='f2')
        ]
        for texture in chain(self.ping_pong_textures, hdr_textures):
            texture.repeat_x = False
            texture.repeat_y = False
            texture.filter = moderngl.LINEAR, moderngl.LINEAR

        self.hdr_framebuffer = ctx.framebuffer(
            depth_attachment=hdr_depth_texture, color_attachments=hdr_textures
        )

        self.ping_pong_framebuffers = [
            ctx.framebuffer(color_attachments=[self.ping_pong_textures[0]]),
            ctx.framebuffer(color_attachments=[self.ping_pong_textures[1]])
        ]

        self.blur_program = state.shader_program_repo['blur']

        prog_name = 'bloom_final_fxaa2'
        prog = state.shader_program_repo[prog_name].program
        prog['scene'].value = 0
        prog['bloomBlur'].value = 1

        self.quad = InstancedObject(
            1, geom.quad_2d, prog_name, None, [Transform3D()], vao_generator_kwargs={'size': (2.0, 2.0)}
        )

    def blur(self) -> bool:
        horizontal = True
        first_iteration = True
        amount = 12
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

        for renderable in renderables:
            renderable.render()

        horizontal = self.blur()

        mglw.ctx().screen.use()
        self.scene_texture.use(location=0)
        self.ping_pong_textures[horizontal].use(location=1)
        self.quad.render(value_uniforms={'exposure': 1.0})


class Scene(Renderable):
    def __init__(self, game: Game, theme: Theme, aspect_ratio: float=16 / 9):
        ctx = mglw.ctx()

        self.aspect_ratio = aspect_ratio
        self.theme = theme
        self.game = game
        self.board_width = game.board_width
        self.board_height = game.board_height

        max_dim = max(self.board_width / aspect_ratio, self.board_height)
        
        self.camera_pos = 0.0, max_dim * 2.25, max_dim * 2.1
        self.camera_pos_buffer = ctx.buffer(Vector3(self.camera_pos, dtype='f4'))

        self.proj = Matrix44.perspective_projection(
            45.0, self.aspect_ratio, 0.1, self.camera_pos[2] * 2, dtype='f4'
        )
        self.view = Matrix44.look_at(
            self.camera_pos, (0, 0, 0), (0, 1, 0), dtype='f4'
        )
        self.view_proj = self.proj * self.view

        self.shadow_map = ShadowMap()

        light_pos = -max_dim * 0.9, max_dim * 1.1, -max_dim * 0.65
        # light_pos = self.camera_pos
        self.light = Light(
            Vector3(light_pos, dtype='f4'), Color(255, 255, 255), self.board_width, self.board_height,
            radius=max_dim / 4
        )
        self.light.update(self.view_proj)

        # Create board
        self.board_renderer = BoardRenderer(self.board_width, self.board_height, theme)
        self.board_renderer.update(self.view_proj)

        # Create walls
        self.wall_renderer = WallRenderer(game, theme)

        # Create snake
        self.snake_renderer = SnakeRenderer(game, theme)

        # Create food
        self.food_renderer = FoodRenderer(game, theme)

        prog = state.shader_program_repo['lighting'].program
        prog['LightPos'].binding = 1
        prog['LightColor'].binding = 2
        prog['ViewPos'].binding = 3
        prog['LightSpaceMatrix'].binding = 4

        prog = state.shader_program_repo['shadow_map_depth'].program
        prog['LightSpaceMatrix'].binding = 4

        self.scope = ctx.scope(
            None, moderngl.DEPTH_TEST | moderngl.CULL_FACE, uniform_buffers=[
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

    @staticmethod
    def grid_position(x: int, y: int, width: int, height: int) -> Vector3:
        return Vector3(
            [2 * x + 1 - width, 1, 2 * y + 1 - height], dtype='f4'
        )

    def update(self) -> None:
        self.snake_renderer.update(self.view_proj, self.board_width, self.board_height)
        self.wall_renderer.update(self.view_proj, self.board_width, self.board_height)
        self.food_renderer.update(self.view_proj, self.board_width, self.board_height)

    def render(self, time: float, frame_time: float) -> None:
        self.update()

        with self.scope:
            # Render shadow map
            self.shadow_map.render(self.shadow_map_pass_renderables)

            self.shadow_map.shadow_map.use(location=0)
            self.hdr_bloom_renderer.render(self.hdr_bloom_pass_renderables)

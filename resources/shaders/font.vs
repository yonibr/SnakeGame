#version 330 core

uniform mat4 projection;

in vec2 in_position;
in vec2 in_offset;
in vec2 in_texcoord_0;

out vec2 TexCoords;

void main()
{
    gl_Position = projection * vec4(in_position + in_offset, 0.0, 1.0);
    TexCoords = in_texcoord_0;
}  
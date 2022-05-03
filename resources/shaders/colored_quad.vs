#version 330

uniform mat4 Mvp;
uniform vec3 in_color;

in vec3 in_position;

out vec3 v_color;

void main() {
    v_color = in_color;
    gl_Position = Mvp * vec4(in_position, 1.0);
}
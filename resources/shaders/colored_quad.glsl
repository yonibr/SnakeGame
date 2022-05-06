#version 330

#if defined VERTEX_SHADER

uniform mat4 Mvp;
uniform vec3 in_color;

in vec3 in_position;

out vec3 v_color;

void main() {
    v_color = in_color;
    gl_Position = Mvp * vec4(in_position, 1.0);
}

#elif defined FRAGMENT_SHADER

in vec3 v_color;

out vec4 f_color;

void main() {
    f_color = vec4(v_color.rgb, 1.0);
}

#endif

#version 330

in vec3 v_color;

out vec4 f_color;

void main() {
    f_color = vec4(v_color.rgb, 1.0);
}
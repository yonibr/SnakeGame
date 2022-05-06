#version 330

#if defined VERTEX_SHADER

in vec3 in_position;
in mat4 model;

uniform LightSpaceMatrix {
    mat4 matrix;
} lightSpace;

void main() {
    gl_Position = lightSpace.matrix * model * vec4(in_position, 1.0);
}

#elif defined FRAGMENT_SHADER

void main() {
}

#endif

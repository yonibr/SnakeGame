#version 330

#if defined VERTEX_SHADER

uniform mat4 Mvp;

in vec3 in_position;

out vec3 Color;

void main() {
    gl_Position = Mvp * vec4(in_position, 1.0);
}

#elif defined FRAGMENT_SHADER

uniform vec3 color;

in vec3 Color;

out vec4 FragColor;

void main() {
    const float gamma = 2.2;
    const vec3 correctionVec = vec3(1.0 / gamma);
    
    // Gamma correction
    vec3 result = pow(color, correctionVec);
    FragColor = vec4(result, 1.0);
}

#endif

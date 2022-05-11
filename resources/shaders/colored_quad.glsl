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
    const float gamma = 2.2;
    const vec3 correction_vec = vec3(1.0 / gamma);
    
    // Gamma correction
    vec3 result = pow(v_color.rgb, correction_vec);
    f_color = vec4(result, 1.0);
}

#endif

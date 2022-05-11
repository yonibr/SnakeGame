#version 330 core

#if defined VERTEX_SHADER

in vec3 in_position;
in vec2 in_texcoord_0;

out vec2 TexCoords;

void main()
{
    TexCoords = in_texcoord_0;
    gl_Position = vec4(in_position, 1.0);
}

#elif defined FRAGMENT_SHADER

uniform sampler2D scene;
uniform sampler2D bloomBlur;

uniform float exposure = 1.0;

in vec2 TexCoords;

out vec4 FragColor;

void main()
{
    const float gamma = 2.2;
    const vec3 correction_vec = vec3(1.0 / gamma);

    vec3 hdrColor = texture(scene, TexCoords).rgb;
    hdrColor += texture(bloomBlur, TexCoords).rgb; // additive blending

    // tone mapping
    vec3 result = vec3(1.0) - exp(-hdrColor * exposure);

    // Gamma correction
    result = pow(result, correction_vec);

    FragColor = vec4(result, 1.0);
}

#endif
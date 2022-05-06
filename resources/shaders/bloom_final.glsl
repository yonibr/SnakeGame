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

uniform bool bloom;
uniform float exposure;

in vec2 TexCoords;

out vec4 FragColor;

void main()
{
    const float gamma = 2.2;

    vec3 hdrColor = texture(scene, TexCoords).rgb;
    if (bloom)
        hdrColor += texture(bloomBlur, TexCoords).rgb; // additive blending
//    hdrColor = texture(bloomBlur, TexCoords).rgb;

    // tone mapping
    vec3 result = vec3(1.0) - exp(-hdrColor * exposure);

    // also gamma correct while we're at it
    result = pow(result, vec3(1.0 / gamma));

    FragColor = vec4(result, 1.0);
}

#endif
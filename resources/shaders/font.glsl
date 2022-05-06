#version 330 core

#if defined VERTEX_SHADER

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

#elif defined FRAGMENT_SHADER

uniform sampler2D texture;
uniform vec3 text_color;

in vec2 TexCoords;
out vec4 FragColor;

void main()
{    
    vec4 sampled = vec4(1.0, 1.0, 1.0, texture(texture, TexCoords).r);
    FragColor = vec4(text_color, 1.0) * sampled;
}

#endif
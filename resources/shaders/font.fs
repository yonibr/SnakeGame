#version 330 core

uniform sampler2D texture;
uniform vec3 text_color;

in vec2 TexCoords;
out vec4 color;

void main()
{    
    vec4 sampled = vec4(1.0, 1.0, 1.0, texture(texture, TexCoords).r);
    color = vec4(text_color, 1.0) * sampled;
}  
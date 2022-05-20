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

uniform sampler2D image;

uniform bool horizontal;
uniform float offsets[3] = float[](0.0, 1.3846153846, 3.2307692308);
uniform float weights[3] = float[](0.2270270270, 0.3162162162, 0.0702702703);

in vec2 TexCoords;

out vec4 FragColor;

void main()
{
     vec2 tex_size = textureSize(image, 0); // gets size of single texel
     vec3 result = texture(image, TexCoords).rgb * weights[0];
     if(horizontal)
     {
         for(int i = 1; i < 3; ++i)
         {
             vec2 offset =  vec2(offsets[i] / tex_size.x, 0.0);
             result += texture(image, TexCoords + offset).rgb * weights[i];
             result += texture(image, TexCoords - offset).rgb * weights[i];
         }
     }
     else
     {
         for(int i = 1; i < 3; ++i)
         {
             vec2 offset =  vec2(0.0, offsets[i] / tex_size.y);
             result += texture(image, TexCoords + offset).rgb * weights[i];
             result += texture(image, TexCoords - offset).rgb * weights[i];
         }
     }
     FragColor = vec4(result, 1.0);
}

#endif
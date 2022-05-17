#version 330 core

#if defined VERTEX_SHADER


in vec2 in_position;
in vec2 in_texcoord_0;

out VS_OUT {
    vec2 TexCoords;
} vs_out;

void main()
{
    gl_Position = vec4(in_position, 0.0, 1.0);
    vs_out.TexCoords = in_texcoord_0;
}

#elif defined GEOMETRY_SHADER

layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

uniform int charWidth;
uniform int charHeight;
uniform int textureWidth;
uniform int textureHeight;

uniform mat4 projection;

in VS_OUT {
    vec2 TexCoords;
} gs_in[];

out vec2 TexCoords;

void main() {
    float dx = float(charWidth) / float(textureWidth);
    float dy = float(charHeight) / float(textureHeight);

    vec4 position = gl_in[0].gl_Position;

    // Bottom left
    gl_Position = projection * (position + vec4(0.0, -charHeight, 0.0, 0.0));
    TexCoords = gs_in[0].TexCoords;
    EmitVertex();

    // Bottom Right
    gl_Position = projection * (position + vec4(charWidth, -charHeight, 0.0, 0.0));
    TexCoords = gs_in[0].TexCoords + vec2(dx, 0.0);
    EmitVertex();

    // Top Left
    gl_Position = projection * (position + vec4(0.0, 0.0, 0.0, 0.0));
    TexCoords = gs_in[0].TexCoords + vec2(0.0, dy);
    EmitVertex();

    // Top right
    gl_Position = projection * (position + vec4(charWidth, 0.0, 0.0, 0.0));
    TexCoords = gs_in[0].TexCoords + vec2(dx, dy);
    EmitVertex();

    EndPrimitive();
}

#elif defined FRAGMENT_SHADER

uniform sampler2D texture;
uniform vec3 text_color;

in vec2 TexCoords;
out vec4 FragColor;

void main()
{
    const float gamma = 2.2;
    const vec3 correction_vec = vec3(1.0 / gamma);

    float alpha = texture(texture, TexCoords).r;
    vec3 gamma_corrected_color = pow(text_color, correction_vec);
    FragColor = vec4(gamma_corrected_color, alpha);
}

#endif
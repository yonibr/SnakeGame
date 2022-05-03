#version 330 core

uniform vec3 in_color;

in mat4 model;
in mat3 normal_mat;
in mat4 mvp;
in vec3 in_position;
in vec3 in_normal;

out vec3 FragPos;
out vec3 Normal;
out vec3 Color;


void main()
{
    Color = in_color;
    Normal = normal_mat * in_normal;
    FragPos = vec3(model * vec4(in_position, 1.0));
    
    gl_Position = mvp * vec4(in_position, 1.0);
}

#version 330 core

uniform float specularStrength;

uniform LightPos
{
    vec3 val; 
} lightPos;

uniform ViewPos
{
    vec3 val; 
} viewPos;

uniform LightColor
{
    vec3 val; 
} lightColor;

in vec3 Normal;  
in vec3 FragPos;  
in vec3 Color;  

out vec4 FragColor;

void main()
{
    // ambient
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor.val;
  	
    // diffuse 
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos.val - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor.val;
    
    // specular
    vec3 viewDir = normalize(viewPos.val - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);  
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 16);
    vec3 specular = specularStrength * spec * lightColor.val;  
        
    vec3 result = (ambient + diffuse + specular) * Color;
    FragColor = vec4(result, 1.0);
} 
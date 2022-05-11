#version 330

#if defined VERTEX_SHADER

uniform LightSpaceMatrix {
    mat4 matrix;
} lightSpace;

uniform vec3 in_color;

in vec3 in_position;
in vec3 in_normal;
in mat4 model;
in mat3 normal_mat;
in mat4 mvp;

out vec2 TexCoords;

out VS_OUT {
    vec3 FragPos;
    vec3 Normal;
    vec3 Color;
    vec4 FragPosLightSpace;
} vs_out;


void main()
{
    vs_out.FragPos = vec3(model * vec4(in_position, 1.0));
    vs_out.Normal = normal_mat * in_normal;
    vs_out.Color = in_color;
    vs_out.FragPosLightSpace = lightSpace.matrix * vec4(vs_out.FragPos, 1.0);
    gl_Position = mvp * vec4(in_position, 1.0);
}

#elif defined FRAGMENT_SHADER

uniform sampler2D shadowMap;
uniform float specularStrength;

uniform ViewPos {
    vec3 val; 
} viewPos;

uniform LightPos {
    vec3 val; 
} lightPos;

uniform LightColor {
    vec3 val; 
} lightColor;

in VS_OUT {
    vec3 FragPos;
    vec3 Normal;
    vec3 Color;
    vec4 FragPosLightSpace;
} fs_in;

out vec4 FragColor;

float ShadowCalculation(vec4 fragPosLightSpace)
{
    // perform perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    // transform to [0,1] range
    projCoords = projCoords * 0.5 + 0.5;
    // get closest depth value from light's perspective (using [0,1] range fragPosLight as coords)
    float closestDepth = texture(shadowMap, projCoords.xy).r; 
    // get depth of current fragment from light's perspective
    float currentDepth = projCoords.z;
    // calculate bias (based on depth map resolution and slope)
    vec3 normal = normalize(fs_in.Normal);
    vec3 lightDir = normalize(lightPos.val - fs_in.FragPos);
    float bias = max(0.05 * (1.0 - dot(normal, lightDir)), 0.005);
    // check whether current frag pos is in shadow
    // float shadow = currentDepth - bias > closestDepth  ? 1.0 : 0.0;
    // PCF
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    for(int x = -1; x <= 1; ++x)
    {
        for(int y = -1; y <= 1; ++y)
        {
            float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r; 
            shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;        
        }    
    }
    shadow /= 9.0;
    
    // keep the shadow at 0.0 when outside the far_plane region of the light's frustum.
    if(projCoords.z > 1.0)
        shadow = 0.0;

    return shadow;
}

void main()
{
    vec3 color = fs_in.Color;
    vec3 normal = normalize(fs_in.Normal);

    // ambient
    vec3 ambient = 0.3 * lightColor.val;

    // diffuse
    vec3 lightDir = normalize(lightPos.val - fs_in.FragPos);
    float diff = max(dot(lightDir, normal), 0.0);
    vec3 diffuse = diff * lightColor.val;

    // specular
    vec3 viewDir = normalize(viewPos.val - fs_in.FragPos);
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = 0.0;
    vec3 halfwayDir = normalize(lightDir + viewDir);
    spec = pow(max(dot(normal, halfwayDir), 0.0), 64.0);
    vec3 specular = specularStrength * spec * lightColor.val;

    // calculate shadow
    float shadow = ShadowCalculation(fs_in.FragPosLightSpace);

    vec3 lighting = (ambient + (1.0 - shadow) * (diffuse + specular)) * color;

    FragColor = vec4(lighting, 1.0);
    // FragColor = vec4(shadow, shadow, shadow, 1.0);
}

#endif
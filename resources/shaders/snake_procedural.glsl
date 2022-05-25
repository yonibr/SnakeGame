#version 330 core

#if defined VERTEX_SHADER

in vec4 in_position;
in mat4 model;
in mat3 normal_mat;
in mat4 mvp;

out VS_OUT {
    mat4 Mvp;
    mat3 NormalMat;
    mat4 Model;
} vs_out;

void main()
{
    vs_out.NormalMat = normal_mat;
    vs_out.Model = model;
    vs_out.Mvp = mvp;

    gl_Position = in_position;
}

#elif defined GEOMETRY_SHADER
// Geometry shader is a modified version of the following:
//    https://github.com/torbjoern/polydraw_scripts/blob/master/geometry/drawcone_geoshader.pss

#define PI 3.14159

layout (lines) in;
layout (triangle_strip, max_vertices = 72) out;

uniform LightSpaceMatrix {
    mat4 matrix;
} lightSpace;

in VS_OUT {
    mat4 Mvp;
    mat3 NormalMat;
    mat4 Model;
} gs_in[];

out GS_OUT {
    vec3 FragPos;
    vec3 Normal;
    vec4 FragPosLightSpace;
} gs_out;

// given to points p1 and p2 create a vector out
// that is perpendicular to (p2-p1)
vec3 createPerp(vec3 p1, vec3 p2)
{
    vec3 invec = normalize(p2 - p1);
    vec3 ret = cross(invec, vec3(0.0, 0.0, 1.0));

    if (length(ret) == 0.0)
        ret = cross(invec, vec3(0.0, 1.0, 0.0));

    return ret;
}

void createTube(vec3 pos1, vec3 pos2, float r1, float r2) {
    const int segs = 36;

    vec3 axis = normalize(pos2 - pos1);

    vec3 perpx = createPerp(pos2, pos1);
    vec3 perpy = cross(axis, perpx);

    for (int i = 0; i < segs; i++) {
        float a = i / float(segs - 1) * 2.0 * PI;
        float ca = cos(a);
        float sa = sin(a);
        vec3 normal = vec3(
            ca * perpx.x + sa * perpy.x,
            ca * perpx.y + sa * perpy.y,
            ca * perpx.z + sa * perpy.z
        );

        vec3 p1 = pos1 + r1 * normal;
        vec3 p2 = pos2 + r2 * normal;

        gl_Position = gs_in[0].Mvp * vec4(p1, 1.0);
        gs_out.FragPos = vec3(gs_in[0].Model * vec4(p1, 1.0));
        gs_out.FragPosLightSpace = lightSpace.matrix * vec4(gs_out.FragPos, 1.0);
        gs_out.Normal = normalize(gs_in[0].NormalMat * normal);
        EmitVertex();

        gl_Position = gs_in[1].Mvp * vec4(p2, 1.0);
        gs_out.FragPos = vec3(gs_in[1].Model * vec4(p2, 1.0));
        gs_out.FragPosLightSpace = lightSpace.matrix * vec4(gs_out.FragPos, 1.0);
        gs_out.Normal = normalize(gs_in[1].NormalMat * normal);
        EmitVertex();
    }
    EndPrimitive();
}

void main()
{
    vec4 pos1 = gl_in[0].gl_Position;
    vec4 pos2 = gl_in[1].gl_Position;

    createTube(pos1.xyz, pos2.xyz, pos1.w, pos2.w);
}

#elif defined FRAGMENT_SHADER

uniform sampler2D shadowMap;
uniform float specularStrength;

uniform vec3 color;

uniform ViewPos {
    vec3 val;
} viewPos;

uniform LightPos {
    vec3 val;
} lightPos;

uniform LightColor {
    vec3 val;
} lightColor;

in GS_OUT {
    vec3 FragPos;
    vec3 Normal;
    vec4 FragPosLightSpace;
} fs_in;

layout (location = 0) out vec4 FragColor;
layout (location = 1) out vec4 BrightColor;

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
    vec3 normal = fs_in.Normal;
    vec3 lightDir = normalize(lightPos.val - fs_in.FragPos);
    float bias = max(0.0075 * (1.0 - dot(normal, lightDir)), 0.0025);
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
    vec3 normal = fs_in.Normal;

    // ambient
    vec3 ambient = 0.2 * lightColor.val;

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

    // Calculate lighting value
    vec3 lighting = (ambient + (1.0 - shadow) * (diffuse + specular)) * color;

    // check whether result is higher than some threshold, if so, output as bloom threshold color
    float brightness = dot(lighting, vec3(0.2126, 0.7152, 0.0722));
    if (brightness > 1.0)
        BrightColor = vec4(lighting, 1.0);
    else
        BrightColor = vec4(0.0, 0.0, 0.0, 1.0);

    FragColor = vec4(lighting, 1.0);
}

#endif
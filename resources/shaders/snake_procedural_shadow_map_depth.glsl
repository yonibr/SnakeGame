#version 330

#if defined VERTEX_SHADER

in vec4 in_position;
in mat4 model;

out VS_OUT {
    mat4 model;
} vs_out;

void main() {
    vs_out.model = model;
    gl_Position = in_position;
}
#elif defined GEOMETRY_SHADER
// Geometry shader is a odified version of the following:
//    https://github.com/torbjoern/polydraw_scripts/blob/master/geometry/drawcone_geoshader.pss

#define PI 3.14159

layout (lines) in;
layout (triangle_strip, max_vertices = 32) out;

uniform LightSpaceMatrix {
    mat4 matrix;
} lightSpace;

in VS_OUT {
    mat4 model;
} gs_in[];

// given to points p1 and p2 create a vector out
// that is perpendicular to (p2-p1)
vec3 createPerp(vec3 p1, vec3 p2)
{
    vec3 invec = normalize(p2 - p1);
    vec3 ret = cross( invec, vec3(0.0, 0.0, 1.0) );
    if ( length(ret) == 0.0 )
    {
        ret = cross( invec, vec3(0.0, 1.0, 0.0) );
    }
    return ret;
}

void createTube(vec3 pos1, vec3 pos2, float r1, float r2) {
    const int segs = 16;

    vec3 axis = normalize(pos2.xyz - pos1.xyz);

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

        gl_Position = lightSpace.matrix * gs_in[0].model * vec4(p1, 1.0);
        EmitVertex();

        gl_Position = lightSpace.matrix * gs_in[1].model * vec4(p2, 1.0);
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

void main() {
}

#endif

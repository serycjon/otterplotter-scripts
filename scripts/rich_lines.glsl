#version 330

/*
This example is a port to ModernGL of code by Nicolas P. Rougier from his "Python & OpenGL
for Scientific Visualization" free online book. Available under the (new) BSD License.

Book is available here:
https://github.com/rougier/python-opengl

Background information on this code:
https://github.com/rougier/python-opengl/blob/master/09-lines.rst

Original code on which this example is based:
https://github.com/rougier/python-opengl/blob/master/code/chapter-09/geom-path.py
*/

#if defined VERTEX_SHADER

in vec2 in_position;
in vec4 in_color;
out vec4 v_color;

void main()
{
    gl_Position = vec4(in_position.x, in_position.y, 0.0, 1.0);
    v_color = vec4(in_color);
}

#elif defined FRAGMENT_SHADER

vec4 stroke(float distance, float linewidth, float antialias, vec4 color)
{
    vec4 frag_color;
    float t = linewidth/2.0 - antialias;
    float signed_distance = distance;
    float border_distance = abs(signed_distance) - t;
    float alpha = border_distance/antialias;

    alpha = exp(-alpha*alpha);

    if (border_distance > (linewidth/2.0 + antialias))
    {
        discard;
    }
    else if (border_distance < 0.0)
    {
        frag_color = color;
    }
    else
    {
        frag_color = vec4(color.rgb, color.a * alpha);
    }

    return frag_color;
}

vec4 cap(int type, float dx, float dy, float linewidth, float antialias, vec4 color)
{
    float d = 0.0;
    dx = abs(dx);
    dy = abs(dy);
    float t = linewidth/2.0 - antialias;

    if (type == 0)  // None
        discard;
    else if (type == 1) // Round
        d = sqrt(dx*dx+dy*dy);
    else if (type == 3) // Triangle in
        d = (dx+abs(dy));
    else if (type == 2) // Triangle out
        d = max(abs(dy), (t+dx-abs(dy)));
    else if (type == 4) // Square
        d = max(dx, dy);
    else if (type == 5) // Butt
        d = max(dx+t, dy);

    return stroke(d, linewidth, antialias, color);
}

// uniform vec4  color;
uniform float antialias;
uniform float linewidth;
uniform float miter_limit;

in float v_length;
in vec2  v_caps;
in vec2  v_texcoord;
in vec2  v_bevel_distance;
in vec4  g_color;

out vec4 fragColor;

void main()
{
    float distance = v_texcoord.y;

    if (v_caps.x < 0.0) {
        fragColor = cap(1, v_texcoord.x, v_texcoord.y, linewidth, antialias, g_color);
        return;
    }

    if (v_caps.y > v_length) {
        fragColor = cap(1, v_texcoord.x-v_length, v_texcoord.y, linewidth, antialias, g_color);
        return;
    }
    // Round join (instead of miter)
    if (miter_limit < 0) {
        if (v_texcoord.x < 0.0) {
            distance = length(v_texcoord);
        } else if (v_texcoord.x > v_length) {
            distance = length(v_texcoord - vec2(v_length, 0.0));
        }
    } else {
        // Miter limit
        float t = (miter_limit-1.0)*(linewidth/2.0) + antialias;

        if ((v_texcoord.x < 0.0) && (v_bevel_distance.x > (abs(distance) + t))) {
            distance = v_bevel_distance.x - t;
        } else if ((v_texcoord.x > v_length) && (v_bevel_distance.y > (abs(distance) + t))) {
            distance = v_bevel_distance.y - t;
        }
    }

    fragColor = stroke(distance, linewidth, antialias, g_color);
}

#elif defined GEOMETRY_SHADER

layout(lines_adjacency) in;// 4 points at the time from vertex shader
layout(triangle_strip, max_vertices = 4) out;// Outputs a triangle strip with 4 vertices

uniform mat4 projection;
uniform float antialias;
uniform float linewidth;
uniform float miter_limit;

in vec4 v_color[];
out vec2 v_caps;
out float v_length;
out vec2 v_texcoord;
out vec2 v_bevel_distance;
out vec4 g_color;

float compute_u(vec2 p0, vec2 p1, vec2 p)
{
    // Projection p' of p such that p' = p0 + u*(p1-p0)
    // Then  u *= length(p1-p0)
    vec2 v = p1 - p0;
    float l = length(v);

    return ((p.x-p0.x)*v.x + (p.y-p0.y)*v.y) / l;
}
float line_distance(vec2 p0, vec2 p1, vec2 p)
{
    // Projection p' of p such that p' = p0 + u*(p1-p0)
    vec2 v = p1 - p0;
    float l2 = v.x*v.x + v.y*v.y;
    float u = ((p.x-p0.x)*v.x + (p.y-p0.y)*v.y) / l2;
    // h is the projection of p on (p0,p1)
    vec2 h = p0 + u*v;

    return length(p-h);
}
void main(void)
{
    // Get the four vertices passed to the shader
    vec2 p0 = gl_in[0].gl_Position.xy;// start of previous segment
    vec2 p1 = gl_in[1].gl_Position.xy;// end of previous segment, start of current segment
    vec2 p2 = gl_in[2].gl_Position.xy;// end of current segment, start of next segment
    vec2 p3 = gl_in[3].gl_Position.xy;// end of next segment
    g_color = v_color[0];

    // Determine the direction of each of the 3 segments (previous, current, next)
    vec2 v0 = normalize(p1 - p0);
    vec2 v1 = normalize(p2 - p1);
    vec2 v2 = normalize(p3 - p2);

    // Determine the normal of each of the 3 segments (previous, current, next)
    vec2 n0 = vec2(-v0.y, v0.x);
    vec2 n1 = vec2(-v1.y, v1.x);
    vec2 n2 = vec2(-v2.y, v2.x);

    // Determine miter lines by averaging the normals of the 2 segments
    vec2 miter_a = normalize(n0 + n1);// miter at start of current segment
    vec2 miter_b = normalize(n1 + n2);// miter at end of current segment

    // Determine the length of the miter by projecting it onto normal
    vec2 p, v;
    float d;
    float w = linewidth/2.0 + antialias;
    v_length = length(p2-p1);
    float length_a = w / dot(miter_a, n1);
    float length_b = w / dot(miter_b, n1);
    float m = miter_limit*linewidth/2.0;

    // Angle between prev and current segment (sign only)
    float d0 = -sign(v0.x*v1.y - v0.y*v1.x);

    // Angle between current and next segment (sign only)
    float d1 = -sign(v1.x*v2.y - v1.y*v2.x);

    // Generate the triangle strip
    // First vertex
    // ------------------------------------------------------------------------
    // Cap at start
    if (p0 == p1) {
        p = p1 - w*v1 + w*n1;
        v_texcoord = vec2(-w, +w);
        v_caps.x = v_texcoord.x;
        // Regular join
    } else {
        p = p1 + length_a * miter_a;
        v_texcoord = vec2(compute_u(p1, p2, p), +w);
        v_caps.x = 1.0;
    }
    if (p2 == p3) v_caps.y = v_texcoord.x;
    else v_caps.y = 1.0;
    gl_Position = projection*vec4(p, 0.0, 1.0);
    v_bevel_distance.x = +d0*line_distance(p1+d0*n0*w, p1+d0*n1*w, p);
    v_bevel_distance.y =    -line_distance(p2+d1*n1*w, p2+d1*n2*w, p);
    EmitVertex();

    // Second vertex
    // ------------------------------------------------------------------------
    // Cap at start
    if (p0 == p1) {
        p = p1 - w*v1 - w*n1;
        v_texcoord = vec2(-w, -w);
        v_caps.x = v_texcoord.x;
        // Regular join
    } else {
        p = p1 - length_a * miter_a;
        v_texcoord = vec2(compute_u(p1, p2, p), -w);
        v_caps.x = 1.0;
    }
    if (p2 == p3) v_caps.y = v_texcoord.x;
    else v_caps.y = 1.0;
    gl_Position = projection*vec4(p, 0.0, 1.0);
    v_bevel_distance.x = -d0*line_distance(p1+d0*n0*w, p1+d0*n1*w, p);
    v_bevel_distance.y =    -line_distance(p2+d1*n1*w, p2+d1*n2*w, p);
    EmitVertex();

    // Third vertex
    // ------------------------------------------------------------------------
    // Cap at end
    if (p2 == p3) {
        p = p2 + w*v1 + w*n1;
        v_texcoord = vec2(v_length+w, +w);
        v_caps.y = v_texcoord.x;
        // Regular join
    } else {
        p = p2 + length_b * miter_b;
        v_texcoord = vec2(compute_u(p1, p2, p), +w);
        v_caps.y = 1.0;
    }
    if (p0 == p1)
        v_caps.x = v_texcoord.x;
    else
        v_caps.x = 1.0;
    gl_Position = projection*vec4(p, 0.0, 1.0);
    v_bevel_distance.x = -line_distance(p1+d0*n0*w, p1+d0*n1*w, p);
    v_bevel_distance.y = +d1*line_distance(p2+d1*n1*w, p2+d1*n2*w, p);
    EmitVertex();

    // Fourth vertex
    // ------------------------------------------------------------------------
    // Cap at end
    if (p2 == p3) {
        p = p2 + w*v1 - w*n1;
        v_texcoord = vec2(v_length+w, -w);
        v_caps.y = v_texcoord.x;
        // Regular join
    } else {
        p = p2 - length_b * miter_b;
        v_texcoord = vec2(compute_u(p1, p2, p), -w);
        v_caps.y = 1.0;
    }

    if (p0 == p1)
        v_caps.x = v_texcoord.x;
    else
        v_caps.x = 1.0;

    gl_Position = projection*vec4(p, 0.0, 1.0);
    v_bevel_distance.x =    -line_distance(p1+d0*n0*w, p1+d0*n1*w, p);
    v_bevel_distance.y = -d1*line_distance(p2+d1*n1*w, p2+d1*n2*w, p);
    EmitVertex();
    EndPrimitive();
}

#endif

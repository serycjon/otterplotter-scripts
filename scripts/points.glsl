#version 330

#if defined VERTEX_SHADER

uniform mat4 projection;
uniform float point_r;
in vec2 in_position;
in vec4 in_color;
out vec4 v_color;

void main()
{
    gl_Position = projection * vec4(in_position, 0.0, 1.0);
    gl_PointSize = point_r;
    v_color = in_color;
}

#elif defined FRAGMENT_SHADER

in vec4  v_color;
out vec4 f_color;

void main()
{
  float dist = step(length(gl_PointCoord.xy - vec2(0.5)), 0.5);
  f_color = vec4(v_color * dist);
}

#endif

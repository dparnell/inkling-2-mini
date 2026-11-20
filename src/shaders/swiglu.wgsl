// Softcapped SwiGLU on interleaved gate/up rows: a[i, j] = cap_g(g) * sigmoid(g) * cap_u(u),
// g = h[i, 2j], u = h[i, 2j + 1], cap_c(v) = tanh(v / c) * c.
struct Params { rows: u32, inter: u32, gate_cap: f32, up_cap: f32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> h: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> a: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let i = linear_wg(wid, nwg) * 256u + lid;
    if (i >= p.rows * p.inter) { return; }
    let gu = h[i];
    let g = gu.x;
    let u = gu.y;
    a[i] = tanh_s(g / p.gate_cap) * p.gate_cap * sigmoid(g) * tanh_s(u / p.up_cap) * p.up_cap;
}

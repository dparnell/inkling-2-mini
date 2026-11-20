// In-place interleaved-pair rotary embedding on rows of [n][heads][hd] (q and k):
//   x' = x * cos + rot(x) * sin,  rot(x1, x2) = (-x2, x1);  cs: [n][hd / 2] (cos, sin) pairs
// One thread per (row, head, pair).
struct Params { n: u32, heads: u32, hd: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> cs: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> q: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> k: array<vec2<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let i = linear_wg(wid, nwg) * 256u + lid;
    let half_hd = p.hd / 2u;
    if (i >= p.n * p.heads * half_hd) { return; }
    let row = i / (p.heads * half_hd);
    let j = i % half_hd;
    let c = cs[row * half_hd + j];
    let a = q[i];
    q[i] = vec2<f32>(a.x * c.x - a.y * c.y, a.y * c.x + a.x * c.y);
    let b = k[i];
    k[i] = vec2<f32>(b.x * c.x - b.y * c.y, b.y * c.x + b.x * c.y);
}

// Normalise key (mode 0: L2 clip to thr) or value (mode 1: unweighted RMS norm) heads and store
// them as packed f16 into the cache, layout [kv_head][cap][head_dim / 2] u32, slot = pos % cap.
// One workgroup (64 threads, head_dim = 128) per (token, kv head).
struct Params { t: u32, nkv: u32, pos0: u32, cap: u32, mode: u32, thr: f32, eps: f32, x_stride: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> cache: array<u32>;
var<workgroup> red: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let id = linear_wg(wid, nwg);
    if (id >= p.t * p.nkv) { return; }
    let t = id / p.nkv;
    let h = id % p.nkv;
    let b = t * p.x_stride + h * 128u + lid * 2u;
    let a = x[b];
    let c = x[b + 1u];
    red[lid] = a * a + c * c;
    workgroupBarrier();
    for (var k = 32u; k > 0u; k >>= 1u) {
        if (lid < k) { red[lid] += red[lid + k]; }
        workgroupBarrier();
    }
    let ss = red[0];
    let f = select(min(p.thr / (sqrt(ss) + 1e-6), 1.0), inverseSqrt(ss / 128.0 + p.eps), p.mode == 1u);
    let slot = (p.pos0 + t) % p.cap;
    cache[(h * p.cap + slot) * 64u + lid] = pack2x16float(vec2<f32>(a * f, c * f));
}

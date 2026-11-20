// Merge split-K attention partials with the per-token sink logit:
//   M = max(sink, m_s);  L = exp(sink - M) + sum_s l_s exp(m_s - M);  out = sum_s acc_s exp(m_s - M) / L
// One workgroup (128 threads = head_dim) per (token, head).
struct Params { t: u32, nh: u32, splits: u32, s_stride: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> part: array<f32>;
@group(0) @binding(2) var<storage, read> sinks: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let id = linear_wg(wid, nwg);
    if (id >= p.t * p.nh) { return; }
    let q = id / p.nh;
    let h = id % p.nh;
    let sink = sinks[q * p.s_stride + h];
    let base = (h * p.t + q) * p.splits;
    var mx = sink;
    for (var s = 0u; s < p.splits; s++) { mx = max(mx, part[(base + s) * 130u]); }
    var l = exp(sink - mx);
    var acc = 0.0;
    for (var s = 0u; s < p.splits; s++) {
        let b = (base + s) * 130u;
        let f = exp(part[b] - mx);
        l += part[b + 1u] * f;
        acc += part[b + 2u + lid] * f;
    }
    out[q * p.nh * 128u + h * 128u + lid] = acc / l;
}

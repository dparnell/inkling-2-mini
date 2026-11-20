// out[t, d] = sum_j w[t, j] * y[inv[t * k + j], d]
struct Params { t: u32, d: u32, k: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read> inv: array<u32>;
@group(0) @binding(3) var<storage, read> w: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let i = linear_wg(wid, nwg) * 256u + lid;
    if (i >= p.t * p.d) { return; }
    let t = i / p.d;
    let d = i % p.d;
    var acc = 0.0;
    for (var j = 0u; j < p.k; j++) {
        acc += w[t * p.k + j] * y[inv[t * p.k + j] * p.d + d];
    }
    out[i] = acc;
}

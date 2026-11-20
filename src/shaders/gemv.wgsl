// Single-token matrix-vector product with bf16 weights:
//   y[y_off + n] = sum_k x[x_off + k] * W[n, k]     W row-major [N, K], K % 8 == 0
// Each workgroup computes 4 output rows with 64 threads per row.
struct Params { k: u32, n: u32, x_off: u32, y_off: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let row = linear_wg(wid, nwg) * 4u + lid / 64u;
    let l = lid % 64u;
    let k8 = p.k / 8u;
    let xb = p.x_off / 4u;
    var acc = 0.0;
    if (row < p.n) {
        let wb = row * k8;
        for (var i = l; i < k8; i += 64u) {
            let wv = w[wb + i];
            let xa = x[xb + i * 2u];
            let xc = x[xb + i * 2u + 1u];
            acc += xa.x * bf16_lo(wv.x) + xa.y * bf16_hi(wv.x) + xa.z * bf16_lo(wv.y) + xa.w * bf16_hi(wv.y)
                 + xc.x * bf16_lo(wv.z) + xc.y * bf16_hi(wv.z) + xc.z * bf16_lo(wv.w) + xc.w * bf16_hi(wv.w);
        }
    }
    red[lid] = acc;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s >>= 1u) {
        if (l < s) { red[lid] += red[lid + s]; }
        workgroupBarrier();
    }
    if (l == 0u && row < p.n) { y[p.y_off + row] = red[lid]; }
}

// Multi-token matmul with bf16 weights:
//   Y[t, y_off + n] = sum_k X[t * x_stride + k] * W[n, k]      K % 32 == 0, x_stride % 4 == 0
// 64x64 output tile per workgroup (tokens x outputs), 4x4 per thread, K step 32.
struct Params { n_tokens: u32, k: u32, n: u32, x_stride: u32, y_stride: u32, y_off: u32, tiles_n: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
var<workgroup> xs: array<array<f32, 64>, 32>;   // [k][token]
var<workgroup> ws: array<array<f32, 64>, 32>;   // [k][output]

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let tile = linear_wg(wid, nwg);
    let tn = tile % p.tiles_n;
    let tm = tile / p.tiles_n;
    let t0 = tm * 64u;
    let n0 = tn * 64u;
    let tx = lid % 16u;   // output columns tx*4 .. +4
    let ty = lid / 16u;   // tokens ty*4 .. +4
    // one vec4 of outputs per token row (plain vectors, not arrays: keeps the DX12 compiler fast)
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);

    let k8 = p.k / 8u;
    for (var k0 = 0u; k0 < p.k; k0 += 32u) {
        // X tile: 64 tokens x 32 k = 512 vec4, two per thread
        for (var r = 0u; r < 2u; r++) {
            let e = lid + r * 256u;
            let tok = e / 8u;
            let c = e % 8u;
            var v = vec4<f32>(0.0);
            if (t0 + tok < p.n_tokens) { v = x[((t0 + tok) * p.x_stride + k0) / 4u + c]; }
            xs[c * 4u][tok] = v.x; xs[c * 4u + 1u][tok] = v.y;
            xs[c * 4u + 2u][tok] = v.z; xs[c * 4u + 3u][tok] = v.w;
        }
        // W tile: 64 outputs x 32 k = 256 vec4<u32> (8 bf16 each), one per thread
        {
            let o = lid / 4u;
            let c = lid % 4u;
            var v = vec4<u32>(0u);
            if (n0 + o < p.n) { v = w[(n0 + o) * k8 + k0 / 8u + c]; }
            let kb = c * 8u;
            ws[kb][o] = bf16_lo(v.x); ws[kb + 1u][o] = bf16_hi(v.x);
            ws[kb + 2u][o] = bf16_lo(v.y); ws[kb + 3u][o] = bf16_hi(v.y);
            ws[kb + 4u][o] = bf16_lo(v.z); ws[kb + 5u][o] = bf16_hi(v.z);
            ws[kb + 6u][o] = bf16_lo(v.w); ws[kb + 7u][o] = bf16_hi(v.w);
        }
        workgroupBarrier();
        for (var kk = 0u; kk < 32u; kk++) {
            let a = vec4<f32>(xs[kk][ty * 4u], xs[kk][ty * 4u + 1u], xs[kk][ty * 4u + 2u], xs[kk][ty * 4u + 3u]);
            let b = vec4<f32>(ws[kk][tx * 4u], ws[kk][tx * 4u + 1u], ws[kk][tx * 4u + 2u], ws[kk][tx * 4u + 3u]);
            acc0 += a.x * b;
            acc1 += a.y * b;
            acc2 += a.z * b;
            acc3 += a.w * b;
        }
        workgroupBarrier();
    }
    store_row(t0 + ty * 4u, n0 + tx * 4u, acc0);
    store_row(t0 + ty * 4u + 1u, n0 + tx * 4u, acc1);
    store_row(t0 + ty * 4u + 2u, n0 + tx * 4u, acc2);
    store_row(t0 + ty * 4u + 3u, n0 + tx * 4u, acc3);
}

fn store_row(t: u32, n: u32, v: vec4<f32>) {
    if (t >= p.n_tokens) { return; }
    let base = t * p.y_stride + p.y_off + n;
    if (n < p.n) { y[base] = v.x; }
    if (n + 1u < p.n) { y[base + 1u] = v.y; }
    if (n + 2u < p.n) { y[base + 2u] = v.z; }
    if (n + 3u < p.n) { y[base + 3u] = v.w; }
}

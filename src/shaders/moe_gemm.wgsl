// Grouped expert matmul over tiles of up to 64 assignments that share one expert slot:
//   Y[i, n] = sum_k X[row(i), k] * W_slot[n, k]
// tiles: [slot, i_start, i_count, _]; sorted: assignment ids (token * topk + j) in expert order.
// row(i) = sorted[i] / gather (gather = topk, reading token activations) or i (gather = 0).
@group(0) @binding(0) var<uniform> p: MoeParams;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> pool: array<u32>;
@group(0) @binding(3) var<storage, read> tiles: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> sorted: array<u32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;
var<workgroup> xs: array<array<f32, 64>, 32>;   // [k][assignment]
var<workgroup> ws: array<array<f32, 64>, 32>;   // [k][output]

fn xrow(i: u32) -> u32 {
    if (p.gather > 0u) { return sorted[i] / p.gather; }
    return i;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let id = linear_wg(wid, nwg);
    let tn = id % p.tiles_n;
    let tile = tiles[id / p.tiles_n];
    let base = tile.x * p.slot_words;
    let i0 = tile.y;
    let cnt = tile.z;
    let n0 = tn * 64u;
    let tx = lid % 16u;
    let ty = lid / 16u;
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    let fac = bitcast<f32>(pool[base + p.off_f]);

    for (var k0 = 0u; k0 < p.k; k0 += 32u) {
        for (var r = 0u; r < 2u; r++) {
            let e = lid + r * 256u;
            let a = e / 8u;
            let c = e % 8u;
            var v = vec4<f32>(0.0);
            if (a < cnt) { v = x[(xrow(i0 + a) * p.x_stride + k0) / 4u + c]; }
            xs[c * 4u][a] = v.x; xs[c * 4u + 1u][a] = v.y;
            xs[c * 4u + 2u][a] = v.z; xs[c * 4u + 3u][a] = v.w;
        }
        {
            let o = lid / 4u;
            let c = lid % 4u;
            let row = n0 + o;
            let kb = c * 8u;
            var vals0 = vec4<f32>(0.0);
            var vals1 = vec4<f32>(0.0);
            if (row < p.n) {
                if (p.fmt == 0u) {
                    let wq = pool[base + p.off_q + row * p.qw + k0 / 8u + c];
                    let si = (k0 + kb) / 16u;
                    let sc = fp8_e4m3(byte_of(pool[base + p.off_s + row * p.sw + si / 4u], si % 4u));
                    vals0 = vec4<f32>(fp4_e2m1(wq), fp4_e2m1(wq >> 4u), fp4_e2m1(wq >> 8u), fp4_e2m1(wq >> 12u));
                    vals1 = vec4<f32>(fp4_e2m1(wq >> 16u), fp4_e2m1(wq >> 20u), fp4_e2m1(wq >> 24u), fp4_e2m1(wq >> 28u));
                    vals0 = vals0 * sc * fac;
                    vals1 = vals1 * sc * fac;
                } else {
                    let w0 = pool[base + p.off_q + row * p.qw + k0 / 4u + c * 2u];
                    let w1 = pool[base + p.off_q + row * p.qw + k0 / 4u + c * 2u + 1u];
                    let si = k0 / 32u;
                    let sc = exp2(f32(byte_of(pool[base + p.off_s + row * p.sw + si / 4u], si % 4u)) - 127.0);
                    vals0 = vec4<f32>(fp8_e4m3(byte_of(w0, 0u)), fp8_e4m3(byte_of(w0, 1u)), fp8_e4m3(byte_of(w0, 2u)), fp8_e4m3(byte_of(w0, 3u))) * sc;
                    vals1 = vec4<f32>(fp8_e4m3(byte_of(w1, 0u)), fp8_e4m3(byte_of(w1, 1u)), fp8_e4m3(byte_of(w1, 2u)), fp8_e4m3(byte_of(w1, 3u))) * sc;
                }
            }
            ws[kb][o] = round_bf16(vals0.x); ws[kb + 1u][o] = round_bf16(vals0.y);
            ws[kb + 2u][o] = round_bf16(vals0.z); ws[kb + 3u][o] = round_bf16(vals0.w);
            ws[kb + 4u][o] = round_bf16(vals1.x); ws[kb + 5u][o] = round_bf16(vals1.y);
            ws[kb + 6u][o] = round_bf16(vals1.z); ws[kb + 7u][o] = round_bf16(vals1.w);
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
    store_row(ty * 4u, cnt, i0, n0 + tx * 4u, acc0);
    store_row(ty * 4u + 1u, cnt, i0, n0 + tx * 4u, acc1);
    store_row(ty * 4u + 2u, cnt, i0, n0 + tx * 4u, acc2);
    store_row(ty * 4u + 3u, cnt, i0, n0 + tx * 4u, acc3);
}

fn store_row(a: u32, cnt: u32, i0: u32, n: u32, v: vec4<f32>) {
    if (a >= cnt) { return; }
    let b = (i0 + a) * p.y_stride + n;
    if (n < p.n) { y[b] = v.x; }
    if (n + 1u < p.n) { y[b + 1u] = v.y; }
    if (n + 2u < p.n) { y[b + 2u] = v.z; }
    if (n + 3u < p.n) { y[b + 3u] = v.w; }
}

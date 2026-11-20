// Expert matrix-vector products for a few assignments (decode). One workgroup = 4 output rows
// of one assignment, 64 threads per row:  Y[i, n] = sum_k X[row(i), k] * W_{slot(i)}[n, k]
// items: [slot, x_row, y_row, _] per assignment i; tiles_n = ceil(n / 4).
@group(0) @binding(0) var<uniform> p: MoeParams;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> pool: array<u32>;
@group(0) @binding(3) var<storage, read> items: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let id = linear_wg(wid, nwg);
    let i = id / p.tiles_n;
    let row = (id % p.tiles_n) * 4u + lid / 64u;
    let l = lid % 64u;
    let it = items[min(i, p.n_items - 1u)];
    let base = it.x * p.slot_words;
    let xb = it.y * p.x_stride;
    let fac = bitcast<f32>(pool[base + p.off_f]);
    var acc = 0.0;
    if (row < p.n && i < p.n_items) {
        if (p.fmt == 0u) {
            // one word = 8 values; one scale per 2 words
            for (var wi = l; wi < p.k / 8u; wi += 64u) {
                let wq = pool[base + p.off_q + row * p.qw + wi];
                let si = wi / 2u;
                let sc = fp8_e4m3(byte_of(pool[base + p.off_s + row * p.sw + si / 4u], si % 4u));
                let kb = xb + wi * 8u;
                var s = 0.0;
                for (var j = 0u; j < 8u; j++) {
                    s += x[kb + j] * round_bf16(fp4_e2m1(wq >> (4u * j)) * sc * fac);
                }
                acc += s;
            }
        } else {
            for (var wi = l; wi < p.k / 4u; wi += 64u) {
                let wq = pool[base + p.off_q + row * p.qw + wi];
                let si = wi / 8u;
                let sc = exp2(f32(byte_of(pool[base + p.off_s + row * p.sw + si / 4u], si % 4u)) - 127.0);
                let kb = xb + wi * 4u;
                acc += x[kb] * round_bf16(fp8_e4m3(byte_of(wq, 0u)) * sc)
                     + x[kb + 1u] * round_bf16(fp8_e4m3(byte_of(wq, 1u)) * sc)
                     + x[kb + 2u] * round_bf16(fp8_e4m3(byte_of(wq, 2u)) * sc)
                     + x[kb + 3u] * round_bf16(fp8_e4m3(byte_of(wq, 3u)) * sc);
            }
        }
    }
    red[lid] = acc;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s >>= 1u) {
        if (l < s) { red[lid] += red[lid + s]; }
        workgroupBarrier();
    }
    if (l == 0u && row < p.n && i < p.n_items) { y[it.z * p.y_stride + row] = red[lid]; }
}

// Block attention residual: for each token, softmax-attend with learned queries over the
// unweighted-RMS-normalised stack [blocks[slot_0..nb-1], cur] and output the mix of normed values.
//   out[q][t][:] = sum_i softmax_i(<query_q, n_i> / sqrt(D)) * n_i,   n_i = rmsnorm(v_i)
struct Params { t: u32, d: u32, nb: u32, nq: u32, slots: u32, q_off: u32, slot_stride: u32, out_stride: u32, eps: f32, cur_off: u32, row0: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> blocks: array<f32>;
@group(0) @binding(2) var<storage, read> cur: array<f32>;
@group(0) @binding(3) var<storage, read> queries: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
var<workgroup> red: array<f32, 256>;
var<workgroup> inv: array<f32, 8>;
var<workgroup> sc: array<f32, 32>;

fn reduce(lid: u32, v: f32) -> f32 {
    red[lid] = v;
    workgroupBarrier();
    for (var k = 128u; k > 0u; k >>= 1u) {
        if (lid < k) { red[lid] += red[lid + k]; }
        workgroupBarrier();
    }
    let r = red[0];
    workgroupBarrier();
    return r;
}

fn base_of(i: u32, t: u32) -> u32 {
    let slot = (p.slots >> (4u * i)) & 0xfu;
    let r = t + p.row0;
    return select(slot * p.slot_stride + r * p.d, p.cur_off + r * p.d, i == p.nb);
}

fn val(i: u32, idx: u32) -> f32 {
    if (i == p.nb) { return cur[idx]; }
    return blocks[idx];
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let t = linear_wg(wid, nwg);
    if (t >= p.t) { return; }
    let n = p.nb + 1u;
    for (var i = 0u; i < n; i++) {
        let b = base_of(i, t);
        var s = 0.0;
        for (var d = lid; d < p.d; d += 256u) { let v = val(i, b + d); s += v * v; }
        let tot = reduce(lid, s);
        if (lid == 0u) { inv[i] = inverseSqrt(tot / f32(p.d) + p.eps); }
    }
    workgroupBarrier();
    let scale = inverseSqrt(f32(p.d));
    for (var q = 0u; q < p.nq; q++) {
        let qb = p.q_off + q * p.d;
        for (var i = 0u; i < n; i++) {
            let b = base_of(i, t);
            var s = 0.0;
            for (var d = lid; d < p.d; d += 256u) { s += queries[qb + d] * val(i, b + d); }
            let tot = reduce(lid, s);
            if (lid == 0u) { sc[q * 8u + i] = tot * inv[i] * scale; }
        }
    }
    workgroupBarrier();
    if (lid == 0u) {
        for (var q = 0u; q < p.nq; q++) {
            var m = NEG_BIG;
            for (var i = 0u; i < n; i++) { m = max(m, sc[q * 8u + i]); }
            var z = 0.0;
            for (var i = 0u; i < n; i++) { let e = exp(sc[q * 8u + i] - m); sc[q * 8u + i] = e; z += e; }
            for (var i = 0u; i < n; i++) { sc[q * 8u + i] = sc[q * 8u + i] / z * inv[i]; }
        }
    }
    workgroupBarrier();
    for (var q = 0u; q < p.nq; q++) {
        let ob = q * p.out_stride + t * p.d;
        for (var d = lid; d < p.d; d += 256u) {
            var acc = 0.0;
            for (var i = 0u; i < n; i++) { acc += sc[q * 8u + i] * val(i, base_of(i, t) + d); }
            out[ob + d] = acc;
        }
    }
}

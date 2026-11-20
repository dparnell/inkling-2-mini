// Causal multi-query attention with per-token sink logits, hidden-state-conditioned relative
// position bias and optional sliding window. 8 queries x 1 head per workgroup, 32-key tiles.
//   logit(q, k) = (q . k) * scale + bias(q, dist) ;  bias = rs[q, h, :] . proj[:, dist] for dist < rel_extent
//   softmax over keys plus one sink logit (zero value) per (token, head).
// q:     [t][nh][128] f32 (row stride q_stride)
// cache: [nkv][cap][64] packed f16 (slot = pos % cap)
// rs:    [t][nh][16] f32 (row stride rs_stride) ; proj: [16][rel_extent] f32 ; sinks: [t][nh] (row stride s_stride)
// out:   [t][nh][128]
struct Params { t: u32, nh: u32, nkv: u32, pos0: u32, cap: u32, window: u32, rel_extent: u32, scale: f32,
                q_stride: u32, rs_stride: u32, s_stride: u32, qblocks: u32, splits: u32, split_len: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> kc: array<u32>;
@group(0) @binding(3) var<storage, read> vc: array<u32>;
@group(0) @binding(4) var<storage, read> rs: array<f32>;
@group(0) @binding(5) var<storage, read> proj: array<f32>;
@group(0) @binding(6) var<storage, read> sinks: array<f32>;
@group(0) @binding(7) var<storage, read_write> out: array<f32>;
// split-K partials: [(h * t + q) * splits + split][130] = (m, l, acc[128])
@group(0) @binding(8) var<storage, read_write> part: array<f32>;

const QT: u32 = 8u;
const KB: u32 = 32u;
var<workgroup> qs: array<f32, 1024>;      // [QT][128]
var<workgroup> ks: array<u32, 2048>;      // [KB][64]
var<workgroup> vs: array<u32, 2048>;      // [KB][64]
var<workgroup> rss: array<f32, 128>;      // [QT][16]
var<workgroup> sc: array<f32, 256>;       // [QT][KB]

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let id0 = linear_wg(wid, nwg);
    let split = id0 % p.splits;
    let id = id0 / p.splits;
    let h = id / p.qblocks;
    let qb = id % p.qblocks;
    let q0 = qb * QT;
    let kvh = h / (p.nh / p.nkv);
    let nq = min(QT, p.t - q0);

    // load queries (scaled) and relative states
    for (var i = lid; i < QT * 128u; i += 128u) {
        let qi = i / 128u;
        let d = i % 128u;
        var v = 0.0;
        if (qi < nq) { v = q[(q0 + qi) * p.q_stride + h * 128u + d] * p.scale; }
        qs[i] = v;
    }
    {
        let qi = lid / 16u;
        let r = lid % 16u;
        var v = 0.0;
        if (qi < nq) { v = rs[(q0 + qi) * p.rs_stride + h * 16u + r]; }
        rss[lid] = v;
    }

    // this thread accumulates query `my_q`, dims my_d .. my_d+8
    let my_q = lid / 16u;
    let my_d = (lid % 16u) * 8u;
    let qvalid = my_q < nq;
    let my_pos = p.pos0 + q0 + min(my_q, nq - 1u);
    var m = 0.0;
    if (qvalid) { m = sinks[(q0 + my_q) * p.s_stride + h]; }
    var l = 1.0;
    if (p.splits > 1u) { m = NEG_BIG; l = 0.0; }
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);

    let first_pos = p.pos0 + q0;
    let last_pos = p.pos0 + q0 + nq - 1u;
    var k_start = 0u;
    if (p.window > 0u && first_pos + 1u > p.window) { k_start = first_pos + 1u - p.window; }
    let k_end = last_pos + 1u;
    k_start = k_start / KB * KB;
    var k_stop = k_end;
    if (p.splits > 1u) {
        k_start = k_start + split * p.split_len;
        k_stop = min(k_end, k_start + p.split_len);
    }

    for (var k0 = k_start; k0 < k_stop; k0 += KB) {
        workgroupBarrier();
        for (var i = lid; i < KB * 64u; i += 128u) {
            let kj = i / 64u;
            let c = i % 64u;
            let pos = k0 + kj;
            var kv = 0u;
            var vv = 0u;
            if (pos < k_stop) {
                let base = (kvh * p.cap + pos % p.cap) * 64u + c;
                kv = kc[base];
                vv = vc[base];
            }
            ks[i] = kv;
            vs[i] = vv;
        }
        workgroupBarrier();
        // scores: 256 (q, k) pairs, 2 per thread
        for (var e = lid; e < QT * KB; e += 128u) {
            let qi = e / KB;
            let kj = e % KB;
            let qpos = p.pos0 + q0 + qi;
            let kpos = k0 + kj;
            var s = 0.0;
            for (var c = 0u; c < 64u; c++) {
                let kk = unpack2x16float(ks[kj * 64u + c]);
                s += qs[qi * 128u + c * 2u] * kk.x + qs[qi * 128u + c * 2u + 1u] * kk.y;
            }
            let dist = qpos - min(kpos, qpos);
            var bias = 0.0;
            if (dist < p.rel_extent) {
                for (var r = 0u; r < 16u; r++) { bias += rss[qi * 16u + r] * proj[r * p.rel_extent + dist]; }
            }
            let valid = qi < nq && kpos < k_stop && kpos <= qpos && (p.window == 0u || qpos - kpos < p.window);
            sc[e] = select(NEG_BIG, s + bias, valid);
        }
        workgroupBarrier();
        // online softmax update for my query
        var tmax = m;
        for (var j = 0u; j < KB; j++) { tmax = max(tmax, sc[my_q * KB + j]); }
        let corr = exp(m - tmax);
        l = l * corr;
        acc0 = acc0 * corr;
        acc1 = acc1 * corr;
        for (var j = 0u; j < KB; j++) {
            let s = sc[my_q * KB + j];
            let pj = select(exp(s - tmax), 0.0, s <= -1.0e38);
            l += pj;
            let vb = j * 64u + my_d / 2u;
            let v0 = unpack2x16float(vs[vb]);
            let v1 = unpack2x16float(vs[vb + 1u]);
            let v2 = unpack2x16float(vs[vb + 2u]);
            let v3 = unpack2x16float(vs[vb + 3u]);
            acc0 += pj * vec4<f32>(v0.x, v0.y, v1.x, v1.y);
            acc1 += pj * vec4<f32>(v2.x, v2.y, v3.x, v3.y);
        }
        m = tmax;
    }
    if (qvalid && p.splits > 1u) {
        let pb = ((h * p.t + q0 + my_q) * p.splits + split) * 130u;
        if (my_d == 0u) { part[pb] = m; part[pb + 1u] = l; }
        let ab = pb + 2u + my_d;
        part[ab] = acc0.x; part[ab + 1u] = acc0.y; part[ab + 2u] = acc0.z; part[ab + 3u] = acc0.w;
        part[ab + 4u] = acc1.x; part[ab + 5u] = acc1.y; part[ab + 6u] = acc1.z; part[ab + 7u] = acc1.w;
    } else if (qvalid) {
        let ob = (q0 + my_q) * p.nh * 128u + h * 128u + my_d;
        let inv = 1.0 / l;
        out[ob] = acc0.x * inv; out[ob + 1u] = acc0.y * inv; out[ob + 2u] = acc0.z * inv; out[ob + 3u] = acc0.w * inv;
        out[ob + 4u] = acc1.x * inv; out[ob + 5u] = acc1.y * inv; out[ob + 6u] = acc1.z * inv; out[ob + 7u] = acc1.w * inv;
    }
}

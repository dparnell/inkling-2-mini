// Bidirectional attention for the vision tower (head_dim 64), f32 q/k/v rows [n][heads * 64].
// 8 queries x 1 head per workgroup, 32-key tiles, online softmax.
struct Params { n: u32, heads: u32, scale: f32, qblocks: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

const QT: u32 = 8u;
const KB: u32 = 32u;
const HD: u32 = 64u;
var<workgroup> qs: array<f32, 512>;   // [QT][64]
var<workgroup> ks: array<f32, 2048>;  // [KB][64]
var<workgroup> vs: array<f32, 2048>;  // [KB][64]
var<workgroup> sc: array<f32, 256>;   // [QT][KB]

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let id = linear_wg(wid, nwg);
    let h = id / p.qblocks;
    let q0 = (id % p.qblocks) * QT;
    let w = p.heads * HD;
    let nq = min(QT, p.n - q0);
    for (var i = lid; i < QT * HD; i += 128u) {
        let qi = i / HD;
        var x = 0.0;
        if (qi < nq) { x = q[(q0 + qi) * w + h * HD + i % HD] * p.scale; }
        qs[i] = x;
    }
    // thread: query lid / 16, dims (lid % 16) * 4 .. +4
    let my_q = lid / 16u;
    let my_d = (lid % 16u) * 4u;
    var m = NEG_BIG;
    var l = 0.0;
    var acc = vec4<f32>(0.0);
    for (var k0 = 0u; k0 < p.n; k0 += KB) {
        workgroupBarrier();
        for (var i = lid; i < KB * HD; i += 128u) {
            let kj = i / HD;
            var kv = 0.0;
            var vv = 0.0;
            if (k0 + kj < p.n) {
                let b = (k0 + kj) * w + h * HD + i % HD;
                kv = k[b];
                vv = v[b];
            }
            ks[i] = kv;
            vs[i] = vv;
        }
        workgroupBarrier();
        for (var e = lid; e < QT * KB; e += 128u) {
            let qi = e / KB;
            let kj = e % KB;
            var s = 0.0;
            for (var c = 0u; c < HD; c++) { s += qs[qi * HD + c] * ks[kj * HD + c]; }
            sc[e] = select(NEG_BIG, s, k0 + kj < p.n);
        }
        workgroupBarrier();
        var tmax = m;
        for (var j = 0u; j < KB; j++) { tmax = max(tmax, sc[my_q * KB + j]); }
        let corr = exp(m - tmax);
        l = l * corr;
        acc = acc * corr;
        for (var j = 0u; j < KB; j++) {
            let s = sc[my_q * KB + j];
            let pj = select(exp(s - tmax), 0.0, s <= -1.0e38);
            l += pj;
            acc += pj * vec4<f32>(vs[j * HD + my_d], vs[j * HD + my_d + 1u], vs[j * HD + my_d + 2u], vs[j * HD + my_d + 3u]);
        }
        m = tmax;
    }
    if (my_q < nq) {
        let ob = (q0 + my_q) * w + h * HD + my_d;
        let inv = 1.0 / l;
        out[ob] = acc.x * inv; out[ob + 1u] = acc.y * inv; out[ob + 2u] = acc.z * inv; out[ob + 3u] = acc.w * inv;
    }
}

// Per (token, head) in place:  v = v * scale[h, :] (if has_scale);  v *= min(thr / (|v| + 1e-6), 1);
// then (log_scale) v *= 1 + alpha * log(max((pos + 1) / n_floor, 1)).
// Rows are `stride` floats apart; heads `dh` floats apart starting at `off`.
struct Params { t: u32, h: u32, dh: u32, stride: u32, off: u32, has_scale: u32, thr: f32, log_scale: u32, alpha: f32, n_floor: f32, pos0: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(2) var<storage, read_write> x: array<f32>;
var<workgroup> red: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let id = linear_wg(wid, nwg);
    if (id >= p.t * p.h) { return; }
    let t = id / p.h;
    let h = id % p.h;
    let b = t * p.stride + p.off + h * p.dh;
    var s = 0.0;
    for (var d = lid; d < p.dh; d += 64u) {
        var v = x[b + d];
        if (p.has_scale != 0u) { v = v * scale[h * p.dh + d]; }
        s += v * v;
    }
    red[lid] = s;
    workgroupBarrier();
    for (var k = 32u; k > 0u; k >>= 1u) {
        if (lid < k) { red[lid] += red[lid + k]; }
        workgroupBarrier();
    }
    var f = min(p.thr / (sqrt(red[0]) + 1e-6), 1.0);
    if (p.log_scale != 0u) {
        let pos = f32(p.pos0 + t);
        f = f * (1.0 + p.alpha * log(max((pos + 1.0) / p.n_floor, 1.0)));
    }
    for (var d = lid; d < p.dh; d += 64u) {
        var v = x[b + d];
        if (p.has_scale != 0u) { v = v * scale[h * p.dh + d]; }
        x[b + d] = v * f;
    }
}

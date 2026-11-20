// Sigmoid top-k router: scores = sigmoid(logits); pick top-k of (scores + bias);
// weights = scores[sel] / sum(scores[sel]) * route_scale.   One workgroup (128 threads) per token.
struct Params { t: u32, ne: u32, k: u32, route_scale: f32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> ids: array<u32>;
@group(0) @binding(4) var<storage, read_write> wts: array<f32>;
var<workgroup> key: array<f32, 128>;
var<workgroup> rv: array<f32, 128>;
var<workgroup> ri: array<u32, 128>;
var<workgroup> sel_s: array<f32, 16>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let t = linear_wg(wid, nwg);
    if (t >= p.t) { return; }
    var score = 0.0;
    var kv = NEG_BIG;
    if (lid < p.ne) {
        score = sigmoid(logits[t * p.ne + lid]);
        kv = score + bias[lid];
    }
    key[lid] = kv;
    workgroupBarrier();
    for (var j = 0u; j < p.k; j++) {
        rv[lid] = key[lid];
        ri[lid] = lid;
        workgroupBarrier();
        for (var s = 64u; s > 0u; s >>= 1u) {
            if (lid < s) {
                let a = rv[lid];
                let b = rv[lid + s];
                // ties -> lower index
                if (b > a || (b == a && ri[lid + s] < ri[lid])) { rv[lid] = b; ri[lid] = ri[lid + s]; }
            }
            workgroupBarrier();
        }
        let win = ri[0];
        workgroupBarrier();
        if (lid == win) {
            key[lid] = NEG_BIG;
            sel_s[j] = score;
            ids[t * p.k + j] = lid;
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        var sum = 0.0;
        for (var j = 0u; j < p.k; j++) { sum += sel_s[j]; }
        for (var j = 0u; j < p.k; j++) { wts[t * p.k + j] = sel_s[j] / (sum + 1e-20) * p.route_scale; }
    }
}

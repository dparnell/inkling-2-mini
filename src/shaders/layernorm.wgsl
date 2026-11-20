// y[r, :] = (x[r, :] - mean) * rsqrt(var + eps) * w + b      (one workgroup per row)
struct Params { rows: u32, dim: u32, x_stride: u32, y_stride: u32, eps: f32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let r = linear_wg(wid, nwg);
    if (r >= p.rows) { return; }
    let xb = r * p.x_stride;
    var s = 0.0;
    for (var i = lid; i < p.dim; i += 256u) { s += x[xb + i]; }
    red[lid] = s;
    workgroupBarrier();
    for (var k = 128u; k > 0u; k >>= 1u) {
        if (lid < k) { red[lid] += red[lid + k]; }
        workgroupBarrier();
    }
    let mean = red[0] / f32(p.dim);
    workgroupBarrier();
    var v = 0.0;
    for (var i = lid; i < p.dim; i += 256u) { let d = x[xb + i] - mean; v += d * d; }
    red[lid] = v;
    workgroupBarrier();
    for (var k = 128u; k > 0u; k >>= 1u) {
        if (lid < k) { red[lid] += red[lid + k]; }
        workgroupBarrier();
    }
    let inv = inverseSqrt(red[0] / f32(p.dim) + p.eps);
    let yb = r * p.y_stride;
    for (var i = lid; i < p.dim; i += 256u) { y[yb + i] = (x[xb + i] - mean) * inv * w[i] + b[i]; }
}

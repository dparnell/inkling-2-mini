// Element-wise ops over n = rows * d values (float offsets a_off, b_off, y_off):
//   mode 0: y = a + b
//   mode 1: y = a * sigmoid(b)
//   mode 2: message gate  y = (1 - tanh(c[t, d])) * a + tanh(c[t, D + d]) * b     (c rows are 2D wide)
//   mode 3: y = a
//   mode 4: y = a + b * c[i % D]      (layer-scaled residual)
struct Params { n: u32, mode: u32, d: u32, a_off: u32, b_off: u32, y_off: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read> c: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let i = linear_wg(wid, nwg) * 256u + lid;
    if (i >= p.n) { return; }
    let av = a[p.a_off + i];
    var r = av;
    if (p.mode == 0u) {
        r = av + b[p.b_off + i];
    } else if (p.mode == 1u) {
        r = av * sigmoid(b[p.b_off + i]);
    } else if (p.mode == 2u) {
        let t = i / p.d;
        let d = i % p.d;
        let f = tanh_s(c[t * 2u * p.d + d]);
        let g = tanh_s(c[t * 2u * p.d + p.d + d]);
        r = (1.0 - f) * av + g * b[p.b_off + i];
    } else if (p.mode == 4u) {
        r = av + b[p.b_off + i] * c[i % p.d];
    }
    y[p.y_off + i] = r;
}

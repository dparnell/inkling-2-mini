// y[r, off + c] = act(y[r, off + c] + b[c])     act: 0 none, 1 GELU (erf), 2 SiLU
struct Params { rows: u32, cols: u32, stride: u32, act: u32, off: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

// Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7)
fn erf_approx(x: f32) -> f32 {
    let s = sign(x);
    let a = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    return s * (1.0 - poly * exp(-a * a));
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let i = linear_wg(wid, nwg) * 256u + lid;
    let r = i / p.cols;
    let c = i % p.cols;
    if (r >= p.rows) { return; }
    let idx = r * p.stride + p.off + c;
    var v = y[idx] + b[c];
    if (p.act == 1u) {
        v = 0.5 * v * (1.0 + erf_approx(v * 0.70710678118654752));
    } else if (p.act == 2u) {
        v = v / (1.0 + exp(-v));
    }
    y[idx] = v;
}

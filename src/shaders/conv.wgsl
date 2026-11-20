// Causal depthwise short convolution with residual and output scale, over a chunk of `t` rows
// starting at absolute position pos0. Earlier rows come from `hist`, a ring of the last 4 input
// rows keyed by position % 4.
//   static (dynamic == 0): y = (sum_j w[c*4 + j] * x[t-3+j] + x[t]) / sqrt(5)         (conv1d layout)
//   dynamic (dynamic == 1): y = (sum_lag (sw[lag*C + c] + dw[t*dstride + head*4 + lag]) * x[t-lag] + x[t]) / 3
//                            head = c / head_dim
struct Params { t: u32, c: u32, pos0: u32, dynamic: u32, head_dim: u32, dstride: u32, x_stride: u32, x_off: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> hist: array<f32>;
@group(0) @binding(3) var<storage, read> w: array<f32>;
@group(0) @binding(4) var<storage, read> dw: array<f32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;

fn src(t: i32, c: u32) -> f32 {
    if (t >= 0) { return x[p.x_off + u32(t) * p.x_stride + c]; }
    let pos = i32(p.pos0) + t;
    if (pos < 0) { return 0.0; }
    return hist[(u32(pos) % 4u) * p.c + c];
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let i = linear_wg(wid, nwg) * 256u + lid;
    if (i >= p.t * p.c) { return; }
    let t = i / p.c;
    let c = i % p.c;
    let ti = i32(t);
    let cur = src(ti, c);
    var acc = 0.0;
    if (p.dynamic == 0u) {
        for (var j = 0u; j < 4u; j++) { acc += w[c * 4u + j] * src(ti - 3 + i32(j), c); }
        y[t * p.c + c] = (acc + cur) * inverseSqrt(5.0);
    } else {
        let hd = c / p.head_dim;
        for (var lag = 0u; lag < 4u; lag++) {
            let wt = w[lag * p.c + c] + dw[t * p.dstride + hd * 4u + lag];
            acc += wt * src(ti - i32(lag), c);
        }
        y[t * p.c + c] = (acc + cur) / 3.0;
    }
}

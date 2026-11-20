// Store the last (up to 4) input rows of a chunk into the conv history ring (slot = pos % 4).
struct Params { t: u32, c: u32, pos0: u32, x_stride: u32, x_off: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> hist: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let i = linear_wg(wid, nwg) * 256u + lid;
    let n = min(p.t, 4u);
    if (i >= n * p.c) { return; }
    let r = p.t - n + i / p.c;
    let c = i % p.c;
    hist[((p.pos0 + r) % 4u) * p.c + c] = x[p.x_off + r * p.x_stride + c];
}

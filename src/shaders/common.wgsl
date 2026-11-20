// Shared helpers (prepended to every kernel).
fn bf16_lo(w: u32) -> f32 { return bitcast<f32>(w << 16u); }
fn bf16_hi(w: u32) -> f32 { return bitcast<f32>(w & 0xffff0000u); }
fn linear_wg(wid: vec3<u32>, nwg: vec3<u32>) -> u32 { return wid.x + wid.y * nwg.x; }
const NEG_BIG: f32 = -3.0e38;
const POS_BIG: f32 = 3.0e38;
// Round an f32 to the nearest bf16 (ties to even), returned as f32.
fn round_bf16(x: f32) -> f32 {
    let b = bitcast<u32>(x);
    let r = b + 0x7fffu + ((b >> 16u) & 1u);
    return bitcast<f32>(r & 0xffff0000u);
}
// FP8 E4M3 (fn) byte -> f32.
fn fp8_e4m3(b: u32) -> f32 {
    let s = select(1.0, -1.0, (b & 0x80u) != 0u);
    let e = (b >> 3u) & 0xfu;
    let m = f32(b & 7u);
    let normal = (1.0 + m * 0.125) * exp2(f32(e) - 7.0);
    let sub = m * exp2(-9.0);
    return s * select(normal, sub, e == 0u);
}
// FP4 E2M1 nibble -> f32.
fn fp4_e2m1(n: u32) -> f32 {
    let v = n & 7u;
    let e = v >> 1u;
    let m = f32(v & 1u);
    let mag = select(m * 0.5, (1.0 + m * 0.5) * exp2(f32(e) - 1.0), e > 0u);
    return select(mag, -mag, (n & 8u) != 0u);
}
fn byte_of(w: u32, i: u32) -> u32 { return (w >> (8u * i)) & 0xffu; }
fn sigmoid(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }
fn tanh_s(x: f32) -> f32 { let c = clamp(x, -15.0, 15.0); let e = exp(2.0 * c); return (e - 1.0) / (e + 1.0); }

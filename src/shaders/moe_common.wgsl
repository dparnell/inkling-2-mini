// Quantized expert weights in a slot pool (u32 words). Slot layout: [q13 | s13 | q2 | s2 | f13 f2]
// fmt 0 = NVFP4: two E2M1 per byte (even element in low nibble), FP8 E4M3 scale per 16, factor per matrix
// fmt 1 = MXFP8: FP8 E4M3 values, E8M0 scale 2^(s-127) per 32
struct MoeParams {
    fmt: u32, which: u32, k: u32, n: u32,
    x_stride: u32, y_stride: u32, slot_words: u32, gather: u32,
    off_s: u32, off_q: u32, qw: u32, sw: u32,
    off_f: u32, tiles_n: u32, n_items: u32, pad: u32,
}

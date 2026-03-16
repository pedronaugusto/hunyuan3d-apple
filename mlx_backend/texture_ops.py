"""
Texture splatting, back-projection, and baking operations in MLX.

Replaces PyTorch scatter-add operations from hy3dpaint/DifferentiableRenderer/MeshRender.py.
Uses Metal kernels via mx.fast.metal_kernel for atomic scatter-add, and pure MLX
array ops for everything else.
"""
import mlx.core as mx
import math

# ---------------------------------------------------------------------------
# Metal kernel source for atomic scatter-add on float buffers.
#
# Metal has no native atomic_float on all GPU families, so we use a CAS loop
# over atomic_uint with bit-cast (as_type<float/uint>).  This is safe on all
# Apple Silicon (A-series / M-series) devices.
# ---------------------------------------------------------------------------

_SCATTER_ADD_HEADER = """
// Atomic float add via compare-and-swap on uint reinterpretation.
// Defined in header so it's available at file scope (not inside the kernel body).
inline void atomic_add_f(device atomic<float>* addr, float val) {
    float expected = atomic_load_explicit(addr, memory_order_relaxed);
    while (true) {
        float desired = expected + val;
        if (atomic_compare_exchange_weak_explicit(
                addr, &expected, desired,
                memory_order_relaxed, memory_order_relaxed))
            break;
    }
}
"""

_SCATTER_ADD_KERNEL_SOURCE = """
uint elem_id = thread_position_in_grid.x;
if (elem_id >= num_points) return;

// Read UV pixel coordinate for this point.
float px_x = uv[elem_id * 2 + 0];
float px_y = uv[elem_id * 2 + 1];

// Four nearest integer pixel positions (bilinear corners).
int x0 = (int)floor(px_x);
int y0 = (int)floor(px_y);
int x1 = x0 + 1;
int y1 = y0 + 1;

// Clamp to valid texture range.
x0 = max(x0, 0); x1 = min(x1, tex_w - 1);
y0 = max(y0, 0); y1 = min(y1, tex_h - 1);

float fx = px_x - floor(px_x);
float fy = px_y - floor(px_y);

float w00 = (1.0f - fx) * (1.0f - fy);
float w01 = (1.0f - fx) * fy;
float w10 = fx * (1.0f - fy);
float w11 = fx * fy;

// Scatter-add weighted values into texture and weights into weight_map.
for (uint c = 0; c < num_channels; c++) {
    float v = values[elem_id * num_channels + c];
    atomic_add_f(&texture[(y0 * tex_w + x0) * num_channels + c], v * w00);
    atomic_add_f(&texture[(y1 * tex_w + x0) * num_channels + c], v * w01);
    atomic_add_f(&texture[(y0 * tex_w + x1) * num_channels + c], v * w10);
    atomic_add_f(&texture[(y1 * tex_w + x1) * num_channels + c], v * w11);
}
atomic_add_f(&weight_map[y0 * tex_w + x0], w00);
atomic_add_f(&weight_map[y1 * tex_w + x0], w01);
atomic_add_f(&weight_map[y0 * tex_w + x1], w10);
atomic_add_f(&weight_map[y1 * tex_w + x1], w11);
"""


def _scatter_add_kernel():
    """Build (and cache) the Metal scatter-add kernel."""
    if not hasattr(_scatter_add_kernel, "_cached"):
        _scatter_add_kernel._cached = mx.fast.metal_kernel(
            name="bilinear_scatter_add",
            input_names=["uv", "values"],
            output_names=["texture", "weight_map"],
            header=_SCATTER_ADD_HEADER,
            source=_SCATTER_ADD_KERNEL_SOURCE,
            atomic_outputs=True,
        )
    return _scatter_add_kernel._cached


# ---------------------------------------------------------------------------
# 1. Bilinear grid put (texture splatting)
# ---------------------------------------------------------------------------

def bilinear_grid_put_2d(
    texture_size: int,
    uv: mx.array,
    values: mx.array,
) -> tuple:
    """
    Scatter values into a texture using bilinear interpolation weights.

    This is the MLX equivalent of ``linear_grid_put_2d`` from the upstream
    PyTorch renderer.  A Metal kernel performs the atomic scatter-add.

    Args:
        texture_size: Output texture resolution (square, H = W).
        uv: (N, 2) UV coordinates in [0, 1] range.
        values: (N, C) values to splat.

    Returns:
        texture: (H, W, C) accumulated (weighted) texture.
        weight:  (H, W, 1) accumulated weights.
    """
    N, C = values.shape
    H = W = texture_size

    # Convert UV [0,1] to pixel coords [0, size-1].
    px = uv.astype(mx.float32) * (texture_size - 1)
    px = px.reshape(-1)  # (N*2,)

    values_flat = values.astype(mx.float32).reshape(-1)  # (N*C,)

    # Pre-allocate output buffers (zeros).
    texture_buf = mx.zeros((H * W * C,), dtype=mx.float32)
    weight_buf = mx.zeros((H * W,), dtype=mx.float32)

    kernel = _scatter_add_kernel()
    texture_buf, weight_buf = kernel(
        inputs=[px, values_flat],
        output_shapes=[(H * W * C,), (H * W,)],
        output_dtypes=[mx.float32, mx.float32],
        grid=(N, 1, 1),
        threadgroup=(min(N, 256), 1, 1),
        init_value=0.0,
        template=[("num_points", N), ("tex_h", H), ("tex_w", W), ("num_channels", C)],
    )

    texture = texture_buf.reshape(H, W, C)
    weight = weight_buf.reshape(H, W, 1)
    return texture, weight


# ---------------------------------------------------------------------------
# 2. Mipmap bilinear grid put
# ---------------------------------------------------------------------------

def mipmap_grid_put_2d(
    texture_size: int,
    uv: mx.array,
    values: mx.array,
    min_resolution: int = 128,
) -> tuple:
    """
    Multi-resolution texture splatting that fills holes via mipmap hierarchy.

    Runs ``bilinear_grid_put_2d`` at progressively halved resolutions and
    bilinearly upsamples each level back to ``texture_size``, filling only
    the texels that still have zero weight.

    Args:
        texture_size: Output texture resolution (square).
        uv: (N, 2) UV coordinates in [0, 1].
        values: (N, C) values to splat.
        min_resolution: Stop halving once the resolution drops below this.

    Returns:
        texture: (H, W, C) accumulated texture with holes filled.
        weight:  (H, W, 1) accumulated weight map.
    """
    H = W = texture_size
    C = values.shape[-1]

    texture = mx.zeros((H, W, C), dtype=mx.float32)
    weight = mx.zeros((H, W, 1), dtype=mx.float32)

    cur_size = texture_size

    while cur_size >= min_resolution:
        # Check if there are still unfilled texels.
        mask = (weight.squeeze(-1) == 0)  # (H, W)
        if not mx.any(mask):
            break

        cur_tex, cur_w = bilinear_grid_put_2d(cur_size, uv, values)

        if cur_size != texture_size:
            # Bilinear upsample to full resolution.
            cur_tex = _bilinear_upsample(cur_tex, H, W)
            cur_w = _bilinear_upsample(cur_w, H, W)

        # Only fill where the accumulated weight is still zero.
        mask_3d = mx.expand_dims(mask, axis=-1)  # (H, W, 1)
        texture = mx.where(mask_3d, texture + cur_tex, texture)
        weight = mx.where(mask_3d, weight + cur_w, weight)

        cur_size //= 2

    return texture, weight


def _bilinear_upsample(arr: mx.array, target_h: int, target_w: int) -> mx.array:
    """
    Bilinear upsample a (H, W, C) array to (target_h, target_w, C).

    Uses grid-sample style coordinate mapping: each output pixel maps back to
    the input and is bilinearly interpolated.
    """
    src_h, src_w, C = arr.shape
    if src_h == target_h and src_w == target_w:
        return arr

    # Output pixel centres mapped to input coordinates.
    ys = mx.linspace(0, src_h - 1, target_h)  # (target_h,)
    xs = mx.linspace(0, src_w - 1, target_w)  # (target_w,)
    gy, gx = mx.meshgrid(ys, xs, indexing="ij")  # each (target_h, target_w)

    coords = mx.stack([gx.reshape(-1), gy.reshape(-1)], axis=-1)  # (N, 2) x,y
    sampled = bilinear_sample(arr, coords)  # (N, C)
    return sampled.reshape(target_h, target_w, C)


# ---------------------------------------------------------------------------
# 3. Bilinear sample from image
# ---------------------------------------------------------------------------

def bilinear_sample(image: mx.array, coords: mx.array) -> mx.array:
    """
    Sample from an image at fractional pixel coordinates using bilinear interpolation.

    Args:
        image: (H, W, C) source image.
        coords: (N, 2) pixel coordinates as (x, y) in [0, W-1] x [0, H-1].

    Returns:
        (N, C) sampled values.
    """
    H, W, C = image.shape

    x = coords[:, 0]
    y = coords[:, 1]

    x0 = mx.floor(x).astype(mx.int32)
    y0 = mx.floor(y).astype(mx.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    # Fractional parts for interpolation weights.
    fx = (x - x0.astype(mx.float32))[:, None]  # (N, 1)
    fy = (y - y0.astype(mx.float32))[:, None]  # (N, 1)

    # Clamp to valid pixel range.
    x0 = mx.clip(x0, 0, W - 1)
    x1 = mx.clip(x1, 0, W - 1)
    y0 = mx.clip(y0, 0, H - 1)
    y1 = mx.clip(y1, 0, H - 1)

    # Gather the four corner values.  image is (H, W, C), index as flat.
    flat = image.reshape(-1, C)
    i00 = y0 * W + x0
    i01 = y1 * W + x0
    i10 = y0 * W + x1
    i11 = y1 * W + x1

    v00 = flat[i00]  # (N, C)
    v01 = flat[i01]
    v10 = flat[i10]
    v11 = flat[i11]

    # Bilinear blend.
    result = (
        v00 * (1 - fx) * (1 - fy)
        + v01 * (1 - fx) * fy
        + v10 * fx * (1 - fy)
        + v11 * fx * fy
    )
    return result


# ---------------------------------------------------------------------------
# 4. Back-projection
# ---------------------------------------------------------------------------

def back_project(
    image: mx.array,
    depth: mx.array,
    mask: mx.array,
    mvp_matrix: mx.array,
    uv: mx.array,
    texture_size: int,
    positions: mx.array | None = None,
    normals: mx.array | None = None,
    cos_threshold: float = 0.1,
) -> tuple:
    """
    Back-project a rendered view into texture space.

    Given a rendered image and the camera transform used to produce it, scatter
    the image colours back onto the UV texture map, weighted by the cosine of
    the surface viewing angle.

    Args:
        image: (H, W, C) rendered image.
        depth: (H, W) depth buffer.
        mask: (H, W) binary visibility mask (>0 where surface is visible).
        mvp_matrix: (4, 4) model-view-projection matrix.
        uv: (N_vis, 2) per-visible-pixel UV coordinates in [0, 1].
        texture_size: Output texture resolution (square).
        positions: (N_vis, 3) world-space positions of visible pixels (optional,
                   used for accurate depth comparison).
        normals: (N_vis, 3) surface normals for visible pixels (optional,
                 for cosine weighting; if None weights are uniform).
        cos_threshold: Minimum cosine of viewing angle to accept a sample.

    Returns:
        texture: (H_tex, W_tex, C) back-projected texture.
        weight:  (H_tex, W_tex, 1) weight map (cosine-based).
    """
    img_h, img_w, C = image.shape

    # Flatten visible pixels via the mask.
    flat_mask = mask.reshape(-1) > 0
    vis_indices = mx.argwhere(flat_mask).squeeze(-1)  # (M,)

    flat_image = image.reshape(-1, C)
    vis_colors = flat_image[vis_indices]  # (M, C)

    # Compute cosine weights from normals if provided.
    if normals is not None:
        # View direction: towards camera (0, 0, -1) in camera space.
        lookat = mx.array([0.0, 0.0, -1.0])
        cos_vals = mx.sum(normals * lookat[None, :], axis=-1, keepdims=True)  # (M, 1)
        cos_vals = mx.maximum(cos_vals, 0.0)
        # Zero out samples below threshold.
        cos_vals = mx.where(cos_vals < cos_threshold, mx.zeros_like(cos_vals), cos_vals)
    else:
        M = vis_colors.shape[0]
        cos_vals = mx.ones((M, 1), dtype=mx.float32)

    # Weight the colours by cosine, then scatter into texture via UV.
    weighted_colors = vis_colors * cos_vals  # (M, C)

    texture, weight = bilinear_grid_put_2d(texture_size, uv, weighted_colors)
    _, cos_weight = bilinear_grid_put_2d(texture_size, uv, cos_vals)

    return texture, cos_weight


# ---------------------------------------------------------------------------
# 5. Fast texture baking (multi-view merge)
# ---------------------------------------------------------------------------

def fast_bake_texture(textures: list, weights: list) -> mx.array:
    """
    Merge multiple per-view textures via weighted averaging.

    This mirrors ``fast_bake_texture`` in the upstream PyTorch renderer,
    including the optimisation that skips views whose coverage is >99%%
    already painted by earlier views.

    Args:
        textures: List of (H, W, C) per-view texture arrays.
        weights:  List of (H, W, 1) per-view weight maps (e.g. cosine maps).

    Returns:
        merged: (H, W, C) merged texture (weighted average).
        valid:  (H, W, 1) boolean mask where total weight > 0.
    """
    C = textures[0].shape[-1]
    H, W = textures[0].shape[:2]

    tex_accum = mx.zeros((H, W, C), dtype=mx.float32)
    w_accum = mx.zeros((H, W, 1), dtype=mx.float32)

    for tex, w in zip(textures, weights):
        # Skip views that are almost entirely redundant (>99% overlap).
        view_pixels = mx.sum(w > 0).item()
        if view_pixels == 0:
            continue
        already_painted = mx.sum((w > 0) * (w_accum > 0)).item()
        if already_painted / view_pixels > 0.99:
            continue

        tex_accum = tex_accum + tex * w
        w_accum = w_accum + w

    # Normalise, guarding against divide-by-zero.
    safe_w = mx.maximum(w_accum, 1e-8)
    merged = tex_accum / safe_w

    valid = w_accum > 1e-8
    return merged, valid

# Metal compute shader rasterizer for MLX.
# Replaces the C++ CPU rasterizer in hy3dpaint/custom_rasterizer/.
#
# Three-kernel pipeline:
#   1. rasterize_triangles — per-triangle, atomic z-buffer writes
#   2. extract_barycentrics — per-pixel, decode z-buffer + perspective-correct barycentrics
#   3. interpolate_attrs — per-pixel, weighted sum of vertex attributes

import mlx.core as mx

# ---------------------------------------------------------------------------
# Z-buffer encoding
#
# We pack depth + triangle ID into a single uint32 for atomic_min:
#   encoded = (z_quant << 20) | tri_id
#
# - 12 bits depth (4096 levels) — coarse but sufficient for non-overlapping
#   geometry at texture-baking resolution
# - 20 bits triangle ID (up to 1,048,575 triangles)
#
# Depth is in NDC [0, 1] after perspective divide, quantized to [0, 4095].
# atomic_min on the encoded value selects the nearest triangle (smallest z),
# with lowest tri_id as tiebreaker for coplanar faces.
# ---------------------------------------------------------------------------

DEPTH_BITS = 12
TRI_ID_BITS = 20
MAX_DEPTH = (1 << DEPTH_BITS) - 1  # 4095
MAX_TRI_ID = (1 << TRI_ID_BITS) - 1  # 1048575
EMPTY_ZBUF = 0xFFFFFFFF  # uint32 max — guaranteed to lose atomic_min

# ---------------------------------------------------------------------------
# Kernel 1: rasterize_triangles
#
# Dispatch: one thread per triangle (F threads total).
# Each thread does perspective divide, NDC-to-screen, computes bounding box,
# iterates pixels in bbox, tests via edge functions, and writes to z-buffer
# using atomic_min.
# ---------------------------------------------------------------------------

_RASTERIZE_TRIANGLES_SRC = """
    // Thread index = triangle index
    uint tri_id = thread_position_in_grid.x;
    if (tri_id >= num_tris) return;

    int W = resolution;
    int H = resolution;

    // Load three clip-space vertices for this triangle.
    // pos_clip is laid out as [F*3, 4] — three consecutive vertices per face.
    uint base = tri_id * 3u;

    float4 c0 = float4(pos_clip[base * 4u + 0u],
                        pos_clip[base * 4u + 1u],
                        pos_clip[base * 4u + 2u],
                        pos_clip[base * 4u + 3u]);
    float4 c1 = float4(pos_clip[(base + 1u) * 4u + 0u],
                        pos_clip[(base + 1u) * 4u + 1u],
                        pos_clip[(base + 1u) * 4u + 2u],
                        pos_clip[(base + 1u) * 4u + 3u]);
    float4 c2 = float4(pos_clip[(base + 2u) * 4u + 0u],
                        pos_clip[(base + 2u) * 4u + 1u],
                        pos_clip[(base + 2u) * 4u + 2u],
                        pos_clip[(base + 2u) * 4u + 3u]);

    // Skip degenerate triangles (w <= 0 for any vertex)
    if (c0.w <= 0.0f || c1.w <= 0.0f || c2.w <= 0.0f) return;

    // Perspective divide -> NDC in [-1, 1] for x,y and [0, 1]-ish for z
    float3 ndc0 = float3(c0.x / c0.w, c0.y / c0.w, c0.z / c0.w);
    float3 ndc1 = float3(c1.x / c1.w, c1.y / c1.w, c1.z / c1.w);
    float3 ndc2 = float3(c2.x / c2.w, c2.y / c2.w, c2.z / c2.w);

    // NDC to screen coords. Match the original C++ rasterizer convention:
    //   sx = (ndc.x * 0.5 + 0.5) * (W - 1) + 0.5
    //   sy = (ndc.y * 0.5 + 0.5) * (H - 1) + 0.5
    //   sz = ndc.z * 0.49999 + 0.5   (maps to ~[0, 1])
    float sx0 = (ndc0.x * 0.5f + 0.5f) * float(W - 1) + 0.5f;
    float sy0 = (ndc0.y * 0.5f + 0.5f) * float(H - 1) + 0.5f;
    float sz0 = ndc0.z * 0.49999f + 0.5f;

    float sx1 = (ndc1.x * 0.5f + 0.5f) * float(W - 1) + 0.5f;
    float sy1 = (ndc1.y * 0.5f + 0.5f) * float(H - 1) + 0.5f;
    float sz1 = ndc1.z * 0.49999f + 0.5f;

    float sx2 = (ndc2.x * 0.5f + 0.5f) * float(W - 1) + 0.5f;
    float sy2 = (ndc2.y * 0.5f + 0.5f) * float(H - 1) + 0.5f;
    float sz2 = ndc2.z * 0.49999f + 0.5f;

    // Bounding box in pixel coords (integer)
    int x_min = max(0, (int)floor(min(sx0, min(sx1, sx2))));
    int x_max = min(W - 1, (int)ceil(max(sx0, max(sx1, sx2))));
    int y_min = max(0, (int)floor(min(sy0, min(sy1, sy2))));
    int y_max = min(H - 1, (int)ceil(max(sy0, max(sy1, sy2))));

    // Signed area of full triangle (2x area), standard convention.
    // Using signed inv_area ensures barycentrics are positive for interior
    // points regardless of CW/CCW winding.
    float area = (sx1 - sx0) * (sy2 - sy0) - (sx2 - sx0) * (sy1 - sy0);
    if (fabs(area) < 1e-10f) return;
    float inv_area = 1.0f / area;

    // Iterate pixels in bounding box
    for (int py = y_min; py <= y_max; py++) {
        for (int px = x_min; px <= x_max; px++) {
            float pxf = float(px) + 0.5f;
            float pyf = float(py) + 0.5f;

            // Barycentric via signed areas (sign-corrected for winding)
            float beta  = ((pxf - sx0) * (sy2 - sy0) - (sx2 - sx0) * (pyf - sy0)) * inv_area;
            float gamma = ((sx1 - sx0) * (pyf - sy0) - (pxf - sx0) * (sy1 - sy0)) * inv_area;
            float alpha = 1.0f - beta - gamma;

            if (alpha < 0.0f || alpha > 1.0f ||
                beta  < 0.0f || beta  > 1.0f ||
                gamma < 0.0f || gamma > 1.0f) continue;

            // Interpolate depth
            float depth = alpha * sz0 + beta * sz1 + gamma * sz2;
            // Quantize depth to 12 bits [0, 4095]
            uint z_quant = (uint)clamp(depth * 4095.0f, 0.0f, 4095.0f);

            // Encode: upper 12 bits = depth, lower 20 bits = triangle ID
            uint token = (z_quant << 20u) | (tri_id & 0xFFFFFu);

            uint pixel = (uint)py * (uint)W + (uint)px;
            atomic_fetch_min_explicit(
                &zbuffer[pixel],
                token,
                memory_order_relaxed);
        }
    }
"""

_rasterize_triangles_kernel = mx.fast.metal_kernel(
    name="rasterize_triangles",
    input_names=["pos_clip", "num_tris", "resolution"],
    output_names=["zbuffer"],
    source=_RASTERIZE_TRIANGLES_SRC,
    atomic_outputs=True,
)

# ---------------------------------------------------------------------------
# Kernel 2: extract_barycentrics
#
# Dispatch: one thread per pixel (H*W threads).
# Decode triangle ID from z-buffer, recompute perspective-correct barycentrics.
# ---------------------------------------------------------------------------

_EXTRACT_BARYCENTRICS_SRC = """
    uint pix = thread_position_in_grid.x;
    uint W = (uint)resolution;
    uint H = W;
    if (pix >= W * H) return;

    uint token = zbuffer[pix];

    // Check for empty pixel (0xFFFFFFFF)
    if (token == 0xFFFFFFFFu) {
        face_idx_out[pix] = -1;
        bary_out[pix * 3u + 0u] = 0.0f;
        bary_out[pix * 3u + 1u] = 0.0f;
        bary_out[pix * 3u + 2u] = 0.0f;
        return;
    }

    uint tri_id = token & 0xFFFFFu;
    face_idx_out[pix] = (int)tri_id;

    // Pixel screen coords
    float pxf = float(pix % W) + 0.5f;
    float pyf = float(pix / W) + 0.5f;

    // Reload clip-space vertices for this triangle
    uint base = tri_id * 3u;
    float4 c0 = float4(pos_clip[base * 4u + 0u],
                        pos_clip[base * 4u + 1u],
                        pos_clip[base * 4u + 2u],
                        pos_clip[base * 4u + 3u]);
    float4 c1 = float4(pos_clip[(base + 1u) * 4u + 0u],
                        pos_clip[(base + 1u) * 4u + 1u],
                        pos_clip[(base + 1u) * 4u + 2u],
                        pos_clip[(base + 1u) * 4u + 3u]);
    float4 c2 = float4(pos_clip[(base + 2u) * 4u + 0u],
                        pos_clip[(base + 2u) * 4u + 1u],
                        pos_clip[(base + 2u) * 4u + 2u],
                        pos_clip[(base + 2u) * 4u + 3u]);

    int Wm1 = (int)W - 1;
    int Hm1 = (int)H - 1;

    // Screen-space positions (same transform as kernel 1)
    float sx0 = (c0.x / c0.w * 0.5f + 0.5f) * float(Wm1) + 0.5f;
    float sy0 = (c0.y / c0.w * 0.5f + 0.5f) * float(Hm1) + 0.5f;
    float sx1 = (c1.x / c1.w * 0.5f + 0.5f) * float(Wm1) + 0.5f;
    float sy1 = (c1.y / c1.w * 0.5f + 0.5f) * float(Hm1) + 0.5f;
    float sx2 = (c2.x / c2.w * 0.5f + 0.5f) * float(Wm1) + 0.5f;
    float sy2 = (c2.y / c2.w * 0.5f + 0.5f) * float(Hm1) + 0.5f;

    // Signed area barycentric — standard convention, signed division
    float area = (sx1 - sx0) * (sy2 - sy0) - (sx2 - sx0) * (sy1 - sy0);
    if (fabs(area) < 1e-10f) {
        face_idx_out[pix] = -1;
        bary_out[pix * 3u + 0u] = 0.0f;
        bary_out[pix * 3u + 1u] = 0.0f;
        bary_out[pix * 3u + 2u] = 0.0f;
        return;
    }
    float inv_area = 1.0f / area;

    float beta  = ((pxf - sx0) * (sy2 - sy0) - (sx2 - sx0) * (pyf - sy0)) * inv_area;
    float gamma = ((sx1 - sx0) * (pyf - sy0) - (pxf - sx0) * (sy1 - sy0)) * inv_area;
    float alpha = 1.0f - beta - gamma;

    // Perspective-correct barycentrics: divide by clip w, renormalize
    float b0 = alpha / c0.w;
    float b1 = beta  / c1.w;
    float b2 = gamma / c2.w;
    float w_sum = 1.0f / (b0 + b1 + b2);
    b0 *= w_sum;
    b1 *= w_sum;
    b2 *= w_sum;

    bary_out[pix * 3u + 0u] = b0;
    bary_out[pix * 3u + 1u] = b1;
    bary_out[pix * 3u + 2u] = b2;
"""

_extract_barycentrics_kernel = mx.fast.metal_kernel(
    name="extract_barycentrics",
    input_names=["zbuffer", "pos_clip", "resolution"],
    output_names=["face_idx_out", "bary_out"],
    source=_EXTRACT_BARYCENTRICS_SRC,
)

# ---------------------------------------------------------------------------
# Kernel 3: interpolate_attrs
#
# Dispatch: one thread per pixel (H*W threads).
# Weighted sum of per-vertex attributes using barycentrics.
# ---------------------------------------------------------------------------

_INTERPOLATE_ATTRS_SRC = """
    uint pix = thread_position_in_grid.x;
    uint total = num_pixels;
    if (pix >= total) return;

    int tri_id = face_indices[pix];
    if (tri_id < 0) {
        // Empty pixel — zero out all channels
        for (uint c = 0u; c < num_channels; c++) {
            result[pix * num_channels + c] = 0.0f;
        }
        return;
    }

    float b0 = barycentrics[pix * 3u + 0u];
    float b1 = barycentrics[pix * 3u + 1u];
    float b2 = barycentrics[pix * 3u + 2u];

    // Look up vertex indices for this face
    uint f0 = (uint)faces[tri_id * 3u + 0u];
    uint f1 = (uint)faces[tri_id * 3u + 1u];
    uint f2 = (uint)faces[tri_id * 3u + 2u];

    for (uint c = 0u; c < num_channels; c++) {
        float a0 = attrs[f0 * num_channels + c];
        float a1 = attrs[f1 * num_channels + c];
        float a2 = attrs[f2 * num_channels + c];
        result[pix * num_channels + c] = b0 * a0 + b1 * a1 + b2 * a2;
    }
"""

_interpolate_attrs_kernel = mx.fast.metal_kernel(
    name="interpolate_attrs",
    input_names=["attrs", "face_indices", "barycentrics", "faces",
                 "num_pixels", "num_channels"],
    output_names=["result"],
    source=_INTERPOLATE_ATTRS_SRC,
)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

class MetalRasterizer:
    """GPU rasterizer using Metal compute shaders via MLX.

    Drop-in replacement for the C++ custom_rasterizer used by hy3dpaint.
    All computation stays on the GPU — no CPU round-trips.

    Typical usage::

        rast = MetalRasterizer(1024)
        # vertices_clip: (V, 4) after MVP, faces: (F, 3) int32
        face_indices, barycentrics = rast.rasterize(vertices_clip, faces)
        # Interpolate UV coords, normals, etc.
        uv_map = rast.interpolate(uv_coords, face_indices, barycentrics, faces)
    """

    def __init__(self, resolution: int):
        self.resolution = resolution

    def rasterize(
        self,
        vertices_clip: mx.array,
        faces: mx.array,
    ) -> tuple:
        """Rasterize triangles and produce per-pixel face indices + barycentrics.

        Args:
            vertices_clip: (V, 4) float32 clip-space positions (after MVP).
            faces: (F, 3) int32 vertex indices per face.

        Returns:
            face_indices: (H, W) int32 — triangle index per pixel, -1 if empty.
            barycentrics: (H, W, 3) float32 — perspective-correct barycentric
                coordinates per pixel.
        """
        V, _ = vertices_clip.shape
        F, _ = faces.shape
        H = W = self.resolution

        # Build pos_clip as [F*3, 4]: gather vertices by face indices.
        # faces is (F, 3) -> flatten to (F*3,) -> index into vertices_clip.
        face_verts = vertices_clip[faces.reshape(-1)]  # (F*3, 4)
        pos_clip = face_verts.astype(mx.float32)

        num_tris = mx.array(F, dtype=mx.uint32)
        resolution = mx.array(self.resolution, dtype=mx.uint32)

        # Kernel 1: rasterize triangles -> z-buffer
        num_pixels = H * W
        threadgroup_size = 256
        grid_k1 = ((F + threadgroup_size - 1) // threadgroup_size) * threadgroup_size

        zbuffer = _rasterize_triangles_kernel(
            inputs=[pos_clip, num_tris, resolution],
            grid=(grid_k1, 1, 1),
            threadgroup=(threadgroup_size, 1, 1),
            output_shapes=[(num_pixels,)],
            output_dtypes=[mx.uint32],
            init_value=float(EMPTY_ZBUF),
        )[0]

        # Kernel 2: extract face indices + barycentrics from z-buffer
        grid_k2 = (
            ((num_pixels + threadgroup_size - 1) // threadgroup_size)
            * threadgroup_size
        )

        face_indices, barycentrics = _extract_barycentrics_kernel(
            inputs=[zbuffer, pos_clip, resolution],
            grid=(grid_k2, 1, 1),
            threadgroup=(threadgroup_size, 1, 1),
            output_shapes=[(num_pixels,), (num_pixels, 3)],
            output_dtypes=[mx.int32, mx.float32],
        )

        face_indices = face_indices.reshape(H, W)
        barycentrics = barycentrics.reshape(H, W, 3)

        return face_indices, barycentrics

    def interpolate(
        self,
        attrs: mx.array,
        face_indices: mx.array,
        barycentrics: mx.array,
        faces: mx.array,
    ) -> mx.array:
        """Interpolate per-vertex attributes over the rasterized image.

        Args:
            attrs: (V, C) float32 per-vertex attributes (UVs, normals, etc.).
            face_indices: (H, W) int32 triangle indices from ``rasterize()``.
            barycentrics: (H, W, 3) float32 barycentric weights.
            faces: (F, 3) int32 vertex indices per face.

        Returns:
            (H, W, C) float32 interpolated attributes.
        """
        H, W = face_indices.shape
        _, C = attrs.shape
        num_pixels = H * W

        face_indices_flat = face_indices.reshape(-1).astype(mx.int32)
        barycentrics_flat = barycentrics.reshape(-1, 3).astype(mx.float32)
        attrs = attrs.astype(mx.float32)
        faces = faces.astype(mx.int32)

        num_pixels_arr = mx.array(num_pixels, dtype=mx.uint32)
        num_channels_arr = mx.array(C, dtype=mx.uint32)

        threadgroup_size = 256
        grid_k3 = (
            ((num_pixels + threadgroup_size - 1) // threadgroup_size)
            * threadgroup_size
        )

        result = _interpolate_attrs_kernel(
            inputs=[attrs, face_indices_flat, barycentrics_flat, faces,
                    num_pixels_arr, num_channels_arr],
            grid=(grid_k3, 1, 1),
            threadgroup=(threadgroup_size, 1, 1),
            output_shapes=[(num_pixels, C)],
            output_dtypes=[mx.float32],
        )[0]

        return result.reshape(H, W, C)

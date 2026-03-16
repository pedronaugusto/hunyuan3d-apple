import os
from dataclasses import dataclass
from typing import Iterable


DEFAULT_LOCAL_ROOTS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "tencent", "Hunyuan3D-2.1"),
]


@dataclass(frozen=True)
class ResolvedHunyuanPaths:
    root: str
    shape_dir: str
    texture_dir: str


def _iter_candidate_roots(model_ref: str) -> Iterable[str]:
    if os.path.isdir(model_ref):
        yield os.path.abspath(model_ref)

    env_root = os.environ.get("HUNYUAN_LOCAL_ROOT")
    if env_root:
        yield os.path.abspath(env_root)

    for root in DEFAULT_LOCAL_ROOTS:
        if not os.path.isdir(root):
            continue
        if root.endswith("/snapshots"):
            for entry in sorted(os.listdir(root)):
                candidate = os.path.join(root, entry)
                if os.path.isdir(candidate):
                    yield candidate
        else:
            yield os.path.abspath(root)


def resolve_hunyuan_root(model_ref: str) -> str:
    for root in _iter_candidate_roots(model_ref):
        if os.path.isdir(os.path.join(root, "hunyuan3d-dit-v2-1")) and os.path.isdir(
            os.path.join(root, "hunyuan3d-paintpbr-v2-1")
        ):
            return root
        if os.path.isfile(os.path.join(root, "model_index.json")) and os.path.isdir(os.path.join(root, "scheduler")):
            return os.path.dirname(root)
    raise FileNotFoundError(
        f"Could not resolve local Hunyuan model root for {model_ref!r}. "
        "Set HUNYUAN_LOCAL_ROOT or point --model_path at a local snapshot."
    )


def resolve_hunyuan_paths(model_ref: str) -> ResolvedHunyuanPaths:
    root = resolve_hunyuan_root(model_ref)
    shape_dir = os.path.join(root, "hunyuan3d-dit-v2-1")
    texture_dir = os.path.join(root, "hunyuan3d-paintpbr-v2-1")
    if not os.path.isdir(shape_dir):
        raise FileNotFoundError(f"Missing local shape dir: {shape_dir}")
    if not os.path.isdir(texture_dir):
        raise FileNotFoundError(f"Missing local texture dir: {texture_dir}")
    return ResolvedHunyuanPaths(root=root, shape_dir=shape_dir, texture_dir=texture_dir)


def ensure_paths_exist(paths: Iterable[str]) -> None:
    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("Missing required local assets:\n" + "\n".join(missing))

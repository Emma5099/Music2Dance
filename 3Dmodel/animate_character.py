#!/usr/bin/env python3
"""Animate a static character GLB with MediaPipe-style poses (T, 33, 3).

The TRELLIS mesh is typically unrigged. This script:
  1. Decimates the mesh for realtime-ish rendering
  2. Aligns it to the rest-pose skeleton
  3. Builds automatic distance-based skin weights
  4. Applies linear-blend skinning each frame
  5. Renders an MP4 (optional audio mux via ffmpeg)

Usage:
  # From an existing pose dump + disco ball mesh
  python 3Dmodel/animate_character.py \\
      --poses poses/gBR_sBM_c01_d04_mBR0_ch01_poses.npy \\
      --mesh 3Dmodel/discoball_trellis2.glb \\
      --audio outputs/MA.wav \\
      --out outputs/discoball_dance.mp4

  # Short smoke test
  python 3Dmodel/animate_character.py --poses ... --mesh ... --frames 50 --out outputs/disco_smoke.mp4
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

POSE_FPS = 25

# Mixamo joint → MediaPipe index (-1 mid-hips, -2 mid-shoulders). Spine uses blends.
# Clavicles (Shoulder) intentionally unmapped → keep rest offset from Spine2 (avoids neck pinch).
MIXAMO_TO_MP: dict[str, int | float | tuple] = {
    "mixamorig:Hips": -1,
    "mixamorig:Spine": ("lerp", -1, -2, 0.22),
    "mixamorig:Spine1": ("lerp", -1, -2, 0.48),
    "mixamorig:Spine2": ("lerp", -1, -2, 0.72),
    "mixamorig:Neck": ("lerp", -2, 0, 0.22),
    "mixamorig:Head": ("lerp", -2, 0, 0.65),
    "mixamorig:HeadTop_End": 0,
    "mixamorig:LeftArm": 11,
    "mixamorig:LeftForeArm": 13,
    "mixamorig:LeftHand": 15,
    "mixamorig:RightArm": 12,
    "mixamorig:RightForeArm": 14,
    "mixamorig:RightHand": 16,
    "mixamorig:LeftUpLeg": 23,
    "mixamorig:LeftLeg": 25,
    "mixamorig:LeftFoot": 27,
    "mixamorig:LeftToeBase": 31,
    "mixamorig:LeftToe_End": 31,
    "mixamorig:RightUpLeg": 24,
    "mixamorig:RightLeg": 26,
    "mixamorig:RightFoot": 28,
    "mixamorig:RightToeBase": 32,
    "mixamorig:RightToe_End": 32,
}

# Preferred child for bone aim (orientation). Fingers keep rest orientation.
MIXAMO_AIM_CHILD: dict[str, str] = {
    "mixamorig:Hips": "mixamorig:Spine",
    "mixamorig:Spine": "mixamorig:Spine1",
    "mixamorig:Spine1": "mixamorig:Spine2",
    "mixamorig:Spine2": "mixamorig:Neck",
    "mixamorig:Neck": "mixamorig:Head",
    "mixamorig:Head": "mixamorig:HeadTop_End",
    "mixamorig:LeftShoulder": "mixamorig:LeftArm",
    "mixamorig:LeftArm": "mixamorig:LeftForeArm",
    "mixamorig:LeftForeArm": "mixamorig:LeftHand",
    "mixamorig:RightShoulder": "mixamorig:RightArm",
    "mixamorig:RightArm": "mixamorig:RightForeArm",
    "mixamorig:RightForeArm": "mixamorig:RightHand",
    "mixamorig:LeftUpLeg": "mixamorig:LeftLeg",
    "mixamorig:LeftLeg": "mixamorig:LeftFoot",
    "mixamorig:LeftFoot": "mixamorig:LeftToeBase",
    "mixamorig:RightUpLeg": "mixamorig:RightLeg",
    "mixamorig:RightLeg": "mixamorig:RightFoot",
    "mixamorig:RightFoot": "mixamorig:RightToeBase",
}

# Fallback auto-rig bones when the GLB has no Mixamo skin.
BONES: list[tuple[str, int, int]] = [
    ("spine", -1, -2),
    ("neck", -2, 0),
    ("l_shoulder", -2, 11),
    ("r_shoulder", -2, 12),
    ("l_upper_arm", 11, 13),
    ("l_forearm", 13, 15),
    ("r_upper_arm", 12, 14),
    ("r_forearm", 14, 16),
    ("l_hip", -1, 23),
    ("r_hip", -1, 24),
    ("l_thigh", 23, 25),
    ("l_shin", 25, 27),
    ("r_thigh", 24, 26),
    ("r_shin", 26, 28),
    ("l_foot", 27, 31),
    ("r_foot", 28, 32),
]


def _read_glb(path: Path) -> tuple[dict, np.ndarray]:
    """Return (gltf JSON, binary buffer bytes) from a .glb file."""
    import json
    import struct

    data = path.read_bytes()
    if data[:4] != b"glTF":
        raise ValueError(f"Not a GLB: {path}")
    json_len = struct.unpack_from("<I", data, 12)[0]
    json_chunk = data[20 : 20 + json_len]
    gltf = json.loads(json_chunk)
    # BIN chunk follows (may have padding)
    bin_off = 20 + json_len
    if bin_off % 4:
        bin_off += 4 - (bin_off % 4)
    if bin_off + 8 <= len(data):
        bin_len = struct.unpack_from("<I", data, bin_off)[0]
        blob = data[bin_off + 8 : bin_off + 8 + bin_len]
    else:
        blob = b""
    return gltf, np.frombuffer(blob, dtype=np.uint8)


def _accessor_numpy(gltf: dict, blob: np.ndarray, accessor_idx: int) -> np.ndarray:
    acc = gltf["accessors"][accessor_idx]
    view = gltf["bufferViews"][acc["bufferView"]]
    component = {
        5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16,
        5125: np.uint32, 5126: np.float32,
    }[acc["componentType"]]
    ncomp = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[acc["type"]]
    count = acc["count"]
    byte_offset = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    dtype = np.dtype(component)
    item_size = dtype.itemsize * ncomp
    stride = view.get("byteStride", item_size)
    if stride == item_size:
        arr = np.frombuffer(blob, dtype=component, count=count * ncomp, offset=byte_offset)
        return arr.reshape(count, ncomp).copy() if ncomp > 1 else arr.copy()
    # Strided
    out = np.empty((count, ncomp), dtype=component)
    for i in range(count):
        off = byte_offset + i * stride
        out[i] = np.frombuffer(blob, dtype=component, count=ncomp, offset=off)
    return out


def _primitive_geometry(gltf: dict, blob: np.ndarray, prim: dict):
    """Mesh primitive → positions (V,3), faces (F,3), uvs (V,2)|None."""
    attrs = prim["attributes"]
    positions = _accessor_numpy(gltf, blob, attrs["POSITION"]).astype(np.float64)
    if "indices" in prim:
        faces = _accessor_numpy(gltf, blob, prim["indices"]).astype(np.int32).reshape(-1, 3)
    else:
        faces = np.arange(len(positions), dtype=np.int32).reshape(-1, 3)
    uvs = None
    if "TEXCOORD_0" in attrs:
        uvs = _accessor_numpy(gltf, blob, attrs["TEXCOORD_0"]).astype(np.float64)
    return positions, faces, uvs


def _decode_glb_image(gltf: dict, blob: np.ndarray, image_idx: int) -> np.ndarray | None:
    """Decode glTF image → RGB uint8 (H,W,3), or None."""
    img = gltf["images"][image_idx]
    if "bufferView" in img:
        view = gltf["bufferViews"][img["bufferView"]]
        off = view.get("byteOffset", 0)
        raw = bytes(blob[off : off + view["byteLength"]])
    elif "uri" in img and img["uri"].startswith("data:"):
        import base64

        raw = base64.b64decode(img["uri"].split(",", 1)[1])
    else:
        return None
    arr = np.frombuffer(raw, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
    if decoded is None:
        return None
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def _texture_image(gltf: dict, blob: np.ndarray, texture_idx: int) -> np.ndarray | None:
    tex = gltf["textures"][texture_idx]
    src = tex.get("source")
    if src is None:
        return None
    return _decode_glb_image(gltf, blob, src)


def _load_pbr_maps(gltf: dict, blob: np.ndarray, material_idx: int | None) -> dict:
    """Load base-color / normal maps + metallic-roughness factors from a material."""
    out = {
        "albedo": None,
        "normal": None,
        "metallic": 1.0,
        "roughness": 0.5,
        "base_color": np.array([0.85, 0.88, 0.92], dtype=np.float64),
    }
    if material_idx is None or not gltf.get("materials"):
        return out
    mat = gltf["materials"][material_idx]
    pbr = mat.get("pbrMetallicRoughness", {})
    out["metallic"] = float(pbr.get("metallicFactor", 1.0))
    out["roughness"] = float(pbr.get("roughnessFactor", 1.0))
    if "baseColorFactor" in pbr:
        out["base_color"] = np.array(pbr["baseColorFactor"][:3], dtype=np.float64)
    if "baseColorTexture" in pbr:
        out["albedo"] = _texture_image(gltf, blob, pbr["baseColorTexture"]["index"])
        if out["albedo"] is not None:
            print(
                f"  Albedo texture: {out['albedo'].shape[1]}×{out['albedo'].shape[0]}",
                flush=True,
            )
    if "normalTexture" in mat:
        out["normal"] = _texture_image(gltf, blob, mat["normalTexture"]["index"])
        if out["normal"] is not None:
            print(
                f"  Normal texture: {out['normal'].shape[1]}×{out['normal'].shape[0]}",
                flush=True,
            )
    return out


def _sample_texture(tex: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinear-ish nearest sample. u,v in [0,1] (V,); returns (V,3) RGB float 0-1."""
    h, w = tex.shape[:2]
    # glTF UV: origin bottom-left → OpenCV top-left
    x = np.clip(u, 0.0, 1.0) * (w - 1)
    y = np.clip(1.0 - v, 0.0, 1.0) * (h - 1)
    xi = np.clip(np.round(x).astype(np.int32), 0, w - 1)
    yi = np.clip(np.round(y).astype(np.int32), 0, h - 1)
    return tex[yi, xi].astype(np.float64) / 255.0


def _skin_parent_si(
    node_idx: int,
    parents: dict[int, int | None],
    node_to_skin: dict[int, int],
) -> int | None:
    """Walk up the node tree until we hit a joint that belongs to the skin."""
    p = parents.get(node_idx)
    while p is not None:
        if p in node_to_skin:
            return node_to_skin[p]
        p = parents.get(p)
    return None


def try_load_skinned_glb(path: Path) -> dict | None:
    """Load Mixamo-style skinned mesh from GLB, or None if not skinned."""
    if path.suffix.lower() != ".glb":
        return None
    try:
        gltf, blob = _read_glb(path)
    except Exception:
        return None
    if not gltf.get("skins"):
        return None

    skin = gltf["skins"][0]
    joint_node_ids = skin["joints"]
    ibm = _accessor_numpy(gltf, blob, skin["inverseBindMatrices"]).astype(np.float64)
    ibm = ibm.reshape(-1, 4, 4)
    # glTF matrices are column-major
    ibm = np.transpose(ibm, (0, 2, 1))

    joint_names = [gltf["nodes"][i].get("name", f"joint_{i}") for i in joint_node_ids]
    name_to_skin = {n: si for si, n in enumerate(joint_names)}

    # Find skinned mesh primitive (has JOINTS_0)
    prim = None
    for mesh in gltf.get("meshes", []):
        for p in mesh["primitives"]:
            if "JOINTS_0" in p.get("attributes", {}):
                prim = p
                break
        if prim:
            break
    if prim is None:
        return None

    attrs = prim["attributes"]
    positions = _accessor_numpy(gltf, blob, attrs["POSITION"]).astype(np.float64)
    joints = _accessor_numpy(gltf, blob, attrs["JOINTS_0"]).astype(np.int32)
    weights = _accessor_numpy(gltf, blob, attrs["WEIGHTS_0"]).astype(np.float64)
    weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-8)
    indices = _accessor_numpy(gltf, blob, prim["indices"]).astype(np.int32).reshape(-1, 3)
    uvs = None
    if "TEXCOORD_0" in attrs:
        uvs = _accessor_numpy(gltf, blob, attrs["TEXCOORD_0"]).astype(np.float64)
    pbr = _load_pbr_maps(gltf, blob, prim.get("material"))


    # Rest global matrices from node local TRS hierarchy
    nodes = gltf["nodes"]
    children: dict[int, list[int]] = {i: [] for i in range(len(nodes))}
    parents = {i: None for i in range(len(nodes))}
    for i, n in enumerate(nodes):
        for c in n.get("children", []) or []:
            children[i].append(c)
            parents[c] = i

    def local_matrix(n: dict) -> np.ndarray:
        if "matrix" in n:
            M = np.array(n["matrix"], dtype=np.float64).reshape(4, 4).T
            return M
        T = np.eye(4)
        t = n.get("translation", [0, 0, 0])
        T[:3, 3] = t
        R = np.eye(4)
        if "rotation" in n:  # quat xyzw
            x, y, z, w = n["rotation"]
            R[:3, :3] = np.array([
                [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
            ], dtype=np.float64)
        S = np.eye(4)
        if "scale" in n:
            sx, sy, sz = n["scale"]
            S[0, 0], S[1, 1], S[2, 2] = sx, sy, sz
        return T @ R @ S

    locals_m = [local_matrix(n) for n in nodes]
    globals_m = [np.eye(4) for _ in nodes]

    def dfs(i: int, parent: np.ndarray):
        globals_m[i] = parent @ locals_m[i]
        for c in children[i]:
            dfs(c, globals_m[i])

    roots = [i for i, p in parents.items() if p is None]
    for r in roots:
        dfs(r, np.eye(4))

    rest_globals = np.stack([globals_m[j] for j in joint_node_ids], axis=0)

    # Rest bone head positions (translation of global)
    rest_heads = rest_globals[:, :3, 3].copy()

    # Extra mesh objects parented to bones but not skinned (e.g. disco-ball head).
    node_to_skin = {nid: si for si, nid in enumerate(joint_node_ids)}
    attachments: list[dict] = []
    merged_faces = indices.copy()
    n_body_faces = len(indices)
    n_body_verts = len(positions)
    merged_uvs = uvs.copy() if uvs is not None else None
    vert_offset = len(positions)
    for ni, n in enumerate(nodes):
        if "mesh" not in n:
            continue
        mesh = gltf["meshes"][n["mesh"]]
        att_prim = mesh["primitives"][0]
        if "JOINTS_0" in att_prim.get("attributes", {}):
            continue
        parent_si = _skin_parent_si(ni, parents, node_to_skin)
        if parent_si is None:
            print(f"  [warn] skipping mesh '{n.get('name')}' — no skin parent", flush=True)
            continue
        att_pos, att_faces, att_uvs = _primitive_geometry(gltf, blob, att_prim)
        bind = np.linalg.inv(rest_globals[parent_si]) @ globals_m[ni]
        homo = np.concatenate([att_pos, np.ones((len(att_pos), 1))], axis=1)
        local = (bind @ homo.T).T
        attachments.append({
            "local": local,
            "parent_si": parent_si,
            "name": n.get("name", f"mesh_{ni}"),
            "n_verts": len(att_pos),
        })
        merged_faces = np.concatenate([merged_faces, att_faces + vert_offset])
        if att_uvs is not None:
            if merged_uvs is None:
                merged_uvs = np.zeros((vert_offset, 2), dtype=np.float64)
            merged_uvs = np.concatenate([merged_uvs, att_uvs], axis=0)
        elif merged_uvs is not None:
            merged_uvs = np.concatenate(
                [merged_uvs, np.zeros((len(att_pos), 2), dtype=np.float64)], axis=0
            )
        # Prefer attachment albedo if body had none
        if pbr["albedo"] is None:
            att_pbr = _load_pbr_maps(gltf, blob, att_prim.get("material"))
            if att_pbr["albedo"] is not None:
                pbr = att_pbr
        vert_offset += len(att_pos)
        print(
            f"  + rigid attachment '{n.get('name')}' → "
            f"{joint_names[parent_si]} ({len(att_pos)} verts)",
            flush=True,
        )

    print(
        f"  Skinned Mixamo GLB: {len(positions)} verts, {len(joint_names)} joints, "
        f"{len(merged_faces)} faces ({len(attachments)} rigid attachments)",
        flush=True,
    )
    # Disco/chrome look (albedo atlas often mismatches custom UVs — used as tint only).
    body_metallic = 0.88
    body_roughness = 0.16
    disco_metallic = 0.96
    disco_roughness = 0.05

    # Downsample huge albedo for faster / slightly softer lookup
    albedo = pbr["albedo"]
    if albedo is not None and max(albedo.shape[:2]) > 1024:
        h, w = albedo.shape[:2]
        scale = 1024 / max(h, w)
        albedo = cv2.resize(
            albedo,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
        print(f"  Albedo resized → {albedo.shape[1]}×{albedo.shape[0]}", flush=True)

    return {
        "positions": positions,
        "faces": merged_faces,
        "n_body_faces": n_body_faces,
        "n_body_verts": n_body_verts,
        "uvs": merged_uvs,
        "albedo": albedo,
        "normal_tex": pbr["normal"],
        "metallic": body_metallic,
        "roughness": body_roughness,
        "disco_metallic": disco_metallic,
        "disco_roughness": disco_roughness,
        "base_color": pbr["base_color"],
        "attachments": attachments,
        "joints": joints,
        "weights": weights,
        "ibm": ibm,
        "joint_names": joint_names,
        "name_to_skin": name_to_skin,
        "rest_globals": rest_globals,
        "rest_heads": rest_heads,
        "joint_node_ids": joint_node_ids,
        "nodes": nodes,
        "children": children,
        "parents": parents,
        "locals_rest": locals_m,
    }


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """3x3 rotation taking unit vector a → unit vector b."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.999999:
        axis = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = axis - a * np.dot(axis, a)
        axis /= np.linalg.norm(axis) + 1e-12
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + 2 * (K @ K)
    s = np.linalg.norm(v)
    if s < 1e-8:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s + 1e-12))


def _mp_joint(pose35: np.ndarray, spec) -> np.ndarray:
    """Resolve MIXAMO_TO_MP entry → (3,) position."""
    if isinstance(spec, tuple) and spec and spec[0] == "lerp":
        _, ja, jb, t = spec
        a = pose35[_resolve_idx(ja)]
        b = pose35[_resolve_idx(jb)]
        return (1.0 - t) * a + t * b
    return pose35[_resolve_idx(int(spec))]


def _two_bone_ik(
    root: np.ndarray,
    mid_hint: np.ndarray,
    end_target: np.ndarray,
    len1: float,
    len2: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Place mid (knee) and end (ankle) with fixed bone lengths."""
    to_end = end_target - root
    dist = float(np.linalg.norm(to_end))
    max_reach = len1 + len2 - 1e-4
    min_reach = abs(len1 - len2) + 1e-4
    if dist < 1e-8:
        # Degenerate — keep hint direction
        direction = mid_hint - root
        if np.linalg.norm(direction) < 1e-8:
            direction = np.array([0.0, -1.0, 0.0])
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        mid = root + direction * len1
        end = mid + direction * len2
        return mid, end

    dist_clamped = float(np.clip(dist, min_reach, max_reach))
    direction = to_end / dist
    end = root + direction * dist_clamped

    # Law of cosines: angle at root between end and mid
    cos_a = (len1 * len1 + dist_clamped * dist_clamped - len2 * len2) / (
        2.0 * len1 * dist_clamped
    )
    cos_a = float(np.clip(cos_a, -1.0, 1.0))
    sin_a = float(np.sqrt(max(0.0, 1.0 - cos_a * cos_a)))

    # Pole from hint (side of bend)
    pole = mid_hint - root
    # Orthonormal basis in the IK plane
    axis = direction
    proj = pole - axis * np.dot(pole, axis)
    if np.linalg.norm(proj) < 1e-6:
        # Arbitrary perpendicular
        tmp = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
        proj = np.cross(axis, tmp)
    side = proj / (np.linalg.norm(proj) + 1e-12)

    mid = root + axis * (len1 * cos_a) + side * (len1 * sin_a)
    return mid, end


def _plant_both_feet(
    targets: np.ndarray,
    rest_heads: np.ndarray,
    name_to_skin: dict[str, int],
    floor_y: float,
) -> np.ndarray:
    """Plant feet on floor_y without destroying leg silhouette.

    - Always hard-plant the lower (support) foot.
    - Soft-plant the other foot if it is already near the floor.
    - Otherwise leave it free (avoids the fused-leg / pillar artifact).
    """
    out = targets.copy()
    chains = [
        ("mixamorig:LeftUpLeg", "mixamorig:LeftLeg", "mixamorig:LeftFoot",
         "mixamorig:LeftToeBase", "mixamorig:LeftToe_End"),
        ("mixamorig:RightUpLeg", "mixamorig:RightLeg", "mixamorig:RightFoot",
         "mixamorig:RightToeBase", "mixamorig:RightToe_End"),
    ]
    ready = []
    for up_n, leg_n, foot_n, toe_n, toe_end_n in chains:
        if up_n not in name_to_skin or leg_n not in name_to_skin or foot_n not in name_to_skin:
            continue
        ui, li, fi = name_to_skin[up_n], name_to_skin[leg_n], name_to_skin[foot_n]
        len1 = float(np.linalg.norm(rest_heads[li] - rest_heads[ui]) + 1e-12)
        len2 = float(np.linalg.norm(rest_heads[fi] - rest_heads[li]) + 1e-12)
        ready.append((ui, li, fi, toe_n, toe_end_n, len1, len2, float(out[fi, 1])))

    if not ready:
        return out

    # Support = currently lower foot
    support = int(np.argmin([h for *_, h in ready]))
    # Lift character so support foot can sit on floor
    ui, li, fi, toe_n, toe_end_n, len1, len2, _ = ready[support]
    foot = out[fi].copy()
    foot[1] = floor_y
    reach = len1 + len2 - 1e-3
    need = float(np.linalg.norm(foot - out[ui]))
    if need > reach:
        out[:, 1] += need - reach

    leg_len = len1 + len2
    for i, (ui, li, fi, toe_n, toe_end_n, len1, len2, raw_y) in enumerate(ready):
        height = float(out[fi, 1] - floor_y)
        if i == support:
            blend = 1.0
        else:
            # Soft contact only when near floor (dance steps stay natural)
            blend = float(np.clip(1.0 - height / (0.22 * leg_len + 1e-6), 0.0, 1.0))
            blend = blend * blend  # ease
        if blend < 0.05:
            continue
        foot_tgt = out[fi].copy()
        foot_tgt[1] = (1.0 - blend) * out[fi, 1] + blend * floor_y
        mid, end = _two_bone_ik(out[ui], out[li], foot_tgt, len1, len2)
        out[li] = mid
        out[fi] = end
        for tn in (toe_n, toe_end_n):
            if tn not in name_to_skin:
                continue
            ti = name_to_skin[tn]
            off = rest_heads[ti] - rest_heads[fi]
            out[ti] = out[fi] + np.array([off[0], 0.0, off[2]])
            out[ti, 1] = out[fi, 1]
    return out


def _hips_basis(left: np.ndarray, right: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    """Orthonormal basis (columns = X right, Y up, Z forward) from hip landmarks."""
    x = right - left
    x = x / (np.linalg.norm(x) + 1e-12)
    y = up_hint.astype(np.float64).copy()
    y = y - x * np.dot(y, x)
    yn = np.linalg.norm(y)
    if yn < 1e-8:
        y = np.array([0.0, 1.0, 0.0])
        y = y - x * np.dot(y, x)
        yn = np.linalg.norm(y) + 1e-12
    y = y / yn
    z = np.cross(x, y)
    z = z / (np.linalg.norm(z) + 1e-12)
    # Re-orthogonalize Y in case of numerical drift
    y = np.cross(z, x)
    y = y / (np.linalg.norm(y) + 1e-12)
    return np.stack([x, y, z], axis=1)


def mixamo_skin_frame(
    skinned: dict,
    pose33: np.ndarray,
    *,
    floor_y: float | None = None,
    scale: float | None = None,
    plant_feet: bool = False,
) -> np.ndarray:
    """Apply MediaPipe pose to Mixamo skinned mesh → (V,3) vertices.

    Builds **hierarchy-consistent** joint matrices (parent @ local) so LBS does
    not tear the mesh. Bone lengths stay at Mixamo rest; directions follow MP.
    """
    pose35 = _augment_joints(pose33)
    names = skinned["joint_names"]
    name_to_skin = skinned["name_to_skin"]
    rest_heads = skinned["rest_heads"]
    rest_globals = skinned["rest_globals"]
    joint_node_ids = skinned["joint_node_ids"]
    n_joints = len(names)

    mp_hips = pose35[33]
    mp_shoulders = pose35[34]
    hips_si = name_to_skin.get("mixamorig:Hips", 0)
    head_key = "mixamorig:Head" if "mixamorig:Head" in name_to_skin else "mixamorig:Neck"
    mx_hips = rest_heads[hips_si]
    mx_head = rest_heads[name_to_skin.get(head_key, hips_si)]
    if scale is None:
        mp_h = float(np.linalg.norm(mp_shoulders - mp_hips) + 1e-6)
        mx_h = float(np.linalg.norm(mx_head - mx_hips) + 1e-6)
        scale = float(np.clip(mx_h / mp_h, 0.5, 12.0))

    def mp_to_mx(p: np.ndarray) -> np.ndarray:
        return mx_hips + (p - mp_hips) * scale

    # Desired world positions (raw MediaPipe retarget)
    targets = np.zeros((n_joints, 3), dtype=np.float64)
    mapped = np.zeros(n_joints, dtype=bool)
    for name, spec in MIXAMO_TO_MP.items():
        if name not in name_to_skin:
            continue
        si = name_to_skin[name]
        targets[si] = mp_to_mx(_mp_joint(pose35, spec))
        mapped[si] = True

    # Head tip: extend past head along neck→head direction
    if "mixamorig:HeadTop_End" in name_to_skin and "mixamorig:Head" in name_to_skin:
        hi = name_to_skin["mixamorig:Head"]
        ti = name_to_skin["mixamorig:HeadTop_End"]
        neck_i = name_to_skin.get("mixamorig:Neck", hips_si)
        direction = targets[hi] - targets[neck_i]
        dlen = np.linalg.norm(direction)
        rest_len = np.linalg.norm(rest_heads[ti] - rest_heads[hi])
        if dlen > 1e-8:
            targets[ti] = targets[hi] + direction / dlen * max(rest_len, 1e-4)
            mapped[ti] = True

    parents = skinned["parents"]
    node_to_skin = {nid: si for si, nid in enumerate(joint_node_ids)}

    def skin_parent(si: int) -> int | None:
        nid = joint_node_ids[si]
        p = parents.get(nid)
        while p is not None:
            if p in node_to_skin:
                return node_to_skin[p]
            p = parents.get(p)
        return None

    order: list[int] = []
    visited = set()

    def visit(si: int):
        if si in visited:
            return
        sp = skin_parent(si)
        if sp is not None:
            visit(sp)
        visited.add(si)
        order.append(si)

    for si in range(n_joints):
        visit(si)

    # Unmapped joints: rest offset from parent (fingers, etc.)
    for si in order:
        if mapped[si]:
            continue
        sp = skin_parent(si)
        if sp is None:
            targets[si] = rest_heads[si]
            continue
        targets[si] = targets[sp] + (rest_heads[si] - rest_heads[sp])

    # ── Preserve rest bone lengths (direction from MediaPipe, length from Mixamo)
    fixed = np.zeros((n_joints, 3), dtype=np.float64)
    for si in order:
        sp = skin_parent(si)
        if sp is None:
            fixed[si] = targets[si]
            continue
        rest_len = float(np.linalg.norm(rest_heads[si] - rest_heads[sp]) + 1e-12)
        direction = targets[si] - fixed[sp]
        dlen = float(np.linalg.norm(direction))
        if dlen < 1e-8:
            direction = rest_heads[si] - rest_heads[sp]
            dlen = float(np.linalg.norm(direction) + 1e-12)
        fixed[si] = fixed[sp] + direction / dlen * rest_len
    targets = fixed

    # Optional foot plant (off by default — it fights dance motion)
    if plant_feet:
        if floor_y is None:
            foot_idxs = [
                name_to_skin[n]
                for n in ("mixamorig:LeftFoot", "mixamorig:RightFoot")
                if n in name_to_skin
            ]
            floor_y = (
                float(rest_heads[foot_idxs, 1].min())
                if foot_idxs
                else float(targets[:, 1].min())
            )
        targets = _plant_both_feet(targets, rest_heads, name_to_skin, float(floor_y))

    # ── Hierarchy-consistent joint matrices (critical for LBS) ──────────────
    # Previous code set absolute R/T independently → FK mismatch → mesh tears.
    G = np.zeros((n_joints, 4, 4), dtype=np.float64)
    left_si = name_to_skin.get("mixamorig:LeftUpLeg")
    right_si = name_to_skin.get("mixamorig:RightUpLeg")

    for si in order:
        sp = skin_parent(si)
        if sp is None:
            # Root: translate hips to target, keep Mixamo rest orientation.
            # (Re-deriving hip basis from MediaPipe was folding the mesh.)
            G[si] = np.eye(4)
            G[si][:3, :3] = rest_globals[si][:3, :3]
            G[si][:3, 3] = targets[si]
            continue

        rest_local = np.linalg.inv(rest_globals[sp]) @ rest_globals[si]
        rest_local_t = rest_local[:3, 3]
        rest_local_R = rest_local[:3, :3]

        parent_R = G[sp][:3, :3]
        parent_t = G[sp][:3, 3]
        world_offset = targets[si] - parent_t
        local_pos = parent_R.T @ world_offset

        if np.linalg.norm(rest_local_t) > 1e-8 and np.linalg.norm(local_pos) > 1e-8:
            local_R = _rotation_between(rest_local_t, local_pos) @ rest_local_R
        else:
            local_R = rest_local_R

        local_M = np.eye(4)
        local_M[:3, :3] = local_R
        local_M[:3, 3] = local_pos
        G[si] = G[sp] @ local_M

    ibm = skinned["ibm"]
    skin_mats = np.einsum("nij,njk->nik", G, ibm)

    V = skinned["positions"]
    Jidx = skinned["joints"]
    W = skinned["weights"]
    homo = np.concatenate([V, np.ones((len(V), 1))], axis=1)
    out = np.zeros((len(V), 3), dtype=np.float64)
    for k in range(4):
        ji = Jidx[:, k]
        wk = W[:, k][:, None]
        Mk = skin_mats[ji]
        transformed = np.einsum("vij,vj->vi", Mk, homo)
        out += wk * transformed[:, :3]

    parts = [out]
    for att in skinned.get("attachments", []):
        M = G[att["parent_si"]]
        parts.append((M @ att["local"].T).T[:, :3])
    if len(parts) > 1:
        out = np.concatenate(parts, axis=0)
    return out


def animate_skinned(
    poses: np.ndarray,
    skinned: dict,
    out_path: Path,
    *,
    audio: Path | None = None,
    fps: int = POSE_FPS,
    width: int = 720,
    height: int = 1080,
    frames: int | None = None,
    smooth_sigma: float = 1.25,
    upsample: int = 2,
    vert_ema: float = 0.82,
    plant_feet: bool = False,
) -> Path:
    """Drive the Mixamo mesh with MediaPipe bone-LBS so motion matches the skeleton video.

    Mixamo joint LBS was tearing this mesh on dance poses; distance-weighted
    MediaPipe bones follow ``freak_dance_clean`` faithfully.
    """
    if frames is not None:
        poses = poses[:frames]
    if smooth_sigma > 0:
        print(f"  Smoothing poses (sigma={smooth_sigma}) …", flush=True)
        poses = smooth_poses(poses, sigma=smooth_sigma)
    render_fps = fps
    if upsample > 1:
        print(f"  Upsampling poses ×{upsample} → {fps * upsample} fps …", flush=True)
        poses = upsample_poses(poses, factor=upsample)
        render_fps = fps * upsample

    T = poses.shape[0]
    print(f"Poses: {T} frames @ {render_fps} fps (MP→mesh follow)", flush=True)

    name_to_skin = skinned["name_to_skin"]
    rest_heads = skinned["rest_heads"]
    body = np.asarray(skinned["positions"], dtype=np.float64)
    faces = np.asarray(skinned["faces"], dtype=np.int32)
    n_body_faces = int(skinned.get("n_body_faces") or len(faces))

    hips_si = name_to_skin.get("mixamorig:Hips", 0)
    head_key = "mixamorig:Head" if "mixamorig:Head" in name_to_skin else "mixamorig:Neck"
    mx_hips = rest_heads[hips_si].astype(np.float64)
    mx_head = rest_heads[name_to_skin.get(head_key, hips_si)].astype(np.float64)
    mx_torso = float(np.linalg.norm(mx_head - mx_hips) + 1e-6)

    # Bind pose = Mixamo T-pose joints in MediaPipe slot layout (weights match mesh).
    rest35 = np.zeros((35, 3), dtype=np.float64)
    mx_map = {
        0: "mixamorig:Head",
        11: "mixamorig:LeftArm",
        12: "mixamorig:RightArm",
        13: "mixamorig:LeftForeArm",
        14: "mixamorig:RightForeArm",
        15: "mixamorig:LeftHand",
        16: "mixamorig:RightHand",
        23: "mixamorig:LeftUpLeg",
        24: "mixamorig:RightUpLeg",
        25: "mixamorig:LeftLeg",
        26: "mixamorig:RightLeg",
        27: "mixamorig:LeftFoot",
        28: "mixamorig:RightFoot",
        31: "mixamorig:LeftToeBase",
        32: "mixamorig:RightToeBase",
    }
    for mi, name in mx_map.items():
        if name in name_to_skin:
            rest35[mi] = rest_heads[name_to_skin[name]]
    rest35[33] = mx_hips
    if "mixamorig:LeftArm" in name_to_skin and "mixamorig:RightArm" in name_to_skin:
        rest35[34] = 0.5 * (
            rest_heads[name_to_skin["mixamorig:LeftArm"]]
            + rest_heads[name_to_skin["mixamorig:RightArm"]]
        )
    else:
        rest35[34] = mx_head
    for i in range(1, 11):
        rest35[i] = rest35[0]
    for i in (17, 18, 19, 20, 21, 22):
        rest35[i] = rest35[15] if i % 2 else rest35[16]
    rest35[29], rest35[30] = rest35[27], rest35[28]

    print("  Computing MediaPipe bone skin weights on Mixamo bind …", flush=True)
    weights = compute_skin_weights(body, rest35, power=3.5, top_k=3)

    sample = [
        float(np.linalg.norm(_augment_joints(poses[t])[34] - _augment_joints(poses[t])[33]) + 1e-6)
        for t in range(0, T, max(1, T // 64))
    ]
    scale = float(np.clip(mx_torso / float(np.median(sample)), 0.5, 12.0))
    print(f"  Motion scale={scale:.3f}", flush=True)

    def to_mesh_space(pose33: np.ndarray) -> np.ndarray:
        p = _augment_joints(pose33)
        return mx_hips + (p - p[33]) * scale

    # Disco-ball attachment follows nose/head
    head_attach_local: list[np.ndarray] = []
    for att in skinned.get("attachments", []) or []:
        G0 = skinned["rest_globals"][att["parent_si"]]
        world = (G0 @ att["local"].T).T[:, :3]
        head_attach_local.append(world - rest35[0])

    rest_preview = [body]
    for local_off in head_attach_local:
        rest_preview.append(rest35[0] + local_off)
    rest_all = np.concatenate(rest_preview, axis=0)
    center = np.median(rest_all, axis=0).astype(np.float64)
    center[1] = float(np.percentile(rest_all[:, 1], 48))
    radius = float(np.linalg.norm(
        np.percentile(rest_all, 98, axis=0) - np.percentile(rest_all, 2, axis=0)
    )) * 0.68
    radius = max(radius, 1.0)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    silent = out_path.with_suffix(".silent.mp4")
    writer = cv2.VideoWriter(
        str(silent), cv2.VideoWriter_fourcc(*"mp4v"), render_fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {silent}")

    prev = None
    for t in tqdm(range(T), desc="Animating", unit="frame"):
        pose35 = to_mesh_space(poses[t])
        if plant_feet:
            foot_y = float(min(pose35[27, 1], pose35[28, 1]))
            floor = float(min(rest35[27, 1], rest35[28, 1]))
            pose35 = pose35.copy()
            pose35[:, 1] += floor - foot_y

        mats = bone_matrices(rest35, pose35)
        body_posed = apply_lbs(body, weights, mats)

        parts = [body_posed]
        for local_off in head_attach_local:
            parts.append(pose35[0] + local_off)
        verts = np.concatenate(parts, axis=0) if len(parts) > 1 else body_posed

        if prev is not None and 0.0 < vert_ema < 1.0:
            verts = vert_ema * verts + (1.0 - vert_ema) * prev
        prev = verts.copy()

        lo, hi = verts.min(axis=0), verts.max(axis=0)
        c = center.copy()
        c[0] = float(0.5 * (lo[0] + hi[0]))
        c[2] = float(0.5 * (lo[2] + hi[2]))
        c[1] = float(lo[1] + 0.46 * (hi[1] - lo[1]))
        frame = render_frame(
            verts,
            faces,
            width=width,
            height=height,
            center=c,
            radius=radius,
            floor_y=float(lo[1]),
            uvs=skinned.get("uvs"),
            albedo=skinned.get("albedo"),
            normal_tex=skinned.get("normal_tex"),
            metallic=float(skinned.get("metallic", 0.88)),
            roughness=float(skinned.get("roughness", 0.16)),
            disco_metallic=float(skinned.get("disco_metallic", 0.96)),
            disco_roughness=float(skinned.get("disco_roughness", 0.05)),
            n_body_faces=n_body_faces,
            base_color=skinned.get("base_color"),
            exposure=1.4,
        )
        writer.write(frame)
    writer.release()
    print(f"  Silent video: {silent}", flush=True)
    mux_audio(silent, audio, out_path, render_fps)
    print(f"Saved: {out_path}", flush=True)
    return out_path


def animate_mixamo_skinned(
    poses: np.ndarray,
    skinned: dict,
    out_path: Path,
    *,
    audio: Path | None = None,
    fps: int = POSE_FPS,
    width: int = 720,
    height: int = 1080,
    frames: int | None = None,
    smooth_sigma: float = 1.25,
    upsample: int = 2,
    vert_ema: float = 0.82,
    plant_feet: bool = False,
) -> Path:
    """Legacy Mixamo-joint LBS path (kept for experimentation)."""
    if frames is not None:
        poses = poses[:frames]
    if smooth_sigma > 0:
        print(f"  Smoothing poses (sigma={smooth_sigma}) …", flush=True)
        poses = smooth_poses(poses, sigma=smooth_sigma)
    render_fps = fps
    if upsample > 1:
        print(f"  Upsampling poses ×{upsample} → {fps * upsample} fps …", flush=True)
        poses = upsample_poses(poses, factor=upsample)
        render_fps = fps * upsample

    T = poses.shape[0]
    print(f"Poses: {T} frames @ {render_fps} fps (Mixamo joints)", flush=True)
    faces = skinned["faces"]

    name_to_skin = skinned["name_to_skin"]
    rest_heads = skinned["rest_heads"]
    hips_si = name_to_skin.get("mixamorig:Hips", 0)
    head_key = "mixamorig:Head" if "mixamorig:Head" in name_to_skin else "mixamorig:Neck"
    mx_h = float(np.linalg.norm(
        rest_heads[name_to_skin.get(head_key, hips_si)] - rest_heads[hips_si]
    ) + 1e-6)
    torso = []
    for t in range(0, T, max(1, T // 64)):
        p = _augment_joints(poses[t])
        torso.append(float(np.linalg.norm(p[34] - p[33]) + 1e-6))
    scale = float(np.clip(mx_h / float(np.median(torso)), 0.5, 12.0))
    print(f"  Retarget scale={scale:.3f} (stable)", flush=True)

    floor_y = None
    if plant_feet:
        foot_idxs = [
            name_to_skin[n]
            for n in ("mixamorig:LeftFoot", "mixamorig:RightFoot")
            if n in name_to_skin
        ]
        floor_y = float(rest_heads[foot_idxs, 1].min()) if foot_idxs else None

    rest_v = skinned["positions"]
    if skinned.get("attachments"):
        att_rest = []
        for att in skinned["attachments"]:
            G0 = skinned["rest_globals"][att["parent_si"]]
            att_rest.append((G0 @ att["local"].T).T[:, :3])
        rest_v = np.concatenate([rest_v, *att_rest], axis=0)
    center = np.median(rest_v, axis=0).astype(np.float64)
    center[1] = float(np.percentile(rest_v[:, 1], 48))
    radius = float(np.linalg.norm(
        np.percentile(rest_v, 98, axis=0) - np.percentile(rest_v, 2, axis=0)
    )) * 0.68
    radius = max(radius, 1.0)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    silent = out_path.with_suffix(".silent.mp4")
    writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), render_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {silent}")

    prev = None
    for t in tqdm(range(T), desc="Animating", unit="frame"):
        verts = mixamo_skin_frame(
            skinned, poses[t], floor_y=floor_y, scale=scale, plant_feet=plant_feet
        )
        if prev is not None and vert_ema > 0:
            verts = vert_ema * verts + (1.0 - vert_ema) * prev
        prev = verts.copy()
        lo, hi = verts.min(axis=0), verts.max(axis=0)
        c = center.copy()
        c[0] = float(0.5 * (lo[0] + hi[0]))
        c[2] = float(0.5 * (lo[2] + hi[2]))
        c[1] = float(lo[1] + 0.46 * (hi[1] - lo[1]))
        frame = render_frame(
            verts,
            faces,
            width=width,
            height=height,
            center=c,
            radius=radius,
            floor_y=float(lo[1]),
            uvs=skinned.get("uvs"),
            albedo=skinned.get("albedo"),
            normal_tex=skinned.get("normal_tex"),
            metallic=float(skinned.get("metallic", 0.88)),
            roughness=float(skinned.get("roughness", 0.16)),
            disco_metallic=float(skinned.get("disco_metallic", 0.96)),
            disco_roughness=float(skinned.get("disco_roughness", 0.05)),
            n_body_faces=skinned.get("n_body_faces"),
            base_color=skinned.get("base_color"),
            exposure=1.4,
        )
        writer.write(frame)
    writer.release()
    print(f"  Silent video: {silent}", flush=True)
    mux_audio(silent, audio, out_path, render_fps)
    print(f"Saved: {out_path}", flush=True)
    return out_path



def _augment_joints(joints: np.ndarray) -> np.ndarray:
    """(33,3) → (35,3) with virtual mid-hips (-1→33) and mid-shoulders (-2→34)."""
    out = np.zeros((35, 3), dtype=np.float64)
    out[:33] = joints
    out[33] = 0.5 * (joints[23] + joints[24])  # mid hips
    out[34] = 0.5 * (joints[11] + joints[12])  # mid shoulders
    return out


def _resolve_idx(i: int) -> int:
    if i == -1:
        return 33
    if i == -2:
        return 34
    return i


def load_poses(path: Path, *, flip_y: bool = True) -> np.ndarray:
    """Load (T,33,3) or (T,33,9) poses → float64 xyz.

    AIST++/MediaPipe dumps in this project store Y increasing toward the feet
    (see evaluate.py flipping ``-y`` for display). Set flip_y=True to convert
    to standard Y-up for 3D mesh animation.
    """
    arr = np.load(path)
    if arr.ndim != 3 or arr.shape[1] != 33:
        raise ValueError(f"Expected (T,33,C) poses, got {arr.shape}")
    xyz = arr[:, :, :3].astype(np.float64).copy()
    if flip_y:
        xyz[:, :, 1] *= -1.0
    return xyz


def load_mesh(path: Path, max_faces: int = 18_000):
    import trimesh

    loaded = trimesh.load(path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        # Blender glTF often has broken per-node transforms. Prefer the largest
        # raw geometry (usually the body) when scene fusion looks crushed.
        raw = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh) and len(g.faces)]
        fused = None
        try:
            fused = loaded.to_geometry()
        except Exception:
            pass
        if raw:
            raw_sorted = sorted(raw, key=lambda g: float(np.prod(np.maximum(g.extents, 1e-6))), reverse=True)
            body = raw_sorted[0].copy()
            use_raw = True
            if isinstance(fused, trimesh.Trimesh) and len(fused.faces):
                # If fused mesh keeps a plausible human proportion, use it (includes face sphere).
                fe, be = fused.extents, body.extents
                fused_ok = (
                    fe[1] > 0.5 * max(fe[0], fe[2])  # taller than wide-ish
                    and float(np.prod(fe)) > 0.25 * float(np.prod(be))
                )
                if fused_ok:
                    mesh = fused
                    use_raw = False
                else:
                    mesh = body
                    print("  [warn] scene transforms look wrong — using raw body mesh", flush=True)
            else:
                mesh = body
            if use_raw and len(raw_sorted) > 1:
                # Try to attach nearby props from raw space (same object-space as body).
                bc, bs = body.centroid, float(np.linalg.norm(body.extents) + 1e-6)
                extras = []
                for g in raw_sorted[1:]:
                    if float(np.linalg.norm(g.centroid - bc)) < 2.5 * bs:
                        extras.append(g)
                if extras:
                    mesh = trimesh.util.concatenate([mesh, *extras])
        elif isinstance(fused, trimesh.Trimesh):
            mesh = fused
        else:
            raise ValueError(f"No meshes in {path}")
    else:
        mesh = loaded
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Could not load a triangle mesh from {path}")

    mesh = mesh.copy()
    mesh.remove_unreferenced_vertices()
    print(f"  Loaded extents={mesh.extents}", flush=True)
    if len(mesh.faces) > max_faces:
        print(f"  Decimating {len(mesh.faces)} → ~{max_faces} faces …", flush=True)
        try:
            import fast_simplification

            v, f = fast_simplification.simplify(
                np.asarray(mesh.vertices), np.asarray(mesh.faces), target_count=max_faces
            )
            mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
        except Exception as e:
            print(f"  [warn] quadric decimation failed ({e})", flush=True)
        if len(mesh.faces) > max(max_faces * 3, 60_000):
            print(f"  Face subset {len(mesh.faces)} → {max_faces}", flush=True)
            rng = np.random.default_rng(0)
            keep = np.sort(rng.choice(len(mesh.faces), size=max_faces, replace=False))
            mesh.update_faces(keep)
            mesh.remove_unreferenced_vertices()
    print(f"  Mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces", flush=True)
    return mesh


def rest_joints_from_poses(poses: np.ndarray, frame: int = 0) -> np.ndarray:
    """(35,3) rest joints from a reference frame."""
    return _augment_joints(poses[frame])


def skeleton_height(joints35: np.ndarray) -> float:
    return float(joints35[:, 1].max() - joints35[:, 1].min() + 1e-6)


def _orient_mesh_y_up(v: np.ndarray) -> np.ndarray:
    """glTF from Blender is already Y-up — keep as-is."""
    return v


def align_mesh_to_skeleton(mesh, rest35: np.ndarray):
    """Similarity-align a Y-up mesh to a Y-up rest skeleton (hips + height)."""
    v = np.asarray(mesh.vertices, dtype=np.float64).copy()
    v = _orient_mesh_y_up(v)

    # Center on robust mid-point
    mid = np.percentile(v, 50, axis=0)
    v -= mid

    lo = np.percentile(v[:, 1], 2)
    hi = np.percentile(v[:, 1], 98)
    mesh_h = float(hi - lo + 1e-6)
    skel_h = skeleton_height(rest35)
    scale = (0.95 * skel_h) / mesh_h
    v *= scale

    # Feet → ankles, then hips → mid-hips
    mesh_feet_y = float(np.percentile(v[:, 1], 2))
    skel_feet_y = float(rest35[[27, 28], 1].mean())
    v[:, 1] += skel_feet_y - mesh_feet_y

    mesh_hips_y = float(np.percentile(v[:, 1], 42))
    skel_hips = rest35[33]
    v[:, 1] += skel_hips[1] - mesh_hips_y
    v[:, 0] += skel_hips[0] - float(np.percentile(v[:, 0], 50))
    v[:, 2] += skel_hips[2] - float(np.percentile(v[:, 2], 50))

    mesh.vertices = v
    print(
        f"  Aligned mesh scale={scale:.3f}  "
        f"y=[{v[:,1].min():.2f},{v[:,1].max():.2f}]  "
        f"skel_y=[{rest35[:,1].min():.2f},{rest35[:,1].max():.2f}]",
        flush=True,
    )
    return mesh, scale


def smooth_poses(poses: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Gaussian-smooth joint trajectories along time (T,33,3)."""
    if sigma <= 0 or poses.shape[0] < 3:
        return poses
    from scipy.ndimage import gaussian_filter1d

    out = gaussian_filter1d(poses, sigma=sigma, axis=0, mode="nearest")
    return out


def upsample_poses(poses: np.ndarray, factor: int = 2) -> np.ndarray:
    """Linear interpolate poses along time for higher display fps."""
    if factor <= 1 or poses.shape[0] < 2:
        return poses
    T, J, C = poses.shape
    t_src = np.arange(T, dtype=np.float64)
    t_dst = np.linspace(0, T - 1, (T - 1) * factor + 1)
    out = np.empty((len(t_dst), J, C), dtype=np.float64)
    for j in range(J):
        for c in range(C):
            out[:, j, c] = np.interp(t_dst, t_src, poses[:, j, c])
    return out


def pick_rest_frame(poses: np.ndarray) -> int:
    """Choose a near-upright frame (arms/legs not extreme) as bind pose."""
    # Prefer frames where shoulder–hip vertical span is large and wrists aren't wild.
    hips = 0.5 * (poses[:, 23] + poses[:, 24])
    shoulders = 0.5 * (poses[:, 11] + poses[:, 12])
    torso = np.linalg.norm(shoulders - hips, axis=1)
    arm_span = (
        np.linalg.norm(poses[:, 15] - poses[:, 11], axis=1)
        + np.linalg.norm(poses[:, 16] - poses[:, 12], axis=1)
    )
    score = torso - 0.15 * arm_span
    return int(np.argmax(score))


def compute_skin_weights(
    vertices: np.ndarray,
    rest35: np.ndarray,
    *,
    power: float = 3.5,
    eps: float = 1e-4,
    top_k: int = 3,
) -> np.ndarray:
    """Distance-to-bone soft weights → (V, B), rows sum to 1.

    Higher ``power`` and ``top_k`` sparsification reduce candy-wrapper jitter.
    """
    V = vertices.shape[0]
    B = len(BONES)
    dist = np.empty((V, B), dtype=np.float64)

    for bi, (_name, ja, jb) in enumerate(BONES):
        a = rest35[_resolve_idx(ja)]
        b = rest35[_resolve_idx(jb)]
        ab = b - a
        ab2 = float(np.dot(ab, ab)) + 1e-12
        ap = vertices - a
        t = np.clip((ap @ ab) / ab2, 0.0, 1.0)
        closest = a + t[:, None] * ab
        dist[:, bi] = np.linalg.norm(vertices - closest, axis=1)

    inv = 1.0 / (dist**power + eps)
    if top_k is not None and 0 < top_k < B:
        # Keep only the nearest bones per vertex.
        keep = np.argpartition(-inv, top_k - 1, axis=1)[:, :top_k]
        mask = np.zeros_like(inv, dtype=bool)
        rows = np.arange(V)[:, None]
        mask[rows, keep] = True
        inv = np.where(mask, inv, 0.0)
    weights = inv / (inv.sum(axis=1, keepdims=True) + 1e-12)
    return weights.astype(np.float32)


def _bone_rigid(rest_a, rest_b, pose_a, pose_b, *, scale_clamp: tuple[float, float] = (0.88, 1.12)) -> np.ndarray:
    """4x4 rigid transform mapping rest bone segment → posed bone segment."""
    r_dir = rest_b - rest_a
    p_dir = pose_b - pose_a
    r_len = np.linalg.norm(r_dir) + 1e-12
    p_len = np.linalg.norm(p_dir) + 1e-12
    r_u = r_dir / r_len
    p_u = p_dir / p_len

    # Rotation aligning r_u → p_u (Rodrigues).
    v = np.cross(r_u, p_u)
    c = float(np.dot(r_u, p_u))
    if c < -0.999999:
        # 180° — pick an arbitrary perpendicular axis.
        axis = np.array([1.0, 0.0, 0.0])
        if abs(r_u[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = axis - r_u * np.dot(axis, r_u)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        K = np.array(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]],
            dtype=np.float64,
        )
        R = np.eye(3) + 2 * (K @ K)
    else:
        s = np.linalg.norm(v)
        if s < 1e-8:
            R = np.eye(3)
        else:
            K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)
            R = np.eye(3) + K + K @ K * ((1 - c) / (s * s + 1e-12))

    # Clamp bone stretch — large frame-to-frame length jumps cause mesh popping.
    scale = float(np.clip(p_len / r_len, scale_clamp[0], scale_clamp[1]))
    R = R * scale

    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = pose_a - R @ rest_a
    return M


def bone_matrices(rest35: np.ndarray, pose35: np.ndarray) -> np.ndarray:
    """(B, 4, 4) bone transforms."""
    mats = np.zeros((len(BONES), 4, 4), dtype=np.float64)
    for bi, (_name, ja, jb) in enumerate(BONES):
        ia, ib = _resolve_idx(ja), _resolve_idx(jb)
        mats[bi] = _bone_rigid(rest35[ia], rest35[ib], pose35[ia], pose35[ib])
    return mats


def apply_lbs(vertices: np.ndarray, weights: np.ndarray, mats: np.ndarray) -> np.ndarray:
    """(V,3), (V,B), (B,4,4) → (V,3)."""
    V = vertices.shape[0]
    homo = np.concatenate([vertices, np.ones((V, 1))], axis=1)  # (V,4)
    # posed = sum_b w_b * (M_b @ v)
    # (B,V,4) = mats @ homo.T style
    transformed = np.einsum("bij,vj->vbi", mats, homo)  # (V,B,4)
    out = np.einsum("vb,vbi->vi", weights, transformed)[:, :3]
    return out


def _look_at_camera(center: np.ndarray, eye: np.ndarray):
    """Return world→camera 3x3 rotation (rows = cam axes)."""
    forward = center - eye
    forward = forward / (np.linalg.norm(forward) + 1e-12)
    up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(forward, up)) > 0.95:
        up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right) + 1e-12
    up = np.cross(right, forward)
    # Camera looks down -Z in view space: map world → (right, up, -forward)
    R = np.stack([right, up, -forward], axis=0)
    return R


def _smooth_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals → per-face averaged normals (F,3)."""
    tri = vertices[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    vn = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
    return vn[faces].mean(axis=1)


def _env_color(refl: np.ndarray) -> np.ndarray:
    """Warm club / stage HDRI from reflection directions (F,3) → RGB (F,3)."""
    y = np.clip(refl[:, 1] * 0.5 + 0.5, 0.0, 1.0)
    floor_c = np.array([0.08, 0.07, 0.09])
    horizon = np.array([0.42, 0.38, 0.40])
    ceiling = np.array([0.75, 0.72, 0.78])
    t = y[:, None]
    mid = floor_c * (1 - t) + horizon * t
    hi = horizon * (1 - t) + ceiling * t
    sky = np.where(t < 0.5, mid, hi)
    accents = [
        (np.array([0.55, 0.65, 0.35]), np.array([1.0, 0.72, 0.45]), 10.0),   # amber key
        (np.array([-0.55, 0.35, 0.45]), np.array([0.45, 0.65, 1.0]), 9.0),   # cool fill
        (np.array([0.15, 0.25, -0.85]), np.array([1.0, 0.55, 0.75]), 12.0), # magenta rim
        (np.array([0.0, 1.0, 0.1]), np.array([0.9, 0.9, 1.0]), 6.0),        # top soft
    ]
    for direction, color, power in accents:
        d = direction / (np.linalg.norm(direction) + 1e-12)
        w = np.clip((refl @ d) ** power, 0.0, 1.0)[:, None]
        sky = sky + w * color
    return np.clip(sky, 0.0, 1.6)


def _draw_stage_backdrop(width: int, height: int) -> np.ndarray:
    """Dark stage: vertical gradient + soft vignette + floor band."""
    t = np.linspace(0, 1, height)[:, None]  # (H,1)
    top = np.array([18.0, 14.0, 22.0])
    mid = np.array([32.0, 24.0, 28.0])
    bot = np.array([12.0, 10.0, 14.0])
    upper = (1.0 - t) * top + t * mid
    band = np.clip((t - 0.55) / 0.25, 0.0, 1.0)
    lower = (1.0 - band) * mid + band * bot
    row = np.where(t < 0.55, upper, lower)  # (H,3)
    img = np.repeat(row[:, None, :], width, axis=1).astype(np.float32)
    yy = np.linspace(0, 1, height)[:, None]
    xx = np.linspace(0, 1, width)[None, :]
    # Soft spotlight pool on floor center
    cx, cy = 0.5, 0.82
    rx, ry = 0.55, 0.22
    spot = np.exp(-(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2))
    img += spot[:, :, None] * np.array([28.0, 18.0, 22.0])
    # Vignette
    vx = (xx - 0.5) * 1.15
    vy = (yy - 0.45) * 1.05
    vig = np.clip(1.0 - (vx * vx + vy * vy) * 0.85, 0.35, 1.0)
    img *= vig[:, :, None]
    return np.clip(img, 0, 255).astype(np.uint8)


def _project_point(p, eye, Rcam, f, width, height, y_bias=0.55):
    cam = Rcam @ (p - eye)
    z = max(-cam[2], 1e-3)
    u = (cam[0] * f / z) + width * 0.5
    v = (-cam[1] * f / z) + height * y_bias
    return u, v, z


def render_frame(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    width: int = 720,
    height: int = 1080,
    center: np.ndarray | None = None,
    radius: float | None = None,
    light_dir: np.ndarray | None = None,
    uvs: np.ndarray | None = None,
    albedo: np.ndarray | None = None,
    normal_tex: np.ndarray | None = None,
    metallic: float = 0.88,
    roughness: float = 0.16,
    disco_metallic: float = 0.96,
    disco_roughness: float = 0.05,
    n_body_faces: int | None = None,
    base_color: np.ndarray | None = None,
    exposure: float = 1.35,
    floor_y: float | None = None,
    eye: np.ndarray | None = None,
) -> np.ndarray:
    """Stage-lit chrome character with ground contact shadow."""
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    if center is None:
        center = 0.5 * (lo + hi)
        # Bias center slightly up so feet get margin in frame
        center[1] = lo[1] + 0.46 * (hi[1] - lo[1])
    if radius is None:
        radius = float(np.linalg.norm(hi - lo) * 0.62)
    radius = max(radius, 0.5)
    if base_color is None:
        base_color = np.array([0.84, 0.88, 0.94], dtype=np.float64)
    if floor_y is None:
        floor_y = float(lo[1])

    if eye is None:
        # 3/4 front view, pulled back to fit full body
        eye = center + np.array([0.55 * radius, 0.12 * radius, 2.55 * radius])

    Rcam = _look_at_camera(center, eye)
    cam = (vertices - eye) @ Rcam.T

    # Focal length: fit vertical extent with ~10% margin
    y_span = float(hi[1] - lo[1] + 1e-6)
    f = 0.90 * height * abs(float(np.median(-cam[:, 2]))) / (y_span * 0.55 + 1e-6)
    f = float(np.clip(f, 0.7 * height, 2.2 * height))

    z = np.clip(-cam[:, 2], 1e-3, None)
    screen = np.stack(
        [
            (cam[:, 0] * f / z) + width * 0.5,
            (-cam[:, 1] * f / z) + height * 0.58,
        ],
        axis=1,
    )

    F = len(faces)
    if n_body_faces is None:
        n_body_faces = F

    n_smooth = _smooth_face_normals(vertices, faces)
    tri = vertices[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    n = n_smooth.copy()
    if n_body_faces < F:
        n[n_body_faces:] = fn[n_body_faces:]

    cent_cam = cam[faces].mean(axis=1)
    view_w = eye - tri.mean(axis=1)
    view_w /= np.linalg.norm(view_w, axis=1, keepdims=True) + 1e-12
    visible = (n * view_w).sum(axis=1) > -0.02

    tint = np.broadcast_to(base_color, (F, 3)).astype(np.float64).copy()
    if uvs is not None and albedo is not None and len(uvs) == len(vertices):
        face_uv = uvs[faces].mean(axis=1)
        sampled = _sample_texture(albedo, face_uv[:, 0], face_uv[:, 1])
        luma = sampled.mean(axis=1, keepdims=True)
        tint = np.clip(base_color * (0.80 + 0.35 * luma), 0.0, 1.0)

    if uvs is not None and normal_tex is not None and len(uvs) == len(vertices):
        face_uv = uvs[faces].mean(axis=1)
        nmap = _sample_texture(normal_tex, face_uv[:, 0], face_uv[:, 1]) * 2.0 - 1.0
        boost = np.zeros((F, 1))
        boost[:n_body_faces] = 0.22
        n = n + boost * nmap
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12

    met = np.full(F, metallic, dtype=np.float64)
    rough = np.full(F, roughness, dtype=np.float64)
    if n_body_faces < F:
        met[n_body_faces:] = disco_metallic
        rough[n_body_faces:] = disco_roughness
        tint[n_body_faces:] = np.array([0.94, 0.95, 0.98])

    ndotv = np.clip((n * view_w).sum(axis=1), 0.0, 1.0)
    refl = 2.0 * ndotv[:, None] * n - view_w
    refl /= np.linalg.norm(refl, axis=1, keepdims=True) + 1e-12
    env = _env_color(refl)

    if light_dir is not None:
        lights = [(light_dir / (np.linalg.norm(light_dir) + 1e-12), 1.0, np.ones(3))]
    else:
        lights = [
            (np.array([0.50, 0.85, 0.35]), 1.05, np.array([1.08, 0.98, 0.88])),
            (np.array([-0.60, 0.40, 0.45]), 0.50, np.array([0.70, 0.82, 1.15])),
            (np.array([0.05, 0.40, -0.90]), 0.65, np.array([1.15, 0.70, 0.95])),
        ]
        lights = [(d / (np.linalg.norm(d) + 1e-12), w, c) for d, w, c in lights]

    diffuse_acc = np.full(F, 0.20, dtype=np.float64)
    spec_rgb = np.zeros((F, 3), dtype=np.float64)
    for Ldir, weight, col in lights:
        ndotl = np.clip((n * Ldir).sum(axis=1), 0.0, 1.0)
        wrap = np.clip(ndotl * 0.75 + 0.25, 0.0, 1.0)
        diffuse_acc += weight * 0.38 * wrap
        half = Ldir + view_w
        half /= np.linalg.norm(half, axis=1, keepdims=True) + 1e-12
        ndoth = np.clip((n * half).sum(axis=1), 0.0, 1.0)
        gloss = np.maximum(10.0, (1.0 - rough) * 110.0)
        spec = np.power(ndoth, gloss) * weight
        spec_rgb += spec[:, None] * col[None, :]

    dielectric = tint * diffuse_acc[:, None]
    metal = env * tint
    lit = dielectric * (1.0 - met)[:, None] + metal * met[:, None]
    lit = lit + spec_rgb * (0.22 + 0.78 * met[:, None])
    fresnel = np.power(1.0 - ndotv, 2.8)
    lit = lit + fresnel[:, None] * env * (0.12 + 0.40 * met[:, None])

    lit = lit * exposure
    lit = lit / (1.0 + lit * 0.55)
    lit = np.clip(lit, 0.0, 1.0)
    colors = (lit * 255.0).astype(np.uint8)[:, ::-1].copy()

    img = _draw_stage_backdrop(width, height)

    # Ground disc + soft contact shadow (projected)
    ground_r = radius * 1.15
    gx = np.linspace(center[0] - ground_r, center[0] + ground_r, 48)
    gz = np.linspace(center[2] - ground_r, center[2] + ground_r, 48)
    # Shadow ellipse under character
    foot_c = np.array([float(np.median(vertices[:, 0])), floor_y, float(np.median(vertices[:, 2]))])
    shadow_pts = []
    for ang in np.linspace(0, 2 * np.pi, 36, endpoint=False):
        px = foot_c[0] + np.cos(ang) * radius * 0.38
        pz = foot_c[2] + np.sin(ang) * radius * 0.22
        u, v, _ = _project_point(np.array([px, floor_y, pz]), eye, Rcam, f, width, height, 0.58)
        shadow_pts.append([u, v])
    shadow = np.array(shadow_pts, dtype=np.int32)
    overlay = img.copy()
    cv2.fillConvexPoly(overlay, shadow, color=(8, 6, 10))
    img = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)
    # Thin ground ring
    ring = []
    for ang in np.linspace(0, 2 * np.pi, 64, endpoint=False):
        px = foot_c[0] + np.cos(ang) * radius * 0.85
        pz = foot_c[2] + np.sin(ang) * radius * 0.55
        u, v, _ = _project_point(np.array([px, floor_y, pz]), eye, Rcam, f, width, height, 0.58)
        ring.append([u, v])
    cv2.polylines(img, [np.array(ring, dtype=np.int32)], True, (48, 40, 52), 1, cv2.LINE_AA)

    screen_i = screen.astype(np.int32)
    faces_v = faces[visible]
    cols = colors[visible]
    depths = cent_cam[visible, 2]
    for fi in np.argsort(depths):
        pts = screen_i[faces_v[fi]]
        if pts[:, 0].min() >= width or pts[:, 0].max() < 0:
            continue
        if pts[:, 1].min() >= height or pts[:, 1].max() < 0:
            continue
        cv2.fillConvexPoly(img, pts, color=tuple(int(c) for c in cols[fi]), lineType=cv2.LINE_AA)

    # Soft bloom on highlights
    bright = np.clip(img.astype(np.float32) - 185.0, 0, None)
    if bright.max() > 1:
        bloom = cv2.GaussianBlur(bright, (0, 0), 5.0)
        img = np.clip(img.astype(np.float32) + bloom * 0.55, 0, 255).astype(np.uint8)

    # Final vignette pass
    yy = np.linspace(-1, 1, height)[:, None]
    xx = np.linspace(-1, 1, width)[None, :]
    vig = np.clip(1.15 - 0.55 * (xx * xx * 0.9 + yy * yy), 0.55, 1.0)
    img = np.clip(img.astype(np.float32) * vig[:, :, None], 0, 255).astype(np.uint8)
    return img


def mux_audio(video_silent: Path, audio: Path | None, out: Path, fps: int) -> None:
    """Re-encode to H.264 (smoother playback than raw mp4v) and optional AAC mux."""
    if audio is not None and audio.exists():
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_silent),
            "-i", str(audio),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            "-r", str(fps),
            str(out),
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_silent),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-r", str(fps),
            str(out),
        ]
    print("  Encoding H.264 …", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [warn] ffmpeg encode failed: {r.stderr[-500:]}", flush=True)
        if video_silent.resolve() != out.resolve():
            os.replace(video_silent, out)
    elif video_silent.resolve() != out.resolve() and video_silent.exists():
        video_silent.unlink(missing_ok=True)


def animate(
    poses: np.ndarray,
    mesh_path: Path,
    out_path: Path,
    *,
    audio: Path | None = None,
    fps: int = POSE_FPS,
    max_faces: int = 18_000,
    width: int = 720,
    height: int = 1080,
    rest_frame: int | None = None,
    frames: int | None = None,
    smooth_sigma: float = 1.25,
    upsample: int = 2,
    vert_ema: float = 0.82,
    plant_feet: bool = False,
) -> Path:
    skinned = try_load_skinned_glb(mesh_path)
    if skinned is not None:
        return animate_skinned(
            poses,
            skinned,
            out_path,
            audio=audio,
            fps=fps,
            width=width,
            height=height,
            frames=frames,
            smooth_sigma=smooth_sigma,
            upsample=upsample,
            vert_ema=vert_ema,
            plant_feet=plant_feet,
        )

    if frames is not None:
        poses = poses[:frames]

    if smooth_sigma > 0:
        print(f"  Smoothing poses (sigma={smooth_sigma}) …", flush=True)
        poses = smooth_poses(poses, sigma=smooth_sigma)

    render_fps = fps
    if upsample > 1:
        print(f"  Upsampling poses ×{upsample} → {fps * upsample} fps …", flush=True)
        poses = upsample_poses(poses, factor=upsample)
        render_fps = fps * upsample

    T = poses.shape[0]
    print(f"Poses: {T} frames @ {render_fps} fps", flush=True)

    if rest_frame is None:
        rest_frame = pick_rest_frame(poses)
        print(f"  Auto rest frame: {rest_frame}", flush=True)

    mesh = load_mesh(mesh_path, max_faces=max_faces)
    rest35 = rest_joints_from_poses(poses, frame=min(rest_frame, T - 1))
    mesh, _ = align_mesh_to_skeleton(mesh, rest35)
    rest_verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    print("  Computing skin weights …", flush=True)
    weights = compute_skin_weights(rest_verts, rest35)

    # Camera from rest mesh + typical hip height (ignore pose outliers)
    all_joints = np.stack([_augment_joints(poses[t]) for t in range(T)])
    hips = all_joints[:, 33, :]
    center = np.median(hips, axis=0).astype(np.float64)
    center[1] = float(np.percentile(rest_verts[:, 1], 55))  # mid-torso
    mesh_span = float(np.linalg.norm(
        np.percentile(rest_verts, 98, axis=0) - np.percentile(rest_verts, 2, axis=0)
    ))
    radius = max(mesh_span * 0.55, skeleton_height(rest35) * 0.65)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    silent = out_path.with_suffix(".silent.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(silent), fourcc, render_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {silent}")

    prev_verts = None
    for t in tqdm(range(T), desc="Animating", unit="frame"):
        pose35 = _augment_joints(poses[t])
        mats = bone_matrices(rest35, pose35)
        verts = apply_lbs(rest_verts, weights, mats)
        if prev_verts is not None and vert_ema > 0:
            # Temporal EMA damps LBS popping between frames.
            verts = vert_ema * verts + (1.0 - vert_ema) * prev_verts
        prev_verts = verts
        frame = render_frame(
            verts, faces, width=width, height=height, center=center, radius=radius
        )
        writer.write(frame)
    writer.release()
    print(f"  Silent video: {silent}", flush=True)

    mux_audio(silent, audio, out_path, render_fps)
    print(f"Saved: {out_path}", flush=True)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--poses", type=Path, required=True, help="(T,33,3|9) .npy")
    p.add_argument("--mesh", type=Path, required=True, help="Character .glb/.obj")
    p.add_argument("--out", type=Path, default=Path("outputs/character_dance.mp4"))
    p.add_argument("--audio", type=Path, default=None)
    p.add_argument("--fps", type=int, default=POSE_FPS)
    p.add_argument("--max-faces", type=int, default=18_000)
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--rest-frame", type=int, default=None,
                   help="Bind-pose frame (default: auto upright)")
    p.add_argument("--frames", type=int, default=None, help="Optional frame cap")
    p.add_argument("--smooth", type=float, default=1.25,
                   help="Temporal Gaussian sigma on poses (0=off)")
    p.add_argument("--upsample", type=int, default=2,
                   help="Pose time upsample factor for smoother playback")
    p.add_argument("--vert-ema", type=float, default=0.82,
                   help="Vertex EMA blend with previous frame (1=off / full current)")
    p.add_argument("--plant-feet", action="store_true",
                   help="IK-plant feet on floor (off by default — better dance fidelity)")
    args = p.parse_args()

    poses = load_poses(args.poses)
    animate(
        poses,
        args.mesh,
        args.out,
        audio=args.audio,
        fps=args.fps,
        max_faces=args.max_faces,
        width=args.width,
        height=args.height,
        rest_frame=args.rest_frame,
        frames=args.frames,
        smooth_sigma=args.smooth,
        upsample=args.upsample,
        vert_ema=args.vert_ema,
        plant_feet=args.plant_feet,
    )


if __name__ == "__main__":
    main()

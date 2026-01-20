from __future__ import annotations

import numpy as np
from .exponential import Exponential


from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

##################################################
#
# Util function
# Dongwook Kwak and Jiook Chung
# Last modified: 2026.01.06
#
##################################################

@staticmethod
def compute_constraint_matrix(loop_screws: list[np.ndarray],
                              closed_loop_crease_idx: list[list[int]],
                              theta: np.ndarray) -> np.ndarray:
    """
    Computes the Pfaffian constraint matrix A for a list of closed loops.
    Automatically chooses between full screw (6D) or angular-only (3D) Jacobian.

    Args:
        loop_screws (list of np.ndarray): Each (3 or 6, n_i) screw matrix per loop.
        closed_loop_crease_idx (list of list of int): Indices of creases in each loop.
        theta (np.ndarray): (n,) full joint angle vector.

    Returns:
        A (np.ndarray): (sum_k d_k, n) constraint matrix where d_k = 3 or 6.
    """
    A_blocks = []
    n = len(theta)

    for screw, crease_idx in zip(loop_screws, closed_loop_crease_idx):
        angle = theta[crease_idx]

        if screw.shape[0] == 3:
            Js = Exponential.trunc_jacobian_space(screw, angle)  # (3, n_i)
            A_block = np.zeros((3, n))
        elif screw.shape[0] == 6:
            Js = Exponential.jacobian_space(screw, angle)        # (6, n_i)
            A_block = np.zeros((6, n))
        else:
            raise ValueError("Each screw must be (3, n_i) or (6, n_i).")

        A_block[:, crease_idx] = Js
        A_blocks.append(A_block)

    A = np.vstack(A_blocks)
    return A

def get_centroids(vertices, faces):
    centroids = np.zeros((len(faces), 3))
    for i, face in enumerate(faces):
        x = vertices[face, 0]
        y = vertices[face, 1]
        n = len(face)
        
        area = 0
        cx = 0
        cy = 0

        for j in range(n):
            x0, y0 = x[j], y[j]
            x1, y1 = x[(j + 1) % n], y[(j + 1) % n]

            cross = x0 * y1 - x1 * y0
            area += cross

            cx += (x0 + x1) * cross
            cy += (y0 + y1) * cross

        area *= 0.5
        cx /= (6 * area)
        cy /= (6 * area)

        centroids[i,0] = cx
        centroids[i,1] = cy
        centroids[i,2] = 0
    return centroids

def _as_list(x):
    return x if isinstance(x, list) else list(x)

def _to_np(a, dtype=float):
    return np.asarray(a, dtype=dtype)


####################### Pattern Visualizer ##################
def _centroid_of_polygon(pts_xyz: np.ndarray) -> np.ndarray:
    # pts_xyz: (m, 3)
    return np.mean(pts_xyz, axis=0)

def _resolve_schema(rfs):
    """
    RigidFoldableStructure(GeonHee Cho ver.) compatible resolver.

    Returns:
      V_sheets: list of (Nv_i, 3)
      F_sheets: list of list of faces (each face is list[int] local to that sheet)
      E_sheets: optional (unused)
      C: connections list
    """

    # dict input
    if isinstance(rfs, dict):
        V = rfs.get("V", rfs.get("vert", rfs.get("verts_raw", None)))
        F = rfs.get("F", rfs.get("face", rfs.get("faces_raw", None)))
        E = rfs.get("E", rfs.get("edge", rfs.get("edges_raw", None)))
        C = rfs.get("C", rfs.get("connection", []))
        vertices_flat = rfs.get("vertices", None)
        faces_flat = rfs.get("faces", None)

    # object input
    else:
        # Prefer sheet/raw if available (needed for sheet-colored visualization)
        V = getattr(rfs, "verts_raw", None)
        F = getattr(rfs, "faces_raw", None)
        E = getattr(rfs, "edges_raw", None)
        C = getattr(rfs, "connection", [])
        vertices_flat = getattr(rfs, "vertices", None)
        faces_flat = getattr(rfs, "faces", None)

    # If sheet/raw exists, use it
    if V is not None and F is not None:
        V_sheets = [np.asarray(v, dtype=float) for v in list(V)]
        F_sheets = [list(f) for f in list(F)]
        E_sheets = list(E) if E is not None else []
        C = list(C) if C is not None else []
        return V_sheets, F_sheets, E_sheets, C

    # Fallback: if only flattened mesh is available, wrap as single sheet
    if vertices_flat is not None and faces_flat is not None:
        V0 = np.asarray(vertices_flat, dtype=float)
        F0 = [list(map(int, f)) for f in faces_flat]
        return [V0], [F0], [], list(C) if C is not None else []

    raise ValueError("pattern_visualizer: cannot resolve schema from given input.")


def _sheet_palette_rgba(n: int, alpha: float = 0.35) -> List[Tuple[float, float, float, float]]:
    """
    Deterministic distinct colors using matplotlib tab20.
    Same sheet -> same color, different sheet -> different color.
    """
    cmap = plt.get_cmap("tab20")
    colors = []
    for i in range(n):
        r, g, b, _ = cmap(i % 20)
        colors.append((r, g, b, alpha))
    return colors
# -----------------------------
# Main visualizer
# -----------------------------
def pattern_visualizer(
    rfs: Any,
    sheets: Union[str, Sequence[int]] = "all",
    show_connections: bool = True,
    connection_style: Optional[Dict[str, Dict[str, Any]]] = None,
    show_edges: bool = True,
    show_faces: bool = True,
    show_vertices: bool = False,
    edge_linewidth: float = 0.8,
    vertex_size: float = 8.0,
    ax: Optional[Any] = None,
    title: Optional[str] = None,
    vertex_text_kwargs: Optional[Dict[str, Any]] = None,
    edge_text_kwargs: Optional[Dict[str, Any]] = None,
    face_text_kwargs: Optional[Dict[str, Any]] = None,
):
    V_sheets, F_sheets, E_sheets, C = _resolve_schema(rfs)

    ns = len(V_sheets)
    sheet_ids = list(range(ns)) if sheets == "all" else list(map(int, sheets))

    if connection_style is None:
        connection_style = {
            "hinge": {"linestyle": (0, (3, 3)), "linewidth": 1.5},
            "solder": {"linestyle": "-", "linewidth": 2.0},
        }

    # default text styles
    vertex_text_kwargs = vertex_text_kwargs or dict(color="red", fontsize=9)
    edge_text_kwargs   = edge_text_kwargs   or dict(color="green", fontsize=9)
    face_text_kwargs   = face_text_kwargs   or dict(color="blue", fontsize=10, fontweight="bold")

    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    colors = _sheet_palette_rgba(ns, alpha=0.35)
    all_pts = []

    # -------------------------
    # Plot sheets
    # -------------------------
    for sid in sheet_ids:
        V = np.asarray(V_sheets[sid])
        F = F_sheets[sid]
        E = E_sheets[sid] if sid < len(E_sheets) else []

        polys = []
        for fid, face in enumerate(F):
            pts = V[list(map(int, face))]
            polys.append(pts)
            all_pts.append(pts)

            # draw face boundary
            if show_edges:
                closed = np.vstack([pts, pts[0]])
                ax.plot(closed[:, 0], closed[:, 1], closed[:, 2],
                        linewidth=edge_linewidth,
                        color=(colors[sid][0], colors[sid][1], colors[sid][2], 0.9))

            # face index
            if show_faces:
                c = pts.mean(axis=0)
                ax.text(c[0], c[1], c[2], f"F{fid}", **face_text_kwargs)

        ax.add_collection3d(
            Poly3DCollection(polys, facecolors=colors[sid], edgecolors="none")
        )

        # vertex indices
        if show_vertices:
            ax.scatter(V[:, 0], V[:, 1], V[:, 2], s=vertex_size, color="k")
            for vidx, (x, y, z) in enumerate(V):
                ax.text(x, y, z, f"V{vidx}", **vertex_text_kwargs)

        # edge indices
        if show_edges and E:
            for eidx, (v0, v1) in enumerate(E):
                p0, p1 = V[v0], V[v1]
                mid = 0.5 * (p0 + p1)
                ax.text(mid[0], mid[1], mid[2], f"E{eidx}", **edge_text_kwargs)

    # -------------------------
    # Connections
    # -------------------------
    if show_connections and C:
        centroids = {
            (sid, fid): _centroid_of_polygon(np.asarray(V_sheets[sid])[face])
            for sid in range(ns)
            for fid, face in enumerate(F_sheets[sid])
        }

        for (s1, f1), (s2, f2), ctype in C:
            if s1 not in sheet_ids or s2 not in sheet_ids:
                continue
            p1, p2 = centroids[(s1, f1)], centroids[(s2, f2)]
            st = connection_style["hinge"] if ctype in ("h", "r") else connection_style["solder"]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color="k", **st)

    # -------------------------
    # Axis scaling
    # -------------------------
    if all_pts:
        P = np.vstack(all_pts)
        mid = (P.max(0) + P.min(0)) / 2
        r = max(P.max(0) - P.min(0)) / 2
        ax.set_xlim(mid[0] - r, mid[0] + r)
        ax.set_ylim(mid[1] - r, mid[1] + r)
        ax.set_zlim(mid[2] - r, mid[2] + r)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title or "RFS Pattern Visualization")
    ax.grid(False)

    plt.tight_layout()
    return fig, ax

########## Pattern export and import ############
def save_rfs_artifact_npz(
    filepath: str,
    V, E, F, C,
    theta_init_raw,
    theta_neutral_raw,
    face_colors,
    k_stiffness_raw,
    reference_frame,
    meta: Optional[Dict[str, Any]] = None,
):
    """
    Save only the public artifact needed for examples (no generator code).

    Notes
    -----
    - V/E/F/C are typically list-of-sheets (python lists).
    - theta_*_raw and k_stiffness_raw are typically (sheet, edge) arrays (or lists).
    - face_colors is (num_faces, 4) uint8 or list.
    - reference_frame can be any small array/list (e.g., 4x4).
    """
    payload = dict(
        V=np.array(V, dtype=object),
        E=np.array(E, dtype=object),
        F=np.array(F, dtype=object),
        C=np.array(C, dtype=object),

        theta_init_raw   = np.array(theta_init_raw, dtype=object),
        theta_neutral_raw = np.array(theta_neutral_raw, dtype=object),
        k_stiffness_raw   = np.array(k_stiffness_raw, dtype=object),

        face_colors=np.asarray(face_colors, dtype=np.uint8),
        reference_frame = (
            np.array(reference_frame, dtype=object)
            if reference_frame is not None
            else np.array(None, dtype=object)
        ),

        meta=np.array(meta if meta is not None else {}, dtype=object),
    )
    np.savez_compressed(filepath, **payload)
    print(f"[*] Saved artifact -> {filepath}")

def load_rfs_artifact_npz(filepath: str) -> Dict[str, Any]:
    """
    Load the public artifact and return python-native structures.
    """
    data = np.load(filepath, allow_pickle=True)

    ref_raw = data["reference_frame"]

    if ref_raw.dtype == object and ref_raw.shape == () and ref_raw.item() is None:
        reference_frame = None
    else:
        reference_frame = ref_raw.tolist()

    artifact = {
        "V": data["V"].tolist(),
        "E": data["E"].tolist(),
        "F": data["F"].tolist(),
        "C": data["C"].tolist(),

        "theta_init_raw": data["theta_init_raw"].tolist(),
        "theta_neutral_raw": data["theta_neutral_raw"].tolist(),
        "k_stiffness_raw": data["k_stiffness_raw"].tolist(),

        "face_colors": data["face_colors"],
        "reference_frame": data["reference_frame"].tolist(),

        "meta": data["meta"].item() if "meta" in data.files else {},
    }
    print(f"[*] Loaded artifact <- {filepath}")
    return artifact
"""
sfm.py  -  Structure from motion for a FLAT (planar) object seen from 4 views (Part 2)

HOW TO RUN:   python sfm.py            (uses sfm_data/cameras.json)

Pipeline (math in docs/DERIVATIONS.md):
 1. For each view, estimate the homography H (plane -> image) with the DLT (SVD).
 2. Decompose  H = K [r1 r2 t]  ->  camera rotation R and position C.
 3. Triangulate every point from all 4 views (linear DLT) -> recovered 3D points.
 4. Order the points (boundary_order) -> boundary, area, perimeter. Compare with ruler values.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "sfm_data/cameras.json"


def dlt_homography(world_xy, img_xy):
    """Solve h from  [X Y 1 0 0 0 -xX -xY -x ; 0 0 0 X Y 1 -yX -yY -y] h = 0  using SVD."""
    rows = []
    for (X, Y), (x, y) in zip(world_xy, img_xy):
        rows.append([X, Y, 1, 0, 0, 0, -x * X, -x * Y, -x])
        rows.append([0, 0, 0, X, Y, 1, -y * X, -y * Y, -y])
    _, _, Vt = np.linalg.svd(np.array(rows))
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]


def decompose_homography(H, K):
    """H = lambda K [r1 r2 t]  ->  R, t."""
    M = np.linalg.inv(K) @ H
    lam = 1.0 / np.linalg.norm(M[:, 0])
    r1, r2, t = lam * M[:, 0], lam * M[:, 1], lam * M[:, 2]
    if t[2] < 0:                                   # plane must be in front of the camera
        r1, r2, t = -r1, -r2, -t
    R = np.stack([r1, r2, np.cross(r1, r2)], axis=1)
    U, _, Vt = np.linalg.svd(R)                    # clean R so that R^T R = I
    return U @ Vt, t


def triangulate(Ps, pts2d):
    """Linear triangulation from N views: rows  x*P3 - P1  and  y*P3 - P2."""
    A = []
    for P, (x, y) in zip(Ps, pts2d):
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.array(A))
    X = Vt[-1]
    return X[:3] / X[3]


def project(P, X):
    p = P @ np.append(X, 1)
    return p[:2] / p[2]


def shoelace(xy):
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def run(data_path=DATA, out_png="outputs/sfm_result.png"):
    os.makedirs("outputs", exist_ok=True)
    cfg = json.load(open(data_path))
    K = np.array(cfg["K"], float)
    names = list(cfg["world_points_cm"].keys())
    W = np.array([cfg["world_points_cm"][n] for n in names], float)      # ruler measurements (Z = 0)
    Ps, cams, reproj = [], [], []
    for v in cfg["views"]:
        px = np.array([v["points_px"][n] for n in names], float)
        H = dlt_homography(W, px)
        R, t = decompose_homography(H, K)
        P = K @ np.hstack([R, t[:, None]])
        Ps.append(P)
        C = -R.T @ t                                                     # camera centre in world coords
        err = np.mean([np.linalg.norm(project(P, np.append(w, 0)) - p) for w, p in zip(W, px)])
        cams.append({"image": v["image"], "note": v.get("camera_note", ""),
                     "H": np.round(H, 5).tolist(), "R": np.round(R, 4).tolist(),
                     "t_cm": np.round(t, 2).tolist(), "camera_position_cm": np.round(C, 2).tolist(),
                     "mean_reprojection_error_px": round(float(err), 3)})
        reproj.append(err)
    # triangulate each point from all views (uses ONLY pixel observations + the recovered cameras)
    X = np.array([triangulate(Ps, [v["points_px"][n] for v in cfg["views"]]) for n in names])
    order = [names.index(n) for n in cfg["boundary_order"]]
    est_area, true_area = shoelace(X[order, :2]), shoelace(W[order])
    per_pt = np.linalg.norm(X[:, :2] - W, axis=1)
    # plot: top-down boundary + 3D cameras
    fig = plt.figure(figsize=(11, 4.5))
    ax = fig.add_subplot(1, 2, 1)
    cl = order + [order[0]]
    ax.plot(W[cl, 0], W[cl, 1], "g--", label="ruler (true)")
    ax.plot(X[cl, 0], X[cl, 1], "r.-", label="recovered")
    for n, p in zip(names, X):
        ax.annotate(n, p[:2])
    ax.set_aspect("equal"); ax.legend(); ax.set_title("Recovered boundary (cm)")
    a3 = fig.add_subplot(1, 2, 2, projection="3d")
    a3.plot(X[cl, 0], X[cl, 1], X[cl, 2], "r.-")
    for i, c in enumerate(cams):
        p = c["camera_position_cm"]; a3.scatter(*p, c="b"); a3.text(*p, f"cam{i+1}")
    a3.set_title("Cameras and object")
    fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)
    return {"image": "/" + out_png, "cameras": cams,
            "recovered_points_cm": {n: np.round(p, 2).tolist() for n, p in zip(names, X)},
            "point_error_cm": {n: round(float(e), 3) for n, e in zip(names, per_pt)},
            "area_estimated_cm2": round(float(est_area), 2), "area_true_cm2": round(float(true_area), 2),
            "K": K.tolist()}


if __name__ == "__main__":
    r = run()
    print(json.dumps({k: v for k, v in r.items() if k != "cameras"}, indent=1))
    for c in r["cameras"]:
        print(c["image"], "cam pos", c["camera_position_cm"], "reproj err", c["mean_reprojection_error_px"])

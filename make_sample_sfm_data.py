"""Creates a SAMPLE sfm_data/cameras.json from a synthetic camera setup so we can test the pipeline."""
import json, numpy as np
rng = np.random.default_rng(1)
K = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1.0]])
pts = {"A": [0, 0], "B": [21, 0], "C": [21, 29.7], "D": [0, 29.7], "E": [10.5, 14.8]}   # A4 sheet, cm
def look_at(C, target=np.array([10.5, 14.8, 0])):
    z = target - C; z /= np.linalg.norm(z)
    x = np.cross([0, -1, 0], z); x /= np.linalg.norm(x); y = np.cross(z, x)
    R = np.stack([x, y, z]); return R, -R @ C
views = []
for i, C in enumerate([[10, -40, 60], [60, 10, 55], [10, 70, 60], [-40, 15, 55]]):
    R, t = look_at(np.array(C, float))
    px = {}
    for n, (X, Y) in pts.items():
        p = K @ (R @ np.array([X, Y, 0]) + t); p = p[:2] / p[2] + rng.normal(0, 0.6, 2)
        px[n] = [round(float(v), 1) for v in p]
    views.append({"image": f"view{i+1}.jpg", "camera_note": f"camera approx at {C} cm", "points_px": px})
json.dump({"K": K.tolist(), "world_points_cm": pts, "boundary_order": ["A", "B", "C", "D"], "views": views},
          open("sfm_data/cameras.json", "w"), indent=1)
print("wrote sfm_data/cameras.json")

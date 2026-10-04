"""
calibrate.py - get camera matrix K with a printed checkerboard (Zhang's method).
Take 10+ photos of the checkerboard from different angles with the SAME camera/zoom you use for the object,
put them in sfm_data/calib/, then:  python calibrate.py 9 6 2.5
   9 6 = inner corners (columns rows),  2.5 = square size in cm
"""
import sys, glob, cv2, numpy as np
cols, rows, sq = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
obj = np.zeros((rows * cols, 3), np.float32); obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * sq
op, ip = [], []
for f in glob.glob("sfm_data/calib/*"):
    g = cv2.imread(f, 0)
    if g is None: continue
    ok, c = cv2.findChessboardCorners(g, (cols, rows))
    if ok: op.append(obj); ip.append(c); size = g.shape[::-1]
rms, K, dist, _, _ = cv2.calibrateCamera(op, ip, size, None, None)
print("images used:", len(op), "RMS error:", rms); print("K =", K.tolist()); print("distortion =", dist.ravel().tolist())

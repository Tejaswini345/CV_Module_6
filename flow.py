"""
flow.py  -  Optical flow + manual Lucas-Kanade tracking 

HOW TO RUN (standalone, no web page needed):
    python flow.py videos/my_video.mp4
The web demo (app.py) calls the same functions.

Contents
  dense_flow_video()   -> Farneback dense optical flow, saved as a side-by-side video
  bilinear()           -> bilinear interpolation (derived in docs/DERIVATIONS.md)
  lk_point()           -> Lucas-Kanade written by hand for ONE point (iterative, uses bilinear)
  validate_tracking()  -> compare my LK vs. OpenCV LK on two consecutive frames
"""
import os, sys, shutil, subprocess
import cv2
import numpy as np

OUT = "outputs"


def _resize(frame, width=480):
    h, w = frame.shape[:2]
    s = width / w
    return cv2.resize(frame, (width, int(h * s))) if s < 1 else frame


def flow_to_color(flow):
    """Direction -> hue, speed -> brightness."""
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*flow.shape[:2], 3), np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), mag


def draw_arrows(img, flow, step=24):
    out = img.copy()
    for y in range(step // 2, img.shape[0], step):
        for x in range(step // 2, img.shape[1], step):
            dx, dy = flow[y, x]
            if dx * dx + dy * dy > 0.5:
                cv2.arrowedLine(out, (x, y), (int(x + 3 * dx), int(y + 3 * dy)), (0, 255, 0), 1, tipLength=0.3)
    return out


def dense_flow_video(path, name, max_seconds=30):
    """Writes outputs/<name>_flow.mp4 (original+arrows | colour flow). Returns stats dict."""
    os.makedirs(OUT, exist_ok=True)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    ok, f = cap.read()
    if not ok:
        raise RuntimeError("cannot read video")
    f = _resize(f)
    prev = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    h, w = f.shape[:2]
    raw = f"{OUT}/{name}_raw.mp4"
    writer = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w * 2, h))
    mags, evidence_saved, n = [], False, 1
    while n < fps * max_seconds:
        ok, f = cap.read()
        if not ok:
            break
        f = _resize(f)
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        # Farneback: dense flow for every pixel (args: pyr_scale, levels, winsize, iters, poly_n, poly_sigma, flags)
        flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        color, mag = flow_to_color(flow)
        left = draw_arrows(f, flow)
        writer.write(np.hstack([left, color]))
        mags.append(float(mag.mean()))
        if n % 60 == 0:
            print(f"processed {n} frames", flush=True)
        if not evidence_saved and n > fps * 5:          # evidence image at ~5 s
            cv2.imwrite(f"{OUT}/{name}_evidence.jpg", np.hstack([left, color]))
            evidence_saved = True
        prev, n = gray, n + 1
    writer.release()
    # browsers need H.264; re-encode if ffmpeg exists, otherwise keep mp4v (download still works)
    final = f"{OUT}/{name}_flow.mp4"
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe:
        subprocess.run([ffmpeg_exe, "-y", "-loglevel", "error", "-i", raw, "-vcodec", "libx264",
                        "-pix_fmt", "yuv420p", final])
        os.remove(raw)
    else:
        os.replace(raw, final)
    if not evidence_saved:
        cv2.imwrite(f"{OUT}/{name}_evidence.jpg", np.hstack([left, color]))
    # average speed per second of video (for the "what can we infer" discussion)
    per_sec = [round(float(np.mean(mags[i:i + int(fps)])), 3) for i in range(0, len(mags), int(fps))]
    return {"video": f"/outputs/{name}_flow.mp4", "evidence": f"/outputs/{name}_evidence.jpg",
            "frames": len(mags), "fps": fps, "mean_speed_px_per_frame_each_second": per_sec}


# ---------------------------------------------------------------- bilinear interpolation
def bilinear(img, x, y):
    """Value of img at non-integer (x, y). x, y can be numpy arrays of the same shape."""
    x = np.clip(x, 0, img.shape[1] - 1.001)
    y = np.clip(y, 0, img.shape[0] - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    a, b = x - x0, y - y0                       # fractional parts
    return ((1 - a) * (1 - b) * img[y0, x0] + a * (1 - b) * img[y0, x0 + 1]
            + (1 - a) * b * img[y0 + 1, x0] + a * b * img[y0 + 1, x0 + 1])


# ---------------------------------------------------------------- Lucas-Kanade by hand
def lk_point(I1, I2, pt, win=15, iters=30, d0=None):
    """
    Find displacement d=(u,v) of the window around pt from I1 to I2.
    Solve  (A^T A) delta = A^T b   repeatedly, where
        A = [Ix Iy] (one row per pixel in window), b = I1 - I2(warped by d).
    Returns d plus the first-iteration A^T A, A^T b for the report.
    """
    I1, I2 = I1.astype(np.float32), I2.astype(np.float32)
    gx = cv2.Sobel(I2, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(I2, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    h = win // 2
    xs, ys = np.meshgrid(np.arange(-h, h + 1), np.arange(-h, h + 1))
    xs, ys = xs + pt[0], ys + pt[1]
    T = bilinear(I1, xs, ys)                    # template from frame 1
    d, first = (np.zeros(2) if d0 is None else np.array(d0, float)), None
    for _ in range(iters):
        W = bilinear(I2, xs + d[0], ys + d[1])  # frame 2 sampled at shifted positions
        A = np.stack([bilinear(gx, xs + d[0], ys + d[1]).ravel(),
                      bilinear(gy, xs + d[0], ys + d[1]).ravel()], axis=1)
        b = (T - W).ravel()
        AtA, Atb = A.T @ A, A.T @ b
        if first is None:
            first = (AtA.copy(), Atb.copy())
        delta = np.linalg.solve(AtA + 1e-6 * np.eye(2), Atb)
        d += delta
        if np.linalg.norm(delta) < 1e-3:
            break
    return d, first


def lk_pyramid(I1, I2, pt, levels=4, win=15):
    """
    Coarse-to-fine Lucas-Kanade (handles LARGE motion).
    Shrink both images by 2 at each level, solve at the coarsest level first,
    then double the answer and use it as the starting guess on the next finer level.
    Returns the final d and the A^T A, A^T b of the FINEST level (for the report).
    """
    P1, P2 = [I1.astype(np.float32)], [I2.astype(np.float32)]
    for _ in range(levels - 1):
        P1.append(cv2.pyrDown(P1[-1]))
        P2.append(cv2.pyrDown(P2[-1]))
    d = np.zeros(2)
    for L in range(levels - 1, -1, -1):
        p = np.array(pt, float) / (2 ** L)
        d, first = lk_point(P1[L], P2[L], p, win=win, d0=d)
        if L > 0:
            d = d * 2
    return d, first


def validate_tracking(path, name, frame_idx=150, step=1, n_points=5):
    """Pick two frames, track corners by my LK, compare with OpenCV's LK (the reference)."""
    os.makedirs(OUT, exist_ok=True)
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok1, f1 = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx + step)
    ok2, f2 = cap.read()
    if not (ok1 and ok2):
        raise RuntimeError("frame index beyond end of video")
    f1, f2 = _resize(f1), _resize(f2)
    g1, g2 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(g1); mask[20:-20, 20:-20] = 255          # ignore the image border
    corners = cv2.goodFeaturesToTrack(g1, n_points, 0.05, 30, mask=mask, blockSize=7)
    ref, st, _ = cv2.calcOpticalFlowPyrLK(g1, g2, corners, None, winSize=(21, 21), maxLevel=3)
    vis1, vis2, rows = f1.copy(), f2.copy(), []
    for i, c in enumerate(corners.reshape(-1, 2)):
        d, (AtA, Atb) = lk_pyramid(g1, g2, c)
        mine, theirs = c + d, ref[i, 0]
        ev = np.linalg.eigvalsh(AtA)
        rows.append({"point": i + 1, "frame1_xy": [round(float(c[0]), 2), round(float(c[1]), 2)],
                     "my_flow_uv": [round(float(d[0]), 3), round(float(d[1]), 3)],
                     "predicted_frame2_xy": [round(float(mine[0]), 2), round(float(mine[1]), 2)],
                     "opencv_frame2_xy": [round(float(theirs[0]), 2), round(float(theirs[1]), 2)],
                     "error_px": round(float(np.linalg.norm(mine - theirs)), 3),
                     "AtA": np.round(AtA, 1).tolist(), "eigenvalues": np.round(ev, 1).tolist()})
        cv2.circle(vis1, tuple(int(v) for v in c), 5, (0, 255, 0), 1)
        cv2.circle(vis2, tuple(int(v) for v in mine), 5, (0, 0, 255), 1)      # red = mine
        cv2.circle(vis2, tuple(int(v) for v in theirs), 3, (255, 0, 0), -1)   # blue = OpenCV
        cv2.putText(vis1, str(i + 1), tuple(int(v) + 6 for v in c), 0, 0.5, (0, 255, 0), 1)
    cv2.imwrite(f"{OUT}/{name}_track.jpg", np.hstack([vis1, vis2]))
    return {"image": f"/outputs/{name}_track.jpg", "frames": [frame_idx, frame_idx + step], "points": rows}


if __name__ == "__main__":
    p = sys.argv[1]
    n = os.path.splitext(os.path.basename(p))[0]
    print(dense_flow_video(p, n))
    print(validate_tracking(p, n))
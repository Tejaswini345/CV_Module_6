# Computer Vision - Assignment Module 6
Part 1: optical flow + motion tracking on 2 videos. 
Part 2: structure from motion of a flat object from 4 views.
Language: Python 3.9+.

## Setup
```
pip install -r requirements.txt
```

## Folder guide
| Folder / file | What it is |
|---|---|
| `videos/` | We place our sample 2 videos here. They show up in the dropdown. |
| `sfm_data/images/` |  4 photos of the flat object (`view1.jpg` ... `view4.jpg`) |
| `sfm_data/cameras.json` | K, ruler measurements, pixel coordinates per view, camera notes |
| `flow.py` | Optical flow, bilinear interpolation, hand-written Lucas-Kanade |
| `sfm.py` | Homography (DLT) -> camera poses -> triangulation -> boundary |
| `app.py` + `templates/index.html` | Web demo with the video dropdown |
| `outputs/` | Generated videos/plots (screenshot these for the report) |

## Part 1 - run it
1. Place two clips (>= 30 s with motion) into `videos/`.
2. `python app.py`, open http://127.0.0.1:5000
3. Pick a video in the dropdown -> **Compute optical flow video** (first 30 s) -> **Validate tracking**
   (choose a frame number where something moves; compares Lucas-Kanade with OpenCV's for 5 corners).
4. Repeat for the second video. Screen-record this.

Without the web page: `python flow.py videos/clip1.mp4`

## Part 2 - run it
1. Print or draw a flat shape (e.g. A4 paper with 5-8 dots at corners/edges). Measure the dots with a ruler and put them in `world_points_cm` (origin at one dot, X right, Y up, Z=0).
2. Take 4 photos from different angles, same camera and zoom, save in `sfm_data/images/`. Write down roughly where the camera was (`camera_note`).
3. Camera matrix K: either use `python calibrate.py 9 6 2.5` (checkerboard photos in `sfm_data/calib/`), or approximate fx = fy = 1.2 x image width (px), cx, cy = image centre.
4. Click the dots in each photo in the same order: `python annotate.py sfm_data/images/view1.jpg` and paste the printed dict into `cameras.json`.
5. `python sfm.py` (or the **Run SfM** button).


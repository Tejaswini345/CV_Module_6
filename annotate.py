"""
annotate.py - click the marked points on each of your 4 photos and print the pixel coordinates.
Usage:   python annotate.py sfm_data/images/view1.jpg
Click the points IN THE SAME ORDER (A, B, C, ...) in every image. Press any key when finished.
Copy the printed list into sfm_data/cameras.json -> views[i].points_px
"""
import sys, cv2
img = cv2.imread(sys.argv[1]); pts = []
def click(e, x, y, *_):
    if e == cv2.EVENT_LBUTTONDOWN:
        pts.append([x, y]); cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(img, chr(64 + len(pts)), (x + 6, y), 0, 0.8, (0, 0, 255), 2); cv2.imshow("annotate", img)
cv2.imshow("annotate", img); cv2.setMouseCallback("annotate", click); cv2.waitKey(0)
print({chr(65 + i): p for i, p in enumerate(pts)})

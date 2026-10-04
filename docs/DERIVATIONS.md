# Derivations (rewrite in your own words in the PDF)

## 1. Brightness constancy and the flow equation
A point keeps its brightness when it moves by (u, v) between frames:
I(x+u, y+v, t+1) = I(x, y, t).
First-order Taylor expansion: I(x,y,t) + I_x u + I_y v + I_t = I(x,y,t)
=> **I_x u + I_y v + I_t = 0**. One equation, two unknowns (aperture problem).

## 2. Lucas-Kanade (tracking between two frames)
Assume the same (u, v) for all n pixels in a window around p (e.g. 15x15):
A d = b, with A = [I_x(q_i) I_y(q_i)] (n x 2), d = [u v]^T, b = -[I_t(q_i)] (n x 1).
Least squares: minimise ||A d - b||^2 -> **(A^T A) d = A^T b**, d = (A^T A)^-1 A^T b.
A^T A = [[sum Ix^2, sum IxIy],[sum IxIy, sum Iy^2]] (structure tensor).
Needs both eigenvalues large => corners track well, edges (one eigenvalue ~0) do not, flat regions fail.
Large motion: build image pyramids; small motion violated -> iterate: warp frame 2 by current d, recompute residual, add delta (this is what `lk_point` does).
Prediction in frame 2: p' = p + d.

## 3. Bilinear interpolation
Warping needs I at non-integer (x, y). Let x0 = floor(x), y0 = floor(y), a = x - x0, b = y - y0.
Interpolate along x on both rows:
 I(x, y0)   = (1-a) I(x0,y0)   + a I(x0+1,y0)
 I(x, y0+1) = (1-a) I(x0,y0+1) + a I(x0+1,y0+1)
Then along y: **I(x,y) = (1-b) I(x,y0) + b I(x,y0+1)**
= (1-a)(1-b) I00 + a(1-b) I10 + (1-a)b I01 + ab I11.  (weights sum to 1)

## 4. What flow tells us (give evidence from your frames)
Moving regions and their direction (hue) and speed (brightness / arrow length); camera motion (coherent field across the image) vs object motion (local region different from background); depth (near objects move more under camera translation); expansion from a point (focus of expansion) when moving forward.

## 5. Planar SfM
Pinhole: s [x y 1]^T = K [R | t] [X Y Z 1]^T. For the plane Z = 0:
s [x y 1]^T = K [r1 r2 t] [X Y 1]^T = H [X Y 1]^T, so H = K [r1 r2 t].

**Homography by DLT** (each point gives 2 rows; need >= 4 points):
[X Y 1 0 0 0 -xX -xY -x] h = 0
[0 0 0 X Y 1 -yX -yY -y] h = 0
Stack rows into M (2n x 9); h = last right-singular vector of M (SVD); reshape to 3x3.

**Pose from H:** K^-1 H = lambda [r1 r2 t], lambda = 1/||K^-1 h1||, r3 = r1 x r2.
Fix R with SVD (R = U V^T). Camera position in world: C = -R^T t.

**Triangulation** (view j has P_j = K[R_j|t_j]; observation (x_j, y_j)): rows
x_j P_j[3] - P_j[1] and y_j P_j[3] - P_j[2]; stack over 4 views; X = last singular vector, divide by 4th element.

**Boundary:** join points in order; area by shoelace A = 1/2 |sum(x_i y_{i+1} - x_{i+1} y_i)|; compare with ruler values; report reprojection error (px) per view and point error (cm).

## References to cite
- B. Lucas, T. Kanade, "An Iterative Image Registration Technique...", IJCAI 1981.
- B. Horn, B. Schunck, "Determining Optical Flow", Artificial Intelligence 1981.
- G. Farneback, "Two-Frame Motion Estimation Based on Polynomial Expansion", SCIA 2003.
- R. Hartley, A. Zisserman, Multiple View Geometry in Computer Vision, 2nd ed., 2004.
- Z. Zhang, "A Flexible New Technique for Camera Calibration", IEEE TPAMI 2000.
- OpenCV documentation (https://docs.opencv.org).

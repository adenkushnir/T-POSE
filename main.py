import argparse
import os
import time
import urllib.request
from functools import lru_cache

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "pose_landmarker_full.task")

# mediapipe's landmark numbers
NOSE, L_EAR, R_EAR = 0, 7, 8
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_PINKY, R_PINKY = 17, 18
L_INDEX, R_INDEX = 19, 20
L_HIP, R_HIP = 23, 24

BONES = [
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
    (L_WRIST, L_PINKY), (L_WRIST, L_INDEX), (R_WRIST, R_PINKY), (R_WRIST, R_INDEX),
    (L_SHOULDER, R_SHOULDER), (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP), (L_HIP, R_HIP),
]

HOLD_SECONDS = 2.0
GRID_W, GRID_H = 80, 45


# grabs the pose model the first time you run this
def get_model():
    if not os.path.exists(MODEL_PATH):
        print("downloading pose model...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


# pulls the first person out of mediapipe's result as pixel coords + how visible each joint is
def read_pose(result, w, h):
    if not result.pose_landmarks:
        return None, None
    lms = result.pose_landmarks[0]
    pts = np.array([[l.x * w, l.y * h] for l in lms], dtype=np.float32)
    vis = np.array([l.visibility for l in lms], dtype=np.float32)
    return pts, vis


# blends with last frame so it doesnt jitter. joints it cant see just stay where they were
def smooth(prev, pts, vis):
    if prev is None:
        return pts
    out = prev * 0.4 + pts * 0.6
    out[vis < 0.5] = prev[vis < 0.5]
    return out


# is one arm straight, level, and pointing away from the other shoulder
def arm_is_out(pts, shoulder, elbow, wrist, other_shoulder):
    upper = pts[shoulder] - pts[elbow]
    lower = pts[wrist] - pts[elbow]
    cos = np.dot(upper, lower) / (np.linalg.norm(upper) * np.linalg.norm(lower) + 1e-6)
    straight = cos < np.cos(np.radians(160))
    arm = pts[wrist] - pts[shoulder]
    level = abs(arm[1]) < abs(arm[0]) * np.tan(np.radians(20))
    outward = np.dot(arm, pts[shoulder] - pts[other_shoulder]) > 0
    return straight and level and outward


# checks both arms, and that we can actually see them
def is_tpose(pts, vis):
    joints = [L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST]
    if vis[joints].min() < 0.5:
        return False
    return arm_is_out(pts, L_SHOULDER, L_ELBOW, L_WRIST, R_SHOULDER) and arm_is_out(pts, R_SHOULDER, R_ELBOW, R_WRIST, L_SHOULDER)


# fixed points around the edge of the frame so the edges dont move
def border_points(w, h):
    xs = np.linspace(0, w - 1, 9)
    ys = np.linspace(0, h - 1, 6)[1:-1]
    top = [[x, 0] for x in xs]
    bottom = [[x, h - 1] for x in xs]
    left = [[0, y] for y in ys]
    right = [[w - 1, y] for y in ys]
    return np.array(top + bottom + left + right, dtype=np.float32)


# makes the matching point pairs: where each bit of you should end up (dst) and where it really is right now (src)
def build_handles(ref, live, w, h):
    dst, src = [], []

    # points along each bone so bent arms get pulled straight, not just the joints
    for a, b in BONES:
        for t in np.linspace(0, 1, 5):
            dst.append(ref[a] + (ref[b] - ref[a]) * t)
            src.append(live[a] + (live[b] - live[a]) * t)

    # head is allowed to move, so its target is just "wherever it is compared to your neck"
    # but put on the pinned neck. scale is for when you walk closer or further away
    ref_neck = (ref[L_SHOULDER] + ref[R_SHOULDER]) / 2
    live_neck = (live[L_SHOULDER] + live[R_SHOULDER]) / 2
    ref_width = np.linalg.norm(ref[L_SHOULDER] - ref[R_SHOULDER])
    live_width = np.linalg.norm(live[L_SHOULDER] - live[R_SHOULDER])
    scale = ref_width / (live_width + 1e-6)
    for i in [NOSE, L_EAR, R_EAR]:
        dst.append(ref_neck + (live[i] - live_neck) * scale)
        src.append(live[i])

    edge = border_points(w, h)
    return np.vstack([np.array(dst, dtype=np.float32), edge]), np.vstack([np.array(src, dtype=np.float32), edge])


# coarse grid points plus the lookup for scaling it back up. only needs working out once per frame size.
# cant use cv2.resize for the scaling, it lines up pixel centers not corners and shifts everything a few pixels
@lru_cache(maxsize=1)
def grid_setup(w, h):
    xs, ys = np.meshgrid(np.linspace(0, w - 1, GRID_W), np.linspace(0, h - 1, GRID_H))
    grid = np.stack([xs.ravel(), ys.ravel()], axis=1)
    up_x, up_y = np.meshgrid(np.linspace(0, GRID_W - 1, w), np.linspace(0, GRID_H - 1, h))
    return grid, up_x.astype(np.float32), up_y.astype(np.float32)


# the actual warp math (moving least squares, rigid version).
# for every output pixel on a coarse grid, works out which live pixel to show, then scales that up to full size
def warp_maps(dst, src, w, h):
    grid, up_x, up_y = grid_setup(w, h)
    dst, src = dst.astype(np.float64), src.astype(np.float64)

    dx = grid[:, 0, None] - dst[:, 0]
    dy = grid[:, 1, None] - dst[:, 1]
    weights = 1.0 / (dx * dx + dy * dy + 1.0)
    total = weights.sum(axis=1)

    dst_center = weights @ dst / total[:, None]
    src_center = weights @ src / total[:, None]

    # best rotation for each grid point. same as summing weight * (dst - center) * (src - center)
    # but multiplied out, way faster than making giant arrays
    a = weights @ (dst * src).sum(axis=1) - total * (dst_center * src_center).sum(axis=1)
    b = weights @ (dst[:, 0] * src[:, 1] - dst[:, 1] * src[:, 0]) - total * (dst_center[:, 0] * src_center[:, 1] - dst_center[:, 1] * src_center[:, 0])
    length = np.sqrt(a * a + b * b) + 1e-6
    cos, sin = a / length, b / length

    v = grid - dst_center
    mapped_x = cos * v[:, 0] - sin * v[:, 1] + src_center[:, 0]
    mapped_y = sin * v[:, 0] + cos * v[:, 1] + src_center[:, 1]

    map_x = cv2.remap(mapped_x.reshape(GRID_H, GRID_W).astype(np.float32), up_x, up_y, cv2.INTER_LINEAR)
    map_y = cv2.remap(mapped_y.reshape(GRID_H, GRID_W).astype(np.float32), up_x, up_y, cv2.INTER_LINEAR)
    return map_x, map_y


# opens the webcam and runs the whole thing until you quit
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("couldnt open camera", args.camera)
        return

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=get_model()),
        running_mode=vision.RunningMode.VIDEO,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    window = "t pose mirror"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    pts = None
    ref = None
    maps = None
    hold_start = None

    while True:
        ok, frame = cap.read()
        if not ok:
            print("lost the camera")
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = landmarker.detect_for_video(image, int(time.perf_counter() * 1000))
        new_pts, vis = read_pose(result, w, h)
        if new_pts is not None:
            pts = smooth(pts, new_pts, vis)

        if ref is None:
            # still waiting for you to hold the pose
            text = "get your whole arm span in frame and hold a T pose"
            if new_pts is not None and is_tpose(pts, vis):
                if hold_start is None:
                    hold_start = time.time()
                held = time.time() - hold_start
                text = f"holding... {held:.1f}s"
                if held >= HOLD_SECONDS:
                    ref = pts.copy()
            else:
                hold_start = None
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            output = frame
        else:
            # if nobody is detected we just keep using the last warp
            if new_pts is not None:
                dst, src = build_handles(ref, pts, w, h)
                maps = warp_maps(dst, src, w, h)
            if maps is None:
                output = frame
            else:
                output = cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        cv2.imshow(window, output)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord("r"):
            ref = None
            maps = None
            hold_start = None

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

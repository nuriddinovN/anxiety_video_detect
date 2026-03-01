"""
video_track.py
--------------
Analyzes a pre-recorded video with parallel face/eye/head-pose tracking.

Architecture:
  - Reader thread  : reads frames from disk into a shared queue
  - N Worker procs : each owns its own MediaPipe instance (true parallelism)
  - Collector      : merges results in frame order, writes CSV, updates display

Usage:
    python video_track.py --video path/to/video.mp4
    python video_track.py --video path/to/video.mp4 --no-display
    python video_track.py --video path/to/video.mp4 --workers 6 --skip 2
    python video_track.py --video path/to/video.mp4 --no-display --workers 8 --skip 3

Tips for a 50-min video:
    --skip 2   processes every 2nd frame  (half the work, still good tracking)
    --skip 3   processes every 3rd frame  (great for speed, 10+ fps effective)
    --no-display adds another ~20% speed boost
"""

import cv2 as cv
import numpy as np
import mediapipe as mp
import argparse
import csv
import os
import sys
import time
import queue
import threading
import multiprocessing as mproc
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from AngleBuffer import AngleBuffer

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_OUTPUT_DIR       = "/home/noor/A/research/personality/anxiety/Python-Gaze-Face-Tracker/tracking_outputs"
DEFAULT_WORKERS          = max(2, mproc.cpu_count() - 2)
DEFAULT_SKIP             = 1
READER_QUEUE_SIZE        = 64
RESULT_QUEUE_SIZE        = 128

MIN_DETECTION_CONFIDENCE = 0.8
MIN_TRACKING_CONFIDENCE  = 0.8
BLINK_THRESHOLD          = 0.51
EYE_AR_CONSEC_FRAMES     = 2
MOVING_AVERAGE_WINDOW    = 10

LEFT_EYE_IRIS          = [474, 475, 476, 477]
RIGHT_EYE_IRIS         = [469, 470, 471, 472]
LEFT_EYE_OUTER_CORNER  = [33]
RIGHT_EYE_OUTER_CORNER = [362]
RIGHT_EYE_POINTS       = [33, 160, 159, 158, 133, 153, 145, 144]
LEFT_EYE_POINTS        = [362, 385, 386, 387, 263, 373, 374, 380]
_INDICES_POSE          = [1, 33, 61, 199, 263, 291]

CSV_COLUMNS = [
    "timestamp", "frame_number", "video_time_sec", "total_blinks",
    "left_eye_cx", "left_eye_cy", "right_eye_cx", "right_eye_cy",
    "left_iris_dx", "left_iris_dy", "right_iris_dx", "right_iris_dy",
    "pitch", "yaw", "roll", "gaze_direction",
]

_STOP = "STOP"


# ── Math helpers ──────────────────────────────────────────────────────────────

def _euclidean_3D(points):
    P0,P3,P4,P5,P8,P11,P12,P13 = points
    num = np.linalg.norm(P3-P13)**3 + np.linalg.norm(P4-P12)**3 + np.linalg.norm(P5-P11)**3
    den = 3 * np.linalg.norm(P0-P8)**3
    return num / den

def _blinking_ratio(pts3d):
    return (_euclidean_3D(pts3d[RIGHT_EYE_POINTS]) + _euclidean_3D(pts3d[LEFT_EYE_POINTS]) + 1) / 2

def _vec_pos(p1, p2):
    x1,y1 = p1.ravel(); x2,y2 = p2.ravel()
    return x2-x1, y2-y1

def _gaze_dir(ax, ay, thr=10):
    if   ay < -thr: return "Left"
    elif ay >  thr: return "Right"
    elif ax < -thr: return "Down"
    elif ax >  thr: return "Up"
    else:           return "Forward"


# ── Worker process (one MediaPipe instance per process) ───────────────────────

def _worker(in_q, out_q):
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )
    while True:
        item = in_q.get()
        if item == _STOP:
            out_q.put(_STOP)
            break

        frame_idx, video_time, frame_bgr = item
        img_h, img_w = frame_bgr.shape[:2]
        rgb     = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        payload = {
            "frame_idx" : frame_idx,
            "video_time": video_time,
            "frame_bgr" : frame_bgr,
            "landmarks" : None,
        }

        if results.multi_face_landmarks:
            lm    = results.multi_face_landmarks[0].landmark
            pts2d = np.array([np.multiply([p.x,p.y],[img_w,img_h]).astype(int) for p in lm])
            pts3d = np.array([[n.x,n.y,n.z] for n in lm])
            payload["landmarks"] = (pts2d, pts3d, img_w, img_h)

        out_q.put(payload)


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(rows, proc_sec, video_path, csv_path):
    if not rows:
        print("[video_track] No data collected."); return

    total_blinks = int(rows[-1]["total_blinks"]) if rows[-1]["total_blinks"] != "" else 0
    video_dur    = float(rows[-1]["video_time_sec"])
    blink_rate   = round(total_blinks/(video_dur/60), 1) if video_dur > 0 else "N/A"
    speed_x      = round(video_dur/proc_sec, 1) if proc_sec > 0 else "N/A"

    def fv(k): return [float(r[k]) for r in rows if r.get(k,"") != ""]
    def avg(l): return round(sum(l)/len(l),2) if l else "N/A"
    def rng(l): return (round(min(l),2), round(max(l),2)) if l else ("N/A","N/A")

    pitches = fv("pitch"); yaws = fv("yaw"); rolls = fv("roll")
    gc = {}
    for r in rows:
        g = r.get("gaze_direction","")
        if g: gc[g] = gc.get(g,0)+1
    dominant = max(gc, key=gc.get) if gc else "N/A"

    print("\n" + "═"*58)
    print("  SESSION SUMMARY")
    print("═"*58)
    print(f"  Video file           : {os.path.basename(video_path)}")
    print(f"  Video duration       : {video_dur:.1f} s  ({video_dur/60:.1f} min)")
    print(f"  Processing time      : {proc_sec:.1f} s  ({speed_x}x realtime)")
    print(f"  Frames analysed      : {len(rows)}")
    print(f"  Total blinks         : {total_blinks}")
    print(f"  Blink rate           : {blink_rate} blinks/min")
    print(f"  Avg Pitch            : {avg(pitches)}°")
    print(f"  Avg Yaw              : {avg(yaws)}°")
    print(f"  Avg Roll             : {avg(rolls)}°")
    print(f"  Pitch range          : {rng(pitches)}")
    print(f"  Yaw range            : {rng(yaws)}")
    print(f"  Roll range           : {rng(rolls)}")
    print(f"  Dominant gaze        : {dominant}")
    for direction, count in sorted(gc.items(), key=lambda x:-x[1]):
        pct = round(100*count/len(rows),1)
        print(f"    {direction:<10}: {count} frames ({pct}%)")
    print(f"\n  CSV saved → {csv_path}")
    print("═"*58 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def process_video(video_path, output_dir, show_display, n_workers, frame_skip):
    os.makedirs(output_dir, exist_ok=True)
    ts_str   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    stem     = os.path.splitext(os.path.basename(video_path))[0]
    csv_path = os.path.join(output_dir, f"{stem}_{ts_str}.csv")

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {video_path}"); sys.exit(1)
    fps          = cap.get(cv.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"[video_track] Video   : {video_path}")
    print(f"[video_track] FPS     : {fps:.1f}  |  Total frames: {total_frames}")
    print(f"[video_track] Workers : {n_workers}  |  Frame skip: {frame_skip}")
    print(f"[video_track] Output  : {csv_path}\n")

    # ── Shared queues ─────────────────────────────────────────────────────────
    in_q  = mproc.Queue(maxsize=READER_QUEUE_SIZE)
    out_q = mproc.Queue(maxsize=RESULT_QUEUE_SIZE)

    # ── Worker processes ──────────────────────────────────────────────────────
    workers = [mproc.Process(target=_worker, args=(in_q, out_q), daemon=True)
               for _ in range(n_workers)]
    for w in workers: w.start()

    # ── Reader thread ─────────────────────────────────────────────────────────
    def reader():
        cap2 = cv.VideoCapture(video_path)
        idx  = 0
        while True:
            ret, frame = cap2.read()
            if not ret: break
            idx += 1
            if idx % frame_skip != 0: continue
            in_q.put((idx, idx/fps, frame))
        cap2.release()
        for _ in range(n_workers):
            in_q.put(_STOP)

    threading.Thread(target=reader, daemon=True).start()

    # ── Display window ────────────────────────────────────────────────────────
    WIN = "video_track — press Q to stop"
    if show_display:
        cv.namedWindow(WIN, cv.WINDOW_NORMAL)
        cv.resizeWindow(WIN, 960, 540)

    # ── Collector ─────────────────────────────────────────────────────────────
    angle_buffer        = AngleBuffer(size=MOVING_AVERAGE_WINDOW)
    total_blinks        = 0
    blink_frame_counter = 0
    initial_pitch = initial_yaw = initial_roll = None
    rows                = []
    pending             = {}          # frame_idx -> result (out-of-order buffer)
    next_expected       = None        # set on first result
    stops_received      = 0
    proc_start          = time.time()
    last_progress_t     = proc_start
    user_quit           = False

    def emit(result):
        nonlocal total_blinks, blink_frame_counter
        nonlocal initial_pitch, initial_yaw, initial_roll

        fidx       = result["frame_idx"]
        video_time = result["video_time"]
        frame_bgr  = result["frame_bgr"]
        lm_data    = result["landmarks"]

        row = {col:"" for col in CSV_COLUMNS}
        row["timestamp"]      = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        row["frame_number"]   = fidx
        row["video_time_sec"] = round(video_time, 3)
        row["total_blinks"]   = total_blinks

        overlay = {}

        if lm_data is not None:
            pts2d, pts3d, img_w, img_h = lm_data

            ear = _blinking_ratio(pts3d)
            if ear <= BLINK_THRESHOLD:
                blink_frame_counter += 1
            else:
                if blink_frame_counter > EYE_AR_CONSEC_FRAMES:
                    total_blinks += 1
                blink_frame_counter = 0
            row["total_blinks"] = total_blinks

            (l_cx,l_cy),_ = cv.minEnclosingCircle(pts2d[LEFT_EYE_IRIS])
            (r_cx,r_cy),_ = cv.minEnclosingCircle(pts2d[RIGHT_EYE_IRIS])
            cl = np.array([l_cx,l_cy], dtype=np.int32)
            cr = np.array([r_cx,r_cy], dtype=np.int32)
            l_dx,l_dy = _vec_pos(pts2d[LEFT_EYE_OUTER_CORNER],  cl)
            r_dx,r_dy = _vec_pos(pts2d[RIGHT_EYE_OUTER_CORNER], cr)
            row.update({
                "left_eye_cx" :round(float(l_cx),1), "left_eye_cy" :round(float(l_cy),1),
                "right_eye_cx":round(float(r_cx),1), "right_eye_cy":round(float(r_cy),1),
                "left_iris_dx":int(l_dx), "left_iris_dy":int(l_dy),
                "right_iris_dx":int(r_dx),"right_iris_dy":int(r_dy),
            })
            overlay["eyes"] = (cl, cr)

            hp3d = np.multiply(pts3d[_INDICES_POSE], [img_w,img_h,1])
            hp2d = np.delete(hp3d,2,axis=1).astype(np.float64)
            hp3d = hp3d.astype(np.float64)
            fl   = float(img_w)
            cam  = np.array([[fl,0,img_h/2],[0,fl,img_w/2],[0,0,1]])
            dist = np.zeros((4,1),dtype=np.float64)
            ok,rv,_ = cv.solvePnP(hp3d,hp2d,cam,dist)
            if ok:
                rm,_    = cv.Rodrigues(rv)
                angs,*_ = cv.RQDecomp3x3(rm)
                ax = angs[0]*360; ay = angs[1]*360
                angle_buffer.add([ax, ay, angs[2]*360])
                pitch,yaw,roll = angle_buffer.get_average()
                if initial_pitch is None:
                    initial_pitch,initial_yaw,initial_roll = pitch,yaw,roll
                pitch -= initial_pitch; yaw -= initial_yaw; roll -= initial_roll
                row.update({
                    "pitch":round(pitch,4),"yaw":round(yaw,4),"roll":round(roll,4),
                    "gaze_direction":_gaze_dir(ax,ay),
                })
                overlay["pose"] = (pitch,yaw,roll)

        if show_display and frame_bgr is not None:
            if "eyes" in overlay:
                cv.circle(frame_bgr, tuple(overlay["eyes"][0]), 5, (255,0,255), 2)
                cv.circle(frame_bgr, tuple(overlay["eyes"][1]), 5, (255,0,255), 2)
            cv.putText(frame_bgr, f"Blinks: {total_blinks}",
                       (20,40), cv.FONT_HERSHEY_DUPLEX, 0.8, (0,255,0), 2)
            if "pose" in overlay:
                p,y,r = overlay["pose"]
                cv.putText(frame_bgr, f"P:{int(p)} Y:{int(y)} R:{int(r)}",
                           (20,75), cv.FONT_HERSHEY_DUPLEX, 0.7, (0,255,0), 2)
            h = frame_bgr.shape[0]
            cv.putText(frame_bgr, f"t={video_time:.1f}s  f={fidx}",
                       (20,h-20), cv.FONT_HERSHEY_DUPLEX, 0.6, (180,180,180), 1)
            cv.imshow(WIN, frame_bgr)

        return row

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        try:
            while stops_received < n_workers and not user_quit:
                try:
                    result = out_q.get(timeout=5)
                except Exception:
                    continue

                if result == _STOP:
                    stops_received += 1
                    continue

                fidx = result["frame_idx"]
                if next_expected is None:
                    next_expected = fidx

                pending[fidx] = result

                # Flush ordered frames
                while next_expected in pending:
                    r   = pending.pop(next_expected)
                    row = emit(r)
                    writer.writerow(row)
                    csvfile.flush()
                    rows.append(row)
                    next_expected += frame_skip

                # Key check
                if show_display:
                    if cv.waitKey(1) & 0xFF == ord("q"):
                        print("\n[video_track] Stopped by user.")
                        user_quit = True
                        break

                # Progress every 10 s wall-time
                now = time.time()
                if now - last_progress_t >= 10:
                    last_progress_t = now
                    pct = round(100*fidx/total_frames,1) if total_frames else "?"
                    elapsed = now - proc_start
                    eta_str = ""
                    if isinstance(pct, float) and pct > 0:
                        eta_sec = elapsed/(pct/100) - elapsed
                        eta_str = f"  ETA ~{eta_sec/60:.1f} min"
                    blinks_now = rows[-1]["total_blinks"] if rows else 0
                    print(f"  Progress: {fidx}/{total_frames} ({pct}%)  "
                          f"blinks: {blinks_now}{eta_str}")

        except KeyboardInterrupt:
            print("\n[video_track] Interrupted.")

    for w in workers: w.terminate()
    cv.destroyAllWindows()
    proc_end = time.time()
    print_summary(rows, proc_end-proc_start, video_path, csv_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Parallel gaze/face tracker for pre-recorded video.")
    p.add_argument("--video",      required=True,              help="Input video path")
    p.add_argument("--out-dir",    default=DEFAULT_OUTPUT_DIR, help="CSV output directory")
    p.add_argument("--no-display", action="store_true",        help="Headless — fastest mode")
    p.add_argument("--workers",    type=int, default=DEFAULT_WORKERS,
                   help=f"Worker processes (default: {DEFAULT_WORKERS})")
    p.add_argument("--skip",       type=int, default=DEFAULT_SKIP,
                   help="Process every Nth frame. 2=half frames, 3=third. "
                        "Recommended 2-3 for 50min+ videos.")
    return p.parse_args()


if __name__ == "__main__":
    mproc.set_start_method("spawn", force=True)  # required: MediaPipe + fork = crash
    args = parse_args()
    process_video(args.video, args.out_dir, not args.no_display, args.workers, args.skip)

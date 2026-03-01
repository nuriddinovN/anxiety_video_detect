"""
save_track.py
-------------
Launches main.py as a subprocess, parses its real-time stdout output,
saves structured tracking data to a timestamped CSV file, and prints
a session summary on exit.

Usage:
    python save_track.py
    python save_track.py -c 1   # use camera source 1
"""

import subprocess
import sys
import csv
import os
import re
import time
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR  = "/home/noor/A/research/personality/anxiety/Python-Gaze-Face-Tracker/tracking_outputs"
MAIN_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")

CSV_COLUMNS = [
    "timestamp",
    "elapsed_sec",
    "total_blinks",
    "left_eye_cx",
    "left_eye_cy",
    "right_eye_cx",
    "right_eye_cy",
    "left_iris_dx",
    "left_iris_dy",
    "right_iris_dx",
    "right_iris_dy",
    "pitch",
    "yaw",
    "roll",
]

# ── Regex patterns matching main.py print statements ─────────────────────────
RE_BLINKS    = re.compile(r"Total Blinks:\s*(\d+)")
RE_LEFT_EYE  = re.compile(r"Left Eye Center X:\s*([\d.]+)\s*Y:\s*([\d.]+)")
RE_RIGHT_EYE = re.compile(r"Right Eye Center X:\s*([\d.]+)\s*Y:\s*([\d.]+)")
RE_LEFT_IRIS = re.compile(r"Left Iris Relative Pos Dx:\s*(-?\d+)\s*Dy:\s*(-?\d+)")
RE_RIGHT_IRIS= re.compile(r"Right Iris Relative Pos Dx:\s*(-?\d+)\s*Dy:\s*(-?\d+)")
RE_POSE      = re.compile(r"Head Pose Angles:\s*Pitch=([-\d.]+),\s*Yaw=([-\d.]+),\s*Roll=([-\d.]+)")


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def unique_csv_path():
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(OUTPUT_DIR, f"tracking_{ts}.csv")


def print_summary(rows, session_start, session_end):
    duration    = session_end - session_start
    total_blinks= int(rows[-1]["total_blinks"]) if rows and rows[-1]["total_blinks"] else 0

    pitches = [float(r["pitch"]) for r in rows if r["pitch"] != ""]
    yaws    = [float(r["yaw"])   for r in rows if r["yaw"]   != ""]
    rolls   = [float(r["roll"])  for r in rows if r["roll"]  != ""]

    def avg(lst):   return round(sum(lst) / len(lst), 2) if lst else "N/A"
    def rng(lst):   return (round(min(lst), 2), round(max(lst), 2)) if lst else ("N/A", "N/A")

    blink_rate = round(total_blinks / (duration / 60), 1) if duration > 0 else "N/A"

    print("\n" + "═" * 54)
    print("  SESSION SUMMARY")
    print("═" * 54)
    print(f"  Duration             : {duration:.1f} s  ({duration/60:.1f} min)")
    print(f"  Frames logged        : {len(rows)}")
    print(f"  Total blinks         : {total_blinks}")
    print(f"  Blink rate           : {blink_rate} blinks/min")
    print(f"  Avg Pitch            : {avg(pitches)}°")
    print(f"  Avg Yaw              : {avg(yaws)}°")
    print(f"  Avg Roll             : {avg(rolls)}°")
    print(f"  Pitch range          : {rng(pitches)}")
    print(f"  Yaw range            : {rng(yaws)}")
    print(f"  Roll range           : {rng(rolls)}")
    print("═" * 54 + "\n")


def main():
    ensure_output_dir()
    csv_path = unique_csv_path()

    # Pass through any extra CLI args (e.g. -c 1) to main.py
    extra_args = sys.argv[1:]
    cmd = [sys.executable, MAIN_SCRIPT] + extra_args

    print(f"[save_track] Starting main.py ...")
    print(f"[save_track] Output CSV → {csv_path}\n")

    session_start = time.time()
    rows  = []
    frame = {}   # accumulates parsed fields for one frame

    try:
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            csvfile.flush()

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=sys.stderr,   # mediapipe/OpenCV warnings go straight to terminal
                text=True,
                bufsize=1,
            )

            for raw_line in proc.stdout:
                line = raw_line.rstrip()
                print(line)           # echo so user still sees live output

                # ── parse each recognised pattern ──────────────────────────
                m = RE_BLINKS.search(line)
                if m:
                    frame["total_blinks"] = m.group(1)

                m = RE_LEFT_EYE.search(line)
                if m:
                    frame["left_eye_cx"], frame["left_eye_cy"] = m.group(1), m.group(2)

                m = RE_RIGHT_EYE.search(line)
                if m:
                    frame["right_eye_cx"], frame["right_eye_cy"] = m.group(1), m.group(2)

                m = RE_LEFT_IRIS.search(line)
                if m:
                    frame["left_iris_dx"], frame["left_iris_dy"] = m.group(1), m.group(2)

                m = RE_RIGHT_IRIS.search(line)
                if m:
                    frame["right_iris_dx"], frame["right_iris_dy"] = m.group(1), m.group(2)

                m = RE_POSE.search(line)
                if m:
                    frame["pitch"], frame["yaw"], frame["roll"] = m.group(1), m.group(2), m.group(3)

                # main.py prints a blank line after each frame block → flush row
                if line.strip() == "" and frame:
                    now = time.time()
                    row = {
                        "timestamp"    : datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                        "elapsed_sec"  : round(now - session_start, 3),
                        "total_blinks" : frame.get("total_blinks", ""),
                        "left_eye_cx"  : frame.get("left_eye_cx",  ""),
                        "left_eye_cy"  : frame.get("left_eye_cy",  ""),
                        "right_eye_cx" : frame.get("right_eye_cx", ""),
                        "right_eye_cy" : frame.get("right_eye_cy", ""),
                        "left_iris_dx" : frame.get("left_iris_dx", ""),
                        "left_iris_dy" : frame.get("left_iris_dy", ""),
                        "right_iris_dx": frame.get("right_iris_dx",""),
                        "right_iris_dy": frame.get("right_iris_dy",""),
                        "pitch"        : frame.get("pitch", ""),
                        "yaw"          : frame.get("yaw",   ""),
                        "roll"         : frame.get("roll",  ""),
                    }
                    writer.writerow(row)
                    csvfile.flush()
                    rows.append(row)
                    frame = {}  # reset for next frame

            proc.wait()

    except KeyboardInterrupt:
        print("\n[save_track] KeyboardInterrupt — flushing remaining data ...")
        # write any partial frame that hasn't been flushed yet
        if frame:
            now = time.time()
            row = {col: frame.get(col, "") for col in CSV_COLUMNS}
            row["timestamp"]   = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            row["elapsed_sec"] = round(now - session_start, 3)
            with open(csv_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=CSV_COLUMNS).writerow(row)
            rows.append(row)

    finally:
        session_end = time.time()
        print(f"\n[save_track] CSV saved → {csv_path}")
        print_summary(rows, session_start, session_end)


if __name__ == "__main__":
    main()
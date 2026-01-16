#!/usr/bin/env python3
# Dependencies:
#   python3 -m pip install opencv-python numpy
#
# Example:
#   python3 plot_video_qa.py \
#     --video video.mp4 \
#     --csv video_qa_per_frame.csv \
#     --out_dir plotted_frames \
#     --save_video \
#     --output_video_name plotted.mp4

import argparse
import csv
import os
import re
import sys

import cv2
import numpy as np


# ---------------------------
# Mapping: text -> {-1,0,1}
# ---------------------------
YES_TOKENS = ["yes", "true", "on", "1", "("]
NO_TOKENS  = ["no", "false", "off", "0"]

def answer_to_value(text: str) -> int:
    """
    +1 for yes-like tokens, -1 for no-like tokens, 0 otherwise.
    Uses substring matching, case-insensitive.
    If both appear, 'no' wins if it appears explicitly; otherwise yes wins.
    """
    if text is None:
        return 0
    s = str(text).strip().lower()
    if not s:
        return 0

    has_yes = any(tok in s for tok in YES_TOKENS)
    has_no  = any(tok in s for tok in NO_TOKENS)

    if has_no and not has_yes:
        return -1
    if has_yes and not has_no:
        return 1
    if has_no and has_yes:
        # tie-break: if explicit "no"/"false"/"off"/"0" appears, treat as -1
        return -1
    return 0


# ---------------------------
# Plot helpers (your style)
# ---------------------------
def calculateRelativeValue(y, h, value, minimum, maximum):
    if maximum == minimum:
        return int(y + (h / 2))
    vRange = (maximum - minimum)
    return int(y + (h / 2) - (value / vRange) * (h / 2))


def drawSinglePlotValueList(valueListRAW, color, itemName, image, x, y, w, h, minimumValue=None, maximumValue=None):
    # Make sure to only display last items of list that fit
    margin = 10
    if len(valueListRAW) > w + margin:
        itemsToRemove = len(valueListRAW) - w
        valueList = valueListRAW[itemsToRemove:]
    else:
        valueList = valueListRAW

    # Auto scale if no minimum/maximum
    if minimumValue is None:
        minimumValue = min(valueList) if len(valueList) else 0
    if maximumValue is None:
        maximumValue = max(valueList) if len(valueList) else 0

    if minimumValue == maximumValue:
        color = (40, 40, 40)  # dead plot

    if len(valueList):
        listMaxValue = np.max(valueList)
        if listMaxValue > maximumValue:
            maximumValue = listMaxValue * 2

    # Axes
    cv2.line(image, pt1=(x, y + h), pt2=(x + w, y + h), color=color, thickness=1)  # X-axis
    cv2.line(image, pt1=(x, y),     pt2=(x, y + h),     color=color, thickness=1)  # Y-axis

    # Labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.35
    thickness = 1

    # Title
    msg = f"{itemName}"
    image = cv2.putText(image, msg, (x - 1, y - 1), font, fontScale, (0, 0, 0), thickness, cv2.LINE_AA)
    image = cv2.putText(image, msg, (x, y),         font, fontScale, color,      thickness, cv2.LINE_AA)

    # Min/Max
    msg = f"Max {maximumValue:.2f}"
    image = cv2.putText(image, msg, (x - 1, y + 11 - 1), font, fontScale, (0, 0, 0), thickness, cv2.LINE_AA)
    image = cv2.putText(image, msg, (x,     y + 11),     font, fontScale, color,      thickness, cv2.LINE_AA)

    msg = f"Min {minimumValue:.2f}"
    image = cv2.putText(image, msg, (x - 1, y + h + 11 - 1), font, fontScale, (0, 0, 0), thickness, cv2.LINE_AA)
    image = cv2.putText(image, msg, (x,     y + h + 11),     font, fontScale, color,      thickness, cv2.LINE_AA)

    # Polyline
    if len(valueList) > 2:
        for frameID in range(1, len(valueList)):
            prevV = calculateRelativeValue(y, h, valueList[frameID - 1], minimumValue, maximumValue)
            nextV = calculateRelativeValue(y, h, valueList[frameID],     minimumValue, maximumValue)
            p0 = (int(x + frameID - 1), prevV)
            p1 = (int(x + frameID),     nextV)
            cv2.line(image, pt1=p0, pt2=p1, color=color, thickness=1)

    # Last value label
    if len(valueList):
        last = valueList[-1]
        org = (int(x + len(valueList)), calculateRelativeValue(y, h, last, minimumValue, maximumValue))
        msg = f"{last:.2f}"
        image = cv2.putText(image, msg, org, font, fontScale, (0, 0, 0), thickness, cv2.LINE_AA)
        image = cv2.putText(image, msg, (org[0] + 1, org[1] + 1), font, fontScale, color, thickness, cv2.LINE_AA)

    return image


def safe_header_name(s: str) -> str:
    # Keep plot labels readable
    s = re.sub(r"\s+", " ", str(s).strip())
    if len(s) > 40:
        s = s[:40].rstrip() + "…"
    return s


def load_csv_rows(csv_path: str):
    """
    Returns:
      frame_to_answers: dict[int frame_index] -> dict[colname]->cell
      question_cols: list[str] (all columns except metadata)
    """
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        # expected metadata:
        # video_path, frame_index, timestamp_sec, then question columns
        meta = {"video_path", "frame_index", "timestamp_sec"}
        question_cols = [c for c in cols if c not in meta]

        frame_to_answers = {}
        for row in reader:
            if "frame_index" not in row:
                continue
            try:
                fi = int(float(row["frame_index"]))
            except Exception:
                continue
            frame_to_answers[fi] = row

    return frame_to_answers, question_cols


def ffmpeg_encode_frames(output_dir, frame_prefix, image_ext, fps, w, h, output_video_name):
    input_pattern = os.path.join(output_dir, f"{frame_prefix}_%05d.{image_ext}")
    output_video_path = os.path.join(output_dir, output_video_name)

    ffmpeg_cmd = (
        f"ffmpeg -framerate {int(round(fps))} "
        f"-start_number 1 "
        f"-i \"{input_pattern}\" "
        f"-s {w}x{h} "
        f"-y -r {int(round(fps))} "
        f"-pix_fmt yuv420p "
        f"-threads 8 "
        f"\"{output_video_path}\""
    )
    print("\nRunning ffmpeg:")
    print(ffmpeg_cmd)
    os.system(ffmpeg_cmd)


def main():
    ap = argparse.ArgumentParser(description="Overlay per-question plots next to frames based on per-frame CSV results.")
    ap.add_argument("--video", required=True, help="Input video path (same source used for CSV).")
    ap.add_argument("--csv", required=True, help="CSV produced by per-frame QA (rows=frames, cols=questions).")
    ap.add_argument("--out_dir", default="plot_output", help="Directory to write annotated frames and optional video.")

    ap.add_argument("--plot_width", type=int, default=420, help="Width of the plot panel on the right.")
    ap.add_argument("--plot_height", type=int, default=70, help="Height per plot (stacked vertically).")
    ap.add_argument("--plot_gap", type=int, default=16, help="Vertical gap between plots.")
    ap.add_argument("--max_plots", type=int, default=0, help="Limit number of plotted questions (0 = all).")

    ap.add_argument("--image_ext", default="jpg", choices=["jpg", "png"], help="Frame image format for output.")
    ap.add_argument("--frame_prefix", default="frame", help="Output frame filename prefix.")
    ap.add_argument("--save_video", action="store_true", help="Encode frames into an output video using ffmpeg.")
    ap.add_argument("--output_video_name", default="plotted.mp4", help="Name of output video in out_dir.")
    ap.add_argument("--force_fps", type=float, default=0.0, help="Override FPS for ffmpeg (0 = use input video FPS).")

    args = ap.parse_args()

    if not os.path.isfile(args.video):
        print(f"Error: video not found: {args.video}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.csv):
        print(f"Error: CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    frame_to_answers, question_cols = load_csv_rows(args.csv)
    if not question_cols:
        print("Error: no question columns found in CSV.", file=sys.stderr)
        sys.exit(1)

    if args.max_plots and args.max_plots > 0:
        question_cols = question_cols[:args.max_plots]

    # Initialize per-question time series
    series = {q: [] for q in question_cols}

    # Video reader
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: could not open video: {args.video}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if args.force_fps and args.force_fps > 0:
        fps = args.force_fps
    if fps <= 0:
        fps = 25.0  # fallback for ffmpeg

    base_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    base_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if base_w <= 0 or base_h <= 0:
        print("Error: could not read video dimensions.", file=sys.stderr)
        sys.exit(1)

    out_w = base_w + args.plot_width
    # Make sure plot panel is tall enough; if not, we’ll still draw clipped
    out_h = base_h

    # Colors for plots (BGR)
    palette = [
        (0, 255, 0),
        (0, 200, 255),
        (255, 0, 0),
        (255, 0, 255),
        (255, 255, 0),
        (200, 200, 200),
        (0, 128, 255),
        (128, 0, 255),
        (255, 128, 0),
        (0, 255, 128),
    ]

    frame_idx = 0
    out_index = 1  # for ffmpeg -start_number 1

    print(f"Questions plotted: {len(question_cols)}")
    print(f"Writing frames to: {args.out_dir}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Only process frames that exist in CSV
        row = frame_to_answers.get(frame_idx, None)
        if row is None:
            frame_idx += 1
            continue

        # Update series values for this frame
        for q in question_cols:
            v = answer_to_value(row.get(q, ""))
            series[q].append(v)

        # Create output canvas (frame + right panel)
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        canvas[:, :base_w, :] = frame

        # Right panel background (slightly dark)
        canvas[:, base_w:, :] = (18, 18, 18)

        # Draw plots stacked
        x0 = base_w + 10
        y0 = 20
        plot_w = args.plot_width - 20
        plot_h = args.plot_height

        for i, q in enumerate(question_cols):
            y = y0 + i * (plot_h + args.plot_gap)
            if y + plot_h + 30 >= out_h:
                # Stop if we run out of vertical space
                break

            color = palette[i % len(palette)]
            label = safe_header_name(q)

            # Fixed scale for your {-1,0,1}
            drawSinglePlotValueList(
                valueListRAW=series[q],
                color=color,
                itemName=label,
                image=canvas,
                x=x0,
                y=y,
                w=plot_w,
                h=plot_h,
                minimumValue=-1,
                maximumValue=1
            )

        # Write frame image
        out_path = os.path.join(args.out_dir, f"{args.frame_prefix}_{out_index:05d}.{args.image_ext}")
        cv2.imwrite(out_path, canvas)
        out_index += 1

        if out_index % 50 == 0:
            print(f"Wrote {out_index-1} frames... (last source frame {frame_idx})")

        frame_idx += 1

    cap.release()

    print(f"Done. Total written frames: {out_index-1}")

    if args.save_video:
        ffmpeg_encode_frames(
            output_dir=args.out_dir,
            frame_prefix=args.frame_prefix,
            image_ext=args.image_ext,
            fps=fps,
            w=out_w,
            h=out_h,
            output_video_name=args.output_video_name
        )
        print(f"Saved video to: {os.path.join(args.out_dir, args.output_video_name)}")


if __name__ == "__main__":
    main()


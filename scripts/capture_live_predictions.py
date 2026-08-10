"""
capture_live_predictions.py
───────────────────────────
Capture live Teensy predictions while guiding the user through a sequence
of gestures. Outputs a CSV with timestamps, Teensy prediction, expected
action, and take number.

Output format:
    Timestamp, Prediction, Action, Take

This CSV is compatible with the online_prediction_assessment notebook
(after light adaptation).

Usage:
    python scripts/capture_live_predictions.py --port /dev/cu.usbmodem157347101 --label sherylann_6cl

CLI flags:
    --port        Serial port of the Teensy (mandatory)
    --label       Identifier for this capture session (used in output filename)
    --n-classes   Number of classes the model predicts (4 or 6, default 6)
    --n-takes     Number of repetitions per gesture (default 3)
    --duration    Seconds per gesture per take (default 5)
    --pause       Seconds of pause between gestures (default 2)
    --output-dir  Where to save the CSV (default: benchmarks/)
"""

import sys
import argparse
import time
import csv
from pathlib import Path

import serial


GESTURES_4CL = ['rest', 'flexion', 'extension', 'close']
GESTURES_6CL = ['rest', 'flexion', 'extension', 'close', 'supination', 'pronation']

ACTION_TO_INT = {
    'rest': 0, 'flexion': 1, 'extension': 2,
    'close': 3, 'supination': 4, 'pronation': 5
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port", required=True)
    p.add_argument("--label", required=True,
                   help="Identifier for this capture (e.g. 'sherylann_6cl')")
    p.add_argument("--n-classes", type=int, default=6, choices=[4, 6])
    p.add_argument("--n-takes", type=int, default=3)
    p.add_argument("--duration", type=float, default=5.0,
                   help="Seconds per gesture per take")
    p.add_argument("--pause", type=float, default=2.0,
                   help="Pause between gestures in seconds")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--output-dir", default="benchmarks")
    return p.parse_args()


def parse_teensy_line(line):
    """Parse a Teensy prediction line of the form:
       '0 rest  raw=0  logits: 0.95 0.02 ...'
    Returns int(pred) or None if line is not parseable.
    """
    if not line or line.startswith('#'):
        return None
    parts = line.strip().split()
    if not parts:
        return None
    try:
        return int(parts[0])
    except ValueError:
        return None


def countdown(msg, secs=3):
    """Display a quick countdown."""
    for i in range(secs, 0, -1):
        print(f"  {msg} in {i}...", end='\r', flush=True)
        time.sleep(1)
    print(" " * 60, end='\r')


def show_gesture(gesture, take_num, total_takes):
    """Big, visible display of the gesture to perform."""
    border = "═" * 60
    print()
    print(border)
    print(f"  TAKE {take_num}/{total_takes}")
    print(f"  GESTURE: >>>  {gesture.upper()}  <<<")
    print(border)


def flush_serial(ser):
    """Drain anything pending on the port."""
    try:
        ser.reset_input_buffer()
    except Exception:
        pass


def capture_phase(ser, gesture, duration, take_num, action_int, start_time):
    """Read predictions from Teensy for `duration` seconds.
    Returns a list of dicts: {Timestamp, Prediction, Action, Take}.
    """
    records = []
    t_start = time.time()
    t_end = t_start + duration

    while time.time() < t_end:
        line = ser.readline().decode(errors='replace').strip()
        pred = parse_teensy_line(line)
        if pred is None:
            continue
        elapsed = time.time() - start_time
        records.append({
            'Timestamp': round(elapsed, 4),
            'Prediction': pred,
            'Action': action_int,
            'Take': take_num,
        })
    return records


def main():
    args = parse_args()

    # Select gesture set
    gestures = GESTURES_6CL if args.n_classes == 6 else GESTURES_4CL
    print(f"\n══════ Live prediction capture ══════")
    print(f"  Label    : {args.label}")
    print(f"  Port     : {args.port}")
    print(f"  Classes  : {args.n_classes} ({', '.join(gestures)})")
    print(f"  Takes    : {args.n_takes} per gesture")
    print(f"  Duration : {args.duration}s per take")
    print(f"  Total    : ~{args.n_takes * len(gestures) * (args.duration + args.pause):.0f}s")

    # Open serial
    print(f"\nOpening serial port...")
    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.5)
    except serial.SerialException as e:
        print(f"❌ Could not open port: {e}")
        print("   Is Arduino Serial Monitor open ? Close it first.")
        sys.exit(1)
    time.sleep(2.0)
    flush_serial(ser)

    # Verify Teensy is responsive
    print("Reading 3 lines to confirm Teensy is alive...")
    alive = False
    for _ in range(20):
        line = ser.readline().decode(errors='replace').strip()
        if parse_teensy_line(line) is not None:
            print(f"  ✅ Got: {line[:80]}")
            alive = True
            break
    if not alive:
        print("❌ Teensy is not sending predictions. Check the sketch is running.")
        ser.close()
        sys.exit(1)

    # Ready prompt
    print("\n" + "=" * 60)
    print(" Put on the armband.")
    print(" Hold each gesture as steadily as you can during recording.")
    print(" Rest between recordings.")
    print("=" * 60)
    input(" Press Enter when ready ... ")

    # Record
    all_records = []
    start_time = time.time()

    for take_num in range(1, args.n_takes + 1):
        for gesture in gestures:
            show_gesture(gesture, take_num, args.n_takes)
            countdown("Get ready", secs=3)
            flush_serial(ser)
            print(f"  [REC] {gesture.upper()} — {args.duration:.0f}s ... ", end='', flush=True)
            recs = capture_phase(
                ser, gesture, args.duration,
                take_num, ACTION_TO_INT[gesture], start_time
            )
            print(f"captured {len(recs)} predictions")
            all_records.extend(recs)
            if not (take_num == args.n_takes and gesture == gestures[-1]):
                time.sleep(args.pause)

    ser.close()

    # Save CSV
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"live_{args.label}.csv"
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['Timestamp', 'Prediction', 'Action', 'Take']
        )
        writer.writeheader()
        writer.writerows(all_records)

    print(f"\n══════ Done ══════")
    print(f"  Captured {len(all_records)} prediction samples")
    print(f"  Saved to {out_path}")

    # Quick accuracy
    if all_records:
        correct = sum(1 for r in all_records if r['Prediction'] == r['Action'])
        acc = correct / len(all_records)
        print(f"  Live accuracy (raw, no smoothing): {acc * 100:.1f}%")


if __name__ == "__main__":
    main()

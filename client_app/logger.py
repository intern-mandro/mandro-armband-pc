import csv
import os
import re
import time
from datetime import datetime

import config


class CSVLogger:
    """CSV logger for EMG sessions.

    Format:
      Time(ms), Raw_CH0..N-1, Amp_CH0..N-1, Label, Window

    Output location:
      - With a profile:  data/<PROFILE>_BATCH<N>/<timestamp>_emg.csv
        This matches the folder convention the training pipeline reads
        (DATA_RAW / DATA_TEST_RAW in lib/configs.py): one folder per
        profile+batch, holding one CSV per recorded take.
      - Without a profile: legacy layout data/YYYY-MM-DD/<timestamp>_emg.csv

    Where:
      - Time(ms)  : absolute time since logger creation
      - Label     : current protocol label ("rest", "flexion", "pause", ...)
      - Window    : ms elapsed inside the current label step (resets at each label change)
    """

    def __init__(self, directory=None, buffer_size=600, profile=None, batch=None,
                 gesture_set=None):
        # Recordings go in the project's data/ folder (one level above the app dir)
        if directory is None:
            _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            directory = os.path.join(_project_root, "data")

        # Folder organisation
        if profile:
            safe_profile = re.sub(r"\s+", "_", str(profile).strip())
            try:
                batch_n = int(batch)
            except (TypeError, ValueError):
                batch_n = 1
            self.base_dir = os.path.join(directory, f"{safe_profile}_BATCH{batch_n}")
        else:
            # Fallback: legacy date-based layout
            date_folder = datetime.now().strftime("%Y-%m-%d")
            self.base_dir = os.path.join(directory, date_folder)

        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        # Stamp the gesture set as an explicit, recorded property of the batch.
        # Written once here, at folder creation — completely outside the
        # sample-writing loop, so it cannot affect recording quality/timing.
        if gesture_set:
            try:
                with open(os.path.join(self.base_dir, "gesture_set.txt"),
                          "w", encoding="utf-8") as _f:
                    _f.write(str(gesture_set).strip() + "\n")
            except Exception as _e:
                print(f"[Logger] could not stamp gesture_set: {_e}")

        filename = datetime.now().strftime("%Y%m%d_%H%M%S_emg.csv")
        self.filename = os.path.join(self.base_dir, filename)

        # Timing reference (monotonic)
        self.start_ts = time.perf_counter()

        # File handle
        self.file = open(self.filename, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)

        # Batch buffer
        self.buffer = []
        self.buffer_size = buffer_size
        self._header_written = False

        # Protocol state (controlled externally by ProtocolController)
        self.current_label = "none"          # current step label
        self.label_start_ts = self.start_ts  # when the current label started

    def _write_header(self):
        header = (
            ["Time(ms)"]
            + [f"Raw_CH{i}" for i in range(config.N_CH)]
            + [f"Amp_CH{i}" for i in range(config.N_CH)]
            + ["Label", "Window"]
        )
        self.writer.writerow(header)
        self._header_written = True

    def set_label(self, label: str):
        """Called by the protocol controller when the label changes.
        Resets the Window counter to 0."""
        self.current_label = label
        self.label_start_ts = time.perf_counter()
        print(f"[Logger] Label -> {label}")

    def write_row(self, raw_vals, amp_vals, timestamp=None):
        if not self._header_written:
            self._write_header()

        # Absolute time since logger start, in ms
        if timestamp is not None:
            time_ms = int(round(timestamp))
        else:
            time_ms = int(round((time.perf_counter() - self.start_ts) * 1000))

        # Window = ms elapsed within the current label step
        window_ms = int(round((time.perf_counter() - self.label_start_ts) * 1000))

        try:
            processed_raw = [int(float(v)) for v in raw_vals]
            processed_amp = [int(round(float(v))) for v in amp_vals]
            self.buffer.append(
                [time_ms]
                + processed_raw
                + processed_amp
                + [self.current_label, window_ms]
            )
            if len(self.buffer) >= self.buffer_size:
                self.flush()
        except (ValueError, TypeError) as e:
            print(f"Logger Error: {e}")

    def flush(self):
        if self.buffer:
            try:
                self.writer.writerows(self.buffer)
                self.file.flush()
                self.buffer.clear()
            except Exception as e:
                print(f"Flush Error: {e}")

    def close(self):
        self.flush()
        if not self.file.closed:
            self.file.close()
            print(f"Saved: {self.filename}")
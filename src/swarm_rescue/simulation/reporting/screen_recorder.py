import hashlib
import os
import queue
import threading
import time

import cv2
import numpy as np

from swarm_rescue.simulation.gui_map.top_down_view import TopDownView

# Bound queued frames so a slow encoder cannot grow memory without limit.
_VIDEO_QUEUE_MAX_FRAMES = 4

_DIAG_ENABLED = bool(os.environ.get("SWARM_RESCUE_VIDEO_DIAG", ""))


class ScreenRecorder:
    """
    Used to record a view and save it to a video file.
    It initializes the recorder with the parameters of the
    view, captures frames from the view, and stops the recording when
    needed.

    Example Usage
        # Create a ScreenRecorder object with the desired parameters
        recorder = ScreenRecorder(width=640, height=480, fps=30,
                                  out_file='output.avi')

        # Call the capture_frame method for each frame to record
        recorder.capture_frame(gui)

        # Stop the recording
        recorder.end_recording()
    """

    def __init__(self, width: int, height: int, fps: int, out_file: str):
        """
        Initialize the recorder with parameters of the view.

        Args:
            width (int): Width of the view to capture.
            height (int): Height of the view to capture.
            fps (int): Frames per second.
            out_file (str): Output file to save the recording.
        """
        self.video = None
        self._out_file = out_file
        self._frame_queue: queue.Queue[np.ndarray] | None = None
        self._stop_event: threading.Event | None = None
        self._writer_thread: threading.Thread | None = None
        self._writer_error: BaseException | None = None

        if out_file is None:
            return

        print("Initializing ScreenRecorder with parameters : width:{}, "
              "height:{}, fps:{}.".format(width, height, fps))

        four_cc = cv2.VideoWriter_fourcc(*'XVID')
        self.video = cv2.VideoWriter(out_file, four_cc, float(fps),
                                     (width, height))
        if not self.video.isOpened():
            raise RuntimeError(
                f"Failed to open video writer for {out_file!r} "
                f"({width}x{height} @ {fps} fps, XVID)."
            )

        self._frame_queue = queue.Queue(maxsize=_VIDEO_QUEUE_MAX_FRAMES)
        self._stop_event = threading.Event()
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="ScreenRecorderWriter",
            daemon=False,
        )
        self._writer_thread.start()

        self._frame_seq = 0
        self._hash_mismatches: list[int] = []
        if _DIAG_ENABLED:
            diag_dir = os.path.dirname(out_file)
            self._diag_ref_path = os.path.join(diag_dir, "_diag_ref_frame.png")
            self._diag_log_path = os.path.join(diag_dir, "_diag_hash_log.txt")
            self._diag_ref_saved = False
            print(f"[DIAG] Video diagnostics enabled. Ref={self._diag_ref_path}, Log={self._diag_log_path}")

    def _writer_loop(self) -> None:
        assert self._frame_queue is not None
        assert self.video is not None
        assert self._stop_event is not None

        diag_lines: list[str] = []

        while not (self._stop_event.is_set() and self._frame_queue.empty()):
            try:
                item = self._frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if _DIAG_ENABLED:
                seq, frame, expected_hash = item
                actual_hash = hashlib.md5(frame.tobytes()).hexdigest()
                if actual_hash != expected_hash:
                    self._hash_mismatches.append(seq)
                    msg = (f"[DIAG] HASH MISMATCH frame {seq}: "
                           f"enqueue={expected_hash[:8]}.. "
                           f"dequeue={actual_hash[:8]}.. "
                           f"shape={frame.shape} dtype={frame.dtype}")
                    print(msg)
                    diag_lines.append(msg + "\n")
            else:
                frame = item

            try:
                self.video.write(frame)
            except Exception as exc:
                self._writer_error = exc
                self._frame_queue.task_done()
                break
            else:
                self._frame_queue.task_done()

        if _DIAG_ENABLED and diag_lines:
            with open(self._diag_log_path, "w") as f:
                f.write(f"Total hash mismatches: {len(self._hash_mismatches)}\n")
                f.write(f"Mismatch frames: {self._hash_mismatches}\n")
                f.writelines(diag_lines)

    def _raise_writer_error(self) -> None:
        if self._writer_error is not None:
            raise self._writer_error

    def capture_frame(self, gui: TopDownView) -> None:
        """
        Call this method every frame to capture the current view.

        Args:
            gui (TopDownView): View to capture.
        """
        if self.video is None:
            return

        self._raise_writer_error()

        gui.update_and_draw_in_framebuffer()
        # Flip Y (OpenGL origin) and RGB->BGR for OpenCV; copy for the writer thread.
        img_capture = np.ascontiguousarray(gui.get_np_img()[::-1, :, ::-1])
        assert self._frame_queue is not None

        if _DIAG_ENABLED:
            frame_hash = hashlib.md5(img_capture.tobytes()).hexdigest()
            self._frame_queue.put((self._frame_seq, img_capture, frame_hash))
            if not self._diag_ref_saved:
                cv2.imwrite(self._diag_ref_path, img_capture)
                self._diag_ref_saved = True
                print(f"[DIAG] Reference frame saved to {self._diag_ref_path}")
        else:
            self._frame_queue.put(img_capture)

        self._frame_seq += 1

    def end_recording(self) -> None:
        """
        Call this method to stop recording and save the video.
        """
        if self.video is None:
            return

        assert self._stop_event is not None
        assert self._frame_queue is not None
        assert self._writer_thread is not None

        self._stop_event.set()
        self._frame_queue.join()
        self._writer_thread.join(timeout=120.0)
        if self._writer_thread.is_alive():
            raise TimeoutError(
                f"Timed out waiting for video writer thread ({self._out_file})."
            )

        self._raise_writer_error()
        self.video.release()
        self.video = None

        if _DIAG_ENABLED and self._hash_mismatches:
            print(f"[DIAG] {len(self._hash_mismatches)} hash mismatches: {self._hash_mismatches[:20]}...")
        elif _DIAG_ENABLED:
            print(f"[DIAG] All {self._frame_seq} frames: zero hash mismatches")

        print("\n")
        print("Output of the screen recording saved to {}."
              .format(self._out_file))

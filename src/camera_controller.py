"""
Camera Controller
Manages image capture from the camera placed at the holographic reconstruction plane.

Uses OpenCV as the primary interface. For industrial cameras (Basler, Allied Vision, etc.)
subclass CameraController and override open(), capture(), close().
"""

import time
import numpy as np
import cv2


class CameraController:
    """
    OpenCV-based camera controller for capturing holographic reconstruction images.
    """

    def __init__(
        self,
        device_index: int = 0,
        backend: str = "auto",
        width: int = 0,
        height: int = 0,
        exposure: float = -1,
        gain: float = -1,
        warmup_frames: int = 5,
    ):
        """
        Args:
            device_index:  OpenCV camera device index (0, 1, 2, ...)
            backend:       'auto', 'dshow' (Windows), 'v4l2' (Linux)
            width:         Capture width in pixels (0 = camera default)
            height:        Capture height in pixels (0 = camera default)
            exposure:      Camera exposure value (-1 = auto)
            gain:          Camera gain (-1 = auto)
            warmup_frames: Number of frames to discard at startup
        """
        self.device_index = device_index
        self.backend = backend
        self.width = width
        self.height = height
        self.exposure = exposure
        self.gain = gain
        self.warmup_frames = warmup_frames

        self._cap: cv2.VideoCapture | None = None

    @staticmethod
    def list_cameras(max_index: int = 10) -> list[int]:
        """Probe camera indices and return those that are available."""
        available = []
        for i in range(max_index):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available

    def open(self):
        """Open the camera device and apply settings."""
        backend_map = {
            "auto": cv2.CAP_ANY,
            "dshow": cv2.CAP_DSHOW,
            "v4l2": cv2.CAP_V4L2,
            "msmf": cv2.CAP_MSMF,
        }
        backend_flag = backend_map.get(self.backend, cv2.CAP_ANY)

        self._cap = cv2.VideoCapture(self.device_index, backend_flag)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera device {self.device_index} "
                f"(backend={self.backend})"
            )

        # Apply resolution
        if self.width > 0:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height > 0:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Apply exposure
        if self.exposure >= 0:
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # manual mode
            self._cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)
        else:
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # auto mode

        # Apply gain
        if self.gain >= 0:
            self._cap.set(cv2.CAP_PROP_GAIN, self.gain)

        # Discard warmup frames
        for _ in range(self.warmup_frames):
            self._cap.read()

    def capture(self) -> np.ndarray:
        """
        Capture a single frame.

        Returns:
            numpy array (H, W, 3) BGR uint8
        """
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("Camera is not open. Call open() first.")
        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise RuntimeError("Failed to capture frame from camera.")
        return frame

    def capture_average(self, num_frames: int, interval_ms: int = 100) -> np.ndarray:
        """
        Capture multiple frames and return their average.
        Useful for reducing noise.

        Args:
            num_frames:  Number of frames to average
            interval_ms: Interval between captures in milliseconds

        Returns:
            numpy array (H, W, 3) float64, values in [0, 255]
        """
        frames = []
        for i in range(num_frames):
            frame = self.capture()
            frames.append(frame.astype(np.float64))
            if i < num_frames - 1:
                time.sleep(interval_ms / 1000.0)
        averaged = np.mean(frames, axis=0)
        return averaged

    def capture_and_save(
        self,
        output_path: str,
        num_frames: int = 1,
        interval_ms: int = 100,
    ) -> np.ndarray:
        """
        Capture frame(s), save to disk, and return as numpy array.

        Args:
            output_path: Path to save the captured image (PNG, BMP, etc.)
            num_frames:  Number of frames to average
            interval_ms: Interval between captures

        Returns:
            Captured image as numpy array (H, W, 3) uint8
        """
        if num_frames > 1:
            frame_f = self.capture_average(num_frames, interval_ms)
            frame = np.clip(frame_f, 0, 255).astype(np.uint8)
        else:
            frame = self.capture()

        success = cv2.imwrite(output_path, frame)
        if not success:
            raise RuntimeError(f"Failed to save captured image to: {output_path}")
        return frame

    def get_resolution(self) -> tuple[int, int]:
        """Return current camera resolution as (width, height)."""
        if self._cap is None:
            return (0, 0)
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    def close(self):
        """Release the camera device."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

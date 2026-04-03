"""
Measurement Worker (QThread)
Performs the full measurement sequence in a background thread:
  1. Display CGH on SLM
  2. Wait for optical stabilization
  3. Capture reconstruction with camera
  4. Save captured image
  5. Compute LPIPS with target image
  6. Emit progress signals to GUI
"""

import time
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import cv2
from PyQt5.QtCore import QThread, pyqtSignal

from .slm_controller import SLMWindow
from .camera_controller import CameraController
from .lpips_calculator import LPIPSCalculator


@dataclass
class MeasurementConfig:
    """All parameters needed for a measurement run."""
    cgh_folder: str = ""
    target_folder: str = ""
    output_folder: str = ""

    # SLM
    slm_screen_index: int = 1
    slm_display_mode: str = "fit"

    # Camera
    camera_device_index: int = 0
    camera_backend: str = "auto"
    camera_width: int = 0
    camera_height: int = 0
    camera_exposure: float = -1
    camera_gain: float = -1
    camera_warmup_frames: int = 5

    # Timing
    stabilization_delay_ms: int = 500
    num_captures: int = 1
    capture_interval_ms: int = 100

    # LPIPS
    lpips_network: str = "alex"
    lpips_spatial: bool = False
    lpips_resize_to: int = 256

    # Pairing
    pair_mode: str = "sorted"  # 'sorted' or 'name'

    # Additional metrics
    compute_psnr: bool = True
    compute_ssim: bool = True


@dataclass
class MeasurementResult:
    """Result of a single CGH-target pair measurement."""
    index: int
    cgh_path: str
    target_path: str
    captured_path: str
    lpips: float
    psnr: float = float("nan")
    ssim: float = float("nan")
    error: Optional[str] = None


SUPPORTED_EXTENSIONS = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"}


def collect_image_paths(folder: str) -> list[Path]:
    """Return sorted list of image paths in folder."""
    p = Path(folder)
    paths = sorted(
        f for f in p.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    return paths


def pair_images(
    cgh_paths: list[Path],
    target_paths: list[Path],
    mode: str,
) -> list[tuple[Path, Path]]:
    """
    Pair CGH and target images.

    mode='sorted': pair by sort order (index 0 <-> 0, 1 <-> 1, ...)
    mode='name':   pair by filename stem (exact match required)
    """
    if mode == "name":
        target_map = {p.stem: p for p in target_paths}
        pairs = []
        for cgh in cgh_paths:
            tgt = target_map.get(cgh.stem)
            if tgt is None:
                raise ValueError(
                    f"No matching target found for CGH '{cgh.name}'. "
                    f"Expected a file named '{cgh.stem}.*' in the target folder."
                )
            pairs.append((cgh, tgt))
        return pairs
    else:  # sorted
        n = min(len(cgh_paths), len(target_paths))
        if n == 0:
            raise ValueError("No image pairs found.")
        return list(zip(cgh_paths[:n], target_paths[:n]))


class MeasurementWorker(QThread):
    """
    Background thread that runs the full holographic measurement sequence.
    """

    # Emitted when a pair is about to start; provides (index, total, cgh_path)
    progress = pyqtSignal(int, int, str)

    # Emitted after CGH is displayed on SLM; provides (cgh_image_as_ndarray,)
    slm_updated = pyqtSignal(object)

    # Emitted after camera capture; provides (captured_image_as_ndarray,)
    captured = pyqtSignal(object)

    # Emitted when a measurement result is ready
    result_ready = pyqtSignal(object)  # MeasurementResult

    # Emitted with a log message
    log = pyqtSignal(str)

    # Emitted when all measurements are finished
    finished = pyqtSignal(list)  # list[MeasurementResult]

    # Emitted on fatal error
    error = pyqtSignal(str)

    def __init__(self, config: MeasurementConfig, slm_window: SLMWindow):
        super().__init__()
        self.config = config
        self.slm_window = slm_window
        self._stop_flag = False

    def stop(self):
        """Request graceful stop."""
        self._stop_flag = True

    def run(self):
        """Main measurement loop (runs in background thread)."""
        config = self.config
        results: list[MeasurementResult] = []

        try:
            # Collect image paths
            cgh_paths = collect_image_paths(config.cgh_folder)
            target_paths = collect_image_paths(config.target_folder)

            if not cgh_paths:
                self.error.emit(f"No images found in CGH folder: {config.cgh_folder}")
                return
            if not target_paths:
                self.error.emit(f"No images found in target folder: {config.target_folder}")
                return

            pairs = pair_images(cgh_paths, target_paths, config.pair_mode)
            total = len(pairs)
            self.log.emit(f"Found {total} CGH-target pairs. Starting measurement...")

            # Create output folder
            os.makedirs(config.output_folder, exist_ok=True)

            # Initialize LPIPS calculator
            self.log.emit(f"Loading LPIPS model ({config.lpips_network})...")
            calc = LPIPSCalculator(
                network=config.lpips_network,
                spatial=config.lpips_spatial,
                resize_to=config.lpips_resize_to,
            )
            self.log.emit("LPIPS model loaded.")

            # Open camera
            camera = CameraController(
                device_index=config.camera_device_index,
                backend=config.camera_backend,
                width=config.camera_width,
                height=config.camera_height,
                exposure=config.camera_exposure,
                gain=config.camera_gain,
                warmup_frames=config.camera_warmup_frames,
            )
            self.log.emit(f"Opening camera (device {config.camera_device_index})...")
            camera.open()
            w, h = camera.get_resolution()
            self.log.emit(f"Camera opened: {w}x{h}")

            try:
                for idx, (cgh_path, target_path) in enumerate(pairs):
                    if self._stop_flag:
                        self.log.emit("Measurement stopped by user.")
                        break

                    self.progress.emit(idx + 1, total, str(cgh_path))
                    self.log.emit(
                        f"[{idx+1}/{total}] CGH: {cgh_path.name} | Target: {target_path.name}"
                    )

                    result = self._measure_one(
                        idx, cgh_path, target_path, camera, calc
                    )
                    results.append(result)
                    self.result_ready.emit(result)

                    if result.error:
                        self.log.emit(f"  ERROR: {result.error}")
                    else:
                        msg = f"  LPIPS={result.lpips:.4f}"
                        if not np.isnan(result.psnr):
                            msg += f"  PSNR={result.psnr:.2f}dB"
                        if not np.isnan(result.ssim):
                            msg += f"  SSIM={result.ssim:.4f}"
                        self.log.emit(msg)

            finally:
                camera.close()
                self.slm_window.clear()

        except Exception as e:
            self.error.emit(str(e))
            return

        self.log.emit(f"\nMeasurement complete. {len(results)} results.")
        self.finished.emit(results)

    def _measure_one(
        self,
        idx: int,
        cgh_path: Path,
        target_path: Path,
        camera: CameraController,
        calc: LPIPSCalculator,
    ) -> MeasurementResult:
        config = self.config

        # Output filename: same stem as CGH
        out_name = f"captured_{cgh_path.stem}.png"
        captured_path = str(Path(config.output_folder) / out_name)

        try:
            # 1. Display CGH on SLM
            cgh_img = cv2.imread(str(cgh_path))
            self.slm_window.display_image(str(cgh_path))

            if cgh_img is not None:
                self.slm_updated.emit(cv2.cvtColor(cgh_img, cv2.COLOR_BGR2RGB))

            # 2. Wait for optical stabilization
            time.sleep(config.stabilization_delay_ms / 1000.0)

            # 3. Capture
            frame = camera.capture_and_save(
                captured_path,
                num_frames=config.num_captures,
                interval_ms=config.capture_interval_ms,
            )
            self.captured.emit(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # 4. LPIPS
            lpips_score = calc.compute(captured_path, str(target_path))

            # 5. Supplementary metrics
            psnr = float("nan")
            ssim = float("nan")
            if config.compute_psnr:
                psnr = LPIPSCalculator.compute_psnr(
                    captured_path, str(target_path), config.lpips_resize_to
                )
            if config.compute_ssim:
                ssim = LPIPSCalculator.compute_ssim(
                    captured_path, str(target_path), config.lpips_resize_to
                )

            return MeasurementResult(
                index=idx,
                cgh_path=str(cgh_path),
                target_path=str(target_path),
                captured_path=captured_path,
                lpips=lpips_score,
                psnr=psnr,
                ssim=ssim,
            )

        except Exception as e:
            return MeasurementResult(
                index=idx,
                cgh_path=str(cgh_path),
                target_path=str(target_path),
                captured_path=captured_path,
                lpips=float("nan"),
                error=str(e),
            )

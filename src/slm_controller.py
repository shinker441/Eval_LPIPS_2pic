"""
SLM Controller
Manages the full-screen display window projected onto the SLM (Spatial Light Modulator).
The SLM is connected as a secondary monitor via HDMI/DVI.

Important: For phase-only SLMs, CGH pixel values (0-255) directly encode phase (0-2π).
The OS must not apply gamma correction or color space conversion.
"""

import numpy as np
from PyQt5.QtWidgets import QWidget, QLabel, QApplication
from PyQt5.QtGui import QPixmap, QImage, QPalette, QColor
from PyQt5.QtCore import Qt


class SLMWindow(QWidget):
    """
    Borderless full-screen window displayed on the SLM monitor.
    The SLM appears as a secondary display to the OS.
    """

    def __init__(self, screen_index: int = 1, display_mode: str = "fit"):
        """
        Args:
            screen_index: Index of the display to use as SLM
                          (0=primary monitor, 1=first secondary, ...)
            display_mode: How to scale the CGH image on the SLM
                          'fit'   - preserve aspect ratio, letterbox
                          'fill'  - stretch to fill screen
                          'native'- display at original pixel size, centered
        """
        super().__init__()
        self.screen_index = screen_index
        self.display_mode = display_mode
        self._setup_ui()

    @staticmethod
    def available_screens() -> list[str]:
        """Return descriptions of all available screens."""
        screens = QApplication.screens()
        result = []
        for i, screen in enumerate(screens):
            g = screen.geometry()
            name = screen.name()
            result.append(f"[{i}] {name}  {g.width()}x{g.height()} @ ({g.x()},{g.y()})")
        return result

    def _setup_ui(self):
        self.setWindowTitle("SLM Display")
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
        )

        # Black background
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor("black"))
        self.setPalette(palette)
        self.setAutoFillBackground(True)

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("background-color: black;")

        self._move_to_screen()

    def _move_to_screen(self):
        screens = QApplication.screens()
        if self.screen_index >= len(screens):
            raise ValueError(
                f"Screen index {self.screen_index} is out of range. "
                f"Available screens: {len(screens)}"
            )
        screen = screens[self.screen_index]
        geometry = screen.geometry()
        self.setGeometry(geometry)
        self.label.setGeometry(0, 0, geometry.width(), geometry.height())

    def show_fullscreen(self):
        """Make the SLM window visible and full-screen."""
        screens = QApplication.screens()
        if self.screen_index < len(screens):
            screen = screens[self.screen_index]
            geometry = screen.geometry()
            self.setGeometry(geometry)
            self.label.setGeometry(0, 0, geometry.width(), geometry.height())
        self.showFullScreen()
        QApplication.processEvents()

    def display_image(self, image: "str | np.ndarray"):
        """
        Display a CGH image on the SLM.

        Args:
            image: File path (str) or numpy array (H,W) grayscale uint8
                   or (H,W,3) RGB uint8
        """
        if isinstance(image, str):
            pixmap = QPixmap(image)
            if pixmap.isNull():
                raise FileNotFoundError(f"Cannot load image: {image}")
        elif isinstance(image, np.ndarray):
            pixmap = self._ndarray_to_pixmap(image)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        screens = QApplication.screens()
        if self.screen_index < len(screens):
            screen = screens[self.screen_index]
            geometry = screen.geometry()
            screen_w, screen_h = geometry.width(), geometry.height()
        else:
            screen_w, screen_h = self.width(), self.height()

        if self.display_mode == "fit":
            scaled = pixmap.scaled(
                screen_w, screen_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        elif self.display_mode == "fill":
            scaled = pixmap.scaled(
                screen_w, screen_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
            )
        else:  # native
            scaled = pixmap  # display at original size, centered

        self.label.setPixmap(scaled)
        QApplication.processEvents()

    def clear(self):
        """Clear the SLM display (show solid black)."""
        self.label.clear()
        QApplication.processEvents()

    @staticmethod
    def _ndarray_to_pixmap(arr: np.ndarray) -> QPixmap:
        arr = np.ascontiguousarray(arr)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        if arr.ndim == 2:
            h, w = arr.shape
            qimg = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
        elif arr.ndim == 3 and arr.shape[2] == 3:
            h, w, _ = arr.shape
            qimg = QImage(arr.data, w, h, w * 3, QImage.Format_RGB888)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            h, w, _ = arr.shape
            qimg = QImage(arr.data, w, h, w * 4, QImage.Format_RGBA8888)
        else:
            raise ValueError(f"Unsupported array shape: {arr.shape}")

        return QPixmap.fromImage(qimg)

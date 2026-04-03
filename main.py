"""
LPIPS Holographic Evaluation System
Entry point

Usage:
    python main.py

Requirements:
    pip install -r requirements.txt

Setup:
    1. Connect SLM to PC as a secondary display (HDMI/DVI).
    2. Connect camera to PC (USB / GigE).
    3. Place CGH images in a folder (BMP/PNG/TIFF).
    4. Place target images in another folder (same number of files).
    5. Run this application and configure via the GUI.

Notes for SLM:
    - The SLM must be set to "Extended Desktop" mode in Windows display settings,
      NOT mirroring. The application displays the CGH as a full-screen window
      on the selected secondary display.
    - For phase-only SLMs: ensure Windows color management and gamma correction
      are disabled for the SLM display to preserve CGH phase values.
    - Typical SLM display index: 1 (primary=0, SLM=1).
"""

import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("LPIPS Holographic Evaluation System")
    app.setOrganizationName("HolographyLab")

    # Set default font
    font = QFont("Yu Gothic UI", 9)
    app.setFont(font)

    # Dark palette (optional)
    app.setStyle("Fusion")

    from src.main_window import MainWindow
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

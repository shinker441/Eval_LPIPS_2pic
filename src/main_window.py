"""
Main Application Window
PyQt5 GUI for the LPIPS holographic evaluation system.
"""

import csv
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QGroupBox, QProgressBar, QTextEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog,
    QSplitter, QTabWidget, QMessageBox, QStatusBar, QFrame,
    QSizePolicy,
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor
from PyQt5.QtCore import Qt, pyqtSlot

from .slm_controller import SLMWindow
from .measurement_worker import MeasurementConfig, MeasurementResult, MeasurementWorker


# ──────────────────────────────────────────────────────────────────────────────
# Small helper: convert numpy (H,W,3) RGB array to QPixmap for preview
# ──────────────────────────────────────────────────────────────────────────────
def ndarray_to_pixmap(arr: np.ndarray, max_size: int = 300) -> QPixmap:
    if arr is None:
        return QPixmap()
    arr = np.ascontiguousarray(arr)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.shape[2] == 4:
        arr = arr[:, :, :3]
    h, w, _ = arr.shape
    qimg = QImage(arr.data, w, h, w * 3, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(qimg)
    return pixmap.scaled(max_size, max_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)


class ImagePreviewLabel(QLabel):
    """Fixed-size preview label with title."""
    def __init__(self, title: str, size: int = 280):
        super().__init__()
        self._size = size
        self.setFixedSize(size, size + 20)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            "border: 1px solid #aaa; background: #1a1a1a; color: #888;"
        )
        self.setText(f"[{title}]")
        self._title = title

    def set_image(self, arr: np.ndarray):
        pixmap = ndarray_to_pixmap(arr, self._size)
        self.setPixmap(pixmap)

    def clear_image(self):
        self.setPixmap(QPixmap())
        self.setText(f"[{self._title}]")


# ──────────────────────────────────────────────────────────────────────────────
# Folder selector row
# ──────────────────────────────────────────────────────────────────────────────
class FolderSelector(QWidget):
    def __init__(self, label: str, placeholder: str = ""):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.line_edit = QLineEdit()
        self.line_edit.setPlaceholderText(placeholder)
        btn = QPushButton("...")
        btn.setFixedWidth(36)
        btn.clicked.connect(self._browse)

        layout.addWidget(QLabel(label))
        layout.addWidget(self.line_edit, 1)
        layout.addWidget(btn)

    def _browse(self):
        folder = QFileDialog.getExistingDirectory(self, "フォルダを選択")
        if folder:
            self.line_edit.setText(folder)

    @property
    def path(self) -> str:
        return self.line_edit.text().strip()

    @path.setter
    def path(self, value: str):
        self.line_edit.setText(value)


# ──────────────────────────────────────────────────────────────────────────────
# Settings Panel
# ──────────────────────────────────────────────────────────────────────────────
class SettingsPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumWidth(340)
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Folder settings
        folder_group = QGroupBox("フォルダ設定")
        fg = QVBoxLayout(folder_group)
        self.cgh_folder = FolderSelector("CGH フォルダ :", "CGH画像が格納されたフォルダ")
        self.target_folder = FolderSelector("ターゲット    :", "ターゲット画像フォルダ")
        self.output_folder = FolderSelector("出力フォルダ :", "撮像画像の保存先")
        fg.addWidget(self.cgh_folder)
        fg.addWidget(self.target_folder)
        fg.addWidget(self.output_folder)
        layout.addWidget(folder_group)

        # SLM settings
        slm_group = QGroupBox("SLM 設定")
        sg = QGridLayout(slm_group)
        sg.addWidget(QLabel("ディスプレイ番号:"), 0, 0)
        self.slm_screen = QSpinBox()
        self.slm_screen.setRange(0, 8)
        self.slm_screen.setValue(1)
        self.slm_screen.setToolTip("0=プライマリ, 1=SLM (通常), ...")
        sg.addWidget(self.slm_screen, 0, 1)

        sg.addWidget(QLabel("表示モード:"), 1, 0)
        self.slm_mode = QComboBox()
        self.slm_mode.addItems(["fit (アスペクト比保持)", "fill (引き伸ばし)", "native (等倍)"])
        sg.addWidget(self.slm_mode, 1, 1)

        self.btn_detect_screens = QPushButton("接続ディスプレイを確認")
        sg.addWidget(self.btn_detect_screens, 2, 0, 1, 2)
        self.btn_detect_screens.clicked.connect(self._detect_screens)
        self.screen_info_label = QLabel("")
        self.screen_info_label.setWordWrap(True)
        self.screen_info_label.setStyleSheet("font-size: 10px; color: #555;")
        sg.addWidget(self.screen_info_label, 3, 0, 1, 2)
        layout.addWidget(slm_group)

        # Camera settings
        cam_group = QGroupBox("カメラ設定")
        cg = QGridLayout(cam_group)
        cg.addWidget(QLabel("デバイス番号:"), 0, 0)
        self.cam_device = QSpinBox()
        self.cam_device.setRange(0, 16)
        cg.addWidget(self.cam_device, 0, 1)

        cg.addWidget(QLabel("バックエンド:"), 1, 0)
        self.cam_backend = QComboBox()
        self.cam_backend.addItems(["auto", "dshow", "msmf", "v4l2"])
        cg.addWidget(self.cam_backend, 1, 1)

        cg.addWidget(QLabel("解像度 W:"), 2, 0)
        self.cam_width = QSpinBox()
        self.cam_width.setRange(0, 7680)
        self.cam_width.setValue(0)
        self.cam_width.setSpecialValueText("自動")
        cg.addWidget(self.cam_width, 2, 1)

        cg.addWidget(QLabel("解像度 H:"), 3, 0)
        self.cam_height = QSpinBox()
        self.cam_height.setRange(0, 4320)
        self.cam_height.setValue(0)
        self.cam_height.setSpecialValueText("自動")
        cg.addWidget(self.cam_height, 3, 1)

        cg.addWidget(QLabel("露出 (-1=自動):"), 4, 0)
        self.cam_exposure = QDoubleSpinBox()
        self.cam_exposure.setRange(-1, 1000)
        self.cam_exposure.setValue(-1)
        self.cam_exposure.setSingleStep(0.5)
        cg.addWidget(self.cam_exposure, 4, 1)

        cg.addWidget(QLabel("ゲイン (-1=自動):"), 5, 0)
        self.cam_gain = QDoubleSpinBox()
        self.cam_gain.setRange(-1, 1000)
        self.cam_gain.setValue(-1)
        self.cam_gain.setSingleStep(1.0)
        cg.addWidget(self.cam_gain, 5, 1)

        cg.addWidget(QLabel("ウォームアップ枚数:"), 6, 0)
        self.cam_warmup = QSpinBox()
        self.cam_warmup.setRange(0, 50)
        self.cam_warmup.setValue(5)
        cg.addWidget(self.cam_warmup, 6, 1)
        layout.addWidget(cam_group)

        # Measurement settings
        meas_group = QGroupBox("計測設定")
        mg = QGridLayout(meas_group)
        mg.addWidget(QLabel("安定待機時間 (ms):"), 0, 0)
        self.stab_delay = QSpinBox()
        self.stab_delay.setRange(0, 30000)
        self.stab_delay.setValue(500)
        self.stab_delay.setSingleStep(100)
        mg.addWidget(self.stab_delay, 0, 1)

        mg.addWidget(QLabel("撮像枚数 (平均用):"), 1, 0)
        self.num_captures = QSpinBox()
        self.num_captures.setRange(1, 100)
        self.num_captures.setValue(1)
        mg.addWidget(self.num_captures, 1, 1)

        mg.addWidget(QLabel("撮像間隔 (ms):"), 2, 0)
        self.capture_interval = QSpinBox()
        self.capture_interval.setRange(0, 10000)
        self.capture_interval.setValue(100)
        mg.addWidget(self.capture_interval, 2, 1)

        mg.addWidget(QLabel("ペアリング方式:"), 3, 0)
        self.pair_mode = QComboBox()
        self.pair_mode.addItems(["sorted (ソート順)", "name (ファイル名一致)"])
        mg.addWidget(self.pair_mode, 3, 1)
        layout.addWidget(meas_group)

        # LPIPS settings
        lpips_group = QGroupBox("LPIPS 設定")
        lg = QGridLayout(lpips_group)
        lg.addWidget(QLabel("ネットワーク:"), 0, 0)
        self.lpips_network = QComboBox()
        self.lpips_network.addItems(["alex", "vgg"])
        lg.addWidget(self.lpips_network, 0, 1)

        lg.addWidget(QLabel("リサイズ (px, 0=なし):"), 1, 0)
        self.lpips_resize = QSpinBox()
        self.lpips_resize.setRange(0, 4096)
        self.lpips_resize.setValue(256)
        self.lpips_resize.setSingleStep(64)
        lg.addWidget(self.lpips_resize, 1, 1)

        self.chk_psnr = QCheckBox("PSNR も計算")
        self.chk_psnr.setChecked(True)
        self.chk_ssim = QCheckBox("SSIM も計算")
        self.chk_ssim.setChecked(True)
        lg.addWidget(self.chk_psnr, 2, 0)
        lg.addWidget(self.chk_ssim, 2, 1)
        layout.addWidget(lpips_group)

        layout.addStretch()

    def _detect_screens(self):
        from PyQt5.QtWidgets import QApplication
        screens = QApplication.screens()
        lines = []
        for i, s in enumerate(screens):
            g = s.geometry()
            lines.append(f"[{i}] {s.name()} {g.width()}×{g.height()} @({g.x()},{g.y()})")
        self.screen_info_label.setText("\n".join(lines))

    def slm_display_mode(self) -> str:
        return self.slm_mode.currentText().split()[0]

    def pair_mode_value(self) -> str:
        return self.pair_mode.currentText().split()[0]

    def build_config(self) -> MeasurementConfig:
        return MeasurementConfig(
            cgh_folder=self.cgh_folder.path,
            target_folder=self.target_folder.path,
            output_folder=self.output_folder.path,
            slm_screen_index=self.slm_screen.value(),
            slm_display_mode=self.slm_display_mode(),
            camera_device_index=self.cam_device.value(),
            camera_backend=self.cam_backend.currentText(),
            camera_width=self.cam_width.value(),
            camera_height=self.cam_height.value(),
            camera_exposure=self.cam_exposure.value(),
            camera_gain=self.cam_gain.value(),
            camera_warmup_frames=self.cam_warmup.value(),
            stabilization_delay_ms=self.stab_delay.value(),
            num_captures=self.num_captures.value(),
            capture_interval_ms=self.capture_interval.value(),
            lpips_network=self.lpips_network.currentText(),
            lpips_resize_to=self.lpips_resize.value(),
            compute_psnr=self.chk_psnr.isChecked(),
            compute_ssim=self.chk_ssim.isChecked(),
            pair_mode=self.pair_mode_value(),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Results Table
# ──────────────────────────────────────────────────────────────────────────────
class ResultsTable(QTableWidget):
    COLUMNS = ["#", "CGH ファイル", "ターゲット", "撮像画像", "LPIPS ↓", "PSNR (dB) ↑", "SSIM ↑"]

    def __init__(self):
        super().__init__(0, len(self.COLUMNS))
        self.setHorizontalHeaderLabels(self.COLUMNS)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setAlternatingRowColors(True)

    def add_result(self, r: MeasurementResult):
        row = self.rowCount()
        self.insertRow(row)

        def cell(text, align=Qt.AlignCenter):
            item = QTableWidgetItem(str(text))
            item.setTextAlignment(align)
            return item

        self.setItem(row, 0, cell(r.index + 1))
        self.setItem(row, 1, cell(Path(r.cgh_path).name, Qt.AlignLeft | Qt.AlignVCenter))
        self.setItem(row, 2, cell(Path(r.target_path).name, Qt.AlignLeft | Qt.AlignVCenter))
        self.setItem(row, 3, cell(Path(r.captured_path).name, Qt.AlignLeft | Qt.AlignVCenter))

        if r.error:
            err_item = cell(f"ERROR: {r.error}", Qt.AlignLeft | Qt.AlignVCenter)
            err_item.setForeground(QColor("red"))
            self.setItem(row, 4, err_item)
        else:
            lpips_item = cell(f"{r.lpips:.4f}")
            # Color coding: green < 0.1, yellow < 0.3, red >= 0.3
            if r.lpips < 0.1:
                lpips_item.setForeground(QColor("#4caf50"))
            elif r.lpips < 0.3:
                lpips_item.setForeground(QColor("#ff9800"))
            else:
                lpips_item.setForeground(QColor("#f44336"))
            self.setItem(row, 4, lpips_item)

            psnr_text = f"{r.psnr:.2f}" if not (r.psnr != r.psnr) else "—"
            ssim_text = f"{r.ssim:.4f}" if not (r.ssim != r.ssim) else "—"
            self.setItem(row, 5, cell(psnr_text))
            self.setItem(row, 6, cell(ssim_text))

        self.scrollToBottom()

    def clear_results(self):
        self.setRowCount(0)

    def export_csv(self, path: str):
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(self.COLUMNS)
            for row in range(self.rowCount()):
                writer.writerow(
                    [self.item(row, col).text() if self.item(row, col) else ""
                     for col in range(self.columnCount())]
                )


# ──────────────────────────────────────────────────────────────────────────────
# Main Window
# ──────────────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LPIPS Holographic Evaluation System")
        self.resize(1400, 860)

        self._worker: MeasurementWorker | None = None
        self._slm_window: SLMWindow | None = None
        self._results: list[MeasurementResult] = []

        self._build_ui()
        self._build_menu()

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setSpacing(8)

        # Left: settings panel
        self.settings_panel = SettingsPanel()
        root.addWidget(self.settings_panel)

        # Vertical separator
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setFrameShadow(QFrame.Sunken)
        root.addWidget(sep)

        # Right: content area
        right = QVBoxLayout()
        right.setSpacing(6)

        # Image previews
        preview_group = QGroupBox("画像プレビュー")
        prev_layout = QHBoxLayout(preview_group)
        self.prev_cgh = ImagePreviewLabel("CGH")
        self.prev_captured = ImagePreviewLabel("撮像 (カメラ)")
        self.prev_target = ImagePreviewLabel("ターゲット")
        prev_layout.addWidget(self._titled_preview("CGH", self.prev_cgh))
        prev_layout.addWidget(self._titled_preview("撮像 (カメラ)", self.prev_captured))
        prev_layout.addWidget(self._titled_preview("ターゲット", self.prev_target))
        prev_layout.addStretch()
        right.addWidget(preview_group)

        # Progress + controls
        ctrl_layout = QHBoxLayout()
        self.btn_start = QPushButton("▶ 計測開始")
        self.btn_start.setFixedHeight(36)
        self.btn_start.setStyleSheet("background:#4caf50; color:white; font-weight:bold;")
        self.btn_stop = QPushButton("■ 停止")
        self.btn_stop.setFixedHeight(36)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("background:#f44336; color:white; font-weight:bold;")
        self.btn_export = QPushButton("CSV エクスポート")
        self.btn_export.setFixedHeight(36)
        self.btn_export.setEnabled(False)
        self.btn_clear = QPushButton("結果をクリア")
        self.btn_clear.setFixedHeight(36)

        self.btn_start.clicked.connect(self._start_measurement)
        self.btn_stop.clicked.connect(self._stop_measurement)
        self.btn_export.clicked.connect(self._export_csv)
        self.btn_clear.clicked.connect(self._clear_results)

        ctrl_layout.addWidget(self.btn_start)
        ctrl_layout.addWidget(self.btn_stop)
        ctrl_layout.addStretch()
        ctrl_layout.addWidget(self.btn_export)
        ctrl_layout.addWidget(self.btn_clear)
        right.addLayout(ctrl_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setValue(0)
        right.addWidget(self.progress_bar)

        # Tabs: results table / log
        tabs = QTabWidget()

        self.results_table = ResultsTable()
        tabs.addTab(self.results_table, "計測結果")

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFont(QFont("Consolas", 9))
        self.log_edit.setStyleSheet("background:#1e1e1e; color:#d4d4d4;")
        tabs.addTab(self.log_edit, "ログ")

        right.addWidget(tabs)

        # Summary label
        self.summary_label = QLabel("計測前")
        self.summary_label.setAlignment(Qt.AlignRight)
        right.addWidget(self.summary_label)

        root.addLayout(right, 1)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("準備完了")

    def _titled_preview(self, title: str, label: ImagePreviewLabel) -> QWidget:
        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(2)
        t = QLabel(title)
        t.setAlignment(Qt.AlignCenter)
        t.setStyleSheet("font-weight:bold; font-size:11px;")
        vl.addWidget(t)
        vl.addWidget(label)
        return w

    def _build_menu(self):
        menu = self.menuBar()
        file_menu = menu.addMenu("ファイル(&F)")

        act_export = file_menu.addAction("CSV エクスポート(&E)")
        act_export.triggered.connect(self._export_csv)
        file_menu.addSeparator()
        act_quit = file_menu.addAction("終了(&Q)")
        act_quit.triggered.connect(self.close)

        help_menu = menu.addMenu("ヘルプ(&H)")
        act_about = help_menu.addAction("このアプリについて")
        act_about.triggered.connect(self._show_about)

    # ── Measurement Control ───────────────────────────────────────────────────

    def _start_measurement(self):
        config = self.settings_panel.build_config()

        # Validate inputs
        errors = []
        if not config.cgh_folder or not Path(config.cgh_folder).is_dir():
            errors.append("CGH フォルダが無効です。")
        if not config.target_folder or not Path(config.target_folder).is_dir():
            errors.append("ターゲットフォルダが無効です。")
        if not config.output_folder:
            errors.append("出力フォルダを指定してください。")
        if errors:
            QMessageBox.warning(self, "入力エラー", "\n".join(errors))
            return

        # Create/re-open SLM window
        try:
            if self._slm_window is not None:
                self._slm_window.close()
            self._slm_window = SLMWindow(
                screen_index=config.slm_screen_index,
                display_mode=config.slm_display_mode,
            )
            self._slm_window.show_fullscreen()
        except ValueError as e:
            QMessageBox.critical(self, "SLM エラー", str(e))
            return

        # Clear previous results
        self._results.clear()
        self.results_table.clear_results()
        self.log_edit.clear()
        self.progress_bar.setValue(0)
        self.btn_export.setEnabled(False)

        # Start worker thread
        self._worker = MeasurementWorker(config, self._slm_window)
        self._worker.progress.connect(self._on_progress)
        self._worker.slm_updated.connect(self._on_slm_updated)
        self._worker.captured.connect(self._on_captured)
        self._worker.result_ready.connect(self._on_result_ready)
        self._worker.log.connect(self._on_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        self._worker.start()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.status_bar.showMessage("計測中...")

    def _stop_measurement(self):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self.status_bar.showMessage("停止処理中...")

    # ── Worker Slots ──────────────────────────────────────────────────────────

    @pyqtSlot(int, int, str)
    def _on_progress(self, current: int, total: int, cgh_path: str):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.progress_bar.setFormat(f"{current}/{total}  —  {Path(cgh_path).name}")

    @pyqtSlot(object)
    def _on_slm_updated(self, img: np.ndarray):
        self.prev_cgh.set_image(img)

    @pyqtSlot(object)
    def _on_captured(self, img: np.ndarray):
        self.prev_captured.set_image(img)

    @pyqtSlot(object)
    def _on_result_ready(self, result: MeasurementResult):
        self._results.append(result)
        self.results_table.add_result(result)

        # Load target thumbnail
        if Path(result.target_path).exists():
            tgt = cv2.imread(result.target_path)
            if tgt is not None:
                self.prev_target.set_image(cv2.cvtColor(tgt, cv2.COLOR_BGR2RGB))

        self._update_summary()

    @pyqtSlot(str)
    def _on_log(self, message: str):
        self.log_edit.append(message)

    @pyqtSlot(list)
    def _on_finished(self, results: list):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_export.setEnabled(len(results) > 0)
        self.status_bar.showMessage(f"完了: {len(results)} ペア計測済み")
        self._update_summary()

        if self._slm_window:
            self._slm_window.clear()

        # Auto-save CSV
        if results:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            auto_path = str(Path(self.settings_panel.output_folder.path) / f"results_{ts}.csv")
            try:
                self.results_table.export_csv(auto_path)
                self._on_log(f"\nCSV 自動保存: {auto_path}")
            except Exception as e:
                self._on_log(f"\nCSV 自動保存失敗: {e}")

    @pyqtSlot(str)
    def _on_error(self, message: str):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status_bar.showMessage("エラーが発生しました")
        QMessageBox.critical(self, "計測エラー", message)

    # ── Actions ───────────────────────────────────────────────────────────────

    def _export_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "CSV として保存", "results.csv", "CSV ファイル (*.csv)"
        )
        if path:
            self.results_table.export_csv(path)
            self.status_bar.showMessage(f"保存完了: {path}")

    def _clear_results(self):
        self._results.clear()
        self.results_table.clear_results()
        self.log_edit.clear()
        self.progress_bar.setValue(0)
        self.summary_label.setText("計測前")
        self.btn_export.setEnabled(False)
        self.prev_cgh.clear_image()
        self.prev_captured.clear_image()
        self.prev_target.clear_image()

    def _update_summary(self):
        if not self._results:
            return
        valid = [r for r in self._results if r.error is None]
        if not valid:
            self.summary_label.setText(f"計測: {len(self._results)}  |  全てエラー")
            return
        avg_lpips = sum(r.lpips for r in valid) / len(valid)
        psnr_vals = [r.psnr for r in valid if r.psnr == r.psnr]
        ssim_vals = [r.ssim for r in valid if r.ssim == r.ssim]
        parts = [
            f"計測: {len(self._results)}",
            f"平均 LPIPS: {avg_lpips:.4f}",
        ]
        if psnr_vals:
            parts.append(f"平均 PSNR: {sum(psnr_vals)/len(psnr_vals):.2f} dB")
        if ssim_vals:
            parts.append(f"平均 SSIM: {sum(ssim_vals)/len(ssim_vals):.4f}")
        self.summary_label.setText("  |  ".join(parts))

    def _show_about(self):
        QMessageBox.about(
            self,
            "このアプリについて",
            "LPIPS Holographic Evaluation System\n\n"
            "CGH をSLM に投影し、カメラで撮像した再生像と\n"
            "ターゲット画像の LPIPS を計測するアプリケーションです。\n\n"
            "指標:\n"
            "  LPIPS : 知覚的類似度 (低いほど類似)\n"
            "  PSNR  : 信号雑音比 (高いほど類似)\n"
            "  SSIM  : 構造的類似度 (高いほど類似)",
        )

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(3000)
        if self._slm_window:
            self._slm_window.close()
        event.accept()

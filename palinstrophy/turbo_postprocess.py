"""
Standalone PySide6 post-processing viewer for palinstrophy PGM dump folders.

Reads .pgm files (P5 binary grayscale) saved by the main turbulence app
and displays them with QLabel + indexed color tables (same as the simulator).

Usage:
    uv run python -m palinstrophy.turbo_postprocess
"""

import math
import sys
import os
import colorsys
import numpy as np
from PySide6.QtCore import Qt, QStandardPaths, QTimer, QSize, Signal
from PySide6.QtGui import QImage, QPixmap, qRgb
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QPushButton,
    QFileDialog,
    QStatusBar,
    QLineEdit,
    QStyle,
    QSizePolicy,
    QSlider,
    QDialog,
    QGridLayout,
    QFrame,
)


# ======================================================================
# Color LUTs  (identical to turbo_main.py)
# ======================================================================

def _make_lut_from_stops(stops, size: int = 256) -> np.ndarray:
    stops = sorted(stops, key=lambda s: s[0])
    lut = np.zeros((size, 3), dtype=np.uint8)
    positions = [int(round(p * (size - 1))) for p, _ in stops]
    colors = [np.array(c, dtype=np.float32) for _, c in stops]
    for i in range(len(stops) - 1):
        x0 = positions[i]
        x1 = positions[i + 1]
        c0 = colors[i]
        c1 = colors[i + 1]
        if x1 <= x0:
            lut[x0] = c0.astype(np.uint8)
            continue
        length = x1 - x0
        for j in range(length):
            t = j / float(length)
            c = (1.0 - t) * c0 + t * c1
            lut[x0 + j] = c.astype(np.uint8)
    lut[positions[-1]] = colors[-1].astype(np.uint8)
    return lut


def _make_gray_lut() -> np.ndarray:
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        lut[i] = (i, i, i)
    return lut


def _make_fire_lut() -> np.ndarray:
    lut = np.zeros((256, 3), dtype=np.uint8)
    for x in range(256):
        h_deg = 85.0 * (x / 255.0)
        h = h_deg / 360.0
        s = 1.0
        l = min(1.0, x / 128.0)
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        lut[x] = (int(r * 255), int(g * 255), int(b * 255))
    return lut


def _make_doom_fire_lut() -> np.ndarray:
    key_colors = np.array([
        [0, 0, 0], [7, 7, 7], [31, 7, 7], [47, 15, 7],
        [71, 15, 7], [87, 23, 7], [103, 31, 7], [119, 31, 7],
        [143, 39, 7], [159, 47, 7], [175, 63, 7], [191, 71, 7],
        [199, 71, 7], [223, 79, 7], [223, 87, 7], [223, 87, 7],
        [215, 95, 7], [215, 95, 7], [215, 103, 15], [207, 111, 15],
        [207, 119, 15], [207, 127, 15], [207, 135, 23], [199, 135, 23],
        [199, 143, 23], [199, 151, 31], [191, 159, 31], [191, 159, 31],
        [191, 167, 39], [191, 167, 39], [191, 175, 47], [183, 175, 47],
        [183, 183, 47], [183, 183, 55], [207, 207, 111], [223, 223, 159],
        [239, 239, 199], [255, 255, 255],
    ], dtype=np.uint8)
    stops = []
    n_keys = key_colors.shape[0]
    for i in range(n_keys):
        pos = i / (n_keys - 1)
        stops.append((pos, key_colors[i].tolist()))
    return _make_lut_from_stops(stops)


def _make_viridis_lut() -> np.ndarray:
    stops = [
        (0.0, (68, 1, 84)), (0.25, (59, 82, 139)),
        (0.50, (33, 145, 140)), (0.75, (94, 201, 98)), (1.0, (253, 231, 37)),
    ]
    return _make_lut_from_stops(stops)


def _make_inferno_lut() -> np.ndarray:
    stops = [
        (0.0, (0, 0, 4)), (0.25, (87, 16, 110)),
        (0.50, (188, 55, 84)), (0.75, (249, 142, 9)), (1.0, (252, 255, 164)),
    ]
    return _make_lut_from_stops(stops)


def _make_ocean_lut() -> np.ndarray:
    stops = [
        (0.0, (0, 0, 0)), (0.25, (0, 0, 100)),
        (0.50, (0, 100, 180)), (0.75, (0, 200, 200)), (1.0, (255, 255, 255)),
    ]
    return _make_lut_from_stops(stops)


def _make_cividis_lut() -> np.ndarray:
    stops = [
        (0.0, (0, 32, 77)), (0.25, (61, 78, 112)),
        (0.50, (125, 123, 115)), (0.75, (194, 172, 88)), (1.0, (253, 232, 37)),
    ]
    return _make_lut_from_stops(stops)


def _make_jet_lut() -> np.ndarray:
    stops = [
        (0.0, (0, 0, 131)), (0.125, (0, 0, 255)), (0.375, (0, 255, 255)),
        (0.625, (255, 255, 0)), (0.875, (255, 0, 0)), (1.0, (128, 0, 0)),
    ]
    return _make_lut_from_stops(stops)


def _make_coolwarm_lut() -> np.ndarray:
    stops = [
        (0.0, (59, 76, 192)), (0.25, (141, 176, 254)),
        (0.50, (221, 221, 221)), (0.75, (245, 160, 105)), (1.0, (180, 4, 38)),
    ]
    return _make_lut_from_stops(stops)


def _make_rdbu_lut() -> np.ndarray:
    stops = [
        (0.0, (5, 48, 97)), (0.25, (67, 147, 195)),
        (0.50, (247, 247, 247)), (0.75, (214, 96, 77)), (1.0, (103, 0, 31)),
    ]
    return _make_lut_from_stops(stops)


def _make_plasma_lut() -> np.ndarray:
    stops = [
        (0.0, (13, 8, 135)), (0.25, (126, 3, 168)),
        (0.50, (204, 71, 120)), (0.75, (248, 149, 64)), (1.0, (240, 249, 33)),
    ]
    return _make_lut_from_stops(stops)


def _make_magma_lut() -> np.ndarray:
    stops = [
        (0.0, (0, 0, 4)), (0.25, (81, 18, 124)),
        (0.50, (183, 55, 121)), (0.75, (254, 159, 109)), (1.0, (252, 253, 191)),
    ]
    return _make_lut_from_stops(stops)


def _make_turbo_lut() -> np.ndarray:
    stops = [
        (0.0, (48, 18, 59)), (0.25, (31, 120, 180)),
        (0.50, (78, 181, 75)), (0.75, (241, 208, 29)), (1.0, (133, 32, 26)),
    ]
    return _make_lut_from_stops(stops)


GRAY_LUT = _make_gray_lut()
INFERNO_LUT = _make_inferno_lut()
OCEAN_LUT = _make_ocean_lut()
VIRIDIS_LUT = _make_viridis_lut()
PLASMA_LUT = _make_plasma_lut()
MAGMA_LUT = _make_magma_lut()
TURBO_LUT = _make_turbo_lut()
FIRE_LUT = _make_fire_lut()
DOOM_FIRE_LUT = _make_doom_fire_lut()
CIVIDIS_LUT = _make_cividis_lut()
JET_LUT = _make_jet_lut()
COOLWARM_LUT = _make_coolwarm_lut()
RDBU_LUT = _make_rdbu_lut()

COLOR_MAPS = {
    "Gray": GRAY_LUT,
    "Inferno": INFERNO_LUT,
    "Ocean": OCEAN_LUT,
    "Viridis": VIRIDIS_LUT,
    "Plasma": PLASMA_LUT,
    "Magma": MAGMA_LUT,
    "Turbo": TURBO_LUT,
    "Fire": FIRE_LUT,
    "Doom": DOOM_FIRE_LUT,
    "Cividis": CIVIDIS_LUT,
    "Jet": JET_LUT,
    "Coolwarm": COOLWARM_LUT,
    "RdBu": RDBU_LUT,
}

DEFAULT_CMAP_NAME = "Inferno"
CUSTOM_CMAP_NAME = "Custom"

QT_COLOR_TABLES = {
    name: [qRgb(int(rgb[0]), int(rgb[1]), int(rgb[2])) for rgb in lut]
    for name, lut in COLOR_MAPS.items()
}
QT_GRAY_TABLE = [qRgb(i, i, i) for i in range(256)]

# ======================================================================
# Custom Colors Dialog
# ======================================================================

class CustomColorsDialog(QDialog):
    """Non-modal dialog with sliders to edit a custom GLUT in real time."""

    lut_changed = Signal(list)  # emits a 256-entry Qt color table

    NUM_STOPS = 5
    FIXED_POSITIONS = [0.0, 0.25, 0.50, 0.75, 1.0]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Custom Colors")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        # Default stops (Inferno-ish)
        self._stop_colors: list[list[int]] = [
            [0, 0, 4],
            [87, 16, 110],
            [187, 55, 84],
            [249, 142, 9],
            [252, 255, 164],
        ]

        self._sliders: list[dict[str, QSlider]] = []
        self._previews: list[QFrame] = []
        self._value_labels: list[QLabel] = []

        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        grid = QGridLayout()
        grid.setSpacing(4)
        headers = ["", "R", "G", "B", ""]
        for col, hdr in enumerate(headers):
            lbl = QLabel(hdr)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(lbl, 0, col)

        for i in range(self.NUM_STOPS):
            row = i + 1
            # Color preview swatch
            preview = QFrame()
            preview.setFixedSize(28, 28)
            preview.setFrameShape(QFrame.Shape.Box)
            self._previews.append(preview)
            grid.addWidget(preview, row, 0)

            sliders = {}
            r, g, b = self._stop_colors[i]
            for ci, (ch, val) in enumerate([("r", r), ("g", g), ("b", b)]):
                sl = QSlider(Qt.Orientation.Horizontal)
                sl.setRange(0, 255)
                sl.setValue(val)
                sl.setFixedWidth(160)
                sl.valueChanged.connect(self._on_slider_changed)
                sliders[ch] = sl
                grid.addWidget(sl, row, ci + 1)

            self._sliders.append(sliders)

            # Numeric value label
            val_lbl = QLabel()
            val_lbl.setFixedWidth(90)
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            val_lbl.setStyleSheet("font-size: 10px; color: #888;")
            self._value_labels.append(val_lbl)
            grid.addWidget(val_lbl, row, 4)

        layout.addLayout(grid)

        # Reset button
        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self._on_reset)
        layout.addWidget(reset_btn)

        self._update_previews()
        self.setFixedWidth(640)

    # ----------------------------------------------------------------
    def _on_slider_changed(self) -> None:
        for i in range(self.NUM_STOPS):
            sl = self._sliders[i]
            self._stop_colors[i] = [sl["r"].value(), sl["g"].value(), sl["b"].value()]
        self._update_previews()
        self._emit_lut()

    def _update_previews(self) -> None:
        for i, color in enumerate(self._stop_colors):
            r, g, b = color
            self._previews[i].setStyleSheet(
                f"background-color: rgb({r},{g},{b}); border: 1px solid #555;"
            )
            self._value_labels[i].setText(f"{r}, {g}, {b}")

    def _emit_lut(self) -> None:
        stops = [
            (self.FIXED_POSITIONS[i], tuple(self._stop_colors[i]))
            for i in range(self.NUM_STOPS)
        ]
        lut = _make_lut_from_stops(stops)
        table = [qRgb(int(lut[j][0]), int(lut[j][1]), int(lut[j][2])) for j in range(256)]
        self.lut_changed.emit(table)

    def _on_reset(self) -> None:
        defaults = [
            [0, 0, 4], [87, 16, 110], [187, 55, 84], [249, 142, 9], [252, 255, 164],
        ]
        self.set_stops_from_colors(defaults)

    def set_stops_from_lut(self, lut: np.ndarray) -> None:
        """Sample *lut* (256×3 uint8) at the five fixed positions and update sliders."""
        indices = [int(round(p * 255)) for p in self.FIXED_POSITIONS]
        colors = [list(int(c) for c in lut[idx]) for idx in indices]
        self.set_stops_from_colors(colors)

    def set_stops_from_colors(self, colors: list[list[int]]) -> None:
        """Set all five stops from a list of [R, G, B] triples and update sliders."""
        # Block slider signals so we don't fire per-channel updates
        for sl_dict in self._sliders:
            for sl in sl_dict.values():
                sl.blockSignals(True)
        for i in range(self.NUM_STOPS):
            self._stop_colors[i] = list(colors[i])
            self._sliders[i]["r"].setValue(colors[i][0])
            self._sliders[i]["g"].setValue(colors[i][1])
            self._sliders[i]["b"].setValue(colors[i][2])
        for sl_dict in self._sliders:
            for sl in sl_dict.values():
                sl.blockSignals(False)
        self._update_previews()
        self._emit_lut()


# ======================================================================
# PGM variables  –  combo label → filename
# ======================================================================
VARIABLES = {
    "U": "u_velocity.pgm",
    "V": "v_velocity.pgm",
    "K": "kinetic.pgm",
    "Ω": "omega.pgm",
}


def read_pgm(filename: str) -> np.ndarray:
    with open(filename, "rb") as f:
        magic = f.readline().strip()
        assert magic == b"P5", f"Not a P5 PGM file: {magic}"
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        w, h = map(int, line.split())
        maxval = int(f.readline().strip())
        data = f.read(w * h)
    return np.frombuffer(data, dtype=np.uint8).reshape((h, w))


# ======================================================================
# Main window
# ======================================================================

class PostProcessWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Palinstrophy Post-Process Viewer")
        self.current_cmap_name = DEFAULT_CMAP_NAME
        self.folder_path: str = ""
        self._pgm_data: dict[str, np.ndarray] = {}

        # --- central image label ---
        self.image_label = QLabel()
        self.image_label.setContentsMargins(0, 0, 0, 0)
        self.image_label.setSizePolicy(
            self.image_label.sizePolicy().horizontalPolicy(),
            self.image_label.sizePolicy().verticalPolicy()
        )
        self.image_label.setMinimumSize(1, 1)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        style = QApplication.style()

        # Folder button
        self.folder_button = QPushButton()
        self.folder_button.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        self.folder_button.setToolTip("Open PGM folder")
        self.folder_button.setFixedSize(28, 28)
        self.folder_button.setIconSize(QSize(14, 14))

        # Variable selector
        self.variable_combo = QComboBox()
        self.variable_combo.setToolTip("V: Variable")
        self.variable_combo.addItems(list(VARIABLES.keys()))
        self.variable_combo.setCurrentText("Ω")

        # Colormap selector
        self.cmap_combo = QComboBox()
        self.cmap_combo.setToolTip("C: Colormap")
        self.cmap_combo.addItems(list(COLOR_MAPS.keys()))
        idx = self.cmap_combo.findText(DEFAULT_CMAP_NAME)
        if idx >= 0:
            self.cmap_combo.setCurrentIndex(idx)

        # Save button
        self.save_button = QPushButton()
        self.save_button.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton))
        self.save_button.setToolTip("Save current frame")
        self.save_button.setFixedSize(28, 28)
        self.save_button.setIconSize(QSize(14, 14))

        # Custom Colors button
        self.custom_colors_button = QPushButton("Custom Colors")
        self.custom_colors_button.setToolTip("Open custom GLUT editor")
        self._custom_colors_dialog: CustomColorsDialog | None = None
        self._custom_qt_table: list[int] | None = None

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)

        # --- layout  (matches turbo_main.py _build_layout) ---
        central = QWidget()
        main = QVBoxLayout(central)
        main.setSpacing(3)
        main.addWidget(self.image_label)

        # Button row
        row1 = QHBoxLayout()
        row1.setContentsMargins(10, 0, 0, 0)
        row1.setAlignment(Qt.AlignmentFlag.AlignLeft)
        row1.addWidget(self.folder_button)
        row1.addWidget(self.save_button)
        row1.addSpacing(5)
        row1.addWidget(self.variable_combo)
        row1.addWidget(self.cmap_combo)
        row1.addSpacing(5)
        row1.addWidget(self.custom_colors_button)
        row1.addStretch(1)
        main.addLayout(row1)

        self.setCentralWidget(central)

        # --- connections ---
        self.save_button.clicked.connect(self.on_save_clicked)
        self.folder_button.clicked.connect(self.on_folder_clicked)
        self.variable_combo.currentTextChanged.connect(lambda _: self._refresh_image())
        self.cmap_combo.currentTextChanged.connect(self.on_cmap_changed)
        self.custom_colors_button.clicked.connect(self.on_custom_colors_clicked)

        if sys.platform == "darwin":
            from PySide6.QtWidgets import QStyleFactory
            self.variable_combo.setStyle(QStyleFactory.create("Fusion"))
            self.cmap_combo.setStyle(QStyleFactory.create("Fusion"))

        self.resize(800, 700)

    # ------------------------------------------------------------------
    def on_save_clicked(self) -> None:
        var_name = self.variable_combo.currentText()
        cmap_name = self.cmap_combo.currentText()
        default_name = f"palinstrophy_{var_name}_{cmap_name}.png"

        desktop = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.DesktopLocation
        )
        initial_path = desktop + "/" + default_name

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save frame",
            initial_path,
            "PNG images (*.png);;All files (*)",
        )

        if path:
            pm = self.image_label.pixmap()
            if pm:
                pm.save(path, "PNG")

    # ------------------------------------------------------------------
    def on_folder_clicked(self) -> None:
        desktop = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.DesktopLocation
        )

        dlg = QFileDialog(self)
        dlg.setWindowTitle("Select PGM folder")
        dlg.setFileMode(QFileDialog.FileMode.Directory)
        dlg.setOption(QFileDialog.Option.ShowDirsOnly, True)
        dlg.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dlg.setDirectory(desktop)

        if dlg.exec():
            chosen = dlg.selectedFiles()[0]
        else:
            return

        self._load_folder(chosen)

    # ------------------------------------------------------------------
    def _load_folder(self, folder: str) -> None:
        self.folder_path = folder
        self._pgm_data.clear()

        for label, fname in VARIABLES.items():
            path = os.path.join(folder, fname)
            if os.path.isfile(path):
                self._pgm_data[label] = read_pgm(path)

        if not self._pgm_data:
            self.status.showMessage(f"No PGM files found in {folder}")
            return

        # select Ω if available, otherwise first available variable
        preferred = "Ω"
        keys = list(VARIABLES.keys())
        if preferred in self._pgm_data:
            self.variable_combo.setCurrentIndex(keys.index(preferred))
        else:
            for i, label in enumerate(keys):
                if label in self._pgm_data:
                    self.variable_combo.setCurrentIndex(i)
                    break

        self.setWindowTitle(f"Post-Process — {os.path.basename(folder)}")
        self.status.showMessage(
            f"Loaded {len(self._pgm_data)} field(s) from {folder}"
        )
        self._refresh_image()

    # ------------------------------------------------------------------
    @staticmethod
    def _display_scale(N: int) -> float:
        screen_h = 1024
        ui_margin = 320
        max_h = max(128, screen_h - ui_margin)

        if N >= max_h:
            down = int(math.ceil(N / max_h))
            return float(down)

        up = int(math.floor(max_h / N))
        if up < 1:
            up = 1
        return 1.0 / float(up)

    @staticmethod
    def _upscale_downscale_u8(pix: np.ndarray) -> np.ndarray:
        N = pix.shape[0]
        scale = PostProcessWindow._display_scale(N)

        if scale == 1.0:
            return np.ascontiguousarray(pix)

        if scale < 1.0:
            up = int(round(1.0 / scale))
            return np.ascontiguousarray(np.repeat(np.repeat(pix, up, axis=0), up, axis=1))

        s = int(scale)
        return np.ascontiguousarray(pix[::s, ::s])

    # ------------------------------------------------------------------
    def _refresh_image(self) -> None:
        label = self.variable_combo.currentText()
        arr = self._pgm_data.get(label)
        if arr is None:
            return

        pixels = self._upscale_downscale_u8(np.ascontiguousarray(arr))
        h, w = pixels.shape

        qimg = QImage(
            pixels.data,
            w,
            h,
            w,
            QImage.Format.Format_Indexed8,
        )
        if self.current_cmap_name == CUSTOM_CMAP_NAME and self._custom_qt_table is not None:
            table = self._custom_qt_table
        else:
            table = QT_COLOR_TABLES.get(self.current_cmap_name, QT_GRAY_TABLE)
        qimg.setColorTable(table)
        pix = QPixmap.fromImage(qimg, Qt.ImageConversionFlag.NoFormatConversion)
        self.image_label.setPixmap(pix)

        # Resize window to fit the image
        new_w = pix.width() + 40
        new_h = pix.height() + 120
        self.setMinimumSize(0, 0)
        self.setMaximumSize(16777215, 16777215)
        self.resize(new_w, new_h)
        # Don't re-centre when the custom-colors dialog is visible
        # (the user / on_custom_colors_clicked already placed the window)
        if self._custom_colors_dialog is not None and self._custom_colors_dialog.isVisible():
            return
        screen = QApplication.primaryScreen().availableGeometry()
        g = self.geometry()
        g.moveCenter(screen.center())
        self.setGeometry(g)

    # ------------------------------------------------------------------
    def on_cmap_changed(self, name: str) -> None:
        if name in COLOR_MAPS:
            self.current_cmap_name = name
            self._refresh_image()
            # Sync the custom-colors dialog sliders to the newly chosen map
            if self._custom_colors_dialog is not None:
                self._custom_colors_dialog.set_stops_from_lut(COLOR_MAPS[name])

    # ------------------------------------------------------------------
    def on_custom_colors_clicked(self) -> None:
        if self._custom_colors_dialog is None:
            self._custom_colors_dialog = CustomColorsDialog(self)
            self._custom_colors_dialog.lut_changed.connect(self._on_custom_lut_changed)
        dlg = self._custom_colors_dialog
        # Move main window ~1/4 to the left so the dialog fits on the right
        screen = QApplication.primaryScreen().availableGeometry()
        pos = self.pos()  # frame top-left (move() also targets the frame)
        shifted_x = max(screen.left(), pos.x() - screen.width() // 4)
        self.move(shifted_x, pos.y())
        # Position dialog to the right of the (shifted) main window
        g = self.frameGeometry()
        dlg.move(g.right() + 10, g.top())
        dlg.show()
        dlg.raise_()
        # Switch to custom colormap immediately
        self._on_custom_lut_changed(None)

    def _on_custom_lut_changed(self, table) -> None:
        if table is not None:
            self._custom_qt_table = table
        self.current_cmap_name = CUSTOM_CMAP_NAME
        self._refresh_image()

    # ------------------------------------------------------------------
    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key.Key_V:
            idx = self.variable_combo.currentIndex()
            count = self.variable_combo.count()
            self.variable_combo.setCurrentIndex((idx + 1) % count)
            return
        if key == Qt.Key.Key_C:
            idx = self.cmap_combo.currentIndex()
            count = self.cmap_combo.count()
            self.cmap_combo.setCurrentIndex((idx + 1) % count)
            return
        super().keyPressEvent(event)


def main():
    app = QApplication(sys.argv)
    win = PostProcessWindow()
    win.show()
    QTimer.singleShot(0, win.on_folder_clicked)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

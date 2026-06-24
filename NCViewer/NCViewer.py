import sys
import os
import numpy as np

try:
    import netCDF4 as nc
    NETCDF4_AVAILABLE = True
except ImportError:
    NETCDF4_AVAILABLE = False

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTreeWidget, QTreeWidgetItem, QLabel, QComboBox,
    QPushButton, QStatusBar, QFileDialog, QGroupBox,
    QGridLayout, QSizePolicy, QToolBar, QAction,
    QCheckBox, QMessageBox, QTextEdit, QTabWidget
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QColor

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# ── Spatial dimension name heuristics ────────────────────────────────────────
SPATIAL_DIM_KEYWORDS = {
    'lat', 'latitude', 'y', 'rlat', 'south_north', 'nj',
    'lon', 'longitude', 'x', 'rlon', 'west_east', 'ni',
    'row', 'col', 'column', 'nx', 'ny',
}

COLORMAPS = [
    'viridis', 'plasma', 'inferno', 'magma', 'cividis',
    'RdBu_r', 'coolwarm', 'bwr', 'seismic',
    'rainbow', 'jet', 'turbo',
    'Blues', 'Greens', 'Reds', 'Oranges', 'Purples',
    'YlOrRd', 'YlGnBu', 'Spectral',
    'gray', 'bone', 'hot', 'cool',
]

DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #1a1d23;
    color: #dce1e7;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
}
QMenuBar {
    background-color: #12141a;
    color: #dce1e7;
    border-bottom: 1px solid #2e3340;
}
QMenuBar::item:selected { background-color: #2a6496; }
QMenu { background-color: #1f2330; border: 1px solid #2e3340; }
QMenu::item:selected { background-color: #2a6496; }
QToolBar {
    background-color: #12141a;
    border-bottom: 1px solid #2e3340;
    spacing: 4px;
    padding: 2px 6px;
}
QToolButton {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 4px 8px;
    color: #dce1e7;
}
QToolButton:hover  { background-color: #2a3044; border-color: #3a4460; }
QToolButton:pressed { background-color: #2a6496; }
QSplitter::handle { background-color: #2e3340; width: 2px; }
QTreeWidget {
    background-color: #1f2330;
    border: 1px solid #2e3340;
    border-radius: 4px;
    alternate-background-color: #1a1d23;
    color: #dce1e7;
}
QTreeWidget::item:selected { background-color: #2a5080; color: #ffffff; }
QTreeWidget::item:hover    { background-color: #2a3044; }
QTreeWidget QHeaderView::section {
    background-color: #12141a;
    color: #9ba8bb;
    border: none;
    border-bottom: 1px solid #2e3340;
    padding: 4px 8px;
    font-weight: bold;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
QGroupBox {
    border: 1px solid #2e3340;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 8px;
    color: #9ba8bb;
    font-size: 11px;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QComboBox {
    background-color: #252936;
    border: 1px solid #3a4460;
    border-radius: 4px;
    padding: 4px 8px;
    color: #dce1e7;
    min-width: 100px;
}
QComboBox:hover { border-color: #4a88c7; }
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView {
    background-color: #252936;
    border: 1px solid #3a4460;
    selection-background-color: #2a5080;
}
QPushButton {
    background-color: #252936;
    border: 1px solid #3a4460;
    border-radius: 4px;
    padding: 5px 14px;
    color: #dce1e7;
}
QPushButton:hover    { background-color: #2a3044; border-color: #4a88c7; }
QPushButton:pressed  { background-color: #2a6496; }
QPushButton:disabled { color: #555d6e; border-color: #252936; background: #1a1d23; }
QPushButton#nav_btn {
    padding: 2px 8px;
    font-size: 14px;
    min-width: 26px;
    max-width: 26px;
    min-height: 24px;
    max-height: 24px;
}
QLabel { color: #dce1e7; }
QLabel#dim_label  { color: #9ba8bb; font-size: 11px; }
QLabel#idx_label  { color: #4a88c7; font-family: monospace; font-size: 12px; min-width: 80px; }
QLabel#val_label  { color: #4a88c7; font-family: monospace; font-size: 12px; }
QStatusBar {
    background-color: #12141a;
    color: #9ba8bb;
    border-top: 1px solid #2e3340;
    font-size: 12px;
}
QStatusBar::item { border: none; }
QTabWidget::pane { border: 1px solid #2e3340; }
QTabBar::tab {
    background: #1f2330;
    border: 1px solid #2e3340;
    padding: 6px 14px;
    color: #9ba8bb;
    margin-right: 2px;
}
QTabBar::tab:selected { background: #252936; color: #dce1e7; border-bottom: 2px solid #4a88c7; }
QScrollBar:vertical { background: #1a1d23; width: 8px; border-radius: 4px; }
QScrollBar::handle:vertical { background: #3a4460; border-radius: 4px; min-height: 20px; }
QScrollBar::handle:vertical:hover { background: #4a88c7; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QTextEdit {
    background-color: #1f2330;
    border: 1px solid #2e3340;
    border-radius: 4px;
    color: #dce1e7;
    font-family: 'Courier New', monospace;
    font-size: 12px;
}
QCheckBox { color: #dce1e7; spacing: 6px; }
QCheckBox::indicator {
    width: 14px; height: 14px;
    border: 1px solid #3a4460;
    border-radius: 3px;
    background: #252936;
}
QCheckBox::indicator:checked { background: #2a6496; border-color: #4a88c7; }
"""


def _is_spatial(dim_name: str) -> bool:
    """Return True if dim_name looks like a spatial (lat/lon/x/y) dimension."""
    return dim_name.lower() in SPATIAL_DIM_KEYWORDS


# ── Compact arrow navigator (only for non-spatial dims) ───────────────────────

class DimArrows(QWidget):
    """◀  idx / total  ▶  navigator for a single non-spatial dimension."""
    valueChanged = pyqtSignal(int)

    def __init__(self, dim_name: str, size: int, parent=None):
        super().__init__(parent)
        self.dim_name = dim_name
        self.size = size
        self._index = 0

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 1, 0, 1)
        row.setSpacing(4)

        lbl = QLabel(dim_name)
        lbl.setObjectName("dim_label")
        lbl.setFixedWidth(68)
        lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row.addWidget(lbl)

        self.btn_prev = QPushButton("◀")
        self.btn_prev.setObjectName("nav_btn")
        self.btn_prev.setToolTip("Previous step  (←)")
        self.btn_prev.clicked.connect(self._prev)
        row.addWidget(self.btn_prev)

        self.idx_lbl = QLabel(self._fmt())
        self.idx_lbl.setObjectName("idx_label")
        self.idx_lbl.setAlignment(Qt.AlignCenter)
        row.addWidget(self.idx_lbl)

        self.btn_next = QPushButton("▶")
        self.btn_next.setObjectName("nav_btn")
        self.btn_next.setToolTip("Next step  (→)")
        self.btn_next.clicked.connect(self._next)
        row.addWidget(self.btn_next)

        row.addStretch()
        self._refresh()

    def _fmt(self):
        return f"{self._index} / {self.size - 1}"

    def _prev(self):
        if self._index > 0:
            self._index -= 1
            self._emit()

    def _next(self):
        if self._index < self.size - 1:
            self._index += 1
            self._emit()

    def _emit(self):
        self.idx_lbl.setText(self._fmt())
        self._refresh()
        self.valueChanged.emit(self._index)

    def _refresh(self):
        self.btn_prev.setEnabled(self._index > 0)
        self.btn_next.setEnabled(self._index < self.size - 1)

    def value(self):
        return self._index

    def setValue(self, v: int):
        self._index = max(0, min(v, self.size - 1))
        self.idx_lbl.setText(self._fmt())
        self._refresh()


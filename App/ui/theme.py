"""High-contrast graphite/blue industrial console theme."""

from __future__ import annotations

PALETTE = {
    "background": "#0D151D",
    "topbar": "#101F2B",
    "sidebar": "#132633",
    "panel": "#1B2A36",
    "panel_alt": "#223542",
    "border": "#344B5A",
    "cyan": "#20C7C2",
    "blue": "#4D9DE0",
    "orange": "#F2A93B",
    "yellow": "#F0B429",
    "red": "#E05252",
    "text": "#E8F0F5",
    "muted": "#9EB2C1",
}


def stylesheet(font_family: str = "Microsoft YaHei UI") -> str:
    p = PALETTE
    return f"""
    QMainWindow, QWidget {{ background:{p['background']}; color:{p['text']}; font-family:'{font_family}'; font-size:13px; }}
    QFrame#topbar {{ background:{p['topbar']}; border-bottom:1px solid {p['border']}; }}
    QFrame#sidebar {{ background:{p['sidebar']}; border-right:1px solid {p['border']}; }}
    QListWidget {{ background:{p['sidebar']}; color:{p['muted']}; border:0; padding:14px 10px; outline:0; }}
    QListWidget::item {{ padding:13px 14px; margin:3px 0; border-radius:5px; }}
    QListWidget::item:selected {{ background:#1D4655; color:{p['cyan']}; border-left:3px solid {p['cyan']}; font-weight:700; }}
    QLabel {{ color:{p['text']}; background:transparent; }}
    QLabel#pageTitle {{ font-size:24px; font-weight:800; color:{p['text']}; }}
    QLabel#subtitle, QLabel#muted {{ color:{p['muted']}; }}
    QLabel#sectionTitle {{ color:{p['cyan']}; font-size:15px; font-weight:800; }}
    QFrame#panel, QFrame#metricCard, QFrame#chartPanel, QFrame#decisionCard, QFrame#statusCard {{ background:{p['panel']}; border:1px solid {p['border']}; border-radius:7px; }}
    QLabel#metricValue {{ color:{p['cyan']}; font-size:25px; font-weight:800; }}
    QLabel#metricCaption, QLabel#key, QLabel#caption {{ color:{p['muted']}; }}
    QLabel#value {{ color:{p['text']}; font-weight:700; }}
    QLabel#notice {{ background:#16333D; color:#BCEDEA; border:1px solid #27616A; padding:10px; border-radius:5px; }}
    QLabel#warning {{ background:#3A2C1B; color:#FFD999; border:1px solid #9A6B28; padding:10px; border-radius:5px; }}
    QPushButton {{ background:#1E5962; color:{p['text']}; border:1px solid #2A7A80; padding:8px 13px; border-radius:4px; font-weight:700; }}
    QPushButton:hover {{ background:#24737A; }}
    QPushButton:pressed {{ background:#17474E; }}
    QPushButton:disabled {{ background:#34434D; color:#71818C; }}
    QComboBox, QSpinBox, QLineEdit {{ background:{p['panel_alt']}; color:{p['text']}; border:1px solid {p['border']}; padding:6px 8px; border-radius:4px; }}
    QSlider::groove:horizontal {{ height:6px; background:{p['border']}; border-radius:3px; }}
    QSlider::sub-page:horizontal {{ background:{p['cyan']}; border-radius:3px; }}
    QSlider::handle:horizontal {{ width:16px; margin:-5px 0; background:{p['cyan']}; border-radius:8px; }}
    QTableWidget {{ background:{p['panel_alt']}; color:{p['text']}; gridline-color:{p['border']}; border:1px solid {p['border']}; }}
    QHeaderView::section {{ background:{p['sidebar']}; color:{p['muted']}; padding:6px; border:0; }}
    QScrollArea {{ background:{p['background']}; border:0; }}
    QStatusBar {{ background:{p['topbar']}; color:{p['muted']}; }}
    """

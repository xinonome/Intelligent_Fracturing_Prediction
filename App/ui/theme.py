"""Theme definitions for the acceptance APP.

The original dark industrial theme remains available as a fallback. The light
theme is intended for demonstrations and screenshots: blue accents, white
cards, and a pale blue canvas are used consistently by Qt widgets and custom
painted charts.
"""

from __future__ import annotations

THEME_PALETTES = {
    "dark": {
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
    "chart_bg": "#1B2A36",
    "chart_grid": "#344B5A",
    "chart_axis": "#9EB2C1",
    "button": "#1E5962",
    "button_border": "#2A7A80",
    "button_hover": "#24737A",
    "button_pressed": "#17474E",
    "notice_bg": "#16333D",
    "notice_text": "#BCEDEA",
    "notice_border": "#27616A",
    "warning_bg": "#3A2C1B",
    "warning_text": "#FFD999",
    "warning_border": "#9A6B28",
    },
    "light": {
    "background": "#F3F7FB",
    "topbar": "#FFFFFF",
    "sidebar": "#EAF2FA",
    "panel": "#FFFFFF",
    "panel_alt": "#F7FAFD",
    "border": "#C9D9E8",
    "cyan": "#1769AA",
    "blue": "#2878C8",
    "orange": "#D78900",
    "yellow": "#B97800",
    "red": "#D24B4B",
    "text": "#17324D",
    "muted": "#607C96",
    "chart_bg": "#FFFFFF",
    "chart_grid": "#D8E4EF",
    "chart_axis": "#607C96",
    "button": "#1769AA",
    "button_border": "#0F5C97",
    "button_hover": "#0F5C97",
    "button_pressed": "#0B4778",
    "notice_bg": "#FFFFFF",
    "notice_text": "#1769AA",
    "notice_border": "#C9D9E8",
    "warning_bg": "#FFFFFF",
    "warning_text": "#1769AA",
    "warning_border": "#C9D9E8",
    },
}

PALETTE = dict(THEME_PALETTES["dark"])


def set_theme(name: str = "dark") -> str:
    """Activate a named theme and return the normalized theme name."""

    normalized = str(name or "dark").strip().lower()
    if normalized not in THEME_PALETTES:
        normalized = "dark"
    PALETTE.clear()
    PALETTE.update(THEME_PALETTES[normalized])
    return normalized


def stylesheet(font_family: str = "Microsoft YaHei UI", theme: str | None = None) -> str:
    if theme is not None:
        set_theme(theme)
    p = PALETTE
    return f"""
    QMainWindow, QWidget {{ background:{p['background']}; color:{p['text']}; font-family:'{font_family}'; font-size:13px; }}
    QFrame#topbar {{ background:{p['topbar']}; border-bottom:1px solid {p['border']}; }}
    QLabel#appHeaderTitle {{ font-size:20px; font-weight:800; color:{p['cyan']}; }}
    QLabel#headerLabel {{ color:{p['muted']}; padding:5px 10px; }}
    QLabel#sourceContext {{ background:{p['notice_bg']}; color:{p['notice_text']}; border:1px solid {p['notice_border']}; padding:6px 10px; border-radius:11px; }}
    QToolButton#sourcesButton {{ background:{p['panel_alt']}; color:{p['text']}; border:1px solid {p['border']}; padding:7px 12px; border-radius:4px; font-weight:700; }}
    QToolButton#sourcesButton:hover {{ border-color:{p['cyan']}; }}
    QToolButton#themeToggle {{ background:{p['panel_alt']}; color:{p['muted']}; border:1px solid {p['border']}; padding:2px; border-radius:16px; min-width:34px; max-width:34px; min-height:30px; max-height:30px; font-size:17px; font-weight:700; }}
    QToolButton#themeToggle:hover {{ background:{p['notice_bg']}; color:{p['text']}; }}
    QToolButton#themeToggle:checked {{ background:{p['notice_bg']}; color:{p['cyan']}; border-color:{p['notice_border']}; }}
    QToolButton#workspaceModeButton {{ background:{p['panel_alt']}; color:{p['muted']}; border:1px solid {p['border']}; padding:7px 14px; min-width:72px; font-weight:700; }}
    QToolButton#workspaceModeButton:hover {{ color:{p['text']}; border-color:{p['cyan']}; }}
    QToolButton#workspaceModeButton:checked {{ background:{p['button']}; color:#FFFFFF; border-color:{p['button_border']}; }}
    QFrame#sidebar {{ background:{p['sidebar']}; border-right:1px solid {p['border']}; }}
    QWidget#sidebarShell {{ background:{p['sidebar']}; border-right:1px solid {p['border']}; }}
    QListWidget#sidebar {{ background:{p['sidebar']}; color:{p['muted']}; border-right:1px solid {p['border']}; }}
    QListWidget {{ background:{p['sidebar']}; color:{p['muted']}; border:0; padding:14px 10px; outline:0; }}
    QListWidget::item {{ padding:13px 14px; margin:3px 0; border-radius:5px; }}
    QListWidget::item:hover {{ background:{p['panel_alt']}; color:{p['text']}; }}
    QListWidget::item:selected {{ background:{p['notice_bg']}; color:{p['cyan']}; border-left:3px solid {p['cyan']}; font-weight:700; }}
    QLabel {{ color:{p['text']}; background:transparent; }}
    QLabel#pageTitle {{ font-size:24px; font-weight:800; color:{p['text']}; }}
    QLabel#subtitle, QLabel#muted {{ color:{p['muted']}; }}
    QLabel#sectionTitle {{ color:{p['cyan']}; font-size:15px; font-weight:800; }}
    QFrame#panel, QFrame#metricCard, QFrame#chartPanel, QFrame#decisionCard, QFrame#statusCard, QFrame#noDasGifPanel {{ background:{p['panel']}; border:1px solid {p['border']}; border-radius:7px; }}
    QLabel#noDasGifImage, QGraphicsView#noDasGifImage {{ background:{p['chart_bg']}; color:{p['muted']}; border:1px solid {p['border']}; border-radius:5px; }}
    QFrame#metricCard:hover, QFrame#chartPanel:hover, QFrame#decisionCard:hover {{ border:1px solid {p['cyan']}; }}
    QLabel#metricValue {{ color:{p['cyan']}; font-size:25px; font-weight:800; }}
    QLabel#metricCaption, QLabel#key, QLabel#caption {{ color:{p['muted']}; }}
    QLabel#value {{ color:{p['text']}; font-weight:700; }}
    QLabel#notice {{ background:{p['notice_bg']}; color:{p['notice_text']}; border:1px solid {p['notice_border']}; padding:10px; border-radius:5px; }}
    QLabel#warning {{ background:{p['warning_bg']}; color:{p['warning_text']}; border:1px solid {p['warning_border']}; padding:10px; border-radius:5px; }}
    QPushButton {{ background:{p['button']}; color:#FFFFFF; border:1px solid {p['button_border']}; padding:8px 13px; border-radius:4px; font-weight:700; }}
    QPushButton:hover {{ background:{p['button_hover']}; }}
    QPushButton:pressed {{ background:{p['button_pressed']}; }}
    QPushButton:disabled {{ background:{p['panel_alt']}; color:{p['muted']}; border-color:{p['border']}; }}
    QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{ background:{p['panel_alt']}; color:{p['text']}; border:1px solid {p['border']}; padding:6px 8px; border-radius:4px; }}
    QComboBox QAbstractItemView {{ background:{p['panel']}; color:{p['text']}; selection-background-color:{p['notice_bg']}; selection-color:{p['text']}; border:1px solid {p['border']}; }}
    QSlider::groove:horizontal {{ height:6px; background:{p['border']}; border-radius:3px; }}
    QSlider::sub-page:horizontal {{ background:{p['cyan']}; border-radius:3px; }}
    QSlider::handle:horizontal {{ width:16px; margin:-5px 0; background:{p['cyan']}; border-radius:8px; }}
    QTableWidget {{ background:{p['panel']}; alternate-background-color:{p['panel_alt']}; color:{p['text']}; gridline-color:{p['border']}; border:1px solid {p['border']}; }}
    QTableWidget::item:selected {{ background:{p['button']}; color:#FFFFFF; }}
    QPlainTextEdit#taskConsole {{ background:{p['chart_bg']}; color:{p['text']}; border:1px solid {p['border']}; border-radius:4px; padding:6px; font-family:Consolas,'Microsoft YaHei UI',monospace; font-size:12px; }}
    QHeaderView::section {{ background:{p['sidebar']}; color:{p['text']}; padding:6px; border:0; border-bottom:1px solid {p['border']}; }}
    QScrollArea {{ background:{p['background']}; border:0; }}
    QTabWidget::pane {{ background:{p['background']}; border:1px solid {p['border']}; border-radius:5px; top:-1px; }}
    QTabBar::tab {{ background:{p['panel_alt']}; color:{p['muted']}; border:1px solid {p['border']}; padding:7px 18px; margin-right:3px; border-top-left-radius:4px; border-top-right-radius:4px; }}
    QTabBar::tab:selected {{ background:{p['panel']}; color:{p['cyan']}; border-bottom-color:{p['panel']}; font-weight:700; }}
    QTabBar::tab:hover {{ color:{p['text']}; }}
    QStatusBar {{ background:{p['topbar']}; color:{p['muted']}; }}
    QToolTip {{ background:{p['panel']}; color:{p['text']}; border:1px solid {p['border']}; padding:5px; }}
    """

from __future__ import annotations


def create_parameter_panel(title: str, fields: list[tuple[str, str]]):
    from PySide6.QtWidgets import QFormLayout, QLabel, QFrame

    panel = QFrame()
    panel.setObjectName("panel")
    layout = QFormLayout(panel)
    layout.setContentsMargins(12, 10, 12, 10)
    heading = QLabel(title)
    heading.setObjectName("sectionTitle")
    layout.addRow(heading, QLabel(""))
    labels = {}
    for key, caption in fields:
        key_label = QLabel(caption)
        key_label.setObjectName("key")
        value = QLabel("--")
        value.setObjectName("value")
        value.setWordWrap(True)
        value.setTextInteractionFlags(value.textInteractionFlags())
        layout.addRow(key_label, value)
        labels[key] = value
    panel._value_labels = labels
    return panel


def update_parameter_panel(panel, values: dict, formats: dict[str, str] | None = None):
    formats = formats or {}
    for key, label in getattr(panel, "_value_labels", {}).items():
        value = values.get(key)
        if value is None:
            label.setText("缺失 · 未接入")
        elif key in formats:
            label.setText(formats[key].format(value))
        elif isinstance(value, float):
            label.setText(f"{value:.3f}")
        else:
            label.setText(str(value))

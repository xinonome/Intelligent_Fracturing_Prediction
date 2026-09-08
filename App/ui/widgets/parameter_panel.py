from __future__ import annotations


def create_parameter_panel(
    title: str,
    fields: list[tuple[str, str]] | None = None,
    *,
    columns: list[tuple[str, list[tuple[str, str]]]] | None = None,
    column_stretches: list[int] | None = None,
):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QFormLayout, QHBoxLayout, QLabel, QFrame, QVBoxLayout

    panel = QFrame()
    panel.setObjectName("panel")
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(12, 10, 12, 10)
    heading = QLabel(title)
    heading.setObjectName("sectionTitle")
    layout.addWidget(heading)
    labels = {}
    captions = {}

    if columns:
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(18)
        for index, (column_title, column_fields) in enumerate(columns):
            column = QVBoxLayout()
            column.setSpacing(4)
            column_heading = QLabel(column_title)
            column_heading.setObjectName("key")
            column.addWidget(column_heading)
            for key, caption in column_fields:
                if caption:
                    key_label = QLabel(caption)
                    key_label.setObjectName("muted")
                    column.addWidget(key_label)
                    captions[key] = key_label
                value = QLabel("")
                value.setObjectName("value")
                value.setWordWrap(True)
                value.setTextInteractionFlags(value.textInteractionFlags())
                value.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
                column.addWidget(value)
                labels[key] = value
            column.addStretch(1)
            body.addLayout(column)
            if index < len(columns) - 1:
                separator = QFrame()
                separator.setFrameShape(QFrame.VLine)
                separator.setFrameShadow(QFrame.Plain)
                separator.setObjectName("panelSeparator")
                body.addWidget(separator)
        for index, stretch in enumerate(column_stretches or [1] * len(columns)):
            body.setStretch(index * 2, stretch)
        layout.addLayout(body, 1)
    else:
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        for key, caption in fields or []:
            key_label = QLabel(caption)
            key_label.setObjectName("key")
            value = QLabel("")
            value.setObjectName("value")
            value.setWordWrap(True)
            value.setTextInteractionFlags(value.textInteractionFlags())
            form.addRow(key_label, value)
            labels[key] = value
            captions[key] = key_label
        layout.addLayout(form)
    panel._value_labels = labels
    panel._caption_labels = captions
    return panel


def update_parameter_panel(panel, values: dict, formats: dict[str, str] | None = None):
    formats = formats or {}
    for key, label in getattr(panel, "_value_labels", {}).items():
        value = values.get(key)
        if value is None:
            label.clear()
            label.setVisible(False)
            caption = getattr(panel, "_caption_labels", {}).get(key)
            if caption is not None:
                caption.setVisible(False)
        elif key in formats:
            label.setVisible(True)
            caption = getattr(panel, "_caption_labels", {}).get(key)
            if caption is not None:
                caption.setVisible(True)
            label.setText(formats[key].format(value))
        elif isinstance(value, float):
            label.setVisible(True)
            caption = getattr(panel, "_caption_labels", {}).get(key)
            if caption is not None:
                caption.setVisible(True)
            label.setText(f"{value:.3f}")
        else:
            label.setVisible(True)
            caption = getattr(panel, "_caption_labels", {}).get(key)
            if caption is not None:
                caption.setVisible(True)
            label.setText(str(value))

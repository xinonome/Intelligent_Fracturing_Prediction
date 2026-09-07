"""Lazy two-state container used by the integrated application shell."""

from __future__ import annotations


def create_mode_center(runtime_factory, research_factory, *, initial_mode: str = "runtime"):
    """Create one business center with runtime and research implementations.

    The two implementations are built only when first shown.  This keeps the
    first paint fast and, more importantly, prevents heavy WebEngine/PyFrac
    widgets from being constructed for an inactive work state.
    """

    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QLabel, QStackedWidget

    stack = QStackedWidget()
    stack.setObjectName("workspaceModeCenter")
    factories = (runtime_factory, research_factory)
    built = {}
    last_dataset_id = ""

    for text in ("正在准备运行工作台…", "正在准备模型研发工作台…"):
        placeholder = QLabel(text)
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setObjectName("muted")
        stack.addWidget(placeholder)

    def ensure(index: int):
        if index in built:
            return built[index]
        page = factories[index]()
        old = stack.widget(index)
        built[index] = page
        stack.removeWidget(old)
        stack.insertWidget(index, page)
        old.deleteLater()
        if last_dataset_id:
            callback = getattr(page, "set_global_dataset", None)
            if callable(callback):
                callback(last_dataset_id)
        return page

    def set_workspace_mode(mode: str):
        index = 1 if str(mode).lower() == "research" else 0
        current = stack.currentWidget()
        pause = getattr(current, "pause_playback", None)
        if callable(pause):
            pause()
        page = ensure(index)
        stack.setCurrentWidget(page)

    def set_global_dataset(dataset_id: str):
        nonlocal last_dataset_id
        last_dataset_id = str(dataset_id or "")
        for page in built.values():
            callback = getattr(page, "set_global_dataset", None)
            if callable(callback):
                callback(last_dataset_id)

    def forward(name: str):
        def call(*args, **kwargs):
            for page in built.values():
                callback = getattr(page, name, None)
                if callable(callback):
                    callback(*args, **kwargs)
        return call

    def dataset_selection_locked() -> bool:
        page = stack.currentWidget()
        callback = getattr(page, "dataset_selection_locked", None)
        return bool(callback()) if callable(callback) else False

    def dataset_selection_message() -> str:
        page = stack.currentWidget()
        callback = getattr(page, "dataset_selection_message", None)
        return str(callback() or "") if callable(callback) else ""

    stack.set_workspace_mode = set_workspace_mode
    stack.set_global_dataset = set_global_dataset
    stack.pause_playback = forward("pause_playback")
    stack.prepare_dataset_change = forward("prepare_dataset_change")
    stack.refresh_dataset_visual = forward("refresh_dataset_visual")
    stack.refresh_theme = forward("refresh_theme")
    stack.refresh_catalog = forward("refresh_catalog")
    stack.dataset_selection_locked = dataset_selection_locked
    stack.dataset_selection_message = dataset_selection_message
    stack._mode_pages = built
    set_workspace_mode(initial_mode)
    return stack


def create_tabbed_center(tabs: list[tuple[str, object]]):
    """Combine related operational pages and forward application hooks."""

    from PySide6.QtWidgets import QTabWidget

    root = QTabWidget()
    root.setObjectName("businessCenterTabs")
    pages = []
    for label, page in tabs:
        root.addTab(page, label)
        pages.append(page)

    def refresh_current(_index: int = 0):
        if 0 <= _index < len(pages):
            callback = getattr(pages[_index], "refresh_dataset_visual", None)
            if callable(callback):
                callback()

    root.currentChanged.connect(refresh_current)

    def forward(name: str):
        def call(*args, **kwargs):
            for page in pages:
                callback = getattr(page, name, None)
                if callable(callback):
                    callback(*args, **kwargs)
        return call

    root.set_global_dataset = forward("set_global_dataset")
    root.pause_playback = forward("pause_playback")
    root.prepare_dataset_change = forward("prepare_dataset_change")
    root.refresh_dataset_visual = forward("refresh_dataset_visual")
    root.refresh_theme = forward("refresh_theme")
    root.refresh_catalog = forward("refresh_catalog")
    root._business_pages = pages
    return root


__all__ = ["create_mode_center", "create_tabbed_center"]

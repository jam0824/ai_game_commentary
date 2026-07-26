from __future__ import annotations

import ctypes
import time
import unicodedata
from ctypes import wintypes
from dataclasses import dataclass

from PIL import Image, ImageGrab


user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

SW_RESTORE = 9
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
VK_RETURN = 0x0D
VK_DOWN = 0x28
ENTER_SCAN_CODE = 0x1C
DOWN_SCAN_CODE = 0x50


class POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]


WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

user32.EnumWindows.argtypes = [WNDENUMPROC, wintypes.LPARAM]
user32.EnumWindows.restype = wintypes.BOOL
user32.IsWindowVisible.argtypes = [wintypes.HWND]
user32.IsWindowVisible.restype = wintypes.BOOL
user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
user32.GetWindowTextLengthW.restype = ctypes.c_int
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetWindowTextW.restype = ctypes.c_int
user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(RECT)]
user32.GetClientRect.restype = wintypes.BOOL
user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(POINT)]
user32.ClientToScreen.restype = wintypes.BOOL
user32.IsIconic.argtypes = [wintypes.HWND]
user32.IsIconic.restype = wintypes.BOOL
user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
user32.ShowWindow.restype = wintypes.BOOL
user32.SetForegroundWindow.argtypes = [wintypes.HWND]
user32.SetForegroundWindow.restype = wintypes.BOOL
user32.BringWindowToTop.argtypes = [wintypes.HWND]
user32.BringWindowToTop.restype = wintypes.BOOL
user32.GetForegroundWindow.restype = wintypes.HWND
user32.GetWindowThreadProcessId.argtypes = [
    wintypes.HWND,
    ctypes.POINTER(wintypes.DWORD),
]
user32.GetWindowThreadProcessId.restype = wintypes.DWORD
user32.AttachThreadInput.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]
user32.AttachThreadInput.restype = wintypes.BOOL
user32.PostMessageW.argtypes = [
    wintypes.HWND,
    wintypes.UINT,
    wintypes.WPARAM,
    wintypes.LPARAM,
]
user32.PostMessageW.restype = wintypes.BOOL
kernel32.GetCurrentThreadId.restype = wintypes.DWORD


@dataclass(frozen=True)
class WindowInfo:
    hwnd: int
    title: str
    width: int
    height: int


def _check(ok: int, api_name: str) -> None:
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error(), api_name)


def enable_dpi_awareness() -> None:
    """Keep Win32 coordinates aligned with physical screenshot pixels."""
    try:
        # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
        user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except (AttributeError, OSError):
        try:
            user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


def _window_title(hwnd: int) -> str:
    length = user32.GetWindowTextLengthW(hwnd)
    if length <= 0:
        return ""
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buffer, len(buffer))
    return buffer.value


def _client_bbox(hwnd: int) -> tuple[int, int, int, int]:
    rect = RECT()
    _check(user32.GetClientRect(hwnd, ctypes.byref(rect)), "GetClientRect")
    top_left = POINT(rect.left, rect.top)
    bottom_right = POINT(rect.right, rect.bottom)
    _check(user32.ClientToScreen(hwnd, ctypes.byref(top_left)), "ClientToScreen")
    _check(user32.ClientToScreen(hwnd, ctypes.byref(bottom_right)), "ClientToScreen")
    return top_left.x, top_left.y, bottom_right.x, bottom_right.y


def list_windows() -> list[WindowInfo]:
    windows: list[WindowInfo] = []

    @WNDENUMPROC
    def callback(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        title = _window_title(hwnd)
        if not title:
            return True
        try:
            left, top, right, bottom = _client_bbox(hwnd)
        except OSError:
            return True
        width, height = right - left, bottom - top
        if width > 0 and height > 0:
            windows.append(WindowInfo(int(hwnd), title, width, height))
        return True

    _check(user32.EnumWindows(callback, 0), "EnumWindows")
    return windows


def normalize_title(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    for symbol in ("×", "✕", "✖", "＊", "*"):
        normalized = normalized.replace(symbol, "x")
    return "".join(normalized.split())


def find_window(title_query: str) -> WindowInfo:
    query = normalize_title(title_query)
    candidates = [
        window for window in list_windows() if query in normalize_title(window.title)
    ]
    if not candidates:
        raise LookupError(f"タイトルに {title_query!r} を含むウィンドウが見つかりません。")
    exact = [window for window in candidates if normalize_title(window.title) == query]
    if len(exact) == 1:
        return exact[0]
    if len(candidates) == 1:
        return candidates[0]
    titles = "\n".join(f"  - {window.title}" for window in candidates)
    raise LookupError(
        "候補が複数あります。--title をより具体的に指定してください:\n" + titles
    )


def activate_window(hwnd: int, wait_seconds: float = 0.35) -> None:
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, SW_RESTORE)

    foreground = user32.GetForegroundWindow()
    current_thread = kernel32.GetCurrentThreadId()
    foreground_thread = (
        user32.GetWindowThreadProcessId(foreground, None) if foreground else 0
    )
    attached = False
    if foreground_thread and foreground_thread != current_thread:
        attached = bool(user32.AttachThreadInput(current_thread, foreground_thread, True))
    try:
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
    finally:
        if attached:
            user32.AttachThreadInput(current_thread, foreground_thread, False)
    time.sleep(max(wait_seconds, 0))


def capture_client(
    window: WindowInfo,
    *,
    activate: bool = True,
    wait_seconds: float = 0.35,
) -> Image.Image:
    if activate:
        activate_window(window.hwnd, wait_seconds=wait_seconds)
    bbox = _client_bbox(window.hwnd)
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise RuntimeError("ウィンドウのクライアント領域サイズが不正です。")
    return ImageGrab.grab(bbox=bbox, all_screens=True).convert("RGB")


def _post_key(
    hwnd: int,
    virtual_key: int,
    scan_code: int,
    *,
    extended: bool = False,
) -> None:
    key_down_lparam = 1 | (scan_code << 16)
    if extended:
        key_down_lparam |= 1 << 24
    key_up_lparam = key_down_lparam | (1 << 30) | (1 << 31)
    _check(
        user32.PostMessageW(hwnd, WM_KEYDOWN, virtual_key, key_down_lparam),
        "PostMessageW(WM_KEYDOWN)",
    )
    time.sleep(0.03)
    _check(
        user32.PostMessageW(hwnd, WM_KEYUP, virtual_key, key_up_lparam),
        "PostMessageW(WM_KEYUP)",
    )


def press_enter(hwnd: int, *, activate: bool = True) -> None:
    """Post a scoped Enter key press to the selected window."""
    if activate:
        activate_window(hwnd, wait_seconds=0.1)
    _post_key(hwnd, VK_RETURN, ENTER_SCAN_CODE)


def select_choice(
    hwnd: int,
    choice_index: int,
    *,
    activate: bool = True,
    key_interval: float = 0.08,
) -> None:
    """Select a zero-based menu item with Down presses followed by Enter."""
    if choice_index < 0:
        raise ValueError("choice_index must be zero or greater")
    if activate:
        activate_window(hwnd, wait_seconds=0.1)
    for _ in range(choice_index):
        _post_key(
            hwnd,
            VK_DOWN,
            DOWN_SCAN_CODE,
            extended=True,
        )
        time.sleep(max(key_interval, 0))
    _post_key(hwnd, VK_RETURN, ENTER_SCAN_CODE)

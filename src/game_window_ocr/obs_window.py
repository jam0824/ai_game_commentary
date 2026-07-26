from __future__ import annotations

import ctypes
import os
import sys
import threading
from ctypes import wintypes
from types import TracebackType


OBS_WINDOW_TITLE = "AIゲーム実況 - OBS音声キャプチャ"

_WM_CLOSE = 0x0010
_WM_DESTROY = 0x0002
_WM_SETFONT = 0x0030
_WM_APP_STOP = 0x8001
_WS_CAPTION = 0x00C00000
_WS_SYSMENU = 0x00080000
_WS_MINIMIZEBOX = 0x00020000
_WS_CHILD = 0x40000000
_WS_VISIBLE = 0x10000000
_SS_CENTER = 0x00000001
_SW_SHOW = 5
_SW_MINIMIZE = 6
_COLOR_WINDOW = 5
_DEFAULT_GUI_FONT = 17
_CW_USEDEFAULT = -2147483648


class ObsCaptureWindow:
    """Expose the commentary process as a selectable top-level OBS window."""

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled and sys.platform == "win32"
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._is_open = False
        self._hwnd: int | None = None

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def error(self) -> BaseException | None:
        return self._error

    def __enter__(self) -> ObsCaptureWindow:
        if not self.enabled:
            return self

        self._thread = threading.Thread(
            target=self._run,
            name="obs-capture-window",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait(timeout=5.0)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._stop.set()
        if self._hwnd is not None:
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            user32.PostMessageW(
                wintypes.HWND(self._hwnd),
                _WM_APP_STOP,
                0,
                0,
            )
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
        wnd_proc_type = ctypes.WINFUNCTYPE(
            ctypes.c_ssize_t,
            wintypes.HWND,
            wintypes.UINT,
            ctypes.c_size_t,
            ctypes.c_ssize_t,
        )

        class WindowClass(ctypes.Structure):
            _fields_ = [
                ("style", wintypes.UINT),
                ("lpfnWndProc", wnd_proc_type),
                ("cbClsExtra", ctypes.c_int),
                ("cbWndExtra", ctypes.c_int),
                ("hInstance", wintypes.HINSTANCE),
                ("hIcon", wintypes.HICON),
                ("hCursor", wintypes.HANDLE),
                ("hbrBackground", wintypes.HBRUSH),
                ("lpszMenuName", wintypes.LPCWSTR),
                ("lpszClassName", wintypes.LPCWSTR),
            ]

        user32.DefWindowProcW.argtypes = [
            wintypes.HWND,
            wintypes.UINT,
            ctypes.c_size_t,
            ctypes.c_ssize_t,
        ]
        user32.DefWindowProcW.restype = ctypes.c_ssize_t
        user32.CreateWindowExW.argtypes = [
            wintypes.DWORD,
            wintypes.LPCWSTR,
            wintypes.LPCWSTR,
            wintypes.DWORD,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.HWND,
            wintypes.HMENU,
            wintypes.HINSTANCE,
            wintypes.LPVOID,
        ]
        user32.CreateWindowExW.restype = wintypes.HWND
        user32.RegisterClassW.argtypes = [ctypes.POINTER(WindowClass)]
        user32.RegisterClassW.restype = wintypes.ATOM
        user32.UnregisterClassW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.HINSTANCE,
        ]
        user32.LoadCursorW.argtypes = [wintypes.HINSTANCE, wintypes.LPCWSTR]
        user32.LoadCursorW.restype = wintypes.HANDLE
        user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.DestroyWindow.argtypes = [wintypes.HWND]
        user32.PostMessageW.argtypes = [
            wintypes.HWND,
            wintypes.UINT,
            ctypes.c_size_t,
            ctypes.c_ssize_t,
        ]
        user32.SendMessageW.argtypes = [
            wintypes.HWND,
            wintypes.UINT,
            ctypes.c_size_t,
            ctypes.c_ssize_t,
        ]
        user32.GetMessageW.argtypes = [
            ctypes.POINTER(wintypes.MSG),
            wintypes.HWND,
            wintypes.UINT,
            wintypes.UINT,
        ]
        kernel32.GetModuleHandleW.restype = wintypes.HMODULE
        kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
        gdi32.GetStockObject.restype = wintypes.HANDLE
        gdi32.GetStockObject.argtypes = [ctypes.c_int]
        class_name = f"AI_GAME_COMMENTARY_OBS_{os.getpid()}"
        instance = kernel32.GetModuleHandleW(None)
        hwnd: int | None = None
        class_registered = False

        @wnd_proc_type
        def wnd_proc(
            window: wintypes.HWND,
            message: int,
            wparam: int,
            lparam: int,
        ) -> int:
            if message == _WM_CLOSE:
                user32.ShowWindow(window, _SW_MINIMIZE)
                return 0
            if message == _WM_APP_STOP:
                user32.DestroyWindow(window)
                return 0
            if message == _WM_DESTROY:
                user32.PostQuitMessage(0)
                return 0
            return user32.DefWindowProcW(window, message, wparam, lparam)

        try:
            cursor = user32.LoadCursorW(
                None,
                ctypes.cast(ctypes.c_void_p(32512), wintypes.LPCWSTR),
            )
            window_class = WindowClass(
                style=0,
                lpfnWndProc=wnd_proc,
                cbClsExtra=0,
                cbWndExtra=0,
                hInstance=instance,
                hIcon=None,
                hCursor=cursor,
                hbrBackground=wintypes.HBRUSH(_COLOR_WINDOW + 1),
                lpszMenuName=None,
                lpszClassName=class_name,
            )
            if not user32.RegisterClassW(ctypes.byref(window_class)):
                raise ctypes.WinError(ctypes.get_last_error())
            class_registered = True

            window_handle = user32.CreateWindowExW(
                0,
                class_name,
                OBS_WINDOW_TITLE,
                _WS_CAPTION | _WS_SYSMENU | _WS_MINIMIZEBOX,
                _CW_USEDEFAULT,
                _CW_USEDEFAULT,
                450,
                170,
                None,
                None,
                instance,
                None,
            )
            if not window_handle:
                raise ctypes.WinError(ctypes.get_last_error())
            hwnd = int(window_handle)
            self._hwnd = hwnd

            heading = user32.CreateWindowExW(
                0,
                "STATIC",
                "AIゲーム実況を実行しています",
                _WS_CHILD | _WS_VISIBLE | _SS_CENTER,
                10,
                22,
                425,
                28,
                window_handle,
                None,
                instance,
                None,
            )
            description = user32.CreateWindowExW(
                0,
                "STATIC",
                (
                    "OBSのアプリケーション音声キャプチャで、このウィンドウを"
                    "選択してください。\r\n"
                    "最小化しても構いません。実況終了時に自動で閉じます。"
                ),
                _WS_CHILD | _WS_VISIBLE | _SS_CENTER,
                10,
                58,
                425,
                45,
                window_handle,
                None,
                instance,
                None,
            )
            font = gdi32.GetStockObject(_DEFAULT_GUI_FONT)
            if heading:
                user32.SendMessageW(heading, _WM_SETFONT, font, True)
            if description:
                user32.SendMessageW(description, _WM_SETFONT, font, True)

            user32.ShowWindow(window_handle, _SW_SHOW)
            user32.UpdateWindow(window_handle)
            self._is_open = True
            self._ready.set()

            if self._stop.is_set():
                user32.PostMessageW(window_handle, _WM_APP_STOP, 0, 0)

            message = wintypes.MSG()
            while True:
                result = user32.GetMessageW(ctypes.byref(message), None, 0, 0)
                if result <= 0:
                    break
                user32.TranslateMessage(ctypes.byref(message))
                user32.DispatchMessageW(ctypes.byref(message))
        except BaseException as exc:
            self._error = exc
        finally:
            self._is_open = False
            self._hwnd = None
            self._ready.set()
            if hwnd is not None and user32.IsWindow(wintypes.HWND(hwnd)):
                user32.DestroyWindow(wintypes.HWND(hwnd))
            if class_registered:
                user32.UnregisterClassW(class_name, instance)

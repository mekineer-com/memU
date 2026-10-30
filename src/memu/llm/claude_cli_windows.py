"""Windows ownership for one scoped Claude invocation (import only on Windows)."""

import asyncio
import ctypes
from ctypes import wintypes


_kernel = ctypes.WinDLL("kernel32", use_last_error=True)


class _ThreadEntry(ctypes.Structure):
    _fields_ = [
        ("size", wintypes.DWORD), ("usage", wintypes.DWORD),
        ("thread_id", wintypes.DWORD), ("process_id", wintypes.DWORD),
        ("base_priority", wintypes.LONG), ("delta_priority", wintypes.LONG),
        ("flags", wintypes.DWORD),
    ]


class _Accounting(ctypes.Structure):
    _fields_ = [
        ("times", ctypes.c_longlong * 4), ("page_faults", wintypes.DWORD),
        ("total", wintypes.DWORD), ("active", wintypes.DWORD),
        ("terminated", wintypes.DWORD),
    ]


for _name, _args, _result in [
    ("CreateJobObjectW", [wintypes.LPVOID, wintypes.LPCWSTR], wintypes.HANDLE),
    ("OpenProcess", [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE),
    ("AssignProcessToJobObject", [wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
    ("TerminateJobObject", [wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
    ("QueryInformationJobObject", [wintypes.HANDLE, wintypes.INT, wintypes.LPVOID,
                                   wintypes.DWORD, wintypes.LPVOID], wintypes.BOOL),
    ("CreateToolhelp32Snapshot", [wintypes.DWORD, wintypes.DWORD], wintypes.HANDLE),
    ("Thread32First", [wintypes.HANDLE, ctypes.POINTER(_ThreadEntry)], wintypes.BOOL),
    ("Thread32Next", [wintypes.HANDLE, ctypes.POINTER(_ThreadEntry)], wintypes.BOOL),
    ("OpenThread", [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE),
    ("ResumeThread", [wintypes.HANDLE], wintypes.DWORD),
    ("CloseHandle", [wintypes.HANDLE], wintypes.BOOL),
]:
    _function = getattr(_kernel, _name)
    _function.argtypes = _args
    _function.restype = _result


def _check(value):
    if not value:
        raise ctypes.WinError(ctypes.get_last_error())
    return value


class WindowsJob:
    def __init__(self):
        # No kill-on-close: losing the server must not cancel active model work.
        self.handle = _check(_kernel.CreateJobObjectW(None, None))
        self.assigned = False

    def assign_and_resume(self, pid: int) -> None:
        process = _check(_kernel.OpenProcess(0x0101, False, pid))  # SET_QUOTA | TERMINATE
        try:
            _check(_kernel.AssignProcessToJobObject(self.handle, process))
            self.assigned = True
        finally:
            _kernel.CloseHandle(process)

        snapshot = _kernel.CreateToolhelp32Snapshot(4, 0)  # TH32CS_SNAPTHREAD
        if snapshot == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        entry = _ThreadEntry()
        entry.size = ctypes.sizeof(entry)
        threads = []
        try:
            found = _kernel.Thread32First(snapshot, ctypes.byref(entry))
            while found:
                if entry.process_id == pid:
                    threads.append(entry.thread_id)
                found = _kernel.Thread32Next(snapshot, ctypes.byref(entry))
            if ctypes.get_last_error() != 18:  # ERROR_NO_MORE_FILES
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            _kernel.CloseHandle(snapshot)
        # CREATE_SUSPENDED leaves exactly the initial thread, before user code.
        if len(threads) != 1:
            raise RuntimeError("Suspended Claude process has no unique initial thread")
        thread = _check(_kernel.OpenThread(2, False, threads[0]))  # THREAD_SUSPEND_RESUME
        try:
            if _kernel.ResumeThread(thread) != 1:
                raise RuntimeError("Claude initial thread was not suspended exactly once")
        finally:
            _kernel.CloseHandle(thread)

    def terminate(self) -> None:
        _check(_kernel.TerminateJobObject(self.handle, 1))

    async def wait_empty(self, deadline: float) -> bool:
        while True:
            accounting = _Accounting()
            _check(_kernel.QueryInformationJobObject(
                self.handle, 1, ctypes.byref(accounting), ctypes.sizeof(accounting), None,
            ))
            if accounting.active == 0:
                return self.assigned
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return False
            await asyncio.sleep(min(0.01, remaining))

    def close(self) -> None:
        _kernel.CloseHandle(self.handle)

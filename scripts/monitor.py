"""Resource monitor: appends one CSV line every N seconds (flushed + fsync'd, so the last readings survive a crash).

    python scripts/monitor.py [interval_s] [out.csv]          # run in the background during GPU jobs
    python scripts/monitor.py --check [min_free_ram_gb] [max_gpu_used_mb]   # preflight: exit 1 if resources are short
Columns: time, host RAM used %, host RAM free GB, GPU mem used MiB, GPU util %, GPU temp C, GPU power W
"""
import ctypes
import os
import subprocess
import sys
import time


class MEMSTATUS(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]


def host_ram():
    s = MEMSTATUS()
    s.dwLength = ctypes.sizeof(MEMSTATUS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return s.dwMemoryLoad, s.ullAvailPhys / 2**30


def gpu():
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
    return [float(x) for x in r.stdout.strip().split(",")]


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        min_free = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
        max_gpu = float(sys.argv[3]) if len(sys.argv) > 3 else 2500.0
        load, free = host_ram()
        used, util, temp, pw = gpu()
        ok = free >= min_free and used <= max_gpu
        print(f"preflight: free RAM {free:.1f} GB (need >= {min_free}), GPU used {used:.0f} MiB (need <= {max_gpu:.0f}), "
              f"GPU temp {temp:.0f} C -> {'OK' if ok else 'ABORT'}")
        sys.exit(0 if ok else 1)
    interval = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join("results", "logs", "monitor.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    new = not os.path.exists(out)
    with open(out, "a") as f:
        if new:
            f.write("time,ram_used_pct,ram_free_gb,gpu_mem_mib,gpu_util_pct,gpu_temp_c,gpu_power_w\n")
        while True:
            load, free = host_ram()
            try:
                g = gpu()
            except Exception:  # noqa: BLE001
                g = [float("nan")] * 4
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{load},{free:.2f},{g[0]:.0f},{g[1]:.0f},{g[2]:.0f},{g[3]:.0f}\n")
            f.flush()
            os.fsync(f.fileno())
            time.sleep(interval)


if __name__ == "__main__":
    main()

"""Step 0: env probe -> runs/h100-session/env.json (GPU name/memory, batch preset).

Usage: python scripts/h100_00_env.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from h100_common import SESS, mark  # noqa: E402


def main():
    SESS.mkdir(parents=True, exist_ok=True)
    info = {}
    try:
        import torch  # noqa: PLC0415
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_mem_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    except Exception as e:  # noqa: BLE001
        info["torch_error"] = str(e)
    mem = info.get("gpu_mem_gb", 0)
    info["preset_batch"] = 8 if mem >= 130 else (4 if mem >= 70 else 2)
    if mem < 70:
        print(f"WARN: {mem:.0f}GB GPU is neither H100 nor H200; batch=2, expect slow/OOM.", flush=True)
    (SESS / "env.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps(info, indent=1), flush=True)
    mark("env", info)


if __name__ == "__main__":
    main()

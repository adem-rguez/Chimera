"""Step 1: check_2b mask battery, fp32, on the CORRECT backend per base.

E2B runs --attn eager: the H200 SDPA path drops the sliding-window isolation
mask (directional leak 0.78, eager 1e-7; see debug_iso). E4B runs the default
(sdpa), on which it is GREEN.

STOP the session if E2B fails. E4B failure only blocks E4B items.

Usage: python scripts/h100_01_check.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from h100_common import E2B, E4B, PY, SESS, SUITE, mark, run  # noqa: E402


def main():
    rc = run([PY, "scripts/check_2b.py", "--base", E2B, "--suite", SUITE, "--dtype", "fp32",
              "--attn", "eager"],
             str(SESS / "check-e2b.log"))
    if rc != 0:
        print("STOP: E2B check_2b failed on the GPU box. Do not train. Paste the log back.", flush=True)
        sys.exit(1)
    rc4 = run([PY, "scripts/check_2b.py", "--base", E4B, "--suite", SUITE, "--dtype", "fp32"],
              str(SESS / "check-e4b.log"))
    mark("check1", {"e4b_rc": rc4,
                    "note": "e4b nonzero = blocked (likely memory); E2B items still valid" if rc4 else "both green"})
    if rc4 != 0:
        print("WARN: E4B check_2b failed; continuing with E2B items only.", flush=True)


if __name__ == "__main__":
    main()

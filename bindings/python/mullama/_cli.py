"""Launcher shim that execs the bundled `mullama` daemon binary."""

import os
import sys
from pathlib import Path


def _binary_path() -> Path:
    name = "mullama.exe" if sys.platform == "win32" else "mullama"
    return Path(__file__).resolve().parent / "bin" / name


def main() -> "int | None":
    binary = _binary_path()
    if not binary.exists():
        sys.stderr.write(
            f"mullama: bundled binary not found at {binary}. "
            "This wheel was built without the daemon binary; install a platform "
            "wheel from PyPI or download a release from "
            "https://github.com/cognisoc/mullama/releases\n"
        )
        return 1

    args = [str(binary), *sys.argv[1:]]
    if sys.platform == "win32":
        import subprocess

        return subprocess.call(args)

    os.execv(str(binary), args)


if __name__ == "__main__":
    sys.exit(main() or 0)

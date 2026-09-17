"""One-command launcher for the local face scanner."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import threading
import webbrowser


def ensure_dependencies() -> None:
    """Install missing runtime packages for this Windows-friendly launcher.

    `face-recognition` normally declares the source-only `dlib` package as a
    dependency.  The project uses the prebuilt `dlib-bin` runtime instead, so
    it is installed after the pinned requirements without dependency solving.
    """

    base_packages = ("flask", "cv2", "numpy", "PIL", "dlib", "face_recognition_models")
    needs_base_packages = any(importlib.util.find_spec(package) is None for package in base_packages)
    needs_face_recognition = importlib.util.find_spec("face_recognition") is None

    if needs_base_packages:
        print("Installing local face-scanner dependencies…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.local.txt"])
    if needs_face_recognition:
        print("Installing face-recognition…")
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "face-recognition==1.3.0",
        ])


ensure_dependencies()

from app import app  # noqa: E402  (must load after the first-run dependency check)


def open_browser() -> None:
    webbrowser.open_new("http://127.0.0.1:5000")


if __name__ == "__main__":
    # Opening in a short-lived thread lets Flask bind the port immediately.
    threading.Timer(0.8, open_browser).start()
    # 0.0.0.0 permits the companion Android app to connect on the same Wi-Fi.
    # --local-only keeps all biometric frames private to this computer.
    host = "127.0.0.1" if "--local-only" in sys.argv else "0.0.0.0"
    app.run(host=host, port=5000, debug=False, use_reloader=False, threaded=True)

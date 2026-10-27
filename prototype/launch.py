"""Start the Streamlit app on a remote machine (Datalab, JupyterHub, Kaggle, any server) and make it
reachable from your browser.

    python launch.py                    # JupyterHub proxy link if available, else a Cloudflare tunnel
    python launch.py --tunnel cloudflare
    python launch.py --tunnel none      # local only, http://localhost:8501

Cloudflare quick tunnels need no account. cloudflared is downloaded once to ~/.cache/cloudflared.
A tunnel URL is public, so the app is protected by a random access key that is part of the printed link.
Stop with Ctrl+C (or interrupt the notebook cell).
"""
import argparse
import importlib.util
import os
import platform
import re
import secrets
import stat
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
RELEASES = "https://github.com/cloudflare/cloudflared/releases/latest/download/"


def cloudflared_binary():
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("aarch64", "arm64") else "amd64"
    system = platform.system()
    if system == "Linux":
        name = f"cloudflared-linux-{arch}"
    elif system == "Windows":
        name = f"cloudflared-windows-{arch}.exe"
    else:
        sys.exit("Automatic cloudflared download supports Linux and Windows. On macOS: brew install cloudflared")
    path = Path.home() / ".cache" / "cloudflared" / name
    if not path.exists():
        print(f"Downloading {name} ...", flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(RELEASES + name, path)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def jupyter_proxy_url(port):
    """Link through jupyter-server-proxy, when running inside JupyterHub (behind the hub's own login)."""
    prefix = os.environ.get("JUPYTERHUB_SERVICE_PREFIX")
    if not prefix or importlib.util.find_spec("jupyter_server_proxy") is None:
        return None
    host = os.environ.get("JUPYTERHUB_PUBLIC_HUB_URL") or os.environ.get("JUPYTERHUB_EXTERNAL_URL") or ""
    return f"{host.rstrip('/')}{prefix}proxy/{port}/"


def wait_for_port(port, timeout=60):
    import socket
    end = time.time() + timeout
    while time.time() < end:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.5)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tunnel", choices=["auto", "cloudflare", "jupyter", "none"], default="auto")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()

    proxy = jupyter_proxy_url(args.port)
    mode = args.tunnel
    if mode == "auto":
        mode = "jupyter" if proxy else "cloudflare"
    if mode == "jupyter" and not proxy:
        sys.exit("No JupyterHub with jupyter-server-proxy found. Use --tunnel cloudflare.")

    env = os.environ.copy()
    if mode == "cloudflare":
        env["APP_ACCESS_KEY"] = secrets.token_urlsafe(16)
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(args.port),
           "--server.address", "127.0.0.1", "--server.headless", "true",
           "--browser.gatherUsageStats", "false"]
    if mode == "jupyter":
        # the proxy rewrites the origin, which Streamlit's CORS/XSRF checks would reject
        cmd += ["--server.enableCORS", "false", "--server.enableXsrfProtection", "false"]
    app = subprocess.Popen(cmd, cwd=HERE, env=env)
    procs = [app]
    try:
        if not wait_for_port(args.port):
            sys.exit("Streamlit did not start; see the output above.")
        if mode == "jupyter":
            print(f"\nOpen: {proxy}\n(relative link if the host part is missing: append it to your Datalab URL)",
                  flush=True)
        elif mode == "cloudflare":
            tunnel = subprocess.Popen([str(cloudflared_binary()), "tunnel", "--no-autoupdate", "--url",
                                       f"http://127.0.0.1:{args.port}"],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            procs.append(tunnel)
            found = threading.Event()

            def read_log():  # cloudflared prints the public URL on stderr
                for line in tunnel.stderr:
                    m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
                    if m and not found.is_set():
                        found.set()
                        print(f"\nOpen: {m.group(0)}/?key={env['APP_ACCESS_KEY']}\n"
                              "Anyone with this full link can use the app (and your Earth Engine quota). "
                              "Do not share it.", flush=True)

            threading.Thread(target=read_log, daemon=True).start()
            if not found.wait(60):
                print("No tunnel URL after 60 s. Outbound connections may be blocked; try --tunnel jupyter.",
                      flush=True)
        else:
            print(f"\nOpen: http://localhost:{args.port}", flush=True)
        app.wait()
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    main()

"""Fixed host HTTP destination via one Unix socket in an unshared namespace.

The host serves filtered HTTP, never CONNECT or arbitrary proxy destinations.
The same stdlib-only file runs the sandbox's TCP-to-Unix relay. Credentials stay
in the host subscription adapter or upstream proxy and never cross this socket.
"""

from __future__ import annotations

import http.client
import http.server
import json
import math
import selectors
import socket
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

SANDBOX_PORT = 8765
SANDBOX_AUTHORITY = f"127.0.0.1:{SANDBOX_PORT}"
MAX_BODY = 8 * 1024 * 1024


def _request_body(handler):
    lengths = handler.headers.get_all("Content-Length", [])
    if handler.command == "GET" and not lengths and not handler.headers.get("Transfer-Encoding"):
        return b""
    if handler.headers.get("Transfer-Encoding") or len(lengths) != 1:
        raise ValueError("explicit unambiguous content length required")
    if not lengths[0].isascii() or not lengths[0].isdigit():
        raise ValueError("invalid content length")
    length = int(lengths[0])
    if length > MAX_BODY:
        raise ValueError("request too large")
    body = handler.rfile.read(length)
    if len(body) != length:
        raise ValueError("truncated request")
    if handler.command == "POST":
        payload = json.loads(body, object_pairs_hook=_unique_fields)
        if not isinstance(payload, dict) or payload.get("model") != handler.server.endpoint.model:
            raise ValueError("model outside provider grant")
        if payload.get("background", False) is not False:
            raise ValueError("background provider jobs are not granted")
    return body


def _unique_fields(pairs):
    fields = dict(pairs)
    if len(fields) != len(pairs):
        raise ValueError("ambiguous JSON fields")
    return fields


class _ProviderHandler(http.server.BaseHTTPRequestHandler):
    def setup(self):
        self.request.settimeout(max(0.001, self.server.deadline - time.monotonic()))
        super().setup()

    def log_message(self, *_args):
        pass  # Neither request bodies nor upstream errors enter controller logs.

    def do_GET(self):
        self._forward()

    def do_POST(self):
        self._forward()

    def _forward(self):
        policy = self.server.endpoint
        if (
            self.command not in policy.methods
            or self.path not in policy.paths
            or self.requestline.split()[1] != self.path
            or self.headers.get_all("Host", []) != [SANDBOX_AUTHORITY]
            or self.headers.get("Upgrade")
        ):
            self.send_error(403, "route outside provider grant")
            return
        try:
            body = _request_body(self)
            self._upstream(body)
        except (OSError, ValueError, http.client.HTTPException):
            self.close_connection = True

    def _upstream(self, body):
        seconds = self.server.deadline - time.monotonic()
        if seconds <= 0:
            return
        subscription = self.server.subscription_headers
        upstream = (http.client.HTTPSConnection("chatgpt.com", timeout=seconds) if subscription
                    else http.client.HTTPConnection("127.0.0.1", self.server.endpoint.port, timeout=seconds))
        connection = None
        try:
            upstream.connect()
            connection = upstream.sock
            self.server.track(connection)
            # Rebuild headers. In particular, no worker auth, proxy, forwarding,
            # hop-by-hop headers or hostile Host reaches the credential proxy.
            headers = {"Content-Type": "application/json", "Accept": "text/event-stream, application/json"}
            headers.update(subscription)
            path = "/backend-api/codex/responses" if subscription else self.path
            self.server.require_active()
            upstream.request(self.command, path, body=body, headers=headers)
            response = upstream.getresponse()
            if 300 <= response.status < 400 or response.status == 101:
                self.send_error(502, "provider redirect or upgrade refused")
                return
            self.send_response(response.status)
            self.send_header("Content-Type", response.getheader("Content-Type", "application/json"))
            self.send_header("Connection", "close")
            self.end_headers()
            while chunk := response.read1(65536):
                self.wfile.write(chunk)
            self.close_connection = True
        finally:
            self.server.untrack(connection)
            upstream.close()


class _ProviderServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    block_on_close = False

    def __init__(self, path, endpoint, deadline, cancel_event):
        self.endpoint, self.deadline, self.cancel_event = endpoint, deadline, cancel_event
        self.subscription_headers = {}
        self.connections = set()
        self.lock = threading.Lock()
        self.stopping = False
        self.slots = threading.BoundedSemaphore(8)
        super().__init__(str(path), _ProviderHandler)

    def require_active(self):
        with self.lock:
            if self.stopping or self.cancel_event.is_set() or time.monotonic() >= self.deadline:
                raise OSError("provider grant ended")

    def track(self, connection):
        with self.lock:
            if self.stopping or self.cancel_event.is_set() or time.monotonic() >= self.deadline:
                connection.close()
                raise OSError("provider grant ended")
            self.connections.add(connection)

    def untrack(self, connection):
        with self.lock:
            self.connections.discard(connection)

    def process_request(self, request, address):
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        super().process_request(request, address)

    def process_request_thread(self, request, address):
        try:
            self.track(request)
            super().process_request_thread(request, address)
        finally:
            self.untrack(request)
            self.slots.release()

    def handle_error(self, *_args):
        pass

    def stop(self):
        with self.lock:
            self.stopping = True
            for connection in self.connections:
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                connection.close()
        self.shutdown()
        self.server_close()


class ProviderBridge:
    """Owned by the controller process; threads cannot outlive parent death."""

    def __init__(self, endpoint, seconds, cancel_event, *, subscription=(None, None)):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("provider_grant_invalid_duration")
        self.endpoint, self.seconds, self.cancel_event = endpoint, seconds, cancel_event
        self.subscription = subscription
        self.finished = threading.Event()

    def __enter__(self):
        from slm_training.harness_core.codex_subscription import _check_cancel

        _check_cancel(self.cancel_event)
        deadline = time.monotonic() + self.seconds
        headers = {}
        if self.endpoint.authentication == "codex_subscription":
            from slm_training.harness_core.codex_subscription import subscription_headers

            headers = subscription_headers(*self.subscription, deadline, cancel_event=self.cancel_event)
        _check_cancel(self.cancel_event)
        if deadline <= time.monotonic():
            raise ValueError("provider_grant_expired")
        self.directory = tempfile.TemporaryDirectory(prefix="slm-provider-")
        self.path = Path(self.directory.name) / "provider.sock"
        try:
            self.server = _ProviderServer(self.path, self.endpoint, deadline, self.cancel_event)
            self.server.subscription_headers = headers
        except BaseException:
            self.directory.cleanup()
            raise
        self.path.chmod(0o600)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self.thread.start()
        self.watcher = threading.Thread(target=self._watch, daemon=True)
        self.watcher.start()
        return self.path

    def _watch(self):
        while not self.finished.wait(0.05):
            if self.cancel_event.is_set() or time.monotonic() >= self.server.deadline:
                break
        self.server.stop()

    def __exit__(self, *_args):
        self.finished.set()
        self.watcher.join(timeout=1)
        self.thread.join(timeout=1)
        self.directory.cleanup()


class _RelayHandler(socketserver.BaseRequestHandler):
    def handle(self):
        with socket.socket(socket.AF_UNIX) as upstream, selectors.DefaultSelector() as ready:
            upstream.connect(self.server.unix_path)
            ready.register(self.request, selectors.EVENT_READ, upstream)
            ready.register(upstream, selectors.EVENT_READ, self.request)
            while True:
                for key, _mask in ready.select():
                    data = key.fileobj.recv(65536)
                    if not data:
                        return
                    key.data.sendall(data)


def relay_main(unix_path, argv):
    """Relay and workload live inside Bubblewrap's parent-bound PID namespace."""
    with socketserver.ThreadingTCPServer(("127.0.0.1", SANDBOX_PORT), _RelayHandler) as server:
        server.daemon_threads = True
        server.block_on_close = False
        server.unix_path = unix_path
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            return subprocess.call(argv)
        finally:
            server.shutdown()


if __name__ == "__main__":
    sys.exit(relay_main(sys.argv[1], sys.argv[2:]))

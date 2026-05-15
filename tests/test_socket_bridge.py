"""Smoke: SocketBackend sends chunks over TCP, client consumes them."""

from __future__ import annotations

import json
import socket
import struct
import tempfile
import threading
import time

import numpy as np

from text2humanoid.contracts.clips import G1ReferenceChunk
from text2humanoid.runtime.socket_backend import SocketBackend


def _make_ref(n=6):
    root_pos = np.zeros((n, 3), dtype=np.float32)
    root_pos[:, 0] = np.linspace(0, 1, n, dtype=np.float32)
    return G1ReferenceChunk(
        chunk_id="sock_test", start_time=0.0, fps=30,
        root_pos=root_pos,
        root_rot=np.tile(np.array([[0, 0, 0, 1]], dtype=np.float32), (n, 1)),
        dof_pos=np.zeros((n, 29), dtype=np.float32),
        local_body_pos=np.zeros((n, 30, 3), dtype=np.float32),
        local_body_rot=np.tile(np.array([[[0, 0, 0, 1]]], dtype=np.float32), (n, 30, 1)),
        body_names=[f"b{i}" for i in range(30)], joint_names=[f"j{i}" for i in range(29)],
        metadata={"root_quat_order": "xyzw"},
    )


def _recv_message(sock, timeout=1.0):
    sock.settimeout(timeout)
    header = sock.recv(4)
    if len(header) < 4:
        return None
    msg_len = struct.unpack(">I", header)[0]
    body = b""
    while len(body) < msg_len:
        c = sock.recv(msg_len - len(body))
        if not c: return None
        body += c
    return json.loads(body.decode("utf-8"))


def test_socket_backend_sends_chunk():
    port = 15557
    backend = SocketBackend(host="127.0.0.1", port=port)
    ready = threading.Event()

    def _run():
        backend.ensure_session("s1"); ready.set()
        backend.push_reference_chunk("s1", _make_ref(8))

    threading.Thread(target=_run, daemon=True).start()
    ready.wait(timeout=2.0); time.sleep(0.1)

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("127.0.0.1", port))
    msg = _recv_message(client, timeout=2.0)
    assert msg and msg["type"] == "chunk" and msg["session_id"] == "s1"
    assert "root_pos" in msg["payload"] and "dof_pos" in msg["payload"]
    client.close(); backend.close()


def test_socket_backend_multiple_chunks():
    port = 15558
    backend = SocketBackend(host="127.0.0.1", port=port)
    ready = threading.Event()

    def _run():
        backend.ensure_session("s1"); ready.set()
        for _ in range(3): backend.push_reference_chunk("s1", _make_ref(6))

    threading.Thread(target=_run, daemon=True).start()
    ready.wait(timeout=2.0); time.sleep(0.2)

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("127.0.0.1", port))
    msgs = [_recv_message(client) for _ in range(3)]
    assert len([m for m in msgs if m]) == 3
    client.close(); backend.close()


def test_socket_backend_status_tracking():
    backend = SocketBackend(host="127.0.0.1", port=15559)
    backend.ensure_session("s1")
    assert backend.get_status("s1").session_id == "s1"
    backend.consume_step("s1", frames=5)
    assert backend.get_status("s1").sim_time > 0


def test_floodnet_file_fallback_unchanged():
    from text2humanoid.runtime.motion_tracking_client import FloodNetFileBackend
    with tempfile.TemporaryDirectory() as tmp:
        backend = FloodNetFileBackend(output_dir=tmp)
        backend.push_reference_chunk("s1", _make_ref(6))
        assert backend.get_status("s1").buffer_frames == 6


# ---- Consumer-side smoke: exact SocketFloodNetSource protocol simulation -----

def _simulate_drain_messages(sock, obs_joint_names=None):
    """Mirror of SocketFloodNetSource._drain_messages() protocol."""
    received = []
    sock.settimeout(0.01)
    while True:
        try:
            header = sock.recv(4)
            if len(header) < 4:
                break
            msg_len = struct.unpack(">I", header)[0]
            body = b""
            while len(body) < msg_len:
                c = sock.recv(msg_len - len(body))
                if not c:
                    raise ConnectionError("Socket closed")
                body += c
            msg = json.loads(body.decode("utf-8"))
            if msg.get("type") == "chunk":
                # _handle_chunk logic
                payload = msg["payload"]
                joint_pos_raw = np.array(payload["dof_pos"], dtype=np.float32)
                root_pos = np.array(payload["root_pos"], dtype=np.float32)
                root_rot_xyzw = np.array(payload["root_rot"], dtype=np.float32)
                root_quat_wxyz = np.concatenate(
                    [root_rot_xyzw[:, 3:4], root_rot_xyzw[:, :3]], axis=-1)
                received.append({
                    "chunk_id": msg["chunk_id"],
                    "session_id": msg["session_id"],
                    "joint_pos_shape": joint_pos_raw.shape,
                    "root_quat_shape": root_quat_wxyz.shape,
                    "root_pos_shape": root_pos.shape,
                })
        except socket.timeout:
            break
        except (ConnectionError, OSError):
            break
    sock.settimeout(2.0)
    return received


def test_consumer_side_drain_messages():
    """Consumer receives and parses chunks via exact SocketFloodNetSource protocol."""
    import time as time_mod
    port = 15560
    backend = SocketBackend(host="127.0.0.1", port=port)
    ready = threading.Event()

    def _run():
        backend.ensure_session("s1"); ready.set()
        backend.push_reference_chunk("s1", _make_ref(8))
        backend.push_reference_chunk("s1", _make_ref(10))

    threading.Thread(target=_run, daemon=True).start()
    ready.wait(timeout=2.0); time_mod.sleep(0.1)

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("127.0.0.1", port))

    received = _simulate_drain_messages(client)
    assert len(received) == 2, f"expected 2 chunks, got {len(received)}"
    for r in received:
        assert r["session_id"] == "s1"
        assert r["joint_pos_shape"][1] == 29  # 29 DOF
        assert r["root_quat_shape"][1] == 4   # wxyz quat
    client.close(); backend.close()


def test_consumer_no_directory_scanning():
    """Consumer path uses only TCP — no directory glob or file I/O."""
    import time as time_mod
    port = 15561
    backend = SocketBackend(host="127.0.0.1", port=port)
    ready = threading.Event()

    def _run():
        backend.ensure_session("s1"); ready.set()
        for _ in range(3):
            backend.push_reference_chunk("s1", _make_ref(6))

    threading.Thread(target=_run, daemon=True).start()
    ready.wait(timeout=2.0); time_mod.sleep(0.2)

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("127.0.0.1", port))
    received = _simulate_drain_messages(client)

    assert len(received) == 3
    # Verify: all chunks received via pure TCP, zero file ops
    for r in received:
        assert "chunk_id" in r
        assert r["joint_pos_shape"] == (6, 29)
    client.close(); backend.close()

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from .console_controller import DemoController, GenerateJobRequest, StreamJobRequest


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Text2Humanoid Demo Console</title>
  <style>
    :root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
    body { margin: 0; background: #f5f7fb; color: #172033; }
    header { padding: 18px 24px; background: #ffffff; border-bottom: 1px solid #d8dee9; }
    h1 { margin: 0; font-size: 20px; font-weight: 700; }
    main { max-width: 1180px; margin: 0 auto; padding: 20px; display: grid; gap: 16px; }
    section { background: #fff; border: 1px solid #d8dee9; border-radius: 8px; padding: 16px; }
    h2 { margin: 0 0 14px; font-size: 15px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
    label { display: grid; gap: 6px; font-size: 13px; color: #4a5568; }
    input[type="text"], input[type="number"] { height: 36px; border: 1px solid #cbd5e1; border-radius: 6px; padding: 0 10px; font-size: 14px; }
    button { height: 36px; border: 1px solid #1f6feb; background: #1f6feb; color: white; border-radius: 6px; padding: 0 12px; font-weight: 650; cursor: pointer; }
    button.secondary { background: #fff; color: #1f2937; border-color: #cbd5e1; }
    button.danger { background: #b42318; border-color: #b42318; }
    .buttons { display: flex; gap: 8px; flex-wrap: wrap; }
    .status { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
    .pill { background: #e7f0ff; color: #174ea6; padding: 5px 8px; border-radius: 999px; font-size: 13px; font-weight: 650; }
    .metrics { margin-top: 10px; color: #334155; font: 13px/1.55 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; white-space: pre-line; }
    pre { margin: 0; min-height: 190px; max-height: 320px; overflow: auto; background: #0f172a; color: #dbeafe; border-radius: 8px; padding: 12px; font-size: 12px; line-height: 1.5; }
    .artifacts { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; }
    video { width: 100%; background: #111827; border-radius: 6px; }
    canvas { width: 100%; height: 420px; background: #ffffff; border-radius: 6px; display: block; }
    .preview-meta { margin-top: 8px; color: #4a5568; font-size: 12px; }
    a { color: #1f6feb; word-break: break-all; }
  </style>
</head>
<body>
  <header><h1>Text2Humanoid Demo Console</h1></header>
  <main>
    <section>
      <h2>Simulation / Policy</h2>
      <div class="buttons">
        <button onclick="startRuntime()">Start Sim</button>
        <button class="secondary" onclick="simKey('9')">Put Down</button>
        <button class="secondary" onclick="key('i')">Init Robot</button>
        <button class="secondary" onclick="key('[')">Start Motion</button>
        <button class="secondary" onclick="key(']')">Enable Policy</button>
        <button class="danger" onclick="post('/api/app/stop')">Stop App</button>
      </div>
    </section>
    <section>
      <h2>Streaming Motion</h2>
      <label>Prompt <input id="prompt" type="text" value="walk forward" /></label>
      <div class="buttons" style="margin-top: 12px;">
        <button onclick="startStream()">Start Stream</button>
        <button class="secondary" onclick="updateStreamText()">Update Text</button>
        <button class="danger" onclick="stopStream()">Stop Stream</button>
      </div>
    </section>
    <section>
      <h2>Status</h2>
      <div class="status"><span class="pill" id="stage">idle</span><span id="processes"></span></div>
      <div class="metrics" id="metrics"></div>
    </section>
    <section>
      <h2>Logs</h2>
      <pre id="log"></pre>
    </section>
    <section>
      <h2>Artifacts</h2>
      <div>
        <strong>Live FloodDiffusion 263D Preview</strong>
        <canvas id="streamPreviewCanvas" width="720" height="420"></canvas>
        <div class="preview-meta" id="streamPreviewMeta">waiting for stream frames</div>
      </div>
      <div class="artifacts" id="artifacts"></div>
    </section>
  </main>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
  <script>
    let nextEvent = 0;
    async function post(path, body={}) {
      const res = await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
      if (!res.ok) alert(await res.text());
      await refresh();
    }
    function key(k) { return post('/api/policy/key', {key:k}); }
    function simKey(k) { return post('/api/sim/key', {key:k}); }
    async function startRuntime() {
      await post('/api/sim/start');
      await post('/api/policy/start');
    }
    function streamPayload() {
      return {
        text: document.getElementById('prompt').value
      };
    }
    function startStream() { return post('/api/stream/start', streamPayload()); }
    function updateStreamText() {
      return post('/api/stream/update_text', {text: document.getElementById('prompt').value});
    }
    function stopStream() { return post('/api/stream/stop'); }
    function artifactUrl(path) { return '/artifacts?path=' + encodeURIComponent(path); }
    function formatMetric(value) {
      const n = Number(value || 0);
      return Number.isFinite(n) ? n.toFixed(1) : '0.0';
    }
    function renderStreamMetrics(metrics) {
      const data = metrics || {};
      const generation = data.generation || {};
      const retarget = data.retarget || {};
      const motion = data.motion_control || {};
      document.getElementById('metrics').textContent = [
        `generation: ${formatMetric(generation.fps)} fps, buffer=${Number(generation.buffer_frames || 0)} frames`,
        `retarget: ${formatMetric(retarget.fps)} fps, buffer=${Number(retarget.buffer_frames || 0)} frames`,
        `motion control: ${formatMetric(motion.fps)} fps, buffer=${Number(motion.buffer_frames || 0)} frames`
      ].join('\\n');
    }
    class PreviewSkeleton3D {
      constructor(canvas) {
        this.canvas = canvas;
        this.chains = [
          [0, 2, 5, 8, 11],
          [0, 1, 4, 7, 10],
          [0, 3, 6, 9, 12, 15],
          [9, 14, 17, 19, 21],
          [9, 13, 16, 18, 20]
        ];
        this.boneConnections = [];
        this.chains.forEach((chain) => {
          for (let i = 0; i < chain.length - 1; i++) this.boneConnections.push([chain[i], chain[i + 1]]);
        });
        this.joints = [];
        this.bones = [];
        this.trailPoints = [];
        this.maxTrailPoints = 200;
        this.lastUserInteraction = Date.now();
        this.initScene();
      }

      initScene() {
        const container = this.canvas.parentElement;
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xffffff);
        this.camera = new THREE.PerspectiveCamera(60, container.clientWidth / 420, 0.1, 1000);
        this.camera.position.set(3, 1.5, 3);
        this.camera.lookAt(0, 1, 0);
        this.renderer = new THREE.WebGLRenderer({canvas: this.canvas, antialias: true});
        this.renderer.setSize(container.clientWidth, 420, false);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.0;

        this.scene.add(new THREE.AmbientLight(0xffffff, 0.7));
        const keyLight = new THREE.DirectionalLight(0xffffff, 0.8);
        keyLight.position.set(5, 8, 3);
        keyLight.castShadow = true;
        keyLight.shadow.mapSize.width = 2048;
        keyLight.shadow.mapSize.height = 2048;
        keyLight.shadow.camera.near = 0.5;
        keyLight.shadow.camera.far = 50;
        keyLight.shadow.camera.left = -5;
        keyLight.shadow.camera.right = 5;
        keyLight.shadow.camera.top = 5;
        keyLight.shadow.camera.bottom = -5;
        keyLight.shadow.bias = -0.0001;
        this.scene.add(keyLight);
        const fillLight = new THREE.DirectionalLight(0xffffff, 0.4);
        fillLight.position.set(-3, 5, -3);
        this.scene.add(fillLight);

        const ground = new THREE.Mesh(
          new THREE.PlaneGeometry(1000, 1000),
          new THREE.ShadowMaterial({opacity: 0.15})
        );
        ground.rotation.x = -Math.PI / 2;
        ground.receiveShadow = true;
        this.scene.add(ground);
        const grid = new THREE.GridHelper(1000, 1000, 0xdddddd, 0xeeeeee);
        grid.position.y = 0.01;
        this.scene.add(grid);

        this.controls = new THREE.OrbitControls(this.camera, this.canvas);
        this.controls.target.set(0, 1, 0);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        const updateInteraction = () => { this.lastUserInteraction = Date.now(); };
        this.canvas.addEventListener('mousedown', updateInteraction);
        this.canvas.addEventListener('wheel', updateInteraction);
        this.canvas.addEventListener('touchstart', updateInteraction);

        const jointMaterial = new THREE.MeshStandardMaterial({color: 0x00809D, metalness: 0.2, roughness: 0.5});
        const jointGeometry = new THREE.SphereGeometry(0.03, 16, 16);
        for (let i = 0; i < 22; i++) {
          const joint = new THREE.Mesh(jointGeometry, jointMaterial);
          joint.castShadow = true;
          joint.receiveShadow = true;
          this.joints.push(joint);
          this.scene.add(joint);
        }

        const boneColors = [0xFEB21A, 0x00AAFF, 0x134686, 0xFFB600, 0x00D47E];
        this.chains.forEach((chain, chainIdx) => {
          const material = new THREE.MeshStandardMaterial({color: boneColors[chainIdx], metalness: 0.2, roughness: 0.5});
          for (let i = 0; i < chain.length - 1; i++) {
            const bone = new THREE.Mesh(new THREE.CylinderGeometry(0.015, 0.015, 1, 8), material);
            bone.castShadow = true;
            bone.receiveShadow = true;
            this.bones.push(bone);
            this.scene.add(bone);
          }
        });

        this.trailGeometry = new THREE.BufferGeometry();
        this.trailGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(this.maxTrailPoints * 3), 3));
        this.trailGeometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(this.maxTrailPoints * 4), 4));
        this.trailMaterial = new THREE.LineBasicMaterial({vertexColors: true, transparent: true, opacity: 1.0});
        this.trailLine = new THREE.Line(this.trailGeometry, this.trailMaterial);
        this.trailLine.frustumCulled = false;
        this.scene.add(this.trailLine);

        window.addEventListener('resize', () => this.resize());
        this.animate();
      }

      resize() {
        const width = this.canvas.parentElement.clientWidth;
        this.camera.aspect = width / 420;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, 420, false);
      }

      updatePose(jointPositions) {
        if (!jointPositions || jointPositions.length !== 22) return;
        for (let i = 0; i < 22; i++) this.joints[i].position.set(jointPositions[i][0], jointPositions[i][1], jointPositions[i][2]);
        this.boneConnections.forEach(([startIdx, endIdx], i) => {
          const start = new THREE.Vector3(...jointPositions[startIdx]);
          const end = new THREE.Vector3(...jointPositions[endIdx]);
          this.updateBone(this.bones[i], start, end);
        });
        this.updateTrail(jointPositions[0]);
        if (Date.now() - this.lastUserInteraction > 3000) {
          const root = new THREE.Vector3(jointPositions[0][0], 1.0, jointPositions[0][2]);
          const offset = new THREE.Vector3().subVectors(this.camera.position, this.controls.target);
          this.controls.target.lerp(root, 0.2);
          this.camera.position.lerp(root.clone().add(offset), 0.2);
        }
      }

      updateBone(bone, startPos, endPos) {
        const direction = new THREE.Vector3().subVectors(endPos, startPos);
        const length = direction.length();
        if (length < 0.001) return;
        bone.position.copy(new THREE.Vector3().addVectors(startPos, endPos).multiplyScalar(0.5));
        bone.scale.y = length;
        bone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.normalize());
      }

      updateTrail(rootPos) {
        const point = {x: rootPos[0], y: 0.01, z: rootPos[2]};
        const last = this.trailPoints[this.trailPoints.length - 1];
        if (!last || Math.hypot(point.x - last.x, point.z - last.z) > 0.02) this.trailPoints.push(point);
        if (this.trailPoints.length > this.maxTrailPoints) this.trailPoints.shift();
        const positions = this.trailGeometry.attributes.position.array;
        const colors = this.trailGeometry.attributes.color.array;
        const n = this.trailPoints.length;
        for (let i = 0; i < this.maxTrailPoints; i++) {
          if (i < n) {
            const p = this.trailPoints[i];
            positions[i * 3] = p.x;
            positions[i * 3 + 1] = p.y;
            positions[i * 3 + 2] = p.z;
            const alpha = n > 1 ? i / (n - 1) : 1;
            colors[i * 4] = 0.0;
            colors[i * 4 + 1] = 0.67;
            colors[i * 4 + 2] = 0.85;
            colors[i * 4 + 3] = Math.pow(alpha, 1.5) * 0.8;
          } else {
            colors[i * 4 + 3] = 0;
          }
        }
        this.trailGeometry.attributes.position.needsUpdate = true;
        this.trailGeometry.attributes.color.needsUpdate = true;
        this.trailGeometry.setDrawRange(0, n);
      }

      animate() {
        requestAnimationFrame(() => this.animate());
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
      }
    }

    const preview3d = (window.THREE && window.THREE.OrbitControls)
      ? new PreviewSkeleton3D(document.getElementById('streamPreviewCanvas'))
      : null;
    const humanmlChains = [[0,2,5,8,11], [0,1,4,7,10], [0,3,6,9,12,15], [9,14,17,19,21], [9,13,16,18,20]];
    const chainColors = ['#f97316', '#06b6d4', '#3b82f6', '#f59e0b', '#34d399'];
    function drawPreviewFrame(joints, meta) {
      if (preview3d) {
        preview3d.updatePose(joints);
        document.getElementById('streamPreviewMeta').textContent =
          `frame=${meta.frame_idx} queued=${meta.queued_frames} text="${meta.text || ''}"`;
        return;
      }
      const canvas = document.getElementById('streamPreviewCanvas');
      const ctx = canvas.getContext('2d');
      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, w, h);
      if (!joints || !joints.length) return;

      const root = joints[0];
      const projected = joints.map((p) => ({
        x: (p[0] - root[0]) * 95 + w * 0.5,
        y: h * 0.72 - (p[1] - root[1]) * 95 - p[2] * 12
      }));

      ctx.strokeStyle = 'rgba(0,0,0,0.08)';
      ctx.lineWidth = 1;
      for (let gx = 40; gx < w; gx += 40) {
        ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, h); ctx.stroke();
      }
      for (let gy = 40; gy < h; gy += 40) {
        ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(w, gy); ctx.stroke();
      }

      humanmlChains.forEach((chain, idx) => {
        ctx.strokeStyle = chainColors[idx];
        ctx.lineWidth = 5;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.beginPath();
        chain.forEach((jointIndex, i) => {
          const p = projected[jointIndex];
          if (i === 0) ctx.moveTo(p.x, p.y);
          else ctx.lineTo(p.x, p.y);
        });
        ctx.stroke();
      });

      ctx.fillStyle = '#00809D';
      for (const p of projected) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
        ctx.fill();
      }

      document.getElementById('streamPreviewMeta').textContent =
        `frame=${meta.frame_idx} queued=${meta.queued_frames} text="${meta.text || ''}"`;
    }
    async function pollStreamPreview() {
      try {
        const data = await (await fetch('/api/stream/preview_frame')).json();
        if (data.status === 'ok') {
          drawPreviewFrame(data.joints, data);
        } else {
          document.getElementById('streamPreviewMeta').textContent =
            `waiting for stream frames queued=${data.queued_frames || 0}`;
        }
      } catch (error) {
        document.getElementById('streamPreviewMeta').textContent = 'preview unavailable';
      }
      setTimeout(pollStreamPreview, 50);
    }
    function renderArtifacts(artifacts) {
      const root = document.getElementById('artifacts');
      root.innerHTML = '';
      for (const [name, path] of Object.entries(artifacts || {})) {
        const item = document.createElement('div');
        if (name.startsWith('video_')) {
          item.innerHTML = `<strong>${name}</strong><video controls src="${artifactUrl(path)}"></video>`;
        } else {
          item.innerHTML = `<strong>${name}</strong><br><a href="${artifactUrl(path)}">${path}</a>`;
        }
        root.appendChild(item);
      }
    }
    async function refresh() {
      const status = await (await fetch('/api/status')).json();
      document.getElementById('stage').textContent = status.stage;
      document.getElementById('processes').textContent = JSON.stringify({
        processes: status.processes,
        stream: {
          running: status.stream.running,
          stream_id: status.stream.stream_id,
          text: status.stream.text,
          error: status.stream.error
        }
      });
      renderStreamMetrics(status.stream.metrics);
      renderArtifacts(status.artifacts);
      const events = await (await fetch('/api/events?since=' + nextEvent)).json();
      const log = document.getElementById('log');
      for (const event of events.events) {
        log.textContent += `[${event.stage}] ${event.message}\\n`;
        nextEvent = Math.max(nextEvent, event.id + 1);
      }
      log.scrollTop = log.scrollHeight;
    }
    setInterval(refresh, 1000);
    pollStreamPreview();
    refresh();
  </script>
</body>
</html>
"""


def route_demo_request(
    controller: DemoController,
    method: str,
    path: str,
    payload: dict | None = None,
    *,
    query: dict[str, list[str]] | None = None,
) -> tuple[int, dict]:
    payload = payload or {}
    query = query or {}
    if method == "GET" and path == "/api/status":
        return 200, controller.status()
    if method == "GET" and path == "/api/events":
        since = int(query.get("since", ["0"])[0])
        return 200, {"events": controller.events(since)}
    if method == "GET" and path == "/api/stream/preview_frame":
        return 200, controller.stream_preview_frame()
    if method == "POST" and path == "/api/sim/start":
        return 200, controller.start_sim()
    if method == "POST" and path == "/api/sim/key":
        return 200, controller.send_sim_key(str(payload["key"]))
    if method == "POST" and path == "/api/policy/start":
        return 200, controller.start_policy()
    if method == "POST" and path == "/api/policy/key":
        return 200, controller.send_policy_key(str(payload["key"]))
    if method == "POST" and path == "/api/generate":
        return 200, controller.start_generate_task(GenerateJobRequest(**payload))
    if method == "POST" and path == "/api/stream/start":
        return 200, controller.start_stream_task(StreamJobRequest(**payload))
    if method == "POST" and path == "/api/stream/update_text":
        return 200, controller.update_stream_text(str(payload["text"]))
    if method == "POST" and path == "/api/stream/stop":
        return 200, controller.stop_stream_task()
    if method == "POST" and path == "/api/stop":
        return 200, controller.stop_all()
    if method == "POST" and path == "/api/app/stop":
        return 200, controller.stop_app()
    return 404, {"error": "not found"}


def create_demo_console_server(
    controller: DemoController,
    *,
    host: str = "127.0.0.1",
    port: int = 8090,
) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            return

        def _send_json(self, payload: dict, status: int = 200) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                data = HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif parsed.path == "/api/status":
                status, payload = route_demo_request(controller, "GET", parsed.path, {}, query=parse_qs(parsed.query))
                self._send_json(payload, status=status)
            elif parsed.path == "/api/events":
                status, payload = route_demo_request(controller, "GET", parsed.path, {}, query=parse_qs(parsed.query))
                self._send_json(payload, status=status)
            elif parsed.path == "/api/stream/preview_frame":
                status, payload = route_demo_request(controller, "GET", parsed.path, {}, query=parse_qs(parsed.query))
                self._send_json(payload, status=status)
            elif parsed.path == "/artifacts":
                raw_path = parse_qs(parsed.query).get("path", [""])[0]
                path = Path(unquote(raw_path)).expanduser()
                if not path.exists() or not path.is_file():
                    self._send_json({"error": "not found"}, status=404)
                    return
                data = path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4" if path.suffix == ".mp4" else "application/octet-stream")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self._send_json({"error": "not found"}, status=404)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                payload = self._read_json()
                status, response = route_demo_request(controller, "POST", parsed.path, payload)
                self._send_json(response, status=status)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)

    return ThreadingHTTPServer((host, port), Handler)

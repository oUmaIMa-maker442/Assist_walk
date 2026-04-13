import os
os.environ['GROQ_API_KEY'] = 'dummy'
#!/usr/bin/env python3
# webcam_server.py — AssistWalk debug viewer
# Usage:
#   python webcam_server.py                          # built-in webcam
#   python webcam_server.py --source 0               # webcam index
#   python webcam_server.py --source video.mp4       # local file
#   python webcam_server.py --source http://IP:8080/video  # IP cam

import argparse
import sys
import os
import time
import threading
import json
import cv2
import numpy as np
from flask import Flask, Response, render_template_string, jsonify

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'assistwalk_vision'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'text_module'))
from src.step3_yolo_detection import YOLODetector
from src.step4_filtering      import filter_objects, HIGH_PRIORITY, MEDIUM_PRIORITY
from speech                   import speak_if_new, speak_lecture
from pipeline                 import process_lecture
import speech as _speech_mod          # to flip lecture_mode flag

# ── Config ───────────────────────────────────────────────
DEFAULT_SOURCE = 0
DEFAULT_PORT   = 5000
MODEL_NAME     = 'yolov8n.pt'
CONFIDENCE     = 0.30
YOLO_EVERY_N   = 8
FRAME_WIDTH    = 640
FRAME_HEIGHT   = 480

COLOR_HIGH   = (0,  60, 220)
COLOR_MEDIUM = (0, 165, 255)
COLOR_LOW    = (60, 180,  60)

# ── Voice behaviour ──────────────────────────────────────
CLEAR_DELAY         = 3.0
MIN_REPEAT_INTERVAL = 3.0

# ── Distance estimation (inline) ─────────────────────────
def _estimate_distance(bbox, fw, fh):
    x1, y1, x2, y2 = bbox
    ratio = max((x2-x1)/fw, (y2-y1)/fh)
    if ratio > 0.5:    return "very close",         "danger"
    elif ratio > 0.25: return "nearby",              "warning"
    elif ratio > 0.10: return "at medium distance",  "info"
    else:              return "far away",             "info"

def _spoken_message(obj_name, bbox, fw, fh):
    dist, level = _estimate_distance(bbox, fw, fh)
    if level == "danger":   return f"danger!  {obj_name}  {dist}  ahead"
    elif level == "warning": return f"warning,  {obj_name}  {dist}"
    else:                    return f"{obj_name}  detected,  {dist}"


# ══════════════════════════════════════════════════════════
# HTML page — live video + detection panel
# ══════════════════════════════════════════════════════════
HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AssistWalk — Debug Viewer</title>
  <style>
    *    { box-sizing:border-box; margin:0; padding:0; }
    body { background:#111; color:#eee; font-family:monospace;
           display:flex; flex-direction:column; align-items:center;
           min-height:100vh; padding:16px 16px 24px; gap:12px; }
    h1   { font-size:1.1rem; letter-spacing:.08em; color:#7ec8e3; margin-top:8px; }

    .legend { display:flex; gap:18px; font-size:.8rem; }
    .dot { width:10px; height:10px; border-radius:50%;
           display:inline-block; margin-right:5px; vertical-align:middle; }
    .high   { background:#dc3c14; }
    .medium { background:#ffa500; }
    .low    { background:#3cb43c; }

    /* ── Two-column layout ── */
    .main-layout {
      display: flex;
      gap: 12px;
      width: 100%;
      max-width: 980px;
      align-items: flex-start;
    }
    .left-col {
      width: 30%;
      min-width: 200px;
      flex-shrink: 0;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .right-col {
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }

    #stream { width:100%; border-radius:8px;
              border:1px solid #333; display:block; }

    #panel  { background:#1a1a1a;
              border:1px solid #333; border-radius:8px; padding:14px;
              min-height:80px; }
    #panel h2 { font-size:.75rem; color:#555; margin-bottom:10px;
                text-transform:uppercase; letter-spacing:.1em; }
    #msg-list { list-style:none; display:flex; flex-direction:column; gap:6px; }
    #msg-list li {
      padding:9px 14px; border-radius:0 6px 6px 0; font-size:.85rem;
      display:flex; align-items:flex-start; gap:8px; flex-wrap:wrap;
      animation: fadein .2s ease;
    }
    @keyframes fadein { from { opacity:0; transform:translateY(-3px); } to { opacity:1; } }
    .level-danger  { background:#3a1010; border-left:3px solid #dc3c14; color:#f99; }
    .level-warning { background:#2e2000; border-left:3px solid #ffa500; color:#fcc; }
    .level-info    { background:#0e1e0e; border-left:3px solid #3cb43c; color:#aea; }
    .level-clear   { background:#1c1c1c; border-left:3px solid #444;    color:#555; font-style:italic; }
    .badge { font-size:.68rem; padding:2px 7px; border-radius:4px;
             font-weight:700; letter-spacing:.05em; white-space:nowrap; flex-shrink:0; }
    .badge-danger  { background:#dc3c14; color:#fff; }
    .badge-warning { background:#ffa500; color:#111; }
    .badge-info    { background:#3cb43c; color:#111; }
    .badge-clear   { background:#333;    color:#888; }

    #status { font-size:.68rem; color:#444; }
    footer  { font-size:.7rem; color:#444; padding-bottom:8px; }

    /* ── Reading mode ── */
    #read-btn {
      width:100%; padding:12px;
      background:#1a2a1a; border:1px solid #3cb43c; border-radius:8px;
      color:#6ed86e; font-family:monospace; font-size:.9rem;
      cursor:pointer; letter-spacing:.05em; transition: background .15s;
    }
    #read-btn:hover:not(:disabled) { background:#243a24; }
    #read-btn:disabled { opacity:.45; cursor:not-allowed; }

    #ocr-panel { background:#12121e;
                 border:1px solid #334; border-radius:8px; padding:14px;
                 display:none; }
    #ocr-panel h2 { font-size:.75rem; color:#557; margin-bottom:8px;
                    text-transform:uppercase; letter-spacing:.1em; }
    #ocr-text { font-size:.9rem; color:#cce; line-height:1.6;
                white-space:pre-wrap; word-break:break-word; }
    #ocr-meta { font-size:.68rem; color:#446; margin-top:6px; }

    /* ── Responsive: stack columns on small screens ── */
    @media (max-width: 620px) {
      .main-layout { flex-direction: column; }
      .left-col { width: 100%; min-width: unset; }
    }
  </style>
</head>
<body>
  <h1>AssistWalk — Live Detection</h1>

  <div class="legend">
    <span><span class="dot high"></span>High priority</span>
    <span><span class="dot medium"></span>Medium priority</span>
    <span><span class="dot low"></span>Low priority</span>
  </div>

  <!-- Two-column layout -->
  <div class="main-layout">

    <!-- LEFT: detection panel -->
    <div class="left-col">
      <div id="panel">
        <h2>Detection messages</h2>
        <ul id="msg-list">
          <li class="level-clear">
            <span class="badge badge-clear">—</span>
            <span>Waiting for detections…</span>
          </li>
        </ul>
      </div>
      <span id="status"></span>
    </div>

    <!-- RIGHT: video + read button + OCR result -->
    <div class="right-col">
      <img id="stream" src="/video_feed" alt="Live stream">

      <!-- Widget taille de texte (optionnel) -->
      <div style="background:#1a1a1a;border:1px solid #333;border-radius:8px;padding:10px;margin-bottom:10px;">
        <div style="font-size:.7rem;color:#555;margin-bottom:6px;text-transform:uppercase;letter-spacing:.1em;">
          Taille du texte
        </div>
        <div style="background:#111;border-radius:4px;height:10px;overflow:hidden;margin-bottom:6px;">
          <div id="text-size-bar" style="height:100%;width:0%;transition:width .4s,background .4s;border-radius:4px;"></div>
        </div>
        <div id="text-size-label" style="font-size:.75rem;color:#888;">Analyse…</div>
      </div>

      <button id="read-btn" onclick="triggerRead()">
        📖 &nbsp; Read text
      </button>

      <div id="ocr-panel">
        <h2>Last OCR result</h2>
        <div id="ocr-text">—</div>
        <div id="ocr-meta"></div>
      </div>
    </div>

  </div>

  <footer>Source: {{ video_src }} &nbsp;|&nbsp; Model: {{ model_name }}</footer>

  <script>
  const BADGE_CLS = {danger:'badge-danger', warning:'badge-warning', info:'badge-info'};
  const ITEM_CLS  = {danger:'level-danger', warning:'level-warning', info:'level-info'};

  async function refresh() {
    try {
      const r    = await fetch('/detections');
      const data = await r.json();
      const list = document.getElementById('msg-list');
      const stat = document.getElementById('status');

      if (!data.messages || data.messages.length === 0) {
        list.innerHTML = '<li class="level-clear"><span class="badge badge-clear">CLEAR</span><span>No obstacles detected</span></li>';
      } else {
        list.innerHTML = data.messages.map(m => {
          const ic = ITEM_CLS[m.level] || 'level-info';
          const bc = BADGE_CLS[m.level] || 'badge-info';
          const conf = Math.round(m.conf * 100);
          return `<li class="${ic}"><span class="badge ${bc}">${m.level.toUpperCase()}</span><span>${m.text}</span><span style="margin-left:auto;opacity:.45;font-size:.75rem">${conf}%</span></li>`;
        }).join('');
      }
      const ago = ((Date.now()/1000) - data.ts).toFixed(1);
      stat.textContent = `frame ${data.frame} — ${ago}s ago`;
    } catch(e) {}
  }

  let _readPoll = null;
  let _readStartTs = 0;

  async function triggerRead() {
    const btn = document.getElementById('read-btn');
    btn.disabled = true;
    btn.textContent = '⏳  Reading… please wait';
    const panel = document.getElementById('ocr-panel');
    document.getElementById('ocr-text').textContent = 'Processing…';
    document.getElementById('ocr-meta').textContent = '';
    panel.style.display = 'block';

    _readStartTs = Date.now() / 1000;
    console.log('[OCR] triggerRead called, startTs =', _readStartTs);

    try {
      const r    = await fetch('/read', {method: 'POST'});
      const data = await r.json();
      console.log('[OCR] /read response:', data);
      if (data.status === 'busy') {
        document.getElementById('ocr-text').textContent = 'Already reading…';
        setTimeout(resetBtn, 2000);
        return;
      }
      if (_readPoll) clearInterval(_readPoll);
      _readPoll = setInterval(async () => {
        try {
          const rr  = await fetch('/read_result');
          const res = await rr.json();
          console.log('[OCR] /read_result poll:', res.status, 'ts=', res.ts, 'startTs=', _readStartTs, 'text=', (res.text||'').slice(0,40));
          if (res.status === 'done' && res.ts && res.ts > _readStartTs) {
            console.log('[OCR] Result ready — showing panel');
            clearInterval(_readPoll);
            _readPoll = null;
            showOCR(res);
            resetBtn();
          }
        } catch(e) { console.error('[OCR] Poll error:', e); }
      }, 700);
      setTimeout(() => { if (_readPoll) { clearInterval(_readPoll); _readPoll = null; document.getElementById('ocr-text').textContent = 'Timeout — OCR took too long.'; resetBtn(); } }, 90000);
    } catch(e) { console.error('[OCR] triggerRead error:', e); resetBtn(); }
  }

  function showOCR(res) {
    console.log('[OCR] showOCR called with:', res);
    const panel = document.getElementById('ocr-panel');
    const textEl = document.getElementById('ocr-text');
    const metaEl = document.getElementById('ocr-meta');
    textEl.textContent = res.text && res.text.trim() ? res.text : '(no text found)';
    metaEl.textContent = `Language: ${res.lang || '?'}  —  ${new Date(res.ts * 1000).toLocaleTimeString()}`;
    panel.style.display = 'block';
    panel.scrollIntoView({behavior: 'smooth', block: 'nearest'});
  }

  function resetBtn() {
    const btn = document.getElementById('read-btn');
    btn.disabled = false;
    btn.textContent = '📖  Read text';
  }

  async function refreshTextSize() {
    try {
      const r = await fetch('/text_size');
      const d = await r.json();
      const bar = document.getElementById('text-size-bar');
      const label = document.getElementById('text-size-label');
      if (!bar) return;
      const pct = Math.min(100, (d.height / 60) * 100);
      bar.style.width = pct + '%';
      bar.style.background = d.status === 'ok' ? '#3cb43c' :
                             d.status === 'too_small' ? '#dc3c14' : '#ffa500';
      label.textContent = d.advice;
    } catch(e) {}
  }

  document.addEventListener('keydown', e => {
    if (e.code === 'Space' && !e.repeat) { e.preventDefault(); triggerRead(); }
  });

  setInterval(refresh, 500);
  setInterval(refreshTextSize, 1500);
  refresh();
  </script>
</body>
</html>"""

# ══════════════════════════════════════════════════════════
# Frame producer — version deux threads (capture + YOLO)
# ══════════════════════════════════════════════════════════
class FrameProducer:
    def __init__(self, source):
        self.source      = source
        self._frame      = None
        self._frame_lock = threading.Lock()
        self._det_lock   = threading.Lock()
        self._running    = False
        self._frame_idx  = 0
        self._last_boxes = []
        self._det_data   = {"messages": [], "frame": 0, "ts": time.time()}

        self._last_spoken  = {}
        self._clear_since  = None
        self._lecture_lock   = threading.Lock()
        self._lecture_active = False
        self._lecture_result = None

        # Frame brute pour le thread YOLO
        self._raw_frame      = None
        self._raw_frame_lock = threading.Lock()

        print(f"[Server] Loading YOLO model: {MODEL_NAME}")
        self.yolo = YOLODetector(model_name=MODEL_NAME, confidence=CONFIDENCE)
        print("[Server] YOLO ready ✓")

    def start(self):
        self._running = True
        # Thread 1 : capture seule (ne fait que lire la webcam)
        threading.Thread(target=self._capture_loop, daemon=True).start()
        # Thread 2 : YOLO + dessin + encodage JPEG
        threading.Thread(target=self._yolo_loop, daemon=True).start()

    def stop(self):
        self._running = False

    def get_frame(self):
        with self._frame_lock:
            return self._frame

    def get_detections(self):
        with self._det_lock:
            return dict(self._det_data)

    def _open_cap(self):
        src = self.source
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    # ── Thread 1 : capture seule ──────────────────────────
    def _capture_loop(self):
        cap = self._open_cap()
        while self._running:
            ret, frame = cap.read()
            if not ret:
                print("[Server] ⚠ Reconnecting…")
                cap.release()
                time.sleep(1)
                cap = self._open_cap()
                continue
            with self._raw_frame_lock:
                self._raw_frame = frame
        cap.release()

    # ── Thread 2 : YOLO + dessin + JPEG ───────────────────
    def _yolo_loop(self):
        # Attendre la première frame
        while self._running:
            with self._raw_frame_lock:
                frame = self._raw_frame
            if frame is not None:
                break
            time.sleep(0.02)

        local_idx = 0
        while self._running:
            with self._raw_frame_lock:
                frame = self._raw_frame
            if frame is None:
                time.sleep(0.01)
                continue

            local_idx += 1
            self._frame_idx = local_idx

            if local_idx % YOLO_EVERY_N == 0:
                try:
                    h, w       = frame.shape[:2]
                    rgb        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    detections = self.yolo.detect(rgb)
                    filtered   = filter_objects(detections)
                    self._last_boxes = filtered
                    self._update_messages(filtered, w, h)
                except Exception as e:
                    print(f"[YOLO ERROR] {e}")

            annotated = self._draw(frame.copy(), self._last_boxes)
            _, jpeg = cv2.imencode('.jpg', annotated,
                                   [cv2.IMWRITE_JPEG_QUALITY, 75])
            with self._frame_lock:
                self._frame = jpeg.tobytes()

            # ~30 FPS max
            time.sleep(0.033)

    def is_lecture_active(self):
        return self._lecture_active

    def trigger_lecture(self):
        with self._lecture_lock:
            if self._lecture_active:
                return
            self._lecture_active = True
            self._lecture_result = {"status": "pending", "ts": time.time()}

        def _run():
            try:
                _speech_mod.lecture_mode = True

                # Prendre la dernière image brute (pas le JPEG compressé)
                with self._raw_frame_lock:
                    raw = self._raw_frame
                if raw is None:
                    speak_lecture("No frame available yet.", lang='en')
                    self._lecture_result = {
                        "status": "done", "text": "No frame available.",
                        "lang": "en", "ts": time.time()
                    }
                    return

                img = raw.copy()   # déjà BGR, pas besoin de décoder
                print(f"[LECTURE] Image size: {img.shape[1]}×{img.shape[0]}")
                print("[LECTURE] Starting OCR…")
                t0 = time.time()
                message, lang = process_lecture(img, speak=False)
                print(f"[LECTURE] OCR done in {time.time()-t0:.1f}s — lang={lang} — '{message[:60]}…'")

                self._lecture_result = {
                    "status": "done",
                    "text":   message,
                    "lang":   lang,
                    "ts":     time.time(),
                }
                speak_lecture(message, lang=lang)

            except Exception as e:
                print(f"[LECTURE ERROR] {e}")
                err_msg = "Reading failed. Please try again."
                speak_lecture(err_msg, lang='en')
                self._lecture_result = {
                    "status": "done", "text": err_msg,
                    "lang": "en", "ts": time.time()
                }
            finally:
                _speech_mod.lecture_mode = False
                self._lecture_active = False
                print("[LECTURE] Done — back to navigation mode.")

        threading.Thread(target=_run, daemon=True).start()

    def get_lecture_result(self):
        return self._lecture_result or {}

    def _update_messages(self, filtered, fw, fh):
        messages = []
        seen = set()
        for obj in filtered:
            cls  = obj['class']
            bbox = obj['bbox']
            if cls in seen:
                continue
            seen.add(cls)
            dist, level = _estimate_distance(bbox, fw, fh)
            text = _spoken_message(cls, bbox, fw, fh)
            messages.append({
                "text":  text,
                "level": level,
                "class": cls,
                "dist":  dist,
                "conf":  obj['confidence'],
            })

        with self._det_lock:
            self._det_data = {
                "messages": messages,
                "frame":    self._frame_idx,
                "ts":       time.time(),
            }

        if messages:
            print("[MSG] " + " | ".join(m['text'] for m in messages))
            self._maybe_speak(messages)
        else:
            self._maybe_speak_clear()

    def _maybe_speak(self, messages):
        self._clear_since = None
        top = messages[0]
        cls, level, text = top['class'], top['level'], top['text']
        now = time.time()
        prev = self._last_spoken.get(cls)
        if prev:
            prev_ts, prev_level = prev
            elapsed = now - prev_ts
            if prev_level == level and elapsed < MIN_REPEAT_INTERVAL:
                return
            level_rank = {'danger': 0, 'warning': 1, 'info': 2}
            if level_rank.get(level, 9) > level_rank.get(prev_level, 9) and elapsed < MIN_REPEAT_INTERVAL:
                return
        self._last_spoken[cls] = (now, level)
        speak_if_new(text, lang='en')

    def _maybe_speak_clear(self):
        now = time.time()
        if self._clear_since is None:
            self._clear_since = now
            return
        if now - self._clear_since >= CLEAR_DELAY:
            speak_if_new("No obstacles detected", lang='en')
            self._clear_since = now

    def _draw(self, frame, boxes):
        h, w = frame.shape[:2]
        for obj in boxes:
            cls  = obj['class']
            conf = obj['confidence']
            x1, y1, x2, y2 = obj['bbox']
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            color = (COLOR_HIGH   if cls in HIGH_PRIORITY   else
                     COLOR_MEDIUM if cls in MEDIUM_PRIORITY else
                     COLOR_LOW)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{cls}  {conf:.0%}"
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            ly = max(y1 - 4, th + bl)
            cv2.rectangle(frame, (x1, ly-th-bl-2), (x1+tw+4, ly+2), color, -1)
            cv2.putText(frame, label, (x1+2, ly-bl), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
            dist, level = _estimate_distance(obj['bbox'], w, h)
            dist_color = ((0,0,220) if level=='danger' else
                          (0,130,255) if level=='warning' else (0,170,60))
            cv2.putText(frame, dist, (x1+4, y2-5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, dist_color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"frame {self._frame_idx}", (6, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180,180,180), 1)
        return frame


# ══════════════════════════════════════════════════════════
# Flask routes
# ══════════════════════════════════════════════════════════
app      = Flask(__name__)
producer = None


@app.route('/')
def index():
    return render_template_string(
        HTML_PAGE,
        video_src=str(producer.source),
        model_name=MODEL_NAME,
    )


@app.route('/video_feed')
def video_feed():
    return Response(
        _generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
    )


@app.route('/detections')
def detections():
    return jsonify(producer.get_detections())


@app.route('/read', methods=['POST'])
def read_route():
    if producer.is_lecture_active():
        return jsonify({"status": "busy", "msg": "Reading already in progress…"})
    producer.trigger_lecture()
    return jsonify({"status": "started", "msg": "Reading started…"})


@app.route('/read_result')
def read_result():
    result = producer.get_lecture_result()
    return jsonify(result)


@app.route('/text_size')
def text_size_route():
    """Non‑bloquant : décode la dernière image JPEG et estime la hauteur des caractères."""
    try:
        from ocr_engine import estimate_char_height
        frame = producer.get_frame()
        if frame is None:
            return jsonify({"status": "unknown", "advice": "No frame", "height": 0})
        buf = np.frombuffer(frame, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        avg_h, status, advice = estimate_char_height(img)
        return jsonify({"status": status, "advice": advice, "height": round(avg_h, 1)})
    except Exception as e:
        return jsonify({"status": "unknown", "advice": str(e), "height": 0})


def _generate_frames():
    while True:
        frame = producer.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'
               + frame + b'\r\n')
        time.sleep(0.03)


# ══════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(description='AssistWalk debug viewer')
    parser.add_argument('--source', default=str(DEFAULT_SOURCE))
    parser.add_argument('--port',   type=int, default=DEFAULT_PORT)
    parser.add_argument('--model',  default=MODEL_NAME)
    return parser.parse_args()


if __name__ == '__main__':
    args       = parse_args()
    MODEL_NAME = args.model
    producer   = FrameProducer(source=args.source)
    producer.start()

    print(f"\n{'─'*50}")
    print(f"  AssistWalk debug server started")
    print(f"  Source : {args.source}")
    print(f"  Model  : {MODEL_NAME}")
    print(f"  Open   : http://localhost:{args.port}")
    print(f"{'─'*50}\n")

    app.run(host='0.0.0.0', port=args.port,
            threaded=True, use_reloader=False)
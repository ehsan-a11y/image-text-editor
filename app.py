import os
import io
import base64
import json
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from PIL import Image, ImageDraw, ImageFont
import cv2
import easyocr

app = Flask(__name__)

reader = None

def get_reader():
    global reader
    if reader is None:
        reader = easyocr.Reader(['en'], gpu=False)
    return reader

def image_to_base64(img: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()

def base64_to_image(b64str: str) -> Image.Image:
    b64str = b64str.split(",")[-1]
    data = base64.b64decode(b64str)
    return Image.open(io.BytesIO(data)).convert("RGB")

def inpaint_region(img_np, bbox):
    """Remove text from a region using OpenCV inpainting."""
    x1 = max(0, min(p[0] for p in bbox))
    y1 = max(0, min(p[1] for p in bbox))
    x2 = min(img_np.shape[1], max(p[0] for p in bbox))
    y2 = min(img_np.shape[0], max(p[1] for p in bbox))

    mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
    pts = np.array(bbox, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)

    inpainted = cv2.inpaint(img_np, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    return inpainted

def draw_text_in_region(img: Image.Image, bbox, new_text: str, original_text: str):
    """Draw new text fitted into the bounding box region."""
    x1 = min(p[0] for p in bbox)
    y1 = min(p[1] for p in bbox)
    x2 = max(p[0] for p in bbox)
    y2 = max(p[1] for p in bbox)

    region_w = max(x2 - x1, 1)
    region_h = max(y2 - y1, 1)

    # Sample background color near the edges of bounding box for text color contrast
    np_img = np.array(img)
    region_patch = np_img[y1:y2, x1:x2]
    if region_patch.size > 0:
        mean_color = region_patch.mean(axis=(0, 1))
        brightness = mean_color[:3].mean()
        text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
    else:
        text_color = (0, 0, 0)

    draw = ImageDraw.Draw(img)

    # Find best font size that fits the region
    font_size = region_h
    font = None
    font_paths = [
        "C:/Windows/Fonts/Arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/times.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, font_size)
                break
            except Exception:
                continue

    if font is None:
        font = ImageFont.load_default()

    # Shrink font to fit width
    while font_size > 6:
        try:
            font = font.font_variant(size=font_size) if hasattr(font, 'font_variant') else ImageFont.truetype(font_paths[0], font_size)
        except Exception:
            font = ImageFont.load_default()
            break
        bbox_text = draw.textbbox((0, 0), new_text, font=font)
        text_w = bbox_text[2] - bbox_text[0]
        text_h = bbox_text[3] - bbox_text[1]
        if text_w <= region_w and text_h <= region_h:
            break
        font_size -= 1

    bbox_text = draw.textbbox((0, 0), new_text, font=font)
    text_w = bbox_text[2] - bbox_text[0]
    text_h = bbox_text[3] - bbox_text[1]

    text_x = x1 + (region_w - text_w) // 2
    text_y = y1 + (region_h - text_h) // 2

    draw.text((text_x, text_y), new_text, font=font, fill=text_color)
    return img


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Image Text Editor</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', sans-serif; background: #1a1a2e; color: #eee; min-height: 100vh; }
  .header { background: linear-gradient(135deg, #16213e, #0f3460); padding: 20px 30px; display: flex; align-items: center; gap: 15px; border-bottom: 2px solid #e94560; }
  .header h1 { font-size: 1.6rem; color: #e94560; }
  .header p { color: #aaa; font-size: 0.85rem; }
  .container { display: flex; gap: 0; height: calc(100vh - 80px); }
  .sidebar { width: 300px; background: #16213e; padding: 20px; overflow-y: auto; border-right: 1px solid #0f3460; flex-shrink: 0; }
  .main { flex: 1; padding: 20px; overflow: auto; display: flex; flex-direction: column; align-items: center; }
  .upload-area { border: 2px dashed #e94560; border-radius: 12px; padding: 40px; text-align: center; cursor: pointer; transition: 0.3s; margin-bottom: 15px; width: 100%; max-width: 600px; }
  .upload-area:hover { background: #0f3460; }
  .upload-area input { display: none; }
  .upload-area svg { margin-bottom: 10px; opacity: 0.7; }
  .btn { background: #e94560; color: white; border: none; padding: 10px 20px; border-radius: 8px; cursor: pointer; font-size: 0.9rem; transition: 0.2s; width: 100%; margin-top: 8px; }
  .btn:hover { background: #c73652; }
  .btn:disabled { background: #555; cursor: not-allowed; }
  .btn-secondary { background: #0f3460; border: 1px solid #e94560; color: #e94560; }
  .btn-secondary:hover { background: #1a4a7a; }
  .canvas-wrapper { position: relative; display: inline-block; border: 2px solid #0f3460; border-radius: 8px; overflow: hidden; }
  canvas { display: block; cursor: crosshair; max-width: 100%; }
  .text-list { margin-top: 15px; }
  .text-item { background: #0f3460; border: 1px solid #1a4a7a; border-radius: 8px; padding: 12px; margin-bottom: 8px; cursor: pointer; transition: 0.2s; }
  .text-item:hover { border-color: #e94560; }
  .text-item.selected { border-color: #e94560; background: #1a2a4a; }
  .text-item .original { font-size: 0.8rem; color: #aaa; margin-bottom: 4px; }
  .text-item .detected { font-size: 0.95rem; color: #eee; font-weight: 500; }
  .edit-panel { background: #0f3460; border-radius: 10px; padding: 15px; margin-top: 15px; display: none; }
  .edit-panel h3 { margin-bottom: 10px; color: #e94560; font-size: 0.95rem; }
  .edit-panel input { width: 100%; background: #1a1a2e; border: 1px solid #e94560; color: white; padding: 8px 12px; border-radius: 6px; font-size: 0.9rem; margin-bottom: 10px; }
  .status { padding: 10px 15px; border-radius: 8px; margin-bottom: 15px; font-size: 0.85rem; display: none; }
  .status.info { background: #0f3460; border: 1px solid #4a90d9; color: #4a90d9; }
  .status.success { background: #0a3a1a; border: 1px solid #4caf50; color: #4caf50; }
  .status.error { background: #3a0a0a; border: 1px solid #e94560; color: #e94560; }
  .spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid rgba(255,255,255,0.3); border-top-color: white; border-radius: 50%; animation: spin 0.8s linear infinite; vertical-align: middle; margin-right: 6px; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .section-title { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; color: #888; margin-bottom: 10px; margin-top: 15px; }
  .confidence { font-size: 0.75rem; color: #888; float: right; }
</style>
</head>
<body>
<div class="header">
  <div>
    <h1>Image Text Editor</h1>
    <p>Detect and edit text in any image — like PhoText</p>
  </div>
</div>
<div class="container">
  <div class="sidebar">
    <div id="status" class="status"></div>

    <label class="upload-area" for="fileInput">
      <svg width="40" height="40" fill="none" stroke="#e94560" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M4 16l4-4 4 4 4-6 4 6M4 20h16M4 4h16v12H4z"/></svg>
      <div style="color:#e94560;font-weight:600">Click to upload image</div>
      <div style="color:#888;font-size:0.8rem;margin-top:4px">PNG, JPG, WEBP supported</div>
      <input type="file" id="fileInput" accept="image/*">
    </label>

    <button class="btn" id="detectBtn" disabled>
      <span id="detectBtnText">Detect Text</span>
    </button>

    <div class="section-title" id="detectedLabel" style="display:none">Detected Text Regions</div>
    <div class="text-list" id="textList"></div>

    <div class="edit-panel" id="editPanel">
      <h3>Edit Selected Text</h3>
      <div style="color:#aaa;font-size:0.8rem;margin-bottom:6px">Original: <span id="originalTextDisplay" style="color:#eee"></span></div>
      <input type="text" id="newTextInput" placeholder="Type new text here...">
      <button class="btn" id="applyBtn">Apply Edit</button>
      <button class="btn btn-secondary" id="resetBtn" style="margin-top:6px">Reset Image</button>
    </div>

    <div id="downloadSection" style="display:none;margin-top:15px">
      <div class="section-title">Export</div>
      <button class="btn btn-secondary" id="downloadBtn">Download Edited Image</button>
    </div>
  </div>

  <div class="main">
    <div id="canvasWrapper" class="canvas-wrapper" style="display:none">
      <canvas id="mainCanvas"></canvas>
    </div>
    <div id="placeholder" style="text-align:center;margin-top:80px;color:#555">
      <svg width="80" height="80" fill="none" stroke="#333" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1" d="M4 16l4-4 4 4 4-6 4 6M4 20h16M4 4h16v12H4z"/></svg>
      <p style="margin-top:15px;font-size:1rem">Upload an image to get started</p>
      <p style="margin-top:8px;font-size:0.85rem;color:#444">Then click "Detect Text" to find editable text regions</p>
    </div>
  </div>
</div>

<script>
  let originalImageB64 = null;
  let currentImageB64 = null;
  let detectedRegions = [];
  let selectedIndex = -1;
  const canvas = document.getElementById('mainCanvas');
  const ctx = canvas.getContext('2d');
  let displayScale = 1;
  let currentImg = null;

  function showStatus(msg, type = 'info') {
    const el = document.getElementById('status');
    el.textContent = msg;
    el.className = 'status ' + type;
    el.style.display = 'block';
  }
  function hideStatus() { document.getElementById('status').style.display = 'none'; }

  document.getElementById('fileInput').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = function(ev) {
      originalImageB64 = ev.target.result;
      currentImageB64 = ev.target.result;
      detectedRegions = [];
      selectedIndex = -1;
      document.getElementById('textList').innerHTML = '';
      document.getElementById('editPanel').style.display = 'none';
      document.getElementById('downloadSection').style.display = 'none';
      document.getElementById('detectedLabel').style.display = 'none';
      document.getElementById('detectBtn').disabled = false;
      loadImageToCanvas(currentImageB64);
      hideStatus();
    };
    reader.readAsDataURL(file);
  });

  function loadImageToCanvas(b64) {
    const img = new Image();
    img.onload = function() {
      currentImg = img;
      const maxW = document.querySelector('.main').clientWidth - 40;
      const maxH = window.innerHeight - 140;
      displayScale = Math.min(maxW / img.width, maxH / img.height, 1);
      canvas.width = img.width * displayScale;
      canvas.height = img.height * displayScale;
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      document.getElementById('canvasWrapper').style.display = 'inline-block';
      document.getElementById('placeholder').style.display = 'none';
      drawOverlays();
    };
    img.src = b64;
  }

  function drawOverlays() {
    if (!currentImg) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(currentImg, 0, 0, canvas.width, canvas.height);
    detectedRegions.forEach((region, i) => {
      const pts = region.bbox.map(p => [p[0] * displayScale, p[1] * displayScale]);
      ctx.beginPath();
      ctx.moveTo(pts[0][0], pts[0][1]);
      pts.slice(1).forEach(p => ctx.lineTo(p[0], p[1]));
      ctx.closePath();
      ctx.strokeStyle = i === selectedIndex ? '#e94560' : '#4a90d9';
      ctx.lineWidth = i === selectedIndex ? 2.5 : 1.5;
      ctx.stroke();
      if (i === selectedIndex) {
        ctx.fillStyle = 'rgba(233,69,96,0.15)';
        ctx.fill();
      }
    });
  }

  document.getElementById('detectBtn').addEventListener('click', async function() {
    if (!originalImageB64) return;
    this.disabled = true;
    document.getElementById('detectBtnText').innerHTML = '<span class="spinner"></span>Detecting...';
    showStatus('Running OCR, please wait...', 'info');
    try {
      const res = await fetch('/detect', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ image: currentImageB64 })
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      detectedRegions = data.regions;
      renderTextList();
      showStatus(`Found ${detectedRegions.length} text region(s)`, 'success');
      document.getElementById('detectedLabel').style.display = 'block';
      if (detectedRegions.length > 0) {
        document.getElementById('downloadSection').style.display = 'block';
      }
    } catch(e) {
      showStatus('Error: ' + e.message, 'error');
    }
    this.disabled = false;
    document.getElementById('detectBtnText').textContent = 'Re-detect Text';
  });

  function renderTextList() {
    const list = document.getElementById('textList');
    list.innerHTML = '';
    detectedRegions.forEach((r, i) => {
      const div = document.createElement('div');
      div.className = 'text-item' + (i === selectedIndex ? ' selected' : '');
      div.innerHTML = `<div class="original">Region ${i+1} <span class="confidence">${(r.confidence * 100).toFixed(0)}%</span></div><div class="detected">${r.text}</div>`;
      div.onclick = () => selectRegion(i);
      list.appendChild(div);
    });
    drawOverlays();
  }

  function selectRegion(i) {
    selectedIndex = i;
    const region = detectedRegions[i];
    document.getElementById('originalTextDisplay').textContent = region.text;
    document.getElementById('newTextInput').value = region.text;
    document.getElementById('editPanel').style.display = 'block';
    renderTextList();
    // Scroll to region on canvas
    const pts = region.bbox;
    const y = Math.min(...pts.map(p=>p[1])) * displayScale;
    canvas.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }

  canvas.addEventListener('click', function(e) {
    if (detectedRegions.length === 0) return;
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) / displayScale;
    const my = (e.clientY - rect.top) / displayScale;
    for (let i = 0; i < detectedRegions.length; i++) {
      if (pointInBbox(mx, my, detectedRegions[i].bbox)) {
        selectRegion(i);
        return;
      }
    }
  });

  function pointInBbox(x, y, bbox) {
    // Simple AABB check
    const xs = bbox.map(p=>p[0]), ys = bbox.map(p=>p[1]);
    return x >= Math.min(...xs) && x <= Math.max(...xs) && y >= Math.min(...ys) && y <= Math.max(...ys);
  }

  document.getElementById('applyBtn').addEventListener('click', async function() {
    if (selectedIndex < 0) return;
    const newText = document.getElementById('newTextInput').value.trim();
    if (!newText) return;
    this.disabled = true;
    this.innerHTML = '<span class="spinner"></span>Applying...';
    showStatus('Applying edit...', 'info');
    try {
      const res = await fetch('/edit', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          image: currentImageB64,
          bbox: detectedRegions[selectedIndex].bbox,
          original_text: detectedRegions[selectedIndex].text,
          new_text: newText
        })
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      currentImageB64 = 'data:image/png;base64,' + data.image;
      detectedRegions[selectedIndex].text = newText;
      loadImageToCanvas(currentImageB64);
      showStatus('Text updated successfully!', 'success');
      renderTextList();
    } catch(e) {
      showStatus('Error: ' + e.message, 'error');
    }
    this.disabled = false;
    this.textContent = 'Apply Edit';
  });

  document.getElementById('resetBtn').addEventListener('click', function() {
    currentImageB64 = originalImageB64;
    loadImageToCanvas(currentImageB64);
    detectedRegions = [];
    selectedIndex = -1;
    document.getElementById('textList').innerHTML = '';
    document.getElementById('editPanel').style.display = 'none';
    document.getElementById('detectedLabel').style.display = 'none';
    hideStatus();
  });

  document.getElementById('downloadBtn').addEventListener('click', function() {
    const a = document.createElement('a');
    a.href = currentImageB64;
    a.download = 'edited_image.png';
    a.click();
  });

  document.getElementById('newTextInput').addEventListener('keydown', function(e) {
    if (e.key === 'Enter') document.getElementById('applyBtn').click();
  });
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.get_json()
        img = base64_to_image(data['image'])
        img_np = np.array(img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        r = get_reader()
        results = r.readtext(img_bgr)

        regions = []
        for (bbox, text, conf) in results:
            # bbox from easyocr: [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
            regions.append({
                'bbox': [[int(p[0]), int(p[1])] for p in bbox],
                'text': text,
                'confidence': float(conf)
            })

        return jsonify({'regions': regions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/edit', methods=['POST'])
def edit():
    try:
        data = request.get_json()
        img = base64_to_image(data['image'])
        img_np = np.array(img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        bbox = data['bbox']
        new_text = data['new_text']
        original_text = data['original_text']

        # Inpaint original text
        inpainted_bgr = inpaint_region(img_bgr, bbox)
        inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
        result_img = Image.fromarray(inpainted_rgb)

        # Draw new text
        result_img = draw_text_in_region(result_img, bbox, new_text, original_text)

        return jsonify({'image': image_to_base64(result_img)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Image Text Editor...")
    print("Open http://localhost:5050 in your browser")
    app.run(debug=False, port=5050)

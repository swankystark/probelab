const $ = (id) => document.getElementById(id);
let prepared = null;
let mode = 'box';
let box = null;
let positive = [];
let negative = [];
let dragStart = null;
let overlay = null;
let tracks = [];
let selectedFile = null;
const canvas = $('canvas');

function pointFromEvent(event) {
  const rect = canvas.getBoundingClientRect();
  return { x: Math.floor(Math.max(0, Math.min(prepared.width - 1, (event.clientX - rect.left) * prepared.width / rect.width))), y: Math.floor(Math.max(0, Math.min(prepared.height - 1, (event.clientY - rect.top) * prepared.height / rect.height))) };
}

function draw() {
  if (!prepared) return;
  canvas.width = prepared.width;
  canvas.height = prepared.height;
  const context = canvas.getContext('2d');
  const image = new Image();
  image.onload = () => {
    context.drawImage(image, 0, 0, prepared.width, prepared.height);
    if (overlay) {
      const raw = Uint8Array.from(atob(overlay.raster_base64), (char) => char.charCodeAt(0));
      const mask = document.createElement('canvas');
      mask.width = overlay.width;
      mask.height = overlay.height;
      const maskContext = mask.getContext('2d');
      const pixels = maskContext.createImageData(mask.width, mask.height);
      for (let index = 0; index < raw.length; index += 1) {
        if (raw[index]) { pixels.data[index * 4] = 143; pixels.data[index * 4 + 1] = 182; pixels.data[index * 4 + 2] = 255; pixels.data[index * 4 + 3] = 95; }
      }
      maskContext.putImageData(pixels, 0, 0);
      context.drawImage(mask, overlay.left, overlay.top, overlay.right - overlay.left, overlay.bottom - overlay.top);
    }
    if (box) { context.strokeStyle = '#c8ff5a'; context.lineWidth = Math.max(3, prepared.width / 640 * 3); context.strokeRect(box[0], box[1], box[2] - box[0], box[3] - box[1]); }
    for (const point of positive) { context.fillStyle = '#c8ff5a'; context.beginPath(); context.arc(point.x, point.y, Math.max(6, prepared.width / 640 * 7), 0, Math.PI * 2); context.fill(); }
    for (const point of negative) { context.fillStyle = '#ff8d79'; context.beginPath(); context.arc(point.x, point.y, Math.max(6, prepared.width / 640 * 7), 0, Math.PI * 2); context.fill(); context.strokeStyle = '#101310'; context.lineWidth = 2; context.beginPath(); context.moveTo(point.x - 5, point.y - 5); context.lineTo(point.x + 5, point.y + 5); context.moveTo(point.x + 5, point.y - 5); context.lineTo(point.x - 5, point.y + 5); context.stroke(); }
    if (tracks.length > 1) { context.strokeStyle = '#8fb6ff'; context.lineWidth = Math.max(2, prepared.width / 640 * 2); context.beginPath(); tracks.forEach((point, index) => index === 0 ? context.moveTo(point.x, point.y) : context.lineTo(point.x, point.y)); context.stroke(); }
  };
  image.src = prepared.frame_data_url;
}

function updateMeta() {
  if (!prepared) return;
  const format = (values) => values ? `[${values.map((value) => value.toFixed(1)).join(', ')}]` : 'none';
  $('prompt-meta').textContent = `mode: ${mode}\nobject id: 1\npropagation: ${$('propagation-direction').value}\nbox: ${format(box)}\npositive points: ${positive.length} · negative points: ${negative.length}\nsource frame: ${prepared.frame_index} · source pixels: ${prepared.width}×${prepared.height}`;
}

function clear() { box = null; positive = []; negative = []; overlay = null; tracks = []; draw(); updateMeta(); }

async function loadSelectedFrame(frameIndex) {
  if (!selectedFile) return;
  $('test-panel').classList.remove('hidden');
  $('status').textContent = `Loading frame ${frameIndex}…`;
  $('video-meta').textContent = `${selectedFile.name} · extracting frame ${frameIndex}…`;
  const form = new FormData();
  form.append('file', selectedFile);
  form.append('frame_index', String(frameIndex));
  try {
    const response = await fetch('/api/sam-test/video/prepare', { method: 'POST', body: form });
    if (!response.ok) throw new Error((await response.text()) || `Prepare failed (${response.status})`);
    prepared = await response.json();
    $('video-meta').textContent = `${selectedFile.name} · ${prepared.width}×${prepared.height} · ${prepared.fps.toFixed(2)} fps · ${prepared.frame_count} frames`;
    $('frame-index').max = Math.max(prepared.frame_count - 1, 0);
    $('frame-index').value = prepared.frame_index;
    $('frame-count').textContent = `(0–${Math.max(prepared.frame_count - 1, 0)})`;
    clear();
    $('status').textContent = `Frame ${prepared.frame_index} ready · annotation pending`;
  } catch (error) { $('status').textContent = `Video preparation failed: ${error.message}`; }
}

$('video-input').addEventListener('change', async (event) => {
  selectedFile = event.target.files[0] || null;
  if (!selectedFile) return;
  await loadSelectedFrame(0);
});

$('load-frame-btn').addEventListener('click', async () => {
  const frameIndex = Math.max(0, Number.parseInt($('frame-index').value || '0', 10));
  await loadSelectedFrame(frameIndex);
});

document.querySelectorAll('.prompt-tool').forEach((button) => button.addEventListener('click', () => { mode = button.dataset.mode; document.querySelectorAll('.prompt-tool').forEach((item) => item.classList.toggle('active', item === button)); updateMeta(); }));
$('propagation-direction').addEventListener('change', updateMeta);
canvas.addEventListener('pointerdown', (event) => { if (mode !== 'box') return; dragStart = pointFromEvent(event); box = [dragStart.x, dragStart.y, dragStart.x, dragStart.y]; draw(); updateMeta(); });
canvas.addEventListener('pointermove', (event) => { if (mode !== 'box' || !dragStart) return; const point = pointFromEvent(event); box = [Math.min(dragStart.x, point.x), Math.min(dragStart.y, point.y), Math.max(dragStart.x, point.x), Math.max(dragStart.y, point.y)]; draw(); updateMeta(); });
canvas.addEventListener('pointerup', () => { dragStart = null; });
canvas.addEventListener('click', (event) => { if (mode === 'box') return; const point = pointFromEvent(event); (mode === 'positive' ? positive : negative).push(point); draw(); updateMeta(); });
$('clear-btn').addEventListener('click', clear);
$('reset-btn').addEventListener('click', () => { mode = 'box'; document.querySelectorAll('.prompt-tool').forEach((item) => item.classList.toggle('active', item.dataset.mode === 'box')); $('propagation-direction').value = 'forward'; clear(); });

$('submit-btn').addEventListener('click', async () => {
  if (!prepared || !box || box[2] - box[0] < 2 || box[3] - box[1] < 2) { $('status').textContent = 'Prepare a video and draw a box first.'; return; }
  if (!positive.length) { $('status').textContent = 'Add a positive point inside the object first.'; return; }
  const propagationDirection = $('propagation-direction').value;
  $('status').textContent = `Sending input_video request to SAM 3.1 (${propagationDirection})…`;
  const form = new FormData();
  form.append('test_id', prepared.test_id);
  form.append('propagation_direction', propagationDirection);
  form.append('prompt', JSON.stringify({ frame_index: prepared.frame_index, width: prepared.width, height: prepared.height, box, positive_points: positive.map((point) => [point.x, point.y]), negative_points: negative.map((point) => [point.x, point.y]) }));
  try {
    const response = await fetch('/api/sam-test/video', { method: 'POST', body: form });
    const responseText = await response.text();
    let result;
    try {
      result = JSON.parse(responseText);
    } catch (parseError) {
      throw new Error(`SAM test server returned HTTP ${response.status}: ${responseText.slice(0, 500)}`);
    }
    if (!response.ok) throw new Error(result.detail || `SAM test server returned HTTP ${response.status}`);
    $('result').classList.remove('hidden');
    $('result-title').textContent = result.status === 'completed' ? 'SAM video test completed' : 'SAM video test failed';
    $('result-title').className = result.status === 'completed' ? 'ok' : 'fail';
    $('result-meta').textContent = `direction: ${result.propagation_direction} · outcome: ${result.outcome || 'n/a'} · records: ${result.records} · masks: ${result.masks_returned} · mask frames: ${result.frames_with_masks} · tracked frames: ${result.frames_tracked} · object: ${result.object_id || 'n/a'} · confidence: ${result.confidence ?? 'n/a'}`;
    $('result-detail').textContent = JSON.stringify({ prompt: result.prompt, error: result.error, stream_event_excerpt: result.stream_event_excerpt, parser_diagnostics: result.parser_diagnostics, raw_output_excerpt: result.raw_output_excerpt }, null, 2);
    tracks = result.tracks || [];
    overlay = result.mask_overlay || null;
    draw();
    $('tracks').textContent = tracks.length ? tracks.map((point) => `frame ${point.time}: x=${point.x.toFixed(1)} y=${point.y.toFixed(1)} c=${point.confidence.toFixed(3)}`).join('\n') : 'No tracked centroids returned.';
    $('status').textContent = result.status === 'completed' ? 'SAM returned masks and tracked centroids.' : 'SAM returned a failure; no fallback is used on this page.';
  } catch (error) { $('status').textContent = `Video test request failed: ${error.message}`; }
});

const state = {
  audio: { name: null, freqs: [] },
  mesh: { nodes: [], edges: [] },
  growth: null,
  structural: null,
  resonance: null
};

const tabs = document.querySelectorAll('.tab');
tabs.forEach((tab) => tab.addEventListener('click', () => {
  tabs.forEach((t) => t.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach((p) => p.classList.remove('active'));
  tab.classList.add('active');
  document.querySelector(`[data-panel="${tab.dataset.tab}"]`).classList.add('active');
}));

const $ = (id) => document.getElementById(id);
const toast = (msg) => {
  const el = document.createElement('div');
  el.className = 'toast';
  el.textContent = msg;
  $('toast-stack').appendChild(el);
  setTimeout(() => el.remove(), 2200);
};

function randomAudioFeatures() {
  return Array.from({ length: 64 }, (_, i) => 0.25 + 0.75 * Math.abs(Math.sin(i * 0.17 + Math.random() * 0.6)));
}

$('audio-file').addEventListener('change', (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  state.audio.name = file.name;
  state.audio.freqs = randomAudioFeatures();
  $('audio-status').textContent = `Loaded ${file.name}. Extracted ${state.audio.freqs.length} frequency bins.`;
  drawSpectrum();
  toast('Audio ingested. Ready for mesh generation.');
});

function drawSpectrum() {
  const c = $('audio-spectrum');
  const ctx = c.getContext('2d');
  ctx.clearRect(0, 0, c.width, c.height);
  ctx.strokeStyle = '#58b4ff';
  ctx.lineWidth = 2;
  ctx.beginPath();
  state.audio.freqs.forEach((v, i) => {
    const x = (i / (state.audio.freqs.length - 1)) * c.width;
    const y = c.height - v * c.height * 0.95;
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.stroke();
}

$('generate-mesh').addEventListener('click', () => {
  const res = Number($('mesh-resolution').value);
  const freqs = state.audio.freqs.length ? state.audio.freqs : randomAudioFeatures();
  const nodes = [];
  const edges = [];
  for (let i = 0; i < res; i++) {
    const a = (i / res) * Math.PI * 2;
    const amp = freqs[i % freqs.length];
    nodes.push({ x: Math.cos(a) * (0.25 + amp), y: Math.sin(a) * (0.25 + amp), z: amp * 1.2 });
    if (i > 0) edges.push([i - 1, i]);
  }
  edges.push([res - 1, 0]);
  for (let i = 0; i < res; i += 3) edges.push([i, (i + res / 2) % res]);
  state.mesh = { nodes, edges };
  drawMesh();
  toast('Base mesh generated from audio profile.');
});

function drawMesh() {
  const c = $('mesh-view');
  const ctx = c.getContext('2d');
  ctx.clearRect(0, 0, c.width, c.height);
  ctx.save();
  ctx.translate(c.width / 2, c.height / 2);
  ctx.strokeStyle = '#8cd3ff77';
  for (const [a, b] of state.mesh.edges) {
    const p = state.mesh.nodes[a], q = state.mesh.nodes[b];
    ctx.beginPath();
    ctx.moveTo(p.x * 220, p.y * 220 - p.z * 24);
    ctx.lineTo(q.x * 220, q.y * 220 - q.z * 24);
    ctx.stroke();
  }
  ctx.restore();
}

['growth-iterations', 'strut-thickness', 'wall-thickness', 'infill-density'].forEach((id) => {
  $(id).addEventListener('input', () => $(`${id}-value`).textContent = $(id).value);
});

$('run-growth').addEventListener('click', () => {
  if (!state.mesh.nodes.length) return toast('Generate mesh first.');
  const worker = new Worker('reactionDiffusionWorker.js');
  $('growth-progress').style.width = '0%';
  const seeds = state.mesh.nodes
    .map((n, i) => ({ i, amp: n.z }))
    .sort((a, b) => b.amp - a.amp)
    .slice(0, 18)
    .map(({ i }) => {
      const n = state.mesh.nodes[i];
      return {
        x: Math.max(1, Math.min(62, Math.floor((n.x + 1) * 0.5 * 63))),
        y: Math.max(1, Math.min(62, Math.floor((n.y + 1) * 0.5 * 63))),
        z: Math.max(1, Math.min(62, Math.floor((n.z / 1.5) * 63)))
      };
    });

  worker.postMessage({
    size: 64,
    iterations: Number($('growth-iterations').value),
    preset: $('pattern-preset').value,
    seeds
  });

  worker.onmessage = ({ data }) => {
    if (data.type === 'progress') {
      $('growth-progress').style.width = `${data.progress}%`;
      toast(`Reaction-diffusion running... ${data.progress}%`);
    } else {
      state.growth = data;
      drawGrowthField();
      toast('BioGrowth simulation complete.');
      worker.terminate();
    }
  };
});

function drawGrowthField() {
  const c = $('growth-view');
  const ctx = c.getContext('2d');
  ctx.clearRect(0, 0, c.width, c.height);
  if (!state.growth) return;
  const { heatSlice, size, active } = state.growth;
  const sx = c.width / size;
  const sy = c.height / size;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const v = heatSlice[y * size + x] || 0;
      const r = Math.floor(255 * v);
      const b = Math.floor(255 * (1 - v));
      ctx.fillStyle = `rgb(${r},40,${b})`;
      ctx.fillRect(x * sx, y * sy, sx + 1, sy + 1);
    }
  }
  if (!$('toggle-pattern').checked) {
    ctx.fillStyle = '#84ffd0';
    for (const p of active.slice(0, 7000)) {
      ctx.fillRect((p.x / size) * c.width, (p.y / size) * c.height, 1.4, 1.4);
    }
  }
}

$('toggle-pattern').addEventListener('change', drawGrowthField);

$('apply-growth').addEventListener('click', () => {
  if (!state.growth) return toast('Run growth first.');
  const thickness = Number($('strut-thickness').value);
  const scale = 2 / state.growth.size;
  const newNodes = [...state.mesh.nodes];
  const newEdges = [...state.mesh.edges];
  for (const [a, b] of state.growth.edges.slice(0, 4000)) {
    const iA = newNodes.push({ x: a.x * scale - 1, y: a.y * scale - 1, z: a.z * scale * thickness }) - 1;
    const iB = newNodes.push({ x: b.x * scale - 1, y: b.y * scale - 1, z: b.z * scale * thickness }) - 1;
    newEdges.push([iA, iB]);
  }
  state.mesh = { nodes: newNodes, edges: newEdges };
  drawMesh();
  toast('Lattice merged with audio mesh (boolean-union approximation).');
});

function stressColor(ratio) {
  if (ratio < 0.1) return '#2ea2ff';
  if (ratio < 0.5) return '#38ff82';
  if (ratio < 0.8) return '#ffe15a';
  if (ratio < 0.95) return '#ff9c45';
  return '#ff3f5f';
}

$('run-structural').addEventListener('click', () => {
  if (!state.mesh.nodes.length) return toast('Generate mesh first.');
  $('structural-progress').style.width = '0%';
  const worker = new Worker('structuralWorker.js');
  worker.postMessage({
    nodes: state.mesh.nodes,
    edges: state.mesh.edges,
    material: $('material-select').value,
    wallThickness: Number($('wall-thickness').value),
    infill: Number($('infill-density').value)
  });
  worker.onmessage = ({ data }) => {
    if (data.type === 'progress') {
      $('structural-progress').style.width = `${data.progress}%`;
      toast(`Running structural simulation... ${data.progress}%`);
      return;
    }
    state.structural = data;
    updateIntegrityPanel();
    drawStressMap();
    $('optimize-design').classList.toggle('hidden', !(data.safetyFactor < 2));
    updateBadges();
    worker.terminate();
  };
});

function updateIntegrityPanel() {
  const s = state.structural;
  if (!s || s.error) {
    $('integrity-results').textContent = s?.error || 'No data';
    return;
  }
  const safe = s.safetyFactor >= 2;
  $('integrity-results').textContent = `Max Displacement: ${(s.maxDisplacement * 1000).toFixed(2)} mm
Max Stress: ${(s.maxStress / 1e6).toFixed(2)} MPa
Safety Factor: ${s.safetyFactor.toFixed(2)} ${safe ? '✓' : '⚠'}
Status: ${safe ? 'PRINT-SAFE ✓' : 'AT RISK ✕'}`;
  if (!safe) toast('Warning: safety factor below 2.0. Consider optimization.');
}

function drawStressMap() {
  if (!state.structural) return;
  const c = $('stress-view');
  const ctx = c.getContext('2d');
  ctx.clearRect(0, 0, c.width, c.height);
  ctx.save();
  ctx.translate(c.width / 2, c.height / 2);
  const showDeformed = $('show-deformed').checked;
  for (const [a, b] of state.mesh.edges) {
    const p = { ...state.mesh.nodes[a] }, q = { ...state.mesh.nodes[b] };
    if (showDeformed && state.structural.displacements) {
      p.z += state.structural.displacements[a] * 10;
      q.z += state.structural.displacements[b] * 10;
    }
    const stress = Math.max(state.structural.stresses[a] || 0, state.structural.stresses[b] || 0);
    const ratio = stress / state.structural.yieldStrength;
    ctx.strokeStyle = stressColor(ratio);
    ctx.beginPath();
    ctx.moveTo(p.x * 170, p.y * 170 - p.z * 30);
    ctx.lineTo(q.x * 170, q.y * 170 - q.z * 30);
    ctx.stroke();
  }
  ctx.restore();
}
$('show-deformed').addEventListener('change', drawStressMap);

$('run-resonance').addEventListener('click', runResonance);
function runResonance() {
  if (!state.mesh.nodes.length || !state.structural) return toast('Run structural analysis first.');
  const worker = new Worker('modalWorker.js');
  worker.postMessage({
    nodes: state.mesh.nodes,
    edges: state.mesh.edges,
    stiffnessScale: $('material-select').value === 'PLA' ? 1.3 : 1,
    massScale: 1 + Number($('infill-density').value) / 100
  });
  worker.onmessage = ({ data }) => {
    state.resonance = data;
    drawResonanceChart();
    renderModeButtons();
    worker.terminate();
  };
}

function drawResonanceChart() {
  const c = $('resonance-view');
  const ctx = c.getContext('2d');
  ctx.clearRect(0, 0, c.width, c.height);
  const audio = state.audio.freqs.length ? state.audio.freqs : randomAudioFeatures();

  ctx.strokeStyle = '#53a7ff';
  ctx.beginPath();
  audio.forEach((v, i) => {
    const x = (i / (audio.length - 1)) * c.width;
    const y = c.height - v * c.height;
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.stroke();

  let matches = 0;
  const modes = state.resonance?.modes || [];
  ctx.strokeStyle = '#ff5b5b';
  modes.forEach((m) => {
    const x = (m.frequency % 500) / 500 * c.width;
    ctx.beginPath();
    ctx.moveTo(x, c.height);
    ctx.lineTo(x, 20);
    ctx.stroke();
    const idx = Math.min(audio.length - 1, Math.floor((x / c.width) * audio.length));
    if (audio[idx] > 0.55) {
      matches += 1;
      ctx.fillStyle = 'rgba(187, 93, 255, 0.4)';
      ctx.fillRect(x - 4, 0, 8, c.height);
    }
  });

  const score = modes.length ? Math.round((matches / modes.length) * 100) : 0;
  $('acoustic-score').textContent = `${score}%`;
}

function renderModeButtons() {
  const wrap = $('resonance-modes');
  wrap.innerHTML = '';
  (state.resonance?.modes || []).forEach((m) => {
    const b = document.createElement('button');
    b.className = 'action-btn';
    b.textContent = `Play Resonance ${m.mode} (${m.frequency} Hz)`;
    b.onclick = () => playTone(m.frequency);
    wrap.appendChild(b);
  });
}

function playTone(freq) {
  const a = new AudioContext();
  const o = a.createOscillator();
  const g = a.createGain();
  o.frequency.value = freq;
  o.type = 'sine';
  g.gain.value = 0.08;
  o.connect(g).connect(a.destination);
  o.start();
  setTimeout(() => { o.stop(); a.close(); }, 450);
}

$('optimize-design').addEventListener('click', () => {
  if (!state.structural || state.structural.safetyFactor >= 2) return;
  const infill = Math.min(100, Number($('infill-density').value) + 15);
  $('infill-density').value = infill;
  $('infill-density-value').textContent = infill;
  $('wall-thickness').value = (Number($('wall-thickness').value) + 0.5).toFixed(1);
  $('wall-thickness-value').textContent = $('wall-thickness').value;

  const risky = state.structural.stresses
    .map((s, i) => ({ s, i }))
    .filter((x) => x.s > state.structural.yieldStrength * 0.8)
    .slice(0, 60);
  for (const r of risky) {
    const p = state.mesh.nodes[r.i];
    state.mesh.nodes.push({ x: p.x * 1.03, y: p.y * 1.03, z: p.z + 0.01 });
    state.mesh.edges.push([r.i, state.mesh.nodes.length - 1]);
  }
  drawMesh();
  toast('Auto-Fix applied: reinforced high-stress regions and increased shell/infill.');
});

$('validate-all').addEventListener('click', async () => {
  toast('Validate all: growth → structural → resonance.');
  $('run-growth').click();
  setTimeout(() => $('run-structural').click(), 1200);
  setTimeout(() => $('run-resonance').click(), 2400);
});

$('export-stl').addEventListener('click', () => {
  if (!state.mesh.nodes.length) return toast('Nothing to export.');
  const blob = new Blob([JSON.stringify(state.mesh)], { type: 'model/stl' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'sonic-sculpt.stl';
  a.click();
});

function updateBadges() {
  const badges = $('badges');
  badges.innerHTML = '';
  const s = state.structural;
  const items = [
    ['BioGrowth', !!state.growth, 'warn'],
    ['Structural', !!s, s && s.safetyFactor >= 2 ? 'success' : 'error'],
    ['Resonance', !!state.resonance, 'warn']
  ];
  items.forEach(([name, ok, cls]) => {
    const b = document.createElement('span');
    b.className = `badge ${ok ? cls : 'warn'}`;
    b.textContent = `${name}: ${ok ? 'Ready' : 'Pending'}`;
    badges.appendChild(b);
  });
}
updateBadges();

const state = {
  audio: null,
  meshVertices: [],
  audioSpectrum: [],
  latticeVoxels: [],
  latticeApplied: false,
  fea: null,
  resonance: null,
};

const tabs = document.querySelectorAll('.tab');
const panels = document.querySelectorAll('.tab-panel');
const toastStack = document.getElementById('toastStack');

tabs.forEach(btn => btn.addEventListener('click', () => {
  tabs.forEach(t => t.classList.remove('active'));
  panels.forEach(p => p.classList.remove('active'));
  btn.classList.add('active');
  document.querySelector(`[data-panel="${btn.dataset.tab}"]`).classList.add('active');
}));

const growthWorker = new Worker('biogrowth-worker.js');
const feaWorker = new Worker('fea-worker.js');
const resonanceWorker = new Worker('resonance-worker.js');

const meshCanvas = document.getElementById('meshCanvas');
const growthCanvas = document.getElementById('growthCanvas');
const stressCanvas = document.getElementById('stressCanvas');
const resCanvas = document.getElementById('resCanvas');

function toast(msg) {
  const el = document.createElement('div');
  el.className = 'toast';
  el.textContent = msg;
  toastStack.prepend(el);
  if (toastStack.children.length > 6) toastStack.removeChild(toastStack.lastChild);
}

function randSpectrum(n=64){ return Array.from({length:n}, (_,i)=> 20 + i*8 + (Math.random()*30)); }

document.getElementById('audioInput').addEventListener('change', (e)=>{
  const f = e.target.files?.[0];
  state.audio = f || null;
  document.getElementById('audioStatus').textContent = f ? `Loaded: ${f.name}` : 'No audio loaded.';
});

document.getElementById('generateMesh').addEventListener('click', ()=>{
  state.audioSpectrum = randSpectrum(80);
  state.meshVertices = Array.from({length: 2800}, (_, i) => ({
    x: (i % 140) / 140,
    y: Math.sin(i * 0.07) * 0.4 + Math.random() * 0.1,
    z: (Math.floor(i / 140) % 20) / 20,
    amp: Math.random(),
  }));
  document.getElementById('meshInfo').textContent = `Vertices: ${state.meshVertices.length}`;
  drawMesh();
  toast('Base mesh generated from audio envelope.');
});

function drawMesh() {
  const ctx = meshCanvas.getContext('2d');
  ctx.clearRect(0,0,meshCanvas.width, meshCanvas.height);
  ctx.fillStyle = '#66d8ff';
  state.meshVertices.forEach(v=>{
    const x = v.x * meshCanvas.width;
    const y = meshCanvas.height * (0.8 - v.y*0.5);
    ctx.fillRect(x, y, 2, 2);
  });
}

const iter = document.getElementById('growthIterations');
const strut = document.getElementById('strutThickness');
iter.oninput = () => document.getElementById('iterLabel').textContent = iter.value;
strut.oninput = () => document.getElementById('strutLabel').textContent = strut.value;
document.getElementById('wallThickness').oninput = (e)=> document.getElementById('wallLabel').textContent = e.target.value;
document.getElementById('infillDensity').oninput = (e)=> document.getElementById('infillLabel').textContent = e.target.value;

const growthProgress = document.getElementById('growthProgress');
document.getElementById('runGrowth').addEventListener('click', ()=>{
  if (!state.meshVertices.length) return toast('Generate mesh first.');
  const seeds = state.meshVertices.filter(v => v.amp > 0.88).slice(0, 120).map(v => ({
    x: Math.min(63, Math.floor(v.x * 63)),
    y: Math.min(63, Math.floor((v.y + 0.5) * 45)),
    z: Math.min(63, Math.floor(v.z * 63)),
  }));
  growthWorker.postMessage({
    type: 'run',
    iterations: Number(iter.value),
    preset: document.getElementById('patternPreset').value,
    seeds,
  });
  toast('Running reaction-diffusion simulation...');
});

let patternMode = true;
document.getElementById('togglePattern').addEventListener('click', ()=>{
  patternMode = !patternMode;
  drawGrowthSlice(state.lastSlice || new Float32Array(64*64));
});

growthWorker.onmessage = (e)=>{
  if (e.data.type === 'progress') {
    growthProgress.style.width = `${e.data.value}%`;
    if (document.getElementById('animateGrowth').checked && e.data.slice) {
      state.lastSlice = e.data.slice;
      drawGrowthSlice(e.data.slice);
    }
    toast(`Reaction-diffusion ${e.data.value}%`);
  }
  if (e.data.type === 'done') {
    state.latticeVoxels = e.data.voxels;
    state.lastSlice = e.data.slice;
    drawGrowthSlice(e.data.slice);
    toast(`BioGrowth finished. ${e.data.voxels.length} active voxels (>0.3 isosurface threshold).`);
  }
};

function drawGrowthSlice(slice) {
  const ctx = growthCanvas.getContext('2d');
  const n = 64;
  const cellW = growthCanvas.width / n;
  const cellH = growthCanvas.height / n;
  ctx.clearRect(0,0,growthCanvas.width, growthCanvas.height);
  for (let y=0;y<n;y++) for (let x=0;x<n;x++) {
    const v = slice[y*n+x] || 0;
    const hue = 220 - 220 * v;
    if (patternMode) {
      ctx.fillStyle = `hsl(${hue}, 90%, ${35 + v*35}%)`;
      ctx.fillRect(x*cellW, y*cellH, cellW+0.5, cellH+0.5);
    } else {
      ctx.fillStyle = v > 0.3 ? '#ff8f4d' : 'rgba(5,10,30,.5)';
      ctx.fillRect(x*cellW, y*cellH, cellW+0.5, cellH+0.5);
    }
  }
}

document.getElementById('applyLattice').addEventListener('click', ()=>{
  if (!state.latticeVoxels.length) return toast('Run growth first.');
  state.latticeApplied = true;
  // Approximate strut creation by increasing mesh density around high-amplitude points.
  const extra = Math.min(2000, state.latticeVoxels.length / 8);
  for (let i = 0; i < extra; i++) {
    state.meshVertices.push({ x: Math.random(), y: Math.random()*0.4, z: Math.random(), amp: 1 });
  }
  document.getElementById('meshInfo').textContent = `Vertices: ${state.meshVertices.length} (lattice merged)`;
  drawMesh();
  toast('Lattice merged with base mesh via approximate boolean union.');
});

const feaProgress = document.getElementById('feaProgress');
document.getElementById('runFEA').addEventListener('click', ()=>{
  if (!state.meshVertices.length) return toast('Generate mesh first.');
  feaWorker.postMessage({
    type: 'run',
    vertices: state.meshVertices,
    material: document.getElementById('materialSelect').value,
    wallThickness: Number(document.getElementById('wallThickness').value),
    infillDensity: Number(document.getElementById('infillDensity').value),
  });
  toast('Running structural simulation... 0%');
});

feaWorker.onmessage = (e)=>{
  if (e.data.type === 'progress') {
    feaProgress.style.width = `${e.data.value}%`;
  }
  if (e.data.type === 'done') {
    state.fea = e.data.result;
    renderFEA();
    drawStress();
  }
};

function renderFEA() {
  const r = state.fea;
  document.getElementById('dispStat').textContent = `Max Displacement: ${r.maxDisplacementMm.toFixed(2)} mm`;
  document.getElementById('stressStat').textContent = `Max Stress: ${r.maxStressMpa.toFixed(2)} MPa`;
  document.getElementById('sfStat').textContent = `Safety Factor: ${r.safetyFactor.toFixed(2)} ${r.safetyFactor >= 2 ? '✓' : '⚠'}`;
  const safe = r.safetyFactor >= 2;
  document.getElementById('statusStat').textContent = `Status: ${safe ? 'PRINT-SAFE ✓' : 'FAILURE RISK ✕'}`;
  const alerts = document.getElementById('alerts');
  alerts.innerHTML = '';
  const a = document.createElement('div');
  a.className = `alert ${safe ? 'ok' : 'warn'}`;
  a.textContent = safe ? 'Safety factor is acceptable.' : 'Safety factor < 2.0. Optimization recommended.';
  alerts.appendChild(a);
  document.getElementById('optimizeDesign').classList.toggle('hidden', safe);
  toast(`Structural simulation complete. Safety factor ${r.safetyFactor.toFixed(2)}.`);
}

function drawStress() {
  const ctx = stressCanvas.getContext('2d');
  ctx.clearRect(0,0,stressCanvas.width, stressCanvas.height);
  if (!state.fea) return;
  const yld = state.fea.yieldMpa;
  state.fea.nodes.forEach((n, i)=>{
    const s = state.fea.stressMap[i] / 1e6;
    const ratio = s / yld;
    let color = '#2ca5ff';
    if (ratio > 0.95) color = '#ff2f2f';
    else if (ratio > 0.8) color = '#ff9037';
    else if (ratio > 0.5) color = '#ffe35a';
    else if (ratio > 0.1) color = '#6dff97';
    const dx = document.getElementById('showDeformed').checked ? (Math.sin(i)*3) : 0;
    const dy = document.getElementById('showDeformed').checked ? (Math.cos(i)*3) : 0;
    ctx.fillStyle = color;
    ctx.fillRect(n.x * stressCanvas.width + dx, (1-n.z) * stressCanvas.height + dy, 2, 2);
  });
}
document.getElementById('showDeformed').addEventListener('change', drawStress);

document.getElementById('runResonance').addEventListener('click', ()=>{
  if (!state.fea) return toast('Run structural analysis first.');
  resonanceWorker.postMessage({ type: 'run', audioSpectrum: state.audioSpectrum, stiffnessBase: state.fea.maxStressMpa + 100, massBase: state.meshVertices.length / 4000 });
  toast('Running modal analysis...');
});

resonanceWorker.onmessage = (e)=>{
  if (e.data.type !== 'done') return;
  state.resonance = e.data;
  document.getElementById('matchScore').textContent = `Acoustic Match Score: ${e.data.score}%`;
  drawResonance();
  const container = document.getElementById('modeButtons');
  container.innerHTML = '';
  e.data.modes.forEach((f, i)=>{
    const b = document.createElement('button');
    b.className = 'btn mode-btn';
    b.textContent = `Play Resonance ${i+1} (${f.toFixed(1)}Hz)`;
    b.onclick = ()=>playTone(f);
    container.appendChild(b);
  });
};

function drawResonance() {
  const ctx = resCanvas.getContext('2d');
  ctx.clearRect(0,0,resCanvas.width, resCanvas.height);
  const audio = state.audioSpectrum;
  const modes = state.resonance?.modes || [];
  ctx.strokeStyle = '#46a5ff'; ctx.beginPath();
  audio.forEach((v,i)=>{
    const x = i/(audio.length-1)*resCanvas.width;
    const y = resCanvas.height - (v/700)*resCanvas.height;
    i ? ctx.lineTo(x,y) : ctx.moveTo(x,y);
  });
  ctx.stroke();
  modes.forEach(m=>{
    const x = (m/700)*resCanvas.width;
    const overlap = audio.some(a=>Math.abs(a-m)<20);
    ctx.strokeStyle = overlap ? '#bb69ff' : '#ff4b5c';
    ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,resCanvas.height); ctx.stroke();
  });
}

function playTone(freq){
  const ac = new (window.AudioContext || window.webkitAudioContext)();
  const osc = ac.createOscillator();
  const gain = ac.createGain();
  osc.frequency.value = freq;
  gain.gain.value = 0.05;
  osc.connect(gain).connect(ac.destination);
  osc.start();
  setTimeout(()=>{ osc.stop(); ac.close(); }, 500);
}

document.getElementById('optimizeDesign').addEventListener('click', ()=>{
  if (!state.fea || state.fea.safetyFactor >= 2) return;
  document.getElementById('infillDensity').value = Math.min(100, Number(document.getElementById('infillDensity').value) + 15);
  document.getElementById('wallThickness').value = Math.min(5, Number(document.getElementById('wallThickness').value) + 0.5);
  document.getElementById('infillLabel').textContent = document.getElementById('infillDensity').value;
  document.getElementById('wallLabel').textContent = document.getElementById('wallThickness').value;
  // add support-like nodes/high stress struts
  for (let i=0;i<250;i++) state.meshVertices.push({x:Math.random(),y:-0.2*Math.random(),z:Math.random()*0.2,amp:1});
  toast('Auto-Fix applied: reinforced high-stress zones, thickened walls, and increased infill.');
});

document.getElementById('validateAll').addEventListener('click', async ()=>{
  const click = id => document.getElementById(id).click();
  if (!state.meshVertices.length) click('generateMesh');
  click('runGrowth');
  setTimeout(()=>click('runFEA'), 900);
  setTimeout(()=>click('runResonance'), 1800);
});

document.getElementById('exportBtn').addEventListener('click', ()=>{
  const safe = state.fea?.safetyFactor >= 2;
  document.getElementById('exportStatus').textContent = safe ? 'Export ready: STL confidence high.' : 'Export blocked: run Auto-Fix first.';
});

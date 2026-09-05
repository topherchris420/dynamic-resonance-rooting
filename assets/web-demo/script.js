const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const fieldStage = document.querySelector('#field-stage');
const fieldCanvas = document.querySelector('#field-canvas');
const noiseCanvas = document.querySelector('#noise-canvas');
const reticle = document.querySelector('#reticle');
const motionButton = document.querySelector('#motion-button');
const pointerReadout = document.querySelector('#pointer-readout');
const context = fieldCanvas.getContext('2d');
const noiseContext = noiseCanvas.getContext('2d');

let running = !reducedMotion.matches;
let animationFrame = 0;
let particles = [];
let pulse = 0;
let fieldWidth = 0;
let fieldHeight = 0;
const pointer = { x: 0, y: 0, active: false };

function fitCanvas(canvas, canvasContext, width, height) {
  const scale = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.floor(width * scale);
  canvas.height = Math.floor(height * scale);
  canvasContext.setTransform(scale, 0, 0, scale, 0, 0);
}

function seedParticles() {
  const count = Math.min(150, Math.max(65, Math.floor(fieldWidth / 8)));
  particles = Array.from({ length: count }, (_, index) => {
    const angle = index * 2.39996;
    const radius = Math.sqrt(index / count) * Math.min(fieldWidth, fieldHeight) * 0.46;
    return {
      x: fieldWidth / 2 + Math.cos(angle) * radius * 1.55,
      y: fieldHeight / 2 + Math.sin(angle) * radius,
      phase: Math.random() * Math.PI * 2,
      size: Math.random() * 1.6 + 0.7,
    };
  });
}

function resize() {
  const bounds = fieldStage.getBoundingClientRect();
  fieldWidth = bounds.width;
  fieldHeight = bounds.height;
  fitCanvas(fieldCanvas, context, fieldWidth, fieldHeight);
  fitCanvas(noiseCanvas, noiseContext, window.innerWidth, window.innerHeight);
  pointer.x = fieldWidth / 2;
  pointer.y = fieldHeight / 2;
  seedParticles();
}

function drawNoise(time) {
  const width = window.innerWidth;
  const height = window.innerHeight;
  noiseContext.clearRect(0, 0, width, height);
  noiseContext.strokeStyle = 'rgba(201, 255, 69, 0.11)';
  noiseContext.lineWidth = 1;
  for (let line = 0; line < 7; line += 1) {
    noiseContext.beginPath();
    for (let x = 0; x <= width; x += 10) {
      const y = height * (0.2 + line * 0.095) + Math.sin(x * 0.012 + time * 0.00035 + line) * (22 + line * 4);
      if (x === 0) noiseContext.moveTo(x, y);
      else noiseContext.lineTo(x, y);
    }
    noiseContext.stroke();
  }
}

function drawField(time) {
  context.clearRect(0, 0, fieldWidth, fieldHeight);
  const centerX = pointer.active ? pointer.x : fieldWidth / 2;
  const centerY = pointer.active ? pointer.y : fieldHeight / 2;
  const wave = pulse > 0 ? (1 - pulse) * Math.max(fieldWidth, fieldHeight) : 0;

  particles.forEach((particle, index) => {
    const drift = Math.sin(time * 0.0007 + particle.phase) * 7;
    const x = particle.x + Math.cos(particle.phase) * drift;
    const y = particle.y + Math.sin(particle.phase) * drift;
    const distance = Math.hypot(x - centerX, y - centerY);
    const influence = Math.max(0, 1 - distance / 190);
    const pulseGlow = pulse > 0 ? Math.max(0, 1 - Math.abs(distance - wave) / 55) : 0;
    const alpha = 0.2 + influence * 0.72 + pulseGlow * 0.65;

    if (influence > 0.13 && index % 2 === 0) {
      context.beginPath();
      context.moveTo(x, y);
      context.lineTo(centerX, centerY);
      context.strokeStyle = `rgba(201, 255, 69, ${influence * 0.1})`;
      context.stroke();
    }
    context.beginPath();
    context.arc(x, y, particle.size + influence * 1.8, 0, Math.PI * 2);
    context.fillStyle = `rgba(201, 255, 69, ${alpha})`;
    context.shadowBlur = influence * 14;
    context.shadowColor = '#c9ff45';
    context.fill();
  });
  context.shadowBlur = 0;
  if (pulse > 0) {
    context.beginPath();
    context.arc(centerX, centerY, wave, 0, Math.PI * 2);
    context.strokeStyle = `rgba(201, 255, 69, ${pulse * 0.45})`;
    context.lineWidth = 1;
    context.stroke();
    pulse = Math.max(0, pulse - 0.018);
  }
}

function animate(time) {
  drawNoise(time);
  drawField(time);
  if (running) animationFrame = requestAnimationFrame(animate);
}

function updateMetrics(x, y) {
  const normalizedX = x / fieldWidth;
  const normalizedY = y / fieldHeight;
  const coupling = 78 + normalizedX * 18;
  const coherence = 0.86 + (1 - Math.abs(normalizedY - 0.5)) * 0.12;
  const depth = 4.8 + normalizedY * 3.2;
  document.querySelector('#coupling-value').textContent = `${coupling.toFixed(1)}%`;
  document.querySelector('#coherence-value').textContent = coherence.toFixed(3);
  document.querySelector('#depth-value').textContent = depth.toFixed(2);
  document.querySelector('#hero-index').textContent = (coupling / 100).toFixed(3);
  document.querySelector('#coupling-bar').style.width = `${coupling}%`;
  document.querySelector('#coherence-bar').style.width = `${coherence * 100}%`;
  document.querySelector('#depth-bar').style.width = `${depth * 10}%`;
  pointerReadout.innerHTML = `X ${(normalizedX * 100).toFixed(1)}&nbsp;&nbsp;Y ${(normalizedY * 100).toFixed(1)}`;
}

function setPointer(event) {
  const bounds = fieldStage.getBoundingClientRect();
  pointer.x = Math.max(0, Math.min(fieldWidth, event.clientX - bounds.left));
  pointer.y = Math.max(0, Math.min(fieldHeight, event.clientY - bounds.top));
  pointer.active = true;
  reticle.style.left = `${pointer.x}px`;
  reticle.style.top = `${pointer.y}px`;
  updateMetrics(pointer.x, pointer.y);
  if (!running) drawField(performance.now());
}

function setRunning(next) {
  running = next;
  document.body.classList.toggle('paused', !running);
  motionButton.setAttribute('aria-pressed', String(!running));
  motionButton.lastElementChild.textContent = running ? 'Pause motion' : 'Resume motion';
  cancelAnimationFrame(animationFrame);
  if (running) animationFrame = requestAnimationFrame(animate);
  else {
    drawNoise(performance.now());
    drawField(performance.now());
  }
}

fieldStage.addEventListener('pointermove', setPointer);
fieldStage.addEventListener('pointerleave', () => { pointer.active = false; });
fieldStage.addEventListener('click', (event) => { setPointer(event); pulse = 1; });
fieldStage.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); pulse = 1; }
});
motionButton.addEventListener('click', () => setRunning(!running));
document.querySelector('#enter-button').addEventListener('click', () => fieldStage.scrollIntoView({ behavior: reducedMotion.matches ? 'auto' : 'smooth', block: 'center' }));
reducedMotion.addEventListener('change', (event) => setRunning(!event.matches));
window.addEventListener('resize', resize);
document.addEventListener('visibilitychange', () => {
  if (document.hidden) cancelAnimationFrame(animationFrame);
  else if (running) animationFrame = requestAnimationFrame(animate);
});

setInterval(() => { document.querySelector('#clock').textContent = new Date().toISOString().slice(11, 19); }, 1000);
resize();
setRunning(running);

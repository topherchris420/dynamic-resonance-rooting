const axes = document.querySelectorAll('.axis');
const pulseButton = document.querySelector('#pulse-button');
const focusButton = document.querySelector('#focus-button');
const field = document.querySelector('#field');
const state = document.querySelector('#field-state');
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

let frameId = 0;
let hue = 190;
let running = !reducedMotion.matches;
let lastPaint = 0;

function paint(timestamp) {
  if (!running) return;
  if (timestamp - lastPaint > 42) {
    hue = (hue + 1.2) % 360;
    document.documentElement.style.setProperty('--signal-hue', hue);
    axes.forEach((axis) => axis.style.setProperty('--glow-hue', hue));
    lastPaint = timestamp;
  }
  frameId = requestAnimationFrame(paint);
}

function setRunning(nextRunning) {
  running = nextRunning;
  pulseButton.setAttribute('aria-pressed', String(!running));
  pulseButton.textContent = running ? 'Pause resonance' : 'Resume resonance';
  state.textContent = running ? 'tracking · 60 fps' : 'field paused · still';
  if (running) {
    cancelAnimationFrame(frameId);
    frameId = requestAnimationFrame(paint);
  } else {
    cancelAnimationFrame(frameId);
  }
}

pulseButton.addEventListener('click', () => setRunning(!running));
focusButton.addEventListener('click', () => {
  field.scrollIntoView({ behavior: reducedMotion.matches ? 'auto' : 'smooth', block: 'center' });
  field.querySelector('.field-visual').focus({ preventScroll: true });
});
field.querySelector('.field-visual').setAttribute('tabindex', '0');

document.addEventListener('visibilitychange', () => {
  if (document.hidden) cancelAnimationFrame(frameId);
  else if (running) frameId = requestAnimationFrame(paint);
});
reducedMotion.addEventListener('change', (event) => setRunning(!event.matches));
setRunning(running);

/*
 * Live DRR lab and evidence charts for index.html.
 *
 * The lab simulates a three-channel system with a known answer, streams it,
 * and every WINDOW new samples runs the real operators from drr-engine.js:
 * resonance depth for each channel and lagged-correlation rooting with 99
 * circular-shift surrogates. Evidence charts read the JSON block that
 * scripts/sync_site_evidence.py copies from results/expected.
 */
(() => {
  'use strict';

  const DRR = window.DRR;
  const FS = 100;
  const WINDOW = 512;
  const DEPTH_WINDOW = 256;
  const MAX_LAG = 6;
  const SURROGATES = 99;
  const ALPHA = 0.05;
  const SAMPLES_PER_SECOND = 360;
  const SOURCE_HZ = 7;
  const DISTRACTOR_HZ = 11;
  const TWO_PI = Math.PI * 2;
  const COLORS = { acid: '#c9ff45', paper: '#e8e9e4', neutral: '#7d877f', muted: '#8b938b', line: 'rgba(232,233,228,0.13)', surface: '#0b100e' };
  const LANES = [
    { name: 'dim_0', role: 'source · 7 Hz + broadband', color: COLORS.acid },
    { name: 'dim_1', role: 'response', color: COLORS.paper },
    { name: 'dim_2', role: 'distractor · 11 Hz', color: COLORS.neutral },
  ];

  const $ = (selector) => document.querySelector(selector);
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

  // ------------------------------------------------------------- the system

  const params = { coupling: 0.5, lag: 2, rhythm: 1.0, noise: 0.3 };

  class System {
    constructor(seed) {
      this.rng = DRR.createRng(seed);
      this.phase = this.rng.next() * TWO_PI;
      this.phase2 = this.rng.next() * TWO_PI;
      this.broadband = 0;
      this.own = 0;
      this.history = new Float64Array(8);
      this.t = 0;
    }

    step() {
      const n = this.rng.normal;
      const { coupling, lag, rhythm, noise } = params;
      this.phase += (TWO_PI * SOURCE_HZ) / FS + 0.04 * n();
      this.broadband = 0.8 * this.broadband + 0.6 * n();
      const x0 = rhythm * Math.sin(this.phase) + this.broadband + noise * n();
      const delayed = this.t >= lag ? this.history[(this.t - lag) % 8] : 0;
      this.history[this.t % 8] = x0;
      // Unit-variance AR(1) scaled to dim_0's spread, so coupling is a mixing weight.
      this.own = 0.5 * this.own + 0.866 * n();
      const scale = Math.sqrt((rhythm * rhythm) / 2 + 1 + noise * noise);
      const x1 = coupling * delayed + Math.sqrt(1 - coupling * coupling) * this.own * scale + noise * n();
      this.phase2 += (TWO_PI * DISTRACTOR_HZ) / FS + 0.04 * n();
      const x2 = 0.7 * Math.sin(this.phase2) + 0.6 * n();
      this.t += 1;
      return [x0, x1, x2];
    }
  }

  const buffers = LANES.map(() => new Float64Array(WINDOW));
  let head = 0;
  let system = new System(20260925);
  let sinceAnalysis = 0;
  let windowsAnalyzed = 0;
  const tally = { runs: 0, hit: 0, miss: 0, falseAlarm: 0 };

  function push(values) {
    for (let c = 0; c < LANES.length; c += 1) buffers[c][head] = values[c];
    head = (head + 1) % WINDOW;
  }

  function chronological(c) {
    const out = new Float64Array(WINDOW);
    for (let i = 0; i < WINDOW; i += 1) out[i] = buffers[c][(head + i) % WINDOW];
    return out;
  }

  function freshWindow() {
    for (let i = 0; i < 64; i += 1) system.step();
    for (let i = 0; i < WINDOW; i += 1) push(system.step());
    sinceAnalysis = 0;
  }

  // --------------------------------------------------------------- analysis

  const fmt = (value, digits = 2) => (Number.isFinite(value) ? value.toFixed(digits) : '—');
  const edgeText = (e) => `dim_${e.source} → dim_${e.target} · lag ${e.lag} · p ${e.adjusted_p_value.toFixed(3)}`;

  let latest = null;

  function analyze() {
    const columns = LANES.map((_, c) => chronological(c));
    const started = performance.now();
    windowsAnalyzed += 1;
    const rooting = DRR.rooting(columns, { maxLag: MAX_LAG, nSurrogates: SURROGATES, alpha: ALPHA, seed: windowsAnalyzed });
    const depths = columns.map((column) => DRR.depth(column, DEPTH_WINDOW, FS, null));
    const spectrum = DRR.welch(columns[0], FS, 256);
    const elapsed = performance.now() - started;
    sinceAnalysis = 0;
    latest = { rooting, depths, spectrum };

    const truth = params.coupling > 0 ? { source: 0, target: 1, lag: params.lag } : null;
    const significant = rooting.significant_edges;
    const isTruthPair = (e) => truth && e.source === 0 && e.target === 1;
    const exact = significant.some((e) => isTruthPair(e) && e.lag === truth.lag);
    const rightPair = significant.some(isTruthPair);
    const extras = significant.filter((e) => !isTruthPair(e));

    tally.runs += 1;
    if (truth) tally[exact ? 'hit' : 'miss'] += 1;
    if (extras.length) tally.falseAlarm += 1;

    let state = 'good';
    let status = 'Recovered the true edge';
    if (!truth) {
      state = extras.length ? 'bad' : 'good';
      status = extras.length ? 'False alarm: there is no real link' : 'Correctly silent';
    } else if (exact && extras.length) {
      state = 'partial'; status = 'Recovered it, plus an extra edge';
    } else if (!exact && rightPair) {
      state = 'partial'; status = 'Right direction, wrong lag';
    } else if (!exact && extras.length) {
      state = 'bad'; status = 'Reported the wrong edge';
    } else if (!exact) {
      state = 'partial'; status = 'Missed in this window';
    }

    const found = $('#verdict-found');
    found.replaceChildren();
    if (!significant.length) found.textContent = `No edge passes α = ${ALPHA}`;
    significant.slice(0, 3).forEach((e) => {
      const line = document.createElement('span');
      line.className = 'edge-line';
      line.innerHTML = `dim_${e.source} → dim_${e.target} <small>lag ${e.lag} · adjusted p ${e.adjusted_p_value.toFixed(3)}</small>`;
      found.append(line);
    });
    const statusEl = $('#verdict-status');
    statusEl.dataset.state = state;
    statusEl.lastElementChild.textContent = status;
    $('#tally-runs').textContent = tally.runs;
    $('#tally-hit').textContent = truth ? tally.hit : '—';
    $('#tally-miss').textContent = truth ? tally.miss : '—';
    $('#tally-false').textContent = tally.falseAlarm;

    $('#hero-depth').textContent = fmt(depths[0].resonance_depth);
    $('#hero-verdict').textContent = significant.length
      ? `Latest window: ${edgeText(significant[0])}`
      : 'Latest window: no edge passes the test';
    $('#engine-status').textContent = `Engine live · ${elapsed.toFixed(0)} ms per window`;

    renderGraph(rooting);
    renderDepth(depths);
    drawSpectrum();
  }

  function updateTruthAndNote() {
    $('#verdict-truth').innerHTML = params.coupling > 0
      ? `dim_0 → dim_1 <small>lag ${params.lag}</small>`
      : 'No link <small>coupling 0</small>';
    let note = 'Recovered means the exact edge and lag passed the max-statistic test. Lower the coupling to find where power runs out.';
    if (params.coupling === 0) {
      const size = evidence && evidence.calibration.size[0];
      const measured = size ? ` The calibration study measured ${(size.circular_shift * 100).toFixed(1)}% on white noise.` : '';
      note = `Every edge reported now is a false alarm. At α = 0.05, expect about one window in twenty.${measured}`;
    } else if (params.rhythm >= 2.2) {
      note = 'A strong pure rhythm repeats every 14 samples, so a delayed copy also looks like an advanced one. Watch direction get harder to call.';
    } else if (params.rhythm === 0) {
      note = 'With no rhythm, dim_0 is broadband red noise. Its memory makes dim_1 appear to lead it too: lead–lag is not causation.';
    }
    $('#verdict-note').textContent = note;
  }

  // ----------------------------------------------------------------- graph

  const NODES = [
    { x: 58, y: 64, label: 'dim_0' },
    { x: 262, y: 64, label: 'dim_1' },
    { x: 160, y: 196, label: 'dim_2' },
  ];
  const SVG_NS = 'http://www.w3.org/2000/svg';
  const el = (name, attrs, text) => {
    const node = document.createElementNS(SVG_NS, name);
    for (const [key, value] of Object.entries(attrs || {})) node.setAttribute(key, value);
    if (text !== undefined) node.textContent = text;
    return node;
  };

  function renderGraph(rooting) {
    const svg = $('#edge-graph');
    svg.replaceChildren();
    const defs = el('defs');
    for (const [id, color] of [['arrow-sig', COLORS.acid], ['arrow-cand', COLORS.neutral]]) {
      const marker = el('marker', { id, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' });
      marker.append(el('path', { d: 'M0 0 L10 5 L0 10 z', fill: color }));
      defs.append(marker);
    }
    svg.append(defs);

    const k = NODES.length;
    const significant = new Set(rooting.significant_edges.map((e) => e.source * k + e.target));
    const candidate = new Set(rooting.candidate_edges.map((e) => e.source * k + e.target));
    const labels = [];
    for (let s = 0; s < k; s += 1) {
      for (let d = 0; d < k; d += 1) {
        if (s === d) continue;
        const a = NODES[s];
        const b = NODES[d];
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const length = Math.hypot(dx, dy);
        const ux = dx / length;
        const uy = dy / length;
        const bend = 16;
        const start = { x: a.x + ux * 28 - uy * 6, y: a.y + uy * 28 + ux * 6 };
        const end = { x: b.x - ux * 30 - uy * 6, y: b.y - uy * 30 + ux * 6 };
        const control = { x: (start.x + end.x) / 2 - uy * bend, y: (start.y + end.y) / 2 + ux * bend };
        const index = s * k + d;
        const isSig = significant.has(index);
        const isCand = candidate.has(index);
        const path = el('path', {
          d: `M${start.x} ${start.y} Q${control.x} ${control.y} ${end.x} ${end.y}`,
          fill: 'none',
          stroke: isSig ? COLORS.acid : COLORS.neutral,
          'stroke-width': isSig ? 2.5 : 1.25,
          'stroke-linecap': 'round',
          opacity: isSig ? 1 : isCand ? 0.6 : 0.18,
          'marker-end': isSig ? 'url(#arrow-sig)' : isCand ? 'url(#arrow-cand)' : '',
        });
        path.append(el('title', {}, `dim_${s} → dim_${d}: score ${rooting.scores[index].toFixed(3)}, lag ${rooting.lags[index]}, adjusted p ${Number.isFinite(rooting.adjusted_p_values[index]) ? rooting.adjusted_p_values[index].toFixed(3) : 'n/a'}`));
        svg.append(path);
        if (isSig) {
          labels.push(el('text', {
            x: control.x - uy * 6, y: control.y + ux * 6, 'text-anchor': 'middle', 'dominant-baseline': 'middle',
            fill: COLORS.paper, 'font-size': 11, 'font-family': 'DM Mono, monospace',
          }, `lag ${rooting.lags[index]} · p ${rooting.adjusted_p_values[index].toFixed(2)}`));
        }
      }
    }
    NODES.forEach((node, i) => {
      svg.append(el('circle', { cx: node.x, cy: node.y, r: 24, fill: '#111815', stroke: LANES[i].color, 'stroke-width': 1.5 }));
      svg.append(el('text', { x: node.x, y: node.y + 1, 'text-anchor': 'middle', 'dominant-baseline': 'middle', fill: COLORS.paper, 'font-size': 12, 'font-family': 'DM Mono, monospace' }, node.label));
    });
    labels.forEach((label) => {
      const halo = label.cloneNode(true);
      halo.setAttribute('stroke', COLORS.surface);
      halo.setAttribute('stroke-width', 5);
      halo.setAttribute('stroke-linejoin', 'round');
      svg.append(halo, label);
    });
  }

  // ----------------------------------------------------------------- depth

  const COMPONENTS = [
    ['spectral_concentration', 'Spectral concentration'],
    ['temporal_persistence', 'Temporal persistence'],
    ['phase_coherence', 'Phase coherence'],
    ['amplitude_stability', 'Amplitude stability'],
  ];

  function renderDepth(depths) {
    $('#depth-value').textContent = fmt(depths[0].resonance_depth);
    const list = $('#depth-meters');
    if (!list.children.length) {
      for (const [, label] of COMPONENTS) {
        const item = document.createElement('li');
        item.innerHTML = `<span>${label}</span><b>—</b><span class="meter"><i></i></span>`;
        list.append(item);
      }
    }
    COMPONENTS.forEach(([key], i) => {
      const value = depths[0].components[key];
      const item = list.children[i];
      item.querySelector('b').textContent = fmt(value);
      item.querySelector('.meter i').style.width = `${(value * 100).toFixed(1)}%`;
    });
    $('#depth-others').textContent = `dim_1 ${fmt(depths[1].resonance_depth)} · dim_2 ${fmt(depths[2].resonance_depth)} · target ${fmt(depths[0].target_frequency)} Hz`;
  }

  // --------------------------------------------------------------- canvases

  function fit(canvas) {
    const scale = Math.min(window.devicePixelRatio || 1, 2);
    const { width, height } = canvas.getBoundingClientRect();
    if (canvas.width !== Math.round(width * scale) || canvas.height !== Math.round(height * scale)) {
      canvas.width = Math.round(width * scale);
      canvas.height = Math.round(height * scale);
    }
    const context = canvas.getContext('2d');
    context.setTransform(scale, 0, 0, scale, 0, 0);
    return { context, width, height };
  }

  const laneScale = [3, 3, 3];

  function drawTraces() {
    const { context, width, height } = fit($('#trace-canvas'));
    context.clearRect(0, 0, width, height);
    const laneHeight = height / LANES.length;
    const left = 12;
    const plotWidth = width - left - 12;
    LANES.forEach((lane, c) => {
      const top = c * laneHeight;
      const mid = top + laneHeight / 2 + 8;
      let peak = 0;
      for (let i = 0; i < WINDOW; i += 1) peak = Math.max(peak, Math.abs(buffers[c][i]));
      laneScale[c] = 0.9 * laneScale[c] + 0.1 * Math.max(peak, 1e-6);
      const gain = (laneHeight * 0.36) / laneScale[c];
      context.strokeStyle = COLORS.line;
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(left, mid);
      context.lineTo(left + plotWidth, mid);
      context.stroke();
      context.fillStyle = COLORS.muted;
      context.font = '11px "DM Mono", monospace';
      context.fillText(`${lane.name} · ${lane.role}`, left, top + 18);
      context.strokeStyle = lane.color;
      context.lineWidth = 1.5;
      context.lineJoin = 'round';
      context.beginPath();
      for (let i = 0; i < WINDOW; i += 1) {
        const x = left + (i / (WINDOW - 1)) * plotWidth;
        const y = mid - buffers[c][(head + i) % WINDOW] * gain;
        if (i === 0) context.moveTo(x, y);
        else context.lineTo(x, y);
      }
      context.stroke();
    });
  }

  function drawHero() {
    const canvas = $('#hero-canvas');
    const { context, width, height } = fit(canvas);
    context.clearRect(0, 0, width, height);
    LANES.forEach((lane, c) => {
      const mid = height * (0.3 + c * 0.2);
      const gain = (height * 0.07) / laneScale[c];
      context.strokeStyle = c === 0 ? 'rgba(201,255,69,0.22)' : 'rgba(232,233,228,0.08)';
      context.lineWidth = 1;
      context.beginPath();
      for (let i = 0; i < WINDOW; i += 1) {
        const x = (i / (WINDOW - 1)) * width;
        const y = mid - buffers[c][(head + i) % WINDOW] * gain;
        if (i === 0) context.moveTo(x, y);
        else context.lineTo(x, y);
      }
      context.stroke();
    });
  }

  function drawSpectrum() {
    if (!latest) return;
    const { context, width, height } = fit($('#spectrum-canvas'));
    context.clearRect(0, 0, width, height);
    const { freqs, psd } = latest.spectrum;
    const pad = { left: 50, right: 12, top: 12, bottom: 24 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;
    let max = 0;
    for (let i = 1; i < psd.length; i += 1) max = Math.max(max, psd[i]);
    const floor = -40;
    const db = (p) => Math.max(floor, 10 * Math.log10(Math.max(p, 1e-30) / max));
    const x = (f) => pad.left + (f / (FS / 2)) * plotW;
    const y = (d) => pad.top + (d / floor) * plotH;

    context.font = '10.5px "DM Mono", monospace';
    context.fillStyle = COLORS.muted;
    context.strokeStyle = COLORS.line;
    context.lineWidth = 1;
    for (const d of [0, -20, -40]) {
      context.beginPath(); context.moveTo(pad.left, y(d)); context.lineTo(width - pad.right, y(d)); context.stroke();
      context.textAlign = 'right'; context.fillText(`${d < 0 ? '−' : ''}${Math.abs(d)} dB`, pad.left - 6, y(d) + 3);
    }
    context.textAlign = 'center';
    for (const f of [0, 10, 20, 30, 40]) context.fillText(`${f}`, x(f), height - 6);
    context.textAlign = 'right';
    context.fillText('50 Hz', width - pad.right, height - 6);

    context.beginPath();
    context.moveTo(x(freqs[1]), y(floor));
    for (let i = 1; i < psd.length; i += 1) context.lineTo(x(freqs[i]), y(db(psd[i])));
    context.lineTo(x(freqs[psd.length - 1]), y(floor));
    context.closePath();
    context.fillStyle = 'rgba(201,255,69,0.1)';
    context.fill();
    context.beginPath();
    for (let i = 1; i < psd.length; i += 1) {
      if (i === 1) context.moveTo(x(freqs[i]), y(db(psd[i])));
      else context.lineTo(x(freqs[i]), y(db(psd[i])));
    }
    context.strokeStyle = COLORS.acid;
    context.lineWidth = 2;
    context.lineJoin = 'round';
    context.stroke();

    const target = latest.depths[0].target_frequency;
    if (target > 0) {
      context.strokeStyle = COLORS.paper;
      context.lineWidth = 1;
      context.beginPath(); context.moveTo(x(target), pad.top); context.lineTo(x(target), y(floor)); context.stroke();
      context.fillStyle = COLORS.paper;
      context.textAlign = target > 38 ? 'right' : 'left';
      context.fillText(`${target.toFixed(2)} Hz`, x(target) + (target > 38 ? -6 : 6), pad.top + 10);
      $('#spectrum-peak').textContent = `peak ${target.toFixed(2)} Hz · true ${SOURCE_HZ} Hz`;
    }
  }

  // --------------------------------------------------------------- streaming

  let running = !reducedMotion.matches;
  let frame = 0;
  let lastTime = 0;
  let carry = 0;
  let heroVisible = true;

  function tick(time) {
    const dt = lastTime ? Math.min(0.1, (time - lastTime) / 1000) : 0;
    lastTime = time;
    carry += dt * SAMPLES_PER_SECOND;
    const steps = Math.floor(carry);
    carry -= steps;
    for (let i = 0; i < steps; i += 1) push(system.step());
    sinceAnalysis += steps;
    if (sinceAnalysis >= WINDOW) analyze();
    drawTraces();
    if (heroVisible) drawHero();
    updateProgress();
    if (running) frame = requestAnimationFrame(tick);
  }

  function updateProgress() {
    const remaining = Math.max(0, WINDOW - sinceAnalysis);
    $('#next-analysis').textContent = running ? `next analysis in ${remaining} samples` : 'stream paused';
    $('#window-progress').style.width = `${((sinceAnalysis / WINDOW) * 100).toFixed(1)}%`;
  }

  function setRunning(next) {
    running = next;
    document.body.classList.toggle('paused', !running);
    const button = $('#motion-button');
    button.setAttribute('aria-pressed', String(!running));
    button.lastElementChild.textContent = running ? 'Pause stream' : 'Resume stream';
    cancelAnimationFrame(frame);
    lastTime = 0;
    if (running) frame = requestAnimationFrame(tick);
    else { drawTraces(); drawHero(); updateProgress(); }
  }

  function restart() {
    tally.runs = 0; tally.hit = 0; tally.miss = 0; tally.falseAlarm = 0;
    freshWindow();
    analyze();
    drawTraces();
    drawHero();
    updateProgress();
    updateTruthAndNote();
  }

  // ---------------------------------------------------------------- controls

  const controls = {
    coupling: (v) => v.toFixed(2),
    lag: (v) => `${v} sample${v === 1 ? '' : 's'}`,
    rhythm: (v) => v.toFixed(1),
    noise: (v) => v.toFixed(2),
  };
  let restartTimer = 0;
  for (const [name, format] of Object.entries(controls)) {
    const input = document.getElementById(name);
    const output = document.getElementById(`${name}-out`);
    input.addEventListener('input', () => {
      params[name] = Number(input.value);
      output.textContent = format(params[name]);
      updateTruthAndNote();
      clearTimeout(restartTimer);
      restartTimer = setTimeout(restart, 120);
    });
  }
  $('#cut-link').addEventListener('click', () => {
    const input = document.getElementById('coupling');
    input.value = '0';
    input.dispatchEvent(new Event('input'));
  });
  $('#new-window').addEventListener('click', () => { freshWindow(); analyze(); drawTraces(); drawHero(); updateProgress(); });
  $('#lab-controls').addEventListener('submit', (event) => event.preventDefault());
  $('#motion-button').addEventListener('click', () => setRunning(!running));
  reducedMotion.addEventListener('change', (event) => setRunning(!event.matches));
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) cancelAnimationFrame(frame);
    else if (running) { lastTime = 0; frame = requestAnimationFrame(tick); }
  });
  window.addEventListener('resize', () => { drawTraces(); drawHero(); drawSpectrum(); });
  if ('IntersectionObserver' in window) {
    new IntersectionObserver(([entry]) => { heroVisible = entry.isIntersecting; }).observe($('.hero'));
  }

  // ---------------------------------------------------------------- evidence

  let evidence = null;
  try {
    evidence = JSON.parse($('#evidence-data').textContent);
  } catch (error) {
    evidence = null;
  }

  const tooltip = $('#chart-tooltip');
  function showTip(target, event) {
    tooltip.innerHTML = target.dataset.tip;
    tooltip.hidden = false;
    const rect = target.getBoundingClientRect();
    const px = event && event.clientX ? event.clientX : rect.left + rect.width / 2;
    const py = event && event.clientY ? event.clientY : rect.top;
    const tw = tooltip.offsetWidth;
    const th = tooltip.offsetHeight;
    tooltip.style.left = `${Math.min(window.innerWidth - tw - 8, Math.max(8, px + 14))}px`;
    tooltip.style.top = `${Math.max(8, py - th - 12)}px`;
  }
  const hideTip = () => { tooltip.hidden = true; };
  function bindTips(svg) {
    svg.querySelectorAll('[data-tip]').forEach((node) => {
      node.setAttribute('tabindex', '0');
      node.addEventListener('pointerenter', (e) => showTip(node, e));
      node.addEventListener('pointermove', (e) => showTip(node, e));
      node.addEventListener('pointerleave', hideTip);
      node.addEventListener('focus', () => showTip(node));
      node.addEventListener('blur', hideTip);
    });
  }

  const pct = (v, digits = 1) => `${(v * 100).toFixed(digits)}%`;

  // Charts draw in CSS pixels at the container's width, so text stays legible on phones.
  function chartWidth(svg) {
    svg.replaceChildren();
    svg.removeAttribute('viewBox');
    return Math.max(280, Math.round(svg.getBoundingClientRect().width));
  }

  // Column with a 4px rounded data end and a square baseline.
  function barPath(x, y, w, h) {
    const r = Math.min(4, w / 2, h);
    return `M${x} ${y + h}V${y + r}Q${x} ${y} ${x + r} ${y}H${x + w - r}Q${x + w} ${y} ${x + w} ${y + r}V${y + h}Z`;
  }

  function sizeChart(data) {
    const svg = $('#size-chart');
    const W = chartWidth(svg); const H = W < 600 ? 260 : 300;
    const m = { left: 44, right: 56, top: 18, bottom: 44 };
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    const y = (v) => m.top + (1 - v) * (H - m.top - m.bottom);
    for (const t of [0, 0.25, 0.5, 0.75, 1]) {
      svg.append(el('line', { class: 'grid', x1: m.left, x2: W - m.right, y1: y(t), y2: y(t) }));
      svg.append(el('text', { x: m.left - 8, y: y(t) + 4, 'text-anchor': 'end' }, pct(t, 0)));
    }
    const groups = data.size;
    const band = (W - m.left - m.right) / groups.length;
    const bar = Math.min(24, Math.floor(band / 3));
    groups.forEach((row, g) => {
      const cx = m.left + band * (g + 0.5);
      [['circular_shift', COLORS.acid, -1], ['permutation', COLORS.neutral, 1]].forEach(([key, color, side]) => {
        const value = row[key];
        const x = side < 0 ? cx - bar - 1 : cx + 1;
        const h = Math.max(1.5, y(0) - y(value));
        const label = key === 'circular_shift' ? 'Circular shift' : 'Permutation';
        const ci = row[`${key}_ci`];
        const tip = `<b>${label} · AR ${row.ar.toFixed(2)}</b>False-alarm rate ${pct(value)}<br>95% interval ${pct(ci[0])}–${pct(ci[1])}`;
        svg.append(el('path', { class: 'mark', d: barPath(x, y(0) - h, bar, h), fill: color }));
        svg.append(el('rect', { class: 'hit', x: x - 6, y: m.top, width: bar + 12, height: y(0) - m.top, 'data-tip': tip, 'aria-label': `${label}, AR ${row.ar}: ${pct(value)}` }));
        if (key === 'circular_shift') {
          svg.append(el('text', { class: 'value', x: x + bar, y: y(value) - 8, 'text-anchor': 'end' }, pct(value)));
        } else if (value === Math.max(...groups.map((r) => r.permutation))) {
          svg.append(el('text', { class: 'value', x: x + bar / 2, y: y(value) - 8, 'text-anchor': 'middle' }, pct(value)));
        }
      });
      svg.append(el('text', { x: cx, y: H - 18, 'text-anchor': 'middle' }, g === 0 && W >= 600 ? 'AR 0.00 · white' : `AR ${row.ar.toFixed(2)}`));
    });
    svg.append(el('text', { x: m.left + (W - m.left - m.right) / 2, y: H - 2, 'text-anchor': 'middle' }, 'autocorrelation of each series →'));
    svg.append(el('line', { class: 'ref', x1: m.left, x2: W - m.right, y1: y(data.alpha), y2: y(data.alpha) }));
    svg.append(el('text', { class: 'value', x: W - m.right + 8, y: y(data.alpha) + 4, 'text-anchor': 'start' }, `α ${pct(data.alpha, 0)}`));
    bindTips(svg);

    $('#size-caption').textContent = `${data.size_trials} simulated systems of three independent AR(1) series per bar, ${data.n_samples} samples, ${data.n_surrogates} surrogates. Any reported edge is a false alarm. Shuffling time points destroys each series' memory, so the permutation null fails as soon as the data are autocorrelated.`;
    const table = $('#size-table');
    table.innerHTML = '<tr><th>AR(1) coefficient</th><th>Circular shift</th><th>95% interval</th><th>Permutation</th><th>95% interval</th></tr>' +
      groups.map((r) => `<tr><td>${r.ar.toFixed(2)}</td><td>${pct(r.circular_shift)}</td><td>${pct(r.circular_shift_ci[0])}–${pct(r.circular_shift_ci[1])}</td><td>${pct(r.permutation)}</td><td>${pct(r.permutation_ci[0])}–${pct(r.permutation_ci[1])}</td></tr>`).join('');
  }

  function powerChart(data) {
    const svg = $('#power-chart');
    const W = chartWidth(svg); const H = 260;
    const m = { left: 44, right: 16, top: 16, bottom: 40 };
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    const maxC = Math.max(...data.power.map((p) => p.coupling));
    const x = (c) => m.left + (c / maxC) * (W - m.left - m.right);
    const y = (v) => m.top + (1 - v) * (H - m.top - m.bottom);
    for (const t of [0, 0.5, 1]) {
      svg.append(el('line', { class: 'grid', x1: m.left, x2: W - m.right, y1: y(t), y2: y(t) }));
      svg.append(el('text', { x: m.left - 8, y: y(t) + 4, 'text-anchor': 'end' }, pct(t, 0)));
    }
    let lastTick = -Infinity;
    data.power.forEach((p) => {
      if (x(p.coupling) - lastTick < 36) return;
      lastTick = x(p.coupling);
      svg.append(el('text', { x: x(p.coupling), y: H - 20, 'text-anchor': 'middle' }, p.coupling.toFixed(2)));
    });
    svg.append(el('text', { x: m.left + (W - m.left - m.right) / 2, y: H - 2, 'text-anchor': 'middle' }, 'coupling →'));
    const upper = data.power.map((p) => `${x(p.coupling)} ${y(p.ci[1])}`);
    const lower = data.power.map((p) => `${x(p.coupling)} ${y(p.ci[0])}`).reverse();
    svg.append(el('path', { d: `M${upper.join('L')}L${lower.join('L')}Z`, fill: 'rgba(201,255,69,0.1)' }));
    svg.append(el('path', { d: `M${data.power.map((p) => `${x(p.coupling)} ${y(p.rate)}`).join('L')}`, fill: 'none', stroke: COLORS.acid, 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }));
    data.power.forEach((p) => {
      svg.append(el('circle', { class: 'mark', cx: x(p.coupling), cy: y(p.rate), r: 4, fill: COLORS.acid, stroke: COLORS.surface, 'stroke-width': 2 }));
      svg.append(el('circle', { class: 'hit', cx: x(p.coupling), cy: y(p.rate), r: 14, 'data-tip': `<b>Coupling ${p.coupling.toFixed(2)}</b>Recovered ${pct(p.rate)} of ${data.power_trials} trials<br>95% interval ${pct(p.ci[0])}–${pct(p.ci[1])}`, 'aria-label': `Coupling ${p.coupling}: ${pct(p.rate)}` }));
    });
    const mid = data.power.find((p) => p.rate >= 0.5 && p.rate < 0.95) || data.power[Math.floor(data.power.length / 2)];
    svg.append(el('text', { class: 'value', x: x(mid.coupling) + 10, y: y(mid.rate) + 16 }, `${pct(mid.rate, 0)} at ${mid.coupling.toFixed(2)}`));
    bindTips(svg);

    $('#power-caption').textContent = `An AR(0.5) source drives the target at lag ${data.power_lag}; a third channel is a distractor. ${data.power_trials} trials per point; the band is a 95% Wilson interval. Recovered means the exact edge at the exact lag.`;
    $('#power-table').innerHTML = '<tr><th>Coupling</th><th>Recovered</th><th>95% interval</th></tr>' +
      data.power.map((p) => `<tr><td>${p.coupling.toFixed(2)}</td><td>${pct(p.rate)}</td><td>${pct(p.ci[0])}–${pct(p.ci[1])}</td></tr>`).join('');
  }

  function depthChart(data) {
    const svg = $('#depth-chart');
    const rows = [
      ['ar1_0.00', 'White noise', 'noise'],
      ['ar1_0.50', 'Red noise AR 0.5', 'noise'],
      ['ar1_0.90', 'Red noise AR 0.9', 'noise'],
      ['tone_amp_0.25', 'Tone at −15 dB', 'tone'],
      ['tone_amp_0.50', 'Tone at −9 dB', 'tone'],
      ['tone_amp_1.00', 'Tone at −3 dB', 'tone'],
    ].filter(([key]) => data.depth[key]);
    const W = chartWidth(svg); const rowH = 34;
    const m = { left: 128, right: 40, top: 12, bottom: 34 };
    const H = m.top + rows.length * rowH + m.bottom;
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    const x = (v) => m.left + v * (W - m.left - m.right);
    for (const t of [0, 0.25, 0.5, 0.75, 1]) {
      svg.append(el('line', { class: 'grid', x1: x(t), x2: x(t), y1: m.top, y2: H - m.bottom }));
      svg.append(el('text', { x: x(t), y: H - m.bottom + 16, 'text-anchor': 'middle' }, t.toFixed(2)));
    }
    svg.append(el('text', { x: x(0.5), y: H - 2, 'text-anchor': 'middle' }, 'resonance depth →'));
    rows.forEach(([key, label, kind], i) => {
      const d = data.depth[key];
      const cy = m.top + rowH * (i + 0.5);
      const color = kind === 'tone' ? COLORS.acid : COLORS.neutral;
      svg.append(el('text', { x: m.left - 12, y: cy + 4, 'text-anchor': 'end', class: 'value' }, label));
      svg.append(el('line', { x1: x(d.q05), x2: x(d.q95), y1: cy, y2: cy, stroke: color, 'stroke-width': 2, 'stroke-linecap': 'round' }));
      svg.append(el('circle', { class: 'mark', cx: x(d.median), cy, r: 5, fill: color, stroke: COLORS.surface, 'stroke-width': 2 }));
      svg.append(el('rect', { class: 'hit', x: x(d.q05) - 8, y: cy - rowH / 2, width: Math.max(16, x(d.q95) - x(d.q05) + 16), height: rowH, 'data-tip': `<b>${label}</b>Median ${d.median.toFixed(3)}<br>5–95%: ${d.q05.toFixed(3)}–${d.q95.toFixed(3)}`, 'aria-label': `${label}: median ${d.median.toFixed(2)}` }));
      if (key === 'ar1_0.90' || key === 'tone_amp_1.00') {
        svg.append(el('text', { class: 'value', x: x(d.q95) + 8, y: cy + 4 }, d.median.toFixed(2)));
      }
    });
    bindTips(svg);
    $('#depth-table').innerHTML = '<tr><th>Signal</th><th>5%</th><th>Median</th><th>95%</th></tr>' +
      rows.map(([key, label]) => { const d = data.depth[key]; return `<tr><td>${label}</td><td>${d.q05.toFixed(3)}</td><td>${d.median.toFixed(3)}</td><td>${d.q95.toFixed(3)}</td></tr>`; }).join('');
  }

  function qboCard(qbo) {
    $('#qbo-status').textContent = qbo.status === 'not_supported' ? 'Not supported' : qbo.status.replace(/_/g, ' ');
    $('#qbo-rate').textContent = pct(qbo.drr_holdout_false_alarm_rate);
    $('#qbo-tolerance').textContent = pct(qbo.tolerance, 0);
  }

  function renderEvidence() {
    if (!evidence) return;
    sizeChart(evidence.calibration);
    powerChart(evidence.calibration);
    depthChart(evidence.calibration);
    qboCard(evidence.qbo);
  }
  renderEvidence();
  let chartTimer = 0;
  let chartWidthSeen = window.innerWidth;
  window.addEventListener('resize', () => {
    if (window.innerWidth === chartWidthSeen) return;
    chartWidthSeen = window.innerWidth;
    clearTimeout(chartTimer);
    chartTimer = setTimeout(renderEvidence, 150);
  });

  // -------------------------------------------------------------------- start

  restart();
  setRunning(running);
})();

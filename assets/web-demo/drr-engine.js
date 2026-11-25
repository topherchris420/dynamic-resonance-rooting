/*
 * DRR engine for the browser: a line-for-line port of the Python operators the
 * landing page runs live.
 *
 *   depth(...)   -> drr_framework.modules.DepthCalculator.calculate  (drr_composite_v2)
 *   rooting(...) -> drr_framework.modules.RootingAnalyzer.analyze    (lagged correlation,
 *                   uniform circular-shift surrogates, max-statistic correction)
 *
 * tests/test_web_engine_parity.py runs this file under Node and checks every
 * deterministic output against the Python package. Surrogate draws use a
 * seeded JavaScript generator, so p-values agree in distribution, not bit for bit.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.DRR = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const EPS = 2.220446049250313e-16;
  const TINY = 2.2250738585072014e-308;
  const TWO_PI = 2 * Math.PI;
  const DEPTH_METHOD_VERSION = 'drr_composite_v2';

  // ---------------------------------------------------------------- numerics

  const clip01 = (v) => Math.min(1, Math.max(0, v));

  function mean(a) {
    let s = 0;
    for (let i = 0; i < a.length; i += 1) s += a[i];
    return s / a.length;
  }

  // Population standard deviation (NumPy's default ddof=0).
  function std(a, m) {
    const mu = m === undefined ? mean(a) : m;
    let s = 0;
    for (let i = 0; i < a.length; i += 1) s += (a[i] - mu) * (a[i] - mu);
    return Math.sqrt(s / a.length);
  }

  function centered(a) {
    const mu = mean(a);
    const out = new Float64Array(a.length);
    for (let i = 0; i < a.length; i += 1) out[i] = a[i] - mu;
    return out;
  }

  // In-place complex FFT. Radix-2 for powers of two, direct DFT otherwise.
  function fft(re, im, inverse) {
    const n = re.length;
    if (n <= 1) return;
    const sign = inverse ? 1 : -1;
    if ((n & (n - 1)) !== 0) {
      const outRe = new Float64Array(n);
      const outIm = new Float64Array(n);
      for (let k = 0; k < n; k += 1) {
        let sr = 0;
        let si = 0;
        for (let t = 0; t < n; t += 1) {
          const angle = (sign * TWO_PI * ((k * t) % n)) / n;
          const c = Math.cos(angle);
          const s = Math.sin(angle);
          sr += re[t] * c - im[t] * s;
          si += re[t] * s + im[t] * c;
        }
        outRe[k] = sr;
        outIm[k] = si;
      }
      re.set(outRe);
      im.set(outIm);
    } else {
      for (let i = 1, j = 0; i < n; i += 1) {
        let bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
          let tmp = re[i]; re[i] = re[j]; re[j] = tmp;
          tmp = im[i]; im[i] = im[j]; im[j] = tmp;
        }
      }
      for (let size = 2; size <= n; size <<= 1) {
        const half = size >> 1;
        const step = (sign * TWO_PI) / size;
        for (let start = 0; start < n; start += size) {
          for (let k = 0; k < half; k += 1) {
            const c = Math.cos(step * k);
            const s = Math.sin(step * k);
            const a = start + k;
            const b = a + half;
            const tr = re[b] * c - im[b] * s;
            const ti = re[b] * s + im[b] * c;
            re[b] = re[a] - tr; im[b] = im[a] - ti;
            re[a] += tr; im[a] += ti;
          }
        }
      }
    }
    if (inverse) {
      for (let i = 0; i < n; i += 1) { re[i] /= n; im[i] /= n; }
    }
  }

  // scipy.signal.welch with a periodic Hann window, constant detrend,
  // 50% overlap, density scaling, one-sided spectrum.
  function welch(x, fs, nperseg) {
    const n = x.length;
    const seg = Math.max(1, Math.min(nperseg || Math.min(256, n), n));
    const step = seg - Math.floor(seg / 2);
    const window = new Float64Array(seg);
    let windowPower = 0;
    for (let k = 0; k < seg; k += 1) {
      window[k] = seg > 1 ? 0.5 - 0.5 * Math.cos((TWO_PI * k) / seg) : 1;
      windowPower += window[k] * window[k];
    }
    const bins = Math.floor(seg / 2) + 1;
    const psd = new Float64Array(bins);
    const re = new Float64Array(seg);
    const im = new Float64Array(seg);
    let count = 0;
    for (let start = 0; start + seg <= n; start += step) {
      let mu = 0;
      for (let k = 0; k < seg; k += 1) mu += x[start + k];
      mu /= seg;
      for (let k = 0; k < seg; k += 1) { re[k] = (x[start + k] - mu) * window[k]; im[k] = 0; }
      fft(re, im, false);
      for (let b = 0; b < bins; b += 1) {
        let p = (re[b] * re[b] + im[b] * im[b]) / (fs * windowPower);
        const nyquist = seg % 2 === 0 && b === bins - 1;
        if (b > 0 && !nyquist) p *= 2;
        psd[b] += p;
      }
      count += 1;
    }
    const freqs = new Float64Array(bins);
    for (let b = 0; b < bins; b += 1) { psd[b] /= count; freqs[b] = (b * fs) / seg; }
    return { freqs, psd };
  }

  // scipy.signal.hilbert: the analytic signal.
  function hilbert(x) {
    const n = x.length;
    const re = Float64Array.from(x);
    const im = new Float64Array(n);
    fft(re, im, false);
    const limit = n % 2 === 0 ? n / 2 : (n + 1) / 2;
    for (let k = 1; k < n; k += 1) {
      const h = k < limit ? 2 : k === n / 2 ? 1 : 0;
      re[k] *= h; im[k] *= h;
    }
    fft(re, im, true);
    return { re, im };
  }

  // numpy.unwrap with the default discontinuity of pi.
  function unwrap(phase) {
    const out = Float64Array.from(phase);
    let correction = 0;
    for (let i = 1; i < phase.length; i += 1) {
      const d = phase[i] - phase[i - 1];
      let dmod = ((((d + Math.PI) % TWO_PI) + TWO_PI) % TWO_PI) - Math.PI;
      if (dmod === -Math.PI && d > 0) dmod = Math.PI;
      if (Math.abs(d) >= Math.PI) correction += dmod - d;
      out[i] = phase[i] + correction;
    }
    return out;
  }

  // ------------------------------------------------------------------- depth

  function interpolatePeak(freqs, power, peak) {
    const center = freqs[peak];
    if (peak <= 0 || peak >= power.length - 1) return center;
    const l = Math.log(Math.max(power[peak - 1], TINY));
    const c = Math.log(Math.max(power[peak], TINY));
    const r = Math.log(Math.max(power[peak + 1], TINY));
    const curvature = l - 2 * c + r;
    if (!Number.isFinite(curvature) || curvature >= -EPS) return center;
    const offset = Math.min(0.5, Math.max(-0.5, (0.5 * (l - r)) / curvature));
    return center + offset * (freqs[peak + 1] - freqs[peak]);
  }

  function selectTargetFrequency(data, fs, resonanceFrequencies) {
    if (resonanceFrequencies) {
      for (const f of resonanceFrequencies) if (Number.isFinite(f) && f > 0) return f;
    }
    const nperseg = Math.min(256, data.length);
    if (nperseg < 8) return 0;
    const { freqs, psd } = welch(centered(data), fs, nperseg);
    let peak = -1;
    for (let b = 0; b < freqs.length; b += 1) {
      if (freqs[b] > 0 && (peak < 0 || psd[b] > psd[peak])) peak = b;
    }
    if (peak < 0) return 0;
    return Math.max(interpolatePeak(freqs, psd, peak), 0);
  }

  function spectralConcentration(data, fs, target) {
    const nperseg = Math.min(256, data.length);
    if (nperseg < 8) return 0;
    const { freqs, psd } = welch(centered(data), fs, nperseg);
    let total = 0;
    let peakPower = 0;
    for (let b = 0; b < freqs.length; b += 1) {
      if (freqs[b] > 0) { total += psd[b]; peakPower = Math.max(peakPower, psd[b]); }
    }
    if (total <= EPS) return 0;
    if (target <= 0) return clip01(peakPower / total);
    let nearest = 0;
    for (let b = 1; b < freqs.length; b += 1) {
      if (Math.abs(freqs[b] - target) < Math.abs(freqs[nearest] - target)) nearest = b;
    }
    let band = 0;
    for (let b = Math.max(0, nearest - 1); b < Math.min(psd.length, nearest + 2); b += 1) band += psd[b];
    return clip01(band / total);
  }

  function temporalPersistence(data, windowSize, fs, target) {
    if (target <= 0) return 0;
    const segment = Math.min(windowSize, Math.max(32, Math.floor(data.length / 4)));
    if (data.length < segment) return 0;
    const tolerance = fs / Math.min(256, segment);
    let sum = 0;
    let count = 0;
    for (let start = 0; start + segment <= data.length; start += segment) {
      const f = selectTargetFrequency(data.subarray(start, start + segment), fs, null);
      if (f <= 0) continue;
      sum += Math.exp(-(((f - target) / tolerance) ** 2));
      count += 1;
    }
    return count ? clip01(sum / count) : 0;
  }

  function phaseCoherence(data, fs, target) {
    if (target <= 0 || data.length < 4) return 0;
    const { re, im } = hilbert(centered(data));
    const phase = new Float64Array(data.length);
    for (let i = 0; i < data.length; i += 1) phase[i] = Math.atan2(im[i], re[i]);
    const unwrapped = unwrap(phase);
    let sr = 0;
    let si = 0;
    for (let i = 0; i < data.length; i += 1) {
      const residual = unwrapped[i] - (TWO_PI * target * i) / fs;
      sr += Math.cos(residual);
      si += Math.sin(residual);
    }
    return clip01(Math.hypot(sr, si) / data.length);
  }

  function amplitudeStability(data) {
    if (data.length < 4) return 0;
    const { re, im } = hilbert(centered(data));
    const amplitude = new Float64Array(data.length);
    for (let i = 0; i < data.length; i += 1) amplitude[i] = Math.hypot(re[i], im[i]);
    const m = mean(amplitude);
    if (m <= EPS) return 0;
    return clip01(1 - std(amplitude, m) / m);
  }

  function depth(series, windowSize, fs, resonanceFrequencies) {
    const data = Float64Array.from(series).filter(Number.isFinite);
    const components = {
      spectral_concentration: 0,
      temporal_persistence: 0,
      phase_coherence: 0,
      amplitude_stability: 0,
    };
    if (data.length < windowSize) {
      return { method: DEPTH_METHOD_VERSION, resonance_depth: 0, components, target_frequency: 0 };
    }
    const windowData = data.subarray(data.length - windowSize);
    const target = selectTargetFrequency(windowData, fs, resonanceFrequencies);
    components.spectral_concentration = spectralConcentration(windowData, fs, target);
    components.temporal_persistence = temporalPersistence(data, windowSize, fs, target);
    components.phase_coherence = phaseCoherence(windowData, fs, target);
    components.amplitude_stability = amplitudeStability(windowData);
    const value = clip01(
      0.35 * components.spectral_concentration +
        0.25 * components.temporal_persistence +
        0.25 * components.phase_coherence +
        0.15 * components.amplitude_stability,
    );
    return { method: DEPTH_METHOD_VERSION, resonance_depth: value, components, target_frequency: target };
  }

  // ----------------------------------------------------------------- rooting

  // Seeded generator (mulberry32) so a page load is reproducible.
  function createRng(seed) {
    let a = seed >>> 0;
    const next = () => {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const int = (n) => Math.floor(next() * n);
    let spare = null;
    const normal = () => {
      if (spare !== null) { const v = spare; spare = null; return v; }
      let u = 0;
      while (u === 0) u = next();
      const r = Math.sqrt(-2 * Math.log(u));
      const theta = TWO_PI * next();
      spare = r * Math.sin(theta);
      return r * Math.cos(theta);
    };
    return { next, int, normal };
  }

  // columns: array of Float64Array channels of equal length.
  function laggedCorrelationScores(columns, maxLag) {
    const k = columns.length;
    const n = columns[0].length;
    const scores = new Float64Array(k * k);
    const lags = new Int32Array(k * k).fill(1);
    for (let i = 0; i < k; i += 1) lags[i * k + i] = 0;
    const zx = columns.map(() => new Float64Array(n));
    const zy = columns.map(() => new Float64Array(n));
    const sx = new Float64Array(k);
    const sy = new Float64Array(k);
    for (let lag = 1; lag <= maxLag; lag += 1) {
      const pairs = n - lag;
      if (pairs < 2) break;
      for (let v = 0; v < k; v += 1) {
        const col = columns[v];
        const x = col.subarray(0, pairs);
        const y = col.subarray(lag);
        const mx = mean(x);
        const my = mean(y);
        sx[v] = std(x, mx);
        sy[v] = std(y, my);
        const dx = sx[v] > EPS ? sx[v] : 1;
        const dy = sy[v] > EPS ? sy[v] : 1;
        for (let t = 0; t < pairs; t += 1) { zx[v][t] = (x[t] - mx) / dx; zy[v][t] = (y[t] - my) / dy; }
      }
      for (let s = 0; s < k; s += 1) {
        for (let d = 0; d < k; d += 1) {
          if (s === d || sx[s] <= EPS || sy[d] <= EPS) continue;
          let acc = 0;
          const a = zx[s];
          const b = zy[d];
          for (let t = 0; t < pairs; t += 1) acc += a[t] * b[t];
          const corr = Math.abs(acc) / pairs;
          if (corr > scores[s * k + d]) { scores[s * k + d] = corr; lags[s * k + d] = lag; }
        }
      }
    }
    return { scores, lags };
  }

  // Uniform draw over every shift configuration whose pairwise circular
  // distances are at least `separation` (see RootingAnalyzer._circular_shift_offsets).
  function circularShiftOffsets(n, k, separation, rng) {
    const slack = n - k * separation;
    const pool = slack + k - 1;
    const chosen = new Set();
    while (chosen.size < k - 1) chosen.add(rng.int(pool));
    const bars = Array.from(chosen).sort((a, b) => a - b);
    const edges = [-1, ...bars, pool];
    const offsets = new Array(k).fill(0);
    const order = Array.from({ length: k - 1 }, (_, i) => i + 1);
    for (let i = order.length - 1; i > 0; i -= 1) {
      const j = rng.int(i + 1);
      [order[i], order[j]] = [order[j], order[i]];
    }
    let position = 0;
    for (let g = 0; g < k - 1; g += 1) {
      position += separation + (edges[g + 1] - edges[g] - 1);
      offsets[order[g]] = position;
    }
    return offsets;
  }

  function roll(col, shift) {
    const n = col.length;
    const out = new Float64Array(n);
    for (let t = 0; t < n; t += 1) out[(t + shift) % n] = col[t];
    return out;
  }

  function rooting(columns, options) {
    const opts = Object.assign({ maxLag: 1, nSurrogates: 0, alpha: 0.05, seed: 0 }, options || {});
    const k = columns.length;
    const n = columns[0].length;
    const cols = columns.map((c) => Float64Array.from(c));
    const { scores, lags } = laggedCorrelationScores(cols, opts.maxLag);

    const pValues = new Float64Array(k * k).fill(NaN);
    const adjusted = new Float64Array(k * k).fill(NaN);
    for (let i = 0; i < k; i += 1) { pValues[i * k + i] = 1; adjusted[i * k + i] = 1; }
    if (opts.nSurrogates > 0) {
      const separation = opts.maxLag + 1;
      if (n < k * separation) throw new Error('Circular-shift surrogate assignment is impossible');
      const rng = createRng(opts.seed);
      const exceed = new Float64Array(k * k);
      const maxExceed = new Float64Array(k * k);
      for (let b = 0; b < opts.nSurrogates; b += 1) {
        const offsets = circularShiftOffsets(n, k, separation, rng);
        const surrogate = cols.map((c, v) => roll(c, offsets[v]));
        const sur = laggedCorrelationScores(surrogate, opts.maxLag).scores;
        let familyMax = -Infinity;
        for (let i = 0; i < k * k; i += 1) if (i % (k + 1) !== 0) familyMax = Math.max(familyMax, sur[i]);
        for (let i = 0; i < k * k; i += 1) {
          if (sur[i] >= scores[i]) exceed[i] += 1;
          if (familyMax >= scores[i]) maxExceed[i] += 1;
        }
      }
      for (let i = 0; i < k * k; i += 1) {
        const diagonal = i % (k + 1) === 0;
        pValues[i] = diagonal ? 1 : (exceed[i] + 1) / (opts.nSurrogates + 1);
        adjusted[i] = diagonal ? 1 : (maxExceed[i] + 1) / (opts.nSurrogates + 1);
      }
    }

    const threshold = mean(scores) + std(scores);
    const candidates = [];
    for (let s = 0; s < k; s += 1) {
      for (let d = 0; d < k; d += 1) {
        const i = s * k + d;
        if (s !== d && scores[i] > threshold) {
          candidates.push({ source: s, target: d, weight: scores[i], lag: lags[i],
            p_value: pValues[i], adjusted_p_value: adjusted[i] });
        }
      }
    }
    candidates.sort((a, b) => b.weight - a.weight);
    const significant = candidates.filter(
      (e) => Number.isFinite(e.adjusted_p_value) && e.adjusted_p_value <= opts.alpha,
    );
    return {
      method: 'lagged_correlation',
      scores, lags, p_values: pValues, adjusted_p_values: adjusted,
      edge_threshold: threshold,
      candidate_edges: candidates,
      significant_edges: significant,
      minimum_attainable_p_value: 1 / (opts.nSurrogates + 1),
    };
  }

  return {
    DEPTH_METHOD_VERSION,
    createRng,
    fft,
    welch,
    hilbert,
    unwrap,
    interpolatePeak,
    selectTargetFrequency,
    depth,
    laggedCorrelationScores,
    circularShiftOffsets,
    rooting,
  };
});

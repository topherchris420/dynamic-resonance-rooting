const PRESETS = {
  Coral: { feed: 0.055, kill: 0.062, du: 0.16, dv: 0.08 },
  Spots: { feed: 0.035, kill: 0.065, du: 0.16, dv: 0.08 },
  Waves: { feed: 0.014, kill: 0.054, du: 0.16, dv: 0.08 },
  Stripes: { feed: 0.03, kill: 0.062, du: 0.16, dv: 0.08 }
};

onmessage = (e) => {
  const { size = 64, iterations = 120, preset = 'Coral', seeds = [] } = e.data;
  const n = size * size * size;
  const U = new Float32Array(n).fill(1);
  const V = new Float32Array(n);
  const nextU = new Float32Array(n);
  const nextV = new Float32Array(n);
  const { feed, kill, du, dv } = PRESETS[preset] || PRESETS.Coral;

  for (const s of seeds) {
    const idx = (s.z * size + s.y) * size + s.x;
    V[idx] = 1;
    U[idx] = 0.5;
  }

  const lap = (field, x, y, z) => {
    const idx = (z * size + y) * size + x;
    const xm = x > 0 ? idx - 1 : idx;
    const xp = x < size - 1 ? idx + 1 : idx;
    const ym = y > 0 ? idx - size : idx;
    const yp = y < size - 1 ? idx + size : idx;
    const zm = z > 0 ? idx - size * size : idx;
    const zp = z < size - 1 ? idx + size * size : idx;
    return field[xm] + field[xp] + field[ym] + field[yp] + field[zm] + field[zp] - 6 * field[idx];
  };

  for (let it = 0; it < iterations; it++) {
    for (let z = 1; z < size - 1; z++) {
      for (let y = 1; y < size - 1; y++) {
        for (let x = 1; x < size - 1; x++) {
          const idx = (z * size + y) * size + x;
          const u = U[idx];
          const v = V[idx];
          const uvv = u * v * v;
          nextU[idx] = u + (du * lap(U, x, y, z) - uvv + feed * (1 - u));
          nextV[idx] = v + (dv * lap(V, x, y, z) + uvv - (feed + kill) * v);
        }
      }
    }
    U.set(nextU);
    V.set(nextV);
    if (it % 10 === 0 || it === iterations - 1) {
      postMessage({ type: 'progress', progress: Math.round(((it + 1) / Math.max(1, iterations)) * 100) });
    }
  }

  const threshold = 0.3;
  const active = [];
  const heatSlice = [];
  const z = Math.floor(size / 2);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const idx = (z * size + y) * size + x;
      heatSlice.push(Math.max(0, Math.min(1, V[idx])));
    }
  }
  for (let zz = 1; zz < size - 1; zz++) {
    for (let yy = 1; yy < size - 1; yy++) {
      for (let xx = 1; xx < size - 1; xx++) {
        const idx = (zz * size + yy) * size + xx;
        if (V[idx] > threshold) active.push({ x: xx, y: yy, z: zz, v: V[idx] });
      }
    }
  }

  const edges = [];
  const key = (x, y, z) => `${x}|${y}|${z}`;
  const set = new Set(active.map(p => key(p.x, p.y, p.z)));
  for (const p of active) {
    const nbs = [[1,0,0],[0,1,0],[0,0,1]];
    for (const [dx, dy, dz] of nbs) {
      if (set.has(key(p.x + dx, p.y + dy, p.z + dz))) {
        edges.push([p, { x: p.x + dx, y: p.y + dy, z: p.z + dz }]);
      }
    }
  }

  postMessage({ type: 'done', active, heatSlice, size, edges });
};

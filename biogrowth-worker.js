const PRESETS = {
  Coral: { feed: 0.055, kill: 0.062, du: 0.16, dv: 0.08 },
  Spots: { feed: 0.035, kill: 0.065, du: 0.16, dv: 0.08 },
  Waves: { feed: 0.014, kill: 0.054, du: 0.16, dv: 0.08 },
  Stripes: { feed: 0.030, kill: 0.062, du: 0.16, dv: 0.08 },
};

self.onmessage = (e) => {
  const { type, iterations, preset, seeds } = e.data;
  if (type !== 'run') return;
  const n = 64;
  const size = n * n * n;
  let U = new Float32Array(size).fill(1);
  let V = new Float32Array(size);
  for (const s of seeds || []) {
    const idx = ((s.z * n + s.y) * n + s.x) | 0;
    V[idx] = 1;
    U[idx] = 0.5;
  }
  const p = PRESETS[preset] || PRESETS.Coral;
  const get = (arr, x, y, z) => arr[((z * n + y) * n + x) | 0];
  const dt = 1;

  for (let it = 0; it < iterations; it++) {
    const nextU = U.slice();
    const nextV = V.slice();
    for (let z = 1; z < n - 1; z++) {
      for (let y = 1; y < n - 1; y++) {
        for (let x = 1; x < n - 1; x++) {
          const i = ((z * n + y) * n + x) | 0;
          const u = U[i], v = V[i];
          const lapU = get(U,x+1,y,z)+get(U,x-1,y,z)+get(U,x,y+1,z)+get(U,x,y-1,z)+get(U,x,y,z+1)+get(U,x,y,z-1)-6*u;
          const lapV = get(V,x+1,y,z)+get(V,x-1,y,z)+get(V,x,y+1,z)+get(V,x,y-1,z)+get(V,x,y,z+1)+get(V,x,y,z-1)-6*v;
          const uvv = u * v * v;
          nextU[i] = Math.min(1, Math.max(0, u + (p.du * lapU - uvv + p.feed * (1 - u)) * dt));
          nextV[i] = Math.min(1, Math.max(0, v + (p.dv * lapV + uvv - (p.feed + p.kill) * v) * dt));
        }
      }
    }
    U = nextU;
    V = nextV;
    if (it % 5 === 0 || it === iterations - 1) {
      self.postMessage({ type: 'progress', value: Math.round(((it + 1) / iterations) * 100), slice: buildSlice(V, n, Math.floor(n/2)) });
    }
  }
  const voxels = [];
  for (let i = 0; i < size; i++) if (V[i] > 0.3) voxels.push(i);
  self.postMessage({ type: 'done', voxels, slice: buildSlice(V, n, Math.floor(n/2)) });
};

function buildSlice(V, n, z) {
  const slice = new Float32Array(n * n);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) slice[y*n+x] = V[(z*n+y)*n+x];
  return slice;
}

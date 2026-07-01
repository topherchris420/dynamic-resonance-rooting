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

  // Pre-calculate strides for direct array access
  const strideY = size;
  const strideZ = size * size;

  for (let it = 0; it < iterations; it++) {
    // Iterate over interior points only (1 to size-2)
    for (let z = 1; z < size - 1; z++) {
      let rowIdx = (z * size + 1) * size;
      for (let y = 1; y < size - 1; y++) {
        for (let x = 1; x < size - 1; x++) {
          const idx = rowIdx + x;
          const u = U[idx];
          const v = V[idx];
          const uvv = u * v * v;

          // Optimization: Inline Laplacian calculation
          // Removes function call overhead and redundant boundary checks.
          // Since we iterate strictly over the interior, neighbor indices are always valid.
          const lapU = (
            U[idx - 1] + U[idx + 1] +
            U[idx - strideY] + U[idx + strideY] +
            U[idx - strideZ] + U[idx + strideZ] -
            6 * u
          );

          const lapV = (
            V[idx - 1] + V[idx + 1] +
            V[idx - strideY] + V[idx + strideY] +
            V[idx - strideZ] + V[idx + strideZ] -
            6 * v
          );

          nextU[idx] = u + (du * lapU - uvv + feed * (1 - u));
          nextV[idx] = v + (dv * lapV + uvv - (feed + kill) * v);
        }
        rowIdx += size;
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
  // Optimization: Direct array lookup instead of string Set for neighbor checking
  for (const p of active) {
    const {x, y, z} = p;
    const idx = (z * size + y) * size + x;

    // Check +x neighbor
    if (V[idx + 1] > threshold) {
      edges.push([p, { x: x + 1, y: y, z: z }]);
    }
    // Check +y neighbor
    if (V[idx + strideY] > threshold) {
      edges.push([p, { x: x, y: y + 1, z: z }]);
    }
    // Check +z neighbor
    if (V[idx + strideZ] > threshold) {
      edges.push([p, { x: x, y: y, z: z + 1 }]);
    }
  }

  postMessage({ type: 'done', active, heatSlice, size, edges });
};

const MATERIALS = {
  PLA: { E: 3.5e9, density: 1240, yield: 50e6 },
  ABS: { E: 2.3e9, density: 1050, yield: 40e6 },
  PETG: { E: 2.1e9, density: 1270, yield: 50e6 },
  Resin: { E: 2.8e9, density: 1150, yield: 65e6 }
};

onmessage = (e) => {
  const { nodes = [], edges = [], material = 'PLA', wallThickness = 1.8, infill = 35 } = e.data;
  const m = MATERIALS[material] || MATERIALS.PLA;
  const n = nodes.length;
  if (!n) return postMessage({ type: 'done', error: 'No mesh nodes available.' });

  const mass = new Float64Array(n);
  const force = new Float64Array(n);
  const disp = new Float64Array(n);
  const fixed = new Uint8Array(n);
  const area = Math.max(0.8, wallThickness) * (0.5 + infill / 100) * 1e-6;

  for (let i = 0; i < n; i++) {
    const z = nodes[i].z;
    if (z < 0.01) fixed[i] = 1;
  }

  const diagK = new Float64Array(n);
  const connectivity = Array.from({ length: n }, () => []);
  for (let eIdx = 0; eIdx < edges.length; eIdx++) {
    const [a, b] = edges[eIdx];
    const dx = nodes[a].x - nodes[b].x;
    const dy = nodes[a].y - nodes[b].y;
    const dz = nodes[a].z - nodes[b].z;
    const L = Math.max(1e-3, Math.hypot(dx, dy, dz));
    const k = (m.E * area) / L;
    diagK[a] += k;
    diagK[b] += k;
    connectivity[a].push([b, k, L]);
    connectivity[b].push([a, k, L]);
    const beamMass = m.density * area * L;
    mass[a] += beamMass * 0.5;
    mass[b] += beamMass * 0.5;
  }

  for (let i = 0; i < n; i++) force[i] = fixed[i] ? 0 : -mass[i] * 9.81;

  const multiply = (x, out) => {
    out.fill(0);
    for (let i = 0; i < n; i++) {
      if (fixed[i]) { out[i] = x[i]; continue; }
      let acc = diagK[i] * x[i];
      for (const [j, k] of connectivity[i]) acc -= k * x[j];
      out[i] = acc;
    }
  };

  const b = force;
  const r = Float64Array.from(b);
  const p = Float64Array.from(r);
  const Ap = new Float64Array(n);
  let rsOld = r.reduce((s, v) => s + v * v, 0);
  const maxIter = 160;

  for (let iter = 0; iter < maxIter; iter++) {
    multiply(p, Ap);
    const denom = Math.max(1e-12, Ap.reduce((s, v, i) => s + p[i] * v, 0));
    const alpha = rsOld / denom;
    for (let i = 0; i < n; i++) {
      disp[i] += alpha * p[i];
      r[i] -= alpha * Ap[i];
    }
    const rsNew = r.reduce((s, v) => s + v * v, 0);
    if (iter % 15 === 0) postMessage({ type: 'progress', progress: Math.round((iter / maxIter) * 100) });
    if (Math.sqrt(rsNew) < 1e-6) break;
    const beta = rsNew / Math.max(1e-12, rsOld);
    for (let i = 0; i < n; i++) p[i] = r[i] + beta * p[i];
    rsOld = rsNew;
  }

  let maxStress = 0;
  let maxDisp = 0;
  const stressNode = new Float64Array(n);
  for (const [a, b] of edges) {
    const L = Math.max(1e-3, Math.hypot(nodes[a].x - nodes[b].x, nodes[a].y - nodes[b].y, nodes[a].z - nodes[b].z));
    const strain = Math.abs(disp[a] - disp[b]) / L;
    const stress = m.E * strain;
    maxStress = Math.max(maxStress, stress);
    stressNode[a] = Math.max(stressNode[a], stress);
    stressNode[b] = Math.max(stressNode[b], stress);
  }
  for (const d of disp) maxDisp = Math.max(maxDisp, Math.abs(d));

  const safetyFactor = m.yield / Math.max(1, maxStress);
  const vonMises = maxStress;
  postMessage({
    type: 'done',
    maxDisplacement: maxDisp,
    maxStress: vonMises,
    safetyFactor,
    yieldStrength: m.yield,
    stresses: Array.from(stressNode),
    displacements: Array.from(disp)
  });
};

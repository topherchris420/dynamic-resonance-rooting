const MATERIALS = {
  PLA: { E: 3.5e9, density: 1240, yield: 50e6 },
  ABS: { E: 2.3e9, density: 1050, yield: 40e6 },
  PETG: { E: 2.1e9, density: 1270, yield: 50e6 },
  Resin: { E: 2.8e9, density: 1150, yield: 65e6 },
};

self.onmessage = (e) => {
  if (e.data.type !== 'run') return;
  const { vertices, material, wallThickness, infillDensity } = e.data;
  const mat = MATERIALS[material] || MATERIALS.PLA;
  const n = vertices.length;
  const simplified = n > 10000;
  const sampled = simplified ? vertices.filter((_, i) => i % 3 === 0) : vertices;

  let maxDisp = 0;
  let maxStress = 0;
  const stressMap = new Float32Array(sampled.length);
  for (let i = 0; i < sampled.length; i++) {
    const v = sampled[i];
    const mass = Math.max(1e-6, mat.density * 1e-8 * (wallThickness + infillDensity / 100));
    const force = mass * 9.81 * (1 + Math.abs(v.y));
    const k = mat.E * wallThickness * 1e-6;
    const disp = force / Math.max(1, k);
    const strain = disp / Math.max(1e-3, Math.abs(v.z) + 1e-3);
    const stress = mat.E * strain;
    stressMap[i] = stress;
    if (disp > maxDisp) maxDisp = disp;
    if (stress > maxStress) maxStress = stress;
    if (i % 200 === 0) self.postMessage({ type: 'progress', value: Math.round(i / sampled.length * 100) });
  }

  const safetyFactor = mat.yield / Math.max(maxStress, 1);
  const vonMises = maxStress; // simplified beam assumption
  self.postMessage({
    type: 'done',
    result: {
      maxDisplacementMm: maxDisp * 1000,
      maxStressMpa: maxStress / 1e6,
      safetyFactor,
      vonMisesMpa: vonMises / 1e6,
      yieldMpa: mat.yield / 1e6,
      simplified,
      stressMap: Array.from(stressMap),
      nodes: sampled,
    }
  });
};

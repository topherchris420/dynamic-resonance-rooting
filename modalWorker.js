onmessage = (e) => {
  const { nodes = [], edges = [], stiffnessScale = 1, massScale = 1 } = e.data;
  const nModes = 10;
  let totalL = 0;
  for (const [a, b] of edges) {
    const dx = nodes[a].x - nodes[b].x;
    const dy = nodes[a].y - nodes[b].y;
    const dz = nodes[a].z - nodes[b].z;
    totalL += Math.hypot(dx, dy, dz);
  }
  const geomFactor = Math.max(0.1, totalL / Math.max(1, edges.length));
  const base = (35 * Math.sqrt(stiffnessScale / Math.max(0.01, massScale))) / geomFactor;
  const modes = [];
  for (let i = 1; i <= nModes; i++) {
    const freq = base * i * (1 + Math.sin(i * 0.73) * 0.07);
    modes.push({ mode: i, frequency: Number(freq.toFixed(2)) });
  }
  postMessage({ type: 'done', modes });
};

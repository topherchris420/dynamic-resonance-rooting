onmessage = (e) => {
  const { nodes = [], edges = [], stiffnessScale = 1, massScale = 1 } = e.data;
  const nModes = 10;
  let totalL = 0;

  const flatNodes =
    nodes instanceof Float32Array || nodes instanceof Float64Array
      ? nodes
      : Float32Array.from(nodes.flatMap((node) => [node.x, node.y, node.z]));

  for (let i = 0; i < edges.length; i++) {
    const a = edges[i][0] * 3;
    const b = edges[i][1] * 3;

    const dx = flatNodes[a] - flatNodes[b];
    const dy = flatNodes[a + 1] - flatNodes[b + 1];
    const dz = flatNodes[a + 2] - flatNodes[b + 2];
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

self.onmessage = (e) => {
  if (e.data.type !== 'run') return;
  const { audioSpectrum, stiffnessBase = 100, massBase = 1 } = e.data;
  const modes = [];
  for (let i = 1; i <= 10; i++) {
    const omega = Math.sqrt((stiffnessBase * i * i) / (massBase + i * 0.2));
    const freq = omega / (2 * Math.PI);
    modes.push(freq * 20);
  }
  const overlaps = modes.filter(m => audioSpectrum.some(a => Math.abs(a - m) < 20));
  const score = Math.round((overlaps.length / modes.length) * 100);
  self.postMessage({ type: 'done', modes, score, overlaps });
};

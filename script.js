const reveals = document.querySelectorAll('.reveal');

const observer = new IntersectionObserver(
  entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('is-visible');
        observer.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.16 }
);

reveals.forEach(section => observer.observe(section));

const metrics = document.querySelectorAll('.metric');
const easeOut = t => 1 - Math.pow(1 - t, 3);

function animateMetric(node) {
  const target = Number(node.dataset.target || 0);
  const duration = 1200;
  const started = performance.now();

  function tick(now) {
    const progress = Math.min((now - started) / duration, 1);
    const value = Math.round(target * easeOut(progress));
    node.textContent = String(value);
    if (progress < 1) requestAnimationFrame(tick);
  }

  requestAnimationFrame(tick);
}

const metricObserver = new IntersectionObserver(
  entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const metric = entry.target;
        animateMetric(metric);
        metricObserver.unobserve(metric);
      }
    });
  },
  { threshold: 0.7 }
);

metrics.forEach(metric => metricObserver.observe(metric));

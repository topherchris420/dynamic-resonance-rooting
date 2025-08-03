// Optional enhancement: pulse effect
const axes = document.querySelectorAll('.axis');
let pulse = 0;
function animate() {
  pulse = (pulse + 1) % 360;
  const glow = `0 0 15px 3px hsla(${pulse}, 100%, 70%, 0.8)`;
  axes.forEach(axis => {
    axis.style.boxShadow = glow;
  });
  requestAnimationFrame(animate);
}
animate();

import { useRef, useEffect } from 'react';

const W = 256;
const H = 128;

export default function RadialProfileCanvas({ radialProfile, spectralEntropy }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !radialProfile) return;

    requestAnimationFrame(() => {
      canvas.width = W;
      canvas.height = H;
      const ctx = canvas.getContext('2d');

      // Background
      ctx.fillStyle = '#0a0a0a';
      ctx.fillRect(0, 0, W, H);

      // Grid lines
      ctx.strokeStyle = 'rgba(255,255,255,0.05)';
      ctx.lineWidth = 1;
      for (const level of [0.25, 0.5, 0.75]) {
        const y = H - level * H;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(W, y);
        ctx.stroke();
      }

      // Find max for normalisation to [0,1]
      let maxVal = 0;
      for (let i = 0; i < radialProfile.length; i++) {
        if (radialProfile[i] > maxVal) maxVal = radialProfile[i];
      }
      if (maxVal < 1e-30) maxVal = 1;

      // Polyline
      ctx.strokeStyle = '#00e5a0';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      const len = radialProfile.length;
      for (let i = 0; i < len; i++) {
        const x = (i / (len - 1)) * W;
        const y = H - (radialProfile[i] / maxVal) * H;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Entropy label
      ctx.fillStyle = '#00e5a0';
      ctx.font = '10px Geist Mono, monospace';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'top';
      ctx.fillText(`H: ${spectralEntropy.toFixed(3)}`, W - 4, 4);
    });
  }, [radialProfile, spectralEntropy]);

  if (!radialProfile) {
    return (
      <div style={{ width: '100%', height: H, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-dim)', fontSize: 10, background: 'var(--bg)' }}>
        WAITING FOR DATA...
      </div>
    );
  }

  return (
    <div>
      <canvas
        ref={canvasRef}
        style={{ width: '100%', display: 'block', background: 'var(--bg)' }}
      />
      <div style={{ fontSize: 9, color: 'var(--text-dim)', marginTop: 4, letterSpacing: '0.05em' }}>
        LOW FREQ &mdash;&mdash;&mdash;&mdash;&mdash;&mdash;&mdash;&mdash; HIGH FREQ
      </div>
    </div>
  );
}

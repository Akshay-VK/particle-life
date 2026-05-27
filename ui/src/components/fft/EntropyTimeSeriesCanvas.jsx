import { useRef, useEffect } from 'react';

const W = 256;
const H = 64;

export default function EntropyTimeSeriesCanvas({ entropyHistory }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !entropyHistory || entropyHistory.length < 2) return;

    requestAnimationFrame(() => {
      canvas.width = W;
      canvas.height = H;
      const ctx = canvas.getContext('2d');

      ctx.fillStyle = '#0a0a0a';
      ctx.fillRect(0, 0, W, H);

      // Reference lines
      ctx.strokeStyle = 'rgba(255,255,255,0.05)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, H * 0.75);
      ctx.lineTo(W, H * 0.75);
      ctx.moveTo(0, H * 0.25);
      ctx.lineTo(W, H * 0.25);
      ctx.stroke();

      // Fill under the line
      ctx.beginPath();
      const len = entropyHistory.length;
      const stepX = W / (len - 1);
      ctx.moveTo(0, H);
      for (let i = 0; i < len; i++) {
        const x = i * stepX;
        const y = H - entropyHistory[i] * H;
        ctx.lineTo(x, y);
      }
      ctx.lineTo((len - 1) * stepX, H);
      ctx.closePath();
      ctx.fillStyle = 'rgba(0, 229, 160, 0.06)';
      ctx.fill();

      // Line
      ctx.strokeStyle = '#00e5a0';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 0; i < len; i++) {
        const x = i * stepX;
        const y = H - entropyHistory[i] * H;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Axis labels
      ctx.fillStyle = '#555555';
      ctx.font = '9px Geist Mono, monospace';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText('1.0', 1, 1);
      ctx.textBaseline = 'bottom';
      ctx.fillText('0.0', 1, H - 1);
    });
  }, [entropyHistory]);

  if (!entropyHistory || entropyHistory.length < 2) {
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
      <div style={{ fontSize: 9, color: 'var(--text-dim)', marginTop: 4 }}>
        SPECTRAL ENTROPY  (60s)
      </div>
    </div>
  );
}

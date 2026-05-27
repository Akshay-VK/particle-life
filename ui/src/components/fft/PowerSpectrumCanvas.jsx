import { useRef, useEffect, useMemo } from 'react';

const INFERNO = [
  [0, 0, 4],
  [40, 11, 84],
  [187, 55, 84],
  [249, 142, 9],
  [252, 255, 164],
];

function lerpColor(t) {
  const clamped = Math.max(0, Math.min(1, t));
  const seg = clamped * (INFERNO.length - 1);
  const idx = Math.floor(seg);
  const frac = seg - idx;
  const a = INFERNO[idx];
  const b = INFERNO[Math.min(idx + 1, INFERNO.length - 1)];
  return [
    Math.round(a[0] + (b[0] - a[0]) * frac),
    Math.round(a[1] + (b[1] - a[1]) * frac),
    Math.round(a[2] + (b[2] - a[2]) * frac),
  ];
}

export default function PowerSpectrumCanvas({ powerSpectrum2d, gridSize }) {
  const canvasRef = useRef(null);
  const luts = useMemo(() => {
    const t = new Uint8Array(256 * 3);
    for (let i = 0; i < 256; i++) {
      const [r, g, b] = lerpColor(i / 255);
      t[i * 3] = r;
      t[i * 3 + 1] = g;
      t[i * 3 + 2] = b;
    }
    return t;
  }, []);

  const maxPower = useMemo(() => {
    if (!powerSpectrum2d) return 0;
    let mx = 0;
    for (let i = 0; i < powerSpectrum2d.length; i++) {
      if (powerSpectrum2d[i] > mx) mx = powerSpectrum2d[i];
    }
    return mx;
  }, [powerSpectrum2d]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !powerSpectrum2d) return;

    const W = gridSize || 256;
    const H = W;

    requestAnimationFrame(() => {
      canvas.width = W;
      canvas.height = H;
      const ctx = canvas.getContext('2d');
      const imageData = ctx.createImageData(W, H);
      const pixels = imageData.data;

      let maxVal = 0;
      for (let i = 0; i < powerSpectrum2d.length; i++) {
        if (powerSpectrum2d[i] > maxVal) maxVal = powerSpectrum2d[i];
      }
      if (maxVal < 1e-30) maxVal = 1;

      for (let r = 0; r < H; r++) {
        for (let c = 0; c < W; c++) {
          const idx = r * W + c;
          const norm = Math.min(1, powerSpectrum2d[idx] / maxVal);
          const lutIdx = Math.floor(norm * 255);
          const base = lutIdx * 3;
          const pixelOff = idx * 4;
          pixels[pixelOff] = luts[base];
          pixels[pixelOff + 1] = luts[base + 1];
          pixels[pixelOff + 2] = luts[base + 2];
          pixels[pixelOff + 3] = 255;
        }
      }

      ctx.putImageData(imageData, 0, 0);

      // Centre crosshair
      const cx = W / 2;
      const cy = H / 2;
      ctx.strokeStyle = 'rgba(255,255,255,0.2)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, cy);
      ctx.lineTo(W, cy);
      ctx.moveTo(cx, 0);
      ctx.lineTo(cx, H);
      ctx.stroke();
    });
  }, [powerSpectrum2d, gridSize, luts]);

  if (!powerSpectrum2d) {
    return (
      <div style={{ width: '100%', height: 256, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-dim)', fontSize: 10, background: 'var(--bg)' }}>
        WAITING FOR DATA...
      </div>
    );
  }

  return (
    <div>
      <canvas
        ref={canvasRef}
        style={{ width: '100%', imageRendering: 'pixelated', background: 'var(--bg)' }}
      />
      <div style={{ fontSize: 9, color: 'var(--text-dim)', marginTop: 4 }}>
        POWER SPECTRUM  &mdash;  max: {maxPower.toExponential(2)}
      </div>
    </div>
  );
}

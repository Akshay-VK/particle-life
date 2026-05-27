import { useRef, useEffect } from 'react';
import { drawPowerSpectrumToCanvas } from './fftDrawUtils';

export default function PowerSpectrumCanvas({ powerSpectrum2d, gridSize }) {
  const canvasRef = useRef(null);
  const size = gridSize || 256;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !powerSpectrum2d) return;

    requestAnimationFrame(() => {
      drawPowerSpectrumToCanvas(canvas, powerSpectrum2d, size);
    });
  }, [powerSpectrum2d, size]);

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
        POWER SPECTRUM  &mdash;  normalised
      </div>
    </div>
  );
}

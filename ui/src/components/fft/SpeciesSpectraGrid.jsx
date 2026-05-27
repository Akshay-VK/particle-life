import { useRef, useEffect, useMemo } from 'react';
import { drawPowerSpectrumToCanvas } from './fftDrawUtils';

const CELL_SIZE = 80;
const FULL_GRID = 256;

export default function SpeciesSpectraGrid({ channelPowers, channelEntropies, numSpecies }) {
  const nCh = numSpecies + 1;
  const cols = 3;
  const rows = Math.ceil(nCh / cols);

  if (!channelPowers || numSpecies === undefined) return null;

  return (
    <div style={{ display: 'grid', gridTemplateColumns: `repeat(${cols}, ${CELL_SIZE}px)`, gap: 8 }}>
      {Array.from({ length: nCh }, (_, i) => {
        const idx = 1 + i;
        const label = i < numSpecies ? `S${i}` : 'STATE';
        const ent = channelEntropies ? channelEntropies[idx] : 0;
        return (
          <SpeciesCell
            key={idx}
            powerData={channelPowers[idx]}
            label={label}
            entropy={ent}
          />
        );
      })}
    </div>
  );
}

function SpeciesCell({ powerData, label, entropy }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !powerData) return;
    const offscreen = document.createElement('canvas');
    offscreen.width = FULL_GRID;
    offscreen.height = FULL_GRID;
    drawPowerSpectrumToCanvas(offscreen, powerData, FULL_GRID);
    requestAnimationFrame(() => {
      canvas.width = CELL_SIZE;
      canvas.height = CELL_SIZE;
      const ctx = canvas.getContext('2d');
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(offscreen, 0, 0, CELL_SIZE, CELL_SIZE);
    });
  }, [powerData]);

  return (
    <div>
      <canvas
        ref={canvasRef}
        style={{ width: CELL_SIZE, height: CELL_SIZE, imageRendering: 'pixelated', background: 'var(--bg)', display: 'block' }}
      />
      <div style={{ fontSize: 9, color: 'var(--text-dim)', marginTop: 2 }}>
        {label}
      </div>
      <div style={{ fontSize: 9, color: 'var(--text-dim)' }}>
        H: {entropy.toFixed(2)}
      </div>
    </div>
  );
}

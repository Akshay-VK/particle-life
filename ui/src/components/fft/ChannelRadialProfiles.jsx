import { useRef, useEffect } from 'react';

const W = 256;
const H = 128;

const SPECIES_COLORS = [
  '#e05', '#0e5', '#05e', '#ee0', '#e50', '#e05', '#e0e', '#0ee',
];

export default function ChannelRadialProfiles({ channelRadialProfiles, channelEntropies, numSpecies }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !channelRadialProfiles || numSpecies === undefined) return;

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

      // Find global max across all channels for consistent scaling
      let maxVal = 0;
      for (let ch = 0; ch < numSpecies + 2; ch++) {
        const prof = channelRadialProfiles[ch];
        if (!prof) continue;
        for (let i = 0; i < prof.length; i++) {
          if (prof[i] > maxVal) maxVal = prof[i];
        }
      }
      if (maxVal < 1e-30) maxVal = 1;

      const len = channelRadialProfiles[0] ? channelRadialProfiles[0].length : 128;

      // All-particle channel (dim reference)
      if (channelRadialProfiles[0]) {
        ctx.strokeStyle = 'rgba(255,255,255,0.3)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let i = 0; i < len; i++) {
          const x = (i / (len - 1)) * W;
          const y = H - (channelRadialProfiles[0][i] / maxVal) * H;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }

      // Species channels
      const nCh = numSpecies + 1;
      for (let ch = 1; ch < nCh; ch++) {
        const prof = channelRadialProfiles[ch];
        if (!prof) continue;
        ctx.strokeStyle = SPECIES_COLORS[(ch - 1) % SPECIES_COLORS.length];
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let i = 0; i < len; i++) {
          const x = (i / (len - 1)) * W;
          const y = H - (prof[i] / maxVal) * H;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }

      // State-weighted channel
      const stateCh = numSpecies + 1;
      if (channelRadialProfiles[stateCh]) {
        ctx.strokeStyle = '#00e5a0';
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let i = 0; i < len; i++) {
          const x = (i / (len - 1)) * W;
          const y = H - (channelRadialProfiles[stateCh][i] / maxVal) * H;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }

      // Legend (bottom-right)
      const legendX = W - 50;
      let legendY = H - 12;
      ctx.font = '8px Geist Mono, monospace';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'bottom';

      // All
      ctx.fillStyle = 'rgba(255,255,255,0.3)';
      ctx.fillText('ALL', legendX, legendY);
      legendY -= 10;

      // State
      ctx.fillStyle = '#00e5a0';
      ctx.fillText('STATE', legendX, legendY);
      legendY -= 10;

      // Species (last few)
      for (let i = Math.min(numSpecies - 1, 3); i >= 0; i--) {
        ctx.fillStyle = SPECIES_COLORS[i % SPECIES_COLORS.length];
        ctx.fillText(`S${i}`, legendX, legendY);
        legendY -= 10;
      }
    });
  }, [channelRadialProfiles, numSpecies]);

  if (!channelRadialProfiles || numSpecies === undefined) {
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

const INFERNO_CTRL = [
  [0, 0, 4],
  [40, 11, 84],
  [187, 55, 84],
  [249, 142, 9],
  [252, 255, 164],
];

function lerpColor(t) {
  const clamped = Math.max(0, Math.min(1, t));
  const seg = clamped * (INFERNO_CTRL.length - 1);
  const idx = Math.floor(seg);
  const frac = seg - idx;
  const a = INFERNO_CTRL[idx];
  const b = INFERNO_CTRL[Math.min(idx + 1, INFERNO_CTRL.length - 1)];
  return [
    Math.round(a[0] + (b[0] - a[0]) * frac),
    Math.round(a[1] + (b[1] - a[1]) * frac),
    Math.round(a[2] + (b[2] - a[2]) * frac),
  ];
}

const INFERNO_LUT = (() => {
  const t = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const [r, g, b] = lerpColor(i / 255);
    t[i * 3] = r;
    t[i * 3 + 1] = g;
    t[i * 3 + 2] = b;
  }
  return t;
})();

export function drawPowerSpectrumToCanvas(canvas, powerData, gridSize) {
  const W = gridSize || 256;
  const H = W;

  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext('2d');
  const imageData = ctx.createImageData(W, H);
  const pixels = imageData.data;

  for (let r = 0; r < H; r++) {
    for (let c = 0; c < W; c++) {
      const idx = r * W + c;
      const val = powerData[idx];
      const norm = val < 0 ? 0 : val > 1 ? 1 : val;
      const lutIdx = Math.floor(norm * 255);
      const base = lutIdx * 3;
      const pixelOff = idx * 4;
      pixels[pixelOff] = INFERNO_LUT[base];
      pixels[pixelOff + 1] = INFERNO_LUT[base + 1];
      pixels[pixelOff + 2] = INFERNO_LUT[base + 2];
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
}

export function approxPercentile99(arr) {
  let maxVal = 0;
  for (let i = 0; i < arr.length; i++) if (arr[i] > maxVal) maxVal = arr[i];
  if (maxVal < 1e-10) return 1.0;

  const bins = 256;
  const hist = new Uint32Array(bins);
  for (let i = 0; i < arr.length; i++) {
    const b = Math.min(bins - 1, Math.floor((arr[i] / maxVal) * bins));
    hist[b]++;
  }
  const target = Math.floor(arr.length * 0.99);
  let cum = 0;
  for (let b = 0; b < bins; b++) {
    cum += hist[b];
    if (cum >= target) return (b / bins) * maxVal;
  }
  return maxVal;
}

import { useState, useEffect, useRef } from "react";
import FFT from "fft.js";

const GRID = 256;
const HALF = GRID / 2;
const LOG2_GRID2 = Math.log2(GRID * GRID);
const POLL_MS = 500;
const MAX_ENTROPY_HISTORY = 120;

export function useFftAnalysis() {
  const entropyHistoryRef = useRef([]);
  const fftRef = useRef(null);

  // Pre-allocated buffers reused each tick — never recreated
  const grid = useRef(new Float32Array(GRID * GRID));
  const real = useRef(new Float32Array(GRID * GRID));
  const imag = useRef(new Float32Array(GRID * GRID));
  const power = useRef(new Float32Array(GRID * GRID));
  const rowIn = useRef(new Float32Array(GRID * 2));
  const rowOut = useRef(new Float32Array(GRID * 2));
  const colIn = useRef(new Float32Array(GRID * 2));
  const colOut = useRef(new Float32Array(GRID * 2));
  const rProfile = useRef(new Float32Array(HALF));
  const bCount = useRef(new Float32Array(HALF));
  const hann = useRef(null);

  const [result, setResult] = useState(null);

  useEffect(() => {
    fftRef.current = new FFT(GRID);

    const h = new Float32Array(GRID);
    for (let i = 0; i < GRID; i++) {
      h[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (GRID - 1)));
    }
    hann.current = h;

    let cancelled = false;

    const tick = async () => {
      if (cancelled) return;
      try {
        const res = await fetch("/api/snapshot");
        if (cancelled) return;
        const data = await res.json();
        if (cancelled) return;

        const { particles, world_size } = data;
        const fftResult = computeFft(particles, world_size);

        const history = entropyHistoryRef.current;
        history.push(fftResult.spectralEntropy);
        if (history.length > MAX_ENTROPY_HISTORY) history.shift();
        setResult({
          powerSpectrum2d: fftResult.powerSpectrum2d,
          radialProfile: fftResult.radialProfile,
          spectralEntropy: fftResult.spectralEntropy,
          entropyHistory: [...history],
          gridSize: GRID,
        });
      } catch (_) {
        // connection not ready yet
      }
      setTimeout(tick, POLL_MS);
    };

    tick();
    return () => {
      cancelled = true;
    };
  }, []);

  function computeFft(particles, worldSize) {
    const g = grid.current;
    const realBuf = real.current;
    const imagBuf = imag.current;
    const p = power.current;
    const h = hann.current;
    const fft = fftRef.current;
    // Step 1 — fill density grid
    g.fill(0);
    for (let idx = 0; idx < particles.length; idx++) {
      const [x, y] = particles[idx];
      const col = Math.floor((x / worldSize) * GRID);
      const row = Math.floor((y / worldSize) * GRID);
      if (col >= 0 && col < GRID && row >= 0 && row < GRID) {
        g[row * GRID + col] += 1;
      }
    }

    // Step 2 — DC removal
    const mean = g.reduce((a, b) => a + b, 0) / (GRID * GRID);
    for (let i = 0; i < g.length; i++) g[i] -= mean;

    // Step 3 — 2D Hann window
    for (let r = 0; r < GRID; r++) {
      const hr = h[r];
      for (let c = 0; c < GRID; c++) {
        g[r * GRID + c] *= hr * h[c];
      }
    }

    const ri = rowIn.current;
    const ro = rowOut.current;
    const ci = colIn.current;
    const co = colOut.current;

    // Step 4 — Row-wise FFT
    for (let r = 0; r < GRID; r++) {
      const off = r * GRID;
      for (let c = 0; c < GRID; c++) {
        ri[2 * c] = g[off + c];
        ri[2 * c + 1] = 0;
      }
      fft.transform(ro, ri);
      for (let c = 0; c < GRID; c++) {
        realBuf[off + c] = ro[2 * c];
        imagBuf[off + c] = ro[2 * c + 1];
      }
    }

    // Step 5 — Column-wise FFT
    for (let c = 0; c < GRID; c++) {
      for (let r = 0; r < GRID; r++) {
        ci[2 * r] = realBuf[r * GRID + c];
        ci[2 * r + 1] = imagBuf[r * GRID + c];
      }
      fft.transform(co, ci);
      for (let r = 0; r < GRID; r++) {
        realBuf[r * GRID + c] = co[2 * r];
        imagBuf[r * GRID + c] = co[2 * r + 1];
      }
    }

    // Step 6 — Power spectrum
    for (let i = 0; i < p.length; i++) {
      p[i] = realBuf[i] * realBuf[i] + imagBuf[i] * imagBuf[i];
    }

    // Step 7 — FFT shift (DC to centre)
    for (let r = 0; r < HALF; r++) {
      for (let c = 0; c < HALF; c++) {
        const a = r * GRID + c;
        const b = (r + HALF) * GRID + (c + HALF);
        let tmp = p[a];
        p[a] = p[b];
        p[b] = tmp;

        const c2 = r * GRID + (c + HALF);
        const d2 = (r + HALF) * GRID + c;
        tmp = p[c2];
        p[c2] = p[d2];
        p[d2] = tmp;
      }
    }

    // Step 8 — Spectral entropy (on raw shifted power BEFORE log)
    const total = p.reduce((a, b) => a + b, 0);
    let entropy = 0;
    if (total > 1e-30) {
      for (let i = 0; i < p.length; i++) {
        const prob = p[i] / total;
        if (prob > 1e-10) entropy -= prob * Math.log2(prob);
      }
    }
    entropy /= LOG2_GRID2;

    // Step 9 — Log scale (for display)
    for (let i = 0; i < p.length; i++) {
      p[i] = Math.log1p(p[i]);
    }

    // Step 10 — Radial profile
    const rp = rProfile.current;
    const bc = bCount.current;
    rp.fill(0);
    bc.fill(0);
    const cx = HALF,
      cy = HALF;
    for (let r = 0; r < GRID; r++) {
      for (let c = 0; c < GRID; c++) {
        const dr = Math.sqrt((r - cy) ** 2 + (c - cx) ** 2);
        const bin = Math.floor(dr);
        if (bin < HALF) {
          rp[bin] += p[r * GRID + c];
          bc[bin] += 1;
        }
      }
    }
    for (let i = 0; i < HALF; i++) {
      if (bc[i] > 0) rp[i] /= bc[i];
    }

    return {
      powerSpectrum2d: new Float32Array(p),
      radialProfile: new Float32Array(rp),
      spectralEntropy: entropy,
    };
  }

  return result;
}

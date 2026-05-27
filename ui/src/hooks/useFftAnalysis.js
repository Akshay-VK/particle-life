import { useState, useEffect, useRef } from "react";
import FFT from "fft.js";
import { approxPercentile99 } from "../components/fft/fftDrawUtils";

const GRID = 256;
const HALF = GRID / 2;
const LOG2_GRID2 = Math.log2(GRID * GRID);
const POLL_MS = 500;
const MAX_ENTROPY_HISTORY = 120;
const MAX_SPECIES = 8;

export function useFftAnalysis() {
  const entropyHistoryRef = useRef([]);
  const fftRef = useRef(null);

  // Pre-allocated buffers — never recreated
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

  // Multi-channel buffers
  const speciesGrids = useRef(
    Array.from({ length: MAX_SPECIES }, () => new Float32Array(GRID * GRID))
  );
  const stateGrid = useRef(new Float32Array(GRID * GRID));
  const channelPowers = useRef(
    Array.from({ length: MAX_SPECIES + 2 }, () => new Float32Array(GRID * GRID))
  );
  const channelRadialProfiles = useRef(
    Array.from({ length: MAX_SPECIES + 2 }, () => new Float32Array(HALF))
  );
  const channelEntropies = useRef(new Float32Array(MAX_SPECIES + 2));

  const [result, setResult] = useState(null);

  function computeChannelFft(inputGrid, powerOut, radialOut, h, fft, realBuf, imagBuf, ri, ro, ci, co, rp, bc) {
    const N = GRID;

    // Step 1 — DC removal
    let sum = 0;
    for (let i = 0; i < inputGrid.length; i++) sum += inputGrid[i];
    const mean = sum / inputGrid.length;
    for (let i = 0; i < inputGrid.length; i++) inputGrid[i] -= mean;

    // Step 2 — 2D Hann window
    for (let r = 0; r < N; r++) {
      const hr = h[r];
      for (let c = 0; c < N; c++) {
        inputGrid[r * N + c] *= hr * h[c];
      }
    }

    // Step 3 — Row-wise FFT
    for (let r = 0; r < N; r++) {
      const off = r * N;
      for (let c = 0; c < N; c++) {
        ri[2 * c] = inputGrid[off + c];
        ri[2 * c + 1] = 0;
      }
      fft.transform(ro, ri);
      for (let c = 0; c < N; c++) {
        realBuf[off + c] = ro[2 * c];
        imagBuf[off + c] = ro[2 * c + 1];
      }
    }

    // Step 4 — Column-wise FFT
    for (let c = 0; c < N; c++) {
      for (let r = 0; r < N; r++) {
        ci[2 * r] = realBuf[r * N + c];
        ci[2 * r + 1] = imagBuf[r * N + c];
      }
      fft.transform(co, ci);
      for (let r = 0; r < N; r++) {
        realBuf[r * N + c] = co[2 * r];
        imagBuf[r * N + c] = co[2 * r + 1];
      }
    }

    // Step 5 — Power spectrum
    for (let i = 0; i < powerOut.length; i++) {
      powerOut[i] = realBuf[i] * realBuf[i] + imagBuf[i] * imagBuf[i];
    }

    // Step 6 — FFT shift (DC to centre)
    for (let r = 0; r < HALF; r++) {
      for (let c = 0; c < HALF; c++) {
        const a = r * N + c;
        const b = (r + HALF) * N + (c + HALF);
        let tmp = powerOut[a];
        powerOut[a] = powerOut[b];
        powerOut[b] = tmp;

        const c2 = r * N + (c + HALF);
        const d2 = (r + HALF) * N + c;
        tmp = powerOut[c2];
        powerOut[c2] = powerOut[d2];
        powerOut[d2] = tmp;
      }
    }

    // Step 7 — Zero DC bin (centre pixel after shift)
    powerOut[HALF * N + HALF] = 0;

    // Step 8 — Spectral entropy (on raw shifted power BEFORE log)
    let total = 0;
    for (let i = 0; i < powerOut.length; i++) total += powerOut[i];
    let entropy = 0;
    if (total > 1e-30) {
      for (let i = 0; i < powerOut.length; i++) {
        const prob = powerOut[i] / total;
        if (prob > 1e-10) entropy -= prob * Math.log2(prob);
      }
    }
    entropy /= LOG2_GRID2;

    // Step 9 — Log scale (for display)
    for (let i = 0; i < powerOut.length; i++) {
      powerOut[i] = Math.log1p(powerOut[i]);
    }

    // Step 10 — Percentile normalisation to [0, 1]
    const p99 = approxPercentile99(powerOut);
    const invP99 = p99 > 1e-10 ? 1.0 / p99 : 1.0;
    for (let i = 0; i < powerOut.length; i++) {
      powerOut[i] = Math.min(1.0, powerOut[i] * invP99);
    }

    // Step 11 — Radial profile
    rp.fill(0);
    bc.fill(0);
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        const dr = Math.sqrt((r - HALF) ** 2 + (c - HALF) ** 2);
        const bin = Math.floor(dr);
        if (bin < HALF) {
          rp[bin] += powerOut[r * N + c];
          bc[bin] += 1;
        }
      }
    }
    for (let i = 0; i < HALF; i++) {
      if (bc[i] > 0) radialOut[i] = rp[i] / bc[i];
    }

    return entropy;
  }

  function computeFft(particles, worldSize, numSpecies) {
    const g = grid.current;
    const realBuf = real.current;
    const imagBuf = imag.current;
    const p = power.current;
    const h = hann.current;
    const fft = fftRef.current;
    const ri = rowIn.current;
    const ro = rowOut.current;
    const ci = colIn.current;
    const co = colOut.current;
    const rp = rProfile.current;
    const bc = bCount.current;

    // --- Fill all-particle density grid ---
    g.fill(0);
    for (let idx = 0; idx < particles.length; idx++) {
      const [x, y] = particles[idx];
      const col = Math.floor((x / worldSize) * GRID);
      const row = Math.floor((y / worldSize) * GRID);
      if (col >= 0 && col < GRID && row >= 0 && row < GRID) {
        g[row * GRID + col] += 1;
      }
    }

    // --- Fill species grids and state grid (single pass) ---
    const numActiveSpecies = Math.min(numSpecies, MAX_SPECIES);
    const sgs = speciesGrids.current;
    const sg = stateGrid.current;
    sg.fill(0);
    for (let s = 0; s < numActiveSpecies; s++) sgs[s].fill(0);

    for (let idx = 0; idx < particles.length; idx++) {
      const px = particles[idx][0];
      const py = particles[idx][1];
      const species = Math.round(particles[idx][2]);
      const state = particles[idx][3];

      const col = Math.floor((px / worldSize) * GRID);
      const row = Math.floor((py / worldSize) * GRID);
      if (col >= 0 && col < GRID && row >= 0 && row < GRID) {
        const cellIdx = row * GRID + col;
        if (species >= 0 && species < MAX_SPECIES) {
          sgs[species][cellIdx] += 1;
        }
        sg[cellIdx] += state;
      }
    }

    // --- Process channels ---
    const nChannels = 1 + numActiveSpecies + 1; // all + species + state
    const cps = channelPowers.current;
    const crps = channelRadialProfiles.current;
    const ces = channelEntropies.current;

    // Channel 0: all-particle
    ces[0] = computeChannelFft(g, cps[0], crps[0], h, fft, realBuf, imagBuf, ri, ro, ci, co, rp, bc);

    // Channels 1..numActiveSpecies: per-species
    for (let s = 0; s < numActiveSpecies; s++) {
      const ch = 1 + s;
      ces[ch] = computeChannelFft(sgs[s], cps[ch], crps[ch], h, fft, realBuf, imagBuf, ri, ro, ci, co, rp, bc);
    }

    // Last channel: state-weighted
    const stateCh = 1 + numActiveSpecies;
    ces[stateCh] = computeChannelFft(sg, cps[stateCh], crps[stateCh], h, fft, realBuf, imagBuf, ri, ro, ci, co, rp, bc);

    const history = entropyHistoryRef.current;
    history.push(ces[0]);
    if (history.length > MAX_ENTROPY_HISTORY) history.shift();

    return {
      powerSpectrum2d: new Float32Array(cps[0]),
      radialProfile: new Float32Array(crps[0]),
      spectralEntropy: ces[0],
      entropyHistory: [...history],
      gridSize: GRID,
      numSpecies: numActiveSpecies,
      channelPowers: Array.from({ length: nChannels }, (_, i) => new Float32Array(cps[i])),
      channelRadialProfiles: Array.from({ length: nChannels }, (_, i) => new Float32Array(crps[i])),
      channelEntropies: Array.from(ces.slice(0, nChannels)),
    };
  }

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

        const { particles, world_size, num_species } = data;
        const fftResult = computeFft(particles, world_size, num_species);

        setResult(fftResult);
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

  return result;
}

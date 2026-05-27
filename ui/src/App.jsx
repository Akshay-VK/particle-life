import React, { useState, useEffect, useRef, useCallback } from 'react';
import MatrixEditor from './components/MatrixEditor';
import SpeciesPanel from './components/SpeciesPanel';
import ParamSlider from './components/ParamSlider';
import SimControls from './components/SimControls';
import Presets from './components/Presets';
import PowerSpectrumCanvas from './components/fft/PowerSpectrumCanvas';
import RadialProfileCanvas from './components/fft/RadialProfileCanvas';
import EntropyTimeSeriesCanvas from './components/fft/EntropyTimeSeriesCanvas';
import { useFftAnalysis } from './hooks/useFftAnalysis';

const API = '/api';

function toServer(payload) {
  fetch(`${API}/params`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  }).catch(() => {});
}

const DEBOUNCE_MS = 50;

export default function App() {
  const [state, setState] = useState(null);
  const [tab, setTab] = useState('CONTROLS');
  const debounceTimers = useRef({});
  const fftResult = useFftAnalysis();

  // Initial fetch
  useEffect(() => {
    fetch(`${API}/state`)
      .then((r) => r.json())
      .then((data) => setState(data))
      .catch(() => {});
  }, []);

  // Poll every 2s
  useEffect(() => {
    if (!state) return;
    const id = setInterval(() => {
      fetch(`${API}/state`)
        .then((r) => r.json())
        .then((data) => setState((prev) => ({ ...prev, ...data })))
        .catch(() => {});
    }, 2000);
    return () => clearInterval(id);
  }, [state !== null]);

  const updateParam = useCallback((key, value) => {
    setState((prev) => {
      if (!prev) return prev;
      return { ...prev, [key]: value };
    });

    if (debounceTimers.current[key]) {
      clearTimeout(debounceTimers.current[key]);
    }
    debounceTimers.current[key] = setTimeout(() => {
      toServer({ [key]: value });
    }, DEBOUNCE_MS);
  }, []);

  const updateMultiple = useCallback((payload) => {
    setState((prev) => {
      if (!prev) return prev;
      return { ...prev, ...payload };
    });
    toServer(payload);
  }, []);

  if (!state) {
    return (
      <div className="app">
        <div className="stream-panel">
          <div className="stream-placeholder">STREAM</div>
        </div>
        <div className="control-panel">
          <div className="panel-section" style={{ color: 'var(--text-dim)', fontSize: 10 }}>
            Connecting...
          </div>
        </div>
      </div>
    );
  }

  const n = state.num_species;
  const interactionMatrix = state.interaction_matrix || new Array(n * n).fill(0);
  const stateTransferMatrix = state.state_transfer_matrix || new Array(n * n).fill(0);
  const speciesColors = state.species_colors || [];

  const handleMatrixChange = (key) => (arr) => {
    setState((prev) => {
      if (!prev) return prev;
      return { ...prev, [key]: arr };
    });

    if (debounceTimers.current[key]) {
      clearTimeout(debounceTimers.current[key]);
    }
    debounceTimers.current[key] = setTimeout(() => {
      toServer({ [key]: arr });
    }, DEBOUNCE_MS);
  };

  return (
    <div className="app">
      <div className="stream-panel">
        <div className="stream-placeholder">STREAM</div>
      </div>

      <div className="control-panel">
        {/* Tab bar */}
        <div style={{ borderBottom: '1px solid var(--border)', display: 'flex' }}>
          <button
            className={`tab${tab === 'CONTROLS' ? ' active' : ''}`}
            onClick={() => setTab('CONTROLS')}
          >
            CONTROLS
          </button>
          <button
            className={`tab${tab === 'ANALYSIS' ? ' active' : ''}`}
            onClick={() => setTab('ANALYSIS')}
          >
            ANALYSIS
          </button>
        </div>

        {tab === 'CONTROLS' ? (
          <>
            {/* Interaction Matrix */}
            <div className="panel-section">
              <div className="section-header">INTERACTION MATRIX  i→j</div>
              <MatrixEditor
                label=""
                matrix={interactionMatrix}
                size={n}
                onChange={handleMatrixChange('interaction_matrix')}
              />
            </div>

            {/* State Transfer */}
            <div className="panel-section">
              <div className="section-header">STATE TRANSFER  i→j</div>
              <MatrixEditor
                label=""
                matrix={stateTransferMatrix}
                size={n}
                onChange={handleMatrixChange('state_transfer_matrix')}
              />
            </div>

            {/* Species */}
            <div className="panel-section">
              <div className="section-header">SPECIES</div>
              <SpeciesPanel
                species={n}
                colors={speciesColors}
                counts={new Array(n).fill('—')}
                onChange={(colors) => updateParam('species_colors', colors)}
              />
            </div>

            {/* Parameters */}
            <div className="panel-section">
              <div className="section-header">PARAMETERS</div>
              <ParamSlider
                label="DT"
                value={state.dt}
                min={0.0005}
                max={0.005}
                step={0.0001}
                onChange={(v) => updateParam('dt', v)}
              />
              <ParamSlider
                label="FRICTION"
                value={state.friction}
                min={0.01}
                max={0.5}
                step={0.01}
                onChange={(v) => updateParam('friction', v)}
              />
              <ParamSlider
                label="FORCE RADIUS"
                value={state.max_force_radius}
                min={0.01}
                max={0.3}
                step={0.005}
                onChange={(v) => updateParam('max_force_radius', v)}
              />
              <ParamSlider
                label="MIN DISTANCE"
                value={state.min_distance}
                min={0.005}
                max={0.1}
                step={0.001}
                onChange={(v) => updateParam('min_distance', v)}
              />
              <ParamSlider
                label="SPEED"
                value={state.speed_multiplier}
                min={1}
                max={10}
                step={1}
                onChange={(v) => updateParam('speed_multiplier', v)}
              />
            </div>

            {/* Controls */}
            <div className="panel-section">
              <div className="section-header">CONTROLS</div>
              <SimControls
                paused={state.paused}
                onTogglePause={() => updateParam('paused', !state.paused)}
                onReset={() => updateMultiple({ reset_requested: true })}
                onRandomise={() => updateMultiple({ randomise_matrix_requested: true })}
              />
            </div>

            {/* Presets */}
            <div className="panel-section">
              <div className="section-header">PRESETS</div>
              <Presets
                state={state}
                onLoad={(parsed) => updateMultiple(parsed)}
              />
            </div>
          </>
        ) : (
          <>
            {/* POWER SPECTRUM */}
            <div className="panel-section">
              <div className="section-header">POWER SPECTRUM</div>
              <PowerSpectrumCanvas
                powerSpectrum2d={fftResult ? fftResult.powerSpectrum2d : null}
                gridSize={fftResult ? fftResult.gridSize : 256}
              />
            </div>

            {/* RADIAL PROFILE */}
            <div className="panel-section">
              <div className="section-header">
                RADIAL PROFILE
                {fftResult ? `  H: ${fftResult.spectralEntropy.toFixed(3)}` : ''}
              </div>
              <RadialProfileCanvas
                radialProfile={fftResult ? fftResult.radialProfile : null}
                spectralEntropy={fftResult ? fftResult.spectralEntropy : 0}
              />
            </div>

            {/* SPECTRAL ENTROPY */}
            <div className="panel-section">
              <div className="section-header">SPECTRAL ENTROPY  (60s)</div>
              <EntropyTimeSeriesCanvas
                entropyHistory={fftResult ? fftResult.entropyHistory : []}
              />
            </div>
          </>
        )}
      </div>
    </div>
  );
}

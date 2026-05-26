import React, { useState, useCallback } from 'react';

const PREFIX = 'preset_';

export default function Presets({ state, onLoad }) {
  const [loaded, setLoaded] = useState(null);

  const presetKeys = useCallback(() => {
    const keys = [];
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key.startsWith(PREFIX)) keys.push(key.slice(PREFIX.length));
    }
    return keys;
  }, []);

  const handleSave = () => {
    const name = window.prompt('Preset name:');
    if (!name) return;
    localStorage.setItem(PREFIX + name, JSON.stringify(state));
    setLoaded(name);
  };

  const handleLoad = (e) => {
    const name = e.target.value;
    if (!name) return;
    const raw = localStorage.getItem(PREFIX + name);
    if (!raw) return;
    try {
      const parsed = JSON.parse(raw);
      onLoad(parsed);
      setLoaded(name);
    } catch {
      // ignore corrupt data
    }
  };

  return (
    <div>
      <div className="presets-row">
        <button className="ctrl-btn" onClick={handleSave}>
          SAVE
        </button>
        <select className="preset-select" defaultValue="" onChange={handleLoad}>
          <option value="" disabled>
            LOAD \u25BE
          </option>
          {presetKeys().map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
      </div>
      <div className="preset-name">
        {loaded ? `Loaded: ${loaded}` : '\u2014'}
      </div>
    </div>
  );
}

import React from 'react';

export default function ParamSlider({ label, value, min, max, step, onChange, unit }) {
  return (
    <div className="param-row">
      <div className="param-header">
        <span className="param-label">{label}</span>
        <span className="param-value">
          {value.toFixed(step < 0.01 ? 3 : step < 0.1 ? 2 : 0)}
          {unit || ''}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
    </div>
  );
}

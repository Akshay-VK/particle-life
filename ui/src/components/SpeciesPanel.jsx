import React from 'react';

function floatToHex(r, g, b) {
  const toHex = (v) =>
    Math.round(Math.max(0, Math.min(1, v)) * 255)
      .toString(16)
      .padStart(2, '0');
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}

function hexToFloat(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [
    ((n >> 16) & 0xff) / 255,
    ((n >> 8) & 0xff) / 255,
    (n & 0xff) / 255,
  ];
}

export default function SpeciesPanel({ species, colors, counts, onChange }) {
  const handleColorChange = (index, hex) => {
    const newColors = colors.map((c, i) =>
      i === index ? hexToFloat(hex) : [...c]
    );
    onChange(newColors);
  };

  return (
    <div>
      {Array.from({ length: species }, (_, i) => (
        <div className="species-row" key={i}>
          <span className="species-label">S{i}</span>
          <input
            type="color"
            className="species-swatch"
            value={floatToHex(colors[i][0], colors[i][1], colors[i][2])}
            onChange={(e) => handleColorChange(i, e.target.value)}
          />
          <span className="species-count">
            {counts && counts[i] !== undefined ? counts[i] : '—'}
          </span>
        </div>
      ))}
    </div>
  );
}

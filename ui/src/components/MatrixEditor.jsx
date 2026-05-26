import React, { useRef, useCallback, useState } from 'react';

const NEG_RGB = [1.0, 0.251, 0.376];
const POS_RGB = [0.0, 0.898, 0.627];

function lerpColor(t) {
  const r = NEG_RGB[0] + (POS_RGB[0] - NEG_RGB[0]) * t;
  const g = NEG_RGB[1] + (POS_RGB[1] - NEG_RGB[1]) * t;
  const b = NEG_RGB[2] + (POS_RGB[2] - NEG_RGB[2]) * t;
  return `rgb(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255)})`;
}

function cellColor(value) {
  const t = (value + 1) / 2;
  if (t <= 0.5) {
    const u = t / 0.5;
    const r = 0.067 + (NEG_RGB[0] - 0.067) * u;
    const g = 0.067 + (NEG_RGB[1] - 0.067) * u;
    const b = 0.067 + (NEG_RGB[2] - 0.067) * u;
    return `rgb(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255)})`;
  }
  const u = (t - 0.5) / 0.5;
  const r = NEG_RGB[0] + (POS_RGB[0] - NEG_RGB[0]) * u;
  const g = NEG_RGB[1] + (POS_RGB[1] - NEG_RGB[1]) * u;
  const b = NEG_RGB[2] + (POS_RGB[2] - NEG_RGB[2]) * u;
  return `rgb(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255)})`;
}

export default function MatrixEditor({ label, matrix, size, onChange }) {
  const [dragging, setDragging] = useState(null);
  const lastY = useRef(0);

  const getIndex = useCallback((r, c) => r * size + c, [size]);

  const handleMouseDown = useCallback((r, c, e) => {
    setDragging({ r, c });
    lastY.current = e.clientY;
  }, []);

  const handleMouseMove = useCallback((e) => {
    if (!dragging) return;
    const delta = lastY.current - e.clientY;
    if (Math.abs(delta) < 1) return;
    lastY.current = e.clientY;
    const idx = getIndex(dragging.r, dragging.c);
    const step = 0.005 * Math.sign(delta);
    const newVal = Math.max(-1, Math.min(1, matrix[idx] + step));
    const updated = [...matrix];
    updated[idx] = Math.round(newVal * 1000) / 1000;
    onChange(updated);
  }, [dragging, matrix, onChange, getIndex]);

  const handleMouseUp = useCallback(() => {
    setDragging(null);
  }, []);

  return (
    <div>
      <div className="matrix-label">{label}</div>
      <div
        className="matrix-grid"
        style={{
          gridTemplateColumns: `repeat(${size}, 36px)`,
        }}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        {Array.from({ length: size }, (_, r) =>
          Array.from({ length: size }, (_, c) => {
            const idx = getIndex(r, c);
            const value = matrix[idx];
            const isDiagonal = r === c;
            return (
              <Cell
                key={idx}
                value={value}
                isDiagonal={isDiagonal}
                onMouseDown={(e) => handleMouseDown(r, c, e)}
              />
            );
          })
        )}
      </div>
    </div>
  );
}

function Cell({ value, isDiagonal, onMouseDown }) {
  const [hover, setHover] = useState(false);
  return (
    <div
      className={`matrix-cell${isDiagonal ? ' diagonal' : ''}`}
      style={{ background: cellColor(value) }}
      onMouseDown={onMouseDown}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
    >
      {hover && (
        <span className="matrix-cell-value">
          {value >= 0 ? `+${value.toFixed(2)}` : `${value.toFixed(2)}`}
        </span>
      )}
    </div>
  );
}

import React from 'react';

export default function SimControls({ paused, onTogglePause, onReset, onRandomise }) {
  return (
    <div className="controls-row">
      <button
        className={`ctrl-btn${paused ? ' active' : ''}`}
        onClick={onTogglePause}
      >
        {paused ? '\u25B6 PLAY' : '\u23F8 PAUSE'}
      </button>
      <button className="ctrl-btn" onClick={onReset}>
        \u21BA RESET
      </button>
      <button className="ctrl-btn" onClick={onRandomise}>
        \u2684 RANDOMISE
      </button>
      <button className="ctrl-btn disabled" disabled>
        \u25A0 STOP
      </button>
    </div>
  );
}

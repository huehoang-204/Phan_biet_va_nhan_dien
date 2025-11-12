(() => {
  const colors = [
    '#ff7b7b', '#27ae60', '#f39c12', '#2980b9', '#9b59b6', '#16a085', '#f1c40f'
  ];

  function hexToRgba(hex, alpha) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }

  function renderWaveAndSpec(waveEl, specEl, url, colorIdx = 0, height = 110) {
    if (!waveEl || !specEl || !url) return null;
    try {
      const ws = WaveSurfer.create({
        container: waveEl,
        height,
        waveColor: colors[colorIdx % colors.length] + '66',
        progressColor: colors[colorIdx % colors.length],
        cursorColor: '#00e5ff',
        normalize: true,
        interact: true,
        barWidth: 2,
        barGap: 1,
        minPxPerSec: 50,
        autoCenter: true,
        responsive: true,
      });
      ws.registerPlugin(WaveSurfer.Spectrogram.create({
        container: specEl,
        labels: true,
        height,
        frequencyMin: 20,
        frequencyMax: 8000,
        //colorMap: createHotColormap(),
        //colorMap: 'hot',
      }));
      ws.on('ready', () => {
        if (waveEl.id === 'outputWave' && window.__DIARIZATION__ && window.__DIARIZATION__.length > 0) {
          const regions = ws.registerPlugin(WaveSurfer.Regions.create());
          
          // Tạo bảng màu theo speaker
          const spkSet = [...new Set(window.__DIARIZATION__.map(d => d.spk))];
          const spkColor = {};
          spkSet.forEach((spk, i) => {
            spkColor[spk] = colors[i % colors.length];
          });
      
          // Vẽ regions + overlay trên spectrogram
          window.__DIARIZATION__.forEach(d => {
            // Region trên waveform
            regions.addRegion({
              start: d.start,
              end: d.end,
              color: hexToRgba(spkColor[d.spk], 0.3),
              drag: false,
              resize: false,
            });
      
            // Overlay trên spectrogram
            const overlay = document.createElement('div');
            overlay.style.position = 'absolute';
            overlay.style.top = '0';
            overlay.style.height = '100%';
            overlay.style.backgroundColor = hexToRgba(spkColor[d.spk], 0.15);
            overlay.style.pointerEvents = 'none';
            overlay.style.left = `${(d.start / ws.getDuration()) * 100}%`;
            overlay.style.width = `${((d.end - d.start) / ws.getDuration()) * 100}%`;
            specEl.appendChild(overlay);
          });
        }
      });
      ws.on('error', (e) => console.error('WaveSurfer error:', e));
      ws.load(url);
      return ws;
    } catch (e) {
      console.error('Render error:', e);
      return null;
    }
  }

  // Input audio
  window.addEventListener('DOMContentLoaded', () => {
    // Ensure overlay is hidden on page load
    const overlay = document.getElementById('overlay');
    if (overlay) {
      overlay.hidden = true;
    }
    
    // Animate cards with staggered delay
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, idx) => {
      card.classList.add('card-animate');
      card.style.setProperty('--card-delay', `${0.05 + idx * 0.08}s`);
    });
    document.querySelectorAll('.speaker-card').forEach((card, idx) => {
      card.classList.add('card-animate');
      card.style.setProperty('--card-delay', `${0.3 + idx * 0.06}s`);
    });

    // Show overlay while submitting
    const form = document.getElementById('uploadForm');
    if (form && overlay) {
      form.addEventListener('submit', () => {
        overlay.hidden = false;
      });
    }

    if (window.__INPUT_AUDIO__) {
      const waveEl = document.getElementById('inputWave');
      const specEl = document.getElementById('inputSpec');
      renderWaveAndSpec(waveEl, specEl, window.__INPUT_AUDIO__, 0, 140);
    }

    if (window.__OUTPUT_AUDIO__) {
      const waveEl = document.getElementById('outputWave');
      const specEl = document.getElementById('outputSpec');
      renderWaveAndSpec(waveEl, specEl, window.__OUTPUT_AUDIO__, 1, 140);
    }

    // Speaker audios
    const labelToIdx = {};
    if (document.querySelectorAll('.speaker-card[data-audio]').length > 0) {
      const speakerLabels = Array.from(document.querySelectorAll('.speaker-card__label')).map(l => l.textContent.trim());
      const uniqueSorted = [...new Set(speakerLabels)].sort();
      uniqueSorted.forEach((l, i) => {
        labelToIdx[l] = i;
      });
    }
    document.querySelectorAll('.speaker-card[data-audio]').forEach((card) => {
      const waveEl = card.querySelector('.speaker-card__wave');
      const specEl = card.querySelector('.speaker-card__spec');
      const audioUrl = card.dataset.audio;
      const label = card.querySelector('.speaker-card__label').textContent.trim();
      const colorIdx = labelToIdx[label] ?? 0;
      if (audioUrl) renderWaveAndSpec(waveEl, specEl, audioUrl, colorIdx);
    });

    // After rendering a result page, replace URL to root so reload resets UI
    if (window.__HAS_RESULTS__) {
      try { history.replaceState(null, '', '/'); } catch (e) {}
    }
  });
})();
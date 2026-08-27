"""Standalone interactive HTML dashboard for reviewing a diarization result.

Self-contained dark-theme page: color-coded speaker timeline, click-to-seek,
live active-speaker highlight, per-turn dialogue cards, and a track switcher
that swaps between the full mix and each isolated speaker stem.
"""

import html as _html
import json
import os
from typing import Dict, List, Optional

COLOR_PALETTE = [
    {"name": "cyan", "badge": "rgba(0, 242, 254, 0.15)", "border": "#00f2fe", "text": "#00f2fe", "grad": "linear-gradient(135deg, #00f2fe, #4facfe)"},
    {"name": "purple", "badge": "rgba(168, 85, 247, 0.15)", "border": "#a855f7", "text": "#c084fc", "grad": "linear-gradient(135deg, #c084fc, #9333ea)"},
    {"name": "amber", "badge": "rgba(245, 158, 11, 0.15)", "border": "#f59e0b", "text": "#fbbf24", "grad": "linear-gradient(135deg, #fbbf24, #d97706)"},
    {"name": "emerald", "badge": "rgba(16, 185, 129, 0.15)", "border": "#10b981", "text": "#34d399", "grad": "linear-gradient(135deg, #34d399, #059669)"},
    {"name": "rose", "badge": "rgba(244, 63, 94, 0.15)", "border": "#f43f5e", "text": "#fb7185", "grad": "linear-gradient(135deg, #fb7185, #e11d48)"},
    {"name": "blue", "badge": "rgba(59, 130, 246, 0.15)", "border": "#3b82f6", "text": "#60a5fa", "grad": "linear-gradient(135deg, #60a5fa, #2563eb)"},
]

def generate_html_player(
    media_filename: str,
    total_duration_s: float,
    segments: List[Dict],
    output_html_path: str,
    audio_sources: Optional[Dict[str, str]] = None,
):
    """Write a standalone HTML player dashboard next to the exported stems.

    ``audio_sources`` maps track keys to hrefs RELATIVE TO THE HTML FILE:
    ``"full"`` for the complete mix plus one entry per speaker label for that
    speaker's isolated stem. Every present track gets a switcher button. When
    omitted, the player falls back to the original media filename and offers
    no per-speaker tracks.
    """
    audio_sources = dict(audio_sources or {})
    audio_sources.setdefault("full", media_filename)

    # Extract unique speakers
    speakers = sorted(list(set(seg['speaker'] for seg in segments)))
    spk_colors = {}
    for idx, spk in enumerate(speakers):
        spk_colors[spk] = COLOR_PALETTE[idx % len(COLOR_PALETTE)]

    # Calculate speaker statistics
    spk_stats = {spk: 0.0 for spk in speakers}
    for seg in segments:
        spk_stats[seg['speaker']] += seg['duration']

    # Build segment data for JS
    js_segments = []
    for seg in segments:
        spk = seg['speaker']
        c = spk_colors[spk]
        js_segments.append({
            "id": seg['segment_id'],
            "spk": spk,
            "label": seg.get('speaker_label', spk),
            "start": seg['start'],
            "end": seg['end'],
            "dur": seg['duration'],
            "transcript": _html.escape(seg.get('transcript', '') or ''),
            "color": c['border'],
            "grad": c['grad']
        })

    speakers_json = json.dumps(speakers)
    audio_sources_json = json.dumps(audio_sources)
    segments_json = json.dumps(js_segments, ensure_ascii=False)
    spk_colors_json = json.dumps(spk_colors)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Speaker Diarization Player - {_html.escape(media_filename)}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg-dark: #0a0e17;
      --bg-card: rgba(20, 27, 45, 0.75);
      --border-card: rgba(255, 255, 255, 0.08);
      --text-main: #f1f5f9;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
      background-color: var(--bg-dark);
      background-image: 
        radial-gradient(circle at 15% 15%, rgba(0, 242, 254, 0.08) 0%, transparent 40%),
        radial-gradient(circle at 85% 85%, rgba(168, 85, 247, 0.08) 0%, transparent 40%);
      color: var(--text-main);
      min-height: 100vh;
      padding: 2.5rem 1.5rem;
      line-height: 1.6;
    }}
    .container {{ max-width: 1100px; margin: 0 auto; }}
    header {{ margin-bottom: 2rem; }}
    .badge-pill {{
      display: inline-flex; align-items: center; gap: 0.5rem;
      padding: 0.35rem 0.85rem; border-radius: 9999px; font-size: 0.8rem;
      font-weight: 600; text-transform: uppercase; background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(255, 255, 255, 0.1); margin-bottom: 0.75rem;
    }}
    .badge-pulse {{
      width: 8px; height: 8px; border-radius: 50%;
      background: #10b981; box-shadow: 0 0 10px #10b981;
    }}
    h1 {{
      font-size: 2.25rem; font-weight: 800;
      background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #94a3b8 100%);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem;
    }}
    .subtitle {{ color: var(--text-muted); font-size: 1.05rem; display: flex; gap: 1.5rem; flex-wrap: wrap; }}
    .metrics-grid {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1rem; margin-bottom: 2rem;
    }}
    .metric-card {{
      background: var(--bg-card); border: 1px solid var(--border-card);
      backdrop-filter: blur(12px); border-radius: 16px; padding: 1.25rem 1.5rem;
    }}
    .metric-title {{ font-size: 0.85rem; color: var(--text-dim); font-weight: 600; text-transform: uppercase; margin-bottom: 0.35rem; }}
    .metric-value {{ font-size: 1.6rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }}
    .metric-sub {{ font-size: 0.85rem; color: var(--text-muted); margin-top: 0.25rem; }}

    .player-card {{
      background: var(--bg-card); border: 1px solid var(--border-card);
      backdrop-filter: blur(16px); border-radius: 20px; padding: 2rem; margin-bottom: 2rem;
    }}
    .player-top {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem; flex-wrap: wrap; gap: 1rem; }}
    .speaker-pill {{
      font-size: 0.85rem; font-weight: 600; padding: 0.3rem 0.85rem; border-radius: 9999px;
      display: inline-flex; align-items: center; gap: 0.5rem;
    }}
    .timeline-container {{ margin-bottom: 1.5rem; position: relative; }}
    .timeline-track {{
      height: 52px; background: rgba(15, 23, 42, 0.8); border-radius: 12px;
      position: relative; overflow: hidden; cursor: pointer; border: 1px solid rgba(255, 255, 255, 0.08);
    }}
    .timeline-segment {{
      position: absolute; top: 4px; bottom: 4px; border-radius: 8px;
      cursor: pointer; display: flex; align-items: center; justify-content: center;
      font-size: 0.75rem; font-weight: 700; color: #fff; padding: 0 4px;
    }}
    .timeline-segment:hover {{ filter: brightness(1.25); z-index: 10; }}
    .playhead {{
      position: absolute; top: 0; bottom: 0; width: 2px; background: #ffffff;
      box-shadow: 0 0 10px #ffffff; pointer-events: none; z-index: 20; left: 0%;
      transition: left 0.1s linear;
    }}
    .time-indicators {{
      display: flex; justify-content: space-between; font-size: 0.8rem;
      color: var(--text-dim); font-family: 'JetBrains Mono', monospace; margin-top: 0.5rem;
    }}
    .controls-row {{ display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1.25rem; }}
    .play-btn-group {{ display: flex; align-items: center; gap: 1rem; }}
    .btn-main {{
      width: 52px; height: 52px; border-radius: 50%;
      background: linear-gradient(135deg, #00f2fe, #4facfe); border: none;
      color: #04101e; display: flex; align-items: center; justify-content: center;
      cursor: pointer; box-shadow: 0 8px 24px rgba(0, 242, 254, 0.4);
    }}
    .btn-main svg {{ width: 22px; height: 22px; fill: currentColor; }}
    .time-display {{ font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 600; }}
    .filter-pills {{ display: flex; gap: 0.5rem; flex-wrap: wrap; }}
    .filter-btn {{
      padding: 0.4rem 0.9rem; border-radius: 9999px; border: 1px solid rgba(255, 255, 255, 0.1);
      background: rgba(255, 255, 255, 0.04); color: var(--text-muted); font-size: 0.85rem;
      cursor: pointer; transition: all 0.2s;
    }}
    .filter-btn:hover, .filter-btn.active {{
      color: #fff; background: rgba(255, 255, 255, 0.15); border-color: rgba(255, 255, 255, 0.3);
    }}
    .legend {{ display: flex; gap: 1.5rem; margin-top: 1rem; flex-wrap: wrap; font-size: 0.85rem; }}
    .legend-item {{ display: flex; align-items: center; gap: 0.5rem; }}
    .legend-color {{ width: 12px; height: 12px; border-radius: 3px; }}

    .segment-list {{ display: flex; flex-direction: column; gap: 0.85rem; margin-top: 1.25rem; }}
    .dialogue-card {{
      background: var(--bg-card); border: 1px solid var(--border-card); border-radius: 16px;
      padding: 1.15rem 1.5rem; display: grid; grid-template-columns: 140px 1fr auto;
      gap: 1.5rem; align-items: center; cursor: pointer; transition: all 0.2s;
    }}
    .dialogue-card:hover {{ background: rgba(30, 41, 69, 0.85); transform: translateX(4px); }}
    .dialogue-card.active {{ border-color: #00f2fe; background: rgba(0, 242, 254, 0.06); }}
    .timestamp-badge {{ font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: var(--text-dim); }}
    .dialogue-text {{ font-size: 1.05rem; font-weight: 500; color: #fff; }}
    .play-icon-btn {{
      width: 38px; height: 38px; border-radius: 50%; border: 1px solid rgba(255, 255, 255, 0.12);
      background: rgba(255, 255, 255, 0.05); color: #fff; display: flex; align-items: center;
      justify-content: center; cursor: pointer;
    }}
    .play-icon-btn:hover {{ background: #fff; color: #000; transform: scale(1.1); }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="badge-pill">
        <span class="badge-pulse"></span>
        <span>Universal Speaker Diarization • Multi-Speaker Pipeline</span>
      </div>
      <h1>Speaker Diarization Dashboard</h1>
      <div class="subtitle">
        <span><strong>Media:</strong> {_html.escape(media_filename)}</span>
        <span>•</span>
        <span><strong>Total Duration:</strong> {total_duration_s:.2f}s</span>
        <span>•</span>
        <span><strong>Speakers Detected:</strong> {len(speakers)}</span>
      </div>
    </header>

    <!-- Metrics -->
    <div class="metrics-grid">
      <div class="metric-card">
        <div class="metric-title">Duration</div>
        <div class="metric-value">{total_duration_s:.2f}s</div>
        <div class="metric-sub">Total Media Length</div>
      </div>
      <div class="metric-card">
        <div class="metric-title">Detected Speakers</div>
        <div class="metric-value" style="color: #00f2fe;">{len(speakers)}</div>
        <div class="metric-sub">Clustered Neural Profiles</div>
      </div>
      <div class="metric-card">
        <div class="metric-title">Total Turns</div>
        <div class="metric-value">{len(segments)}</div>
        <div class="metric-sub">Diarized Segments</div>
      </div>
      <div class="metric-card">
        <div class="metric-title">Noise Handling</div>
        <div class="metric-value" style="color: #10b981;">Preserved</div>
        <div class="metric-sub">100% Original Acoustics</div>
      </div>
    </div>

    <!-- Audio Player -->
    <div class="player-card">
      <audio id="audioElement" src="{_html.escape(audio_sources['full'])}" preload="auto"></audio>
      <div class="player-top">
        <div class="now-playing">
          <span class="time-display"><span id="currentTime">00:00.0</span> / <span id="duration">{int(total_duration_s//60):02d}:{total_duration_s%60:04.1f}</span></span>
          <span id="activeSpeakerPill" class="speaker-pill" style="display: none;">Speaker</span>
        </div>
        <div class="filter-pills" id="filterPills"></div>
      </div>

      <div class="timeline-container">
        <div class="timeline-track" id="timelineTrack">
          <div class="playhead" id="playhead"></div>
        </div>
        <div class="time-indicators">
          <span>00:00</span>
          <span>{int(total_duration_s//60):02d}:{total_duration_s%60:04.1f}</span>
        </div>
      </div>

      <div class="legend" id="legendContainer"></div>

      <div class="controls-row" style="margin-top: 1.5rem;">
        <div class="play-btn-group">
          <button class="btn-main" id="playBtn" onclick="togglePlay()">
            <svg id="playIcon" viewBox="0 0 24 24"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
          </button>
          <button class="filter-btn" onclick="seekRelative(-5)">-5s</button>
          <button class="filter-btn" onclick="seekRelative(5)">+5s</button>
        </div>
        <div style="color: var(--text-dim); font-size: 0.85rem;">
          Click any block on the timeline or card below to jump and listen
        </div>
      </div>
    </div>

    <!-- Dialogue List -->
    <div>
      <h2>Speaker Turns & Timestamps</h2>
      <div class="segment-list" id="segmentList"></div>
    </div>
  </div>

  <script>
    const totalDuration = {total_duration_s};
    const audio = document.getElementById('audioElement');
    const playBtn = document.getElementById('playBtn');
    const playIcon = document.getElementById('playIcon');
    const currentTimeEl = document.getElementById('currentTime');
    const playhead = document.getElementById('playhead');
    const timelineTrack = document.getElementById('timelineTrack');
    const segmentList = document.getElementById('segmentList');
    const activeSpeakerPill = document.getElementById('activeSpeakerPill');
    const filterPills = document.getElementById('filterPills');
    const legendContainer = document.getElementById('legendContainer');

    const speakers = {speakers_json};
    const segments = {segments_json};
    const spkColors = {spk_colors_json};
    const audioSources = {audio_sources_json};

    // Track switcher: full mix plus one button per isolated speaker stem.
    let activeTrack = 'full';
    function switchTrack(key) {{
      if (!(key in audioSources) || key === activeTrack) return;
      const wasPlaying = !audio.paused;
      const t = audio.currentTime;
      activeTrack = key;
      audio.src = audioSources[key];
      audio.currentTime = t;
      if (wasPlaying) audio.play();
      document.querySelectorAll('.filter-btn[data-track]').forEach(b =>
        b.classList.toggle('active', b.dataset.track === key));
    }}
    Object.keys(audioSources).forEach(key => {{
      const btn = document.createElement('button');
      btn.className = 'filter-btn' + (key === 'full' ? ' active' : '');
      btn.dataset.track = key;
      btn.textContent = key === 'full' ? 'Full Audio' : key + ' (solo)';
      btn.onclick = () => switchTrack(key);
      filterPills.appendChild(btn);
    }});

    // Render Filter Buttons & Legend
    speakers.forEach(spk => {{
      const c = spkColors[spk];
      
      // Legend item
      const leg = document.createElement('div');
      leg.className = 'legend-item';
      leg.innerHTML = `<div class="legend-color" style="background: ${{c.grad}}"></div><span>${{spk}}</span>`;
      legendContainer.appendChild(leg);
    }});

    // Render Timeline Segments
    segments.forEach(seg => {{
      const leftPct = (seg.start / totalDuration) * 100;
      const widthPct = (seg.dur / totalDuration) * 100;
      const block = document.createElement('div');
      block.className = 'timeline-segment';
      block.style.left = `${{leftPct}}%`;
      block.style.width = `${{Math.max(widthPct, 0.6)}}%`;
      block.style.background = seg.grad;
      block.title = `${{seg.label}}: ${{seg.start.toFixed(1)}}s - ${{seg.end.toFixed(1)}}s`;
      block.onclick = (e) => {{
        e.stopPropagation();
        seekTo(seg.start);
      }};
      timelineTrack.appendChild(block);
    }});

    // Render Segment List
    segments.forEach(seg => {{
      const c = spkColors[seg.spk];
      const card = document.createElement('div');
      card.className = 'dialogue-card';
      card.id = `card-${{seg.id}}`;
      card.onclick = () => seekTo(seg.start);

      card.innerHTML = `
        <div>
          <span class="speaker-pill" style="background: ${{c.badge}}; color: ${{c.text}}; border: 1px solid ${{c.border}};">${{seg.spk}}</span>
          <div class="timestamp-badge" style="margin-top: 4px;">${{formatTime(seg.start)}} - ${{formatTime(seg.end)}}</div>
        </div>
        <div class="dialogue-text">${{seg.transcript || '&mdash;'}}</div>
        <div class="play-icon-btn">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
        </div>
      `;
      segmentList.appendChild(card);
    }});

    timelineTrack.onclick = (e) => {{
      const rect = timelineTrack.getBoundingClientRect();
      const clickX = e.clientX - rect.left;
      const pct = clickX / rect.width;
      seekTo(pct * totalDuration);
    }};

    function togglePlay() {{
      if (audio.paused) {{
        audio.play();
        playIcon.innerHTML = '<rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect>';
      }} else {{
        audio.pause();
        playIcon.innerHTML = '<polygon points="5 3 19 12 5 21 5 3"></polygon>';
      }}
    }}

    function seekTo(sec) {{
      audio.currentTime = sec;
      if (audio.paused) {{
        audio.play();
        playIcon.innerHTML = '<rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect>';
      }}
    }}

    function seekRelative(delta) {{
      audio.currentTime = Math.max(0, Math.min(totalDuration, audio.currentTime + delta));
    }}

    function formatTime(sec) {{
      const m = Math.floor(sec / 60);
      const s = (sec % 60).toFixed(1);
      return `${{String(m).padStart(2, '0')}}:${{s.padStart(4, '0')}}`;
    }}

    audio.addEventListener('timeupdate', () => {{
      const cur = audio.currentTime;
      currentTimeEl.textContent = formatTime(cur);
      const pct = (cur / totalDuration) * 100;
      playhead.style.left = `${{pct}}%`;

      let currentSeg = segments.find(s => cur >= s.start && cur <= s.end);
      document.querySelectorAll('.dialogue-card').forEach(c => c.classList.remove('active'));

      if (currentSeg) {{
        const c = spkColors[currentSeg.spk];
        activeSpeakerPill.style.display = 'inline-flex';
        activeSpeakerPill.style.background = c.badge;
        activeSpeakerPill.style.color = c.text;
        activeSpeakerPill.style.border = `1px solid ${{c.border}}`;
        activeSpeakerPill.textContent = currentSeg.spk;

        const card = document.getElementById(`card-${{currentSeg.id}}`);
        if (card) card.classList.add('active');
      }} else {{
        activeSpeakerPill.style.display = 'none';
      }}
    }});

    audio.addEventListener('ended', () => {{
      playIcon.innerHTML = '<polygon points="5 3 19 12 5 21 5 3"></polygon>';
      playhead.style.left = '0%';
    }});
  </script>
</body>
</html>
"""
    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    return output_html_path

/* ===========================
   世界杯预测 - 前端交互
   v4 - 清理自动轮播，保留手动箭头切换
   =========================== */

// API 基础路径
const API = '/api';

// 状态管理
const state = {
  homeTeam: null,
  awayTeam: null,
  teamsCache: [],
  liveData: [],
  refreshTimer: null,
  chatStreaming: false,
  chatAbortController: null,
  chatRenderPending: false,    // SSE 批量渲染标志
  lastChatRenderAt: 0,         // 上次 SSE 渲染时间戳
};

// ===========================
// 工具函数
// ===========================
function $(sel) { return document.querySelector(sel); }
function $$(sel) { return document.querySelectorAll(sel); }

function formatDate(iso) {
  if (!iso) return '-';
  const d = new Date(iso);
  const now = new Date();
  const isToday = d.toDateString() === now.toDateString();
  const tomorrow = new Date(now.getTime() + 86400000);
  const isTomorrow = d.toDateString() === tomorrow.toDateString();
  if (isToday) return `今天 ${d.getHours().toString().padStart(2,'0')}:${d.getMinutes().toString().padStart(2,'0')}`;
  if (isTomorrow) return `明天 ${d.getHours().toString().padStart(2,'0')}:${d.getMinutes().toString().padStart(2,'0')}`;
  return `${d.getMonth()+1}月${d.getDate()}日 ${d.getHours().toString().padStart(2,'0')}:${d.getMinutes().toString().padStart(2,'0')}`;
}

function pct(v, digits) {
  if (digits === undefined) digits = 1;
  return (v * 100).toFixed(digits) + '%';
}

async function fetchJSON(url, options) {
  const resp = await fetch(url, options);
  if (!resp.ok) {
    throw new Error(`HTTP ${resp.status}: ${await resp.text()}`);
  }
  return resp.json();
}

// ===========================
// 加载球队列表
// ===========================
async function loadTeams() {
  try {
    const data = await fetchJSON(`${API}/teams?top_n=120`);
    state.teamsCache = data;
  } catch (e) {
    console.error('加载球队失败:', e);
  }
}

// ===========================
// 球队搜索（带中文）—— 使用事件委托避免每按键重建 listener
// ===========================
function setupTeamSearch(inputId, suggestionsId, side) {
  const input = $(`#${inputId}`);
  const suggestions = $(`#${suggestionsId}`);
  if (!input || !suggestions) return;

  // 输入时刷新建议列表
  input.addEventListener('input', () => {
    const q = input.value.trim().toLowerCase();
    if (q.length < 1) {
      suggestions.classList.remove('show');
      return;
    }
    const filtered = state.teamsCache.filter(t =>
      t.team_en.toLowerCase().includes(q) || t.team_zh.includes(q)
    ).slice(0, 8);

    if (filtered.length === 0) {
      suggestions.innerHTML = '<div class="suggestion-item"><span class="suggestion-zh">无匹配球队</span></div>';
      suggestions.classList.add('show');
      return;
    }

    suggestions.innerHTML = filtered.map(t => `
      <div class="suggestion-item" data-team="${t.team_en}" data-zh="${t.team_zh}" data-elo="${t.elo}">
        <div>
          <div class="suggestion-zh">${t.team_zh}</div>
          <div class="suggestion-en">${t.team_en}</div>
        </div>
        <div class="suggestion-elo">Elo ${t.elo}</div>
      </div>
    `).join('');
    suggestions.classList.add('show');
  });

  // ★ 事件委托：只注册一个 click listener，不再每按键重建
  suggestions.addEventListener('click', (e) => {
    const item = e.target.closest('.suggestion-item');
    if (!item) return;
    const team = item.dataset.team;
    const zh = item.dataset.zh;
    const elo = item.dataset.elo;
    if (!team) return;

    if (side === 'home') {
      state.homeTeam = team;
      const hd = $('#homeDisplay');
      if (hd) hd.innerHTML = `
        <div class="team-name-zh">${zh}</div>
        <div class="team-name-en">${team}</div>
        <div class="team-elo">Elo ${elo}</div>
      `;
    } else {
      state.awayTeam = team;
      const ad = $('#awayDisplay');
      if (ad) ad.innerHTML = `
        <div class="team-name-zh">${zh}</div>
        <div class="team-name-en">${team}</div>
        <div class="team-elo">Elo ${elo}</div>
      `;
    }
    input.value = '';
    suggestions.classList.remove('show');
  });

  // 失焦隐藏
  document.addEventListener('click', (e) => {
    if (!input.contains(e.target) && !suggestions.contains(e.target)) {
      suggestions.classList.remove('show');
    }
  });
}

// ===========================
// 实时比赛加载
// ===========================
async function loadLiveMatches() {
  const grid = $('#liveGrid');
  if (!grid) return;
  try {
    const data = await fetchJSON(`${API}/live`);
    state.liveData = data.matches || [];

    // 更新hero stats
    const liveCount = state.liveData.filter(m => m.status === 'live').length;
    const todayCount = state.liveData.length;
    const statLive = $('#statLive');
    const heroStatus = $('#heroStatus');
    if (statLive) statLive.textContent = todayCount;
    if (heroStatus) {
      if (liveCount > 0) {
        heroStatus.textContent = `🔴 ${liveCount} 场比赛进行中 · 共 ${todayCount} 场今日比赛`;
      } else {
        heroStatus.textContent = `今日 ${todayCount} 场世界杯比赛 · 实时数据`;
      }
    }

    if (state.liveData.length === 0) {
      grid.innerHTML = `
        <div class="loading-card glass">
          <p>暂无实时比赛数据，稍后再试</p>
        </div>
      `;
      return;
    }

    // 分组
    const finished = state.liveData.filter(m => m.status === 'finished');
    const upcoming = state.liveData.filter(m => m.status !== 'finished');

    let html = '';
    if (upcoming.length > 0) {
      html += renderLiveCarousel('upcoming', '🔴', '即将 / 进行中', upcoming.length, upcoming);
    }
    if (finished.length > 0) {
      html += renderLiveCarousel('finished', '✅', '已结束', finished.length, finished);
    }
    grid.innerHTML = html;

    // 轮播事件委托已在 init() 中一次性设置
    // 仅需对新渲染的箭头刷新禁用状态
    $$('.carousel-track').forEach(track => updateArrowState(track));
  } catch (e) {
    console.error('加载实时数据失败:', e);
    grid.innerHTML = `
      <div class="loading-card glass">
        <p style="color: var(--accent-danger);">实时数据加载失败</p>
        <p style="font-size: 12px; margin-top: 8px;">${e.message}</p>
      </div>
    `;
  }
}

// 渲染一组比赛的横向轮播
function renderLiveCarousel(id, icon, label, count, matches) {
  return `
    <div class="live-carousel" data-group="${id}">
      <div class="live-section-title">
        <span class="live-section-icon">${icon}</span> ${label}
        <span class="live-section-count">${count} 场</span>
      </div>
      <div class="carousel-wrapper">
        <button class="carousel-arrow carousel-arrow-left" data-target="${id}" aria-label="向左滑动" disabled>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 18 9 12 15 6"/></svg>
        </button>
        <div class="carousel-track" id="track-${id}">
          ${matches.map(m => renderLiveCard(m)).join('')}
        </div>
        <button class="carousel-arrow carousel-arrow-right" data-target="${id}" aria-label="向右滑动">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"/></svg>
        </button>
      </div>
    </div>
  `;
}

// ★ 事件委托：轮播箭头点击 — 只注册一次在 #liveGrid 上
function setupCarouselDelegation() {
  const grid = $('#liveGrid');
  if (!grid) return;

  // 箭头点击（委托）
  grid.addEventListener('click', (e) => {
    const arrow = e.target.closest('.carousel-arrow');
    if (!arrow || arrow.disabled) return;  // ★ disabled 状态静默跳过
    const target = arrow.dataset.target;
    const track = $(`#track-${target}`);
    if (!track) return;
    const cardWidth = track.querySelector('.live-card')?.offsetWidth || 360;
    const gap = 20;
    const step = cardWidth + gap;
    const dir = arrow.classList.contains('carousel-arrow-right') ? 1 : -1;
    track.scrollBy({ left: dir * step, behavior: 'smooth' });
    // ★ 兜底：smooth 动画完成后更新箭头状态（同时 scroll 事件也会更新）
    setTimeout(() => updateArrowState(track), 450);
  });

  // 滚动时更新箭头状态（委托）
  grid.addEventListener('scroll', (e) => {
    const track = e.target.closest('.carousel-track');
    if (track) updateArrowState(track);
  }, { passive: true });

}

// 根据滚动位置更新左右箭头禁用状态
function updateArrowState(track) {
  const wrapper = track.closest('.carousel-wrapper');
  if (!wrapper) return;
  const leftBtn = wrapper.querySelector('.carousel-arrow-left');
  const rightBtn = wrapper.querySelector('.carousel-arrow-right');
  const maxScroll = track.scrollWidth - track.clientWidth;
  if (leftBtn) leftBtn.disabled = track.scrollLeft <= 2;
  if (rightBtn) rightBtn.disabled = track.scrollLeft >= maxScroll - 2;
}

// ===========================
// 轮播左右箭头点击（仅手动，无自动滚动）

// ===========================
function renderLiveCard(m) {
  const statusBadge = {
    'live': '<span class="status-badge live"><span class="pulse-dot" style="width:6px;height:6px;"></span>进行中</span>',
    'scheduled': '<span class="status-badge scheduled">未开始</span>',
    'finished': '<span class="status-badge finished">已结束</span>',
  }[m.status] || `<span class="status-badge scheduled">${m.status}</span>`;

  let scoreDisplay;
  if (m.status === 'finished' || (m.home_score !== null && m.away_score !== null && m.status === 'live')) {
    scoreDisplay = `<div class="score-display">${m.home_score} - ${m.away_score}</div>`;
  } else {
    scoreDisplay = `<div class="score-display vs">VS</div>`;
  }

  let predHTML = '';
  if (m.prediction) {
    const p = m.prediction;
    const homePct = Math.round(p.home_win * 100);
    const drawPct = Math.round(p.draw * 100);
    const awayPct = Math.round(p.away_win * 100);

    // 激进预测范围（如有）
    const aggRange = p.aggressive_score && p.aggressive_score !== p.predicted_score
      ? `<span class="predicted-score agg-range" title="激进预测">→ ${p.aggressive_score}</span>`
      : '';

    predHTML = `
      <div class="match-prediction">
        <div class="prediction-label">AI 预测</div>
        <div class="probability-bar">
          <div class="prob-segment prob-home" style="flex: ${homePct};">${homePct >= 12 ? homePct + '%' : ''}</div>
          <div class="prob-segment prob-draw" style="flex: ${drawPct};">${drawPct >= 12 ? drawPct + '%' : ''}</div>
          <div class="prob-segment prob-away" style="flex: ${awayPct};">${awayPct >= 12 ? awayPct + '%' : ''}</div>
        </div>
        <div class="prediction-detail">
          <span>预测比分: <span class="predicted-score">${p.predicted_score}</span> ${aggRange}</span>
          <span style="color: var(--accent-gold);">Elo差: ${p.elo_diff > 0 ? '+' : ''}${p.elo_diff.toFixed(0)}</span>
        </div>
      </div>
    `;
  }

  const venue = m.venue ? m.venue : (m.city ? `${m.city}, ${m.country}` : '');

  return `
    <div class="live-card glass ${m.status === 'live' ? 'live-now' : ''}">
      <div class="live-status">
        <span>${formatDate(m.date)}</span>
        ${statusBadge}
      </div>
      <div class="match-teams">
        <div class="team-block">
          <div class="team-name-zh">${m.home_team_zh}</div>
          <div class="team-name-en">${m.home_team}</div>
        </div>
        <div class="score-block">
          ${scoreDisplay}
        </div>
        <div class="team-block">
          <div class="team-name-zh">${m.away_team_zh}</div>
          <div class="team-name-en">${m.away_team}</div>
        </div>
      </div>
      ${predHTML}
      ${venue ? `<div class="match-venue">📍 ${venue}</div>` : ''}
      <button class="ai-ask-btn" data-home="${m.home_team}" data-away="${m.away_team}" data-status="${m.status}" data-home-zh="${m.home_team_zh}" data-away-zh="${m.away_team_zh}" data-home-score="${m.home_score ?? ''}" data-away-score="${m.away_score ?? ''}" data-date="${m.date ?? ''}" data-venue="${(m.venue || '') + (m.city ? ', ' + m.city : '') + (m.country ? ', ' + m.country : '')}">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
        AI问赛 · ${m.status === 'finished' ? '分析过程' : 'AI预测'}
      </button>
    </div>
  `;
}

// ===========================
// 自定义预测
// ===========================
async function predictMatch() {
  const btn = $('#predictBtn');
  const resultDiv = $('#predictResult');

  if (!state.homeTeam || !state.awayTeam) {
    resultDiv.innerHTML = `
      <div class="predict-placeholder">
        <span class="placeholder-icon">⚠️</span>
        <p style="color: var(--accent-warning);">请选择两支球队</p>
      </div>
    `;
    return;
  }
  if (state.homeTeam === state.awayTeam) {
    resultDiv.innerHTML = `
      <div class="predict-placeholder">
        <span class="placeholder-icon">⚠️</span>
        <p style="color: var(--accent-danger);">请选择两支不同的球队</p>
      </div>
    `;
    return;
  }

  btn.disabled = true;
  btn.querySelector('span').textContent = '预测中...';
  resultDiv.innerHTML = `
    <div class="predict-placeholder">
      <div class="loader"></div>
      <p>AI 正在分析中...</p>
    </div>
  `;

  try {
    const tournament = $('#tournamentSelect').value;
    const neutral = $('#neutralSelect').value === 'true';
    const url = `${API}/predict?home=${encodeURIComponent(state.homeTeam)}&away=${encodeURIComponent(state.awayTeam)}&tournament=${encodeURIComponent(tournament)}&neutral=${neutral}&mode=both`;
    const data = await fetchJSON(url);

    resultDiv.innerHTML = renderPrediction(data);
  } catch (e) {
    resultDiv.innerHTML = `
      <div class="predict-placeholder">
        <span class="placeholder-icon">❌</span>
        <p style="color: var(--accent-danger);">预测失败: ${e.message}</p>
      </div>
    `;
  } finally {
    btn.disabled = false;
    btn.querySelector('span').textContent = '开始预测';
  }
}

function renderPrediction(p) {
  // 检测是否为双预测格式 (predict_both 返回值)
  const isDual = !!(p.conservative && p.aggressive);
  const cons = isDual ? p.conservative : p;
  const agg = isDual ? p.aggressive : null;
  const adjFactors = isDual ? (p.adjustment_factors || {}) : {};

  const probs = cons.probabilities;
  const homePct = (probs.home_win * 100).toFixed(1);
  const drawPct = (probs.draw * 100).toFixed(1);
  const awayPct = (probs.away_win * 100).toFixed(1);

  const totalElo = cons.elo_ratings.home + cons.elo_ratings.away;
  const homeEloPct = (cons.elo_ratings.home / totalElo * 100).toFixed(1);
  const awayEloPct = 100 - homeEloPct;

  const topScoresHTML = cons.top_scores.map((s, i) => `
    <div class="top-score-item">
      <div class="top-score-rank">#${i+1}</div>
      <div class="top-score-value">${s.score}</div>
      <div class="top-score-prob">${(s.probability * 100).toFixed(1)}%</div>
    </div>
  `).join('');

  const explanationHTML = cons.explanation.map(e => `
    <div class="explanation-item">${e}</div>
  `).join('');

  // 激进预测区域（仅双预测模式显示）
  let aggressiveHTML = '';
  if (agg) {
    const aggProbs = agg.probabilities;
    const aggTopScoresHTML = agg.top_scores.map((s, i) => `
      <div class="top-score-item agg">
        <div class="top-score-rank">#${i+1}</div>
        <div class="top-score-value">${s.score}</div>
        <div class="top-score-prob">${(s.probability * 100).toFixed(1)}%</div>
      </div>
    `).join('');

    const aggExplanationHTML = agg.explanation.map(e => `
      <div class="explanation-item agg">${e}</div>
    `).join('');

    // 调整因子说明
    const inflationNote = adjFactors.tournament_inflation > 1.05
      ? `本届场均 ${(adjFactors.tournament_inflation * (adjFactors.wc2026_avg_goals || 2.6)).toFixed(1)} 球（历史${adjFactors.historical_avg_goals || 2.6}球），通胀 ${adjFactors.tournament_inflation?.toFixed(2) || '-'}x`
      : '';

    const formNotes = [];
    if (adjFactors.home_form_deviation > 1.1) formNotes.push(`${p.home_team_zh} 进球率高于历史 x${adjFactors.home_form_deviation?.toFixed(1)}`);
    if (adjFactors.away_form_deviation > 1.1) formNotes.push(`${p.away_team_zh} 进球率高于历史 x${adjFactors.away_form_deviation?.toFixed(1)}`);
    const formNote = formNotes.length > 0 ? formNotes.join('；') : '';

    aggressiveHTML = `
      <div class="dual-prediction-section">
        <div class="dual-section-title">
          <span>⚡ 激进预测（基于本届走势）</span>
          ${inflationNote ? `<span class="dual-inflation-badge">${inflationNote}</span>` : ''}
          ${formNote ? `<span class="dual-inflation-badge" style="background:rgba(255,107,53,0.2);">${formNote}</span>` : ''}
        </div>

        <div class="dual-pred-grid">
          <div class="dual-pred-card conservative">
            <div class="dual-pred-label">🛡️ 保守预测</div>
            <div class="dual-pred-score">${cons.predicted_score.home} - ${cons.predicted_score.away}</div>
            <div class="dual-pred-expected">期望 ${cons.predicted_score.home_expected?.toFixed(2)} - ${cons.predicted_score.away_expected?.toFixed(2)}</div>
            <div class="dual-prob-row">
              <span class="prob-dot" style="background:var(--accent-success);"></span>胜 ${(cons.probabilities.home_win * 100).toFixed(0)}%
              <span class="prob-dot" style="background:var(--accent-warning);"></span>平 ${(cons.probabilities.draw * 100).toFixed(0)}%
              <span class="prob-dot" style="background:var(--accent-danger);"></span>负 ${(cons.probabilities.away_win * 100).toFixed(0)}%
            </div>
          </div>
          <div class="dual-pred-card aggressive">
            <div class="dual-pred-label">🔥 激进预测</div>
            <div class="dual-pred-score agg-score">${agg.predicted_score.home} - ${agg.predicted_score.away}</div>
            <div class="dual-pred-expected">期望 ${agg.predicted_score.home_expected?.toFixed(2)} - ${agg.predicted_score.away_expected?.toFixed(2)}</div>
            <div class="dual-prob-row">
              <span class="prob-dot" style="background:var(--accent-success);"></span>胜 ${(aggProbs.home_win * 100).toFixed(0)}%
              <span class="prob-dot" style="background:var(--accent-warning);"></span>平 ${(aggProbs.draw * 100).toFixed(0)}%
              <span class="prob-dot" style="background:var(--accent-danger);"></span>负 ${(aggProbs.away_win * 100).toFixed(0)}%
            </div>
          </div>
        </div>

        <div class="dual-top-scores">
          <div style="flex:1">
            <div class="dual-section-subtitle">保守 Top 3 比分</div>
            ${cons.top_scores.slice(0, 3).map(s => `
              <div class="top-score-item">${s.score} <span class="top-score-prob">${(s.probability * 100).toFixed(1)}%</span></div>
            `).join('')}
          </div>
          <div style="flex:1">
            <div class="dual-section-subtitle">激进 Top 3 比分</div>
            ${agg.top_scores.slice(0, 3).map(s => `
              <div class="top-score-item agg">${s.score} <span class="top-score-prob">${(s.probability * 100).toFixed(1)}%</span></div>
            `).join('')}
          </div>
        </div>
      </div>
    `;
  }

  // 单预测模式下的 Top 5（向后兼容）
  const topScoresSection = !isDual ? `
    <div class="result-section-title">Top 5 最可能比分</div>
    <div class="top-scores">${topScoresHTML}</div>
  ` : '';

  const scorePredSection = !isDual ? `
    <div class="result-section-title">预测比分 & 期望进球</div>
    <div class="score-prediction">
      <div class="score-pred-card">
        <div class="win-rate-label">${p.home_team_zh}</div>
        <div class="score-pred-value">${p.predicted_score.home}</div>
        <div class="score-pred-expected">期望 ${p.predicted_score.home_expected}</div>
      </div>
      <div class="score-pred-card">
        <div class="win-rate-label">${p.away_team_zh}</div>
        <div class="score-pred-value">${p.predicted_score.away}</div>
        <div class="score-pred-expected">期望 ${p.predicted_score.away_expected}</div>
      </div>
    </div>
  ` : '';

  const winRateSection = !isDual ? `
    <div class="result-section-title">综合胜率 (胜+0.5×平)</div>
    <div class="win-rate-section">
      <div class="win-rate-card home">
        <div class="win-rate-label">${p.home_team_zh} 胜率</div>
        <div class="win-rate-value">${(p.win_rates.home * 100).toFixed(1)}%</div>
      </div>
      <div class="win-rate-card away">
        <div class="win-rate-label">${p.away_team_zh} 胜率</div>
        <div class="win-rate-value">${(p.win_rates.away * 100).toFixed(1)}%</div>
      </div>
    </div>
  ` : '';

  return `
    <div class="result-container">
      <div class="result-header">
        <div class="result-matchup">
          <div class="result-team">
            <div class="result-team-name">${p.home_team_zh}</div>
            <div class="result-team-en">${p.home_team}</div>
          </div>
          <div class="result-vs-score">${cons.predicted_score.home} - ${cons.predicted_score.away}</div>
          <div class="result-team">
            <div class="result-team-name">${p.away_team_zh}</div>
            <div class="result-team-en">${p.away_team}</div>
          </div>
        </div>
        <div style="text-align: right; font-size: 11px; color: var(--text-muted);">
          ${p.tournament}<br>
          ${p.neutral ? '中立场' : '主客场'}
        </div>
      </div>

      ${isDual ? '' : `
      <div class="result-section-title">Elo 等级分对比</div>
      <div class="elo-comparison">
        <div class="elo-team-label home">${cons.elo_ratings.home.toFixed(0)}</div>
        <div class="elo-bar">
          <div class="elo-bar-home" style="width: ${homeEloPct}%;"></div>
        </div>
        <div class="elo-team-label away">${cons.elo_ratings.away.toFixed(0)}</div>
      </div>
      `}

      ${isDual ? aggressiveHTML : `
      <div class="result-section-title">胜平负概率</div>
      <div class="result-probabilities">
        <div class="big-prob-bar">
          <div class="big-prob-segment prob-home" style="flex: ${probs.home_win * 100};">
            <div class="label">主胜</div>
            <div class="value">${homePct}%</div>
          </div>
          <div class="big-prob-segment prob-draw" style="flex: ${probs.draw * 100};">
            <div class="label">平局</div>
            <div class="value">${drawPct}%</div>
          </div>
          <div class="big-prob-segment prob-away" style="flex: ${probs.away_win * 100};">
            <div class="label">客胜</div>
            <div class="value">${awayPct}%</div>
          </div>
        </div>
        <div class="prob-legend">
          <span><span class="dot" style="background: var(--accent-success);"></span>${p.home_team_zh} 胜</span>
          <span><span class="dot" style="background: var(--accent-warning);"></span>平局</span>
          <span><span class="dot" style="background: var(--accent-danger);"></span>${p.away_team_zh} 胜</span>
        </div>
      </div>
      ${winRateSection}
      ${scorePredSection}
      ${topScoresSection}
      `}

      <div class="result-section-title">AI 分析</div>
      <div class="explanation-list">
        ${explanationHTML}
        ${agg ? agg.explanation.map(e => `<div class="explanation-item agg">${e}</div>`).join('') : ''}
      </div>

      <div class="result-disclaimer">
        <span class="result-disclaimer-icon">⚠️</span>
        <span>预测仅供参考，未考虑球员伤病、红牌、伤停补时、天气等临场因素。足球比赛存在不确定性，请理性看待。</span>
      </div>
    </div>
  `;
}

// ===========================
// 赛程加载（完整赛程：小组赛 + 淘汰赛）
// ===========================
let scheduleData = [];
let schedulePhaseFilter = 'all';
let scheduleDateFilter = 'all';      // 'all' | 'upcoming' | 'played' | 'YYYY-MM-DD'
let scheduleLastUpdated = null;      // 最近一次成功拉取时间
let scheduleRefreshTimer = null;     // 定时刷新 timer

// ── 工具：把 ISO 日期/时间字符串转成 'YYYY-MM-DD'（北京时间 UTC+8）
function toLocalDateStr(iso) {
  if (!iso) return '';
  // iso 有时带时区、有时不带，统一偏移到 +8
  const d = new Date(iso);
  if (isNaN(d.getTime())) return iso.slice(0, 10);
  const offset = 8 * 60;   // CST = UTC+8
  const local = new Date(d.getTime() + (offset - d.getTimezoneOffset()) * 60000);
  return local.toISOString().slice(0, 10);
}

// ── 工具：把 'YYYY-MM-DD' 格式化为 "6月11日（星期三）"
function formatDateLabel(dateStr) {
  if (!dateStr) return '';
  const [y, m, d] = dateStr.split('-').map(Number);
  const dt = new Date(y, m - 1, d);
  const weekDays = ['日', '一', '二', '三', '四', '五', '六'];
  const now = new Date();
  const nowStr = `${now.getFullYear()}-${String(now.getMonth()+1).padStart(2,'0')}-${String(now.getDate()).padStart(2,'0')}`;
  let extra = '';
  if (dateStr === nowStr) extra = ' · 今天';
  else {
    const tomorrow = new Date(now.getTime() + 86400000);
    const tStr = `${tomorrow.getFullYear()}-${String(tomorrow.getMonth()+1).padStart(2,'0')}-${String(tomorrow.getDate()).padStart(2,'0')}`;
    if (dateStr === tStr) extra = ' · 明天';
  }
  return `${m}月${d}日（周${weekDays[dt.getDay()]}）${extra}`;
}

async function loadSchedule() {
  const grid = $('#scheduleGrid');
  try {
    const data = await fetchJSON(`${API}/worldcup/full-schedule`);
    scheduleData = data.matches || [];
    scheduleLastUpdated = new Date();
    updateScheduleLastUpdatedUI();
    renderSchedule();
    setupScheduleFilters();
  } catch (e) {
    if (grid) grid.innerHTML = `
      <div class="loading-card glass">
        <p style="color: var(--accent-danger);">赛程加载失败</p>
        <p style="font-size: 12px; margin-top: 8px;">${e.message}</p>
      </div>
    `;
  }
}

// ── 更新"最后更新时间"文字
function updateScheduleLastUpdatedUI() {
  const el = $('#scheduleLastUpdated');
  if (!el || !scheduleLastUpdated) return;
  const h = String(scheduleLastUpdated.getHours()).padStart(2, '0');
  const min = String(scheduleLastUpdated.getMinutes()).padStart(2, '0');
  el.textContent = `上次更新：${h}:${min}`;
}

// ── 启动定时自动刷新（页面可见时每 10 分钟刷新一次赛程）
function startScheduleAutoRefresh() {
  if (scheduleRefreshTimer) clearInterval(scheduleRefreshTimer);
  scheduleRefreshTimer = setInterval(() => {
    if (document.visibilityState === 'visible') {
      loadSchedule();
    }
  }, 10 * 60 * 1000);   // 10 分钟
  // 标签页从后台切换回来时也触发一次
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
      const now = Date.now();
      const last = scheduleLastUpdated ? scheduleLastUpdated.getTime() : 0;
      // 超过 5 分钟未更新则刷新
      if (now - last > 5 * 60 * 1000) loadSchedule();
    }
  });
}

// ── 日期滚动箭头控制
function updateDateScrollArrows() {
  const container = $('#scheduleDateFilter');
  const leftArrow = $('#dateScrollLeft');
  const rightArrow = $('#dateScrollRight');
  if (!container || !leftArrow || !rightArrow) return;

  // 判断是否需要显示箭头（内容宽度 > 容器可视宽度）
  const needsScroll = container.scrollWidth > container.clientWidth + 2;
  const wrap = $('#scheduleDateScrollWrap');
  if (wrap) {
    wrap.classList.toggle('has-overflow', needsScroll);
  }

  if (!needsScroll) {
    leftArrow.classList.add('disabled');
    rightArrow.classList.add('disabled');
    return;
  }

  // 左侧箭头：未滚到最左时可用
  leftArrow.classList.toggle('disabled', container.scrollLeft <= 2);
  // 右侧箭头：未滚到最右时可用
  rightArrow.classList.toggle('disabled', container.scrollLeft + container.clientWidth >= container.scrollWidth - 2);
}

function scrollDateTabs(direction) {
  const container = $('#scheduleDateFilter');
  if (!container) return;
  const scrollAmount = container.clientWidth * 0.65;
  const target = direction === 'left'
    ? container.scrollLeft - scrollAmount
    : container.scrollLeft + scrollAmount;
  container.scrollTo({ left: target, behavior: 'smooth' });
}

function scrollActiveDateTabIntoView() {
  const container = $('#scheduleDateFilter');
  if (!container) return;
  const activeTab = container.querySelector('.date-tab.active');
  if (!activeTab) return;

  // 计算 active tab 在容器中的位置
  const tabLeft = activeTab.offsetLeft;
  const tabRight = tabLeft + activeTab.offsetWidth;
  const viewLeft = container.scrollLeft;
  const viewRight = viewLeft + container.clientWidth;

  // 如果 tab 在可视区域内，不需要滚动
  if (tabLeft >= viewLeft && tabRight <= viewRight) return;

  // 滚动使 active tab 居中
  const targetLeft = tabLeft - container.clientWidth / 2 + activeTab.offsetWidth / 2;
  container.scrollTo({ left: Math.max(0, targetLeft), behavior: 'smooth' });
}

function setupScheduleFilters() {
  const filterContainer = $('#scheduleFilters');
  const dateContainer = $('#scheduleDateFilter');
  if (!filterContainer || !dateContainer) return;

  // 避免重复绑定
  if (filterContainer.dataset.bound) return;
  filterContainer.dataset.bound = '1';

  filterContainer.addEventListener('click', (e) => {
    const tab = e.target.closest('.filter-tab');
    if (!tab) return;
    filterContainer.querySelectorAll('.filter-tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    schedulePhaseFilter = tab.dataset.phase;
    renderSchedule();
  });

  // 日期 tab 使用事件委托
  if (!dateContainer.dataset.bound) {
    dateContainer.dataset.bound = '1';
    dateContainer.addEventListener('click', (e) => {
      const tab = e.target.closest('.date-tab');
      if (!tab) return;
      dateContainer.querySelectorAll('.date-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      scheduleDateFilter = tab.dataset.date;
      renderSchedule();
    });

    // 监听滚动以更新箭头状态
    dateContainer.addEventListener('scroll', () => {
      updateDateScrollArrows();
    }, { passive: true });

    // 鼠标滚轮横向滚动（premium touch）
    dateContainer.addEventListener('wheel', (e) => {
      if (Math.abs(e.deltaY) > Math.abs(e.deltaX)) {
        e.preventDefault();
        dateContainer.scrollLeft += e.deltaY;
      }
    }, { passive: false });

    // 窗口 resize 时更新箭头
    window.addEventListener('resize', () => {
      updateDateScrollArrows();
    }, { passive: true });
  }

  // 左右滚动箭头
  const leftArrow = $('#dateScrollLeft');
  const rightArrow = $('#dateScrollRight');
  if (leftArrow && !leftArrow.dataset.bound) {
    leftArrow.dataset.bound = '1';
    leftArrow.addEventListener('click', () => scrollDateTabs('left'));
  }
  if (rightArrow && !rightArrow.dataset.bound) {
    rightArrow.dataset.bound = '1';
    rightArrow.addEventListener('click', () => scrollDateTabs('right'));
  }

  // 初始更新箭头状态
  updateDateScrollArrows();

  // 更新比分按钮
  const refreshBtn = $('#btnRefreshScores');
  if (refreshBtn && !refreshBtn.dataset.bound) {
    refreshBtn.dataset.bound = '1';
    refreshBtn.addEventListener('click', async () => {
      if (refreshBtn.classList.contains('loading')) return;
      refreshBtn.classList.add('loading');
      const originalHTML = refreshBtn.innerHTML;
      refreshBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/><path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16"/><path d="M16 16h5v5"/></svg> 更新中...';
      try {
        const resp = await fetch(`${API}/worldcup/update-scores`, { method: 'POST' });
        const data = await resp.json();
        if (data.status === 'ok') {
          refreshBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg> ' + (data.message || '已更新');
          await loadSchedule();
        } else {
          refreshBtn.innerHTML = originalHTML;
        }
      } catch (e) {
        refreshBtn.innerHTML = originalHTML;
      }
      setTimeout(() => {
        refreshBtn.classList.remove('loading');
        refreshBtn.innerHTML = originalHTML;
      }, 3000);
    });
  }
}

// ── 赛程卡片上的手动更新比分按钮
let scheduleEditOverlay = null;

function setupScheduleEditButtons() {
  const grid = $('#scheduleGrid');
  if (!grid) return;

  // 事件委托：点击编辑按钮
  grid.addEventListener('click', (e) => {
    const btn = e.target.closest('.schedule-edit-btn');
    if (!btn) return;
    e.stopPropagation();

    const matchDate = btn.dataset.date;
    const homeTeam = btn.dataset.home;
    const awayTeam = btn.dataset.away;

    showScoreEditPopup(btn, matchDate, homeTeam, awayTeam);
  });
}

function showScoreEditPopup(triggerBtn, matchDate, homeTeam, awayTeam) {
  // 移除旧弹窗
  if (scheduleEditOverlay) scheduleEditOverlay.remove();

  const existingData = findExistingScore(matchDate, homeTeam, awayTeam);

  const overlay = document.createElement('div');
  overlay.className = 'score-edit-overlay';
  overlay.innerHTML = `
    <div class="score-edit-popup glass">
      <div class="score-edit-header">
        <strong>${homeTeam} vs ${awayTeam}</strong>
        <span class="score-edit-date">${matchDate}</span>
      </div>
      <div class="score-edit-body">
        <div class="score-edit-inputs">
          <input type="number" class="score-input" id="editHomeScore" placeholder="主" min="0" max="20"
            value="${existingData ? existingData.home_score : ''}">
          <span class="score-edit-sep">:</span>
          <input type="number" class="score-input" id="editAwayScore" placeholder="客" min="0" max="20"
            value="${existingData ? existingData.away_score : ''}">
        </div>
      </div>
      <div class="score-edit-actions">
        <button class="btn-secondary" id="editCancel">取消</button>
        <button class="btn-primary" id="editSubmit">更新比分</button>
      </div>
      <div class="score-edit-status" id="editStatus"></div>
    </div>
  `;

  document.body.appendChild(overlay);
  scheduleEditOverlay = overlay;

  // 事件绑定
  overlay.querySelector('#editCancel').addEventListener('click', () => overlay.remove());
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) overlay.remove();
  });

  overlay.querySelector('#editSubmit').addEventListener('click', async () => {
    const hsEl = overlay.querySelector('#editHomeScore');
    const asEl = overlay.querySelector('#editAwayScore');
    const statusEl = overlay.querySelector('#editStatus');

    const hs = parseInt(hsEl.value);
    const as = parseInt(asEl.value);

    if (isNaN(hs) || isNaN(as) || hs < 0 || as < 0) {
      statusEl.textContent = '请输入有效的比分（≥0）';
      statusEl.className = 'score-edit-status error';
      return;
    }

    const submitBtn = overlay.querySelector('#editSubmit');
    submitBtn.disabled = true;
    submitBtn.textContent = '更新中...';
    statusEl.textContent = '';
    statusEl.className = 'score-edit-status';

    try {
      const params = new URLSearchParams({
        home_team: homeTeam,
        away_team: awayTeam,
        match_date: matchDate,
        home_score: hs,
        away_score: as,
      });
      const resp = await fetch(`${API}/worldcup/manual-score?${params}`, { method: 'POST' });
      const data = await resp.json();

      if (data.status === 'ok') {
        statusEl.textContent = `✓ ${data.message}`;
        statusEl.className = 'score-edit-status success';
        // 延迟关闭并刷新赛程
        setTimeout(() => {
          overlay.remove();
          loadSchedule();
        }, 1000);
      } else {
        statusEl.textContent = data.message || '更新失败';
        statusEl.className = 'score-edit-status error';
        submitBtn.disabled = false;
        submitBtn.textContent = '更新比分';
      }
    } catch (e) {
      statusEl.textContent = `网络错误: ${e.message}`;
      statusEl.className = 'score-edit-status error';
      submitBtn.disabled = false;
      submitBtn.textContent = '更新比分';
    }
  });

  // 自动聚焦第一个输入框
  setTimeout(() => overlay.querySelector('#editHomeScore')?.focus(), 100);
}

function findExistingScore(matchDate, homeTeam, awayTeam) {
  const match = scheduleData.find(m =>
    m.home_team === homeTeam &&
    m.away_team === awayTeam &&
    m.played
  );
  if (match) return { home_score: match.home_score, away_score: match.away_score };

  // 也检查可能互换的
  const swapped = scheduleData.find(m =>
    m.home_team === awayTeam &&
    m.away_team === homeTeam &&
    m.played
  );
  if (swapped) return { home_score: swapped.away_score, away_score: swapped.home_score };

  return null;
}

// ── 动态生成日期快速跳转 tabs
function buildDateTabs() {
  const dateContainer = $('#scheduleDateFilter');
  if (!dateContainer) return;

  // 从当前筛选后的数据提取所有出现的日期
  let pool = scheduleData;
  if (schedulePhaseFilter !== 'all') {
    pool = pool.filter(m => m.phase === schedulePhaseFilter);
  }

  const uniqueDates = [...new Set(pool.map(m => toLocalDateStr(m.date)).filter(Boolean))].sort();
  const today = (() => {
    const n = new Date();
    return `${n.getFullYear()}-${String(n.getMonth()+1).padStart(2,'0')}-${String(n.getDate()).padStart(2,'0')}`;
  })();

  // 固定前三个 tab，动态日期跟在后面
  const fixedTabs = [
    { date: 'all',      label: '全部' },
    { date: 'upcoming', label: '未开始' },
    { date: 'played',   label: '已完赛' },
  ];

  const dateTabs = uniqueDates.map(d => ({
    date: d,
    label: d === today ? `今天` : `${parseInt(d.slice(5,7))}/${parseInt(d.slice(8,10))}`,
  }));

  const allTabs = [...fixedTabs, ...dateTabs];

  dateContainer.innerHTML = allTabs.map(t => {
    const active = t.date === scheduleDateFilter ? ' active' : '';
    return `<button class="date-tab${active}" data-date="${t.date}" title="${t.date.length === 10 ? formatDateLabel(t.date) : ''}">${t.label}</button>`;
  }).join('');

  // 渲染完成后：自动滚动 active tab 到可见区域 + 更新箭头状态
  requestAnimationFrame(() => {
    scrollActiveDateTabIntoView();
    updateDateScrollArrows();
  });
}

function renderSchedule() {
  const grid = $('#scheduleGrid');
  if (!grid) return;

  // 重建日期 tab（轮次变化会影响可见日期范围）
  buildDateTabs();

  let filtered = scheduleData;

  // 轮次筛选
  if (schedulePhaseFilter !== 'all') {
    filtered = filtered.filter(m => m.phase === schedulePhaseFilter);
  }

  // 日期筛选
  if (scheduleDateFilter === 'played') {
    filtered = filtered.filter(m => m.played);
  } else if (scheduleDateFilter === 'upcoming') {
    filtered = filtered.filter(m => !m.played);
  } else if (scheduleDateFilter.length === 10 && scheduleDateFilter.includes('-')) {
    // 精确日期筛选
    filtered = filtered.filter(m => toLocalDateStr(m.date) === scheduleDateFilter);
  }

  if (filtered.length === 0) {
    grid.innerHTML = `<div class="schedule-empty glass">该筛选条件下暂无比赛</div>`;
    return;
  }

  // ── 按日期分组渲染
  const groups = {};
  filtered.forEach(m => {
    const key = toLocalDateStr(m.date) || 'unknown';
    if (!groups[key]) groups[key] = [];
    groups[key].push(m);
  });

  const sortedKeys = Object.keys(groups).sort();
  let html = '';

  sortedKeys.forEach(dateKey => {
    const matches = groups[dateKey];
    const labelHTML = dateKey !== 'unknown'
      ? `<div class="schedule-date-group-header">
           <span class="schedule-date-group-label">${formatDateLabel(dateKey)}</span>
           <span class="schedule-date-group-count">${matches.length} 场</span>
         </div>`
      : '';
    html += labelHTML;
    html += `<div class="schedule-date-group">`;
    html += matches.map(m => renderScheduleCard(m)).join('');
    html += `</div>`;
  });

  grid.innerHTML = html;
}

function renderScheduleCard(m) {
  const isKnockout = m.phase !== 'group';
  const isFinal = m.phase === 'final';
  const isTBD = m.home_team === 'TBD';
  const played = m.played;

  // 轮次标签
  let phaseTagClass = 'schedule-phase-tag';
  if (isKnockout) phaseTagClass += ' knockout';
  if (isFinal) phaseTagClass += ' final';

  let phaseLabel = m.phase_zh;

  // 比分区域
  let scoreArea;
  if (played) {
    const homeWin = m.home_score > m.away_score;
    const awayWin = m.away_score > m.home_score;
    scoreArea = `
      <div class="schedule-score">
        <span class="score-num ${homeWin ? 'win' : 'lose'}">${m.home_score}</span>
        <span class="score-sep">:</span>
        <span class="score-num ${awayWin ? 'win' : 'lose'}">${m.away_score}</span>
      </div>
    `;
  } else {
    scoreArea = `<div class="schedule-vs">VS</div>`;
  }

  // 场馆信息
  const venueParts = [];
  if (m.stadium) venueParts.push(m.stadium);
  if (m.city) venueParts.push(m.city);
  if (m.country) venueParts.push(m.country);
  const venueStr = venueParts.join(' · ');

  const teamClass = isTBD ? 'schedule-team tbd' : 'schedule-team';
  const cardClass = `schedule-card glass${played ? ' played' : ''}${isKnockout ? ' knockout' : ''}`;

  // 编辑按钮（非 TBD 比赛显示）
  const editBtn = isTBD ? '' : `
    <button class="schedule-edit-btn" title="手动更新比分"
      data-match-id="${m.match_id || ''}"
      data-date="${m.date.split('T')[0] || ''}"
      data-home="${m.home_team}"
      data-away="${m.away_team}"
      data-played="${played}"
    >✎</button>`;

  return `
    <div class="${cardClass}" data-home="${m.home_team}" data-away="${m.away_team}" data-match-id="${m.match_id || ''}">
      <span class="${phaseTagClass}">${phaseLabel}</span>
      ${editBtn}
      <div class="schedule-date">
        <span class="schedule-date-text">${formatDate(m.date)}</span>
      </div>
      <div class="schedule-match">
        <div class="${teamClass}">
          <div class="schedule-team-zh">${m.home_team_zh}</div>
          <div class="schedule-team-en">${m.home_team}</div>
        </div>
        ${scoreArea}
        <div class="${teamClass}">
          <div class="schedule-team-zh">${m.away_team_zh}</div>
          <div class="schedule-team-en">${m.away_team}</div>
        </div>
      </div>
      <div class="schedule-venue">${venueStr}</div>
    </div>
  `;
}

// ===========================
// 最近结果加载
// ===========================
async function loadRecentResults() {
  const list = $('#resultsList');
  try {
    const data = await fetchJSON(`${API}/results/recent?limit=15`);
    if (data.length === 0) {
      list.innerHTML = `<div class="loading-card glass"><p>暂无数据</p></div>`;
      return;
    }
    list.innerHTML = data.map(r => {
      const homeWinner = r.result === 'home_win';
      const awayWinner = r.result === 'away_win';
      return `
        <div class="result-row">
          <div class="result-date">${formatDate(r.date)}</div>
          <div class="result-team home ${homeWinner ? 'winner' : ''}">${r.home_team_zh}</div>
          <div class="result-score">${r.home_score} - ${r.away_score}</div>
          <div class="result-team away ${awayWinner ? 'winner' : ''}">${r.away_team_zh}</div>
          <div class="result-tournament">${r.tournament}</div>
        </div>
      `;
    }).join('');
  } catch (e) {
    list.innerHTML = `<div class="loading-card glass"><p>加载失败: ${e.message}</p></div>`;
  }
}

// ===========================
// 模型信息加载
// ===========================
async function loadModelInfo() {
  try {
    const data = await fetchJSON(`${API}/health`);
    const m = data.model_loaded ? data.training_metadata : null;
    if (m) {
      const accEl = $('#statAccuracy');
      const teamsEl = $('#statTeams');
      if (accEl) accEl.textContent = (m.outcome_accuracy * 100).toFixed(1) + '%';
      if (teamsEl) teamsEl.textContent = m.total_teams;
      const metrics = [
        { value: (m.outcome_accuracy * 100).toFixed(1) + '%', label: '胜平负准确率' },
        { value: m.outcome_log_loss.toFixed(3), label: 'Log Loss' },
        { value: m.home_score_mae.toFixed(2), label: '主队进球MAE' },
        { value: m.away_score_mae.toFixed(2), label: '客队进球MAE' },
        { value: m.train_size.toLocaleString(), label: '训练样本' },
        { value: m.test_size.toLocaleString(), label: '测试样本' },
      ];
      const metricsGrid = $('#metricsGrid');
      if (metricsGrid) {
        metricsGrid.innerHTML = metrics.map(m => `
          <div class="metric">
            <div class="metric-value">${m.value}</div>
            <div class="metric-label">${m.label}</div>
          </div>
        `).join('');
      }
    }
  } catch (e) {
    console.error('加载模型信息失败:', e);
  }
}

// ===========================
// 刷新
// ===========================
async function refresh() {
  const btn = $('#refreshBtn');
  btn.classList.add('spinning');
  await Promise.all([
    loadLiveMatches(),
    loadSchedule(),
    loadRecentResults(),
  ]);
  setTimeout(() => btn.classList.remove('spinning'), 1000);
}

// ===========================
// 赛程卡片点击 -> 自动填充预测
// ===========================
function setupScheduleClicks() {
  document.addEventListener('click', (e) => {
    const card = e.target.closest('.schedule-card');
    if (!card) return;
    const home = card.dataset.home;
    const away = card.dataset.away;
    if (!home || !away || home === 'TBD' || away === 'TBD') return;
    const predictEl = $('#predict');
    if (predictEl) predictEl.scrollIntoView({ behavior: 'smooth' });
    setTimeout(() => {
      const homeTeam = state.teamsCache.find(t => t.team_en === home);
      const awayTeam = state.teamsCache.find(t => t.team_en === away);
      const hd = $('#homeDisplay');
      const ad = $('#awayDisplay');
      if (homeTeam) {
        state.homeTeam = homeTeam.team_en;
        if (hd) hd.innerHTML = `
          <div class="team-name-zh">${homeTeam.team_zh}</div>
          <div class="team-name-en">${homeTeam.team_en}</div>
          <div class="team-elo">Elo ${homeTeam.elo}</div>
        `;
      }
      if (awayTeam) {
        state.awayTeam = awayTeam.team_en;
        if (ad) ad.innerHTML = `
          <div class="team-name-zh">${awayTeam.team_zh}</div>
          <div class="team-name-en">${awayTeam.team_en}</div>
          <div class="team-elo">Elo ${awayTeam.elo}</div>
        `;
      }
      const ts = $('#tournamentSelect');
      const ns = $('#neutralSelect');
      if (ts) ts.value = 'FIFA World Cup';
      if (ns) ns.value = 'true';
      predictMatch();
    }, 500);
  });
}

// ===========================
// 导航高亮
// ===========================
function setupNavHighlight() {
  const sections = ['live', 'predict', 'schedule', 'results', 'about'];
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const id = entry.target.id;
        $$('.nav-link').forEach(link => {
          link.classList.toggle('active', link.dataset.section === id);
        });
      }
    });
  }, { rootMargin: '-30% 0px -60% 0px' });
  sections.forEach(id => {
    const el = $(`#${id}`);
    if (el) observer.observe(el);
  });
}

// ===========================
// LLM 配置面板
// ===========================
const PROVIDER_PRESETS = {
  openai: { base_url: 'https://api.openai.com/v1', model: 'gpt-4o-mini' },
  deepseek: { base_url: 'https://api.deepseek.com/v1', model: 'deepseek-chat' },
  qwen: { base_url: 'https://dashscope.aliyuncs.com/compatible-mode/v1', model: 'qwen-plus' },
  zhipu: { base_url: 'https://open.bigmodel.cn/api/paas/v4', model: 'glm-4-flash' },
  moonshot: { base_url: 'https://api.moonshot.cn/v1', model: 'moonshot-v1-8k' },
  custom: { base_url: '', model: '' },
};

async function openConfigModal() {
  const modal = $('#configModal');
  if (modal) modal.classList.add('open');
  try {
    const resp = await fetchJSON(`${API}/config/llm`);
    const prov = $('#configProvider');
    const bu = $('#configBaseUrl');
    const cm = $('#configModel');
    const ck = $('#configApiKey');
    const kh = $('#configKeyHint');
    const cs = $('#configStatus');
    if (prov) prov.value = resp.provider || 'openai';
    if (bu) bu.value = resp.base_url || '';
    if (cm) cm.value = resp.model || '';
    if (ck) ck.value = '';  // ★ 始终留空，让用户手动输入（安全策略）
    if (kh) kh.textContent = resp.api_key_masked ? `当前: ${resp.api_key_masked}` : '未配置';
    // 记录服务端是否已有 key（测试按钮需要）
    state._serverHasKey = !!resp.api_key_masked;
    if (cs) { cs.className = 'config-status'; cs.textContent = ''; }
  } catch (e) {
    console.error('加载配置失败:', e);
  }
}

function closeConfigModal() {
  const modal = $('#configModal');
  if (modal) modal.classList.remove('open');
}

// 按钮绑定
const configBtn = $('#configBtn');
const configModalClose = $('#configModalClose');
const configModal = $('#configModal');
if (configBtn) configBtn.addEventListener('click', openConfigModal);
if (configModalClose) configModalClose.addEventListener('click', closeConfigModal);
if (configModal) {
  configModal.addEventListener('click', (e) => {
    if (e.target === configModal) closeConfigModal();
  });
}

const configProvider = $('#configProvider');
if (configProvider) {
  configProvider.addEventListener('change', () => {
    const provider = configProvider.value;
    const preset = PROVIDER_PRESETS[provider];
    const bu = $('#configBaseUrl');
    const cm = $('#configModel');
    if (preset && preset.base_url && bu) bu.value = preset.base_url;
    if (preset && preset.model && cm) cm.value = preset.model;
  });
}

const configSaveBtn = $('#configSaveBtn');
if (configSaveBtn) {
  configSaveBtn.addEventListener('click', async () => {
    const config = {
      provider: ($('#configProvider')||{}).value,
      base_url: ($('#configBaseUrl')||{}).value?.trim() || '',
      api_key: ($('#configApiKey')||{}).value?.trim() || '',
      model: ($('#configModel')||{}).value?.trim() || '',
    };
    try {
      await fetchJSON(`${API}/config/llm`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });
      showConfigStatus('配置已保存 ✓', 'success');
      setTimeout(closeConfigModal, 1000);
    } catch (e) {
      showConfigStatus(`保存失败: ${e.message}`, 'error');
    }
  });
}

const configTestBtn = $('#configTestBtn');
if (configTestBtn) {
  configTestBtn.addEventListener('click', async () => {
    const btn = configTestBtn;
    const provider = ($('#configProvider')||{}).value;
    const baseUrl = ($('#configBaseUrl')||{}).value?.trim() || '';
    const apiKey = ($('#configApiKey')||{}).value?.trim() || '';
    const model = ($('#configModel')||{}).value?.trim() || '';

    // ★ 如果输入框为空但服务端已有 key → 允许测试
    if (!apiKey && !state._serverHasKey) {
      showConfigStatus('请先填写 API Key', 'error'); return;
    }
    if (!baseUrl) { showConfigStatus('请填写 Base URL', 'error'); return; }

    btn.disabled = true;
    btn.textContent = '保存并测试...';
    showConfigStatus('正在保存配置...', 'info');

    try {
      const saveResp = await fetch(`${API}/config/llm`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ provider, base_url: baseUrl, api_key: apiKey, model }),
      });
      if (!saveResp.ok) {
        showConfigStatus('配置保存失败', 'error');
        btn.disabled = false;
        btn.textContent = '测试连接';
        return;
      }
    } catch (e) {
      showConfigStatus(`保存异常: ${e.message}`, 'error');
      btn.disabled = false;
      btn.textContent = '测试连接';
      return;
    }

    showConfigStatus('正在测试连接...', 'info');

    try {
      const response = await fetch(`${API}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: '你好，请简单回答"连接成功"', thinking: false, web_search: false }),
      });

      if (!response.ok) {
        const text = await response.text();
        showConfigStatus(`连接失败 (HTTP ${response.status})`, 'error');
        btn.disabled = false;
        btn.textContent = '测试连接';
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let hasError = false;
      let errorMsg = '';
      let hasContent = false;

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop();
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.substring(6));
                if (data.type === 'error') {
                  hasError = true;
                  errorMsg = data.error || data.detail || '未知错误';
                } else if (data.type === 'text') {
                  hasContent = true;
                }
              } catch (e) { /* ignore parse errors */ }
            }
          }
        }
      } finally {
        try { reader.releaseLock(); } catch (e) { /* already released */ }
      }

      if (hasError) {
        showConfigStatus(`连接失败: ${errorMsg}`, 'error');
      } else if (hasContent) {
        showConfigStatus('连接成功 ✓', 'success');
      } else {
        showConfigStatus('连接异常: 未收到有效响应', 'error');
      }
    } catch (e) {
      showConfigStatus(`连接异常: ${e.message}`, 'error');
    } finally {
      btn.disabled = false;
      btn.textContent = '测试连接';
    }
  });
}

function showConfigStatus(msg, type) {
  const el = $('#configStatus');
  if (el) {
    el.className = `config-status ${type}`;
    el.textContent = msg;
  }
}

// ===========================
// AI问赛 聊天助手
// ===========================

function openChatPanel() {
  const panel = $('#chatPanel');
  const bubble = $('#chatBubble');
  if (panel) panel.classList.add('open');
  if (bubble) bubble.style.display = 'none';
  setTimeout(() => {
    const input = $('#chatInput');
    if (input) input.focus();
  }, 300);
}

function closeChatPanel() {
  const panel = $('#chatPanel');
  const bubble = $('#chatBubble');
  if (panel) panel.classList.remove('open');
  if (bubble) bubble.style.display = 'flex';
}

function toggleChatPanel() {
  const panel = $('#chatPanel');
  if (panel && panel.classList.contains('open')) {
    closeChatPanel();
  } else {
    openChatPanel();
  }
}

const chatBubble = $('#chatBubble');
const chatPanelClose = $('#chatPanelClose');
if (chatBubble) chatBubble.addEventListener('click', toggleChatPanel);
if (chatPanelClose) chatPanelClose.addEventListener('click', closeChatPanel);

// ★ 清除历史对话 —— 一站式清理，无需二次确认
function clearChatHistory() {
  const messages = $('#chatMessages');
  if (!messages) return;
  messages.innerHTML = `
    <div class="chat-welcome">
      <div class="chat-welcome-icon">⚽</div>
      <p>你好！我是 <strong>AI问赛</strong>，你的世界杯足球AI助手。</p>
      <p>可以问我关于任何比赛的分析、预测、球队数据等问题。</p>
      <div class="chat-quick-questions">
        <button class="quick-q" data-q="法国队和巴西队谁更强？分析两队实力对比。">🇫🇷 法国 vs 🇧🇷 巴西 实力分析</button>
        <button class="quick-q" data-q="请分析本届世界杯的夺冠热门球队，并给出理由。">🏆 本届世界杯夺冠热门</button>
        <button class="quick-q" data-q="阿根廷队近期的状态如何？分析他们的优势和劣势。">🇦🇷 阿根廷队近期状态</button>
      </div>
    </div>
  `;
  // 完全重置状态
  state.chatMatchContext = null;
  state.chatStreaming = false;
  if (state.chatAbortController) {
    state.chatAbortController.abort();
    state.chatAbortController = null;
  }
  const sendBtn = $('#chatSendBtn');
  const stopBtn = $('#chatStopBtn');
  if (sendBtn) sendBtn.style.display = 'flex';
  if (stopBtn) stopBtn.style.display = 'none';
  showToast('历史对话已清除');
}

// ★ 清除按钮 —— 直接清除（垃圾桶图标已表明意图）
const chatClearBtn = $('#chatClearBtn');
if (chatClearBtn) {
  chatClearBtn.addEventListener('click', () => {
    // 正在生成时不允许清除
    if (state.chatStreaming) {
      showToast('正在生成回复，请稍后再试');
      return;
    }
    // 检查是否有历史对话
    const messages = $('#chatMessages');
    if (!messages || !messages.querySelector('.chat-message')) {
      showToast('暂无历史对话');
      return;
    }
    clearChatHistory();
  });
}

// 轻量 toast 提示
function showToast(text, duration) {
  if (duration === undefined) duration = 1800;
  let toast = $('#chatToast');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'chatToast';
    toast.className = 'chat-toast';
    document.body.appendChild(toast);
  }
  toast.textContent = text;
  toast.classList.add('show');
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => toast.classList.remove('show'), duration);
}

// 快捷问题 — 事件委托
document.addEventListener('click', (e) => {
  const qBtn = e.target.closest('.quick-q');
  if (!qBtn) return;
  const q = qBtn.dataset.q;
  if (q) {
    const messages = $('#chatMessages');
    const welcome = messages ? messages.querySelector('.chat-welcome') : null;
    if (welcome) welcome.remove();
    sendChatMessage(q);
  }
});

// 发送按钮
const chatSendBtn = $('#chatSendBtn');
if (chatSendBtn) {
  chatSendBtn.addEventListener('click', () => {
    const input = $('#chatInput');
    if (!input) return;
    const msg = input.value.trim();
    if (!msg || state.chatStreaming) return;
    input.value = '';
    sendChatMessage(msg);
  });
}

// 回车发送
const chatInput = $('#chatInput');
if (chatInput) {
  chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      const sendBtn = $('#chatSendBtn');
      if (sendBtn) sendBtn.click();
    }
  });
}

// ★ SSE 批量渲染阈值
const CHAT_RENDER_INTERVAL_MS = 80;   // 最多每80ms渲染一次
const CHAT_RENDER_CHAR_THRESHOLD = 60; // 或每累积60个字符

async function sendChatMessage(message) {
  if (state.chatStreaming) return;

  const messagesContainer = $('#chatMessages');
  if (!messagesContainer) return;

  // 移除欢迎界面
  const welcome = messagesContainer.querySelector('.chat-welcome');
  if (welcome) welcome.remove();

  // 添加用户消息
  appendChatMessage('user', message);
  state.chatStreaming = true;

  // 添加AI消息容器
  const assistantContainer = appendAssistantContainer();

  // 获取开关状态
  const thinking = ($('#chatThinkingToggle')||{}).checked || false;
  const webSearch = ($('#chatSearchToggle')||{}).checked || false;

  // 切换发送/停止按钮
  const sendBtn = $('#chatSendBtn');
  const stopBtn = $('#chatStopBtn');
  if (sendBtn) sendBtn.style.display = 'none';
  if (stopBtn) stopBtn.style.display = 'flex';

  // AbortController
  const controller = new AbortController();
  state.chatAbortController = controller;

  // 状态追踪
  let thinkingEl = null;
  let thinkingContent = '';
  let answerContent = '';
  let fullText = '';

  // ★ 批量渲染相关
  state.lastChatRenderAt = performance.now();
  let renderScheduled = false;
  let streamDone = false;

  function scheduleRender() {
    if (renderScheduled) return;
    renderScheduled = true;
    requestAnimationFrame(() => {
      renderScheduled = false;
      const now = performance.now();
      // 仅在间隔足够或流结束时渲染
      if (now - state.lastChatRenderAt >= CHAT_RENDER_INTERVAL_MS || streamDone) {
        doRender();
        state.lastChatRenderAt = now;
      }
    });
  }

  function doRender() {
    if (thinking) {
      let remaining = fullText;
      const thinkMatch = remaining.match(/<thinking>([\s\S]*?)<\/thinking>/);
      const answerMatch = remaining.match(/<answer>([\s\S]*?)(?:<\/answer>|$)/);

      if (thinkMatch) {
        thinkingContent = thinkMatch[1];
        if (!thinkingEl) thinkingEl = createThinkingUI(assistantContainer);
        updateThinkingUI(thinkingEl, thinkingContent);
      }
      if (answerMatch) {
        answerContent = answerMatch[1];
        setAnswerText(assistantContainer, answerContent, true);
      } else if (!thinkMatch) {
        answerContent = remaining;
        setAnswerText(assistantContainer, answerContent, true);
      }
    } else {
      answerContent = fullText;
      setAnswerText(assistantContainer, answerContent, true);
    }

    // ★ 批量渲染后始终滚到底（用户若手动上滚超过50px则不强制跟底）
    const msgContainer = $('#chatMessages');
    if (msgContainer && isNearBottom(msgContainer)) {
      msgContainer.scrollTo({ top: msgContainer.scrollHeight, behavior: 'instant' });
    }
  }

  // ★ 判断用户是否在底部（阈值50px内视为"在底部"）
  function isNearBottom(el) {
    return el.scrollHeight - el.scrollTop - el.clientHeight < 50;
  }

  let reader = null;

  try {
    const body = {
      message,
      thinking,
      web_search: webSearch,
      match_context: state.chatMatchContext || null,
    };
    state.chatMatchContext = null;

    const response = await fetch(`${API}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: controller.signal,
    });

    if (!response.ok) {
      const errText = await response.text();
      setAnswerText(assistantContainer, `**错误**: ${errText.substring(0, 200)}`, false);
      return;
    }

    reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.substring(6));
            if (data.type === 'error') {
              setAnswerText(assistantContainer, `**错误**: ${data.error}`, false);
              return;
            } else if (data.type === 'text') {
              fullText += data.content;
              scheduleRender();
            } else if (data.type === 'done') {
              // 流结束
            }
          } catch (e) {
            // 忽略解析错误
          }
        }
      }
    }

    // 流结束：最后一次完整渲染
    streamDone = true;
    fullText += ''; // 确保最后一次 schedule 能通过阈值检查
    doRender();
    removeCursor(assistantContainer);

  } catch (e) {
    if (e.name === 'AbortError') {
      const currentAnswer = assistantContainer.querySelector('.answer-text');
      if (currentAnswer) {
        currentAnswer.innerHTML = renderMarkdown(answerContent || '') + ' <em style="color:var(--text-muted);font-size:11px;">(已停止生成)</em>';
      } else {
        setAnswerText(assistantContainer, '*(已停止生成)*', false);
      }
    } else {
      setAnswerText(assistantContainer, `**网络错误**: ${e.message}`, false);
    }
  } finally {
    // ★ 确保 reader 释放
    if (reader) {
      try { reader.releaseLock(); } catch (e) { /* already released */ }
    }
    state.chatStreaming = false;
    state.chatAbortController = null;
    const sBtn = $('#chatSendBtn');
    const stBtn = $('#chatStopBtn');
    if (sBtn) sBtn.style.display = 'flex';
    if (stBtn) stBtn.style.display = 'none';
    removeCursor(assistantContainer);
  }
}

// 中止对话
function stopChatMessage() {
  if (state.chatAbortController) {
    state.chatAbortController.abort();
  }
}

// 创建AI回答容器
function appendAssistantContainer() {
  const container = $('#chatMessages');
  const div = document.createElement('div');
  div.className = 'chat-message assistant';
  div.innerHTML = `
    <div class="chat-assistant-body">
      <div class="thinking-area" style="display:none;"></div>
      <div class="answer-area">
        <div class="answer-text chat-bubble-msg streaming-cursor"></div>
      </div>
    </div>
    <div class="chat-message-time">${new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })}</div>
  `;
  container.appendChild(div);
  // ★ 使用 scrollTo 代替 scrollTop 避免强制 reflow
  container.scrollTo({ top: container.scrollHeight, behavior: 'instant' });
  return div;
}

// 创建可折叠的Thinking UI
function createThinkingUI(container) {
  const area = container.querySelector('.thinking-area');
  area.style.display = 'block';
  area.innerHTML = `
    <button class="thinking-toggle open" onclick="
      this.classList.toggle('open');
      this.nextElementSibling.classList.toggle('open');
    ">
      <span class="thinking-toggle-icon">▶</span>
      🧠 AI思考中...
    </button>
    <div class="thinking-content open"></div>
  `;
  return area.querySelector('.thinking-content');
}

function updateThinkingUI(el, text) {
  if (el) {
    el.textContent = text;
    const container = $('#chatMessages');
    if (container) container.scrollTo({ top: container.scrollHeight, behavior: 'instant' });
  }
}

// ★ 设置回答文本 — 支持增量渲染（skipScroll=true时跳过滚动）
function setAnswerText(container, text, skipScroll) {
  const answerEl = container.querySelector('.answer-text');
  if (answerEl) {
    answerEl.innerHTML = renderMarkdown(text);
    answerEl.classList.add('streaming-cursor');
    if (!skipScroll) {
      const msgContainer = $('#chatMessages');
      if (msgContainer) msgContainer.scrollTo({ top: msgContainer.scrollHeight, behavior: 'instant' });
    }
  }
  const toggle = container.querySelector('.thinking-toggle');
  if (toggle && toggle.classList.contains('open')) {
    const contentEl = container.querySelector('.thinking-content');
    toggle.innerHTML = `<span class="thinking-toggle-icon">▶</span> 🧠 思考过程（${(contentEl?.textContent?.length || 0)} 字）`;
  }
}

function removeCursor(container) {
  const answerEl = container.querySelector('.answer-text');
  if (answerEl) answerEl.classList.remove('streaming-cursor');
}

// Markdown渲染
function renderMarkdown(text) {
  if (!text) return '';
  try {
    if (typeof marked !== 'undefined' && marked.parse) {
      return marked.parse(text, { breaks: true });
    }
  } catch (e) {
    console.warn('marked.parse failed, fallback to basic:', e);
  }
  return text
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/`(.+?)`/g, '<code>$1</code>')
    .replace(/\n/g, '<br>');
}

function appendChatMessage(role, content) {
  const container = $('#chatMessages');
  const div = document.createElement('div');
  div.className = `chat-message ${role}`;
  div.innerHTML = `
    <div class="chat-bubble-msg">${escapeHtml(content)}</div>
    <div class="chat-message-time">${new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })}</div>
  `;
  container.appendChild(div);
  container.scrollTo({ top: container.scrollHeight, behavior: 'instant' });
  return div;
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// ===========================
// AI问赛 按钮处理（live卡片中的按钮）
// ===========================
document.addEventListener('click', async (e) => {
  const btn = e.target.closest('.ai-ask-btn');
  if (!btn) return;

  const home = btn.dataset.home;
  const away = btn.dataset.away;
  const status = btn.dataset.status;
  const homeZh = btn.dataset.homeZh;
  const awayZh = btn.dataset.awayZh;
  const homeScore = btn.dataset.homeScore;
  const awayScore = btn.dataset.awayScore;
  const matchDate = btn.dataset.date || '';
  const matchVenue = btn.dataset.venue || '';

  if (!home || !away) return;

  // ★ 构建分数上下文（已结束/进行中比赛携带实际比分）
  const scoreContext = { home_team: home, away_team: away, home_team_zh: homeZh, away_team_zh: awayZh, status };
  if (homeScore !== undefined && homeScore !== '' && homeScore !== null) {
    scoreContext.home_score = homeScore;
    scoreContext.away_score = awayScore;
  }
  if (matchDate) scoreContext.date = matchDate;
  if (matchVenue) scoreContext.venue = matchVenue;

  // 构建API URL，附带比分信息
  let apiUrl = `${API}/chat/match-analysis?home=${encodeURIComponent(home)}&away=${encodeURIComponent(away)}&status=${encodeURIComponent(status)}`;
  if (homeScore !== undefined && homeScore !== '' && homeScore !== null) {
    apiUrl += `&home_score=${encodeURIComponent(homeScore)}&away_score=${encodeURIComponent(awayScore)}`;
  }

  try {
    const resp = await fetchJSON(apiUrl);

    // 合并后端预测数据到上下文
    if (resp.prediction) {
      scoreContext.prediction = resp.prediction;
    }

    state.chatMatchContext = scoreContext;
    openChatPanel();

    const messages = $('#chatMessages');
    const welcome = messages ? messages.querySelector('.chat-welcome') : null;
    if (welcome) welcome.remove();

    sendChatMessage(resp.question);
  } catch (e) {
    console.error('获取分析提示失败:', e);
    state.chatMatchContext = scoreContext;
    openChatPanel();
    const messages = $('#chatMessages');
    const welcome = messages ? messages.querySelector('.chat-welcome') : null;
    if (welcome) welcome.remove();

    // ★ 已结束/进行中：携带实际比分到问题中
    let q;
    if (status === 'finished' || (status === 'live' && homeScore !== undefined && homeScore !== '')) {
      const scoreStr = (homeScore !== undefined && homeScore !== '') ? `（最终比分 ${homeScore}-${awayScore}）` : '';
      q = `请分析 ${homeZh||home} vs ${awayZh||away} 这场比赛的过程和胜负原因。${scoreStr}`;
    } else {
      q = `请预测 ${homeZh||home} vs ${awayZh||away} 这场比赛的结果。`;
    }
    sendChatMessage(q);
  }
});

// ===========================
// 自定义下拉选择器 — 替换原生 <select>，弹出菜单完全可控制样式
// ===========================
function initCustomSelects() {
  document.querySelectorAll('select.option-select').forEach(select => {
    // 防重复初始化
    if (select.closest('.custom-select-wrapper')) return;

    const wrapper = document.createElement('div');
    wrapper.className = 'custom-select-wrapper';

    // Trigger — 显示当前选中项
    const trigger = document.createElement('div');
    trigger.className = 'custom-select-trigger';
    trigger.setAttribute('tabindex', '0');
    const selectedOpt = select.options[select.selectedIndex];
    trigger.textContent = selectedOpt ? selectedOpt.textContent : '';

    // Dropdown — 选项列表
    const dropdown = document.createElement('div');
    dropdown.className = 'custom-select-dropdown';

    for (const opt of select.options) {
      const item = document.createElement('div');
      item.className = 'custom-select-option';
      if (opt.selected) item.classList.add('selected');
      item.textContent = opt.textContent;
      item.dataset.value = opt.value;
      dropdown.appendChild(item);
    }

    wrapper.appendChild(trigger);
    wrapper.appendChild(dropdown);

    // 插入 DOM，隐藏原生 select
    select.parentNode.insertBefore(wrapper, select);
    wrapper.appendChild(select);
    select.style.display = 'none';

    // ---- 事件绑定 ----

    // 监听原生 select change → 同步 trigger 文字（处理 JS 直接 set value 的场景）
    select.addEventListener('change', () => {
      const opt = select.options[select.selectedIndex];
      if (opt) trigger.textContent = opt.textContent;
      dropdown.querySelectorAll('.custom-select-option').forEach(o => {
        o.classList.toggle('selected', o.dataset.value === select.value);
      });
    });

    // 点击 trigger 开关
    trigger.addEventListener('click', (e) => {
      e.stopPropagation();
      const isOpen = wrapper.classList.contains('open');
      closeAllDropdowns();
      if (!isOpen) wrapper.classList.add('open');
    });

    // 点击 option
    dropdown.addEventListener('click', (e) => {
      const item = e.target.closest('.custom-select-option');
      if (!item) return;
      // 更新选中态
      dropdown.querySelectorAll('.custom-select-option').forEach(o => o.classList.remove('selected'));
      item.classList.add('selected');
      trigger.textContent = item.textContent;
      // 同步原生 select 值并触发 change 事件
      select.value = item.dataset.value;
      select.dispatchEvent(new Event('change', { bubbles: true }));
      wrapper.classList.remove('open');
    });

    // 键盘支持
    trigger.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        const isOpen = wrapper.classList.contains('open');
        closeAllDropdowns();
        if (!isOpen) wrapper.classList.add('open');
      }
      if (e.key === 'Escape') {
        wrapper.classList.remove('open');
        trigger.blur();
      }
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        if (!wrapper.classList.contains('open')) wrapper.classList.add('open');
        const items = dropdown.querySelectorAll('.custom-select-option');
        const idx = Array.from(items).findIndex(o => o.classList.contains('selected'));
        const next = items[Math.min(idx + 1, items.length - 1)];
        if (next) {
          items.forEach(o => o.classList.remove('selected'));
          next.classList.add('selected');
          next.scrollIntoView({ block: 'nearest' });
        }
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        if (!wrapper.classList.contains('open')) wrapper.classList.add('open');
        const items = dropdown.querySelectorAll('.custom-select-option');
        const idx = Array.from(items).findIndex(o => o.classList.contains('selected'));
        const prev = items[Math.max(idx - 1, 0)];
        if (prev) {
          items.forEach(o => o.classList.remove('selected'));
          prev.classList.add('selected');
          prev.scrollIntoView({ block: 'nearest' });
        }
      }
    });
  });
}

function closeAllDropdowns() {
  document.querySelectorAll('.custom-select-wrapper.open').forEach(w => w.classList.remove('open'));
}

// 全局点击关闭
document.addEventListener('click', (e) => {
  if (!e.target.closest('.custom-select-wrapper')) {
    closeAllDropdowns();
  }
});

// ===========================
// 初始化
// ===========================
async function init() {
  // 页脚时间
  const ft = $('#footerTime');
  if (ft) ft.textContent = new Date().toLocaleString('zh-CN');

  // 自定义下拉选择器 — 替换所有原生 select.option-select
  initCustomSelects();

  // ★ 一次性设置轮播事件委托（箭头点击，不再自动滚动）
  setupCarouselDelegation();

  // 加载模型信息
  loadModelInfo();

  // 加载球队
  await loadTeams();

  // 加载各模块
  loadLiveMatches();
  loadSchedule();
  loadRecentResults();

  // 启动赛程自动刷新（每 10 分钟 + 标签页可见时）
  startScheduleAutoRefresh();

  // 设置赛程卡片上的手动比分编辑按钮事件
  setupScheduleEditButtons();

  // 设置搜索
  setupTeamSearch('homeInput', 'homeSuggestions', 'home');
  setupTeamSearch('awayInput', 'awaySuggestions', 'away');

  // 预测按钮
  const predictBtn = $('#predictBtn');
  if (predictBtn) predictBtn.addEventListener('click', predictMatch);

  // 刷新按钮
  const refreshBtn = $('#refreshBtn');
  if (refreshBtn) refreshBtn.addEventListener('click', refresh);

  // 赛程卡片点击
  setupScheduleClicks();

  // 导航高亮
  setupNavHighlight();

  // 定时刷新实时数据（每5分钟）
  state.refreshTimer = setInterval(loadLiveMatches, 30 * 60 * 1000);
}

document.addEventListener('DOMContentLoaded', init);

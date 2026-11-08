// Rating page logic. Rendered into page.html by students.shared.human_eval;
// orderFor is pure and exported for tests (node), everything else runs only
// in a browser.

function hashString(s) {
  let h = 2166136261 >>> 0;                       // FNV-1a
  for (const ch of s) {
    h ^= ch.codePointAt(0);
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h;
}

function mulberry32(seed) {
  return function () {
    seed = (seed + 0x6D2B79F5) >>> 0;
    let t = seed;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// A rater-specific order of 0..n-1 in which no two neighbours share a group
// (a pair's outputs never follow each other). Same name, same order.
function orderFor(name, n, groupOf) {
  const rnd = mulberry32(hashString(String(name)));
  const idx = Array.from({length: n}, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  const out = [];
  const rest = idx.slice();
  while (rest.length) {
    const last = out.length ? groupOf[out[out.length - 1]] : null;
    const k = rest.findIndex(x => groupOf[x] !== last);
    if (k < 0) break;
    out.push(rest.splice(k, 1)[0]);
  }
  for (const x of rest) {                         // leftovers of one group: slot them in
    let placed = false;
    for (let i = 0; i <= out.length && !placed; i++) {
      const before = i > 0 ? groupOf[out[i - 1]] : null;
      const after = i < out.length ? groupOf[out[i]] : null;
      if (before !== groupOf[x] && after !== groupOf[x]) { out.splice(i, 0, x); placed = true; }
    }
    if (!placed) out.push(x);
  }
  return out;
}

// The score a key press sets, or null. Cmd/Ctrl/Alt+digit are browser
// shortcuts (switch tab), not ratings.
function scoreKey(e) {
  if (e.metaKey || e.ctrlKey || e.altKey) return null;
  return e.key >= '1' && e.key <= '5' && e.key.length === 1 ? Number(e.key) : null;
}

// Whether more items have been rated than the last export held. A private
// window keeps localStorage until it closes and then drops it silently, so
// this, not a storage error, is what guards against losing a session.
function hasUnexported(state, exportedCount) {
  const rated = Object.values(state.ratings).filter(r => Object.keys(r.scores).length).length;
  return rated > exportedCount;
}

if (typeof module !== 'undefined') module.exports = {orderFor, scoreKey, hasUnexported};

// ── browser ──────────────────────────────────────────────────────────────────

const SCALES = [
  ['macedonian', () => 'Правилен македонски'],
  ['coherent', () => 'Кохерентност (има смисла)'],
  ['poetic', () => 'Поетски квалитет'],
  ['content', () => 'Ја задржува содржината на изворот'],
  ['author', a => 'Звучи како ' + a],
  ['usable', () => 'Употребливо како што е'],
];

function el(tag, attrs, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === 'class') node.className = v;
    else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const c of children) node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return node;
}

function startPage(DATA) {
  const app = document.getElementById('app');
  const items = DATA.items;
  let rater = null, order = null, pos = 0, state = null, focus = 0, shownAt = 0, shownPos = -1;
  let storageOk = true, exportedCount = 0;

  function storageKey() { return 'ratings:' + DATA.version + ':' + rater; }
  function save() {
    if (!storageOk) return;
    try { localStorage.setItem(storageKey(), JSON.stringify({...state, pos})); }
    catch (e) { storageOk = false; render(); }
  }
  function load() {
    try { const raw = localStorage.getItem(storageKey()); return raw ? JSON.parse(raw) : null; }
    catch (e) { storageOk = false; return null; }
  }

  function exportJson() {
    const out = {
      rater, page_version: DATA.version, started: state.started,
      finished: pos >= items.length ? new Date().toISOString() : null,
      ratings: Object.entries(state.ratings).map(([item_id, r]) => ({item_id, ...r})),
    };
    exportedCount = out.ratings.filter(r => Object.keys(r.scores).length).length;
    const blob = new Blob([JSON.stringify(out, null, 1)], {type: 'application/json'});
    const a = el('a', {href: URL.createObjectURL(blob), download: 'ratings_' + rater + '.json'});
    document.body.append(a); a.click(); a.remove();
  }

  function current() { return items[order[pos]]; }
  function rating() {
    const id = current().item_id;
    return state.ratings[id] || (state.ratings[id] = {scores: {}, comment: '', seconds: 0});
  }
  function complete() { return SCALES.every(([k]) => rating().scores[k]); }
  function leave() {
    if (pos < items.length) rating().seconds += Math.round((Date.now() - shownAt) / 1000);
    shownPos = -1;
  }
  function go(delta) {
    leave(); pos = Math.max(0, Math.min(items.length, pos + delta)); focus = 0; save(); render();
  }

  function poemBox(label, text) {
    return el('div', {class: 'box'}, el('div', {class: 'tag'}, label), el('div', {class: 'poem'}, text));
  }

  function render() {
    app.replaceChildren();
    const bar = el('div', {class: 'top'});
    if (!storageOk) bar.append(el('div', {class: 'warn'}, 'Напредокот нема да се зачува во овој прелистувач — извезете пред да ја затворите страницата.'));
    app.append(bar);
    if (rater === null) return renderStart();
    bar.append(el('button', {class: 'export', onclick: exportJson}, 'Извези оценки'));
    if (pos >= items.length) return renderEnd();
    if (shownPos !== pos) { shownAt = Date.now(); shownPos = pos; }
    const it = current(), r = rating();
    app.append(el('div', {class: 'head'}, `Ставка ${pos + 1} / ${items.length} · целен автор: ${it.target_author}`));
    const prog = el('div', {class: 'bar'}); prog.append(el('div', {style: `width:${100 * pos / items.length}%`}));
    app.append(prog);

    const panel = DATA.panels[it.target_author] || [];
    if (panel.length) {
      const det = el('details', {}, el('summary', {}, 'За ' + it.target_author + ' — три вистински песни (по желба)'));
      for (const p of panel) det.append(poemBox(p.title, p.text));
      app.append(det);
    }
    app.append(el('div', {class: 'two'},
      poemBox('ИЗВОР · ' + it.source.author + ', „' + it.source.title + '“', it.source.text),
      poemBox('ГЕНЕРИРАНА ПЕСНА', it.poem)));

    const scales = el('div', {class: 'scales'},
      el('div', {class: 'hint'}, '1 = воопшто не · 5 = целосно. Тастатура: 1–5 оценува, ↑/↓ менува скала, Enter = следно.'));
    SCALES.forEach(([key, label], i) => {
      const row = el('div', {class: 'scale' + (i === focus ? ' focus' : ''), onclick: () => { focus = i; render(); }},
        el('span', {class: 'q'}, label(it.target_author)));
      for (let v = 1; v <= 5; v++) {
        row.append(el('button', {class: 'dot' + (r.scores[key] === v ? ' on' : ''),
          onclick: e => { e.stopPropagation(); r.scores[key] = v; focus = Math.min(i + 1, SCALES.length - 1); save(); render(); }}, v));
      }
      scales.append(row);
    });
    app.append(scales);
    const comment = el('textarea', {placeholder: 'Коментар (по желба)', rows: 2});
    comment.value = r.comment;
    comment.addEventListener('input', () => { r.comment = comment.value; save(); });
    app.append(comment);
    if (it.reveal && complete()) app.append(el('div', {class: 'box reveal'}, it.reveal));
    const next = el('button', {class: 'primary', onclick: () => complete() && go(1)}, 'Следно →');
    if (!complete()) next.setAttribute('disabled', '');
    app.append(el('div', {class: 'nav'}, el('button', {onclick: () => go(-1)}, '← Назад'), next));
  }

  function renderStart() {
    const name = el('input', {placeholder: 'Вашето име', autofocus: ''});
    const start = () => {
      const v = name.value.trim(); if (!v) return;
      rater = v;
      order = orderFor(v, items.length, items.map(i => i.group));
      const saved = load();
      state = saved || {started: new Date().toISOString(), ratings: {}};
      pos = saved ? saved.pos || 0 : 0;
      save(); render();
    };
    name.addEventListener('keydown', e => { if (e.key === 'Enter') start(); });
    app.append(el('div', {class: 'start'},
      el('h1', {}, 'Оценување на песни'),
      el('p', {}, `Ќе видите ${items.length} песни, една по една, секоја покрај изворната песна од која е напишана. ` +
        'Оценете ја секоја на шест скали. ' + (items.some(i => i.reveal) ? 'Откако ќе ја оцените, ќе видите кој ја напишал. ' : 'Не се кажува кој систем ја напишал. ') + 'Напредокот се зачувува автоматски во овој прелистувач (не и во приватен прозорец); ' +
        'на крајот (или во секое време) притиснете „Извези оценки“ и испратете ја датотеката.'),
      name, el('button', {class: 'primary', onclick: start}, 'Започни')));
  }

  function renderEnd() {
    app.append(el('div', {class: 'start'}, el('h1', {}, 'Готово — благодарам!'),
      el('p', {}, 'Притиснете за да ја преземете датотеката со оценки.'),
      el('button', {class: 'primary', onclick: exportJson}, 'Извези оценки'),
      el('button', {onclick: () => go(-1)}, '← Назад')));
  }

  document.addEventListener('keydown', e => {
    if (rater === null || pos >= items.length) return;
    if (['TEXTAREA', 'INPUT'].includes(document.activeElement.tagName)) return;
    const r = rating(), score = scoreKey(e);
    if (score !== null) {
      r.scores[SCALES[focus][0]] = score;
      focus = Math.min(focus + 1, SCALES.length - 1); save(); render();
    } else if (e.key === 'ArrowDown') { focus = Math.min(focus + 1, SCALES.length - 1); render(); e.preventDefault(); }
    else if (e.key === 'ArrowUp') { focus = Math.max(focus - 1, 0); render(); e.preventDefault(); }
    else if (e.key === 'Enter' && complete()) go(1);
  });

  window.addEventListener('beforeunload', e => {
    if (state && hasUnexported(state, exportedCount)) { e.preventDefault(); e.returnValue = ''; }
  });

  render();
}

if (typeof document !== 'undefined' && typeof DATA !== 'undefined') startPage(DATA);

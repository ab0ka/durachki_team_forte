"""Подтверждение типов человеком: HTML + data/labels_types.csv + черновик cache/DATA_FINDINGS.md.

Вход: cache/types_review/vlm_raw.jsonl (tools/dataprep/vlm_types.py), data/labels.csv, data/manifest.csv.
Выход (всё локально, НЕ в git):
  cache/types_review/index.html — открыть в Chrome/Edge; правки автосохраняются в браузере;
      кнопки выгружают labels_types.csv / clean_procedures.csv / review_state.json (в «Загрузки»);
  data/labels_types.csv — VLM-предложения (source=vlm, confirmed=0) или подтверждённое человеком (--confirmed);
  cache/DATA_FINDINGS.md — черновик находок; дописывает и правит человек.
В stdout — только агрегаты.

usage: python -I -X utf8 tools/dataprep/types_review.py --root . [--confirmed ~/Downloads/labels_types.csv]
       [--clean-notes ~/Downloads/clean_procedures.csv] [--state ~/Downloads/review_state.json]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.util
import json
import math
import statistics
from pathlib import Path

# python -I не кладёт каталог скрипта в sys.path — грузим соседний модуль по пути
_spec = importlib.util.spec_from_file_location("vlm_types", Path(__file__).with_name("vlm_types.py"))
_vt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_vt)
CATALOG, fnum, norm_oop, norm_type = _vt.CATALOG, _vt.fnum, _vt.norm_oop, _vt.norm_type

COLS = ["video_id", "t_sec", "type_id", "confidence", "what_is_visible", "actors", "objects", "onset_or_peak",
        "event_time_sec", "new_type", "alt_type_id", "source", "confirmed", "notes"]

RULES = {
    "V01": ("outsider_in_cashier_area, n_persons, collector_present",
            "outsider_in_cashier_area ≥ 0.6 дольше 2 с при collector_present = 0",
            "onset: первый кадр постороннего в зоне кассира"),
    "V02": ("cashier_in_workplace, workplace_unattended_with_cash, cash_visible, safe_open",
            "cashier_in_workplace = 0 дольше 5 с при cash_visible / safe_open / включённом ноутбуке",
            "onset: выход кассира из зоны рабочего места"),
    "V03": ("safe_open, cashier_in_workplace, outsider_in_cashier_area",
            "safe_open ≥ 0.6 и (кассир вне места или посторонний в зоне) дольше 2 с",
            "onset: открытие сейфа или уход кассира при открытом сейфе"),
    "V04": ("door_open, n_persons, outsider_in_cashier_area",
            "door_open дольше 5 с или новый person-трек из зоны двери (не кассир и не инкассатор)",
            "onset: открытие двери / пересечение порога"),
    "V05": ("phone_in_hand, phone_to_ear (cell_phone у кисти по позе)",
            "phone_in_hand или phone_to_ear ≥ 0.6 дольше 1.5 с у трека кассира",
            "peak: телефон явно в руке или у уха"),
    "V06": ("eating_drinking (cup/bottle/food у рта по позе)",
            "eating_drinking ≥ 0.6 дольше 1 с",
            "peak: предмет у рта"),
    "V07": ("cash_bag_visible, collector_present, handover_motion",
            "cash_bag_visible и collector_present, передача сумки без видимой процедуры → проверка VLM",
            "onset: начало передачи сумки"),
    "V08": ("hands_hidden, cashier_facing_away, cash_visible, handover_motion",
            "hands_hidden или cashier_facing_away при cash_visible / handover_motion дольше 1.5 с",
            "onset: деньги уходят из обзора камеры"),
}


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    recs = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            recs[r["key"]] = r
    return list(recs.values())


def s(v) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        return "; ".join(s(x) for x in v)
    return str(v).strip()


def vlm_events(recs: list[dict]) -> list[dict]:
    """Финальное предложение на событие: проход с наибольшей уверенностью; все проходы — в passes."""
    by: dict[tuple, list[dict]] = {}
    for r in recs:
        if r["kind"] == "violation":
            by.setdefault((r["video_id"], r["t_sec"]), []).append(r)
    out = []
    for (vid, t), ps in by.items():
        ps.sort(key=lambda r: r["pass_"] != "win")
        best = max(ps, key=lambda r: fnum(r["parsed"].get("confidence"), 0))
        p = best["parsed"]
        out.append(dict(video_id=vid, t_sec=t, type_id=norm_type(p.get("type_id")),
                        alt_type_id=norm_type(p.get("alt_type_id")) if s(p.get("alt_type_id")) else "",
                        confidence=fnum(p.get("confidence")), onset_or_peak=norm_oop(p.get("onset_or_peak")),
                        event_time_sec=fnum(p.get("event_time_sec")), new_type=s(p.get("new_type")),
                        what_is_visible=s(p.get("what_is_visible")), actors=s(p.get("actors")),
                        objects=s(p.get("objects")), best_pass=best["pass_"], overview=best.get("overview", ""),
                        passes=[dict(pass_=r["pass_"], sheet=r["sheet"], type_id=norm_type(r["parsed"].get("type_id")),
                                     confidence=fnum(r["parsed"].get("confidence")), parse_ok=r["parse_ok"],
                                     raw=r["raw"]) for r in ps]))
    out.sort(key=lambda e: (int(e["video_id"].split("-")[1]), e["t_sec"]))
    return out


def read_confirmed(p: Path | None) -> dict[tuple, dict]:
    if not p or not p.exists():
        return {}
    out = {}
    for d in csv.DictReader(open(p, encoding="utf-8-sig")):
        if str(d.get("confirmed", "")).strip() in ("1", "true", "True"):
            out[(d["video_id"], float(d["t_sec"]))] = d
    return out


# ----------------------------------------------------------------------------- HTML
CSS = """
:root{--bg:#f6f7f9;--card:#fff;--fg:#1d2330;--mut:#5d6677;--line:#dfe3ea;--acc:#2457d6;--ok:#1d8a4e;--warn:#b26a00;
--bad:#c23b3b;--chip:#eef2fb}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#14171d;--card:#1c2029;--fg:#e6e9ef;
--mut:#9aa3b2;--line:#2c3240;--acc:#7aa2ff;--ok:#4cc38a;--warn:#e0a443;--bad:#ff7b7b;--chip:#242b3a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,Segoe UI,Arial}
header{position:sticky;top:0;z-index:5;background:var(--card);border-bottom:1px solid var(--line);padding:10px 16px;
display:flex;gap:12px;align-items:center;flex-wrap:wrap}
header h1{font-size:17px;margin:0 12px 0 0}button{font:inherit;padding:6px 12px;border-radius:6px;border:1px solid var(--line);
background:var(--chip);color:var(--fg);cursor:pointer}button.pri{background:var(--acc);color:#fff;border-color:var(--acc)}
main{max-width:1500px;margin:0 auto;padding:12px 16px 80px}.note{color:var(--mut)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:14px 0}
.card.done{border-left:5px solid var(--ok)}.row{display:grid;grid-template-columns:minmax(260px,330px) 1fr minmax(300px,420px);gap:14px}
@media (max-width:1100px){.row{grid-template-columns:1fr}}
h2{font-size:16px;margin:0 0 8px}h3{margin:28px 0 4px}.chip{display:inline-block;background:var(--chip);border-radius:999px;
padding:2px 9px;margin:0 6px 4px 0;font-size:12.5px}.lo{color:var(--warn);font-weight:600}
video{width:100%;max-height:520px;background:#000;border-radius:6px}.imgs{display:flex;flex-direction:column;gap:8px}
.imgs img{width:100%;border-radius:6px;border:1px solid var(--line);cursor:zoom-in}
label{display:block;font-size:12.5px;color:var(--mut);margin-top:8px}select,input,textarea{width:100%;font:inherit;
padding:6px;border-radius:6px;border:1px solid var(--line);background:var(--bg);color:var(--fg)}textarea{min-height:52px}
details{margin-top:8px}pre{white-space:pre-wrap;font-size:12px;background:var(--bg);padding:8px;border-radius:6px}
table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left;vertical-align:top}
.vlm p{margin:4px 0}.btns button{margin:4px 4px 0 0;padding:3px 8px}
"""

JS = r"""
const D = JSON.parse(document.getElementById('data').textContent);
const KEY = 'kassa_types_review_v1';
let S = {};
try { S = JSON.parse(localStorage.getItem(KEY) || '{}') || {}; } catch (e) { S = {}; }
function save() { try { localStorage.setItem(KEY, JSON.stringify(S)); } catch (e) {} progress(); }
function el(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === 'text') e.textContent = v; else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined) e.setAttribute(k, v);
  }
  for (const k of kids) if (k) e.append(k);
  return e;
}
const fmt = v => (v === null || v === undefined || Number.isNaN(v)) ? '—' : (+v).toFixed(2);
function field(st, k, lab, type, opts, dflt) {
  if (st[k] === undefined) st[k] = dflt ?? '';
  let inp;
  if (type === 'select') {
    inp = el('select', {});
    for (const [v, t] of opts) inp.append(el('option', {value: v, text: t}));
    inp.value = st[k];
  } else if (type === 'textarea') { inp = el('textarea', {}); inp.value = st[k]; }
  else { inp = el('input', {type}); if (type === 'checkbox') inp.checked = !!st[k]; else inp.value = st[k]; }
  inp.addEventListener('input', () => { st[k] = type === 'checkbox' ? inp.checked : inp.value; save(); });
  inp.addEventListener('change', () => {
    st[k] = type === 'checkbox' ? inp.checked : inp.value; save();
    const c = inp.closest('.card');  // точечно, без перерисовки (не сбрасываем видео)
    if (c && ['confirmed', 'verified', 'decision'].includes(k)) c.classList.toggle('done', !!st[k]);
  });
  if (type === 'checkbox') return el('label', {}, inp, ' ' + lab);
  return el('label', {text: lab}, inp);
}
function player(file, t) {
  const v = el('video', {controls: '', preload: 'none', src: '../../data/raw/' + file + '#t=' + Math.max(0, t - 5)});
  const go = d => () => { v.currentTime = Math.max(0, t + d); v.play(); };
  return el('div', {}, v, el('div', {class: 'btns'},
    el('button', {text: '▶ t−10', onclick: go(-10)}), el('button', {text: '▶ t−5', onclick: go(-5)}),
    el('button', {text: '▶ t', onclick: go(0)}), el('button', {text: '▶ t+5', onclick: go(5)})),
    el('div', {class: 'note', text: 'Если видео не играет — откройте data/raw/' + file + ' в плеере.'}));
}
const img = src => src ? el('a', {href: src, target: '_blank'}, el('img', {src, loading: 'lazy'})) : null;
const TYPES = [['', '— выберите —'], ...Object.entries(D.catalog).map(([k, v]) => [k, k + ' — ' + v]),
  ['NEW', 'NEW — новый тип (опишите)'], ['NONE', 'нарушения в этом месте не видно'], ['UNSURE', 'неясно']];
const CONF = [['0.9', 'уверен (0.9)'], ['0.6', 'скорее да (0.6)'], ['0.3', 'сомневаюсь (0.3)']];
const OOP = [['onset', 'onset — в t действие начинается'], ['peak', 'peak — в t пик действия'],
  ['before', 't раньше действия'], ['after', 't позже действия'], ['unclear', 'неясно']];
function vkey(e) { return e.video_id + '@' + e.t_sec; }
function evState(e) {
  const k = vkey(e);
  if (!S[k]) S[k] = {type_id: ['NEW', 'UNK'].includes(e.type_id) ? (e.type_id === 'NEW' ? 'NEW' : '') : e.type_id,
    confidence: '0.6', onset_or_peak: ['other'].includes(e.onset_or_peak) ? 'unclear' : e.onset_or_peak,
    event_time_sec: Number.isNaN(e.event_time_sec) || e.event_time_sec === null ? '' : String(e.event_time_sec),
    new_type: e.new_type, what_is_visible: e.what_is_visible, actors: e.actors, objects: e.objects, notes: '', confirmed: false};
  return S[k];
}
function vlmBlock(e) {
  const c = e.confidence, lo = !(c >= D.doubt);
  return el('div', {class: 'vlm'},
    el('div', {}, el('span', {class: 'chip', text: 'VLM: ' + e.type_id + (e.alt_type_id ? ' (alt ' + e.alt_type_id + ')' : '')}),
      el('span', {class: 'chip ' + (lo ? 'lo' : ''), text: 'conf ' + fmt(c)}), el('span', {class: 'chip', text: e.onset_or_peak}),
      el('span', {class: 'chip', text: 'событие ≈ ' + fmt(e.event_time_sec) + ' с'}), el('span', {class: 'chip', text: 'проход: ' + e.best_pass})),
    e.new_type ? el('p', {}, el('b', {text: 'Новый тип: '}), e.new_type) : null,
    el('p', {}, el('b', {text: 'Видно: '}), e.what_is_visible || '—'),
    el('p', {}, el('b', {text: 'Кто: '}), e.actors || '—'),
    el('p', {}, el('b', {text: 'Предметы: '}), e.objects || '—'),
    el('details', {}, el('summary', {text: 'сырые ответы VLM'}),
      ...e.passes.map(p => el('pre', {text: '[' + p.pass_ + '] ' + p.type_id + ' conf ' + fmt(p.confidence) + '\n' + p.raw}))));
}
function evCard(e) {
  const st = evState(e);
  const file = D.files[e.video_id];
  return el('div', {class: 'card' + (st.confirmed ? ' done' : ''), id: vkey(e)},
    el('h2', {text: e.video_id + ' · t = ' + e.t_sec + ' с' + (D.conflict[e.video_id] ? ' · ⚠ тот же файл, что ' + D.conflict[e.video_id] : '')}),
    el('div', {class: 'row'},
      el('div', {}, player(file, e.t_sec), vlmBlock(e)),
      el('div', {class: 'imgs'}, ...e.passes.map(p => img(p.sheet)), img(e.overview)),
      el('div', {},
        field(st, 'type_id', 'Тип нарушения', 'select', TYPES),
        field(st, 'new_type', 'Новый тип (только для NEW): название и суть', 'text'),
        field(st, 'confidence', 'Уверенность', 'select', CONF),
        field(st, 'onset_or_peak', 'Что происходит в размеченную секунду', 'select', OOP),
        field(st, 'event_time_sec', 'Секунда, где действие видно лучше всего', 'number'),
        field(st, 'what_is_visible', 'Что видно (факты, без домыслов)', 'textarea'),
        field(st, 'actors', 'Кто (роли, количество)', 'textarea'),
        field(st, 'objects', 'Предметы', 'textarea'),
        field(st, 'notes', 'Заметки', 'textarea'),
        field(st, 'confirmed', 'Подтверждаю', 'checkbox'))));
}
function pairCard(p) {
  const k = 'pair:' + p.video_ids.join('+');
  if (!S[k]) S[k] = {decision: '', notes: ''};
  const r = p.parsed || {};
  return el('div', {class: 'card' + (S[k].decision ? ' done' : '')},
    el('h2', {text: 'Дубликат ' + p.video_ids.join(' ≡ ') + ' · таймкоды ' + p.t.join(' и ') + ' с'}),
    el('div', {class: 'row'},
      el('div', {}, player(D.files[p.video_ids[0]], p.t[0]), el('div', {class: 'vlm'},
        el('span', {class: 'chip', text: 'VLM: одно событие = ' + r.same_event}),
        el('span', {class: 'chip', text: 'conf ' + fmt(+r.confidence)}),
        el('p', {}, el('b', {text: 't=' + p.t[0] + ': '}), (r.type_t1 || '') + ' — ' + (r.event_t1 || '')),
        el('p', {}, el('b', {text: 't=' + p.t[1] + ': '}), (r.type_t2 || '') + ' — ' + (r.event_t2 || '')),
        el('details', {}, el('summary', {text: 'сырой ответ'}), el('pre', {text: p.raw})))),
      el('div', {class: 'imgs'}, img(p.sheet)),
      el('div', {},
        field(S[k], 'decision', 'Решение', 'select', [['', '— выберите —'], ['two_events', 'два разных события'],
          ['same_event', 'одно событие, верный таймкод — первый'], ['same_event_second', 'одно событие, верный таймкод — второй'],
          ['unclear', 'неясно']]),
        field(S[k], 'notes', 'Заметки', 'textarea'))));
}
function cleanCard(c) {
  const k = 'clean:' + c.video_id, r = c.parsed || {};
  const steps = Array.isArray(r.procedure_steps) ? r.procedure_steps : [String(r.procedure_steps || '')];
  const fps = Array.isArray(r.fp_risks) ? r.fp_risks : [String(r.fp_risks || '')];
  if (!S[k]) S[k] = {procedure: steps.join('\n'), fp_risks: fps.join('\n'), notes: '', verified: false};
  return el('div', {class: 'card' + (S[k].verified ? ' done' : '')},
    el('h2', {text: c.video_id + ' · без нарушения · ' + D.dur[c.video_id] + ' с'}),
    el('div', {class: 'row'},
      el('div', {}, player(D.files[c.video_id], 5), el('div', {class: 'vlm'},
        el('p', {}, el('b', {text: 'Кто: '}), String(r.actors || '—')),
        el('p', {}, el('b', {text: 'Инкассаторы: '}), String(r.collectors_present)),
        el('p', {}, el('b', {text: 'Сейф: '}), String(r.safe_used || '—')),
        el('p', {}, el('b', {text: 'Деньги: '}), String(r.cash_handling || '—')),
        el('p', {}, el('b', {text: 'Дверь: '}), String(r.door_events || '—')),
        el('p', {}, el('b', {text: 'VLM подозревает нарушение: '}), String(r.suspected_violation || 'нет')),
        el('details', {}, el('summary', {text: 'сырой ответ'}), el('pre', {text: c.raw})))),
      el('div', {class: 'imgs'}, img(c.overview)),
      el('div', {},
        field(S[k], 'procedure', 'Корректная процедура по шагам (поправьте)', 'textarea'),
        field(S[k], 'fp_risks', 'Что может ложно сработать и почему это норма', 'textarea'),
        field(S[k], 'notes', 'Заметки', 'textarea'),
        field(S[k], 'verified', 'Проверено', 'checkbox'))));
}
function progress() {
  const n = D.events.filter(e => S[vkey(e)] && S[vkey(e)].confirmed).length;
  document.getElementById('prog').textContent = 'подтверждено ' + n + ' / ' + D.events.length;
}
function render() {
  const m = document.getElementById('main'); m.replaceChildren();
  m.append(el('p', {class: 'note', text: 'Типы предложены локальной VLM (Qwen3-VL-4B) и НЕ подтверждены. ' +
    'Проверьте каждое событие по видео и кадрам, поправьте поля и отметьте «Подтверждаю». Правки сохраняются в этом ' +
    'браузере. В конце нажмите «labels_types.csv» и запустите: python -I -X utf8 tools/dataprep/types_review.py ' +
    '--root . --confirmed <путь к скачанному файлу>.'}));
  const leg = el('table', {}); for (const [k, v] of Object.entries(D.catalog)) leg.append(el('tr', {}, el('td', {text: k}), el('td', {text: v})));
  m.append(el('details', {}, el('summary', {text: 'Список типов V01–V08'}), leg));
  m.append(el('h3', {text: 'Нарушения (' + D.events.length + ')'}));
  D.events.forEach(e => m.append(evCard(e)));
  if (D.pairs.length) { m.append(el('h3', {text: 'Дубликаты с разными таймкодами'})); D.pairs.forEach(p => m.append(pairCard(p))); }
  if (D.clean.length) { m.append(el('h3', {text: 'Чистые видео: корректная процедура'})); D.clean.forEach(c => m.append(cleanCard(c))); }
  progress(); save();
}
function csvRow(a) { return a.map(v => { v = v === undefined || v === null ? '' : String(v); return /[",\n;]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; }).join(','); }
function download(name, text, type) {
  const b = new Blob(['﻿' + text], {type}); const a = el('a', {href: URL.createObjectURL(b), download: name});
  document.body.append(a); a.click(); a.remove();
}
function exportTypes() {
  const lines = [csvRow(D.cols)];
  for (const e of D.events) {
    const st = evState(e);
    const rec = {video_id: e.video_id, t_sec: e.t_sec, type_id: st.type_id, confidence: st.confidence,
      what_is_visible: st.what_is_visible, actors: st.actors, objects: st.objects, onset_or_peak: st.onset_or_peak,
      event_time_sec: st.event_time_sec, new_type: st.type_id === 'NEW' ? st.new_type : '', alt_type_id: e.alt_type_id,
      source: st.confirmed ? 'human' : 'vlm', confirmed: st.confirmed ? 1 : 0, notes: st.notes};
    lines.push(csvRow(D.cols.map(c => rec[c])));
  }
  download('labels_types.csv', lines.join('\n') + '\n', 'text/csv');
}
function exportClean() {
  const lines = [csvRow(['video_id', 'verified', 'procedure', 'fp_risks', 'notes'])];
  for (const c of D.clean) { const st = S['clean:' + c.video_id] || {}; lines.push(csvRow([c.video_id, st.verified ? 1 : 0, st.procedure, st.fp_risks, st.notes])); }
  download('clean_procedures.csv', lines.join('\n') + '\n', 'text/csv');
}
document.getElementById('bt').onclick = exportTypes;
document.getElementById('bc').onclick = exportClean;
document.getElementById('bs').onclick = () => download('review_state.json', JSON.stringify(S, null, 1), 'application/json');
document.getElementById('br').onclick = () => { if (confirm('Сбросить все правки в этом браузере?')) { S = {}; save(); render(); } };
render();
"""


def build_html(events, pairs, clean, man, conflict, doubt) -> str:
    data = dict(catalog=CATALOG, cols=COLS, doubt=doubt, events=events, pairs=pairs, clean=clean,
                files={k: v["file"] for k, v in man.items()},
                dur={k: round(float(v["duration_s"]), 1) for k, v in man.items()}, conflict=conflict)
    blob = json.dumps(data, ensure_ascii=False, default=lambda o: None, allow_nan=False).replace("</", "<\\/")
    return (f"<!doctype html><html lang='ru'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'><title>Подтверждение типов</title>"
            f"<style>{CSS}</style></head><body><header><h1>Подтверждение типов нарушений</h1>"
            f"<span id='prog' class='chip'></span><button class='pri' id='bt'>labels_types.csv</button>"
            f"<button id='bc'>clean_procedures.csv</button><button id='bs'>review_state.json</button>"
            f"<button id='br'>сбросить</button><span class='note'>локальный файл — не публиковать</span></header>"
            f"<main id='main'></main><script type='application/json' id='data'>{blob}</script>"
            f"<script>{JS}</script></body></html>")


def nan_to_none(o):
    if isinstance(o, float) and math.isnan(o):
        return None
    if isinstance(o, dict):
        return {k: nan_to_none(v) for k, v in o.items()}
    if isinstance(o, list):
        return [nan_to_none(v) for v in o]
    return o


# ----------------------------------------------------------------------------- DATA_FINDINGS (черновик)
def findings(events, conf_rows, pairs, clean, man, clean_notes, state, doubt) -> str:
    L = []
    w = L.append
    src = "подтверждено человеком" if conf_rows else "предложения VLM, НЕ подтверждены"
    w("# DATA_FINDINGS — ЧЕРНОВИК (локально, не в git)\n")
    w(f"> Сгенерировано `tools/dataprep/types_review.py` {dt.datetime.now():%Y-%m-%d %H:%M}. Источник типов: **{src}** "
      f"({len(conf_rows)}/{len(events)} подтверждено). Текст ниже правит и дописывает человек.\n")
    durs = {k: float(v["duration_s"]) for k, v in man.items()}
    pos = [durs[k] for k, v in man.items() if v["label"] == "violation"]
    neg = [durs[k] for k, v in man.items() if v["label"] == "no_violation"]
    w("## 1. Датасет\n")
    w(f"- {len(man)} файлов, {len({v['sha256'] for v in man.values()})} уникальных по sha256; "
      f"violation {len(pos)} / no_violation {len(neg)}.")
    w(f"- Длительность: violation медиана {statistics.median(pos):.1f} с, no_violation {statistics.median(neg):.1f} с "
      f"(длительность — артефакт датасета, как признак не использовать).")
    w("- 464×848 портрет, rotation 0, H.264, VFR ~30 fps (время кадра — по PTS).\n")

    final = []
    for e in events:
        c = conf_rows.get((e["video_id"], e["t_sec"]))
        if c:
            final.append(dict(e, type_id=c["type_id"].strip().upper(), confidence=fnum(c.get("confidence")),
                              onset_or_peak=c.get("onset_or_peak", ""), event_time_sec=fnum(c.get("event_time_sec")),
                              new_type=c.get("new_type", ""), what_is_visible=c.get("what_is_visible", ""),
                              actors=c.get("actors", ""), objects=c.get("objects", ""), src="human"))
        else:
            final.append(dict(e, src="vlm"))
    types = [f["type_id"] for f in final]
    w("## 2. Частоты типов\n")
    w("| тип | название | событий | доля | ср. уверенность | onset / peak |")
    w("|---|---|---|---|---|---|")
    for t in sorted(set(types)):
        fs = [f for f in final if f["type_id"] == t]
        cs = [f["confidence"] for f in fs if not math.isnan(f["confidence"])]
        oo = [norm_oop(f["onset_or_peak"]) for f in fs]
        w(f"| {t} | {CATALOG.get(t, 'новый / неясно')} | {len(fs)} | {len(fs) / len(final):.0%} | "
          f"{statistics.mean(cs) if cs else float('nan'):.2f} | {oo.count('onset')} / {oo.count('peak')} |")
    w("")
    w("## 3. Признаки по типам (что видно в размеченную секунду)\n")
    for t in sorted(set(types)):
        w(f"### {t} — {CATALOG.get(t, 'новый / неясно')}\n")
        for f in [f for f in final if f["type_id"] == t]:
            extra = f" — новый тип: {f['new_type']}" if f.get("new_type") else ""
            w(f"- **{f['video_id']} @ {f['t_sec']:g} с** [{f['src']}, conf {f['confidence']:.2f}, "
              f"{norm_oop(f['onset_or_peak'])}, событие ≈ {f['event_time_sec']:.0f} с]{extra}: {f['what_is_visible']} "
              f"_Кто:_ {f['actors']}. _Предметы:_ {f['objects']}.")
        w("")
    oo = [norm_oop(f["onset_or_peak"]) for f in final]
    offs = [f["event_time_sec"] - f["t_sec"] for f in final if not math.isnan(f["event_time_sec"])]
    w("## 4. Таймкод: onset vs peak\n")
    w("- В размеченную секунду: " + ", ".join(f"{k} {oo.count(k)}" for k in sorted(set(oo))) + ".")
    if offs:
        w(f"- (секунда лучшей видимости − метка): медиана {statistics.median(offs):+.1f} с, "
          f"|Δ| ≤ 1 с у {sum(abs(o) <= 1 for o in offs)}/{len(offs)}, ≤ 3 с у {sum(abs(o) <= 3 for o in offs)}/{len(offs)}.")
    major = "onset" if oo.count("onset") >= oo.count("peak") else "peak"
    w(f"- Черновая рекомендация: разметчики чаще ставили **{major}** → отдавать `t_peak` контракта как "
      f"{'момент начала действия (первый кадр, где признак устойчиво виден)' if major == 'onset' else 'пик действия'}; "
      "кадры-доказательства: метка, +1 с и пик. Проверить на подтверждённых данных.\n")
    w("## 5. Чистые видео: корректная процедура\n")
    cn = {}
    if clean_notes and clean_notes.exists():
        cn = {d["video_id"]: d for d in csv.DictReader(open(clean_notes, encoding="utf-8-sig"))}
    sus = 0
    for c in clean:
        r = c.get("parsed") or {}
        h = cn.get(c["video_id"])
        w(f"### {c['video_id']} ({durs[c['video_id']]:.0f} с){' — проверено человеком' if h and h.get('verified') == '1' else ''}\n")
        steps = (h["procedure"].splitlines() if h else s(r.get("procedure_steps")).split("; "))
        for st in steps:
            if st.strip():
                w(f"- {st.strip()}")
        w(f"- Инкассаторы: {s(r.get('collectors_present'))}; сейф: {s(r.get('safe_used'))}; "
          f"деньги: {s(r.get('cash_handling'))}; дверь: {s(r.get('door_events'))}.")
        w(f"- Риски ложного срабатывания: {h['fp_risks'] if h else s(r.get('fp_risks'))}")
        if s(r.get("suspected_violation")):
            sus += 1
            w(f"- ⚠ VLM заподозрила нарушение: {s(r.get('suspected_violation'))}")
        w("")
    w(f"VLM заподозрила нарушение в {sus} из {len(clean)} чистых видео — оценка склонности к ложным срабатываниям.\n")
    w("## 6. Сложные случаи\n")
    for f in final:
        flags = []
        if not (f["confidence"] >= doubt):
            flags.append(f"низкая уверенность {f['confidence']:.2f}")
        if f["type_id"] in ("NEW", "UNK", "UNSURE", "NONE"):
            flags.append(f"тип {f['type_id']}")
        if f.get("best_pass") == "wide":
            flags.append("понадобилось окно ±10 с")
        if flags:
            w(f"- {f['video_id']} @ {f['t_sec']:g} с: " + ", ".join(flags))
    for p in pairs:
        r = p.get("parsed") or {}
        dec = (state or {}).get("pair:" + "+".join(p["video_ids"]), {}).get("decision", "")
        w(f"- Дубликат {' ≡ '.join(p['video_ids'])} (таймкоды {', '.join(f'{x:g}' for x in p['t'])} с): VLM — "
          f"same_event={r.get('same_event')}, типы {norm_type(r.get('type_t1'))}/{norm_type(r.get('type_t2'))}; "
          f"решение человека: {dec or 'нет'}. {s(r.get('event_t1'))} | {s(r.get('event_t2'))}")
    w("")
    w("## 7. Гипотезы V01–V08\n")
    seen = sorted({t for t in types if t in CATALOG})
    w(f"- Встретились: {', '.join(seen) or '—'}.")
    w(f"- Не встретились: {', '.join(k for k in CATALOG if k not in seen) or '—'}.")
    news = "; ".join(f"{f['video_id']}: {f['new_type']}" for f in final if f["type_id"] == "NEW")
    w(f"- Новые типы: {news or '—'}.\n")
    w("## 8. Рекомендации для правил (сигналы контракта, стартовые пороги — подстроить на данных)\n")
    w("| тип | сигналы | условие (старт) | таймкод |")
    w("|---|---|---|---|")
    for k in sorted(RULES, key=lambda k: -types.count(k)):
        a, b, c = RULES[k]
        w(f"| {k} ({types.count(k)}) | {a} | {b} | {c} |")
    w("")
    w("## 9. Дописать человеку\n")
    w("- [ ] Где в кадре кассир, сейф, дверь (сколько раскладок мебели и ракурсов, зоны для `Zone`).")
    w("- [ ] Свет: есть ли смены освещения или экспозиции и в каких роликах.")
    w("- [ ] Подтвердить или исправить типы с низкой уверенностью, NEW и UNK; решение по дубликату с двумя таймкодами.")
    w("- [ ] Что из V01–V08 подтвердилось, какие новые типы оставить в `norms/catalog.yaml` (V99 или новые id).")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--confirmed", type=Path, default=None, help="labels_types.csv, выгруженный из HTML")
    ap.add_argument("--clean-notes", type=Path, default=None)
    ap.add_argument("--state", type=Path, default=None, help="review_state.json из HTML (решения по дубликатам)")
    ap.add_argument("--doubt", type=float, default=0.6)
    a = ap.parse_args()
    root = Path(a.root)
    rdir = root / "cache/types_review"
    recs = load_jsonl(rdir / "vlm_raw.jsonl")
    man = {r["video_id"]: r for r in csv.DictReader(open(root / "data/manifest.csv", encoding="utf-8"))}
    conflict = {k: v["dup_of"] for k, v in man.items() if v.get("needs_review") == "1"}
    events = vlm_events(recs)
    pairs = [r for r in recs if r["kind"] == "pair"]
    clean = sorted([r for r in recs if r["kind"] == "clean"], key=lambda r: int(r["video_id"].split("-")[1]))
    conf_rows = read_confirmed(a.confirmed)
    state = json.loads(a.state.read_text(encoding="utf-8-sig")) if a.state and a.state.exists() else None

    (rdir / "index.html").write_text(build_html(nan_to_none(events), pairs, clean, man, conflict, a.doubt),
                                     encoding="utf-8")
    out = root / "data/labels_types.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=COLS, lineterminator="\n", extrasaction="ignore")
        wr.writeheader()
        for e in events:
            c = conf_rows.get((e["video_id"], e["t_sec"]))
            if c:
                wr.writerow({**{k: c.get(k, "") for k in COLS}, "source": "human", "confirmed": 1})
            else:
                wr.writerow({**{k: e.get(k, "") for k in COLS}, "t_sec": f"{e['t_sec']:g}",
                             "confidence": "" if math.isnan(e["confidence"]) else f"{e['confidence']:.2f}",
                             "event_time_sec": "" if math.isnan(e["event_time_sec"]) else f"{e['event_time_sec']:g}",
                             "source": "vlm", "confirmed": 0, "notes": ""})
    (root / "cache/DATA_FINDINGS.md").write_text(
        findings(events, conf_rows, pairs, clean, man, a.clean_notes, state, a.doubt), encoding="utf-8")
    print(f"html: {rdir / 'index.html'} | events={len(events)} pairs={len(pairs)} clean={len(clean)}")
    print(f"labels_types.csv: {len(events)} rows, confirmed by human: {len(conf_rows)}")
    print(f"DATA_FINDINGS draft: {root / 'cache/DATA_FINDINGS.md'}")


if __name__ == "__main__":
    main()

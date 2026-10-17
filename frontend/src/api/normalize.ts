// Защитный разбор ответа GET /…/jobs/{id}. Бэкенд собирают 18 агентов —
// UI не должен падать от лишних/пропущенных полей, строковых чисел и т.п.
import type {
  EvidenceFrame,
  JobMeta,
  JobResult,
  JobStatus,
  RuleRef,
  TimelinePoint,
  Verdict,
  Violation,
} from "./types";

type Obj = Record<string, unknown>;

const isObj = (v: unknown): v is Obj => typeof v === "object" && v !== null && !Array.isArray(v);

export function num(v: unknown, fallback = 0): number {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  if (typeof v === "string" && v.trim() !== "") {
    const n = Number(v.replace(",", "."));
    if (Number.isFinite(n)) return n;
  }
  return fallback;
}

function optNum(v: unknown): number | undefined {
  const n = num(v, Number.NaN);
  return Number.isFinite(n) ? n : undefined;
}

function str(v: unknown, fallback = ""): string {
  if (typeof v === "string") return v;
  if (typeof v === "number" || typeof v === "boolean") return String(v);
  return fallback;
}

function strList(v: unknown): string[] {
  if (Array.isArray(v)) return v.map((x) => str(x)).filter((s) => s.trim() !== "");
  if (typeof v === "string" && v.trim() !== "") return [v];
  return [];
}

export function normalizeStatus(v: unknown): JobStatus {
  const s = str(v).toLowerCase().trim();
  if (["done", "completed", "complete", "finished", "success", "succeeded", "ok"].includes(s)) return "done";
  if (["failed", "error", "failure", "cancelled", "canceled", "timeout"].includes(s)) return "failed";
  if (["queued", "pending", "waiting", "accepted"].includes(s)) return "queued";
  return "processing";
}

export function normalizeVerdict(v: unknown, violations: Violation[] = []): Verdict | null {
  if (typeof v === "boolean") return v ? "violation" : "no_violation";
  const s = str(v).toLowerCase().trim().replace(/[\s-]+/g, "_");
  if (["violation", "violations", "violated", "true", "1", "нарушение"].includes(s)) return "violation";
  if (["no_violation", "noviolation", "clean", "ok", "none", "false", "0", "нет_нарушения"].includes(s))
    return "no_violation";
  if (s === "") return null;
  // Неизвестная строка: решаем по наличию нарушений.
  return violations.length > 0 ? "violation" : null;
}

/** Прогресс: 0..1 или 0..100 → 0..1. */
export function normalizeProgress(v: unknown): number | undefined {
  const n = optNum(v);
  if (n === undefined) return undefined;
  const p = n > 1 ? n / 100 : n;
  return Math.min(1, Math.max(0, p));
}

function normalizeFrame(v: unknown): EvidenceFrame | null {
  if (typeof v === "string") return v ? { timestamp_sec: 0, image_base64: v } : null;
  if (!isObj(v)) return null;
  const img = str(v.image_base64 ?? v.image ?? v.jpeg_base64 ?? v.data ?? v.url);
  if (!img) return null;
  return { timestamp_sec: num(v.timestamp_sec ?? v.t ?? v.timestamp, 0), image_base64: img };
}

function normalizeRule(v: unknown): RuleRef | null {
  if (!isObj(v)) return null;
  const document = str(v.document ?? v.doc ?? v.title);
  const url = str(v.url ?? v.link ?? v.href);
  if (!document && !url) return null;
  const clause = v.clause === null || v.clause === undefined ? null : str(v.clause);
  return {
    document: document || url,
    clause,
    url,
    quote: str(v.quote ?? v.text),
    verified: v.verified === true || v.verified === "true",
  };
}

export function normalizeViolation(v: unknown): Violation | null {
  if (!isObj(v)) return null;
  const timecode = num(v.timecode_sec ?? v.t_peak ?? v.timecode ?? v.t, 0);
  const frames = Array.isArray(v.evidence_frames)
    ? v.evidence_frames.map(normalizeFrame).filter((f): f is EvidenceFrame => f !== null)
    : [];
  const rulesRaw: unknown[] = Array.isArray(v.rules) && v.rules.length ? v.rules : isObj(v.rule) ? [v.rule] : [];
  const rules = rulesRaw.map(normalizeRule).filter((r): r is RuleRef => r !== null);
  const sources = strList(v.sources ?? v.source);
  const out: Violation = {
    type: str(v.type, "") || str(v.type_name, "") || str(v.type_id, "Возможное нарушение"),
    timecode_sec: Math.max(0, timecode),
    description: str(v.description),
    evidence_frames: frames,
    type_id: str(v.type_id) || undefined,
    confidence: optNum(v.confidence),
    t_start: optNum(v.t_start),
    t_end: optNum(v.t_end),
    rules,
    facts: strList(v.facts),
    sources,
    severity: str(v.severity) || undefined,
  };
  if (out.confidence !== undefined) out.confidence = Math.min(1, Math.max(0, out.confidence));
  return out;
}

export function normalizeTimeline(v: unknown): TimelinePoint[] {
  if (!Array.isArray(v)) return [];
  const pts: TimelinePoint[] = [];
  for (const p of v) {
    let t: number | undefined;
    let r: number | undefined;
    if (Array.isArray(p) && p.length >= 2) {
      t = optNum(p[0]);
      r = optNum(p[1]);
    } else if (isObj(p)) {
      t = optNum(p.t ?? p.time ?? p.timestamp_sec);
      r = optNum(p.risk ?? p.score ?? p.value);
    }
    if (t === undefined || r === undefined || t < 0) continue;
    pts.push([t, Math.min(1, Math.max(0, r))]);
  }
  pts.sort((a, b) => a[0] - b[0]);
  return pts;
}

/** Главная функция: сырой JSON → JobResult. Бросает Error, если это не похоже на ответ API. */
export function normalizeJob(raw: unknown, fallbackJobId = ""): JobResult {
  if (!isObj(raw)) throw new Error("Некорректный ответ сервера: ожидался JSON-объект");
  const violations = Array.isArray(raw.violations)
    ? raw.violations.map(normalizeViolation).filter((v): v is Violation => v !== null)
    : [];
  violations.sort((a, b) => a.timecode_sec - b.timecode_sec);
  const status = normalizeStatus(raw.status);
  let verdict = normalizeVerdict(raw.verdict, violations);
  if (status === "done" && verdict === null) verdict = violations.length > 0 ? "violation" : "no_violation";
  const meta: JobMeta = isObj(raw.meta) ? (raw.meta as JobMeta) : {};
  const error = str(raw.error ?? raw.detail ?? meta.error ?? "") || undefined;
  let partialResult: JobResult | undefined;
  if (status === "failed" && isObj(raw.partial)) {
    try {
      partialResult = normalizeJob({ ...raw.partial, status: "done" }, fallbackJobId);
    } catch {
      partialResult = undefined;
    }
  }
  const qp = optNum(raw.queue_position);
  const el = optNum(raw.elapsed_s);
  return {
    job_id: str(raw.job_id ?? raw.id, fallbackJobId),
    queuePosition: qp,
    elapsedSec: el,
    partialResult,
    status,
    verdict,
    violations,
    timeline: normalizeTimeline(raw.timeline),
    meta,
    progress: status === "done" ? 1 : normalizeProgress(raw.progress),
    stage: str(raw.stage ?? meta.stage) || undefined,
    error,
  };
}

/** image_base64 → src для <img>: распознаём JPEG/PNG/SVG/WebP, data-URI и URL. */
export function imageSrc(b64: string): string {
  const s = b64.trim();
  if (s.startsWith("data:") || s.startsWith("blob:") || /^https?:\/\//.test(s)) return s;
  // JPEG base64 начинается с "/9j/" — это НЕ путь. Путь — если есть символы вне алфавита base64.
  if (s.startsWith("/") && !/^[A-Za-z0-9+/=\s]+$/.test(s)) return s;
  if (s.startsWith("iVBOR")) return `data:image/png;base64,${s}`;
  if (s.startsWith("PHN2") || s.startsWith("PD94")) return `data:image/svg+xml;base64,${s}`;
  if (s.startsWith("UklGR")) return `data:image/webp;base64,${s}`;
  return `data:image/jpeg;base64,${s}`;
}

/** Длительность видео: meta.duration → последняя точка timeline → последний t_end. */
export function inferDuration(job: JobResult): number {
  const m = job.meta;
  const fromMeta =
    optNum(m.duration) ??
    optNum(m.duration_sec) ??
    (isObj(m.video) ? optNum((m.video as Obj).duration) : undefined);
  if (fromMeta && fromMeta > 0) return fromMeta;
  let d = 0;
  for (const [t] of job.timeline) d = Math.max(d, t);
  for (const v of job.violations) d = Math.max(d, v.t_end ?? 0, v.timecode_sec + 1);
  return d;
}

/** Источники события: из самого нарушения или из meta.events (trace фьюжна). */
export function violationSources(v: Violation, job: JobResult): string[] {
  const own = v.sources ?? [];
  if (own.length) return own;
  const events = (job.meta as Obj).events;
  if (Array.isArray(events)) {
    const found = new Set<string>();
    for (const e of events) {
      if (!isObj(e)) continue;
      if (v.type_id && str(e.type_id) !== v.type_id) continue;
      const tp = num(e.t_peak ?? e.timecode_sec, Number.NaN);
      if (Number.isFinite(tp) && Math.abs(tp - v.timecode_sec) > 15) continue;
      strList(e.source ?? e.sources).forEach((s) => found.add(s));
    }
    return [...found];
  }
  return [];
}

/** Факты: явные facts → предложения описания. */
export function violationFacts(v: Violation, job: JobResult): string[] {
  if (v.facts && v.facts.length) return v.facts;
  const events = (job.meta as Obj).events;
  if (Array.isArray(events)) {
    const facts: string[] = [];
    for (const e of events) {
      if (!isObj(e) || (v.type_id && str(e.type_id) !== v.type_id)) continue;
      const tp = num(e.t_peak ?? e.timecode_sec, Number.NaN);
      if (Number.isFinite(tp) && Math.abs(tp - v.timecode_sec) > 15) continue;
      facts.push(...strList(e.facts));
    }
    if (facts.length) return [...new Set(facts)];
  }
  return v.description
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 3)
    .slice(0, 5);
}

/**
 * Анализ неполный (часть стадий пропущена по таймауту/ошибке)? Тогда «нарушений нет»
 * нельзя показывать зелёным. Флаги — договорённость с A02/A03 (см. README → «Для интегратора»).
 */
export function isPartial(job: JobResult): boolean {
  const m = job.meta as Obj;
  if (m.partial === true || m.incomplete === true || m.degraded === true || m.timed_out === true) return true;
  if (m.complete === false) return true;
  for (const k of ["skipped_stages", "stages_skipped", "failed_stages"]) {
    const v = m[k];
    if (Array.isArray(v) && v.length > 0) return true;
  }
  const warnings = Array.isArray(m.warnings) ? m.warnings.map((w) => str(w).toLowerCase()) : [];
  return warnings.some((w) => /partial|timeout|timed out|skipped|неполн|таймаут|пропущ|не заверш/.test(w));
}

/** Внешний JSON жюри: строго обязательные поля ТЗ (без timeline/meta/норм). */
export function toJuryJson(job: JobResult): Record<string, unknown> {
  return {
    job_id: job.job_id,
    status: job.status,
    verdict: job.verdict ?? null,
    violations: job.violations.map((v) => ({
      type: v.type,
      timecode_sec: v.timecode_sec,
      description: v.description,
      evidence_frames: v.evidence_frames.map((f) => ({ timestamp_sec: f.timestamp_sec, image_base64: f.image_base64 })),
    })),
  };
}

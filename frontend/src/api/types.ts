// Типы ответа API Kassa Sentinel.
// Обязательные поля — строго по ТЗ; остальные — доп. поля контракта v1
// (contracts/sentinel_contracts.py → contracts.serialize.result_to_api).

export type JobStatus = "queued" | "processing" | "done" | "failed";
export type Verdict = "violation" | "no_violation";

/** Кадр-доказательство. image_base64 — JPEG base64 (без префикса data:) или data-URI. */
export interface EvidenceFrame {
  timestamp_sec: number;
  image_base64: string;
}

/** Ссылка на норму РК — только из norms/catalog.yaml (A13). */
export interface RuleRef {
  document: string;
  clause?: string | null;
  url: string;
  quote?: string;
  verified?: boolean;
}

export interface Violation {
  // --- обязательные (ТЗ) ---
  type: string;
  timecode_sec: number;
  description: string;
  evidence_frames: EvidenceFrame[];
  // --- доп. поля контракта ---
  type_id?: string;
  confidence?: number;
  t_start?: number;
  t_end?: number;
  rules?: RuleRef[];
  // --- необязательные расширения (если бэкенд их отдаёт) ---
  facts?: string[];
  sources?: string[];
  severity?: string;
}

export type TimelinePoint = [number, number]; // (t сек, risk 0..1)

export interface JobMeta {
  timings?: Record<string, number>;
  models?: unknown;
  warnings?: string[];
  version?: string;
  duration?: number;
  file_name?: string;
  [key: string]: unknown;
}

export interface JobResult {
  job_id: string;
  status: JobStatus;
  verdict?: Verdict | null;
  violations: Violation[];
  timeline: TimelinePoint[];
  meta: JobMeta;
  /** 0..1 */
  progress?: number;
  stage?: string;
  error?: string;
}

export interface AskResponse {
  answer: string;
  sources?: string[];
}

export interface SessionInfo {
  token: string | null;
  /** Сервер ответил (любой HTTP-статус). */
  reachable: boolean;
  /** Есть действующая сессия (HttpOnly cookie) или API-ключ в памяти вкладки. */
  authenticated: boolean;
  features: { copilot?: boolean; [k: string]: unknown };
  version?: string;
}

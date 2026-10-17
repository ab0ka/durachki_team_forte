// HTTP-клиент к бэкенду (A02). Все пути — в ENDPOINTS, чтобы интегратор правил в одном месте.
//
// Авторизация UI: GET /ui/v1/session → токен → заголовок X-UI-Token для /ui/v1/*.
// Запасной путь: API-ключ из настроек → X-API-Key для /api/v1/*.
import { normalizeJob } from "./normalize";
import type { AskResponse, JobResult, SessionInfo } from "./types";

export const ENDPOINTS = {
  session: "/ui/v1/session",
  uiAnalyze: "/ui/v1/analyze",
  uiJob: (id: string) => `/ui/v1/jobs/${encodeURIComponent(id)}`,
  uiAsk: (id: string) => `/ui/v1/jobs/${encodeURIComponent(id)}/ask`,
  apiAnalyze: "/api/v1/analyze",
  apiJob: (id: string) => `/api/v1/jobs/${encodeURIComponent(id)}`,
  apiAsk: (id: string) => `/api/v1/jobs/${encodeURIComponent(id)}/ask`,
  video: (id: string) => `/api/v1/jobs/${encodeURIComponent(id)}/video`,
};

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export class CopilotUnavailable extends Error {
  constructor() {
    super("Copilot недоступен на этом сервере");
    this.name = "CopilotUnavailable";
  }
}

// ---------- настройки (localStorage; всё в try/catch — приватный режим) ----------
const LS = {
  get(k: string): string | null {
    try {
      return localStorage.getItem(k);
    } catch {
      return null;
    }
  },
  set(k: string, v: string | null) {
    try {
      if (v === null || v === "") localStorage.removeItem(k);
      else localStorage.setItem(k, v);
    } catch {
      /* noop */
    }
  },
};

export const settings = {
  get apiKey(): string {
    return LS.get("ks.apiKey") ?? "";
  },
  set apiKey(v: string) {
    LS.set("ks.apiKey", v.trim());
  },
  get apiBase(): string {
    const env = (import.meta as { env?: Record<string, string | undefined> }).env?.VITE_API_BASE ?? "";
    return (LS.get("ks.apiBase") ?? env).replace(/\/+$/, "");
  },
  set apiBase(v: string) {
    LS.set("ks.apiBase", v.trim().replace(/\/+$/, ""));
  },
};

export function messageForStatus(status: number, fallback = ""): string {
  if (status === 401 || status === 403) return "Нет доступа: проверьте API-ключ в настройках или обновите страницу.";
  if (status === 404) return "Задача не найдена на сервере (возможно, сервер перезапускался).";
  if (status === 413) return "Файл слишком большой для сервера.";
  if (status === 415 || status === 422) return fallback || "Сервер не принял файл: нужен видеофайл .mov или .mp4.";
  if (status === 429) return "Сервер занят: слишком много задач. Повторите через минуту.";
  if (status >= 500) return fallback || "Ошибка сервера при обработке запроса.";
  if (status === 0) return "Нет связи с сервером.";
  return fallback || `Ошибка запроса (HTTP ${status}).`;
}

async function errorText(res: Response): Promise<string> {
  try {
    const j = (await res.json()) as { detail?: unknown; error?: unknown; message?: unknown };
    const d = j.detail ?? j.error ?? j.message;
    if (typeof d === "string") return d;
  } catch {
    /* noop */
  }
  return "";
}

export interface UploadHandle {
  promise: Promise<string>;
  abort: () => void;
}

export interface Backend {
  kind: "http" | "demo";
  init(): Promise<SessionInfo>;
  upload(file: File, onProgress: (fraction: number) => void): UploadHandle;
  getJob(id: string): Promise<{ job: JobResult; raw: unknown }>;
  ask(id: string, question: string, history: { role: "user" | "assistant"; text: string }[]): Promise<AskResponse>;
  /** URL исходного видео для <video>. fallback — загрузка blob с заголовками, если прямой URL не играет. */
  videoSource(id: string): { url: string; fallback?: () => Promise<string> } | null;
}

export class HttpBackend implements Backend {
  kind = "http" as const;
  private token: string | null = null;
  private features: SessionInfo["features"] = {};

  private url(path: string): string {
    return `${settings.apiBase}${path}`;
  }

  private headers(path: string, extra?: Record<string, string>): Record<string, string> {
    const h: Record<string, string> = { Accept: "application/json", ...extra };
    if (this.token) h["X-UI-Token"] = this.token;
    if (path.startsWith("/api/") && settings.apiKey) h["X-API-Key"] = settings.apiKey;
    return h;
  }

  async init(): Promise<SessionInfo> {
    try {
      const res = await fetch(this.url(ENDPOINTS.session), {
        credentials: "same-origin",
        headers: { Accept: "application/json" },
      });
      if (res.ok) {
        const ct = res.headers.get("content-type") ?? "";
        let token: string | null = null;
        let info: Record<string, unknown> = {};
        if (ct.includes("json")) {
          info = (await res.json()) as Record<string, unknown>;
          const t = info.token ?? info.ui_token ?? info.access_token;
          token = typeof t === "string" ? t : null;
        } else {
          token = (await res.text()).trim() || null;
        }
        this.token = token;
        const f = info.features;
        this.features = typeof f === "object" && f !== null ? (f as SessionInfo["features"]) : {};
        if (typeof info.copilot === "boolean") this.features.copilot = info.copilot;
        return { token, features: this.features, version: typeof info.version === "string" ? info.version : undefined };
      }
    } catch {
      /* сервер недоступен — работаем с API-ключом, если он задан */
    }
    this.token = null;
    return { token: null, features: {} };
  }

  /** GET/POST с автоповтором через /api/v1/*, если /ui/v1/* не реализован (404/405). */
  private async fetchWithFallback(uiPath: string, apiPath: string, init: RequestInit = {}): Promise<Response> {
    const doFetch = (p: string) =>
      fetch(this.url(p), {
        ...init,
        credentials: "same-origin",
        headers: this.headers(p, (init.headers as Record<string, string>) ?? {}),
      });
    if (this.token || !settings.apiKey) {
      let res = await doFetch(uiPath);
      if (res.status === 401 && this.token) {
        await this.init(); // токен мог протухнуть после перезапуска сервера
        res = await doFetch(uiPath);
      }
      if (res.status !== 404 && res.status !== 405) return res;
      if (!settings.apiKey && !this.token) return res;
    }
    return doFetch(apiPath);
  }

  upload(file: File, onProgress: (fraction: number) => void): UploadHandle {
    let xhr: XMLHttpRequest | null = null;
    let aborted = false;
    const send = (path: string): Promise<{ status: number; body: string }> =>
      new Promise((resolve, reject) => {
        xhr = new XMLHttpRequest();
        xhr.open("POST", this.url(path));
        for (const [k, v] of Object.entries(this.headers(path))) xhr.setRequestHeader(k, v);
        xhr.upload.onprogress = (e) => {
          if (e.lengthComputable && e.total > 0) onProgress(Math.min(1, e.loaded / e.total));
        };
        xhr.onload = () => resolve({ status: xhr!.status, body: xhr!.responseText });
        xhr.onerror = () => reject(new ApiError(0, messageForStatus(0)));
        xhr.onabort = () => reject(new DOMException("Загрузка отменена", "AbortError"));
        const fd = new FormData();
        fd.append("file", file, file.name);
        xhr.send(fd);
      });
    const promise = (async () => {
      let r: { status: number; body: string };
      if (this.token || !settings.apiKey) {
        r = await send(ENDPOINTS.uiAnalyze);
        if ((r.status === 404 || r.status === 405 || (r.status === 401 && !this.token)) && !aborted) {
          onProgress(0);
          r = await send(ENDPOINTS.apiAnalyze);
        }
      } else {
        r = await send(ENDPOINTS.apiAnalyze);
      }
      if (r.status < 200 || r.status >= 300) {
        let detail = "";
        try {
          const j = JSON.parse(r.body) as { detail?: unknown };
          if (typeof j.detail === "string") detail = j.detail;
        } catch {
          /* noop */
        }
        throw new ApiError(r.status, messageForStatus(r.status, detail));
      }
      const j = JSON.parse(r.body) as { job_id?: unknown; id?: unknown };
      const id = j.job_id ?? j.id;
      if (typeof id !== "string" && typeof id !== "number") throw new ApiError(500, "Сервер не вернул job_id");
      onProgress(1);
      return String(id);
    })();
    return {
      promise,
      abort: () => {
        aborted = true;
        (xhr as XMLHttpRequest | null)?.abort();
      },
    };
  }

  async getJob(id: string): Promise<{ job: JobResult; raw: unknown }> {
    const res = await this.fetchWithFallback(ENDPOINTS.uiJob(id), ENDPOINTS.apiJob(id));
    if (!res.ok) throw new ApiError(res.status, messageForStatus(res.status, await errorText(res)));
    const raw: unknown = await res.json();
    return { job: normalizeJob(raw, id), raw };
  }

  async ask(
    id: string,
    question: string,
    history: { role: "user" | "assistant"; text: string }[],
  ): Promise<AskResponse> {
    if (this.features.copilot === false) throw new CopilotUnavailable();
    const res = await this.fetchWithFallback(ENDPOINTS.uiAsk(id), ENDPOINTS.apiAsk(id), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history }),
    });
    if (res.status === 501) throw new CopilotUnavailable();
    if (!res.ok) throw new ApiError(res.status, messageForStatus(res.status, await errorText(res)));
    const j = (await res.json()) as Record<string, unknown>;
    const answer = j.answer ?? j.text ?? j.response ?? j.message;
    const sources = Array.isArray(j.sources) ? j.sources.map((s) => (typeof s === "string" ? s : JSON.stringify(s))) : [];
    return { answer: typeof answer === "string" ? answer : JSON.stringify(j), sources };
  }

  copilotEnabled(): boolean {
    return this.features.copilot !== false;
  }

  videoSource(id: string): { url: string; fallback?: () => Promise<string> } {
    const path = ENDPOINTS.video(id);
    const q = this.token ? `?token=${encodeURIComponent(this.token)}` : "";
    return {
      url: this.url(path) + q,
      // <video> не умеет слать заголовки: если прямой URL отвергнут, тянем blob с X-UI-Token/X-API-Key.
      fallback: async () => {
        const res = await fetch(this.url(path), { headers: this.headers(path), credentials: "same-origin" });
        if (!res.ok) throw new ApiError(res.status, messageForStatus(res.status));
        return URL.createObjectURL(await res.blob());
      },
    };
  }
}

// ---------- локальные видео (только что загруженные файлы: играем без сервера) ----------
const localVideos = new Map<string, string>();
export function rememberLocalVideo(jobId: string, file: File) {
  try {
    localVideos.set(jobId, URL.createObjectURL(file));
  } catch {
    /* SSR/тесты */
  }
}
export function localVideo(jobId: string): string | undefined {
  return localVideos.get(jobId);
}

// ---------- история проверок ----------
export interface HistoryItem {
  job_id: string;
  file_name: string;
  size?: number;
  created_at: string;
  verdict?: "violation" | "no_violation" | null;
  n_violations?: number;
  status?: string;
  demo?: boolean;
}

const HKEY = "ks.history";
export function loadHistory(): HistoryItem[] {
  try {
    const v = JSON.parse(LS.get(HKEY) ?? "[]") as unknown;
    return Array.isArray(v) ? (v as HistoryItem[]).filter((x) => x && typeof x.job_id === "string") : [];
  } catch {
    return [];
  }
}
export function saveHistoryItem(item: Partial<HistoryItem> & { job_id: string }) {
  const list = loadHistory();
  const i = list.findIndex((x) => x.job_id === item.job_id);
  const merged: HistoryItem = {
    file_name: "видео",
    created_at: new Date().toISOString(),
    ...(i >= 0 ? list[i] : {}),
    ...item,
  } as HistoryItem;
  if (i >= 0) list.splice(i, 1);
  list.unshift(merged);
  LS.set(HKEY, JSON.stringify(list.slice(0, 20)));
}
export function clearHistory() {
  LS.set(HKEY, null);
}

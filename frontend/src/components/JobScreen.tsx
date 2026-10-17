import { useEffect, useRef, useState } from "react";
import { AlertTriangle, RefreshCw, Upload } from "lucide-react";
import { ApiError, loadHistory, saveHistoryItem } from "../api/client";
import type { JobResult } from "../api/types";
import { useApp } from "../context";
import ProcessingView from "./ProcessingView";
import ResultView from "./ResultView";
import { Spinner } from "./ui";

export const POLL_MS = 1500;

export default function JobScreen({ jobId }: { jobId: string }) {
  const { backend, demo, setLastResult, authenticated, serverOk } = useApp();
  const [job, setJob] = useState<JobResult | null>(null);
  const [raw, setRaw] = useState<unknown>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [attempt, setAttempt] = useState(0);
  const fileName = useRef(loadHistory().find((h) => h.job_id === jobId)?.file_name);

  useEffect(() => {
    let alive = true;
    let timer: number | undefined;
    let failures = 0;
    setError(null);
    const tick = async () => {
      try {
        const r = await backend.getJob(jobId);
        if (!alive) return;
        failures = 0;
        setJob(r.job);
        setRaw(r.raw);
        setError(null);
        if (r.job.status === "done" || r.job.status === "failed") {
          saveHistoryItem({
            job_id: jobId,
            status: r.job.status,
            verdict: r.job.status === "done" ? r.job.verdict : null,
            n_violations: r.job.violations.length,
            demo: demo || undefined,
          });
          if (r.job.status === "done") setLastResult(r.job);
          return;
        }
      } catch (e) {
        if (!alive) return;
        failures++;
        const err = e instanceof ApiError ? e : new ApiError(0, "Не удалось получить статус задачи.");
        // 404/401/403 — опрос бессмысленен; сетевые сбои — терпим и продолжаем.
        if ([401, 403, 404].includes(err.status) || failures >= 10) {
          setError(err);
          return;
        }
      }
      timer = window.setTimeout(tick, POLL_MS);
    };
    void tick();
    return () => {
      alive = false;
      window.clearTimeout(timer);
    };
  }, [backend, jobId, attempt, demo, setLastResult]);

  if (error && (!job || job.status !== "done")) {
    return (
      <StateCard
        title="Не удалось получить результат"
        text={error.message + (!authenticated && serverOk !== false ? " Войдите с ключом доступа на главной странице." : "")}
        onRetry={() => setAttempt((a) => a + 1)}
      />
    );
  }
  if (!job) {
    return (
      <div className="flex min-h-[40vh] items-center justify-center gap-3 text-slate-500">
        <Spinner className="h-5 w-5" /> Получаем статус задачи…
      </div>
    );
  }
  if (job.status === "failed" && job.partialResult && (job.partialResult.violations.length > 0 || job.partialResult.timeline.length > 0)) {
    // A02 отдаёт для UI частичный результат упавшего анализа: показываем найденное, баннер — «Анализ не завершён».
    const shown: JobResult = { ...job.partialResult, job_id: job.job_id, status: "failed", error: job.error };
    return <ResultView job={shown} raw={raw} fileName={fileName.current} />;
  }
  if (job.status === "failed") {
    return (
      <StateCard
        title="Анализ не завершён"
        text={`${job.error ? `Сервер сообщил: «${job.error}». ` : ""}Вывод об отсутствии нарушений сделать нельзя — просмотрите запись вручную или загрузите её повторно.`}
        onRetry={() => setAttempt((a) => a + 1)}
        jobId={job.job_id}
      />
    );
  }
  if (job.status !== "done") return <ProcessingView job={job} fileName={fileName.current} />;
  return <ResultView job={job} raw={raw} fileName={fileName.current} />;
}

function StateCard({ title, text, onRetry, jobId }: { title: string; text: string; onRetry: () => void; jobId?: string }) {
  const { demo } = useApp();
  return (
    <div className="mx-auto max-w-xl animate-fade-in">
      <div className="card overflow-hidden">
        <div className="flex items-center gap-3 bg-amber-400 px-6 py-4 text-amber-950">
          <AlertTriangle className="h-6 w-6" />
          <h1 className="text-lg font-extrabold">{title}</h1>
        </div>
        <div className="space-y-4 p-6">
          <p className="text-sm leading-relaxed text-slate-700 dark:text-slate-200">{text}</p>
          {jobId && <p className="font-mono text-xs text-slate-500">задача {jobId}</p>}
          <div className="flex flex-wrap gap-2">
            <button className="btn-ghost" onClick={onRetry}>
              <RefreshCw className="h-4 w-4" /> Обновить статус
            </button>
            <a className="btn-primary" href={demo ? "?demo=1#/" : "#/"}>
              <Upload className="h-4 w-4" /> Загрузить видео
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}

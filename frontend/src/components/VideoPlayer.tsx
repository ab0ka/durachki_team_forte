import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from "react";
import { AlertTriangle, Pause, Play, RotateCcw, RotateCw, VideoOff } from "lucide-react";
import type { TimelinePoint, Violation } from "../api/types";
import { demoSceneDataUri } from "../api/demo";
import { formatTimecode } from "../lib/format";
import RiskTimeline from "./RiskTimeline";
import { cx, Spinner } from "./ui";

export interface PlayerHandle {
  seek: (t: number, play?: boolean) => void;
}

interface Props {
  source: { url: string; fallback?: () => Promise<string> } | null;
  /** Синтетическая сцена вместо видео (только демо). */
  synthetic?: boolean;
  duration: number;
  timeline: TimelinePoint[];
  violations: Violation[];
  onTime?: (t: number) => void;
}

const SPEEDS = [0.5, 1, 2];

const VideoPlayer = forwardRef<PlayerHandle, Props>(function VideoPlayer(
  { source, synthetic, duration, timeline, violations, onTime },
  ref,
) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [src, setSrc] = useState<string | null>(source?.url ?? null);
  const [fallbackTried, setFallbackTried] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(!!source);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [realDuration, setRealDuration] = useState(0);
  const d = realDuration > 0 ? realDuration : duration;
  const virtual = !source && !!synthetic;

  useEffect(() => {
    setSrc(source?.url ?? null);
    setFallbackTried(false);
    setError(null);
    setLoading(!!source);
  }, [source]);

  // виртуальные часы синтетической сцены
  useEffect(() => {
    if (!virtual || !playing) return;
    let last = performance.now();
    const id = window.setInterval(() => {
      const now = performance.now();
      setT((x) => {
        const nx = x + ((now - last) / 1000) * speed;
        if (nx >= d) {
          setPlaying(false);
          return d;
        }
        return nx;
      });
      last = now;
    }, 100);
    return () => window.clearInterval(id);
  }, [virtual, playing, speed, d]);

  useEffect(() => onTime?.(t), [t, onTime]);

  const seek = useCallback(
    (time: number, play = false) => {
      const tt = Math.min(Math.max(0, time), d || time);
      const v = videoRef.current;
      if (v && src) {
        v.currentTime = tt;
        if (play) void v.play().catch(() => undefined);
      }
      setT(tt);
      if (play && virtual) setPlaying(true);
    },
    [d, src, virtual],
  );

  useImperativeHandle(ref, () => ({ seek }), [seek]);

  const toggle = () => {
    const v = videoRef.current;
    if (virtual) {
      if (t >= d) setT(0);
      setPlaying((p) => !p);
      return;
    }
    if (!v) return;
    if (v.paused) void v.play().catch(() => undefined);
    else v.pause();
  };

  const onVideoError = async () => {
    if (source?.fallback && !fallbackTried) {
      setFallbackTried(true);
      try {
        setSrc(await source.fallback());
        return;
      } catch {
        /* ниже — сообщение */
      }
    }
    setLoading(false);
    setError("Не удалось загрузить исходное видео. Кадры-доказательства доступны в карточках эпизодов.");
  };

  const active = violations.findIndex(
    (v) => t >= (v.t_start ?? v.timecode_sec - 1) && t <= (v.t_end ?? v.timecode_sec + 1),
  );

  return (
    <div>
      <div className="relative mx-auto flex max-h-[68vh] min-h-[220px] items-center justify-center overflow-hidden rounded-xl bg-ink-950">
        {src ? (
          <video
            ref={videoRef}
            src={src}
            className="max-h-[68vh] w-full object-contain"
            playsInline
            preload="metadata"
            onLoadedMetadata={(e) => {
              setLoading(false);
              const dur = e.currentTarget.duration;
              if (Number.isFinite(dur) && dur > 0) setRealDuration(dur);
            }}
            onTimeUpdate={(e) => setT(e.currentTarget.currentTime)}
            onPlay={() => setPlaying(true)}
            onPause={() => setPlaying(false)}
            onError={() => void onVideoError()}
            onClick={toggle}
          />
        ) : virtual ? (
          <img src={demoSceneDataUri(t)} alt="Синтетическая сцена (демо)" className="max-h-[68vh] w-auto max-w-full object-contain" onClick={toggle} />
        ) : (
          <div className="flex flex-col items-center gap-2 p-8 text-center text-sm text-slate-400">
            <VideoOff className="h-8 w-8" />
            Исходное видео недоступно для этой задачи.
          </div>
        )}
        {loading && src && !error && (
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center text-slate-300">
            <Spinner className="h-7 w-7" />
          </div>
        )}
        {error && (
          <div className="absolute inset-x-3 bottom-3 flex items-start gap-2 rounded-lg bg-black/70 p-3 text-xs text-amber-200">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" /> {error}
          </div>
        )}
        {virtual && (
          <div className="pointer-events-none absolute left-3 top-3 rounded bg-amber-400 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider text-amber-950">
            синтетическая сцена
          </div>
        )}
        {active >= 0 && (
          <div className="pointer-events-none absolute right-3 top-3 max-w-[70%] animate-fade-in rounded-lg bg-brand-600/95 px-2.5 py-1.5 text-xs font-semibold text-white shadow-lift">
            Эпизод {active + 1}: {violations[active].type}
          </div>
        )}
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-2">
        <button className="btn-primary h-10 w-10 rounded-full p-0" onClick={toggle} aria-label={playing ? "Пауза" : "Воспроизвести"} disabled={!src && !virtual}>
          {playing ? <Pause className="h-4 w-4" /> : <Play className="ml-0.5 h-4 w-4" />}
        </button>
        <button className="rounded-lg p-2 text-slate-500 hover:bg-slate-100 dark:hover:bg-white/10" onClick={() => seek(t - 5)} aria-label="Назад 5 с" title="Назад 5 с">
          <RotateCcw className="h-4 w-4" />
        </button>
        <button className="rounded-lg p-2 text-slate-500 hover:bg-slate-100 dark:hover:bg-white/10" onClick={() => seek(t + 5)} aria-label="Вперёд 5 с" title="Вперёд 5 с">
          <RotateCw className="h-4 w-4" />
        </button>
        <div className="ml-1 font-mono text-sm tnum">
          <span className="font-semibold">{formatTimecode(t)}</span>
          <span className="text-slate-400"> / {formatTimecode(d)}</span>
        </div>
        <div className="ml-auto flex overflow-hidden rounded-lg border border-slate-200 text-xs dark:border-white/10">
          {SPEEDS.map((s) => (
            <button
              key={s}
              className={cx("px-2.5 py-1.5 font-semibold tnum", s === speed ? "bg-ink-900 text-white dark:bg-white dark:text-ink-900" : "text-slate-500 hover:bg-slate-50 dark:hover:bg-white/5")}
              onClick={() => {
                setSpeed(s);
                if (videoRef.current) videoRef.current.playbackRate = s;
              }}
            >
              {s}×
            </button>
          ))}
        </div>
      </div>
      {d > 0 ? (
        <div className="mt-5">
          <RiskTimeline duration={d} timeline={timeline} violations={violations} currentTime={t} onSeek={(x) => seek(x)} />
        </div>
      ) : (
        <p className="mt-4 text-xs text-slate-500">Длительность записи неизвестна — шкала риска недоступна.</p>
      )}
    </div>
  );
});

export default VideoPlayer;

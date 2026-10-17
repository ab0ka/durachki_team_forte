import { useEffect, useState } from "react";
import { ArrowRight, Crosshair, ImageOff } from "lucide-react";
import type { EvidenceFrame, Violation } from "../api/types";
import { imageSrc, type HighlightBox } from "../api/normalize";
import { formatTimecode } from "../lib/format";
import { cx, Spinner } from "./ui";

/** Кадр с необязательной подсветкой рамками (A12 shapes; координаты нормированы к кадру). */
export function EvidenceImage({
  src,
  boxes,
  alt,
  className,
  imgClassName,
}: {
  src: string;
  boxes: HighlightBox[];
  alt: string;
  className?: string;
  imgClassName?: string;
}) {
  return (
    <div className={cx("relative", className ?? "block")}>
      <img src={src} alt={alt} draggable={false} className={cx("block select-none", imgClassName ?? "h-auto w-full")} />
      {boxes.length > 0 && (
        <svg className="pointer-events-none absolute inset-0 h-full w-full" viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden>
          {boxes.map((b, i) => (
            <rect
              key={i}
              x={b.bbox[0]}
              y={b.bbox[1]}
              width={b.bbox[2] - b.bbox[0]}
              height={b.bbox[3] - b.bbox[1]}
              fill="none"
              stroke={b.color}
              strokeWidth={2.5}
              vectorEffect="non-scaling-stroke"
            />
          ))}
        </svg>
      )}
      {boxes.length > 0 && (
        <div className="pointer-events-none absolute left-1 top-1 flex max-w-[95%] flex-wrap gap-1">
          {boxes.slice(0, 3).map((b, i) =>
            b.label ? (
              <span key={i} className="rounded px-1 py-0.5 text-[10px] font-semibold leading-none text-white" style={{ background: b.color }}>
                {b.label}
              </span>
            ) : null,
          )}
        </div>
      )}
    </div>
  );
}

/** Рамки применимы к кадру, близкому к пиковому моменту (A12 рисует их на кадре t_peak). */
export function boxesForFrame(frame: EvidenceFrame, v: Violation, boxes: HighlightBox[], maxDt = 0.6): HighlightBox[] {
  return Math.abs(frame.timestamp_sec - v.timecode_sec) <= maxDt ? boxes : [];
}

/**
 * «До / после» для эпизодов-переходов («ушла и оставила», сейф оставлен открытым, сумка оставлена).
 * «До» — кадр сервера до начала эпизода, иначе кадр из видео в браузере (t_start − beforeS); если видео
 * не воспроизводится — кнопка перехода к этому моменту. «После» — кадр сервера в момент/после начала.
 */
export function BeforeAfter({
  v,
  boxes,
  highlight,
  capture,
  onSeek,
  onOpen,
  beforeS = 2,
}: {
  v: Violation;
  boxes: HighlightBox[];
  highlight: boolean;
  capture?: (t: number) => Promise<string | null>;
  onSeek: (t: number) => void;
  onOpen: (index: number) => void;
  beforeS?: number;
}) {
  const start = v.t_start ?? v.timecode_sec;
  const beforeT = Math.max(0, start - beforeS);
  const frames = v.evidence_frames;
  const serverBeforeIdx = frames.findIndex((f) => f.timestamp_sec <= start - 0.3);
  const afterCandidates = frames.map((f, i) => ({ f, i })).filter(({ f }) => f.timestamp_sec >= start - 0.05);
  const after =
    afterCandidates.sort((a, b) => Math.abs(a.f.timestamp_sec - v.timecode_sec) - Math.abs(b.f.timestamp_sec - v.timecode_sec))[0] ??
    (frames.length ? { f: frames[frames.length - 1], i: frames.length - 1 } : undefined);
  const [captured, setCaptured] = useState<string | null | "loading" | "error">(null);

  useEffect(() => {
    let alive = true;
    if (serverBeforeIdx >= 0 || !capture) return;
    setCaptured("loading");
    void capture(beforeT).then((r) => alive && setCaptured(r ?? "error"));
    return () => {
      alive = false;
    };
  }, [serverBeforeIdx, capture, beforeT]);

  const slot = "min-w-0 flex-1 overflow-hidden rounded-lg bg-ink-950 ring-1 ring-slate-200 dark:ring-white/10";
  return (
    <div className="max-w-md">
      <div className="label mb-2">До и после начала эпизода</div>
      <div className="flex items-stretch gap-2">
        <figure className="min-w-0 flex-1">
          <div className={slot}>
            {serverBeforeIdx >= 0 ? (
              <button className="block w-full" onClick={() => onOpen(serverBeforeIdx)} aria-label="Открыть кадр «до»">
                <EvidenceImage src={imageSrc(frames[serverBeforeIdx].image_base64)} boxes={[]} alt="Кадр до начала эпизода" />
              </button>
            ) : captured && captured !== "loading" && captured !== "error" ? (
              <button className="block w-full" onClick={() => onSeek(beforeT)} aria-label="Перейти к моменту «до» в видео">
                <EvidenceImage src={captured} boxes={[]} alt="Кадр из видео до начала эпизода" />
              </button>
            ) : (
              <div className="flex aspect-[3/4] flex-col items-center justify-center gap-2 p-3 text-center text-xs text-slate-300">
                {captured === "loading" ? <Spinner className="h-5 w-5" /> : <ImageOff className="h-5 w-5" />}
                {captured === "loading" ? "берём кадр из видео…" : "кадра «до» нет"}
                <button className="btn bg-white/10 px-2 py-1 text-[11px] text-white hover:bg-white/20" onClick={() => onSeek(beforeT)}>
                  <Crosshair className="h-3 w-3" /> к {formatTimecode(beforeT)}
                </button>
              </div>
            )}
          </div>
          <figcaption className="mt-1 text-[11px] text-slate-500 tnum">
            до · {formatTimecode(serverBeforeIdx >= 0 ? frames[serverBeforeIdx].timestamp_sec : beforeT)}
            {serverBeforeIdx < 0 && captured && captured !== "loading" && captured !== "error" && " · кадр из видео, без размытия"}
          </figcaption>
        </figure>
        <ArrowRight className="mt-[25%] h-5 w-5 shrink-0 text-slate-400" aria-hidden />
        <figure className="min-w-0 flex-1">
          <div className={slot}>
            {after ? (
              <button className="block w-full" onClick={() => onOpen(after.i)} aria-label="Открыть кадр «после»">
                <EvidenceImage
                  src={imageSrc(after.f.image_base64)}
                  boxes={highlight ? boxesForFrame(after.f, v, boxes, 1.5) : []}
                  alt="Кадр после начала эпизода"
                />
              </button>
            ) : (
              <div className="flex aspect-[3/4] items-center justify-center p-3 text-center text-xs text-slate-300">кадров нет</div>
            )}
          </div>
          <figcaption className="mt-1 text-[11px] text-slate-500 tnum">после · {after ? formatTimecode(after.f.timestamp_sec) : "—"}</figcaption>
        </figure>
      </div>
    </div>
  );
}

import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { ChevronLeft, ChevronRight, Crosshair, Minus, Plus, X } from "lucide-react";
import type { EvidenceFrame } from "../api/types";
import { imageSrc, type HighlightBox } from "../api/normalize";
import { EvidenceImage } from "./Evidence";
import { formatTimecode } from "../lib/format";

interface Props {
  frames: EvidenceFrame[];
  index: number;
  caption: string;
  onClose: () => void;
  onIndex: (i: number) => void;
  onSeek: (t: number) => void;
  boxesFor?: (f: EvidenceFrame) => HighlightBox[];
}

/** Просмотр кадра-доказательства: зум (колесо/кнопки/двойной клик), панорама, стрелки, Esc. */
export default function Lightbox({ frames, index, caption, onClose, onIndex, onSeek, boxesFor }: Props) {
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const drag = useRef<{ x: number; y: number; px: number; py: number } | null>(null);
  const f = frames[index];

  useEffect(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, [index]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowRight" && index < frames.length - 1) onIndex(index + 1);
      if (e.key === "ArrowLeft" && index > 0) onIndex(index - 1);
      if (e.key === "+" || e.key === "=") setZoom((z) => Math.min(6, z * 1.25));
      if (e.key === "-") setZoom((z) => Math.max(1, z / 1.25));
    };
    window.addEventListener("keydown", onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = prev;
    };
  }, [index, frames.length, onClose, onIndex]);

  if (!f) return null;
  if (typeof document === "undefined") return null;
  const btn = "flex h-10 w-10 items-center justify-center rounded-full bg-white/10 text-white backdrop-blur hover:bg-white/20 disabled:opacity-30";

  // Портал в body: fixed-оверлей не зависит от transform/overflow предков.
  return createPortal(
    <div className="no-print fixed inset-0 z-[70] flex flex-col bg-black/90" role="dialog" aria-modal="true" aria-label="Кадр-доказательство">
      <div className="flex items-center gap-3 px-4 py-3 text-sm text-white">
        <div className="min-w-0 flex-1">
          <div className="truncate font-semibold">{caption}</div>
          <div className="text-xs text-white/60 tnum">
            Кадр {index + 1} из {frames.length} · {formatTimecode(f.timestamp_sec)} · головы размыты на сервере
          </div>
        </div>
        <button className={btn} onClick={() => setZoom((z) => Math.max(1, z / 1.25))} aria-label="Уменьшить" disabled={zoom <= 1}>
          <Minus className="h-4 w-4" />
        </button>
        <span className="w-12 text-center text-xs tnum">{Math.round(zoom * 100)} %</span>
        <button className={btn} onClick={() => setZoom((z) => Math.min(6, z * 1.25))} aria-label="Увеличить">
          <Plus className="h-4 w-4" />
        </button>
        <button className={btn} onClick={onClose} aria-label="Закрыть">
          <X className="h-5 w-5" />
        </button>
      </div>
      <div
        className="relative flex flex-1 touch-none items-center justify-center overflow-hidden"
        onWheel={(e) => setZoom((z) => Math.min(6, Math.max(1, z * (e.deltaY < 0 ? 1.12 : 1 / 1.12))))}
        onDoubleClick={() => {
          setZoom((z) => (z > 1 ? 1 : 2.5));
          setPan({ x: 0, y: 0 });
        }}
        onPointerDown={(e) => {
          drag.current = { x: e.clientX, y: e.clientY, px: pan.x, py: pan.y };
          (e.currentTarget as Element).setPointerCapture?.(e.pointerId);
        }}
        onPointerMove={(e) => {
          if (!drag.current || zoom <= 1) return;
          setPan({ x: drag.current.px + (e.clientX - drag.current.x), y: drag.current.py + (e.clientY - drag.current.y) });
        }}
        onPointerUp={() => (drag.current = null)}
        onClick={(e) => e.target === e.currentTarget && zoom === 1 && onClose()}
      >
        <div
          className="max-h-full max-w-full transition-transform duration-75"
          style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`, cursor: zoom > 1 ? "grab" : "zoom-in" }}
        >
          <EvidenceImage
            src={imageSrc(f.image_base64)}
            boxes={boxesFor ? boxesFor(f) : []}
            alt={`Кадр-доказательство ${formatTimecode(f.timestamp_sec)}`}
            className="relative inline-block max-h-[calc(100vh-140px)]"
            imgClassName="max-h-[calc(100vh-140px)] w-auto max-w-[min(92vw,1100px)] object-contain"
          />
        </div>
        <button className={`${btn} absolute left-3 top-1/2 -translate-y-1/2`} onClick={() => onIndex(index - 1)} disabled={index === 0} aria-label="Предыдущий кадр">
          <ChevronLeft className="h-5 w-5" />
        </button>
        <button className={`${btn} absolute right-3 top-1/2 -translate-y-1/2`} onClick={() => onIndex(index + 1)} disabled={index >= frames.length - 1} aria-label="Следующий кадр">
          <ChevronRight className="h-5 w-5" />
        </button>
      </div>
      <div className="flex justify-center gap-2 px-4 py-3">
        <button
          className="btn bg-white text-ink-900 hover:bg-slate-200"
          onClick={() => {
            onSeek(f.timestamp_sec);
            onClose();
          }}
        >
          <Crosshair className="h-4 w-4" /> Перейти к {formatTimecode(f.timestamp_sec)} в видео
        </button>
      </div>
    </div>,
    document.body,
  );
}

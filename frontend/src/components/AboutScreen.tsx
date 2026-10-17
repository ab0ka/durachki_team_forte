import { ArrowRight, Cpu, EyeOff, FileSearch, Lock, Scale } from "lucide-react";
import { useApp } from "../context";
import { DEFAULT_MODELS, TYPE_CATALOG } from "../lib/catalog";
import { LocalBadge } from "./ui";

const PIPELINE = [
  ["Видео", "декодирование, поворот, 5 к/с"],
  ["Люди и предметы", "детекция, трекинг, роли по зонам"],
  ["Поза и сцена", "руки, сейф, дверь, наличные"],
  ["Правила + временная модель", "события и кривая риска"],
  ["Локальная VLM", "верификация эпизодов"],
  ["Отчёт", "вердикт, кадры, нормы РК"],
];

export default function AboutScreen() {
  const { lastResult } = useApp();
  const serverModels = Array.isArray(lastResult?.meta.models) ? (lastResult!.meta.models as unknown[]) : [];
  return (
    <div className="animate-fade-in space-y-6">
      <section className="card p-6 sm:p-8">
        <LocalBadge />
        <h1 className="mt-4 text-3xl font-extrabold">О системе Kassa Sentinel</h1>
        <p className="mt-3 max-w-3xl text-[15px] leading-relaxed text-slate-600 dark:text-slate-300">
          Локальный AI-сервис для проверяющего: анализирует запись потолочной камеры кассы и подсвечивает эпизоды с признаками
          возможных нарушений — с таймкодом, кадрами-доказательствами и ссылкой на норму законодательства РК. Видео и кадры не
          передаются во внешние облачные сервисы; все модели работают на сервере банка.
        </p>
        <div className="mt-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {[
            [Lock, "Локально", "Нет облачных AI API в рантайме, веса загружены заранее"],
            [EyeOff, "Без распознавания лиц", "Роли — по зоне, поведению и предметам; головы на кадрах размыты"],
            [FileSearch, "Только видимое", "Описания — наблюдаемые факты, без суждений об умысле и суммах"],
            [Scale, "Нормы из каталога", "Ссылки только на проверенный каталог норм с adilet.zan.kz"],
          ].map(([I, t, d]) => {
            const Icon = I as typeof Lock;
            return (
              <div key={t as string} className="rounded-xl border border-slate-200 p-4 dark:border-white/10">
                <Icon className="h-5 w-5 text-brand-600 dark:text-brand-300" />
                <div className="mt-2 text-sm font-bold">{t as string}</div>
                <div className="mt-1 text-xs text-slate-500">{d as string}</div>
              </div>
            );
          })}
        </div>
      </section>

      <section className="card p-6">
        <h2 className="mb-4 flex items-center gap-2 text-lg font-bold">
          <Cpu className="h-5 w-5 text-slate-400" /> Архитектура анализа
        </h2>
        <ol className="grid gap-2 sm:grid-cols-2 lg:grid-cols-6">
          {PIPELINE.map(([t, d], i) => (
            <li key={t} className="relative rounded-xl bg-slate-50 p-3 dark:bg-white/[.04]">
              <div className="text-[11px] font-bold text-brand-600 dark:text-brand-300">{String(i + 1).padStart(2, "0")}</div>
              <div className="text-sm font-semibold">{t}</div>
              <div className="text-xs text-slate-500">{d}</div>
              {i < PIPELINE.length - 1 && <ArrowRight className="absolute -right-2.5 top-1/2 z-10 hidden h-4 w-4 -translate-y-1/2 text-slate-300 lg:block" />}
            </li>
          ))}
        </ol>
      </section>

      <div className="grid gap-6 lg:grid-cols-2">
        <section className="card p-6">
          <h2 className="mb-3 text-lg font-bold">Типы нарушений</h2>
          <ul className="divide-y divide-slate-100 dark:divide-white/5">
            {TYPE_CATALOG.map((t) => (
              <li key={t.id} className="flex gap-3 py-2.5">
                <span className="w-10 shrink-0 font-mono text-xs font-bold text-brand-700 dark:text-brand-300">{t.id}</span>
                <span>
                  <span className="block text-sm font-semibold">{t.name}</span>
                  <span className="block text-xs text-slate-500">{t.observable}</span>
                </span>
              </li>
            ))}
          </ul>
        </section>
        <section className="card p-6">
          <h2 className="mb-3 text-lg font-bold">Модели и лицензии</h2>
          <p className="mb-3 text-xs text-slate-500">
            {serverModels.length ? "По данным последнего анализа на этом сервере." : "Состав по архитектуре; фактический список сервер отдаёт в meta.models."}
          </p>
          <ul className="divide-y divide-slate-100 dark:divide-white/5">
            {(serverModels.length
              ? serverModels.map((m) =>
                  typeof m === "string"
                    ? { name: m, license: "", purpose: "" }
                    : { name: String((m as { name?: unknown }).name ?? "?"), license: String((m as { license?: unknown }).license ?? ""), purpose: String((m as { purpose?: unknown }).purpose ?? "") },
                )
              : DEFAULT_MODELS
            ).map((m, i) => (
              <li key={i} className="flex items-center justify-between gap-3 py-2.5">
                <span>
                  <span className="block text-sm font-semibold">{m.name}</span>
                  <span className="block text-xs text-slate-500">{m.purpose}</span>
                </span>
                {m.license && <span className="chip shrink-0 border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-400/20 dark:bg-emerald-400/10 dark:text-emerald-300">{m.license}</span>}
              </li>
            ))}
          </ul>
        </section>
      </div>
    </div>
  );
}

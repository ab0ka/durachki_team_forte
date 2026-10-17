import { useEffect, useState } from "react";
import { Eye, EyeOff, KeyRound, Trash2 } from "lucide-react";
import { clearHistory, settings } from "../api/client";
import { useApp } from "../context";
import { Modal } from "./ui";

export default function SettingsDialog({ open, onClose, onSaved }: { open: boolean; onClose: () => void; onSaved: () => void }) {
  const { session, demo, toast } = useApp();
  const [apiKey, setApiKey] = useState("");
  const [apiBase, setApiBase] = useState("");
  const [show, setShow] = useState(false);

  useEffect(() => {
    if (open) {
      setApiKey(settings.apiKey);
      setApiBase(settings.apiBase);
    }
  }, [open]);

  return (
    <Modal open={open} onClose={onClose} title="Настройки">
      <form
        className="space-y-5"
        onSubmit={(e) => {
          e.preventDefault();
          settings.apiKey = apiKey;
          settings.apiBase = apiBase;
          onSaved();
        }}
      >
        <div className="rounded-xl bg-slate-50 p-3 text-sm text-slate-600 dark:bg-white/5 dark:text-slate-300">
          {demo
            ? "Демо-режим: запросы к серверу не выполняются."
            : session?.token
              ? "Сессия интерфейса активна (X-UI-Token). API-ключ нужен только как запасной путь."
              : "Сессия интерфейса не получена. Укажите API-ключ, чтобы работать через /api/v1."}
        </div>
        <label className="block">
          <span className="label mb-1.5 flex items-center gap-1.5">
            <KeyRound className="h-3.5 w-3.5" /> API-ключ (X-API-Key)
          </span>
          <div className="relative">
            <input
              className="input pr-11 font-mono"
              type={show ? "text" : "password"}
              autoComplete="off"
              spellCheck={false}
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="не задан"
            />
            <button
              type="button"
              className="absolute inset-y-0 right-0 flex w-11 items-center justify-center text-slate-500"
              onClick={() => setShow((s) => !s)}
              aria-label={show ? "Скрыть ключ" : "Показать ключ"}
            >
              {show ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
            </button>
          </div>
          <span className="mt-1 block text-xs text-slate-500">Хранится только в этом браузере (localStorage).</span>
        </label>
        <label className="block">
          <span className="label mb-1.5 block">Адрес API (необязательно)</span>
          <input
            className="input font-mono"
            value={apiBase}
            onChange={(e) => setApiBase(e.target.value)}
            placeholder="тот же адрес, что и у интерфейса"
            spellCheck={false}
          />
        </label>
        <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
          <button
            type="button"
            className="btn-ghost"
            onClick={() => {
              clearHistory();
              toast("История проверок очищена", "success");
            }}
          >
            <Trash2 className="h-4 w-4" /> Очистить историю
          </button>
          <div className="flex gap-2">
            <button type="button" className="btn-ghost" onClick={onClose}>
              Отмена
            </button>
            <button type="submit" className="btn-primary">
              Сохранить
            </button>
          </div>
        </div>
      </form>
    </Modal>
  );
}

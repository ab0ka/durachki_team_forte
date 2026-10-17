import type { JobResult, Verdict } from "../api/types";

// Формулировки ТЗ — дословно.
export const VERDICT_TEXT: Record<Verdict, string> = {
  violation: "Обнаружены признаки возможных нарушений",
  no_violation: "Нарушение не обнаружено",
};

export interface VerdictView {
  /** success (зелёный) — ТОЛЬКО для завершённого полного анализа без нарушений. */
  tone: "danger" | "success" | "warning" | "neutral";
  title: string;
  subtitle: string;
}

export function verdictView(
  job: Pick<JobResult, "verdict" | "violations" | "status">,
  partial = false,
): VerdictView {
  if (job.status === "failed") {
    return {
      tone: "warning",
      title: "Анализ не завершён",
      subtitle: "Сервер сообщил об ошибке обработки. Вывод об отсутствии нарушений сделать нельзя — нужен ручной просмотр.",
    };
  }
  if (job.status !== "done") {
    return { tone: "neutral", title: "Анализ выполняется", subtitle: "Результат появится после завершения обработки." };
  }
  if (job.verdict === "violation") {
    const n = job.violations.length;
    const tail =
      n === 0
        ? "Требуется просмотр записи проверяющим."
        : `${n} ${n === 1 ? "эпизод" : n < 5 ? "эпизода" : "эпизодов"} для проверки. Результат — подсказка, не юридическое заключение.`;
    return {
      tone: "danger",
      title: VERDICT_TEXT.violation,
      subtitle: partial ? `${tail} Анализ неполный — часть стадий пропущена.` : tail,
    };
  }
  if (job.verdict === "no_violation") {
    if (partial) {
      return {
        tone: "warning",
        title: "Анализ неполный",
        subtitle: "Признаков нарушений не найдено, но часть стадий анализа пропущена. Нужен ручной просмотр записи.",
      };
    }
    return {
      tone: "success",
      title: VERDICT_TEXT.no_violation,
      subtitle: "Видимых признаков нарушений регламента в записи не выявлено.",
    };
  }
  return { tone: "warning", title: "Вердикт не получен", subtitle: "Сервер не вернул вердикт — нужен ручной просмотр." };
}

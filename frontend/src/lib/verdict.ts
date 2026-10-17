import type { JobResult, Verdict } from "../api/types";

// Формулировки ТЗ — дословно.
export const VERDICT_TEXT: Record<Verdict, string> = {
  violation: "Обнаружены признаки возможных нарушений",
  no_violation: "Нарушение не обнаружено",
};

export interface VerdictView {
  tone: "danger" | "success" | "neutral";
  title: string;
  subtitle: string;
}

export function verdictView(job: Pick<JobResult, "verdict" | "violations" | "status">): VerdictView {
  if (job.status === "failed") {
    return { tone: "neutral", title: "Анализ не завершён", subtitle: "Сервер сообщил об ошибке обработки видео." };
  }
  if (job.verdict === "violation") {
    const n = job.violations.length;
    const tail =
      n === 0
        ? "Требуется просмотр записи проверяющим."
        : `${n} ${n === 1 ? "эпизод" : n < 5 ? "эпизода" : "эпизодов"} для проверки. Результат — подсказка, не юридическое заключение.`;
    return { tone: "danger", title: VERDICT_TEXT.violation, subtitle: tail };
  }
  if (job.verdict === "no_violation") {
    return {
      tone: "success",
      title: VERDICT_TEXT.no_violation,
      subtitle: "Видимых признаков нарушений регламента в записи не выявлено.",
    };
  }
  return { tone: "neutral", title: "Вердикт не получен", subtitle: "Сервер не вернул вердикт." };
}

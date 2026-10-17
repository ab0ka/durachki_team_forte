// Встроенный справочник типов (для иконок/подписей). Источник истины по названиям
// и нормам — сервер (norms/catalog.yaml, A13); здесь — только запасные подписи.

export interface TypeInfo {
  id: string;
  name: string;
  short: string;
  observable: string;
}

export const TYPE_CATALOG: TypeInfo[] = [
  {
    id: "V01",
    name: "Посторонний в кассовом помещении",
    short: "Посторонний",
    observable: "Человек, не похожий по поведению/зоне на кассира, находится за кассовым столом или у ценностей",
  },
  {
    id: "V02",
    name: "Рабочее место оставлено с ценностями",
    short: "Без присмотра",
    observable: "Кассир покинул зону рабочего места, на столе видны наличные или ценности",
  },
  {
    id: "V03",
    name: "Сейф открыт без присмотра / при посторонних",
    short: "Сейф",
    observable: "Дверца сейфа открыта, рядом нет кассира либо в зоне есть посторонние",
  },
  {
    id: "V04",
    name: "Дверь кассового помещения открыта",
    short: "Дверь",
    observable: "Дверь/перегородка кассы открыта во время операций, возможен вход постороннего",
  },
  {
    id: "V05",
    name: "Мобильный телефон при работе с ценностями",
    short: "Телефон",
    observable: "Телефон в руке или у уха кассира, рядом наличные",
  },
  {
    id: "V06",
    name: "Приём пищи/напитков на рабочем месте",
    short: "Еда/напитки",
    observable: "Еда или напиток в руках/у лица при наличных на столе",
  },
  {
    id: "V07",
    name: "Нарушение порядка передачи инкассаторской сумки",
    short: "Инкассация",
    observable: "Сумка без присмотра, передача вне процедуры, вскрытие при посторонних",
  },
  {
    id: "V08",
    name: "Пересчёт/передача наличных вне обзора камеры",
    short: "Вне обзора",
    observable: "Руки с наличными скрыты от камеры или операция вынесена из поля зрения",
  },
  {
    id: "V99",
    name: "Иное подозрительное действие",
    short: "Иное",
    observable: "Конкретные видимые факты, не подпадающие под типы V01–V08",
  },
];

export function typeInfo(id: string | undefined): TypeInfo | undefined {
  if (!id) return undefined;
  return TYPE_CATALOG.find((t) => t.id === id.toUpperCase());
}

export const SOURCE_LABELS: Record<string, { label: string; hint: string }> = {
  rules: { label: "Правила", hint: "Детерминированные правила по сигналам сцены (зоны, предметы, присутствие)" },
  temporal: { label: "Временная модель", hint: "Обученная модель по последовательности кадров (SigLIP2 + temporal head)" },
  vlm: { label: "Локальная VLM", hint: "Визуально-языковая модель на этом сервере проверила кадры события" },
  fusion: { label: "Слияние", hint: "Итоговое решение по совокупности источников" },
  demo: { label: "Демо", hint: "Встроенный демонстрационный результат" },
};

export function sourceLabel(s: string): { label: string; hint: string } {
  return SOURCE_LABELS[s.toLowerCase()] ?? { label: s, hint: "Источник сигнала" };
}

/** Модели по умолчанию (если сервер не прислал meta.models). Только лицензии с коммерческим использованием. */
export const DEFAULT_MODELS: { name: string; license: string; purpose: string }[] = [
  { name: "RT-DETRv2 / D-FINE", license: "Apache-2.0", purpose: "Детекция людей и предметов" },
  { name: "OWLv2 / Grounding DINO", license: "Apache-2.0", purpose: "Open-vocabulary: сейф, сумка, наличные" },
  { name: "ByteTrack (supervision)", license: "MIT", purpose: "Трекинг людей и предметов" },
  { name: "RTMPose (rtmlib)", license: "Apache-2.0", purpose: "Поза и жесты рук" },
  { name: "SigLIP2 + temporal head", license: "Apache-2.0", purpose: "Временная модель риска" },
  { name: "Qwen2.5-VL-7B / Qwen3-VL", license: "Apache-2.0", purpose: "Локальная верификация событий" },
  { name: "Qwen2.5-7B / Qwen3 (llama.cpp)", license: "Apache-2.0 / MIT", purpose: "Copilot проверяющего" },
];

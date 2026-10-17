import { describe, expect, it } from "vitest";
import { normalizeJob } from "../src/api/normalize";
import { analyzeCompleteness, readCoverage, readStages } from "../src/lib/completeness";

const base = { job_id: "c", status: "done", verdict: "no_violation", violations: [] };

describe("analyzeCompleteness (complete / partial / unknown)", () => {
  it("unknown, если сервер не подтвердил полноту — unknown ≠ «нарушений нет»", () => {
    const r = analyzeCompleteness(normalizeJob(base));
    expect(r.state).toBe("unknown");
    expect(r.reasons[0]).toMatch(/не подтвердил полноту/);
  });
  it("complete по meta.complete=true (A03)", () => {
    expect(analyzeCompleteness(normalizeJob({ ...base, meta: { complete: true } })).state).toBe("complete");
  });
  it("complete по coverage ≥ 0.98 и всем обязательным стадиям ok/cached", () => {
    const meta = { coverage: 0.99, required_stages: ["ingest"], stages: { ingest: { status: "ok" }, vlm_verify: { status: "skipped" } } };
    expect(analyzeCompleteness(normalizeJob({ ...base, meta })).state).toBe("complete");
  });
  it("partial: coverage < 0.98 (A04), с процентом", () => {
    const r = analyzeCompleteness(normalizeJob({ ...base, meta: { complete: true, coverage: 0.874 } }));
    expect(r.state).toBe("partial");
    expect(r.reasons.join(" ")).toMatch(/87 %/);
  });
  it("partial: обязательная стадия не выполнена (A03 stages)", () => {
    const meta = { stages: { detect: { status: "timeout", required: true, seconds: 30 } } };
    const r = analyzeCompleteness(normalizeJob({ ...base, meta }));
    expect(r.state).toBe("partial");
    expect(r.reasons.join(" ")).toMatch(/Поиск людей и объектов.*таймаут/);
  });
  it("partial: complete=false / degraded / deadline / failed", () => {
    expect(analyzeCompleteness(normalizeJob({ ...base, meta: { complete: false } })).state).toBe("partial");
    expect(analyzeCompleteness(normalizeJob({ ...base, meta: { complete: true, degraded: true } })).state).toBe("partial");
    expect(analyzeCompleteness(normalizeJob({ ...base, meta: { complete: true, deadline_exceeded_s: 3.2 } })).state).toBe("partial");
    expect(analyzeCompleteness(normalizeJob({ ...base, status: "failed", verdict: null })).state).toBe("partial");
  });
  it("mock: meta.mock, origin=mock у стадии, FAKE PIPELINE в предупреждениях", () => {
    expect(analyzeCompleteness(normalizeJob({ ...base, meta: { mock: true } })).mock).toBe(true);
    expect(analyzeCompleteness(normalizeJob({ ...base, meta: { stages: { detect: { status: "ok", origin: "mock" } } } })).mock).toBe(true);
    expect(analyzeCompleteness(normalizeJob({ ...base, meta: { warnings: ["FAKE PIPELINE: result is synthetic"] } })).mock).toBe(true);
    expect(analyzeCompleteness(normalizeJob({ ...base, meta: { complete: true } })).mock).toBe(false);
  });
  it("кэш-стадии перечисляются", () => {
    const r = analyzeCompleteness(normalizeJob({ ...base, meta: { complete: true, stages: { detect: { status: "cached" } } } }));
    expect(r.state).toBe("complete");
    expect(r.cached).toEqual(["Поиск людей и объектов"]);
  });
});

describe("readCoverage / readStages", () => {
  it("форматы coverage", () => {
    expect(readCoverage({ coverage: 0.5 })).toBe(0.5);
    expect(readCoverage({ coverage: "87%" })).toBeCloseTo(0.87);
    expect(readCoverage({ coverage: { ratio: 0.9 } })).toBe(0.9);
    expect(readCoverage({ video: { coverage: 1 } })).toBe(1);
    expect(readCoverage({ coverage: "abc" })).toBeUndefined();
    expect(readCoverage({})).toBeUndefined();
  });
  it("стадии: подписи и статусы по-русски, обязательность из required_stages", () => {
    const st = readStages({ required_stages: ["fusion"], stages: { fusion: { status: "crashed", error: "x" }, temporal: { status: "disabled" } } });
    expect(st.find((x) => x.name === "fusion")).toMatchObject({ label: "Формирование отчёта", statusLabel: "сбой", required: true, ok: false });
    expect(st.find((x) => x.name === "temporal")).toMatchObject({ statusLabel: "выключена", required: false });
  });
});

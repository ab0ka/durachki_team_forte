// Мок-e2e реального HTTP-пути (НЕ демо): бэкенд A02 подменён page.route с синтетическими ответами.
// Проверяет: вход по ключу (cookie-сессия) → загрузка → поллинг → результат → перемотка по таймкоду,
// Copilot скрывается при 501, failed не показывается зелёным, в URL нет ключа.
// Запуск: npm run build && npm run e2e:mock
import assert from "node:assert/strict";
import { mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { launch, loadPlaywright, startPreview } from "./pw.mjs";

const shots = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../screenshots");
mkdirSync(shots, { recursive: true });

const KEY = "ks_test_key_e2e";
const svg = (label) =>
  Buffer.from(
    `<svg xmlns="http://www.w3.org/2000/svg" width="232" height="424"><rect width="232" height="424" fill="#cfd8de"/><text x="20" y="200" font-size="20">${label}</text></svg>`,
  ).toString("base64");

function doneJob(id) {
  return {
    job_id: id,
    status: "done",
    verdict: "violation",
    violations: [
      {
        type: "Синтетический тип A",
        timecode_sec: 12.4,
        description: "Синтетическое описание для e2e.",
        evidence_frames: [{ timestamp_sec: 12.4, image_base64: svg("A") }],
        type_id: "V99",
        confidence: 0.66,
        t_start: 10,
        t_end: 15,
        rule: { document: "Синтетический документ", clause: "п. 0", url: "https://adilet.zan.kz/rus", quote: "синтетическая цитата", verified: false },
      },
    ],
    progress: 1,
    stage: "done",
    timeline: [
      [0, 0.1],
      [12.4, 0.8],
      [30, 0.1],
    ],
    meta: { version: "e2e", duration_sec: 30, timings: { total: 3 } },
  };
}

const pw = await loadPlaywright();
const { server, url } = await startPreview(4182);
const browser = await launch(pw);
const failures = [];
const step = async (name, fn) => {
  try {
    await fn();
    console.log("✓", name);
  } catch (e) {
    failures.push(name);
    console.error("✗", name, "\n ", e.message);
  }
};

try {
  // ---------- сценарий 1: полный путь ----------
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 860 } });
  const page = await ctx.newPage();
  let loggedIn = false;
  let polls = 0;
  let sawKeyInUrl = false;
  page.on("request", (r) => {
    if (r.url().includes(KEY)) sawKeyInUrl = true;
  });
  await page.route("**/ui/v1/**", async (route) => {
    const req = route.request();
    const u = new URL(req.url());
    const p = u.pathname;
    if (p.endsWith("/ui/v1/session")) {
      if (req.method() === "POST") {
        const ok = req.headers()["x-api-key"] === KEY;
        loggedIn = ok;
        return route.fulfill({
          status: ok ? 200 : 401,
          contentType: "application/json",
          headers: ok ? { "set-cookie": "ks_ui=1; HttpOnly; Path=/; SameSite=Strict" } : {},
          body: JSON.stringify(ok ? { authenticated: true, features: { copilot: true } } : { detail: "bad key" }),
        });
      }
      return route.fulfill({ status: loggedIn ? 200 : 401, contentType: "application/json", body: JSON.stringify(loggedIn ? { authenticated: true } : { detail: "no session" }) });
    }
    if (!loggedIn) return route.fulfill({ status: 401, body: "{}" });
    if (p.endsWith("/ui/v1/analyze")) return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ job_id: "e2e-1" }) });
    if (p.endsWith("/ui/v1/jobs/e2e-1")) {
      polls++;
      const body = polls < 3 ? { job_id: "e2e-1", status: "processing", verdict: null, violations: [], progress: polls * 0.3, stage: polls === 1 ? "detect" : "vlm" } : doneJob("e2e-1");
      return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(body) });
    }
    if (p.endsWith("/ask")) return route.fulfill({ status: 501, contentType: "application/json", body: JSON.stringify({ detail: "copilot disabled" }) });
    if (p.endsWith("/video")) return route.fulfill({ status: 404, body: "" });
    return route.fulfill({ status: 404, body: "{}" });
  });

  await page.goto(`${url}#/`);
  await step("нет демо-ленты в реальном режиме", async () => {
    assert.equal(await page.getByText("Синтетическое демо", { exact: false }).count(), 0);
  });
  await step("форма входа при отсутствии сессии", async () => {
    await page.getByText("Вход для проверяющего").waitFor({ timeout: 5000 });
    await page.screenshot({ path: path.join(shots, "11-login.png") });
  });
  await step("неверный ключ → ошибка", async () => {
    await page.getByLabel("Ключ доступа").fill("wrong");
    await page.getByRole("button", { name: "Войти" }).click();
    await page.getByText("Неверный ключ доступа.").waitFor({ timeout: 5000 });
  });
  await step("верный ключ → зона загрузки; ключ не в localStorage", async () => {
    await page.getByLabel("Ключ доступа").fill(KEY);
    await page.getByRole("button", { name: "Войти" }).click();
    await page.getByText("Перетащите видео сюда").waitFor({ timeout: 5000 });
    const stored = await page.evaluate(() => JSON.stringify(Object.assign({}, localStorage)));
    assert.ok(!stored.includes(KEY), "ключ найден в localStorage");
  });
  await step("загрузка → обработка (реальный progress) → результат", async () => {
    await page.locator('input[type="file"]').setInputFiles({ name: "synthetic.mov", mimeType: "video/quicktime", buffer: Buffer.alloc(2048, 1) });
    await page.getByRole("button", { name: "Начать проверку" }).click();
    await page.getByText("Идёт анализ записи").waitFor({ timeout: 8000 });
    await page.getByText("Обнаружены признаки возможных нарушений").waitFor({ timeout: 15000 });
    assert.ok(polls >= 3, `опросов: ${polls}`);
  });
  await step("клик по таймкоду → перемотка", async () => {
    await page.locator("#episode-1 header button.btn-brand").click();
    await page.waitForTimeout(300);
    const txt = await page.locator("div.font-mono.text-sm").first().innerText();
    assert.ok(txt.includes("00:12.4"), `время плеера: ${txt}`);
  });
  await step("кадр открывается в лайтбоксе, норма видна", async () => {
    await page.locator("article button[aria-label^='Открыть кадр']").first().click();
    await page.getByRole("dialog", { name: "Кадр-доказательство" }).waitFor({ timeout: 3000 });
    await page.keyboard.press("Escape");
    await page.getByText("Синтетический документ").waitFor({ timeout: 3000 });
    await page.getByText("цитата не сверена").waitFor({ timeout: 3000 });
  });
  await step("Copilot: 501 → панель скрыта", async () => {
    await page.getByRole("button", { name: "Почему это нарушение?" }).click();
    await page.waitForTimeout(800);
    assert.equal(await page.getByText("Copilot проверяющего").count(), 0);
  });
  await step("ключ ни разу не попал в URL запросов", async () => {
    assert.equal(sawKeyInUrl, false);
  });
  await ctx.close();

  // ---------- сценарий 2: failed и неполный анализ не зелёные ----------
  for (const [name, body, expectText] of [
    ["failed", { job_id: "f1", status: "failed", verdict: null, violations: [], error: "синтетическая ошибка" }, "Анализ не завершён"],
    ["partial", { job_id: "f1", status: "done", verdict: "no_violation", violations: [], meta: { partial: true } }, "Анализ неполный"],
    ["clean", { job_id: "f1", status: "done", verdict: "no_violation", violations: [], meta: {} }, "Нарушение не обнаружено"],
  ]) {
    const c = await browser.newContext({ viewport: { width: 1280, height: 860 } });
    const pg = await c.newPage();
    await pg.route("**/ui/v1/**", (route) => {
      const p = new URL(route.request().url()).pathname;
      if (p.endsWith("/session")) return route.fulfill({ status: 200, contentType: "application/json", body: '{"authenticated":true}' });
      if (p.endsWith("/jobs/f1")) return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(body) });
      return route.fulfill({ status: 404, body: "" });
    });
    await pg.goto(`${url}#/job/f1`);
    await step(`${name}: «${expectText}»`, async () => {
      await pg.getByText(expectText, { exact: false }).first().waitFor({ timeout: 6000 });
      if (name !== "clean") assert.equal(await pg.getByText("Нарушение не обнаружено").count(), 0, "зелёная формулировка при неполном анализе");
      await pg.waitForTimeout(600);
      if (name === "failed") await pg.screenshot({ path: path.join(shots, "12-failed.png") });
      if (name === "partial") await pg.screenshot({ path: path.join(shots, "13-partial.png") });
    });
    await c.close();
  }
} finally {
  await browser.close();
  await server.close();
}

if (failures.length) {
  console.error(`\n${failures.length} проверок не прошло`);
  process.exit(1);
}
console.log("\nВсе мок-e2e проверки прошли");

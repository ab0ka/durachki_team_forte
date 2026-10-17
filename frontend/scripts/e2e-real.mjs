// e2e против ЖИВОГО сервера A02 (интегратору): вход → загрузка → обработка → результат → перемотка → кадр → норма.
//   E2E_BASE_URL=http://127.0.0.1:8000/ E2E_KEY=<api key> E2E_VIDEO=path/to/clip.mov node scripts/e2e-real.mjs
// Скриншоты — только если задан E2E_SHOTS=<папка> (с реальными видео не коммитить и никуда не отправлять!).
import assert from "node:assert/strict";
import path from "node:path";
import { launch, loadPlaywright } from "./pw.mjs";

const BASE = process.env.E2E_BASE_URL ?? "http://127.0.0.1:8000/";
const KEY = process.env.E2E_KEY;
const VIDEO = process.env.E2E_VIDEO;
const SHOTS = process.env.E2E_SHOTS;
if (!KEY || !VIDEO) {
  console.error("Нужны E2E_KEY и E2E_VIDEO");
  process.exit(2);
}

const pw = await loadPlaywright();
const browser = await launch(pw);
const context = await browser.newContext({ viewport: { width: 1366, height: 900 } });
const page = await context.newPage();
const consoleErrors = [];
page.on("pageerror", (e) => consoleErrors.push(String(e)));
const statuses = [];
page.on("response", (r) => {
  const u = new URL(r.url());
  if (u.pathname.startsWith("/ui/") || u.pathname.startsWith("/api/")) statuses.push(`${r.request().method()} ${u.pathname} ${r.status()}`);
});

try {
  await page.goto(BASE);
  await page.getByText("Вход для проверяющего").waitFor({ timeout: 10000 });
  await page.getByLabel("Ключ доступа").fill(KEY);
  await page.getByRole("button", { name: "Войти" }).click();
  await page.getByText("Перетащите видео сюда").waitFor({ timeout: 10000 });
  console.log("✓ вход по ключу → cookie-сессия");

  await page.locator('input[type="file"]').setInputFiles(VIDEO);
  const t0 = Date.now();
  await page.getByRole("button", { name: "Начать проверку" }).click();
  await page.waitForFunction(() => location.hash.startsWith("#/job/"), null, { timeout: 120000 });
  console.log(`✓ загрузка (${((Date.now() - t0) / 1000).toFixed(1)} с) → ${page.url().split("#")[1]}`);

  const banner = page.locator("h1").filter({ hasText: /Обнаружены признаки|Нарушение не обнаружено|Анализ не завершён|Анализ неполный/ });
  await banner.first().waitFor({ timeout: 300000 });
  console.log(`✓ результат за ${((Date.now() - t0) / 1000).toFixed(1)} с: «${(await banner.first().innerText()).trim()}»`);

  const episodes = await page.locator("article[id^='episode-']").count();
  console.log(`  эпизодов: ${episodes}`);
  if (episodes > 0) {
    await page.locator("#episode-1 header button.btn-brand").click();
    await page.waitForTimeout(500);
    console.log(`✓ перемотка: ${await page.locator("div.font-mono.text-sm").first().innerText()}`);
    const thumbs = page.locator("#episode-1 button[aria-label^='Открыть кадр']");
    if (await thumbs.count()) {
      await thumbs.first().click();
      await page.getByRole("dialog", { name: "Кадр-доказательство" }).waitFor({ timeout: 5000 });
      await page.keyboard.press("Escape");
      console.log("✓ кадр-доказательство открывается");
    }
    console.log(`  норм в эпизоде 1: ${await page.locator("#episode-1 a[href*='adilet']").count()}`);
  }
  const video = page.locator("video");
  if (await video.count()) {
    const ready = await video.evaluate((v) => new Promise((res) => (v.readyState >= 1 ? res(v.readyState) : (v.onloadedmetadata = () => res(v.readyState), v.onerror = () => res(-1), setTimeout(() => res(v.readyState), 8000)))));
    console.log(ready >= 1 ? `✓ только что загруженное видео играет локально (readyState=${ready})` : "  локальное видео не декодировано браузером (кодек?)");
  }
  // Повторное открытие задачи (без локального файла) → плеер берёт защищённый cookie-маршрут /ui/v1/jobs/{id}/video
  const jobUrl = page.url();
  const p2 = await context.newPage();
  const videoResp = p2.waitForResponse((r) => r.url().includes("/video"), { timeout: 15000 }).catch(() => null);
  await p2.goto(jobUrl);
  const vr = await videoResp;
  console.log(vr ? `✓ видео-маршрут: ${new URL(vr.url()).pathname} → HTTP ${vr.status()} (${vr.headers()["content-type"] ?? "?"})` : "✗ запроса видео не было");
  const v2 = p2.locator("video");
  if (await v2.count()) {
    const ready = await v2.evaluate((v) => new Promise((res) => (v.readyState >= 1 ? res(v.readyState) : (v.onloadedmetadata = () => res(v.readyState), v.onerror = () => res(-1), setTimeout(() => res(v.readyState), 8000)))));
    console.log(ready >= 1 ? `✓ видео с сервера играет (readyState=${ready})` : "  видео с сервера не декодировано браузером (кодек?)");
  }
  await p2.close();
  if (SHOTS) await page.screenshot({ path: path.join(SHOTS, "e2e-real-result.png"), fullPage: true });
  assert.equal(consoleErrors.length, 0, `ошибки JS: ${consoleErrors.join("; ")}`);
  console.log("✓ без ошибок JS");
} catch (e) {
  console.error("✗", e.message);
  process.exitCode = 1;
} finally {
  console.log("запросы API:", [...new Set(statuses)].join(" | "));
  await browser.close();
}

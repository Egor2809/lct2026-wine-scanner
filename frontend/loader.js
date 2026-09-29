/* ═══════════════════════════════════════════════════════════════
   Лоадер «evolve» — перенос boot-экрана с evolve-scripts.online
   (React-компонент Boot) на чистый JS.

   Как в оригинале, пропускается любой клавишей или касанием, при
   prefers-reduced-motion не показывается вовсе. В отличие от оригинала
   показывается при каждой загрузке, а не раз за сессию. Строки лога — не выдуманные: это ответ /health
   (каталог, индекс, эмбеддер, OCR), с которым поднялся сервис.

   По завершении: .frame получает is-ready (запускает появление
   первого экрана), body — loader-gone, документ — событие loader:gone.
   ═══════════════════════════════════════════════════════════════ */

(function () {
  "use strict";

  const root = document.getElementById("page-loader");

  function reveal() { document.querySelector(".frame")?.classList.add("is-ready"); }

  function gone() {
    document.body.classList.remove("loading");
    document.body.classList.add("loader-gone");
    document.dispatchEvent(new CustomEvent("loader:gone"));
  }

  /* Решение «показывать ли» принято ещё в <head> — до первой отрисовки */
  if (!root || document.documentElement.classList.contains("boot-skip")) {
    root?.remove();
    gone();
    reveal();
    return;
  }

  /* ── Тайминги оригинала ── */
  const FIRST_STEP = 250;   // первая строка лога
  const STEP       = 150;   // шаг между строками
  const HOLD       = 750;   // выдержка после последней строки
  const REVEAL_AT  = 200;   // шторки начинают разъезжаться
  const LEAVE_MS   = 860;   // полный уход
  const HEALTH_MS  = 1200;  // дольше /health не ждём

  const MARK = "M8225 6969 c170 -20 360 -102 476 -207 l50 -45 22 20 c62 57 102 74 173 74 75 0 128 -19 177 -66 47 -45 39 -65 -80 -189 -229 -236 -700 -694 -732 -711 -50 -25 -127 -17 -190 22 -68 43 -112 57 -147 49 -28 -7 -234 -200 -234 -219 0 -5 16 -24 35 -41 93 -86 284 -139 420 -118 91 14 102 17 155 41 55 24 138 82 224 155 89 75 109 84 182 80 118 -7 174 -50 155 -119 -21 -79 -183 -255 -314 -344 -231 -157 -551 -181 -822 -63 -64 27 -243 140 -264 166 -11 13 -19 11 -64 -23 -67 -49 -116 -65 -173 -56 -68 10 -107 30 -147 75 -36 39 -37 42 -23 68 8 15 44 56 81 92 95 93 129 138 117 153 -24 28 -55 185 -59 302 -11 266 75 477 268 660 113 108 185 151 339 203 141 48 232 58 375 41 M8075 6654 c-153 -19 -266 -75 -361 -178 -76 -82 -120 -161 -140 -255 -14 -68 -19 -221 -7 -221 4 0 85 78 180 172 224 223 245 231 383 149 98 -59 110 -62 153 -43 38 18 227 193 227 211 0 16 -100 91 -155 116 -84 38 -201 59 -280 49";

  /* ── Разметка ── */

  /* Сетка 31×13: задержка и яркость растут от центра, каждая ~десятая
     точка — акцентная (тот же хеш, что в оригинале) */
  function gridCells() {
    const far = Math.hypot(15, 6);
    let html = "";
    for (let row = 0; row < 13; row++) {
      for (let col = 0; col < 31; col++) {
        const dist = Math.hypot(col - 15, (row - 6) * 1.6);
        const k = 31 * row + col;
        const accent = (0x9e3779b1 * k) % 97 < 9;
        const fade = Math.max(0, 1 - dist / (1.25 * far));
        const peak = (accent ? 1 : .55) * (.35 + .65 * fade);
        const rest = ((accent ? .5 : .14) * fade).toFixed(3);
        html += `<span class="boot__cell${accent ? " boot__cell--accent" : ""}" ` +
                `style="--d:${120 + Math.round(48 * dist)}ms;--peak:${peak};--rest:${rest}"></span>`;
      }
    }
    return html;
  }

  const coarse = matchMedia("(pointer: coarse)").matches;
  const word = "evolve".split("").map((ch, i) =>
    `<span class="boot__letter" style="--d:${700 + 55 * i}ms">${ch}</span>`).join("");

  root.className = "boot";
  root.setAttribute("aria-hidden", "true");
  root.innerHTML = `
    <div class="boot__half boot__half--top"></div>
    <div class="boot__half boot__half--bottom"></div>
    <div class="boot__seam"></div>
    <div class="boot__stage">
      <div class="boot__emblem">
        <div class="boot__grid">${gridCells()}</div>
        <span class="boot__glow"></span>
        <span class="boot__shade"></span>
        <svg class="boot__mark" viewBox="603 422 423 365">
          <g transform="translate(810 608) scale(2) translate(-810 -608)">
            <g transform="translate(0,1216) scale(0.1,-0.1)">
              <path class="boot__trace" d="${MARK}" pathLength="1" fill-rule="evenodd"
                    fill="currentColor" stroke="currentColor" stroke-width="34" stroke-linejoin="round"/>
            </g>
          </g>
        </svg>
      </div>
      <div class="boot__word">${word}</div>
      <div class="boot__progress">
        <div class="boot__meta">
          <span class="boot__label boot__count">00/00</span>
          <span class="boot__label boot__version">team</span>
        </div>
        <div class="boot__bar"></div>
        <div class="boot__log"></div>
      </div>
      <span class="boot__label boot__hint">${coarse ? "коснись, чтобы пропустить" : "нажми любую клавишу"}</span>
    </div>`;

  const count = root.querySelector(".boot__count");
  const bar = root.querySelector(".boot__bar");
  const log = root.querySelector(".boot__log");

  /* ── Строки лога из /health ── */

  function plural(n, one, few, many) {
    const m10 = n % 10, m100 = n % 100;
    if (m10 === 1 && m100 !== 11) return one;
    if (m10 >= 2 && m10 <= 4 && (m100 < 12 || m100 > 14)) return few;
    return many;
  }

  function linesFrom(h) {
    const num = (n) => Number(n).toLocaleString("ru-RU");
    const out = [["health", h.status || "—", h.status === "ok"]];
    if (h.catalog_size != null) {
      out.push(["catalog", `${num(h.catalog_size)} ${plural(h.catalog_size, "вино", "вина", "вин")}`, h.catalog_size > 0]);
    }
    if (h.index_size != null) {
      out.push(["index", `${num(h.index_size)} ${plural(h.index_size, "эталон", "эталона", "эталонов")}`, h.index_size > 0]);
    }
    if (h.embedder) out.push(["embedder", h.embedder.replace(/^[a-z]+:google\//, ""), true]);
    if (h.ocr) out.push(["ocr", h.ocr.split("|").join(" · "), true]);
    if (h.error) out.push(["error", h.error, false]);
    return out;
  }

  function fetchHealth() {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), HEALTH_MS);
    return fetch("/health", { signal: ctl.signal, cache: "no-store" })
      .then((r) => r.json())
      .then(linesFrom)
      .catch(() => [["api", "нет ответа", false]])
      .finally(() => clearTimeout(timer));
  }

  /* ── Ход ── */

  const timers = [];
  const later = (fn, ms) => timers.push(setTimeout(fn, ms));
  const pad = (n) => String(n).padStart(2, "0");
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  let lines = [];
  let leaving = false;

  function step(p) {
    const total = lines.length;
    const done = p >= total;
    count.textContent = done ? "готово" : `${pad(p)}/${pad(total)}`;
    count.classList.toggle("is-ready", done);
    bar.classList.toggle("is-ready", done);
    [...bar.children].forEach((seg, i) => {
      seg.className = "boot__seg" + (i < p ? " is-done" : "") + (i === p ? " is-live" : "");
    });
    if (p > 0) {
      const [name, value, ok] = lines[p - 1];
      log.innerHTML = `<span class="boot__line"><span class="boot__n">${pad(p)}</span>` +
        `<span class="boot__ok${ok ? "" : " boot__ok--err"}">${ok ? "ok" : "err"}</span>` +
        `<span class="boot__text">${esc(name)}  ${esc(value)}</span></span>`;
    }
  }

  function leave() {
    if (leaving) return;
    leaving = true;
    timers.forEach(clearTimeout);
    root.classList.add("is-leaving");
    document.body.classList.remove("loading");
    setTimeout(reveal, REVEAL_AT);
    setTimeout(() => { root.remove(); gone(); }, LEAVE_MS);
  }

  const started = performance.now();

  fetchHealth().then((got) => {
    if (leaving) return;
    lines = got;
    bar.innerHTML = lines.map(() => '<span class="boot__seg"></span>').join("");
    step(0);
    const offset = Math.max(0, FIRST_STEP - (performance.now() - started));
    lines.forEach((_, i) => later(() => step(i + 1), offset + STEP * i));
    later(leave, offset + STEP * (lines.length - 1) + HOLD);
  });

  /* Пропуск — как в оригинале: любая клавиша или касание */
  addEventListener("keydown", leave, { once: true });
  addEventListener("pointerdown", leave, { once: true });
})();

/* ═══════════════════════════════════════════════════════════════
   Сканер вин «Своё Вино»

   Все данные о винах — только с бэкенда:
     POST /scan          фото -> { slug, confident, card, candidates, analogs, analogs_profile, ... }
     GET  /wines         каталог: q, style, region, winery, sparkling, limit, offset
     GET  /wines/{slug}  карточка + аналоги
     GET  /sommelier     блюдо -> вина с «почему подходит»: food, region, sparkling, seed

   Метрики (confidence / F1) в интерфейс не выводятся — требование ТЗ:
   «в UI добавлять не нужно, можно показать в API».
   Избранное и история — localStorage, хранят снимок реальной карточки.
   ═══════════════════════════════════════════════════════════════ */

const PAGE_SIZE = 24;
const TIMEOUT_MS = 15000;

/* ═══════════════ API ═══════════════ */

/* kind: timeout | network | http (тогда status — код ответа). */
class ApiError extends Error {
  constructor(kind, status = 0) {
    super(status ? `HTTP ${status}` : kind);
    this.kind = kind;
    this.status = status;
  }
}

/* Без таймаута запрос на слабой сети висит минутами, а экран — на «Распознаём». */
function timeoutSignal(ms) {
  if (AbortSignal.timeout) return AbortSignal.timeout(ms);
  const ctrl = new AbortController();
  setTimeout(() => ctrl.abort(), ms);
  return ctrl.signal;
}

async function getJSON(url, init = {}) {
  let res;
  try {
    res = await fetch(url, { ...init, signal: timeoutSignal(TIMEOUT_MS) });
  } catch (err) {
    /* Свой abort у нас только по таймауту: AbortError — тоже он (старые Safari). */
    const late = err?.name === "TimeoutError" || err?.name === "AbortError";
    throw new ApiError(late ? "timeout" : "network");
  }
  if (!res.ok) throw new ApiError("http", res.status);
  return res.json();
}

const query = (params) => {
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== "" && v != null) qs.set(k, v);
  }
  return qs;
};

const api = {
  scan(file) {
    const body = new FormData();
    body.append("image", file);
    return getJSON("/scan", { method: "POST", body });
  },
  wine: (slug) => getJSON(`/wines/${encodeURIComponent(slug)}`),
  wines: (params) => getJSON(`/wines?${query(params)}`),
  sommelier: (params) => getJSON(`/sommelier?${query(params)}`),
};

/* ═══════════════ ФОТО ПЕРЕД ОТПРАВКОЙ ═══════════════
   Снимок телефона — 3–12 МБ: на мобильной сети это секунды до ответа.
   Крупный файл и HEIC пережимаются в JPEG ~2048 px по длинной стороне
   (модели хватает с запасом: этикетка всё равно сжимается до 384).
   Любая ошибка — уходит оригинал: лучше медленно, чем никак. */

const UPLOAD_MAX_SIDE = 2048;
const UPLOAD_SHRINK_FROM = 2 * 1024 * 1024;
const isHeic = (file) => /hei[cf]/i.test(file.type) || /\.hei[cf]$/i.test(file.name || "");

async function prepareUpload(file) {
  const heic = isHeic(file);
  if (file.size <= UPLOAD_SHRINK_FROM && !heic) return file;
  try {
    /* from-image: поворот по EXIF применяется к пикселям — в JPEG из canvas
       EXIF уже не будет, и без этого снимок «на боку» ушёл бы на боку. */
    const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
    const scale = Math.min(1, UPLOAD_MAX_SIDE / Math.max(bitmap.width, bitmap.height));
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(bitmap.width * scale);
    canvas.height = Math.round(bitmap.height * scale);
    canvas.getContext("2d").drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    bitmap.close?.();
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.92));
    if (!blob || (!heic && blob.size >= file.size)) return file;
    const name = (file.name || "photo").replace(/\.[^.]+$/, "") + ".jpg";
    return new File([blob], name, { type: "image/jpeg" });
  } catch (err) {
    console.warn("Фото уходит без уменьшения:", err);
    return file;
  }
}

/* ═══════════════ ХРАНИЛИЩЕ ═══════════════ */

const KEY = "svoe-vino";

const store = {
  read() {
    try { return JSON.parse(localStorage.getItem(KEY)) || {}; }
    catch { return {}; }
  },
  write(patch) {
    const next = { ...store.read(), ...patch };
    try { localStorage.setItem(KEY, JSON.stringify(next)); } catch {}
    return next;
  },
};

(() => {
  const { user, taste, ...rest } = store.read();
  if (user !== undefined || taste !== undefined) {
    try { localStorage.setItem(KEY, JSON.stringify(rest)); } catch {}
  }
})();

const entries = (name) => (store.read()[name] || []).filter((x) => x && typeof x === "object" && x.slug);
const getLikes = () => entries("likes");
const getHistory = () => entries("history");
const snapshot = ({ slug, card }) => ({ slug, card });

function toggleLike(item) {
  const likes = getLikes();
  const on = likes.some((x) => x.slug === item.slug);
  store.write({ likes: on ? likes.filter((x) => x.slug !== item.slug) : [snapshot(item), ...likes] });
  return !on;
}

function pushHistory(item) {
  const next = [snapshot(item), ...getHistory().filter((x) => x.slug !== item.slug)].slice(0, 12);
  store.write({ history: next });
}

/* ═══════════════ РОУТЕР ═══════════════ */

const PAGES = ["scan", "catalog", "sommelier", "saved", "winery"];
const LABELS = { scan: "Сканер", catalog: "Каталог", sommelier: "Сомелье",
                 saved: "Мои вина", winery: "Винодельня" };

const app = document.getElementById("app");
const stage = document.getElementById("stage");
const sectionLabel = document.getElementById("section-label");

let previousPage = "scan";

function currentRoute() {
  const raw = location.hash.replace(/^#\/?/, "");
  const cut = raw.indexOf("/");
  const name = cut < 0 ? raw : raw.slice(0, cut);
  let arg = cut < 0 ? "" : raw.slice(cut + 1);
  try { arg = decodeURIComponent(arg); } catch {}
  return { name: PAGES.includes(name) ? name : "scan", arg };
}

function route() {
  const { name, arg } = currentRoute();

  app.querySelectorAll(".page").forEach((p) => {
    p.classList.toggle("is-active", p.dataset.page === name);
  });
  document.querySelectorAll(".tab").forEach((t) => {
    const on = t.dataset.tab === name;
    t.classList.toggle("is-active", on);
    if (on) t.setAttribute("aria-current", "page"); else t.removeAttribute("aria-current");
  });

  sectionLabel.textContent = LABELS[name];
  /* Таббар на винодельне прячет CSS и только на мобильном:
     на десктопе это верхнее меню, без него со страницы не уйти. */
  document.body.dataset.page = name;

  if (name === "catalog")   openCatalog();
  if (name === "sommelier") startSommelier();
  if (name === "saved")     renderSaved();
  if (name === "winery")    renderWinery(arg);

  window.scrollTo({ top: 0 });
}

window.addEventListener("hashchange", (e) => {
  const from = new URL(e.oldURL).hash.replace(/^#\/?/, "").split("/")[0];
  if (PAGES.includes(from) && from !== "winery") previousPage = from;
  route();
});

document.querySelectorAll("[data-back]").forEach((b) => {
  b.addEventListener("click", () => { location.hash = "#/" + previousPage; });
});

/* ═══════════════ СКАНЕР ═══════════════ */

const els = {
  preview:    document.getElementById("scan-preview"),
  bottle:     document.getElementById("card-bottle"),
  winery:     document.getElementById("card-winery"),
  eyebrow:    document.getElementById("card-eyebrow"),
  byline:     document.getElementById("card-byline"),
  title:      document.getElementById("card-title"),
  specs:      document.getElementById("card-specs"),
  note:       document.getElementById("card-note"),
  pairing:    document.getElementById("card-pairing"),
  pairList:   document.getElementById("pairing-list"),
  pairNote:   document.getElementById("pairing-note"),
  rating:     document.getElementById("card-rating"),
  ratingValue: document.getElementById("rating-value"),
  link:       document.getElementById("card-link"),
  like:       document.getElementById("card-like"),
  likeLabel:  document.getElementById("card-like-label"),
  analogs:    document.getElementById("card-analogs"),
  analogRail: document.getElementById("analogs-rail"),
  choices:    document.getElementById("choices"),
  choiceAnalogs: document.getElementById("choice-analogs"),
  choiceRail: document.getElementById("choice-rail"),
  choiceSub:  document.getElementById("choice-analogs-sub"),
  errEyebrow: document.getElementById("error-eyebrow"),
  errTitle:   document.getElementById("error-title"),
  errText:    document.getElementById("error-text"),
  errRetry:   document.getElementById("error-retry"),
  choiceTitle: document.getElementById("choice-title"),
  live:       document.getElementById("scan-live"),
};

let current = null;
let retry = null;
let requestSeq = 0;

/* Отменяет ответ скана или карточки, который ещё в пути. */
function cancelPending() {
  requestSeq++;
}

/* Заголовок, на который встаёт фокус после смены экрана, и фраза для экранного
   диктора. Кнопка, с которой начинали, к этому моменту уже скрыта — без переноса
   фокус улетел бы в <body>. */
const ARRIVAL = {
  result: () => [els.title, els.title.textContent],
  choice: () => [els.choiceTitle, "Точного совпадения нет. Выберите вино из списка"],
  empty:  () => [els.errTitle, `${els.errTitle.textContent}. ${els.errText.textContent}`],
};

function setState(name) {
  stage.dataset.state = name;
  const onScan = currentRoute().name === "scan";
  if (onScan) window.scrollTo({ top: 0, behavior: "smooth" });

  const arrival = ARRIVAL[name]?.();
  els.live.textContent = arrival ? arrival[1] : "";
  if (arrival && onScan) arrival[0].focus({ preventScroll: true });
}

const inputCamera  = document.getElementById("file-camera");
const inputGallery = document.getElementById("file-gallery");

document.getElementById("btn-camera").addEventListener("click", () => inputCamera.click());
document.getElementById("btn-gallery").addEventListener("click", () => inputGallery.click());

[inputCamera, inputGallery].forEach((input) => {
  input.addEventListener("change", (e) => {
    const file = e.target.files?.[0];
    if (file) scan(file);
    e.target.value = "";
  });
});

document.querySelectorAll("[data-restart]").forEach((btn) => {
  btn.addEventListener("click", () => { cancelPending(); location.hash = "#/scan"; setState("idle"); });
});

els.errRetry.addEventListener("click", () => retry && retry());

let previewUrl = null;

async function scan(file) {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  els.preview.src = previewUrl;
  cancelPending();
  setState("scanning");
  const started = performance.now();
  const my = requestSeq;

  try {
    const upload = await prepareUpload(file);
    if (my !== requestSeq) return;
    /* HEIC показывает только Safari: в остальных превью — уже из JPEG. */
    if (upload !== file && isHeic(file)) {
      URL.revokeObjectURL(previewUrl);
      previewUrl = URL.createObjectURL(upload);
      els.preview.src = previewUrl;
    }
    const data = await api.scan(upload);
    await holdMinimum(started);
    if (my !== requestSeq) return;
    if (data.error) {
      showError("Фото не прочитано", "Не получилось разобрать снимок",
                "Попробуйте другое фото: этикетка целиком, без сильного размытия.", () => scan(file));
    } else if (data.confident === false) {
      renderChoices(data);
    } else {
      showCard(data);
    }
  } catch (err) {
    console.warn("Сканирование не удалось:", err);
    await holdMinimum(started);
    if (my !== requestSeq) return;
    const [eyebrow, title, text, again] = scanError(err);
    showError(eyebrow, title, text, again ? () => scan(file) : null);
  }
}

/* -> [подпись, заголовок, текст, есть ли смысл повторять то же фото]. */
function scanError(err) {
  if (err?.kind === "timeout") {
    return ["Долгий ответ", "Сервис не ответил вовремя",
            `Ждали ${TIMEOUT_MS / 1000} секунд — похоже, слабая связь. Повторите — фото сохранено.`, true];
  }
  if (err?.status === 413) {
    return ["Слишком большой файл", "Фото не поместилось в лимит",
            "Выберите снимок поменьше или сфотографируйте этикетку ещё раз.", false];
  }
  if (err?.status === 503) {
    return ["Сервис запускается", "Модель ещё загружается",
            "Обычно это меньше минуты. Повторите чуть позже — фото сохранено.", true];
  }
  if (err?.kind === "http") {
    return ["Ошибка сервиса", "Сервис не смог ответить",
            `Код ответа ${err.status}. Повторите — фото сохранено.`, true];
  }
  return ["Нет связи", "Сервис недоступен",
          "Проверьте подключение и повторите — фото сохранено.", true];
}

/* Не даёт интерфейсу моргнуть на мгновенном ответе. Порог низкий:
   скорость — часть оценки, тормозить искусственно нельзя. */
function holdMinimum(started, minMs = 400) {
  const left = minMs - (performance.now() - started);
  return left > 0 ? new Promise((r) => setTimeout(r, left)) : Promise.resolve();
}

function showError(eyebrow, title, text, again) {
  els.errEyebrow.textContent = eyebrow;
  els.errTitle.textContent = title;
  els.errText.textContent = text;
  retry = again;
  els.errRetry.hidden = !again;
  setState("empty");
}

function showCard(data) {
  renderCard(data);
  pushHistory(data);
}

/* Переход по клику: сразу на вкладку сканера, карточка приходит следом.
   Результат скана, наоборот, вкладку не переключает — пользователь мог уйти. */
async function openWine(slug) {
  cancelPending();
  const my = requestSeq;
  location.hash = "#/scan";
  try {
    const data = await api.wine(slug);
    if (my === requestSeq) showCard(data);
  } catch (err) {
    console.warn("Карточка недоступна:", err);
    if (my === requestSeq) {
      showError("Нет связи", "Карточка не загрузилась", "Повторите через пару секунд.", () => openWine(slug));
    }
  }
}

function renderCard(data) {
  const c = data.card || {};
  current = data;

  setImage(els.bottle, c);
  els.bottle.alt = [c.title, c.winery].filter(Boolean).join(", ");
  const eyebrow = [c.style, c.region].filter(Boolean).join(" · ");
  els.eyebrow.textContent = eyebrow;
  els.eyebrow.hidden = !eyebrow;
  els.winery.textContent = c.winery || "";
  els.byline.hidden = !c.winery;
  els.title.textContent = c.title || data.slug;

  const rows = [["Сорт", c.grape], ["Цвет", c.color], ["Урожай", c.vintage]].filter(([, v]) => v);
  els.specs.innerHTML = rows.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join("");

  els.note.textContent = c.note || "";
  els.note.hidden = !c.note;

  const rating = c.rating;
  els.rating.hidden = !rating?.value;
  if (rating?.value) {
    els.ratingValue.textContent = rating.value;
    els.rating.title = rating.note || "";
  }

  const pairs = c.pairing || [];
  els.pairing.hidden = pairs.length === 0;
  els.pairList.innerHTML = pairs.map((p) => `<li>${esc(p)}</li>`).join("");
  /* Откуда гастрономия: в дампе её нет, и гость должен видеть, где совет
     винодельни, а где подсказка сервиса по сорту и стилю. */
  const pairNote = c.pairing_source === "description" ? "Совет винодельни из описания"
    : c.pairing_source === "rules" ? ["Подсказка сервиса", c.pairing_basis].filter(Boolean).join(" · ")
    : "";
  els.pairNote.textContent = pairNote;
  els.pairNote.hidden = !pairNote;

  els.link.hidden = !c.url;
  if (c.url) els.link.href = c.url;

  syncLike();

  const analogs = data.analogs || [];
  els.analogs.hidden = analogs.length === 0;
  renderRail(els.analogRail, analogs);
  setState("result");
}

els.winery.addEventListener("click", () => {
  const name = current?.card?.winery;
  if (name) location.hash = `#/winery/${encodeURIComponent(name)}`;
});

els.like.addEventListener("click", () => {
  if (!current) return;
  toggleLike(current);
  syncLike();
});

function syncLike() {
  const liked = current ? getLikes().some((x) => x.slug === current.slug) : false;
  els.like.classList.toggle("is-on", liked);
  els.like.setAttribute("aria-pressed", String(liked));
  els.likeLabel.textContent = liked ? "В избранном" : "В избранное";
}

function renderChoices(data) {
  const candidates = (data.candidates || []).slice(0, 5);
  els.choices.innerHTML = candidates.map((item, i) => {
    const c = item.card || {};
    const sub = [c.style, c.region, c.vintage].filter(Boolean).join(" · ");
    return `<li>
      <button class="choice" type="button" data-i="${i}" style="animation-delay:${0.06 * i}s">
        <span class="choice__num" aria-hidden="true">${i + 1}</span>
        <span class="choice__meta">
          <span class="choice__winery">${esc(c.winery)}</span>
          <span class="choice__name">${esc(c.title || item.slug)}</span>
          ${sub ? `<span class="choice__sub">${esc(sub)}</span>` : ""}
        </span>
        <span class="choice__plate"><img class="choice__thumb" ${imgAttrs(c)} alt="" width="56" height="78"></span>
      </button></li>`;
  }).join("");

  els.choices.querySelectorAll(".choice").forEach((btn) => {
    btn.addEventListener("click", () => openWine(candidates[+btn.dataset.i].slug));
  });

  const analogs = data.analogs || [];
  els.choiceAnalogs.hidden = analogs.length === 0;
  const basis = profileLine(data.analogs_profile);
  els.choiceSub.textContent = basis;
  els.choiceSub.hidden = !basis;
  renderRail(els.choiceRail, analogs);
  setState("choice");
}

/* Вина нет в каталоге: по какому профилю подобраны аналоги и откуда он. */
const PROFILE_FIELDS = { grapes: "сорт", style: "стиль", sugar: "сахар", region: "регион", sparkling: "игристость" };

function profileLine(p) {
  if (!p) return "";
  const parts = [(p.style || "").toLowerCase(), p.sparkling ? "игристое" : "", p.sugar,
                 (p.grapes || []).join(", "), p.region].filter(Boolean);
  if (!parts.length) return "";
  const read = (p.from_label || []).map((k) => PROFILE_FIELDS[k]).filter(Boolean);
  const source = read.length ? `с этикетки прочитаны ${read.join(", ")}` : "профиль собран по похожим позициям";
  return `Подбор по профилю: ${parts.join(" · ")} — ${source}`;
}

/* ═══════════════ СПИСКИ ВИН ═══════════════ */

/* Под аналогом — причины совпадения, под вином сомелье — почему подходит. */
function renderRail(rail, items) {
  rail.innerHTML = items.map((item) => {
    const c = item.card || {};
    const why = item.why || (item.reasons || []).join(" · ");
    return `<button class="analog" type="button" data-slug="${esc(item.slug)}">
      <span class="analog__plate"><img ${imgAttrs(c)} alt="" width="120" height="140" loading="lazy" decoding="async"></span>
      <span class="analog__winery">${esc(c.winery)}</span>
      <span class="analog__name">${esc(c.title || item.slug)}</span>
      ${why ? `<span class="analog__why">${esc(why)}</span>` : ""}
    </button>`;
  }).join("");
}

/* Рейтинг — демо-значение (в выгрузке его нет): пометка видна всегда. */
function ratingBadge(c, cls) {
  const r = c.rating;
  if (!r?.value) return "";
  return `<span class="${cls}" title="${esc(r.note || "")}">${esc(r.source || "Рейтинг")} ${esc(r.value)}${r.demo ? ` <em>демо</em>` : ""}</span>`;
}

function tiles(items, offset = 0) {
  return items.map((item, i) => {
    const c = item.card || {};
    const meta = [c.style, c.region].filter(Boolean).join(" · ");
    return `<button class="tile" type="button" data-slug="${esc(item.slug)}" style="animation-delay:${0.03 * ((offset + i) % PAGE_SIZE)}s">
      <span class="tile__plate"><img ${imgAttrs(c)} alt="" width="120" height="148" loading="lazy" decoding="async"></span>
      <span class="tile__winery">${esc(c.winery)}</span>
      <span class="tile__name">${esc(c.title || item.slug)}</span>
      <span class="tile__meta">${esc(meta)}</span>
      ${ratingBadge(c, "tile__rating")}
    </button>`;
  }).join("");
}

/* Клики по любой карточке вина — одним делегатом на приложение. */
app.addEventListener("click", (e) => {
  const btn = e.target.closest(".analog[data-slug], .tile[data-slug]");
  if (btn) openWine(btn.dataset.slug);
});

/* Постраничная загрузка /wines в сетку. Устаревшие ответы (пользователь
   успел поменять фильтр) отбрасываются по номеру запроса. */
function pager({ grid, more, onPage }) {
  let params = {}, offset = 0, seq = 0;

  async function load(reset) {
    const my = ++seq;
    if (reset) offset = 0;
    more.disabled = true;
    try {
      const page = await api.wines({ ...params, limit: PAGE_SIZE, offset });
      if (my !== seq) return;
      if (reset) grid.innerHTML = tiles(page.items);
      else grid.insertAdjacentHTML("beforeend", tiles(page.items, offset));
      offset += page.items.length;
      more.hidden = offset >= page.total;
      onPage?.(page, reset);
    } catch (err) {
      if (my !== seq) return;
      console.warn("Каталог недоступен:", err);
      if (reset) grid.innerHTML = "";
      more.hidden = true;
      onPage?.(null, reset);
    } finally {
      if (my === seq) more.disabled = false;
    }
  }

  more.addEventListener("click", () => load(false));
  return {
    set(next) { params = next; return load(true); },
    clear() { seq++; grid.innerHTML = ""; more.hidden = true; },
  };
}

/* ═══════════════ КАТАЛОГ ═══════════════ */

const catalog = { q: "", style: "", region: "" };
const catalogEmpty = document.getElementById("catalog-empty");
const catalogStyles = document.getElementById("catalog-styles");
const catalogRegions = document.getElementById("catalog-regions");
const catalogQ = document.getElementById("catalog-q");

const catalogPager = pager({
  grid: document.getElementById("catalog-grid"),
  more: document.getElementById("catalog-more"),
  onPage(page, reset) {
    if (!reset) return;
    if (!page) {
      catalogEmpty.textContent = "Каталог сейчас недоступен. Обновите страницу чуть позже.";
      catalogEmpty.hidden = false;
      return;
    }
    catalogEmpty.textContent = "Ничего не нашлось. Попробуйте другое слово.";
    catalogEmpty.hidden = page.total > 0;
    chips(catalogStyles, page.styles, "style", "Все стили");
    chips(catalogRegions, page.regions, "region", "Все регионы");
  },
});

/* Ряд чипов — toolbar: в Tab-порядке одна кнопка (выбранная), по ряду —
   стрелками, Home/End. После перерисовки фокус возвращается на выбранный чип,
   иначе при выборе с клавиатуры он пропадал вместе со старыми кнопками. */
function chips(host, facets, key, allLabel) {
  const active = catalog[key];
  const hadFocus = host.contains(document.activeElement);
  const list = facets.some((f) => f.name === active) || !active
    ? facets : [{ name: active, count: 0 }, ...facets];
  host.innerHTML = [{ name: "", label: allLabel }, ...list].map((f) => {
    const on = f.name === active;
    const label = f.label
      ? esc(f.label)
      : `${esc(f.name)}<span class="filter__count"><span class="visually-hidden">, позиций: </span>${f.count}</span>`;
    return `<button class="filter${on ? " is-active" : ""}" type="button" aria-pressed="${on}"
      tabindex="${on ? 0 : -1}" data-key="${key}" data-value="${esc(f.name)}">${label}</button>`;
  }).join("");
  if (hadFocus) host.querySelector(".filter.is-active")?.focus();
}

const CHIP_KEYS = { ArrowRight: 1, ArrowLeft: -1 };

[catalogStyles, catalogRegions].forEach((host) => {
  host.addEventListener("click", (e) => {
    const btn = e.target.closest(".filter");
    if (!btn) return;
    catalog[btn.dataset.key] = btn.dataset.value;
    catalogPager.set({ ...catalog });
  });

  host.addEventListener("keydown", (e) => {
    const all = [...host.querySelectorAll(".filter")];
    const i = all.indexOf(document.activeElement);
    if (i < 0) return;
    let next;
    if (e.key in CHIP_KEYS) next = (i + CHIP_KEYS[e.key] + all.length) % all.length;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = all.length - 1;
    else return;
    e.preventDefault();
    all[i].tabIndex = -1;
    all[next].tabIndex = 0;
    all[next].focus();
  });
});

let searchTimer = null;
catalogQ.addEventListener("input", () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {
    catalog.q = catalogQ.value.trim();
    catalogPager.set({ ...catalog });
  }, 250);
});

let catalogLoaded = false;
function openCatalog() {
  if (catalogLoaded) return;
  catalogLoaded = true;
  catalogPager.set({ ...catalog });
}

/* ═══════════════ СОМЕЛЬЕ ═══════════════
   Подбирает бэкенд (/sommelier) по той же гастрономии, что в карточке:
   к блюду подходят вина нескольких стилей и сортов, у каждого — «почему».
   Если фильтры вместе ничего не дают, бэкенд ослабляет их и говорит об этом.
   id блюд — как FOODS в src/wine_scanner/enrich/sommelier.py. */

const FOOD = [
  { id: "meat",     label: "Красное мясо, дичь" },
  { id: "poultry",  label: "Птица, паста, овощи" },
  { id: "fish",     label: "Рыба, морепродукты" },
  { id: "cheese",   label: "Сыры, пряная кухня" },
  { id: "aperitif", label: "Аперитив, праздник", sparkling: true },
  { id: "dessert",  label: "Десерты, фрукты" },
];

const sommDialog = document.getElementById("somm-dialog");
const sommStep = document.getElementById("somm-step");
const sommReset = document.getElementById("somm-reset");

let somm = {};
let sommSeq = 0;

sommReset.addEventListener("click", () => startSommelier());

function startSommelier() {
  somm = {};
  sommSeq++;
  sommReset.hidden = true;
  ask(1, "Что будет на столе?", FOOD.map((f) => ({ label: f.label, value: f })), (f) => {
    somm.food = f.id;
    somm.foodLabel = f.label;
    if (f.sparkling) somm.sparkling = true;
    askRegion();
  });
}

async function askRegion() {
  const my = sommSeq;
  let regions = [];
  try {
    const page = await api.sommelier({ food: somm.food, sparkling: somm.sparkling, limit: 1 });
    regions = page.regions.slice(0, 5);
  } catch (err) {
    console.warn("Каталог недоступен:", err);
  }
  if (my !== sommSeq) return;
  const options = [...regions.map((r) => ({ label: r.name, value: r.name })),
                   { label: "Не важно", value: "" }];
  ask(2, "Какой регион ближе?", options, (region) => {
    somm.region = region;
    if (somm.sparkling) return pick();
    ask(3, "Тихое или игристое?", [
      { label: "Тихое", value: false },
      { label: "Игристое", value: true },
      { label: "Не важно", value: null },
    ], (sparkling) => { somm.sparkling = sparkling; pick(); });
  });
}

/* Перерисовка диалога уносит кнопку, на которой стоял фокус; с клавиатуры
   он переезжает на новый шаг, а не в <body>. */
function sommRender(html, focusSelector) {
  const hadFocus = sommDialog.contains(document.activeElement);
  sommDialog.innerHTML = html;
  if (hadFocus) sommDialog.querySelector(focusSelector)?.focus({ preventScroll: true });
}

function ask(n, question, options, onAnswer) {
  sommStep.textContent = `Вопрос ${n} из ${somm.sparkling && n > 1 ? 2 : 3}`;
  sommRender(`
    <p class="somm__question">${esc(question)}</p>
    <div class="somm__options">
      ${options.map((o, i) => `
        <button class="somm__option" type="button" data-i="${i}" style="animation-delay:${0.05 * i}s">${esc(o.label)}</button>`).join("")}
    </div>`, ".somm__option");
  sommDialog.querySelectorAll(".somm__option").forEach((btn) => {
    btn.addEventListener("click", () => onAnswer(options[+btn.dataset.i].value));
  });
}

async function pick() {
  const my = sommSeq;
  sommStep.textContent = "Подборка";
  sommRender(`<p class="somm__question" tabindex="-1">Подбираем…</p>`, ".somm__question");

  let page = null;
  try {
    page = await api.sommelier({ food: somm.food, region: somm.region, sparkling: somm.sparkling,
                                 seed: Math.floor(Math.random() * 1e9) });
  } catch (err) {
    console.warn("Каталог недоступен:", err);
  }
  if (my !== sommSeq) return;
  sommReset.hidden = false;

  if (!page?.items?.length) {
    sommRender(`<p class="somm__question" tabindex="-1">Каталог сейчас недоступен</p>
      <p class="list-empty">Попробуйте подобрать ещё раз через минуту.</p>`, ".somm__question");
    return;
  }

  const used = page.used || {};
  const why = [somm.foodLabel, used.region,
               used.sparkling === true ? "игристое" : used.sparkling === false ? "тихое" : ""]
    .filter(Boolean).join(" · ");

  sommRender(`
    <p class="somm__question somm__question--result" tabindex="-1">Вина разных хозяйств под ваш запрос</p>
    <p class="list-empty">${esc(why)}${page.relaxed ? " — точных совпадений нет, условия ослаблены" : ""}</p>
    <div class="analogs__rail" id="somm-rail"></div>`, ".somm__question");
  renderRail(document.getElementById("somm-rail"), page.items);
}

/* ═══════════════ ВИНОДЕЛЬНЯ ═══════════════ */

const wineryName = document.getElementById("winery-name");
const wineryRegion = document.getElementById("winery-region");
const winerySpecs = document.getElementById("winery-specs");

const wineryPager = pager({
  grid: document.getElementById("winery-grid"),
  more: document.getElementById("winery-more"),
  onPage(page, reset) {
    if (!reset) return;
    if (!page) {
      wineryRegion.textContent = "";
      winerySpecs.innerHTML = `<div><dt>Каталог</dt><dd>недоступен</dd></div>`;
      return;
    }
    const regions = page.regions.map((r) => r.name);
    wineryRegion.textContent = regions.join(", ");
    const rows = [
      ["В каталоге", `${page.total} ${plural(page.total, "позиция", "позиции", "позиций")}`],
      ["Стили", page.styles.map((s) => s.name).join(", ")],
    ].filter(([, v]) => v);
    winerySpecs.innerHTML = rows.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join("");
  },
});

function renderWinery(name) {
  wineryName.textContent = name || "Винодельня";
  wineryRegion.textContent = "";
  winerySpecs.innerHTML = "";
  wineryPager.clear();
  if (name) wineryPager.set({ winery: name });
}

/* ═══════════════ МОИ ВИНА ═══════════════ */

function renderSaved() {
  const likes = getLikes();
  document.getElementById("liked-empty").hidden = likes.length > 0;
  renderRail(document.getElementById("liked-rail"), likes);

  const history = getHistory();
  document.getElementById("history-empty").hidden = history.length > 0;
  renderRail(document.getElementById("history-rail"), history);
}

/* ═══════════════ УТИЛИТЫ ═══════════════ */

function esc(v) {
  return String(v ?? "").replace(/[&<>"']/g, (ch) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
}

function plural(n, one, few, many) {
  const m10 = n % 10, m100 = n % 100;
  if (m10 === 1 && m100 !== 11) return one;
  if (m10 >= 2 && m10 <= 4 && (m100 < 12 || m100 > 14)) return few;
  return many;
}

/* Фото каталога есть не у всех позиций, часть файлов может не отдаться —
   тогда силуэт бутылки в цвете стиля (подмена в обработчике error ниже). */
function imgAttrs(c) {
  return `src="${esc(c.image || bottleArt(c.style))}" data-style="${esc(c.style)}"`;
}

function setImage(img, c) {
  img.dataset.style = c.style || "";
  img.src = c.image || bottleArt(c.style);
}

document.addEventListener("error", (e) => {
  const img = e.target;
  if (img.tagName !== "IMG" || img.dataset.style === undefined || img.src.startsWith("data:")) return;
  img.src = bottleArt(img.dataset.style);
}, true);

function bottleArt(style = "") {
  const s = (style || "").toLowerCase();
  const glass =
    s.includes("бел")    ? "#c9b86a" :
    s.includes("роз")    ? "#d98f97" :
    s.includes("оранж")  ? "#d49a4e" :
                           "#6e2730";

  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 90 260">
      <defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%" stop-color="${glass}" stop-opacity=".72"/>
        <stop offset="34%" stop-color="${glass}"/>
        <stop offset="62%" stop-color="${glass}" stop-opacity=".82"/>
        <stop offset="100%" stop-color="#1d1113" stop-opacity=".55"/>
      </linearGradient></defs>
      <path d="M36 6h18v52c0 10 16 22 16 44v146a6 6 0 0 1-6 6H26a6 6 0 0 1-6-6V102c0-22 16-34 16-44V6Z" fill="url(#g)"/>
      <rect x="34" y="4" width="22" height="13" rx="2" fill="#2c2a28" opacity=".78"/>
      <rect x="22" y="132" width="46" height="70" rx="1.5" fill="#fdf9ed" opacity=".93"/>
      <rect x="29" y="148" width="32" height="2.4" fill="${glass}" opacity=".62"/>
      <rect x="29" y="158" width="24" height="2" fill="#857e79" opacity=".45"/>
      <rect x="29" y="182" width="18" height="1.8" fill="#857e79" opacity=".38"/>
    </svg>`;

  return "data:image/svg+xml;charset=utf-8," + encodeURIComponent(svg.trim());
}

/* ═══════════════ СТАРТ ═══════════════ */

route();

/* Размер каталога на первом экране — с бэкенда, не из вёрстки. */
api.wines({ limit: 1 }).then((page) => {
  const n = page.total;
  if (n) document.getElementById("catalog-total").textContent =
    ` · ${n.toLocaleString("ru-RU")} ${plural(n, "позиция", "позиции", "позиций")}`;
}).catch(() => {});

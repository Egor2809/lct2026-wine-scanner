#!/usr/bin/env python3
"""Стенд оценки: повторяет логику скрипта кейсодержателя.

    python scripts/evaluate.py --labels data/labels.csv
    python scripts/evaluate.py --labels data/field/labels.csv --wait 180 \
        --min-top1 0.952 --max-p95 1500      # оракул приёмки: код выхода 1 при провале

labels.csv: две колонки — путь к фотографии и эталонный slug
(`image,slug` или `path,slug`; заголовок опционален).

Почему именно так, а не иначе:
- запросы строго последовательные — на проверке параллелизма не будет,
  оптимизировать нужно латентность одного запроса;
- сравнение slug нечёткое: другой винтаж того же вина засчитывается как
  попадание (подтверждено кейсодержателем);
- в CSV ошибок выводится margin до второго кандидата — по нему сразу видно,
  спутаны ли near-duplicates одной серии или пайплайн промахнулся вообще.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wine_scanner.catalog.models import same_wine  # noqa: E402


def read_labels(path: Path) -> list[tuple[Path, str]]:
    rows: list[tuple[Path, str]] = []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.reader(fh):
            if len(row) < 2:
                continue
            image, slug = row[0].strip(), row[1].strip()
            if not image or not slug or slug.lower() in {"slug", "label", "expected"}:
                continue  # строка заголовка
            candidate = Path(image)
            if not candidate.is_absolute():
                candidate = path.parent / candidate
            rows.append((candidate, slug))
    return rows


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(round(q / 100 * (len(ordered) - 1))), len(ordered) - 1)
    return ordered[index]


def wait_healthy(client: httpx.Client, url: str, seconds: float) -> bool:
    health = httpx.URL(url).copy_with(path="/health")
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            if client.get(health, timeout=5).status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(1)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8080/scan")
    parser.add_argument("--out", type=Path, default=Path("artifacts/eval_errors.csv"))
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--wait", type=float, default=0.0,
                        help="сколько секунд ждать GET /health после холодного старта")
    parser.add_argument("--min-top1", type=float, default=None, help="порог top-1, доля")
    parser.add_argument("--max-p95", type=float, default=None, help="порог p95, мс")
    parser.add_argument("--sla", type=float, default=3000.0, help="предел одного запроса, мс")
    args = parser.parse_args()

    pairs = read_labels(args.labels)
    if not pairs:
        print(f"В {args.labels} нет ни одной пары (фото, slug)")
        return 1

    top1 = top5 = failed = 0
    latencies: list[float] = []
    errors: list[dict[str, object]] = []

    with httpx.Client(timeout=args.timeout) as client:
        if args.wait and not wait_healthy(client, args.url, args.wait):
            print(f"Сервис не поднялся за {args.wait:.0f} c")
            return 1
        for position, (image_path, expected) in enumerate(pairs, start=1):
            if not image_path.exists():
                print(f"  ! файл не найден: {image_path}")
                failed += 1
                continue

            started = time.perf_counter()
            try:
                response = client.post(
                    args.url,
                    files={"image": (image_path.name, image_path.read_bytes())},
                )
                response.raise_for_status()
                data = response.json()
            except Exception as exc:
                print(f"  ! {image_path.name}: {exc}")
                failed += 1
                continue
            elapsed = (time.perf_counter() - started) * 1000
            latencies.append(elapsed)

            # Ровно то, что делает скрипт кейсодержателя: читает корневой ключ.
            predicted = data.get("slug")
            if not predicted:
                print(f"  ! {image_path.name}: в ответе нет ключа 'slug' в корне")
                failed += 1
                continue

            hit1 = same_wine(predicted, expected)
            candidates = [c.get("slug", "") for c in data.get("candidates", [])][:5]
            hit5 = hit1 or any(same_wine(c, expected) for c in candidates)

            top1 += hit1
            top5 += hit5

            if not hit1:
                errors.append(
                    {
                        "image": str(image_path),
                        "expected": expected,
                        "predicted": predicted,
                        "confidence": data.get("confidence"),
                        "margin": data.get("margin"),
                        "in_top5": hit5,
                        "top5": " ".join(candidates),
                        "elapsed_ms": round(elapsed),
                    }
                )

            mark = "+" if hit1 else ("~" if hit5 else "-")
            print(f"[{position:3d}/{len(pairs)}] {mark} {image_path.name} -> {predicted}")

    total = len(latencies)
    if not total:
        print("\nНи одного успешного ответа — сервис поднят? (GET /health)")
        return 1

    print("\n" + "─" * 56)
    print(f"Фотографий обработано : {total} (ошибок запроса: {failed})")
    print(f"top-1 accuracy        : {top1 / total:6.1%}  ({top1}/{total})")
    print(f"top-5 accuracy        : {top5 / total:6.1%}  ({top5}/{total})")
    p50, p95 = percentile(latencies, 50), percentile(latencies, 95)
    slow = sum(1 for value in latencies if value > args.sla)
    print(f"Латентность p50 / p95 : {p50:.0f} / {p95:.0f} мс")
    print(f"max / первый запрос   : {max(latencies):.0f} / {latencies[0]:.0f} мс"
          f"  (дольше {args.sla / 1000:.0f} c: {slow})")
    print("Baseline заказчика    : 60.0%   Цель: 85%+")
    print("─" * 56)

    if errors:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(errors[0]))
            writer.writeheader()
            writer.writerows(errors)
        print(f"Разбор {len(errors)} ошибок: {args.out}")

        # Промах, но правильный ответ был в top-5 -> визуальный поиск сработал,
        # а разводить кандидатов должен ре-ранкинг. Отдельная строка потому,
        # что это разные задачи и разные шаги плана.
        recoverable = sum(1 for e in errors if e["in_top5"])
        if recoverable:
            print(f"Из них {recoverable} — правильный ответ был в top-5: работа для ре-ранкинга.")

    problems = []
    if failed:
        problems.append(f"ошибок запроса или null: {failed}")
    if slow:
        problems.append(f"запросов дольше SLA: {slow}")
    if args.min_top1 is not None and top1 / total < args.min_top1:
        problems.append(f"top-1 {top1 / total:.1%} < {args.min_top1:.1%}")
    if args.max_p95 is not None and p95 >= args.max_p95:
        problems.append(f"p95 {p95:.0f} мс >= {args.max_p95:.0f} мс")
    if args.min_top1 is not None or args.max_p95 is not None:
        print("ПРИЁМКА: " + ("ПРОВАЛ — " + "; ".join(problems) if problems else "OK"))
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

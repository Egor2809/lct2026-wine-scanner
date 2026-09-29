"""Рейтинг Роскачества — демо-значение для механики карточки.

В выгрузке каталога рейтинга нет; кейсодержатель разрешил сгенерировать его,
чтобы показать, где он живёт в карточке. Значение детерминированное (от
хэша slug без винтажа: разные винтажи одного вина получают одну цифру) и
везде помечено `demo: true` — в API и в интерфейсе. Настоящий рейтинг
подставляется сюда же, когда появится колонка в дампе.
"""

from __future__ import annotations

import hashlib
from typing import Any

from ..catalog.models import Wine, slug_stem

LOW, HIGH = 83, 95
DEMO_NOTE = "Демо-значение: рейтинга Роскачества нет в выгрузке каталога"


def demo_rating(slug: str) -> int:
    """83–95, чаще середина: сумма двух равномерных даёт «колокол» около 89."""
    digest = hashlib.sha256(slug_stem(slug).encode("utf-8")).digest()
    half = (HIGH - LOW) // 2
    return LOW + digest[0] % (half + 1) + digest[1] % (HIGH - LOW - half + 1)


def rating_for(wine: Wine) -> dict[str, Any]:
    return {
        "value": demo_rating(wine.slug),
        "max": 100,
        "source": "Роскачество",
        "demo": True,
        "note": DEMO_NOTE,
    }

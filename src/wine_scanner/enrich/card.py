"""Карточка для ответа API: поля каталога + то, что выводит enrich/.

Wine.card() отдаёт только данные дампа; здесь к ним добавляются гастрономия
и рейтинг. Ключи Wine.card() не удаляются и не переименовываются — их читают
frontend/app.js и скрипт кейсодержателя.
"""

from __future__ import annotations

from typing import Any

from ..catalog.models import Wine
from .pairing import pairing_for
from .rating import rating_for


def card_for(wine: Wine) -> dict[str, Any]:
    card = wine.card()
    if wine.pairing:
        card["pairing_source"] = "catalog"
        card["pairing_basis"] = ""
    else:
        pairing = pairing_for(wine)
        card["pairing"] = pairing.labels
        card["pairing_source"] = pairing.source
        card["pairing_basis"] = pairing.basis
    card["rating"] = rating_for(wine)
    return card

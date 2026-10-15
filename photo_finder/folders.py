"""Arayuzdeki klasor secimi icin yardimcilar: gruplama ve Turkce'ye duyarsiz anahtar kelime eslestirme."""

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable

ROOT_LABEL = "(ana klasor)"


def normalize(text: str) -> str:
    """'ÖDÜL TÖRENİ' -> 'odul toreni', 'Çocuk Koşusu' -> 'cocuk kosusu'.

    Kullanici Turkce karakterle de, karaktersiz de yazabilir; ikisi de eslesmeli.
    Python'un lower()'i Turkce I/İ'yi bilmez ('İ'.lower() -> 'i' + nokta isareti), o yuzden once elle.
    """
    text = text.replace("İ", "i").replace("I", "i").replace("ı", "i").lower()
    decomposed = unicodedata.normalize("NFKD", text)  # 'ö' -> 'o' + iki nokta isareti
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def parse_keywords(text: str) -> list[str]:
    """'21k, Finish ,  ' -> ['21k', 'finish']"""
    return [k for k in (normalize(part.strip()) for part in text.split(",")) if k]


def _compact(text: str) -> str:
    """Harf/rakam disindakileri at: '21 K' ve '21-K' de '21k' olsun."""
    return re.sub(r"[^a-z0-9]", "", text)


def matches_any(folder: str, keywords: list[str]) -> bool:
    """Klasor yolu (ust klasorler dahil) anahtar kelimelerden birini iceriyor mu?"""
    path = _compact(normalize(folder))
    return any(_compact(k) in path for k in keywords if _compact(k))


def group_folders(folders: Iterable[str]) -> dict[str, list[str]]:
    """Klasor yollarini en ust seviyeye gore gruplar: {'BURAK': ['BURAK/SATURDAY/1-START', ...]}."""
    groups: dict[str, list[str]] = {}
    for folder in sorted(set(folders)):
        top = folder.split("/", 1)[0] or ROOT_LABEL
        groups.setdefault(top, []).append(folder)
    return groups


def label(folder: str, top: str, counts: Counter) -> str:
    """Grup icindeki klasorun etiketi: ust klasor adi tekrar edilmez."""
    rest = folder[len(top) + 1:] if folder.startswith(top + "/") else ""
    return f"{rest or '(dogrudan bu klasor)'} — {counts[folder]} foto"

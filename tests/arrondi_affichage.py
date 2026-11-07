# -*- coding: utf-8 -*-
"""Comparer un nombre AFFICHÉ par le site à celui qu'affiche l'écran Streamlit, quand les deux règles d'arrondi
diffèrent : le site arrondit DEMI VERS L'EXTÉRIEUR (api_web/commun.py, comme les formules des moteurs, acier.formats.F),
l'écran Streamlit AU PAIR (format Python, pandas). Elles ne divergent que sur un demi EXACT dont le voisin inférieur est
pair : 0,625 s'écrit « 0,63 » sur le site et « 0,62 » dans Streamlit ; 29,25 → « 29,3 » et « 29,2 ».

La règle est celle de tests/test_api_web_soudures.py (meme_nombre) : un seul écart est permis — le chiffre de Streamlit
est PAIR et celui du site le dépasse d'une unité du dernier rang, en valeur absolue ; quand la valeur brute est connue,
elle doit être un demi EXACT à ce rang (sa valeur binaire). Tout autre écart (un chiffre de plus, un rang, un signe)
reste un écart : la tolérance n'avale pas une vraie différence de calcul."""
import re
from decimal import Decimal

_NOMBRE = re.compile(r"(?<![\d,])(-|−)?\d(?:[\d \xa0 ]*\d)?(?:,\d+)?")


def _chiffre(s: str) -> str:
    return str(s).replace(" ", "").replace("\xa0", "").replace(" ", "").replace("−", "-").replace(".", ",")


def meme_nombre(st, site, brut=None) -> bool:
    """Le même nombre, au demi exact près (voir le module)."""
    a, b = _chiffre(st), _chiffre(site)
    if a == b:
        return True
    if not re.fullmatch(r"-?\d+(,\d+)?", a) or not re.fullmatch(r"-?\d+(,\d+)?", b):
        return False
    nd = len(a.partition(",")[2])
    if nd != len(b.partition(",")[2]):
        return False
    ua, ub = int(a.replace(",", "").lstrip("-")), int(b.replace(",", "").lstrip("-"))
    meme_signe = a.startswith("-") == b.startswith("-") or ua == 0
    if brut is not None:
        d = abs(Decimal(float(brut))).scaleb(nd)
        if d - int(d) != Decimal("0.5"):
            return False
    return meme_signe and ua % 2 == 0 and ub == ua + 1


def memes_nombres(st, site, bruts=None) -> bool:
    """Deux listes de cellules : les textes égaux, les nombres égaux au demi exact près."""
    bruts = list(bruts) if bruts is not None else [None] * len(st)
    return len(st) == len(site) == len(bruts) and all(
        x == y or meme_nombre(x, y, z) for x, y, z in zip(st, site, bruts))


def variantes_demi(texte: str):
    """Le texte de Streamlit, puis chacune de ses variantes où UN nombre au dernier chiffre pair est écrit comme le site
    l'écrirait au demi exact (dernier chiffre + 1, jamais de retenue puisqu'il est pair)."""
    yield texte
    for m in _NOMBRE.finditer(texte):
        n = m.group(0)
        if n[-1].isdigit() and int(n[-1]) % 2 == 0:
            yield texte[:m.start()] + n[:-1] + str(int(n[-1]) + 1) + texte[m.end():]


def present(texte: str, ou: str) -> bool:
    """Le texte de l'écran Streamlit se lit sur le site, au demi exact près sur un de ses nombres."""
    return any(v in ou for v in variantes_demi(texte))


def memes_cellules(st, site) -> bool:
    """Deux rangées de cellules texte (« 2,2–9,0 », « 18,0 », « Limon ») : chaque cellule du site est celle de l'écran,
    ou l'une de ses variantes au demi exact (un seul nombre au dernier chiffre pair, + 1)."""
    return len(st) == len(site) and all(y in set(variantes_demi(x)) for x, y in zip(st, site))

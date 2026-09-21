# -*- coding: utf-8 -*-
"""Sémantique numérique JavaScript, reproduite en Python.

Le moteur de référence est écrit en JavaScript ; sa transcription doit rendre
EXACTEMENT les mêmes nombres et les mêmes chaînes. Les différences entre les
deux langages sont concentrées ici, chacune avec la règle qu'elle reproduit :

- ``dv(a, b)``      division qui ne lève jamais : b = 0 donne ±Infinity ou NaN ;
- ``mn`` / ``mx``   ``Math.min`` / ``Math.max`` : NaN dès qu'un argument est NaN ;
- ``js_round``      ``Math.round`` : demi vers +∞ (2,5 → 3 ; −2,5 → −2) ;
- ``ceil5``         arrondi au multiple de 5 supérieur, avec sa tolérance ;
- ``N``             ``parseFloat`` tolérant : chaîne, None, vide → nombre ou 0 ;
- ``to_fixed``      ``Number.prototype.toFixed`` : demi-supérieur sur la valeur
                    binaire EXACTE (0,125 → « 0.13 » ; 1,005 → « 1.00 ») ;
- ``js_str``        ``String(nombre)`` : « 60 » pour 60.0, « 60.5 » pour 60.5.
"""
import math
from decimal import Decimal, ROUND_HALF_UP

INF = 1e9          # sentinelle « sans objet » du moteur (valeur finie, comme en JS)


class AttrDict(dict):
    """Dictionnaire à accès attribut : ``R.Fz_S`` ≡ ``R['Fz_S']``.

    Permet une transcription lisible, ligne à ligne, de l'objet ``R`` du
    moteur JavaScript, tout en restant un ``dict`` ordinaire (sérialisable,
    comparable clé par clé par les tests de parité)."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]


def N(x):
    """``parseFloat`` puis 0 si non fini (None, '', texte, NaN, ±Infinity)."""
    if x is None or isinstance(x, bool):
        return 0.0 if not isinstance(x, bool) else float(x)
    if isinstance(x, (int, float)):
        return float(x) if math.isfinite(x) else 0.0
    try:
        s = str(x).strip().replace(",", ".")
        # parseFloat lit le plus long préfixe numérique ; ici on accepte les
        # formes usuelles d'un champ de saisie (« 12 », « 12.5 », « -3 »)
        v = float(s)
    except (TypeError, ValueError):
        return 0.0
    return v if math.isfinite(v) else 0.0


def dv(a, b):
    """Division au sens JavaScript : jamais d'exception.

    b = 0 → NaN si a = 0 ou a non fini, sinon ±Infinity selon le signe de a."""
    try:
        return a / b
    except ZeroDivisionError:
        if a == 0 or (isinstance(a, float) and not math.isfinite(a)):
            return math.nan
        return math.copysign(math.inf, a)


def _nan_in(args):
    return any(isinstance(v, float) and math.isnan(v) for v in args)


def mn(*args):
    """``Math.min`` : NaN si un argument est NaN, sinon le minimum."""
    if _nan_in(args):
        return math.nan
    return min(args)


def mx(*args):
    """``Math.max`` : NaN si un argument est NaN, sinon le maximum."""
    if _nan_in(args):
        return math.nan
    return max(args)


def js_round(x):
    """``Math.round`` : l'entier le plus proche, demi vers +∞."""
    if isinstance(x, float) and not math.isfinite(x):
        return x
    r = math.floor(x)
    return int(r) if x - r < 0.5 else int(r) + 1


def ceil5(x):
    """Multiple de 5 supérieur ou égal, avec la tolérance du moteur (1e-9)."""
    return int(math.ceil(x / 5 - 1e-9)) * 5


def hyp(a, b, c=0.0):
    """``Math.sqrt(a² + b² + c²)`` — ne lève jamais (NaN/Infinity propagés)."""
    s = a * a + b * b + (c or 0.0) * (c or 0.0)
    if isinstance(s, float) and (math.isnan(s) or math.isinf(s)):
        return s
    return math.sqrt(s)


def to_fixed(x, n):
    """``x.toFixed(n)`` de JavaScript (point décimal, signe « - »).

    Règle de la norme ECMAScript : on choisit l'entier k tel que k/10ⁿ − x
    soit le plus proche de zéro ; en cas d'égalité, le plus grand k. Le signe
    est traité à part, d'où un arrondi « demi vers l'extérieur » appliqué à
    la valeur binaire EXACTE du flottant — ce que fait ``Decimal(x)``."""
    if isinstance(x, bool):
        x = float(x)
    if x == 0:
        x = 0.0                                   # toFixed(-0) → « 0.00 »
    d = Decimal(x) if isinstance(x, float) else Decimal(int(x))
    q = Decimal(1).scaleb(-int(n))
    r = d.quantize(q, rounding=ROUND_HALF_UP)
    return format(r, "f")


def js_str(x):
    """``String(x)`` pour un nombre : entier sans décimales, sinon la plus
    courte représentation qui se relit à l'identique (comme en JavaScript)."""
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if math.isnan(x):
            return "NaN"
        if math.isinf(x):
            return "Infinity" if x > 0 else "-Infinity"
        if x == 0:
            return "0"
        if x.is_integer() and abs(x) < 1e21:
            return str(int(x))
        s = repr(x)
        if "e" in s:                              # 1e-07 → 1e-7 (forme JS)
            m, e = s.split("e")
            e = int(e)
            s = f"{m}e{'+' if e > 0 else '-'}{abs(e)}"
        return s
    return str(x)

# -*- coding: utf-8 -*-
"""Formatage des nombres, identique au HTML de référence.

- ``F(x, n)``  : ``toFixed(n)`` avec la virgule décimale ; « — » si non fini ;
- ``f0(x)``    : 0 ou 1 décimale (1 si la partie décimale dépasse 0,05), signe
                 typographique « − » ;
- ``pct(e, n)`` : pourcentage, « > 999 % » au-delà de 9,99, « — » si absent ; ``a_pct`` : « à 107,7 % » ;
- ``f_ecart(x, ref)`` : ``f0``, ou la décimale qui départage ``x`` de ``ref`` (« 80,04 mm > 80 mm ») ;
- ``mm_fr(texte)`` : les épaisseurs d'un message de la table (« 260.5 mm ») écrites à la virgule ;
- ``nb(x, dec)`` : une valeur saisie écrite telle quelle, à la virgule (« 260,5 ») — le formateur de l'écran
  (copie nommée de ``api_web.commun.nb``), pour les dimensions des en-têtes, textes et notes.

Ces trois fonctions écrivent les chaînes ``vals`` des vérifications, les
explications chiffrées des alertes, l'export texte et les cotes des dessins :
elles font partie de la parité.
"""
import decimal
import math
import numbers
import re

from .js import to_fixed, js_round


def _fini(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) \
        and math.isfinite(x)


def F(x, n=1):
    """``F(x, n)`` du moteur JS : nombre → chaîne à n décimales, virgule."""
    if x is None or not _fini(x):
        return "—"
    return to_fixed(x, n).replace(".", ",")


def f0(x):
    """``f0(x)`` : une décimale seulement si nécessaire, signe « − »."""
    if x is None or not _fini(x):
        return "—"
    n = 1 if abs(x - js_round(x)) > 0.05 else 0
    return F(x, n).replace("-", "−", 1)


def fixe(x, nd):
    """``x`` écrit avec ``nd`` décimales (point décimal), arrondi DEMI VERS L'EXTÉRIEUR sur la valeur binaire exacte :
    la règle des moteurs (``acier.js.to_fixed``), appelée — pas recopiée. La précision décimale est élargie pour qu'un
    très grand nombre (1e30) s'écrive au lieu de lever. Le signe d'un zéro est ôté : jamais « -0 ». Un nombre de numpy
    passe d'abord en nombre Python (``to_fixed`` tronquait un numpy.float32 à l'entier). Copie NOMMÉE de
    ``api_web.commun._fixe`` (l'écran) ; tests/test_api_web_assemblages.py § 14 les lie."""
    x = int(x) if isinstance(x, numbers.Integral) else float(x)
    with decimal.localcontext() as ctx:
        ctx.prec = 420                            # 309 chiffres pour le plus grand flottant, plus les décimales
        s = to_fixed(x, int(nd))
    if s.startswith("-") and not s.strip("-0."):
        s = s[1:]
    return s


def nb(x, dec=3):
    """Une valeur saisie (coefficient, dimension) écrite telle quelle, sans arrondi au-delà de ``dec`` décimales :
    1,25 reste 1,25 (un format de taux en ferait « 1,2 »), 210000 reste 210000 (seuls les zéros APRÈS la virgule
    s'ôtent), 0,5 à zéro décimale donne 1 (demi vers l'extérieur). Pas de séparateur de milliers (``num`` les pose
    dans un champ). Le chiffre écrit est relu et comparé à la valeur ; jamais « -0 » ; « — » si non fini. Le
    formateur de l'écran (copie NOMMÉE de ``api_web.commun.nb``, liée par tests/test_api_web_assemblages.py § 14), ici
    pour que les en-têtes, textes et notes des moteurs écrivent une dimension comme son champ (« platine 500×500×260,5 »,
    jamais « 260.5 » ; lot B)."""
    if x is None:
        return ""
    if not _fini(float(x)):
        return "—"
    s = fixe(float(x), dec)
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    assert abs(float(s) - float(x)) <= 0.5 * 10 ** -dec * (1 + 1e-9) + 1e-12 * max(1.0, abs(float(x))), (x, s)
    return s.replace(".", ",")


def pct(e, n=0):
    """``pct(e, n)`` de l'interface : taux → « 35,4 % »."""
    if e is None or (isinstance(e, float) and math.isnan(e)):
        return "—"
    if e > 9.99:
        return "> 999 %"
    return F(e * 100, n or 0) + " %"


def _valeur(texte):
    """Le nombre qu'un texte de ``F`` / ``f0`` donne à lire (« −0 » se lit 0)."""
    return float(texte.replace("−", "-").replace(",", "."))


def a_pct(e, n=0):
    """« à 107,7 % » — le taux de ``pct`` précédé de « à » ; au-delà de l'échelle de ``pct`` (« > 999 % ») : « au-delà de
    999 % » (jamais « à > 999 % »). La raison courte d'un « À vérifier » (pied de poteau, lot B)."""
    t = pct(e, n)
    return "au-delà de " + t[2:] if t.startswith("> ") else "à " + t


def f_ecart(x, ref):
    """``x`` écrit par ``f0``, sauf quand ``f0`` l'écrirait avec la VALEUR de ``ref`` : alors avec la décimale qui les
    DÉPARTAGE (``F``, jusqu'à 6 décimales) — « épaisseur 80,04 mm > 80 mm », jamais « 80 mm > 80 mm » ; « H = −0,04 mm »,
    jamais « −0 mm » (le zéro signé se lit 0 : lot B, brief v12 § 19.2). Même formateur que le reste du moteur : ``F``
    et ``f0``, aucun autre. ``x`` égal à ``ref`` : ``f0`` (« 0 », jamais « 0,000000 »)."""
    t = f0(x)
    if x is None or not _fini(x) or ref is None or not _fini(ref) or x == ref or _valeur(t) != _valeur(f0(ref)):
        return t
    for n in range(1, 7):
        if _valeur(F(x, n)) != _valeur(F(ref, n)):
            return F(x, n).replace("-", "−", 1)
    return F(x, 6).replace("-", "−", 1)


# un nombre décimal écrit avec le point, suivi de « mm » (« 260.5 mm ») ; « tableau 3.3 ANB » n'est pas une épaisseur
_MM_POINT = re.compile(r"(?<![\d.])(\d+)\.(\d+)(?= mm(?![\w²³]))")


def mm_fr(texte):
    """Un message de la table unique (``acier.materiaux.HorsTable``, écrit avec le point de Python : « S355 laminé de
    260.5 mm ») dans l'écriture du site : « 260,5 mm ». Seules les épaisseurs (un nombre suivi de « mm ») changent ; le
    reste du message est gardé tel quel. ``acier/materiaux.py`` (la table) ne se modifie pas : l'appelant qui montre le
    message le passe ici (lot B, brief v12 § 19.2)."""
    return _MM_POINT.sub(r"\1,\2", str(texte))

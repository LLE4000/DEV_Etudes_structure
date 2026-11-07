# -*- coding: utf-8 -*-
"""Cornière ancrée — l'ENTRÉE du moteur : les valeurs par défaut et le domaine de chaque champ de l'écran
Streamlit ``modules/corniere.py`` (ses ``st.number_input``, ``st.radio``, ``st.toggle``), le catalogue des cornières
(``acier/profiles``, source unique des profilés de l'application) et la normalisation stricte d'une cornière.

Bibliothèque standard ; le catalogue (``acier.profiles.catalogue``) charge numpy, présent dans le navigateur.
Une valeur hors du domaine de l'écran lève ValueError : l'écran Streamlit la refuse (bornes de ses champs), le
moteur aussi — il ne la ramène jamais en silence à une autre. Un seul refus de plus que l'écran : une saisie libre
où l'épaisseur atteint une aile (t ≥ A ou t ≥ B), qui n'est pas une cornière (docs/acier/CORNIERE_MOTEUR.md, § 2).
"""
import math
import re
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from acier.profiles import catalogue as CAT

REFERENCE = "modules/corniere.py"

# ------------------------------------------------------------------ les choix de l'écran
LINEAIRE, PONCTUELLE = "lineaire", "ponctuelle"
CHARGES = (LINEAIRE, PONCTUELLE)
# les libellés du bouton radio « Type de charge » de l'écran (admis à la relecture, même sens)
LIBELLES_CHARGE = {"Linéaire (kN/m)": LINEAIRE, "Ponctuelle (kN)": PONCTUELLE}
FAMILLES = ("L", "LI")                      # cornières à ailes égales, à ailes inégales (EN 10056-1)
LIBELLES_FAMILLE = {"L": "Ailes égales", "LI": "Ailes inégales"}

# ------------------------------------------------------------------ les champs numériques : (défaut, min, max, unité, libellé)
# défauts et bornes = ceux des st.number_input de l'écran (None : pas de borne) ; tests/test_corniere_moteur.py les
# relit dans l'écran lui-même (faux st) : une valeur changée ici ou là-bas fait rougir le test.
CHAMPS: Dict[str, Tuple[float, Optional[float], Optional[float], str, str]] = {
    "A": (60.0, 10.0, 300.0, "mm", "largeur de l'aile A"),
    "B": (60.0, 10.0, 300.0, "mm", "largeur de l'aile B"),
    "t": (6.0, 2.0, 20.0, "mm", "épaisseur t"),
    "V": (5.0, 0.0, None, "kN/m ou kN", "effort tranchant V"),
    "L": (1.0, 0.1, 5.0, "m", "longueur d'influence L"),
    "e": (2.3, 0.0, None, "cm", "excentricité e"),
    "gamma_ELU": (1.35, 1.0, None, "", "coefficient global ELU"),
    "s": (30.0, 10.0, None, "cm", "espacement des fixations s"),
    "bord": (2.0, 0.0, None, "cm", "distance de l'âme au bord du mur"),
    "sig_lim_ELS": (155.0, 10.0, None, "N/mm²", "σ_lim,ELS"),
    "tau_lim_ELS": (0.8, 0.1, None, "N/mm²", "τ_lim,ELS"),
    "sig_lim_ELU": (186.0, 10.0, None, "N/mm²", "σ_eq,lim,ELU"),
}
# la section proposée à l'ouverture : la PREMIÈRE de la liste de l'écran, sorted(ANGLES_STD) = « 100x100x10 »
# (tri alphabétique ; voir docs/acier/CORNIERE_MOTEUR.md, défauts) — la même cornière au catalogue
PROFIL_DEFAUT = "L 100x100x10"
CLES = ("nom", "standard", "profil", "A", "B", "t", "charge", "V", "L", "e", "gamma_ELU", "s", "bord",
        "sig_lim_ELS", "tau_lim_ELS", "sig_lim_ELU")


# ------------------------------------------------------------------ le catalogue
@lru_cache(maxsize=1)
def _catalogue() -> Tuple[Dict, ...]:
    """Les cornières du catalogue (acier/donnees/profiles.json), triées par masse puis par nom."""
    recs = [p for p in CAT.profils() if p["famille"] in FAMILLES]
    return tuple(sorted(recs, key=lambda p: (p["masse"], p["nom"])))


def cornieres(familles=None) -> List[Dict]:
    """Les cornières du catalogue (les familles demandées ; toutes par défaut), triées par masse croissante."""
    fams = tuple(familles) if familles else FAMILLES
    return [p for p in _catalogue() if p["famille"] in fams]


def section_catalogue(nom: str) -> Dict:
    """Une cornière du catalogue par son nom (« L 100x100x10 ») ; ValueError si ce n'en est pas une."""
    for p in _catalogue():
        if p["nom"] == nom:
            return p
    raise ValueError(f"profil : « {nom} » n'est pas une cornière du catalogue")


def designation(nom: str) -> str:
    """« L 100x100x10 » → « L 100×100×10 », « L 45x45x4.5 » → « L 45×45×4,5 » (le signe de multiplication et la
    virgule du bureau)."""
    return nom.replace("x", "×").replace(".", ",")


def nom_catalogue(texte: str) -> Optional[str]:
    """Une désignation tapée (« 80x80x8 », « L 80×80×8 », « l80*80*8 », « 100 x 65 x 8 », ailes dans un ordre ou
    dans l'autre, « 4,5 ») → le nom de la cornière au catalogue (« L 80x80x8 ») ; None si elle n'y est pas."""
    if not isinstance(texte, str):
        return None
    s = texte.strip().lower().replace("×", "x").replace("*", "x").replace(",", ".").replace(" ", "").replace("\xa0", "")
    s = s[1:] if s.startswith("l") else s
    m = re.fullmatch(r"(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)", s)
    if not m:
        return None
    a, b, t = (float(x) for x in m.groups())
    noms = {p["nom"] for p in _catalogue()}
    for x, y in ((a, b), (b, a)):
        nom = f"L {x:g}x{y:g}x{t:g}"
        if nom in noms:
            return nom
    return None


# ------------------------------------------------------------------ la cornière neuve, la normalisation
def nouvelle_corniere() -> Dict:
    """Une cornière aux valeurs par défaut de l'écran Streamlit."""
    c = {"nom": "Cornière 1", "standard": True, "profil": PROFIL_DEFAUT, "charge": LINEAIRE}
    c.update({k: v[0] for k, v in CHAMPS.items()})
    return {k: c[k] for k in CLES}


def _fr(x: float) -> str:
    return f"{x:g}".replace(".", ",")


def _nombre(cle: str, x) -> float:
    defaut, mini, maxi, unite, libelle = CHAMPS[cle]
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"{libelle} : un nombre est attendu, pas {x!r}")
    x = float(x)
    if not math.isfinite(x):
        raise ValueError(f"{libelle} : nombre infini ou indéterminé")
    u = f" {unite}" if unite and unite != "kN/m ou kN" else ""
    if (mini is not None and x < mini) or (maxi is not None and x > maxi):
        borne = f"au moins {_fr(mini)}{u}" if maxi is None else f"de {_fr(mini)} à {_fr(maxi)}{u}"
        raise ValueError(f"{libelle} = {_fr(x)}{u} hors du domaine de l'écran Streamlit ({borne})")
    return x + 0.0                  # « -0 » relu 0 (sinon le champ écrirait « -0 »)


def normaliser(c: Dict) -> Dict:
    """Une copie complétée (une clé absente ou None prend le défaut de l'écran) et contrôlée : une clé inconnue, un
    type faux, un nombre hors des bornes de l'écran, un profil absent du catalogue lèvent ValueError."""
    if not isinstance(c, dict):
        raise ValueError("une cornière est un objet JSON")
    inconnues = sorted(set(c) - set(CLES))
    if inconnues:
        raise ValueError(f"clé(s) inconnue(s) : {', '.join(inconnues)}")
    d = nouvelle_corniere()
    out = {}
    for k in CLES:
        v = c.get(k)
        if v is None:
            v = d[k]
        if k == "nom":
            if not isinstance(v, str):
                raise ValueError("nom : un texte est attendu")
        elif k == "standard":
            if not isinstance(v, bool):
                raise ValueError("standard : vrai (cornière du catalogue) ou faux (saisie libre)")
        elif k == "profil":
            if not isinstance(v, str):
                raise ValueError("profil : un nom de cornière est attendu")
            section_catalogue(v)
        elif k == "charge":
            v = LIBELLES_CHARGE.get(v, v)
            if v not in CHARGES:
                raise ValueError(f"type de charge : « {v} » (linéaire ou ponctuelle)")
        else:
            v = _nombre(k, v)
        out[k] = v
    # une cornière a des ailes plus larges que son épaisseur : l'écran Streamlit accepte t ≥ A (il ne lit que t), mais
    # ce n'est plus une cornière (ni dessin ni aile B où poser la charge) — refus ajouté par le site, écrit dans la doc
    aile, nom_aile = min((out["A"], "A"), (out["B"], "B"))
    if out["t"] >= aile:
        raise ValueError(f"épaisseur t = {_fr(out['t'])} mm ≥ aile {nom_aile} = {_fr(aile)} mm : ce n'est pas une cornière "
                         "(les ailes doivent être plus larges que l'épaisseur)")
    return out

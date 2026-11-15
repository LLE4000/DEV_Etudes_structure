# -*- coding: utf-8 -*-
"""Fondation de grue — les EFFORTS SCIA recopiés (lot 5, cahier des charges du 15/11/2026, § 8) : « SCIA donne le niveau,
le module donne la forme ».

Six valeurs SCIA (trois quand tout est symétrique : la case « x = y ») : M+ le plus fort des bandes BX et des bandes BY
(kNm, sur la largeur de la bande), V à d le plus fort des bandes BX et BY (kN), mx et my les plus négatifs de la carte 2D
(kNm/m) ; et, pour comparer seulement, σ max de chaque k coché. Pour chaque ligne, r = SCIA ÷ module :

  - en bas, en x : r des bandes BX, pour les bandes BX et pour la carte (mx en bas) ; en bas, en y : r des bandes BY ;
  - en haut, en x : r de mx, pour les bandes BX (M−) et pour la carte ; en haut, en y : r de my ;
  - les moments du module sont multipliés par r et le principe d'armatures est refait (``armatures.principe``,
    paramètre ``echelle``) ; l'effort tranchant SCIA (÷ largeur) va directement au moteur ; σ max : comparaison seule.

Un champ vide reste vide : r = 1 (la valeur du module est gardée) et la note dit « SCIA non saisi ». r hors de 0,85 à
1,15 : « Écart module / SCIA : x % — vérifier le modèle SCIA (k, bandes, unités) ».

    symetrique(el) -> bool                      la dalle carrée sous un châssis carré, centré, non tourné
    sym(el) -> bool                             la case « x = y » (cochée par défaut quand la dalle est symétrique)
    module(ef, par_k) -> {ligne: valeur}        les valeurs grises du module (T14)
    rapports(el, ef, par_k) -> {...}            les lignes du tableau, r et l'écart de chacune, l'échelle, V SCIA
"""
import math
from typing import Dict, List, Optional

ECART = 0.15                    # r hors de 1 ± 0,15 : écart à vérifier (§ 8.2)
# les lignes du tableau (§ 8.1) : (clé, libellé, unité, ce que le module en lit)
LIGNES = (("bx_m", "Bandes BX : M+ le plus fort", "kN·m"), ("by_m", "Bandes BY : M+ le plus fort", "kN·m"),
          ("bx_v", "Bandes BX : V à d le plus fort", "kN"), ("by_v", "Bandes BY : V à d le plus fort", "kN"),
          ("mx", "Dalle : m_x le plus négatif", "kN·m/m"), ("my", "Dalle : m_y le plus négatif", "kN·m/m"))
# x = y : les lignes y prennent la valeur de leur ligne x
JUMELLES = {"by_m": "bx_m", "by_v": "bx_v", "my": "mx"}


def symetrique(el: Dict) -> bool:
    """La dalle est-elle symétrique en x et y (§ 8.1) : rectangle carré, châssis carré, centré, β = 0 (à 90° près) ?"""
    d, g = el["dalle"], el["grue"]
    if d["forme"] != "rectangle" or abs(d["Lx"] - d["Ly"]) > 1e-9:
        return False
    from . import chassis as CH
    ch = CH.lire(g, el["options"]["rond"])
    if abs(ch["a"] - ch["b"]) > 1e-9 or abs(g["ex"]) > 1e-9 or abs(g["ey"]) > 1e-9:
        return False
    return abs(math.sin(math.radians(g["beta"]) * 2)) < 1e-9 or abs(math.sin(math.radians(g["beta"]))) < 1e-9


def sym(el: Dict) -> bool:
    """La case « x = y » : celle de l'ingénieur, ou, tant qu'il n'y a pas touché, la symétrie de la dalle."""
    s = el["scia"]["sym"]
    return symetrique(el) if s is None else bool(s)


def module(ef: Dict, par_k: Optional[List[Dict]] = None) -> Dict[str, Optional[float]]:
    """Les valeurs du module (les valeurs grises, § 8.1 ; T14) : le plus fort des bandes de chaque sens, les extrêmes de
    la carte des moments au point."""
    def plus_fort(prefixe, cle):
        v = [b[cle] for b in ef["bandes"] if b["nom"].startswith(prefixe) and b[cle] is not None]
        return max(v) if v else None
    ex = ef["extremes"]
    return {"bx_m": plus_fort("BX", "M_plus"), "by_m": plus_fort("BY", "M_plus"), "bx_v": plus_fort("BX", "V_d"),
            "by_v": plus_fort("BY", "V_d"), "mx": ex["sup_x"], "my": ex["sup_y"]}


def saisie(el: Dict) -> Dict[str, Optional[float]]:
    """Les valeurs SCIA tapées ; avec « x = y », les lignes y sont celles des lignes x."""
    s = dict((c, el["scia"][c]) for c, _, _ in LIGNES)
    if sym(el):
        for y, x in JUMELLES.items():
            s[y] = s[x]
    return s


def rapports(el: Dict, ef: Dict, par_k: Optional[List[Dict]] = None) -> Dict:
    """Le tableau module / SCIA (§ 8.2) :
    {"lignes": [{"cle", "libelle", "unite", "module", "scia" (ou None), "r" (ou None), "ecart" (|r − 1|, ou None),
                 "alerte" (bool)}], "echelle": {"inf_x", "inf_y", "sup_x", "sup_y"} (r, ou 1 sans valeur SCIA),
     "V": {"x", "y"} (kN SCIA, ou None), "sigma": [{"k", "module", "scia", "r"}], "saisi" (au moins une valeur),
     "complet" (toutes les valeurs à taper le sont), "sym", "alertes": [textes]}"""
    mod = module(ef, par_k)
    sc = saisie(el)
    lignes = []
    for cle, lib, unite in LIGNES:
        m, s = mod[cle], sc[cle]
        r = s / m if (s is not None and m not in (None, 0.0)) else None
        e = abs(r - 1.0) if r is not None else None
        lignes.append({"cle": cle, "libelle": lib, "unite": unite, "module": m, "scia": s, "r": r, "ecart": e,
                       "alerte": e is not None and e > ECART + 1e-12})
    r = {li["cle"]: li["r"] for li in lignes}
    echelle = {"inf_x": r["bx_m"] or 1.0, "inf_y": r["by_m"] or 1.0, "sup_x": r["mx"] or 1.0, "sup_y": r["my"] or 1.0}
    sigma = []
    for x in el["sol"]["k"]:
        if not x["scia"]:
            continue
        rk = next((p for p in (par_k or []) if abs(p["k"] - x["k"]) < 1e-9), None)
        v = el["scia"]["sigma"].get(_cle_k(x["k"]))
        sm = rk["sigma_max"] if rk else None
        sigma.append({"k": x["k"], "module": sm, "scia": v, "r": (v / sm) if (v is not None and sm) else None})
    a_taper = [c for c, _, _ in LIGNES if not (sym(el) and c in JUMELLES)]
    alertes = [f"Écart module / SCIA : {_pct((li['r'] - 1) * 100)} % sur {li['libelle'][0].lower() + li['libelle'][1:]} — "
               "vérifier le modèle SCIA (k, bandes, unités)" for li in lignes if li["alerte"] and not (sym(el) and li["cle"] in JUMELLES)]
    return {"lignes": lignes, "echelle": echelle, "V": {"x": sc["bx_v"], "y": sc["by_v"]}, "sigma": sigma,
            "saisi": any(sc[c] is not None for c in a_taper), "complet": all(sc[c] is not None for c in a_taper),
            "sym": sym(el), "a_taper": a_taper, "alertes": alertes}


def _cle_k(k: float) -> str:
    return f"{k:g}"


def cle_k(k: float) -> str:
    """La clé d'un k dans « sigma » (le texte de sa valeur : « 2 », « 2.5 »)."""
    return _cle_k(k)


def _pct(x: float) -> str:
    return (f"{x:+.1f}").replace(".", ",")

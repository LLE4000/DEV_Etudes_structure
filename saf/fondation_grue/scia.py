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

Revue du 15/11/2026 (trois défauts NON CONSERVATIFS, chacun figé par tests/test_saf_grue_armatures.py) :
  - r = |SCIA| ÷ |module| (comme V) : un moment tapé avec le signe opposé ne retourne plus l'échelle (une échelle ≤ 0
    effaçait tous les renforts de la face) ; le signe opposé est une alerte ;
  - « x = y » n'est cochée d'office que si la dalle, les surfaces d'appui, les plots et les ressorts du sol sont
    symétriques par l'échange x ↔ y (des appuis 0,70 × 0,35 à 0° ne le sont pas) ; l'alerte d'une ligne y n'est plus
    cachée quand son écart diffère de celui de sa ligne x ;
  - les valeurs SCIA sont liées au modèle pour lequel elles ont été tapées (``signature``) : les données du modèle
    changent, elles ne servent plus (r = 1, V du module), « à ressaisir ».

    signature(el) -> str                        ce dont dépendent les efforts (le modèle SCIA) — pas le nom, la limite, les
                                                treillis, la vue
    perime(el) -> bool                          des valeurs tapées pour un AUTRE modèle
    symetrique(el) -> bool                      la dalle, les appuis, les plots et le sol inchangés par l'échange x ↔ y
    sym(el) -> bool                             la case « x = y » (cochée par défaut quand la dalle est symétrique)
    module(ef, par_k) -> {ligne: valeur}        les valeurs grises du module (T14)
    rapports(el, ef, par_k) -> {...}            les lignes du tableau, r et l'écart de chacune, l'échelle, V SCIA
"""
import hashlib
import json
import math
from typing import Dict, List, Optional

ECART = 0.15                    # r hors de 1 ± 0,15 : écart à vérifier (§ 8.2)
PERIME = "Valeurs SCIA tapées pour un autre modèle (les données ont changé depuis) : non utilisées — à ressaisir"
# les lignes du tableau (§ 8.1) : (clé, libellé, unité, ce que le module en lit)
LIGNES = (("bx_m", "Bandes BX : M+ le plus fort", "kN·m"), ("by_m", "Bandes BY : M+ le plus fort", "kN·m"),
          ("bx_v", "Bandes BX : V à d le plus fort", "kN"), ("by_v", "Bandes BY : V à d le plus fort", "kN"),
          ("mx", "Dalle : m_x le plus négatif", "kN·m/m"), ("my", "Dalle : m_y le plus négatif", "kN·m/m"))
# x = y : les lignes y prennent la valeur de leur ligne x
JUMELLES = {"by_m": "bx_m", "by_v": "bx_v", "my": "mx"}


# ce qui, de la fondation, ne change pas les efforts : le nom, les textes du projet, la limite du sol, les treillis, les
# valeurs SCIA elles-mêmes, la vue, les étapes, la proposition ; et, des options, l'unité et la plage de « Proposer »
HORS_MODELE = ("nom", "projet", "limite", "armatures", "scia", "vue", "etapes", "proposition")
OPTIONS_HORS_MODELE = ("unite", "h_min", "h_max", "h_pas")
GRUE_HORS_MODELE = ("dmin", "repere")


def signature(el: Dict) -> str:
    """L'empreinte du modèle dont SCIA a tiré ses efforts : tout ce qui change les efforts (dalle, grue, sol, réactions,
    bandes, options de calcul) ; 16 caractères."""
    m = {k: v for k, v in el.items() if k not in HORS_MODELE}
    m["options"] = {k: v for k, v in (el.get("options") or {}).items() if k not in OPTIONS_HORS_MODELE}
    m["grue"] = {k: v for k, v in (el.get("grue") or {}).items() if k not in GRUE_HORS_MODELE}
    return hashlib.sha256(json.dumps(m, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def _valeurs(el: Dict) -> bool:
    sc = el["scia"]
    return any(sc.get(c) is not None for c, _, _ in LIGNES) or bool(sc.get("sigma"))


def perime(el: Dict) -> bool:
    """Des valeurs SCIA tapées pour un AUTRE modèle (les données ont changé depuis) : elles ne servent plus."""
    base = el["scia"].get("base")
    return bool(base) and _valeurs(el) and base != signature(el)


def _echange(cx: float, cy: float, poly) -> List:
    """Un polygone renvoyé par l'échange x ↔ y autour du centre (la symétrie par la diagonale)."""
    return [(cx + (y - cy), cy + (x - cx)) for x, y in poly]


def _memes_polygones(A, B, tol: float = 1e-6) -> bool:
    """Deux listes de polygones égales comme ensembles (l'ordre des polygones et de leurs sommets ne compte pas)."""
    def cle(poly):
        return sorted((round(x / tol) * tol, round(y / tol) * tol) for x, y in poly)
    a, b = sorted(cle(p) for p in A), sorted(cle(p) for p in B)
    return len(a) == len(b) and all(len(p) == len(q) and all(abs(u[0] - v[0]) <= 10 * tol and abs(u[1] - v[1]) <= 10 * tol
                                                             for u, v in zip(p, q)) for p, q in zip(a, b))


def symetrique(el: Dict) -> bool:
    """La fondation est-elle inchangée par l'échange x ↔ y (§ 8.1 : dalle carrée, châssis carré, centré, β = 0 à 90°
    près) ? On regarde aussi ce que la phrase du cahier suppose : les surfaces d'appui, les plots et leurs carrés de
    diffusion se retrouvent eux-mêmes, et les ressorts du sol sont les mêmes en x et en y."""
    d, g, o = el["dalle"], el["grue"], el["options"]
    if d["forme"] != "rectangle" or abs(d["Lx"] - d["Ly"]) > 1e-9:
        return False
    from . import chassis as CH
    ch = CH.lire(g, o["rond"])
    if abs(ch["a"] - ch["b"]) > 1e-9 or abs(g["ex"]) > 1e-9 or abs(g["ey"]) > 1e-9:
        return False
    if not (abs(math.sin(math.radians(g["beta"]) * 2)) < 1e-9 or abs(math.sin(math.radians(g["beta"]))) < 1e-9):
        return False
    if abs(o["c1x"] - o["c1y"]) > 1e-12 or abs(o["c2x"] - o["c2y"]) > 1e-12:
        return False
    from . import projet as PR
    m = PR.modele(el)
    cx, cy = m["centre"]
    groupes = [m["plaques"]]
    if m["plots"]:
        groupes += [m["plots"]["polys"], m["plots"]["diffusion"]]
    return all(_memes_polygones(gr, [_echange(cx, cy, p) for p in gr]) for gr in groupes)


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
    {"lignes": [{"cle", "libelle", "unite", "module", "scia" (ou None), "r" (|SCIA| ÷ |module|, ou None), "ecart"
                 (|r − 1|, ou None), "oppose" (signe opposé), "alerte" (bool)}], "echelle": {"inf_x", "inf_y", "sup_x",
                 "sup_y"} (r > 0, ou 1 sans valeur SCIA), "V": {"x", "y"} (kN SCIA, ou None), "sigma": [{"k", "module",
                 "scia", "r", "ecart", "alerte"}], "saisi" (au moins une valeur, σ compris), "complet" (toutes les valeurs à
                 taper le sont ; σ est facultatif), "sym",
     "alertes": [textes], "perime" (valeurs d'un autre modèle : aucune n'est utilisée)}"""
    mod = module(ef, par_k)
    vieux = perime(el)
    sc = {c: None for c, _, _ in LIGNES} if vieux else saisie(el)
    lignes = []
    for cle, lib, unite in LIGNES:
        m, s = mod[cle], sc[cle]
        # (r = |SCIA| ÷ |module| : SCIA donne le niveau ; un signe opposé à celui du module est une alerte, jamais une
        #  échelle négative qui effacerait les renforts de la face)
        r = abs(s) / abs(m) if (s is not None and m not in (None, 0.0)) else None
        e = abs(r - 1.0) if r is not None else None
        oppose = s is not None and m not in (None, 0.0) and s * m < 0
        lignes.append({"cle": cle, "libelle": lib, "unite": unite, "module": m, "scia": s, "r": r, "ecart": e, "oppose": oppose,
                       "alerte": oppose or (e is not None and e > ECART + 1e-12)})
    r = {li["cle"]: li["r"] for li in lignes}

    def ech(v):
        return v if (v is not None and v > 0) else 1.0
    echelle = {"inf_x": ech(r["bx_m"]), "inf_y": ech(r["by_m"]), "sup_x": ech(r["mx"]), "sup_y": ech(r["my"])}
    sigma = []
    for x in el["sol"]["k"]:
        if not x["scia"]:
            continue
        rk = next((p for p in (par_k or []) if abs(p["k"] - x["k"]) < 1e-9), None)
        v = None if vieux else el["scia"]["sigma"].get(_cle_k(x["k"]))
        sm = rk["sigma_max"] if rk else None
        rs = (v / sm) if (v is not None and sm) else None
        es = abs(rs - 1.0) if rs is not None else None
        sigma.append({"k": x["k"], "module": sm, "scia": v, "r": rs, "ecart": es, "alerte": es is not None and es > ECART + 1e-12})
    s_ = sym(el)
    a_taper = [c for c, _, _ in LIGNES if not (s_ and c in JUMELLES)]
    par_cle = {li["cle"]: li for li in lignes}

    def montree(li):
        # x = y : la ligne y est celle de x ; son alerte n'est cachée que si elle dit la même chose que celle de x
        if not (s_ and li["cle"] in JUMELLES):
            return True
        lx = par_cle[JUMELLES[li["cle"]]]
        return not (lx["alerte"] and lx["r"] is not None and li["r"] is not None and abs(lx["r"] - li["r"]) < 1e-6
                    and lx["oppose"] == li["oppose"])

    def texte(li):
        nom = li["libelle"][0].lower() + li["libelle"][1:]
        if li["oppose"]:
            return f"Signe SCIA opposé à celui du module sur {nom} — vérifier la convention de signe (repère local) dans SCIA"
        return f"Écart module / SCIA : {_pct((li['r'] - 1) * 100)} % sur {nom} — vérifier le modèle SCIA (k, bandes, unités)"
    alertes = [texte(li) for li in lignes if li["alerte"] and montree(li)]
    # (σ max : comparaison seule, mais un écart hors de 0,85 à 1,15 se dit aussi — revue du 15/11/2026)
    alertes += [f"Écart module / SCIA : {_pct((x['r'] - 1) * 100)} % sur σ max, k = {_k_fr(x['k'])} MN/m³ — vérifier le modèle SCIA "
                "(k, bandes, unités)" for x in sigma if x["alerte"]]
    if vieux:
        alertes.insert(0, PERIME)
    return {"lignes": lignes, "echelle": echelle, "V": {"x": sc["bx_v"], "y": sc["by_v"]}, "sigma": sigma,
            "saisi": any(sc[c] is not None for c in a_taper) or any(x["scia"] is not None for x in sigma),
            "complet": all(sc[c] is not None for c in a_taper),
            "sym": s_, "a_taper": a_taper, "alertes": alertes, "perime": vieux}


def _cle_k(k: float) -> str:
    return f"{k:g}"


def cle_k(k: float) -> str:
    """La clé d'un k dans « sigma » (le texte de sa valeur : « 2 », « 2.5 »)."""
    return _cle_k(k)


def _k_fr(k: float) -> str:
    return f"{k:g}".replace(".", ",")


def _pct(x: float) -> str:
    return (f"{x:+.1f}").replace(".", ",")

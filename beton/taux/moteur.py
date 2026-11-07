# -*- coding: utf-8 -*-
"""Taux d'armature d'une poutre en béton armé — le MOTEUR, sans Streamlit, extrait de ``modules/taux_armature.py``
(la référence de parité : ``tests/test_taux_moteur.py``). Bibliothèque standard seule.

Le module est un QUANTITATIF : il ne vérifie rien. Il calcule le volume de béton d'une poutre, la masse de ses
armatures (familles de barres longitudinales, zones d'étriers) et leur rapport, le taux d'armature en kg/m³, avec une
majoration globale. Chaque expression est celle du module, dans le même ordre d'opérations (flottants identiques au
bit près).

ENTRÉE — une poutre (cm, m, %, Ø en mm ; une clé absente ou null prend le défaut de l'écran) :

    {"nom": "P1", "forme": "rectangulaire" | "talon_gauche" | "talon_droit" | "deux_talons",
     "enrobage": 3.0, "L": 6.0, "b": 30.0, "h": 60.0,                      # c (cm), portée (m), âme b × h (cm)
     "talon_gauche": {"b": 10.0, "h": 15.0}, "talon_droit": {"b": 10.0, "h": 15.0},   # (cm) lus selon la forme
     "longueurs": "pourcent" | "metre",          # « Saisie des longueurs de lit » (le stockage reste en %)
     "acier": "B500A" | "B500B" | "B500C",       # n'entre dans aucun calcul (comme l'écran)
     "majoration": 5.0,                          # « Majoration globale (%) »
     "familles": [{"role": "principal" | "renfort" | "lateral", "position": "ame_inf" | "ame_sup" | "talon_g_inf" |
                   "talon_g_sup" | "talon_d_inf" | "talon_d_sup" | "lat_g" | "lat_d" | "lat_2",
                   "n": 2, "d": 16, "retours": 0 | 1 | 2, "h_retour": 0.0, "de": 0.0, "a": 100.0}],  # de, a : % de L
     "types_etriers": [{"nom": "E1", "d": 8.0}],
     "zones": [{"nom": "Z1", "de": 0.0, "a": 33.0, "s": 15.0, "type": "E1"}]}                       # s : cm

SORTIE de ``calculer`` (JSON strict, nombres non arrondis — le site formate) :

- ``reference``, ``poutre`` (normalisée) ;
- ``geometrie`` : ``b``, ``h``, ``c``, ``L``, ``bg``, ``hg``, ``bd``, ``hd`` (talons CALCULÉS : 0 sans talon), ``aire_cm2``,
  ``polygone`` ([[x, y], …] cm, origine au milieu de l'âme, fibre inférieure y = 0), ``talons_ignores`` (deux talons) ;
- ``familles`` : pour chacune, dans l'ordre de la poutre : ``index``, ``code`` (INF1, SUP1, LAT1…), ``role``,
  ``position``, ``libelle_role``, ``libelle_position``, ``groupe`` (bas, haut, cote), ``rang`` (dans son groupe),
  ``couleur`` (celle du dessin de l'écran), ``n``, ``n_eff`` (2·n pour « Latérales 2 côtés »), ``d``, ``retours``,
  ``h_retour``, ``de``, ``a`` (%), ``de_m``, ``a_m`` (m), ``lit`` (rang du lit à sa position), ``L_active``, ``L_tot`` (m),
  ``m_lin`` (kg/m), ``volume`` (m³), ``masse_majoree`` (kg, la colonne « kg tot »), ``part`` (du volume d'acier),
  ``apport`` (masse majorée / Vc : la contribution de la famille au taux, kg/m³ — ajout du site), ``libelle``
  (« 3Ø16 », « 2 × 2Ø12 »), ``As`` (mm²), ``barres`` ([[x, y], …] cm : les centres que l'écran dessine), ``formules`` ;
- ``etriers`` : une ligne par zone CALCULÉE : ``index`` (de la zone), ``zone``, ``type``, ``d``, ``de``, ``a``, ``s``,
  ``L_zone``, ``n``, ``L_un`` (longueur d'un étrier, m), ``L_tot``, ``m_lin``, ``volume``, ``masse_majoree``, ``part``,
  ``apport``, ``quotient`` (L_zone / s, avant troncature), ``x1``, ``x2`` (De, À bornés à 0–100 %), ``s_m``,
  ``formules`` ; ``zones_ignorees`` : [{``index``, ``zone``, ``motif``}] ;
- ``quantites`` : ``aire_cm2``, ``Vc`` (m³), ``masse_beton`` (kg), ``V_longitudinales``, ``V_etriers``, ``V_acier`` (m³),
  ``masse_acier``, ``kg_m3`` (sans majoration : calculés par l'écran, non affichés), ``majoration``,
  ``masse_acier_majoree``, ``kg_m3_majore`` (les trois chiffres de l'écran), ``par_categorie`` (m³ : principal,
  renfort, lateral, etriers), ``masses_categories`` (kg majorés : le site les affiche, l'écran les calcule sans les
  montrer), ``formules`` ;
- ``avertissements`` : [{``code``, ``message``, ``grave``}] — ce que l'écran tait (défauts reproduits, zones ignorées,
  chevauchements…) ; ils ne changent aucun chiffre.

Une famille placée sur un talon que la géométrie CALCULÉE n'a pas (« deux talons » : défaut D2) lève ValueError
(« Talon gauche non défini. »), comme l'écran, qui s'arrête sur cette erreur.
"""
import copy
import math
from typing import Dict, List

from . import entree as EN
from . import geometrie as GE
from .entree import normaliser, nouveau_type_etrier, nouvelle_famille, nouvelle_poutre, nouvelle_zone, options_positions  # noqa: F401

REFERENCE = EN.REFERENCE
RHO_S = EN.MASSE_VOLUMIQUE_ACIER
RHO_C = EN.MASSE_VOLUMIQUE_BETON
CATEGORIES = ("principal", "renfort", "lateral", "etriers")


# ============================================================ les barres longitudinales (RebarFamily)
def n_effectif(f: dict) -> int:
    """Le nombre réel de barres (effective_n_bars) : « Latérales 2 côtés » compte n barres de chaque côté."""
    if f["position"] == "lat_2":
        return f["n"] * 2
    return f["n"]


def longueur_active(f: dict, L: float) -> float:
    """La partie droite d'une barre, en m (active_length_m) : de « De » à « À », bornés à 0–100 % ; nulle si À ≤ De."""
    x1 = max(0.0, min(100.0, f["de"]))
    x2 = max(0.0, min(100.0, f["a"]))
    if x2 <= x1:
        return 0.0
    return L * (x2 - x1) / 100.0


def longueur_totale(f: dict, L: float) -> float:
    """L'acier de la famille, en m (total_bar_length_m) : n_eff barres de longueur active, plus retours × hauteur."""
    L_active = longueur_active(f, L)
    if L_active <= 0.0 or f["n"] <= 0:
        return 0.0
    n_eff = n_effectif(f)
    L_hooks = (f["retours"] * max(f["h_retour"], 0.0) / 100.0) * n_eff
    return L_active * n_eff + L_hooks


def section_acier_mm2(f: dict) -> float:
    """La section d'acier de la famille (steel_area_mm2), en mm²."""
    return n_effectif(f) * math.pi * (f["d"] / 2.0) ** 2


def masse_lineique(d_mm: float) -> float:
    """La masse d'un mètre de barre Ø d (steel_linear_mass_kg_m), en kg/m."""
    r_m = d_mm / 1000.0 / 2.0
    area_m2 = math.pi * r_m ** 2
    return area_m2 * RHO_S


def volume_barre(longueur_m: float, d_mm: float) -> float:
    """Le volume d'acier, en m³ (rebar_volume_m3_from_length)."""
    r_m = d_mm / 1000.0 / 2.0
    area_m2 = math.pi * r_m ** 2
    return longueur_m * area_m2


# ============================================================ les étriers (StirrupZone, compute_stirrups_for_zones)
def longueur_etrier(d_mm: float, g: dict) -> float:
    """La longueur d'un étrier, en m (stirrup_length_m) : le contour de l'ÂME à l'enrobage, plus deux crochets de 8 Ø.
    Les talons n'y entrent pas."""
    c = g["c"]
    b_cm = g["b"] - 2 * c
    h_cm = g["h"] - 2 * c
    b_mm = b_cm * 10.0
    h_mm = h_cm * 10.0
    perim_mm = 2.0 * (b_mm + h_mm)
    hook_mm = 8.0 * d_mm * 2
    L_mm = perim_mm + hook_mm
    return L_mm / 1000.0


def _zones(p: dict, g: dict) -> (List[dict], List[dict]):
    """Les zones calculées et les zones ignorées. Le type d'une zone est cherché par son NOM (le dernier de ce nom,
    comme le dictionnaire du module) ; une zone sans type, ou dont À ≤ De, est ignorée sans un mot par l'écran."""
    par_nom = {t["nom"]: t for t in p["types_etriers"]}
    out, ignorees = [], []
    L = p["L"]
    for i, z in enumerate(p["zones"]):
        t = par_nom.get(z["type"])
        if t is None:
            ignorees.append({"index": i, "zone": z["nom"], "motif": "sans_type"})
            continue
        x1 = max(0.0, min(100.0, z["de"]))
        x2 = max(0.0, min(100.0, z["a"]))
        if x2 <= x1 or z["s"] <= 0.0:
            ignorees.append({"index": i, "zone": z["nom"], "motif": "longueur_nulle"})
            continue
        L_zone_m = L * (x2 - x1) / 100.0
        s_m = z["s"] / 100.0
        q = L_zone_m / s_m
        n = int(L_zone_m / s_m) + 1
        L_one = longueur_etrier(t["d"], g)
        total = n * L_one
        out.append({"index": i, "zone": z["nom"], "type": t["nom"], "d": t["d"], "de": z["de"], "a": z["a"], "s": z["s"],
                    "x1": x1, "x2": x2, "L_zone": L_zone_m, "s_m": s_m, "quotient": q, "n": n, "L_un": L_one, "L_tot": total})
    return out, ignorees


# ============================================================ couleurs, codes, lits (le dessin et le tableau de l'écran)
def _hex_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _degrade(debut: str, fin: str, n: int) -> List[str]:
    """Le dégradé de l'écran (_gradient) : n couleurs de ``debut`` à ``fin``, composantes tronquées."""
    if n <= 1:
        return [debut]
    s = _hex_rgb(debut)
    e = _hex_rgb(fin)
    out = []
    for i in range(n):
        t = i / (n - 1)
        rgb = tuple(int(s[k] + (e[k] - s[k]) * t) for k in range(3))
        out.append("#{:02x}{:02x}{:02x}".format(*rgb))
    return out


def couleurs(familles: List[dict]) -> List[str]:
    """La couleur de chaque famille (compute_family_colors_for_families) : un dégradé de rouges pour les lits du bas,
    de bleus pour ceux du haut, gris pour les latérales."""
    bas = [i for i, f in enumerate(familles) if GE.groupe(f["position"]) == "bas"]
    haut = [i for i, f in enumerate(familles) if GE.groupe(f["position"]) == "haut"]
    out = ["#666666"] * len(familles)
    rouges = _degrade("#ffcccc", "#b30000", max(len(bas), 1))
    bleus = _degrade("#cce0ff", "#0033cc", max(len(haut), 1))
    for k, i in enumerate(bas):
        out[i] = rouges[k]
    for k, i in enumerate(haut):
        out[i] = bleus[k]
    return out


def codes(familles: List[dict]) -> List[str]:
    """Les codes INF1, SUP1, LAT1… (compute_family_codes), numérotés par groupe dans l'ordre de la poutre."""
    compte = {"bas": 0, "haut": 0, "cote": 0}
    prefixe = {"bas": "INF", "haut": "SUP", "cote": "LAT"}
    out = []
    for f in familles:
        gr = GE.groupe(f["position"])
        compte[gr] += 1
        out.append(f"{prefixe[gr]}{compte[gr]}")
    return out


def lits(familles: List[dict]) -> List[int]:
    """Le rang du lit de chaque famille (compute_layer_indices) : 0, 1, 2… parmi les familles de MÊME position."""
    vus: Dict[str, int] = {}
    out = []
    for f in familles:
        out.append(vus.get(f["position"], 0))
        vus[f["position"]] = vus.get(f["position"], 0) + 1
    return out


def rangs(familles: List[dict]) -> List[int]:
    """Le rang de chaque famille dans son groupe (bas, haut, côté) : celui de sa couleur."""
    compte = {"bas": 0, "haut": 0, "cote": 0}
    out = []
    for f in familles:
        gr = GE.groupe(f["position"])
        out.append(compte[gr])
        compte[gr] += 1
    return out


def libelle(f: dict) -> str:
    """« 3Ø16 », « 2 × 2Ø12 » (latérales des deux côtés)."""
    return f"2 × {f['n']}Ø{f['d']}" if f["position"] == "lat_2" else f"{f['n']}Ø{f['d']}"


# ============================================================ le calcul
def _t(symbole, valeur, unite=""):
    return {"symbole": symbole, "valeur": valeur, "unite": unite}


def calculer(poutre: dict) -> dict:
    """Le quantitatif d'une poutre : la sortie décrite en tête de ce fichier. ``poutre`` est normalisée d'abord."""
    p = EN.normaliser(poutre)
    g = GE.geometrie(p)
    L = p["L"]
    maj = p["majoration"]
    fam = p["familles"]
    cols, cds, lts, rgs = couleurs(fam), codes(fam), lits(fam), rangs(fam)

    # les quantités (compute_quantities), dans l'ordre du module
    area_cm2 = GE.aire_cm2(g)
    Vc_m3 = area_cm2 / 10000.0 * L
    mass_concrete = Vc_m3 * RHO_C
    V_long = 0.0
    cat_vol = {"principal": 0.0, "renfort": 0.0, "lateral": 0.0, "etriers": 0.0}
    familles = []
    for i, f in enumerate(fam):
        barres = GE.centres(g, f["position"], f["n"], f["d"], lts[i])     # ValueError : talon non défini (D2)
        L_a = longueur_active(f, L)
        L_m = longueur_totale(f, L)
        V = volume_barre(L_m, f["d"])
        V_long += V
        cat_vol[f["role"]] += V
        m_lin = masse_lineique(f["d"])
        m_tot = L_m * m_lin * (1.0 + maj / 100.0)
        n_eff = n_effectif(f)
        familles.append({
            "index": i, "code": cds[i], "role": f["role"], "position": f["position"],
            "libelle_role": EN.ROLES[f["role"]][1], "libelle_position": EN.POSITIONS[f["position"]][1],
            "groupe": GE.groupe(f["position"]), "rang": rgs[i], "couleur": cols[i], "libelle": libelle(f),
            "n": f["n"], "n_eff": n_eff, "d": f["d"], "retours": f["retours"], "h_retour": f["h_retour"],
            "de": f["de"], "a": f["a"], "de_m": L * f["de"] / 100.0, "a_m": L * f["a"] / 100.0, "lit": lts[i],
            "L_active": L_a, "L_tot": L_m, "m_lin": m_lin, "volume": V, "masse_majoree": m_tot, "As": section_acier_mm2(f),
            "barres": [[x, y] for x, y in barres],
            "formules": [
                {"cle": "L_active", "termes": [_t("L", L, "m"), _t("de", f["de"], "%"), _t("a", f["a"], "%")], "resultat": L_a, "unite": "m"},
                {"cle": "n_eff", "termes": [_t("n", f["n"]), _t("k", 2 if f["position"] == "lat_2" else 1)], "resultat": n_eff, "unite": ""},
                {"cle": "L_tot", "termes": [_t("L_active", L_a, "m"), _t("n_eff", n_eff), _t("r", f["retours"]), _t("h_r", f["h_retour"], "cm")],
                 "resultat": L_m, "unite": "m"},
                {"cle": "m_lin", "termes": [_t("d", f["d"], "mm"), _t("rho", RHO_S, "kg/m³")], "resultat": m_lin, "unite": "kg/m"},
                {"cle": "masse", "termes": [_t("L_tot", L_m, "m"), _t("m_lin", m_lin, "kg/m"), _t("maj", maj, "%")], "resultat": m_tot, "unite": "kg"},
            ]})
    zones, ignorees = _zones(p, g)
    V_st = 0.0
    etriers = []
    for r in zones:
        V = volume_barre(r["L_tot"], r["d"])
        V_st += V
        cat_vol["etriers"] += V
        m_lin = masse_lineique(r["d"])
        m_tot = r["L_tot"] * m_lin * (1.0 + maj / 100.0)
        e = dict(r, volume=V, m_lin=m_lin, masse_majoree=m_tot)
        e["formules"] = [
            {"cle": "L_zone", "termes": [_t("L", L, "m"), _t("de", r["de"], "%"), _t("a", r["a"], "%")], "resultat": r["L_zone"], "unite": "m"},
            {"cle": "n", "termes": [_t("L_zone", r["L_zone"], "m"), _t("s", r["s"], "cm")], "resultat": r["n"], "unite": ""},
            {"cle": "L_un", "termes": [_t("b", g["b"], "cm"), _t("h", g["h"], "cm"), _t("c", g["c"], "cm"), _t("d", r["d"], "mm")],
             "resultat": r["L_un"], "unite": "m"},
            {"cle": "L_tot_etriers", "termes": [_t("n", r["n"]), _t("L_un", r["L_un"], "m")], "resultat": r["L_tot"], "unite": "m"},
            {"cle": "m_lin", "termes": [_t("d", r["d"], "mm"), _t("rho", RHO_S, "kg/m³")], "resultat": m_lin, "unite": "kg/m"},
            {"cle": "masse", "termes": [_t("L_tot", r["L_tot"], "m"), _t("m_lin", m_lin, "kg/m"), _t("maj", maj, "%")], "resultat": m_tot, "unite": "kg"},
        ]
        etriers.append(e)
    V_s_total = V_long + V_st
    mass_steel = V_s_total * RHO_S
    kg_per_m3 = mass_steel / Vc_m3 if Vc_m3 > 0 else 0.0
    # la majoration (show) : sur la masse totale, puis le taux
    mass_steel_majorated = mass_steel * (1.0 + maj / 100.0)
    kg_m3_majorated = mass_steel_majorated / Vc_m3 if Vc_m3 > 0 else 0.0
    for x in familles + etriers:
        x["part"] = x["volume"] / V_s_total if V_s_total > 0 else 0.0
        x["apport"] = x["masse_majoree"] / Vc_m3 if Vc_m3 > 0 else 0.0
    q = {"aire_cm2": area_cm2, "Vc": Vc_m3, "masse_beton": mass_concrete, "V_longitudinales": V_long, "V_etriers": V_st,
         "V_acier": V_s_total, "masse_acier": mass_steel, "kg_m3": kg_per_m3, "majoration": maj,
         "masse_acier_majoree": mass_steel_majorated, "kg_m3_majore": kg_m3_majorated, "par_categorie": cat_vol,
         "masses_categories": {k: v * RHO_S * (1.0 + maj / 100.0) for k, v in cat_vol.items()}}
    q["formules"] = [
        {"cle": "A_c", "termes": [_t("b", g["b"], "cm"), _t("h", g["h"], "cm"), _t("b_g", g["bg"], "cm"), _t("h_g", g["hg"], "cm"),
                                  _t("b_d", g["bd"], "cm"), _t("h_d", g["hd"], "cm")], "resultat": area_cm2, "unite": "cm²"},
        {"cle": "V_c", "termes": [_t("A_c", area_cm2, "cm²"), _t("L", L, "m")], "resultat": Vc_m3, "unite": "m³"},
        {"cle": "M_c", "termes": [_t("V_c", Vc_m3, "m³"), _t("rho", RHO_C, "kg/m³")], "resultat": mass_concrete, "unite": "kg"},
        {"cle": "V_s", "termes": [_t("V_long", V_long, "m³"), _t("V_et", V_st, "m³")], "resultat": V_s_total, "unite": "m³"},
        {"cle": "M_s", "termes": [_t("V_s", V_s_total, "m³"), _t("rho", RHO_S, "kg/m³")], "resultat": mass_steel, "unite": "kg"},
        {"cle": "M_s_maj", "termes": [_t("M_s", mass_steel, "kg"), _t("maj", maj, "%")], "resultat": mass_steel_majorated, "unite": "kg"},
        {"cle": "taux", "termes": [_t("M_s_maj", mass_steel_majorated, "kg"), _t("V_c", Vc_m3, "m³")], "resultat": kg_m3_majorated, "unite": "kg/m³"},
    ]
    geo = {"b": g["b"], "h": g["h"], "c": g["c"], "L": L, "bg": g["bg"], "hg": g["hg"], "bd": g["bd"], "hd": g["hd"],
           "aire_cm2": area_cm2, "polygone": [[x, y] for x, y in GE.polygone(g)], "talons_ignores": p["forme"] == "deux_talons"}
    return {"reference": REFERENCE, "poutre": p, "geometrie": geo, "familles": familles, "etriers": etriers,
            "zones_ignorees": ignorees, "quantites": q, "avertissements": _avertissements(p, familles, etriers, ignorees)}


# ============================================================ ce que l'écran tait
def _pc(x: float) -> str:
    return f"{x:g}".replace(".", ",")


def _avertissements(p: dict, familles: List[dict], etriers: List[dict], ignorees: List[dict]) -> List[dict]:
    """Ce que l'écran calcule sans le dire. Rien de tout cela ne change un chiffre : le site le MONTRE."""
    out = []

    def av(code, message, grave=False):
        out.append({"code": code, "message": message, "grave": grave})
    if p["forme"] == "deux_talons":
        av("talons_ignores", "Deux talons : le calcul de l'écran Streamlit ne les compte pas — le volume de béton, son poids "
           "et le dessin sont ceux de l'âme seule (défaut du module, reproduit ici).", True)
    if not p["familles"] and not p["zones"]:
        av("sans_armature", "Aucune armature : ajouter un lit, un renfort, des latérales ou des étriers.")
    for f in familles:
        if f["L_active"] <= 0.0:
            av("longueur_nulle", f"{f['code']} : « À » ({_pc(f['a'])} %) n'est pas au-delà de « De » ({_pc(f['de'])} %) — "
               "longueur nulle, la famille ne pèse rien.", True)
        if f["retours"] and f["h_retour"] == 0.0:
            av("retour_nul", f"{f['code']} : {f['retours']} retour{'s' if f['retours'] > 1 else ''} de 0 cm — saisir leur hauteur "
               "(l'écran Streamlit propose 0 aussi).")
    for z in ignorees:
        if z["motif"] == "sans_type":
            av("zone_sans_type", f"Zone {z['zone'] or '?'} : aucun type d'étrier — elle n'est pas comptée.", True)
        else:
            av("zone_nulle", f"Zone {z['zone'] or '?'} : « À » n'est pas au-delà de « De » — elle n'est pas comptée.", True)
    bg, _, bd, _ = GE.talons(p)
    if etriers and (bg > 0 or bd > 0):
        # D11 : stirrup_length_m prend le contour de l'ÂME ; un cadre ou une épingle de talon n'existe pas dans le module
        av("etriers_talon", "Étriers : le module compte le contour de l'âme seule — les cadres ou épingles du talon ne sont pas "
           "comptés (les ajouter par la majoration).")
    for e in etriers:
        frac = e["quotient"] - math.floor(e["quotient"])
        if frac > 1.0 - 1e-9:
            av("troncature_etriers", f"Zone {e['zone']} : L/s = {repr(e['quotient']).replace('.', ',')} (soit {math.ceil(e['quotient']):d} en exact) — "
               f"la troncature du module compte {e['n']} étriers au lieu de {e['n'] + 1}.", True)
    # chevauchements et parties sans étriers (zones calculées)
    for i, a in enumerate(etriers):
        for b in etriers[i + 1:]:
            lo, hi = max(a["x1"], b["x1"]), min(a["x2"], b["x2"])
            if hi > lo:
                av("zones_chevauchement", f"Zones {a['zone']} et {b['zone']} : elles se recouvrent de {_pc(lo)} à {_pc(hi)} % de la "
                   "portée — les étriers y sont comptés deux fois.")
    if etriers:
        bouts = sorted((e["x1"], e["x2"]) for e in etriers)
        trous, x = [], 0.0
        for lo, hi in bouts:
            if lo > x:
                trous.append((x, lo))
            x = max(x, hi)
        if x < 100.0:
            trous.append((x, 100.0))
        if trous:
            av("zones_trou", "Sans étriers : " + ", ".join(f"de {_pc(lo)} à {_pc(hi)} %" for lo, hi in trous) + " de la portée.")
    return out


# ============================================================ le choix du diamètre d'une famille, la note entière
def candidats_diametre(poutre: dict, index: int) -> List[dict]:
    """Les diamètres de l'écran pour la famille ``index`` : pour chacun, la poutre recalculée — masse de la famille
    (majorée), taux d'armature de la poutre, part de la famille, section d'acier. Triés par diamètre."""
    p = EN.normaliser(poutre)
    if not 0 <= index < len(p["familles"]):
        raise ValueError(f"famille {index} : la poutre en a {len(p['familles'])}")
    out = []
    for d in EN.DIAMETRES:
        q = copy.deepcopy(p)
        q["familles"][index]["d"] = d
        r = calculer(q)
        f = r["familles"][index]
        out.append({"d": d, "nom": f["libelle"], "masse_majoree": f["masse_majoree"], "kg_m3_majore": r["quantites"]["kg_m3_majore"],
                    "part": f["part"], "As": f["As"], "m_lin": f["m_lin"], "retenu": d == p["familles"][index]["d"]})
    return out


def synthese_note(resultats: List[dict]) -> dict:
    """La NOTE entière (ajout du site : l'écran n'a qu'une poutre) : la somme des volumes de béton et des masses d'acier
    majorées des poutres calculées, et leur rapport."""
    vc = sum(r["quantites"]["Vc"] for r in resultats)
    m = sum(r["quantites"]["masse_acier_majoree"] for r in resultats)
    return {"n": len(resultats), "Vc": vc, "masse_acier_majoree": m, "kg_m3_majore": m / vc if vc > 0 else 0.0}

# -*- coding: utf-8 -*-
"""Cornière ancrée — le MOTEUR, sans Streamlit. Extrait de l'écran ``modules/corniere.py``, qui reste la RÉFÉRENCE :
``tests/test_corniere_moteur.py`` rejoue l'écran avec un faux ``st`` et prouve, cas par cas, que ce moteur rend les
mêmes grandeurs AU BIT PRÈS, les mêmes ✅ / ❌ et les mêmes pourcentages. Méthode, provenance de chaque grandeur,
défauts constatés dans l'écran : ``docs/acier/CORNIERE_MOTEUR.md``.

Bibliothèque standard (math) ; le catalogue des cornières vient d'``acier.profiles`` (numpy, présent dans le
navigateur).

API
---
    nouvelle_corniere() -> dict                  une cornière aux valeurs par défaut de l'écran
    normaliser(corniere) -> dict                 copie complétée et contrôlée (ValueError hors du domaine de l'écran,
                                                 épaisseur ≥ aile, calcul qui déborde)
    calculer(t, V, charge, L, e, gamma_ELU, s, sig_lim_ELS, tau_lim_ELS, sig_lim_ELU) -> dict
                                                 les grandeurs de l'écran, mot pour mot
    verifier(corniere) -> dict                   le calcul complet, ses vérifications et ses formules
    candidats(corniere, familles=None) -> list   le sélecteur : les cornières du catalogue triées par masse
    optimal(corniere, familles=None) -> dict | None   la plus légère qui tient
    cornieres(familles=None), section_catalogue(nom)   le catalogue (acier/profiles)
    designation(nom) -> str                      « L 45x45x4.5 » → « L 45×45×4,5 »
    nom_catalogue(texte) -> str | None           une désignation tapée (« 80x80x8 », « L 80×80×8 ») → le nom au catalogue

ENTRÉE — un dict JSON ; une clé absente ou ``null`` prend le défaut de l'écran :
------
    {"nom": "Cornière 1",
     "standard": true,              le bouton « Utiliser une cornière standard » : true = du catalogue
     "profil": "L 100x100x10",      la cornière du catalogue (acier/profiles, familles L et LI)
     "A": 60.0, "B": 60.0,          largeurs des ailes en saisie libre (mm, 10 à 300)
     "t": 6.0,                      épaisseur en saisie libre (mm, 2 à 20)
     "charge": "lineaire",          « lineaire » (V en kN/m) ou « ponctuelle » (V en kN, réparti sur L)
     "V": 5.0,                      effort tranchant caractéristique (kN/m ou kN, ≥ 0)
     "L": 1.0,                      longueur d'influence d'une charge ponctuelle (m, 0,1 à 5)
     "e": 2.3,                      excentricité, distance charge – cornière (cm, ≥ 0)
     "gamma_ELU": 1.35,             coefficient global ELU (≥ 1)
     "s": 30.0,                     espacement des fixations (cm, ≥ 10)
     "bord": 2.0,                   distance de l'âme au bord du mur (cm, ≥ 0) — l'écran ne s'en sert pas
     "sig_lim_ELS": 155.0,          σ_lim,ELS (N/mm², ≥ 10)
     "tau_lim_ELS": 0.8,            τ_lim,ELS (N/mm², ≥ 0,1)
     "sig_lim_ELU": 186.0}          σ_eq,lim,ELU (N/mm², ≥ 10)

SORTIE de verifier — JSON-sérialisable, nombres non arrondis (le site formate) :
------
    {"reference": "modules/corniere.py",
     "corniere": {...},             l'entrée normalisée
     "section": {"source": "catalogue" | "saisie", "nom", "designation", "famille", "A", "B", "t", "r1", "r2",
                 "masse" (kg/m, None en saisie libre comme à l'écran), "aire" (cm²), "yc", "zc", "norme"},
     "grandeurs": {...},            toutes les grandeurs de l'écran (ci-dessous)
     "verifications": [<verification>, ...],   els_sigma, els_tau, elu, long_elu (l'ordre de l'écran)
     "verdicts": {"els", "elu", "long"},        les ✅ (true) / ❌ (false) de l'écran
     "taux_max", "statut": "ok" | "nok", "gouvernant": {"cle", "nom", "groupe", "taux"},
     "calcul": {"transversal": [<bloc>, ...], "longitudinal": [<bloc>, ...]},
     "avertissements": [{"code", "message"}, ...]}
        codes : « ancrages_non_verifies » et « longitudinal_provisoire » (toujours : ce que l'écran dit en légende),
        « excentricite_hors_aile » (e > B : un constat géométrique, aucun chiffre ne change)

  <verification> = {"cle", "nom", "groupe" ("transversal" | "longitudinal"), "etat_limite" ("ELS" | "ELU"),
                    "symbole", "valeur", "limite", "symbole_limite", "unite" ("N/mm²"),
                    "taux" (valeur / limite : le pourcentage de l'écran divisé par 100), "ok" (valeur ≤ limite),
                    "etat" ("ok" | "nok")}
  <bloc> = {"cle", "titre", "formules": [{"cle", "symbole", "termes": [{"symbole", "valeur", "unite"}, ...],
                                          "resultat", "unite"}, ...]}
     chaque résultat se retrouve à partir de ses seuls termes (le test le prouve) : le site écrit la phrase
     « symbole = expression = valeurs = résultat », il ne recalcule rien.

  grandeurs (mm, N/mm², kN, kN/m, kN·m, kN·m/m, N·mm/m) — les noms des variables de l'écran :
     modèle lame     b_trans t I_trans W_trans Az_trans
     efforts         V_ELS V_ELU e_m M_ELS M_ELU M_ELS_Nmm M_ELU_Nmm
     transversal     sigma_ELS sigma_ELU tau_ELS tau_ELU sigma_eq_ELS sigma_eq_ELU
     longitudinal    b_long I_long W_long Az_long Vloc_ELS Vloc_ELU Mlong_ELS Mlong_ELU sigma_long_ELS
                     sigma_long_ELU tau_long_ELS tau_long_ELU sigma_eq_long_ELS sigma_eq_long_ELU
     taux et verdicts U_sigma U_tau U_ELU U_long_ELU ok_els ok_elu ok_long

SORTIE de candidats — les cornières du catalogue, triées par masse (puis par nom), chacune calculée par
``calculer`` : [{"nom", "designation", "famille", "A", "B", "t", "masse", "aire", "taux" (le plus grand des
quatre), "ok" (les trois ✅ de l'écran), "aile" (e ≤ B : la charge tombe sur l'aile B), "retenue" (la cornière
retenue), "optimal" (la plus légère « ok » dont l'aile B reçoit la charge)}, ...]
"""
import math
from typing import Dict, List, Optional

from . import entree as E
from .entree import (CHARGES, FAMILLES, LINEAIRE, PONCTUELLE, REFERENCE, cornieres, designation,  # noqa: F401 — l'API
                     nom_catalogue, nouvelle_corniere, section_catalogue)

OK, NOK = "ok", "nok"
TRANSVERSAL, LONGITUDINAL = "transversal", "longitudinal"
B_TRANS = 1000.0                 # le modèle lame : une bande de 1 m de cornière (vérification par mètre courant)
T_MIN = 2.0                      # l'épaisseur la plus faible du domaine (saisie libre) : les contraintes y sont les plus fortes


def normaliser(c: Dict) -> Dict:
    """La cornière relue par l'entrée (défauts et domaine de l'écran, ``entree.normaliser``), puis un dernier refus :
    des valeurs si grandes que le calcul DÉBORDE (une contrainte infinie ou indéterminée ; l'écran Streamlit s'arrête
    alors sur une erreur Python) lèvent ValueError. Le contrôle est mené à l'épaisseur de la cornière et à la plus
    faible du domaine (2 mm) : les contraintes décroissent avec t, aucune cornière du sélecteur ne déborde donc."""
    c = E.normaliser(c)
    for t in (c["t"], T_MIN):
        try:
            g = _calcul_de(c, t)
            fini = all(math.isfinite(x) for x in g.values() if not isinstance(x, bool))
        except OverflowError:
            fini = False
        if not fini:
            raise ValueError("valeurs trop grandes : le calcul déborde (une contrainte devient infinie) — revoir V, e, s, L et γ_ELU")
    return c

# les quatre vérifications : (clé, nom, groupe, état limite) — l'ordre des taux de l'écran
VERIFICATIONS = (("els_sigma", "Contrainte équivalente", TRANSVERSAL, "ELS"),
                 ("els_tau", "Cisaillement", TRANSVERSAL, "ELS"),
                 ("elu", "Contrainte équivalente", TRANSVERSAL, "ELU"),
                 ("long_elu", "Contrainte équivalente", LONGITUDINAL, "ELU"))


# ------------------------------------------------------------------ le calcul de l'écran, mot pour mot
def calculer(t, V_char, charge, L, e, gamma_ELU, s, sig_lim_ELS, tau_lim_ELS, sig_lim_ELU) -> Dict:
    """Les grandeurs de l'écran ``modules/corniere.py`` (« CALCULS utilitaires » à « Taux »), chaque expression
    reprise DANS LE MÊME ORDRE D'OPÉRATIONS : les flottants sont identiques au bit près. ``t`` en mm, ``V_char`` en
    kN/m (linéaire) ou kN (ponctuelle), ``L`` en m, ``e`` et ``s`` en cm, les limites en N/mm²."""
    # Modèle lame : b = 1000 mm (vérif par mètre courant)
    b_trans_mm = B_TRANS
    t_mm = float(t)
    I_trans = b_trans_mm * t_mm**3 / 12.0
    W_trans = b_trans_mm * t_mm**2 / 6.0
    Az_trans = b_trans_mm * t_mm
    # Efforts (ELS et ELU) ; une charge ponctuelle est ramenée par mètre de cornière sur la longueur L
    if charge == LINEAIRE:
        V_ELS = V_char
        V_ELU = V_char * gamma_ELU
    else:
        V_ELS = V_char / L
        V_ELU = (V_char * gamma_ELU) / L
    # Moment transversal M = V * e (par mètre) → N·mm/mm
    e_m = e / 100.0
    M_ELS_Nmm = V_ELS * e_m * 1e6
    M_ELU_Nmm = V_ELU * e_m * 1e6
    # Contraintes transversales
    sigma_ELS = M_ELS_Nmm / W_trans if W_trans > 0 else 0.0
    sigma_ELU = M_ELU_Nmm / W_trans if W_trans > 0 else 0.0
    tau_ELS = (V_ELS * 1e3) / Az_trans if Az_trans > 0 else 0.0
    tau_ELU = (V_ELU * 1e3) / Az_trans if Az_trans > 0 else 0.0
    sigma_eq_ELS = math.sqrt(sigma_ELS**2 + 3.0 * tau_ELS**2)
    sigma_eq_ELU = math.sqrt(sigma_ELU**2 + 3.0 * tau_ELU**2)
    # Longitudinal (entre fixations) : un panneau de largeur s
    b_long_mm = s * 10.0
    I_long = b_long_mm * t_mm**3 / 12.0
    W_long = b_long_mm * t_mm**2 / 6.0
    Az_long = b_long_mm * t_mm
    Vloc_ELS = V_ELS * (s / 100.0)
    Vloc_ELU = V_ELU * (s / 100.0)
    Mlong_ELS_kNm = Vloc_ELS * e_m
    Mlong_ELU_kNm = Vloc_ELU * e_m
    sigma_long_ELS = (Mlong_ELS_kNm * 1e6) / W_long if W_long > 0 else 0.0
    sigma_long_ELU = (Mlong_ELU_kNm * 1e6) / W_long if W_long > 0 else 0.0
    tau_long_ELS = (Vloc_ELS * 1e3) / Az_long if Az_long > 0 else 0.0
    tau_long_ELU = (Vloc_ELU * 1e3) / Az_long if Az_long > 0 else 0.0
    sigma_eq_long_ELS = math.sqrt(sigma_long_ELS**2 + 3.0 * tau_long_ELS**2)
    sigma_eq_long_ELU = math.sqrt(sigma_long_ELU**2 + 3.0 * tau_long_ELU**2)
    # Taux
    U_sigma = sigma_eq_ELS / sig_lim_ELS if sig_lim_ELS > 0 else 0.0
    U_tau = tau_ELS / tau_lim_ELS if tau_lim_ELS > 0 else 0.0
    U_ELU = sigma_eq_ELU / sig_lim_ELU if sig_lim_ELU > 0 else 0.0
    U_long_ELU = sigma_eq_long_ELU / sig_lim_ELU if sig_lim_ELU > 0 else 0.0
    # les ✅ / ❌ de l'écran
    ok_els = sigma_eq_ELS <= sig_lim_ELS and tau_ELS <= tau_lim_ELS
    ok_elu = sigma_eq_ELU <= sig_lim_ELU
    ok_long = sigma_eq_long_ELU <= sig_lim_ELU
    return {"b_trans": b_trans_mm, "t": t_mm, "I_trans": I_trans, "W_trans": W_trans, "Az_trans": Az_trans,
            "V_ELS": V_ELS, "V_ELU": V_ELU, "e_m": e_m, "M_ELS": V_ELS * e_m, "M_ELU": V_ELU * e_m,
            "M_ELS_Nmm": M_ELS_Nmm, "M_ELU_Nmm": M_ELU_Nmm,
            "sigma_ELS": sigma_ELS, "sigma_ELU": sigma_ELU, "tau_ELS": tau_ELS, "tau_ELU": tau_ELU,
            "sigma_eq_ELS": sigma_eq_ELS, "sigma_eq_ELU": sigma_eq_ELU,
            "b_long": b_long_mm, "I_long": I_long, "W_long": W_long, "Az_long": Az_long,
            "Vloc_ELS": Vloc_ELS, "Vloc_ELU": Vloc_ELU, "Mlong_ELS": Mlong_ELS_kNm, "Mlong_ELU": Mlong_ELU_kNm,
            "sigma_long_ELS": sigma_long_ELS, "sigma_long_ELU": sigma_long_ELU,
            "tau_long_ELS": tau_long_ELS, "tau_long_ELU": tau_long_ELU,
            "sigma_eq_long_ELS": sigma_eq_long_ELS, "sigma_eq_long_ELU": sigma_eq_long_ELU,
            "U_sigma": U_sigma, "U_tau": U_tau, "U_ELU": U_ELU, "U_long_ELU": U_long_ELU,
            "ok_els": ok_els, "ok_elu": ok_elu, "ok_long": ok_long}


def _calcul_de(c: Dict, t: float) -> Dict:
    return calculer(t, c["V"], c["charge"], c["L"], c["e"], c["gamma_ELU"], c["s"], c["sig_lim_ELS"], c["tau_lim_ELS"],
                    c["sig_lim_ELU"])


# ------------------------------------------------------------------ la section
def section(c: Dict) -> Dict:
    """La cornière calculée : celle du catalogue (``standard``) ou la saisie libre A × B × t (sans masse, comme à
    l'écran, qui n'en affiche pas en saisie libre)."""
    if c["standard"]:
        p = section_catalogue(c["profil"])
        return {"source": "catalogue", "nom": p["nom"], "designation": designation(p["nom"]), "famille": p["famille"],
                "A": p["h"], "B": p["b"], "t": p["t"], "r1": p.get("r1") or 0.0, "r2": p.get("r2") or 0.0,
                "masse": p["masse"], "aire": p["A"], "yc": p.get("yc"), "zc": p.get("zc"), "norme": "EN 10056-1"}
    return {"source": "saisie", "nom": None, "designation": f"{_g(c['A'])} × {_g(c['B'])} × {_g(c['t'])}",
            "famille": None, "A": c["A"], "B": c["B"], "t": c["t"], "r1": 0.0, "r2": 0.0, "masse": None, "aire": None,
            "yc": None, "zc": None, "norme": None}


def _g(x: float) -> str:
    return f"{x:g}".replace(".", ",")


# ------------------------------------------------------------------ les formules, pour le détail du calcul
def _f(cle: str, symbole: str, termes, resultat: float, unite: str) -> Dict:
    return {"cle": cle, "symbole": symbole, "termes": [{"symbole": s, "valeur": v, "unite": u} for s, v, u in termes],
            "resultat": resultat, "unite": unite}


def _blocs(c: Dict, g: Dict) -> Dict[str, List[Dict]]:
    """Les formules de l'écran, groupées comme une note faite à la main ; chaque résultat est une grandeur de
    ``calculer`` (aucun calcul nouveau)."""
    V, gam, e, s, L = c["V"], c["gamma_ELU"], c["e"], c["s"], c["L"]
    b, t = g["b_trans"], g["t"]
    if c["charge"] == LINEAIRE:
        eff = [_f("V_ELS_lin", "V_ELS", [("V", V, "kN/m")], g["V_ELS"], "kN/m"),
               _f("V_ELU_lin", "V_ELU", [("γ_ELU", gam, ""), ("V", V, "kN/m")], g["V_ELU"], "kN/m")]
    else:
        eff = [_f("V_ELS_ponct", "V_ELS", [("V", V, "kN"), ("L", L, "m")], g["V_ELS"], "kN/m"),
               _f("V_ELU_ponct", "V_ELU", [("γ_ELU", gam, ""), ("V", V, "kN"), ("L", L, "m")], g["V_ELU"], "kN/m")]
    eff += [_f("M", "M_ELS", [("V_ELS", g["V_ELS"], "kN/m"), ("e", e, "cm")], g["M_ELS"], "kN·m/m"),
            _f("M", "M_ELU", [("V_ELU", g["V_ELU"], "kN/m"), ("e", e, "cm")], g["M_ELU"], "kN·m/m")]
    sec = [_f("I", "I", [("b", b, "mm"), ("t", t, "mm")], g["I_trans"], "mm⁴"),
           _f("W", "W", [("b", b, "mm"), ("t", t, "mm")], g["W_trans"], "mm³"),
           _f("Az", "A_z", [("b", b, "mm"), ("t", t, "mm")], g["Az_trans"], "mm²")]

    def contraintes(el, M, Vx, sig, tau, eq):
        return [_f("sigma", f"σ_{el}", [("M", M, "kN·m/m"), ("W", g["W_trans"], "mm³")], sig, "N/mm²"),
                _f("tau", f"τ_{el}", [("V", Vx, "kN/m"), ("A_z", g["Az_trans"], "mm²")], tau, "N/mm²"),
                _f("sigma_eq", f"σ_eq,{el}", [("σ", sig, "N/mm²"), ("τ", tau, "N/mm²")], eq, "N/mm²")]
    els = contraintes("ELS", g["M_ELS"], g["V_ELS"], g["sigma_ELS"], g["tau_ELS"], g["sigma_eq_ELS"])
    els += [_f("taux", "U_σ", [("σ_eq,ELS", g["sigma_eq_ELS"], "N/mm²"), ("σ_lim,ELS", c["sig_lim_ELS"], "N/mm²")], g["U_sigma"], ""),
            _f("taux", "U_τ", [("τ_ELS", g["tau_ELS"], "N/mm²"), ("τ_lim,ELS", c["tau_lim_ELS"], "N/mm²")], g["U_tau"], "")]
    elu = contraintes("ELU", g["M_ELU"], g["V_ELU"], g["sigma_ELU"], g["tau_ELU"], g["sigma_eq_ELU"])
    elu += [_f("taux", "U_ELU", [("σ_eq,ELU", g["sigma_eq_ELU"], "N/mm²"), ("σ_eq,lim,ELU", c["sig_lim_ELU"], "N/mm²")], g["U_ELU"], "")]
    bl = g["b_long"]
    pan = [_f("b_long", "b", [("s", s, "cm")], bl, "mm"),
           _f("I", "I_long", [("b", bl, "mm"), ("t", t, "mm")], g["I_long"], "mm⁴"),
           _f("W", "W_long", [("b", bl, "mm"), ("t", t, "mm")], g["W_long"], "mm³"),
           _f("Az", "A_z,long", [("b", bl, "mm"), ("t", t, "mm")], g["Az_long"], "mm²")]
    eff_l = [_f("Vloc", "V_loc,ELS", [("V_ELS", g["V_ELS"], "kN/m"), ("s", s, "cm")], g["Vloc_ELS"], "kN"),
             _f("Vloc", "V_loc,ELU", [("V_ELU", g["V_ELU"], "kN/m"), ("s", s, "cm")], g["Vloc_ELU"], "kN"),
             _f("Mlong", "M_long,ELS", [("V_loc,ELS", g["Vloc_ELS"], "kN"), ("e", e, "cm")], g["Mlong_ELS"], "kN·m"),
             _f("Mlong", "M_long,ELU", [("V_loc,ELU", g["Vloc_ELU"], "kN"), ("e", e, "cm")], g["Mlong_ELU"], "kN·m")]

    def contraintes_long(el, M, Vx, sig, tau, eq):
        return [_f("sigma_long", f"σ_long,{el}", [("M_long", M, "kN·m"), ("W_long", g["W_long"], "mm³")], sig, "N/mm²"),
                _f("tau_long", f"τ_long,{el}", [("V_loc", Vx, "kN"), ("A_z,long", g["Az_long"], "mm²")], tau, "N/mm²"),
                _f("sigma_eq", f"σ_eq,long,{el}", [("σ", sig, "N/mm²"), ("τ", tau, "N/mm²")], eq, "N/mm²")]
    lng = contraintes_long("ELS", g["Mlong_ELS"], g["Vloc_ELS"], g["sigma_long_ELS"], g["tau_long_ELS"], g["sigma_eq_long_ELS"])
    lng += contraintes_long("ELU", g["Mlong_ELU"], g["Vloc_ELU"], g["sigma_long_ELU"], g["tau_long_ELU"], g["sigma_eq_long_ELU"])
    lng += [_f("taux", "U_long,ELU", [("σ_eq,long,ELU", g["sigma_eq_long_ELU"], "N/mm²"), ("σ_eq,lim,ELU", c["sig_lim_ELU"], "N/mm²")],
               g["U_long_ELU"], "")]
    return {TRANSVERSAL: [{"cle": "efforts", "titre": "Efforts par mètre courant", "formules": eff},
                          {"cle": "section", "titre": "Section · modèle lame (b = 1 000 mm)", "formules": sec},
                          {"cle": "els", "titre": "Contraintes à l'ELS", "formules": els},
                          {"cle": "elu", "titre": "Contraintes à l'ELU", "formules": elu}],
            LONGITUDINAL: [{"cle": "panneau", "titre": "Panneau entre fixations (largeur s)", "formules": pan},
                           {"cle": "efforts", "titre": "Efforts du panneau", "formules": eff_l},
                           {"cle": "contraintes", "titre": "Contraintes", "formules": lng}]}


# ------------------------------------------------------------------ la vérification
def _verifications(c: Dict, g: Dict) -> List[Dict]:
    lignes = {"els_sigma": (g["sigma_eq_ELS"], c["sig_lim_ELS"], g["U_sigma"], "σ_eq,ELS", "σ_lim,ELS"),
              "els_tau": (g["tau_ELS"], c["tau_lim_ELS"], g["U_tau"], "τ_ELS", "τ_lim,ELS"),
              "elu": (g["sigma_eq_ELU"], c["sig_lim_ELU"], g["U_ELU"], "σ_eq,ELU", "σ_eq,lim,ELU"),
              "long_elu": (g["sigma_eq_long_ELU"], c["sig_lim_ELU"], g["U_long_ELU"], "σ_eq,long,ELU", "σ_eq,lim,ELU")}
    out = []
    for cle, nom, groupe, el in VERIFICATIONS:
        val, lim, taux, sym, sym_l = lignes[cle]
        ok = val <= lim
        out.append({"cle": cle, "nom": nom, "groupe": groupe, "etat_limite": el, "symbole": sym, "valeur": val,
                    "limite": lim, "symbole_limite": sym_l, "unite": "N/mm²", "taux": taux, "ok": ok,
                    "etat": OK if ok else NOK})
    return out


def verifier(c: Dict) -> Dict:
    """Le calcul de l'écran pour une cornière : grandeurs, quatre vérifications, verdicts, formules."""
    c = normaliser(c)
    sec = section(c)
    g = _calcul_de(c, sec["t"])
    ver = _verifications(c, g)
    gov = max(ver, key=lambda v: v["taux"])            # à égalité, la première dans l'ordre de l'écran
    av = [{"code": "ancrages_non_verifies",
           "message": "Ancrages non vérifiés, comme dans l'écran Streamlit : efforts par fixation, entraxes, béton et bords "
                      "à justifier selon l'ETA."},
          {"code": "longitudinal_provisoire",
           "message": "Côté non sûr : la vérification « longitudinale » de l'écran Streamlit redonne le calcul transversal "
                      "(s se simplifie) ; la flexion entre fixations (q·s²/8) n'est vérifiée nulle part."}]
    if c["e"] * 10.0 > sec["B"]:
        # un CONSTAT géométrique (aucun chiffre ne change) : l'écran ne compare jamais e à l'aile B
        av.append({"code": "excentricite_hors_aile",
                   "message": f"La charge tombe au-delà de l'aile B : e = {_g(c['e'] * 10.0)} mm > B = {_g(sec['B'])} mm "
                              "(e compté depuis le dos de l'aile fixée). L'écran Streamlit ne le contrôle pas ; le calcul "
                              "est mené quand même."})
    return {"reference": REFERENCE, "corniere": c, "section": sec, "grandeurs": g, "verifications": ver,
            "verdicts": {"els": g["ok_els"], "elu": g["ok_elu"], "long": g["ok_long"]},
            "taux_max": gov["taux"], "statut": OK if all(v["ok"] for v in ver) else NOK,
            "gouvernant": {"cle": gov["cle"], "nom": gov["nom"], "groupe": gov["groupe"], "etat_limite": gov["etat_limite"],
                           "taux": gov["taux"]},
            "calcul": _blocs(c, g), "avertissements": av}


# ------------------------------------------------------------------ le sélecteur
def candidats(c: Dict, familles=None) -> List[Dict]:
    """Les cornières du catalogue (familles L et LI par défaut), TRIÉES PAR MASSE, chacune calculée avec les
    charges et les critères de ``c`` ; ``optimal`` marque la plus légère qui tient (les trois ✅ de l'écran) ET dont
    l'aile B reçoit la charge (e ≤ B : l'écran ne le contrôle pas, mais proposer une cornière où la charge tombe dans
    le vide serait une faute ; ``aile`` le dit pour chaque ligne)."""
    c = normaliser(c)
    out = []
    for p in cornieres(familles):
        g = _calcul_de(c, p["t"])
        ok = g["ok_els"] and g["ok_elu"] and g["ok_long"]
        out.append({"nom": p["nom"], "designation": designation(p["nom"]), "famille": p["famille"], "A": p["h"], "B": p["b"],
                    "t": p["t"], "r1": p.get("r1") or 0.0, "masse": p["masse"], "aire": p["A"],
                    "taux": max(g["U_sigma"], g["U_tau"], g["U_ELU"], g["U_long_ELU"]), "ok": ok,
                    "aile": c["e"] * 10.0 <= p["b"],
                    "retenue": bool(c["standard"] and p["nom"] == c["profil"]), "optimal": False})
    premier = next((x for x in out if x["ok"] and x["aile"]), None)
    if premier is not None:
        premier["optimal"] = True
    return out


def optimal(c: Dict, familles=None) -> Optional[Dict]:
    """La cornière la plus légère qui tient (None si aucune du catalogue ne tient)."""
    return next((x for x in candidats(c, familles) if x["optimal"]), None)

# -*- coding: utf-8 -*-
"""Le moteur « Choix de profilé » : le calcul de l'écran Streamlit ``modules/choix_profile.py`` (``show``), sans
Streamlit ni pandas. Rien n'y est inventé : chaque grandeur est celle du module, calculée par les fonctions recopiées
(``calcul``) dans le même ordre ; ce que le moteur AJOUTE est nommé comme tel (le plus léger qui tient, le gouvernant,
les formules, les avertissements).

    verifier(element) -> dict        # le calcul d'un élément (entree.normaliser d'abord)
    candidats(element, familles)     # les lignes du sélecteur : une autre sélection de familles, triée par masse

Sortie de ``verifier`` (JSON strict, nombres non arrondis sauf les colonnes arrondies de l'écran) :

- ``reference`` : la référence de parité ; ``element`` : l'élément normalisé ;
- ``acier`` : ``nuance``, ``fy`` (int, comme ``int(acier[1:])``) ;
- ``efforts`` : ``N``, ``My``, ``Mz``, ``Vy``, ``Vz``, ``T`` — ceux que le calcul reçoit (0 quand la case est décochée) ;
- ``flambement`` : ``actif`` (la case), ``calcule`` (case cochée ET N > 0), ``Lcr_y``, ``Lcr_z``, ``courbe``,
  ``alpha`` (ceux du calcul : 0 et 0,34 case décochée) ; ``deversement`` : ``actif``, ``L_LT`` (jamais utilisé) ;
- ``familles_disponibles``, ``familles`` (cochées ; toutes si aucune), ``Iy_min`` ;
- ``lignes`` : le tableau de l'écran TRIÉ comme lui (``Utilisation [%]`` croissante), une ligne par profilé retenu par
  les filtres : ``nom``, ``famille``, ``masse`` (kg/m), ``taux`` (η global exact), ``taux_section``,
  ``taux_flambement`` (None sans flambement calculé), les colonnes ARRONDIES de l'écran ``utilisation``
  (= round(η·100, 3)), ``u_section`` (2 décimales), ``u_flamb``, les propriétés ``h`` (int, comme l'écran), ``Wel_y``,
  ``Wel_z``, ``Avy``, ``Avz``, ``A``, ``Iy``, ``Iz``, et ``tient`` (``utilisation`` ≤ 100 : le critère du module) ;
- ``n_profils``, ``n_tiennent`` ;
- ``optimal`` : le « Profilé optimal » du module (le plus CHARGÉ dont l'utilisation arrondie ≤ 100 %, le premier
  du tri à égalité ; None si aucun) ; ``plus_leger`` : AJOUT du moteur, le plus léger dont η ≤ 1 (exact) ;
- ``retenu`` : le profilé de la colonne de droite (le choix de l'élément s'il est dans le tableau, sinon l'index par
  défaut de l'écran : l'optimal, ou la première ligne) ; ``retenu_auto`` ; ``choix_hors_tableau`` (le choix de
  l'élément quand il n'est plus dans le tableau, sinon None) ;
- ``profil`` : la fiche du retenu — ``nom``, ``famille``, ``dimensions`` [[libellé, valeur]] et ``proprietes`` (les deux
  tableaux de l'écran), ``resistances`` (``N_Rd`` kN, ``My_Rd`` kN·m, ``Mz_Rd`` kN·m, ``Vy_Rd`` kN, ``Vz_Rd`` kN ;
  None quand la donnée manque), ``ratios`` (``rN``, ``rMy``, ``rMz``, ``rVy``, ``rVz``, ``util_NM``, ``util`` du
  module), ``axes`` (flambement par axe : ``axe``, ``Lcr``, ``I``, ``Ncr`` kN, ``lambdabar``, ``phi``, ``chi``,
  ``Nb_Rd`` kN, ``Nb_Rd_N`` en N comme le module), ``Nb_Rd`` (le plus petit, kN), ``taux_flambement``, ``taux``
  (η global), ``formules`` ;
- ``verifications`` : une par taux du retenu (``cle`` N, My, Mz, Vz, Vy, NM, section, flambement ; ``nom``,
  ``symbole``, ``valeur``, ``limite``, ``unite``, ``taux``, ``actif`` (l'effort est saisi), ``ecran`` (le module
  écrit ce taux), ``resistance_absente`` (la résistance manque : le module prend un taux nul), ``formules`` : les
  clés de ``profil.formules`` qui y mènent) ;
- ``taux_max`` (η global du retenu), ``statut`` (``ok`` : η ≤ 1 ; ``nok`` ; ``vide`` : aucun profilé),
  ``gouvernant`` (``cle``, ``nom``, ``taux`` : le premier maximum dans l'ordre du module — NM, Vy, Vz, puis le
  flambement s'il dépasse strictement) ;
- ``avertissements`` : [{``code``, ``message``, ``module`` (le texte que l'écran affiche, None si l'écran se tait)}] ;
  ``classe_4`` (AJOUT : une paroi du retenu au-delà de la classe 3, que l'écran ne classe pas — défaut D5).

Chaque formule (``profil.formules``) : ``cle``, ``symbole``, ``termes`` [{symbole, valeur, unite}] dans l'ordre de la
formule, ``resultat``, ``unite`` — le site écrit la phrase, il ne recalcule rien.
"""
import math
from typing import Dict, List, Optional, Sequence

from . import calcul as CA
from . import entree as EN

REFERENCE = "modules/choix_profile.py (écran Streamlit « Choix de profilé métallique optimisé (EC3) »)"

# les taux du profilé retenu, dans l'ordre de l'écran (colonne de droite, « Utilisations (ratios) »)
NOMS = {"N": "Effort normal", "My": "Flexion autour de y", "Mz": "Flexion autour de z", "Vz": "Cisaillement selon z",
        "Vy": "Cisaillement selon y", "NM": "Interaction N + M", "section": "Section", "flambement": "Flambement"}
SYMBOLES = {"N": "η_N", "My": "η_My", "Mz": "η_Mz", "Vz": "η_Vz", "Vy": "η_Vy", "NM": "η_NM", "section": "η_section",
            "flambement": "η_b"}

# ce que l'écran affiche quand une donnée manque (colonne de droite) ou quand rien ne tient (colonne de gauche)
TEXTES_MODULE = {
    "N_Rd_absent": "N_Rd : A manquant dans la base (champ `A`).",
    "My_Rd_absent": "My,Rd : Wel_y manquant.",
    "Mz_Rd_absent": "Mz,Rd : Wel_z manquant dans la base (champ `Welz`/`Wel_z`).",
    "Vz_Rd_absent": "Vz,Rd : Avz manquant dans la base (champ `Avz`).",
    "Vy_Rd_absent": "Vy,Rd : Avy manquant dans la base (champ `Avy`).",
    "flambement_impossible": "Impossible de calculer le flambement : vérifier A, Iy/Iz et Lcr,y/Lcr,z.",
    "deversement_non_calcule": ("LTB complet à activer dès que ta base JSON contient Iz, It, Iw et qu’on implémente Mcr + χLT. "
                                "(La structure UI est prête.)"),
    "aucun_profil": "Aucun profilé ne satisfait aux critères.",
}
LEGENDE_LTB = ("LTB complet nécessite : Iz, It, Iw + choix C1 / diagramme de moments. "
               "(Ajoutable dès que ta base JSON les contient.)")
LEGENDE_FLAMBEMENT = "N_b,Rd = χ·A·fy/γM1 (axe le plus défavorable)."

# EN 1993-1-1 tableau 3.1 : f_y des aciers laminés pour 40 mm < t ≤ 80 mm. L'écran prend TOUJOURS celui de t ≤ 40 mm
# (``int(acier[1:])``) : pour une paroi plus épaisse (HD, HL), ses résistances sont surestimées. Le moteur le DIT
# (avertissement ``fy_epaisseur``) sans rien recalculer : la parité avec l'écran reste entière.
EPAISSEUR_LIMITE, EPAISSEUR_MAX_TABLEAU = 40.0, 80.0         # mm
FY_EPAIS = {"S235": 215, "S275": 255, "S355": 335}            # N/mm², 40 < t ≤ 80 mm
# EN 1993-1-1 tableau 5.2 : les limites de la classe 3 (au-delà, classe 4 : la section efficace A_eff, W_eff remplace la
# section brute). L'écran ne classe pas les sections (défaut D5) ; le moteur DIT quand une paroi du retenu dépasse, sans
# rien recalculer. Contrôle simple et du côté sûr : la semelle en console COMPRIMÉE (14ε) dès qu'un moment ou une
# compression est saisi ; l'âme en compression PURE (42ε) dès que N_Ed > 0 (sous N + M, la limite réelle est plus haute).
LIMITE_SEMELLE_CL3, LIMITE_AME_COMPRESSION_CL3 = 14.0, 42.0    # × ε = √(235 / f_y)


# ------------------------------------------------------------------ le tableau de l'écran
def _efforts(el: Dict) -> Dict[str, float]:
    """Les efforts que le calcul reçoit : un effort « optionnel » vaut 0.0 quand sa case est décochée."""
    return {"N": el["N"] if el["avec_N"] else 0.0, "My": el["My"], "Mz": el["Mz"] if el["avec_Mz"] else 0.0,
            "Vy": el["Vy"] if el["avec_Vy"] else 0.0, "Vz": el["Vz"], "T": el["T"] if el["avec_T"] else 0.0}


def _flambement(el: Dict) -> Dict:
    if el["flambement"]:
        return {"Lcr_y": el["Lcr_y"], "Lcr_z": el["Lcr_z"], "alpha": EN.ALPHA[el["courbe"]], "courbe": el["courbe"]}
    return {"Lcr_y": 0.0, "Lcr_z": 0.0, "alpha": EN.ALPHA_SANS_FLAMBEMENT, "courbe": None}


def _ligne(nom: str, prof: Dict, fy: int, el: Dict, ef: Dict, fl: Dict) -> Dict:
    """Une ligne du tableau : le corps de la boucle de ``show`` pour un profilé."""
    gamma_M0, gamma_M1 = el["gamma_M0"], el["gamma_M1"]
    NEd = ef["N"]
    resist = CA.resistances_section(prof, fy=fy, gamma_M0=gamma_M0)
    ratios = CA.taux_section(NEd, ef["My"], ef["Mz"], ef["Vy"], ef["Vz"], resist)
    util_buck = None
    if el["flambement"] and NEd > 0:
        A, Iy, Iz = prof.get("A"), prof.get("Iy"), prof.get("Iz")
        buck_y = CA.flambement_chi(A, Iy, fl["Lcr_y"], fy=fy, alpha=fl["alpha"]) if (fl["Lcr_y"] > 0 and Iy) else None
        buck_z = CA.flambement_chi(A, Iz, fl["Lcr_z"], fy=fy, alpha=fl["alpha"]) if (fl["Lcr_z"] > 0 and Iz) else None
        candidates = []
        for buck in (buck_y, buck_z):
            if buck:
                Nb_Rd = buck["chi"] * (A * 1e2) * fy / gamma_M1
                candidates.append(Nb_Rd)
        if candidates:
            Nb_Rd_min = min(candidates)
            util_buck = (NEd * 1e3) / Nb_Rd_min
    util_global = ratios["util"]
    if util_buck is not None:
        util_global = max(util_global, util_buck)
    utilisation = round(util_global * 100, 3)
    return {"nom": nom, "famille": prof["type"], "masse": prof.get("Poids"),
            "taux": util_global, "taux_section": ratios["util"], "taux_flambement": util_buck,
            "utilisation": utilisation, "u_section": round(ratios["util"] * 100, 2),
            "u_flamb": round(util_buck * 100, 2) if util_buck is not None else None,
            "h": int(prof["h"]) if prof.get("h") else None, "Wel_y": prof.get("Wel_y"), "Wel_z": prof.get("Wel_z"),
            "Avy": prof.get("Avy"), "Avz": prof.get("Avz"), "A": prof.get("A"), "Iy": prof.get("Iy"), "Iz": prof.get("Iz"),
            "tient": utilisation <= 100.0}


def _filtres(el: Dict, familles: Optional[Sequence[str]]) -> Dict[str, Dict]:
    """``profils_filtres`` puis le filtre « Iy min. » : les profilés des familles (toutes si aucune)."""
    profiles = CA._profils()
    filtres = {k: v for k, v in profiles.items() if v["type"] in familles} if familles else profiles
    Iv_min = el["Iy_min"]
    return {k: v for k, v in filtres.items() if not (v.get("Iy") is not None and Iv_min > 0 and v["Iy"] < Iv_min)}


def _lignes(el: Dict, familles: Optional[Sequence[str]]) -> List[Dict]:
    fy = int(el["acier"][1:])
    ef, fl = _efforts(el), _flambement(el)
    return [_ligne(nom, prof, fy, el, ef, fl) for nom, prof in _filtres(el, familles).items()]


def _optimal(lignes: List[Dict]) -> Optional[str]:
    """``(100.0 - util[util <= 100.0]).idxmin()`` : le plus chargé sous 100 % (utilisation ARRONDIE), le premier à
    égalité dans l'ordre du tableau."""
    best, ecart = None, None
    for li in lignes:
        if li["utilisation"] <= 100.0:
            e = 100.0 - li["utilisation"]
            if ecart is None or e < ecart:
                best, ecart = li["nom"], e
    return best


def _plus_leger(lignes: List[Dict]) -> Optional[str]:
    """AJOUT du moteur : le plus léger dont η ≤ 1 (exact), le premier à égalité de masse."""
    ok = [li for li in lignes if li["taux"] <= 1.0 and li["masse"] is not None]
    return min(ok, key=lambda li: li["masse"])["nom"] if ok else None


# ------------------------------------------------------------------ la fiche du profilé retenu
def _f(cle, symbole, termes, resultat, unite) -> Dict:
    return {"cle": cle, "symbole": symbole, "termes": [{"symbole": s, "valeur": v, "unite": u} for s, v, u in termes],
            "resultat": resultat, "unite": unite}


def _fiche(nom: str, el: Dict, ef: Dict, fl: Dict, ligne: Dict) -> Dict:
    """La colonne de droite de l'écran pour le profilé retenu, avec les grandeurs de ses formules."""
    prof = CA._profils()[nom]
    fy = int(el["acier"][1:])
    g0, g1 = el["gamma_M0"], el["gamma_M1"]
    resist = CA.resistances_section(prof, fy=fy, gamma_M0=g0)
    ratios = CA.taux_section(ef["N"], ef["My"], ef["Mz"], ef["Vy"], ef["Vz"], resist)

    def kn(x, k):
        return x / k if x is not None else None
    R = {"N_Rd": kn(resist["N_Rd_N"], 1e3), "My_Rd": kn(resist["My_Rd_Nmm"], 1e6), "Mz_Rd": kn(resist["Mz_Rd_Nmm"], 1e6),
         "Vy_Rd": kn(resist["Vy_Rd_N"], 1e3), "Vz_Rd": kn(resist["Vz_Rd_N"], 1e3)}
    F = []
    if R["N_Rd"] is not None:
        F.append(_f("N_Rd", "N_Rd", [("A", prof["A"], "cm²"), ("fy", fy, "N/mm²"), ("gamma_M0", g0, "")], R["N_Rd"], "kN"))
    if R["My_Rd"] is not None:
        F.append(_f("My_Rd", "M_y,Rd", [("Wel_y", prof["Wel_y"], "cm³"), ("fy", fy, "N/mm²"), ("gamma_M0", g0, "")], R["My_Rd"], "kN·m"))
    if R["Mz_Rd"] is not None:
        F.append(_f("Mz_Rd", "M_z,Rd", [("Wel_z", prof["Wel_z"], "cm³"), ("fy", fy, "N/mm²"), ("gamma_M0", g0, "")], R["Mz_Rd"], "kN·m"))
    if R["Vz_Rd"] is not None:
        F.append(_f("Vz_Rd", "V_z,Rd", [("Avz", prof["Avz"], "cm²"), ("fy", fy, "N/mm²"), ("gamma_M0", g0, "")], R["Vz_Rd"], "kN"))
    if R["Vy_Rd"] is not None:
        F.append(_f("Vy_Rd", "V_y,Rd", [("Avy", prof["Avy"], "cm²"), ("fy", fy, "N/mm²"), ("gamma_M0", g0, "")], R["Vy_Rd"], "kN"))
    for k, s, e, res in (("rN", "η_N", "N", "N_Rd"), ("rMy", "η_My", "My", "My_Rd"), ("rMz", "η_Mz", "Mz", "Mz_Rd"),
                         ("rVz", "η_Vz", "Vz", "Vz_Rd"), ("rVy", "η_Vy", "Vy", "Vy_Rd")):
        if R[res] is not None:
            F.append(_f(k, s, [(e + "_Ed", ef[e] if k == "rN" else abs(ef[e]), ""), (res, R[res], "")], ratios[k], ""))
        else:   # résistance absente : le module prend 0 (et le dit, ou non, selon l'effort)
            F.append(_f(k, s, [], ratios[k], ""))
    F.append(_f("util_NM", "η_NM", [("rN", ratios["rN"], ""), ("rMy", ratios["rMy"], ""), ("rMz", ratios["rMz"], "")], ratios["util_NM"], ""))
    F.append(_f("util", "η_section", [("util_NM", ratios["util_NM"], ""), ("rVy", ratios["rVy"], ""), ("rVz", ratios["rVz"], "")],
                ratios["util"], ""))
    # le flambement de la colonne de droite (case cochée et N > 0)
    axes, Nb_min = [], None
    if el["flambement"] and ef["N"] > 0:
        for axe, I, L in (("y", prof.get("Iy"), fl["Lcr_y"]), ("z", prof.get("Iz"), fl["Lcr_z"])):
            b = CA.flambement_chi(prof.get("A"), I, L, fy=fy, alpha=fl["alpha"]) if (L > 0 and I) else None
            if not b:
                continue
            Nb = b["chi"] * (prof["A"] * 1e2) * fy / g1
            axes.append({"axe": axe, "Lcr": L, "I": I, "Ncr": b["Ncr_N"] / 1e3, "lambdabar": b["lambdabar"], "phi": b["phi"],
                         "chi": b["chi"], "Nb_Rd": Nb / 1e3, "Nb_Rd_N": Nb})
            F.append(_f(f"Ncr_{axe}", f"N_cr,{axe}", [("E", CA.E_ACIER, "N/mm²"), (f"I_{axe}", I, "cm⁴"), (f"Lcr_{axe}", L, "m")],
                        b["Ncr_N"] / 1e3, "kN"))
            F.append(_f(f"lambda_{axe}", f"λ̄_{axe}", [("A", prof["A"], "cm²"), ("fy", fy, "N/mm²"), (f"Ncr_{axe}", b["Ncr_N"] / 1e3, "kN")],
                        b["lambdabar"], ""))
            F.append(_f(f"phi_{axe}", f"φ_{axe}", [("alpha", fl["alpha"], ""), (f"lambda_{axe}", b["lambdabar"], "")], b["phi"], ""))
            F.append(_f(f"chi_{axe}", f"χ_{axe}", [(f"phi_{axe}", b["phi"], ""), (f"lambda_{axe}", b["lambdabar"], "")], b["chi"], ""))
            F.append(_f(f"Nb_Rd_{axe}", f"N_b,Rd,{axe}", [(f"chi_{axe}", b["chi"], ""), ("A", prof["A"], "cm²"), ("fy", fy, "N/mm²"),
                                                           ("gamma_M1", g1, "")], Nb / 1e3, "kN"))
        if axes:
            Nb_min = min(a["Nb_Rd_N"] for a in axes)
            F.append(_f("util_buck", "η_b", [("N_Ed", ef["N"], "kN"), ("Nb_Rd", Nb_min / 1e3, "kN")], ligne["taux_flambement"], ""))
    F.append(_f("util_global", "η", [("util", ratios["util"], ""), ("util_buck", ligne["taux_flambement"], "")], ligne["taux"], ""))
    dims = [("h [mm]", prof.get("h")), ("b [mm]", prof.get("b")), ("tw [mm]", prof.get("tw")), ("tf [mm]", prof.get("tf")), ("r [mm]", prof.get("r"))]
    props = [("Poids [kg/m]", prof.get("Poids")), ("A [cm²]", prof.get("A")), ("Wel_y [cm³]", prof.get("Wel_y")), ("Wel_z [cm³]", prof.get("Wel_z")),
             ("Avy [cm²]", prof.get("Avy")), ("Avz [cm²]", prof.get("Avz")), ("Iy [cm⁴]", prof.get("Iy")), ("Iz [cm⁴]", prof.get("Iz"))]
    return {"nom": nom, "famille": prof["type"], "masse": prof.get("Poids"), "dimensions": [list(x) for x in dims],
            "proprietes": [list(x) for x in props], "resistances": R, "ratios": ratios, "axes": axes,
            "Nb_Rd": Nb_min / 1e3 if Nb_min is not None else None, "taux_flambement": ligne["taux_flambement"],
            "taux": ligne["taux"], "formules": F}


def _verifications(fiche: Dict, el: Dict, ef: Dict) -> List[Dict]:
    R, r = fiche["resistances"], fiche["ratios"]
    actifs = {"N": el["avec_N"], "My": True, "Mz": el["avec_Mz"], "Vz": True, "Vy": el["avec_Vy"]}
    ecran = {"N": True, "My": True, "Mz": el["avec_Mz"], "Vz": True, "Vy": el["avec_Vy"]}
    unites = {"N": "kN", "My": "kN·m", "Mz": "kN·m", "Vz": "kN", "Vy": "kN"}
    out = []
    for cle, rk in (("N", "rN"), ("My", "rMy"), ("Mz", "rMz"), ("Vz", "rVz"), ("Vy", "rVy")):
        res = R[f"{cle}_Rd"]
        out.append({"cle": cle, "nom": NOMS[cle], "symbole": SYMBOLES[cle], "valeur": ef[cle] if cle == "N" else abs(ef[cle]),
                    "limite": res, "unite": unites[cle], "taux": r[rk], "actif": actifs[cle], "ecran": ecran[cle],
                    "resistance_absente": res is None, "formules": [f"{cle}_Rd", rk] if res is not None else [rk]})
    out.append({"cle": "NM", "nom": NOMS["NM"], "symbole": SYMBOLES["NM"], "valeur": r["util_NM"], "limite": 1.0, "unite": "",
                "taux": r["util_NM"], "actif": True, "ecran": True, "resistance_absente": False, "formules": ["util_NM"]})
    out.append({"cle": "section", "nom": NOMS["section"], "symbole": SYMBOLES["section"], "valeur": r["util"], "limite": 1.0, "unite": "",
                "taux": r["util"], "actif": True, "ecran": True, "resistance_absente": False, "formules": ["util"]})
    if fiche["taux_flambement"] is not None:
        out.append({"cle": "flambement", "nom": NOMS["flambement"], "symbole": SYMBOLES["flambement"], "valeur": ef["N"],
                    "limite": fiche["Nb_Rd"], "unite": "kN", "taux": fiche["taux_flambement"], "actif": True, "ecran": False,
                    "resistance_absente": False,
                    "formules": [f["cle"] for f in fiche["formules"] if f["cle"].split("_")[0] in ("Ncr", "lambda", "phi", "chi", "Nb")]
                    + ["util_buck"]})
    return out


def _gouvernant(fiche: Dict) -> Dict:
    """Le taux qui fait η, dans l'ordre du module : max(η_NM, η_Vy, η_Vz) rend le PREMIER maximum ; le flambement ne
    gouverne que s'il dépasse strictement (max(util, util_buck) rend util à égalité)."""
    r = fiche["ratios"]
    cle, t = "NM", r["util_NM"]
    for k, x in (("Vy", r["rVy"]), ("Vz", r["rVz"])):
        if x > t:
            cle, t = k, x
    if fiche["taux_flambement"] is not None and fiche["taux_flambement"] > r["util"]:
        cle, t = "flambement", fiche["taux_flambement"]
    return {"cle": cle, "nom": NOMS[cle], "taux": t}


# ------------------------------------------------------------------ l'élément
def _classe_4(p: Dict, fy: int, ef: Dict) -> List[tuple]:
    """Les parois élancées du profilé ``p`` (EN 1993-1-1 tableau 5.2) que l'écran ignore (défaut D5) : [(code, message)].
    c_f = (b − t_w − 2r)/2 (semelle en console), c_w = h − 2t_f − 2r (âme) ; ε = √(235 / f_y)."""
    h, b, tw, tf, r = (p.get(k) or 0.0 for k in ("h", "b", "tw", "tf", "r"))
    if not (tw > 0 and tf > 0):
        return []
    eps = math.sqrt(235.0 / fy)

    def v(x):
        return f"{x:.1f}".replace(".", ",")
    out = []
    c_f, c_w = (b - tw - 2 * r) / 2, h - 2 * tf - 2 * r
    flechi = any(ef[k] != 0 for k in ("My", "Mz")) or ef["N"] > 0
    if flechi and c_f / tf > LIMITE_SEMELLE_CL3 * eps:
        out.append(("classe_4", f"Semelle élancée : c/t_f = {v(c_f / tf)} > 14ε = {v(LIMITE_SEMELLE_CL3 * eps)} (EN 1993-1-1 tableau 5.2, "
                                f"console comprimée) : classe 4 ; l'écran ne réduit pas la section (W_eff, A_eff) — M_Rd et N_Rd "
                                f"surestimés (non conservatif)."))
    if ef["N"] > 0 and c_w / tw > LIMITE_AME_COMPRESSION_CL3 * eps:
        out.append(("classe_4", f"Âme élancée en compression : c/t_w = {v(c_w / tw)} > 42ε = {v(LIMITE_AME_COMPRESSION_CL3 * eps)} "
                                f"(EN 1993-1-1 tableau 5.2, compression pure) : classe 4 ; l'écran ne réduit pas la section (A_eff) — "
                                f"N_Rd et N_b,Rd surestimés (non conservatif)."))
    return out


def _avertissement(code: str, message: str, module: Optional[str] = None) -> Dict:
    return {"code": code, "message": message, "module": module}


def verifier(element: Dict) -> Dict:
    """Le calcul d'un élément : le tableau trié, l'optimal, le retenu et sa fiche, ses taux, ses avertissements."""
    el = EN.normaliser(element)
    fy = int(el["acier"][1:])
    ef, fl = _efforts(el), _flambement(el)
    familles = el["familles"] or EN.familles_disponibles()
    lignes = sorted(_lignes(el, el["familles"]), key=lambda x: x["utilisation"])
    noms = [li["nom"] for li in lignes]
    optimal = _optimal(lignes)
    out = {"reference": REFERENCE, "element": el, "acier": {"nuance": el["acier"], "fy": fy}, "efforts": ef,
           "flambement": {"actif": el["flambement"], "calcule": el["flambement"] and ef["N"] > 0, **fl},
           "deversement": {"actif": el["deversement"], "L_LT": el["L_LT"] if el["deversement"] else 0.0},
           "familles_disponibles": EN.familles_disponibles(), "familles": familles, "Iy_min": el["Iy_min"],
           "lignes": lignes, "n_profils": len(lignes), "n_tiennent": sum(li["tient"] for li in lignes),
           "optimal": optimal, "plus_leger": _plus_leger(lignes), "retenu": None, "retenu_auto": el["profil"] is None,
           "choix_hors_tableau": el["profil"] if el["profil"] is not None and el["profil"] not in noms else None,
           "profil": None, "verifications": [], "taux_max": None, "statut": "vide", "gouvernant": None, "avertissements": []}
    av = out["avertissements"]
    if el["deversement"]:
        av.append(_avertissement("deversement_non_calcule",
                                 "Déversement non vérifié : le module ne calcule ni M_cr ni χ_LT ; la longueur L_LT saisie n'est pas utilisée.",
                                 TEXTES_MODULE["deversement_non_calcule"]))
    if el["avec_T"]:
        av.append(_avertissement("torsion_ignoree", "T_Ed n'entre dans aucun calcul du module : la torsion n'est pas vérifiée."))
    if not lignes:
        av.append(_avertissement("aucun_profil", "Aucun profilé ne satisfait aux critères (familles cochées, I_y min).",
                                 TEXTES_MODULE["aucun_profil"]))
        return out
    nom = el["profil"] if el["profil"] in noms else (optimal if optimal in noms else noms[0])
    ligne = lignes[noms.index(nom)]
    fiche = _fiche(nom, el, ef, fl, ligne)
    out.update(retenu=nom, profil=fiche, verifications=_verifications(fiche, el, ef), taux_max=ligne["taux"],
               statut="ok" if ligne["taux"] <= 1.0 else "nok", gouvernant=_gouvernant(fiche))
    if out["choix_hors_tableau"]:
        av.append(_avertissement("choix_hors_tableau", f"{el['profil']} n'est plus dans la sélection (familles, I_y min) : "
                                                       f"le profilé retenu est celui de l'écran, {nom}."))
    R = fiche["resistances"]
    # ce que la colonne de droite de l'écran dit d'une résistance absente (Mz et Vy seulement si l'effort est ajouté)
    for k, quand in (("N_Rd", True), ("My_Rd", True), ("Mz_Rd", el["avec_Mz"]), ("Vz_Rd", True), ("Vy_Rd", el["avec_Vy"])):
        if R[k] is None and quand:
            msg = {"Vy_Rd": "V_y,Ed n'est PAS vérifié : le module lit A_v,y dans la forme ancienne du catalogue, qui ne le porte pas "
                            "(η_Vy = 0 quel que soit V_y,Ed — non conservatif)."}.get(k, f"{k} : donnée absente, le taux correspondant vaut 0.")
            av.append(_avertissement(f"{k}_absent", msg, TEXTES_MODULE[f"{k}_absent"]))
    p = CA.profil(nom)
    t_max = max(p.get("tf") or 0.0, p.get("tw") or 0.0)
    if t_max > EPAISSEUR_LIMITE:
        t_txt = f"{t_max:g}".replace(".", ",")
        if t_max <= EPAISSEUR_MAX_TABLEAU:
            fy_n = FY_EPAIS[el["acier"]]
            ecart = f"{round((fy / fy_n - 1) * 100, 1):g}".replace(".", ",")
            msg = (f"Paroi de {t_txt} mm > 40 mm : l'écran prend f_y = {fy} N/mm², EN 1993-1-1 (tableau 3.1) donne {fy_n} N/mm² "
                   f"pour 40 < t ≤ 80 mm — résistances surestimées de {ecart} % (non conservatif).")
        else:
            msg = (f"Paroi de {t_txt} mm > 80 mm : hors du tableau 3.1 d'EN 1993-1-1 ; l'écran prend f_y = {fy} N/mm², la valeur des "
                   f"épaisseurs ≤ 40 mm — résistances surestimées (non conservatif) ; f_y est à lire dans la norme du produit (EN 10025).")
        av.append(_avertissement("fy_epaisseur", msg))
    for code_, message_ in _classe_4(p, fy, ef):
        av.append(_avertissement(code_, message_))
    if el["avec_N"] and ef["N"] < 0:
        av.append(_avertissement("traction_retranchee", "N_Ed négatif (traction) : η_N est négatif et se RETRANCHE de la flexion "
                                                        "dans η_NM = η_N + η_My + η_Mz (module, non conservatif)."))
    if el["flambement"] and ef["N"] <= 0:
        av.append(_avertissement("flambement_sans_compression", "Flambement demandé, mais N_Ed ≤ 0 : le module ne le calcule pas."))
    if el["flambement"] and ef["N"] > 0 and not fiche["axes"]:
        av.append(_avertissement("flambement_impossible", "Flambement non calculé : L_cr,y et L_cr,z sont nuls (ou A, I manquent).",
                                 TEXTES_MODULE["flambement_impossible"]))
    if optimal is None:
        # l'écran retient le moins chargé quand on ne choisit rien ; un profilé choisi dans le tableau reste le retenu
        quoi = (f"l'écran retient le moins chargé, {noms[0]}" if nom == noms[0] else f"le moins chargé est {noms[0]} ; {nom} est celui choisi")
        av.append(_avertissement("aucun_ne_tient", f"Aucun profilé de la sélection ne tient : {quoi}."))
    if all(ef[k] == 0 for k in ("N", "My", "Mz", "Vy", "Vz")):
        av.append(_avertissement("sans_effort", "Aucun effort saisi : tous les profilés « tiennent » et l'écran retient le premier du tableau."))
    return out


def candidats(element: Dict, familles: Optional[Sequence[str]] = None) -> List[Dict]:
    """Les lignes du sélecteur : les profilés de ``familles`` (celles de l'élément si None ; toutes si aucune), filtre
    I_y min compris, TRIÉS PAR MASSE (l'ordre du catalogue à égalité), chacune avec ``retenu`` (le profilé retenu de
    l'élément) et ``plus_leger`` (le plus léger qui tient PARMI ces lignes)."""
    el = EN.normaliser(element)
    if familles is not None:
        hors = [f for f in familles if f not in EN.familles_disponibles()]
        if hors:
            raise ValueError(f"famille(s) inconnue(s) : {', '.join(hors)}")
    fams = list(familles) if familles else el["familles"]
    lignes = _lignes(el, fams)
    rang = {n: i for i, n in enumerate(CA._profils())}
    lignes.sort(key=lambda li: (li["masse"] if li["masse"] is not None else math.inf, rang[li["nom"]]))
    retenu = verifier(el)["retenu"]
    leger = _plus_leger(lignes)
    for li in lignes:
        li["retenu"] = li["nom"] == retenu
        li["plus_leger"] = li["nom"] == leger
    return lignes

# -*- coding: utf-8 -*-
"""La famille PIED DE POTEAU du module « Assemblages métalliques » du site : platine de pied de poteau sur béton
(N + Mx + My), en deux MODES — le calcul « platine à la main » (boulons, éléments soudés et raidisseurs posés
librement : l'écran neuf de Streamlit) et la vérification complète EN 1993-1-8 + EN 1992-4 (grille régulière de
tiges, un profilé en I ou RHS). Fichier d'AIDE (sans MODULE) : il décrit la configuration à la carte commune
(api_web/assemblages.py).

Le moteur est celui du dépôt (``calcul.compute`` choisit le mode : ``moteur`` ou ``manuel.moteur``), importé SANS
ses fichiers d'écran (``interface.py``, ``ecran_saisie.py``, ``ecran_resultats.py``, ``manuel/ecran.py``, ``note.py``
chargent streamlit ou reportlab). Ce que l'écran Streamlit compose dans ces fichiers et que le site doit montrer est
une COPIE NOMMÉE, liée à l'original par tests/test_api_web_assemblages_pied.py (faux « st ») :

- les options courtes (``AFFICHAGE``, ``AFF_TYPES``), les aides « ? » (``AIDES_CO``, ``AIDES_MA``), les listes du
  périmètre (``NON_REVERIFIES``, ``NON_VERIFIES``), le sens des natures, la règle du Tableau 3.3 (``LEGENDE_33``) ;
- l'état de la bascule (``etat_bascule``), le boulon ajouté (``ajouter_boulon``), la grille posée (``poser_grille``) ;
- « l'épaisseur de la platine, le calcul à la main » (``platine_complet`` = ``ecran_resultats.bloc_platine``,
  ``platine_manuel`` = ``manuel/ecran.bloc_etat``, ``detail_main`` = ``manuel/ecran.detail_main``) ;
- les paramètres retenus des deux modes, la répartition (``repartition`` = ``onglet_repartition``), les efforts des
  tiges (``efforts_manuel`` = ``onglet_efforts``), les deux onglets Benchmark (``validation_*``).

La bascule de mode reprend la disposition comme l'écran Streamlit (``manuel/conversion`` : la grille et le profilé
deviennent des boulons et un élément ; vers la vérification complète seulement si la disposition est régulière, sinon
la raison est dite et le mode reste).
"""
import copy
import functools
import math

from acier.assemblages.pied_poteau.platine_beton import (benchmark as BCO, calcul as CA, entrees as E, formules as FO,
                                                        geometrie as GE, methode as MEC, notation as NO, schemas as SCO,
                                                        synthese as YCO, texte as TCO)
from acier.assemblages.pied_poteau.platine_beton.manuel import (benchmark as BMA, conversion as CV, elements as EL,
                                                               methode as MEM, schemas as SMA, synthese as YMA,
                                                               texte as TMA)
from acier.bibliotheques import DB, PERSO, find
from acier.formats import F, pct
from acier.js import N, js_str

from . import assemblages as A
from . import commun as C

INF = float("inf")

# ================================================================== copies nommées de l'écran Streamlit
# Les options courtes à l'affichage (ecran_saisie.AFFICHAGE) ; les valeurs brutes ne changent pas.
AFFICHAGE = {"mode": {E.MODE_MANUEL: "Calcul manuel", E.MODE_COMPLET: "Vérif. complète"},
             "perim": {E.PERIM_EC3: "EC3 seul", E.PERIM_TOUT: "EC3 + EC2-4"},
             "crit": {E.CRIT_ELAS: "Élastique", E.CRIT_EC3: "Plastique"},
             "appui": {E.APPUI_EC3: "Bande c", E.APPUI_PLEIN: "Toute la platine"},
             "forme": {E.FORME_I: "I / H", E.FORME_RHS: "Tube RHS / SHS"},
             "typ_a": {E.TYPE_TETE: "Tête / plaque", E.TYPE_CROSSE: "Crosse", E.TYPE_ETA: "Scellée (ETA)"}}
# Sur le site, le segment Mode dit « Manuel / Complet » : « Calcul manuel / Vérif. complète » ne tient pas dans la
# colonne Données au format tablette (mesuré à 1024 px : la colonne défilait de côté, comme le cas d'appui poutre–
# poteau) ; le libellé « Mode » et son aide disent le reste. Les autres options sont celles de l'écran.
AFFICHAGE_SITE = dict(AFFICHAGE, mode={E.MODE_MANUEL: "Manuel", E.MODE_COMPLET: "Complet"})
# Les types d'élément courts (manuel/ecran.AFF_TYPES).
AFF_TYPES = {E.TYPE_I: "Profilé I/H", E.TYPE_RHS: "Tube RHS", E.TYPE_CHS: "Tube CHS", E.TYPE_PLAT: "Plat",
             E.TYPE_ROND: "Rond", E.TYPE_RECT: "Rectangle"}
# Les « ? » de la carte (ecran_saisie.AIDES : vérification complète ; AIDES_MANUEL : calcul manuel) — ceux que le
# site montre ; les autres (dessin interactif, panneau Cotes) n'ont pas d'objet ici.
AIDES_CO = {
    "mode": "Calcul manuel : boulons, éléments et raidisseurs posés librement, épaisseur requise comme à la "
            "main. Complète : grille régulière, Tab. 6.7, EN 1992-4.",
    "perim": "EC3 seul : platine, tiges, poteau, cordons et appui fjd décident ; béton d'ancrage (EN 1992-4) en "
             "information. EC3 + EC2-4 : il entre dans le verdict.",
    "typ_a": "Tête / plaque : hef et dh se règlent sur la coupe A-A. Crosse : adhérence. Scellée : valeurs de "
             "l'ETA (τRk, k1, NRk,s, VRk,s).",
    "a_cc": "Coefficient de l'ANB belge pour fcd = αcc·fck/γc ; 0,85 par défaut, à confirmer.",
    "C_f": "Frottement platine / mortier, déduit du cisaillement des tiges (V EC3).",
    "L_c": "Hauteur du poteau pour la classification de rigidité du pied (EN 1993-1-8 §5.2.2.5), en information.",
    "visserie": "Rondelles, écrou et contre-écrou par tige, portés au cartouche du plan de principe.",
    "EFFORTS ELU": "N > 0 compression ; Mx > 0 comprime le côté y > 0, My > 0 comprime x > 0 ; T = torsion. "
                   "Appliqués en (xN ; yN), (0 ; 0) = centre de la platine.",
    "ETA": "Valeurs de l'ETA de la tige scellée (état du béton, classe) ; k1 = kcr ou kucr (7,7 / 11,0 à défaut) ; "
           "NRk,s et VRk,s : 0 = calculé par l'outil.",
    "COEFF. γ": "γM0 1,00 et γM2 1,25 (ANB belge) ; γc = γMc 1,50 ; αcc 0,85 (fcd = αcc·fck/γc) — hypothèse à "
                "confirmer sur le texte NBN.",
}
AIDES_MA = dict(AIDES_CO, **{
    "crit": "Élastique : Von Mises ≤ fy/γM0 sans effet de levier — ≈ 1,22 × l'épaisseur plastique (√1,5), "
            "jusqu'à 1,73 (√3) avec levier. Plastique : tronçon en T Mpl.",
    "appui": "Bande c autour des parois (EN 1993-1-8 §6.2.5, sûr) ou toute la platine (rigide : efforts des tiges "
             "comparables aux logiciels d'ancrage).",
    "xN": "Point d'application de N, Mx, My, Vx, Vy ; (0 ; 0) = centre de la platine. Un élément excentré crée le "
          "moment N·e tout seul.",
    "visserie": "Rondelles et écrous par boulon, portés au cartouche du plan. Ex. : « 1 rondelle + 2 écrous "
                "(contre-écrou) », « 2 rondelles + 1 écrou ».",
    "tiges": "Origine au centre de la platine, x vers la droite, y vers le haut (vue en plan). Chaque boulon se "
             "déplace aussi sur le dessin.",
    "ÉLÉMENTS": "Élément soudé = profilé I/H, tube, plat, rond ou rectangle posé en (x ; y). Rotation et dimensions : "
                "dans sa fenêtre sur le dessin (ici si le dessin est fixe).",
    "RAIDISSEURS": "Plat vertical soudé sur la platine et l'élément, parti de (x ; y) dans le sens donné ; h sert "
                   "aux coupes.",
})
# La note de la fenêtre Soudure du calcul manuel (manuel/ecran._panneaux).
NOTE_GORGE_MA = "La gorge ne sert qu'au bras de levier (0,8·a·√2 déduit) : les cordons ne sont pas vérifiés ici."
# Le périmètre (ecran_resultats.NON_REVERIFIES ; manuel/ecran.NON_VERIFIES) et le sens des natures.
NON_REVERIFIES = ("fendage et éclatement latéral du massif (EC2-4 §7.2.1.7-8, signalés en alerte)",
                  "tiges post-installées, raidisseurs, bêche", "flambement et classe de section du poteau",
                  "poinçonnement du massif au-delà de l'EN 1992-1-1 §6.7")
NON_VERIFIES = ("cordons", "raidisseurs eux-mêmes", "poteau",
                "béton d'ancrage (cône, arrachement, bord, pry-out) — logiciel d'ancrage ou vérification complète")
NATURES = ("EC", "EC + INT", "INT")
SENS_EC = "formule de l'Eurocode"                                            # ui_commun.NATURES["EC"]
TEXTES_NATURES_CO = {"EC + INT": "Eurocode et interprétation nommée",
                     "INT": "modèle d'ingénieur (niveau 2, tiges de coin ou latérales)"}
TEXTES_NATURES_MA = {"EC + INT": "Eurocode et interprétation nommée",
                     "INT": "modèle d'ingénieur (sections au nu, bandes, angles)"}
# La légende du Tableau 3.3 des deux modes (ecran_resultats._tableau_33).
LEGENDE_33 = ("e ≥ 1,2·d0 au bord de la platine (Tableau 3.3) ; les distances aux bords du MASSIF relèvent de "
              "l'EN 1992-4 (cône, bord, fendage).")
# Les légendes des vues en plan (ecran_resultats.CAPTION_LEGENDE ; manuel/ecran.CAPTION_LEGENDE).
LEGENDE_PLAN_CO = "Vue en plan : aire efficace (bleu), zone comprimée (rouge), axe neutre, C et T des tiges."
LEGENDE_PLAN_MA = ("Vue en plan : aire d'appui (bleu), zone comprimée (rouge), C, axe neutre, T et bras m de chaque "
                   "boulon, bande comprimée (mEd).")
# Les quatre critères de l'épaisseur requise : calcul manuel (manuel/ecran.CRITERES_NOMS) et vérification complète
# (ecran_resultats.bloc_platine, « crit »).
CRITERES_NOMS = {"T": "tronçon en T du boulon", "S": "section au nu d'une paroi", "c": "bande comprimée",
                 "b": "pression diamétrale"}
CRITERES_CO = {"c": "console comprimée", "N": "appui N ≤ fjd·Aeff", "M": "flexion composée (niveau 1)",
               "T": "tronçon en T"}
ORDRE_MA, ORDRE_CO = ("T", "S", "c", "b"), ("c", "N", "M", "T")
# Le vocabulaire par mode (ecran_saisie.LIBELLE_MODE) : la même clé « ancr » est une tige ou un boulon.
LIBELLE_MODE = {E.MODE_COMPLET: {"ancr": "Tige", "visserie": "Visserie par tige"},
                E.MODE_MANUEL: {"ancr": "Boulon", "visserie": "Visserie par boulon"}}
DIMS_ELEM = {E.TYPE_I: ("h", "b", "tw", "tf", "r"), E.TYPE_RHS: ("h", "b", "t"), E.TYPE_CHS: ("d", "t"),
             E.TYPE_PLAT: ("b", "t"), E.TYPE_ROND: ("d",), E.TYPE_RECT: ("b", "h")}       # manuel/ecran.DIMS_PAR_TYPE
LIB_DIM = {"h": "h", "b": "b", "tw": "t<sub>w</sub>", "tf": "t<sub>f</sub>", "r": "r", "t": "t", "d": "Ø"}


def etat_bascule(u) -> str:
    """La phrase sous l'en-tête du calcul manuel (manuel/ecran.etat_bascule) : la vérification complète est-elle
    disponible pour cette disposition ?"""
    reg = CV.disposition_reguliere(u)
    return ("Vérification complète : disponible pour cette disposition." if reg["ok"]
            else "Vérification complète : indisponible — " + reg["raison"] + ".")


def ajouter_boulon(u) -> dict:
    """Le boulon que « ➕ Boulon » ajoute (manuel/ecran._ajouter_boulon) : au bord bas, à 60 mm, à droite du dernier
    boulon posé sur cette ligne."""
    lp = N(u.get("lp_u")) or 500.0
    tiges = u.get("tiges") or []
    y = -(lp / 2 - 60.0)
    xs = [float(N(t.get("x")) or 0) for t in tiges if abs(float(N(t.get("y")) or 0) - y) < 1e-6]
    x = (max(xs) + 80.0) if xs else 0.0
    return dict(x=x, y=y)


GRILLE_DEFAUT = {"nx": 2, "ny": 2, "wx": 400.0, "wy": 400.0}     # manuel/ecran.bloc_boulons (setdefault de ppb_ui_g*)


def poser_grille(g) -> list:
    """« Poser la grille (remplace les boulons) » (manuel/ecran._poser_grille)."""
    nx_, ny_ = int(g.get("nx") or 2), int(g.get("ny") or 2)
    wx, wy = float(g.get("wx") or 400), float(g.get("wy") or 400)
    return [dict(x=float(x), y=float(y)) for (x, y, _i, _j) in GE.positions_tiges(nx_, ny_, wx, wy)]


# ---------------------------------------------------------- « l'épaisseur de la platine, le calcul à la main »
def _lig(lab, val, fort=False):
    return (lab, val, fort)


def platine_complet(R) -> dict:
    """ecran_resultats.bloc_platine : l'épaisseur requise (le plus grand des quatre critères), puis de chaque côté le
    moment pris, sa résistance et l'épaisseur qu'il demande, et la comparaison de Navier."""
    P, u, c = R.plaque, R.u, R.ck["tpq"]
    if not c.active:
        return {"actif": False, "note": "Géométrie non valide : pas d'épaisseur requise."}
    pire = max(("c", "N", "M", "T"), key=lambda k: P["t_" + k])
    req = "∞ (aucune épaisseur ne suffit : agrandir la platine ou les tiges)" if c.Ed == INF else F(c.Ed, 1) + " mm"
    out = {"actif": True, "ok": bool(c.ok),
           "titre": "Épaisseur requise " + req + (" ≤ " if c.ok else " > ") + "tp = " + F(u.tp_u, 0) + " mm",
           "complement": "— " + pct(c.eta, 1) + " · critère : " + CRITERES_CO[pire] + " · fy = " + F(R.fy_p, 0)
                         + " MPa (épaisseur retenue)",
           "criteres": [(CRITERES_CO[k], P["t_" + k] / u.tp_u if u.tp_u else 0.0, P["t_" + k] <= u.tp_u + 1e-9)
                        for k in ("c", "N", "M", "T")]}
    out["comprime"] = ("Côté comprimé — console au nu de l'aile", [
        _lig("Pression sous la platine (répartition élastique)",
             "p max = " + F(P["p_max"], 1) + " MPa ≤ fjd = " + F(R.fjd, 1) + " MPa"),
        _lig("Console mobilisée", "bande c = tp·√(fy/(3·fjd·γM0)) = " + F(P["c"], 0) + " mm (EC3 §6.2.5) — débord "
             "géométrique " + F(P["L_geo"], 0) + " mm"),
        _lig("Moment pris (" + P["ligne"] + ")", "mEd = ∫p·s ds sur c = " + F(P["m_Ed"], 1) + " kNm/m", True),
        _lig("Résistance élastique", "mRd = tp²·fy/(6·γM0) = " + F(P["m_Rd_el"], 1) + " kNm/m → "
             + pct(R.ck["pc"].eta, 0) if R.ck["pc"].active else "—"),
        _lig("Épaisseur pour ce moment", "tc = √(6·mEd·γM0/fy) = " + F(P["t_c"], 1) + " mm"),
        _lig("Appui : N ≤ fjd·Aeff(c)", "tN = " + F(P["t_N"], 1) + " mm" if P["t_N"] != INF else "tN = ∞"),
        _lig("Flexion composée : niveau 1 ≤ 1", "tM = " + F(P["t_M"], 1) + " mm" if P["t_M"] != INF else "tM = ∞")])
    T = P["tendu"]
    if T and T["T"] > 1e-9:
        lignes = [
            _lig("Rangée tendue", NO.rangee_lisible(T["cle"]) + " · " + T["classe"] + " : T = " + F(T["T"], 1) + " kN"),
            _lig("Bras au nu de l'aile (cordon déduit)", "m = e − 0,8·a·√2 = " + F(T["m"], 1) + " mm"),
            _lig("Moment pris", "M = T·m = " + F(T["T"], 1) + " × " + F(T["m"], 1) + " = " + F(T["M"], 2) + " kNm", True),
            _lig("Longueur efficace", "Σleff,1 = " + F(T["leff1"], 0) + " mm"
                 + (" (plafond de coin 0,25·b par tige)" if T["classe"] == "coin" else "")),
            _lig("Résistance plastique", "Mpl,1 = 0,25·Σleff,1·tp²·fy/γM0 = " + F(T["Mpl1"], 2) + " kNm"),
            _lig("Effet de levier", ("Lb = " + F(T["Lb"], 0) + (" ≤ " if T["levier"] else " > ") + "Lb* = "
                 + F(min(T["Lb_star"], 9999), 0) + " mm : " + ("avec" if T["levier"] else "sans")
                 + " → FT,Rd = " + ("4" if T["levier"] else "2") + "·Mpl,1/m… = " + F(T["FT_Rd"], 0)
                 + " kN (mode " + str(T["mode"]) + ")")),
            _lig("Épaisseur pour ce moment", "tp ≥ √(" + ("1" if T["levier"] else "2") + "·T·m·γM0/(Σleff,1·fy)) = "
                 + F(T["t_formule"], 1) + " mm ; tous modes : tT = " + F(P["t_T"], 1) + " mm")]
    else:
        lignes = [_lig("Rangée tendue", "aucune tige tendue : la platine ne fléchit pas en traction"),
                  _lig("Épaisseur pour ce côté", "tT = " + F(P["t_T"], 1) + " mm (minimum du balayage)")]
    out["tendu"] = ("Côté tendu — tronçon en T de la rangée la plus sollicitée", lignes)
    NV = P["navier"]
    base = ("Comparaison manuelle — Navier sur TOUTE la platine (repère, pas un critère) : σ = N/A ± 6·Mx/(bp·lp²) ± "
            "6·My/(lp·bp²) = " + F(NV["sN"], 2) + " ± " + F(NV["sMx"], 2) + " ± " + F(NV["sMy"], 2) + " MPa")
    if NV["applicable"]:
        out["navier"] = (base + " → σ max = " + F(NV["s_max"], 2) + " MPa, débord L = " + F(NV["L"], 0) + " mm, m = σ max·L²/2 = "
                         + F(NV["m"], 1) + " kNm/m → t = √(6·m·γM0/fy) = " + F(NV["t"], 1) + " mm (pression uniforme sur "
                         "toute la platine : plus de bras, moins d'appui que l'aire efficace).")
    else:
        out["navier"] = (base + " → σ min = " + F(NV["s_min"], 2) + " MPa " + ("< 0 : soulèvement, Navier ne s'applique pas"
                         if u.N_Ed > 0 else ": pas de compression") + " — la répartition sans traction ci-dessus fait foi.")
    return out


def platine_manuel(R) -> dict:
    """manuel/ecran.bloc_etat : l'épaisseur requise et ses quatre critères (tronçon en T du boulon, section au nu
    d'une paroi, bande comprimée, pression diamétrale)."""
    P, u, c = R.plaque, R.u, R.ck["tpq"]
    if not c.active:
        return {"actif": False, "note": "Géométrie non valide ou équilibre introuvable : pas d'épaisseur requise."}
    req = "∞ (aucune épaisseur ne suffit)" if c.Ed == INF else F(c.Ed, 1) + " mm"
    return {"actif": True, "ok": bool(c.ok),
            "titre": "Épaisseur requise " + req + (" ≤ " if c.ok else " > ") + "tp = " + F(u.tp_u, 0) + " mm",
            "complement": "— " + pct(c.eta, 1) + " · critère : " + CRITERES_NOMS[P["pire"]] + " · " + P["critere"]
                          + " · fy = " + F(R.fy_p, 0) + " MPa",
            "criteres": [(CRITERES_NOMS[k], P["t_" + k] / u.tp_u if u.tp_u else 0.0, P["t_" + k] <= u.tp_u + 1e-9)
                         for k in ("T", "S", "c", "b")]}


def detail_main(R) -> dict:
    """manuel/ecran.detail_main : le détail comme à la main — boulon par boulon, face par face, la bande comprimée,
    les trous. Chaque table : (titre, en-têtes, lignes, légende) ; une cellule (η, vérifié) est un taux."""
    P = R.plaque

    def mm(v):
        return "∞" if v == INF else F(v, 1)
    tendus = [t for t in R.tiges if t["T"] > 1e-9]
    el = R.elastique
    out = {}
    ent = ["n°", "T (kN)", "Appui", "m (mm)", "T·m (kNm)", "leff (mm)", "σvm (MPa)" if el else "FT,Rd (kN)", "η",
           "tp plast. / élast."]
    if tendus:
        lignes = []
        for t in tendus:
            lignes.append([js_str(t["i"] + 1), F(t["T"], 1), t["appui"]["face"].replace("élément ", "É"),
                           F(t["m"], 1) + (" (angle)" if t["coin"] else ""), F(t["M"], 2), F(t["leff1"], 0),
                           F(t["svm"], 0) if el else F(t["FT_Rd"], 0) + " (m" + str(t["mode_ts"]).split(" ")[0] + ")",
                           (t["eta"], t["eta"] <= 1 + 1e-9), F(t["t_pl"], 1) + " / " + F(t["t_el"], 1)])
        pt = R.pire_tige
        levier = (f"Boulon {pt['i'] + 1} : Lb = {F(R.Lb, 0)} / Lb* = {F(min(pt['Lb_star'], 9999), 0)} mm → "
                  + ("avec" if pt["levier"] else "sans") + " effet de levier")
        if el:
            cap = (f"Critère élastique retenu : η = σvm / (fy/γM0), fy/γM0 = {F(R.ck['ptl'].Rd, 0)} MPa — σ de "
                   f"Von Mises de la bande (2·leff). {levier} ; Mpl,1 = 0,25·leff·tp²·fy/γM0 = {F(pt['Mpl1'], 2)} kNm, "
                   f"FT,Rd = {F(pt['FT_Rd'], 0)} kN, mode {pt['mode_ts']} (plastique, information).")
        else:
            cap = (levier + ", mode " + str(pt["mode_ts"])
                   + f" ; Mpl,1 = 0,25·leff·tp²·fy/γM0 = {F(pt['Mpl1'], 2)} kNm ; σ Von Mises de la bande (2·leff) = {F(pt['svm'], 0)} MPa.")
    else:
        lignes, cap = [], "Aucun boulon tendu : la platine ne fléchit pas en traction."
    out["tendus"] = ("Côté tendu — boulon par boulon (tronçon en T)", ent, lignes, cap)
    ent = ["Face", "Boulons", "m⊥ (mm)", "ΣT·m (kNm)", "b (mm)", "σvm (MPa)" if el else "Mpl (kNm)", "η",
           "tp plast. / élast."]
    if R.sections:
        lignes = []
        for s in R.sections:
            lignes.append([s["nom"].replace("élément ", "É").replace("raidisseur ", "R"),
                           ", ".join(str(i + 1) for i in s["idx"]), ", ".join(F(m, 0) for m in s["m"]),
                           F(s["M"], 2), F(s["b"], 0), F(s["svm"], 0) if el else F(s["Mpl"], 2),
                           (s["eta"], s["eta"] <= 1 + 1e-9), F(s["t_pl"], 1) + " / " + F(s["t_el"], 1)])
        ps = R.pire_section
        cap = ("b = min(platine ; L + 2·m) : la face de longueur L et la diffusion à 45° sur le bras."
               + (f" Critère élastique retenu : η = σvm / (fy/γM0), fy/γM0 = {F(R.ck['pts'].Rd, 0)} MPa ; "
                  f"Mpl = 0,25·b·tp²·fy/γM0 = {F(ps['Mpl'], 2)} kNm pour "
                  f"{ps['nom'].replace('élément ', 'É').replace('raidisseur ', 'R')} (plastique, information)."
                  if el else ""))
    else:
        lignes, cap = [], "Aucune section : pas de boulon tendu devant une paroi."
    out["sections"] = ("Côté tendu — section au nu de chaque paroi", ent, lignes, cap)
    cs = R.console
    out["console"] = ("Côté comprimé — la bande la plus sollicitée", [
        _lig("Pression sous la platine (répartition rigide)", "p max = " + F(R.pmax, 2) + " MPa ≤ fjd = " + F(R.fjd, 1) + " MPa"),
        _lig("Aire d'appui", ("toute la platine" if R.plein else "bande c = tp·√(fy/(3·fjd·γM0)) = " + F(R.c, 0) + " mm autour des parois")),
        _lig("Bande", cs["nom"].replace("élément ", "É") + " : p au nu = " + F(cs["p0"], 2) + " MPa, console de " + F(cs["L"], 0) + " mm"),
        _lig("Moment pris", "mEd = ∫p·t dt = " + F(cs["m_Ed"], 2) + " kNm/m", True),
        _lig("Résistance élastique", "mRd = tp²·fy/(6·γM0) = " + F(cs["m_Rd"], 2) + " kNm/m → " + (pct(R.ck["pc"].eta, 0) if R.ck["pc"].active else "—")),
        _lig("Épaisseur pour ce moment", "tc = √(6·mEd·γM0/fy) = " + F(cs["t"], 1) + " mm (à tp donné) ; balayage : " + mm(P["t_c"]) + " mm")])
    pb = R.pire_appui
    if pb and pb["Fv"] > 1e-9:
        rows = [_lig("Boulon le plus cisaillé", f"n° {pb['i'] + 1} : Fv = {F(pb['Fv'], 1)} kN (V partagé, torsion, frottement déduit)"),
                _lig("Pinces et entraxes dans le sens de l'effort", f"e1 = {F(pb['e1'], 0)}, p1 = {('∞' if pb['p1'] == INF or pb['p1'] > 1e8 else F(pb['p1'], 0))}, e2 = {F(pb['e2'], 0)}, "
                     f"p2 = {('∞' if pb['p2'] == INF or pb['p2'] > 1e8 else F(pb['p2'], 0))} mm"),
                _lig("Fb,Rd = k1·αb·fu·d·tp/γM2", f"αb = {F(pb['ab'], 2)}, k1 = {F(pb['k1'], 2)} → {F(pb['Fb'], 1)} kN → tb = {F(pb['t_b'], 1)} mm", True)]
    else:
        rows = [_lig("Cisaillement par boulon", "nul (frottement Cf,d·N = " + F(R.Ff_Rd / 1e3, 0) + " kN ≥ V) : pas de pression diamétrale")]
    out["trous"] = ("Trous — pression diamétrale (Tab. 3.4)", rows)
    out["pied"] = ("Chaque épaisseur « balayage » est la plus petite qui satisfait son critère, la répartition étant "
                   "recalculée à chaque épaisseur essayée ; « à tp donné » : l'épaisseur que demande le moment calculé "
                   "avec l'épaisseur retenue.")
    return out


# ---------------------------------------------------------- paramètres retenus
def parametres_complet(R) -> list:
    """ecran_resultats._parametres_retenus (vérification complète)."""
    u = R.u
    g = [["Périmètre", "EN 1993-1-8 seule — béton d'ancrage (EN 1992-4) en information" if R.perim_ec3
          else "Complet — EN 1993-1-8 + EN 1992-4"],
         ["Poteau", R.nom_c + " " + str(u.nu_c) + " – fy = " + F(R.fy_c, 0) + " MPa"],
         ["Platine", js_str(N(u.bp_u)) + " × " + js_str(N(u.lp_u)) + " × " + js_str(N(u.tp_u)) + " mm " + str(u.nu_pl)
          + " – fy = " + F(R.fy_p, 0) + " MPa (épaisseur)"],
         ["Tiges", js_str(R.n_t) + " × " + str(u.ancr) + " " + str(u.classe) + " – d0 = " + F(R.d0, 0) + " mm, As = "
          + F(R.As, 0) + " mm² – " + E.ancrage_txt(u) + (" mm" if E.champ_visible(u, "d_h") else "")],
         ["Béton", "fck = " + F(R.fck, 0) + " MPa, fcd = αcc·fck/γc = " + F(R.fcd, 1) + " MPa, Ecm = "
          + F(R.E_cm / 1e3, 1) + " GPa, " + ("fissuré" if R.fiss else "non fissuré")],
         ["Appui : fjd = βj·fcd·√(Ac1/Ac0)", F(R.fjd, 1) + " MPa (βj = " + F(R.beta_j, 3) + ", √(Ac1/Ac0) = "
          + F(R.ratio_c, 2) + ")"],
         ["Bande c et aire efficace", "c = " + F(R.c, 1) + " mm ; Aeff = " + F(R.A_eff / 100, 0) + " cm² ; NRd = "
          + F(R.N_Rd / 1e3, 0) + " kN"],
         ["Raideurs de la répartition", "kc = " + F(R.k_c, 1) + " N/mm³ ; kb = " + F(R.k_b / 1e3, 0) + " kN/mm"],
         ["Répartition élastique", "C = " + F(R.C, 1) + " kN en (" + F(R.xC, 0) + " ; " + F(R.yC, 0) + ") mm ; pmax = "
          + F(R.pmax, 1) + " MPa ; zone comprimée " + F(R.A_c / 100, 0) + " cm² ; ΣT = " + F(R.T_tot, 1) + " kN"],
         ["Longueur d'allongement Lb = 8d + tg + tp + 0,6d", F(R.Lb, 0) + " mm"],
         ["Résistance d'une tige Ft,anc = min(EC3 ; EC2-4" + (" ; adhérence" if R.Ft_bond is not None else "") + ")",
          F(R.Ft_anc / 1e3, 1) + " kN"],
         ["Cordons : point le plus sollicité", str(R.cordons.get("point", "—")) + " – L = " + F(R.cordons.get("L", 0), 0) + " mm"]]
    if R.rig:
        g.append(["Rigidité Sj,ini / seuil rigide (§5.2.2.5)", F(R.Sj / 1e3, 1) + " / " + F(R.Sj_seuil / 1e3, 1)
                  + " MNm/rad → " + ("RIGIDE" if R.rig["rigide"] else "SEMI-RIGIDE") + " (information)"])
    return g


def parametres_manuel(R) -> list:
    """manuel/ecran._parametres_retenus (calcul manuel)."""
    u = R.u
    return [["Mode", "Calcul manuel — critère " + R.plaque["critere"] + " ; appui : " + ("toute la platine" if R.plein else "bande c")],
            ["Platine", js_str(N(u.bp_u)) + " × " + js_str(N(u.lp_u)) + " × " + js_str(N(u.tp_u)) + " mm " + str(u.nu_pl)
             + " – fy = " + F(R.fy_p, 0) + " MPa (épaisseur), fu = " + F(R.fu_p, 0)],
            ["Éléments soudés", " + ".join(f"{EL.nom(e)} en ({F(e['x'], 0)} ; {F(e['y'], 0)})" + (" tourné 90°" if e["rot"] else "") for e in R.elems) or "—"],
            ["Raidisseurs", ", ".join(f"({F(r['x'], 0)} ; {F(r['y'], 0)}) {r['dir']} L {F(r['L'], 0)} t {F(r['t'], 0)}" for r in R.raid) or "—"],
            ["Boulons", js_str(R.n_t) + " × " + str(u.ancr) + " " + str(u.classe) + " – d0 = " + F(R.d0, 0) + " mm, As = " + F(R.As, 0)
             + " mm² – " + ", ".join(f"({F(t['x'], 0)} ; {F(t['y'], 0)})" for t in R.tiges)],
            ["Béton", "fck = " + F(R.fck, 0) + " MPa, fcd = " + F(R.fcd, 1) + " MPa, fjd = βj·fcd·√(Ac1/Ac0) = " + F(R.fjd, 1) + " MPa"],
            ["Bras : cordon déduit", "0,8·a·√2 = " + F(R.cord, 1) + " mm"],
            ["Raideurs de la répartition", "kc = " + F(R.k_c, 1) + " N/mm³ ; kb = " + F(R.k_b / 1e3, 0) + " kN/mm (" + ("avec" if R.levier else "sans") + " effet de levier)"],
            ["Efforts au point (" + F(R.xN, 0) + " ; " + F(R.yN, 0) + ")", "Mx = " + F(R.Mx_o, 1) + " kNm, My = " + F(R.My_o, 1) + " kNm à l'axe de la platine"],
            ["Longueur d'allongement Lb = 8d + tg + tp + 0,6d", F(R.Lb, 0) + " mm"],
            ["Résistance d'un boulon (acier)", "Ft,Rd = " + F(R.Ft_EC3 / 1e3, 1) + " kN ; NRd,s = " + F(R.N_Rd_s / 1e3, 1) + " kN ; F2,vb,Rd = " + F(R.F2vb / 1e3, 1) + " kN"]]


# ---------------------------------------------------------- la répartition et les efforts des tiges
def repartition(R) -> dict:
    """ecran_resultats.onglet_repartition (vérification complète) : le domaine par direction, les tiges, les rangées."""
    out = {"intro": "La répartition élastique (platine rigide, lit sans traction, tiges tendues) donne l'effort de "
                    "chaque tige — c'est elle qui alimente les vérifications d'EN 1992-4. Le domaine (Mx, My) compare "
                    "le niveau 1 (verdict) et le niveau 2 (information).",
           "titre_domaine": "Domaine résistant (Mx, My) à N = " + F(R.u.N_Ed, 0) + " kN"}
    lignes = []
    for nom, lab in (("x+", "Mx > 0 (comprime y > 0)"), ("x-", "Mx < 0"), ("y+", "My > 0 (comprime x > 0)"),
                     ("y-", "My < 0")):
        lignes.append([lab, F(R.MRd1.get(nom, 0), 1), str(R.branche1.get(nom, "—")), F(R.MRd2.get(nom, 0), 1)])
    out["directions"] = (["Direction", "Niveau 1 (kNm)", "Branche", "Niveau 2 (kNm)"], lignes)
    out["niveaux"] = ("Niveau 1 : " + pct(R.eta1, 1) + " (verdict) · niveau 2 : " + pct(R.eta2, 1)
                      + " (information, M,Rd exact dans la direction appliquée = " + F(R.M_R2, 1) + " kNm).")
    lignes = []
    for t in R.tiges:
        tendue = t["T"] > 1e-6
        lignes.append([F(t["x"], 0) + " ; " + F(t["y"], 0), t["classe"], NO.rangee_lisible(t.get("rangee", "—")),
                       F(t["T"], 1) if tendue else "0", F(t["FT_Rd"] / 1e3, 1),
                       (t["T"] / (t["FT_Rd"] / 1e3), t["T"] <= t["FT_Rd"] / 1e3 + 1e-9) if tendue and t["FT_Rd"] > 0 else "—",
                       F(t["V_ec3"], 1) + " / " + F(t["V_en"], 1)])
    out["tiges"] = ("Tiges : position, classe, efforts",
                    ["x ; y (mm)", "Classe", "Rangée", "T (kN)", "FT,Rd (kN)", "η", "V EC3 / EC2-4 (kN)"], lignes,
                    "Classes : ext = au-delà d'une paroi (Tab. 6.6) ; coin = au-delà de l'about d'aile ; lat = entre "
                    "les niveaux d'aile ; int = entre les ailes (Tab. 6.4). FT,Rd : résistance par tige de sa rangée "
                    "(tronçon en T, acier compris).")
    lignes = []
    for r in R.rangees:
        lignes.append([NO.rangee_lisible(r["cle"]), r["classe"], F(r["m"], 1), F(r["leff1"], 0) + " / " + F(r["leff2"], 0),
                       F(r["Lb"], 0) + " / " + ("∞" if r["Lb_star"] == INF else F(r["Lb_star"], 0)),
                       str(r["mode"]), F(r["FT_Rd"] / 1e3, 1)])
    out["rangees"] = ("Rangées : tronçons en T", ["Rangée", "Classe", "m (mm)", "Σleff,1 / Σleff,2 (mm)", "Lb / Lb* (mm)",
                                                  "Mode", "FT,Rd (kN)"], lignes)
    return out


def efforts_manuel(R) -> dict:
    """manuel/ecran.onglet_efforts : l'effort de chaque boulon (platine rigide) et la répartition."""
    out = {"intro": "Les efforts des tiges viennent de la platine rigide (lit de ressorts sans traction, tiges tendues) — "
                    "le modèle des logiciels d'ancrage : ce sont ces valeurs que l'on compare à Spit ou Hilti. "
                    + ("Appui sur toute la platine, comme eux." if R.plein else "Appui sur la bande c (EN 1993-1-8) : un peu plus "
                       "de pression et de traction que sur toute la platine.")}
    lignes = []
    for t in R.tiges:
        tendu = t["T"] > 1e-6
        lignes.append([js_str(t["i"] + 1), F(t["x"], 0) + " ; " + F(t["y"], 0), F(t["T"], 1) if tendu else "0",
                       (t["T"] / (R.Ft_anc / 1e3), t["T"] <= R.Ft_anc / 1e3 + 1e-9) if tendu else "—",
                       F(t["V_ec3"], 1) + " / " + F(t["V_en"], 1),
                       t["appui"]["face"].replace("élément ", "É") if tendu else "—", F(t["m"], 0) if tendu else "—"])
    out["boulons"] = ("Boulons : position, effort, appui", ["n°", "x ; y (mm)", "T (kN)", "T / Ft,Rd", "V EC3 / brut (kN)",
                                                            "Appui", "m (mm)"], lignes,
                      "V EC3 : frottement Cf,d·N déduit ; brut : sans frottement (valeur des logiciels d'ancrage).")
    out["repartition"] = ("Répartition", [
        ["Résultante comprimée C", F(R.C, 1) + " kN en (" + F(R.xC, 0) + " ; " + F(R.yC, 0) + ") mm"],
        ["Pression maximale", F(R.pmax, 2) + " MPa (fjd = " + F(R.fjd, 1) + ")"],
        ["Zone comprimée", F(R.A_c / 100, 0) + " cm² sur " + F(R.A_eff / 100, 0) + " cm² d'appui"],
        ["Traction totale ΣT", F(R.T_tot, 1) + " kN (T max = " + F(R.T_max, 1) + " kN)"],
        ["Raideurs", "kc = " + F(R.k_c, 1) + " N/mm³ ; kb = " + F(R.k_b / 1e3, 0) + " kN/mm"],
        ["Torsion autour du centre du groupe", F(R.T_G, 2) + " kNm (centre en " + F(R.gx, 0) + " ; " + F(R.gy, 0) + ")"]])
    return out


def note_information(R):
    """La ligne d'information sous le bandeau de l'écran Streamlit — copies nommées de ``ecran_resultats.information``
    (vérification complète : le béton d'ancrage hors périmètre) et de ``manuel/ecran.information`` (calcul à la main :
    tiges et appui) : la vérification d'information la plus sollicitée, hors verdict. Le périmètre est « en tête de
    la carte » dans Streamlit, dans la section Assemblage ici."""
    if CA.est_manuel_R(R):
        pire = YMA.pire_information(R)
        if pire is None:
            return ""
        return ("En information, hors verdict : " + YMA.court(pire) + " à " + pct(pire.eta, 1)
                + (" — non satisfait" if not pire.ok else " — satisfait")
                + ". Le verdict porte sur la platine ; les tiges et le béton relèvent du logiciel d'ancrage ou de "
                  "la vérification complète.")
    pire = YCO.pire_information(R)
    if pire is None:
        return ""
    return ("Béton d'ancrage (EN 1992-4, en information, hors verdict) : " + YCO.court(pire) + " à " + pct(pire.eta, 1)
            + (" — non satisfait" if not pire.ok else " — satisfait")
            + ". Périmètre « EC3 + EC2-4 » (section Assemblage) pour le compter dans le verdict.")


# ================================================================== la mise en forme (HTML du site)
def _cellule(x) -> str:
    """Une cellule : un texte (échappé) ou un taux (η, vérifié) en pastille de couleur."""
    if isinstance(x, tuple):
        eta, ok = x
        return C.chip(A.taux(A.eta_vu(eta, ok), 1), "r" if not ok else C.niveau(A.eta_vu(eta)))
    return C.esc(x)


def _entete(h: str) -> str:
    """Un en-tête de table : le texte de l'écran, ses symboles indicés (« FT,Rd » → F<sub>T,Rd</sub>)."""
    for brut, joli in (("FT,Rd", "F<sub>T,Rd</sub>"), ("Ft,Rd", "F<sub>t,Rd</sub>"), ("Σleff,1 / Σleff,2", "Σl<sub>eff,1</sub> / Σl<sub>eff,2</sub>"),
                       ("leff", "l<sub>eff</sub>"), ("σvm", "σ<sub>vm</sub>"), ("Mpl", "M<sub>pl</sub>"), ("tp ", "t<sub>p</sub> "),
                       ("Lb / Lb*", "L<sub>b</sub> / L<sub>b</sub>*")):
        if brut in h:
            return C.esc(h.replace(brut, "\0")).replace("\0", joli)
    return C.esc(h)


def _table(entetes, lignes, droite=()) -> str:
    """Un tableau du module, qui défile de côté plutôt que de déborder."""
    return ('<div class="asm-defile">' + A.tableau([_entete(h) for h in entetes], [[_cellule(x) for x in li] for li in lignes],
                                                  droite=droite, classe="asm-tab") + "</div>")


def _table_kv(lignes) -> str:
    """Un tableau « libellé · valeur » du calcul à la main (les lignes fortes en gras)."""
    tr = "".join(f'<tr class="{"fort" if fort else ""}"><td>{C.esc(a)}</td><td>{C.esc(b)}</td></tr>' for a, b, fort in lignes)
    return f'<table class="vtab asm-tab asm-kv"><tbody>{tr}</tbody></table>'


def _etat_html(d) -> str:
    """L'épaisseur requise et ses quatre critères : la ligne d'état, puis une pastille par critère (t / tp)."""
    if not d["actif"]:
        return C.message(C.esc(d["note"]))
    crit = "".join(f'<span>{C.esc(nom)} <b class="pct {"r" if not ok else C.niveau(A.eta_vu(r))}">{C.pc(A.taux(A.eta_vu(r, ok)))}{C.NBSP}%</b></span>'
                   for nom, r, ok in d["criteres"])
    return (f'<div class="asm-etat {"p" if d["ok"] else "r"}"><b>{C.esc(d["titre"])}</b><span>{C.esc(d["complement"])}</span></div>'
            f'<div class="asm-crit">{crit}</div>')


def _bloc(titre, corps, large=True) -> str:
    return f'<div class="ck-g{" asm-large" if large else ""}"><div class="ck-t">{C.esc(titre)}</div>{corps}</div>'


def bloc_platine(R) -> str:
    """En tête de la fenêtre Platine : l'épaisseur de la platine comme on la calcule à la main (les deux modes)."""
    if CA.est_manuel_R(R):
        d = platine_manuel(R)
        out = _bloc("Épaisseur de la platine — le calcul à la main", _etat_html(d))
        if not d["actif"]:
            return out
        D = detail_main(R)
        for cle in ("tendus", "sections"):
            titre, ent, lignes, cap = D[cle]
            corps = (_table(ent, lignes, droite=tuple(range(1, len(ent)))) if lignes else "") + C.aide(C.esc(cap))
            out += _bloc(titre, corps)
        for cle in ("console", "trous"):
            titre, lignes = D[cle]
            out += _bloc(titre, _table_kv(lignes), large=False)
        return out + f'<div class="ck-g asm-large">{C.aide(C.esc(D["pied"]))}</div>'
    d = platine_complet(R)
    out = _bloc("Épaisseur de la platine — le calcul à la main", _etat_html(d))
    if not d["actif"]:
        return out
    for cle in ("comprime", "tendu"):
        titre, lignes = d[cle]
        out += _bloc(titre, _table_kv(lignes), large=False)
    return out + f'<div class="ck-g asm-large">{C.aide(C.esc(d["navier"]))}</div>'


def bloc_tiges(R) -> str:
    """La fenêtre Tiges : l'effort de chaque tige (et, en vérification complète, ses rangées en tronçon en T)."""
    if CA.est_manuel_R(R):
        d = efforts_manuel(R)
        titre, ent, lignes, cap = d["boulons"]
        return _bloc(titre, _table(ent, lignes, droite=(2, 3, 4, 6)) + C.aide(C.esc(cap)))
    d = repartition(R)
    titre, ent, lignes, cap = d["tiges"]
    out = _bloc(titre, _table(ent, lignes, droite=(3, 4, 5, 6)) + C.aide(C.esc(cap)))
    titre, ent, lignes = d["rangees"]
    return out + _bloc(titre, _table(ent, lignes, droite=(2, 3, 4, 6)))


def bloc_pied(R) -> str:
    """La fenêtre Pied (vérification complète) : le domaine résistant par direction, niveaux 1 et 2."""
    d = repartition(R)
    ent, lignes = d["directions"]
    return _bloc(d["titre_domaine"], _table(ent, lignes, droite=(1, 3)) + C.aide(C.esc(d["niveaux"])))


def vue_repartition(R) -> str:
    """La vue « Répartition » (onglet Répartition de la vérification complète)."""
    d = repartition(R)
    ent, lignes = d["directions"]
    out = f'<p class="asm-note">{C.esc(d["intro"])}</p><h5>{C.esc(d["titre_domaine"])}</h5>'
    out += _table(ent, lignes, droite=(1, 3)) + f'<p class="asm-note">{C.esc(d["niveaux"])}</p>'
    titre, ent, lignes, cap = d["tiges"]
    out += f"<h4>{C.esc(titre)}</h4>" + _table(ent, lignes, droite=(3, 4, 5, 6)) + f'<p class="asm-note">{C.esc(cap)}</p>'
    titre, ent, lignes = d["rangees"]
    out += f"<h4>{C.esc(titre)}</h4>" + _table(ent, lignes, droite=(2, 3, 4, 6))
    return out


def vue_efforts(R) -> str:
    """La vue « Efforts des tiges » (onglet du calcul manuel)."""
    d = efforts_manuel(R)
    titre, ent, lignes, cap = d["boulons"]
    out = (f'<p class="asm-note">{C.esc(d["intro"])}</p><h4>{C.esc(titre)}</h4>' + _table(ent, lignes, droite=(2, 3, 4, 6))
           + f'<p class="asm-note">{C.esc(cap)}</p>')
    titre, g = d["repartition"]
    return out + f"<h4>{C.esc(titre)}</h4>" + A.tableau(["Grandeur", "Valeur"], [[C.esc(a), f"<b>{C.esc(b)}</b>"] for a, b in g],
                                                        droite=(1,), classe="asm-tab asm-cle")


class _Doc:
    """Une vue « document » de la toile (un tableau, pas un dessin) : rendue comme les vues document de la carte."""

    def __init__(self, html_):
        self.html = html_

    def svg(self, identifiant=""):
        return f'<div class="asm-doc">{self.html}</div>'


def _vues(S, complet):
    def vues(R, niveau, hl):
        opt = S.options_ecran(R, niveau, hl)
        opt.interactive, opt.poignees, opt.editables, opt.locked = False, False, niveau >= 1, None
        out = [("plan", "Vue en plan", "Plan", S.plan(R, opt), LEGENDE_PLAN_CO if complet else LEGENDE_PLAN_MA),
               ("coupe_aa", "Coupe A-A", "Coupe A-A", S.coupe_aa(R, opt), "coupe selon x : platine, mortier, tiges, massif"),
               ("coupe_bb", "Coupe B-B", "Coupe B-B", S.coupe_bb(R, opt), "coupe selon y")]
        if complet:
            out.append(("domaine", "Domaine résistant (Mx, My)", "Domaine", S.domaine(R),
                        f"à N = {F(R.u.N_Ed, 0)} kN · niveau 1 (verdict) et niveau 2 (information)"))
            out.append(("repartition", "Répartition", "Répartition", _Doc(vue_repartition(R)),
                        "efforts des tiges, tronçons en T, domaine par direction"))
        else:
            out.append(("efforts", "Efforts des tiges", "Efforts", _Doc(vue_efforts(R)),
                        "platine rigide, lit sans traction : les valeurs des logiciels d'ancrage"))
        return out
    return vues


# ---------------------------------------------------------- la validation (onglets Benchmark)
@functools.lru_cache(maxsize=1)
def _validation_complet() -> str:
    """ecran_resultats.onglet_benchmark : les calculs manuels (VAL-A, VAL-B), les fiches à compléter, la validation
    croisée ; recalculés par le moteur (ils ne dépendent pas des données de l'assemblage)."""
    B = BCO.run_bench()
    S = BCO.SRC
    out = (f'<div class="asm-kpi"><div><span>Benchmark du module</span><b>{C.esc(B["statut"])}</b></div></div>'
           f'<p class="asm-note">Ces tableaux sont recalculés par le moteur à chaque ouverture ; ils ne dépendent pas des '
           f'données. Source : {C.esc(S["titre"])}. {C.esc(S["org"])}. Norme : {C.esc(S["norme"])}.</p>')
    if not B["bench"]:
        out += ('<p class="asm-note">Aucun exemple publié n\'est rejoué en V1 (lot C : MSB Part 5 pied articulé, SCI P398 '
                'pied encastré, avec le document sous les yeux). Seuil des calculs manuels : écart ≤ 0,5 %.</p>')
    for b in B["bench"]:
        lignes = [[C.esc(r["lab"]), C.esc(r["page"]), C.esc(A.fr_js(r["ref"], _nd(r["ref"])) + ("" if r["unit"] == "-" else " " + r["unit"])),
                   A.fr_js(r["val"], _nd(r["ref"]) + 1), f'{A.fr_js(r["ec"] * 100, 2)} %', C.esc(r["st"])] for r in b["rows"]]
        out += (f"<h5>{C.esc(b['id'])}</h5><p class=\"asm-note\">{C.esc(b['titre'])} — données : {C.esc(b['data'])}</p>"
                + A.tableau(["Vérification", "Réf.", "Référence", "Outil", "Écart", "Statut"], lignes, droite=(2, 3, 4), classe="asm-tab"))
    for f in B.get("fiches", []):
        out += (f"<h5>{C.esc(f['id'])} – fiche à compléter</h5>"
                f"<p class=\"asm-note\">{C.esc(f['titre'] + ' — ' + f.get('source', '') + '. ' + f.get('data', ''))}</p>")
    out += "<h4>Vérifications sans exemple publié consulté</h4><ul>" + "".join(
        f"<li><b>{C.esc(x[0])}</b> – {C.esc(x[1])}</li>" for x in BCO.NOCOVER) + "</ul>"
    V = B.get("croisee")
    if V:
        out += (f"<h5>Validation croisée indépendante du {C.esc(V['date'])} — {V['configurations']} configurations, "
                f"{V['directions']} directions</h5>"
                f"<p class=\"asm-note\">Hors moteur (numpy + scipy) : grille de ressorts pour la répartition élastique, "
                f"programme linéaire (HiGHS) pour le domaine résistant — le domaine exact des champs admissibles bornés. "
                f"Rapport : {C.esc(V['rapport'])} (ses figures y sont).</p>"
                + A.tableau(["Grandeur", "Écart maximal", "Tolérance"], [[C.esc(x) for x in e] for e in V["ecarts"]],
                            droite=(1, 2), classe="asm-tab"))
    for c in B["valid"]:
        lignes = []
        for r in c["rows"]:
            n = 4 if abs(r["main"]) < 10 else (2 if abs(r["main"]) < 1000 else 0)
            lignes.append([C.esc(r["lab"]), C.esc(A.fr_js(r["main"], n) + ("" if r["unit"] == "-" else " " + r["unit"])),
                           A.fr_js(r["ref"], n), A.fr_js(r["val"], n), f'{A.fr_js(r["ec"] * 100, 3)} %',
                           "OK" if r["ok"] else "À CONTRÔLER"])
        out += (f"<h5>{C.esc(c['id'])} – calcul manuel indépendant</h5>"
                f"<p class=\"asm-note\">{C.esc(c['titre'])}. Colonne « manuel » : calcul indépendant hors moteur ; colonne "
                f"« contrôle » : même valeur recomposée à la main (arrondie).</p>"
                + A.tableau(["Grandeur", "Manuel", "Contrôle", "Outil", "Écart", ""], lignes, droite=(1, 2, 3, 4), classe="asm-tab"))
    out += ('<p class="asm-note">Pour rejouer un cas : section « Cartouche et affichage », « Exemples et cas de '
            'validation ».</p>')
    return f'<div class="asm-doc">{out}</div>'


def _nd(x):
    return 2 if abs(x) < 1 else (1 if abs(x) < 10 else 0)


@functools.lru_cache(maxsize=1)
def _validation_manuel() -> str:
    """manuel/ecran.onglet_benchmark : les calculs à la main VAL-M1 à VAL-M4 rejoués par le moteur."""
    B = BMA.run_bench_manuel()
    out = (f'<div class="asm-kpi"><div><span>Benchmark du calcul manuel</span><b>{C.esc(B["statut"])}</b></div></div>'
           '<p class="asm-note">Chaque cas est recalculé à la main dans tests/test_platine_manuelle.py (formule par '
           'formule) et rejoué ici. Seuil : écart ≤ 0,5 %.</p>')
    for c in B["cas"]:
        lignes = [[C.esc(r["lab"]), C.esc(A.fr_js(r["main"], 3) + ("" if r["unit"] == "-" else " " + r["unit"])),
                   A.fr_js(r["val"], 3), f'{A.fr_js((r["ec"] if abs(r["ec"]) >= 5e-6 else 0.0) * 100, 3)} %',
                   "OK" if r["ok"] else "À CONTRÔLER"] for r in c["rows"]]
        out += (f"<h5>{C.esc(c['id'])} – {C.esc(c['titre'])}</h5><p class=\"asm-note\">{C.esc(c['data'])}</p>"
                + A.tableau(["Grandeur", "Manuel", "Outil", "Écart", ""], lignes, droite=(1, 2, 3), classe="asm-tab"))
    out += ('<p class="asm-note">Pour rejouer un cas : section « Cartouche et affichage », « Exemples et cas de '
            'validation ».</p>')
    return f'<div class="asm-doc">{out}</div>'


def exemples():
    """« Charger ces données dans l'outil » des deux onglets Benchmark : les exemples publiés et les calculs manuels
    de la vérification complète (les défauts, les données du cas, le mode complet — ecran_resultats._charger), puis
    les calculs manuels de la platine à la main (les défauts, les données du cas — manuel/ecran._charger)."""
    out = []
    for x in list(BCO.BENCH) + list(BCO.VALID):
        d = E.defaults()
        d.update(copy.deepcopy(x["inp"]))
        d["mode"] = E.MODE_COMPLET
        out.append((x["id"], f"{x['id']} · vérification complète · {x['titre'][:46]}", d))
    for x in BMA.CAS:
        d = E.defaults()
        d.update(copy.deepcopy(x["inp"]))
        out.append((x["id"], f"{x['id']} · calcul manuel · {x['titre'][:46]}", d))
    return out


# ================================================================== le cadre Données
LIBELLES = {
    "id_projet": ("", "Projet"), "id_rep": ("", "Repère"), "id_red": ("", "Rédacteur"), "id_date": ("", "Date"),
    "mode": ("", "Mode"), "crit": ("", "Critère"), "appui": ("", "Appui"), "perim": ("", "Périmètre"),
    "xN": ("x<sub>N</sub>", "selon x"), "yN": ("y<sub>N</sub>", "selon y"),
    "forme": ("", "Forme"), "prof_c": ("", "Profilé"), "h_u": ("h", "hauteur"), "b_u": ("b", "largeur"),
    "tw_u": ("t<sub>w</sub>", "âme"), "tf_u": ("t<sub>f</sub>", "aile"), "r_u": ("r", "congé"), "t_u": ("t", "paroi"),
    "nu_c": ("", "Nuance"), "L_c": ("L<sub>c</sub>", "hauteur"), "contrev": ("", "Ossature contreventée"),
    "bp_u": ("b<sub>p</sub>", "largeur (x)"), "lp_u": ("l<sub>p</sub>", "longueur (y)"), "tp_u": ("t<sub>p</sub>", "épaisseur"),
    "nu_pl": ("", "Nuance"), "a_w": ("a", "gorge"),
    "n_ax": ("n<sub>x</sub>", "files (x)"), "n_ay": ("n<sub>y</sub>", "rangées (y)"),
    "w_x": ("w<sub>x</sub>", "entraxe (x)"), "w_y": ("w<sub>y</sub>", "entraxe (y)"),
    "ancr": ("", "Tige"), "classe": ("", "Classe"), "typ_a": ("", "Ancrage"),
    "hef": ("h<sub>ef</sub>", "profondeur"), "d_h": ("d<sub>h</sub>", "tête / plaque"),
    "eta_tau": ("τ<sub>Rk</sub>", "adhérence"), "eta_k1": ("k<sub>1</sub>", "cône de béton"),
    "eta_nrks": ("N<sub>Rk,s</sub>", "traction"), "eta_vrks": ("V<sub>Rk,s</sub>", "cisaillement"),
    "beton": ("", "Béton"), "fiss": ("", "Béton fissuré"), "a_cc": ("α<sub>cc</sub>", "f<sub>cd</sub> = α<sub>cc</sub>·f<sub>ck</sub>/γ<sub>c</sub>"),
    "bc_u": ("b<sub>c</sub>", "largeur (x)"), "lc_u": ("l<sub>c</sub>", "longueur (y)"), "hc_u": ("h<sub>c</sub>", "hauteur"),
    "t_g": ("t<sub>g</sub>", "épaisseur"), "f_g": ("f<sub>g</sub>", "résistance"),
    "N_Ed": ("N<sub>Ed</sub>", "effort normal"), "Mx_Ed": ("M<sub>x,Ed</sub>", "moment"),
    "My_Ed": ("M<sub>y,Ed</sub>", "moment"), "Vx_Ed": ("V<sub>x,Ed</sub>", "tranchant"),
    "Vy_Ed": ("V<sub>y,Ed</sub>", "tranchant"), "T_Ed": ("T<sub>Ed</sub>", "torsion"),
    "g_M0": ("γ<sub>M0</sub>", "acier"), "g_M2": ("γ<sub>M2</sub>", "boulons"), "g_c": ("γ<sub>c</sub>", "béton"),
    "g_Mc": ("γ<sub>Mc</sub>", "EN 1992-4"), "C_f": ("C<sub>f,d</sub>", "frottement"),
    "fy_u": ("f<sub>y</sub>", "limite élastique"), "fu_u": ("f<sub>u</sub>", "résistance"), "bw_u": ("β<sub>w</sub>", "corrélation"),
}
ENTIERS = frozenset({"n_ax", "n_ay"})
SEGMENTS = frozenset({"mode", "perim", "crit", "appui", "forme"})


# Les aides de l'écran qui renvoient au dessin interactif (« hef et dh se règlent sur la coupe A-A », « chaque boulon se
# déplace aussi sur le dessin ») : le site les saisit dans le cadre Données, son aide le dit.
AIDES_SITE = {"typ_a": "Tête / plaque : hef et dh (cotes de la coupe A-A), dans cette section. Crosse : adhérence. "
                       "Scellée : valeurs de l'ETA (τRk, k1, NRk,s, VRk,s).",
              "tiges": "Origine au centre de la platine, x vers la droite, y vers le haut (vue en plan)."}


def aides(u) -> dict:
    """Les « ? » de la carte Streamlit, ceux du mode de ``u`` (le calcul manuel ajoute les siens) ; celles qui
    renvoient au dessin interactif disent où se fait la saisie sur le site (``AIDES_SITE``)."""
    return dict(AIDES_CO if E.est_complet(u) else AIDES_MA, **AIDES_SITE)


def _moins(chemin, titre):
    return (f'<button type="button" class="moins" data-chemin="{C._attr_chemin(chemin)}" data-valeur="1" title="{C.esc(titre)}">'
            f'{C.icon("x")}</button>')


def _ajout(chemin, texte):
    return f'<button type="button" class="addrow" data-chemin="{C._attr_chemin(chemin)}" data-valeur="1">{C.icon("plus")}{C.esc(texte)}</button>'


def _cellule_trio(lab, champ_):
    return f"<label><span>{lab}</span>{champ_}</label>"


def table_tiges(u, el) -> str:
    """Les boulons du calcul manuel (une ligne par boulon : n°, x, y, retirer), « Ajouter un boulon » (au bord bas,
    comme l'écran Streamlit), puis « Poser une grille » (files, rangées, entraxes)."""
    tiges = u["tiges"]
    out = f'<div class="grp">Position des boulons <span class="aide" title="{C.esc(AIDES_SITE["tiges"])}">{C.icon("info")}</span></div>'
    if tiges:
        out += '<div class="lit asm-th"><span>n°</span><span>x</span><span>y</span><span></span></div>'
    for i, t in enumerate(tiges):
        out += (f'<div class="lit"><span>n° {i + 1}</span>{C.num(["v", "tiges", i, "x"], t["x"], "mm")}'
                f'{C.num(["v", "tiges", i, "y"], t["y"], "mm")}{_moins(["v", "tiges", i, "-"], "Retirer ce boulon")}</div>')
    if not tiges:
        out += C.message("Aucun boulon : la platine n'est pas ancrée (vérification impossible).", grave=True)
    out += _ajout(["v", "tiges", "+"], "Ajouter un boulon")
    g = el.get("grille") or GRILLE_DEFAUT
    out += ('<div class="grp">Poser une grille</div>'
            + C.ligne("files (x)", C.num(["grille", "nx"], g["nx"], "", entier=True, mini=1), "n<sub>x</sub>")
            + C.ligne("rangées (y)", C.num(["grille", "ny"], g["ny"], "", entier=True, mini=1), "n<sub>y</sub>")
            + C.ligne("entraxe (x)", C.num(["grille", "wx"], g["wx"], "mm", mini=0), "w<sub>x</sub>")
            + C.ligne("entraxe (y)", C.num(["grille", "wy"], g["wy"], "mm", mini=0), "w<sub>y</sub>")
            + f'<button type="button" class="addrow" data-chemin="grille/poser" data-valeur="1" '
              f'title="Remplace les {len(tiges)} boulons par la grille centrée">{C.icon("grid-2x2")}'
              f'Poser la grille {g["nx"]} × {g["ny"]}</button>' + C.aide("La grille remplace les boulons posés."))
    return out


def table_elements(u) -> str:
    """Les éléments soudés : une carte par élément (type, profilé, position, rotation, dimensions d'une section sans
    catalogue), « Ajouter un élément soudé »."""
    cartes = ""
    profils = [(p, p) for p in [PERSO] + [x["n"] for x in DB["profils"]]]
    for i, e in enumerate(u["elems"]):
        ch = ["v", "elems", i]
        perso = e["type"] != E.TYPE_I or e["prof"] == PERSO
        corps = C.ligne("Type", C.choix(ch + ["type"], e["type"], [(t, AFF_TYPES[t]) for t in E.TYPES_ELEM]), "", "", full=True)
        if e["type"] == E.TYPE_I:
            corps += C.ligne("Profilé", C.choix(ch + ["prof"], e["prof"], profils), "", "", full=True)
        corps += ('<div class="trio">' + _cellule_trio("x (mm)", C.num(ch + ["x"], e["x"], ""))
                  + _cellule_trio("y (mm)", C.num(ch + ["y"], e["y"], ""))
                  + _cellule_trio("rotation", C.choix(ch + ["rot"], str(int(e["rot"])), [("0", "0°"), ("90", "90°")])) + "</div>")
        if perso:
            dims = DIMS_ELEM[e["type"]]
            corps += '<div class="trio">' + "".join(_cellule_trio(f"{LIB_DIM[d]} (mm)", C.num(ch + [d], e[d], "")) for d in dims) + "</div>"
        cartes += (f'<div class="secard"><div class="sh2"><span class="indice">É{i + 1}</span>{C.esc(EL.nom(e))}'
                   f'<span class="muted">· ({C.nb(e["x"])} ; {C.nb(e["y"])})</span>'
                   f'<span class="asm-fin">{_moins(ch + ["-"], "Retirer cet élément")}</span></div>{corps}</div>')
    vide = "" if u["elems"] else C.message("Aucun élément soudé : rien ne porte sur la platine (vérification impossible).", grave=True)
    return (C.aide(C.esc(AIDES_MA["ÉLÉMENTS"].split(". Rotation")[0] + ".")) + vide
            + (f'<div class="secards">{cartes}</div>' if cartes else "") + _ajout(["v", "elems", "+"], "Ajouter un élément soudé"))


def table_raidisseurs(u) -> str:
    """Les raidisseurs : une carte par raidisseur (position, sens, longueur, épaisseur, hauteur)."""
    cartes = ""
    for i, r in enumerate(u["raid"]):
        ch = ["v", "raid", i]
        corps = ('<div class="trio">' + _cellule_trio("x (mm)", C.num(ch + ["x"], r["x"], ""))
                 + _cellule_trio("y (mm)", C.num(ch + ["y"], r["y"], ""))
                 + _cellule_trio("sens", C.choix(ch + ["dir"], r["dir"], [(d, d) for d in E.DIRS_RAID])) + "</div>"
                 + '<div class="trio">' + _cellule_trio("L (mm)", C.num(ch + ["L"], r["L"], ""))
                 + _cellule_trio("t (mm)", C.num(ch + ["t"], r["t"], ""))
                 + _cellule_trio("h (mm)", C.num(ch + ["h"], r["h"], "")) + "</div>")
        cartes += (f'<div class="secard"><div class="sh2"><span class="indice">R{i + 1}</span>{C.esc(r["dir"])} · '
                   f'L {C.nb(r["L"])} × t {C.nb(r["t"])}<span class="muted">· ({C.nb(r["x"])} ; {C.nb(r["y"])})</span>'
                   f'<span class="asm-fin">{_moins(ch + ["-"], "Retirer ce raidisseur")}</span></div>{corps}</div>')
    return (C.aide(C.esc(AIDES_MA["RAIDISSEURS"])) + (f'<div class="secards">{cartes}</div>' if cartes else "")
            + _ajout(["v", "raid", "+"], "Ajouter un raidisseur"))


def _resume_efforts(u) -> str:
    morceaux = []
    for k, sym, un in (("N_Ed", "N", "kN"), ("Mx_Ed", "M<sub>x</sub>", "kN·m"), ("My_Ed", "M<sub>y</sub>", "kN·m"),
                       ("Vx_Ed", "V<sub>x</sub>", "kN"), ("Vy_Ed", "V<sub>y</sub>", "kN"), ("T_Ed", "T", "kN·m")):
        x = N(u.get(k))
        if x or k == "N_Ed":
            morceaux.append(f"{sym} {C.nb(x or 0)}")
    return " · ".join(morceaux)


def _platine_resume(u) -> str:
    return f"{C.nb(N(u.get('bp_u')))} × {C.nb(N(u.get('lp_u')))} × {C.nb(N(u.get('tp_u')))} · {C.esc(u.get('nu_pl') or '')}"


def _beton_resume(u) -> str:
    return f"{C.esc(u.get('beton') or '')} · {C.nb(N(u.get('bc_u')))} × {C.nb(N(u.get('lc_u')))} × {C.nb(N(u.get('hc_u')))}"


def _coefficients(u, complet) -> list:
    items = ["g_M0", "g_M2", "g_c"] + (["g_Mc"] if complet else []) + ["a_cc", "C_f", ("aide", C.esc(AIDES_CO["COEFF. γ"]))]
    if E.acier_perso(u):
        items += [("grp", "Acier personnalisé"), "fy_u", "fu_u", "bw_u"]
    return items


def _sections_manuel(u, R, el):
    cfg = A.CONFIGS["platine_beton"]
    n_t, n_e, n_r = len(u["tiges"]), len(u["elems"]), len(u["raid"])
    elems = ", ".join(C.esc(EL.nom(EL.normaliser(e))) for e in u["elems"][:2]) + (" …" if n_e > 2 else "") or "aucun"
    return [
        ("as", "Assemblage", f"calcul manuel · {C.esc(AFFICHAGE['crit'].get(u.get('crit'), '')).lower()} · "
                             f"{C.esc(AFFICHAGE['appui'].get(u.get('appui'), '')).lower()}",
         ["id_rep", "mode", ("aide", C.esc(etat_bascule(u))), "crit", "appui"], True),
        ("pl", "Platine", _platine_resume(u),
         ["bp_u", "lp_u", "tp_u", "nu_pl", ("grp", "Soudure sur les éléments"), "a_w", ("aide", C.esc(NOTE_GORGE_MA))], True),
        ("bo", "Boulons", f"{n_t} × {C.esc(u.get('ancr') or '')} {C.esc(u.get('classe') or '')} · {C.esc(E.ancrage_txt(u, True))}",
         [("lib", "ancr", ("", "Boulon")), "classe", "typ_a", "hef", "d_h", ("html", table_tiges(u, el))], True),
        ("el", "Éléments soudés", elems, [("html", table_elements(u))], True),
        ("ra", "Raidisseurs", f"{n_r} raidisseur{'s' if n_r > 1 else ''}" if n_r else "aucun",
         [("html", table_raidisseurs(u))], bool(n_r)),
        ("be", "Béton et massif", _beton_resume(u),
         ["beton", "bc_u", "lc_u", "hc_u", ("grp", "Mortier de calage"), "t_g", "f_g"], False),
        ("ef", "Efforts ELU", _resume_efforts(u),
         ["N_Ed", "Mx_Ed", "My_Ed", "Vx_Ed", "Vy_Ed", "T_Ed", ("aide", C.esc(AIDES_CO["EFFORTS ELU"])),
          ("grp", "Point d'application"), "xN", "yN"], True),
        ("co", "Coefficients", f"γ<sub>M0</sub> {C.nb(N(u.get('g_M0')))} · γ<sub>M2</sub> {C.nb(N(u.get('g_M2')))} · "
                               f"γ<sub>c</sub> {C.nb(N(u.get('g_c')))}", _coefficients(u, False), False),
        ("ca", "Cartouche et affichage", C.esc(u.get("id_red") or "rédacteur à saisir"),
         A.section_cartouche(cfg, u, el, libelle_visserie="Visserie",
                             aide_visserie=AIDES_MA["visserie"]), False),
    ]


def _sections_complet(u, R, el):
    cfg = A.CONFIGS["platine_beton"]
    prof = u.get("prof_c") if E.forme_I(u) else "tube RHS / SHS"
    eta = [("grp", "Valeurs de l'ETA"), ("aide", C.esc(AIDES_CO["ETA"])), "eta_tau", "eta_k1", "eta_nrks", "eta_vrks"] if E.eta(u) else []
    return [
        ("as", "Assemblage", f"vérification complète · {C.esc(AFFICHAGE['perim'].get(u.get('perim'), ''))}",
         ["id_rep", "mode", "perim"], True),
        ("po", "Poteau", f"{C.esc(prof)} · {C.esc(u.get('nu_c') or '')}",
         ["forme", "prof_c", "h_u", "b_u", "tw_u", "tf_u", "r_u", "t_u", "nu_c", ("grp", "Rigidité du pied (information)"),
          "L_c", "contrev"], True),
        ("pl", "Platine", _platine_resume(u),
         ["bp_u", "lp_u", "tp_u", "nu_pl", ("grp", "Soudure poteau / platine"), "a_w"], True),
        ("ti", "Tiges d'ancrage", f"{C.nb(N(u.get('n_ax')))} × {C.nb(N(u.get('n_ay')))} {C.esc(u.get('ancr') or '')} "
                                  f"{C.esc(u.get('classe') or '')} · {C.esc(E.ancrage_txt(u, True))}",
         [("grp", "Grille centrée sur le poteau"), "n_ax", "n_ay", "w_x", "w_y", ("grp", "Tiges"),
          ("lib", "ancr", ("", "Tige")), "classe", "typ_a", "hef", "d_h"] + eta, True),
        ("be", "Béton et massif", _beton_resume(u),
         ["beton", "fiss", "bc_u", "lc_u", "hc_u", ("grp", "Mortier de calage"), "t_g", "f_g"], False),
        ("ef", "Efforts ELU", _resume_efforts(u),
         ["N_Ed", "Mx_Ed", "My_Ed", "Vx_Ed", "Vy_Ed", "T_Ed", ("aide", C.esc(AIDES_CO["EFFORTS ELU"]))], True),
        ("co", "Coefficients", f"γ<sub>M0</sub> {C.nb(N(u.get('g_M0')))} · γ<sub>M2</sub> {C.nb(N(u.get('g_M2')))} · "
                               f"γ<sub>Mc</sub> {C.nb(N(u.get('g_Mc')))}", _coefficients(u, True), False),
        ("ca", "Cartouche et affichage", C.esc(u.get("id_red") or "rédacteur à saisir"),
         A.section_cartouche(cfg, u, el, libelle_visserie="Visserie",
                             aide_visserie=AIDES_CO["visserie"]), False),
    ]


def _sections(u, R, el):
    return _sections_complet(u, R, el) if E.est_complet(u) else _sections_manuel(u, R, el)


# ================================================================== la saisie : tables, grille, bascule
def _liste(normal):
    def normaliser(x):
        if not isinstance(x, list):
            raise ValueError("table du calcul à la main illisible")
        if len(x) > A.LIGNES_MAXI:
            raise ValueError(f"table du calcul à la main trop longue : {len(x)} lignes, {A.LIGNES_MAXI} au plus")
        return [normal(e) for e in x]
    return normaliser


def _tige(t):
    if not isinstance(t, dict):
        raise ValueError("boulon illisible")
    return {"x": float(N(t.get("x")) or 0.0), "y": float(N(t.get("y")) or 0.0)}


# Les champs des tables : les positions sont signées, les dimensions positives ou nulles (l'écran Streamlit les
# accepte négatives : un raidisseur les ramène à 0 en silence, un élément les garde) ; la rotation vaut 0 ou 90°.
POSITIONS = frozenset({"x", "y"})
CHOIX_LISTES = {"type": lambda v: v in E.TYPES_ELEM, "dir": lambda v: v in E.DIRS_RAID,
                "prof": lambda v: v == PERSO or any(p["n"] == v for p in DB["profils"])}


def _modifier_liste(cle, nouveau, normal):
    """Une saisie dans une table : [« + »] ajoute (``nouveau(u)``), [i, « - »] retire (jusqu'au dernier, comme le 🗑️
    de l'écran Streamlit : le moteur dit alors « aucun boulon » ou « aucun élément »), [i, champ] écrit."""
    def modifier(u, reste, valeur):
        lst = u[cle]
        if reste == ["+"]:
            if len(lst) >= A.LIGNES_MAXI:
                return f"Table : {A.LIGNES_MAXI} lignes au plus"
            lst.append(normal(nouveau(u)))
            return None
        if len(reste) != 2:
            return "Saisie refusée : chemin de table inconnu"
        i, champ_ = int(reste[0]), reste[1]
        if not 0 <= i < len(lst):
            return "Saisie refusée : ligne inconnue"
        if champ_ == "-":
            del lst[i]
            return None
        e = dict(lst[i])
        if champ_ in CHOIX_LISTES:
            v = str(valeur).replace("-", "−") if champ_ == "dir" else str(valeur)
            if not CHOIX_LISTES[champ_](v):
                return f"Option inconnue : « {valeur} »"
            e[champ_] = v
        elif champ_ in e:
            x = C.lire_nombre(valeur)
            if x is None:
                return f"Nombre illisible : « {valeur} »"
            if champ_ == "rot" and x not in (0, 90):
                return "Rotation : 0° ou 90°"
            if champ_ not in POSITIONS and x < 0:
                return f"{champ_} : une dimension positive ou nulle est attendue (« {valeur} »)"
            e[champ_] = x
        else:
            return f"Champ inconnu : {champ_}"
        lst[i] = normal(e)
        return None
    return modifier


LISTES = {
    "tiges": {"normaliser": _liste(_tige), "modifier": _modifier_liste("tiges", ajouter_boulon, _tige)},
    "elems": {"normaliser": _liste(EL.normaliser), "modifier": _modifier_liste("elems", lambda u: E.element_defaut(), EL.normaliser)},
    "raid": {"normaliser": _liste(EL.normaliser_raid), "modifier": _modifier_liste("raid", lambda u: E.raidisseur_defaut(), EL.normaliser_raid)},
}


# la grille à poser : 20 × 20 boulons au plus (l'écran Streamlit n'a pas de maximum ; 10⁵ × 10⁵ boulons épuiseraient la
# mémoire de l'onglet) — les mêmes bornes que les tiges de la vérification complète (assemblages.COMPTES_MAXI)
GRILLE_MAXI = 20


def _grille(g):
    """La grille à poser, relue (un nombre de files ou de rangées entier ≥ 1, des entraxes ≥ 0)."""
    out = dict(GRILLE_DEFAUT)
    if isinstance(g, dict):
        for k in ("nx", "ny"):
            x = C.lire_nombre(g.get(k))
            if x is not None and 1 <= x <= GRILLE_MAXI and x == int(x):
                out[k] = int(x)
        for k in ("wx", "wy"):
            x = C.lire_nombre(g.get(k))
            if x is not None and x >= 0:
                out[k] = float(x)
    return out


def _action_grille(cfg, el, reste, valeur):
    """[« grille », nx | ny | wx | wy] : un paramètre de la grille ; [« grille », « poser »] : la grille remplace les
    boulons (« Poser la grille » de l'écran Streamlit)."""
    g = _grille(el.get("grille"))
    if reste == ["poser"]:
        if E.est_complet(el["values"]):
            return "La grille se pose dans le calcul manuel (vérification complète : files, rangées, entraxes)"
        el["values"]["tiges"] = poser_grille(g)
        el["grille"] = g
        return None
    if len(reste) != 1 or reste[0] not in GRILLE_DEFAUT:
        return f"Champ inconnu : grille/{'/'.join(str(c) for c in reste)}"
    k = reste[0]
    x = C.lire_nombre(valeur)
    if x is None:
        return f"Nombre illisible : « {valeur} »"
    if k in ("nx", "ny") and (x < 1 or x != int(x)):
        return "Un nombre entier ≥ 1 de files ou de rangées est attendu"
    if k in ("nx", "ny") and x > GRILLE_MAXI:
        return f"Grille : {GRILLE_MAXI} files ou rangées au plus (« {valeur} »)"
    if x < 0:
        return "Un entraxe positif ou nul est attendu"
    g[k] = int(x) if k in ("nx", "ny") else float(x)
    el["grille"] = g
    return None


def _changer_mode(cfg, el, x):
    """La bascule de mode (ecran_saisie.changer_mode) : la disposition est reprise d'un mode à l'autre ; vers la
    vérification complète seulement si elle est régulière — sinon le mode reste manuel et la raison est dite."""
    u = el["values"]
    if x == u.get("mode"):
        return None
    if x == E.MODE_COMPLET:
        v = CV.vers_complet(dict(u, mode=E.MODE_MANUEL))
        if v is None:
            return "Vérification complète impossible : " + CV.disposition_reguliere(u)["raison"] + "."
    else:
        v = CV.vers_manuel(dict(u, mode=E.MODE_COMPLET))
    el["values"] = A.valeurs_normales(cfg, v)
    return None


# ================================================================== le résultat
def _piece(R, u, grp):
    complet = not CA.est_manuel_R(R)
    if grp == "Tiges":
        n = R.get("n_t") or len(R.get("tiges") or [])
        return C.esc(f"{n} × {u.get('ancr')} {u.get('classe')}"), "ancr"
    if grp == "Platine":
        return C.esc(f"{C.nb(N(u.get('bp_u')))} × {C.nb(N(u.get('lp_u')))} × {C.nb(N(u.get('tp_u')))} mm · {u.get('nu_pl')}"), ""
    if grp == "Poteau" and complet:
        return C.esc(f"{R.nom_c} · {u.get('nu_c')}"), ("prof_c" if E.forme_I(u) else "")
    if grp in ("Béton", "Appui"):
        return f"{C.esc(u.get('beton'))} · f<sub>jd</sub> = {C.fr(R.fjd, 1)} N/mm²", ""
    if grp == "Ancrages":
        return C.esc(f"{E.ancrage_txt(u, True)} · hef {C.nb(N(u.get('hef')))} mm"), ""
    if grp == "Soudure":
        return f"a = {C.nb(N(u.get('a_w')))} mm", ""
    if grp == "Trous":
        return C.esc(f"{u.get('ancr')} · d0 = {C.nb(R.d0)} mm"), ""
    if grp == "Pied":
        return "N + M<sub>x</sub> + M<sub>y</sub> · Tab. 6.7", ""
    return "", ""


def lignes_resultat(R, u) -> dict:
    """Les quatre critères de l'épaisseur requise, SOUS la ligne Platine (les pastilles du bandeau de l'écran
    Streamlit : t / t_p de chaque critère) ; chacun ouvre la fenêtre Platine."""
    c = R.ck.get("tpq")
    tp = N(u.get("tp_u")) or 0.0
    if c is None or not c.active or tp <= 0:
        return {}
    P = R.plaque
    manuel = CA.est_manuel_R(R)
    noms, ordre = (CRITERES_NOMS, ORDRE_MA) if manuel else (CRITERES_CO, ORDRE_CO)
    out = ""
    for k in ordre:
        t = P["t_" + k]
        r = t / tp
        ok = t <= tp + 1e-9
        piece = f't<sub>{k}</sub> = {"∞" if t == INF else C.fr(t, 1)} mm'
        out += C.ligne_res(noms[k][0].upper() + noms[k][1:], piece, A.taux(A.eta_vu(r, ok)), "r" if not ok else C.niveau(A.eta_vu(r)),
                           calcul="platine")
    return {"Platine": f'<div class="asm-sous" title="Épaisseur requise : les quatre critères (t / t_p)">{out}</div>'}


def complement_verdict(R) -> str:
    """Le critère qui gouverne l'épaisseur, quand l'épaisseur requise gouverne le verdict (le bandeau de l'écran)."""
    g = R.gov
    if g is None or g.key != "tpq" or not g.active:
        return ""
    P = R.plaque
    if CA.est_manuel_R(R):
        return " · " + C.esc(CRITERES_NOMS[P["pire"]])
    return " · " + C.esc(CRITERES_CO[max(ORDRE_CO, key=lambda k: P["t_" + k])])


def hypotheses(R, u) -> list:
    """Les pastilles de plus : les tiges (ou boulons), le béton, le mode."""
    manuel = CA.est_manuel_R(R)
    n = R.get("n_t") or len(R.get("tiges") or [])
    h = [f"{'Boulons' if manuel else 'Tiges'} <b>{n} × {C.esc(u.get('ancr'))} {C.esc(u.get('classe'))}</b> · d<sub>0</sub> = "
         f"{C.nb(R.d0)} mm · A<sub>s</sub> = {C.nb(R.As)} mm² · f<sub>ub</sub> = {C.nb(R.f_ub)} N/mm²",
         f"Béton {C.esc(u.get('beton'))} · f<sub>ck</sub> = {C.nb(R.fck)} · f<sub>cd</sub> = {C.fr(R.fcd, 1)} · f<sub>jd</sub> = "
         f"{C.fr(R.fjd, 1)} N/mm² · γ<sub>c</sub> = {C.nb(N(u.get('g_c')))}"]
    if manuel:
        h.append(f"Critère {C.esc(AFFICHAGE['crit'].get(u.get('crit'), '')).lower()} · appui "
                 f"{'sur toute la platine' if R.plein else 'sur la bande c'}")
    else:
        h.append("Périmètre " + ("EN 1993-1-8, béton d'ancrage en information" if R.perim_ec3 else "EN 1993-1-8 + EN 1992-4"))
    return h


def blocs_tete(R, grp) -> str:
    return bloc_platine(R) if grp == "Platine" else ""


def blocs(R, grp) -> str:
    if grp == "Tiges":
        return bloc_tiges(R)
    if grp == "Pied" and not CA.est_manuel_R(R):
        return bloc_pied(R)
    return ""


# ================================================================== les sélecteurs : la tige, le poteau
def sel_tige(cfg, u, familles=None) -> str:
    """Le sélecteur des tiges (boulons du calcul manuel) : chaque diamètre du catalogue, le taux de l'assemblage
    recalculé par le moteur."""
    rows = []
    for b in DB["boulons"]:
        R = A._avec(cfg, u, ancr=b["n"])
        rows.append(A._ligne_sel(b["n"], C.fr(b["As"], 0), R, b["n"] == u.get("ancr"),
                                 f"{b['n']} · d<sub>0</sub> = {C.nb(b['d0'])} mm · A<sub>s</sub> = {C.fr(b['As'], 1)} mm²"))
    A._marquer_optimal(rows)
    nom = LIBELLE_MODE[E.MODE_COMPLET if E.est_complet(u) else E.MODE_MANUEL]["ancr"]
    return C.selecteur("ancr", f"{nom}s · classe {C.esc(u.get('classe') or '')}", "triées par diamètre · un clic retient le diamètre",
                       "", rows, A._debut(rows), "A<sub>s</sub> mm²", etoile="le plus petit qui tient")


def _dimensions():
    """Les valeurs PROVISOIRES des dimensions vides du poteau personnalisé (``ecran_saisie.DIMENSIONS`` — copie
    nommée, liée par tests/test_api_web_assemblages.py) : celles du profilé par défaut, la paroi d'un tube = t_f."""
    p = find(DB["profils"], E.defaults()["prof_c"])
    d = {"h_u": p["h"], "b_u": p["b"], "tw_u": p["tw"], "tf_u": p["tf"]}
    d["t_u"] = d["tf_u"]
    return d


def refus_tables(u):
    """Les dimensions NÉGATIVES des éléments soudés d'une note relue (l'écran Streamlit les garde : géométrie absurde,
    calculée en silence ; la saisie du site les refuse) : la vérification est suspendue (``assemblages.saisies_refusees``).
    Les raidisseurs n'en ont pas : leur normalisation de l'écran les ramène déjà à 0."""
    out = []
    for i, e in enumerate(u.get("elems") or []):
        for k in ("h", "b", "tw", "tf", "r", "t", "d"):
            x = e.get(k)
            if isinstance(x, (int, float)) and x < 0:
                out.append((f"elems/{i}/{k}", f"Élément soudé {i + 1} : {k} = {C.nb(x)} mm — une dimension positive ou nulle "
                                              f"est attendue"))
    return out


def _variante(R):
    return MANUEL if CA.est_manuel_R(R) else A.CONFIGS["platine_beton"]


COMMUN = dict(E=E, calculer=CA.compute, N=NO, formules=lambda R, c: A.lignes_texte(R, c, FO.textes),
              piece=_piece, sections=_sections, libelles=LIBELLES, affichage=AFFICHAGE_SITE, segments=SEGMENTS,
              entiers=ENTIERS, listes=LISTES, dimensions=_dimensions(), note_information=note_information,
              visserie_defaut=E.VISSERIE_DEFAUT, exemples=exemples, etat={"grille": _grille},
              actions={"grille": _action_grille}, sur_saisie={"mode": _changer_mode}, aides=aides, refus=refus_tables,
              hypotheses=hypotheses, blocs_tete=blocs_tete, blocs=blocs, lignes_resultat=lignes_resultat,
              complement_verdict=complement_verdict, legende_33=C.esc(LEGENDE_33),
              selecteurs={"ancr": {"rendre": sel_tige, "choisir": A._choisir_liste("ancr", "boulons")},
                          "prof_c": {"rendre": A.sel_profil("prof_c", "Poteau · profilé"), "choisir": A._choisir_liste("prof_c", "profils")}})

A.enregistrer(A.Config(cle="platine_beton", nom_menu="Platine sur béton", S=YCO, T=TCO, methode=MEC.METHODE, bench=BCO, vues=_vues(SCO, True),
                       variante=_variante, parametres=parametres_complet, non_reverifies=NON_REVERIFIES,
                       normes={"Ancrages": "EN 1992-4", "Béton": "EN 1993-1-8 §6.2.5"},
                       natures=(("EC", SENS_EC),) + tuple((n, TEXTES_NATURES_CO[n]) for n in NATURES[1:]),
                       validation=lambda cfg: _validation_complet(), **COMMUN))
# la variante « platine à la main » : même saisie ; synthèse, texte, méthode, dessins, périmètre et validation du
# calcul manuel
MANUEL = A.Config(cle="platine_beton", nom_menu="Platine sur béton", S=YMA, T=TMA, methode=MEM.METHODE, bench=None, vues=_vues(SMA, False),
                  parametres=parametres_manuel, non_reverifies=NON_VERIFIES, titre_non_verifies="Non vérifiés ici",
                  normes={"Appui": "EN 1993-1-8 §6.2.5", "Trous": "EN 1993-1-8 Tab. 3.4"},
                  natures=(("EC", SENS_EC),) + tuple((n, TEXTES_NATURES_MA[n]) for n in NATURES[1:]),
                  validation=lambda cfg: _validation_manuel(), **COMMUN)

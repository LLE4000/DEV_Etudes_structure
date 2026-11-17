# -*- coding: utf-8 -*-
"""Les PROFILÉS RECONSTITUÉS SOUDÉS (PRS) en I : trois plats — semelle haute b1 × t1, âme hw × tw, semelle basse
b2 × t2 —, l'âme soudée à chaque semelle par deux cordons d'angle de gorge a. Semelles égales (bisymétrique) ou
différentes (monosymétrique, symétrique par rapport au plan de l'âme). Brief v12 § 18 ; docs/acier/BARRES_ACIER.md § 5.10.

LE CONTRAT (les blocs de la méthode ELS, de la stabilité et de l'EN 1993-1-5 le lisent) :

    section_prs(b1, t1, hw, tw, b2, t2, a=None, nuance="S355", bords=None, nom=None) -> une section au format de
    acier/barre/section.py (forme « I »), plus :
        "fabrication"      « soude_oxycoupe », « soude_larges_plats », ou None (bords des semelles non saisis)
        "prs"              {"b1", "t1", "hw", "tw", "b2", "t2", "a"} (mm ; 1 = semelle HAUTE dans la barre posée)
        "monosymetrique"   bool ; "h" = t1 + hw + t2
        "z_g"              centre de gravité depuis la face inférieure (mm)
        "z_s"              centre de cisaillement AU-DESSUS du centre de gravité (mm, signé)
        "int_z_y2z2"       ∫z·(y² + z²)dA, z vers le haut depuis le centre de gravité (mm⁵)
        "Wel_y_haut", "Wel_y_bas", "Wel_y" (le plus petit), "Wpl_y" (axe neutre plastique qui partage l'aire)
        "Iw", "It" (18.4), "If_haut" = t1·b1³/12, "If_bas" = t2·b2³/12
        "Avz" = η·hw·tw (η = 1,20, PRS), "Avy" = A − hw·tw, "eta" = 1,20 — au η du jeu de paramètres PAR DÉFAUT ;
                           l'Eurocode 3 de la barre recalcule A_v,z avec SON jeu (ec3_ame.eta_de) : un test lie les deux
    zj_psi(section, semelle_comprimee) -> {"z_j", "psi_f", …}       par TRONÇON : ils changent avec la semelle comprimée

LES PROPRIÉTÉS (brief § 18.4), sur les trois rectangles, CORDONS NON COMPTÉS (côté sûr) :
    A, centre de gravité, I_y, I_z, i_y, i_z ; W_el,y aux deux fibres extrêmes ; W_el,z = I_z/(b_max/2) ;
    W_pl,y autour de l'axe neutre PLASTIQUE (il partage l'aire en deux moitiés) ; W_pl,z = Σ t·b²/4 ;
    I_t = Σ (b·t³/3)·(1 − 0,63·t/b) sur les TROIS plats, âme comprise, chacun compté comme un rectangle séparé (b son
        grand côté, t le petit) — jamais Σ b·t³/3 seul (+4,2 % sur PRS-A, côté dangereux) ;
    I_w = h_s²·I₁·I₂/(I₁ + I₂), h_s = h − t1/2 − t2/2 (parois minces, semelles seules) ; centre de cisaillement à
        h_s·I₂/(I₁ + I₂) sous le plan moyen de la semelle haute ;
    ψ_f = (I_fc − I_ft)/(I_fc + I_ft) ; z_j = z_s − 0,5·∫z·(y² + z²)dA/I_y (annexe E ANB, 2(2)), z positif du centre de
        gravité vers la semelle COMPRIMÉE (2(7)) ; (E.4) et (E.5) : pour information.
    Pour un rectangle de largeur b entre z1 et z2 : ∫z·y²dA = b³·(z2² − z1²)/24 et ∫z³dA = b·(z2⁴ − z1⁴)/4.

LA SAISIE (brief § 18.2) — contrôles BLOQUANTS (Refus, avec le message) : dimensions positives ; t_w < b1 et t_w < b2 ;
c de l'âme (h_w − 2·s) et des semelles ((b − t_w)/2 − s) positifs, s = a·√2 (côté d'un cordon d'angle à côtés égaux).
Gorge non saisie : s = 0 (c plus grand, côté sûr), avec un message. Bords non saisis : « fabrication » None — le verdict
de la méthode ELS est bloqué (elle en tire ses courbes) ; l'Eurocode 3 ne les emploie pas.

LA BIBLIOTHÈQUE (brief § 18.2) : le format des « profilés personnalisés » du Catalogue (api_web/catalogue.py,
rubrique 6 ; acier/profiles/consultation.py) — {"famille": "PRS", "dims": {"h", "b", "tw", "tf"}} — étendu, pour une
section monosymétrique, de la semelle BASSE ("b2", "tf2") et de la gorge ("a"), plus la nuance et les bords : seules les
dimensions et la nuance sont stockées, les propriétés sont TOUJOURS recalculées ici. Le nom porte « PRS ».

LA SECTION EFFICACE (EN 1993-1-5 § 4.4, brief § 7.4 et 18.5 : b̄ = b_w, largeur libre entre les soudures) d'un PRS de
classe 4 : ``aire_efficace`` (compression seule, avec le décalage e_N du centre de gravité efficace) et
``module_efficace`` (flexion seule : la semelle comprimée d'abord, puis l'âme avec le ψ de la section à semelle
réduite, § 4.4(3)).

Unités : N, mm. Bibliothèque standard seule.
"""
import math
from typing import Dict, List, Optional, Sequence, Tuple

from . import nuances as NU
from . import parametres as PA
from .resultat import Refus, fr

FAMILLE = "PRS"
DIMS = ("b1", "t1", "hw", "tw", "b2", "t2")
BORDS = {"oxycoupes": "soude_oxycoupe", "larges_plats": "soude_larges_plats"}       # brief § 18.2 (doc p. 4)
COEF_IT = 0.63                   # I_t d'un rectangle : (b·t³/3)·(1 − 0,63·t/b) (brief § 18.4)
RACINE2 = math.sqrt(2.0)         # s = a·√2 : cordon d'angle à côtés égaux (45°)
E4, E5 = 0.8, 1.0                # (E.4) z_j ≈ 0,8·ψ_f·h_s/2 si ψ_f ≥ 0 ; (E.5) z_j ≈ ψ_f·h_s/2 si ψ_f < 0 (pour information)
MASSE_VOLUMIQUE = 7850.0         # kg/m³ (la masse affichée ; le poids propre de la barre est celui de statique.py)
HAUT, BAS = "haut", "bas"
DECIMALES_MM = 3                 # une dimension de plat s'écrit avec ses décimales, au millième au plus
METHODE_IT_CATALOGUE = "Σ (b·t³/3)·(1 − 0,63·t/b) sur les trois plats, côté sûr"      # la légende « I_t : … » du Catalogue
MSG_GORGE = "Gorge a non saisie : s = 0 (largeurs c plus grandes, côté sûr)."
MSG_BORDS = "Bords des semelles non saisis (oxycoupés ou larges plats) : verdict de la méthode ELS bloqué."
# l'ε d'un plat dans les blocs de formules : « ε_f1 » (semelle haute), « ε_f2 » (semelle basse), « ε_w » (âme)
ROLES = {"semelle_haut": ("semelle haute", "f1"), "semelle_bas": ("semelle basse", "f2"), "ame": ("âme", "w")}


def mm(x: float) -> str:
    """Une dimension de plat dans une phrase : « 30 », « 30,5 », « 30,25 » — les décimales de la saisie (jusqu'à
    DECIMALES_MM), virgule, sans zéro inutile : le nom du PRS et ses phrases donnent le même plat."""
    n = next((k for k in range(DECIMALES_MM + 1) if abs(x - round(x, k)) < 1e-9), DECIMALES_MM)
    return fr(x, n)


def _dimension(v, nom: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or not float(v) > 0:
        raise Refus(f"PRS : {nom} doit être une dimension positive (mm), reçu {v!r}")
    return float(v)


def _rectangles(b1: float, t1: float, hw: float, tw: float, b2: float, t2: float) -> List[Tuple[float, float, float]]:
    """Les trois plats [(largeur, z bas, z haut)], z depuis la face inférieure : semelle basse, âme, semelle haute."""
    return [(b2, 0.0, t2), (tw, t2, t2 + hw), (b1, t2 + hw, t2 + hw + t1)]


def axe_plastique(rects: Sequence[Tuple[float, float, float]], A: float) -> float:
    """La cote (depuis la face inférieure) de l'axe neutre plastique : il partage l'aire en deux moitiés."""
    moitie, cumul = A / 2.0, 0.0
    for b, z1, z2 in rects:
        a = b * (z2 - z1)
        if cumul + a >= moitie:
            return z1 + (moitie - cumul) / b
        cumul += a
    return rects[-1][2]


def module_plastique(rects: Sequence[Tuple[float, float, float]], zp: float) -> float:
    """W_pl = Σ ∫|z − z_p|dA autour de la cote z_p."""
    w = 0.0
    for b, z1, z2 in rects:
        for u, v in ((z1, min(z2, zp)), (max(z1, zp), z2)):
            if v > u:
                w += b * (v - u) * abs((u + v) / 2.0 - zp)
    return w


def torsion_rectangle(b: float, t: float) -> float:
    """I_t d'un plat compté seul : (b·t³/3)·(1 − 0,63·t/b), b le grand côté (brief § 18.4)."""
    b, t = max(b, t), min(b, t)
    return b * t ** 3 / 3.0 * (1.0 - COEF_IT * t / b)


def nom_de(b1: float, t1: float, hw: float, tw: float, b2: float, t2: float) -> str:
    """« PRS 600x300x15x30 » (h × b × t_w × t_f, le nom des profilés personnalisés du Catalogue) ; monosymétrique :
    « PRS 600x300x15x30x200x30 » (la semelle basse ajoutée)."""
    h = t1 + hw + t2
    base = f"{FAMILLE} " + "x".join(f"{float(x):g}" for x in (h, b1, tw, t1))
    return base if (b1, t1) == (b2, t2) else base + "x" + "x".join(f"{float(x):g}" for x in (b2, t2))


def section_prs(b1, t1, hw, tw, b2, t2, a=None, nuance: str = "S355", bords: Optional[str] = None, nom: Optional[str] = None) -> Dict:
    """La section d'un PRS en I, au format de acier/barre/section.py, ses propriétés CALCULÉES (brief § 18.4) ;
    Refus pour une saisie impossible (brief § 18.2), avec le message."""
    b1, t1, hw, tw, b2, t2 = (_dimension(v, n) for v, n in zip((b1, t1, hw, tw, b2, t2), DIMS))
    messages = []
    if a is None or (isinstance(a, (int, float)) and not isinstance(a, bool) and float(a) == 0.0):
        a, s = None, 0.0
        messages.append(MSG_GORGE)
    else:
        a = _dimension(a, "la gorge a")
        s = a * RACINE2
    if nuance not in NU.NUANCES:              # (lot C, C1b, reprise) la liste du module Barres : S420 compris
        raise Refus(f"PRS : nuance inconnue {nuance!r} ({', '.join(NU.NUANCES[:-1])} ou {NU.NUANCES[-1]})")
    bords = {v: k for k, v in BORDS.items()}.get(bords, bords)    # (la fabrication du contrat, « soude_oxycoupe », admise aussi)
    if bords is not None and bords not in BORDS:
        raise Refus(f"PRS : bords des semelles « oxycoupes » ou « larges_plats » — reçu {bords!r}")
    if not (tw < b1 and tw < b2):
        raise Refus("PRS : l'âme doit être moins épaisse que les semelles ne sont larges (t_w < b1 et t_w < b2)")
    cw = hw - 2.0 * s
    cf1, cf2 = (b1 - tw) / 2.0 - s, (b2 - tw) / 2.0 - s
    if not (cw > 0 and cf1 > 0 and cf2 > 0):
        raise Refus("PRS : les cordons couvrent toute la paroi (c ≤ 0) — h_w − 2·s et (b − t_w)/2 − s doivent être positifs")
    if bords is None:
        messages.append(MSG_BORDS)
    h = t1 + hw + t2
    rects = _rectangles(b1, t1, hw, tw, b2, t2)
    A = sum(b * (z2 - z1) for b, z1, z2 in rects)
    zg = sum(b * (z2 - z1) * (z1 + z2) / 2.0 for b, z1, z2 in rects) / A
    Iy = sum(b * (z2 - z1) ** 3 / 12.0 + b * (z2 - z1) * ((z1 + z2) / 2.0 - zg) ** 2 for b, z1, z2 in rects)
    I1, I2 = t1 * b1 ** 3 / 12.0, t2 * b2 ** 3 / 12.0
    Iz = I1 + hw * tw ** 3 / 12.0 + I2
    zp = axe_plastique(rects, A)
    hs = h - t1 / 2.0 - t2 / 2.0
    z_sc = h - t1 / 2.0 - hs * I2 / (I1 + I2)                   # le centre de cisaillement, depuis la face inférieure
    integrale = sum(b ** 3 * ((z2 - zg) ** 2 - (z1 - zg) ** 2) / 24.0 + b * ((z2 - zg) ** 4 - (z1 - zg) ** 4) / 4.0
                    for b, z1, z2 in rects)
    eta = PA.valeur("eta_soude")                                 # (le défaut : voir « Avz » dans le contrat)
    mono = (b1, t1) != (b2, t2)
    sec = {"nom": nom if (nom and FAMILLE in nom) else (f"{FAMILLE} {nom}" if nom else nom_de(b1, t1, hw, tw, b2, t2)),
           "famille": FAMILLE, "forme": "I", "fabrication": BORDS.get(bords), "axes": ("y-y", "z-z"), "nuance": nuance,
           "bords": bords,
           # les clés communes d'un I : b = la PLUS PETITE largeur (h/b du déversement, brief § 18.5), t_f = la PLUS GRANDE
           # épaisseur (tableau 6.2, choix prudent déclaré ; et la f_y de calcul de la section : la plus petite des plats)
           "h": h, "b": min(b1, b2), "tw": tw, "tf": max(t1, t2), "r": 0.0, "t": None, "d": None,
           "prs": {"b1": b1, "t1": t1, "hw": hw, "tw": tw, "b2": b2, "t2": t2, "a": a}, "s": s, "monosymetrique": mono,
           "A": A, "Iy": Iy, "Iz": Iz, "iy": math.sqrt(Iy / A), "iz": math.sqrt(Iz / A),
           "z_g": zg, "z_pl": zp, "z_s": z_sc - zg, "int_z_y2z2": integrale,
           "Wel_y_haut": Iy / (h - zg), "Wel_y_bas": Iy / zg, "Wel_y": min(Iy / (h - zg), Iy / zg), "Wel_z": Iz / (max(b1, b2) / 2.0),
           "Wpl_y": module_plastique(rects, zp), "Wpl_z": (t1 * b1 ** 2 + hw * tw ** 2 + t2 * b2 ** 2) / 4.0,
           "It": torsion_rectangle(b1, t1) + torsion_rectangle(hw, tw) + torsion_rectangle(b2, t2),
           "Iw": hs ** 2 * I1 * I2 / (I1 + I2), "h_s": hs, "If_haut": I1, "If_bas": I2,
           "eta": eta, "Avz": eta * hw * tw, "Avy": A - hw * tw, "masse": A * 1e-6 * MASSE_VOLUMIQUE,
           # (méthode ELS : l'âme et les semelles, comme les classeurs — τ = V_z/A_âme, V_y/A_ailes)
           "A_ame": hw * tw, "A_ailes": b1 * t1 + b2 * t2,
           "cw_tw": cw / tw, "cf_tf": max(cf1 / t1, cf2 / t2),
           "messages": messages}
    return sec


def est_prs(s: Dict) -> bool:
    return bool(s.get("prs"))


def monosymetrique(s: Dict) -> bool:
    return bool(s.get("prs")) and bool(s.get("monosymetrique"))


def zj_psi(s: Dict, semelle_comprimee: str = HAUT) -> Dict:
    """z_j et ψ_f du TRONÇON dont la semelle ``semelle_comprimee`` (« haut » ou « bas ») est comprimée (annexe E ANB,
    2(2) et 2(7)) : {"z_j" (mm, positif du centre de gravité vers la semelle comprimée), "psi_f", "I_fc", "I_ft",
    "z_j_approche" ((E.4) ou (E.5), pour information), "formule_approche", "semelle_comprimee"}. Section bisymétrique :
    z_j = 0, ψ_f = 0."""
    if semelle_comprimee not in (HAUT, BAS):
        raise ValueError(f"semelle comprimée : « haut » ou « bas » — reçu {semelle_comprimee!r}")
    if not est_prs(s):
        return {"z_j": 0.0, "psi_f": 0.0, "I_fc": None, "I_ft": None, "z_j_approche": 0.0, "formule_approche": None,
                "semelle_comprimee": semelle_comprimee}
    haut = semelle_comprimee == HAUT
    Ifc, Ift = (s["If_haut"], s["If_bas"]) if haut else (s["If_bas"], s["If_haut"])
    psi = (Ifc - Ift) / (Ifc + Ift)
    zj_haut = s["z_s"] - 0.5 * s["int_z_y2z2"] / s["Iy"]
    zj = zj_haut if haut else -zj_haut
    approche = (E4 if psi >= 0 else E5) * psi * s["h_s"] / 2.0
    return {"z_j": zj, "psi_f": psi, "I_fc": Ifc, "I_ft": Ift, "z_j_approche": approche,
            "formule_approche": "(E.4)" if psi >= 0 else "(E.5)", "semelle_comprimee": semelle_comprimee}


def iw_annexe_e(s: Dict, semelle_comprimee: str = HAUT) -> float:
    """I_w par (E.3) de l'annexe E ANB, (1 − ψ_f²)·I_z·(h_s/2)², POUR INFORMATION : l'âme compte dans I_z (+0,11 % sur
    PRS-A, +0,17 % sur PRS-B) ; le module emploie la formule des semelles seules (brief § 18.4)."""
    return (1.0 - zj_psi(s, semelle_comprimee)["psi_f"] ** 2) * s["Iz"] * (s["h_s"] / 2.0) ** 2


def zj_haut(s: Dict) -> float:
    """z_j quand la semelle HAUTE est comprimée (le repère fixe du solveur de M_cr : z vers le haut, M_y > 0 quand la
    semelle haute est comprimée — le même z_j sert pour les deux signes du moment) ; 0 hors d'un PRS monosymétrique."""
    return zj_psi(s, HAUT)["z_j"] if monosymetrique(s) else 0.0


def cotes(s: Dict) -> Dict[str, float]:
    """Les cotes, AU-DESSUS du centre de cisaillement (mm), où se repèrent les charges et les maintiens (brief § 7.7) :
    {"face_haut", "face_bas", "semelle_haut", "semelle_bas" (plans moyens des semelles), "z_cisaillement" (depuis la
    face inférieure)}."""
    p = s["prs"]
    zc = s["z_g"] + s["z_s"]
    return {"face_haut": s["h"] - zc, "face_bas": -zc, "semelle_haut": s["h"] - p["t1"] / 2.0 - zc, "semelle_bas": p["t2"] / 2.0 - zc,
            "z_cisaillement": zc}


# ------------------------------------------------------------------ les parois (classification, section efficace)
def parois(s: Dict) -> List[Dict]:
    """Les trois parois d'un PRS, c depuis le PIED DES CORDONS (tableau 5.2, sections soudées) : [{"role", "nom",
    "type", "c", "t", "c_t", "n" (nombre de consoles ou d'âmes), "z" (cote du plan moyen, depuis la face inférieure)}]."""
    p, sv = s["prs"], s["s"]
    tw = p["tw"]
    cw = p["hw"] - 2.0 * sv
    cf1, cf2 = (p["b1"] - tw) / 2.0 - sv, (p["b2"] - tw) / 2.0 - sv
    return [{"role": "semelle_haut", "nom": "semelle haute", "type": "console", "t": p["t1"], "c": cf1, "c_t": cf1 / p["t1"], "n": 2,
             "z": s["h"] - p["t1"] / 2.0, "b": p["b1"]},
            {"role": "ame", "nom": "âme", "type": "interne", "t": tw, "c": cw, "c_t": cw / tw, "n": 1, "z": p["t2"] + p["hw"] / 2.0,
             "z_bas": p["t2"] + sv, "z_haut": p["t2"] + p["hw"] - sv},
            {"role": "semelle_bas", "nom": "semelle basse", "type": "console", "t": p["t2"], "c": cf2, "c_t": cf2 / p["t2"], "n": 2,
             "z": p["t2"] / 2.0, "b": p["b2"]}]


def aires_semelles(s: Dict, semelle_comprimee: str) -> Tuple[float, float]:
    """(A_fc, A_ft) : l'aire de la semelle comprimée et de la semelle tendue, chacune avec la bande d'âme entre elle et
    le pied de ses cordons (s·t_w) — ce qui est hors de la largeur c de l'âme."""
    p, sv = s["prs"], s["s"]
    haut = p["b1"] * p["t1"] + sv * p["tw"]
    bas = p["b2"] * p["t2"] + sv * p["tw"]
    return (haut, bas) if semelle_comprimee == HAUT else (bas, haut)


def alpha_ame(s: Dict, N: float, fy: float, semelle_comprimee: str) -> Dict:
    """α de l'âme d'un PRS fléchi (et comprimé, N > 0), par l'axe neutre PLASTIQUE (tableau 5.2) : la part comprimée de
    sa largeur c. Les deux semelles plastifiées (l'une comprimée, l'autre tendue), l'équilibre donne
        α = 0,5 + (N/f_y − A_fc + A_ft) / (2·c·t_w)        (bisymétrique : 0,5 + N/(2·c·t_w·f_y))
    {"alpha" (borné à [0 ; 1]), "brut", "A_fc", "A_ft", "c", "tw"}."""
    p = s["prs"]
    c = p["hw"] - 2.0 * s["s"]
    Afc, Aft = aires_semelles(s, semelle_comprimee)
    brut = 0.5 + (N / fy - Afc + Aft) / (2.0 * c * p["tw"])
    return {"alpha": min(max(brut, 0.0), 1.0), "brut": brut, "A_fc": Afc, "A_ft": Aft, "c": c, "tw": p["tw"]}


def psi_ame(s: Dict, N: float, fy: float, semelle_comprimee: str) -> Dict:
    """ψ de l'âme d'un PRS fléchi (et comprimé), ÉLASTIQUE : le bord de sa largeur c du côté de la semelle comprimée à
    f_y, l'effort normal uniforme — σ(z) = N/A + k·(z − z_G) :
        ψ = (N/A − (f_y − N/A)·v_t/v_c) / f_y          (bisymétrique : 2·N/(A·f_y) − 1)
    v_c, v_t : distances du centre de gravité au bord comprimé et à l'autre bord de c. {"psi" (≤ 1), "v_c", "v_t"} ;
    v_c ≤ 0 : le moment ne comprime pas l'âme (None)."""
    w = next(x for x in parois(s) if x["role"] == "ame")
    zg = s["z_g"]
    if semelle_comprimee == HAUT:
        vc, vt = w["z_haut"] - zg, zg - w["z_bas"]
    else:
        vc, vt = zg - w["z_bas"], w["z_haut"] - zg
    if not vc > 0:
        return {"psi": None, "v_c": vc, "v_t": vt}
    sN = N / s["A"]
    psi = (sN - (fy - sN) * vt / vc) / fy
    return {"psi": min(psi, 1.0), "v_c": vc, "v_t": vt}


# ------------------------------------------------------------------ EN 1993-1-5 § 4.4 : la section efficace d'un PRS
def _assemble(pieces: Sequence[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    """(A, z_G, I) d'un assemblage [(aire, z du centre, inertie propre)]."""
    A = sum(a for a, _, _ in pieces)
    zg = sum(a * z for a, z, _ in pieces) / A
    return A, zg, sum(i0 + a * (z - zg) ** 2 for a, z, i0 in pieces)


def _pieces(s: Dict) -> List[Tuple[float, float, float]]:
    return [(b * (z2 - z1), (z1 + z2) / 2.0, b * (z2 - z1) ** 3 / 12.0) for b, z1, z2 in _rectangles(*(s["prs"][k] for k in DIMS))]


def aire_efficace(s: Dict, eps, rho_interne, rho_console, lambda_p, k_console: float) -> Dict:
    """A_eff d'un PRS sous compression SEULE (ψ = 1) — ``eps`` : {rôle: ε} ; les fonctions ρ et λ̄_p de
    acier/barre/ec3_classe.py (une seule écriture des tableaux 4.1 et 4.2). Les bouts des quatre consoles et le milieu
    de la largeur c de l'âme sont retirés ; e_N : le décalage (mm, vers le haut) du centre de gravité efficace.
    {"A_eff", "reduite", "e_N", "parois": [{…, "epsilon", "k_sigma", "lambda_p", "rho", "retrait"}]}."""
    out, trous = [], []
    for p in parois(s):
        p = dict(p)
        e = eps[p["role"]] if isinstance(eps, dict) else eps
        ks = 4.0 if p["type"] == "interne" else k_console
        lp = lambda_p(p["c_t"], e, ks)
        rho = rho_interne(lp, 1.0) if p["type"] == "interne" else rho_console(lp)
        retrait = p["n"] * (1.0 - rho) * p["c"] * p["t"]
        p.update({"epsilon": e, "psi": 1.0, "k_sigma": ks, "lambda_p": lp, "rho": rho, "retrait": retrait})
        if retrait > 0:
            # (console : le bout des deux consoles, au plan moyen de la semelle ; âme : le milieu de c, à mi-hauteur de c)
            z = p["z"] if p["type"] == "console" else (p["z_bas"] + p["z_haut"]) / 2.0
            trous.append((retrait, z))
        out.append(p)
    A_eff = s["A"] - sum(a for a, _ in trous)
    # (semelles égales : les trous sont symétriques, e_N = 0 exactement — pas le bruit d'arrondi d'un calcul à 10⁻¹³ mm)
    eN = (s["A"] * s["z_g"] - sum(a * z for a, z in trous)) / A_eff - s["z_g"] if trous and s.get("monosymetrique") else 0.0
    return {"A_eff": A_eff, "reduite": A_eff < s["A"], "e_N": eN, "parois": out}


def module_efficace(s: Dict, eps, axe: str, semelle_comprimee: str, Fns: Dict) -> Dict:
    """W_eff d'un PRS sous flexion SEULE autour de ``axe`` ; ``semelle_comprimee`` : celle que M_y comprime ; ``Fns`` :
    {"rho_interne", "rho_console", "lambda_p", "k_sigma_interne", "k_sigma_console", "k_console", "be1"} (les fonctions de
    acier/barre/ec3_classe.py). Axe fort : la semelle comprimée réduite (console, ψ = 1), puis l'âme avec le ψ que donne
    l'axe neutre de cette section (§ 4.4(3)) ; W_eff = I_eff / la plus grande distance de l'axe neutre efficace à une
    fibre extrême (≤ W_el,y). Axe faible : la console comprimée de chaque semelle (ψ de la console). Même forme que
    ec3_classe.module_efficace : {"W_eff", "I_eff", "e", "v", "reduit", "membrure", "ame"}."""
    w = {p["role"]: p for p in parois(s)}
    pr = s["prs"]
    if axe == "z":
        vmax = max(pr["b1"], pr["b2"]) / 2.0
        trous_y = []                                     # [(aire, y du centre (vers le côté comprimé), inertie propre autour de z)]
        membrure = None
        for role in ("semelle_haut", "semelle_bas"):
            p = w[role]
            v = p["b"] / 2.0
            psi_m = (v - p["c"]) / v
            ks = Fns["k_sigma_console"](psi_m)
            e = eps[role] if isinstance(eps, dict) else eps
            lp = Fns["lambda_p"](p["c_t"], e, ks)
            rho = Fns["rho_console"](lp)
            long = (1.0 - rho) * p["c"]
            if long > 0:
                trous_y.append((long * p["t"], v - long / 2.0, p["t"] * long ** 3 / 12.0))
            m = dict(p, epsilon=e, psi=psi_m, k_sigma=ks, lambda_p=lp, rho=rho)
            if membrure is None or rho < membrure["rho"]:
                membrure = m
        Wel = s["Wel_z"]
        if not trous_y:
            return {"W_eff": Wel, "I_eff": s["Iz"], "e": 0.0, "v": vmax, "reduit": False, "membrure": membrure, "ame": None}
        A_net = s["A"] - sum(a for a, _, _ in trous_y)
        yG = -sum(a * y for a, y, _ in trous_y) / A_net
        I_eff = s["Iz"] - sum(i0 + a * y * y for a, y, i0 in trous_y) - A_net * yG * yG
        return {"W_eff": min(I_eff / (vmax - yG), Wel), "I_eff": I_eff, "e": -yG, "v": vmax, "reduit": True, "membrure": membrure, "ame": None}
    haut = semelle_comprimee == HAUT
    f = w["semelle_haut" if haut else "semelle_bas"]
    e_f = eps[f["role"]] if isinstance(eps, dict) else eps
    ks = Fns["k_console"]
    lp = Fns["lambda_p"](f["c_t"], e_f, ks)
    rho = Fns["rho_console"](lp)
    pieces = _pieces(s)
    retire = []                                          # [(aire, z, inertie propre)] retirées de la section brute
    long_f = (1.0 - rho) * f["c"]
    if long_f > 0:
        retire.append((f["n"] * long_f * f["t"], f["z"], f["n"] * long_f * f["t"] ** 3 / 12.0))
    membrure = dict(f, epsilon=e_f, psi=1.0, k_sigma=ks, lambda_p=lp, rho=rho)

    def section(trous):
        return _assemble(pieces + [(-a, z, -i0) for a, z, i0 in trous])

    _, z1, _ = section(retire)                           # l'axe neutre de la section à semelle comprimée réduite (âme brute)
    a = w["ame"]
    zc_bord, zt_bord = (a["z_haut"], a["z_bas"]) if haut else (a["z_bas"], a["z_haut"])
    d_c, d_t = abs(zc_bord - z1), (zt_bord - z1) * (1.0 if haut else -1.0)       # d_t < 0 : l'autre bord est tendu
    sens = -1.0 if haut else 1.0                         # du bord comprimé vers l'intérieur de l'âme
    e_a = eps["ame"] if isinstance(eps, dict) else eps
    if (zc_bord - z1) * (1.0 if haut else -1.0) <= 0:    # le moment ne comprime pas l'âme : elle reste entière
        ame_out = dict(a, epsilon=e_a, psi=None, k_sigma=None, lambda_p=0.0, rho=1.0, b_c=0.0, b_eff=0.0)
    else:
        psi_w = d_t / d_c
        ks_w = Fns["k_sigma_interne"](psi_w)
        lp_w = Fns["lambda_p"](a["c_t"], e_a, ks_w)
        rho_w = Fns["rho_interne"](lp_w, psi_w)
        if psi_w < 0:
            bc = d_c
            beff = rho_w * bc
            be1 = Fns["be1"] * beff
        else:
            bc = a["c"]
            beff = rho_w * bc
            be1 = 2.0 / (5.0 - psi_w) * beff
        long_w = bc - beff
        if long_w > 0:
            debut = zc_bord + sens * be1                 # le trou commence à b_e1 du bord comprimé, vers l'intérieur
            retire.append((long_w * a["t"], debut + sens * long_w / 2.0, a["t"] * long_w ** 3 / 12.0))
        ame_out = dict(a, epsilon=e_a, psi=psi_w, k_sigma=ks_w, lambda_p=lp_w, rho=rho_w, b_c=bc, b_eff=beff)
    Wel = s["Wel_y"]
    if not retire:
        v = max(s["h"] - s["z_g"], s["z_g"])
        return {"W_eff": Wel, "I_eff": s["Iy"], "e": 0.0, "v": v, "reduit": False, "membrure": membrure, "ame": ame_out}
    _, zG, I_eff = section(retire)
    # la fibre extrême la plus éloignée de l'axe neutre efficace : « v » (brute, depuis z_G) + « e » (le décalage)
    haut_d, bas_d = s["h"] - zG, zG
    if haut_d >= bas_d:
        v, e = s["h"] - s["z_g"], s["z_g"] - zG
    else:
        v, e = s["z_g"], zG - s["z_g"]
    return {"W_eff": min(I_eff / (v + e), Wel), "I_eff": I_eff, "e": e, "v": v, "reduit": True, "membrure": membrure, "ame": ame_out}


def module_plastique_ame_reduite(s: Dict, rho: float) -> float:
    """W_pl,y d'un PRS dont l'âme (h_w·t_w, l'aire de cisaillement entre les semelles) a la limite d'élasticité réduite
    (1 − ρ)·f_y (EN 1993-1-1 (6.29), § 6.2.8(3)) : l'âme comptée d'épaisseur (1 − ρ)·t_w, l'axe neutre plastique de
    cette section recalculé — la forme exacte de (6.30) quand les semelles sont égales."""
    p = s["prs"]
    rects = _rectangles(p["b1"], p["t1"], p["hw"], (1.0 - rho) * p["tw"], p["b2"], p["t2"])
    A = sum(b * (z2 - z1) for b, z1, z2 in rects)
    return module_plastique(rects, axe_plastique(rects, A))


# ------------------------------------------------------------------ la bibliothèque des profilés personnalisés
def vers_bibliotheque(s: Dict) -> Dict:
    """L'enregistrement d'un PRS dans la bibliothèque des profilés personnalisés — le format du Catalogue
    ({"famille": "PRS", "dims": {"h", "b", "tw", "tf"}}), la semelle basse (« b2 », « tf2 ») et la gorge (« a ») en
    plus quand elles comptent ; la nuance et les bords. AUCUNE propriété : elles sont recalculées au chargement."""
    p = s["prs"]
    dims = {"h": s["h"], "b": p["b1"], "tw": p["tw"], "tf": p["t1"]}
    if s.get("monosymetrique"):
        dims.update({"b2": p["b2"], "tf2": p["t2"]})
    if p["a"] is not None:
        dims["a"] = p["a"]
    out = {"famille": FAMILLE, "dims": dims, "nom": s["nom"], "nuance": s["nuance"]}
    if s.get("bords"):
        out["bords"] = s["bords"]
    return out


def de_bibliotheque(entree: Dict, nuance: Optional[str] = None) -> Dict:
    """Le PRS d'un enregistrement de la bibliothèque (``vers_bibliotheque``, ou un PRS du Catalogue : semelles égales,
    sans gorge) : propriétés RECALCULÉES. ``nuance`` : celle de la barre quand l'enregistrement n'en porte pas."""
    if not isinstance(entree, dict) or entree.get("famille") != FAMILLE or not isinstance(entree.get("dims"), dict):
        raise Refus(f"profilé personnalisé illisible : un PRS est attendu ({entree!r})"[:160])
    d = entree["dims"]
    h, b1, tw, t1 = (_dimension(d.get(k), k) for k in ("h", "b", "tw", "tf"))
    b2 = _dimension(d.get("b2", b1), "b2")
    t2 = _dimension(d.get("tf2", t1), "tf2")
    hw = h - t1 - t2
    if not hw > 0:
        raise Refus("PRS : h doit dépasser l'épaisseur des deux semelles")
    return section_prs(b1, t1, hw, tw, b2, t2, a=d.get("a"), nuance=entree.get("nuance") or nuance or "S355",
                       bords=entree.get("bords"), nom=entree.get("nom"))


def proprietes_catalogue(dims: Dict) -> Dict:
    """Ce que le Catalogue (api_web/catalogue.py) prend ICI pour son PRS à semelles égales {"h", "b", "tw", "tf"} —
    une seule source (R2) — en unités du catalogue : I_t (18.4, cm⁴), I_w (cm⁶), A_v,z = η·h_w·t_w et
    A_v,y = A − h_w·t_w (cm²), et la méthode de I_t. Une âme au moins aussi épaisse que les semelles sont larges
    (section_prs la refuse ; le Catalogue écrit que ce n'est plus un I) : I_t seul, par la même formule."""
    h, b, tw, tf = (float(dims[k]) for k in ("h", "b", "tw", "tf"))
    hw = h - 2.0 * tf
    out = {"It": (2.0 * torsion_rectangle(b, tf) + torsion_rectangle(hw, tw)) / 1e4, "methode_It": METHODE_IT_CATALOGUE}
    try:
        s = section_prs(b, tf, hw, tw, b, tf, bords="oxycoupes")
    except Refus:
        return out
    out.update({"It": s["It"] / 1e4, "Iw": s["Iw"] / 1e6, "Avz": s["Avz"] / 1e2, "Avy": s["Avy"] / 1e2})
    return out


def chercher(bibliotheque: Sequence[Dict], nom: str, nuance: Optional[str] = None) -> Optional[Dict]:
    """Le PRS de nom ``nom`` dans la bibliothèque (une liste d'enregistrements) ; None s'il n'y est pas."""
    for e in bibliotheque or ():
        if isinstance(e, dict) and e.get("famille") == FAMILLE:
            s = de_bibliotheque(e, nuance)
            if s["nom"] == nom:
                return s
    return None

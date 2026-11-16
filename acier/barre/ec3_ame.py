# -*- coding: utf-8 -*-
"""L'ÂME d'une barre selon l'EN 1993-1-5:2006 — ce que l'EN 1993-1-1 renvoie à la partie 1-5 (docs/acier/BARRES_ACIER.md
§ 5.9 ; brief § 8) :

    § 5    le voilement par CISAILLEMENT d'une âme élancée, h_w/t_w > 72·ε/η :          V_Ed ≤ V_b,Rd
               λ̄_w = h_w/(86,4·t·ε)       χ_w = η si λ̄_w < 0,83/η, sinon 0,83/λ̄_w      V_bw,Rd = χ_w·f_yw·h_w·t/(√3·γ_M1)
               η = 1,00 pour un I ou H laminé et un tube, 1,20 pour une section soudée (``eta_de`` ; NBN EN 1993-1-5
               ANB:2011, 5.1(2), note complémentaire) ; ε : celui de l'ÂME (sa f_y selon t_w — brief § 7.1)
    § 7.1  … et la flexion, quand η̄₃ = V_Ed/V_bw,Rd > 0,5 :          η̄₁ + (1 − M_f,Rd/M_pl,Rd)·(2·η̄₃ − 1)² ≤ 1
    § 6    l'âme sous une CHARGE CONCENTRÉE ou une réaction d'appui :                   F_Ed ≤ F_Rd
               types : (a) sur une semelle, reprise par le cisaillement de l'âme      k_F = 6 + 2·(h_w/a)²
                       (b) transmise d'une semelle à l'autre                           k_F = 3,5 + 2·(h_w/a)²
                       (c) près d'une extrémité non raidie, à la distance c            k_F = 2 + 6·(s_s + c)/h_w ≤ 6
               F_cr = 0,9·k_F·E·t_w³/h_w      m₁ = f_yf·b_f/(f_yw·t_w)
               m₂ = 0,02·(h_w/t_f)², branche retenue seulement si λ̄_F > 0,5 ; m₂ = 0, seulement si λ̄_F ≤ 0,5 : les DEUX
                    branches sont calculées ; quand les deux sont cohérentes, la plus petite F_Rd est retenue, et dite
                    (règle déclarée du modèle, brief § 8.2)
               (a), (b) : l_y = s_s + 2·t_f·(1 + √(m₁ + m₂))
               (c) : l_e = k_F·E·t_w²/(2·f_yw·h_w) ≤ s_s + c ; l_y = le plus petit de l'expression ci-dessus (6.10), de
                     l_e + t_f·√(m₁/2 + (l_e/t_f)² + m₂) et de l_e + t_f·√(m₁ + m₂) ; CONTACT d'extrémité « inconnu »
                     (DÉCISION BUREAU n° 2, défaut) : s_s = 0 dans (6.10) seulement — k_F et le plafond de l_e gardent
                     s_s (commentaire JRC EUR 22898, p. 79) ; la variante avec contact est donnée pour information
               λ̄_F = √(l_y·t_w·f_yw/F_cr) — f_yw de l'âme selon t_w      χ_F = 0,5/λ̄_F ≤ 1
               F_Rd = f_y·χ_F·l_y·t_w/γ_M1 — f_y de calcul de la section
    § 7.2  … et la flexion :                                                          η₂ + 0,8·η₁ ≤ 1,4
    § 8    le voilement de l'âme induit par la semelle comprimée :      h_w/t_w ≤ k·(E/f_yf)·√(A_w/A_fc)

Ce fichier CALCULE et NOMME, journal en main. Il ne décide pas OÙ vérifier (chaque appui, chaque charge ponctuelle),
ni avec quelle combinaison, ni s'il faut un raidisseur : c'est acier/barre/moteur.py ; le raidisseur lui-même est
acier/barre/raidisseurs.py (§ 9).

DOMAINE. Charges concentrées : les I et H laminés (h_w = h − 2·t_f) et les PRS en I (h_w entre semelles, b_f et t_f
de la semelle chargée ; contrat de acier/barre/prs.py). Cisaillement : l'âme d'un I, les deux âmes d'un tube carré ou
rectangulaire. La résistance d'un PROFIL CREUX sous une charge concentrée relève de
l'EN 1993-1-8 : elle n'est pas vérifiée ici, et le moteur le dit.

CE QUI EST PRIS DU CÔTÉ SÛR, et dit dans les hypothèses :
- aucun raidisseur transversal intermédiaire : a → ∞ dans k_F (6 ; 3,5) ; la contribution des semelles V_bf,Rd est
  négligée (elle tend vers zéro quand a grandit) ;
- montant d'extrémité NON rigide : χ_w = 0,83/λ̄_w au-delà de 0,83/η (tableau 5.1) ;
- η₁ ÉLASTIQUE dans (7.2) : N_Ed/(f_y·A/γ_M0) + M_Ed/(f_y·W_el/γ_M0), A_eff et W_eff en classe 4 (brief § 8.2) ;
- la longueur d'appui rigide s_s est bornée à h_w (§ 6.3(1)) ; non saisie, le moteur de l'âme (ame.py) ne décide pas ;
- les RÉSISTANCES prennent la f_y de calcul de la section (la plus petite des parois, brief § 7.1) : m₁ = b_f/t_w,
  F_Rd, V_bw,Rd, l_e ; les ÉLANCEMENTS de plaque (λ̄_w, λ̄_F) et la limite 72·ε/η, la f_y de l'âme selon son épaisseur.

Unités internes : N, mm. Blocs de formules en kN, kN·m, mm (gabarits « a_… » de acier/barre/formules.py).
"""
import math
from typing import Dict, List, Optional

from acier.flambement import normes as NO

from . import ec3_classe as CL
from . import section as SE
from .resultat import NON_COUVERT, Journal, fr as _fr, terme as _t

METHODE = "ame"
NORME = "NBN EN 1993-1-5"
REF = "EN 1993-1-5"
TYPES = ("a", "b", "c")
CONTACTS = ("oui", "non", "inconnu")       # contact d'extrémité, type (c) — DÉCISION BUREAU n° 2 (brief § 8.2) : « inconnu » par défaut
LIBELLES_TYPE = {"a": "sur une semelle, reprise par le cisaillement de l'âme", "b": "transmise d'une semelle à l'autre",
                 "c": "près d'une extrémité non raidie"}
VOILEMENT_AME = 72.0             # h_w/t_w ≤ 72·ε/η (§ 5.1(2))
LAMBDA_W = 86.4                  # λ̄_w = h_w/(86,4·t·ε) : raidisseurs transversaux aux appuis seulement (5.5)
CHI_W = 0.83                     # tableau 5.1
SEUIL_ETA3 = 0.5                 # § 7.1(1) : pas d'interaction tant que η̄₃ ≤ 0,5
KF = {"a": 6.0, "b": 3.5}        # k_F sans raidisseur transversal voisin (figure 6.1, a → ∞)
KF_C_MAX = 6.0
SEUIL_M2 = 0.5                   # m₂ = 0,02·(h_w/t_f)² si λ̄_F > 0,5 ; 0 sinon (6.9)
CHI_F = 0.5                      # χ_F = 0,5/λ̄_F ≤ 1 (6.3)
INTERACTION_72 = (0.8, 1.4)      # η₂ + 0,8·η₁ ≤ 1,4 (7.2)
K_SEMELLE = 0.4                  # § 8(1) : 0,4 quand le moment plastique est employé (0,55 pour le moment élastique) — du côté sûr
SOUDES = ("soude_oxycoupe", "soude_larges_plats")                 # les I soudés (section.FABRICATIONS)
SOUDES_ETA = SOUDES + ("caisson", "caisson_recuit")              # les sections soudées chargées dans le plan de l'âme
# brief § 8.3, point 8 : quand le cisaillement gouverne — le voilement (A-Vb) ou, pour un laminé, V_pl,Rd (E-Vz, moteur.py)
MOT_CISAILLEMENT = "Le cisaillement gouverne : un raidisseur transversal ne réduit pas τ — plaque de doublage d'âme ou profil plus lourd."


def est_soude(s: Dict) -> bool:
    """Une section SOUDÉE chargée dans le plan de l'âme : un PRS (contrat acier/barre/prs.py : la clé « prs », quelle que
    soit sa « fabrication », None tant que les bords des semelles ne sont pas saisis), un I soudé, un caisson."""
    return bool(s.get("prs")) or s.get("fabrication") in SOUDES_ETA


def eta_de(s: Dict, P: Dict) -> float:
    """η de la section (NBN EN 1993-1-5 ANB:2011, 5.1(2), et sa note complémentaire qui renvoie au 6.2.6(3) de
    l'EN 1993-1-1 ; brief § 8) : « eta_soude » (1,20) pour une section SOUDÉE — PRS, I soudé, caisson —, « eta » (1,00)
    pour un I ou H laminé et un tube."""
    return P["eta_soude"] if est_soude(s) else P["eta"]


def hw_de(s: Dict) -> Optional[float]:
    """h_w, la hauteur de l'âme ENTRE SEMELLES d'un I (mm) : celle saisie d'un PRS (s["prs"]["hw"] : ses deux semelles
    peuvent différer d'épaisseur), h − 2·t_f d'un laminé ; None pour une autre forme."""
    p = s.get("prs")
    if s.get("forme") == "I" and p and p.get("hw"):
        return float(p["hw"])
    return SE.hauteur_ame(s)


def semelles_de(s: Dict) -> Dict[str, Dict]:
    """Les deux semelles d'un I : {"haut": {"bf", "tf"}, "bas": {…}} (mm) — celles d'un PRS (1 = semelle haute dans la
    barre posée), les mêmes deux fois pour un laminé."""
    p = s.get("prs")
    if p:
        return {"haut": {"bf": float(p["b1"]), "tf": float(p["t1"])}, "bas": {"bf": float(p["b2"]), "tf": float(p["t2"])}}
    return {k: {"bf": s["b"], "tf": s["tf"]} for k in ("haut", "bas")}


def ame_de(s: Dict, cote: str = "haut") -> Optional[Dict]:
    """L'âme d'un I ou d'un H pour les charges concentrées, chargée par la semelle ``cote`` (« haut » | « bas ») :
    {"hw", "tw", "tf", "bf"} (mm ; b_f et t_f de la semelle CHARGÉE, brief § 8.2) ; None pour une autre forme."""
    hw = hw_de(s)
    if hw is None or not s.get("tw") or not s.get("b"):
        return None
    sm = semelles_de(s)[cote]
    return {"hw": hw, "tw": s["tw"], "tf": sm["tf"], "bf": sm["bf"]}


def moment_semelles(s: Dict, fy: float, gM0: float) -> Dict:
    """M_f,Rd, le moment résistant des SEULES semelles (§ 5.4(1), § 7.1) : la plus petite semelle (aire A_f) sur le
    bras h_f entre plans moyens — b·t_f·(h − t_f) pour un laminé ; {"Af", "hf", "Mf"} (mm², mm, N·mm)."""
    sm = semelles_de(s)
    Af = min(x["bf"] * x["tf"] for x in sm.values())
    hf = s["h"] - sm["haut"]["tf"] / 2.0 - sm["bas"]["tf"] / 2.0
    return {"Af": Af, "hf": hf, "Mf": Af * hf * fy / gM0}


# ------------------------------------------------------------------ § 5 : le cisaillement
def elancement(s: Dict, fy: float, P: Dict, direction: str = "z", fy_ame: Optional[float] = None) -> Optional[Dict]:
    """L'élancement de l'âme qui porte l'effort tranchant de ``direction`` et sa limite (§ 5.1(2)) : {"hw", "tw", "n"
    (nombre d'âmes), "rapport", "limite", "elancee", "eta", "fy_ame", "eps_ame"} ; None quand la question ne se pose pas
    (tube rond, semelles d'un I sous V_y, forme hors du module). ``fy_ame`` : la f_y de l'âme selon son épaisseur (son
    ε fait la limite 72·ε/η) ; à défaut ``fy``."""
    a = CL.hauteur_cisaillee(s, direction) if s.get("forme") in ("I", "tube_carre", "tube_rect") else None
    if a is not None and s.get("forme") == "I":
        a = {"hw": hw_de(s), "tw": a["tw"]}                    # un PRS : l'âme saisie entre ses semelles
    if a is None or not a["hw"] or not a["tw"]:
        return None
    fyw = fy if fy_ame is None else float(fy_ame)
    eta, eps = eta_de(s, P), NO.epsilon(fyw)
    lim = VOILEMENT_AME * eps / eta
    r = a["hw"] / a["tw"]
    return {"hw": a["hw"], "tw": a["tw"], "n": 1 if s["forme"] == "I" else 2, "rapport": r, "limite": lim, "elancee": r > lim,
            "eta": eta, "fy_ame": fyw, "eps_ame": eps}


def resistance_cisaillement(e: Dict, fy: float, P: Dict) -> Dict:
    """V_bw,Rd d'une âme élancée (``elancement``), sans raidisseur intermédiaire, montant d'extrémité non rigide :
    {"lambda_w", "chi_w", "plafonne" (χ_w = η), "Vbw"} (N) — λ̄_w avec l'ε de l'âme, V_bw,Rd avec la f_y de calcul."""
    eta = e["eta"]
    lw = e["hw"] / (LAMBDA_W * e["tw"] * e["eps_ame"])
    plafonne = lw < CHI_W / eta
    chi = eta if plafonne else CHI_W / lw
    return {"lambda_w": lw, "chi_w": chi, "plafonne": plafonne, "Vbw": e["n"] * chi * fy * e["hw"] * e["tw"] / (math.sqrt(3.0) * P["gamma_M1"])}


def cisaillement(J: Journal, s: Dict, fy: float, P: Dict, V: float, M: float = 0.0, N: float = 0.0, direction: str = "z",
                 fy_ame: Optional[float] = None) -> None:
    """§ 5 et § 7.1 dans le journal : « A-Vb » (V_Ed ≤ V_b,Rd ; « A-Vb-y » pour l'effort tranchant parallèle aux
    semelles d'un tube) et, quand V_Ed > 0,5·V_bw,Rd sous un moment qui dépasse celui des semelles, « A-VbM ». Rien pour
    une âme qui n'est pas élancée. ``V``, ``N`` en N, ``M`` en N·mm (le moment qui fléchit CETTE âme), en valeur absolue ;
    ``fy_ame`` : la f_y de l'âme selon son épaisseur (ε de la limite et de λ̄_w)."""
    e = elancement(s, fy, P, direction, fy_ame)
    if e is None or not e["elancee"] or V <= 0:
        return
    eps, eta, gM1, gM0 = e["eps_ame"], e["eta"], P["gamma_M1"], P["gamma_M0"]
    se = "ε" if e["fy_ame"] == fy else "ε_w"
    r = resistance_cisaillement(e, fy, P)
    suf = "" if direction == "z" else "-y"
    cles = [J.f("hw_tw", "h_w/t_w", [_t("h_w", e["hw"], "mm"), _t("t_w", e["tw"], "mm")], e["rapport"], "", "a_elancement")]
    if se != "ε":
        cles.append(J.f("eps_w", "ε_w", [_t("f_y,w", e["fy_ame"], "N/mm²")], eps, "", "e_eps"))
    cles += [J.f("lim_cis", f"72·{se}/η", [_t(se, eps), _t("η", eta)], e["limite"], "", "a_limite_cis"),
             J.f("lambda_w", "λ̄_w", [_t("h_w", e["hw"], "mm"), _t("t_w", e["tw"], "mm"), _t(se, eps)], r["lambda_w"], "", "a_lambda_w")]
    if not r["plafonne"]:
        cles.append(J.f("chi_w", "χ_w", [_t("λ̄_w", r["lambda_w"])], r["chi_w"], "", "a_chi_w"))
    communs = [_t("χ_w", r["chi_w"]), _t("f_y", fy, "N/mm²"), _t("h_w", e["hw"], "mm"), _t("t_w", e["tw"], "mm"), _t("γ_M1", gM1)]
    if e["n"] == 1:
        cles.append(J.f("V_bw", "V_bw,Rd", communs, r["Vbw"] / 1e3, "kN", "a_V_bw"))
    else:
        cles.append(J.f("V_bw", "V_bw,Rd", [_t("n", e["n"])] + communs, r["Vbw"] / 1e3, "kN", "a_V_bw_n"))
    J.G["voilement_cisaillement" + suf.replace("-", "_")] = dict(e, **r)
    mots = ["χ_w = η : l'âme n'est pas réduite."] if r["plafonne"] else []
    if e["n"] > 1:
        mots.append("Les deux âmes du tube (n = 2).")
    if V > r["Vbw"]:                                         # brief § 8.3, point 8 : le raidisseur transversal n'y peut rien
        mots.append(MOT_CISAILLEMENT)
    J.verif("A-Vb" + suf, "Voilement de l'âme par cisaillement", f"{REF} § 5.2", "V_Ed ≤ V_b,Rd", V / 1e3, r["Vbw"] / 1e3, "kN", cles, messages=mots)
    eta3 = V / r["Vbw"]
    if eta3 <= SEUIL_ETA3 or M <= 0:
        return
    if s["forme"] != "I" or N != 0:
        quoi = "d'un tube" if s["forme"] != "I" else "sous un effort normal"
        J.verif("A-VbM" + suf, "Âme élancée : flexion et effort tranchant", f"{REF} § 7.1", "", None, None, "", [], statut=NON_COUVERT,
                messages=[J.alerte("interaction_71", "attention", f"Interaction flexion – cisaillement {quoi} (V_Ed > 0,5·V_bw,Rd) : non vérifiée.")])
        return
    mf = moment_semelles(s, fy, gM0)
    Mf = mf["Mf"]
    if M <= Mf:                                              # les semelles seules portent le moment : (7.1) est sans objet
        return
    Mpl = s["Wpl_y"] * fy / gM0
    eta1b = M / Mpl
    v = eta1b + (1.0 - Mf / Mpl) * (2.0 * eta3 - 1.0) ** 2
    cles2 = [J.f("eta3", "η̄₃", [_t("V_Ed", V / 1e3, "kN"), _t("V_bw,Rd", r["Vbw"] / 1e3, "kN")], eta3, "", "a_eta3"),
             (J.f("M_f", "M_f,Rd", [_t("b", s["b"], "mm"), _t("t_f", s["tf"], "mm"), _t("h", s["h"], "mm"), _t("f_y", fy, "N/mm²"), _t("γ_M0", gM0)],
                  Mf / 1e6, "kN·m", "a_M_f") if not s.get("prs") else
              J.f("M_f", "M_f,Rd", [_t("A_f", mf["Af"], "mm²"), _t("h_f", mf["hf"], "mm"), _t("f_y", fy, "N/mm²"), _t("γ_M0", gM0)],
                  Mf / 1e6, "kN·m", "a_M_f_prs")),
             J.f("M_pl", "M_pl,Rd", [_t("W_pl,y", s["Wpl_y"] / 1e3, "cm³"), _t("f_y", fy, "N/mm²"), _t("γ_M0", gM0)], Mpl / 1e6, "kN·m", "e_M_Rd"),
             J.f("eta1_bar", "η̄₁", [_t("M_Ed", M / 1e6, "kN·m"), _t("M_pl,Rd", Mpl / 1e6, "kN·m")], eta1b, "", "a_eta1_bar"),
             J.f("i71", "η", [_t("η̄₁", eta1b), _t("M_f,Rd", Mf / 1e6, "kN·m"), _t("M_pl,Rd", Mpl / 1e6, "kN·m"), _t("η̄₃", eta3)], v, "", "a_71")]
    J.verif("A-VbM", "Âme élancée : flexion et effort tranchant", f"{REF} § 7.1", "η̄₁ + (1 − M_f,Rd/M_pl,Rd)·(2·η̄₃ − 1)² ≤ 1", v, 1.0, "",
            cles2, taux=v)


# ------------------------------------------------------------------ § 6 : la charge concentrée
def resistance(a: Dict, fy: float, P: Dict, ss: float, type_: str, c: float = 0.0, contact=False,
               fy_ame: Optional[float] = None, panneau: Optional[float] = None) -> Dict:
    """F_Rd de l'âme ``a`` (``ame_de``) sous une force appliquée sur la longueur d'appui rigide ``ss`` (mm), selon le
    ``type_`` de la figure 6.1 ; ``c`` : la distance de l'extrémité au bord de la longueur chargée, type (c) ;
    ``contact`` (type (c), DÉCISION BUREAU n° 2, brief § 8.2) : « oui » (ou True) — s_s entre dans (6.10) ; « non » —
    s_s = 0 dans (6.10) seulement ; « inconnu » (ou False, le défaut) — le verdict de « non », et F_Rd avec contact
    donnée pour information ; ``fy_ame`` : la f_y de l'âme selon t_w, celle de λ̄_F (à défaut ``fy``) ; ``panneau`` :
    a, la distance entre raidisseurs transversaux (mm ; None : aucun raidisseur voisin, a → ∞) — k_F = 6 + 2·(h_w/a)²
    ou 3,5 + 2·(h_w/a)², et l_y ≤ a (§ 6.5(2)).

    Rend {"type", "ss" (bornée à h_w), "ss_610" (le s_s de (6.10)), "c", "contact", "kF", "kF_plafonne", "Fcr", "m1",
    "m2" (la branche retenue), "deux_branches", "FRd_autre", "m2_autre" (l'autre branche cohérente, None sinon), "le",
    "le_plafonne", "ly1", "ly2", "ly3", "ly", "lambda_F", "chi_F", "FRd", "FRd_contact" (type (c) sans contact, s_s > 0 :
    F_Rd avec contact, pour information ; None sinon), "fy_ame"} — N et mm ; ``le``, ``ly2``, ``ly3`` : None hors du
    type (c)."""
    if type_ not in TYPES:
        raise ValueError(f"type de charge concentrée inconnu : {type_!r}")
    etat = {True: "oui", False: "inconnu"}.get(contact, contact) if isinstance(contact, bool) else contact
    if etat not in CONTACTS:
        raise ValueError(f"contact d'extrémité : {', '.join(CONTACTS)} — reçu {contact!r}")
    contact = etat == "oui"
    hw, tw, tf, bf = a["hw"], a["tw"], a["tf"], a["bf"]
    E = P["E"]
    fyw = fy if fy_ame is None else float(fy_ame)
    ss = min(max(float(ss), 0.0), hw)                          # § 6.3(1) : s_s ≤ h_w
    c = max(float(c), 0.0)
    ss610 = ss if (type_ != "c" or contact) else 0.0            # contact non établi : s_s = 0 dans (6.10) seulement (JRC p. 79)
    brut = 2.0 + 6.0 * (ss + c) / hw
    voisin = 0.0 if panneau is None else 2.0 * (hw / float(panneau)) ** 2      # (6.4) : le raidisseur transversal voisin
    kF = KF[type_] + voisin if type_ != "c" else min(brut, KF_C_MAX)
    Fcr = 0.9 * kF * E * tw ** 3 / hw
    m1 = bf / tw                                               # f_yf = f_yw
    le = le_plafonne = None
    if type_ == "c":
        le_libre = kF * E * tw ** 2 / (2.0 * fy * hw)           # f_y de calcul (brief § 7.1 : une résistance)
        le_plafonne = le_libre > ss + c
        le = min(le_libre, ss + c)

    def longueurs(m2: float) -> Dict:
        ly1 = ss610 + 2.0 * tf * (1.0 + math.sqrt(m1 + m2))
        if type_ != "c":
            return {"ly1": ly1, "ly2": None, "ly3": None, "ly": ly1 if panneau is None else min(ly1, float(panneau))}
        ly2 = le + tf * math.sqrt(m1 / 2.0 + (le / tf) ** 2 + m2)
        ly3 = le + tf * math.sqrt(m1 + m2)
        return {"ly1": ly1, "ly2": ly2, "ly3": ly3, "ly": min(ly1, ly2, ly3)}

    def branche(m2: float) -> Dict:
        L = longueurs(m2)
        lF = math.sqrt(L["ly"] * tw * fyw / Fcr)
        chi = min(CHI_F / lF, 1.0)
        return dict(L, m2=m2, lambda_F=lF, chi_F=chi, FRd=fy * chi * L["ly"] * tw / P["gamma_M1"])

    # (6.9) : m₂ = 0,02·(h_w/t_f)² ne vaut que si λ̄_F > 0,5, m₂ = 0 que si λ̄_F ≤ 0,5. Les DEUX branches sont calculées ;
    # au moins une est cohérente (λ̄_F croît avec m₂) ; quand les deux le sont, la plus petite F_Rd est retenue (brief § 8.2)
    avec, sans = branche(0.02 * (hw / tf) ** 2), branche(0.0)
    coherentes = [x for x, ok in ((sans, sans["lambda_F"] <= SEUIL_M2), (avec, avec["lambda_F"] > SEUIL_M2)) if ok]
    B = min(coherentes, key=lambda x: x["FRd"])
    autre = next((x for x in coherentes if x is not B), None)
    out = dict(B, type=type_, ss=ss, ss_610=ss610, c=c, contact=bool(contact), contact_etat=etat, panneau=panneau, kF=kF, kF_plafonne=type_ == "c" and brut > KF_C_MAX, Fcr=Fcr,
               m1=m1, le=le, le_plafonne=le_plafonne, deux_branches=autre is not None, FRd_autre=None if autre is None else autre["FRd"],
               m2_autre=None if autre is None else autre["m2"], FRd_contact=None, fy_ame=fyw)
    if type_ == "c" and etat == "inconnu" and ss > 0:
        out["FRd_contact"] = resistance(a, fy, P, ss, type_, c, contact=True, fy_ame=fy_ame, panneau=panneau)["FRd"]
    return out


def ecrire_resistance(J: Journal, a: Dict, fy: float, P: Dict, r: Dict, prefixe: str = "") -> List[str]:
    """Les blocs de formules de ``resistance`` dans le journal ; rend leurs clés, dans l'ordre du calcul. ``prefixe`` :
    devant chaque clé, quand le journal porte déjà une autre résistance (la force transmise, type (b))."""
    hw, tw, tf, bf = a["hw"], a["tw"], a["tf"], a["bf"]
    cles = []

    def f(cle: str, *reste) -> str:
        return J.f(prefixe + cle, *reste)

    if r["type"] == "c" and not r["kF_plafonne"]:
        cles.append(f("kF", "k_F", [_t("s_s", r["ss"], "mm"), _t("c", r["c"], "mm"), _t("h_w", hw, "mm")], r["kF"], "", "a_kF_c"))
    cles.append(f("F_cr", "F_cr", [_t("k_F", r["kF"]), _t("E", P["E"], "N/mm²"), _t("t_w", tw, "mm"), _t("h_w", hw, "mm")], r["Fcr"] / 1e3, "kN",
                    "a_F_cr"))
    cles.append(f("m1", "m₁", [_t("b_f", bf, "mm"), _t("t_w", tw, "mm")], r["m1"], "", "a_m1"))
    if r["m2"] > 0:
        cles.append(f("m2", "m₂", [_t("h_w", hw, "mm"), _t("t_f", tf, "mm")], r["m2"], "", "a_m2"))
    m2 = _t("m₂", r["m2"])
    if r["type"] != "c":
        cles.append(f("ly", "l_y", [_t("s_s", r["ss"], "mm"), _t("t_f", tf, "mm"), _t("m₁", r["m1"]), m2], r["ly"], "mm", "a_ly"))
    else:
        if not r["le_plafonne"]:
            cles.append(f("le", "l_e", [_t("k_F", r["kF"]), _t("E", P["E"], "N/mm²"), _t("t_w", tw, "mm"), _t("f_y", fy, "N/mm²"), _t("h_w", hw, "mm")],
                            r["le"], "mm", "a_le"))
        le = _t("l_e", r["le"], "mm")
        cles.append(f("ly1", "l_y,1", [_t("s_s", r["ss_610"], "mm"), _t("t_f", tf, "mm"), _t("m₁", r["m1"]), m2], r["ly1"], "mm", "a_ly"))
        cles.append(f("ly2", "l_y,2", [le, _t("t_f", tf, "mm"), _t("m₁", r["m1"]), m2], r["ly2"], "mm", "a_ly_c2"))
        cles.append(f("ly3", "l_y,3", [le, _t("t_f", tf, "mm"), _t("m₁", r["m1"]), m2], r["ly3"], "mm", "a_ly_c3"))
    fyw = r.get("fy_ame", fy)
    cles.append(f("lambda_F", "λ̄_F", [_t("l_y", r["ly"], "mm"), _t("t_w", tw, "mm"), _t("f_y" if fyw == fy else "f_y,w", fyw, "N/mm²"),
                                       _t("F_cr", r["Fcr"] / 1e3, "kN")], r["lambda_F"], "", "a_lambda_F"))
    if r["chi_F"] < 1.0:
        cles.append(f("chi_F", "χ_F", [_t("λ̄_F", r["lambda_F"])], r["chi_F"], "", "a_chi_F"))
    cles.append(f("F_Rd", "F_Rd", [_t("f_y", fy, "N/mm²"), _t("χ_F", r["chi_F"]), _t("l_y", r["ly"], "mm"), _t("t_w", tw, "mm"),
                                     _t("γ_M1", P["gamma_M1"])], r["FRd"] / 1e3, "kN", "a_F_Rd"))
    return cles


def mots_resistance(r: Dict) -> List[str]:
    """Ce que les formules ne disent pas d'elles-mêmes, en UNE phrase : le type de la force, les plafonds et les choix
    du calcul de F_Rd."""
    mots = [f"Type ({r['type']}) : force {LIBELLES_TYPE[r['type']]}"]
    if r["type"] != "c":
        mots.append(f"k_F = {_fr(r['kF'], 1)} sans raidisseur transversal voisin" if r.get("panneau") is None
                    else f"raidisseurs transversaux à a = {_fr(r['panneau'])} mm")
    else:
        if r["kF_plafonne"]:
            mots.append("k_F plafonné à 6")
        if r["le_plafonne"]:
            mots.append("l_e plafonné à s_s + c")
        mots.append("l_y : la plus petite des trois expressions")
        if r.get("FRd_contact") is not None:
            mots.append("hypothèse de contact non établie : s_s = 0 dans (6.10) (verdict) ; avec contact, "
                        f"F_Rd = {_fr(r['FRd_contact'] / 1e3, 1)} kN, pour information")
        elif r.get("contact_etat") == "non" and r["ss"] > 0:
            mots.append("sans contact d'extrémité : s_s = 0 dans (6.10)")
        elif r.get("contact_etat") == "oui":
            mots.append("contact d'extrémité établi : s_s dans (6.10)")
    if r["m2"] <= 0:
        mots.append("λ̄_F ≤ 0,5 : m₂ = 0")
    if r.get("deux_branches"):
        mots.append(f"deux branches de m₂ cohérentes : la plus petite F_Rd est retenue (l'autre : {_fr(r['FRd_autre'] / 1e3, 1)} kN)")
    if r["chi_F"] >= 1.0:
        mots.append("χ_F = 1")
    return [" ; ".join(mots) + "."]


def eta1(N: float, M: float, A: float, W: float, fy: float, gM0: float) -> float:
    """η₁ élastique (§ 4.6, sans excentrement) : |N|/(f_y·A/γ_M0) + |M|/(f_y·W/γ_M0) — A_eff et W_eff en classe 4."""
    return abs(N) / (A * fy / gM0) + abs(M) / (W * fy / gM0)


def ecrire_eta1(J: Journal, N: float, M: float, A: float, W: float, fy: float, gM0: float, nomA: str = "A", nomW: str = "W_el,y") -> str:
    """Le bloc de η₁ ; sa clé. Un terme nul ne s'écrit pas."""
    v = eta1(N, M, A, W, fy, gM0)
    tN = [_t("N_Ed", abs(N) / 1e3, "kN"), _t(nomA, A, "mm²")]
    tM = [_t("M_Ed", abs(M) / 1e6, "kN·m"), _t(nomW, W / 1e3, "cm³")]
    fin = [_t("f_y", fy, "N/mm²"), _t("γ_M0", gM0)]
    if N != 0 and M != 0:
        return J.f("eta1", "η₁", tN + tM + fin, v, "", "a_eta1")
    if N != 0:
        return J.f("eta1", "η₁", tN + fin, v, "", "a_eta1_N")
    return J.f("eta1", "η₁", tM + fin, v, "", "a_eta1_M")


def interaction(e2: float, e1: float) -> float:
    """Le taux de (7.2) : (η₂ + 0,8·η₁)/1,4."""
    k, lim = INTERACTION_72
    return (e2 + k * e1) / lim


def seuil_local(FRd: float, e1: float) -> float:
    """La force au-delà de laquelle l'âme ne passe plus, à M_Ed et N_Ed IMPOSÉS (brief § 8.2, « seuil de résistance
    locale ») : min(F_Rd ; (1,4 − 0,8·η₁)·F_Rd) — η₂ ≤ 1 et (7.2). Ce n'est PAS la capacité de la barre : quand la
    charge croît, le moment croît avec elle."""
    k, lim = INTERACTION_72
    return max(0.0, min(FRd, (lim - k * e1) * FRd))


# ------------------------------------------------------------------ § 7.2(2) : la charge sur la semelle TENDUE
COMPRIMEE, TENDUE, LES_DEUX = "comprimee", "tendue", "les_deux"
ETATS_SEMELLE = {COMPRIMEE: "semelle comprimée", TENDUE: "semelle tendue", LES_DEUX: "σ_x nul ou de signes mêlés"}
# ce que dit la vérification « -EL », en une ligne
MOTS_ETAT = {TENDUE: "Semelle chargée tendue : pas d'interaction (7.2), le critère 6.2.1(5) décide.",
             LES_DEUX: "σ_x nul ou de signes mêlés dans la semelle chargée : les deux contrôles."}


def niveaux_semelle(s: Dict, cote: str) -> Dict:
    """Les cotes de la semelle ``cote`` depuis le centre de gravité, z vers le haut (mm) : {"z_moyen" (son plan moyen),
    "z_jonction" (la jonction âme-semelle), "S" (le moment statique de la semelle, mm³), "bf", "tf"} — z_g d'un PRS
    (contrat de prs.py), h/2 d'un laminé."""
    sm = semelles_de(s)[cote]
    zg = s.get("z_g") or s["h"] / 2.0
    if cote == "haut":
        zm, zj = s["h"] - zg - sm["tf"] / 2.0, s["h"] - zg - sm["tf"]
    else:
        zm, zj = -(zg - sm["tf"] / 2.0), -(zg - sm["tf"])
    return {"z_moyen": zm, "z_jonction": zj, "S": sm["bf"] * sm["tf"] * abs(zm), "bf": sm["bf"], "tf": sm["tf"]}


def sigma_x(N: float, M: float, A: float, Iy: float, z: float, Mz: float = 0.0, Iz: Optional[float] = None, y: float = 0.0) -> float:
    """σ_x = N/A + M_y·z/I_y + M_z·y/I_z (N/mm², COMPRESSION POSITIVE : N > 0 comprime, M_y > 0 comprime la fibre haute —
    les conventions de acier/barre/statique.py) — brief § 8.2."""
    return N / A + M * z / Iy + (Mz * y / Iz if Mz and Iz else 0.0)


def etat_semelle(sigmas) -> str:
    """L'état de la semelle chargée d'après σ_x sur toute sa largeur (brief § 8.2) : comprimée partout, tendue partout,
    ou « les deux » — signes mêlés ou contrainte nulle (un appui d'extrémité) : les deux contrôles, le plus défavorable
    retenu (choix déclaré du brief)."""
    v = list(sigmas)
    if all(x > EPS_SIGMA for x in v):
        return COMPRIMEE
    if all(x < -EPS_SIGMA for x in v):
        return TENDUE
    return LES_DEUX


EPS_SIGMA = 1e-6                # N/mm² : une contrainte plus petite est nulle


def sigma_z(F: float, ss: float, tf: float, tw: float, z: float = 0.0, ast1: float = 0.0) -> Dict:
    """La compression locale de l'âme sous une force concentrée (EN 1993-1-5 § 3.2.3, (3.2)) :
    σ_z,Ed = F_Ed/(b_eff·(t_w + a_st,1)), b_eff = s_e·√(1 + (z/(s_e·n))²), n = 0,636·√(1 + 0,878·a_st,1/t_w), s_e = s_s + 2·t_f ;
    à la jonction (z = 0), sans raidisseur : F_Ed/(s_e·t_w). {"se", "n", "beff", "sigma_z"} (mm, N/mm²)."""
    se = ss + 2.0 * tf
    n = 0.636 * math.sqrt(1.0 + 0.878 * ast1 / tw)
    beff = se * math.sqrt(1.0 + (z / (se * n)) ** 2)
    return {"se": se, "n": n, "beff": beff, "sigma_z": F / (beff * (tw + ast1))}


def critere_elastique(sx: float, sz: float, tau: float, fy: float, gM0: float) -> float:
    """Le critère de l'EN 1993-1-1 § 6.2.1(5) (6.1), au point considéré :
    (σ_x/f)² + (σ_z/f)² − (σ_x/f)·(σ_z/f) + 3·(τ/f)², f = f_y/γ_M0 — σ_x et σ_z avec leur signe (compression positive)."""
    f = fy / gM0
    return (sx / f) ** 2 + (sz / f) ** 2 - (sx / f) * (sz / f) + 3.0 * (tau / f) ** 2


# ------------------------------------------------------------------ § 8 : la semelle
def semelle(s: Dict, fy: float, P: Dict, k: float = K_SEMELLE, fy_semelle: Optional[float] = None) -> Optional[Dict]:
    """Le voilement de l'âme induit par la semelle comprimée (§ 8) : {"rapport", "limite", "k", "ok", "bf", "tf",
    "fyf"} ; None hors des I. Semelles inégales (un PRS) : la PLUS GRANDE des deux aires — NBN EN 1993-1-5 ANB:2011,
    8(2) : A_ft remplace A_fc quand elle la dépasse ; la semelle comprimée n'est pas connue ici, d'où le côté sûr.
    ``fy_semelle`` : la f_y de la semelle (f_yf), à défaut ``fy``."""
    a = ame_de(s)
    if a is None:
        return None
    fl = max(semelles_de(s).values(), key=lambda x: x["bf"] * x["tf"])
    fyf = fy if fy_semelle is None else float(fy_semelle)
    lim = k * P["E"] / fyf * math.sqrt(a["hw"] * a["tw"] / (fl["bf"] * fl["tf"]))
    return {"rapport": a["hw"] / a["tw"], "limite": lim, "k": k, "ok": a["hw"] / a["tw"] <= lim, "bf": fl["bf"], "tf": fl["tf"], "fyf": fyf}


def ecrire_semelle(J: Journal, s: Dict, fy: float, P: Dict, v: Dict) -> None:
    """« A-S » : seulement quand la limite est DÉPASSÉE (jamais sur un laminé courant) ; sinon le moteur de l'âme le dit
    en une ligne d'hypothèse."""
    a = ame_de(s)
    cles = [J.f("hw_tw_s", "h_w/t_w", [_t("h_w", a["hw"], "mm"), _t("t_w", a["tw"], "mm")], v["rapport"], "", "a_elancement"),
            J.f("lim_semelle", "k·(E/f_yf)·√(A_w/A_fc)", [_t("k", v["k"]), _t("E", P["E"], "N/mm²"), _t("f_yf", v.get("fyf", fy), "N/mm²"),
                                                           _t("h_w", a["hw"], "mm"), _t("t_w", a["tw"], "mm"), _t("b_f", v.get("bf", a["bf"]), "mm"),
                                                           _t("t_f", v.get("tf", a["tf"]), "mm")],
                v["limite"], "", "a_semelle")]
    J.verif("A-S", "Voilement de l'âme induit par la semelle", f"{REF} § 8", "h_w/t_w ≤ k·(E/f_yf)·√(A_w/A_fc)", v["rapport"], v["limite"], "", cles)

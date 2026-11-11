# -*- coding: utf-8 -*-
"""Résistances du plancher (§ 2 et § 3 de l'étude).

- Béton armé (dalle, prédalle) : les formules de ``modules/poutre.py`` et
  ``modules/dalle.py`` INVERSÉES — M_adm = min(αb·μ·b·d² ; As·fyd·0,9·d),
  V_adm = τadm,I·0,75·b·h — sur les classes de ``beton_classes.json``. Ces
  formules existent dans ``poutre.py`` à l'intérieur d'une fonction liée à
  l'état Streamlit : elles sont RECOPIÉES ici, nommées, et le test
  ``tests/test_charges_plancher_moteur.py`` les lie au texte de
  ``poutre.py`` (copie tenue = vue, R2).
- Hourdis : résistances de catalogue avec leur nature (P-1) : valeurs de
  CALCUL (M_Rd, V_Rd), comparées à des efforts ELU — γG sur les poids
  propres, γQ sur les charges de chantier (P-2, tranché le 02/11/2026) —
  ou valeurs ADMISSIBLES de service, comparées aux efforts de service
  (coefficients 1) ; valeurs « avec chape » seulement si la chape les
  autorise (P-17).
- Effets locaux : EN 1992-1-1 §6.7 (pression localisée) et §6.4
  (poinçonnement), en ELU avec γQ sur la charge (P-9).
"""
import json
import math
import os
from typing import Dict, Optional

from .modele import Hourdis, DalleBA, Chape, RD, SERVICE

_RACINE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CHEMIN_BETON = os.path.join(_RACINE, "beton_classes.json")


def _charger_beton():
    with open(_CHEMIN_BETON, "r", encoding="utf-8") as fh:
        return json.load(fh)


BETON = _charger_beton()
"""Classes de béton de l'application : fck, fck_cube, tau_lim, alpha_b, mu_a500, mu_a400."""


# ---------------------------------------------------------------- classes de béton
def classe_pour_fck(fck_t: float) -> Optional[str]:
    """La classe dont fck est la plus grande ≤ fck_t (arrondi vers le bas) ;
    ``None`` sous la plus petite classe."""
    meilleure, fck_best = None, -1.0
    for nom, v in BETON.items():
        if v["fck"] <= fck_t + 1e-9 and v["fck"] > fck_best:
            meilleure, fck_best = nom, v["fck"]
    return meilleure


def fck_de_classe(classe: str) -> float:
    return float(BETON[classe]["fck"])


def Ecm(fck: float) -> float:
    """Module sécant du béton, EN 1992-1-1 Tableau 3.1 : Ecm = 22·((fck + 8)/10)^0,3 GPa → MPa."""
    return 22.0 * ((fck + 8.0) / 10.0) ** 0.3 * 1.0e3


# ---------------------------------------------------------------- formules BA de l'application (copie nommée)
def fyd(fyk: float, gamma_s: float) -> float:
    """fyd = fyk / γs — poutre.py : ``fyd = fyk / gamma_s``."""
    return fyk / gamma_s


def hu_min_cm(M_kNm: float, alpha_b: float, b_cm: float, mu: float) -> float:
    """poutre.py : ``math.sqrt((m * 1e6) / (alpha_b * b * 10 * mu_val)) / 10`` (cm)."""
    return math.sqrt((M_kNm * 1e6) / (alpha_b * b_cm * 10 * mu)) / 10 if M_kNm > 0 else 0.0


def as_req_mm2(M_kNm: float, fyd_: float, d_cm: float) -> float:
    """poutre.py : ``(M * 1e6) / (fyd * 0.9 * d * 10)`` (mm²)."""
    return (M_kNm * 1e6) / (fyd_ * 0.9 * d_cm * 10) if M_kNm > 0 else 0.0


def tau_N_mm2(V_kN: float, b_cm: float, h_cm: float) -> float:
    """poutre.py : ``V * 1e3 / (0.75 * b * h * 100)``."""
    return V_kN * 1e3 / (0.75 * b_cm * h_cm * 100)


def tau_adm_I(fck_cube: float) -> float:
    """poutre.py : ``tau_1 = 0.016 * fck_cube / 1.05`` — seuil « pas besoin d'étriers »."""
    return 0.016 * fck_cube / 1.05


def m_adm_beton_kNm(alpha_b: float, mu: float, b_cm: float, d_cm: float) -> float:
    """hᵤ,min résolue en M : M = αb·μ·(b·10)·(d·10)² / 1e6 (kN·m)."""
    return alpha_b * mu * (b_cm * 10) * (d_cm * 10) ** 2 / 1e6


def m_adm_acier_kNm(As_mm2: float, fyd_: float, d_cm: float) -> float:
    """As,req résolue en M : M = As·fyd·0,9·(d·10) / 1e6 (kN·m)."""
    return As_mm2 * fyd_ * 0.9 * (d_cm * 10) / 1e6


def v_adm_kN(tau: float, b_cm: float, h_cm: float) -> float:
    """τ résolue en V : V = τ·0,75·(b·h·100) / 1e3 (kN)."""
    return tau * 0.75 * b_cm * h_cm * 100 / 1e3


def resistance_dalle(dalle: DalleBA) -> Dict:
    """Résistances PAR MÈTRE de largeur d'une dalle ou d'une prédalle chapée :
    M_adm (kN·m/m), V_adm (kN/m), avec le détail (classe retenue, αb, μ)."""
    classe = dalle.classe_calcul()
    v = BETON[classe]
    mu_key = "mu_a500" if abs(dalle.fyk - 500) < 1e-9 else ("mu_a400" if abs(dalle.fyk - 400) < 1e-9 else "mu_a500")
    mu = v[mu_key]
    f = fyd(dalle.fyk, dalle.gamma_s)
    b_cm, d_cm, h_cm = 100.0, dalle.d * 100.0, dalle.h * 100.0
    m_b = m_adm_beton_kNm(v["alpha_b"], mu, b_cm, d_cm)
    m_s = m_adm_acier_kNm(dalle.As_inf, f, d_cm)
    tau1 = tau_adm_I(v["fck_cube"])
    return {
        "classe": classe, "alpha_b": v["alpha_b"], "mu": mu, "fyd": f, "fck": v["fck"], "fck_cube": v["fck_cube"],
        "M_adm_beton": m_b, "M_adm_acier": m_s, "M_adm": min(m_b, m_s),
        "gouverne_M": "béton" if m_b <= m_s else "acier",
        "tau_adm_I": tau1, "V_adm": v_adm_kN(tau1, b_cm, h_cm),
        "methode": "contraintes admissibles de l'application (αb, μ, τadm,I), charges de service",
    }


# ---------------------------------------------------------------- hourdis
def resistance_hourdis(h: Hourdis, gamma_G: float = 1.35, gamma_Q: float = 1.5) -> Dict:
    """Résistances PAR ÉLÉMENT et coefficients à appliquer aux efforts :
    - nature Rd : M_adm = M_Rd, V_adm = V_Rd, efforts à l'ELU — γG sur les
      poids propres, γQ sur les charges de chantier (P-2, 02/11/2026) ;
    - nature service : valeurs admissibles telles quelles, efforts de
      service (coefficients 1 et 1) ;
    - valeurs « avec chape » si la chape les autorise (P-17 : collaborante,
      classe exigée atteinte, épaisseur du fabricant ``e_chape`` si elle est
      donnée), sinon valeurs nues, la chape ne comptant que comme poids ;
      M et V sont jugés SÉPARÉMENT : une valeur « avec chape » que le
      fabricant ne publie pas (V_Rd,c chez Ergon) reste la valeur nue, la
      plus faible (prudent)."""
    admises = bool(h.chape and h.chape.valeurs_avec_chape_admises(h.e_chape))
    M_chape = admises and h.M_Rd_chape is not None
    V_chape = admises and h.V_Rd_chape is not None
    avec_chape = M_chape or V_chape
    M_Rd = h.M_Rd_chape if M_chape else h.M_Rd
    V_Rd = h.V_Rd_chape if V_chape else h.V_Rd
    if h.nature == RD:
        gG, gQ = gamma_G, gamma_Q
        methode = (f"résistances de calcul du catalogue : M_Ed = γG·M_G + γQ·M_Q ≤ M_Rd, γG = {gamma_G:g} sur les poids "
                   f"propres, γQ = {gamma_Q:g} sur les charges de chantier (ELU, P-2)")
    elif h.nature == SERVICE:
        gG, gQ = 1.0, 1.0
        methode = "charges admissibles de service du catalogue : M_G + M_Q ≤ M_adm, coefficients 1 (P-2)"
    else:
        raise ValueError(f"Nature de catalogue non prise en charge en V1 : {h.nature!r} (Rd ou service)")
    note = ""
    if h.chape and not admises:
        note = (f"chape présente mais résistances « avec chape » non admises (P-17 : {h.chape.motif_refus(h.e_chape)}) : "
                "valeurs nues, chape comptée en poids")
    elif h.chape and not avec_chape:
        note = "chape admise mais aucune valeur « avec chape » au catalogue (P-17) : valeurs nues, chape comptée en poids"
    elif h.chape and not (M_chape and V_chape):
        nue = "V_Rd" if M_chape else "M_Rd"
        note = (f"{'M_Rd' if M_chape else 'V_Rd'} avec chape ; {nue} nu : la valeur avec chape n'est pas au catalogue, "
                "la valeur nue est la plus faible (P-17)")
    return {"M_adm": M_Rd, "V_adm": V_Rd, "M_Rd": M_Rd, "V_Rd": V_Rd, "avec_chape": avec_chape,
            "M_avec_chape": M_chape, "V_avec_chape": V_chape,
            "nature": h.nature, "gamma_G": gG, "gamma_Q": gQ, "methode": methode, "note": note,
            "M_cr": h.M_cr, "p_loc_adm": h.p_loc_adm}


# ---------------------------------------------------------------- effets locaux EN 1992-1-1
def f_cd(fck: float, gamma_c: float = 1.5, alpha_cc: float = 1.0) -> float:
    return alpha_cc * fck / gamma_c


def pression_localisee_Rd(a0_mm: float, b0_mm: float, h_mm: float, fck: float,
                          gamma_c: float = 1.5, alpha_cc: float = 1.0) -> Dict:
    """EN 1992-1-1 §6.7, éq. (6.63) : F_Rdu = A_c0·f_cd·√(A_c1/A_c0) ≤ 3·f_cd·A_c0,
    l'aire de diffusion A_c1 ayant la forme de A_c0 avec b2 ≤ 3·b1 et
    h ≥ b2 − b1 (diffusion dans l'épaisseur disponible). Résultat en kN."""
    fcd = f_cd(fck, gamma_c, alpha_cc)
    a1 = min(3.0 * a0_mm, a0_mm + h_mm)
    b1 = min(3.0 * b0_mm, b0_mm + h_mm)
    Ac0, Ac1 = a0_mm * b0_mm, a1 * b1
    F = min(Ac0 * fcd * math.sqrt(Ac1 / Ac0), 3.0 * fcd * Ac0)
    return {"F_Rdu_kN": F / 1e3, "A_c0": Ac0, "A_c1": Ac1, "f_cd": fcd, "a1": a1, "b1": b1}


def poinconnement(V_Ed_kN: float, a0_mm: float, b0_mm: float, d_mm: float, fck: float, rho_l: float,
                  gamma_c: float = 1.5, alpha_cc: float = 1.0, beta: float = 1.0,
                  coef_vrd_max: Optional[float] = 0.4) -> Dict:
    """EN 1992-1-1 §6.4 pour une charge concentrée sur une dalle :
    v_Ed = β·V_Ed/(u1·d) sur le contour u1 à 2d de l'aire chargée ;
    v_Rd,c = C_Rd,c·k·(100·ρl·fck)^(1/3) ≥ v_min (éq. 6.47, 6.3N), avec
    C_Rd,c = 0,18/γc, k = 1 + √(200/d) ≤ 2, v_min = 0,035·k^(3/2)·fck^(1/2) ;
    au nu de l'aire chargée (u0), v_Ed ≤ coef·ν·f_cd avec ν = 0,6·(1 − fck/250)
    (§6.4.5(3), coefficient à confirmer sur l'ANB : ``None`` = non vérifié).
    Contraintes en MPa, V en kN, longueurs en mm."""
    d = d_mm
    u1 = 2.0 * (a0_mm + b0_mm) + 2.0 * math.pi * 2.0 * d
    u0 = 2.0 * (a0_mm + b0_mm)
    v_Ed = beta * V_Ed_kN * 1e3 / (u1 * d)
    k = min(1.0 + math.sqrt(200.0 / d), 2.0)
    C = 0.18 / gamma_c
    rho = min(rho_l, 0.02)
    v_Rdc = max(C * k * (100.0 * rho * fck) ** (1.0 / 3.0), 0.035 * k ** 1.5 * math.sqrt(fck))
    out = {"u1_mm": u1, "u0_mm": u0, "v_Ed": v_Ed, "v_Rd_c": v_Rdc, "k": k, "rho_l": rho, "taux": v_Ed / v_Rdc}
    if coef_vrd_max is not None:
        nu = 0.6 * (1.0 - fck / 250.0)
        v_Rdmax = coef_vrd_max * nu * f_cd(fck, gamma_c, alpha_cc)
        v_Ed0 = beta * V_Ed_kN * 1e3 / (u0 * d)
        out.update({"v_Ed0": v_Ed0, "v_Rd_max": v_Rdmax, "taux_u0": v_Ed0 / v_Rdmax})
    return out

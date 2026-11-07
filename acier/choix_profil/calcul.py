# -*- coding: utf-8 -*-
"""Les fonctions de calcul de l'écran Streamlit ``modules/choix_profile.py``, RECOPIÉES sans Streamlit ni pandas.

Chaque fonction reprend l'expression du module DANS LE MÊME ORDRE D'OPÉRATIONS : les flottants sont identiques au bit
près (``tests/test_choix_profil_moteur.py`` le vérifie en appelant les deux). Les noms sont français ; la fonction du
module est citée dans chaque docstring, et les clés des dictionnaires rendus sont celles du module (``N_Rd_N``,
``rMy``…) pour que la correspondance se lise sans table.

Unités du module : efforts en kN, moments en kN·m, f_y en N/mm² ; aires en cm², modules en cm³, inerties en cm⁴ ;
longueurs de flambement en m. Les résistances rendues sont en N et N·mm (comme le module).
"""
import math
from functools import lru_cache
from typing import Dict, Optional

# les familles que le module charge (load_profiles → CAT.legacy) : les I et H du catalogue
FAMILLES_CHARGEES = ("IPE", "IPE A", "IPE O", "IPN", "HEA", "HEB", "HEM", "HEAA", "HL", "HD", "HP")
E_ACIER = 210000.0          # N/mm², la valeur par défaut d'ec3_buckling_chi (le module ne la passe jamais)


# ------------------------------------------------------------------ aides du module
def lire_cle(d: Dict, *keys, default=None):
    """``pick`` : la première clé présente et renseignée (ni None, ni "None", ni "")."""
    for k in keys:
        if k in d and d[k] not in (None, "None", ""):
            return d[k]
    return default


def en_reel(x, default=None):
    """``to_float`` : un réel, ou ``default`` si la valeur manque ou ne se lit pas."""
    try:
        if x in (None, "None", ""):
            return default
        return float(x)
    except Exception:  # noqa: BLE001 — le module avale toute erreur de lecture
        return default


def fmt_sans_zeros(x, digits: int = 3) -> str:
    """``fmt_no_trailing_zeros`` : le format des tableaux de l'écran (« 21.238 », « 990 », « — »)."""
    if x is None:
        return "—"
    try:
        xf = float(x)
    except Exception:  # noqa: BLE001
        return str(x)
    if xf.is_integer():
        return str(int(round(xf)))
    return f"{xf:.{digits}f}".rstrip("0").rstrip(".")


# ------------------------------------------------------------------ les profilés
def nettoyer(raw: Dict[str, Dict]) -> Dict[str, Dict]:
    """La boucle de ``load_profiles`` : chaque enregistrement « ancien » du catalogue relu champ par champ ; un profilé
    sans W_el,y > 0 est écarté. L'ordre est celui du catalogue."""
    cleaned = {}
    for name, p in raw.items():
        # Dimensions (mm)
        h = en_reel(lire_cle(p, "h"))
        b = en_reel(lire_cle(p, "b"), default=0.0)
        tw = en_reel(lire_cle(p, "tw"), default=0.0)
        tf = en_reel(lire_cle(p, "tf"), default=0.0)
        r = en_reel(lire_cle(p, "r"), default=0.0)
        # Aires en cm² ; inerties en cm⁴ ; modules en cm³
        A = en_reel(lire_cle(p, "A"), default=None)
        Avy = en_reel(lire_cle(p, "Avy"), default=None)        # absent de la forme ancienne : toujours None
        Avz = en_reel(lire_cle(p, "Avz"), default=None)
        Iy = en_reel(lire_cle(p, "Iy", "Iv"), default=None)
        Iz = en_reel(lire_cle(p, "Iz"), default=None)
        Wel_y = en_reel(lire_cle(p, "Wely", "Wel", "Wel_y"), default=None)
        Wel_z = en_reel(lire_cle(p, "Welz", "Wel_z"), default=None)
        It = en_reel(lire_cle(p, "It"), default=None)
        Iw = en_reel(lire_cle(p, "Iw"), default=None)
        poids = en_reel(lire_cle(p, "Poids", "masse"), default=None)
        typ = str(lire_cle(p, "type", default=""))
        if Wel_y is None or Wel_y <= 0:
            continue
        cleaned[name] = {"type": typ, "h": h, "b": b, "tw": tw, "tf": tf, "r": r, "A": A, "Avy": Avy, "Avz": Avz,
                         "Iy": Iy, "Iz": Iz, "Wel_y": Wel_y, "Wel_z": Wel_z, "It": It, "Iw": Iw, "Poids": poids}
    return cleaned


@lru_cache(maxsize=1)
def _profils() -> Dict[str, Dict]:
    from acier.profiles import catalogue as CAT      # la source unique : acier/donnees/profiles.json
    return nettoyer(CAT.legacy(FAMILLES_CHARGEES))


def profils() -> Dict[str, Dict]:
    """``load_profiles`` : les profilés que l'écran propose, {nom: enregistrement}, dans l'ordre du catalogue. Une
    COPIE : la modifier ne change pas le calcul."""
    return {k: dict(v) for k, v in _profils().items()}


def profil(nom: str) -> Dict:
    """Un profilé de ``profils()`` (copie) ; KeyError s'il n'y est pas."""
    return dict(_profils()[nom])


def noms() -> list:
    return list(_profils())


# ------------------------------------------------------------------ résistances et taux de section
def resistances_section(profile: Dict, fy, gamma_M0=1.0) -> Dict[str, Optional[float]]:
    """``ec3_section_resistance`` : résistances ÉLASTIQUES de section (N, N·mm), None quand la donnée manque."""
    A_mm2 = profile["A"] * 1e2 if profile.get("A") else None
    Wy_mm3 = profile["Wel_y"] * 1e3 if profile.get("Wel_y") else None
    Wz_mm3 = profile["Wel_z"] * 1e3 if profile.get("Wel_z") else None
    Avy_mm2 = profile["Avy"] * 1e2 if profile.get("Avy") else None
    Avz_mm2 = profile["Avz"] * 1e2 if profile.get("Avz") else None
    N_Rd = (A_mm2 * fy / gamma_M0) if A_mm2 else None
    My_Rd = (Wy_mm3 * fy / gamma_M0) if Wy_mm3 else None
    Mz_Rd = (Wz_mm3 * fy / gamma_M0) if Wz_mm3 else None
    Vy_Rd = (Avy_mm2 * fy / (math.sqrt(3) * gamma_M0)) if Avy_mm2 else None
    Vz_Rd = (Avz_mm2 * fy / (math.sqrt(3) * gamma_M0)) if Avz_mm2 else None
    return {"N_Rd_N": N_Rd, "My_Rd_Nmm": My_Rd, "Mz_Rd_Nmm": Mz_Rd, "Vy_Rd_N": Vy_Rd, "Vz_Rd_N": Vz_Rd}


def taux_section(NEd_kN, My_kNm, Mz_kNm, Vy_kN, Vz_kN, resist: Dict) -> Dict[str, float]:
    """``utilisation_ratio_section`` : taux de chaque effort, interaction LINÉAIRE N + M (rN SANS valeur absolue :
    une traction le rend négatif), et le taux de la section max(η_NM ; η_Vy ; η_Vz). Une résistance absente donne 0."""
    NEd = NEd_kN * 1e3
    My = My_kNm * 1e6
    Mz = Mz_kNm * 1e6
    Vy = Vy_kN * 1e3
    Vz = Vz_kN * 1e3
    rN = (NEd / resist["N_Rd_N"]) if (resist.get("N_Rd_N") and resist["N_Rd_N"] > 0) else 0.0
    rMy = (abs(My) / resist["My_Rd_Nmm"]) if (resist.get("My_Rd_Nmm") and resist["My_Rd_Nmm"] > 0) else 0.0
    rMz = (abs(Mz) / resist["Mz_Rd_Nmm"]) if (resist.get("Mz_Rd_Nmm") and resist["Mz_Rd_Nmm"] > 0) else 0.0
    rVy = (abs(Vy) / resist["Vy_Rd_N"]) if (resist.get("Vy_Rd_N") and resist["Vy_Rd_N"] > 0) else 0.0
    rVz = (abs(Vz) / resist["Vz_Rd_N"]) if (resist.get("Vz_Rd_N") and resist["Vz_Rd_N"] > 0) else 0.0
    interaction_NM = rN + rMy + rMz
    interaction = max(interaction_NM, rVy, rVz)
    return {"rN": rN, "rMy": rMy, "rMz": rMz, "rVy": rVy, "rVz": rVz, "util": interaction, "util_NM": interaction_NM}


# ------------------------------------------------------------------ flambement
def chi_courbe(lambdabar: float, alpha: float):
    """φ et χ d'une courbe de flambement (les deux lignes d'``ec3_buckling_chi``, mot pour mot) : (φ, χ ≤ 1).
    Le dessin des courbes χ(λ̄) passe par CETTE fonction : une seule expression."""
    phi = 0.5 * (1.0 + alpha * (lambdabar - 0.2) + lambdabar**2)
    chi = 1.0 / (phi + math.sqrt(max(phi**2 - lambdabar**2, 0.0)))
    chi = min(chi, 1.0)
    return phi, chi


def flambement_chi(A_cm2, I_cm4, Lcr_m, fy, E=E_ACIER, alpha=0.34) -> Optional[Dict[str, float]]:
    """``ec3_buckling_chi`` : N_cr (N), λ̄, χ pour un axe ; None si A, I manquent ou L_cr ≤ 0. Le moteur rend aussi
    φ (clé ``phi``, ajoutée pour la note : même expression, même ordre)."""
    if A_cm2 is None or I_cm4 is None or Lcr_m <= 0:
        return None
    A_mm2 = A_cm2 * 1e2
    I_mm4 = I_cm4 * 1e4
    L_mm = Lcr_m * 1e3
    Ncr = (math.pi**2 * E * I_mm4) / (L_mm**2)
    lambdabar = math.sqrt((A_mm2 * fy) / Ncr)
    phi, chi = chi_courbe(lambdabar, alpha)
    return {"chi": chi, "lambdabar": lambdabar, "Ncr_N": Ncr, "phi": phi}

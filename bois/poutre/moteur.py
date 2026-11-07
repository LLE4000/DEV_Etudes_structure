# -*- coding: utf-8 -*-
"""Poutre bois — le CALCUL de l'écran Streamlit ``modules/poutre_bois.py``, sans Streamlit ni numpy.

Chaque grandeur reprend l'expression du module DANS LE MÊME ORDRE D'OPÉRATIONS : les flottants sont identiques au bit
près (``tests/test_poutre_bois_moteur.py`` le prouve sur plus de 700 poutres, l'écran rejoué avec un faux ``st``).
Les écarts du module avec l'EC5 sont REPRODUITS (parité) et NOMMÉS (``ECARTS``) ; ils sont discutés dans
``docs/bois/POUTRE_BOIS_MOTEUR.md``.

    from bois.poutre import moteur as M
    r = M.verifier(M.nouvelle_poutre())

ENTRÉE (une poutre ; mm, m, kN/m ; une clé absente prend le défaut de l'écran — voir ``entree``) ::

    {"nom": "P1", "classe": "C24", "service": 1, "duree": "long",
     "standard": true, "section": "45x220", "b": 80.0, "h": 240.0,       # b, h : la section sur mesure (standard: false)
     "L": 4.0, "G": 2.0, "Q": 3.0, "gamma_G": 1.35, "gamma_Q": 1.5, "psi2": 0.3,
     "lim_inst": "L/300", "lim_fin": "L/200", "appui": false, "a": 60.0}

SORTIE de ``verifier`` (JSON strict, nombres non arrondis — le site formate) :

- ``reference``, ``poutre`` (normalisée) ;
- ``materiau`` : ``classe`` et les neuf valeurs de la base (``fm_k`` … ``rho_k``) ;
- ``coefficients`` : ``service``, ``duree``, ``kmod``, ``kdef``, ``kh``, ``gamma_M`` ;
- ``geometrie`` : ``b``, ``h`` (mm ; entiers pour une section standard, comme le module), ``tag`` (« 45x220 », ou
  None), ``standard``, ``A`` (mm²), ``I`` (mm⁴), ``W`` (mm³), ``L`` (m), ``L_mm`` ;
- ``charges`` : ``G``, ``Q``, ``gamma_G``, ``gamma_Q``, ``psi2``, ``q_ELU``, ``q_inst`` (G + Q), ``q_qp`` (G + ψ₂Q) ;
- ``grandeurs`` : toutes les grandeurs du module, à plat : ``A``, ``I``, ``W``, ``kh``, ``kmod``, ``kdef``, ``fm_d``,
  ``fv_d``, ``fc90_d``, ``q_ELU``, ``M_Ed``, ``V_Ed``, ``q_inst``, ``q_qp``, ``sigma_m_Ed``, ``tau_Ed``, ``w_inst``,
  ``w_qp`` (flèche élastique sous G + ψ₂Q), ``w_fin``, ``L_mm``, ``lim_inst_mm``, ``lim_fin_mm``, ``R_appui``,
  ``sigma_c90_Ed``, ``u_flex``, ``u_shear``, ``u_c90``, ``u_wi``, ``u_wf`` (les deux grandeurs d'appui valent None
  quand l'option est éteinte) ;
- ``verifications`` : dans l'ordre de l'écran — ``flexion``, ``cisaillement``, ``appui`` (si l'option est cochée),
  ``fleche_inst``, ``fleche_fin`` ; chacune ``{cle, nom, etat ("ok" ≤ 100 %, "nok" sinon : le st.success /
  st.error de l'écran), valeur, limite, unite, taux (= valeur / limite, l'expression du module), formules}`` ;
  une formule : ``{cle, termes: [{symbole, valeur, unite}], resultat, unite}`` ;
- ``taux_max``, ``statut`` (``"ok"`` ou ``"nok"``), ``gouvernant`` (``{cle, nom, taux}``, la première au taux
  maximal), ``ecarts`` (les codes de ``ECARTS`` qui jouent pour cette poutre), ``indicatif`` — des grandeurs que
  l'écran ne calcule pas, données À TITRE INDICATIF, sans effet sur les vérifications : ``masse_lineique`` kg/m,
  ``poids_propre`` kN/m (ρ_mean·A — l'écran affiche ρ mais ne le compte pas), ``masse`` kg, et ``combinaison_G`` (la
  combinaison γ_G·G seule avec k_mod permanent, que l'écran ne fait pas : ``{q, kmod, sigma_m_Ed, fm_d, u_flex,
  tau_Ed, fv_d, u_shear}``, ou None quand G = 0 ou que la durée choisie est déjà « permanent »), ``w_inst_unite`` et
  ``w_fin_unite`` (les deux flèches de l'écran divisées par 1 000 : sans l'écart d'unité « fleche_unite »).

``candidats_sections(poutre)`` : les sections standard de la liste, chacune calculée (taux maximal), triées par aire
(donc par masse), la retenue et la plus légère qui tient marquées.
"""
from typing import Dict, List, Optional

from . import donnees as D
from .donnees import CLASSES, GAMMA_M, KDEF, KMOD, REFERENCE  # noqa: F401 — REFERENCE : l'API
from .entree import normaliser, nouvelle_poutre, section_standard  # noqa: F401 — l'API

OK, NOK = "ok", "nok"
G_PESANTEUR = 9.81      # m/s² — seulement pour le poids propre INDICATIF (l'écran ne le compte pas)

# Les écarts du module avec l'EC5 (NF EN 1995-1-1), reproduits pour la parité. « sens » : ce que l'écart fait au
# résultat. La note du site les rappelle ; la doc les chiffre.
ECARTS = {
    "fleche_unite": ("la flèche est calculée avec q·1 000 alors que q est déjà en N/mm (1 kN/m = 1 N/mm) : les deux "
                     "flèches de l'écran sont 1 000 fois trop grandes", "côté sûr, mais les vérifications de flèche sont faussées"),
    "kh_reduit": ("k_h = (150/h)^0,2 est appliqué aussi au-dessus de 150 mm (k_h < 1) ; l'EC5 §3.2 le limite aux "
                  "hauteurs inférieures à 150 mm (k_h = 1 au-delà)", "côté sûr"),
    "cisaillement_kcr": ("τ calculé sur la largeur b, sans k_cr = 0,67 (EC5 §6.1.7, amendement A1)", "non conservatif"),
    "appui_els": ("compression à l'appui sous la réaction G + Q NON pondérée, sans k_c,90 ni longueur efficace",
                  "non conservatif à l'ELU"),
    "fleche_finale": ("w_fin = (1 + k_def)·w(G + ψ₂Q) : il manque (1 − ψ₂)·w_inst,Q par rapport à l'EC5 §2.2.3 "
                      "(w_fin = w_inst,G(1 + k_def) + w_inst,Q(1 + ψ₂·k_def))", "non conservatif"),
    "combinaison_G": ("seule la combinaison γ_G·G + γ_Q·Q est vérifiée, avec le k_mod de la durée choisie ; la combinaison "
                      "γ_G·G seule, avec k_mod permanent, n'est pas faite (EC5 §3.1.3 : chaque combinaison avec son k_mod) : "
                      "elle gouverne dès que G domine Q", "non conservatif"),
    "poids_propre": ("le poids propre de la poutre n'est pas compté : l'inclure dans G", "non conservatif"),
    "deversement": ("ni déversement (k_crit), ni effort tranchant dans la flèche (G_mean inutilisé)", "non vérifié"),
}


# ------------------------------------------------------------------ les fonctions du module (mêmes expressions)
def section_rect(b_mm, h_mm):
    """sect_rect : A (mm²), I (mm⁴), W (mm³) d'un rectangle."""
    A = b_mm*h_mm
    I = b_mm*(h_mm**3)/12.0
    W = I/(h_mm/2.0)
    return A, I, W


def fleche_isostatique(q_kN_m, L_m, E_MPa, I_mm4):
    """defl_simply : flèche à mi-portée d'une poutre sur deux appuis sous q uniforme (mm)."""
    return (5.0 * (q_kN_m*1000.0) * (L_m*1000.0)**4) / (384.0 * E_MPa * I_mm4)


def tau_rect(V_kN, b_mm, h_mm):
    """shear_tau_rect : τ ≈ 1,5·V/A (MPa)."""
    A = b_mm*h_mm
    return 1.5 * (V_kN*1000.0) / A


def kh_hauteur(h_mm):
    """kh_depth : k_h = (150/h)^0,2 borné entre 0,6 et 1,3."""
    kh = (D.H_REFERENCE/max(h_mm, 1.0))**0.2
    return min(max(kh, D.KH_BORNES[0]), D.KH_BORNES[1])


def diviseur_limite(lim: str) -> float:
    """« L/300 » → 300.0 (l'écran : float(lim.split("/")[1]))."""
    return float(lim.split("/")[1])


def _t(symbole: str, valeur, unite: str = "") -> Dict:
    return {"symbole": symbole, "valeur": valeur, "unite": unite}


def _f(cle: str, termes: List[Dict], resultat, unite: str) -> Dict:
    return {"cle": cle, "termes": termes, "resultat": resultat, "unite": unite}


def _taux(valeur, limite):
    """u = valeur / limite si limite > 0 (l'écran : np.nan sinon, ce que le domaine de l'écran n'atteint pas ; le
    moteur rend alors None, JSON strict)."""
    return valeur / limite if limite > 0 else None


# ------------------------------------------------------------------ le calcul
def verifier(poutre: Dict) -> Dict:
    """Le calcul de l'écran pour une poutre (voir le format en tête du fichier)."""
    p = normaliser(poutre)
    mat = CLASSES[p["classe"]]
    sc, duration = p["service"], p["duree"]
    kmod = KMOD[sc][duration]
    kdef = KDEF[sc]
    if p["standard"]:
        sel = section_standard(p["section"])
        b_mm, h_mm = sel["b"], sel["h"]
    else:
        b_mm, h_mm = p["b"], p["h"]
    L_m = p["L"]
    qG, qQ, gammaG, gammaQ, psi2 = p["G"], p["Q"], p["gamma_G"], p["gamma_Q"], p["psi2"]

    A, I, W = section_rect(b_mm, h_mm)
    kh = kh_hauteur(h_mm)
    # résistances de calcul
    fm_d = kmod * mat["fm_k"] * kh / GAMMA_M
    fv_d = kmod * mat["fv_k"] / GAMMA_M
    fc90_d = kmod * mat["fc90_k"] / GAMMA_M if mat.get("fc90_k") else None
    # efforts, poutre sur deux appuis
    q_ELU = gammaG*qG + gammaQ*qQ
    M_ELU = q_ELU * L_m**2 / 8.0
    V_ELU = q_ELU * L_m / 2.0
    q_SLS_inst = qG + qQ
    q_SLS_quasi = qG + psi2*qQ
    # contraintes ELU
    sigma_m_Ed = (M_ELU*1e6) / W
    tau_Ed = tau_rect(V_ELU, b_mm, h_mm)
    # flèches
    E = mat["E0_mean"]
    w_inst = fleche_isostatique(q_SLS_inst, L_m, E, I)
    w_qp = fleche_isostatique(q_SLS_quasi, L_m, E, I)
    w_fin = (1.0 + kdef) * w_qp
    # limites ELS
    L_mm = L_m*1000.0
    n_inst, n_fin = diviseur_limite(p["lim_inst"]), diviseur_limite(p["lim_fin"])
    lim_inst_mm = L_mm / n_inst
    lim_fin_mm = L_mm / n_fin
    # compression perpendiculaire (option)
    R_sup = sigma_c90_Ed = u_c90 = None
    if p["appui"] and fc90_d:
        R_sup = (q_SLS_inst * L_m) / 2.0
        sigma_c90_Ed = (R_sup*1000.0) / (b_mm*p["a"])
        u_c90 = sigma_c90_Ed / fc90_d
    # taux d'utilisation
    u_flex = _taux(sigma_m_Ed, fm_d)
    u_shear = _taux(tau_Ed, fv_d)
    u_wi = _taux(w_inst, lim_inst_mm)
    u_wf = _taux(w_fin, lim_fin_mm)
    # la combinaison γ_G·G seule, avec k_mod permanent : l'écran ne la fait pas (écart « combinaison_G »). Calculée À
    # TITRE INDICATIF (flexion et cisaillement, mêmes formules que l'écran) : elle ne change ni les vérifications, ni
    # le verdict, ni la parité. Sans objet quand G = 0 ou quand la durée choisie est déjà « permanent ».
    km_p = KMOD[sc]["permanent"]
    comb_G = None
    if qG > 0 and kmod > km_p:
        q_G = gammaG*qG
        sig_G = (q_G * L_m**2 / 8.0 * 1e6) / W
        tau_G = tau_rect(q_G * L_m / 2.0, b_mm, h_mm)
        fmd_G = km_p * mat["fm_k"] * kh / GAMMA_M
        fvd_G = km_p * mat["fv_k"] / GAMMA_M
        comb_G = {"q": q_G, "kmod": km_p, "sigma_m_Ed": sig_G, "fm_d": fmd_G, "u_flex": sig_G / fmd_G,
                  "tau_Ed": tau_G, "fv_d": fvd_G, "u_shear": tau_G / fvd_G}

    # ---- les formules du détail (chaque résultat se retrouve à partir de ses seuls termes)
    f_q = _f("q_ELU", [_t("gamma_G", gammaG), _t("G", qG, "kN/m"), _t("gamma_Q", gammaQ), _t("Q", qQ, "kN/m")], q_ELU, "kN/m")
    f_kmod = _f("kmod", [_t("service", sc), _t("duree", duration)], kmod, "")
    f_kh = _f("kh", [_t("h", h_mm, "mm")], kh, "")
    f_I = _f("I", [_t("b", b_mm, "mm"), _t("h", h_mm, "mm")], I, "mm⁴")
    f_W = _f("W", [_t("I", I, "mm⁴"), _t("h", h_mm, "mm")], W, "mm³")
    verifs = [
        {"cle": "flexion", "nom": "Flexion", "valeur": sigma_m_Ed, "limite": fm_d, "unite": "MPa", "taux": u_flex,
         "formules": [f_q, _f("M_Ed", [_t("q_ELU", q_ELU, "kN/m"), _t("L", L_m, "m")], M_ELU, "kN·m"), f_I, f_W,
                      _f("sigma_m_Ed", [_t("M_Ed", M_ELU, "kN·m"), _t("W", W, "mm³")], sigma_m_Ed, "MPa"), f_kmod, f_kh,
                      _f("fm_d", [_t("kmod", kmod), _t("fm_k", mat["fm_k"], "MPa"), _t("kh", kh), _t("gamma_M", GAMMA_M)], fm_d, "MPa"),
                      _f("u", [_t("valeur", sigma_m_Ed, "MPa"), _t("limite", fm_d, "MPa")], u_flex, "")]},
        {"cle": "cisaillement", "nom": "Cisaillement", "valeur": tau_Ed, "limite": fv_d, "unite": "MPa", "taux": u_shear,
         "formules": [_f("V_Ed", [_t("q_ELU", q_ELU, "kN/m"), _t("L", L_m, "m")], V_ELU, "kN"),
                      _f("tau_Ed", [_t("V_Ed", V_ELU, "kN"), _t("b", b_mm, "mm"), _t("h", h_mm, "mm")], tau_Ed, "MPa"),
                      _f("fv_d", [_t("kmod", kmod), _t("fv_k", mat["fv_k"], "MPa"), _t("gamma_M", GAMMA_M)], fv_d, "MPa"),
                      _f("u", [_t("valeur", tau_Ed, "MPa"), _t("limite", fv_d, "MPa")], u_shear, "")]},
    ]
    if u_c90 is not None:
        verifs.append(
            {"cle": "appui", "nom": "Compression ⟂ fil (appui)", "valeur": sigma_c90_Ed, "limite": fc90_d, "unite": "MPa", "taux": u_c90,
             "formules": [_f("R_appui", [_t("q_inst", q_SLS_inst, "kN/m"), _t("L", L_m, "m")], R_sup, "kN"),
                          _f("sigma_c90_Ed", [_t("R_appui", R_sup, "kN"), _t("b", b_mm, "mm"), _t("a", p["a"], "mm")], sigma_c90_Ed, "MPa"),
                          _f("fc90_d", [_t("kmod", kmod), _t("fc90_k", mat["fc90_k"], "MPa"), _t("gamma_M", GAMMA_M)], fc90_d, "MPa"),
                          _f("u", [_t("valeur", sigma_c90_Ed, "MPa"), _t("limite", fc90_d, "MPa")], u_c90, "")]})
    verifs += [
        {"cle": "fleche_inst", "nom": "Flèche instantanée", "valeur": w_inst, "limite": lim_inst_mm, "unite": "mm", "taux": u_wi,
         "limite_texte": p["lim_inst"],
         "formules": [_f("q_inst", [_t("G", qG, "kN/m"), _t("Q", qQ, "kN/m")], q_SLS_inst, "kN/m"), f_I,
                      _f("w_inst", [_t("q", q_SLS_inst, "kN/m"), _t("L", L_m, "m"), _t("E", E, "MPa"), _t("I", I, "mm⁴")], w_inst, "mm"),
                      _f("lim_inst", [_t("L", L_mm, "mm"), _t("n", n_inst)], lim_inst_mm, "mm"),
                      _f("u", [_t("valeur", w_inst, "mm"), _t("limite", lim_inst_mm, "mm")], u_wi, "")]},
        {"cle": "fleche_fin", "nom": "Flèche finale", "valeur": w_fin, "limite": lim_fin_mm, "unite": "mm", "taux": u_wf,
         "limite_texte": p["lim_fin"],
         "formules": [_f("q_qp", [_t("G", qG, "kN/m"), _t("psi2", psi2), _t("Q", qQ, "kN/m")], q_SLS_quasi, "kN/m"),
                      _f("w_qp", [_t("q", q_SLS_quasi, "kN/m"), _t("L", L_m, "m"), _t("E", E, "MPa"), _t("I", I, "mm⁴")], w_qp, "mm"),
                      _f("kdef", [_t("service", sc)], kdef, ""),
                      _f("w_fin", [_t("kdef", kdef), _t("w_qp", w_qp, "mm")], w_fin, "mm"),
                      _f("lim_fin", [_t("L", L_mm, "mm"), _t("n", n_fin)], lim_fin_mm, "mm"),
                      _f("u", [_t("valeur", w_fin, "mm"), _t("limite", lim_fin_mm, "mm")], u_wf, "")]},
    ]
    for v in verifs:
        # l'écran : (st.success if u <= 1.0 else st.error) ; un taux indéfini (np.nan) tombe dans st.error
        v["etat"] = OK if v["taux"] is not None and v["taux"] <= 1.0 else NOK
    taux = [v["taux"] for v in verifs if v["taux"] is not None]
    taux_max = max(taux) if taux else None
    gov = next((v for v in verifs if v["taux"] == taux_max), verifs[0])
    ecarts = ["fleche_unite"] + (["kh_reduit"] if kh < 1.0 else []) + ["cisaillement_kcr"] + (["combinaison_G"] if comb_G else []) \
        + (["appui_els"] if u_c90 is not None else []) + ["fleche_finale", "poids_propre", "deversement"]
    masse = mat["rho_mean"] * A * 1e-6                      # kg/m (ρ en kg/m³, A en mm²)
    return {
        "reference": REFERENCE, "poutre": p,
        "materiau": dict(mat, classe=p["classe"]),
        "coefficients": {"service": sc, "duree": duration, "kmod": kmod, "kdef": kdef, "kh": kh, "gamma_M": GAMMA_M},
        "geometrie": {"b": b_mm, "h": h_mm, "tag": p["section"] if p["standard"] else None, "standard": p["standard"],
                      "A": A, "I": I, "W": W, "L": L_m, "L_mm": L_mm},
        "charges": {"G": qG, "Q": qQ, "gamma_G": gammaG, "gamma_Q": gammaQ, "psi2": psi2, "q_ELU": q_ELU,
                    "q_inst": q_SLS_inst, "q_qp": q_SLS_quasi},
        "grandeurs": {"A": A, "I": I, "W": W, "kh": kh, "kmod": kmod, "kdef": kdef, "fm_d": fm_d, "fv_d": fv_d, "fc90_d": fc90_d,
                      "q_ELU": q_ELU, "M_Ed": M_ELU, "V_Ed": V_ELU, "q_inst": q_SLS_inst, "q_qp": q_SLS_quasi,
                      "sigma_m_Ed": sigma_m_Ed, "tau_Ed": tau_Ed, "w_inst": w_inst, "w_qp": w_qp, "w_fin": w_fin, "L_mm": L_mm,
                      "lim_inst_mm": lim_inst_mm, "lim_fin_mm": lim_fin_mm, "R_appui": R_sup, "sigma_c90_Ed": sigma_c90_Ed,
                      "u_flex": u_flex, "u_shear": u_shear, "u_c90": u_c90, "u_wi": u_wi, "u_wf": u_wf},
        "verifications": verifs,
        "taux_max": taux_max,
        "statut": OK if all(v["etat"] == OK for v in verifs) else NOK,
        "gouvernant": {"cle": gov["cle"], "nom": gov["nom"], "taux": gov["taux"]},
        "ecarts": ecarts,
        "indicatif": {"masse_lineique": masse, "poids_propre": masse * G_PESANTEUR / 1000.0, "masse": masse * L_m,
                      "combinaison_G": comb_G,
                      # les flèches de l'écran sans l'écart « fleche_unite » (q en N/mm) : affichées, jamais retenues
                      "w_inst_unite": w_inst / 1000.0, "w_fin_unite": w_fin / 1000.0},
    }


def candidats_sections(poutre: Dict) -> List[Dict]:
    """Les sections standard de la liste de l'écran, chacune calculée avec le reste de la poutre : ``{nom, b, h, A,
    masse (kg/m), taux (le maximal), gouvernant, ok, retenue, optimal}``, triées par aire croissante (l'ordre de la
    liste départage). ``optimal`` : la plus légère qui tient (taux ≤ 100 %) ; ``retenue`` : la section standard
    choisie (aucune quand la section est sur mesure)."""
    p = normaliser(poutre)
    out = []
    for k, s in enumerate(D.SECTIONS_STD):
        r = verifier(dict(p, standard=True, section=s["tag"]))
        out.append({"nom": s["tag"], "b": s["b"], "h": s["h"], "A": r["geometrie"]["A"], "masse": r["indicatif"]["masse_lineique"],
                    "I": r["geometrie"]["I"], "W": r["geometrie"]["W"], "taux": r["taux_max"], "gouvernant": r["gouvernant"]["nom"],
                    "ok": r["statut"] == OK, "retenue": p["standard"] and s["tag"] == p["section"], "optimal": False, "_k": k})
    out.sort(key=lambda c: (c["A"], c["_k"]))
    best = next((c for c in out if c["ok"]), None)
    for c in out:
        c["optimal"] = c is best
        del c["_k"]
    return out


def optimal(poutre: Dict) -> Optional[Dict]:
    """La section standard la plus légère qui tient (None si aucune)."""
    return next((c for c in candidats_sections(poutre) if c["optimal"]), None)

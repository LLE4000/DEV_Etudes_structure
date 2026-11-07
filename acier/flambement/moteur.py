# -*- coding: utf-8 -*-
"""La vérification d'une barre comprimée au flambement par flexion (EN 1993-1-1 §6.3.1) et la résistance de sa
section (§6.2.4), et la liste des profilés candidats (le sélecteur, trié par masse, le plus léger qui tient).

    N_cr = π²·E·I / L_cr²        λ̄ = √(A·f_y / N_cr)          (A_eff en classe 4)
    φ = 0,5·[1 + α·(λ̄ − 0,2) + λ̄²]     χ = 1 / (φ + √(φ² − λ̄²)) ≤ 1
    N_b,Rd = χ·A·f_y / γ_M1       N_c,Rd = A·f_y / γ_M0       η = N_Ed / N_Rd

Unités du calcul : N, mm (comme ec3_buckling_chi) ; rendues en kN, m, cm, cm², cm⁴ pour l'écran. Le moteur CALCULE et
NOMME (la règle du Tableau 6.2 retenue, la classe, les parois réduites) ; l'écran compose et formate.
"""
from typing import Dict, List, Optional, Sequence

from . import entree as EN
from . import normes as NO
from . import section as SE

OK, NOK, IMPOSSIBLE = "conforme", "non conforme", "impossible"
NOUVEAU = "Nouveau calcul EN 1993-1-1 §6.3.1, à valider par le bureau"
LIMITE_NEGLIGEABLE = 0.2       # §6.3.1.2(4) : λ̄ ≤ 0,2 — les effets du flambement peuvent être négligés


def _section(p: Dict, nuance: str) -> Dict:
    """Matériau, classe et aire efficace de la section en compression pure."""
    fab = SE.FABRICATION[p["famille"]]
    t = SE.epaisseur(p)
    fy = NO.limite_elastique(nuance, t, froid=fab == "creux_froid")
    eps = NO.epsilon(fy)
    parois = []
    for pa in SE.parois(p):
        d = dict(pa)
        if pa["type"] == "console" and fab == "L":
            d["classe"] = 0          # posée plus bas, pour la cornière entière
        else:
            d["classe"] = NO.classe_paroi(pa["type"], pa["c_t"], eps)
        parois.append(d)
    if fab == "L":
        h_t = p["h"] / p["t"]
        bh_2t = (p["b"] + p["h"]) / (2.0 * p["t"])
        classe = NO.classe_corniere(h_t, bh_2t, eps)
        for d in parois:
            d["classe"] = classe
    else:
        h_t = bh_2t = None
        classe = max(d["classe"] for d in parois)
    A = p["A"] * 1e2                                       # mm²
    A_eff = A
    if classe == 4:
        if p["famille"].startswith("CHS"):
            raise NO.HorsDomaine(f"tube rond de classe 4 (d/t = {p['cf_tf']:.1f} > 90·ε²) : EN 1993-1-6, hors du champ du module".replace(".", ",", 1))
        for d in parois:
            d["lambda_p"] = NO.elancement_paroi(d["type"], d["c_t"], eps)
            d["rho"] = NO.rho(d["type"], d["lambda_p"])
            d["retrait"] = d["n"] * (1.0 - d["rho"]) * d["c"] * d["t"]      # mm²
        A_eff = A - sum(d["retrait"] for d in parois)
    return {"fabrication": fab, "t": t, "fy": fy, "epsilon": eps, "parois": parois, "classe": classe, "A": A, "A_eff": A_eff,
            "h_t": h_t, "bh_2t": bh_2t}


def _axe(cle: str, ax_sec: Dict, ax_in: Dict, L: float, A_calc: float, fy: float, alpha: float, courbe: str, regle: str,
         N: float, gamma_M1: float) -> Dict:
    """Un axe : N_cr, λ̄, φ, χ, N_b,Rd, η."""
    Lcr = EN.longueur_flambement(ax_in, L)                # m
    I_mm4 = ax_sec["I"] * 1e4
    L_mm = Lcr * 1e3
    Ncr, lb = NO.euler(A_calc, I_mm4, L_mm, fy)
    phi, chi = NO.phi_chi(lb, alpha)
    NbRd = chi * A_calc * fy / gamma_M1                    # N
    taux = (N * 1e3) / NbRd
    return {"cle": cle, "nom": ax_sec["nom"], "appuis": ax_in["appuis"], "maintiens": ax_in["maintiens"],
            "k": EN.coefficient(ax_in, L), "Lcr": Lcr, "I": ax_sec["I"], "i": ax_sec["i"], "Ncr": Ncr / 1e3,
            "lambda": Lcr * 1e2 / ax_sec["i"], "lambda1": NO.lambda_1(fy), "lambdabar": lb, "courbe": courbe, "alpha": alpha,
            "regle": regle, "phi": phi, "chi": chi, "NbRd": NbRd / 1e3, "taux": taux, "negligeable": lb <= LIMITE_NEGLIGEABLE,
            "Ncr_N": Ncr, "NbRd_N": NbRd}


def verifier(barre: Dict) -> Dict:
    """La barre vérifiée ; ``statut`` IMPOSSIBLE (et ``motif``) quand la norme ne couvre pas le profilé."""
    b = EN.normaliser(barre)
    p = SE.profil(b["profil"])
    out = {"nom": b["nom"], "profil": p["nom"], "famille": p["famille"], "nuance": b["nuance"], "N": b["N"], "L": b["L"],
           "gamma_M0": b["gamma_M0"], "gamma_M1": b["gamma_M1"], "E": NO.E_ACIER, "masse_lineique": p["masse"],
           "masse": p["masse"] * b["L"], "avertissements": [NOUVEAU]}
    try:
        sec = _section(p, b["nuance"])
        fab = sec["fabrication"]
        cy, cz, regle = NO.courbes(fab, b["nuance"], p.get("h"), p.get("b"), p.get("tf"))
    except NO.HorsDomaine as e:
        out.update({"statut": IMPOSSIBLE, "motif": str(e), "taux_max": None, "gouvernant": None, "verifications": []})
        return out
    A_calc = sec["A_eff"]
    axes_sec = SE.axes(p)
    axes = {}
    for cle, courbe in (("y", cy), ("z", cz)):
        axes[cle] = _axe(cle, axes_sec[cle], b["axes"][cle], b["L"], A_calc, sec["fy"], NO.ALPHA[courbe], courbe, regle,
                         b["N"], b["gamma_M1"])
    NcRd = A_calc * sec["fy"] / b["gamma_M0"] / 1e3       # kN
    t_sec = b["N"] / NcRd
    verifs = [{"cle": "y", "nom": f"Flambement autour de {axes['y']['nom']}", "taux": axes["y"]["taux"], "valeur": b["N"], "limite": axes["y"]["NbRd"]},
              {"cle": "z", "nom": f"Flambement autour de {axes['z']['nom']}", "taux": axes["z"]["taux"], "valeur": b["N"], "limite": axes["z"]["NbRd"]},
              {"cle": "section", "nom": "Résistance de la section", "taux": t_sec, "valeur": b["N"], "limite": NcRd}]
    gov = max(verifs, key=lambda v: v["taux"])
    sec.update({"NcRd": NcRd})
    out.update({"statut": OK if gov["taux"] <= 1.0 else NOK, "motif": "", "section": sec, "axes": axes, "verifications": verifs,
                "taux_max": gov["taux"], "gouvernant": gov["cle"], "courbes": {"y": cy, "z": cz, "regle": regle}})
    av = out["avertissements"]
    reduite = sec["classe"] == 4 and sec["A_eff"] < sec["A"]
    if reduite:
        av.append("Section de classe 4 en compression : l'aire efficace A_eff (EN 1993-1-5 §4.4) remplace A dans λ̄, N_b,Rd et N_c,Rd.")
    elif sec["classe"] == 4:
        av.append("Section de classe 4 au sens du Tableau 5.2, mais aucune paroi n'est réduite par l'EN 1993-1-5 §4.4 (ρ = 1) : A_eff = A.")
    if reduite:
        if not SE.symetrique(p):
            av.append("Classe 4 et section non bi-symétrique : le moment dû au décalage e_N du centre de gravité efficace "
                      "(§6.2.2.5(4)) n'est pas pris en compte.")
    if fab == "U":
        av.append("Section en U : le flambement par flexion-torsion (§6.3.1.4) n'est pas vérifié ici.")
    elif fab == "L":
        av.append("Cornière : flambement autour des axes principaux u-u et v-v ; le flambement par torsion et flexion-torsion "
                  "(§6.3.1.4) et l'excentricité des attaches (annexe BB.1.2) ne sont pas pris en compte.")
    return out


# ------------------------------------------------------------------ le sélecteur
def candidats(barre: Dict, familles: Sequence[str]) -> List[Dict]:
    """Chaque profilé des familles données, vérifié avec les autres données de la barre : {nom, famille, masse, taux,
    statut, motif}, TRIÉS PAR MASSE. Les profilés que la norme ne couvre pas restent (statut IMPOSSIBLE, taux None)."""
    b = EN.normaliser(barre)
    out = []
    for nom in SE.noms(list(familles)):
        b["profil"] = nom
        r = verifier(b)
        out.append({"nom": nom, "famille": r["famille"], "masse": r["masse_lineique"], "taux": r["taux_max"], "statut": r["statut"],
                    "motif": r.get("motif", ""), "courbes": r.get("courbes"), "gouvernant": r.get("gouvernant")})
    out.sort(key=lambda c: (c["masse"], c["nom"]))
    return out


def plus_leger(cands: Sequence[Dict]) -> Optional[Dict]:
    """Le plus léger qui tient (η ≤ 1) ; None si aucun."""
    tient = [c for c in cands if c["taux"] is not None and c["taux"] <= 1.0]
    return min(tient, key=lambda c: (c["masse"], c["nom"])) if tient else None

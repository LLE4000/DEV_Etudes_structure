# -*- coding: utf-8 -*-
"""Contrôles PARTAGÉS du lot A (16/11/2026) pour les sept moteurs d'assemblage : f_y et f_u de chaque pièce selon
SON épaisseur, par la table unique du site (NBN EN 1993-1-1 ANB:2018 3.2.1(1), option a ; tableau 3.3 ANB), et le
domaine de la poutre grugée lu sur la NUANCE.

Ce fichier n'est pas une suite (pas de « test_ » devant son nom) : chaque suite d'un moteur l'appelle avec ses cas,
puis rejoue les MÊMES contrôles sous chaque mutation (``MUTATIONS``) et exige qu'ils rougissent.

Les valeurs attendues sont écrites ici en clair (elles se lisent sur le tableau 3.3 ANB) : une copie NOMMÉE, que
``jouer`` lie à la table unique avant tout contrôle (``lien_table``).
"""
import json
import os

from acier import materiaux as MA
from acier.assemblages import materiau as MT
from acier.assemblages.poutre_poutre.doubles_cornieres.parite import materiau_du_corrige

# tableau 3.3 ANB (EN 10025-2:2004), lu sur le texte : S355 355 (t ≤ 16) / 345 (16 < t ≤ 40) / 325 (63 < t ≤ 80) ;
# S275 275 / 265 ; S460 N/NL (tableau 3.4 ANB) 460 / 440 ; f_u de 3 à 100 mm : S355 470, S275 410, S460 540
FY_S355, FY_S355_40, FY_S355_80 = 355.0, 345.0, 325.0
FY_S275_40 = 265.0
FY_S460_40 = 440.0
FU_S355, FU_S275, FU_S460 = 470.0, 410.0, 540.0
# l'ANCIENNE table des assemblages (EN 1993-1-1 tableau 3.1, sans épaisseur), lue dans le corrigé de référence (sa
# source, aucune recopie) : le matériau des calculs à la main et des exemples publiés des suites (validation de
# FORMULES), et la mutation « ancienne table »
with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "acier", "reference",
                       "reference_double_corniere.json"), encoding="utf-8") as _fh:
    TABLEAU_3_1 = materiau_du_corrige(json.load(_fh))


def proche(a, b, tol=1e-9):
    return a is not None and b is not None and abs(a - b) <= tol * max(1.0, abs(b))


def piece(R, X):
    """(t, f_y, f_u) retenus pour la pièce X."""
    m = (R.get("materiau") or {}).get(X) or {}
    return m.get("t"), R.get("fy_" + X), R.get("fu_" + X)


def controles(compute, attendus, ligne_fu=None, grugeage=None, perso=None, hors_table=None):
    """Les contrôles du lot A d'un moteur ; rend [(nom, condition, info)].

    ``attendus`` : [(libellé, saisies, {X: (t, fy, fu)})] — le matériau retenu par pièce ;
    ``ligne_fu`` : (libellé, saisies, clé) — une vérification proportionnelle à f_u : sa résistance vaut exactement
      470/510 de celle du matériau des exemples (EN 1993-1-1 tableau 3.1) ;
    ``grugeage`` : (saisies, X) — une poutre S275 de plus de 16 mm, grugée : f_y = 265 et le domaine reste celui de
      la S275 (« Valide », limites S275) ;
    ``perso`` : (saisies, X, fy, fu, libellé) — un acier « Personnalisé » : la valeur saisie (inchangé) ;
    ``hors_table`` : (saisies, mot) — une pièce au-delà de la dernière ligne du tableau : refus nommé (HorsTable)."""
    out = []
    for lib, u, att in attendus:
        R = compute(u)
        for X, (t, fy, fu) in att.items():
            tt, fyy, fuu = piece(R, X)
            out.append((f"{lib} — pièce {X} : t = {t:g} mm → f_y = {fy:g}, f_u = {fu:g} (tableaux 3.3 / 3.4 ANB)",
                        proche(tt, t) and proche(fyy, fy) and proche(fuu, fu), f"{(tt, fyy, fuu)}"))
    if ligne_fu:
        lib, u, k = ligne_fu
        Rn, Re = compute(u), compute(u, MT.EXEMPLES)
        a, b = Rn.ck[k].Rd, Re.ck[k].Rd
        out.append((f"{lib} — {k} : Rd = {b:.2f} × 470/510 = {a:.2f} kN (f_u = 470, tableau 3.3 ANB)",
                    proche(a, b * 470 / 510, 1e-12), f"{a} / {b}"))
    if grugeage:
        u, X = grugeage
        R = compute(u)
        out.append(("poutre grugée S275 de plus de 16 mm : f_y = 265 pour la résistance, domaine lu sur la NUANCE "
                    "(« Valide », limites S275 : 54,3)",
                    proche(R["fy_" + X], FY_S275_40) and R.val_N == "Valide" and R.lim_N == 54.3,
                    f"{R['fy_' + X]} ; {R.val_N} ; {R.lim_N}"))
    if perso:
        u, X, fy, fu, lib = perso
        R = compute(u)
        out.append((f"acier « Personnalisé » inchangé : f_y = {fy:g}, f_u = {fu:g} ({lib})",
                    proche(R["fy_" + X], fy) and proche(R["fu_" + X], fu), f"{(R['fy_' + X], R['fu_' + X])}"))
    if hors_table:
        u, mot = hors_table
        try:
            compute(u)
            msg = None
        except MA.HorsTable as e:
            msg = str(e)
        out.append((f"au-delà du tableau : refus nommé (HorsTable « {mot} : … »)", bool(msg) and mot in msg
                    and "ANB" in msg, str(msg)))
    return out


# Les mutations : chacune remet un défaut corrigé par le lot A ; les contrôles doivent rougir.
def _mut_table(on):
    """f_y et f_u nominaux sans épaisseur (l'ancienne table, EN 1993-1-1 tableau 3.1)."""
    if on:
        _mut_table.avant = MT.table_du_site
        MT.table_du_site = TABLEAU_3_1
    else:
        MT.table_du_site = _mut_table.avant


def _mut_epaisseur(on):
    """Le profilé lu à son âme (la paroi la plus mince) au lieu de max(tf, tw)."""
    if on:
        _mut_epaisseur.avant = MT.t_profil
        MT.t_profil = lambda tf, tw: min(tf, tw)
    else:
        MT.t_profil = _mut_epaisseur.avant


def _mut_domaine(on):
    """Le domaine de la poutre grugée lu sur la f_y réduite par l'épaisseur (et non sur la nuance)."""
    if on:
        _mut_domaine.avant = MT.nominale
        MT.nominale = lambda R, X: R["fy_" + X]
    else:
        MT.nominale = _mut_domaine.avant


def _regle_historique(nom, t):
    """L'ancienne règle du pied de poteau (fy_epaisseur, entrees.py) : f_y nominale − 10 MPa de 16 à 40 mm, − 20 au-delà
    (sans borne), f_u nominale."""
    a = TABLEAU_3_1(nom, t)
    return dict(a, fy=a["fy"] if t <= 16 else a["fy"] - 10 if t <= 40 else a["fy"] - 20)


def _mut_regle_pied(on):
    """La règle « −10 / −20 » écrite en dur à la place de la table (pied de poteau)."""
    if on:
        _mut_regle_pied.avant = MT.table_du_site
        MT.table_du_site = _regle_historique
    else:
        MT.table_du_site = _mut_regle_pied.avant


MUTATION_REGLE_PIED = ("règle « −10 / −20 MPa » écrite en dur (fausse en S460 et au-delà de 63 mm)", _mut_regle_pied)

MUTATIONS = [("f_y et f_u nominaux, sans épaisseur (ancienne table)", _mut_table),
             ("profilé lu à sa paroi la plus mince", _mut_epaisseur),
             ("domaine du grugeage lu sur la f_y réduite", _mut_domaine)]


# ---------------------------------------------------------------- reprise A2 : la gorge de PLEINE RÉSISTANCE
# a/t_w = β_w·γ_M2·f_y/(√2·f_u) (MSB P5 §2.2) est une exigence de SUR-RÉSISTANCE : une f_y plus BASSE la rendrait
# MOINS sévère. Elle prend la f_y de l'ÂME à SA propre épaisseur t_w (MT.t_ame), pas la plus petite f_y du profilé
# (max(t_f, t_w), qui vaut pour les résistances). Poutre portée HEB 400 S355 : t_w = 13,5 ≤ 16 → 355 ; t_f = 24 → 345.
# Mesuré au commit de base f3a5f6d : a/t_w = 0,5839, a_plein = 7,88 mm, gorge proposée 8 mm (non conservatif).
K_PLEIN_S355 = 0.9 * 1.25 * FY_S355 / (2 ** 0.5 * FU_S355)        # β_w = 0,9 ; γ_M2 = 1,25 : 0,6009
TW_HEB400 = 13.5


def controles_gorge(compute, predim):
    """Les contrôles de la gorge de pleine résistance (platines d'about) ; ``predim`` : les saisies du mode
    prédimensionnement. Rend [(nom, condition, info)]."""
    R = compute(dict(prof_S="HEB 400"))
    Rp = compute(dict(predim, prof_S="HEB 400"))
    Rn = compute({})
    a = K_PLEIN_S355 * TW_HEB400
    return [
        ("HEB 400 portée S355 : la gorge de pleine résistance prend la f_y de l'ÂME (t_w = 13,5 mm : 355), les "
         "résistances celle du profilé (max(t_f, t_w) = 24 mm : 345)",
         proche(R.fy_wS, FY_S355) and proche(R.fy_S, FY_S355_40), f"{R.fy_wS} / {R.fy_S}"),
        (f"HEB 400 portée : a/t_w = 0,9·1,25·355/(√2·470) = {K_PLEIN_S355:.4f} ; a_plein = {a:.2f} mm (et non 7,88)",
         proche(R.k_plein, K_PLEIN_S355) and proche(R.a_plein, a), f"{R.k_plein} / {R.a_plein}"),
        ("HEB 400 portée : gorge proposée par le prédimensionnement = 9 mm (et non 8)",
         Rp.pd.a == 9 and Rp.a_w == 9, f"{Rp.pd.a} / {Rp.a_w}"),
        (f"note neuve (âme et ailes ≤ 16 mm) : inchangée, a/t_w = {K_PLEIN_S355:.4f}",
         proche(Rn.k_plein, K_PLEIN_S355) and proche(Rn.fy_wS, Rn.fy_S), f"{Rn.k_plein}"),
    ]


def _mut_gorge(on):
    """La gorge de pleine résistance lue sur la f_y du profilé à max(t_f, t_w) (la règle des résistances)."""
    if on:
        _mut_gorge.avant = MT.t_ame
        MT.t_ame = lambda tf, tw: max(tf, tw)
    else:
        MT.t_ame = _mut_gorge.avant


MUTATION_GORGE = ("f_y du profilé à max(t_f, t_w) dans la gorge de pleine résistance", _mut_gorge)


def jouer_gorge(chk, compute, predim):
    """Joue les contrôles de la gorge de pleine résistance, puis la mutation (ils doivent rougir)."""
    for nom, cond, info in controles_gorge(compute, predim):
        chk(nom, cond, info)
    nom, mut = MUTATION_GORGE
    mut(True)
    try:
        rouges = [n for n, c, _ in controles_gorge(compute, predim) if not c]
    finally:
        mut(False)
    chk(f"mutation « {nom} » : les contrôles de la gorge rougissent ({len(rouges)})", len(rouges) == 3, str(rouges))


def lien_table():
    """Les valeurs écrites ci-dessus sont celles de la table unique (acier/materiaux.py) : la copie est tenue."""
    return ([FY_S355, FY_S355_40, FY_S355_80, FY_S275_40, FY_S460_40, FU_S355, FU_S275, FU_S460]
            == [MA.fy("S355", 10), MA.fy("S355", 20), MA.fy("S355", 70), MA.fy("S275", 20), MA.fy("S460 N/NL", 20),
                MA.fu("S355", 10), MA.fu("S275", 10), MA.fu("S460 N/NL", 10)])


def jouer(chk, compute, mutations=None, **kw):
    """Joue les contrôles, puis chaque mutation (ses contrôles doivent rougir) ; ``chk`` : la fonction de la suite."""
    chk("lot A : les valeurs attendues (tableaux 3.3 et 3.4 ANB) sont celles de la table unique", lien_table())
    for nom, cond, info in controles(compute, **kw):
        chk(nom, cond, info)
    for nom, mut in (mutations or MUTATIONS):
        mut(True)
        try:
            rouges = [n for n, c, _ in controles(compute, **kw) if not c]
        except Exception as e:  # noqa: BLE001 — une mutation qui fait planter le calcul a aussi été vue
            rouges = [f"exception {type(e).__name__}"]
        finally:
            mut(False)
        chk(f"mutation « {nom} » : les contrôles du lot A rougissent ({len(rouges)})", bool(rouges))


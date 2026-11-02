# -*- coding: utf-8 -*-
"""Export texte de l'assemblage poutre–poteau par platine d'about :
l'essentiel (statut, taux, données, vérifications essentielles) ou la
version complète (toutes les vérifications actives avec formules)."""
import re
import unicodedata

from acier.js import js_str
from acier.formats import F, pct
from . import synthese, formules
from .notation import ref_courte


def nom_fichier(u, ext):
    base = "assemblage_platine_about_poteau"
    rep = (u.get("id_rep") or "").strip()
    if rep:
        s = unicodedata.normalize("NFKD", rep).encode("ascii", "ignore").decode()
        s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
        if s:
            base += "_" + s
    return base + ext


def construire_texte(R, complet=False):
    u = R.u
    L = ["ASSEMBLAGE POUTRE–POTEAU — PLATINE D'ABOUT DE HAUTEUR PARTIELLE",
         "=" * 63,
         R.statut + " — taux maximal " + pct(R.eta_max, 1)
         + (" — " + R.gov.label if R.gov else ""), ""]
    L += ["Données : poteau " + str(u.prof_P) + " " + str(u.nu_P) + " ; poutre portée "
          + str(u.prof_S) + " " + str(u.nu_S)
          + " ; appui : " + str(R.appui).lower()
          + " ; platine " + js_str(R.h_p) + "x" + js_str(R.b_p) + "x" + js_str(R.t_p) + " " + str(u.nu_pl)
          + " ; boulons " + R.boulon + " " + str(u.classe) + " (" + js_str(R.n_1) + " × 2, p3 = "
          + F(R.p_3, 0) + " mm)"
          + " ; a = " + js_str(R.a_w) + " mm ; VEd = " + F(u.V_Ed, 1) + " kN"
          + (" ; NEd = " + F(u.N_Ed, 1) + " kN (" + str(u.nat_N) + ")" if u.N_Ed > 0 else ""),
          "Groupe CENTRÉ (deux files symétriques) : aucun moment d'excentricité ; VEd/n par boulon = "
          + F(R.Fv_Ed, 2) + " kN",
          "Moment apporté au poteau (information, hors verdict) : VEd·e_col = "
          + F(R.M_col, 2) + " kNm avec e_col = " + F(R.e_col, 1) + " mm"]
    if R.aile:
        L.append("Cas (a) : gauge minimal (l'écrou dégage le congé du poteau) p3 ≥ twP + 2·rP + dw = "
                 + F(R.p3_congeP, 1) + " mm ; pince au bout d'aile e = " + F(R.e_P, 1) + " mm")
    else:
        L.append("Cas (b) : largeur droite entre congés " + F(R.libre_droit, 0)
                 + " mm (platine ≤ " + F(R.bp_maxP, 0) + " mm) ; passage de la semelle hP − 2·tfP − 2·j = "
                 + F(R.libre_ailes, 0) + " mm")
    # Le rédacteur saisi au cartouche (id_red) est imprimé ici aussi — un champ
    # saisi et jamais lu était un défaut (charte du 31/10/2026, amendement 3).
    if str(u.get("id_red") or "").strip():
        L.append("Rédacteur : " + str(u.get("id_red")).strip())
    L.append("")
    for a in R.alerts:
        L.append(("[BLOQUANT] " if a.block else "[note] ") + a.msg)
    if R.alerts:
        L.append("")
    for cle in ("boulons", "platine", "portee", "poteau", "soudure"):
        ls = synthese.lignes(R, cle)
        if complet:
            keep = ls
        else:
            keep = [l for l in ls if l["c"].ess or not l["ok"]]
        if not keep:
            continue
        L.append(synthese.titre_table(cle).upper())
        for l in keep:
            L.append("  " + l["lab"] + " : " + l["Ed"] + " / " + l["Rd"]
                     + (" " + l["unit"] if l["unit"] else "") + " = " + l["pct"]
                     + (" OK" if l["ok"] else " NON OK") + "  [" + l["ref"] + "]")
            if complet:
                for t in formules.textes(R, l["c"])[:-1]:
                    L.append("      " + t)
        L.append("")
    if R.ame:
        L.append("Poteau porteur, cas (b) : la flexion HORS PLAN de l'âme sous la traction des boulons "
                 "n'est PAS vérifiée (EN 1993-1-8 ne la couvre pas pour une âme non raidie) — voir "
                 "Méthode › Limites.")
        L.append("")
    L.append("Tableau 3.3 : " + ("conforme" if R.dist_ok else "NON CONFORME"))
    for x in R.dist:
        L.append("  " + x.lab + " : " + F(x.val, 1) + " mm (min " + F(x.min, 1)
                 + (" ; max " + F(x.max, 0) if x.max is not None else "") + ") "
                 + ("OK" if x.ok else "NON OK"))
    L += ["", "Références : " + " ; ".join(sorted({ref_courte(c.ref) for c in R.checks if c.active}))]
    return "\n".join(L) + "\n"

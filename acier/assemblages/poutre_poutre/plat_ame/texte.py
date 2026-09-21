# -*- coding: utf-8 -*-
"""Export texte de l'assemblage par plat d'âme : l'essentiel (statut, taux,
données, vérifications essentielles) ou la version complète (toutes les
vérifications actives avec formules)."""
import re
import unicodedata

from acier.js import js_str
from acier.formats import F, pct
from . import synthese, formules
from .notation import ref_courte


def nom_fichier(u, ext):
    base = "assemblage_plat_ame"
    rep = (u.get("id_rep") or "").strip()
    if rep:
        s = unicodedata.normalize("NFKD", rep).encode("ascii", "ignore").decode()
        s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
        if s:
            base += "_" + s
    return base + ext


def construire_texte(R, complet=False):
    u = R.u
    L = ["ASSEMBLAGE POUTRE–POUTRE — PLAT D'ÂME SOUDÉ (FIN PLATE)",
         "=" * 58,
         R.statut + " — taux maximal " + pct(R.eta_max, 1)
         + (" — " + R.gov.label if R.gov else ""), ""]
    L += ["Données : principale " + str(u.prof_P) + " " + str(u.nu_P) + " ; secondaire "
          + str(u.prof_S) + " " + str(u.nu_S)
          + " ; plat " + js_str(R.h_p) + "x" + js_str(R.b_p) + "x" + js_str(R.t_p) + " " + str(u.nu_pl)
          + " ; boulons " + R.boulon + " " + str(u.classe) + " (" + js_str(R.n_1) + " × " + js_str(R.n_2) + ")"
          + " ; a = " + js_str(R.a_w) + " mm ; VEd = " + F(u.V_Ed, 1) + " kN"
          + (" ; NEd = " + F(u.N_Ed, 1) + " kN" if u.N_Ed else "")
          + (" ; MEd = " + F(u.M_Ed, 2) + " kNm" if u.M_Ed else ""),
         "z = " + F(R.zeff, 1) + " mm ; MS = VEd·z + |MEd| = " + F(R.M_S, 2) + " kNm ; Ip = "
         + F(R.Ip, 0) + " mm²", ""]
    for a in R.alerts:
        L.append(("[BLOQUANT] " if a.block else "[note] ") + a.msg)
    if R.alerts:
        L.append("")
    for cle in ("boulons", "plat", "portee", "porteuse", "soudure"):
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
    L.append("Tableau 3.3 : " + ("conforme" if R.dist_ok else "NON CONFORME"))
    for x in R.dist:
        L.append("  " + x.lab + " : " + F(x.val, 1) + " mm (min " + F(x.min, 1)
                 + (" ; max " + F(x.max, 0) if x.max is not None else "") + ") "
                 + ("OK" if x.ok else "NON OK"))
    L += ["", "Références : " + " ; ".join(sorted({ref_courte(c.ref) for c in R.checks if c.active}))]
    return "\n".join(L) + "\n"

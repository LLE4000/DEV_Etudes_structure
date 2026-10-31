# -*- coding: utf-8 -*-
"""Export texte du pied de poteau : l'essentiel (statut, taux, données,
vérifications essentielles) ou la version complète (toutes les vérifications
actives avec formules)."""
import re
import unicodedata

from acier.js import js_str, N
from acier.formats import F, pct
from . import synthese, formules
from .notation import ref_courte
from .entrees import TYPE_CROSSE, OUI


def nom_fichier(u, ext):
    base = "pied_poteau_platine"
    rep = (u.get("id_rep") or "").strip()
    if rep:
        s = unicodedata.normalize("NFKD", rep).encode("ascii", "ignore").decode()
        s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
        if s:
            base += "_" + s
    return base + ext


def ligne_donnees(R):
    u = R.u
    tige = (js_str(R.n_t) + " tiges " + str(u.ancr) + " " + str(u.classe) + " (" + js_str(int(u.n_ax)) + " files × "
            + js_str(int(u.n_ay)) + " rangées, wx = " + js_str(N(u.w_x)) + ", wy = " + js_str(N(u.w_y)) + " mm), "
            + ("crosse" if u.typ_a == TYPE_CROSSE else "tête dh = " + js_str(N(u.d_h))) + ", hef = " + js_str(N(u.hef)) + " mm")
    return ("Poteau " + R.nom_c + " " + str(u.nu_c) + " ; platine " + js_str(N(u.bp_u)) + "x" + js_str(N(u.lp_u)) + "x"
            + js_str(N(u.tp_u)) + " " + str(u.nu_pl) + " ; " + tige + " ; béton " + str(u.beton)
            + (" fissuré" if u.fiss == OUI else " non fissuré") + ", massif " + js_str(N(u.bc_u)) + "x"
            + js_str(N(u.lc_u)) + "x" + js_str(N(u.hc_u)) + ", mortier tg = " + js_str(N(u.t_g)) + " mm ; a = "
            + js_str(N(u.a_w)) + " mm ; N = " + F(u.N_Ed, 1) + " kN, Mx = " + F(u.Mx_Ed, 1) + ", My = "
            + F(u.My_Ed, 1) + " kNm, Vx = " + F(u.Vx_Ed, 1) + ", Vy = " + F(u.Vy_Ed, 1) + " kN"
            + (", T = " + F(u.T_Ed, 2) + " kNm" if N(u.T_Ed) else ""))


def construire_texte(R, complet=False):
    L = ["PIED DE POTEAU — PLATINE SUR BÉTON (N + Mx + My)",
         "=" * 64,
         R.statut + " — taux maximal " + pct(R.eta_max, 1)
         + (" — " + R.gov.label if R.gov else ""), ""]
    L += ["Données : " + ligne_donnees(R) + ".",
          "Répartition élastique : fjd = " + F(R.fjd, 1) + " MPa, c = " + F(R.c, 1) + " mm, Aeff = "
          + F(R.A_eff / 100, 0) + " cm² ; C = " + F(R.C, 1) + " kN en (" + F(R.xC, 0) + " ; " + F(R.yC, 0)
          + ") mm, pmax = " + F(R.pmax, 1) + " MPa, ΣT = " + F(R.T_tot, 1) + " kN (T max = " + F(R.T_max, 1) + " kN).",
          "Niveau 1 : " + pct(R.eta1, 1) + " ; niveau 2 (information) : " + pct(R.eta2, 1) + ".", ""]
    for a in R.alerts:
        L.append(("[BLOQUANT] " if a.block else "[note] ") + a.msg)
    if R.alerts:
        L.append("")
    for cle in ("pied", "beton", "platine", "tiges", "poteau", "soudure"):
        ls = synthese.lignes(R, cle)
        keep = ls if complet else [l for l in ls if l["c"].ess or not l["ok"]]
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
        L.append("  " + x.lab + " : " + F(x.val, 1) + " mm (min " + F(x.min, 1) + ") " + ("OK" if x.ok else "NON OK"))
    if R.rig:
        L.append("Rigidité : Sj,ini = " + F(R.Sj / 1e3, 1) + " MNm/rad, seuil rigide " + F(R.Sj_seuil / 1e3, 1)
                 + " MNm/rad → " + ("RIGIDE" if R.rig["rigide"] else "SEMI-RIGIDE") + " (information).")
    L += ["", "Références : " + " ; ".join(sorted({ref_courte(c.ref) for c in R.checks if c.active}))]
    return "\n".join(L) + "\n"

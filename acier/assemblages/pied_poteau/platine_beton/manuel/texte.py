# -*- coding: utf-8 -*-
"""Export texte du calcul manuel : la note à la main en clair — données,
efforts des tiges, chaque boulon tendu (bras, moment, longueur efficace,
résistance, épaisseur demandée), chaque section au nu, la bande comprimée,
les trous, l'épaisseur requise, puis les tableaux de vérification."""
from acier.js import js_str, N
from acier.formats import F, pct
from .. import formules
from ..notation import ref_courte
from . import synthese
from . import elements as EL


def ligne_donnees(R):
    u = R.u
    return ("Platine " + js_str(N(u.bp_u)) + "x" + js_str(N(u.lp_u)) + "x" + js_str(N(u.tp_u)) + " " + str(u.nu_pl)
            + " ; éléments soudés : " + " + ".join(f"{EL.nom(e)} en ({F(e['x'], 0)} ; {F(e['y'], 0)})" + (" tourné" if e["rot"] else "")
                                                   for e in R.elems)
            + (" ; raidisseurs : " + ", ".join(f"({F(r['x'], 0)} ; {F(r['y'], 0)}) {r['dir']} L {F(r['L'], 0)} t {F(r['t'], 0)}"
                                               for r in R.raid) if R.raid else "")
            + " ; " + js_str(R.n_t) + " boulons " + str(u.ancr) + " " + str(u.classe) + " en "
            + ", ".join(f"({F(t['x'], 0)} ; {F(t['y'], 0)})" for t in R.tiges)
            + " ; béton " + str(u.beton) + ", massif " + js_str(N(u.bc_u)) + "x" + js_str(N(u.lc_u)) + "x" + js_str(N(u.hc_u))
            + ", mortier tg = " + js_str(N(u.t_g)) + " mm ; a = " + js_str(N(u.a_w)) + " mm ; N = " + F(u.N_Ed, 1)
            + " kN, Mx = " + F(u.Mx_Ed, 1) + ", My = " + F(u.My_Ed, 1) + " kNm, Vx = " + F(u.Vx_Ed, 1) + ", Vy = "
            + F(u.Vy_Ed, 1) + " kN" + (", T = " + F(u.T_Ed, 2) + " kNm" if N(u.T_Ed) else "")
            + " en (" + F(R.xN, 0) + " ; " + F(R.yN, 0) + ")")


def lignes_main(R):
    """Le calcul à la main, ligne par ligne."""
    P = R.plaque
    L = []

    def mm(v):
        return "∞" if v == float("inf") else F(v, 1)
    L.append("Répartition (platine rigide, " + ("toute la platine" if R.plein else "bande c = " + F(R.c, 1) + " mm") + ") : C = "
             + F(R.C, 1) + " kN en (" + F(R.xC, 0) + " ; " + F(R.yC, 0) + "), p max = " + F(R.pmax, 2) + " MPa (fjd = "
             + F(R.fjd, 1) + ") ; tiges : " + ", ".join(f"n°{t['i'] + 1} T = {F(t['T'], 1)} kN" for t in R.tiges) + ".")
    for t in R.tiges:
        if t["T"] <= 1e-9:
            continue
        L.append(f"Boulon {t['i'] + 1} ({F(t['x'], 0)} ; {F(t['y'], 0)}) : appui = {t['appui']['face']}, bras m = {F(t['m'], 1)} mm"
                 + (" (angle)" if t["coin"] else "") + f", M = T·m = {F(t['T'], 1)} × {F(t['m'], 1)} = {F(t['M'], 2)} kNm ; "
                 f"leff,1 = {F(t['leff1'], 0)} mm, Mpl,1 = {F(t['Mpl1'], 2)} kNm, Lb = {F(R.Lb, 0)} / Lb* = "
                 f"{F(min(t['Lb_star'], 9999), 0)} ({'avec' if t['levier'] else 'sans'} effet de levier), FT,Rd = {F(t['FT_Rd'], 1)} kN "
                 f"(mode {t['mode_ts']}) → {pct(t['T'] / t['FT_Rd'] if t['FT_Rd'] > 0 else 9.99, 0)} ; σvm = {F(t['svm'], 0)} MPa ; "
                 f"épaisseur pour ce boulon : plastique {F(t['t_pl'], 1)} mm, élastique {F(t['t_el'], 1)} mm.")
    for s in R.sections:
        L.append(f"Section au nu {s['nom']} : boulons {', '.join(str(i + 1) for i in s['idx'])}, bras ⊥ "
                 f"{', '.join(F(m, 1) for m in s['m'])} mm, M = ΣT·m = {F(s['M'], 2)} kNm, b = min(platine ; L + 2m) = "
                 f"{F(s['b'], 0)} mm, Mpl = {F(s['Mpl'], 2)} kNm → {pct(s['eta'], 0)} ; σvm = {F(s['svm'], 0)} MPa ; "
                 f"épaisseur : plastique {F(s['t_pl'], 1)}, élastique {F(s['t_el'], 1)} mm.")
    cs = R.console
    if cs["m_Ed"] > 1e-9:
        L.append(f"Bande comprimée la plus sollicitée : {cs['nom']}, p au nu = {F(cs['p0'], 2)} MPa, console de "
                 f"{F(cs['L'], 0)} mm, mEd = ∫p·t dt = {F(cs['m_Ed'], 2)} kNm/m ≤ tp²·fy/(6·γM0) = {F(cs['m_Rd'], 2)} → "
                 f"tc = {F(cs['t'], 1)} mm.")
    pb = R.pire_appui
    if pb and pb["Fv"] > 1e-9:
        L.append(f"Pression diamétrale, boulon {pb['i'] + 1} : Fv = {F(pb['Fv'], 1)} kN, αb = {F(pb['ab'], 2)}, k1 = {F(pb['k1'], 2)}, "
                 f"Fb,Rd = {F(pb['Fb'], 1)} kN → tb = {F(pb['t_b'], 1)} mm.")
    L.append("ÉPAISSEUR REQUISE = max(tT " + mm(P["t_T"]) + " ; tS " + mm(P["t_S"]) + " ; tc " + mm(P["t_c"]) + " ; tb "
             + mm(P["t_b"]) + ") = " + mm(P["t_req"]) + " mm pour tp = " + F(R.u.tp_u, 0) + " mm — critère " + P["critere"]
             + " (répartition recalculée à chaque épaisseur).")
    return L


def construire_texte(R, complet=False):
    L = ["PLATINE SUR BÉTON — CALCUL MANUEL DE LA PLATINE",
         "=" * 64,
         R.statut + " — taux maximal " + pct(R.eta_max, 1) + (" — " + R.gov.label if R.gov else ""), "",
         "Données : " + ligne_donnees(R) + ".", ""]
    L += lignes_main(R)
    L.append("")
    for a in R.alerts:
        L.append(("[BLOQUANT] " if a.block else "[note] ") + a.msg)
    if R.alerts:
        L.append("")
    for cle in synthese.ORDRE:
        ls = synthese.lignes(R, cle)
        keep = ls if complet else [l for l in ls if l["c"].ess or not l["ok"]]
        if not keep:
            continue
        L.append(synthese.titre_table(cle).upper())
        for l in keep:
            L.append("  " + l["lab"] + " : " + l["Ed"] + " / " + l["Rd"] + (" " + l["unit"] if l["unit"] else "")
                     + " = " + l["pct"] + (" OK" if l["ok"] else " NON OK") + "  [" + l["ref"] + "]")
            if complet:
                for t in formules.textes(R, l["c"])[:-1]:
                    L.append("      " + t)
        L.append("")
    L.append("Tableau 3.3 : " + ("conforme" if R.dist_ok else "NON CONFORME"))
    for x in R.dist:
        L.append("  " + x.lab + " : " + F(x.val, 1) + " mm (min " + F(x.min, 1) + ") " + ("OK" if x.ok else "NON OK"))
    L += ["", "Références : " + " ; ".join(sorted({ref_courte(c.ref) for c in R.checks if c.active}))]
    return "\n".join(L) + "\n"

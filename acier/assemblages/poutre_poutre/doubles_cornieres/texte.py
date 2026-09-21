# -*- coding: utf-8 -*-
"""Export texte de l'assemblage — même contenu, même structure que le HTML.

Transcription de ``dataLines``, ``hypLines`` et ``buildTxt`` : identification
(si renseignée), profilés, nuances, cornières, boulons, soudures, efforts,
hypothèses et excentricités, vérifications (essentielles : celles marquées
essentielles et toutes celles qui ne sont pas vérifiées ; complètes : toutes
les actives) avec formule, valeurs, Ed, Rd, η, référence et nature, pinces et
entraxes, alertes, vérification dimensionnante, taux maximal, conclusion.
Les cinq exports du corrigé (``exports_texte``) servent de témoins.
"""
import re

from acier.js import N, js_str
from acier.formats import F, pct
from acier.bibliotheques import PERSO


def lignes_donnees(R):
    """``[(libellé, valeur), …]`` des données de l'assemblage."""
    u = R.u
    L = []
    nt = N(u.d_nt); nb = N(u.d_nb)
    for lab, val in (("Projet", u.id_projet), ("Repère", u.id_rep), ("Rédacteur", u.id_red), ("Date", u.id_date)):
        if val:
            L.append((lab, str(val)))
    L.append(("Poutre principale", ("h " + F(R.h_P, 0) + " b " + F(R.b_P, 0) + " tw " + F(R.tw_P, 1) + " tf " + F(R.tf_P, 1)
                                    if u.prof_P == PERSO else u.prof_P) + " – " + u.nu_P))
    L.append(("Poutre secondaire", ("h " + F(R.h_S, 0) + " b " + F(R.b_S, 0) + " tw " + F(R.tw_S, 1) + " tf " + F(R.tf_S, 1)
                                    if u.prof_S == PERSO else u.prof_S) + " – " + u.nu_S
              + ((" – grugeage " + ("sup. " + F(nt, 0) if nt else "") + (" / " if nt and nb else "")
                  + ("inf. " + F(nb, 0) if nb else "") + " × " + F(N(u.l_n), 0) + " mm") if (nt or nb) else " – non grugée")))
    L.append(("Cornières", "2 × " + R.corn_txt + " – " + u.nu_C + " – Lc = " + F(R.L_C, 0) + " mm"))
    if R.bolt_S or R.bolt_P:
        L.append(("Boulons", R.boulon + " classe " + u.classe + ", catégorie " + u.cat + " – "
                  + ("âme secondaire : " + js_str(R.n_S) + " (" + js_str(R.n1_S) + " × " + js_str(R.n2_S) + ", 2 plans)" if R.bolt_S else "")
                  + (" ; " if R.bolt_S and R.bolt_P else "")
                  + ("âme principale : 2 × " + js_str(R.n_P) + " (1 plan)" if R.bolt_P else "")))
    if not R.bolt_S:
        L.append(("Soudures ailes B / âme secondaire", "a = " + F(N(u.a_S), 0) + " mm, cordon vertical Lc + retours " + F(N(u.lh_S), 0) + " mm"))
    if not R.bolt_P:
        L.append(("Soudures ailes A / âme principale", "a = " + F(N(u.a_P), 0) + " mm, cordon vertical Lc + retours " + F(N(u.lh_P), 0) + " mm"))
    L.append(("Efforts ELU", "VEd = " + F(abs(N(u.V_Ed)), 1) + " kN ; NEd = " + F(N(u.N_Ed), 1) + " kN ; HEd = " + F(N(u.H_Ed), 1)
              + " kN ; MEd = " + F(N(u.M_Ed), 2) + " kNm"))
    return L


def lignes_hypotheses(R):
    """Les hypothèses et excentricités, en clair."""
    u = R.u
    return [
        "Assemblage nominalement articulé ; ligne de transfert (rotule) à la face de l'âme de la poutre principale (MSB Part 5 §4.2.1.1).",
        ("Groupe S : répartition élastique (inertie polaire) sous VEd, NEd et MS = VEd·z + |MEd| ; z = " + F(R.z_S, 1) + " mm ; MS = " + F(R.M_S, 2) + " kNm."
         if R.bolt_S else "Cordons S : groupe de cordons en répartition élastique ; z = " + F(R.zw_S, 1) + " mm ; MS = " + F(R.M_S, 2) + " kNm."),
        ("Groupe P : excentricité eP reprise par chaque cornière (option conservatrice) ; MP = " + F(R.M_P, 2) + " kNm."
         if R.M_P > 0 else "Groupe P : cisaillement centré" + (", facteur " + F(N(u.k_rot), 2) + " sur Fv,Rd pour la traction parasite due à la rotation." if R.bolt_P else ".")),
        ("Poutre grugée : flexion de la section réduite sous MEd = VEd·(gh + ln) = " + F(R.ck["mN"].Ed, 2) + " kNm."
         if R.cas_g > 0 else "Poutre secondaire non grugée."),
        "Pression diamétrale : interaction quadratique des composantes verticale et horizontale.",
        "Coefficients partiels : γM0 = " + F(N(u.g_M0), 2) + " ; γM2 = " + F(N(u.g_M2), 2) + " (sections nettes : " + F(N(u.g_M2n), 2) + ").",
    ]


def verifications_listees(R, complet=False):
    """Les vérifications retenues dans l'export : toutes les actives (version
    complète) ou les essentielles et celles qui ne sont pas vérifiées."""
    return [c for c in R.checks if c.active and (complet or c.ess or not c.ok)]


def construire_texte(R, complet=False):
    """Le résumé texte (``buildTxt``)."""
    T = ["ASSEMBLAGE POUTRE–POUTRE PAR DOUBLE CORNIÈRE D'ÂME", ""]
    for lab, val in lignes_donnees(R):
        T.append(lab + " : " + val)
    T.extend(["", "HYPOTHÈSES ET EXCENTRICITÉS :"])
    for x in lignes_hypotheses(R):
        T.append("- " + x)
    T.extend(["", "VÉRIFICATIONS" + ("" if complet else " ESSENTIELLES") + " (EC = Eurocode direct ; COMP = modèle complémentaire ; INT = interprétation) :", ""])
    for i, c in enumerate(verifications_listees(R, complet), start=1):
        nd = 3 if c.unit == "-" else 1
        T.append(js_str(i) + ". " + c.label)
        T.append("   " + c.formula)
        if c.vals:
            T.append("   " + c.vals)
        T.append("   Ed = " + F(c.Ed, nd) + " ; Rd = " + F(c.Rd, nd) + ("" if c.unit == "-" else " " + c.unit)
                 + " ; η = " + pct(c.eta, 1) + " → " + ("OK" if c.ok else "NON OK"))
        T.append("   Réf. : " + c.ref + " [" + c.nat + "]")
        T.append("")
    T.append("PINCES ET ENTRAXES (EN 1993-1-8 Tableau 3.3) : " + ("conformes" if R.dist_ok else "NON CONFORMES"))
    for a in R.alerts:
        T.append(("ALERTE BLOQUANTE : " if a.block else "ATTENTION : ") + a.msg)
    T.extend(["", "VÉRIFICATION DIMENSIONNANTE :", R.gov.label if R.gov else "—", "Taux maximum = " + pct(R.eta_max, 1),
              "", "CONCLUSION :",
              ("ASSEMBLAGE VÉRIFIÉ À L'ELU SELON EN 1993-1-8 ET EN 1993-1-1"
               + (" – SOUS RÉSERVE DES POINTS SIGNALÉS CI-DESSUS." if (R.reserve or R.alerts) else "."))
              if R.verified else "ASSEMBLAGE NON VÉRIFIÉ."])
    return "\n".join(T)


def nom_fichier(u, ext):
    """« assemblage_<repère><ext> », caractères sûrs (comme ``fname``)."""
    base = "assemblage_" + (str(u.get("id_rep") or "") or "double_corniere")
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", base) + ext

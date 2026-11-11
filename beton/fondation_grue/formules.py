# -*- coding: utf-8 -*-
"""Fondations de grues — les FORMULES telles qu'elles s'écrivent dans la note et dans les fenêtres de calcul :
« grandeur = formule = valeurs = résultat », avec la cellule du classeur en regard.

Rien n'est calculé ici : chaque résultat est une valeur du relevé (``classeur.Releve``) ; ce fichier dit seulement de
QUELS termes elle se retrouve et par QUELLE formule (``gabarit`` : une expression Python des identifiants des termes,
lue par le kit des notes ``Sub``). ``tests/test_fondation_grue_moteur.py`` ré-évalue chaque gabarit avec ses termes
non arrondis et exige le résultat du relevé à 1e-9 : c'est la preuve que la formule imprimée EST celle du classeur.

Unités imprimées : m pour la statique (L, e, s… = cm du classeur / 100), mm pour les sections (b = cm × 10), kN,
kN·m, N/mm², mm². Les facteurs ``*1000`` et ``*1e6`` sont des conversions d'unités (le kit les imprime 10³, 10⁶).

    blocs(f, r) -> [{"cle", "nom", "piece", "sollicitations": [formule…], "verifications": [vérification…]}]
    vérification = {"cle", "nom", "piece", "valeur", "limite", "unite", "taux", "etat_classeur", "cellule_etat",
                    "texte" (facultatif), "formules": [formule…]}
    formule = {"nom", "gabarit", "termes": [[identifiant, symbole, valeur]…], "resultat", "unite", "nd", "cellule"}
"""
import math

from . import classeur as K
from . import entree as E


def _fx(nom, gabarit, termes, resultat, unite, nd=None, cellule=None):
    return {"nom": nom, "gabarit": gabarit, "termes": [list(t) for t in termes], "resultat": resultat, "unite": unite,
            "nd": nd, "cellule": cellule}


def _verif(cle, nom, piece, valeur, limite, unite, etat, cellule_etat, formules, taux=None, texte=None):
    if taux is None and valeur is not None and limite:
        taux = valeur / limite
    return {"cle": cle, "nom": nom, "piece": piece, "valeur": valeur, "limite": limite, "unite": unite, "taux": taux,
            "etat_classeur": etat, "cellule_etat": cellule_etat, "texte": texte, "formules": formules}


def _barres(lit) -> str:
    """« 10Ø20 », ou « 8Ø20 + 2Ø16 » pour deux groupes (un groupe à 0 barre ne s'écrit pas)."""
    lits = lit if isinstance(lit, list) else [lit]
    return " + ".join(f"{g['n']}Ø{g['d']}" for g in lits if g["n"] > 0) or "—"


def _cadres(et) -> str:
    return f"{et['brins']} brins Ø{et['d']} – {et['pas']:g} mm"


# ------------------------------------------------------------------ formules de section, communes
def _pp(nom, b, h, res, cel):
    return _fx(nom, "25*b*h", (("b", "b", b / 100), ("h", "h", h / 100)), res, "kN/m", 2, cel)


def _h_calc(sym_M, M, m, b, res, cel, nom="h_calc", sym_b="b"):
    """La hauteur calculée ; ``sym_b`` nomme la largeur que le classeur y met (au droit du pieu : son diamètre D)."""
    return _fx(nom, "sqrt(M*1e6/(mu*b*sb))", (("M", sym_M, M), ("mu", "μ", m["mu"]), ("b", sym_b, b * 10),
                                              ("sb", "σ'_b", m["sigma_b"])), res, "mm", 0, cel)


def _omega(nom, sym_M, M, m, sym_d, d, res, cel):
    return _fx(nom, "M*1e6/(sa*0.9*d)", (("M", sym_M, M), ("sa", "σ_adm", m["sigma_adm"]), ("d", sym_d, d)),
               res, "mm²", 0, cel)


def _omega_choisie(nom, lit, res, cel):
    if isinstance(lit, list):
        return _fx(nom, "pi*(n1*(D1/2)**2 + n2*(D2/2)**2)",
                   (("n1", "n_1", lit[0]["n"]), ("D1", "Ø_1", lit[0]["d"]), ("n2", "n_2", lit[1]["n"]),
                    ("D2", "Ø_2", lit[1]["d"])), res, "mm²", 0, cel)
    return _fx(nom, "pi*n*(D/2)**2", (("n", "n", lit["n"]), ("D", "Ø", lit["d"])), res, "mm²", 0, cel)


def _renfort(f, r, m, b, Dp, sym_P, piece_base, cellules, etats, om_h_formule, etriers_formule, limite_h) -> dict:
    """Le bloc « Renforcement au droit du pieu », commun aux trois fiches (b : largeur de la poutre sur le pieu)."""
    P, d = r["renfort.P"], r["renfort.d"]
    c = cellules
    soll = [
        _fx("p_1", "P/b", (("P", sym_P, P), ("b", "b", b / 100)), r["renfort.p1"], "kN/m", 1, c["p1"]),
        _fx("p_2a", "P/D", (("P", sym_P, P), ("D", "D", Dp / 100)), r["renfort.p2a"], "kN/m", 1, c["p2a"]),
        _fx("p_2b", "P/(D/2)", (("P", sym_P, P), ("D", "D", Dp / 100)), r["renfort.p2b"], "kN/m", 1, c["p2b"]),
        _fx("M_A", "p1*(b/2)**2/2 - p2*(D/2)**2/2", (("p1", "p_1", r["renfort.p1"]), ("b", "b", b / 100),
                                                     ("p2", "p_2a", r["renfort.p2a"]), ("D", "D", Dp / 100)),
            r["renfort.M_A"], "kN·m", 2, c["M_A"]),
        _fx("M_B", "p1*(b/2)**2/2 - p2*(D/4)**2/2", (("p1", "p_1", r["renfort.p1"]), ("b", "b", b / 100),
                                                     ("p2", "p_2b", r["renfort.p2b"]), ("D", "D", Dp / 100)),
            r["renfort.M_B"], "kN·m", 2, c["M_B"]),
    ]
    hA, hB = r["renfort.h_calc_A"], r["renfort.h_calc_B"]
    oA, oB, oh = r["renfort.omega_sup_A"], r["renfort.omega_sup_B"], r["renfort.omega_brins_horizontaux"]
    verifs = [
        _verif("renfort.hauteur", "Hauteur au droit du pieu", f"d = {d:g} mm",
               None if hA is None or hB is None else max(hA, hB), d, "mm", etats["hauteur"][0], etats["hauteur"][1],
               ([] if hA is None else [_h_calc("M_A", r["renfort.M_A"], m, Dp, hA, c["h_A"], "h_calc,A", "D")])
               + ([] if hB is None else [_h_calc("M_B", r["renfort.M_B"], m, Dp, hB, c["h_B"], "h_calc,B", "D")]),
               texte=None if (hA is not None and hB is not None) else
               "moment négatif : le pieu est aussi large que la poutre, le classeur ne calcule pas (#NOMBRE!)"),
        _verif("renfort.sup", "Armature supérieure au droit du pieu", "brins horizontaux des étriers",
               max(oA, oB), oh, "mm²", etats["sup"][0], etats["sup"][1],
               [_omega("Ω_sup,A", "M_A", r["renfort.M_A"], m, "d", d, oA, c["om_A"]),
                _omega("Ω_sup,B", "M_B", r["renfort.M_B"], m, "d", d, oB, c["om_B"]), om_h_formule]),
        _verif("renfort.etriers", "Étriers au droit du pieu", piece_base,
               r["renfort.omega_etriers"], r["renfort.omega_etriers_choisie"], "mm²", etats["etriers"][0],
               etats["etriers"][1],
               [_fx("Ω_étriers", "P*1000/sa", (("P", sym_P, P), ("sa", "σ_adm", m["sigma_adm"])),
                    r["renfort.omega_etriers"], "mm²", 0, c["om_e"]), etriers_formule]),
    ]
    tau = _fx("τ", "P*1000/(0.75*b*d)", (("P", sym_P, P), ("b", "b", limite_h * 10), ("d", "d", d)),
              r["renfort.tau"], "N/mm²", 2, c["tau"])
    return {"cle": "renfort", "nom": "Renforcement au droit du pieu", "piece": f"pieu Ø{Dp:g} cm",
            "sollicitations": soll + [tau], "verifications": verifs,
            "informations": [{"nom": "τ au droit du pieu", "valeur": r["renfort.tau"], "unite": "N/mm²",
                              "texte": r["renfort.tau_texte"], "cellule": c["tau"]}]}


def _horizontal(f, r, cellules, gabarit, termes, etat) -> dict:
    """Le bloc « Effort horizontal » : effort par pied, section des perfos (taux de travail 150 N/mm²)."""
    c = cellules
    H = r["horizontal.H_pied"]
    return {"cle": "horizontal", "nom": "Effort horizontal", "piece": f"{r['horizontal.perfos_par_plaque']}Ø{f['perfos']['d']} par plaque",
            "sollicitations": [_fx("H_pied", gabarit, termes, H, "kN", 1, c["H"])],
            "verifications": [_verif(
                "perfos", "Perfos par pied", f"{f['perfos']['n']}Ø{f['perfos']['d']}",
                r["horizontal.omega_perfos"], r["horizontal.omega_perfos_choisie"], "mm²", etat, c["etat"],
                [_fx("Ω_perfos", "H*1000/150", (("H", "H_pied", H),), r["horizontal.omega_perfos"], "mm²", 0, c["om"]),
                 _omega_choisie("Ω_choisie", f["perfos"], r["horizontal.omega_perfos_choisie"], c["om_ch"])])],
            "informations": [{"nom": "Perfos autour des plaques de répartition",
                              "texte": f"{r['horizontal.perfos_par_plaque']}Ø{f['perfos']['d']} par plaque "
                                       f"(3 × {f['perfos']['n']})", "cellule": c["plaque"]}]}


def _un_pied(prefixe, r, L, e, P, pp, cel, avec_travee) -> list:
    """Les sollicitations de la poutre à un pied (cadre de 4 pieux, poutre secondaire) : CAS A puis CAS B."""
    a, b = f"{prefixe}.cas_A", f"{prefixe}.cas_B"
    T = (("P", "P", P), ("L", "L", L), ("pp", "pp", pp), ("e", "e", e))
    out = [
        _fx("R_A (CAS A)", "(P*L + pp*L**2/2)/(L - e)", T, r[f"{a}.RA"], "kN", 1, cel["A.RA"]),
        _fx("R_B (CAS A)", "(pp*(L - e)**2/2 - P*e - pp*e**2/2)/(L - e)", T, r[f"{a}.RB"], "kN", 1, cel["A.RB"]),
        _fx("M_min (CAS A)", "-P*e - pp*e**2/2", (("P", "P", P), ("e", "e", e), ("pp", "pp", pp)), r[f"{a}.M_min"],
            "kN·m", 1, cel["A.M"]),
        _fx("V_max (CAS A)", "P + pp*e", (("P", "P", P), ("pp", "pp", pp), ("e", "e", e)), r[f"{a}.V_max"], "kN", 1,
            cel["A.V"]),
        _fx("R_A (CAS B)", "(P*L + pp*(L + e)**2/2)/(L + e)", T, r[f"{b}.RA"], "kN", 1, cel["B.RA"]),
        _fx("R_B (CAS B)", "(P*e + pp*(L + e)**2/2)/(L + e)", T, r[f"{b}.RB"], "kN", 1, cel["B.RB"]),
        _fx("M_max (CAS B)", "RA*e - pp*e**2/2", (("RA", "R_A", r[f"{b}.RA"]), ("e", "e", e), ("pp", "pp", pp)),
            r[f"{b}.M_max"], "kN·m", 1, cel["B.M"]),
    ]
    if avec_travee:
        out.insert(4, _fx("V_travée (CAS A)", "max(abs(P + pp*0.8 - RA), abs(RB - pp*0.8))",
                          (("P", "P", P), ("pp", "pp", pp), ("RA", "R_A", r[f"{a}.RA"]), ("RB", "R_B", r[f"{a}.RB"])),
                          r[f"{a}.V_travee"], "kN", 1, cel["A.Vt"]))
        out.append(_fx("V_travée (CAS B)", "max(abs(RA - 0.8*pp - P), abs(RB - 0.8*pp))",
                       (("RA", "R_A", r[f"{b}.RA"]), ("pp", "pp", pp), ("P", "P", P), ("RB", "R_B", r[f"{b}.RB"])),
                       r[f"{b}.V_travee"], "kN", 1, cel["B.Vt"]))
    return out


# ================================================================== VAR4
def _cadre4(f: dict, r: K.Releve, m: dict) -> list:
    po = f["poutre"]
    L, e, P, pp = f["chassis"] / 100, f["e"] / 100, f["charges"]["P"], r["pp.poutre"]
    b, d = po["b"], r["section.poutre.d"]
    Mmax, Mmin, V, Vt = (r["poutre.dim.M_max"], r["poutre.dim.M_min"], r["poutre.dim.V_max"], r["poutre.dim.V_travee"])
    ea, et = po["etriers_appui"], po["etriers_travee"]
    soll = [_pp("pp", po["b"], po["h"], pp, "C21")] + _un_pied(
        "poutre", r, L, e, P, pp, {"A.RA": "C34", "A.RB": "C35", "A.M": "C37", "A.V": "C38", "A.Vt": "C39",
                                   "B.RA": "G34", "B.RB": "G35", "B.M": "G37", "B.Vt": "G39"}, True)
    M_h = max(abs(Mmax), abs(Mmin))
    verifs = [
        _verif("poutre.hauteur", "Hauteur de la poutre", f"d = {d:g} mm", r["section.poutre.h_calc"], d, "mm",
               r["etat.poutre.hauteur"], "G75",
               [_h_calc("max(|M_max| ; |M_min|)", M_h, m, b, r["section.poutre.h_calc"], "C53"),
                _fx("d", "h - 50", (("h", "h", po["h"] * 10),), d, "mm", 0, "G53")]),
        _verif("poutre.inf", "Armatures inférieures", _barres(po["inf"]), r["section.poutre.omega_inf"],
               r["section.poutre.omega_inf_choisie"], "mm²", r["etat.poutre.inf"], "G77",
               [_omega("Ω_inf", "M_max", Mmax, m, "d", d, r["section.poutre.omega_inf"], "C55"),
                _omega_choisie("Ω_choisie", po["inf"], r["section.poutre.omega_inf_choisie"], "G55")]),
        _verif("poutre.sup", "Armatures supérieures", _barres(po["sup"]), r["section.poutre.omega_sup"],
               r["section.poutre.omega_sup_choisie"], "mm²", r["etat.poutre.sup"], "G78",
               [_fx("Ω_sup", "abs(M)*1e6/(sa*0.9*(d - 50))", (("M", "M_min", Mmin), ("sa", "σ_adm", m["sigma_adm"]),
                                                             ("d", "d", d)), r["section.poutre.omega_sup"], "mm²", 0, "C59"),
                _omega_choisie("Ω_choisie", po["sup"], r["section.poutre.omega_sup_choisie"], "G59")]),
        _verif("poutre.etriers_appui", "Étriers à l'appui", _cadres(ea), ea["pas"], r["section.poutre.pas_appui"], "mm",
               r["etat.poutre.etriers_appui"], "G79",
               [_fx("pas calculé", "n*pi*(D/2)**2*sa*d*0.9/(V*1000)",
                    (("n", "n", ea["brins"]), ("D", "Ø", ea["d"]), ("sa", "σ_adm", m["sigma_adm"]), ("d", "d", d),
                     ("V", "V_max", V)), r["section.poutre.pas_appui"], "mm", 0, "C66")]),
        _verif("poutre.etriers_travee", "Étriers en travée", _cadres(et), et["pas"], r["section.poutre.pas_travee"], "mm",
               r["etat.poutre.etriers_travee"], "G80",
               [_fx("pas calculé", "n*pi*(D/2)**2*sa*d*0.9/(V*1000)",
                    (("n", "n", et["brins"]), ("D", "Ø", et["d"]), ("sa", "σ_adm", m["sigma_adm"]), ("d", "d", d),
                     ("V", "V_travée", Vt)), r["section.poutre.pas_travee"], "mm", 0, "C70")]),
    ]
    tau = _fx("τ", "V*k/100*1000/(0.75*b*d)", (("V", "V_max", V), ("k", "part de V (%)", po["part_V"]),
                                               ("b", "b", b * 10), ("d", "d", d)), r["section.poutre.tau"], "N/mm²", 2, "C63")
    poutre = {"cle": "poutre", "nom": "Poutre", "piece": f"{po['b']:g} × {po['h']:g} cm",
              "sollicitations": soll, "verifications": verifs,
              "informations": [{"nom": "τ cisaillement", "valeur": r["section.poutre.tau"], "unite": "N/mm²",
                                "texte": r["section.poutre.tau_texte"], "cellule": "C63", "formule": tau}]}
    pieux = {"cle": "pieux", "nom": "Capacité portante des pieux", "piece": f"4 pieux Ø{f['pieux']['D']:g} cm",
             "sollicitations": [], "informations": [],
             "verifications": [_verif(
                 "pieux", "Charge sur un pieu", f"{r['pieux.capacite_T']:g} T", r["pieux.charge"], f["pieux"]["capacite"],
                 "kN", r["etat.pieux"], "G88",
                 [_fx("R_pieu", "R + L/2*pp", (("R", "R_max", r["poutre.dim.R_max"]), ("L", "L", L), ("pp", "pp", pp)),
                      r["pieux.charge"], "kN", 1, "C84")])]}
    oh = _fx("Ω_brins horizontaux", "pi*(D/2)**2*n/2*(Dp/2/s)",
             (("D", "Ø", ea["d"]), ("n", "n", ea["brins"]), ("Dp", "D", f["pieux"]["D"] * 10), ("s", "pas", ea["pas"])),
             r["renfort.omega_brins_horizontaux"], "mm²", 0, "G108")
    oe = _omega_choisie("Ω_choisie", {"n": f["renfort"]["brins"], "d": f["renfort"]["d"]},
                        r["renfort.omega_etriers_choisie"], "G114")
    renfort = _renfort(
        f, r, m, po["b"], f["pieux"]["D"], "V_max", f"{f['renfort']['brins']} brins Ø{f['renfort']['d']}",
        {"p1": "C95", "p2a": "C96", "p2b": "C97", "M_A": "C101", "M_B": "C102", "h_A": "C105", "h_B": "C106",
         "om_A": "C108", "om_B": "C109", "om_e": "C114", "tau": "C112"},
        {"hauteur": (r["etat.renfort.hauteur"], "G120"), "sup": (r["etat.renfort.sup"], "G122"),
         "etriers": (r["etat.renfort.etriers"], "G123")}, oh, oe, po["b"])
    ch = f["charges"]
    horizontal = _horizontal(
        f, r, {"H": "C130", "om": "C133", "om_ch": "G133", "plaque": "D139", "etat": "G139"},
        "max(Hh/3, Hs/3 + M/2/L/100)",
        (("Hh", "H_hs", ch["H_hs"]), ("Hs", "H_s", ch["H_s"]), ("M", "M_s", ch["M_s"]),
         ("L", "L (cm)", f["chassis"])), r["etat.perfos"])
    return [poutre, pieux, renfort, horizontal]


# ================================================================== VAR8
def _diagonales8(f: dict, r: K.Releve, m: dict) -> list:
    se, pr = f["secondaire"], f["primaire"]
    L, e, P, pp = f["chassis"] / 100, f["e_secondaire"] / 100, f["charges"]["P"], r["pp.secondaire"]
    d = r["section.secondaire.d"]
    Mmax, Mmin, V = r["secondaire.dim.M_max"], r["secondaire.dim.M_min"], r["secondaire.dim.V_max"]
    es = se["etriers"]
    soll = [_pp("pp", se["b"], se["h"], pp, "C22")] + _un_pied(
        "secondaire", r, L, e, P, pp, {"A.RA": "C36", "A.RB": "C37", "A.M": "C39", "A.V": "C40",
                                       "B.RA": "G36", "B.RB": "G37", "B.M": "G39"}, False)
    verifs = [
        _verif("secondaire.hauteur", "Hauteur de la poutre secondaire", f"d = {d:g} mm", r["section.secondaire.h_calc"],
               d, "mm", r["etat.secondaire.hauteur"], "G86",
               [_h_calc("max(|M_max| ; |M_min|)", max(abs(Mmax), abs(Mmin)), m, se["b"], r["section.secondaire.h_calc"], "C68"),
                _fx("d", "h - 50", (("h", "h", se["h"] * 10),), d, "mm", 0, "G68")]),
        _verif("secondaire.inf", "Armatures inférieures", _barres(se["inf"]), r["section.secondaire.omega_inf"],
               r["section.secondaire.omega_inf_choisie"], "mm²", r["etat.secondaire.inf"], "G88",
               [_omega("Ω_inf", "M_max", Mmax, m, "d", d, r["section.secondaire.omega_inf"], "C70"),
                _omega_choisie("Ω_choisie", se["inf"], r["section.secondaire.omega_inf_choisie"], "G70")]),
        _verif("secondaire.sup", "Armatures supérieures", _barres(se["sup"]), r["section.secondaire.omega_sup"],
               r["section.secondaire.omega_sup_choisie"], "mm²", r["etat.secondaire.sup"], "G89",
               [_fx("Ω_sup", "abs(M)*1e6/(sa*0.9*(d - 50))", (("M", "M_min", Mmin), ("sa", "σ_adm", m["sigma_adm"]),
                                                             ("d", "d", d)), r["section.secondaire.omega_sup"], "mm²", 0, "C74"),
                _omega_choisie("Ω_choisie", se["sup"], r["section.secondaire.omega_sup_choisie"], "G74")]),
        _verif("secondaire.etriers", "Étriers", _cadres(es), es["pas"], r["section.secondaire.pas"], "mm",
               r["etat.secondaire.etriers"], "G90",
               [_fx("pas calculé", "n*pi*(D/2)**2*sa*d*0.9/(V*1000*k/100)",
                    (("n", "n", es["brins"]), ("D", "Ø", es["d"]), ("sa", "σ_adm", m["sigma_adm"]), ("d", "d", d),
                     ("V", "V_max", V), ("k", "part de V (%)", se["part_V"])), r["section.secondaire.pas"], "mm", 0, "C81")]),
    ]
    tau = _fx("τ", "V*k/100*1000/(0.75*b*d)", (("V", "V_max", V), ("k", "part de V (%)", se["part_V"]),
                                               ("b", "b", se["b"] * 10), ("d", "d", d)), r["section.secondaire.tau"],
              "N/mm²", 2, "C78")
    secondaire = {"cle": "secondaire", "nom": "Poutre secondaire", "piece": f"{se['b']:g} × {se['h']:g} cm",
                  "sollicitations": soll, "verifications": verifs,
                  "informations": [{"nom": "τ cisaillement", "valeur": r["section.secondaire.tau"], "unite": "N/mm²",
                                    "texte": r["section.secondaire.tau_texte"], "cellule": "C78", "formule": tau}]}

    Lp, s, e2, Pp, pp2 = pr["L"] / 100, f["pieux"]["entraxe"] / 100, f["e_primaire"] / 100, r["primaire.charge"], r["pp.primaire"]
    T = (("P", "P", Pp), ("s", "s", s), ("e", "e", e2), ("pp", "pp", pp2), ("Lp", "L_p", Lp))
    A, B = "primaire.cas_A", "primaire.cas_B"
    soll_p = [
        _pp("pp", pr["b"], pr["h"], pp2, "C23"),
        _fx("P", "R + pp*L/2", (("R", "R_max secondaire", r["secondaire.dim.R_max"]), ("pp", "pp secondaire", pp),
                                ("L", "L", L)), Pp, "kN", 1, "C50"),
        _fx("R_A (CAS A)", "P*((s - e)/2 + e/2)/(s - e) + pp*Lp/2", T, r[f"{A}.RA"], "kN", 1, "C55"),
        _fx("R_B (CAS A)", "P*((s - e)/2 - e/2)/(s - e) + pp*Lp/2", T, r[f"{A}.RB"], "kN", 1, "C56"),
        _fx("M_max (CAS A)", "RA*((s - e)/2 - e/2) - pp*(Lp/2 - e/2)**2/2",
            (("RA", "R_A", r[f"{A}.RA"]), ("s", "s", s), ("e", "e", e2), ("pp", "pp", pp2), ("Lp", "L_p", Lp)),
            r[f"{A}.M_max"], "kN·m", 1, "C58"),
        _fx("V_max (CAS A)", "RA - pp*(Lp - (s - e))/2",
            (("RA", "R_A", r[f"{A}.RA"]), ("pp", "pp", pp2), ("Lp", "L_p", Lp), ("s", "s", s), ("e", "e", e2)),
            r[f"{A}.V_max"], "kN", 1, "C59"),
        _fx("R_A = R_B (CAS B)", "P/2 + pp*Lp/2", (("P", "P", Pp), ("pp", "pp", pp2), ("Lp", "L_p", Lp)), r[f"{B}.RA"],
            "kN", 1, "G55"),
        _fx("M_max (CAS B)", "RA*(s/2 + e/2) - pp*(Lp/2)**2/2",
            (("RA", "R_A", r[f"{B}.RA"]), ("s", "s", s), ("e", "e", e2), ("pp", "pp", pp2), ("Lp", "L_p", Lp)),
            r[f"{B}.M_max"], "kN·m", 1, "G58"),
        _fx("V_max (CAS B)", "RA - pp*(Lp - s - e)/2",
            (("RA", "R_A", r[f"{B}.RA"]), ("pp", "pp", pp2), ("Lp", "L_p", Lp), ("s", "s", s), ("e", "e", e2)),
            r[f"{B}.V_max"], "kN", 1, "G59"),
    ]
    dp, Mp, Vp, ep = r["section.primaire.d"], r["primaire.dim.M_max"], r["primaire.dim.V_max"], pr["etriers"]
    verifs_p = [
        _verif("primaire.hauteur", "Hauteur de la poutre primaire", f"d = {dp:g} mm", r["section.primaire.h_calc"], dp,
               "mm", r["etat.primaire.hauteur"], "G110",
               [_h_calc("M_max", Mp, m, pr["b"], r["section.primaire.h_calc"], "C93"),
                _fx("d", "h - 75", (("h", "h", pr["h"] * 10),), dp, "mm", 0, "G93")]),
        _verif("primaire.inf", "Armatures inférieures", _barres(pr["inf"]), r["section.primaire.omega_inf"],
               r["section.primaire.omega_inf_choisie"], "mm²", r["etat.primaire.inf"], "G112",
               [_omega("Ω_inf", "M_max", Mp, m, "d", dp, r["section.primaire.omega_inf"], "C95"),
                _omega_choisie("Ω_choisie", pr["inf"], r["section.primaire.omega_inf_choisie"], "G95")]),
        _verif("primaire.sup", "Armatures supérieures", _barres(pr["sup"]), r["section.primaire.omega_sup"],
               r["section.primaire.omega_sup_choisie"], "mm²", r["etat.primaire.sup"], "G113",
               [_fx("Ω_sup", "0.4*O", (("O", "Ω_inf", r["section.primaire.omega_inf"]),), r["section.primaire.omega_sup"],
                    "mm²", 0, "C99"),
                _omega_choisie("Ω_choisie", pr["sup"], r["section.primaire.omega_sup_choisie"], "G99")]),
        _verif("primaire.etriers", "Étriers", _cadres(ep), ep["pas"], r["section.primaire.pas"], "mm",
               r["etat.primaire.etriers"], "G114",
               [_fx("pas calculé", "n*pi*(D/2)**2*sa*d*0.9/(V*1000)",
                    (("n", "n", ep["brins"]), ("D", "Ø", ep["d"]), ("sa", "σ_adm", m["sigma_adm"]), ("d", "d", dp),
                     ("V", "V_max", Vp)), r["section.primaire.pas"], "mm", 0, "C105")]),
    ]
    tau_p = _fx("τ", "V*1000/(0.75*b*d)", (("V", "V_max", Vp), ("b", "b", pr["b"] * 10), ("d", "d", dp)),
                r["section.primaire.tau"], "N/mm²", 2, "C103")
    primaire = {"cle": "primaire", "nom": "Poutre primaire", "piece": f"{pr['b']:g} × {pr['h']:g} cm, L = {pr['L']:g} cm",
                "sollicitations": soll_p, "verifications": verifs_p,
                "informations": [{"nom": "τ cisaillement", "valeur": r["section.primaire.tau"], "unite": "N/mm²",
                                  "texte": r["section.primaire.tau_texte"], "cellule": "C103", "formule": tau_p},
                                 {"nom": "Réaction maximale d'un pieu", "valeur": r["primaire.dim.R_max"], "unite": "kN",
                                  "texte": "calculée par le classeur, comparée à aucune capacité", "cellule": "C62"}]}
    oh = _fx("Ω_brins horizontaux", "pi*(D/2)**2*n/2*(Dp/2/s)",
             (("D", "Ø", ep["d"]), ("n", "n", ep["brins"]), ("Dp", "D", f["pieux"]["D"] * 10), ("s", "pas", ep["pas"])),
             r["renfort.omega_brins_horizontaux"], "mm²", 0, "G133")
    oe = _omega_choisie("Ω_choisie", {"n": f["renfort"]["brins"], "d": f["renfort"]["d"]},
                        r["renfort.omega_etriers_choisie"], "G139")
    renfort = _renfort(
        f, r, m, pr["b"], f["pieux"]["D"], "V_max", f"{f['renfort']['brins']} brins Ø{f['renfort']['d']}",
        {"p1": "C120", "p2a": "C121", "p2b": "C122", "M_A": "C126", "M_B": "C127", "h_A": "C130", "h_B": "C131",
         "om_A": "C133", "om_B": "C134", "om_e": "C139", "tau": "C137"},
        {"hauteur": (r["etat.renfort.hauteur"], "G145"), "sup": (r["etat.renfort.sup"], "G147"),
         "etriers": (r["etat.renfort.etriers"], "G148")}, oh, oe, pr["b"])
    ch = f["charges"]
    horizontal = _horizontal(
        f, r, {"H": "C155", "om": "C158", "om_ch": "G158", "plaque": "D164", "etat": "G164"},
        "max(Hh/3, Hs/3 + M/2/L/100)",
        (("Hh", "H_hs", ch["H_hs"]), ("Hs", "H_s", ch["H_s"]), ("M", "M_s", ch["M_s"]),
         ("L", "L (cm)", f["chassis"])), r["etat.perfos"])
    return [secondaire, primaire, renfort, horizontal]


# ================================================================== BASE / GWW
def _colonne(r: K.Releve, g: str):
    """Le sous-cas et la colonne qui donnent la valeur dimensionnante ``g`` de la poutre support."""
    A, B = r.cas_support["A"][g], r.cas_support["B"][g]
    src = A if A["valeur"] >= B["valeur"] else B
    if g == "RB_resp":
        cle = "A" if r.cas_support["A"]["RB"]["valeur"] == r["support.dim.RB_max"] else "B"
        src = r.cas_support[cle][g]
    t = r.tableaux[src["sous_cas"]]
    j = t["max"][g]["colonne"]
    return t, (None if j is None else t["colonnes"][j]), src


def _semelles8(f: dict, r: K.Releve, m: dict) -> list:
    su, sm = f["support"], f["semelle"]
    pp, pps = r["pp.support"], r["pp.semelle"]

    def geo(t):
        g = t["geometrie"]
        return (("Lc", "L_c", g["Lc"]), ("a", "a'", g["a"]), ("b", "b'", g["b"]), ("L", "L", g["L"]), ("pp", "pp", g["pp"]))

    def cas(src):
        return f"{src['sous_cas']} {src['libelle'] or ''}".strip()

    soll = [_pp("pp", su["b"], su["h"], pp, "C43")]
    t, col, src = _colonne(r, "RA")
    P = (("P1", "P_1", col["P1"]), ("P2", "P_2", col["P2"]))
    soll.append(_fx(f"R_A,max — {cas(src)}", "(P1*(Lc + b) + P2*b)/L + pp*L/2", P + geo(t), r["support.dim.RA_max"],
                    "kN", 1, "C90"))
    t, col, src = _colonne(r, "RB")
    P = (("P1", "P_1", col["P1"]), ("P2", "P_2", col["P2"]))
    soll.append(_fx(f"R_B,max — {cas(src)}", "(P2*(Lc + a) + P1*a)/L + pp*L/2", P + geo(t), r["support.dim.RB_max"],
                    "kN", 1, "C91"))
    t, col, src = _colonne(r, "V_central")
    soll.append(_fx(f"V_central — {cas(src)}", "max(RA - pp*a - P1, RB - pp*b - P2)",
                    (("RA", "R_A", col["RA"]), ("pp", "pp", pp), ("a", "a'", t["geometrie"]["a"]), ("P1", "P_1", col["P1"]),
                     ("RB", "R_B", col["RB"]), ("b", "b'", t["geometrie"]["b"]), ("P2", "P_2", col["P2"])),
                    r["support.dim.V_central"], "kN", 1, "C97"))
    t, col, src = _colonne(r, "M_max")
    g = t["geometrie"]
    i = max(range(K.N_X), key=lambda k: col["M"][k])
    x = i / 20
    if x > g["a"] + g["Lc"]:
        gab, T = "RA*x - pp*x**2/2 - P1*(x - a) - P2*(x - a - Lc)", (("P2", "P_2", col["P2"]), ("Lc", "L_c", g["Lc"]))
    elif x > g["a"]:
        gab, T = "RA*x - pp*x**2/2 - P1*(x - a)", ()
    else:
        gab, T = "RA*x - pp*x**2/2", ()
    T = (("RA", "R_A", col["RA"]), ("x", "x", x), ("pp", "pp", pp)) + ((("P1", "P_1", col["P1"]), ("a", "a'", g["a"]))
                                                                      if x > g["a"] else ()) + T
    soll.append(_fx(f"M_max — {cas(src)}, x = {f'{x:.2f}'.replace('.', ',')} m", gab, T, r["support.dim.M_max"], "kN·m", 1, "C94"))
    t, col, src = _colonne(r, "M_max_eq")
    g = t["geometrie"]
    soll.append(_fx(f"M_hor,éq — {cas(src)}", "max(((b + Lc) + b)*a*H/L, ((a + Lc) + a)*b*H/L)*h/bp*100/125",
                    (("b", "b'", g["b"]), ("Lc", "L_c", g["Lc"]), ("a", "a'", g["a"]), ("H", "H/4", col["H4"]),
                     ("L", "L", g["L"]), ("h", "h", su["h"] / 100), ("bp", "b_poutre", su["b"] / 100)),
                    col["M_hor_eq"], "kN·m", 2, None))
    soll.append(_fx("M_max,éq", "Mh + M", (("Mh", "M_hor,éq", col["M_hor_eq"]), ("M", "M_max", col["M_max"])),
                    r["support.dim.M_max_eq"], "kN·m", 1, "C95"))

    d = r["section.support.d"]
    ee, ec = su["etriers_extremites"], su["etriers_centre"]
    inf_max = max(gp["d"] if gp["n"] > 0 else 0 for gp in su["inf"])
    V, Vc, k = r["support.dim.V_max"], r["support.dim.V_central"], su["part_V"]
    verifs = [
        _verif("support.hauteur", "Hauteur de la poutre support", f"d = {d:g} mm", r["section.support.h_calc"], d, "mm",
               r["etat.support.hauteur"], "G161",
               [_h_calc("M_max", r["support.dim.M_max"], m, su["b"], r["section.support.h_calc"], "C126"),
                _fx("d", "h - c - De - Di/2", (("h", "h", su["h"] * 10), ("c", "enrobage", f["enrobage"]),
                                               ("De", "Ø_étrier", max(ee["d"], ec["d"])), ("Di", "Ø_inf", inf_max)),
                    d, "mm", 0, "G126")]),
        _verif("support.inf", "Armatures inférieures", _barres(su["inf"]), r["section.support.omega_inf"],
               r["section.support.omega_inf_choisie"], "mm²", r["etat.support.inf"], "G165",
               [_omega("Ω_inf", "M_max,éq", r["support.dim.M_max_eq"], m, "d", d, r["section.support.omega_inf"], "C128"),
                _omega_choisie("Ω_choisie", su["inf"], r["section.support.omega_inf_choisie"], "G128")]),
        _verif("support.sup", "Armatures supérieures", _barres(su["sup"]), r["section.support.omega_sup"],
               r["section.support.omega_sup_choisie"], "mm²", r["etat.support.sup"], "G166",
               [_fx("Ω_sup", "0.25*O", (("O", "Ω_inf", r["section.support.omega_inf"]),), r["section.support.omega_sup"],
                    "mm²", 0, "C135"),
                _omega_choisie("Ω_choisie", su["sup"], r["section.support.omega_sup_choisie"], "G135")]),
        _verif("support.etriers_extremites", "Étriers aux extrémités", _cadres(ee), ee["pas"],
               r["section.support.pas_extremites"], "mm", r["etat.support.etriers_extremites"], "G167",
               [_fx("hauteur de l'étrier", "h - 2*c - D", (("h", "h", su["h"] * 10), ("c", "enrobage", f["enrobage"]),
                                                        ("D", "Ø", ee["d"])),
                    r["section.support.hauteur_etrier_extremites"], "mm", 0, "G149"),
                _fx("pas calculé", "n*pi*(D/2)**2*sa*z/(V*1000*k/100)",
                    (("n", "n", ee["brins"]), ("D", "Ø", ee["d"]), ("sa", "σ_adm", m["sigma_adm"]),
                     ("z", "hauteur de l'étrier", r["section.support.hauteur_etrier_extremites"]), ("V", "V_max", V),
                     ("k", "part de V (%)", k)), r["section.support.pas_extremites"], "mm", 0, "C145")]),
        _verif("support.etriers_centre", "Étriers en partie centrale", _cadres(ec), ec["pas"],
               r["section.support.pas_centre"], "mm", r["etat.support.etriers_centre"], "G168",
               [_fx("pas calculé", "n*pi*(D/2)**2*sa*(d - c)/(V*1000*k/100)",
                    (("n", "n", ec["brins"]), ("D", "Ø", ec["d"]), ("sa", "σ_adm", m["sigma_adm"]), ("d", "d", d),
                     ("c", "enrobage", f["enrobage"]), ("V", "V_central", Vc), ("k", "part de V (%)", k)),
                    r["section.support.pas_centre"], "mm", 0, "C152")]),
    ]
    tau = _fx("τ", "V*k/100*1000/(0.75*b*d)", (("V", "V_max", V), ("k", "part de V (%)", k), ("b", "b", su["b"] * 10),
                                               ("d", "d", d)), r["section.support.tau"], "N/mm²", 2, "C142")
    infos = [{"nom": "τ cisaillement", "valeur": r["section.support.tau"], "unite": "N/mm²",
              "texte": r["section.support.tau_zone"], "zone": r["etat.support.tau_zone"], "cellule": "C142", "formule": tau},
             {"nom": "Armature latérale", "texte": "prévoir 2Ø12 tous les 30 à 40 cm (prescription du classeur)",
              "cellule": "H169"}]
    reac = "paroi de pieux sécants" if f["cote_b"] == E.COTE_PAROI else "appui du côté b"
    infos += [{"nom": f"R_B,max ({reac})", "valeur": r["support.dim.RB_max"], "unite": "kN", "cellule": "C91"},
              {"nom": "R_B respectif (autre poutre, même cas)", "valeur": r["support.dim.RB_resp"], "unite": "kN", "cellule": "C92"},
              {"nom": "R_B symétrique", "valeur": r["support.dim.RB_symm"], "unite": "kN", "cellule": "C93"}]
    support = {"cle": "support", "nom": "Poutre support châssis",
               "piece": f"{su['b']:g} × {su['h']:g} cm, portée {r['geometrie.longueur_support']:g} cm",
               "sollicitations": soll, "verifications": verifs, "informations": infos}

    # --- semelle sur pieux
    Ls, s, e, c, Pc = sm["L"] / 100, f["pieux"]["entraxe"] / 100, f["e_semelle"] / 100, f["c_appui"] / 100, r["semelle.charge"]
    T = (("P", "P", Pc), ("s", "s", s), ("e", "e", e), ("pp", "pp", pps), ("Ls", "L_s", Ls))
    i_t = max(range(K.N_T), key=lambda k_: r.M_semelle[k_])
    tt = i_t / 100
    soll_s = [
        _pp("pp", sm["b"], sm["h"], pps, "C44"),
        _fx("P", "R", (("R", "R_A,max poutre support", r["support.dim.RA_max"]),), Pc, "kN", 1, "C104"),
        _fx("R_A (CAS A)", "P*(s/2)/(s - e) + pp*Ls/2", T, r["semelle.cas_A.RA"], "kN", 1, "C112"),
        _fx("R_B (CAS A)", "P*(s/2 - e)/(s - e) + pp*Ls/2", T, r["semelle.cas_A.RB"], "kN", 1, "C113"),
        _fx(f"M_max (CAS A), t = {tt:.2f}".replace(".", ","), "RA*(b + t*c) - pp*(a + t*c)**2/2 - P*t**2*c/2",
            (("RA", "R_A", r["semelle.cas_A.RA"]), ("b", "b_0", r["semelle.cas_A.feuille.b"]), ("t", "t", tt),
             ("c", "c", c), ("pp", "pp", pps), ("a", "a_0", r["semelle.cas_A.feuille.a"]), ("P", "P", Pc)),
            r["semelle.cas_A.M_max"], "kN·m", 1, "C115"),
        _fx("V_max (CAS A)", "RA - pp*(Ls - (s - e))/2",
            (("RA", "R_A", r["semelle.cas_A.RA"]), ("pp", "pp", pps), ("Ls", "L_s", Ls), ("s", "s", s), ("e", "e", e)),
            r["semelle.cas_A.V_max"], "kN", 1, "C116"),
        _fx("R_A = R_B (CAS B)", "P/2 + pp*Ls/2", (("P", "P", Pc), ("pp", "pp", pps), ("Ls", "L_s", Ls)),
            r["semelle.cas_B.RA"], "kN", 1, "G112"),
        _fx("M_max (CAS B)", "RA*(s/2 + e/2) - pp*(Ls/2)**2/2 - P*c/8",
            (("RA", "R_A", r["semelle.cas_B.RA"]), ("s", "s", s), ("e", "e", e), ("pp", "pp", pps), ("Ls", "L_s", Ls),
             ("P", "P", Pc), ("c", "c", c)), r["semelle.cas_B.M_max"], "kN·m", 1, "G115"),
        _fx("V_max (CAS B)", "RA - pp*(Ls - s - e)/2",
            (("RA", "R_A", r["semelle.cas_B.RA"]), ("pp", "pp", pps), ("Ls", "L_s", Ls), ("s", "s", s), ("e", "e", e)),
            r["semelle.cas_B.V_max"], "kN", 1, "G116"),
    ]
    ds, es = r["section.semelle.d"], sm["etriers"]
    Ms, Vs = r["semelle.dim.M_max"], r["semelle.dim.V_max"]
    sinf_max = max(gp["d"] if gp["n"] > 0 else 0 for gp in sm["inf"])
    verifs_s = [
        _verif("semelle.hauteur", "Hauteur de la semelle", f"d = {ds:g} mm", r["section.semelle.h_calc"], ds, "mm",
               r["etat.semelle.hauteur"], "G197",
               [_h_calc("M_max", Ms, m, sm["b"], r["section.semelle.h_calc"], "C172"),
                _fx("d", "h - c - De - Di/2", (("h", "h", sm["h"] * 10), ("c", "enrobage", f["enrobage"]),
                                               ("De", "Ø_étrier", es["d"]), ("Di", "Ø_inf", sinf_max)), ds, "mm", 0, "G172")]),
        _verif("semelle.inf", "Armatures inférieures", _barres(sm["inf"]), r["section.semelle.omega_inf"],
               r["section.semelle.omega_inf_choisie"], "mm²", r["etat.semelle.inf"], "G201",
               [_omega("Ω_inf", "M_max", Ms, m, "d", ds, r["section.semelle.omega_inf"], "C174"),
                _omega_choisie("Ω_choisie", sm["inf"], r["section.semelle.omega_inf_choisie"], "G174")]),
        _verif("semelle.sup", "Armatures supérieures", _barres(sm["sup"]), r["section.semelle.omega_sup"],
               r["section.semelle.omega_sup_choisie"], "mm²", r["etat.semelle.sup"], "G202",
               [_fx("Ω_sup", "0.25*O", (("O", "Ω_inf", r["section.semelle.omega_inf"]),), r["section.semelle.omega_sup"],
                    "mm²", 0, "C180"),
                _omega_choisie("Ω_choisie", sm["sup"], r["section.semelle.omega_sup_choisie"], "G180")]),
        _verif("semelle.etriers", "Étriers", _cadres(es), es["pas"], r["section.semelle.pas"], "mm",
               r["etat.semelle.etriers"], "G203",
               [_fx("pas calculé", "n*pi*(D/2)**2*sa*(d - c)/(V*1000)",
                    (("n", "n", es["brins"]), ("D", "Ø", es["d"]), ("sa", "σ_adm", m["sigma_adm"]), ("d", "d", ds),
                     ("c", "enrobage", f["enrobage"]), ("V", "V_max", Vs)), r["section.semelle.pas"], "mm", 0, "C189")]),
    ]
    tau_s = _fx("τ", "V*1000/(0.75*b*d)", (("V", "V_max", Vs), ("b", "b", sm["b"] * 10), ("d", "d", ds)),
                r["section.semelle.tau"], "N/mm²", 2, "C187")
    semelle = {"cle": "semelle", "nom": "Semelle sur pieux",
               "piece": f"{sm['L']:g} × {sm['b']:g} × {sm['h']:g} cm, 2 pieux à {f['pieux']['entraxe']:g} cm",
               "sollicitations": soll_s, "verifications": verifs_s,
               "informations": [{"nom": "τ cisaillement", "valeur": r["section.semelle.tau"], "unite": "N/mm²",
                                 "texte": r["section.semelle.tau_zone"], "zone": r["etat.semelle.tau_zone"],
                                 "cellule": "C187", "formule": tau_s},
                                {"nom": "Armature latérale", "cellule": "H204",
                                 "texte": "prévoir 2Ø12 tous les 30 à 40 cm (prescription du classeur)"},
                                {"nom": "Réaction maximale d'un pieu R_max", "valeur": r["semelle.dim.R_max"], "unite": "kN",
                                 "texte": "calculée par le classeur, non utilisée ensuite", "cellule": "C119"}]}
    pieux = {"cle": "pieux", "nom": "Capacité portante des pieux", "sollicitations": [], "informations": [],
             "piece": f"{4 if f['cote_b'] == E.COTE_PAROI else 8} pieux Ø{f['pieux']['D']:g} cm",
             "verifications": [_verif(
                 "pieux", "Sollicitation d'un pieu", f"{f['pieux']['capacite']:g} T", r["pieux.sollicitation_T"],
                 r["pieux.capacite_T"], "T", r["etat.pieux"], "G241",
                 [_fx("sollicitation", "V/10", (("V", "V_max semelle", Vs),), r["pieux.sollicitation_T"], "T", 1, "C222")],
                 texte="OK si sollicitation × 1,10 < capacité ; LIMITE si × 1,05 < capacité")]}
    oh = _fx("Ω_brins horizontaux", "pi*(D/2)**2*(Dp/s)",
             (("D", "Ø", es["d"]), ("Dp", "D", f["pieux"]["D"] * 10), ("s", "pas", es["pas"])),
             r["renfort.omega_brins_horizontaux"], "mm²", 0, "G227")
    re = f["renfort"]
    oe = _fx("Ω_choisie", "pi*(nb*(Db/2)**2) + pi*(nc*(Dc/2)**2)",
             (("nb", "n_base", re["brins_base"]), ("Db", "Ø_base", es["d"]), ("nc", "n_compl", re["brins"]),
              ("Dc", "Ø_compl", re["d"])), r["renfort.omega_etriers_choisie"], "mm²", 0, "G233")
    renfort = _renfort(
        f, r, m, sm["b"], f["pieux"]["D"], "V_max", f"{re['brins']} brins Ø{re['d']} + {re['brins_base']} brins Ø{es['d']} de base",
        {"p1": "C212", "p2a": "C213", "p2b": "C214", "M_A": "C218", "M_B": "C219", "h_A": "C224", "h_B": "C225",
         "om_A": "C227", "om_B": "C228", "om_e": "C233", "tau": "C231"},
        {"hauteur": (r["etat.renfort.hauteur"], "G243"), "sup": (r["etat.renfort.sup"], "G245"),
         "etriers": (r["etat.renfort.etriers"], "G246")}, oh, oe, sm["b"])
    ch = f["charges"]
    horizontal = _horizontal(
        f, r, {"H": "C254", "om": "C257", "om_ch": "G257", "plaque": "D263", "etat": "G263"},
        "max(Hh/3, Hs/3 + M/(2*L/s45))",
        (("Hh", "H_hs", ch["H_hs"]), ("Hs", "H_s", ch["H_s"]), ("M", "M_s", ch["M_s"]),
         ("L", "L", f["chassis"] / 100), ("s45", "sin 45°", math.sin(math.radians(45)))), r["etat.perfos"])
    return [support, semelle, pieux, renfort, horizontal]


_PAR_TYPE = {E.CADRE4: _cadre4, E.DIAGONALES8: _diagonales8, E.SEMELLES8: _semelles8}


def blocs(f: dict, r: K.Releve, m: dict) -> list:
    """Les blocs de résultats de la fondation NORMALISÉE ``f`` (relevé ``r``, matériaux ``m``), dans l'ordre de la
    fiche de calcul du classeur."""
    return _PAR_TYPE[f["type"]](f, r, m)

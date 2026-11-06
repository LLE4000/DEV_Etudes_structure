# -*- coding: utf-8 -*-
"""Maquettes du site « Stab Méthodes » : l'établi « Poutres béton armé », bâti avec les MÊMES composants que le
garde-corps (cadres flottants, ligne de résultat sur deux lignes, grande fenêtre de calcul, sélecteur) — c'est la règle
d'homogénéité entre modules. Les chiffres sont ceux du module poutre de Streamlit (modules/poutre.py), piloté sans écran
par extraire_donnees.py ; l'exemple est une poutre 30 × 50 avec une section en travée et une sur appui.
"""
from construire import (D, MOD, NBSP, barre_haut, cartouche, chip, elements_rapport, esc, etabli, fenetre_calcul, feuilles_dessin,
                        go, icon, legende, ligne, ligne_res, nb, num, pc, section, seg, sel, selecteur, statut, SCRIPT_ETABLI)


def _champ_txt(v: str) -> str:
    return f'<div class="field text"><input value="{v}"></div>'


def _secard(s: dict, actif: bool) -> str:
    """Une section de la poutre : son rôle et ses sollicitations (M en travée, M sur appui, V)."""
    def f(lab, v, u):
        # le libellé dans son propre span : seul, le texte et son indice seraient deux cases de la colonne
        return f'<label><span>{lab}</span><div class="field"><input value="{v}"><span class="u">{u}</span></div></label>'
    return (f'<div class="secard{" on" if actif else ""}"><div class="sh2"><span class="indice">{s["nom"]}</span>{s["role"]}'
            f'<span class="muted">· {s["inf"]} / {s["sup"]}</span><span class="pct {s["niveau"]}">{pc(s["eta"])}{NBSP}%</span></div>'
            f'<div class="trio">{f("M<sub>inf</sub>", s["M_inf"], "kN·m")}{f("M<sub>sup</sub>", s["M_sup"], "kN·m")}{f("V", s["V"], "kN")}</div></div>')


def _court(v: dict, s: dict) -> str:
    """La pièce sur la ligne du résultat : courte (la phrase complète est dans la fenêtre de calcul)."""
    return f"V = {s['V']} kN" if v["cle"] == "tau" else v["spec"]


def _lit(lab: str, n: int, d: int) -> str:
    return (f'<div class="lit"><span>{lab}</span><div class="field"><input value="{n}"><span class="u">barres</span></div>'
            f'<div class="field select"><span class="v">Ø{d}</span>{icon("chevron-down")}</div></div>')


def poutre() -> tuple:
    p = D["poutre"]
    hyp, dft = p["hyp"], p["defauts"]
    A, B = p["sections"]
    # le rapport : trois poutres (une conforme, une qui ne tient pas, une à compléter) et un garde-corps d'un autre module
    items = [("poutre", [("V1", f"{hyp['b']} × {hyp['h']} · {hyp['beton']}", p["eta"], p["niveau"], True, ""),
                         ("V2", "25 × 45 · C25/30", p["v2"]["eta"], p["v2"]["niveau"], False, ""),
                         ("V3", "sollicitations à saisir", None, "c", False, "")]),
             ("garde_corps", [("G1 Terrasse", "barreaudage", D["gc"]["eta"], D["gc"]["niveau"], False, "garde-corps")])]
    # les données : la poutre, ses sections, ses armatures, la méthode ; « bureau » / « modifié » disent d'où vient la valeur
    s1 = (ligne("Nom", _champ_txt("V1"), full=True)
          + ligne("Béton", sel(hyp["beton"]), full=True, src="bureau")
          + ligne("Acier", seg(["B400", "B500"], f"B{int(hyp['fyk'])}"), full=True, src="bureau")
          + ligne("largeur", num(nb(hyp["b"]), "cm"), "b") + ligne("hauteur", num(nb(hyp["h"]), "cm", focus=True), "h")
          + ligne("Enrobage", num(nb(hyp["c"]), "cm"), "c", src="bureau"))
    s2 = (f'<div class="secards">{_secard(A, True)}{_secard(B, False)}</div>'
          f'<div class="addrow">{icon("plus")}Ajouter une section</div>')
    s3 = ('<div class="grp">Inférieures</div>' + _lit("Lit 1", 3, 20) + f'<div class="addrow">{icon("plus")}Ajouter un lit</div>'
          + '<div class="grp">Supérieures</div>' + _lit("Lit 1", 2, 14)
          + '<div class="grp">Effort tranchant</div>'
          + '<div class="lit"><span>Étrier</span><div class="field select"><span class="v">Ø8</span>' + icon("chevron-down")
          + '</div><div class="field"><input value="20"><span class="u">cm</span></div></div>')
    s4 = (ligne("Coefficient acier", num(nb(dft["gamma_s"]), ""), "γ<sub>s</sub>", src="bureau")
          + ligne("Jeu entre lits", num("1,5", "cm"), "", src=f"modifié:{nb(dft['jeu_lits'])} cm")
          + ligne("Montage", sel(dft["techno"]), full=True, src="bureau"))
    donnees = (f'<div class="phead"><span class="h3">Données</span><span class="muted" style="font-size:12px">Méthode du bureau</span>'
               f'<button class="pcoll" data-coll="g" title="Replier les données">{icon("panel-left")}</button></div>'
               f'<div class="pbody">{elements_rapport(items, "Ajouter un élément")}'
               + section(1, "Poutre V1", "", s1, True)
               + section(2, "Sections et sollicitations", "", s2, True)
               + section(3, "Armatures · section A", "", s3, True)
               + section(4, "Armatures · section B", f"{B['inf']} / {B['sup']} · étrier Ø8 e 10")
               + section(5, "Méthode", "", s4, True) + "</div>")
    # le dessin : les coupes à l'échelle, dessinées sur la géométrie que le module a retenue
    vues = [{"cle": "A", "titre": f"Section A · {A['role']}", "court": "Section A", "svg": p["svg"]["A"], "legende": f"{A['inf']} · {A['sup']} · {A['etrier']}"},
            {"cle": "B", "titre": f"Section B · {B['role']}", "court": "Section B", "svg": p["svg"]["B"], "legende": f"{B['inf']} · {B['sup']} · {B['etrier']}"},
            {"cle": "AB", "titre": "Sections A et B", "court": "A et B", "svg": p["svg"]["AB"], "legende": "même échelle"}]
    dessin = feuilles_dessin(vues, "A", cartouche("Poutres R+1 · V1", "Réalex", "A"))
    # le résultat : section par section, une vérification par ligne ; la ligne ouvre le calcul de sa section
    gov_t = max(v["t"] for s in p["sections"] for v in s["verifs"])
    rows = ""
    for s in p["sections"]:
        rows += f'<div class="elgrp">Section {s["nom"]} · {s["role"]}<span class="pct {s["niveau"]}">{pc(s["eta"])}{NBSP}%</span></div>'
        for v in s["verifs"]:
            pk = f"inf-{s['nom']}" if (v["cle"] == "inf" and s["nom"] == "A") else ""
            rows += ligne_res(v["nom"], esc(_court(v, s)), v["taux"], v["niveau"], calcul=s["nom"], pk=pk, gov=v["t"] == gov_t)
    gv = next(v for s in p["sections"] for v in s["verifs"] if v["t"] == gov_t)
    resultat = (f'<div class="phead"><span class="h3">Résultat</span>{statut(p["niveau"])}'
                f'<button class="pcoll" data-coll="d" title="Replier le résultat">{icon("panel-right")}</button></div>'
                f'<div class="pbody"><div class="verdict">{chip(p["eta"], p["niveau"], xl=True)}<div class="gov"><b>{esc(p["gouvernant"])}</b>'
                f'{gv["valeur"]} ≤ {gv["limite"]}</div></div><div class="els">{rows}</div></div>'
                f'<div class="pfoot"><div class="mass"><span>Méthode du bureau · module poutre</span><b>v{p["version"]}</b></div>'
                f'<div class="legend">{legende()}</div></div>')
    # les volets : le sélecteur d'armatures (même geste que les profilés) et la grande fenêtre de calcul de chaque section
    i_o = next((i for i, c in enumerate(p["selecteur"]) if c["optimal"]), 0)
    rows_pk = [{"nom": c["nom"], "col": str(c["aire"]), "taux": c["eta"], "niveau": c["niveau"], "cur": c["retenue"], "opt": c["optimal"],
                "prop": f"{c['nom']} · A<sub>s</sub> = {c['aire']} mm² · un lit"} for c in p["selecteur"]]
    filtres = "".join(f'<span class="fchip on">Ø{d}</span>' for d in (12, 14, 16, 20, 25)) + '<span class="fchip on">1 lit</span><span class="fchip">2 lits</span>'
    volets = selecteur("inf-A", "Armatures inférieures · section A", "triées par section d'acier · la molette fait défiler", filtres, rows_pk,
                       max(0, i_o - 4), "mm²")
    hyps = [f"Béton <b>{hyp['beton']}</b> · f<sub>ck,cube</sub> = {hyp['fck_cube']} N/mm²",
            f"Acier <b>B{int(hyp['fyk'])}</b> · f<sub>yd</sub> = f<sub>yk</sub> / γ<sub>s</sub> = {int(hyp['fyk'])} / {nb(hyp['gamma_s'])} = {nb(hyp['fyd'], 1)} N/mm²",
            f"α<sub>b</sub> = {nb(hyp['alpha_b'])} · μ = {nb(hyp['mu'], 4)}",
            f"b × h = {hyp['b']} × {hyp['h']} cm · enrobage {nb(hyp['c'])} cm"]
    for s in p["sections"]:
        t_max = max(v["t"] for v in s["verifs"])
        lignes = [{"nom": v["nom"], "sous": esc(v["spec"]), "formule": v["lit"], "valeur": v["valeur"], "limite": v["limite"], "taux": v["taux"],
                   "niveau": v["niveau"], "gov": v["t"] == t_max} for v in s["verifs"]]
        detail = "".join(f'<div class="ck-g"><div class="ck-t">{esc(v["nom"])}</div>' + "".join(f'<div class="fx"><span>{f}</span></div>' for f in v["formules"]) + "</div>"
                         for v in s["verifs"])
        efforts = " · ".join(x for x in (f"M<sub>inf</sub> = {s['M_inf']} kN·m" if s["M_inf"] else "", f"M<sub>sup</sub> = {s['M_sup']} kN·m" if s["M_sup"] else "",
                                         f"V = {s['V']} kN") if x)
        volets += fenetre_calcul(s["nom"], f"Poutre V1 · section {s['nom']} · {s['role']}", f"{efforts} · méthode du bureau", hyps, lignes, detail)
    top = barre_haut([("Réalex", "projets"), ("Étude de la dalle", "projets")], "Poutres R+1", "A", ("draft", "Brouillon"), "poutre")
    mtop = (f'<header class="mtop"><a class="btn icon ghost"{go("projets")} title="Retour aux projets">{icon("arrow-left")}</a><div class="t"><b>Poutres R+1 · V1</b>'
            f'<span>Réalex · indice A · brouillon</span></div>{chip(p["eta"], p["niveau"])}</header>')
    corps = etabli("modules", top, mtop, donnees, dessin, resultat, volets, "inf-A", "A")
    return "Poutres R+1", corps, SCRIPT_ETABLI

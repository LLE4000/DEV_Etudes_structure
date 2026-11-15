# -*- coding: utf-8 -*-
"""Générateur SAF — fondation de grue : l'EXPORT SAF (saf/classeur.py, saf/fondation_grue/ecriture.py) et la lecture de la
BIBLIOTHÈQUE DES CHÂSSIS (grues/chassis/acces.py, saf/fondation_grue/chassis.py). Cahier des charges § 8 et § 11.

Ce que la suite prouve :
  1. S1 — NON-RÉGRESSION : la dalle type, anneau 2 cm, k = 12 → mêmes feuilles, mêmes en-têtes, mêmes tableaux Excel,
     mêmes valeurs et mêmes Id que le fichier SAF du bureau (annexe B), hors les différences admises.
  2. S1b — LES PLOTS : préfabriqués (charges FF sur les plots, poids FP) et coulés en place (nœuds, sous-régions PL).
  3. LES RÈGLES D'ÉCRITURE — des valeurs seulement, les nombres en cellules numériques, les dates en cellules date, les
     listes au mm ; la règle des Id ; un fichier par k ; les options (arêtes internes, Nonlinear, anneau) ; un contour.
  4. LA BIBLIOTHÈQUE — l'accès Python rend ce que rend le moteur d'origine (JS) sur les 21 châssis ; B1 à B5 côté moteur
     et SAF ; la base n'est jamais modifiée.
  5. MUTATIONS — arbre miroir, contrôles rapides.

Le classeur écrit est relu par openpyxl, un lecteur indépendant de l'écrivain.

    /usr/bin/python3.12 tests/test_saf_grue_export.py [--rapide] [--racine dossier]
"""
import datetime
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(ICI)
ARGS = sys.argv[1:]
RACINE_MOTEUR = ARGS[ARGS.index("--racine") + 1] if "--racine" in ARGS else RACINE
RAPIDE = "--rapide" in ARGS
os.chdir(RACINE)
sys.path.insert(0, RACINE_MOTEUR)

import openpyxl  # noqa: E402

from grues.chassis import acces as BIB  # noqa: E402
from saf import classeur as CL  # noqa: E402
from saf.fondation_grue import chassis as CH  # noqa: E402
from saf.fondation_grue import ecriture as EC  # noqa: E402
from saf.fondation_grue import geometrie as G  # noqa: E402
from saf.fondation_grue import projet as PR  # noqa: E402

OK, KO = [], []
JEUX = [(350, 440, 440, 350), (150, 150, 550, 550), (135, 100, 800, 135)]
DATE = datetime.datetime(2026, 11, 14, 9, 30)
REF = EC.reference()
ADMISES = {("StructuralSurfaceMember", "Area [m2]"), ("StructuralSurfaceMemberOpening", "Area [m2]"), ("StructuralSurfaceConnection", "C1y [MN/m3]"),
           ("StructuralLoadCase", "Description")}


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{str(info)[:500]}]" if info != "" and not cond else ""))


def projet(jeux=JEUX, **k):
    el = PR.nouveau()
    el["reactions"] = [{"nom": "", "P": list(j)} for j in jeux]
    for cle, v in k.items():
        if isinstance(v, dict):
            if cle == "grue" and "chassis" in v and "appuis" not in v:      # un châssis CHOISI dans la bibliothèque : ses appuis réels
                el["grue"]["appuis"], el["grue"]["rect"]["angle"] = "reels", None
            if cle == "grue" and v.get("appuis") == "rectangle" and "rect" not in v:
                el["grue"]["rect"]["angle"] = None                           # … et un rectangle sans orientation : celle de la platine
            el[cle].update(v)
        else:
            el[cle] = v
    return PR.lire(el)


def lire(octets):
    """Le classeur relu : {feuille: {"lignes", "tableau": (nom, plage, colonnes) | None, "formules", "types"}}, et l'ordre."""
    wb = openpyxl.load_workbook(io.BytesIO(octets))
    out = {}
    for ws in wb.worksheets:
        t = list(ws.tables.values())
        out[ws.title] = {"lignes": [[c.value for c in r] for r in ws.iter_rows()],
                         "tableau": (t[0].name, t[0].ref, [c.name for c in t[0].tableColumns]) if t else None,
                         "formules": sum(1 for r in ws.iter_rows() for c in r if c.data_type == "f"),
                         "cellules": {c.coordinate: c for r in ws.iter_rows() for c in r}}
    return wb.sheetnames, out


def classeur(el, k=12.0):
    return lire(EC.fichier(el, k, DATE))


def objets(f, feuille):
    """{nom: {colonne: valeur}} d'une feuille à tableau."""
    L = f[feuille]["lignes"]
    return {li[0]: dict(zip(L[0], li)) for li in L[1:]}


def pareil(x, y):
    if x in (None, "") and y in (None, ""):
        return True
    if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
        return abs(x - y) < 1e-9
    return x == y


# ------------------------------------------------------------------ les contrôles RAPIDES (rejoués par les mutations)
def rapides():
    ordre, f = classeur(projet(projet={"projet": "Essai", "grue": "125EC-B6", "indice": "A"}))
    chk("S1 dalle type, anneau 2 cm, k = 12, sans arêtes internes : les 14 feuilles du fichier du bureau, dans son ordre, LLE_DONNEE en tête",
        ordre == REF["ordre"] and ordre[0] == "LLE_DONNEE", ordre)
    ecarts, hors_admis = [], []
    for nom in REF["ordre"][1:]:
        R, mien = REF["feuilles"][nom], f[nom]
        if "tableau" not in R:                                    # Project, Model : deux colonnes, sans tableau
            for li, a in zip(mien["lignes"], R["lignes"]):
                if li[0] != a[0]:
                    hors_admis.append((nom, "ligne", a[0], li[0]))
                elif not pareil(li[1], a[1]) and li[0] not in ("Created", "Last update", "Name", "Description"):
                    hors_admis.append((nom, li[0], a[1], li[1]))
            if mien["tableau"] is not None or len(mien["lignes"]) != len(R["lignes"]):
                hors_admis.append((nom, "forme"))
            continue
        if mien["tableau"] is None or mien["tableau"][0] != R["tableau"] or mien["tableau"][2] != R["colonnes"] or mien["lignes"][0] != R["colonnes"]:
            hors_admis.append((nom, "tableau ou en-têtes", mien["tableau"]))
            continue
        miens = objets(f, nom)
        for a in R["lignes"]:
            att = dict(zip(R["colonnes"], a))
            if a[0] not in miens:
                hors_admis.append((nom, "objet absent", a[0]))
                continue
            for col in R["colonnes"]:
                if not pareil(miens[a[0]][col], att[col]):
                    (ecarts if (nom, col) in ADMISES or (nom == "StructuralSurfaceActionFree" and a[0] in ("FF21", "FF22", "FF23", "FF24"))
                     else hors_admis).append((nom, a[0], col, att[col], miens[a[0]][col]))
    chk("S1 mêmes en-têtes, mêmes tableaux Excel (noms et colonnes), mêmes valeurs et mêmes Id que le fichier du bureau pour LC1 à LC6, FF1 à "
        "FF20, ELS-Car (auto)1 à 7, les nœuds, D1, D2, O1, SS1, SS2, SF1, CS1, le matériau et les groupes — aucun écart hors des "
        "différences admises", not hors_admis, hors_admis[:6])
    chk(f"S1 les {len(ecarts)} écarts sont ceux que le cahier des charges admet : les aires calculées (D1 : 64 m², le fichier disait 52,5), "
        "C1y = k/10 (l'erreur de l'Excel, corrigée), les descriptions des cas, et LC7 / FF21 à FF24 (l'ancien 3.b est le nouveau 3.d)",
        sorted({(e[0], e[2]) for e in ecarts}) == sorted(ADMISES | {("StructuralSurfaceActionFree", "q [kN/m2]")}) and len(ecarts) == 15,
        sorted({(e[0], e[2]) for e in ecarts}))
    lc, ff, co = objets(f, "StructuralLoadCase"), objets(f, "StructuralSurfaceActionFree"), objets(f, "StructuralLoadCombination")
    chk("S1 les cas ajoutés : LC1 à LC17 (16 positions), FF1 à FF64 (FF(4·(n−2)+p) pour le cas LCn et le pied p), ELS-Car (auto)1 à 17 "
        "(LC1 seul, puis LC1 + LCn) ; les plages des tableaux suivent (A1:G18, A1:M65, A1:L18)",
        len(lc) == 17 and len(ff) == 64 and len(co) == 17 and ff["FF64"]["Load case"] == "LC17" and ff["FF61"]["Load case"] == "LC17"
        and ff["FF60"]["Load case"] == "LC16" and co["ELS-Car (auto)17"]["Load Case name 2"] == "LC17" and co["ELS-Car (auto)1"]["Load Case name 2"] is None
        and (f["StructuralLoadCase"]["tableau"][1], f["StructuralSurfaceActionFree"]["tableau"][1], f["StructuralLoadCombination"]["tableau"][1])
        == ("A1:G18", "A1:M65", "A1:L18"), (len(lc), len(ff), len(co)))
    chk("§ 8 Id : un objet du fichier du bureau garde son Id (LC2 : d110d185-…) ; un objet nouveau reçoit un UUID v5 — « StructuralLoadCase/LC8 » "
        "→ 9a654ef6-5e95-5fbd-8fce-f0aa4a71a1bb ; « StructuralPointConnection/N9 » → 2b0348d8-b7a8-58d1-8037-7444d012217a",
        lc["LC2"]["Id"] == "d110d185-8420-4c42-8bab-34d3ed438063" and lc["LC8"]["Id"] == "9a654ef6-5e95-5fbd-8fce-f0aa4a71a1bb"
        and EC.identifiant("StructuralPointConnection", "N9") == "2b0348d8-b7a8-58d1-8037-7444d012217a"
        and EC.identifiant("StructuralPointConnection", "N1") == "21674468-c3ae-4da1-b328-9394b3714726", (lc["LC2"]["Id"], lc["LC8"]["Id"]))
    mo = dict(li[:2] for li in f["Model"]["lignes"])
    pj = dict(li[:2] for li in f["Project"]["lignes"])
    chk("Model : Name = le projet ; Description = la grue + « / CHASSIS a x b m » généré depuis les entraxes (« 125EC-B6 / CHASSIS 6,00x6,00m ») ; "
        "Revision number = l'indice ; SAF Version 2.1.0 ; Created et Last update = la date d'export, en cellules date ; Project type "
        "« Fondation grue » ; les autres lignes copiées (« Scia enginneer », l'Id)",
        mo["Name"] == "Essai" and mo["Description"] == "125EC-B6 / CHASSIS 6,00x6,00m" and mo["Revision number"] == "A" and mo["SAF Version"] == "2.1.0"
        and mo["Created"] == DATE and pj["Last update"] == DATE and f["Model"]["cellules"]["B8"].is_date and f["Project"]["cellules"]["B4"].is_date
        and pj["Project type"] == "Fondation grue" and mo["Source application"] == "Scia enginneer" and mo["Id"] == "b7a9b9c1-f38a-46ee-bd2a-5d5fc12fcda4", mo)
    vides = dict(li[:2] for li in classeur(projet())[1]["Model"]["lignes"])
    chk("un champ vide reste vide : sans projet, sans grue, sans indice — Name et Revision number vides, Description « CHASSIS 6,00x6,00m »",
        vides["Name"] is None and vides["Revision number"] is None and vides["Description"] == "CHASSIS 6,00x6,00m", vides)
    chk("des VALEURS seulement : aucune formule dans aucune feuille ; les nombres en cellules numériques (q, coordonnées, épaisseur), les "
        "listes en texte séparé par « ; » sans espace",
        all(x["formules"] == 0 for x in f.values()) and f["StructuralSurfaceActionFree"]["cellules"]["E2"].data_type == "n"
        and f["StructuralPointConnection"]["cellules"]["B3"].data_type == "n" and f["StructuralSurfaceMember"]["cellules"]["E2"].value == 500
        and ff["FF1"]["Coordinate X [m]"] == "0.75;1.25;1.25;0.75;0.75" and ff["FF1"]["Coordinate Z [m]"] == "0;0;0;0;0")
    chk("§ 8 coordonnées des listes : arrondies au mm, point décimal, sans zéro inutile — « 0.75 », « 7 », « 1.017 », jamais « 0.750 » ni « -0 »",
        [EC.coord(v) for v in (0.75, 7.0, 1.0174, 0.7500001, -0.0001, 12.3456)] == ["0.75", "7", "1.017", "0.75", "0", "12.346"])
    ss = objets(classeur(projet(), 2.0)[1], "StructuralSurfaceConnection")
    chk("§ 13 le sol, un fichier par k : pour k = 2 — Subsoil « 2 MN/m³ », C1z = 2, C1x = C1y = 0,2, C2x = C2y = 0,2 ; le C1y = k du fichier "
        "actuel est une erreur de l'Excel, corrigée", (ss["SS1"]["Subsoil"], ss["SS1"]["C1z [MN/m3]"], ss["SS1"]["C1x [MN/m3]"], ss["SS1"]["C1y [MN/m3]"],
                                                       ss["SS1"]["C2x [MN/m]"], ss["SS2"]["C2y [MN/m]"], ss["SS1"]["C1z Spring"]) == ("2 MN/m³", 2, 0.2, 0.2, 0.2, 0.2, "Linear"), ss["SS1"])
    # (lot 4, § 7 : UN SEUL SAF par fondation, « <Projet>.xlsx », écrit avec le PREMIER k coché « SCIA » — le bureau fait varier k
    # dans SCIA ; avant, un fichier par k coché, « <Projet>_k02.xlsx »)
    el_k = projet()
    el_k2 = projet(sol={"k": [{"k": 30, "scia": False}, {"k": 12, "scia": True}, {"k": 5, "scia": True}], "reprise": None})
    chk("S0 / § 7 le nom du SEUL fichier : « <Projet>.xlsx » — le projet de la barre du haut, à défaut celui d'une note d'avant le lot 4, à "
        "défaut le nom de la note ; il porte le PREMIER k coché « SCIA » (dalle type : 2 MN/m³ ; 30 non coché, puis 12 et 5 cochés : 12)",
        EC.nom_fichier(projet(projet={"projet": "Blanc"})) == "Blanc.xlsx" and EC.nom_fichier(projet(), "Ma note") == "Ma note.xlsx"
        and EC.nom_fichier(projet(projet={"projet": "Blanc"}), "Ma note", "Chantier A/B") == "Chantier A-B.xlsx"
        and EC.k_du_fichier(el_k) == 2 and EC.k_du_fichier(el_k2) == 12,
        (EC.k_du_fichier(el_k), EC.k_du_fichier(el_k2)))
    # ---- S1b : les plots
    _, fp = classeur(projet(grue={"plots": {"type": "prefabrique", "a": 2.0, "b": 2.0, "h": 0.35}}))
    fa = objets(fp, "StructuralSurfaceActionFree")
    # (lot 4, § 3.3 et § 10 S1b : la réaction d'un plot préfabriqué arrive sur le carré de diffusion à 45° — 1,20 × 1,20 m centré
    # sur le pied —, q = −R ÷ 1,44 ; avant, sur tout le plot : −87,5 ; −110 ; −110 ; −87,5 kN/m², FF1 sur « 0;2;2;0;0 »)
    chk("S1b dalle type + plots préfabriqués 2,00 × 2,00 × 0,35 : pour LC2, FF1 à FF4 = −243,056 ; −305,556 ; −305,556 ; −243,056 kN/m² (la "
        "réaction diffusée à 45° : carré de 1,20 m) ; FF1 : X « 0.4;1.6;1.6;0.4;0.4 », Y « 7.6;7.6;6.4;6.4;7.6 » ; FP1 à FP4 = −8,75 kN/m² dans "
        "LC1 sur TOUT le plot (« 0;2;2;0;0 », « 8;8;6;6;8 ») ; Id de FP1 = 48716bfe-954d-504f-8402-7a427ebfd99c ; pas de feuille "
        "StructuralSurfaceMemberRegion",
        [fa[f"FF{i}"]["q [kN/m2]"] for i in (1, 2, 3, 4)] == [-243.056, -305.556, -305.556, -243.056] and fa["FF1"]["Coordinate X [m]"] == "0.4;1.6;1.6;0.4;0.4"
        and fa["FF1"]["Coordinate Y [m]"] == "7.6;7.6;6.4;6.4;7.6" and [fa[f"FP{i}"]["q [kN/m2]"] for i in (1, 2, 3, 4)] == [-8.75] * 4
        and fa["FP1"]["Coordinate X [m]"] == "0;2;2;0;0" and fa["FP1"]["Coordinate Y [m]"] == "8;8;6;6;8"
        and all(fa[f"FP{i}"]["Load case"] == "LC1" for i in (1, 2, 3, 4))
        and fa["FP1"]["Id"] == "48716bfe-954d-504f-8402-7a427ebfd99c" and "StructuralSurfaceMemberRegion" not in fp,
        ([fa[f"FF{i}"]["q [kN/m2]"] for i in (1, 2, 3, 4)], fa["FF1"]["Coordinate X [m]"], fa["FP1"]))
    oc, fc = classeur(projet(dalle={"Lx": 10.0, "Ly": 10.0}, grue={"plots": {"type": "coule", "a": 2.0, "b": 2.0, "h": 0.35}}))
    nc = {li[0]: (li[1], li[2]) for li in fc["StructuralPointConnection"]["lignes"][1:]}
    rg = objets(fc, "StructuralSurfaceMemberRegion")
    fac = objets(fc, "StructuralSurfaceActionFree")
    chk("S1b dalle 10 × 10 × 0,50, châssis 6 × 6 centré, plots coulés en place 2,00 × 2,00 × 0,35 : nœuds N9 (1 ; 7), N10 (3 ; 7), N11 (3 ; 9), "
        "N12 (1 ; 9), puis N13 à N16 (P2), N17 à N20 (P3), N21 (1 ; 1), N22 (3 ; 1), N23 (3 ; 3), N24 (1 ; 3)",
        [nc[f"N{i}"] for i in (9, 10, 11, 12, 21, 22, 23, 24)] == [(1, 7), (3, 7), (3, 9), (1, 9), (1, 1), (3, 1), (3, 3), (1, 3)]
        and nc["N13"] == (7, 7) and nc["N17"] == (7, 1) and len(nc) == 24, nc)
    chk("S1b PL1 = C30/37 ; 850 ; Centre ; D1 ; « N9;N10;N11;N12 » ; « Line;Line;Line;Line » ; 0 ; 4 ; Id = 3f1c3d78-9b4f-5266-ae8b-2eee54702c67 ; "
        "la feuille StructuralSurfaceMemberRegion est juste après StructuralSurfaceMemberOpening, son tableau et ses en-têtes sont ceux du "
        "cahier des charges ; les FF restent sur les plaques 0,50 × 0,50 ; FP1 à FP4 = −8,75 kN/m²",
        [rg["PL1"][c] for c in EC.COLONNES_REGION] == ["PL1", "C30/37", 850, "Centre", "D1", "N9;N10;N11;N12", "Line;Line;Line;Line", 0, 4, None,
                                                         "3f1c3d78-9b4f-5266-ae8b-2eee54702c67"]
        and oc[oc.index("StructuralSurfaceMemberOpening") + 1] == "StructuralSurfaceMemberRegion"
        and fc["StructuralSurfaceMemberRegion"]["tableau"] == ("tblStructuralSurfaceMemberRegion", "A1:K5", EC.COLONNES_REGION)
        and rg["PL4"]["Nodes"] == "N21;N22;N23;N24" and fac["FF1"]["Coordinate X [m]"] == "1.75;2.25;2.25;1.75;1.75" and fac["FF1"]["q [kN/m2]"] == -1400
        and [fac[f"FP{i}"]["q [kN/m2]"] for i in (1, 2, 3, 4)] == [-8.75] * 4, rg.get("PL1"))
    # ---- les options
    oa, fa2 = classeur(projet(options={"aretes": True, "nonlinear": True}, dalle={"Lx": 10.0, "Ly": 10.0},
                              grue={"plots": {"type": "coule", "a": 2.0, "b": 2.0, "h": 0.35}}))
    ar = objets(fa2, "StructuralCurveEdge")
    na = {li[0]: (li[1], li[2]) for li in fa2["StructuralPointConnection"]["lignes"][1:]}
    chk("option « Arêtes internes des bandes » : la feuille StructuralCurveEdge vient après StructuralSurfaceMemberOpening et la sous-région, "
        "tableau tblStructuralCurveEdge, en-têtes exacts (« 2D member », m minuscule) ; BX1 sur D1 : début, centres de P1 et P2, fin — "
        "« N25;N26;N27;N28 », « Line;Line;Line » ; un centre commun à deux bandes est le même nœud ; les nœuds des arêtes viennent après "
        "ceux des plots",
        oa[oa.index("StructuralSurfaceMemberRegion") + 1] == "StructuralCurveEdge"
        and fa2["StructuralCurveEdge"]["tableau"] == ("tblStructuralCurveEdge", "A1:F5", ["Name", "2D member", "Nodes", "Segments", "Parent ID", "Id"])
        and list(ar) == ["BX1", "BX2", "BY1", "BY2"] and ar["BX1"]["Nodes"] == "N25;N26;N27;N28" and ar["BX1"]["Segments"] == "Line;Line;Line"
        and [na[n] for n in ("N25", "N26", "N27", "N28")] == [(0, 8), (2, 8), (8, 8), (10, 8)] and ar["BX1"]["2D member"] == "D1"
        and ar["BY1"]["Nodes"].split(";")[2] == "N26" and len(na) == 36, (ar.get("BX1"), ar.get("BY1"), len(na)))
    ca = objets(fa2, "StructuralLoadCombination")
    chk("option « Combinaisons non linéaires » : Type = « Nonlinear » sur chaque combinaison, et la ligne « SAF Version » = 2.2.0 ; sans "
        "l'option : « Linear », 2.1.0", {c["Type"] for c in ca.values()} == {"Nonlinear"}
        and dict(li[:2] for li in fa2["Model"]["lignes"])["SAF Version"] == "2.2.0" and {c["Type"] for c in co.values()} == {"Linear"})
    _, f3 = classeur(projet(options={"anneau": 0.03}))
    n3 = {li[0]: (li[1], li[2]) for li in f3["StructuralPointConnection"]["lignes"][1:]}
    chk("option anneau D2 de 3 cm : N5 à N8 à (−0,03 ; −0,03) … (−0,03 ; 8,03) ; aire de D2 = 8,06² − 64 = 0,9636 m²",
        [n3[f"N{i}"] for i in (5, 6, 7, 8)] == [(-0.03, -0.03), (8.03, -0.03), (8.03, 8.03), (-0.03, 8.03)]
        and abs(objets(f3, "StructuralSurfaceMember")["D2"]["Area [m2]"] - 0.9636) < 1e-9, n3)
    _, f7 = classeur(projet(dalle={"forme": "contour", "sommets": [[0, 0], [12, 0], [12, 7], [7, 7], [7, 12], [0, 12]], "beton": "C25/30"},
                            grue={"a": 4.5, "b": 4.5, "rect": {"a": 0.6, "b": 0.6, "angle": None}, "ex": -1.8, "ey": -1.8, "beta": 30.0}))
    d7, m7, n7 = objets(f7, "StructuralSurfaceMember"), objets(f7, "StructuralMaterial"), {li[0]: (li[1], li[2]) for li in f7["StructuralPointConnection"]["lignes"][1:]}
    f71 = objets(f7, "StructuralSurfaceActionFree")["FF1"]
    xs, ys = [float(v) for v in f71["Coordinate X [m]"].split(";")], [float(v) for v in f71["Coordinate Y [m]"].split(";")]
    chk("un contour (la dalle en L de T7) en C25/30 : N1 à N6 = ses sommets, N7 à N12 = l'anneau ; D1 « N1;…;N6 », six « Line », aire 119 m² (144 − 25) ; O1 "
        "sur les mêmes nœuds ; le matériau C25/30 (E = 31 500, G = E/2,4, Id nouveau) ; une plaque tournée de 30° : 5 valeurs X et Y, sens "
        "horaire, |q| × aire du polygone écrit = R",
        [n7[f"N{i}"] for i in range(1, 7)] == [(0, 0), (12, 0), (12, 7), (7, 7), (7, 12), (0, 12)] and n7["N7"] == (-0.02, -0.02) and len(n7) == 12
        and d7["D1"]["Nodes"] == "N1;N2;N3;N4;N5;N6" and d7["D1"]["Edges"] == ";".join(["Line"] * 6) and d7["D1"]["Area [m2]"] == 119
        and d7["D1"]["Material"] == "C25/30" and m7["C25/30"]["E modulus [MPa]"] == 31500 and abs(m7["C25/30"]["G modulus [MPa]"] - 13125) < 1e-9
        and m7["C25/30"]["Id"] == str(uuid.uuid5(EC.ESPACE, "StructuralMaterial/C25/30")) and objets(f7, "StructuralSurfaceMemberOpening")["O1"]["Nodes"] == d7["D1"]["Nodes"]
        and len(xs) == 5 and G.aire_signee(list(zip(xs[:4], ys[:4]))) < 0
        and abs(abs(f71["q [kN/m2]"]) * G.aire(list(zip(xs[:4], ys[:4]))) - 350) < 0.035, (n7, d7.get("D1"), f71))
    ld = f["LLE_DONNEE"]
    plat = [str(v) for li in ld["lignes"] for v in li if v is not None]
    chk("LLE_DONNEE : la première feuille, sans tableau — le récapitulatif en valeurs (le châssis « saisie », les appuis, les plots, l'anneau, "
        "les positions et leurs cas, les bandes et leurs coordonnées)",
        ld["tableau"] is None and ld["formules"] == 0 and all(t in plat for t in ("saisie", "Anneau D2", "POSITIONS DES CHARGES", "BANDES D'INTÉGRATION",
                                                                                "BX1", "3.h", "LC17", "Plots", "Aucun")), plat[:30])
    # ---- la bibliothèque des châssis, côté moteur et SAF
    c132 = CH.lire({"chassis": "liebherr-132hc-chassis-4600"})
    s0 = c132["surfaces"][0]
    cotes = sorted(round(((s0[i][0] - s0[(i + 1) % 4][0]) ** 2 + (s0[i][1] - s0[(i + 1) % 4][1]) ** 2) ** 0.5, 3) for i in range(4))
    travers = next((s0[i], s0[(i + 1) % 4]) for i in range(4) if round(((s0[i][0] - s0[(i + 1) % 4][0]) ** 2 + (s0[i][1] - s0[(i + 1) % 4][1]) ** 2) ** 0.5, 3) == 0.58)
    c390 = CH.lire({"chassis": "liebherr-390hc-chassis-8000"})
    chk("B1 appuis réels — 132 HC : 4 surfaces 0,58 × 0,40 à 45°, 0,58 en travers du bras (ce côté est perpendiculaire à la diagonale du pied "
        "P1), aire 0,2320 m² ; 390 HC : 0,70 × 0,70 parallèle aux axes (sommets à ± 0,35 du centre)",
        cotes == [0.4, 0.4, 0.58, 0.58] and round(c132["aire"], 4) == 0.2320 and c132["appuis"] == "reels" and c132["angle"] == 45
        and abs((travers[1][0] - travers[0][0]) * (-1) + (travers[1][1] - travers[0][1]) * 1) < 1e-9
        and c390["angle"] == 0 and round(c390["aire"], 4) == 0.49 and sorted({round(abs(p[0] - c390["centres"][0][0]), 3) for p in c390["surfaces"][0]}) == [0.35],
        (cotes, c132["aire"], c390["surfaces"][0]))
    el132 = projet(dalle={"Lx": 9.0, "Ly": 9.0}, grue={"chassis": "liebherr-132hc-chassis-4600"})
    f132 = objets(classeur(el132)[1], "StructuralSurfaceActionFree")
    cas132 = PR.modele(el132)["cas"]
    faux = []
    for i, c in enumerate(cas132):
        for p in range(4):
            li = f132[f"FF{4 * i + p + 1}"]
            xs, ys = [float(v) for v in li["Coordinate X [m]"].split(";")], [float(v) for v in li["Coordinate Y [m]"].split(";")]
            aire = G.aire(list(zip(xs[:-1], ys[:-1])))
            if abs(abs(li["q [kN/m2]"]) * aire - c["R"][p]) > 1e-4 * c["R"][p] + 0.0005 * aire or li["Load case"] != c["lc"]:
                faux.append((li["Name"], li["q [kN/m2]"], aire, c["R"][p]))
    chk(f"B1 dans le SAF d'un 132 HC, pour chacune des {4 * len(cas132)} lignes FF : |q| × aire du polygone écrit = R (écart < 0,01 %, à "
        "l'arrondi de q au millième près)", len(cas132) == 16 and not faux, faux[:3])
    c290 = CH.lire({"chassis": "liebherr-290hc-chassis-6000-pieds-a-vis"})
    c48 = CH.lire({"chassis": "liebherr-290hc-chassis-6000-pieds-a-vis"}, rond="48")
    f48 = objets(classeur(projet(grue={"chassis": "liebherr-290hc-chassis-6000-pieds-a-vis"}, options={"rond": "48"}))[1], "StructuralSurfaceActionFree")["FF1"]
    x48, y48 = [float(v) for v in f48["Coordinate X [m]"].split(";")], [float(v) for v in f48["Coordinate Y [m]"].split(";")]
    chk("B2 appuis réels, 290 HC sur pieds à vis (Ø 500) : carré de même aire par défaut (côté 0,443 m, celui de la bibliothèque, côtés "
        "parallèles aux axes) ; option 48 côtés : 48 « Line », 49 valeurs X et 49 valeurs Y, aire 0,1958 m² avant l'arrondi au mm, ≈ 0,196 écrite",
        len(c290["surfaces"][0]) == 4 and round(c290["surfaces"][0][1][0] - c290["surfaces"][0][0][0], 3) == 0.443 and c290["forme"] == "carré de même aire"
        and len(c48["surfaces"][0]) == 48 and round(c48["aire"], 4) == 0.1958 and f48["Edges"] == ";".join(["Line"] * 48) and len(x48) == 49 and len(y48) == 49
        and round(G.aire(list(zip(x48[:-1], y48[:-1]))), 3) == 0.196 and abs(abs(f48["q [kN/m2]"]) * G.aire(list(zip(x48[:-1], y48[:-1]))) - 350) < 0.01,
        (c290["surfaces"][0], c48["aire"], len(x48)))
    avant = hashlib.sha256(open(BIB.FICHIER, "rb").read()).hexdigest()
    base_avant = json.dumps(BIB.chassis("wolff-krv-10-60-a60"), sort_keys=True)
    ckrv = CH.lire({"chassis": "wolff-krv-10-60-a60", "appuis": "reels"})
    ckrv2 = CH.lire({"chassis": "wolff-krv-10-60-a60", "rect": {"a": 0.6, "b": 0.4, "angle": 45}})
    chk("B3 Wolff KRV 10-60 (platine non indiquée dans la bibliothèque) : appuis « Rectangle » d'office, même si « Réels » est demandé — "
        "0,50 × 0,50 m par défaut à 0°, signalés (« impose ») et marqués « saisie » ; des dimensions saisies sont prises ; la bibliothèque "
        "n'a pas changé (ni son fichier, ni le châssis rendu)",
        ckrv["appuis"] == "rectangle" and ckrv["impose"] and ckrv["saisie"] and ckrv["platine"] is None and ckrv["dims"] == "0,50 × 0,50 m"
        and ckrv2["dims"] == "0,60 × 0,40 m à 45°" and hashlib.sha256(open(BIB.FICHIER, "rb").read()).hexdigest() == avant
        and json.dumps(BIB.chassis("wolff-krv-10-60-a60"), sort_keys=True) == base_avant and BIB.chassis("wolff-krv-10-60-a60")["pied"]["appui"] is None, ckrv)
    ch200 = BIB.chassis("liebherr-200hc-chassis-6000")
    ap = {a["n"]: (a["x"] / 1000, a["y"] / 1000) for a in BIB.appuis(ch200)}
    c200 = CH.lire({"chassis": "liebherr-200hc-chassis-6000"})
    m_bib = PR.modele(projet(grue={"chassis": "liebherr-200hc-chassis-6000", "appuis": "rectangle", "rect": {"a": 0.5, "b": 0.5, "angle": 0}}))
    m_main = PR.modele(projet())
    chk("B4 un châssis de la bibliothèque (200 HC, 6 × 6 m), β = 0 : appui 4 (−x, +y) = P1 en haut à gauche, appui 3 = P2, appui 2 = P3, appui 1 "
        "(−x, −y) = P4 ; mêmes positions de charge, mêmes cas et mêmes plaques que le même châssis saisi à la main",
        [c200["centres"][i] for i in range(4)] == [ap[4], ap[3], ap[2], ap[1]] and ap[4] == (-3.0, 3.0) and ap[1] == (-3.0, -3.0)
        and [c["R"] for c in m_bib["cas"]] == [c["R"] for c in m_main["cas"]] and [c["nom"] for c in m_bib["cas"]] == [c["nom"] for c in m_main["cas"]]
        and all(max(abs(u[0] - v[0]) + abs(u[1] - v[1]) for u, v in zip(p, q)) < 1e-9 for p, q in zip(m_bib["plaques"], m_main["plaques"])),
        (c200["centres"], ap))
    r132 = CH.lire({"chassis": "liebherr-132hc-chassis-4600", "appuis": "rectangle"})
    fr = objets(classeur(projet(dalle={"Lx": 9.0, "Ly": 9.0}, grue={"chassis": "liebherr-132hc-chassis-4600", "appuis": "rectangle"}))[1], "StructuralSurfaceActionFree")
    chk("B5 132 HC, appuis « Rectangle » : 0,50 × 0,50 m à 45° par défaut (l'orientation de la platine réelle), même centre qu'elle ; "
        "q ≈ −R ÷ 0,25 (−1400 pour 350 kN, à 0,5 % près : l'aire est celle du polygone ÉCRIT, sommets arrondis au mm) ; en repassant à "
        "« Réels » : 0,58 × 0,40 à 45°, q ≈ −R ÷ 0,232 (−1508,6, à 0,5 % près)",
        r132["dims"] == "0,50 × 0,50 m à 45°" and r132["centres"] == c132["centres"] and abs(r132["aire"] - 0.25) < 1e-9
        and abs(fr["FF1"]["q [kN/m2]"] + 1400) < 0.005 * 1400 and abs(f132["FF1"]["q [kN/m2]"] + 350 / 0.232) < 0.005 * 350 / 0.232
        and fr["FF1"]["q [kN/m2]"] != f132["FF1"]["q [kN/m2]"], (r132["dims"], fr["FF1"]["q [kN/m2]"], f132["FF1"]["q [kN/m2]"]))
    ex = CH.lire({"chassis": "liebherr-290hc-21hc290-chassis-6000"})
    ex_r = CH.lire({"chassis": "liebherr-290hc-21hc290-chassis-6000", "appuis": "rectangle"})
    chk("§ 3.1 excentrement radial (290 HC 21HC290 : 10 mm vers l'extérieur) : le centre de la surface d'appui P1 est à (−3,0071 ; 3,0071), "
        "pas sur l'axe d'appui ; en appuis « Rectangle », le rectangle garde ce centre ; les bandes passent par ces centres",
        tuple(round(v, 4) for v in ex["centres"][0]) == (-3.0071, 3.0071) and tuple(round(v, 4) for v in ex_r["centres"][0]) == (-3.0071, 3.0071)
        and ex_r["appuis"] == "rectangle"
        and round(PR.modele(projet(grue={"chassis": "liebherr-290hc-21hc290-chassis-6000"}))["bandes"][0]["debut"][1], 4) == 7.0071, ex["centres"][0])
    oct_ = CH.lire({"chassis": "liebherr-63lc-croix-3800-pyramides"})
    chk("§ 3.1 un octogone (croix 63 LC sur pyramides, 1,10 m sur plats) : 8 sommets, aire 1,0024 m² — celle de la bibliothèque",
        len(oct_["surfaces"][0]) == 8 and round(oct_["aire"], 4) == 1.0024 and oct_["forme"] == "octogone")


def complets():
    print("== 4. la bibliothèque des châssis : l'accès Python face au moteur d'origine")
    js = ("const CD = require('./grues/chassis/src/chassis_dessin.js');\n"
          "const base = require('./grues/chassis/donnees/chassis_grues.json');\n"
          "const out = {};\n"
          "for (const c of base.chassis) out[c.id] = { appuis: CD.appuis(c), emprises: CD.emprises(c), ht: CD.horsToutEmprises(c) };\n"
          "console.log(JSON.stringify(out));\n")
    p = subprocess.run(["node", "-e", js], capture_output=True, text=True, cwd=RACINE)
    ref = json.loads(p.stdout) if p.returncode == 0 else {}
    pire = [0.0]

    def egal(a, b):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
            pire[0] = max(pire[0], abs(a - b))
            return abs(a - b) <= 1e-9
        if isinstance(a, list):
            return isinstance(b, list) and len(a) == len(b) and all(egal(x, y) for x, y in zip(a, b))
        if isinstance(a, dict):
            return isinstance(b, dict) and set(a) == set(b) and all(egal(a[k], b[k]) for k in a)
        return a == b
    faux = [x["id"] for x in BIB.liste()
            if not egal({"appuis": BIB.appuis(BIB.chassis(x["id"])), "emprises": BIB.emprises(BIB.chassis(x["id"])),
                         "ht": BIB.hors_tout_emprises(BIB.chassis(x["id"]))}, ref.get(x["id"]))]
    chk(f"les 21 châssis de la base : les appuis, les surfaces d'appui et le hors-tout rendus par l'accès Python sont ceux du moteur de dessin "
        f"d'origine (chassis_dessin.js, joué par node), au nanomètre près (plus grand écart {pire[0]:.1e} mm)",
        p.returncode == 0 and len(BIB.liste()) == 21 and not faux, (p.stderr[-200:], faux))
    p = subprocess.run(["node", "tests/verifier_bibliotheque.js"], capture_output=True, text=True, cwd=os.path.join(RACINE, "grues", "chassis"))
    chk("les contrôles livrés avec la bibliothèque passent tels quels : « 460 contrôles réussis, 0 échec(s) – 417 valeurs dans la base. »",
        p.returncode == 0 and "460 contrôles réussis, 0 échec(s) – 417 valeurs dans la base." in p.stdout, p.stdout[-200:])
    ch = BIB.chassis("liebherr-132hc-chassis-4600")
    ch["pied"]["appui"]["a"]["valeur"] = 1
    chk("la base n'est jamais modifiée : ce qu'on fait d'un châssis rendu ne touche pas le suivant (132 HC : a = 580 mm)",
        BIB.valeur(BIB.chassis("liebherr-132hc-chassis-4600")["pied"]["appui"]["a"]) == 580)
    chk("un classeur mal formé est refusé : deux feuilles du même nom, un tableau sans ligne de données, un nombre non fini",
        all(_refuse(x) for x in ([{"nom": "A", "lignes": [], "tableau": None}] * 2, [{"nom": "A", "lignes": [["Name"]], "tableau": "t"}],
                                 [{"nom": "A", "lignes": [[float("nan")]], "tableau": None}])))
    b1, b2 = EC.fichier(projet(), 12.0, DATE), EC.fichier(projet(), 12.0, DATE)
    chk("le fichier est reproductible : même projet, même date → mêmes octets", b1 == b2 and len(b1) > 20000)


def _refuse(feuilles):
    try:
        CL.ecrire(feuilles)
        return False
    except ValueError:
        return True


MUTATIONS = [
    ("ecriture.py", 'ESPACE = uuid.UUID("cd663d86-b1df-5f4f-b40e-3c708345d746")', 'ESPACE = uuid.UUID("cd663d86-b1df-5f4f-b40e-3c708345d747")', "un autre espace de noms pour les Id"),
    ("ecriture.py", "            if li[i_nom] == nom:\n                return li[i_id]", "            if False:\n                return li[i_id]", "l'Id du fichier du bureau non repris"),
    ("ecriture.py", '            nom = f"FF{4 * i_cas + p + 1}"', '            nom = f"FF{4 * i_cas + 4 - p}"', "les FF numérotées à l'envers dans un cas"),
    ("ecriture.py", 'libres.append([nom, "Z", "Standard", "Uniform", _entier(round(-c["R"][p] / pg["aire"], 3))', 'libres.append([nom, "Z", "Standard", "Uniform", _entier(round(c["R"][p] / pg["aire"], 3))',
     "la réaction vers le haut"),
    ("ecriture.py", '_entier(k * o["c1y"]), "Linear"', '_entier(k), "Linear"', "C1y = k : l'erreur de l'Excel revenue"),
    ("ecriture.py", '    anneau = G.decaler(contour, o["anneau"])', "    anneau = G.decaler(contour, 0.02)", "la largeur de l'anneau ignorée"),
    ("ecriture.py", '    surfaces = VE.surfaces_chargees(m)', '    surfaces = m["plaques"]', "la réaction d'un plot préfabriqué écrite sur la plaque"),
    ("ecriture.py", '    return ks[0] if ks else None', '    return ks[-1] if ks else None', "le SAF écrit avec le dernier k coché (lot 4, § 7 : le premier)"),
    ("ecriture.py", '    surfaces = VE.surfaces_chargees(m)', '    surfaces = plots["polys"] if plots and plots["type"] == "prefabrique" else m["plaques"]',
     "la réaction d'un plot préfabriqué écrite sur tout le plot (la règle d'avant le lot 4)"),
    ("ecriture.py", '";".join(N.nom(p) for p in (q[3], q[2], q[1], q[0]))', '";".join(N.nom(p) for p in q)', "les sommets d'une sous-région dans l'ordre des charges"),
    ("ecriture.py", '    s = f"{round(v, 3):.3f}".rstrip("0").rstrip(".")', '    s = f"{round(v, 3):.3f}"', "des zéros inutiles dans les listes"),
    ("ecriture.py", '    type_combi = "Nonlinear" if o["nonlinear"] else "Linear"', '    type_combi = "Linear"', "l'option Nonlinear sans effet"),
    ("ecriture.py", '            li[1] = VERSION_SAF_NONLINEAR if o["nonlinear"] else VERSION_SAF', "            li[1] = VERSION_SAF", "la version SAF d'un fichier Nonlinear"),
    ("ecriture.py", '    if aretes:\n        out.append(', '    if False:\n        out.append(', "la feuille des arêtes internes oubliée"),
    ("ecriture.py", '    return {"X": ";".join(coord(x) for x, _ in pts + pts[:1])', '    return {"X": ";".join(coord(x) for x, _ in pts)', "le polygone d'une charge non refermé"),
    ("chassis.py", "    return [valeurs[3], valeurs[2], valeurs[1], valeurs[0]]", "    return list(valeurs)", "les appuis de la bibliothèque non renumérotés"),
    ("chassis.py", "    surfaces = [list(reversed(s)) for s in surfaces]\n", "", "les surfaces d'appui dans le sens trigonométrique"),
    ("chassis.py", '            angle = 45 if platine and platine["rotation"] == 45 else 0', "            angle = 0", "le rectangle saisi sans l'orientation de la platine réelle"),
    ("chassis.py", '        if platine["forme"] == "cercle" and rond != "48":', '        if False:', "une platine ronde toujours en 48 côtés"),
    ("chassis.py", '        exc = BIB.valeur((reel or {}).get("excentrement_radial")) or 0', "        exc = 0", "le rectangle saisi recentré sur l'axe d'appui"),
    ("chassis.py", "    impose = platine is None ", "    impose = False ", "une platine non indiquée laissée en « Réels »"),
]
TEMOIN = ("ecriture.py", 'VERSION_SAF, VERSION_SAF_NONLINEAR = "2.1.0", "2.2.0"', 'VERSION_SAF, VERSION_SAF_NONLINEAR = "2.1.0", "2.2.0"')


def muter(fichier, ancien, nouveau):
    with tempfile.TemporaryDirectory() as tmp:
        for paquet in ("saf", "grues"):
            shutil.copytree(os.path.join(RACINE, paquet), os.path.join(tmp, paquet), ignore=shutil.ignore_patterns("__pycache__"))
        cible = os.path.join(tmp, "saf", "fondation_grue", fichier)
        src = open(cible, encoding="utf-8").read()
        if src.count(ancien) != 1:
            return False, None
        open(cible, "w", encoding="utf-8").write(src.replace(ancien, nouveau))
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--rapide", "--racine", tmp], capture_output=True, text=True, cwd=RACINE)
        return True, p.returncode


def main():
    print("== 1 à 3. S1, S1b, les règles d'écriture, la bibliothèque côté moteur (contrôles rapides)")
    rapides()
    if RAPIDE:
        print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
        return 1 if KO else 0
    complets()
    print("== 5. mutations")
    trouve, code = muter(*TEMOIN)
    chk("témoin : le moteur recopié sans rien changer passe les contrôles rapides", trouve and code == 0, code)
    for fichier, ancien, nouveau, nom in MUTATIONS:
        trouve, code = muter(fichier, ancien, nouveau)
        chk(f"mutation « {nom} » ({fichier}) : un contrôle rougit", trouve and code == 1,
            "texte absent ou répété" if not trouve else f"code {code} : AUCUN contrôle ne rougit")
    print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
    for nom, info in KO:
        print("   -", nom[:160], "|", str(info)[:300])
    return 1 if KO else 0


if __name__ == "__main__":
    sys.exit(main())

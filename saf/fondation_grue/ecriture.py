# -*- coding: utf-8 -*-
"""Fondation de grue — les FEUILLES DU SAF d'un projet : un fichier par raideur de sol k.

Les règles (cahier des charges § 8) :
- mêmes feuilles, même ordre, mêmes en-têtes (au caractère près) que le fichier SAF du bureau (``ndc_type.json`` : le
  contenu de NDC_TYPE.xlsx) ; chaque plage déclarée en tableau Excel nommé comme dans ce fichier ; Project et Model n'ont
  pas de tableau ;
- des VALEURS seulement ; les nombres en cellules numériques ; Created et Last update en cellules date ; les listes
  (nœuds, coordonnées, segments) en texte séparé par « ; » sans espace ; les coordonnées de ces listes arrondies au mm,
  point décimal, sans zéro inutile ;
- l'Id d'un objet qui existe dans le fichier du bureau (même feuille, même nom) est repris tel quel ; un objet nouveau
  reçoit un UUID v5 (espace de noms ``ESPACE``, nom « feuille/nom de l'objet ») — le modèle SCIA type reconnaît ses
  objets par leur Id ;
- un champ vide du fichier du bureau reste vide.

    feuilles(el, k, date, resultats=None, nom_note="") -> les feuilles, prêtes pour saf/classeur.ecrire
    fichier(el, k, date, resultats=None, nom_note="")  -> les octets du .xlsx
    nom_fichier(el, k, nom_note="")                    -> « <Projet>_k02.xlsx »
"""
import datetime
import json
import os
import uuid
from functools import lru_cache
from typing import Dict, List, Optional, Sequence

from .. import classeur as CL
from . import geometrie as G
from . import projet as PR
from .verification import KG_CM2, premier_max

ESPACE = uuid.UUID("cd663d86-b1df-5f4f-b40e-3c708345d746")
VERSION_SAF, VERSION_SAF_NONLINEAR = "2.1.0", "2.2.0"       # « Nonlinear » existe depuis SAF 2.2.0
COLONNES_REGION = ["Name", "Material", "Thickness [mm]", "System plane at", "2D Member", "Nodes", "Edges", "Eccentricity ez [mm]",
                   "Area [m2]", "Parent ID", "Id"]
COLONNES_ARETE = ["Name", "2D member", "Nodes", "Segments", "Parent ID", "Id"]


@lru_cache(maxsize=1)
def reference() -> Dict:
    """Le fichier SAF du bureau : {"ordre" (des feuilles), "feuilles": {nom: {"tableau", "colonnes", "lignes"}}}."""
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ndc_type.json"), encoding="utf-8") as f:
        d = json.load(f)
    return {"ordre": d["entree_LLE_DONNEE"]["ordre_des_feuilles"], "entree": d["entree_LLE_DONNEE"],
            "feuilles": {x["feuille"]: x for x in d["feuilles"]}}


def identifiant(feuille: str, nom: str) -> str:
    """L'Id d'un objet : celui du fichier du bureau s'il y existe (même feuille, même nom), sinon un UUID v5."""
    ref = reference()["feuilles"].get(feuille)
    if ref and "colonnes" in ref:
        i_nom, i_id = ref["colonnes"].index("Name"), ref["colonnes"].index("Id")
        for li in ref["lignes"]:
            if li[i_nom] == nom:
                return li[i_id]
    return str(uuid.uuid5(ESPACE, f"{feuille}/{nom}"))


def _ligne(feuille: str, nom: str) -> List:
    """La ligne d'un objet du fichier du bureau (une copie)."""
    ref = reference()["feuilles"][feuille]
    return list(next(li for li in ref["lignes"] if li[ref["colonnes"].index("Name")] == nom))


def coord(v: float) -> str:
    """Une coordonnée dans une liste : arrondie au mm, point décimal, sans zéro inutile (« 0.75 », « 7 », « 1.017 »)."""
    s = f"{round(v, 3):.3f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _mm(v: float) -> float:
    """Un nombre arrondi au mm, sans bruit de virgule flottante (8.020000000000001 → 8.02) ; jamais « -0 »."""
    r = round(v, 3)
    return 0.0 if r == 0 else r


def _entier(v: float):
    """Un nombre écrit en entier quand il l'est (500 et non 500.0), comme le fichier du bureau."""
    r = round(v, 9)
    return int(r) if r == int(r) else r


def _k_texte(k: float) -> str:
    return f"{k:.3f}".rstrip("0").rstrip(".").replace(".", ",")


def nom_fichier(el: Dict, k: float, nom_note: str = "") -> str:
    """« <Projet>_k02.xlsx » ; sans nom de projet, celui de la note."""
    base = el["projet"]["projet"] or nom_note or "Fondation de grue"
    base = "".join("-" if c in '\\/:*?"<>|' else c for c in base).strip(" .") or "Fondation de grue"
    kk = f"{k:02.0f}" if float(k).is_integer() else f"{int(k):02d}_{f'{k % 1:.3f}'[2:].rstrip('0')}"
    return f"{base}_k{kk}.xlsx"


class _Noeuds:
    """Les nœuds du SAF, dans l'ordre où ils sont écrits ; un point qui tombe sur un nœud déjà écrit (à 1 mm près)
    reprend ce nœud."""

    def __init__(self):
        self.liste: List[tuple] = []

    def nom(self, p: Sequence[float]) -> str:
        x, y = _mm(p[0]), _mm(p[1])
        for i, (qx, qy) in enumerate(self.liste, 1):
            if abs(qx - x) <= G.TOLERANCE + 1e-12 and abs(qy - y) <= G.TOLERANCE + 1e-12:
                return f"N{i}"
        self.liste.append((x, y))
        return f"N{len(self.liste)}"

    def nouveau(self, p: Sequence[float]) -> str:
        """Un nœud écrit sans chercher s'il existe déjà (les sommets du contour et de l'anneau)."""
        self.liste.append((_mm(p[0]), _mm(p[1])))
        return f"N{len(self.liste)}"


def _polygone(poly: Sequence) -> Dict:
    """Un polygone tel qu'il est écrit : coordonnées arrondies au mm, retour au premier sommet ; son aire est celle du
    polygone ÉCRIT (la somme d'une charge répartie dessus fait alors exactement la réaction)."""
    pts = [(_mm(x), _mm(y)) for x, y in poly]
    return {"X": ";".join(coord(x) for x, _ in pts + pts[:1]), "Y": ";".join(coord(y) for _, y in pts + pts[:1]),
            "Z": ";".join("0" for _ in range(len(pts) + 1)), "edges": ";".join("Line" for _ in pts), "aire": G.aire(pts)}


def feuilles(el: Dict, k: float, date: Optional[datetime.datetime] = None, resultats: Optional[List[Dict]] = None,
             nom_note: str = "") -> List[Dict]:
    """Les feuilles du SAF du projet pour la raideur ``k`` (MN/m³). ``date`` : la date d'export (Created, Last update) ;
    ``resultats`` : la vérification rapide par k (pour la feuille LLE_DONNEE)."""
    ref = reference()["feuilles"]
    m = PR.modele(el)
    o, plots, cas = el["options"], m["plots"], m["cas"]
    h = el["dalle"]["h"]
    beton = el["dalle"]["beton"]
    contour = m["contour"]
    n = len(contour)
    date = date or datetime.datetime.now().replace(microsecond=0)

    def table(nom: str, lignes: List[List], colonnes: Optional[List[str]] = None, tableau: Optional[str] = None) -> Dict:
        return {"nom": nom, "tableau": tableau or ref[nom]["tableau"], "lignes": [list(colonnes or ref[nom]["colonnes"])] + lignes}

    # ---- Project, Model : deux colonnes, sans tableau
    project = [list(li) for li in ref["Project"]["lignes"]]
    for li in project:
        if li[0] in ("Created", "Last update"):
            li[1] = date
    model = [list(li) for li in ref["Model"]["lignes"]]
    ch = m["chassis"]
    texte_chassis = f"CHASSIS {ch['a']:.2f}x{ch['b']:.2f}m".replace(".", ",")
    grue = el["projet"]["grue"]
    for li in model:
        if li[0] == "Name":
            li[1] = el["projet"]["projet"] or None
        elif li[0] == "Description":
            li[1] = f"{grue} / {texte_chassis}" if grue else texte_chassis
        elif li[0] == "Revision number":
            li[1] = el["projet"]["indice"] or None
        elif li[0] in ("Created", "Last update"):
            li[1] = date
        elif li[0] == "SAF Version":
            li[1] = VERSION_SAF_NONLINEAR if o["nonlinear"] else VERSION_SAF
    # ---- le matériau : la classe choisie ; C30/37 = les valeurs du fichier du bureau
    E = PR.module_E(beton)
    materiau = [beton, "Concrete", None, beton, 2500, _entier(E), E / (2 * (1 + PR.NU)), PR.NU, 1e-05, None, identifiant("StructuralMaterial", beton)]
    # ---- les nœuds : le contour, l'anneau, les plots coulés en place, les arêtes internes
    N = _Noeuds()
    noeuds_d1 = [N.nouveau(p) for p in contour]
    anneau = G.decaler(contour, o["anneau"])
    noeuds_d2 = [N.nouveau(p) for p in anneau]
    regions = []
    if plots and plots["type"] == "coule":
        for i, q in enumerate(plots["polys"], 1):                 # bas-gauche puis sens trigonométrique (avant rotation)
            regions.append([f"PL{i}", beton, _entier((h + plots["hauteur"]) * 1000), "Centre", "D1",
                            ";".join(N.nom(p) for p in (q[3], q[2], q[1], q[0])), "Line;Line;Line;Line", 0,
                            _entier(round(plots["a"] * plots["b"], 9)), None, identifiant("StructuralSurfaceMemberRegion", f"PL{i}")])
    aretes = []
    if o["aretes"]:
        for b in m["bandes"]:                                     # début ; centres des surfaces d'appui ; fin
            ux, uy = b["fin"][0] - b["debut"][0], b["fin"][1] - b["debut"][1]
            centres = sorted((m["centres"][i] for i in b["pieds"]), key=lambda c: (c[0] - b["debut"][0]) * ux + (c[1] - b["debut"][1]) * uy)
            noms = []
            for p in [b["debut"]] + centres + [b["fin"]]:
                nom = N.nom(p)
                if not noms or noms[-1] != nom:
                    noms.append(nom)
            aretes.append([b["nom"], "D1", ";".join(noms), ";".join("Line" for _ in noms[1:]), None, identifiant("StructuralCurveEdge", b["nom"])])
    lignes_noeuds = [[f"N{i}", _entier(x), _entier(y), 0, identifiant("StructuralPointConnection", f"N{i}")] for i, (x, y) in enumerate(N.liste, 1)]
    # ---- D1 (la dalle) et D2 (l'anneau), O1 (l'ouverture de D2)
    aire_d1 = G.aire(contour)
    aire_d2 = G.aire(anneau) - aire_d1
    membres = []
    for nom, noeuds, aire_ in (("D1", noeuds_d1, aire_d1), ("D2", noeuds_d2, aire_d2)):
        li = _ligne("StructuralSurfaceMember", nom)
        c = ref["StructuralSurfaceMember"]["colonnes"]
        li[c.index("Material")] = beton
        li[c.index("Thickness [mm]")] = _entier(h * 1000)
        li[c.index("Nodes")] = ";".join(noeuds)
        li[c.index("Edges")] = ";".join("Line" for _ in noeuds)
        li[c.index("Area [m2]")] = _entier(round(aire_, 9))
        membres.append(li)
    ouverture = ["O1", "D2", ";".join(noeuds_d1), ";".join("Line" for _ in noeuds_d1), _entier(round(aire_d1, 9)), None,
                 identifiant("StructuralSurfaceMemberOpening", "O1")]
    # ---- le sol : C1z = k ; C1x, C1y, C2x, C2y = k × leur rapport (k/10 par défaut)
    sol = [[nom, d, None, f"{_k_texte(k)} MN/m³", None, _entier(k * o["c1x"]), _entier(k * o["c1y"]), "Linear", _entier(k),
            _entier(k * o["c2x"]), _entier(k * o["c2y"]), None, identifiant("StructuralSurfaceConnection", nom)]
           for nom, d in (("SS1", "D1"), ("SS2", "D2"))]
    # ---- les cas de charge et les combinaisons
    lc = [_ligne("StructuralLoadCase", "LC1")]
    for c in cas:
        lc.append([c["lc"], c["description"], "Variable", "LG2", "Static", "Short", identifiant("StructuralLoadCase", c["lc"])])
    type_combi = "Nonlinear" if o["nonlinear"] else "Linear"
    combis = [["ELS-Car (auto)1", None, "SLS (Serviceability Limit State)", None, type_combi, 1, 1, "LC1", None, None, None,
               identifiant("StructuralLoadCombination", "ELS-Car (auto)1")]]
    for i, c in enumerate(cas, 2):
        nom = f"ELS-Car (auto){i}"
        combis.append([nom, None, "SLS (Serviceability Limit State)", None, type_combi, 1, 1, "LC1", 1, 1, c["lc"],
                       identifiant("StructuralLoadCombination", nom)])
    # ---- les charges : le poids propre (SF1) ; les réactions (FF) ; le poids des plots (FP)
    sf = _ligne("StructuralSurfaceAction", "SF1")
    sf[ref["StructuralSurfaceAction"]["colonnes"].index("Value [kN/m2]")] = _entier(round(-PR.GAMMA_BETON * h, 9))
    surfaces = plots["polys"] if plots and plots["type"] == "prefabrique" else m["plaques"]
    ecrits = [_polygone(s) for s in surfaces]
    libres = []
    for i_cas, c in enumerate(cas):
        for p in range(4):
            nom = f"FF{4 * i_cas + p + 1}"
            pg = ecrits[p]
            libres.append([nom, "Z", "Standard", "Uniform", _entier(round(-c["R"][p] / pg["aire"], 3)), c["lc"], pg["X"], pg["Y"], pg["Z"],
                           pg["edges"], "Global", "Length", identifiant("StructuralSurfaceActionFree", nom)])
    if plots:
        for p, q in enumerate(plots["polys"], 1):
            pg = _polygone(q)
            libres.append([f"FP{p}", "Z", "Standard", "Uniform", _entier(round(-PR.GAMMA_BETON * plots["hauteur"], 9)), "LC1", pg["X"], pg["Y"],
                           pg["Z"], pg["edges"], "Global", "Length", identifiant("StructuralSurfaceActionFree", f"FP{p}")])
    out = [{"nom": "LLE_DONNEE", "tableau": None, "lignes": _recapitulatif(el, m, k, resultats, nom_note)},
           {"nom": "Project", "tableau": None, "lignes": project},
           {"nom": "Model", "tableau": None, "lignes": model},
           table("StructuralMaterial", [materiau]),
           table("StructuralCrossSection", [_ligne("StructuralCrossSection", "CS1")]),
           table("StructuralPointConnection", lignes_noeuds),
           table("StructuralSurfaceMember", membres),
           table("StructuralSurfaceMemberOpening", [ouverture])]
    if regions:
        out.append(table("StructuralSurfaceMemberRegion", regions, COLONNES_REGION, "tblStructuralSurfaceMemberRegion"))
    if aretes:
        out.append(table("StructuralCurveEdge", aretes, COLONNES_ARETE, "tblStructuralCurveEdge"))
    out += [table("StructuralSurfaceConnection", sol),
            table("StructuralLoadGroup", [list(li) for li in ref["StructuralLoadGroup"]["lignes"]]),
            table("StructuralLoadCase", lc),
            table("StructuralLoadCombination", combis),
            table("StructuralSurfaceAction", [sf])]
    if libres:
        out.append(table("StructuralSurfaceActionFree", libres))
    return out


def _recapitulatif(el: Dict, m: Dict, k: float, resultats: Optional[List[Dict]], nom_note: str) -> List[List]:
    """La feuille LLE_DONNEE : le récapitulatif du projet, en valeurs (SCIA ne la lit pas)."""
    p, d, g, o, ch = el["projet"], el["dalle"], el["grue"], el["options"], m["chassis"]
    L: List[List] = [["FONDATION DE GRUE — récapitulatif du Générateur SAF"], []]
    L += [["PROJET"], ["Projet", p["projet"] or None], ["Grue", p["grue"] or None], ["Date création", p["creation"] or None],
          ["Date mise à jour", p["mise_a_jour"] or None], ["Indice", p["indice"] or None], ["Note", nom_note or None], []]
    L += [["DALLE"], ["Forme", "Rectangle" if d["forme"] == "rectangle" else "Contour"]]
    if d["forme"] == "rectangle":
        L += [["Lx", d["Lx"], "m"], ["Ly", d["Ly"], "m"]]
    else:
        L += [["Sommets (X ; Y)", ";".join(coord(x) for x, _ in m["contour"]), ";".join(coord(y) for _, y in m["contour"]), "m"]]
    L += [["Épaisseur", d["h"], "m"], ["Béton", d["beton"]], ["E", PR.module_E(d["beton"]), "MPa"], ["Aire", round(G.aire(m["contour"]), 6), "m²"], []]
    L += [["SOL DE CE FICHIER"], ["k (C1z)", k, "MN/m³"], ["C1x", round(k * o["c1x"], 9), "MN/m³"], ["C1y", round(k * o["c1y"], 9), "MN/m³"],
          ["C2x", round(k * o["c2x"], 9), "MN/m"], ["C2y", round(k * o["c2y"], 9), "MN/m"], ["Anneau D2", o["anneau"], "m"], []]
    L += [["GRUE"], ["Châssis", ch["id"] or "saisie"], ["Nom du châssis", ch["nom"] or None], ["Entraxe a", ch["a"], "m"], ["Entraxe b", ch["b"], "m"],
          ["Position ex", g["ex"], "m"], ["Position ey", g["ey"], "m"], ["Centre X", round(m["centre"][0], 6), "m"],
          ["Centre Y", round(m["centre"][1], 6), "m"], ["Angle β", g["beta"], "°"],
          ["Appuis", ("Rectangle" if ch["appuis"] == "rectangle" else "Réels") + (" (saisie)" if ch["saisie"] else "")],
          ["Surface d'appui", ch["dims"]], ["Aire d'une surface", round(ch["aire"], 6), "m²"],
          ["Distance mini plaque–bord", round(m["controles"]["distance"], 3), "m"], ["Distance mini demandée", g["dmin"], "m"]]
    if m["plots"]:
        pl = m["plots"]
        L += [["Plots", "Préfabriqués" if pl["type"] == "prefabrique" else "Coulés en place"], ["Plot a", pl["a"], "m"], ["Plot b", pl["b"], "m"],
              ["Plot hauteur", pl["hauteur"], "m"]]
    else:
        L += [["Plots", "Aucun"]]
    L += [[], ["RÉACTIONS (kN)"], ["Jeu", "Nom", "P1", "P2", "P3", "P4"]]
    L += [[j["numero"], j["nom"] or None] + list(j["P"]) for j in m["jeux"]]
    L += [[], ["POSITIONS DES CHARGES"], ["Cas", "Cas de charge", "Jeu", "P1", "P2", "P3", "P4"]]
    L += [[c["nom"], c["lc"], c["jeu_saisi"]] + list(c["R"]) for c in m["cas"]]
    L += [[], ["BANDES D'INTÉGRATION"], ["Bande", "X début", "Y début", "X fin", "Y fin", "Largeur"]]
    L += [[b["nom"], _mm(b["debut"][0]), _mm(b["debut"][1]), _mm(b["fin"][0]), _mm(b["fin"][1]), b["largeur"]] for b in m["bandes"]]
    L += [[], ["VÉRIFICATION RAPIDE (estimation avant SCIA)"], ["Limite", el["limite"], "kg/cm²", round(el["limite"] * KG_CM2, 3), "kPa"]]
    if resultats:
        L += [["k (MN/m³)", "σ max (kPa)", "σ max (kg/cm²)", "Tassement max (mm)", "Décollé (%)", "SCIA", "Cas"]]
        scia = {x["k"] for x in el["sol"]["k"] if x["scia"]}
        for r in resultats:
            L.append([r["k"], round(r["sigma_max"], 1), round(r["sigma_max"] / KG_CM2, 2), round(r["w_max"], 1), round(r["decolle"], 1),
                      "oui" if r["k"] in scia else None, r["cas_max"]])
        retenus = [r for r in resultats if r["k"] in scia] or resultats
        pire = retenus[premier_max([r["sigma_max"] for r in retenus])]         # (à égalité, le premier : le même partout)
        L += [[], ["CAS DIMENSIONNANT"], ["Cas", pire["cas_max"]], ["k", pire["k"], "MN/m³"], ["σ max", round(pire["sigma_max"], 1), "kPa"]]
    return L


def fichier(el: Dict, k: float, date: Optional[datetime.datetime] = None, resultats: Optional[List[Dict]] = None, nom_note: str = "") -> bytes:
    """Les octets du fichier SAF (.xlsx) du projet pour la raideur ``k``."""
    return CL.ecrire(feuilles(el, k, date, resultats, nom_note))

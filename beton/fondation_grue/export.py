# -*- coding: utf-8 -*-
"""Fondations de grues — l'EXPORT vers le dessin : ce que le calcul transmet à RBAR (plugin AutoCAD d'armatures) et à
AutoCAD, sans rien recalculer. Bibliothèque standard seulement.

Tout part du résultat de ``moteur.verifier`` : les armatures exportées sont les armatures CHOISIES dans la note (celles
que le classeur vérifie), jamais une valeur refaite ici. Trois fichiers :

    rbar(res, infos)   le JSON d'échange : les « modèles de poutre » au format que RBAR importe déjà (palette ›
                       modèles › Importer JSON… : ``{"modelesPoutre": [...]}``, champs nom, bas, haut, basSuivants,
                       lat, etrier {d, espacement, brins}, enrobage — en mm), les types de pieux et le gabarit de massif
                       au format de son ``types.json``, puis la géométrie en plan, toutes les armatures et le contrôle ;
    tableau(res)       le tableau des armatures (CSV, point-virgule, virgule décimale) ;
    dxf(res, infos)    le plan de principe (DXF R12, unités : cm) : pieux en cercles au diamètre réel, contours, axes.

``controle(res)`` dit, AVANT l'envoi, ce qui part (donnée, valeur, cellule de saisie, état), ce qui MANQUE (ce que ni
le classeur ni l'export ne fixent : RBAR le prend dans le standard du bureau, ou le dessinateur le pose) et ce qui
BLOQUE (une vérification non conforme : le fichier RBAR est alors refusé).

Ce que RBAR ne sait pas recevoir aujourd'hui est écrit dans ``LIMITES_RBAR`` et dans le fichier lui-même.
"""
import json

from . import entree as E
from . import geometrie as G

FORMAT = "stab.fondation_grue"
VERSION = 1
LIMITES_RBAR = (
    "RBAR importe les modèles de poutre (« modelesPoutre ») : palette › liste des modèles › Importer JSON. Il ne lit "
    "pas encore le reste de ce fichier (géométrie, pieux, armatures de renfort, perfos).",
    "Un modèle RBAR porte un seul nombre de brins : deux zones d'étriers différentes font deux modèles, donc deux coupes.",
    "Formes, retours, ancrages, recouvrements et longueur des zones d'étriers ne viennent pas du calcul : RBAR applique "
    "le standard du bureau.",
)
CALQUES = {"beton": "STAB_BETON", "pieux": "STAB_PIEUX", "axes": "STAB_AXES", "grue": "STAB_GRUE", "cotes": "STAB_COTES",
           "textes": "STAB_TEXTES"}


def _cellules(f: dict, chemin: str) -> str:
    """Les cellules de saisie d'une armature (« G129, G130 ») : la correspondance de ``entree.SAISIES``."""
    s = E.SAISIES[f["type"]]
    return ", ".join(cel for cle, cel in s.items() if cle == chemin or cle.startswith(chemin + "/"))


def controle(res: dict) -> dict:
    """Ce qui part, ce qui manque, ce qui bloque."""
    f = res["fondation"]
    lignes = [{"element": a["nom_element"], "poste": a["designation"], "valeur": a["texte"], "etat": a["etat"],
               "cellules": _cellules(f, a["saisie"]) if a.get("saisie") else "", "genre": a["genre"]}
              for a in res["armatures"]]
    manques = []
    if f["type"] == E.SEMELLES8:
        manques.append("Armatures latérales : le classeur prescrit « 2Ø12 tous les 30 à 40 cm » sans en fixer le nombre "
                       "de rangées — à poser par le dessinateur (elles partent dans le tableau, pas dans le modèle RBAR).")
    else:
        manques.append("Enrobage : le classeur de cette variante n'en a pas (hauteur utile forfaitaire h − 50 mm) — le "
                       "modèle RBAR part sans enrobage, RBAR prend celui du standard du bureau.")
    manques += ["Longueur des zones d'étriers (extrémités, partie centrale) : non fixée par le calcul.",
                "Formes, retours, ancrages et recouvrements des barres : non fixés par le calcul (standard du bureau).",
                "Étriers de renfort au droit des pieux et perfos : hors du modèle de poutre de RBAR — à poser d'après le "
                "tableau des armatures."]
    bloquants = [f"{v['nom']} ({next(b['nom'] for b in res['blocs'] if b['cle'] == v['bloc'])}) : {v['etat_classeur']}"
                 for v in res["verifications"] if v["etat"] == "nok"]
    bloquants += [f"{b['nom']} : cisaillement au-delà de la zone 3" for b in res["blocs"]
                  if any(i.get("etat") == "nok" for i in b["informations"])]
    return {"lignes": lignes, "manques": manques, "bloquants": bloquants, "pret": not bloquants,
            "a_regarder": [f"{v['nom']} : {v['etat_classeur']}" for v in res["verifications"] if v["etat"] == "warn"],
            "ecarts": [f"{e['code']} — {e['titre']}" for e in res["ecarts"]]}


def _lit(groupes):
    return {"n": groupes[0]["n"], "d": groupes[0]["d"]} if groupes else {"n": 0, "d": 0}


def modeles_rbar(res: dict) -> list:
    """Un modèle de poutre RBAR par pièce et par zone d'étriers : nombre et Ø des barres basses et hautes (le second
    groupe d'une face part en 2e lit), étriers (Ø, pas, brins), enrobage quand le classeur en a un. Longueurs en mm."""
    f = res["fondation"]
    note = f["nom"]
    par = {}
    for a in res["armatures"]:
        par.setdefault(a["element"], {})[a["poste"]] = a
    modeles = []
    for piece in res["pieces"]:
        cle = piece["cle"]
        if cle not in par or "inf" not in par[cle]:
            continue
        p = par[cle]
        zones = [z for z in ("etriers_extremites", "etriers_appui", "etriers", "etriers_centre", "etriers_travee") if z in p]
        for z in zones:
            et = p[z]
            nom = f"{note} — {piece['nom']} {piece['b']:g}x{piece['h']:g}" + (f" — {et['zone']}" if len(zones) > 1 else "")
            m = {"nom": nom, "bas": _lit(p["inf"]["groupes"]), "haut": _lit(p["sup"]["groupes"]),
                 "lat": {"nParFace": 0, "d": 12},
                 "etrier": {"d": et["d"], "espacement": et["pas"], "brins": et["brins"]}}
            for face, champ in (("inf", "basSuivants"), ("sup", "hautSuivants")):
                if len(p[face]["groupes"]) > 1:
                    m[champ] = [{"n": g["n"], "d": g["d"]} for g in p[face]["groupes"][1:]]
            if "enrobage" in f:
                m["enrobage"] = f["enrobage"]
            m["domaine"] = {"largeurMin_mm": piece["b"] * 10, "largeurMax_mm": piece["b"] * 10,
                            "hauteurMin_mm": piece["h"] * 10, "hauteurMax_mm": piece["h"] * 10}
            m["stab"] = {"module": "fondation_grue", "note": note, "piece": cle, "nombre_de_pieces": piece["nombre"],
                         "zone": et["zone"], "section_cm": [piece["b"], piece["h"]],
                         "etats": {"inf": p["inf"]["etat"], "sup": p["sup"]["etat"], "etriers": et["etat"]},
                         "saisies": {"inf": _cellules(f, p["inf"]["saisie"]), "sup": _cellules(f, p["sup"]["saisie"]),
                                     "etriers": _cellules(f, et["saisie"])}}
            modeles.append(m)
    return modeles


def _pieux_rbar(res: dict) -> list:
    """Le type de pieu au format du ``types.json`` de RBAR (tableau « pieux »)."""
    p = next(x for x in res["pieces"] if x["cle"] == "pieux")
    D = p["D"]
    t = {"code": f"PX{D:g}".replace(".", "_"), "diametre_mm": D * 10, "etat": "ELS"}
    if "capacite" in p:
        charge_t = p["capacite"] if p["unite_capacite"] == "T" else p["capacite"] / 10
        t["charge_t"] = charge_t
        t["nom"] = f"Pieu Ø{D:g} — {charge_t:g} T (ELS)"
    else:
        t["nom"] = f"Pieu Ø{D:g}"
    return [t]


def _massifs_rbar(res: dict) -> list:
    """La semelle sur pieux au format du ``types.json`` de RBAR (tableau « fondations », sorte « massif ») quand son
    débord au nu des pieux est le même dans les deux sens — sinon rien (RBAR ne connaît qu'un débord)."""
    f = res["fondation"]
    if f["type"] != E.SEMELLES8:
        return []
    sm, pi_ = f["semelle"], f["pieux"]
    d1, d2 = (sm["b"] - pi_["D"]) / 2, (sm["L"] - pi_["entraxe"] - pi_["D"]) / 2
    if abs(d1 - d2) > 0.05 or d1 < 0:
        return []
    return [{"code": f"MA{sm['h']:g}".replace(".", "_"), "sorte": "massif", "hauteur_mm": sm["h"] * 10, "debord_mm": d1 * 10,
             "nom": "Massif sur pieux — h. " + f"{sm['h']:g}".replace(".", ",") + ", débord " + f"{d1:g}".replace(".", ",")}]


def _mm(points):
    return [[round(x * 10, 3), round(y * 10, 3)] for x, y in points]


def rbar(res: dict, infos: dict = None) -> dict:
    """Le fichier d'échange complet (JSON-sérialisable). ValueError si une vérification est non conforme : on ne
    transmet pas au dessin une armature que la note refuse."""
    c = controle(res)
    if not c["pret"]:
        raise ValueError("Export refusé : " + " ; ".join(c["bloquants"]) + ". Corriger la note d'abord.")
    f = res["fondation"]
    g = G.plan(f)
    infos = infos or {}
    return {
        "format": FORMAT, "version": VERSION,
        "note": {"nom": f["nom"], "projet": f.get("projet", ""), "grue": f.get("grue", ""), "schema": f["type"],
                 "libelle": E.LIBELLES_TYPE[f["type"]], "classeur": res["classeur"], "statut": res["statut"],
                 "date": infos.get("date", ""), "indice": infos.get("indice", ""), "chantier": infos.get("chantier", "")},
        "limites": list(LIMITES_RBAR),
        "modelesPoutre": modeles_rbar(res),
        "pieux": _pieux_rbar(res), "fondations": _massifs_rbar(res),
        "materiaux": {"beton": f["beton"], "acier": f["acier"]},
        "pieces": res["pieces"],
        "implantation": {
            "unite": "mm", "repere": "origine au centre du châssis ; x vers la droite, y vers le haut",
            "pieds": [{"nom": p["nom"], "x": p["x"] * 10, "y": p["y"] * 10} for p in g["pieds"]],
            "pieux": [{"nom": p["nom"], "x": round(p["x"] * 10, 3), "y": round(p["y"] * 10, 3), "diametre": p["D"] * 10}
                      for p in g["pieux"]],
            "pieces": [{"cle": p["cle"], "nom": p["nom"], "largeur": p["b"] * 10, "contour": _mm(p["contour"]),
                        "axe": _mm(p["axe"])} for p in g["pieces"]],
            "paroi": None if not g["paroi"] else {k: v * 10 for k, v in g["paroi"].items()},
            "conventions": g["conventions"]},
        "armatures": res["armatures"],
        "controle": c,
    }


def rbar_texte(res: dict, infos: dict = None) -> str:
    return json.dumps(rbar(res, infos), ensure_ascii=False, indent=2)


# ------------------------------------------------------------------ le tableau des armatures (CSV)
def _fr(x, nd=0):
    if x is None:
        return ""
    return f"{x:.{nd}f}".replace(".", ",")


def tableau(res: dict) -> str:
    """Le tableau des armatures : une ligne par poste, dans l'ordre de la note."""
    f = res["fondation"]
    lignes = [["Élément", "Poste", "Armature", "Nombre ou brins", "Ø (mm)", "Pas (mm)", "Zone", "Section choisie (mm²)",
               "Section requise (mm²)", "Pas calculé (mm)", "État", "Cellules de saisie du classeur"]]
    mots = {"ok": "OK", "warn": "LIMITE", "nok": "PAS OK", "info": ""}
    for a in res["armatures"]:
        if a["genre"] == "barres":
            for g in a["groupes"]:
                lignes.append([a["nom_element"], a["designation"], f"{g['n']}Ø{g['d']}", str(g["n"]), str(g["d"]), "", "",
                               _fr(a["section"]), _fr(a["requis"]), "", mots[a["etat"]], _cellules(f, a["saisie"])])
        elif a["genre"] == "etriers":
            lignes.append([a["nom_element"], a["designation"], a["texte"], str(a["brins"]), str(a["d"]), _fr(a["pas"]),
                           a["zone"], "", "", _fr(a["pas_calcule"]), mots[a["etat"]], _cellules(f, a["saisie"])])
        elif a["genre"] == "prescription":
            lignes.append([a["nom_element"], a["designation"], a["texte"], "", str(a["d"]), "300 à 400", "", "", "", "",
                           "prescription", ""])
        else:
            n = a.get("brins", a.get("n"))
            lignes.append([a["nom_element"], a["designation"], a["texte"], str(n), str(a["d"]), "", "", _fr(a["section"]),
                           _fr(a["requis"]), "", mots[a["etat"]], _cellules(f, a["saisie"])])
    return "\n".join(";".join('"' + c.replace('"', '""') + '"' if (";" in c or '"' in c) else c for c in ligne)
                     for ligne in lignes) + "\n"


def tableau_cas(res: dict) -> str:
    """Toutes les colonnes des grands tableaux du classeur (8 pieux sur semelles) : pour chaque position de tolérance
    (A.a, A.b, B.a, B.b), chaque cas de la grue dans chaque orientation — les charges des deux pieds de la poutre, les
    réactions, les efforts tranchants et les moments, tels que les feuilles « R+V+M support chassis » les donnent."""
    lignes = [["Position", "Feuille du classeur", "Colonne", "Cas", "Orientation", "P1 (kN)", "P2 (kN)", "H/4 (kN)", "RA (kN)", "RB (kN)",
               "RB respectif (kN)", "Symétrique", "V max (kN)", "V central (kN)", "M horizontal équivalent (kN.m)", "M max (kN.m)",
               "M max équivalent (kN.m)"]]
    for t in res["tableau_cas"]:
        for c in t["colonnes"]:
            lignes.append([t["sous_cas"], t["feuille"], c["libelle"], str(c["cas"]), c["orientation"], _fr(c["P1"], 2), _fr(c["P2"], 2),
                           _fr(c["H4"], 3), _fr(c["RA"], 3), _fr(c["RB"], 3), _fr(c["RB_resp"], 3), "oui" if c["symetrique"] else "non",
                           _fr(c["V_max"], 3), _fr(c["V_central"], 3), _fr(c["M_hor_eq"], 3), _fr(c["M_max"], 3), _fr(c["M_max_eq"], 3)])
    return "\n".join(";".join('"' + x.replace('"', '""') + '"' if (";" in x or '"' in x) else x for x in ligne)
                     for ligne in lignes) + "\n"


# ------------------------------------------------------------------ le plan de principe (DXF R12, cm)
def _ligne(x1, y1, x2, y2, calque):
    return ["0", "LINE", "8", calque, "10", f"{x1:.3f}", "20", f"{y1:.3f}", "11", f"{x2:.3f}", "21", f"{y2:.3f}"]


def _cercle(x, y, r, calque):
    return ["0", "CIRCLE", "8", calque, "10", f"{x:.3f}", "20", f"{y:.3f}", "40", f"{r:.3f}"]


def _texte(x, y, h, s, calque):
    return ["0", "TEXT", "8", calque, "10", f"{x:.3f}", "20", f"{y:.3f}", "40", f"{h:.3f}", "1", _ascii(s)]


def _polygone(points, calque):
    out = ["0", "POLYLINE", "8", calque, "66", "1", "70", "1"]
    for x, y in points:
        out += ["0", "VERTEX", "8", calque, "10", f"{x:.3f}", "20", f"{y:.3f}"]
    return out + ["0", "SEQEND", "8", calque]


_SANS_ACCENT = str.maketrans("àâäçéèêëîïôöùûüÀÂÄÇÉÈÊËÎÏÔÖÙÛÜØø’«»—–×≤≥·", "aaaceeeeiioouuuAAACEEEEIIOOUUUOo'\"\"--x<>.")


def _ascii(s: str) -> str:
    """Un texte sûr pour un DXF R12 (ASCII, une ligne) : les accents tombent, le reste hors ASCII devient « ? »."""
    s = " ".join(str(s).split()).translate(_SANS_ACCENT)
    return "".join(ch if 32 <= ord(ch) < 127 else "?" for ch in s)


def dxf(res: dict, infos: dict = None) -> str:
    """Le plan de principe en DXF R12 (ASCII), en cm : contours des poutres et semelles, axes, pieux en cercles au
    diamètre réel (``RBARPIEU › Objets`` les reprend), pieds de la grue, cotes principales en lignes et textes."""
    f = res["fondation"]
    g = G.plan(f)
    infos = infos or {}
    C = CALQUES
    x0, y0, x1, y1 = g["etendue"]
    ht = max((x1 - x0), (y1 - y0)) / 60                      # hauteur des textes
    e = []
    for p in g["pieces"]:
        e += _polygone(p["contour"], C["beton"])
        (ax, ay), (bx, by) = p["axe"]
        e += _ligne(ax, ay, bx, by, C["axes"])
    for p in g["pieux"]:
        e += _cercle(p["x"], p["y"], p["D"] / 2, C["pieux"])
        e += _texte(p["x"] + p["D"] / 2 + ht / 2, p["y"] + p["D"] / 2, ht, p["nom"], C["textes"])
    if g["paroi"]:
        w = g["paroi"]
        y, pas = w["y0"], 0.8 * w["D"]
        while y <= w["y1"] + 1e-9:
            e += _cercle(w["x"], y, w["D"] / 2, C["pieux"])
            y += pas
        e += _texte(w["x"] + w["D"], w["y1"], ht, "paroi de pieux secants (figuree)", C["textes"])
    c = g["cote_pied"] / 2
    for p in g["pieds"]:
        e += _polygone([(p["x"] - c, p["y"] - c), (p["x"] + c, p["y"] - c), (p["x"] + c, p["y"] + c), (p["x"] - c, p["y"] + c)], C["grue"])
        e += _texte(p["x"] - ht / 3, p["y"] - ht / 2, ht, p["nom"], C["grue"])
    for k in g["cotes"]:
        (ax, ay), (bx, by) = k["a"], k["b"]
        e += _ligne(ax, ay, bx, by, C["cotes"])
        e += _texte((ax + bx) / 2, (ay + by) / 2 + ht / 3, ht, k["texte"], C["cotes"])
    lignes = [f"{f['nom']} - fondation de grue - plan de principe (unites : cm)",
              f"{E.LIBELLES_TYPE[f['type']]} - beton {f['beton']}, acier {f['acier']}",
              f"note du {infos.get('date', '')} indice {infos.get('indice', '')} - etat : {res['statut']}"]
    lignes += [a["nom_element"] + " : " + a["designation"] + " " + a["texte"] for a in res["armatures"]]
    for i, s in enumerate(lignes):
        e += _texte(x0, y0 - (i + 2) * ht * 1.6, ht, s, C["textes"])
    tables = ["0", "SECTION", "2", "TABLES", "0", "TABLE", "2", "LAYER", "70", str(len(C) + 1),
              "0", "LAYER", "2", "0", "70", "0", "62", "7", "6", "CONTINUOUS"]
    for nom, couleur in zip(C.values(), (8, 1, 5, 3, 2, 7)):
        tables += ["0", "LAYER", "2", nom, "70", "0", "62", str(couleur), "6", "CONTINUOUS"]
    tables += ["0", "ENDTAB", "0", "ENDSEC"]
    return "\n".join(["0", "SECTION", "2", "HEADER", "9", "$ACADVER", "1", "AC1009", "0", "ENDSEC"] + tables
                     + ["0", "SECTION", "2", "ENTITIES"] + e + ["0", "ENDSEC", "0", "EOF"]) + "\n"

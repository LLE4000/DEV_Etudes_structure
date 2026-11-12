# -*- coding: utf-8 -*-
"""Fondations de grues — le fichier « rbar.ouvrage » : l'ouvrage calculé par le site, tel que RBAR le dessine (plan de
coffrage, plan d'armatures, bordereau). Bibliothèque standard seulement.

LE FORMAT est celui de RBAR, pas celui du site : dépôt RBAR, ``docs/reference/ouvrages/regles/format-du-fichier.md``
(version 1, avec les ajouts 0.4.27 à 0.4.35), et les quatre ajouts demandés par la conversation RBAR le 12/11/2026
(``feuilles.echelles``, ``cartouche``, ``revisions`` ; la charge des pieux attend sa clé — en attendant, un encadré).
RBAR ne sait pas ce qu'est une fondation de grue : les noms, les titres et les textes sont ceux de ce fichier.

D'OÙ VIENT CHAQUE VALEUR — rien n'est recalculé ici, rien n'est inventé :
- du RÉSULTAT du moteur (``moteur.verifier``) : les armatures CHOISIES dans la note (celles que le classeur vérifie),
  les sections, les matériaux, l'enrobage, le type de pieu et sa capacité, la charge du pieu le plus chargé ;
- de la GÉOMÉTRIE en plan du site (``geometrie.plan``) : les centres des pieux, les bouts de chaque poutre, les pieds de
  la grue — la même que le dessin de l'écran et le DXF ;
- des OPTIONS DE CONCEPTION de la fondation (``conception.py``, valeurs de Greenwood par défaut) : ce que le plan
  demande et que le calcul ne fixe pas — retours, zones d'étriers serrés, barres latérales, coupes, cages de renfort,
  barres scellées, niveaux, textes, mise en page ;
- de la NOTE du site (``infos`` : son nom, son chantier, son indice, sa date) : le cartouche et l'indice.
Ce qu'aucune de ces sources ne donne reste ABSENT du fichier : RBAR le prend dans le standard du poste, ou ne le dessine
pas, et le dit.

LIMITES (dites par ``possible`` et par le contrôle) : seul le schéma « 8 pieux — 2 poutres support sur 4 semelles »
avec des semelles des deux côtés s'écrit aujourd'hui (Greenwood, BASE) ; la paroi de pieux sécants (côté b) et les
variantes 4 et 8 pieux n'ont pas encore de fichier.

    possible(f)             None, ou la raison pour laquelle cette fondation n'a pas de fichier
    controle(res, infos)    ce qui part (et d'où), ce qui manque, ce qui est à regarder, ce qui bloque
    ouvrage(res, infos)     le fichier (un dict prêt pour json.dumps) ; ValueError si l'export est bloqué
    fichier(res, infos)     {"nom", "type", "texte"} : le fichier à télécharger
"""
import json
import re

from . import conception as CO
from . import entree as E
from . import export as X
from . import geometrie as G

FORMAT, VERSION, UNITE = "rbar.ouvrage", 1, "mm"
SPEC = ("dépôt RBAR, docs/reference/ouvrages/regles/format-du-fichier.md (version 1, ajouts 0.4.27 à 0.4.35) ; "
        "ajouts du 12/11/2026 : feuilles.echelles, cartouche, revisions")
PREFIXES = {"support": "PT", "semelle": "PL"}
NOM_PIEU_MAX, NOM_POUTRE_MAX = 16, 20
ORIGINE_LATERALES = "prescription du classeur"


# ------------------------------------------------------------------ ce qui peut partir
def possible(f: dict):
    """None si la fondation a un fichier « rbar.ouvrage » ; sinon la phrase qui dit pourquoi pas encore."""
    if f["type"] != E.SEMELLES8:
        return (f"le fichier pour RBAR n'existe pas encore pour le schéma « {E.COURTS_TYPE[f['type']]} » : seul le "
                "schéma 8 pieux sur semelles (Greenwood) est décrit aujourd'hui")
    if f.get("cote_b") == E.COTE_PAROI:
        return ("le fichier pour RBAR ne décrit pas encore une paroi de pieux sécants (le format de RBAR n'a ni paroi ni "
                "massif en version 1) : seul le côté b à semelles s'écrit aujourd'hui")
    return None


def _mm(x_cm: float) -> float:
    """cm → mm, arrondi au millième (les coordonnées du site sont en cm)."""
    return round(float(x_cm) * 10.0, 3)


def _num(x: float):
    """Un nombre écrit sans « .0 » quand il est entier (le fichier se lit aussi à la main)."""
    x = float(x)
    return int(x) if x.is_integer() else x


def _fr(x: float, nd: int = 1) -> str:
    s = f"{x:.{nd}f}".replace(".", ",")
    return s[:-2] if nd == 1 and s.endswith(",0") else s


def _date_iso(texte: str):
    """« 12/11/2026 » (la date de la note du site) → « 2026-11-12 » ; None si ce n'est pas une date lisible."""
    m = re.fullmatch(r"\s*(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\s*", str(texte or ""))
    if not m:
        return None
    j, mo, a = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (1 <= j <= 31 and 1 <= mo <= 12):
        return None
    return f"{a:04d}-{mo:02d}-{j:02d}"


# ------------------------------------------------------------------ ce que le calcul et la géométrie donnent
def _armatures(res: dict) -> dict:
    """{élément: {poste: armature}} du résultat du moteur."""
    par = {}
    for a in res["armatures"]:
        par.setdefault(a["element"], {})[a["poste"]] = a
    return par


def _pieces_geo(f: dict) -> dict:
    """La géométrie du plan (cm) : les poutres support et les semelles, dans l'ordre du dessin de l'écran, avec leurs
    bouts d'about en about ; les pieux ; les pieds de la grue."""
    g = G.plan(f)
    out = {"support": [], "semelle": [], "pieux": g["pieux"], "pieds": g["pieds"]}
    for p in g["pieces"]:
        if p["cle"] == "support":
            xs = [x for x, _ in p["contour"]]
            y = p["axe"][0][1]
            out["support"].append({"debut": (min(xs), y), "fin": (max(xs), y), "longueur": max(xs) - min(xs)})
        elif p["cle"] == "semelle":
            (x0, y0), (x1, y1) = p["axe"]
            out["semelle"].append({"debut": (x0, y0), "fin": (x1, y1), "longueur": ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5})
    return out


def _disposition(piece: str, brins: int):
    """La disposition des étriers (règle fixe de Greenwood, ``conception``) ; None : aucune ne convient (RBAR prend
    son standard, et le contrôle le dit)."""
    if brins == 2:
        return "cadreInterieur"
    if brins % 2 or brins < 2:
        return None
    if piece == "semelle" and brins == 4:
        return "deuxCadres"
    return "cadres"


def _longitudinal(retour_mm: float) -> dict:
    """Les extrémités d'une barre longitudinale : un retour à chaque about (règle fixe de Greenwood) ; un retour nul de
    l'option : des barres droites."""
    if retour_mm <= 0:
        return {"extremites": "droites"}
    return {"extremites": "deux", "retour_mm": _num(round(retour_mm, 1))}


def _lit(groupe: dict, retour_mm: float) -> dict:
    return {"n": int(groupe["n"]), "d": int(groupe["d"]), "longitudinal": _longitudinal(retour_mm)}


def _faces(m: dict, arm: dict, retour_mm: float) -> None:
    """Les lits inférieur et supérieur : le premier groupe d'une face est son premier lit, les suivants ses lits
    suivants (comme l'export des modèles de poutre du 11/11)."""
    for poste, cle, suite in (("inf", "bas", "basSuivants"), ("sup", "haut", "hautSuivants")):
        gs = [g for g in arm[poste]["groupes"] if g["n"] > 0]
        if not gs:
            continue
        m[cle] = _lit(gs[0], retour_mm)
        if len(gs) > 1:
            m[suite] = [_lit(g, retour_mm) for g in gs[1:]]


def _laterales(m: dict, arm: dict, c: dict, longueur_mm: float) -> None:
    lat = arm.get("laterales")
    if not lat or c["laterales"] <= 0:
        return
    lg = _longitudinal(c["retour_lat_mm"])
    if c["recouvrement_lat"]:
        lg["recouvrements_mm"] = [_num(round(longueur_mm / 2, 1))]
    m["lat"] = {"nParFace": int(c["laterales"]), "d": int(lat["d"]), "longitudinal": lg}


# ------------------------------------------------------------------ le contrôle avant l'envoi
def _ligne(quoi: str, valeur: str, source: str) -> dict:
    return {"quoi": quoi, "valeur": valeur, "source": source}


def controle(res: dict, infos: dict = None) -> dict:
    """Ce qui part (chaque valeur et d'où elle vient), ce qui manque (absent du fichier), ce qui est à regarder, ce qui
    bloque l'envoi. ``pret`` : le fichier peut partir."""
    f = res["fondation"]
    infos = infos or {}
    base = X.controle(res)
    bloquants = list(base["bloquants"])
    raison = possible(f)
    if raison:
        bloquants.insert(0, raison[:1].upper() + raison[1:])
        return {"lignes": [], "manques": [], "a_regarder": [], "bloquants": bloquants, "pret": False}
    c = f["conception"]
    arm = _armatures(res)
    geo = _pieces_geo(f)
    lignes, manques, a_regarder = [], [], []
    su, sm = f["support"], f["semelle"]
    L_su = _mm(geo["support"][0]["longueur"])
    L_sm = _mm(geo["semelle"][0]["longueur"])
    ext, cen = arm["support"]["etriers_extremites"], arm["support"]["etriers_centre"]
    if ext["d"] != cen["d"]:
        bloquants.append(f"Poutre support : étriers Ø{ext['d']} aux extrémités et Ø{cen['d']} au centre — un modèle de "
                         "RBAR n'a qu'un diamètre d'étrier ; choisir le même Ø dans la note")
    z = c["support"]["zone_serree_mm"]
    if (ext["brins"], ext["pas"]) != (cen["brins"], cen["pas"]) and not 0 < 2 * z < L_su:
        bloquants.append(f"Poutre support : deux zones d'étriers serrés de {_fr(z / 1000, 2)} m ne tiennent pas dans "
                         f"la poutre de {_fr(L_su / 1000, 2)} m (option de conception)")
    for piece, h_cm, et in (("support", su["h"], cen), ("semelle", sm["h"], arm["semelle"]["etriers"])):
        try:
            CO.retour(c, piece, h_cm * 10, f["enrobage"], et["d"])
        except ValueError as e:
            bloquants.append(str(e)[:1].upper() + str(e)[1:])
    for piece, et in (("support", cen), ("support", ext), ("semelle", arm["semelle"]["etriers"])):
        if _disposition(piece, et["brins"]) is None:
            a_regarder.append(f"{'Poutre support' if piece == 'support' else 'Semelle'} : {et['brins']} brins — aucune "
                              "disposition de RBAR ne convient (nombre impair) ; RBAR prend celle de son standard")

    # ce qui part, et d'où
    lignes.append(_ligne("Pieux", f"{len(geo['pieux'])} pieux Ø{f['pieux']['D']:g} cm, capacité "
                         f"{_fr(res['valeurs']['pieux.capacite_T'])} T", "calcul et géométrie du site"))
    for piece, nom in (("support", "Poutres support"), ("semelle", "Semelles")):
        for poste in ("inf", "sup"):
            a = arm[piece][poste]
            lignes.append(_ligne(f"{nom} · {a['designation'].lower()}", a["texte"],
                                 f"calcul · classeur {X.plage(X._cellules(f, a['saisie']))}"))
    for a in (ext, cen, arm["semelle"]["etriers"]):
        nom = "Poutres support" if a["element"] == "support" else "Semelles"
        lignes.append(_ligne(f"{nom} · {a['designation'].lower()}", a["texte"],
                             f"calcul · classeur {X.plage(X._cellules(f, a['saisie']))}"))
    rf = arm["semelle"]["renfort_pieu"]
    cs = c["semelle"]
    lignes.append(_ligne("Cages de renfort au droit des pieux",
                         f"{cs['cage_cadres_long']} + {cs['cage_cadres_travers']} cadres Ø{rf['d']}, "
                         f"{_fr(cs['cage_long_mm'] / 10, 0)} × {_fr(cs['cage_larg_mm'] / 10, 0)} cm",
                         "Ø : calcul ; cadres et emprise : option de conception (Greenwood)"))
    a_regarder.append(f"Cages de renfort : le classeur vérifie {rf['texte']} ; les cages dessinées "
                      f"({cs['cage_cadres_long']} + {cs['cage_cadres_travers']} cadres Ø{rf['d']}) viennent des options "
                      "de conception — la règle de passage des brins aux cadres est à confirmer")
    pf = arm["perfos"]["perfos"]
    if pf["n"] == CO.PERFOS_PAR_MOTIF:
        lignes.append(_ligne("Barres scellées (perfos)", f"{pf['n']}Ø{pf['d']} par plaque, "
                             f"{_fr(c['perfos']['longueur_mm'] / 10, 0)} cm dont {_fr(c['perfos']['scelle_mm'] / 10, 0)} scellés",
                             "nombre et Ø : calcul ; longueurs, motif de pose : option de conception (Greenwood)"))
    else:
        manques.append(f"Barres scellées : {pf['n']}Ø{pf['d']} par plaque — le motif de pose connu est celui de "
                       f"{CO.PERFOS_PAR_MOTIF} barres par plaque : elles ne partent pas, à poser par le dessinateur")
    ligne, pourquoi = CO.mise_en_page(f)
    lignes.append(_ligne("Mise en page des plans", "RBAR choisit" if ligne is None else
                         f"{ligne[1]} · " + " · ".join(f"{k} 1/{v}" for k, v in zip(CO.ECHELLES, ligne[2:])), pourquoi))
    for cle, nom in (("dessus_m", "Niveau du dessus des poutres"), ("recepage_m", "Niveau de recépage des pieux")):
        x = c["niveaux"][cle]
        if x is None:
            manques.append(f"{nom} : non saisi (options de conception) — absent du plan")
        elif x == CO.defauts()["niveaux"][cle]:
            a_regarder.append(f"{nom} : {_fr(x, 2)} m, la valeur de Greenwood (options de conception) — à vérifier "
                              "pour ce chantier")
    manques.append("Charge horizontale des pieux : le classeur ne répartit pas l'effort horizontal entre les pieux — "
                   "absente du plan")
    manques.append("Charge verticale des pieux : RBAR n'a pas encore de clé pour la charge calculée — elle part dans "
                   "l'encadré « Pieux »")
    vides = [k for k, v in (("le chantier", infos.get("chantier")), ("le nom de la note", infos.get("nom")))
             if not str(v or "").strip()]
    if vides:
        manques.append(f"Cartouche : {' et '.join(vides)} " + ("ne sont pas saisis — absents" if len(vides) > 1 else
                                                               "n'est pas saisi — absent") + " du cartouche")
    manques.append("Cartouche : numéro de projet, bâtiment, phase, discipline, vérificateur et intervenants — inconnus "
                   "du site, absents")
    if base["a_regarder"]:
        a_regarder += [f"Calcul : {x}" for x in base["a_regarder"]]
    return {"lignes": lignes, "manques": manques, "a_regarder": a_regarder, "bloquants": bloquants,
            "pret": not bloquants}


# ------------------------------------------------------------------ le fichier
def _origine(f: dict, a: dict) -> str:
    return X.origine(f, a)


def _modele_support(f: dict, arm: dict, c: dict, geo: dict, nom: str) -> dict:
    su, cs = f["support"], c["support"]
    L = _mm(geo["support"][0]["longueur"])
    m = {"nom": nom}
    ext, cen = arm["etriers_extremites"], arm["etriers_centre"]
    _faces(m, arm, CO.retour(c, "support", su["h"] * 10, f["enrobage"], cen["d"]))
    _laterales(m, arm, cs, L)
    et = {"d": int(cen["d"]), "espacement": _num(cen["pas"]), "brins": int(cen["brins"])}
    disp = _disposition("support", max(ext["brins"], cen["brins"]))
    if disp:
        et["disposition"] = disp
    if (ext["brins"], ext["pas"]) != (cen["brins"], cen["pas"]):
        z = _num(cs["zone_serree_mm"])
        et["zones"] = [{"longueur_mm": z, "espacement": _num(ext["pas"]), "brins": int(ext["brins"])},
                       {"espacement": _num(cen["pas"]), "brins": int(cen["brins"])},
                       {"longueur_mm": z, "espacement": _num(ext["pas"]), "brins": int(ext["brins"])}]
    m["etrier"] = et
    m["montage"] = {"d": -1}
    pf = arm.get("perfos")
    if pf and pf["n"] == CO.PERFOS_PAR_MOTIF:
        x0 = geo["support"][0]["debut"][0]
        a0 = _mm(-f["chassis"] / 2 - x0)                   # abscisses des deux pieds le long de la poutre (mm)
        a1 = _mm(f["chassis"] / 2 - x0)
        pts = [[_num(a0 - dx), _num(dy)] for dx, dy in CO.MOTIF_PERFOS]
        # au second pied, le motif retourné, lu de l'about vers l'extérieur comme au premier (l'ordre du plan)
        pts += [[_num(a1 + dx), _num(dy)] for dx, dy in sorted(CO.MOTIF_PERFOS, key=lambda t: (t[0], -t[1]))]
        m["scellements"] = [{"d": int(pf["d"]), "longueur_mm": _num(c["perfos"]["longueur_mm"]),
                             "scelle_mm": _num(c["perfos"]["scelle_mm"]), "remarque": c["perfos"]["remarque"],
                             "points_mm": pts}]
    m["enrobage"] = _num(f["enrobage"])
    return m


def _modele_semelle(f: dict, arm: dict, c: dict, geo: dict, nom: str) -> dict:
    sm, cs = f["semelle"], c["semelle"]
    L = _mm(geo["semelle"][0]["longueur"])
    m = {"nom": nom}
    e = arm["etriers"]
    _faces(m, arm, CO.retour(c, "semelle", sm["h"] * 10, f["enrobage"], e["d"]))
    _laterales(m, arm, cs, L)
    et = {"d": int(e["d"]), "espacement": _num(e["pas"]), "brins": int(e["brins"])}
    disp = _disposition("semelle", e["brins"])
    if disp:
        et["disposition"] = disp
    m["etrier"] = et
    m["montage"] = {"d": -1}
    rf = arm.get("renfort_pieu")
    if rf and cs["cage_cadres_long"] + cs["cage_cadres_travers"] > 0:
        s = _mm(f["pieux"]["entraxe"])
        m["cages"] = [{"d": int(rf["d"]), "positions_mm": [_num(round(L / 2 - s / 2, 3)), _num(round(L / 2 + s / 2, 3))],
                       "longueur_mm": _num(cs["cage_long_mm"]), "largeur_mm": _num(cs["cage_larg_mm"]),
                       "cadresLong": int(cs["cage_cadres_long"]), "cadresTravers": int(cs["cage_cadres_travers"]),
                       "jambe_mm": _num(cs["cage_jambe_mm"])}]
    m["enrobage"] = _num(f["enrobage"])
    return m


def _coupes(piece: str, choix: str, L: float, c: dict, f: dict):
    if choix == "aucune":
        return None
    if choix == "milieu":
        return [_num(round(L / 2, 1))]
    if piece == "support":                                   # « zone serrée et milieu » : au milieu de la zone serrée
        return [_num(round(c["support"]["zone_serree_mm"] / 2, 1)), _num(round(L / 2, 1))]
    return [_num(round(L / 2 + _mm(f["pieux"]["entraxe"]) / 2, 1))]   # semelle : au droit du second pieu


def _famille(piece: str, f: dict, c: dict, geo: dict, modele: str, origines: list) -> dict:
    sec = f[piece]
    designation = c["textes"][piece]
    elements = [{"nom": f"{PREFIXES[piece]}{i + 1}", "debut": [_num(_mm(p["debut"][0])), _num(_mm(p["debut"][1]))],
                 "fin": [_num(_mm(p["fin"][0])), _num(_mm(p["fin"][1]))]} for i, p in enumerate(geo[piece])]
    fam = {"titre": f"Détail {designation}" if designation else "", "designation": designation, "position": "retombee",
           "largeur_mm": _num(_mm(sec["b"])), "hauteur_mm": _num(_mm(sec["h"]))}
    if c["niveaux"]["dessus_m"] is not None:
        fam["dessus_m"] = _num(c["niveaux"]["dessus_m"])
    fam["modele"] = modele
    coupes = _coupes(piece, c[piece]["coupes"], _mm(geo[piece][0]["longueur"]), c, f)
    if coupes:
        fam["coupes_mm"] = coupes
    fam["elements"] = elements
    fam["origines"] = origines
    if not fam["titre"]:
        del fam["titre"]
    if not fam["designation"]:
        del fam["designation"]
    return fam


def ouvrage(res: dict, infos: dict = None) -> dict:
    """Le fichier « rbar.ouvrage » de la fondation (un dict prêt pour ``json.dumps``). ValueError si l'envoi est
    bloqué (une vérification non conforme, un schéma pas encore décrit, une option impossible) : on ne transmet pas au
    dessin ce que la note refuse."""
    ctl = controle(res, infos)
    if not ctl["pret"]:
        raise ValueError("Fichier pour RBAR refusé : " + " ; ".join(ctl["bloquants"]) + ".")
    f = res["fondation"]
    infos = infos or {}
    c = f["conception"]
    arm = _armatures(res)
    geo = _pieces_geo(f)
    su, sm = f["support"], f["semelle"]
    nom_su = f"{f['nom']} — {c['textes']['support'] or 'Poutre support'} {su['b']:g}x{su['h']:g}"
    nom_sm = f"{f['nom']} — {c['textes']['semelle'] or 'Semelle'} {sm['b']:g}x{sm['h']:g}"
    a_su, a_sm = arm["support"], arm["semelle"]
    ext, cen = a_su["etriers_extremites"], a_su["etriers_centre"]
    or_su = [{"poste": "inf", "texte": _origine(f, a_su["inf"])}, {"poste": "sup", "texte": _origine(f, a_su["sup"])}]
    if (ext["brins"], ext["pas"]) != (cen["brins"], cen["pas"]):
        or_su += [{"poste": "etriers", "entreAxes_mm": _num(ext["pas"]), "texte": _origine(f, ext)},
                  {"poste": "etriers", "entreAxes_mm": _num(cen["pas"]), "texte": _origine(f, cen)}]
    else:
        or_su.append({"poste": "etriers", "texte": _origine(f, ext)})
    if "laterales" in a_su and c["support"]["laterales"] > 0:
        or_su.append({"poste": "lat", "texte": _origine(f, a_su["laterales"])})
    m_su = _modele_support(f, a_su | {"perfos": arm.get("perfos", {}).get("perfos")}, c, geo, nom_su)
    if "scellements" in m_su:
        or_su.append({"poste": "scellements", "texte": _origine(f, arm["perfos"]["perfos"])})
    or_sm = [{"poste": "inf", "texte": _origine(f, a_sm["inf"])}, {"poste": "sup", "texte": _origine(f, a_sm["sup"])},
             {"poste": "etriers", "texte": _origine(f, a_sm["etriers"])}]
    if "laterales" in a_sm and c["semelle"]["laterales"] > 0:
        or_sm.append({"poste": "lat", "texte": _origine(f, a_sm["laterales"])})
    m_sm = _modele_semelle(f, a_sm, c, geo, nom_sm)
    if "cages" in m_sm:
        or_sm.append({"poste": "cages", "texte": _origine(f, a_sm["renfort_pieu"])})

    type_pieu = {k: (_num(v) if isinstance(v, float) else v) for k, v in X._pieux_rbar(res)[0].items()}
    pieux = []
    for p in geo["pieux"]:
        q = {"nom": p["nom"], "type": type_pieu["code"], "centre": [_num(_mm(p["x"])), _num(_mm(p["y"]))]}
        if c["niveaux"]["recepage_m"] is not None:
            q["recepage_m"] = _num(c["niveaux"]["recepage_m"])
        pieux.append(q)

    encadres = [{"titre": "Matériaux", "lignes": [{"libelle": "Béton", "valeur": f["beton"]},
                                                  {"libelle": "Acier", "valeur": f["acier"]},
                                                  {"libelle": "Enrobage", "valeur": f"{f['enrobage']:g} mm"},
                                                  "distance entre bord béton et extérieur barres"]}]
    pose = [l for l in c["textes"]["pose"].split("\n") if l.strip()]
    if pose:
        encadres.append({"titre": "Remarques — Placement des barres", "lignes": pose})
    v = res["valeurs"]
    encadres.append({"titre": "Pieux", "lignes": [
        {"libelle": "Pieu le plus chargé", "valeur": f"{_fr(v['pieux.sollicitation_T'])} T (service, sans coefficient)"},
        {"libelle": "Capacité portante", "valeur": f"{_fr(v['pieux.capacite_T'])} T"}]})
    grue = [(lib, str(f.get(cle) or "").strip()) for lib, cle in (("Grue", "grue"), ("Flèche", "fleche"),
                                                                  ("Hauteur sous crochet", "hauteur_crochet"),
                                                                  ("Châssis", "dimensions_chassis"), ("Niveau", "niveau_beton"))]
    grue = [{"libelle": lib, "valeur": val} for lib, val in grue if val]
    grue.append({"libelle": "Charge maximale par pied", "valeur": f"{_fr(X.charge_max_par_pied(f), 0)} kN"})
    encadres.append({"titre": "Grue", "lignes": grue})

    h = _num(_mm(f["chassis"] / 2))
    nom_grue = str(f.get("grue") or "").strip()
    texte_chassis = " ".join(t for t in (c["textes"]["chassis"], nom_grue) if t)
    trace = {"nom": "Châssis", "ferme": True, "points": [[-h, -h], [h, -h], [h, h], [-h, h]]}
    if texte_chassis:
        trace["texte"] = texte_chassis
    traces = [trace, {"nom": "Châssis diagonale 1", "points": [[-h, -h], [h, h]]},
              {"nom": "Châssis diagonale 2", "points": [[-h, h], [h, -h]]}]

    out = {"_doc": (f"Fichier « rbar.ouvrage » écrit par le site du bureau (module Fondations de grues) depuis la note "
                    f"de calcul : la fondation « {f['nom']} ». Les armatures sont celles que la note vérifie ; ce que le "
                    "calcul ne fixe pas vient des options de conception de la fondation. Ne pas le modifier à la main : "
                    "corriger la note et exporter de nouveau."),
           "_source": {"note": str(infos.get("nom") or ""), "fondation": f["nom"], "classeur": res["classeur"],
                       "statut": res["statut"], "format": SPEC},
           "format": FORMAT, "version": VERSION, "unite": UNITE, "nom": f["nom"]}
    if c["textes"]["plan"]:
        out["plan"] = {"titre": c["textes"]["plan"]}
    out["materiaux"] = {"beton": f["beton"], "acier": f["acier"], "enrobage_mm": _num(f["enrobage"])}
    ligne, _ = CO.mise_en_page(f)
    if ligne is not None:
        out["feuilles"] = CO.feuilles(ligne)
    cartouche = {}
    for cle, val in (("projet", infos.get("chantier")), ("titre", infos.get("nom"))):
        if str(val or "").strip():
            cartouche[cle] = str(val).strip()
    date = _date_iso(infos.get("date"))
    if date:
        cartouche["date"] = date
    if cartouche:
        out["cartouche"] = cartouche
    indice = str(infos.get("indice") or "").strip()
    if indice:
        rev = {"indice": indice}
        if date:
            rev["date"] = date
        out["revisions"] = [rev]
    out["modelesPoutre"] = [m_su, m_sm]
    out["typesPieux"] = [type_pieu]
    out["pieux"] = pieux
    out["poutres"] = [_famille("support", f, c, geo, nom_su, or_su), _famille("semelle", f, c, geo, nom_sm, or_sm)]
    out["encadres"] = encadres
    out["traces"] = traces
    return out


def nom_fichier(res: dict, infos: dict = None) -> str:
    return f"{res['fondation']['nom']} - ouvrage RBAR"


def fichier(res: dict, infos: dict = None) -> dict:
    """Le fichier à télécharger : {"nom" (sans extension : le module la pose), "type", "texte"} ; ValueError si bloqué."""
    o = ouvrage(res, infos)
    return {"nom": nom_fichier(res, infos), "type": "application/json", "texte": json.dumps(o, ensure_ascii=False, indent=2) + "\n"}

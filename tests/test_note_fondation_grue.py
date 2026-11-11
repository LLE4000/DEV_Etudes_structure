# -*- coding: utf-8 -*-
"""La NOTE PDF du module « Fondations de grues » sur le site (api_web/note_fondation_grue.py), vérifiée en Python
ordinaire par ce que pdftotext relit du PDF.

  1. TOUTE la note en un PDF valide : la page de garde du site (projet, partie, date, indice, une ligne par fondation
     avec son statut, son taux, ce qui gouverne) puis, par fondation, ses pages de calcul, sa page de diagrammes, son
     plan de principe — sur 13 états : les quatre classeurs, la note riche du relevé (trois fondations), un pieu PAS OK,
     la paroi de pieux sécants avec un second groupe de barres et douze cas de la grue, des noms longs, le cisaillement
     au-delà de la zone 3, un contrôle de géométrie non respecté, les variantes à la limite et non conformes.
  2. PARITÉ avec le moteur de l'écran (``moteur.verifier``), fondation par fondation : le statut, η max et ce qui
     gouverne ; les hypothèses ; CHAQUE donnée saisie avec SA CELLULE du classeur (aucune cellule de
     ``entree.SAISIES`` n'est oubliée) ; chaque formule « symbole = expression = valeurs = résultat » à la lettre
     (l'écriture de ``ecriture.py``), suivie de sa cellule ; chaque verdict (la comparaison de l'écran, le taux, l'état
     du classeur et sa cellule) ; le cisaillement et sa zone ; les contrôles de géométrie ; les écarts (code, titre,
     cellule, les deux chiffres, le message) ; le récapitulatif ; ce que la note ne couvre pas ; la conclusion.
  3. DESSINS : chaque texte des scènes de l'écran (``schemas.vues`` : le plan, M et V) est dans la page qui la porte ;
     le tableau des positions de tolérance (portée, réactions, extremums, l'enveloppe, ce que le classeur retient) ;
     le plan à une échelle NORMALISÉE qui tient dans sa place ; le peintre (repère, échelle des traits et des textes).
  4. MISE EN PAGE : la répartition des blocs (ordre, hauteur des colonnes, un titre jamais seul en bas d'une colonne),
     aucun débordement, la taille du texte (la plus grande qui garde le nombre de pages de la taille normale), une
     grille de fondations tirées au sort sans un refus.
  5. IDENTIFICATION : celle de la note du site (chantier et nom piégés, indice « C », date), sur la garde et dans
     l'en-tête de chaque page ; le nom du fichier ; le contrat du routeur.
  6. REFUS : une note sans fondation, une fondation hors domaine.
  7. COMME DANS LE NAVIGATEUR : sans fontTools ni accélérateurs C, avec le Pillow factice et les seules polices du
     paquet, sans numpy — la note se fait, son texte est celui de CPython, elle ne charge aucun fichier du dépôt hors
     des paquets du site.
  8. MUTATIONS : chacune doit faire rougir un contrôle ; un témoin, aucun.

    /usr/bin/python3.12 tests/test_note_fondation_grue.py
"""
import base64
import collections
import copy
import html
import io
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)
sys.path.insert(0, os.path.join(RACINE, "tests"))

import note_cas_outils as O  # noqa: E402
from api_web import commun as C  # noqa: E402
from api_web import fondation_grue as FG  # noqa: E402
from api_web import note_fondation_grue as NF  # noqa: E402
from api_web import routeur as R  # noqa: E402
from beton.fondation_grue import ecarts as EC  # noqa: E402
from beton.fondation_grue import ecriture as W  # noqa: E402
from beton.fondation_grue import entree as EN  # noqa: E402
from beton.fondation_grue import moteur as MO  # noqa: E402
from beton.fondation_grue import schemas as SC  # noqa: E402

sys.path.insert(0, os.path.join(RACINE, "site", "web"))
try:
    import construire as K  # noqa: E402
finally:
    sys.path.pop(0)

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:700]}"))


INFOS = {"nom": "Façade « Est » & l'atelier \"Nord\"", "chantier": "Résidence d'Été & Cie — « Lot 2 »", "indice": "C", "date": "03/04/2026"}


# ------------------------------------------------------------------ les états : construits par les saisies de l'écran
def etat(saisies=(), exemple="BASE"):
    e = FG.nouvel_etat()
    assert FG.modifier(e, ["exemple"], exemple) is None
    for ch, v in saisies:
        r = FG.modifier(e, ch.split("/"), v)
        assert r is None, (ch, v, r)
    return e


def riche():
    ns = {}
    exec(K.NOTES_RELEVE["fondation_grue"], ns)
    return ns["etat"]


ETATS = {
    "classeur BASE": etat(),
    "classeur Greenwood": etat(exemple="GWW"),
    "variante 4 pieux": etat(exemple="VAR4"),
    "variante 8 pieux": etat(exemple="VAR8"),
    "la note riche du relevé (trois fondations)": riche(),
    "pieu PAS OK": etat([("pieux/capacite", "40")]),
    "paroi, second groupe de barres, douze cas": etat([("cote_b", "paroi"), ("support/inf/1/n", "2"), ("semelle/sup/1/n", "3")] + [("charges/cas/+", "1")] * 6
                                                      + [("charges/cas/7/B", "950"), ("charges/cas/7/A", "400")]),
    "noms longs": etat([("nom", "Fondation de la grue à tour du lot gros œuvre – bâtiment B2"),
                        ("projet", "Chantier de la rue des Deux-Églises, phase 2 — infrastructure et parkings (04)"),
                        ("grue", "Liebherr 280 EC-H 12 Litronic, flèche de 60 m, hauteur sous crochet 52,3 m env.")]),
    "cisaillement au-delà de la zone 3": etat([("support/b", "60")]),
    "contrôle de géométrie non respecté": etat([("semelle/h", "70")]),
    "variante 4 pieux à la limite": etat([("poutre/inf/n", "3")], "VAR4"),
    "variante 4 pieux, pieu PAS OK": etat([("pieux/capacite", "500")], "VAR4"),
    "variante 8 pieux non conforme": etat([("primaire/inf/n", "4")], "VAR8"),
}

# ------------------------------------------------------------------ relire le PDF
_EXPOSANTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
_BLANCS = re.compile(r"[\s\u00a0\u202f]+")
_RETIRES = re.compile(r"[()_√|${}]")


def plat(t: str) -> str:
    """Un texte comparable à ce que pdftotext relit : sans blanc (les insécables compris), sans parenthèses ni barres
    (le compositeur les trace), sans marque d'indice ; un exposant est un chiffre."""
    return _RETIRES.sub("", _BLANCS.sub("", str(t))).translate(_EXPOSANTS)


def formule(fx) -> str:
    """Une formule telle qu'elle doit se relire : ses morceaux joints par « = », son résultat, puis sa cellule."""
    parts, res = W.morceaux(fx)
    return plat(" = ".join(parts + [res])) + (fx.get("cellule") or "")


def pages_de(octets):
    return O.pages(octets, brut=True)


def decouper(tx, fondations):
    """[(calcul, diagrammes, plan)] : les pages de chaque fondation, reconnues à leur titre, dans l'ordre ; None si
    l'ordre ou les titres ne sont pas ceux attendus."""
    out, i = [], 1
    for f in fondations:
        nom = f.get("nom") or NF.TITRE
        calc, diag = [], []
        while i < len(tx) and f"{NF.TITRE} — {nom}" in O.norm(tx[i]) and "DIAGRAMMES M ET V" not in tx[i] and "PLAN DE PRINCIPE" not in tx[i]:
            calc.append(tx[i])
            i += 1
        while i < len(tx) and f"DIAGRAMMES M ET V — {nom}" in O.norm(tx[i]):
            diag.append(tx[i])
            i += 1
        if i >= len(tx) or f"PLAN DE PRINCIPE — {nom}" not in O.norm(tx[i]) or not calc or not diag:
            return None
        out.append((calc, diag, [tx[i]]))
        i += 1
    return out if i == len(tx) else None


def cellules_lues(texte: str) -> set:
    """Les cellules du classeur écrites dans un texte : « C34 », et toutes celles d'une plage « G129:G132 »."""
    out = set(re.findall(r"(?<![A-Za-z0-9])[A-Z]\d{1,3}(?![0-9])", texte))
    for col, a, b in re.findall(r"(?<![A-Za-z0-9])([A-Z])(\d{1,3}):\1(\d{1,3})(?![0-9])", texte):
        out |= {f"{col}{k}" for k in range(int(a), int(b) + 1)}
    return out


def val_ecart(x) -> str:
    return html.unescape(FG._val_ecart(x))


# --- des attendus écrits ICI, sans passer par le générateur (un contrôle qui appellerait la fonction qu'il contrôle ne
#     verrait pas sa mutation) : la comparaison est celle de l'ÉCRAN, les dimensions et les efforts sont recomposés
def comparaison_ecran(v) -> str:
    return html.unescape(FG._comparaison(v, etat=False))


def dimensions_attendues(p) -> str:
    g = FG._g
    if p["cle"] == "pieux":
        return (f"Ø {g(p['D'])} cm" + (f" · entraxe {g(p['entraxe'])} cm" if p.get("entraxe") else "")
                + (f" · capacité {g(p['capacite'])} {p.get('unite_capacite', '')}".rstrip() if p.get("capacite") else ""))
    return (f"{g(p['b'])} × {g(p['h'])} cm" + (f" · longueur {g(p['longueur'])} cm" if p.get("longueur") else "")
            + (f" · portée {g(p['portee'])} cm" if p.get("portee") else ""))


def efforts_attendus(r, b) -> str:
    s = b["piece"]
    el = next((x for x in r["elements"] if x["cle"] == b["cle"]), None)
    for sym, cle, unite in (("Mmax", "M_max", "kN·m"), ("Mmin", "M_min", "kN·m"), ("Vmax", "V_max", "kN")):
        if el is not None and el["dimensionnant"].get(cle) is not None:
            s += f" · {sym} = {FG._nu(el['dimensionnant'][cle], unite)}"
    return s


def origine_attendue(el) -> str:
    """D'où les abscisses d'un élément sont comptées, lu sur ses appuis : une pièce dont les deux appuis encadrent
    x = 0 à égale distance est comptée de son milieu ; les autres, de l'appui A à sa place prévue."""
    xa, xb = el["positions"][-1]["appuis"]
    return "depuis le milieu de la pièce" if abs(xa + xb) < 1e-9 else "depuis l'appui A à sa place prévue"


def fautes(e, octets) -> list:
    """Ce que la note ``octets`` de l'état ``e`` ne dit pas comme le moteur de l'écran ; liste vide = parité."""
    tx = pages_de(octets)
    fs = e["fondations"]
    out = []
    groupes = decouper(tx, fs)
    if groupes is None:
        return [f"les pages ne sont pas « calcul, diagrammes, plan » de chaque fondation, dans l'ordre ({len(tx)} pages)"]
    garde = plat(tx[0])
    for f, (p_calc, p_diag, p_plan) in zip(fs, groupes):
        r = MO.verifier(f)
        ff, m, nom = r["fondation"], r["materiaux"], f.get("nom") or NF.TITRE
        calc, diag, plan = plat(" ".join(p_calc)), plat(" ".join(p_diag)), plat(" ".join(p_plan))
        brut = " ".join(p_calc + p_diag + p_plan)

        def a(zone, attendu, quoi):
            if plat(attendu) not in zone:
                out.append(f"{nom} · {quoi} : « {attendu} » absent")

        # --- la garde, le bandeau
        v_gov = next(x for x in r["verifications"] if x["cle"] == r["gouvernant"]["cle"])
        b_gov = next(b for b in r["blocs"] if b["cle"] == v_gov["bloc"])
        eta = "η max " + FG._pct(r["taux_max"])
        gouverne = f"gouverne : {FG._nom_bloc(b_gov).lower()}, {FG._court(v_gov).lower()}"
        if plat(f"{nom}{EN.COURTS_TYPE[ff['type']]} · {m['beton']} · {m['acier']} · {eta} · {gouverne}") not in garde:
            out.append(f"{nom} · garde : la ligne du sommaire (schéma, matériaux, η max, ce qui gouverne) absente")
        if len(p_diag) != (len(r["elements"]) + 1) // 2:
            out.append(f"{nom} : {len(p_diag)} page(s) de diagrammes pour {len(r['elements'])} élément(s)")
        tete = plat(p_calc[0])
        a(tete, NF.STATUTS[r["statut"]] + eta, "bandeau (statut et η max)")
        a(tete, f"— {gouverne} · {comparaison_ecran(v_gov)} · état du classeur : {v_gov['etat_classeur']}", "bandeau (ce qui gouverne)")
        # --- les hypothèses
        a(calc, f"fiche {r['classeur']}", "hypothèses (le classeur)")
        a(calc, EN.LIBELLES_TYPE[ff["type"]], "hypothèses (le schéma)")
        if ff["type"] == EN.SEMELLES8:
            a(calc, f"côté b : {EN.LIBELLES_COTE_B[ff['cote_b']]}", "hypothèses (le côté b)")
        cel = EN.SAISIES[ff["type"]]
        a(calc, f"{m['beton']}{cel['beton']} · R'wk = {FG._g(m['Rwk'])} N/mm² · σ'b = {C.fr(m['sigma_b'], 2)} N/mm²", "hypothèses (béton)")
        a(calc, f"{m['acier']}{cel['acier']} · σadm = {C.fr(m['sigma_adm'], 2)} N/mm² · μ = {C.fr(m['mu'], 4)}", "hypothèses (acier)")
        a(calc, "zones 1 / 2 / 3 jusqu'à " + " / ".join(C.fr(x, 2) for x in m["tau_zones"]) + " N/mm²", "hypothèses (cisaillement)")
        for cle in EN.CHAMPS_TEXTE:
            if cle in cel and str(ff.get(cle) or "").strip():
                a(calc, f"{ff[cle]}{cel[cle]}", f"texte « {cle} » et sa cellule")
        # --- les données saisies, chacune avec sa cellule
        for _titre, champs in NF.DONNEES[ff["type"]]:
            for lib, chemin, unite in champs:
                a(calc, f"{lib} = {FG._g(EN.lire(ff, chemin))} {unite}{cel[chemin]}", f"donnée « {chemin} » et sa cellule")
        for lib, chemin, unite in NF.CHARGES:
            a(calc, f"{lib} = {FG._g(EN.lire(ff, chemin))} {unite}{cel[chemin]}", f"charge « {chemin} » et sa cellule")
        if ff["type"] == EN.SEMELLES8:
            for i, cas in enumerate(ff["charges"]["cas"]):
                cs = [cel.get(f"charges/cas/{i}/{p}") for p in EN.PIEDS]
                plage = f"{cs[0]}:{cs[-1]}" if all(cs) else "—"
                a(calc, cas["nom"] + EN.LIBELLES_ETAT_CAS[cas["etat"]] + "".join(FG._g(cas[p]) for p in EN.PIEDS) + plage, f"cas de la grue n° {i + 1}")
        else:
            a(calc, f"P = {FG._g(ff['charges']['P'])} kN{cel['charges/P']}", "charge par pied et sa cellule")
        manquent = sorted(set(cel.values()) - cellules_lues(brut))
        if manquent:
            out.append(f"{nom} : cellules de saisie du classeur absentes de la note : {manquent[:8]}")
        # --- les blocs du classeur : sollicitations, vérifications, cisaillement, informations
        for i, b in enumerate(r["blocs"], 1):
            t = NF.taux_bloc(b)
            a(calc, f"{i} · {b['nom'].upper()}", f"titre du bloc {i}")
            a(calc, NF.STATUTS[b["etat"]] + (f" · {FG._pct(t)}" if t is not None else ""), f"pastille du bloc {i}")
            a(calc, efforts_attendus(r, b), f"pièce et sollicitations retenues du bloc {i}")
            for fx in b["sollicitations"]:
                if formule(fx) not in calc:
                    out.append(f"{nom} · {b['cle']} : sollicitation absente ou sans sa cellule — {' = '.join(W.morceaux(fx)[0])}")
            for v in b["verifications"]:
                a(calc, v["nom"].upper(), f"nom de {v['cle']}")
                a(calc, v["piece"], f"pièce de {v['cle']}")
                for fx in v["formules"]:
                    if formule(fx) not in calc:
                        out.append(f"{nom} · {v['cle']} : formule absente ou sans sa cellule — {' = '.join(W.morceaux(fx)[0])}")
                droite = (f"{FG._pct(v['taux'])} · " if v["taux"] is not None else "") + f"{v['etat_classeur']} ({v['cellule_etat']})"
                a(calc, comparaison_ecran(v) + droite, f"verdict de {v['cle']}")
            for info in b["informations"]:
                if info.get("formule"):
                    if formule(info["formule"]) not in calc:
                        out.append(f"{nom} · {b['cle']} : formule du cisaillement absente")
                    a(calc, (info.get("texte") or "") + str(info.get("zone") or "").capitalize(), f"zone de cisaillement de {b['cle']}")
                else:
                    txt = info["nom"]
                    if info.get("valeur") is not None:
                        txt += f" = {FG._nu(info['valeur'], info.get('unite', ''))}"
                    if info.get("texte"):
                        txt += f" · {info['texte']}"
                    a(calc, txt + (info.get("cellule") or ""), f"information « {info['nom']} » de {b['cle']}")
            if b.get("ecarts"):
                a(calc, "dans ce bloc : " + ", ".join(dict.fromkeys(x["code"] for x in b["ecarts"])), f"renvoi des écarts de {b['cle']}")
        # --- contrôles, écarts, récapitulatif, limites, conclusion
        for c in r["controles"]:
            signe = "≥" if " ≥ " in c["texte"] else "≤"
            vu = signe if c["ok"] else ("<" if signe == "≥" else ">")
            a(calc, f"{FG._g(c['valeur'])} {vu} {FG._g(c['limite'])} {c['unite']}{c.get('cellule') or ''}{'OK' if c['ok'] else 'NON'}", f"contrôle « {c['cle']} »")
            mots = plat(c["texte"])
            if mots not in calc and not all(plat(x) in calc for x in c["texte"].split(" ")):
                out.append(f"{nom} · contrôle « {c['cle']} » : son texte absent")
        ko = sum(1 for c in r["controles"] if not c["ok"])
        if r["controles"]:
            a(calc, f"{ko} NON RESPECTÉ{'S' if ko > 1 else ''}" if ko else "RESPECTÉS", "pastille des contrôles de géométrie")
        for x in r["ecarts"]:
            mot = "change le verdict" if x["change_verdict"] else EC.LIBELLES_GRAVITE.get(x["gravite"], "écart")
            if plat(mot) + plat(f"{x['code']} · {x['titre']}") not in calc and plat(f"{x['code']} · {x['titre']}") not in calc:
                out.append(f"{nom} · écart {x['code']} : son titre absent")
            u = f" {x.get('unite') or ''}".rstrip()
            if x.get("cellule"):
                a(calc, f"cellule {x['cellule']}", f"écart {x['code']} (cellule)")
            if x.get("classeur") is not None:
                a(calc, f"classeur {val_ecart(x['classeur'])}{u}", f"écart {x['code']} (valeur du classeur)")
            if x.get("redemontre") is not None:
                a(calc, f"redémontré {val_ecart(x['redemontre'])}{u}", f"écart {x['code']} (valeur redémontrée)")
            a(calc, x["message"], f"écart {x['code']} (message)")
        if r["ecarts"]:
            a(calc, f"{len(r['ecarts'])} À CONNAÎTRE", "pastille des écarts")
        noms = {b["cle"]: FG._nom_bloc(b) for b in r["blocs"]}
        for v in r["verifications"]:
            a(calc, f"{noms[v['bloc']]} · {FG._court(v).lower()}{comparaison_ecran(v)}{v['etat_classeur']} ({v['cellule_etat']})"
                    + (FG._pct(v["taux"]) if v["taux"] is not None else "—"), f"récapitulatif, ligne de {v['cle']}")
        for x in MO.NON_COUVERT:
            a(calc, x, "ce que la note ne couvre pas")
        a(calc, f"CONCLUSION Taux maximal {FG._pct(r['taux_max'])} — {FG._nom_bloc(b_gov).lower()}, {FG._court(v_gov).lower()}", "conclusion (taux et ce qui gouverne)")
        if r["statut"] == MO.NOK:
            a(calc, "La fondation n'est PAS conforme en l'état", "conclusion (non conforme)")
        for x in FG._a_regarder(r):
            a(calc, x, "conclusion (à regarder)")
        non_tenus = [c["texte"] for c in r["controles"] if not c["ok"]]
        if non_tenus:
            a(calc, ("Contrôles de géométrie non respectés : " if len(non_tenus) > 1 else "Contrôle de géométrie non respecté : ")
              + " ; ".join(non_tenus) + ".", "conclusion (contrôles de géométrie : l'accord, la liste)")
        if r["ecarts"]:
            n_ec, pl = len(r["ecarts"]), "s" if len(r["ecarts"]) > 1 else ""
            fois = collections.Counter(x["code"] for x in r["ecarts"])
            a(calc, f"{n_ec} écart{pl} relevé{pl} dans le classeur (" + ", ".join(c if k == 1 else f"{c} × {k}" for c, k in fois.items())
              + f") : signalé{pl}, non corrigé{pl}.", "conclusion (écarts : le compte se retrouve dans les codes, l'accord)")
        # --- les diagrammes : les textes de chaque scène, le tableau des positions
        vues = {cle: scene for cle, _t, _c, _e, scene, _l in SC.vues(r)}
        for el in r["elements"]:
            a(diag, el["nom"].upper() + " · M ET V", f"titre des diagrammes de {el['cle']}")
            for it in vues["mv_" + el["cle"]].items:
                if it[0] == "texte" and plat(it[3]) not in diag:
                    out.append(f"{nom} · diagrammes de {el['cle']} : le texte « {it[3]} » du dessin de l'écran est absent")
            for p in el["positions"]:
                if "max" in p:
                    mx = p["max"]
                    vals = [mx["RA"]["valeur"], mx["RB"]["valeur"], mx["M_max"]["valeur"], None, mx["V_max"]["valeur"]]
                    fin = FG._colonne(mx["M_max"])
                else:
                    x = p["extremes"]
                    vals, fin = [p["RA"], p["RB"], x["M_max"]["valeur"], x["M_min"]["valeur"], x["V_max"]["valeur"]], ""
                ligne = p["cle"] + C.fr(p["portee"], 2) + "".join(FG._n(v, u) if v is not None and abs(v) > 5e-7 else "—"
                                                                    for v, u in zip(vals, ("kN", "kN", "kN·m", "kN·m", "kN"))) + fin
                a(diag, ligne, f"positions de {el['cle']}, ligne {p['cle']}")
            d = el["dimensionnant"]
            a(diag, "classeur" + "".join(FG._n(d[k], u) if d.get(k) is not None else "—" for k, u in (("M_max", "kN·m"), ("M_min", "kN·m"), ("V_max", "kN"))),
              f"positions de {el['cle']}, ce que le classeur retient")
            x = el["extremes"]
            a(diag, "enveloppe" + FG._n(x["M_max"]["valeur"], "kN·m"), f"positions de {el['cle']}, l'enveloppe")
        for i in range(0, len(r["elements"]), 2):
            page = r["elements"][i:i + 2]
            origines = [origine_attendue(el) for el in page]
            a(diag, f"abscisses en m {origines[0]}." if len(set(origines)) == 1 else
              "Abscisses en m : " + " ; ".join(f"{el['nom'].lower()}, {o}" for el, o in zip(page, origines)) + ".",
              f"l'origine des abscisses des diagrammes ({' et '.join(el['cle'] for el in page)})")
        # --- le plan de principe
        for it in vues["plan"].items:
            if it[0] == "texte" and plat(it[3]) not in plan:
                out.append(f"{nom} · plan : le texte « {it[3]} » du dessin de l'écran est absent")
        a(plan, "ni plan de coffrage, ni plan d'armatures", "plan (ce qu'il n'est pas)")
        for p in r["pieces"]:
            a(plan, f"{p['nom']}{p['nombre']}{dimensions_attendues(p)}", f"pièce « {p['cle']} »")
        ctl = NF.X.controle(r)
        for l in ctl["lignes"]:
            a(plan, f"{l['valeur']}{NF.plage(l.get('cellules'))}", f"armature « {l['element']} · {l['poste']} » et ses cellules")
        for x in ctl["manques"]:
            a(plan, x, "ce que le calcul ne fixe pas")
        n_ech = re.search(r"ÉCHELLE1:(\d+)", plan)
        if not n_ech or int(n_ech.group(1)) not in NF.ECHELLES:
            out.append(f"{nom} · plan : pas d'échelle normalisée au cartouche")
    return out


def generer(e, infos=None):
    return NF.generer_note(copy.deepcopy(e), NF.infos_doc(e, infos or INFOS))


# ================================================================== 1 et 2. la note entière, la parité
print("=== 1 et 2. Toute la note en un PDF ; parité avec le moteur de l'écran ===")
NOTES = {}
for nom_etat, e in ETATS.items():
    try:
        octets = generer(e)
    except Exception as err:  # noqa: BLE001
        chk(f"{nom_etat} : la note se fait", False, f"{type(err).__name__} : {err}")
        continue
    NOTES[nom_etat] = octets
    tx = pages_de(octets)
    ec = fautes(e, octets)
    chk(f"{nom_etat} : un PDF de {len(tx)} pages ({NF.dernieres_variantes}) — la garde, puis le calcul, les diagrammes et le plan de chaque "
        "fondation ; tout ce que le moteur calcule y est, à la lettre, avec les cellules du classeur", octets[:5] == b"%PDF-" and not ec, ec[:5])
    chk(f"{nom_etat} : aucun débordement, aucune page tronquée", NF.derniers_avertissements == [], NF.derniers_avertissements)

tailles = subprocess.run(["pdfinfo", "-f", "1", "-l", "9", "-"], input=NOTES["classeur Greenwood"], capture_output=True).stdout.decode()
dims = re.findall(r"Page\s+\d+ size:\s+([\d.]+) x ([\d.]+)", tailles)
chk("la garde est en A4 portrait, les pages de chaque fondation en A4 paysage",
    len(dims) == 6 and float(dims[0][0]) < float(dims[0][1]) and all(float(w) > float(h) for w, h in dims[1:]), dims)
tg = pages_de(NOTES["classeur Greenwood"])
p1 = O.norm(tg[1])
chk("Greenwood, à la lettre : « Lc = 600 cm C34 », « a = 25 cm C35 », « capacité = 75 T C31 », « CAS 1 en service 369 1004 369 0 C48:C51 », "
    "« Hs = 51 kN C60 · Hhs = 129 kN G60 · Ms = 430 kN·m C61 »",
    all(x in p1 for x in ("Lc = 600 cm C34", "a = 25 cm C35", "capacité = 75 T C31", "CAS 1 en service 369 1004 369 0 C48:C51",
                          "Hs = 51 kN C60 · Hhs = 129 kN G60 · Ms = 430 kN·m C61")), p1[600:1500])
chk("Greenwood, à la lettre : « = 1032,4 × 1,15 − 24,5 × 1,152 /2 − 1004 × 1,15 − 0,45 = 468,3 kN·m C94 » (le moment de la poutre support, "
    "refait à la main) et « néc. 2 458 ≤ posé 2 513 mm² 97,8 % · OK (G165) »",
    "= 1032,4 × 1,15 − 24,5 × 1,152 /2 − 1004 × 1,15 − 0,45 = 468,3 kN·m C94" in p1 and "néc. 2 458 ≤ posé 2 513 mm² 97,8 % · OK (G165)" in p1)
tv = " ".join(O.norm(t) for t in pages_de(NOTES["variante 4 pieux"]))
chk("variante 4 pieux : l'écart E16 se lit — le classeur retient 72,6 kN·m, l'enveloppe donne 76,3 kN·m (tableau des positions, dessin, bloc Écarts)",
    "classeur 72,6 −71,2 484,9" in tv and "enveloppe 76,3 −71,2 484,9" in tv and "E16" in tv and "classeur : M max = 72,6" in tv
    and "-71,2" not in tv, [x for x in ("classeur 72,6 −71,2 484,9", "enveloppe 76,3 −71,2 484,9", "E16", "classeur : M max = 72,6") if x not in tv])
tn = pages_de(NOTES["pieu PAS OK"])
chk("pieu PAS OK : « NON CONFORME », la bande « sollicitation 60,2 > capacité 40,0 T … PAS OK (G241) », la conclusion le dit",
    "NON CONFORME" in tn[1] and plat("sollicitation 60,2 > capacité 40,0 T150,5 % · PAS OK (G241)") in plat(" ".join(tn))
    and "La fondation n'est PAS conforme en l'état" in O.norm(" ".join(tn)))
tr_ = pages_de(NOTES["la note riche du relevé (trois fondations)"])
g = O.norm(tr_[0])
chk("la note riche : le sommaire donne la page de chaque fondation (2, puis la suite des pages de la précédente) et son statut",
    all(x in g for x in ("F1", "F2", "F3", "À VÉRIFIER", "NON CONFORME", "CONFORME", "page 2")) and len(tr_) == 14, g[:600])

# ================================================================== 3. les dessins
print("\n=== 3. Les dessins : le peintre, l'échelle du plan ===")


class Toile:
    """Un canevas qui note ce qu'on lui demande (les seules méthodes que le peintre appelle)."""

    def __init__(self):
        self.appels = []

    def __getattr__(self, nom):
        def f(*a, **k):
            self.appels.append((nom, a))
            return self
        return f


class DocEspion:
    def __init__(self):
        self.c, self.textes = Toile(), []

    def t(self, x, y, s, f, size, col, align="left", track=0.0):
        self.textes.append((round(x, 3), round(y, 3), s, f, round(size, 3), align))


sc = SC.Scene()
sc.cadre = (0.0, 0.0, 100.0, 50.0)
sc.ligne(0, 0, 100, 0, "#000000", 1.0)
sc.ligne(0, 50, 100, 50, "#000000", 0.01, "4 2")
sc.texte(50, 25, "milieu", 10, "middle", "#000000", True)
sc.texte(0, 50, "bas", 4, "start", "#000000", False, angle=-90.0)
de = DocEspion()
s_ = NF.peindre(de, sc, 10, 20, 200, 100)
lignes_ = [a for n, a in de.c.appels if n == "line"]
epais = [a[0] for n, a in de.c.appels if n == "setLineWidth"]
chk("le peintre : l'échelle est la plus grande qui tient (2 pt par unité) ; le haut de la scène est en haut de la boîte (y retourné) ; un "
    "trait de 1 unité fait 2 pt, un trait trop fin reste à 0,25 pt ; les tirets sont à l'échelle",
    s_ == 2.0 and lignes_ == [(10.0, 120.0, 210.0, 120.0), (10.0, 20.0, 210.0, 20.0)] and epais == [2.0, 0.25]
    and [a[0] for n, a in de.c.appels if n == "setDash" and a] == [[8.0, 4.0]], (s_, lignes_, epais))
chk("le peintre : un texte est à l'échelle (10 unités → 20 pt), gras en Carlito-Bold, centré sur son ancre ; un texte tourné de −90° (lu de "
    "bas en haut) est posé par une rotation de +90° autour de son ancre",
    de.textes == [(110.0, 70.0, "milieu", "Carlito-Bold", 20.0, "center"), (0, 0, "bas", "Carlito", 8.0, "left")]
    and ("translate", (10.0, 20.0)) in de.c.appels and ("rotate", (90.0,)) in de.c.appels, de.textes)
de2 = DocEspion()
chk("le peintre : une échelle imposée est respectée, la scène reste centrée", NF.peindre(de2, sc, 0, 0, 300, 100, echelle=1.0) == 1.0
    and [a for n, a in de2.c.appels if n == "line"][0] == (100.0, 75.0, 200.0, 75.0))
for nom_ex, attendu in (("GWW", 75), ("VAR4", 50), ("VAR8", 75)):
    calc_ = NF.Calcul(MO.exemple(nom_ex))
    scene = calc_.vues["plan"][1]
    n_, s_pl = NF.echelle_plan(scene, 430.0, 440.0)
    _, _, cw, ch = scene.cadre
    plus_grande = NF.ECHELLES[NF.ECHELLES.index(n_) - 1]
    chk(f"{nom_ex} : le plan (en mm) est au 1:{n_} — il tient dans sa place, et pas au 1:{plus_grande}",
        n_ == attendu and cw * s_pl <= 430 and ch * s_pl <= 440 and math.isclose(s_pl, 72 / 25.4 / n_)
        and (cw * 72 / 25.4 / plus_grande > 430 or ch * 72 / 25.4 / plus_grande > 440), (n_, cw, ch))
chk("les cellules qui se suivent s'écrivent en plage : « G129, G130, G131, G132 » → « G129:G132 » ; pas « C5, G5 » ni « G1, G3 » ; rien → « — »",
    (NF.plage("G129, G130, G131, G132"), NF.plage("C5, G5"), NF.plage("G1, G3"), NF.plage("G7"), NF.plage(None)) == ("G129:G132", "C5, G5", "G1, G3", "G7", "—"))
chk("le nom d'une formule : le symbole est composé, ce qui le qualifie est en texte droit",
    NF.nom_formule("M_max — A.a (CAS 1.a-B/A), x = 1,15 m") == r"M_{max} \t{ — A.a (CAS 1.a-B/A), x = 1,15 m}"
    and NF.nom_formule("R_A (CAS A)") == r"R_{A} \t{ (CAS A)}" and NF.nom_formule("Ω_inf") == "Ω_{inf}")

# ================================================================== 4. la mise en page
print("\n=== 4. La mise en page : les blocs mesurés puis posés ===")
rnd = random.Random(7)
mauvais = []
for _ in range(300):
    n = rnd.randint(1, 40)
    hs = [rnd.uniform(5, 180) for _ in range(n)]
    lies = [rnd.random() < 0.25 for _ in range(n)]
    lies[-1] = False
    h1, h2 = rnd.uniform(380, 480), rnd.uniform(480, 500)
    rep = NF.repartir(hs, lies, h1, h2)
    if rep is None:
        if max(hs) <= h1 - 180:
            mauvais.append(("refus sans raison", n))
        continue
    plat_ = [i for page in rep for col in page for i in col]
    if plat_ != list(range(n)):
        mauvais.append(("ordre", plat_[:6]))
    for k, page in enumerate(rep):
        if len(page) > 2:
            mauvais.append(("trois colonnes", k))
        for col in page:
            if sum(hs[i] for i in col) > (h1 if k == 0 else h2) + 0.5:
                mauvais.append(("colonne trop haute", k))
            if lies[col[-1]] and col[-1] + 1 < n:
                # un bloc lié en bas de colonne : sa chaîne (jusqu'au premier bloc non lié) ne tenait pas dans une colonne
                # entière, ou elle commence la colonne — sinon elle devait partir ensemble
                debut_ = col[-1]
                while debut_ - 1 in col and lies[debut_ - 1]:
                    debut_ -= 1
                fin_ = col[-1]
                while lies[fin_] and fin_ + 1 < n:
                    fin_ += 1
                if debut_ != col[0] and sum(hs[debut_:fin_ + 1]) <= (h1 if k == 0 else h2):
                    mauvais.append(("un bloc lié reste en bas de sa colonne sans le suivant", col[-1]))
chk("repartir, sur 300 tirages : l'ordre est gardé, chaque bloc est posé une fois, deux colonnes par page, aucune colonne ne dépasse sa "
    "hauteur, un bloc « lié » ne reste jamais en bas d'une colonne sans le suivant", not mauvais, mauvais[:4])
chk("repartir : un bloc plus haut qu'une colonne entière → None (la note est refusée, jamais tronquée)",
    NF.repartir([10.0, 600.0, 10.0], [False, False, False], 480.0, 500.0) is None)
calc_g = NF.Calcul(MO.exemple("GWW"))
doc_meta = NF.ndc_data.construire_doc(NF.infos_doc(ETATS["classeur Greenwood"], INFOS))
k_, nom_k, rep_ = NF.mise_en_page(calc_g, doc_meta)


def pages_a(k):
    note = NF.Note(calc_g, doc_meta, k)
    brouillon = NF.Doc(io.BytesIO(), NF.A4L)
    h1, h2 = (top - NF.BAS for top in note.hauts(brouillon))
    us = note.unites()
    r_ = NF.repartir(NF._hauteurs(note, us, (NF.A4L[0] - 2 * NF.M - NF.GAP) / 2.0), [lie for _c, lie, _f in us], h1, h2)
    return None if r_ is None else len(r_)


plus_grandes = [k for k, _n in NF.VARIANTES if k > k_]
chk(f"Greenwood : la taille retenue ({nom_k}, × {k_}) est la plus grande qui garde le nombre de pages de la taille normale "
    f"({pages_a(NF.NORMALE)} pages de calcul) — les tailles au-dessus en demanderaient plus",
    len(rep_) == pages_a(NF.NORMALE) == pages_a(k_) and all((pages_a(k) or 99) > len(rep_) for k in plus_grandes), (k_, [pages_a(k) for k, _ in NF.VARIANTES]))
us_g = NF.Note(calc_g, doc_meta, 1.0).unites()
chk("les blocs du calcul, dans l'ordre : hypothèses, données, charges, puis chaque bloc du classeur (son en-tête lié à ce qui suit), les "
    "contrôles, les écarts, le récapitulatif, les limites, la conclusion",
    [c for c, _l, _f in us_g][:3] == ["hypotheses", "geometrie", "charges"] and [c for c, _l, _f in us_g][-3:] == ["recapitulatif", "limites", "conclusion"]
    and all(lie for c, lie, _f in us_g if c.endswith(".tete")) and [c for c, _l, _f in us_g if c.endswith(".tete")] == [b["cle"] + ".tete" for b in calc_g.res["blocs"]])
rnd = random.Random(20261111)
refus, n_grille = [], 0
while n_grille < 40:
    f = EN.exemple(rnd.choice(("BASE", "GWW", "VAR4", "VAR8")))
    champs = {EN.CADRE4: ("chassis", "poutre/b", "poutre/h", "pieux/D", "charges/P", "charges/H_s", "charges/M_s", "e"),
              EN.DIAGONALES8: ("chassis", "secondaire/b", "secondaire/h", "primaire/b", "primaire/h", "pieux/D", "charges/P", "e_secondaire", "e_primaire"),
              EN.SEMELLES8: ("chassis", "support/b", "support/h", "semelle/b", "semelle/h", "dist_a", "dist_b", "e_support", "e_semelle", "c_appui", "charges/H_s")}[f["type"]]
    for c in champs:
        EN.poser(f, c, round(EN.lire(f, c) * rnd.uniform(0.6, 1.5), 1))
    try:
        f = EN.normaliser(f)
    except ValueError:
        continue
    n_grille += 1
    e_ = {"version": 1, "fondations": [f], "actif": 0}
    try:
        o_ = generer(e_)
        ec = fautes(e_, o_)
        if ec or NF.derniers_avertissements:
            refus.append((f["type"], ec[:2], NF.derniers_avertissements))
    except Exception as err:  # noqa: BLE001
        refus.append((f["type"], type(err).__name__, str(err)[:200]))
chk(f"une grille de {n_grille} fondations tirées au sort autour des quatre classeurs : aucune note refusée, aucun débordement, la parité partout",
    not refus, refus[:3])

# ================================================================== 5. l'identification, le fichier
print("\n=== 5. L'identification de la note du site, le fichier ===")
e3 = ETATS["la note riche du relevé (trois fondations)"]
tx = pages_de(NOTES["la note riche du relevé (trois fondations)"])
g = plat(tx[0])
chk("la garde : PROJET = le chantier, PARTIE = le nom de la note (accents, guillemets, « & », tiret long), DATE, INDICE C",
    all(plat(lab + val) in g for lab, val in (("PROJET", INFOS["chantier"]), ("PARTIE", INFOS["nom"]), ("DATE", "03/04/2026"), ("INDICE", "C"))), tx[0][:400])
chk("chaque page porte « Note de calcul · 03/04/2026 · indice C » et « chantier · nom » ; nulle part un autre indice",
    all("Note de calcul · 03/04/2026 · indice C" in O.norm(t) and f"{INFOS['chantier']} · {INFOS['nom']}" in O.norm(t) or "DIAGRAMMES" in t or "PLAN DE PRINCIPE" in t
        for t in tx[1:]) and all("03/04/2026 · indice C" in O.norm(t) for t in tx[1:])
    and not re.search(r"\bindice\s*:?\s*(?!C\b)[A-Z0-9]\b", O.norm(" ".join(tx)), re.I))
chk("le cartouche du plan : la fondation, « chantier · nom », la date, « indice C », l'échelle",
    all(x in O.norm(tx[-1]) for x in ("F O N D A T I O N F3 " + INFOS["chantier"], "03/04/2026 indice C", "É C H E L L E 1:75")), O.norm(tx[-1])[-500:])
rep = json.loads(R.traiter(json.dumps({"module": "fondation_grue", "action": "telecharger", "etat": e3, "quoi": "note_pdf", "infos": INFOS})))
fichier = rep.get("fichier") or {}
chk("le routeur rend la note : « Note de calcul - <nom de la note>.pdf », application/pdf, les mêmes pages que le générateur",
    rep.get("ok") and fichier.get("type") == "application/pdf" and fichier.get("nom", "").startswith("Note de calcul - ") and fichier["nom"].endswith(".pdf")
    and [O.norm(t) for t in pages_de(base64.b64decode(fichier["base64"]))] == [O.norm(t) for t in tx], (rep.get("erreur"), fichier.get("nom")))
chk("sans chantier : PROJET = le nom de la note, PARTIE = le titre du module ; sans nom de note et une seule fondation : son nom",
    NF.infos_doc(e3, {"nom": "N", "indice": "D"}) == {"nom_projet": "N", "partie": NF.TITRE_MODULE, "indice": "D"}
    and NF.nom_note(ETATS["classeur BASE"], {}) == ETATS["classeur BASE"]["fondations"][0]["nom"] and NF.nom_note(e3, {}) == NF.TITRE_MODULE
    and NF.infos_doc(e3, {})["indice"] == "A")
chk("la note ne change pas l'état qu'on lui donne ; deux tirages donnent le même texte",
    (lambda a: (generer(a), a)[1])(copy.deepcopy(e3)) == e3 and [O.norm(t) for t in pages_de(generer(e3))] == [O.norm(t) for t in tx])

# ================================================================== 6. les refus
print("\n=== 6. Les refus ===")


def refuse(e, mot):
    try:
        NF.generer_note(e, {})
    except ValueError as err:
        return mot in str(err)
    return False


hors = copy.deepcopy(ETATS["classeur BASE"])
hors["fondations"][0]["e_support"] = 300.0
chk("une note sans fondation est refusée (« aucune fondation ») ; une fondation hors domaine arrête la note avec le message du moteur",
    refuse({"version": 1, "fondations": []}, "aucune fondation") and refuse(hors, "La note n'est pas préparée : fondation « F1 »"))
rep = json.loads(R.traiter(json.dumps({"module": "fondation_grue", "action": "telecharger", "etat": hors, "quoi": "note_pdf", "infos": INFOS})))
chk("le routeur ne rend pas de fichier pour une fondation hors domaine", not rep.get("ok") or not rep.get("fichier"), rep.get("erreur"))

# ================================================================== 7. comme dans le navigateur
print("\n=== 7. Comme dans le navigateur ===")
SONDE = r"""
import builtins, json, os, sys
RACINE, STUB, REFUS = {racine!r}, {stub!r}, {refus!r}
sys.path.insert(0, STUB); sys.path.insert(1, RACINE); os.chdir(RACINE)
class Refus:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in REFUS or name == "reportlab.lib._rl_accel":
            raise ImportError("refusé dans le navigateur : " + name)
sys.meta_path.insert(0, Refus())
lus = set(); _open = builtins.open
def espion(f, *a, **k):
    try: lus.add(os.path.abspath(f))
    except Exception: pass
    return _open(f, *a, **k)
builtins.open = espion
def depot():
    mods = {{os.path.abspath(m.__file__) for m in list(sys.modules.values()) if getattr(m, "__file__", None)}}
    return {{os.path.relpath(f, RACINE) for f in mods | lus if f.startswith(RACINE + os.sep) and not f.endswith(".pyc")}}
from api_web import fondation_grue as FG
with _open({etats!r}, encoding="utf-8") as f:
    etats = json.load(f)
for e in etats:
    for i in range(len(e["fondations"])):
        e["actif"] = i
        FG.resultat(e)
ecran = depot()
import base64
sorties = [FG.telecharger(e, "note_pdf", {infos!r})["base64"] for e in etats]
note = depot()
import PIL
print(json.dumps({{"note": sorted(f for f in note - ecran if not f.endswith(".ttf")), "numpy": "numpy" in sys.modules,
                   "polices": sorted(os.path.basename(f) for f in lus if f.endswith(".ttf")), "pil": PIL.__file__,
                   "fonttools": "fontTools" in sys.modules, "rl_accel": any(m.split(".")[-1] == "_rl_accel" for m in sys.modules),
                   "sorties": sorties}}))
"""
e_nav = [ETATS["la note riche du relevé (trois fondations)"], ETATS["classeur BASE"], ETATS["pieu PAS OK"]]
dossier = tempfile.mkdtemp()
try:
    for f in K.NOTE_POLICES:
        shutil.copy(os.path.join(RACINE, "ndc_pdf", "fonts", f), dossier)
    ch = os.path.join(dossier, "etats.json")
    with open(ch, "w", encoding="utf-8") as fh:
        json.dump(e_nav, fh, ensure_ascii=False)
    p = subprocess.run([sys.executable, "-c", SONDE.format(racine=RACINE, stub=O.STUB, refus=K.NOTE_REFUS, etats=ch, infos=INFOS)],
                       capture_output=True, text=True, cwd=RACINE, env={**os.environ, "NDC_FONT_DIR": dossier})
finally:
    shutil.rmtree(dossier, ignore_errors=True)
try:
    s = json.loads(p.stdout.strip().splitlines()[-1])
except Exception:  # noqa: BLE001
    s = {"erreur": p.stderr[-1500:]}
chk("navigateur : sans fontTools ni accélérateurs C, avec le Pillow factice et les seules polices du paquet, SANS numpy, la note se fait",
    "erreur" not in s and s["fonttools"] is False and s["rl_accel"] is False and s["pil"].startswith(O.STUB) and s["numpy"] is False,
    s.get("erreur") or {k: v for k, v in s.items() if k != "sorties"})
if "erreur" not in s:
    sorties = [base64.b64decode(x) for x in s["sorties"]]
    cpy = [[O.norm(t) for t in pages_de(NF.pdf(copy.deepcopy(x), INFOS))] for x in e_nav]
    chk("navigateur : le texte de chaque page est celui de la note faite en CPython", [[O.norm(t) for t in pages_de(o)] for o in sorties] == cpy)
    pol = sorted({x for o in sorties for x in O.polices(o)})
    chk(f"navigateur : les polices de la note sont celles du paquet : {pol}",
        set(pol) <= {f[:-4] for f in K.NOTE_POLICES} | {"Helvetica"} and set(s["polices"]) <= set(K.NOTE_POLICES), (pol, s["polices"]))
    hors_paquets = sorted(f for f in s["note"] if not f.startswith(("ndc_pdf/", "site/note_pdf/")) and f.split("/")[0] not in K.PAQUETS_ENTIERS)
    chk("navigateur : la note ne charge aucun fichier du dépôt hors des paquets du site (le paquet de la note n'a rien à recevoir de plus)",
        not hors_paquets and "api_web/note_fondation_grue.py" in s["note"], hors_paquets)
chk("la construction : le module est dans NOTE_PDF (son bouton s'allume) et a sa note riche (trois fondations, les trois schémas)",
    "fondation_grue" in K.NOTE_PDF and [f["type"] for f in riche()["fondations"]] == [EN.SEMELLES8, EN.CADRE4, EN.DIAGONALES8])

# ================================================================== 8. les mutations
print("\n=== 8. Les mutations : chacune vue rouge ===")
E_MUT = ETATS["la note riche du relevé (trois fondations)"]


E_BASE = ETATS["classeur BASE"]              # trois contrôles de géométrie non tenus, l'écart E7 relevé deux fois


def rouge(nom_objet, objet, attribut, remplacement, etats=(E_MUT, E_BASE)) -> bool:
    """Le contrôle de parité (ou la génération) rougit-il quand ``objet.attribut`` est remplacé ?"""
    vrai = getattr(objet, attribut)
    setattr(objet, attribut, remplacement(vrai))
    try:
        for e in etats:
            try:
                o = generer(e)
            except Exception:  # noqa: BLE001 — une note qui ne se fait plus est vue
                return True
            if fautes(e, o) or NF.derniers_avertissements:
                return True
        return False
    finally:
        setattr(objet, attribut, vrai)


def sans_cellule(vrai):
    def f(self, d, fr, fx, indent=4):
        return vrai(self, d, fr, dict(fx, cellule=None), indent)
    return f


def verdict_du_premier(vrai):
    def f(self, v, d, fr):
        b = next(x for x in self.C.res["blocs"] if x["cle"] == v["bloc"])
        return vrai(self, dict(v, taux=b["verifications"][0]["taux"]), d, fr)
    return f


def calcul_sur_un_autre(vrai):
    def f(self, fondation):
        vrai(self, dict(fondation, chassis=fondation["chassis"] + 10.0))
    return f


def taux_fausse(vrai):
    def f(self, fondation):
        vrai(self, fondation)
        self.res["taux_max"] *= 0.9
    return f


def premiere_verification(vrai):
    def f(self):
        v = self.res["verifications"][-1]
        return v, next(b for b in self.res["blocs"] if b["cle"] == v["bloc"])
    return f


def poser_oublieux(vrai):
    def f(d, calc, doc_meta, k, repartition):
        return [] if calc.nom() == "F2" else vrai(d, calc, doc_meta, k, repartition)
    return f


def repartir_sans_mesure(vrai):
    return lambda hauteurs, lies, h1, h2, n_cols=2: [[list(range(len(hauteurs)))]]


def positions_du_premier(vrai):
    def f(self, el):
        cols, lignes, note = vrai(self, el)
        return cols, [lignes[0]] * len(lignes), note
    return f


def conclusion_sans_accord(vrai):
    return lambda self: vrai(self).replace("Contrôles de géométrie non respectés", "Contrôle de géométrie non respecté")


MUTATIONS = (
    ("une fondation oubliée (ses pages ne sont pas posées)", NF, "poser", poser_oublieux),
    ("la cellule du classeur n'est plus écrite après une formule", NF.Note, "formule", sans_cellule),
    ("une donnée saisie sans sa cellule", NF, "cellule_de", lambda vrai: (lambda f, chemin: "")),
    ("le verdict d'une vérification porte le taux de la première de son bloc", NF.Note, "verification", verdict_du_premier),
    ("la note calcule une autre fondation que celle de l'écran (châssis + 10 cm)", NF.Calcul, "__init__", calcul_sur_un_autre),
    ("la comparaison imprimée n'est plus celle de l'écran", NF, "comparaison", lambda vrai: (lambda v: "néc. 0 ≤ posé 0")),
    ("η max imprimé n'est pas celui du moteur (taux de tête × 0,9)", NF.Calcul, "__init__", taux_fausse),
    ("ce qui gouverne est mal nommé (la première vérification)", NF.Calcul, "gouvernant", premiere_verification),
    ("les écarts relevés ne sont plus imprimés", NF.Note, "ecart", lambda vrai: (lambda self, e, d, fr: None)),
    ("les contrôles de géométrie ne sont plus imprimés", NF.Note, "controles", lambda vrai: (lambda self, d, fr: None)),
    ("le récapitulatif n'est plus imprimé", NF.Note, "recapitulatif", lambda vrai: (lambda self, d, fr: None)),
    ("ce que la note ne couvre pas n'est plus dit", NF.Note, "limites", lambda vrai: (lambda self, d, fr: None)),
    ("le dessin n'est plus peint (plan et diagrammes vides)", NF, "peindre", lambda vrai: (lambda d, scene, x, y, w, h, echelle=None, haut=False: 1.0)),
    ("le tableau des positions répète sa première ligne", NF.Note, "lignes_positions", positions_du_premier),
    ("les blocs sont posés sans être mesurés (tout dans une colonne)", NF, "repartir", repartir_sans_mesure),
    ("l'échelle du plan n'est plus normalisée", NF, "echelle_plan", lambda vrai: (lambda scene, w, h: (77, vrai(scene, w, h)[1]))),
    ("les cellules des armatures ne sont plus écrites", NF, "plage", lambda vrai: (lambda cellules: "—")),
    ("le nom des pièces du plan est perdu", NF, "dimensions", lambda vrai: (lambda p: "")),
    ("la conclusion annonce « 5 écarts » et n'en nomme que quatre (un code relevé deux fois n'est pas compté)", NF, "codes_ecarts",
     lambda vrai: (lambda ecarts: ", ".join(dict.fromkeys(x["code"] for x in ecarts)))),
    ("la conclusion écrit « Contrôle de géométrie non respecté » au singulier devant trois contrôles", NF.Note, "texte_conclusion", conclusion_sans_accord),
    ("la page des diagrammes dit « depuis l'appui A » pour toutes ses pièces (la semelle est comptée de son milieu)", NF.SC, "legende_mv",
     lambda vrai: (lambda elements: vrai(elements[:1]))),
)
survivants = [nom for nom, objet, attribut, rempl in MUTATIONS if not rouge(nom, objet, attribut, rempl)]
chk(f"{len(MUTATIONS)} mutations (une fondation oubliée, une cellule, un verdict, une autre fondation calculée, η max, ce qui gouverne, les écarts, les "
    "contrôles, le récapitulatif, les limites, les dessins, les positions, la mise en page, l'échelle, les armatures, les pièces, la conclusion, "
    "l'origine des abscisses) : TOUTES vues rouges",
    not survivants, survivants)
chk("le témoin (une fonction remplacée par elle-même) n'est vu rouge par aucun contrôle", not rouge("témoin", NF, "plage", lambda vrai: vrai))
indice = NF.infos_doc
NF.infos_doc = lambda e, infos: dict(indice(e, infos), indice="0")
try:
    perdu = pages_de(NF.pdf(copy.deepcopy(E_MUT), INFOS))
finally:
    NF.infos_doc = indice
chk("mutation : l'indice de la note perdu (« 0 » imprimé) est vu par le contrôle de l'identification",
    not all("03/04/2026 · indice C" in O.norm(t) for t in perdu[1:]))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:500])
sys.exit(1 if KO else 0)

# -*- coding: utf-8 -*-
"""Le module « Poutres béton armé » du site : l'état d'une note (une ou plusieurs poutres, chacune avec ses sections),
les DONNÉES en rubriques (colonne de gauche), le DESSIN cliquable de chaque section (au centre), les RÉSULTATS compacts
et leurs fenêtres de détail (colonne de droite), le sélecteur d'armatures — rendus en fragments HTML.

Le calcul est celui du moteur ``beton/poutre`` (extrait de l'écran Streamlit modules/poutre.py, dont il reproduit les
résultats — tests/test_poutre_moteur.py), complété des DISPOSITIONS CONSTRUCTIVES du ferraillage détaillé (lits
positionnés, cadres multiples, armatures latérales : ``beton/poutre/dispositions.py``). Ce fichier MET EN FORME : il ne
calcule rien ; il CHOISIT parmi ce que le moteur a nommé (la proposition d'armatures retient la plus petite section qui
tient ET se pose) et COMPOSE les phrases.

Les pièces du dessin et les lignes des données portent la MÊME clé (``s<i>.lit-inf-<j>``, ``s<i>.cadre-<k>``,
``s<i>.lat``, ``s<i>.efforts``, ``s<i>.beton``) : ``data-sel`` sur le dessin, ``data-cible`` dans les données. Un clic
sur l'une montre l'autre (js/compact.js).
"""
import copy
from typing import Dict, List, Optional, Sequence

from beton.poutre import dispositions as DI
from beton.poutre import entree as EN
from beton.poutre import geometrie as G
from beton.poutre import moteur as MO

from . import commun as C
from . import coupe_ba as CB

MODULE = "poutre"
PAGE = "poutre"


# ------------------------------------------------------------------ l'état de la note : des poutres
def nouvel_etat() -> Dict:
    p = EN.nouvelle_poutre()
    p["nom"] = "V1"
    return {"version": 1, "poutres": [p], "actif": 0}


def normaliser(etat: Dict) -> Dict:
    """L'état reçu de l'écran relu par le moteur ; une poutre illisible (fichier abîmé) lève une erreur claire. Un
    fichier de l'écran Streamlit ouvert sur le site arrive tel quel ({"streamlit": {"beams", "values"}}, js/notes.js) :
    il est traduit ici, une fois (api_web/poutre_streamlit.py)."""
    if isinstance(etat, dict) and isinstance(etat.get("streamlit"), dict):
        from . import poutre_streamlit as PS
        etat = PS.depuis_streamlit(etat["streamlit"])
    poutres = etat.get("poutres") if isinstance(etat, dict) else None
    if not poutres:
        return nouvel_etat()
    out = {"version": 1, "poutres": [_noms_uniques(EN.normaliser(p)) for p in poutres]}
    out["actif"] = min(max(0, int(etat.get("actif", 0) or 0)), len(out["poutres"]) - 1)
    return out


def _noms_uniques(p: Dict) -> Dict:
    """Le site désigne une section par son NOM (le sélecteur d'armatures, la ligne gouvernante) : deux sections du
    même nom, ou une sans nom — possible dans un fichier .stab, pas à l'écran — prennent la première lettre libre."""
    pris: List[str] = []
    for s in p["sections"]:
        if not s["nom"].strip() or s["nom"] in pris:
            s["nom"] = EN.nom_section_suivant({"sections": [{"nom": n} for n in pris + [x["nom"] for x in p["sections"]]]})
        pris.append(s["nom"])
    return p


def _poutre(etat: Dict) -> Dict:
    return etat["poutres"][etat["actif"]]


def gerer_elements(etat: Dict, action: str, index: Optional[int] = None) -> None:
    """Les poutres de la note : en ouvrir une, en ajouter, copier l'ouverte, en supprimer une (jamais la dernière)."""
    ps = etat["poutres"]
    if action == "activer":
        etat["actif"] = min(max(0, int(index or 0)), len(ps) - 1)
    elif action == "ajouter":
        p = EN.nouvelle_poutre()
        p["nom"] = f"V{len(ps) + 1}"
        ps.append(p)
        etat["actif"] = len(ps) - 1
    elif action == "dupliquer":
        p = copy.deepcopy(ps[etat["actif"]])
        p["nom"] = f"{p.get('nom') or 'Poutre'} (copie)"
        ps.append(p)
        etat["actif"] = len(ps) - 1
    elif action == "supprimer" and len(ps) > 1:
        i = min(max(0, int(index or 0)), len(ps) - 1)
        del ps[i]
        if etat["actif"] > i or etat["actif"] >= len(ps):
            etat["actif"] = max(0, etat["actif"] - 1)


# ------------------------------------------------------------------ la saisie
ENTIERS = {"n", "d", "fyk", "de", "a", "peau_d"}
# les champs « vide = automatique » (commun.num(..., auto=…)) : un champ vidé rend la valeur au calcul du module
AUTO = {"pos", "sh", "de", "a", "yG_inf", "yG_sup"}
TEXTES = {"nom", "beton", "type", "rep"}


def modifier(etat: Dict, chemin: Sequence, valeur) -> Optional[str]:
    """Écrit une saisie dans la poutre ouverte. ``chemin`` : les clés (« sections », 0, « M_inf ») ; les boutons
    d'ajout, de copie et de retrait passent par le même chemin, terminé par « + » (ajouter ; pour les cadres, la valeur
    dit « etrier » ou « epingle »), « dup » (copier la section) ou « - » (retirer). Le moteur relit la poutre : une valeur
    hors de son domaine est refusée avec un message, et la poutre reste intacte."""
    p = _poutre(etat)
    avant = copy.deepcopy(p)
    chemin = [int(c) if isinstance(c, str) and c.lstrip("-").isdigit() and c != "-" else c for c in chemin]
    try:
        if chemin[-1] == "+":
            liste = _aller(p, chemin[:-1])
            if chemin[:-1] == ["sections"]:
                liste.append(EN.nouvelle_section(EN.nom_section_suivant(p)))
            elif chemin[-2] in ("inf", "sup"):
                if len(liste) >= EN.MAX_LITS:
                    return f"Au plus {EN.MAX_LITS} lits par face"
                # le lit ajouté reprend les barres du précédent : on ajoute « un lit de plus », pas un défaut
                liste.append({"n": liste[-1]["n"], "d": liste[-1]["d"]} if liste else dict(EN.LIT_DEFAUT))
            elif chemin[-2] == "etriers":
                s = _aller(p, chemin[:-2])
                liste.append(_nouvelle_ligne(s, "epingle" if valeur == "epingle" else "etrier"))
        elif chemin[-1] == "dup":
            liste = _aller(p, chemin[:-2])
            i = int(chemin[-2])
            s = copy.deepcopy(liste[i])
            s["nom"] = EN.nom_section_suivant(p)
            liste.insert(i + 1, s)
        elif chemin[-1] == "-":
            liste = _aller(p, chemin[:-2])
            if len(liste) <= 1:
                return "Il en faut au moins un"
            del liste[int(chemin[-2])]
        elif len(chemin) >= 2 and chemin[-2] == "laterales":
            s = _aller(p, chemin[:-2])
            lat = dict(s.get("laterales") or {})
            k = chemin[-1]
            x = C.lire_auto(valeur) if k == "n" else C.lire_nombre(valeur)
            if x is None and k != "n":
                return f"Nombre illisible : « {valeur} »"
            # le Ø affiché sans Ø imposé est celui des latérales automatiques (Options) : le réécrire ne l'impose pas
            if x is None or (k == "d" and "d" not in lat and int(round(x)) == int(p.get("peau_d") or EN.DEFAUTS_DETAIL["peau_d"])):
                lat.pop(k, None)
            else:
                lat[k] = int(round(x))
            if lat:
                s["laterales"] = lat
            else:
                s.pop("laterales", None)
        else:
            d = _aller(p, chemin[:-1])
            k = chemin[-1]
            if k in TEXTES:
                d[k] = str(valeur)
            elif k in AUTO:
                x = C.lire_auto(valeur)
                if x is None and k in ("yG_inf", "yG_sup"):
                    d[k] = None                     # la clé du module : null = calculé
                elif x is None:
                    d.pop(k, None)                  # une clé du détail : absente = automatique
                else:
                    d[k] = int(round(x)) if k in ENTIERS else x
            else:
                x = C.lire_nombre(valeur)
                if x is None:
                    return f"Nombre illisible : « {valeur} »"
                d[k] = int(round(x)) if k in ENTIERS else x
        etat["poutres"][etat["actif"]] = _noms_uniques(EN.normaliser(p))
    except (ValueError, KeyError, IndexError, TypeError) as e:
        etat["poutres"][etat["actif"]] = avant
        return f"Saisie refusée : {e}"
    return None


def _nouvelle_ligne(s: Dict, type_: str) -> Dict:
    """Une ligne d'effort tranchant de plus, placée là où on l'attend : une épingle sur la barre du milieu ; un 2ᵉ cadre
    à l'INTÉRIEUR du premier (il entoure les barres 2 à n − 1 du lit 1 inférieur, dès 4 barres) ; les suivants côte à
    côte (moitié gauche, puis moitié droite). Ø : celui de la première ligne. Les barres entourées se changent ensuite."""
    d = s["etriers"][0]["d"] if s["etriers"] else EN.ETRIER_DEFAUT["d"]
    n1 = s["inf"][0]["n"]
    if type_ == "epingle":
        return {"type": "epingle", "d": d}
    fermes = sum(1 for e in s["etriers"] if e["type"] == "etrier")
    if fermes == 0 or n1 < 3:
        return {"type": "etrier", "d": d}
    if fermes == 1 and n1 >= 4:
        return {"type": "etrier", "d": d, "de": 2, "a": n1 - 1}
    moitie = (n1 + 1) // 2
    return {"type": "etrier", "d": d, "de": 1, "a": moitie} if fermes % 2 else {"type": "etrier", "d": d, "de": n1 - moitie + 1, "a": n1}


def _aller(d, chemin: Sequence):
    for c in chemin:
        d = d[c]
    return d


def choisir_section(etat: Dict, role: str, nom: str) -> None:
    """Le sélecteur d'armatures : « inf-0 » (face inférieure de la section 0) reçoit la disposition « 3Ø20 » en un lit."""
    face, i = role.split("-")
    n, d = nom.split("Ø")
    p = _poutre(etat)
    p["sections"][int(i)][face] = [{"n": int(n), "d": int(d)}]
    etat["poutres"][etat["actif"]] = EN.normaliser(p)


# ------------------------------------------------------------------ petits morceaux d'écran
def _ch(*c) -> str:
    return C._attr_chemin(list(c))


def _bouton(chemin: Sequence, icone: str, titre: str, valeur: str = "1", classe: str = "moins", texte: str = "") -> str:
    return (f'<button type="button" class="{classe}" data-chemin="{C._attr_chemin(chemin)}" data-valeur="{C.esc(valeur)}" '
            f'title="{C.esc(titre)}">{C.icon(icone)}{texte}</button>')


def _cm(x: Optional[float]) -> str:
    return C.fr(x, 1) if x is not None else "—"


def _mm(x: Optional[float]) -> str:
    return C.fr(x, 0) if x is not None else "—"


def _items(dispo: Dict, face, lit, cle) -> List[Dict]:
    return [it for it in dispo["items"] if it["face"] == face and it["lit"] == lit and it["cle"] == cle]


def _etat_txt(e: str) -> str:
    return {DI.OK: "OK", DI.WARN: "À vérifier", DI.NOK: "Non conforme"}[e]


def _niv_dispo(e: str) -> str:
    """La couleur d'un état de disposition : « v » (ambre, à vérifier) n'est pas le vert foncé des taux de 95 à 100 %."""
    return {DI.OK: "p", DI.WARN: "v", DI.NOK: "r"}[e]


# ------------------------------------------------------------------ les données (colonne de gauche)
REPS = (("uniforme", "Répartie"), ("aligne", "Sur L1"), ("entraxe", "Entraxe"))


def _lit(i: int, face: str, j: int, lit: Dict, n_lits: int, rs: Dict) -> str:
    """Un lit : la ligne courte (L1 · n · Ø · aire), puis son détail (répartition, position, espacements) — ouvert
    quand le lit est choisi, ici ou sur le dessin."""
    ch = ["sections", i, face, j]
    geo = rs["geometrie"][face]["lits"][j]
    lits = rs["geometrie"][face]["lits"]
    dispo = rs["dispositions"]
    cible = f"s{i}.lit-{face}-{j}"
    retirer = _bouton(ch + ["-"], "x", "Retirer ce lit") if n_lits > 1 else '<span class="moins"></span>'
    tete = (f'<div class="et-l1"><button type="button" class="et-tag" data-deplier="1" title="Détail du lit">L{j + 1}</button>'
            f'{C.num(ch + ["n"], lit["n"], "", entier=True, mini=1)}'
            f'{C.choix(ch + ["d"], lit["d"], [(x, f"Ø{x}") for x in EN.DIAMETRES])}'
            f'<span class="et-as">{_mm(geo["As"])}{C.NBSP}mm²</span>'
            f'<button type="button" class="et-dep" data-deplier="1" title="Répartition et position">{C.icon("chevron-down")}</button>{retirer}</div>')
    reps = [r for r in REPS if j > 0 or r[0] != "aligne"]
    rep = lit.get("rep", "uniforme")
    det = C.ligne("Répartition", C.segments(ch + ["rep"], rep, reps), full=True)
    if rep == "entraxe":
        xs = geo["x"]
        pas = (xs[1] - xs[0]) if len(xs) > 1 else None
        det += C.ligne("entraxe horizontal", C.num(ch + ["sh"], lit.get("sh"), "cm", auto=f"auto · {_cm(pas)}" if pas else "auto"),
                       "s<sub>h</sub>", "Axe à axe des barres du lit, groupe centré dans la largeur.")
    if j == 0:
        det += C.ligne("axe / parement", C.num(ch + ["pos"], lit.get("pos"), "cm", auto=f"auto · {_cm(geo['distance'])}"),
                       "e<sub>1</sub>", "Distance du parement à l'axe du lit 1 ; vide : enrobage + Ø étrier + Ø/2 arrondis + jeu (méthode du bureau).")
    else:
        auto = geo["distance"] - lits[j - 1]["distance"]
        det += C.ligne(f"entraxe / L{j}", C.num(ch + ["pos"], lit.get("pos"), "cm", auto=f"auto · {_cm(auto)}"),
                       "e<sub>v</sub>", "Axe à axe depuis le lit précédent ; vide : demi-Ø arrondis + jeu entre lits (méthode du bureau).")
    infos = [f"axe à {_cm(geo['distance'])}{C.NBSP}cm du parement"]
    for it in _items(dispo, face, j, "lit") + _items(dispo, face, j, "entre_lits"):
        quoi = "entre barres" if it["cle"] == "lit" else f"avec L{j}"
        infos.append(f'<span class="et-e {_niv_dispo(it["etat"])}">libre {quoi} {_mm(it["libre"])}{C.NBSP}mm '
                     f'(min {_mm(it["mini"])})</span>')
    for it in _items(dispo, face, j, "cadre"):
        infos.append(f'<span class="et-e r">barres hors de l\'étrier</span>')
    det += f'<div class="hint et-info">{" · ".join(infos)}</div>'
    return f'<div class="et-piece" data-cible="{cible}">{tete}<div class="et-det">{det}</div></div>'


def _face(i: int, face: str, s: Dict, rs: Dict) -> str:
    nom = "Inférieures" if face == "inf" else "Supérieures"
    geo = rs["geometrie"][face]
    corps = "".join(_lit(i, face, j, lit, len(s[face]), rs) for j, lit in enumerate(s[face]))
    ajouter = (f'<button type="button" class="addrow" data-chemin="{_ch("sections", i, face, "+")}" data-valeur="1">'
               f'{C.icon("plus")}Ajouter un lit</button>') if len(s[face]) < EN.MAX_LITS else ""
    yg = C.ligne("centre de gravité imposé", C.num(["sections", i, f"yG_{face}"], s[f"yG_{face}"], "cm",
                                                   auto=f"auto · {_cm(geo['yG_calcule'])}"),
                 "y<sub>G</sub>", "Distance du parement au centre de gravité des lits de la face ; vide : calculé (Σ A·e / Σ A).")
    prop = (f'<button type="button" class="et-prop" data-pk="{face}-{i}" title="Les dispositions d\'un lit qui tiennent, triées par section">'
            f'{C.icon("lightbulb")}Proposer</button>')
    return (f'<div class="et-grp"><div class="grp">{nom}<span class="et-grp-s">{C.esc(geo["libelle"])} · {_mm(geo["As"])}{C.NBSP}mm²</span>{prop}</div>'
            f'{corps}{ajouter}<div class="et-avance">{yg}</div></div>')


def _cadres(i: int, s: Dict, rs: Dict) -> str:
    geo = rs["geometrie"]["etriers"]
    n1 = s["inf"][0]["n"]
    out = ""
    for k, (e, cd) in enumerate(zip(s["etriers"], geo["cadres"])):
        ch = ["sections", i, "etriers", k]
        retirer = _bouton(ch + ["-"], "x", "Retirer") if len(s["etriers"]) > 1 else '<span class="moins"></span>'
        tete = (f'<div class="et-l1"><button type="button" class="et-tag c" data-deplier="1" title="Détail">C{k + 1}</button>'
                f'{C.choix(ch + ["type"], e["type"], [("etrier", "Étrier"), ("epingle", "Épingle")])}'
                f'{C.choix(ch + ["d"], e["d"], [(x, f"Ø{x}") for x in EN.DIAMETRES_ETRIERS])}'
                f'<span class="et-as">{cd["brins"]}{C.NBSP}brin{"s" if cd["brins"] > 1 else ""}</span>'
                f'<button type="button" class="et-dep" data-deplier="1" title="Barres entourées">{C.icon("chevron-down")}</button>{retirer}</div>')
        if e["type"] == "epingle":
            det = C.ligne("sur la barre n°", C.num(ch + ["de"], e.get("de"), f"/ {n1}", entier=True, auto=f"auto · {cd['de']}"), "",
                          "La barre du lit 1 inférieur que l'épingle longe ; vide : la barre du milieu.")
        else:
            det = (C.ligne("entoure de la barre", C.num(ch + ["de"], e.get("de"), f"/ {n1}", entier=True, auto=f"auto · {cd['de']}"), "",
                           "Les barres du lit 1 inférieur que le cadre entoure ; vides : toute la largeur.")
                   + C.ligne("à la barre", C.num(ch + ["a"], e.get("a"), f"/ {n1}", entier=True, auto=f"auto · {cd['a']}"), ""))
        largeur = (cd["x1"] - cd["x0"]) if e["type"] != "epingle" else None
        det += (f'<div class="hint et-info">{"cadre sur toute la largeur" if cd["complet"] else ("épingle" if e["type"] == "epingle" else "cadre intérieur")}'
                f'{f" · emprise {_cm(largeur)}{C.NBSP}cm" if largeur else ""} · {cd["brins"]} brin{"s" if cd["brins"] > 1 else ""} vertica{"ux" if cd["brins"] > 1 else "l"}</div>')
        out += f'<div class="et-piece" data-cible="s{i}.cadre-{k}">{tete}<div class="et-det">{det}</div></div>'
    out += (f'<div class="et-adds"><button type="button" class="addrow" data-chemin="{_ch("sections", i, "etriers", "+")}" data-valeur="etrier">'
            f'{C.icon("plus")}Cadre</button><button type="button" class="addrow" data-chemin="{_ch("sections", i, "etriers", "+")}" '
            f'data-valeur="epingle">{C.icon("plus")}Épingle</button></div>')
    out += C.ligne("pas des étriers", C.num(["sections", i, "pas"], s["pas"], "cm"), "e", "Espacement LONGITUDINAL des cadres le long de la poutre.")
    return (f'<div class="et-grp"><div class="grp">Effort tranchant<span class="et-grp-s">{geo["brins"]}{C.NBSP}brins · '
            f'A<sub>st</sub> = {_mm(geo["Ast"])}{C.NBSP}mm²</span></div>{out}</div>')


def _laterales(i: int, s: Dict, rs: Dict) -> str:
    lat = rs["geometrie"]["laterales"]
    imp = s.get("laterales") or {}
    ch = ["sections", i, "laterales"]
    auto = f"auto · {lat['n']}"
    tete = (f'<div class="et-l1"><span class="et-tag l">LAT</span>'
            f'{C.num(ch + ["n"], imp.get("n"), "/ face", entier=True, auto=auto)}'
            f'{C.choix(ch + ["d"], int(lat["d"]), [(x, f"Ø{x}") for x in EN.DIAMETRES_LATERAUX])}</div>')
    regle = (f"automatique : écart entre lits 1 de {_cm(lat['ecart'])}{C.NBSP}cm"
             + (f" &gt; s<sub>max</sub> = {_cm(lat['s_max'])}{C.NBSP}cm" if lat["n"] else f" ≤ s<sub>max</sub> = {_cm(lat['s_max'])}{C.NBSP}cm : aucune")
             if lat["auto"] else "imposées")
    for it in [x for x in rs["dispositions"]["items"] if x["cle"] == "laterales" and x["etat"] == DI.NOK]:
        regle += ' · <span class="et-e r">touchent une barre</span>'
    return (f'<div class="et-grp"><div class="grp">Latérales</div><div class="et-piece" data-cible="s{i}.lat">{tete}'
            f'<div class="hint et-info">{regle} · dessin et note, hors calcul</div></div></div>')


def formulaire(etat: Dict, res: Optional[Dict] = None) -> str:
    """Les quatre rubriques de la colonne de gauche : 1 Poutre, 2 Sections et sollicitations, 3 Armatures (de la
    section courante), 4 Options. L'établi compact n'en ouvre qu'une à la fois. ``res`` : le calcul de la poutre ouverte
    (``MO.verifier``), refait s'il manque."""
    p = _poutre(etat)
    res = res or MO.verifier(p)
    L = C.ligne
    s1 = (f'<div class="et-piece" data-cible="beton">'
          + L("Nom", C.texte(["nom"], p.get("nom"), "ex. V1"), full=True)
          + L("Béton", C.choix(["beton"], p.get("beton"), [(k, k) for k in EN.classes_beton()]), full=True)
          + L("Acier", C.segments(["fyk"], p.get("fyk"), [(n, f"B{n}") for n in EN.NUANCES]), full=True)
          + L("largeur", C.num(["b"], p.get("b"), "cm"), "b")
          + L("hauteur", C.num(["h"], p.get("h"), "cm"), "h")
          + L("enrobage", C.num(["enrobage"], p.get("enrobage"), "cm"), "c", "Distance du parement au nu de l'étrier.")
          + "</div>")
    cartes = ""
    for i, s in enumerate(p["sections"]):
        rs = res["sections"][i]
        niv, txt = _etat_section(rs)
        pct = f'<span class="pct {niv}">{txt}</span>'
        retirer = _bouton(["sections", i, "-"], "x", "Retirer la section") if len(p["sections"]) > 1 else ""
        copier = _bouton(["sections", i, "dup"], "copy", "Copier la section")
        cartes += (f'<div class="secard et-piece" data-cible="s{i}.efforts" data-section="{i}"><div class="sh2"><span class="indice">{C.esc(s["nom"])}</span>'
                   f'{C.texte(["sections", i, "nom"], s["nom"], "nom")}{pct}{copier}{retirer}</div>'
                   f'<div class="trio">'
                   f'<label><span>M<sub>inf</sub></span>{C.num(["sections", i, "M_inf"], s["M_inf"], "kN·m")}</label>'
                   f'<label><span>M<sub>sup</sub></span>{C.num(["sections", i, "M_sup"], s["M_sup"], "kN·m")}</label>'
                   f'<label><span>V</span>{C.num(["sections", i, "V"], s["V"], "kN")}</label></div></div>')
    s2 = (f'<div class="secards">{cartes}</div>'
          f'<button type="button" class="addrow" data-chemin="sections/+" data-valeur="1">{C.icon("plus")}Ajouter une section</button>'
          + C.aide("M<sub>inf</sub> tend le bas (travée), M<sub>sup</sub> tend le haut (appui) ; valeurs de l'état limite de service."))
    onglets = "".join(f'<button type="button" data-section-courante="{i}" class="{"on" if i == 0 else ""}">{C.esc(s["nom"])}</button>'
                      for i, s in enumerate(p["sections"]))
    blocs = ""
    for i, s in enumerate(p["sections"]):
        rs = res["sections"][i]
        blocs += (f'<div class="et-arm{" on" if i == 0 else ""}" data-section="{i}">'
                  + _face(i, "inf", s, rs) + _face(i, "sup", s, rs) + _laterales(i, s, rs) + _cadres(i, s, rs) + "</div>")
    s3 = (f'<div class="seg et-secsel" role="tablist" aria-label="Section">{onglets}</div>' if len(p["sections"]) > 1 else "") + blocs
    resume3 = " · ".join(f'{C.esc(s["nom"])} : {C.esc(_detail_lits(s["inf"]))} / {C.esc(_detail_lits(s["sup"]))}' for s in p["sections"][:2])
    if len(p["sections"]) > 2:
        resume3 += " …"
    s4 = _options(p)
    return (C.section(1, f"Poutre {C.esc(p.get('nom') or '')}", f"{C.nb(p['b'])} × {C.nb(p['h'])} · {p['beton']} · B{p['fyk']}", s1, True, "pt")
            + C.section(2, "Sections et sollicitations", f"{len(p['sections'])} section{'s' if len(p['sections']) > 1 else ''}", s2, False, "so")
            + C.section(3, "Armatures", resume3, s3, False, "arm")
            + C.section(4, "Options", f"méthode du bureau · jeux {C.nb(p['jeu_premier_lit'])} / {C.nb(p['jeu_entre_lits'])} cm", s4, False, "op",
                        secondaire=True))


def _options(p: Dict) -> str:
    L = C.ligne
    dg = float(p.get("dg") or EN.DEFAUTS_DETAIL["dg"])
    # ce que le jeu entre lits donne pour deux lits Ø16 (le cas courant), lu sur les règles du moteur
    libre16 = (2 * G.arrondi_demi_cm(16 / 20.0) + float(p["jeu_entre_lits"])) * 10.0 - 16.0
    smin16 = DI.s_min(16, 16, dg)
    alerte = (f"Avec ce jeu, deux lits Ø16 gardent <b>{_mm(libre16)}{C.NBSP}mm</b> libres, sous l'espacement minimal de l'EC2 "
              f"8.2(2) ({_mm(smin16)}{C.NBSP}mm avec d<sub>g</sub> = {C.nb(dg)}{C.NBSP}mm) : les dispositions le signalent "
              f"« À vérifier ». Imposer l'entraxe vertical des lits, ou porter ce jeu à "
              f"{C.nb((smin16 + 16.0) / 10.0 - 2 * G.arrondi_demi_cm(0.8))}{C.NBSP}cm.") if libre16 < smin16 else ""
    return (f'<div class="row full et-meth"><label><span>Méthode</span></label><b>Méthode du bureau</b><span class="hint">'
            f'{C.esc(MO.REFERENCE)} · ELS, z = 0,9·d</span></div>'
            + L("coefficient acier", C.num(["gamma_s"], p.get("gamma_s"), ""), "γ<sub>s</sub>", "f<sub>yd</sub> = f<sub>yk</sub> / γ<sub>s</sub>")
            + L("jeu au premier lit", C.num(["jeu_premier_lit"], p.get("jeu_premier_lit"), "cm"), "", "Ajouté sous le lit 1, après l'enrobage et l'étrier.")
            + L("jeu entre lits", C.num(["jeu_entre_lits"], p.get("jeu_entre_lits"), "cm"), "", "Ajouté entre les demi-diamètres arrondis de deux lits.")
            + (C.aide(alerte) if alerte else "")
            + '<div class="grp">Dispositions constructives</div>'
            + L("plus gros granulat", C.num(["dg"], dg, "mm"), "d<sub>g</sub>", "Espacement libre minimal EC2 8.2(2) : max(Ø ; d_g + 5 mm ; 20 mm).")
            + L("Ø des latérales (auto)", C.choix(["peau_d"], int(p.get("peau_d") or EN.DEFAUTS_DETAIL["peau_d"]),
                                                    [(x, f"Ø{x}") for x in EN.DIAMETRES_LATERAUX]), "")
            + L("écart maximal des latérales", C.num(["peau_smax"], p.get("peau_smax") or EN.DEFAUTS_DETAIL["peau_smax"], "cm"),
                "s<sub>max</sub>", "Au-delà de cet écart entre les lits 1, des barres latérales sont ajoutées (règle de la note PDF)."))


def _detail_lits(lits: List[Dict]) -> str:
    return " + ".join(f"{lit['n']}Ø{lit['d']}" for lit in lits)


# ------------------------------------------------------------------ le résultat
MOTIFS = {"hauteur_insuffisante": "hauteur insuffisante", "acier_insuffisant": "acier insuffisant",
          "acier_excessif": "acier excessif (A<sub>s</sub> > A<sub>s,max</sub>)", "d_utile_negatif": "hauteur utile négative",
          "barres_inclinees": "barres inclinées nécessaires", "cisaillement_excessif": "cisaillement excessif",
          "pas_trop_grand": "pas trop grand"}
# les noms courts des lignes du résultat (les titres complets du module restent dans le détail du calcul)
COURTS = {"hauteur": "Hauteur", "inf": "Armatures inférieures", "sup": "Armatures supérieures", "tau": "Effort tranchant", "pas": "Étriers"}
BESOINS = {"pas_d_etriers": "pas besoin d'étriers", "etriers": "besoin d'étriers", "barres_inclinees_et_etriers": "barres inclinées et étriers",
           "inacceptable": "inacceptable"}
CRITERES = {"hauteur": "h<sub>min</sub> = h<sub>u,min</sub> + y<sub>G</sub> ≤ h",
            "inf": "A<sub>s,néc</sub> = max(A<sub>s,req</sub> ; A<sub>s,min</sub>) ≤ A<sub>s,posé</sub> ≤ A<sub>s,max</sub>",
            "sup": "A<sub>s,néc</sub> = max(A<sub>s,req</sub> ; A<sub>s,min</sub>) ≤ A<sub>s,posé</sub> ≤ A<sub>s,max</sub>",
            "tau": "τ = V / (0,75·b·h) ≤ τ<sub>adm</sub> (bandes I, II, IV du béton)",
            "pas": "e ≤ s<sub>lim</sub> = min(s<sub>th</sub> ; s<sub>max</sub>)"}


def _niv(v: Dict) -> str:
    """La couleur d'une vérification : le taux du moteur, et rouge dès que le module la déclare non conforme
    (acier excessif, hauteur utile négative…) même sous 100 %."""
    return "r" if v.get("etat") == MO.NOK else C.niveau(v["taux"])


def _etat_section(rs: Dict):
    """(niveau, texte court) d'une section : « Données incomplètes » sans aucun effort (ses vérifications passent toutes
    et ne disent rien), sinon le taux, en rouge si la méthode ou les dispositions la déclarent non conforme."""
    if not rs["sollicitee"]:
        return "c", "à compléter"
    niv = {MO.NOK: "r", MO.WARN: "v"}.get(rs["statut_complet"]) or C.niveau(rs["taux_max"])
    return niv, f"{C.pc(C.pct(rs['taux_max']))}{C.NBSP}%"


def _n(x: Optional[float], unite: str = "") -> str:
    """Un RÉSULTAT écrit avec la précision d'usage : mm² entiers, cm au millimètre, contraintes au centième."""
    if x is None:
        return "—"
    nd = {"mm²": 0, "cm": 1, "N/mm²": 2, "kN": 1, "kN·m": 1}.get(unite)
    if nd is None:
        a = abs(x)
        nd = 0 if a >= 1000 else (1 if a >= 10 else 2)
    return C.fr(x, nd)


def _v(x: Optional[float]) -> str:
    """Une DONNÉE d'une formule (effort, dimension, coefficient) écrite sans la dénaturer : 12,96 reste 12,96,
    0,1709 reste 0,1709, 200 reste 200 ; seules les grandeurs calculées longues (f_yd = 333,33…) sont arrondies."""
    if x is None:
        return "—"
    a = abs(x)
    if a >= 100:
        return C.nb(round(x, 1), 1)
    if a >= 10:
        return C.nb(round(x, 2), 2)
    return C.nb(round(x, 4), 4)


def _formule(f: Dict) -> Dict:
    """Une formule du moteur → les morceaux « symbole = expression = valeurs = résultat » du détail."""
    t = {x["symbole"]: x["valeur"] for x in f["termes"]}
    u = f["unite"]
    r = _n(f["resultat"], u) + (f" {u}" if u else "")
    k = f["cle"]
    if k == "hu_min":
        parts = ["h_u,min", "√(M·10⁶ / (α_b·b·μ)) / 10", f"√({_v(t['M'])} × 10⁶ / ({_v(t['alpha_b'])} × {_v(t['b'])} × {_v(t['mu'])})) / 10"]
    elif k == "h_min":
        parts = ["h_min", "h_u,min + y_G", f"{_v(t['hu_min'])} + {_v(t['yG'])}"]
    elif k == "As_req":
        parts = ["A_s,req", "M·10⁶ / (f_yd·0,9·d)", f"{_v(t['M'])} × 10⁶ / ({_v(t['fyd'])} × 0,9 × {_v(t['d'] * 10)})"]
    elif k == "As_min_ec":
        parts = ["A_s,min,EC", "0,26·f_ctm / f_yk·b·h", f"0,26 × {_v(t['fctm'])} / {_v(t['fyk'])} × {_v(t['b'])} × {_v(t['h'])}"]
    elif k == "As_min_plancher":
        parts = ["A_s,min,pl", "0,0013·b·h", f"0,0013 × {_v(t['b'])} × {_v(t['h'])}"]
    elif k == "As_min_opposee":
        parts = ["A_s,min,opp", "0,25·A_s,req,opp", f"0,25 × {_n(t['As_req_opp'], 'mm²')}"]
    elif k == "As_min":
        parts = ["A_s,min", "max(A_s,min,EC ; A_s,min,pl ; A_s,min,opp)",
                 f"max({_n(t['As_min_ec'], 'mm²')} ; {_n(t['As_min_plancher'], 'mm²')} ; {_n(t['As_min_opposee'], 'mm²')})"]
    elif k == "besoin":
        parts = ["A_s,nécessaire", "max(A_s,req ; A_s,min)", f"max({_n(t['As_req'], 'mm²')} ; {_n(t['As_min'], 'mm²')})"]
    elif k == "As_max":
        parts = ["A_s,max", "0,04·b·h", f"0,04 × {_v(t['b'])} × {_v(t['h'])}"]
    elif k == "tau":
        parts = ["τ", "V·10³ / (0,75·b·h)", f"{_v(t['V'])} × 10³ / (0,75 × {_v(t['b'])} × {_v(t['h'])})"]
    elif k == "tau_lim":
        parts = ["τ_adm", "k·f_ck,cube / 1,05", f"{C.nb(t['k'])} × {_v(t['fck_cube'])} / 1,05"]
    elif k == "s_th":
        parts = ["s_th", "A_st·f_yd·d / (V·10³)", f"{_v(t['Ast'])} × {_v(t['fyd'])} × {_v(t['d'])} / ({_v(t['V'])} × 10³)"]
    elif k == "s_max":
        parts = ["s_max", "min(0,75·d ; 30 cm)", f"min(0,75 × {_v(t['d'])} ; {_v(t['plafond'])})"]
    elif k == "s_lim":
        parts = ["s_lim", "min(s_th ; s_max)", f"min({_v(t['s_th'])} ; {_v(t['s_max'])})"]
    else:                                                        # pragma: no cover — une formule nouvelle du moteur
        parts = [k, " · ".join(f"{s} = {_n(v)}" for s, v in t.items())]
    if f.get("resultat_tronque") is not None and k in ("s_lim",):
        r += f" (retenu : {_n(f['resultat_tronque'])} cm)"
    return {"parts": parts, "res": r, "niveau": ""}


def _piece(v: Dict, s: Dict) -> str:
    geo = s["geometrie"]
    if v["cle"] == "hauteur":
        return f"{C.nb(geo['b'])} × {C.nb(geo['h'])} cm"
    if v["cle"] in ("inf", "sup"):
        return f"{geo[v['cle']]['libelle']} · {C.fr(geo[v['cle']]['As'], 0)} mm²"
    if v["cle"] == "tau":
        return f"V = {C.nb(s['V'])} kN"
    return f"{geo['etriers']['resume']} · e = {C.nb(s['pas'] if 'pas' in s else geo['etriers']['pas'])} cm"


def _comparaison(v: Dict, s: Dict) -> str:
    """La ligne 2 d'une vérification : ce qu'on compare, en clair (le nécessaire face au posé pour les aciers)."""
    g = s["grandeurs"]
    k = v["cle"]
    le = "≤" if v["taux"] <= 1 else "&gt;"
    if k == "hauteur":
        return f"h<sub>min</sub> {_n(v['valeur'])} {le} h {_n(v['limite'])}{C.NBSP}cm"
    if k in ("inf", "sup"):
        t = f"néc. {_n(v['valeur'], 'mm²')} {le} posé {_n(v['limite'], 'mm²')}{C.NBSP}mm²"
        if "acier_excessif" in v.get("motifs", []):
            t += f" &gt; max {_n(v['As_max'], 'mm²')}"
        return t
    if k == "tau":
        return (f"τ {_n(v['valeur'])} {le} τ<sub>adm,{C.esc(v.get('bande') or '')}</sub> {_n(v['limite'])}{C.NBSP}N/mm² · "
                f"{BESOINS.get(v.get('besoin'), '')}")
    return f"e {_n(v['valeur'])} {le} s<sub>lim</sub> {_n(g.get('s_lim_tronque') or v['limite'])}{C.NBSP}cm · {s['geometrie']['etriers']['brins']}{C.NBSP}brins"


def _ligne_tableau(v: Dict, s: Dict, gov: bool) -> Dict:
    g = s["grandeurs"]
    k = v["cle"]
    if k == "hauteur":
        f, val, lim = "h<sub>min</sub> = h<sub>u,min</sub> + y<sub>G</sub> ≤ h", f"h<sub>min</sub> = {_n(v['valeur'])} cm", f"h = {_n(v['limite'])} cm"
    elif k in ("inf", "sup"):
        f, val, lim = ("max(A<sub>s,req</sub> ; A<sub>s,min</sub>) ≤ A<sub>s</sub>", f"A<sub>s,néc</sub> = {_n(v['valeur'], 'mm²')} mm²",
                       f"A<sub>s,posé</sub> = {_n(v['limite'], 'mm²')} mm²")
    elif k == "tau":
        f = "τ = V / (0,75·b·h) ≤ τ<sub>adm</sub>"
        val, lim = f"τ = {_n(v['valeur'])} N/mm²", f"τ<sub>adm,{C.esc(v.get('bande') or '')}</sub> = {_n(v['limite'])} N/mm²"
    else:
        f, val, lim = "e ≤ min(s<sub>th</sub> ; s<sub>max</sub>)", f"e = {_n(v['valeur'])} cm", f"s<sub>lim</sub> = {_n(g.get('s_lim_tronque') or v['limite'])} cm"
    sous = C.esc(_piece(v, s))
    if k == "tau" and v.get("besoin"):
        sous += " · " + BESOINS.get(v["besoin"], v["besoin"])
    if v.get("motifs"):
        sous += " · " + ", ".join(MOTIFS.get(m, m) for m in v["motifs"])
    return {"nom": C.esc(COURTS.get(k, v["nom"])), "sous": sous, "formule": f, "valeur": val, "limite": lim, "taux": C.pct(v["taux"]), "niveau": _niv(v), "gov": gov}


def _efforts_svg(s: Dict) -> str:
    def m(indice, x):
        return f"M<tspan font-size='70%' dy='4'>{indice}</tspan><tspan dy='-4'> = {C.fr(x, 0)} kN·m</tspan>"
    morceaux = ([m("inf", s["M_inf"])] if s["M_inf"] else []) + ([m("sup", s["M_sup"])] if s["M_sup"] else []) + [f"V = {C.fr(s['V'], 0)} kN"]
    return " · ".join(morceaux)


def _coupe(s: Dict, i: int, titre: str = "") -> Dict:
    """La coupe cotée d'une section (api_web/coupe_ba.py), ses pièces cliquables préfixées « s<i>. »."""
    geo = dict(s["geometrie"])
    tendu = "sup" if s["M_sup"] > s["M_inf"] else "inf"
    geo["tendu"] = tendu
    d = s["grandeurs"][f"d_{tendu}"]
    return CB.coupe(geo, i, _efforts_svg(s), d, True, titre)


def _calculer(p: Dict) -> Dict:
    try:
        return MO.verifier(p)
    except ValueError as e:                                     # pragma: no cover — normaliser l'a déjà refusé
        return {"erreur": str(e)}


def _niveau_note(r: Dict) -> str:
    """Le niveau d'une poutre calculée : non conforme (méthode ou dispositions) > données incomplètes (une section sans
    effort) > à vérifier (une disposition) > le taux."""
    if "taux_max" not in r:
        return "x"
    if r["statut_complet"] == MO.NOK:
        return "r"
    if not r["sollicitee"]:
        return "n"
    if r["statut_complet"] == MO.WARN:
        return "v"
    return C.niveau(r["taux_max"])


def resume_note(etat: Dict, resultats: List[Dict]) -> Dict:
    """Le résumé de la NOTE : sa poutre la plus chargée ; « Données incomplètes » dès qu'une section n'a aucun effort."""
    vals = [(r["taux_max"], i) for i, r in enumerate(resultats) if "taux_max" in r]
    if not vals:
        return {"taux": None, "niveau": "x", "statut": "Calcul impossible", "gouvernant": ""}
    t, i = max(vals)
    r = resultats[i]
    nivs = [_niveau_note(x) for x in resultats]
    niv = next((n for n in ("x", "r", "n", "v") if n in nivs), _niveau_note(r))
    g = r["gouvernant"]
    out = {"taux": C.pct(t) if niv != "n" else None, "niveau": niv, "statut": C.STATUTS[niv][1],
           "gouvernant": f"{etat['poutres'][i].get('nom') or 'Poutre'} · section {g['section']} · {g['nom']}"}
    if niv in C.STATUTS_COURTS:
        out["court"] = C.STATUTS_COURTS[niv]   # la pastille du téléphone
    return out


def elements(etat: Dict, resultats: List[Dict]) -> str:
    """« Poutres de la note » : chaque poutre (nom, dimensions, taux) ; l'ouverte porte ses actions (renommer, copier,
    supprimer) ; « Ajouter une poutre » dessous."""
    items = ""
    seule = len(etat["poutres"]) == 1
    for i, (p, r) in enumerate(zip(etat["poutres"], resultats)):
        t = r.get("taux_max")
        niv = _niveau_note(r)
        txt = "—" if t is None or niv == "n" else f"{C.pc(C.pct(t))}{C.NBSP}%"
        cls = {"n": "c", "x": "r"}.get(niv, niv)
        on = i == etat["actif"]
        actions = ""
        if on:
            actions = (f'<button class="x et-ren" data-renommer="1" title="Renommer">{C.icon("pencil")}</button>'
                       f'<button class="x" data-dupliquer="1" title="Copier cette poutre">{C.icon("copy")}</button>')
        if not seule:
            actions += f'<button class="x" data-supprimer="{i}" title="Supprimer">{C.icon("x")}</button>'
        items += (f'<div class="ni{" on" if on else ""}" data-activer="{i}"><span class="t">{C.esc(p.get("nom") or f"Poutre {i + 1}")} '
                  f'<span>· {C.nb(p["b"])} × {C.nb(p["h"])} · {len(p["sections"])} sect.</span></span><span class="pct {cls}">{txt}</span>{actions}</div>')
    return (f'<div class="elements"><div class="nl-h"><span class="kicker">Poutres de la note · {len(etat["poutres"])}</span></div>{items}'
            f'<div class="nl-add" data-ajouter="1">{C.icon("plus")}Ajouter une poutre</div></div>')


def _hyps(p: Dict, r: Dict) -> List[str]:
    m = r["materiaux"]
    return [f"Béton <b>{C.esc(m['beton'])}</b> · f<sub>ck,cube</sub> = {C.nb(m['fck_cube'])} N/mm²",
            f"Acier <b>B{int(m['fyk'])}</b> · f<sub>yd</sub> = f<sub>yk</sub> / γ<sub>s</sub> = {int(m['fyk'])} / {C.nb(m['gamma_s'])} = {C.fr(m['fyd'], 1)} N/mm²",
            f"α<sub>b</sub> = {C.nb(m['alpha_b'])} · μ = {C.nb(m['mu'], 4)}",
            f"b × h = {C.nb(p['b'])} × {C.nb(p['h'])} cm · enrobage {C.nb(p['enrobage'])} cm"]


def _pied(critere: str) -> str:
    return (f'<div class="ck-crit"><span><b>Critère</b> {critere}</span>'
            f'<span><b>Référence</b> méthode du bureau — {C.esc(MO.REFERENCE)}</span></div>')


def _efforts_txt(s: Dict) -> str:
    return " · ".join(x for x in (f"M<sub>inf</sub> = {C.nb(s['M_inf'])} kN·m" if s["M_inf"] else "",
                                  f"M<sub>sup</sub> = {C.nb(s['M_sup'])} kN·m" if s["M_sup"] else "", f"V = {C.nb(s['V'])} kN") if x)


def _fenetre_dispositions(ident: str, titre: str, s: Dict, hyps: List[str]) -> str:
    dispo = s["dispositions"]
    dg = dispo["dg"]
    lignes = []
    for it in dispo["items"]:
        if it["cle"] in ("lit", "entre_lits", "faces"):
            regle = f"s<sub>libre</sub> ≥ max(Ø ; d<sub>g</sub> + {C.nb(DI.K2)} ; {C.nb(DI.S_ABS)}) mm"
        elif it["cle"] == "enrobage":
            regle = "c + Ø<sub>étrier</sub> ≥ Ø (adhérence, c<sub>min,b</sub> = Ø)"
        elif it["cle"] == "cadre":
            regle = "les barres entre les montants de l'étrier"
        elif it["cle"] == "laterales":
            regle = "sans contact avec une barre"
        elif it["cle"] == "angles_cadre":
            regle = "une barre longitudinale dans chaque angle du cadre"
        else:
            regle = "une barre du lit 1 inférieur"
        val = "—" if it["libre"] is None else (f"{_mm(it['libre'])} mm" if it["libre"] > 0 or it["cle"] != "cadre" else f"débord {_mm(-it['libre'])} mm")
        lim = "—" if not it["mini"] else f"{_mm(it['mini'])} mm"
        lignes.append({"nom": C.esc(it["texte"]), "sous": "", "formule": regle, "valeur": val, "limite": lim, "taux": None,
                       "niveau": _niv_dispo(it["etat"]), "etat_txt": _etat_txt(it["etat"]), "gov": it is dispo["gouvernant"]})
    detail = (f'<div class="ck-g"><div class="ck-t">Espacement libre minimal (EC2 8.2(2), valeurs recommandées)</div>'
              + C.lignes_fx([{"parts": ["s_min", "max(k₁·Ø ; d_g + k₂ ; 20 mm)", f"max({C.nb(DI.K1)} × Ø ; {C.nb(dg)} + {C.nb(DI.K2)} ; {C.nb(DI.S_ABS)})"],
                              "res": f"{_mm(dispo['s_min_ref'])} mm pour Ø20", "niveau": ""}])
              + '<div class="hint">Un espacement positif sous le minimum est « À vérifier » ; un chevauchement ou une barre hors '
                "de l'étrier est « Non conforme » (la disposition ne se pose pas). Ces contrôles ne changent pas le calcul de la "
                "méthode du bureau.</div></div>")
    return C.fenetre_calcul(ident, titre, "dispositions constructives · espacements, enrobage, cadres", hyps, lignes, detail,
                            _pied("s<sub>libre</sub> ≥ s<sub>min</sub> ; aucune barre hors de l'étrier ni au contact d'une autre"),
                            colonnes=("Contrôle", "Règle", "Valeur", "Minimum", "État"))


def resultat(etat: Dict) -> Dict:
    p = _poutre(etat)
    r = MO.verifier(p)
    resultats = [r if i == etat["actif"] else _calculer(q) for i, q in enumerate(etat["poutres"])]
    g = r["gouvernant"]
    t = r["taux_max"]
    niv_note = _niveau_note(r)
    nom_p = C.esc(p.get("nom") or "Poutre")
    hyps = _hyps(p, r)
    # ---- le verdict : la vérification qui gouverne (ou ce qui manque)
    if niv_note == "n":
        vides = ", ".join(C.esc(s["nom"]) for s in r["sections"] if not s["sollicitee"])
        verdict = (f'<div class="verdict et-incomplet"><span class="chip xl c" style="flex:none"><small>η</small>—</span>'
                   f'<div class="gov"><b>Données incomplètes</b>Section{"s" if "," in vides else ""} {vides} sans effort : '
                   f'saisir M<sub>inf</sub>, M<sub>sup</sub> ou V (rubrique 2).</div></div>')
    else:
        sec_g = next(s for s in r["sections"] if s["nom"] == g["section"])
        v_g = next(v for v in sec_g["verifications"] if v["cle"] == g["cle"])
        niv_v = "r" if r["statut_complet"] == MO.NOK else C.niveau(t)
        verdict = (f'<div class="verdict" data-calcul="s{r["sections"].index(sec_g)}-{g["cle"]}" title="Voir le détail">{C.chip(C.pct(t), niv_v, xl=True)}'
                   f'<div class="gov"><b>Section {C.esc(g["section"])} · {C.esc(COURTS.get(g["cle"], g["nom"]).lower())}</b>'
                   f'{_comparaison(v_g, sec_g)}</div></div>')
    rows = ""
    volets = ""
    for i, s in enumerate(r["sections"]):
        niv_s, txt_s = _etat_section(s)
        etat_s = "Données incomplètes" if not s["sollicitee"] else C.STATUTS[niv_s][1]
        rows += (f'<div class="elgrp et-sgrp" data-section="{i}"><span class="et-sn">Section {C.esc(s["nom"])}</span>'
                 f'<span class="et-st {niv_s}">{etat_s}</span><span class="et-sef">{_efforts_txt(s)}</span>'
                 f'<button type="button" class="et-tout" data-calcul="s{i}" title="Tout le calcul de la section">{C.icon("file-text")}</button></div>')
        t_s = max(v["taux"] for v in s["verifications"])
        for v in s["verifications"]:
            pk = f"{v['cle']}-{i}" if v["cle"] in ("inf", "sup") else ""
            piece = C.esc(s["geometrie"][v["cle"]]["libelle"]) if pk else ""
            rows += C.ligne_verif(COURTS.get(v["cle"], v["nom"]), _comparaison(v, s), C.pct(v["taux"]), _niv(v), calcul=f"s{i}-{v['cle']}",
                                  pk=pk, piece=piece, gov=(s["nom"] == g["section"] and v["cle"] == g["cle"] and niv_note != "n"))
            ligne = _ligne_tableau(v, s, True)
            detail = f'<div class="ck-g"><div class="ck-t">{C.esc(v["nom"])}</div>{C.lignes_fx([_formule(f) for f in v["formules"]])}</div>'
            volets += C.fenetre_calcul(f"s{i}-{v['cle']}", f"{nom_p} · section {C.esc(s['nom'])} · {C.esc(COURTS.get(v['cle'], v['nom']).lower())}",
                                       f"{_efforts_txt(s)} · méthode du bureau", hyps, [ligne], detail, _pied(CRITERES[v["cle"]]))
        for nv in s["non_verifiees"]:
            rows += C.ligne_verif(COURTS.get(nv["cle"], nv["nom"]), C.esc(nv["raison"]), None, "c", etat_txt="non effectuée")
        dispo = s["dispositions"]
        gv = dispo["gouvernant"]
        cmp_d = C.esc(gv["texte"]) + (f" {_mm(gv['libre'])} mm (min {_mm(gv['mini'])})" if gv["libre"] is not None and gv["mini"] else "") if gv else ""
        if dispo["etat"] == DI.OK:
            cmp_d = "espacements, enrobage et cadres conformes"
        rows += C.ligne_verif("Dispositions constructives", cmp_d, None, _niv_dispo(dispo["etat"]), calcul=f"s{i}-dispo",
                              etat_txt=_etat_txt(dispo["etat"]))
        volets += _fenetre_dispositions(f"s{i}-dispo", f"{nom_p} · section {C.esc(s['nom'])} · dispositions", s, hyps)
        # la fenêtre de toute la section (le tableau complet et toutes les formules)
        lignes = [_ligne_tableau(v, s, v["taux"] == t_s) for v in s["verifications"]]
        detail = "".join(f'<div class="ck-g"><div class="ck-t">{C.esc(v["nom"])}</div>{C.lignes_fx([_formule(f) for f in v["formules"]])}</div>'
                         for v in s["verifications"])
        volets += C.fenetre_calcul(f"s{i}", f"{nom_p} · section {C.esc(s['nom'])}", f"{_efforts_txt(s)} · méthode du bureau", hyps, lignes, detail)
    messages = "".join(C.message(C.esc(a["message"])) for a in r.get("avertissements", []))
    pied = (f'<div class="pfoot"><div class="mass"><span>Méthode du bureau · moteur poutre</span><b>{C.esc(MO.REFERENCE.split(" (")[0].split()[-1])}</b></div>'
            f'<div class="legend">{C.legende()}</div></div>')
    vues = []
    for i, s in enumerate(r["sections"]):
        vues.append({"cle": f"s{i}", "titre": f"Section {s['nom']}", "court": f"Section {s['nom']}", "svg": CB.svg(f"pba_{i}", [_coupe(s, i)]),
                     "legende": f"{s['geometrie']['inf']['libelle']} en bas · {s['geometrie']['sup']['libelle']} en haut · "
                                f"{s['geometrie']['etriers']['resume']}, e = {C.nb(s['geometrie']['etriers']['pas'])} cm"})
    if len(r["sections"]) > 1:
        vues.append({"cle": "toutes", "titre": "Toutes les sections", "court": "Toutes",
                     "svg": CB.svg("pba_toutes", [_coupe(s, i, f"Section {s['nom']}") for i, s in enumerate(r["sections"])]),
                     "legende": "même échelle"})
    statut = C.statut(niv_note)
    return {"resultat": f'<div class="pbody">{verdict}<div class="els">{rows}</div>{messages}</div>{pied}', "volets": volets, "vues": vues,
            "phrase": f"{C.nb(p['b'])} × {C.nb(p['h'])} cm · {C.esc(p['beton'])} · B{p['fyk']}",
            "statut_html": statut, "resume": resume_note(etat, resultats), "donnees": formulaire(etat, r),
            "elements": elements(etat, resultats)}


# ------------------------------------------------------------------ la note PDF
def telecharger(etat: Dict, quoi: str, infos: Optional[Dict] = None) -> Optional[Dict]:
    """« note_pdf » : la note de calcul PDF de TOUTE la note (chaque poutre, chaque section), faite par le générateur
    validé de l'écran Streamlit (api_web/note_poutre.py) ; ``infos`` : le nom, le chantier et l'indice de la note."""
    if quoi != "note_pdf":
        return None
    from . import note_poutre as NP
    return NP.fichier(etat, infos)


# ------------------------------------------------------------------ le sélecteur d'armatures
def selecteur(etat: Dict, role: str, familles=None, froid: bool = True) -> str:
    """Les dispositions d'un lit pour une face d'une section, triées par section d'acier, calculées par le moteur.
    Celles qui ne se POSENT pas (barres qui se chevauchent ou sortent de l'étrier : « possible » faux) ne sont pas
    proposées ; la ★ va à la plus petite section qui tient ET respecte l'espacement minimal (à défaut, qui tient)."""
    face, i = role.split("-")
    p = _poutre(etat)
    s = p["sections"][int(i)]
    cands = [c for c in MO.candidats_armatures(p, s["nom"], face) if c["possible"] or c["retenue"]]
    opt = next((c for c in cands if c["ok"] and c["possible"] and not c["serree"]), None) or next((c for c in cands if c["ok"] and c["possible"]), None)
    i_o = cands.index(opt) if opt else 0
    i_c = next((k for k, c in enumerate(cands) if c["retenue"]), i_o)

    def prop(c):
        t = f"{c['nom']} · A<sub>s</sub> = {C.fr(c['aire'], 0)} mm² · un lit à {C.nb(c['distance'], 2)} cm du parement"
        if c["serree"]:
            t += " · <b>espacement sous le minimum de l'EC2</b>"
        if not c["possible"]:
            t += " · <b>ne se pose pas (barres trop serrées)</b>"
        return t
    rows = [{"nom": c["nom"], "col": C.fr(c["aire"], 0), "taux": C.pct(min(c["taux"], 9.99)), "niveau": "r" if c["etat"] == MO.NOK else C.niveau(c["taux"]),
             "cur": c["retenue"], "opt": c is opt, "prop": prop(c)} for c in cands]
    face_txt = "inférieures" if face == "inf" else "supérieures"
    return C.selecteur(role, f"Armatures {face_txt} · section {C.esc(s['nom'])}", "un lit · triées par section d'acier · un clic retient la disposition",
                       "", rows, max(0, min(i_o, i_c) - 4), "mm²", etoile="la plus petite section qui tient et se pose")

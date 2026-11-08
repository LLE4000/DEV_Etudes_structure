# -*- coding: utf-8 -*-
"""Le module « Poutres béton armé » du site : l'état d'une note (une ou plusieurs poutres, chacune avec ses sections),
les DONNÉES en arborescence (colonne de gauche : chaque poutre, et sous la poutre ouverte ses quatre rubriques), le
DESSIN cliquable de chaque section (au centre, avec son en-tête technique), les RÉSULTATS par section (colonne de
droite : un bloc par section, dont la liste de contrôle des dispositions constructives), le sélecteur d'armatures —
rendus en fragments HTML.

Le calcul est celui du moteur ``beton/poutre`` (extrait de l'écran Streamlit modules/poutre.py, dont il reproduit les
résultats — tests/test_poutre_moteur.py), complété des DISPOSITIONS CONSTRUCTIVES du ferraillage détaillé (lits
positionnés, cadres multiples, armatures latérales : ``beton/poutre/dispositions.py``). Ce fichier MET EN FORME : il ne
calcule rien ; il CHOISIT parmi ce que le moteur a nommé (la proposition d'armatures retient la plus petite section qui
tient ET se pose) et COMPOSE les phrases.

Les pièces du dessin et les lignes des données portent la MÊME clé (``s<i>.lit-inf-<j>``, ``s<i>.cadre-<k>``,
``s<i>.lat``, ``s<i>.efforts``, ``s<i>.beton``) : ``data-sel`` sur le dessin, ``data-cible`` dans les données. Un clic
sur l'une montre l'autre (js/compact.js). La SECTION COURANTE (celle qu'on règle à gauche) est celle du dessin et des
résultats à droite : chaque bloc porte ``data-section``, l'écran n'en montre qu'un (ou tous, vue « Toutes »).
"""
import copy
from typing import Dict, List, Optional, Sequence

from beton.poutre import dispositions as DI
from beton.poutre import entree as EN
from beton.poutre import geometrie as G
from beton.poutre import moteur as MO

from . import commun as C
from . import coupe_ba as CB
from . import etabli_compact as E

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
    """Les poutres de la note : en ouvrir une, en ajouter, en copier une (``index`` ; sans : l'ouverte), en supprimer
    une (jamais la dernière)."""
    ps = etat["poutres"]
    if action == "activer":
        etat["actif"] = min(max(0, int(index or 0)), len(ps) - 1)
    elif action == "ajouter":
        p = EN.nouvelle_poutre()
        p["nom"] = f"V{len(ps) + 1}"
        ps.append(p)
        etat["actif"] = len(ps) - 1
    elif action == "dupliquer":
        i = etat["actif"] if index is None else min(max(0, int(index)), len(ps) - 1)
        p = copy.deepcopy(ps[i])
        p["nom"] = f"{p.get('nom') or 'Poutre'} (copie)"
        ps.insert(i + 1, p)
        etat["actif"] = i + 1
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
    dit « etrier » ou « epingle »), « dup » (copier la section) ou « - » (retirer). Deux saisies composées :
    « …/etriers/<k>/plage » (les barres que le cadre entoure, « 2-4 » ; une seule barre pour une épingle ; vide : toute
    la largeur) et « …/justifs/<observation> » (la justification de l'ingénieur, {"texte", "empreinte"} ; null : la
    retirer). Le moteur relit la poutre : une valeur hors de son domaine est refusée avec un message, et la poutre reste
    intacte. Une saisie qui retire des barres au lit 1 inférieur ramène les cadres qui les entouraient sur les barres
    qui restent, et le dit."""
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
                # le lit ajouté reprend les barres du précédent, posé au-dessus (EC2 8.2(3)) : « un lit de plus »
                lit = {"n": liste[-1]["n"], "d": liste[-1]["d"]} if liste else dict(EN.LIT_DEFAUT)
                liste.append(lit)
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
        elif chemin[-1] == "plage":
            msg = _plage(p, chemin[:-1], valeur)
            if msg:
                return msg
        elif len(chemin) >= 2 and chemin[-2] == "justifs":
            s = _aller(p, chemin[:-2])
            js = dict(s.get("justifs") or {})
            if valeur in (None, "", {}):
                js.pop(str(chemin[-1]), None)
            elif isinstance(valeur, dict):
                js[str(chemin[-1])] = {"texte": str(valeur.get("texte") or ""), "empreinte": str(valeur.get("empreinte") or "")}
            else:
                return "Justification illisible"
            if js:
                s["justifs"] = js
            else:
                s.pop("justifs", None)
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
        ramenes = _cadres_sur_barres(p)
        etat["poutres"][etat["actif"]] = _noms_uniques(EN.normaliser(p))
    except (ValueError, KeyError, IndexError, TypeError) as e:
        etat["poutres"][etat["actif"]] = avant
        return f"Saisie refusée : {e}"
    return ramenes


def _plage(p: Dict, chemin: Sequence, valeur) -> Optional[str]:
    """Les barres du lit 1 inférieur qu'une ligne d'effort tranchant entoure : « de-a » (un cadre, au moins deux
    barres), « k » (une épingle), vide (un cadre sur toute la largeur ; une épingle sur la barre du milieu). Jamais
    une barre qui n'existe pas : la saisie est refusée."""
    e = _aller(p, chemin)
    s = _aller(p, chemin[:-2])
    n1 = int(s["inf"][0]["n"])
    texte = str(valeur or "").strip()
    if not texte:
        e.pop("de", None)
        e.pop("a", None)
        return None
    try:
        morceaux = [int(x) for x in texte.replace("–", "-").split("-")]
    except ValueError:
        return f"Barres illisibles : « {texte} »"
    if any(not 1 <= m <= n1 for m in morceaux):
        return f"Le lit 1 inférieur compte {n1} barre{'s' if n1 > 1 else ''} : barres 1 à {n1}"
    if e["type"] == "epingle":
        e["de"] = morceaux[0]
        e.pop("a", None)
        return None
    if len(morceaux) != 2:
        return "Un cadre entoure une suite de barres : « de-à »"
    de, a = sorted(morceaux)
    if de == a:
        return "Un cadre entoure au moins deux barres (une seule : une épingle)"
    if de == 1 and a == n1:
        e.pop("de", None)
        e.pop("a", None)
    else:
        e["de"], e["a"] = de, a
    return None


def _cadres_sur_barres(p: Dict) -> Optional[str]:
    """Après une saisie : un cadre qui entourait une barre que le lit 1 inférieur n'a plus est ramené sur les barres qui
    restent (le dessin, la note et les dispositions disent alors la même chose que les données). Rend le message, s'il
    y a lieu."""
    dits = []
    for s in p.get("sections") or []:
        n1 = int(s["inf"][0]["n"])
        for k, e in enumerate(s.get("etriers") or []):
            touche = False
            for cle in ("de", "a"):
                if e.get(cle) is not None and int(e[cle]) > n1:
                    e[cle] = n1
                    touche = True
            if not touche:
                continue
            if e.get("type") == "epingle":
                dits.append(f"C{k + 1} : épingle ramenée sur la barre {e['de']}")
                continue
            de, a = int(e.get("de") or 1), int(e.get("a") or n1)
            if de >= a:
                de = max(1, a - 1)
            if (de == 1 and a == n1) or n1 < 2:
                e.pop("de", None)
                e.pop("a", None)
                dits.append(f"C{k + 1} : cadre ramené à toute la largeur")
            else:
                e["de"], e["a"] = de, a
                dits.append(f"C{k + 1} : cadre ramené aux barres {de} à {a}")
    if not dits:
        return None
    return f"Le lit 1 inférieur compte moins de barres — {' ; '.join(dits)}."


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
    _cadres_sur_barres(p)
    etat["poutres"][etat["actif"]] = EN.normaliser(p)


# ------------------------------------------------------------------ petits morceaux d'écran
def _cm(x: Optional[float]) -> str:
    return C.fr(x, 1) if x is not None else "—"


def _mm(x: Optional[float]) -> str:
    return C.fr(x, 0) if x is not None else "—"


def _items(dispo: Dict, face, lit, *cles) -> List[Dict]:
    return [it for it in dispo["items"] if it["face"] == face and it["lit"] == lit and it["cle"] in cles]


# l'état EFFECTIF d'une observation, en mots et en couleur (« v » : l'ambre « à examiner », jamais le vert des taux ;
# « j » : examinée et justifiée par l'ingénieur — une couleur à part, ni conforme ni en défaut)
ETATS_OBS = {DI.OK: ("p", "Conforme"), DI.WARN: ("v", "À examiner"), DI.NOK: ("r", "Non conforme"),
             DI.JUST: ("j", "Examinée — justifiée")}


def _pire(items: List[Dict]) -> Optional[str]:
    rang = {DI.OK: 0, DI.JUST: 0, DI.WARN: 1, DI.NOK: 2}
    e = max((it["etat_eff"] for it in items), key=lambda x: rang[x], default=None)
    return None if e in (None, DI.OK, DI.JUST) else e


def _point(etat: Optional[str]) -> str:
    """La pastille d'une ligne fermée qui porte un défaut (le détail dit lequel) : rien quand tout va bien."""
    if not etat:
        return ""
    niv, txt = ETATS_OBS[etat]
    return f'<span class="et-pt-e {niv}" title="{C.esc(txt)}"></span>'


# ------------------------------------------------------------------ les données : l'arborescence (colonne de gauche)
# les répartitions d'un lit : le nom, un pictogramme (les barres entre les montants de l'étrier) et ce que fait le mode
REPS = (("uniforme", "Uniforme", "Barres réparties entre les montants de l'étrier."),
        ("aligne", "Superposée", "Chaque barre au-dessus d'une barre de L1, les angles d'abord (EC2 8.2(3))."),
        ("entraxe", "Centrée", "Groupe centré dans la largeur, entraxe s<sub>h</sub> imposé."))
_PICTOS = {"uniforme": ((4, 10), (10, 10), (16, 10), (22, 10)),
           "aligne": ((4, 11), (10, 11), (16, 11), (22, 11), (4, 4.5), (22, 4.5)),
           "entraxe": ((9, 10), (13, 10), (17, 10))}


def _picto(mode: str) -> str:
    pts = "".join(f'<circle cx="{x}" cy="{y}" r="1.9"/>' for x, y in _PICTOS[mode])
    return (f'<svg class="et-pic" viewBox="0 0 26 14" aria-hidden="true"><path d="M1 1.5V12.5M25 1.5V12.5" stroke="currentColor" '
            f'stroke-width="1.1" fill="none" opacity=".45"/><g fill="currentColor">{pts}</g></svg>')


def _repartition(ch: List, rep: str, j: int) -> str:
    bt = "".join(f'<button type="button" class="{"on" if k == rep else ""}" data-chemin="{C._attr_chemin(ch + ["rep"])}" data-sorte="choix" '
                 f'data-valeur="{k}" title="{C.esc(aide.replace("<sub>", "").replace("</sub>", ""))}">{_picto(k)}<span>{t}</span></button>'
                 for k, t, aide in REPS if j > 0 or k != "aligne")
    aide = next(a for k, _, a in REPS if k == rep)
    return f'<div class="seg full et-rep">{bt}</div><div class="hint et-rep-a">{aide}</div>'


def _espacements(dispo: Dict, face: str, j: int) -> str:
    """Les espacements libres du lit, contrôlés : dans le lit (↔), avec le lit précédent (↕), la superposition."""
    out = []
    for it in _items(dispo, face, j, "lit", "entre_lits", "superposes", "cadre"):
        niv, txt = ETATS_OBS[it["etat_eff"]]
        if it["cle"] == "lit":
            quoi = f"↔ libre entre barres <b>{_mm(it['libre'])}{C.NBSP}mm</b> {'≥' if it['libre'] >= it['mini'] else '&lt;'} {_mm(it['mini'])}"
        elif it["cle"] == "entre_lits":
            quoi = f"↕ libre avec L{j} <b>{_mm(it['libre'])}{C.NBSP}mm</b> {'≥' if it['libre'] >= it['mini'] else '&lt;'} {_mm(it['mini'])}"
        elif it["cle"] == "superposes":
            quoi = "barres non superposées à L" + str(j) + " (EC2 8.2(3))"
        else:
            quoi = "barres hors de l'étrier"
        out.append(f'<div class="et-esp {niv}"><span class="et-pt-e {niv}"></span>{quoi}<span class="et-esp-t">{txt}</span></div>')
    return "".join(out)


def _lit(i: int, face: str, j: int, lit: Dict, n_lits: int, rs: Dict) -> str:
    """Un lit : la ligne courte (L1 · n · Ø · aire), puis son détail (répartition, position, espacements) — ouvert
    quand le lit est choisi, ici ou sur le dessin."""
    ch = ["sections", i, face, j]
    geo = rs["geometrie"][face]["lits"][j]
    lits = rs["geometrie"][face]["lits"]
    dispo = rs["dispositions"]
    cible = f"s{i}.lit-{face}-{j}"
    pire = _pire(_items(dispo, face, j, "lit", "entre_lits", "superposes", "cadre"))
    tete = (f'<div class="et-l1"><button type="button" class="et-tag" data-deplier="1" title="Détail du lit">L{j + 1}</button>'
            f'{C.num(ch + ["n"], lit["n"], "", entier=True, mini=1)}'
            f'{C.choix(ch + ["d"], lit["d"], [(x, f"Ø{x}") for x in EN.DIAMETRES])}'
            f'<span class="et-as">{_mm(geo["As"])}{C.NBSP}mm²</span>{_point(pire)}'
            f'<button type="button" class="et-dep" data-deplier="1" title="Répartition et espacements">{C.icon("chevron-down")}</button></div>')
    rep = lit.get("rep", "uniforme")
    det = _repartition(ch, rep, j)
    if rep == "entraxe":
        xs = geo["x"]
        pas = (xs[1] - xs[0]) if len(xs) > 1 else None
        det += C.ligne("s<sub>h</sub> entraxe", C.num(ch + ["sh"], lit.get("sh"), "cm", auto=f"auto · {_cm(pas)}" if pas else "auto"),
                       "", "Axe à axe des barres du lit, groupe centré dans la largeur.")
    if j == 0:
        det += C.ligne("e<sub>1</sub> axe", C.num(ch + ["pos"], lit.get("pos"), "cm", auto=f"auto · {_cm(geo['distance'])}"), "",
                       "Distance du parement à l'axe du lit 1 ; vide : enrobage + Ø étrier + Ø/2 arrondis + jeu (méthode du bureau).")
    else:
        auto = geo["distance"] - lits[j - 1]["distance"]
        det += C.ligne(f"e<sub>v</sub> / L{j}", C.num(ch + ["pos"], lit.get("pos"), "cm", auto=f"auto · {_cm(auto)}"), "",
                       "Entraxe vertical, axe à axe depuis le lit précédent ; vide : demi-Ø arrondis + jeu entre lits (méthode du bureau).")
    det += _espacements(dispo, face, j)
    if n_lits > 1:
        det += E.retirer(ch + ["-"], "Retirer ce lit")
    return f'<div class="et-piece" data-cible="{cible}">{tete}<div class="et-det">{det}</div></div>'


def _face(i: int, face: str, s: Dict, rs: Dict) -> str:
    nom = "Inférieures" if face == "inf" else "Supérieures"
    geo = rs["geometrie"][face]
    v = next(x for x in rs["verifications"] if x["cle"] == face)
    corps = "".join(_lit(i, face, j, lit, len(s[face]), rs) for j, lit in enumerate(s[face]))
    ajouter = E.plus(["sections", i, face, "+"], "1", "Lit") if len(s[face]) < EN.MAX_LITS else ""
    yg = C.ligne("y<sub>G</sub> imposé", C.num(["sections", i, f"yG_{face}"], s[f"yG_{face}"], "cm", auto=f"auto · {_cm(geo['yG_calcule'])}"),
                 "", "Centre de gravité des lits de la face, mesuré depuis le parement. Vide : calculé (Σ A·e / Σ A) sur les "
                     "positions réelles des lits. L'imposer remplace le calcul dans d = h − y_G.")
    prop = (f'<button type="button" class="et-prop" data-pk="{face}-{i}" title="Proposer une disposition qui tient (sélecteur)">'
            f'{C.icon("lightbulb")}</button>')
    return (f'<div class="et-grp"><div class="et-gh"><span class="t">{nom}</span><span class="et-req">A<sub>s</sub> requis '
            f'<b>{_mm(v["valeur"])}</b>{C.NBSP}mm²</span>{prop}</div>{corps}<div class="et-pied">{ajouter}</div>{yg}</div>')


def _barres(ch: List, e: Dict, cd: Dict, n1: int) -> str:
    """Le choix des barres entourées : les barres du lit 1 inférieur, la plage du cadre éclairée — un clic pose un bout
    (le plus proche), jamais sur une barre qui n'existe pas. « Toute la largeur » pour un cadre complet."""
    ep = e["type"] == "epingle"
    de, a = cd["de"], cd["a"]
    bt = "".join(f'<button type="button" class="b{" on" if de <= k <= a else ""}{" bout" if k in (de, a) else ""}" data-barre="{k}" '
                 f'title="Barre {k}">{k}</button>' for k in range(1, n1 + 1))
    tout = "" if ep else (f'<button type="button" class="et-large{" on" if cd["complet"] else ""}" data-plage-tout="1">'
                          f'Toute la largeur</button>')
    quoi = "Sur la barre" if ep else "Entoure les barres"
    return (f'<div class="row full et-plage"><label><span>{quoi}</span><span class="aide" title="Les barres du lit 1 inférieur ; un clic pose le bout le plus proche. '
            f'Sur le dessin, les poignées du cadre choisi glissent d\'une barre à l\'autre.">{C.icon("info")}</span></label>'
            f'<div class="et-barres" data-plage="{C._attr_chemin(ch + ["plage"])}" data-type="{e["type"]}" data-de="{de}" data-a="{a}" '
            f'data-n="{n1}">{bt}</div>{tout}</div>')


def _cadres(i: int, s: Dict, rs: Dict) -> str:
    geo = rs["geometrie"]["etriers"]
    dispo = rs["dispositions"]
    n1 = s["inf"][0]["n"]
    out = ""
    for k, (e, cd) in enumerate(zip(s["etriers"], geo["cadres"])):
        ch = ["sections", i, "etriers", k]
        ep = e["type"] == "epingle"
        obs = [it for it in dispo["items"] if it["id"] in (f"angles_cadre:{k}", f"epingle:{k}", f"position_cadre:{k}:de", f"position_cadre:{k}:a")]
        quoi = (f"épingle · barre {cd['de']}" if ep else ("toute la largeur" if cd["complet"] else f"barres {cd['de']}–{cd['a']}"))
        tete = (f'<div class="et-l1"><button type="button" class="et-tag c" data-deplier="1" title="Détail">C{k + 1}</button>'
                f'<span class="et-txt">Ø{e["d"]} · {cd["brins"]}{C.NBSP}brin{"s" if cd["brins"] > 1 else ""} · {quoi}</span>{_point(_pire(obs))}'
                f'<button type="button" class="et-dep" data-deplier="1" title="Barres entourées">{C.icon("chevron-down")}</button></div>')
        det = (C.ligne("Type", C.segments(ch + ["type"], e["type"], [("etrier", "Cadre · 2 brins"), ("epingle", "Épingle · 1 brin")]), full=True)
               + C.ligne("Ø", C.choix(ch + ["d"], e["d"], [(x, f"Ø{x}") for x in EN.DIAMETRES_ETRIERS]))
               + _barres(ch, e, cd, n1))
        if not ep:
            det += f'<div class="hint et-info">emprise {_cm(cd["x1"] - cd["x0"])}{C.NBSP}cm hors tout · {"cadre extérieur" if cd["complet"] else "cadre intérieur"}</div>'
        for it in obs:
            niv, txt = ETATS_OBS[it["etat_eff"]]
            det += f'<div class="et-esp {niv}"><span class="et-pt-e {niv}"></span>{C.esc(it["texte"].split(" : ", 1)[-1])}<span class="et-esp-t">{txt}</span></div>'
        if len(s["etriers"]) > 1:
            det += E.retirer(ch + ["-"], "Retirer cette ligne")
        out += f'<div class="et-piece" data-cible="s{i}.cadre-{k}">{tete}<div class="et-det">{det}</div></div>'
    pas = C.ligne("e espacement", C.num(["sections", i, "pas"], s["pas"], "cm"), "",
                  "Espacement LONGITUDINAL des cadres, le long de la poutre (le pas). La configuration de la section — combien "
                  "de cadres, lesquels entourent quelles barres — se règle ligne par ligne.")
    ajouts = (f'<div class="et-pied">{E.plus(["sections", i, "etriers", "+"], "etrier", "Cadre")}'
              f'{E.plus(["sections", i, "etriers", "+"], "epingle", "Épingle")}</div>')
    return (f'<div class="et-grp"><div class="et-gh"><span class="t">Étriers</span><span class="et-req">{geo["brins"]}{C.NBSP}brins · '
            f'A<sub>st</sub> <b>{_mm(geo["Ast"])}</b>{C.NBSP}mm²</span></div>{pas}{out}{ajouts}</div>')


def _laterales(i: int, s: Dict, rs: Dict) -> str:
    lat = rs["geometrie"]["laterales"]
    imp = s.get("laterales") or {}
    ch = ["sections", i, "laterales"]
    resume = ("aucune" if not lat["n"] else f"2 × {lat['n']}Ø{int(lat['d'])}") + (" · auto" if lat["auto"] else "")
    pire = _pire([x for x in rs["dispositions"]["items"] if x["cle"] == "laterales"])
    tete = (f'<div class="et-l1"><button type="button" class="et-tag l" data-deplier="1" title="Détail">LAT</button>'
            f'<span class="et-txt">Latérales · {resume}</span>{_point(pire)}'
            f'<button type="button" class="et-dep" data-deplier="1" title="Latérales">{C.icon("chevron-down")}</button></div>')
    regle = (f"Automatique : écart entre lits 1 de {_cm(lat['ecart'])}{C.NBSP}cm"
             + (f" &gt; s<sub>max</sub> = {_cm(lat['s_max'])}{C.NBSP}cm" if lat["n"] else f" ≤ s<sub>max</sub> = {_cm(lat['s_max'])}{C.NBSP}cm : aucune")
             if lat["auto"] else "Imposées")
    det = (C.ligne("n par face", C.num(ch + ["n"], imp.get("n"), "", entier=True, auto=f"auto · {lat['n']}"), "",
                   "Barres par face latérale ; vide : la règle automatique de la note PDF (armatures de peau).")
           + C.ligne("Ø", C.choix(ch + ["d"], int(lat["d"]), [(x, f"Ø{x}") for x in EN.DIAMETRES_LATERAUX]))
           + f'<div class="hint et-info">{regle} · dessin et note, hors calcul</div>')
    return f'<div class="et-grp et-grp-l"><div class="et-piece" data-cible="s{i}.lat">{tete}<div class="et-det">{det}</div></div></div>'


def _resume_section(s: Dict) -> str:
    fermes = [e for e in s["etriers"] if e["type"] == "etrier"]
    return f"{_detail_lits(s['inf'])} / {_detail_lits(s['sup'])} · {len(fermes)} cadre{'s' if len(fermes) > 1 else ''}"


def formulaire(etat: Dict, res: Optional[Dict] = None) -> str:
    """Les quatre rubriques de la poutre ouverte : 1 Poutre, 2 Sections et sollicitations, 3 Armatures (de la section
    courante), 4 Options. ``res`` : le calcul de la poutre ouverte (``MO.verifier``), refait s'il manque."""
    p = _poutre(etat)
    res = res or MO.verifier(p)
    L = C.ligne
    # condensée (demande de l'ingénieur, 11/2026) : le nom se modifie dans l'arborescence ; béton et acier, largeur et
    # hauteur, chacun sur une ligne ; l'enrobage dessous
    s1 = ('<div class="et-piece et-plat" data-cible="beton">'
          + C.ligne2(("Béton", C.choix(["beton"], p.get("beton"), [(k, k) for k in EN.classes_beton()]), ""),
                     ("Acier", C.choix(["fyk"], p.get("fyk"), [(n, f"B{n}") for n in EN.NUANCES]), ""))
          + C.ligne2(("b largeur", C.num(["b"], p.get("b"), "cm"), ""), ("h hauteur", C.num(["h"], p.get("h"), "cm"), ""))
          + L("c enrobage", C.num(["enrobage"], p.get("enrobage"), "cm"), "", "Distance du parement au nu de l'étrier.")
          + "</div>")
    seule = len(p["sections"]) == 1
    cartes = "".join(E.carte_section(i, s["nom"], ["sections", i, "dup"], ["sections", i, "-"], seule,
                                     f'<div class="trio">'
                                     f'<label><span>M<sub>inf</sub></span>{C.num(["sections", i, "M_inf"], s["M_inf"], "kN·m")}</label>'
                                     f'<label><span>M<sub>sup</sub></span>{C.num(["sections", i, "M_sup"], s["M_sup"], "kN·m")}</label>'
                                     f'<label><span>V</span>{C.num(["sections", i, "V"], s["V"], "kN")}</label></div>')
                     for i, s in enumerate(p["sections"]))
    s2 = (f'<div class="et-secs">{cartes}</div>'
          + E.pied_sections("M_inf tend le bas (travée), M_sup tend le haut (appui) ; valeurs de service (ELS).",
                            "M<sub>inf</sub> travée · M<sub>sup</sub> appui · ELS"))
    s3 = E.armatures([s["nom"] for s in p["sections"]],
                     [_face(i, "inf", s, res["sections"][i]) + _face(i, "sup", s, res["sections"][i]) + _cadres(i, s, res["sections"][i])
                      + _laterales(i, s, res["sections"][i]) for i, s in enumerate(p["sections"])])
    noms = " · ".join(C.esc(s["nom"]) for s in p["sections"][:3]) + (" …" if len(p["sections"]) > 3 else "")
    resumes3 = "".join(f'<span class="et-r3" data-section="{i}">{C.esc(_resume_section(s))}</span>' for i, s in enumerate(p["sections"]))
    return (C.section(1, "Poutre", f"{C.esc(p['beton'])} · B{p['fyk']} · c = {C.nb(p['enrobage'])} cm", s1, True, "pt")
            + C.section(2, "Sections et sollicitations", noms, s2, False, "so")
            + C.section(3, "Armatures", resumes3, s3, False, "arm")
            + C.section(4, "Options", "méthode du bureau", _options(p), False, "op", secondaire=True))


def arbre(etat: Dict, res: Optional[Dict] = None) -> str:
    """La colonne de gauche : l'ARBORESCENCE de la note (api_web/etabli_compact.py, la même que les dalles et les
    prédalles). Chaque poutre, « nom · b × h cm », sans taux ni pastille (le résultat est à droite) ; la poutre ouverte se
    développe et montre ses quatre rubriques ; un clic sur une autre l'ouvre (la première se replie). Copier, supprimer,
    renommer : dans le menu « ⋯ » de la poutre."""
    elements = [(p.get("nom") or f"Poutre {i + 1}", f"· {C.nb(p['b'])} × {C.nb(p['h'])} cm", False)
                for i, p in enumerate(etat["poutres"])]
    return E.arbre(E.POUTRE, elements, etat["actif"], formulaire(etat, res), "poutre")


def _options(p: Dict) -> str:
    L = C.ligne
    dg = float(p.get("dg") or EN.DEFAUTS_DETAIL["dg"])
    # ce que le jeu entre lits donne pour deux lits Ø16 (le cas courant), lu sur les règles du moteur
    libre16 = (2 * G.arrondi_demi_cm(16 / 20.0) + float(p["jeu_entre_lits"])) * 10.0 - 16.0
    smin16 = DI.s_min(16, 16, dg)
    alerte = (f'<div class="et-avert r"><b>Jeu entre lits insuffisant</b> : deux lits Ø16 gardent {_mm(libre16)}{C.NBSP}mm libres, '
              f'sous le minimum de l\'EC2 8.2(2) ({_mm(smin16)}{C.NBSP}mm) — non conforme. Porter ce jeu à '
              f'{C.nb((smin16 + 16.0) / 10.0 - 2 * G.arrondi_demi_cm(0.8))}{C.NBSP}cm, ou imposer l\'entraxe vertical des lits.</div>'
              if libre16 < smin16 else "")
    return (f'<div class="row full et-meth"><label><span>Méthode</span></label><b>Méthode du bureau</b><span class="hint">'
            f'{C.esc(MO.REFERENCE)} · ELS, z = 0,9·d</span></div>'
            + C.ligne2(("Jeu sous L1", C.num(["jeu_premier_lit"], p.get("jeu_premier_lit"), "cm"), "Ajouté sous le lit 1, après l'enrobage et l'étrier."),
                       ("Jeu entre lits", C.num(["jeu_entre_lits"], p.get("jeu_entre_lits"), "cm"),
                        "Ajouté entre les demi-diamètres arrondis de deux lits."))
            + alerte
            + C.ligne2(("γ<sub>s</sub> acier", C.num(["gamma_s"], p.get("gamma_s"), ""), "Coefficient de l'acier : f_yd = f_yk / γ_s."),
                       ("d<sub>g</sub> granulat", C.num(["dg"], dg, "mm"),
                        "Le plus gros granulat : espacement libre minimal EC2 8.2(2) = max(Ø ; d_g + 5 mm ; 20 mm)."))
            + C.ligne2(("Peau auto Ø", C.choix(["peau_d"], int(p.get("peau_d") or EN.DEFAUTS_DETAIL["peau_d"]),
                                                [(x, f"Ø{x}") for x in EN.DIAMETRES_LATERAUX]),
                        "Le Ø des armatures latérales (de peau) automatiques."),
                       ("Peau s<sub>max</sub>", C.num(["peau_smax"], p.get("peau_smax") or EN.DEFAUTS_DETAIL["peau_smax"], "cm"),
                        "Au-delà de cet écart entre les lits 1, des barres latérales (de peau) sont ajoutées (règle de la note PDF).")))


def _detail_lits(lits: List[Dict]) -> str:
    return " + ".join(f"{lit['n']}Ø{lit['d']}" for lit in lits)


# ------------------------------------------------------------------ le résultat
MOTIFS = {"hauteur_insuffisante": "hauteur insuffisante", "acier_insuffisant": "acier insuffisant",
          "acier_excessif": "acier excessif (A<sub>s</sub> > A<sub>s,max</sub>)", "d_utile_negatif": "hauteur utile négative",
          "barres_inclinees": "barres inclinées nécessaires", "cisaillement_excessif": "cisaillement excessif",
          "pas_trop_grand": "pas trop grand"}
# les noms courts des lignes du résultat (les titres complets du module restent dans le détail du calcul)
COURTS = {"hauteur": "Hauteur", "inf": "Armatures inférieures", "sup": "Armatures supérieures", "tau": "Effort tranchant",
          "pas": "Espacement des étriers"}
# ce que dit la bande de τ atteinte : jamais « pas besoin d'étriers » à côté d'une ligne qui vérifie les étriers posés
BESOINS = {"pas_d_etriers": "étriers de calcul non requis (minimum constructif)", "etriers": "étriers de calcul requis",
           "barres_inclinees_et_etriers": "barres inclinées et étriers", "inacceptable": "section insuffisante"}
# … et sa forme courte, à côté du nom de la vérification (la ligne 1 du résultat)
BESOINS_COURTS = {"pas_d_etriers": "minimum constructif", "etriers": "étriers requis",
                  "barres_inclinees_et_etriers": "barres inclinées + étriers", "inacceptable": "section insuffisante"}
CRITERES = {"hauteur": "h<sub>min</sub> = h<sub>u,min</sub> + y<sub>G</sub> ≤ h",
            "inf": "A<sub>s,requis</sub> = max(A<sub>s,req</sub> ; A<sub>s,min</sub>) ≤ A<sub>s,posé</sub> ≤ A<sub>s,max</sub>",
            "sup": "A<sub>s,requis</sub> = max(A<sub>s,req</sub> ; A<sub>s,min</sub>) ≤ A<sub>s,posé</sub> ≤ A<sub>s,max</sub>",
            "tau": "τ = V / (0,75·b·h) ≤ τ<sub>adm</sub> (bandes I, II, IV du béton)",
            "pas": "e ≤ s<sub>lim</sub> = min(s<sub>th</sub> ; s<sub>max</sub>)"}


def _niv(v: Dict) -> str:
    """La couleur d'une vérification : le taux du moteur, et rouge dès que le module la déclare non conforme
    (acier excessif, hauteur utile négative…) même sous 100 %."""
    return "r" if v.get("etat") == MO.NOK else C.niveau(v["taux"])


def _etat_bloc(rs: Dict):
    """L'état d'une SECTION, la couleur de l'en-tête de son bloc : NON CONFORME d'abord — une non-conformité trouvée
    sans aucun effort (barres trop serrées, acier excessif) n'est jamais cachée par « Données incomplètes » ; puis
    « Données incomplètes » sans aucun effort (ses vérifications passent toutes et ne disent rien) ; sinon à vérifier
    (une observation à examiner), ou conforme — avec le nombre d'observations justifiées par l'ingénieur, s'il y en a."""
    if rs["statut_complet"] == MO.NOK:
        return "nok", "Non conforme"
    if not rs["sollicitee"]:
        return "inc", "Données incomplètes"
    if rs["statut_complet"] == MO.WARN:
        return "warn", "À vérifier"
    nj = rs["dispositions"]["compte"].get(DI.JUST, 0)
    return "ok", "Conforme" + (f" · {nj} justification{'s' if nj > 1 else ''}" if nj else "")


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
        parts = ["A_s,requis", "max(A_s,req ; A_s,min)", f"max({_n(t['As_req'], 'mm²')} ; {_n(t['As_min'], 'mm²')})"]
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


def _limite_pas(s: Dict) -> str:
    """Ce qui fixe s_lim : le calcul (s_th) ou la limite constructive (s_max)."""
    g = s["grandeurs"]
    return "s<sub>max</sub>" if g["s_max"] <= g["s_th"] else "s<sub>th</sub>"


def _comparaison(v: Dict, s: Dict) -> str:
    """La ligne 2 d'une vérification : la valeur calculée et sa limite, nommées, en clair."""
    g = s["grandeurs"]
    k = v["cle"]
    le = "≤" if v["taux"] <= 1 else "&gt;"
    if k == "hauteur":
        return f"h<sub>min</sub> = {_n(v['valeur'])}{C.NBSP}cm {le} h = {_n(v['limite'])}{C.NBSP}cm"
    if k in ("inf", "sup"):
        t = f"A<sub>s,requis</sub> = {_n(v['valeur'], 'mm²')} {le} A<sub>s,posé</sub> = {_n(v['limite'], 'mm²')}{C.NBSP}mm²"
        if "acier_excessif" in v.get("motifs", []):
            t += f" &gt; A<sub>s,max</sub> = {_n(v['As_max'], 'mm²')}"
        return t
    if k == "tau":
        return f"τ = {_n(v['valeur'])} {le} τ<sub>adm,{C.esc(v.get('bande') or '')}</sub> = {_n(v['limite'])}{C.NBSP}N/mm²"
    return f"e = {_n(v['valeur'])} {le} s<sub>lim</sub> = {_n(g.get('s_lim_tronque') or v['limite'])}{C.NBSP}cm ({_limite_pas(s)})"


def _piece_verif(v: Dict, s: Dict) -> str:
    """Ce que la ligne 1 d'une vérification nomme à côté de son titre : les barres posées (un bouton qui ouvre le
    sélecteur, ``_bloc``), la bande de τ et ce qu'elle demande, les brins des étriers."""
    if v["cle"] in ("inf", "sup"):
        return C.esc(s["geometrie"][v["cle"]]["libelle"])
    if v["cle"] == "tau":
        return BESOINS_COURTS.get(v.get("besoin"), "")
    if v["cle"] == "pas":
        n = s["geometrie"]["etriers"]["brins"]
        return f"{n}{C.NBSP}brin{'s' if n > 1 else ''}"
    return ""


def _piece(v: Dict, s: Dict) -> str:
    geo = s["geometrie"]
    if v["cle"] == "hauteur":
        return f"{C.nb(geo['b'])} × {C.nb(geo['h'])} cm"
    if v["cle"] in ("inf", "sup"):
        return f"{geo[v['cle']]['libelle']} · {C.fr(geo[v['cle']]['As'], 0)} mm²"
    if v["cle"] == "tau":
        return f"V = {C.nb(s['V'])} kN"
    return f"{geo['etriers']['resume']} · e = {C.nb(geo['etriers']['pas'])} cm"


def _ligne_tableau(v: Dict, s: Dict, gov: bool) -> Dict:
    g = s["grandeurs"]
    k = v["cle"]
    if k == "hauteur":
        f, val, lim = "h<sub>min</sub> = h<sub>u,min</sub> + y<sub>G</sub> ≤ h", f"h<sub>min</sub> = {_n(v['valeur'])} cm", f"h = {_n(v['limite'])} cm"
    elif k in ("inf", "sup"):
        f, val, lim = ("max(A<sub>s,req</sub> ; A<sub>s,min</sub>) ≤ A<sub>s</sub>", f"A<sub>s,requis</sub> = {_n(v['valeur'], 'mm²')} mm²",
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


# ------------------------------------------------------------------ le dessin et son en-tête
def _efforts_txt(s: Dict) -> str:
    return " · ".join(x for x in (f"M<sub>inf</sub> = {C.nb(s['M_inf'])} kN·m" if s["M_inf"] else "",
                                  f"M<sub>sup</sub> = {C.nb(s['M_sup'])} kN·m" if s["M_sup"] else "", f"V = {C.nb(s['V'])} kN") if x)


def _efforts_plat(s: Dict) -> str:
    """Les efforts en texte simple (le titre d'une coupe de la vue « Toutes »)."""
    return " · ".join(x for x in (f"Minf {C.fr(s['M_inf'], 1)} kN·m" if s["M_inf"] else "",
                                  f"Msup {C.fr(s['M_sup'], 1)} kN·m" if s["M_sup"] else "", f"V {C.fr(s['V'], 1)} kN") if x)


def _resume_etriers(geo: Dict) -> str:
    """« 2 cadres Ø10 + 1 épingle Ø8 · e = 20 cm » : la ligne 3 de l'en-tête du dessin."""
    compte: Dict = {}
    for cd in geo["etriers"]["cadres"]:
        cle = ("épingle" if cd["type"] == "epingle" else "cadre", int(cd["d"]))
        compte[cle] = compte.get(cle, 0) + 1
    morceaux = [f"{n} {nom}{'s' if n > 1 else ''} Ø{d}" for (nom, d), n in compte.items()]
    out = " + ".join(morceaux) + f" · e = {C.nb(geo['etriers']['pas'])} cm"
    lat = geo.get("laterales") or {}
    if lat.get("n"):
        out += f" · latérales 2 × {lat['n']}Ø{int(lat['d'])}" + (" (auto)" if lat.get("auto") else "")
    return out


def _entete(p: Dict, titre: str, lignes: List[str], efforts: Optional[Dict]) -> str:
    """L'en-tête technique d'une vue (api_web/etabli_compact.py) : trois lignes (la section et ses matériaux ; les
    aciers ; les étriers) et, à droite, la bande compacte des efforts — une fois, sans couleur d'alerte."""
    bande = None if efforts is None else [("M", "inf", efforts["M_inf"], "kN·m"), ("M", "sup", efforts["M_sup"], "kN·m"),
                                          ("V", "", efforts["V"], "kN")]
    return E.entete(titre, f'{C.nb(p["b"])} × {C.nb(p["h"])} cm · {C.esc(p["beton"])} · B{p["fyk"]}', lignes, bande)


def _coupe(s: Dict, i: int, titre: str = "", sous_titre: str = "") -> Dict:
    """La coupe cotée d'une section (api_web/coupe_ba.py), ses pièces cliquables préfixées « s<i>. »."""
    geo = dict(s["geometrie"])
    tendu = "sup" if s["M_sup"] > s["M_inf"] else "inf"
    geo["tendu"] = tendu
    d = s["grandeurs"][f"d_{tendu}"]
    return CB.coupe(geo, i, d, True, titre, sous_titre)


def _vues(p: Dict, r: Dict) -> List[Dict]:
    vues = []
    for i, s in enumerate(r["sections"]):
        geo = s["geometrie"]
        lignes = [f"Inf. {geo['inf']['libelle']} · Sup. {geo['sup']['libelle']}", _resume_etriers(geo)]
        vues.append({"cle": f"s{i}", "titre": f"Section {s['nom']}", "court": s["nom"], "svg": CB.svg(f"pba_{i}", [_coupe(s, i)]),
                     "entete": _entete(p, f"SECTION {s['nom']}", lignes, s), "legende": ""})
    if len(r["sections"]) > 1:
        vues.append({"cle": "toutes", "titre": "Toutes les sections", "court": "Toutes",
                     "svg": CB.svg("pba_toutes", [_coupe(s, i, s["nom"], _efforts_plat(s)) for i, s in enumerate(r["sections"])]),
                     "entete": _entete(p, "TOUTES LES SECTIONS", [f"{len(r['sections'])} sections · même échelle"], None), "legende": ""})
    return vues


# ------------------------------------------------------------------ les résultats : un bloc par section
def _calculer(p: Dict) -> Dict:
    try:
        return MO.verifier(p)
    except ValueError as e:                                     # pragma: no cover — normaliser l'a déjà refusé
        return {"erreur": str(e)}


def _niveau_note(r: Dict) -> str:
    """Le niveau d'une poutre calculée : non conforme (méthode ou dispositions) > données incomplètes (une section sans
    effort) > à vérifier (une observation à examiner) > le taux."""
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
    """Rien : les poutres de la note sont l'ARBORESCENCE des données (``arbre``), avec leurs rubriques."""
    return ""


def _hyps(p: Dict, r: Dict) -> List[str]:
    m = r["materiaux"]
    return [f"Béton <b>{C.esc(m['beton'])}</b> · f<sub>ck,cube</sub> = {C.nb(m['fck_cube'])} N/mm²",
            f"Acier <b>B{int(m['fyk'])}</b> · f<sub>yd</sub> = f<sub>yk</sub> / γ<sub>s</sub> = {int(m['fyk'])} / {C.nb(m['gamma_s'])} = {C.fr(m['fyd'], 1)} N/mm²",
            f"α<sub>b</sub> = {C.nb(m['alpha_b'])} · μ = {C.nb(m['mu'], 4)}",
            f"b × h = {C.nb(p['b'])} × {C.nb(p['h'])} cm · enrobage {C.nb(p['enrobage'])} cm"]


def _pied(critere: str) -> str:
    return (f'<div class="ck-crit"><span><b>Critère</b> {critere}</span>'
            f'<span><b>Référence</b> méthode du bureau — {C.esc(MO.REFERENCE)}</span></div>')


# les groupes de la liste de contrôle, dans l'ordre où on les traite : ce qui demande une action d'abord
GROUPES_OBS = ((DI.NOK, "Non conformes"), (DI.WARN, "À examiner"), (DI.JUST, "Examinées et justifiées par l'ingénieur"),
               (DI.OK, "Conformes automatiquement"))
# comment corriger une non-conformité (la liste de contrôle le dit à côté de l'observation)
CORRIGER = {"lit": "réduire le nombre de barres ou leur Ø, ou élargir la section",
            "entre_lits": "imposer un entraxe vertical plus grand au lit (e<sub>v</sub>) ou porter le jeu entre lits (Options)",
            "cadre": "réduire l'entraxe imposé ou le nombre de barres",
            "faces": "réduire les lits ou augmenter la hauteur",
            "laterales": "changer le nombre de latérales ou leur Ø",
            "enrobage": "augmenter l'enrobage ou le Ø des étriers (c + Ø étrier ≥ Ø)"}
# (l'angle sans barre et l'épingle sans barre haute le disent déjà dans leur texte : « barres de montage à prévoir »)
EXAMINER = {"superposes": "choisir la répartition « Superposée », ou justifier",
            "position_cadre": "choisir les barres du cadre parmi celles du lit 1"}


def _piece_obs(i: int, it: Dict) -> str:
    """La pièce qu'une observation désigne (la clé ``data-sel`` du dessin, ``data-cible`` des données) : un clic sur
    l'observation la montre — le lit, le cadre, les latérales ; le béton (b, h, c) pour l'enrobage et les faces."""
    k = it["id"].split(":")
    if it["cle"] in ("lit", "cadre", "entre_lits", "superposes"):
        return f"s{i}.lit-{k[1]}-{k[2]}"
    if it["cle"] in ("angles_cadre", "epingle", "position_cadre"):
        return f"s{i}.cadre-{k[1]}"
    if it["cle"] == "laterales":
        return f"s{i}.lat"
    return f"s{i}.beton"


def _observation(i: int, it: Dict, regle: str) -> str:
    """Une observation de la liste de contrôle : l'état (conforme automatiquement, non conforme, à examiner, examinée et
    justifiée), la valeur et son minimum, ce qu'il faut faire ; « Marquer examinée » pour une observation À EXAMINER
    seulement — jamais pour une non-conformité, qui ne se lève qu'en corrigeant."""
    eff = it["etat_eff"]
    niv, txt = ETATS_OBS[eff]
    val = ""
    if it["libre"] is not None and it["mini"]:
        val = f'<span class="v">{_mm(it["libre"])} {"≥" if it["libre"] >= it["mini"] else "&lt;"} {_mm(it["mini"])} mm</span>'
    elif it["libre"] is not None and it["cle"] == "cadre":
        val = f'<span class="v">débord {_mm(-it["libre"])} mm</span>'
    bas = f'<span class="et-mot {niv}">{txt}</span>' if eff != DI.OK else ""
    act = ""
    if eff == DI.NOK and it["cle"] in CORRIGER:
        bas += f'<span class="o-a">Corriger : {CORRIGER[it["cle"]]}</span>'
    elif eff == DI.WARN:
        if it.get("justification_perimee") is not None:
            bas += f'<span class="o-a">La disposition a changé : justification à revoir (« {C.esc(it["justification_perimee"])} »)</span>'
        elif it["cle"] in EXAMINER:
            bas += f'<span class="o-a">{EXAMINER[it["cle"]]}</span>'
        act = (f'<button type="button" class="et-just" data-justifier="{i}" data-obs="{C.esc(it["id"])}" data-empreinte="{C.esc(it["empreinte"])}">'
               f'{C.icon("check")}Marquer examinée…</button>')
    elif eff == DI.JUST:
        j = it.get("justification") or ""
        bas += f'<span class="o-a">{"« " + C.esc(j) + " »" if j else "sans commentaire"}</span>'
        act = (f'<button type="button" class="et-just et-dej" data-dejustifier="{i}" data-obs="{C.esc(it["id"])}">'
               f'{C.icon("x")}Annuler</button>')
    return (f'<div class="et-obs {niv}" data-obs="{C.esc(it["id"])}" data-piece="{_piece_obs(i, it)}" title="Montrer la pièce">'
            f'<span class="et-pt-e {niv}"></span><div class="ob">'
            f'<div class="o1"><span class="t">{C.esc(it["texte"])}</span>{val}</div>'
            f'{f"<div class=o2>{bas}</div>" if bas else ""}{f"<div class=o3>{act}</div>" if act else ""}'
            f'{f"<div class=o-r>{regle}</div>" if regle else ""}</div></div>')


def _dispositions(i: int, s: Dict) -> str:
    """La ligne « Dispositions constructives » et sa LISTE DE CONTRÔLE (dépliée au clic) : d'abord ce qui demande une
    action (non conforme, à examiner), puis ce qui est justifié, puis ce qui est conforme automatiquement."""
    dispo = s["dispositions"]
    cpt = dispo["compte"]
    morceaux = [f"{cpt[DI.NOK]} non conforme{'s' if cpt[DI.NOK] > 1 else ''}" if cpt[DI.NOK] else "",
                f"{cpt[DI.WARN]} à examiner" if cpt[DI.WARN] else "",
                f"{cpt[DI.JUST]} justifiée{'s' if cpt[DI.JUST] > 1 else ''}" if cpt[DI.JUST] else "",
                f"{cpt[DI.OK]} conforme{'s' if cpt[DI.OK] > 1 else ''}" if cpt[DI.OK] else ""]
    resume = " · ".join(m for m in morceaux if m)
    etat = DI.NOK if cpt[DI.NOK] else (DI.WARN if cpt[DI.WARN] else (DI.JUST if cpt[DI.JUST] else DI.OK))
    niv, txt = ETATS_OBS[etat]
    if etat == DI.JUST:
        txt = "Conforme · justifiées"
    regle_smin = (f"s<sub>min</sub> = max(Ø ; d<sub>g</sub> + {C.nb(DI.K2)} ; {C.nb(DI.S_ABS)}) mm — "
                  f"{_mm(dispo['s_min_ref'])} mm pour Ø20, d<sub>g</sub> = {C.nb(dispo['dg'])} mm (EC2 8.2(2))")
    obs = ""
    for e, titre in GROUPES_OBS:
        its = [it for it in dispo["items"] if it["etat_eff"] == e]
        if not its:
            continue
        obs += f'<div class="et-obs-g {ETATS_OBS[e][0]}">{titre}<span>{len(its)}</span></div>'
        obs += "".join(_observation(i, it, regle_smin if it["cle"] in ("lit", "entre_lits", "faces") and e == DI.NOK else "") for it in its)
    return (f'<div class="el vc et-dispo" data-dispo="{i}" title="Afficher chaque observation"><div class="l1"><b>Dispositions constructives</b>'
            f'<span class="et-mot {niv}">{txt}</span></div><div class="l2"><span class="cmp">{resume}</span>'
            f'{C.icon("chevron-down")}</div></div><div class="et-obs-l" data-dispo-l="{i}">{obs}</div>')


def _bloc(i: int, s: Dict, nom_p: str, hyps: List[str], gouv: Dict, niv_note: str) -> tuple:
    """Le bloc d'UNE section : l'en-tête (son nom, son état, coloré), les efforts, les vérifications (deux lignes, un
    clic ouvre le détail), la liste de contrôle des dispositions. Rend (html, volets)."""
    cls, etat_txt = _etat_bloc(s)
    rows, volets = "", ""
    t_s = max(v["taux"] for v in s["verifications"])
    for v in s["verifications"]:
        pk = f"{v['cle']}-{i}" if v["cle"] in ("inf", "sup") else ""
        piece = _piece_verif(v, s)
        rows += C.ligne_verif(COURTS.get(v["cle"], v["nom"]), _comparaison(v, s), C.pct(v["taux"]), _niv(v), calcul=f"s{i}-{v['cle']}",
                              pk=pk, piece=piece, gov=(s["nom"] == gouv["section"] and v["cle"] == gouv["cle"] and niv_note != "n"))
        detail = f'<div class="ck-g"><div class="ck-t">{C.esc(v["nom"])}</div>{C.lignes_fx([_formule(f) for f in v["formules"]])}</div>'
        volets += C.fenetre_calcul(f"s{i}-{v['cle']}", f"{nom_p} · section {C.esc(s['nom'])} · {C.esc(COURTS.get(v['cle'], v['nom']).lower())}",
                                   f"{_efforts_txt(s)} · méthode du bureau", hyps, [_ligne_tableau(v, s, True)], detail, _pied(CRITERES[v["cle"]]))
    for nv in s["non_verifiees"]:
        rows += C.ligne_verif(COURTS.get(nv["cle"], nv["nom"]), C.esc(nv["raison"]), None, "c", etat_txt="non effectuée")
    rows += _dispositions(i, s)
    lignes = [_ligne_tableau(v, s, v["taux"] == t_s) for v in s["verifications"]]
    detail = "".join(f'<div class="ck-g"><div class="ck-t">{C.esc(v["nom"])}</div>{C.lignes_fx([_formule(f) for f in v["formules"]])}</div>'
                     for v in s["verifications"])
    volets += C.fenetre_calcul(f"s{i}", f"{nom_p} · section {C.esc(s['nom'])}", f"{_efforts_txt(s)} · méthode du bureau", hyps, lignes, detail)
    return E.bloc(i, f"Section {s['nom']}", cls, etat_txt, _efforts_txt(s), rows), volets


def resultat(etat: Dict) -> Dict:
    p = _poutre(etat)
    r = MO.verifier(p)
    resultats = [r if i == etat["actif"] else _calculer(q) for i, q in enumerate(etat["poutres"])]
    niv_note = _niveau_note(r)
    nom_p = C.esc(p.get("nom") or "Poutre")
    hyps = _hyps(p, r)
    blocs, volets = "", ""
    for i, s in enumerate(r["sections"]):
        h, v = _bloc(i, s, nom_p, hyps, r["gouvernant"], niv_note)
        blocs += h
        volets += v
    messages = "".join(C.message(C.esc(a["message"])) for a in r.get("avertissements", []))
    tete = E.tete_resultats(nom_p, f'{C.nb(p["b"])} × {C.nb(p["h"])} cm · {C.esc(p["beton"])} · B{p["fyk"]}')
    pied = (f'<div class="pfoot"><div class="mass"><span>Méthode du bureau · moteur poutre</span><b>{C.esc(MO.REFERENCE.split(" (")[0].split()[-1])}</b></div>'
            f'<div class="legend">{C.legende()}</div></div>')
    # l'état de la NOTE (toutes ses poutres) dans l'en-tête du panneau : utile dès deux poutres ; seule, chaque section dit le sien
    statut = C.statut(_niveau_note_note(resultats)) if len(etat["poutres"]) > 1 else ""
    return {"resultat": f'<div class="pbody">{tete}<div class="et-blocs">{blocs}</div>{messages}</div>{pied}', "volets": volets,
            "vues": _vues(p, r), "phrase": "", "statut_html": statut, "resume": resume_note(etat, resultats),
            "donnees": arbre(etat, r), "elements": elements(etat, resultats)}


def _niveau_note_note(resultats: List[Dict]) -> str:
    nivs = [_niveau_note(x) for x in resultats]
    return next((n for n in ("x", "r", "n", "v") if n in nivs), max(nivs, key=lambda n: {"p": 0, "f": 1}.get(n, 0)))


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
            t += " · <b>espacement sous le minimum de l'EC2 (non conforme)</b>"
        if not c["possible"]:
            t += " · <b>ne se pose pas (barres trop serrées)</b>"
        return t
    rows = [{"nom": c["nom"], "col": C.fr(c["aire"], 0), "taux": C.pct(min(c["taux"], 9.99)), "niveau": "r" if c["etat"] == MO.NOK else C.niveau(c["taux"]),
             "cur": c["retenue"], "opt": c is opt, "prop": prop(c)} for c in cands]
    face_txt = "inférieures" if face == "inf" else "supérieures"
    return C.selecteur(role, f"Armatures {face_txt} · section {C.esc(s['nom'])}", "un lit · triées par section d'acier · un clic retient la disposition",
                       "", rows, max(0, min(i_o, i_c) - 4), "mm²", etoile="la plus petite section qui tient et se pose")

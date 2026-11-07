# -*- coding: utf-8 -*-
"""Le module « Enrobage » du site : l'état d'une note (un ou plusieurs éléments : poutre, poteau, dalle, voile), les
champs de saisie, le résultat (l'enrobage nominal retenu et sa décomposition), les fenêtres de calcul (durabilité,
feu, valeur retenue), le sélecteur de la largeur au feu et les vues (le détail coté à l'échelle, les tableaux de
l'écran) — rendus en fragments HTML.

Le calcul est celui du moteur ``beton/enrobage`` (extrait de l'écran Streamlit modules/enrobage.py, dont il reproduit
les résultats, les effets de bord et l'affichage — tests/test_enrobage_moteur.py). Ce fichier MET EN FORME : il ne
calcule rien ; il choisit parmi ce que le moteur nomme et compose les phrases. Pas de vérification dans l'écran
d'origine (aucun enrobage prévu n'y est comparé) : c'est une CONSULTATION (niveau « i »), les barres disent la part de
chaque exigence dans l'enrobage retenu.
"""
import copy
import html
import math
import re
from typing import Dict, List, Optional, Sequence

from beton.enrobage import donnees as D
from beton.enrobage import entree as EN
from beton.enrobage import moteur as M

from . import commun as C
from . import dessin_enrobage as DE

MODULE = "enrobage"
PAGE = "enrobage"

GOUV = {"durabilite": "la durabilité", "feu": "le feu"}
NOM_MAX = 80
LIBELLES = {"nom": "Nom", "type": "Type d'élément", "position": "Position", "diam_barre": "Ø max des armatures principales",
            "diam_etrier": "Ø de l'étrier ou du cadre", "delta_dev": "Tolérance Δc_dev", "classe_expo": "Classe d'exposition",
            "classe_struct": "Classe structurale", "feu": "Résistance au feu", "r": "Durée de résistance", "b_min": "Largeur minimale"}
TITRES_VUE = {"Poutre": "Détail d'angle de la poutre", "Poteau": "Détail d'angle du poteau", "Dalle": "Coupe de la dalle",
              "Voile": "Coupe horizontale du voile"}
# un état de démonstration pour le parcours : feu vérifié sur une poutre (le sélecteur de la largeur s'y ouvre)
_DEMO = {"version": 1, "actif": 0, "elements": [{**EN.nouvel_element("E1"), "feu": True, "r": "R90", "b_min": 200,
                                                 "classe_expo": "XC4", "classe_struct": "S4"}]}
PARCOURS = [
    {"action": "selecteur", "role": "feu", "etat": _DEMO},
    {"action": "choisir", "role": "feu", "nom": "b ≥ 300 mm", "etat": _DEMO},
    {"action": "modifier", "chemin": ["type"], "valeur": "Poteau", "etat": _DEMO},
    {"action": "modifier", "chemin": ["type"], "valeur": "Voile", "etat": _DEMO},
    {"action": "modifier", "chemin": ["diam_barre"], "valeur": "4"},
    {"action": "modifier", "chemin": ["diam_barre"], "valeur": "20,5"},
    {"action": "modifier", "chemin": ["diam_etrier"], "valeur": "-2"},
]


# ------------------------------------------------------------------ l'état de la note : des éléments
def _nouveau(n: int) -> Dict:
    return EN.nouvel_element(f"E{n}")


def nouvel_etat() -> Dict:
    return {"version": 1, "elements": [_nouveau(1)], "actif": 0}


def normaliser(etat: Dict) -> Dict:
    """L'état reçu de l'écran relu par le moteur ; un élément illisible (fichier abîmé) lève une erreur claire."""
    if not isinstance(etat, dict):
        raise ValueError("note d'enrobage illisible : un objet {version, elements, actif} est attendu")
    els = etat.get("elements")
    if not els:
        return nouvel_etat()
    if not isinstance(els, list):
        raise ValueError("note d'enrobage illisible : « elements » doit être une liste")
    out = {"version": 1, "elements": [EN.normaliser(e) for e in els]}
    for e in out["elements"]:
        if len(e["nom"]) > NOM_MAX:
            raise ValueError(f"note d'enrobage illisible : un nom d'élément dépasse {NOM_MAX} caractères")
    try:
        actif = int(etat.get("actif", 0) or 0)
    except (TypeError, ValueError, OverflowError):
        actif = 0
    out["actif"] = min(max(0, actif), len(out["elements"]) - 1)
    return out


def _element(etat: Dict) -> Dict:
    return etat["elements"][etat["actif"]]


def _index(index) -> int:
    """Le rang d'un élément reçu de l'écran ; un rang illisible est refusé clairement."""
    if index is None:
        return 0
    if isinstance(index, bool) or not isinstance(index, (int, float, str)):
        raise ValueError(f"rang d'élément illisible : {index!r}")
    try:
        return int(float(index))
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"rang d'élément illisible : {index!r}") from None


def gerer_elements(etat: Dict, action: str, index: Optional[int] = None) -> None:
    """Les éléments de la note : en ouvrir un, en ajouter, copier l'ouvert, en supprimer un (jamais le dernier)."""
    els = etat["elements"]
    if action == "activer":
        etat["actif"] = min(max(0, _index(index)), len(els) - 1)
    elif action == "ajouter":
        els.append(_nouveau(len(els) + 1))
        etat["actif"] = len(els) - 1
    elif action == "dupliquer":
        e = copy.deepcopy(els[etat["actif"]])
        e["nom"] = f"{e.get('nom') or 'Élément'} (copie)"
        els.append(e)
        etat["actif"] = len(els) - 1
    elif action == "supprimer" and len(els) > 1:
        i = min(max(0, _index(index)), len(els) - 1)
        del els[i]
        if etat["actif"] > i or etat["actif"] >= len(els):
            etat["actif"] = max(0, etat["actif"] - 1)


# ------------------------------------------------------------------ la saisie
def _message(cle: str, e: ValueError) -> str:
    """Le refus du moteur, dit avec le libellé du champ : « Ø max des armatures principales : au moins 6 mm … »."""
    texte = str(e)
    return f"{LIBELLES.get(cle, cle)} : {texte.split(' : ', 1)[1] if ' : ' in texte else texte}"


def modifier(etat: Dict, chemin: Sequence, valeur) -> Optional[str]:
    """Écrit une saisie dans l'élément ouvert, avec les effets de bord de l'écran Streamlit (la classe structurale
    suit l'exposition, la largeur au feu suit le type et la durée — beton/enrobage/entree.py). Rend un message si la
    saisie est refusée (l'élément reste intact) : un Ø non entier ou sous le minimum du champ, que l'écran Streamlit
    ne laisse pas saisir, est REFUSÉ ici avec son motif."""
    if not chemin or len(chemin) != 1 or chemin[0] not in EN.CLES:
        return f"Champ inconnu : « {'/'.join(map(str, chemin or []))} »"
    cle = chemin[0]
    el = _element(etat)
    if cle == "nom" and valeur is not None and len(str(valeur)) > NOM_MAX:
        return f"Nom : {NOM_MAX} caractères au plus ({len(str(valeur))} reçus)"
    if cle in ("diam_barre", "diam_etrier", "delta_dev", "b_min"):
        x = C.lire_nombre(valeur)
        if x is None:
            return f"{LIBELLES[cle]} : nombre illisible « {valeur} »"
        if not x.is_integer():
            return f"{LIBELLES[cle]} : un nombre entier de millimètres est attendu (« {valeur} » refusé)"
        v = int(x)
    elif cle == "feu":
        if not isinstance(valeur, bool):   # la case envoie vrai ou faux ; « false » (un texte) serait VRAI pour bool()
            return f"{LIBELLES[cle]} : vrai ou faux attendu (« {valeur} » refusé)"
        v = valeur
    else:
        v = "" if valeur is None else str(valeur)
    try:
        etat["elements"][etat["actif"]] = EN.appliquer(el, cle, v)
    except ValueError as e:
        return _message(cle, e)
    return None


def choisir_section(etat: Dict, role: str, nom: str) -> None:
    """Le sélecteur de la largeur au feu : « b ≥ 300 mm » devient la largeur minimale retenue."""
    if role != "feu":
        raise ValueError(f"sélecteur inconnu : {role}")
    m = re.search(r"(\d+)", str(nom))
    if not m:
        raise ValueError(f"largeur illisible : {nom!r}")
    etat["elements"][etat["actif"]] = EN.appliquer(_element(etat), "b_min", int(m.group(1)))


# ------------------------------------------------------------------ petites écritures
def _mm(x) -> str:
    """Une longueur du calcul en mm, exacte (les valeurs sont entières ou au demi-millimètre) : 30, 18,5."""
    return C.nb(x, 1)


def _s(t: str) -> str:
    """Un symbole avec indice, écrit à la main pour les HTML bruts : c_nom,dur → c<sub>nom,dur</sub>."""
    return C.symb_math(t)


def _md(texte: str) -> str:
    """Le petit markdown des textes de l'écran (gras, italique, listes à tirets, paragraphes) → HTML, échappé."""
    out = []
    for bloc in texte.split("\n\n"):
        lignes = bloc.split("\n")
        items = [ln[2:] for ln in lignes if ln.startswith("- ")]
        tete = [ln for ln in lignes if not ln.startswith("- ")]

        def fmt(s):
            s = html.escape(s, quote=False)
            s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
            return re.sub(r"\*(.+?)\*", r"<i>\1</i>", s)
        h = "".join(f"<p>{fmt(t)}</p>" for t in tete)
        if items:
            h += "<ul>" + "".join(f"<li>{fmt(i)}</li>" for i in items) + "</ul>"
        out.append(h)
    return "".join(out)


def _note(f: Dict) -> str:
    """La caption de l'écran sous « B — Feu » (l'hypothèse du tableau), espacée comme le reste de la page."""
    return re.sub(r"\s*([=≈])\s*", r" \1 ", f["note"])


def _proposee(expo: str) -> str:
    return D.CLASSE_STRUCTURALE_DEFAULT[expo]


# ------------------------------------------------------------------ le cadre Données
def formulaire(etat: Dict, res: Dict) -> str:
    el = _element(etat)
    L = C.ligne
    expo, struct = el["classe_expo"], el["classe_struct"]
    idx = D.EXPO_CLASSES.index(expo)
    x = res["exposition"]
    # 1 · l'élément
    s1 = (L("Nom", C.texte(["nom"], el["nom"], "ex. E1, poutre V3"), full=True)
          + L("Type", C.segments(["type"], el["type"], [(t, t) for t in D.TYPES]), full=True)
          + L("Position", C.segments(["position"], el["position"], [(p, p) for p in D.POSITIONS]), full=True)
          + C.aide("La position n'entre dans aucun calcul (comme dans l'écran Streamlit) : elle aide à choisir la classe d'exposition."))
    # 2 · les armatures et la tolérance d'exécution (l'écran les groupe dans ses « Paramètres généraux » : Δc_dev
    # s'ajoute aux DEUX enrobages, durabilité et feu)
    s2 = (L("barre principale", C.num(["diam_barre"], el["diam_barre"], "mm", mini=D.DIAM_BARRE_MIN, maxi=D.DIAM_BARRE_MAX), "Ø",
            f"Le plus gros Ø des armatures principales (« Ø max » de l'écran Streamlit), en mm entiers, de {D.DIAM_BARRE_MIN} à "
            f"{D.DIAM_BARRE_MAX}. c_min,b = Ø (barres individuelles, EN 1992-1-1 tab. 4.2) ; Ø/2 se retire de la distance à l'axe au feu.")
          + L("étrier / cadre", C.num(["diam_etrier"], el["diam_etrier"], "mm", mini=D.DIAM_ETRIER_MIN, maxi=D.DIAM_ETRIER_MAX), "Ø<sub>e</sub>",
              D.AIDE_ETRIER + f" 0 : pas d'étrier (dalle, voile sans barre extérieure) ; {D.DIAM_ETRIER_MAX} mm au plus.")
          + '<div class="grp">Tolérance d\'exécution</div>'
          + L("Δc<sub>dev</sub>", C.segments(["delta_dev"], el["delta_dev"], [(v, f"{v} mm") for v in D.DELTAS_DEV]), full=True,
              aide=D.AIDE_DELTA_DEV + " Elle s'ajoute à l'enrobage de durabilité ET à celui du feu."))
    # 3 · la durabilité
    s3 = ('<div class="grp">Classe d\'exposition</div>'
          + L("Classe", C.choix(["classe_expo"], expo, [(k, f"{k} — {D.EXPOSITION_INFO[k][0]}") for k in D.EXPO_CLASSES]), full=True)
          + C.aide(f"<b>{C.esc(expo)} — {C.esc(x['titre'])}</b><br><b>Mécanisme</b> : {C.esc(x['description'])}<br>"
                   f"<b>Quand la choisir</b> : {C.esc(x['quand'])}<br><i>Exemples</i> : {C.esc(x['exemples'])}")
          + '<div class="grp">Classe structurale</div>'
          + L("Classe", C.choix(["classe_struct"], struct,
                                            [(k, f"{k} — {D.C_MIN_DUR[k][idx]} mm" + (" (proposée)" if k == _proposee(expo) else ""))
                                             for k in D.CLASSES_STRUCTURALES]), full=True,
              aide=D.AIDE_CLASSE_STRUCT + " Le menu donne c_min,dur pour l'exposition choisie.")
          + C.aide(f"Classe proposée pour {C.esc(expo)} : <b>{_proposee(expo)}</b> — elle est reprise quand l'exposition change et "
                   "que la classe proposée change (comme dans l'écran Streamlit)."))
    # 4 · le feu
    s4 = C.case(["feu"], el["feu"], C.esc(D.LIBELLE_FEU))
    f = res["feu"]
    if el["feu"]:
        s4 += L("Durée", C.choix(["r"], el["r"], [(r, r) for r in D.R_LEVELS]), full=True)
        vs = EN.variantes(el["type"], el["r"])
        if vs:
            s4 += L("Largeur", C.choix(["b_min"], el["b_min"], [(b, f"b ≥ {b} mm → a = {a} mm") for b, a in vs]), full=True,
                    aide=D.AIDE_LARGEUR)
        # la caption de l'écran sous « B — Feu » : l'hypothèse du tableau (appuis simples, une face, une direction…)
        s4 += C.aide(f"<b>{_s(_note(f))}</b><br>{_s(D.CAPTION_FEU)}")
    resume4 = f"{el['r']} · tab. {f['tableau']}" + (f" · b ≥ {f['b_min']}" if f["b_min"] else "") if f else "non vérifiée"
    # 5 · le guide de l'écran
    s5 = (f'<div class="en-guide">{_md(D.GUIDE_INTRO)}{_md(D.GUIDE_CLASSES)}'
          f'<p class="en-regle"><b>{C.esc(D.GUIDE_REGLE[0])}</b> : {C.esc(D.GUIDE_REGLE[1])}</p>{_md(D.GUIDE_PLUSIEURS)}</div>')
    return (C.section(1, f"Élément {C.esc(el['nom'])}", f"{el['type']} · {el['position'].lower()}", s1, True, "el")
            + C.section(2, "Armatures et tolérance", f"Ø{el['diam_barre']} · étrier Ø{el['diam_etrier']} · Δc<sub>dev</sub> {el['delta_dev']} mm",
                        s2, True, "ar")
            + C.section(3, "Durabilité", f"{C.esc(expo)} · {struct}", s3, True, "du")
            + C.section(4, "Résistance au feu", resume4, s4, True, "fe")
            + C.section(5, "Guide · classe d'exposition", "comment la choisir", s5, False, "gu"))


# ------------------------------------------------------------------ le résultat
def _part(x, res) -> float:
    """La part d'une exigence dans l'enrobage retenu (la barre et le pourcentage des lignes)."""
    return x / res["c_nom"] if res["c_nom"] else 0.0


def _pc(t: float) -> int:
    """Une part en pour cent, arrondie au plus proche : les barres sont neutres (une consultation), l'arrondi de C.pct
    qui garde la bande de couleur d'un taux (94,9 % → 94) n'a pas lieu d'être ici."""
    return int(math.floor(min(t, 9.99) * 100 + 0.5))


def _lignes_dur(res: Dict) -> List[Dict]:
    d = res["durabilite"]
    # « Limite » porte ici la RÉFÉRENCE de la norme (une consultation n'a pas de limite), « Formule » le calcul
    ref = {"c_min_b": "EC2 tab. 4.2", "c_min_dur": "EC2 tab. 4.4N", "plancher": "EC2 éq. (4.2)", "c_min": "EC2 éq. (4.2)",
           "delta_dev": "EC2 §4.4.1.3", "c_nom_dur": "EC2 éq. (4.1)"}
    rows = [("c_min_b", "c<sub>min,b</sub>", "adhérence", f"Ø de la barre principale = {_mm(d['c_min_b'])}", d["c_min_b"]),
            ("c_min_dur", "c<sub>min,dur</sub>", f"durabilité · {d['classe_structurale']} / {C.esc(d['classe_exposition'])}",
             f"classe {d['classe_structurale']} × exposition {C.esc(d['classe_exposition'])}", d["c_min_dur"]),
            ("plancher", "10 mm", "minimum absolu", "c<sub>min</sub> ≥ 10 mm", d["plancher"]),
            ("c_min", "c<sub>min</sub>", "enrobage minimal", "max(c<sub>min,b</sub> ; c<sub>min,dur</sub> + Δc<sub>dur</sub> ; 10)", d["c_min"]),
            ("delta_dev", "Δc<sub>dev</sub>", "tolérance d'exécution", "tolérance choisie (10, 5 ou 0 mm)", d["delta_dev"]),
            ("c_nom_dur", "c<sub>nom,dur</sub>", "nominal · durabilité", "c<sub>min</sub> + Δc<sub>dev</sub>", d["c_nom"])]
    out = []
    for cle, sym, sous, formule, v in rows:
        out.append({"nom": sym, "sous": sous, "formule": formule, "valeur": f"{_mm(v)} mm", "limite": ref[cle],
                    "taux": _pc(_part(v, res)), "niveau": "c",
                    "gov": cle == "c_nom_dur" and res["gouvernant"] == "durabilite" or cle == d["terme_c_min"]})
    return out


def _lignes_feu(res: Dict) -> List[Dict]:
    f = res["feu"]
    rows = [("a", "a", f"distance à l'axe · {f['r']}", f"lue pour {f['r']}" + (f" · b ≥ {f['b_min']} mm" if f["b_min"] else ""), f["a"],
             f"EC2-1-2 tab. {f['tableau']}"),
            ("c_feu_min", "c<sub>feu,min</sub>", "au nu de l'étrier", "max(a − Ø/2 − Ø<sub>e</sub> ; 0)", f["c_min"], "convention de l'écran"),
            ("c_nom_feu", "c<sub>nom,feu</sub>", "nominal · feu", "c<sub>feu,min</sub> + Δc<sub>dev</sub>", f["c_nom"], "convention de l'écran")]
    return [{"nom": sym, "sous": sous, "formule": fo, "valeur": f"{_mm(v)} mm", "limite": ref, "taux": _pc(_part(v, res)),
             "niveau": "c", "gov": cle == "c_nom_feu" and res["gouvernant"] == "feu"} for cle, sym, sous, fo, v, ref in rows]


def _fx_dur(res: Dict) -> str:
    d = res["durabilite"]
    lignes = [{"parts": ["c_min,dur", f"tab. 4.4N [{d['classe_structurale']} ; {d['classe_exposition']}]"], "res": f"{_mm(d['c_min_dur'])} mm"},
              {"parts": ["c_min,b", "Ø"], "res": f"{_mm(d['c_min_b'])} mm"},
              {"parts": ["c_min", "max(c_min,b ; c_min,dur + Δc_dur ; 10)",
                         f"max({_mm(d['c_min_b'])} ; {_mm(d['c_min_dur'])} + 0 ; 10)"], "res": f"{_mm(d['c_min'])} mm"},
              {"parts": ["c_nom,dur", "c_min + Δc_dev", f"{_mm(d['c_min'])} + {_mm(d['delta_dev'])}"], "res": f"{_mm(d['c_nom'])} mm"}]
    return f'<div class="ck-g"><div class="ck-t">Durabilité · EN 1992-1-1 §4.4.1</div>{C.lignes_fx(lignes)}</div>'


def _fx_feu(res: Dict) -> str:
    f, el = res["feu"], res["element"]
    var = f" ; b ≥ {f['b_min']}" if f["b_min"] else ""
    lignes = [{"parts": ["a", f"tab. {f['tableau']} [{f['r']}{var}]"], "res": f"{_mm(f['a'])} mm"},
              {"parts": ["c_feu,min", "max(a − Ø/2 − Ø étrier ; 0)",
                         f"max({_mm(f['a'])} − {_mm(f['demi_barre'])} − {_mm(el['diam_etrier'])} ; 0)"], "res": f"{_mm(f['c_min'])} mm"},
              {"parts": ["c_nom,feu", "c_feu,min + Δc_dev", f"{_mm(f['c_min'])} + {_mm(f['delta_dev'])}"], "res": f"{_mm(f['c_nom'])} mm"}]
    return (f'<div class="ck-g"><div class="ck-t">Feu · EN 1992-1-2 {f["section"]}, méthode tabulée</div>{C.lignes_fx(lignes)}'
            f'<div class="hint">{_s(_note(f))}</div></div>')


def _fx_retenu(res: Dict) -> str:
    if res["feu"] is None:
        lignes = [{"parts": ["c_nom", "c_nom,dur"], "res": f"{_mm(res['c_nom'])} mm"}]
        pied = '<div class="hint">Feu non vérifié : l\'enrobage retenu est celui de la durabilité.</div>'
    else:
        lignes = [{"parts": ["c_nom", "max(c_nom,dur ; c_nom,feu)", f"max({_mm(res['c_nom_dur'])} ; {_mm(res['c_nom_feu'])})"],
                   "res": f"{_mm(res['c_nom'])} mm"}]
        pied = f'<div class="hint">Gouverné par {GOUV[res["gouvernant"]]} (à égalité, la durabilité, comme l\'écran).</div>'
    return f'<div class="ck-g"><div class="ck-t">Enrobage nominal retenu</div>{C.lignes_fx(lignes)}{pied}</div>'


def _regles(res: Dict, cles: Sequence[str]) -> str:
    """Les règles des termes (nommées par le moteur), en pied de fenêtre."""
    items = "".join(f"<li><b>{_s(t['symbole'])}</b> — {_s(t['regle'])}</li>" for t in res["termes"] if t["cle"] in cles)
    if "c_min" in cles:
        items += f"<li><b>Δc<sub>dur</sub></b> — {_s(M.REGLES['delta_dur'])}</li>"
    items = items.replace("c<sub>dur</sub>,γ", "c<sub>dur,γ</sub>")  # γ n'est pas un indice pour symb_math
    return f'<div class="ck-sub">Les règles</div><ul class="en-regles">{items}</ul>'


def _hyps(res: Dict) -> List[str]:
    el, d = res["element"], res["durabilite"]
    prop = d["classe_structurale_proposee"]
    h = [f"{C.esc(el['type'])} · {C.esc(el['position'].lower())}",
         f"Exposition <b>{C.esc(d['classe_exposition'])}</b> · classe structurale <b>{d['classe_structurale']}</b>"
         + (" (proposée)" if prop == d["classe_structurale"] else f" (proposée : {prop})"),
         f"Ø <b>{el['diam_barre']} mm</b> · étrier Ø{el['diam_etrier']} mm",
         f"Δc<sub>dev</sub> = <b>{el['delta_dev']} mm</b> · Δc<sub>dur,γ</sub> = Δc<sub>dur,st</sub> = Δc<sub>dur,add</sub> = 0"]
    if res["feu"]:
        f = res["feu"]
        # μ_fi (taux d'exploitation au feu) n'entre que dans les tableaux des poteaux et des voiles
        h.append(f"Feu <b>{f['r']}</b> · tab. {f['tableau']}" + (f" · b ≥ {f['b_min']} mm" if f["b_min"] else "")
                 + (" · μ<sub>fi</sub> ≈ 0,7" if el["type"] in ("Poteau", "Voile") else ""))
    return h


def _fenetres(res: Dict, nom: str) -> str:
    hyps = _hyps(res)
    n = C.esc(nom)
    out = C.fenetre_calcul("dur", f"{n} · durabilité", "EN 1992-1-1 §4.4.1 · taux : part de l'enrobage retenu", hyps, _lignes_dur(res),
                           _fx_dur(res), _regles(res, ("c_min_b", "c_min_dur", "plancher", "c_min", "delta_dev", "c_nom_dur")))
    if res["feu"]:
        out += C.fenetre_calcul("feu", f"{n} · feu {res['feu']['r']}", f"EN 1992-1-2 {res['feu']['section']} · méthode tabulée", hyps,
                                _lignes_feu(res), _fx_feu(res), _regles(res, ("a", "c_feu_min", "c_nom_feu")))
    lignes = [li for li in _lignes_dur(res) if li["nom"] == "c<sub>nom,dur</sub>"]
    if res["feu"]:
        lignes += [li for li in _lignes_feu(res) if li["nom"] == "c<sub>nom,feu</sub>"]
    lignes.append({"nom": "c<sub>nom</sub>", "sous": f"retenu · {GOUV[res['gouvernant']]}", "formule": "max(c<sub>nom,dur</sub> ; c<sub>nom,feu</sub>)",
                   "valeur": f"{_mm(res['c_nom'])} mm", "limite": "le plus grand", "taux": 100 if res["c_nom"] else 0, "niveau": "c", "gov": True})
    detail = _fx_retenu(res) + _fx_dur(res) + (_fx_feu(res) if res["feu"] else "")
    out += C.fenetre_calcul("retenu", f"{n} · enrobage nominal retenu", "max(durabilité EN 1992-1-1 ; feu EN 1992-1-2) · comme l'écran Streamlit",
                            hyps, lignes,
                            detail, _regles(res, ("c_nom",)))
    return out


def _synthese(res: Dict) -> str:
    """La décomposition sous le verdict : les termes de c_min (le gouvernant marqué), puis la tolérance."""
    d = res["durabilite"]
    cases = [("c_min_b", f"c<sub>min,b</sub> · Ø barre", d["c_min_b"]),
             ("c_min_dur", f"c<sub>min,dur</sub> · {d['classe_structurale']} / {C.esc(d['classe_exposition'])}", d["c_min_dur"]),
             ("delta_dur", "Δc<sub>dur</sub> · γ, st, add", 0),
             ("plancher", "minimum absolu", d["plancher"]),
             ("c_min", "c<sub>min</sub> · le plus grand", d["c_min"]),
             ("delta_dev", "Δc<sub>dev</sub> · tolérance", d["delta_dev"])]
    h = "".join(f'<div class="{"g" if k == d["terme_c_min"] else ""}"><span>{t}</span><b>{_mm(v)} mm</b></div>' for k, t, v in cases)
    return f'<div class="en-dec" data-calcul="dur" title="Voir le calcul">{h}</div>'


def _verdict(res: Dict) -> str:
    d, f = res["durabilite"], res["feu"]
    if f is None:
        cmp_ = f"c<sub>min</sub> + Δc<sub>dev</sub> = {_mm(d['c_min'])} + {_mm(d['delta_dev'])} mm<br>feu non vérifié"
    else:
        cmp_ = (f"durabilité {_mm(res['c_nom_dur'])} mm · feu {_mm(res['c_nom_feu'])} mm"
                f"<br>{f['r']} · tab. {f['tableau']}" + (f" · b ≥ {f['b_min']} mm" if f["b_min"] else ""))
    return (f'<div class="verdict en-v" data-calcul="retenu" title="Voir le calcul"><span class="chip xl en-c">{_mm(res["c_nom"])}'
            f'<small>{C.NBSP}mm</small></span><div class="gov"><b>Enrobage nominal · gouverné par {GOUV[res["gouvernant"]]}</b>{cmp_}</div></div>')


def _lignes(res: Dict) -> str:
    d, f, el = res["durabilite"], res["feu"], res["element"]
    rows = C.ligne_res("Durabilité", f"{d['classe_structurale']} / {C.esc(d['classe_exposition'])} · {_mm(d['c_nom'])} mm",
                       _pc(_part(d["c_nom"], res)), "c", calcul="dur", gov=res["gouvernant"] == "durabilite")
    if f is not None:
        piece = (f"b ≥ {f['b_min']} · a {_mm(f['a'])} · {_mm(f['c_nom'])} mm" if f["b_min"] else f"a {_mm(f['a'])} · {_mm(f['c_nom'])} mm")
        leger = ""
        if f["b_min"]:
            o = next((c for c in M.candidats_feu(el) if c["optimale"]), None)
            if o and o["b_min"] != f["b_min"] and o["c_nom"] < res["c_nom"]:
                leger = f"À partir de b ≥ {o['b_min']} mm, l'enrobage retenu descend à {C.nb(o['c_nom'], 1)} mm"
        rows += C.ligne_res(f"Feu · {f['r']}", piece, _pc(_part(f["c_nom"], res)), "c", calcul="feu",
                            pk="feu" if f["b_min"] else "", gov=res["gouvernant"] == "feu", leger=leger)
    return rows


def _messages(res: Dict) -> str:
    out = ""
    if res["feu"] is None:
        out += C.message("Feu non vérifié : cochez « Vérifier la résistance au feu » (section 4) pour l'inclure.")
    for a in res["avertissements"]:
        out += C.message(_s(a["message"]), a["grave"])
    return out


def resume_note(etat: Dict, resultats: List[Dict]) -> Dict:
    """Le résumé de la NOTE : une consultation (pas de verdict) ; l'élément au plus grand enrobage est nommé."""
    i = max(range(len(resultats)), key=lambda k: resultats[k]["c_nom"])
    r = resultats[i]
    nom = etat["elements"][i].get("nom") or f"Élément {i + 1}"
    return {"taux": None, "niveau": "i", "statut": C.STATUTS["i"][1],
            "gouvernant": f"{nom} · c_nom = {C.nb(r['c_nom'], 1)} mm ({GOUV[r['gouvernant']]})"}


def elements(etat: Dict, resultats: List[Dict]) -> str:
    items = ""
    for i, (e, r) in enumerate(zip(etat["elements"], resultats)):
        sup = f'<button class="x" data-supprimer="{i}" title="Supprimer">{C.icon("x")}</button>' if len(etat["elements"]) > 1 else ""
        nom = e.get("nom") or f"Élément {i + 1}"
        feu = f" · {e['r']}" if e["feu"] else ""
        # le type, sauf quand le nom le dit déjà (« Dalle parking ») : la ligne tient dans la colonne d'une tablette
        typ = "" if e["type"].lower() in nom.lower() else f"{C.esc(e['type'])} · "
        items += (f'<div class="ni{" on" if i == etat["actif"] else ""}" data-activer="{i}"><span class="t">{C.esc(nom)} '
                  f'<span>· {typ}{C.esc(e["classe_expo"])}{feu}</span></span><span class="pct">{_mm(r["c_nom"])}{C.NBSP}mm</span>{sup}</div>')
    return (f'<div class="elements"><div class="nl-h"><span class="kicker">Dans cette note · {len(etat["elements"])}</span>'
            f'<button class="nl-add" data-dupliquer="1" title="Copier l\'élément ouvert">{C.icon("copy")}Copier</button></div>{items}'
            f'<div class="nl-add" data-ajouter="1">{C.icon("plus")}Ajouter un élément</div></div>')


# ------------------------------------------------------------------ la vue « Tableaux »
def vue_tableaux(res: Dict) -> str:
    """Les tableaux de l'écran, la case retenue marquée : c_min,dur (classe structurale × exposition) et, pour le type
    de l'élément, les distances à l'axe au feu (durée × largeur)."""
    d, el = res["durabilite"], res["element"]
    tete = "".join(f"<th>{C.esc(x)}</th>" for x in D.EXPO_CLASSES)
    corps = ""
    for s in D.CLASSES_STRUCTURALES:
        cel = ""
        for j, x in enumerate(D.EXPO_CLASSES):
            cls = " ".join(c for c in ("on" if (s, x) == (d["classe_structurale"], d["classe_exposition"]) else "",
                                       "col" if x == d["classe_exposition"] else "", "lig" if s == d["classe_structurale"] else "") if c)
            cel += f'<td class="{cls}">{D.C_MIN_DUR[s][j]}</td>'
        prop = " · proposée" if s == d["classe_structurale_proposee"] else ""
        corps += f'<tr><th>{s}<small>{prop}</small></th>{cel}</tr>'
    t1 = (f'<div class="en-bloc"><div class="en-tt">c<sub>min,dur</sub> (mm) · tableau de l\'écran (EN 1992-1-1 tab. 4.4N)</div>'
          f'<table class="en-tab"><thead><tr><th>Classe</th>{tete}</tr></thead><tbody>{corps}</tbody></table></div>')
    t = el["type"]
    f = res["feu"]
    if t in D.TYPES_VARIANTES:
        tab = D.FEU_POUTRE if t == "Poutre" else D.FEU_POTEAU
        n = max(len(v) for v in tab.values())
        lignes = ""
        for r in D.R_LEVELS:
            cel = ""
            for k in range(n):
                if k < len(tab[r]):
                    b, a = tab[r][k]
                    on = f is not None and r == f["r"] and b == f["b_min"]
                    cel += f'<td class="{"on" if on else ""}"><small>b ≥ {b}</small>{a}</td>'
                else:
                    cel += "<td></td>"
            lignes += f'<tr class="{"lig" if f is not None and r == f["r"] else ""}"><th>{r}</th>{cel}</tr>'
        t2 = (f'<div class="en-bloc"><div class="en-tt">a (mm) · {t.lower()} · tableau {M.TABLEAUX_FEU[t]} de l\'écran (variantes de largeur b, mm)</div>'
              f'<table class="en-tab feu"><thead><tr><th>Durée</th>{"".join(f"<th>variante {k + 1}</th>" for k in range(n))}</tr></thead>'
              f'<tbody>{lignes}</tbody></table></div>')
    else:
        tab = D.FEU_DALLE if t == "Dalle" else D.FEU_VOILE
        cel = "".join(f'<td class="{"on" if f is not None and r == f["r"] else ""}">{tab[r]}</td>' for r in D.R_LEVELS)
        t2 = (f'<div class="en-bloc"><div class="en-tt">a (mm) · {t.lower()} · tableau {M.TABLEAUX_FEU[t]} de l\'écran</div>'
              f'<table class="en-tab feu"><thead><tr><th>Durée</th>{"".join(f"<th>{r}</th>" for r in D.R_LEVELS)}</tr></thead>'
              f'<tbody><tr><th>a</th>{cel}</tr></tbody></table></div>')
    pied = ("Valeurs reprises telles quelles de l'écran Streamlit (« alignées ANB — à valider par l'ingénieur ») ; leurs écarts "
            "avec les tableaux de la norme sont relevés dans la documentation du moteur.")
    return f'<div class="en-vue">{t1}{t2}<div class="en-pied">{pied}</div></div>'


# ------------------------------------------------------------------ la réponse
def resultat(etat: Dict) -> Dict:
    el = _element(etat)
    res = M.calculer(el)
    resultats = [res if i == etat["actif"] else M.calculer(x) for i, x in enumerate(etat["elements"])]
    nom = el.get("nom") or f"Élément {etat['actif'] + 1}"
    pied = (f'<div class="pfoot"><div class="mass"><span>EN 1992-1-1 §4.4.1 · EN 1992-1-2 §5</span><b>écran v2.0</b></div>'
            f'<div class="legend en-leg"><span>Barres : part de c<sub>nom</sub> retenu · valeurs ANB de l\'écran, à valider par l\'ingénieur</span></div></div>')
    corps = f'<div class="pbody">{_verdict(res)}{_synthese(res)}<div class="els">{_lignes(res)}</div>{_messages(res)}</div>{pied}'
    f = res["feu"]
    phrase = (f"{C.esc(el['type'])} · {C.esc(el['classe_expo'])} / {el['classe_struct']} · Ø{el['diam_barre']} · étrier Ø{el['diam_etrier']}"
              f" · Δc<sub>dev</sub> {el['delta_dev']} mm" + (f" · {f['r']}" if f else ""))
    vues = [{"cle": "coupe", "titre": TITRES_VUE[el["type"]], "court": "Coupe", "svg": DE.coupe("en_coupe", res),
             "legende": f"c_nom = {C.nb(res['c_nom'], 1)} mm"},
            {"cle": "tableaux", "titre": "Tableaux de l'écran", "court": "Tableaux", "svg": vue_tableaux(res),
             "legende": "c_min,dur et distances à l'axe au feu · la case retenue marquée"}]
    return {"resultat": corps, "volets": _fenetres(res, nom), "vues": vues, "phrase": phrase, "statut_html": C.statut("i"),
            "resume": resume_note(etat, resultats), "donnees": formulaire(etat, res), "elements": elements(etat, resultats)}


# ------------------------------------------------------------------ le sélecteur de la largeur au feu
def selecteur(etat: Dict, role: str, familles=None, froid: bool = True) -> str:
    """Les variantes de largeur du tableau feu (poutre, poteau), triées par largeur, chacune recalculée par le moteur :
    la retenue (✓) et l'optimale (★ : la plus petite largeur qui donne le plus petit enrobage retenu). Le pourcentage
    est c_nom,feu / c_nom,dur : au-delà de 100 %, le feu gouverne."""
    if role != "feu":
        raise ValueError(f"sélecteur inconnu : {role}")
    el = _element(etat)
    cands = M.candidats_feu(el)
    dur = M.calculer({**el, "feu": False})["c_nom"]
    i_o = next((k for k, c in enumerate(cands) if c["optimale"]), 0)
    i_c = next((k for k, c in enumerate(cands) if c["retenue"]), i_o)
    rows = [{"nom": f"b ≥ {c['b_min']} mm", "col": f"a = {c['a']}", "taux": _pc(c["c_nom_feu"] / dur), "niveau": "c",
             "cur": c["retenue"], "opt": c["optimale"],
             "prop": (f"b ≥ {c['b_min']} mm · a = {c['a']} mm · c<sub>feu,min</sub> = {_mm(c['c_min_feu'])} mm · c<sub>nom,feu</sub> = "
                      f"{_mm(c['c_nom_feu'])} mm · retenu {_mm(c['c_nom'])} mm ({GOUV[c['gouvernant']]})")} for c in cands]
    f = M.TABLEAUX_FEU[el["type"]]
    return C.selecteur("feu", f"Largeur minimale · {C.esc(el['type'].lower())} {el['r']} (tab. {f})",
                       "triées par largeur · % = c<sub>nom,feu</sub> / c<sub>nom,dur</sub> : au-delà de 100 %, le feu gouverne",
                       "", rows, max(0, min(i_o, i_c) - 4), "a (mm)", etoile="la plus petite largeur au plus petit enrobage")

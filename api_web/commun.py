# -*- coding: utf-8 -*-
"""Ce que les modules du site partagent : le code couleur, les nombres à la française et les COMPOSANTS de l'écran
(ligne de résultat, grande fenêtre de calcul, sélecteur, champs de saisie, sections repliables).

Ils sont écrits ici, en Python, et tournent dans le navigateur avec le moteur (Pyodide) : l'écran ne fait qu'afficher
les fragments HTML qu'on lui rend et renvoyer ce que l'utilisateur saisit. Il ne calcule rien, ne recopie aucun seuil.
Les classes CSS sont celles de la charte (site/web/css/charte.css), reprise des maquettes validées par le bureau.
"""
import html
import math
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from acier.garde_corps import membres as ME

from .icones import ICONES

NBSP = " "
NIVEAUX = {ME.NOK: "r", ME.ATTENTION: "f", ME.OK: "p"}


# ------------------------------------------------------------------ code couleur et nombres
def niveau(t: Optional[float]) -> str:
    """Le code couleur du bureau : « r » au-dessus de 100 %, « f » (vert foncé) de 95 à 100 %, « p » (vert pâle) en
    dessous ; c'est le statut du moteur qui tranche (membres.statut, seuils en source unique)."""
    if t is None:
        return "c"
    return NIVEAUX.get(ME.statut(t), "p")


def pct(t: float, nd: int = 0):
    """Un taux à afficher, arrondi au plus proche SANS quitter la bande de sa couleur : 94,98 % s'affiche 94 % (95 % se
    lirait vert foncé), 100,2 % s'affiche 101 % (100 % se lirait admis), 63,7 % reste 64 %."""
    x, n, k = t * 100, niveau(t), 10 ** nd
    v = math.floor(x * k + 0.5) / k
    if n == "r":
        v = max(v, 100 + 1 / k)
    elif n == "f":
        v = min(max(v, 95.0), 100.0)
    else:
        v = min(v, 95 - 1 / k)
    return int(v) if nd == 0 else round(v, nd)


def pc(v) -> str:
    """Un taux déjà arrondi (pct), écrit à la française : « 64 » ou « 63,7 »."""
    return f"{v:.1f}".replace(".", ",") if isinstance(v, float) and v != int(v) else f"{int(v)}"


def fr(x: Optional[float], nd: int = 1) -> str:
    """Un nombre à la française : virgule décimale, espace insécable fine entre les milliers."""
    if x is None:
        return "—"
    return f"{x:,.{nd}f}".replace(",", " ").replace(".", ",")


def nb(x: Optional[float], dec: int = 3) -> str:
    """Une valeur saisie (coefficient, dimension) écrite telle quelle, sans arrondi au-delà de ``dec`` décimales :
    1,25 reste 1,25 (un format de taux en ferait « 1,2 »). Le chiffre écrit est relu et comparé à la valeur."""
    if x is None:
        return ""
    s = f"{float(x):.{dec}f}".rstrip("0").rstrip(".")
    assert abs(float(s) - float(x)) <= 0.5 * 10 ** -dec + 1e-12, (x, s)
    return s.replace(".", ",")


def lire_nombre(v) -> Optional[float]:
    """Ce que l'utilisateur a tapé (« 1 100 », « 0,5 », 12) → un nombre ; None si ce n'en est pas un."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    if not isinstance(v, str):
        return None
    s = v.replace(" ", "").replace("\xa0", "").replace(" ", "").replace(",", ".")
    try:
        x = float(s)
    except ValueError:
        return None
    return x if math.isfinite(x) else None


# ------------------------------------------------------------------ texte
def esc(s) -> str:
    return html.escape(str(s), quote=True)


def symb_math(s: str) -> str:
    """Un symbole du moteur écrit comme dans la note : q_h → q<sub>h</sub>, W_el,y → W<sub>el,y</sub> (texte échappé)."""
    return re.sub(r"([A-Za-zσητγαμ])_([A-Za-z0-9éèàùστη]+(?:,[A-Za-z0-9éèàù]+)*)", r"\1<sub>\2</sub>", html.escape(str(s), quote=False))


def icon(nom: str) -> str:
    return ICONES[nom]


def chip(taux, niv: str, xl: bool = False) -> str:
    if xl:
        return f'<span class="chip xl {niv}"><small>η</small>{pc(taux)}{NBSP}%</span>'
    return f'<span class="chip {niv}">{pc(taux)}{NBSP}%</span>'


STATUTS = {"p": ("p", "Conforme"), "f": ("f", "Conforme, attention"), "r": ("r", "Non conforme"), "c": ("c", "À compléter"),
           "x": ("r", "Calcul impossible"),
           "i": ("c", "Consultation")}


def statut(niv: str) -> str:
    """La pastille d'état de la note : conforme, attention (95 à 100 %), non conforme, à compléter."""
    cls, txt = STATUTS[niv]
    return f'<span class="status {cls}"><span class="dot"></span>{txt}</span>'


def legende() -> str:
    return '<span class="chip r">&gt;{0}100{0}%</span><span class="chip f">95–100{0}%</span><span class="chip p">&lt;{0}95{0}%</span>'.format(NBSP)


# ------------------------------------------------------------------ résultat, calcul, sélecteur
def ligne_res(nom: str, piece: str, taux, niv: str, calcul: str = "", pk: str = "", gov: bool = False, leger: str = "") -> str:
    """Une vérification du résultat, sur DEUX lignes : le nom et la pièce, puis la jauge et le taux. La ligne entière
    ouvre le calcul détaillé ; la pièce, quand elle se choisit (profilé, armatures), ouvre le sélecteur. ``taux`` est
    déjà arrondi (pct) ; None : à compléter. ``piece`` est déjà du HTML."""
    etoile = f'<span class="leger" title="{esc(leger)}">{icon("star")}</span>' if leger else ""
    if pk:
        p = f'<button class="prof" data-pk="{esc(pk)}" title="Choisir">{etoile}<span>{piece}</span>{icon("chevron-down")}</button>'
    else:
        p = f'<span class="prof"><span>{piece}</span></span>'
    ouvre = f' data-calcul="{esc(calcul)}" title="Voir le calcul"' if calcul else ""
    txt = f"{pc(taux)}{NBSP}%" if taux is not None else "à compléter"
    largeur = min(taux or 0, 100)
    return (f'<div class="el{" gov" if gov else ""}"{ouvre}><div class="l1"><b>{esc(nom)}</b>{p}</div>'
            f'<div class="l2"><div class="bar"><b class="{niv}" style="width:{largeur}%"></b></div><span class="pct {niv}">{txt}</span></div></div>')


def lignes_fx(lignes: List[Dict]) -> str:
    """Les lignes « symbole = formule = valeurs = résultat » d'un groupe (les taux en pastille de couleur)."""
    out = ""
    for li in lignes:
        parts = li["parts"]
        h = f'<span class="s">{symb_math(parts[0])}</span>' + "".join(f'<span class="eq">=</span><span>{symb_math(p)}</span>' for p in parts[1:])
        res = f'<span class="chip {li["niveau"]}">{esc(li["res"])}</span>' if li.get("niveau") else f'<span class="r">{esc(li["res"])}</span>'
        out += f'<div class="fx">{h}<span class="eq">=</span>{res}</div>'
    return out


def fenetre_calcul(ident: str, titre: str, sous_titre: str, hyps: Sequence[str], lignes: List[Dict], detail: str, pied: str = "") -> str:
    """La grande fenêtre de calcul, LA MÊME pour tous les modules : les hypothèses, un tableau (vérification, formule,
    valeur, limite, taux), puis le détail des formules sur deux colonnes."""
    hyp = "".join(f"<span>{h}</span>" for h in hyps)
    rows = "".join(
        f'<tr class="{"gov" if li.get("gov") else ""}"><td class="nm">{li["nom"]}{("<span>" + li["sous"] + "</span>") if li.get("sous") else ""}</td>'
        f'<td class="fo">{li["formule"]}</td><td class="n v">{li["valeur"]}</td><td class="n">{li["limite"]}</td>'
        f'<td class="n">{chip(li["taux"], li["niveau"])}</td></tr>' for li in lignes)
    table = (f'<table class="vtab"><thead><tr><th>Vérification</th><th class="fo">Formule</th><th class="n">Valeur</th><th class="n">Limite</th>'
             f'<th class="n">Taux</th></tr></thead><tbody>{rows}</tbody></table>')
    return (f'<div class="flyout calcul" id="ck-{esc(ident)}"><div class="fl-h"><div><b>{titre}</b><span>{sous_titre}</span></div>'
            f'<button class="pcoll" data-fermer="1" title="Fermer">{icon("x")}</button></div><div class="ck-hyp">{hyp}</div>'
            f'<div class="ck-body">{table}<div class="ck-sub">Le détail du calcul</div><div class="ck-cols">{detail}</div>{pied}</div></div>')


def selecteur(ident: str, titre: str, sous_titre: str, filtres: str, rows: List[Dict], debut: int, unite: str,
              etoile: str = "le plus léger qui tient") -> str:
    """Le sélecteur, LE MÊME pour les profilés et les armatures : trié par ordre de grandeur, le retenu (✓) et l'optimal
    (★, le plus léger qui tient) marqués ; un clic choisit (le calcul est relancé) ; sous la liste, les propriétés de la
    ligne survolée. ``rows`` : nom, col (la colonne d'ordre de grandeur), taux (arrondi), niveau, cur, opt, prop."""
    lignes = "".join(
        f'<button class="pk-row{" cur" if r["cur"] else ""}{" opt" if r["opt"] else ""}" data-choix="{esc(r["nom"])}" data-prop="{esc(r.get("prop", ""))}">'
        f'<span class="mk">{icon("check") if r["cur"] else (icon("star") if r["opt"] else "")}</span>'
        f'<span class="nm">{esc(r["nom"])}</span><span class="kg">{r["col"]}</span>{chip(r["taux"], r["niveau"])}</button>' for r in rows)
    prop0 = next((r.get("prop", "") for r in rows if r["cur"]), "")
    vide = "" if rows else '<div class="pk-vide">Aucune section dans cette sélection.</div>'
    return (f'<div class="flyout picker" id="pk-{esc(ident)}" data-pk="{esc(ident)}" data-debut="{debut}"><div class="fl-h"><div><b>{titre}</b><span>{sous_titre}</span></div>'
            f'<button class="pcoll" data-fermer="1" title="Fermer">{icon("x")}</button></div><div class="filters">{filtres}</div>'
            f'<div class="pk-list">{lignes}{vide}</div><div class="pk-prop">{prop0}</div>'
            f'<div class="fl-foot"><span>{icon("check")}retenu</span><span>{icon("star")}{etoile}</span><span>{unite} · taux</span></div></div>')


# ------------------------------------------------------------------ saisie
def _attr_chemin(chemin: Sequence) -> str:
    return esc("/".join(str(c) for c in chemin))


def ligne(label: str, champ: str, sym: str = "", aide: str = "", full: bool = False) -> str:
    """Une ligne de saisie : le symbole, le libellé (l'aide au survol de l'icône), le champ."""
    s = f'<span class="sym">{sym}</span>' if sym else ""
    a = f'<span class="aide" title="{esc(aide)}">{icon("info")}</span>' if aide else ""
    return f'<div class="row{" full" if full else ""}"><label>{s}<span>{label}</span>{a}</label>{champ}</div>'


def num(chemin: Sequence, valeur, unite: str = "", dec: int = 3, entier: bool = False, desactive: bool = False, mini=None, maxi=None) -> str:
    """Un champ numérique lié à un CHEMIN de l'état (« geometrie/H ») : l'écran renvoie le texte tapé, le moteur le lit."""
    v = "" if valeur is None else (str(int(round(float(valeur)))) if entier else nb(valeur, dec))
    if v and not entier and abs(float(valeur)) >= 1000:
        ent, _, dec_ = v.partition(",")
        v = f"{int(ent):,}".replace(",", " ") + ("," + dec_ if dec_ else "")
    lim = (f' data-min="{mini}"' if mini is not None else "") + (f' data-max="{maxi}"' if maxi is not None else "")
    u = f'<span class="u">{unite}</span>' if unite else ""
    lk = icon("lock") if desactive else ""
    return (f'<div class="field{" locked" if desactive else ""}">{lk}<input data-chemin="{_attr_chemin(chemin)}" data-sorte="{"entier" if entier else "nombre"}"'
            f' inputmode="decimal" value="{esc(v)}"{lim}{" disabled" if desactive else ""}>{u}</div>')


def texte(chemin: Sequence, valeur: str, indice: str = "") -> str:
    return (f'<div class="field text"><input data-chemin="{_attr_chemin(chemin)}" data-sorte="texte" value="{esc(valeur or "")}"'
            f' placeholder="{esc(indice)}"></div>')


def zone_texte(chemin: Sequence, valeur: str, lignes: int = 5, indice: str = "") -> str:
    """Un texte sur plusieurs lignes (un tableau collé depuis Excel, une liste de couches…) : il part au moteur en
    quittant la zone, comme un champ."""
    return (f'<div class="field area"><textarea data-chemin="{_attr_chemin(chemin)}" data-sorte="texte" rows="{int(lignes)}"'
            f' placeholder="{esc(indice)}" spellcheck="false">{esc(valeur or "")}</textarea></div>')


def fichier(chemin: Sequence, libelle: str, accept: str = ".csv,.txt") -> str:
    """Un bouton qui lit un fichier du PC : le moteur reçoit {"nom": …, "texte": …} au chemin donné (2 Mo au plus)."""
    return (f'<label class="btn sm fichier">{icon("upload")}{esc(libelle)}<input type="file" data-chemin="{_attr_chemin(chemin)}"'
            f' accept="{esc(accept)}" hidden></label>')


def choix(chemin: Sequence, valeur, options: Iterable[Tuple], desactive: bool = False) -> str:
    """Un menu déroulant ; ``options`` : (valeur, libellé)."""
    opts = "".join(f'<option value="{esc(k)}"{" selected" if k == valeur else ""}>{esc(t)}</option>' for k, t in options)
    return (f'<div class="field select"><select data-chemin="{_attr_chemin(chemin)}" data-sorte="choix"{" disabled" if desactive else ""}>{opts}</select>'
            f'{icon("chevron-down")}</div>')


def segments(chemin: Sequence, valeur, options: Iterable[Tuple]) -> str:
    """Un choix court en segments (deux à quatre options)."""
    bt = "".join(f'<button type="button" class="{"on" if k == valeur else ""}" data-chemin="{_attr_chemin(chemin)}" data-sorte="choix" data-valeur="{esc(k)}">{esc(t)}</button>'
                 for k, t in options)
    return f'<div class="seg full">{bt}</div>'


def case(chemin: Sequence, valeur: bool, texte_: str) -> str:
    return (f'<label class="case"><input type="checkbox" data-chemin="{_attr_chemin(chemin)}" data-sorte="case"{" checked" if valeur else ""}>'
            f'<span>{texte_}</span></label>')


def section(n: int, titre: str, resume: str, corps: str, ouvert: bool = True, cle: str = "") -> str:
    """Une section repliable du cadre Données (le numéro, le titre, un résumé quand elle est repliée)."""
    return (f'<div class="sec{" open" if ouvert else ""}" data-sec="{esc(cle or n)}"><div class="sechd" data-replier="1"><span class="n">{n}</span>'
            f'<span class="tt">{titre}</span><span class="sum">{resume}</span>{icon("chevron-down")}</div>'
            f'<div class="rows">{corps}</div></div>')


def aide(texte_: str) -> str:
    return f'<div class="hint">{texte_}</div>'


def message(texte_: str, grave: bool = False) -> str:
    return f'<div class="msg{" grave" if grave else ""}">{texte_}</div>'

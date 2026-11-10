# -*- coding: utf-8 -*-
"""La BIBLIOTHÈQUE BÉTON ARMÉ du site (docs/site/BIBLIOTHEQUES.md § 2.2) : une page, cinq rubriques, chacune un
TABLEAU complet à sélection et une FICHE RAPIDE (les paramètres allument la case, la bande sous le tableau donne la
valeur retenue, « Pourquoi ? » déplie la justification). Ce fichier n'est pas un module de calcul du registre (il ne
déclare ni MODULE ni PAGE : une consultation n'est pas une note) ; le routeur l'appelle par l'action « consulter »
(api_web/routeur.py), la page bibliotheque-beton (site/web/construire.py, page_bibliotheque) et, plus tard, le panneau
« Bibliothèque » d'une note (§ 2.4) appellent la même fonction : un seul HTML, deux hôtes.

    consulter(rubrique, params, hote="page") → {rubrique, titre, barre, tableau, bande, pourquoi, sources, fiche,
                                                  params, message}

Les valeurs viennent des moteurs (beton/donnees/ec2.py, beton/donnees/consultation.py, beton/armatures,
beton/bibliotheque/{aciers, enrobage, feu, ancrages}.py), jamais d'ici ; les paramètres illisibles reprennent leur défaut et
``message`` le dit ; tout texte venu des paramètres est échappé.
"""
import html
import json
import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

from api_web import commun as C
from beton.armatures import moteur as AM
from beton.armatures import tableau as AT
from beton.bibliotheque import aciers as SA
from beton.bibliotheque import ancrages as AN
from beton.bibliotheque import enrobage as EN
from beton.donnees import consultation as K
from beton.donnees import ec2 as EC

BIBLIOTHEQUE = "beton"
TITRE = "Bibliothèque Béton armé"
PAGE = "bibliotheque-beton"
VERSION = "2026-11-10"       # la date des données de la bibliothèque : une fiche copiée dans une note dit quel moteur l'a rendue
# (clé, titre du menu, sous-titre du menu, icône)
RUBRIQUES = (
    ("betons", "Bétons", "Classes, résistances, déformations", "layers"),
    ("aciers", "Aciers", "Nuances, ductilité, treillis", "sliders-horizontal"),
    ("sections", "Sections", "Sections et masses des armatures", "grid-2x2"),
    ("enrobages", "Enrobages", "Durabilité et feu", "shield-check"),
    ("ancrages", "Ancrages", "Ancrages et recouvrements", "anchor"),
)
CLES = tuple(r[0] for r in RUBRIQUES)
NOMS = {r[0]: r[1] for r in RUBRIQUES}
RUBRIQUE_DEFAUT = "betons"
HOTES = ("page", "note")
NBSP = C.NBSP


# ================================================================================================ les paramètres
class Lecture:
    """La relecture des paramètres d'une rubrique : chaque paramètre illisible reprend son défaut, et le dit."""

    def __init__(self, params: Optional[Dict]):
        self.brut = params if isinstance(params, dict) else {}
        self.messages: List[str] = []
        self.out: Dict = {}

    def _dit(self, cle: str, pourquoi: str) -> None:
        self.messages.append(f"« {cle} » {pourquoi}")

    def choix(self, cle: str, permis: Sequence, defaut):
        v = self.brut.get(cle, defaut)
        if v is None:
            v = defaut
        # une valeur numérique proposée (Ø, f_yk, γ, ρ1) se reconnaît aussi écrite en texte (« 1 », « 1,0 », « 1.0 »)
        for p in permis:
            if v == p and type(v) is type(p):
                self.out[cle] = p
                return p
            if not isinstance(p, str) and not isinstance(p, bool) and not isinstance(v, bool):
                try:
                    if float(str(v).replace(",", ".")) == float(p):
                        self.out[cle] = p
                        return p
                except ValueError:
                    pass
        self._dit(cle, f"ignoré : « {str(v)[:40]} » n'est pas proposé")
        self.out[cle] = defaut
        return defaut

    def nombre(self, cle: str, defaut: Optional[float], mini: float, maxi: float, entier: bool = False, vide=None):
        v = self.brut.get(cle, defaut)
        if v is None or (isinstance(v, str) and not v.strip()):
            self.out[cle] = vide if defaut is None else defaut
            return self.out[cle]
        try:
            if isinstance(v, bool):
                raise ValueError
            x = float(str(v).replace(",", ".").replace(NBSP, "").replace(" ", ""))
            if not math.isfinite(x):
                raise ValueError
        except (ValueError, TypeError):
            self._dit(cle, f"ignoré : « {str(v)[:40]} » n'est pas un nombre")
            self.out[cle] = defaut if defaut is not None else vide
            return self.out[cle]
        if x < mini or x > maxi:
            self._dit(cle, f"ramené dans {C.nb(mini, 2)} à {C.nb(maxi, 2)}")
            x = min(maxi, max(mini, x))
        if entier:
            x = int(round(x))
        self.out[cle] = x
        return x

    def case(self, cle: str, defaut: bool = False) -> bool:
        v = self.brut.get(cle, defaut)
        if isinstance(v, str):
            v = v.strip().lower() in ("1", "true", "oui", "on", "vrai")
        self.out[cle] = bool(v)
        return self.out[cle]

    def message(self) -> Optional[str]:
        return " ; ".join(self.messages) if self.messages else None


# ================================================================================================ les briques HTML
def _th(sym: str, unite: str = "", titre: str = "", cls: str = "") -> str:
    t = f' title="{C.esc(titre)}"' if titre else ""
    u = f'<span class="u">{C.esc(unite)}</span>' if unite else ""
    return f'<th{(" class=" + chr(34) + cls + chr(34)) if cls else ""}{t}><span class="s">{C.symb_math(sym)}</span>{u}</th>'


def _sel(d: Dict) -> str:
    """L'attribut data-sel d'une case : les paramètres qu'un clic règle (JSON échappé)."""
    return f' data-sel="{C.esc(json.dumps(d, ensure_ascii=False))}"'


def _td(texte: str, sel: Optional[Dict] = None, cls: str = "", titre: str = "") -> str:
    t = f' title="{C.esc(titre)}"' if titre else ""
    return f'<td{(" class=" + chr(34) + cls + chr(34)) if cls else ""}{_sel(sel) if sel else ""}{t}>{texte}</td>'


def _tableau(thead: str, tbody: str, cls: str = "") -> str:
    return f'<table class="bib-tab{(" " + cls) if cls else ""}"><thead>{thead}</thead><tbody>{tbody}</tbody></table>'


_MATH = re.compile(r"<sub>|[α-ω]")


def _champ(cle: str, libelle: str, controle: str, aide: str = "") -> str:
    """Un paramètre de la barre : son libellé (en capitales, sauf un symbole — σ_sd ne devient pas Σ), l'aide ⓘ au
    survol, le contrôle. Sans libellé (une case à cocher), l'aide est dans la case elle-même."""
    if not libelle:
        return f'<label class="bib-p bib-p-case">{controle}</label>'
    a = f'<span class="aide" title="{C.esc(aide)}">{C.icon("info")}</span>' if aide else ""
    cls = "bib-pl m" if _MATH.search(libelle) else "bib-pl"
    return f'<label class="bib-p"><span class="{cls}">{libelle}{a}</span>{controle}</label>'


def _select(cle: str, valeur, options: Sequence[Tuple]) -> str:
    opts = "".join(f'<option value="{C.esc(str(k))}"{" selected" if str(k) == str(valeur) else ""}>{C.esc(str(t))}</option>' for k, t in options)
    return f'<div class="field select"><select data-param="{C.esc(cle)}">{opts}</select>{C.icon("chevron-down")}</div>'


def _segments(cle: str, valeur, options: Sequence[Tuple]) -> str:
    bt = "".join(f'<button type="button"{" class=" + chr(34) + "on" + chr(34) if str(k) == str(valeur) else ""} data-valeur="{C.esc(str(k))}">{t}</button>'
                 for k, t in options)
    return f'<div class="seg" data-param="{C.esc(cle)}">{bt}</div>'


def _num(cle: str, valeur, unite: str = "", vide: str = "") -> str:
    v = "" if valeur is None else (C.nb(valeur, 2) if isinstance(valeur, float) and valeur != int(valeur) else str(int(valeur)) if isinstance(valeur, (int, float)) else str(valeur))
    u = f'<span class="u">{unite}</span>' if unite else ""
    ph = f' placeholder="{C.esc(vide)}"' if vide else ""
    return f'<div class="field"><input data-param="{C.esc(cle)}" inputmode="decimal" value="{C.esc(v)}"{ph}>{u}</div>'


def _case(cle: str, valeur: bool, texte: str, aide: str = "") -> str:
    a = f'<span class="aide" title="{C.esc(aide)}">{C.icon("info")}</span>' if aide else ""
    return f'<span class="case bib-case"><input type="checkbox" data-param="{C.esc(cle)}"{" checked" if valeur else ""}><span>{texte}</span>{a}</span>'


def _barre(*lignes: str) -> str:
    return "".join(f'<div class="bib-barre-l">{li}</div>' for li in lignes if li)


def _bande(valeur: str, unite: str, symbole: str, nom: str, phrase: str, hote: str, avert: Sequence[str] = (), pourquoi: bool = True) -> str:
    """La bande sous le tableau : la valeur retenue en grand, son symbole et son nom, la phrase des paramètres, les
    avertissements, puis les boutons (« Pourquoi ? » ; dans une note : « Inclure dans la note », « Utiliser cette
    valeur »)."""
    av = "".join(f'<div class="bib-av">{C.icon("triangle-alert")}<span>{a}</span></div>' for a in avert)
    boutons = ""
    if pourquoi:
        boutons += f'<button type="button" class="btn" data-pourquoi="1">{C.icon("lightbulb")}<span>Pourquoi ?</span></button>'
    if hote == "note":
        boutons += (f'<button type="button" class="btn" data-inclure="fiche">{C.icon("file-plus-2")}<span>Inclure dans la note</span></button>'
                    f'<button type="button" class="btn" data-inclure="extrait">{C.icon("grid-2x2")}<span>Inclure l\'extrait du tableau</span></button>'
                    f'<button type="button" class="btn primary" data-utiliser="1">{C.icon("check")}<span>Utiliser cette valeur</span></button>')
    return (f'<div class="bib-bande-v"><b class="bib-val">{valeur}<small>{unite}</small></b>'
            f'<div class="bib-bande-t"><span class="bib-sym">{C.symb_math(symbole)}</span><span class="bib-nom">{nom}</span>'
            f'<span class="bib-phr">{phrase}</span></div></div>{av}<div class="bib-bande-b">{boutons}</div>')


DECIMALES = {"mm": 1, "MPa": 2, "GPa": 0, "‰": 2, "%": 1, "": 3, "mm²": 0, "mm²/m": 0, "kg/m": 3, "kg/m²": 2, "kg/m³": 0, "cm": 1, "barres/m": 2}


def _valeur_txt(v, unite: str, nd: int) -> str:
    """La valeur d'un terme telle que la justification l'écrit — et telle que la fiche la garde pour la note : « oui »
    ou « non », un nombre aux décimales de son unité (DECIMALES ; ``nd`` pour une unité inconnue ; un entier ou un
    grand nombre rond sans décimale), ou le texte tel quel."""
    if isinstance(v, bool):
        return "oui" if v else "non"
    if isinstance(v, (int, float)):
        d = DECIMALES.get(unite, nd)
        return C.fr(v, 0 if (isinstance(v, int) or (abs(v - round(v)) < 1e-9 and abs(v) >= 10)) else d)
    return str(v)


def _texte_brut(html_: str) -> str:
    """Une phrase de la bande (du HTML : <sub>, <b>) en texte pour la fiche : « Δc<sub>dev</sub> » devient « Δc_dev »,
    que la note PDF compose à son tour ; les autres balises tombent, les entités reviennent à leurs caractères."""
    t = re.sub(r"<sub>(.*?)</sub>", r"_\1", html_ or "")
    return html.unescape(re.sub(r"<[^>]+>", "", t)).strip()


def _pourquoi(titre: str, sous: str, termes: Sequence[Dict], sources: Sequence[str], avert: Sequence[str] = (), nd: int = 2) -> str:
    """La justification : une ligne par terme (symbole, nom, valeur, règle), les avertissements, les sources. Les
    décimales suivent l'unité (un mm au dixième, un coefficient au millième) ; ``nd`` pour une unité inconnue."""
    lignes = ""
    for t in termes:
        vt = C.esc(_valeur_txt(t["valeur"], t.get("unite", ""), nd))
        u = f' <span class="u">{C.esc(t.get("unite", ""))}</span>' if t.get("unite") else ""
        maj = ' class="maj"' if t["nom"].isupper() else ""
        lignes += (f'<tr{maj}><td class="s">{C.symb_math(t["symbole"])}</td><td class="nm">{C.esc(t["nom"].capitalize() if t["nom"].isupper() else t["nom"])}</td>'
                   f'<td class="n">{vt}{u}</td><td class="rg">{C.esc(t["regle"])}</td></tr>')
    av = "".join(f'<div class="bib-av">{C.icon("triangle-alert")}<span>{a}</span></div>' for a in avert)
    src = "".join(f'<div class="bib-src">{C.icon("file-text")}<span>{s}</span></div>' for s in sources)
    return (f'<div class="bib-pq-h"><div><b>{titre}</b><span>{sous}</span></div>'
            f'<button type="button" class="btn icon ghost" data-fermer-pourquoi="1" aria-label="Fermer">{C.icon("x")}</button></div>'
            f'<div class="bib-pq-b"><table class="bib-pq-t"><tbody>{lignes}</tbody></table>{av}<div class="bib-pq-s">{src}</div></div>')


def _sources(*textes: str) -> str:
    return "".join(f'<div class="bib-src">{C.icon("file-text")}<span>{t}</span></div>' for t in textes if t)


def _fiche(titre: str, symbole: str, valeur, unite: str, params: Dict, termes: Sequence[Dict], sources: Sequence[str],
           extrait: Optional[Dict] = None, avertissements: Sequence[str] = (), texte: str = "", phrase: str = "", nd: int = 2) -> Dict:
    """La fiche figée que « Inclure dans la note » copie dans l'état d'une note (§ 2.5, api_web/fiches.py la relit) :
    tout ce qu'il faut pour la relire sans le moteur — la valeur (le nombre) et ``texte`` (la valeur telle que la bande
    l'affiche), la ``phrase`` des paramètres en texte, chaque ligne avec sa valeur écrite comme dans « Pourquoi ? »,
    les sources, l'extrait du tableau, et le moteur qui l'a rendue (VERSION)."""
    return {"titre": titre, "symbole": symbole, "valeur": valeur, "texte": texte, "unite": unite, "phrase": _texte_brut(phrase), "params": params,
            "lignes": [{"symbole": t["symbole"], "nom": t["nom"], "valeur": t["valeur"], "texte": _valeur_txt(t["valeur"], t.get("unite", ""), nd),
                        "unite": t.get("unite", ""), "regle": t["regle"]} for t in termes],
            "sources": list(sources), "avertissements": list(avertissements), "extrait": extrait, "moteur": f"{TITRE} {VERSION}"}


def _extrait(entetes: Sequence[str], lignes: Sequence[Sequence], case: Optional[Tuple[int, int]], titre: str, source: str) -> Dict:
    return {"titre": titre, "entetes": list(entetes), "lignes": [list(li) for li in lignes], "case": list(case) if case else None, "source": source}


def _mm(x: float) -> str:
    return C.fr(x, 0)


# ================================================================================================ 1 · bétons
CLASSES_BETON = [f"C{a}/{b}" for a, b in EC.CLASSES_3_1]
GAMMAS_C = ((1.5, "1,5 · durable"), (1.2, "1,2 · accidentelle"))
ALPHAS_CC = ((1.0, "1,0 · recommandé"), (0.85, "0,85 · usage"))
COLONNES_EC2 = (
    ("fck", "f_ck", "résistance caractéristique sur cylindre", "MPa", 0),
    ("fck_cube", "f_ck,cube", "résistance caractéristique sur cube", "MPa", 0),
    ("fcm", "f_cm", "résistance moyenne en compression", "MPa", 0),
    ("fctm", "f_ctm", "résistance moyenne en traction", "MPa", 1),
    ("fctk_005", "f_ctk,0,05", "traction caractéristique, fractile 5 %", "MPa", 1),
    ("fctk_095", "f_ctk,0,95", "traction caractéristique, fractile 95 %", "MPa", 1),
    ("Ecm", "E_cm", "module d'élasticité sécant", "GPa", 0),
    ("eps_c2", "ε_c2", "déformation au pic, parabole-rectangle", "‰", 1),
    ("eps_cu2", "ε_cu2", "déformation ultime, parabole-rectangle", "‰", 1),
    ("n", "n", "exposant du diagramme parabole-rectangle", "", 2),
    ("fcd", "f_cd", "résistance de calcul en compression", "MPa", 1),
)
COLONNES_BUREAU = tuple(c for c in K.COLONNES_CLASSES if c[5] == "bureau")


def _ligne_beton(fck: float, cube: int, alpha_cc: float, gamma_c: float) -> Dict:
    return {"nom": f"C{int(fck)}/{cube}", "fck": fck, "fck_cube": float(cube), "fcm": EC.fcm(fck), "fctm": EC.fctm(fck),
            "fctk_005": EC.fctk_005(fck), "fctk_095": EC.fctk_095(fck), "Ecm": EC.Ecm(fck), "eps_c2": EC.eps_c2(fck),
            "eps_cu2": EC.eps_cu2(fck), "n": EC.n_parabole(fck), "fcd": EC.fcd(fck, alpha_cc, gamma_c),
            "fctd": EC.fctk_005(fck) / gamma_c}


def _r_betons(params: Dict, hote: str) -> Dict:
    L = Lecture(params)
    classe = L.choix("classe", CLASSES_BETON, "C30/37")
    groupe = L.choix("groupe", ("ec2", "bureau"), "ec2")
    alpha_cc = L.choix("alpha_cc", tuple(a for a, _ in ALPHAS_CC), 1.0)
    gamma_c = L.choix("gamma_c", tuple(g for g, _ in GAMMAS_C), 1.5)
    p = L.out
    lignes = [_ligne_beton(float(a), b, alpha_cc, gamma_c) for a, b in EC.CLASSES_3_1]
    bureau = {x["nom"]: x for x in K.classes()}
    x = next(li for li in lignes if li["nom"] == classe)
    barre = _barre(_champ("classe", "Classe", _select("classe", classe, [(c, c) for c in CLASSES_BETON]))
                   + _champ("groupe", "Valeurs", _segments("groupe", groupe, (("ec2", "EN 1992-1-1"), ("bureau", "Méthode du bureau"))))
                   + _champ("alpha_cc", "α<sub>cc</sub>", _segments("alpha_cc", alpha_cc, ALPHAS_CC), "f_cd = α_cc · f_ck / γ_c (§3.1.6(1)P) ; l'annexe belge est à confirmer par le bureau")
                   + _champ("gamma_c", "γ<sub>c</sub>", _segments("gamma_c", gamma_c, GAMMAS_C), "tableau 2.1N : 1,5 en situation durable et transitoire, 1,2 en situation accidentelle"))
    if groupe == "ec2":
        th = '<tr><th class="nm">Classe</th>' + "".join(_th(s, u, n) for _, s, n, u, _ in COLONNES_EC2) + "</tr>"
        corps = ""
        for li in lignes:
            on = ' class="on"' if li["nom"] == classe else ""
            corps += (f'<tr{on}><th class="nm">{li["nom"]}</th>'
                      + "".join(_td(C.fr(li[k], nd), {"classe": li["nom"]}, "cur" if (li["nom"] == classe and k == "fcd") else "") for k, _, _, _, nd in COLONNES_EC2) + "</tr>")
        entetes = ["Classe"] + [s for _, s, _, _, _ in COLONNES_EC2]
        extrait_lignes = [[li["nom"]] + [C.fr(li[k], nd) for k, _, _, _, nd in COLONNES_EC2] for li in lignes]
        i = CLASSES_BETON.index(classe)
        extrait = _extrait(entetes, extrait_lignes, (i, len(entetes) - 1), "EN 1992-1-1, tableau 3.1 (calculé)", EC.SOURCE_BETON)
    else:
        th = '<tr><th class="nm">Classe</th>' + _th("f_ck", "MPa") + _th("f_ck,cube", "MPa") + "".join(_th(s, u, n) for _, s, n, u, _, _ in COLONNES_BUREAU) + "</tr>"
        corps = ""
        for li in lignes:
            b = bureau.get(li["nom"])
            on = ' class="on"' if li["nom"] == classe else ""
            if b:
                cases = "".join(_td(C.fr(b[k], nd), {"classe": li["nom"]}) for k, _, _, _, nd, _ in COLONNES_BUREAU)
            else:
                cases = "".join(_td("—", {"classe": li["nom"]}, "vide", "classe hors des tableaux du bureau") for _ in COLONNES_BUREAU)
            corps += (f'<tr{on}><th class="nm">{li["nom"]}</th>{_td(C.fr(li["fck"], 0), {"classe": li["nom"]})}{_td(C.fr(li["fck_cube"], 0), {"classe": li["nom"]})}{cases}</tr>')
        extrait = None
    tableau = _tableau(th, corps, "bib-betons")
    termes = [
        {"symbole": "f_ck", "nom": "résistance caractéristique sur cylindre à 28 jours", "valeur": x["fck"], "unite": "MPa", "regle": "classe de résistance — tableau 3.1"},
        {"symbole": "f_cm", "nom": "résistance moyenne", "valeur": x["fcm"], "unite": "MPa", "regle": "f_ck + 8 — tableau 3.1"},
        {"symbole": "f_ctm", "nom": "traction moyenne", "valeur": x["fctm"], "unite": "MPa", "regle": "0,30 · f_ck^(2/3) jusqu'à C50/60 ; 2,12 · ln(1 + f_cm/10) au-delà — tableau 3.1"},
        {"symbole": "f_ctk,0,05", "nom": "traction caractéristique, fractile 5 %", "valeur": x["fctk_005"], "unite": "MPa", "regle": "0,7 · f_ctm — tableau 3.1"},
        {"symbole": "f_ctk,0,95", "nom": "traction caractéristique, fractile 95 %", "valeur": x["fctk_095"], "unite": "MPa", "regle": "1,3 · f_ctm — tableau 3.1"},
        {"symbole": "E_cm", "nom": "module d'élasticité sécant", "valeur": x["Ecm"], "unite": "GPa", "regle": "22 · (f_cm/10)^0,3 — tableau 3.1"},
        {"symbole": "ε_c2", "nom": "déformation au pic, parabole-rectangle", "valeur": x["eps_c2"], "unite": "‰", "regle": "2,0 jusqu'à C50/60 ; 2,0 + 0,085 · (f_ck − 50)^0,53 au-delà — tableau 3.1"},
        {"symbole": "ε_cu2", "nom": "déformation ultime, parabole-rectangle", "valeur": x["eps_cu2"], "unite": "‰", "regle": "3,5 jusqu'à C50/60 ; 2,6 + 35 · ((90 − f_ck)/100)^4 au-delà — tableau 3.1"},
        {"symbole": "n", "nom": "exposant du diagramme parabole-rectangle", "valeur": x["n"], "unite": "", "regle": "2,0 jusqu'à C50/60 ; 1,4 + 23,4 · ((90 − f_ck)/100)^4 au-delà — tableau 3.1"},
        {"symbole": "f_cd", "nom": "RÉSISTANCE DE CALCUL EN COMPRESSION", "valeur": x["fcd"], "unite": "MPa", "regle": f"α_cc · f_ck / γ_c = {C.nb(alpha_cc, 2)} · {C.fr(x['fck'], 0)} / {C.nb(gamma_c, 2)} — §3.1.6(1)P, éq. (3.15)"},
        {"symbole": "f_ctd", "nom": "traction de calcul", "valeur": x["fctd"], "unite": "MPa", "regle": f"α_ct · f_ctk,0,05 / γ_c = 1,0 · f_ctk,0,05 / {C.nb(gamma_c, 2)} — §3.1.6(2)P, éq. (3.16)"},
    ]
    b = bureau.get(classe)
    if b:
        termes += [{"symbole": s, "nom": n, "valeur": b[k], "unite": u, "regle": f"méthode du bureau (beton_classes.json, moteur des poutres)"} for k, s, n, u, _, _ in COLONNES_BUREAU]
    sources = [EC.SOURCE_BETON, EC.SOURCE_FCD]
    if b:
        sources.append(K.SOURCE_BUREAU)
    phrase = f"{classe} · α<sub>cc</sub> = {C.nb(alpha_cc, 2)} · γ<sub>c</sub> = {C.nb(gamma_c, 2)}"
    avert = ["Classe hors des tableaux du bureau (α_b, μ, τ_adm : C20/25 à C40/50)."] if groupe == "bureau" and not b else []
    return {"titre": "Bétons", "barre": barre, "tableau": tableau,
            "bande": _bande(C.fr(x["fcd"], 1), "MPa", "f_cd", "résistance de calcul en compression", phrase, hote, avert),
            "pourquoi": _pourquoi(f"Béton {classe}", "EN 1992-1-1, tableau 3.1 et §3.1.6", termes, sources, avert),
            "sources": _sources(*sources),
            "fiche": _fiche(f"Béton {classe}", "f_cd", x["fcd"], "MPa", p, termes, sources, extrait, avert, texte=C.fr(x["fcd"], 1), phrase=phrase),
            "params": p, "message": L.message()}


# ================================================================================================ 2 · aciers
GAMMAS_S = ((EC.GAMMA_S, f"{C.nb(EC.GAMMA_S, 2)} · EN 1992-1-1"), (K.GAMMA_S_BUREAU, f"{C.nb(K.GAMMA_S_BUREAU, 2)} · bureau"))
VUES_ACIERS = (("nuances", "Nuances"), ("treillis", "Treillis soudés"))


def _r_aciers(params: Dict, hote: str) -> Dict:
    from beton import treillis as TR
    L = Lecture(params)
    noms = [n["nom"] for n in SA.nuances()]
    nuance = L.choix("nuance", noms, "B500B")
    gamma_s = L.choix("gamma_s", tuple(g for g, _ in GAMMAS_S), EC.GAMMA_S)
    vue = L.choix("vue", tuple(k for k, _ in VUES_ACIERS), "nuances")
    treillis = L.choix("treillis", [TR.designation(t) for t in TR.TREILLIS_STANDARDS], TR.TREILLIS_DEFAUT)
    p = L.out
    f = SA.fiche(int(nuance[1:4]), nuance[4])
    fyd = f["fyk"] / gamma_s
    eps_yd = EC.eps_yd(fyd)
    barre = _barre(_champ("nuance", "Nuance", _select("nuance", nuance, [(n, n) for n in noms]))
                   + _champ("gamma_s", "γ<sub>s</sub>", _segments("gamma_s", gamma_s, GAMMAS_S), "tableau 2.1N : 1,15 à l'ELU ; le bureau prend 1,5 dans ses modules béton")
                   + _champ("vue", "Tableau", _segments("vue", vue, VUES_ACIERS))
                   + (_champ("treillis", "Treillis", _select("treillis", treillis, [(TR.designation(t), TR.designation(t)) for t in TR.TREILLIS_STANDARDS])) if vue == "treillis" else ""))
    if vue == "nuances":
        th = ('<tr><th class="nm">Nuance</th>' + _th("f_yk", "MPa", "limite d'élasticité caractéristique") + _th("f_yd", "MPa", f"f_yk / γ_s, γ_s = {C.nb(gamma_s, 2)}")
              + _th("E_s", "GPa", "module d'élasticité") + _th("ε_yd", "‰", "allongement à f_yd") + _th("k", "", "(f_t / f_y)_k, minimum (classe C : minimum et maximum)")
              + _th("ε_uk", "%", "allongement caractéristique sous charge maximale, minimum") + '<th class="tx">Classe de ductilité · usage</th></tr>')
        corps = ""
        for n in SA.nuances():
            on = ' class="on"' if n["nom"] == nuance else ""
            fy = n["fyk"] / gamma_s
            k = f"≥ {C.nb(n['k_min'], 2)}" + (f" ; &lt; {C.nb(n['k_max'], 2)}" if n["k_max"] else "")
            sel = {"nuance": n["nom"]}
            corps += (f'<tr{on}><th class="nm">{n["nom"]}</th>{_td(C.fr(n["fyk"], 0), sel)}{_td(C.fr(fy, 1), sel, "cur" if n["nom"] == nuance else "")}'
                      f'{_td(C.fr(SA.E_S, 0), sel)}{_td(C.fr(EC.eps_yd(fy), 2), sel)}{_td(k, sel)}{_td("≥ " + C.fr(n["eps_uk_min"], 1), sel)}'
                      f'{_td(C.esc(n["classe"] + " · " + n["usage"]), sel, "tx")}</tr>')
        tableau = _tableau(th, corps, "bib-aciers")
        extrait = _extrait(["Nuance", "f_yk", "f_yd", "E_s", "ε_yd", "k", "ε_uk"],
                           [[n["nom"], C.fr(n["fyk"], 0), C.fr(n["fyk"] / gamma_s, 1), C.fr(SA.E_S, 0), C.fr(EC.eps_yd(n["fyk"] / gamma_s), 2),
                             f"≥ {C.nb(n['k_min'], 2)}", f"≥ {C.fr(n['eps_uk_min'], 1)} %"] for n in SA.nuances()],
                           (noms.index(nuance), 2), f"Aciers d'armature (γ_s = {C.nb(gamma_s, 2)})", SA.SOURCE)
    else:
        th = ('<tr><th class="nm">Treillis</th>' + _th("Ø_l", "mm", "fils porteurs") + _th("Ø_t", "mm", "fils de répartition") + _th("e_l", "mm", "maille des fils porteurs")
              + _th("e_t", "mm", "maille des fils de répartition") + _th("A_s,l", "mm²/m", "section des fils porteurs par mètre") + _th("A_s,t", "mm²/m", "section des fils de répartition par mètre")
              + _th("masse", "kg/m²", f"ρ = {C.fr(SA.MASSE_VOLUMIQUE, 0)} kg/m³") + "</tr>")
        corps = ""
        for t in TR.TREILLIS_STANDARDS:
            des = TR.designation(t)
            on = ' class="on"' if des == treillis else ""
            asl = TR.as_barres_mm2_m(t[0], t[2])
            ast = TR.as_barres_mm2_m(t[1], t[3])
            masse = (asl + ast) * 1e-6 * SA.MASSE_VOLUMIQUE
            sel = {"treillis": des}
            corps += (f'<tr{on}><th class="nm">{des}</th>{_td(str(t[0]), sel)}{_td(str(t[1]), sel)}{_td(str(t[2]), sel)}{_td(str(t[3]), sel)}'
                      f'{_td(C.fr(asl, 0), sel, "cur" if des == treillis else "")}{_td(C.fr(ast, 0), sel)}{_td(C.fr(masse, 2), sel)}</tr>')
        tableau = _tableau(th, corps, "bib-treillis")
        extrait = None
    termes = [
        {"symbole": "f_yk", "nom": "limite d'élasticité caractéristique", "valeur": f["fyk"], "unite": "MPa", "regle": f"nuance {nuance} — §3.2.2"},
        {"symbole": "γ_s", "nom": "coefficient partiel de l'acier", "valeur": gamma_s, "unite": "", "regle": "tableau 2.1N (1,15 à l'ELU) ; 1,5 : méthode du bureau"},
        {"symbole": "f_yd", "nom": "LIMITE D'ÉLASTICITÉ DE CALCUL", "valeur": fyd, "unite": "MPa", "regle": f"f_yk / γ_s = {C.fr(f['fyk'], 0)} / {C.nb(gamma_s, 2)} — §3.2.7(2)"},
        {"symbole": "E_s", "nom": "module d'élasticité", "valeur": SA.E_S, "unite": "GPa", "regle": "valeur de calcul — §3.2.7(4)"},
        {"symbole": "ε_yd", "nom": "allongement à la limite élastique de calcul", "valeur": eps_yd, "unite": "‰", "regle": "f_yd / E_s"},
        {"symbole": "k", "nom": "rapport (f_t / f_y)_k", "valeur": f"≥ {C.nb(f['k_min'], 2)}" + (f" et < {C.nb(f['k_max'], 2)}" if f["k_max"] else ""), "unite": "", "regle": f"classe {f['classe']} — annexe C, tableau C.1"},
        {"symbole": "ε_uk", "nom": "allongement caractéristique sous charge maximale", "valeur": f"≥ {C.fr(f['eps_uk_min'], 1)}", "unite": "%", "regle": f"classe {f['classe']} — annexe C, tableau C.1"},
        {"symbole": "ρ", "nom": "masse volumique", "valeur": SA.MASSE_VOLUMIQUE, "unite": "kg/m³", "regle": "§3.2.7(3)"},
    ]
    if vue == "treillis":
        t = TR.parse_designation(treillis)
        termes += [{"symbole": "A_s,l", "nom": "SECTION DES FILS PORTEURS PAR MÈTRE", "valeur": TR.as_barres_mm2_m(t[0], t[2]), "unite": "mm²/m",
                    "regle": f"π Ø_l²/4 · 1000 / e_l = π · {t[0]}²/4 · 1000 / {t[2]} — beton/treillis.py"}]
    sources = [SA.SOURCE, EC.SOURCE_GAMMA_S]
    phrase = f"{nuance} · γ<sub>s</sub> = {C.nb(gamma_s, 2)}" + (f" · treillis {treillis}" if vue == "treillis" else "")
    if vue == "treillis":
        t = TR.parse_designation(treillis)
        valeur, unite, sym, nom = C.fr(TR.as_barres_mm2_m(t[0], t[2]), 0), "mm²/m", "A_s,l", f"treillis {treillis} : fils porteurs"
    else:
        valeur, unite, sym, nom = C.fr(fyd, 1), "MPa", "f_yd", "limite d'élasticité de calcul"
    return {"titre": "Aciers d'armature", "barre": barre, "tableau": tableau,
            "bande": _bande(valeur, unite, sym, nom, phrase, hote),
            "pourquoi": _pourquoi(f"Acier {nuance}", "EN 1992-1-1 §3.2 et annexe C", termes, sources),
            "sources": _sources(*sources),
            "fiche": _fiche(f"Acier {nuance}", sym, fyd if vue == "nuances" else TR.as_barres_mm2_m(t[0], t[2]), unite, p, termes, sources, extrait,
                            texte=valeur, phrase=phrase),
            "params": p, "message": L.message()}


# ================================================================================================ 3 · sections
MODES = (("barres", "Barres"), ("nappe", "Par mètre"))
NOMS_FORMULES = {"a": "aire d'une barre", "As": "section", "m": "masse d'une barre", "M": "masse", "e": "espacement", "n": "barres par mètre",
                 "taux": "taux A_s,req / A_s"}


def _r_sections(params: Dict, hote: str) -> Dict:
    L = Lecture(params)
    mode = L.choix("mode", tuple(k for k, _ in MODES), "barres")
    d = L.choix("d", AT.DIAMETRES, 12)
    n = L.nombre("n", 4, 1, AT.NOMBRES[-1], entier=True)
    esp = L.choix("espacement", tuple(x["cle"] for x in AM.ESPACEMENTS), "e15")
    as_req = L.nombre("As_req", None, 0, AM.AS_REQ_MAX, vide=0.0)
    p = L.out
    besoin = {"mode": mode, "d": d, "n": int(n), "espacement": esp, "As_req": float(as_req or 0.0)}
    r = AM.calculer(besoin)
    ret, opt, tab = r["disposition"], r.get("optimale"), r["tableau"]
    unite = AM.UNITES[mode]
    barre = _barre(_champ("mode", "Lecture", _segments("mode", mode, MODES))
                   + _champ("d", "Ø", _select("d", d, [(x, f"Ø {x}") for x in AT.DIAMETRES]))
                   + (_champ("n", "Barres", _num("n", int(n))) if mode == "barres"
                      else _champ("espacement", "Espacement", _select("espacement", esp, [(x["cle"], x["libelle"] + " cm") for x in AM.ESPACEMENTS])))
                   + _champ("As_req", f"A<sub>s</sub> requis", _num("As_req", as_req if as_req else None, unite, "facultatif"),
                            "la plus petite section du tableau qui suffit est marquée d'une étoile"))
    # le tableau de l'image du bureau : Ø × (n barres | n barres par mètre et espacements fixes)
    cols = tab["colonnes"]
    th1 = ('<tr class="r1"><th class="nm" rowspan="2">Ø<span class="u">mm</span></th><th rowspan="2" class="m" title="aire d\'une barre">A<span class="u">mm²</span></th>'
           f'<th rowspan="2" class="m" title="masse linéique, ρ = {C.fr(tab["masse_volumique"], 0)} kg/m³">kg/m</th>'
           + (f'<th colspan="{len(AT.NOMBRES)}">Nombre de barres</th>' if mode == "barres"
              else f'<th colspan="{len(AT.NOMBRES)}">Barres par mètre · espacement (cm)</th><th colspan="{len(AT.ESPACEMENTS_FIXES)}">Espacement fixe (cm)</th>') + "</tr>")
    th2 = "<tr class=\"r2\">" + "".join(f'<th class="k"><b>{k}</b>' + (f'<small>{tab["espacements_n_txt"][k - 1]}</small>' if mode == "nappe" else "") + "</th>" for k in AT.NOMBRES)
    if mode == "nappe":
        th2 += "".join(f'<th class="k"><b>{e}</b></th>' for e in AT.ESPACEMENTS_FIXES)
    th2 += "</tr>"
    corps = ""
    for li in tab["lignes"]:
        on = ' class="on"' if li["d"] == d else ""
        cases = ""
        for ca in li["cases"]:
            if not ca["utile"]:
                continue
            col = ca["colonne"]
            sel = {"d": li["d"], **({"n": int(col[1:])} if mode == "barres" else {"espacement": col})}
            cls = " ".join(k for k, v in (("cur", ca["retenue"]), ("opt", ca["optimale"]), ("ok", ca["suffit"] and not ca["retenue"] and not ca["optimale"])) if v)
            txt = C.fr(ca["arrondi"], 0) + ('<i class="mk">★</i>' if ca["optimale"] else "")
            cases += _td(txt, sel, cls)
        corps += (f'<tr{on}><th class="nm">{li["d"]}</th><td class="m">{C.fr(li["aire"], 1)}</td><td class="m">{C.esc(li["masse_txt"])}</td>{cases}</tr>')
    tableau = _tableau(th1 + th2, corps, "bib-sections")
    avert = [a["message"] for a in r["avertissements"]]
    termes = [{"symbole": t["symbole"], "nom": NOMS_FORMULES.get(t["cle"], t["cle"]), "valeur": t["resultat"], "unite": t["unite"],
               "regle": t["expression"] + " ; " + " ; ".join(f"{x['symbole']} = {C.nb(x['valeur'], 2)} {x['unite']}".rstrip() for x in t["termes"])} for t in ret["formules"]]
    termes.insert(0, {"symbole": "A_s", "nom": "SECTION DE LA DISPOSITION RETENUE", "valeur": ret["As"], "unite": unite, "regle": ret["nom"]})
    if opt:
        termes.append({"symbole": "A_s,opt", "nom": "la plus économique qui suffit", "valeur": opt["As"], "unite": unite, "regle": f"{opt['nom']} pour {C.fr(as_req, 0)} {unite} requis"})
    sources = [f"{AT.LEGENDE_IMAGE} — tableau du bureau recalculé (ρ = {C.fr(tab['masse_volumique'], 0)} kg/m³)"]
    phrase = C.esc(ret["nom"]) + (f" — la plus économique qui suffit pour {C.fr(as_req, 0)} {unite} : <b>{C.esc(opt['nom'])}</b> ★" if opt else "")
    return {"titre": "Sections et masses", "barre": barre, "tableau": tableau,
            "bande": _bande(C.fr(ret["As"], 0), unite, "A_s", "section de la disposition retenue", phrase, hote, avert),
            "pourquoi": _pourquoi(C.esc(ret["nom"]), "Tableau des sections d'armatures du bureau", termes, sources, avert),
            "sources": _sources(*sources),
            "fiche": _fiche(f"Armatures {ret['nom']}", "A_s", ret["As"], unite, p, termes, sources, None, avert, texte=C.fr(ret["As"], 0), phrase=phrase),
            "params": p, "message": L.message()}


# ================================================================================================ 4 · enrobages
def _r_enrobages(params: Dict, hote: str) -> Dict:
    """La rubrique « Enrobages » et ses deux onglets (« Calcul », « Tableaux ») : api_web/biblio_enrobages.py (importé
    ici, à l'appel : il prend ses briques dans ce fichier)."""
    from api_web import biblio_enrobages
    return biblio_enrobages.rendre(params, hote)


# ================================================================================================ 5 · ancrages
GRANDEURS = (("lbd", "l_bd · ancrage"), ("l0", "l_0 · recouvrement"))
RHO1 = ((20.0, "< 25 %"), (33.0, "33 %"), (50.0, "50 %"), (100.0, "> 50 %"))


def _r_ancrages(params: Dict, hote: str) -> Dict:
    L = Lecture(params)
    grandeur = L.choix("grandeur", tuple(k for k, _ in GRANDEURS), "lbd")
    phi = L.choix("phi", AN.DIAMETRES, 16)
    classe = L.choix("classe", AN.CLASSES, "C30/37")
    fyk = L.choix("fyk", AN.NUANCES, 500)
    cond = L.choix("conditions", tuple(k for k, _ in AN.CONDITIONS), "bonnes")
    soll = L.choix("sollicitation", tuple(k for k, _ in AN.SOLLICITATIONS), "traction")
    forme = L.choix("forme", tuple(k for k, _ in AN.FORMES), "droit")
    c_d = L.nombre("c_d", None, 0, 1000, vide=None)
    soude = L.case("soude", False)
    taux = L.nombre("taux", 1.0, 0.0, 1.0)
    rho1 = L.choix("rho1", tuple(r for r, _ in RHO1), 100.0)
    pres = L.nombre("p", 0.0, 0, 100)
    p = L.out
    res = AN.calculer({"phi": phi, "classe": classe, "fyk": fyk, "conditions": cond, "sollicitation": soll, "forme": forme, "c_d": c_d,
                       "soude": soude, "taux": taux, "rho1": rho1, "p": pres})
    t = AN.tableau(grandeur, cond, fyk, rho1)
    barre = _barre(
        _champ("grandeur", "Longueur", _segments("grandeur", grandeur, [(k, C.symb_math(v)) for k, v in GRANDEURS]))
        + _champ("phi", "Ø", _select("phi", phi, [(x, f"Ø {x}") for x in AN.DIAMETRES]))
        + _champ("classe", "Béton", _select("classe", classe, [(c, c) for c in AN.CLASSES]))
        + _champ("fyk", "Acier", _segments("fyk", fyk, [(n, f"B{n}") for n in AN.NUANCES]))
        + _champ("conditions", "Adhérence", _segments("conditions", cond, (("bonnes", "bonnes"), ("mediocres", "autres"))), dict(AN.CONDITIONS)["mediocres"])
        + (_champ("rho1", "Recouvert", _select("rho1", rho1, RHO1), "proportion de barres recouvertes dans la section (α6, tableau 8.3)") if grandeur == "l0" else ""),
        _champ("sollicitation", "Barre", _segments("sollicitation", soll, (("traction", "tendue"), ("compression", "comprimée"))))
        + _champ("forme", "Ancrage", _segments("forme", forme, (("droit", "droit"), ("courbe", "crochet · coude"))))
        + _champ("c_d", "c<sub>d</sub>", _num("c_d", c_d, "mm", "—"), "enrobage ou demi-entraxe des barres (figure 8.3) : α1 et α2")
        + _champ("taux", "σ<sub>sd</sub> / f<sub>yd</sub>", _num("taux", taux), "la contrainte de la barre à l'origine de l'ancrage rapportée à f_yd (ou A_s,req / A_s,prov)")
        + _champ("p", "Pression p", _num("p", pres, "MPa"), "pression transversale le long de l'ancrage : α5 = 1 − 0,04 p")
        + _champ("soude", "", _case("soude", soude, "Barre transversale soudée", "α4 = 0,7 (§8.6)")))
    # le tableau Ø × classe de la grandeur, la case [Ø ; classe] allumée ; la ligne des coefficients l/Ø
    th = ('<tr><th class="nm">Ø<span class="u">mm</span></th>' + "".join(f'<th class="{"on" if c == classe else ""}"><span class="s">{c}</span><span class="u">{C.fr(t["coefficients"][c], 1)} Ø</span></th>' for c in t["classes"]) + "</tr>")
    corps = ""
    for d in t["diametres"]:
        on = ' class="on"' if d == phi else ""
        cases = "".join(_td(_mm(t["cellules"][d][c]["cm"]), {"phi": d, "classe": c}, "cur" if (d == phi and c == classe) else ("col" if c == classe else ""),
                            f"{C.fr(t['cellules'][d][c]['mm'], 0)} mm calculés ({C.fr(t['cellules'][d][c]['k'], 1)} Ø)") for c in t["classes"])
        gros = ' <small title="gros diamètre (§8.8) : recouvrement sous conditions, tableau approximatif au-delà de Ø 32">gros Ø</small>' if d > AN.PHI_GRANDE else ""
        corps += f'<tr{on}><th class="nm">{d}{gros}</th>{cases}</tr>'
    tableau = _tableau(th, corps, "bib-anc")
    val = res[grandeur]["valeur"]
    case_mm = t["cellules"][phi][classe]["cm"]
    sym, nom = ("l_bd", "longueur d'ancrage de calcul") if grandeur == "lbd" else ("l_0", "longueur de recouvrement de calcul")
    avert = [a["message"] for a in res["avertissements"]]
    reduit = abs(val - t["cellules"][phi][classe]["mm"]) > 1e-9
    phrase = (f"Ø {phi} · {classe} · B{fyk} · {'bonnes' if cond == 'bonnes' else 'autres'} conditions · barre {'tendue' if soll == 'traction' else 'comprimée'}, "
              f"{'droite' if forme == 'droit' else 'crochet ou coude'}" + (f" · σ<sub>sd</sub> = {C.nb(taux, 2)} f<sub>yd</sub>" if taux < 1 else "")
              + (f" · ρ<sub>1</sub> {dict(RHO1)[rho1]}" if grandeur == "l0" else "")
              + (f" — la case du tableau ({_mm(case_mm)} mm) vaut pour une barre droite tendue pleinement sollicitée, sans réduction" if reduit else " · arrondi au cm supérieur"))
    sources = [AN.SOURCE, "Contrôle : " + AN.CONTROLE]
    extrait = _extrait(["Ø"] + list(t["classes"]), [[str(d)] + [_mm(t["cellules"][d][c]["cm"]) for c in t["classes"]] for d in t["diametres"]],
                       (t["diametres"].index(phi), t["classes"].index(classe) + 1),
                       f"{'l_bd' if grandeur == 'lbd' else 'l_0'} (mm) — B{fyk}, {'bonnes' if cond == 'bonnes' else 'autres'} conditions d'adhérence, arrondi au cm supérieur", AN.SOURCE)
    return {"titre": "Ancrages et recouvrements", "barre": barre, "tableau": tableau,
            "bande": _bande(_mm(AN.arrondi_cm(val)), "mm", sym, nom + f" ({C.fr(val, 0)} mm calculés)", phrase, hote, avert),
            "pourquoi": _pourquoi(f"{C.symb_math(sym)} · Ø {phi} {classe} B{fyk}", "EN 1992-1-1 §8.4 et §8.7.3", res["termes"], sources, avert, nd=3),
            "sources": _sources(*sources),
            "fiche": _fiche(f"{'Ancrage' if grandeur == 'lbd' else 'Recouvrement'} Ø {phi} {classe} B{fyk}", sym, val, "mm", p, res["termes"], sources, extrait, avert,
                            texte=_mm(AN.arrondi_cm(val)), phrase=phrase, nd=3),
            "params": p, "message": L.message()}


# ================================================================================================ la porte
RENDUS = {"betons": _r_betons, "aciers": _r_aciers, "sections": _r_sections, "enrobages": _r_enrobages, "ancrages": _r_ancrages}


def consulter(rubrique: Optional[str], params: Optional[Dict], hote: str = "page") -> Dict:
    """La rubrique rendue pour ces paramètres (les absents prennent leur défaut, les illisibles aussi — ``message`` le
    dit) ; une rubrique inconnue ouvre la première. ``hote`` : « page » ou « note » (les boutons de la bande)."""
    if hote not in HOTES:
        raise ValueError(f"hôte inconnu : {hote!r} (page ou note)")
    r = rubrique if rubrique in RENDUS else RUBRIQUE_DEFAUT
    out = RENDUS[r](params, hote)
    out["rubrique"] = r
    out["bibliotheque"] = BIBLIOTHEQUE
    if rubrique is not None and rubrique != r:
        out["message"] = " ; ".join(m for m in (f"rubrique « {str(rubrique)[:40]} » inconnue : « {NOMS[r]} » ouverte", out.get("message")) if m)
    return out


def menu(actif: str = RUBRIQUE_DEFAUT) -> str:
    """Le menu des rubriques (la page le porte en dur, js/bibliotheque.js allume l'entrée ouverte)."""
    return "".join(f'<button type="button" class="bib-m{" on" if k == actif else ""}" data-rubrique="{k}" aria-pressed="{"true" if k == actif else "false"}">'
                   f'{C.icon(ic)}<span class="bib-mt">{t}</span><span class="bib-ms">{st}</span></button>' for k, t, st, ic in RUBRIQUES)

# -*- coding: utf-8 -*-
"""Le module « Tableau des armatures » du site : le tableau des sections d'acier du bureau (l'image que l'écran
Streamlit modules/tableau_armatures.py affiche), et ce que l'image ne fait pas — une section requise surligne les
dispositions qui suffisent et propose la plus économique ; une disposition (Ø et nombre de barres, ou Ø et
espacement) donne sa section, sa masse et son entraxe, avec le détail des formules.

Module de CONSULTATION : pas de verdict de la note (statut « Consultation ») ; les taux A_s,req / A_s colorent les
lignes. Une note porte plusieurs REPÈRES (une nappe de dalle, les barres d'une poutre…), chacun avec son besoin.

Les nombres viennent du moteur ``beton/armatures`` (preuve de parité avec l'image : tests/test_armatures_moteur.py) ;
ce fichier les MET EN FORME : il ne calcule rien.
"""
import copy
import re
from typing import Dict, List, Optional, Sequence

from beton.armatures import dessins as DS
from beton.armatures import moteur as MO
from beton.armatures import tableau as T

from . import commun as C

MODULE = "armatures"
PAGE = "tableau-armatures"

ENTIERS = {"d", "n", "d_min", "d_max", "n_min", "n_max"}
REELS = {"As_req", "e_libre", "e_min", "e_max"}
TEXTES = {"nom", "mode", "espacement"}
U = {"barres": "mm²", "nappe": "mm²/m"}


# ------------------------------------------------------------------ l'état de la note : des repères
def nouvel_etat() -> Dict:
    return {"version": 1, "reperes": [MO.nouveau_besoin("Repère 1")], "actif": 0}


def normaliser(etat: Dict) -> Dict:
    """L'état reçu de l'écran (ou d'un fichier .stab) relu par le moteur ; un repère illisible lève une erreur claire."""
    reps = etat.get("reperes") if isinstance(etat, dict) else None
    if reps is None or reps == []:
        return nouvel_etat()
    if not isinstance(reps, list):
        raise ValueError("note abîmée : « reperes » doit être une liste")
    out = {"version": 1, "reperes": [MO.normaliser(r) for r in reps]}
    try:
        actif = int(etat.get("actif", 0) or 0)
    except (TypeError, ValueError):
        actif = 0
    out["actif"] = min(max(0, actif), len(out["reperes"]) - 1)
    return out


def _rep(etat: Dict) -> Dict:
    return etat["reperes"][etat["actif"]]


def gerer_elements(etat: Dict, action: str, index: Optional[int] = None) -> None:
    """Les repères de la note : en ouvrir un, en ajouter, copier l'ouvert, en supprimer un (jamais le dernier)."""
    rs = etat["reperes"]
    if action == "activer":
        etat["actif"] = min(max(0, int(index or 0)), len(rs) - 1)
    elif action == "ajouter":
        pris = {r["nom"] for r in rs}
        k = len(rs) + 1
        while f"Repère {k}" in pris:
            k += 1
        rs.append(MO.nouveau_besoin(f"Repère {k}"))
        etat["actif"] = len(rs) - 1
    elif action == "dupliquer":
        r = copy.deepcopy(rs[etat["actif"]])
        r["nom"] = f"{r.get('nom') or 'Repère'} (copie)"
        rs.append(r)
        etat["actif"] = len(rs) - 1
    elif action == "supprimer" and len(rs) > 1:
        i = min(max(0, int(index or 0)), len(rs) - 1)
        del rs[i]
        if etat["actif"] > i or etat["actif"] >= len(rs):
            etat["actif"] = max(0, etat["actif"] - 1)


# ------------------------------------------------------------------ la saisie
def _retenir(r: Dict, x: Dict) -> None:
    """Retient la disposition ``x`` du moteur dans le repère ``r``."""
    r["d"] = x["d"]
    if x["mode"] == "barres":
        r["n"] = x["n"]
    else:
        r["espacement"] = x["espacement"]


def modifier(etat: Dict, chemin: Sequence, valeur) -> Optional[str]:
    """Écrit une saisie dans le repère ouvert. ``chemin`` : la clé du champ (« As_req », « d »…) ; « optimal » retient la
    disposition la plus économique. Le moteur relit le repère : une valeur hors de son domaine est refusée avec un
    message, et le repère reste intact (rien n'est ramené en silence à une autre valeur)."""
    r = _rep(etat)
    avant = copy.deepcopy(r)
    k = chemin[-1] if chemin else ""
    if len(chemin) != 1 or k not in ENTIERS | REELS | TEXTES | {"optimal"}:
        return f"Champ inconnu : « {'/'.join(map(str, chemin))} »"
    try:
        if k == "optimal":
            opt = MO.optimale(MO.normaliser(r))
            if opt is None:
                return "Aucune disposition à proposer : saisir une section requise, ou élargir les critères de recherche."
            _retenir(r, opt)
        elif k in TEXTES:
            r[k] = str(valeur if valeur is not None else "")
        else:
            x = C.lire_nombre(valeur)
            if x is None:
                return f"Nombre illisible : « {valeur} »"
            if k in ENTIERS:
                if x != int(x):
                    return f"Un nombre entier est attendu : « {valeur} »"
                x = int(x)
            r[k] = x
        etat["reperes"][etat["actif"]] = MO.normaliser(r)
    except (ValueError, KeyError, TypeError) as e:
        etat["reperes"][etat["actif"]] = avant
        return f"Saisie refusée : {e}"
    return None


_BARRES = re.compile(r"^(\d+)Ø(\d+)$")
_NAPPE = re.compile(r"^Ø(\d+) · ([\d,]+) cm$")


def choisir_section(etat: Dict, role: str, nom: str) -> None:
    """Le sélecteur des dispositions : « 4Ø12 » (barres) ou « Ø10 · 15 cm » (nappe) devient la disposition retenue."""
    r = _rep(etat)
    m = _BARRES.match(nom or "")
    if m:
        r.update(mode="barres", n=int(m.group(1)), d=int(m.group(2)))
    else:
        m = _NAPPE.match(nom or "")
        if not m:
            raise ValueError(f"disposition illisible : {nom!r}")
        d, lib = int(m.group(1)), m.group(2)
        cle = next((x["cle"] for x in MO.ESPACEMENTS if x["libelle"] == lib), None)
        r.update(mode="nappe", d=d)
        if cle:
            r["espacement"] = cle
        else:
            r.update(espacement=MO.LIBRE, e_libre=float(lib.replace(",", ".")))
    etat["reperes"][etat["actif"]] = MO.normaliser(r)


# ------------------------------------------------------------------ les nombres
def _mm2(x: float) -> str:
    """Une section comme le tableau l'écrit : au mm² (la moitié vers le haut), espace fine des milliers."""
    return C.fr(T.arrondi(x), 0)


def _e(e: float) -> str:
    """Un espacement en cm : celui d'une colonne comme l'en-tête du tableau (« 15 », « 16,7 »), un autre au centième."""
    return MO.libelle_espacement(e)


def _req(x: float) -> str:
    """Une section requise, écrite telle que saisie (C.nb), les milliers séparés comme dans le tableau : « 1 250,5 »."""
    ent, _, dec = C.nb(x).partition(",")
    return f"{int(ent):,}".replace(",", "\u202f") + ("," + dec if dec else "")


def _As(x: Dict, req: float) -> str:
    """La section d'une disposition face au besoin : au mm² comme le tableau, au dixième quand l'arrondi tromperait la
    comparaison (1Ø40 = 1 256,6 mm² face à 1 257 mm² requis)."""
    return MO.section_txt(x["As"], req) if req > 0 else _mm2(x["As"])


TAUX_MAX = 9.99     # au-delà de 999 %, un taux n'apprend plus rien : il s'écrit « > 999 % »


def _pct_txt(t: float) -> str:
    """Un taux A_s,req / A_s écrit en texte : « 88 % », « > 999 % » (un « > » se lit tel quel en HTML, et C.esc
    l'échappe dans un attribut)."""
    return f">{C.NBSP}999{C.NBSP}%" if t > TAUX_MAX else f"{C.pc(C.pct(t))}{C.NBSP}%"


def _v(x: float, nd: int) -> str:
    return C.fr(x, nd)


def _niv(res: Dict, x: Dict) -> str:
    """La couleur d'une disposition : le taux A_s,req / A_s ; grise sans section requise."""
    return C.niveau(x["taux"]) if res["besoin"]["As_req"] > 0 else "c"


def _taux(res: Dict, x: Dict):
    return C.pct(min(x["taux"], TAUX_MAX)) if res["besoin"]["As_req"] > 0 else None


def _meme(a: Dict, b: Dict) -> bool:
    """Deux dispositions identiques : même Ø, même case du tableau."""
    return a["d"] == b["d"] and a["dans_tableau"] and b["dans_tableau"] and a["colonne"] == b["colonne"]


def _piece(x: Dict, req: float = 0.0) -> str:
    return f"{C.esc(x['nom'])} · {_As(x, req)} {x['unite']}"


# ------------------------------------------------------------------ le cadre Données
def _options_espacement(r: Dict) -> List:
    opts = []
    for x in MO.ESPACEMENTS:
        k = x["barres_par_metre"]
        opts.append((x["cle"], f"{x['libelle']} cm · {k} barre{'s' if k > 1 else ''}/m" if k else f"{x['libelle']} cm"))
    return opts + [(MO.LIBRE, "autre espacement…")]


def formulaire(etat: Dict, res: Dict) -> str:
    r = _rep(etat)
    L = C.ligne
    mode = r["mode"]
    u = U[mode]
    dsp = res["disposition"]
    opt = res["optimale"]
    # 1 · le besoin
    s1 = (L("Repère", C.texte(["nom"], r["nom"], "ex. Dalle D1 · nappe inférieure x"), full=True)
          + L("Armatures", C.segments(["mode"], mode, [("barres", "Barres"), ("nappe", "Nappe")]), full=True)
          + L("requise", C.num(["As_req"], r["As_req"], u, mini=0), "A<sub>s,req</sub>",
              "La section d'acier à placer, issue du calcul (poutre, dalle, voile…) ; 0 : pas de recherche, le tableau seul.")
          + C.aide("Des <b>barres</b> se comptent (section en mm²) ; une <b>nappe</b> se lit par mètre (mm²/m), "
                   "à espacement constant."))
    if r["As_req"] > 0:
        s1 += C.aide(f"{res['suffisantes']} disposition{'s' if res['suffisantes'] > 1 else ''} du tableau suffi"
                     f"{'sent' if res['suffisantes'] > 1 else 't'} dans les critères ; elles sont surlignées en vert.")
    resume1 = f"{C.esc(r['nom'])} · " + (f"A<sub>s,req</sub> {_req(r['As_req'])} {u}" if r["As_req"] > 0 else "consultation")
    # 2 · la disposition retenue
    s2 = L("Diamètre", C.choix(["d"], r["d"], [(d, f"Ø{d} · {_mm2(T.aire(d))} mm²") for d in T.DIAMETRES]), full=True)
    if mode == "barres":
        s2 += L("nombre de barres", C.num(["n"], r["n"], "barres", entier=True, mini=1, maxi=MO.N_MAX_SAISIE), "n",
                "De 1 à 10, la disposition est une case du tableau ; au-delà, même formule.")
    else:
        s2 += L("Espacement", C.choix(["espacement"], r["espacement"], _options_espacement(r)), full=True)
        if r["espacement"] == MO.LIBRE:
            s2 += L("espacement", C.num(["e_libre"], r["e_libre"], "cm", mini=0), "e",
                    "Un espacement hors des colonnes du tableau : même formule a·100/e.")
    s2 += C.aide(f"<b>{C.esc(dsp['nom'])}</b> = {_As(dsp, r['As_req'])} {u} · "
                 + (f"{_v(dsp['masse'], 3)} kg/m" if mode == "barres" else f"{_v(dsp['masse'], 2)} kg/m²")
                 + (f" · e = {_e(dsp['e'])} cm (lu par mètre)" if mode == "barres" else f" · {_v(dsp['barres_par_metre'], 2)} barres/m"))
    if opt and not _meme(opt, dsp):
        s2 += (f'<button type="button" class="addrow" data-chemin="optimal" data-valeur="1">{C.icon("star")}'
               f'Retenir la plus économique : {C.esc(opt["nom"])}</button>')
    resume2 = f"{C.esc(dsp['nom'])} · {_mm2(dsp['As'])} {u}"
    # 3 · les critères de la recherche
    ds = [(d, f"Ø{d}") for d in T.DIAMETRES]
    s3 = (L("Ø mini", C.choix(["d_min"], r["d_min"], ds), full=True)
          + L("Ø maxi", C.choix(["d_max"], r["d_max"], ds), full=True))
    if mode == "barres":
        s3 += (L("barres, au moins", C.num(["n_min"], r["n_min"], "barres", entier=True, mini=1, maxi=10), "n<sub>min</sub>")
               + L("barres, au plus", C.num(["n_max"], r["n_max"], "barres", entier=True, mini=1, maxi=10), "n<sub>max</sub>"))
        resume3 = f"Ø{r['d_min']}–{r['d_max']} · {r['n_min']}–{r['n_max']} barres"
    else:
        s3 += (L("espacement, au moins", C.num(["e_min"], r["e_min"], "cm", mini=0), "e<sub>min</sub>")
               + L("espacement, au plus", C.num(["e_max"], r["e_max"], "cm", mini=0), "e<sub>max</sub>",
                   "40 cm : le plus grand espacement fixe du tableau."))
        resume3 = f"Ø{r['d_min']}–{r['d_max']} · e {C.nb(r['e_min'])}–{C.nb(r['e_max'])} cm"
    s3 += C.aide("La <b>plus économique</b> : la plus petite section qui suffit, parmi les cases du tableau dans ces "
                 "critères ; à section égale, le moins de barres, puis le plus petit Ø.")
    # 4 · le tableau du bureau (rien à saisir : sa source et ses formules)
    s4 = C.aide(f"Les valeurs sont celles du tableau du bureau (« {C.esc(T.IMAGE)} », l'écran Streamlit), relues au "
                "chiffre près : a = π·Ø²/4 ; n barres : n·a (mm²), ou n barres par mètre, e = 100/n cm (mm²/m) ; "
                "espacement fixe e : a·100/e (mm²/m) ; arrondi au mm².")
    s4 += C.aide(f"Masse linéique : ρ·a avec <b>ρ = {C.fr(T.MASSE_VOLUMIQUE, 0)} kg/m³</b>, la valeur du tableau "
                 "(l'usage courant est 7 850 kg/m³ : les masses du tableau sont plus fortes de 0,13 %).")
    return (C.section(1, "Besoin", resume1, s1, True, "be")
            + C.section(2, "Disposition retenue", resume2, s2, True, "di")
            + C.section(3, "Critères de recherche", resume3, s3, False, "cr")
            + C.section(4, "Le tableau du bureau", f"ρ = {C.fr(T.MASSE_VOLUMIQUE, 0)} kg/m³", s4, False, "ta"))


# ------------------------------------------------------------------ la fenêtre de calcul
def _lignes_detail(res: Dict, x: Dict) -> str:
    """Le détail des formules d'une disposition, « symbole = expression = valeurs = résultat »."""
    t = {f["cle"]: f for f in x["formules"]}
    a = t["a"]["resultat"]
    m = t["m"]["resultat"]
    barre = [{"parts": ["a", "π·Ø²/4", f"π × {x['d']}² / 4"], "res": f"{_v(a, 1)} mm²", "niveau": ""},
             {"parts": ["m", "ρ·a·10⁻⁶", f"{C.fr(T.MASSE_VOLUMIQUE, 0)} × {_v(a, 1)} × 10⁻⁶"], "res": f"{_v(m, 3)} kg/m", "niveau": ""}]
    u = x["unite"]
    if x["mode"] == "barres":
        n = x["n"]
        dispo = [{"parts": ["A_s", "n·a", f"{n} × {_v(a, 1)}"], "res": f"{_mm2(x['As'])} {u}", "niveau": ""},
                 {"parts": ["M", "n·m", f"{n} × {_v(m, 3)}"], "res": f"{_v(x['masse'], 3)} kg/m", "niveau": ""},
                 {"parts": ["e", "100 / n", f"100 / {n}"], "res": f"{_v(x['e'], 1)} cm (lu par mètre)", "niveau": ""}]
        titre = f"{n} barres Ø{x['d']}"
    else:
        if "e" in t and t["e"]["termes"][0]["symbole"] == "n":
            k = t["e"]["termes"][0]["valeur"]
            dispo = [{"parts": ["e", "100 / n", f"100 / {k}"], "res": f"{_v(x['e'], 1)} cm", "niveau": ""},
                     {"parts": ["A_s", "n·a", f"{k} × {_v(a, 1)}"], "res": f"{_mm2(x['As'])} {u}", "niveau": ""}]
        else:
            dispo = [{"parts": ["n", "100 / e", f"100 / {C.nb(x['e'])}"], "res": f"{_v(x['barres_par_metre'], 2)} barres/m", "niveau": ""},
                     {"parts": ["A_s", "a·100 / e", f"{_v(a, 1)} × 100 / {C.nb(x['e'])}"], "res": f"{_mm2(x['As'])} {u}", "niveau": ""}]
        dispo.append({"parts": ["M", "n·m", f"{_v(x['barres_par_metre'], 2)} × {_v(m, 3)}"], "res": f"{_v(x['masse'], 2)} kg/m²", "niveau": ""})
        titre = f"Nappe Ø{x['d']}, par mètre"
    out = (f'<div class="ck-g"><div class="ck-t">Une barre Ø{x["d"]}</div>{C.lignes_fx(barre)}</div>'
           f'<div class="ck-g"><div class="ck-t">{C.esc(titre)}</div>{C.lignes_fx(dispo)}</div>')
    if "taux" in t:
        req = res["besoin"]["As_req"]
        cmp_ = [{"parts": ["η", "A_s,req / A_s", f"{_req(req)} / {_As(x, req)}"], "res": _pct_txt(x["taux"]),
                 "niveau": _niv(res, x)}]
        out += f'<div class="ck-g"><div class="ck-t">Section requise</div>{C.lignes_fx(cmp_)}</div>'
    return out


def _fenetre(ident: str, titre: str, res: Dict, x: Dict, gov: bool) -> str:
    b = res["besoin"]
    u = x["unite"]
    if x["mode"] == "barres":
        f = "A<sub>s,req</sub> ≤ A<sub>s</sub> = n·π·Ø²/4"
    elif x["barres_par_metre"] == int(x["barres_par_metre"]) and x.get("espacement", "").startswith("n"):
        f = "A<sub>s,req</sub> ≤ A<sub>s</sub> = n·π·Ø²/4, n = 100/e"
    else:
        f = "A<sub>s,req</sub> ≤ A<sub>s</sub> = π·Ø²/4 · 100/e"
    if b["As_req"] > 0:
        val, lim = f"A<sub>s,req</sub> = {_req(b['As_req'])} {u}", f"A<sub>s</sub> = {_As(x, b['As_req'])} {u}"
        taux, niv = C.pct(min(x["taux"], TAUX_MAX)), _niv(res, x)
    else:
        val, lim, taux, niv = f"A<sub>s</sub> = {_mm2(x['As'])} {u}", "pas de section requise", 0, "c"
    sous = f"{C.esc(x['nom'])} · {'case du tableau' if x['dans_tableau'] else 'hors du tableau'}"
    lignes = [{"nom": "Section", "sous": sous, "formule": f, "valeur": val, "limite": lim, "taux": taux, "niveau": niv, "gov": gov}]
    hyps = [f"Tableau du bureau · ρ = <b>{C.fr(T.MASSE_VOLUMIQUE, 0)} kg/m³</b>",
            f"{'Barres' if x['mode'] == 'barres' else 'Nappe par mètre'} · <b>{C.esc(x['nom'])}</b>",
            f"Sections arrondies au mm² · masses au gramme"]
    lu = (f"lu par mètre : {x['n']} barre{'s' if x['n'] > 1 else ''}/m, e = {_e(x['e'])} cm" if x["mode"] == "barres"
          else f"{_v(x['barres_par_metre'], 2)} barres/m · jour e − Ø = {C.fr(x['jour'], 0)} mm")
    return C.fenetre_calcul(ident, titre, f"{C.esc(x['nom'])} = {_mm2(x['As'])} {u} · {lu}", hyps, lignes, _lignes_detail(res, x))


# ------------------------------------------------------------------ la vue Tableau (un bloc HTML dans la toile)
def _cellule(res: Dict, li: Dict, c: Dict) -> str:
    b = res["besoin"]
    d = li["d"]
    col = c["colonne"]
    cls = []
    if not c["utile"]:
        cls.append("hs")
    elif not c["dans_criteres"]:
        cls.append("hc")
    if c["suffit"]:
        cls.append("ok")
    if c["optimale"]:
        cls.append("opt")
    if c["retenue"]:
        cls.append("cur")
    if col.startswith("n"):
        n = int(col[1:])
        nom = f"{n}Ø{d}" if b["mode"] == "barres" else f"Ø{d} · {_e(T.espacement(n))} cm ({n} barre{'s' if n > 1 else ''}/m)"
        u = U[b["mode"]]
    else:
        nom, u = f"Ø{d} · {col[1:]} cm", "mm²/m"
    titre = f"{nom} = {C.fr(c['valeur'], 1)} {u}"
    if b["As_req"] > 0 and c["utile"]:
        titre += f" · {_pct_txt(b['As_req'] / c['valeur'])}" + (" · suffit" if c["suffit"] else " · ne suffit pas")
    if c["optimale"]:
        titre += " · la plus économique"
    if c["retenue"]:
        titre += " · retenue"
    if not c["utile"]:
        titre += " · colonne d'une nappe (mode Nappe)"
    elif not c["dans_criteres"]:
        titre += " · hors des critères de recherche"
    marque = '<i class="mk">★</i>' if c["optimale"] else ('<i class="mk">✓</i>' if c["retenue"] else "")
    return f'<td class="{" ".join(cls)}" title="{C.esc(titre)}">{marque}{C.fr(c["arrondi"], 0)}</td>'


def vue_tableau(res: Dict) -> str:
    """Le tableau de l'image, au chiffre près, avec ses deux lectures (n barres en mm² ; n barres par mètre et
    espacements fixes en mm²/m) ; les cases qui suffisent, la plus économique (★) et la retenue (✓) marquées."""
    tab = res["tableau"]
    b = res["besoin"]
    dsp, opt = res["disposition"], res["optimale"]
    u = U[b["mode"]]
    cur_col = dsp["colonne"] if dsp["dans_tableau"] else None
    # la légende : ce qui est surligné
    if b["As_req"] > 0:
        leg = (f'<span class="lg ok"><i></i>{res["suffisantes"]} case{"s" if res["suffisantes"] > 1 else ""} suffi{"sent" if res["suffisantes"] > 1 else "t"} · ≥ {_req(b["As_req"])} {u}</span>'
               + (f'<span class="lg opt"><i>★</i>la plus économique · {C.esc(opt["nom"])} = {_mm2(opt["As"])} {u}</span>' if opt else
                  '<span class="lg">aucune case ne suffit dans les critères</span>'))
    else:
        leg = '<span class="lg">Saisir une section requise : les cases qui suffisent se surlignent.</span>'
    leg += f'<span class="lg cur"><i>✓</i>retenue · {C.esc(dsp["nom"])}{"" if dsp["dans_tableau"] else " (hors du tableau)"}</span>'
    lecture = ("Barres : la section de n barres, en mm² ; la masse d'une barre, en kg/m." if b["mode"] == "barres"
               else "Nappe : n barres par mètre (e = 100/n cm) et espacements fixes, en mm²/m ; la masse d'une barre, en kg/m.")
    # l'en-tête sur deux rangées, comme l'image
    h1 = (f'<th rowspan="2" class="d">Ø<small>mm</small></th>'
          f'<th colspan="{len(T.NOMBRES)}" class="g{" on" if b["mode"] == "barres" else ""}">Nombre de barres <small>mm²</small>'
          f'<span class="ou">ou barres par mètre · e en cm <small>mm²/m</small></span></th>'
          f'<th rowspan="2" class="m">Masse<small>kg/m</small></th>'
          f'<th colspan="{len(T.ESPACEMENTS_FIXES)}" class="g{" on" if b["mode"] == "nappe" else " hs"}">Espacement fixe<small>mm²/m</small></th>')
    h2 = ""
    for n, et in zip(T.NOMBRES, tab["espacements_n_txt"]):
        on = " cur" if cur_col == f"n{n}" else ""
        h2 += f'<th class="n{on}"><b>{n}</b><small>{et}</small></th>'
    for e in T.ESPACEMENTS_FIXES:
        on = " cur" if cur_col == f"e{e}" else ""
        h2 += f'<th class="n{on}{"" if b["mode"] == "nappe" else " hs"}"><b>{e}</b><small>cm</small></th>'
    corps = ""
    for li in tab["lignes"]:
        d = li["d"]
        cases = li["cases"]
        nb_n = len(T.NOMBRES)
        cells = "".join(_cellule(res, li, c) for c in cases[:nb_n])
        cells += f'<td class="m">{li["masse_txt"]}</td>'
        cells += "".join(_cellule(res, li, c) for c in cases[nb_n:])
        hors = "" if b["d_min"] <= d <= b["d_max"] else " hc"
        corps += f'<tr class="{"cur" if d == dsp["d"] else ""}"><th class="d{hors}">{d}</th>{cells}</tr>'
    return (f'<div class="arm-vue" id="arm-tableau"><div class="arm-leg"><span class="lect">{lecture}</span>{leg}</div>'
            f'<div class="arm-defile"><table class="arm-tab"><thead><tr class="r1">{h1}</tr><tr class="r2">{h2}</tr></thead>'
            f'<tbody>{corps}</tbody></table></div>'
            f'<div class="arm-pied"><b>{C.esc(T.LEGENDE_IMAGE)}.</b> Sections arrondies au mm² ; masse linéique ρ·π·Ø²/4 avec ρ = {C.fr(T.MASSE_VOLUMIQUE, 0)} kg/m³ '
            f'(tableau du bureau). Survoler une case : sa valeur exacte et son taux.</div></div>')


# ------------------------------------------------------------------ le résultat
def _fiche(res: Dict, x: Dict) -> str:
    """Les propriétés de la disposition retenue (consultation) : section, masse, entraxe, la barre seule."""
    u = x["unite"]
    if x["mode"] == "barres":
        items = [("Section", f"{_mm2(x['As'])} {u}"), ("Masse des barres", f"{_v(x['masse'], 3)} kg/m"),
                 ("Lues par mètre", f"{x['n']} barre{'s' if x['n'] > 1 else ''}/m · e = {_e(x['e'])} cm")]
    else:
        items = [("Section par mètre", f"{_mm2(x['As'])} {u}"), ("Masse de la nappe", f"{_v(x['masse'], 2)} kg/m²"),
                 ("Barres par mètre", f"{_v(x['barres_par_metre'], 2)} · e = {_e(x['e'])} cm"),
                 ("Jour entre barres", f"e − Ø = {C.fr(x['jour'], 0)} mm")]
    items += [(f"Une barre Ø{x['d']}", f"{_v(x['aire'], 1)} mm² · {T.masse_txt(x['masse_barre'])} kg/m")]
    return '<div class="arm-fiche">' + "".join(f'<div><span>{k}</span><b>{v}</b></div>' for k, v in items) + "</div>"


def _calcul(etat: Dict) -> List[Dict]:
    return [MO.calculer(r) for r in etat["reperes"]]


def resume_note(etat: Dict, resultats: List[Dict]) -> Dict:
    """Le résumé de la NOTE : une consultation (pas de verdict) ; le repère le plus sollicité est nommé."""
    i = max(range(len(resultats)), key=lambda k: resultats[k]["disposition"]["taux"] if resultats[k]["besoin"]["As_req"] > 0 else -1)
    r = resultats[i]
    return {"taux": None, "niveau": "i", "statut": C.STATUTS["i"][1],
            "gouvernant": f"{etat['reperes'][i].get('nom') or 'Repère'} · {r['disposition']['nom']} · {_mm2(r['disposition']['As'])} {r['disposition']['unite']}"}


def elements(etat: Dict, resultats: List[Dict]) -> str:
    items = ""
    for i, (rp, r) in enumerate(zip(etat["reperes"], resultats)):
        x = r["disposition"]
        if r["besoin"]["As_req"] > 0:
            niv, txt = C.niveau(x["taux"]), _pct_txt(x["taux"])
        else:
            niv, txt = "c", "—"
        sup = f'<button class="x" data-supprimer="{i}" title="Supprimer">{C.icon("x")}</button>' if len(etat["reperes"]) > 1 else ""
        items += (f'<div class="ni{" on" if i == etat["actif"] else ""}" data-activer="{i}"><span class="t">{C.esc(rp.get("nom") or f"Repère {i + 1}")} '
                  f'<span>· {C.esc(x["nom"])}</span></span><span class="pct {niv}">{txt}</span>{sup}</div>')
    return (f'<div class="elements"><div class="nl-h"><span class="kicker">Dans cette note · {len(etat["reperes"])}</span>'
            f'<button class="nl-add" data-dupliquer="1" title="Copier le repère ouvert">{C.icon("copy")}Copier</button></div>{items}'
            f'<div class="nl-add" data-ajouter="1">{C.icon("plus")}Ajouter un repère</div></div>')


def resultat(etat: Dict) -> Dict:
    resultats = _calcul(etat)
    res = resultats[etat["actif"]]
    b, x, opt = res["besoin"], res["disposition"], res["optimale"]
    u = x["unite"]
    req = b["As_req"] > 0
    niv = _niv(res, x)
    # l'en-tête : la disposition retenue (pas de verdict : une consultation)
    if req:
        cmp_ = "≥" if x["suffit"] else "&lt;"
        sous = (f'{cmp_} A<sub>s,req</sub> = {_req(b["As_req"])} {u}<br>taux {_pct_txt(x["taux"])} · '
                + ("suffit" if x["suffit"] else "ne suffit pas"))
    else:
        sous = "aucune section requise :<br>le tableau en consultation"
    entete = (f'<div class="verdict arm-entete"><span class="chip xl {niv}">{_As(x, b["As_req"])}<small>{u}</small></span>'
              f'<div class="gov"><b>{C.esc(b["nom"])} · {C.esc(x["nom"])}</b>{sous}</div></div>')
    leger = (f"La plus économique : {opt['nom']} · {_mm2(opt['As'])} {u} · {_pct_txt(opt['taux'])}"
             if opt and not _meme(opt, x) else "")
    rows = C.ligne_res("Retenue", _piece(x, b["As_req"]), _taux(res, x), niv, calcul="disp", pk="disp", gov=True, leger=leger)
    if opt:
        # « Économique » et non « Plus économique » : à 1 024 px, la colonne du résultat ne laisse que 167 px à
        # « ★ Ø40 · 33,3 cm · 3 770 mm²/m » (184 px, mesurés) derrière le nom long ; il en reste 197 derrière le court
        rows += C.ligne_res("Économique", f'<span class="arm-or">★</span> {_piece(opt)}', _taux(res, opt), _niv(res, opt), calcul="opt")
    rows += f'<div class="elgrp">{"Barres retenues · " + C.esc(x["nom"]) if b["mode"] == "barres" else "Nappe retenue · Ø" + str(x["d"])}</div>' + _fiche(res, x)
    messages = "".join(C.message(C.esc(a["message"]), a["code"] in ("insuffisante", "chevauchement")) for a in res["avertissements"])
    pied = (f'<div class="pfoot"><div class="mass"><span>Masse volumique du tableau</span>'
            f'<b>ρ = {C.fr(T.MASSE_VOLUMIQUE, 0)} kg/m³</b></div>'
            f'<div class="legend">{C.legende()}</div></div>')
    volets = _fenetre("disp", f"{C.esc(b['nom'])} · disposition retenue", res, x, True)
    if opt:
        volets += _fenetre("opt", f"{C.esc(b['nom'])} · la plus économique", res, opt, False)
    # la toile : le tableau, la nappe d'un mètre, les diamètres
    if b["mode"] == "barres":
        t_n = f"{x['nom']} lus par mètre · {x['n']} barre{'s' if x['n'] > 1 else ''}/m"
        p_n = f"A = {_mm2(x['As'])} mm²/m · {_v(x['masse'], 2)} kg/m²"
    else:
        t_n = f"Nappe Ø{x['d']} · e = {_e(x['e'])} cm · {_v(x['barres_par_metre'], 2)} barres/m"
        p_n = f"A = {_mm2(x['As'])} mm²/m · {_v(x['masse'], 2)} kg/m²"
    vues = [{"cle": "tableau", "titre": T.TITRE_ECRAN, "court": "Tableau", "svg": vue_tableau(res),
             "legende": T.LEGENDE_IMAGE},
            {"cle": "nappe", "titre": "Nappe d'un mètre", "court": "Par mètre", "svg": DS.nappe("arm_nappe", x["d"], x["e"], t_n, p_n),
             "legende": t_n},
            {"cle": "diametres", "titre": "Diamètres à l'échelle", "court": "Diamètres",
             "svg": DS.diametres("arm_diam", x["d"], opt["d"] if opt else None), "legende": "Ø6 à Ø40 · section et masse linéique"}]
    phrase = (f"{C.esc(b['nom'])} · {C.esc(x['nom'])} retenu"
              + (f" · A<sub>s,req</sub> = {_req(b['As_req'])} {u}" if req else "")
              + (f" · ★ {C.esc(opt['nom'])}" if opt else ""))
    return {"resultat": f'<div class="pbody">{entete}<div class="els">{rows}</div>{messages}</div>{pied}', "volets": volets, "vues": vues,
            "phrase": phrase, "statut_html": C.statut("i"), "resume": resume_note(etat, resultats), "donnees": formulaire(etat, res),
            "elements": elements(etat, resultats)}


# ------------------------------------------------------------------ le sélecteur des dispositions
def selecteur(etat: Dict, role: str, familles=None, froid: bool = True) -> str:
    """Les dispositions du tableau dans les critères, triées par section (le moins de barres d'abord à section égale),
    la retenue (✓) et la plus économique (★) marquées."""
    r = MO.normaliser(_rep(etat))
    res = {"besoin": r}
    cands = MO.candidats(r, avec_retenue=True)
    u = U[r["mode"]]
    i_o = next((k for k, c in enumerate(cands) if c["optimale"]), None)
    i_c = next((k for k, c in enumerate(cands) if c["retenue"]), 0)
    rows = []
    for c in cands:
        if r["mode"] == "barres":
            prop = (f"{c['nom']} · A<sub>s</sub> = {_As(c, r['As_req'])} mm² · {_v(c['masse'], 3)} kg/m · lues par mètre : e = {_e(c['e'])} cm")
        else:
            prop = (f"{c['nom']} · {_v(c['barres_par_metre'], 2)} barres/m · A<sub>s</sub> = {_As(c, r['As_req'])} mm²/m · {_v(c['masse'], 2)} kg/m²")
        if not c["dans_tableau"]:
            prop += " · hors du tableau"
        elif not c["dans_criteres"]:
            prop += " · hors des critères"
        rows.append({"nom": c["nom"], "col": _mm2(c["As"]), "taux": C.pct(min(c["taux"], 9.99)) if r["As_req"] > 0 else 0,
                     "niveau": _niv(res, c), "cur": c["retenue"], "opt": c["optimale"], "prop": prop})
    debut = max(0, min(i_c, i_o if i_o is not None else i_c) - 4)
    titre = "Barres · disposition" if r["mode"] == "barres" else "Nappe · disposition"
    sous = ("triées par section · un clic retient la disposition"
            + (f" · A<sub>s,req</sub> = {_req(r['As_req'])} {u}" if r["As_req"] > 0 else ""))
    return C.selecteur(role, titre, sous, "", rows, debut, u, etoile="la plus économique qui suffit")

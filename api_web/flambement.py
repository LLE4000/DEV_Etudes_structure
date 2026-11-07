# -*- coding: utf-8 -*-
"""Le module « Flambement » du site : une note de barres comprimées (une ou plusieurs), chacune vérifiée au flambement
par flexion autour de ses deux axes (EN 1993-1-1 §6.3.1) et en résistance de section (§6.2.4) — les champs de saisie,
le résultat, la grande fenêtre de calcul (chaque formule substituée), le sélecteur de profilés trié par masse et les
dessins (la barre et sa déformée par axe, la section, les courbes χ(λ̄)), rendus en fragments HTML.

NOUVEAU CALCUL : l'écran Streamlit (modules/flambement.py) affiche « Page en construction » ; il n'y a donc aucune
parité à tenir, et l'avertissement « nouveau calcul, à valider par le bureau » est posé en tête du résultat. Le calcul
est celui du moteur ``acier/flambement`` ; ce fichier MET EN FORME, il ne calcule rien.
"""
import copy
import math
import re
from typing import Dict, List, Optional, Sequence

from acier.flambement import entree as EN
from acier.flambement import moteur as MO
from acier.flambement import normes as NO
from acier.flambement import section as SE
from acier.profiles import familles as FA

from . import commun as C
from . import dessin_flambement as DF

MODULE = "flambement"
PAGE = "flambement"
PARCOURS = [
    # une barre élancée en S355 (classe 4 : l'aire efficace), une cornière, un tube formé à froid, un U, des maintiens,
    # un coefficient et une longueur saisis, et les refus
    {"action": "modifier", "chemin": ["profil"], "valeur": "IPE 400"},
    {"action": "modifier", "chemin": ["famille"], "valeur": "L"},
    {"action": "modifier", "chemin": ["famille"], "valeur": "CHS-F"},
    {"action": "modifier", "chemin": ["profil"], "valeur": "UPN 200"},
    {"action": "modifier", "chemin": ["profil"], "valeur": "HD 400x1086.0"},
    {"action": "modifier", "chemin": ["axes", "z", "maintiens"], "valeur": "2"},
    {"action": "modifier", "chemin": ["N"], "valeur": "-5"},
    {"action": "modifier", "chemin": ["L"], "valeur": "0"},
    {"action": "modifier", "chemin": ["gamma_M1"], "valeur": "0,9"},
    {"action": "selecteur", "role": "profil", "familles": ["L", "CHS"], "froid": False},
    {"action": "selecteur", "role": "profil", "familles": ["AUTRES"], "froid": True},
]

# les puces du sélecteur : (clé, libellé, familles du catalogue)
PUCES = [("IPE", "IPE", ("IPE",)), ("HEA", "HE A", ("HEA",)), ("HEB", "HE B", ("HEB",)), ("HEM", "HE M", ("HEM",)), ("HD", "HD", ("HD",)),
         ("UPE", "UPE", ("UPE",)), ("UPN", "UPN", ("UPN",)), ("L", "Cornières", ("L", "LI")), ("CHS", "CHS", ("CHS",)),
         ("SHS", "SHS", ("SHS",)), ("RHS", "RHS", ("RHS",)), ("AUTRES", "Autres séries", ("IPE A", "IPE O", "IPN", "HEAA", "HL", "HP"))]
FAMILLES_PUCE = {k: f for k, _, f in PUCES}
PUCE_DE = {f: k for k, _, fs in PUCES for f in fs}
PUCE_DE.update({"CHS-F": "CHS", "SHS-F": "SHS", "RHS-F": "RHS"})
FROIDS = {"CHS": "CHS-F", "SHS": "SHS-F", "RHS": "RHS-F"}
PUCES_DEFAUT = {"lamine_I": ("IPE", "HEA", "HEB"), "U": ("UPE", "UPN"), "L": ("L",), "creux_chaud": ("CHS", "SHS", "RHS"),
                "creux_froid": ("CHS", "SHS", "RHS")}
NUANCES = list(NO.NUANCES)
PLANS = {"y": "la barre fléchit dans le plan de l'âme (axe fort)", "z": "la barre fléchit hors du plan de l'âme (axe faible)"}
PLANS_L = {"y": "axe principal fort de la cornière", "z": "axe principal faible de la cornière (il gouverne d'ordinaire)"}
PLANS_CREUX = {"y": "axe parallèle au côté b", "z": "axe parallèle au côté h"}


# ------------------------------------------------------------------ l'état de la note : des barres
def nouvel_etat() -> Dict:
    return {"version": 1, "barres": [EN.nouvelle_barre("P1")], "actif": 0}


def normaliser(etat: Dict) -> Dict:
    """L'état reçu de l'écran relu par le moteur ; une barre illisible (fichier abîmé) lève une erreur claire."""
    barres = etat.get("barres") if isinstance(etat, dict) else None
    if not isinstance(barres, list) or not barres:
        return nouvel_etat()
    out = {"version": 1, "barres": [EN.normaliser(b) for b in barres]}
    try:
        actif = int(etat.get("actif", 0) or 0)
    except (TypeError, ValueError):
        actif = 0
    out["actif"] = min(max(0, actif), len(out["barres"]) - 1)
    return out


def _barre(etat: Dict) -> Dict:
    return etat["barres"][etat["actif"]]


def _nom_suivant(etat: Dict) -> str:
    pris = {b.get("nom") for b in etat["barres"]}
    i = len(etat["barres"]) + 1
    while f"P{i}" in pris:
        i += 1
    return f"P{i}"


def gerer_elements(etat: Dict, action: str, index: Optional[int] = None) -> None:
    """Les barres de la note : en ouvrir une, en ajouter, copier l'ouverte, en supprimer une (jamais la dernière)."""
    bs = etat["barres"]
    if action == "activer":
        etat["actif"] = min(max(0, int(index or 0)), len(bs) - 1)
    elif action == "ajouter":
        bs.append(EN.nouvelle_barre(_nom_suivant(etat)))
        etat["actif"] = len(bs) - 1
    elif action == "dupliquer":
        b = copy.deepcopy(bs[etat["actif"]])
        b["nom"] = f"{b.get('nom') or 'Barre'} (copie)"
        bs.append(b)
        etat["actif"] = len(bs) - 1
    elif action == "supprimer" and len(bs) > 1:
        i = min(max(0, int(index or 0)), len(bs) - 1)
        del bs[i]
        if etat["actif"] > i or etat["actif"] >= len(bs):
            etat["actif"] = max(0, etat["actif"] - 1)


# ------------------------------------------------------------------ la saisie
NOMBRES = {("N",), ("L",), ("gamma_M0",), ("gamma_M1",)}


def _choisir_dans_famille(b: Dict, famille: str) -> str:
    """Un changement de famille retient, dans la nouvelle famille, le plus léger qui tient ; à défaut le moins chargé."""
    cands = [c for c in MO.candidats(b, [famille]) if c["taux"] is not None]
    if not cands:
        raise ValueError(f"aucun profilé {famille} calculable")
    best = MO.plus_leger(cands)
    return (best or min(cands, key=lambda c: c["taux"]))["nom"]


def modifier(etat: Dict, chemin: Sequence, valeur) -> Optional[str]:
    """Écrit une saisie dans la barre ouverte. Une valeur hors domaine (traction, longueur nulle, γ < 1…) est REFUSÉE
    avec un message, jamais ramenée en silence à une autre ; la barre reste alors intacte."""
    b = _barre(etat)
    avant = copy.deepcopy(b)
    chemin = [str(c) for c in chemin]
    try:
        if chemin == ["nom"]:
            b["nom"] = str(valeur if valeur is not None else "")
        elif chemin == ["famille"]:
            f = str(valeur)
            if f not in SE.FABRICATION:
                return f"Famille inconnue : « {valeur} »"
            if SE.profil(b["profil"])["famille"] != f:
                b["profil"] = _choisir_dans_famille(b, f)
        elif chemin == ["profil"]:
            SE.profil(str(valeur))
            b["profil"] = str(valeur)
        elif chemin == ["nuance"]:
            if valeur not in NO.NUANCES:
                return f"Nuance inconnue : « {valeur} »"
            b["nuance"] = valeur
        elif tuple(chemin) in NOMBRES or (len(chemin) == 3 and chemin[0] == "axes" and chemin[1] in ("y", "z")
                                          and chemin[2] in ("k", "Lcr", "maintiens")):
            x = C.lire_nombre(valeur)
            if x is None:
                return f"Nombre illisible : « {valeur} »"
            cle = chemin[-1]
            if cle == "maintiens":
                if x != int(x):
                    return "Nombre entier de maintiens attendu"
                x = int(x)
            msg = EN.verifier_valeur(cle, x)
            if msg:
                return f"Saisie refusée : {msg}"
            if chemin[0] == "axes":
                b["axes"][chemin[1]][cle] = x
            else:
                b[cle] = x
        elif len(chemin) == 3 and chemin[0] == "axes" and chemin[1] in ("y", "z") and chemin[2] == "appuis":
            if valeur not in EN.K_APPUIS:
                return f"Appuis inconnus : « {valeur} »"
            ax = b["axes"][chemin[1]]
            if valeur != ax["appuis"]:
                # passer à un coefficient ou une longueur saisis part de la longueur de flambement en cours
                if valeur == "k":
                    ax["k"] = round(EN.coefficient(ax, b["L"]), 4)
                elif valeur == "L":
                    ax["Lcr"] = round(EN.longueur_flambement(ax, b["L"]), 4)
                ax["appuis"] = valeur
        else:
            return f"Champ inconnu : {'/'.join(chemin)}"
        etat["barres"][etat["actif"]] = EN.normaliser(b)
    except (ValueError, KeyError, TypeError) as e:
        etat["barres"][etat["actif"]] = avant
        return f"Saisie refusée : {e}"
    return None


def choisir_section(etat: Dict, role: str, nom: str) -> None:
    """La ligne choisie dans le sélecteur de profilés."""
    b = _barre(etat)
    SE.profil(nom)
    b["profil"] = nom
    etat["barres"][etat["actif"]] = EN.normaliser(b)


# ------------------------------------------------------------------ les nombres
def _kN(x: float) -> str:
    return C.fr(x, 1)


def _m(x: float) -> str:
    return C.fr(x, 2)


def _k(k: float) -> str:
    s = f"{k:.3f}".rstrip("0")
    return (s + "0" if s.endswith(".") else s).replace(".", ",")


def _g(x: float) -> str:
    """Un coefficient γ : 1,0 · 1,05 · 1,1."""
    return _k(x)


def _n3(x: float) -> str:
    return C.fr(x, 3)


def _nom(nom: str) -> str:
    """Un nom de profilé À L'ÉCRAN : « L 180x180x13 » → « L 180×180×13 » (les capitales des en-têtes n'en font plus
    « 180X180 ») ; le nom du catalogue reste la clé des requêtes."""
    return re.sub(r"(?<=\d)x(?=\d)", "×", str(nom))


BARRE = '<i class="fl-barre">λ</i>'
GRECS = re.compile(r"([λχφερψα])(̄?)_([A-Za-z0-9éèàùâ]+(?:,[A-Za-z0-9éèàùâ]+)*)")


def _grecs(h: str) -> str:
    """Les indices des lettres grecques que la brique commune (C.symb_math) laisse en ligne (λ_y, χ_z, φ, ρ, ε),
    et le λ̄ surligné (le macron combinant se décale selon la police)."""
    return GRECS.sub(lambda m: f"{m.group(1)}{m.group(2)}<sub>{m.group(3)}</sub>", h).replace("λ̄", BARRE)


def _fx(lignes: List[Dict]) -> str:
    return _grecs(C.lignes_fx(lignes))


def _li(parts: List[str], res: str, niveau: str = "") -> Dict:
    return {"parts": parts, "res": res, "niveau": niveau}


# ------------------------------------------------------------------ la saisie : le cadre Données
def _plans(fab: str) -> Dict[str, str]:
    return PLANS_L if fab == "L" else (PLANS_CREUX if fab.startswith("creux") else PLANS)


def _libelle_famille(f: str) -> str:
    return FA.FAMILLES[f]["libelle"]


def _contour(p: Dict):
    """Le contour exact du profilé (acier/profiles, source unique du catalogue), en mm."""
    return FA.contours(p["famille"], p["dims"])


def _axe_form(a: str, ax: Dict, r: Optional[Dict], nom_axe: str, fab: str, L: float) -> str:
    L_ = C.ligne
    s = (L_("Appuis", C.choix(["axes", a, "appuis"], ax["appuis"], [(k, t) for k, t, _ in EN.APPUIS]), full=True)
         + C.aide(f"Autour de {nom_axe} : {_plans(fab)[a]}. Le premier appui nommé est le pied de la barre."))
    if ax["appuis"] == "AA":
        s += L_("maintiens", C.num(["axes", a, "maintiens"], ax["maintiens"], "", entier=True, mini=0, maxi=EN.MAINTIENS_MAX), "n",
                "Maintiens latéraux intermédiaires, équidistants, dans ce plan (lisses, butons) : L_cr = L / (n + 1).")
    elif ax["appuis"] == "k":
        s += L_("coefficient", C.num(["axes", a, "k"], ax["k"], "", dec=4), "k", "L_cr = k·L (0,85 de pratique pour encastrée–articulée…).")
    elif ax["appuis"] == "L":
        s += L_("longueur", C.num(["axes", a, "Lcr"], ax["Lcr"], "m", dec=4), "L<sub>cr</sub>", "Saisie directement, en mètres.")
    else:
        s += C.aide(f"Coefficient théorique d'Euler k = {_k(EN.K_APPUIS[ax['appuis']])} ; un coefficient de pratique se saisit "
                    "avec « Coefficient k saisi ».")
    lcr = EN.longueur_flambement(ax, L)
    # k et L_cr rappelés sous le menu : sur une tablette en portrait, le libellé du menu est coupé avant « k = … »
    s += C.aide(f"k = {_k(EN.coefficient(ax, L))} · L<sub>cr</sub> = {_m(lcr)} m"
                + (f" · courbe <b>{r['axes'][a]['courbe']}</b>" if r and r.get("axes") else ""))
    return s


def formulaire(etat: Dict, r: Optional[Dict] = None) -> str:
    b = _barre(etat)
    p = SE.profil(b["profil"])
    fab = SE.FABRICATION[p["famille"]]
    L_ = C.ligne
    fams = [(f, _libelle_famille(f)) for f in SE.FAMILLES]
    profs = [(n, f"{n} · {C.fr(SE.profil(n)['masse'], 1)} kg/m") for n in SE.noms([p["famille"]])]
    sec = (r or {}).get("section")
    if sec:
        t = sec["t"]
        fy_txt = (f"f<sub>y</sub> = {C.fr(sec['fy'], 0)} N/mm² (Tableau 3.1, t = {C.nb(t, 1)} mm "
                  f"{'≤ 40' if t <= 40 else '≤ 80'} mm) · classe {sec['classe'] if fab != 'L' or sec['classe'] == 4 else '1 à 3'} en compression")
    else:
        fy_txt = f"Calcul impossible : {C.esc((r or {}).get('motif', ''))}"
    s1 = (L_("Nom", C.texte(["nom"], b.get("nom"), "ex. P1"), full=True)
          + L_("Famille", C.choix(["famille"], p["famille"], fams), full=True)
          + L_("Profilé", C.choix(["profil"], b["profil"], profs), full=True)
          + C.aide("Le sélecteur du résultat (la ligne « Section ») trie les profilés par masse et marque le plus léger qui tient (★).")
          + L_("Nuance", C.segments(["nuance"], b["nuance"], [(n, n) for n in NUANCES]), full=True)
          + C.aide(fy_txt))
    s2 = (L_("compression", C.num(["N"], b["N"], "kN"), "N<sub>Ed</sub>", "Valeur de calcul à l'ELU ; compression positive.")
          + L_("longueur", C.num(["L"], b["L"], "m", dec=4), "L", "Longueur d'épure entre les appuis d'extrémité."))
    noms_axes = SE.NOMS_AXES[fab]
    secs_axes = ""
    for i, a in enumerate(("y", "z")):
        ax = b["axes"][a]
        lcr = EN.longueur_flambement(ax, b["L"])
        resume = f"{C.esc(EN.LIB_APPUIS[ax['appuis']].split(' · ')[0])} · L<sub>cr</sub> {_m(lcr)} m"
        secs_axes += C.section(3 + i, f"Flambement autour de {noms_axes[i]}", resume, _axe_form(a, ax, r, noms_axes[i], fab, b["L"]), True, f"ax{a}")
    s5 = (L_("sections", C.num(["gamma_M0"], b["gamma_M0"], ""), "γ<sub>M0</sub>", "Résistance de la section, §6.2.4 ; ANB belge : 1,0.")
          + L_("instabilités", C.num(["gamma_M1"], b["gamma_M1"], ""), "γ<sub>M1</sub>", "Flambement, §6.3.1 ; ANB belge : 1,0.")
          + C.aide("E = 210 000 N/mm² · f<sub>y</sub> du Tableau 3.1 selon l'épaisseur · courbes du Tableau 6.2 selon le profilé, l'axe et "
                   "l'épaisseur · classe 4 : aire efficace de l'EN 1993-1-5 §4.4."))
    return (C.section(1, f"Barre {C.esc(b.get('nom') or '')}", f"{C.esc(b['profil'])} · {b['nuance']}", s1, True, "br")
            + C.section(2, "Effort et longueur", f"N<sub>Ed</sub> {C.nb(b['N'])} kN · L {_m(b['L'])} m", s2, True, "ef")
            + secs_axes
            + C.section(5, "Méthode", f"γ<sub>M0</sub> {_g(b['gamma_M0'])} · γ<sub>M1</sub> {_g(b['gamma_M1'])} · ANB", s5, False, "me"))


# ------------------------------------------------------------------ le résultat
def _calculer(b: Dict) -> Dict:
    return MO.verifier(b)


def _taux(r: Dict) -> Optional[float]:
    return r.get("taux_max")


def _niv(t: Optional[float]) -> str:
    return C.niveau(t) if t is not None else "x"


def _gov_txt(r: Dict) -> str:
    g = r.get("gouvernant")
    if g in ("y", "z"):
        return f"flambement {r['axes'][g]['nom']}"
    return "résistance de la section" if g == "section" else ""


def resume_note(etat: Dict, resultats: List[Dict]) -> Dict:
    """Le résumé de la NOTE : sa barre la plus chargée."""
    vals = [(t, i) for i, r in enumerate(resultats) if (t := _taux(r)) is not None]
    if not vals:
        return {"taux": None, "niveau": "x", "statut": "Calcul impossible", "gouvernant": ""}
    t, i = max(vals)
    niv = C.niveau(t)
    return {"taux": C.pct(t), "niveau": niv, "statut": C.STATUTS[niv][1],
            "gouvernant": f"{etat['barres'][i].get('nom') or 'Barre'} · {_nom(resultats[i]['profil'])} · {_gov_txt(resultats[i])}"}


def elements(etat: Dict, resultats: List[Dict]) -> str:
    items = ""
    for i, (b, r) in enumerate(zip(etat["barres"], resultats)):
        t = _taux(r)
        txt = f"{C.pc(C.pct(t))}{C.NBSP}%" if t is not None else "—"
        sup = f'<button class="x" data-supprimer="{i}" title="Supprimer">{C.icon("x")}</button>' if len(etat["barres"]) > 1 else ""
        items += (f'<div class="ni{" on" if i == etat["actif"] else ""}" data-activer="{i}"><span class="t">{C.esc(b.get("nom") or f"Barre {i + 1}")} '
                  f'<span>· {C.esc(_nom(b["profil"]))} · {_m(b["L"])} m</span></span><span class="pct {_niv(t)}">{txt}</span>{sup}</div>')
    return (f'<div class="elements"><div class="nl-h"><span class="kicker">Dans cette note · {len(etat["barres"])}</span>'
            f'<button class="nl-add" data-dupliquer="1" title="Copier la barre ouverte">{C.icon("copy")}Copier</button></div>{items}'
            f'<div class="nl-add" data-ajouter="1">{C.icon("plus")}Ajouter une barre</div></div>')


def _nouveau() -> str:
    return (f'<div class="fl-nouveau">{C.message(C.icon("info") + "<span><b>Nouveau calcul</b> EN 1993-1-1 §6.3.1, à valider par le bureau</span>")}'
            '</div>')


def _symb_A(sec: Dict) -> str:
    return "A_eff" if sec["classe"] == 4 else "A"


def _classe_lignes(r: Dict) -> List[Dict]:
    sec = r["section"]
    eps = sec["epsilon"]
    out = []
    t = sec["t"]
    borne = "≤ 40 mm" if t <= 40 else "≤ 80 mm (40 < t)"
    out.append(_li(["f_y", f"Tableau 3.1 · {r['nuance']} · t = {C.nb(t, 1)} mm {borne}"], f"{C.fr(sec['fy'], 0)} N/mm²"))
    out.append(_li(["ε", "√(235 / f_y)", f"√(235 / {C.fr(sec['fy'], 0)})"], _n3(eps)))
    if sec["fabrication"] == "L":
        p = SE.profil(r["profil"])
        cl = sec["classe"]
        out.append(_li(["h / t", f"{C.nb(p['h'], 1)} / {C.nb(p['t'], 1)}"], f"{C.fr(sec['h_t'], 2)} {'≤' if sec['h_t'] <= 15 * eps else '>'} 15·ε = {C.fr(15 * eps, 2)}"))
        out.append(_li(["(b + h) / (2·t)", f"({C.nb(p['b'], 1)} + {C.nb(p['h'], 1)}) / (2 × {C.nb(p['t'], 1)})"],
                       f"{C.fr(sec['bh_2t'], 2)} {'≤' if sec['bh_2t'] <= 11.5 * eps else '>'} 11,5·ε = {C.fr(11.5 * eps, 2)} · "
                       + ("classe 1 à 3" if cl < 4 else "classe 4")))
    else:
        for pa in sec["parois"]:
            typ = pa["type"]
            lims = NO.LIMITES_CLASSE[typ]
            k = eps ** 2 if typ == "tube" else eps
            cl = pa["classe"]
            lim = lims[min(cl, 3) - 1]
            sym = "d / t" if typ == "tube" else f"c / t ({pa['nom']})"
            e_txt = "ε²" if typ == "tube" else "ε"
            sgn = ">" if cl == 4 else "≤"
            if pa.get("cote"):                     # tube rectangulaire : c = b − 3t (EN 1993-1-5 §4.4(2))
                d, v = pa["cote"]
                expr = [f"({d} − 3·t) / t", f"({C.nb(v, 1)} − 3 × {C.nb(pa['t'], 1)}) / {C.nb(pa['t'], 1)}"]
            else:
                expr = [f"{C.nb(pa['c'], 1)} / {C.nb(pa['t'], 1)}"]
            out.append(_li([sym] + expr, f"{C.fr(pa['c_t'], 2)} {sgn} {C.nb(lim, 1)}·{e_txt} = {C.fr(lim * k, 2)} · classe {cl}"))
    if sec["classe"] == 4:
        # cornière : la largeur de référence est h pour les deux ailes (EN 1993-1-5 §4.4(2)), une seule ligne
        corniere = sec["fabrication"] == "L"
        for pa in (sec["parois"][:1] if corniere else sec["parois"]):
            ind = "ailes" if corniere else (pa["cote"][0] if pa.get("cote") else pa["nom"])
            ks = NO.K_SIGMA[pa["type"]]
            out.append(_li([f"λ̄_p,{ind}", "(h/t) / (28,4·ε·√k_σ)" if corniere else "(c/t) / (28,4·ε·√k_σ)",
                            f"{C.fr(pa['c_t'], 2)} / (28,4 × {_n3(eps)} × √{C.nb(ks, 2)})"], _n3(pa["lambda_p"])))
            if pa["lambda_p"] <= NO.LAMBDA_P_LIM[pa["type"]]:
                out.append(_li([f"ρ_{ind}", f"λ̄_p ≤ {C.nb(NO.LAMBDA_P_LIM[pa['type']], 3)}"], "1,000"))
            else:
                f_txt = "(λ̄_p − 0,22) / λ̄_p²" if pa["type"] == "interne" else "(λ̄_p − 0,188) / λ̄_p²"
                v = _n3(pa["lambda_p"])
                out.append(_li([f"ρ_{ind}", f_txt, f"({v} − {'0,22' if pa['type'] == 'interne' else '0,188'}) / {v}²"], _n3(pa["rho"])))
        reduites = [pa for pa in sec["parois"] if pa["rho"] < 1.0]
        if reduites:
            termes = " − ".join(f"{pa['n']} × (1 − {_n3(pa['rho'])}) × {C.nb(pa['c'], 1)} × {C.nb(pa['t'], 1)}" for pa in reduites)
            out.append(_li(["A_eff", "A − Σ n·(1 − ρ)·c·t", f"{C.fr(sec['A'], 0)} − {termes}"], f"{C.fr(sec['A_eff'], 0)} mm²"))
        else:
            out.append(_li(["A_eff", "A, aucune paroi réduite (ρ = 1)"], f"{C.fr(sec['A_eff'], 0)} mm²"))
    else:
        out.append(_li(["A", "catalogue", f"{C.fr(sec['A'] / 100, 2)} cm²"], f"{C.fr(sec['A'], 0)} mm²"))
    sA = _symb_A(sec)
    out.append(_li(["N_c,Rd", f"{sA}·f_y / γ_M0", f"{C.fr(sec['A_eff'], 0)} × {C.fr(sec['fy'], 0)} / {_g(r['gamma_M0'])}"], f"{_kN(sec['NcRd'])} kN"))
    t_s = next(v for v in r["verifications"] if v["cle"] == "section")["taux"]
    out.append(_li(["η_N", "N_Ed / N_c,Rd", f"{C.nb(r['N'])} / {_kN(sec['NcRd'])}"], f"{C.pc(C.pct(t_s, 1))} %", C.niveau(t_s)))
    return out


def _axe_lignes(r: Dict, a: str) -> List[Dict]:
    ax = r["axes"][a]
    sec = r["section"]
    i = "y" if a == "y" else "z"
    ia = ax["nom"][0]                              # y, z, u, v : l'indice écrit
    out = []
    if ax["appuis"] == "AA" and ax["maintiens"]:
        out.append(_li([f"L_cr,{ia}", "L / (n + 1)", f"{_m(r['L'])} / ({ax['maintiens']} + 1)"], f"{_m(ax['Lcr'])} m"))
    elif ax["appuis"] == "L":
        out.append(_li([f"L_cr,{ia}", "saisie"], f"{_m(ax['Lcr'])} m"))
    else:
        out.append(_li([f"L_cr,{ia}", "k·L", f"{_k(ax['k'])} × {_m(r['L'])}"], f"{_m(ax['Lcr'])} m"))
    sA = _symb_A(sec)
    out.append(_li([f"N_cr,{ia}", f"π²·E·I_{ia} / L_cr,{ia}²", f"π² × 210 000 × {C.fr(ax['I'], 1)}·10⁴ / {C.fr(ax['Lcr'] * 1000, 0)}²"],
                   f"{_kN(ax['Ncr'])} kN"))
    out.append(_li([f"λ̄_{ia}", f"√({sA}·f_y / N_cr,{ia})", f"√({C.fr(sec['A_eff'], 0)} × {C.fr(sec['fy'], 0)} / ({_kN(ax['Ncr'])}·10³))"],
                   _n3(ax["lambdabar"])))
    if sec["classe"] < 4:
        out.append(_li([f"λ̄_{ia}", f"L_cr,{ia} / (i_{ia}·λ_1)", f"{C.fr(ax['Lcr'] * 1000, 0)} / ({C.fr(ax['i'] * 10, 1)} × {C.fr(ax['lambda1'], 1)})"],
                       f"{_n3(ax['lambdabar'])} (contrôle)"))
    out.append(_li([f"α_{ia}", f"courbe {ax['courbe']} · {ax['regle']}"], C.nb(ax["alpha"], 2)))
    lb = _n3(ax["lambdabar"])
    out.append(_li([f"φ_{ia}", "½·[1 + α(λ̄ − 0,2) + λ̄²]", f"½·[1 + {C.nb(ax['alpha'], 2)}·({lb} − 0,2) + {lb}²]"], _n3(ax["phi"])))
    ph = _n3(ax["phi"])
    out.append(_li([f"χ_{ia}", "1 / (φ + √(φ² − λ̄²)) ≤ 1", f"1 / ({ph} + √({ph}² − {lb}²))"], _n3(ax["chi"])))
    out.append(_li([f"N_b,{ia},Rd", f"χ_{ia}·{sA}·f_y / γ_M1", f"{_n3(ax['chi'])} × {C.fr(sec['A_eff'], 0)} × {C.fr(sec['fy'], 0)} / {_g(r['gamma_M1'])}"],
                   f"{_kN(ax['NbRd'])} kN"))
    out.append(_li([f"η_{ia}", f"N_Ed / N_b,{ia},Rd", f"{C.nb(r['N'])} / {_kN(ax['NbRd'])}"], f"{C.pc(C.pct(ax['taux'], 1))} %", C.niveau(ax["taux"])))
    del i
    return out


def _sub(s: str) -> str:
    """« N_b,y,Rd » → N<sub>b,y,Rd</sub> (HTML)."""
    return _grecs(C.symb_math(s))


def _tableau(r: Dict) -> List[Dict]:
    sec = r["section"]
    sA = "A<sub>eff</sub>" if sec["classe"] == 4 else "A"
    g = r["gouvernant"]
    out = []
    for a in ("y", "z"):
        ax = r["axes"][a]
        ia = ax["nom"][0]
        out.append({"nom": f"Flambement {ax['nom']}", "sous": f"courbe {ax['courbe']} · {BARRE} = {_n3(ax['lambdabar'])} · χ = {_n3(ax['chi'])}",
                    "formule": f"N<sub>Ed</sub> ≤ χ<sub>{ia}</sub>·{sA}·f<sub>y</sub> / γ<sub>M1</sub>", "valeur": f"N<sub>Ed</sub> = {C.nb(r['N'])} kN",
                    "limite": f"N<sub>b,{ia},Rd</sub> = {_kN(ax['NbRd'])} kN", "taux": C.pct(ax["taux"], 1), "niveau": C.niveau(ax["taux"]), "gov": g == a})
    t_s = next(v for v in r["verifications"] if v["cle"] == "section")["taux"]
    cl = "classe 1 à 3" if sec["fabrication"] == "L" and sec["classe"] < 4 else f"classe {sec['classe']}"
    out.append({"nom": "Section", "sous": f"{C.esc(_nom(r['profil']))} · {cl}", "formule": f"N<sub>Ed</sub> ≤ {sA}·f<sub>y</sub> / γ<sub>M0</sub>",
                "valeur": f"N<sub>Ed</sub> = {C.nb(r['N'])} kN", "limite": f"N<sub>c,Rd</sub> = {_kN(sec['NcRd'])} kN", "taux": C.pct(t_s, 1),
                "niveau": C.niveau(t_s), "gov": g == "section"})
    return out


def _hypotheses(r: Dict) -> List[str]:
    sec = r["section"]
    cl = "classe 1 à 3" if sec["fabrication"] == "L" and sec["classe"] < 4 else f"classe {sec['classe']}"
    aire = (f"A<sub>eff</sub> = {C.fr(sec['A_eff'] / 100, 2)} cm²" if sec["classe"] == 4 else f"A = {C.fr(sec['A'] / 100, 2)} cm²")
    return [f"Acier <b>{r['nuance']}</b> · f<sub>y</sub> = {C.fr(sec['fy'], 0)} N/mm² · E = 210 000 N/mm²",
            f"<b>{C.esc(_nom(r['profil']))}</b> · {aire} · {cl}",
            f"N<sub>Ed</sub> = <b>{C.nb(r['N'])} kN</b> · L = {_m(r['L'])} m",
            f"γ<sub>M0</sub> = {_g(r['gamma_M0'])} · γ<sub>M1</sub> = {_g(r['gamma_M1'])} (ANB)"]


def _fiche(p: Dict, r: Dict) -> List[tuple]:
    sec = r.get("section")
    out = [("A", C.fr(p["A"], 2), "cm²")]
    if sec and sec["classe"] == 4:
        out.append(("A_eff", C.fr(sec["A_eff"] / 100, 2), "cm²"))
    for cle, ax in SE.axes(p).items():
        ia = ax["nom"][0]
        out.append((f"I_{ia}", C.fr(ax["I"], 1), "cm⁴"))
    for cle, ax in SE.axes(p).items():
        ia = ax["nom"][0]
        out.append((f"i_{ia}", C.fr(ax["i"], 2), "cm"))
    out.append(("masse", C.fr(p["masse"], 1), "kg/m"))
    if sec:
        out.append(("f_y", C.fr(sec["fy"], 0), "N/mm²"))
        out.append(("classe", ("1 à 3" if sec["fabrication"] == "L" and sec["classe"] < 4 else str(sec["classe"])), ""))
    return out


def _vue_note(etat: Dict, resultats: List[Dict]) -> str:
    """La vue « Toutes les barres » : une ligne par barre (un tableau, posé dans la feuille)."""
    lignes = ""
    for i, (b, r) in enumerate(zip(etat["barres"], resultats)):
        t = _taux(r)
        if t is None:
            chip, nb, gv = '<span class="chip c">—</span>', "—", C.esc(r.get("motif", ""))
            lcr = "—"
        else:
            chip = C.chip(C.pct(t), C.niveau(t))
            nb = _kN(min(v["limite"] for v in r["verifications"]))
            gv = r["axes"][r["gouvernant"]]["nom"] if r["gouvernant"] in ("y", "z") else "section"
            lcr = " / ".join(_m(r["axes"][a]["Lcr"]) for a in ("y", "z"))
        lignes += (f'<tr class="{"on" if i == etat["actif"] else ""}"><td class="nm">{C.esc(b.get("nom") or f"Barre {i + 1}")}</td>'
                   f'<td>{C.esc(_nom(b["profil"]))} · {b["nuance"]}</td><td class="n">{C.nb(b["N"])}</td><td class="n">{_m(b["L"])}</td>'
                   f'<td class="n">{lcr}</td><td class="n">{nb}</td><td>{gv}</td><td class="n">{chip}</td></tr>')
    return (f'<div class="fl-vue"><table class="vtab"><thead><tr><th>Barre</th><th>Profilé · nuance</th><th class="n">N<sub>Ed</sub> kN</th>'
            f'<th class="n">L m</th><th class="n">L<sub>cr</sub> fort / faible m</th><th class="n">N<sub>Rd</sub> kN</th>'
            f'<th>Gouverne</th><th class="n">Taux</th></tr></thead><tbody>{lignes}</tbody></table></div>')


def _optimal(b: Dict) -> Optional[Dict]:
    """Le plus léger qui tient parmi les familles par défaut du sélecteur (l'étoile de la ligne « Section »)."""
    fams = _familles_de(_puces_defaut(b))
    return MO.plus_leger(MO.candidats(b, fams))


def _extents(p: Dict) -> Dict[str, float]:
    """La largeur de la section dans chaque plan de flambement (mm) : autour de y, la barre fléchit selon z. Pour une
    cornière, les plans sont ceux des axes PRINCIPAUX : autour de u-u elle fléchit selon v, autour de v-v selon u
    (u fait l'angle α du catalogue avec y-y) ; la largeur est l'étendue du contour exact projeté sur cette direction."""
    if p["famille"] in ("L", "LI"):
        a = math.radians(float(p.get("alpha_deg") or 0.0))
        pts = [(float(q[0]), float(q[1])) for ring in _contour(p) for q in ring]

        def etendue(dy: float, dz: float) -> float:
            proj = [y * dy + z * dz for y, z in pts]
            return max(proj) - min(proj)
        return {"y": etendue(-math.sin(a), math.cos(a)), "z": etendue(math.cos(a), math.sin(a))}
    ez = float(p.get("z_sup") or 0) + float(p.get("z_inf") or 0)
    ey = float(p.get("y_droite") or 0) + float(p.get("y_gauche") or 0)
    return {"y": ez or float(p.get("h") or p.get("d") or 100), "z": ey or float(p.get("b") or p.get("d") or 100)}


def resultat(etat: Dict) -> Dict:
    b = _barre(etat)
    r = _calculer(b)
    resultats = [r if i == etat["actif"] else _calculer(q) for i, q in enumerate(etat["barres"])]
    p = SE.profil(b["profil"])
    nom = b.get("nom") or "Barre"
    vues = []
    if r["statut"] == MO.IMPOSSIBLE:
        niv = "x"
        corps = (f'{_nouveau()}<div class="verdict"><div class="gov"><b>Calcul impossible</b>{C.esc(r["motif"])}</div></div>'
                 f'<div class="els"><div class="elgrp">{C.esc(nom)} · {C.esc(_nom(r["profil"]))} · {r["nuance"]}</div>'
                 + C.ligne_res("Section", C.esc(_nom(r["profil"])), None, "c", pk="profil") + '</div>'
                 + C.message("Choisissez un autre profilé dans le sélecteur, ou une autre nuance.", True))
        volets = ""
        vues.append({"cle": "section", "titre": f"Section {_nom(p['nom'])}", "court": "Section", "svg": DF.section("fl_section", p, r, _fiche(p, r)),
                     "legende": r["motif"]})
    else:
        niv = C.niveau(r["taux_max"])
        g = r["gouvernant"]
        v_g = next(v for v in r["verifications"] if v["cle"] == g)
        li_g = next(li for li in _tableau(r) if li["gov"])
        verdict = (f'<div class="verdict">{C.chip(C.pct(r["taux_max"]), niv, xl=True)}<div class="gov"><b>{C.esc(v_g["nom"])}</b>'
                   f'{li_g["valeur"]} {"≤" if v_g["taux"] <= 1 else "&gt;"} {li_g["limite"]}</div></div>')
        opt = _optimal(b)
        leger = f"Plus léger qui tient : {_nom(opt['nom'])} · {C.pc(C.pct(opt['taux']))} %" if opt and opt["nom"] != b["profil"] else ""
        sec = r["section"]
        rows = f'<div class="elgrp">{C.esc(nom)} · {C.esc(_nom(r["profil"]))} · {r["nuance"]}<span class="pct {niv}">{C.pc(C.pct(r["taux_max"]))}{C.NBSP}%</span></div>'
        for v in r["verifications"]:
            if v["cle"] in ("y", "z"):
                ax = r["axes"][v["cle"]]
                piece = f"L<sub>cr</sub> = {_m(ax['Lcr'])} m · courbe {ax['courbe']}"
                rows += C.ligne_res(f"Flambement {ax['nom']}", piece, C.pct(v["taux"]), C.niveau(v["taux"]), calcul="fl", gov=g == v["cle"])
            else:
                cl = "classe 1 à 3" if sec["fabrication"] == "L" and sec["classe"] < 4 else f"classe {sec['classe']}"
                rows += C.ligne_res("Section", f"{C.esc(_nom(r['profil']))} · {cl}", C.pct(v["taux"]), C.niveau(v["taux"]), calcul="fl", pk="profil",
                                    gov=g == "section", leger=leger)
        messages = "".join(C.message(_grecs(C.symb_math(a))) for a in r["avertissements"][1:])
        if any(ax["negligeable"] for ax in r["axes"].values()):
            messages += C.message(f"{BARRE} ≤ 0,2 autour d'un axe : χ = 1, le flambement y est sans effet (§6.3.1.2(4)).")
        corps = f'{_nouveau()}{verdict}<div class="els">{rows}</div>{messages}'
        detail = (f'<div class="ck-g"><div class="ck-t">Matériau et section</div>{_fx(_classe_lignes(r))}</div>'
                  + "".join(f'<div class="ck-g"><div class="ck-t">Flambement autour de {r["axes"][a]["nom"]}</div>{_fx(_axe_lignes(r, a))}</div>'
                            for a in ("y", "z")))
        pied_ck = ('<div class="ck-autres">Nouveau calcul EN 1993-1-1 §6.3.1, à valider par le bureau. Non vérifiés ici : flambement par '
                   'torsion et flexion-torsion (§6.3.1.4), déversement, flexion composée (§6.3.3).</div>')
        volets = C.fenetre_calcul("fl", f"{C.esc(nom)} · {C.esc(_nom(r['profil']))} · {r['nuance']}",
                                  f"N<sub>Ed</sub> = {C.nb(r['N'])} kN · L = {_m(r['L'])} m · EN 1993-1-1 §6.3.1",
                                  _hypotheses(r), _tableau(r), detail, pied_ck)
        ay, az = r["axes"]["y"], r["axes"]["z"]
        vues.append({"cle": "barre", "titre": f"Barre {nom}", "court": "Barre",
                     "svg": DF.barre("fl_barre", r, _extents(p), nom),
                     "legende": f"{_nom(r['profil'])} · L = {_m(r['L'])} m"})
        vues.append({"cle": "section", "titre": f"Section {_nom(p['nom'])}", "court": "Section", "svg": DF.section("fl_section", p, r, _fiche(p, r)),
                     "legende": f"{'classe 1 à 3' if sec['fabrication'] == 'L' and sec['classe'] < 4 else 'classe ' + str(sec['classe'])} · "
                                f"courbes {ay['courbe']} ({ay['nom']}) et {az['courbe']} ({az['nom']}) · {r['courbes']['regle']}"})
        vues.append({"cle": "courbes", "titre": "Courbes de flambement", "court": "Courbes χ", "svg": DF.courbes("fl_courbes", r),
                     "legende": "χ selon l'élancement réduit, courbes a0 à d (Tableau 6.1), et le point de chaque axe",
                     "echelle": False})                  # un graphique : pas de « mm · à l'échelle » (site/web/js/etabli.js)
    if len(etat["barres"]) > 1:
        vues.append({"cle": "note", "titre": "Toutes les barres", "court": "Toutes", "svg": _vue_note(etat, resultats),
                     "legende": f"{len(etat['barres'])} barres · la ligne surlignée est ouverte"})
    pied = (f'<div class="pfoot"><div class="mass"><span>Masse de la barre · {C.fr(p["masse"], 1)} kg/m × {_m(b["L"])} m</span>'
            f'<b>{C.fr(p["masse"] * b["L"], 0)}{C.NBSP}kg</b></div><div class="legend">{C.legende()}</div></div>')
    # la phrase sous le titre de la feuille : courte (les L_cr sont sur le dessin), sinon un nom long la coupe
    phrase = f"{C.esc(_nom(r['profil']))} · {r['nuance']} · N<sub>Ed</sub> = {C.nb(b['N'])} kN · L = {_m(b['L'])} m"
    return {"resultat": f'<div class="pbody">{corps}</div>{pied}', "volets": volets, "vues": vues, "phrase": phrase,
            "statut_html": C.statut(niv), "resume": resume_note(etat, resultats), "donnees": formulaire(etat, r),
            "elements": elements(etat, resultats)}


# ------------------------------------------------------------------ le sélecteur de profilés
def _puces_defaut(b: Dict) -> List[str]:
    f = SE.profil(b["profil"])["famille"]
    puces = list(PUCES_DEFAUT[SE.FABRICATION[f]])
    if PUCE_DE[f] not in puces:
        puces.append(PUCE_DE[f])
    return puces


def _familles_de(puces: Sequence[str], froid: bool = True) -> List[str]:
    out = []
    for k in puces:
        out += list(FAMILLES_PUCE.get(k, ()))
        if froid and k in FROIDS:
            out.append(FROIDS[k])
    return out


def selecteur(etat: Dict, role: str, familles: Optional[Sequence[str]] = None, froid: bool = True) -> str:
    """Les profilés des familles cochées, vérifiés avec les données de la barre ouverte, TRIÉS PAR MASSE ; le retenu
    (✓) et le plus léger qui tient (★) marqués ; une fenêtre autour d'eux."""
    b = _barre(etat)
    puces = [k for k in (familles or []) if k in FAMILLES_PUCE] or _puces_defaut(b)
    cands_tous = MO.candidats(b, _familles_de(puces, froid))
    cands = [c for c in cands_tous if c["taux"] is not None]
    ecartes = len(cands_tous) - len(cands)
    best = MO.plus_leger(cands)
    cur = b["profil"]
    noms = [c["nom"] for c in cands]
    i_o = noms.index(best["nom"]) if best else 0
    i_c = noms.index(cur) if cur in noms else i_o

    def prop(c):
        p = SE.profil(c["nom"])
        ax = SE.axes(p)
        ia, iz = ax["y"]["nom"][0], ax["z"]["nom"][0]
        return (f"{_libelle_famille(p['famille'])} · A = {C.fr(p['A'], 2)} cm² · i<sub>{ia}</sub> = {C.fr(ax['y']['i'], 2)} cm · "
                f"i<sub>{iz}</sub> = {C.fr(ax['z']['i'], 2)} cm · courbes {c['courbes']['y']} / {c['courbes']['z']}")
    rows = [{"nom": c["nom"], "col": C.fr(c["masse"], 2), "taux": C.pct(min(c["taux"], 9.99)), "niveau": C.niveau(c["taux"]),
             "cur": c["nom"] == cur, "opt": best is not None and c["nom"] == best["nom"], "prop": prop(c)} for c in cands]
    filtres = "".join(f'<button type="button" class="fchip{" on" if k in puces else ""}" data-famille="{k}">{C.esc(t)}</button>' for k, t, _ in PUCES)
    filtres += f'<button type="button" class="fchip{" on" if froid else ""}" data-froid="1">Creux formés à froid</button>'
    sous = "triés par masse · un clic retient le profilé"
    if ecartes:
        sous += f" · {ecartes} hors du champ écarté{'s' if ecartes > 1 else ''} (t &gt; 80 mm, tube de classe 4)"
    return C.selecteur(role, f"Barre {C.esc(b.get('nom') or '')} · profilé", sous, filtres, rows, max(0, min(i_o, i_c) - 4), "kg/m")

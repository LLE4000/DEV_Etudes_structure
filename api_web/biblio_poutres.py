# -*- coding: utf-8 -*-
"""La rubrique « POUTRES » de la bibliothèque « Autres » : le FORMULAIRE DE RDM (docs/site/BIBLIOTHEQUE_AUTRES.md § 3).
Le moteur est rdm/poutres.py ; ce fichier LIT les paramètres (l'adresse de la page), COMPOSE l'écran — le modèle et la
rigidité dans la barre ; le dessin, les charges, les résultats et la feuille de calcul dans le cadre — et FORMATE.

Les paramètres sont plats (une adresse se partage) : modele, nt, L1…L5, n, puis pour chaque charge i : t{i} (q, tri, P,
C), v{i}, a{i}, b{i}, s{i} ; rig (acier, bois, libre, sans) et ses données ; x (la section lue) ; g (la grandeur de la
bande).

LA FEUILLE DE CALCUL écrit chaque résultat en trois temps — formule générale, valeurs, résultat. Pour un CAS D'ÉCOLE
(une seule charge, cas usuel) c'est la formule du formulaire ; sinon la SUPERPOSITION : travée isostatique charge par
charge, liaisons (rotations nulles, trois moments), réactions, puis V, M et f dans la section lue et aux extremums.
Les nombres viennent tous du moteur ; les formules écrites ici sont celles qu'il évalue (tests/
test_api_web_bibliotheque_autres.py relit chaque « = valeurs » et le compare au résultat affiché).
"""
import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

from acier.profiles import catalogue as KA
from api_web import biblio_autres as A
from api_web import biblio_beton as B
from api_web import biblio_poutres_dessin as D
from api_web import commun as C
from bois.poutre import donnees as BD
from rdm import poutres as P

MODELES = (("appuis", "Deux appuis"), ("console", "Console"), ("enc_app", "Encastrée–appuyée"), ("bi_enc", "Bi-encastrée"), ("continue", "Continue"))
TITRES_MODELE = {"appuis": "Poutre sur deux appuis simples", "console": "Console (encastrée en A, libre en B)",
                 "enc_app": "Poutre encastrée en A, appuyée en B", "bi_enc": "Poutre bi-encastrée", "continue": "Poutre continue sur appuis simples"}
TYPES = (("q", "Répartie"), ("tri", "Triangle"), ("P", "Ponctuelle"), ("C", "Couple"))
NOM_TYPE = dict(TYPES)
UNITE = {"q": "kN/m", "tri": "kN/m", "P": "kN", "C": "kN·m"}
LETTRE = {"q": "q", "tri": "q", "P": "P", "C": "C"}
DEFAUT_V = {"q": 10.0, "tri": 10.0, "P": 20.0, "C": 10.0}
SENS = (("croissante", "0 → q"), ("decroissante", "q → 0"))
RIGIDITES = (("acier", "Profilé acier"), ("bois", "Section bois"), ("libre", "E et I"), ("sans", "Sans"))
GRANDEURS = ("M", "V", "f", "R")
E_ACIER = 210000.0                     # MPa — EN 1993-1-1 §3.2.6
FAMILLES_ACIER = [f for f, _ in KA.familles_presentes()]
SECTIONS_BOIS = [f"{b}x{h}" for b, h in tuple(BD.SECTIONS_MASSIF) + tuple(BD.SECTIONS_LAMELLE)]
n, r, m = A.n, A.r, A.m

SOURCES = ("Formules des cas de charge : <i>Techniques de l'ingénieur</i> C 2 060, « Formulaire de la résistance des matériaux — poutres droites » ; "
           "Roark, <i>Formulas for Stress and Strain</i>, tableau 8.1 ; Timoshenko, <i>Résistance des matériaux</i>, t. 1.",
           "Paramètres initiaux (Clebsch ; Macaulay, 1919) pour V(x), M(x), f(x) ; théorème des trois moments (Clapeyron, 1857) pour la poutre continue.",
           "Profilés : catalogue de la Bibliothèque Acier (E = 210 000 MPa, EN 1993-1-1 §3.2.6). Bois : E<sub>0,mean</sub> de l'EN 338:2016 et de l'EN 14080:2013.")


# ================================================================================================ lecture
def lire(params: Optional[Dict]) -> Tuple[B.Lecture, Dict]:
    L = B.Lecture(params)
    o: Dict = {}
    o["modele"] = L.choix("modele", [k for k, _ in MODELES], "appuis")
    if o["modele"] == "continue":
        nt = L.nombre("nt", 2, 2, P.TRAVEES_MAX, entier=True)
        L1 = L.nombre("L1", 6.0, 0.1, 200.0)
        o["Ls"] = [L1] + [L.nombre(f"L{i + 1}", L1, 0.1, 200.0) for i in range(1, nt)]
    else:
        o["Ls"] = [L.nombre("L1", 6.0, 0.1, 200.0)]
    Lt = sum(o["Ls"])
    nch = L.nombre("n", 1, 0, P.CHARGES_MAX, entier=True)
    ch = []
    for i in range(1, nch + 1):
        t = L.choix(f"t{i}", [k for k, _ in TYPES], "q")
        v = L.nombre(f"v{i}", DEFAUT_V[t], -1e6, 1e6)
        if t in ("P", "C"):
            ch.append({"type": t, "v": v, "a": L.nombre(f"a{i}", Lt / 2.0, 0.0, Lt)})
            continue
        a = L.nombre(f"a{i}", 0.0, 0.0, Lt)
        # b absent = « jusqu'au bout de la poutre » : une charge sur toute la portée SUIT la portée quand on la change
        b = L.nombre(f"b{i}", Lt, 0.0, Lt)
        if not a < b:
            L._dit(f"a{i}", "et b remis à toute la poutre : il faut a < b")
            a, b = 0.0, Lt
            L.out[f"a{i}"] = 0.0
        if b >= Lt:
            L.out.pop(f"b{i}", None)
        else:
            L.out[f"b{i}"] = b
        c = {"type": t, "v": v, "a": a, "b": b}
        if t == "tri":
            c["sens"] = L.choix(f"s{i}", [k for k, _ in SENS], "croissante")
        ch.append(c)
    o["charges"] = ch
    # ---- la rigidité (pour la flèche seulement)
    o["rig"] = L.choix("rig", [k for k, _ in RIGIDITES], "acier")
    o["E"] = o["I"] = o["W"] = None
    if o["rig"] == "acier":
        fam = L.choix("fam", FAMILLES_ACIER, "IPE")
        noms = [p["nom"] for p in KA.profils() if p["famille"] == fam]
        defaut = "IPE 300" if "IPE 300" in noms else noms[len(noms) // 2]
        if L.brut.get("profil") not in noms:                 # la famille vient de changer : son profilé médian, sans message
            L.brut.pop("profil", None)
        nom = L.choix("profil", noms, defaut)
        axe = L.choix("axe", ("y", "z"), "y")
        p = KA.par_nom()[nom]
        o.update(fam=fam, noms=noms, profil=nom, axe=axe, E=E_ACIER, I=p[f"I{axe}"], W=p.get(f"Wel_{axe}"), p=p)
    elif o["rig"] == "bois":
        classe = L.choix("classe", list(BD.CLASSES), "C24")
        sec = L.choix("sec", SECTIONS_BOIS + ["libre"], "75x225" if "75x225" in SECTIONS_BOIS else SECTIONS_BOIS[0])
        if sec == "libre":
            bw, hw = L.nombre("bw", 75.0, 10.0, 2000.0), L.nombre("hw", 225.0, 10.0, 3000.0)
        else:
            bw, hw = (float(x) for x in sec.split("x"))
        pose = L.choix("pose", ("chant", "plat"), "chant")
        b_, h_ = (bw, hw) if pose == "chant" else (hw, bw)
        o.update(classe=classe, sec=sec, bw=bw, hw=hw, pose=pose, E=BD.CLASSES[classe]["E0_mean"], I=b_ * h_ ** 3 / 12.0 / 1e4, W=b_ * h_ ** 2 / 6.0 / 1e3,
                 b_=b_, h_=h_)
    elif o["rig"] == "libre":
        o.update(E=L.nombre("E", E_ACIER, 1.0, 1e7), I=L.nombre("I", 8356.0, 1e-3, 1e9))
    o["EI"] = o["E"] * o["I"] * 1e-5 if o["E"] else None          # kN·m² : E [MPa] × I [cm⁴] × 10⁻⁵
    o["x"] = L.nombre("x", None, 0.0, Lt, vide=None)
    o["g"] = L.choix("g", GRANDEURS, "M")
    return L, o


def noms_charges(charges: Sequence[Dict]) -> List[str]:
    return [f"{LETTRE[c['type']]}_{i + 1}" for i, c in enumerate(charges)]


# ================================================================================================ la barre
def _barre(o: Dict) -> str:
    modele = B._champ("modele", "Modèle", B._segments("modele", o["modele"], MODELES))
    if o["modele"] == "continue":
        portees = B._champ("nt", "Travées", B._select("nt", len(o["Ls"]), [(k, str(k)) for k in range(2, P.TRAVEES_MAX + 1)]))
        portees += "".join(B._champ(f"L{i + 1}", f"L<sub>{i + 1}</sub>", B._num(f"L{i + 1}", x, "m")) for i, x in enumerate(o["Ls"]))
    else:
        portees = B._champ("L1", "Portée L", B._num("L1", o["Ls"][0], "m"))
    l1 = B._groupe("Modèle", modele) + B._groupe("Portées", portees)
    rig = B._champ("rig", "Pour la flèche", B._segments("rig", o["rig"], RIGIDITES),
                   "Les efforts d'une poutre de rigidité constante ne dépendent pas de E·I : la rigidité ne sert qu'à la flèche (et à la contrainte).")
    if o["rig"] == "acier":
        rig += (B._champ("fam", "Famille", B._select("fam", o["fam"], [(f, f) for f in FAMILLES_ACIER]))
                + B._champ("profil", "Profilé", B._select("profil", o["profil"], [(x, x) for x in o["noms"]]))
                + B._champ("axe", "Flexion autour de", B._segments("axe", o["axe"], (("y", "y (axe fort)"), ("z", "z (axe faible)")))))
    elif o["rig"] == "bois":
        rig += (B._champ("classe", "Classe", B._select("classe", o["classe"], [(k, k) for k in BD.CLASSES]))
                + B._champ("sec", "Section b × h", B._select("sec", o["sec"], [(s, s.replace("x", " × ") + " mm") for s in SECTIONS_BOIS] + [("libre", "autre…")])))
        if o["sec"] == "libre":
            rig += B._champ("bw", "b", B._num("bw", o["bw"], "mm")) + B._champ("hw", "h", B._num("hw", o["hw"], "mm"))
        rig += B._champ("pose", "Pose", B._segments("pose", o["pose"], (("chant", "sur chant"), ("plat", "à plat"))))
    elif o["rig"] == "libre":
        rig += B._champ("E", "E", B._num("E", o["E"], "MPa")) + B._champ("I", "I", B._num("I", o["I"], "cm⁴"))
    return B._barre(l1, B._groupe("Rigidité", rig))


# ================================================================================================ les charges (cadre)
def _params_sans(o_params: Dict, i: int, nch: int) -> Dict:
    """Les paramètres après le RETRAIT de la charge i : les suivantes remontent d'un rang."""
    d: Dict = {"n": nch - 1}
    for j in range(i, nch):
        for c in ("t", "v", "a", "b", "s"):
            d[f"{c}{j}"] = o_params.get(f"{c}{j + 1}")
    for c in ("t", "v", "a", "b", "s"):
        d[f"{c}{nch}"] = None
    return d


def _charges(o: Dict, params: Dict) -> str:
    ch = o["charges"]
    noms = noms_charges(ch)
    Lt = sum(o["Ls"])
    lignes = ""
    for i, (c, nom) in enumerate(zip(ch, noms), 1):
        t = c["type"]
        cases = A.menu_champ(f"t{i}", t, TYPES) + A.champ(f"v{i}", c["v"], UNITE[t], m(nom))
        if t in ("P", "C"):
            cases += A.champ(f"a{i}", c["a"], "m", "en x =")
        else:
            cases += A.champ(f"a{i}", c["a"], "m", "de") + A.champ(f"b{i}", c["b"], "m", "à")
            if t == "tri":
                cases += A.menu_champ(f"s{i}", c["sens"], SENS)
        lignes += (f'<div class="rdm-ch" data-charge="{i}">{cases}'
                   f'{A.bouton("", _params_sans(params, i, len(ch)), "trash-2", "btn icon ghost sm", f"Retirer la charge {i}")}</div>')
    if not ch:
        lignes = '<p class="rdm-vide">Aucune charge : la poutre est au repos. Ajouter une charge ci-dessous.</p>'
    ajout = ""
    if len(ch) < P.CHARGES_MAX:
        k = len(ch) + 1
        ajout = "".join(A.bouton(NOM_TYPE[t], {"n": k, f"t{k}": t, f"v{k}": None, f"a{k}": None, f"b{k}": None, f"s{k}": None}, "plus", "btn sm")
                        for t, _ in TYPES)
    aide = ('<p class="rdm-aide">Vers le bas : positif. Les charges se superposent (jusqu\'à huit) et se glissent sur le dessin. '
            f'Abscisses depuis A, de 0 à {r(Lt)} m.</p>')
    return A.bloc("Charges", f'{lignes}<div class="rdm-plus">{ajout}</div>{aide}', "rdm-charges")


# ================================================================================================ la feuille : cas d'école
# (clé de la valeur du moteur, formule générale) — les symboles q, P, L, a, b, c, R_B, x_f sont remplacés par leurs valeurs
ECOLE = {
    "appuis_q": ("deux appuis, charge uniforme sur toute la portée",
                 [("R_A", "q·L/2"), ("R_B", "q·L/2"), ("M_max", "q·L²/8"), ("EIf_max", "5·q·L⁴/384")]),
    "appuis_P": ("deux appuis, charge ponctuelle en a (b = L − a ; c = min(a ; b))",
                 [("R_A", "P·b/L"), ("R_B", "P·a/L"), ("M_max", "P·a·b/L"), ("EIf_a", "P·a²·b²/(3·L)"), ("EIf_max", "P·c·(L² − c²)^1,5/(9·√3·L)")]),
    "appuis_P_milieu": ("deux appuis, charge ponctuelle au milieu",
                        [("R_A", "P/2"), ("R_B", "P/2"), ("M_max", "P·L/4"), ("EIf_max", "P·L³/48")]),
    "appuis_tri+": ("deux appuis, charge triangulaire 0 → q sur toute la portée (x_f = L·√(1 − √(8/15)) depuis l'extrémité déchargée)",
                    [("R_A", "q·L/6"), ("R_B", "q·L/3"), ("M_max", "q·L²/(9·√3)"), ("EIf_max", "q·x_f·(3·x_f⁴ − 10·L²·x_f² + 7·L⁴)/(360·L)")]),
    "appuis_tri-": ("deux appuis, charge triangulaire q → 0 sur toute la portée (x_f = L·√(1 − √(8/15)) depuis l'extrémité déchargée)",
                    [("R_A", "q·L/3"), ("R_B", "q·L/6"), ("M_max", "q·L²/(9·√3)"), ("EIf_max", "q·x_f·(3·x_f⁴ − 10·L²·x_f² + 7·L⁴)/(360·L)")]),
    "console_q": ("console, charge uniforme sur toute la portée", [("R_A", "q·L"), ("M_A", "−q·L²/2"), ("EIf_max", "q·L⁴/8")]),
    "console_P": ("console, charge ponctuelle en a", [("R_A", "P"), ("M_A", "−P·a"), ("EIf_a", "P·a³/3"), ("EIf_max", "P·a²·(3·L − a)/6")]),
    "console_tri": ("console, charge triangulaire q → 0 (maximale à l'encastrement)", [("R_A", "q·L/2"), ("M_A", "−q·L²/6"), ("EIf_max", "q·L⁴/30")]),
    "enc_app_q": ("encastrée–appuyée, charge uniforme",
                  [("M_A", "−q·L²/8"), ("R_A", "5·q·L/8"), ("R_B", "3·q·L/8"), ("M_max", "9·q·L²/128"), ("EIf_max", "(39 + 55·√33)·q·L⁴/65536")]),
    "enc_app_P": ("encastrée–appuyée, charge ponctuelle en a depuis l'encastrement (b = L − a)",
                  [("M_A", "−P·a·b·(L + b)/(2·L²)"), ("R_B", "P·a²·(3·L − a)/(2·L³)"), ("R_A", "P − R_B"), ("M_max", "R_B·b")]),
    "enc_app_P_milieu": ("encastrée–appuyée, charge ponctuelle au milieu",
                         [("M_A", "−3·P·L/16"), ("R_A", "11·P/16"), ("R_B", "5·P/16"), ("M_max", "5·P·L/32"), ("EIf_max", "P·L³/(48·√5)")]),
    "bi_enc_q": ("bi-encastrée, charge uniforme",
                 [("M_A", "−q·L²/12"), ("M_B", "−q·L²/12"), ("R_A", "q·L/2"), ("R_B", "q·L/2"), ("M_max", "q·L²/24"), ("EIf_max", "q·L⁴/384")]),
    "bi_enc_P": ("bi-encastrée, charge ponctuelle en a (b = L − a)",
                 [("M_A", "−P·a·b²/L²"), ("M_B", "−P·a²·b/L²"), ("R_A", "P·b²·(3·a + b)/L³"), ("R_B", "P·a²·(a + 3·b)/L³"), ("M_max", "2·P·a²·b²/L³"),
                  ("EIf_a", "P·a³·b³/(3·L³)")]),
    "bi_enc_P_milieu": ("bi-encastrée, charge ponctuelle au milieu",
                        [("M_A", "−P·L/8"), ("M_B", "−P·L/8"), ("R_A", "P/2"), ("R_B", "P/2"), ("M_max", "P·L/8"), ("EIf_max", "P·L³/192")]),
    "continue_2_q": ("deux travées égales, charge uniforme partout",
                     [("M_B", "−q·L²/8"), ("R_A", "3·q·L/8"), ("R_B", "5·q·L/4"), ("R_C", "3·q·L/8"), ("M_max", "9·q·L²/128"),
                      ("EIf_max", "(39 + 55·√33)·q·L⁴/65536")]),
    "continue_3_q": ("trois travées égales, charge uniforme partout",
                     [("M_B", "−q·L²/10"), ("M_C", "−q·L²/10"), ("R_A", "0,4·q·L"), ("R_B", "1,1·q·L"), ("R_C", "1,1·q·L"), ("R_D", "0,4·q·L"),
                      ("M_max", "0,08·q·L²"), ("M_milieu", "0,025·q·L²")]),
}
NOM_GRANDEUR = {"M_max": "moment maximal en travée", "M_milieu": "moment au milieu de la travée centrale", "EIf_max": "flèche maximale",
                "EIf_a": "flèche sous la charge"}
_JETON = re.compile(r"R_B|x_f|[qPLabc]")


def _remplacer(formule: str, vals: Dict[str, float]) -> str:
    """La formule générale où chaque symbole prend sa valeur : « 5·q·L⁴/384 » → « 5 × 10 × 6⁴ / 384 »."""
    s = _JETON.sub(lambda mm: n(vals[mm.group()]), formule)
    return s.replace("·", " × ").replace("/", " / ")


def _flèche_ligne(sym: str, nom: str, formule_EI: str, valeurs_EI: str, EIf: float, o: Dict, cle: str, ref: str, on: bool = False) -> Dict:
    """Une ligne de flèche : avec rigidité, f = (…)/(E·I) en mm ; sans, E·I·f en kN·m³."""
    if o["EI"]:
        return A.ligne(sym, nom, f"({formule_EI})/(E·I)" if formule_EI else "", f"({valeurs_EI}) / {n(o['EI'], 1)} × 1000" if valeurs_EI else "",
                       r(EIf / o["EI"] * 1000.0, 2), "mm", ref, on, cle)
    return A.ligne(f"E·I·{sym}", nom, formule_EI, valeurs_EI, r(EIf, 2), "kN·m³", ref, on, cle)


def _lignes_ecole(sol, ce: Dict, o: Dict) -> Tuple[str, List[Dict]]:
    c = sol.charges[0]
    cle = ce["cle"]
    if cle == "appuis_tri":
        cle += "+" if c["sens"] == "croissante" else "-"
    titre, formules = ECOLE[cle]
    L = sol.travees[0].L
    a = c.get("a", 0.0)
    vals = {"q": c["v"], "P": c["v"], "L": L, "a": a, "b": L - a, "c": min(a, L - a), "R_B": ce["valeurs"].get("R_B", 0.0),
            "x_f": L * math.sqrt(1.0 - math.sqrt(8.0 / 15.0))}
    out = []
    for k, f in formules:
        v = ce["valeurs"][k]
        val = _remplacer(f, vals)
        pos = ""
        if k == "M_max":
            pos = f" · en x = {r(ce['general']['x_M'])} m" + (" (et sa symétrique)" if cle.startswith("continue") else "")
        if k == "EIf_max" and "x_f" in ce["general"]:
            pos = f" · en x = {r(ce['general']['x_f'])} m"
        if k.startswith("EIf"):
            out.append(_flèche_ligne("f_max" if k == "EIf_max" else "f(a)", NOM_GRANDEUR[k] + pos, f, val, v, o, k, "formulaire", o["g"] == "f" and k == "EIf_max"))
        elif k.startswith("R_"):
            out.append(A.ligne(k, f"réaction en {k[2:]}", f, val, r(v), "kN", "formulaire", False, k))
        else:
            nom = NOM_GRANDEUR.get(k, f"moment en {k[2:]}" + (" (encastrement)" if sol.modele != "continue" else " (sur appui)")) + pos
            out.append(A.ligne(k, nom, f, val, r(v), "kN·m", "formulaire", o["g"] == "M" and k == _cle_M(sol), k))
    return titre, out


def _cle_M(sol) -> str:
    """La clé de la ligne du moment GOUVERNANT (le plus grand en valeur absolue)."""
    e = sol.extremes
    if abs(e["M_max"][0]) >= abs(e["M_min"][0]):
        return "M_max"
    app = min(sol.appuis, key=lambda a: abs(a["x"] - e["M_min"][1]))
    return f"M_{app['nom']}" if abs(app["x"] - e["M_min"][1]) < 1e-9 * sol.L else "M_min"


# ================================================================================================ la feuille : superposition
def _terme(sym: str, num: str, signe: float = 1.0) -> Tuple[str, str]:
    return sym, num


def _joindre(termes: Sequence[str]) -> str:
    """« a + b − c » depuis des termes qui portent leur signe (« −C/L » commence par un moins)."""
    s = ""
    for t in termes:
        if not t:
            continue
        if not s:
            s = t
        elif t.startswith(A.MOINS):
            s += f" {A.MOINS} {t[1:]}"
        else:
            s += f" + {t}"
    return s or "0"


def _iso_termes(t, mo: Dict, nom: str) -> Dict[str, Tuple[str, str]]:
    """Pour un morceau de charge de la travée : {R, wA, wB : (terme général, terme en valeurs)} — le formulaire."""
    L = t.L
    typ = mo["type"]
    if typ == "P":
        a = mo["a"]
        b = L - a
        return {"R": (f"{nom}·(L − a)/L", f"{n(mo['v'])} × ({n(L)} − {n(a)}) / {n(L)}"),
                "wA": (f"{nom}·a·b·(L + b)/(6·L)", f"{n(mo['v'])} × {n(a)} × {n(b)} × ({n(L)} + {n(b)}) / (6 × {n(L)})"),
                "wB": (f"{nom}·a·b·(L + a)/(6·L)", f"{n(mo['v'])} × {n(a)} × {n(b)} × ({n(L)} + {n(a)}) / (6 × {n(L)})")}
    if typ == "C":
        a = mo["a"]
        b = L - a
        return {"R": (f"{A.MOINS}{nom}/L", f"{A.MOINS}{n(mo['v'])} / {n(L)}"),
                "wA": (f"{nom}·(3·b² − L²)/(6·L)", f"{n(mo['v'])} × (3 × {n(b)}² − {n(L)}²) / (6 × {n(L)})"),
                "wB": (f"{nom}·(L² − 3·a²)/(6·L)", f"{n(mo['v'])} × ({n(L)}² − 3 × {n(a)}²) / (6 × {n(L)})")}
    a, b, q1, q2 = mo["a"], mo["b"], mo["q1"], mo["q2"]
    total = a == 0.0 and abs(b - L) < 1e-12 * L
    if typ == "q" or abs(q1 - q2) < 1e-12 * max(abs(q1), abs(q2), 1e-300):
        if total:
            return {"R": (f"{nom}·L/2", f"{n(q1)} × {n(L)} / 2"), "wA": (f"{nom}·L³/24", f"{n(q1)} × {n(L)}³ / 24"),
                    "wB": (f"{nom}·L³/24", f"{n(q1)} × {n(L)}³ / 24")}
        c, xc = b - a, (a + b) / 2.0
        return {"R": (f"{nom}·c·(L − x_c)/L", f"{n(q1)} × {n(c)} × ({n(L)} − {n(xc)}) / {n(L)}"),
                "wA": (f"{nom}·[F(b) − F(a)]/(6·L)", f"{n(q1)} × ({n(P._F(b, L))} − {n(P._F(a, L))}) / (6 × {n(L)})"),
                "wB": (f"{nom}·[G(b) − G(a)]/(6·L)", f"{n(q1)} × ({n(P._G(b, L))} − {n(P._G(a, L))}) / (6 × {n(L)})")}
    if total and (q1 == 0.0 or q2 == 0.0):
        q = q2 if q1 == 0.0 else q1
        cA, cB, rA = ("7", "8", "6") if q1 == 0.0 else ("8", "7", "3")
        return {"R": (f"{nom}·L/{rA}", f"{n(q)} × {n(L)} / {rA}"), "wA": (f"{cA}·{nom}·L³/360", f"{cA} × {n(q)} × {n(L)}³ / 360"),
                "wB": (f"{cB}·{nom}·L³/360", f"{cB} × {n(q)} × {n(L)}³ / 360")}
    # trapèze ou triangle partiel : q_g en a, q_d en b, pente k ; résultante W en x_c
    k = (q2 - q1) / (b - a)
    Wr = (q1 + q2) / 2.0 * (b - a)
    xc = (b - a) * (q1 * (2 * a + b) + q2 * (a + 2 * b)) / 6.0 / Wr if Wr else (a + b) / 2.0
    dF, dG = P._F(b, L) - P._F(a, L), P._G(b, L) - P._G(a, L)
    dF1, dG1 = P._F1(b, L) - P._F1(a, L), P._G1(b, L) - P._G1(a, L)
    return {"R": ("W·(L − x_c)/L", f"{n(Wr)} × ({n(L)} − {n(xc)}) / {n(L)}"),
            "wA": ("{q_g·[F(b) − F(a)] + k·([F_1(b) − F_1(a)] − a·[F(b) − F(a)])}/(6·L)",
                   f"({n(q1)} × {n(dF)} + {n(k)} × ({n(dF1)} − {n(a)} × {n(dF)})) / (6 × {n(L)})"),
            "wB": ("{q_g·[G(b) − G(a)] + k·([G_1(b) − G_1(a)] − a·[G(b) − G(a)])}/(6·L)",
                   f"({n(q1)} × {n(dG)} + {n(k)} × ({n(dG1)} − {n(a)} × {n(dG)})) / (6 × {n(L)})")}


def _groupe_iso(sol, noms: Sequence[str]) -> List:
    """Par travée : R_A°, E·I·ω_A, E·I·ω_B de la travée rendue isostatique, charge par charge (formulaire)."""
    groupes = []
    for t in sol.travees:
        if not t.morceaux:
            continue
        tt = [_iso_termes(t, mo, noms[mo["i"]]) for mo in t.morceaux]
        suf = "" if len(sol.travees) == 1 else f",{sol.travees.index(t) + 1}"
        lignes = [A.ligne(f"R_{t.nom_A}°", "réaction isostatique", _joindre([x["R"][0] for x in tt]), _joindre([x["R"][1] for x in tt]), r(t.R_A0), "kN",
                          "formulaire", False, f"RA0{suf}"),
                  A.ligne(f"E·I·ω_A{suf}", f"rotation isostatique en {t.nom_A}", _joindre([x["wA"][0] for x in tt]), _joindre([x["wA"][1] for x in tt]),
                          r(t.wA0, 3), "kN·m²", "formulaire", False, f"wA0{suf}"),
                  A.ligne(f"E·I·ω_B{suf}", f"rotation isostatique en {t.nom_B}", _joindre([x["wB"][0] for x in tt]), _joindre([x["wB"][1] for x in tt]),
                          r(t.wB0, 3), "kN·m²", "formulaire", False, f"wB0{suf}")]
        sous = (f"L = {r(t.L)} m · abscisses locales depuis {t.nom_A}" if len(sol.travees) > 1 else "chaque charge seule, puis la somme")
        if any(("F(" in x["wA"][0]) for x in tt):
            sous += " · F(ξ) = L²ξ² − Lξ³ + ξ⁴/4 ; G(ξ) = L²ξ²/2 − ξ⁴/4"
        if any(("F_1" in x["wA"][0]) for x in tt):
            sous += " ; F_1(ξ) = 2L²ξ³/3 − 3Lξ⁴/4 + ξ⁵/5 ; G_1(ξ) = L²ξ³/3 − ξ⁵/5"
        groupes.append((f"Travée {t.nom_A}{t.nom_B} isostatique" if len(sol.travees) > 1 else "Travée isostatique (sur deux appuis)", sous, lignes))
    return groupes


def _groupe_liaisons(sol, o: Dict) -> List:
    t0 = sol.travees[0]
    L = t0.L
    gM = o["g"] == "M"
    cm = _cle_M(sol)
    if sol.modele == "enc_app":
        return [("Encastrement en A", "la rotation en A est nulle : E·I·ω_A + M_A·L/3 = 0",
                 [A.ligne("M_A", "moment d'encastrement", f"{A.MOINS}3·E·I·ω_A/L", f"{A.MOINS}3 × {n(t0.wA0)} / {n(L)}", r(t0.M_A), "kN·m",
                          "méthode des forces", gM and cm == "M_A", "M_A")])]
    if sol.modele == "bi_enc":
        return [("Encastrements en A et B", "rotations nulles : E·I·ω_A + M_A·L/3 + M_B·L/6 = 0 ; E·I·ω_B + M_A·L/6 + M_B·L/3 = 0",
                 [A.ligne("M_A", "moment d'encastrement en A", f"{A.MOINS}2·(2·E·I·ω_A − E·I·ω_B)/L", f"{A.MOINS}2 × (2 × {n(t0.wA0)} − {n(t0.wB0)}) / {n(L)}",
                          r(t0.M_A), "kN·m", "méthode des forces", gM and cm == "M_A", "M_A"),
                  A.ligne("M_B", "moment d'encastrement en B", f"{A.MOINS}2·(2·E·I·ω_B − E·I·ω_A)/L", f"{A.MOINS}2 × (2 × {n(t0.wB0)} − {n(t0.wA0)}) / {n(L)}",
                          r(t0.M_B), "kN·m", "méthode des forces", gM and cm == "M_B", "M_B")])]
    if sol.modele == "console":
        termes_s, termes_n = [], []
        for mo, nom in ((mo, noms_charges(sol.charges)[mo["i"]]) for mo in t0.morceaux):
            if mo["type"] == "P":
                termes_s.append(f"{nom}·a")
                termes_n.append(f"{n(mo['v'])} × {n(mo['a'])}")
            elif mo["type"] == "C":
                termes_s.append(nom)
                termes_n.append(n(mo["v"]))
            else:
                Wr = (mo["q1"] + mo["q2"]) / 2.0 * (mo["b"] - mo["a"])
                xc = (mo["b"] - mo["a"]) * (mo["q1"] * (2 * mo["a"] + mo["b"]) + mo["q2"] * (mo["a"] + 2 * mo["b"])) / 6.0 / Wr if Wr else 0.0
                termes_s.append("W·x_c")
                termes_n.append(f"{n(Wr)} × {n(xc)}")
        return [("Encastrement en A", "équilibre : la console est isostatique (W : résultante d'une charge répartie, x_c : son centre de gravité)",
                 [A.ligne("R_A", "réaction", "Σ charges", "", r(t0.R_A), "kN", "équilibre", False, "R_A"),
                  A.ligne("M_A", "moment d'encastrement", f"{A.MOINS}(Σ F·x + Σ C)", f"{A.MOINS}({_joindre(termes_n)})" if termes_n else "", r(t0.M_A), "kN·m",
                          "équilibre", gM and cm == "M_A", "M_A")])]
    if sol.modele == "continue":
        lignes = []
        for e in sol.equations:
            i = "ABCDEF".index(e["appui"])
            g_, d_ = "ABCDEF"[i - 1], "ABCDEF"[i + 1]
            lignes.append(A.ligne(f"appui {e['appui']}", "trois moments",
                                  f"M_{g_}·L_{i} + 2·M_{e['appui']}·(L_{i} + L_{i + 1}) + M_{d_}·L_{i + 1} = {A.MOINS}6·(E·I·ω_B,{i} + E·I·ω_A,{i + 1})",
                                  f"{A.MOINS}6 × ({n(e['wB_g'])} + {n(e['wA_d'])})", r(e["second"], 3), "kN·m²", "Clapeyron", False, f"eq{e['appui']}"))
        for e in sol.equations:
            lignes.append(A.ligne(f"M_{e['appui']}", f"moment sur l'appui {e['appui']}", "solution du système",
                                  f"contrôle : {n(e['M_g'])} × {n(e['L_g'])} + 2 × {n(e['M'])} × ({n(e['L_g'])} + {n(e['L_d'])}) + {n(e['M_d'])} × {n(e['L_d'])}",
                                  r(e["M"]), "kN·m", "Clapeyron", gM and cm == f"M_{e['appui']}", f"M_{e['appui']}"))
        return [("Continuité sur les appuis", "la rotation est continue sur chaque appui intermédiaire ; M = 0 aux appuis de rive", lignes)]
    return []


def _groupe_reactions(sol) -> List:
    lignes = []
    tr = sol.travees
    if sol.modele == "console":
        return []
    for j, a in enumerate(sol.appuis):
        g_ = tr[j - 1] if j > 0 else None
        d_ = tr[j] if j < len(tr) else None
        fs, vs = [], []
        if g_ is not None:
            fs.append(f"Σ F_{g_.nom_A}{g_.nom_B} − R_{g_.nom_A}°" + (f" − (M_{g_.nom_B} − M_{g_.nom_A})/L" if (g_.M_A or g_.M_B) else ""))
            vs.append(f"{n(g_.charge_totale())} − {n(g_.R_A0)}" + (f" − ({n(g_.M_B)} − {n(g_.M_A)}) / {n(g_.L)}" if (g_.M_A or g_.M_B) else ""))
        if d_ is not None:
            fs.append(f"R_{d_.nom_A}°" + (f" + (M_{d_.nom_B} − M_{d_.nom_A})/L" if (d_.M_A or d_.M_B) else ""))
            vs.append(f"{n(d_.R_A0)}" + (f" + ({n(d_.M_B)} − {n(d_.M_A)}) / {n(d_.L)}" if (d_.M_A or d_.M_B) else ""))
        lignes.append(A.ligne(f"R_{a['nom']}", f"réaction en {a['nom']}", _joindre(fs), _joindre(vs), r(a["R"]), "kN", "équilibre", False, f"R_{a['nom']}"))
    return [("Réactions d'appui", "vers le haut : positif · Σ R = Σ charges (contrôlé)", lignes)]


def _v0(t) -> str:
    """Le nom du tranchant en x = 0 d'une travée : la réaction R_A pour la première ; « V_Bd » (à droite de B) pour une
    travée intérieure — la réaction de l'appui B, elle, additionne les deux travées qu'il porte."""
    return f"R_{t.nom_A}" if t.x0 == 0.0 else f"V_{t.nom_A}d"


def _expr_section(t, xl: float, noms: Sequence[str], quoi: str, droite: bool = True) -> Tuple[str, str]:
    """L'écriture des paramètres initiaux dans la section d'abscisse LOCALE xl : (forme générale, valeurs). ``quoi`` :
    « V », « M » ou « f » (E·I·f)."""
    L = t.L
    x = n(xl)
    v0 = _v0(t)
    if quoi == "V":
        fs, vs = [v0], [n(t.R_A)]
    elif quoi == "M":
        fs, vs = [f"M_{t.nom_A}", f"{v0}·x"], [n(t.M_A), f"{n(t.R_A)} × {x}"]
        if not t.M_A:
            fs, vs = fs[1:], vs[1:]
    else:
        fs, vs = ["E·I·ω_0·x", f"{A.MOINS}M_{t.nom_A}·x²/2", f"{A.MOINS}{v0}·x³/6"], [f"{n(t.w0)} × {x}", f"{A.MOINS}{n(t.M_A)} × {x}² / 2", f"{A.MOINS}{n(t.R_A)} × {x}³ / 6"]
        if not t.M_A:
            del fs[1], vs[1]
    mo_ = A.MOINS
    for mo in t.morceaux:
        nom = noms[mo["i"]]
        a = mo["a"]
        apres = xl > a or (droite and xl >= a and mo["type"] in ("P", "C") and quoi != "f")
        if not apres:
            continue
        da = f"({x} − {n(a)})"
        if mo["type"] == "P":
            v = n(mo["v"])
            if quoi == "V":
                fs.append(f"{mo_}{nom}"); vs.append(f"{mo_}{v}")
            elif quoi == "M":
                fs.append(f"{mo_}{nom}·(x − a)"); vs.append(f"{mo_}{v} × {da}")
            else:
                fs.append(f"{nom}·(x − a)³/6"); vs.append(f"{v} × {da}³ / 6")
        elif mo["type"] == "C":
            v = n(mo["v"])
            if quoi == "M":
                fs.append(nom); vs.append(v)
            elif quoi == "f":
                fs.append(f"{mo_}{nom}·(x − a)²/2"); vs.append(f"{mo_}{v} × {da}² / 2")
        else:
            b, q1, q2 = mo["b"], mo["q1"], mo["q2"]
            k = (q2 - q1) / (b - a)
            uni = abs(k) < 1e-12 * max(abs(q1), abs(q2), 1e-300) / max(b - a, 1e-300) or mo["type"] == "q"
            dedans = xl <= b
            db = f"({x} − {n(b)})"
            if uni:
                q = n(q1)
                if quoi == "V":
                    fs.append(f"{mo_}{nom}·(x − a)" if dedans else f"{mo_}{nom}·(b − a)")
                    vs.append(f"{mo_}{q} × {da}" if dedans else f"{mo_}{q} × ({n(b)} − {n(a)})")
                elif quoi == "M":
                    fs.append(f"{mo_}{nom}·(x − a)²/2" if dedans else f"{mo_}{nom}·[(x − a)² − (x − b)²]/2")
                    vs.append(f"{mo_}{q} × {da}² / 2" if dedans else f"{mo_}{q} × ({da}² − {db}²) / 2")
                else:
                    fs.append(f"{nom}·(x − a)⁴/24" if dedans else f"{nom}·[(x − a)⁴ − (x − b)⁴]/24")
                    vs.append(f"{q} × {da}⁴ / 24" if dedans else f"{q} × ({da}⁴ − {db}⁴) / 24")
            else:
                # densité q_g + k·(x − a) à partir de a ; après b on retire q_d et la pente k
                p = {"V": (1, 2, 1, 2), "M": (2, 3, 2, 6), "f": (4, 5, 24, 120)}[quoi]
                e1, e2 = "⁰¹²³⁴⁵"[p[0]], "⁰¹²³⁴⁵"[p[1]]
                sg = "" if quoi == "f" else mo_
                sg2 = mo_ if quoi == "f" else ""
                e1 = "" if p[0] == 1 else e1
                d1 = "" if p[2] == 1 else f"/{p[2]}"
                d1n = "" if p[2] == 1 else f" / {p[2]}"
                fs.append(f"{sg}q_g·(x − a){e1}{d1}"); vs.append(f"{sg}{n(q1)} × {da}{e1}{d1n}")
                fs.append(f"{sg}k·(x − a){e2}/{p[3]}"); vs.append(f"{sg}{n(k)} × {da}{e2} / {p[3]}")
                if not dedans:
                    fs.append(f"{sg2}q_d·(x − b){e1}{d1}"); vs.append(f"{sg2}{n(q2)} × {db}{e1}{d1n}")
                    fs.append(f"{sg2}k·(x − b){e2}/{p[3]}"); vs.append(f"{sg2}{n(k)} × {db}{e2} / {p[3]}")
    return _joindre(fs), _joindre(vs)


def _groupe_section(sol, o: Dict, noms: Sequence[str], x: float, titre: str, cles: Sequence[str], on: Optional[str] = None, suf: str = "") -> Tuple:
    """Les lignes V, M, f dans la section d'abscisse GLOBALE x (les paramètres initiaux de sa travée)."""
    t, xl = sol._travee(x, True)
    if xl >= t.L and x < sol.L:
        t, xl = sol._travee(x, True)
    e = sol.evaluer(x)
    multi = len(sol.travees) > 1
    sous = (f"travée {t.nom_A}{t.nom_B}, abscisse locale x = {r(x)} − {r(t.x0)} = {r(xl)} m" if multi else f"x = {r(xl)} m depuis A")
    sous += " · ⟨x − a⟩ : un terme n'entre que si la section est à droite de la charge"
    lignes = []
    droite = x < sol.L
    xl_e = xl
    if "V" in cles:
        saut = abs(e["V_g"] - e["V_d"]) > 1e-9 * max(abs(e["V_g"]), abs(e["V_d"]), 1e-300) and 0 < x < sol.L
        f, v = _expr_section(t, xl_e, noms, "V", droite)
        lignes.append(A.ligne("V(x)", "effort tranchant" + (f" (à gauche : {r(e['V_g'])} kN)" if saut else ""), f, v, r(e["V_d"] if droite else e["V_g"]), "kN",
                              "paramètres initiaux", on == "V", f"V{suf}"))
    if "M" in cles:
        f, v = _expr_section(t, xl_e, noms, "M", droite)
        lignes.append(A.ligne("M(x)", "moment fléchissant", f, v, r(e["M_d"] if droite else e["M_g"]), "kN·m", "paramètres initiaux", on == "M", f"M{suf}"))
    if "f" in cles:
        f, v = _expr_section(t, xl_e, noms, "f", droite)
        lignes.append(_flèche_ligne("f(x)", "flèche", f, v, e["EIf"], o, f"f{suf}", "paramètres initiaux", on == "f"))
    return titre, sous, lignes


def _groupe_parametres(sol) -> List:
    """Les paramètres initiaux de chaque travée (ce que V(x), M(x), f(x) emploient)."""
    lignes = []
    multi = len(sol.travees) > 1
    for t in sol.travees:
        suf = f",{sol.travees.index(t) + 1}" if multi else ""
        if sol.modele == "console":
            lignes.append(A.ligne("E·I·ω_0", "rotation à l'encastrement", "0", "", "0", "kN·m²", "encastrement", False, "w0"))
        else:
            f, v = f"E·I·ω_A{suf}", n(t.wA0)
            if t.M_A:
                f += f" + M_{t.nom_A}·L/3"
                v += f" + {n(t.M_A)} × {n(t.L)} / 3"
            if t.M_B:
                f += f" + M_{t.nom_B}·L/6"
                v += f" + {n(t.M_B)} × {n(t.L)} / 6"
            lignes.append(A.ligne(f"E·I·ω_0{suf}", f"rotation en {t.nom_A}" + (f" (travée {t.nom_A}{t.nom_B})" if multi else ""), f, v if (t.M_A or t.M_B) else "",
                                  r(t.w0, 3), "kN·m²", "superposition", False, f"w0{suf}"))
            if t.x0 > 0.0:
                lignes.append(A.ligne(_v0(t), f"tranchant à droite de {t.nom_A}", f"R_{t.nom_A}° + (M_{t.nom_B} − M_{t.nom_A})/L",
                                      f"{n(t.R_A0)} + ({n(t.M_B)} − {n(t.M_A)}) / {n(t.L)}", r(t.R_A), "kN", "superposition", False, f"V0{suf}"))
    return [("Paramètres initiaux", "par travée, x depuis son appui de gauche : V(x) = V_0 − Σ ; M(x) = M_0 + V_0·x − Σ ; "
             "E·I·f(x) = E·I·ω_0·x − M_0·x²/2 − V_0·x³/6 + Σ (V_0 = R_A pour la première travée)", lignes)]


# ================================================================================================ résultats, bande
def _resultats(sol, o: Dict) -> Tuple[str, Dict]:
    """Le bloc des résultats (un clic sur une ligne choisit la grandeur de la bande) et les valeurs clés."""
    e = sol.extremes
    Mg = e["M_max"] if abs(e["M_max"][0]) >= abs(e["M_min"][0]) else e["M_min"]
    Vg = e["V_max"] if abs(e["V_max"][0]) >= abs(e["V_min"][0]) else e["V_min"]
    fg = e["EIf_max"] if abs(e["EIf_max"][0]) >= abs(e["EIf_min"][0]) else e["EIf_min"]
    cle = {"M": Mg, "V": Vg, "f": fg}

    def lig(g, sym, nom, val, unite, pos=""):
        on = o["g"] == g
        return (f'<tr{A.sel({"g": g})} class="rdm-r{" on" if on else ""}" tabindex="0"><th>{m(sym)}</th><td class="nm">{nom}</td>'
                f'<td class="v"><b>{val}</b> {unite}</td><td class="p">{pos}</td></tr>')

    corps = ""
    reac = " · ".join(f'{m("R_" + a["nom"])} = {r(a["R"])}' for a in sol.appuis)
    corps += (f'<tr{A.sel({"g": "R"})} class="rdm-r{" on" if o["g"] == "R" else ""}" tabindex="0"><th>R</th><td class="nm">réactions</td>'
              f'<td class="v" colspan="2"><b>{reac}</b> kN</td></tr>')
    enc = [a for a in sol.appuis if a["type"] == "encastrement"]
    if enc:
        corps += ('<tr class="rdm-r fixe"><th>M</th><td class="nm">encastrement</td><td class="v" colspan="2"><b>'
                  + " · ".join(f'{m("M_" + a["nom"])} = {r(a["M"])}' for a in enc) + "</b> kN·m</td></tr>")
    corps += lig("M", "M_max", "moment en travée", r(e["M_max"][0]), "kN·m", f"x = {r(e['M_max'][1])} m")
    if e["M_min"][0] < 0:
        corps += lig("M", "M_min", "moment négatif", r(e["M_min"][0]), "kN·m", f"x = {r(e['M_min'][1])} m")
    corps += lig("V", "V_max", "effort tranchant (valeur absolue)", r(abs(Vg[0])), "kN", f"x = {r(Vg[1])} m")
    info = {"M": Mg, "V": Vg, "f": fg, "sigma": None, "ratio": None}
    if o["EI"]:
        fmm = fg[0] / o["EI"] * 1000.0
        ratio = ""
        if abs(fmm) > 1e-9:
            portee = max(t.L for t in sol.travees) * (2.0 if sol.modele == "console" else 1.0)
            info["ratio"] = portee * 1000.0 / abs(fmm)
            ratio = f' · L/{C.fr(info["ratio"], 0)}' if info["ratio"] < 1e5 else ""
        corps += lig("f", "f_max", "flèche", r(fmm), "mm", f"x = {r(fg[1])} m{ratio}")
    else:
        corps += lig("f", "E·I·f_max", "flèche × rigidité", r(fg[0]), "kN·m³", f"x = {r(fg[1])} m")
    if o["W"]:
        info["sigma"] = abs(Mg[0]) * 1e3 / o["W"]                        # kN·m → N·mm (×10⁶) / cm³ (×10³) = MPa × 10³/10³
        corps += (f'<tr class="rdm-r fixe"><th>{m("σ_max")}</th><td class="nm">contrainte de flexion</td><td class="v"><b>{r(info["sigma"], 1)}</b> MPa</td>'
                  f'<td class="p">M<sub>max</sub>/W<sub>el</sub></td></tr>')
    return A.bloc("Résultats", f'<table class="rdm-res"><tbody>{corps}</tbody></table>', "rdm-resultats"), info


def _lecture(sol, o: Dict, x: float) -> str:
    """Le bloc « Section lue » : le curseur et V, M, f dans la section."""
    e = sol.evaluer(x)
    Lt = sol.L
    pas = D._pas(Lt)
    saut = abs(e["V_g"] - e["V_d"]) > 1e-9 * max(abs(e["V_g"]), abs(e["V_d"]), 1e-300) and 0 < x < Lt
    V = f"{r(e['V_g'])} | {r(e['V_d'])}" if saut else r(e["V_d"] if x < Lt else e["V_g"])
    Mx = e["M_d"] if x < Lt else e["M_g"]
    fx = (f"{r(e['f'] * 1000.0)} mm" if o["EI"] else f"{r(e['EIf'])} kN·m³ (E·I·f)")
    curseur = (f'<input class="rdm-cur" type="range" data-param="x" min="0" max="{C.nb(Lt, 4).replace(",", ".")}" step="{pas}" value="{x:.4f}" '
               f'aria-label="Abscisse de la section lue">')
    return A.bloc("Section lue", f'<div class="rdm-lu">{A.champ("x", round(x, 4), "m", "x =")}{curseur}</div>'
                  f'<div class="rdm-lu-v"><span>V = <b>{V}</b> kN</span><span>M = <b>{r(Mx)}</b> kN·m</span><span>f = <b>{fx}</b></span></div>', "rdm-lecture")


def _hypotheses(sol, o: Dict) -> List[str]:
    h = [f"{TITRES_MODELE[sol.modele]} · " + (" + ".join(f"{r(t.L)} m" for t in sol.travees) + (f" = {r(sol.L)} m" if len(sol.travees) > 1 else "")),
         "Poutre droite de rigidité E·I constante, matériau élastique linéaire, petites déformations (Euler-Bernoulli : la déformation d'effort tranchant est négligée).",
         "Conventions : x depuis A ; charge vers le bas positive ; V = somme des forces à gauche ; M positif s'il tend la fibre inférieure (dessiné du côté tendu) ; "
         "flèche positive vers le bas ; couple positif dans le sens horaire."]
    if o["rig"] == "acier":
        h.append(f"Rigidité : {C.esc(o['profil'])}, flexion autour de {o['axe']} — E = 210 000 MPa, I<sub>{o['axe']}</sub> = {r(o['I'], 1)} cm⁴"
                 + (f", W<sub>el,{o['axe']}</sub> = {r(o['W'], 1)} cm³" if o["W"] else "") + f" → E·I = {r(o['EI'], 1)} kN·m².")
    elif o["rig"] == "bois":
        h.append(f"Rigidité : bois {C.esc(o['classe'])}, {r(o['bw'], 0)} × {r(o['hw'], 0)} mm {'sur chant' if o['pose'] == 'chant' else 'à plat'} — E<sub>0,mean</sub> = "
                 f"{r(o['E'], 0)} MPa, I = b·h³/12 = {r(o['I'], 1)} cm⁴, W = b·h²/6 = {r(o['W'], 1)} cm³ → E·I = {r(o['EI'], 1)} kN·m². "
                 "Flèche instantanée élastique : ni fluage (k<sub>def</sub>) ni effort tranchant.")
    elif o["rig"] == "libre":
        h.append(f"Rigidité saisie : E = {r(o['E'], 0)} MPa, I = {r(o['I'], 1)} cm⁴ → E·I = {r(o['EI'], 1)} kN·m².")
    else:
        h.append("Sans rigidité : réactions, V et M (qui n'en dépendent pas) ; la flèche est donnée multipliée par E·I.")
    return h


VOLET = [
    ("Conventions de signe", ["x se compte depuis l'appui de gauche A.", "Une charge vers le bas est positive ; une réaction vers le haut est positive.",
                              "V(x) est la somme des forces à gauche de la section : V = dM/dx.",
                              "M(x) est positif quand il tend la fibre inférieure ; les diagrammes le dessinent du côté tendu.",
                              "La flèche f est positive vers le bas : E·I·f″ = −M. Un couple est positif dans le sens horaire."]),
    ("Hypothèses", ["Poutre droite, rigidité E·I constante sur toute la longueur, matériau élastique linéaire.",
                    "Petites déformations, sections planes (Euler-Bernoulli) : la déformation d'effort tranchant est négligée.",
                    "Appuis ponctuels indéformables, sans dénivellation ; encastrements parfaits."]),
    ("Méthode", ["Un cas d'école (une charge, cas usuel) est écrit par sa formule de formulaire ; le moteur vérifie à chaque calcul qu'elle donne le même "
                 "nombre que l'écriture générale.",
                 "Plusieurs charges : superposition — travée isostatique charge par charge, puis égalité des rotations (encastrements) ou théorème des "
                 "trois moments (poutre continue).",
                 "V(x), M(x), f(x) : paramètres initiaux, en formules fermées. Les extremums sont cherchés sur ces formules (V = 0, rotation nulle) ; "
                 "aucune intégration numérique."]),
    ("Non traité dans cette version", ["Porte-à-faux d'une poutre sur appuis ; encastrement d'extrémité d'une poutre continue.",
                                       "Appuis élastiques ou dénivelés ; rigidité variable ; déformation d'effort tranchant.",
                                       "Aucune vérification de résistance : la contrainte σ = M/W<sub>el</sub> est une indication élastique."]),
]


# ================================================================================================ la rubrique
def rendre(params: Optional[Dict], hote: str = "page") -> Dict:
    L, o = lire(params)
    noms = noms_charges(o["charges"])
    barre = _barre(o)
    titre = "Formulaire de poutres"
    try:
        sol = P.resoudre(o["modele"], o["Ls"], o["charges"], o["EI"])
        ce = P.cas_ecole(sol)
    except (P.ErreurDonnees, P.ErreurControle) as ex:
        tableau = f'<div class="rdm rdm-poutres">{A.refus("Calcul refusé", C.esc(str(ex)))}{_charges(o, L.out)}</div>'
        return {"titre": titre, "barre": barre, "tableau": tableau, "bande": "", "pourquoi": A.volet(titre, "", VOLET, SOURCES),
                "sources": A.sources(*SOURCES), "params": L.out, "message": L.message()}
    e = sol.extremes
    res_html, info = _resultats(sol, o)
    x = o["x"] if o["x"] is not None else info["M"][1]
    k_f, unite_f, nd_f = (1000.0 / o["EI"], "mm", 2) if o["EI"] else (1.0, "kN·m³", 2)
    dessin = D.dessin(sol, sol.charges, noms, x, unite_f, k_f, nd_f)
    # ---- la feuille de calcul
    groupes: List = []
    if ce:
        t_e, lignes = _lignes_ecole(sol, ce, o)
        groupes.append((f"Formulaire · {t_e}", "formule classique ; le moteur a vérifié qu'elle égale l'écriture générale ci-dessous", lignes))
    else:
        if sol.modele != "console":
            groupes += _groupe_iso(sol, noms)
        groupes += _groupe_liaisons(sol, o)
        groupes += _groupe_reactions(sol)
    if sol.charges:
        groupes += _groupe_parametres(sol)
        if not ce:
            sur_appui = lambda xx: any(abs(a["x"] - xx) < 1e-9 * sol.L for a in sol.appuis)                # noqa: E731
            for cle_e, nom_e in (("M_max", "Moment maximal en travée"), ("M_min", "Moment négatif en travée")):
                val_e, xm = e[cle_e]
                if val_e != 0.0 and not sur_appui(xm) and (cle_e == "M_max") == (val_e > 0):
                    groupes.append(_groupe_section(sol, o, noms, xm, f"{nom_e} · x = {r(xm)} m (V = 0 ou saut de V)", ("V", "M"),
                                                   "M" if o["g"] == "M" and _cle_M(sol) == cle_e else None, "_" + cle_e))
            xf = info["f"][1]
            groupes.append(_groupe_section(sol, o, noms, xf, f"Flèche extrême · x = {r(xf)} m (rotation nulle)", ("f",), "f" if o["g"] == "f" else None, "_f"))
        groupes.append(_groupe_section(sol, o, noms, x, f"Section lue · x = {r(x)} m", ("V", "M", "f"), None, "_x"))
    if o["W"] and sol.charges:
        groupes.append(("Contrainte de flexion", "indication élastique, sans coefficient",
                        [A.ligne("σ_max", "contrainte (le plus grand moment, en valeur absolue)", "M_max/W_el", f"{n(abs(info['M'][0]))} × 1000 / {n(o['W'], 1)}", r(info["sigma"], 1), "MPa", "Navier", False, "sigma")]))
    hyp = _hypotheses(sol, o)
    entete = A.entete_impression(titre, TITRES_MODELE[sol.modele], hyp)
    feuille = (A.bloc("Feuille de calcul", f'<ul class="rdm-hyp">{"".join(f"<li>{h}</li>" for h in hyp)}</ul>{A.feuille(groupes)}', "rdm-feuille"))
    tableau = (f'<div class="rdm rdm-poutres">{entete}<div class="rdm-g"><figure class="rdm-dessin">{dessin}</figure>{_lecture(sol, o, x)}</div>'
               f'<div class="rdm-d">{_charges(o, L.out)}{res_html}</div><div class="rdm-f">{feuille}</div></div>')
    # ---- la bande
    g = o["g"]
    av = []
    if info["ratio"] is not None and info["ratio"] < 100:
        av.append("Flèche supérieure à L/100 : hors du domaine des petites déformations, le résultat n'est qu'indicatif.")
    if g == "V":
        bande = A.bande(r(abs(info["V"][0])), "kN", "V_max", "effort tranchant maximal (valeur absolue)", f"en x = {r(info['V'][1])} m", av)
    elif g == "f":
        if o["EI"]:
            ratio = f" · L/{C.fr(info['ratio'], 0)}" if info["ratio"] and info["ratio"] < 1e5 else ""
            bande = A.bande(r(info["f"][0] / o["EI"] * 1000.0), "mm", "f_max", "flèche maximale", f"en x = {r(info['f'][1])} m{ratio} · E·I = {r(o['EI'], 1)} kN·m²", av)
        else:
            bande = A.bande(r(info["f"][0]), "kN·m³", "E·I·f_max", "flèche maximale × rigidité", f"en x = {r(info['f'][1])} m · choisir une rigidité pour la flèche en mm", av)
    elif g == "R":
        gros = max(sol.appuis, key=lambda a: abs(a["R"]))
        bande = A.bande(r(gros["R"]), "kN", f"R_{gros['nom']}", "plus grande réaction d'appui",
                        " · ".join(f"{m('R_' + a['nom'])} = {r(a['R'])} kN" for a in sol.appuis), av)
    else:
        cm = _cle_M(sol)
        nomM = ("moment maximal en travée" if cm == "M_max" else "moment négatif maximal" if cm == "M_min"
                else f"moment {'à l' + chr(39) + 'encastrement' if sol.modele != 'continue' else 'sur l' + chr(39) + 'appui'} {cm[2:]} (le plus grand en valeur absolue)")
        sg = f" · σ = {r(info['sigma'], 1)} MPa" if info["sigma"] is not None else ""
        bande = A.bande(r(info["M"][0]), "kN·m", cm, nomM, f"en x = {r(info['M'][1])} m{sg}", av)
    return {"titre": titre, "barre": barre, "tableau": tableau, "bande": bande, "pourquoi": A.volet(titre, TITRES_MODELE[sol.modele], VOLET, SOURCES),
            "sources": A.sources(*SOURCES), "params": L.out, "message": L.message()}

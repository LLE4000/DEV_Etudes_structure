# -*- coding: utf-8 -*-
"""Le module « Cornières » du site : l'état d'une note (une ou plusieurs cornières ancrées), les champs de saisie, le
résultat (transversal, longitudinal), les fenêtres de calcul aux formules substituées, le sélecteur de cornières trié
par masse et les dessins — rendus en fragments HTML.

Le calcul est celui du moteur ``acier/corniere`` (extrait de l'écran Streamlit modules/corniere.py, dont il reproduit
les résultats au bit près — tests/test_corniere_moteur.py). Ce fichier MET EN FORME : il ne calcule rien.

Inventaire de l'écran Streamlit et où chaque chose se retrouve ici (docs/acier/CORNIERE_MOTEUR.md, § 6) :
champs (cornière standard ou saisie libre A, B, t ; type de charge ; V ; L de la barre latérale ; e ; γ_ELU ; s ;
distance au bord ; σ_lim,ELS, τ_lim,ELS, σ_eq,lim,ELU) → cadre Données, sections 1 à 4 ; résultats (t, b, masse ;
efforts, section, contraintes ELS / ELU ; bandeaux ELS, ELU, longitudinal) → lignes du résultat et fenêtres de
calcul ; ancrages « à compléter » et notes → avertissements, hypothèses et pied de la fenêtre longitudinale.
"""
import copy
import math
from typing import Dict, List, Optional, Sequence

from acier.corniere import entree as EN
from acier.corniere import moteur as MO

from . import commun as C
from . import dessin_corniere as DC

MODULE = "corniere"
PAGE = "corniere"

CHARGES = {MO.LINEAIRE: "Linéaire", MO.PONCTUELLE: "Ponctuelle"}
ORIGINES = {"catalogue": "Catalogue", "saisie": "Saisie libre"}
LIB_FAMILLES = {"L": "Ailes égales", "LI": "Ailes inégales"}
NUMERIQUES = tuple(EN.CHAMPS)
NOM_MAX = 60                    # un nom de cornière tient sur une ligne de la liste « Dans cette note »
DEC = 6                         # les champs écrivent la valeur saisie sans l'arrondir (0,0315 reste 0,0315)
# les requêtes de plus pour le parcours (api_web/parcours.py) : les chemins rares que les choix du cadre n'atteignent pas
PARCOURS = [
    {"action": "modifier", "chemin": ["profil"], "valeur": "L 45×45×4,5"},
    {"action": "modifier", "chemin": ["profil"], "valeur": "65 x 100 x 8"},
    {"action": "modifier", "chemin": ["e"], "valeur": "25"},
    {"action": "modifier", "chemin": ["V"], "valeur": "0"},
    {"action": "choisir", "role": "corniere", "nom": "L 250x250x35"},
    {"action": "selecteur", "role": "corniere", "familles": ["LI"]},
]


# ------------------------------------------------------------------ l'état de la note : des cornières
def nouvel_etat() -> Dict:
    return {"version": 1, "cornieres": [MO.nouvelle_corniere()], "actif": 0}


def normaliser(etat: Dict) -> Dict:
    """L'état reçu de l'écran (ou d'un fichier .stab) relu par le moteur ; une cornière illisible lève une erreur
    claire (le routeur la rend, l'écran l'affiche)."""
    cornieres = etat.get("cornieres") if isinstance(etat, dict) else None
    if not cornieres:
        return nouvel_etat()
    if not isinstance(cornieres, list):
        raise ValueError("note abîmée : « cornieres » doit être une liste")
    out = {"version": 1, "cornieres": [MO.normaliser(c) for c in cornieres]}
    out["actif"] = min(max(0, _indice(etat.get("actif"))), len(out["cornieres"]) - 1)
    return out


def _indice(x) -> int:
    """L'indice de la cornière ouverte, lu sans faire tomber la note : un nombre entier fini, sinon la première."""
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        return 0
    return int(x)


def _corniere(etat: Dict) -> Dict:
    return etat["cornieres"][etat["actif"]]


def _nom_libre(cs: List[Dict]) -> str:
    pris = {c.get("nom") for c in cs}
    k = len(cs) + 1
    while f"Cornière {k}" in pris:
        k += 1
    return f"Cornière {k}"


def gerer_elements(etat: Dict, action: str, index: Optional[int] = None) -> None:
    """Les cornières de la note : en ouvrir une, en ajouter, copier l'ouverte, en supprimer une (jamais la dernière)."""
    cs = etat["cornieres"]
    if action == "activer":
        etat["actif"] = min(max(0, _indice(index)), len(cs) - 1)
    elif action == "ajouter":
        c = MO.nouvelle_corniere()
        c["nom"] = _nom_libre(cs)
        cs.append(c)
        etat["actif"] = len(cs) - 1
    elif action == "dupliquer":
        c = copy.deepcopy(cs[etat["actif"]])
        c["nom"] = f"{c.get('nom') or 'Cornière'} (copie)"
        cs.append(c)
        etat["actif"] = len(cs) - 1
    elif action == "supprimer" and len(cs) > 1:
        i = min(max(0, _indice(index)), len(cs) - 1)
        del cs[i]
        if etat["actif"] > i or etat["actif"] >= len(cs):
            etat["actif"] = max(0, etat["actif"] - 1)


# ------------------------------------------------------------------ la saisie
def modifier(etat: Dict, chemin: Sequence, valeur) -> Optional[str]:
    """Écrit une saisie dans la cornière ouverte. Le moteur la relit : une valeur hors du domaine de l'écran Streamlit
    est REFUSÉE avec un message (jamais ramenée en silence à une autre), et la cornière reste intacte."""
    c = _corniere(etat)
    avant = copy.deepcopy(c)
    k = chemin[-1] if chemin else None
    try:
        if len(chemin) != 1:
            return f"Champ inconnu : « {'/'.join(str(x) for x in chemin)} »"
        if k == "standard":
            v = {"catalogue": True, "saisie": False, True: True, False: False, "True": True, "False": False}.get(valeur)
            if v is None:
                return f"Origine de la cornière : « {valeur} » (catalogue ou saisie libre)"
            c["standard"] = v
        elif k == "charge":
            c["charge"] = valeur
        elif k == "nom":
            nom = str(valeur if valeur is not None else "").strip()
            if len(nom) > NOM_MAX:
                return f"Nom trop long : {len(nom)} caractères (au plus {NOM_MAX})"
            c["nom"] = nom
        elif k == "profil":
            nom = EN.nom_catalogue(valeur)
            if nom is None:
                return f"Cornière absente du catalogue : « {valeur} » (ex. L 80×80×8, 100×65×8)"
            c["profil"] = nom
            c["standard"] = True
        elif k in NUMERIQUES:
            x = C.lire_nombre(valeur)
            if x is None:
                return f"Nombre illisible : « {valeur} »"
            c[k] = x
        else:
            return f"Champ inconnu : « {k} »"
        etat["cornieres"][etat["actif"]] = MO.normaliser(c)
    except (ValueError, KeyError, TypeError) as e:
        etat["cornieres"][etat["actif"]] = avant
        return f"Saisie refusée : {e}"
    return None


def choisir_section(etat: Dict, role: str, nom: str) -> None:
    """Le clic dans le sélecteur : la cornière du catalogue est retenue (et la cornière passe en « catalogue »)."""
    c = _corniere(etat)
    n = EN.nom_catalogue(nom)
    if n is None:
        raise ValueError(f"cornière absente du catalogue : « {nom} »")
    c["standard"] = True
    c["profil"] = n
    etat["cornieres"][etat["actif"]] = MO.normaliser(c)


def formulaire(etat: Dict, res: Dict) -> str:
    """Le cadre Données : quatre sections repliables, dans l'ordre de l'écran Streamlit."""
    c, sec = _corniere(etat), res["section"]
    L = C.ligne
    # 1 · la cornière
    s1 = (L("Nom", C.texte(["nom"], c.get("nom"), "ex. Cornière de rive"), full=True)
          + L("Origine", C.segments(["standard"], "catalogue" if c["standard"] else "saisie", list(ORIGINES.items())), full=True))
    if c["standard"]:
        s1 += (L("Désignation", C.texte(["profil"], sec["designation"], "ex. L 80×80×8"), full=True)
               + _fiche(sec)
               + C.aide("La liste des cornières, triée par masse, s'ouvre depuis la ligne <b>Cornière</b> du résultat : "
                        "✓ la retenue, ★ la plus légère qui tient."))
    else:
        s1 += (f'<div class="trio">'
               f'<label><span>A · aile fixée</span>{C.num(["A"], c["A"], "mm", dec=DEC)}</label>'
               f'<label><span>B · aile chargée</span>{C.num(["B"], c["B"], "mm", dec=DEC)}</label>'
               f'<label><span>t · épaisseur</span>{C.num(["t"], c["t"], "mm", dec=DEC)}</label></div>'
               + C.aide("Seule l'épaisseur t entre dans le calcul (modèle lame, comme l'écran Streamlit) ; A et B "
                        "servent au dessin. Domaine de l'écran : A et B de 10 à 300 mm, t de 2 à 20 mm."))
    resume1 = f"{C.esc(sec['designation'])} · t {C.nb(sec['t'])} mm"
    # 2 · les charges
    lin = c["charge"] == MO.LINEAIRE
    s2 = (L("Type", C.segments(["charge"], c["charge"], list(CHARGES.items())), full=True)
          + L("effort tranchant", C.num(["V"], c["V"], "kN/m" if lin else "kN", dec=DEC), "V",
              "Valeur caractéristique (service) ; l'ELU la multiplie par le coefficient global."))
    if not lin:
        s2 += L("réparti sur", C.num(["L"], c["L"], "m", dec=DEC), "L",
                "Longueur d'influence : la charge ponctuelle est répartie sur cette longueur de cornière (0,1 à 5 m) ; "
                "c'était un champ de la barre latérale de l'écran Streamlit.")
    s2 += (L("excentricité", C.num(["e"], c["e"], "cm", dec=DEC), "e",
             "Distance de la charge à la cornière, comptée depuis le dos de l'aile fixée (la face du mur).")
           + L("coefficient ELU", C.num(["gamma_ELU"], c["gamma_ELU"], "", dec=DEC), "γ<sub>ELU</sub>")
           + C.aide("Le moment transversal est pris <b>M = V × e</b>, par mètre de cornière."))
    resume2 = (f"{'linéaire' if lin else 'ponctuelle'} · V {C.nb(c['V'])} {'kN/m' if lin else 'kN'} · e {C.nb(c['e'])} cm")
    # 3 · les ancrages
    s3 = (L("entraxe", C.num(["s"], c["s"], "cm", dec=DEC), "s",
            "Espacement des fixations : la largeur du panneau de la vérification longitudinale.")
          + L("bord du mur", C.num(["bord"], c["bord"], "cm", dec=DEC), "c",
              "Distance de l'âme au bord du mur : saisie de l'écran Streamlit, sans effet sur le calcul (les ancrages "
              "n'y sont pas vérifiés).")
          + C.aide("<b>Ancrages non vérifiés</b>, comme dans l'écran Streamlit : efforts par fixation (cisaillement, traction) "
                   "et leur interaction face aux résistances de l'ETA, entraxes, profondeur, béton, distances au bord et "
                   "épaisseur du support à justifier à part."))
    # 4 · les critères
    s4 = (f'<div class="trio">'
          f'<label><span>σ<sub>lim,ELS</sub></span>{C.num(["sig_lim_ELS"], c["sig_lim_ELS"], "N/mm²", dec=DEC)}</label>'
          f'<label><span>τ<sub>lim,ELS</sub></span>{C.num(["tau_lim_ELS"], c["tau_lim_ELS"], "N/mm²", dec=DEC)}</label>'
          f'<label><span>σ<sub>eq,lim,ELU</sub></span>{C.num(["sig_lim_ELU"], c["sig_lim_ELU"], "N/mm²", dec=DEC)}</label></div>'
          + C.aide("Contrainte équivalente admissible à l'ELS, cisaillement admissible à l'ELS, contrainte équivalente "
                   "admissible à l'ELU : ceux de la note ; par défaut ceux de l'écran Streamlit (acier S235, et "
                   "τ<sub>lim,ELS</sub> = 0,8 N/mm², voir la documentation du moteur)."))
    resume4 = f"{C.nb(c['sig_lim_ELS'])} · {C.nb(c['tau_lim_ELS'])} · {C.nb(c['sig_lim_ELU'])} N/mm²"
    return (C.section(1, "Cornière", resume1, s1, True, "co")
            + C.section(2, "Charges", resume2, s2, True, "ch")
            + C.section(3, "Ancrages", f"s {C.nb(c['s'])} cm", s3, False, "an")
            + C.section(4, "Critères", resume4, s4, False, "cr"))


def _fiche(sec: Dict) -> str:
    """La fiche de la cornière du catalogue (ce que l'écran Streamlit écrivait dans son bandeau « Dimensions »)."""
    lignes = [("Ailes A × B", f"{C.nb(sec['A'])} × {C.nb(sec['B'])} mm"), ("Épaisseur t", f"{C.nb(sec['t'])} mm"),
              ("Congés r<sub>1</sub> / r<sub>2</sub>", f"{C.nb(sec['r1'])} / {C.nb(sec['r2'])} mm"),
              ("Masse", f"{C.fr(sec['masse'], 2)} kg/m"), ("Aire", f"{C.fr(sec['aire'], 2)} cm²")]
    dl = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in lignes)
    fam = LIB_FAMILLES.get(sec.get("famille"), "")
    return (f'<div class="secard"><div class="sh2">{C.esc(sec["designation"])}<span class="muted">{fam} · {C.esc(sec["norme"])}</span></div>'
            f'<dl class="facts">{dl}</dl></div>')


# ------------------------------------------------------------------ les nombres d'un résultat
def _nd(x: float, nd: int, sig: int = 2) -> int:
    """Assez de décimales pour garder ``sig`` chiffres significatifs aux petites valeurs (0,0345 kN·m, pas « 0,03 »)."""
    a = abs(x)
    while a and nd < 6 and a < 10 ** (sig - 1 - nd):
        nd += 1
    return nd


SIGNIFICATIFS = {"kN·m/m": 3, "kN·m": 3}


DECIMALES = {"N/mm²": 2, "kN/m": 2, "kN": 2, "kN·m/m": 3, "kN·m": 3, "mm⁴": 0, "mm³": 0, "mm²": 0, "mm": 0, "": 3}


def _r(x: float, unite: str = "") -> str:
    """Un résultat à la française, à la précision de l'écran Streamlit (contraintes au centième, moments au millième…)."""
    nd = DECIMALES.get(unite, 2)
    return C.fr(x, _nd(x, nd, SIGNIFICATIFS.get(unite, 2)) if nd else 0)


def _d(x: float) -> str:
    """Une DONNÉE (saisie, ou exacte comme b = 1 000 mm) écrite sans la dénaturer : 1,35 reste 1,35, 0,0315 reste
    0,0315 ; les milliers séparés comme dans les résultats."""
    s = C.nb(x, 5)
    return C.fr(x, len(s.partition(",")[2]))


def _ru(x: float, unite: str) -> str:
    return f"{_r(x, unite)} {unite}".strip()


def _taux(t: float) -> str:
    return f"{C.pc(C.pct(t, 1))}{C.NBSP}%"


def _niv(v: Dict) -> str:
    """La couleur d'une vérification : rouge dès que le moteur la déclare non conforme, sinon le taux."""
    return "r" if v["etat"] == MO.NOK else C.niveau(v["taux"])


def _niv_res(res: Dict) -> str:
    return "r" if res["statut"] == MO.NOK else C.niveau(res["taux_max"])


# ------------------------------------------------------------------ le détail des formules
def _fx(f: Dict) -> Dict:
    """Une formule du moteur → « symbole = expression = valeurs substituées = résultat » (rien n'est recalculé : le
    résultat est celui du moteur, les termes sont ses termes)."""
    t = {x["symbole"]: x["valeur"] for x in f["termes"]}
    u = {x["symbole"]: x["unite"] for x in f["termes"]}
    k, s = f["cle"], f["symbole"]
    noms = list(t)

    def v(nom):                     # une grandeur CALCULÉE, à la précision de son unité
        return _r(t[nom], u[nom])

    def d(nom):                     # une DONNÉE saisie, écrite telle quelle (5, 1,35, 0,6, 155)
        return _d(t[nom])
    if k == "V_ELS_lin":
        parts = [s, "V"]
    elif k == "V_ELU_lin":
        parts = [s, "γ_ELU·V", f"{d('γ_ELU')} × {d('V')}"]
    elif k == "V_ELS_ponct":
        parts = [s, "V / L", f"{d('V')} / {d('L')}"]
    elif k == "V_ELU_ponct":
        parts = [s, "γ_ELU·V / L", f"{d('γ_ELU')} × {d('V')} / {d('L')}"]
    elif k in ("M", "Mlong"):
        a = noms[0]
        parts = [s, f"{a}·e", f"{v(a)} × {_d(t['e'] / 100.0)}"]
    elif k in ("I", "W", "Az"):
        expr = {"I": ("b·t³ / 12", "{b} × {t}³ / 12"), "W": ("b·t² / 6", "{b} × {t}² / 6"), "Az": ("b·t", "{b} × {t}")}[k]
        parts = [s, expr[0], expr[1].format(b=_d(t["b"]), t=_d(t["t"]))]
    elif k in ("sigma", "sigma_long"):
        m, w = noms
        parts = [s, f"{m} / {w}", f"{v(m)} × 10⁶ / {v(w)}"]
    elif k in ("tau", "tau_long"):
        vv, a = noms
        parts = [s, f"{vv} / {a}", f"{v(vv)} × 10³ / {v(a)}"]
    elif k == "sigma_eq":
        parts = [s, "√(σ² + 3·τ²)", f"√({v('σ')}² + 3 × {v('τ')}²)"]
    elif k == "b_long":
        parts = [s, "10·s", f"10 × {d('s')}"]
    elif k == "Vloc":
        a = noms[0]
        parts = [s, f"{a}·s", f"{v(a)} × {_d(t['s'] / 100.0)}"]
    elif k == "taux":
        a, b = noms
        parts = [s, f"{a} / {b}", f"{v(a)} / {d(b)}"]
        return {"parts": parts, "res": _taux(f["resultat"]), "niveau": C.niveau(f["resultat"])}
    else:                                                        # pragma: no cover — une formule nouvelle du moteur
        parts = [s, " · ".join(f"{n} = {v(n)}" for n in noms)]
    return {"parts": parts, "res": _ru(f["resultat"], f["unite"]), "niveau": ""}


def _detail(blocs: List[Dict]) -> str:
    return "".join(f'<div class="ck-g"><div class="ck-t">{C.esc(b["titre"])}</div>{C.lignes_fx([_fx(f) for f in b["formules"]])}</div>'
                   for b in blocs)


# ------------------------------------------------------------------ le résultat
NOMS = {"els_sigma": "Contrainte équivalente", "els_tau": "Cisaillement", "elu": "Contrainte équivalente",
        "long_elu": "Contrainte équivalente"}
LIMITES_COURTES = {"els_sigma": "σ_lim", "els_tau": "τ_lim", "elu": "σ_lim", "long_elu": "σ_lim"}
GROUPES = {MO.TRANSVERSAL: ("Transversal · par mètre courant", "transversal"),
           MO.LONGITUDINAL: ("Longitudinal · entre fixations", "longitudinal")}
FORMULES = {"els_sigma": "σ<sub>eq</sub> = √(σ² + 3·τ²) ≤ σ<sub>lim,ELS</sub>", "els_tau": "τ = V / A<sub>z</sub> ≤ τ<sub>lim,ELS</sub>",
            "elu": "σ<sub>eq</sub> = √(σ² + 3·τ²) ≤ σ<sub>eq,lim,ELU</sub>",
            "long_elu": "σ<sub>eq,long</sub> = √(σ<sub>long</sub>² + 3·τ<sub>long</sub>²) ≤ σ<sub>eq,lim,ELU</sub>"}


def _ligne_tableau(v: Dict, gov: bool) -> Dict:
    return {"nom": C.esc(NOMS[v["cle"]]), "sous": f"{v['etat_limite']} · {'par mètre courant' if v['groupe'] == MO.TRANSVERSAL else 'panneau de largeur s'}",
            "formule": FORMULES[v["cle"]], "valeur": f"{C.symb_math(v['symbole'])} = {_ru(v['valeur'], 'N/mm²')}",
            "limite": f"{C.symb_math(v['symbole_limite'])} = {C.nb(v['limite'])} N/mm²", "taux": C.pct(v["taux"]), "niveau": _niv(v), "gov": gov}


def _piece(sec: Dict) -> str:
    return C.esc(sec["designation"]) if sec["source"] == "catalogue" else f"Saisie {C.nb(sec['A'])}×{C.nb(sec['B'])}×{C.nb(sec['t'])}"


def _charge(c: Dict) -> str:
    if c["charge"] == MO.LINEAIRE:
        return f"Charge linéaire V = {C.nb(c['V'])} kN/m"
    return f"Charge ponctuelle V = {C.nb(c['V'])} kN répartie sur L = {C.nb(c['L'])} m"


def _hypotheses(res: Dict) -> List[str]:
    c, sec = res["corniere"], res["section"]
    masse = f" · {C.fr(sec['masse'], 2)} kg/m" if sec.get("masse") else " · saisie libre"
    return [f"Cornière <b>{_piece(sec)}</b> · t = {C.nb(sec['t'])} mm{masse}",
            "Modèle lame · b = 1 000 mm (par mètre courant)",
            f"{_charge(c)} · e = {C.nb(c['e'])} cm · γ<sub>ELU</sub> = {C.nb(c['gamma_ELU'])}",
            f"σ<sub>lim,ELS</sub> = {C.nb(c['sig_lim_ELS'])} · τ<sub>lim,ELS</sub> = {C.nb(c['tau_lim_ELS'])} · "
            f"σ<sub>eq,lim,ELU</sub> = {C.nb(c['sig_lim_ELU'])} N/mm²"]


# le début de chaque avertissement du moteur, écrit en gras (le reste est échappé tel quel)
TITRES_AVERTISSEMENTS = {"excentricite_hors_aile": "La charge tombe au-delà de l'aile B",
                         "longitudinal_provisoire": "Côté non sûr", "ancrages_non_verifies": "Ancrages non vérifiés"}


def _avertissement(a: Dict) -> str:
    """Un avertissement du moteur, son titre en gras (« Côté non sûr : … »), tout le texte échappé."""
    msg, titre = C.esc(a["message"]), C.esc(TITRES_AVERTISSEMENTS.get(a["code"], ""))
    return f"<b>{titre}</b>{msg[len(titre):]}" if titre and msg.startswith(titre) else msg


def _calculer(c: Dict) -> Dict:
    try:
        return MO.verifier(c)
    except ValueError as e:                                     # pragma: no cover — normaliser l'a déjà refusé
        return {"erreur": str(e)}


def resume_note(etat: Dict, resultats: List[Dict]) -> Dict:
    """Le résumé de la NOTE (l'accueil, la liste des notes) : sa cornière la plus chargée."""
    vals = [(r["taux_max"], i) for i, r in enumerate(resultats) if "taux_max" in r]
    if not vals:
        return {"taux": None, "niveau": "x", "statut": "Calcul impossible", "gouvernant": ""}
    t, i = max(vals)
    r = resultats[i]
    niv = _niv_res(r)
    g = r["gouvernant"]
    return {"taux": C.pct(t), "niveau": niv, "statut": C.STATUTS[niv][1],
            "gouvernant": f"{etat['cornieres'][i].get('nom') or 'Cornière'} · {g['groupe']} {g['etat_limite']} · {g['nom'].lower()}"}


def elements(etat: Dict, resultats: List[Dict]) -> str:
    """« Dans cette note » : chaque cornière avec son taux ; un clic l'ouvre."""
    items = ""
    for i, (c, r) in enumerate(zip(etat["cornieres"], resultats)):
        t = r.get("taux_max")
        niv = _niv_res(r) if t is not None else "x"
        txt = f"{C.pc(C.pct(t))}{C.NBSP}%" if t is not None else "—"
        sup = f'<button class="x" data-supprimer="{i}" title="Supprimer">{C.icon("x")}</button>' if len(etat["cornieres"]) > 1 else ""
        piece = _piece(r["section"]) if "section" in r else ""
        items += (f'<div class="ni{" on" if i == etat["actif"] else ""}" data-activer="{i}"><span class="t">{C.esc(c.get("nom") or f"Cornière {i + 1}")} '
                  f'<span>· {piece}</span></span><span class="pct {niv}">{txt}</span>{sup}</div>')
    return (f'<div class="elements"><div class="nl-h"><span class="kicker">Dans cette note · {len(etat["cornieres"])}</span>'
            f'<button class="nl-add" data-dupliquer="1" title="Copier la cornière ouverte">{C.icon("copy")}Copier</button></div>{items}'
            f'<div class="nl-add" data-ajouter="1">{C.icon("plus")}Ajouter une cornière</div></div>')


def resultat(etat: Dict) -> Dict:
    c = _corniere(etat)
    r = MO.verifier(c)
    resultats = [r if i == etat["actif"] else _calculer(x) for i, x in enumerate(etat["cornieres"])]
    sec, g = r["section"], r["gouvernant"]
    niv = _niv_res(r)
    ver = {v["cle"]: v for v in r["verifications"]}
    vg = ver[g["cle"]]
    verdict = (f'<div class="verdict">{C.chip(C.pct(r["taux_max"]), niv, xl=True)}<div class="gov">'
               f'<b>{GROUPES[vg["groupe"]][0].split(" · ")[0]} · {C.esc(NOMS[vg["cle"]].lower())} à l\'{vg["etat_limite"]}</b>'
               f'{C.symb_math(vg["symbole"])} = {_r(vg["valeur"], "N/mm²")} {"≤" if vg["ok"] else "&gt;"} '
               f'{C.symb_math(LIMITES_COURTES[vg["cle"]])} = {C.nb(vg["limite"])} N/mm²</div></div>')
    # la ligne de la cornière : son sélecteur, et l'étoile quand une plus légère tient
    opt = MO.optimal(c)
    leger = (f"Plus légère qui tient : {MO.designation(opt['nom'])} · {C.pc(C.pct(opt['taux']))} %"
             if opt and not opt["retenue"] else "")
    rows = C.ligne_res("Cornière", _piece(sec), C.pct(r["taux_max"]), niv, calcul="transversal", pk="corniere", leger=leger)
    for groupe, (titre, cle_calc) in GROUPES.items():
        vs = [v for v in r["verifications"] if v["groupe"] == groupe]
        t_g = max(v["taux"] for v in vs)
        n_g = "r" if any(v["etat"] == MO.NOK for v in vs) else C.niveau(t_g)
        rows += f'<div class="elgrp">{titre}<span class="pct {n_g}">{C.pc(C.pct(t_g))}{C.NBSP}%</span></div>'
        for v in vs:
            rows += C.ligne_res(NOMS[v["cle"]], f"{v['etat_limite']} · {_ru(v['valeur'], 'N/mm²')}", C.pct(v["taux"]), _niv(v),
                                calcul=cle_calc, gov=v["cle"] == g["cle"])
    graves = {"excentricite_hors_aile"}
    # l'ordre : le grave (charge hors de l'aile), puis le défaut non conservatif de l'écran, puis les ancrages
    messages = "".join(C.message(_avertissement(a), a["code"] in graves)
                       for a in sorted(r["avertissements"], key=lambda a: (a["code"] not in graves, a["code"] != "longitudinal_provisoire")))
    masse = f"{C.fr(sec['masse'], 2)}{C.NBSP}kg/m" if sec.get("masse") else "saisie libre"
    pied = (f'<div class="pfoot"><div class="mass"><span>Masse de la cornière</span><b>{masse}</b></div>'
            f'<div class="legend">{C.legende()}</div></div>')
    # les fenêtres de calcul : transversal (efforts, section, ELS, ELU) et longitudinal (panneau, efforts, contraintes)
    hyps = _hypotheses(r)
    volets = ""
    for groupe, (titre, cle_calc) in GROUPES.items():
        vs = [v for v in r["verifications"] if v["groupe"] == groupe]
        t_g = max(v["taux"] for v in vs)
        lignes = [_ligne_tableau(v, v["taux"] == t_g) for v in vs]
        if groupe == MO.TRANSVERSAL:
            sous, h = "modèle lame b = 1 000 mm · méthode de l'écran Streamlit", hyps
            pied_ck = ""
        else:
            sous, h = f"panneau de largeur s = {C.nb(c['s'])} cm · méthode de l'écran Streamlit", hyps + [
                f"Fixations tous les <b>s = {C.nb(c['s'])} cm</b>"]
            pied_ck = ('<div class="ck-autres"><b>Côté non sûr</b> — hypothèse provisoire de l\'écran Streamlit : '
                       'M<sub>long</sub> = V<sub>loc</sub>·e sur un panneau de largeur s — s se simplifie, σ<sub>long</sub> = σ du '
                       'calcul transversal, quel que soit l\'entraxe. La flexion de la cornière entre fixations (q·s²/8, le '
                       '« TODO » de l\'écran) n\'est vérifiée nulle part : à justifier à part.</div>')
        volets += C.fenetre_calcul(cle_calc, f"{C.esc(c.get('nom') or 'Cornière')} · {titre.lower()}", sous, h, lignes,
                                   _detail(r["calcul"][groupe]), pied_ck)
    charge = _charge(c)
    legendes = {"coupe": f"e compté depuis le dos de l'aile fixée · {charge[0].lower()}{charge[1:]}",
                "elevation": f"entraxe s = {C.nb(c['s'])} cm · en gris, le panneau de largeur s",
                "efforts": "par mètre courant · valeurs de service (ELS) et de calcul (ELU)",
                "section": (f"{sec['designation']} · {sec['norme']} · {C.fr(sec['masse'], 2)} kg/m" if sec.get("masse")
                            else f"saisie libre {C.nb(sec['A'])} × {C.nb(sec['B'])} × {C.nb(sec['t'])} mm · angles vifs")}
    vues = [dict(v, legende=legendes[v["cle"]]) for v in DC.vues(r, f"cor{etat['actif']}")]
    phrase = (f"{_piece(sec)} · V = {C.nb(c['V'])} {'kN/m' if c['charge'] == MO.LINEAIRE else 'kN'} · e = {C.nb(c['e'])} cm · "
              f"s = {C.nb(c['s'])} cm")
    return {"resultat": f'<div class="pbody">{verdict}<div class="els">{rows}</div>{messages}</div>{pied}', "volets": volets,
            "vues": vues, "phrase": phrase, "statut_html": C.statut(niv), "resume": resume_note(etat, resultats),
            "donnees": formulaire(etat, r), "elements": elements(etat, resultats)}


# ------------------------------------------------------------------ le sélecteur
def selecteur(etat: Dict, role: str, familles: Optional[Sequence[str]] = None, froid: bool = True) -> str:
    """Les cornières du catalogue TRIÉES PAR MASSE, chacune calculée avec les charges et les critères de la cornière
    ouverte ; la retenue (✓) et la plus légère qui tient (★) marquées ; les familles cochées filtrent."""
    c = _corniere(etat)
    fams = tuple(f for f in (familles or EN.FAMILLES) if f in EN.FAMILLES) or EN.FAMILLES
    cands = MO.candidats(c, fams)
    noms = [x["nom"] for x in cands]
    i_o = next((k for k, x in enumerate(cands) if x["optimal"]), 0)
    i_c = next((k for k, x in enumerate(cands) if x["retenue"]), i_o)

    e_mm = c["e"] * 10.0

    def prop(x):
        hors = ("" if x["aile"] else f" · <b>la charge tombe au-delà de l'aile B</b> (e = {C.nb(e_mm)} mm &gt; B = {C.nb(x['B'])} mm)")
        return (f"{LIB_FAMILLES[x['famille']]} · EN 10056-1 · A × B × t = {C.nb(x['A'])} × {C.nb(x['B'])} × {C.nb(x['t'])} mm · "
                f"r<sub>1</sub> = {C.nb(x['r1'])} mm · aire {C.fr(x['aire'], 2)} cm²{hors}")

    def niveau(x):
        """Rouge : ne tient pas ; gris : tient, mais la charge tombe au-delà de l'aile B (jamais proposée) ; sinon le taux."""
        return "r" if not x["ok"] else ("c" if not x["aile"] else C.niveau(x["taux"]))
    # la ligne porte la désignation du bureau (« L 45×45×4,5 ») ; choisir_section la relit (nom_catalogue)
    rows = [{"nom": x["designation"], "col": C.fr(x["masse"], 2), "taux": C.pct(min(x["taux"], 9.99)),
             "niveau": niveau(x), "cur": x["retenue"], "opt": x["optimal"], "prop": prop(x)}
            for x in cands]
    courtes = sum(not x["aile"] for x in cands)
    sous = f"{len(noms)} cornières · " + ("par masse · gris : aile B &lt; e" if courtes else "triées par masse")
    filtres = "".join(f'<button type="button" class="fchip{" on" if f in fams else ""}" data-famille="{f}">{LIB_FAMILLES[f]}</button>'
                      for f in EN.FAMILLES)
    return C.selecteur(role, "Cornière · profilé", sous, filtres,
                       rows, max(0, min(i_o, i_c) - 4), "kg/m", etoile="la plus légère qui tient")

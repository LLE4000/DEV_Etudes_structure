# -*- coding: utf-8 -*-
"""Le module « Poutres bois » du site : l'état d'une note (une ou plusieurs poutres en bois massif), les champs de
saisie, le résultat vérification par vérification, les fenêtres de calcul (ELU, ELS), le sélecteur de sections
standard et les dessins — rendus en fragments HTML.

Le calcul est celui du moteur ``bois/poutre`` (extrait de l'écran Streamlit modules/poutre_bois.py, dont il reproduit
les résultats au bit près — tests/test_poutre_bois_moteur.py), écarts avec l'EC5 compris : ils sont rappelés à
l'écran, pas corrigés. Ce fichier MET EN FORME : il ne calcule rien.
"""
import copy
from typing import Dict, List, Optional, Sequence

from bois.poutre import donnees as D
from bois.poutre import entree as EN
from bois.poutre import moteur as MO

from . import commun as C
from . import dessin_bois as DB

MODULE = "bois"
PAGE = "poutre-bois"
# des requêtes de plus pour le parcours : une section sur mesure, l'appui coché, une saisie refusée
PARCOURS = [{"action": "modifier", "chemin": ["standard"], "valeur": "non"},
            {"action": "modifier", "chemin": ["appui"], "valeur": True},
            {"action": "modifier", "chemin": ["L"], "valeur": "0,2"},
            {"action": "choisir", "role": "section", "nom": "75x225"}]

DUREES = {"permanent": "Permanente", "long": "Long terme", "moyen": "Moyen terme", "court": "Court terme", "instant": "Instantanée"}
PERIODES = {"permanent": "plus de 10 ans", "long": "6 mois à 10 ans", "moyen": "1 semaine à 6 mois", "court": "moins d'une semaine",
            "instant": "instantanée"}
DUREES_COURTES = {"permanent": "permanente", "long": "long terme", "moyen": "moyen terme", "court": "court terme", "instant": "instantanée"}
SERVICES = {1: "intérieur chauffé (humidité ≤ 65 %)", 2: "abrité, non chauffé (humidité ≤ 85 %)", 3: "extérieur, exposé"}
COURTS = {"flexion": "Flexion", "cisaillement": "Cisaillement", "appui": "Compression à l'appui", "fleche_inst": "Flèche instantanée",
          "fleche_fin": "Flèche finale"}
ELU, ELS = ("flexion", "cisaillement", "appui"), ("fleche_inst", "fleche_fin")
PLAFOND = 9.99      # un taux au-delà de 999 % s'affiche 999 % (pastilles, jauges) ; la valeur exacte est dans le détail


# ------------------------------------------------------------------ l'état de la note : des poutres
def nouvel_etat() -> Dict:
    p = MO.nouvelle_poutre()
    p["nom"] = "P1"
    return {"version": 1, "poutres": [p], "actif": 0}


def normaliser(etat: Dict) -> Dict:
    """L'état reçu de l'écran (ou d'un fichier .stab) relu par le moteur ; une poutre illisible lève une erreur claire."""
    poutres = etat.get("poutres") if isinstance(etat, dict) else None
    if not poutres or not isinstance(poutres, list):
        return nouvel_etat()
    out = {"version": 1, "poutres": [MO.normaliser(p) for p in poutres]}
    out["actif"] = min(max(0, _entier(etat.get("actif"))), len(out["poutres"]) - 1)
    return out


def _entier(x) -> int:
    """Un rang lu dans une note ou une requête (« 1 », 1.0, null, « abc », 1e309…) : un entier, 0 s'il est illisible."""
    try:
        return int(x or 0)
    except (TypeError, ValueError, OverflowError):
        return 0


def _poutre(etat: Dict) -> Dict:
    return etat["poutres"][etat["actif"]]


def gerer_elements(etat: Dict, action: str, index: Optional[int] = None) -> None:
    """Les poutres de la note : en ouvrir une, en ajouter, copier l'ouverte, en supprimer une (jamais la dernière)."""
    ps = etat["poutres"]
    if action == "activer":
        etat["actif"] = min(max(0, _entier(index)), len(ps) - 1)
    elif action == "ajouter":
        p = MO.nouvelle_poutre()
        p["nom"] = f"P{len(ps) + 1}"
        ps.append(p)
        etat["actif"] = len(ps) - 1
    elif action == "dupliquer":
        p = copy.deepcopy(ps[etat["actif"]])
        p["nom"] = f"{p.get('nom') or 'Poutre'} (copie)"
        ps.append(p)
        etat["actif"] = len(ps) - 1
    elif action == "supprimer" and len(ps) > 1:
        i = min(max(0, _entier(index)), len(ps) - 1)
        del ps[i]
        if etat["actif"] > i or etat["actif"] >= len(ps):
            etat["actif"] = max(0, etat["actif"] - 1)


# ------------------------------------------------------------------ la saisie
NOMBRES = tuple(D.CHAMPS)
OUI = {"oui": True, "1": True, "true": True, "non": False, "0": False, "false": False}


def modifier(etat: Dict, chemin: Sequence, valeur) -> Optional[str]:
    """Écrit une saisie dans la poutre ouverte ; le moteur relit la poutre : une valeur hors du domaine de l'écran
    Streamlit (les bornes de ses champs) est REFUSÉE avec un message, et la poutre reste intacte."""
    p = _poutre(etat)
    avant = copy.deepcopy(p)
    if len(chemin) != 1 or chemin[0] not in EN.CLES:
        return f"Champ inconnu : « {'/'.join(str(c) for c in chemin)} »"
    k = chemin[0]
    try:
        if k in NOMBRES:
            x = C.lire_nombre(valeur)
            if x is None:
                return f"Nombre illisible : « {valeur} »"
            p[k] = x
        elif k == "service":
            x = C.lire_nombre(valeur)
            if x is None or x != int(x):
                return f"Classe de service illisible : « {valeur} »"
            p[k] = int(x)
        elif k in ("standard", "appui"):
            p[k] = valeur if isinstance(valeur, bool) else OUI.get(str(valeur).strip().lower(), valeur)
        else:
            p[k] = valeur if isinstance(valeur, str) else str(valeur)
        etat["poutres"][etat["actif"]] = MO.normaliser(p)
    except ValueError as e:
        etat["poutres"][etat["actif"]] = avant
        texte = str(e)
        return f"Saisie refusée : {texte[:1].lower() + texte[1:]}"
    return None


def choisir_section(etat: Dict, role: str, nom: str) -> None:
    """Le sélecteur : la section standard ``nom`` (« 45x220 ») est retenue (et la section repasse en « standard »)."""
    tag = str(nom).replace("×", "x").replace(" ", "")
    p = dict(_poutre(etat), standard=True, section=tag)
    etat["poutres"][etat["actif"]] = MO.normaliser(p)


def _tag(t: str) -> str:
    """« 45x220 » → « 45 × 220 »."""
    return t.replace("x", " × ")


def formulaire(etat: Dict, r: Dict) -> str:
    """Le cadre Données : cinq sections repliables, dans l'ordre de l'écran Streamlit."""
    p = _poutre(etat)
    L = C.ligne
    mat, co, geo, ch, g = r["materiau"], r["coefficients"], r["geometrie"], r["charges"], r["grandeurs"]
    # 1 · la poutre, le bois, le service
    s1 = (L("Nom", C.texte(["nom"], p.get("nom"), "ex. P1 solive"), full=True)
          + L("Classe", C.choix(["classe"], p["classe"], [(c, f"{c} · résineux") for c in D.CLASSES]), full=True)
          + C.aide(f"f<sub>m,k</sub> {C.nb(mat['fm_k'])} · f<sub>v,k</sub> {C.nb(mat['fv_k'])} · f<sub>c,90,k</sub> {C.nb(mat['fc90_k'])} · "
                   f"E<sub>0,mean</sub> {C.fr(mat['E0_mean'], 0)} · G<sub>mean</sub> {C.nb(mat['G_mean'])} MPa · "
                   f"ρ<sub>mean</sub> {C.nb(mat['rho_mean'])} kg/m³ (base de l'écran Streamlit)")
          + L("Service", C.segments(["service"], str(p["service"]), [(str(s), str(s)) for s in D.CLASSES_SERVICE]), full=True)
          + L("Durée", C.choix(["duree"], p["duree"], [(d, DUREES[d]) for d in D.DUREES]), full=True,
              aide="La durée de la charge la plus courte de la combinaison fixe k_mod (EC5 §3.1.3).")
          + C.aide(f"Classe de service {co['service']} : {SERVICES[co['service']]} · durée : {PERIODES[co['duree']]} · k<sub>mod</sub> = {C.nb(co['kmod'])} · "
                   f"k<sub>def</sub> = {C.nb(co['kdef'])} · γ<sub>M</sub> = {C.nb(co['gamma_M'])}"))
    # 2 · la section et la portée
    std = p["standard"]
    s2 = L("Section", C.segments(["standard"], "oui" if std else "non", [("oui", "Standard"), ("non", "Sur mesure")]), full=True)
    if std:
        s2 += (L("b × h", C.choix(["section"], p["section"], [(s["tag"], f"{_tag(s['tag'])} mm") for s in D.SECTIONS_STD]), full=True)
               + C.aide("Ou choisir dans le résultat (ligne « Flexion ») : les sections triées par masse, la retenue ✓ et la plus légère qui "
                        "tient ★."))
    else:
        s2 += (L("largeur", C.num(["b"], p["b"], "mm"), "b")
               + L("hauteur", C.num(["h"], p["h"], "mm"), "h"))
    s2 += (L("portée", C.num(["L"], p["L"], "m"), "L", "Entre axes des appuis ; poutre sur deux appuis.")
           + C.aide(f"A = {C.fr(geo['A'] / 100, 1)} cm² · I = {C.fr(geo['I'] / 1e4, 1)} cm⁴ · W = {C.fr(geo['W'] / 1e3, 1)} cm³ · "
                    f"k<sub>h</sub> = {C.fr(co['kh'], 3)}"))
    # 3 · les charges
    s3 = (L("permanente", C.num(["G"], p["G"], "kN/m"), "G", "Hors poids propre : l'écran ne le compte pas.")
          + L("d'exploitation", C.num(["Q"], p["Q"], "kN/m"), "Q")
          + L("coefficient sur G", C.num(["gamma_G"], p["gamma_G"], ""), "γ<sub>G</sub>")
          + L("coefficient sur Q", C.num(["gamma_Q"], p["gamma_Q"], ""), "γ<sub>Q</sub>")
          + L("quasi-permanent", C.num(["psi2"], p["psi2"], ""), "ψ<sub>2</sub>",
              "Part de Q présente en permanence : elle flue avec G (flèche finale).")
          + C.aide(f"q<sub>ELU</sub> = {C.nb(ch['gamma_G'])} × {C.nb(ch['G'])} + {C.nb(ch['gamma_Q'])} × {C.nb(ch['Q'])} = "
                   f"<b>{C.fr(ch['q_ELU'], 2)} kN/m</b> · G + Q = {C.fr(ch['q_inst'], 2)} · G + ψ<sub>2</sub>Q = {C.fr(ch['q_qp'], 2)} kN/m"))
    # 4 · les flèches
    s4 = (L("Instantanée", C.segments(["lim_inst"], p["lim_inst"], [(x, x) for x in sorted(D.LIMITES_INST, key=MO.diviseur_limite)]), full=True)
          + L("Finale", C.segments(["lim_fin"], p["lim_fin"], [(x, x) for x in sorted(D.LIMITES_FIN, key=MO.diviseur_limite)]), full=True)
          + C.aide("w<sub>inst</sub> sous G + Q ; w<sub>fin</sub> = (1 + k<sub>def</sub>)·w(G + ψ<sub>2</sub>Q), comme l'écran Streamlit."))
    # 5 · l'appui
    s5 = (C.case(["appui"], p["appui"], "Vérifier la compression perpendiculaire au fil à l'appui")
          + L("largeur d'appui", C.num(["a"], p["a"], "mm", desactive=not p["appui"]), "a",
              "Longueur de poutre posée sur l'appui ; grisée tant que la vérification n'est pas demandée (comme l'écran).")
          + C.aide("Réaction sous G + Q <b>non pondérées</b> et sans k<sub>c,90</sub>, comme l'écran Streamlit : non conservatif à l'ELU "
                   "(voir le détail du calcul)."))
    sec = _tag(geo["tag"]) if geo["tag"] else f"{C.nb(geo['b'])} × {C.nb(geo['h'])}"
    return (C.section(1, f"Poutre {C.esc(p.get('nom') or '')}", f"{C.esc(p['classe'])} · classe {p['service']} · {DUREES_COURTES[p['duree']]}", s1, True, "pb")
            + C.section(2, "Section et portée", f"{sec} mm · L {C.nb(p['L'])} m", s2, True, "geo")
            + C.section(3, "Charges", f"G {C.nb(p['G'])} · Q {C.nb(p['Q'])} kN/m", s3, True, "ch")
            + C.section(4, "Flèches admissibles", f"{p['lim_inst']} · {p['lim_fin']}", s4, False, "els")
            + C.section(5, "Appuis", f"a = {C.nb(p['a'])} mm" if p["appui"] else "non vérifiés", s5, False, "ap"))


# ------------------------------------------------------------------ le résultat
def _aff(t: Optional[float]):
    """Le taux affiché dans une pastille ou une jauge : arrondi dans sa bande, plafonné à 999 %."""
    return None if t is None else C.pct(min(t, PLAFOND))


def _niv(v: Dict) -> str:
    return "r" if v["etat"] == MO.NOK else C.niveau(v["taux"])


def _n(x: Optional[float], unite: str = "") -> str:
    """Un RÉSULTAT écrit avec la précision d'usage (celle de l'écran Streamlit, au moins) : contraintes et efforts au
    centième, flèches au dixième, aires et inerties entières."""
    if x is None:
        return "—"
    nd = {"MPa": 2, "kN": 2, "kN·m": 2, "kN/m": 2, "mm": 1, "mm²": 0, "mm³": 0, "mm⁴": 0, "": 3}.get(unite, 2)
    return C.fr(x, _decimales(x, nd))


def _decimales(x: float, nd: int) -> int:
    """Les décimales d'un petit résultat non nul : au moins deux chiffres significatifs, quatre décimales au plus —
    M_Ed = 0,0024 kN·m ne s'écrit pas « 0,00 » sous un diagramme de moment tracé en entier."""
    while nd and nd < 4 and x != 0 and abs(x) < 10 ** (1 - nd):
        nd += 1
    return nd


def _v(x) -> str:
    """Une DONNÉE d'une formule écrite sans la dénaturer : 1,35 reste 1,35, 11 000 reste 11 000 ; seules les grandeurs
    calculées longues (12,923…) sont arrondies."""
    if isinstance(x, int):
        return C.fr(x, 0) if abs(x) >= 1000 else str(x)
    a = abs(x)
    if a >= 1000:
        return C.fr(x, 0) if x == int(x) else C.fr(x, 1)
    if a >= 100:
        return C.nb(round(x, 1), 1)
    if a >= 10:
        return C.nb(round(x, 2), 2)
    return C.nb(round(x, 4), 4)


def _pc(u: Optional[float]) -> str:
    """Un taux EXACT (le détail du calcul) : au dixième sous 1 000 %, entier au-delà."""
    return "—" if u is None else f"{C.fr(u * 100, 1 if u < 10 else 0)}{C.NBSP}%"


RAPPORTS = {"flexion": "σ_m,Ed / f_m,d", "cisaillement": "τ_Ed / f_v,d", "appui": "σ_c,90,Ed / f_c,90,d", "fleche_inst": "w_inst / w_lim",
            "fleche_fin": "w_fin / w_lim"}


def _formule(f: Dict, v: Dict) -> Dict:
    """Une formule du moteur → « symbole = expression = valeurs substituées = résultat »."""
    t = {x["symbole"]: x["valeur"] for x in f["termes"]}
    u, k = f["unite"], f["cle"]
    res = _n(f["resultat"], u) + (f" {u}" if u else "")
    niveau = ""
    if k == "q_ELU":
        parts = ["q_ELU", "γ_G·G + γ_Q·Q", f"{_v(t['gamma_G'])} × {_v(t['G'])} + {_v(t['gamma_Q'])} × {_v(t['Q'])}"]
    elif k == "M_Ed":
        parts = ["M_Ed", "q_ELU·L² / 8", f"{_v(t['q_ELU'])} × {_v(t['L'])}² / 8"]
    elif k == "V_Ed":
        parts = ["V_Ed", "q_ELU·L / 2", f"{_v(t['q_ELU'])} × {_v(t['L'])} / 2"]
    elif k == "I":
        parts = ["I", "b·h³ / 12", f"{_v(t['b'])} × {_v(t['h'])}³ / 12"]
    elif k == "W":
        parts = ["W", "I / (h/2)", f"{_n(t['I'], 'mm⁴')} / ({_v(t['h'])} / 2)"]
    elif k == "sigma_m_Ed":
        parts = ["σ_m,Ed", "M_Ed·10⁶ / W", f"{_v(t['M_Ed'])} × 10⁶ / {_n(t['W'], 'mm³')}"]
    elif k == "kmod":
        parts = ["k_mod", "tableau (classe de service, durée)", f"classe {t['service']} · {DUREES_COURTES[t['duree']]}"]
        res = C.nb(f["resultat"], 2)
    elif k == "kh":
        parts = ["k_h", "min(max((150/h)^0,2 ; 0,6) ; 1,3)", f"min(max((150 / {_v(t['h'])})^0,2 ; 0,6) ; 1,3)"]
    elif k == "fm_d":
        parts = ["f_m,d", "k_mod·f_m,k·k_h / γ_M", f"{_v(t['kmod'])} × {_v(t['fm_k'])} × {_v(t['kh'])} / {_v(t['gamma_M'])}"]
    elif k == "tau_Ed":
        parts = ["τ_Ed", "1,5·V_Ed·10³ / (b·h)", f"1,5 × {_v(t['V_Ed'])} × 10³ / ({_v(t['b'])} × {_v(t['h'])})"]
    elif k == "fv_d":
        parts = ["f_v,d", "k_mod·f_v,k / γ_M", f"{_v(t['kmod'])} × {_v(t['fv_k'])} / {_v(t['gamma_M'])}"]
    elif k == "R_appui":
        parts = ["R", "(G + Q)·L / 2", f"{_v(t['q_inst'])} × {_v(t['L'])} / 2"]
    elif k == "sigma_c90_Ed":
        parts = ["σ_c,90,Ed", "R·10³ / (b·a)", f"{_v(t['R_appui'])} × 10³ / ({_v(t['b'])} × {_v(t['a'])})"]
    elif k == "fc90_d":
        parts = ["f_c,90,d", "k_mod·f_c,90,k / γ_M", f"{_v(t['kmod'])} × {_v(t['fc90_k'])} / {_v(t['gamma_M'])}"]
    elif k == "q_inst":
        parts = ["q_inst", "G + Q", f"{_v(t['G'])} + {_v(t['Q'])}"]
    elif k == "q_qp":
        parts = ["q_qp", "G + ψ₂·Q", f"{_v(t['G'])} + {_v(t['psi2'])} × {_v(t['Q'])}"]
    elif k in ("w_inst", "w_qp"):
        sym = "w_inst" if k == "w_inst" else "w(G+ψ₂Q)"
        parts = [sym, "5·(q·10³)·(L·10³)⁴ / (384·E·I)",
                 f"5 × ({_v(t['q'])} × 10³) × ({_v(t['L'])} × 10³)⁴ / (384 × {_v(t['E'])} × {_n(t['I'], 'mm⁴')})"]
    elif k == "kdef":
        parts = ["k_def", "tableau (classe de service)", f"classe {t['service']}"]
        res = C.nb(f["resultat"], 2)
    elif k == "w_fin":
        parts = ["w_fin", "(1 + k_def)·w(G+ψ₂Q)", f"(1 + {_v(t['kdef'])}) × {_v(t['w_qp'])}"]
    elif k in ("lim_inst", "lim_fin"):
        parts = ["w_lim", f"L / {_v(t['n'])}", f"{_v(t['L'])} / {_v(t['n'])}"]
        sym = v.get("limite_texte") or ""
        parts[1] = sym or parts[1]
    elif k == "u":
        uu = f["termes"][0]["unite"]
        parts = ["η", RAPPORTS[v["cle"]], f"{_n(t['valeur'], uu)} / {_n(t['limite'], uu)}"]
        res, niveau = _pc(f["resultat"]), _niv(v)
    else:                                                        # pragma: no cover — une formule nouvelle du moteur
        parts = [k, " · ".join(f"{s} = {_v(x)}" for s, x in t.items() if isinstance(x, (int, float)))]
    return {"parts": parts, "res": res, "niveau": niveau}


FORMULES_TABLE = {
    "flexion": ("σ<sub>m,Ed</sub> = M<sub>Ed</sub> / W ≤ f<sub>m,d</sub>", "σ<sub>m,Ed</sub>", "f<sub>m,d</sub>"),
    "cisaillement": ("τ<sub>Ed</sub> = 1,5·V<sub>Ed</sub> / (b·h) ≤ f<sub>v,d</sub>", "τ<sub>Ed</sub>", "f<sub>v,d</sub>"),
    "appui": ("σ<sub>c,90,Ed</sub> = R / (b·a) ≤ f<sub>c,90,d</sub>", "σ<sub>c,90,Ed</sub>", "f<sub>c,90,d</sub>"),
    "fleche_inst": ("w<sub>inst</sub> = 5·q·L⁴ / (384·E·I) ≤ w<sub>lim</sub>", "w<sub>inst</sub>", None),
    "fleche_fin": ("w<sub>fin</sub> = (1 + k<sub>def</sub>)·w(G + ψ<sub>2</sub>Q) ≤ w<sub>lim</sub>", "w<sub>fin</sub>", None),
}


def _valeur_limite(v: Dict) -> tuple:
    f, sv, sl = FORMULES_TABLE[v["cle"]]
    u = v["unite"]
    val = f"{sv} = {_n(v['valeur'], u)} {u}"
    lim = f"{sl} = {_n(v['limite'], u)} {u}" if sl else f"{C.esc(v['limite_texte'])} = {_n(v['limite'], u)} {u}"
    return f, val, lim


def _piece(v: Dict, r: Dict) -> str:
    g, geo, p = r["grandeurs"], r["geometrie"], r["poutre"]
    k = v["cle"]
    if k == "flexion":
        sec = _tag(geo["tag"]) if geo["tag"] else f"{C.nb(geo['b'])} × {C.nb(geo['h'])} sur mesure"
        return f"{sec} · {C.esc(r['materiau']['classe'])}"
    if k == "cisaillement":
        return f"V<sub>Ed</sub> = {_n(g['V_Ed'], 'kN')} kN"
    if k == "appui":
        return f"a = {C.nb(p['a'])} mm"
    return f"w = {_n(v['valeur'], 'mm')} mm · {C.esc(v['limite_texte'])}"


def _ligne_tableau(v: Dict, r: Dict, gov: bool) -> Dict:
    f, val, lim = _valeur_limite(v)
    return {"nom": C.esc(COURTS[v["cle"]]), "sous": _piece(v, r), "formule": f, "valeur": val, "limite": lim, "taux": _aff(v["taux"]),
            "niveau": _niv(v), "gov": gov}


def _ecarts(r: Dict, cles: Sequence[str]) -> str:
    """Le pied d'une fenêtre de calcul : les écarts du module Streamlit avec l'EC5 qui jouent ici, reproduits."""
    items = ""
    for c in r["ecarts"]:
        if c not in cles:
            continue
        items += f"<li><b>{C.esc(MO.ECARTS[c][1])}</b> · {C.symb_math(MO.ECARTS[c][0])}"
        cg = r["indicatif"].get("combinaison_G")
        if c == "combinaison_G" and cg:
            items += (f" — ici, à titre indicatif : q = γ<sub>G</sub>·G = {_n(cg['q'], 'kN/m')} kN/m, k<sub>mod</sub> = {C.nb(cg['kmod'])} ; "
                      f"flexion {_n(cg['sigma_m_Ed'], 'MPa')} / {_n(cg['fm_d'], 'MPa')} = <b>{_pc(cg['u_flex'])}</b>, "
                      f"cisaillement {_n(cg['tau_Ed'], 'MPa')} / {_n(cg['fv_d'], 'MPa')} = <b>{_pc(cg['u_shear'])}</b> (non retenu)")
        items += "</li>"
    if not items:
        return ""
    return (f'<div class="ck-autres bo-ecarts"><b>Écarts connus avec l\'EC5, reproduits de l\'écran Streamlit</b>'
            f'<ul>{items}</ul></div>')


def _fenetres(r: Dict, nom: str) -> str:
    mat, co, geo, ch, p = r["materiau"], r["coefficients"], r["geometrie"], r["charges"], r["poutre"]
    vs = {v["cle"]: v for v in r["verifications"]}
    sec = f"{C.nb(geo['b'])} × {C.nb(geo['h'])} mm"
    out = ""
    for ident, titre, cles, ecarts in (("elu", "résistance à l'ELU", ELU, ("kh_reduit", "cisaillement_kcr", "combinaison_G", "appui_els", "poids_propre",
                                                                           "deversement")),
                                        ("els", "flèches à l'ELS", ELS, ("fleche_unite", "fleche_finale", "poids_propre", "deversement"))):
        presentes = [vs[c] for c in cles if c in vs]
        t_max = max(v["taux"] for v in presentes)
        lignes = [_ligne_tableau(v, r, v["taux"] == t_max) for v in presentes]
        if ident == "elu":
            hyps = [f"Bois <b>{C.esc(mat['classe'])}</b> · f<sub>m,k</sub> = {C.nb(mat['fm_k'])} · f<sub>v,k</sub> = {C.nb(mat['fv_k'])} · "
                    f"f<sub>c,90,k</sub> = {C.nb(mat['fc90_k'])} MPa",
                    f"Classe de service <b>{co['service']}</b> · {DUREES_COURTES[co['duree']]} · k<sub>mod</sub> = {C.nb(co['kmod'])}",
                    f"γ<sub>M</sub> = {C.nb(co['gamma_M'])} · k<sub>h</sub> = {C.fr(co['kh'], 3)}",
                    f"Section {sec} · L = {C.fr(geo['L'], 2)} m · q<sub>ELU</sub> = {C.fr(ch['q_ELU'], 2)} kN/m"]
            groupes = [("Charge et efforts", [f for f in vs["flexion"]["formules"] if f["cle"] in ("q_ELU", "M_Ed")]
                        + [f for f in vs["cisaillement"]["formules"] if f["cle"] == "V_Ed"], vs["flexion"]),
                       ("Flexion", [f for f in vs["flexion"]["formules"] if f["cle"] not in ("q_ELU", "M_Ed")], vs["flexion"]),
                       ("Cisaillement", [f for f in vs["cisaillement"]["formules"] if f["cle"] != "V_Ed"], vs["cisaillement"])]
            if "appui" in vs:
                groupes.append(("Compression perpendiculaire au fil, à l'appui", vs["appui"]["formules"], vs["appui"]))
        else:
            hyps = [f"Bois <b>{C.esc(mat['classe'])}</b> · E<sub>0,mean</sub> = {C.fr(mat['E0_mean'], 0)} MPa",
                    f"Classe de service <b>{co['service']}</b> · k<sub>def</sub> = {C.nb(co['kdef'])} · ψ<sub>2</sub> = {C.nb(ch['psi2'])}",
                    f"Limites <b>{C.esc(p['lim_inst'])}</b> (instantanée) · <b>{C.esc(p['lim_fin'])}</b> (finale)",
                    f"Section {sec} · L = {C.fr(geo['L'], 2)} m"]
            groupes = [("Flèche instantanée", vs["fleche_inst"]["formules"], vs["fleche_inst"]),
                       ("Flèche finale", vs["fleche_fin"]["formules"], vs["fleche_fin"])]
        detail = "".join(f'<div class="ck-g"><div class="ck-t">{C.esc(t)}</div>{C.lignes_fx([_formule(f, v) for f in fs])}</div>'
                         for t, fs, v in groupes)
        out += C.fenetre_calcul(ident, f"{C.esc(nom)} · {titre}", "les formules de l'écran Streamlit, valeurs substituées", hyps, lignes,
                                detail, _ecarts(r, ecarts))
    return out


def _facts(r: Dict) -> str:
    """Les chiffres clés de l'écran Streamlit (ses quatre métriques, M_Ed et V_Ed)."""
    geo, mat, g = r["geometrie"], r["materiau"], r["grandeurs"]
    rows = (("Section", f"{C.nb(geo['b'])} × {C.nb(geo['h'])} mm"), ("Portée", f"{C.fr(geo['L'], 2)} m"),
            ("E<sub>0,mean</sub>", f"{C.fr(mat['E0_mean'], 0)} MPa"), ("ρ<sub>mean</sub>", f"{C.fr(mat['rho_mean'], 0)} kg/m³"),
            ("M<sub>Ed</sub>", f"{_n(g['M_Ed'], 'kN·m')} kN·m"), ("V<sub>Ed</sub>", f"{_n(g['V_Ed'], 'kN')} kN"))
    return '<div class="bo-cles"><dl class="facts">' + "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in rows) + "</dl></div>"


def _calculer(p: Dict) -> Dict:
    try:
        return MO.verifier(p)
    except ValueError as e:                                     # pragma: no cover — normaliser l'a déjà refusé
        return {"erreur": str(e)}


def resume_note(etat: Dict, resultats: List[Dict]) -> Dict:
    """Le résumé de la NOTE : sa poutre la plus chargée."""
    vals = [(r["taux_max"], i) for i, r in enumerate(resultats) if r.get("taux_max") is not None]
    if not vals:
        return {"taux": None, "niveau": "x", "statut": "Calcul impossible", "gouvernant": ""}
    t, i = max(vals)
    r = resultats[i]
    niv = "r" if r["statut"] == MO.NOK else C.niveau(t)
    return {"taux": _aff(t), "niveau": niv, "statut": C.STATUTS[niv][1],
            "gouvernant": f"{etat['poutres'][i].get('nom') or 'Poutre'} · {COURTS[r['gouvernant']['cle']].lower()}"}


def elements(etat: Dict, resultats: List[Dict]) -> str:
    items = ""
    for i, (p, r) in enumerate(zip(etat["poutres"], resultats)):
        t = r.get("taux_max")
        niv = ("r" if r.get("statut") == MO.NOK else C.niveau(t)) if t is not None else "x"
        txt = f"{C.pc(_aff(t))}{C.NBSP}%" if t is not None else "—"
        geo = r.get("geometrie") or {}
        dims = f"{C.nb(geo['b'])} × {C.nb(geo['h'])}" if geo else ""
        sup = f'<button class="x" data-supprimer="{i}" title="Supprimer">{C.icon("x")}</button>' if len(etat["poutres"]) > 1 else ""
        items += (f'<div class="ni{" on" if i == etat["actif"] else ""}" data-activer="{i}"><span class="t">{C.esc(p.get("nom") or f"Poutre {i + 1}")} '
                  f'<span>· {dims} · {C.esc(p["classe"])}</span></span><span class="pct {niv}">{txt}</span>{sup}</div>')
    return (f'<div class="elements"><div class="nl-h"><span class="kicker">Dans cette note · {len(etat["poutres"])}</span>'
            f'<button class="nl-add" data-dupliquer="1" title="Copier la poutre ouverte">{C.icon("copy")}Copier</button></div>{items}'
            f'<div class="nl-add" data-ajouter="1">{C.icon("plus")}Ajouter une poutre</div></div>')


def resultat(etat: Dict) -> Dict:
    p = _poutre(etat)
    r = MO.verifier(p)
    resultats = [r if i == etat["actif"] else _calculer(q) for i, q in enumerate(etat["poutres"])]
    nom = p.get("nom") or "Poutre"
    vs = {v["cle"]: v for v in r["verifications"]}
    g, geo = r["grandeurs"], r["geometrie"]
    t = r["taux_max"]
    niv = "r" if r["statut"] == MO.NOK else C.niveau(t)
    v_g = vs[r["gouvernant"]["cle"]]
    _, val, lim = _valeur_limite(v_g)
    verdict = (f'<div class="verdict">{C.chip(_aff(t), niv, xl=True)}<div class="gov"><b>{C.esc(COURTS[v_g["cle"]])}</b>'
               f'{val} {"≤" if v_g["taux"] <= 1 else "&gt;"} {lim}</div></div>')
    # la plus légère qui tient (l'étoile de la ligne « Flexion »)
    opt = MO.optimal(p)
    leger = (f"Plus légère qui tient : {_tag(opt['nom'])} · {C.pc(_aff(opt['taux']))} %"
             if opt and not (p["standard"] and opt["nom"] == p["section"]) else "")
    rows = ""
    for titre, cles, cal in (("ELU · résistance", ELU, "elu"), ("ELS · flèches", ELS, "els")):
        presentes = [vs[c] for c in cles if c in vs]
        t_g = max(v["taux"] for v in presentes)
        n_g = "r" if any(v["etat"] == MO.NOK for v in presentes) else C.niveau(t_g)
        rows += f'<div class="elgrp">{titre}<span class="pct {n_g}">{C.pc(_aff(t_g))}{C.NBSP}%</span></div>'
        for v in presentes:
            rows += C.ligne_res(COURTS[v["cle"]], _piece(v, r), _aff(v["taux"]), _niv(v), calcul=cal,
                                pk="section" if v["cle"] == "flexion" else "", gov=v["cle"] == v_g["cle"],
                                leger=leger if v["cle"] == "flexion" else "")
    ind, ch = r["indicatif"], r["charges"]
    # en tête, le défaut NON conservatif : la combinaison γ_G·G seule, quand elle gouverne la résistance et atteint la
    # bande d'attention (95 %) — en rouge au-delà de 100 % ; ses chiffres sont toujours au pied du calcul ELU
    messages = ""
    cg = ind.get("combinaison_G")
    u_elu = max(vs["flexion"]["taux"], vs["cisaillement"]["taux"])
    u_G = max(cg["u_flex"], cg["u_shear"]) if cg else 0.0
    if u_G > u_elu and u_G >= 0.95:
        messages += C.message(
            f"<b>Combinaison γ<sub>G</sub>·G seule non vérifiée par l'écran Streamlit</b> : avec k<sub>mod</sub> permanent = {C.nb(cg['kmod'])} "
            f"(au lieu de {C.nb(r['coefficients']['kmod'])}), q = {C.nb(ch['gamma_G'])} × {C.nb(ch['G'])} = {_n(cg['q'], 'kN/m')} kN/m "
            f"gouverne la résistance : flexion {_pc(cg['u_flex'])}, cisaillement {_pc(cg['u_shear'])} (à titre indicatif, non retenu ; "
            f"détail au pied du calcul ELU).", grave=u_G > 1.0)
    messages += C.message(
        f"<b>Flèches de l'écran Streamlit 1 000 fois trop grandes</b> : il multiplie q (kN/m) par 1 000, alors que 1 kN/m vaut déjà "
        f"1 N/mm. Le site reprend ses chiffres tels quels (l'écran reste la référence) ; corrigées de l'unité, elles vaudraient "
        f"w<sub>inst</sub> = {_n(ind['w_inst_unite'], 'mm')} mm et w<sub>fin</sub> = {_n(ind['w_fin_unite'], 'mm')} mm "
        f"(à titre indicatif, non retenues).", grave=True)
    messages += C.message(f"Poids propre non compté (≈ {C.fr(ind['poids_propre'], 3)} kN/m) : l'inclure dans G. Vérifications EC5 simplifiées "
                          "de l'écran Streamlit (deux appuis, charge uniforme) ; la base des classes et les coefficients sont à ajuster selon "
                          "l'Annexe nationale, comme le rappelle l'écran. Les écarts connus avec l'EC5 sont au pied du détail du calcul.")
    pied = (f'<div class="pfoot"><div class="mass"><span>Masse indicative du bois · ρ<sub>mean</sub></span>'
            f'<b>{C.fr(r["indicatif"]["masse"], 1)}{C.NBSP}kg</b></div><div class="legend">{C.legende()}</div></div>')
    html_res = f'<div class="pbody">{verdict}<div class="els">{rows}</div>{_facts(r)}{messages}</div>{pied}'
    sec = f"{C.nb(geo['b'])} × {C.nb(geo['h'])} mm"
    vues = [{"cle": "poutre", "titre": "Schéma et efforts", "court": "Schéma et efforts", "svg": DB.poutre("bo_poutre", r),
             "legende": "charges G et Q, appuis et portée à l'échelle ; V et M à l'ELU"},
            {"cle": "deformee", "titre": "Déformée", "court": "Déformée", "svg": DB.deformee("bo_deformee", r),
             "legende": f"w inst. {_n(g['w_inst'], 'mm')} mm · w fin. {_n(g['w_fin'], 'mm')} mm (valeurs de l'écran Streamlit)"},
            {"cle": "section", "titre": "Section", "court": "Section", "svg": DB.section("bo_section", r),
             "legende": f"{sec} · {r['materiau']['classe']} · contraintes de calcul à l'ELU"}]
    phrase = (f"{sec} · {C.esc(p['classe'])} · L = {C.fr(geo['L'], 2)} m · q<sub>ELU</sub> = {C.fr(r['charges']['q_ELU'], 2)} kN/m")
    return {"resultat": html_res, "volets": _fenetres(r, nom), "vues": vues, "phrase": phrase, "statut_html": C.statut(niv),
            "resume": resume_note(etat, resultats), "donnees": formulaire(etat, r), "elements": elements(etat, resultats)}


# ------------------------------------------------------------------ le sélecteur de sections
def selecteur(etat: Dict, role: str, familles=None, froid: bool = True) -> str:
    """Les sections standard de l'écran, chacune calculée par le moteur, triées par masse (= par aire) ; la retenue ✓
    et la plus légère qui tient ★."""
    p = _poutre(etat)
    cands = MO.candidats_sections(p)
    i_o = next((k for k, c in enumerate(cands) if c["optimal"]), 0)
    i_c = next((k for k, c in enumerate(cands) if c["retenue"]), i_o)
    rows = [{"nom": _tag(c["nom"]), "col": C.fr(c["masse"], 2), "taux": _aff(c["taux"]), "niveau": "r" if not c["ok"] else C.niveau(c["taux"]),
             "cur": c["retenue"], "opt": c["optimal"],
             "prop": (f"{_tag(c['nom'])} mm · A = {C.fr(c['A'] / 100, 1)} cm² · I = {C.fr(c['I'] / 1e4, 1)} cm⁴ · "
                      f"W = {C.fr(c['W'] / 1e3, 1)} cm³ · gouverne : {C.esc(c['gouvernant'].lower())} ({_pc(c['taux'])})")} for c in cands]
    sous = "triées par masse · un clic retient la section" + ("" if any(c["ok"] for c in cands) else " · aucune ne tient")
    return C.selecteur(role, f"Section · bois massif {C.esc(p['classe'])}", sous, "", rows, max(0, min(i_o, i_c) - 4), "kg/m",
                       etoile="la plus légère qui tient")

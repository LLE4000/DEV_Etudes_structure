# -*- coding: utf-8 -*-
"""LA REVUE FINALE du module Barres (lot C, bloc C8) : les deux méthodes côte à côte sur tous les cas du brief v12
(§ 11), les écarts avec les classeurs du bureau (§ 6.7), la qualification externe, les listes restantes. Écrit LE
rapport daté du lot C : docs/validation/barres_revue_finale_AAAA-MM-JJ.md (une mesure datée ne se réécrit jamais).

Hors moteur, comme tools/validation_pied_poteau.py : ce script CHOISIT et COMPOSE ce que le moteur NOMME. Une barre du
catalogue passe par acier.barre.moteur.verifier ; un cas hors catalogue ou hors nuance (les données d'un classeur, une
section de la doc) passe par historique.verifier et ec3.verifier, avec section.saisie. Les sept sujets de la
comparaison sont ceux du moteur (acier/barre/comparaison.py : SUJETS, case) — jamais une seconde liste.

UNE DONNÉE propre au rapport, déclarée en tête (jamais une règle du module, jamais une ligne de la note — brief § 1,
règle 1) : un cas des classeurs ne donne que des efforts de SERVICE ; l'Eurocode 3 le reçoit à N_Ed = 1,5 × P et
M_Ed = 1,5 × M (la même sécurité que s = 1,5 : test 12 du § 11.5).

    /usr/bin/python3.12 tools/comparaison_barres.py                 # le rapport du jour, dans docs/validation/
    /usr/bin/python3.12 tools/comparaison_barres.py --sortie F.md   # ailleurs (rien n'est réécrit)
    /usr/bin/python3.12 tools/comparaison_barres.py --parametres    # réécrit le § 5.8 de docs/acier/BARRES_ACIER.md

Le rapport daté n'est JAMAIS comparé à l'égalité (tests/test_barres_revue.py) : le prochain changement du moteur le
ferait rougir pour toujours. Le test rejoue le script et compare au BRIEF.
"""
import argparse
import datetime
import math
import os
import re
import sys
import unicodedata
from typing import Dict, List, Optional

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

from acier.barre import comparaison as CP  # noqa: E402
from acier.barre import ec3 as EC  # noqa: E402
from acier.barre import ec3_ame as AM  # noqa: E402
from acier.barre import entree as EN  # noqa: E402
from acier.barre import historique as HI  # noqa: E402
from acier.barre import historique_deversement as HD  # noqa: E402
from acier.barre import mcr as MC  # noqa: E402
from acier.barre import moteur as MO  # noqa: E402
from acier.barre import parametres as PA  # noqa: E402
from acier.barre import prs as PR  # noqa: E402
from acier.barre import resultat as RE  # noqa: E402
from acier.barre import section as SE  # noqa: E402
from acier.garde_corps import membres as ME  # noqa: E402

DOSSIER = os.path.join(RACINE, "docs", "validation")
DOC = os.path.join(RACINE, "docs", "acier", "BARRES_ACIER.md")
S = 1.5                          # LA DONNÉE du rapport : N_Ed = S·P et M_Ed = S·M pour un cas des classeurs
E_, G_ = 210000.0, 81000.0
SEUIL_ECART = 10.0               # points de % : au-delà, une phrase dit d'où vient l'écart
TOL_BRIEF = 0.01                 # brief, règle 7 : ≤ 1 % conforme
DONNEE = ("Donnée du rapport (pas une règle du module, ni une ligne de la note) : un cas des classeurs ne donne que des "
          "efforts de service ; l'Eurocode 3 le reçoit à N_Ed = 1,5 × P et M_Ed = 1,5 × M, la même sécurité que s = 1,5.")
Z = {"N": 0.0, "My": 0.0, "Mz": 0.0, "Vz": 0.0, "Vy": 0.0}
PRS_A = {"b1": 300.0, "t1": 30.0, "hw": 540.0, "tw": 15.0, "b2": 300.0, "t2": 30.0, "a": 6.0, "bords": "oxycoupes", "nom": None}
PRS_B = dict(PRS_A, b2=200.0)
PRS_C = {"b1": 300.0, "t1": 30.0, "hw": 1420.0, "tw": 25.0, "b2": 500.0, "t2": 50.0, "a": None, "bords": "oxycoupes", "nom": None}


# ------------------------------------------------------------------ écriture
def fr(x, nd: int = 1) -> str:
    """Un nombre à la française (l'arrondi du module, resultat.fixe) ; « — » pour None."""
    return "—" if x is None else RE.fr(x, nd)


def pc(t) -> str:
    """Un taux en % (une décimale)."""
    return "—" if t is None else fr(100.0 * t, 1) + " %"


def mot_statut(statut: Optional[str], taux: Optional[float] = None) -> str:
    """Les six statuts du brief § 9 : vérifié, à la limite (le seuil des notes du site, membres.SEUIL_ATTENTION),
    non vérifié, hors domaine, non couvert, bloqué."""
    if statut == RE.OK:
        return "à la limite" if taux is not None and taux >= ME.SEUIL_ATTENTION else "vérifié"
    return {RE.NOK: "non vérifié", RE.HORS_DOMAINE: "hors domaine", RE.NON_COUVERT: "non couvert", RE.BLOQUE: "bloqué"}.get(statut, "—")


# ------------------------------------------------------------------ les barres
def _barre(**k) -> Dict:
    b = EN.nouvelle_barre("B")
    prs = k.pop("prs", None)
    b.update(k)
    if prs:
        b["prs"] = dict(prs)
        b.pop("profil")
    return b


def _q(v, a, b):
    return {"type": "q", "v": v, "a": a, "b": b, "niveau": None}


def _P(v, a, ss=None):
    return {"type": "P", "v": v, "a": a, "ss": ss, "niveau": None}


def efforts(profil, nuance, L, N=0.0, My=0.0, Mz=0.0, Lc=None, u=1.0, courbes=(None, None), diagramme="constant",
            niveau="centre", deversement=True, historique=None, prs=None, elu=None, Lz=None) -> Dict:
    """Une barre du CATALOGUE (ou un PRS) en efforts imposés, par le moteur : service (kN, kN·m) ; ELU = S × service
    (la donnée du rapport), sauf ``elu`` donné (un cas de logiciel, qui donne les efforts de calcul)."""
    serv = dict(Z, N=N, My=My, Mz=Mz)
    b = _barre(profil=profil, nuance=nuance, mode="efforts", Lc=L if Lc is None else Lc, niveau=niveau, prs=prs,
               flambement={"y": {"L": L, "u": u, "courbe": courbes[0]}, "z": {"L": L if Lz is None else Lz, "u": u, "courbe": courbes[1]}},
               diagramme={"y": {"forme": diagramme, "psi": 1.0}, "z": {"forme": "constant", "psi": 1.0}},
               efforts={"service": serv, "elu": dict(Z, **elu) if elu else {k: S * v for k, v in serv.items()}})
    b["historique"]["deversement"] = deversement
    b["historique"].update(historique or {})
    return MO.verifier(b)


def _essayer(calcul):
    """(résultat, None) ou (None, le motif du refus) : une méthode qui refuse a un statut NOMMÉ, jamais rien."""
    try:
        return calcul(), None
    except (RE.Refus, ValueError) as e:
        return None, str(e)


def deux(section, fy, nuance, P=0.0, My=0.0, Mz=0.0, L=0.0, u=1.0, courbes=(None, None), deversement=None, C=None, Lz=None):
    """Un cas HORS catalogue (section.saisie) : la méthode ELS (historique.verifier, f_y donnée) et l'Eurocode 3
    (ec3.verifier, la nuance ; None : pas de nuance du module) — N, N·mm, mm. Rend (ELS, EC3, motifs des refus)."""
    Lz = L if Lz is None else Lz
    axes = {"y": {"L": L, "u": u, "courbe": courbes[0]}, "z": {"L": Lz, "u": u, "courbe": courbes[1]}}
    h, mh = _essayer(lambda: HI.verifier({"section": section, "fy": fy, "s": S, "P": P, "My": My, "Mz": Mz, "axes": axes,
                                          "C": C or {}, "deversement": deversement or {"actif": False}}))
    if nuance is None:
        e, me = None, f"non couvert : f_y = {fr(fy, 0)} N/mm² n'est pas une nuance du module (S235 à S460)"
    else:
        ax = {a: {"L": axes[a]["L"], "u": u} for a in ("y", "z")}
        dv = {"Lc": (deversement or {}).get("Lc", 0.0 if not (deversement or {}).get("actif", True) else Lz)}
        e, me = _essayer(lambda: EC.verifier({"section": section, "nuance": nuance, "NEd": S * P, "MyEd": S * My, "MzEd": S * Mz,
                                              "axes": ax, "deversement": dv}))
    return h, e, {"historique": mh and f"non calculé : {mh}", "ec3": me if nuance is None or not me else f"non calculé : {me}"}


def _v(r, ident):
    return None if r is None else next((v for v in r["verifications"] if v["id"] == ident), None)


def _taux(r, ident):
    v = _v(r, ident)
    return None if v is None else v["taux"]


def _limite(r, ident):
    v = _v(r, ident)
    return None if v is None else v["limite"]


def _g(r, *cles):
    """Une grandeur nommée par le moteur (r["grandeurs"][…]) ; None si elle manque."""
    x = None if r is None else r.get("grandeurs")
    for k in cles:
        x = x.get(k) if isinstance(x, dict) else None
    return x


def cas(groupe, nom, donnees, historique=None, ec3=None, ame=None, absent=None, brief=(), note=None, capacite=None, causes=None) -> Dict:
    """Un cas du rapport. ``donnees`` : {profil, nuance, fy_els, fy_ec3, L, mode, efforts} ; ``absent`` : le motif d'une
    méthode sans résultat (un statut NOMMÉ) ; ``brief`` : [(quoi, attendu, obtenu, tolérance relative)] ; ``capacite`` :
    un cas sans charge (capacité ELS × s contre la résistance de l'Eurocode 3)."""
    ab = {k: v for k, v in (absent or {}).items() if v}            # (une méthode sans résultat ni motif : le test le refuse)
    return {"groupe": groupe, "nom": nom, "donnees": donnees, "historique": historique, "ec3": ec3, "ame": ame, "absent": ab,
            "brief": [dict(quoi=q, attendu=a, obtenu=o, tol=t) for q, a, o, t in brief], "note": note, "capacite": capacite, "causes": causes}


def _du_moteur(groupe, nom, r, donnees, **k):
    return cas(groupe, nom, donnees, historique=r["methodes"]["historique"], ec3=r["methodes"]["ec3"], ame=r["ame"], **k)


NOMS = {"historique": "ELS", "ec3": "Eurocode 3"}


def _fy(r, cote):
    """La f_y que la méthode a prise (N/mm²), nommée par le moteur."""
    return None if r is None else _g(r, "fy")


def _donnees(profil, nuance, L, mode, eff, fy_els=None, fy_ec3=None):
    return {"profil": profil, "nuance": nuance, "fy_els": fy_els, "fy_ec3": fy_ec3, "L": L, "mode": mode, "efforts": eff}


def _completer_fy(c):
    d = c["donnees"]
    if d["fy_els"] is None:
        d["fy_els"] = _fy(c["historique"], "historique")
    if d["fy_ec3"] is None:
        d["fy_ec3"] = _fy(c["ec3"], "ec3")
    return c


# ------------------------------------------------------------------ 1. les classeurs « Flambement » (brief § 11.1)
G1 = "1. Classeurs « Flambement » (11.1, efforts imposés, u = 1)"
G2 = "2. Exemples de la doc théorique (11.1)"
G3 = "3. Déversement des classeurs (11.1 bis) : capacité, sans charge"
G4 = "4. Eurocode 3 : 11.2, R2 et R3 (IPE 300, 6 m)"
G5 = "5. PRS (11.8, 11.9)"
G6 = "6. Logiciels du bureau (SCIA S1 à S3)"
GROUPES = (G1, G2, G3, G4, G5, G6)


def cas_classeurs() -> List[Dict]:
    out = []
    r = efforts("HEB 200", "S235", 7.2, N=432.72)
    out.append(_du_moteur(G1, "HEB 200", r, _donnees("HEB 200 (catalogue)", "S235", 7.2, "efforts imposés", "P = 432,72 kN ; N_Ed = 649,08 kN"),
                          brief=[("ELS, taux de flambement z (σ 175,09 / 156,67)", 1.118, _taux(r["methodes"]["historique"], "H-N-z"), TOL_BRIEF),
                                 ("EC3, taux de flambement z (649,08 / 570,1)", 1.139, _taux(r["methodes"]["ec3"], "E-Nb-z"), TOL_BRIEF),
                                 ("EC3, N_b,z,Rd (kN, 11.2)", 570.1, _limite(r["methodes"]["ec3"], "E-Nb-z"), TOL_BRIEF),
                                 ("EC3, λ̄_z (11.2)", 1.5122, _g(r["methodes"]["ec3"], "flambement", "z", "lambda_bar"), TOL_BRIEF),
                                 ("EC3, χ_z (11.2)", 0.3106, _g(r["methodes"]["ec3"], "flambement", "z", "chi"), TOL_BRIEF)]))
    r = efforts("HEA 120", "S235", 1.0, N=112.0)
    h = r["methodes"]["historique"]
    out.append(_du_moteur(G1, "HEA 120 Chancel", r, _donnees("HEA 120 (catalogue)", "S235", 1.0, "efforts imposés", "P = 112 kN"),
                          brief=[("ELS, φ_z", 0.9180656, _g(h, "z", "phi"), TOL_BRIEF), ("ELS, φ_y", 0.9821556, _g(h, "y", "phi"), TOL_BRIEF),
                                 ("ELS, σ (N/mm²)", 48.1435, _g(h, "sigma"), TOL_BRIEF)]))
    # le CHS 168,3 × 6 du classeur : A = 2 520 mm² (donnée fausse, dite au § 2) — ses données, saisies ; M = 0,05·P + poids propre
    chs = SE.saisie("tube_rond", "creux_froid", nom="CHS 168,3 × 6 (classeur)", t=6.0, d=168.3, A=2520.0, Iy=1009e4, Iz=1009e4, iy=57.4, iz=57.4,
                    Wel_y=1009e4 / 84.15, Wel_z=1009e4 / 84.15)
    Mz = 35000 * 0.05 + 78600 * 2520 / 1e6 * 7.0 ** 2 / 8
    h, e, ab = deux(chs, 235.0, "S235", P=35000.0, Mz=Mz * 1e3, L=7000.0, courbes=("c", "c"))
    out.append(cas(G1, "CHS 168,3 × 6 (A du classeur)", _donnees("CHS 168,3 × 6, données du classeur (saisies)", "S235", 7.0, "efforts imposés",
                                                                  "P = 35 kN ; M_z = 2,963 kN·m", 235.0, None),
                   historique=h, ec3=e, absent=ab,
                   brief=[("ELS, φ", 0.3930340, _g(h, "z", "phi"), TOL_BRIEF), ("ELS, PE", 0.8769887, _g(h, "z", "PE"), TOL_BRIEF),
                          ("ELS, σ (N/mm²)", 63.5168, _g(h, "sigma"), TOL_BRIEF)]))
    # la lecture cohérente : le tube du catalogue (formé à froid, A = 3 059 mm²), M autour de l'axe fort
    s = SE.du_catalogue("CHS 168.3x6 (froid)")
    My = (35000 * 0.05 + 78600 * s["A"] / 1e6 * 7.0 ** 2 / 8) / 1e3
    r = efforts("CHS 168.3x6 (froid)", "S235", 7.0, N=35.0, My=My)
    out.append(_du_moteur(G1, "CHS 168,3 × 6 (A = 3 059 mm²)", r, _donnees("CHS 168.3x6 (froid) (catalogue)", "S235", 7.0, "efforts imposés",
                                                                           f"P = 35 kN ; M_y = {fr(My, 3)} kN·m"),
                          brief=[("ELS, σ (N/mm²)", 59.75, _g(r["methodes"]["historique"], "sigma"), TOL_BRIEF),
                                 ("ELS, taux", 0.3814, r["methodes"]["historique"]["taux_max"], TOL_BRIEF)]))
    # REALEX : f_y 450, hors des nuances ; le tube Ø610 × 12,5 n'est pas au catalogue (décision n° 6 : fabrication, position)
    def realex(fab, W):
        return SE.saisie("tube_rond", fab, nom="Tube Ø610 × 12,5", t=12.5, d=610.0, A=23464.0, Iy=104754e4, Iz=104754e4, iy=211.3, iz=211.3,
                         Wel_y=W * 1e3, Wel_z=W * 1e3)
    h, e, ab = deux(realex("creux_chaud", 3506.0), 450.0, None, P=5e6, L=10000.0, courbes=("a", "a"))
    q = 78600 * 23464 / 1e6
    lect = {}
    for cle, fab, W, M, C in (("a", "creux_chaud", 3506.0, 0.0, "unite"), ("c", "creux_froid", 3506.0, 0.0, "unite"),
                              ("c+pp", "creux_froid", 3434.6, q * 100 / 8, "unite"), ("a+pp", "creux_chaud", 3434.6, q * 100 / 8, "unite")):
        lect[cle] = HI.verifier({"section": realex(fab, W), "fy": 450.0, "P": 5e6, "My": M * 1e3,
                                 "axes": {"y": {"L": 10000.0}, "z": {"L": 10000.0}}, "C": {"y": {"mode": C}}})["taux_max"]
    out.append(cas(G1, "REALEX Ø610 × 12,5", _donnees("Tube Ø610 × 12,5 (absent du catalogue, saisi)", "f_y 450", 10.0, "efforts imposés",
                                                       "P = 5 000 kN", 450.0, None),
                   historique=h, ec3=e, absent=ab,
                   brief=[("ELS, φ", 0.8552328, _g(h, "z", "phi"), TOL_BRIEF), ("ELS, P_adm (N)", 6020154.0, _g(h, "z", "P_adm"), TOL_BRIEF),
                          ("ELS, PE", 0.6545608, _g(h, "z", "PE"), TOL_BRIEF), ("ELS, σ (N/mm²)", 249.1630, _g(h, "sigma"), TOL_BRIEF),
                          ("ELS, lecture a (fini à chaud), taux", 0.831, lect["a"], TOL_BRIEF), ("ELS, lecture c, taux", 0.984, lect["c"], TOL_BRIEF),
                          ("ELS, lecture c + poids propre, taux", 1.018, lect["c+pp"], TOL_BRIEF),
                          ("ELS, lecture a + poids propre, taux", 0.865, lect["a+pp"], TOL_BRIEF)],
                   note="Décision n° 6 ouverte : le taux ELS va de 0,831 à 1,018 selon la fabrication et la position (test 6 du § 11.5)."))
    # FLAMBECBD (HEA 160) : i = 3,98 cm sur les deux axes, valeurs d'axe faible dans les cases « fort » (donnée dite au § 2)
    cbd = SE.saisie("I", "lamine", nom="HEA 160 (FLAMBECBD)", h=152, b=160, tf=9, tw=6, A=3877.0, Iy=615.6e4, Iz=3877 * 3.98 ** 2 * 100,
                    iy=39.8, iz=39.8, Wel_y=76.95e3, Wel_z=76.95e3)
    h, e, ab = deux(cbd, 235.0, "S235", P=300e3, My=3e6, L=1750.0, courbes=("b", "c"))
    out.append(cas(G1, "FLAMBECBD (HEA 160)", _donnees("HEA 160, données du classeur (saisies)", "S235", 1.75, "efforts imposés",
                                                        "P = 300 kN ; M = 3 kN·m", 235.0, None),
                   historique=h, ec3=e, absent=ab,
                   brief=[("ELS, φ_z", 0.8604919, _g(h, "z", "phi"), TOL_BRIEF), ("ELS, φ_y", 0.9131044, _g(h, "y", "phi"), TOL_BRIEF),
                          ("ELS, P_adm (N)", 522659.9, _g(h, "z", "P_adm"), TOL_BRIEF), ("ELS, σ avec φ_min (N/mm²)", 133.6, _g(h, "sigma"), TOL_BRIEF)]))
    # … ses lectures cohérentes : le HEA 160 du catalogue, M autour de l'axe faible, puis de l'axe fort (déversement des classeurs, fascicule)
    rz = efforts("HEA 160", "S235", 1.75, N=300.0, Mz=3.0)
    ry = efforts("HEA 160", "S235", 1.75, N=300.0, My=3.0)
    rf = efforts("HEA 160", "S235", 1.75, N=300.0, My=3.0, historique={"variante": "fascicule"})
    out.append(_du_moteur(G1, "FLAMBECBD, lecture axe faible", rz, _donnees("HEA 160 (catalogue)", "S235", 1.75, "efforts imposés", "P = 300 kN ; M_z = 3 kN·m"),
                          brief=[("ELS, σ (N/mm²)", 133.63, _g(rz["methodes"]["historique"], "sigma"), TOL_BRIEF)]))
    out.append(_du_moteur(G1, "FLAMBECBD, lecture axe fort", ry, _donnees("HEA 160 (catalogue)", "S235", 1.75, "efforts imposés", "P = 300 kN ; M_y = 3 kN·m ; L_d 1,75 m"),
                          brief=[("ELS, σ, classeurs (N/mm²)", 107.52, _g(ry["methodes"]["historique"], "sigma"), TOL_BRIEF),
                                 ("ELS, σ, variante fascicule (N/mm²)", 105.96, _g(rf["methodes"]["historique"], "sigma"), TOL_BRIEF)]))
    # FLAMBEBAD (IPE 270, S355) : I_y 5 790, W_y 428 (données du classeur)
    bad = SE.saisie("I", "lamine", nom="IPE 270 (FLAMBEBAD)", h=270, b=135, tf=10.2, tw=6.6, A=4594.0, Iy=5790e4, Iz=4594 * 3.02 ** 2 * 100,
                    iy=112.3, iz=30.2, Wel_y=428e3, Wel_z=62.2e3)
    Mb = 361 * 3.4 ** 2 / 8
    h, e, ab = deux(bad, 355.0, "S355", P=185e3, My=Mb * 1e3, L=3400.0, courbes=("a", "b"))
    out.append(cas(G1, "FLAMBEBAD (IPE 270)", _donnees("IPE 270, données du classeur (saisies)", "S355", 3.4, "efforts imposés",
                                                        "P = 185 kN ; M = 0,522 kN·m", 355.0, None),
                   historique=h, ec3=e, absent=ab,
                   brief=[("ELS, φ_z", 0.3519139, _g(h, "z", "phi"), TOL_BRIEF), ("ELS, φ_y", 0.9600961, _g(h, "y", "phi"), TOL_BRIEF),
                          ("ELS, P_adm,z (N)", 382617.2, _g(h, "z", "P_adm"), TOL_BRIEF), ("ELS, σ avec φ_min (N/mm²)", 115.7, _g(h, "sigma"), TOL_BRIEF)]))
    rb = efforts("IPE 270", "S355", 3.4, N=185.0, My=Mb / 1e3)
    rbf = efforts("IPE 270", "S355", 3.4, N=185.0, My=Mb / 1e3, historique={"variante": "fascicule"})
    out.append(_du_moteur(G1, "FLAMBEBAD, lecture cohérente", rb, _donnees("IPE 270 (catalogue)", "S355", 3.4, "efforts imposés", "P = 185 kN ; M_y = 0,522 kN·m ; L_d 3,40 m"),
                          brief=[("ELS, σ, classeurs (N/mm²)", 117.42, _g(rb["methodes"]["historique"], "sigma"), TOL_BRIEF),
                                 ("ELS, σ, variante fascicule (N/mm²)", 117.20, _g(rbf["methodes"]["historique"], "sigma"), TOL_BRIEF)]))
    return out


def cas_doc() -> List[Dict]:
    out = []
    iz = math.sqrt((101 ** 4 - 99 ** 4) / 12 / 400) * 10
    cais = SE.saisie("autre", "caisson", nom="caisson 100 × 100 × 1 cm", t=10.0, h=1010.0, b=1010.0, A=400e2, Iy=(101 ** 4 - 99 ** 4) / 12 * 1e4, Iz=(101 ** 4 - 99 ** 4) / 12 * 1e4,
                     iy=iz, iz=iz)
    h, e, ab = deux(cais, 235.0, "S235", P=4000e3, L=20000.0)
    out.append(cas(G2, "p. 6, caisson soudé", _donnees("caisson 100 × 100 × 1 cm (saisi)", "S235", 20.0, "efforts imposés", "P = 4 000 kN", 235.0, None),
                   historique=h, ec3=e, absent=ab,
                   brief=[("ELS, λ̄", 0.522, _g(h, "z", "lambda_bar"), TOL_BRIEF), ("ELS, φ (courbe b)", 0.891, _g(h, "z", "phi"), TOL_BRIEF),
                          ("ELS, P_adm (kN)", 5582.0, (_g(h, "z", "P_adm") or 0) / 1e3, TOL_BRIEF)]))
    cor = SE.saisie("L", "lamine", nom="L 60 × 60 × 8 (doc)", A=903.0, Iy=903 * 22.9 ** 2, Iz=903 * 11.7 ** 2, iy=22.9, iz=11.7)
    h, _e, _ab = deux(cor, 235.0, None, P=30e3, L=2000.0)
    rc = efforts("L 60x60x8", "S235", 2.0, N=30.0)
    out.append(cas(G2, "p. 8, cornière L 60 × 60 × 8", _donnees("L 60 × 60 × 8 (doc, saisie)", "S235", 2.0, "efforts imposés", "P = 30 kN", 235.0, None),
                   historique=h, ec3=rc["methodes"]["ec3"],
                   brief=[("ELS, λ̄", 1.820, _g(h, "z", "lambda_bar"), TOL_BRIEF), ("ELS, φ (courbe c)", 0.236, _g(h, "z", "phi"), TOL_BRIEF),
                          ("ELS, P_adm (kN)", 33.4, (_g(h, "z", "P_adm") or 0) / 1e3, TOL_BRIEF)]))
    s = SE.du_catalogue("HEA 220")
    h, e, ab = deux(s, 235.0, "S235", P=500e3, My=25e6, L=5000.0, Lz=0.0)
    out.append(cas(G2, "p. 14, HEA 220", _donnees("HEA 220 (catalogue)", "S235", 5.0, "efforts imposés, plan vertical seul (L_z = 0)",
                                                   "P = 500 kN ; M = 25 kN·m", 235.0, None),
                   historique=h, ec3=e, absent=ab,
                   brief=[("ELS, σ (N/mm²)", 148.3, _g(h, "sigma"), TOL_BRIEF), ("ELS, P_cr (kN)", 4485.0, (_g(h, "y", "P_cr") or 0) / 1e3, TOL_BRIEF)]))
    s = SE.du_catalogue("HEA 200")
    h, e, ab = deux(s, 235.0, "S235", P=38e3, My=48e6, L=8000.0)
    out.append(cas(G2, "p. 23, poutre BC HEA 200", _donnees("HEA 200 (catalogue)", "S235", 8.0, "efforts imposés, déversement non compté (la doc)",
                                                             "P = 38 kN ; M = 48 kN·m", 235.0, None),
                   historique=h, ec3=e, absent=ab,
                   brief=[("ELS, σ_P (N/mm²)", 27.0, _g(h, "sigma_P"), TOL_BRIEF), ("ELS, σ_M (N/mm²)", 129.7, _g(h, "sigma_My"), TOL_BRIEF),
                          ("ELS, σ (N/mm²)", 156.7, _g(h, "sigma"), TOL_BRIEF)],
                   note="La doc écarte le déversement (dispense de la p. 20, non reprise) ; l'Eurocode 3 le compte sur 8 m.",
                   causes={"Flexion composée": "déversement écarté par la doc, compté par l'Eurocode 3 : χ_LT dans (6.61)"}))
    s = SE.du_catalogue("HEA 300")
    h, e, ab = deux(s, 235.0, "S235", P=168e3, My=110e6, L=6000.0, u=1.5, C={"y": {"mode": "charges"}},
                    deversement={"actif": True, "Lc": 6000.0, "variante": "fascicule"})
    out.append(cas(G2, "p. 25, colonne AB HEA 300", _donnees("HEA 300 (catalogue)", "S235", 6.0, "efforts imposés, u = 1,5, C = 0,85, fascicule p. 19",
                                                              "P = 168 kN ; M = 110 kN·m", 235.0, None),
                   historique=h, ec3=e, absent=ab,
                   brief=[("ELS, φ_ed", 0.676, _g(h, "deversement", "phi"), TOL_BRIEF), ("ELS, σ_P (N/mm²)", 37.3, _g(h, "sigma_P"), TOL_BRIEF),
                          ("ELS, σ_M (N/mm²)", 116.1, _g(h, "sigma_My"), TOL_BRIEF), ("ELS, σ (N/mm²)", 153.3, _g(h, "sigma"), TOL_BRIEF)]))
    return out


def _capacite(r_els, r_ec3, W):
    """Capacité de service de la méthode ELS (σ_adm,d · W_el,y, kN·m), × s, contre M_b,Rd de l'Eurocode 3 (kN·m)."""
    sad = _limite(r_els, "H-DEV")
    Mb = _limite(r_ec3, "E-Mb")
    Ms = None if sad is None else sad * W / 1e6
    return {"sigma_adm_d": sad, "M_service": Ms, "M_els_s": None if Ms is None else S * Ms, "M_b_Rd": Mb,
            "rapport": None if Ms is None or Mb is None else S * Ms / Mb}


def cas_deversement() -> List[Dict]:
    """11.1 bis : la barre n'a pas de charge — la capacité ELS (σ_adm,d·W_el,y) × s contre M_b,Rd (moment constant, C_d = 1)."""
    out = []

    def lamine(nom, profil, nuance, l, att, **k):
        r = efforts(profil, nuance, l, My=10.0, **k)
        h, e = r["methodes"]["historique"], r["methodes"]["ec3"]
        cap = _capacite(h, e, SE.du_catalogue(profil)["Wel_y"])
        out.append(_du_moteur(G3, nom, r, _donnees(f"{profil} (catalogue)", nuance, l, "efforts imposés, moment constant", "sans charge"),
                              brief=[("ELS, σ_adm,d (N/mm²)", att, cap["sigma_adm_d"], TOL_BRIEF)], capacite=cap))
    lamine("HE 200 B, feuille automatique", "HEB 200", "S235", 4.0, 115.70)
    lamine("DEVERSEMENT_CAD, feuille S355 (180 × 180 × 14)", "HEB 180", "S355", 4.0, 147.50)
    lamine("φ_d = 1 (l = 0,30 m)", "HEB 200", "S235", 0.3, 156.67)
    lamine("IPE 600, C_d = 1", "IPE 600", "S235", 12.0, 42.58)
    lamine("IPE 600, C_d = 1,12, charge sur la semelle haute", "IPE 600", "S235", 12.0, 46.86, niveau="haut", historique={"Cd": 1.12})
    # les sections données par leurs seules dimensions (section.saisie) : la fonction des classeurs ; l'Eurocode 3 n'a ni I_t, ni I_w, ni W_pl
    for nom, forme, h_, b_, tf, l, att in (("U 140 × 60 × 10 (feuille S235)", "U", 140.0, 60.0, 10.0, 3800.0, 65.32),
                                           ("I 600 × 300 × 50 (t_f > 40, h/b > 1,2)", "I", 600.0, 300.0, 50.0, 6000.0, 119.03),
                                           ("I 300 × 300 × 80 (t_f = 80)", "I", 300.0, 300.0, 80.0, 6000.0, 143.92)):
        tw = 7.0 if forme == "U" else 10.0
        s = SE.saisie(forme, "lamine", nom=nom, h=h_, b=b_, tf=tf, tw=tw, A=1000.0, Iy=1e8, Iz=1e6, iy=100.0, iz=30.0, Wel_y=1e6)
        d = HD.classeurs(s, l, HI.lambda_E(235.0))
        sad = 235.0 / S * d["phi"]
        cap = {"sigma_adm_d": sad, "M_service": None, "M_els_s": None, "M_b_Rd": None, "rapport": None}
        out.append(cas(G3, nom, _donnees(f"{nom} (saisi : h, b, t_f)", "S235", l / 1e3, "fonction des classeurs", "sans charge", 235.0, None),
                       absent={"historique": f"fonction seule : classe {d['classe']}, φ_d {fr(d['phi'], 4)}, σ_adm,d {fr(sad, 2)} N/mm²",
                               "ec3": "non calculé : la section n'est donnée que par h, b et t_f (ni I_t, ni I_w, ni W_pl)"},
                       brief=[("ELS, σ_adm,d (N/mm²)", att, sad, TOL_BRIEF)], capacite=cap))
    return out


def cas_r2_r3() -> List[Dict]:
    out = []
    R2 = dict(poids_propre=False, appuis={"mode": "poses", "ss": 100.0, "c": 0.0, "detail": None},
              cas=[{"nom": "G", "nature": "permanent", "N": 0.0, "charges": [_q(2.42, 0.0, 6.0), _P(10.0, 3.0, 100.0)]},
                   {"nom": "Q", "nature": "exploitation_A", "N": 0.0, "charges": [_P(15.0, 3.0, 100.0)]}])
    for niveau, Mcr, taux in (("centre", 118.18, 0.887), ("haut", 86.98, 1.093)):
        r = MO.verifier(_barre(niveau=niveau, **R2))
        e = r["methodes"]["ec3"]
        out.append(_du_moteur(G4, f"R2 et R3, charges {'au centre de cisaillement' if niveau == 'centre' else 'sur la semelle haute'}", r,
                              _donnees("IPE 300 (catalogue)", "S235", 6.0, "barre chargée (6.10)", "G 2,42 kN/m + 10 kN ; Q 15 kN à 3 m"),
                              brief=[("EC3, M_Ed (kN·m)", 68.70, _v(e, "E-My") and _v(e, "E-My")["valeur"], TOL_BRIEF),
                                     ("EC3, V_Ed (kN)", 27.80, _v(e, "E-Vz") and _v(e, "E-Vz")["valeur"], TOL_BRIEF),
                                     ("EC3, M_cr (kN·m)", Mcr, (_g(e, "deversement", "M_cr") or 0) / 1e6, TOL_BRIEF),
                                     ("EC3, taux de déversement", taux, _taux(e, "E-Mb"), TOL_BRIEF)],
                              note="Catalogue du site : I_t 19,77 cm⁴ au lieu de 20,12 (brief) ; M_cr 0,6 % plus bas, accepté à 1 %."))
    return out


def r4() -> List[Dict]:
    """R4 (brief § 11.3) : l'âme de l'IPE 300 S235 sous une force concentrée — EN 1993-1-5, commune aux deux méthodes."""
    a = AM.ame_de(SE.du_catalogue("IPE 300"))
    P_ = PA.jeu()
    lignes = []
    for quoi, ss, t, c, contact, att in (("(a) s_s = 50", 50, "a", 0.0, False, 283.2), ("(a) s_s = 100", 100, "a", 0.0, False, 387.8),
                                         ("(a) s_s = 150", 150, "a", 0.0, False, 425.1), ("(c) s_s = 100, c = 0, contact « oui »", 100, "c", 0.0, True, 248.9),
                                         ("(c) s_s = 100, c = 0, contact « non » (défaut, n° 2)", 100, "c", 0.0, False, 199.82),
                                         ("(c) s_s = 100, c = 50, contact « oui »", 100, "c", 50.0, True, 335.9)):
        lignes.append(dict(quoi="F_Rd (kN), " + quoi, attendu=att, obtenu=AM.resistance(a, 235.0, P_, ss, t, c, contact=contact)["FRd"] / 1e3, tol=TOL_BRIEF))
    return lignes


def cas_prs() -> List[Dict]:
    out = []
    for nom, prs, att_els, att_mb in (("PRS-A, oxycoupé, 10 m", PRS_A, 90.31, 797.39), ("PRS-A, larges plats, 10 m", dict(PRS_A, bords="larges_plats"), 78.79, None),
                                      ("PRS-B, monosymétrique, 10 m", PRS_B, None, 617.55)):
        r = efforts(None, "S355", 10.0, My=100.0, diagramme="reparti", niveau="haut", prs=prs)
        h, e = r["methodes"]["historique"], r["methodes"]["ec3"]
        s = PR.section_prs(prs["b1"], prs["t1"], prs["hw"], prs["tw"], prs["b2"], prs["t2"], a=prs["a"], nuance="S355", bords=prs["bords"])
        cap = _capacite(h, e, s["Wel_y"])
        br = []
        if att_els:
            br.append(("ELS, σ_adm,d (N/mm²)", att_els, cap["sigma_adm_d"], TOL_BRIEF))
        if att_mb:
            br.append(("EC3, M_b,Rd (kN·m)", att_mb, cap["M_b_Rd"], TOL_BRIEF))
        if prs is PRS_A:
            br += [("ELS, capacité de service σ_adm,d·W_el,y (kN·m)", 499.8, cap["M_service"], TOL_BRIEF), ("ELS × 1,5 (kN·m)", 749.7, cap["M_els_s"], TOL_BRIEF)]
        out.append(_du_moteur(G5, nom, r, _donnees(f"{s['nom']} ({'oxycoupé' if prs['bords'] == 'oxycoupes' else 'larges plats'})", "S355", 10.0,
                                                   "efforts imposés, charge répartie sur la semelle haute", "sans charge"),
                              brief=br, capacite=cap))
    for nom, n in (("PRS-C, 8 maintiens", 8), ("PRS-C, maintiens aux tiers", 2), ("PRS-C, 18 maintiens", 18)):
        b = _barre(prs=PRS_C, nuance="S235", travees=[28.0], poids_propre=True, niveau="haut",
                   maintiens=[{"x": 28.0 * i / (n + 1), "genre": "haut", "neglige": False} for i in range(1, n + 1)],
                   cas=[{"nom": "E", "nature": "execution", "psi0": 1.0, "N": 0.0, "charges": [_q(20.80 + 2.40, 0.0, 28.0), _q(2.25, 12.5, 15.5)]}])
        r = MO.verifier(b)
        h, e = r["methodes"]["historique"], r["methodes"]["ec3"]
        att = {8: 89.68, 2: 42.91, 18: 118.3}[n]
        br = [("ELS, σ_adm,d (N/mm²)", att, _limite(h, "H-DEV"), TOL_BRIEF), ("ELS, σ = M/W (N/mm²)", 117.08, _v(h, "H-DEV") and _v(h, "H-DEV")["valeur"], TOL_BRIEF)]
        if n == 8:
            # (mise à jour 1) le 0,967 du brief est celui du SOLVEUR et de la section ÉLASTIQUE : les variantes des décisions
            # n° 12 et n° 11 ; les défauts d'aujourd'hui (formule de l'annexe E, classe 2 efficace) sont au tableau du cas
            eb = MO.verifier(dict(b, parametres={"mcr_retenu": "solveur", "ame_classe_3": "elastique"}))["methodes"]["ec3"]
            br += [("EC3, M_Ed (kN·m)", 4199.3, _v(e, "E-Mb") and _v(e, "E-Mb")["valeur"], TOL_BRIEF),
                   ("EC3, taux de déversement (solveur, section élastique)", 0.967, _taux(eb, "E-Mb"), TOL_BRIEF)]
        out.append(_du_moteur(G5, nom, r, _donnees("PRS 1500 (300 × 30 / 1 420 × 25 / 500 × 50, oxycoupé)", "S235", 28.0,
                                                   "barre chargée, exécution (ψ₀ = 1,0)", "béton + travail 23,20 kN/m ; zone 2,25 kN/m"),
                              brief=br))
    return out


def cas_scia() -> List[Dict]:
    out = []
    S1 = dict(poids_propre=False, cas=[{"nom": "G", "nature": "permanent", "N": 0.0, "charges": [_q(2.42, 0.0, 6.0), _P(10.0, 3.0)]},
                                       {"nom": "Q", "nature": "exploitation_A", "N": 0.0, "charges": [_P(15.0, 3.0)]}])
    for niveau, att in (("haut", 1.0993), ("centre", 0.8899)):
        r = MO.verifier(_barre(niveau=niveau, **S1))
        out.append(_du_moteur(G6, f"S1, charges {'sur la semelle haute' if niveau == 'haut' else 'au centre de cisaillement'}", r,
                              _donnees("IPE 300 (catalogue)", "S235", 6.0, "barre chargée (6.10)", "G 2,42 kN/m + 10 kN ; Q 15 kN à 3 m"),
                              brief=[("EC3, taux de déversement (non-régression)", att, _taux(r["methodes"]["ec3"], "E-Mb"), 0.0005)]))
    # S2 : des efforts de CALCUL donnés, M_y linéaire de 70 kN·m à 0 (ψ = 0) — ec3.verifier, comme tests/test_barre_benchmarks.py
    e = EC.verifier({"section": SE.du_catalogue("HEB 300"), "nuance": "S235", "axes": {"y": {"L": 5000.0}, "z": {"L": 5000.0}}, "NEd": 1400e3,
                     "MyEd": 70e6, "VzEd": 14e3, "flexion": {"y": {"diagramme": "lineaire", "psi": 0.0}}})
    out.append(cas(G6, "S2, poteau HEB 300", _donnees("HEB 300 (catalogue)", "S235", 5.0, "efforts de calcul (ELU) donnés", "N_Ed 1 400 kN ; M_y,Ed 70 kN·m (ψ = 0)",
                                                       None, None),
                   ec3=e, absent={"historique": "sans objet : SCIA ne donne que les efforts de calcul (aucun effort de service)"},
                   brief=[("EC3, (6.61) (non-régression, sans borne de C_mi,0)", 0.600, _taux(e, "E-661"), 0.001),
                          ("EC3, (6.62) (non-régression)", 0.648, _taux(e, "E-662"), 0.001)]))
    b = _barre(profil="IPE 360", nuance="S355", modele="continue", travees=[6.0, 6.0], poids_propre=False, niveau="haut",
               cas=[{"nom": "G", "nature": "permanent", "N": 0.0, "charges": [_q(6.0, 0.0, 12.0)]},
                    {"nom": "Q1", "nature": "exploitation_B", "N": 0.0, "charges": [_q(10.0, 0.0, 6.0)]},
                    {"nom": "Q2", "nature": "exploitation_B", "N": 0.0, "charges": [_q(10.0, 6.0, 12.0)]}])
    r = MO.verifier(b)
    out.append(_du_moteur(G6, "S3, continue IPE 360, 2 × 6 m", r, _donnees("IPE 360 (catalogue)", "S355", 12.0, "barre chargée (6.10)", "G 6 kN/m ; Q1, Q2 10 kN/m"),
                          brief=[("EC3, taux de déversement (non-régression)", 0.7266, _taux(r["methodes"]["ec3"], "E-Mb"), 0.0005)]))
    return out


def mesurer() -> List[Dict]:
    """Tous les cas, dans l'ordre du rapport, f_y complétées."""
    return [_completer_fy(c) for c in cas_classeurs() + cas_doc() + cas_deversement() + cas_r2_r3() + cas_prs() + cas_scia()]


# ------------------------------------------------------------------ les sept sujets (acier/barre/comparaison.py)
def lignes(c: Dict) -> List[Dict]:
    """Les sujets d'un cas : [{"sujet", "historique", "ec3", "commun" (cases de comparaison.case, ou None), "ecart"}] — la
    forme de comparaison.comparer, mais aussi quand une méthode manque (sa case est alors son statut NOMMÉ, c["absent"])."""
    out = []
    for nom, h, e, a in CP.SUJETS:
        li = {"sujet": nom}
        for cote, motifs, r in (("historique", h, c["historique"]), ("ec3", e, c["ec3"]), ("commun", a, c["ame"])):
            li[cote] = CP.case(r, motifs) if r is not None else None
        if all(li[k] is None for k in CP.COTES):
            continue
        th, te = ((li[k] or {}).get("taux") for k in ("historique", "ec3"))
        li["ecart"] = None if th is None or te is None else 100.0 * (te - th)
        out.append(li)
    return out


CAUSES = {
    "Flambement, axe faible": "courbes α, β du fascicule et s = 1,5 (ELS) contre χ du § 6.3.1 et γ_M1",
    "Flambement, axe fort": "courbes α, β du fascicule et s = 1,5 (ELS) contre χ du § 6.3.1 et γ_M1",
    "Flexion composée": "W_el et amplification 1/PE (ELS) contre W_pl et k_ij de l'annexe D ANB (EC3)",
    "Déversement": "formule des classeurs (λ_e,d, sans gauchissement) contre M_cr du solveur",
    "Cisaillement": "contrôle hérité H-EQ (σ de flambement et τ moyens, brief § 6.1) contre V_pl,Rd",
}


def phrase_ecart(c: Dict, li: Dict) -> Optional[str]:
    """Une phrase par écart de plus de SEUIL_ECART points : le sens, la cause (f_y nominale contre f_y selon l'épaisseur
    quand elles diffèrent)."""
    if li["ecart"] is None or abs(li["ecart"]) <= SEUIL_ECART:
        return None
    plus = "l'Eurocode 3" if li["ecart"] > 0 else "la méthode ELS"
    d = c["donnees"]
    fy = (f" ; f_y nominale {fr(d['fy_els'], 0)} contre {fr(d['fy_ec3'], 0)} N/mm² selon l'épaisseur"
          if d["fy_els"] and d["fy_ec3"] and abs(d["fy_els"] - d["fy_ec3"]) > 0.5 else "")
    cause = (c.get("causes") or {}).get(li["sujet"]) or CAUSES.get(li["sujet"], "méthodes différentes")
    return f"{c['nom']}, {li['sujet'].lower()} : {plus} est plus sévère de {fr(abs(li['ecart']), 1)} points ({cause}{fy})."


def raison(r: Dict) -> Optional[str]:
    """Pourquoi une méthode est bloquée ou non couverte : sa première alerte (le moteur la NOMME : « Appui A, appui B :
    décision impossible : s_s à saisir. ») ; à défaut, le premier message de la première vérification qui l'est."""
    v = next((v for v in r["verifications"] if v["statut"] == r["statut"] and v.get("messages")), None)
    m = next((a["message"] for a in r.get("alertes") or ()), None) or (v["messages"][0] if v else None)
    if m and m.startswith("Eurocode 3 : "):
        m = m[len("Eurocode 3 : "):]
    return None if not m else m if m.endswith(".") else m + "."


def _md(x) -> str:
    """Une cellule de tableau : la barre verticale échappée."""
    return str(x).replace("|", "\\|")


def _cellule(case: Optional[Dict], motif: Optional[str]) -> str:
    if case is None:
        return f"_{motif}_" if motif else "—"
    t = case["taux"]
    return f"{pc(t)} · {mot_statut(case['statut'], t)}" if t is not None else mot_statut(case["statut"])


def _gouverne(li: Dict) -> str:
    if li["historique"] is None and li["ec3"] is None and li["commun"] is not None:
        return li["commun"]["gouvernant"]
    g = [(li[k] or {}).get("gouvernant") for k in ("historique", "ec3")]
    return " / ".join(x or "—" for x in g)


def tableau_sujets(cs: List[Dict]) -> List[str]:
    out = ["| Cas | Sujet | ELS | Eurocode 3 | Écart (points) | Gouverne (ELS / EC3) |", "|---|---|---|---|---|---|"]
    for c in cs:
        if c["capacite"] is not None:
            continue
        for li in lignes(c):
            els = _cellule(li["historique"], c["absent"].get("historique") if c["historique"] is None else None)
            ec3 = _cellule(li["ec3"], c["absent"].get("ec3") if c["ec3"] is None else None)
            sujet = li["sujet"]
            if li["historique"] is None and li["ec3"] is None and li["commun"] is not None:
                els = ec3 = _cellule(li["commun"], None)
                sujet += " (EN 1993-1-5, commun)"
            out.append(f"| {c['nom']} | {sujet} | {els} | {ec3} | {'—' if li['ecart'] is None else fr(li['ecart'], 1)} | {_gouverne(li)} |")
    return out


def tableau_donnees(cs: List[Dict]) -> List[str]:
    out = ["| Cas | Profilé | Nuance | f_y ELS / EC3 | L (m) | Mode | Efforts |", "|---|---|---|---|---|---|---|"]
    for c in cs:
        d = c["donnees"]
        out.append(f"| {c['nom']} | {d['profil']} | {d['nuance']} | {fr(d['fy_els'], 0)} / {fr(d['fy_ec3'], 0)} | {fr(d['L'], 2)} | {d['mode']} | {d['efforts']} |")
    return out


def tableau_capacite(cs: List[Dict]) -> List[str]:
    out = ["| Cas | σ_adm,d (N/mm²) | Capacité ELS (kN·m) | × 1,5 (kN·m) | M_b,Rd EC3 (kN·m) | ELS × 1,5 / EC3 |", "|---|---|---|---|---|---|"]
    for c in cs:
        k = c["capacite"]
        if k is None:
            continue
        out.append(f"| {c['nom']} | {fr(k['sigma_adm_d'], 2)} | {fr(k['M_service'], 1)} | {fr(k['M_els_s'], 1)} | {fr(k['M_b_Rd'], 2)} | "
                   f"{fr(k['rapport'], 3)} |")
    return out


def tableau_brief(cs: List[Dict]) -> List[str]:
    out = ["| Cas | Valeur | Brief | Site | Écart |", "|---|---|---|---|---|"]
    for c in cs:
        for b in c["brief"]:
            e = None if b["obtenu"] is None else 100.0 * (b["obtenu"] / b["attendu"] - 1.0)
            out.append(f"| {c['nom']} | {b['quoi']} | {_nb(b['attendu'])} | {_nb(b['obtenu'])} | {'—' if e is None else fr(e, 2) + ' %'} |")
    return out


def _nb(x):
    if x is None:
        return "—"
    a = abs(x)
    return fr(x, 0 if a >= 10000 else 2 if a >= 10 else 4)


def hors_tolerance(cs: List[Dict]) -> List[str]:
    """Les valeurs du brief qui sortent de leur tolérance (règle 7) : vide quand tout est conforme."""
    return [f"{c['nom']} — {b['quoi']}" for c in cs for b in c["brief"]
            if b["obtenu"] is None or abs(b["obtenu"] / b["attendu"] - 1.0) > b["tol"]]


# ------------------------------------------------------------------ 2. écarts avec les classeurs (brief § 6.7)
CLASSEURS = (  # (cas, valeur du classeur, valeur corrigée, correction de 6.7 : la clé de historique_deversement.CORRECTIONS)
    ("FLAMBECBD (HEA 160)", "σ = 128,45 N/mm² (φ de l'axe fort)", "133,6 N/mm²", "phi_min"),
    ("FLAMBEBAD (IPE 270)", "σ = 43,20 N/mm² (φ de l'axe fort)", "115,7 N/mm²", "phi_min"),
    ("HEB 200, HEA 120, CHS 168,3, REALEX (u = 1)", "P_cr sans u", "identique à u = 1 ; × u⁻² sinon", "pcr_u"),
    ("DEVERSEMENT_CAD, feuille S355", "σ_adm,d = 97,8 N/mm² (157 × φ_d)", "147,50 N/mm²", "fy_nuance"),
    ("DEVERSEMENT_CAD, feuille S235 (U 140)", "classe b, 72,06 N/mm²", "classe c, 65,32 N/mm²", "classe_type"),
    ("HE 200 B, l = 0,30 m", "φ_d = 1,0073 (formule prolongée)", "φ_d = 1 ; 156,67 N/mm²", "phid_plafond"),
    ("I 600 × 300 × 50", "aucune classe (« 4 < t_f < 8 » toujours faux)", "classe c ; 119,03 N/mm²", "seuils_epaisseur"),
    ("I 300 × 300 × 80", "aucune classe (« t_f < 8 »)", "classe c ; 143,92 N/mm²", "seuils_epaisseur"),
    ("feuille « Calcul automatique », t_f > 80 mm", "seuils de l'Eurocode (100 mm)", "40 et 80 mm (tableau 12)", "seuils_eurocode"),
    ("toute poutre des classeurs Déversement", "lignes « axe fort » et « axe faible »", "« axe faible » seule (σ_adm 6 à 11 % plus bas)", "axe_faible"),
    ("contrôle hérité H11", "V_y + V_z signés", "|V_y| + |V_z|", "cisaillement_abs"),
    ("saisie des courbes", "aucune case : φ = 1 ; deux : α + β", "une courbe par axe (refus)", "une_courbe"),
)
VALEURS_CORRECTIONS = {"fy_nuance": {"fy": "355"}, "classe_type": {"classe": "c", "classe_lamine": "b"}, "phid_plafond": {"prolonge": "1,0073"}}
DONNEES_FAUSSES = (
    ("CHS 168,3 × 6", "A = 2 520 mm²", "3 059 mm² pour 6 mm", "le cas du classeur reste à 2 520 (σ 63,52) ; la lecture du catalogue donne 59,75"),
    ("HEA 120 Chancel", "section des ailes 960 mm² (une aile)", "1 920 mm²", "sans effet ici (V = 0)"),
    ("REALEX Ø610 × 12,5", "W = 3 506 cm³ (rayon moyen)", "I/(D/2) = 3 435 cm³", "les lectures avec poids propre prennent 3 434,6"),
    ("FLAMBECBD (HEA 160)", "valeurs d'axe faible dans les cases « fort »", "I_y 1 673 cm⁴, W_y 220,1 cm³", "le mélange bloque le verdict (test 5)"),
)


# ------------------------------------------------------------------ 3. qualification externe
def _mcr(L, s, moment, **k):
    return MC.moment_critique(L, E_ * s["Iz"], E_ * s["Iw"], k.pop("G", G_) * s["It"], moment, **k)["M_cr"] / 1e6


def qualification() -> List[Dict]:
    """[{cas, exterieur (valeur reçue ou None), site, attendu (la valeur que le site annonce), statut}] — L1 à L17, S1 à S3,
    ECCS 17 à 22, JRC."""
    out = []
    L = 6000.0
    ipe = {"Iz": 603.8e4, "It": 20.12e4, "Iw": 125.9e9}        # brief § 11.3 (les catalogues des fabricants)

    def Mq(x, LL=L):
        return x * (LL - x) / 2.0

    def Mp(x, LL=L):
        return (x if x <= LL / 2 else LL - x) / 2.0
    for n, quoi, f, att in (("L1", "moment constant", lambda: _mcr(L, ipe, lambda x: 1e6), 90.47),
                            ("L2", "répartie, centre", lambda: _mcr(L, ipe, Mq, reparties=[(0, L, 1.0, 0.0)]), 102.31),
                            ("L3", "répartie, semelle haute", lambda: _mcr(L, ipe, Mq, reparties=[(0, L, 1.0, 150.0)]), 78.83),
                            ("L4", "répartie, semelle basse", lambda: _mcr(L, ipe, Mq, reparties=[(0, L, 1.0, -150.0)]), 132.69),
                            ("L5", "centrale, centre", lambda: _mcr(L, ipe, Mp, ponctuelles=[(L / 2, 1.0, 0.0)]), 123.12),
                            ("L6", "centrale, semelle haute", lambda: _mcr(L, ipe, Mp, ponctuelles=[(L / 2, 1.0, 150.0)]), 89.19),
                            ("L7", "centrale, semelle basse", lambda: _mcr(L, ipe, Mp, ponctuelles=[(L / 2, 1.0, -150.0)]), 168.95)):
        out.append({"cas": f"{n} IPE 300, {quoi}", "exterieur": None, "site": f(), "attendu": att, "source": "LTBeam au bureau"})
    L10 = 10000.0
    A = PR.section_prs(300, 30, 540, 15, 300, 30, a=6, nuance="S355", bords="oxycoupes")
    B = PR.section_prs(300, 30, 540, 15, 200, 30, a=6, nuance="S355", bords="oxycoupes")
    zjB = PR.zj_haut(B)
    cot = PR.cotes(B)
    for n, quoi, f, att in (("L8", "PRS-A, moment constant", lambda: _mcr(L10, A, lambda x: 1e6), 1385.68),
                            ("L9", "PRS-A, répartie, centre", lambda: _mcr(L10, A, lambda x: Mq(x, L10), reparties=[(0, L10, 1.0, 0.0)]), 1567.12),
                            ("L10", "PRS-A, répartie, semelle haute", lambda: _mcr(L10, A, lambda x: Mq(x, L10), reparties=[(0, L10, 1.0, 300.0)]), 1191.42),
                            ("L11", "PRS-A, répartie, semelle basse", lambda: _mcr(L10, A, lambda x: Mq(x, L10), reparties=[(0, L10, 1.0, -300.0)]), 2059.63),
                            ("L12", "PRS-B, moment constant, grande semelle comprimée", lambda: _mcr(L10, B, lambda x: 1e6, zj=zjB), 1241.73),
                            ("L13", "PRS-B, moment constant, petite semelle comprimée", lambda: _mcr(L10, B, lambda x: -1e6, zj=zjB), 715.69),
                            ("L14", "PRS-B, répartie, centre de cisaillement", lambda: _mcr(L10, B, lambda x: Mq(x, L10), reparties=[(0, L10, 1.0, 0.0)], zj=zjB), 1234.80),
                            ("L15", "PRS-B, répartie, semelle haute", lambda: _mcr(L10, B, lambda x: Mq(x, L10), reparties=[(0, L10, 1.0, cot["face_haut"])], zj=zjB), 1086.25),
                            ("L16", "PRS-B, répartie, semelle basse", lambda: _mcr(L10, B, lambda x: Mq(x, L10), reparties=[(0, L10, 1.0, cot["face_bas"])], zj=zjB), 1803.30)):
        out.append({"cas": f"{n} {quoi}", "exterieur": None, "site": f(), "attendu": att, "source": "LTBeam au bureau"})
    s360 = SE.du_catalogue("IPE 360")
    q, MB = 23.1, -103.95e6

    def M17(x):
        return q * x * (L - x) / 2.0 + MB * x / L
    for n, zg, att, ext in (("L17a", 180.0, 223.30, 221.1), ("L17b", 0.0, 385.37, None), ("L17c", -180.0, 650.95, None)):
        v = _mcr(L, s360, M17, reparties=[(0.0, L, q, zg)], M_max=103.95e6, G=80769.0)
        out.append({"cas": f"{n} IPE 360, répartie + moment d'appui (z_g {fr(zg, 0)} mm, G 80 769)", "exterieur": ext, "site": v, "attendu": att,
                    "source": "LTBeamN, reçu le 16/11/2026" if ext else "LTBeam au bureau",
                    "lecture": "propriétés du catalogue du site (I_t 37,90 au lieu de 37,32 cm⁴) ; avec celles des fabricants, ligne suivante" if ext else None})
    fab = {"Iz": 1043e4, "It": 37.32e4, "Iw": 313.6e9}
    out.append({"cas": "L17a, propriétés des fabricants (I_t 37,32 ; I_w 313,6·10³)", "exterieur": 221.1,
                "site": _mcr(L, fab, M17, reparties=[(0.0, L, q, 180.0)], M_max=103.95e6, G=80769.0), "attendu": 221.49, "source": "LTBeamN, reçu le 16/11/2026"})
    return out


SCIA = (  # (cas, valeur de SCIA (reçue le 16/11/2026), valeur du site après le lot C — relue dans les cas du § 1, lecture)
    ("S1, semelle haute : taux de déversement", 1.08, "S1, charges sur la semelle haute", "E-Mb", "−1,7 % : M_cr approché de SCIA et son f (k_c tiré de C₁)"),
    ("S2 : (6.61)", 0.60, "S2, poteau HEB 300", "E-661", "borne de C_mi,0 retirée au lot B"),
    ("S2 : (6.62)", 0.65, "S2, poteau HEB 300", "E-662", ""),
    ("S3, travée Q1 : taux de déversement", 0.72, "S3, continue IPE 360, 2 × 6 m", "E-Mb", "−1 % : SCIA lit le moment aux nœuds de son maillage"),
)
ECCS = ((17, "exemple 1"), (18, "exemple 2"), (19, "exemple 3"), (20, "exemple 4"), (22, "ossature de bâtiment"))


def eccs21() -> List[Dict]:
    """ECCS N°119, benchmark 21 (exemple 5) : les quatorze coefficients que le brief § 11.4 en donne (diagramme de M_y
    déduit de C_my,0 : À VÉRIFIER sur la page, Q8)."""
    ref = {"μ_y": ("mu_y", 1.00), "w_y": ("w_y", 1.138), "μ_z": ("mu_z", 0.937), "w_z": ("w_z", 1.5), "C_my,0": ("C_my0", 0.999),
           "C_mz,0": ("C_mz0", 0.771), "a_LT": ("a_LT", 0.998), "b_LT": ("b_LT", 0.043), "c_LT": ("c_LT", 0.468), "d_LT": ("d_LT", 0.347),
           "e_LT": ("e_LT", 0.719), "C_yy": ("C_yy", 0.981)}
    r = EC.verifier({"section": SE.du_catalogue("IPE 500"), "nuance": "S235", "axes": {"y": {"L": 3750.0}, "z": {"L": 3750.0}}, "NEd": 500e3,
                     "MyEd": 200e6, "MzEd": 25e6, "flexion": {"y": {"diagramme": "ponctuel"}, "z": {"diagramme": "lineaire", "psi": 0.0}}})
    A, D = r["grandeurs"]["annexe_A"], r["grandeurs"]["deversement"]
    out = [{"quoi": k, "ref": v, "site": A[c]} for k, (c, v) in ref.items()]
    out += [{"quoi": "M_cr,0 (kN·m)", "ref": 899.0, "site": D["M_cr0"] / 1e6}, {"quoi": "λ̄_0", "ref": 0.757, "site": D["lambda_0"]}]
    return out


def regle7(e: float) -> str:
    """Brief, règle 7 : ≤ 1 % conforme ; 1 à 3 % à expliquer ; > 3 % à corriger ou à justifier."""
    a = abs(e)
    return "conforme" if a <= 1.0 else "à expliquer" if a <= 3.0 else "à corriger ou à justifier"


# ------------------------------------------------------------------ 4. listes restantes
def sans_accents(t: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", t) if unicodedata.category(ch) != "Mn").lower()


FICHIERS_CHERCHES = ("acier/barre", "api_web/profil.py", "api_web/profil_mots.py", "api_web/profil_prs.py", "api_web/profil_resultat.py",
                     "api_web/profil_saisie.py")


def a_verifier_du_code() -> Dict[str, int]:
    """La recherche de « à vérifier » (sans accents ni casse) dans le moteur des barres et l'écran du profil : {fichier: nombre}."""
    vus = {}
    for cible in FICHIERS_CHERCHES:
        chemin = os.path.join(RACINE, cible)
        fichiers = sorted(os.path.join(cible, f) for f in os.listdir(chemin) if f.endswith(".py")) if os.path.isdir(chemin) else [cible]
        for f in fichiers:
            n = sans_accents(open(os.path.join(RACINE, f), encoding="utf-8").read()).count("a verifier")
            if n:
                vus[f] = n
    return vus


# chaque fichier où le code dit « À VÉRIFIER » : ce qu'il y prend (la ligne de la liste qui le couvre) — un fichier neuf
# que la recherche trouve sans être ici fait rougir tests/test_barres_revue.py : la revue doit le lire et le classer
FICHIERS_A_VERIFIER = {
    "acier/barre/ame.py": "âme : borne s_s = 0, maintien de la semelle chargée, appui par l'âme, raidisseurs (B4, C1)",
    "acier/barre/combinaisons.py": "γ_G,sup ou γ_G,inf par source permanente (C2)",
    "acier/barre/ec3_ame.py": "règle des branches de m₂ ; borne haute des forces rapprochées (B4)",
    "acier/barre/ec3_classe.py": "ψ et α d'une semelle en console pris à leur borne prudente (B2)",
    "acier/barre/ec3_deversement.py": "M < 0 en efforts imposés : le plus petit M_cr des deux sens (intégration B)",
    "acier/barre/ec3_interaction.py": "décision n° 9 : console en N + M, aucune méthode fixée",
    "acier/barre/historique.py": "cornière u-u et v-v ; console L_d = 2,1·L ; colonne f_y ≤ 430 du tableau 12 (B1, C1b)",
    "acier/barre/historique_deversement.py": "correction 4 dite en S355 seule ; C_d borné à [1 ; 3] ; colonne f_y > 430 ; ligne « axe fort » (B1)",
    "acier/barre/mcr.py": "colonne C₃ du tableau E.1 ANB, relue sur un texte extrait abîmé",
    "acier/barre/moteur.py": "console 2,1·L, débords, sens de M, semelle comprimée, « le cisaillement gouverne » (intégration B, C1)",
    "acier/barre/nuances.py": "valeurs M/ML et norme produit des plats d'un PRS en S420 et S460 (C1b)",
    "acier/barre/parametres.py": "décision n° 9 sans variante calculable",
    "acier/barre/raidisseurs.py": "gorge d'un PRS non saisie : congé pris à t_w (B4)",
    "acier/barre/statique.py": "« sections à vérifier » (le mot, pas une hypothèse) ; poids propre vertical, N par tronçon, débord (C1)",
    "api_web/profil.py": "« à vérifier (v) », le niveau de couleur (le mot, pas une hypothèse)",
    "api_web/profil_mots.py": "angle de 45° d'une barre inclinée (C3) ; le niveau « v » (le mot)",
    "api_web/profil_prs.py": "la page Catalogue ne lit pas la bibliothèque des PRS (C6)",
    "api_web/profil_resultat.py": "le mot « À vérifier » de l'écran (pas une hypothèse)",
}


def table_parametres() -> List[str]:
    """Le § 5.8 de docs/acier/BARRES_ACIER.md, RECOPIÉ de acier/barre/parametres.py (copie NOMMÉE : tests/test_barres_revue.py
    la lie à la table ; « --parametres » la réécrit)."""
    out = ["| Clé | Grandeur | Valeur | Statut | Source |", "|---|---|---|---|---|"]
    statut = {PA.POINTE: "pointé", PA.A_CONFIRMER: "**à confirmer**", PA.DECISION: "**décision n° {point}**", PA.A_POINTER: "à pointer"}
    for k, p in PA.PARAMETRES.items():
        v = p["valeur"]
        val = " ; ".join(_valeur_doc(x) for x in v) if isinstance(v, tuple) else _valeur_doc(v)
        st = statut[p["statut"]].format(point=p.get("point"))
        if p["statut"] == PA.DECISION:
            st += f" (variante : {_valeur_doc(p['variante'])})"
        out.append(f"| `{k}` | {p['nom']} | {val} | {st} | {p['source']} |")
    return out


def _valeur_doc(v) -> str:
    if v is None:
        return "aucune"
    if isinstance(v, bool):
        return "oui" if v else "non"
    if isinstance(v, (int, float)):
        s = RE.fr(v, 3)
        return s.rstrip("0").rstrip(",") if "," in s else s
    return f"« {v} »"


DEBUT_58, FIN_58 = "<!-- parametres.py : debut (tools/comparaison_barres.py --parametres) -->", "<!-- parametres.py : fin -->"


def bloc_58(texte: str) -> Optional[List[str]]:
    """Les lignes du § 5.8 entre ses deux repères ; None s'ils manquent."""
    if DEBUT_58 not in texte or FIN_58 not in texte:
        return None
    return [li for li in texte.split(DEBUT_58, 1)[1].split(FIN_58, 1)[0].strip().splitlines()]


def reecrire_58():
    t = open(DOC, encoding="utf-8").read()
    avant, reste = t.split(DEBUT_58, 1)
    apres = reste.split(FIN_58, 1)[1]
    open(DOC, "w", encoding="utf-8").write(avant + DEBUT_58 + "\n" + "\n".join(table_parametres()) + "\n" + FIN_58 + apres)


def gamma_vivant() -> List[Dict]:
    """Le seul « à confirmer » (npl_gamma) GARDÉ VIVANT : un cas γ_M0 = 1,00 et γ_M1 = 1,10 où il joue (IPE 500, N + M) —
    ce que la note imprime alors dans ses hypothèses."""
    r = EC.verifier({"section": SE.du_catalogue("IPE 500"), "nuance": "S235", "axes": {"y": {"L": 3750.0}, "z": {"L": 3750.0}}, "NEd": 500e3,
                     "MyEd": 200e6, "MzEd": 25e6, "flexion": {"y": {"diagramme": "ponctuel"}, "z": {"diagramme": "lineaire", "psi": 0.0}},
                     "parametres": {"gamma_M1": 1.10}})
    return r["grandeurs"].get("a_confirmer") or []


# ------------------------------------------------------------------ le rapport
def rapport(cs: List[Dict], date: str) -> str:
    L = []
    w = L.append
    nc = hors_tolerance(cs)
    w(f"# Barres acier — la revue finale du lot C : les deux méthodes sur tous les cas\n")
    w(f"*Mesure datée du {date}. Elle ne se réécrit pas : une correction s'y porte en révision datée, dans un commit à part. "
      f"Produite par `tools/comparaison_barres.py` (rejoué par `tests/test_barres_revue.py`, qui compare au brief, jamais à ce "
      f"fichier).*\n")
    w(f"**{DONNEE}**\n")
    w("## 0. En bref\n")
    w("- **Défaut non conservatif trouvé par la revue, corrigé à l'intégration du lot C** : une barre à débord gauche ne pouvait "
      "pas être chargée jusqu'au bout de sa travée (`acier/barre/entree.py`, `normaliser` bornait les charges, les maintiens et "
      "les points à Σ portées, sans les débords ; l'écran mesurait la barre de même). La barre va maintenant de 0 à "
      "g + Σ portées + d (`statique.longueur`) ; contrôles et mutations : `tests/test_barre_moteur.py`, `tests/test_api_web_profil.py`.")
    w(f"- {len(cs)} cas, {sum(len(c['brief']) for c in cs) + len(R4_LIGNES)} valeurs du brief : "
      + ("**toutes à 1 % au plus** (règle 7)." if not nc else f"**{len(nc)} hors tolérance** : " + " ; ".join(nc) + "."))
    w("- Chaque cas a ses deux méthodes, ou un statut nommé (non couvert, sans objet, fonction seule) ; jamais une case vide.")
    w(f"- Écarts de plus de {fr(SEUIL_ECART, 0)} points entre les méthodes : {sum(1 for c in cs for li in lignes(c) if phrase_ecart(c, li))}, "
      "chacun dit par une phrase (§ 1).")
    w("- Qualification externe : L17a reçu, conforme avec les propriétés des fabricants (+0,2 %) ; S1 à S3 reçus ; L1 à L16, "
      "L17b, L17c, ECCS 17 à 20 et 22, JRC attendus (§ 3).")
    w("- À confirmer : `npl_gamma` seul ; sept décisions du bureau dans la table, une dans sa source du site (§ 4).\n")
    w("## 1. Les deux méthodes, cas par cas\n")
    w("Statuts du brief § 9 : vérifié, à la limite (95 % à 100 %, le seuil des notes du site), non vérifié, hors domaine, "
      "non couvert, bloqué. Écart = taux EC3 − taux ELS, en points de %.\n")
    for g in GROUPES:
        cg = [c for c in cs if c["groupe"] == g]
        w(f"### {g}\n")
        w("\n".join(tableau_donnees(cg)) + "\n")
        if any(c["capacite"] is None for c in cg):
            w("\n".join(tableau_sujets(cg)) + "\n")
        if any(c["capacite"] is not None for c in cg):
            w("\n".join(tableau_capacite(cg)) + "\n")
        ph = [p for c in cg for li in lignes(c) for p in [phrase_ecart(c, li)] if p]
        ph += [f"{c['nom']} : {c['note']}" for c in cg if c["note"]]
        ph += [f"{c['nom']} : {NOMS[k]} — {m}." for c in cg for k, m in sorted(c["absent"].items()) if c[k] is None]
        ph += [f"{c['nom']} : {NOMS.get(k, 'âme')} {mot_statut(r['statut'])} — {raison(r)}" for c in cg
               for k, r in (("historique", c["historique"]), ("ec3", c["ec3"]), ("ame", c["ame"]))
               if r is not None and r["statut"] in (RE.BLOQUE, RE.NON_COUVERT) and raison(r)]
        if ph:
            w("\n".join(f"- {p}" for p in ph) + "\n")
        w("Valeurs du brief :\n")
        w("\n".join(tableau_brief(cg)) + "\n")
        if g == G4:
            w("R4, l'âme de l'IPE 300 S235 (EN 1993-1-5, commune aux deux méthodes ; la méthode ELS ne traite pas l'âme, brief § 6.6) :\n")
            w("| Valeur | Brief | Site | Écart |\n|---|---|---|---|")
            w("\n".join(f"| {b['quoi']} | {_nb(b['attendu'])} | {_nb(b['obtenu'])} | {fr(100 * (b['obtenu'] / b['attendu'] - 1), 2)} % |" for b in R4_LIGNES) + "\n")
    w("## 2. Écarts avec les classeurs du bureau (brief § 6.7)\n")
    w("Chaque correction est nommée par le moteur (`historique_deversement.CORRECTIONS`) et dite dans la note quand elle joue.\n")
    w("| Cas | Classeur | Module | Correction |\n|---|---|---|---|")
    w("\n".join(f"| {_md(a)} | {_md(b)} | {_md(c)} | {_md(HD.texte_correction(k, **VALEURS_CORRECTIONS.get(k, {})))} |" for a, b, c, k in CLASSEURS) + "\n")
    w("Données fausses des exemples, dites et NON corrigées (ce ne sont pas des erreurs de formule ; les tests de 11.1 les gardent) :\n")
    w("| Cas | Donnée du classeur | Valeur juste | Effet |\n|---|---|---|---|")
    w("\n".join(f"| {a} | {b} | {c} | {d} |" for a, b, c, d in DONNEES_FAUSSES) + "\n")
    w("Les choix du bureau ne sont pas des corrections : f_y nominale, courbes et classes de chaque classeur, s = 1,5, C = 1, "
      "C_d = 1, charges de service.\n")
    w("## 3. Qualification externe (brief § 11.4, règle 7)\n")
    w("| Cas | Extérieur | Site | Attendu | Écart | Lecture |\n|---|---|---|---|---|---|")
    for q in QUALIF:
        if q["exterieur"] is None:
            w(f"| {q['cas']} | attendu — {q['source']} | {fr(q['site'], 2)} | {fr(q['attendu'], 2)} | — | attendu |")
        else:
            e = 100.0 * (q["site"] / q["exterieur"] - 1.0)
            w(f"| {q['cas']} | {fr(q['exterieur'], 2)} — {q['source']} | {fr(q['site'], 2)} | {fr(q['attendu'], 2)} | {fr(e, 2)} % | {regle7(e)}, reçu{' : ' + q['lecture'] if q.get('lecture') else ''} |")
    for quoi, ext, nom, ident, lecture in SCIA:
        c = next(c for c in cs if c["nom"] == nom)
        t = _taux(c["ec3"], ident)
        e = 100.0 * (t / ext - 1.0)
        w(f"| SCIA {quoi} | {fr(ext, 2)} — SCIA 26, reçu le 16/11/2026 | {fr(t, 4)} | — | {fr(e, 1)} % | {regle7(e)}{' : ' + lecture if lecture else ''} |")
    for n, quoi in ECCS:
        w(f"| ECCS {n} ({quoi}) | bloqué — données à fournir | — | — | — | attendu (Q8) |")
    w("| JRC EUR 22898 (cisaillement, charges transversales, raidisseurs) | bloqué — données à fournir | — | — | — | attendu (Q14) |\n")
    w("ECCS 21 (exemple 5 ; diagramme de M_y déduit de C_my,0, À VÉRIFIER sur la page) :\n")
    w("| Coefficient | Référence | Site | Écart | Règle 7 |\n|---|---|---|---|---|")
    for x in ECCS21:
        e = 100.0 * (x["site"] / x["ref"] - 1.0)
        w(f"| {x['quoi']} | {fr(x['ref'], 3)} | {fr(x['site'], 4)} | {fr(e, 2)} % | {regle7(e)} |")
    w("\nc_LT, d_LT, e_LT ∝ 1/χ_LT : ils impliquent χ_LT ≈ 0,826 dans la référence, le module prend 0,811 ; aucune lecture essayée "
      "ne retrouve 0,826 (`tests/test_barre_benchmarks.py`). Les taux de la référence ne sont pas lus.\n")
    w("## 4. Ce qui reste ouvert\n")
    w("### 4.1 À CONFIRMER (`parametres.a_confirmer()`)\n")
    w("\n".join(f"- `{p['cle']}` — {p['nom']} : {p['valeur']} ({p['source']})." for p in PA.a_confirmer()))
    w(f"- Gardé vivant : IPE 500, N + M, γ_M0 = 1,00 et γ_M1 = 1,10 — le moteur le rend dans ses hypothèses "
      f"({', '.join(p['cle'] for p in GAMMA) or 'RIEN'}).\n")
    w("### 4.2 DÉCISION BUREAU (brief § 13.2)\n")
    w("| n° | Décision | Défaut | Variante | Où |\n|---|---|---|---|---|")
    for d in PA.decisions():
        w(f"| {d['point']} | {d['nom']} | {_valeur_doc(d['valeur'])} | {_valeur_doc(d['variante'])} | `parametres.{d['cle']}` |")
    w(f"| 5 | Méthode ELS : les classeurs du bureau, tels quels | tranché le 16/11/2026 (une seule valeur) | — | brief § 13.2 point 5 |")
    w(f"| 6 | Tube REALEX : fabrication et position | ouvert (ELS 0,831 à 1,018 ; EC3 non couvert) | — | hors code (une donnée) |")
    w(f"| 7 | Seuil « à la limite » | {fr(100 * ME.SEUIL_ATTENTION, 0)} % | — | `acier/garde_corps/membres.SEUIL_ATTENTION` (le seuil du site) |")
    w("| 8 | Mise en ligne après le lot C ou après le lot 8 ; lot 9 | après le lot C (défaut) | — | hors code |\n")
    w("Aucun défaut de décision n'est écrit en dur hors de la table (`tests/test_barres_revue.py`).\n")
    w("### 4.3 À VÉRIFIER\n")
    w("Les hypothèses prudentes des rapports des lots B et C, une par ligne : hypothèse, endroit, effet sur un résultat, "
      "réponse proposée.\n")
    w("| Hypothèse | Endroit | Effet | Réponse proposée |\n|---|---|---|---|")
    w("\n".join(f"| {a} | `{b}` | {c} | {d} |" for a, b, c, d in A_VERIFIER) + "\n")
    vus = a_verifier_du_code()
    w(f"Recherche de « à vérifier » (sans accents ni casse) dans le moteur et l'écran du profil : {sum(vus.values())} mentions "
      f"dans {len(vus)} fichiers, chacun classé dans la liste de `tools/comparaison_barres.py` (un fichier neuf fait rougir le test) :\n")
    w("\n".join(f"- `{f}` ({n}) : {FICHIERS_A_VERIFIER.get(f, 'NON CLASSÉ')}" for f, n in sorted(vus.items())) + "\n")
    w("### 4.4 Questions au bureau (`docs/acier/BARRES_ACIER.md` § 10)\n")
    for titre, qs in QUESTIONS:
        w(f"**{titre}** : " + " ; ".join(qs) + ".\n")
    w("### 4.5 Hors du module (nommés, non traités)\n")
    w("\n".join(f"- {x}" for x in HORS_MODULE) + "\n")
    w("## 5. Avant / après : ce qui change à la réouverture d'une note enregistrée\n")
    w("Une note se relit sans être réécrite (format VERSION 2, clés neuves avec leur défaut, aucune migration). À sa "
      "réouverture, le site recalcule avec le moteur du jour :\n")
    w("| Changement | Lot | Effet mesuré | Preuve |\n|---|---|---|---|")
    w("\n".join(f"| {a} | {b} | {c} | `{d}` |" for a, b, c, d in AVANT_APRES) + "\n")
    return "\n".join(L).rstrip() + "\n"


R4_LIGNES: List[Dict] = []
QUALIF: List[Dict] = []
ECCS21: List[Dict] = []
GAMMA: List[Dict] = []

A_VERIFIER = (  # (hypothèse prudente, endroit, effet, réponse proposée) — tirées des rapports des blocs B1 à B5, C0 à C7 et des intégrations
    ("R2 sous la charge sans maintien : « décision impossible » (EN 1993-1-5 § 6.1(1)), le brief disait « pas de raidisseur »", "acier/barre/ame.py",
     "âme « bloqué » au lieu d'OK (43 % si la semelle est tenue)", "garder la règle (côté sûr) et corriger le brief"),
    ("s_s et c non saisis : calcul à la borne 0 ; « décision impossible » si la borne ne passe pas", "acier/barre/ame.py",
     "barre neuve « bloqué » (157 % à la borne, 22 % à s_s = 100 mm)", "garder ; ou pré-remplir s_s (Q11)"),
    ("Console : L_cr = 2,1·L sur l'axe à bout libre, maintiens intermédiaires non comptés", "acier/barre/moteur.py",
     "6,30 m au lieu de 3,77 m (console de 3 m tenue à mi-longueur)", "garder ; l'ingénieur saisit sa longueur"),
    ("Débord à bout libre : L_cr par Euler sur la barre entière, au moins 2,1 × le débord ; L_d du débord = L_cr,z, ELS hors domaine",
     "acier/barre/statique.py", "7,895 m pour 4 + 2,5 m", "garder"),
    ("Formule des classeurs pour les poutres hautes peu maintenues", "acier/barre/historique.py",
     "IPE 750 de 8 m : 171 % ELS contre 83 % EC3 ; PRS-C : 18 maintiens contre 8", "étendre le message « formule prudente » aux laminés hauts ?"),
    ("Appui « par l'âme », âme élancée : raidisseur d'appui sous la réaction entière", "acier/barre/ame.py",
     "un raidisseur demandé (§ 5.1(2))", "garder"),
    ("γ_G,sup ou γ_G,inf par cas permanent (chaque cas une source, sauf deux de même répartition)", "acier/barre/combinaisons.py",
     "plus de combinaisons, bilans plus longs ; jamais moins sévère", "garder"),
    ("Action de courte durée : seulement si TOUS ses cas sont déclarés", "acier/barre/combinaisons.py", "ψ₀ = 0,3 plus rare", "garder"),
    ("Poids propre d'une barre verticale : q·L entier dans N, en compression", "acier/barre/statique.py", "N plus fort au pied", "garder"),
    ("N par tronçon : flambement et (6.61)/(6.62) au plus grand N de la barre", "acier/barre/statique.py", "côté sûr", "garder"),
    ("PRS-C : béton frais en nature « exécution » (γ_Q 1,5, ψ₀ 1,0)", "tests/test_note_profil.py", "M_Ed 4 199,3 (3 893,5 en permanent)",
     "confirmer avec la note du bureau"),
    ("Verdict ELS quand le contrôle hérité H-EQ dépasse la principale : la principale en tête, « + contrainte équivalente » dessous",
     "api_web/profil_resultat.py", "pastille 111 %, H-EQ 113 % écrit dessous", "garder ; H-EQ ne décide jamais seul (brief § 6.1)"),
    ("Norme produit des plats d'un PRS en S420/S460 : ligne « laminé » de la nuance", "acier/barre/nuances.py", "N/NL par défaut", "garder"),
    ("Cornière : u-u et v-v prennent la plus grande longueur saisie", "acier/barre/historique.py", "v-v gouverne", "garder"),
    ("Bibliothèque des PRS dans le navigateur ; la page Catalogue ne la lit pas", "api_web/profil_prs.py", "deux bibliothèques",
     "trancher (question de C6)"),
    ("z_j et ψ_f du tableau du PRS pour la semelle haute comprimée", "api_web/profil_prs.py", "le moteur, lui, les recalcule par tronçon",
     "ajouter le cas inverse ?"),
    ("Inclinée sans angle saisi : α = 45° écrit à l'écran", "api_web/profil_mots.py", "poids propre réparti à 45°", "garder (champ visible)"),
    ("Charge dont la valeur de tableau manque (classe à fournir, ψ₀ d'exécution) : calcul « bloqué »", "acier/barre/combinaisons.py",
     "aucun verdict deviné", "garder"),
    ("Décision n° 9 : console en N + M « non couvert », aucune variante calculable", "acier/barre/parametres.py",
     "E-661 et E-662 non couverts", "le bureau fixe une méthode"),
    ("NBN B 51-002, tableau 12 : S460 lu dans la colonne f_y ≤ 430", "acier/barre/historique_deversement.py", "classe du S460", "lire la colonne"),
    ("Charges à hauteurs mêlées : M_cr par tronçon, k_c = 1 hors tableau 6.6", "acier/barre/ec3_deversement.py", "S1 1,099 (SCIA 1,08)", "garder"),
    ("M < 0 en efforts imposés : le sens de la charge est inconnu, le plus petit M_cr des deux sens", "acier/barre/ec3_deversement.py",
     "IPE 300, 6 m, −60 kN·m : 0,705 → 1,023", "garder ; une ligne « À VÉRIFIER » dans la note"),
    ("Type (c) à la borne : branche m₂ = 0", "acier/barre/ec3_ame.py", "jusqu'à −24 % sur des profilés hauts en S355", "garder"),
    ("Semelle en console : ψ et α à leur borne prudente (classe 3 à 13,77·ε sous M_z)", "acier/barre/ec3_classe.py",
     "six laminés en classe 4 sous M_z, taux inchangés", "garder"),
    ("Correction 4 (σ_adm avec la f_y de la nuance) dite pour S355 seulement ; C_d borné à [1 ; 3]", "acier/barre/historique_deversement.py",
     "texte de la note", "garder"),
    ("Tableau E.1 ANB : colonne C₃ (k_z = 1) relue sur un texte extrait abîmé", "acier/barre/mcr.py", "C₃ des moments d'extrémité",
     "relire sur l'image du tableau"),
    ("Gorge d'un PRS non saisie : congé du raidisseur pris à t_w", "acier/barre/raidisseurs.py", "chanfrein du raidisseur", "garder"),
    ("I soudé non PRS : la clé prs_soudee_equivalente vaut pour tout I soudé (§ 6.3.2.2 par défaut)", "acier/barre/parametres.py",
     "I soudé h/b = 3, 6 m : M_b,Rd −1,8 à −5,4 %", "garder"),
)
QUESTIONS = (
    ("Tranchées par le brief v12", ["Q1 méthode 1 et borne de C_mi,0", "Q2 ψ belges", "Q3 f_u et A_v,y", "Q4 méthode ELS des classeurs",
                                     "Q7 fichiers hors du dépôt", "Q13 raidisseurs proposés", "Q9 et Q11 caduques (impression au choix ; s_s à la borne)"]),
    ("Ouvertes", ["Q5 = décision n° 6 (REALEX)", "Q6 = décision n° 7 (seuil 95 %)", "Q8 pages SCIA 17 à 22", "Q10 η₁ (décision n° 1)",
                  "Q12 appuis posés ou par l'âme", "Q14 exemples JRC"]),
    ("Nouvelles des lots B et C", ["R2 sans maintien sous la charge (brief à corriger ?)", "barre neuve : pré-remplir s_s ?",
                                   "message « formule prudente » pour les laminés hauts ?", "Catalogue et bibliothèque des PRS : une seule ?",
                                   "béton frais du PRS-C en exécution ?", "écran Streamlit et I_t des PRS (+6,2 %) : le retirer ?"]),
)
HORS_MODULE = (
    "γ_Mu = 1,1 des attaches (assemblages) : à pointer sur l'ANB de l'EN 1993-1-8.",
    "α_cc = 0,85 (béton) : la valeur belge, à pointer.",
    "Garde-corps, cornière, enrobage : leurs suites sont aux chantiers de leurs modules.",
    "RBAR : le moteur de plans, hors des barres.",
    "Ancien écran Streamlit (`modules/tableau_profiles.py`) : I_t d'un PRS = Σ b·t³/3 (+6,2 %, non conservatif) ; à retirer ou à porter sur `prs.py`.",
)
AVANT_APRES = (  # (changement, lot, effet mesuré, preuve)
    ("f_y nominale en méthode ELS", "B1", "HEB 300 S235 : 225 → 235 N/mm²", "tests/test_barre_moteur.py"),
    ("Déversement ELS des classeurs (au lieu du fascicule)", "B1", "barre neuve 74 → 88 % ; IPE 750x134 8 m 93 → 171 %", "tests/test_barre_historique.py"),
    ("C_mi,0 sans borne inférieure", "B3", "S2 : 0,631 / 0,667 → 0,600 / 0,648", "tests/test_barre_benchmarks.py"),
    ("Console : 2,1·L, bout libre non compté en ELS", "intégration B", "console IPE 300 3 m : 94,9 % ok → 185 % hors domaine (ELS)", "tests/test_barre_moteur.py"),
    ("Statut « bloqué » (PRS sans bords, classe à fournir, s_s à saisir, ψ₀ à fournir)", "C0, C2", "le verdict attend la donnée, le taux reste montré",
     "tests/test_barre_moteur.py"),
    ("Les deux méthodes toujours calculées ; l'impression au choix", "C0, C5", "une note « ec3 » seule montre aussi l'ELS à l'écran",
     "tests/test_note_profil.py"),
    ("γ_G,sup ou γ_G,inf par cas permanent", "C2", "bilans plus longs, aucun verdict changé sur les photos", "tests/test_barre_combinaisons.py"),
    ("Poids propre en efforts imposés (deux interrupteurs, éteints par défaut)", "C1", "aucun pour une note ancienne", "tests/test_barre_moteur.py"),
    ("s_s et c non saisis pris à leur borne 0", "B4", "barre neuve : « s_s à saisir », 157 % à la borne", "tests/test_barre_ame.py"),
    ("Longueurs copiées au passage en efforts imposés remises en automatique", "C0",
     "une note ancienne (sans « longueurs_copiees ») garde ses longueurs, aucun changement ; console neuve : L_z 3 → 6,3 m",
     "tests/test_api_web_profil.py"),
    ("Tolérance de 5 % retirée", "C0", "une note qui la cochait le dit, le taux n'est plus toléré", "tests/test_note_profil.py"),
    ("Format de la note", "B4, intégration B", "VERSION 2 ; clé zero_saisi (0 saisi ≠ non saisi) ; aucune migration", "tests/test_api_web_profil.py"),
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sortie", help="le fichier à écrire (défaut : docs/validation/barres_revue_finale_<date>.md)")
    p.add_argument("--date", default=datetime.date.today().isoformat())
    p.add_argument("--parametres", action="store_true", help="réécrire le § 5.8 de docs/acier/BARRES_ACIER.md depuis parametres.py")
    a = p.parse_args(argv)
    if a.parametres:
        reecrire_58()
        print("§ 5.8 réécrit depuis acier/barre/parametres.py")
        return 0
    cs = preparer()
    sortie = a.sortie or os.path.join(DOSSIER, f"barres_revue_finale_{a.date}.md")
    if os.path.exists(sortie) and not a.sortie:
        print(f"{sortie} existe : une mesure datée ne se réécrit pas (--sortie pour écrire ailleurs)")
        return 1
    j, m, an = a.date[8:10], a.date[5:7], a.date[:4]
    open(sortie, "w", encoding="utf-8").write(rapport(cs, f"{j}/{m}/{an}"))
    print(f"écrit : {sortie} ({len(cs)} cas ; hors tolérance : {len(hors_tolerance(cs))})")
    return 0


def preparer() -> List[Dict]:
    """Tout ce que le rapport lit : les cas, R4, la qualification, ECCS 21, le « à confirmer » gardé vivant."""
    R4_LIGNES[:] = r4()
    QUALIF[:] = qualification()
    ECCS21[:] = eccs21()
    GAMMA[:] = gamma_vivant()
    return mesurer()


if __name__ == "__main__":
    sys.exit(main())

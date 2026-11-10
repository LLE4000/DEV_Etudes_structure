# -*- coding: utf-8 -*-
"""Cornière d'appui — le MOTEUR refondu sur la méthode du classeur de l'ingénieur (feuille ``P8``), décisions du
10/11/2026 (``docs/acier/CORNIERE_DECISIONS.md``) ; ce qui est livré et la preuve : ``docs/acier/CORNIERE_MOTEUR.md``.

Deux configurations (A : aile horizontale en bas, celle du classeur ; B : en haut, retournée), deux modes pour A
(« historique » : P8 mot pour mot, chaque grandeur avec la cellule du classeur ; « équilibre » : un corps libre, pivot
y_c et ligne d'action u_V, N et V exacts) — B se calcule toujours en équilibre. Les jeux par l'ENVELOPPE : la charge à
X − dx, X, X + dx, l'ancrage à Y − dy, Y, Y + dy (neuf combinaisons, une seule quand dx = dy = 0) ; chaque grandeur
vérifiée garde la plus défavorable avec la combinaison qui la gouverne. La statique d'une combinaison : ``statique.py``.

Bibliothèque standard (math) ; le catalogue des cornières (``acier.profiles``) charge numpy, présent dans le
navigateur. Unités : mm, kN, kN·m (moments rendus), N/mm². Nombres non arrondis ; sortie JSON-sérialisable.

API
---
    nouvelle_corniere(nom="Cornière 1") -> dict      les défauts du § 2 (L 120×120×15, A, historique, 35 kN, 75 ± 20, 85 ± 10…)
    normaliser(corniere) -> dict                     copie complétée et contrôlée ; ValueError hors domaine (entree.normaliser)
    calculer(corniere, diagrammes=True, annexes=True) -> dict   le calcul complet (ci-dessous) ; sans les tableaux des
                                                     diagrammes ni la comparaison / la garde quand on les coupe (le sélecteur)
    verifier(corniere) -> dict                       = calculer(corniere) : tout
    combinaisons(corniere) -> list                   les 1, 3 ou 9 combinaisons de jeux, la nominale d'abord
    candidats(corniere, familles=None) -> list       le sélecteur : les cornières du catalogue triées par masse, chacune
                                                     calculée avec les charges de ``corniere``
    optimal(corniere, familles=None) -> dict | None  la plus légère qui tient ET dont la charge et l'ancrage sont dans le domaine
    cornieres(), section_catalogue(nom), designation(nom), nom_catalogue(texte), section(c), est_v1(c), depuis_v1(c)
                                                     (entree)

ENTRÉE — un dict JSON ; une clé absente ou ``null`` prend le défaut (a, R_ELU, uV : vide = auto) ; mm, kN, N/mm² :
------
    {"nom": "Cornière 1",
     "standard": true, "profil": "L 120x120x15",   catalogue (acier/profiles, familles L et LI) …
     "A": 120, "B": 120, "t": 15, "r": 13,          … ou saisie libre (ailes, épaisseur, congé)
     "aile_horizontale": "A",                       laquelle des ailes porte la charge (A = la première du nom, h au catalogue)
     "configuration": "A",                          A = aile horizontale en bas (classeur), B = en haut (retournée)
     "etat_limite": "ELS",                          ELS : σ_lim = f_y/coef, τ_lim = σ_lim/√3, R est la réaction ELS ;
                                                    "ELU" : R est la réaction ELU, σ_lim = f_y/γ_M0, τ_lim = f_y/(√3·γ_M0), γ_M0 = 1,0
                                                    (coef et R_ELU sans effet) — décisions § 9.5
     "fy": 355, "coef": 1.5,                        nuance du site (235, 275, 355, 460) ; coef : le coefficient de l'ELS
     "R": 35, "L": 300, "X": 75, "dx": 10,          réaction (kN, dans l'état limite choisi) sur la longueur reprise L ; charge à X ± dx
     "a": null, "R_ELU": null,                      largeur d'appui réelle (vide = L) ; ELS : réaction ELU pour les ancrages (vide = 1,5·R)
     "Y": 85, "dy": 5, "n": 3, "d": 16,             ancrage à Y ± dy de la face extérieure de l'aile horizontale ; n ancrages Ø d
     "entraxe": null,                               entraxe s des ancrages, centrés sur L (vide = L/n) ; EN 1993-1-8 Tab. 3.3 : s ≥ 2,2·d0, e₁ ≥ 1,2·d0
     "type_ancrage": "mecanique",                   ou "chimique", "traversant" (dessin et note ; chevilles non vérifiées)
     "profondeur": 100, "rondelles": 1,             h_ef (ou l'épaisseur traversée) ; 0, 1 (ISO 7089) ou 2 (large ISO 7093-1 + ISO 7089)
     "support": "voile",                            ou "poutre" : support_haut / support_bas = l'étendue de la poutre au-dessus / au-dessous
     "support_haut": 150, "support_bas": 150,       de la face extérieure de l'aile horizontale (mm, vers le haut / vers le bas) — § 9.3
     "mode": "historique",                          ou "equilibre" (sans effet en B : toujours équilibre)
     "jeu_ancrage": 10,                             mode historique : le C76 du classeur, bras de la traction Y − jeu_ancrage
     "yc": 0, "uV": null,                           mode équilibre : pivot (A : depuis le dessous ; B : depuis le bout libre),
                                                    ligne d'action du cisaillement d'ancrage depuis le nu du mur (vide = e/2)
     "v1": false}                                   vrai : relue d'une note du module d'avant (modèle lame) → avertissement note_v1

SORTIE de calculer / verifier
------
    {"reference": …, "corniere": {…} (normalisée),
     "section": {"source": "catalogue"|"saisie", "nom", "designation", "famille", "A", "B", "t", "r1", "r2", "masse" (kg/m),
                 "aire" (cm²), "masse_source": "catalogue"|"calculée", "norme", "aile_horizontale", "larg", "haut", "e", "r"},
     "configuration": "A"|"B", "mode": "historique"|"equilibre" (le mode EFFECTIF), "mode_demande",
     "materiau": {"fy", "nuance", "coef", "etat_limite", "gamma_M0", "sigma_adm", "tau_adm" (les LIMITES de l'état limite :
                  ELS f_y/coef et σ_adm/√3 ; ELU f_y/γ_M0 et f_y/(√3·γ_M0)), "symbole_sigma" ("σ_adm" | "f_y/γ_M0"), "symbole_tau"},
     "geometrie": {"e", "r", "larg", "haut", "L", "X", "dx", "Y", "dy", "n", "d", "d0", "a" (None = L), "L_eff" (la plus
                   petite des combinaisons), "facteur_L_eff" (L/L_eff), "combinaison_L_eff", "yc", "uV" (effectif),
                   "jeu_ancrage", "sections": [{"cle", "nom", "aile", "s", "t"}] (S1 à S4), "conge": [e, e + r],
                   "bloc_angle": [0, e], "positions_charge": [X′…], "positions_ancrage": [Y′…], "s_contact" (B),
                   "entraxe" (effectif), "entraxe_auto" (bool), "distance_bout" (e₁), "positions_z": [z_i depuis le bout],
                   "type_ancrage", "profondeur", "rondelles", "rondelle": {"normale": (Ø, ép.), "large": (Ø, ép.), "nominal"},
                   "support", "support_haut", "support_bas" (None pour un voile), "bords_poutre": {"haut", "bas"} (mm, distance
                   de l'axe d'ancrage le plus proche à chaque bord ; None pour un voile)},
     "combinaisons": [<combinaison>, …]  — la nominale d'abord (1, 3 ou 9),
     "efforts": {"R", "R_ELU", "R_ELU_auto" (bool), "rapport_ELU" (R_ELU/R, None si R = 0), "Mmax" (kN·m, enveloppe = R·(X + dx)),
                 "H" (kN, traction totale des ancrages = compression du mur, enveloppe), "combinaison_H", "bras_H" (mm),
                 "H_sections" (historique : Mmax/(YH + dy), le H implicite des sections, None sinon), "C" (= H),
                 "M_angle" (B : R·(X′ − e/2) enveloppe, kN·m, None en A)},
     "verifications": [<verification>, …]  — S1, S2, S3, S4, max_h, max_v, et nette (B) ; ``verdict`` dit si la ligne
                 entre dans le verdict (en historique, les maxima du congé sont en information : le classeur ne les a pas),
     "informations": [<information>, …]  — pression diamétrale, arrachement (A), compression du mur (A) ou pression sur le
                 support (B), taux de l'autre mode (A), L_eff (si a),
     "ancrages": {"n", "d", "d0", "positions_z": [e₁ + i·s], "entraxe", "distance_bout", "etat_limite",
                  "N", "V" (kN par ancrage, dans l'état limite choisi), "N_ELS", "V_ELS", "N_ELU", "V_ELU" (kN, par ancrage ;
                  en ELU N_ELS = V_ELS = None et N_ELU = H/n, V_ELU = R/n), "combinaison_N", "bras" (mm : Y − jeu_ancrage en
                  historique, Y′ − y_c ou y_a − c en équilibre), "H",
                  "pression_diametrale", "arrachement" (N/mm², information ; A seulement — None en B, où l'effort du boulon sur
                  l'aile va vers l'angle), "compression_mur" (kN, A : résultante à y_c),
                  "pression_support" (N/mm, B), "methode": "historique"|"equilibre"},
     "diagrammes": {"horizontale": <diagramme>, "verticale": <diagramme>} (None quand diagrammes=False),
     "maillage": {"horizontale": <champ>, "verticale": <champ>} (None quand diagrammes=False) — § 9.6 :
                  <champ> = {"aile", "s": [du nu e à la fin : le congé en 12 pas, la partie droite en 36], "t": [t(s)],
                  "zeta": [−1 … +1 en 9 niveaux ; −1 = face extérieure, +1 = face du congé], "sigma_eq": [[σ_vm(s, ζ) par ζ] par s],
                  "taux": [[σ_vm/σ_lim]], "max": {"s", "zeta", "sigma_eq", "taux"}} ; pour chaque combinaison k :
                  σ_x = (M_k·ζ)/W + N_k/A (M signé), τ = |V_k|/A uniforme (le classeur), σ_vm,k = √(σ_x² + 3τ²) ; la maille
                  garde max_k (aucun mélange de positions) ; max_ζ σ_vm(s, ζ) = |M|/W + |N|/A = la vérification de la coupe,
     "maxima": {"horizontale": <verification max_h>, "verticale": <verification max_v>},
     "comparaison": {"mode", "taux_max", "gouvernant": {"cle", "nom"}} — l'autre mode, en A ; None en B,
     "garde": {"taux_historique", "taux_equilibre_P1", "depasse", "yc", "uV"} — mode historique ; None sinon,
     "taux_max", "statut": "ok"|"nok", "gouvernant": {"cle", "nom", "aile", "s", "taux", "combinaison"},
     "calcul": [<bloc>, …], "hypotheses": [str, …], "avertissements": [{"code", "niveau", "message"}, …]}

  <combinaison> = {"jeu_charge": −dx|0|+dx, "jeu_ancrage": −dy|0|+dy, "X": X′, "Y": Y′, "libelle": "charge à X + 20, ancrage à Y − 10",
                   "position_charge": "proche"|"nominale"|"loin", "position_ancrage": "haut"|"nominal"|"bas"}
     (en A, Y + dy est l'ancrage HAUT ; en B, le BAS : Y se compte depuis la face extérieure de l'aile horizontale)
  <verification> = {"cle": "S1"|"S2"|"S3"|"S4"|"max_h"|"max_v"|"nette", "nom", "aile": "horizontale"|"verticale", "s" (mm, distance
                    au dos de l'autre aile), "t" (mm), "largeur" (mm : L, L_eff, ou L − n·d0), "W" (mm³), "A" (mm²), "M" (kN·m,
                    signé : > 0 = face côté congé tendue), "N" (kN, > 0 traction), "V" (kN), "sigma", "tau", "sigma_eq" (N/mm²),
                    "sigma_adm", "taux" (σ_éq/σ_adm), "taux_sigma" (σ/σ_adm), "taux_tau" (τ/τ_adm), "ok", "etat": "ok"|"nok",
                    "verdict" (bool), "combinaison": <combinaison>, "cellule" (historique : "I37"…, sinon None),
                    et pour max_h / max_v : "reference" ("S1"|"S3"), "ecart" (taux/taux_ref − 1), "recul" (mm avant le pied du
                    congé, e + r − s), "dans_conge" (bool)}
  <information> = {"cle", "nom", "valeur", "unite", "combinaison" (ou None), "texte"}
  <diagramme> = {"aile", "fin" (mm), "largeur" (mm, la plus petite des combinaisons), "points": [<point>, …] (l'ENVELOPPE :
                 point par point, la plus défavorable des combinaisons), "courbes": [{"jeu_charge", "X", "ancrage" (le jeu
                 d'ancrage qui gouverne la section de référence pour cette charge), "points": [[s, M], …]}, …] (une courbe
                 par position de la charge, chacune enveloppe des trois positions de l'ancrage ; l'enveloppe des points
                 est leur maximum point par point), "troncons": [{"de", "a", "N", "V"}] (étiquettes des efforts, enveloppe),
                 "reperes": {"sections": [{"cle", "s", "t"}], "conge": [e, e + r], "bloc_angle": [0, e], "charge": {"X",
                 "positions"} (horizontale), "ancrage": {"Y", "positions", "bande", "d0"} (verticale), "contact" (B),
                 "maximum": {"s", "taux", "sigma_eq"}}}
  <point> = {"s", "t", "M" (kN·m), "N", "V" (kN), "sigma", "tau", "sigma_eq" (N/mm²), "taux"}
  <bloc> = {"cle", "titre", "combinaison" (ou None), "formules": [{"cle", "symbole", "termes": [{"symbole", "valeur", "unite"}],
            "resultat", "unite", "cellule" (historique)}]} — chaque résultat se retrouve à partir de ses seuls termes :
     sigma_adm f_y/coef · tau_adm σ_adm/√3 · Xp X + Δx · Yp Y + Δy · Xe X − r − e (historique : X nominal, C19 ; équilibre :
     X′) · Xer X − e (C20 ; équilibre X′ − e) · Ye Y − e − r · Yer Y − e · YH Y − e/2 · M_bras_j R·(bras + Xj)/1000 (C35,
     C45) · Mmax_j R·(X + Xj)/1000 (C55, C78) · rho (Y_réf + Yj)/(YH + Yj) · M_rho ρ·Mmax · H_hist Mmax·1000/(YH + Yj) ·
     M_hist_v H_s·(Y′ − s)/1000 · M_bras R·bras/1000 · M_lin R·(X′ − s)/1000 · bras_A Y′ − y_c · H_eq R·(X′ − u_V)/bras ·
     M_eq_A (H·(Y′ − s) − R·(e/2 − u_V))/1000 · y_a haut − Y′ · bras_B y_a − c · M_angle_B R·(X′ − e/2)/1000 · M_bas_B H·(s_c − s)/1000 ·
     t_conge e + r − √(r² − (e + r − s)²) · W b·t²/6 · A b·t · sigma_M M·10⁶/W · sigma_MN |M|·10⁶/W + |N|·10³/A ·
     tau_RLt R·10³/(b·t) · tau_VA |V|·10³/A · sigma_eq √(σ² + 3τ²) · taux valeur/limite · N_ELS_hist Mmax·1000/((Y − jeu)·n) ·
     N_ELS_eq H/n · V_ELS R/n · rapport_ELU R_ELU/R · ELU x·ρ_ELU · R_ELU_auto 1,5·R · p_diam V·10³/(d·e) ·
     sigma_lim_ELU f_y/γ_M0 · tau_lim_ELU f_y/(√3·γ_M0) · N_ELU_direct H/n · V_ELU_direct R/n (ELU) ·
     arrachement V·10³/(2·(haut − Y)·e) (A) · L_eff min(L ; a + 2·Xe) · d0 d + 2 · b_net L − n·d0 · pression_support C·10³/L.

  avertissements (code · niveau) : chevilles_non_verifiees · info ; pic_conge · info ; effet_levier · info ;
     pression_support · attention (B) ; equilibre_plus_severe · rouge (historique : la garde) ; mode_impose_B · info ;
     R_ELU_auto · info ; note_v1 · attention ; fy_epaisseur · info (e > 40 mm).

SORTIE de candidats — les cornières du catalogue, triées par masse : [{"nom", "designation", "famille", "A", "B", "t", "r1",
"masse", "aire", "taux" (taux_max, None hors domaine), "ok", "domaine" (bool), "raison" (le refus, ou None), "retenue",
"optimal"}, …] ; optimal = la plus légère « ok » et dans le domaine.
"""
import math
from typing import Callable, Dict, List, Optional, Tuple

from . import entree as E
from . import statique as S
from .entree import (AVIS_V1, ELS, ELU, EQUILIBRE, GAMMA_M0, HISTORIQUE, POUTRE, REFERENCE, cornieres, depuis_v1,  # noqa: F401
                     designation, est_v1, nom_catalogue, nouvelle_corniere, normaliser, section, section_catalogue)
from .statique import HORIZONTALE, VERTICALE

OK, NOK = "ok", "nok"
INFO, ATTENTION, ROUGE = "info", "attention", "rouge"
EPS = 1e-12                       # à égalité (à EPS relatif près), la combinaison nominale garde la main

# les sections repères : (clé, nom, aile, s(e, r), t(e, r), cellule du taux dans P8)
SECTIONS = (("S1", "Section 1 · aile horizontale, pied du congé", HORIZONTALE, lambda e, r: e + r, lambda e, r: e, "I37"),
            ("S2", "Section 2 · aile horizontale, nu de l'aile verticale", HORIZONTALE, lambda e, r: e, lambda e, r: e + r, "I47"),
            ("S3", "Section 3 · aile verticale, pied du congé", VERTICALE, lambda e, r: e + r, lambda e, r: e, "I57"),
            ("S4", "Section 4 · aile verticale, nu de l'aile horizontale", VERTICALE, lambda e, r: e, lambda e, r: e + r, "I67"))
NOMS = {"max_h": "Maximum le long de l'aile horizontale (congé)", "max_v": "Maximum le long de l'aile verticale (congé)",
        "nette": "Section nette au droit des ancrages (aile verticale)"}


# ------------------------------------------------------------------ les combinaisons de jeux
def _signe(x: float) -> str:
    return f" + {E._g(x)}" if x > 0 else f" − {E._g(-x)}" if x < 0 else ""


def combinaisons(c: Dict) -> List[Dict]:
    """Les positions de la charge (X − dx, X, X + dx) croisées avec celles de l'ancrage (Y − dy, Y, Y + dy) : une seule
    combinaison quand dx = dy = 0, trois quand un seul jeu est nul, neuf sinon. La nominale d'abord, puis par jeux
    croissants en valeur absolue — l'ordre dans lequel une égalité est départagée."""
    return _combinaisons(normaliser(c))


def _combinaisons(c: Dict) -> List[Dict]:
    dx, dy, cfg = c["dx"], c["dy"], c["configuration"]
    jx = [0.0] + ([-dx, dx] if dx > 0 else [])
    jy = [0.0] + ([-dy, dy] if dy > 0 else [])
    out = []
    for a in jx:
        for b in jy:
            pos_c = "nominale" if a == 0 else ("loin" if a > 0 else "proche")
            haut_bas = ("haut", "bas") if cfg == "A" else ("bas", "haut")
            pos_a = "nominal" if b == 0 else (haut_bas[0] if b > 0 else haut_bas[1])
            out.append({"jeu_charge": a, "jeu_ancrage": b, "X": c["X"] + a, "Y": c["Y"] + b,
                        "libelle": f"charge à X{_signe(a)}, ancrage à Y{_signe(b)}",
                        "position_charge": pos_c, "position_ancrage": pos_a})
    return out


# ------------------------------------------------------------------ le contexte d'un calcul
def _contexte(c: Dict, mode: Optional[str] = None, yc: Optional[float] = None, uV: Optional[float] = None) -> Dict:
    """Tout ce que le calcul lit : la section, le matériau, le mode effectif, les combinaisons et leur statique.
    ``mode``, ``yc``, ``uV`` forcés servent à la comparaison des modes et à la garde."""
    sec = section(c)
    e, r, larg, haut = sec["e"], sec["r"], sec["larg"], sec["haut"]
    cfg = c["configuration"]
    mode_eff = EQUILIBRE if cfg == "B" else (mode or c["mode"])
    uV_eff = (c["uV"] if c["uV"] is not None else e / 2.0) if uV is None else uV
    yc_eff = c["yc"] if yc is None else yc
    # les limites de l'état limite (§ 9.5) : ELS f_y/coef et σ_lim/√3 ; ELU f_y/γ_M0 et f_y/(√3·γ_M0), γ_M0 = 1,0
    etat = c["etat_limite"]
    sigma_adm = c["fy"] / c["coef"] if etat == ELS else c["fy"] / GAMMA_M0
    g = {"e": e, "r": r, "larg": larg, "haut": haut, "L": c["L"], "R": c["R"], "Y": c["Y"], "yc": yc_eff, "uV": uV_eff,
         "jeu_ancrage": c["jeu_ancrage"]}
    combos = _combinaisons(c)
    stats = [S.statique(g, mode_eff, cfg, cb["X"], cb["Y"]) for cb in combos]

    def largeur(aile: str, st: Dict) -> float:
        """La largeur d'une coupe : L, ou L_eff = min(L ; a + 2·Xe) sur l'aile horizontale quand a est donnée."""
        if aile == HORIZONTALE and c["a"] is not None:
            return min(c["L"], c["a"] + 2.0 * (st["Xp"] - r - e))
        return c["L"]

    return {"c": c, "sec": sec, "e": e, "r": r, "larg": larg, "haut": haut, "cfg": cfg, "mode": mode_eff, "uV": uV_eff,
            "yc": yc_eff, "etat": etat, "sigma_adm": sigma_adm, "tau_adm": sigma_adm / math.sqrt(3.0), "g": g, "combos": combos,
            "stats": stats, "largeur": largeur}


def _etat(ctx: Dict, aile: str, s: float, t: float, st: Dict, largeur: Optional[float] = None) -> Dict:
    """Les grandeurs d'une coupe pour UNE combinaison."""
    M, N, V = S.efforts(st, aile, s)
    Lw = ctx["largeur"](aile, st) if largeur is None else largeur
    W, A = S.module_aire(Lw, t)
    sigma, tau, seq = S.contraintes(M, N, V, Lw, t)
    return {"s": s, "t": t, "largeur": Lw, "W": W, "A": A, "M": M / 1e3, "N": N, "V": V, "sigma": sigma, "tau": tau,
            "sigma_eq": seq, "sigma_adm": ctx["sigma_adm"], "taux": seq / ctx["sigma_adm"], "taux_sigma": sigma / ctx["sigma_adm"],
            "taux_tau": tau / ctx["tau_adm"]}


def _pire(ctx: Dict, cle: str, nom: str, aile: str, s: float, t: float, verdict: bool = True, cellule: Optional[str] = None,
          largeur: Optional[float] = None, etats: Optional[Callable[[Dict], List[Dict]]] = None) -> Dict:
    """L'ENVELOPPE d'une coupe : la combinaison au plus grand taux (à EPS près, la nominale garde la main)."""
    best, best_cb = None, None
    for cb, st in zip(ctx["combos"], ctx["stats"]):
        for et in (etats(st) if etats else [_etat(ctx, aile, s, t, st, largeur)]):
            if best is None or et["taux"] > best["taux"] * (1.0 + EPS):
                best, best_cb = et, cb
    out = {"cle": cle, "nom": nom, "aile": aile}
    out.update(best)
    out.update({"ok": best["taux"] <= 1.0, "etat": OK if best["taux"] <= 1.0 else NOK, "verdict": verdict,
                "combinaison": best_cb, "cellule": cellule})
    return out


# ------------------------------------------------------------------ les vérifications
def _sections(ctx: Dict) -> List[Dict]:
    e, r, hist = ctx["e"], ctx["r"], ctx["mode"] == HISTORIQUE
    return [_pire(ctx, cle, nom, aile, fs(e, r), ft(e, r), cellule=cel if hist else None) for cle, nom, aile, fs, ft, cel in SECTIONS]


def _coudes(ctx: Dict, aile: str) -> List[float]:
    """Les abscisses où les efforts changent de pente : la charge, l'ancrage, le contact (B)."""
    if aile == HORIZONTALE:
        return [st["Xp"] for st in ctx["stats"]]
    pts = [st["Yp"] for st in ctx["stats"]]
    if ctx["cfg"] == "B":
        pts += [st["s_contact"] for st in ctx["stats"]]
    return pts


def _maximum(ctx: Dict, aile: str, verdict: bool) -> Dict:
    """Le maximum de σ_éq (enveloppe des combinaisons) le long d'une aile : le congé au pas ≤ 0,1 mm, les coudes de
    la partie droite (les efforts y sont linéaires), puis la section dorée."""
    e, r = ctx["e"], ctx["r"]
    fin = ctx["larg"] if aile == HORIZONTALE else ctx["haut"]
    ref = "S1" if aile == HORIZONTALE else "S3"

    def f(s: float) -> float:
        t = S.t_conge(s, e, r)
        return max(S.contraintes(*S.efforts(st, aile, s), ctx["largeur"](aile, st), t)[2] for st in ctx["stats"])

    absc = S.abscisses_exploration(e, r, fin, _coudes(ctx, aile))
    s_max, _ = S.maximum(f, absc)
    v = _pire(ctx, "max_h" if aile == HORIZONTALE else "max_v", NOMS["max_h" if aile == HORIZONTALE else "max_v"], aile, s_max,
              S.t_conge(s_max, e, r), verdict=verdict)
    v.update({"reference": ref, "recul": e + r - s_max, "dans_conge": s_max < e + r - 1e-9})
    return v


def _nette(ctx: Dict) -> Dict:
    """Configuration B : la section nette au droit du trou (largeur L − n·d0, épaisseur e), la face la plus
    sollicitée du trou — juste au-dessus (M = R·(X′ − e/2), N = −R, V = 0) ou juste au-dessous (M = H·bras, V = H)."""
    c = ctx["c"]
    b_net = c["L"] - c["n"] * E.d0(c)

    def etats(st: Dict) -> List[Dict]:
        dessus = dict(_etat(ctx, VERTICALE, st["Yp"], ctx["e"], st, largeur=b_net), face="dessus")
        dessous = dict(_etat(ctx, VERTICALE, st["Yp"] + 1e-9, ctx["e"], st, largeur=b_net), s=st["Yp"], face="dessous")
        return [dessus, dessous]

    return _pire(ctx, "nette", NOMS["nette"], VERTICALE, 0.0, ctx["e"], largeur=b_net, etats=etats)


def _verifications(ctx: Dict) -> List[Dict]:
    """S1 à S4, les deux maxima (dans le verdict en équilibre, en information en historique : le classeur ne les a
    pas, et son cas de contrôle doit rendre 88,86 %), la section nette en B."""
    hist = ctx["mode"] == HISTORIQUE
    out = _sections(ctx)
    for aile in (HORIZONTALE, VERTICALE):
        v = _maximum(ctx, aile, verdict=not hist)
        ref = next(x for x in out if x["cle"] == v["reference"])
        v["ecart"] = v["taux"] / ref["taux"] - 1.0 if ref["taux"] > 0 else 0.0
        out.append(v)
    if ctx["cfg"] == "B":
        out.append(_nette(ctx))
    return out


def _taux_max(verifs: List[Dict]) -> Dict:
    """La vérification gouvernante parmi celles du verdict (la première à égalité)."""
    return max((v for v in verifs if v["verdict"]), key=lambda v: v["taux"])


# ------------------------------------------------------------------ les ancrages
def _ancrages(ctx: Dict) -> Dict:
    c, n, R = ctx["c"], ctx["c"]["n"], ctx["c"]["R"]
    best, best_cb = None, None
    for cb, st in zip(ctx["combos"], ctx["stats"]):
        if best is None or st["H"] > best["H"] * (1.0 + EPS):
            best, best_cb = st, cb
    N_par, V_par = best["H"] / n, R / n                  # par ancrage, dans l'état limite de la saisie
    if ctx["etat"] == ELS:
        R_ELU = c["R_ELU"] if c["R_ELU"] is not None else E.RAPPORT_ELU_DEFAUT * R
        rapport = (R_ELU / R) if R > 0 else None
        N_ELS, V_ELS = N_par, V_par
        N_ELU = N_ELS * rapport if rapport is not None else 0.0
        V_ELU = V_ELS * rapport if rapport is not None else 0.0
        auto = c["R_ELU"] is None
    else:                                                # ELU saisi : rien n'est inventé pour l'ELS
        R_ELU, rapport, N_ELS, V_ELS, N_ELU, V_ELU, auto = R, None, None, None, N_par, V_par, False
    e, haut, Y = ctx["e"], ctx["haut"], c["Y"]
    # l'arrachement vers le bord (C90) : le classeur lit Y nominal ; en équilibre, l'ancrage le plus près du bord (enveloppe).
    # En A l'effort du boulon sur l'aile va vers le bout libre (haut − Y) ; en B il va vers l'ANGLE (le boulon retient la
    # cornière qui descend), où la matière continue dans l'autre aile : sans objet, None (relecture de la mécanique)
    Y_bord = Y if ctx["mode"] == HISTORIQUE else max(cb["Y"] for cb in ctx["combos"])
    out = {"n": n, "d": c["d"], "d0": E.d0(c), "positions_z": E.positions_ancrages(c), "entraxe": E.entraxe_effectif(c),
           "distance_bout": E.distance_bout(c), "etat_limite": ctx["etat"], "N": N_par, "V": V_par,
           "N_ELS": N_ELS, "V_ELS": V_ELS, "N_ELU": N_ELU, "V_ELU": V_ELU, "rapport_ELU": rapport, "R_ELU": R_ELU,
           "R_ELU_auto": auto, "combinaison_N": best_cb, "bras": best["bras"], "H": best["H"],
           "pression_diametrale": V_par * 1e3 / (c["d"] * e), "Y_bord": Y_bord,
           "arrachement": V_par * 1e3 / (2.0 * (haut - Y_bord) * e) if (ctx["cfg"] == "A" and haut > Y_bord) else None,
           "methode": ctx["mode"], "compression_mur": best["H"] if ctx["cfg"] == "A" else None,
           "pression_support": best["H"] * 1e3 / c["L"] if ctx["cfg"] == "B" else None}
    return out


# ------------------------------------------------------------------ les diagrammes
def _enveloppe_points(ctx: Dict, aile: str, absc: List[float]) -> List[Dict]:
    e, r = ctx["e"], ctx["r"]
    pts = []
    for s in absc:
        t = S.t_conge(s, e, r)
        M = N = V = 0.0
        sig = tau = seq = 0.0
        for st in ctx["stats"]:
            m, nn, vv = S.efforts(st, aile, s)
            sg, ta, sq = S.contraintes(m, nn, vv, ctx["largeur"](aile, st), t)
            if abs(m) > abs(M):
                M = m
            if abs(nn) > abs(N):
                N = nn
            if abs(vv) > abs(V):
                V = vv
            sig, tau, seq = max(sig, sg), max(tau, ta), max(seq, sq)
        pts.append({"s": s, "t": t, "M": M / 1e3, "N": N, "V": V, "sigma": sig, "tau": tau, "sigma_eq": seq, "taux": seq / ctx["sigma_adm"]})
    return pts


def _diagramme(ctx: Dict, aile: str, verifs: List[Dict]) -> Dict:
    c, e, r = ctx["c"], ctx["e"], ctx["r"]
    fin = ctx["larg"] if aile == HORIZONTALE else ctx["haut"]
    coudes = _coudes(ctx, aile)
    vmax = next(v for v in verifs if v["cle"] == ("max_h" if aile == HORIZONTALE else "max_v"))
    absc = (S.grille(e, e + r, max(r / 26.0, 1e-6)) if r > 0 else [e]) + S.grille(e + r, fin, max((fin - e - r) / 40.0, 1e-6))
    absc += [x for x in coudes if e <= x <= fin] + [vmax["s"]]
    absc = sorted(set(round(x, 9) for x in absc))
    points = _enveloppe_points(ctx, aile, absc)
    # une courbe de moment par position de la charge, enveloppe des positions de l'ancrage
    courbes = []
    s_ref = e + r
    for jc in sorted(set(cb["jeu_charge"] for cb in ctx["combos"])):
        paires = [(cb, st) for cb, st in zip(ctx["combos"], ctx["stats"]) if cb["jeu_charge"] == jc]
        pts = []
        for s in absc:
            M = 0.0
            for _, st in paires:
                m = S.efforts(st, aile, s)[0]
                if abs(m) > abs(M):
                    M = m
            pts.append([s, M / 1e3])
        gouv = max(paires, key=lambda p: (abs(S.efforts(p[1], aile, s_ref)[0]), -abs(p[0]["jeu_ancrage"])))[0]
        courbes.append({"jeu_charge": jc, "X": paires[0][0]["X"], "ancrage": gouv["jeu_ancrage"] if aile == VERTICALE else None,
                        "points": pts})
    # les tronçons : N et V en étiquettes (enveloppe)
    R, Hmax = c["R"], max(st["H"] for st in ctx["stats"])
    Xmax, Ymin, Ymax = max(st["Xp"] for st in ctx["stats"]), min(st["Yp"] for st in ctx["stats"]), max(st["Yp"] for st in ctx["stats"])
    if aile == HORIZONTALE:
        troncons = [{"de": e, "a": Xmax, "N": 0.0, "V": R}]
    elif ctx["cfg"] == "B":
        troncons = [{"de": e, "a": Ymax, "N": -R, "V": 0.0}, {"de": Ymin, "a": max(st["s_contact"] for st in ctx["stats"]), "N": 0.0, "V": Hmax}]
    elif ctx["mode"] == HISTORIQUE:
        troncons = [{"de": e, "a": Ymax, "N": 0.0, "V": R}]
    else:
        troncons = [{"de": e, "a": Ymax, "N": R, "V": Hmax}]
    reperes = {"sections": [{"cle": v["cle"], "s": v["s"], "t": v["t"]} for v in verifs if v["aile"] == aile and v["cle"].startswith("S")],
               "conge": [e, e + r], "bloc_angle": [0.0, e], "maximum": {"s": vmax["s"], "taux": vmax["taux"], "sigma_eq": vmax["sigma_eq"]}}
    if aile == HORIZONTALE:
        reperes["charge"] = {"X": c["X"], "positions": sorted(set(st["Xp"] for st in ctx["stats"]))}
    else:
        reperes["ancrage"] = {"Y": c["Y"], "positions": sorted(set(st["Yp"] for st in ctx["stats"])), "bande": [c["Y"] - c["dy"], c["Y"] + c["dy"]],
                              "d0": E.d0(c)}
        if ctx["cfg"] == "B":
            reperes["contact"] = ctx["stats"][0]["s_contact"]
    return {"aile": aile, "fin": fin, "largeur": min(ctx["largeur"](aile, st) for st in ctx["stats"]), "points": points,
            "courbes": courbes, "troncons": troncons, "reperes": reperes}


# ------------------------------------------------------------------ le champ de contraintes (§ 9.6)
PAS_MAILLAGE = (12, 36)                 # mailles dans le congé, puis dans la partie droite
ZETA = tuple(-1.0 + 0.25 * i for i in range(9))   # la position dans l'épaisseur : −1 face extérieure, +1 face du congé


def _maillage(ctx: Dict, aile: str) -> Dict:
    """Le champ de von Mises de l'ENVELOPPE sur une aile : pour chaque maille (s, ζ) et chaque combinaison k,
    σ_x = (M_k·ζ)/W + N_k/A (M signé : > 0 tend la face du congé, ζ = +1), τ = |V_k|/A uniforme (le classeur),
    σ_vm,k = √(σ_x² + 3τ²) ; la maille garde le maximum des combinaisons — jamais un moment d'une position avec le
    tranchant d'une autre. Contraintes de poutre, pas un calcul par éléments finis ; sous s = e (bloc d'angle) rien."""
    e, r = ctx["e"], ctx["r"]
    fin = ctx["larg"] if aile == HORIZONTALE else ctx["haut"]
    absc = (S.grille(e, e + r, r / PAS_MAILLAGE[0]) if r > 0 else [e]) + S.grille(e + r, fin, max((fin - e - r) / PAS_MAILLAGE[1], 1e-6))
    absc = sorted(set(round(x, 9) for x in absc))
    ts, seqs, taux = [], [], []
    best = {"s": absc[0], "zeta": 0.0, "sigma_eq": -1.0, "taux": -1.0}
    for s in absc:
        t = S.t_conge(s, e, r)
        ts.append(t)
        ligne = []
        for z in ZETA:
            vm = 0.0
            for st in ctx["stats"]:
                M, N, V = S.efforts(st, aile, s)
                W, A = S.module_aire(ctx["largeur"](aile, st), t)
                sx = M * 1e3 * z / W + N * 1e3 / A
                tau = abs(V) * 1e3 / A
                vm = max(vm, math.sqrt(sx * sx + 3.0 * tau * tau))
            ligne.append(vm)
            if vm > best["sigma_eq"]:
                best = {"s": s, "zeta": z, "sigma_eq": vm, "taux": vm / ctx["sigma_adm"]}
        seqs.append(ligne)
        taux.append([v / ctx["sigma_adm"] for v in ligne])
    return {"aile": aile, "s": absc, "t": ts, "zeta": list(ZETA), "sigma_eq": seqs, "taux": taux, "max": best}


# ------------------------------------------------------------------ les formules du détail
def _f(cle: str, symbole: str, termes, resultat: float, unite: str, cellule: Optional[str] = None) -> Dict:
    return {"cle": cle, "symbole": symbole, "termes": [{"symbole": s, "valeur": v, "unite": u} for s, v, u in termes],
            "resultat": resultat, "unite": unite, "cellule": cellule}


def _bloc(cle: str, titre: str, combinaison: Optional[Dict], formules: List[Dict]) -> Dict:
    return {"cle": cle, "titre": titre, "combinaison": combinaison, "formules": formules}


def _formules_coupe(ctx: Dict, v: Dict, hist: bool, cellules: Tuple[str, ...] = ()) -> List[Dict]:
    """W, A, σ, τ, σ_éq, taux d'une coupe (les cellules du classeur, en historique : W, σ, τ, σ_éq, taux, taux σ, taux τ)."""
    cel = [x or None for x in cellules] + [None] * 7
    f = [_f("W", "W", [("b", v["largeur"], "mm"), ("t", v["t"], "mm")], v["W"], "mm³", cel[0])]
    if hist:
        f += [_f("sigma_M", "σ", [("M", v["M"], "kN·m"), ("W", v["W"], "mm³")], v["sigma"], "N/mm²", cel[1]),
              _f("tau_RLt", "τ", [("R", ctx["c"]["R"], "kN"), ("b", v["largeur"], "mm"), ("t", v["t"], "mm")], v["tau"], "N/mm²", cel[2])]
    else:
        f += [_f("A", "A", [("b", v["largeur"], "mm"), ("t", v["t"], "mm")], v["A"], "mm²"),
              _f("sigma_MN", "σ", [("M", v["M"], "kN·m"), ("W", v["W"], "mm³"), ("N", v["N"], "kN"), ("A", v["A"], "mm²")], v["sigma"], "N/mm²"),
              _f("tau_VA", "τ", [("V", v["V"], "kN"), ("A", v["A"], "mm²")], v["tau"], "N/mm²")]
    f += [_f("sigma_eq", "σ_éq", [("σ", v["sigma"], "N/mm²"), ("τ", v["tau"], "N/mm²")], v["sigma_eq"], "N/mm²", cel[3]),
          _f("taux", "taux", [("σ_éq", v["sigma_eq"], "N/mm²"), ("σ_adm", v["sigma_adm"], "N/mm²")], v["taux"], "", cel[4]),
          _f("taux", "σ/σ_adm", [("σ", v["sigma"], "N/mm²"), ("σ_adm", v["sigma_adm"], "N/mm²")], v["taux_sigma"], "", cel[5]),
          _f("taux", "τ/τ_adm", [("τ", v["tau"], "N/mm²"), ("τ_adm", ctx["tau_adm"], "N/mm²")], v["taux_tau"], "", cel[6])]
    return f


def _formules_moment(ctx: Dict, v: Dict, st: Dict) -> List[Dict]:
    """Les formules qui mènent au moment d'une coupe (mode et configuration) ; s = v["s"]. En historique, P8 mot pour
    mot : les bras C19, C20, C23, C24, C25 sont NOMINAUX et le jeu Xj (C14) s'ajoute dans C35, C45, C55 ; Yj (C17)
    entre dans ρ (C56, C64)."""
    c, e, r, hist = ctx["c"], ctx["e"], ctx["r"], ctx["mode"] == HISTORIQUE
    cb = v["combinaison"]
    s, Xp, Yp, R = v["s"], st["Xp"], st["Yp"], c["R"]
    jx, jy = cb["jeu_charge"], cb["jeu_ancrage"]
    f: List[Dict] = []
    if hist:
        if v["aile"] == HORIZONTALE:
            if v["cle"] == "S1":
                f += [_f("Xe", "Xe", [("X", c["X"], "mm"), ("r", r, "mm"), ("e", e, "mm")], c["X"] - r - e, "mm", "C19"),
                      _f("M_bras_j", "M1", [("R", R, "kN"), ("Xe", c["X"] - r - e, "mm"), ("Xj", jx, "mm")], abs(v["M"]), "kN·m", "C35")]
            elif v["cle"] == "S2":
                f += [_f("Xer", "Xe+r", [("X", c["X"], "mm"), ("e", e, "mm")], c["X"] - e, "mm", "C20"),
                      _f("M_bras_j", "M2", [("R", R, "kN"), ("Xe+r", c["X"] - e, "mm"), ("Xj", jx, "mm")], abs(v["M"]), "kN·m", "C45")]
            else:
                f += [_f("Xp", "X′", [("X", c["X"], "mm"), ("Xj", jx, "mm")], Xp, "mm", "C13 + C14"),
                      _f("M_lin", "M(s)", [("R", R, "kN"), ("X′", Xp, "mm"), ("s", s, "mm")], abs(v["M"]), "kN·m")]
            return f
        f += [_f("Mmax_j", "Mmax", [("R", R, "kN"), ("X", c["X"], "mm"), ("Xj", jx, "mm")], st["Mmax"] / 1e3, "kN·m", "C55"),
              _f("YH", "YH", [("Y", c["Y"], "mm"), ("e", e, "mm")], st["YH"], "mm", "C25")]
        if v["cle"] == "S3":
            f += [_f("Ye", "Ye", [("Y", c["Y"], "mm"), ("e", e, "mm"), ("r", r, "mm")], c["Y"] - e - r, "mm", "C23"),
                  _f("rho", "ρ₃", [("Ye", c["Y"] - e - r, "mm"), ("Yj", jy, "mm"), ("YH", st["YH"], "mm")], st["rho3"], "", "C56"),
                  _f("M_rho", "M3", [("ρ₃", st["rho3"], ""), ("Mmax", st["Mmax"] / 1e3, "kN·m")], v["M"], "kN·m", "C57")]
        elif v["cle"] == "S4":
            f += [_f("Yer", "Ye+r", [("Y", c["Y"], "mm"), ("e", e, "mm")], c["Y"] - e, "mm", "C24"),
                  _f("rho", "ρ₄", [("Ye+r", c["Y"] - e, "mm"), ("Yj", jy, "mm"), ("YH", st["YH"], "mm")], st["rho4"], "", "C64"),
                  _f("M_rho", "M4", [("ρ₄", st["rho4"], ""), ("Mmax", st["Mmax"] / 1e3, "kN·m")], v["M"], "kN·m", "C65")]
        else:
            f += [_f("Yp", "Y′", [("Y", c["Y"], "mm"), ("Yj", jy, "mm")], Yp, "mm", "C16 + C17"),
                  _f("H_hist", "H_s", [("Mmax", st["Mmax"] / 1e3, "kN·m"), ("YH", st["YH"], "mm"), ("Yj", jy, "mm")], st["H_sections"], "kN"),
                  _f("M_hist_v", "M(s)", [("H_s", st["H_sections"], "kN"), ("Y′", Yp, "mm"), ("s", s, "mm")], v["M"], "kN·m")]
        return f
    f.append(_f("Xp", "X′", [("X", c["X"], "mm"), ("Δx", jx, "mm")], Xp, "mm"))
    if v["aile"] == HORIZONTALE:
        if v["cle"] == "S1":
            f += [_f("Xe", "Xe", [("X′", Xp, "mm"), ("r", r, "mm"), ("e", e, "mm")], Xp - r - e, "mm"),
                  _f("M_bras", "M1", [("R", R, "kN"), ("Xe", Xp - r - e, "mm")], abs(v["M"]), "kN·m")]
        elif v["cle"] == "S2":
            f += [_f("Xer", "Xe+r", [("X′", Xp, "mm"), ("e", e, "mm")], Xp - e, "mm"),
                  _f("M_bras", "M2", [("R", R, "kN"), ("Xe+r", Xp - e, "mm")], abs(v["M"]), "kN·m")]
        else:
            f.append(_f("M_lin", "M(s)", [("R", R, "kN"), ("X′", Xp, "mm"), ("s", s, "mm")], abs(v["M"]), "kN·m"))
        return f
    f.append(_f("Yp", "Y′", [("Y", c["Y"], "mm"), ("Δy", jy, "mm")], Yp, "mm"))
    if ctx["cfg"] == "A":
        f.append(_f("bras_A", "Y′ − y_c", [("Y′", Yp, "mm"), ("y_c", ctx["yc"], "mm")], st["bras"], "mm"))
        f.append(_f("H_eq", "H", [("R", R, "kN"), ("X′", Xp, "mm"), ("u_V", ctx["uV"], "mm"), ("bras", st["bras"], "mm")], st["H"], "kN"))
        f.append(_f("M_eq_A", "M(s)", [("H", st["H"], "kN"), ("Y′", Yp, "mm"), ("s", s, "mm"), ("R", R, "kN"), ("e", e, "mm"), ("u_V", ctx["uV"], "mm")],
                    v["M"], "kN·m"))
        return f
    f.append(_f("y_a", "y_a", [("haut", ctx["haut"], "mm"), ("Y′", Yp, "mm")], st["ya"], "mm"))
    f.append(_f("bras_B", "y_a − c", [("y_a", st["ya"], "mm"), ("c", ctx["yc"], "mm")], st["bras"], "mm"))
    f.append(_f("H_eq", "H", [("R", R, "kN"), ("X′", Xp, "mm"), ("u_V", ctx["uV"], "mm"), ("bras", st["bras"], "mm")], st["H"], "kN"))
    if s <= Yp + 1e-12 and abs(abs(v["M"]) * 1e3 - st["M_angle"]) <= 1e-9 * max(1.0, st["M_angle"]):
        f.append(_f("M_angle_B", "M", [("R", R, "kN"), ("X′", Xp, "mm"), ("e", e, "mm")], abs(v["M"]), "kN·m"))
    else:
        f.append(_f("M_bas_B", "M(s)", [("H", st["H"], "kN"), ("s_c", st["s_contact"], "mm"), ("s", s, "mm")], abs(v["M"]), "kN·m"))
    return f


def _statique_de(ctx: Dict, cb: Dict) -> Dict:
    return ctx["stats"][ctx["combos"].index(cb)]


def _calcul(ctx: Dict, verifs: List[Dict], anc: Dict, geo: Dict) -> List[Dict]:
    c, hist = ctx["c"], ctx["mode"] == HISTORIQUE
    if ctx["etat"] == ELS:
        blocs = [_bloc("materiau", "Matériau · contraintes admissibles (ELS)", None,
                       [_f("sigma_adm", "σ_adm", [("f_y", c["fy"], "N/mm²"), ("coef", c["coef"], "")], ctx["sigma_adm"], "N/mm²", "H31" if hist else None),
                        _f("tau_adm", "τ_adm", [("σ_adm", ctx["sigma_adm"], "N/mm²")], ctx["tau_adm"], "N/mm²", "H34" if hist else None)])]
    else:
        blocs = [_bloc("materiau", "Matériau · limites ELU (EN 1993-1-1 § 6.2.1(5), γ_M0 = 1,0)", None,
                       [_f("sigma_lim_ELU", "f_y/γ_M0", [("f_y", c["fy"], "N/mm²"), ("γ_M0", GAMMA_M0, "")], ctx["sigma_adm"], "N/mm²"),
                        _f("tau_lim_ELU", "f_y/(√3·γ_M0)", [("f_y", c["fy"], "N/mm²"), ("γ_M0", GAMMA_M0, "")], ctx["tau_adm"], "N/mm²")])]
    if c["a"] is not None:
        st = _statique_de(ctx, geo["combinaison_L_eff"])
        blocs.append(_bloc("largeur_efficace", "Largeur efficace de l'aile horizontale (diffusion à 45° jusqu'à la section 1)", geo["combinaison_L_eff"],
                           [_f("Xp", "X′", [("X", c["X"], "mm"), ("Δx", geo["combinaison_L_eff"]["jeu_charge"], "mm")], st["Xp"], "mm"),
                            _f("Xe", "Xe′", [("X′", st["Xp"], "mm"), ("r", ctx["r"], "mm"), ("e", ctx["e"], "mm")], st["Xp"] - ctx["r"] - ctx["e"], "mm"),
                            _f("L_eff", "L_eff", [("L", c["L"], "mm"), ("a", c["a"], "mm"), ("Xe′", st["Xp"] - ctx["r"] - ctx["e"], "mm")], geo["L_eff"], "mm")]))
    CEL = {"S1": ("C32", "G31", "G34", "G37", "I37", "I31", "I34"), "S2": ("C42", "G41", "G44", "G47", "I47", "I41", "I44"),
           "S3": ("C52", "G51", "G54", "G57", "I57", "I51", "I54"), "S4": ("C62", "G61", "G64", "G67", "I67", "I61", "I64")}
    for v in verifs:
        st = _statique_de(ctx, v["combinaison"])
        f = []
        if v["cle"] == "nette":
            f += [_f("d0", "d0", [("d", c["d"], "mm")], anc["d0"], "mm"),
                  _f("b_net", "b_net", [("L", c["L"], "mm"), ("n", c["n"], ""), ("d0", anc["d0"], "mm")], v["largeur"], "mm")]
            f += _formules_moment(ctx, v, st)
        else:
            if v["cle"] in ("max_h", "max_v") and v["dans_conge"]:
                f.append(_f("t_conge", "t(s)", [("e", ctx["e"], "mm"), ("r", ctx["r"], "mm"), ("s", v["s"], "mm")], v["t"], "mm"))
            f += _formules_moment(ctx, v, st)
        f += _formules_coupe(ctx, v, hist, CEL.get(v["cle"], ()) if hist else ())
        blocs.append(_bloc(v["cle"], v["nom"], v["combinaison"], f))
    # ancrages
    st = _statique_de(ctx, anc["combinaison_N"])
    fa = []
    els = ctx["etat"] == ELS
    sN, sV = ("N_ELS", "V_ELS") if els else ("N_ELU", "V_ELU")   # le nom de l'effort par ancrage dans l'état limite saisi
    if hist:
        fa.append(_f("Mmax_j", "Mmax", [("R", c["R"], "kN"), ("X", c["X"], "mm"), ("Xj", anc["combinaison_N"]["jeu_charge"], "mm")], st["Mmax"] / 1e3, "kN·m", "C78"))
        if els:
            fa.append(_f("N_ELS_hist", "N_ELS", [("Mmax", st["Mmax"] / 1e3, "kN·m"), ("Y", c["Y"], "mm"), ("jeu", c["jeu_ancrage"], "mm"), ("n", c["n"], "")],
                         anc["N_ELS"], "kN", "C82"))
        else:
            fa.append(_f("H_hist", "H", [("Mmax", st["Mmax"] / 1e3, "kN·m"), ("Y − jeu", st["bras"], "mm"), ("Yj", 0.0, "mm")], st["H"], "kN", "C82·n"))
    else:
        if ctx["cfg"] == "A":
            fa.append(_f("bras_A", "Y′ − y_c", [("Y′", st["Yp"], "mm"), ("y_c", ctx["yc"], "mm")], st["bras"], "mm"))
        else:
            fa.append(_f("y_a", "y_a", [("haut", ctx["haut"], "mm"), ("Y′", st["Yp"], "mm")], st["ya"], "mm"))
            fa.append(_f("bras_B", "y_a − c", [("y_a", st["ya"], "mm"), ("c", ctx["yc"], "mm")], st["bras"], "mm"))
        fa.append(_f("H_eq", "H", [("R", c["R"], "kN"), ("X′", st["Xp"], "mm"), ("u_V", ctx["uV"], "mm"), ("bras", st["bras"], "mm")], st["H"], "kN"))
        if els:
            fa.append(_f("N_ELS_eq", "N_ELS", [("H", st["H"], "kN"), ("n", c["n"], "")], anc["N_ELS"], "kN"))
    if els:
        fa.append(_f("V_ELS", "V_ELS", [("R", c["R"], "kN"), ("n", c["n"], "")], anc["V_ELS"], "kN", "C83" if hist else None))
        if anc["R_ELU_auto"]:
            fa.append(_f("R_ELU_auto", "R_ELU", [("R", c["R"], "kN")], anc["R_ELU"], "kN"))
        if anc["rapport_ELU"] is not None:
            fa += [_f("rapport_ELU", "ρ_ELU", [("R_ELU", anc["R_ELU"], "kN"), ("R", c["R"], "kN")], anc["rapport_ELU"], "", "C71/C70" if hist else None),
                   _f("ELU", "N_ELU", [("N_ELS", anc["N_ELS"], "kN"), ("ρ_ELU", anc["rapport_ELU"], "")], anc["N_ELU"], "kN", "C85" if hist else None),
                   _f("ELU", "V_ELU", [("V_ELS", anc["V_ELS"], "kN"), ("ρ_ELU", anc["rapport_ELU"], "")], anc["V_ELU"], "kN", "C86" if hist else None)]
    else:
        fa += [_f("N_ELU_direct", "N_ELU", [("H", st["H"], "kN"), ("n", c["n"], "")], anc["N_ELU"], "kN"),
               _f("V_ELU_direct", "V_ELU", [("R", c["R"], "kN"), ("n", c["n"], "")], anc["V_ELU"], "kN")]
    fa.append(_f("p_diam", "p_diam", [(sV, anc["V"], "kN"), ("d", c["d"], "mm"), ("e", ctx["e"], "mm")], anc["pression_diametrale"], "N/mm²", "C88" if hist else None))
    if anc["arrachement"] is not None:
        fa.append(_f("arrachement", "σ_arr", [(sV, anc["V"], "kN"), ("haut", ctx["haut"], "mm"), ("Y", anc["Y_bord"], "mm"), ("e", ctx["e"], "mm")],
                     anc["arrachement"], "N/mm²", "C90" if hist else None))
    if anc["pression_support"] is not None:
        fa.append(_f("pression_support", "p_support", [("C", st["H"], "kN"), ("L", c["L"], "mm")], anc["pression_support"], "N/mm"))
    blocs.append(_bloc("ancrages", "Ancrages · efforts par ancrage", anc["combinaison_N"], fa))
    return blocs


# ------------------------------------------------------------------ hypothèses, avertissements
def _hypotheses(ctx: Dict, anc: Dict, geo: Dict) -> List[str]:
    c = ctx["c"]
    if ctx["etat"] == ELS:
        h = [f"ELS, méthode aux contraintes admissibles : σ_adm = f_y/{E._g(c['coef'])} = {E._f(ctx['sigma_adm'])} N/mm², τ_adm = σ_adm/√3 = "
             f"{E._f(ctx['tau_adm'])} N/mm², σ_éq = √(σ² + 3τ²) ≤ σ_adm sur les efforts ELS non majorés — ce n'est pas une justification "
             "EC3 aux états limites."]
    else:
        h = [f"ELU, critère élastique de l'EN 1993-1-1 § 6.2.1(5) : σ_éq = √(σ² + 3τ²) ≤ f_y/γ_M0 = {E._f(ctx['sigma_adm'])} N/mm² et "
             f"τ ≤ f_y/(√3·γ_M0) = {E._f(ctx['tau_adm'])} N/mm², γ_M0 = {E._g(GAMMA_M0)} (NBN EN 1993-1-1 ANB), sur la réaction R saisie à l'ELU "
             "— le coefficient 1,5 de l'ELS ne s'applique pas."]
    h += [f"Configuration {ctx['cfg']} : {E.LIBELLES_CONFIGURATION[ctx['cfg']]} ; {E.LIBELLES_MODE[ctx['mode']]}.",
          "Tranche courte : la longueur reprise L est la largeur de toutes les sections ; les n ancrages se partagent R et la traction."]
    if ctx["mode"] == HISTORIQUE:
        h.append(f"Formules de la feuille P8 : sections avec le pivot e/2 (YH = Y − e/2) et τ = R/(L·t) dans les deux ailes ; traction "
                 f"d'ancrage sur le bras Y − jeu = {E._g(c['Y'] - c['jeu_ancrage'])} mm (C76 = {E._g(c['jeu_ancrage'])} mm).")
    elif ctx["cfg"] == "A":
        h.append(f"Un seul corps libre, frottement nul au mur : compression du mur à y_c = {E._g(ctx['yc'])} mm du dessous, cisaillement "
                 f"d'ancrage à u_V = {E._g(ctx['uV'])} mm du nu du mur ; N = +R et V = H dans l'aile verticale.")
    else:
        h.append(f"Corps libre de la cornière retournée, frottement nul : contact au bout libre (c = {E._g(ctx['yc'])} mm), cisaillement "
                 f"d'ancrage à u_V = {E._g(ctx['uV'])} mm ; moment constant R·(X′ − e/2) de l'ancrage à l'angle, N = −R ; section nette au trou ; "
                 "pas d'arrachement vers le bord (l'effort du boulon sur l'aile va vers l'angle).")
    jeux = (f"Jeux par l'enveloppe : charge à X − {E._g(c['dx'])}, X, X + {E._g(c['dx'])} ; ancrage à Y − {E._g(c['dy'])}, Y, Y + {E._g(c['dy'])} ; "
            f"{len(ctx['combos'])} combinaison(s), chaque grandeur garde la plus défavorable.") if (c["dx"] > 0 or c["dy"] > 0) else \
        "Sans jeu (dx = dy = 0) : une seule combinaison, la charge à X et l'ancrage à Y."
    h.append(jeux)
    if c["a"] is not None:
        h.append(f"Largeur d'appui a = {E._g(c['a'])} mm : largeur efficace de l'aile horizontale L_eff = min(L ; a + 2·Xe) = "
                 f"{E._f(geo['L_eff'])} mm (σ × {E._f(geo['facteur_L_eff'], 3)}).")
    if ctx["etat"] == ELS:
        h.append("R_ELU = 1,5·R (auto)." if anc["R_ELU_auto"] else f"R_ELU = {E._g(anc['R_ELU'])} kN (saisie).")
    h.append(f"Ancrages : {c['n']} × {E.LIBELLES_TYPE_ANCRAGE[c['type_ancrage']]} Ø{E._g(c['d'])}, profondeur {E._g(c['profondeur'])} mm, "
             f"entraxe {E._f(anc['entraxe'])} mm{' (L/n)' if c['entraxe'] is None else ''}, e₁ = {E._f(anc['distance_bout'])} mm au bout, "
             f"{E.LIBELLES_RONDELLES[c['rondelles']]} ; support : {E.LIBELLES_SUPPORT[c['support']]}"
             + (f" ({E._g(c['support_haut'])} mm au-dessus et {E._g(c['support_bas'])} mm au-dessous de la face extérieure de l'aile horizontale)."
                if c["support"] == POUTRE else "."))
    return h


def _avertissements(ctx: Dict, anc: Dict, garde: Optional[Dict]) -> List[Dict]:
    c = ctx["c"]
    av = [{"code": "chevilles_non_verifiees", "niveau": INFO,
           "message": ("Chevilles non vérifiées : efforts N et V par ancrage fournis pour Hilti PROFIS, ELS et ELU séparés."
                       if ctx["etat"] == ELS else
                       "Chevilles non vérifiées : efforts N et V par ancrage à l'ELU fournis pour Hilti PROFIS (R saisie à l'ELU).")},
          {"code": "pic_conge", "niveau": INFO,
           "message": "Pic de contrainte au congé non vérifié (contrainte de pointe ≈ 1,3 × la contrainte de poutre : statique seulement, "
                      "à considérer en fatigue)."},
          {"code": "effet_levier", "niveau": INFO,
           "message": "Effet de levier non couvert si les chevilles sont très raides (> ≈ 250 kN/mm)."}]
    if c.get("v1"):
        av.insert(0, {"code": "note_v1", "niveau": ATTENTION, "message": AVIS_V1})
    if garde is not None and garde["depasse"]:
        av.insert(0, {"code": "equilibre_plus_severe", "niveau": ROUGE,
                      "message": f"Équilibre P1 (pivot e/2, u_V = e/2) plus sévère que la méthode historique sur l'aile verticale : "
                                 f"{E._f(100 * garde['taux_equilibre_P1'])} % contre {E._f(100 * garde['taux_historique'])} % — ancrage bas, la méthode "
                                 "historique n'est plus du côté sûr."})
    if ctx["cfg"] == "B":
        av.append({"code": "pression_support", "niveau": ATTENTION,
                   "message": f"Pression du bout libre sur le support à vérifier : C = {E._f(anc['H'], 2)} kN, soit {E._f(anc['pression_support'])} N/mm "
                              "de cornière (le support n'est pas connu du module)."})
        if c["mode"] == HISTORIQUE:
            av.append({"code": "mode_impose_B", "niveau": INFO,
                       "message": "En configuration B, les formules du classeur ne s'appliquent pas (elles donneraient −18 % sur M3) : "
                                  "calcul en mode équilibre."})
    if anc["R_ELU_auto"] and ctx["etat"] == ELS:
        av.append({"code": "R_ELU_auto", "niveau": INFO,
                   "message": f"R_ELU = 1,5·R = {E._f(anc['R_ELU'], 2)} kN (auto) : à remplacer par la combinaison réelle 1,35·G + 1,5·Q."})
    if ctx["e"] > E.T_FY_REDUIT:
        av.append({"code": "fy_epaisseur", "niveau": INFO,
                   "message": f"Épaisseur {E._g(ctx['e'])} mm > 40 mm : le Tableau 3.1 réduit f_y au-delà de 40 mm ; la valeur saisie est gardée."})
    return av


# ------------------------------------------------------------------ le calcul complet
def _geometrie(ctx: Dict) -> Dict:
    c, e, r = ctx["c"], ctx["e"], ctx["r"]
    best, best_cb = None, None
    for cb, st in zip(ctx["combos"], ctx["stats"]):
        Lw = ctx["largeur"](HORIZONTALE, st)
        if best is None or Lw < best * (1.0 - EPS):
            best, best_cb = Lw, cb
    geo = {"e": e, "r": r, "larg": ctx["larg"], "haut": ctx["haut"], "L": c["L"], "X": c["X"], "dx": c["dx"], "Y": c["Y"], "dy": c["dy"],
           "n": c["n"], "d": c["d"], "d0": E.d0(c), "a": c["a"], "L_eff": best, "facteur_L_eff": c["L"] / best, "combinaison_L_eff": best_cb,
           "yc": ctx["yc"], "uV": ctx["uV"], "jeu_ancrage": c["jeu_ancrage"],
           "sections": [{"cle": cle, "nom": nom, "aile": aile, "s": fs(e, r), "t": ft(e, r)} for cle, nom, aile, fs, ft, _ in SECTIONS],
           "conge": [e, e + r], "bloc_angle": [0.0, e],
           "positions_charge": sorted(set(st["Xp"] for st in ctx["stats"])), "positions_ancrage": sorted(set(st["Yp"] for st in ctx["stats"])),
           "s_contact": ctx["stats"][0]["s_contact"] if ctx["cfg"] == "B" else None,
           # § 9.2 et 9.3 : les ancrages (positions, type, profondeur, rondelles) et le support, pour le dessin et la note
           "entraxe": E.entraxe_effectif(c), "entraxe_auto": c["entraxe"] is None, "distance_bout": E.distance_bout(c),
           "positions_z": E.positions_ancrages(c), "type_ancrage": c["type_ancrage"], "profondeur": c["profondeur"],
           "rondelles": c["rondelles"], "rondelle": E.rondelle(c["d"]), "support": c["support"],
           "support_haut": c["support_haut"] if c["support"] == POUTRE else None,
           "support_bas": c["support_bas"] if c["support"] == POUTRE else None, "bords_poutre": _bords_poutre(ctx)}
    return geo


def _bords_poutre(ctx: Dict) -> Optional[Dict]:
    """Poutre : la distance de l'axe d'ancrage le plus proche au bord supérieur et au bord inférieur (mm, pour Hilti) ;
    None pour un voile. En A les ancrages sont à Y ± dy au-dessus de la face extérieure de l'aile horizontale, en B à
    Y ± dy au-dessous."""
    c = ctx["c"]
    if c["support"] != POUTRE:
        return None
    Y, dy, hs, bs = c["Y"], c["dy"], c["support_haut"], c["support_bas"]
    if ctx["cfg"] == "A":
        return {"haut": hs - (Y + dy), "bas": bs + (Y - dy)}
    return {"haut": hs + (Y - dy), "bas": bs - (Y + dy)}


def _efforts(ctx: Dict, anc: Dict) -> Dict:
    c = ctx["c"]
    hist = ctx["mode"] == HISTORIQUE
    return {"R": c["R"], "R_ELU": anc["R_ELU"], "R_ELU_auto": anc["R_ELU_auto"], "rapport_ELU": anc["rapport_ELU"],
            "Mmax": max(st["Mmax"] for st in ctx["stats"]) / 1e3, "H": anc["H"], "combinaison_H": anc["combinaison_N"], "bras_H": anc["bras"],
            "H_sections": max(st["H_sections"] for st in ctx["stats"]) if hist else None, "C": anc["H"],
            "M_angle": max(st["M_angle"] for st in ctx["stats"]) / 1e3 if ctx["cfg"] == "B" else None}


def _informations(ctx: Dict, anc: Dict, geo: Dict, comparaison: Optional[Dict]) -> List[Dict]:
    c = ctx["c"]
    et = ctx["etat"]
    inf = [{"cle": "pression_diametrale", "nom": f"Pression diamétrale ({et})", "valeur": anc["pression_diametrale"], "unite": "N/mm²",
            "combinaison": None, "texte": f"V_{et}/(d·e), sans limite (C88 du classeur)"}]
    if anc["arrachement"] is not None:
        inf.append({"cle": "arrachement", "nom": f"Arrachement vers le bord ({et})", "valeur": anc["arrachement"], "unite": "N/mm²",
                    "combinaison": None, "texte": f"V_{et}/(2·(haut − Y)·e) avec Y = {E._g(anc['Y_bord'])} mm, sans limite (C90 du classeur)"})
    if geo["bords_poutre"] is not None:
        inf.append({"cle": "bord_poutre_haut", "nom": "Ancrage : distance au bord supérieur de la poutre", "valeur": geo["bords_poutre"]["haut"],
                    "unite": "mm", "combinaison": None, "texte": "axe de l'ancrage le plus haut (Y ± dy), pour Hilti PROFIS"})
        inf.append({"cle": "bord_poutre_bas", "nom": "Ancrage : distance au bord inférieur de la poutre", "valeur": geo["bords_poutre"]["bas"],
                    "unite": "mm", "combinaison": None, "texte": "axe de l'ancrage le plus bas (Y ± dy), pour Hilti PROFIS"})
    if ctx["cfg"] == "A":
        inf.append({"cle": "compression_mur", "nom": "Compression du mur", "valeur": anc["compression_mur"], "unite": "kN",
                    "combinaison": anc["combinaison_N"],
                    "texte": f"C = H, résultante à y_c = {E._g(ctx['yc'])} mm du dessous" if ctx["mode"] == EQUILIBRE
                    else f"C = H = Mmax/(Y − jeu), le couple des ancrages (bras {E._g(anc['bras'])} mm)"})
    else:
        inf.append({"cle": "pression_support", "nom": "Pression du bout libre sur le support", "valeur": anc["pression_support"], "unite": "N/mm",
                    "combinaison": anc["combinaison_N"], "texte": f"C = H = {E._f(anc['H'], 2)} kN au bout libre, rapportée à L"})
    if comparaison is not None:
        inf.append({"cle": "autre_mode", "nom": f"Taux maximal en mode {E.LIBELLES_MODE[comparaison['mode']]}", "valeur": comparaison["taux_max"],
                    "unite": "", "combinaison": None, "texte": f"gouvernant : {comparaison['gouvernant']['nom']}"})
    if c["a"] is not None:
        inf.append({"cle": "L_eff", "nom": "Largeur efficace de l'aile horizontale", "valeur": geo["L_eff"], "unite": "mm",
                    "combinaison": geo["combinaison_L_eff"], "texte": f"min(L ; a + 2·Xe) : σ de l'aile horizontale × {E._f(geo['facteur_L_eff'], 3)}"})
    return inf


def calculer(c: Dict, diagrammes: bool = True, annexes: bool = True) -> Dict:
    """Le calcul complet d'une cornière (contrat dans la docstring du module). ``diagrammes`` : les tableaux de points
    des deux ailes ; ``annexes`` : la comparaison avec l'autre mode (A) et la garde (historique) — coupés pour le
    sélecteur, qui n'en a pas besoin."""
    c = normaliser(c)
    ctx = _contexte(c)
    verifs = _verifications(ctx)
    anc = _ancrages(ctx)
    geo = _geometrie(ctx)
    gov = _taux_max(verifs)
    comparaison = garde = None
    if annexes and ctx["cfg"] == "A":
        autre = EQUILIBRE if ctx["mode"] == HISTORIQUE else HISTORIQUE
        v2 = _verifications(_contexte(c, mode=autre))
        g2 = _taux_max(v2)
        comparaison = {"mode": autre, "taux_max": g2["taux"], "gouvernant": {"cle": g2["cle"], "nom": g2["nom"]}}
    if annexes and ctx["mode"] == HISTORIQUE:
        v3 = _sections(_contexte(c, mode=EQUILIBRE, yc=ctx["e"] / 2.0, uV=ctx["e"] / 2.0))
        t_eq = max(v["taux"] for v in v3 if v["aile"] == VERTICALE)
        t_hist = max(v["taux"] for v in verifs if v["aile"] == VERTICALE and v["cle"].startswith("S"))
        garde = {"taux_historique": t_hist, "taux_equilibre_P1": t_eq, "depasse": t_eq > t_hist * (1.0 + 1e-9), "yc": ctx["e"] / 2.0, "uV": ctx["e"] / 2.0}
    res = {"reference": REFERENCE, "corniere": c, "section": ctx["sec"], "configuration": ctx["cfg"], "mode": ctx["mode"], "mode_demande": c["mode"],
           "materiau": {"fy": c["fy"], "nuance": E.nuance_de(c["fy"]), "coef": c["coef"], "etat_limite": ctx["etat"], "gamma_M0": GAMMA_M0,
                        "sigma_adm": ctx["sigma_adm"], "tau_adm": ctx["tau_adm"],
                        "symbole_sigma": "σ_adm" if ctx["etat"] == ELS else "f_y/γ_M0", "symbole_tau": "τ_adm" if ctx["etat"] == ELS else "f_y/(√3·γ_M0)"},
           "geometrie": geo, "combinaisons": ctx["combos"], "efforts": _efforts(ctx, anc), "verifications": verifs,
           "informations": _informations(ctx, anc, geo, comparaison), "ancrages": anc,
           "diagrammes": {HORIZONTALE: _diagramme(ctx, HORIZONTALE, verifs), VERTICALE: _diagramme(ctx, VERTICALE, verifs)} if diagrammes else None,
           "maillage": {HORIZONTALE: _maillage(ctx, HORIZONTALE), VERTICALE: _maillage(ctx, VERTICALE)} if diagrammes else None,
           "maxima": {HORIZONTALE: next(v for v in verifs if v["cle"] == "max_h"), VERTICALE: next(v for v in verifs if v["cle"] == "max_v")},
           "comparaison": comparaison, "garde": garde,
           "taux_max": gov["taux"], "statut": OK if all(v["ok"] for v in verifs if v["verdict"]) else NOK,
           "gouvernant": {"cle": gov["cle"], "nom": gov["nom"], "aile": gov["aile"], "s": gov["s"], "taux": gov["taux"], "combinaison": gov["combinaison"]},
           "calcul": _calcul(ctx, verifs, anc, geo), "hypotheses": _hypotheses(ctx, anc, geo), "avertissements": _avertissements(ctx, anc, garde)}
    return res


def verifier(c: Dict) -> Dict:
    """Le calcul complet, diagrammes et annexes compris."""
    return calculer(c, diagrammes=True, annexes=True)


# ------------------------------------------------------------------ le sélecteur
def candidats(c: Dict, familles=None) -> List[Dict]:
    """Les cornières du catalogue (familles L et LI par défaut), TRIÉES PAR MASSE, chacune calculée avec les charges,
    l'ancrage et les hypothèses de ``c`` ; hors domaine (la charge dans l'angle ou au-delà de l'aile, l'ancrage hors
    de la partie droite…), la raison est donnée et la cornière n'est pas calculée ; ``optimal`` = la plus légère qui
    tient ET dans le domaine ; ``retenue`` = celle de ``c``."""
    c = normaliser(c)
    out = []
    for p in cornieres(familles):
        cp = dict(c, standard=True, profil=p["nom"])
        ligne = {"nom": p["nom"], "designation": designation(p["nom"]), "famille": p["famille"], "A": p["h"], "B": p["b"], "t": p["t"],
                 "r1": p.get("r1") or 0.0, "masse": p["masse"], "aire": p["A"], "taux": None, "ok": False, "domaine": True, "raison": None,
                 "retenue": bool(c["standard"] and p["nom"] == c["profil"]), "optimal": False}
        try:
            r = calculer(cp, diagrammes=False, annexes=False)
            ligne.update(taux=r["taux_max"], ok=r["statut"] == OK)
        except ValueError as err:
            ligne.update(domaine=False, raison=str(err))
        out.append(ligne)
    premier = next((x for x in out if x["ok"] and x["domaine"]), None)
    if premier is not None:
        premier["optimal"] = True
    return out


def optimal(c: Dict, familles=None) -> Optional[Dict]:
    """La cornière la plus légère qui tient et dans le domaine (None si aucune)."""
    return next((x for x in candidats(c, familles) if x["optimal"]), None)

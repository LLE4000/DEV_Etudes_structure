# -*- coding: utf-8 -*-
"""Moteur « Cornières » refondu (acier/corniere) : la méthode du classeur de l'ingénieur (feuille P8), les deux
configurations, les deux modes, l'enveloppe des jeux — décisions docs/acier/CORNIERE_DECISIONS.md § 7. Chaque chiffre
vient de l'analyse docs/acier/corniere_excel/ANALYSE_CORNIERE_EXCEL.md (sorties synthese/sorties/*.txt) ; chaque
contrôle rougit si l'on retire ce qu'il protège (§ 14, mutations).

  1. CAS DE CONTRÔLE (L120×120×15, S355, L = 300, R = 42,76 kN, X = 75, Y = 85, dx = dy = 0, jeu 10, R_ELU = 68,9) :
     M1 = 2,009720, M2 = 2,565600, M3 = 2,358697, M4 = 2,896645 kN·m, 88,8621 % en section 3, N_ELS = V_ELS = 14,2533,
     N_ELU = V_ELU = 22,9667 kN, pression diamétrale 59,389, arrachement 13,575 — à 1e-9 face aux formules de P8
     recalculées ici, et aux chiffres de l'analyse à leur dernière décimale ; les cellules dans les formules.
  2. P8 MOT POUR MOT : les 93 formules de l'annexe A, CODÉES TELLES QU'ELLES SONT ÉCRITES (cellules, unités cm du
     classeur) et évaluées par un petit interpréteur indépendant du moteur, sur ≥ 200 saisies tirées au sort
     (dx, dy, jeu, L, X, Y, R, R_ELU, e, r, f_y, coef, n, d) : dx = dy = 0 d'abord (une seule combinaison, P8 exactement,
     cellule par cellule), puis avec des jeux (l'enveloppe = P8 à Xj = +dx, Yj = +dy pour les sections, à Xj = +dx
     pour les ancrages, que Yj ne touche pas).
  3. L'ENVELOPPE : une combinaison quand dx = dy = 0, trois, neuf ; aucune grandeur ne baisse quand dx ou dy croît
     (balayage, A historique, A équilibre, B) ; la combinaison gouvernante est rendue et juste (ancrage HAUT pour S3
     en A, ancrage BAS pour la traction, en B l'ancrage bas) ; les jeux du tableau § 2.2 (dx = 20 → 112,43 %, dy = 20 →
     95,38 %, les deux → 120,69 %, jeu_ancrage = 20 → 16,446 kN, dy = −20 refusé) ; les diagrammes : les trois courbes
     de moment sont celles des trois positions de la charge (statique recalculée ici) et l'enveloppe est leur maximum
     point par point, l'enveloppe de σ_éq celle des neuf combinaisons.
  4. ÉQUILIBRE A : P0 (pivot à l'arête, u_V = e/2) 76,91 %, H = 33,956 kN ; P1 (pivot e/2) 83,96 % ; Q1 (u_V = 0) 80,84 % ;
     la comparaison des modes ; la GARDE (ancrage bas Y = 40 : avertissement rouge avec les deux chiffres ; Y = 85 : rien).
  5. CONFIGURATION B : H = 82,47 kN, M3 = M4 = 2,886 kN·m constant de l'ancrage à l'angle, N = −R, taux 112,42 %,
     section nette (3 Ø18) 137,1 % qui gouverne, Y = 45 → 38,48 kN, c = 10 → 115,45 kN, jeux (ancrage vers le bout
     libre → traction ×1,4), mode imposé, pression sur le support.
  6. AILES INÉGALES L150×100×12 (r = 12) : 150 horizontal, Y = 65 → taux 1 128,3 %, taux 3 131,1 %, traction 58,31 kN ;
     100 horizontal, Y = 115 → 157,4 %, 30,54 kN.
  7. LE MAXIMUM DU CONGÉ : S1 +2,11 % à 1,98 mm du pied, S3 +1,45 % à 1,66 mm ; en information en historique
     (88,86 % reste le verdict), dans le verdict en équilibre (77,94 % en P0) ; jamais sous une grille fine indépendante.
  8. L_eff : a = 100 → L_eff = 194, σ1 × 1,55, 117,2 % ; a vide → L.
  9. FORMULES : chaque résultat se retrouve à partir de ses seuls termes (table des clés) ; les blocs portent leur
     combinaison ; les cellules en historique.
 10. DOMAINE : les refus (charge dans l'angle, au-delà de l'aile, ancrage dans le congé, hors de l'aile, t ≥ aile,
     nuance, coef, dy < 0…) avec message ; les bornes admises ; -0 ; idempotence ; JSON strict ; entrée intacte ;
     les défauts du § 2 ; les nuances du site (source unique acier/flambement/normes).
 11. RELECTURE V1 : les vraies notes du gel d'avant la refonte (tests/reference_refonte/corniere.json au 10/11,
     copiées ici) s'ouvrent : nom et section gardés, charge et ancrage posés dans les ailes, v1 vrai, avertissement.
 12. SÉLECTEUR : les 233 cornières triées par masse, chacune calculée avec les charges (taux = celui du calcul),
     hors domaine avec la raison, optimal = la plus légère qui tient et dans le domaine, retenue, familles.
 13. PURETÉ : sources (bibliothèque standard, acier.profiles, acier.flambement.normes) ; sous-processus où sys.meta_path
     REFUSE streamlit, pandas, reportlab, matplotlib, scipy, PIL, fitz — sous Python 3.12 et 3.13, résultats identiques.
 14. MUTATIONS : le moteur rechargé depuis ses sources, modifié d'UNE formule clé (ρ₃ sans dy, pivot, N en B, section
     nette sans trous, L_eff, dx oublié, u_V, t(s), σ_éq, τ, bras, ELU, enveloppe, maximum, garde, gouvernant, verdict,
     refus, défaut, sélecteur, v1, diagrammes, positions, nuance), est rejeté ; un témoin ne l'est pas.

    /usr/bin/python3.12 tests/test_corniere_moteur.py
"""
import ast
import copy
import json
import math
import os
import random
import re
import subprocess
import sys
import types

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from acier.corniere import entree as E  # noqa: E402
from acier.corniere import moteur as M  # noqa: E402
from acier.corniere import statique as ST  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{str(info)[:700]}]" if info and not cond else ""))


def pres(a, b, tol=1e-9):
    """Égalité relative (tol), absolue sous 1."""
    return a is not None and b is not None and abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def dec(a, b, nd):
    """a écrit avec nd décimales vaut b (l'analyse cite ses chiffres arrondis)."""
    return abs(a - b) <= 0.5 * 10 ** (-nd) + 1e-12


def lire(*chemin):
    with open(os.path.join(RACINE, *chemin), encoding="utf-8") as fh:
        return fh.read()


def V(res, cle):
    return next(v for v in res["verifications"] if v["cle"] == cle)


def formule(res, bloc, symbole):
    b = next(x for x in res["calcul"] if x["cle"] == bloc)
    return next(f for f in b["formules"] if f["symbole"] == symbole)


# le cas de contrôle du classeur
CAS = dict(M.nouvelle_corniere(), R=42.76, dx=0.0, dy=0.0, R_ELU=68.9)


# ------------------------------------------------------------------ P8 : les 93 formules, telles qu'elles sont écrites
P8 = {
    "D3": "=MAX(I37,I47,I57,I67)",
    "C4": "=IF($B$3=\"NON STANDARD\",K4,VLOOKUP('P8 '!$B$3,CORNIERES!$A$2:$E$59,2,FALSE()))",
    "C5": "=IF($B$3=\"NON STANDARD\",K5,VLOOKUP('P8 '!$B$3,CORNIERES!$A$2:$E$59,3,FALSE()))",
    "C6": "=IF($B$3=\"NON STANDARD\",K6,VLOOKUP('P8 '!$B$3,CORNIERES!$A$2:$E$59,4,FALSE()))",
    "C7": "=IF($B$3=\"NON STANDARD\",K7,VLOOKUP('P8 '!$B$3,CORNIERES!$A$2:$E$59,5,FALSE()))",
    "C19": "=$C$13-$C$7-$C$6", "A20": "=C4-C13", "C20": "=$C$13-$C$6", "A21": "=C5-C16", "L22": "=3.7*0.19*1230",
    "C23": "=$C$16-$C$6-$C$7", "C24": "=C16-$C$6", "C25": "=$C$16-$C$6/2",
    "C31": "=$C$6", "F31": "=C35", "G31": "=F31*10^6/F32", "H31": "=C3/$C$10", "I31": "=G31/H31",
    "C32": "=$C$8*10*C31^2/6", "F32": "=C32", "C33": "=$C$19", "C34": "=C14", "F34": "=$C$12",
    "G34": "=($C$12)*1000/($C$8*10*$C$31)", "H34": "=C3/$C$10/SQRT(3)", "I34": "=G34/H34",
    "C35": "=$C$12*(($C$33+$C$34)/1000)", "G37": "=(G31^2+3*G34^2)^0.5", "H37": "=C3/$C$10", "I37": "=G37/H37",
    "C41": "=$C$6+$C$7", "F41": "=C45", "G41": "=F41*10^6/F42", "H41": "=$C$3/$C$10", "I41": "=G41/H41",
    "C42": "=$C$8*10*C41^2/6", "F42": "=C42", "C43": "=$C$20", "C44": "=C14", "F44": "=$C$12",
    "G44": "=(F44)*1000/(C8*10*C41)", "H44": "=$C$3/$C$10/SQRT(3)", "I44": "=G44/H44",
    "C45": "=$C$12*(($C$43+$C$44)/1000)", "G47": "=(G41^2+3*G44^2)^0.5", "H47": "=$C$3/$C$10", "I47": "=G47/H47",
    "C51": "=$C$6", "F51": "=C57", "G51": "=F51*10^6/F52", "H51": "=$C$3/$C$10", "I51": "=G51/H51",
    "C52": "=$C$8*10*C51^2/6", "F52": "=C52", "C53": "=$C$13", "C54": "=C14", "F54": "=$C$12",
    "G54": "=($F$54)*1000/($C$8*10*$C$51)", "H54": "=$C$3/$C$10/SQRT(3)", "I54": "=G54/H54",
    "C55": "=($C$13+$C$14)/1000*$C$12", "C56": "=($C$23+$C$17)/($C$25+$C$17)", "C57": "=C56*C55",
    "G57": "=(G51^2+3*G54^2)^0.5", "H57": "=$C$3/$C$10", "I57": "=G57/H57",
    "C61": "=$C$6+$C$7", "F61": "=C65", "G61": "=F61*10^6/F62", "H61": "=$C$3/$C$10", "I61": "=G61/H61",
    "C62": "=$C$8*10*C61^2/6", "F62": "=C62", "C63": "=($C$13+$C$14)/1000*$C$12", "C64": "=($C$24+$C$17)/($C$25+$C$17)",
    "F64": "=$C$12", "G64": "=($F$64)*1000/($C$8*10*$C$61)", "H64": "=$C$3/$C$10/SQRT(3)", "I64": "=G64/H64",
    "C65": "=C64*C63", "G67": "=(G61^2+3*G64^2)^0.5", "H67": "=$C$3/$C$10", "I67": "=G67/H67",
    "C70": "=C12", "D73": "=C8", "C75": "=C16/10", "C78": "=($C$13+$C$14)/1000*$C$12",
    "C82": "=C78/(C75*0.01-C76*0.01)/C73", "C83": "=C70/C73", "C85": "=(C71/C70)*C82", "C86": "=(C71/C70)*C83",
    "C88": "=C83*10^3/(C74*C6)", "C90": "=C83*1000/(2*(C5-C16)*C6)",
}
assert len(P8) == 93


def _traduire(fx):
    """Une formule Excel → une expression Python (références → V("…"), IF, VLOOKUP, MAX, SQRT, ^)."""
    s = fx[1:].replace("$", "").replace("'P8 '!", "")
    s = re.sub(r"CORNIERES!([A-Z]+\d+):([A-Z]+\d+)", "CAT", s)
    s = s.replace("FALSE()", "False")
    s = re.sub(r"\b([A-Z]{1,2})(\d+)\b", r'V("\1\2")', s)
    s = s.replace('")="', '")=="')
    s = s.replace("^", "**")
    s = s.replace("IF(", "IF_(").replace("VLOOKUP(", "VLOOKUP_(").replace("MAX(", "max(").replace("SQRT(", "SQRT_(")
    return s


_EXPR = {cle: compile(_traduire(fx), cle, "eval") for cle, fx in P8.items()}


def p8(saisies):
    """Toutes les cellules de P8 pour des saisies {cellule: valeur} (unités DU CLASSEUR : C8 et C76 en cm)."""
    memo = {}

    def V(ref):
        if ref in saisies:
            return saisies[ref]
        if ref in memo:
            return memo[ref]
        if ref not in _EXPR:
            raise KeyError(ref)
        memo[ref] = eval(_EXPR[ref], env)
        return memo[ref]

    def IF_(c, a, b):
        return a if c else b

    def VLOOKUP_(*a):
        return float("nan")                 # la branche jamais retenue (B3 = NON STANDARD : K4..K7) ; Python évalue les deux

    env = {"V": V, "IF_": IF_, "VLOOKUP_": VLOOKUP_, "CAT": None, "SQRT_": math.sqrt, "max": max, "__builtins__": {}}
    cellules = {cle: V(cle) for cle in P8}
    cellules.update({k: v for k, v in saisies.items() if isinstance(v, (int, float))})
    return cellules


def saisies_p8(c, sec, Xj, Yj):
    """Les saisies du classeur pour une cornière du moteur (B3 = NON STANDARD, dimensions dans K4..K7)."""
    jeu = c["jeu_ancrage"]
    return {"B3": "NON STANDARD", "K4": sec["larg"], "K5": sec["haut"], "K6": sec["e"], "K7": sec["r"], "C3": c["fy"], "C8": c["L"] / 10,
            "C10": c["coef"], "C12": c["R"], "C13": c["X"], "C14": Xj, "C16": c["Y"], "C17": Yj,
            "C71": c["R_ELU"] if c["R_ELU"] is not None else 1.5 * c["R"], "C73": c["n"], "C74": c["d"], "C76": jeu / 10}


# cellule de P8 ↔ grandeur du moteur (historique) ; S = section, V = verification, R = résultat
CORRESPONDANCE = [
    ("D3", lambda r: r["taux_max"]), ("C4", lambda r: r["section"]["larg"]), ("C5", lambda r: r["section"]["haut"]),
    ("C6", lambda r: r["section"]["e"]), ("C7", lambda r: r["section"]["r"]),
    ("C19", lambda r: formule(r, "S1", "Xe")["resultat"]), ("C20", lambda r: formule(r, "S2", "Xe+r")["resultat"]),
    ("C23", lambda r: formule(r, "S3", "Ye")["resultat"]), ("C24", lambda r: formule(r, "S4", "Ye+r")["resultat"]),
    ("C25", lambda r: formule(r, "S3", "YH")["resultat"]),
    ("H31", lambda r: r["materiau"]["sigma_adm"]), ("H34", lambda r: r["materiau"]["tau_adm"]),
    ("C55", lambda r: r["efforts"]["Mmax"]), ("C56", lambda r: formule(r, "S3", "ρ₃")["resultat"]), ("C64", lambda r: formule(r, "S4", "ρ₄")["resultat"]),
    ("C78", lambda r: formule(r, "ancrages", "Mmax")["resultat"]),
    ("C82", lambda r: r["ancrages"]["N_ELS"]), ("C83", lambda r: r["ancrages"]["V_ELS"]), ("C85", lambda r: r["ancrages"]["N_ELU"]),
    ("C86", lambda r: r["ancrages"]["V_ELU"]), ("C88", lambda r: r["ancrages"]["pression_diametrale"]), ("C90", lambda r: r["ancrages"]["arrachement"]),
]
for _k, _l0, _M in (("S1", 31, "C35"), ("S2", 41, "C45"), ("S3", 51, "C57"), ("S4", 61, "C65")):
    CORRESPONDANCE += [(f"C{_l0}", lambda r, k=_k: V(r, k)["t"]), (f"C{_l0 + 1}", lambda r, k=_k: V(r, k)["W"]),
                       (_M, lambda r, k=_k: V(r, k)["M"]), (f"G{_l0}", lambda r, k=_k: V(r, k)["sigma"]),
                       (f"I{_l0}", lambda r, k=_k: V(r, k)["taux_sigma"]), (f"G{_l0 + 3}", lambda r, k=_k: V(r, k)["tau"]),
                       (f"I{_l0 + 3}", lambda r, k=_k: V(r, k)["taux_tau"]), (f"G{_l0 + 6}", lambda r, k=_k: V(r, k)["sigma_eq"]),
                       (f"I{_l0 + 6}", lambda r, k=_k: V(r, k)["taux"])]
COPIES = {"F31": "C35", "F32": "C32", "C33": "C19", "F41": "C45", "F42": "C42", "C43": "C20", "F51": "C57", "F52": "C52",
          "F61": "C65", "F62": "C62", "C63": "C55", "C70": "C12", "C53": "C13", "F34": "C12", "F44": "C12", "F54": "C12", "F64": "C12",
          "C34": "C14", "C44": "C14", "C54": "C14"}


def ecarts_p8(mod, c, Xj, Yj, tol=1e-9):
    """Le moteur ``mod`` (historique, enveloppe) face à P8 évalué à (Xj, Yj) : la liste des cellules qui diffèrent."""
    r = mod.calculer(c, diagrammes=False, annexes=False)
    cel = p8(saisies_p8(r["corniere"], r["section"], Xj, Yj))
    out = []
    for ref, lecteur in CORRESPONDANCE:
        if ref == "C90" and lecteur(r) is None:
            continue
        if not pres(lecteur(r), cel[ref], tol):
            out.append(f"{ref} : moteur {lecteur(r)!r} ≠ P8 {cel[ref]!r}")
    return out, cel


# ------------------------------------------------------------------ 1. le cas de contrôle
print("=== 1. Cas de contrôle du classeur (A, historique, dx = dy = 0) ===")
R0 = M.verifier(CAS)
S = {k: V(R0, k) for k in ("S1", "S2", "S3", "S4")}
ecarts, CEL0 = ecarts_p8(M, CAS, 0.0, 0.0)
chk(f"les {len(CORRESPONDANCE)} cellules de P8 qui ont une grandeur dans le moteur : égales à 1e-9 (interpréteur indépendant)", not ecarts, ecarts[:4])
chk("toutes les 93 formules sont évaluées à un nombre, les copies sont des copies, L22 = 864,69",
    all(isinstance(CEL0[k], (int, float)) and math.isfinite(CEL0[k]) for k in P8) and all(CEL0[a] == CEL0[b] for a, b in COPIES.items())
    and pres(CEL0["L22"], 864.69, 1e-12) and CEL0["C33"] == CEL0["C19"] and CEL0["C63"] == CEL0["C78"] == CEL0["C55"])
chk("M1 = 2,009720 ; M2 = 2,565600 ; M3 = 2,358697 ; M4 = 2,896645 kN·m (six décimales de l'analyse)",
    dec(S["S1"]["M"], 2.009720, 6) and dec(S["S2"]["M"], 2.565600, 6) and dec(S["S3"]["M"], 2.358697, 6) and dec(S["S4"]["M"], 2.896645, 6),
    [S[k]["M"] for k in S])
chk("taux 75,8021 / 27,9043 / 88,8621 / 31,4443 % ; taux_max = 88,8621 % en section 3 (statut ok)",
    dec(100 * S["S1"]["taux"], 75.8021, 4) and dec(100 * S["S2"]["taux"], 27.9043, 4) and dec(100 * S["S3"]["taux"], 88.8621, 4)
    and dec(100 * S["S4"]["taux"], 31.4443, 4) and dec(100 * R0["taux_max"], 88.8621, 4) and R0["gouvernant"]["cle"] == "S3" and R0["statut"] == "ok",
    (100 * R0["taux_max"], R0["gouvernant"]))
chk("σ₁ = 178,642 ; τ₁ = 9,502 ; σ_éq,1 = 179,398 ; σ₃ = 209,662 ; σ_éq,3 = 210,307 N/mm² ; σ_adm = 236,667 ; τ_adm = 136,640",
    dec(S["S1"]["sigma"], 178.642, 3) and dec(S["S1"]["tau"], 9.502, 3) and dec(S["S1"]["sigma_eq"], 179.398, 3) and dec(S["S3"]["sigma"], 209.662, 3)
    and dec(S["S3"]["sigma_eq"], 210.307, 3) and dec(R0["materiau"]["sigma_adm"], 236.667, 3) and dec(R0["materiau"]["tau_adm"], 136.640, 3))
A0 = R0["ancrages"]
chk("ancrages : N_ELS = V_ELS = 14,2533 kN ; N_ELU = V_ELU = 22,9667 kN (R_ELU = 68,9 : rapport 1,61132) ; z = 50, 150, 250 mm",
    dec(A0["N_ELS"], 14.2533, 4) and dec(A0["V_ELS"], 14.2533, 4) and dec(A0["N_ELU"], 22.9667, 4) and dec(A0["V_ELU"], 22.9667, 4)
    and dec(A0["rapport_ELU"], 1.61132, 5) and A0["positions_z"] == [50.0, 150.0, 250.0] and A0["bras"] == 75.0, A0)
chk("information : pression diamétrale 59,389 N/mm², arrachement 13,575 N/mm² (sans limite), compression du mur 42,760 kN",
    dec(A0["pression_diametrale"], 59.389, 3) and dec(A0["arrachement"], 13.575, 3) and dec(A0["compression_mur"], 42.760, 3)
    and {i["cle"] for i in R0["informations"]} >= {"pression_diametrale", "arrachement", "compression_mur", "autre_mode"})
chk("H_sections = Mmax/YH = 41,381 kN ; Mmax = 3,2070 kN·m ; ρ₃ = 0,73548 ; ρ₄ = 0,90323",
    dec(R0["efforts"]["H_sections"], 41.381, 3) and dec(R0["efforts"]["Mmax"], 3.2070, 4) and dec(CEL0["C56"], 0.73548, 5) and dec(CEL0["C64"], 0.90323, 5))
cellules = {f["cellule"] for b in R0["calcul"] for f in b["formules"] if f.get("cellule")}
chk("les cellules du classeur sont nommées dans les formules du mode historique (C19, C20, C23, C24, C25, C35, C45, C55, C56, C57, C64, C65, "
    "I37…I67, C82, C83, C85, C86, C88, C90, H31, H34)",
    {"C19", "C20", "C23", "C24", "C25", "C35", "C45", "C55", "C56", "C57", "C64", "C65", "I37", "I47", "I57", "I67", "C82", "C83", "C85", "C86",
     "C88", "C90", "H31", "H34", "C32", "G31", "G34", "G37", "I31", "I34"} <= cellules, sorted(cellules))
chk("une seule combinaison (dx = dy = 0) : « charge à X, ancrage à Y », nominale", len(R0["combinaisons"]) == 1
    and R0["combinaisons"][0]["libelle"] == "charge à X, ancrage à Y" and R0["combinaisons"][0]["position_ancrage"] == "nominal")
chk("mode historique, configuration A, S355, coef 1,5 ; hypothèses dites (contraintes admissibles, P8, sans jeu)",
    R0["mode"] == "historique" and R0["configuration"] == "A" and R0["materiau"]["nuance"] == "S355"
    and any("contraintes admissibles" in h for h in R0["hypotheses"]) and any("P8" in h for h in R0["hypotheses"]) and any("Sans jeu" in h for h in R0["hypotheses"]))


# ------------------------------------------------------------------ 2. P8 sur des saisies au sort
print("\n=== 2. P8 mot pour mot : ≥ 200 saisies tirées au sort ===")
FY = list(E.FY_ADMIS)


def saisie_au_sort(rng, k, avec_jeux):
    """Une cornière dans le domaine : saisie libre (ailes, e, r) ou catalogue, charges et ancrage tirés au sort."""
    for _ in range(200):
        if rng.random() < 0.5:
            p = rng.choice(M.cornieres())
            base = dict(standard=True, profil=p["nom"], aile_horizontale=rng.choice(("A", "B")))
            larg, haut = (p["h"], p["b"]) if base["aile_horizontale"] == "A" else (p["b"], p["h"])
            e, r = p["t"], p.get("r1") or 0.0
        else:
            e = round(rng.uniform(4.0, 25.0), 1)
            r = round(rng.uniform(0.0, 1.4 * e), 1)
            larg, haut = round(rng.uniform(3 * (e + r) + 20, 250), 0), round(rng.uniform(3 * (e + r) + 20, 250), 0)
            base = dict(standard=False, A=larg, B=haut, t=e, r=r, aile_horizontale="A")
        d = rng.choice((8.0, 10.0, 12.0, 16.0, 20.0))
        dx = round(rng.uniform(0.0, 15.0), 1) if avec_jeux else 0.0
        dy = round(rng.uniform(0.0, 10.0), 1) if avec_jeux else 0.0
        X_lo, X_hi = e + r + dx + 1.0, larg - dx
        Y_lo, Y_hi = e + r + (d + 2) / 2 + dy, haut - (d + 2) / 2 - dy
        if X_hi <= X_lo or Y_hi <= Y_lo:
            continue
        R = round(rng.uniform(1.0, 150.0), 2)
        L = round(rng.uniform(100.0, 600.0), 1)
        n = rng.choice((1, 2, 3, 4, 5))
        if n * (d + 2) >= L:
            continue
        Y = round(rng.uniform(Y_lo, Y_hi), 1)
        c = dict(M.nouvelle_corniere(f"Tirage {k}"), **base, fy=rng.choice(FY), coef=round(rng.uniform(1.0, 2.0), 2), R=R, L=L,
                 X=round(rng.uniform(X_lo, X_hi), 1), dx=dx, R_ELU=round(rng.uniform(R, 2 * R), 2), Y=Y, dy=dy, n=n, d=d,
                 jeu_ancrage=0.0 if k % 10 == 0 else round(rng.uniform(0.0, max(0.0, Y - 5.0)), 1), mode="historique", configuration="A")
        try:
            return M.normaliser(c)
        except ValueError:
            continue
    raise AssertionError("pas de saisie dans le domaine")


RNG = random.Random(20261110)
TIRAGES_SANS = [saisie_au_sort(RNG, k, False) for k in range(1, 111)]
TIRAGES_AVEC = [saisie_au_sort(RNG, k, True) for k in range(111, 241)]
fautes = []
for c in TIRAGES_SANS:
    ec, _ = ecarts_p8(M, c, 0.0, 0.0)
    fautes += [f"{c['nom']} {x}" for x in ec[:1]]
chk(f"{len(TIRAGES_SANS)} saisies sans jeu : chaque cellule de P8 (sections, ancrages, D3) = le moteur à 1e-9 (une seule combinaison)",
    not fautes, fautes[:4])
chk("ces saisies couvrent catalogue et saisie libre, les deux ailes horizontales, 1 à 5 ancrages, les quatre nuances, des jeux d'ancrage nuls",
    {c["standard"] for c in TIRAGES_SANS} == {True, False} and {c["aile_horizontale"] for c in TIRAGES_SANS} == {"A", "B"}
    and {c["n"] for c in TIRAGES_SANS} >= {1, 2, 3, 4, 5} and {c["fy"] for c in TIRAGES_SANS} == set(FY) and any(c["jeu_ancrage"] == 0 for c in TIRAGES_SANS))
fautes = []
for c in TIRAGES_AVEC:
    ec, _ = ecarts_p8(M, c, c["dx"], c["dy"])
    fautes += [f"{c['nom']} {x}" for x in ec[:1]]
chk(f"{len(TIRAGES_AVEC)} saisies avec jeux (dx jusqu'à 15, dy jusqu'à 10) : l'enveloppe = P8 à Xj = +dx, Yj = +dy (sections), Xj = +dx (ancrages)",
    not fautes and sum(1 for c in TIRAGES_AVEC if c["dx"] > 0 and c["dy"] > 0) >= 100, fautes[:4])
chk(f"{len(TIRAGES_SANS) + len(TIRAGES_AVEC)} saisies en tout : au moins 200", len(TIRAGES_SANS) + len(TIRAGES_AVEC) >= 200)
# avec jeux, l'ancrage de P8 ne lit pas Yj : l'enveloppe des ancrages ne change pas avec dy
c = TIRAGES_AVEC[0]
ra, rb = M.calculer(dict(c, dy=0.0), diagrammes=False), M.calculer(dict(c, dy=10.0), diagrammes=False)
chk("les ancrages du classeur (C82…C90) ne lisent pas Yj : dy ne change ni N_ELS, ni V_ELS, ni la pression, ni l'arrachement en historique",
    ra["ancrages"]["N_ELS"] == rb["ancrages"]["N_ELS"] and ra["ancrages"]["arrachement"] == rb["ancrages"]["arrachement"]
    and ra["ancrages"]["pression_diametrale"] == rb["ancrages"]["pression_diametrale"])


# ------------------------------------------------------------------ 3. l'enveloppe
print("\n=== 3. L'enveloppe des neuf combinaisons ===")
chk("combinaisons : 1 (dx = dy = 0), 3 (dx seul), 3 (dy seul), 9 (les deux), la nominale d'abord",
    len(M.combinaisons(CAS)) == 1 and len(M.combinaisons(dict(CAS, dx=20.0))) == 3 and len(M.combinaisons(dict(CAS, dy=10.0))) == 3
    and len(M.combinaisons(dict(CAS, dx=20.0, dy=10.0))) == 9 and M.combinaisons(dict(CAS, dx=20.0, dy=10.0))[0]["libelle"] == "charge à X, ancrage à Y")
cb9 = M.combinaisons(dict(CAS, dx=20.0, dy=10.0))
chk("chaque combinaison porte X′, Y′, son libellé et ses positions (A : Y + dy = ancrage HAUT ; charge X + dx = loin)",
    {(x["jeu_charge"], x["jeu_ancrage"]) for x in cb9} == {(a, b) for a in (-20.0, 0.0, 20.0) for b in (-10.0, 0.0, 10.0)}
    and all(x["X"] == 75 + x["jeu_charge"] and x["Y"] == 85 + x["jeu_ancrage"] for x in cb9)
    and next(x for x in cb9 if x["jeu_charge"] == 20 and x["jeu_ancrage"] == 10)["libelle"] == "charge à X + 20, ancrage à Y + 10"
    and next(x for x in cb9 if x["jeu_ancrage"] == 10)["position_ancrage"] == "haut" and next(x for x in cb9 if x["jeu_charge"] == 20)["position_charge"] == "loin")
cbB = M.combinaisons(dict(CAS, configuration="B", dy=10.0))
chk("en B, Y + dy est l'ancrage BAS (vers le bout libre) et Y − dy le haut", next(x for x in cbB if x["jeu_ancrage"] == 10)["position_ancrage"] == "bas"
    and next(x for x in cbB if x["jeu_ancrage"] == -10)["position_ancrage"] == "haut")
J = {lib: M.verifier(dict(CAS, **kw)) for lib, kw in (("dx20", dict(dx=20.0)), ("dy20", dict(dy=20.0)), ("dxdy20", dict(dx=20.0, dy=20.0)),
                                                       ("jeu20", dict(jeu_ancrage=20.0)))}
chk("jeux du § 2.2 : dx = 20 → 112,43 % (M1 2,86492, M3 2,98768, traction 18,054 kN) ; dy = 20 → 95,38 % (M3 2,53271, traction inchangée) ; "
    "les deux → 120,69 % (M3 3,20810) ; jeu_ancrage = 20 → 16,446 kN (taux inchangé)",
    dec(100 * J["dx20"]["taux_max"], 112.43, 2) and dec(V(J["dx20"], "S1")["M"], 2.86492, 5) and dec(V(J["dx20"], "S3")["M"], 2.98768, 5)
    and dec(J["dx20"]["ancrages"]["N_ELS"], 18.054, 3)
    and dec(100 * J["dy20"]["taux_max"], 95.38, 2) and dec(V(J["dy20"], "S3")["M"], 2.53271, 5) and dec(J["dy20"]["ancrages"]["N_ELS"], 14.253, 3)
    and dec(100 * J["dxdy20"]["taux_max"], 120.69, 2) and dec(V(J["dxdy20"], "S3")["M"], 3.20810, 5)
    and dec(J["jeu20"]["ancrages"]["N_ELS"], 16.446, 3) and dec(100 * J["jeu20"]["taux_max"], 88.86, 2),
    {k: (100 * r["taux_max"], r["ancrages"]["N_ELS"]) for k, r in J.items()})
chk("la combinaison gouvernante est rendue et juste : S3 → charge loin + ancrage HAUT ; S1 → charge loin, ancrage nominal (indifférent) ; "
    "traction → charge loin, ancrage nominal (les ancrages du classeur ne lisent pas Yj)",
    V(J["dxdy20"], "S3")["combinaison"]["libelle"] == "charge à X + 20, ancrage à Y + 20" and V(J["dxdy20"], "S3")["combinaison"]["position_ancrage"] == "haut"
    and V(J["dxdy20"], "S1")["combinaison"]["libelle"] == "charge à X + 20, ancrage à Y"
    and J["dxdy20"]["ancrages"]["combinaison_N"]["libelle"] == "charge à X + 20, ancrage à Y")
try:
    M.normaliser(dict(CAS, dy=-20.0))
    chk("dy = −20 est refusé (dy ≥ 0 : le jeu joue dans les deux sens par l'enveloppe)", False)
except ValueError as e:
    chk("dy = −20 est refusé (dy ≥ 0 : le jeu joue dans les deux sens par l'enveloppe)", "dy" in str(e), e)


def grandeurs(r):
    """Les grandeurs vérifiées qui ne doivent pas baisser quand un jeu croît."""
    g = {v["cle"]: v["taux"] for v in r["verifications"]}
    g.update(N_ELS=r["ancrages"]["N_ELS"], H=r["ancrages"]["H"], Mmax=r["efforts"]["Mmax"], taux_max=r["taux_max"])
    if r["efforts"]["M_angle"] is not None:
        g["M_angle"] = r["efforts"]["M_angle"]
    return g


def balayage(base, cle, valeurs):
    """Les grandeurs qui BAISSENT (au-delà de 1e-12) quand ``cle`` croît."""
    prev, baisses = None, []
    for x in valeurs:
        g = grandeurs(M.calculer(dict(base, **{cle: x}), diagrammes=False, annexes=False))
        if prev is not None:
            baisses += [f"{cle} = {x} : {k} {prev[k]} → {g[k]}" for k in g if g[k] < prev[k] * (1 - 1e-12) - 1e-12]
        prev = g
    return baisses


baisses = []
for lib, base in (("A historique", CAS), ("A équilibre", dict(CAS, mode="equilibre")), ("B", dict(CAS, configuration="B")),
                  ("A équilibre, u_V = 0, y_c = 5", dict(CAS, mode="equilibre", uV=0.0, yc=5.0))):
    baisses += [f"{lib} : {x}" for x in balayage(base, "dx", [0.0, 5.0, 10.0, 15.0, 20.0])]
    baisses += [f"{lib} : {x}" for x in balayage(base, "dy", [0.0, 2.5, 5.0, 7.5, 10.0])]
    baisses += [f"{lib} : {x}" for x in balayage(dict(base, dx=10.0), "dy", [0.0, 5.0, 10.0])]
chk("balayage : aucune grandeur vérifiée (S1…S4, maxima, nette, N_ELS, H, Mmax, M_angle, taux_max) ne baisse quand dx ou dy croît — "
    "A historique, A équilibre (y_c = 0 et y_c = 5, u_V = 0), B", not baisses, baisses[:5])


# les diagrammes : statique indépendante
def M_attendu(r, cb, aile, s):
    """Le moment (kN·m, signé : + = face côté congé tendue) de la combinaison ``cb`` à l'abscisse s, recalculé ici."""
    c, sec, e, r_ = r["corniere"], r["section"], r["section"]["e"], r["section"]["r"]
    R, Xp, Yp = c["R"], cb["X"], cb["Y"]
    uV = c["uV"] if c["uV"] is not None else e / 2
    if aile == "horizontale":
        return (1 if r["configuration"] == "A" else -1) * R * (Xp - s) / 1e3 if s <= Xp else 0.0
    if r["configuration"] == "B":
        ya, cc = sec["haut"] - Yp, c["yc"]
        H = R * (Xp - uV) / (ya - cc)
        if s <= Yp:
            return -R * (Xp - e / 2) / 1e3
        return -H * (sec["haut"] - cc - s) / 1e3 if s <= sec["haut"] - cc else 0.0
    if s >= Yp:
        return 0.0
    if r["mode"] == "historique":
        Hs = R * Xp / (c["Y"] - e / 2 + (Yp - c["Y"]))
        return Hs * (Yp - s) / 1e3
    H = R * (Xp - uV) / (Yp - c["yc"])
    return (H * (Yp - s) - R * (e / 2 - uV)) / 1e3


def seq_attendu(r, cb, aile, s):
    """σ_éq d'une combinaison à s, recalculée ici (N et V de la statique, t(s) du congé, largeur L ou L_eff)."""
    c, sec, e, r_ = r["corniere"], r["section"], r["section"]["e"], r["section"]["r"]
    R, Xp, Yp = c["R"], cb["X"], cb["Y"]
    uV = c["uV"] if c["uV"] is not None else e / 2
    Mi = M_attendu(r, cb, aile, s) * 1e3
    if aile == "horizontale":
        N, Vv = 0.0, (R if s <= Xp else 0.0)
        Lw = min(c["L"], c["a"] + 2 * (Xp - r_ - e)) if c["a"] is not None else c["L"]
    else:
        Lw = c["L"]
        if r["configuration"] == "B":
            H = R * (Xp - uV) / (sec["haut"] - Yp - c["yc"])
            N, Vv = (-R, 0.0) if s <= Yp else ((0.0, H) if s <= sec["haut"] - c["yc"] else (0.0, 0.0))
        elif r["mode"] == "historique":
            N, Vv = 0.0, (R if s < Yp else 0.0)
        else:
            H = R * (Xp - uV) / (Yp - c["yc"])
            N, Vv = (R, H) if s < Yp else (0.0, 0.0)
    t = e + r_ - math.sqrt(max(r_ * r_ - (e + r_ - s) ** 2, 0.0)) if s < e + r_ else e
    W, A = Lw * t * t / 6, Lw * t
    sig = abs(Mi) * 1e3 / W + abs(N) * 1e3 / A
    tau = abs(Vv) * 1e3 / A
    return math.sqrt(sig * sig + 3 * tau * tau)


def fautes_diagrammes(r):
    """Les écarts entre les diagrammes rendus et la statique recalculée : trois courbes (une par charge, à l'ancrage
    qui gouverne), enveloppe = max point par point des courbes et des neuf combinaisons."""
    out = []
    for aile in ("horizontale", "verticale"):
        d = r["diagrammes"][aile]
        jeux_charge = sorted(set(cb["jeu_charge"] for cb in r["combinaisons"]))
        if [cc["jeu_charge"] for cc in d["courbes"]] != jeux_charge:
            out.append(f"{aile} : {len(d['courbes'])} courbes pour {len(jeux_charge)} positions de la charge")
            continue
        for cc in d["courbes"]:
            cbs = [cb for cb in r["combinaisons"] if cb["jeu_charge"] == cc["jeu_charge"]]
            for s, Mc in cc["points"]:
                attendu = max((M_attendu(r, cb, aile, s) for cb in cbs), key=abs)
                if not pres(abs(Mc), abs(attendu), 1e-9):
                    out.append(f"{aile} courbe Δx = {cc['jeu_charge']} s = {s} : {Mc} ≠ {attendu}")
                    break
            if aile == "verticale" and cc["ancrage"] is not None:
                cb_g = next(cb for cb in cbs if cb["jeu_ancrage"] == cc["ancrage"])
                s_ref = r["section"]["e"] + r["section"]["r"]
                if any(abs(M_attendu(r, cb, aile, s_ref)) > abs(M_attendu(r, cb_g, aile, s_ref)) * (1 + 1e-9) for cb in cbs):
                    out.append(f"{aile} courbe Δx = {cc['jeu_charge']} : l'ancrage rendu ({cc['ancrage']}) ne gouverne pas S3")
        for i, pt in enumerate(d["points"]):
            s = pt["s"]
            Mmax_courbes = max((cc["points"][i][1] for cc in d["courbes"]), key=abs)
            if not pres(abs(pt["M"]), abs(Mmax_courbes), 1e-9) or any(cc["points"][i][0] != s for cc in d["courbes"]):
                out.append(f"{aile} s = {s} : enveloppe M {pt['M']} ≠ max des courbes {Mmax_courbes}")
                break
            seq = max(seq_attendu(r, cb, aile, s) for cb in r["combinaisons"])
            if not pres(pt["sigma_eq"], seq, 1e-9) or not pres(pt["taux"], seq / r["materiau"]["sigma_adm"], 1e-9):
                out.append(f"{aile} s = {s} : enveloppe σ_éq {pt['sigma_eq']} ≠ {seq}")
                break
            if not pres(pt["t"], ST.t_conge(s, r["section"]["e"], r["section"]["r"]), 1e-12):
                out.append(f"{aile} s = {s} : t")
                break
    return out


DIAG = {lib: M.verifier(dict(CAS, **kw)) for lib, kw in (("A hist 20/10", dict(dx=20.0, dy=10.0)), ("A éq 20/10", dict(mode="equilibre", dx=20.0, dy=10.0)),
                                                          ("B 20/10", dict(configuration="B", dx=20.0, dy=10.0)), ("A hist 0/0", {}),
                                                          ("A éq u_V = 0", dict(mode="equilibre", uV=0.0, dx=10.0, dy=5.0)),
                                                          ("B c = 10", dict(configuration="B", yc=10.0, dx=5.0, dy=5.0, a=120.0)))}
fd = [f"{lib} : {x}" for lib, r in DIAG.items() for x in fautes_diagrammes(r)[:2]]
chk("diagrammes : une courbe de moment par position de la charge (la statique recalculée ici, à l'ancrage qui gouverne S3 — nommé), "
    "l'enveloppe = le maximum point par point des courbes, σ_éq = le maximum des neuf combinaisons, t(s) du congé — A historique, A équilibre "
    "(u_V = 0 aussi), B (c = 10, a = 120 aussi)", not fd, fd[:4])
d = DIAG["A hist 20/10"]["diagrammes"]["verticale"]
chk("diagrammes : repères (sections, congé, bloc d'angle, charge et ses trois positions, ancrage et sa bande, maximum), tronçons N/V, "
    "les trois courbes de l'aile verticale prennent l'ancrage haut (Δy = +10)",
    d["reperes"]["ancrage"]["bande"] == [75.0, 95.0] and d["reperes"]["ancrage"]["positions"] == [75.0, 85.0, 95.0]
    and DIAG["A hist 20/10"]["diagrammes"]["horizontale"]["reperes"]["charge"]["positions"] == [55.0, 75.0, 95.0]
    and d["reperes"]["conge"] == [15.0, 28.0] and d["reperes"]["bloc_angle"] == [0.0, 15.0] and [x["cle"] for x in d["reperes"]["sections"]] == ["S3", "S4"]
    and all(cc["ancrage"] == 10.0 for cc in d["courbes"]) and d["troncons"] == [{"de": 15.0, "a": 95.0, "N": 0.0, "V": 42.76}]
    and pres(d["reperes"]["maximum"]["taux"], V(DIAG["A hist 20/10"], "max_v")["taux"]), d["reperes"])
dB = DIAG["B 20/10"]["diagrammes"]["verticale"]
chk("B : à la section 3 le moment ne dépend pas de l'ancrage (égalité → l'ancrage nominal est nommé), sous l'ancrage la courbe suit l'ancrage bas ; "
    "tronçons N = −R puis V = H ; contact repéré au bout libre (s = haut)",
    all(cc["ancrage"] == 0.0 for cc in dB["courbes"]) and dB["troncons"][0]["N"] == -42.76 and dB["troncons"][1]["V"] > 0 and dB["reperes"]["contact"] == 120.0
    and all(pres(abs(M_), abs(M_attendu(DIAG["B 20/10"], next(cb for cb in DIAG["B 20/10"]["combinaisons"] if cb["jeu_charge"] == cc["jeu_charge"]
                                                                  and cb["jeu_ancrage"] == 10.0), "verticale", s_)))
            for cc in dB["courbes"] for s_, M_ in cc["points"] if 95.0 < s_ < 120.0))
chk("sans jeu, une seule courbe, égale à l'enveloppe", len(DIAG["A hist 0/0"]["diagrammes"]["horizontale"]["courbes"]) == 1
    and all(pres(abs(p["M"]), abs(q[1])) for p, q in zip(DIAG["A hist 0/0"]["diagrammes"]["horizontale"]["points"],
                                                        DIAG["A hist 0/0"]["diagrammes"]["horizontale"]["courbes"][0]["points"])))


# ------------------------------------------------------------------ 4. mode équilibre, A
print("\n=== 4. Mode équilibre (A), comparaison, garde ===")
P0 = M.verifier(dict(CAS, mode="equilibre", yc=0.0))
P1 = M.verifier(dict(CAS, mode="equilibre", yc=7.5))
Q1 = M.verifier(dict(CAS, mode="equilibre", yc=7.5, uV=0.0))
chk("P0 (pivot à l'arête, u_V = e/2) : H = 33,956 kN, M3 = 1,93552, M4 = 2,37695 kN·m, taux 3 = 76,91 % (σ = M/W + N/A, τ = H/A), N/ancrage 11,319 kN",
    dec(P0["ancrages"]["H"], 33.956, 3) and dec(V(P0, "S3")["M"], 1.93552, 5) and dec(V(P0, "S4")["M"], 2.37695, 5) and dec(100 * V(P0, "S3")["taux"], 76.91, 2)
    and dec(P0["ancrages"]["N_ELS"], 11.319, 3) and V(P0, "S3")["N"] == 42.76 and pres(V(P0, "S3")["V"], P0["ancrages"]["H"]),
    (P0["ancrages"]["H"], V(P0, "S3")))
chk("P1 (pivot e/2) : H = 37,243 kN, M3 = 2,12283, taux 3 = 83,96 % ; Q1 (pivot e/2, u_V = 0) : H = 41,381, M3 = 2,03800, taux 3 = 80,84 %",
    dec(P1["ancrages"]["H"], 37.243, 3) and dec(V(P1, "S3")["M"], 2.12283, 5) and dec(100 * V(P1, "S3")["taux"], 83.96, 2)
    and dec(Q1["ancrages"]["H"], 41.381, 3) and dec(V(Q1, "S3")["M"], 2.03800, 5) and dec(100 * V(Q1, "S3")["taux"], 80.84, 2),
    (P1["ancrages"]["H"], V(P1, "S3")["taux"], Q1["ancrages"]["H"], V(Q1, "S3")["M"], V(Q1, "S3")["taux"]))
chk("en équilibre, les maxima du congé entrent dans le verdict : P0 → 77,94 % (maximum de l'aile verticale, à 1,61 mm du pied) ; "
    "les formules sont sans cellule", dec(100 * P0["taux_max"], 77.94, 2) and P0["gouvernant"]["cle"] == "max_v" and V(P0, "max_v")["verdict"]
    and dec(V(P0, "max_v")["recul"], 1.61, 2) and not any(f.get("cellule") for b in P0["calcul"] for f in b["formules"]))
chk("comparaison des modes (information) : en historique le taux de l'équilibre (77,94 %), en équilibre celui de l'historique (88,86 %)",
    R0["comparaison"]["mode"] == "equilibre" and dec(100 * R0["comparaison"]["taux_max"], 77.94, 2)
    and P0["comparaison"]["mode"] == "historique" and dec(100 * P0["comparaison"]["taux_max"], 88.86, 2)
    and next(i for i in R0["informations"] if i["cle"] == "autre_mode")["valeur"] == R0["comparaison"]["taux_max"])
chk("garde du mode historique : équilibre P1 (83,96 %) sous l'historique (88,86 %) sur le cas de contrôle → pas d'avertissement",
    R0["garde"]["depasse"] is False and dec(100 * R0["garde"]["taux_equilibre_P1"], 83.96, 2) and dec(100 * R0["garde"]["taux_historique"], 88.86, 2)
    and not any(a["code"] == "equilibre_plus_severe" for a in R0["avertissements"]) and R0["garde"]["yc"] == 7.5 and R0["garde"]["uV"] == 7.5)
RG = M.verifier(dict(CAS, Y=40.0))
ag = next((a for a in RG["avertissements"] if a["code"] == "equilibre_plus_severe"), None)
chk("ancrage bas (Y = 40 < 43,5 mm) : l'équilibre P1 dépasse l'historique → avertissement ROUGE avec les deux chiffres",
    RG["garde"]["depasse"] and ag is not None and ag["niveau"] == "rouge" and f"{100 * RG['garde']['taux_equilibre_P1']:.1f}" in ag["message"]
    and f"{100 * RG['garde']['taux_historique']:.1f}" in ag["message"] and RG["avertissements"][0] is ag, (RG["garde"], ag))
chk("Y = 45 : l'historique reste du côté sûr (seuil P1 à 43,5 mm)", M.verifier(dict(CAS, Y=45.0))["garde"]["depasse"] is False)
chk("le mode équilibre n'a ni garde ni cellule ; ses hypothèses nomment y_c et u_V", P0["garde"] is None and any("y_c = 0" in h and "u_V = 7,5" in h for h in P0["hypotheses"]))


# ------------------------------------------------------------------ 5. configuration B
print("\n=== 5. Configuration B (aile horizontale en haut) ===")
B = M.verifier(dict(CAS, configuration="B"))
vB = {v["cle"]: v for v in B["verifications"]}
chk("B : H = 82,47 kN (bras 35 mm), M3 = M4 = 2,886 kN·m, N = −R, V = 0 entre l'ancrage et l'angle, σ = 256,56 + 9,50 = 266,06, taux 3 = 112,42 %",
    dec(B["ancrages"]["H"], 82.47, 2) and B["ancrages"]["bras"] == 35.0 and dec(abs(vB["S3"]["M"]), 2.886, 3) and dec(abs(vB["S4"]["M"]), 2.886, 3)
    and vB["S3"]["N"] == -42.76 and vB["S3"]["V"] == 0.0 and dec(vB["S3"]["sigma"], 266.06, 2) and dec(100 * vB["S3"]["taux"], 112.42, 2)
    and vB["S3"]["M"] < 0 and vB["S1"]["M"] < 0, (B["ancrages"]["H"], vB["S3"]))
chk("B : section nette (3 trous Ø18 sur 300) 137,1 %, face du dessus du trou (M = R·(X′ − e/2), N = −R), elle GOUVERNE ; statut nok",
    dec(100 * vB["nette"]["taux"], 137.1, 1) and vB["nette"]["face"] == "dessus" and vB["nette"]["largeur"] == 246.0 and B["gouvernant"]["cle"] == "nette"
    and B["statut"] == "nok" and vB["nette"]["s"] == 85.0, vB["nette"])
chk("B : le moment est constant de l'ancrage à l'angle (statique de la partie au-dessus de la coupe) et ne dépend pas de l'ancrage : Y = 45 et Y = 105 → même M3",
    pres(abs(V(M.verifier(dict(CAS, configuration="B", Y=45.0)), "S3")["M"]), abs(vB["S3"]["M"]))
    and pres(abs(V(M.verifier(dict(CAS, configuration="B", Y=105.0)), "S3")["M"]), abs(vB["S3"]["M"])))
chk("B : Y = 45 → H = 38,48 kN ; c = 10 (u_V = 7,5) → 115,45 kN ; u_V = 0 → 91,63 kN ; ancrage à Y + 10 (vers le bout libre) → 115,45, Y − 10 → 64,14",
    dec(M.verifier(dict(CAS, configuration="B", Y=45.0))["ancrages"]["H"], 38.48, 2) and dec(M.verifier(dict(CAS, configuration="B", yc=10.0))["ancrages"]["H"], 115.45, 2)
    and dec(M.verifier(dict(CAS, configuration="B", uV=0.0))["ancrages"]["H"], 91.63, 2) and dec(M.verifier(dict(CAS, configuration="B", Y=95.0))["ancrages"]["H"], 115.45, 2)
    and dec(M.verifier(dict(CAS, configuration="B", Y=75.0))["ancrages"]["H"], 64.14, 2))
BJ = M.verifier(dict(CAS, configuration="B", dy=10.0, dx=20.0))
chk("B avec jeux : la traction est gouvernée par l'ancrage BAS et la charge loin (115,45 × (95 − 7,5)/(75 − 7,5) = 149,6 kN) ; "
    "les sections par la charge loin, ancrage nominal (indifférent) ; dx = 20 → taux 3 144,54 %",
    BJ["ancrages"]["combinaison_N"]["position_ancrage"] == "bas" and BJ["ancrages"]["combinaison_N"]["position_charge"] == "loin"
    and dec(BJ["ancrages"]["H"], 42.76 * 87.5 / 25.0, 2) and V(BJ, "S3")["combinaison"]["libelle"] == "charge à X + 20, ancrage à Y"
    and dec(100 * V(BJ, "S3")["taux"], 144.54, 2), (BJ["ancrages"], V(BJ, "S3")["taux"]))
chk("B : mode équilibre imposé (historique demandé → avertissement mode_impose_B), pression sur le support C/L = 274,9 N/mm en information et en avertissement",
    B["mode"] == "equilibre" and B["mode_demande"] == "historique" and any(a["code"] == "mode_impose_B" for a in B["avertissements"])
    and dec(B["ancrages"]["pression_support"], 274.9, 1) and any(a["code"] == "pression_support" and a["niveau"] == "attention" for a in B["avertissements"])
    and next(i for i in B["informations"] if i["cle"] == "pression_support")["valeur"] == B["ancrages"]["pression_support"] and B["comparaison"] is None)
chk("B : l'arrachement se compte vers le bout libre (haut − Y′, ancrage le plus bas) ; M_angle rendu dans les efforts",
    pres(B["ancrages"]["arrachement"], 14.253333333333334 * 1e3 / (2 * 35 * 15)) and pres(BJ["ancrages"]["arrachement"], 14.253333333333334 * 1e3 / (2 * 25 * 15))
    and dec(B["efforts"]["M_angle"], 2.886, 3) and R0["efforts"]["M_angle"] is None)


# ------------------------------------------------------------------ 6. ailes inégales
print("\n=== 6. Ailes inégales (L 150x100x12, r1 = 12) ===")
I1 = M.verifier(dict(CAS, profil="L 150x100x12", aile_horizontale="A", Y=65.0))
I2 = M.verifier(dict(CAS, profil="L 150x100x12", aile_horizontale="B", Y=115.0))
chk("150 horizontal × 100 vertical, Y = 65 : taux 1 = 128,3 %, taux 3 = 131,1 %, traction totale 58,31 kN ; la fiche dit larg = 150, haut = 100, r = 12",
    dec(100 * V(I1, "S1")["taux"], 128.3, 1) and dec(100 * V(I1, "S3")["taux"], 131.1, 1) and dec(I1["ancrages"]["H"], 58.31, 2)
    and (I1["section"]["larg"], I1["section"]["haut"], I1["section"]["r"]) == (150.0, 100.0, 12.0), (V(I1, "S1")["taux"], V(I1, "S3")["taux"], I1["ancrages"]["H"]))
chk("100 horizontal × 150 vertical, Y = 115 : taux 1 = 128,3 %, taux 3 = 157,4 %, traction 30,54 kN (statut nok)",
    dec(100 * V(I2, "S1")["taux"], 128.3, 1) and dec(100 * V(I2, "S3")["taux"], 157.4, 1) and dec(I2["ancrages"]["H"], 30.54, 2) and I2["statut"] == "nok"
    and (I2["section"]["larg"], I2["section"]["haut"]) == (100.0, 150.0))
chk("l'orientation est un paramètre : la même désignation, les deux orientations, deux calculs différents ; le catalogue reste la source (r1 = 12, pas 6)",
    V(I1, "S3")["taux"] != V(I2, "S3")["taux"] and M.section_catalogue("L 150x100x12")["r1"] == 12.0 and I1["section"]["r2"] == 6.0)


# ------------------------------------------------------------------ 7. le maximum du congé
print("\n=== 7. Le maximum le long du congé ===")
mh, mv = V(R0, "max_h"), V(R0, "max_v")
chk("aile horizontale : maximum à 1,98 mm avant le pied du congé (s = 26,02), +2,11 % sur la section 1, 77,40 % ; t = 15,152",
    dec(mh["recul"], 1.98, 2) and dec(mh["s"], 26.02, 2) and dec(100 * mh["ecart"], 2.11, 2) and dec(100 * mh["taux"], 77.40, 2) and dec(mh["t"], 15.152, 3)
    and mh["reference"] == "S1" and mh["dans_conge"], mh)
chk("aile verticale (méthode du classeur) : maximum à 1,66 mm du pied (s = 26,34), +1,45 % sur la section 3, 90,15 %, M = 2,427 kN·m, t = 15,106",
    dec(mv["recul"], 1.66, 2) and dec(mv["s"], 26.34, 2) and dec(100 * mv["ecart"], 1.45, 2) and dec(100 * mv["taux"], 90.15, 2) and dec(mv["M"], 2.427, 3)
    and dec(mv["t"], 15.106, 3), mv)
chk("en historique les maxima sont en INFORMATION (verdict faux) : le verdict reste 88,8621 % ; maxima rendus aussi sous « maxima »",
    not mh["verdict"] and not mv["verdict"] and R0["maxima"]["horizontale"] is mh and R0["maxima"]["verticale"] is mv)


def max_grille_fine(r, aile, pas=0.002):
    """Le maximum de σ_éq sur une grille fine indépendante, sur tout le congé et la partie droite (coudes compris)."""
    sec, c = r["section"], r["corniere"]
    e, rr = sec["e"], sec["r"]
    fin = sec["larg"] if aile == "horizontale" else sec["haut"]
    pts = [e + i * pas for i in range(int(rr / pas) + 1)] + [e + rr + i * 0.5 for i in range(int((fin - e - rr) / 0.5) + 1)]
    pts += [cb["X" if aile == "horizontale" else "Y"] + d_ for cb in r["combinaisons"] for d_ in (-1e-9, 0.0, 1e-9)]
    pts = [p for p in pts if e <= p <= fin]
    return max(max(seq_attendu(r, cb, aile, s) for cb in r["combinaisons"]) for s in pts)


fautes = []
for lib, r in list(DIAG.items()) + [("P0", P0), ("B", B), ("I2", I2)]:
    for aile in ("horizontale", "verticale"):
        vmax = V(r, "max_h" if aile == "horizontale" else "max_v")
        fin = max_grille_fine(r, aile)
        if vmax["sigma_eq"] < fin * (1 - 1e-7):
            fautes.append(f"{lib} {aile} : {vmax['sigma_eq']} < grille fine {fin}")
chk("le maximum rendu n'est jamais sous celui d'une grille fine indépendante (pas 0,002 mm dans le congé, coudes compris) — six cas, deux ailes",
    not fautes, fautes[:3])
chk("le maximum porte sa combinaison et le moment, l'effort normal et le tranchant du point ; ses formules commencent par t(s)",
    mv["combinaison"]["libelle"] == "charge à X, ancrage à Y" and formule(R0, "max_v", "t(s)")["termes"][2]["valeur"] == mv["s"]
    and pres(formule(R0, "max_v", "t(s)")["resultat"], mv["t"]))


# ------------------------------------------------------------------ 8. largeur d'appui
print("\n=== 8. Largeur d'appui a et L_eff ===")
LA = M.verifier(dict(CAS, a=100.0))
chk("a = 100 : L_eff = min(300 ; 100 + 2·47) = 194 mm, σ₁ × 1,55 (300/194 = 1,546), taux 1 = 117,2 % — l'aile horizontale seule (S3 inchangée)",
    LA["geometrie"]["L_eff"] == 194.0 and dec(LA["geometrie"]["facteur_L_eff"], 1.546, 3) and dec(100 * V(LA, "S1")["taux"], 117.2, 1)
    and V(LA, "S1")["largeur"] == 194.0 and V(LA, "S3")["largeur"] == 300.0 and pres(V(LA, "S3")["taux"], V(R0, "S3")["taux"])
    and LA["gouvernant"]["cle"] == "S1" and any(i["cle"] == "L_eff" for i in LA["informations"]) and formule(LA, "largeur_efficace", "L_eff")["resultat"] == 194.0,
    (LA["geometrie"], V(LA, "S1")["taux"]))
chk("a vide → L_eff = L, facteur 1, sans information L_eff ; a = 206 → tout L efficace ; a = 50 → 157,9 %",
    R0["geometrie"]["L_eff"] == 300.0 and R0["geometrie"]["facteur_L_eff"] == 1.0 and not any(i["cle"] == "L_eff" for i in R0["informations"])
    and M.verifier(dict(CAS, a=206.0))["geometrie"]["L_eff"] == 300.0 and dec(100 * V(M.verifier(dict(CAS, a=50.0)), "S1")["taux"], 157.9, 1))
LAJ = M.verifier(dict(CAS, a=100.0, dx=20.0))
chk("avec dx, L_eff suit X′ : la plus petite largeur (charge proche, Xe = 27 → 154 mm) est rendue avec sa combinaison, mais S1 garde la combinaison qui "
    "la gouverne (charge loin : M1 plus grand malgré L_eff plus large)",
    LAJ["geometrie"]["L_eff"] == 154.0 and LAJ["geometrie"]["combinaison_L_eff"]["position_charge"] == "proche"
    and V(LAJ, "S1")["combinaison"]["position_charge"] == "loin" and V(LAJ, "S1")["largeur"] == 234.0, (LAJ["geometrie"]["L_eff"], V(LAJ, "S1")))


# ------------------------------------------------------------------ 9. formules
print("\n=== 9. Formules : chaque résultat à partir de ses seuls termes ===")
FORMULES = {
    "sigma_adm": lambda t, v: t["f_y"] / t["coef"], "tau_adm": lambda t, v: t["σ_adm"] / math.sqrt(3),
    "Xp": lambda t, v: v[0] + v[1], "Yp": lambda t, v: v[0] + v[1],
    "Xe": lambda t, v: v[0] - t["r"] - t["e"], "Xer": lambda t, v: v[0] - t["e"], "Ye": lambda t, v: t["Y"] - t["e"] - t["r"],
    "Yer": lambda t, v: t["Y"] - t["e"], "YH": lambda t, v: t["Y"] - t["e"] / 2,
    "M_bras_j": lambda t, v: t["R"] * (v[1] + t["Xj"]) / 1000, "Mmax_j": lambda t, v: t["R"] * (t["X"] + t["Xj"]) / 1000,
    "rho": lambda t, v: (v[0] + t["Yj"]) / (t["YH"] + t["Yj"]), "M_bras": lambda t, v: t["R"] * v[1] / 1000, "M_rho": lambda t, v: v[0] * t["Mmax"],
    "H_hist": lambda t, v: t["Mmax"] * 1000 / (t["YH"] + t["Yj"]), "M_hist_v": lambda t, v: t["H_s"] * (t["Y′"] - t["s"]) / 1000,
    "M_lin": lambda t, v: t["R"] * (t["X′"] - t["s"]) / 1000, "bras_A": lambda t, v: t["Y′"] - t["y_c"],
    "H_eq": lambda t, v: t["R"] * (t["X′"] - t["u_V"]) / t["bras"],
    "M_eq_A": lambda t, v: (t["H"] * (t["Y′"] - t["s"]) - t["R"] * (t["e"] / 2 - t["u_V"])) / 1000,
    "y_a": lambda t, v: t["haut"] - t["Y′"], "bras_B": lambda t, v: t["y_a"] - t["c"], "M_angle_B": lambda t, v: t["R"] * (t["X′"] - t["e"] / 2) / 1000,
    "M_bas_B": lambda t, v: t["H"] * (t["s_c"] - t["s"]) / 1000,
    "t_conge": lambda t, v: t["e"] + t["r"] - math.sqrt(t["r"] ** 2 - (t["e"] + t["r"] - t["s"]) ** 2),
    "W": lambda t, v: t["b"] * t["t"] ** 2 / 6, "A": lambda t, v: t["b"] * t["t"], "sigma_M": lambda t, v: t["M"] * 1e6 / t["W"],
    "sigma_MN": lambda t, v: abs(t["M"]) * 1e6 / t["W"] + abs(t["N"]) * 1e3 / t["A"], "tau_RLt": lambda t, v: t["R"] * 1e3 / (t["b"] * t["t"]),
    "tau_VA": lambda t, v: abs(t["V"]) * 1e3 / t["A"], "sigma_eq": lambda t, v: math.sqrt(t["σ"] ** 2 + 3 * t["τ"] ** 2),
    "taux": lambda t, v: v[0] / v[1], "N_ELS_hist": lambda t, v: t["Mmax"] * 1000 / ((t["Y"] - t["jeu"]) * t["n"]),
    "N_ELS_eq": lambda t, v: t["H"] / t["n"], "V_ELS": lambda t, v: t["R"] / t["n"], "rapport_ELU": lambda t, v: t["R_ELU"] / t["R"],
    "ELU": lambda t, v: v[0] * t["ρ_ELU"], "R_ELU_auto": lambda t, v: 1.5 * t["R"], "p_diam": lambda t, v: t["V_ELS"] * 1e3 / (t["d"] * t["e"]),
    "arrachement": lambda t, v: t["V_ELS"] * 1e3 / (2 * (t["haut"] - t["Y"]) * t["e"]), "L_eff": lambda t, v: min(t["L"], t["a"] + 2 * v[2]),
    "d0": lambda t, v: t["d"] + 2, "b_net": lambda t, v: t["L"] - t["n"] * t["d0"], "pression_support": lambda t, v: t["C"] * 1e3 / t["L"],
}


def fautes_formules(resultats):
    """Les formules dont le résultat ne se retrouve pas à partir de leurs termes (1e-9), les blocs sans combinaison là
    où il en faut une, les vérifications dont taux, ok, résultat de formule ne se tiennent pas ; (fautes, nombre)."""
    out, n = [], 0
    for res in resultats:
        for b in res["calcul"]:
            for f in b["formules"]:
                n += 1
                t = {x["symbole"]: x["valeur"] for x in f["termes"]}
                vals = [x["valeur"] for x in f["termes"]]
                try:
                    attendu = FORMULES[f["cle"]](t, vals)
                except Exception as err:  # noqa: BLE001
                    out.append(f"{res['corniere']['nom']}/{b['cle']}/{f['symbole']} : {type(err).__name__} {err}")
                    continue
                if not pres(attendu, f["resultat"], 1e-9):
                    out.append(f"{res['corniere']['nom']}/{b['cle']}/{f['symbole']} : {f['resultat']!r} ≠ {attendu!r}")
            if b["cle"] not in ("materiau",) and b["combinaison"] is None:
                out.append(f"{res['corniere']['nom']}/{b['cle']} : sans combinaison")
        for v in res["verifications"]:
            bloc = next(x for x in res["calcul"] if x["cle"] == v["cle"])
            ftx = next(f for f in bloc["formules"] if f["symbole"] == "taux")
            if not (pres(v["taux"], v["sigma_eq"] / v["sigma_adm"]) and v["ok"] == (v["taux"] <= 1.0) and v["etat"] == ("ok" if v["ok"] else "nok")
                    and ftx["resultat"] == v["taux"] and bloc["combinaison"] == v["combinaison"]):
                out.append(f"{res['corniere']['nom']}/{v['cle']} : taux, ok, état ou bloc")
        gov = max((v for v in res["verifications"] if v["verdict"]), key=lambda v: v["taux"])
        if not (res["gouvernant"]["cle"] == gov["cle"] and res["taux_max"] == gov["taux"]
                and res["statut"] == ("ok" if all(v["ok"] for v in res["verifications"] if v["verdict"]) else "nok")):
            out.append(f"{res['corniere']['nom']} : gouvernant / taux_max / statut")
    return out, n


TOUS = [R0, P0, P1, Q1, B, BJ, I1, I2, LA, LAJ, RG] + list(J.values()) + list(DIAG.values()) + [M.verifier(c) for c in TIRAGES_AVEC[:40]] + \
       [M.verifier(dict(c, mode="equilibre", yc=round(0.3 * (c["Y"] - c["dy"]), 1))) for c in TIRAGES_AVEC[40:70]] + \
       [M.verifier(dict(c, configuration="B")) for c in TIRAGES_SANS[:40] if c["Y"] + c["dy"] + c["d"] / 2 + 1 <= M.section(c)["haut"]]
ff, nf = fautes_formules(TOUS)
chk(f"« symbole = expression = valeurs = résultat » : les {nf} formules de {len(TOUS)} cornières se recalculent à partir de leurs seuls termes ; "
    "chaque bloc porte sa combinaison ; taux, ok, état, gouvernant, statut se tiennent", not ff and nf > 3000, ff[:4])
chk("le bloc d'une section nomme sa combinaison (S3 : « charge à X + 20, ancrage à Y + 20 ») et le bloc des ancrages la sienne",
    next(b for b in J["dxdy20"]["calcul"] if b["cle"] == "S3")["combinaison"]["libelle"] == "charge à X + 20, ancrage à Y + 20"
    and next(b for b in J["dxdy20"]["calcul"] if b["cle"] == "ancrages")["combinaison"]["libelle"] == "charge à X + 20, ancrage à Y")


# ------------------------------------------------------------------ 10. domaine
print("\n=== 10. Domaine : refus, bornes, défauts, nuances, JSON ===")
D = M.nouvelle_corniere()
chk("les défauts du § 2 : L 120x120x15 du catalogue, aile A, configuration A, S355, coef 1,5, R = 35, L = 300, X = 75 ± 20, a vide, R_ELU vide, "
    "Y = 85 ± 10, n = 3, d = 16, historique, jeu 10, y_c = 0, u_V vide, v1 faux",
    D == {"nom": "Cornière 1", "standard": True, "profil": "L 120x120x15", "A": 120.0, "B": 120.0, "t": 15.0, "r": 13.0, "aile_horizontale": "A",
          "configuration": "A", "fy": 355.0, "coef": 1.5, "R": 35.0, "L": 300.0, "X": 75.0, "dx": 20.0, "a": None, "R_ELU": None, "Y": 85.0, "dy": 10.0,
          "n": 3, "d": 16.0, "mode": "historique", "jeu_ancrage": 10.0, "yc": 0.0, "uV": None, "v1": False}, D)
RD = M.verifier(D)
chk("la cornière neuve tient (taux < 100 %), R_ELU auto = 52,5 kN dit en information, L_eff = L",
    RD["statut"] == "ok" and RD["taux_max"] < 1.0 and RD["ancrages"]["R_ELU"] == 52.5 and RD["ancrages"]["R_ELU_auto"]
    and any(a["code"] == "R_ELU_auto" for a in RD["avertissements"]) and RD["geometrie"]["L_eff"] == 300.0, (RD["taux_max"], RD["ancrages"]["R_ELU"]))
chk("les nuances du site viennent d'une source unique (acier/flambement/normes, Tableau 3.1) : 235, 275, 355, 460 ; S355 ↔ 355",
    E.FY_ADMIS == (235.0, 275.0, 355.0, 460.0) and E.nuance_de(355.0) == "S355" and E.nuance_de(300.0) is None
    and "from acier.flambement import normes" in lire("acier", "corniere", "entree.py") and not re.search(r"\b235\.0\b.*\b275\.0\b", lire("acier", "corniere", "entree.py")))
REFUS = (("charge dans l'angle (X − dx = 28 ≤ e + r)", {"X": 48.0, "dx": 20.0}, "dans l'angle"), ("charge au-delà de l'aile (X + dx = 125)", {"X": 105.0}, "au-delà"),
         ("ancrage dans le congé (Y − dy − 9 < 28)", {"Y": 45.0}, "congé"), ("ancrage hors de l'aile (Y + dy + 9 > 120)", {"Y": 105.0}, "sort"),
         ("t ≥ aile (saisie libre)", {"standard": False, "A": 12.0, "t": 12.0}, "cornière"), ("e + r atteint l'aile", {"standard": False, "A": 30.0, "B": 30.0, "t": 15.0, "r": 15.0}, "partie droite"),
         ("congé négatif", {"standard": False, "r": -1.0}, "congé"), ("nuance inconnue (300)", {"fy": 300.0}, "nuance"), ("coef < 1", {"coef": 0.9}, "au moins 1"),
         ("R négative", {"R": -1.0}, "négative"), ("L nulle", {"L": 0.0}, "positive"), ("dx négatif", {"dx": -1.0}, "dx"), ("dy négatif", {"dy": -1.0}, "dy"),
         ("n = 0", {"n": 0}, "au moins 1"), ("n non entier", {"n": 2.5}, "entier"), ("d nul", {"d": 0.0}, "positif"), ("trous qui mangent L (n = 20, Ø16)", {"n": 20}, "section nette"),
         ("a nulle", {"a": 0.0}, "positive"), ("R_ELU négative", {"R_ELU": -5.0}, "négative"), ("jeu d'ancrage ≥ Y", {"jeu_ancrage": 85.0}, "jeu"),
         ("jeu d'ancrage négatif", {"jeu_ancrage": -1.0}, "jeu"), ("y_c au-delà du bras (A : 75)", {"yc": 75.0}, "pivot"), ("y_c négatif", {"yc": -1.0}, "pivot"),
         ("y_c au-delà du bras en B (haut − Y − dy = 25)", {"configuration": "B", "yc": 25.0}, "pivot"), ("u_V > e", {"uV": 16.0}, "u_V"), ("u_V < 0", {"uV": -1.0}, "u_V"),
         ("aile horizontale « C »", {"aile_horizontale": "C"}, "choix"), ("configuration « haut »", {"configuration": "haut"}, "choix"), ("mode « lame »", {"mode": "lame"}, "choix"),
         ("profil inconnu", {"profil": "L 99x99x9"}, "catalogue"), ("profil qui n'est pas une cornière", {"profil": "IPE 200"}, "catalogue"),
         ("clé inconnue", {"epaisseur": 6}, "inconnue"), ("R en texte", {"R": "35"}, "nombre"), ("t booléen", {"standard": False, "t": True}, "nombre"),
         ("X infini", {"X": float("inf")}, "infini"), ("Y indéterminé", {"Y": float("nan")}, "indéterminé"), ("nom numérique", {"nom": 3}, "texte"),
         ("nom trop long", {"nom": "x" * 61}, "60"), ("standard « oui »", {"standard": "oui"}, "vrai ou faux"), ("v1 en texte", {"v1": "oui"}, "vrai ou faux"))


def refus_manques(mod):
    out = []
    for nom, champ, mot in REFUS:
        try:
            mod.normaliser(dict(mod.nouvelle_corniere(), **champ))
            out.append(f"{nom} : accepté")
        except ValueError as err:
            if mot.lower() not in str(err).lower():
                out.append(f"{nom} : message « {err} » sans « {mot} »")
        except Exception as err:  # noqa: BLE001
            out.append(f"{nom} : {type(err).__name__} {err}")
    return out


rf = refus_manques(M)
chk(f"{len(REFUS)} saisies hors du domaine lèvent ValueError avec un message qui dit quoi (jamais une valeur ramenée en silence)", not rf, rf[:4])
for nom, champ in (("X − dx juste au-delà de l'angle (X = 48,1)", {"X": 48.1}), ("X + dx = larg (X = 100)", {"X": 100.0}), ("ancrage au ras du congé (Y − dy − 9 = 28 : Y = 47)", {"Y": 47.0}),
                   ("ancrage au ras du bord (Y + dy + 9 = 120 : Y = 101)", {"Y": 101.0}), ("dx = dy = 0", {"dx": 0.0, "dy": 0.0}), ("coef = 1", {"coef": 1.0}),
                   ("R = 0", {"R": 0.0}), ("n = 1", {"n": 1}), ("n = 3.0 (entier écrit en flottant)", {"n": 3.0}), ("jeu d'ancrage 0", {"jeu_ancrage": 0.0}),
                   ("y_c juste sous le bras (74,9)", {"yc": 74.9}), ("u_V = e", {"uV": 15.0}), ("u_V = 0", {"uV": 0.0}), ("a = L", {"a": 300.0}), ("a > L", {"a": 400.0}),
                   ("R_ELU = 0", {"R_ELU": 0.0}), ("saisie libre r = 0", {"standard": False, "r": 0.0}), ("t = 44 mm (f_y réduit : avertissement)", {"standard": False, "A": 200.0, "B": 200.0, "t": 44.0, "r": 10.0, "X": 100.0, "Y": 150.0})):
    try:
        rr = M.verifier(dict(M.nouvelle_corniere(), **champ))
        chk(f"borne admise : {nom}", True)
    except ValueError as err:
        chk(f"borne admise : {nom}", False, err)
RZ = M.verifier(dict(CAS, R=0.0))
chk("R = 0 : tout est nul, statut ok, rapport ELU indéfini (None), N_ELU = 0", RZ["taux_max"] == 0.0 and RZ["statut"] == "ok" and RZ["ancrages"]["rapport_ELU"] is None
    and RZ["ancrages"]["N_ELU"] == 0.0 and RZ["ancrages"]["H"] == 0.0)
chk("t = 44 mm : avertissement fy_epaisseur (Tableau 3.1), sans refus",
    any(a["code"] == "fy_epaisseur" for a in M.verifier(dict(M.nouvelle_corniere(), standard=False, A=200.0, B=200.0, t=44.0, r=10.0, X=100.0, Y=150.0))["avertissements"]))
chk("« -0 » est relu 0", math.copysign(1, M.normaliser(dict(CAS, yc=-0.0))["yc"]) == 1.0)
chk("normaliser() est idempotente, {} donne la cornière neuve, None vaut absent, a/R_ELU/uV vides restent vides",
    all(M.normaliser(M.normaliser(r["corniere"])) == r["corniere"] for r in TOUS) and M.normaliser({}) == D and M.normaliser({"R": None, "a": None}) == D)
entree = copy.deepcopy(CAS)
M.verifier(entree)
M.candidats(entree)
chk("verifier() et candidats() ne modifient pas la cornière reçue", entree == CAS)
chk("la sortie est du JSON strict (sans NaN) et se relit à l'identique", all(json.loads(json.dumps(r, allow_nan=False)) == r for r in TOUS))
chk("les avertissements permanents : chevilles non vérifiées, pic au congé, effet de levier (niveau info), avec leurs textes",
    all({"chevilles_non_verifiees", "pic_conge", "effet_levier"} <= {a["code"] for a in r["avertissements"]} for r in TOUS)
    and any("Hilti PROFIS" in a["message"] for a in R0["avertissements"]) and any("1,3" in a["message"] for a in R0["avertissements"])
    and any("250 kN/mm" in a["message"] for a in R0["avertissements"]))
chk("saisie libre : masse et aire calculées (120×120×15, r = 13 : 26,8 kg/m, +0,5 % sur le catalogue 26,64 qui a les arrondis de bout d'aile), dites « calculée »",
    dec(M.section(dict(D, standard=False))["masse"], 26.8, 1) and M.section(dict(D, standard=False))["masse_source"] == "calculée"
    and M.section(D)["masse_source"] == "catalogue" and M.section(dict(D, standard=False))["source"] == "saisie")
chk("une désignation tapée se relit au catalogue, la désignation du bureau (×, virgule)",
    M.nom_catalogue("120x120x15") == "L 120x120x15" and M.nom_catalogue("L 100×65×8") == "L 100x65x8" and M.nom_catalogue("65 x 100 x 8") == "L 100x65x8"
    and M.nom_catalogue("L 45×45×4,5") == "L 45x45x4.5" and M.nom_catalogue("IPE 200") is None and M.designation("L 45x45x4.5") == "L 45×45×4,5")


# ------------------------------------------------------------------ 11. relecture d'une note v1
print("\n=== 11. Relecture d'une note du module d'avant (v1, modèle lame) ===")
# les cornières des photos de tests/reference_refonte/corniere.json AVANT la refonte (état au 10/11/2026), copiées telles quelles
V1 = [
    {"nom": "Cornière 1", "standard": True, "profil": "L 100x100x10", "A": 60.0, "B": 60.0, "t": 6.0, "charge": "lineaire", "V": 5.0, "L": 1.0, "e": 2.3,
     "gamma_ELU": 1.35, "s": 30.0, "bord": 2.0, "sig_lim_ELS": 155.0, "tau_lim_ELS": 0.8, "sig_lim_ELU": 186.0},
    {"nom": "Cornière 1", "standard": True, "profil": "L 45x45x4.5", "A": 60.0, "B": 60.0, "t": 6.0, "charge": "lineaire", "V": 5.0, "L": 1.0, "e": 2.3,
     "gamma_ELU": 1.35, "s": 30.0, "bord": 2.0, "sig_lim_ELS": 155.0, "tau_lim_ELS": 0.8, "sig_lim_ELU": 186.0},
    {"nom": "Cornière 1", "standard": True, "profil": "L 100x100x10", "A": 60.0, "B": 60.0, "t": 6.0, "charge": "lineaire", "V": 8.0, "L": 1.0, "e": 2.3,
     "gamma_ELU": 1.35, "s": 30.0, "bord": 2.0, "sig_lim_ELS": 155.0, "tau_lim_ELS": 0.8, "sig_lim_ELU": 186.0},
    {"nom": "Cornière 1", "standard": True, "profil": "L 100x65x8", "A": 60.0, "B": 60.0, "t": 6.0, "charge": "ponctuelle", "V": 5.0, "L": 1.0, "e": 2.3,
     "gamma_ELU": 1.35, "s": 30.0, "bord": 2.0, "sig_lim_ELS": 155.0, "tau_lim_ELS": 0.8, "sig_lim_ELU": 186.0},
    {"nom": "Cornière 1", "standard": True, "profil": "L 100x100x10", "A": 60.0, "B": 60.0, "t": 6.0, "charge": "lineaire", "V": 5.0, "L": 1.0, "e": 25.0,
     "gamma_ELU": 1.35, "s": 30.0, "bord": 2.0, "sig_lim_ELS": 155.0, "tau_lim_ELS": 0.8, "sig_lim_ELU": 186.0},
    {"nom": "Cornière 3", "standard": True, "profil": "L 100x100x10", "A": 60.0, "B": 60.0, "t": 6.0, "charge": "lineaire", "V": 5.0, "L": 1.0, "e": 2.3,
     "gamma_ELU": 1.35, "s": 30.0, "bord": 2.0, "sig_lim_ELS": 155.0, "tau_lim_ELS": 0.8, "sig_lim_ELU": 186.0},
    # une saisie libre du module d'avant (65 × 100 × 8)
    {"nom": "Libre", "standard": False, "profil": "L 100x100x10", "A": 65.0, "B": 100.0, "t": 8.0, "charge": "ponctuelle", "V": 12.0, "L": 0.6, "e": 4.0,
     "gamma_ELU": 1.35, "s": 30.0, "bord": 2.0, "sig_lim_ELS": 155.0, "tau_lim_ELS": 0.8, "sig_lim_ELU": 186.0},
]
fautes, relues = [], []
for v1 in V1:
    try:
        c1 = M.normaliser(v1)
        r1 = M.verifier(c1)
        relues.append((v1, c1, r1))
    except ValueError as err:
        fautes.append(f"{v1['profil']} : refusée ({err})")
chk(f"les {len(V1)} notes v1 (dont les six photos du gel d'avant la refonte) s'ouvrent : rien n'est refusé", not fautes and len(relues) == len(V1), fautes)
chk("le nom et la section sont gardés (catalogue : profil ; saisie libre : A, B, t, congé 0) ; tout le reste prend les défauts (mode, configuration, nuance, n, d…) "
    "et v1 est vrai",
    all(c1["nom"] == v1["nom"] and c1["standard"] == v1["standard"] and c1["profil"] == v1["profil"] and c1["v1"] is True and c1["mode"] == "historique"
        and c1["configuration"] == "A" and c1["fy"] == 355.0 and c1["coef"] == 1.5 and c1["R"] == 35.0 and c1["L"] == 300.0 and c1["n"] == 3 for v1, c1, _ in relues)
    and relues[6][1]["A"] == 65.0 and relues[6][1]["B"] == 100.0 and relues[6][1]["t"] == 8.0 and relues[6][1]["r"] == 0.0)
chk("la charge et l'ancrage sont posés DANS les ailes, au rapport du cas de contrôle (L 100 : X = 62,5 ± 16,5, Y = 71 ± 8,5, Ø16 ; L 45x45x4,5 : X = 28 ± 7,5, "
    "Y = 32 ± 4, Ø16 au ras du bord) ; la L 120 redonnerait les défauts exacts",
    (relues[0][1]["X"], relues[0][1]["dx"], relues[0][1]["Y"], relues[0][1]["dy"], relues[0][1]["d"]) == (62.5, 16.5, 71.0, 8.5, 16.0)
    and (relues[1][1]["X"], relues[1][1]["dx"], relues[1][1]["Y"], relues[1][1]["dy"], relues[1][1]["d"]) == (28.0, 7.5, 32.0, 4.0, 16.0)
    and {k: M.normaliser(dict(V1[0], profil="L 120x120x15"))[k] for k in ("X", "dx", "Y", "dy", "d")} == {"X": 75.0, "dx": 20.0, "Y": 85.0, "dy": 10.0, "d": 16.0},
    [(c1["X"], c1["dx"], c1["Y"], c1["dy"], c1["d"]) for _, c1, _ in relues])
chk("l'avertissement « note_v1 » (attention) porte le texte des décisions ; il est en tête ; une note v2 ne l'a pas ; v1 traverse normaliser (idempotent)",
    all(r1["avertissements"][0]["code"] == "note_v1" and r1["avertissements"][0]["niveau"] == "attention" and r1["avertissements"][0]["message"] == E.AVIS_V1 for _, _, r1 in relues)
    and "modèle lame" in E.AVIS_V1 and "ressaisir" in E.AVIS_V1 and not any(a["code"] == "note_v1" for a in R0["avertissements"])
    and all(M.normaliser(c1)["v1"] is True for _, c1, _ in relues) and all(E.est_v1(v1) for v1 in V1) and not E.est_v1(CAS) and not E.est_v1({"nom": "x"}))
chk("les anciens chiffres ne sont plus calculés : aucune clé du modèle lame dans la sortie (V, e, s, gamma_ELU, sig_lim…), le résultat est celui du nouveau calcul",
    all(not ({"V", "e", "s", "gamma_ELU", "sig_lim_ELS", "tau_lim_ELS", "sig_lim_ELU", "bord", "charge"} & set(c1)) for _, c1, _ in relues)
    and all("taux_max" in r1 and r1["mode"] == "historique" for _, _, r1 in relues))


# ------------------------------------------------------------------ 12. le sélecteur
print("\n=== 12. Sélecteur : les 233 cornières du catalogue ===")
CANDS = M.candidats(CAS)
masses = [x["masse"] for x in CANDS]
hors = [x for x in CANDS if not x["domaine"]]
dedans = [x for x in CANDS if x["domaine"]]
chk(f"{len(CANDS)} cornières (toutes celles du catalogue, L et LI), triées par masse ; {len(hors)} hors domaine avec leur raison (la charge au-delà de l'aile, "
    "l'ancrage hors de l'aile…), aucune calculée ; les autres calculées",
    len(CANDS) == len(M.cornieres()) == 233 and masses == sorted(masses) and {x["famille"] for x in CANDS} == {"L", "LI"} and len(hors) > 50
    and all(x["raison"] and x["taux"] is None and not x["ok"] for x in hors) and all(x["taux"] is not None and x["raison"] is None for x in dedans)
    and any("au-delà" in x["raison"] for x in hors) and any("sort" in x["raison"] or "congé" in x["raison"] for x in hors), [x["raison"] for x in hors[:2]])
ecarts = []
for x in RNG.sample(dedans, 12) + [x for x in CANDS if x["retenue"]]:
    r = M.verifier(dict(CAS, profil=x["nom"]))
    if not (x["taux"] == r["taux_max"] and x["ok"] == (r["statut"] == "ok")):
        ecarts.append(x["nom"])
chk("chaque cornière du sélecteur est calculée avec les charges de la cornière ouverte : taux = taux_max du calcul complet, ok = statut (13 vérifiées)", not ecarts, ecarts)
premier = next((x for x in CANDS if x["ok"] and x["domaine"]), None)
chk(f"optimal = la plus légère qui tient ET dans le domaine ({premier and premier['nom']}) ; une seule ; retenue = L 120x120x15 (qui tient) ; optimal() la rend",
    premier is not None and premier["optimal"] and sum(x["optimal"] for x in CANDS) == 1 and [x["nom"] for x in CANDS if x["retenue"]] == ["L 120x120x15"]
    and M.optimal(CAS)["nom"] == premier["nom"] and all(not (x["ok"] and x["domaine"]) for x in CANDS[:CANDS.index(premier)]))
chk("candidats(familles=['LI']) : les seules cornières à ailes inégales ; aucune ne tient (R = 500) → pas d'optimal ; en B et en équilibre le sélecteur calcule aussi",
    {x["famille"] for x in M.candidats(CAS, ["LI"])} == {"LI"} and M.optimal(dict(CAS, R=500.0)) is None
    and any(x["ok"] for x in M.candidats(dict(CAS, configuration="B"))) and any(x["ok"] for x in M.candidats(dict(CAS, mode="equilibre"))))
CB = M.candidats(dict(CAS, configuration="B"))
chk("en B, le taux du sélecteur est celui du calcul complet (section nette comprise)",
    all(x["taux"] == M.calculer(dict(CAS, configuration="B", profil=x["nom"]), diagrammes=False)["taux_max"] for x in RNG.sample([x for x in CB if x["domaine"]], 5)))


# ------------------------------------------------------------------ 13. pureté
print("\n=== 13. Pureté : bibliothèque standard, le catalogue, les nuances ; sous-processus sans streamlit ===")
non_std = []
# le MOTEUR : entree, statique, moteur (et l'init du paquet) ; les vues (schemas.py, sur le kit de dessin) et la note
# (reportlab) vivent dans le même paquet mais ne sont pas le calcul — leur pureté a ses propres contrôles
MODULES_MOTEUR = ("__init__.py", "entree.py", "statique.py", "moteur.py")
for f in sorted(os.listdir(os.path.join(RACINE, "acier", "corniere"))):
    if f in MODULES_MOTEUR:
        arbre = ast.parse(lire("acier", "corniere", f))
        for n in ast.walk(arbre):
            noms = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module] if isinstance(n, ast.ImportFrom) and n.level == 0 and n.module else []
            non_std += [f"{f}: {x}" for x in noms if x.split(".")[0] not in sys.stdlib_module_names and not x.startswith(("acier.profiles", "acier.flambement"))]
chk("acier/corniere n'importe que la bibliothèque standard, le catalogue acier.profiles et les nuances acier.flambement.normes (lecture des sources)", not non_std, non_std)
SOUS_PROCESSUS = r'''
import json, sys
avant = set(sys.modules)
INTERDITS = ("streamlit", "pandas", "reportlab", "matplotlib", "scipy", "PIL", "fitz", "pymupdf")
class Refus:
    def find_spec(self, nom, chemin=None, cible=None):
        if nom.split(".")[0] in INTERDITS:
            raise ImportError("chargement refusé : " + nom)
        return None
sys.meta_path.insert(0, Refus())
try:
    import streamlit  # noqa: F401
    refus = False
except ImportError:
    refus = True
sys.path.insert(0, sys.argv[1])
from acier.corniere import moteur as M
cs = json.loads(sys.stdin.read())
sortie = {"refus": refus, "version": list(sys.version_info[:2]), "res": [M.verifier(c) for c in cs], "cands": M.candidats(cs[0])}
charges = {m.split(".")[0] for m in set(sys.modules) - avant}
sortie["tiers"] = sorted(charges - set(sys.stdlib_module_names) - {"acier"})
sortie["interdits"] = sorted(m for m in sys.modules if m.split(".")[0] in INTERDITS)
print(json.dumps(sortie))
'''
cs = [CAS, dict(CAS, configuration="B"), dict(CAS, mode="equilibre", dx=20.0, dy=10.0), dict(CAS, profil="L 150x100x12", aile_horizontale="B", Y=115.0, a=100.0)]
attendu = json.loads(json.dumps({"res": [M.verifier(c) for c in cs], "cands": M.candidats(cs[0])}))
interpretes = [sys.executable] + [x for x in ("/usr/bin/python3.13",) if os.path.exists(x) and os.path.realpath(x) != os.path.realpath(sys.executable)]
for exe in interpretes:
    pr = subprocess.run([exe, "-I", "-c", SOUS_PROCESSUS, RACINE], input=json.dumps(cs), capture_output=True, text=True, check=False)
    if pr.returncode != 0:
        chk(f"{exe} : le moteur se charge et calcule sans streamlit", False, pr.stderr.strip().splitlines()[-3:])
        continue
    out = json.loads(pr.stdout)
    v = ".".join(map(str, out["version"]))
    chk(f"Python {v} : streamlit, pandas, reportlab… REFUSÉS par sys.meta_path, et le moteur calcule quand même (4 cornières, le sélecteur)",
        out["refus"] and not out["interdits"], (out["refus"], out["interdits"]))
    chk(f"Python {v} : seuls la bibliothèque standard et numpy (le catalogue acier/profiles) sont chargés", set(out["tiers"]) <= {"numpy"}, out["tiers"])
    diff = [i for i, (a, b) in enumerate(zip(out["res"], attendu["res"])) if a != b] + (["cands"] if out["cands"] != attendu["cands"] else [])
    chk(f"Python {v} : résultats identiques à ceux de ce processus", not diff, diff[:3])
if len(interpretes) == 1:
    print("     (python3.13 absent : le calcul sous la version de Pyodide n'est pas rejoué sur cette machine)")


# ------------------------------------------------------------------ 14. mutations
print("\n=== 14. Mutations : le moteur modifié d'UNE formule clé est rejeté ===")
PAQUET = sys.modules["acier.corniere"]
NOMS_MODULES = ("entree", "statique", "moteur")


def charger_mutant(mutations):
    """Les trois modules rechargés depuis leurs SOURCES après des remplacements [(module, avant, après)] ; chaque
    « avant » doit figurer UNE fois dans la source (sinon la mutation est périmée). Rend le moteur mutant."""
    sauve = {n: sys.modules[f"acier.corniere.{n}"] for n in NOMS_MODULES}
    sauve_attr = {n: getattr(PAQUET, n) for n in NOMS_MODULES}
    mods = {}
    try:
        for n in NOMS_MODULES:
            chemin = os.path.join(RACINE, "acier", "corniere", n + ".py")
            src = lire("acier", "corniere", n + ".py")
            for module, avant, apres in mutations:
                if module == n:
                    if src.count(avant) != 1:
                        raise AssertionError(f"mutation périmée : « {avant} » ({src.count(avant)} fois dans {n}.py)")
                    src = src.replace(avant, apres)
            m = types.ModuleType(f"acier.corniere.{n}")
            m.__package__, m.__file__ = "acier.corniere", chemin
            sys.modules[m.__name__] = m
            setattr(PAQUET, n, m)
            exec(compile(src, chemin, "exec"), m.__dict__)
            mods[n] = m
    finally:
        sys.modules.update({f"acier.corniere.{n}": sauve[n] for n in NOMS_MODULES})
        for n in NOMS_MODULES:
            setattr(PAQUET, n, sauve_attr[n])
    return mods["moteur"]


def rejets(mod):
    """Ce que les contrôles de ce fichier reprochent au moteur ``mod`` : le cas de contrôle, P8 sur des tirages, B,
    P0/P1/Q1, les jeux, les maxima, L_eff, la garde, les refus, les défauts, les formules, les diagrammes, le sélecteur,
    la relecture v1, les positions."""
    out = []
    try:
        r0 = mod.verifier(CAS)
        if not (dec(100 * r0["taux_max"], 88.8621, 4) and r0["gouvernant"]["cle"] == "S3" and dec(r0["ancrages"]["N_ELS"], 14.2533, 4) and dec(r0["ancrages"]["N_ELU"], 22.9667, 4)
                and all(dec(V(r0, k)["M"], m, 6) for k, m in (("S1", 2.009720), ("S2", 2.565600), ("S3", 2.358697), ("S4", 2.896645)))
                and dec(r0["ancrages"]["pression_diametrale"], 59.389, 3) and dec(r0["ancrages"]["arrachement"], 13.575, 3) and r0["ancrages"]["positions_z"] == [50.0, 150.0, 250.0]):
            out.append("cas de contrôle")
        for c in TIRAGES_SANS[:8] + TIRAGES_AVEC[:12]:
            ec, _ = ecarts_p8(mod, c, c["dx"], c["dy"])
            if ec:
                out.append(f"P8 {c['nom']} : {ec[0]}")
                break
        b = mod.verifier(dict(CAS, configuration="B"))
        if not (dec(b["ancrages"]["H"], 82.47, 2) and dec(100 * V(b, "S3")["taux"], 112.42, 2) and dec(100 * V(b, "nette")["taux"], 137.1, 1) and b["gouvernant"]["cle"] == "nette"):
            out.append("configuration B")
        if not dec(mod.verifier(dict(CAS, configuration="B", yc=10.0))["ancrages"]["H"], 115.45, 2):
            out.append("B avec contact à c = 10")
        p0 = mod.verifier(dict(CAS, mode="equilibre"))
        if not (dec(100 * V(p0, "S3")["taux"], 76.91, 2) and dec(100 * p0["taux_max"], 77.94, 2) and dec(100 * V(mod.verifier(dict(CAS, mode="equilibre", yc=7.5)), "S3")["taux"], 83.96, 2)
                and dec(100 * V(mod.verifier(dict(CAS, mode="equilibre", yc=7.5, uV=0.0)), "S3")["taux"], 80.84, 2)):
            out.append("équilibre P0 / P1 / Q1")
        j = mod.verifier(dict(CAS, dx=20.0, dy=20.0))
        if not (dec(100 * j["taux_max"], 120.69, 2) and dec(100 * mod.verifier(dict(CAS, dx=20.0))["taux_max"], 112.43, 2)
                and dec(100 * mod.verifier(dict(CAS, dy=20.0))["taux_max"], 95.38, 2) and V(j, "S3")["combinaison"]["position_ancrage"] == "haut"):
            out.append("jeux et enveloppe")
        if not (dec(V(r0, "max_h")["recul"], 1.98, 2) and dec(100 * V(r0, "max_h")["ecart"], 2.11, 2) and dec(V(r0, "max_v")["recul"], 1.66, 2) and dec(100 * V(r0, "max_v")["ecart"], 1.45, 2)):
            out.append("maximum du congé")
        la = mod.verifier(dict(CAS, a=100.0))
        if not (la["geometrie"]["L_eff"] == 194.0 and dec(100 * V(la, "S1")["taux"], 117.2, 1)):
            out.append("L_eff")
        if not (mod.verifier(dict(CAS, Y=40.0))["garde"]["depasse"] and any(a["code"] == "equilibre_plus_severe" for a in mod.verifier(dict(CAS, Y=40.0))["avertissements"])
                and not r0["garde"]["depasse"]):
            out.append("garde")
        out += [f"refus : {x}" for x in refus_manques(mod)[:2]]
        if mod.nouvelle_corniere() != D:
            out.append("défauts")
        ff_, _ = fautes_formules([r0, b, p0, j, la])
        out += [f"formules : {x}" for x in ff_[:2]]
        out += [f"diagrammes : {x}" for x in fautes_diagrammes(mod.verifier(dict(CAS, dx=20.0, dy=10.0)))[:1]]
        cands = mod.candidats(CAS)
        ms = [x["masse"] for x in cands]
        prem = next((x for x in cands if x["ok"] and x["domaine"]), None)
        if ms != sorted(ms) or not (prem and prem["optimal"]) or sum(x["optimal"] for x in cands) != 1 or len(cands) != 233:
            out.append("sélecteur : tri, optimal")
        c1 = mod.normaliser(V1[0])
        if not (c1["v1"] is True and (c1["X"], c1["Y"]) == (62.5, 71.0) and mod.verifier(c1)["avertissements"][0]["code"] == "note_v1"):
            out.append("relecture v1")
        baisses_ = balayage(dict(CAS, mode="equilibre"), "dy", [0.0, 5.0, 10.0]) if False else []
        out += baisses_
    except AssertionError:
        raise
    except Exception as err:  # noqa: BLE001 — un mutant qui casse est rejeté
        out.append(f"{type(err).__name__} : {err}")
    return out


MUTATIONS = (
    ("ρ₃ sans Δy : H_s = Mmax/YH, l'ancrage haut ne charge plus la section 3", [("statique", "Hs = Mmax / (YH + Yj)", "Hs = Mmax / YH")]),
    ("pivot des sections YH = Y − e au lieu de Y − e/2", [("statique", "YH = g[\"Y\"] - e / 2.0", "YH = g[\"Y\"] - e")]),
    ("N oublié dans l'aile verticale en B", [("statique", "N_v=lambda s: -R if s <= Yp else 0.0", "N_v=lambda s: 0.0")]),
    ("section nette sans les trous", [("moteur", "b_net = c[\"L\"] - c[\"n\"] * E.d0(c)", "b_net = c[\"L\"]")]),
    ("L_eff oubliée", [("moteur", "return min(c[\"L\"], c[\"a\"] + 2.0 * (st[\"Xp\"] - r - e))", "return c[\"L\"]")]),
    ("dx oublié dans les combinaisons", [("moteur", "\"X\": c[\"X\"] + a, \"Y\": c[\"Y\"] + b", "\"X\": c[\"X\"], \"Y\": c[\"Y\"] + b")]),
    ("moment constant en B : R·X′ au lieu de R·(X′ − e/2)", [("statique", "M_angle = R * (Xp - e / 2.0)", "M_angle = R * Xp")]),
    ("H de l'équilibre sans u_V", [("statique", "H = R * (Xp - uV) / bras\n        st.update(H=H, bras=bras,", "H = R * Xp / bras\n        st.update(H=H, bras=bras,")]),
    ("moment de l'aile verticale (équilibre A) sans le terme R·(e/2 − u_V)", [("statique", "M_v=lambda s: H * (Yp - s) - R * (e / 2.0 - uV) if s < Yp else 0.0", "M_v=lambda s: H * (Yp - s) if s < Yp else 0.0")]),
    ("épaisseur du congé t(s) = e partout", [("statique", "return e + r - math.sqrt(max(r * r - w * w, 0.0))", "return e")]),
    ("σ_éq = √(σ² + τ²)", [("statique", "math.sqrt(sigma * sigma + 3.0 * tau * tau)", "math.sqrt(sigma * sigma + tau * tau)")]),
    ("τ du classeur oublié dans l'aile verticale", [("statique", "V_v=lambda s: R if s < Yp else 0.0,\n                  N_v=lambda s: 0.0)", "V_v=lambda s: 0.0,\n                  N_v=lambda s: 0.0)")]),
    ("bras de la traction historique Y au lieu de Y − jeu", [("statique", "H=Mmax / (g[\"Y\"] - g[\"jeu_ancrage\"])", "H=Mmax / g[\"Y\"]")]),
    ("rapport ELU figé à 1,5", [("moteur", "rapport = (R_ELU / R) if R > 0 else None", "rapport = 1.5 if R > 0 else None")]),
    ("enveloppe : la combinaison la moins défavorable", [("moteur", "if best is None or et[\"taux\"] > best[\"taux\"] * (1.0 + EPS):", "if best is None or et[\"taux\"] < best[\"taux\"] * (1.0 + EPS):")]),
    ("maximum au pas de 1 mm sans affinage", [("statique", "PAS_CONGE = 0.1 ", "PAS_CONGE = 1.0 "), ("statique", "affiner: bool = True", "affiner: bool = False")]),
    ("garde muette", [("moteur", "\"depasse\": t_eq > t_hist * (1.0 + 1e-9)", "\"depasse\": False")]),
    ("gouvernant : le plus petit taux", [("moteur", "return max((v for v in verifs if v[\"verdict\"]), key=lambda v: v[\"taux\"])", "return min((v for v in verifs if v[\"verdict\"]), key=lambda v: v[\"taux\"])")]),
    ("les maxima du congé entrent dans le verdict historique", [("moteur", "v = _maximum(ctx, aile, verdict=not hist)", "v = _maximum(ctx, aile, verdict=True)")]),
    ("charge au-delà de l'aile admise", [("entree", "_exiger(X + dx <= larg,", "_exiger(True or X + dx <= larg,")]),
    ("défaut de dx : 0", [("entree", "\"dx\": (20.0, \"mm\"", "\"dx\": (0.0, \"mm\"")]),
    ("sélecteur trié par nom", [("entree", "key=lambda p: (p[\"masse\"], p[\"nom\"])", "key=lambda p: (p[\"nom\"], p[\"masse\"])")]),
    ("optimal : la plus LOURDE qui tient", [("moteur", "premier = next((x for x in out if x[\"ok\"] and x[\"domaine\"]), None)", "premier = next((x for x in reversed(out) if x[\"ok\"] and x[\"domaine\"]), None)")]),
    ("relecture v1 sans l'indicateur", [("entree", "out[\"v1\"] = True", "out[\"v1\"] = False")]),
    ("diagrammes : l'enveloppe ne regarde que la combinaison nominale", [("moteur", "        for st in ctx[\"stats\"]:\n            m, nn, vv = S.efforts(st, aile, s)", "        for st in ctx[\"stats\"][:1]:\n            m, nn, vv = S.efforts(st, aile, s)")]),
    ("positions des ancrages depuis le bord", [("moteur", "\"positions_z\": [c[\"L\"] / (2 * n) + i * c[\"L\"] / n for i in range(n)]", "\"positions_z\": [i * c[\"L\"] / n for i in range(n)]")]),
    ("nuance quelconque admise", [("entree", "_exiger(out[\"fy\"] in FY_ADMIS,", "_exiger(True or out[\"fy\"] in FY_ADMIS,")]),
    ("B : le contact ignore c", [("statique", "bras = ya - c\n", "bras = ya\n")]),
)
TEMOIN = ("témoin : un commentaire changé", [("moteur", "# les sections repères :", "# les sections repères (témoin) :")])
survivants, n_rejets = [], []
for nom, muts in MUTATIONS:
    try:
        rej = rejets(charger_mutant(muts))
    except AssertionError as err:
        survivants.append(f"{nom} : {err}")
        continue
    n_rejets.append(len(rej))
    if not rej:
        survivants.append(nom)
chk(f"{len(MUTATIONS)} mutations volontaires (une par formule clé : ρ₃, pivot, N en B, nette, L_eff, dx, M_angle, u_V, t(s), σ_éq, τ, bras, ELU, enveloppe, maximum, "
    f"garde, gouvernant, verdict, refus, défaut, sélecteur, optimal, v1, diagrammes, positions, nuance, contact) : TOUTES rejetées "
    f"(de {min(n_rejets or [0])} à {max(n_rejets or [0])} reproches chacune)", not survivants, survivants)
chk("la mutation témoin (un commentaire) n'est rejetée par aucun contrôle : les rejets viennent bien des mutations", not rejets(charger_mutant(TEMOIN[1])))
chk("après les mutations, le moteur chargé est intact (mêmes résultats)", M.verifier(CAS) == R0 and M.candidats(CAS) == CANDS)

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:400])
sys.exit(1 if KO else 0)

# -*- coding: utf-8 -*-
"""Socle des scripts de SYNTHÈSE (document ANALYSE_CORNIERE_EXCEL.md).

Tout chiffre du document sort d'un script de ce dossier. Ce module rassemble :

- le cas de contrôle de l'ingénieur ;
- la méthode historique du classeur, RÉÉCRITE à partir des symboles (N, mm), sans
  lire une formule du classeur (`historique`) ;
- la projection de cette méthode sur les cellules de P8, dans les unités du
  classeur (`cellules_modele`) : sert à la comparaison cellule par cellule ;
- un petit interpréteur des formules de P8 (`Classeur`), qui relit les formules
  telles qu'elles sont écrites et les recalcule pour n'importe quelle saisie ;
- l'épaisseur exacte du congé t(s), le corps libre paramétré (pivot y_P, ligne
  d'action du cisaillement u_V) ;
- la mise en forme française des nombres et l'écriture des sorties.

Unités internes : N, mm, N/mm². Lancement des scripts : depuis un autre dossier,
  cd /tmp && /usr/bin/python3.12 -I -B <ce dossier>/sNN_xxx.py
"""
import json
import math
import os
import re

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(ICI)
XLSX = os.path.join(RACINE, "source", "realex_cornieres.xlsx")
SORTIES = os.path.join(ICI, "sorties")
FIGURES = os.path.join(RACINE, "figures")
PROFILES = "/home/user/integ/acier/donnees/profiles.json"
FEUILLE = "P8 "                       # le nom de la feuille finit par une espace

# ------------------------------------------------------------------ cas de contrôle (demande de l'ingénieur)
CAS = dict(
    nom="L120x120x15", larg=120.0, haut=120.0, e=15.0, r=13.0,
    L=300.0,                 # longueur reprise : C8 = 30 cm
    fy=355.0, gM=1.5,        # C3, C10
    R=42760.0,               # C12 = 42,76 kN (ELS)
    X=75.0, Xj=0.0,          # C13, C14 (mm)
    Y=85.0, Yj=0.0,          # C16, C17 (mm)
    n=3, d=16.0,             # C73, C74
    jeu=10.0,                # C76 = 1 cm
    R_ELU=68900.0,           # C71 = 68,9 kN
)
ATTENDU = dict(M1=2.00972, M2=2.56560, M3=2.35870, M4=2.89665, taux=0.8886)


# ------------------------------------------------------------------ méthode historique, en symboles
def historique(p):
    """La méthode du classeur, symbole par symbole (N, mm). Aucune formule du classeur n'est lue."""
    e, r, L = p["e"], p["r"], p["L"]
    R, X, Xj, Y, Yj = p["R"], p["X"], p["Xj"], p["Y"], p["Yj"]
    o = {}
    o["Xe"] = X - r - e                  # bras de R jusqu'à la section 1 (pied du congé)
    o["Xer"] = X - e                     # bras de R jusqu'à la section 2 (nu intérieur de l'aile verticale)
    o["Ye"] = Y - e - r                  # de l'ancrage à la section 3
    o["Yer"] = Y - e                     # de l'ancrage à la section 4
    o["YH"] = Y - e / 2                  # de l'ancrage au plan moyen de l'aile horizontale
    o["Mmax"] = R * (X + Xj)             # moment au nu du mur
    o["rho3"] = (o["Ye"] + Yj) / (o["YH"] + Yj)
    o["rho4"] = (o["Yer"] + Yj) / (o["YH"] + Yj)
    o["H"] = o["Mmax"] / (o["YH"] + Yj)  # effort horizontal implicite (couple ancrage–compression)
    M = [R * (o["Xe"] + Xj), R * (o["Xer"] + Xj), o["rho3"] * o["Mmax"], o["rho4"] * o["Mmax"]]
    t = [e, e + r, e, e + r]
    W = [L * ti ** 2 / 6 for ti in t]
    sig = [m / w for m, w in zip(M, W)]
    tau = [R / (L * ti) for ti in t]
    seq = [math.sqrt(s * s + 3 * u * u) for s, u in zip(sig, tau)]
    sadm = p["fy"] / p["gM"]
    tadm = sadm / math.sqrt(3)
    taux = [s / sadm for s in seq]
    o.update(M=M, t=t, W=W, sig=sig, tau=tau, seq=seq, sadm=sadm, tadm=tadm, taux=taux,
             taux_sig=[s / sadm for s in sig], taux_tau=[u / tadm for u in tau])
    o["taux_max"] = max(taux)
    o["section_max"] = 1 + taux.index(max(taux))
    # ancrages : C82 (bras Y − jeu), C83 (R/n), C85/C86 (× R_ELU/R)
    o["N_els"] = o["Mmax"] / ((Y - p["jeu"]) * p["n"])
    o["V_els"] = R / p["n"]
    o["rapport_elu"] = p["R_ELU"] / R
    o["N_elu"] = o["rapport_elu"] * o["N_els"]
    o["V_elu"] = o["rapport_elu"] * o["V_els"]
    o["p_diam"] = o["V_els"] / (p["d"] * e)                        # C88
    o["arrach"] = o["V_els"] / (2 * (p["haut"] - Y) * e)           # C90
    return o


def cellules_modele(p):
    """La méthode en symboles, rendue cellule par cellule dans les UNITÉS DU CLASSEUR."""
    h = historique(p)
    kNm = 1e-6
    c = {
        "C4": p["larg"], "C5": p["haut"], "C6": p["e"], "C7": p["r"],
        "C19": h["Xe"], "A20": p["larg"] - p["X"], "C20": h["Xer"], "A21": p["haut"] - p["Y"],
        "C23": h["Ye"], "C24": h["Yer"], "C25": h["YH"],
        "C55": h["Mmax"] * kNm, "C63": h["Mmax"] * kNm, "C78": h["Mmax"] * kNm,
        "C56": h["rho3"], "C64": h["rho4"],
        "D3": h["taux_max"],
        "C70": p["R"] / 1e3, "D73": p["L"] / 10, "C75": p["Y"] / 10,
        "C82": h["N_els"] / 1e3, "C83": h["V_els"] / 1e3, "C85": h["N_elu"] / 1e3, "C86": h["V_elu"] / 1e3,
        "C88": h["p_diam"], "C90": h["arrach"],
    }
    lignes = {1: 31, 2: 41, 3: 51, 4: 61}
    for k, l0 in lignes.items():
        i = k - 1
        c[f"C{l0}"] = h["t"][i]
        c[f"C{l0 + 1}"] = h["W"][i]
        c[f"F{l0}"] = h["M"][i] * kNm
        c[f"F{l0 + 1}"] = h["W"][i]
        c[f"G{l0}"] = h["sig"][i]
        c[f"H{l0}"] = h["sadm"]
        c[f"I{l0}"] = h["taux_sig"][i]
        c[f"F{l0 + 3}"] = p["R"] / 1e3
        c[f"G{l0 + 3}"] = h["tau"][i]
        c[f"H{l0 + 3}"] = h["tadm"]
        c[f"I{l0 + 3}"] = h["taux_tau"][i]
        c[f"G{l0 + 6}"] = h["seq"][i]
        c[f"H{l0 + 6}"] = h["sadm"]
        c[f"I{l0 + 6}"] = h["taux"][i]
    c["C33"], c["C34"], c["C35"] = h["Xe"], p["Xj"], h["M"][0] * kNm
    c["C43"], c["C44"], c["C45"] = h["Xer"], p["Xj"], h["M"][1] * kNm
    c["C53"], c["C54"], c["C57"] = p["X"], p["Xj"], h["M"][2] * kNm
    c["C65"] = h["M"][3] * kNm
    return c


# ------------------------------------------------------------------ interpréteur des formules du classeur
class Classeur:
    """Relit les formules NUMÉRIQUES de P8 et les recalcule pour une saisie quelconque.

    Fonctions reconnues : références (avec ou sans $ et nom de feuille), plages de CORNIERES, IF, VLOOKUP exact
    (première ligne trouvée, comme Excel), MAX, SQRT, FALSE(), opérateurs + − × / ^ et comparaison « = ».
    Les formules de texte (&, TEXT) sont ignorées.
    """

    def __init__(self, chemin=XLSX):
        import openpyxl
        wf = openpyxl.load_workbook(chemin, data_only=False)
        wv = openpyxl.load_workbook(chemin, data_only=True)
        self.noms = wf.sheetnames
        self.formules, self.constantes, self.cache = {}, {}, {}
        self.autres = {}
        for nom in wf.sheetnames:
            f, v = wf[nom], wv[nom]
            for ligne in f.iter_rows():
                for cel in ligne:
                    if cel.value is None:
                        continue
                    cle = (nom, cel.coordinate)
                    val = v[cel.coordinate].value
                    if isinstance(cel.value, str) and cel.value.startswith("="):
                        self.formules[cle] = cel.value
                        self.cache[cle] = val
                    else:
                        self.constantes[cle] = cel.value
        self.cat = []
        ws = wf["CORNIERES"]
        for row in ws.iter_rows(min_row=2, max_row=59, max_col=5, values_only=True):
            self.cat.append(row)
        self.saisies = {}

    # --- traduction
    @staticmethod
    def traduire(fx):
        s = fx[1:]
        if "&" in s or "TEXT(" in s:
            return None
        s = s.replace("$", "").replace("'P8 '!", "")
        s = re.sub(r"CORNIERES!([A-Z]+\d+):([A-Z]+\d+)", r"CAT", s)
        s = s.replace("FALSE()", "False")
        s = re.sub(r"\b([A-Z]{1,2})(\d+)\b", r'V("\1\2")', s)
        s = s.replace('")="', '")=="')
        s = s.replace("^", "**")
        s = s.replace("IF(", "IF_(").replace("VLOOKUP(", "VLOOKUP_(").replace("MAX(", "max(").replace("SQRT(", "SQRT_(")
        return s

    def evaluer(self, saisies=None):
        """Recalcule toutes les formules numériques de P8 avec des saisies modifiées {cellule: valeur}."""
        saisies = saisies or {}
        memo = {}
        cat = self.cat

        def V(ref):
            if ref in saisies:
                return saisies[ref]
            if ref in memo:
                return memo[ref]
            cle = (FEUILLE, ref)
            if cle in self.formules:
                expr = self.traduire(self.formules[cle])
                val = None if expr is None else eval(expr, env)
                memo[ref] = val
                return val
            return self.constantes.get(cle)

        def IF_(c, a, b):
            return a if c else b

        def VLOOKUP_(cle, plage, col, exact):
            for row in cat:
                if row[0] == cle:
                    return row[col - 1]
            return float("nan")

        env = dict(V=V, IF_=IF_, VLOOKUP_=VLOOKUP_, CAT=None, SQRT_=math.sqrt, max=max, False_=False)
        env["__builtins__"] = {}
        res = {}
        for (nom, ref), fx in self.formules.items():
            if nom != FEUILLE:
                continue
            if self.traduire(fx) is None:
                continue
            try:
                res[ref] = V(ref)
            except ZeroDivisionError:
                res[ref] = float("nan")
        return res


def saisies_classeur(p, nom=None):
    """Saisies de P8 correspondant à un jeu de paramètres (unités du classeur)."""
    s = {"C3": p["fy"], "C8": p["L"] / 10, "C10": p["gM"], "C12": p["R"] / 1e3, "C13": p["X"], "C14": p["Xj"],
         "C16": p["Y"], "C17": p["Yj"], "C71": p["R_ELU"] / 1e3, "C73": p["n"], "C74": p["d"], "C76": p["jeu"] / 10}
    if nom is None:
        s.update({"B3": "NON STANDARD", "K4": p["larg"], "K5": p["haut"], "K6": p["e"], "K7": p["r"]})
    else:
        s["B3"] = nom
    return s


# ------------------------------------------------------------------ géométrie du congé
def t_conge(s, e, r):
    """Épaisseur d'une aile à la distance s du DOS de l'autre aile (congé de rayon r centré en (e + r ; e + r)).

    s ≥ e + r : e ; e ≤ s ≤ e + r : e + r − √(r² − (e + r − s)²) ; s < e : bloc d'angle (None).
    """
    if s >= e + r:
        return e
    if s < e:
        return None
    w = e + r - s
    return e + r - math.sqrt(max(r * r - w * w, 0.0))


# ------------------------------------------------------------------ corps libre (configuration A)
def corps_libre_A(p, yP, uV):
    """Cornière isolée, aile horizontale en bas. Frottement au mur nul.

    R vers le bas à u = X' ; ancrage à y = Y' : traction H (vers le mur), cisaillement V = R (vers le haut) de
    ligne d'action u = uV ; compression du mur C = H au point (0 ; yP).
    Moments autour du pivot : H·(Y' − yP) = R·(X' − uV).
    Aile verticale, coupe à la hauteur y (e + r ≤ y < Y') : M(y) = H·(Y' − y) − R·(e/2 − uV) ; N = +R ; V = H.
    """
    e, R = p["e"], p["R"]
    Xp, Yp = p["X"] + p["Xj"], p["Y"] + p["Yj"]
    H = R * (Xp - uV) / (Yp - yP)
    M = lambda y: H * (Yp - y) - R * (e / 2 - uV) if y < Yp else 0.0
    return dict(H=H, Xp=Xp, Yp=Yp, M=M, N=R, V=H, yP=yP, uV=uV)


def taux_section_v(p, M, N, V, t=None):
    """Taux d'une coupe de l'aile verticale : σ = M/W + N/A (fibre côté congé), τ = V/A ; σ_éq/σ_adm."""
    t = p["e"] if t is None else t
    W, A = p["L"] * t * t / 6, p["L"] * t
    s = M / W + N / A
    u = V / A
    return math.sqrt(s * s + 3 * u * u) / (p["fy"] / p["gM"]), s, u


# ------------------------------------------------------------------ catalogue du site (lecture seule)
def profils_site(familles=("L", "LI")):
    with open(PROFILES, encoding="utf-8") as fh:
        d = json.load(fh)
    return [q for q in d["profils"] if q.get("famille") in familles]


# ------------------------------------------------------------------ sorties
def fr(x, nd=2):
    """Nombre à la française : virgule décimale, espace fine insécable pour les milliers."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    s = f"{x:,.{nd}f}"
    return s.replace(",", " ").replace(".", ",")


def pc(x, nd=2):
    return fr(100 * x, nd) + " %"


class Sortie:
    """Collecte des lignes de texte et des valeurs nommées ; écrit sNN_xxx.txt et sNN_xxx.json."""

    def __init__(self, nom):
        self.nom = nom
        self.lignes = []
        self.val = {}

    def __call__(self, *a):
        self.lignes.append(" ".join(str(x) for x in a))

    def v(self, cle, valeur):
        self.val[cle] = valeur
        return valeur

    def ecrire(self):
        os.makedirs(SORTIES, exist_ok=True)
        with open(os.path.join(SORTIES, self.nom + ".txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(self.lignes) + "\n")
        with open(os.path.join(SORTIES, self.nom + ".json"), "w", encoding="utf-8") as fh:
            json.dump(self.val, fh, ensure_ascii=False, indent=1, default=float)
        print("\n".join(self.lignes))

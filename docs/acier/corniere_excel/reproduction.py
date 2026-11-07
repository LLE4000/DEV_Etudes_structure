# -*- coding: utf-8 -*-
"""Reproduction INDÉPENDANTE de la feuille P8 du classeur REALEX « Dimensionnement cornières ».

Trois parties :

A. LE MODÈLE, écrit à partir des symboles du schéma (N, mm), sans lire une seule formule du classeur :
   géométrie des sections 1 à 4, moments, contraintes, taux, efforts d'ancrage. Il est comparé à CHAQUE valeur
   numérique en cache de P8 (93 cellules), écart relatif cellule par cellule.

B. LE CLASSEUR RÉÉVALUÉ : un petit interpréteur relit les formules Excel de P8 telles qu'elles sont écrites
   (références, MAX, IF, VLOOKUP, SQRT, ^) ; il redonne d'abord les valeurs en cache (preuve qu'il lit juste), puis
   sert d'arbitre pour les cas que le classeur n'a jamais calculés : jeux Xj et Yj non nuls, et 500 cas tirés au
   hasard (cornière, nuance, longueur, charge, X, Xj, Y, Yj, ancrages) où le modèle A doit retrouver le classeur.

C. L'EFFET DES JEUX Xj et Yj sur chaque grandeur (0 / 20 mm, seuls et ensemble), et le jeu qui porte le taux à 100 %.

Lancement : cd /tmp && /usr/bin/python3.12 -I <ce dossier>/reproduction.py   → <ce dossier>/sorties/reproduction.txt
"""
import math
import os
import random
import re

import openpyxl

ICI = os.path.dirname(os.path.abspath(__file__))
XLSX = os.path.join(ICI, "source", "realex_cornieres.xlsx")
SORTIE = os.path.join(ICI, "sorties", "reproduction.txt")
P8 = "P8 "

# =====================================================================================================================
# A. LE MODÈLE (symboles du schéma ; unités de calcul : N et mm)
# =====================================================================================================================
# Cas de contrôle donné par l'ingénieur
CAS = dict(larg=120.0, haut=120.0, e=15.0, r=13.0,   # cornière L120×120×15, congé r (mm)
           L=300.0,                                  # longueur de cornière reprise (mm)
           fy=355.0, gM=1.5,                         # nuance (N/mm²), coefficient appliqué à fy
           R=42.76e3,                                # réaction ELS (N)
           X=75.0, Xj=0.0,                           # excentrement de la charge depuis le dos de l'aile verticale, jeu
           Y=85.0, Yj=0.0,                           # hauteur de l'axe d'ancrage depuis le dessous de la cornière, jeu
           n=3, d=16.0, jeu_anc=10.0,                # ancrages : nombre, diamètre (mm), jeu pour le bras de levier (mm)
           R_ELU=68.9e3)                             # réaction ELU donnée (logiciel Hilti) (N)


def modele(p):
    """Toutes les grandeurs de la méthode, à partir des seuls symboles. Rend un dict en unités du classeur
    (mm, mm³, kN, kN·m, N/mm², taux sans dimension)."""
    larg, haut, e, r, L = p["larg"], p["haut"], p["e"], p["r"], p["L"]
    fy, gM, R, X, Xj, Y, Yj = p["fy"], p["gM"], p["R"], p["X"], p["Xj"], p["Y"], p["Yj"]
    # --- géométrie : position des sections critiques ---
    # aile horizontale : section 1 au pied du congé (à e + r du dos), section 2 au nu intérieur de l'aile verticale
    Xe = X - (e + r)            # bras de la charge jusqu'à la section 1
    Xer = X - e                 # bras de la charge jusqu'à la section 2 (« Xe+r » = Xe + r)
    # aile verticale : distances depuis l'axe d'ancrage vers le bas
    Ye = Y - (e + r)            # jusqu'à la section 3 (pied du congé, à e + r au-dessus du dessous)
    Yer = Y - e                 # jusqu'à la section 4 (niveau de la face supérieure de l'aile horizontale)
    YH = Y - e / 2              # jusqu'au plan moyen de l'aile horizontale (le « pivot » du modèle)
    # --- modules de flexion d'une lame de largeur L ---
    W_e = L * e ** 2 / 6        # épaisseur e (sections 1 et 3)
    W_er = L * (e + r) ** 2 / 6  # épaisseur e + r (sections 2 et 4, à travers le congé)
    # --- moments ---
    M1 = R * (Xe + Xj)
    M2 = R * (Xer + Xj)
    Mmax = R * (X + Xj)          # moment de la charge au nu du mur
    # aile verticale : l'effort horizontal H équilibre Mmax sur le bras (YH + Yj) ; M(s) = H·s depuis l'ancrage
    H = Mmax / (YH + Yj)
    M3 = H * (Ye + Yj)
    M4 = H * (Yer + Yj)
    # --- contraintes ---
    s1, s2, s3, s4 = M1 / W_e, M2 / W_er, M3 / W_e, M4 / W_er
    t_e = R / (L * e)            # cisaillement moyen, épaisseur e
    t_er = R / (L * (e + r))     # cisaillement moyen, épaisseur e + r
    sadm = fy / gM
    tadm = sadm / math.sqrt(3)
    eq = lambda s, t: math.sqrt(s * s + 3 * t * t)
    q1, q2, q3, q4 = eq(s1, t_e), eq(s2, t_er), eq(s3, t_e), eq(s4, t_er)
    # --- ancrages ---
    n, d, jeu, R_ELU = p["n"], p["d"], p["jeu_anc"], p["R_ELU"]
    N_ELS = Mmax / (Y - jeu) / n     # traction par ancrage : Mmax sur le bras (Y − jeu), partagé entre n ancrages
    V_ELS = R / n
    k = R_ELU / R
    p_diam = V_ELS / (d * e)                 # pression diamétrale (ELS) sur l'aile verticale
    t_arr = V_ELS / (2 * (haut - Y) * e)     # cisaillement de deux plans au-dessus du trou (ELS)
    kN, kNm = 1e-3, 1e-6
    return {
        "larg": larg, "haut": haut, "e": e, "r": r, "e+r": e + r, "X": X, "Xj": Xj,
        "Xe": Xe, "Xe+r": Xer, "larg-X": larg - X, "haut-Y": haut - Y, "Ye": Ye, "Ye+r": Yer, "YH": YH,
        "W_e": W_e, "W_er": W_er, "R": R * kN, "M1": M1 * kNm, "M2": M2 * kNm, "Mmax": Mmax * kNm,
        "H": H * kN, "rho3": (Ye + Yj) / (YH + Yj), "rho4": (Yer + Yj) / (YH + Yj), "M3": M3 * kNm, "M4": M4 * kNm,
        "sigma1": s1, "sigma2": s2, "sigma3": s3, "sigma4": s4, "tau_e": t_e, "tau_er": t_er,
        "seq1": q1, "seq2": q2, "seq3": q3, "seq4": q4, "sadm": sadm, "tadm": tadm,
        "taux_s1": s1 / sadm, "taux_s2": s2 / sadm, "taux_s3": s3 / sadm, "taux_s4": s4 / sadm,
        "taux_t_e": t_e / tadm, "taux_t_er": t_er / tadm,
        "taux1": q1 / sadm, "taux2": q2 / sadm, "taux3": q3 / sadm, "taux4": q4 / sadm,
        "taux_max": max(q1, q2, q3, q4) / sadm,
        "L_cm": L / 10, "Y_cm": Y / 10, "N_ELS": N_ELS * kN, "V_ELS": V_ELS * kN,
        "N_ELU": N_ELS * k * kN, "V_ELU": V_ELS * k * kN, "p_diam": p_diam, "tau_arr": t_arr,
    }


# cellule numérique de P8 → grandeur du modèle (les copies d'affichage pointent vers la même grandeur)
CORRESPONDANCE = {
    "C4": "larg", "C5": "haut", "C6": "e", "C7": "r", "C19": "Xe", "A20": "larg-X", "C20": "Xe+r", "A21": "haut-Y",
    "C23": "Ye", "C24": "Ye+r", "C25": "YH",
    # section 1
    "C31": "e", "C32": "W_e", "C33": "Xe", "C34": "Xj", "C35": "M1", "F31": "M1", "F32": "W_e", "G31": "sigma1",
    "H31": "sadm", "I31": "taux_s1", "F34": "R", "G34": "tau_e", "H34": "tadm", "I34": "taux_t_e",
    "G37": "seq1", "H37": "sadm", "I37": "taux1",
    # section 2
    "C41": "e+r", "C42": "W_er", "C43": "Xe+r", "C44": "Xj", "C45": "M2", "F41": "M2", "F42": "W_er", "G41": "sigma2",
    "H41": "sadm", "I41": "taux_s2", "F44": "R", "G44": "tau_er", "H44": "tadm", "I44": "taux_t_er",
    "G47": "seq2", "H47": "sadm", "I47": "taux2",
    # section 3
    "C51": "e", "C52": "W_e", "C53": "X", "C54": "Xj", "C55": "Mmax", "C56": "rho3", "C57": "M3", "F51": "M3",
    "F52": "W_e", "G51": "sigma3", "H51": "sadm", "I51": "taux_s3", "F54": "R", "G54": "tau_e", "H54": "tadm",
    "I54": "taux_t_e", "G57": "seq3", "H57": "sadm", "I57": "taux3",
    # section 4
    "C61": "e+r", "C62": "W_er", "C63": "Mmax", "C64": "rho4", "C65": "M4", "F61": "M4", "F62": "W_er",
    "G61": "sigma4", "H61": "sadm", "I61": "taux_s4", "F64": "R", "G64": "tau_er", "H64": "tadm", "I64": "taux_t_er",
    "G67": "seq4", "H67": "sadm", "I67": "taux4",
    # synthèse et ancrages
    "D3": "taux_max", "C70": "R", "D73": "L_cm", "C75": "Y_cm", "C78": "Mmax", "C82": "N_ELS", "C83": "V_ELS",
    "C85": "N_ELU", "C86": "V_ELU", "C88": "p_diam", "C90": "tau_arr",
}
HORS_MODELE = {"L22": "3,7 × 0,19 × 1 230 : cellule isolée, lue par rien, sans lien avec la cornière"}

# =====================================================================================================================
# B. L'INTERPRÉTEUR DES FORMULES DU CLASSEUR (arbitre des cas non calculés)
# =====================================================================================================================
TOK = re.compile(r"""
  (?P<str>"[^"]*")
| (?P<ref>(?:'[^']+'!|[A-Za-z0-9_]+!)?\$?[A-Z]{1,3}\$?\d+(?::\$?[A-Z]{1,3}\$?\d+)?)
| (?P<num>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)
| (?P<func>[A-Z][A-Z0-9.]*(?=\())
| (?P<op><=|>=|<>|[-+*/^&=<>(),])
| (?P<sp>\s+)
""", re.X)
UNREF = re.compile(r"(?:'([^']+)'!|([A-Za-z0-9_]+)!)?\$?([A-Z]{1,3})\$?(\d+)(?::\$?([A-Z]{1,3})\$?(\d+))?")


def _col(c):
    n = 0
    for ch in c:
        n = n * 26 + ord(ch) - 64
    return n


def _lettre(n):
    s = ""
    while n:
        n, m = divmod(n - 1, 26)
        s = chr(65 + m) + s
    return s


def traduire(formule, feuille):
    """Formule Excel → expression Python (None pour une formule de texte)."""
    if "&" in formule:
        return None
    out, pos, f = [], 0, formule[1:]
    while pos < len(f):
        m = TOK.match(f, pos)
        if not m:
            raise ValueError(f"jeton illisible dans {formule!r} à {pos}")
        pos = m.end()
        k, v = m.lastgroup, m.group()
        if k == "sp":
            continue
        if k == "str":
            out.append(repr(v[1:-1]))
        elif k == "num":
            out.append(v)
        elif k == "func":
            out.append(f"_f_{v.replace('.', '_')}")
        elif k == "ref":
            r = UNREF.fullmatch(v)
            fe = r.group(1) or r.group(2) or feuille
            if r.group(5):
                out.append(f"_r({fe!r},{r.group(3)!r},{int(r.group(4))},{r.group(5)!r},{int(r.group(6))})")
            else:
                out.append(f"_v({fe!r},'{r.group(3)}{int(r.group(4))}')")
        else:
            out.append({"^": "**", "=": "==", "<>": "!="}.get(v, v))
    return "".join(out)


class Classeur:
    """Les cellules du classeur ; une formule est réévaluée à la demande avec des saisies modifiées."""

    def __init__(self, chemin):
        wf = openpyxl.load_workbook(chemin, data_only=False)
        wv = openpyxl.load_workbook(chemin, data_only=True)
        self.const, self.form, self.cache = {}, {}, {}
        for ws in wf.worksheets:
            for row in ws.iter_rows():
                for c in row:
                    if c.value is None:
                        continue
                    cle = (ws.title, c.coordinate)
                    self.cache[cle] = wv[ws.title][c.coordinate].value
                    if isinstance(c.value, str) and c.value.startswith("="):
                        self.form[cle] = traduire(c.value, ws.title)
                    else:
                        self.const[cle] = c.value

    def evaluer(self, saisies=None):
        """Rend une fonction valeur(feuille, adresse) pour un jeu de saisies {adresse P8: valeur}."""
        memo = {}
        sais = {(P8, k): v for k, v in (saisies or {}).items()}

        def _v(fe, ad):
            cle = (fe, ad)
            if cle in sais:
                return sais[cle]
            if cle in memo:
                return memo[cle]
            if cle in self.form and self.form[cle] is not None:
                val = eval(self.form[cle], env)
            else:
                val = self.const.get(cle)
            memo[cle] = val
            return val

        def _r(fe, c1, l1, c2, l2):
            return [[_v(fe, f"{_lettre(ci)}{li}") for ci in range(_col(c1), _col(c2) + 1)] for li in range(l1, l2 + 1)]

        def _f_VLOOKUP(cle, plage, col, exact):
            for ligne in plage:
                a = ligne[0]
                if (a.upper() if isinstance(a, str) else a) == (cle.upper() if isinstance(cle, str) else cle):
                    v = ligne[col - 1]
                    return 0 if v is None else v
            raise LookupError(f"#N/A : {cle!r}")

        env = {"_v": _v, "_r": _r, "_f_VLOOKUP": _f_VLOOKUP, "_f_IF": lambda c, a, b: a if c else b,
               "_f_MAX": lambda *a: max(a), "_f_SQRT": math.sqrt, "_f_FALSE": lambda: False,
               "__builtins__": {}}
        return _v


# saisies P8 ↔ symboles du modèle (unités du classeur : C8 en cm, C12 et C71 en kN, C76 en cm)
def saisies_classeur(p, nom):
    s = {"B3": nom, "C3": p["fy"], "C8": p["L"] / 10, "C10": p["gM"], "C12": p["R"] / 1e3, "C13": p["X"],
         "C14": p["Xj"], "C16": p["Y"], "C17": p["Yj"], "C71": p["R_ELU"] / 1e3, "C73": p["n"], "C74": p["d"],
         "C76": p["jeu_anc"] / 10}
    if nom == "NON STANDARD":
        s.update({"K4": p["larg"], "K5": p["haut"], "K6": p["e"], "K7": p["r"]})
    return s


def ecart(a, b):
    if a == b:
        return 0.0
    return abs(a - b) / max(abs(a), abs(b))


# =====================================================================================================================
def main():
    L = []
    p = L.append
    cl = Classeur(XLSX)
    m0 = modele(CAS)

    # ---------------------------------------------------------------- A. modèle face au cache
    p("REPRODUCTION DE LA FEUILLE P8 — produit par reproduction.py")
    p("")
    p("== A. Le modèle (symboles, N et mm) face à CHAQUE valeur numérique en cache de P8 ==")
    num = [(ad, v) for (fe, ad), v in cl.cache.items() if fe == P8 and (fe, ad) in cl.form and isinstance(v, (int, float))]
    p(f"cellules de formule numériques dans P8 : {len(num)} ; cellules de formule de texte : "
      f"{sum(1 for (fe, ad), f in cl.form.items() if fe == P8 and f is None)}")
    pire = (0.0, None)
    absentes = []
    p(f"{'cellule':8s} {'grandeur':10s} {'cache classeur':>22s} {'modèle':>22s} {'écart relatif':>14s}")
    ordre = sorted(num, key=lambda x: (int(re.sub(r'\D', '', x[0])), x[0]))
    for ad, v in ordre:
        if ad in HORS_MODELE:
            p(f"{ad:8s} {'—':10s} {v!r:>22} {'(hors modèle)':>22s}   {HORS_MODELE[ad]} ; 3.7*0.19*1230 = {3.7 * 0.19 * 1230!r}")
            continue
        g = CORRESPONDANCE.get(ad)
        if g is None:
            absentes.append(ad)
            continue
        mv = m0[g]
        er = ecart(v, mv)
        if er > pire[0]:
            pire = (er, ad)
        p(f"{ad:8s} {g:10s} {v!r:>22} {mv!r:>22} {er:14.2e}")
    p(f"cellules sans correspondance : {absentes or 'aucune'}")
    p(f"PLUS GRAND ÉCART RELATIF modèle / cache : {pire[0]:.2e} (cellule {pire[1]})")
    p("")
    p("Valeurs attendues par l'ingénieur :")
    for g, attendu in (("M1", 2.00972), ("M2", 2.56560), ("M3", 2.35870), ("M4", 2.89665), ("taux_max", 0.8886)):
        p(f"   {g:8s} modèle = {m0[g]:.6f}   attendu = {attendu}   écart = {m0[g] - attendu:+.2e}")
    gouv = max(("taux1", "taux2", "taux3", "taux4"), key=lambda k: m0[k])
    p(f"   section gouvernante : {gouv} ({m0[gouv]:.4%})")
    p("")
    p("Pourquoi les quatre moments diffèrent (cas de contrôle) :")
    R_kN, r_m = CAS["R"] / 1e3, CAS["r"] / 1e3
    p(f"   aile horizontale, effort R = {R_kN} kN : M2 − M1 = {m0['M2'] - m0['M1']:.5f} kN·m = R·r = {R_kN * r_m:.5f}")
    p(f"   aile verticale, effort H = R·(X+Xj)/(YH+Yj) = {m0['H']:.5f} kN (H/R = {m0['H'] / R_kN:.5f} = X/YH = "
      f"{CAS['X'] / m0['YH']:.5f}) : M4 − M3 = {m0['M4'] - m0['M3']:.5f} kN·m = H·r = {m0['H'] * r_m:.5f}")
    p(f"   M3/M1 = {m0['M3'] / m0['M1']:.5f} = (H/R)·(Ye/Xe) = {m0['H'] / R_kN:.5f} × {m0['Ye'] / m0['Xe']:.5f}")
    p(f"   nœud : Mmax = R·(X+Xj) = {m0['Mmax']:.5f} = H·(YH+Yj) = {m0['H'] * m0['YH'] / 1e3:.5f} kN·m")
    p(f"   W(e+r)/W(e) = ((e+r)/e)² = {m0['W_er'] / m0['W_e']:.5f} ; σ2/σ1 = {m0['sigma2'] / m0['sigma1']:.5f} ; "
      f"σ4/σ3 = {m0['sigma4'] / m0['sigma3']:.5f}")
    p(f"   ordre des moments : " + " < ".join(f"{k} = {m0[k]:.5f}" for k in sorted(("M1", "M2", "M3", "M4", "Mmax"), key=lambda k: m0[k])))

    # ---------------------------------------------------------------- B. l'interpréteur face au cache
    p("")
    p("== B1. L'interpréteur des formules du classeur face au cache (il doit le redonner) ==")
    val = cl.evaluer()
    pire_b = 0.0
    n_b = 0
    for (fe, ad), v in cl.cache.items():
        if (fe, ad) in cl.form and cl.form[(fe, ad)] is not None:
            w = val(fe, ad)
            if isinstance(v, (int, float)):
                pire_b = max(pire_b, ecart(v, w))
                n_b += 1
    p(f"{n_b} formules numériques des trois feuilles réévaluées ; plus grand écart au cache : {pire_b:.2e}")

    # ---------------------------------------------------------------- B2. équivalence sur des cas tirés au hasard
    p("")
    p("== B2. Équivalence modèle ⇔ classeur sur 500 cas tirés au hasard (graine fixe) ==")
    ws = openpyxl.load_workbook(XLSX, data_only=True)["CORNIERES"]
    cat = []
    for i in range(2, 59):
        cat.append((ws.cell(i, 1).value, ws.cell(i, 2).value, ws.cell(i, 3).value, ws.cell(i, 4).value, ws.cell(i, 5).value))
    premier = {}
    for nom, a, b, t, r in cat:
        premier.setdefault(nom, (a, b, t, r))       # VLOOKUP exact : la PREMIÈRE ligne du nom
    rnd = random.Random(20261107)
    pire_c = (0.0, None, None)
    n_cmp = 0
    for k in range(500):
        if rnd.random() < 0.2:
            nom = "NON STANDARD"
            t = rnd.choice([4, 5, 6, 8, 10, 12, 15, 20])
            r = rnd.choice([0, 4, 6.5, 10, 13])
            a, b = rnd.uniform(t + r + 20, 250), rnd.uniform(t + r + 20, 250)
        else:
            nom = rnd.choice(sorted(premier))
            a, b, t, r = premier[nom]
        q = dict(CAS, larg=a, haut=b, e=t, r=r, L=rnd.choice([100, 200, 300, 500, 1000]),
                 fy=rnd.choice([235, 355]), gM=rnd.choice([1.0, 1.1, 1.5]), R=rnd.uniform(1e3, 1e5),
                 X=rnd.uniform(t + r + 1, a), Xj=rnd.choice([0, 5, 10, 20, 30]),
                 Y=rnd.uniform(t + r + 2, b - 1), Yj=rnd.choice([0, 5, 10, 20, 30]),
                 n=rnd.choice([1, 2, 3, 4, 6]), d=rnd.choice([8, 10, 12, 16, 20]), jeu_anc=rnd.choice([0, 5, 10, 20]),
                 R_ELU=0)
        q["R_ELU"] = q["R"] * rnd.uniform(1.3, 1.7)
        if q["Y"] - q["jeu_anc"] <= 1:
            q["jeu_anc"] = 0.0
        mm = modele(q)
        vv = cl.evaluer(saisies_classeur(q, nom))
        for ad, g in CORRESPONDANCE.items():
            w = vv(P8, ad)
            er = ecart(w, mm[g])
            n_cmp += 1
            if er > pire_c[0]:
                pire_c = (er, ad, k)
    p(f"{n_cmp} comparaisons (500 cas × {len(CORRESPONDANCE)} cellules) ; plus grand écart relatif : {pire_c[0]:.2e} "
      f"(cellule {pire_c[1]}, cas {pire_c[2]})")

    # ---------------------------------------------------------------- C. effet des jeux
    p("")
    p("== C. Effet des jeux Xj et Yj (cas de contrôle) ==")
    variantes = [("base Xj=0 Yj=0", 0.0, 0.0), ("Xj=20", 20.0, 0.0), ("Yj=20", 0.0, 20.0), ("Xj=20 Yj=20", 20.0, 20.0)]
    res = []
    for nom_v, xj, yj in variantes:
        q = dict(CAS, Xj=xj, Yj=yj)
        mm = modele(q)
        vv = cl.evaluer(saisies_classeur(q, "L120x120x15"))
        acc = max(ecart(vv(P8, ad), mm[g]) for ad, g in CORRESPONDANCE.items())
        res.append((nom_v, mm, acc))
    p("accord modèle / classeur réévalué pour chaque variante : "
      + " ; ".join(f"{n} {a:.1e}" for n, _, a in res))
    lignes = [("M1", "kN·m", "C35"), ("M2", "kN·m", "C45"), ("Mmax", "kN·m", "C55=C63=C78"), ("rho3", "", "C56"),
              ("M3", "kN·m", "C57"), ("rho4", "", "C64"), ("M4", "kN·m", "C65"), ("H", "kN", "(implicite)"),
              ("sigma1", "N/mm²", "G31"), ("sigma2", "N/mm²", "G41"), ("sigma3", "N/mm²", "G51"), ("sigma4", "N/mm²", "G61"),
              ("tau_e", "N/mm²", "G34=G54"), ("tau_er", "N/mm²", "G44=G64"),
              ("seq1", "N/mm²", "G37"), ("seq2", "N/mm²", "G47"), ("seq3", "N/mm²", "G57"), ("seq4", "N/mm²", "G67"),
              ("taux1", "", "I37"), ("taux2", "", "I47"), ("taux3", "", "I57"), ("taux4", "", "I67"),
              ("taux_max", "", "D3"), ("N_ELS", "kN", "C82"), ("V_ELS", "kN", "C83"), ("N_ELU", "kN", "C85"),
              ("V_ELU", "kN", "C86"), ("p_diam", "N/mm²", "C88"), ("tau_arr", "N/mm²", "C90")]
    p(f"{'grandeur':9s} {'unité':6s} {'cellule':12s} " + " ".join(f"{n:>24s}" for n, _, _ in res))
    for g, u, c in lignes:
        b0 = res[0][1][g]
        cells = []
        for _, mm, _ in res:
            v = mm[g]
            d = (v / b0 - 1) if b0 else 0.0
            if g.startswith("taux"):
                cells.append(f"{v:9.4%} ({d:+7.2%})")
            else:
                cells.append(f"{v:10.5f} ({d:+7.2%})")
        p(f"{g:9s} {u:6s} {c:12s} " + " ".join(f"{x:>24s}" for x in cells))
    # pente et jeu limite
    p("")
    for nom_j in ("Xj", "Yj"):
        f = lambda j: modele(dict(CAS, **{nom_j: j}))["taux_max"]
        pente = (f(1.0) - f(0.0))
        lo, hi = 0.0, 500.0
        for _ in range(200):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) < 1 else (lo, mid)
        jlim = (lo + hi) / 2
        q = dict(CAS, **{nom_j: jlim})
        vv = cl.evaluer(saisies_classeur(q, "L120x120x15"))
        p(f"{nom_j} : +1 mm fait passer le taux max de {f(0.0):.4%} à {f(1.0):.4%} ({pente * 100:+.4f} point) ; "
          f"le taux max atteint 100 % pour {nom_j} = {jlim:.3f} mm (classeur réévalué : D3 = {vv(P8, 'D3'):.6f})")
    p(f"Yj → ∞ : le rapport ρ3 tend vers 1, le taux de la section 3 vers {modele(dict(CAS, Yj=1e9))['taux3']:.4%} (borne)")

    os.makedirs(os.path.dirname(SORTIE), exist_ok=True)
    with open(SORTIE, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

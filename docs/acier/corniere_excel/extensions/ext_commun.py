# -*- coding: utf-8 -*-
"""Socle commun des scripts « modèle étendu » (cornière REALEX).

Unités internes : N, mm, N/mm² (MPa). Les moments sont rendus en kN·m quand
c'est dit.

Repère de la coupe (configuration A, celle du classeur) :
  u = distance horizontale depuis le NU DU MUR (dos de l'aile verticale) ;
  y = hauteur depuis le DESSOUS de la cornière (face inférieure de l'aile
      horizontale).

Rien ici n'est lu dans le classeur : les formules sont réécrites à partir des
symboles ; `releve_excel()` relit seulement les valeurs en cache du relevé
texte pour la comparaison.
"""
import math
import os
import re

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(ICI)
FIGURES = os.path.join(RACINE, "figures")
RELEVE = os.path.join(RACINE, "releve_cellules.txt")

# ------------------------------------------------------------------ cas de contrôle
CAS = dict(
    nom="L120x120x15", larg=120.0, haut=120.0, e=15.0, r=13.0, fy=355.0,
    L=300.0,            # longueur reprise (mm) = C8 (cm) × 10
    gM=1.5,             # C10
    R=42760.0,          # C12 (kN → N)
    X=75.0, Xj=0.0,     # C13, C14
    Y=85.0, Yj=0.0,     # C16, C17
    n_anc=3, jeu_anc=10.0,  # C73, C76 (cm → mm)
    R_elu=68900.0,      # C71
    d_anc=16.0,         # C74
)

ATTENDU = dict(M1=2.00972, M2=2.56560, M3=2.35870, M4=2.89665, taux_max=0.8886)


def releve_excel(chemin=RELEVE):
    """Valeurs en cache de la feuille P8 (relevé texte) : {cellule: valeur}."""
    vals = {}
    feuille = None
    with open(chemin, encoding="utf-8") as fh:
        for ligne in fh:
            m = re.match(r"===== FEUILLE (\S+)", ligne)
            if m:
                feuille = m.group(1)
                continue
            if feuille != "P8":
                continue
            m = re.match(r"([A-Z]+\d+)\tFORMULE .*\t=> (.+)$", ligne.rstrip("\n"))
            if m:
                try:
                    vals[m.group(1)] = float(m.group(2))
                except ValueError:
                    pass
                continue
            m = re.match(r"([A-Z]+\d+)\tVALEUR (\S+)$", ligne.rstrip("\n"))
            if m:
                try:
                    vals[m.group(1)] = float(m.group(2))
                except ValueError:
                    pass
    return vals


# ------------------------------------------------------------------ méthode du classeur
def excel(c=CAS):
    """La méthode historique, symbole par symbole (cellules citées)."""
    e, r, L = c["e"], c["r"], c["L"]
    Xp = c["X"] + c["Xj"]
    Xe = c["X"] - r - e            # C19
    Xer = c["X"] - e               # C20
    Ye = c["Y"] - e - r            # C23
    Yer = c["Y"] - e               # C24
    YH = c["Y"] - e / 2            # C25
    R = c["R"]
    M1 = R * (Xe + c["Xj"])        # C35  (N·mm)
    M2 = R * (Xer + c["Xj"])       # C45
    Mmax = R * Xp                  # C55 = C63 = C78
    M3 = (Ye + c["Yj"]) / (YH + c["Yj"]) * Mmax   # C56·C55 = C57
    M4 = (Yer + c["Yj"]) / (YH + c["Yj"]) * Mmax  # C64·C63 = C65
    sadm = c["fy"] / c["gM"]       # H31
    tadm = sadm / math.sqrt(3)     # H34
    out = dict(Xe=Xe, Xer=Xer, Ye=Ye, Yer=Yer, YH=YH, Mmax=Mmax,
               M=[M1, M2, M3, M4], sadm=sadm, tadm=tadm)
    t = [e, e + r, e, e + r]       # C31, C41, C51, C61
    W = [L * ti ** 2 / 6 for ti in t]      # C32 …
    sig = [m / w for m, w in zip(out["M"], W)]
    tau = [R / (L * ti) for ti in t]       # G34 …
    seq = [math.sqrt(s ** 2 + 3 * ta ** 2) for s, ta in zip(sig, tau)]
    taux = [s / sadm for s in seq]
    out.update(t=t, W=W, sig=sig, tau=tau, seq=seq, taux=taux,
               taux_max=max(taux), section_max=1 + taux.index(max(taux)))
    # ancrages (C82, C83)
    out["N_anc"] = Mmax / ((c["Y"] - c["jeu_anc"]) * c["n_anc"])
    out["V_anc"] = R / c["n_anc"]
    return out


# ------------------------------------------------------------------ géométrie du congé
def t_conge(s, e, r):
    """Épaisseur locale d'une aile à la distance s du dos de l'AUTRE aile.

    s ≥ e + r : épaisseur courante e ; e ≤ s ≤ e + r : congé de rayon r, centré
    en (e + r, e + r) ; s < e : on est dans le bloc d'angle (renvoie None).
    """
    if s >= e + r:
        return e
    if s < e:
        return None
    w = e + r - s
    return e + r - math.sqrt(max(r * r - w * w, 0.0))


# ------------------------------------------------------------------ corps libre
def corps_libre_A(c=CAS, yP=None, uV=None, R=None):
    """Équilibre d'une cornière isolée, configuration A (aile horizontale en bas).

    Forces : R vers le bas à u = X + Xj (face supérieure de l'aile horizontale) ;
    ancrage à y = Y + Yj : traction H (vers le mur) et cisaillement V = R (vers
    le haut) dont la ligne d'action est à u = uV ; compression du mur C = H au
    pivot (u = 0, y = yP). Frottement négligé (comme le classeur).

    Moments autour du pivot : H·(Y' − yP) = R·(X' − uV).
    """
    e = c["e"]
    R = c["R"] if R is None else R
    yP = e / 2 if yP is None else yP
    uV = 0.0 if uV is None else uV
    Xp = c["X"] + c["Xj"]
    Yp = c["Y"] + c["Yj"]
    H = R * (Xp - uV) / (Yp - yP)
    return dict(R=R, H=H, V=R, C=H, yP=yP, uV=uV, Xp=Xp, Yp=Yp)


def moment_aile_h(u, fb):
    """Moment (N·mm par la longueur reprise) dans l'aile horizontale, section u."""
    return fb["R"] * max(fb["Xp"] - u, 0.0)


def moment_aile_v(y, fb, e):
    """Moment dans l'aile verticale, section à la hauteur y (y ≥ e).

    Partie au-dessus de la coupe : H à Y' (bras Y' − y) et V à uV (bras uV − e/2
    par rapport à l'axe de l'aile, compté avec son signe).
    """
    if y >= fb["Yp"]:
        return 0.0
    return fb["H"] * (fb["Yp"] - y) - fb["V"] * (e / 2 - fb["uV"])


def efforts_A(fb):
    """N et V par aile (configuration A) : signes et sens physiques."""
    return dict(
        aile_h=dict(N=0.0, V=fb["R"], note="porte-à-faux : V = R, N = 0"),
        aile_v=dict(N=+fb["R"], V=fb["H"],
                    note="entre l'angle et l'ancrage : traction N = R, tranchant H"),
    )


def ecrire(chemin, lignes):
    with open(chemin, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lignes) + "\n")


# ------------------------------------------------------------------ charte des figures
COUL = dict(
    s1="#2a78d6", s2="#eb6834", s3="#1baf7a", s4="#6250d6",
    encre="#0b0b0b", encre2="#52514e", muet="#898781", grille="#e1e0d9",
    axe="#c3c2b7", fond="#fcfcfb", neutre="#f0efec", critique="#d03b3b",
    acier="#d9d8d2", beton="#ecebe6",
)


def style_axes(ax):
    ax.set_facecolor(COUL["fond"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(COUL["axe"])
    ax.tick_params(colors=COUL["encre2"], labelsize=8)
    ax.grid(True, color=COUL["grille"], lw=0.6)
    ax.set_axisbelow(True)


def fr(x, nd=2):
    """Nombre à la française (virgule décimale, espace fine pour les milliers)."""
    s = f"{x:,.{nd}f}"
    return s.replace(",", " ").replace(".", ",")

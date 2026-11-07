# -*- coding: utf-8 -*-
"""Relecture adverse — 2. Tous les chiffres analytiques du document, recalculés à partir des formules du classeur
(P8) et de la statique, sans rien importer de l'analyse relue."""
import math

fy, gM = 355.0, 1.5
SADM = fy / gM
TADM = SADM / math.sqrt(3)
BASE = dict(R=42.76, X=75.0, Xj=0.0, Y=85.0, Yj=0.0, e=15.0, r=13.0, L=300.0, n=3, jeu=10.0, larg=120.0, haut=120.0)


def classeur(R=42.76, X=75.0, Xj=0.0, Y=85.0, Yj=0.0, e=15.0, r=13.0, L=300.0, n=3, jeu=10.0, **_):
    """Les formules de P8 (jeu en mm ici : C76·10)."""
    Mmax = R * (X + Xj) / 1000
    M = [R * (X - r - e + Xj) / 1000, R * (X - e + Xj) / 1000,
         Mmax * (Y - e - r + Yj) / (Y - e / 2 + Yj), Mmax * (Y - e + Yj) / (Y - e / 2 + Yj)]
    t = [e, e + r, e, e + r]
    taux = []
    for Mk, tk in zip(M, t):
        s = Mk * 1e6 / (L * tk ** 2 / 6)
        ta = R * 1e3 / (L * tk)
        taux.append(math.sqrt(s ** 2 + 3 * ta ** 2) / SADM)
    Nanc = Mmax / ((Y - jeu) / 1000) / n
    return dict(M=M, taux=taux, Nanc=Nanc, Mmax=Mmax)


def equilibre(yP, uV, R=42.76, X=75.0, Xj=0.0, Y=85.0, Yj=0.0, e=15.0, r=13.0, L=300.0, **_):
    """Corps libre cohérent : pivot (compression du mur) à la hauteur yP, cisaillement d'ancrage à uV du mur.
    Aile verticale à la hauteur y : M = H(Y'-y) - R(e/2-uV), N = +R (traction), V = H."""
    Xp, Yp = X + Xj, Y + Yj
    H = R * (Xp - uV) / (Yp - yP)
    def M(y):
        return (H * (Yp - y) - R * (e / 2 - uV)) / 1000
    def taux(y, t):
        A = L * t
        s = abs(M(y)) * 1e6 / (L * t ** 2 / 6) + R * 1e3 / A
        ta = H * 1e3 / A
        return math.sqrt(s ** 2 + 3 * ta ** 2) / SADM
    return dict(H=H, M3=M(e + r), M4=M(e), taux3=taux(e + r, e), taux4=taux(e, e + r), Nanc=H / 3)


def p(*a):
    print(*a)


c0 = classeur(**BASE)
p("== §3.1 cas de contrôle")
p("  M =", [f"{m:.6f}" for m in c0["M"]], " taux =", [f"{t*100:.4f}" for t in c0["taux"]], " Mmax", c0["Mmax"])
p(f"  sadm {SADM:.4f} tadm {TADM:.4f}")

p("== §3.2")
R, e, r = 42.76, 15.0, 13.0
H = c0["Mmax"] * 1000 / (85 - 7.5)
p(f"  M2-M1 {c0['M'][1]-c0['M'][0]:.5f}  H {H:.4f}  M4-M3 {c0['M'][3]-c0['M'][2]:.5f}  M3/M1 {c0['M'][2]/c0['M'][0]:.5f}"
  f"  H/R {H/R:.5f} Ye/Xe {57/47:.5f}  ((e+r)/e)^2 {(28/15)**2:.4f}")

p("== §1.2 variantes de l'équilibre (N par ancrage, M3, taux 3)")
for nom, yP, uV, moment_classeur in [("classeur sections", 7.5, 0, True), ("classeur ancrages", 10, 0, True),
                                      ("Hilti 15,21", 15.21, 0, True), ("P0", 0, 7.5, False), ("P1", 7.5, 7.5, False),
                                      ("Q1", 7.5, 0, False), ("P15", 15, 7.5, False)]:
    q = equilibre(yP, uV)
    if moment_classeur:
        M3 = q["H"] * 57 / 1000
        s = M3 * 1e6 / 11250
        t = math.sqrt(s ** 2 + 3 * (R * 1e3 / 4500) ** 2) / SADM
        p(f"  {nom:20s} H {q['H']:.3f}  N/anc {q['H']/3:.3f}  M3(H·Ye) {M3:.5f}  taux3 {t*100:.2f}")
    else:
        p(f"  {nom:20s} H {q['H']:.3f}  N/anc {q['H']/3:.3f}  M3 {q['M3']:.5f}  taux3 {q['taux3']*100:.2f}  taux4 {q['taux4']*100:.2f}")
p(f"  pivot y_P* {85 - (75-7.5)*(85-7.5)/75:.3f} ; Hilti bras {68.9*75/(3*24.68):.3f} -> pivot {85-68.9*75/(3*24.68):.3f}"
  f" ; G85/C85 {24.68/(68.9/42.76*14.253333)-1:.4%}")
p(f"  saut au nœud : R(X-e/2) {R*67.5/1000:.5f} vs H(Y-e/2) {H*77.5/1000:.5f}  rapport {H*77.5/(R*67.5):.4f}")

p("== §2.2 jeux")
for nom, kw in [("contrôle", {}), ("Xj=20", dict(Xj=20)), ("Yj=20", dict(Yj=20)), ("Xj=Yj=20", dict(Xj=20, Yj=20)),
                ("C76=2", dict(jeu=20)), ("Yj=-20", dict(Yj=-20)), ("Xj=5", dict(Xj=5)), ("Xj=10", dict(Xj=10)),
                ("Yj=1", dict(Yj=1)), ("Yj=1e6", dict(Yj=1e6)), ("C76=20", dict(jeu=200)), ("L=3000", dict(L=3000))]:
    q = classeur(**{**BASE, **kw})
    p(f"  {nom:9s} M1 {q['M'][0]:.5f} M3 {q['M'][2]:.5f} taux1 {q['taux'][0]*100:.2f} taux3 {q['taux'][2]*100:.2f}"
      f" max {max(q['taux'])*100:.2f} N/anc {q['Nanc']:.3f}")


def bissection(f, a, b, n=200):
    fa = f(a)
    for _ in range(n):
        m = (a + b) / 2
        if (f(m) > 0) == (fa > 0):
            a, fa = m, f(m)
        else:
            b = m
    return (a + b) / 2


p(f"  Xj à 100 % : {bissection(lambda x: max(classeur(**{**BASE, 'Xj': x})['taux']) - 1, 0, 50):.4f} mm")
p(f"  Yj à 100 % : {bissection(lambda x: max(classeur(**{**BASE, 'Yj': x})['taux']) - 1, 0, 500):.4f} mm")
p(f"  pente Xj (pt/mm) {(classeur(**{**BASE,'Xj':20})['taux'][2]-c0['taux'][2])*100/20:.3f}"
  f" ; pente Yj secante 20 mm {(classeur(**{**BASE,'Yj':20})['taux'][2]-c0['taux'][2])*100/20:.3f}"
  f" ; 1er mm {(classeur(**{**BASE,'Yj':1})['taux'][2]-c0['taux'][2])*100:.3f}")

p("== domaine")
for X in (120, 125, 130, 135, 140, 145, 150):
    q = classeur(**{**BASE, 'X': X})
    p(f"  X={X}: taux {[round(t*100,2) for t in q['taux']]}")
q = classeur(**{**BASE, 'Y': 25})
p(f"  Y=25 : M3 {q['M'][2]:.4f}")

p("== seuil Y sous lequel le classeur n'est plus sûr en section 3")
for nom, yP, uV in [("P0", 0, 7.5), ("P1", 7.5, 7.5), ("Q1", 7.5, 0)]:
    f = lambda Y: classeur(**{**BASE, 'Y': Y})['taux'][2] - equilibre(yP, uV, **{**BASE, 'Y': Y})['taux3']
    p(f"  {nom}: Y* = {bissection(f, 29, 84):.3f} mm ; vérif à Y=85 : classeur-équilibre = {f(85)*100:.2f} pt")
# sans cisaillement, P1 : formule 1,25e+1,5r
p(f"  1,25e+1,5r = {1.25*15+1.5*13}")

p("== §3.3 ancrages")
p(f"  C88 {14.253333*1e3/(16*15):.3f}  C90 {14.253333*1e3/(2*35*15):.4f}  demi-trou {14.253333*1e3/(2*(35-9)*15):.3f}"
  f" (+{(35/26-1)*100:.1f} %)  ELU/ELS {68.9/42.76:.5f}  N_ELU {68.9/42.76*14.253333:.4f}  G85/C85 {24.68/(68.9/42.76*14.253333):.4f}")

p("== §3.4 EC3")
sig3, M3 = c0['taux'][2] * SADM, c0['M'][2]
Mpl = 300 * 15 ** 2 / 4 * fy / 1e6
for g in (1.35, 1.5, 68.9 / 42.76):
    p(f"  gamma {g:.4f}: admissible {c0['taux'][2]*100:.2f}  EC3 élast {g*sig3/fy*100:.2f}  EC3 plast {g*M3/Mpl*100:.2f}"
      f"  rapport adm/plast {c0['taux'][2]/(g*M3/Mpl):.3f}")
p(f"  V/Vpl ELU {68.9e3/(4500*fy/math.sqrt(3)):.4f}")

p("== module du site (M = V·e, b = 1 m, e = X)")
V = 42.76 / 0.3
for e_cm in (7.5, 4.7):
    s = V * e_cm / 100 * 1e6 / (1000 * 225 / 6)
    ta = V * 1e3 / 15000
    seq = math.sqrt(s ** 2 + 3 * ta ** 2)
    p(f"  e = {e_cm} cm : σeq {seq:.3f} -> /236,67 {seq/SADM*100:.2f} %  ; /155 (défaut du site) {seq/155*100:.2f} %"
      f" ; τ/0,8 {ta/0.8*100:.1f} % ; ELU 1,35 /186 {math.sqrt((1.35*s)**2+3*(1.35*ta)**2)/186*100:.1f} %")

p("== §5.3 largeur d'appui (diffusion 45°)")
p(f"  b_min milieu {300-2*47} ; bout {300-47} ; b=100 -> facteur {300/(100+2*47):.4f}")
for Leff in (194.0, 250.0):
    q = classeur(**{**BASE, 'L': Leff})
    p(f"  L = {Leff}: taux {[round(t*100,2) for t in q['taux']]}")

p("== §5.5 ELS/ELU")
k = 68.9 / 42.76
for psi in (0.5, 0.3):
    p(f"  psi {psi}: G/Q = {(1.5 - k*psi)/(k-1.35):.3f}")
p(f"  PROFIS V·X = {68.9*0.075:.4f} kN·m")
# volée 12
G12, Q12 = 57.93 / 2 / 1.345, 24.56 / 2 / 1.345
p(f"  volée 12 : G {G12:.4f} Q {Q12:.4f} 1,35G+1,5Q {1.35*G12+1.5*Q12:.4f} écart/C12 {(1.35*G12+1.5*Q12)/42.76-1:.4%}")
kq = (1.35 * G12 + 1.5 * Q12) / (G12 + Q12)
q0 = equilibre(0, 7.5)
p(f"  rapport volée 12 {kq:.5f} ; P0 N_ELU par ancrage {q0['H']/3*kq:.3f} ; V_ELU {42.76/3*kq:.3f} ; avec 1,6113 : {q0['H']/3*k:.3f}")

p("== Annexe C — E10A BD")
masses = {1: 1184.5, 2: 2773.33, 4: 1109.67, 6: 2747.87, 7: 2284.97, 9: 2777.01, 10: 3628.21, 12: 5793}
surf = {1: 1.79, 2: 4.14, 4: 1.78, 6: 4.14, 7: 3.44, 9: 4.14, 10: 4.84, 12: 6.14}
for v in masses:
    G = masses[v] / 100 / 2 / 1.345
    Q = surf[v] * 4 / 2 / 1.345
    p(f"  volée {v:2d}: G {G:.3f} Q {Q:.3f} G+Q {G+Q:.3f} ELU {1.35*G+1.5*Q:.3f} rapport {(1.35*G+1.5*Q)/(G+Q):.4f}")
p(f"  (poids+charge)/2 volée 12 = {(57.93+24.56)/2:.3f} ({(57.93+24.56)/2/42.76-1:.3%}) ; volées 10+12 = {(36.2821+19.36+57.93+24.56)/2:.3f}"
  f" ({(36.2821+19.36+57.93+24.56)/2/68.9-1:.3%}) ; 42,76/0,3 = {42.76/0.3:.2f} ; rapport {42.76/0.3/(G12+Q12):.2f}")

p("== §6.1 configuration B")
for ya in (35, 75, 25, 45):
    p(f"  y_a = {ya}: H = {42.76*67.5/ya:.3f}")
M3B = 42.76 * 67.5 / 1000
sB = M3B * 1e6 / 11250 + 42.76e3 / 4500
p(f"  M3B {M3B:.4f} σ {sB:.3f} taux {sB/SADM*100:.2f} ; classeur {c0['M'][2]:.4f} ({c0['M'][2]/M3B-1:.2%})"
  f" ; B/A(P0) {42.76*67.5/35/q0['H']:.3f}")
Ln = 300 - 3 * 18
sn = M3B * 1e6 / (Ln * 225 / 6) + 42.76e3 / (Ln * 15)
p(f"  section nette {sn:.3f} -> {sn/SADM*100:.2f} % ; pression {42.76*67.5/35/0.3:.1f} N/mm")
sX = 42.76 * (95 - 7.5) / 1000 * 1e6 / 11250 + 42.76e3 / 4500
p(f"  Xj = 20 en B : {sX/SADM*100:.2f} %")
# au nœud des lignes moyennes en B : continuité
p("  (B) moment aile horizontale au nœud = R(X - e/2) = moment aile verticale : continuité exacte")

p("== §6.2 ailes inégales L150×100×12")
for nom, X, Y, r_ in [("150h×100v r12", 75, 65, 12), ("100h×150v r12", 75, 115, 12), ("150h×100v r6", 75, 65, 6)]:
    kw = dict(R=42.76, X=X, Xj=0, Y=Y, Yj=0, e=12.0, r=r_, L=300, n=3, jeu=10)
    q = classeur(**kw)
    qe = equilibre(0, 6.0, **kw)
    p(f"  {nom}: taux1 {q['taux'][0]*100:.2f} taux3 {q['taux'][2]*100:.2f} traction tot {q['Nanc']*3:.2f}"
      f" | P0 taux3 {qe['taux3']*100:.2f} H {qe['H']:.2f}")

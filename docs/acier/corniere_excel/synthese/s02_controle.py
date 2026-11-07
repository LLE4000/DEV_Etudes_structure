# -*- coding: utf-8 -*-
"""S02 — Le cas de contrôle : sections, ancrages, jeux, pivots, aile verticale, sensibilités, EC3, moteur du site.

Toutes les grandeurs sont calculées ici (méthode en symboles de s_commun) ; les cas « classeur » sont recoupés par
l'interpréteur des formules (s_commun.Classeur) pour les saisies que le classeur n'a jamais calculées.

Lancer : cd /tmp && /usr/bin/python3.12 -I -B <ce dossier>/s02_controle.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s_commun as K  # noqa: E402

S = K.Sortie("s02_controle")
cl = K.Classeur()
C = K.CAS
e, r, L, R, X, Y = C["e"], C["r"], C["L"], C["R"], C["X"], C["Y"]
h = K.historique(C)
sadm = h["sadm"]

S("# S02 — cas de contrôle L120x120x15, S355, L = 300 mm, R = 42,76 kN, X = 75, Y = 85, Xj = Yj = 0, γ = 1,50")
S("")
S("## 1. Les quatre sections (méthode du classeur)")
S("| section | t (mm) | W (mm³) | bras (mm) | M (kN·m) | σ | τ | σ_éq (N/mm²) | taux |")
bras = [h["Xe"], h["Xer"], h["Ye"], h["Yer"]]
for i in range(4):
    S(f"| {i + 1} | {h['t'][i]:.0f} | {h['W'][i]:.0f} | {bras[i]:.1f} | {h['M'][i] / 1e6:.5f} | {h['sig'][i]:.3f} | "
      f"{h['tau'][i]:.3f} | {h['seq'][i]:.3f} | {100 * h['taux'][i]:.2f} % |")
S(f"σ_adm = f_y/γ = {sadm:.3f} ; τ_adm = σ_adm/√3 = {h['tadm']:.3f} N/mm² ; Mmax = R·(X + Xj) = {h['Mmax'] / 1e6:.4f} kN·m")
S(f"taux max = {100 * h['taux_max']:.4f} % en section {h['section_max']}")
S(f"σ_éq/σ − 1 : " + " ; ".join(f"S{i + 1} {100 * (h['seq'][i] / h['sig'][i] - 1):+.2f} %" for i in range(4)))
S.v("hist", {k: h[k] for k in ("M", "t", "W", "sig", "tau", "seq", "taux", "taux_max", "Mmax", "H", "rho3", "rho4")})

S("")
S("## 2. Ancrages (classeur)")
S(f"N_ELS = Mmax/((Y − jeu)·n) = {h['Mmax'] / 1e6:.4f}/(({Y:.0f} − {C['jeu']:.0f})·{C['n']}) = {h['N_els'] / 1e3:.4f} kN (C82)")
S(f"V_ELS = R/n = {h['V_els'] / 1e3:.4f} kN (C83)")
S(f"R_ELU/R_ELS = {C['R_ELU'] / 1e3:.1f}/{R / 1e3:.2f} = {h['rapport_elu']:.5f} ; N_ELU = {h['N_elu'] / 1e3:.4f} ; V_ELU = {h['V_elu'] / 1e3:.4f} kN")
S(f"G85 = 24,68 kN (tapé) / C85 = {24.68 / (h['N_elu'] / 1e3):.4f} ; G86 = 22,97 / C86 = {22.97 / (h['V_elu'] / 1e3):.5f}")
S(f"C88 pression diamétrale V_ELS/(d·e) = {h['p_diam']:.3f} N/mm² (×ρ_ELU {h['p_diam'] * h['rapport_elu']:.2f}) ; "
  f"C90 arrachement V_ELS/(2·(haut. − Y)·e) = {h['arrach']:.3f} N/mm² (×ρ_ELU {h['arrach'] * h['rapport_elu']:.2f})")
d0 = 18.0
arr_net = h["V_els"] / (2 * (C["haut"] - Y - d0 / 2) * e)
S(f"arrachement en déduisant un demi-trou Ø{d0:.0f} (hypothèse) : {arr_net:.2f} N/mm² ({100 * (arr_net / h['arrach'] - 1):+.1f} %)")
S.v("anc", dict(N_els=h["N_els"], V_els=h["V_els"], N_elu=h["N_elu"], V_elu=h["V_elu"], rho=h["rapport_elu"],
                p_diam=h["p_diam"], arrach=h["arrach"]))

S("")
S("## 3. Pourquoi les quatre moments diffèrent")
H = h["H"]
S(f"H = Mmax/YH = {H / 1e3:.3f} kN ; H/R = X/YH = {H / R:.5f}")
S(f"M2 − M1 = R·r = {(h['M'][1] - h['M'][0]) / 1e6:.5f} kN·m ; M4 − M3 = H·r = {(h['M'][3] - h['M'][2]) / 1e6:.5f} kN·m")
S(f"M3/M1 = (H/R)·(Ye/Xe) = {H / R:.5f} × {h['Ye'] / h['Xe']:.5f} = {h['M'][2] / h['M'][0]:.5f}")
S(f"W(e+r)/W(e) = ((e+r)/e)² = {(h['W'][1] / h['W'][0]):.4f} ; σ2/σ1 = {h['sig'][1] / h['sig'][0]:.4f} ; σ4/σ3 = {h['sig'][3] / h['sig'][2]:.4f}")
ordre = sorted([(h["M"][i], f"M{i + 1}") for i in range(4)] + [(h["Mmax"], "Mmax")])
S("ordre : " + " < ".join(f"{n} = {m / 1e6:.5f}" for m, n in ordre))
S(f"au nœud filaire (u = y = e/2) : aile horizontale R·(X − e/2) = {R * (X - e / 2) / 1e6:.5f} ; aile verticale du classeur "
  f"H·(Y − e/2) = {H * (Y - e / 2) / 1e6:.5f} kN·m ; saut R·e/2 = {R * e / 2 / 1e6:.5f} kN·m ({100 * (X / (X - e / 2) - 1):.1f} %)")

S("")
S("## 4. Les jeux (méthode du classeur ; recoupée par les formules réévaluées)")


def cas(**kw):
    p = dict(C, **kw)
    hh = K.historique(p)
    ev = cl.evaluer(K.saisies_classeur(p, "L120x120x15"))
    assert abs(ev["D3"] - hh["taux_max"]) < 1e-12 and abs(ev["C82"] - hh["N_els"] / 1e3) < 1e-9, kw
    return hh


S("| saisie | M1 | M3 | M4 (kN·m) | H (kN) | taux 1 | taux 3 | taux max | N_ELS | V_ELS (kN) |")
jeux = {}
for nom, kw in (("cas de contrôle", {}), ("Xj = 20", dict(Xj=20.0)), ("Yj = 20", dict(Yj=20.0)),
                ("Xj = Yj = 20", dict(Xj=20.0, Yj=20.0)), ("jeu C76 = 2 cm", dict(jeu=20.0)),
                ("Yj = −20", dict(Yj=-20.0)), ("Xj = −20", dict(Xj=-20.0)),
                ("C76 = 20 (« 20 mm » tapé dans une case en cm)", dict(jeu=200.0)),
                ("C8 = 300 (« 300 mm » tapé dans une case en cm)", dict(L=3000.0))):
    hh = cas(**kw)
    jeux[nom] = dict(M=hh["M"], H=hh["H"], taux=hh["taux"], taux_max=hh["taux_max"], N=hh["N_els"], V=hh["V_els"])
    S(f"| {nom} | {hh['M'][0] / 1e6:.5f} | {hh['M'][2] / 1e6:.5f} | {hh['M'][3] / 1e6:.5f} | {hh['H'] / 1e3:.3f} | "
      f"{100 * hh['taux'][0]:.2f} % | {100 * hh['taux'][2]:.2f} % | {100 * hh['taux_max']:.2f} % | {hh['N_els'] / 1e3:.3f} | {hh['V_els'] / 1e3:.3f} |")
S.v("jeux", jeux)


def seuil(cle, cible=1.0):
    a, b = 0.0, 500.0
    for _ in range(200):
        m = (a + b) / 2
        if K.historique(dict(C, **{cle: m}))["taux_max"] < cible:
            a = m
        else:
            b = m
    return (a + b) / 2


xs, ys = seuil("Xj"), seuil("Yj")
d_x = 100 * (K.historique(dict(C, Xj=1.0))["taux_max"] - h["taux_max"])
d_y = 100 * (K.historique(dict(C, Yj=1.0))["taux_max"] - h["taux_max"])
lim_y = K.historique(dict(C, Yj=1e9))["taux"][2]
S(f"taux max = 100 % pour Xj = {xs:.3f} mm ou pour Yj = {ys:.2f} mm ; +1 mm de Xj : {d_x:+.2f} point ; +1 mm de Yj : {d_y:+.2f} point")
S(f"Yj → ∞ : taux 3 → {100 * lim_y:.2f} % (Mmax entier sur l'épaisseur e)")
S.v("seuils", dict(Xj=xs, Yj=ys, dx=d_x, dy=d_y, lim_y=lim_y))
for xj in (5.0, 10.0, 20.0):
    S(f"Xj = {xj:.0f} mm : taux max {100 * K.historique(dict(C, Xj=xj))['taux_max']:.2f} %")

S("")
S("## 5. Pivots et corps libre (aile verticale)")
S("Statique cohérente d'une section de l'aile verticale : σ = M/W + N/A (N = R, traction), τ = V/A (V = H).")
S("| variante | y_P (mm) | u_V (mm) | H (kN) | N/ancrage (kN) | M3 (kN·m) | M4 (kN·m) | taux 3 cohérent | taux 3 « convention classeur » |")
piv = {}


def ligne(nom, yP, uV, M_classeur=False):
    fb = K.corps_libre_A(C, yP, uV)
    if M_classeur:                       # la convention du classeur : H avec uV = 0, M avec uV = e/2
        M3, M4 = fb["H"] * (Y - (e + r)), fb["H"] * (Y - e)
    else:
        M3, M4 = fb["M"](e + r), fb["M"](e)
    t_coh, s_coh, u_coh = K.taux_section_v(C, M3, R, fb["H"])
    t_cl = math.sqrt((M3 / h["W"][2]) ** 2 + 3 * h["tau"][2] ** 2) / sadm
    piv[nom] = dict(yP=yP, uV=uV, H=fb["H"], N=fb["H"] / C["n"], M3=M3, M4=M4, taux_coh=t_coh, taux_cl=t_cl, sig=s_coh, tau=u_coh)
    S(f"| {nom} | {yP:.2f} | {uV:.1f} | {fb['H'] / 1e3:.3f} | {fb['H'] / C['n'] / 1e3:.3f} | {M3 / 1e6:.5f} | {M4 / 1e6:.5f} | "
      f"{100 * t_coh:.2f} % | {100 * t_cl:.2f} % |")


ligne("classeur, sections (H avec u_V = 0, M avec u_V = e/2)", e / 2, 0.0, M_classeur=True)
ligne("classeur, ancrages (C76 : pivot à 1 cm)", 10.0, 0.0, M_classeur=True)
yh = Y - 3 * 68900 * 0 + 0                         # placeholder remplacé ci-dessous
z_hilti = 68900 * X / (3 * 24680)                  # bras qui redonne G85 = 24,68 kN à l'ELU (u_V = 0)
ligne("Hilti (G85 = 24,68 kN ⇒ pivot)", Y - z_hilti, 0.0, M_classeur=True)
ligne("P0 : pivot à l'arête, V sur l'axe (maillage, mur raide)", 0.0, e / 2)
ligne("P1 : pivot à e/2, V sur l'axe (nœud filaire)", e / 2, e / 2)
ligne("Q1 : pivot à e/2, V au nu du mur", e / 2, 0.0)
ligne("P10 : pivot à 1 cm, V sur l'axe", 10.0, e / 2)
ligne("P15 : pivot à e (mur souple), V sur l'axe", 15.0, e / 2)
yPs = Y - (X - e / 2) * (Y - e / 2) / X
S(f"bras implicite de Hilti z = R_ELU·X/(n·G85) = {z_hilti:.2f} mm ⇒ pivot à {Y - z_hilti:.2f} mm au-dessus du dessous ; "
  f"N_ELS correspondant = {R * X / z_hilti / 3 / 1e3:.3f} kN ; G85/C85 − 1 = {100 * (24.68 / (h['N_elu'] / 1e3) - 1):.2f} %")
S(f"pivot qui redonne exactement M3 et M4 du classeur avec V sur l'axe : y_P* = Y − (X − e/2)(Y − e/2)/X = {yPs:.3f} mm")
S.v("pivots", piv)
S.v("z_hilti", z_hilti)
S.v("yP_star", yPs)
# double compte
t_dc = math.sqrt((h["sig"][2] + R / (L * e)) ** 2 + 3 * (H / (L * e)) ** 2) / sadm
S(f"« correction » partielle à NE PAS faire (moment du classeur + N/A) : taux 3 = {100 * t_dc:.2f} % (double compte)")
S(f"σ_N = R/(L·e) = {R / (L * e):.3f} N/mm² (le « τ » du classeur en section 3) ; τ_H = H/(L·e) = {H / (L * e):.3f} N/mm²")
S.v("double_compte", t_dc)


# seuil en Y : en deçà, la statique cohérente dépasse le classeur
def ecart_Y(yv, yP, uV):
    p = dict(C, Y=yv)
    hh = K.historique(p)
    fb = K.corps_libre_A(p, yP, uV)
    t_coh = K.taux_section_v(p, fb["M"](e + r), R, fb["H"])[0]
    return t_coh / hh["taux"][2] - 1


for nom, (yP, uV) in (("Q1", (e / 2, 0.0)), ("P1", (e / 2, e / 2)), ("P0", (0.0, e / 2))):
    a, b = e + r + 0.5, 120.0
    if ecart_Y(a, yP, uV) < 0:
        S(f"{nom} : la statique cohérente reste sous le classeur pour tout Y de {a:.1f} à 120 mm")
        continue
    for _ in range(100):
        m = (a + b) / 2
        if ecart_Y(m, yP, uV) > 0:
            a = m
        else:
            b = m
    S(f"{nom} : la statique cohérente dépasse le classeur si Y < {a:.1f} mm (Y = 30 mm : {100 * ecart_Y(30.0, yP, uV):+.1f} %)")
    S.v(f"seuil_Y_{nom}", a)


# même seuil, en flexion + effort normal seulement (sans τ) : c'est la comparaison de l'analyse critique
def ecart_sig(yv):
    p = dict(C, Y=yv)
    hh = K.historique(p)
    fb = K.corps_libre_A(p, e / 2, e / 2)
    return (fb["M"](e + r) / hh["W"][2] + R / (L * e)) / hh["sig"][2] - 1


a_, b_ = e + r + 0.5, 120.0
for _ in range(100):
    m_ = (a_ + b_) / 2
    if ecart_sig(m_) > 0:
        a_ = m_
    else:
        b_ = m_
S(f"P1, flexion + effort normal sans τ : seuil Y < {a_:.2f} mm (formule fermée 1,25·e + 1,5·r = {1.25 * e + 1.5 * r:.2f} mm)")
S.v("seuil_Y_P1_sans_tau", a_)

S("")
S("## 6. Sensibilités")
for rr in (13.0, 6.5, 0.0):
    S(f"r = {rr:4.1f} mm : taux 3 = {100 * K.historique(dict(C, r=rr))['taux'][2]:.2f} % ; taux max {100 * K.historique(dict(C, r=rr))['taux_max']:.2f} %")
    S.v(f"taux_r_{rr}", K.historique(dict(C, r=rr))["taux_max"])
for LL in (250.0, 300.0, 350.0):
    S(f"L = {LL:.0f} mm : taux max = {100 * K.historique(dict(C, L=LL))['taux_max']:.2f} %")
    S.v(f"taux_L_{LL:.0f}", K.historique(dict(C, L=LL))["taux_max"])

S("Largeur d'appui c de la charge, diffusion à 45° jusqu'à la section 1 : b_eff = min(L, c + 2·Xe) (charge loin des bouts)")
for c in (50.0, 100.0, 150.0, 206.0, 300.0):
    beff = min(L, c + 2 * h["Xe"])
    S(f"   c = {c:.0f} mm : b_eff = {beff:.0f} mm ; taux 1 = {100 * h['taux'][0] * L / beff:.1f} % (σ1 × {L / beff:.2f})")
S(f"   largeur d'appui qui rend toute la longueur reprise efficace : {L - 2 * h['Xe']:.0f} mm (charge au milieu), "
  f"{L - h['Xe']:.0f} mm (charge en bout de cornière)")

S("")
S("## 7. Domaine : ce que le classeur rend sans alerte")
for nom, kw in (("X = 20 mm (charge sur le congé)", dict(X=20.0)), ("X + Xj = 125 > larg. = 120", dict(Xj=50.0)),
                ("X = 140 > larg.", dict(X=140.0)), ("Y = 25 mm (ancrage dans le congé)", dict(Y=25.0)),
                ("Y = 130 > haut.", dict(Y=130.0))):
    hh = cas(**kw)
    S(f"{nom} : M1 = {hh['M'][0] / 1e6:.3f}, M3 = {hh['M'][2] / 1e6:.3f} kN·m, ρ3 = {hh['rho3']:.3f}, taux max "
      f"{100 * hh['taux_max']:.1f} %, N_ELS = {hh['N_els'] / 1e3:.2f} kN, C90 = {hh['arrach']:.2f}")
p = dict(C, Y=C["jeu"])
ev = cl.evaluer(K.saisies_classeur(p, "L120x120x15"))
S(f"Y = jeu = 10 mm : C82 = {ev['C82']} (division par zéro dans le classeur)")

S("")
S("## 8. Situer la méthode : EC3 (section 3, statique du classeur, γ_M0 = 1,0)")
Mpl = L * e * e / 4 * C["fy"]
for g in (1.35, 1.50, C["R_ELU"] / R):
    t_el = g * h["seq"][2] / C["fy"]
    t_pl = g * h["M"][2] / Mpl
    S(f"γ = {g:.4f} : admissible {100 * h['taux'][2]:.2f} % ; EC3 élastique {100 * t_el:.2f} % ; EC3 plastique {100 * t_pl:.2f} %")
    S.v(f"ec3_{g:.4f}", dict(el=t_el, pl=t_pl))
S(f"rapport admissible / EC3 plastique : de {h['taux'][2] / (1.5 * h['M'][2] / Mpl):.3f} (γ = 1,50) à {h['taux'][2] / (1.35 * h['M'][2] / Mpl):.3f} (γ = 1,35)")
Vpl = L * e * C["fy"] / math.sqrt(3)
S(f"V_Ed/V_pl à l'ELU (R_ELU) = {C['R_ELU'] / Vpl:.3f}")

S("")
S("## 9. Le moteur actuel du site sur le cas de contrôle (lecture seule)")
sys.dont_write_bytecode = True
sys.path.insert(0, "/home/user/integ")
from acier.corniere import moteur as M  # noqa: E402

base = dict(M.nouvelle_corniere(), standard=True, profil="L 120x120x15", charge="ponctuelle", V=R / 1e3, L=L / 1000,
            gamma_ELU=1.0)
for nom, ee in (("e = X = 7,5 cm (moment au nu du mur)", X / 10), ("e = Xe = 4,7 cm", h["Xe"] / 10),
                ("e = X·Ye/YH = 5,516 cm", X * h["Ye"] / h["YH"] / 10)):
    res = M.verifier(dict(base, e=ee, sig_lim_ELS=sadm, tau_lim_ELS=h["tadm"], sig_lim_ELU=sadm))
    g = res["grandeurs"]
    S(f"{nom} : σ_ELS = {g['sigma_ELS']:.2f} N/mm², σ_éq = {g['sigma_eq_ELS']:.2f}, U_σ = {100 * g['U_sigma']:.2f} % "
      f"(classeur {100 * h['taux_max']:.2f} % ; écart {100 * (g['U_sigma'] / h['taux_max'] - 1):+.1f} %)")
    S.v(f"site_{ee:.3f}", g["U_sigma"])
S("   (les trois lignes ci-dessus SAISISSENT σ_lim,ELS = τ_lim·√3 = σ_eq,lim,ELU = f_y/1,5 et γ_ELU = 1,0 ; ce ne sont pas les défauts du site)")
from acier.corniere import entree as EN  # noqa: E402
S("défauts de l'écran du site (acier/corniere/entree.py, CHAMPS) : " + ", ".join(f"{k} = {EN.CHAMPS[k][0]:g}" for k in ("gamma_ELU", "sig_lim_ELS", "tau_lim_ELS", "sig_lim_ELU")))
res = M.verifier(dict(base, e=X / 10, gamma_ELU=EN.CHAMPS["gamma_ELU"][0]))
g = res["grandeurs"]
S(f"TOUTES valeurs par défaut du site (γ_ELU 1,35 ; σ_lim,ELS 155 ; τ_lim,ELS 0,8 ; σ_eq,lim,ELU 186 N/mm²), e = 7,5 cm : "
  f"U_σ = {100 * g['U_sigma']:.2f} %, U_τ = {100 * g['U_tau']:.1f} %, U_ELU = {100 * g['U_ELU']:.1f} %")
worst = 0.0
for s_ in [10 + 7.3 * k for k in range(200)]:
    gg = M.verifier(dict(base, e=X / 10, s=s_))["grandeurs"]
    worst = max(worst, abs(gg["U_long_ELU"] / gg["U_ELU"] - 1))
S(f"vérification « longitudinale » sur 200 entraxes de 10 à {10 + 7.3 * 199:.0f} cm : max |U_long/U_ELU − 1| = {worst:.1e}")
S.v("site_long", worst)
S.ecrire()

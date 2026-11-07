# -*- coding: utf-8 -*-
"""Contrôles mécaniques COMPLÉMENTAIRES du classeur (ce qu'il ne calcule pas, pour situer ses hypothèses).
Tout part du modèle de reproduction.py (même dossier, chargé par son chemin) et du cas de contrôle.

  1. équilibre du modèle et position du « pivot » : e/2 pour les sections 3-4, jeu de 1 cm pour les ancrages,
     et celui qu'impliquent les 24,68 kN de Hilti ; effet d'un pivot unique sur M3, M4 et le taux ;
  2. efforts intérieurs de l'aile verticale dans le modèle du classeur : effort normal R (traction), effort
     tranchant H ; ce que donnerait σ = σ_M + σ_N avec τ = H/(L·e) ;
  3. le maximum de σ le long des congés (épaisseur variable e + r − √(r² − u²)) face aux sections 1 et 3 ;
  4. part de τ dans σ_éq ; sensibilité à L ;
  5. pour situer : ELU (68,9 kN) à f_y/1,0 en élastique et en plastique ; coefficient global implicite ;
  6. ancrages : bras, équilibre, écart à Hilti ; pression diamétrale et arrachement (avec et sans demi-trou) ;
  7. domaine : ce que le classeur calcule hors géométrie (X < e + r, Y < e + r, X + Xj > largeur) ;
  8. E10A BD : peut-on retrouver Rmax = 42,76 kN et 68,9 kN ? ;
  9. le moteur actuel du site (modèle « lame », acier/corniere/moteur.py) sur le même cas — importé en LECTURE
     seule, sans écrire de bytecode.

Lancement : cd /tmp && /usr/bin/python3.12 -I -B <ce dossier>/complements.py   → <ce dossier>/sorties/complements.txt
"""
import importlib.util
import itertools
import math
import os
import sys

sys.dont_write_bytecode = True
ICI = os.path.dirname(os.path.abspath(__file__))
SORTIE = os.path.join(ICI, "sorties", "complements.txt")

_spec = importlib.util.spec_from_file_location("reproduction_corniere", os.path.join(ICI, "reproduction.py"))
REP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(REP)
CAS, modele = REP.CAS, REP.modele


def main():
    L = []
    p = L.append
    c = CAS
    m = modele(c)
    R, X, Y, e, r, Lr = c["R"], c["X"], c["Y"], c["e"], c["r"], c["L"]
    sadm = c["fy"] / c["gM"]
    Mmax = R * X                                         # N·mm
    p("CONTRÔLES MÉCANIQUES COMPLÉMENTAIRES — produit par complements.py (cas de contrôle L120×120×15)")

    # ------------------------------------------------------------------ 1. équilibre et pivot
    p("")
    p("== 1. Équilibre du modèle et position du pivot ==")
    YH = Y - e / 2
    H = Mmax / YH
    p(f"modèle des sections 3-4 : pivot au nu du mur, au plan moyen de l'aile horizontale (e/2 = {e / 2} mm du dessous)")
    p(f"   ΣM/pivot : R·X − H·YH = {R * X:.1f} − {H:.3f}·{YH} = {R * X - H * YH:.2e} N·mm ; H = {H / 1e3:.4f} kN au total, "
      f"{H / 1e3 / c['n']:.4f} kN par ancrage")
    z_anc = Y - c["jeu_anc"]
    N_tot = Mmax / z_anc
    p(f"modèle des ancrages (P8!C82) : bras Y − jeu = {Y} − {c['jeu_anc']} = {z_anc} mm → traction totale "
      f"{N_tot / 1e3:.4f} kN, {N_tot / 1e3 / c['n']:.4f} kN par ancrage (pivot à {c['jeu_anc']} mm du dessous)")
    p(f"   rapport des deux tractions : {N_tot / H:.5f} (= YH/(Y − jeu) = {YH / z_anc:.5f})")
    N_hilti = 24.68e3                                     # P8!G85, valeur tapée
    k = c["R_ELU"] / R
    z_hilti = (c["R_ELU"] * X) / (c["n"] * N_hilti)
    p(f"Hilti (P8!G85 = 24,68 kN/ancrage, ELU) : si l'excentrement est X = {X} mm et la traction également partagée, "
      f"bras = R_ELU·X/(n·N) = {z_hilti:.3f} mm → pivot à {Y - z_hilti:.3f} mm du dessous")
    p(f"   écart classeur / Hilti sur la traction ELU : {m['N_ELU']:.4f} / 24,68 − 1 = {m['N_ELU'] / 24.68 - 1:+.3%} ; "
      f"cisaillement : {m['V_ELU']:.4f} / 22,97 − 1 = {m['V_ELU'] / 22.97 - 1:+.4%}")
    p("effet d'un pivot unique à la hauteur c (au-dessus du dessous) sur l'aile verticale :")
    p(f"   {'c (mm)':>8s} {'H (kN)':>9s} {'M3 (kN·m)':>10s} {'M4 (kN·m)':>10s} {'taux 3':>9s} {'taux 4':>9s} {'N/ancrage ELS (kN)':>19s}")
    for cc, lib in ((0.0, "arête inférieure"), (e / 2, "e/2 (classeur, sections)"), (c["jeu_anc"], "jeu 1 cm (classeur, ancrages)"),
                    (Y - z_hilti, "implicite Hilti")):
        Hc = Mmax / (Y - cc)
        M3 = Hc * (Y - e - r)
        M4 = Hc * (Y - e) if cc <= e else float("nan")
        s3 = M3 / (Lr * e ** 2 / 6)
        t3 = math.sqrt(s3 ** 2 + 3 * (R / (Lr * e)) ** 2) / sadm
        s4 = M4 / (Lr * (e + r) ** 2 / 6)
        t4 = math.sqrt(s4 ** 2 + 3 * (R / (Lr * (e + r))) ** 2) / sadm
        p(f"   {cc:8.3f} {Hc / 1e3:9.4f} {M3 / 1e6:10.5f} {M4 / 1e6:10.5f} {t3:9.4%} {t4:9.4%} {Hc / 1e3 / c['n']:19.4f}   {lib}")
    p("   (M4 n'a de sens que si le pivot est sous la section 4, c ≤ e)")

    # ------------------------------------------------------------------ 2. efforts intérieurs de l'aile verticale
    p("")
    p("== 2. Aile verticale : efforts intérieurs du modèle du classeur (entre l'ancrage et la section 3) ==")
    sN = R / (Lr * e)
    tH = H / (Lr * e)
    sM = m["sigma3"]
    p(f"effort normal (traction) = R = {R / 1e3} kN → σ_N = R/(L·e) = {sN:.4f} N/mm² (le classeur l'appelle τ, P8!G54)")
    p(f"effort tranchant = H = {H / 1e3:.4f} kN → τ_H = H/(L·e) = {tH:.4f} N/mm²")
    q_cl = m["seq3"]
    q_c = math.sqrt((sM + sN) ** 2 + 3 * tH ** 2)
    p(f"classeur : √(σ_M² + 3·(R/Le)²) = {q_cl:.4f} N/mm² → {q_cl / sadm:.4%}")
    p(f"σ_M + σ_N et τ_H : √(({sM:.4f} + {sN:.4f})² + 3·{tH:.4f}²) = {q_c:.4f} N/mm² → {q_c / sadm:.4%} "
      f"(+{(q_c / q_cl - 1):.3%})")
    for cc in (c["jeu_anc"],):
        Hc = Mmax / (Y - cc)
        sMc = Hc * (Y - e - r) / (Lr * e ** 2 / 6)
        qq = math.sqrt((sMc + sN) ** 2 + 3 * (Hc / (Lr * e)) ** 2)
        p(f"avec en plus le pivot des ancrages (c = {cc} mm) : σ_éq = {qq:.4f} N/mm² → {qq / sadm:.4%}")

    # ------------------------------------------------------------------ 3. le maximum dans les congés
    p("")
    p("== 3. Le maximum de σ le long des congés (théorie des poutres, épaisseur variable) ==")
    def ep(u):                           # épaisseur à la distance u du dos (u entre e et e + r)
        return e + r - math.sqrt(max(r * r - (u - e - r) ** 2, 0.0))
    for nom, Mfun, tf, s_ref, q_ref, lib in (
            ("aile horizontale", lambda u: R * (X - u), lambda u: R / (Lr * ep(u)), m["sigma1"], m["seq1"], "section 1"),
            ("aile verticale", lambda u: H * (Y - u), lambda u: R / (Lr * ep(u)), m["sigma3"], m["seq3"], "section 3")):
        best = (0, None, None)
        N = 200000
        for i in range(N + 1):
            u = e + r * i / N
            s = 6 * Mfun(u) / (Lr * ep(u) ** 2)
            q = math.sqrt(s * s + 3 * tf(u) ** 2)
            if q > best[0]:
                best = (q, u, s)
        q, u, s = best
        p(f"{nom} : σ_éq maximal à {u - e:.3f} mm du nu de l'autre aile, soit {e + r - u:.3f} mm AVANT le pied du congé "
          f"(côté angle), épaisseur {ep(u):.4f} mm (+{ep(u) - e:.4f}), σ = {s:.4f}, σ_éq = {q:.4f} N/mm² → {q / sadm:.4%} "
          f"(classeur {lib} : {q_ref / sadm:.4%}, +{q / q_ref - 1:.3%})")

    # quand la section « e + r » (2 ou 4) gouverne-t-elle la section « e » (1 ou 3) ? σ2 > σ1 ⇔ a·e² > (a − r)(e + r)²
    a_lim = r * (e + r) ** 2 / ((e + r) ** 2 - e ** 2)
    p(f"la section 2 (e + r) dépasse la section 1 (e) seulement si X − e < r·(e+r)²/((e+r)² − e²) = {a_lim:.3f} mm, "
      f"soit X < {e + a_lim:.3f} mm (le pied du congé est à {e + r} mm) ; même seuil pour Y entre les sections 4 et 3")

    # balayage du catalogue du classeur : pic réel dans le congé / σ au pied du congé, selon la distance s charge–pied
    import openpyxl
    ws = openpyxl.load_workbook(os.path.join(ICI, "source", "realex_cornieres.xlsx"), data_only=True)["CORNIERES"]
    vus = {}
    for i in range(2, 59):
        nom = ws.cell(i, 1).value
        vus.setdefault(nom, (ws.cell(i, 2).value, ws.cell(i, 4).value, ws.cell(i, 5).value))
    p("rapport (pic de σ dans le congé) / (σ au pied du congé), cornières du classeur, charge à s du pied du congé :")
    for s_min in (5.0, 10.0, 20.0, 40.0):
        pire = (0.0, None)
        for nom, (a_, t_, r_) in vus.items():
            if r_ <= 0 or a_ - t_ - r_ < s_min:
                continue
            Xs = t_ + r_ + s_min
            def ep2(u):
                return t_ + r_ - math.sqrt(max(r_ * r_ - (u - t_ - r_) ** 2, 0.0))
            pic = max((Xs - (t_ + r_ * i / 4000)) / ep2(t_ + r_ * i / 4000) ** 2 for i in range(4001))
            rap = pic / (s_min / t_ ** 2)
            if rap > pire[0]:
                pire = (rap, nom)
        p(f"   s = {s_min:4.0f} mm : pic/pied au plus {pire[0]:.4f} ({pire[1]})")

    # ------------------------------------------------------------------ 3 bis. diagrammes continus chiffrés
    p("")
    p("== 3 bis. Diagrammes continus du modèle du classeur (mêmes hypothèses, aucune interpolation) ==")
    p("aile horizontale, u = distance au dos (nu du mur) ; M(u) = R·(X + Xj − u) ; épaisseur e + r − √(r² − (u − e − r)²) dans le congé")
    p(f"   {'u (mm)':>8s} {'repère':22s} {'M (kN·m)':>9s} {'t (mm)':>8s} {'σ (N/mm²)':>10s} {'taux σ_éq':>10s}")
    pts_h = [(0.0, "nu du mur (nœud)"), (e / 2, "plan moyen aile vert."), (e, "section 2 (nu int.)"), (e + r / 2, "milieu du congé"),
             (e + r - 2.0, "≈ maximum (2 mm avant)"), (e + r, "section 1 (pied congé)"), ((e + r + X) / 2, "mi-portée"),
             (X, "charge")]
    for u, lib in pts_h:
        M = R * (X - u)
        t = ep(u) if e <= u <= e + r else (e if u > e + r else float("nan"))
        s = 6 * M / (Lr * t ** 2) if t == t else float("nan")
        q = math.sqrt(s * s + 3 * (R / (Lr * t)) ** 2) if t == t else float("nan")
        p(f"   {u:8.2f} {lib:22s} {M / 1e6:9.5f} {t:8.3f} {s:10.3f} {q / sadm:10.3%}")
    p("aile verticale, y = hauteur au-dessus du dessous ; M(y) = H·(Y + Yj − y), nul à l'ancrage")
    p(f"   {'y (mm)':>8s} {'repère':22s} {'M (kN·m)':>9s} {'t (mm)':>8s} {'σ (N/mm²)':>10s} {'taux σ_éq':>10s}")
    pts_v = [(e / 2, "pivot (nœud)"), (e, "section 4"), (e + r / 2, "milieu du congé"), (e + r - 2.0, "≈ maximum (2 mm avant)"),
             (e + r, "section 3 (pied congé)"), ((e + r + Y) / 2, "mi-hauteur"), (Y, "ancrage")]
    for y, lib in pts_v:
        M = H * (Y - y)
        t = ep(y) if e <= y <= e + r else (e if y > e + r else float("nan"))
        s = 6 * M / (Lr * t ** 2) if t == t else float("nan")
        q = math.sqrt(s * s + 3 * (R / (Lr * t)) ** 2) if t == t else float("nan")
        p(f"   {y:8.2f} {lib:22s} {M / 1e6:9.5f} {t:8.3f} {s:10.3f} {q / sadm:10.3%}")
    p("   (t = nan : la coupe traverse l'autre aile — le nœud d'angle, où la théorie des poutres ne s'applique pas)")

    # ------------------------------------------------------------------ 4. part de τ, sensibilité
    p("")
    p("== 4. Part du cisaillement dans σ_éq ; sensibilité à la longueur reprise ==")
    for i in (1, 2, 3, 4):
        p(f"section {i} : σ_éq/σ − 1 = {m[f'seq{i}'] / m[f'sigma{i}'] - 1:+.4%}")
    for Lx in (250.0, 300.0, 350.0):
        p(f"L = {Lx:.0f} mm : taux max = {modele(dict(c, L=Lx))['taux_max']:.4%}")
    p(f"cisaillement maximal de Jourawski au plan neutre (1,5 × moyen), section 1 : {1.5 * m['tau_e']:.4f} N/mm² ; "
      f"σ_éq correspondant (σ = 0) = √3·τ = {math.sqrt(3) * 1.5 * m['tau_e']:.4f} N/mm²")

    # ------------------------------------------------------------------ 5. ELU
    p("")
    p("== 5. Pour situer : l'ELU de la note (68,9 kN, P8!C71) ==")
    p(f"R_ELU / R_ELS = {k:.5f} ; le classeur vérifie σ_éq(ELS) ≤ f_y/{c['gM']} soit {c['gM']}·σ_éq(ELS) ≤ f_y")
    q_elu = k * m["seq3"]
    p(f"section 3 à l'ELU, élastique, f_y/1,0 : σ_éq = {q_elu:.4f} N/mm² → {q_elu / c['fy']:.4%}")
    p(f"   rapport au taux du classeur : {q_elu / c['fy'] / m['taux3']:.5f} (= {k:.5f}/{c['gM']})")
    M3_elu = k * m["M3"] * 1e6
    Wpl = Lr * e ** 2 / 4
    V_elu = c["R_ELU"]
    Vpl = Lr * e * c["fy"] / math.sqrt(3)
    p(f"section 3 à l'ELU, plastique : M3,Ed = {M3_elu / 1e6:.5f} kN·m ; M_pl = L·e²/4·f_y = {Wpl * c['fy'] / 1e6:.5f} kN·m "
      f"→ {M3_elu / (Wpl * c['fy']):.4%} ; V_Ed/V_pl = {V_elu / Vpl:.4%}")
    for i in (1, 2, 4):
        Wp = Lr * (e if i == 1 else e + r) ** 2 / 4
        p(f"section {i} à l'ELU, plastique : {k * m[f'M{i}'] * 1e6 / (Wp * c['fy']):.4%} ; élastique σ_éq : "
          f"{k * m[f'seq{i}'] / c['fy']:.4%}")

    # ------------------------------------------------------------------ 6. ancrages
    p("")
    p("== 6. Ancrages ==")
    p(f"N_ELS = Mmax/((Y − jeu)·n) = {Mmax / 1e6:.4f} kN·m / ({z_anc / 1e3} m × {c['n']}) = {m['N_ELS']:.5f} kN ; "
      f"V_ELS = R/n = {m['V_ELS']:.5f} kN")
    p(f"N = V ici parce que X = Y − jeu ({X} = {z_anc}) : coïncidence du cas, pas une règle")
    p(f"pression diamétrale (C88) = V_ELS/(d·e) = {m['p_diam']:.4f} N/mm² ; à l'ELU ×{k:.4f} = {m['p_diam'] * k:.4f} N/mm²")
    d0 = c["d"] + 2
    p(f"arrachement (C90) = V_ELS/(2·(haut − Y)·e) = {m['tau_arr']:.4f} N/mm² ; à l'ELU = {m['tau_arr'] * k:.4f} N/mm²")
    t_arr_net = m["V_ELS"] * 1e3 / (2 * (c["haut"] - Y - d0 / 2) * e)
    p(f"   en retranchant le demi-trou (d0 = d + 2 = {d0} mm, hypothèse) : longueur {c['haut'] - Y - d0 / 2} mm → "
      f"{t_arr_net:.4f} N/mm² ELS ({t_arr_net / m['tau_arr'] - 1:+.2%})")
    p("   aucune de ces deux contraintes n'est comparée à une limite (les mises en forme de C88 et C90 visent #REF!)")

    # ------------------------------------------------------------------ 7. domaine
    p("")
    p("== 7. Ce que le classeur calcule hors de la géométrie (aucun contrôle de domaine) ==")
    for lib, q in (("X = 20 mm < e + r = 28 mm (charge sur le congé)", dict(c, X=20.0)),
                   ("Y = 25 mm < e + r = 28 mm (ancrage dans le congé)", dict(c, Y=25.0)),
                   ("X + Xj = 75 + 50 = 125 mm > largeur 120 mm", dict(c, Xj=50.0)),
                   ("Y = jeu = 10 mm (bras d'ancrage nul)", dict(c, Y=10.0 + 1e-9))):
        try:
            mm = modele(q)
            p(f"{lib} : M1 = {mm['M1']:+.4f}, M3 = {mm['M3']:+.4f} kN·m, ρ3 = {mm['rho3']:+.4f}, taux max = {mm['taux_max']:.2%}, "
              f"N_ELS = {mm['N_ELS']:.4g} kN")
        except ZeroDivisionError:
            p(f"{lib} : division par zéro")

    # ------------------------------------------------------------------ 8. E10A BD
    p("")
    p("== 8. E10A BD : retrouver 42,76 kN (ELS) et 68,9 kN (ELU) ? ==")
    B = {1: 1184.5, 2: 2773.33, 4: 1109.67, 6: 2747.87, 7: 2284.97, 9: 2777.01, 10: 3628.21, 12: 5793.0}
    S = {1: 1.79, 2: 4.14, 4: 1.78, 6: 4.14, 7: 3.44, 9: 4.14, 10: 4.84, 12: 6.14}
    q_expl, larg = 4.0, 1.345
    cand = {}
    for v in B:
        G, Q = B[v] / 100, S[v] * q_expl
        cand[f"volée {v} : (G+Q)/2"] = (G + Q) / 2
        cand[f"volée {v} : (1,35G+1,5Q)/2"] = (1.35 * G + 1.5 * Q) / 2
        cand[f"volée {v} : (G+Q)/2/1,345 (kN/m)"] = (G + Q) / 2 / larg
        cand[f"volée {v} : (G+Q)/2/1,345 × 0,30 m"] = (G + Q) / 2 / larg * 0.30
    for v1, v2 in itertools.combinations(B, 2):
        cand[f"volées {v1}+{v2} : (G+Q)/2"] = (B[v1] / 100 + S[v1] * q_expl + B[v2] / 100 + S[v2] * q_expl) / 2
    for cible in (42.76, 68.9):
        proches = sorted(cand.items(), key=lambda kv: abs(kv[1] / cible - 1))[:4]
        p(f"cible {cible} kN : " + " ; ".join(f"{n} = {v:.3f} ({v / cible - 1:+.2%})" for n, v in proches))
    G12, Q12 = B[12] / 100, S[12] * q_expl
    p(f"volée 12 : rapport ELU/ELS avec 1,35 G + 1,5 Q = {(1.35 * G12 + 1.5 * Q12) / (G12 + Q12):.4f} ; "
      f"rapport de la note 68,9/42,76 = {68.9 / 42.76:.4f} (supérieur à 1,5, le plus grand coefficient d'EN 1990)")
    p(f"42,76 kN sur 0,30 m = {42.76 / 0.30:.2f} kN/m ; plus grande réaction linéaire de E10A BD (D+G) = "
      f"{max((B[v] / 100 + S[v] * q_expl) / 2 / larg for v in B):.4f} kN/m")

    # ------------------------------------------------------------------ 9. le moteur actuel du site
    p("")
    p("== 9. Le moteur actuel du site (modèle « lame », lecture seule) sur le même cas ==")
    racine = "/home/user/integ"
    avant = {os.path.join(d, f): os.stat(os.path.join(d, f)).st_mtime for d, _, fs in os.walk(os.path.join(racine, "acier")) for f in fs}
    sys.path.insert(0, racine)
    try:
        from acier.corniere import moteur as MS       # noqa: E402
        g = MS.calculer(t=e, V_char=R / 1e3, charge="ponctuelle", L=Lr / 1e3, e=X / 10, gamma_ELU=k, s=10.0,
                        sig_lim_ELS=sadm, tau_lim_ELS=sadm / math.sqrt(3), sig_lim_ELU=c["fy"])
        p(f"entrée : t = {e} mm, V = {R / 1e3} kN ponctuelle sur L = {Lr / 1e3} m, e = {X / 10} cm (= X), γ_ELU = {k:.5f}, "
          f"limites σ = {sadm:.4f}, τ = {sadm / math.sqrt(3):.4f}, σ_éq,ELU = {c['fy']} N/mm²")
        p(f"site : M_ELS = {g['M_ELS']:.5f} kN·m/m (× 0,30 m = {g['M_ELS'] * 0.3:.5f} kN·m = Mmax du classeur {m['Mmax']:.5f}) ; "
          f"σ_ELS = {g['sigma_ELS']:.4f} ; τ_ELS = {g['tau_ELS']:.4f} ; σ_éq,ELS = {g['sigma_eq_ELS']:.4f} N/mm²")
        p(f"site : U_σ = {g['U_sigma']:.4%} ; U_ELU = {g['U_ELU']:.4%} ; classeur (section 3) : {m['taux3']:.4%}")
        p(f"   le site vérifie la section au NU DU MUR, épaisseur e, moment entier R·X : c'est M_max/W_e "
          f"= {m['Mmax'] * 1e6 / m['W_e']:.4f} N/mm², sans congé ni rapport Ye/YH")
        p(f"   avec ses défauts (σ_lim,ELS = 155, τ_lim,ELS = 0,8) : U_τ = {g['tau_ELS'] / 0.8:.1%}")
    finally:
        sys.path.remove(racine)
    apres = {os.path.join(d, f): os.stat(os.path.join(d, f)).st_mtime for d, _, fs in os.walk(os.path.join(racine, "acier")) for f in fs}
    p(f"contrôle : fichiers de {racine}/acier ajoutés ou modifiés par ce script : "
      f"{sorted(set(apres) - set(avant)) + [f for f in avant if f in apres and apres[f] != avant[f]] or 'aucun'}")

    os.makedirs(os.path.dirname(SORTIE), exist_ok=True)
    with open(SORTIE, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

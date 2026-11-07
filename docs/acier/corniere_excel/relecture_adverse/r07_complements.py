# -*- coding: utf-8 -*-
"""Relecture adverse — 7. Compléments : hauteur d'ancrage et section 3 ; statique des sorties de c03c ;
contrainte locale au droit de l'ancrage ; K_FI."""
import math
R, X, e, r, L = 42760.0, 75.0, 15.0, 13.0, 300.0
SADM = 355 / 1.5
def classeur3(Y):
    M = R * X * (Y - e - r) / (Y - e / 2)
    s = M / (L * e * e / 6); t = R / (L * e)
    return math.sqrt(s * s + 3 * t * t) / SADM
def p0_3(Y):
    H = R * (X - e / 2) / Y
    M = H * (Y - e - r)
    s = M / (L * e * e / 6) + R / (L * e); t = H / (L * e)
    return math.sqrt(s * s + 3 * t * t) / SADM, H
for Y in (65, 85, 105):
    t0, H = p0_3(Y)
    print(f"Y = {Y}: classeur S3 {classeur3(Y)*100:.2f} % ; P0 S3 {t0*100:.2f} % ; H(P0) {H/1e3:.2f} kN ; traction classeur {R*X/(Y-10)/1e3:.2f} kN")
# statique des sorties c03c : R (X - e/2) = T (yT - yc)
for nom, T, yc in (("mur rigide", 40.777467876195296, 0.0), ("30 N/mm³", 33.07646972689814, 11.75628234108912)):
    print(f"c03c {nom}: T = {T:.2f} kN, y_c = {yc:.2f} -> résultante d'ancrage à y_T = {yc + R*(X-e/2)/(T*1e3):.1f} mm (axe à 85, rondelle 70 à 100)")
# poutre P(y_P = 11,8 ; u_V = e/2)
H = R * (X - e / 2) / (85 - 11.8); M = H * 57
s = M / 11250 + R / 4500; t = H / 4500
print(f"poutre pivot 11,8 mm : H {H/1e3:.2f} kN, S3 {math.sqrt(s*s+3*t*t)/SADM*100:.2f} %")
# contrainte locale au droit de l'ancrage (§5.3)
leff, m = 282.1, (85 - e) - 0.8 * r
for nom, N1, V1 in (("rigides", 25.16e3, 29.93e3), ("souples", 20.87e3, 30.16e3)):
    a = 6 * N1 * 57 / (leff * e * e) + V1 / (leff * e)
    b = 6 * N1 * m / (leff * e * e) + V1 / (leff * e)
    print(f"{nom}: bras 57 -> {a:.1f} N/mm² ; bras m = {m:.1f} -> {b:.1f} N/mm² ({b/SADM:.0%})")
print(f"k_pl = 3E(l_eff e³/12)/Y³ = {3*210000*(leff*e**3/12)/85**3/1e3:.1f} kN/mm")
# K_FI
print(f"rapport ELU/ELS max avec K_FI = 1,1 : {1.1*1.5:.3f} ; volée 12 : {1.1*(1.35*21.5353+1.5*9.1301)/(21.5353+9.1301):.3f}")
# poutre continue : raideur de flexion d'une travée
EI = 210000 * 444.881e4
print(f"48EI/s³ (s = 800) = {48*EI/800**3/1e3:.1f} kN/mm")
# seuil de levier (2D, déformation plane) : k > 720·EI'/(Y³·(H_aile-Y)) par tranche de 300 mm
EIp = 210000 / (1 - 0.09) * 300 * 15 ** 3 / 12
k = 6 * EIp * 120 / (85 ** 2 * 85 * 35)
print(f"seuil analytique de levier (talon articulé, ancrage, débord 35 mm) : {k/3/1e3:.0f} kN/mm par cheville")

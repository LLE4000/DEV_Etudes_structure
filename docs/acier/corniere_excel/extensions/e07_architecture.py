# -*- coding: utf-8 -*-
"""E07 — Architecture : réutiliser le catalogue partagé, préparer les cornières PRS, situer le moteur actuel.

Lit le code du site EN LECTURE SEULE (import avec -B : aucun fichier .pyc écrit dans /home/user/integ).

1. Le moteur actuel du site (acier/corniere/moteur.py) sur le cas de contrôle.
2. Les fonctions de géométrie partagées (acier/profiles/geometrie.py) sur :
   - la L120×120×15 du catalogue (contrôle : on retrouve la fiche) ;
   - une cornière PRS de deux plats soudés (aile horizontale 15 mm, aile verticale 12 mm, cordon intérieur a = 6).
3. La fonction « épaisseur locale » t(s) pour un congé laminé et pour un cordon : ce qui change pour les 4 sections.

Lancer : /usr/bin/python3.12 -I -B e07_architecture.py
"""
import math
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, "/home/user/integ")
import numpy as np  # noqa: E402

import ext_commun as K  # noqa: E402
from acier.profiles import geometrie as GEO  # noqa: E402
from acier.corniere import moteur as MOT  # noqa: E402

out = []
P = out.append
c = K.CAS

P("# E07 — architecture")
P("")
P("## 1. Le moteur actuel du site sur le cas de contrôle")
sadm = c["fy"] / c["gM"]
cor = dict(nom="contrôle", standard=True, profil="L 120x120x15", charge="ponctuelle", V=c["R"] / 1e3, L=c["L"] / 1e3,
           e=c["X"] / 10.0, gamma_ELU=1.0, s=30.0, bord=0.0, sig_lim_ELS=sadm, tau_lim_ELS=sadm / math.sqrt(3),
           sig_lim_ELU=c["fy"])
v = MOT.verifier(cor)
g = v["grandeurs"]
vv = {x["cle"]: x for x in v["verifications"]}
P(f"moteur actuel (modèle « lame », M = V·e) : σ_ELS = {g['sigma_ELS']:.2f} N/mm², σ_éq,ELS = {g['sigma_eq_ELS']:.2f} ; "
  f"U_σ = {vv['els_sigma']['taux'] * 100:.2f} % ; la cornière est vérifiée au nu du mur avec "
  f"M = R·X = {g['M_ELS'] * c['L'] / 1e3:.4f} kN·m sur 300 mm et l'épaisseur e")
ex = K.excel(c)
P(f"classeur : section 3 = {ex['taux'][2] * 100:.2f} % ; écart du moteur actuel : "
  f"{(vv['els_sigma']['taux'] / ex['taux'][2] - 1) * 100:+.1f} %")

P("")
P("## 2. Géométrie partagée : L laminée et cornière PRS")
lam = GEO.proprietes([GEO.contour_L(120.0, 120.0, 15.0, 13.0, 6.5)], plastique=False)
It_lam = GEO.torsion_saint_venant([GEO.contour_L(120.0, 120.0, 15.0, 13.0, 6.5)], 0.75)
P(f"L120×120×15 (contour_L, r1 = 13, r2 = 6,5) : A = {lam['A'] / 100:.3f} cm², Iy = {lam['Iy'] / 1e4:.2f} cm⁴, "
  f"Iyz = {lam['Iyz'] / 1e4:.2f} cm⁴, yc = {lam['yc']:.2f} mm, It = {It_lam / 1e4:.2f} cm⁴ (fiche : 33,93 ; 444,88 ; "
  f"−260,22 ; 35,15 ; 26,43)")
# PRS : plat horizontal 120 × th (u de 0 à 120, y de 0 à th), plat vertical tv × (120 − th) posé dessus contre le mur,
# cordon d'angle intérieur de gorge a (côté z = a·√2), compté comme matière
th, tv, a = 15.0, 12.0, 6.0
z = a * math.sqrt(2)
B = H = 120.0
pts = np.array([(0, 0), (B, 0), (B, th), (tv + z, th), (tv, th + z), (tv, H), (0, H)], float)
prs = GEO.proprietes([pts], plastique=False)
It_prs = GEO.torsion_saint_venant([pts], 0.5)
P(f"PRS {B:.0f}×{H:.0f}, aile horizontale {th:.0f} mm, aile verticale {tv:.0f} mm, cordon intérieur a = {a:.0f} mm "
  f"(côté {z:.2f} mm) : A = {prs['A'] / 100:.3f} cm², Iy = {prs['Iy'] / 1e4:.2f} cm⁴, Iz = {prs['Iz'] / 1e4:.2f} cm⁴, "
  f"Iyz = {prs['Iyz'] / 1e4:.2f} cm⁴, centre ({prs['yc']:.2f} ; {prs['zc']:.2f}) mm, It = {It_prs / 1e4:.2f} cm⁴ "
  f"(It de deux plats minces séparés : {(B * th ** 3 + (H - th) * tv ** 3) / 3 / 1e4:.2f} cm⁴)")
P("Remarque : la section continue (cordon compté comme matière) est une idéalisation — entre le plat vertical et le "
  "plat horizontal, la continuité n'est assurée que par le cordon et le contact.")

P("")
P("## 3. Épaisseur locale t(s) : congé laminé et cordon")


def t_lamine(s, e, r):
    return K.t_conge(s, e, r)


def t_cordon(s, e_aile, t_autre, zc):
    """Aile d'épaisseur e_aile ; s mesuré depuis le dos de l'autre aile (épaisseur t_autre) ; cordon de côté zc."""
    if s >= t_autre + zc:
        return e_aile
    if s < t_autre:
        return None
    return e_aile + (t_autre + zc - s)


for s_ in (12.0, 15.0, 18.0, 20.0, 22.0, 24.0, 28.0):
    tl = t_lamine(s_, 15.0, 13.0)
    tc = t_cordon(s_, th, tv, z)
    P(f"  s = {s_:5.1f} mm : L laminée (e = 15, r = 13) t = {('—' if tl is None else f'{tl:.3f}')} ; "
      f"PRS aile horizontale (th = 15, plat vertical 12, cordon {z:.2f}) t = {('—' if tc is None else f'{tc:.3f}')}")
P("Sections du classeur transposées au PRS : 1 = pied du cordon sur l'aile horizontale (s = tv + z), 2 = nu du plat "
  "vertical (s = tv, t = th + z), 3 = pied du cordon sur le plat vertical (y = th + z), 4 = interface des deux plats "
  "(y = th) : cette dernière n'est PAS une section d'acier continue — elle se vérifie comme un assemblage (cordon + contact).")
K.ecrire(os.path.join(K.ICI, "e07_sortie.txt"), out)
print("\n".join(out))

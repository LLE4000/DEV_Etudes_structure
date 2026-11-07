# -*- coding: utf-8 -*-
"""c03 — Éléments finis plans de la coupe du cas de contrôle (L120×120×15, r = 13, q = 42,76 kN / 300 mm).

1. validation du maillage contre la théorie des poutres, loin du coin ;
2. le cas de base (mur rigide, cheville rigide) : contact au mur (pivot réel), effort dans la cheville, effet de
   levier en tête d'aile, contraintes aux quatre sections du classeur, pic dans le congé ;
3. la section 2 : distribution réelle de σ_xx sur la coupe x = e, module « effectif » M / σ_max ;
4. variantes : cheville souple (déplacement sous l'effort de service), mur souple (Winkler) ;
5. figures : carte de von Mises sur la déformée, σ le long des coupes.
"""
import json
import math
import os
import sys
import time

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import numpy as np  # noqa: E402
import commun as K  # noqa: E402
import ef_coupe as EF  # noqa: E402

RACINE = os.path.dirname(ICI)
SORT = os.path.join(RACINE, "sorties")
p = K.CAS
e, r, X, Y, larg, haut = p["e"], p["r"], p["X"], p["Y"], p["larg"], p["haut"]
b = p["b_cm"] * 10
q = p["R"] * 1000 / b
xl = K.classeur(**p)
H = float(os.environ.get("EF_PAS", "0.5"))
res = {"pas_maillage_mm": H}

t0 = time.time()
mod = EF.Coupe(larg, haut, e, r, h=H)
print(f"Maillage : {len(mod.nodes)} nœuds, {len(mod.els)} éléments (pas {H} mm) — {time.time() - t0:.1f} s")
res["maillage"] = dict(noeuds=len(mod.nodes), elements=len(mod.els))

# ---------------------------------------------------------------- 2. cas de base
# cheville : δ = 0,1 mm sous son effort de service (C82 = 14,25 kN) → raideur par mm de longueur n·k₁/b ;
# réaction verticale de la tige au milieu de l'épaisseur (x_V = e/2) ; mur rigide.
T_anc = xl["N_ELS"] * 1000
DELTA0 = 0.1
K0 = p["n_anc"] * (T_anc / DELTA0) / b
R0 = mod.resoudre(q, X, Y, k_tige=K0)
yT0 = (R0["rond_f"] * R0["rond_y"]).sum() / R0["rond_f"].sum()
print(f"Cas de base (cheville δ = {DELTA0} mm, mur rigide) : résultante sous la rondelle à y = {yT0:.2f} mm "
      f"(axe Y = {Y}) ; T par le ressort = {R0['T_ressort'] * b / 1000:.2f} kN")
print(f"Cas de base : {R0['iterations']} itérations de contact ; ΣV tige = {R0['V']:.3f} N/mm (q = {q:.3f}) ; "
      f"T = {R0['T']:.3f} N/mm ; C = {R0['C']:.3f} N/mm")
mur_y, mur_f = R0["mur_y"], R0["mur_f"]
bas = mur_y < Y - 15 - 1e-6
haut_ = mur_y > Y + 15 + 1e-6
sous_rond = ~bas & ~haut_
C_bas = mur_f[bas].sum()
yc_bas = (mur_f[bas] * mur_y[bas]).sum() / C_bas
ext_bas = mur_y[bas][mur_f[bas] > 1e-9 * q].max()
Q_tete = mur_f[haut_].sum()
C_rond = mur_f[sous_rond].sum()
print(f"  mur, sous la rondelle : C = {C_bas:.3f} N/mm, résultante à y_c = {yc_bas:.2f} mm, contact jusqu'à "
      f"y = {ext_bas:.2f} mm ; au droit de la rondelle : {C_rond:.3f} N/mm ; en tête (y > Y + 15) : {Q_tete:.3f} N/mm")
T_tot = R0["T"] * b / 1000
print(f"  traction totale des chevilles T = {T_tot:.2f} kN (classeur, sections : R·X/YH = "
      f"{p['R'] * X / xl['C25_YH']:.2f} kN ; ancrages : R·X/(Y − 10) = {p['R'] * X / xl['z_ancrage_mm']:.2f} kN)")
z_ef = p["R"] * X / T_tot
res["base"] = dict(delta=DELTA0, y_rondelle=yT0, T_kN=T_tot, T_par_mm=R0["T"], V_par_mm=R0["V"], C_bas=C_bas, yc=yc_bas, contact_jusqua=ext_bas,
                   C_sous_rondelle=C_rond, Q_tete=Q_tete, bras_equivalent_RX_sur_T=z_ef,
                   T_classeur_sections=p["R"] * X / xl["C25_YH"], T_classeur_ancrages=p["R"] * X / xl["z_ancrage_mm"])

# ---------------------------------------------------------------- 1. validation loin du coin
v = mod.coupe((50, 0), (50, e))
beam = 6 * q * (X - 50) / e ** 2
print(f"Validation x = 50 mm : M_EF = {abs(v['M']):.1f} N·mm/mm (statique {q * (X - 50):.1f}) ; σ fibre EF = "
      f"{abs(v['sn_ext']).max():.2f} MPa, poutre = {beam:.2f} MPa ({(abs(v['sn_ext']).max() / beam - 1) * 100:+.2f} %)")
res["validation"] = dict(M_EF=abs(v["M"]), M_statique=q * (X - 50), sigma_EF=float(abs(v["sn_ext"]).max()),
                         sigma_poutre=beam)
vv = mod.coupe((0, 50), (e, 50))
print(f"Validation y = 50 mm (aile verticale) : N = {vv['N']:.2f} N/mm (q = {q:.2f}), V = {abs(vv['V']):.2f} "
      f"(T = {R0['T']:.2f}), M = {abs(vv['M']):.1f}")
res["validation_verticale"] = dict(N=vv["N"], V=abs(vv["V"]), M=abs(vv["M"]))


# ---------------------------------------------------------------- 2'. les quatre sections du classeur
def resume(nom, cut, sig_classeur, M_classeur_kNm):
    sn_max = cut["sn_max"]
    print(f"  {nom} : L = {cut['L']:.2f} mm ; N = {cut['N']:+.2f} N/mm, |M| = {abs(cut['M']):.1f} N·mm/mm "
          f"(classeur {M_classeur_kNm * 1e6 / b:.1f}) ; σ linéarisé = {cut['lin_max']:.2f} ; σ max réel = "
          f"{sn_max:.2f} MPa ; classeur σ = {sig_classeur:.2f} MPa → réel/classeur = {sn_max / sig_classeur:.3f}")
    return dict(L=cut["L"], N=cut["N"], M=abs(cut["M"]), M_classeur=M_classeur_kNm * 1e6 / b,
                sigma_lin=cut["lin_max"], sigma_max=sn_max, sigma_classeur=sig_classeur,
                rapport=sn_max / sig_classeur, sn_ext=list(map(float, cut["sn_ext"])))


cuts = {"S1": ((e + r, 0), (e + r, e)), "S2": ((e, 0), (e, e + r)),
        "S3": ((0, e + r), (e, e + r)), "S4": ((0, e), (e + r, e))}
res["sections"] = {}
coupes_ef = {}
print("Sections du classeur, contraintes EF (cas de base) :")
for s, (A, B_) in cuts.items():
    cut = mod.coupe(A, B_, n=300)
    coupes_ef[s] = cut
    res["sections"][s] = resume(s, cut, xl[s]["sigma"], xl[s]["M"])

# section 2 : forme de la distribution
c2 = coupes_ef["S2"]
i_max = int(np.nanargmax(np.abs(c2["sn"])))
W_eff = abs(c2["M"]) / c2["sn_max"]
W_er = (e + r) ** 2 / 6
W_e = e ** 2 / 6
print(f"  S2 : σ_xx max = {c2['sn_max']:.2f} MPa à y = {c2['s'][i_max]:.2f} mm ; σ_xx en fibre haute (y = e + r) = "
      f"{c2['sn_ext'][1]:.2f} MPa (bord libre vertical) ; W effectif = M/σ_max = {W_eff:.2f} mm³/mm contre "
      f"W(e+r) = {W_er:.2f} et W(e) = {W_e:.2f} → épaisseur équivalente √(6 W_eff) = {math.sqrt(6 * W_eff):.2f} mm")
res["S2_detail"] = dict(y_max=float(c2["s"][i_max]), sigma_haut=float(c2["sn_ext"][1]), W_eff=W_eff, W_er=W_er,
                        W_e=W_e, t_equivalente=math.sqrt(6 * W_eff))
c4 = coupes_ef["S4"]
W4 = abs(c4["M"]) / c4["sn_max"]
print(f"  S4 : W effectif = {W4:.2f} mm³/mm → épaisseur équivalente {math.sqrt(6 * W4):.2f} mm (classeur : e + r = 28)")
res["S4_detail"] = dict(W_eff=W4, t_equivalente=math.sqrt(6 * W4))

# ---------------------------------------------------------------- pic dans le congé
x, y = mod.nodes[:, 0], mod.nodes[:, 1]
arc = np.where(np.abs(np.hypot(x - (e + r), y - (e + r)) - r) < 1e-6)[0]
ang = np.degrees(np.arctan2(y[arc] - (e + r), x[arc] - (e + r)))
i1 = arc[np.argmax(mod.s1[arc])]
iv = arc[np.argmax(mod.vm[arc])]
a1 = math.degrees(math.atan2(y[i1] - e - r, x[i1] - e - r))
print(f"Congé (surface) : σ1 max = {mod.s1[i1]:.2f} MPa à ({x[i1]:.2f}, {y[i1]:.2f}), angle {a1:.1f}° "
      f"(270° = pied côté aile horizontale, 180° = pied côté aile verticale) ; von Mises max = {mod.vm[iv]:.2f}")
# faces intérieures et extérieures, hors zones d'application (charge ± 5 mm, rondelle, tige)
zone = ((x <= e + r + 15) & (y <= e + r + 15))
ivm = np.where(zone)[0][np.argmax(mod.vm[zone])]
print(f"Coin (x, y ≤ e + r + 15) : von Mises max = {mod.vm[ivm]:.2f} MPa en ({x[ivm]:.2f}, {y[ivm]:.2f})")
# plus grande σ1 sur la face intérieure de l'aile verticale (x = e, y ∈ [e+r, Y − 15]) et de l'aile horizontale
fv = np.where((np.abs(x - e) < 1e-9) & (y >= e + r) & (y <= Y - 15))[0]
fh = np.where((np.abs(y - e) < 1e-9) & (x >= e + r) & (x <= X - 5))[0]
print(f"Face intérieure aile verticale : σ1 max = {mod.s1[fv].max():.2f} à y = {y[fv][np.argmax(mod.s1[fv])]:.2f} ; "
      f"aile horizontale : |σ| max = {np.abs(mod.sig[fh, 0]).max():.2f} à x = {x[fh][np.argmax(np.abs(mod.sig[fh, 0]))]:.2f}")
vm_tous = [(mod.vm[iv], "congé")]
res["conge"] = dict(s1_max=float(mod.s1[i1]), angle=a1, x=float(x[i1]), y=float(y[i1]), vm_max_arc=float(mod.vm[iv]),
                    vm_max_coin=float(mod.vm[ivm]), vm_max_coin_xy=[float(x[ivm]), float(y[ivm])],
                    s1_face_verticale=float(mod.s1[fv].max()),
                    sxx_face_horizontale=float(np.abs(mod.sig[fh, 0]).max()))
taux_ef = max(mod.vm[ivm], mod.s1[i1]) / xl["adm"]
print(f"Taux « pic élastique » : {taux_ef * 100:.1f} % (classeur : {xl['D3_taux_max'] * 100:.2f} %)")
res["taux_pic"] = taux_ef

# balayage continu des coupes verticales et horizontales (σ max sur chaque coupe) pour la figure
bal_v, bal_h = [], []
for xx in np.r_[np.linspace(e + 0.05, e + r, 27), np.linspace(e + r + 1, X - 5, 20)]:
    t = K.epaisseur_verticale(xx, e, r) if xx < e + r else e
    cu = mod.coupe((xx, 0), (xx, t), n=120)
    bal_v.append((float(xx), float(cu["sn_max"]), float(cu["lin_max"]), float(abs(cu["M"]))))
for yy in np.r_[np.linspace(e + 0.05, e + r, 27), np.linspace(e + r + 1, Y - 16, 20)]:
    t = K.epaisseur_verticale(yy, e, r) if yy < e + r else e
    cu = mod.coupe((0, yy), (t, yy), n=120)
    bal_h.append((float(yy), float(cu["sn_max"]), float(cu["lin_max"]), float(abs(cu["M"])), float(cu["N"])))
bv = max(bal_v, key=lambda t: t[1])
bh = max(bal_h, key=lambda t: t[1])
print(f"Balayage EF : aile horizontale σ max = {bv[1]:.2f} MPa à x = {bv[0]:.2f} ; aile verticale σ max = "
      f"{bh[1]:.2f} MPa à y = {bh[0]:.2f} (classeur S3 = {xl['S3']['sigma']:.2f})")
res["balayage"] = dict(vertical=bal_v, horizontal=bal_h, max_h=bv, max_v=bh)

# ---------------------------------------------------------------- 4. variantes : cheville et mur souples
var = []
for delta in (None, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
    kc = None if delta is None else p["n_anc"] * (T_anc / delta) / b
    Rv = mod.resoudre(q, X, Y, k_tige=kc)
    my, mf = Rv["mur_y"], Rv["mur_f"]
    bs = my < Y - 15 - 1e-6
    hs = my > Y + 15 + 1e-6
    Cb = mf[bs].sum()
    ycb = (mf[bs] * my[bs]).sum() / Cb
    s3 = mod.coupe((0, e + r), (e, e + r), n=200)
    yT = (Rv["rond_f"] * Rv["rond_y"]).sum() / Rv["rond_f"].sum()
    var.append(dict(cheville_delta_mm=delta, y_rondelle=yT, T_kN=Rv["T"] * b / 1000, yc=ycb, Q_tete_kN=mf[hs].sum() * b / 1000,
                    contact_haut=float(my[hs][mf[hs] > 1e-9].min()) if (mf[hs] > 1e-9).any() else None,
                    sigma_S3=s3["sn_max"], u_rondelle=Rv["u_rondelle"]))
    print(f"Cheville {'rigide' if delta is None else f'δ = {delta} mm':>12s} : T = {var[-1]['T_kN']:.2f} kN, "
          f"y_c = {ycb:.2f} mm, résultante rondelle y = {yT:.1f} mm, levier en tête Q = {var[-1]['Q_tete_kN']:.2f} kN, "
          f"σ_S3 = {s3['sn_max']:.1f} MPa")
res["variantes_cheville"] = var
varm = []
for kw in (None, 5000.0, 1000.0, 300.0, 100.0, 30.0):
    Rv = mod.resoudre(q, X, Y, k_mur=kw, k_tige=p["n_anc"] * (T_anc / 0.1) / b)
    my, mf = Rv["mur_y"], Rv["mur_f"]
    bs = my < Y - 15 - 1e-6
    Cb = mf[bs].sum()
    ycb = (mf[bs] * my[bs]).sum() / Cb
    ext = float(my[bs][mf[bs] > 1e-9 * q].max())
    s3 = mod.coupe((0, e + r), (e, e + r), n=200)
    varm.append(dict(k_mur=kw, T_kN=Rv["T"] * b / 1000, yc=ycb, contact_jusqua=ext, sigma_S3=s3["sn_max"]))
    print(f"Mur {'rigide' if kw is None else f'k = {kw:g} N/mm³':>16s} (cheville δ = 0,1 mm) : T = {varm[-1]['T_kN']:.2f} kN, "
          f"y_c = {ycb:.2f} mm, contact jusqu'à {ext:.1f} mm, σ_S3 = {s3['sn_max']:.1f} MPa")
res["variantes_mur"] = varm
for xv in (0.0, e / 2, e):
    Rv = mod.resoudre(q, X, Y, x_V=xv, k_tige=p["n_anc"] * (T_anc / 0.1) / b)
    s3 = mod.coupe((0, e + r), (e, e + r), n=200)
    print(f"Réaction verticale de la tige à x_V = {Rv['x_tige']:.1f} mm (cheville δ = 0,1 mm) : T = {Rv['T'] * b / 1000:.2f} kN, "
          f"σ_S3 = {s3['sn_max']:.1f} MPa, N_S3 = {s3['N']:.1f} N/mm, M_S3 = {abs(s3['M']):.0f} N·mm/mm")
    res.setdefault("variantes_xV", []).append(dict(x_V=Rv["x_tige"], T_kN=Rv["T"] * b / 1000, sigma_S3=s3["sn_max"],
                                                   N_S3=s3["N"], M_S3=abs(s3["M"])))

# ---------------------------------------------------------------- 5. figures
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    mod.resoudre(q, X, Y, k_tige=K0)                                  # revient au cas de base
    amp = 15.0 / np.abs(mod.u).max()
    U = mod.u.reshape(-1, 2)
    fig, axs = plt.subplots(1, 2, figsize=(13, 6.2))
    for ax, deform, titre in ((axs[0], False, "Contrainte de von Mises (MPa) — géométrie initiale"),
                              (axs[1], True, f"Déformée amplifiée ×{amp:.0f}")):
        P = mod.nodes + (amp * U if deform else 0)
        polys = P[mod.els]
        val = mod.vm[mod.els].mean(1)
        pc = PolyCollection(polys, array=val, cmap="turbo", edgecolors="none")
        pc.set_clim(0, 240)
        ax.add_collection(pc)
        ax.set_xlim(-12, larg + 5)
        ax.set_ylim(-8, haut + 5)
        ax.set_aspect("equal")
        ax.axvline(0, color="0.3", lw=2)
        ax.set_title(titre, fontsize=10)
        ax.annotate("", xy=(X, e), xytext=(X, e + 25), arrowprops=dict(arrowstyle="->", color="r", lw=2))
        ax.plot([e, e], [Y - 15, Y + 15], color="k", lw=4)
    fig.colorbar(pc, ax=axs, shrink=0.7, label="σ von Mises (MPa) — σ_adm = 236,7")
    for s, (A, B_) in cuts.items():
        axs[0].plot([A[0], B_[0]], [A[1], B_[1]], "w--", lw=1)
        axs[0].text((A[0] + B_[0]) / 2 + 1, (A[1] + B_[1]) / 2 + 1, s, color="w", fontsize=8)
    fig.savefig(os.path.join(SORT, "c03_carte_vm.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(1, 4, figsize=(15, 4))
    for ax, s in zip(axs, ("S1", "S2", "S3", "S4")):
        cu = mod.coupe(*cuts[s], n=300)
        ax.plot(cu["sn"], cu["s"], label="EF (σ normale à la coupe)")
        L = cu["L"]
        lin = cu["N"] / L + 12 * cu["M"] / L ** 3 * (cu["s"] - L / 2)
        ax.plot(lin, cu["s"], "--", label="linéarisée EF")
        sc = xl[s]["sigma"]
        ax.plot([-sc, sc], [0, L] if s in ("S1", "S3") else [0, L], ":", label=f"classeur ±{sc:.0f}")
        ax.set_title(f"{s} (longueur {L:.0f} mm)")
        ax.set_xlabel("σ (MPa)")
        ax.grid(alpha=.3)
    axs[0].set_ylabel("position le long de la coupe (mm)")
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(SORT, "c03_sections.png"), dpi=120)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
    cv = json.load(open(os.path.join(SORT, "c02_conge.json")))
    xs_ = [t[0] for t in bal_v]
    axs[0].plot(xs_, [t[1] for t in bal_v], "o-", ms=3, label="EF : σ max sur la coupe verticale")
    axs[0].plot([t[0] for t in cv["courbe_verticale"]], [t[2] for t in cv["courbe_verticale"]], "-",
                label="poutre, épaisseur exacte du congé")
    axs[0].axvline(e + r, color="0.5", ls=":")
    axs[0].axvline(e, color="0.5", ls=":")
    axs[0].plot([e + r], [xl["S1"]["sigma"]], "s", color="k", label="S1 classeur")
    axs[0].plot([e], [xl["S2"]["sigma"]], "^", color="k", label="S2 classeur")
    axs[0].set_xlabel("x (mm) depuis le nu du mur")
    axs[0].set_ylabel("σ (MPa)")
    axs[0].set_title("Aile horizontale")
    axs[0].legend(fontsize=8)
    axs[0].grid(alpha=.3)
    axs[1].plot([t[0] for t in bal_h], [t[1] for t in bal_h], "o-", ms=3, label="EF : σ max sur la coupe horizontale")
    axs[1].plot([t[0] for t in cv["courbe_horizontale_classeur"]], [6 * 0 + t[2] for t in cv["courbe_horizontale_classeur"]],
                "-", label="statique du classeur, épaisseur exacte")
    axs[1].axvline(e + r, color="0.5", ls=":")
    axs[1].axvline(e, color="0.5", ls=":")
    axs[1].plot([e + r], [xl["S3"]["sigma"]], "s", color="k", label="S3 classeur")
    axs[1].plot([e], [xl["S4"]["sigma"]], "^", color="k", label="S4 classeur")
    axs[1].set_xlabel("y (mm) depuis la sous-face")
    axs[1].set_title("Aile verticale")
    axs[1].legend(fontsize=8)
    axs[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(SORT, "c03_balayage.png"), dpi=120)
    plt.close(fig)
    print("Figures : c03_carte_vm.png, c03_sections.png, c03_balayage.png")
except Exception as exc:  # les figures ne conditionnent aucun chiffre
    print("Figures non produites :", exc)

with open(os.path.join(SORT, f"c03_ef_h{H}.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=1, default=float)
print(f"→ c03_ef_h{H}.json ({time.time() - t0:.0f} s)")

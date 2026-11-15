# -*- coding: utf-8 -*-
"""Générateur SAF — fondation de grue : les EFFORTS ELS pour le ferraillage (saf/fondation_grue/efforts.py).
Cahier des charges du lot 4 (15/11/2026), § 4 et § 11 : les tests T14, T15 et T17 du sous-lot 4b.

Ce que la suite prouve :
  1. T14, T15, T17 — les valeurs du cahier (sortie de l'annexe A) : moments et efforts tranchants intégrés sur les bandes,
     zone courante, prolongements, bande de 0,50 m ; l'intégration sur la largeur EXACTE ; les zones de renfort.
  2. LES RÈGLES — le pas en s (maille / 4), l'enveloppe des k, l'option « torsion », V lu à d du bord des zones chargées,
     la zone chargée de chaque type de plot, l'épaisseur des éléments sous un plot coulé, la rotation des moments.
  3. LA PARITÉ AVEC LE CALCUL DE RÉFÉRENCE — tests/fondation_grue_ref.py (``efforts``, numpy + scipy), indépendant : mêmes
     M(s), V(s), mêmes abscisses lues, mêmes zones, mêmes champs aux centres des éléments, cas par cas (dalle type avec et
     sans plots, bandes de 0,50 m, dalle en L avec châssis tourné de 30°, grue décalée).
  4. MUTATIONS — chaque altération plausible d'une règle fait rougir un contrôle (arbre miroir, contrôles rapides).

    /usr/bin/python3.12 tests/test_saf_grue_efforts.py
    /usr/bin/python3.12 tests/test_saf_grue_efforts.py --rapide [--racine dossier]   (les contrôles rapides seuls)
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(ICI)
ARGS = sys.argv[1:]
RACINE_MOTEUR = ARGS[ARGS.index("--racine") + 1] if "--racine" in ARGS else RACINE
RAPIDE = "--rapide" in ARGS
os.chdir(RACINE)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, RACINE_MOTEUR)
sys.path.insert(1, ICI)

import numpy as np  # noqa: E402

from saf.fondation_grue import efforts as EF  # noqa: E402
from saf.fondation_grue import plaque as PL  # noqa: E402
from saf.fondation_grue import projet as PR  # noqa: E402
from saf.fondation_grue import verification as V  # noqa: E402

OK, KO = [], []
JEUX = [(350, 440, 440, 350), (150, 150, 550, 550), (135, 100, 800, 135)]
KS = (2.0, 5.0, 10.0)                       # T14 : k = 2 ; 5 ; 10 MN/m³, C2 = k/10
D_V = 0.44                                  # T14 : d = 0,44 m
H, K = 0.5, 12.0
L_T7 = [(0, 0), (12, 0), (12, 7), (7, 7), (7, 12), (0, 12)]
PREF = {"type": "prefabrique", "a": 2.0, "b": 2.0, "h": 0.35}
COULE = {"type": "coule", "a": 2.0, "b": 2.0, "h": 0.35}


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{str(info)[:400]}]" if info != "" and not cond else ""))


def rect(Lx, Ly):
    return [(0, 0), (Lx, 0), (Lx, Ly), (0, Ly)]


def projet(maille=0.20, largeur=1.0, plots=None, jeux=JEUX, **k):
    """La dalle type du cahier (8 × 8 × 0,50 m, châssis 6 × 6 m centré, appuis 0,5 × 0,5 m, 3 jeux), avec la maille et la
    largeur de bande ÉCRITES dans T14 (0,20 m ; 1,00 m), modifiée par ``k`` (dalle, grue… fusionnés)."""
    el = PR.nouveau()
    el["options"]["maille"], el["bandes"]["largeur"] = maille, largeur
    el["reactions"] = [{"nom": "", "P": list(j)} for j in jeux]
    if plots:
        el["grue"]["plots"] = dict(plots)
    for cle, v in k.items():
        el[cle].update(v)
    return PR.lire(el)


def efforts(el, ks=KS, d_v=D_V, torsion=True):
    m = PR.modele(el)
    return m, EF.enveloppe(m, [V.calculer(m, k, k / 10, k / 10) for k in ks], d_v, torsion)


def r1(v):
    return round(float(v), 1)


def resume(ef):
    """Les valeurs imprimées par l'annexe A pour T14 : bande BX1, écart entre les 4 bandes, zone courante, prolongements."""
    B, c = ef["bandes"], ef["courante"]
    ecart = max(max(x[q] for x in B) - min(x[q] for x in B) for q in ("M_plus", "M_moins", "V_d"))
    return {"M+": r1(B[0]["M_plus"]), "M-": r1(B[0]["M_moins"]), "Vd": r1(B[0]["V_d"]), "ecart": round(ecart, 2),
            "inf": (r1(c["inf_x"]), r1(c["inf_y"])), "sup": (r1(c["sup_x"]), r1(c["sup_y"])),
            "plot": (r1(c["plot_inf"]), r1(c["plot_sup"])),
            "prol": (r1(max(x["prol_inf"] for x in B)), r1(min(x["prol_sup"] for x in B)))}


# ------------------------------------------------------------------ les contrôles RAPIDES (rejoués par les mutations)
def rapides():
    print("== 1. T14 — efforts ELS, dalle type, k = 2 ; 5 ; 10, C2 = k/10, maille 0,20, d = 0,44 m (sortie de l'annexe A)")
    t = time.time()
    m0, e0 = efforts(projet())
    s = resume(e0)
    chk("T14 sans plot, bandes de 1,00 m : M+ = 127,1 kNm ; M− = −126,5 kNm ; V à d = 172,8 kN ; les 4 bandes égales (écart 0,00)",
        (s["M+"], s["M-"], s["Vd"], s["ecart"]) == (127.1, -126.5, 172.8, 0.0), s)
    chk("T14 sans plot, zone courante (centres d'éléments, torsion comprise) : inf x / y = 92,2 / 92,2 ; sup x / y = −164,1 / −164,1 kNm/m ; "
        "prolongements jusqu'au bord : inf 141,0 ; sup −175,9 kNm/m",
        s["inf"] == (92.2, 92.2) and s["sup"] == (-164.1, -164.1) and s["prol"] == (141.0, -175.9), s)
    _, ep = efforts(projet(plots=PREF))
    s = resume(ep)
    chk("T14 plots préfabriqués 2,00 × 2,00 × 0,35 (diffusion à 45°) : M+ = 74,7 ; M− = −134,9 kNm ; V à d = 123,0 kN ; 4 bandes égales",
        (s["M+"], s["M-"], s["Vd"], s["ecart"]) == (74.7, -134.9, 123.0, 0.0), s)
    chk("T14 plots préfabriqués, zone courante : inf 86,2 / 86,2 ; sup −172,4 / −172,4 kNm/m ; prolongements : inf 128,0 ; sup −186,4",
        s["inf"] == (86.2, 86.2) and s["sup"] == (-172.4, -172.4) and s["prol"] == (128.0, -186.4), s)
    mc, ec = efforts(projet(plots=COULE))
    s = resume(ec)
    chk("T14 plots coulés en place 2,00 × 2,00 × 0,35 : M+ = 144,2 ; M− = −129,5 kNm ; V à d = 102,6 kN ; 4 bandes égales",
        (s["M+"], s["M-"], s["Vd"], s["ecart"]) == (144.2, -129.5, 102.6, 0.0), s)
    chk("T14 plots coulés, zone courante : inf 26,5 / 26,5 ; sup −174,4 / −174,4 ; dans les plots : inf 152,4 / sup −342,1 ; "
        "prolongements : inf 226,2 ; sup −185,1 kNm/m",
        s["inf"] == (26.5, 26.5) and s["sup"] == (-174.4, -174.4) and s["plot"] == (152.4, -342.1) and s["prol"] == (226.2, -185.1), s)
    _, e5 = efforts(projet(largeur=0.5))
    b = e5["bandes"][0]
    v = (r1(b["M_plus"]), r1(b["M_plus"] / 0.5), r1(b["M_moins"]), r1(b["M_moins"] / 0.5), r1(b["V_d"]), r1(b["V_d"] / 0.5))
    chk("T14 bandes de 0,50 m, sans plot : M+ = 71,6 kNm (143,2 kNm/m) ; M− = −63,3 kNm (−126,7 kNm/m) ; V à d = 94,0 kN (187,9 kN/m)",
        v == (71.6, 143.2, -63.3, -126.7, 94.0, 187.9), v)
    print(f"     (les quatre enveloppes en {time.time() - t:.1f} s)")

    print("== 2. T15 — l'intégrale sur la largeur EXACTE, bords interpolés")
    for mm, naif_attendu in ((0.20, 3.200), (0.25, 4.000)):
        d = PL.Dalle(rect(8, 8), H, K, maille=mm)
        f = 3 + d.y0 + (np.arange(d.ndof // 3) // (d.nx + 1)) * d.b
        v = EF.integrer_bande(d, f, (0.0, 1.0), np.array([1.0, 0.0]), -0.5, 0.5, np.array([2.0, 4.1, 7.0]))
        yc = d.y0 + (d.J + 0.5) * d.b
        sel = (d.I == 10) & (np.abs(yc - 1.0) < 0.5)
        naif = float(((3 + yc[sel]) * d.b).sum())
        chk(f"T15 maille {mm:.2f} m : f = 3 + y intégré sur y = 0,50 à 1,50 → 4,000 en s = 2,0 ; 4,1 ; 7,0 (la méthode « centres d'éléments "
            f"dans la bande », interdite, donne {naif_attendu:.3f})",
            [round(float(x), 3) for x in v] == [4.0, 4.0, 4.0] and round(naif, 3) == naif_attendu, (list(v), naif))

    print("== 3. T17 — les zones de renfort (bandes de 1,00 m)")
    z8 = EF.zones_renfort(m0["contour"], m0["bandes"])
    m10 = PR.modele(projet(dalle={"Lx": 10.0, "Ly": 10.0}))
    z10 = EF.zones_renfort(m10["contour"], m10["bandes"])
    arr = lambda z: [(round(a, 2), round(b_, 2)) for a, b_ in z]  # noqa: E731
    chk("T17 dalle 8 × 8 : BX1 t = −0,50 à +1,00 (prolongée jusqu'au bord) ; BX2 −1,00 à +0,50 ; BY1 −0,50 à +1,00 ; BY2 −1,00 à +0,50",
        [x["nom"] for x in m0["bandes"]] == ["BX1", "BX2", "BY1", "BY2"]
        and arr(z8) == [(-0.5, 1.0), (-1.0, 0.5), (-0.5, 1.0), (-1.0, 0.5)], arr(z8))
    chk("T17 dalle 10 × 10 : toutes de −0,50 à +0,50 (la lisière restante, 1,50 m, est plus large que la bande)",
        arr(z10) == [(-0.5, 0.5)] * 4, arr(z10))
    bx1 = m0["bandes"][0]
    y_zone = sorted(bx1["debut"][1] + t_ for t_ in z8[0])
    chk("T17 dalle type : la zone de BX1 va de y = 6,50 à 8,00 m (le bord de la dalle)", [round(y, 2) for y in y_zone] == [6.5, 8.0], y_zone)

    print("== 4. les règles")
    g = EF.geometrie_bandes(m0, None, D_V)[0]
    chk("le pas en s le long d'une bande est maille / 4 = 0,05 m (§ 4.3), de 0 à la longueur de la bande (8,00 m)",
        abs(g["s"][1] - g["s"][0] - 0.05) < 1e-12 and abs(g["s"][-1] - g["L"]) < 1e-9 and round(g["L"], 6) == 8.0, (g["s"][:3], g["L"]))
    chk("V à d : lu à d = 0,44 m ou plus du bord des surfaces d'appui de BX1 (0,75 à 1,25 et 6,75 à 7,25 m) : s ≤ 0,31 ; 1,69 à 6,31 ; ≥ 7,69",
        [tuple(round(c, 3) for c in x) for x in g["charge"]] == [(0.75, 1.25), (6.75, 7.25)]
        and bool(g["ok_v"][np.isclose(g["s"], 0.30)][0]) and not g["ok_v"][np.isclose(g["s"], 0.35)][0]
        and not g["ok_v"][np.isclose(g["s"], 1.65)][0] and bool(g["ok_v"][np.isclose(g["s"], 1.70)][0])
        and bool(g["ok_v"][np.isclose(g["s"], 7.70)][0]) and not g["ok_v"][np.isclose(g["s"], 7.65)][0], g["charge"])
    b0 = e0["bandes"][0]
    v30, v60 = EF.v_a_d(b0, 0.30), EF.v_a_d(b0, 0.60)
    chk(f"V à d se RELIT pour un autre d (celui que le moteur poutre donne pour la bande, § 4.3) : d = 0,44 → {r1(b0['V_d'])} kN (T14) ; "
        f"d = 0,30 → {r1(v30)} ; d = 0,60 → {r1(v60)} kN (plus d est grand, plus on lit loin du pied) ; sans les abscisses à d, le pic "
        f"{r1(b0['Vabs'].max())} kN serait lu",
        abs(EF.v_a_d(b0, D_V) - b0["V_d"]) < 1e-12 and v30 >= b0["V_d"] >= v60 and v30 > v60 and b0["Vabs"].max() > v30 + 1, (v30, v60))
    zp = EF.zones_chargees(PR.modele(projet(plots=PREF)))
    zc = EF.zones_chargees(mc)
    cote = lambda q: round(max(p[0] for p in q) - min(p[0] for p in q), 3)  # noqa: E731
    chk("la zone chargée d'un pied (V à d) : la surface d'appui sans plot (0,50 m) ; le carré de diffusion sous un plot préfabriqué "
        "(0,50 + 2 × 0,35 = 1,20 m) ; le plot coulé en place (2,00 m)",
        [cote(q) for q in EF.zones_chargees(m0)] == [0.5] * 4 and [cote(q) for q in zp] == [1.2] * 4 and [cote(q) for q in zc] == [2.0] * 4,
        ([cote(q) for q in zp], [cote(q) for q in zc]))
    bc = EF.geometrie_bandes(mc, None, D_V)[0]
    chk("plots coulés : V se lit à d du bord du PLOT (0,00 à 2,00 m le long de BX1 → s ≥ 2,44 m entre les pieds)",
        [tuple(round(c, 3) for c in x) for x in bc["charge"]] == [(0.0, 2.0), (6.0, 8.0)]
        and not bc["ok_v"][np.isclose(bc["s"], 2.40)][0] and bool(bc["ok_v"][np.isclose(bc["s"], 2.45)][0]), bc["charge"])
    ch = ec["champ"]
    chk("plots coulés : les éléments sous un plot ont l'épaisseur h + 0,35 = 0,85 m, les autres 0,50 m (champ des moments au centre)",
        set(np.round(ch["h"][ch["plot"]], 6)) == {0.85} and set(np.round(ch["h"][~ch["plot"]], 6)) == {0.5} and ch["plot"].sum() == 4 * 100,
        (int(ch["plot"].sum()), set(np.round(ch["h"], 3))))
    # l'enveloppe des k : chaque k seul est sous l'enveloppe, et l'enveloppe EST le plus grand (le plus petit) des k
    seuls = [EF.enveloppe(m0, [V.calculer(m0, k, k / 10, k / 10)], D_V) for k in KS]
    chk("l'enveloppe des k = 2 ; 5 ; 10 est le plus grand (le plus petit) des trois calculs séparés, en chaque abscisse et chaque élément",
        np.allclose(e0["bandes"][0]["Mmax"], np.max([x["bandes"][0]["Mmax"] for x in seuls], 0))
        and np.allclose(e0["bandes"][0]["Mmin"], np.min([x["bandes"][0]["Mmin"] for x in seuls], 0))
        and np.allclose(e0["bandes"][2]["Vabs"], np.max([x["bandes"][2]["Vabs"] for x in seuls], 0))
        and np.allclose(e0["champ"]["inf_x"], np.max([x["champ"]["inf_x"] for x in seuls], 0))
        and np.allclose(e0["champ"]["sup_y"], np.min([x["champ"]["sup_y"] for x in seuls], 0)) and e0["ks"] == list(KS),
        e0["ks"])
    chk("… et elle diffère de chaque k seul (les trois k comptent : M+ de 127,1 ≠ " + " ; ".join(f"{r1(x['bandes'][0]['M_plus'])}" for x in seuls) + ")",
        len({r1(x["bandes"][0]["M_plus"]) for x in seuls}) == 3 and r1(e0["bandes"][0]["M_plus"]) == max(r1(x["bandes"][0]["M_plus"]) for x in seuls))
    _, en = efforts(projet(), torsion=False)
    cn, ch0 = en["courante"], e0["champ"]
    cour = ~ch0["zone_renfort"]
    chk("option « torsion » non : les moments de calcul du treillis sont mx, my sans |mxy| (zone courante = max de mx hors zones de "
        f"renfort : {r1(cn['inf_x'])} kNm/m, contre 92,2 avec la torsion)",
        abs(cn["inf_x"] - float(ch0["mx_max"][cour].max())) < 1e-9 and abs(cn["sup_y"] - float(ch0["my_min"][cour].min())) < 1e-9
        and cn["inf_x"] < 92.2 - 1 and np.allclose(en["champ"]["inf_x"], ch0["mx_max"]) and en["torsion"] is False,
        (cn["inf_x"], float(ch0["mx_max"][cour].max())))
    chk("… les bandes ne dépendent pas de l'option (M, V intégrés sans la torsion)",
        np.allclose(en["bandes"][0]["Mmax"], e0["bandes"][0]["Mmax"]) and np.allclose(en["bandes"][0]["Vabs"], e0["bandes"][0]["Vabs"]))
    chk("avec la torsion : inf = m + |mxy| et sup = m − |mxy| en chaque élément (§ 4.4 : en bas mx + |mxy|, en haut mx − |mxy|)",
        bool(np.all(ch0["inf_x"] >= ch0["mx_max"] - 1e-9)) and bool(np.all(ch0["sup_x"] <= ch0["mx_min"] + 1e-9))
        and float((ch0["inf_x"] - ch0["mx_max"]).max()) > 50.0, float((ch0["inf_x"] - ch0["mx_max"]).max()))
    M = np.array([[1.0, 2.0, 3.0]])
    u30 = np.array([np.cos(np.pi / 6), np.sin(np.pi / 6)])
    muu, mnn, mun = EF.rot_moments(M, u30)
    chk("la rotation des moments (§ 4.3 : m_uu = mx·cos² + my·sin² + 2·mxy·sin·cos) : mx = 1, my = 2, mxy = 3 à 30° → m_uu = 3,848 ; "
        "m_nn = −0,848 ; m_un = 1,933 ; la trace se conserve",
        (round(float(muu[0]), 3), round(float(mnn[0]), 3), round(float(mun[0]), 3)) == (3.848, -0.848, 1.933)
        and abs(float(muu[0] + mnn[0]) - 3.0) < 1e-12, (muu, mnn, mun))
    d = PL.Dalle(rect(4, 4), H, K, maille=0.25)
    w = np.zeros(d.ndof)
    xn = d.x0 + (np.arange(d.ndof // 3) % (d.nx + 1)) * d.a
    w[0::3], w[1::3] = 1e-3 * xn ** 2, 2e-3 * xn                   # les degrés de liberté d'un nœud : w, ∂w/∂x, ∂w/∂y
    Mn, Vn = EF.champs_noeuds(d, w)
    w2 = np.zeros(d.ndof)
    w2[0::3], w2[1::3] = 1e-3 * xn ** 3, 3e-3 * xn ** 2
    _, Vn3 = EF.champs_noeuds(d, w2)
    Dp = 32800e3 * H ** 3 / (12 * (1 - 0.04))
    chk("les champs aux nœuds sur une flexion cylindrique w = 10⁻³ x² : mx = −D·w,xx = −2·10⁻³ D, my = ν·mx, mxy = 0, vx = 0 ; "
        "w = 10⁻³ x³ : vx = −D·w,xxx = −6·10⁻³ D (convention : m > 0 fibre inférieure tendue)",
        _cylindrique(Mn, Vn, Vn3, Dp), (Mn[d.nx // 2], Vn[d.nx // 2], Vn3[d.nx // 2], Dp))


def _cylindrique(Mn, Vn, Vn3, Dp):
    """La flexion cylindrique, en TOUS les nœuds (bords et coins compris : la moyenne de valeurs exactes est exacte)."""
    mx = Mn[:, 0]
    return (np.allclose(mx, -2e-3 * Dp, rtol=1e-6) and np.allclose(Mn[:, 1], 0.2 * mx, rtol=1e-6) and np.allclose(Mn[:, 2], 0, atol=1e-6 * Dp)
            and np.allclose(Vn[:, 0], 0, atol=1e-6 * Dp) and np.allclose(Vn3[:, 0], -6e-3 * Dp, rtol=1e-6))


# ------------------------------------------------------------------ les contrôles COMPLETS
def complets():
    import fondation_grue_ref as REF
    print("== 5. la parité avec le calcul de référence (tests/fondation_grue_ref.py : efforts, numpy + scipy)")

    def paire(nom, el, ref, ks=KS):
        _, mien = efforts(el, ks)
        pire = 0.0
        meme = len(mien["bandes"]) == len(ref["bandes"])
        for a, b in zip(mien["bandes"], ref["bandes"]):
            meme = meme and a["nom"] == b["nom"] and len(a["s"]) == len(b["s"]) and bool(np.array_equal(a["ok_v"], b["ok_v"]))
            meme = meme and np.allclose(a["zone"], b["zone"], atol=1e-12)
            for q in ("Mmax", "Mmin", "Vabs"):
                pire = max(pire, float(np.abs(a[q] - b[q]).max()))
            for q in ("M_plus", "M_moins", "V_d", "prol_inf", "prol_sup"):
                pire = max(pire, abs(a[q] - b[q]))
        for q in ref["courante"]:
            pire = max(pire, abs(mien["courante"][q] - ref["courante"][q]))
        for q in ("inf_x", "inf_y", "sup_x", "sup_y", "mx_max", "my_max", "mx_min", "my_min", "h", "x", "y"):
            pire = max(pire, float(np.abs(mien["champ"][q] - ref["champ"][q]).max()))
        meme = meme and bool(np.array_equal(mien["champ"]["zone_renfort"], ref["champ"]["zone_renfort"]))
        meme = meme and bool(np.array_equal(mien["champ"]["plot"], ref["champ"]["plot"]))
        chk(f"parité · {nom} : 4 bandes, M(s), V(s), abscisses lues, zones, zone courante, champs aux centres — plus grand écart "
            f"{pire:.1e} (kNm, kN)", meme and pire < 1e-6, (meme, pire))

    t = time.time()
    paire("dalle type sans plot", projet(), REF.efforts(rect(8, 8), H, KS, (6, 6), (0.5, 0.5), JEUX, d_v=D_V))
    paire("plots préfabriqués", projet(plots=PREF), REF.efforts(rect(8, 8), H, KS, (6, 6), (0.5, 0.5), JEUX, plots=("prefabrique", 2.0, 2.0, 0.35), d_v=D_V))
    paire("plots coulés en place", projet(plots=COULE), REF.efforts(rect(8, 8), H, KS, (6, 6), (0.5, 0.5), JEUX, plots=("coule", 2.0, 2.0, 0.35), d_v=D_V))
    paire("bandes de 0,50 m", projet(largeur=0.5), REF.efforts(rect(8, 8), H, KS, (6, 6), (0.5, 0.5), JEUX, largeur=0.5, d_v=D_V))
    paire("grue décalée de + 0,25 m en x et en y", projet(grue={"ex": 0.25, "ey": 0.25}),
          REF.efforts(rect(8, 8), H, KS, (6, 6), (0.5, 0.5), JEUX, centre=(4.25, 4.25), d_v=D_V))
    t7 = projet(dalle={"forme": "contour", "sommets": [list(p) for p in L_T7]},
                grue={"a": 4.5, "b": 4.5, "rect": {"a": 0.6, "b": 0.6, "angle": None}, "ex": -1.8, "ey": -1.8, "beta": 30.0})
    paire("dalle en L, châssis tourné de 30°, k = 5 ; 12", t7,
          REF.efforts(L_T7, H, (5.0, 12.0), (4.5, 4.5), (0.6, 0.6), JEUX, centre=(4.2, 4.2), beta=30.0, d_v=D_V), ks=(5.0, 12.0))
    print(f"     (six parités en {time.time() - t:.1f} s)")


# ------------------------------------------------------------------ mutations (arbre miroir, contrôles rapides)
MUTATIONS = [
    ("efforts.py", "    t = np.linspace(t0, t1, nt)", "    t = np.linspace(t0 + min(d.a, d.b) / 2, t1 - min(d.a, d.b) / 2, nt)",
     "l'intégrale sur les seuls éléments entiers de la bande (la méthode interdite : 20 % perdus)"),
    ("efforts.py", "    r = (1 - u) * (1 - v) * val[n00] + u * (1 - v) * val[n00 + 1] + u * v * val[n00 + 1 + nnx] + (1 - u) * v * val[n00 + nnx]",
     "    r = val[n00]", "les bords de la bande non interpolés (le nœud voisin)"),
    ("efforts.py", "    muu = mx * c * c + my * s_ * s_ + 2 * mxy * c * s_", "    muu = mx * c * c + my * s_ * s_", "la rotation des moments sans mxy"),
    ("efforts.py", "                    t_ = np.abs(mun) if torsion else 0.0", "                    t_ = 0.0", "les prolongements sans la torsion"),
    ("efforts.py", "            ax = np.abs(Me[:, 2]) if torsion else np.zeros(len(Me))", "            ax = np.zeros(len(Me))", "le treillis sans la torsion"),
    ("efforts.py", "            ax = np.abs(Me[:, 2]) if torsion else np.zeros(len(Me))", "            ax = np.abs(Me[:, 2])", "l'option « torsion : non » ignorée"),
    ("efforts.py", '    if pl and pl["type"] == "coule":\n        return pl["polys"]', '    if pl and pl["type"] == "coule":\n        return modele["plaques"]',
     "V lu à d de la platine sous un plot coulé (au lieu du plot)"),
    ("efforts.py", '        return pl["diffusion"]\n', '        return pl["polys"]\n', "V lu à d du plot préfabriqué (au lieu du carré de diffusion)"),
    ("efforts.py", "    return np.all([(s <= s1 - d_v + 1e-9) | (s >= s2 + d_v - 1e-9) for s1, s2 in charge], axis=0)",
     "    return np.all([(s <= s1 + 1e-9) | (s >= s2 - 1e-9) for s1, s2 in charge], axis=0)", "V lu au bord de la zone chargée (et non à d)"),
    ("efforts.py", '    return float(bande["Vabs"][ok].max()) if ok.any() else 0.0', '    return float(bande["Vabs"].max())',
     "V à d relu sans les abscisses à d (le pic sous le pied)"),
    ("efforts.py", "            lim.append(sg * (b / 2 + e) if e < b - 1e-9 else sg * b / 2)", "            lim.append(sg * b / 2)",
     "la zone de renfort jamais prolongée jusqu'au bord"),
    ("efforts.py", "            lim.append(sg * (b / 2 + e) if e < b - 1e-9 else sg * b / 2)", "            lim.append(sg * (b / 2 + e) if e < b / 2 else sg * b / 2)",
     "la lisière comparée à une demi-bande"),
    ("efforts.py", "PAS_S = 0.25 ", "PAS_S = 0.5 ", "le pas en s de maille / 2"),
    ("efforts.py", "    for r in resultats:\n", "    for r in resultats[-1:]:\n", "l'enveloppe réduite au dernier k"),
    ("efforts.py", "    vx = -d.De * (ue @ ((pxxx + pxyy) @ d.Ci))", "    vx = -d.De * (ue @ (pxxx @ d.Ci))", "vx sans le terme w,xyy"),
    ("efforts.py", "    return np.stack([-d.De * (c[:, 0] + d.nu * c[:, 1]), -d.De * (c[:, 1] + d.nu * c[:, 0]), -d.De * (1 - d.nu) * c[:, 2]], 1)",
     "    return np.stack([-d.De * c[:, 0], -d.De * c[:, 1], -d.De * (1 - d.nu) * c[:, 2]], 1)", "les moments au centre sans l'effet de Poisson"),
    ("efforts.py", '"h": modele["h"] + np.where(dans_plot, pl["hauteur"] if dans_plot.any() else 0.0, 0.0)', '"h": modele["h"] + np.zeros(len(xc))',
     "l'épaisseur du plot coulé oubliée dans le champ"),
    ("efforts.py", "            prol.append(dz & (np.abs(T) > g[\"bande\"][\"largeur\"] / 2 + 1e-6))", "            prol.append(dz & (np.abs(T) > 1e9))",
     "les prolongements jamais lus"),
]
TEMOIN = ("efforts.py", "PAS_S = 0.25 ", "PAS_S = 0.25 ", "témoin : rien ne change")


def muter(fichier, ancien, nouveau):
    """Les contrôles rapides rejoués sur une copie du moteur où ``ancien`` est remplacé : (texte trouvé, code de sortie)."""
    with tempfile.TemporaryDirectory() as tmp:
        for paquet in ("saf", "grues"):
            shutil.copytree(os.path.join(RACINE, paquet), os.path.join(tmp, paquet), ignore=shutil.ignore_patterns("__pycache__"))
        os.symlink(os.path.join(RACINE, "beton"), os.path.join(tmp, "beton"))
        os.symlink(os.path.join(RACINE, "beton_classes.json"), os.path.join(tmp, "beton_classes.json"))
        cible = os.path.join(tmp, "saf", "fondation_grue", fichier)
        src = open(cible, encoding="utf-8").read()
        if src.count(ancien) != 1:
            return False, None
        open(cible, "w", encoding="utf-8").write(src.replace(ancien, nouveau))
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--rapide", "--racine", tmp], capture_output=True, text=True, cwd=RACINE)
        return True, p.returncode


def main():
    rapides()
    if RAPIDE:
        print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
        return 1 if KO else 0
    complets()
    print("== 6. mutations")
    trouve, code = muter(*TEMOIN[:3])
    chk("témoin : le moteur recopié sans rien changer passe les contrôles rapides", trouve and code == 0, code)
    for fichier, ancien, nouveau, nom in MUTATIONS:
        trouve, code = muter(fichier, ancien, nouveau)
        chk(f"mutation « {nom} » ({fichier}) : un contrôle rougit", trouve and code == 1,
            "texte absent ou répété" if not trouve else f"code {code} : AUCUN contrôle ne rougit")
    print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
    for nom, info in KO:
        print("   -", nom[:160], "|", str(info)[:300])
    return 1 if KO else 0


if __name__ == "__main__":
    sys.exit(main())

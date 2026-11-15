# -*- coding: utf-8 -*-
"""Générateur SAF — fondation de grue : les EFFORTS ELS pour le principe d'armatures (saf/fondation_grue/efforts.py).
Cahier des charges du lot 5, version 4 (15/11/2026), § 7, § 13 et annexe A : les tests T14 et T15 (T17 retiré avec les zones de
renfort du lot 4).

Ce que la suite prouve :
  1. T14, T15 — les valeurs du cahier (sortie de l'annexe A) : moments et efforts tranchants intégrés sur les bandes de 0,50 et
     1,00 m, avec et sans plots, le cas et le k qui donnent chaque valeur, les moments au point les plus forts ; l'intégration
     sur la largeur EXACTE.
  2. LES RÈGLES — le pas en s (maille / 4), l'enveloppe des k, V lu à d du bord des zones chargées, la zone chargée de chaque
     type de plot, M+ lu hors des plots coulés, leurs nœuds exclus des extrêmes, PAS DE TORSION, la rotation des moments.
  3. LA PARITÉ AVEC LE CALCUL DE RÉFÉRENCE — tests/fondation_grue_ref.py (``efforts``, numpy + scipy), indépendant : mêmes
     M(s), V(s), mêmes abscisses lues, mêmes cas dimensionnants, mêmes moments aux nœuds, cas par cas (dalle type avec et sans
     plots, bandes de 1,00 m, dalle en L avec châssis tourné de 30°, grue décalée).
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


def projet(maille=0.20, largeur=0.5, plots=None, jeux=JEUX, **k):
    """La dalle type du cahier (8 × 8 × 0,50 m, châssis 6 × 6 m centré, appuis 0,5 × 0,5 m, 3 jeux), avec la maille et la
    largeur de bande ÉCRITES dans T14 (0,20 m ; 0,50 m), modifiée par ``k`` (dalle, grue… fusionnés)."""
    el = PR.nouveau()
    el["options"]["maille"], el["bandes"]["largeur"] = maille, largeur
    el["reactions"] = [{"nom": "", "P": list(j)} for j in jeux]
    if plots:
        el["grue"]["plots"] = dict(plots)
    for cle, v in k.items():
        el[cle].update(v)
    return PR.lire(el)


def efforts(el, ks=KS, d_v=D_V):
    m = PR.modele(el)
    return m, EF.enveloppe(m, [V.calculer(m, k, k / 10, k / 10) for k in ks], d_v)


def r1(v):
    return round(float(v), 1)


def resume(ef):
    """Les valeurs imprimées par l'annexe A pour T14 : la bande BX1, l'écart entre les 4 bandes, les cas dimensionnants, les
    moments au point les plus forts."""
    B, e = ef["bandes"], ef["extremes"]
    ecart = max(max(x[q] for x in B) - min(x[q] for x in B) for q in ("M_plus", "M_moins", "V_d"))
    return {"M+": r1(B[0]["M_plus"]), "M-": r1(B[0]["M_moins"]), "Vd": r1(B[0]["V_d"]), "ecart": round(ecart, 2),
            "cas": (B[0]["cas_M_plus"], B[0]["cas_M_moins"], B[0]["cas_V_d"]),
            "plot": (r1(B[0]["M_plus_plot"]), r1(B[0]["M_moins_plot"])) if B[0]["M_plus_plot"] is not None else None,
            "mx": (r1(e["inf_x"]), r1(e["sup_x"])), "my": (r1(e["inf_y"]), r1(e["sup_y"]))}


# ------------------------------------------------------------------ les contrôles RAPIDES (rejoués par les mutations)
def rapides():
    print("== 1. T14 — efforts ELS, dalle type, k = 2 ; 5 ; 10, C2 = k/10, maille 0,20, d = 0,44 m (sortie de l'annexe A)")
    t = time.time()
    m0, e0 = efforts(projet())
    s = resume(e0)
    # (le cas dimensionnant : à égalité entre deux positions symétriques — 3.b et 3.g, 3.c et 3.h —, le PREMIER dans l'ordre de
    # l'annexe A, jeu par jeu : exactement les cas imprimés par l'annexe)
    def cas_ok(s_, attendus):
        return all(c is not None and c[0] == a and c[1] == 2.0 for c, a in zip(s_["cas"], attendus))
    chk("T14 bandes de 0,50 m, sans plot : M+ = 71,6 kNm ; M− = −63,3 kNm ; V à d = 94,0 kN ; les 4 bandes égales (écart 0,00) ; cas "
        "dimensionnants de BX1 : M+ 3.b, M− 2.c, V 3.c, tous à k = 2 (ceux de l'annexe) ; mx = my = 153,1 / −140,8 kNm/m au point",
        (s["M+"], s["M-"], s["Vd"], s["ecart"]) == (71.6, -63.3, 94.0, 0.0) and cas_ok(s, ("3.b", "2.c", "3.c"))
        and s["mx"] == (153.1, -140.8) and s["my"] == (153.1, -140.8) and s["plot"] is None, s)
    b = e0["bandes"][0]
    v = (r1(b["M_plus"] / 0.5), r1(b["M_moins"] / 0.5), r1(b["V_d"] / 0.5))
    chk("T14 … par mètre : 143,2 kNm/m ; −126,7 kNm/m ; 187,9 kN/m", v == (143.2, -126.7, 187.9), v)
    _, ep = efforts(projet(plots=PREF))
    s = resume(ep)
    chk("T14 plots préfabriqués 2,00 × 2,00 × 0,35 : M+ = 38,6 ; M− = −67,5 kNm ; V à d = 63,6 kN ; 4 bandes égales ; M+ 3.b, M− 2.c, "
        "V 3.c (k = 2) ; mx = 83,0 / −150,3 kNm/m", (s["M+"], s["M-"], s["Vd"], s["ecart"]) == (38.6, -67.5, 63.6, 0.0)
        and cas_ok(s, ("3.b", "2.c", "3.c")) and s["mx"] == (83.0, -150.3), s)
    mc, ec = efforts(projet(plots=COULE))
    s = resume(ec)
    chk("T14 plots coulés en place : M+ = 8,2 ; M− = −64,7 kNm ; V à d = 51,2 kN au nu du plot ; dans le plot (information) 46,4 / −42,0 ; "
        "4 bandes égales ; M+ 3.g, M− 2.c, V 3.c ; mx = 199,6 / −169,0 kNm/m (hors des plots)",
        (s["M+"], s["M-"], s["Vd"], s["ecart"]) == (8.2, -64.7, 51.2, 0.0) and s["plot"] == (46.4, -42.0)
        and cas_ok(s, ("3.g", "2.c", "3.c")) and s["mx"] == (199.6, -169.0), s)
    _, e1 = efforts(projet(largeur=1.0))
    s = resume(e1)
    chk("T14 bandes de 1,00 m, sans plot : M+ = 127,1 ; M− = −126,5 kNm ; V à d = 172,8 kN ; mx = 153,1 / −140,8 (les moments au point ne "
        "dépendent pas de la largeur)", (s["M+"], s["M-"], s["Vd"], s["ecart"]) == (127.1, -126.5, 172.8, 0.0) and s["mx"] == (153.1, -140.8), s)
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

    print("== 3. les règles")
    g = EF.geometrie_bandes(m0, D_V)[0]
    chk("le pas en s le long d'une bande est maille / 4 = 0,05 m, de 0 à la longueur de la bande (8,00 m)",
        abs(g["s"][1] - g["s"][0] - 0.05) < 1e-12 and abs(g["s"][-1] - g["L"]) < 1e-9 and round(g["L"], 6) == 8.0, (g["s"][:3], g["L"]))
    chk("V à d : lu à d = 0,44 m ou plus du bord des surfaces d'appui de BX1 (0,75 à 1,25 et 6,75 à 7,25 m) : s ≤ 0,31 ; 1,69 à 6,31 ; ≥ 7,69",
        [tuple(round(c, 3) for c in x) for x in g["charge"]] == [(0.75, 1.25), (6.75, 7.25)]
        and bool(g["ok_v"][np.isclose(g["s"], 0.30)][0]) and not g["ok_v"][np.isclose(g["s"], 0.35)][0]
        and not g["ok_v"][np.isclose(g["s"], 1.65)][0] and bool(g["ok_v"][np.isclose(g["s"], 1.70)][0])
        and bool(g["ok_v"][np.isclose(g["s"], 7.70)][0]) and not g["ok_v"][np.isclose(g["s"], 7.65)][0], g["charge"])
    b0 = e0["bandes"][0]
    chk(f"V à d est le plus grand |V| aux seules abscisses à d ({r1(b0['V_d'])} kN), pas le pic sous le pied ({r1(b0['Vabs'].max())} kN)",
        abs(b0["V_d"] - float(b0["Vabs"][b0["ok_v"]].max())) < 1e-12 and b0["Vabs"].max() > b0["V_d"] + 1, (b0["V_d"], b0["Vabs"].max()))
    zp = EF.zones_chargees(PR.modele(projet(plots=PREF)))
    zc = EF.zones_chargees(mc)
    cote = lambda q: round(max(p[0] for p in q) - min(p[0] for p in q), 3)  # noqa: E731
    chk("la zone chargée d'un pied (V à d) : la surface d'appui sans plot (0,50 m) ; le carré de diffusion sous un plot préfabriqué "
        "(0,50 + 2 × 0,35 = 1,20 m) ; le plot coulé en place (2,00 m)",
        [cote(q) for q in EF.zones_chargees(m0)] == [0.5] * 4 and [cote(q) for q in zp] == [1.2] * 4 and [cote(q) for q in zc] == [2.0] * 4,
        ([cote(q) for q in zp], [cote(q) for q in zc]))
    bc = EF.geometrie_bandes(mc, D_V)[0]
    chk("plots coulés : V se lit à d du bord du PLOT (0,00 à 2,00 m le long de BX1 → s ≥ 2,44 m entre les pieds) ; l'axe de la bande est "
        "« dans le plot » de 0 à 2,00 m et de 6,00 à 8,00 m (M+ et M− lus hors de ces tronçons)",
        [tuple(round(c, 3) for c in x) for x in bc["charge"]] == [(0.0, 2.0), (6.0, 8.0)]
        and not bc["ok_v"][np.isclose(bc["s"], 2.40)][0] and bool(bc["ok_v"][np.isclose(bc["s"], 2.45)][0])
        and bool(bc["dans_plot"][np.isclose(bc["s"], 1.0)][0]) and not bc["dans_plot"][np.isclose(bc["s"], 4.0)][0], bc["charge"])
    bec = ec["bandes"][0]
    chk("… M+ = le plus grand M(s) HORS des plots coulés (le nu du plot), M+ dans le plot = celui des tronçons dans le plot",
        abs(bec["M_plus"] - float(bec["Mmax"][~bec["dans_plot"]].max())) < 1e-12
        and abs(bec["M_plus_plot"] - float(bec["Mmax"][bec["dans_plot"]].max())) < 1e-12 and bec["M_plus_plot"] > bec["M_plus"] + 10)
    a = ec["grille"]["actif"]
    chk("plots coulés : les nœuds STRICTEMENT dans un plot sont exclus des extrêmes (4 × 9 × 9 = 324 nœuds : le plot de 2,00 m, maille "
        "0,20, son bord gardé)", int((~a).sum()) == 4 * 9 * 9, int((~a).sum()))
    # l'enveloppe des k : chaque k seul est sous l'enveloppe, et l'enveloppe EST le plus grand (le plus petit) des k
    seuls = [EF.enveloppe(m0, [V.calculer(m0, k, k / 10, k / 10)], D_V) for k in KS]
    chk("l'enveloppe des k = 2 ; 5 ; 10 est le plus grand (le plus petit) des trois calculs séparés, en chaque abscisse et chaque nœud",
        np.allclose(e0["bandes"][0]["Mmax"], np.max([x["bandes"][0]["Mmax"] for x in seuls], 0))
        and np.allclose(e0["bandes"][0]["Mmin"], np.min([x["bandes"][0]["Mmin"] for x in seuls], 0))
        and np.allclose(e0["bandes"][2]["Vabs"], np.max([x["bandes"][2]["Vabs"] for x in seuls], 0))
        and np.allclose(e0["noeuds"]["inf_x"], np.max([x["noeuds"]["inf_x"] for x in seuls], 0))
        and np.allclose(e0["noeuds"]["sup_y"], np.min([x["noeuds"]["sup_y"] for x in seuls], 0)) and e0["ks"] == list(KS), e0["ks"])
    chk("… et elle diffère de chaque k seul (les trois k comptent : M+ de 71,6 ≠ " + " ; ".join(f"{r1(x['bandes'][0]['M_plus'])}" for x in seuls) + ")",
        len({r1(x["bandes"][0]["M_plus"]) for x in seuls}) == 3 and r1(e0["bandes"][0]["M_plus"]) == max(r1(x["bandes"][0]["M_plus"]) for x in seuls))
    # sans torsion (lot 5, § 2) : les moments au point sont mx, my aux nœuds — le champ de champs_noeuds, sans |mxy|
    r10 = V.calculer(m0, 10.0, 1.0, 1.0)
    mx_n = np.max([EF.champs_noeuds(r10["dalle"], x)[0][:, 0] for x in r10["deplacements"]], 0).reshape(e0["grille"]["ny"], e0["grille"]["nx"])
    e10 = EF.enveloppe(m0, [r10], D_V)
    chk("PAS DE TORSION : les moments au point d'un k sont mx (my) aux nœuds, en enveloppe des cas — aucun |mxy| ajouté",
        np.allclose(e10["noeuds"]["inf_x"], mx_n) and "torsion" not in e10 and "champ" not in e10 and "courante" not in e10)
    M = np.array([[1.0, 2.0, 3.0]])
    u30 = np.array([np.cos(np.pi / 6), np.sin(np.pi / 6)])
    muu, mnn, mun = EF.rot_moments(M, u30)
    chk("la rotation des moments (m_uu = mx·cos² + my·sin² + 2·mxy·sin·cos) : mx = 1, my = 2, mxy = 3 à 30° → m_uu = 3,848 ; "
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
    print("== 4. la parité avec le calcul de référence (tests/fondation_grue_ref.py : efforts, numpy + scipy)")

    def paire(nom, el, ref, ks=KS):
        m_, mien = efforts(el, ks)
        pire = 0.0
        meme = len(mien["bandes"]) == len(ref["bandes"])
        resultats = {k: V.calculer(m_, k, k / 10, k / 10) for k in ks}

        def valeur_du_cas(i_bande, cle, cas):
            """La valeur que le MODULE donne pour le cas (nom, k) retenu par la référence : deux positions symétriques sont
            à égalité au bruit près, et chaque calcul les départage par SON bruit — le cas de la référence doit être, dans le
            module aussi, un maximum (à 10⁻⁶ près)."""
            r = resultats[cas[1]]
            seul = {"k": r["k"], "dalle": r["dalle"], "cas": [c for c in r["cas"] if c["nom"] == cas[0]],
                    "deplacements": [x for c, x in zip(r["cas"], r["deplacements"]) if c["nom"] == cas[0]]}
            return EF.enveloppe(m_, [seul], D_V)["bandes"][i_bande][cle]
        egalites = 0
        for i, (a, b) in enumerate(zip(mien["bandes"], ref["bandes"])):
            meme = meme and a["nom"] == b["nom"] and len(a["s"]) == len(b["s"]) and bool(np.array_equal(a["ok_v"], b["ok_v"]))
            meme = meme and bool(np.array_equal(a["dans_plot"], b["dans_plot"]))
            for c, cle in (("cas_M_plus", "M_plus"), ("cas_M_moins", "M_moins"), ("cas_V_d", "V_d")):
                if a[c] is None or tuple(a[c]) != tuple(b[c]):
                    egalites += 1
                    meme = meme and a[c] is not None and abs(valeur_du_cas(i, cle, b[c]) - a[cle]) < 1e-6 * max(1.0, abs(a[cle]))
            for q in ("Mmax", "Mmin", "Vabs"):
                pire = max(pire, float(np.abs(a[q] - b[q]).max()))
            for q in ("M_plus", "M_moins", "V_d"):
                pire = max(pire, abs(a[q] - b[q]))
            for q in ("M_plus_plot", "M_moins_plot"):
                meme = meme and ((a[q] is None) == (b[q] is None))
                if a[q] is not None:
                    pire = max(pire, abs(a[q] - b[q]))
        for q in ("inf_x", "inf_y", "sup_x", "sup_y"):
            pire = max(pire, float(np.abs(mien["noeuds"][q] - ref["noeuds"][q]).max()), abs(mien["extremes"][q] - ref["extremes"][q]))
        meme = meme and bool(np.array_equal(mien["grille"]["actif"], ref["grille"]["actif"]))
        chk(f"parité · {nom} : 4 bandes, M(s), V(s), abscisses lues, tronçons dans les plots, cas dimensionnants (les mêmes, ou à égalité "
            f"dans le module : {egalites}), moments au point, extrêmes — plus grand écart {pire:.1e} (kNm, kN)", meme and pire < 1e-6, (meme, pire))

    t = time.time()
    paire("dalle type sans plot", projet(), REF.efforts(rect(8, 8), H, KS, (6, 6), (0.5, 0.5), JEUX, d_v=D_V))
    paire("plots préfabriqués", projet(plots=PREF), REF.efforts(rect(8, 8), H, KS, (6, 6), (0.5, 0.5), JEUX, plots=("prefabrique", 2.0, 2.0, 0.35), d_v=D_V))
    paire("plots coulés en place", projet(plots=COULE), REF.efforts(rect(8, 8), H, KS, (6, 6), (0.5, 0.5), JEUX, plots=("coule", 2.0, 2.0, 0.35), d_v=D_V))
    paire("bandes de 1,00 m", projet(largeur=1.0), REF.efforts(rect(8, 8), H, KS, (6, 6), (0.5, 0.5), JEUX, largeur=1.0, d_v=D_V))
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
    ("efforts.py", '    if pl and pl["type"] == "coule":\n        return pl["polys"]', '    if pl and pl["type"] == "coule":\n        return modele["plaques"]',
     "V lu à d de la platine sous un plot coulé (au lieu du plot)"),
    ("efforts.py", '    if pl:\n        return pl["diffusion"]\n', '    if pl:\n        return pl["polys"]\n', "V lu à d du plot préfabriqué (au lieu du carré de diffusion)"),
    ("efforts.py", "    return np.all([(s <= s1 - d_v + 1e-9) | (s >= s2 + d_v - 1e-9) for s1, s2 in charge], axis=0)",
     "    return np.all([(s <= s1 + 1e-9) | (s >= s2 - 1e-9) for s1, s2 in charge], axis=0)", "V lu au bord de la zone chargée (et non à d)"),
    ("efforts.py", '        rb["V_d"] = float(rb["Vabs"][rb["ok_v"]].max()) if rb["ok_v"].any() else None', '        rb["V_d"] = float(rb["Vabs"].max())',
     "V à d lu sans les abscisses à d (le pic sous le pied)"),
    ("efforts.py", '        rb["M_plus"] = float(rb["Mmax"][hp].max()) if hp.any() else None', '        rb["M_plus"] = float(rb["Mmax"].max())',
     "M+ lu dans le plot coulé (la section plus haute)"),
    ("efforts.py", "PAS_S = 0.25 ", "PAS_S = 0.5 ", "le pas en s de maille / 2"),
    ("efforts.py", "    for r in resultats:\n", "    for r in resultats[-1:]:\n", "l'enveloppe réduite au dernier k"),
    ("efforts.py", "    vx = -d.De * (ue @ ((pxxx + pxyy) @ d.Ci))", "    vx = -d.De * (ue @ (pxxx @ d.Ci))", "vx sans le terme w,xyy"),
    ("efforts.py", '            no["inf_x"] = np.maximum(no["inf_x"], mx)', '            no["inf_x"] = np.maximum(no["inf_x"], mx + np.abs(M[:, 2].reshape(nny, nnx)))',
     "la torsion remise dans les moments au point (m + |mxy|)"),
    ("efforts.py", '            actif &= ~dans_polygone(X, Y, G.rectangle(*G.centre_gravite(q), pl["a"] - 1e-6, pl["b"] - 1e-6, modele.get("beta", 0.0)))',
     "            pass", "les nœuds des plots coulés gardés dans les extrêmes"),
    ("efforts.py", "    if (v > ref + tol) if plus else (v < ref - tol):", "    if (v >= ref - tol) if plus else (v <= ref + tol):",
     "le cas dimensionnant : le DERNIER à égalité"),
    ("efforts.py", "EGALITE_CAS = 1e-6 ", "EGALITE_CAS = 0.0 ", "le cas dimensionnant départagé par le bruit du calcul (« 3.h » au lieu de « 3.c »)"),
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
    print("== 5. mutations")
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

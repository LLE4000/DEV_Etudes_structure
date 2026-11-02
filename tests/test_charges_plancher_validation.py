# -*- coding: utf-8 -*-
"""Validation croisée des charges sur plancher (lot 4) : le script
``tools/validation_charges_plancher.py`` — grilles fines de sections,
créneaux sur grille, point fixe de la bande plastique, éléments finis de
poutre, formules de l'EN 1992-1-1 et du Heft 240 réécrites à part, hors
moteur — rend VALIDÉ en mode rapide (6 filières, 60 jeux de créneaux).

Il faut numpy et matplotlib (hors chemin de production) : sans eux, le test
le dit et passe (la MESURE de référence est le rapport daté de
docs/validation/, produit par le script complet).

Lancement : python3 tests/test_charges_plancher_validation.py
"""
import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)
sys.path.insert(0, os.path.join(RACINE, "tools"))

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{info}]" if info and not cond else ""))


try:
    import numpy  # noqa: F401
    import matplotlib  # noqa: F401
    DEPS = True
except ImportError:
    DEPS = False

DOSSIER = os.path.join("docs", "validation")
print("=== Validation croisée des charges sur plancher (lot 4) ===")
chk("le rapport daté de la validation croisée existe (MESURE, docs/validation/)",
    any(f.startswith("charges_plancher_validation_") and f.endswith(".md") and not f.endswith("_rapide.md")
        for f in os.listdir(DOSSIER)))
chk("les trois figures de la mesure complète existent",
    all(os.path.exists(os.path.join(DOSSIER, "figures", f)) for f in
        ("plancher_validation_A_roue.png", "plancher_validation_C_dalle.png", "plancher_validation_D_filiere.png")))
if not DEPS:
    print("  SAUT  numpy / matplotlib absents : seul le rapport daté est contrôlé")
else:
    import validation_charges_plancher as V
    verdict = V.main(rapide=True)
    chk("mode rapide (6 filières, 60 jeux de créneaux) : VALIDÉ — roue sur hourdis, stabilisateur sur tôle à cheval, "
        "Manitou sur dalle, filière contre éléments finis, créneaux exhaustifs", verdict)
    chk("figures produites (mode rapide, à côté de la mesure complète)",
        all(os.path.exists(os.path.join(V.FIG, f)) for f in
            ("plancher_validation_A_roue_rapide.png", "plancher_validation_C_dalle_rapide.png",
             "plancher_validation_D_filiere_rapide.png")))
    # le cas A tient le défaut non conservatif du 02/11/2026 : le maximum de M_Ed est ENTRE la charge et la mi-portée
    A = V.cas_A()
    m_fin, m_moteur = float(max(A["m_ed"])), A["lignes"][0][1]
    chk("cas A : le maximum de M_Ed n'est ni à la charge ni à mi-portée, et le moteur le trouve (pas L/60)",
        A["m_a_la_charge"] < m_fin * (1 - 5e-3) and A["m_mi_portee"] < m_fin * (1 - 5e-3) and m_moteur >= m_fin * (1 - 5e-4),
        f"charge {A['m_a_la_charge']:.3f}, mi-portée {A['m_mi_portee']:.3f}, fin {m_fin:.3f}, moteur {m_moteur:.3f}")
    for f in os.listdir(V.FIG):
        if f.endswith("_rapide.png"):
            os.remove(os.path.join(V.FIG, f))
    for f in os.listdir(V.DOSSIER):
        if f.startswith("charges_plancher_validation_") and f.endswith("_rapide.md"):
            os.remove(os.path.join(V.DOSSIER, f))
chk("le moteur de production n'importe ni scipy ni matplotlib ni streamlit (numpy seul, R3)",
    not any(m in open(os.path.join("plancher", "charges", f), encoding="utf-8").read()
            for f in os.listdir(os.path.join("plancher", "charges")) if f.endswith(".py")
            for m in ("import scipy", "import matplotlib", "import streamlit")))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

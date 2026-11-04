# -*- coding: utf-8 -*-
"""Harnais AppTest de la page « Catalogue des profilés » (modules/tableau_profiles.py)
sur l'application réelle : ouverture, compteur, familles, recherche, filtre
numérique, tri, colonnes, fiche et dessin, comparaison, profilé
personnalisé, préférences (sur un fichier temporaire), export CSV, bouton
Accueil, étanchéité des clés.

Lancement : /usr/bin/python3.12 tests/test_tableau_profiles_ecran.py
"""
import os
import sys
import tempfile

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from streamlit.testing.v1 import AppTest  # noqa: E402

from acier.profiles import catalogue as CAT  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{info}]" if info and not cond else ""))


def ouvrir():
    at = AppTest.from_file(os.path.join(RACINE, "streamlit_app.py"), default_timeout=180)
    at.session_state["page"] = "Tableau profilés"
    at.run()
    return at


def md(at):
    return "\n".join(str(m.value) for m in at.markdown)


def compteur(at):
    import re
    m = re.search(r"\*\*(\d+) profilé", md(at))
    return int(m.group(1)) if m else -1


def widget(at, kind, key):
    return next(w for w in getattr(at, kind) if w.key == key)


print("=== 1. Ouverture ===")
at = ouvrir()
chk("aucune exception", not at.exception, str([str(e.value)[:200] for e in at.exception]))
chk("titre, compteur, tableau, fiche avec dessin SVG", "Catalogue des profilés" in md(at) and compteur(at) > 500 and len(at.dataframe) == 1
    and "PROFILÉ" in md(at) and "<svg" in md(at), str(compteur(at)))
n0 = compteur(at)
chk("le tableau a autant de lignes que le compteur", len(at.dataframe[0].value) == n0, f"{len(at.dataframe[0].value)} vs {n0}")
chk("colonnes par défaut dans le tableau (Profilé, Famille, h, Iy, Wpl,y…)",
    all(c in list(at.dataframe[0].value.columns) for c in ("Profilé", "Famille", "h (mm)", "Iy (cm⁴)", "Wpl,y (cm³)")), str(list(at.dataframe[0].value.columns)[:8]))
chk("pastilles de familles par groupe (I/H, U, L, creux)", {str(p.label) for p in at.pills} >= {"I/H", "U", "L", "creux"}, str([str(p.label) for p in at.pills]))

print("\n=== 2. Recherche, filtre, tri ===")
widget(at, "text_input", "prf_recherche").set_value("IPE 200").run()
n1 = compteur(at)
chk("recherche « IPE 200 » : le compteur baisse, IPE 200 dans le tableau", 0 < n1 < n0 and "IPE 200" in list(at.dataframe[0].value["Profilé"]), str(n1))
widget(at, "text_input", "prf_recherche").set_value("").run()
widget(at, "selectbox", "prf_w_f_cle_0").select("Iy").run()
widget(at, "number_input", "prf_w_f_min_0").set_value(10000.0).run()
n2 = compteur(at)
vals = list(at.dataframe[0].value["Iy (cm⁴)"])
chk("filtre Iy ≥ 10 000 cm⁴ : compteur réduit, toutes les lignes ≥ 10 000", 0 < n2 < n0 and all(v >= 10000 for v in vals), f"{n2} ; min {min(vals) if vals else None}")
widget(at, "selectbox", "prf_tri").select("masse").run()
widget(at, "toggle", "prf_decroissant").set_value(True).run()
m = list(at.dataframe[0].value["Masse (kg/m)"])
chk("tri par masse décroissante", all(a >= b for a, b in zip(m, m[1:])))
widget(at, "button", "prf_btn_filtres_rien").click().run()
chk("effacer les filtres : compteur de départ", compteur(at) == n0, str(compteur(at)))

print("\n=== 3. Familles et colonnes ===")
widget(at, "button", "prf_btn_rien").click().run()
chk("« Aucune » famille : tableau vide et message", compteur(at) == 0 and any("Aucun profilé" in str(i.value) for i in at.info))
widget(at, "button", "prf_btn_tout").click().run()
chk("« Toutes les familles » : tout le catalogue", compteur(at) == len(CAT.profils()), str(compteur(at)))
widget(at, "toggle", "prf_courants").set_value(True).run()
chk("« Séries courantes seulement » : moins de profilés, aucun HL ni IPE A", compteur(at) < len(CAT.profils())
    and not any(f in ("HL", "IPE A") for f in at.dataframe[0].value["Famille"]))
widget(at, "toggle", "prf_courants").set_value(False).run()
pil = widget(at, "pills", "prf_w_col_Torsion")
pil.set_value(["It", "Wt", "Iw"]).run()
chk("colonnes : Wt et Iw ajoutées au tableau", all(c in list(at.dataframe[0].value.columns) for c in ("Wt (cm³)", "Iw (cm⁶)")))

print("\n=== 4. Fiche, comparaison, personnalisé ===")
widget(at, "selectbox", "prf_w_fiche").select("HEB 300").run()
chk("fiche HEB 300 : titre, dessin, It et Iw dans la fiche", "<h3>HEB 300</h3>" in md(at) and "<svg" in md(at) and "It :" in "\n".join(str(c.value) for c in at.caption))
widget(at, "multiselect", "prf_compare").set_value(["HEB 300", "HEA 300", "IPE 300"]).run()
chk("comparaison de trois profilés : tableau avec les trois noms", all(n in md(at) for n in ("HEB 300", "HEA 300", "IPE 300")) and "Propriété" in md(at))
widget(at, "number_input", "prf_perso_PRS_h").set_value(500.0).run()
widget(at, "button", "prf_perso_btn").click().run()
chk("profilé personnalisé PRS 500x200x8x12 : ajouté, fiche ouverte, pastille « personnalisé », liste déroulante alignée",
    "<h3>PRS 500x200x8x12</h3>" in md(at) and "personnalisé</span>" in md(at) and "PRS 500x200x8x12" in list(at.dataframe[0].value["Profilé"])
    and widget(at, "selectbox", "prf_w_fiche").value == "PRS 500x200x8x12")
widget(at, "selectbox", "prf_perso_fam").select("CAISSON").run()
widget(at, "number_input", "prf_perso_CAISSON_t").set_value(200.0).run()
widget(at, "button", "prf_perso_btn").click().run()
chk("dimensions impossibles (t ≥ demi-dimension) : message d'erreur, pas d'exception", any("impossibles" in str(e.value) for e in at.error) and not at.exception)

print("\n=== 5. Préférences et exports ===")
orig = CAT.CHEMIN_PREFS
with tempfile.TemporaryDirectory() as d:
    CAT.CHEMIN_PREFS = os.path.join(d, "prefs.json")
    try:
        widget(at, "button", "prf_btn_save").click().run()
        chk("« Garder par défaut » : fichier écrit avec les colonnes de l'écran (Wt, Iw comprises)",
            os.path.exists(CAT.CHEMIN_PREFS) and "Wt" in CAT.preferences()["colonnes"] and "Iw" in CAT.preferences()["colonnes"])
        widget(at, "button", "prf_btn_reset").click().run()
        n_persos = len(at.session_state["prf_persos"])          # le PRS ajouté plus haut reste au tableau
        chk("« Rétablir » : défauts livrés, Wt hors du tableau, familles par défaut (+ les personnalisés de la session)",
            CAT.preferences()["colonnes"] == CAT.DEFAUT_COLONNES and "Wt (cm³)" not in list(at.dataframe[0].value.columns)
            and compteur(at) == len(CAT.filtrer(CAT.profils(), CAT.PREFS_DEFAUT["familles"])) + n_persos, f"{compteur(at)} (persos {n_persos})")
    finally:
        CAT.CHEMIN_PREFS = orig
chk("boutons CSV et fiche JSON présents", {b.label for b in at.download_button} >= {"⬇️ CSV", "⬇️ Fiche (JSON)"})
chk("bouton Accueil présent, toutes les clés préfixées prf_",
    any("Accueil" in str(b.label) for b in at.button) and all(str(k).startswith("prf_") for k in at.session_state if "prf" in str(k)))
chk("méthode et sources : expander avec les sources des dimensions et les définitions",
    any("Méthode et sources" in str(e.label) for e in at.expander) and "structuralcodes" in md(at) and "Saint-Venant" in md(at))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

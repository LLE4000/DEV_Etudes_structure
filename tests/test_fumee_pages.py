# -*- coding: utf-8 -*-
"""Test de fumée : chaque page de l'application se charge sans exception
(AppTest sur streamlit_app.py, page par page, y compris la nouvelle page
« Assemblages métalliques » et l'écran de l'assemblage ouvert).

Lancement : python3 tests/test_fumee_pages.py (depuis la racine).
"""
import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from streamlit.testing.v1 import AppTest  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom
          + (f"   [{info}]" if info and not cond else ""))


PAGES = ["Accueil", "Poutre", "Dalle", "Prédalle", "Cornière", "Garde-corps", "Poutre bois",
         "Tableau armatures", "Age béton", "Choix profilé", "Flambement", "Tableau profilés",
         "Enrobage", "Rigidité du sol", "Taux d'armature", "Assemblages métalliques"]

print("=== Chargement de chaque page ===")
for page in PAGES:
    at = AppTest.from_file(os.path.join(RACINE, "streamlit_app.py"), default_timeout=180)
    at.session_state["page"] = page
    try:
        at.run()
        chk(page, not at.exception, str(at.exception[0].message)[:160] if at.exception else "")
    except Exception as e:  # noqa: BLE001
        chk(page, False, f"{type(e).__name__}: {e}"[:160])

print("\n=== Écran de l'assemblage ouvert ===")
at = AppTest.from_file(os.path.join(RACINE, "streamlit_app.py"), default_timeout=180)
at.session_state["page"] = "Assemblages métalliques"
at.session_state["asm_courant"] = "doubles_cornieres"
at.run()
chk("Doubles cornières d'âme", not at.exception, str(at.exception[0].message)[:160] if at.exception else "")
chk("page inconnue → accueil (comportement inchangé)",
    (lambda a: (a.run(), not a.exception)[1])(
        (lambda a: (a.session_state.__setitem__("page", "n'existe pas"), a)[1])(
            AppTest.from_file(os.path.join(RACINE, "streamlit_app.py"), default_timeout=180))))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:200])
sys.exit(1 if KO else 0)

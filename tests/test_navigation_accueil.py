# -*- coding: utf-8 -*-
"""Le bouton « 🏠 Accueil » ramène à l'accueil depuis CHAQUE page, y compris
quand la page a été ouverte par une tuile (paramètre d'URL ``?page=…``).

Défaut du 04/11/2026 : ``streamlit_app.py`` réappliquait le paramètre
d'URL à chaque exécution, et la page revenait à celle de l'URL au rerun
qui suivait le clic ; de plus huit modules n'avaient pas de bouton. Le
test ouvre chaque page de ``pages`` par le paramètre d'URL, clique le
premier bouton « 🏠 Accueil » et exige l'accueil à l'écran.

Lancement : python3 tests/test_navigation_accueil.py   (python ≥ 3.12 : f-strings de poutre.py)
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
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{info}]" if info and not cond else ""))


def accueil_affiche(at):
    return any("Études Structure" in str(getattr(e, "value", "")) for e in at.markdown) and at.session_state["page"] == "Accueil"


import streamlit_app  # noqa: E402  — la liste des pages, source unique

PAGES = [p for p in streamlit_app.pages if p != "Accueil"]
print(f"=== {len(PAGES)} pages : ouverture par ?page=…, clic sur 🏠 Accueil ===")
for page in PAGES:
    at = AppTest.from_file(os.path.join(RACINE, "streamlit_app.py"), default_timeout=120)
    at.query_params["page"] = page
    at.run()
    boutons = [b for b in at.button if "Accueil" in str(b.label)]
    if not boutons:
        chk(f"{page} : un bouton 🏠 Accueil existe", False, "aucun bouton")
        continue
    chk(f"{page} : ouverte par l'URL, hors accueil", at.session_state["page"] == page and not accueil_affiche(at), str(at.session_state["page"]))
    boutons[0].click().run()
    chk(f"{page} : 🏠 Accueil ramène à l'accueil (paramètre d'URL effacé)",
        accueil_affiche(at) and not at.exception and "page" not in dict(at.query_params),
        f"page={at.session_state.get('page')} qp={dict(at.query_params)} exc={[str(e.value)[:80] for e in at.exception]}")

print("\n=== le second retour tient aussi (pas de rebond vers l'URL) ===")
at = AppTest.from_file(os.path.join(RACINE, "streamlit_app.py"), default_timeout=120)
at.query_params["page"] = "Tableau armatures"
at.run()
[b for b in at.button if "Accueil" in str(b.label)][0].click().run()
at.run()
chk("après un rerun supplémentaire, toujours l'accueil", accueil_affiche(at))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

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

from acier import registre as registre_f  # noqa: E402

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

print("\n=== Écran de chaque assemblage ouvert ===")
for ident, nom in (("doubles_cornieres", "Doubles cornières d'âme"), ("plat_ame", "Plat d'âme soudé"),
                   ("platine_about", "Platine d'about"), ("platine_beton", "Pied de poteau"),
                   ("plat_ame_poteau", "Plat d'âme soudé sur poteau"),
                   ("doubles_cornieres_poteau", "Doubles cornières d'âme sur poteau"),
                   ("platine_about_poteau", "Platine d'about sur poteau")):
    at = AppTest.from_file(os.path.join(RACINE, "streamlit_app.py"), default_timeout=180)
    at.session_state["page"] = "Assemblages métalliques"
    at.session_state["asm_courant"] = ident
    at.run()
    chk(nom, not at.exception and at.session_state["asm_courant"] == ident,
        str(at.exception[0].message)[:160] if at.exception else "")
print("\n=== Catalogue : la famille « Poutre – Poteau » ===")
# `modules/assemblages._catalogue` rend un bouton « Créer » (clé
# asm_btn_open_<id>) pour un assemblage DISPONIBLE, et la mention « À venir »
# sinon (registre.Assemblage.disponible : état « disponible » ET module non
# vide). Depuis le lot 2, `plat_ame_poteau` est DISPONIBLE ; `doubles_cornieres_poteau`
# et `platine_about_poteau` le sont aussi depuis leurs lots (le harnais suit). Une
# attache encore « à venir » paraît au catalogue, sans bouton, et son module n'est
# pas importé.
at = AppTest.from_file(os.path.join(RACINE, "streamlit_app.py"), default_timeout=180)
at.session_state["page"] = "Assemblages métalliques"
at.run()
TXT = ("\n".join(str(m.value) for m in at.markdown) + "\n"
       + "\n".join(str(c.value) for c in at.caption))   # le pitch est un st.caption
BOUTONS = {b.key for b in at.button if b.key}
chk("la page du catalogue se charge", not at.exception,
    str(at.exception[0].message)[:160] if at.exception else "")
chk("la famille « Poutre – Poteau » est titrée, avec son accroche",
    "#### Poutre – Poteau" in TXT
    and "Attaches d'une poutre portée sur l'aile ou dans l'âme d'un poteau" in TXT)
chk("plat_ame_poteau : titre, vignette (role=img + aria-label) et bouton « Créer »",
    "Plat d'âme soudé sur poteau (fin plate)" in TXT
    and 'aria-label="Plat d\'âme soudé sur poteau, vue en plan"' in TXT and 'role="img"' in TXT
    and "asm_btn_open_plat_ame_poteau" in BOUTONS, str(sorted(BOUTONS)))
chk("doubles_cornieres_poteau : titre, vignette (role=img + aria-label) et bouton « Créer »",
    "Doubles cornières d'âme sur poteau" in TXT
    and 'aria-label="Doubles cornières d\'âme sur poteau, vue en plan"' in TXT and 'role="img"' in TXT
    and "asm_btn_open_doubles_cornieres_poteau" in BOUTONS, str(sorted(BOUTONS)))
chk("platine_about_poteau : titre, vignette (role=img + aria-label) et bouton « Créer »",
    "Platine d'about (hauteur partielle) sur poteau" in TXT
    and 'aria-label="Platine d\'about sur poteau, vue en plan"' in TXT and 'role="img"' in TXT
    and "asm_btn_open_platine_about_poteau" in BOUTONS, str(sorted(BOUTONS)))
# Le catalogue écrit « À venir » pour un assemblage non disponible et un bouton
# « Créer » pour chacun des disponibles : les deux attendus se lisent sur le
# REGISTRE, de sorte que les lots 3, 4 et 5 (un module chacun) ne se marchent
# pas dessus dans ce fichier partagé.
_AVENIR = [a.id for f in registre_f.FAMILLES for a in f.assemblages if not a.disponible]
_DISPO = [a.id for a in registre_f.disponibles()]
chk(f"une mention « À venir » par attache encore à écrire ({len(_AVENIR)})",
    TXT.count("À venir") == len(_AVENIR), f"{TXT.count('À venir')} pour {_AVENIR}")
chk(f"les {len(_DISPO)} assemblages disponibles gardent leur bouton « Créer »",
    all(f"asm_btn_open_{i}" in BOUTONS for i in _DISPO),
    str([i for i in _DISPO if f"asm_btn_open_{i}" not in BOUTONS]))
for ident in _AVENIR:
    atv = AppTest.from_file(os.path.join(RACINE, "streamlit_app.py"), default_timeout=180)
    atv.session_state["page"] = "Assemblages métalliques"
    atv.session_state["asm_courant"] = ident
    atv.run()
    chk(f"{ident} : ouvrir un assemblage « à venir » ramène à la page projet, sans "
        "importer son module", not atv.exception
        and "asm_courant" not in atv.session_state.to_dict()
        and "Assemblages (0)" in "\n".join(str(m.value) for m in atv.markdown),
        str(atv.exception[0].message)[:160] if atv.exception else "")

chk("page inconnue → accueil (comportement inchangé)",
    (lambda a: (a.run(), not a.exception)[1])(
        (lambda a: (a.session_state.__setitem__("page", "n'existe pas"), a)[1])(
            AppTest.from_file(os.path.join(RACINE, "streamlit_app.py"), default_timeout=180))))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:200])
sys.exit(1 if KO else 0)

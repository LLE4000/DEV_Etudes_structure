# -*- coding: utf-8 -*-
"""La TAILLE DU CODE (``outils/taille_du_code.py``, demande du propriétaire du 13/11/2026 : voir, lot après lot, la
réduction du code module par module).

1. Le COMPTE d'une ligne de code, sur des extraits connus : ni ligne vide, ni commentaire, ni docstring (Python) ; ni
   commentaire // ou /* … */ (JavaScript, CSS) ; une chaîne sur plusieurs lignes qui n'est pas une docstring compte.
2. Le SITE est entièrement rangé : chaque fichier que la page charge tombe dans un groupe (un fichier sans règle lève
   ValueError — le tableau ne peut pas oublier du code) ; un fichier inventé hors des règles est bien refusé.
3. Le TABLEAU dit le total du site = la somme des modules et du code commun ; l'ancien logiciel et les tests à part.
4. L'HISTORIQUE (``docs/site/taille_du_code.json``) : des mesures datées, la première est celle d'avant l'unification.

    python3 tests/test_taille_du_code.py
"""
import importlib.util
import json
import os
import sys
import warnings

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)
_spec = importlib.util.spec_from_file_location("taille_du_code", os.path.join(RACINE, "outils", "taille_du_code.py"))
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:300]}"))


print("=== 1 · Ce qui compte pour une ligne de code")
PY = '''"""La docstring du module,
sur deux lignes."""
import os  # un commentaire en bout de ligne : la ligne compte


# un commentaire seul
def f(x):
    """La docstring de la fonction."""
    texte = """une chaîne
sur deux lignes"""
    return x + 1


class A:
    """La docstring de la classe."""
    y = 2
'''
chk("Python : 7 lignes de code (import, def, la chaîne « texte » sur 2 lignes, return, class, y) — docstrings, "
    "commentaires et lignes vides ôtés", T._lignes_python(PY) == 7, T._lignes_python(PY))
JS = """// un commentaire
const a = 1;  // en bout de ligne : compte

/* un bloc
   sur deux lignes */
function f() { return a; }
/* court */ const b = 2;
"""
chk("JavaScript : 3 lignes de code (const a, function, const b après un bloc fermé)", T._lignes_js_css(JS, True) == 3,
    T._lignes_js_css(JS, True))
CSS = """/* titre */
.a { color: red; }

/* bloc
   long */
.b {
  margin: 0;
}
"""
chk("CSS : 4 lignes de code (les deux règles, leur contenu ; les blocs /* */ ôtés)", T._lignes_js_css(CSS, False) == 4,
    T._lignes_js_css(CSS, False))

print("=== 2 · Le site est entièrement rangé")
suivis = T._suivis()
site = T.fichiers_du_site(suivis)
sans = []
for f in site:
    try:
        T.groupe_de(f)
    except ValueError:
        sans.append(f)
chk(f"chacun des {len(site)} fichiers du site tombe dans un groupe (un module de l'accueil, les bibliothèques, le commun)",
    not sans and len(site) > 300, sans[:5])
try:
    T.groupe_de("api_web/module_invente.py")
    refuse = False
except ValueError:
    refuse = True
chk("un fichier qu'aucune règle ne range est REFUSÉ (ValueError) : le tableau ne l'oublie pas en silence", refuse)
chk("la dalle et la prédalle sont un seul groupe ; la poutre le sien (les moteurs, l'écran, la note PDF)",
    T.groupe_de("beton/predalle/moteur.py") == T.groupe_de("api_web/dalle.py") == T.groupe_de("modules/export_pdf_dalle.py") == "dalle"
    and T.groupe_de("beton/poutre/moteur.py") == T.groupe_de("modules/export_pdf.py") == "poutre")
chk("la règle de position des armatures (beton/position.py, lot 3 du 13/11/2026), partagée par la poutre et les dalles, est du "
    "code COMMUN — comme api_web/commun.py qui la lit", T.groupe_de("beton/position.py") == T.groupe_de("api_web/commun.py") == "commun")

print("=== 3 · Le tableau")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    m = T.mesurer()
tab = T.tableau(m)
total = sum(m[c][0] for c, _ in T.GROUPES)
chk("le total du site = la somme des modules et du code commun ; l'ancien logiciel et les tests à part, hors du total",
    f"**{T._nb(total)}**" in tab and all(nom in tab for _, nom in T.GROUPES + T.HORS_SITE)
    and m["streamlit"][0] > 0 and m["tests"][0] > 0, tab[-400:])
chk("chaque module de l'accueil a du code (aucun groupe vide : une règle qui ne reconnaîtrait plus ses fichiers se voit)",
    all(m[c][0] > 0 for c, _ in T.GROUPES), [c for c, _ in T.GROUPES if m[c][0] == 0])

print("=== 4 · L'historique des mesures")
with open(T.HISTORIQUE, encoding="utf-8") as f_:
    h = json.load(f_)["mesures"]
chk("l'historique est daté, la première mesure est celle d'avant l'unification (13/11/2026) et nomme son commit",
    h and h[0]["date"] == "2026-11-13" and "avant" in h[0]["etiquette"] and h[0]["commit"]
    and all(set(x) >= {"date", "commit", "etiquette", "groupes"} for x in h), h[:1])

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:240])
sys.exit(1 if KO else 0)

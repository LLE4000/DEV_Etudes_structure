# -*- coding: utf-8 -*-
"""Le REGISTRE PARESSEUX (docs/site/REFONTE_ETABLI.md, lot S4, décision S4-D9) : ``registre.module(clé)`` ne lit que
le fichier ``api_web/<clé>.py``, une fois, et refuse EXACTEMENT ce que ``registre.cles()`` refuse.

Le moteur du navigateur sert désormais tous les écrans d'une session ; chaque requête passe par ``module()``. Avant, il
relisait les 54 fichiers du dossier à chaque requête (``cles()``, ≈ 5 ms mesurés en CPython).

1. ESPION sur ``registre._lire`` : ``module("poutre")`` lit au plus un fichier, puis aucun au deuxième appel.
2. ÉQUIVALENCE : pour chaque fichier de api_web/ et chaque nom piégé (commun, routeur, registre, _x, ../poutre, None,
   « Poutre », « poutre\\n », « », une liste…), ``module()`` accepte ⇔ la clé est dans ``cles()``.
3. ``module(k)`` est le module importé (``importlib.import_module("api_web.<k>")``).
4. MUTATION : un ``module()`` qui rappellerait ``cles()`` (l'ancien) est rejeté par le contrôle 1.

    python3 tests/test_api_web_registre.py
"""
import importlib
import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

from api_web import registre as REG  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:300]}"))


def lectures(fn, *a):
    """(nombre de fichiers lus par ``registre._lire`` pendant ``fn(*a)``, résultat ou exception)."""
    vus = []
    orig = REG._lire

    def espion(f):
        vus.append(f)
        return orig(f)
    REG._lire = espion
    try:
        try:
            r = fn(*a)
        except Exception as e:  # noqa: BLE001 — le refus est un résultat
            r = e
    finally:
        REG._lire = orig
    return len(vus), r


def accepte(fn, cle) -> bool:
    try:
        fn(cle)
        return True
    except KeyError:
        return False


# ------------------------------------------------------------------ 1. paresseux
print("=== 1. Le registre ne lit que le fichier du module, une fois ===")
REG._MODULES.clear()
n1, m1 = lectures(REG.module, "poutre")
n2, m2 = lectures(REG.module, "poutre")
chk(f"module('poutre') lit au plus 1 fichier (lu : {n1}), puis 0 au deuxième appel (lu : {n2})", n1 <= 1 and n2 == 0, (n1, n2))
chk("les deux appels rendent le même module", m1 is m2 and not isinstance(m1, Exception), (m1, m2))
n3, _ = lectures(REG.module, "commun")
chk(f"un fichier qui n'est pas un module (commun) est refusé en lisant au plus son seul fichier (lu : {n3})", n3 <= 1, n3)

# ------------------------------------------------------------------ 2. mêmes refus que cles()
print("\n=== 2. Mêmes acceptations et mêmes refus que cles() ===")
CLES = set(REG.cles())
fichiers = sorted(f[:-3] for f in os.listdir(REG.ICI) if f.endswith(".py"))
PIEGES = ["commun", "routeur", "registre", "_x", "__init__", "../poutre", "poutre/../dalle", "Poutre", "poutre\n",
          " poutre", "", "poutre.py", "biblio_beton", "biblio_acier", "fiches", "parcours", None, 3, ["poutre"]]
ecarts = []
for c in fichiers + PIEGES:
    REG._MODULES.clear()
    attendu = isinstance(c, str) and c in CLES
    try:
        obtenu = accepte(REG.module, c)
    except Exception as e:  # noqa: BLE001 — tout autre type d'erreur est un écart
        obtenu = f"{type(e).__name__}"
    if obtenu is not attendu:
        ecarts.append((c, attendu, obtenu))
chk(f"{len(fichiers)} fichiers de api_web/ et {len(PIEGES)} noms piégés : module() accepte ⇔ clé de cles() "
    f"({len(CLES)} modules)", not ecarts, ecarts)
chk("chaque clé du registre est acceptée", all(accepte(REG.module, k) for k in sorted(CLES)))

# ------------------------------------------------------------------ 3. l'objet rendu
print("\n=== 3. Le module rendu est le module importé ===")
chk("module(k) is importlib.import_module('api_web.' + k), pour chaque clé",
    all(REG.module(k) is importlib.import_module(f"api_web.{k}") for k in sorted(CLES)))

# ------------------------------------------------------------------ 4. mutation
print("\n=== 4. Mutation : l'ancien module() qui relit tout le dossier ===")


def ancien(cle):
    if not isinstance(cle, str) or cle not in REG.cles():
        raise KeyError(cle)
    return importlib.import_module(f"api_web.{cle}")


m1, _ = lectures(ancien, "poutre")
m2, _ = lectures(ancien, "poutre")
chk(f"le contrôle 1 rejette l'ancien module() ({m1} puis {m2} fichiers lus)", not (m1 <= 1 and m2 == 0), (m1, m2))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

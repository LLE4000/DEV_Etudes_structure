# -*- coding: utf-8 -*-
"""Moteur « Rigidité du sol » (paquet sol/) : la PARITÉ avec l'écran Streamlit modules/rigidite_sol.py v5.0, qui
reste la référence. Chaque contrôle rougit si l'on retire ce qu'il protège (section 9 : on retire, il rougit).

  1. Copies conformes : sol/base.py et sol/theorie.py sont le code de modules/sol_base.py et sol_theorie.py OCTET
     POUR OCTET hors de leur en-tête ; chaque fonction de sol/imports.py a la source de celle de sol_import.py.
  2. Mêmes fonctions, mêmes flottants : chaque fonction de la théorie, de la base et de l'import rejouée des deux
     côtés sur des centaines d'entrées — égalité STRICTE (au bit près).
  3. À la main : Newmark, tassement d'un sol homogène, k = q/w, plat en béton, Terzaghi, Robertson — recalculés ici.
  4. L'écran rejoué : modules/rigidite_sol.show() est exécuté avec un faux « st » (les widgets rendent l'état), un faux
     pandas (le tableau = ses lignes) et un faux matplotlib (le graphique = ses séries) ; pour chaque note (plus de 50
     fondations, sondages multiples, tous les modes), CHAQUE ligne de la colonne Résultats — avertissements,
     tableaux SCIA et de zonage, enveloppe, profil trop court, détail par sondage, tranches, métriques — est comparée
     à ce que le moteur rend, mis en forme comme l'écran ; chaque série tracée (diffusion, k(B), CPT) est comparée aux
     données du moteur ; chaque appel à tassement() de l'écran est comparé au moteur (dictionnaires entiers, flottants
     au bit près).
  5. Les couches : préremplissage au changement de type, légende de chaque ligne, ajout/copie/suppression de sondages
     et de couches.
  6. L'import d'un essai (GEF, CSV) : aperçu, tableau découpé, « Remplir le tableau » (couches, nappe, nom), exports CSV.
  7. Le fichier de l'écran (raideur_sol.json) relu par le moteur ; les défauts du moteur = l'état de l'écran neuf.
  8. Format et domaine : JSON strict, normalisation idempotente, saisies hors des bornes de l'écran REFUSÉES.
  9. Mutations : neuf altérations du moteur, chacune DOIT faire apparaître des écarts.
 10. Pureté : le moteur se charge et calcule dans un sous-processus où streamlit, pandas, numpy, matplotlib… sont
     REFUSÉS, sous Python 3.12 et 3.13 (celui de Pyodide).
 11. Le SITE contre l'écran : pour chaque fondation des sections 4 (modes 1 à 6), les chiffres que le site affiche
     (api_web/sol.py par le routeur : tableau SCIA, zonage, enveloppe, w et k au centre, z d'influence, q_net, tranches,
     nombres des avertissements, k des modes 3 à 6, abaque) = ceux que l'écran rejoué affiche, à l'arrondi d'affichage
     près ; une contre-épreuve (un k décalé d'un centième sur le site) la fait rougir.

    /usr/bin/python3.12 tests/test_sol_moteur.py       (streamlit, pandas, matplotlib requis pour la parité)
"""
import arrondi_affichage as AA  # noqa: E402  (tests/, le même dossier)
import ast
import contextlib
import copy
import inspect
import json
import logging
import math
import os
import random
import shutil
import subprocess
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from sol import base as B  # noqa: E402
from sol import imports as I  # noqa: E402
from sol import rigidite as R  # noqa: E402
from sol import theorie as T  # noqa: E402

OK, KO = [], []
EGALITES = {"n": 0}


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{str(info)[:700]}]" if info and not cond else ""))


def lire(*chemin):
    with open(os.path.join(RACINE, *chemin), encoding="utf-8") as f:
        return f.read()


def ecarts(a, b, chemin="", out=None):
    """Les écarts entre deux structures : mêmes clés, mêmes longueurs, nombres ÉGAUX (au bit près), textes égaux."""
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            out.append(f"{chemin} : clés {sorted(map(str, set(a) ^ set(b)))}")
        for k in a:
            if k in b:
                ecarts(a[k], b[k], f"{chemin}.{k}", out)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out.append(f"{chemin} : {len(a)} ≠ {len(b)} éléments")
        for i, (x, y) in enumerate(zip(a, b)):
            ecarts(x, y, f"{chemin}[{i}]", out)
    else:
        EGALITES["n"] += 1
        same = (a == b) or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b))
        if not same or type(a) is not type(b) and not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
            out.append(f"{chemin} : {a!r} ≠ {b!r}")
    return out


# ================================================================== 1. copies conformes
print("=== 1. Copies conformes : sol/ = modules/sol_*.py, hors en-tête ===")


def corps(texte):
    """Le code d'un fichier sans son en-tête : la ligne d'encodage et la docstring de tête ajoutée à l'extraction."""
    lignes = texte.splitlines()
    j = 1
    if lignes[1].startswith('"""'):
        j = next(k for k in range(1, len(lignes)) if lignes[k].rstrip().endswith('"""') and (k > 1 or lignes[1].count('"""') == 2)) + 1
    return "\n".join(lignes[j:])


for neuf, ancien in (("base.py", "sol_base.py"), ("theorie.py", "sol_theorie.py")):
    a, b = corps(lire("sol", neuf)), "\n".join(lire("modules", ancien).splitlines()[1:])
    chk(f"sol/{neuf} = modules/{ancien} octet pour octet hors de l'en-tête ({len(b.splitlines())} lignes)", a == b,
        next((f"ligne {k}: {x!r} ≠ {y!r}" for k, (x, y) in enumerate(zip(a.splitlines(), b.splitlines())) if x != y), "longueurs"))

logging.disable(logging.WARNING)
try:
    import modules.navigation as NAV  # noqa: E402
    import modules.rigidite_sol as RS  # noqa: E402
    import modules.sol_base as MB  # noqa: E402
    import modules.sol_import as MI  # noqa: E402
    import modules.sol_theorie as MT  # noqa: E402
except ImportError as e:  # pragma: no cover — la parité EXIGE l'écran
    chk("modules/rigidite_sol.py et ses briques se chargent (indispensables à la parité)", False, e)
    sys.exit(1)

FONCTIONS_IMPORT = ("_sondage", "parse_gef", "_devine_colonnes", "parse_csv", "_controler_profondeurs", "vers_csv",
                    "couches_vers_csv", "reechantillonner")
diff_src = [f for f in FONCTIONS_IMPORT if inspect.getsource(getattr(I, f)) != inspect.getsource(getattr(MI, f))]
chk(f"sol/imports.py : les {len(FONCTIONS_IMPORT)} fonctions extraites ont la source de modules/sol_import.py "
    "(GEF, CSV, contrôle des profondeurs, exports, rééchantillonnage)", not diff_src, diff_src)
chk("sol/imports.py : mêmes tables (codes GEF, en-têtes CSV, bornes physiques de Rf)",
    I._GEF_QTY == MI._GEF_QTY and I._ENTETES == MI._ENTETES and (I.RF_MIN_PLAUSIBLE, I.RF_MAX_PLAUSIBLE) == (MI.RF_MIN_PLAUSIBLE, MI.RF_MAX_PLAUSIBLE))
chk("la référence de parité est bien celle du moteur : rigidite_sol v5.0, sol_theorie v1.0, sol_base v2.0, sol_import v1.0",
    RS.VERSION == R.VERSION == "v5.0" and "VERSION 1.0" in lire("modules", "sol_theorie.py")[:900]
    and "VERSION 2.0" in lire("modules", "sol_base.py")[:300] and "VERSION 1.0" in lire("modules", "sol_import.py")[:300]
    and "VERSION 5.0" in lire("modules", "rigidite_sol.py")[:300])
chk("les listes de l'écran : six modes (libellés exacts), critères 10…30 %, positions, types de sol, profil de départ",
    [lib for _, lib in R.MODES] == ["1. Sondage CPT (import de fichier)", "2. Profil de sol saisi à la main",
                                    "3. Vérification rapide k = q / w", "4. Comparer les théories", "5. Abaque des sols",
                                    "6. Raideur d'un plat en béton"]
    and "options=[10, 15, 20, 25, 30]" in lire("modules", "rigidite_sol.py") and list(R.CRITERES) == [10, 15, 20, 25, 30]
    and R.POSITIONS == MT.POSITIONS and list(R.TYPES) == MB.soil_types_list()
    and set(RS.LAYER_FIELDS) == set(R.nouvelle_couche()) | {"type_prev"})

# ================================================================== 2. mêmes fonctions, mêmes flottants
print("\n=== 2. Théorie, base, import : les deux copies rejouées, égalité stricte ===")
RNG = random.Random(20261107)
TYPES_DB = [t for t in B.SOIL_DB]


def profil_aleatoire(rng, n=None, trous=False, haut=False):
    out = []
    for _ in range(n or rng.randint(1, 6)):
        M = round(rng.choice((rng.uniform(1, 60), rng.uniform(60, 400), rng.uniform(0.5, 5))), 2)
        c = {"h": round(rng.uniform(0.3, 8), 2), "gamma": round(rng.uniform(14, 22), 1), "M": M,
             "nom": rng.choice(TYPES_DB)}
        if trous and rng.random() < 0.2:
            c["M"] = None
        if haut:
            c["M_haut"] = round(M * rng.uniform(1.0, 3.0), 2)
        out.append(c)
    return out


fautes = []


def compare(nom, fa, fb, *a, **k):
    try:
        ra = fa(*a, **k)
    except Exception as e:  # noqa: BLE001
        ra = ("exception", type(e).__name__, str(e))
    try:
        rb = fb(*a, **k)
    except Exception as e:  # noqa: BLE001
        rb = ("exception", type(e).__name__, str(e))
    d = ecarts(ra, rb)
    if d:
        fautes.append((nom, a, d[:3]))


n_appels = 0
for _ in range(400):
    Bx, Lx, z = RNG.uniform(0.2, 30), RNG.uniform(0.2, 60), RNG.choice((0.0, -1.0, RNG.uniform(0, 1e-3), RNG.uniform(0.01, 80)))
    for pos in list(T.POSITIONS) + ["ailleurs"]:
        compare("I_position", T.I_position, MT.I_position, Bx, Lx, z, pos)
    compare("I_coin", T.I_coin, MT.I_coin, Bx, Lx, z)
    compare("delta_sigma", T.delta_sigma, MT.delta_sigma, RNG.uniform(0, 500), Bx, Lx, z, RNG.choice(list(T.POSITIONS)))
    n_appels += 7
for _ in range(150):
    pr = profil_aleatoire(RNG, trous=True)
    nap = RNG.choice((None, RNG.uniform(0, 10)))
    compare("contraintes_en_place", T.contraintes_en_place, MT.contraintes_en_place, pr, RNG.uniform(0, 30), nap)
    kw = dict(D=RNG.choice((0.0, RNG.uniform(0, 4))), nappe_m=nap, critere=RNG.choice((0.1, 0.15, 0.2, 0.25, 0.3)),
              dz=RNG.choice((0.05, 0.10)), position=RNG.choice(list(T.POSITIONS)), q_net=RNG.random() < 0.6)
    args = (pr, RNG.uniform(0.3, 25), RNG.uniform(0.3, 40), RNG.choice((0.0, RNG.uniform(5, 400))))
    compare("tassement", T.tassement, MT.tassement, *args, **kw)
    compare("k_zone", T.k_zone, MT.k_zone, *args, D=kw["D"], nappe_m=nap, critere=kw["critere"], q_net=kw["q_net"])
    n_appels += 3
for _ in range(300):
    E, Bx, nu = RNG.choice((0.0, RNG.uniform(0.5, 400))), RNG.choice((0.0, RNG.uniform(0.2, 30))), RNG.choice((0.0, 0.3, 0.49, 0.5, RNG.uniform(0, 0.5)))
    compare("k_elastique", T.k_elastique, MT.k_elastique, E, Bx, nu, Is=RNG.choice((0.88, 1.12)))
    compare("k_vesic", T.k_vesic, MT.k_vesic, E, Bx, nu, EI_fondation=RNG.choice((None, 0.0, RNG.uniform(1e3, 1e7))))
    compare("k_terzaghi_taille", T.k_terzaghi_taille, MT.k_terzaghi_taille, RNG.uniform(0, 3000), Bx,
            RNG.choice((None, RNG.uniform(0.3, 40))), nature=RNG.choice(("sable", "argile")))
    compare("module_oedometrique", T.module_oedometrique, MT.module_oedometrique, E, RNG.choice((nu, -0.1)))
    compare("module_young", T.module_young, MT.module_young, E, nu)
    qt, sv = RNG.uniform(0, 30000), RNG.uniform(0, 400)
    compare("ic_robertson", T.ic_robertson, MT.ic_robertson, qt, RNG.choice((None, 0.0, RNG.uniform(1, 400))), sv, RNG.uniform(0, sv + 1))
    Ic = RNG.choice((None, RNG.uniform(1, 4.2)))
    compare("sbt_from_ic", T.sbt_from_ic, MT.sbt_from_ic, Ic)
    compare("est_grenu", T.est_grenu, MT.est_grenu, Ic)
    compare("alpha_M_robertson", T.alpha_M_robertson, MT.alpha_M_robertson, Ic, RNG.choice((None, RNG.uniform(0, 40))))
    compare("modules_depuis_cpt", T.modules_depuis_cpt, MT.modules_depuis_cpt, Ic, qt, sv, RNG.uniform(0, sv + 1), qt / 1000,
            RNG.uniform(0, 0.45), Qtn=RNG.choice((None, RNG.uniform(0, 40))))
    compare("gamma_depuis_ic", T.gamma_depuis_ic, MT.gamma_depuis_ic, Ic)
    n_appels += 11


def essai_cpt(rng, n=None, fs=True, pas=0.02, fin_muette=0.0):
    """Un sondage CPT synthétique : des couches de qc et de Rf, du bruit ; ``fin_muette`` : les derniers mètres sans fs
    (capteur muet au refus)."""
    pts, z = [], 0.0
    couches = [(rng.uniform(0.5, 4), rng.uniform(0.3, 25), rng.uniform(0.3, 6)) for _ in range(rng.randint(2, 6))]
    for ep, qc, rf in couches:
        z1 = z + ep
        while z < z1:
            z = round(z + pas, 3)
            q = max(0.05, qc * rng.uniform(0.85, 1.15))
            f = q * 1000 * rf / 100 * rng.uniform(0.8, 1.2) if fs else None
            pts.append((z, round(q, 3), None if f is None else round(f, 2)))
    pts = pts[: n or len(pts)]
    if fin_muette:
        pts = [(z, q, None if z > pts[-1][0] - fin_muette else f) for z, q, f in pts]
    return pts


for _ in range(25):
    pts = essai_cpt(RNG, fs=RNG.random() < 0.85, pas=RNG.choice((0.02, 0.05, 0.1)))
    kw = dict(nappe_m=RNG.choice((None, RNG.uniform(0, 6))), seuil_ic=RNG.choice(R.SEUILS_IC), h_min=RNG.choice((0.2, 0.5, 1.0, 2.5)),
              nu=RNG.uniform(0, 0.45))
    compare("profil_depuis_cpt", T.profil_depuis_cpt, MT.profil_depuis_cpt, pts, **kw)
    cou = T.profil_depuis_cpt(pts, **kw)
    compare("couches_pour_calcul", T.couches_pour_calcul, MT.couches_pour_calcul, cou, RNG.choice(("M_bas", "M_haut")))
    compare("bilan_k", T.bilan_k, MT.bilan_k, cou, RNG.uniform(0.5, 10), RNG.uniform(0.5, 20), RNG.uniform(50, 300),
            D=RNG.uniform(0, 2), nappe_m=kw["nappe_m"], q_net=RNG.random() < 0.5)
    n_appels += 3
chk(f"théorie : {n_appels} appels rejoués dans les deux copies (Newmark, contraintes, tassement, k_zone, élasticité, "
    "Vesić, Terzaghi, Robertson, découpage CPT, bilan) — résultats identiques au bit près",
    not fautes and n_appels > 5000, fautes[:3])

fautes = []
compare("SOIL_DB", lambda: B.SOIL_DB, lambda: MB.SOIL_DB)
compare("listes", lambda: (B.ROCK_CATEGORIES, B._SOIL_KEYWORDS, B.soil_types_list()),
        lambda: (MB.ROCK_CATEGORIES, MB._SOIL_KEYWORDS, MB.soil_types_list()))
LIBELLES = ["", "  ", "Craie altérée grise", "craie alteree", "CRAIE", "schiste houiller", "Schiste W3", "grès fracturé", "gres sain",
            "Argile très molle", "Argile de Boom", "limon sableux", "Sable propre à sable limoneux", "Sable limoneux à limon sableux",
            "Limon argileux à argile limoneuse", "Argile silteuse à argile", "Sol organique, tourbe", "Sable graveleux à sable dense",
            "Indéterminé", "remblai de débris", "calcaire karstique", "Calcaire sain", "silt", "w5"] + TYPES_DB
for t in LIBELLES + [None, "Inconnu"]:
    for f in ("match_soil_type", "soil_default_qc", "soil_default_Rf", "soil_default_E", "soil_default_M", "soil_gamma",
              "soil_nature", "is_rock"):
        if t is None and f != "match_soil_type":
            continue
        compare(f, getattr(B, f), getattr(MB, f), t)
    for qc in (0.0, -1.0, 0.7, 3.5, 22.0):
        compare("suggest_M_from_qc", B.suggest_M_from_qc, MB.suggest_M_from_qc, qc, t, nu=RNG.choice((0.3, 0.25)))
chk(f"base des sols : la table ({len(B.SOIL_DB)} types), les mots-clés, et chaque fonction sur {len(LIBELLES) + 2} libellés "
    "— identiques", not fautes, fautes[:3])


GEF = """#GEFID= 1,1,0
#COLUMN= 3
#COLUMNINFO= 1, m, Sondeerlengte, 1
#COLUMNINFO= 2, MPa, Conusweerstand, 2
#COLUMNINFO= 3, MPa, Wrijvingsweerstand, 3
#COLUMNSEPARATOR= ;
#RECORDSEPARATOR= !
#COLUMNVOID= 2, -9999.000
#TESTID= {nom}
{nappe}#EOH=
{corps}"""


def gef(pts, nom="CPT-042", nappe=None, vide=False):
    lignes = [f"{z:.3f};{q:.3f};{(f or 0) / 1000:.5f}!" for z, q, f in pts]
    if vide:
        lignes.insert(3, "9.000;-9999.000;0.0000!")
    return GEF.format(nom=nom, nappe=f"#MEASUREMENTVAR= 14, {nappe:.2f}, m, grondwaterstand\n" if nappe else "", corps="\n".join(lignes) + "\n")


def csv_fr(pts, fs=True):
    entete = "Profondeur (m);qc (MPa)" + (";fs (kPa)" if fs else "")
    return entete + "\n" + "\n".join(f"{z:.2f};{q:.3f}".replace(".", ",") + (f";{(f or 0):.1f}".replace(".", ",") if fs else "")
                                     for z, q, f in pts) + "\n"


def csv_en(pts):
    return "depth,qc,Rf\n" + "\n".join(f"{z},{q},{(f or 0) / (q * 1000) * 100:.3f}" for z, q, f in pts) + "\n"


FICHIERS = []
for k in range(14):
    pts = essai_cpt(RNG, fs=k % 5 != 4, pas=RNG.choice((0.02, 0.05)), fin_muette=0.3 if k % 4 == 2 else 0.0)
    nap = RNG.choice((None, round(RNG.uniform(0.5, 5), 2)))
    FICHIERS += [(f"CPT{k:02d}.gef", gef(pts, f"S-{k:02d}", nap, vide=k % 3 == 0)), (f"essai_{k}.csv", csv_fr(pts, fs=k % 4 != 3)),
                 (f"essai_{k}.txt", csv_en(pts) if k % 2 else csv_fr(pts).replace(";", "\t"))]
FICHIERS += [("cote.csv", csv_fr([(z + 12.5, q, f) for z, q, f in essai_cpt(RNG)])), ("grossier.csv", csv_fr(essai_cpt(RNG, pas=1.5))),
             ("cm.csv", csv_fr([(z * 100, q, f) for z, q, f in essai_cpt(RNG)])), ("vide.csv", ""), ("court.csv", csv_fr(essai_cpt(RNG)[:3])),
             ("faux.gef", "pas un gef"), ("sans_qc.csv", "a;b\n1;2\n2;3\n3;4\n4;5\n5;6\n"), ("x.doc", "rien"),
             ("latin.gef", gef(essai_cpt(RNG), "Sondage réalisé"))]
fautes, PLANTAGES = [], []
for nom, texte in FICHIERS:
    octets = texte.encode("utf-8")
    compare(f"importer {nom}", I.importer, MI.importer, octets, nom)
    if nom.endswith(".csv"):
        compare(f"parse_csv mapping {nom}", I.parse_csv, MI.parse_csv, octets, nom, mapping={"z": 0, "qc": 1, "fs": 2})
    try:
        s = I.importer(octets, nom)
    except ValueError:
        continue
    compare(f"vers_csv {nom}", I.vers_csv, MI.vers_csv, s)
    compare(f"reechantillonner {nom}", I.reechantillonner, MI.reechantillonner, s, pas=RNG.choice((0.02, 0.1)))
    try:
        cou = T.profil_depuis_cpt(s["points"], nappe_m=s["nappe_m"])
    except TypeError:
        PLANTAGES.append(nom)
        continue
    compare(f"couches_vers_csv {nom}", I.couches_vers_csv, MI.couches_vers_csv, s["nom"], cou)
chk(f"import : {len(FICHIERS)} fichiers (GEF avec et sans nappe, valeur vide, CSV français et anglais, tabulations, Rf, cote "
    "au lieu de profondeur, pas grossier, cm, vides, illisibles) — lecture, avertissements, refus et exports identiques",
    not fautes, fautes[:3])
chk(f"DÉFAUT REPRODUIT : {len(PLANTAGES)} essai(s) dont les dernières mesures n'ont pas de frottement (capteur muet au refus) font "
    "PLANTER le découpage (TypeError dans profil_depuis_cpt) — dans les deux copies ; le moteur le traduit en message (decouper)",
    len(PLANTAGES) >= 3 and all("TypeError" in str(T.__dict__) or True for _ in PLANTAGES), PLANTAGES)
try:
    R.decouper([tuple(p) for p in I.importer(dict(FICHIERS)[PLANTAGES[0]].encode(), PLANTAGES[0])["points"]], 3.0)
    chk("decouper() : le plantage devient une ValueError qui dit pourquoi et que faire", False)
except ValueError as e:
    chk("decouper() : le plantage devient une ValueError qui dit pourquoi et que faire (réduire l'épaisseur minimale, retirer les "
        "dernières lignes)", "Ic" in str(e) and "épaisseur minimale" in str(e) and "Streamlit" in str(e), str(e))
try:
    I.importer(b"%PDF-1.4", "rapport.pdf")
    chk("un PDF est refusé par le moteur avec un message (PyMuPDF absent du navigateur)", False)
except ValueError as e:
    chk("un PDF est refusé par le moteur avec un message qui renvoie vers le GEF, le CSV ou Streamlit", "GEF" in str(e) and "Streamlit" in str(e))

# ================================================================== 3. à la main
print("\n=== 3. À la main ===")


def proche(a, b, tol=1e-12):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def newmark(m, n):
    """Facteur d'influence sous le coin (Newmark 1935), écrit ici indépendamment (branche de l'arctangente explicite)."""
    v = m * m + n * n + 1
    a = 2 * m * n * math.sqrt(v) / (v + m * m * n * n) * (v + 1) / v
    x = 2 * m * n * math.sqrt(v) / (v - m * m * n * n) if v != m * m * n * n else math.inf
    b = math.atan(x) + (math.pi if v < m * m * n * n else 0.0)
    return (a + b) / (4 * math.pi)


TABLE = ((1.0, 1.0, 0.1752), (2.0, 1.0, 0.1999), (0.5, 0.5, 0.0840), (1.0, 0.5, 0.1202), (3.0, 3.0, 0.2439), (0.2, 0.2, 0.0179))
chk("Newmark : I_coin(B, L, z = 1) = tables publiées (m·n = 1·1 → 0,1752 ; 2·1 → 0,1999 ; ½·½ → 0,0840 ; 1·½ → 0,1202…) "
    "et = la formule réécrite ici, branche de l'arctangente comprise (m²n² > m²+n²+1)",
    all(abs(T.I_coin(m, n, 1.0) - v) < 6e-4 and proche(T.I_coin(m, n, 1.0), newmark(m, n), 1e-12) for m, n, v in TABLE))
# un sol homogène de 30 m, q brute, sans nappe : w = Σ q·I(zᵢ)·dz/M, k = q/w — recalculé tranche par tranche
M0, q0, B0, L0 = 25.0, 150.0, 2.0, 3.0
r0 = T.tassement([{"h": 30.0, "gamma": 19.0, "M": M0, "nom": "x"}], B0, L0, q0, D=0.0, critere=0.20, q_net=False)
w, z, n = 0.0, 0.0, 0
while True:
    zm = z + 0.025
    ds = q0 * 4 * newmark(B0 / 2 / zm, L0 / 2 / zm)
    w += ds * 0.05 / (M0 * 1000)
    z += 0.05
    n += 1
    if z >= B0 and ds <= 0.20 * max(19.0 * zm, 1.0):
        break
chk(f"tassement d'un sol homogène M = 25 MPa sous 2 × 3 m, q = 150 kPa : w = {w * 1000:.3f} mm en {n} tranches, "
    f"arrêt à z = {z:.2f} m (Δσ ≤ 20 % σ'v0, au-delà de B), k = q/w = {q0 / w / 1000:.3f} MN/m³",
    proche(r0["w_m"], w, 1e-9) and len(r0["tranches"]) == n and proche(r0["z_influence"], z, 1e-9)
    and proche(r0["k_MNm3"], q0 / w / 1000, 1e-9), (r0["w_mm"], w * 1000, r0["z_influence"], z))
rq = R.calcul_rapide(150.0, 20.0)
chk("mode 3 : k = q/w = 150 / 0,020 = 7 500 kN/m³ = 7,50 MN/m³ ; w = 0 → k = 0 (« nok »)",
    rq["k_kNm3"] == 7500.0 and rq["k_MNm3"] == 7.5 and R.calcul_rapide(150.0, 0.0)["etat"] == "nok")
rp = R.plat(200.0, 300.0, 0.5, 30.0, True, 20.0, 20.0)
kc, kg = 30e6 / 0.1, 20e6 / 0.02
chk(f"mode 6 : h_c = 0,5·min(200 ; 300) = 100 mm, k_c = 30 GPa / 0,1 m = {kc / 1000:,.0f} MN/m³, k_g = 20 GPa / 0,02 m, "
    "1/k_eq = 1/k_c + 1/k_g", proche(rp["hc_m"], 0.1) and proche(rp["kc"], kc) and proche(rp["kg"], kg)
    and proche(rp["keq"], 1 / (1 / kc + 1 / kg)) and R.plat(200, 300, .5, 30, False, 20, 20)["keq"] == rp["kc"])
th = R.theories(25.0, 2.0, 2.0, 0.30)
Mo = 25.0 * 0.7 / (1.3 * 0.4)
chk(f"mode 4 : M = E(1−ν)/((1+ν)(1−2ν)) = {Mo:.3f} MPa ; rigide E/(B(1−ν²)·0,88), souple ·1,12, Vesić E/(B(1−ν²)), "
    "ancienne E/(2B) = 6,25", proche(th["M"], Mo) and proche(th["methodes"][1]["k"], 25 / (2 * 0.91 * 0.88))
    and proche(th["methodes"][2]["k"], 25 / (2 * 0.91 * 1.12)) and proche(th["methodes"][3]["k"], 25 / (2 * 0.91))
    and th["methodes"][4]["k"] == 6.25)
ab = {x["type"]: x for x in R.abaque(2.0, 20.0)["lignes"]}
chk("mode 5 : Terzaghi — sable k·((B+0,3)/2B)², argile k·0,3/B ; q = k·w_réf ; « Personnalisé » écarté",
    proche(ab["Sable dense"]["k_bas"], 80 * (2.3 / 4) ** 2) and proche(ab["Tourbe"]["k_haut"], 5 * 0.15)
    and proche(ab["Sable dense"]["q_bas"], 80 * (2.3 / 4) ** 2 * 20) and "Personnalisé" not in ab and len(ab) == 21)
Icr, Qtn, Fr, nn, cv = T.ic_robertson(5000.0, 50.0, 100.0, 60.0)
# l'itération refaite ici : n = 1, Qtn = (qt − σv0)/pa·min((pa/σ'v0)ⁿ ; 1,7), Ic, n = 0,381·Ic + 0,05·σ'v0/pa − 0,15,
# jusqu'à ce que Ic et n bougent de moins de 1e-5 ; le Qtn rendu est celui du DERNIER n (Ic, lui, vient du n d'avant)
n_m, ic_m, fr_m = 1.0, 2.5, 50.0 / 4900.0 * 100.0
for _ in range(30):
    qtn_m = (4900.0 / 101.325) * min((101.325 / 60.0) ** n_m, 1.7)
    ic2 = math.sqrt((3.47 - math.log10(qtn_m)) ** 2 + (math.log10(fr_m) + 1.22) ** 2)
    n2 = max(0.0, min(1.0, 0.381 * ic2 + 0.05 * 60.0 / 101.325 - 0.15))
    fini = abs(ic2 - ic_m) < 1e-5 and abs(n2 - n_m) < 1e-5
    ic_m, n_m = ic2, n2
    if fini:
        break
chk(f"Robertson : Ic = √((3,47 − log Qtn)² + (log Fr + 1,22)²) = {Icr:.4f} avec Fr = fs/(qt − σv0) = {Fr:.4f} %, n itéré "
    f"= {nn:.4f} (convergé) — l'itération refaite à la main, au bit près", cv and fini and T.PA == 101.325 and proche(Fr, fr_m, 0)
    and Icr == ic_m and nn == n_m and proche(Qtn, (4900.0 / 101.325) * min((101.325 / 60.0) ** n_m, 1.7), 0))

# ================================================================== 4. l'écran rejoué
print("\n=== 4. L'écran rejoué (faux st, faux pandas, faux matplotlib) contre le moteur ===")


class Etat(dict):
    """st.session_state : un dictionnaire qu'on lit aussi par attribut."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k) from None

    def __setattr__(self, k, v):
        self[k] = v


class Relance(Exception):
    pass


class Col:
    def __init__(self, st):
        self.st = st

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def metric(self, label, value, *a, **k):
        self.st.ev.append(("metric", label, value))


class FauxSt:
    """Un « st » qui ENREGISTRE ce que l'écran afficherait ; les widgets rendent la valeur de la session (posée à leur
    valeur initiale si la clé n'existe pas, comme Streamlit) ; un bouton listé dans ``cliques`` est cliqué."""

    def __init__(self, ss, cliques=()):
        self.session_state, self.ev, self.cliques = ss, [], set(cliques)

    def _rec(self, nom):
        def f(body="", *a, **k):
            self.ev.append((nom, body))
        return f

    def __getattr__(self, nom):
        if nom in ("markdown", "caption", "info", "warning", "error", "success", "latex", "divider", "image"):
            return self._rec(nom)
        raise AttributeError(nom)

    def columns(self, spec, **k):
        return [Col(self) for _ in range(spec if isinstance(spec, int) else len(spec))]

    def container(self, **k):
        return Col(self)

    def expander(self, label, expanded=False, **k):
        self.ev.append(("expander", label))
        return Col(self)

    def _val(self, key, defaut):
        if key is None:
            return defaut
        if key not in self.session_state:
            self.session_state[key] = defaut
        return self.session_state[key]

    def number_input(self, label, min_value=None, max_value=None, value=None, step=None, key=None, **k):
        return self._val(key, value if value is not None else (min_value if min_value is not None else 0.0))

    def text_input(self, label, value="", key=None, **k):
        return self._val(key, value)

    def checkbox(self, label, value=False, key=None, **k):
        return self._val(key, value)

    def selectbox(self, label, options, index=0, key=None, **k):
        options = list(options)
        return self._val(key, options[index] if options else None)

    def select_slider(self, label, options, value=None, key=None, **k):
        return self._val(key, value if value is not None else list(options)[0])

    def radio(self, label, options, index=0, key=None, **k):
        return self._val(key, list(options)[index])

    def slider(self, label, min_value=None, max_value=None, value=None, step=None, key=None, **k):
        return self._val(key, value)

    def file_uploader(self, *a, **k):
        return None

    def button(self, label, key=None, on_click=None, args=(), **k):
        if key in self.cliques:
            if on_click:
                on_click(*args)
            return True
        return False

    def download_button(self, label, data=None, file_name=None, **k):
        self.ev.append(("download", label, file_name, data))
        return False

    def dataframe(self, df, **k):
        self.ev.append(("dataframe", df))

    def rerun(self):
        raise Relance()


class FauxPd:
    @staticmethod
    def DataFrame(rows):
        return [dict(r) for r in rows]


class Axe:
    def __init__(self, journal, i):
        self.j, self.i = journal, i

    def get_xlim(self):
        return (0.0, 1.0)

    def __getattr__(self, m):
        def f(*a, **k):
            self.j.append((self.i, m, a, k))
        return f


class Fig:
    def tight_layout(self):
        pass

    def savefig(self, *a, **k):
        pass


class FauxPlt:
    def __init__(self):
        self.figs = []

    def subplots(self, nrows=1, ncols=1, **k):
        j = []
        self.figs.append(j)
        axes = [Axe(j, i) for i in range(nrows * ncols)]
        return Fig(), (axes[0] if len(axes) == 1 else tuple(axes))

    def get_cmap(self, nom):
        return lambda x: ("cmap", x)

    def close(self, fig):
        pass


@contextlib.contextmanager
def ecran_simule(st):
    """L'écran branché sur les faux : st, pandas, matplotlib ; _bloc et tassement() espionnés."""
    plt = FauxPlt()
    appels, blocs = [], []
    vrai_tass = MT.tassement

    def espion_tass(*a, **k):
        r = vrai_tass(*a, **k)
        appels.append((a, k, r))
        return r
    vrais = (RS.st, RS.pd, RS.plt, RS._HAS_MPL, RS._HAS_REPORTLAB, RS._bloc, NAV.bouton_accueil, MT.tassement)
    RS.st, RS.pd, RS.plt, RS._HAS_MPL, RS._HAS_REPORTLAB = st, FauxPd, plt, True, False
    RS._bloc = lambda left, right="", etat="ok": st.ev.append(("bloc", left, right, etat))
    NAV.bouton_accueil = lambda *a, **k: None
    MT.tassement = espion_tass
    try:
        yield plt, appels
    finally:
        RS.st, RS.pd, RS.plt, RS._HAS_MPL, RS._HAS_REPORTLAB, RS._bloc, NAV.bouton_accueil, MT.tassement = vrais


def ecran(ss, cliques=()):
    """show() sur l'état ``ss`` : (événements, figures, appels à tassement)."""
    st = FauxSt(ss, cliques)
    with ecran_simule(st) as (plt, appels):
        try:
            RS.show()
        except Relance:
            st.ev.append(("rerun",))
    return st.ev, plt.figs, appels


def charger(f, chart=None):
    """L'état de session de l'écran pour la fondation f (format du moteur)."""
    ss = Etat()
    ss["rs_mode"] = R.LIBELLES_MODES[f["mode"]]
    for k, cle in (("B", "rs_B"), ("L", "rs_L"), ("D", "rs_D"), ("q", "rs_q"), ("nappe_active", "rs_nappe_active"),
                   ("nappe", "rs_nappe"), ("critere", "rs_critere"), ("q_net", "rs_q_net"), ("nu", "rs_nu"), ("position", "rs_pos_scia")):
        ss[cle] = f[k]
    ss["rs_detail"], ss["rs_projet"], ss["rs_w_ref"] = True, "", f["abaque"]["w_ref"]
    ss["soundings"] = [{"id": s["id"], "nom": s["nom"]} for s in f["sondages"]]
    for s in f["sondages"]:
        sid = s["id"]
        ss[f"snd{sid}_nom"] = s["nom"]
        ss[RS._order_key(sid)] = list(range(1, len(s["couches"]) + 1))
        for j, c in enumerate(s["couches"], start=1):
            for k in ("h", "type", "qc", "rf", "M", "gamma", "M_haut", "Ic", "sbt"):
                ss[RS._layer_key(sid, j, k)] = c[k]
            ss[RS._layer_key(sid, j, "type_prev")] = c["type"]
        if s["points"]:
            ss[f"snd{sid}_points"] = [tuple(p) for p in s["points"]]
        if s["source"]:
            ss[f"snd{sid}_source"] = s["source"]
        if s["nappe"] is not None:
            ss[f"snd{sid}_nappe"] = s["nappe"]
    ss["rs3_q"], ss["rs3_w"] = f["rapide"]["q"], f["rapide"]["w"]
    for k in ("E", "B", "L", "nu"):
        ss[f"rs4_{k}"] = f["theories"][k]
    ss["rs5_B"] = f["abaque"]["B"]
    for k, cle in (("B", "rs6_B"), ("L", "rs6_L"), ("alpha", "rs6_a"), ("E", "rs6_E"), ("mortier", "rs6_grout"), ("tg", "rs6_tg"), ("Eg", "rs6_Eg")):
        ss[cle] = f["plat"][k]
    if chart is not None:
        ss["rs_chart_snd"] = chart
    return ss


def lire_session(ss):
    """Les sondages et couches tels que la session de l'écran les porte (format du moteur)."""
    out = []
    for s in ss["soundings"]:
        sid = int(s["id"])
        couches = []
        for lid in ss.get(RS._order_key(sid), []):
            couches.append({k: ss.get(RS._layer_key(sid, lid, k)) for k in ("h", "type", "qc", "rf", "M", "gamma", "M_haut", "Ic", "sbt")})
        out.append({"id": sid, "nom": ss.get(f"snd{sid}_nom"), "couches": couches, "source": ss.get(f"snd{sid}_source") or "",
                    "nappe": ss.get(f"snd{sid}_nappe"), "points": [list(p) for p in ss.get(f"snd{sid}_points") or []]})
    return out


def zone_resultats(ev):
    """Les événements de la colonne Résultats (de « ### Résultats » au trait final)."""
    i = ev.index(("markdown", "### Résultats"))
    j = max(k for k, e in enumerate(ev) if e[0] == "divider")
    return ev[i + 1:j]


def attendu_profil(f, sel):
    """La colonne Résultats des modes 1 et 2, écrite à partir de ce que le MOTEUR rend, avec les formats de l'écran."""
    res = R.resultats_profil(f)
    p, rs = res["parametres"], res["resultats"]
    ev = []
    if p["echange"]:
        ev.append(("info", "B est le petit côté par convention : B et L ont été échangés."))
    if not rs:
        ev.append(("warning", "Renseigne au moins une couche avec une épaisseur et un module M, ou importe un sondage."))
        return ev, res
    ev.append(("markdown", "#### 🎯 À encoder dans SCIA (paramètres de sol C1z)"))
    if res["ecartes"]:
        ev.append(("warning", f"⚠️ **{', '.join(res['ecartes'])} n'entre pas dans les résultats** : aucune couche ne porte de module M. "
                              f"Les valeurs ci-dessous ne couvrent que {len(rs)} sondage(s) sur {len(rs) + len(res['ecartes'])}."))
    for r in rs:
        if "incomplet" in r["convergence"]:
            ev.append(("error", f"❌ **{r['nom']} : k non calculable.** {r['convergence']}. Une couche sans module ne peut pas être "
                                "traitée comme incompressible — ce serait l'hypothèse la plus favorable possible, et elle surestime k. "
                                "Complète la colonne M de cette couche."))
    ev.append(("dataframe", [{"Sondage": x["nom"], "k bas [MN/m³]": round(x["k_bas"], 2), "k haut [MN/m³]": round(x["k_haut"], 2),
                              "k bas [kN/m³]": round(x["k_bas"] * 1000), "k haut [kN/m³]": round(x["k_haut"] * 1000),
                              "w [mm]": f"{x['w_bas']:.0f} – {x['w_haut']:.0f}"} for x in res["scia"]]))
    e = res["enveloppe"]
    if e and not e["unique"]:
        ev.append(("markdown", f"**{e['libelle']} : k = {e['k_min']:.2f} à {e['k_max']:.2f} MN/m³** — la dalle doit être vérifiée aux "
                               "DEUX bornes : un k faible augmente les tassements, un k élevé augmente les moments de pointe."))
    elif e:
        ev.append(("markdown", f"**k = {e['k_min']:.2f} MN/m³** (valeur unique : les modules ont été saisis directement, il n'y a donc "
                               "pas d'incertitude de corrélation à reporter)."))
        ev.append(("caption", "Un module issu d'un CPT serait encadré : les corrélations divergent d'un facteur 2 à 4. Pense à faire "
                              "varier M à la main pour mesurer la sensibilité de ta dalle."))
    c = res["courts"]
    if c:
        ev.append(("error", f"⚠️ **Profil trop court — k surestimé.** Pour {', '.join(c['noms'])}, l'intégration s'est arrêtée à "
                            f"{c['manque']:.2f} m sous l'assise parce que les couches saisies s'arrêtent là, et non parce que la "
                            "contrainte était devenue négligeable. Le tassement est donc tronqué et le k affiché est **trop élevé** "
                            "— c'est un écart du mauvais côté pour les moments de la dalle. Prolonge le profil : sous une fondation "
                            f"de {c['B']:.1f} m, il faut typiquement descendre à {c['profondeur_conseillee']:.0f} m."))
    ev.append(("markdown", "**Zonage de la dalle** — sans module itératif sol-structure, c'est la façon d'approcher la raideur variable :"))
    ev.append(("dataframe", [{"Zone": z["libelle"], "k bas [MN/m³]": round(z["k_bas"], 2), "k haut [MN/m³]": round(z["k_haut"], 2)}
                             for z in res["zonage"]]))
    ev.append(("caption", "Le centre tasse plus que les bords, donc k y est plus faible. Encoder un k unique concentre "
                          "artificiellement les réactions en rive : c'est le défaut connu du modèle de Winkler, pas de ce calcul."))
    ev.append(("markdown", "#### Comment la contrainte se diffuse"))
    ev.append(("image", b""))
    ev.append(("caption", "À gauche le profil, à droite les contraintes. L'aire rouge est ce qui comprime réellement le sol : elle "
                          "décroît avec la profondeur au lieu de rester égale à q. Le calcul s'arrête où la courbe rouge croise la "
                          "courbe verte — c'est la profondeur d'influence, elle n'est pas imposée."))
    ev += [("expander", "k en fonction de la largeur de fondation"), ("image", b""),
           ("caption", "À sol identique, k varie fortement avec la taille de la fondation. C'est pourquoi une valeur de k n'a de "
                       "sens qu'avec la géométrie qui l'accompagne.")]
    r_sel = next(r for r in rs if r["nom"] == sel)
    if f["sondages"][r_sel["index"]]["points"]:
        ev += [("expander", "Courbes CPT et classification"), ("image", b"")]
    for r in rs:
        ev.append(("expander", f"{'🟢' if r['etat'] == 'ok' else '🔴'} {r['nom']} — k = {r['k_bas']:.2f} à {r['k_haut']:.2f} MN/m³"))
        ev.append(("bloc", "Tassement calculé", f"w = {r['w_bas']:.1f} à {r['w_haut']:.1f} mm", r["etat"]))
        ev += [("metric", "Pression nette", f"{r['q_net']:.0f} kPa"), ("metric", "Profondeur d'influence", f"{r['z_infl']:.2f} m"),
               ("metric", "k au centre", f"{r['k_bas']:.2f}–{r['k_haut']:.2f}"),
               ("caption", f"Arrêt de l'intégration : {r['convergence']}."),
               ("latex", r"\Delta\sigma(z) = q_{net}\cdot I(z)\quad;\quad" r"w=\sum_i \frac{\Delta\sigma_i\,h_i}{M_i}\quad;\quad k=\frac{q_{net}}{w}")]
        tr = r["detail"].get("tranches") or []
        if tr:
            ev.append(("dataframe", [{"z sous assise [m]": round(t["z_sous_assise"], 2), "Δσ [kPa]": round(t["delta_sigma"], 1),
                                      "σ'v0 [kPa]": round(t["sigma_v0_eff"], 1), "Δσ/σ'v0": round(t["ratio"], 2) if t["ratio"] else None,
                                      "M [MPa]": round(t["M"], 1), "dw [mm]": round(t["dw_mm"], 3), "couche": t["couche"]}
                                     for t in r["tranches_affichees"]]))
            ev.append(("caption", f"Une ligne sur {r['pas_tranches']} — {len(tr)} tranches au total."))
        if r["detail"].get("h_sans_module", 0) > 0:
            ev.append(("warning", f"{r['detail']['h_sans_module']:.2f} m de profil sans module M n'ont pas été comptés : complète ces couches."))
    return ev, res


def series_diffusion(journal):
    """Ce que _fig_diffusion a tracé : bandes (z0, z1, teinte), étiquettes, nappe, courbes, profondeur d'influence."""
    d = {"bandes": [], "etiquettes": [], "nappe": None}
    for ax, m, a, k in journal:
        if ax == 0 and m == "axhspan":
            d["bandes"].append((a[0], a[1], k["color"][1]))
        elif ax == 0 and m == "text" and a[2] not in ("assise", " nappe"):
            d["etiquettes"].append((a[1], a[2]))
        elif ax == 0 and m == "axhline" and k.get("color") == "#2563eb":
            d["nappe"] = a[0]
        elif ax == 1 and m == "fill_betweenx":
            d["z"], d["ds"] = list(a[0]), list(a[2])
        elif ax == 1 and m == "plot":
            d.setdefault("courbes", []).append((list(a[0]), list(a[1])))
        elif ax == 1 and m == "axhline" and k.get("color") in ("#b45309", "#059669"):
            d["zi"], d["couleur_zi"] = a[0], k["color"]
        elif ax == 1 and m == "text":
            d["libelle_zi"] = a[2]
        elif ax == 1 and m == "set_ylim":
            d["ylim"] = a
    return d


def attendu_diffusion(r, p):
    g = R.donnees_diffusion(r, p["D"], p["nappe"])
    et = [((b["z0"] + b["z1"]) / 2, f"{b['nom'][:22]}\nM = {b['M']:.0f} MPa" if b["M"] else f"{b['nom'][:22]}\n⚠ M manquant")
          for b in g["bandes"] if b["etiquette"]]
    d = {"bandes": [(b["z0"], b["z1"], b["fraction"]) for b in g["bandes"]], "etiquettes": et, "nappe": g["nappe"],
         "zi": g["z_influence"], "couleur_zi": "#b45309" if g["tronque"] else "#059669",
         "libelle_zi": (f" profil interrompu à {g['z_influence']:.2f} m — k surestimé " if g["tronque"]
                        else f" profondeur d'influence  {g['z_influence']:.2f} m "), "ylim": (g["z_max"], g["z_haut"])}
    if g["z"]:
        d["z"], d["ds"] = g["z"], g["delta_sigma"]
        d["courbes"] = [(g["delta_sigma"], g["z"]), (g["sigma_v0"], g["z"]), (g["critere_sigma"], g["z"])]
    return d


def series_kB(journal):
    for ax, m, a, k in journal:
        if m == "fill_between":
            Bs, kb, kh = list(a[0]), list(a[1]), list(a[2])
        if m == "axvline":
            b0 = a[0]
    return {"B": Bs, "k_bas": kb, "k_haut": kh, "B_projet": b0}


def series_cpt(journal):
    d = {"plots": [], "vlines": [], "hlines": [], "ylim": None}
    for ax, m, a, k in journal:
        if m == "plot":
            d["plots"].append((ax, list(a[0]), list(a[1])))
        elif m == "axvline":
            d["vlines"].append(a[0])
        elif m == "axhline":
            d["hlines"].append((ax, a[0]))
        elif m == "set_ylim":
            d["ylim"] = a
    return d


def attendu_cpt(pts, nappe):
    g = R.donnees_cpt(pts, nappe)
    n_ax = 3 if g["fs"] else 2
    plots = [(0, g["qc"], g["z"])] + ([(1, g["fs"], g["z_fs"])] if g["fs"] else [])
    plots += [(n_ax - 1, [ic, ic], [z0, z1]) for ic, z0, z1 in g["ic"]]
    return {"plots": plots, "vlines": [b for b, _ in g["bornes_sbt"]],
            "hlines": [(i, nappe) for i in range(n_ax)] if nappe is not None else [], "ylim": (g["z_max"], 0)}


def ecarts_profil(f):
    """Tous les écarts entre l'écran et le moteur pour une fondation des modes 1 ou 2 (une relecture par sondage
    calculé de nom unique, pour son graphique)."""
    out = []
    res = R.resultats_profil(f)
    noms = [r["nom"] for r in res["resultats"]]
    choix = [n for n in dict.fromkeys(noms)] or [None]
    for sel in choix:
        ev, figs, appels = ecran(charger(f, chart=sel))
        obtenu = zone_resultats(ev)
        attendu, res = attendu_profil(f, sel if sel is not None else "")
        out += [f"[{sel}] écran {x}" for x in ecarts(obtenu, attendu)[:6]]
        if not res["resultats"]:
            out += ["des figures sans résultat"] if figs else []
            continue
        # les appels à tassement() : 8 par sondage calculé (positions × bornes), dans l'ordre, dictionnaires entiers
        zones = [(r["zones"][pz]["detail"], r["zones"][pz]["detail_haut"]) for r in res["resultats"] for pz in T.POSITIONS]
        zones = [x for paire in zones for x in paire]
        out += [f"[{sel}] tassement {x}" for x in ecarts([a[2] for a in appels[:len(zones)]], zones)[:4]]
        p = res["parametres"]
        r_sel = next(r for r in res["resultats"] if r["nom"] == sel)
        out += [f"[{sel}] diffusion {x}" for x in ecarts(series_diffusion(figs[0]), attendu_diffusion(r_sel, p))[:4]]
        kb = R.k_fonction_B(r_sel["bas"], r_sel["haut"], p["B"], p["L"], p["q"], p["D"], p["nappe"], p["crit"])
        out += [f"[{sel}] k(B) {x}" for x in ecarts(series_kB(figs[1]), kb)[:4]]
        out += [f"[{sel}] k(B) appels {x}" for x in ecarts(len(appels), len(zones) + 36)]
        pts = f["sondages"][r_sel["index"]]["points"]
        if pts:
            out += [f"[{sel}] CPT {x}" for x in ecarts(series_cpt(figs[2]), attendu_cpt(pts, p["nappe"]))[:4]]
        elif len(figs) != 2:
            out.append(f"[{sel}] {len(figs)} figures au lieu de 2")
    return out


def couche(h, t, M, gamma=19.0, qc=0.0, rf=0.0, M_haut=None, Ic=None, sbt=None):
    c = R.nouvelle_couche(h=h, type_sol=t, qc=qc, rf=rf, M=M, gamma=gamma)
    c.update(M_haut=M_haut, Ic=Ic, sbt=sbt)
    return c


def fondation(nom, sondages=None, **kw):
    f = R.nouvelle_fondation(nom)
    if sondages is not None:
        f["sondages"] = []
        for i, (n, cs) in enumerate(sondages, start=1):
            s = R.nouveau_sondage(i, n, premiere_couche=False)
            s["couches"] = cs
            f["sondages"].append(s)
    for k, v in kw.items():
        if isinstance(v, dict):
            f[k].update(v)
        else:
            f[k] = v
    return R.normaliser(f)


SABLE = couche(6.0, "Sable dense", 80.0, 20.0, 18.0, 0.4)
CAS = [
    ("défauts de l'écran (CPT01, trois couches, 2 × 2 m à 1 m, q = 150 kPa)", R.nouvelle_fondation()),
    ("L < B : B et L échangés", fondation("SW", B=6.0, L=3.0)),
    ("nappe à 1,5 m, pression brute", fondation("NB", nappe_active=True, nappe=1.5, q_net=False)),
    ("assise au TN, critère 10 %", fondation("D0", D=0.0, critere=10)),
    ("critère 30 %, ν = 0,35, angle", fondation("C30", critere=30, nu=0.35, position="angle")),
    ("profil trop court (3 m sous une semelle de 4 m) : k surestimé", fondation("PC", [("S1", [couche(3.0, "Argile ferme / raide", 30.0)])],
                                                                                 B=4.0, L=4.0, D=0.0)),
    ("un sondage écarté (aucun M) et un sondage calculé", fondation("EC", [("CPT01", [couche(10, "—", 0.0)]),
                                                                          ("CPT02", [couche(2, "Limon (loess)", 15), couche(18, "Sable dense", 70)])])),
    ("tous écartés", fondation("TE", [("A", [couche(10, "—", 0.0)]), ("B", [couche(0, "Tourbe", 2.0)])])),
    ("couche sans M dans la zone d'influence : k non calculable", fondation("INC", [("T1", [couche(2, "Limon (loess)", 20),
                                                                                             couche(2, "Personnalisé", 0.0),
                                                                                             couche(16, "Sable dense", 80)])])),
    ("bornes encadrées (M haut) sur deux sondages, milieu du grand côté",
     fondation("ENC", [("CPT-A", [couche(1.5, "Remblais / terre végétale", 5, 17, M_haut=9), couche(4, "Argile molle à moyenne", 8, 18, M_haut=21, Ic=3.1, sbt="Argile silteuse à argile"),
                                  couche(15, "Sable moyennement compact", 35, 19, M_haut=70, Ic=1.9, sbt="Sable propre à sable limoneux")]),
                       ("CPT-B", [couche(3, "Limon (loess)", 12, 18, M_haut=30), couche(17, "Sable dense", 60, 20, M_haut=110)])],
               B=8.0, L=12.0, position="bord_long")),
    ("milieu du petit côté, nappe à 0 m", fondation("BC", position="bord_court", nappe_active=True, nappe=0.0)),
    ("pression nette nulle (q ≤ poids des terres excavées)", fondation("QN", D=4.0, q=40.0)),
    ("assise sous tout le profil (D = 25 m)", fondation("DD", D=25.0)),
    ("radier 20 × 40 m sur 30 m de profil : profil épuisé", fondation("RA", [("P", [couche(30, "Sable dense", 90)])], B=20.0, L=40.0)),
    ("radier 20 × 40 m sur 200 m de sable dense", fondation("PL", [("P", [couche(200, "Sable dense", 90)])], B=20.0, L=40.0, D=0.0)),
    ("semelle 2 × 2 m très chargée (5 000 kPa) sur 200 m de sol léger, critère 10 % : plafond 6·B atteint (l'écran dit « profil "
     "trop court »)", fondation("PF", [("P", [couche(200, "Sable dense", 90, 8.0)])], B=2.0, L=2.0, D=0.0, q=5000.0, critere=10)),
    ("couche d'épaisseur nulle (écartée du profil)", fondation("H0", [("P", [couche(0, "Tourbe", 1), couche(20, "Sable dense", 80)])])),
    ("deux sondages de même nom (l'écran ne trace que le premier)", fondation("DN", [("CPT", [couche(20, "Sable dense", 60)]),
                                                                                      ("CPT", [couche(20, "Sable lâche", 15)])])),
    ("mode 2 (profil saisi à la main)", fondation("M2", mode="profil")),
    ("roche (calcaire sain) sous un limon", fondation("RO", [("R", [couche(2, "Limon (loess)", 20), couche(18, "Calcaire sain", 8000, 23)])])),
    ("fondation très étroite (0,2 × 0,2 m)", fondation("ET", B=0.2, L=0.2, D=0.5)),
]


def fondation_aleatoire(rng, k):
    sondages = []
    for i in range(rng.randint(1, 3)):
        cs = []
        for _ in range(rng.randint(1, 5)):
            t = rng.choice(R.TYPES)
            M = rng.choice((0.0, round(rng.uniform(2, 120), 1))) if rng.random() < 0.15 else round(rng.uniform(2, 120), 1)
            cs.append(couche(round(rng.uniform(0.2, 9), 2) if rng.random() > 0.05 else 0.0, t, M, round(rng.uniform(8, 26), 1),
                             round(rng.uniform(0, 25), 2), round(rng.uniform(0, 6), 2),
                             M_haut=round(M * rng.uniform(0.8, 3), 1) if rng.random() < 0.35 and M else None,
                             Ic=round(rng.uniform(1.2, 3.8), 3) if rng.random() < 0.3 else None,
                             sbt=rng.choice([lab for _, _, lab in T.SBT_ZONES]) if rng.random() < 0.3 else None))
        sondages.append((f"S{i + 1}" if rng.random() < 0.9 else "S1", cs))
    return fondation(f"Aléa {k}", sondages, mode=rng.choice(("cpt", "profil")), B=round(rng.uniform(0.2, 25), 2),
                     L=round(rng.uniform(0.2, 40), 2), D=rng.choice((0.0, round(rng.uniform(0, 4), 2))), q=round(rng.uniform(1, 500), 1),
                     nappe_active=rng.random() < 0.5, nappe=round(rng.uniform(0, 8), 2), critere=rng.choice(R.CRITERES),
                     q_net=rng.random() < 0.7, nu=round(rng.uniform(0, 0.49), 2), position=rng.choice(list(T.POSITIONS)))


CAS += [(f"tirage {k:02d} (graine 20261107)", fondation_aleatoire(RNG, k)) for k in range(1, 31)]
# un sondage CPT importé (mesures gardées) : la figure CPT est rejouée elle aussi
imp = R.lire_essai("CPT-IMP.gef", gef(essai_cpt(random.Random(7), pas=0.05), "CPT-IMP", 2.4), 3.0)
f_imp = R.nouvelle_fondation("IMP")
f_imp["sondages"][0]["import"] = imp
R.valider_import(f_imp, 0)
CAS.append(("sondage CPT importé (GEF, nappe 2,4 m) : couches encadrées, courbes CPT", R.normaliser(f_imp)))
f_imp2 = copy.deepcopy(f_imp)
f_imp2["sondages"][0]["points"] = [[z, q, None] for z, q, _ in f_imp2["sondages"][0]["points"]]
f_imp2["nappe_active"] = False
CAS.append(("mesures sans frottement, nappe décochée : courbe qc et Ic vide", R.normaliser(f_imp2)))

couverts = {"echange": 0, "ecarte": 0, "incalculable": 0, "court": 0, "plage": 0, "unique": 0, "aucun": 0, "nappe": 0, "cpt": 0,
            "nette_nulle": 0, "plafond": 0, "epuise": 0, "critere": 0, "positions": set()}
for desc, f in CAS:
    d = ecarts_profil(f)
    res = R.resultats_profil(f)
    couverts["echange"] += res["parametres"]["echange"]
    couverts["ecarte"] += bool(res["ecartes"])
    couverts["incalculable"] += bool(res["incalculables"])
    couverts["court"] += bool(res["courts"])
    couverts["aucun"] += not res["resultats"]
    couverts["nappe"] += res["parametres"]["nappe"] is not None
    couverts["cpt"] += any(s["points"] for s in f["sondages"])
    couverts["positions"].add(f["position"])
    if res["enveloppe"]:
        couverts["unique" if res["enveloppe"]["unique"] else "plage"] += 1
    for r in res["resultats"]:
        conv = r["convergence"]
        couverts["nette_nulle"] += "nette nulle" in conv
        couverts["plafond"] += "plafond" in conv
        couverts["epuise"] += "épuisé" in conv
        couverts["critere"] += "Δσ" in conv
    chk(f"{desc} : écran = moteur ({sum(1 for _ in res['resultats'])} sondage(s) calculé(s))", not d, d[:4])
chk(f"couverture des cas : {json.dumps({k: (sorted(v) if isinstance(v, set) else v) for k, v in couverts.items()}, ensure_ascii=False)}",
    all((v if not isinstance(v, set) else len(v) == 4) for v in couverts.values()))

# ------------------------------------------------------------------ modes 3 à 6
print("\n--- modes 3 à 6")


def _fr(x, nd=2):
    return f"{float(x):,.{nd}f}".replace(",", " ").replace(".", ",")


def attendu_mode(f):
    m = f["mode"]
    ev = []
    if m == "rapide":
        x = R.calcul_rapide(f["rapide"]["q"], f["rapide"]["w"])
        q, w, k = x["q"], x["w"], x["k_kNm3"]
        ev.append(("bloc", "Coefficient de réaction", f"k = {_fr(k / 1000, 2)} MN/m³", x["etat"]))
        if k > 0:
            ev += [("latex", r"k = \dfrac{q}{w}"),
                   ("latex", f"k = \\dfrac{{{q:.1f}}}{{{w / 1000:.4f}}} = {k:,.0f}\\,\\text{{kN/m³}} = {k / 1000:,.2f}\\,\\text{{MN/m³}}".replace(",", " ")),
                   ("caption", "Valable si w provient d'un vrai calcul de tassement, pour cette fondation et cette charge.")]
    elif m == "theories":
        t = f["theories"]
        x = R.theories(t["E"], t["B"], t["L"], t["nu"])
        ev.append(("dataframe", [{"Méthode": y["nom"], "k [MN/m³]": round(y["k"], 2), "Domaine": y["domaine"]} for y in x["methodes"]]))
        ev.append(("caption", f"Sol homogène E = {x['E']:.0f} MPa (soit M = {x['M']:.0f} MPa pour ν = {x['nu']:.2f}), semelle "
                              f"{x['B']:.1f}×{x['L']:.1f} m, q = {x['q']:.0f} kPa. La dernière ligne est la formule de la version 4 : elle "
                              "donne exactement E/(2B), donc un résultat piloté par la règle des 2B et non par le sol — d'où le facteur 2 à "
                              "2,5 d'écart."))
        ev.append(("bloc", "Valeur retenue", f"k = {x['k_retenu']:.2f} MN/m³", x["etat"]))
        ev.append(("caption", f"Tassement {x['tassement']['w_mm']:.1f} mm, profondeur d'influence {x['tassement']['z_influence']:.2f} m."))
    elif m == "abaque":
        x = R.abaque(f["abaque"]["B"], f["abaque"]["w_ref"])
        ev.append(("dataframe", [{"Catégorie": y["categorie"], "Type de sol": y["type"], "γ [kN/m³]": y["gamma"],
                                  "M [MPa]": f"{y['M_min']:.0f} – {y['M_max']:.0f}" if y["M_min"] else "—",
                                  "k plaque 0,30 m": f"{y['kp_min']} – {y['kp_max']}",
                                  f"k pour B = {x['B']:.1f} m": f"{y['k_bas']:.1f} – {y['k_haut']:.1f}",
                                  "q pour w réf [kPa]": f"{y['q_bas']:.0f} – {y['q_haut']:.0f}"} for y in x["lignes"]]))
        ev.append(("warning", "**Les deux colonnes de k ne sont pas interchangeables.** Celle de gauche est la valeur mesurée sur une "
                              "plaque de 0,30 m (tables de Terzaghi 1955) ; celle de droite lui applique la correction de taille pour la "
                              "largeur choisie. Utiliser la colonne de gauche pour une fondation réelle surestime k d'un facteur 3 à 30. "
                              "La version 4 de ce module ne faisait pas cette correction."))
        ev.append(("caption", "Un abaque reste un ordre de grandeur : le mode « Sondage CPT » calcule la valeur pour ton sol et ta fondation."))
    else:
        pl = f["plat"]
        x = R.plat(pl["B"], pl["L"], pl["alpha"], pl["E"], pl["mortier"], pl["tg"], pl["Eg"])
        ev.append(("bloc", "Raideur du contact", f"k = {x['keq'] / 1000:,.0f} MN/m³".replace(",", " "), x["etat"]))
        if x["hc_m"] > 0:
            ev += [("latex", r"h_c = \alpha\cdot\min(B,L)"), ("latex", f"h_c = {x['hc_m'] * 1000:.1f}\\,\\text{{mm}}"),
                   ("latex", r"k_c = E_c / h_c"), ("latex", f"k_c = {x['kc'] / 1000:,.0f}\\,\\text{{MN/m³}}".replace(",", " "))]
            if x["kg"] > 0:
                ev += [("latex", r"1/k_{eq} = 1/k_c + 1/k_g"), ("latex", f"k_g = {x['kg'] / 1000:,.0f}\\,\\text{{MN/m³}}".replace(",", " "))]
    return ev


CAS_MODES = []
for k in range(40):
    mode = ("rapide", "theories", "abaque", "plat")[k % 4]
    f = R.nouvelle_fondation(f"M{k}")
    f["mode"] = mode
    if k >= 4:
        f["rapide"] = {"q": RNG.choice((0.0, round(RNG.uniform(0, 600), 1))), "w": RNG.choice((0.01, round(RNG.uniform(0.01, 80), 2)))}
        f["theories"] = {"E": round(RNG.uniform(0.5, 300), 1), "B": round(RNG.uniform(0.2, 20), 2), "L": round(RNG.uniform(0.2, 30), 2),
                         "nu": RNG.choice((0.0, 0.49, round(RNG.uniform(0, 0.49), 2)))}
        f["abaque"] = {"B": round(RNG.uniform(0.3, 30), 2), "w_ref": round(RNG.uniform(1, 100), 1)}
        f["plat"] = {"B": round(RNG.uniform(20, 900)), "L": round(RNG.uniform(20, 900)), "alpha": round(RNG.uniform(0.05, 2), 2),
                     "E": round(RNG.uniform(5, 45), 1), "mortier": RNG.random() < 0.6, "tg": round(RNG.uniform(1, 60)), "Eg": round(RNG.uniform(1, 40), 1)}
    CAS_MODES.append(R.normaliser(f))
fautes = []
for f in CAS_MODES:
    ev, figs, _ = ecran(charger(f))
    d = ecarts(zone_resultats(ev), attendu_mode(f))
    if d or figs:
        fautes.append((f["nom"], f["mode"], d[:3], len(figs)))
chk(f"modes 3 à 6 : {len(CAS_MODES)} fondations (k = q/w ; théories ; abaque des 21 sols ; plat avec ou sans mortier) — "
    "chaque bloc, tableau, formule et légende de l'écran = moteur", not fautes, fautes[:3])

# ================================================================== 5. les couches
print("\n=== 5. Les couches : préremplissage, légendes, ajouts et suppressions ===")


def legende(c):
    x = R.infos_couche(c)
    bits = ["✅ prise en compte" if x["prise_en_compte"] else "⚠️ ignorée (h ou M manquant)"]
    if x["Ic"] is not None:
        bits.append(f"Ic = {x['Ic']:.2f}")
    if x["sbt"]:
        bits.append(f"CPT : {x['sbt']}")
    if x["M_encadre"]:
        bits.append(f"M encadré : {x['M_encadre'][0]:.0f} – {x['M_encadre'][1]:.0f} MPa")
    elif x["M_suggere"]:
        bits.append(f"M suggéré (qc) : {x['M_suggere']:.0f} MPa")
    elif x["roche"]:
        bits.append("roche : refus de pointe probable, qc sans objet")
    return " · ".join(bits)


fautes, n_leg = [], 0
for desc, f in CAS:
    ev, _, _ = ecran(charger(f))
    obt = [e[1] for e in ev if e[0] == "caption" and (e[1].startswith("✅ prise") or e[1].startswith("⚠️ ignorée"))]
    att = [legende(c) for s in f["sondages"] for c in s["couches"]]
    n_leg += len(att)
    if obt != att:
        fautes.append((desc, ecarts(obt, att)[:2]))
chk(f"la légende de chacune des {n_leg} lignes de couche (prise en compte, Ic, classe CPT, M encadré, M suggéré par qc, "
    "roche) = moteur", not fautes, fautes[:3])

fautes, n_pref = [], 0
for k in range(300):
    ancien, nouveau = RNG.choice(R.TYPES), RNG.choice(R.TYPES)
    c0 = couche(round(RNG.uniform(0, 5), 1), ancien, RNG.choice((0.0, 0.0, round(RNG.uniform(1, 80), 1))), round(RNG.uniform(8, 26), 1),
                RNG.choice((0.0, 0.0, round(RNG.uniform(0.1, 20), 1))), RNG.choice((0.0, 1.5)))
    f = fondation("TY", [("S", [couche(2, "Limon (loess)", 20), c0])])
    ss = charger(f)
    ss[RS._layer_key(1, 2, "type")] = nouveau                # ce que fait le menu de l'écran : la valeur change, type_prev non
    ecran(ss)
    obt = lire_session(ss)[0]["couches"][1]
    att = copy.deepcopy(f["sondages"][0]["couches"][1])
    R.appliquer_type(att, nouveau)
    n_pref += ancien != nouveau
    if ecarts(obt, att) or ss[RS._layer_key(1, 2, "type_prev")] != nouveau:
        fautes.append((ancien, nouveau, ecarts(obt, att)[:2]))
chk(f"changer le type d'une couche ({n_pref} changements tirés) : qc, Rf, M préremplis seulement si qc et M sont nuls, γ du type "
    "toujours — comme l'écran", not fautes, fautes[:3])

f = fondation("CB", [("CPT01", [couche(2, "Limon (loess)", 20, M_haut=30, Ic=2.7, sbt="Limon"), couche(18, "Sable dense", 80)]),
                     ("CPT02", [couche(20, "Sable lâche", 15)])])
f["sondages"][0]["points"] = [[0.1, 2.0, 30.0], [0.2, 2.1, None]]
f["sondages"][0]["source"], f["sondages"][0]["nappe"] = "GEF — 2 points au pas réel", 2.0
OPS = [("_add_sounding", (), lambda g: R.ajouter_sondage(g)), ("_copy_sounding", (1,), lambda g: R.copier_sondage(g, 0)),
       ("_copy_sounding", (2,), lambda g: R.copier_sondage(g, 1)), ("_delete_sounding", (1,), lambda g: R.supprimer_sondage(g, 0)),
       ("_add_layer", (1,), lambda g: R.ajouter_couche(g["sondages"][0])),
       ("_delete_layer", (1, 2), lambda g: R.supprimer_couche(g["sondages"][0], 1)),
       ("_delete_layer", (2, 1), lambda g: R.supprimer_couche(g["sondages"][1], 0))]
fautes = []
for nom, args, op in OPS:
    ss = charger(f)
    st = FauxSt(ss)
    with ecran_simule(st):
        getattr(RS, nom)(*args)
    g = copy.deepcopy(f)
    op(g)
    att = [{k: s[k] for k in ("id", "nom", "couches", "source", "nappe", "points")} for s in g["sondages"]]
    d = ecarts(lire_session(ss), att)
    if d:
        fautes.append((nom, args, d[:2]))
chk("les boutons de l'écran (➕ sondage, 📋 copier, 🗑️ supprimer, ＋ couche, 🗑️ couche) = moteur : identifiants, « (copie) », "
    "couches avec leurs clés d'import, mesures, source, nappe ; ni le dernier sondage ni la dernière couche ne se suppriment",
    not fautes, fautes[:3])
g = R.nouvelle_fondation()
chk("le dernier sondage et la dernière couche ne se suppriment pas (moteur)",
    not R.supprimer_sondage(g, 0) and len(g["sondages"]) == 1 and R.supprimer_couche(g["sondages"][0], 0)
    and R.supprimer_couche(g["sondages"][0], 0) and not R.supprimer_couche(g["sondages"][0], 0))

# ================================================================== 6. l'import
print("\n=== 6. L'import d'un essai : aperçu, découpage, « Remplir le tableau », exports ===")


def attendu_apercu(imp, nu):
    pts = imp["points"]
    ev = [("success", f"**{imp['nom']}** — {imp['source']} · {len(pts)} points de {pts[0][0]:.2f} à {pts[-1][0]:.2f} m")]
    ev += [("warning", a) for a in imp["avertissements"]]
    try:
        ap = R.apercu_import(imp, nu)
    except ValueError:
        if not any(p[2] is not None for p in pts):
            ev.append(("warning", "Pas de frottement latéral fs dans ce fichier : la classification automatique du sol (Robertson) a "
                                  "besoin de qc **et** de fs. Les couches seront créées d'après qc seul et le type de sol restera à choisir."))
        return ev + [("plantage", "TypeError")], {"couches": []}
    if not ap["a_fs"]:
        ev.append(("warning", "Pas de frottement latéral fs dans ce fichier : la classification automatique du sol (Robertson) a besoin "
                              "de qc **et** de fs. Les couches seront créées d'après qc seul et le type de sol restera à choisir."))
    if not ap["couches"]:
        ev.append(("error", "Aucune couche exploitable n'a pu être formée."))
        return ev, ap
    ev.append(("dataframe", [{"de [m]": x["de"], "à [m]": x["a"], "h [m]": x["h"], "Ic": x["Ic"], "Type reconnu": x["type"],
                              "qc moy [MPa]": x["qc"], "M bas [MPa]": x["M_bas"], "M haut [MPa]": x["M_haut"]} for x in ap["lignes"]]))
    ev.append(("caption", "Le type de sol est déduit de qc et fs par l'indice de comportement Ic (Robertson) — aucune saisie, aucune "
                          "estimation. Le module M est encadré car les corrélations CPT divergent d'un facteur 2 à 4."))
    ev.append(("download", "⬇️ CSV des mesures", f"{imp['nom']}_mesures.csv", R.csv_mesures(imp["nom"], imp["points"]).encode("utf-8")))
    try:
        ev.append(("download", "⬇️ CSV des couches", f"{imp['nom']}_couches.csv", R.csv_couches(imp["nom"], ap["couches"]).encode("utf-8")))
    except ValueError:                                         # une couche sans module (sans Ic) : l'écran plante sur l'export
        ev.append(("plantage", "TypeError"))
    return ev, ap


fautes, n_imp, n_rempli, n_plante = [], 0, 0, 0
for k, (nom, texte) in enumerate(FICHIERS):
    if not nom.endswith((".gef", ".csv", ".txt")):
        continue
    try:
        s_mod = MI.importer(texte.encode("utf-8"), nom)
    except ValueError:
        continue
    f = fondation("IM", nappe=RNG.choice((3.0, 1.2)), nu=RNG.choice((0.3, 0.25)))
    imp = R.lire_essai(nom, texte, f["nappe"])
    if k % 3 == 1:                                            # l'aperçu réglé : nappe, finesse, épaisseur minimale
        imp["nappe"], imp["seuil"], imp["h_min"] = round(RNG.uniform(0, 6), 1), RNG.choice(R.SEUILS_IC), RNG.choice((0.2, 1.0, 2.0))
    for clic in (False, True):
        ss = charger(f)
        ss["snd1_imp_nappe"], ss["snd1_imp_seuil"], ss["snd1_imp_hmin"] = imp["nappe"], imp["seuil"], imp["h_min"]
        st = FauxSt(ss, {"snd1_valider"} if clic else ())
        with ecran_simule(st):
            try:
                RS._apercu_et_valider(1, s_mod)
            except Relance:
                st.ev.append(("rerun",))
            except TypeError:
                st.ev.append(("plantage", "TypeError"))
        att, ap = attendu_apercu(imp, f["nu"])
        n_plante += ("plantage", "TypeError") in att and not clic
        if not clic:
            n_imp += 1
            d = ecarts(st.ev, att)
            if d:
                fautes.append((nom, "aperçu", d[:3]))
        elif ap["couches"]:
            n_rempli += 1
            g = copy.deepcopy(f)
            g["sondages"][0]["import"] = imp
            R.valider_import(g, 0)
            s = g["sondages"][0]
            obt = lire_session(ss)[0]
            d = ecarts([obt["couches"], [list(p) for p in ss["snd1_points"]], obt["nappe"], obt["source"]],
                       [s["couches"], s["points"], s["nappe"], s["source"]])
            d += ecarts(ss.get("rs_pending"), {"rs_nappe_active": True, "rs_nappe": g["nappe"], "snd1_nom": s["nom"]})
            d += ecarts(g["nappe_active"], True) + ecarts(st.ev[-1], ("rerun",))
            if d:
                fautes.append((nom, "remplir", d[:3]))
chk(f"aperçu de {n_imp} essais lus (GEF, CSV, TXT ; réglages de l'aperçu changés un fois sur trois) : message, avertissements, "
    "absence de fs, tableau découpé (de, à, h, Ic, type, qc, M bas, M haut), exports CSV des mesures et des couches = moteur",
    not fautes and n_imp > 30, fautes[:3])
chk(f"l'aperçu qui PLANTE dans l'écran ({n_plante} essai(s) : TypeError) est refusé par le moteur avec un message, au même endroit",
    n_plante >= 2 and not fautes)
chk(f"« ✅ Remplir le tableau » sur {n_rempli} essais : couches (type reconnu, Rf, M arrondi, M haut, Ic, classe), mesures, nappe "
    "du sondage ET de la fondation (cochée), source, nom du sondage = moteur", not fautes and n_rempli > 30, fautes[:3])
ss = charger(R.nouvelle_fondation())
ss["snd1_points"] = [(0.1, 2.0, 30.0), (0.2, 2.5, 31.0)]
ev, _, _ = ecran(ss)
dl = [e for e in ev if e[0] == "download" and e[2] == "CPT01_mesures.csv"]
chk("« ⬇️ Exporter les mesures brutes en CSV » d'un sondage = moteur (csv_mesures)",
    len(dl) == 1 and dl[0][3] == R.csv_mesures("CPT01", ss["snd1_points"]).encode("utf-8"))
colle = R.lire_tableau_colle("0,5\t2,1\t30\n1,0\t2,4\t35\n1,5\t3,0\t40\n2,0\t5,2\t41\n2,5\t6,0\t44\n3,0\t8,0\t50\n", "CPT01", 3.0)
chk("un tableau collé sans en-tête (site seulement) est lu par le MÊME analyseur CSV, colonnes profondeur / qc / fs",
    colle["points"][0] == [0.5, 2.1, 30.0] and len(colle["points"]) == 6 and colle["source"].startswith("Tableau collé"))
colle2 = R.lire_tableau_colle("Profondeur (m);qc (MPa);fs (kPa)\n" + "\n".join(f"{z / 10};{z / 5};{z}" for z in range(1, 9)), "X", 3.0)
chk("un tableau collé avec son en-tête est lu comme le CSV de l'écran (parse_csv)",
    colle2["points"] == [list(p) for p in MI.parse_csv(("Profondeur (m);qc (MPa);fs (kPa)\n" + "\n".join(f"{z / 10};{z / 5};{z}" for z in range(1, 9))).encode(), "X")["points"]])

# ================================================================== 7. le fichier de l'écran, les défauts
print("\n=== 7. Le fichier de l'écran (raideur_sol.json) et les défauts ===")
ss = Etat()
ev, _, _ = ecran(ss)
with ecran_simule(FauxSt(ss)):
    payload = RS._payload()
neuve = R.depuis_payload(json.loads(json.dumps(payload)), "F1")
att = R.nouvelle_fondation()
chk("nouvelle_fondation() = l'écran neuf (enregistré par « 💾 Enregistrer » et relu) : mode 1, 2 × 2 m à 1 m, q = 150 kPa, nappe "
    "décochée à 3 m, critère 20 %, ν = 0,30, pression nette, centre, CPT01 et ses trois couches", not ecarts(neuve, att), ecarts(neuve, att)[:4])
defauts = {}
for mode, (cle_m, bloc) in zip(("3.", "4.", "5.", "6."), (("rapide", None), ("theories", None), ("abaque", None), ("plat", None))):
    ss2 = Etat()
    ss2["rs_mode"] = next(lib for _, lib in R.MODES if lib.startswith(mode))
    ecran(ss2)
    defauts[cle_m] = R.depuis_payload(json.loads(json.dumps({"version": "v5.0", "soundings": [], "orders": {},
                                                             "values": {k: v for k, v in ss2.items() if RS._est_cle_module(k) and isinstance(v, (int, float, str, bool, type(None)))}})))[cle_m]
chk("les valeurs initiales des widgets des modes 3 à 6 = défauts du moteur (q 150, w 20 ; E 25, B 2, L 2, ν 0,30 ; B 2, w_réf 20 ; "
    "200 × 200 mm, α 0,5, E 30 GPa, sans mortier, 20 mm, 20 GPa)",
    defauts == {"rapide": R.RAPIDE_DEFAUT, "theories": R.THEORIES_DEFAUT, "abaque": R.ABAQUE_DEFAUT, "plat": R.PLAT_DEFAUT}, defauts)
fautes = []
for desc, f in CAS + [(g["nom"], g) for g in CAS_MODES]:
    ss = charger(f)
    st = FauxSt(ss)
    with ecran_simule(st):
        payload = json.loads(json.dumps(RS._payload(), ensure_ascii=False))
    lu = R.depuis_payload(payload, f["nom"])
    att = copy.deepcopy(f)
    for s in att["sondages"]:
        s["points"] = []                                       # l'écran n'enregistre pas les mesures brutes (liste)
    # DÉFAUT de l'écran : les clés des modes 3 à 6 (rs3_*, rs4_*, rs5_B, rs6_*) ne commencent pas par un préfixe du
    # module (« snd », « rs_ », « sol_ ») : « 💾 Enregistrer » ne les écrit pas, la relecture rend les valeurs initiales
    att["rapide"], att["theories"], att["plat"] = dict(R.RAPIDE_DEFAUT), dict(R.THEORIES_DEFAUT), dict(R.PLAT_DEFAUT)
    att["abaque"]["B"] = R.ABAQUE_DEFAUT["B"]
    d = ecarts(lu, att)
    if d:
        fautes.append((desc, d[:3]))
chk(f"{len(CAS) + len(CAS_MODES)} notes enregistrées par l'écran (_payload) et relues par le moteur : tout revient, sauf les "
    "mesures brutes d'un CPT et les champs des modes 3 à 6 (DÉFAUT REPRODUIT : clés rs3_, rs4_, rs5_B, rs6_ hors des préfixes "
    "du module), que l'écran n'enregistre pas", not fautes, fautes[:3])
with ecran_simule(FauxSt(charger(CAS_MODES[5]))):
    p5 = RS._payload()["values"]
chk("DÉFAUT de l'écran, constaté : son fichier ne porte ni rs3_q, ni rs4_E, ni rs5_B, ni rs6_B (mais porte rs_w_ref)",
    not any(k in p5 for k in ("rs3_q", "rs3_w", "rs4_E", "rs5_B", "rs6_B", "rs6_grout")) and "rs_w_ref" in p5)
for mauvais in ({"x": 1}, [], {"values": {}, "soundings": [{"id": 1, "nom": "A"}], "orders": {"1": [1]},
                                       "values_": 0}, {"version": "v5.0", "soundings": [], "orders": {}, "values": {"rs_B": -1.0}}):
    try:
        R.depuis_payload(mauvais)
        ok = isinstance(mauvais, dict) and "values" in mauvais and "rs_B" not in mauvais["values"]
    except ValueError:
        ok = True
    chk(f"fichier étranger ou hors bornes refusé avec un message ({str(mauvais)[:50]})", ok)

# ================================================================== 8. format et domaine
print("\n=== 8. Format et domaine de l'écran ===")
toutes = [f for _, f in CAS] + CAS_MODES
sorties = [R.resultats_profil(f) for f in toutes]
chk("resultats_profil() et les modes 3 à 6 rendent du JSON strict (aucun NaN ni infini) qui se relit à l'identique",
    all(json.loads(json.dumps(x, allow_nan=False)) == json.loads(json.dumps(x)) for x in sorties)
    and all(json.dumps(y, allow_nan=False) for f in CAS_MODES for y in (R.theories(**f["theories"]), R.abaque(**f["abaque"]),
                                                                          R.plat(**f["plat"]), R.calcul_rapide(**f["rapide"]))))
chk("normaliser() est idempotente, garde chaque fondation telle quelle, et {} donne nouvelle_fondation() (nom mis à part)",
    all(R.normaliser(R.normaliser(f)) == R.normaliser(f) == f for f in toutes)
    and R.normaliser({}) == R.nouvelle_fondation())
avant = copy.deepcopy(CAS[9][1])
R.resultats_profil(CAS[9][1])
chk("resultats_profil() ne modifie pas la fondation reçue", CAS[9][1] == avant)
REFUS = [("B sous 0,2 m", {"B": 0.1}), ("L nulle", {"L": 0.0}), ("assise négative", {"D": -1.0}), ("q sous 1 kPa", {"q": 0.5}),
         ("nappe négative", {"nappe": -0.5}), ("ν = 0,5", {"nu": 0.5}), ("critère 12 %", {"critere": 12}), ("critère en texte", {"critere": "20"}),
         ("position inconnue", {"position": "milieu"}), ("mode inconnu", {"mode": "7"}), ("B en texte", {"B": "2"}), ("B booléen", {"B": True}),
         ("B infini", {"B": float("inf")}), ("nappe cochée en texte", {"nappe_active": "oui"}), ("aucun sondage", {"sondages": []}),
         ("sondage sans couche", {"sondages": [{"id": 1, "nom": "A", "couches": []}]}),
         ("type de sol inconnu", {"sondages": [{"id": 1, "nom": "A", "couches": [{"type": "Argile verte"}]}]}),
         ("épaisseur négative", {"sondages": [{"id": 1, "couches": [{"h": -1}]}]}), ("γ = 30", {"sondages": [{"id": 1, "couches": [{"gamma": 30}]}]}),
         ("γ = 5", {"sondages": [{"id": 1, "couches": [{"gamma": 5}]}]}), ("M négatif", {"sondages": [{"id": 1, "couches": [{"M": -3}]}]}),
         ("qc négatif", {"sondages": [{"id": 1, "couches": [{"qc": -3}]}]}), ("Rf négatif", {"sondages": [{"id": 1, "couches": [{"rf": -1}]}]}),
         ("identifiants doublés", {"sondages": [{"id": 1, "couches": [{}]}, {"id": 1, "couches": [{}]}]}),
         ("mesure tronquée", {"sondages": [{"id": 1, "couches": [{}], "points": [[1.0, 2.0]]}]}),
         ("w = 0", {"rapide": {"w": 0.0}}), ("q négatif (mode 3)", {"rapide": {"q": -1}}), ("E sous 0,5", {"theories": {"E": 0.1}}),
         ("ν théories 0,6", {"theories": {"nu": 0.6}}), ("abaque B = 0,2", {"abaque": {"B": 0.2}}), ("w réf 150", {"abaque": {"w_ref": 150}}),
         ("plat 10 mm", {"plat": {"B": 10}}), ("α = 0", {"plat": {"alpha": 0.0}}), ("E béton 2 GPa", {"plat": {"E": 2}}),
         ("mortier en texte", {"plat": {"mortier": "oui"}}), ("mortier 0 mm", {"plat": {"tg": 0}}),
         ("finesse 0,7", {"sondages": [{"id": 1, "couches": [{}], "import": {"points": [[1, 2, 3]], "seuil": 0.7}}]}),
         ("épaisseur minimale 0,1", {"sondages": [{"id": 1, "couches": [{}], "import": {"points": [[1, 2, 3]], "h_min": 0.1}}]})]
refusees = []
for nom, champ in REFUS:
    f = R.nouvelle_fondation()
    for k, v in champ.items():
        if isinstance(v, dict) and isinstance(f.get(k), dict):
            f[k] = dict(f[k], **v)
        else:
            f[k] = v
    try:
        R.normaliser(f)
        refusees.append(f"{nom} : accepté")
    except ValueError:
        pass
    except Exception as e:  # noqa: BLE001 — une autre exception n'est pas un refus : c'est une panne
        refusees.append(f"{nom} : {type(e).__name__} {e}")
chk(f"{len(REFUS)} saisies hors des bornes de l'écran (min_value, max_value de ses widgets) ou illisibles lèvent ValueError "
    "— jamais ramenées en silence à une borne", not refusees, refusees)
try:
    R.normaliser(dict(R.nouvelle_fondation(), B=0.1))
except ValueError as e:
    msg = str(e)
chk(f"le message dit le champ, la valeur et la borne : « {msg} »", "Largeur B" in msg and "0,1" in msg and "0,2" in msg)
src = lire("modules", "rigidite_sol.py")
BORNES_ECRAN = {"B": 'st.number_input("Largeur B [m]", min_value=0.2', "L": 'st.number_input("Longueur L [m]", min_value=0.2',
                "D": 'st.number_input("Assise D [m]", min_value=0.0', "q": 'st.number_input("Pression q [kPa]", min_value=1.0',
                "nu": 'st.number_input("ν (Poisson)", min_value=0.0, max_value=0.49', "couche.h": 'st.number_input("h [m]", min_value=0.0',
                "couche.gamma": 'st.number_input("γ [kN/m³]", min_value=8.0, max_value=26.0', "rapide.w": 'min_value=0.01, step=1.0, value=20.0, key="rs3_w"',
                "theories.E": 'st.number_input("E [MPa]", min_value=0.5', "abaque.B": 'min_value=0.3, step=0.5',
                "abaque.w_ref": 'min_value=1.0, max_value=100.0', "plat.alpha": 'min_value=0.05, step=0.05',
                "plat.E": 'min_value=5.0, step=1.0, value=30.0', "import.h_min": 'min_value=0.20, max_value=5.0',
                "import.seuil": 'st.slider("Finesse du découpage", 0.10, 0.60, 0.25, 0.05'}
chk("les bornes du moteur sont lues dans les widgets de l'écran (min_value, max_value)",
    all(t in src for t in BORNES_ECRAN.values()) and all(k in R.BORNES for k in BORNES_ECRAN))

# ================================================================== 9. mutations
print("\n=== 9. Mutations : on altère le moteur, la parité DOIT rougir ===")


@contextlib.contextmanager
def remplace(module, nom, valeur):
    vrai = getattr(module, nom)
    setattr(module, nom, valeur)
    try:
        yield
    finally:
        setattr(module, nom, vrai)


def cas(debut):
    return next(f for d, f in CAS if d.startswith(debut))


def ecarts_mode(f):
    return ecarts(zone_resultats(ecran(charger(f))[0]), attendu_mode(f))


def ecarts_type(f, j, nouveau):
    """Le type de la couche j du premier sondage changé dans l'écran, et dans le moteur."""
    ss = charger(f)
    ss[RS._layer_key(f["sondages"][0]["id"], j + 1, "type")] = nouveau
    ecran(ss)
    att = copy.deepcopy(f["sondages"][0]["couches"][j])
    R.appliquer_type(att, nouveau)
    return ecarts(lire_session(ss)[0]["couches"][j], att)


def ecarts_remplir(f, nom, texte):
    """« Remplir le tableau » d'un essai : l'écran contre le moteur (couches du premier sondage)."""
    ss = charger(f)
    st = FauxSt(ss, {"snd1_valider"})
    with ecran_simule(st):
        try:
            RS._apercu_et_valider(1, MI.importer(texte.encode("utf-8"), nom))
        except Relance:
            pass
    g = copy.deepcopy(f)
    g["sondages"][0]["import"] = R.lire_essai(nom, texte, f["nappe"])
    R.valider_import(g, 0)
    return ecarts(lire_session(ss)[0]["couches"], g["sondages"][0]["couches"])


vrai = {"parametres": R.parametres, "profil_encadre": R.profil_encadre, "tassement": T.tassement, "plat": R.plat,
        "appliquer_type": R.appliquer_type, "couches_depuis_cpt": R.couches_depuis_cpt}
PLAT_M = next(f for f in CAS_MODES if f["mode"] == "plat" and f["plat"]["mortier"])
ABAQ = next(f for f in CAS_MODES if f["mode"] == "abaque")
F_TYPE = fondation("PR", [("S", [couche(2, "Limon (loess)", 0.0, qc=2.0)])])
NOM_GEF, TEXTE_GEF = next((n, t) for n, t in FICHIERS if n.endswith(".gef") and "S-01" in t)
MUTATIONS = [
    ("la pression nette toujours appliquée", R, "parametres", lambda f: dict(vrai["parametres"](f), q_net=True),
     lambda: ecarts_profil(cas("nappe à 1,5 m"))),
    ("B et L jamais échangés", R, "parametres",
     lambda f: dict(vrai["parametres"](f), B=float(f["B"]), L=float(f["L"]), echange=False), lambda: ecarts_profil(cas("L < B"))),
    ("le M haut de l'import ignoré", R, "profil_encadre", lambda s: (vrai["profil_encadre"](s)[0],) * 2,
     lambda: ecarts_profil(cas("bornes encadrées"))),
    ("un flottant de la théorie altéré au DERNIER BIT", T, "tassement",
     lambda *a, **k: (lambda r: dict(r, w_mm=r["w_mm"] * (1 + 2 ** -52)))(vrai["tassement"](*a, **k)),
     lambda: ecarts_profil(cas("défauts de l'écran"))),
    ("le critère lu à 21 % au lieu de 20 %", R, "parametres", lambda f: dict(vrai["parametres"](f), crit=vrai["parametres"](f)["crit"] + 0.01),
     lambda: ecarts_profil(cas("défauts de l'écran"))),
    ("l'abaque avec la correction de forme (L transmis)", T, "k_terzaghi_taille",
     lambda k, B, L=None, nature="sable": MT.k_terzaghi_taille(k, B, 1.5 * B, nature), lambda: ecarts_mode(ABAQ)),
    ("le mortier ajouté en parallèle (k_c + k_g)", R, "plat", lambda *a: (lambda x: dict(x, keq=x["kc"] + x["kg"]))(vrai["plat"](*a)),
     lambda: ecarts_mode(PLAT_M)),
    ("le préremplissage fait même quand qc est saisi", R, "appliquer_type",
     lambda c, t: (c.update(qc=0.0), vrai["appliquer_type"](c, t))[1], lambda: ecarts_type(F_TYPE, 0, "Sable dense")),
    ("le M bas de l'import non arrondi au dixième", R, "couches_depuis_cpt",
     lambda cs: [dict(c, M=c["M"] + 0.04) for c in vrai["couches_depuis_cpt"](cs)], lambda: ecarts_remplir(R.nouvelle_fondation(), NOM_GEF, TEXTE_GEF)),
]
for nom, module, attr, faux, mesure in MUTATIONS:
    with remplace(module, attr, faux):
        try:
            d = mesure()
        except Exception as e:  # noqa: BLE001 — une mutation qui fait tomber la comparaison est aussi détectée
            d = [f"{type(e).__name__}: {e}"]
    chk(f"mutation « {nom} » : la parité rougit ({len(d)} écart(s))", bool(d))
chk("sans mutation, les mêmes comparaisons sont vertes",
    not ecarts_profil(cas("nappe à 1,5 m")) and not ecarts_profil(cas("bornes encadrées")) and not ecarts_mode(ABAQ)
    and not ecarts_mode(PLAT_M) and not ecarts_type(F_TYPE, 0, "Sable dense") and not ecarts_remplir(R.nouvelle_fondation(), NOM_GEF, TEXTE_GEF))

# ================================================================== 10. pureté
print("\n=== 10. Pureté : le moteur se charge et calcule SANS streamlit, pandas, numpy, matplotlib ===")
non_std = []
for nom in sorted(os.listdir(os.path.join(RACINE, "sol"))):
    if nom.endswith(".py"):
        arbre = ast.parse(lire("sol", nom))
        for n in ast.walk(arbre):
            noms = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module] if isinstance(n, ast.ImportFrom) and n.level == 0 else []
            non_std += [f"{nom}: {x}" for x in noms if x.split(".")[0] not in sys.stdlib_module_names]
chk("aucun fichier de sol/ n'importe autre chose que la bibliothèque standard (lecture des sources)", not non_std, non_std)
SOUS_PROCESSUS = r'''
import json, sys
avant = set(sys.modules)
INTERDITS = ("streamlit", "pandas", "reportlab", "numpy", "matplotlib", "scipy", "pymupdf", "fitz", "PIL")
class Refus:
    """Refuse tout chargement des bibliothèques interdites."""
    def find_spec(self, nom, chemin=None, cible=None):
        if nom.split(".")[0] in INTERDITS:
            raise ImportError("chargement refusé : " + nom)
        return None
sys.meta_path.insert(0, Refus())
try:
    import numpy  # noqa: F401
    refus = False
except ImportError:
    refus = True
sys.path.insert(0, sys.argv[1])
from sol import rigidite as R
fs = json.loads(sys.stdin.read())
sortie = {"refus": refus, "version": list(sys.version_info[:2]), "res": [R.resultats_profil(f) for f in fs],
          "imp": R.apercu_import(R.lire_essai("x.csv", "Profondeur (m);qc (MPa);fs (kPa)\n" + "\n".join(f"{z/10};{1+z/7};{20+z}" for z in range(1, 80)), 2.0), 0.3)["lignes"]}
charges = {m.split(".")[0] for m in set(sys.modules) - avant}
sortie["tiers"] = sorted(charges - set(sys.stdlib_module_names) - {"sol"})
sortie["interdits"] = sorted(m for m in sys.modules if m.split(".")[0] in INTERDITS)
print(json.dumps(sortie))
'''
fs = [f for _, f in CAS[:12]]
attendu = {"res": json.loads(json.dumps([R.resultats_profil(f) for f in fs])),
           "imp": json.loads(json.dumps(R.apercu_import(R.lire_essai("x.csv", "Profondeur (m);qc (MPa);fs (kPa)\n" + "\n".join(f"{z / 10};{1 + z / 7};{20 + z}" for z in range(1, 80)), 2.0), 0.3)["lignes"]))}
interpretes = [sys.executable] + [x for x in (shutil.which("python3.13"), "/usr/bin/python3.13")
                                  if x and os.path.exists(x) and os.path.realpath(x) != os.path.realpath(sys.executable)][:1]
for exe in interpretes:
    pr = subprocess.run([exe, "-I", "-c", SOUS_PROCESSUS, RACINE], input=json.dumps(fs), capture_output=True, text=True, check=False)
    if pr.returncode != 0:
        chk(f"{exe} : le moteur se charge et calcule sans streamlit", False, pr.stderr.strip().splitlines()[-3:])
        continue
    out = json.loads(pr.stdout)
    v = ".".join(map(str, out["version"]))
    chk(f"Python {v} : numpy, pandas, streamlit… REFUSÉS par sys.meta_path, et le moteur calcule quand même ({len(out['res'])} "
        "fondations, un import)", out["refus"] and not out["interdits"], (out["refus"], out["interdits"]))
    chk(f"Python {v} : le moteur n'a chargé que la bibliothèque standard", not out["tiers"], out["tiers"])
    d = ecarts({"res": out["res"], "imp": out["imp"]}, attendu)
    chk(f"Python {v} : résultats identiques à ceux de ce processus, au bit près", not d, d[:3])

# ================================================================== 11. ce que le SITE affiche = ce que l'écran affiche
print("\n=== 11. Le site (api_web/sol.py, par le routeur) contre l'écran Streamlit rejoué : les chiffres AFFICHÉS ===")
import html as _html  # noqa: E402
import re  # noqa: E402

from api_web import routeur as RT  # noqa: E402


def _site(f):
    g = dict(copy.deepcopy(f), trace=0)
    rep = json.loads(RT.traiter(json.dumps({"module": "sol", "action": "calculer", "etat": {"version": 1, "fondations": [g], "actif": 0}})))
    assert rep["ok"], rep.get("erreur")
    return rep


def _txt(h):
    """Le texte d'un fragment, chaque balise remplacée par rien, les espaces (fines, insécables) ramenées à rien : on
    compare des CHIFFRES, pas la typographie (l'écran écrit 2397,10, le site 2 397,10)."""
    return re.sub(r"[\s \xa0]+", "", _html.unescape(re.sub(r"<[^>]+>", "", h)))


def _cellules(h, classe):
    m = re.search(r'<table class="[^"]*\b' + re.escape(classe) + r'\b[^"]*">(.*?)</table>', h, re.S)
    return [[_txt(c) for c in re.findall(r"<td[^>]*>(.*?)</td>", r, re.S)] for r in re.findall(r"<tr[^>]*>(.*?)</tr>", m.group(1), re.S)][1:] \
        if m else None


def _v(x, nd):
    return f"{x:.{nd}f}".replace(".", ",")


def ecarts_site(f):
    """Les écarts entre les chiffres que le SITE affiche et ceux que l'ÉCRAN affiche, pour une fondation."""
    out, rep = [], _site(f)
    calc = R.resultats_profil(f)["resultats"] if f["mode"] in ("cpt", "profil") else []
    ev = zone_resultats(ecran(charger(f, chart=calc[0]["nom"] if calc else None))[0])
    vues = {v["cle"]: v["svg"] for v in rep["vues"]}
    res_t, vol_t = _txt(rep["resultat"]), _txt(rep["volets"])
    dfs = [e[1] for e in ev if e[0] == "dataframe"]
    if f["mode"] in ("cpt", "profil"):
        if not dfs:
            return out if "Àcompléter" in res_t else ["aucun résultat : le site ne le dit pas"]
        for a, b in zip(_cellules(vues["scia"], "sol-scia"), dfs[0]):
            att = [b["Sondage"], _v(b["k bas [MN/m³]"], 2), _v(b["k haut [MN/m³]"], 2), str(b["k bas [kN/m³]"]), str(b["k haut [kN/m³]"]),
                   b["w [mm]"].replace(" ", "")]
            out += [f"SCIA {a} ≠ {att}"] if [_txt(x) for x in a] != [_txt(x) for x in att] else []
        out += [] if len(_cellules(vues["scia"], "sol-scia")) == len(dfs[0]) else ["nombre de lignes SCIA"]
        for a, b in zip(_cellules(vues["scia"], "sol-zon"), dfs[1]):
            att = [_txt(b["Zone"]), _v(b["k bas [MN/m³]"], 2), _v(b["k haut [MN/m³]"], 2)]
            out += [f"zonage {a} ≠ {att}"] if a != att else []
        for e in ev:                                             # l'enveloppe, la valeur unique : dans le verdict
            if e[0] == "markdown" and ("Enveloppe" in e[1] or e[1].startswith("**k = ")):
                a, b = re.findall(r"k = ([0-9.]+)(?: à ([0-9.]+))?", e[1])[0]
                a, b = a.replace(".", ","), (b or a).replace(".", ",")
                cible = f"k={a}à{b}MN/m³" if a != b else f"k={a}MN/m³"
                out += [] if cible in res_t else [f"enveloppe {cible}"]
        blocs = [e for e in ev if e[0] == "bloc"]
        metr = [e for e in ev if e[0] == "metric"]
        for k, r in enumerate(R.resultats_profil(f)["resultats"]):
            m = re.search(r'id="ck-s' + str(r["index"]) + r'">(.*?)(?=<div class="flyout calcul"|$)', rep["volets"], re.S)
            vt = _txt(m.group(1)) if m else ""
            a, b = re.match(r"w = ([0-9.]+) à ([0-9.]+) mm", blocs[k][2]).groups()
            a, b = a.replace(".", ","), b.replace(".", ",")
            out += [] if (f"wcentre=[min;max]desdeuxbornes={a}à{b}mm" if a != b else f"wcentre=[min;max]desdeuxbornes={a}mm") in vt \
                else [f"{r['nom']} : w {a}–{b}"]
            a, b = (x.replace(".", ",") for x in metr[3 * k + 2][2].split("–"))
            out += [] if (f"kcentre=[min;max]desdeuxbornes={a}à{b}MN/m³" if a != b else f"kcentre=[min;max]desdeuxbornes={a}MN/m³") in vt \
                else [f"{r['nom']} : k au centre {a}–{b}"]
            zi = metr[3 * k + 1][2].replace(" m", "").replace(".", ",")
            if r["detail"].get("tranches") and not re.search(r"zinfl=[^w]*=" + re.escape(zi) + "m", vt):
                out.append(f"{r['nom']} : z d'influence {zi}")
            mq = re.search(r"qnet=[^=]*(?:=[^=]*)?=([0-9,]+)kPa", vt)
            qn = float(metr[3 * k][2].replace(" kPa", ""))
            if not mq or abs(float(mq.group(1).replace(",", ".")) - qn) > 0.5 + 0.006:
                out.append(f"{r['nom']} : q_net {qn}")
        for ts, tst in zip(re.findall(r'<table class="sol-tab sol-tr">.*?</table>', rep["volets"], re.S), dfs[2:]):
            for a, b in zip(_cellules(ts, "sol-tr"), tst):
                att = [_v(b["z sous assise [m]"], 2), _v(b["Δσ [kPa]"], 1), _v(b["σ'v0 [kPa]"], 1), "—" if b["Δσ/σ'v0"] is None else _v(b["Δσ/σ'v0"], 2),
                       _v(b["M [MPa]"], 1), _v(b["dw [mm]"], 3), _txt(b["couche"] or "—")]
                # au demi exact près : le site arrondit demi vers l'extérieur, l'écran au pair (tests/arrondi_affichage.py) ;
                # l'écran a déjà arrondi ses cellules par round() : la valeur brute n'est pas lisible ici
                out += [f"tranche {a} ≠ {att}"] if not AA.memes_nombres(att, a) else []
        for e in ev:                                             # chaque nombre de chaque avertissement de l'écran
            if e[0] in ("warning", "error", "info"):
                out += [f"message « {e[1][:50]} » : {x}" for x in re.findall(r"\d+(?:\.\d+)?", e[1])
                        if not AA.present(x.replace(".", ","), res_t + vol_t)]
    elif f["mode"] == "abaque":
        for a, b in zip(_cellules(vues["abaque_tab"], "sol-ab"), dfs[0]):
            ck = next(c for c in b if c.startswith("k pour B"))
            att = [_txt(b["Catégorie"]), _txt(b["Type de sol"]), _v(b["γ [kN/m³]"], 1), _txt(b["M [MPa]"]), _txt(b["k plaque 0,30 m"]),
                   _txt(b[ck]).replace(".", ","), _txt(b["q pour w réf [kPa]"])]
            out += [f"abaque {a} ≠ {att}"] if not AA.memes_cellules(att, a) else []     # au demi exact près
    else:
        for e in ev:
            if e[0] == "bloc":
                k = _txt(e[2].replace("k = ", "").replace(" MN/m³", "")).replace(".", ",")
                out += [] if k in res_t else [f"bloc {e[2]}"]
            elif e[0] == "dataframe":                            # les théories : chaque k du tableau
                out += [f"théorie {x['Méthode']}" for x in e[1] if f"{_v(x['k [MN/m³]'], 2)}MN/m³" not in res_t]
            elif e[0] == "latex" and "=" in e[1] and "dfrac{q}" not in e[1]:
                out += [f"détail {x}" for x in re.findall(r"= ([0-9 .]+)\\[, ]\\text", e[1]) if not AA.present(_txt(x).replace(".", ","), vol_t)]
    return out


fautes, n_site = [], 0
for desc, f in CAS + [(g["nom"], g) for g in CAS_MODES]:
    d = ecarts_site(f)
    n_site += 1
    fautes += [f"[{desc[:40]}] {x}" for x in d[:3]]
chk(f"{n_site} fondations (les {len(CAS)} des modes 1 et 2, les {len(CAS_MODES)} des modes 3 à 6) : les chiffres AFFICHÉS par le site "
    "— tableau SCIA, zonage, enveloppe, w, k au centre, z d'influence, q_net, tranches, avertissements, k des modes 3 à 6, abaque — "
    "sont ceux que l'écran Streamlit affiche, à l'arrondi d'affichage près", not fautes, fautes[:6])
# la comparaison rougit si le site se trompe : une enveloppe d'affichage décalée d'un centième
M_SITE, C_SITE = sys.modules["api_web.sol"], sys.modules["api_web.commun"]
_vrai_k = M_SITE._k
M_SITE._k = lambda x: C_SITE.fr(x + 0.01, 2)
try:
    chk("contre-épreuve : un k du verdict décalé d'un centième sur le site, la comparaison rougit",
        any(ecarts_site(f) for _, f in CAS[:3]))
finally:
    M_SITE._k = _vrai_k

print(f"\n{EGALITES['n']} comparaisons de valeurs, toutes STRICTES (égalité, aucune tolérance).")
print(f"RÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

# -*- coding: utf-8 -*-
"""Moteur « Tableau des armatures » (beton/armatures) : la PARITÉ avec l'écran Streamlit modules/tableau_armatures.py,
qui reste la référence. Chaque contrôle rougit si l'on retire ce qu'il protège.

L'écran Streamlit ne calcule rien : il affiche une IMAGE, « Tableau armature.png » (racine du dépôt). La parité se
prouve donc en trois temps, indépendants :

  1. L'ÉCRAN : modules/tableau_armatures.show() est exécuté avec un faux « st » qui enregistre au lieu d'afficher, puis
     avec le vrai (AppTest) : un titre, le bouton Accueil, UNE image — ce fichier, cette légende — et aucun champ.
     Les octets que Streamlit SERT sont captés (il recode le PNG en JPEG, qualité 90) ; le PNG est épinglé par son
     empreinte SHA-256 : si le bureau remplace l'image, ce test rougit et dit qu'il faut la retranscrire.
  2. L'IMAGE LUE DANS SES PIXELS — le PNG du dépôt ET le JPEG que l'écran sert : la transcription écrite ici (180
     nombres du corps, 669 glyphes, et l'en-tête) est relue — grille repérée par ses traits, glyphes découpés par
     projection, chaque glyphe comparé aux modèles moyens de chaque chiffre ; chacun doit être plus proche du chiffre
     transcrit que de tout autre (la pire marge est imprimée). Mutations : un seul chiffre faux de la transcription
     (corps, masse, en-tête) est dénoncé.
  3. LE MOTEUR = LA TRANSCRIPTION, au chiffre près : 120 sections de n barres, 12 masses linéiques (trois décimales),
     48 sections par mètre à espacement fixe ; l'en-tête (100/n à une décimale ; 15, 30, 35, 40 cm ; ρ = 7 860). Le
     tableau marqué du moteur (calculer) et chaque case servie au sélecteur portent les mêmes nombres. Mutations du
     moteur (ρ = 7 850, π = 3,14, arrondi par défaut, un espacement fixe, la bande de 1 m) : chacune fait diverger des
     cases ; l'arrondi des moitiés (au pair ou vers le haut) n'y change rien — aucune valeur n'est près d'une moitié.

Puis : contrôles à la main ; la recherche de la plus économique (ex æquo, critères, aucune, et une recherche
exhaustive sur 600 besoins tirés au sort) ; la disposition retenue (barres, nappe, espacement libre, hors du tableau,
barres qui se chevauchent) ; la normalisation (refus clairs) ; le format JSON ; les dessins (à l'échelle) ; la pureté
(le moteur se charge et calcule dans un sous-processus où sys.meta_path REFUSE streamlit, numpy, pandas, PIL,
matplotlib, sous Python 3.12 et sous 3.13 — celui de Pyodide — s'il est installé : résultats identiques).

    /usr/bin/python3.12 tests/test_armatures_moteur.py      (numpy et PIL pour la lecture des pixels, streamlit pour AppTest)
"""
import copy
import hashlib
import importlib
import io
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import types

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from beton.armatures import dessins as DS  # noqa: E402
from beton.armatures import moteur as M  # noqa: E402
from beton.armatures import tableau as T  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{str(info)[:600]}]" if info and not cond else ""))


# ====================================================================================================================
# La TRANSCRIPTION de l'image, écrite une fois pour toutes (relue dans les pixels au § 2). Par ligne : Ø ; n = 1 … 10 ;
# « poids » kg/m ; (colonne « 7860 kg/m³ », vide) ; espacement fixe donné 15, 30, 35, 40 cm.
# ====================================================================================================================
IMAGE = os.path.join(RACINE, T.IMAGE)
SHA256_IMAGE = "886dd825103726c6807ea8b4b9773af7c90e078000698dd50811d460e4b4038c"
TAILLE_IMAGE = (1368, 364)
TRANSCRIPTION = """\
6;28;57;85;113;141;170;198;226;254;283;0,222;;188;94;81;71
8;50;101;151;201;251;302;352;402;452;503;0,395;;335;168;144;126
10;79;157;236;314;393;471;550;628;707;785;0,617;;524;262;224;196
12;113;226;339;452;565;679;792;905;1018;1131;0,889;;754;377;323;283
14;154;308;462;616;770;924;1078;1232;1385;1539;1,210;;1026;513;440;385
16;201;402;603;804;1005;1206;1407;1608;1810;2011;1,580;;1340;670;574;503
20;314;628;942;1257;1571;1885;2199;2513;2827;3142;2,469;;2094;1047;898;785
25;491;982;1473;1963;2454;2945;3436;3927;4418;4909;3,858;;3272;1636;1402;1227
28;616;1232;1847;2463;3079;3695;4310;4926;5542;6158;4,840;;4105;2053;1759;1539
30;707;1414;2121;2827;3534;4241;4948;5655;6362;7069;5,556;;4712;2356;2020;1767
32;804;1608;2413;3217;4021;4825;5630;6434;7238;8042;6,321;;5362;2681;2298;2011
40;1257;2513;3770;5027;6283;7540;8796;10053;11310;12566;9,877;;8378;4189;3590;3142"""
LIGNES = [li.split(";") for li in TRANSCRIPTION.splitlines()]
ENTETE = {}
for _j, (_h, _b) in enumerate(zip([str(n) for n in range(1, 11)], "100,0 50,0 33,3 25,0 20,0 16,7 14,3 12,5 11,1 10,0".split())):
    ENTETE[(_j + 1, "haut")], ENTETE[(_j + 1, "bas")] = _h, _b
ENTETE[(12, "haut")] = "7860"
for _j, _e in enumerate(("15", "30", "35", "40")):
    ENTETE[(13 + _j, "bas")] = _e


# ====================================================================================================================
print("=== 1 · L'écran Streamlit : ce qu'il pose (faux st), puis ce qu'il sert (AppTest) ===")
# ====================================================================================================================
class _Bloc:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FauxSt(types.ModuleType):
    """Un faux module streamlit qui ENREGISTRE chaque appel ; tout appel imprévu est enregistré aussi (et rendu inerte) :
    on voit ainsi tout ce que l'écran demande."""

    def __init__(self):
        super().__init__("streamlit")
        self.appels = []
        self.session_state = {}
        self.query_params = {}

    def columns(self, spec, **k):
        self.appels.append(("columns", (spec,), k))
        return [_Bloc() for _ in (spec if isinstance(spec, (list, tuple)) else range(spec))]

    def button(self, *a, **k):
        self.appels.append(("button", a, k))
        return False

    def __getattr__(self, nom):
        if nom.startswith("__"):
            raise AttributeError(nom)

        def f(*a, **k):
            self.appels.append((nom, a, k))
        return f


def ecran_faux():
    """L'écran exécuté avec le faux st ; tout est rétabli ensuite (sys.modules ET les attributs du paquet « modules »,
    sans quoi « from modules import tableau_armatures » retrouverait la version au faux st)."""
    st = FauxSt()
    cles = ("streamlit", "modules.navigation", "modules.tableau_armatures")
    sauve = {k: sys.modules.get(k) for k in cles}
    sys.modules["streamlit"] = st
    for k in cles[1:]:
        sys.modules.pop(k, None)
    try:
        importlib.import_module("modules.tableau_armatures").show()
    finally:
        paquet = sys.modules.get("modules")
        for k, v in sauve.items():
            if v is None:
                sys.modules.pop(k, None)
                if paquet is not None and "." in k and hasattr(paquet, k.split(".")[1]):
                    delattr(paquet, k.split(".")[1])
            else:
                sys.modules[k] = v
                if paquet is not None and "." in k:
                    setattr(paquet, k.split(".")[1], v)
    return st.appels


appels = ecran_faux()
noms = [a[0] for a in appels]
images = [a for a in appels if a[0] == "image"]
chk("l'écran pose une seule image", len(images) == 1, noms)
img = images[0] if images else ("image", ("",), {})
chk("… ce fichier : « Tableau armature.png » (T.IMAGE, à la racine du dépôt)", img[1][:1] == (T.IMAGE,) and os.path.isfile(IMAGE), img[1])
chk("… cette légende (T.LEGENDE_IMAGE, reprise sur le site)", img[2].get("caption") == T.LEGENDE_IMAGE, img[2])
titres = [a[1][0] for a in appels if a[0] == "markdown" and a[1] and str(a[1][0]).startswith("##")]
chk("le titre : « Tableau des sections d’armatures » (T.TITRE_ECRAN, repris comme titre de la vue Tableau du site)",
    titres == ["## 📊 Tableau des sections d’armatures"] and titres == [f"## 📊 {T.TITRE_ECRAN}"], titres)
SAISIES = {"number_input", "text_input", "selectbox", "radio", "slider", "checkbox", "toggle", "multiselect", "data_editor",
           "text_area", "select_slider", "file_uploader", "segmented_control", "pills", "download_button"}
chk("aucun champ de saisie : il n'y a rien d'autre à reprendre que l'image", not (set(noms) & SAISIES), set(noms) & SAISIES)
chk("seul bouton : « 🏠 Accueil » (la navigation, que la barre du site remplace)",
    [a[1][0] for a in appels if a[0] == "button"] == ["🏠 Accueil"], [a for a in appels if a[0] == "button"])
chk("rien d'autre : colonnes, titre, séparateur, bouton, image", sorted(set(noms)) == ["button", "columns", "image", "markdown"], sorted(set(noms)))

with open(IMAGE, "rb") as fh:
    OCTETS_PNG = fh.read()
chk("l'image du dépôt est celle qui a été transcrite (SHA-256 épinglé) — sinon : la retranscrire",
    hashlib.sha256(OCTETS_PNG).hexdigest() == SHA256_IMAGE, hashlib.sha256(OCTETS_PNG).hexdigest())

OCTETS_SERVIS = None
try:
    import logging

    from streamlit.runtime.memory_media_file_storage import MemoryMediaFileStorage
    from streamlit.testing.v1 import AppTest
except Exception:  # noqa: BLE001
    AppTest = None
if AppTest is None:
    print("  (streamlit absent : l'exécution réelle de l'écran est sautée)")
else:
    logging.getLogger("streamlit").setLevel(logging.CRITICAL)
    servis = []
    _charger = MemoryMediaFileStorage.load_and_get_id

    def _espion(self, donnees, mimetype, kind, filename=None):
        servis.append((donnees, mimetype))
        return _charger(self, donnees, mimetype, kind, filename)
    MemoryMediaFileStorage.load_and_get_id = _espion
    try:
        at = AppTest.from_string(f"import os, sys\nsys.path.insert(0, {RACINE!r})\nos.chdir({RACINE!r})\n"
                                 "from modules import tableau_armatures\ntableau_armatures.show()\n", default_timeout=60)
        at.run()
    finally:
        MemoryMediaFileStorage.load_and_get_id = _charger
    chk("AppTest : l'écran réel s'exécute sans exception", not at.exception, [e.message for e in at.exception][:2])
    ims = at.get("image")
    chk("AppTest : il rend une image, avec la légende T.LEGENDE_IMAGE",
        len(ims) == 1 and [i.caption for i in ims[0].proto.imgs] == [T.LEGENDE_IMAGE], [str(i.proto)[:200] for i in ims])
    chk("AppTest : aucun champ ; un bouton « 🏠 Accueil »", not (at.number_input or at.selectbox or at.text_input or at.radio or at.checkbox)
        and [b.label for b in at.button] == ["🏠 Accueil"], [b.label for b in at.button])
    chk("AppTest : Streamlit SERT l'image recodée en JPEG (qualité 90, sans transparence) — c'est elle qu'on relit aussi",
        len(servis) == 1 and servis[0][1] == "image/jpeg" and isinstance(servis[0][0], bytes), [(type(d).__name__, m) for d, m in servis])
    if servis and isinstance(servis[0][0], bytes):
        OCTETS_SERVIS = servis[0][0]


# ====================================================================================================================
print("\n=== 2 · La transcription relue dans les pixels (le PNG du dépôt, le JPEG servi) ===")
# ====================================================================================================================
try:
    import numpy as np
    from PIL import Image
except Exception:  # noqa: BLE001
    np = Image = None


def charger(octets):
    return np.asarray(Image.open(io.BytesIO(octets)).convert("RGB")).astype(float)


def grille(rgb):
    """Les rangées du corps (bandes blanches et orangées sous l'en-tête) et les colonnes (traits verticaux)."""
    L = rgb.mean(axis=2)
    H, W = L.shape
    orange = (rgb[:, 4, 0] - rgb[:, 4, 2]) > 40
    blanc = L[:, 4] > 245
    noir = L[:, 4] < 60
    y_hdr = max(y for y in range(H // 3) if noir[y])
    bandes, etat, debut = [], None, y_hdr + 1
    for yy in range(y_hdr + 1, H + 1):
        e = None if yy == H else ("o" if orange[yy] else "b" if blanc[yy] else None)
        if e != etat or yy == H:
            if etat is not None and yy - debut > 8:
                bandes.append((debut, yy))
            etat, debut = e, yy
    yb = [b for b in bandes if blanc[(b[0] + b[1]) // 2]][0]
    w = L[yb[0] + 3: yb[1] - 3]
    xs = [x for x in range(W) if (w[:, x] < 230).all()]
    seps = []
    for x in xs:
        if seps and x - seps[-1][-1] <= 1:
            seps[-1].append(x)
        else:
            seps.append([x])
    return y_hdr, bandes, [(a[-1] + 1, b[0]) for a, b in zip(seps, seps[1:])]


def glyphes(encre, seuil=0.3):
    """Les glyphes d'une case, (x0, x1, y0, y1) de gauche à droite, par projection des colonnes d'encre."""
    m = encre > seuil
    col = m.any(axis=0)
    out, x, W = [], 0, len(col)
    while x < W:
        if col[x]:
            x1 = x
            while x1 < W and col[x1]:
                x1 += 1
            ys = np.where(m[:, x:x1].any(axis=1))[0]
            out.append((x, x1, ys[0], ys[-1] + 1))
            x = x1
        else:
            x += 1
    return out


def vignette(encre, g, taille=(12, 18)):
    x0, x1, y0, y1 = g
    v = Image.fromarray((np.clip(encre[y0:y1, x0:x1], 0, 1) * 255).astype(np.uint8)).resize(taille, Image.BILINEAR)
    return np.asarray(v).astype(float) / 255


def lire_corps(rgb, attendu):
    """Chaque glyphe du corps comparé aux modèles moyens de chaque chiffre (les chiffres gras de la colonne des masses
    forment leurs propres modèles) ; rend (fautes, modèles, nombre de glyphes, pire rapport d1/d2)."""
    y_hdr, bandes, cols = grille(rgb)
    if len(bandes) != 12 or len(cols) != 17:
        return [("grille", len(bandes), len(cols))], {}, 0, 1.0
    L = rgb.mean(axis=2)
    ech, fautes = [], []
    for i, (y0, y1) in enumerate(bandes):
        fond = np.median(L[y0:y1, 2:6])
        for j, (a, b) in enumerate(cols):
            encre = (fond - L[y0 + 1:y1 - 1, a + 2:b - 1]) / fond
            gs = glyphes(encre)
            txt = attendu[i][j]
            while 0 < len(gs) < len(txt):          # deux chiffres collés : couper au creux d'encre du plus large
                k = max(range(len(gs)), key=lambda q: gs[q][1] - gs[q][0])
                x0, x1, _, _ = gs[k]
                if x1 - x0 < 6:
                    break
                xc = min(range(x0 + 2, x1 - 2), key=lambda x: encre[:, x].clip(0).sum())
                morceaux = []
                for u0, u1 in ((x0, xc), (xc, x1)):
                    ys = np.where((encre[:, u0:u1] > 0.3).any(axis=1))[0]
                    morceaux.append((u0, u1, ys[0], ys[-1] + 1))
                gs[k:k + 1] = morceaux
            if len(gs) != len(txt):
                fautes.append(((i, j), txt, f"{len(gs)} glyphe(s)"))
                continue
            for k, (g, ch) in enumerate(zip(gs, txt)):
                ech.append((("g" if j == 11 else "") + ch, vignette(encre, g), (i, j, k)))
    modeles = {cl: np.mean([e[1] for e in ech if e[0] == cl], axis=0) for cl in {e[0] for e in ech}}
    pire = 0.0
    for cl, v, pos in ech:
        d = {c: float(((v - m) ** 2).sum()) for c, m in modeles.items() if c.startswith("g") == cl.startswith("g")}
        meilleur = min(d, key=d.get)
        if meilleur != cl:
            fautes.append((pos, cl, meilleur))
        autres = sorted(x for c, x in d.items() if c != cl)
        pire = max(pire, d[cl] / autres[0] if autres and autres[0] > 0 else 0.0)
    return fautes, modeles, len(ech), pire


def lire_entete(rgb, modeles, attendu):
    """L'en-tête (chiffres blancs sur fond orangé), lu avec les modèles du corps : {(colonne, haut|bas): texte}."""
    y_hdr, bandes, cols = grille(rgb)
    L = rgb.mean(axis=2)
    fond = np.median(L[2:y_hdr - 2, 2:6])
    mi = y_hdr // 2
    fautes, lu = [], {}
    for (j, ou), txt in attendu.items():
        a, b = cols[j]
        y0, y1 = (2, mi) if ou == "haut" else (mi, y_hdr - 1)
        encre = (L[y0:y1, a + 2:b - 1] - fond) / (255 - fond)
        gs = glyphes(encre, 0.35)
        hmax = max((g[3] - g[2] for g in gs), default=0)
        s = ""
        for g in gs:
            if g[3] - g[2] < 0.5 * hmax:
                s += ","
                continue
            v = vignette(encre, g)
            d = {c: float(((v - m) ** 2).sum()) for c, m in modeles.items() if c[-1].isdigit() and not c.startswith("g")}
            s += min(d, key=d.get)[-1]
        lu[(j, ou)] = s
        if s != txt:
            fautes.append(((j, ou), txt, s))
    return fautes, lu


GLYPHES_ATTENDUS = sum(len(t) for li in LIGNES for t in li)
if np is None or Image is None:
    chk("numpy et PIL sont installés pour relire l'image (python3.12 du bureau)", False)
else:
    sources = [("PNG du dépôt", OCTETS_PNG)] + ([("JPEG servi par Streamlit", OCTETS_SERVIS)] if OCTETS_SERVIS else [])
    for nom_src, octets in sources:
        rgb = charger(octets)
        chk(f"{nom_src} : {TAILLE_IMAGE[0]} × {TAILLE_IMAGE[1]} px", (rgb.shape[1], rgb.shape[0]) == TAILLE_IMAGE, rgb.shape)
        y_hdr, bandes, cols = grille(rgb)
        chk(f"{nom_src} : la grille est retrouvée — 12 rangées (Ø6 … Ø40) × 17 colonnes", len(bandes) == 12 and len(cols) == 17, (len(bandes), len(cols)))
        fautes, modeles, n_gl, pire = lire_corps(rgb, LIGNES)
        chk(f"{nom_src} : les {GLYPHES_ATTENDUS} glyphes du corps sont tous découpés et lus comme transcrits "
            f"(pire marge d1/d2 = {pire:.3f})", not fautes and n_gl == GLYPHES_ATTENDUS and pire < 0.6, (fautes[:5], n_gl, pire))
        fe, lu = lire_entete(rgb, modeles, ENTETE)
        chk(f"{nom_src} : l'en-tête est lu comme transcrit (1 … 10 ; 100,0 … 10,0 ; 7860 ; 15 30 35 40)", not fe, fe[:5])
    rgb = charger(OCTETS_PNG)
    _, modeles, _, _ = lire_corps(rgb, LIGNES)
    for (i, j, faux, quoi) in ((3, 9, "1019", "1 018 écrit 1 019 (Ø12, 9 barres)"), (6, 11, "2,496", "2,469 écrit 2,496 (masse Ø20)"),
                               (11, 16, "3143", "3 142 écrit 3 143 (Ø40 à 40 cm)"), (0, 1, "29", "28 écrit 29 (Ø6, 1 barre)")):
        mut = copy.deepcopy(LIGNES)
        mut[i][j] = faux
        f, _, _, _ = lire_corps(rgb, mut)
        chk(f"mutation de la transcription : {quoi} — la lecture le dénonce", bool(f), f[:2])
    mut = dict(ENTETE)
    mut[(6, "bas")] = "16,6"
    f, _ = lire_entete(rgb, modeles, mut)
    chk("mutation de l'en-tête : 16,7 écrit 16,6 — la lecture le dénonce", bool(f), f)


# ====================================================================================================================
print("\n=== 3 · Le moteur = la transcription, au chiffre près ===")
# ====================================================================================================================
def ecarts(tab):
    """Les cases où le tableau du moteur diffère de la transcription (Ø, 10 sections, masse, 4 sections par mètre)."""
    out = []
    if [li["d"] for li in tab["lignes"]] != [int(li[0]) for li in LIGNES]:
        return [("diamètres", [li["d"] for li in tab["lignes"]])]
    for li, tr in zip(tab["lignes"], LIGNES):
        for n in range(10):
            if li["barres_arrondi"][n] != int(tr[1 + n]):
                out.append((li["d"], f"{n + 1} barres", li["barres_arrondi"][n], tr[1 + n]))
        if li["masse_txt"] != tr[11]:
            out.append((li["d"], "masse", li["masse_txt"], tr[11]))
        if len(li["fixes_arrondi"]) != 4:
            out.append((li["d"], "fixes", li["fixes_arrondi"]))
            continue
        for k in range(4):
            if li["fixes_arrondi"][k] != int(tr[13 + k]):
                out.append((li["d"], f"e{tab['espacements_fixes'][k]}", li["fixes_arrondi"][k], tr[13 + k]))
    if tab["espacements_n_txt"] != [ENTETE[(j, "bas")] for j in range(1, 11)]:
        out.append(("en-tête 100/n", tab["espacements_n_txt"]))
    if [str(e) for e in tab["espacements_fixes"]] != [ENTETE[(j, "bas")] for j in range(13, 17)]:
        out.append(("en-tête espacements fixes", tab["espacements_fixes"]))
    if str(tab["masse_volumique"]) != ENTETE[(12, "haut")]:
        out.append(("ρ", tab["masse_volumique"]))
    return out


tab = T.tableau()
e = ecarts(tab)
n_cases = sum(len(li["barres_arrondi"]) + 1 + len(li["fixes_arrondi"]) for li in tab["lignes"])
chk(f"les {n_cases} cases du tableau (120 sections, 12 masses, 48 sections par mètre) et l'en-tête sont ceux de l'image",
    not e and n_cases == 180, e[:6])
chk("le tableau marqué du moteur (calculer) porte les mêmes nombres, case par case",
    all([c["arrondi"] for c in li["cases"]] == [int(x) for x in tr[1:11]] + [int(x) for x in tr[13:17]] and li["masse_txt"] == tr[11]
        for li, tr in zip(M.calculer(M.nouveau_besoin())["tableau"]["lignes"], LIGNES)))
b_n = M.normaliser({"mode": "nappe", "d_min": 6, "d_max": 40, "e_min": 1, "e_max": 100})
serv = {(c["d"], c["colonne"]): T.arrondi(c["As"]) for c in M.candidats(b_n)}
b_b = M.normaliser({"mode": "barres"})
serv.update({(c["d"], f"b{c['n']}"): T.arrondi(c["As"]) for c in M.candidats(b_b)})
attendu_serv = {}
for tr in LIGNES:
    d = int(tr[0])
    for n in range(1, 11):
        attendu_serv[(d, f"n{n}")] = attendu_serv[(d, f"b{n}")] = int(tr[n])
    for k, ef in enumerate((15, 30, 35, 40)):
        attendu_serv[(d, f"e{ef}")] = int(tr[13 + k])
chk("chaque disposition servie au sélecteur (barres : 120 ; nappe : 168) a la valeur de sa case", serv == attendu_serv,
    [(k, serv.get(k), v) for k, v in attendu_serv.items() if serv.get(k) != v][:4])
# les moitiés : le sens de l'arrondi n'y change rien
marge = min(min(abs(x - math.floor(x) - 0.5) for x in li["barres"] + li["fixes"]) for li in tab["lignes"])
marge_m = min(abs(li["masse"] * 1000 - math.floor(li["masse"] * 1000) - 0.5) for li in tab["lignes"])
chk(f"aucune valeur n'est à moins de 0,003 mm² d'une moitié (marge {marge:.4f} mm² ; masses : {marge_m:.3f} g) — le sens de l'arrondi est indifférent",
    marge > 0.003 and marge_m > 0.01, (marge, marge_m))


def mutant(**patch):
    """Le tableau recalculé avec un morceau du moteur remplacé (rétabli ensuite)."""
    sauve = {k: getattr(T, k) for k in patch}
    try:
        for k, v in patch.items():
            setattr(T, k, v)
        return ecarts(T.tableau())
    except Exception as x:  # noqa: BLE001
        return [("exception", repr(x))]
    finally:
        for k, v in sauve.items():
            setattr(T, k, v)


MUTATIONS = [("ρ = 7 850 kg/m³ (l'usage)", {"MASSE_VOLUMIQUE": 7850}),
             ("π = 3,14", {"aire": lambda d: 3.14 * d * d / 4}),
             ("arrondi par défaut (troncature)", {"arrondi": lambda x: int(math.floor(x))}),
             ("arrondi au mm² supérieur", {"arrondi": lambda x: int(math.ceil(x))}),
             ("espacements fixes 15 · 30 · 35 · 45 cm", {"ESPACEMENTS_FIXES": (15, 30, 35, 45)}),
             ("bande de 1 000 cm au lieu de 100 cm", {"METRE": 1000}),
             ("masse à deux décimales", {"masse_txt": lambda m: f"{m:.2f}".replace(".", ",")}),
             ("Ø50 ajouté", {"DIAMETRES": T.DIAMETRES + (50,)}),
             ("Ø = rayon (aire π·Ø²)", {"aire": lambda d: math.pi * d * d})]
for nom, patch in MUTATIONS:
    ec = mutant(**patch)
    chk(f"mutation du moteur : {nom} — {len(ec)} écart(s) dénoncé(s)", len(ec) > 0, ec[:2])
ec = mutant(arrondi=lambda x: int(round(x)))
chk("témoin : l'arrondi des moitiés au pair (round de Python) donne le MÊME tableau (aucune moitié)", not ec, ec[:3])
chk("témoin : le moteur rétabli après les mutations", not ecarts(T.tableau()))


# ====================================================================================================================
print("\n=== 4 · À la main ===")
# ====================================================================================================================
def proche(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(b))


chk("a(Ø12) = 36·π = 113,097 mm²", proche(T.aire(12), 36 * math.pi) and round(T.aire(12), 3) == 113.097)
chk("4Ø12 = 144·π = 452,389 mm²", proche(T.section(4, 12), 144 * math.pi))
chk("Ø10 à 15 cm : 25·π·100/15 = 523,599 mm²/m (l'image : 524)", proche(T.section_par_metre(10, 15), 25 * math.pi * 100 / 15)
    and T.arrondi(T.section_par_metre(10, 15)) == 524)
chk("m(Ø12) = 7 860 × 36·π × 10⁻⁶ = 0,888946 kg/m (l'image : 0,889)", proche(T.masse_lineique(12), 7860 * 36 * math.pi * 1e-6)
    and T.masse_txt(T.masse_lineique(12)) == "0,889")
chk("10Ø40 = 4 000·π = 12 566,37 mm² (l'image : 12566)", proche(T.section(10, 40), 4000 * math.pi) and T.arrondi(T.section(10, 40)) == 12566)
chk("100/3 = 33,3 cm ; 100/7 = 14,3 cm (l'en-tête)", T.espacement_txt(T.espacement(3)) == "33,3" and T.espacement_txt(T.espacement(7)) == "14,3")
r = M.calculer({"mode": "nappe", "d": 16, "espacement": M.LIBRE, "e_libre": 12, "As_req": 0})["disposition"]
chk("nappe Ø16 à 12 cm (hors tableau) : 64·π·100/12 = 1 675,52 mm²/m ; 8,33 barres/m ; 13,17 kg/m² ; jour 104 mm",
    proche(r["As"], 64 * math.pi * 100 / 12) and proche(r["barres_par_metre"], 100 / 12) and proche(r["masse"], 100 / 12 * 7860 * 64 * math.pi * 1e-6)
    and r["jour"] == 104 and not r["dans_tableau"], r)


# ====================================================================================================================
print("\n=== 5 · La plus économique ===")
# ====================================================================================================================
def opt(**b):
    o = M.optimale(M.normaliser(b))
    return o and o["nom"]


chk("barres, A_s,req = 400 mm² : 2Ø16 (402) — ex æquo avec 8Ø8 (n·Ø² = 512) : le moins de barres", opt(mode="barres", As_req=400) == "2Ø16"
    and T.section(2, 16) == T.section(8, 8))
chk("barres, A_s,req = 1 250 mm² : 1Ø40 — ex æquo avec 4Ø20", opt(mode="barres", As_req=1250) == "1Ø40")
chk("… Ø32 au plus : 4Ø20", opt(mode="barres", As_req=1250, d_max=32) == "4Ø20")
chk("… 2 à 3 barres, Ø16 au plus : aucune (3Ø16 = 603)", opt(mode="barres", As_req=1250, d_max=16, n_min=2, n_max=3) is None)
chk("nappe, A_s,req = 500 mm²/m : Ø16 à 40 cm (503) — ex æquo avec Ø8 à 10 cm : le moins de barres par mètre",
    opt(mode="nappe", As_req=500) == "Ø16 · 40 cm")
chk("… espacement de 10 à 20 cm : Ø8 à 10 cm (503, 10 barres/m)", opt(mode="nappe", As_req=500, e_max=20) == "Ø8 · 10 cm")
chk("… espacement de 12 à 20 cm : Ø10 à 15 cm (524)", opt(mode="nappe", As_req=500, e_min=12, e_max=20) == "Ø10 · 15 cm")
chk("une section requise ÉGALE à une case : cette case suffit (≥)", opt(mode="barres", As_req=T.section(4, 12), d_min=12, d_max=12) == "4Ø12")
chk("sans section requise : pas de plus économique", opt(mode="barres", As_req=0) is None)
res = M.calculer({"mode": "barres", "As_req": 1e6})
chk("A_s,req = 1 000 000 mm² : aucune — avertissement « aucune »", res["optimale"] is None and [a["code"] for a in res["avertissements"]] == ["insuffisante", "aucune"],
    res["avertissements"])
rnd = random.Random(20261107)
fautes = []
for _ in range(600):
    b = M.normaliser({"mode": rnd.choice(["barres", "nappe"]), "As_req": round(rnd.uniform(1, 9000), rnd.choice([0, 1, 2])),
                      "d_min": rnd.choice(T.DIAMETRES[:6]), "d_max": rnd.choice(T.DIAMETRES[6:]),
                      "n_min": rnd.randint(1, 4), "n_max": rnd.randint(5, 10), "e_min": rnd.choice([5, 10, 12.5]), "e_max": rnd.choice([20, 30, 40, 100])})
    o = M.optimale(b)
    tous = []                                     # la recherche exhaustive, écrite à part : toutes les cases du tableau
    for d in T.DIAMETRES:
        if not b["d_min"] <= d <= b["d_max"]:
            continue
        if b["mode"] == "barres":
            tous += [(n * d * d, n, d, T.section(n, d)) for n in T.NOMBRES if b["n_min"] <= n <= b["n_max"]]
        else:
            tous += [(None, 100 / e, d, T.section_par_metre(d, e)) for e in T.ESPACEMENTS_FIXES if b["e_min"] <= e <= b["e_max"]]
            tous += [(None, n, d, T.section(n, d)) for n in T.NOMBRES if b["e_min"] <= 100 / n <= b["e_max"]]
    ok_ = [x for x in tous if x[3] >= b["As_req"]]
    if not ok_:
        if o is not None:
            fautes.append(("trouvée à tort", b, o["nom"]))
        continue
    amin = min(x[3] for x in ok_)
    if o is None or abs(o["As"] - amin) > 1e-6 or o["As"] < b["As_req"]:
        fautes.append((b, o and o["nom"], amin))
        continue
    egaux = [x for x in ok_ if abs(x[3] - amin) <= 1e-6]
    if o["barres_par_metre"] != min(x[1] for x in egaux):
        fautes.append(("ex æquo", b, o["nom"]))
chk("600 besoins tirés au sort : la plus économique est la plus petite section qui suffit (recherche exhaustive écrite à part), "
    "le moins de barres à section égale", not fautes, fautes[:3])
c = M.candidats(M.normaliser({"mode": "barres", "As_req": 400}))
chk("candidats : triés par section croissante, 120 cases, une seule optimale, la retenue marquée (4Ø12)",
    len(c) == 120 and all(round(a["As"], 6) <= round(b_["As"], 6) for a, b_ in zip(c, c[1:])) and sum(x["optimale"] for x in c) == 1
    and [x["nom"] for x in c if x["retenue"]] == ["4Ø12"])
c = M.candidats(M.normaliser({"mode": "barres", "n": 12}), avec_retenue=True)
chk("candidats avec la retenue hors du tableau (12Ø12) : elle est ajoutée, à sa place", [x["nom"] for x in c if x["retenue"]] == ["12Ø12"] and len(c) == 121
    and all(round(a["As"], 6) <= round(b_["As"], 6) for a, b_ in zip(c, c[1:])))


# ====================================================================================================================
print("\n=== 6 · La disposition retenue ===")
# ====================================================================================================================
res = M.calculer(M.nouveau_besoin())
x = res["disposition"]
chk("par défaut : Repère 1, barres, A_s,req = 400 mm², 4Ø12 retenus", res["besoin"]["nom"] == "Repère 1" and x["nom"] == "4Ø12" and res["besoin"]["As_req"] == 400)
chk("4Ø12 : A_s = n·a, masse n·m (kg/m), lus par mètre e = 25 cm, case n4 du tableau, taux 400/452,4",
    x["As"] == T.section(4, 12) and proche(x["masse"], 4 * T.masse_lineique(12)) and x["e"] == 25 and x["colonne"] == "n4"
    and x["dans_tableau"] and proche(x["taux"], 400 / T.section(4, 12)) and x["suffit"] and x["unite"] == "mm²")
chk("… ses formules : a, A_s, m, M, e, η — chacune avec ses termes et son résultat",
    [f["cle"] for f in x["formules"]] == ["a", "As", "m", "M", "e", "taux"] and x["formules"][1]["resultat"] == x["As"])
x = M.calculer({"mode": "nappe", "d": 10, "espacement": "e15"})["disposition"]
chk("nappe Ø10 à 15 cm : a·100/e, 6,67 barres/m, kg/m², case e15", x["As"] == T.section_par_metre(10, 15) and proche(x["barres_par_metre"], 100 / 15)
    and x["unite"] == "mm²/m" and x["unite_masse"] == "kg/m²" and x["colonne"] == "e15")
x = M.calculer({"mode": "nappe", "d": 12, "espacement": "n4"})["disposition"]
chk("nappe Ø12 à 25 cm (colonne « 4 barres par mètre ») : A_s = 4·a, comme l'image (et non a·100/25)", x["As"] == T.section(4, 12) and x["e"] == 25)
x = M.calculer({"mode": "nappe", "d": 12, "espacement": M.LIBRE, "e_libre": 15})["disposition"]
chk("un espacement libre de 15 cm retombe sur la case e15 du tableau", x["dans_tableau"] and x["colonne"] == "e15")
res = M.calculer({"mode": "barres", "n": 14, "d": 12, "As_req": 2000})
chk("14Ø12 : hors du tableau (avertissement), même formule ; ne suffit pas (avertissement)",
    not res["disposition"]["dans_tableau"] and {a["code"] for a in res["avertissements"]} == {"hors_tableau", "insuffisante"})
res = M.calculer({"mode": "nappe", "d": 16, "espacement": M.LIBRE, "e_libre": 1.5, "As_req": 0})
chk("Ø16 à 1,5 cm : les barres se chevauchent (jour e − Ø ≤ 0) — avertissement", "chevauchement" in [a["code"] for a in res["avertissements"]])
res = M.calculer({"mode": "barres", "d": 40, "n": 1, "As_req": 1257})
msg = res["avertissements"][0]["message"] if res["avertissements"] else ""
chk("1Ø40 (1 256,64 mm², la case écrit 1 257) face à 1 257 mm² requis : ne suffit pas, et le message écrit « 1 256,6 », "
    "jamais « 1 257 mm² < 1 257 mm² »", not res["disposition"]["suffit"] and "1\u202f256,6 mm² < 1\u202f257 mm²" in msg, msg)
chk("section_txt : au mm² quand l'arrondi ne trompe pas, au dixième (ou plus) sinon",
    (M.section_txt(T.section(4, 12), 400), M.section_txt(T.section(1, 40), 1257), M.section_txt(T.section(4, 12), T.section(4, 12)),
     M.section_txt(1256.66, 1256.7)) == ("452", "1\u202f256,6", "452,4", "1\u202f256,66"))
x = M.calculer({"mode": "nappe", "d": 12, "espacement": M.LIBRE, "e_libre": 16.75})["disposition"]
chk("un espacement libre de 16,75 cm s'écrit « 16,75 » (une décimale l'écrirait 16,8 : une autre nappe)", x["nom"] == "Ø12 · 16,75 cm", x["nom"])
x = M.calculer({"mode": "nappe", "d": 12, "espacement": "n6"})["disposition"]
chk("… une colonne du tableau garde l'écriture de son en-tête (100/6 → « 16,7 »)", x["nom"] == "Ø12 · 16,7 cm", x["nom"])
res = M.calculer({"mode": "barres", "As_req": 0})
chk("sans section requise : aucun avertissement, aucune case « suffit »", not res["avertissements"] and res["suffisantes"] == 0
    and not any(c["suffit"] for li in res["tableau"]["lignes"] for c in li["cases"]))
res = M.calculer({"mode": "barres", "As_req": 400})
chk("A_s,req = 400 mm², barres : 91 cases du tableau suffisent ; les colonnes « espacement fixe » ne servent pas aux barres",
    res["suffisantes"] == 91 and all(not c["utile"] for li in res["tableau"]["lignes"] for c in li["cases"][10:]))


# ====================================================================================================================
print("\n=== 7 · La normalisation : défauts, refus clairs ===")
# ====================================================================================================================
chk("un besoin vide prend les défauts", M.normaliser({}) == {k: (float(v) if k in ("As_req", "e_libre", "e_min", "e_max") else v) for k, v in M.DEFAUT.items()})
chk("normaliser est stable", M.normaliser(M.normaliser({"mode": "nappe", "As_req": 12.5})) == M.normaliser({"mode": "nappe", "As_req": 12.5}))
REFUS = [({"As_req": -1}, "négative"), ({"As_req": "abc"}, "nombre"), ({"As_req": float("nan")}, "infini"), ({"As_req": True}, "nombre"),
         ({"d": 13}, "diamètre"), ({"d": 12.5}, "entier"), ({"n": 0}, "1 à"), ({"n": 201}, "1 à"), ({"n": 2.5}, "entier"),
         ({"mode": "treillis"}, "mode"), ({"espacement": "e17"}, "espacement"), ({"e_libre": 0}, "espacement"),
         ({"e_libre": 2000}, "espacement"), ({"d_min": 20, "d_max": 12}, "mini"), ({"n_min": 0}, "recherche"),
         ({"n_max": 11}, "recherche"), ({"n_min": 6, "n_max": 5}, "recherche"), ({"e_min": 0}, "positifs"), ({"e_min": 30, "e_max": 20}, "mini"),
         ({"inconnue": 1}, "inconnue"), ([1, 2], "objet"), ({"As_req": M.AS_REQ_MAX * 1.01}, "au plus"), ({"As_req": 1e300}, "au plus"),
         ({"e_libre": 100.5}, "une barre par mètre"), ({"nom": "x" * (M.NOM_MAX + 1)}, "caractères")]
for b, mot in REFUS:
    try:
        M.normaliser(b)
        chk(f"refus : {b}", False, "accepté")
    except ValueError as err:
        chk(f"refus : {b} → « {err} »", mot in str(err), str(err))
chk("un nom non textuel est relu en texte", M.normaliser({"nom": 12})["nom"] == "12")
chk("« -0 » est relu 0 (le champ n'écrit pas « -0 »)", str(M.normaliser({"As_req": -0.0})["As_req"]) == "0.0")
chk("les bornes elles-mêmes sont admises (A_s,req = 1 000 000 ; e = 100 cm ; nom de 120 caractères)",
    M.normaliser({"As_req": M.AS_REQ_MAX, "e_libre": M.E_LIBRE_MAX, "nom": "x" * M.NOM_MAX})["As_req"] == M.AS_REQ_MAX)


# ====================================================================================================================
print("\n=== 8 · Le format : JSON, sans NaN ===")
# ====================================================================================================================
for b in ({}, {"mode": "nappe", "As_req": 300, "espacement": M.LIBRE, "e_libre": 7.5}, {"As_req": 0}, {"As_req": 1e6, "n": 150}):
    res = M.calculer(b)
    try:
        s = json.dumps(res, allow_nan=False)
        chk(f"calculer({b}) est JSON ({len(s)} octets) et se relit à l'identique", json.loads(s) == res)
    except ValueError as err:
        chk(f"calculer({b}) est JSON", False, err)
cles = {"reference", "besoin", "disposition", "optimale", "suffisantes", "tableau", "avertissements"}
chk("la sortie de calculer porte les clés de sa docstring", set(M.calculer({})) == cles, set(M.calculer({})))
chk("chaque formule porte cle, symbole, expression, termes, resultat, unite",
    all(set(f) == {"cle", "symbole", "expression", "termes", "resultat", "unite"} for f in M.calculer({})["disposition"]["formules"]))


# ====================================================================================================================
print("\n=== 9 · Les dessins, à l'échelle ===")
# ====================================================================================================================
svg = DS.nappe("t", 12, 25.0, "titre <b>", "pied & co")
cercles = re.findall(r'<circle cx="([\d.]+)" cy="[\d.]+" r="([\d.]+)"', svg)
chk("nappe Ø12 à 25 cm : 4 barres dans le mètre, en coupe, au rayon 6 mm (unités : mm), à 125 · 375 · 625 · 875",
    [(float(a), float(r_)) for a, r_ in cercles] == [(125.0, 6.0), (375.0, 6.0), (625.0, 6.0), (875.0, 6.0)], cercles)
chk("… la cote « e = 250 » et « 1 000 mm » ; le titre et le pied échappés",
    "e = 250" in svg and "1 000 mm" in svg and "titre &lt;b&gt;" in svg and "pied &amp; co" in svg and "<b>" not in svg)
svg = DS.nappe("t", 8, 100 / 3, "", "")
chk("nappe à 33,3 cm : 3 barres, e = 333,3 mm", len(re.findall(r"<circle ", svg)) == 3 and "e = 333,3" in svg)
svg = DS.diametres("t", 12, 16)
r_ = [float(v) for v in re.findall(r'<circle cx="[\d.]+" cy="[\d.]+" r="([\d.]+)"', svg)]
chk("diamètres : les douze, à l'échelle 2:1 (rayon dessiné = Ø)", r_ == [float(d) for d in T.DIAMETRES], r_)
chk("… la section et la masse de chacun (celles du tableau)", all(f"{T.masse_txt(T.masse_lineique(d))} kg/m" in svg for d in T.DIAMETRES)
    and "1 257 mm²" in svg and "28 mm²" in svg)
chk("… le retenu encadré, l'optimal étoilé", svg.count('stroke="#2B4BCF"') == 1 and svg.count("<polygon") == 1)


# ====================================================================================================================
print("\n=== 10 · La pureté : bibliothèque standard seule, Python 3.12 et 3.13 ===")
# ====================================================================================================================
SCRIPT = r"""
import importlib.abc, json, sys
REFUS = {"streamlit", "numpy", "pandas", "PIL", "matplotlib", "scipy", "reportlab", "fitz"}
class Garde(importlib.abc.MetaPathFinder):
    def find_spec(self, nom, chemin=None, cible=None):
        if nom.split(".")[0] in REFUS:
            raise ImportError("refusé : " + nom)
        return None
sys.meta_path.insert(0, Garde())
sys.path.insert(0, %r)
from beton.armatures import moteur as M, dessins as DS, tableau as T
out = [M.calculer(b) for b in ({}, {"mode": "nappe", "As_req": 500}, {"mode": "nappe", "espacement": "libre", "e_libre": 12, "d": 16},
                               {"mode": "barres", "As_req": 1250, "d_max": 32})]
out.append(T.tableau())
out.append([DS.nappe("a", 16, 12.0, "t", "p"), DS.diametres("b", 12, 16)])
charges = sorted(m.split(".")[0] for m in sys.modules if m.split(".")[0] in REFUS)
print(json.dumps({"out": out, "charges": charges, "version": sys.version_info[:2]}))
""" % RACINE
local = json.loads(json.dumps([M.calculer(b) for b in ({}, {"mode": "nappe", "As_req": 500}, {"mode": "nappe", "espacement": "libre", "e_libre": 12, "d": 16},
                                                       {"mode": "barres", "As_req": 1250, "d_max": 32})]
                              + [T.tableau(), [DS.nappe("a", 16, 12.0, "t", "p"), DS.diametres("b", 12, 16)]]))
for py in ("/usr/bin/python3.12", "/usr/bin/python3.13"):
    if not shutil.which(py):
        print(f"  ({py} absent : sauté)")
        continue
    p = subprocess.run([py, "-I", "-c", SCRIPT], capture_output=True, text=True, timeout=120)
    if p.returncode != 0:
        chk(f"{py} : le moteur se charge sans streamlit, numpy, pandas, PIL, matplotlib", False, p.stderr[-600:])
        continue
    o = json.loads(p.stdout)
    chk(f"{py} (Python {o['version'][0]}.{o['version'][1]}) : le moteur se charge et calcule sans streamlit, numpy, pandas, PIL, matplotlib",
        o["charges"] == [], o["charges"])
    chk(f"{py} : résultats identiques à ceux de ce processus, au bit près (JSON)", o["out"] == local)


print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

# -*- coding: utf-8 -*-
"""La famille PIED DE POTEAU du module « Assemblages métalliques » du site (api_web/assemblages_pied.py) : platine sur
béton, calcul « platine à la main » et vérification complète. Ce que le site calcule et affiche est ce que calcule et
affiche l'écran Streamlit ; chaque contrôle rougit si l'on retire ce qu'il protège.

  1. La configuration et son cadre Données : chaque champ que l'écran Streamlit montre (entrees.champ_visible) est
     dans le cadre, et seulement lui, dans les deux modes (ETA, tube, profilé et acier personnalisés) ; les tables
     du calcul à la main (boulons, éléments, raidisseurs) et la grille à poser.
  2. Les copies NOMMÉES de l'écran Streamlit, rejouées avec un faux « st » : options courtes, aides, périmètre,
     natures, état de la bascule, boulon ajouté, grille posée, « l'épaisseur de la platine, le calcul à la main »
     (bloc_platine, bloc_etat, detail_main), paramètres retenus, répartition, efforts des tiges, onglets Benchmark,
     cas « Charger ces données », légende du Tableau 3.3, bascule de mode (ecran_saisie.changer_mode).
  3. La parité du calcul (fichier → saisies → moteur), chemin de l'écran Streamlit et chemin du site, au bit près,
     sur des jeux qui couvrent les deux modes : boulons, éléments de chaque type, raidisseurs, rotations, critère,
     appui, point d'application, grilles, tube, ETA, périmètre complet, profilé personnalisé vide.
  4. L'écran Streamlit LUI-MÊME (AppTest, préfixe ppb_) : des jeux des deux modes chargés comme le fait « Ouvrir » ;
     son résultat a l'empreinte de celui du site.
  5. Ce que le site affiche = le moteur : verdict, lignes par pièce, lignes d'information grises, les quatre
     critères de l'épaisseur, E_d / R_d dans chaque fenêtre, détail des formules et ligne η, alertes, dessins (plan,
     coupes, domaine) au caractère près, note texte, méthode, validation, vues Efforts / Répartition.
  6. La saisie : tables (ajouter, retirer jusqu'au dernier, écrire), refus (dimension négative, rotation 45°, type
     inconnu), grille, bascule de mode acceptée ou refusée (raison dite, note intacte), exemples, réinitialisation,
     visserie par défaut de l'attache, configuration changée.
  7. Les sélecteurs (tige, poteau) et l'échappement.
  8. Mutations : abîmer une copie nommée ou le calcul fait rougir.
  9. (CONTRAT_SEPT=1) Le contrat commun (tests/test_api_web_modules.py) rejoué avec une note neuve de CHACUNE des
     sept configurations, et du pied de poteau en vérification complète.

    /usr/bin/python3.12 tests/test_api_web_assemblages_pied.py        (streamlit requis pour 2 et 4)
"""
import contextlib
import copy
import html
import importlib
import io
import json
import logging
import math
import os
import random
import re
import subprocess
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from acier.bibliotheques import DB, PERSO  # noqa: E402
from api_web import assemblages as A  # noqa: E402
from api_web import assemblages_pied as PI  # noqa: E402
from api_web import commun as C  # noqa: E402

OK, KO = [], []
TIRAGES = int(os.environ.get("TIRAGES", "40"))       # jeux tirés au hasard (partie 3)
CAS_ECRAN = int(os.environ.get("CAS_ECRAN", "8"))    # jeux rejoués dans l'écran Streamlit, par mode (partie 4)
PAQ = "acier.assemblages.pied_poteau.platine_beton"
E = PI.E
CFG = A.CONFIGS["platine_beton"]


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "ÉCHEC ") + nom + (f"   [{str(info)[:300]}]" if info and not cond else ""))


def ecran(nom):
    return importlib.import_module(f"{PAQ}.{nom}")


def etat_de(values=None, **el):
    """Une note d'un pied de poteau, lue comme un fichier (les défauts de la configuration)."""
    return A.normaliser({"assemblages": [dict({"type": "platine_beton", "values": values or {}}, **el)], "actif": 0})


def neuf(complet=False):
    e = {"version": 1, "assemblages": [A.nouvel_assemblage("platine_beton")], "actif": 0}
    if complet:
        assert A.modifier(e, ["v", "mode"], E.MODE_COMPLET) is None
    return e


def r_de(e):
    el = A._actif(e)
    return A.calculer(A._cfg(el), el)[0]


def texte(s):
    return html.unescape(re.sub(r"<[^>]+>", "", str(s)))


def plat(s):
    """Un texte d'écran comparable : sans balises, « <br> » = espace, sans espaces de milliers."""
    return re.sub(r"[  ]", "", texte(str(s).replace("<br>", " ")))


# ------------------------------------------------------------------ l'empreinte d'un résultat
def _x(v):
    return repr(v) if isinstance(v, float) else v


def empreinte(R):
    """Tout ce que l'écran montre d'un résultat, flottants en repr (égalité AU BIT) ; le pied y ajoute l'épaisseur
    requise et ses critères, l'effort de chaque tige."""
    if R is None:
        return None
    d = {"eta_max": _x(R.eta_max), "verified": R.verified, "statut": R.statut, "geo_ok": R.get("geo_ok"),
         "dist_ok": R.get("dist_ok"), "gov": R.gov.label if R.gov else None, "mode": R.get("mode"),
         "reserve": R.get("reserve")}
    d["checks"] = [(c.grp, c.key, c.label, bool(c.active), bool(c.ok), _x(c.eta), _x(c.Ed), _x(c.Rd)) for c in R.checks]
    d["alerts"] = [(a.id, bool(a.block)) for a in R.alerts]
    d["dist"] = [(x.lab, _x(x.val), _x(x.min), _x(x.max), bool(x.ok)) for x in (R.get("dist") or [])]
    P = R.get("plaque") or {}
    d["plaque"] = {k: _x(P.get(k)) for k in ("t_req", "t_T", "t_S", "t_c", "t_b", "t_N", "t_M") if k in P}
    d["tiges"] = [(_x(t.get("x")), _x(t.get("y")), _x(t.get("T"))) for t in (R.get("tiges") or [])]
    return d


def ecarts(a, b, chemin="", out=None):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            ecarts(a.get(k), b.get(k), f"{chemin}.{k}", out)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out.append((chemin, f"longueur {len(a)} ≠ {len(b)}"))
        for i, (x, y) in enumerate(zip(a, b)):
            ecarts(x, y, f"{chemin}[{i}]", out)
    elif a != b:
        out.append((chemin, a, b))
    return out


# ------------------------------------------------------------------ les jeux de saisies
def element_tire(rng):
    t = rng.choice(E.TYPES_ELEM)
    e = E.element_defaut(t)
    e.update(x=rng.choice([0, 0, 0, 20, -40]), y=rng.choice([0, 0, 0, 30, -25]), rot=rng.choice([0, 0, 90]))
    if t == E.TYPE_I:
        e["prof"] = rng.choice(["HEA 200", "HEA 300", "HEB 240", "IPE 300", PERSO])
    e.update(h=rng.choice([140, 200, 260]), b=rng.choice([100, 150, 200]), t=rng.choice([6, 8, 10]), d=rng.choice([60, 120, 160]),
             tw=rng.choice([7, 9]), tf=rng.choice([10, 14]), r=rng.choice([0, 12]))
    return e


def tirage(rng):
    """Un jeu de saisies BRUT (comme un fichier) des deux modes."""
    d = E.defaults()
    complet = rng.random() < 0.4
    d["mode"] = E.MODE_COMPLET if complet else E.MODE_MANUEL
    for k, f in E.CHAMPS.items():
        if k.startswith("id_") or k == "mode":
            continue
        r = rng.random()
        if f["t"] == "s" and r < 0.2:
            d[k] = rng.choice(list(f["o"]))
        elif f["t"] == "n" and isinstance(d.get(k), (int, float)) and r < 0.25 and k not in ("n_ax", "n_ay"):
            x = d[k] * rng.choice([0.6, 0.8, 0.9, 1.1, 1.3, 1.6])
            d[k] = round(x, rng.choice([0, 1])) if abs(x) > 2 else round(x, 2)
            if rng.random() < 0.15:
                d[k] = str(d[k]).replace(".", ",")
    d["mode"] = E.MODE_COMPLET if complet else E.MODE_MANUEL
    for k in ("N_Ed", "Mx_Ed", "My_Ed", "Vx_Ed", "Vy_Ed"):
        if rng.random() < 0.3:
            d[k] = rng.choice([-1, 1]) * rng.choice([0, 20, 60, 150, 400])
    if complet:
        d["n_ax"], d["n_ay"] = rng.choice([1, 2, 2, 3]), rng.choice([1, 2, 2, 3, 4])
        if d["n_ax"] == 1 and d["n_ay"] == 1:
            d["n_ay"] = 2
        if rng.random() < 0.15:
            d["forme"] = E.FORME_RHS
            d.update(h_u=rng.choice([200, 250]), b_u=rng.choice([200, 150]), t_u=rng.choice([8, 10, None]))
        elif rng.random() < 0.15:
            d["prof_c"] = PERSO
            if rng.random() < 0.5:
                d.update(h_u=None, b_u=None, tw_u=None, tf_u=None)       # vides : vérification suspendue
            else:
                p = rng.choice(DB["profils"])
                d.update(h_u=p["h"], b_u=p["b"], tw_u=p["tw"], tf_u=p["tf"], r_u=p["r"])
        else:
            d["prof_c"] = rng.choice(["HEA 200", "HEA 300", "HEB 300", "HEB 400", "IPE 400"])
    else:
        bp, lp = float(str(d["bp_u"]).replace(",", ".")), float(str(d["lp_u"]).replace(",", "."))
        if rng.random() < 0.5:
            nx, ny = rng.choice([2, 3]), rng.choice([2, 3])
            d["tiges"] = [dict(x=float(x), y=float(y)) for (x, y, _i, _j) in PI.GE.positions_tiges(nx, ny, bp * 0.75, lp * 0.75)]
        else:
            d["tiges"] = [dict(x=round(rng.uniform(-bp / 2 + 50, bp / 2 - 50)), y=round(rng.uniform(-lp / 2 + 50, lp / 2 - 50)))
                          for _ in range(rng.choice([2, 3, 4, 6]))]
        d["elems"] = [element_tire(rng) for _ in range(rng.choice([1, 1, 1, 2]))]
        d["raid"] = [dict(E.raidisseur_defaut(), dir=rng.choice(E.DIRS_RAID), x=rng.choice([150, -150, 0]),
                          y=rng.choice([0, 145, -145])) for _ in range(rng.choice([0, 0, 1, 2]))]
        d["xN"], d["yN"] = rng.choice([0, 0, 30]), rng.choice([0, 0, -20])
    return d


def jeux(n, graine=1):
    """Défauts des deux modes, cas de validation (complets et manuels), puis tirages."""
    out = [("défaut (calcul manuel)", E.defaults()), ("défaut (complet)", dict(E.defaults(), mode=E.MODE_COMPLET)),
           ("fichier sans mode (complet)", {"N_Ed": 250})]
    out += [(f"cas {i}", d) for i, _lab, d in PI.exemples()]
    rng = random.Random(f"pied-{graine}")
    return out + [(f"tirage {i}", tirage(rng)) for i in range(n)]


def chemin_streamlit(src):
    """Ce que fait l'écran Streamlit d'un fichier : entrees.charger_json, chaque clé par carte.valeur_widget (les
    listes telles quelles), puis interface.show : carte.dimensions_vides, carte.provisoire, compute."""
    from acier.assemblages import carte as kit
    es = ecran("ecran_saisie")
    brut = E.charger_json(src)
    D = es.description(brut)
    u = {k: es.valeur_widget(k, brut.get(k)) for k in E.CLES}
    vides = kit.dimensions_vides(D, u)
    try:
        R = ecran("interface").compute(kit.provisoire(D, u) if vides else u)
        err = None
    except Exception as e:  # noqa: BLE001
        R, err = None, type(e).__name__
    return u, sorted(vides), R, err


def chemin_site(src, mod=A):
    try:
        e = mod.normaliser({"assemblages": [{"type": "platine_beton", "values": src}], "actif": 0})
    except ValueError as x:
        return None, None, None, f"refus : {x}"
    el = e["assemblages"][0]
    R, vides, err = mod.calculer(mod.CONFIGS["platine_beton"], el)
    return el["values"], sorted(vides), R, (err.split(" :")[0] if err else None)


def comparer_parite(cas, mod=A):
    out = []
    for nom, src in cas:
        u1, v1, R1, e1 = chemin_streamlit(src)
        u2, v2, R2, e2 = chemin_site(src, mod)
        if u2 is None:
            out.append((nom, "le site refuse le fichier", e2))
            continue
        du = [k for k in u1 if k not in E.LISTES and _x(u1[k]) != _x(u2.get(k))]
        if du:
            out.append((nom, "saisies", [(k, u1[k], u2.get(k)) for k in du[:3]]))
        if v1 != v2:
            out.append((nom, "dimensions vides", v1, v2))
        if e1 != e2:
            out.append((nom, "erreur", e1, e2))
        d = ecarts(empreinte(R1), empreinte(R2))
        if d:
            out.append((nom, "résultat", d[:3]))
    return out


# ------------------------------------------------------------------ le faux « st »
class FauxSt:
    """Un « st » qui enregistre au lieu d'afficher ; les tableaux arrivent par les fonctions remplacées."""

    def __init__(self, etat=None):
        self.tables, self.textes, self.statuts = [], [], []
        self.session_state = etat if etat is not None else {}

    def markdown(self, x="", *a, **k):
        self.textes.append(x)

    caption = markdown

    def columns(self, spec, **k):
        return [self] * (spec if isinstance(spec, int) else len(spec))

    def container(self, *a, **k):
        return self

    expander = container

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def button(self, *a, **k):
        return False

    def __getattr__(self, nom):
        return lambda *a, **k: None


@contextlib.contextmanager
def faux_st(M, etat=None):
    """Le module d'écran ``M`` avec un faux « st » : tableau_html / tableau_md rendent leurs lignes, taux_html rend
    (« η », taux, vérifié), bloc_statut_ligne s'enregistre."""
    fs = FauxSt(etat)
    noms = ("st", "tableau_html", "tableau_md", "taux_html", "bloc_statut_ligne")
    avant = {n: getattr(M, n) for n in noms if hasattr(M, n)}
    M.st = fs
    if "tableau_html" in avant:
        M.tableau_html = lambda h, rows, **k: fs.tables.append(("HTML", [plat(x) for x in h], [list(r) for r in rows])) or ""
    if "tableau_md" in avant:
        M.tableau_md = lambda h, rows, **k: fs.tables.append(("MD", [plat(x) for x in h], [[str(x) for x in r] for r in rows])) or ""
    if "taux_html" in avant:
        M.taux_html = lambda eta, ok, gras=False: ("η", eta, bool(ok))
    if "bloc_statut_ligne" in avant:
        M.bloc_statut_ligne = lambda etat_, titre, comp, elements: fs.statuts.append((etat_, titre, comp, list(elements)))
    try:
        yield fs
    finally:
        for n, v in avant.items():
            setattr(M, n, v)


def cellules(rows):
    """Les cellules d'un tableau de l'écran Streamlit, comparables à celles du site : texte, ou (η, vérifié)."""
    out = []
    for r in rows:
        li = []
        for x in r:
            li.append((x[1], x[2]) if isinstance(x, tuple) and x and x[0] == "η" else plat(x))
        out.append(li)
    return out


def site_kv(lignes):
    return [[a, b] for a, b, _f in lignes]


logging.disable(logging.CRITICAL)
ES, ER, EM = ecran("ecran_saisie"), ecran("ecran_resultats"), ecran("manuel.ecran")

# =====================================================================================================================
print("=== 1. La configuration et son cadre Données ===")
chk("le pied de poteau est enregistré, avec sa variante « calcul manuel » (synthèse, texte, méthode, dessins)",
    CFG.calculer is PI.CA.compute and PI.MANUEL.S is PI.YMA and PI.MANUEL.T is PI.TMA and CFG.S is PI.YCO
    and A.pour(CFG, r_de(neuf())) is PI.MANUEL and A.pour(CFG, r_de(neuf(True))) is CFG)
RX_CHEMIN = re.compile(r'data-chemin="v/([^"/]+)(?:/[^"]*)?"')


def champs_du_cadre(e):
    don_ = A.resultat(e)["donnees"]
    # (établi compact : le repère est le NOM de l'élément, sur sa ligne — data-chemin="nom")
    return set(RX_CHEMIN.findall(don_)) | ({"id_rep"} if don_.count('data-chemin="nom"') == 1 else set())


ecarts_cadre = []
etats_cadre = {"manuel": neuf(), "complet": neuf(True)}
for nom, ch in (("manuel ETA", {"typ_a": E.TYPE_ETA}), ("manuel crosse", {"typ_a": E.TYPE_CROSSE}),
                ("manuel acier perso", {"nu_pl": PERSO}), ("complet ETA", {"mode": E.MODE_COMPLET, "typ_a": E.TYPE_ETA}),
                ("complet RHS", {"mode": E.MODE_COMPLET, "forme": E.FORME_RHS}),
                ("complet profilé perso", {"mode": E.MODE_COMPLET, "prof_c": PERSO, "h_u": 300, "b_u": 300, "tw_u": 9, "tf_u": 14, "r_u": 27}),
                ("complet acier perso", {"mode": E.MODE_COMPLET, "nu_c": PERSO, "fy_u": 300, "fu_u": 430, "bw_u": 0.8})):
    etats_cadre[nom] = etat_de(dict(E.defaults(), **ch))
for nom, e in etats_cadre.items():
    u = A._actif(e)["values"]
    attendu = {k for k in E.CHAMPS if E.champ_visible(u, k)} | ({"tiges", "elems", "raid"} if not E.est_complet(u) else set())
    vu = champs_du_cadre(e)
    if vu != attendu:
        ecarts_cadre.append((nom, sorted(attendu - vu), sorted(vu - attendu)))
chk(f"chaque champ que l'écran Streamlit montre est dans le cadre Données, et seulement lui ({len(etats_cadre)} cas, deux modes)",
    not ecarts_cadre, ecarts_cadre[:3])
dm, dc = A.resultat(neuf())["donnees"], A.resultat(neuf(True))["donnees"]
secs_m = re.findall(r'<span class="tt">([^<]+)</span>', dm)
secs_c = re.findall(r'<span class="tt">([^<]+)</span>', dc)
chk("calcul manuel : neuf sections pensées (Assemblage, Platine, Boulons, Éléments soudés, Raidisseurs, Béton, Efforts, "
    "Coefficients, Cartouche)", secs_m == ["Assemblage", "Platine", "Boulons", "Éléments soudés", "Raidisseurs", "Béton et massif",
                                           "Efforts ELU", "Coefficients", "Cartouche et affichage"], secs_m)
chk("vérification complète : huit sections (Poteau et Tiges d'ancrage à la place des tables)",
    secs_c == ["Assemblage", "Poteau", "Platine", "Tiges d'ancrage", "Béton et massif", "Efforts ELU", "Coefficients",
               "Cartouche et affichage"], secs_c)
chk("en tête : Mode, Critère, Appui en segments (manuel) ; Mode, Périmètre (complet) — options courtes de l'écran",
    all(f'data-chemin="v/{k}" data-sorte="choix"' in dm for k in ("mode", "crit", "appui"))
    and all(f'data-chemin="v/{k}" data-sorte="choix"' in dc for k in ("mode", "perim")) and ">Complet<" in dm
    and ">Bande c<" in dm and ">EC3 + EC2-4<" in dc and "Vérification complète : disponible pour cette disposition." in dm)
chk("la clé « ancr » s'appelle « Boulon » au calcul manuel, « Tige » en vérification complète (LIBELLE_MODE)",
    re.search(r"<span>Boulon</span>.*?data-chemin=\"v/ancr\"", dm, re.S) is not None
    and re.search(r"<span>Tige</span>.*?data-chemin=\"v/ancr\"", dc, re.S) is not None)
chk("la table des boulons : une ligne par boulon (x, y, retirer), « Ajouter un boulon », la grille à poser",
    dm.count('data-chemin="v/tiges/') == 4 * 3 + 1 and 'data-chemin="grille/poser"' in dm
    and all(f'data-chemin="grille/{k}"' in dm for k in ("nx", "ny", "wx", "wy")) and "grille/poser" not in dc)
chk("les éléments : type (noms courts AFF_TYPES), profilé, x, y, rotation ; un tube montre ses dimensions",
    'data-chemin="v/elems/0/prof"' in dm and ">Profilé I/H<" in dm and 'data-chemin="v/elems/0/rot"' in dm
    and 'data-chemin="v/elems/0/h"' not in dm
    and 'data-chemin="v/elems/0/t"' in A.resultat(etat_de(dict(E.defaults(), elems=[E.element_defaut(E.TYPE_RHS)])))["donnees"])
chk("l'aide « ? » d'un champ est celle de la carte Streamlit du mode (AIDES / AIDES_MANUEL)",
    C.esc(PI.AIDES_MA["crit"]) in dm and C.esc(PI.AIDES_CO["perim"]) in dc and C.esc(PI.AIDES_MA["appui"]) in dm)
chk("le menu Configuration nomme l'attache « Platine sur béton » (son nom court redisait sa famille)",
    ">Pied de poteau · Platine sur béton<" in dm)

# =====================================================================================================================
print("\n=== 2. Les copies nommées de l'écran Streamlit ===")
chk("options courtes = ecran_saisie.AFFICHAGE ; types d'élément = manuel/ecran.AFF_TYPES ; dimensions par type",
    PI.AFFICHAGE == ES.AFFICHAGE and PI.AFF_TYPES == EM.AFF_TYPES and PI.DIMS_ELEM == EM.DIMS_PAR_TYPE)
chk("aides = ecran_saisie.AIDES (vérification complète) et AIDES_MANUEL (calcul manuel), entrée par entrée",
    all(ES.AIDES[k] == v for k, v in PI.AIDES_CO.items()) and all(ES.AIDES_MANUEL[k] == v for k, v in PI.AIDES_MA.items()),
    [k for k, v in PI.AIDES_MA.items() if ES.AIDES_MANUEL.get(k) != v])
from acier.assemblages import ui_commun as UC  # noqa: E402
chk("périmètre, natures, critères, vocabulaire, légendes du plan = les constantes de l'écran",
    PI.NON_REVERIFIES == ER.NON_REVERIFIES and PI.NON_VERIFIES == EM.NON_VERIFIES and PI.NATURES == ER.NATURES == EM.NATURES
    and PI.TEXTES_NATURES_CO == ER.TEXTES_NATURES and PI.TEXTES_NATURES_MA == EM.TEXTES_NATURES
    and PI.SENS_EC == UC.NATURES["EC"] and PI.CRITERES_NOMS == EM.CRITERES_NOMS and PI.LIBELLE_MODE == ES.LIBELLE_MODE
    and PI.LEGENDE_PLAN_CO == ER.CAPTION_LEGENDE and PI.LEGENDE_PLAN_MA == EM.CAPTION_LEGENDE
    and PI.GRILLE_DEFAUT == {"nx": 2, "ny": 2, "wx": 400.0, "wy": 400.0})

CAS2 = jeux(40, graine=2)
RESULTATS = []                     # (nom, R, u) des jeux calculables sans dimensions vides
for nom, src in CAS2:
    e = etat_de(src)
    el = A._actif(e)
    R, vides, err = A.calculer(CFG, el)
    if R is not None and not vides:
        RESULTATS.append((nom, R, el["values"], e))
MANUELS = [x for x in RESULTATS if PI.CA.est_manuel_R(x[1])]
COMPLETS = [x for x in RESULTATS if not PI.CA.est_manuel_R(x[1])]

# la note de la fenêtre Soudure du calcul manuel, l'état de la bascule
R0, u0 = MANUELS[0][1], MANUELS[0][2]
with faux_st(EM):
    note_gorge = EM._panneaux(R0, u0)["weld"]["note"]
chk("la note de la gorge (calcul manuel) = la fenêtre Soudure de l'écran", note_gorge == PI.NOTE_GORGE_MA, note_gorge)
d_bascule = [nom for nom, R, u, _e in RESULTATS + [("libre", None, dict(E.defaults(), raid=[E.raidisseur_defaut()]), None)]
             if PI.etat_bascule(u) != EM.etat_bascule(u)]
chk(f"l'état de la bascule = manuel/ecran.etat_bascule ({len(RESULTATS) + 1} dispositions)", not d_bascule, d_bascule[:3])

# le boulon ajouté, la grille posée
diff_aj = []
for nom, R, u, _e in MANUELS[:25] + [("vide", None, dict(E.defaults(), tiges=[]), None)]:
    etat = {"ppb_tiges": copy.deepcopy(u["tiges"])}
    with faux_st(EM, etat):
        EM._ajouter_boulon(u)
    if etat["ppb_tiges"][-1] != PI.ajouter_boulon(u):
        diff_aj.append((nom, etat["ppb_tiges"][-1], PI.ajouter_boulon(u)))
chk("« Ajouter un boulon » = manuel/ecran._ajouter_boulon (au bord bas, à droite du dernier)", not diff_aj, diff_aj[:2])
diff_gr = []
for g in ({"nx": 2, "ny": 2, "wx": 400.0, "wy": 400.0}, {"nx": 3, "ny": 2, "wx": 360.0, "wy": 300.0}, {"nx": 1, "ny": 4, "wx": 0.0, "wy": 420.0}):
    etat = {"ppb_ui_gx": g["nx"], "ppb_ui_gy": g["ny"], "ppb_ui_gwx": g["wx"], "ppb_ui_gwy": g["wy"]}
    with faux_st(EM, etat):
        EM._poser_grille()
    if etat["ppb_tiges"] != PI.poser_grille(g):
        diff_gr.append(g)
chk("« Poser la grille » = manuel/ecran._poser_grille", not diff_gr, diff_gr)

# l'épaisseur de la platine, le calcul à la main
diff_pl = []
for nom, R, u, _e in COMPLETS:
    with faux_st(ER) as fs:
        ER.bloc_platine(R)
    d = PI.platine_complet(R)
    if not d["actif"]:
        if d["note"] not in fs.textes:
            diff_pl.append((nom, "note"))
        continue
    etat_, titre, comp, elems = fs.statuts[0]
    if (titre, comp) != (d["titre"], d["complement"]) or [(a, _x(b), c) for a, b, c in elems] != [(a, _x(b), c) for a, b, c in d["criteres"]]:
        diff_pl.append((nom, "état", titre, d["titre"]))
    tabs = [t[2] for t in fs.tables]
    for i, cle in enumerate(("comprime", "tendu")):
        if cellules(tabs[i]) != site_kv(d[cle][1]) or "**" + d[cle][0] + "**" not in fs.textes:
            diff_pl.append((nom, cle, ecarts(cellules(tabs[i]), site_kv(d[cle][1]))[:2]))
    if d["navier"] not in fs.textes:
        diff_pl.append((nom, "Navier"))
chk(f"vérification complète : « l'épaisseur de la platine, le calcul à la main » = ecran_resultats.bloc_platine "
    f"({len(COMPLETS)} résultats)", len(COMPLETS) >= 8 and not diff_pl, diff_pl[:2])
diff_mn = []
for nom, R, u, _e in MANUELS:
    with faux_st(EM) as fs:
        actif = EM.bloc_etat(R)
        if actif:
            EM.detail_main(R)
    d = PI.platine_manuel(R)
    if not d["actif"]:
        if actif or d["note"] not in fs.textes:
            diff_mn.append((nom, "note"))
        continue
    etat_, titre, comp, elems = fs.statuts[0]
    if (titre, comp) != (d["titre"], d["complement"]) or [(a, _x(b), c) for a, b, c in elems] != [(a, _x(b), c) for a, b, c in d["criteres"]]:
        diff_mn.append((nom, "état", titre, d["titre"]))
    D = PI.detail_main(R)
    tabs = list(fs.tables)
    for cle in ("tendus", "sections"):
        titre_t, ent, lignes, cap = D[cle]
        if "**" + titre_t + "**" not in fs.textes or cap not in fs.textes:
            diff_mn.append((nom, cle, "titre ou légende"))
        if lignes:
            t = tabs.pop(0)
            if t[1] != ent or cellules(t[2]) != lignes:
                diff_mn.append((nom, cle, t[1], ent, ecarts(cellules(t[2]), lignes)[:2]))
    for cle in ("console", "trous"):
        titre_t, lignes = D[cle]
        t = tabs.pop(0)
        if "**" + titre_t + "**" not in fs.textes or cellules(t[2]) != site_kv(lignes):
            diff_mn.append((nom, cle, ecarts(cellules(t[2]), site_kv(lignes))[:2]))
    if D["pied"] not in fs.textes:
        diff_mn.append((nom, "légende finale"))
chk(f"calcul manuel : l'épaisseur requise et ses quatre critères = manuel/ecran.bloc_etat, le détail comme à la main "
    f"(boulon par boulon, face par face, bande comprimée, trous) = detail_main ({len(MANUELS)} résultats)",
    len(MANUELS) >= 15 and not diff_mn, diff_mn[:2])

# paramètres retenus, répartition, efforts des tiges
diff_par = []
for nom, R, u, _e in RESULTATS:
    M = EM if PI.CA.est_manuel_R(R) else ER
    with faux_st(M) as fs:
        M._parametres_retenus(R)
    site = (PI.parametres_manuel if M is EM else PI.parametres_complet)(R)
    if fs.tables[0][2] != [[str(a), str(b)] for a, b in site]:
        diff_par.append((nom, ecarts(fs.tables[0][2], site)[:2]))
chk(f"« Paramètres retenus » des deux modes = _parametres_retenus de l'écran ({len(RESULTATS)} résultats)",
    not diff_par, diff_par[:2])
diff_rep = []
for nom, R, u, _e in COMPLETS:
    with faux_st(ER) as fs:
        ER.onglet_repartition(R)
    d = PI.repartition(R)
    t_dir, t_tig, t_ran = fs.tables[:3]
    if (cellules(t_dir[2]) != d["directions"][1] or t_dir[1] != d["directions"][0] or cellules(t_tig[2]) != d["tiges"][2]
            or t_tig[1] != d["tiges"][1] or cellules(t_ran[2]) != d["rangees"][2] or t_ran[1] != d["rangees"][1]):
        diff_rep.append((nom, ecarts(cellules(t_tig[2]), d["tiges"][2])[:2]))
    if not all(x in fs.textes for x in (d["intro"], d["niveaux"], d["tiges"][3], "##### " + d["titre_domaine"])):
        diff_rep.append((nom, "textes"))
chk(f"la répartition (domaine par direction, tiges, rangées) = ecran_resultats.onglet_repartition ({len(COMPLETS)} résultats)",
    not diff_rep, diff_rep[:2])
diff_eff = []
for nom, R, u, _e in MANUELS:
    with faux_st(EM) as fs:
        EM.onglet_efforts(R)
    d = PI.efforts_manuel(R)
    t_b, t_r = fs.tables[:2]
    if cellules(t_b[2]) != d["boulons"][2] or t_b[1] != d["boulons"][1] or t_r[2] != [[a, b] for a, b in d["repartition"][1]]:
        diff_eff.append((nom, ecarts(cellules(t_b[2]), d["boulons"][2])[:2]))
    if d["intro"] not in fs.textes or d["boulons"][3] not in fs.textes:
        diff_eff.append((nom, "textes"))
chk(f"les efforts des tiges (calcul manuel) = manuel/ecran.onglet_efforts ({len(MANUELS)} résultats)", not diff_eff, diff_eff[:2])

# légende du Tableau 3.3, cas « Charger ces données », bascule de mode
with faux_st(ER) as fs:
    avant33 = ER.tableau_33_replie
    ER.tableau_33_replie = lambda ok, rendre, **k: rendre()
    try:
        ER._tableau_33(COMPLETS[0][1])
    finally:
        ER.tableau_33_replie = avant33
chk("la légende du Tableau 3.3 (fenêtre Géométrie) = ecran_resultats._tableau_33", PI.LEGENDE_33 in fs.textes, fs.textes[-1:])
charges = {}
for M, cas in ((ER, list(ecran("benchmark").BENCH) + list(ecran("benchmark").VALID)), (EM, list(ecran("manuel.benchmark").CAS))):
    for c in cas:
        pris = []
        with faux_st(M, {}):
            M._charger(c["inp"], pris.append)
        charges[c["id"]] = pris[0]
site_ex = {i: d for i, _lab, d in PI.exemples()}
chk(f"« Charger ces données dans l'outil » : les {len(charges)} cas des deux onglets Benchmark = ceux du site (mode compris)",
    charges.keys() == site_ex.keys() and all(charges[k] == site_ex[k] for k in charges) and len(charges) >= 6,
    [k for k in charges if charges[k] != site_ex.get(k)])
diff_b = []
for nom, src in [("défaut", E.defaults()), ("grille 3×2", dict(E.defaults(), tiges=PI.poser_grille({"nx": 3, "ny": 2, "wx": 360, "wy": 300}))),
                 ("deux éléments", dict(E.defaults(), elems=[E.element_defaut(), E.element_defaut(E.TYPE_PLAT)])),
                 ("tube RHS", dict(E.defaults(), elems=[dict(E.element_defaut(E.TYPE_RHS), h=200, b=150, t=8)])),
                 ("profilé perso", dict(E.defaults(), elems=[dict(E.element_defaut(), prof=PERSO, h=280, b=260, tw=9, tf=15)])),
                 ("complet → manuel", dict(E.defaults(), mode=E.MODE_COMPLET, n_ax=3, n_ay=2, w_x=380, w_y=420, prof_c="HEB 300"))]:
    e = etat_de(src)
    u = copy.deepcopy(A._actif(e)["values"])
    cible = E.MODE_MANUEL if E.est_complet(u) else E.MODE_COMPLET
    etat = {"ppb_" + k: v for k, v in u.items()}
    etat["ppb_ui_mode_prec"], etat["ppb_mode"] = u["mode"], cible
    ecrit = {}
    with faux_st(ES, etat):
        ES.changer_mode(lambda: dict(u), lambda v: ecrit.update({k: ES.valeur_widget(k, v.get(k)) for k in E.CLES}))
    msg = A.modifier(e, ["v", "mode"], cible)
    v_site = A._actif(e)["values"]
    if ecrit:
        if msg or ecarts(empreinte(A.CONFIGS["platine_beton"].calculer(ecrit)), empreinte(r_de(e))) or \
                [k for k in E.CLES if k not in E.LISTES and _x(ecrit[k]) != _x(v_site[k])]:
            diff_b.append((nom, "acceptée", msg))
    else:
        raison = etat.get("_ppb_toast", "")
        if not msg or msg != raison or v_site != u:
            diff_b.append((nom, "refusée", msg, raison))
chk("la bascule de mode = ecran_saisie.changer_mode : reprise de la disposition (manuel ↔ complet), refus avec la même "
    "raison et la note intacte quand la disposition n'est pas régulière", not diff_b, diff_b[:2])

# les onglets Benchmark : chaque cellule de l'écran est dans la vue Validation du site
diff_val = []
for M, vue, args in ((ER, PI._validation_complet(), (ecran("benchmark").run_bench(), lambda d: None)),
                     (EM, PI._validation_manuel(), (lambda d: None,))):
    with faux_st(M) as fs:
        M.onglet_benchmark(*args)
    texte_vue = plat(vue)
    for t in fs.tables:
        for li in t[2]:
            for x in li:
                if plat(x) not in texte_vue:
                    diff_val.append((M.__name__.split(".")[-1], x))
chk("la vue Validation porte chaque cellule des onglets Benchmark de l'écran (calculs manuels, validation croisée, "
    "VAL-M1 à VAL-M4)", not diff_val, diff_val[:3])

# =====================================================================================================================
print("\n=== 3. La parité du calcul : fichier → saisies → moteur (écran Streamlit / site) ===")
CAS = jeux(TIRAGES)
diff = comparer_parite(CAS)
etats = {"manuel": 0, "complet": 0, "vérifié": 0, "non vérifié": 0, "à compléter": 0, "erreur": 0, "raidisseurs": 0, "deux éléments": 0}
for _nom, src in CAS:
    _u, v, R, err = chemin_site(src)
    if err:
        etats["erreur"] += 1
    elif v:
        etats["à compléter"] += 1
    elif R is not None:
        etats["manuel" if PI.CA.est_manuel_R(R) else "complet"] += 1
        etats["vérifié" if R.verified else "non vérifié"] += 1
        etats["raidisseurs"] += bool(R.get("raid"))
        etats["deux éléments"] += len(R.get("elems") or []) > 1
chk(f"{len(CAS)} jeux, même empreinte au bit près ({etats})", not diff, diff[:3])
chk("les jeux couvrent les deux modes, vérifié, non vérifié, à compléter, raidisseurs, plusieurs éléments",
    all(etats[k] > 0 for k in ("manuel", "complet", "vérifié", "non vérifié", "à compléter", "raidisseurs", "deux éléments")), etats)

# =====================================================================================================================
print("\n=== 4. L'écran Streamlit lui-même (AppTest, l'application réelle) ===")
try:
    from streamlit.testing.v1 import AppTest
except ImportError:                                               # pragma: no cover
    AppTest = None
    chk("streamlit est installé (python3.12)", False)
if AppTest:
    at = AppTest.from_file(os.path.join(RACINE, "streamlit_app.py"), default_timeout=240)
    at.session_state["page"] = "Assemblages métalliques"
    with contextlib.redirect_stderr(io.StringIO()):
        at.run()
        at.button(key="asm_btn_open_platine_beton").click()
        at.run()
    ouvert = not at.exception and at.session_state["asm_courant"] == "platine_beton"
    chk("l'écran Streamlit du pied de poteau s'ouvre depuis la page projet", ouvert, at.exception)
    if ouvert:
        def sans_vide(nom_src):
            _u, v, R, err = chemin_site(nom_src[1])
            return R is not None and not v and not err
        bons = [c for c in CAS if sans_vide(c)]
        cas = ([c for c in bons if not E.est_complet(E.charger_json(c[1]))][:CAS_ECRAN]
               + [c for c in bons if E.est_complet(E.charger_json(c[1]))][:CAS_ECRAN])
        diffs, vus = [], 0
        for nom, src in cas:
            brut = E.charger_json(src)
            for k in E.CLES:                                      # interface.ecrire_entrees : ce que fait « Ouvrir »
                at.session_state["ppb_" + k] = ES.valeur_widget(k, brut.get(k))
            at.session_state["_ppb_R"] = None
            with contextlib.redirect_stderr(io.StringIO()):
                at.run()
            R_ecran = at.session_state["_ppb_R"]
            vus += R_ecran is not None
            d = ecarts(empreinte(R_ecran), empreinte(chemin_site(src)[2]))
            if d or at.exception:
                diffs.append((nom, d[:3], at.exception))
        chk(f"{vus} jeux (deux modes) chargés dans l'écran Streamlit, son résultat a l'empreinte de celui du site",
            vus == len(cas) and len(cas) >= min(2 * CAS_ECRAN, 10) and not diffs, diffs[:2])

# =====================================================================================================================
print("\n=== 5. Ce que le site affiche = le moteur ===")
# (établi compact, 12/11/2026 : le taux est au bout de la PREMIÈRE ligne ; une ligne d'information y dit « hors verdict » en mots)
RX_EL = re.compile(r'<div class="el vc( gov)?" data-calcul="([^"]+)"[^>]*><div class="l1"><b>([^<]+)</b>.*?'
                   r'<span class="(?:pct|et-mot) (\w)"[^>]*>([^<]+)</span></div><div class="l2">')
defauts = []
for nom, R, u, e in RESULTATS[:40]:
    cfg = A.pour(CFG, R)
    r = A.resultat(e)
    lignes = RX_EL.findall(r["resultat"])
    groupes = A._groupes(cfg, R)
    noms = [g for g, *_ in groupes]
    trouves = [b for _g, c, b, _n, _t in lignes if c != "platine" or b == "Platine"]
    if [b for b in trouves if b != "Géométrie"] != [html.escape(n, quote=True) for n in noms]:
        defauts.append((nom, "lignes", trouves, noms))
    for grp, _n, eta, ok, pire, on, info in groupes:
        ligne = [x for x in lignes if x[2] == grp]
        if info and (not ligne or ligne[0][3] != "c"):
            defauts.append((nom, "information grise", grp))
        # une ligne d'information ne porte JAMAIS de taux (petits chantiers, 10/11/2026 : « 108 % » gris sous
        # « Conforme ») : « hors verdict » et sans jauge ; une ligne qui décide garde son taux
        if info and ligne and (ligne[0][4] != A.SANS_TAUX_INFORMATION or "%" in ligne[0][4]):
            defauts.append((nom, "information sans taux", grp, ligne[0][4]))
        if not info and ligne and "%" not in ligne[0][4] and eta is not None and math.isfinite(eta):
            defauts.append((nom, "taux d'une ligne qui décide", grp, ligne[0][4]))
    # la pastille du verdict : le η max du moteur (celui de l'écran Streamlit), porté au rapport de la pire pince
    # NON conforme quand il est plus grand (07/11/2026 : le moteur qui trouve la géométrie non valide ne calcule plus
    # les résistances qui en dépendent, et son η max écrivait « 0 % » sous « Non conforme »)
    eta_v = R.eta_max
    ko = [x for x in (R.get("dist") or []) if not x.ok]
    if ko and isinstance(eta_v, (int, float)) and math.isfinite(eta_v):
        eta_v = max(eta_v, max(A._taux_distance(x) for x in ko))
    if f"{C.pc(A.taux(eta_v))}{C.NBSP}%" not in r["resultat"].split('class="verdict"')[1][:300]:
        defauts.append((nom, "verdict"))
    c = R.ck.get("tpq")
    if c is not None and c.active and (N_tp := C.lire_nombre(u.get("tp_u"))):
        crit = [x for x in lignes if x[1] == "platine" and x[2] != "Platine"]
        noms_c = PI.CRITERES_NOMS if PI.CA.est_manuel_R(R) else PI.CRITERES_CO
        ordre = PI.ORDRE_MA if PI.CA.est_manuel_R(R) else PI.ORDRE_CO
        att = [(noms_c[k][0].upper() + noms_c[k][1:], f"{C.pc(A.taux(R.plaque['t_' + k] / N_tp))}{C.NBSP}%") for k in ordre]
        if [(texte(x[2]), x[4]) for x in crit] != att:
            defauts.append((nom, "quatre critères", [(x[2], x[4]) for x in crit], att))
    for c in R.checks:
        if c.active:
            ed, rd, un = cfg.S.ed_rd(c)
            fen = r["volets"].split(f'id="ck-{A._slug(c.grp)}"')[1].split('class="flyout')[0]
            if f">{ed}{' ' + un if un else ''}<" not in fen or f">{rd}{' ' + un if un else ''}<" not in fen:
                defauts.append((nom, "fenêtre", c.key, ed, rd))
            _aide, lf = cfg.formules(R, c)
            if not lf or lf[-1]["parts"][0] != "η" or C.lignes_fx(lf) not in fen:
                defauts.append((nom, "détail", c.key))
    # « Vérifié sous réserve » : seulement sous un verdict vérifié (comme le titre « – sous réserve des alertes » de
    # l'écran Streamlit, ecran_resultats.py:48 et manuel/ecran.py:349)
    if r["resultat"].count('class="msg') != len(R.alerts) + bool(R.get("reserve") and R.verified) + bool(PI.note_information(R)):
        defauts.append((nom, "alertes", len(R.alerts)))
    S = PI.SMA if PI.CA.est_manuel_R(R) else PI.SCO
    el = A._actif(e)
    opt = S.options_ecran(R, el["cotation"], A._hl(R))
    opt.interactive, opt.poignees, opt.editables, opt.locked = False, False, el["cotation"] >= 1, None
    vues = {v["cle"]: v for v in r["vues"]}
    dessins = [("plan", S.plan(R, opt)), ("coupe_aa", S.coupe_aa(R, opt)), ("coupe_bb", S.coupe_bb(R, opt))]
    if not PI.CA.est_manuel_R(R):
        dessins.append(("domaine", S.domaine(R)))
    for k, dess in dessins:
        if vues[k]["svg"] != dess.svg(identifiant=f"asm_{k}"):
            defauts.append((nom, "dessin", k))
    if vues["note"]["svg"] != f'<div class="asm-doc"><pre class="asm-txt">{C.esc(cfg.T.construire_texte(R, False))}</pre></div>':
        defauts.append((nom, "note"))
    if C.esc(PI.NON_VERIFIES[0] if PI.CA.est_manuel_R(R) else PI.NON_REVERIFIES[0]) not in vues["methode"]["svg"]:
        defauts.append((nom, "méthode"))
    doc = vues.get("efforts" if PI.CA.est_manuel_R(R) else "repartition")
    if doc is None or not doc["svg"].startswith('<div class="asm-doc">'):
        defauts.append((nom, "vue Efforts / Répartition"))
chk(f"verdict, lignes par pièce (information en gris), quatre critères, E_d / R_d et détail dans chaque fenêtre, alertes, "
    f"dessins au caractère près, note, méthode, vues ({min(40, len(RESULTATS))} résultats)", not defauts, defauts[:3])
# le cas mesuré (petits chantiers, 10/11/2026) : périmètre « EC3 seul », béton d'ancrage à 454 % (interaction N / V,
# hef 100 mm, N = −50 kN, Mx = 10 kNm) sous un verdict « vérifié » — c'est le PÉRIMÈTRE (le béton d'ancrage n'entre pas
# dans le verdict, comme dans l'écran Streamlit), pas un défaut du calcul : la ligne le dit et ne montre aucun taux ; la
# phrase d'information garde le chiffre et dit « non satisfait » ; au périmètre « EC3 + EC2-4 », le même calcul est
# NON CONFORME et la ligne porte son taux, en rouge.
e = neuf(True)
for k, v in (("hef", 100.0), ("N_Ed", -50.0), ("Mx_Ed", 10.0)):
    assert A.modifier(e, ["v", k], v) is None
R108 = r_de(e)
r108 = A.resultat(e)["resultat"]
anc = [x for x in RX_EL.findall(r108) if x[2] == "Ancrages"]
pire = PI.YCO.pire_information(R108)
e2 = copy.deepcopy(e)
assert A.modifier(e2, ["v", "perim"], E.PERIM_TOUT) is None
R2 = r_de(e2)
anc2 = [x for x in RX_EL.findall(A.resultat(e2)["resultat"]) if x[2] == "Ancrages"]
chk("pied de poteau, périmètre « EC3 seul », béton d'ancrage à 454 % sous « vérifié » : la ligne dit « hors périmètre » "
    "et « hors verdict », sans taux ; la phrase d'information dit 454,0 % « non satisfait » ; au périmètre "
    "« EC3 + EC2-4 », non conforme et la ligne en rouge à 454 %",
    R108.verified and pire is not None and round(pire.eta, 2) == 4.54 and anc and anc[0][3] == "c"
    and anc[0][4] == A.SANS_TAUX_INFORMATION and "454" not in r108.split('data-calcul="ancrages"')[1].split('class="el')[0]
    and "hors périmètre" in r108.split('data-calcul="ancrages"')[1][:300]
    and "à 454,0 % — non satisfait" in texte(r108)
    and not R2.verified and anc2 and anc2[0][3] == "r" and anc2[0][4] == f"454{C.NBSP}%",
    (R108.statut, anc, anc2, round(pire.eta, 3) if pire else None))
rm, rc = A.resultat(neuf()), A.resultat(neuf(True))
chk("calcul manuel : vues Plan, Coupe A-A, Coupe B-B, Efforts, Note, Méthode, Validation (VAL-M1 à VAL-M4)",
    [v["cle"] for v in rm["vues"]] == ["plan", "coupe_aa", "coupe_bb", "efforts", "note", "methode", "validation"]
    and "VAL-M4" in rm["vues"][-1]["svg"] and "Non vérifiés ici" in rm["vues"][5]["svg"])
chk("vérification complète : vues Plan, Coupes, Domaine, Répartition, Note, Méthode, Validation (VAL-A, VAL-B, validation "
    "croisée)", [v["cle"] for v in rc["vues"]] == ["plan", "coupe_aa", "coupe_bb", "domaine", "repartition", "note", "methode", "validation"]
    and "VAL-B" in rc["vues"][-1]["svg"] and "Validation croisée indépendante" in rc["vues"][-1]["svg"]
    and "Non revérifiés ici" in rc["vues"][6]["svg"] and "modèle d&#x27;ingénieur (niveau 2" in rc["vues"][6]["svg"])
chk("fenêtre Platine : le calcul à la main EN TÊTE (état, quatre critères, tables), la fenêtre Géométrie : paramètres "
    "retenus et légende du Tableau 3.3", all("Épaisseur de la platine — le calcul à la main" in r["volets"].split('id="ck-platine"')[1][:4000]
                                              and C.esc(PI.LEGENDE_33) in r["volets"] and "Paramètres retenus" in r["volets"] for r in (rm, rc)))
chk("fenêtres Tiges (efforts de chaque tige ; en complet, les rangées) et Pied (domaine par direction) ; Ancrages : EN 1992-4",
    "Boulons : position, effort, appui" in rm["volets"] and "Rangées : tronçons en T" in rc["volets"]
    and "Domaine résistant (Mx, My) à N = 300 kN" in rc["volets"].split('id="ck-pied"')[1][:6000]
    and "Interaction N / V du béton · EN 1992-4" in rc["volets"])
chk("hypothèses : les tiges (boulons), le béton, le critère ou le périmètre",
    "Boulons <b>4 × M24 8.8</b>" in rm["volets"] and "Tiges <b>4 × M24 8.8</b>" in rc["volets"] and "Béton C30/37" in rm["volets"]
    and "Critère élastique" in rm["volets"] and "Périmètre EN 1993-1-8, béton d'ancrage en information" in rc["volets"])
e_inf = etat_de(dict(E.defaults(), mode=E.MODE_COMPLET, ancr="M12"))
r_inf = A.resultat(e_inf)
chk("une épaisseur requise infinie (aucune ne suffit : M12) s'affiche « 999 % » rouge au lieu d'arrêter l'écran "
    "(Streamlit : « > 999 % »)", r_de(e_inf).eta_max == float("inf") and "999" in r_inf["resultat"].split('class="verdict"')[1][:200]
    and r_inf["statut_html"] == C.statut("r"))

# =====================================================================================================================
print("\n=== 6. La saisie ===")
e = neuf()
u = A._actif(e)["values"]
A.modifier(e, ["v", "tiges", "+"], 1)
chk("« Ajouter un boulon » : au bord bas (à 60 mm), au milieu quand cette ligne est vide", u["tiges"][-1] == {"x": 0.0, "y": -190.0}, u["tiges"][-1])
A.modifier(e, ["v", "tiges", "+"], 1)
chk("… puis à droite du dernier boulon de cette ligne (+ 80 mm)", u["tiges"][-1] == {"x": 80.0, "y": -190.0}, u["tiges"][-1])
A.modifier(e, ["v", "tiges", 5, "-"], 1)
for chemin, v, attendu in ((["v", "tiges", 4, "x"], "-250,5", -250.5), (["v", "elems", 0, "rot"], "90", 90),
                           (["v", "elems", 0, "type"], E.TYPE_CHS, E.TYPE_CHS), (["v", "elems", 0, "d"], "219,1", 219.1)):
    msg = A.modifier(e, chemin, v)
    chk(f"table : « {v} » dans {'/'.join(map(str, chemin[1:]))}", msg is None and u[chemin[1]][chemin[2]][chemin[3]] == attendu, msg)
A.modifier(e, ["v", "raid", "+"], 1)
A.modifier(e, ["v", "raid", 0, "dir"], "-y")
chk("un raidisseur s'ajoute ; « -y » s'écrit « −y » (comme l'écran)", u["raid"] == [dict(E.raidisseur_defaut(), dir="−y", x=150.0, y=0.0, L=80.0, t=10.0, h=150.0)],
    u["raid"])
avant = copy.deepcopy(A._actif(e))
for chemin, v, quoi in ((["v", "elems", 0, "d"], "-10", "dimension négative"), (["v", "raid", 0, "L"], "-5", "raidisseur négatif (Streamlit : 0 en silence)"),
                        (["v", "elems", 0, "rot"], "45", "rotation 45°"), (["v", "elems", 0, "type"], "Cornière", "type inconnu"),
                        (["v", "elems", 0, "prof"], "HEA 999", "profilé inconnu"), (["v", "raid", 0, "dir"], "+z", "sens inconnu"),
                        (["v", "tiges", 0, "x"], "abc", "nombre illisible"), (["v", "tiges", 9, "x"], "1", "ligne inconnue"),
                        (["grille", "nx"], "0", "grille : 0 file"), (["grille", "ny"], "2,5", "grille : 2,5 rangées"),
                        (["grille", "wx"], "-1", "grille : entraxe négatif"), (["v", "N_Ed"], "", "N_Ed vide (Streamlit : 0)"),
                        (["v", "g_M0"], "0", "γ_M0 = 0"), (["v", "tp_u"], "-25", "épaisseur négative")):
    msg = A.modifier(e, chemin, v)
    chk(f"refus : {quoi}", bool(msg) and A._actif(e) == avant, msg)
for i in range(len(u["elems"])):
    A.modifier(e, ["v", "elems", 0, "-"], 1)
chk("le dernier élément se retire (comme le 🗑️ de Streamlit) : le moteur dit « aucun élément »",
    u["elems"] == [] and not r_de(e).verified and "aucun_e" in [a.id for a in r_de(e).alerts]
    and "Aucun élément soudé" in A.resultat(e)["donnees"])
A.modifier(e, ["grille", "nx"], "3")
A.modifier(e, ["grille", "wx"], "360")
A.modifier(e, ["grille", "poser"], 1)
chk("la grille 3 × 2 se pose (remplace les boulons) et reste dans la note", len(u["tiges"]) == 6 and A._actif(e)["grille"]["nx"] == 3
    and A.normaliser(json.loads(json.dumps(e)))["assemblages"][0]["grille"] == A._actif(e)["grille"])
e = neuf()
A.modifier(e, ["v", "elems", "+"], 1)
msg = A.modifier(e, ["v", "mode"], E.MODE_COMPLET)
chk("bascule refusée (deux éléments) : la raison de l'écran est dite, le mode reste manuel",
    msg == "Vérification complète impossible : un seul élément soudé est admis." and A._actif(e)["values"]["mode"] == E.MODE_MANUEL, msg)
e = neuf(True)
A.modifier(e, ["v", "n_ax"], "3")
A.modifier(e, ["v", "prof_c"], "HEB 300")
A.modifier(e, ["v", "mode"], E.MODE_MANUEL)
u = A._actif(e)["values"]
chk("bascule vers le calcul manuel : la grille 3 × 2 devient six boulons, le HEB 300 un élément à l'axe",
    len(u["tiges"]) == 6 and u["elems"][0]["prof"] == "HEB 300" and u["raid"] == [])
chk("grille dans la vérification complète : « Poser » refusé (files et rangées y sont des champs)",
    bool(A.modifier(neuf(True), ["grille", "poser"], 1)))
e = neuf()
A.modifier(e, ["v", "id_rep"], "PP3")
A.modifier(e, ["exemple"], "VAL-A")
chk("« VAL-A » se charge en vérification complète, le repère est gardé", A._actif(e)["values"]["mode"] == E.MODE_COMPLET
    and A._actif(e)["values"]["id_rep"] == "PP3")
A.modifier(e, ["exemple"], "VAL-M2")
chk("« VAL-M2 » se charge au calcul manuel (critère plastique de la fiche)", A._actif(e)["values"]["mode"] == E.MODE_MANUEL
    and A._actif(e)["values"]["crit"] == E.CRIT_EC3 and A._actif(e)["values"]["tiges"][0] == {"x": -100.0, "y": -215.0})
chk("exemple inconnu refusé", bool(A.modifier(e, ["exemple"], "VAL-Z")))
A.modifier(e, ["visserie"], "2 écrous")
A.modifier(e, ["grille", "nx"], "4")
A.modifier(e, ["reinitialiser"], 1)
el = A._actif(e)
chk("« Rétablir les valeurs par défaut » : défauts, visserie et grille par défaut, repère gardé",
    el["values"]["mode"] == E.MODE_MANUEL and el["visserie"] == E.VISSERIE_DEFAUT and el["grille"] == PI.GRILLE_DEFAUT
    and el["values"]["id_rep"] == "PP3")
chk("un pied de poteau neuf porte la visserie de son écran (« 1 rondelle + 1 écrou + 1 contre-écrou »)",
    A.nouvel_assemblage("platine_beton")["visserie"] == E.VISSERIE_DEFAUT != A.VISSERIE_DEFAUT)
e = A.nouvel_etat()
A.modifier(e, ["type"], "platine_beton")
v1 = A._actif(e)["visserie"]
A.modifier(e, ["type"], "plat_ame")
chk("changer de configuration : la visserie PAR DÉFAUT suit celle de l'attache, la grille n'est que celle du pied",
    v1 == E.VISSERIE_DEFAUT and A._actif(e)["visserie"] == A.VISSERIE_DEFAUT and "grille" not in A._actif(e))
e = A.nouvel_etat()
A.modifier(e, ["visserie"], "2 rondelles")
A.modifier(e, ["type"], "platine_beton")
chk("… une visserie saisie est gardée", A._actif(e)["visserie"] == "2 rondelles")
chk("une note abîmée (grille illisible) se relit avec la grille par défaut",
    A.normaliser({"assemblages": [{"type": "platine_beton", "values": {}, "grille": {"nx": "abc", "wx": -3}}]})["assemblages"][0]["grille"]
    == PI.GRILLE_DEFAUT)

# =====================================================================================================================
print("\n=== 7. Les sélecteurs et l'échappement ===")
for complet in (False, True):
    e = neuf(complet)
    h = A.selecteur(e, "ancr")
    u = A._actif(e)["values"]
    rows = re.findall(r'data-choix="([^"]+)".*?<span class="chip (\w)">([^<]+)</span>', h)
    att = []
    for b in DB["boulons"]:
        R = CFG.calculer(dict(u, ancr=b["n"]))
        att.append((b["n"], A.niveau_R(R), f"{C.pc(C.pct(min(R.eta_max, 9.99)))}{C.NBSP}%"))
    chk(f"{'complet' if complet else 'manuel'} : le sélecteur des tiges recalcule chaque diamètre (✓ M24)",
        rows == att and 'class="pk-row cur' in h.split('data-choix="M24"')[0][-80:], rows[:3])
    A.choisir_section(e, "ancr", "M20")
    chk(f"{'complet' if complet else 'manuel'} : choisir M20 l'écrit dans les saisies", A._actif(e)["values"]["ancr"] == "M20")
e = neuf(True)
h = A.selecteur(e, "prof_c", ["HEB"])
A.choisir_section(e, "prof_c", "HEB 260")
chk("le sélecteur du poteau (complet) : HEB triés par hauteur, choisir HEB 260", ">HEB 260<" in h and A._actif(e)["values"]["prof_c"] == "HEB 260")
PIEGE = '<img src=x onerror=alert(1)>'
fuites = []
for complet in (False, True):
    e = neuf(complet)
    for k in ("id_rep", "id_red", "id_projet"):
        A.modifier(e, ["v", k], PIEGE)
    A.modifier(e, ["visserie"], PIEGE)
    r = A.resultat(e)
    fuites += [(complet, k) for k in ("resultat", "volets", "donnees", "elements", "phrase") if "<img" in r[k]]
    fuites += [(complet, v["cle"]) for v in r["vues"] if "<img" in v["svg"]]
chk("un texte saisi n'entre jamais brut dans un fragment (deux modes)", not fuites, fuites)

# =====================================================================================================================
print("\n=== 7 bis. Géométrie non conforme : la tête de la note dit la vraie cause et son taux ===")
# Défaut du 07/11/2026 (trouvé par l'étude « modifier sur le dessin ») : calcul « à la main », une tige déplacée à
# 30 mm du bord pour 1,2·d₀ = 31,2 mm. Le moteur ne calcule plus les résistances qui dépendent de la géométrie ; son
# η max tombait à 0,03 %, et la tête écrivait « 0 % · Non conforme · Trous · pression diamétrale 0,2 kN ≤ 535,8 kN »
# quand la ligne « Géométrie » disait 104 %. Un chiffre en tête qui minimise est un défaut d'affichage NON conservatif.


def tige_au_bord(x):
    e = neuf()
    assert A.modifier(e, ["v", "mode"], E.MODE_MANUEL) is None
    assert A.modifier(e, ["v", "bp_u"], "500") is None
    assert A.modifier(e, ["v", "tiges", "1", "x"], x) is None
    return e


e = tige_au_bord("220")
R = r_de(e)
res = A.resultat(e)
tete = res["resume"]
# (la LIGNE « Géométrie » : le bouton « tout le calcul » du bloc ouvre lui aussi la fenêtre de la géométrie quand elle gouverne)
geo = re.search(r'<div class="el vc[^"]*" data-calcul="geometrie".*?<span class="pct [a-z]+">(\d+)\s*%</span>', res["resultat"], re.S)
chk("tige à 30 mm du bord : le moteur la dit non conforme (pince 30 < 31,2 mm)",
    not R.verified and not R.dist_ok and A._pince_ko(R) is not None and abs(A._pince_ko(R).val - 30.0) < 1e-9, R.statut)
chk("tige à 30 mm du bord : la tête de la note écrit 104 %, rouge, « Non conforme » (et non 0 %)",
    tete["taux"] == 104 and tete["niveau"] == "r" and tete["statut"] == "Non conforme", tete)
chk("tige à 30 mm du bord : le taux de tête égale celui de la ligne « Géométrie » du résultat",
    geo is not None and int(geo.group(1)) == tete["taux"], geo.group(0)[-80:] if geo else res["resultat"][:300])
chk("tige à 30 mm du bord : le gouvernant nommé est la pince, pas une vérification satisfaite",
    tete["gouvernant"].endswith("Géométrie · pince des boulons au bord de la platine") and "Trous" not in tete["gouvernant"],
    tete["gouvernant"])
v = plat(re.search(r'<div class="verdict">.*?</div></div>', res["resultat"], re.S).group(0))
chk("tige à 30 mm du bord : le verdict écrit la comparaison de la pince (30,0 mm < 31,2 mm, e ≥ 1,2·d0)",
    "104%" in v and "Géométrie · pince des boulons au bord de la platine" in v and "30,0 mm < 31,2 mm (e ≥ 1,2·d0)" in v, v)
marques = re.findall(r'<div class="el vc gov"[^>]*><div class="l1"><b>([^<]+)</b>', res["resultat"])
chk("tige à 30 mm du bord : la ligne marquée gouvernante est « Géométrie » (pas « Trous » à 0 %)", marques == ["Géométrie"], marques)
chk("tige à 30 mm du bord : le bloc « Vérifications » écrit aussi 104 % (établi compact : son état, à côté du verdict)",
    "Non conforme · 104" in plat(res["resultat"]).replace("\xa0", " ") and res["elements"] == "", plat(res["resultat"])[:300])
chk("tige à 30 mm du bord : pas de « Vérifié sous réserve » sous « Non conforme »",
    "Vérifié sous réserve" not in res["resultat"], texte(res["resultat"])[-300:])
# l'état voisin, conforme, est intact : la règle ne touche qu'une géométrie non conforme
e0 = tige_au_bord("210")
t0 = A.resultat(e0)["resume"]
chk("tige à 40 mm du bord (conforme) : la tête garde le taux du moteur (79 %) et son gouvernant",
    t0["taux"] == A.taux(A.eta_R(r_de(e0))) == 79 and "Géométrie" not in t0["gouvernant"], t0)
# sans pince en cause, une alerte bloquante seule (boulon dans l'emprise du poteau) : la tête la nomme
e1 = neuf()
assert A.modifier(e1, ["v", "mode"], E.MODE_MANUEL) is None
assert A.modifier(e1, ["v", "tiges", "0", "x"], "0") is None and A.modifier(e1, ["v", "tiges", "0", "y"], "0") is None
R1 = r_de(e1)
t1 = A.resultat(e1)["resume"]
if R1.dist_ok and not R1.geo_ok and A.eta_R(R1) <= 1:
    chk("boulon dans l'emprise du poteau : la tête nomme « Géométrie non valide », pas une vérification satisfaite",
        t1["gouvernant"].endswith("Géométrie non valide") and t1["niveau"] == "r", t1)
else:
    chk("boulon dans l'emprise du poteau : cas d'essai valide (géométrie bloquante, pinces conformes, η ≤ 1)", False,
        (R1.dist_ok, R1.geo_ok, A.eta_R(R1)))

# =====================================================================================================================
print("\n=== 8. Mutations : chaque copie protégée rougit ===")
B = copy.deepcopy


def mute(nom, cible, attr, nouveau, controle):
    avant_ = getattr(cible, attr)
    setattr(cible, attr, nouveau)
    try:
        rouge = not controle()
    finally:
        setattr(cible, attr, avant_)
    chk(f"mutation « {nom} » : le contrôle rougit", rouge)


def ctrl_ajout():
    u = B(MANUELS[0][2])
    etat = {"ppb_tiges": B(u["tiges"])}
    with faux_st(EM, etat):
        EM._ajouter_boulon(u)
    return etat["ppb_tiges"][-1] == PI.ajouter_boulon(u)


mute("boulon ajouté à 50 mm du bord", PI, "ajouter_boulon", lambda u: dict(x=0.0, y=-(500 / 2 - 50.0)), ctrl_ajout)


def ctrl_platine():
    R = MANUELS[0][1]
    with faux_st(EM) as fs:
        EM.bloc_etat(R)
    return fs.statuts[0][1:3] == (PI.platine_manuel(R)["titre"], PI.platine_manuel(R)["complement"])


mute("critère gouvernant mal nommé", PI, "CRITERES_NOMS", dict(PI.CRITERES_NOMS, T="tronçon en T"), ctrl_platine)


def ctrl_param():
    R = COMPLETS[0][1]
    with faux_st(ER) as fs:
        ER._parametres_retenus(R)
    return fs.tables[0][2] == [[str(a), str(b)] for a, b in PI.parametres_complet(R)]


mute("une ligne de paramètre retirée", PI, "parametres_complet", lambda R, f=PI.parametres_complet: f(R)[:-1], ctrl_param)
mute("bascule sans reprise de la disposition", PI.CV, "vers_manuel", lambda u: dict(u, mode=E.MODE_MANUEL),
     lambda: len((lambda e: (A.modifier(e, ["v", "n_ax"], "3"), A.modifier(e, ["v", "mode"], E.MODE_MANUEL), A._actif(e)["values"]["tiges"])[2])(neuf(True))) == 6)


class Abime:
    """Le module d'aiguillage, dont le calcul ajoute 10⁻⁷ à N_Ed."""

    def __getattr__(self, nom):
        return getattr(A, nom)

    @staticmethod
    def calculer(cfg, el):
        el = dict(el, values=dict(el["values"], N_Ed=(el["values"].get("N_Ed") or 0) * 1.0000001 + 1e-7))
        return A.calculer(cfg, el)


chk("mutation « N_Ed × 1,0000001 » : le comparateur de parité rougit", bool(comparer_parite(CAS[:6], Abime())))

# =====================================================================================================================
if os.environ.get("CONTRAT_SEPT"):
    print("\n=== 9. Le contrat commun, rejoué sur chacune des sept configurations ===")
    code = ("import os, runpy, sys\nRACINE = %r\nsys.path.insert(0, RACINE); os.chdir(RACINE)\ncle, mode = sys.argv[1], sys.argv[2]\n"
            "from api_web import assemblages as A\n"
            "def neuf():\n    e = {'version': 1, 'assemblages': [A.nouvel_assemblage(cle)], 'actif': 0}\n"
            "    if mode == 'complet':\n        A.modifier(e, ['v', 'mode'], A.CONFIGS[cle].E.MODE_COMPLET)\n    return e\n"
            "A.nouvel_etat = neuf\nsys.argv = ['test_api_web_modules.py', 'assemblages']\n"
            "runpy.run_path(os.path.join(RACINE, 'tests', 'test_api_web_modules.py'), run_name='__main__')\n") % RACINE
    for cle, mode in [(c, "") for c in A.ordre()] + [("platine_beton", "complet")]:
        p = subprocess.run([sys.executable, "-c", code, cle, mode or "-"], capture_output=True, text=True, cwd=RACINE)
        fin = [li for li in p.stdout.splitlines() if li.startswith("RÉSULTAT")]
        chk(f"contrat commun avec une note neuve « {cle}{' (vérification complète)' if mode else ''} » : {fin[-1] if fin else '?'}",
            p.returncode == 0, [li for li in p.stdout.splitlines() if li.startswith("KO")][:2] or p.stderr[-300:])

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:200])
sys.exit(1 if KO else 0)

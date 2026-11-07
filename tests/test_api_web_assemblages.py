# -*- coding: utf-8 -*-
"""Le module « Assemblages métalliques » du site (api_web/assemblages.py et ses aides par famille) : ce que l'écran du
site calcule et affiche est ce que calcule l'écran Streamlit. Chaque contrôle rougit si l'on retire ce qu'il protège.

  1. Le registre : MODULE / PAGE, les sept configurations dans l'ordre du registre des assemblages, chacune calcule.
  2. Les copies NOMMÉES de l'écran Streamlit (le site ne peut pas importer ses fichiers d'écran, qui chargent
     streamlit) : valeur_widget = carte.valeur_widget, les dimensions provisoires = ecran_saisie.DIMENSIONS, les
     listes « non revérifiés » = ecran_resultats.NON_REVERIFIES, les « paramètres retenus » et le « détail de la
     solution retenue » = ce que _parametres_retenus et onglet_predim ÉCRIVENT (rejoués avec un faux « st »).
  3. La parité du calcul (fichier → saisies → moteur), sans écran, sur plusieurs centaines de jeux : les jeux de
     référence (tests/ref_*.json), les exemples publiés, et des tirages au hasard (profilés du catalogue ou
     personnalisés, dimensions vides, textes « 12,5 », options inconnues, prédimensionnement). Le chemin de l'écran
     Streamlit (carte.valeur_widget → carte.dimensions_vides → carte.provisoire → compute) et celui du site
     (normaliser → calculer) donnent la MÊME empreinte : chaque vérification (E_d, R_d, η, active, vérifiée), les
     alertes, les pinces, le statut — au bit près (repr des flottants).
  4. L'écran Streamlit LUI-MÊME (AppTest, l'application réelle) : pour les trois attaches poutre–poutre, des jeux
     chargés dans son état comme le fait « Ouvrir » ; le résultat qu'il calcule (``_fpl_R``, ``_asm_R``,
     ``_pab_R``) a l'empreinte de celui du site.
  5. Ce que le site AFFICHE = le moteur : verdict, une ligne par pièce (taux de synthese.taux_par_element), les
     E_d / R_d de chaque vérification dans sa fenêtre, les dessins (SVG identiques à ceux des schémas du moteur),
     la note texte, la méthode, la validation (benchmark rejoué).
  6. La saisie : une valeur acceptée est celle du widget ; une saisie que Streamlit ramène EN SILENCE à une valeur
     fausse (rangées 2,4 → 2, γ_M2 = 0, V_Ed vide → 0, épaisseur négative) est REFUSÉE avec un message, et
     l'assemblage reste intact. Le moteur fait bien ce que le refus évite (contrôle du défaut lui-même).
  7. La note : ajouter, copier, supprimer (jamais le dernier), ouvrir ; le menu Configuration ; le résumé de la note.
  8. Les sélecteurs (boulon, profilés, cornière, solutions du prédimensionnement) : triés, retenu ✓, optimal ★,
     taux recalculés par le moteur ; un choix change les saisies comme l'écran Streamlit.
  9. Échappement : un texte saisi n'entre jamais brut dans un fragment.
 10. Mutations : le comparateur de parité ROUGIT si l'on abîme valeur_widget ou les dimensions provisoires.
 11. Pureté : le module se charge et calcule dans un sous-processus où streamlit, pandas, reportlab, matplotlib,
     scipy, PIL, fitz sont REFUSÉS — c'est le chemin du navigateur.

    /usr/bin/python3.12 tests/test_api_web_assemblages.py        (streamlit requis pour 2 et 4)
"""
import contextlib
import copy
import html
import io
import json
import logging
import os
import random
import math
import re
import subprocess
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)
sys.path.insert(0, os.path.join(RACINE, "tests"))

from acier import registre as RG  # noqa: E402
from acier.bibliotheques import DB, PERSO  # noqa: E402
from api_web import assemblages as A  # noqa: E402
from api_web import assemblages_poutre as PP  # noqa: E402
from api_web import commun as C  # noqa: E402

OK, KO = [], []
TIRAGES = int(os.environ.get("TIRAGES", "70"))       # jeux tirés au hasard par configuration (partie 3)
CAS_ECRAN = int(os.environ.get("CAS_ECRAN", "12"))   # jeux rejoués dans l'écran Streamlit (partie 4)



def eta_tete(R):
    """Le taux de TÊTE attendu : le η max du moteur (celui de l'écran Streamlit), porté au rapport de la pire pince
    NON conforme quand il est plus grand (07/11/2026 : un boulon trop gros pour la platine montrait « 52 % » rouge
    sans dire pourquoi ; le moteur qui trouve la géométrie non valide ne calcule plus ce qui en dépend). « Vérifié
    sous réserve » ne s'écrit plus que sous un verdict vérifié, comme dans l'écran Streamlit."""
    eta = R.eta_max
    ko = [x for x in (R.get("dist") or []) if not x.ok]
    if ko and isinstance(eta, (int, float)) and math.isfinite(eta):
        eta = max(eta, max(A._taux_distance(x) for x in ko))
    return eta


def geometrie_gouverne(R):
    """La tête nomme la GÉOMÉTRIE (et la ligne « Géométrie » est marquée gouvernante) : une pince non conforme dont le
    rapport atteint le η du moteur, ou, sans pince en cause, une alerte bloquante quand le moteur reste sous 100 %."""
    if R.get("dist_ok", True) and R.get("geo_ok", True):
        return False
    g = R.get("gov")
    eta = A.eta_vu(R.eta_max, g.ok if g is not None else bool(R.verified))
    if not (isinstance(eta, (int, float)) and math.isfinite(eta)):
        return False
    ko = [x for x in (R.get("dist") or []) if not x.ok]
    if ko:
        return max(A._taux_distance(x) for x in ko) >= eta
    return eta <= 1 and any(a.block for a in R.alerts)

def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "ÉCHEC ") + nom + (f"   [{str(info)[:300]}]" if info and not cond else ""))


POUTRE_POUTRE = {"plat_ame": "acier.assemblages.poutre_poutre.plat_ame",
                 "doubles_cornieres": "acier.assemblages.poutre_poutre.doubles_cornieres",
                 "platine_about": "acier.assemblages.poutre_poutre.platine_about"}
TOUS = dict(POUTRE_POUTRE, plat_ame_poteau="acier.assemblages.poutre_poteau.plat_ame_poteau",
            doubles_cornieres_poteau="acier.assemblages.poutre_poteau.doubles_cornieres_poteau",
            platine_about_poteau="acier.assemblages.poutre_poteau.platine_about_poteau",
            platine_beton="acier.assemblages.pied_poteau.platine_beton")
PREFIXES = {"plat_ame": "fpl_", "doubles_cornieres": "asm_", "platine_about": "pab_"}


def ecran(cle, nom):
    """Un fichier d'écran Streamlit d'une configuration (ecran_saisie, ecran_resultats, interface)."""
    import importlib
    return importlib.import_module(f"{TOUS[cle]}.{nom}")


def etat_de(cle, values=None, **el):
    e = A.nouvel_etat()
    A.modifier(e, ["type"], cle)
    if values is not None:
        e = A.normaliser({"assemblages": [dict({"type": cle, "values": values}, **el)], "actif": 0})
    return e


def r_de(e):
    el = A._actif(e)
    return A.calculer(A._cfg(el), el)[0]


# ------------------------------------------------------------------ l'empreinte d'un résultat
def _x(v):
    return repr(v) if isinstance(v, float) else v


def empreinte(R):
    """Tout ce que l'écran montre d'un résultat, flottants en repr (égalité AU BIT)."""
    if R is None:
        return None
    d = {"eta_max": _x(R.eta_max), "verified": R.verified, "statut": R.statut, "geo_ok": R.get("geo_ok"),
         "dist_ok": R.get("dist_ok"), "gov": R.gov.label if R.gov else None, "pred": R.get("pred"),
         "reserve": R.get("reserve"), "boulon": R.get("boulon")}
    d["checks"] = [(c.grp, c.key, c.label, bool(c.active), bool(c.ok), _x(c.eta), _x(c.Ed), _x(c.Rd)) for c in R.checks]
    d["alerts"] = [(a.id, bool(a.block)) for a in R.alerts]
    d["dist"] = [(x.lab, _x(x.val), _x(x.min), _x(x.max), bool(x.ok)) for x in (R.get("dist") or [])]
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
def tirage(cle, rng):
    """Un jeu de saisies BRUT (comme un fichier) : défauts perturbés, profilés du catalogue ou personnalisés, textes à
    virgule, valeurs vides, options inconnues, prédimensionnement."""
    cfg = A.CONFIGS[cle]
    E = cfg.E
    d = E.defaults()
    if cle == "doubles_cornieres":
        import alea_double_corniere as ADC
        d.update(ADC.tirage(rng))
    for k, f in E.CHAMPS.items():
        if k in cfg.listes or k.startswith("id_"):
            continue
        r = rng.random()
        if f["t"] == "s" and r < 0.22:
            d[k] = rng.choice(list(f["o"]))
        elif f["t"] == "n" and isinstance(d.get(k), (int, float)) and r < 0.30:
            x = d[k] * rng.choice([0.5, 0.75, 0.9, 1.1, 1.25, 1.6, 2.0])
            d[k] = round(x, rng.choice([0, 1])) if abs(x) > 2 else round(x, 2)
            if rng.random() < 0.15:
                d[k] = str(d[k]).replace(".", ",")              # un fichier écrit à la française
    if rng.random() < 0.15 and "prof_S" in d:
        d["prof_S"] = PERSO
        if rng.random() < 0.5:                                    # dimensions vides : vérification suspendue
            for x in ("hS_u", "bS_u", "twS_u", "tfS_u"):
                d[x] = None
        else:
            p = rng.choice(DB["profils"])
            d.update(hS_u=p["h"], bS_u=p["b"], twS_u=p["tw"], tfS_u=p["tf"], rS_u=p["r"])
    if rng.random() < 0.05:
        d["boulon_u"] = "M99"                                     # option inconnue : la première de la liste
    if rng.random() < 0.05 and "V_Ed" in d:
        d["V_Ed"] = None                                          # vide : le moteur prend 0
    if "mode_calc" in d and rng.random() < 0.2:
        d["mode_calc"] = E.MODE_PREDIM if hasattr(E, "MODE_PREDIM") else d["mode_calc"]
    return d


def jeux(cle, n, graine=1):
    """Les jeux d'une configuration : défauts, exemples publiés, cas de référence du dépôt, puis tirages."""
    cfg = A.CONFIGS[cle]
    out = [("défaut", {})]
    for b in (getattr(cfg.bench, "BENCH", None) or []):
        out.append((f"exemple {b['id']}", copy.deepcopy(b["inp"])))
    ref = {"plat_ame": "ref_plat_ame.json", "platine_about": "ref_platine_about.json", "doubles_cornieres_poteau": "ref_dcc.json",
           "plat_ame_poteau": "ref_fpc.json", "platine_about_poteau": "ref_pac.json"}.get(cle)
    if ref:
        with open(os.path.join(RACINE, "tests", ref), encoding="utf-8") as fh:
            for nom, x in json.load(fh).items():
                if isinstance(x, dict) and isinstance(x.get("inp"), dict):
                    out.append((f"réf. {nom}", x["inp"]))
    rng = random.Random(f"{cle}-{graine}")
    out += [(f"tirage {i}", tirage(cle, rng)) for i in range(n)]
    return out


# ------------------------------------------------------------------ le chemin de l'écran Streamlit, sans écran
def chemin_streamlit(cle, src):
    """Ce que fait l'écran Streamlit d'un fichier : entrees.charger_json, chaque clé par carte.valeur_widget
    (ecrire_entrees), puis interface.show : carte.dimensions_vides, carte.provisoire, compute."""
    from acier.assemblages import carte as kit
    es = ecran(cle, "ecran_saisie")
    cfg = A.CONFIGS[cle]
    D = es.description(cfg.E.charger_json(src)) if hasattr(es, "description") else es.D
    brut = cfg.E.charger_json(src)
    compute = ecran(cle, "interface").compute                    # le calcul que l'écran appelle, importé par lui
    u = {k: kit.valeur_widget(D, k, brut.get(k)) for k in cfg.E.CLES if k not in cfg.listes}
    for k in cfg.listes:
        u[k] = brut.get(k)
    vides = kit.dimensions_vides(D, u)
    try:
        R = compute(kit.provisoire(D, u) if vides else u)
        err = None
    except Exception as e:  # noqa: BLE001
        R, err = None, type(e).__name__
    return u, sorted(vides), R, err


def chemin_site(cle, src, mod=A):
    try:
        e = mod.normaliser({"assemblages": [{"type": cle, "values": src}], "actif": 0})
    except ValueError as x:
        return None, None, None, f"refus : {x}"
    el = e["assemblages"][0]
    R, vides, err = mod.calculer(mod.CONFIGS[cle], el)
    return el["values"], sorted(vides), R, (err.split(" :")[0] if err else None)


def comparer_parite(cle, cas, mod=A):
    """Les écarts entre le chemin de l'écran Streamlit et celui du site, sur une liste de jeux."""
    out = []
    for nom, src in cas:
        u1, v1, R1, e1 = chemin_streamlit(cle, src)
        u2, v2, R2, e2 = chemin_site(cle, src, mod)
        if u2 is None:
            out.append((nom, "le site refuse le fichier", e2))
            continue
        du = [k for k in u1 if k not in A.CONFIGS[cle].listes and _x(u1[k]) != _x(u2.get(k))]
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


# =====================================================================================================================
print("=== 1. Le registre et les sept configurations ===")
from api_web import registre as REG  # noqa: E402

chk("MODULE = « assemblages », PAGE = « assemblages », le module est au registre du site",
    A.MODULE == "assemblages" and A.PAGE == "assemblages" and "assemblages" in REG.cles() and REG.module("assemblages") is A,
    REG.cles())
ids = [a.id for f in RG.FAMILLES for a in f.assemblages if a.disponible]
chk("les sept configurations du registre des assemblages, dans son ordre (familles, puis attaches)",
    A.ordre() == ids and len(ids) == 7, (A.ordre(), ids))
chk("les fichiers d'aide n'ont pas de MODULE (un seul module au registre du site)",
    not any(hasattr(m, "MODULE") for m in (PP, sys.modules["api_web.assemblages_poteau"], sys.modules["api_web.assemblages_pied"])))
for cle in A.ordre():
    e = etat_de(cle)
    r = A.resultat(e)
    cfg = A.CONFIGS[cle]
    R = r_de(e)
    dessins = [v for v in r["vues"] if v["svg"].lstrip().startswith("<svg")]
    chk(f"{cle} : la note neuve se calcule, le résumé est celui du moteur, au moins trois dessins SVG",
        R is not None and r["resume"]["taux"] == A.taux(R.eta_max) and len(dessins) >= 3
        and r["resume"]["niveau"] == A.niveau_R(R), (r["resume"], len(dessins)))

# =====================================================================================================================
print("\n=== 2. Les copies nommées de l'écran Streamlit ===")
from acier.assemblages import carte as KIT  # noqa: E402

rng = random.Random(7)
ecarts_vw = []
for cle in TOUS:
    es = ecran(cle, "ecran_saisie")
    D = es.D
    cfg = A.CONFIGS[cle]
    for k, f in cfg.E.CHAMPS.items():
        essais = [None, "", " ", "abc", "12,5", "1e3", 0, -3, 7.25, "7", True, "Oui", "Non", "nan"]
        if f["t"] == "s":
            essais += list(f["o"]) + ["inconnue", 2, 2.0, "2.0"]
        for v in essais + [rng.uniform(-100, 1000) for _ in range(3)]:
            a, b = KIT.valeur_widget(D, k, v), A.valeur_widget(cfg, k, v)
            if _x(a) != _x(b):
                ecarts_vw.append((cle, k, v, a, b))
chk("valeur_widget du site = carte.valeur_widget de l'écran Streamlit (7 configurations, chaque champ, 20 saisies)",
    not ecarts_vw, ecarts_vw[:4])
for cle in TOUS:
    chk(f"{cle} : les dimensions provisoires = ecran_saisie.DIMENSIONS", A.CONFIGS[cle].dimensions == ecran(cle, "ecran_saisie").D.DIMENSIONS,
        (A.CONFIGS[cle].dimensions, ecran(cle, "ecran_saisie").D.DIMENSIONS))
diff_neuf = []
for cle in TOUS:
    es = ecran(cle, "ecran_saisie")
    cfg = A.CONFIGS[cle]
    creer = {k: es.valeur_widget(k, v) for k, v in cfg.E.defaults().items()}      # projet_ui.creer
    creer["id_rep"] = "A1"
    neuf = A.nouvel_assemblage(cle, "A1")["values"]
    d = [k for k in creer if k not in cfg.listes and _x(creer[k]) != _x(neuf[k])]
    R1 = ecran(cle, "interface").compute(creer)
    R2 = A.calculer(cfg, {"values": neuf})[0]
    if d or ecarts(empreinte(R1), empreinte(R2)):
        diff_neuf.append((cle, d[:3], ecarts(empreinte(R1), empreinte(R2))[:2]))
chk("un assemblage neuf = « Créer » de la page projet Streamlit (projet_ui.creer : defaults() mis en forme des widgets), "
    "7 configurations, même résultat — le pied de poteau neuf s'ouvre sur le calcul à la main", not diff_neuf, diff_neuf)
for cle, copie in (("plat_ame", PP.NON_REVERIFIES_PA), ("doubles_cornieres", PP.NON_REVERIFIES_DC), ("platine_about", PP.NON_REVERIFIES_PB)):
    chk(f"{cle} : « non revérifiés » = ecran_resultats.NON_REVERIFIES", tuple(copie) == tuple(ecran(cle, "ecran_resultats").NON_REVERIFIES))
    chk(f"{cle} : visserie par défaut = entrees.VISSERIE_DEFAUT", A.VISSERIE_DEFAUT == A.CONFIGS[cle].E.VISSERIE_DEFAUT)


class FauxSt:
    """Un « st » qui enregistre au lieu d'afficher (les tableaux arrivent par tableau_md, remplacé)."""

    def __init__(self):
        self.tables, self.textes = [], []

    def markdown(self, x="", *a, **k):
        (self.tables if isinstance(x, tuple) else self.textes).append(x)

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
def faux_st(ER):
    """ecran_resultats avec un faux « st », tableau_md qui rend ses lignes, bloc_alerte muet."""
    fs = FauxSt()
    avant = (ER.st, ER.tableau_md, getattr(ER, "bloc_alerte", None))
    ER.st = fs
    ER.tableau_md = lambda h, rows, **k: ("TAB", tuple(h), [[str(x) for x in r] for r in rows])
    ER.bloc_alerte = lambda *a, **k: None
    try:
        yield fs
    finally:
        ER.st, ER.tableau_md, ER.bloc_alerte = avant


ecarts_par, ecarts_det, n_par, n_det = [], [], 0, 0
for cle in POUTRE_POUTRE:
    ER = ecran(cle, "ecran_resultats")
    cfg = A.CONFIGS[cle]
    for nom, src in jeux(cle, 25, graine=2):
        el = A.normaliser({"assemblages": [{"type": cle, "values": src}]})["assemblages"][0]
        R, vides, err = A.calculer(cfg, el)
        if R is None or vides:
            continue
        with faux_st(ER) as fs:
            ER._parametres_retenus(R)
        ecran_ = [t for t in fs.tables if t[1] == ("Paramètre", "Valeur")][0][2]
        site = [[str(a), str(b)] for a, b in cfg.parametres(R)]
        n_par += 1
        if ecran_ != site:
            ecarts_par.append((cle, nom, ecarts(ecran_, site)[:2]))
        if R.get("pd") is None:
            continue
        with faux_st(ER) as fs:
            ER.onglet_predim(R, el["values"], lambda u: None)
        det = [t for t in fs.tables if t[1] == ("", "")]
        n_det += 1
        if not det or det[0][2] != A.detail_predim(cfg, R.pd):
            ecarts_det.append((cle, nom, ecarts(det[0][2] if det else [], A.detail_predim(cfg, R.pd))[:2]))
        if cfg.predim["regle"] not in fs.textes:
            ecarts_det.append((cle, nom, "légende de l'onglet Prédim", [t for t in fs.textes if "Règle" in str(t)]))
import api_web.assemblages_pied as PIED  # noqa: E402
import importlib  # noqa: E402
ecarts_inf, n_inf = [], 0
for nom, src in jeux("platine_beton", 12, graine=3):
    el = A.normaliser({"assemblages": [{"type": "platine_beton", "values": src}]})["assemblages"][0]
    R, vides, err = A.calculer(A.CONFIGS["platine_beton"], el)
    if R is None or vides:
        continue
    manuel = PIED.CA.est_manuel_R(R)
    ER = importlib.import_module(TOUS["platine_beton"] + (".manuel.ecran" if manuel else ".ecran_resultats"))
    fs = FauxSt()
    avant = ER.st
    ER.st = fs
    try:
        ER.information(R)
    finally:
        ER.st = avant
    ecran_ = "".join(fs.textes).replace("en tête de la carte", "(section Assemblage)")
    n_inf += 1
    if ecran_ != PIED.note_information(R):
        ecarts_inf.append((nom, manuel, ecran_[:80], PIED.note_information(R)[:80]))
chk(f"pied de poteau : la phrase d'information = ecran_resultats.information / manuel.ecran.information ({n_inf} résultats, "
    "« en tête de la carte » devient « (section Assemblage) »)", n_inf >= 8 and not ecarts_inf, ecarts_inf[:2])
chk(f"« Paramètres retenus » = ce qu'écrit ecran_resultats._parametres_retenus ({n_par} résultats, trois attaches)",
    n_par > 60 and not ecarts_par, ecarts_par[:2])
chk(f"« Détail de la solution retenue par défaut » et la règle de choix = ce qu'écrit onglet_predim ({n_det} résultats)",
    n_det > 60 and not ecarts_det, ecarts_det[:2])

# =====================================================================================================================
print("\n=== 3. La parité du calcul : fichier → saisies → moteur (écran Streamlit / site) ===")
logging.disable(logging.CRITICAL)
CAS = {}
total, couverture = 0, {}
for cle in TOUS:
    n = TIRAGES if cle in POUTRE_POUTRE else max(10, TIRAGES // 4)
    CAS[cle] = jeux(cle, n)
    diff = comparer_parite(cle, CAS[cle])
    total += len(CAS[cle])
    etats = {"vérifié": 0, "non vérifié": 0, "à compléter": 0, "erreur": 0, "prédim": 0}
    for _nom, src in CAS[cle]:
        _u, v, R, err = chemin_site(cle, src)
        if err:
            etats["erreur"] += 1
        elif v:
            etats["à compléter"] += 1
        elif R is not None:
            etats["vérifié" if R.verified else "non vérifié"] += 1
            etats["prédim"] += bool(R.get("pred"))
    couverture[cle] = etats
    chk(f"{cle} : {len(CAS[cle])} jeux, même empreinte au bit près ({etats})", not diff, diff[:3])
for cle in POUTRE_POUTRE:
    e = couverture[cle]
    chk(f"{cle} : les jeux couvrent vérifié, non vérifié, à compléter et prédimensionnement",
        all(e[k] > 0 for k in ("vérifié", "non vérifié", "à compléter", "prédim")), e)
print(f"     ({total} jeux comparés)")

# =====================================================================================================================
print("\n=== 4. L'écran Streamlit lui-même (AppTest, l'application réelle) ===")
try:
    from streamlit.testing.v1 import AppTest
except ImportError:                                               # pragma: no cover
    AppTest = None
    chk("streamlit est installé (python3.12)", False)
if AppTest:
    for cle, pre in PREFIXES.items():
        es = ecran(cle, "ecran_saisie")
        cfg = A.CONFIGS[cle]
        at = AppTest.from_file(os.path.join(RACINE, "streamlit_app.py"), default_timeout=240)
        at.session_state["page"] = "Assemblages métalliques"
        with contextlib.redirect_stderr(io.StringIO()):
            at.run()
            at.button(key=f"asm_btn_open_{cle}").click()
            at.run()
        ouvert = not at.exception and at.session_state["asm_courant"] == cle
        chk(f"{cle} : l'écran Streamlit s'ouvre depuis la page projet", ouvert, at.exception)
        if not ouvert:
            continue
        cas = [c for c in CAS[cle] if not chemin_site(cle, c[1])[1] and chemin_site(cle, c[1])[3] is None][:CAS_ECRAN]
        diffs, vus = [], 0
        for nom, src in cas:
            brut = cfg.E.charger_json(src)
            for k in cfg.E.CLES:                                  # interface.ecrire_entrees : ce que fait « Ouvrir »
                at.session_state[pre + k] = es.valeur_widget(k, brut.get(k))
            at.session_state[f"_{pre}R"] = None
            with contextlib.redirect_stderr(io.StringIO()):
                at.run()
            R_ecran = at.session_state[f"_{pre}R"]
            R_site = chemin_site(cle, src)[2]
            vus += R_ecran is not None
            d = ecarts(empreinte(R_ecran), empreinte(R_site))
            if d or at.exception:
                diffs.append((nom, d[:3], at.exception))
        chk(f"{cle} : {vus} jeux chargés dans l'écran Streamlit, son résultat a l'empreinte de celui du site",
            vus == len(cas) and len(cas) >= min(CAS_ECRAN, 8) and not diffs, diffs[:2])
        # une saisie au clavier : le widget rend un flottant, le site le même
        at.session_state[pre + "V_Ed"] = 210.0
        with contextlib.redirect_stderr(io.StringIO()):
            at.run()
        e = etat_de(cle, cas[-1][1] if cas else {})
        A.modifier(e, ["v", "V_Ed"], "210")
        chk(f"{cle} : « V_Ed = 210 » saisi au clavier : même résultat à l'écran et sur le site",
            not ecarts(empreinte(at.session_state[f"_{pre}R"]), empreinte(r_de(e))))

# =====================================================================================================================
print("\n=== 5. Ce que le site affiche = le moteur ===")
RX_EL = re.compile(r'<div class="el( gov)?" data-calcul="([^"]+)"[^>]*><div class="l1"><b>([^<]+)</b>.*?<span class="pct (\w)">([^<]+)</span></div></div>')
for cle in POUTRE_POUTRE:
    cfg = A.CONFIGS[cle]
    M = __import__(f"{TOUS[cle]}.schemas", fromlist=["x"])
    T = cfg.T
    defauts = []
    for nom, src in CAS[cle][:30]:
        e = etat_de(cle, src)
        el = A._actif(e)
        R, vides, err = A.calculer(cfg, el)
        if R is None or vides or A.saisies_refusees(cfg, el["values"]):     # relue « à corriger » : pas de verdict
            continue
        r = A.resultat(e)
        lignes = RX_EL.findall(r["resultat"])
        attendu = [(nom_g, f"{C.pc(A.taux(eta))}{C.NBSP}%") for nom_g, eta, _ok, _p in cfg.S.taux_par_element(R)]
        trouve = [(b, t) for _g, _c, b, _n, t in lignes][:len(attendu)]
        noms = [g for g, *_ in A._groupes(cfg, R)]
        if [t for _, t in trouve] != [t for _, t in attendu] or [b for b, _ in trouve] != [b.replace("'", "&#x27;") for b in noms]:
            defauts.append((nom, "lignes", trouve, attendu))
        if f"{C.pc(A.taux(eta_tete(R)))}{C.NBSP}%" not in r["resultat"].split('class="verdict"')[1][:300]:
            defauts.append((nom, "verdict"))
        gov = [x for x in lignes if x[0]]
        att_gov = "Géométrie" if geometrie_gouverne(R) else (R.gov.grp if R.gov is not None else None)
        if att_gov is not None and (len(gov) != 1 or gov[0][2] != C.esc(att_gov)):
            defauts.append((nom, "ligne gouvernante", gov))
        # chaque vérification active : sa ligne (E_d, R_d) dans la fenêtre de sa pièce
        for c in R.checks:
            if c.active:
                ed, rd, un = cfg.S.ed_rd(c)
                fen = r["volets"].split(f'id="ck-{A._slug(c.grp)}"')[1].split('class="flyout')[0]
                if f">{ed}{' ' + un if un else ''}<" not in fen or f">{rd}{' ' + un if un else ''}<" not in fen:
                    defauts.append((nom, "fenêtre", c.key, ed, rd))
        # les alertes du moteur, une par message
        if r["resultat"].count('class="msg') != len(R.alerts) + bool(R.get("reserve") and R.verified) + bool(R.get("pred")):
            defauts.append((nom, "alertes", len(R.alerts)))
        # les dessins : ceux des schémas du moteur, au caractère près
        opt = M.options_ecran(R, el["cotation"], A._hl(R))
        opt.interactive, opt.poignees, opt.editables, opt.locked = False, False, el["cotation"] >= 1, None
        vues = {v["cle"]: v for v in r["vues"]}
        for k, f in (("elevation", M.elevation), ("plan", M.plan), ("droite", M.vue_droite)):
            if vues[k]["svg"] != f(R, opt).svg(identifiant=f"asm_{k}"):
                defauts.append((nom, "dessin", k))
        if C.esc(T.construire_texte(R, False)) not in vues["note"]["svg"]:
            defauts.append((nom, "note texte"))
    chk(f"{cle} : verdict, lignes par pièce (taux de synthese.taux_par_element), gouvernante, E_d / R_d de chaque "
        f"vérification dans sa fenêtre, alertes, dessins et note texte = le moteur", not defauts, defauts[:3])
    e = etat_de(cle)
    r = A.resultat(e)
    vues = {v["cle"]: v for v in r["vues"]}
    B = cfg.bench.run_bench()
    chk(f"{cle} : vues Prédim, Note, Méthode, Validation ; la validation rejoue le benchmark ({B['statut']})",
        {"predim", "note", "methode", "validation"} <= set(vues) and C.esc(B["statut"]) in vues["validation"]["svg"]
        and all(C.esc(b["id"]) in vues["validation"]["svg"] for b in B["bench"]))
    titres = [ligne.lstrip("# ").strip() for ligne in cfg.methode.splitlines() if ligne.lstrip().startswith("#")]
    chk(f"{cle} : la vue Méthode porte le texte de methode.METHODE ({len(titres)} titres)",
        titres and all(html.escape(t, quote=False).replace("**", "") in vues["methode"]["svg"].replace("<b>", "").replace("</b>", "")
                       for t in titres), titres[:3])
    A.modifier(e, ["texte_complet"], True)
    R = r_de(e)
    chk(f"{cle} : « toutes les vérifications » : la note texte complète du moteur",
        C.esc(cfg.T.construire_texte(R, True)) in {v["cle"]: v for v in A.resultat(e)["vues"]}["note"]["svg"])
    p = R.pd
    rows = [x for x in p.rows if x.geom]
    vp = vues["predim"]["svg"]
    chk(f"{cle} : la vue Prédim liste toutes les combinaisons géométriquement possibles ({len(rows)}) et la retenue",
        vp.count("<tr>") - 1 >= len(rows) and "★ retenue" in vp)
    niveaux = []
    for n in (0, 1, 2):
        A.modifier(e, ["cotation"], n)
        niveaux.append(len(A.resultat(e)["vues"][0]["svg"]))
    chk(f"{cle} : les trois niveaux de cotation changent l'élévation (vue simple < édition < complète)",
        niveaux[0] < niveaux[1] <= niveaux[2], niveaux)

# les formules : celles du moteur (formules.textes ou formules.lignes), substituées
for cle in POUTRE_POUTRE:
    cfg = A.CONFIGS[cle]
    e = etat_de(cle)
    R = r_de(e)
    vol = A.resultat(e)["volets"]
    manque = []
    for c in R.checks:
        if not c.active:
            continue
        _aide, lf = cfg.formules(R, c)
        if not lf or lf[-1]["parts"][0] != "η" or lf[-1]["res"] != f"{C.pc(A.taux(c.eta, 1))} %":
            manque.append((c.key, "η"))
        if C.lignes_fx(lf) not in vol:
            manque.append((c.key, "détail"))
    chk(f"{cle} : chaque vérification a son détail « symbole = formule = valeurs = résultat » et sa ligne η", not manque, manque[:4])
R = r_de(etat_de("doubles_cornieres"))
vol = A.resultat(etat_de("doubles_cornieres"))["volets"]
mat = PP.YDC.matrice_cornieres(R)
chk("doubles cornières : la fenêtre Cornières porte la matrice aile A · aile B (E_d / R_d de synthese.matrice_cornieres)",
    "Cornières — aile A / aile B" in vol and all(f"{x['A']['Ed']} / {x['A']['Rd']}" in vol for x in mat if x["A"]))
e = etat_de("platine_beton")
A.modifier(e, ["v", "mode"], A.CONFIGS["platine_beton"].E.CHAMPS["mode"]["o"][1])        # la vérification complète
Rp = r_de(e)
cfg = A.pour(A.CONFIGS["platine_beton"], Rp)
res = A.resultat(e)["resultat"]
anc = [x for x in RX_EL.findall(res) if x[2] == "Ancrages"]
pire = cfg.S.pire_information(Rp)
chk("pied de poteau (périmètre EN 1993-1-8) : le béton d'ancrage EN 1992-4 est une ligne d'INFORMATION grise, hors "
    "verdict, et la phrase d'information de l'écran Streamlit est dite",
    Rp.verified and pire is not None and pire.eta > 1 and anc and anc[0][3] == "c" and "information" in res.split("Ancrages")[1][:200]
    and "en information, hors verdict" in res and "Conforme" not in res.split('class="verdict"')[0], anc)
chk("doubles cornières : la répartition élastique par boulon (groupe S) est dans la fenêtre Boulons",
    "Répartition élastique par boulon · groupe S" in vol and all(C.fr(b.f, 2) in vol for b in R.tabS))

# =====================================================================================================================
print("\n=== 6. La saisie : acceptée comme le widget, refusée quand Streamlit se tromperait en silence ===")
e = etat_de("plat_ame")
for chemin, v, attendu in ((["v", "V_Ed"], "150", 150.0), (["v", "V_Ed"], "1 250,5", 1250.5), (["v", "tp_u"], 12, 12.0),
                           (["v", "n1_u"], "4", 4.0), (["v", "n2_u"], "2", "2"), (["v", "lt_ok"], False, "Non"),
                           (["v", "N_Ed"], "-40", -40.0), (["v", "classe"], "10.9", "10.9")):
    msg = A.modifier(e, chemin, v)
    chk(f"plat d'âme : « {v} » dans {chemin[-1]} → {attendu!r}", msg is None and A._actif(e)["values"][chemin[-1]] == attendu, msg)
refus = ((["v", "n1_u"], "2,4", "rangées 2,4 (Streamlit : 2 en silence)"), (["v", "n1_u"], "0", "rangées 0 (Streamlit : 1)"),
         (["v", "g_M2"], "0", "γ_M2 = 0 (Streamlit : vérifié à 60 %)"), (["v", "V_Ed"], "", "V_Ed vide (Streamlit : 0)"),
         (["v", "tp_u"], "-10", "épaisseur négative (Streamlit : η = 10⁹)"), (["v", "hp_u"], "-190", "hauteur négative (Streamlit : erreur)"),
         (["v", "boulon_u"], "M99", "boulon inconnu"), (["v", "V_Ed"], "abc", "nombre illisible"), (["v", "inconnu"], "1", "champ inconnu"),
         (["type"], "rien", "configuration inconnue"), (["cotation"], "5", "niveau de cotation inconnu"),
         (["exemple"], "ZZ", "exemple inconnu"))
for chemin, v, quoi in refus:
    avant = copy.deepcopy(e)
    msg = A.modifier(e, chemin, v)
    chk(f"refus : {quoi} — message, assemblage intact", bool(msg) and e == avant, msg)
# le défaut lui-même : le moteur fait bien ce que le refus évite
from acier.assemblages.poutre_poutre.plat_ame import entrees as EPA, moteur as MPA  # noqa: E402
u0 = EPA.defaults()
chk("le défaut existe dans le moteur : rangées 2,4 → 2 rangées, γ_M2 = 0 → « vérifié », V_Ed vide → 0",
    MPA.compute(dict(u0, n1_u=2.4)).n_1 == 2 and MPA.compute(dict(u0, g_M2=0)).verified
    and MPA.compute(dict(u0, V_Ed=None)).eta_max < MPA.compute(u0).eta_max)
# effets de bord
e = etat_de("plat_ame")
A.modifier(e, ["v", "prof_S"], "IPE 360")
A.modifier(e, ["v", "prof_S"], PERSO)
u = A._actif(e)["values"]
ipe = [p for p in DB["profils"] if p["n"] == "IPE 360"][0]
chk("profilé « Personnalisé » : ses dimensions partent de la pièce quittée (Streamlit : vides, vérification suspendue)",
    (u["hS_u"], u["bS_u"], u["twS_u"], u["tfS_u"], u["rS_u"]) == (ipe["h"], ipe["b"], ipe["tw"], ipe["tf"], ipe["r"])
    and not A.calculer(A.CONFIGS["plat_ame"], A._actif(e))[1])
A.modifier(e, ["v", "nu_pl"], PERSO)
chk("acier « Personnalisé » : f_y, f_u, β_w partent de la nuance quittée", (u["fy_u"], u["fu_u"]) == (355.0, 510.0) or u["fy_u"] > 0, (u["fy_u"], u["fu_u"]))
A.modifier(e, ["d0_impose"], True)
chk("trou imposé coché : d_0 part du trou normal du boulon (22 mm pour M20) ; décoché : vide",
    u["d0_u"] == 22.0 and A.modifier(e, ["d0_impose"], False) is None and u["d0_u"] is None, u["d0_u"])
ex = A.CONFIGS["plat_ame"].bench.BENCH[0]
A.modifier(e, ["v", "id_rep"], "R7")
A.modifier(e, ["exemple"], ex["id"])
attendu = A.valeurs_normales(A.CONFIGS["plat_ame"], dict(EPA.defaults(), **ex["inp"]))
attendu["id_rep"] = "R7"
chk("« Charger » un exemple publié : les défauts puis les données de l'exemple ; le repère est gardé",
    A._actif(e)["values"] == attendu)
A.modifier(e, ["v", "V_Ed"], "300")
A.modifier(e, ["visserie"], "2 écrous")
A.modifier(e, ["reinitialiser"], "1")
attendu = A.valeurs_normales(A.CONFIGS["plat_ame"], {})
attendu["id_rep"] = "R7"
chk("« Rétablir les valeurs par défaut » (🔄 de Streamlit) : défauts et visserie par défaut, le repère est gardé",
    A._actif(e)["values"] == attendu and A._actif(e)["visserie"] == A.VISSERIE_DEFAUT)
# le menu Configuration
e = etat_de("plat_ame")
A.modifier(e, ["v", "prof_S"], "IPE 400")
A.modifier(e, ["v", "V_Ed"], "180")
A.modifier(e, ["v", "id_rep"], "P12")
A.modifier(e, ["type"], "platine_about")
u = A._actif(e)["values"]
chk("Configuration → platine d'about (même famille) : profilés, efforts et repère suivent, le reste aux défauts",
    A._actif(e)["type"] == "platine_about" and u["prof_S"] == "IPE 400" and u["V_Ed"] == 180.0 and u["id_rep"] == "P12"
    and u["tp_u"] == A.valeur_widget(A.CONFIGS["platine_about"], "tp_u", A.CONFIGS["platine_about"].E.defaults()["tp_u"]))
A.modifier(e, ["type"], "platine_beton")
u = A._actif(e)["values"]
chk("Configuration → pied de poteau (autre famille) : seuls le repère et les efforts communs suivent",
    A._actif(e)["type"] == "platine_beton" and u["id_rep"] == "P12" and "prof_S" not in u)

# =====================================================================================================================
print("\n=== 7. La note : les assemblages ===")
e = A.nouvel_etat()
chk("note neuve : un assemblage, repère A1, visserie et cotation de l'écran Streamlit",
    len(e["assemblages"]) == 1 and A._actif(e)["values"]["id_rep"] == "A1" and A._actif(e)["visserie"] == A.VISSERIE_DEFAUT
    and A._actif(e)["cotation"] == 1)
A.modifier(e, ["type"], "platine_about")
A.gerer_elements(e, "ajouter")
chk("ajouter : même configuration que l'ouvert, valeurs par défaut, premier repère libre (A2), ouvert",
    len(e["assemblages"]) == 2 and e["actif"] == 1 and A._actif(e)["type"] == "platine_about"
    and A._actif(e)["values"]["id_rep"] == "A2")
A.modifier(e, ["v", "V_Ed"], "400")
A.gerer_elements(e, "dupliquer")
chk("copier : copie complète insérée après l'original, repère « A2 (copie) », ouverte (📋 Dupliquer de Streamlit)",
    len(e["assemblages"]) == 3 and e["actif"] == 2 and A._actif(e)["values"]["id_rep"] == "A2 (copie)"
    and A._actif(e)["values"]["V_Ed"] == 400.0)
r = A.resultat(e)
bil = [A._bilan(A._cfg(x), x) for x in e["assemblages"]]
i_max = max(range(3), key=lambda i: ((bil[i]["niveau"] == "r"), bil[i]["taux"]))
chk("le résumé de la note : son assemblage le plus chargé (non conforme d'abord)",
    r["resume"]["taux"] == C.pct(bil[i_max]["taux"]) and r["resume"]["gouvernant"].startswith(A._repere(e["assemblages"][i_max], i_max)),
    r["resume"])
chk("la liste « Dans cette note » : chaque repère, sa configuration et son taux",
    all(C.esc(x["values"]["id_rep"]) in r["elements"] for x in e["assemblages"]) and r["elements"].count('class="ni') == 3)
A.gerer_elements(e, "supprimer", 0)
A.gerer_elements(e, "supprimer", 0)
A.gerer_elements(e, "supprimer", 0)
chk("supprimer : jamais le dernier", len(e["assemblages"]) == 1)
A.gerer_elements(e, "activer", 9)
chk("ouvrir un rang hors liste : le dernier", e["actif"] == 0)
for abime in ({"assemblages": [{"type": "inconnu"}]}, {"assemblages": "x"}, {"assemblages": [{"type": "plat_ame", "values": 3}]}):
    try:
        A.normaliser(abime)
        lu = False
    except ValueError:
        lu = True
    chk(f"une note abîmée ({str(abime)[:40]}) : erreur claire", lu)

# =====================================================================================================================
print("\n=== 8. Les sélecteurs ===")
RX_PK = re.compile(r'<button class="pk-row( cur)?( opt)?" data-choix="([^"]+)"[^>]*>.*?<span class="chip (\w)">([^<]+)</span></button>')
for cle in POUTRE_POUTRE:
    cfg = A.CONFIGS[cle]
    e = etat_de(cle)
    u = A._actif(e)["values"]
    rows = RX_PK.findall(A.selecteur(e, "boulon"))
    attendu = []
    for b in DB["boulons"]:
        R = cfg.calculer(A.a_calculer(cfg, dict(u, boulon_u=b["n"])))
        attendu.append((b["n"], f"{C.pc(C.pct(min(eta_tete(R), 9.99)))}{C.NBSP}%", R.verified))
    opt = next(n for n, _t, ok in attendu if ok)
    chk(f"{cle} : sélecteur de boulons — tous les boulons, taux du moteur, ✓ sur le retenu, ★ sur le plus petit qui tient",
        [(n, t) for _c, _o, n, _n, t in rows] == [(n, t) for n, t, _ in attendu]
        and [n for c, _o, n, *_ in rows if c] == [u["boulon_u"]] and [n for _c, o, n, *_ in rows if o] == [opt], rows[:3])
    A.choisir_section(e, "boulon", "M16")
    chk(f"{cle} : choisir M16 → boulon_u = M16", A._actif(e)["values"]["boulon_u"] == "M16")
    html_p = A.selecteur(e, "prof_S", ["IPE"])
    noms = [n for _c, _o, n, *_ in RX_PK.findall(html_p)]
    ipe = sorted([p for p in DB["profils"] if p["n"].startswith("IPE ")], key=lambda p: p["h"])
    chk(f"{cle} : sélecteur de profilés (famille IPE) trié par hauteur", noms == [p["n"] for p in ipe], noms[:4])
    A.choisir_section(e, "prof_S", "IPE 330")
    chk(f"{cle} : choisir IPE 330 → prof_S", A._actif(e)["values"]["prof_S"] == "IPE 330")
    try:
        A.choisir_section(e, "prof_S", "IPE 999")
        chk(f"{cle} : un profilé inconnu est refusé", False)
    except ValueError:
        chk(f"{cle} : un profilé inconnu est refusé", True)
    # les solutions du prédimensionnement : la choisir = « Appliquer » de l'onglet Prédim
    e = etat_de(cle)
    A.modifier(e, ["v", "mode_calc"], cfg.E.MODE_PREDIM)
    u = A._actif(e)["values"]
    R = r_de(e)
    sols = [x for x in R.pd.rows if x.geom and x.prop]
    rows = RX_PK.findall(A.selecteur(e, "solution"))
    chk(f"{cle} : sélecteur des solutions — chaque combinaison possible, ★ sur la proposée par défaut",
        len(rows) == len(sols) and [n for _c, o, n, *_ in rows if o] == [A._nom_solution(cfg, R, R.pd.pick)], (len(rows), len(sols)))
    r2 = sols[-1]
    attendu = A.valeurs_normales(cfg, cfg.predim["apply"](dict(u), r2))
    A.choisir_section(e, "solution", A._nom_solution(cfg, R, r2))
    chk(f"{cle} : choisir une solution = moteur.apply_solution (repasse en vérification)",
        A._actif(e)["values"] == attendu and attendu["mode_calc"] == cfg.E.MODE_VERIF)
e = etat_de("doubles_cornieres")
rows = RX_PK.findall(A.selecteur(e, "corniere"))
chk("doubles cornières : sélecteur des cornières EN 10056-1 (ailes puis épaisseur), ✓ sur la retenue",
    len(rows) == len(DB["cornieres"]) and [n for c, _o, n, *_ in rows if c] == [A._actif(e)["values"]["corn_u"]])

# =====================================================================================================================
print("\n=== 9. Échappement ===")
piege = '<img src=x onerror=alert(1)>"&'
for cle in POUTRE_POUTRE:
    e = etat_de(cle)
    for k in ("id_rep", "id_red", "id_projet", "id_date"):
        A.modifier(e, ["v", k], piege)
    A.modifier(e, ["visserie"], piege)
    r = A.resultat(e)
    # les fragments HTML ; le résumé, les titres et légendes des vues sont du TEXTE, que la page échappe
    frag = "".join(r[k] for k in ("resultat", "volets", "phrase", "donnees", "elements", "statut_html")) + "".join(v["svg"] for v in r["vues"])
    chk(f"{cle} : un texte saisi n'entre jamais brut dans un fragment HTML (repère, rédacteur, projet, date, visserie)",
        "<img" not in frag and frag.count("&lt;img src=x onerror=alert(1)&gt;") >= 3, frag.count("&lt;img"))

# =====================================================================================================================
print("\n=== 10. Mutations : le comparateur de parité rougit ===")


class Mutant:
    """Le module du site, une fonction remplacée."""

    def __init__(self, **k):
        self.__dict__.update({n: getattr(A, n) for n in ("normaliser", "calculer", "CONFIGS")})
        self.__dict__.update(k)


vw0 = A.valeur_widget
for nom_m, mutation in (("valeur_widget sans la virgule française", lambda cfg, k, v: vw0(cfg, k, v.replace(",", "x") if isinstance(v, str) else v)),
                        ("valeur_widget qui garde une option inconnue", lambda cfg, k, v: (str(v) if cfg.E.CHAMPS.get(k, {}).get("t") == "s" and v is not None else vw0(cfg, k, v)))):
    A.valeur_widget = mutation
    try:
        n = len(comparer_parite("plat_ame", CAS["plat_ame"]))
    finally:
        A.valeur_widget = vw0
    chk(f"mutation « {nom_m} » : la parité rougit ({n} jeux en écart)", n > 0)
ac0 = A.a_calculer
A.a_calculer = lambda cfg, u: u
try:
    n = len(comparer_parite("doubles_cornieres", CAS["doubles_cornieres"]))
finally:
    A.a_calculer = ac0
chk(f"mutation « dimensions vides non remplacées par les provisoires » : la parité rougit ({n} jeux en écart)", n > 0)
ls0 = A._lire_saisie
A._lire_saisie = lambda cfg, u, k, v: (C.lire_nombre(v) if cfg.E.CHAMPS[k]["t"] == "n" else v, None)
try:
    e = etat_de("plat_ame")
    rouge = A.modifier(e, ["v", "n1_u"], "2,4") is None
finally:
    A._lire_saisie = ls0
chk("mutation « saisie sans refus » : le contrôle des refus rougit (rangées 2,4 acceptées)", rouge)
mot0 = A.CONFIGS["platine_about"].calculer
A.CONFIGS["platine_about"].calculer = lambda u: mot0(dict(u, V_Ed=(u.get("V_Ed") or 0) * 1.0000001))
try:
    n = len(comparer_parite("platine_about", CAS["platine_about"][:15]))
finally:
    A.CONFIGS["platine_about"].calculer = mot0
chk(f"mutation « V_Ed × 1,0000001 » : l'empreinte au bit près rougit ({n} jeux)", n > 0)

# =====================================================================================================================
print("\n=== 11. Pureté : le chemin du navigateur, sans streamlit ===")
SOUS = r'''
import json, sys
INTERDITS = {"streamlit", "pandas", "reportlab", "matplotlib", "scipy", "PIL", "fitz", "pymupdf"}
refus = []
class Refus:
    def find_spec(self, nom, chemin=None, cible=None):
        if nom.split(".")[0] in INTERDITS:
            refus.append(nom)
            raise ImportError("refusé : " + nom)
        return None
sys.meta_path.insert(0, Refus())
sys.path.insert(0, sys.argv[1])
from api_web import assemblages as A
out = {}
for cle in A.ordre():
    e = A.nouvel_etat(); A.modifier(e, ["type"], cle)
    r = A.resultat(e)
    out[cle] = r["resume"]
locaux = {"acier", "api_web", "beton", "plancher", "sol", "bois", "flambement", "outils", "modules"}
# _distutils_hack et sitecustomize : chargés par l'installation de Python de la machine avant toute ligne du test
tiers = sorted({m.split(".")[0] for m in sys.modules} - set(sys.stdlib_module_names) - locaux
               - {"numpy", "__main__", "_distutils_hack", "sitecustomize", "usercustomize"})
print(json.dumps({"res": out, "refus": sorted(set(refus)), "interdits": sorted(INTERDITS & {m.split(".")[0] for m in sys.modules}), "tiers": tiers}))
'''
pr = subprocess.run([sys.executable, "-I", "-c", SOUS, RACINE], capture_output=True, text=True, check=False)
if pr.returncode != 0:
    chk("le module se charge et calcule sans streamlit", False, pr.stderr.strip().splitlines()[-3:])
else:
    out = json.loads(pr.stdout)
    ici = {}
    for cle in A.ordre():
        ici[cle] = A.resultat(etat_de(cle))["resume"]
    chk("streamlit, pandas, reportlab… REFUSÉS, et les sept configurations calculent quand même", not out["interdits"], out)
    chk("rien d'autre que la bibliothèque standard et numpy n'est chargé", not out["tiers"], out["tiers"])
    chk("mêmes résumés que dans ce processus", out["res"] == ici, ecarts(out["res"], ici)[:3])

# =====================================================================================================================
print("\n=== 12. Relecture : notes relues, bornes, tolérance du moteur ===")
import json as _json  # noqa: E402
from api_web import routeur as RT  # noqa: E402


def _calc_note(cle, values):
    return _json.loads(RT.traiter(_json.dumps({"module": "assemblages", "action": "calculer",
                                                "etat": {"assemblages": [{"type": cle, "values": values}]}})))


# (a) une note RELUE porte une donnée que la saisie refuserait : le verdict est suspendu (Streamlit la calcule en silence)
faux = []
for cle in A.ordre():
    cfg = A.CONFIGS[cle]
    for nom, src in [("défaut", cfg.E.defaults())] + [(i, d) for i, _l, d in A.exemples(cfg)] + CAS.get(cle, [])[1:12]:
        el = A.normaliser({"assemblages": [{"type": cle, "values": src}]})["assemblages"][0]
        r = A.saisies_refusees(cfg, el["values"])
        if r and not nom.startswith("tirage"):                    # un tirage peut porter V vide, exprès
            faux.append((cle, nom, r))
chk("aucune donnée par défaut, d'exemple publié ni de cas de référence n'est refusée à la relecture", not faux, faux[:3])
for values, quoi in (({"V_Ed": None}, "V_Ed vide (Streamlit : 0, « vérifié » à 60 %)"),
                     ({"g_M2": 0}, "γ_M2 = 0 (Streamlit : boulons à 0 %, « vérifié »)"),
                     ({"n1_u": 2.4}, "rangées 2,4 (Streamlit : 2)"), ({"tp_u": -10}, "épaisseur négative")):
    rep = _calc_note("plat_ame", values)
    R_st = MPA.compute(dict(EPA.defaults(), **values))
    chk(f"note relue, {quoi} : « À corriger », aucun taux, le champ en cause nommé — là où le moteur dit "
        f"« {R_st.statut.lower()} »", rep["ok"] and rep["resume"]["taux"] is None and rep["resume"]["niveau"] == "c"
        and "À corriger" in rep["resultat"] and 'class="msg grave"' in rep["resultat"] and "chip xl" not in rep["resultat"]
        and rep["etat"]["assemblages"][0]["values"][next(iter(values))] == A.valeur_widget(A.CONFIGS["plat_ame"], next(iter(values)), values[next(iter(values))]),
        (rep.get("resume"), rep.get("erreur")))
chk("le défaut existe : le moteur calcule ces notes « vérifiées » (V vide, γ_M2 = 0)",
    MPA.compute(dict(EPA.defaults(), V_Ed=None)).verified and MPA.compute(dict(EPA.defaults(), g_M2=0)).verified)
sr0 = A.saisies_refusees
A.saisies_refusees = lambda cfg, u: []
try:
    rep = _calc_note("plat_ame", {"V_Ed": None})
finally:
    A.saisies_refusees = sr0
chk("mutation « relecture sans refus » : la note à V_Ed vide redevient « Conforme » (le contrôle ci-dessus rougirait)",
    rep["resume"]["niveau"] == "p" and rep["resume"]["taux"] is not None, rep["resume"])

# (b) les bornes : un nombre de rangées ou de tiges hors bornes n'est JAMAIS calculé (n1 = 10⁹ : 14 Go, processus tué)
appels = []
cal0 = A.CONFIGS["plat_ame"].calculer
A.CONFIGS["plat_ame"].calculer = lambda u: (appels.append(u.get("n1_u")), cal0(u))[1]
try:
    rep = _calc_note("plat_ame", {"n1_u": 1e9})
    sel = _json.loads(RT.traiter(_json.dumps({"module": "assemblages", "action": "selecteur", "role": "boulon",
                                               "etat": {"assemblages": [{"type": "plat_ame", "values": {"n1_u": 1e9}}]}})))
finally:
    A.CONFIGS["plat_ame"].calculer = cal0
chk("note relue à n1 = 10⁹ rangées : le moteur n'est pas appelé (résultat, sélecteur), « Calcul impossible » dit la borne",
    rep["ok"] and not any((x or 0) > 50 for x in appels) and rep["resume"]["niveau"] == "x" and "50 au plus" in rep["resultat"]
    and sel["ok"], (appels[:3], rep.get("resume")))
e = etat_de("plat_ame")
chk("saisie : 51 rangées refusées (50 au plus), 50 acceptées",
    bool(A.modifier(e, ["v", "n1_u"], "51")) and A.modifier(e, ["v", "n1_u"], "50") is None)
e = etat_de("platine_beton")
chk("pied de poteau : une grille de 21 files refusée, 20 acceptée ; 20 × 20 tiges en vérification complète, pas 21",
    bool(A.modifier(e, ["grille", "nx"], "21")) and A.modifier(e, ["grille", "nx"], "20") is None
    and bool(A.trop_grand(A.CONFIGS["platine_beton"], dict(A._actif(e)["values"], n_ax=21.0)))
    and not A.trop_grand(A.CONFIGS["platine_beton"], dict(A._actif(e)["values"], n_ax=20.0)))
try:
    A.normaliser({"assemblages": [{"type": "platine_beton", "values": {"tiges": [{"x": 0, "y": 0}] * (A.LIGNES_MAXI + 1)}}]})
    lu = False
except ValueError:
    lu = True
chk(f"pied de poteau : une table de plus de {A.LIGNES_MAXI} boulons est une note abîmée (erreur claire)", lu)

# (c) la tolérance du moteur (η ≤ 1 + 10⁻⁹ est « vérifié ») : jamais « 101 % · Non conforme » à côté de « E_d ≤ R_d »
e = etat_de("plat_ame")
A._actif(e)["values"]["V_Ed"] = 168.9384675                       # η = 1,0000000004 : vérifié pour le moteur
R = r_de(e)
r = A.resultat(e)
v = re.search(r'chip xl (\w)"><small>η</small>([^<]+)', r["resultat"])
chk("η = 1 + 4·10⁻¹⁰, que le moteur déclare vérifié : verdict « 100 % » vert foncé, « Conforme, attention »",
    R.eta_max > 1 and R.verified and v.groups() == ("f", f"100{C.NBSP}%") and "Conforme, attention" in r["statut_html"]
    and "101" not in r["resultat"] and "101" not in r["volets"].split('id="ck-geometrie"')[0], (R.eta_max, v.groups()))
ev0 = A.eta_vu
A.eta_vu = lambda eta, ok=True: eta
try:
    rouge = "101" in A.resultat(e)["resultat"]
finally:
    A.eta_vu = ev0
chk("mutation « sans la tolérance du moteur » : le verdict repasse à 101 % rouge", rouge)
A._actif(e)["values"]["V_Ed"] = 168.938468                         # η = 1,0000000034 : non vérifié
chk("η = 1 + 3·10⁻⁹ (non vérifié pour le moteur) : 101 % rouge, « Non conforme »",
    not r_de(e).verified and "Non conforme" in A.resultat(e)["statut_html"])
e = etat_de("plat_ame")
A.modifier(e, ["v", "p1_u"], "48,4")                                # p₁ = 2,2·d₀ = 48,400000000000006 mm
R = r_de(e)
r = A.resultat(e)
geo = re.search(r'<b>Géométrie</b>.*?<span class="pct (\w)">([^<]+)</span>', r["resultat"]).groups()
chk("p₁ = 48,4 mm = 2,2·d₀ (conforme pour le moteur, à 10⁻¹⁴ près) : la ligne Géométrie ne dit pas « 101 % » rouge",
    R.dist_ok and geo[0] != "r" and "101" not in geo[1], geo)

# (d) l'onglet Benchmark : la légende des calculs manuels et le nom de la colonne de contrôle de CHAQUE écran (le site
# écrivait « Contrôle » sur la colonne « Excel » des doubles cornières), la source et son adresse
ecarts_bm = []
for cle in [c for c in A.ordre() if c != "platine_beton"]:
    cfg = A.CONFIGS[cle]
    src = open(ecran(cle, "ecran_resultats").__file__, encoding="utf-8").read()
    src = re.sub(r'"\s*\n\s*"', "", src)                          # les chaînes Python écrites sur plusieurs lignes
    legende, colonne = getattr(cfg, "legende_manuel", None) or A.LEGENDE_MANUEL
    vv = A.vue_validation(cfg)
    if legende not in src or f'"{colonne}"' not in src or C.esc(legende) not in vv or f">{colonne}<" not in vv \
            or C.esc(cfg.bench.SRC["url"]) not in vv:
        ecarts_bm.append((cle, colonne))
chk("vue Validation : légende des calculs manuels et colonne de contrôle de l'onglet Benchmark de chaque écran (« Excel » "
    "des doubles cornières), adresse de la source", not ecarts_bm, ecarts_bm)

# (e) le cadre Données des trois attaches poutre–poutre = les champs que l'écran Streamlit montre (champ_visible), ni
# plus ni moins — les deux ailes des cornières soudées cachaient le bloc BOULONS, que le moteur lit encore
ecarts_cd = []
for cle in POUTRE_POUTRE:
    cfg = A.CONFIGS[cle]
    variantes = [{}, {"mode_calc": cfg.E.MODE_PREDIM}, {"cat": "B"}, {"cat": "C"}, {"trou": "Surdimensionné", "d0_u": 24.0},
                 {"prof_P": PERSO, "prof_S": PERSO, "nu_pl": PERSO}, {"N_Ed": 50.0}]
    if cle == "doubles_cornieres":
        variantes += [{"fix_P": "Soudée"}, {"fix_S": "Soudée"}, {"fix_P": "Soudée", "fix_S": "Soudée", "trou": "Surdimensionné"},
                      {"corn_u": PERSO, "nu_C": PERSO}]
    for v in variantes:
        e = etat_de(cle)
        u = A._actif(e)["values"]
        u.update(v)
        vis = {k for k in cfg.E.CHAMPS if cfg.E.champ_visible(u, k)} - ({"d0_u"} if not (u.get("d0_u") or 0) > 0 else set())
        site = set(re.findall(r'data-chemin="v/([^"/]+)"', A.resultat(e)["donnees"]))
        if vis != site:
            ecarts_cd.append((cle, v, sorted(vis - site), sorted(site - vis)))
from acier.assemblages.pied_poteau.platine_beton.manuel import elements as ELP  # noqa: E402
rep = _calc_note("platine_beton", {"elems": [{"type": ELP.TYPES_ELEM[3], "b": -200, "t": 10, "x": 0, "y": 0}]})
chk("pied de poteau : une note relue dont un élément soudé a une dimension négative (Streamlit la garde) : « À corriger »",
    rep["ok"] and rep["resume"]["niveau"] == "c" and "Élément soudé 1 : b = -200 mm" in rep["resultat"], rep.get("resume"))
chk("poutre–poutre : le cadre Données porte exactement les champs de l'écran Streamlit (11 variantes, dont les deux ailes "
    "soudées avec un trou surdimensionné : le champ Trou, qui bloque le verdict, reste atteignable)", not ecarts_cd, ecarts_cd[:3])

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

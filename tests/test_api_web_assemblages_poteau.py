# -*- coding: utf-8 -*-
"""La famille POUTRE–POTEAU du module « Assemblages métalliques » du site (api_web/assemblages_poteau.py) : plat
d'âme soudé, doubles cornières, platine d'about partielle — sur l'aile ou dans l'âme du poteau. Ce que le site calcule
et affiche est ce que calcule et affiche l'écran Streamlit ; chaque contrôle rougit si l'on retire ce qu'il protège.

  1. Les trois configurations : enregistrées avec LEUR moteur (acier/assemblages/poutre_poteau/…), sections pensées
     (le cas d'appui en tête, le poteau, la poutre portée et sa position, la pièce, les boulons…), chaque champ que
     l'écran Streamlit montre est dans le cadre Données, et seulement lui (entrees.champ_visible), aile et âme.
  2. Les copies NOMMÉES de l'écran Streamlit, rejouées avec un faux « st » sur des dizaines de résultats (aile, âme,
     traction, tying, prédimensionnement, cornières soudées) : PERIMETRE_AILE, « paramètres retenus », « détail de la
     solution retenue », règle de choix, ce qui est sans objet et non revérifié (repli Périmètre), la règle du Tableau
     3.3, la légende de la matrice des ailes, les tableaux de répartition par boulon, les décimales du Benchmark.
  3. La parité du calcul (fichier → saisies → moteur), chemin de l'écran Streamlit et chemin du site, au bit près,
     sur des jeux qui couvrent les deux cas d'appui, la traction et le tying, les cornières soudées, les profilés
     personnalisés vides, le prédimensionnement.
  4. L'écran Streamlit LUI-MÊME (AppTest) : les trois écrans poutre–poteau (fpc_, dcc_, pac_), des jeux chargés dans
     leur état comme le fait « Ouvrir » ; leur résultat a l'empreinte de celui du site.
  5. Ce que le site affiche = le moteur : verdict, une ligne par pièce, la gouvernante, E_d / R_d dans chaque fenêtre,
     détail des formules et ligne η, alertes, les trois dessins (élévation, plan, coupe A-A) au caractère près, note
     texte, méthode (limites de methode.limites(ame)), validation, prédim, matrice des ailes, répartition par boulon,
     hypothèses (« poteau », « poutre portée »).
  6. La saisie : cas d'appui, refus (rangées 2,4, γ = 0, V vide, recoupe négative…), configuration changée (la pièce
     porteuse et le grugeage d'une poutre ne suivent pas sur un poteau ; le cas d'appui suit dans la famille).
  7. Les sélecteurs (poteau, poutre portée, boulon, cornière, solutions) et l'échappement.
  8. Mutations : abîmer une copie nommée, le calcul de 10⁻⁷, les colonnes du prédimensionnement fait rougir.

    /usr/bin/python3.12 tests/test_api_web_assemblages_poteau.py        (streamlit requis pour 2 et 4)
"""
import contextlib
import copy
import html
import importlib
import io
import json
import logging
import os
import random
import re
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from acier.bibliotheques import DB, PERSO  # noqa: E402
from api_web import assemblages as A  # noqa: E402
from api_web import assemblages_poteau as PT  # noqa: E402
from api_web import commun as C  # noqa: E402

OK, KO = [], []
TIRAGES = int(os.environ.get("TIRAGES", "40"))       # jeux tirés au hasard par configuration (partie 3)
CAS_ECRAN = int(os.environ.get("CAS_ECRAN", "10"))   # jeux rejoués dans l'écran Streamlit (partie 4)


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "ÉCHEC ") + nom + (f"   [{str(info)[:300]}]" if info and not cond else ""))


POTEAU = {"plat_ame_poteau": "acier.assemblages.poutre_poteau.plat_ame_poteau",
          "doubles_cornieres_poteau": "acier.assemblages.poutre_poteau.doubles_cornieres_poteau",
          "platine_about_poteau": "acier.assemblages.poutre_poteau.platine_about_poteau"}
PREFIXES = {"plat_ame_poteau": "fpc_", "doubles_cornieres_poteau": "dcc_", "platine_about_poteau": "pac_"}
REFS = {"plat_ame_poteau": "ref_fpc.json", "doubles_cornieres_poteau": "ref_dcc.json", "platine_about_poteau": "ref_pac.json"}
AME = "Âme du poteau"


def ecran(cle, nom):
    return importlib.import_module(f"{POTEAU[cle]}.{nom}")


def etat_de(cle, values=None, **el):
    """Une note d'un assemblage de la configuration ``cle``, lu comme un fichier (les défauts de la configuration)."""
    return A.normaliser({"assemblages": [dict({"type": cle, "values": values or {}}, **el)], "actif": 0})


def r_de(e):
    el = A._actif(e)
    return A.calculer(A._cfg(el), el)[0]


def sans_balises(s):
    return html.unescape(re.sub(r"<[^>]+>", "", s))


def chiffres(s):
    """Un nombre écrit, sans ses espaces de milliers (fine, insécable, ordinaire)."""
    return re.sub(r"[\s  ]", "", str(s))


# ------------------------------------------------------------------ l'empreinte d'un résultat
def _x(v):
    return repr(v) if isinstance(v, float) else v


def empreinte(R):
    """Tout ce que l'écran montre d'un résultat, flottants en repr (égalité AU BIT)."""
    if R is None:
        return None
    d = {"eta_max": _x(R.eta_max), "verified": R.verified, "statut": R.statut, "geo_ok": R.get("geo_ok"),
         "dist_ok": R.get("dist_ok"), "gov": R.gov.label if R.gov else None, "pred": R.get("pred"),
         "reserve": R.get("reserve"), "boulon": R.get("boulon"), "appui": R.get("appui"), "M_col": _x(R.get("M_col"))}
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
    """Un jeu de saisies BRUT (comme un fichier) : défauts perturbés, le cas d'appui, la traction et le tying, les
    fixations des cornières, des profilés du catalogue ou personnalisés (vides parfois), des textes à virgule, le
    prédimensionnement."""
    cfg = A.CONFIGS[cle]
    E = cfg.E
    d = E.defaults()
    for k, f in E.CHAMPS.items():
        if k.startswith("id_"):
            continue
        r = rng.random()
        if f["t"] == "s" and r < 0.2:
            d[k] = rng.choice(list(f["o"]))
        elif f["t"] == "n" and isinstance(d.get(k), (int, float)) and r < 0.25:
            x = d[k] * rng.choice([0.5, 0.75, 0.9, 1.1, 1.25, 1.6, 2.0])
            d[k] = round(x, rng.choice([0, 1])) if abs(x) > 2 else round(x, 2)
            if rng.random() < 0.15:
                d[k] = str(d[k]).replace(".", ",")
    d["appui"] = E.APPUI_AME if rng.random() < 0.45 else E.APPUI_AILE
    if rng.random() < 0.35:
        d["N_Ed"] = rng.choice([20, 60, 150, "35,5"])
        d["nat_N"] = rng.choice(list(E.CHAMPS["nat_N"]["o"]))
    if rng.random() < 0.3:
        d["prof_P"] = rng.choice([p["n"] for p in DB["profils"] if p["n"].split()[0] in ("HEA", "HEB", "HEM")])
    if rng.random() < 0.25:
        d["prof_S"] = rng.choice([p["n"] for p in DB["profils"] if p["n"].startswith(("IPE 2", "IPE 3", "IPE 4", "HEA 2"))])
    if d["appui"] == E.APPUI_AME and rng.random() < 0.5:
        d["rec_S"] = rng.choice([0, 60, 80, 120])
    if rng.random() < 0.1:
        d["prof_S"] = PERSO
        if rng.random() < 0.5:
            for x in ("hS_u", "bS_u", "twS_u", "tfS_u"):
                d[x] = None
        else:
            p = rng.choice(DB["profils"])
            d.update(hS_u=p["h"], bS_u=p["b"], twS_u=p["tw"], tfS_u=p["tf"], rS_u=p["r"])
    if rng.random() < 0.05:
        d["V_Ed"] = None
    if rng.random() < 0.25:
        d["mode_calc"] = E.MODE_PREDIM
    return d


def jeux(cle, n, graine=1):
    """Les jeux d'une configuration : défauts (aile et âme), exemples publiés, cas de référence du dépôt, tirages."""
    cfg = A.CONFIGS[cle]
    out = [("défaut", {}), ("défaut âme", {"appui": AME}),
           ("tying", {"N_Ed": 120, "nat_N": cfg.E.NAT_TYING}), ("tying âme", {"appui": AME, "N_Ed": 120, "nat_N": cfg.E.NAT_TYING}),
           ("prédim", {"mode_calc": cfg.E.MODE_PREDIM}), ("prédim âme", {"mode_calc": cfg.E.MODE_PREDIM, "appui": AME})]
    if cle == "doubles_cornieres_poteau":
        S = cfg.E.CHAMPS["fix_P"]["o"][1]
        out += [("ailes A soudées", {"fix_P": S}), ("ailes B soudées", {"fix_S": S}), ("tout soudé", {"fix_P": S, "fix_S": S}),
                ("ailes A soudées, âme", {"fix_P": S, "appui": AME, "corn_u": "L80x80x8"})]
    for b in (getattr(cfg.bench, "BENCH", None) or []):
        out.append((f"exemple {b['id']}", copy.deepcopy(b["inp"])))
    with open(os.path.join(RACINE, "tests", REFS[cle]), encoding="utf-8") as fh:
        for nom, x in json.load(fh).items():
            if isinstance(x, dict) and isinstance(x.get("inp"), dict):
                out.append((f"réf. {nom}", x["inp"]))
    rng = random.Random(f"poteau-{cle}-{graine}")
    out += [(f"tirage {i}", tirage(cle, rng)) for i in range(n)]
    return out


# ------------------------------------------------------------------ le chemin de l'écran Streamlit, sans écran
def chemin_streamlit(cle, src):
    """Ce que fait l'écran Streamlit d'un fichier : entrees.charger_json, chaque clé par carte.valeur_widget
    (ecrire_entrees), puis interface.show : carte.dimensions_vides, carte.provisoire, compute."""
    from acier.assemblages import carte as kit
    D = ecran(cle, "ecran_saisie").D
    cfg = A.CONFIGS[cle]
    brut = cfg.E.charger_json(src)
    compute = ecran(cle, "interface").compute
    u = {k: kit.valeur_widget(D, k, brut.get(k)) for k in cfg.E.CLES}
    vides = kit.dimensions_vides(D, u)
    try:
        R = compute(kit.provisoire(D, u) if vides else u)
        err = None
    except Exception as e:  # noqa: BLE001
        R, err = None, type(e).__name__
    return u, sorted(vides), R, err


def chemin_site(cle, src):
    try:
        e = A.normaliser({"assemblages": [{"type": cle, "values": src}], "actif": 0})
    except ValueError as x:
        return None, None, None, f"refus : {x}"
    el = e["assemblages"][0]
    R, vides, err = A.calculer(A.CONFIGS[cle], el)
    return el["values"], sorted(vides), R, (err.split(" :")[0] if err else None)


def comparer_parite(cle, cas):
    out = []
    for nom, src in cas:
        u1, v1, R1, e1 = chemin_streamlit(cle, src)
        u2, v2, R2, e2 = chemin_site(cle, src)
        if u2 is None:
            out.append((nom, "le site refuse le fichier", e2))
            continue
        du = [k for k in u1 if _x(u1[k]) != _x(u2.get(k))]
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


# ------------------------------------------------------------------ un faux « st » pour rejouer l'écran
class FauxSt:
    """Un « st » qui enregistre au lieu d'afficher (les tableaux arrivent par tableau_md, remplacé)."""

    def __init__(self):
        self.tables, self.textes, self.expanders = [], [], []

    def markdown(self, x="", *a, **k):
        (self.tables if isinstance(x, tuple) else self.textes).append(x)

    caption = markdown

    def columns(self, spec, **k):
        return [self] * (spec if isinstance(spec, int) else len(spec))

    def container(self, *a, **k):
        return self

    def expander(self, titre="", *a, **k):
        self.expanders.append(titre)
        return self

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
    """ecran_resultats avec un faux « st » ; tableau_md rend ses lignes ; les briques de ui_commun qui écriraient
    dans le vrai streamlit sont remplacées par des enregistreurs (le repli Périmètre, le Tableau 3.3, la légende)."""
    fs = FauxSt()
    fs.perimetre = []
    noms = ("st", "tableau_md", "bloc_alerte", "expander_perimetre", "tableau_33_replie", "legende_natures")
    avant = {n: getattr(ER, n, None) for n in noms}
    ER.st = fs
    ER.tableau_md = lambda h, rows, **k: ("TAB", tuple(h), [[str(x) for x in r] for r in rows])
    ER.bloc_alerte = lambda *a, **k: None
    ER.expander_perimetre = lambda non, sans, *a, **k: fs.perimetre.append((list(non), list(sans)))
    ER.tableau_33_replie = lambda ok, rendre, *a, **k: rendre()
    ER.legende_natures = lambda *a, **k: None
    try:
        yield fs
    finally:
        for n, v in avant.items():
            setattr(ER, n, v)


# =====================================================================================================================
print("=== 1. Les trois configurations et leur cadre Données ===")
chk("le fichier d'aide n'a pas de MODULE ; ses trois configurations sont au registre de l'aiguillage, dans l'ordre",
    not hasattr(PT, "MODULE") and [c for c in A.ordre() if c in POTEAU] == list(POTEAU))
for cle, chemin in POTEAU.items():
    cfg = A.CONFIGS[cle]
    chk(f"{cle} : le moteur est celui de la configuration (entrees, compute, synthese, schemas, methode, benchmark)",
        cfg.E is ecran(cle, "entrees") and cfg.calculer is ecran(cle, "moteur").compute and cfg.S is ecran(cle, "synthese")
        and cfg.bench is ecran(cle, "benchmark") and cfg.methode == ecran(cle, "methode").METHODE and cfg.famille == "poutre_poteau")
    vus_ko = []
    for nom, v in (("aile", {}), ("âme", {"appui": AME}), ("prédim", {"mode_calc": cfg.E.MODE_PREDIM}),
                   ("tying", {"N_Ed": 50, "nat_N": cfg.E.NAT_TYING, "cat": "B"}),
                   ("perso", {"prof_P": PERSO, "prof_S": PERSO, "nu_P": PERSO, "trou": cfg.E.CHAMPS["trou"]["o"][-1]})):
        e = etat_de(cle, v)
        u = A._actif(e)["values"]
        don = A.resultat(e)["donnees"]
        lus = set(re.findall(r'data-chemin="v/([^"/]+)"', don))
        attendus = {k for k in cfg.E.CHAMPS if cfg.E.champ_visible(u, k)}
        if "d0_u" in attendus and not (u.get("d0_u") or 0) > 0:
            attendus.discard("d0_u")                     # trou normal : la case « d0 imposé » le remplace
        if lus != attendus:
            vus_ko.append((nom, sorted(attendus - lus), sorted(lus - attendus)))
    chk(f"{cle} : chaque champ que l'écran Streamlit montre est dans le cadre Données, et seulement lui (aile, âme, "
        "prédim, tying, personnalisé)", not vus_ko, vus_ko)
    don = A.resultat(etat_de(cle))["donnees"]
    titres = re.findall(r'<span class="tt">([^<]+)</span>', don)
    chk(f"{cle} : sections pensées — Assemblage, Poteau porteur, Poutre portée, Grugeage, la pièce, Boulons, Efforts, "
        f"Coefficients, Cartouche ({len(titres)})",
        titres[:4] == ["Assemblage", "Poteau porteur", "Poutre portée", "Grugeage"] and titres[-3:] ==
        ["Efforts ELU", "Coefficients et modèle", "Cartouche et affichage"], titres)
    tete = don.split('data-sec="po"')[0]
    chk(f"{cle} : le cas d'appui est en tête (section Assemblage), en segments « Sur l'aile / Dans l'âme »",
        'data-chemin="v/appui"' in tete and tete.count('data-chemin="v/appui"') == 2 and 'data-valeur="Âme du poteau">Dans l&#x27;âme<' in tete
        and 'select data-chemin="type"' in tete)
    e = etat_de(cle, {"appui": AME})
    don = A.resultat(e)["donnees"]
    portee = don.split('data-sec="pt"')[1].split('data-sec="gr"')[0]
    chk(f"{cle} : en cas (b), la recoupe des semelles et le jeu latéral sont dans « Poutre portée › Position »",
        'data-chemin="v/rec_S"' in portee and 'data-chemin="v/j_lat"' in portee)
e = etat_de("doubles_cornieres_poteau", {"fix_P": "Soudée", "fix_S": "Soudée"})
don = A.resultat(e)["donnees"]
chk("cornières soudées des deux côtés : pas de section Boulons, une section Soudures, pas de visserie",
    'data-sec="bo"' not in don and 'data-sec="so"' in don and 'data-chemin="visserie"' not in don)

# =====================================================================================================================
print("\n=== 2. Les copies nommées de l'écran Streamlit (faux « st ») ===")
chk("PERIMETRE_AILE du plat d'âme = plat_ame_poteau/ecran_resultats.PERIMETRE_AILE",
    PT.PERIMETRE_AILE_PA == ecran("plat_ame_poteau", "ecran_resultats").PERIMETRE_AILE)
chk("PERIMETRE_AILE des cornières = doubles_cornieres_poteau/ecran_resultats.PERIMETRE_AILE",
    PT.PERIMETRE_AILE_DC == ecran("doubles_cornieres_poteau", "ecran_resultats").PERIMETRE_AILE)
for cle in POTEAU:
    es = ecran(cle, "ecran_saisie")
    cfg = A.CONFIGS[cle]
    chk(f"{cle} : options affichées du cas d'appui et de la nature de N_Ed = les options du moteur ; dimensions "
        "provisoires = ecran_saisie.DIMENSIONS",
        set(cfg.affichage["appui"]) == set(cfg.E.CHAMPS["appui"]["o"]) and set(cfg.affichage["nat_N"]) == set(cfg.E.CHAMPS["nat_N"]["o"])
        and set(cfg.affichage["mode_calc"]) == set(cfg.E.CHAMPS["mode_calc"]["o"]) and cfg.dimensions == es.D.DIMENSIONS)
chk("cornières : options affichées de la grande aile = les options du moteur",
    set(A.CONFIGS["doubles_cornieres_poteau"].affichage["orient"]) == set(A.CONFIGS["doubles_cornieres_poteau"].E.CHAMPS["orient"]["o"]))

CAS2 = {cle: jeux(cle, 22, graine=2) for cle in POTEAU}
for cle in POTEAU:
    ER = ecran(cle, "ecran_resultats")
    cfg = A.CONFIGS[cle]
    e_par, e_det, e_per, e_33, e_tab, e_leg = [], [], [], [], [], []
    n_par = n_det = n_ame = n_aile = 0
    for nom, src in CAS2[cle]:
        el = A.normaliser({"assemblages": [{"type": cle, "values": src}]})["assemblages"][0]
        R, vides, err = A.calculer(cfg, el)
        if R is None or vides:
            continue
        n_ame += bool(R.ame)
        n_aile += bool(R.aile)
        # « Paramètres retenus »
        with faux_st(ER) as fs:
            ER._parametres_retenus(R)
        ecran_ = [t for t in fs.tables if t[1] == ("Paramètre", "Valeur")][0][2]
        site = [[str(a), str(b)] for a, b in cfg.parametres(R)]
        n_par += 1
        if ecran_ != site:
            e_par.append((nom, ecarts(ecran_, site)[:2]))
        # l'onglet Vérifications : repli Périmètre, Tableau 3.3, répartition par boulon, légende des ailes
        with faux_st(ER) as fs:
            ER.onglet_verifications(R)
        non, sans = fs.perimetre[0]
        cfg_r = A.pour(cfg, R)
        vm = A.vue_methode(cfg_r, R)
        if list(A.non_reverifies(cfg_r, R)) != non or any(C.esc(s) not in vm for s in sans + non):
            e_per.append((nom, "périmètre", [s[:40] for s in sans + non if C.esc(s) not in vm][:2]))
        cap33 = [t for t in fs.textes if isinstance(t, str) and t.startswith("e ≥ 1,2·d0")]
        geo = A._volet_geometrie(cfg_r, R, [])
        pied33 = sans_balises(geo.split('class="ck-autres">')[1].split("</div>")[0]) if R.dist else ""
        if R.dist and (len(cap33) != 1 or not pied33.startswith(cap33[0].rstrip("."))):
            e_33.append((nom, cap33, pied33[:120]))
        rep = [x for x in fs.expanders if x.startswith("Répartition élastique par boulon – ")]
        site_rep = [t for t, tab in (cfg.tableaux_boulons(R) if cfg.tableaux_boulons else [])]
        if [x.split(" – ", 1)[1] for x in rep] != site_rep:
            e_tab.append((nom, rep, site_rep))
        if cle == "doubles_cornieres_poteau" and PT.YDC.matrice_cornieres(R):
            if PT.LEGENDE_AILES not in fs.textes or C.esc(PT.LEGENDE_AILES) not in PT.matrice_cornieres(R):
                e_leg.append(nom)
        # l'onglet Prédim : détail de la solution retenue, règle de choix, colonnes du tableau
        if R.get("pd") is None:
            continue
        with faux_st(ER) as fs:
            ER.onglet_predim(R, el["values"], lambda u: None)
        det = [t for t in fs.tables if t[1] == ("", "")]
        n_det += 1
        if not det or det[0][2] != A.detail_predim(cfg, R.pd):
            e_det.append((nom, ecarts(det[0][2] if det else [], A.detail_predim(cfg, R.pd))[:2]))
        if cfg.predim["regle"] not in fs.textes:
            e_det.append((nom, "règle", [t[:60] for t in fs.textes if "Règle" in str(t)]))
        # les colonnes de « Toutes les combinaisons » : les valeurs que l'écran écrit, ligne par ligne
        cols = cfg.predim["colonnes"](R) if callable(cfg.predim["colonnes"]) else cfg.predim["colonnes"]
        rows = [r for r in R.pd.rows if r.geom]
        cellules = [t.strip("*") for t in fs.textes if isinstance(t, str)]
        vp = A.vue_predim(cfg, R, el["values"])
        for r in rows[:6]:
            for _h, f in cols:
                if f(r) not in cellules and cle != "doubles_cornieres_poteau":
                    e_det.append((nom, "colonne", _h, f(r)))
                if C.esc(f(r)) not in vp:
                    e_det.append((nom, "vue Prédim", _h, f(r)))
    chk(f"{cle} : les jeux couvrent l'aile ({n_aile}) et l'âme ({n_ame})", n_aile >= 5 and n_ame >= 5)
    chk(f"{cle} : « Paramètres retenus » = ce qu'écrit ecran_resultats._parametres_retenus ({n_par} résultats)",
        n_par >= 20 and not e_par, e_par[:2])
    chk(f"{cle} : « Détail de la solution retenue », règle de choix et colonnes des combinaisons = onglet_predim "
        f"({n_det} résultats)", n_det >= 20 and not e_det, e_det[:3])
    chk(f"{cle} : Périmètre — non revérifiés (methode.limites) et sans objet (dont l'aile du poteau) = le repli de "
        "l'écran, dits dans la vue Méthode", not e_per, e_per[:2])
    chk(f"{cle} : la règle du Tableau 3.3 de la fenêtre Géométrie = la légende de l'écran", not e_33, e_33[:2])
    chk(f"{cle} : répartition élastique par boulon = les tableaux de l'onglet Vérifications", not e_tab, e_tab[:2])
    if cle == "doubles_cornieres_poteau":
        chk("cornières : la légende de la matrice des ailes = celle de l'écran (aile A sur le POTEAU)", not e_leg, e_leg[:3])

# le Benchmark : les décimales de chaque configuration
for cle in POTEAU:
    ER = ecran(cle, "ecran_resultats")
    cfg = A.CONFIGS[cle]
    B = cfg.bench.run_bench()
    with faux_st(ER) as fs:
        ER.onglet_benchmark(B, lambda u: None)
    ecran_ = [t[2] for t in fs.tables if t[1][:1] == ("Vérification",)]
    vv = A.vue_validation(cfg)
    manque = []
    for tab in ecran_:
        for li in tab:
            ref, val = li[2], li[3]
            ref_n = ref.split(" ")[0]                     # F() n'écrit pas de séparateur de milliers : le nombre, puis l'unité
            if chiffres(ref_n) not in chiffres(sans_balises(vv)) or chiffres(val) not in chiffres(sans_balises(vv)):
                manque.append((li[0][:30], ref, val))
    chk(f"{cle} : vue Validation — chaque référence et chaque valeur de l'outil écrites avec les décimales de l'onglet "
        f"Benchmark ({sum(len(t) for t in ecran_)} lignes)", ecran_ and not manque, manque[:3])
chk("vue Validation : un rapport sans unité ne porte pas « - »", " -</td>" not in A.vue_validation(A.CONFIGS["plat_ame_poteau"]))

# =====================================================================================================================
print("\n=== 3. La parité du calcul : fichier → saisies → moteur (écran Streamlit / site) ===")
logging.disable(logging.CRITICAL)
CAS = {}
total = 0
for cle in POTEAU:
    CAS[cle] = jeux(cle, TIRAGES)
    diff = comparer_parite(cle, CAS[cle])
    total += len(CAS[cle])
    etats = {"vérifié": 0, "non vérifié": 0, "à compléter": 0, "erreur": 0, "prédim": 0, "âme": 0, "traction": 0}
    for _nom, src in CAS[cle]:
        u, v, R, err = chemin_site(cle, src)
        if err:
            etats["erreur"] += 1
        elif v:
            etats["à compléter"] += 1
        elif R is not None:
            etats["vérifié" if R.verified else "non vérifié"] += 1
            etats["prédim"] += bool(R.get("pred"))
            etats["âme"] += bool(R.ame)
            etats["traction"] += bool((C.lire_nombre(u.get("N_Ed")) or 0) > 0)
    chk(f"{cle} : {len(CAS[cle])} jeux, même empreinte au bit près ({etats})", not diff, diff[:3])
    chk(f"{cle} : les jeux couvrent vérifié, non vérifié, à compléter, prédim, âme et traction",
        all(etats[k] > 0 for k in ("vérifié", "non vérifié", "à compléter", "prédim", "âme", "traction")), etats)
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
        bons = [c for c in CAS[cle] if not chemin_site(cle, c[1])[1] and chemin_site(cle, c[1])[3] is None]
        cas = bons[:CAS_ECRAN // 2] + [c for c in bons if c[1].get("appui") == AME][:CAS_ECRAN - CAS_ECRAN // 2]
        diffs, vus, ames = [], 0, 0
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
            ames += bool(R_ecran is not None and R_ecran.ame)
            d = ecarts(empreinte(R_ecran), empreinte(R_site))
            if d or at.exception:
                diffs.append((nom, d[:3], at.exception))
        chk(f"{cle} : {vus} jeux chargés dans l'écran Streamlit (dont {ames} dans l'âme), son résultat a l'empreinte de "
            "celui du site", vus == len(cas) and len(cas) >= min(CAS_ECRAN, 6) and ames >= 2 and not diffs, diffs[:2])
        # le cas d'appui basculé au clavier : même résultat
        at.session_state[pre + "appui"] = AME
        at.session_state[pre + "V_Ed"] = 150.0
        with contextlib.redirect_stderr(io.StringIO()):
            at.run()
        e = etat_de(cle, cas[-1][1] if cas else {})
        A.modifier(e, ["v", "appui"], AME)
        A.modifier(e, ["v", "V_Ed"], "150")
        chk(f"{cle} : « Âme du poteau » et « V_Ed = 150 » saisis : même résultat à l'écran et sur le site",
            not at.exception and not ecarts(empreinte(at.session_state[f"_{pre}R"]), empreinte(r_de(e))))

# =====================================================================================================================
print("\n=== 5. Ce que le site affiche = le moteur ===")
RX_EL = re.compile(r'<div class="el( gov)?" data-calcul="([^"]+)"[^>]*><div class="l1"><b>([^<]+)</b>.*?<span class="pct (\w)">([^<]+)</span></div></div>')
for cle in POTEAU:
    cfg = A.CONFIGS[cle]
    M = ecran(cle, "schemas")
    defauts = []
    n = 0
    for nom, src in CAS[cle][:40]:
        e = etat_de(cle, src)
        el = A._actif(e)
        R, vides, err = A.calculer(cfg, el)
        if R is None or vides or A.saisies_refusees(cfg, el["values"]):     # relue « à corriger » : pas de verdict
            continue
        n += 1
        r = A.resultat(e)
        lignes = RX_EL.findall(r["resultat"])
        attendu = [f"{C.pc(A.taux(eta))}{C.NBSP}%" for _g, eta, _ok, _p in cfg.S.taux_par_element(R)]
        noms = [C.esc(g) for g, *_ in A._groupes(cfg, R)]           # les pièces de synthese.ELEMENTS
        trouve = [(b, t) for _g, _c, b, _n, t in lignes][:len(attendu)]
        if [t for _, t in trouve] != attendu or [b for b, _ in trouve] != noms:
            defauts.append((nom, "lignes", trouve, attendu))
        if f"{C.pc(A.taux(R.eta_max))}{C.NBSP}%" not in r["resultat"].split('class="verdict"')[1][:300]:
            defauts.append((nom, "verdict"))
        gov = [x for x in lignes if x[0]]
        if R.gov is not None and (len(gov) != 1 or gov[0][2] != C.esc(R.gov.grp)):
            defauts.append((nom, "ligne gouvernante", gov))
        pot = [x for x in lignes if x[2] == "Poteau porteur"]
        if pot and f"{'aile' if R.aile else 'âme'}</span>" not in r["resultat"].split("<b>Poteau porteur</b>")[1][:400]:
            defauts.append((nom, "cas d'appui sur la ligne du poteau"))
        for c in R.checks:
            if c.active:
                ed, rd, un = cfg.S.ed_rd(c)
                fen = r["volets"].split(f'id="ck-{A._slug(c.grp)}"')[1].split('class="flyout')[0]
                if f">{ed}{' ' + un if un else ''}<" not in fen or f">{rd}{' ' + un if un else ''}<" not in fen:
                    defauts.append((nom, "fenêtre", c.key, ed, rd))
                _aide, lf = cfg.formules(R, c)
                if not lf or lf[-1]["parts"][0] != "η" or lf[-1]["res"] != f"{C.pc(A.taux(c.eta, 1))} %" or C.lignes_fx(lf) not in fen:
                    defauts.append((nom, "détail", c.key))
        if r["resultat"].count('class="msg') != len(R.alerts) + bool(R.get("reserve")) + bool(R.get("pred")):
            defauts.append((nom, "alertes", len(R.alerts)))
        opt = M.options_ecran(R, el["cotation"], A._hl(R))
        opt.interactive, opt.poignees, opt.editables, opt.locked = False, False, el["cotation"] >= 1, None
        vues = {v["cle"]: v for v in r["vues"]}
        for k, f in (("elevation", M.elevation), ("plan", M.plan), ("coupe", M.coupe_aa)):
            if vues[k]["svg"] != f(R, opt).svg(identifiant=f"asm_{k}"):
                defauts.append((nom, "dessin", k))
        if C.esc(cfg.T.construire_texte(R, False)) not in vues["note"]["svg"]:
            defauts.append((nom, "note texte"))
        lim = ecran(cle, "methode").limites(R.ame)
        if not all(C.esc(t) in vues["methode"]["svg"] for t in lim):
            defauts.append((nom, "limites de la méthode"))
        hyp = r["volets"].split('class="ck-hyp">')[1].split("</div>")[0]
        # un acier « Personnalisé » sans f_y (fichier) vaut f_y = 0 dans le moteur : il n'est pas dit (η = 10⁹ le dit)
        if ((R.fy_P and "poteau f<sub>y</sub>" not in hyp) or (R.fy_S and "poutre portée f<sub>y</sub>" not in hyp)
                or "principale" in hyp or "secondaire" in hyp):
            defauts.append((nom, "hypothèses", hyp[:120]))
    chk(f"{cle} : {n} résultats — verdict, lignes par pièce, gouvernante, cas d'appui, E_d / R_d et détail η de chaque "
        "vérification dans sa fenêtre, alertes, trois dessins, note, limites, hypothèses = le moteur", n >= 25 and not defauts, defauts[:3])
    e = etat_de(cle)
    r = A.resultat(e)
    vues = {v["cle"]: v for v in r["vues"]}
    B = cfg.bench.run_bench()
    chk(f"{cle} : vues Élévation, Plan, Coupe A-A, Prédim, Note, Méthode, Validation ({B['statut']})",
        list(vues) == ["elevation", "plan", "coupe", "predim", "note", "methode", "validation"]
        and C.esc(B["statut"]) in vues["validation"]["svg"] and all(C.esc(b["id"]) in vues["validation"]["svg"] for b in B["bench"]))
    niveaux = []
    for nv in (0, 1, 2):
        A.modifier(e, ["cotation"], nv)
        niveaux.append(len(A.resultat(e)["vues"][0]["svg"]))
    chk(f"{cle} : les trois niveaux de cotation changent l'élévation", niveaux[0] < niveaux[1] <= niveaux[2], niveaux)
    e = etat_de(cle, {"mode_calc": cfg.E.MODE_PREDIM})
    R = r_de(e)
    r = A.resultat(e)
    vp = {v["cle"]: v for v in r["vues"]}["predim"]["svg"]
    rows = [x for x in R.pd.rows if x.geom]
    chk(f"{cle} : prédimensionnement — bandeau, ligne Proposition, vue Prédim ({len(rows)} combinaisons), fenêtre",
        "Mode prédimensionnement" in r["resultat"] and 'data-pk="solution"' in r["resultat"] and vp.count("<tr>") - 1 >= len(rows)
        and "★ retenue" in vp and 'id="ck-predim"' in r["volets"] and C.esc(cfg.predim["regle"]) in r["volets"])
# la matrice des ailes, la répartition par boulon, la pression diamétrale
R = r_de(etat_de("doubles_cornieres_poteau"))
vol = A.resultat(etat_de("doubles_cornieres_poteau"))["volets"]
mat = PT.YDC.matrice_cornieres(R)
chk("cornières : la fenêtre Cornières porte la matrice aile A · aile B (E_d / R_d de synthese.matrice_cornieres)",
    "Cornières — aile A / aile B" in vol and mat and all(f"{x['A']['Ed']} / {x['A']['Rd']}" in vol for x in mat if x["A"])
    and C.esc(PT.LEGENDE_AILES) in vol)
chk("cornières : la répartition élastique par boulon du groupe S est dans la fenêtre Boulons",
    "Répartition élastique par boulon · groupe S" in vol and all(C.fr(b.f, 2) in vol for b in R.tabS))
R = r_de(etat_de("plat_ame_poteau"))
vol = A.resultat(etat_de("plat_ame_poteau"))["volets"]
chk("plat d'âme : la répartition élastique par boulon du groupe est dans la fenêtre Boulons",
    "Répartition élastique par boulon · groupe" in vol and all(C.fr(b.f, 2) in vol for b in R.tab))
e = etat_de("plat_ame_poteau")
r = A.resultat(e)
chk("plat d'âme sur l'aile : pas de ligne « Poteau porteur » (aucune vérification active), et la vue Méthode dit pourquoi",
    "<b>Poteau porteur</b>" not in r["resultat"] and C.esc(PT.PERIMETRE_AILE_PA) in {v["cle"]: v for v in r["vues"]}["methode"]["svg"])
A.modifier(e, ["v", "appui"], AME)
r = A.resultat(e)
chk("plat d'âme dans l'âme : la ligne « Poteau porteur » paraît (cisaillement local), sans le texte de l'aile",
    "<b>Poteau porteur</b>" in r["resultat"] and C.esc(PT.PERIMETRE_AILE_PA) not in {v["cle"]: v for v in r["vues"]}["methode"]["svg"])
R = r_de(etat_de("platine_about_poteau", {"N_Ed": 80}))
vol = A.resultat(etat_de("platine_about_poteau", {"N_Ed": 80}))["volets"]
chk("platine en traction sur l'aile : tronçon en T de l'aile du poteau et âme en traction transversale dans la fenêtre "
    "Poteau porteur", R.aile and all(c.active for c in R.checks if c.key in ("tfP", "twtP"))
    and all(C.esc(PT.YPB.court(c)) in vol.split('id="ck-poteau-porteur"')[1].split('class="flyout')[0] for c in R.checks if c.key in ("tfP", "twtP")))

# =====================================================================================================================
print("\n=== 6. La saisie ===")
for cle in POTEAU:
    e = etat_de(cle)
    msg = A.modifier(e, ["v", "appui"], AME)
    chk(f"{cle} : « Âme du poteau » choisi au segment → le moteur calcule le cas (b)", msg is None and r_de(e).ame)
    refus = ((["v", "g_M2"], "0", "γ_M2 = 0"), (["v", "V_Ed"], "", "V_Ed vide"), (["v", "rec_S"], "-20", "recoupe négative"),
             (["v", "j_lat"], "-2", "jeu négatif"), (["v", "appui"], "Semelle", "cas d'appui inconnu"),
             (["v", "prof_P"], "HEB 999", "profilé inconnu"))
    if cle == "doubles_cornieres_poteau":
        refus += ((["v", "n1S_u"], "2,4", "rangées 2,4"), (["v", "n1P_u"], "0", "rangées 0"))
    else:
        refus += ((["v", "n1_u"], "2,4", "rangées 2,4"), (["v", "n1_u"], "0", "rangées 0"))
    for chemin, v, quoi in refus:
        avant = copy.deepcopy(e)
        msg = A.modifier(e, chemin, v)
        chk(f"{cle} : refus « {quoi} » — message, assemblage intact", bool(msg) and e == avant, msg)
# le défaut lui-même, dans le moteur poutre–poteau
EPA, MPA = A.CONFIGS["plat_ame_poteau"].E, A.CONFIGS["plat_ame_poteau"].calculer
u0 = EPA.defaults()
chk("le défaut existe dans le moteur poutre–poteau : rangées 2,4 → 2 ; γ_M2 = 0 → boulons à 0 % ; V_Ed vide → 0",
    MPA(dict(u0, n1_u=2.4)).n_1 == 2 and next(c for c in MPA(dict(u0, g_M2=0)).checks if c.key == "bv").eta == 0
    and MPA(dict(u0, V_Ed=None)).eta_max < MPA(u0).eta_max)
# changer de configuration
e = A.nouvel_etat()                                 # plat d'âme poutre–poutre : grugeage 50 mm, HEA 400 porteuse
A.modifier(e, ["v", "prof_S"], "IPE 450")
A.modifier(e, ["v", "V_Ed"], "180")
A.modifier(e, ["v", "id_rep"], "PP4")
A.modifier(e, ["type"], "plat_ame_poteau")
u = A._actif(e)["values"]
d = A.CONFIGS["plat_ame_poteau"].E.defaults()
chk("poutre–poutre → poutre–poteau : l'effort, le boulon et le repère suivent ; les pièces (la porteuse, une poutre ; "
    "la poutre portée et son grugeage) repartent des défauts de la famille — l'assemblage neuf est conforme",
    u["V_Ed"] == 180.0 and u["id_rep"] == "PP4" and u["prof_P"] == d["prof_P"] and u["prof_S"] == d["prof_S"]
    and u["d_nt"] == float(d["d_nt"]) and u["l_n"] == float(d["l_n"]) and r_de(e).verified, (u["prof_P"], u["prof_S"], u["d_nt"]))
# ce que le « garder la poutre portée » donnait : la poutre par défaut des poutre–poutre sous la pièce des poutre–poteau
R_hea = MPA(dict(d, prof_S="HEA 300"))
chk("la raison : un HEA 300 (poutre portée des poutre–poutre) sous le plat par défaut des poutre–poteau est « Impossible »",
    not R_hea.verified and any(a.block for a in R_hea.alerts), [a.id for a in R_hea.alerts])
A.modifier(e, ["v", "appui"], AME)
A.modifier(e, ["v", "j_lat"], "8")
A.modifier(e, ["v", "N_Ed"], "40")
A.modifier(e, ["v", "nat_N"], A.CONFIGS["plat_ame_poteau"].E.NAT_TYING)
A.modifier(e, ["type"], "platine_about_poteau")
u = A._actif(e)["values"]
chk("plat d'âme → platine d'about sur poteau : le cas d'appui, le jeu latéral, la traction et sa nature suivent",
    u["appui"] == AME and u["j_lat"] == 8.0 and u["N_Ed"] == 40.0 and u["nat_N"] == A.CONFIGS["platine_about_poteau"].E.NAT_TYING
    and r_de(e).ame, (u["appui"], u["j_lat"], u["nat_N"]))
A.modifier(e, ["type"], "platine_about")
dpb = A.CONFIGS["platine_about"].E.defaults()
chk("poutre–poteau → poutre–poutre : le poteau ne devient pas la poutre principale, la pièce repart des défauts ; "
    "l'effort suit", A._actif(e)["values"]["prof_P"] == dpb["prof_P"] and A._actif(e)["values"]["prof_S"] == dpb["prof_S"]
    and A._actif(e)["values"]["N_Ed"] == 40.0 and r_de(e) is not None)

# =====================================================================================================================
print("\n=== 7. Les sélecteurs et l'échappement ===")
RX_PK = re.compile(r'<button class="pk-row( cur)?( opt)?" data-choix="([^"]+)"[^>]*>.*?<span class="chip (\w)">([^<]+)</span></button>')
for cle in POTEAU:
    cfg = A.CONFIGS[cle]
    e = etat_de(cle)
    u = A._actif(e)["values"]
    h = A.selecteur(e, "prof_P")
    rows = RX_PK.findall(h)
    heb = sorted([p for p in DB["profils"] if p["n"].startswith("HEB ")], key=lambda p: p["h"])
    attendu = []
    for p in heb:
        R = cfg.calculer(A.a_calculer(cfg, dict(u, prof_P=p["n"])))
        attendu.append((p["n"], f"{C.pc(C.pct(min(R.eta_max, 9.99)))}{C.NBSP}%", R.verified))
    opt = next((n for n, _t, ok in attendu if ok), None)
    chk(f"{cle} : sélecteur du poteau (HEB, triés par hauteur), taux recalculés, ✓ sur {u['prof_P']}, ★ sur le plus petit qui tient",
        "Poteau · profilé" in h and [(n, t) for _c, _o, n, _n, t in rows] == [(n, t) for n, t, _ in attendu]
        and [n for c, _o, n, *_ in rows if c] == [u["prof_P"]] and [n for _c, o, n, *_ in rows if o] == ([opt] if opt else []), rows[:3])
    A.choisir_section(e, "prof_P", "HEA 280")
    chk(f"{cle} : choisir HEA 280 → prof_P", A._actif(e)["values"]["prof_P"] == "HEA 280")
    chk(f"{cle} : sélecteur de la poutre portée", "Poutre portée · profilé" in A.selecteur(e, "prof_S", ["IPE"]))
    A.choisir_section(e, "boulon", "M24")
    chk(f"{cle} : choisir M24 → boulon_u", A._actif(e)["values"]["boulon_u"] == "M24")
    e = etat_de(cle, {"mode_calc": cfg.E.MODE_PREDIM, "appui": AME})
    u = A._actif(e)["values"]
    R = r_de(e)
    sols = [x for x in R.pd.rows if x.geom and x.prop]
    rows = RX_PK.findall(A.selecteur(e, "solution"))
    chk(f"{cle} : sélecteur des solutions (âme) — chaque combinaison possible, ★ sur la proposée",
        len(rows) == len(sols) and [n for _c, o, n, *_ in rows if o] == [A._nom_solution(cfg, R, R.pd.pick)], (len(rows), len(sols)))
    r2 = sols[len(sols) // 2]
    attendu = A.valeurs_normales(cfg, cfg.predim["apply"](dict(u), r2))
    A.choisir_section(e, "solution", A._nom_solution(cfg, R, r2))
    chk(f"{cle} : choisir une solution = moteur.apply_solution (repasse en vérification, garde l'âme)",
        A._actif(e)["values"] == attendu and attendu["mode_calc"] == cfg.E.MODE_VERIF and attendu["appui"] == AME)
e = etat_de("doubles_cornieres_poteau")
rows = RX_PK.findall(A.selecteur(e, "corniere"))
chk("cornières sur poteau : sélecteur des cornières EN 10056-1, ✓ sur la retenue",
    len(rows) == len(DB["cornieres"]) and [n for c, _o, n, *_ in rows if c] == [A._actif(e)["values"]["corn_u"]])
piege = '<img src=x onerror=alert(1)>"&'
for cle in POTEAU:
    e = etat_de(cle, {"appui": AME})
    for k in ("id_rep", "id_red", "id_projet", "id_date"):
        A.modifier(e, ["v", k], piege)
    A.modifier(e, ["visserie"], piege)
    r = A.resultat(e)
    frag = "".join(r[k] for k in ("resultat", "volets", "phrase", "donnees", "elements", "statut_html")) + "".join(v["svg"] for v in r["vues"])
    chk(f"{cle} : un texte saisi n'entre jamais brut dans un fragment HTML", "<img" not in frag and frag.count("&lt;img src=x onerror=alert(1)&gt;") >= 3)

# =====================================================================================================================
print("\n=== 8. Mutations : les contrôles rougissent ===")


def rejoue_parametres(cle, cas):
    ER = ecran(cle, "ecran_resultats")
    cfg = A.CONFIGS[cle]
    n = 0
    for _nom, src in cas:
        el = A.normaliser({"assemblages": [{"type": cle, "values": src}]})["assemblages"][0]
        R, vides, err = A.calculer(cfg, el)
        if R is None or vides:
            continue
        with faux_st(ER) as fs:
            ER._parametres_retenus(R)
        n += [t for t in fs.tables if t[1] == ("Paramètre", "Valeur")][0][2] != [[str(a), str(b)] for a, b in cfg.parametres(R)]
    return n


p0 = A.CONFIGS["platine_about_poteau"].parametres
A.CONFIGS["platine_about_poteau"].parametres = lambda R: [x for x in p0(R) if "Tronçon en T de l'AILE" not in x[0]]
try:
    n = rejoue_parametres("platine_about_poteau", CAS2["platine_about_poteau"])
finally:
    A.CONFIGS["platine_about_poteau"].parametres = p0
chk(f"mutation « paramètres retenus sans le tronçon en T de l'aile » : la copie nommée rougit ({n} résultats)", n > 0)
p0 = A.CONFIGS["doubles_cornieres_poteau"].parametres
A.CONFIGS["doubles_cornieres_poteau"].parametres = lambda R: [[a, b.replace(" mm (Tab. 6.4)", " mm")] for a, b in p0(R)]
try:
    n = rejoue_parametres("doubles_cornieres_poteau", CAS2["doubles_cornieres_poteau"])
finally:
    A.CONFIGS["doubles_cornieres_poteau"].parametres = p0
chk(f"mutation « une référence retirée d'un paramètre » : la copie nommée rougit ({n} résultats)", n > 0)
mot0 = A.CONFIGS["plat_ame_poteau"].calculer
A.CONFIGS["plat_ame_poteau"].calculer = lambda u: mot0(dict(u, V_Ed=(u.get("V_Ed") or 0) * 1.0000001))
try:
    n = len(comparer_parite("plat_ame_poteau", CAS["plat_ame_poteau"][:15]))
finally:
    A.CONFIGS["plat_ame_poteau"].calculer = mot0
chk(f"mutation « V_Ed × 1,0000001 » : l'empreinte au bit près rougit ({n} jeux)", n > 0)
so0 = A.CONFIGS["plat_ame_poteau"].sans_objet
A.CONFIGS["plat_ame_poteau"].sans_objet = lambda R: []
try:
    R = r_de(etat_de("plat_ame_poteau"))
    rouge = C.esc(PT.PERIMETRE_AILE_PA) not in A.vue_methode(A.CONFIGS["plat_ame_poteau"], R)
finally:
    A.CONFIGS["plat_ame_poteau"].sans_objet = so0
chk("mutation « sans objet de l'aile oublié » : le contrôle de la vue Méthode rougit", rouge)
co0 = A.CONFIGS["platine_about_poteau"].predim["colonnes"]
A.CONFIGS["platine_about_poteau"].predim["colonnes"] = lambda R: [(h, f) for h, f in co0(R)][:-1] + [("η poteau", lambda r: pct_pdp(r))]


def pct_pdp(r):
    from acier.formats import pct
    return pct(r.e.pdP)                                   # l'écran prend max(p. diam., cisaillement local) dans l'âme


try:
    e = etat_de("platine_about_poteau", {"mode_calc": A.CONFIGS["platine_about_poteau"].E.MODE_PREDIM, "appui": AME, "V_Ed": 400})
    R = r_de(e)
    ER = ecran("platine_about_poteau", "ecran_resultats")
    with faux_st(ER) as fs:
        ER.onglet_predim(R, A._actif(e)["values"], lambda u: None)
    cellules = [t.strip("*") for t in fs.textes if isinstance(t, str)]
    f = A.CONFIGS["platine_about_poteau"].predim["colonnes"](R)[-1][1]
    rouge = any(f(r) not in cellules for r in R.pd.rows if r.geom)
finally:
    A.CONFIGS["platine_about_poteau"].predim["colonnes"] = co0
chk("mutation « η poteau sans le cisaillement local de l'âme » : le contrôle des colonnes rougit", rouge)

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

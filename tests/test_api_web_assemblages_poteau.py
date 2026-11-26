# -*- coding: utf-8 -*-
"""La famille POUTRE–POTEAU du module « Assemblages métalliques » du site (api_web/assemblages_poteau.py) : plat
d'âme soudé, doubles cornières, platine d'about partielle — sur l'aile ou dans l'âme du poteau. Ce que le site calcule
et affiche est ce que calcule et affiche l'écran Streamlit ; chaque contrôle rougit si l'on retire ce qu'il protège.

  1. Les trois configurations : enregistrées avec LEUR moteur (acier/assemblages/poutre_poteau/…), rubriques pensées
     (Assemblage et le cas d'appui en tête · Poteau et poutre, avec la position et le grugeage · la pièce · Boulons ·
     Efforts ELU · Options), chaque champ que l'écran Streamlit montre est dans le cadre Données (entrees.champ_visible),
     aile et âme — sans la bascule « Mode », avec les règles du prédimensionnement (Options).
  2. Les copies NOMMÉES de l'écran Streamlit, rejouées avec un faux « st » sur des dizaines de résultats (aile, âme,
     traction, tying, cornières soudées) : PERIMETRE_AILE, « paramètres retenus », ce qui est sans objet et non
     revérifié (repli Périmètre → le pied de chaque fenêtre de calcul), la règle du Tableau 3.3, la légende de la
     matrice des ailes, les tableaux de répartition par boulon.
  3. La parité du calcul (fichier → saisies → moteur), chemin de l'écran Streamlit et chemin du site, au bit près,
     sur des jeux qui couvrent les deux cas d'appui, la traction et le tying, les cornières soudées, les profilés
     personnalisés vides, et des notes enregistrées en mode prédimensionnement (le site les relit en vérification,
     la proposition du moteur recopiée — ce que fait « Appliquer » dans l'écran Streamlit).
  4. L'écran Streamlit LUI-MÊME (AppTest) : les trois écrans poutre–poteau (fpc_, dcc_, pac_), des jeux chargés dans
     leur état comme le fait « Ouvrir » ; leur résultat a l'empreinte de celui du site.
  5. Ce que le site affiche = le moteur : verdict, une ligne par pièce, la gouvernante, E_d / R_d dans chaque fenêtre,
     détail des formules et ligne η, alertes, les trois dessins (élévation, plan, coupe A-A) au caractère près — la
     toile ne porte QU'EUX —, les limites de methode.limites(ame) au pied des fenêtres, le bloc des solutions, matrice
     des ailes, répartition par boulon, hypothèses (« poteau », « poutre portée »).
  6. La saisie : cas d'appui, refus (rangées 2,4, γ = 0, V vide, recoupe négative…), configuration changée (la pièce
     porteuse et le grugeage d'une poutre ne suivent pas sur un poteau ; le cas d'appui suit dans la famille).
  7. Les sélecteurs (poteau, poutre portée, boulon, cornière, solutions VÉRIFIÉES) et l'échappement.
  8. Mutations : abîmer une copie nommée, le calcul de 10⁻⁷, le périmètre des fenêtres, une proposition qui change le
     cas d'appui fait rougir.

(25/11/2026, revue de l'écran demandée par le propriétaire : la toile ne garde que les dessins ; les onglets Prédim,
Note, Méthode et Validation et le MODE prédimensionnement sont retirés ; les propositions, vérifiées par le calcul
complet, s'ouvrent du résultat — leur suite : tests/test_assemblages_predim.py. Le harnais SUIT.)

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
import math
import re
import sys
import types

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



def eta_tete(R):
    """Le taux de TÊTE attendu : le η max du moteur (celui de l'écran Streamlit), porté au rapport de la pire pince
    NON conforme quand il est plus grand (07/11/2026 : le moteur qui trouve la géométrie non valide ne calcule plus ce
    qui en dépend, et la tête montrait un taux qui minimise). « Vérifié sous réserve » ne s'écrit plus que sous un
    verdict vérifié, comme dans l'écran Streamlit."""
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


# les règles du prédimensionnement : l'écran Streamlit ne les montre que dans son mode ; le site, toujours (Options)
REGLES_PREDIM = {"eta_c", "pd_dmin", "pd_dmax", "k_e1", "k_p1", "k_e2"}
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
    # (25/11/2026) le site n'a plus de MODE prédimensionnement (les propositions s'ouvrent du résultat) : une note
    # enregistrée dans ce mode y est relue avec la proposition du moteur RECOPIÉE (``assemblages.sans_mode_predim``) —
    # ce que fait le bouton « Appliquer » de l'onglet Prédim de l'écran Streamlit (``moteur.apply_solution``)
    if cfg.E.mode_predim(u):
        try:
            vides_ = kit.dimensions_vides(D, u)
            pd = compute(kit.provisoire(D, u) if vides_ else u).get("pd")
        except Exception:  # noqa: BLE001
            pd = None
        if pd is not None and pd.get("pick") is not None and pd.pick.get("prop"):
            b2 = cfg.E.charger_json(cfg.predim["moteur"].apply_solution(u, pd.pick))
            u = {k: kit.valeur_widget(D, k, b2.get(k)) for k in u}
        else:
            u = dict(u, mode_calc=cfg.E.MODE_VERIF)
    vides = kit.dimensions_vides(D, u)
    try:
        R = compute(kit.provisoire(D, u) if vides else u)
        err = None
    except Exception as e:  # noqa: BLE001
        R, err = None, type(e).__name__
    return u, sorted(vides), R, err


def en_predim(cle, src):
    """La note ``src`` est-elle enregistrée en mode prédimensionnement ? (L'écran Streamlit y reste ; le site l'en sort.)"""
    cfg = A.CONFIGS[cle]
    return bool(cfg.E.mode_predim(cfg.E.charger_json(src)))


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
    chk(f"{cle} : le moteur est celui de la configuration (entrees, compute, synthese, benchmark ; les limites de "
        "methode.limites ; le moteur des propositions) — ni texte de méthode ni export texte à l'écran",
        cfg.E is ecran(cle, "entrees") and cfg.calculer is ecran(cle, "moteur").compute and cfg.S is ecran(cle, "synthese")
        and cfg.bench is ecran(cle, "benchmark") and cfg.famille == "poutre_poteau" and cfg.predim["moteur"] is ecran(cle, "moteur")
        and all(cfg.non_reverifies(types.SimpleNamespace(ame=a)) == tuple(ecran(cle, "methode").limites(a)) for a in (False, True))
        and not hasattr(cfg, "methode") and not hasattr(cfg, "T"))
    vus_ko = []
    for nom, v in (("aile", {}), ("âme", {"appui": AME}), ("note enregistrée en prédim", {"mode_calc": cfg.E.MODE_PREDIM}),
                   ("tying", {"N_Ed": 50, "nat_N": cfg.E.NAT_TYING, "cat": "B"}),
                   ("perso", {"prof_P": PERSO, "prof_S": PERSO, "nu_P": PERSO, "trou": cfg.E.CHAMPS["trou"]["o"][-1]})):
        e = etat_de(cle, v)
        u = A._actif(e)["values"]
        don = A.resultat(e)["donnees"]
        # (établi compact : le repère est le NOM de l'élément, sur sa ligne — data-chemin="nom")
        lus = set(re.findall(r'data-chemin="v/([^"/]+)"', don)) | ({"id_rep"} if don.count('data-chemin="nom"') == 1 else set())
        attendus = {k for k in cfg.E.CHAMPS if cfg.E.champ_visible(u, k)}
        if "d0_u" in attendus and not (u.get("d0_u") or 0) > 0:
            attendus.discard("d0_u")                     # trou normal : la case « d0 imposé » le remplace
        # (25/11/2026) le site n'a plus la bascule « Mode » ; les règles du prédimensionnement, que l'écran Streamlit ne
        # montre que dans ce mode, sont toujours dans « Options » (elles règlent les propositions)
        attendus = (attendus - {"mode_calc"}) | REGLES_PREDIM
        if lus != attendus or u["mode_calc"] != cfg.E.MODE_VERIF:
            vus_ko.append((nom, sorted(attendus - lus), sorted(lus - attendus), u["mode_calc"]))
    chk(f"{cle} : chaque champ que l'écran Streamlit montre est dans le cadre Données, et seulement lui (aile, âme, tying, "
        "personnalisé, note enregistrée en prédimensionnement) — sans la bascule « Mode », avec les règles du "
        "prédimensionnement", not vus_ko, vus_ko)
    don = A.resultat(etat_de(cle))["donnees"]
    titres = re.findall(r'<span class="tt">([^<]+)</span>', don)
    rub = re.findall(r'<div class="sec( open)?" data-sec="([^"]+)"( data-secondaire="1")?>', don)
    chk(f"{cle} : rubriques pensées — Assemblage · Poteau et poutre · la pièce · Boulons · Efforts ELU · Options "
        f"({len(titres)}) ; la première seule ouverte, « Options » seule secondaire",
        titres[:2] == ["Assemblage", "Poteau et poutre"] and titres[3:] == ["Boulons", "Efforts ELU", "Options"] and len(titres) == 6
        and [k for _o, k, _s in rub] == ["as", "pt", "cn" if "cornieres" in cle else "pl", "bo", "ef", "op"]
        and [k for o, k, _s in rub if o] == ["as"] and [k for _o, k, s_ in rub if s_] == ["op"], (titres, rub))
    tete = don.split('data-sec="pt"')[0]
    chk(f"{cle} : le cas d'appui est en tête (section Assemblage), en segments « Sur l'aile / Dans l'âme »",
        'data-chemin="v/appui"' in tete and tete.count('data-chemin="v/appui"') == 2 and 'data-valeur="Âme du poteau">Dans l&#x27;âme<' in tete
        and 'select data-chemin="type"' in tete)
    e = etat_de(cle, {"appui": AME})
    don = A.resultat(e)["donnees"]
    pieces = don.split('data-sec="pt"')[1].split('data-sec="')[0]
    chk(f"{cle} : « Poteau et poutre » porte le poteau, la poutre portée, sa position — en cas (b), la recoupe des semelles "
        "et le jeu latéral — et son GRUGEAGE (plus de rubrique à part)",
        all(f'data-chemin="v/{k}"' in pieces for k in ("prof_P", "nu_P", "prof_S", "nu_S", "rec_S", "j_lat", "d_nt", "d_nb", "l_n"))
        and "Grugeage" in pieces and 'data-sec="gr"' not in don and 'data-sec="po"' not in don)
    options = don.split('data-sec="op"')[1]
    chk(f"{cle} : « Options » porte les coefficients, les règles du prédimensionnement, la note (taux affichés ou non), le "
        "cartouche et les cotes des dessins",
        all(f'data-chemin="v/{k}"' in options for k in sorted(REGLES_PREDIM) + ["g_M0", "g_M2", "id_projet", "id_red"])
        and 'data-chemin="note_taux"' in options and C.esc(A.LIBELLE_NOTE_TAUX) in options and 'data-chemin="cotation"' in options
        and 'data-chemin="v/mode_calc"' not in don, options[:200])
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
        and "mode_calc" not in cfg.affichage and cfg.dimensions == es.D.DIMENSIONS)
chk("cornières : options affichées de la grande aile = les options du moteur",
    set(A.CONFIGS["doubles_cornieres_poteau"].affichage["orient"]) == set(A.CONFIGS["doubles_cornieres_poteau"].E.CHAMPS["orient"]["o"]))

CAS2 = {cle: jeux(cle, 22, graine=2) for cle in POTEAU}
for cle in POTEAU:
    ER = ecran(cle, "ecran_resultats")
    cfg = A.CONFIGS[cle]
    e_par, e_per, e_33, e_tab, e_leg = [], [], [], [], []
    n_par = n_per = n_ame = n_aile = 0
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
        # (25/11/2026) l'onglet « Méthode » de la toile, qui portait ce périmètre, est retiré : il est écrit au pied de
        # CHAQUE fenêtre de calcul (``assemblages.perimetre``)
        vm = A.perimetre(cfg_r, R)
        n_per += 1
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
    chk(f"{cle} : les jeux couvrent l'aile ({n_aile}) et l'âme ({n_ame})", n_aile >= 5 and n_ame >= 5)
    chk(f"{cle} : « Paramètres retenus » = ce qu'écrit ecran_resultats._parametres_retenus ({n_par} résultats)",
        n_par >= 20 and not e_par, e_par[:2])
    chk(f"{cle} : Périmètre — non revérifiés (methode.limites) et sans objet (dont l'aile du poteau) = le repli de "
        f"l'écran, dits au pied des fenêtres de calcul ({n_per} résultats)", n_per >= 20 and not e_per, e_per[:2])
    chk(f"{cle} : la règle du Tableau 3.3 de la fenêtre Géométrie = la légende de l'écran", not e_33, e_33[:2])
    chk(f"{cle} : répartition élastique par boulon = les tableaux de l'onglet Vérifications", not e_tab, e_tab[:2])
    if cle == "doubles_cornieres_poteau":
        chk("cornières : la légende de la matrice des ailes = celle de l'écran (aile A sur le POTEAU)", not e_leg, e_leg[:3])

# (25/11/2026 : l'onglet « Validation » de la toile est retiré — les bancs de comparaison sont rejoués par les suites du
# dépôt, à chaque modification : tests/test_*_poteau*.py ; le contrôle des décimales de la vue n'a plus d'objet)

# =====================================================================================================================
print("\n=== 3. La parité du calcul : fichier → saisies → moteur (écran Streamlit / site) ===")
logging.disable(logging.CRITICAL)
CAS = {}
total = 0
for cle in POTEAU:
    CAS[cle] = jeux(cle, TIRAGES)
    diff = comparer_parite(cle, CAS[cle])
    total += len(CAS[cle])
    # « prédim » : les notes ENREGISTRÉES en mode prédimensionnement (le site les relit avec leur proposition recopiée) ;
    # « en mode prédim » : les calculs du site restés dans ce mode — il ne doit plus y en avoir (25/11/2026)
    etats = {"vérifié": 0, "non vérifié": 0, "à compléter": 0, "erreur": 0, "prédim": 0, "en mode prédim": 0, "âme": 0, "traction": 0}
    for _nom, src in CAS[cle]:
        u, v, R, err = chemin_site(cle, src)
        etats["prédim"] += en_predim(cle, src)
        if err:
            etats["erreur"] += 1
        elif v:
            etats["à compléter"] += 1
        elif R is not None:
            etats["vérifié" if R.verified else "non vérifié"] += 1
            etats["en mode prédim"] += bool(R.get("pred"))
            etats["âme"] += bool(R.ame)
            etats["traction"] += bool((C.lire_nombre(u.get("N_Ed")) or 0) > 0)
    chk(f"{cle} : {len(CAS[cle])} jeux, même empreinte au bit près ({etats})", not diff, diff[:3])
    chk(f"{cle} : les jeux couvrent vérifié, non vérifié, à compléter, âme, traction et des notes enregistrées en mode "
        "prédimensionnement — que le site relit en vérification (aucun de ses calculs ne reste dans ce mode)",
        all(etats[k] > 0 for k in ("vérifié", "non vérifié", "à compléter", "prédim", "âme", "traction"))
        and etats["en mode prédim"] == 0, etats)
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
        # (sans les notes en mode prédimensionnement : l'écran Streamlit y reste, le site les en sort — leur parité est
        #  celle du bouton « Appliquer », contrôlée plus haut par ``chemin_streamlit``)
        bons = [c for c in CAS[cle] if not en_predim(cle, c[1]) and not chemin_site(cle, c[1])[1] and chemin_site(cle, c[1])[3] is None]
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
# (établi compact, 12/11/2026 : le taux est au bout de la PREMIÈRE ligne ; une ligne d'information y dit « hors verdict » en mots)
RX_EL = re.compile(r'<div class="el vc( gov)?" data-calcul="([^"]+)"[^>]*><div class="l1"><b>([^<]+)</b>.*?'
                   r'<span class="(?:pct|et-mot) (\w)"[^>]*>([^<]+)</span></div><div class="l2">')
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
        if f"{C.pc(A.taux(eta_tete(R)))}{C.NBSP}%" not in r["resultat"].split('class="verdict"')[1][:300]:
            defauts.append((nom, "verdict"))
        gov = [x for x in lignes if x[0]]
        att_gov = "Géométrie" if geometrie_gouverne(R) else (R.gov.grp if R.gov is not None else None)
        if att_gov is not None and (len(gov) != 1 or gov[0][2] != C.esc(att_gov)):
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
        if r["resultat"].count('class="msg') != len(R.alerts) + bool(R.get("reserve") and R.verified) or R.get("pred"):
            defauts.append((nom, "alertes", len(R.alerts)))
        opt = M.options_ecran(R, el["cotation"], A._hl(R))
        opt.interactive, opt.poignees, opt.editables, opt.locked = False, False, el["cotation"] >= 1, None
        vues = {v["cle"]: v for v in r["vues"]}
        for k, f in (("elevation", M.elevation), ("plan", M.plan), ("coupe", M.coupe_aa)):
            if vues[k]["svg"] != f(R, opt).svg(identifiant=f"asm_{k}"):
                defauts.append((nom, "dessin", k))
        if list(vues) != ["elevation", "plan", "coupe"]:
            defauts.append((nom, "vues", list(vues)))
        # les limites de la méthode (methode.limites(ame)) : au pied de chaque fenêtre de calcul
        lim = ecran(cle, "methode").limites(R.ame)
        pied = "<b>Non revérifiés ici</b> : " + C.esc(" ; ".join(lim)) + "."
        if r["volets"].count(pied) != len(A._groupes(A.pour(cfg, R), R)) or not lim:
            defauts.append((nom, "limites de la méthode", r["volets"].count(pied)))
        hyp = r["volets"].split('class="ck-hyp">')[1].split("</div>")[0]
        # un acier « Personnalisé » sans f_y (fichier) vaut f_y = 0 dans le moteur : il n'est pas dit (η = 10⁹ le dit)
        if ((R.fy_P and "poteau f<sub>y</sub>" not in hyp) or (R.fy_S and "poutre portée f<sub>y</sub>" not in hyp)
                or "principale" in hyp or "secondaire" in hyp):
            defauts.append((nom, "hypothèses", hyp[:120]))
    chk(f"{cle} : {n} résultats — verdict, lignes par pièce, gouvernante, cas d'appui, E_d / R_d et détail η de chaque "
        "vérification dans sa fenêtre, alertes, les trois dessins (et eux seuls), limites au pied des fenêtres, hypothèses = "
        "le moteur", n >= 25 and not defauts, defauts[:3])
    e = etat_de(cle)
    r = A.resultat(e)
    vues = {v["cle"]: v for v in r["vues"]}
    # (25/11/2026, le propriétaire : « dans la partie centrale, on devrait juste avoir élévation, plan, droite » — le
    #  harnais SUIT : il cherchait les onglets Prédim, Note, Méthode et Validation)
    chk(f"{cle} : la toile porte Élévation, Plan, Coupe A-A — ni Prédim, ni Note, ni Méthode, ni Validation",
        [(v["cle"], v["court"]) for v in r["vues"]] == [("elevation", "Élévation"), ("plan", "Plan"), ("coupe", "Coupe A-A")]
        and all(v["svg"].lstrip().startswith("<svg") for v in r["vues"]), [(v["cle"], v["court"]) for v in r["vues"]])
    niveaux = []
    for nv in (0, 1, 2):
        A.modifier(e, ["cotation"], nv)
        niveaux.append(len(A.resultat(e)["vues"][0]["svg"]))
    chk(f"{cle} : les trois niveaux de cotation changent l'élévation", niveaux[0] < niveaux[1] <= niveaux[2], niveaux)
    for appui in ({}, {"appui": AME}):
        e = etat_de(cle, dict({"mode_calc": cfg.E.MODE_PREDIM}, **appui))
        R = r_de(e)
        r = A.resultat(e)
        u0 = cfg.E.charger_json(dict({"mode_calc": cfg.E.MODE_PREDIM}, **appui))
        pick = cfg.calculer(u0).pd.pick
        chk(f"{cle} : une note enregistrée en mode prédimensionnement ({'âme' if appui else 'aile'}) est relue en VÉRIFICATION, "
            "la proposition du moteur recopiée (« Appliquer ») — ni bandeau de mode, ni fenêtre ni vue Prédim ; le bloc "
            "« Prédimensionnement » du résultat ouvre les solutions vérifiées",
            A._actif(e)["values"]["mode_calc"] == cfg.E.MODE_VERIF and not R.get("pred") and bool(R.ame) == bool(appui)
            and A._actif(e)["values"] == A.valeurs_normales(cfg, cfg.predim["moteur"].apply_solution(u0, pick))
            and "Mode prédimensionnement" not in r["resultat"] and 'id="ck-predim"' not in r["volets"]
            and [v["cle"] for v in r["vues"]] == ["elevation", "plan", "coupe"]
            and re.search(r'<div class="el vc fixe"><div class="l1"><b>Solutions</b><button class="prof" data-pk="solution"[^>]*>'
                          r'<span>Proposer</span>', r["resultat"]) is not None and C.esc(A.PROPOSITION_GARDE) in r["resultat"],
            (A._actif(e)["values"]["mode_calc"], R.get("pred")))
    e = etat_de(cle)
    msg = A.modifier(e, ["v", "mode_calc"], cfg.E.MODE_PREDIM)
    chk(f"{cle} : la bascule vers le mode prédimensionnement est refusée (le mode n'existe plus à l'écran)",
        bool(msg) and A._actif(e)["values"]["mode_calc"] == cfg.E.MODE_VERIF, msg)
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
chk("plat d'âme sur l'aile : pas de ligne « Poteau porteur » (aucune vérification active), et le pied des fenêtres de calcul "
    "dit pourquoi", "<b>Poteau porteur</b>" not in r["resultat"] and C.esc(PT.PERIMETRE_AILE_PA) in r["volets"])
A.modifier(e, ["v", "appui"], AME)
r = A.resultat(e)
chk("plat d'âme dans l'âme : la ligne « Poteau porteur » paraît (cisaillement local), sans le texte de l'aile",
    "<b>Poteau porteur</b>" in r["resultat"] and C.esc(PT.PERIMETRE_AILE_PA) not in r["volets"])
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
GARDES = ("appui", "V_Ed", "N_Ed", "nat_N", "prof_P", "nu_P", "prof_S", "nu_S", "z_C", "rec_S", "j_lat", "d_nt", "d_nb", "l_n")


def fautes_solutions(cle):
    """Le sélecteur des solutions d'une attache DANS L'ÂME du poteau, et le clic qui recopie la dernière : ce qui cloche."""
    cfg = A.CONFIGS[cle]
    e = etat_de(cle, {"appui": AME})
    u = dict(A._actif(e)["values"])
    X = A.propositions(cfg, u)
    rows = RX_PK.findall(A.selecteur(e, "solution"))
    out = []
    if not (len(rows) == len(X.solutions) >= 1):
        return [("lignes", len(rows), len(X.solutions))]
    if [n for _c, o, n, *_ in rows if o] != [html.escape(cfg.predim["nom"](X.solutions[0].u))]:
        out.append(("★", [n for _c, o, n, *_ in rows if o]))
    if [t for *_x, t in rows] != [f"{C.pc(C.pct(min(A.eta_R(s.R), 9.99)))}{C.NBSP}%" for s in X.solutions]:
        out.append(("taux", [t for *_x, t in rows]))
    s2 = X.solutions[-1]
    A.choisir_section(e, "solution", cfg.predim["nom"](s2.u))
    v2 = A._actif(e)["values"]
    R2 = r_de(e)
    if not (R2.verified and R2.ame and abs(R2.eta_max - s2.eta) < 1e-9):
        out.append(("recopie", R2.statut, R2.eta_max, s2.eta))
    if v2["mode_calc"] != cfg.E.MODE_VERIF or [k for k in GARDES if k in u and v2[k] != u[k]]:
        out.append(("gardés", [k for k in GARDES if k in u and v2[k] != u[k]], v2["mode_calc"]))
    return out

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
        attendu.append((p["n"], f"{C.pc(C.pct(min(eta_tete(R), 9.99)))}{C.NBSP}%", R.verified))
    opt = next((n for n, _t, ok in attendu if ok), None)
    chk(f"{cle} : sélecteur du poteau (HEB, triés par hauteur), taux recalculés, ✓ sur {u['prof_P']}, ★ sur le plus petit qui tient",
        "Poteau · profilé" in h and [(n, t) for _c, _o, n, _n, t in rows] == [(n, t) for n, t, _ in attendu]
        and [n for c, _o, n, *_ in rows if c] == [u["prof_P"]] and [n for _c, o, n, *_ in rows if o] == ([opt] if opt else []), rows[:3])
    A.choisir_section(e, "prof_P", "HEA 280")
    chk(f"{cle} : choisir HEA 280 → prof_P", A._actif(e)["values"]["prof_P"] == "HEA 280")
    chk(f"{cle} : sélecteur de la poutre portée", "Poutre portée · profilé" in A.selecteur(e, "prof_S", ["IPE"]))
    A.choisir_section(e, "boulon", "M24")
    chk(f"{cle} : choisir M24 → boulon_u", A._actif(e)["values"]["boulon_u"] == "M24")
    # les propositions du prédimensionnement (25/11/2026) : des dispositions VÉRIFIÉES par le calcul complet — leur
    # suite : tests/test_assemblages_predim.py ; ici, le sélecteur de l'écran et le clic qui recopie, DANS L'ÂME
    fautes = fautes_solutions(cle)
    chk(f"{cle} : sélecteur des solutions (âme) — les propositions vérifiées, ★ sur la plus simple ; en choisir une la "
        "recopie : l'assemblage est vérifié, au taux annoncé ; le cas d'appui, le poteau, la poutre, les efforts et les niveaux "
        "gardés ; toujours en vérification", not fautes, fautes)
    try:
        A.choisir_section(etat_de(cle, {"appui": AME}), "solution", "solution inconnue")
        chk(f"{cle} : une solution inconnue est refusée", False)
    except ValueError:
        chk(f"{cle} : une solution inconnue est refusée", True)
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
    rouge = C.esc(PT.PERIMETRE_AILE_PA) not in A.perimetre(A.CONFIGS["plat_ame_poteau"], R)
finally:
    A.CONFIGS["plat_ame_poteau"].sans_objet = so0
chk("mutation « sans objet de l'aile oublié » : le contrôle du périmètre des fenêtres rougit", rouge)
pe0 = A.perimetre
A.perimetre = lambda cfg, R: ""
try:
    r = A.resultat(etat_de("platine_about_poteau", {"appui": AME}))
    lim = ecran("platine_about_poteau", "methode").limites(True)
    rouge = "<b>Non revérifiés ici</b> : " + C.esc(" ; ".join(lim)) + "." not in r["volets"]
finally:
    A.perimetre = pe0
chk("mutation « le périmètre retiré des fenêtres de calcul » : le contrôle des limites de la méthode rougit", rouge)
ch0 = A.PR.chercher


def chercher_sur_l_aile(M, S, u, genre, **k):
    """Une recherche qui rendrait ses propositions avec le cas d'appui changé (sur l'aile)."""
    X = ch0(M, S, u, genre, **k)
    for s_ in X.solutions:
        s_.u["appui"] = A.CONFIGS["platine_about_poteau"].E.APPUI_AILE
    return X


A.PR.chercher = chercher_sur_l_aile
A._RECHERCHE.clear()
try:
    rouge = fautes_solutions("platine_about_poteau") != []
finally:
    A.PR.chercher = ch0
    A._RECHERCHE.clear()
chk("mutation « une proposition qui change le cas d'appui » : le contrôle des solutions rougit", rouge)
chk("… et sans la mutation, il reste vert", fautes_solutions("platine_about_poteau") == [])

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

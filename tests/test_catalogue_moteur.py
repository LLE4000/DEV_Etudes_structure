# -*- coding: utf-8 -*-
"""Moteur de la page « Catalogue des profilés » (acier/profiles/consultation.py) : la PARITÉ avec l'écran Streamlit
modules/tableau_profiles.py v1.0, qui reste la référence. Chaque contrôle rougit si l'on retire ce qu'il protège.

  1. Constantes : celles que le module extrait recopie de l'écran (version, nombre de filtres, dimensions proposées
     du profilé personnalisé, minimum des dimensions, comparaison « jusqu'à 5 », messages, aides) sont LUES dans le
     source de l'écran et comparées.
  2. Parité : l'écran RÉEL est rejoué par Streamlit (AppTest, sans navigateur) sur des dizaines d'états tirés au
     hasard (graine fixe) et sur des cas ciblés (préférences enregistrées, « Rétablir », « Toutes », « Aucune »,
     « Effacer les filtres », deux filtres sur la même propriété, min > max, profilés personnalisés possibles et
     impossibles, aux bornes exactes, le même nom deux fois). Après CHAQUE exécution, ce que l'écran montre est
     comparé à ce que le module calcule depuis les mêmes réglages :
       a) la liste (noms et ordre), les colonnes du tableau, CHAQUE cellule (valeur arrondie, absente) ;
       b) le compteur (texte exact), le message « Aucun profilé », le titre et sa légende ;
       c) l'état qui suit les clics : familles et colonnes remises dans leur ordre, fiche qui passe au premier
          profilé (et ne revient pas), comparaison qui perd les profilés sortis du tableau ;
       d) la fiche : le profilé dessiné (l'enregistrement entier), ses pastilles, la table de toutes ses
          propriétés (HTML de l'écran recomposé depuis le module, au caractère près), la légende « It : … » ;
       e) la comparaison (HTML recomposé, au caractère près) et son aide ;
       f) les deux fichiers téléchargés : le CSV (octet pour octet, BOM compris) et la fiche JSON (et son nom) ;
       g) le profilé personnalisé : l'enregistrement calculé (chaque grandeur, au bit près), son nom, sa place,
          les messages d'erreur de l'écran ;
       h) « Méthode et sources » : chaque phrase de l'écran recomposée depuis le module.
  3. Mutations : chaque fonction du module est faussée à son tour (vingt mutations) ; chacune doit faire rougir au
     moins une comparaison du point 2 — sinon la parité ne la protège pas.
  4. Pureté : le module (et le module du site qui l'appelle) se charge et calcule dans un sous-processus où
     sys.meta_path REFUSE streamlit, pandas, matplotlib, scipy, reportlab, PIL et fitz.

    /usr/bin/python3.12 tests/test_catalogue_moteur.py       (streamlit et pandas requis pour la parité)
"""
import copy
import gc
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from acier.profiles import catalogue as CAT  # noqa: E402
from acier.profiles import consultation as CO  # noqa: E402
from acier.profiles import familles as F  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{str(info)[:400]}]" if info and not cond else ""))


def lire(*chemin):
    with open(os.path.join(RACINE, *chemin), encoding="utf-8") as fh:
        return fh.read()


ECRAN = lire("modules", "tableau_profiles.py")

# ====================================================================== 1. les constantes, lues dans l'écran
print("=== 1. Constantes lues dans l'écran ===")
chk("VERSION de l'écran", re.search(r'^VERSION = "([^"]+)"', ECRAN, re.M).group(1) == CO.VERSION)
chk("N_FILTRES de l'écran", int(re.search(r"^N_FILTRES = (\d+)", ECRAN, re.M).group(1)) == CO.N_FILTRES)
defauts = eval(re.search(r"defauts = (\{[^}]+\})", ECRAN).group(1))  # noqa: S307 — un littéral de l'écran
chk("dimensions proposées du profilé personnalisé (defauts de _perso)", defauts == CO.DIMS_DEFAUT, defauts)
chk("minimum des dimensions (min_value du champ)", float(re.search(r"min_value=([\d.]+)", ECRAN).group(1)) == CO.DIM_MIN)
chk("comparaison « jusqu'à 5 » (max_selections)", int(re.search(r"max_selections=(\d+)", ECRAN).group(1)) == CO.MAX_COMPARES
    and f"jusqu'à {CO.MAX_COMPARES}" in ECRAN)
chk("message « Aucun profilé »", f'st.info("{CO.MESSAGE_VIDE}")' in ECRAN)
chk("aide « Séries courantes seulement »", f'help="{CO.AIDE_COURANTS}"' in ECRAN)
chk("aide des colonnes (deux littéraux de l'écran)", all(x.strip() in ECRAN for x in re.findall(r"[^.]+\.", CO.AIDE_COLONNES_ECRAN)))
chk("aide de la comparaison", f'st.caption("{CO.AIDE_COMPARER}")' in ECRAN)
chk("formes du profilé personnalisé : les familles « non standard », dans l'ordre du registre",
    CO.formes_perso() == ["PRS", "CAISSON", "TUBE", "PLAT", "ROND"] and "d.get(\"non_standard\")" in ECRAN)
chk("contrôles du profilé personnalisé : les deux de l'écran, textes compris",
    "h doit dépasser 2·tf" in ECRAN and "t doit être inférieure à la demi-dimension" in ECRAN
    and 'fam in ("PRS",) and 2 * dims["tf"] >= dims["h"]' in ECRAN)

# ====================================================================== 2. la parité, sur l'écran réel
print("\n=== 2. Parité avec l'écran réel (AppTest) ===")
import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from acier.profiles import dessin as DS  # noqa: E402

ENREG = {}
_dl, _svg = st.download_button, DS.svg_section


def _espion_dl(label, data=None, file_name=None, *a, **k):
    """Le téléchargement de l'écran : on garde ce qu'il enverrait (octets, nom du fichier)."""
    ENREG[label] = (data, file_name)
    return _dl(label, data, file_name, *a, **k)


def _espion_svg(famille, dims, taille=240, props=None, titre=""):
    """Le dessin de la fiche : on garde le profilé dessiné."""
    ENREG["svg"] = (famille, copy.deepcopy(dict(dims)), copy.deepcopy(props))
    return _svg(famille, dims, taille, props, titre)


st.download_button = _espion_dl
DS.svg_section = _espion_svg

SCRIPT = f"""
import sys
sys.path.insert(0, {RACINE!r})
from modules import tableau_profiles as T
T.show()
"""
PREFS_ORIG = CAT.CHEMIN_PREFS
TMP = tempfile.mkdtemp(prefix="cat_prefs_")


def prefs_temp(contenu=None) -> str:
    """Le fichier des préférences de l'écran, détourné vers un fichier temporaire (l'écran l'écrit)."""
    f = os.path.join(TMP, f"prefs_{random.random():.9f}.json")
    if contenu is not None:
        with open(f, "w", encoding="utf-8") as fh:
            json.dump(contenu, fh)
    CAT.CHEMIN_PREFS = f
    return f


def figer_prefs() -> str:
    """Une copie du fichier des préférences tel qu'il est maintenant (l'écran le réécrit ensuite)."""
    f = CAT.CHEMIN_PREFS + f".fige{len(OBS)}.json"
    with open(CAT.CHEMIN_PREFS, encoding="utf-8") as a, open(f, "w", encoding="utf-8") as b:
        b.write(a.read())
    return f


def courir(at):
    ENREG.clear()
    at.run()
    if at.exception:
        raise RuntimeError("exception de l'écran : " + str([str(e.value)[:300] for e in at.exception]))


def ouvrir():
    at = AppTest.from_string(SCRIPT, default_timeout=180)
    courir(at)
    return at


def widget(at, sorte, cle):
    return next(w for w in getattr(at, sorte) if w.key == cle)


def _f(v):
    """Une cellule du DataFrame : None pour une valeur absente (NaN), sinon le flottant."""
    if v is None:
        return None
    v = float(v)
    return None if math.isnan(v) else v


OBS = []          # chaque exécution observée : ce que l'écran montre + les réglages qui l'ont produit


def empreinte(x) -> str:
    """L'empreinte EXACTE d'un tableau (JSON, les flottants en repr qui se relit au bit près) ou d'octets : garder 300
    tableaux de 1 419 × 41 cellules coûterait des centaines de Mo."""
    b = x if isinstance(x, bytes) else json.dumps(x, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def observer(at, quoi, attendu=None):
    """Relève ce que l'écran montre après une exécution ; ``attendu`` porte ce que l'utilisateur a voulu (familles et
    colonnes cochées dans l'ordre des clics, fiche et comparaison avant l'exécution)."""
    ss = at.session_state
    etat = {k: copy.deepcopy(ss["prf_" + k]) for k in ("familles", "colonnes", "tri", "decroissant", "courants", "filtres",
                                                       "recherche", "fiche")}
    etat["compare"] = copy.deepcopy(ss["prf_compare"]) if "prf_compare" in ss else []
    md = [str(m.value) for m in at.markdown]
    o = {"quoi": quoi, "etat": etat, "persos": copy.deepcopy(ss["prf_persos"]), "attendu": attendu or {}, "md": md,
         "captions": [str(c.value) for c in at.caption], "infos": [str(i.value) for i in at.info],
         "erreurs": [str(e.value) for e in at.error], "expanders": [str(e.label) for e in at.expander],
         "svg": copy.deepcopy(ENREG.get("svg")), "json": ENREG.get("⬇️ Fiche (JSON)")}
    csv_ = ENREG.get("⬇️ CSV")
    o["csv"] = (empreinte(csv_[0]), csv_[1], csv_[0][:160]) if csv_ else None
    if len(at.dataframe):
        df = at.dataframe[0].value
        lignes = [{c: (r[c] if c in ("Profilé", "Famille") else _f(r[c])) for c in df.columns} for r in df.to_dict("records")]
        o["colonnes_df"], o["noms_df"], o["n_lignes"] = list(df.columns), [x["Profilé"] for x in lignes], len(lignes)
        o["lignes_df"], o["debut_df"] = empreinte(lignes), lignes[:3]
        o["iy_max"] = max((x.get("Iy (cm⁴)") or 0.0 for x in lignes), default=None)
        o["familles_df"] = sorted({x["Famille"] for x in lignes})
    else:
        o["colonnes_df"], o["lignes_df"], o["noms_df"], o["n_lignes"] = None, None, None, 0
    try:
        sb = widget(at, "selectbox", "prf_w_fiche")
        o["fiche_options"], o["fiche_valeur"] = list(sb.options), sb.value
    except StopIteration:
        o["fiche_options"], o["fiche_valeur"] = None, None
    OBS.append(o)
    return o


# ---------------------------------------------------------------------- ce que le module calcule, recomposé comme l'écran
def html_fiche(p):
    lig = []
    for g, props in CO.fiche(p):
        lig.append(f"<tr><td class='g' colspan='3'>{g.upper()}</td></tr>")
        lig += [f"<tr><td title=\"{aide}\">{lib}</td><td class='n'>{val}</td><td class='prf-note'>{u}</td></tr>"
                for _c, lib, val, u, aide in props]
    return "<table class='prf-tab'>" + "".join(lig) + "</table>"


def html_comparaison(profils, colonnes):
    tete = "".join(f"<th class='n'>{p['nom']}</th>" for p in profils)
    lig = [f"<tr><td>{ent}</td>" + "".join(f"<td class='n'>{x}</td>" for x in vals) + "</tr>" for _c, ent, vals in CO.comparaison(profils, colonnes)]
    return f"<table class='prf-tab'><tr><th>Propriété</th>{tete}</tr>" + "".join(lig) + "</table>"


def html_entete_fiche(p):
    puces = "".join(f"<span class='prf-chip'>{x}</span>" for x in CO.chips_fiche(p))
    return f"<div class='prf-fiche'><div class='t'>PROFILÉ</div><h3>{p['nom']}</h3>{puces}</div>"


def ecarts(o) -> list:
    """Ce que le module calcule depuis les réglages de l'écran, comparé à ce que l'écran a montré ; [] si tout est égal."""
    e, out = o["etat"], []
    a = o["attendu"]
    # c) l'état qui suit les clics
    if "familles" in a and e["familles"] != CO.familles_ordonnees(a["familles"]):
        out.append(("familles", e["familles"], CO.familles_ordonnees(a["familles"])))
    if "colonnes" in a and e["colonnes"] != CO.colonnes_ordonnees(a["colonnes"]):
        out.append(("colonnes", e["colonnes"], CO.colonnes_ordonnees(a["colonnes"])))
    if "reglages" in a:
        garde = CAT.CHEMIN_PREFS
        CAT.CHEMIN_PREFS = a["prefs"]            # le fichier des préférences tel qu'il était à l'ouverture (figé)
        try:
            attendus = getattr(CO, a["reglages"])()      # par son NOM : une mutation du module doit y porter
        finally:
            CAT.CHEMIN_PREFS = garde
        for k, kk in (("familles", "familles"), ("colonnes", "colonnes"), ("tri", "tri"), ("decroissant", "decroissant"),
                      ("courants", "courants"), ("filtres", "filtres")):
            if e[k] != attendus[kk]:
                out.append(("réglages " + k, e[k], attendus[kk]))
    lst = CO.liste(e["familles"], e["recherche"], e["filtres"], e["courants"], o["persos"], e["tri"], e["decroissant"])
    noms = [p["nom"] for p in lst]
    # a) la liste, les colonnes, les cellules
    if o["lignes_df"] is None:
        if noms:
            out.append(("tableau absent", len(noms)))
    else:
        if o["noms_df"] != noms:
            out.append(("liste", o["noms_df"][:6], noms[:6]))
        cols = CO.colonnes_visibles(lst, e["colonnes"])
        lignes = CO.lignes_tableau(lst, cols)
        if o["colonnes_df"] != (list(lignes[0].keys()) if lignes else []):
            out.append(("colonnes du tableau", o["colonnes_df"], list(lignes[0].keys()) if lignes else []))
        elif o["lignes_df"] != empreinte(lignes):
            out.append(("cellules", o["debut_df"][:1], lignes[:1]))
        # f) le CSV, octet pour octet
        txt = CO.texte_tableau(lignes, ";").encode("utf-8-sig")
        if o["csv"] is None or o["csv"][0] != empreinte(txt) or o["csv"][1] != "profiles.csv":
            out.append(("CSV", (o["csv"] or ("", "", b""))[2][:120], txt[:120]))
    # b) le compteur, le message vide, le titre
    n_persos = CO.n_persos_affiches(e["recherche"], e["filtres"], o["persos"])
    if CO.compteur(len(lst), e["familles"], n_persos) not in o["md"]:
        out.append(("compteur", next((m for m in o["md"] if m.startswith("**")), None), CO.compteur(len(lst), e["familles"], n_persos)))
    if (CO.MESSAGE_VIDE in o["infos"]) != (not lst):
        out.append(("message vide", o["infos"], len(lst)))
    if CO.AIDE_COLONNES_ECRAN not in o["captions"]:
        out.append(("aide des colonnes", [c for c in o["captions"] if c.startswith("Les colonnes")]))
    if CO.legende_page() not in o["captions"]:
        out.append(("légende du titre", o["captions"][:1], CO.legende_page()))
    # le titre de la section des colonnes est écrit AVANT que l'écran relise ses pastilles : il a une exécution de
    # retard quand on vient de cocher une colonne (défaut de l'écran, documenté ; le site n'en a pas)
    n_titre = len(a.get("colonnes_avant", e["colonnes"]))
    if f"Colonnes affichées — {n_titre} sur {len(CO.cles_colonnes())}" not in o["expanders"]:
        out.append(("titre des colonnes", [x for x in o["expanders"] if x.startswith("Colonnes")], n_titre))
    # c) la fiche et la comparaison qui suivent le tableau
    if "fiche_avant" in a and e["fiche"] != CO.fiche_retenue(a["fiche_avant"], noms):
        out.append(("fiche retenue", e["fiche"], CO.fiche_retenue(a["fiche_avant"], noms)))
    if "compare_avant" in a and e["compare"] != CO.compares_retenus(a["compare_avant"], noms):
        out.append(("comparaison retenue", e["compare"], CO.compares_retenus(a["compare_avant"], noms)))
    if noms and o["fiche_options"] != noms:
        out.append(("options de la fiche", (o["fiche_options"] or [])[:5], noms[:5]))
    # d) la fiche
    par = {p["nom"]: p for p in lst}
    p = par.get(e["fiche"])
    if noms:
        if p is None:
            out.append(("fiche hors de la liste", e["fiche"]))
        else:
            if o["svg"] is None or o["svg"][2] != p or o["svg"][0] != p["famille"] or o["svg"][1] != dict(p["dims"]):
                out.append(("profilé dessiné", (o["svg"] or ("", {}, {}))[2].get("nom") if o["svg"] else None, p["nom"]))
            if html_entete_fiche(p) not in o["md"]:
                out.append(("en-tête de la fiche", p["nom"]))
            if html_fiche(p) not in o["md"]:
                vu = next((m for m in o["md"] if m.startswith("<table class='prf-tab'><tr><td class='g'")), "")
                out.append(("table de la fiche", vu[:200], html_fiche(p)[:200]))
            leg_it = [c for c in o["captions"] if c.startswith("It : ")]
            if leg_it != ([f"It : {CO.methode_it(p)}."] if CO.methode_it(p) else []):
                out.append(("légende It", leg_it, CO.methode_it(p)))
            if o["json"] is None or o["json"][0] != CO.fiche_json(p).encode("utf-8") or o["json"][1] != CO.nom_fichier_fiche(p):
                out.append(("fiche JSON", (o["json"] or ("", ""))[1], CO.nom_fichier_fiche(p)))
    # e) la comparaison
    comp = [par[n] for n in e["compare"] if n in par]
    if len(e["compare"]) >= 2:
        h = html_comparaison(comp, CO.colonnes_comparaison(CO.colonnes_visibles(lst, e["colonnes"])))
        if h not in o["md"]:
            vu = next((m for m in o["md"] if m.startswith("<table class='prf-tab'><tr><th>Propriété")), "")
            out.append(("comparaison", vu[:200], h[:200]))
    elif CO.AIDE_COMPARER not in o["captions"]:
        out.append(("aide de la comparaison", o["captions"]))
    # g) les profilés personnalisés
    for q in o["persos"]:
        try:
            r = CO.calculer_perso(q["famille"], q["dims"])
        except ValueError as x:
            out.append(("perso refusé par le module", q["nom"], str(x)))
            continue
        if r != q or list(r) != list(q):
            cle = next((k for k in q if r.get(k) != q.get(k)), None)
            out.append(("profilé personnalisé", q["nom"], cle, q.get(cle), r.get(cle)))
    if "persos_attendus" in a:
        noms_p = [CO.nom_perso(f, d) for f, d in a["persos_attendus"]]
        if [q["nom"] for q in o["persos"]] != noms_p:
            out.append(("noms et ordre des personnalisés", [q["nom"] for q in o["persos"]], noms_p))
    if "erreur_perso" in a:
        fam, dims = a["erreur_perso"]
        try:
            CO.controle_ecran(fam, dims)
            msg = None
        except ValueError as x:
            msg = f"Dimensions impossibles : {x}"
        if o["erreurs"] != ([msg] if msg else []):
            out.append(("message du profilé personnalisé", o["erreurs"], msg))
    # h) Méthode et sources
    m = CO.methode_et_sources()
    phrases = [f"**Source unique** : `{m['fichier']}` (version {m['version']}, {m['date']}, {m['nombre']} profilés). Conventions : {m['conventions']}.",
               "**Dimensions** :\n" + "\n".join(f"- {s}" for s in m["sources"]),
               f"**Calcul** : {m['calcul']}. **Contrôle croisé** : {m['controle']}.",
               "**Définitions** :\n" + "\n".join(f"- **{lib}** — {t}" for _c, lib, t in m["definitions"]),
               "**Familles** :\n" + "\n".join(f"- **{f}** — {lib} · {n} · {d}" for f, lib, n, d in m["familles"])]
    manque = [x[:40] for x in phrases if x not in o["md"]]
    if manque:
        out.append(("méthode et sources", manque))
    return out


# ---------------------------------------------------------------------- les gestes de l'utilisateur
CLES = CO.cles_colonnes()
FREQUENTES = ["Iy", "masse", "h", "A", "Wpl_y", "Wel_y", "iz", "It", "b", "tf", "Iz", "cf_tf", "Iw", "alpha_deg", "y_s", "d", "t", "Wt", "perimetre"]
RECHERCHES = ["", "", "", "IPE 200", "ipe200", "100x100", "HEB 3", "heb", "L 50", "rhs", "CHS 1", "upn", "x", "HE 1000", "200 10",
              "IPE,200", "  hea   2", "froid", "PRS", "zzz", "60x60x5", "HD 260", "9", "L"]
GROUPES = CAT.groupes_presents()


def valeurs_de(cle):
    return sorted(p[cle] for p in CAT.profils() if p.get(cle) is not None)


def borne_au_hasard(rng, cle):
    vs = valeurs_de(cle)
    return float(vs[int(rng.random() * (len(vs) - 1))]) if vs else float(rng.randint(1, 100))


def poser_familles(at, familles):
    for g, fams in GROUPES:
        widget(at, "pills", "prf_w_fam_" + g).set_value([f for f in fams if f in familles])


def poser_colonnes(at, colonnes):
    for g in CAT.GROUPES_COLONNES:
        widget(at, "pills", "prf_w_col_" + g).set_value([c for c in CAT.colonnes_du_groupe(g) if c in colonnes])


def poser_filtres(at, filtres):
    for i, f in enumerate(filtres):
        widget(at, "selectbox", f"prf_w_f_cle_{i}").set_value(f["cle"])
        widget(at, "number_input", f"prf_w_f_min_{i}").set_value(f["min"])
        widget(at, "number_input", f"prf_w_f_max_{i}").set_value(f["max"])


def ajouter_perso(at, fam, dims):
    """La forme choisie (une exécution, les champs de la forme apparaissent), puis les dimensions et le bouton."""
    widget(at, "selectbox", "prf_perso_fam").set_value(fam)
    courir(at)
    for k, x in dims.items():
        widget(at, "number_input", f"prf_perso_{fam}_{k}").set_value(float(x))
    widget(at, "button", "prf_perso_btn").click()
    courir(at)


def dims_au_hasard(rng, fam):
    pas = lambda a, b: round(rng.uniform(a, b) * 2) / 2  # noqa: E731 — le pas de 0,5 mm du champ
    if fam == "PRS":
        h = pas(150, 1200)
        return {"h": h, "b": pas(80, 500), "tw": pas(4, 25), "tf": pas(6, 50)}
    if fam == "CAISSON":
        return {"h": pas(80, 600), "b": pas(80, 600), "t": pas(3, 30)}
    if fam == "TUBE":
        return {"d": pas(30, 600), "t": pas(2, 25)}
    if fam == "PLAT":
        return {"h": pas(5, 400), "b": pas(3, 60)}
    return {"d": pas(6, 150)}


def scenario(k, rng):
    """Un état tiré au hasard, posé geste par geste sur l'écran réel ; chaque exécution est observée."""
    at = ouvrir()
    observer(at, f"cas {k} · ouverture", {"fiche_avant": None})
    persos = []
    for _ in range(rng.choice([0, 0, 1, 2])):
        fam = rng.choice(CO.formes_perso())
        dims = dims_au_hasard(rng, fam)
        ajouter_perso(at, fam, dims)
        try:
            CO.controle_ecran(fam, dims)
            persos = [x for x in persos if CO.nom_perso(*x) != CO.nom_perso(fam, dims)] + [(fam, dims)]
            avant = CO.nom_perso(fam, dims)
        except ValueError:
            avant = OBS[-1]["etat"]["fiche"] if OBS else None
        observer(at, f"cas {k} · personnalisé {fam}", {"persos_attendus": list(persos), "erreur_perso": (fam, dims), "fiche_avant": avant})
    # les réglages : familles, recherche, filtres, séries courantes, tri, colonnes — posés ensemble
    toutes = CO.toutes_familles()
    fam_sel = rng.sample(toutes, rng.choice([0, 1, 2, 3, 5, 8, len(toutes)]))
    poser_familles(at, fam_sel)
    widget(at, "text_input", "prf_recherche").set_value(rng.choice(RECHERCHES))
    filtres = CO.filtres_vides()
    for i in range(rng.choice([0, 0, 1, 1, 2, 3])):
        cle = rng.choice(FREQUENTES if rng.random() < 0.8 else CLES)
        lo = borne_au_hasard(rng, cle) if rng.random() < 0.7 else None
        hi = borne_au_hasard(rng, cle) if rng.random() < 0.5 else None
        filtres[i] = {"cle": cle, "min": lo, "max": hi}
    poser_filtres(at, filtres)
    widget(at, "toggle", "prf_courants").set_value(rng.random() < 0.3)
    widget(at, "selectbox", "prf_tri").set_value(rng.choice([""] * 3 + CLES))
    widget(at, "toggle", "prf_decroissant").set_value(rng.random() < 0.4)
    col_sel = rng.sample(CLES, rng.choice([0, 1, 3, 6, 10, 15, 25, len(CLES)])) if rng.random() < 0.6 else list(CAT.DEFAUT_COLONNES)
    poser_colonnes(at, col_sel)
    fiche_avant, col_avant = OBS[-1]["etat"]["fiche"], OBS[-1]["etat"]["colonnes"]
    courir(at)
    o = observer(at, f"cas {k} · réglages", {"familles": fam_sel, "colonnes": col_sel, "fiche_avant": fiche_avant, "colonnes_avant": col_avant})
    noms = o["fiche_options"] or []
    # la fiche choisie dans la liste
    if noms:
        choix = rng.choice(noms)
        widget(at, "selectbox", "prf_w_fiche").set_value(choix)
        courir(at)
        observer(at, f"cas {k} · fiche {choix}", {"fiche_avant": choix})
    # la comparaison
    if noms:
        cmp_sel = rng.sample(noms, min(len(noms), rng.choice([0, 1, 2, 2, 3, 5])))
        widget(at, "multiselect", "prf_compare").set_value(cmp_sel)
        courir(at)
        observer(at, f"cas {k} · comparaison", {"compare_avant": cmp_sel, "fiche_avant": OBS[-1]["etat"]["fiche"]})
    # une autre recherche : la fiche et la comparaison suivent le tableau
    avant = OBS[-1]["etat"]
    widget(at, "text_input", "prf_recherche").set_value(rng.choice(RECHERCHES))
    courir(at)
    observer(at, f"cas {k} · autre recherche", {"fiche_avant": avant["fiche"], "compare_avant": avant["compare"]})
    # un bouton de la barre, parfois
    geste = rng.choice(["", "tout", "rien", "effacer", "courants"])
    if geste:
        avant = OBS[-1]["etat"]
        if geste == "tout":
            widget(at, "button", "prf_btn_tout").click()
        elif geste == "rien":
            widget(at, "button", "prf_btn_rien").click()
        elif geste == "effacer":
            widget(at, "button", "prf_btn_filtres_rien").click()
        else:
            widget(at, "toggle", "prf_courants").set_value(not avant["courants"])
        courir(at)
        att = {"fiche_avant": avant["fiche"], "compare_avant": avant["compare"]}
        if geste == "tout":
            att["familles"] = CO.toutes_familles()
        elif geste == "rien":
            att["familles"] = []
        o = observer(at, f"cas {k} · bouton {geste}", att)
        if geste == "effacer" and o["etat"]["filtres"] != CO.filtres_vides():
            chk(f"cas {k} : « Effacer les filtres » = filtres_vides()", False, o["etat"]["filtres"])
    del at
    gc.collect()


GRAINE = 20261107
rng = random.Random(GRAINE)
N_CAS = int(os.environ.get("CAT_N_CAS", "36"))      # moins de cas pour un essai rapide
print(f"  ({N_CAS} cas tirés au hasard, graine {GRAINE})")
try:
    prefs_temp(CAT.PREFS_DEFAUT)
    for k in range(N_CAS):
        scenario(k, rng)

    # ------------------------------------------------------------------ cas ciblés
    # préférences enregistrées : désordonnées, avec une famille et une colonne inconnues
    prefs_temp({"colonnes": ["Iw", "zz", "h", "masse", "Iy", "alpha_deg"], "familles": ["SHS", "XYZ", "IPE", "L", "HEB"],
                "tri": "Iy", "decroissant": True, "courants_seulement": True, "decimales_fixes": None})
    at = ouvrir()
    o = observer(at, "préférences enregistrées", {"reglages": "reglages_initiaux", "prefs": figer_prefs(), "fiche_avant": None})
    chk("préférences enregistrées (désordonnées, inconnues comprises) : réglages d'ouverture = reglages_initiaux()",
        not ecarts(o), ecarts(o)[:2])
    # « Garder par défaut » écrit le fichier ; « Rétablir » revient aux défauts livrés ET les écrit
    fprefs = CAT.CHEMIN_PREFS
    widget(at, "button", "prf_btn_save").click()
    courir(at)
    with open(fprefs, encoding="utf-8") as fh:
        garde = json.load(fh)
    chk("« Garder par défaut » écrit familles, colonnes, tri, ordre et séries courantes de l'écran",
        garde["familles"] == o["etat"]["familles"] and garde["colonnes"] == o["etat"]["colonnes"] and garde["tri"] == "Iy"
        and garde["decroissant"] is True and garde["courants_seulement"] is True, garde)
    widget(at, "text_input", "prf_recherche").set_value("HEB")
    poser_filtres(at, [{"cle": "Iy", "min": 1000.0, "max": None}])
    courir(at)
    observer(at, "avant Rétablir")
    widget(at, "button", "prf_btn_reset").click()
    courir(at)
    o = observer(at, "Rétablir", {"reglages": "reglages_livres", "prefs": figer_prefs()})
    with open(fprefs, encoding="utf-8") as fh:
        ecrit = json.load(fh)
    chk("« Rétablir » : réglages = reglages_livres() (filtres effacés, décroissant et séries courantes à faux)", not ecarts(o), ecarts(o)[:2])
    chk("« Rétablir » ÉCRIT les défauts livrés dans le fichier des préférences (pour tout le monde : défaut documenté)",
        ecrit == CAT.PREFS_DEFAUT, ecrit)
    chk("« Rétablir » garde la recherche (« HEB ») : l'écran ne l'efface pas", o["etat"]["recherche"] == "HEB")
    del at
    gc.collect()

    # deux filtres sur la MÊME propriété : seul le dernier compte (défaut de l'écran, reproduit)
    prefs_temp(CAT.PREFS_DEFAUT)
    at = ouvrir()
    poser_filtres(at, [{"cle": "Iy", "min": 10000.0, "max": None}, {"cle": "Iy", "min": None, "max": 5000.0}])
    courir(at)
    o = observer(at, "deux filtres sur Iy", {"fiche_avant": None})
    chk("deux filtres sur Iy (≥ 10 000 puis ≤ 5 000) : l'écran ne garde que le second — le module aussi",
        o["n_lignes"] and o["iy_max"] <= 5000 and not ecarts(o), (o["n_lignes"], o["iy_max"], ecarts(o)[:1]))
    # min > max : rien, et le message
    poser_filtres(at, [{"cle": "masse", "min": 50.0, "max": 20.0}, {"cle": "", "min": None, "max": None}])
    courir(at)
    o = observer(at, "min > max", {"fiche_avant": OBS[-1]["etat"]["fiche"]})
    chk("masse ≥ 50 et ≤ 20 : tableau vide, « Aucun profilé », fiche vide", o["lignes_df"] is None and CO.MESSAGE_VIDE in o["infos"]
        and o["etat"]["fiche"] is None and not ecarts(o), ecarts(o)[:2])
    widget(at, "button", "prf_btn_filtres_rien").click()
    courir(at)
    o = observer(at, "effacer après min > max", {"fiche_avant": None})
    chk("« Effacer les filtres » : la fiche revient au premier profilé", o["etat"]["fiche"] == (o["fiche_options"] or [None])[0] and not ecarts(o))

    # une fiche de chaque famille du catalogue (le milieu de la série), toutes familles cochées
    widget(at, "button", "prf_btn_tout").click()
    courir(at)
    observer(at, "toutes les familles", {"familles": CO.toutes_familles(), "fiche_avant": OBS[-1]["etat"]["fiche"]})
    for fam in CO.toutes_familles():
        serie = [p["nom"] for p in CAT.profils() if p["famille"] == fam]
        nom = serie[len(serie) // 2]
        widget(at, "selectbox", "prf_w_fiche").set_value(nom)
        courir(at)
        o = observer(at, f"fiche {nom}", {"fiche_avant": nom})
        if o["etat"]["fiche"] != nom or ecarts(o):
            chk(f"fiche {nom} ({fam}) : la même à l'écran et dans le module", False, ecarts(o)[:2])
    # profilés personnalisés aux bornes exactes des contrôles de l'écran, et le même nom deux fois
    persos = []
    for fam, dims, possible in (("PRS", {"h": 24.0, "b": 200.0, "tw": 8.0, "tf": 12.0}, False),       # 2·tf = h : refusé
                                ("PRS", {"h": 24.5, "b": 200.0, "tw": 8.0, "tf": 12.0}, True),
                                ("CAISSON", {"h": 200.0, "b": 300.0, "t": 100.0}, False),             # 2·t = min(h, b)
                                ("CAISSON", {"h": 200.0, "b": 300.0, "t": 99.5}, True),
                                ("TUBE", {"d": 100.0, "t": 50.0}, False),
                                ("TUBE", {"d": 100.0, "t": 49.5}, True),
                                ("PRS", {"h": 400.0, "b": 200.0, "tw": 300.0, "tf": 12.0}, True),     # t_w > b : calculé, sans le dire
                                ("PLAT", {"h": 0.1, "b": 0.1}, True),
                                ("ROND", {"d": 40.0}, True),
                                ("PRS", {"h": 24.5, "b": 200.0, "tw": 8.0, "tf": 12.0}, True)):      # le même nom : remplacé, en fin
        ajouter_perso(at, fam, dims)
        if possible:
            persos = [x for x in persos if CO.nom_perso(*x) != CO.nom_perso(fam, dims)] + [(fam, dims)]
        o = observer(at, f"personnalisé {CO.nom_perso(fam, dims)}", {"persos_attendus": list(persos), "erreur_perso": (fam, dims),
                                                                       "fiche_avant": CO.nom_perso(fam, dims) if possible else OBS[-1]["etat"]["fiche"]})
        chk(f"profilé personnalisé {CO.nom_perso(fam, dims)} : {'calculé' if possible else 'refusé'} comme l'écran", not ecarts(o)
            and (bool(o["erreurs"]) != possible), (o["erreurs"], ecarts(o)[:2]))
    prs = next(q for q in OBS[-1]["persos"] if q["nom"] == "PRS 400x200x300x12")
    chk("PRS 400x200x300x12 (t_w > b) : l'écran affiche c_f/t_f = −4,17 sans rien dire ; le module le signale",
        round(prs["cf_tf"], 2) == -4.17 and CO.avertissements_perso("PRS", prs["dims"]), prs["cf_tf"])
    # les personnalisés ignorent familles et séries courantes (comme dans l'écran), pas la recherche ni les bornes
    widget(at, "button", "prf_btn_rien").click()
    courir(at)
    o = observer(at, "Aucune famille, des personnalisés", {"familles": [], "fiche_avant": OBS[-1]["etat"]["fiche"]})
    chk("« Aucune » famille : les profilés personnalisés restent au tableau (familles ignorées)",
        o["n_lignes"] and all(x in CO.formes_perso() for x in o["familles_df"]) and not ecarts(o), ecarts(o)[:2])
    widget(at, "text_input", "prf_recherche").set_value("caisson")
    courir(at)
    o = observer(at, "recherche « caisson »", {"fiche_avant": OBS[-1]["etat"]["fiche"]})
    chk("recherche « caisson » : un seul personnalisé, compteur « **1 profilé** · 1 personnalisé »",
        not ecarts(o) and "**1 profilé** · 1 personnalisé" in o["md"], ecarts(o)[:2])
    del at
    gc.collect()
finally:
    CAT.CHEMIN_PREFS = PREFS_ORIG

n_vides = sum(1 for o in OBS if o["lignes_df"] is None)
n_comp = sum(1 for o in OBS if len(o["etat"]["compare"]) >= 2)
n_pers = sum(1 for o in OBS if o["persos"])
n_tri = sum(1 for o in OBS if o["etat"]["tri"])
n_fil = sum(1 for o in OBS if any(f["cle"] for f in o["etat"]["filtres"]))
n_cel = sum(o["n_lignes"] * len(o["colonnes_df"] or []) for o in OBS)
print(f"  {len(OBS)} exécutions observées · {n_cel} cellules · {n_vides} tableaux vides · {n_comp} comparaisons · "
      f"{n_pers} avec personnalisés · {n_tri} triées · {n_fil} filtrées")
mauvais = [(o["quoi"], ecarts(o)) for o in OBS if ecarts(o)]
chk(f"parité : chaque exécution observée ({len(OBS)}) = ce que le module calcule (listes, cellules, textes, fichiers)",
    not mauvais, mauvais[:3])
chk("couverture : des tableaux vides, des comparaisons, des personnalisés, des tris, des filtres, plus de 100 000 cellules",
    n_vides >= 3 and n_comp >= 10 and n_pers >= 10 and n_tri >= 10 and n_fil >= 15 and n_cel > 100000,
    (n_vides, n_comp, n_pers, n_tri, n_fil, n_cel))
formes_vues = {F.FAMILLES[o["svg"][0]]["forme"] for o in OBS if o["svg"]}
chk("couverture : les deux sens du tri, les séries courantes, une fiche de CHAQUE forme de contour (I, I conique, U, "
    "U conique, L, creux rond et rectangulaire, plat, rond)",
    any(o["etat"]["decroissant"] and o["etat"]["tri"] for o in OBS) and any(o["etat"]["courants"] for o in OBS)
    and formes_vues == {d["forme"] for d in F.FAMILLES.values()}, sorted({d["forme"] for d in F.FAMILLES.values()} - formes_vues))
chk("couverture : une fiche hors de la liste revient au premier, une comparaison perd un profilé sorti",
    any(o["attendu"].get("fiche_avant") and o["attendu"]["fiche_avant"] != o["etat"]["fiche"] and o["etat"]["fiche"] for o in OBS)
    and any(len(o["attendu"].get("compare_avant") or []) > len(o["etat"]["compare"]) for o in OBS))

# ====================================================================== 2 bis. bout à bout : le SITE contre l'écran
print("\n=== 2 bis. Bout à bout : ce que le SITE affiche (routeur → HTML) = ce que l'écran Streamlit montre ===")
import html as _html  # noqa: E402

from api_web import catalogue as M  # noqa: E402
from api_web import routeur as R  # noqa: E402


def _site(etat, **k):
    r = json.loads(R.traiter(json.dumps(dict(module="catalogue", etat=etat, **k))))
    if not r["ok"]:
        raise RuntimeError(r["erreur"])
    return r


def _txt(h):
    return _html.unescape(re.sub(r"<[^>]+>", "", h))


def _lu(s):
    """Une cellule du site relue : « 1 317,0 » → 1317.0 ; « — » → None."""
    s = s.replace(" ", "").replace("\xa0", "").replace(" ", "").replace(",", ".").replace("−", "-")
    return None if s in ("—", "") else float(s)


def poser_les_deux(fam=None, rech=None, filtres=None, courants=None, tri=None, desc=None, cols=None, persos=(), fiche=None, compare=None):
    """Le même état posé geste par geste sur l'écran (AppTest) et sur le site (les requêtes du navigateur) ; rend ce
    que l'écran MONTRE (sans rien recalculer) et l'état du site."""
    at = ouvrir()
    e = _site(None, action="nouveau")["etat"]

    def m(chemin, valeur):
        nonlocal e
        r = _site(e, action="modifier", chemin=chemin, valeur=valeur)
        if r["message"]:
            raise RuntimeError(f"{chemin} : {r['message']}")
        e = r["etat"]
    m(["lignes"], "0")
    for f, dims in persos:
        ajouter_perso(at, f, dims)
        m(["perso", "forme"], f)
        for k, x in dims.items():
            m(["perso", "dims", f, k], str(x).replace(".", ","))
        m(["perso", "+"], "1")
    if fam is not None:
        poser_familles(at, fam)
        m(["familles", "aucune"], "1")
        for f in fam:
            m(["familles", f], True)
    if rech is not None:
        widget(at, "text_input", "prf_recherche").set_value(rech)
        m(["recherche"], rech)
    if filtres:
        poser_filtres(at, filtres)
        for i, f in enumerate(filtres):
            m(["filtres", str(i), "cle"], f["cle"])
            for b in ("min", "max"):
                if f[b] is not None:
                    m(["filtres", str(i), b], repr(f[b]).replace(".", ","))
    if courants is not None:
        widget(at, "toggle", "prf_courants").set_value(courants)
        m(["courants"], courants)
    if tri is not None:
        widget(at, "selectbox", "prf_tri").set_value(tri)
        m(["tri"], tri)
    if desc is not None:
        widget(at, "toggle", "prf_decroissant").set_value(desc)
        m(["ordre"], "decroissant" if desc else "croissant")
    if cols is not None:
        poser_colonnes(at, cols)
        for c in CLES:
            m(["colonnes", c], c in cols)
    courir(at)
    if fiche is not None:
        widget(at, "selectbox", "prf_w_fiche").set_value(fiche)
        courir(at)
    # la fiche dépend de l'ORDRE des gestes (l'écran les applique en une exécution, le site un à un) : celle de
    # l'écran est ouverte sur le site ; la règle qui la fait suivre le tableau est prouvée au point 2
    if at.session_state["prf_fiche"] is not None:
        e = _site(e, action="choisir", role="fiche", nom=at.session_state["prf_fiche"])["etat"]
    if compare is not None:
        widget(at, "multiselect", "prf_compare").set_value(compare)
        courir(at)
        for n in compare:
            e = _site(e, action="choisir", role="compare", nom=n)["etat"]
    o = {"md": [str(x.value) for x in at.markdown], "captions": [str(c.value) for c in at.caption], "etat": e,
         "df": at.dataframe[0].value.to_dict("records") if len(at.dataframe) else None,
         "df_cols": list(at.dataframe[0].value.columns) if len(at.dataframe) else None,
         "csv": (ENREG.get("⬇️ CSV") or (None,))[0], "json": (ENREG.get("⬇️ Fiche (JSON)") or (None,))[0]}
    try:
        o["options"] = list(widget(at, "selectbox", "prf_w_fiche").options)
    except StopIteration:
        o["options"] = None
    del at
    gc.collect()
    return o


def ecarts_site(o) -> tuple:
    """(écarts, nombre de valeurs comparées) entre ce que l'écran a montré et ce que le site affiche maintenant."""
    out, n = [], 0
    e = o["etat"]
    res = _site(e, action="calculer")
    vt = next(v for v in res["vues"] if v["cle"] == "tableau")
    cpt = next((x for x in o["md"] if x.startswith("**")), None)
    if cpt is None or cpt.replace("**", "") != vt["legende"]:
        out.append(("compteur", cpt, vt["legende"]))
    if o["df"] is not None:
        th = re.findall(r"<th([^>]*)>(.*?)</th>", re.search(r"<thead>(.*?)</thead>", vt["svg"], re.S).group(1), re.S)
        cols = [(_html.unescape(t.group(1)).split(" — ")[0] if (t := re.search(r'title="([^"]*)"', a)) else _txt(x).strip()) for a, x in th]
        corps = re.search(r"<tbody>(.*?)</tbody>", vt["svg"], re.S).group(1)
        rows = [[_txt(td) for td in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)] for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", corps, re.S)]
        noms = [x["Profilé"] for x in o["df"]]
        vus = [r_[0] if r_[0] in noms else re.sub(r"(perso)?\d?$", "", r_[0]) for r_ in rows]
        if vus != noms:
            out.append(("liste", noms[:4], vus[:4]))
        elif sorted(cols[2:]) != sorted(o["df_cols"][2:]):
            out.append(("colonnes", o["df_cols"][2:], cols[2:]))
        else:
            for ligne, rw in zip(o["df"], rows):
                if rw[1] != ligne["Famille"]:
                    out.append(("famille", ligne["Profilé"], rw[1]))
                for j, c in enumerate(cols[2:]):
                    v = ligne[c]
                    v = None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)
                    s, n = _lu(rw[2 + j]), n + 1
                    if (v is None) != (s is None) or (v is not None and abs(v - s) > 1e-9 * max(1.0, abs(v))):
                        out.append(("cellule", ligne["Profilé"], c, v, rw[2 + j]))
        e2 = copy.deepcopy(e)
        e2["recherches"][e2["actif"]]["export"] = {"tableau": True, "fiche": True}
        d2 = _site(e2, action="calculer")["donnees"]
        z = re.search(r'<textarea data-chemin="export/texte_tableau"[^>]*>(.*?)</textarea>', d2, re.S)
        if o["csv"] is None or not z or _html.unescape(z.group(1)).replace("\t", ";") != o["csv"].decode("utf-8-sig"):
            out.append(("CSV", (o["csv"] or b"")[:60], z and z.group(1)[:60]))
        z = re.search(r'<textarea data-chemin="export/texte_fiche"[^>]*>(.*?)</textarea>', d2, re.S)
        if o["json"] is not None and (not z or _html.unescape(z.group(1)) != o["json"].decode("utf-8")):
            out.append(("fiche JSON", o["json"][:60], z and z.group(1)[:60]))
    elif CO.MESSAGE_VIDE not in _txt(res["resultat"]):
        out.append(("message vide", _txt(res["resultat"])[:80]))
    tab = next((x for x in o["md"] if x.startswith("<table class='prf-tab'><tr><td class='g'")), None)
    if tab:
        a = re.findall(r"<tr><td title=\"[^\"]*\">([^<]*)</td><td class='n'>([^<]*)</td><td class='prf-note'>([^<]*)</td></tr>", tab)
        fi = re.search(r'<div class="cat-fiche">(.*?)</table>', res["resultat"], re.S)
        b = [(_txt(x), _txt(y), _txt(z_)) for x, y, z_ in re.findall(r'<td class="s"[^>]*>(.*?)</td><td class="n">(.*?)</td><td class="u">(.*?)</td>', fi.group(1), re.S)] if fi else []
        n += len(a)
        if [(x.replace(" ", ""), y, z_) for x, y, z_ in a] != [(x.replace(" ", ""), y, z_) for x, y, z_ in b]:
            out.append(("fiche", a[:2], b[:2]))
        h3 = re.search(r"<h3>(.*?)</h3>", next(x for x in o["md"] if "<div class='prf-fiche'>" in x)).group(1)
        if f"<b>{_html.escape(h3)}</b>" not in res["resultat"]:
            out.append(("nom de la fiche", h3))
        it = [c for c in o["captions"] if c.startswith("It : ")]
        if it and it[0][5:-1].replace(".", ",") not in _txt(res["resultat"]).replace(".", ","):
            out.append(("méthode de I_t", it))
    tc = next((x for x in o["md"] if x.startswith("<table class='prf-tab'><tr><th>Propriété")), None)
    vc = next((v for v in res["vues"] if v["cle"] == "comparaison"), None)
    if (tc is None) != (vc is None):
        out.append(("comparaison présente", tc is not None, vc is not None))
    elif tc:
        a = [re.findall(r"<td class='n'>([^<]*)</td>", tr) for tr in re.findall(r"<tr><td>[^<]*</td>(.*?)</tr>", tc)]
        corps = re.search(r"<tbody>(.*?)</tbody>", vc["svg"], re.S).group(1)
        b = [[_txt(td) for td in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)][1:] for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", corps, re.S)]
        n += sum(map(len, a))
        if a != b:
            out.append(("comparaison", a[:1], b[:1]))
    if o["options"] is not None:
        sel = _site(e, action="selecteur", role="fiche")["html"]
        if [_html.unescape(x) for x in re.findall(r'data-choix="([^"]*)"', sel)] != o["options"]:
            out.append(("sélecteur de la fiche", o["options"][:3]))
    return out, n


IPE80 = CAT.par_nom()["IPE 80"]
TOUTES = CO.toutes_familles()
CAS_BOUT = [
    ("ouverture", {}),
    ("« heb 3 », I_y décroissant", dict(rech="heb 3", tri="Iy", desc=True)),
    ("cornières, 50 ≤ I_y ≤ 200, colonnes des L", dict(fam=["L", "LI"], filtres=[{"cle": "Iy", "min": 50.0, "max": 200.0}],
                                                    cols=["h", "b", "t", "Iy", "Iz", "Iu", "Iv", "alpha_deg", "masse"], fiche="L 100x65x8")),
    ("CHS, masse ≤ 30, tri masse, torsion", dict(fam=["CHS"], filtres=[{"cle": "masse", "min": None, "max": 30.0}], tri="masse",
                                                cols=["d", "t", "A", "It", "Wt", "masse"])),
    ("toutes, séries courantes, « 200 », tri A", dict(fam=TOUTES, courants=True, rech="200", tri="A")),
    ("min > max : vide", dict(filtres=[{"cle": "masse", "min": 50.0, "max": 20.0}])),
    ("aucune famille + PRS 600x250x10x20", dict(fam=[], persos=[("PRS", {"h": 600, "b": 250, "tw": 10, "tf": 20})])),
    ("caisson + recherche « caisson »", dict(persos=[("CAISSON", {"h": 300, "b": 200, "t": 10})], rech="caisson")),
    ("tube 168,3x5 (décimale)", dict(persos=[("TUBE", {"d": 168.3, "t": 5})], rech="TUBE")),
    ("plat 0,1x0,1 (minimum exact)", dict(persos=[("PLAT", {"h": 0.1, "b": 0.1})], rech="plat")),
    ("PRS t_w > b", dict(persos=[("PRS", {"h": 400, "b": 200, "tw": 300, "tf": 12})], rech="PRS")),
    ("comparaison IPE/HEA/HEB 300, 39 colonnes", dict(rech="300", cols=CLES, compare=["IPE 300", "HEA 300", "HEB 300"])),
    ("comparaison de 5 mêlés (U, L, creux)", dict(fam=TOUTES, compare=["UPN 200", "L 100x65x8", "RHS 200x100x8", "CHS 168.3x5", "HEB 300"])),
    ("pile à la limite : I_y(IPE 80) ≤ I_y ≤ I_y(IPE 80), valeur exacte", dict(fam=["IPE"], filtres=[{"cle": "Iy", "min": IPE80["Iy"], "max": IPE80["Iy"]}])),
    ("HD (noms à point), 39 colonnes, I_w décroissant", dict(fam=["HD"], cols=CLES, tri="Iw", desc=True)),
    ("UPE/UPN, tri y_s, colonnes des U", dict(fam=["UPE", "UPN"], tri="y_s", cols=["h", "b", "y_s", "Iw", "It", "cf_tf", "cw_tw"])),
    ("formés à froid, trois filtres", dict(fam=["SHS-F", "RHS-F", "CHS-F"], filtres=[{"cle": "h", "min": 100.0, "max": None},
                                                                               {"cle": "t", "min": None, "max": 5.0},
                                                                               {"cle": "masse", "min": None, "max": 20.0}])),
    ("recherche « hea   2 », tri W_pl,y", dict(rech="  hea   2", tri="Wpl_y")),
]
prefs_temp(CAT.PREFS_DEFAUT)
try:
    OBS_SITE, mauvais_site, n_val = [], [], 0
    for nom_cas, k in CAS_BOUT:
        o = poser_les_deux(**k)
        OBS_SITE.append((nom_cas, o))
        ec, n = ecarts_site(o)
        n_val += n
        if ec:
            mauvais_site.append((nom_cas, ec[:2]))
finally:
    CAT.CHEMIN_PREFS = PREFS_ORIG
chk(f"bout à bout : {len(CAS_BOUT)} états, {n_val} valeurs (cellules, fiche, comparaison) + CSV, JSON, compteur, ordre du "
    "sélecteur : ce que le SITE affiche = ce que l'écran Streamlit montre", not mauvais_site and n_val > 40000, (mauvais_site[:2], n_val))
_valeur, _table_fiche, _ent = M._valeur, M._table_fiche, M.colonnes_affichees
for nom_mut, muter in (("cellule écrite avec une décimale de moins", lambda: setattr(M, "_valeur", lambda v, c: _valeur(v, c)[:-1] if v else _valeur(v, c))),
                       ("fiche sans sa dernière ligne", lambda: setattr(M, "_table_fiche", lambda p, f: re.sub(r"<tr><td class=\"s\"(?:(?!<tr).)*</tr></table>", "</table>", _table_fiche(p, f), flags=re.S))),
                       ("colonne vide affichée", lambda: setattr(M, "colonnes_affichees", lambda r, v: _ent(r, v) + ["Iu"] if "Iu" not in _ent(r, v) else _ent(r, v)))):
    muter()
    try:
        rouges = sum(1 for _n, o in OBS_SITE if ecarts_site(o)[0])
    finally:
        M._valeur, M._table_fiche, M.colonnes_affichees = _valeur, _table_fiche, _ent
    chk(f"mutation du SITE « {nom_mut} » : le bout à bout rougit ({rouges} état(s))", rouges > 0)

# la réserve sur I_t des PRS (Σ b·t³/3, non conservatif) : ses chiffres sont MESURÉS contre Saint-Venant
from acier.profiles import geometrie as G  # noqa: E402
ecarts_it = {}
for nom_prs, d in (("PRS 400x200x8x12", {"h": 400.0, "b": 200.0, "tw": 8.0, "tf": 12.0}),
                   ("PRS 200x150x15x30", {"h": 200.0, "b": 150.0, "tw": 15.0, "tf": 30.0})):
    p_ = CO.calculer_perso("PRS", d)
    sv = G.torsion_saint_venant(F.contours("PRS", d), min(d["tw"], d["tf"]) / 16) / 1e4      # mm⁴ → cm⁴
    ecarts_it[nom_prs] = round(100 * (p_["It"] / sv - 1))
chk("réserve I_t des PRS : Σ b·t³/3 surestime de +2 % (400x200x8x12) et +10 % (200x150x15x30) contre Saint-Venant — "
    "les chiffres écrits", ecarts_it == {"PRS 400x200x8x12": 2, "PRS 200x150x15x30": 10}
    and all(f"+{v} %" in CO.reserve_torsion(CO.calculer_perso("PRS", {"h": 400.0, "b": 200.0, "tw": 8.0, "tf": 12.0})) for v in ecarts_it.values()),
    ecarts_it)

# la réserve sur A_v,z des PRS (formule des laminés) : ses chiffres sont MESURÉS contre η·h_w·t_w (§6.2.6(3)(d))
p_ = CO.calculer_perso("PRS", {"h": 300.0, "b": 300.0, "tw": 20.0, "tf": 40.0})
hw_tw = (300.0 - 2 * 40.0) * 20.0 / 100.0                                                   # cm², η = 1
chk("réserve A_v,z des PRS : 52,0 cm² (t_w·(h − t_f)) contre η·h_w·t_w = 44,0 cm² (η = 1,0) : +18 %, le chiffre écrit ; "
    "en dessous avec η = 1,2 (52,8 cm²) ; aucune réserve pour un laminé",
    round(p_["Avz"], 6) == 52.0 and hw_tw == 44.0 and round(100 * (p_["Avz"] / hw_tw - 1)) == 18 and p_["Avz"] <= 1.2 * hw_tw
    and "+18 %" in CO.reserve_cisaillement(p_) and CO.reserve_cisaillement(CAT.par_nom()["HEB 300"]) is None, p_["Avz"])

# ====================================================================== 3. les mutations
print("\n=== 3. Mutations : chacune doit rougir ===")
_B = {k: getattr(CO, k) for k in dir(CO) if callable(getattr(CO, k)) and not k.startswith("_")}
MUTATIONS = {
    "bornes : le PREMIER filtre gagne": lambda: setattr(CO, "bornes", lambda fs: {f["cle"]: (f["min"], f["max"]) for f in reversed(fs) if f["cle"]}),
    "liste : les personnalisés filtrés par familles": lambda: setattr(CO, "liste", lambda fa, re_, fi, co, pe, tr, de: (
        lambda out: CAT.trier(out, tr, de) if tr else out)(CAT.filtrer(CAT.profils(), fa, re_, CO.bornes(fi), co) + CAT.filtrer(pe, fa, re_, CO.bornes(fi), False))),
    "liste : le tri avant les personnalisés": lambda: setattr(CO, "liste", lambda fa, re_, fi, co, pe, tr, de: (
        CAT.trier(CAT.filtrer(CAT.profils(), fa, re_, CO.bornes(fi), co), tr, de) if tr else CAT.filtrer(CAT.profils(), fa, re_, CO.bornes(fi), co))
        + CAT.filtrer(pe, None, re_, CO.bornes(fi), False)),
    "compteur : sans pluriel": lambda: setattr(CO, "compteur", lambda n, fa, np_: _B["compteur"](n, fa, np_).replace("profilés**", "profilé**")),
    "personnalisés comptés sans filtre": lambda: setattr(CO, "n_persos_affiches", lambda re_, fi, pe: len(pe)),
    "lignes du tableau : sans arrondi": lambda: setattr(CO, "lignes_tableau", lambda lst, cols: [
        dict({"Profilé": p["nom"], "Famille": p["famille"]}, **{CO.entete(c): p.get(c) for c in cols}) for p in lst]),
    "colonnes vides gardées": lambda: setattr(CO, "colonnes_visibles", lambda lst, cols: list(cols)),
    "fiche : revient quand elle reparaît": lambda: setattr(CO, "fiche_retenue", lambda f, noms: noms[-1] if noms and f not in noms else f),
    "comparaison : garde les profilés sortis": lambda: setattr(CO, "compares_retenus", lambda c, noms: list(c)),
    "fiche : aide perdue": lambda: setattr(CO, "fiche", lambda p: [(g, [(c, l, v, u, "") for c, l, v, u, _a in pr]) for g, pr in _B["fiche"](p)]),
    "pastilles : sans « personnalisé »": lambda: setattr(CO, "chips_fiche", lambda p: _B["chips_fiche"](p)[:2]),
    "comparaison : lignes vides gardées": lambda: setattr(CO, "comparaison", lambda ps, cols: [
        (c, CO.entete(c), [CAT.formater(p.get(c), c) for p in ps]) for c in cols]),
    "comparaison : toujours les colonnes par défaut": lambda: setattr(CO, "colonnes_comparaison", lambda cv: list(CAT.DEFAUT_COLONNES)),
    "CSV : le défaut des noms « corrigé »": lambda: setattr(CO, "texte_tableau", lambda lig, sep=";": _B["texte_tableau"](lig, sep).replace("x54,1", "x54.1")),
    "fiche JSON : indentation 2": lambda: setattr(CO, "fiche_json", lambda p: json.dumps(p, ensure_ascii=False, indent=2)),
    "nom du personnalisé : une décimale": lambda: setattr(CO, "nom_perso", lambda fam, d: f"{fam} " + "x".join(f"{float(d[k]):.1f}" for k in F.FAMILLES[fam]["dims"])),
    "contrôle du PRS : > au lieu de ≥": lambda: setattr(CO, "controle_ecran", lambda fam, d: (_B["controle_ecran"](fam, d)
                                                                                             if not (fam == "PRS" and 2 * d["tf"] == d["h"]) else None)),
    "familles : ordre des clics": lambda: setattr(CO, "familles_ordonnees", lambda ch: [f for f in ch if f in CO.toutes_familles()]),
    "colonnes : ordre des clics": lambda: setattr(CO, "colonnes_ordonnees", lambda ch: [c for c in ch if c in CO.cles_colonnes()]),
    "Rétablir : décroissant gardé": lambda: setattr(CO, "reglages_livres", lambda: dict(_B["reglages_livres"](), decroissant=True)),
    "légende du titre : version 1.1": lambda: setattr(CO, "legende_page", lambda: _B["legende_page"]().replace("version 1.0", "version 1.1")),
    "méthode de It : texte tronqué": lambda: setattr(CO, "methode_it", lambda p: (p.get("methode_It") or "")[:12] or None),
}
for nom, muter in MUTATIONS.items():
    muter()
    try:
        rouges = sum(1 for o in OBS if ecarts(o))
    except Exception as e:  # noqa: BLE001 — une mutation qui casse le calcul rougit aussi
        rouges = f"exception {type(e).__name__}"
    finally:
        for k, f in _B.items():
            setattr(CO, k, f)
    chk(f"mutation « {nom} » : rougit ({rouges} exécution(s))", rouges != 0, rouges)
chk("après les mutations, le module rétabli redonne la parité", not any(ecarts(o) for o in OBS))

# ====================================================================== 4. la pureté
print("\n=== 4. Pureté : sans streamlit ni pandas ===")
sonde = f"""
import sys, json
sys.path.insert(0, {RACINE!r})
class Refus:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("streamlit", "pandas", "matplotlib", "scipy", "reportlab", "PIL", "fitz"):
            raise ImportError("refusé : " + name)
sys.meta_path.insert(0, Refus())
from acier.profiles import consultation as CO
from api_web import catalogue as M
lst = CO.liste(["IPE", "HEB"], "", [{{"cle": "Iy", "min": 1000.0, "max": None}}], False, [], "masse", False)
p = CO.calculer_perso("PRS", {{"h": 500.0, "b": 200.0, "tw": 8.0, "tf": 12.0}})
r = M.resultat(M.nouvel_etat())
print(json.dumps([len(lst), lst[0]["nom"], round(p["Iy"], 6), len(r["resultat"]) > 100,
                  sorted(m for m in sys.modules if m.split(".")[0] in ("streamlit", "pandas"))]))
"""
pr = subprocess.run([sys.executable, "-c", sonde], capture_output=True, text=True, cwd=RACINE)
try:
    rep = json.loads(pr.stdout.strip().splitlines()[-1])
except (ValueError, IndexError):
    rep = None
lst_ref = CO.liste(["IPE", "HEB"], "", [{"cle": "Iy", "min": 1000.0, "max": None}], False, [], "masse", False)
chk("le module et le site se chargent et calculent sans streamlit ni pandas (sous-processus qui les refuse)",
    rep is not None and rep[0] == len(lst_ref) and rep[1] == lst_ref[0]["nom"] and rep[3] and rep[4] == []
    and rep[2] == round(CO.calculer_perso("PRS", {"h": 500.0, "b": 200.0, "tw": 8.0, "tf": 12.0})["Iy"], 6), (rep, pr.stderr[-600:]))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

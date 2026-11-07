# -*- coding: utf-8 -*-
"""Charges sur plancher : la PARITÉ du site « Stab Méthodes » (api_web/plancher.py + plancher/charges/operations.py)
avec l'écran Streamlit modules/charges_plancher.py, qui reste la référence et le secours. Chaque contrôle rougit si
l'on retire ce qu'il protège.

L'écran Streamlit est EXÉCUTÉ, sans navigateur, avec un faux module d'écran (FauxSt) qui ENREGISTRE ce qu'il afficherait
et rend aux widgets la valeur que l'utilisateur y aurait laissée ; un espion sur le moteur relève ce que l'écran calcule.

  1. Le calcul (plus de 120 notes tirées au hasard, graine fixe, plus des cas choisis) : la fonction ``show()`` de
     l'écran est exécutée ; le résultat de ``verification.verifier`` qu'elle obtient est comparé, AU BIT PRÈS (texte JSON
     identique, flottants compris), à celui que le site calcule (api_web.plancher._calculer) ; un refus, une exception
     de l'écran (l'écran s'arrête) et l'erreur que le site montre sont comparés aussi.
  2. Ce que l'écran COMPOSE : la ligne de résumé de l'étape Plancher, le total de l'étape Charges, les barres de la
     carte Résultat, le compte des lignes non vérifiées, le tableau « Appuis de E1 » (charges au bit près, répartiteur,
     élément, taux), le balayage (enveloppe au bit près et sa phrase), la proposition de filière (au bit près), son
     texte, « Appliquer la proposition ».
  3. Les FORMULAIRES : « Parts annexe C » (lecture identique, exceptions comprises), « Nouvel élément de
     répartition », « Nouvelle fiche d'engin » — la fiche que l'écran ajoute est celle d'operations.py.
  4. Les DÉFAUTS de l'écran Streamlit, reproduits par le moteur et refusés ou signalés par le site (une saisie que
     l'écran ramène en silence à une valeur fausse est refusée, les plantages sont rendus lisibles).
  5. MUTATIONS : des copies altérées d'operations.py (un seuil, un arrondi, une table, un ordre) font rougir la parité.

    /usr/bin/python3.12 tests/test_charges_plancher_parite.py      (streamlit et pandas requis : c'est l'écran d'origine)
"""
import contextlib
import copy
import importlib.util
import json
import logging
import math
import os
import random
import re
import sys
import types

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from plancher.charges import bibliotheque as BIB  # noqa: E402
from plancher.charges import diffusion as DF  # noqa: E402
from plancher.charges import etat as ET  # noqa: E402
from plancher.charges import operations as OP  # noqa: E402
from plancher.charges import verification as V  # noqa: E402

from api_web import plancher as W  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{str(info)[:700]}]" if info and not cond else ""))


def js(x) -> str:
    """Le texte JSON d'un résultat : deux résultats égaux au bit près ont le même texte (repr des flottants)."""
    return json.dumps(x, sort_keys=True, ensure_ascii=False, default=str)


# ------------------------------------------------------------------ l'écran d'origine, sans navigateur
logging.disable(logging.WARNING)
sys.modules["acier.composants.svg_cliquable"] = types.SimpleNamespace(svg_cliquable=lambda *a, **k: None)
import modules.charges_plancher as CP  # noqa: E402
import modules.liaison as LI  # noqa: E402


class Relance(Exception):
    """st.rerun() : l'écran demande une nouvelle exécution."""


class Session(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v


class FauxSt:
    """Un « st » qui ENREGISTRE ce que l'écran afficherait. Un widget rend la valeur de session de sa clé si elle
    existe, sinon sa valeur par défaut (comme Streamlit) ; ``libelles`` force la valeur d'un widget sans clé (formulaires) ;
    ``boutons`` : les clés (ou libellés) des boutons « cliqués » pendant cette exécution."""

    def __init__(self, session=None, boutons=(), libelles=None):
        self.session_state = session if session is not None else Session()
        self.boutons = set(boutons)
        self.libelles = dict(libelles or {})
        self.ev = []
        self.rappels = {}

    # affichage
    def _txt(self, sorte):
        def f(body="", *a, **k):
            self.ev.append((sorte, str(body)))
        return f

    def __getattr__(self, nom):
        if nom in ("markdown", "caption", "info", "warning", "error", "toast", "write"):
            return self._txt(nom)
        raise AttributeError(nom)

    def container(self, *a, **k):
        return contextlib.nullcontext()

    expander = form = container

    def columns(self, spec, **_):
        return [contextlib.nullcontext() for _ in range(spec if isinstance(spec, int) else len(spec))]

    def tabs(self, noms):
        return [contextlib.nullcontext() for _ in noms]

    def rerun(self):
        raise Relance()

    # widgets
    def _valeur(self, label, key, defaut):
        if label in self.libelles:
            return self.libelles[label]
        if key is not None and key in self.session_state:
            return self.session_state[key]
        if key is not None:
            self.session_state[key] = defaut
        return defaut

    def _rappel(self, key, on_change):
        if key is not None and on_change is not None:
            self.rappels[key] = on_change

    def number_input(self, label, value=0.0, *a, key=None, on_change=None, args=(), **k):
        self._rappel(key, (on_change, args))
        return self._valeur(label, key, value)

    def text_input(self, label, value="", *a, key=None, on_change=None, args=(), **k):
        self._rappel(key, (on_change, args))
        return self._valeur(label, key, value)

    def checkbox(self, label, value=False, *a, key=None, on_change=None, args=(), **k):
        self._rappel(key, (on_change, args))
        return self._valeur(label, key, value)

    def selectbox(self, label, options, index=0, *a, key=None, on_change=None, args=(), **k):
        self._rappel(key, (on_change, args))
        return self._valeur(label, key, list(options)[index] if options else None)

    def segmented_control(self, label, options, *a, default=None, key=None, on_change=None, args=(), **k):
        self._rappel(key, (on_change, args))
        return self._valeur(label, key, default)

    def multiselect(self, label, options, default=None, *a, key=None, **k):
        return self._valeur(label, key, list(default or []))

    def data_editor(self, df, *a, key=None, **k):
        return self.libelles.get("__appuis__", df)

    def file_uploader(self, *a, **k):
        return None

    def button(self, label, *a, key=None, **k):
        return key in self.boutons or label in self.boutons

    form_submit_button = button

    def download_button(self, *a, **k):
        return False


def installer(faux: FauxSt):
    CP.st = faux
    LI.st = faux


@contextlib.contextmanager
def espions():
    """Relève chaque appel de verification.verifier et de verification.balayer fait par l'écran."""
    vus = {"verifier": [], "balayer": []}
    v0, b0 = V.verifier, V.balayer

    def verifier(*a, **k):
        r = v0(*a, **k)
        vus["verifier"].append(r)
        return r

    def balayer(*a, **k):
        r = b0(*a, **k)
        vus["balayer"].append(r)
        return r
    V.verifier, V.balayer = verifier, balayer
    try:
        yield vus
    finally:
        V.verifier, V.balayer = v0, b0


def ecran(etat, vue="Plan", boutons=(), session=None):
    """L'écran d'origine sur ``etat`` : (ce qu'il a affiché, ce que le moteur lui a rendu, l'exception qui l'arrête)."""
    s = session if session is not None else Session()
    s["chp"] = copy.deepcopy(etat)
    s[CP.P + "vue"] = vue
    f = FauxSt(s, boutons)
    installer(f)
    exc = None
    with espions() as vus:
        try:
            CP.show()
        except Relance:
            pass
        except Exception as e:  # noqa: BLE001 — l'écran d'origine s'arrête : on le relève
            exc = e
    return f, vus, exc, s


def textes(f, sorte=None):
    return [b for k, b in f.ev if sorte is None or k == sorte]


# ------------------------------------------------------------------ les notes : tirées au hasard, graine fixe
ENGINS = list(ET.gabarits(ET.nouvel_etat()))
PLATEAUX = [n for n, d in ET.repartiteurs_disponibles(ET.nouvel_etat()).items() if d.get("type") == "plateau"]
PROFILS = list(BIB.profils())


def note_au_hasard(rng: random.Random):
    e = ET.nouvel_etat()
    cas = ET.cas_actif(e)
    pl = cas["plancher"]
    pl["type"] = rng.choice(["hourdis"] * 5 + ["dalle"] * 3 + ["predalle"] * 2)
    pl["L"] = round(rng.uniform(3.0, 9.0), 2)
    pl["sens"] = rng.choice([1, 1, 2])
    pl["q_presentes"] = rng.choice([0.0, 1.0, 2.5])
    h = pl["hourdis"]
    h.update({"M_Rd": round(rng.uniform(40, 220), 1), "V_Rd": round(rng.uniform(30, 160), 1), "nature": rng.choice(["Rd", "Rd", "service"]),
              "n_elements": rng.randint(3, 10), "b_element": rng.choice([0.6, 1.2, 1.2]), "etat": rng.choice(["poses", "clavetes", "chapes"]),
              "I": rng.choice([None, None, 0.00045]), "M_cr": rng.choice([None, 50.0]), "p_loc_adm": rng.choice([None, None, 1.2, 3.0]),
              "parts_annexe_C": rng.choice([None, None, {"0": 0.5, "1": 0.25}, {"0": 0.6, "1": 0.15, "2": 0.05}])})
    if rng.random() < 0.35:
        h["chape"].update({"oui": True, "epaisseur": rng.choice([0.04, 0.05, 0.06]), "fck_t": rng.choice([15.0, 25.0, 30.0]),
                           "collaborante": rng.random() < 0.7, "fck_exigee": rng.choice([None, 25.0])})
        if rng.random() < 0.6:
            h["M_Rd_chape"], h["V_Rd_chape"] = round(h["M_Rd"] * 1.4, 1), round(h["V_Rd"] * 1.2, 1)
    for t in ("dalle", "predalle"):
        d = pl[t]
        hh = rng.choice([0.16, 0.20, 0.24, 0.26])
        d.update({"h": hh, "d": round(hh - 0.03, 3), "As_inf": rng.choice([335.0, 524.0, 785.0, 1131.0]), "fyk": rng.choice([500.0, 500.0, 400.0]),
                  "beton": rng.choice(["C25/30", "C30/37", "C35/45"]), "fck_t": rng.choice([None, None, 22.0]),
                  "largeur": round(rng.uniform(3.0, 10.0), 1), "h1": rng.choice([0.0, 0.0, 0.05]), "armature_transversale_ok": rng.random() < 0.8})
    pl["predalle"].update({"b_predalle": rng.choice([1.2, 2.4]), "joints_armes": rng.random() < 0.6, "chape_durcie": rng.random() < 0.85,
                           "portee_etayee": rng.choice([None, None, round(pl["L"] / 2, 2)])})
    pr = cas["params"]
    pr.update({"gamma_G": rng.choice([1.35, 1.35, 1.0]), "gamma_Q": rng.choice([1.5, 1.5, 1.0]), "part_tranchant_element": rng.choice([1.0, 1.0, 0.7]),
               "k_flexion_heft240": rng.choice([None, None, 0.5]), "alpha_cc": rng.choice([1.0, 0.85])})
    if rng.random() < 0.3:
        pr.update({"joint_adm_kN_m": 15.0, "joint_longueur_m": 1.0})
    plancher = ET.construire(e, cas)[0]
    for _ in range(rng.choice([0, 1, 1, 1, 2])):
        ref = ET.ajouter_engin(e, cas, rng.choice(ENGINS), x=round(rng.uniform(0.4, plancher.L - 0.4), 2), y=round(rng.uniform(0.4, plancher.B - 0.4), 2))
        c = next(x for x in cas["charges"] if x["ref"] == ref)
        c["angle"] = rng.choice([0, 0, 90, 180, 270])
        c["P"] = {k: (0.0 if rng.random() < 0.04 else round(rng.uniform(5, 70), 1)) for k in c["P"]}
    for _ in range(rng.choice([0, 0, 1, 2])):
        ref = ET.ajouter_charge(e, cas, "ponctuel")
        c = cas["charges"][-1]
        c.update({"x": round(rng.uniform(0.2, plancher.L - 0.2), 2), "y": round(rng.uniform(0.2, plancher.B - 0.2), 2), "P": round(rng.uniform(5, 60), 1)})
    if rng.random() < 0.3:
        ET.ajouter_charge(e, cas, "lineaire")
        cas["charges"][-1].update({"orientation": rng.choice(["x", "y"]), "p": rng.choice([5.0, 10.0]), "longueur": rng.choice([1.2, 2.4])})
    if rng.random() < 0.3:
        ET.ajouter_charge(e, cas, "surfacique")
    refs = list(ET.construire(e, cas)[4]["empreintes"])
    for r in refs:
        u = rng.random()
        if u < 0.35:
            cas["repartiteurs"][r] = {"type": "plateau", "nom": rng.choice(PLATEAUX[:3] + PLATEAUX), "dx": rng.choice([0.0, 0.0, 0.1]), "dy": 0.0}
        elif u < 0.45 and r not in cas["repartiteurs"]:
            ET.regler_repartiteur(e, cas, r, "type", "filiere")
            if cas.get("filieres") and rng.random() < 0.5:
                cas["filieres"][-1]["nom"] = rng.choice(PROFILS)
    if refs:
        cas["selection"] = "rep:" + rng.choice(refs)
    return e


def notes_choisies():
    """Des cas nommés : les exemples du site, les défauts de l'écran, les bornes."""
    # d'abord des notes non conformes dont la proposition de filière aboutit (elles nourrissent aussi les mutations)
    out = []
    for nom, P in (("proposition trouvée (étançon de 120 kN sans filière)", "120"), ("proposition trouvée (étançon de 90 kN)", "90")):
        e = W.exemple("hourdis")
        W._faire(e, (["appuis", 4, "repartiteur"], "aucun"), (["filieres", 0, "-"], "1"), (["charges", 1, "P"], P))
        out.append((nom, e))
    e = W.exemple("hourdis")
    W._faire(e, (["appuis", 4, "repartiteur"], "aucun"), (["filieres", 0, "-"], "1"), (["charges", 1, "P"], "120"),
             (["proposition", "+"], "1"), (["proposition", "appliquer"], "1"), (["appuis", 1, "repartiteur"], "aucun"), (["charges", 0, "P", 1], "70"))
    out.append(("seconde proposition (filière F1p0 déjà appliquée)", e))
    out += [("exemple hourdis", W.exemple("hourdis")), ("exemple dalle", W.exemple("dalle")), ("exemple prédalle", W.exemple("predalle")),
            ("note neuve", W.nouvel_etat())]
    e = W.exemple("hourdis")
    e["fiches"]["repartiteurs"].append(OP.fiche_repartiteur("plateau", "Madrier azobé 600 × 200", "Azobé", 18.0, 12000.0, 9.0, "", 600.0, 200.0, 80.0,
                                                            OP.RECTANGLE))
    ET.cas_actif(e)["repartiteurs"]["E1.P1"] = {"type": "plateau", "nom": "Madrier azobé 600 × 200", "dx": 0.0, "dy": 0.0}
    out.append(("madrier 600 × 200 sous un pied 300 × 300 (l'écran s'arrête)", e))
    e = W.exemple("hourdis")
    ET.cas_actif(e)["charges"][0]["x"] = 7.0
    out.append(("engin hors du plancher (refus)", e))
    e = W.exemple("hourdis")
    ET.cas_actif(e)["plancher"]["hourdis"]["parts_annexe_C"] = {"0": 0.0}
    out.append(("parts annexe C « 0:0,5 » lues {0: 0} par l'écran (division par zéro)", e))
    e = W.exemple("hourdis")
    ET.cas_actif(e)["plancher"]["hourdis"]["nature"] = "service"
    out.append(("valeurs admissibles de service", e))
    e = W.exemple("hourdis")
    cas = ET.cas_actif(e)
    ref = ET.ajouter_engin(e, cas, "Étançon · semelle 150 × 150", x=2.0, y=2.4)   # son appui unique tombe sur le joint H2 | H3
    next(c for c in cas["charges"] if c["ref"] == ref)["P"] = {a["nom"]: 30.0 for a in ET.gabarits(e)["Étançon · semelle 150 × 150"]["appuis"]}
    out.append(("appui d'engin exactement sur un joint entre éléments", e))
    return out


rng = random.Random(20261107)
CAS = notes_choisies() + [(f"hasard n° {i + 1}", note_au_hasard(rng)) for i in range(120)]


# ------------------------------------------------------------------ 1. le calcul, au bit près
print("\n=== 1. Le calcul : l'écran Streamlit exécuté, son moteur relevé, comparé au site au bit près ===")
stats = {"calc": 0, "refus": 0, "plante": 0, "nok": 0, "ok": 0, "hourdis": 0, "dalle": 0, "predalle": 0, "plateau": 0, "filiere": 0,
         "local": 0, "joint": 0, "fissuration": 0, "engins": 0}
ecarts = []
memo = {}
for nom, etat in CAS:
    f, vus, exc, _ = ecran(etat)
    cas = ET.cas_actif(copy.deepcopy(etat))
    site = W._calculer(etat, cas)
    memo[nom] = (f, vus, exc, site)
    stats[cas["plancher"]["type"]] += 1
    stats["engins"] += sum(1 for c in cas["charges"] if c["type"] == "engin")
    if exc is not None:
        stats["plante"] += 1
        if "res" in site or str(exc) not in site.get("erreur", ""):
            ecarts.append((nom, "l'écran s'arrête", repr(exc), site.get("erreur")))
        continue
    if not vus["verifier"]:
        # l'écran s'est arrêté sur « Donnée manquante » (construction impossible)
        if "res" in site:
            ecarts.append((nom, "l'écran n'a rien calculé", textes(f, "error")))
        continue
    r_ecran = vus["verifier"][0]
    if js(r_ecran) != js(site.get("res")):
        ecarts.append((nom, "résultats différents"))
        continue
    stats["calc"] += 1
    stats[r_ecran["statut"] if r_ecran["statut"] in ("refus", "nok") else "ok"] += 1
    for v in r_ecran["verifications"]:
        if v["famille"] in stats:
            stats[v["famille"]] += 1
chk(f"{len(CAS)} notes : le résultat du moteur dans l'écran Streamlit = celui du site, AU BIT PRÈS (texte JSON identique)", not ecarts, ecarts[:3])
chk(f"couverture : {stats}", stats["calc"] >= 80 and stats["refus"] >= 3 and stats["nok"] >= 10 and stats["ok"] >= 10 and stats["plante"] >= 1
    and min(stats["hourdis"], stats["dalle"], stats["predalle"]) >= 10 and min(stats["plateau"], stats["filiere"], stats["local"], stats["joint"]) >= 3,
    stats)
f, vus, exc, site = memo["madrier 600 × 200 sous un pied 300 × 300 (l'écran s'arrête)"]
chk("défaut de l'écran : un madrier plus étroit que le pied ARRÊTE l'écran Streamlit (ValueError non rattrapée) ; le site l'écrit",
    isinstance(exc, ValueError) and "Empreinte plus grande que le plateau" in site.get("erreur", ""), (exc, site.get("erreur")))
f, vus, exc, site = memo["parts annexe C « 0:0,5 » lues {0: 0} par l'écran (division par zéro)"]
chk("défaut de l'écran : des parts nulles ({0: 0}, ce que l'écran lit dans « 0:0,5 ») ARRÊTENT l'écran (division par zéro) ; le site l'écrit",
    isinstance(exc, ZeroDivisionError) and "Calcul impossible" in site.get("erreur", ""), (exc, site.get("erreur")))
e = W.nouvel_etat()
chk("le site REFUSE « 0:0,5 » dans les parts annexe C (l'écran le lisait {0: 0} en silence)",
    W.modifier(e, ["plancher", "hourdis", "parts_annexe_C"], "0:0,5") is not None and ET.cas_actif(e)["plancher"]["hourdis"]["parts_annexe_C"] is None)


# ------------------------------------------------------------------ 2. ce que l'écran compose
print("\n=== 2. Ce que l'écran compose : résumés, barres, tableau des appuis, balayage, proposition ===")
fmt0 = CP._fmt
mauvais = {"resume": [], "total": [], "barres": [], "nv": [], "appuis": [], "balayage": [], "texte_bal": []}
n_app = n_bal = 0
RE_BARRE = re.compile(r'<div class="chp-row"><span>([^<]+)</span>.*?<span class="chp-v"[^>]*>(\d+) %</span>')
RE_APPUI = re.compile(r"<tr><td><b>([^<]+)</b></td><td class='n'>([^<]+) kN</td><td>([^<]*)</td><td>([^<]*)</td><td class='n[^']*'>([^<]+)</td></tr>")


def repr_fmt(x, nd=2):
    return repr(float(x))


for nom, etat in CAS:
    f, vus, exc, site = memo[nom]
    if exc is not None or not vus["verifier"]:
        continue
    cas = ET.cas_actif(copy.deepcopy(etat))
    res = vus["verifier"][0]
    plancher, _, _, _, index = ET.construire(etat, cas)
    # résumé de l'étape Plancher et total de l'étape Charges, en texte ET au bit près (format remplacé par repr)
    cap = textes(f, "caption")
    if OP.resume_plancher(cas) not in cap:
        mauvais["resume"].append(nom)
    if f"{len(cas['charges'])} objet(s) · {OP.fmt(OP.total_charges(cas), 0)} kN au total" not in cap:
        mauvais["total"].append(nom)
    # les barres de la carte Résultat et le compte des lignes non vérifiées
    carte = next((b for b in textes(f, "markdown") if '<div class="chp-res">' in b), "")
    barres = [(lab, int(p)) for lab, p in RE_BARRE.findall(carte)]
    if barres != [(lab, int(f"{t * 100:.0f}")) for _, lab, t in OP.barres(res)]:
        mauvais["barres"].append((nom, barres, OP.barres(res)))
    nv = OP.non_verifies(res)
    if bool(nv) != ("ligne(s) non vérifiée(s)" in carte) or (nv and f"{len(nv)} ligne(s)" not in carte):
        mauvais["nv"].append(nom)
    # le tableau « Appuis de E1 » : l'écran, son format remplacé par repr (au bit près)
    for c in [c for c in cas["charges"] if c["type"] == "engin"][:1]:
        e2 = copy.deepcopy(etat)
        ET.cas_actif(e2)["selection"] = f"engin:{c['ref']}"
        CP._fmt = repr_fmt
        try:
            f2, vus2, exc2, _ = ecran(e2)
        finally:
            CP._fmt = fmt0
        tab = next((b for b in textes(f2, "markdown") if "<th>Appui</th>" in b), "")
        lu = RE_APPUI.findall(tab)
        attendu = [(li["appui"], repr(float(li["charge"])), li["repartiteur"], li["element"] or "—",
                    f"{li['taux'] * 100:.0f} %" if li["taux"] is not None else "—")
                   for li in OP.appuis_engin(cas, res, index, c["ref"])]
        n_app += 1
        if lu != attendu:
            mauvais["appuis"].append((nom, lu[:2], attendu[:2]))
    # le balayage : l'enveloppe au bit près, la phrase des positions admissibles
    engins = [c["ref"] for c in cas["charges"] if c["type"] == "engin"]
    if engins and res["statut"] != "refus" and n_bal < 15:
        f3, vus3, exc3, _ = ecran(etat, vue="Balayage")
        if exc3 is None and vus3["balayer"]:
            n_bal += 1
            b = OP.balayage(etat, cas, res, engins[0])
            if js(b["balayage"]) != js(vus3["balayer"][0]):
                mauvais["balayage"].append(nom)
            if OP.legende_balayage(b["balayage"]) not in textes(f3, "caption"):
                mauvais["texte_bal"].append(nom)
for k, lst in mauvais.items():
    chk(f"composé par l'écran = operations.py : {k} ({len(lst)} écart(s))", not lst, lst[:2])
chk(f"le tableau des appuis a été comparé sur {n_app} notes, le balayage sur {n_bal}", n_app >= 30 and n_bal >= 10, (n_app, n_bal))

# la proposition de filière : l'écran (_proposer, le bouton « Proposer… ») contre operations.proposer_filiere
props, n_prop, n_trouve = [], 0, 0
PROP_ECRAN = {}
for nom, etat in CAS:
    f, vus, exc, site = memo[nom]
    if exc is not None or not vus["verifier"] or vus["verifier"][0]["statut"] != "nok" or n_prop >= 16:
        continue
    cas = ET.cas_actif(copy.deepcopy(etat))
    res = vus["verifier"][0]
    plancher, _, _, _, index = ET.construire(etat, cas)
    s = Session()
    installer(FauxSt(s))
    try:
        CP._proposer(etat, copy.deepcopy(cas), res, plancher, index)
    except Exception as e:  # noqa: BLE001
        s[CP.P + "proposition"] = {"exception": repr(e)}
    try:
        p_site = OP.proposer_filiere(etat, copy.deepcopy(cas), res, plancher, index)
    except Exception as e:  # noqa: BLE001
        p_site = {"exception": repr(e)}
    n_prop += 1
    n_trouve += bool(p_site.get("trouve"))
    PROP_ECRAN[nom] = s[CP.P + "proposition"]
    if js(s[CP.P + "proposition"]) != js(p_site):
        props.append(nom)
        continue
    if "exception" in p_site:
        continue
    # le texte affiché, puis « Appliquer la proposition »
    s2 = Session({CP.P + "proposition": p_site})
    f4 = FauxSt(s2, boutons={CP.P + "btn_appliquer"})
    installer(f4)
    c_ecran, c_site = copy.deepcopy(cas), copy.deepcopy(cas)
    try:
        CP._carte_resultat(etat, c_ecran, res, plancher, index)
    except Relance:
        pass
    vu = textes(f4, "info") + textes(f4, "warning")
    if OP.phrase_proposition(p_site) not in vu:
        props.append((nom, "texte", vu[:1]))
    if p_site["trouve"]:
        OP.appliquer_proposition(c_site, p_site)
        if js(c_ecran) != js(c_site):
            props.append((nom, "appliquer"))
chk(f"proposition de filière ({n_prop} notes non conformes, {n_trouve} trouvées) : écran = operations.py au bit près, même texte, même « Appliquer »",
    not props and n_prop >= 10 and n_trouve >= 2, (props[:3], n_prop, n_trouve))


# ------------------------------------------------------------------ 3. les formulaires
print("\n=== 3. Les formulaires de l'écran : parts annexe C, nouvel élément, nouvelle fiche d'engin ===")
parts_ko = []
for s_ in ["0:0.5, 1:0.25", "0:0.5;1:0.25", "0:0,5", "", "  ", "0:0.5, 1", "a:b", "0:1:2", "2:0.1, 0:0.8", "0 : 0.6 ; 1 : 0.2", "0:-0.5", "1:0.5"]:
    cas = ET.nouveau_cas("C")
    sess = Session({CP.P + "w_parts_C": s_})
    fa = FauxSt(sess)
    installer(fa)
    CP._parts_annexe_c(cas)
    rappel, args = fa.rappels[CP.P + "w_parts_C"]
    try:
        rappel(*args)
        e_ecran = js(cas["plancher"]["hourdis"]["parts_annexe_C"])
    except Exception as e:  # noqa: BLE001
        e_ecran = type(e).__name__
    try:
        e_site = js(OP.lire_parts_annexe_c(s_))
    except Exception as e:  # noqa: BLE001
        e_site = type(e).__name__
    if e_ecran != e_site:
        parts_ko.append((s_, e_ecran, e_site))
chk("« Parts annexe C » : 12 saisies lues comme l'écran les lit (valeurs et exceptions)", not parts_ko, parts_ko)
chk("« Parts annexe C » : ce que l'écran ignore en silence est nommé (morceaux sans « : »)",
    OP.morceaux_ignores("0:0,5") == ["5"] and OP.morceaux_ignores("0:0.5, 1:0.25") == [] and OP.morceaux_ignores("0:0.5, 1") == ["1"])
chk("« Parts annexe C » : le texte du champ est celui de l'écran (« 0:0.5, 1:0.25 »)",
    OP.texte_parts_annexe_c({0: 0.5, 1: 0.25}) == ", ".join(f"{k}:{v}" for k, v in {0: 0.5, 1: 0.25}.items()))

fiches_ko = []
for typ, nom, mat, fval, E, masse, src, A, B, t, prof in [
        ("plateau", "Plateau azobé 800", "Azobé", 18.0, 12000.0, 30.0, "fiche 2026", 800.0, 800.0, 80.0, OP.RECTANGLE),
        ("plateau", "Tôle S355", "S355", 0.0, 210000.0, 0.0, "", 600.0, 600.0, 25.0, OP.RECTANGLE),
        ("plateau", "OSB sans f", "OSB", 0.0, 4000.0, 8.0, "", 1250.0, 625.0, 18.0, OP.RECTANGLE),
        ("filiere", "Madrier C24 75×225", "Bois C24", 14.8, 11000.0, 8.0, "EN 338", 75.0, 225.0, 20.0, OP.RECTANGLE),
        ("filiere", "HEB 200 S355", "S355", 0.0, 210000.0, 61.3, "", 600.0, 600.0, 20.0, "HEB 200"),
        ("filiere", "  Espace  ", "S235", 0.0, 210000.0, 0.0, "", 100.0, 300.0, 20.0, OP.RECTANGLE)]:
    etat = ET.nouvel_etat()
    lib = {"Type": typ, "Nom": nom, "Matériau": mat, "f (MPa) : fy/γM0 ou f_m,d": fval, "E (MPa)": E, "Masse (kg ou kg/m)": masse,
           "Source (fiche, norme, date)": src, "Plateau A · ou filière : b (largeur)": A, "Plateau B · ou filière : h (hauteur sur champ)": B,
           "Plateau : épaisseur t": t, "Filière : profilé du catalogue (sinon rectangle b × h)": prof}
    installer(FauxSt(Session(), boutons={"Enregistrer l'élément"}, libelles=lib))
    try:
        CP._nouvel_element(etat)
    except Relance:
        pass
    attendu = OP.fiche_repartiteur(typ, nom, mat, fval, E, masse, src, A, B, t, prof)
    if js(etat["fiches"]["repartiteurs"]) != js([attendu]):
        fiches_ko.append((nom, etat["fiches"]["repartiteurs"], attendu))
chk("« Nouvel élément de répartition » : la fiche ajoutée par l'écran = operations.fiche_repartiteur (6 fiches, au bit près)", not fiches_ko, fiches_ko[:2])

import pandas as pd  # noqa: E402  (l'écran d'origine en a besoin pour son tableau d'appuis)
eng_ko = []
for nom, typ, lng, lrg, fl, appuis, src in [
        ("Grue X 12 t", "pieds", 5.5, 2.5, 4.0, [dict(a) for a in OP.APPUIS_FICHE_DEFAUT], "doc 2026 p. 4"),
        ("Nacelle Y", "roues", 2.5, 1.2, 0.0, [{"nom": "R1", "dx": 1.0, "dy": -0.5, "a": 0.2, "b": 0.25, "P": 22.5, "phi": 1.4},
                                             {"nom": "R2", "dx": -1.0, "dy": 0.5, "a": 0.2, "b": 0.25, "P": 18.0, "phi": 1.0}], ""),
        ("Chenillard", "chenilles", 4.0, 2.2, 1.5, [{"nom": "S1", "dx": 0.0, "dy": 0.0, "a": 0.4, "b": 0.4, "P": 80.0, "phi": 1.0}], "x")]:
    etat = ET.nouvel_etat()
    lib = {"Nom (modèle, constructeur)": nom, "Type d'appuis": typ, "Longueur châssis (m)": lng, "Largeur châssis (m)": lrg,
           "Flèche en projection (m)": fl, "Source (document, date, page)": src, "__appuis__": pd.DataFrame(appuis)}
    installer(FauxSt(Session(), boutons={"Enregistrer la fiche"}, libelles=lib))
    try:
        CP._dialogue_bibliotheque.__wrapped__(etat, ET.cas_actif(etat))
    except Relance:
        pass
    attendu = OP.fiche_engin(nom, typ, lng, lrg, fl, appuis, src)
    if js(etat["fiches"]["engins"]) != js([attendu]):
        eng_ko.append((nom, etat["fiches"]["engins"], attendu))
chk("« Nouvelle fiche d'engin » : la fiche ajoutée par l'écran = operations.fiche_engin (3 fiches, au bit près)", not eng_ko, eng_ko[:2])
chk("les défauts du formulaire d'engin sont ceux de l'écran (4 appuis 300 × 300 à ± 2,30 m, châssis 5,0 × 2,4, flèche 3,0)",
    OP.FICHE_ENGIN_DEFAUT == {"longueur": 5.0, "largeur": 2.4, "fleche": 3.0}
    and "{\"nom\": \"P1\", \"dx\": 2.3, \"dy\": -2.3, \"a\": 0.30, \"b\": 0.30, \"P\": 0.0, \"phi\": 1.0}".replace("\"", "\"") is not None
    and all(f'"nom": "{a["nom"]}", "dx": {a["dx"]}, "dy": {a["dy"]}' in open("modules/charges_plancher.py", encoding="utf-8").read()
            for a in OP.APPUIS_FICHE_DEFAUT))
src_ecran = open("modules/charges_plancher.py", encoding="utf-8").read()
chk("les listes de l'écran sont celles d'operations.py (types, états, familles, matériaux, rectangle)",
    OP.TYPES == CP.TYPES and OP.ETATS_H == CP.ETATS_H and OP.FAMILLES == CP.FAMILLES
    and f'["S235", "S355", "Bois C24", "Azobé", "OSB", "Autre"]' in src_ecran and OP.MATERIAUX == ["S235", "S355", "Bois C24", "Azobé", "OSB", "Autre"]
    and OP.RECTANGLE in src_ecran and '{"S235": 235.0, "S355": 355.0}' in src_ecran)


# ------------------------------------------------------------------ 4. ce que le site fait de plus : refuser, dire
print("\n=== 4. Les défauts de l'écran : reproduits par le moteur, refusés ou dits par le site ===")
e = W.nouvel_etat()
refus = {k: W.modifier(copy.deepcopy(e), ch, v) for k, ch, v in [
    ("portée < 0,5 m (l'écran la bornait en silence)", ["plancher", "L"], "0,2"),
    ("nombre d'éléments décimal (l'écran l'arrondissait)", ["plancher", "hourdis", "n_elements"], "7,5"),
    ("M_Rd négatif", ["plancher", "hourdis", "M_Rd"], "-10"),
    ("γQ nul", ["params", "gamma_Q"], "0"),
    ("part du tranchant > 1", ["params", "part_tranchant_element"], "1,2"),
    ("fck(t) sous la plus petite classe (l'écran plantait au calcul)", ["plancher", "dalle", "fck_t"], "5"),
    ("charge d'étançon illisible", ["plancher", "q_presentes"], "beaucoup")]}
chk("le site refuse, avec un message, les saisies que l'écran ramenait en silence à une valeur fausse", all(refus.values()),
    [k for k, v in refus.items() if not v])
e = W.exemple("hourdis")
avant = js(e)
W.modifier(e, ["plancher", "hourdis", "M_Rd"], "-5")
chk("une saisie refusée laisse la note intacte", js(e) == avant)
# total « kN au total » de l'écran : les charges linéaires et surfaciques n'y entrent pas (défaut reproduit et écrit)
e = W.exemple("hourdis")
cas = ET.cas_actif(e)
chk("défaut de l'écran reproduit : le total « kN » de l'étape Charges ignore les files d'étais et les stocks (250 kN : E1 + T1 seulement)",
    OP.total_charges(cas) == 250.0 and any(c["type"] == "surfacique" for c in cas["charges"]))
# proposition : une seconde proposition reprend l'identifiant d'une filière déjà appliquée (« F1p0 »)
base = W.exemple("hourdis")
W._faire(base, (["appuis", 4, "repartiteur"], "aucun"), (["filieres", 0, "-"], "1"), (["charges", 1, "P"], "120"))
W._faire(base, (["proposition", "+"], "1"), (["proposition", "appliquer"], "1"))
W._faire(base, (["appuis", 1, "repartiteur"], "aucun"), (["charges", 0, "P", 1], "70"))
c = ET.cas_actif(base)
ids_avant = [f["id"] for f in c["filieres"]]
r = W._calculer(base, c)
# l'écran d'origine : son bouton « Proposer », puis « Appliquer »
s_ = Session()
installer(FauxSt(s_))
CP._proposer(base, copy.deepcopy(c), r["res"], r["plancher"], r["index"])
p_ecran = s_[CP.P + "proposition"]
c_ecran = copy.deepcopy(c)
installer(FauxSt(Session({CP.P + "proposition": p_ecran}), boutons={CP.P + "btn_appliquer"}))
try:
    CP._carte_resultat(base, c_ecran, r["res"], r["plancher"], r["index"])
except Relance:
    pass
ids_ecran = [f["id"] for f in c_ecran["filieres"]]
site = copy.deepcopy(base)
W._faire(site, (["proposition", "+"], "1"), (["proposition", "appliquer"], "1"))
ids_site = [f["id"] for f in ET.cas_actif(site)["filieres"]]
chk("défaut de l'écran : une seconde proposition reprend l'identifiant d'une filière déjà appliquée (deux « F1p0 ») ; le site en donne un libre",
    p_ecran["trouve"] and len(set(ids_ecran)) < len(ids_ecran) and len(set(ids_site)) == len(ids_site) == len(ids_ecran),
    (ids_avant, ids_ecran, ids_site))
chk("… et le site applique la MÊME proposition (mêmes profilés, mêmes positions), seuls les identifiants changent",
    [(f["nom"], f["x"], f["y"], f["longueur"]) for f in ET.cas_actif(site)["filieres"]] == [(f["nom"], f["x"], f["y"], f["longueur"]) for f in c_ecran["filieres"]])


# ------------------------------------------------------------------ 5. mutations : une copie altérée d'operations.py rougit
print("\n=== 5. Mutations : chaque altération d'operations.py fait rougir la parité ===")
SOURCE_OP = open(os.path.join("plancher", "charges", "operations.py"), encoding="utf-8").read()


def module_mute(ancien, neuf):
    assert SOURCE_OP.count(ancien) == 1, ancien
    spec = importlib.util.spec_from_loader("plancher.charges._mutant", loader=None)
    m = importlib.util.module_from_spec(spec)
    m.__package__ = "plancher.charges"
    exec(compile(SOURCE_OP.replace(ancien, neuf), "operations_mutant.py", "exec"), m.__dict__)
    return m


def parite(op) -> bool:
    """La parité sur un échantillon, avec le module ``op`` à la place d'operations.py : True si tout concorde."""
    for nom, etat in CAS[:60]:
        f, vus, exc, site = memo[nom]
        if exc is not None or not vus["verifier"]:
            continue
        cas = ET.cas_actif(copy.deepcopy(etat))
        res = vus["verifier"][0]
        plancher, _, _, _, index = ET.construire(etat, cas)
        cap = textes(f, "caption")
        if op.resume_plancher(cas) not in cap or f"{len(cas['charges'])} objet(s) · {op.fmt(op.total_charges(cas), 0)} kN au total" not in cap:
            return False
        carte = next((b for b in textes(f, "markdown") if '<div class="chp-res">' in b), "")
        if [(lab, int(p)) for lab, p in RE_BARRE.findall(carte)] != [(lab, int(f"{t * 100:.0f}")) for _, lab, t in op.barres(res)]:
            return False
        for c in [c for c in cas["charges"] if c["type"] == "engin"]:
            vu = op.appuis_engin(cas, res, index, c["ref"])
            ref_ = OP.appuis_engin(cas, res, index, c["ref"])
            if js(vu) != js(ref_):
                return False
    # la proposition : celle que l'écran a faite (section 2), sur les premières notes non conformes
    for nom in list(PROP_ECRAN)[:6]:
        etat = dict(CAS)[nom]
        cas = ET.cas_actif(copy.deepcopy(etat))
        res = memo[nom][1]["verifier"][0]
        plancher, _, _, _, index = ET.construire(etat, cas)
        try:
            if js(PROP_ECRAN[nom]) != js(op.proposer_filiere(etat, copy.deepcopy(cas), res, plancher, index)):
                return False
        except Exception:  # noqa: BLE001
            return False
    for s_ in ["0:0.5, 1:0.25", "0:0,5", "2:0.1, 0:0.8"]:
        try:
            if js(op.lire_parts_annexe_c(s_)) != js(OP.lire_parts_annexe_c(s_)):
                return False
        except Exception:  # noqa: BLE001
            return False
    for args in [("plateau", "P", "S355", 0.0, 210000.0, 0.0, "", 600.0, 600.0, 25.0, OP.RECTANGLE),
                 ("filiere", "F", "Bois C24", 14.8, 11000.0, 8.0, "", 75.0, 225.0, 20.0, OP.RECTANGLE)]:
        if js(op.fiche_repartiteur(*args)) != js(OP.fiche_repartiteur(*args)):
            return False
    return True


chk("la parité d'échantillon passe avec operations.py tel quel (témoin des mutations)", parite(OP))
MUTATIONS = [
    ("seuil du défaut des éléments (> 100 % → > 90 %)", 'if e["taux"] > 1.0}', 'if e["taux"] > 0.9}'),
    ("identifiant des filières proposées (p0 → p1)", 'fid = ET.ref_libre(cas, "F") + f"p{len(fils)}"', 'fid = ET.ref_libre(cas, "F") + f"p{len(fils) + 1}"'),
    ("ordre des candidats (masse décroissante)", "BIB.candidats_filieres(3.6)]", "reversed(BIB.candidats_filieres(3.6))]"),
    ("table des nuances (S355 → 335 MPa)", '"S355": 355.0}', '"S355": 335.0}'),
    ("barres : l'attention n'est plus chiffrée", 'v["statut"] in ("ok", "attention", "nok")]', 'v["statut"] in ("ok", "nok")]'),
    ("total : les étançons oubliés", 'else c.get("P", 0.0) for c', 'else 0.0 for c'),
    ("résumé : la chape en cm", "h['chape']['epaisseur'] * 1e3:.0f} mm", "h['chape']['epaisseur'] * 1e2:.0f} mm"),
    ("annexe C : la virgule décimale acceptée", 'for morceau in s.replace(";", ",").split(","):', 'for morceau in s.replace(", ", ";").split(";"):'),
    ("appuis : l'élément pris à droite (y0 < y ≤ y1)", 'if x["y0"] <= e.y < x["y1"]', 'if x["y0"] < e.y <= x["y1"]'),
    ("répartiteur : plateau seul pris en compte", 'or (el and el.nom in defauts and cas["repartiteurs"].get(ref, {}).get("type") == "plateau")', ''),
]
for desc, ancien, neuf in MUTATIONS:
    try:
        mu = module_mute(ancien, neuf)
        rougit = not parite(mu)
    except Exception as e:  # noqa: BLE001 — une mutation qui casse le module rougit aussi
        rougit, desc = True, desc + f" ({type(e).__name__})"
    chk(f"mutation « {desc} » : la parité rougit", rougit)

installer(FauxSt())
print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

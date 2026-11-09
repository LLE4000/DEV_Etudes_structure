# -*- coding: utf-8 -*-
"""La NOTE PDF du module « Dalles béton armé » sur le site (api_web/note_dalle.py — S4, décision de l'ingénieur du
09/11/2026), les deux types (Dalle CSP, Prédalle + dalle CSP), vérifiée en Python ordinaire.

- PARITÉ : pour chaque cas (CSP et prédalle, une et plusieurs dalles et sections, treillis par nappe, treillis +
  renforts, gros Ø, moments sur les deux faces, axe saisi, centre de gravité imposé, verrouillée…), la planche que le
  générateur Streamlit calcule (modules/export_pdf_dalle._compute_section, sur les valeurs que le site traduit) rend les
  grandeurs du moteur du site — axes, y_G, d, A_s,req, A_s,min, A_s, états, h_u,min, h_min, τ, τ_adm — au 1e-9 près ;
- le TEXTE de chaque planche écrit ces chiffres (aux arrondis de la note : mm² entiers, cm et kN·m au dixième, N/mm² au
  centième), et l'écran les mêmes (A_s,néc, A_s, état du bloc) ;
- les nappes posées l'une sur l'autre passent (l'axe effectif du site, renfort secondaire S3 compris, et la nappe
  secondaire posée sur un renfort principal plus haut que sa base, S5 : le calcul, la planche, la coupe de l'écran et le
  PDF lu par pdftotext disent le même axe, le même d, le même A_s,req — une mutation le garde) ; la coupe dessine
  chaque couche à l'axe du calcul ; le verdict d'une nappe non conforme sous 100 % (acier excessif, d ≤ 0) dit pourquoi ;
  l'état « Données incomplètes » ; les observations de l'écran (avertissements, périmètre de la prédalle et de la dalle CSP) ;
- la traduction est l'inverse de la lecture d'un fichier Streamlit (beton/dalle/fichier.lire) : l'aller-retour garde
  chaque grandeur ;
- le PDF sort (garde + une planche par section, tous types), pdftotext y lit les noms et les chiffres ; le routeur le
  rend en base 64 ;
- la note se fait dans les conditions du navigateur (fontTools, accélérateurs C, streamlit, numpy, matplotlib refusés ;
  Pillow factice ; les seules polices du paquet) — et le contrôle dit ce qu'elle charge de plus que celle de la poutre ;
- MUTATIONS : une hauteur utile décalée, une dalle oubliée, l'axe empilé non transmis, l'état, les niveaux de la coupe,
  le verdict, les observations retirés — chacune fait rougir le contrôle qui la garde.

    /usr/bin/python3.12 tests/test_note_dalle.py
"""
import base64
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

from api_web import commun as C  # noqa: E402
from api_web import dalle as P  # noqa: E402
from api_web import dalle_positions as POS  # noqa: E402
from api_web import note_dalle as N  # noqa: E402
from api_web import routeur as R  # noqa: E402
from beton.dalle import entree as EN  # noqa: E402
from beton.dalle import fichier as FI  # noqa: E402
from beton.dalle import moteur as MO  # noqa: E402
from modules import export_pdf as EP  # noqa: E402
from modules import export_pdf_dalle as EPD  # noqa: E402
from ndc_pdf import data as ND  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:300]}"))


def proche(a, b, tol=1e-9):
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def cas(*saisies, base=None):
    """Une note saisie comme à l'écran (api_web/dalle.modifier) ; chaque saisie doit passer."""
    e = copy.deepcopy(base) if base else P.nouvel_etat()
    for chemin, valeur in saisies:
        msg = P.modifier(e, chemin, valeur)
        assert msg is None, (chemin, valeur, msg)
    return e


S0 = ["sections", 0]


def F(f):
    return S0 + ["faces", f, "couches"]


CAS = {
    "CSP neuve (sans effort)": cas(),
    "CSP · M_y,inf = 28 (treillis par défaut)": cas((S0 + ["My_inf"], 28)),
    "CSP · moments sur les deux faces et les deux directions, V": cas(
        (S0 + ["Mx_inf"], 22), (S0 + ["Mx_sup"], 14), (S0 + ["My_inf"], 35), (S0 + ["My_sup"], 18), (S0 + ["V"], 60)),
    "CSP · principale X": cas((["dir_principale"], "X"), (S0 + ["Mx_inf"], 30), (S0 + ["My_inf"], 12), (S0 + ["V"], 25)),
    "CSP · treillis par nappe 8/10/100/150 en bas, 6/6/150/150 en haut": cas(
        (S0 + ["nappes", "inf", "treillis"], "8/10/100/150"), (S0 + ["nappes", "sup", "treillis"], "6/6/150/150"),
        (S0 + ["My_inf"], 24), (S0 + ["Mx_inf"], 15), (S0 + ["My_sup"], 8)),
    "CSP · treillis + renforts, renfort secondaire empilé (S3)": cas(
        (S0 + ["nappes", "inf", "mode"], "treillis_renforts"), (F("inf_x") + ["+"], 1), (S0 + ["My_inf"], 40), (S0 + ["Mx_inf"], 25)),
    "CSP · gros Ø (Ø25/150 en bas, Ø20/200 en haut), h = 30": cas(
        (["h"], 30), (S0 + ["nappes", "inf", "mode"], "barres"), (F("inf_y") + [0, "d"], 25), (F("inf_x") + [0, "d"], 25),
        (S0 + ["nappes", "sup", "mode"], "barres"), (F("sup_y") + [0, "d"], 20), (F("sup_y") + [0, "esp"], 200), (F("sup_x") + [0, "d"], 20),
        (S0 + ["My_inf"], 120), (S0 + ["My_sup"], 60), (S0 + ["Mx_inf"], 70)),
    "CSP · trois renforts sur une face (n barres, barres, treillis)": cas(
        (F("inf_y") + ["+"], 1), (F("inf_y") + ["+"], 1), (F("inf_y") + [2, "type"], "barres"), (F("inf_y") + [2, "d"], 16),
        (F("inf_y") + ["+"], 1), (F("inf_y") + [3, "type"], "treillis"), (F("inf_y") + [3, "treillis"], "8/8/150/150"),
        (S0 + ["My_inf"], 75), (S0 + ["Mx_inf"], 20)),
    "CSP · axe saisi et centre de gravité imposé": cas(
        (F("inf_y") + ["+"], 1), (F("inf_y") + [1, "dist"], "6,5"), (S0 + ["faces", "sup_x", "yG"], "4,2"), (S0 + ["My_inf"], 30),
        (S0 + ["Mx_sup"], 10)),
    "CSP · enrobage nul (lu 3 cm) et jeu premier lit 0,5": cas((["enrobage"], 0), (["jeu_premier_lit"], "0,5"), (S0 + ["My_inf"], 26)),
    "CSP · B400, γs 1,15, C20/25, bande de 50 cm": cas(
        (["fyk"], "400"), (["gamma_s"], "1,15"), (["beton"], "C20/25"), (["b"], 50), (S0 + ["My_inf"], 14), (S0 + ["V"], 20)),
    "CSP · acier excessif (Ø40/100 en bas)": cas(
        (S0 + ["nappes", "inf", "mode"], "barres"), (F("inf_y") + [0, "d"], 40), (F("inf_y") + [0, "esp"], 100), (S0 + ["My_inf"], 20)),
    "CSP · d utile ≤ 0 (axe saisi au-delà de h)": cas((F("inf_y") + [0, "dist"], "21"), (S0 + ["Mx_sup"], 10)),
    "CSP · hauteur insuffisante et τ > τ_adm,I": cas((S0 + ["My_inf"], 160), (S0 + ["V"], 400)),
    "CSP · nappes X et Y au même niveau (axe saisi)": cas((F("inf_x") + [0, "dist"], "3,5"), (S0 + ["My_inf"], 20), (S0 + ["Mx_inf"], 15)),
    "CSP · trois sections": cas(
        (S0 + ["My_inf"], 20), (["sections", "+"], 1), (["sections", 1, "My_sup"], 32), (["sections", 1, "V"], 45),
        (["sections", "+"], 1), (["sections", 2, "Mx_inf"], 18)),
    "CSP · verrouillée": cas((S0 + ["My_inf"], 28), (S0 + ["Mx_sup"], 9), (["verrou"], True)),
    "Prédalle · neuve sollicitée": cas((["type"], "predalle"), (S0 + ["My_inf"], 28), (S0 + ["Mx_inf"], 12), (S0 + ["V"], 40)),
    "Prédalle · renforts en bas et en haut, principale X": cas(
        (["type"], "predalle"), (["dir_principale"], "X"), (F("inf_x") + ["+"], 1), (F("sup_y") + ["+"], 1), (S0 + ["Mx_inf"], 30),
        (S0 + ["My_sup"], 14), (S0 + ["V"], 30)),
    "Prédalle · peau de 4 cm, base Ø16 qui en sort": cas(
        (["type"], "predalle"), (["h_pre"], 4), (S0 + ["nappes", "inf", "mode"], "barres"), (F("inf_y") + [0, "d"], 16), (S0 + ["My_inf"], 25)),
    "Prédalle · couches croisées sur la prédalle (renfort Y, base X)": cas(
        (["type"], "predalle"), (F("inf_y") + ["+"], 1), (S0 + ["My_inf"], 30), (S0 + ["Mx_inf"], 10)),
    "Prédalle · gros Ø en haut (renfort 4 Ø20 secondaire empilé)": cas(
        (["type"], "predalle"), (["h"], 25), (F("sup_x") + ["+"], 1), (F("sup_x") + [1, "n"], 4), (F("sup_x") + [1, "d"], 20),
        (S0 + ["Mx_sup"], 40), (S0 + ["My_sup"], 30), (S0 + ["My_inf"], 22)),
}
e2 = cas((S0 + ["My_inf"], 28), (S0 + ["V"], 30))
P.gerer_elements(e2, "ajouter")
CAS["deux dalles : CSP puis prédalle"] = cas((["type"], "predalle"), (S0 + ["Mx_inf"], 20), (["sections", "+"], 1),
                                             (["sections", 1, "My_sup"], 16), base=e2)
e3 = copy.deepcopy(CAS["deux dalles : CSP puis prédalle"])
e3["dalles"][1]["gamma_s"] = 1.15                                     # une note relue (un .stab) : γs propre à la dalle
CAS["deux dalles, γs différents (note relue)"] = P.normaliser(e3)
CAS["deux dalles, γs différents (note relue)"].pop(P.CLE_AVIS, None)

BETON = N._beton()
N_SECTIONS = sum(len(d["sections"]) for e in CAS.values() for d in e["dalles"])


# ---------------------------------------------------------------- 1. la PARITÉ : le calcul de la planche = le moteur du site
def ecarts_parite(etat, valeurs=None):
    """Les écarts (planche Streamlit ↔ moteur du site) d'une note : le contrôle INDÉPENDANT du test (il ne lit pas
    ``N.ecarts``). ``valeurs`` : la traduction employée (``N.valeurs`` par défaut ; les mutations la remplacent)."""
    valeurs = valeurs or N.valeurs
    out = []
    for di, d in enumerate(etat["dalles"]):
        r = P._calculer(d)
        dalles, vals = valeurs(d, di + 1)
        for sj, rs in enumerate(r["sections"]):
            Rp = EPD._compute_section(vals, BETON, di + 1, sj + 1)
            g = rs["grandeurs"]
            paires = [("hu_min", Rp["hmin"], g["hu_min"]), ("h_min", Rp["h_min_dalle"], g["h_min"]),
                      ("As_max", Rp["As_max"], g["As_max"]), ("fyd", Rp["fyd"], g["fyd"])]
            for f in EN.FACES:
                cote, dk = f.split("_")
                D = Rp["dirs"][dk]
                geo = D[f"geo_{cote}"]
                paires += [(f"yG_{f}", geo["e_cdg"], g[f"yG_{f}"]), (f"d_{f}", D["di" if cote == "inf" else "ds"], g[f"d_{f}"]),
                           (f"As_req_{f}", D[f"As_req_{cote}"], g[f"As_req_{f}"]), (f"As_min_{f}", D[f"As_min_{cote}"], g[f"As_min_{f}"]),
                           (f"As_{f}", D[f"As_{cote}"], g[f"As_{f}"]),
                           (f"besoin_{f}", max(D[f"As_req_{cote}"], D[f"As_min_{cote}"]), g[f"besoin_{f}"])]
                paires += [(f"axe {f} {j}", c["e"], c_s["e"]) for j, (c, c_s) in enumerate(zip(geo["couches"], rs["faces"][f]["couches"]))]
                if len(geo["couches"]) != len(rs["faces"][f]["couches"]) or D[f"etat_{cote}"] != g[f"etat_{f}"]:
                    out.append((d["nom"], rs["nom"], f, "couches ou état"))
            Sh = Rp["shear"]
            if g["V"] > 0:
                paires += [("tau", Sh["tau"], g["tau"]), ("tau_adm", Sh["tau_adm"], g["tau_adm"])]
            elif Sh is not None:
                out.append((d["nom"], rs["nom"], "τ sans V"))
            out += [(d["nom"], rs["nom"], k, a, b) for k, a, b in paires if not proche(a, b)]
            if Rp["etat_global"] != rs["statut"] or Rp["etat_h"] != g["etat_hauteur"]:
                out.append((d["nom"], rs["nom"], "statut", Rp["etat_global"], rs["statut"]))
    return out


ecarts = []
for nom, e in CAS.items():
    ecarts += [(nom,) + x for x in ecarts_parite(e)]
chk(f"parité : {len(CAS)} cas, {N_SECTIONS} sections — la planche Streamlit rend les axes, y_G, d, A_s,req, A_s,min, A_s, "
    "états, h_min, τ du moteur du site (au 1e-9 près)", not ecarts and len(CAS) >= 15 and N_SECTIONS >= 25, ecarts[:4])
n_couches = sum(len(s["faces"][f]["couches"]) for e in CAS.values() for d in e["dalles"] for s in d["sections"] for f in EN.FACES)
n_empilees = sum(POS.empilee(d, s, f, j) for e in CAS.values() for d in e["dalles"] for s in d["sections"] for f in EN.FACES
                 for j in range(len(s["faces"][f]["couches"])))
chk(f"… dont {n_empilees} couches posées par le site sur la nappe principale (sur {n_couches}) : l'empilement est couvert",
    n_empilees >= 40, n_empilees)
refus = []
for nom, e in CAS.items():
    try:
        N.resultats(e)
    except ValueError as err:
        refus.append((nom, str(err)))
chk("le contrôle de la note (N.ecarts) laisse passer chaque cas : aucune note refusée", not refus, refus[:2])


# ---------------------------------------------------------------- 2. le TEXTE de la planche écrit les chiffres de l'écran
def lire_planches(etat):
    """(sections mises en forme, planches) d'une note."""
    return N.sections(etat), N.resultats(etat)


def textes_planche(sec):
    """Tout le texte d'une planche : formules, lignes, verdicts, données."""
    out = []
    for v in sec["verifs"]:
        out.append(v["titre"])
        for it in v["items"]:
            out += [str(x) for x in it[1:]]
        out += [k["texte"] for k in v["verdicts"]]
    for _, rows in sec["blocs"]:
        out += [" ".join(str(x) for x in row if x) for row in rows]
    return "\n".join(out)


manques, n_chiffres = [], 0
for nom, e in CAS.items():
    secs, pls = lire_planches(e)
    for sec, pl in zip(secs, pls):
        rs, g = pl["site"], pl["site"]["grandeurs"]
        par_cle = {v["cle"]: v for v in rs["verifications"]}
        p = rs["principale"]
        # 1 hauteur : h_u,min et h_min au dixième
        v1 = sec["verifs"][0]
        attendus = [(v1, rf"\res{{{ND.fn(g['hu_min'], 1)} \u{{cm}}}}"), (v1, rf"\res{{{ND.fn(g['h_min'], 1)} \u{{cm}}}}")]
        # 2 à 5 : chaque nappe, dans l'ordre de l'écran
        for v_pdf, f in zip(sec["verifs"][1:5], MO.ordre_faces(p)):
            cote, dk = f.split("_")
            if not v_pdf["items"][0][1].endswith(f" {dk.upper()}") or ("inférieur" in v_pdf["items"][0][1]) != (cote == "inf"):
                manques.append((nom, rs["nom"], f, "ordre", v_pdf["items"][0][1]))
            attendus += [(v_pdf, rf"\res{{{ND.fn(g[f'd_{f}'], 1)} \u{{cm}}}}"), (v_pdf, rf"\res{{{ND.fn(g[f'As_req_{f}'], 0)} \u{{mm}}^{{2}}}}"),
                         (v_pdf, rf"\res{{{ND.fn(g[f'As_min_{f}'], 0)} \u{{mm}}^{{2}}}}"), (v_pdf, f"({ND.fn(g[f'As_pm_{f}'], 0)} mm²/m)")]
            verdict = v_pdf["verdicts"][0]["texte"].replace(C.NBSP, " ")
            if par_cle[f]["etat"] == MO.OK:
                attendus_v = [f"{ND.fn(g[f'As_{f}'], 0)} mm² ≥", f"{ND.fn(g[f'besoin_{f}'], 0)} mm²"]
            elif par_cle[f]["taux"] > 1:
                attendus_v = [f"{ND.fn(g[f'As_{f}'], 0)} mm² <", f"{ND.fn(g[f'besoin_{f}'], 0)} mm²"]
            else:
                attendus_v = []                      # acier excessif, d ≤ 0 : contrôlés plus bas
            manques += [(nom, rs["nom"], f, t, verdict) for t in attendus_v if t not in verdict]
            if (v_pdf["verdicts"][0]["etat"] == "ok") != (par_cle[f]["etat"] == MO.OK):
                manques.append((nom, rs["nom"], f, "couleur du verdict"))
            n_chiffres += 4 + len(attendus_v)
        if g["V"] > 0:
            v6 = sec["verifs"][5]
            attendus += [(v6, rf"\res{{{ND.fn(g['tau'], 2)} \u{{N/mm}}^{{2}}}}")]
            if ND.fn(g["tau_adm"], 2) not in v6["items"][1][3]:
                manques.append((nom, rs["nom"], "tau_adm"))
        for v_pdf, t in attendus:
            n_chiffres += 1
            if t not in "\n".join(str(it[2]) for it in v_pdf["items"] if len(it) > 2) + "\n".join(str(it[1]) for it in v_pdf["items"]):
                manques.append((nom, rs["nom"], v_pdf["titre"], t))
        soll = dict((row[1], row[2]) for row in sec["blocs"][2][1])
        for k, sym in (("Mx_inf", "M_{x,inf}"), ("Mx_sup", "M_{x,sup}"), ("My_inf", "M_{y,inf}"), ("My_sup", "M_{y,sup}"), ("V", "V_{max}")):
            n_chiffres += 1
            if soll.get(sym) != ND.fn(rs[k], 1):
                manques.append((nom, rs["nom"], k, soll.get(sym), rs[k]))
chk(f"le texte de chaque planche écrit les chiffres du moteur du site ({n_chiffres} chiffres : h_u,min, h_min, d, A_s,req, "
    "A_s,min, A_s/m, A_s, A_s,néc, τ, τ_adm, efforts), dans l'ordre des nappes de l'écran", not manques, manques[:4])

# l'écran affiche les mêmes chiffres (la ligne de chaque nappe : A_s,néc et A_s, et l'état du bloc), aux mêmes arrondis
ecran, ecarts_arrondi, n_ecran = [], [], 0
ETATS_ECRAN = {"Conforme": "Vérifié", "Non conforme": "Non vérifié", "Données incomplètes": N.ETAT_INCOMPLET}
BLOC = re.compile(r'<section class="et-bloc (\w+)" data-section="(\d+)">.*?<span class="e">([^<]+)</span>', re.S)
for nom, e in CAS.items():
    for i in range(len(e["dalles"])):
        ee = copy.deepcopy(e)
        ee["actif"] = i
        res = P.resultat(ee)
        html_ = res["resultat"].replace(C.FINE, " ")
        secs = N.sections(ee)
        debut = sum(len(d["sections"]) for d in e["dalles"][:i])
        etats_ecran = [x[2] for x in BLOC.findall(res["resultat"])]
        for k, s in enumerate(P._calculer(e["dalles"][i])["sections"]):
            sec = secs[debut + k]
            if ETATS_ECRAN.get(etats_ecran[k]) != sec["etat"]:
                ecran.append((nom, s["nom"], etats_ecran[k], sec["etat"]))
            for v in s["verifications"]:
                if v["cle"] not in EN.FACES or P._sous_limite_non_conforme(v):
                    continue                      # acier excessif, d ≤ 0 : leur comparaison est contrôlée plus bas
                for x in (v["valeur"], v["limite"]):
                    n_ecran += 1
                    if f"{C.fr(x, 0).replace(C.FINE, ' ')}" not in html_:
                        ecran.append((nom, s["nom"], v["cle"], x))
                    if C.fr(x, 0).replace(C.FINE, "") != ND.fn(x, 0):
                        ecarts_arrondi.append((nom, v["cle"], x))
chk(f"l'écran montre les mêmes A_s,néc et A_s ({n_ecran} chiffres) et le même état de chaque section que la planche",
    not ecran, ecran[:3])
chk("… écrits pareil : aucun chiffre n'est arrondi autrement dans la note (C.fr, demi vers l'extérieur ; fn, demi au pair : "
    "ils ne diffèrent que sur une valeur binaire exactement à mi-chemin)", not ecarts_arrondi, ecarts_arrondi[:3])


# ---------------------------------------------------------------- 3. ce que le site ajoute : treillis par nappe, empilement, coupe, verdict, état
def planche(nom, k=0):
    secs, pls = lire_planches(CAS[nom])
    return secs[k], pls[k]


sec, pl = planche("CSP · treillis par nappe 8/10/100/150 en bas, 6/6/150/150 en haut")
t = textes_planche(sec)
chk("treillis par nappe : la planche dit le treillis de chaque direction (8/10/100/150 en Y, sa transposée 10/8/150/100 en X)",
    "On prend Treillis 8/10/100/150 (503 mm²/m)" in t and "On prend Treillis 10/8/150/100 (524 mm²/m)" in t, t[:200])

nom_s3 = "CSP · treillis + renforts, renfort secondaire empilé (S3)"
d_s3 = CAS[nom_s3]["dalles"][0]
sec, pl = planche(nom_s3)
axes = [c["e"] for c in pl["R"]["dirs"]["x"]["geo_inf"]["couches"]]
chk("S3 et S5 : la direction secondaire (inf. X) est comptée aux axes empilés du site — posée sur le renfort 3 Ø12 que « Treillis + "
    "renforts » met sur la principale (axe 4,0, dessus 5,0, plus haut que le treillis Ø10 : S5) : sa base à 5,0 + 0,5 = 5,5, son renfort "
    "(S3) à 5,0 + 1,0 = 6,0 cm (avant S5 : 4,5 et 5,0, au travers du renfort Y)",
    axes == POS.axes(d_s3, d_s3["sections"][0])["inf_x"] == [5.5, 6.0], axes)
# S5 de bout en bout : un renfort principal 3 Ø25 sous la nappe secondaire — le calcul, la planche, la coupe et le PDF disent la même
# chose (la nappe X sur le renfort Y : axe 6,5, d = 13,5 cm), et pas la base seule (4,5 : d = 15,5)
nom_s5 = "CSP · treillis + renfort principal 3 Ø25 (S5)"
e_s5 = cas((S0 + ["nappes", "inf", "mode"], "treillis_renforts"), (F("inf_y") + [1, "d"], 25), (S0 + ["My_inf"], 30), (S0 + ["Mx_inf"], 20))


def s5_bout_en_bout():
    """Le calcul, la planche, la coupe de l'écran et le PDF (pdftotext) de la note S5 : (passe, ce qu'on a lu)."""
    r_ = P._calculer(e_s5["dalles"][0])
    g_ = r_["sections"][0]["grandeurs"]
    secs_, pls_ = lire_planches(e_s5)
    v_ = next(v for v in secs_[0]["verifs"][1:5] if v["items"][0][1].endswith(" X") and "inférieur" in v["items"][0][1])
    txt_ = "\n".join(str(it[2]) for it in v_["items"] if len(it) > 2) + "\n".join(str(it[1]) for it in v_["items"])
    tmp_ = tempfile.mkdtemp()
    with open(os.path.join(tmp_, "s5.pdf"), "wb") as fh:
        fh.write(N.pdf(e_s5, {"nom_projet": nom_s5}))
    pdf_ = subprocess.run(["pdftotext", "-layout", os.path.join(tmp_, "s5.pdf"), "-"], capture_output=True, text=True).stdout
    shutil.rmtree(tmp_, ignore_errors=True)
    leg_ = re.findall(r"X · Treillis 10/10/100/100 · axe à \d+ · sur [^<]+", P.resultat(copy.deepcopy(e_s5))["vues"][0]["svg"])
    e_ = r_["sections"][0]["faces"]["inf_x"]["couches"][0]["e"]
    passe = (e_ == 6.5 and g_["d_inf_x"] == 13.5 and round(g_["As_req_inf_x"]) == 494
             and [c["e"] for c in pls_[0]["R"]["dirs"]["x"]["geo_inf"]["couches"]] == [6.5] and not ecarts_parite(e_s5)
             and rf"\res{{{ND.fn(13.5, 1)} \u{{cm}}}}" in txt_ and rf"\res{{494 \u{{mm}}^{{2}}}}" in txt_
             and rf"\res{{{ND.fn(15.5, 1)} \u{{cm}}}}" not in txt_ and rf"\res{{430 \u{{mm}}^{{2}}}}" not in txt_
             and leg_.count("X · Treillis 10/10/100/100 · axe à 65 · sur renfort Y") == 2              # inf. X, dans les deux coupes
             and set(leg_) == {"X · Treillis 10/10/100/100 · axe à 65 · sur renfort Y",
                               "X · Treillis 10/10/100/100 · axe à 45 · sur la base Y"}              # sup. X : pas de renfort sup. Y
             and "13,5 cm" in pdf_ and "494 mm" in pdf_)
    return passe, (e_, g_["d_inf_x"], round(g_["As_req_inf_x"], 1), leg_, txt_[:200])


passe5, lu5 = s5_bout_en_bout()
chk("S5 de bout en bout (renfort principal 3 Ø25 en inf. Y, M_x,inf = 20) : le calcul pose la base inf. X sur le renfort (axe 6,5, d = 13,5 cm, "
    "A_s,req = 494 mm²) ; la planche compte la couche à 6,5 et écrit d = 13,5 cm et 494 mm² (et non 15,5 cm, 430 mm² de la base seule) ; "
    "la coupe de l'écran dit « axe à 65 · sur renfort Y » ; pdftotext lit 13,5 cm et 494 mm dans le PDF", passe5, lu5)


def niveaux_coupe(sec, R_):
    """(niveau dessiné, axe du calcul) de chaque couche de chaque schéma (mm)."""
    p = R_["principale"]
    out = []
    for sch, dk in zip(sec["coupe"]["schemas"], [p, MO.direction_secondaire(p)]):
        for cle, dd in (("points", dk), ("filants", MO.direction_secondaire(dk))):
            for face in ("inf", "sup"):
                for c_d, c in zip(sch[f"{cle}_{face}"], R_["dirs"][dd][f"geo_{face}"]["couches"]):
                    out.append((round(c_d["e"], 6), round(c["e"] * 10.0, 6)))
    return out


def coupe_juste(nom):
    sec, pl = planche(nom)
    return all(a == b for a, b in niveaux_coupe(sec, pl["R"]))


faux = [n for n in CAS if not coupe_juste(n)]
chk("la coupe dessine chaque couche (base et renforts, coupée ou filante) à l'axe du calcul — tous les cas", not faux, faux[:3])
brut = ND.construire_sections_dalle(N.resultats(CAS[nom_s3]))[0]
chk("… ce que le générateur seul ne fait pas : il écartait le renfort de sa base pour le dessin (le cas S3 le montre)",
    any(a != b for a, b in niveaux_coupe(brut, pl["R"])), niveaux_coupe(brut, pl["R"]))

sec, pl = planche("CSP · acier excessif (Ø40/100 en bas)")
vd = sec["verifs"][1]["verdicts"][0]
vb = ND.construire_sections_dalle(N.resultats(CAS["CSP · acier excessif (Ø40/100 en bas)"]))[0]["verifs"][1]["verdicts"][0]
chk("acier excessif : le verdict dit A_s > A_s,max (le générateur seul écrirait « A_s < requise », faux)",
    vd["etat"] == "ko" and "12566 mm² > section d'armature maximale : 8000 mm² (acier excessif)" in vd["texte"].replace(C.NBSP, " ")
    and "<" in vb["texte"], (vd, vb["texte"]))
sec, pl = planche("CSP · d utile ≤ 0 (axe saisi au-delà de h)")
vd = sec["verifs"][1]["verdicts"][0]
chk("d utile ≤ 0 : le verdict dit d = −1,0 cm ≤ 0 (position des couches incompatible), en rouge",
    vd["etat"] == "ko" and "d = -1,0 cm ≤ 0" in vd["texte"].replace(C.NBSP, " "), vd)

sec, pl = planche("CSP neuve (sans effort)")
chk("une section sans aucun effort n'est jamais « Vérifié » : « Données incomplètes » (l'écran aussi)",
    sec["etat"] == N.ETAT_INCOMPLET and pl["R"]["etat_global"] == "ok", sec["etat"])
sec, pl = planche("CSP · hauteur insuffisante et τ > τ_adm,I")
chk("hauteur insuffisante et τ > τ_adm,I : « Non vérifié », verdicts rouges (hauteur, τ)",
    sec["etat"] == "Non vérifié" and sec["verifs"][0]["verdicts"][0]["etat"] == "ko" and sec["verifs"][5]["verdicts"][0]["etat"] == "ko",
    sec["etat"])
sec, pl = planche("Prédalle · neuve sollicitée")
dims = dict((row[0], row[2]) for row in sec["blocs"][0][1])
chk("prédalle : DIMENSIONS dit la peau et le coulé en place (6 + 14 cm), la coupe la teinte",
    dims.get("dont prédalle préf.") == "6" and dims.get("dont coulé en place") == "14" and sec["coupe"]["h_pre"] == 60.0, dims)

# ---------------------------------------------------------------- 4. les OBSERVATIONS de l'écran
OBS = {
    "Prédalle · neuve sollicitée": ["Non vérifiés, comme dans l'écran Streamlit : la phase provisoire"],
    "Prédalle · peau de 4 cm, base Ø16 qui en sort": ["sort de la peau de 4,0 cm", "Non vérifiés"],
    "Prédalle · couches croisées sur la prédalle (renfort Y, base X)": ["à des niveaux trop proches pour se croiser", "Non vérifiés"],
    "CSP · enrobage nul (lu 3 cm) et jeu premier lit 0,5": ["Enrobage nul : comme l'écran Streamlit, le calcul pose les couches avec un enrobage de 3 cm."],
    "CSP · nappes X et Y au même niveau (axe saisi)": ["les nappes X et Y inférieures sont calculées au même niveau", "Rendre l'axe saisi à l'automatique"],
    "CSP · d utile ≤ 0 (axe saisi au-delà de h)": ["d utile ≤ 0"],
}
faux = []
for nom, attendus in OBS.items():
    sec, pl = planche(nom)
    obs = " | ".join(N.observations_de(sec))
    faux += [(nom, a, obs[:120]) for a in attendus if a not in obs]
    if "<" in obs or "&" in obs:
        faux.append((nom, "balisage", obs[:120]))
# (petits chantiers, 10/11/2026) une dalle CSP dit AUSSI ce qui n'est pas vérifié (flèche, fissuration) — son périmètre,
# sur sa première planche — : une dalle CSP sans avertissement n'a QUE lui
_csp = N._texte(P.PERIMETRE_CSP)
sans = [n for n in CAS if n not in OBS and not n.startswith(("Prédalle", "deux dalles")) and N.observations_de(planche(n)[0]) != [_csp]]
chk("les observations de la planche sont celles que l'écran affiche sous la section (avertissements, périmètre de la "
    "prédalle ou de la dalle CSP), en texte ; une dalle CSP sans avertissement n'a que son périmètre (flèche, fissuration non vérifiées)",
    not faux and not sans, (faux[:2], sans[:3]))
# le périmètre : une source pour l'écran et la note (api_web/dalle.texte_perimetre)
r_p = P.resultat(copy.deepcopy(CAS["Prédalle · neuve sollicitée"]))
_d_csp = CAS["CSP · M_y,inf = 28 (treillis par défaut)"]
r_c = P.resultat(copy.deepcopy(_d_csp))
chk("le périmètre de la prédalle, et celui de la dalle CSP (flèche, fissuration non vérifiées), sont le texte de l'écran "
    "(api_web/dalle.texte_perimetre, une source pour les deux) — dans le résultat et sur la première planche de la note",
    P.texte_perimetre(CAS["Prédalle · neuve sollicitée"]["dalles"][0]) in r_p["resultat"]
    and N._texte(P.texte_perimetre(CAS["Prédalle · neuve sollicitée"]["dalles"][0])) in N.observations_de(planche("Prédalle · neuve sollicitée")[0])
    and P.texte_perimetre(_d_csp["dalles"][0]) == P.PERIMETRE_CSP and "la flèche et la fissuration" in P.PERIMETRE_CSP
    and P.PERIMETRE_CSP in r_c["resultat"] and _csp in N.observations_de(planche("CSP · M_y,inf = 28 (treillis par défaut)")[0]), "")


# ---------------------------------------------------------------- 5. le DOCUMENT : chaque planche tient, en deux colonnes
def construire(etat):
    fd, chemin = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    try:
        return N.document(etat, {}, chemin)
    finally:
        os.remove(chemin)


debordes, trois, replis, reduites = [], [], [], []
for nom, e in CAS.items():
    d_ = construire(e)
    if d_.warnings:
        debordes.append((nom, d_.warnings))
    if d_.colonnes != 2:
        trois.append(nom)
    if d_.replis:
        replis.append(nom)
    if d_.coupes_reduites:
        reduites.append(nom)
chk("chaque planche de chaque cas tient sur sa page (rien ne déborde)", not debordes, debordes[:2])
chk(f"… en DEUX colonnes, les observations sous les données de la coupe ({len(reduites)} cas où la coupe leur cède de la "
    f"place : {', '.join(reduites)}) ; aucune en fin de planche ni en trois colonnes", not trois and not replis, (trois, replis))
# MESURER CE QUE L'ŒIL VOIT : sous ses observations, la coupe garde la place de ses étiquettes — aucun texte de la coupe
# n'en chevauche un autre (on espionne chaque texte écrit par la coupe, mesuré dans sa police et son corps ; les cotes
# verticales, écrites dans un repère tourné, sont à part). Balayage : épaisseurs de 12 à 60 cm, prédalle et CSP, de 0 à
# 4 observations, une coupe à un schéma (directions identiques) et à deux.
import ndc_pdf.section as SEC  # noqa: E402


def chevauchements_coupe(etat, n_obs):
    boites, vrai_txt, vrai_obs = [], SEC.draw_text, N.observations

    def espion(c, x, y, txt, font, size, color=None, align="left", **k):
        lw = vrai_txt(c, x, y, txt, font, size, color, align, **k)
        if boites and boites[-1][0] is not c:              # un nouveau document (le repli reconstruit la note)
            boites.clear()
        if abs(x) > 1.0:                                   # (0, y) : une cote verticale, dans un repère tourné
            g = x - (lw if align == "right" else (lw / 2 if align == "center" else 0))
            boites.append((c, txt, g, g + lw, y - 0.22 * size, y + 0.78 * size))
        return lw
    SEC.draw_text = espion
    N.observations = lambda d, r, i: ([P.texte_perimetre(d) or "Observation d'essai, assez longue pour tenir sur deux lignes "
                                       "du panneau de gauche de la planche, sous la coupe de la section."] * n_obs)
    try:
        d_ = construire(etat)
    finally:
        SEC.draw_text, N.observations = vrai_txt, vrai_obs
    out = [(a[1], b[1]) for i, a in enumerate(boites) for b in boites[i + 1:]
           if a[2] < b[3] - 0.5 and b[2] < a[3] - 0.5 and a[4] < b[5] - 0.5 and b[4] < a[5] - 0.5]
    return out, d_


faux, n_essais, n_reduites, n_replis, n_trois = [], 0, 0, 0, 0
for type_ in ("coulee", "predalle"):
    for h in (12, 20, 30, 45, 60):
        base_ = cas((["type"], type_), (["h"], h), (S0 + ["My_inf"], 20), (S0 + ["Mx_sup"], 6), (S0 + ["V"], 30),
                    (F("sup_x") + ["+"], 1))
        for n_obs in range(5):
            n_essais += 1
            ch_, d_ = chevauchements_coupe(base_, n_obs)
            n_reduites, n_replis, n_trois = n_reduites + d_.coupes_reduites, n_replis + d_.replis, n_trois + (d_.colonnes == 3)
            if ch_ or d_.warnings:
                faux.append((type_, h, n_obs, ch_[:2], d_.warnings))
une = cas((F("inf_x") + [0, "dist"], "3,5"), (F("sup_x") + [0, "dist"], "3,5"), (S0 + ["My_inf"], 20))   # X et Y au même niveau
chk("… une coupe à UN schéma (directions identiques, axes saisis)", len(N.sections(une)[0]["coupe"]["schemas"]) == 1)
for n_obs in range(5):
    n_essais += 1
    ch_, d_ = chevauchements_coupe(une, n_obs)
    n_reduites, n_replis, n_trois = n_reduites + d_.coupes_reduites, n_replis + d_.replis, n_trois + (d_.colonnes == 3)
    if ch_ or d_.warnings:
        faux.append(("un schéma", 20, n_obs, ch_[:2], d_.warnings))
chk(f"sous ses observations, aucun texte de la coupe n'en chevauche un autre, et rien ne déborde ({n_essais} planches : "
    f"h de 12 à 60 cm, CSP et prédalle, 0 à 4 observations, un ou deux schémas ; {n_reduites} coupes réduites, "
    f"{n_replis} observations en fin de planche, {n_trois} planches en trois colonnes)",
    not faux and n_reduites > 0 and n_replis > 0, faux[:3])


def textes_coupe(secs):
    """Les textes écrits par la coupe (texte, x, y) pendant la construction des planches ``secs``."""
    vus, vrai_txt = [], SEC.draw_text

    def espion(c, x, y, txt, *a, **k):
        vus.append((txt, round(x, 3), round(y, 3)))
        return vrai_txt(c, x, y, txt, *a, **k)
    SEC.draw_text = espion
    try:
        fd, chemin = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        N._style(2).build(chemin, sections=secs, doc=ND.construire_doc({}, "x"))
        os.remove(chemin)
    finally:
        SEC.draw_text = vrai_txt
    return vus


secs_a = N.sections(CAS[nom_s3])
secs_b = N.sections(CAS[nom_s3])
for s_ in secs_b:
    s_["dessin"] = N._hauteur_coupe(s_)
chk("le dessinateur des planches à observations dessine la coupe même du kit (LARGEUR_ETIQUETTES = la place que "
    "styles.planche donne aux étiquettes) : mêmes textes aux mêmes places", textes_coupe(secs_a) == textes_coupe(secs_b)
    and len(textes_coupe(secs_a)) > 10, "")
vrai_min = dict(N.COUPE_MIN)
N.COUPE_MIN.update({1: (40.0, 0.0), 2: (80.0, 0.0)})
try:
    ch_, _ = chevauchements_coupe(cas((["type"], "predalle"), (["h"], 45), (S0 + ["My_inf"], 20), (S0 + ["Mx_sup"], 6),
                                      (S0 + ["V"], 30), (F("sup_x") + ["+"], 1)), 3)
finally:
    N.COUPE_MIN.clear()
    N.COUPE_MIN.update(vrai_min)
chk("mutation · COUPE_MIN trop petit : la coupe tassée sous ses observations, ses étiquettes se chevauchent — le contrôle rougit",
    bool(ch_), ch_)
# le dernier recours : des observations trop longues pour le panneau passent en fin de planche ; la planche qui déborde
# alors passe SEULE en trois colonnes, ses titres raccourcis pour y tenir (ceux du générateur en sortent)
from ndc_pdf.fonts import string_width  # noqa: E402
long_ = cas((["type"], "predalle"), (["h"], 40), (F("inf_y") + ["+"], 1), (S0 + ["My_inf"], 30), (S0 + ["Mx_sup"], 8),
            (S0 + ["V"], 40), (["sections", "+"], 1), (["sections", 1, "My_inf"], 12))
vrai_obs = N.observations
N.observations = lambda d, r, i: vrai_obs(d, r, i) * (3 if i == 0 else 1)
vrai_build = N._style


def espion_style(n_cols):
    s_ = vrai_build(n_cols)
    vrai_planche = s_.planche

    def planche_(d, doc, sec):
        if sec.get("n_cols") == 3:
            gw = (d.W - s_.margin - (s_.margin + s_.coupe_w + s_.gap) - 2 * s_.col_gap) / 3
            larges.extend(v["titre"] for v in sec["verifs"] if string_width(v["titre"], s_.f_h, 7.0) > gw - (s_.s_tit * 1.25 + 8))
            vus.append(sec["section"])
        return vrai_planche(d, doc, sec)
    s_.planche = planche_
    return s_


larges, vus = [], []
N._style = espion_style
try:
    d_ = construire(long_)
finally:
    N.observations, N._style = vrai_obs, vrai_build
chk("des observations trop longues : en fin de planche, la planche seule en trois colonnes (l'autre section en deux), "
    "chaque titre y tient, rien ne déborde", not d_.warnings and d_.replis == 1 and d_.trois_colonnes == 1 and vus
    and not larges and set(vus) == {"Section A"}, (d_.warnings, d_.replis, d_.trois_colonnes, vus, larges))

# ---------------------------------------------------------------- 6. le PDF, lu par pdftotext ; le routeur
e = CAS["deux dalles : CSP puis prédalle"]
octets = N.pdf(e, {"nom_projet": "Chantier Essai", "partie": "Dalles"})
n_pages = octets.count(b"/Type /Page\n") + octets.count(b"/Type /Page ") + octets.count(b"/Type /Page>")
chk("la note est un PDF : la garde et une planche par section (1 + 3 pages)", octets[:5] == b"%PDF-" and n_pages == 4, (octets[:8], n_pages))
tmp = tempfile.mkdtemp()
with open(os.path.join(tmp, "n.pdf"), "wb") as f:
    f.write(octets)
txt = subprocess.run(["pdftotext", "-layout", os.path.join(tmp, "n.pdf"), "-"], capture_output=True, text=True).stdout
shutil.rmtree(tmp, ignore_errors=True)
d0, d1 = e["dalles"]
g0 = P._calculer(d0)["sections"][0]["grandeurs"]
g1 = P._calculer(d1)["sections"][0]["grandeurs"]
attendus = [f"{d0['nom']} — Section A", f"{d1['nom']} — Section A", f"{d1['nom']} — Section B", "Chantier Essai",
            f"{ND.fn(g0['As_req_inf_y'], 0)} mm", f"{ND.fn(g1['d_inf_x'], 1)} cm", "dont prédalle préf.", "OBSERVATIONS",
            "VÉRIFIÉ"]
manque = [a for a in attendus if a not in txt]
chk("pdftotext lit dans la note le nom de chaque dalle et de chaque section, le chantier, des chiffres de l'écran (A_s,req, d), "
    "la peau de la prédalle et ses observations", not manque, manque)

r = json.loads(R.traiter(json.dumps({"module": "dalle", "action": "telecharger", "etat": e, "quoi": "note_pdf",
                                     "infos": {"nom": "Plancher <R+1>", "chantier": "", "indice": "B", "date": "10/11/2026"}})))
f = r.get("fichier") or {}
chk("le routeur rend la note en base 64, nommée d'après la note (nom de fichier sûr)",
    r["ok"] and f.get("type") == "application/pdf" and base64.b64decode(f.get("base64", ""))[:5] == b"%PDF-"
    and f.get("nom") == "Note de calcul - Plancher -R+1.pdf", (r.get("erreur"), f.get("nom")))
r = json.loads(R.traiter(json.dumps({"module": "dalle", "action": "telecharger", "etat": e, "quoi": "autre"})))
chk("un autre fichier que la note est refusé avec un message", not r["ok"] and "inconnu" in r["erreur"], r)
ancienne = copy.deepcopy(CAS["CSP · treillis + renforts, renfort secondaire empilé (S3)"])
ancienne["version"] = 2
ancienne["dalles"][0]["empilement"] = False
r = json.loads(R.traiter(json.dumps({"module": "dalle", "action": "telecharger", "etat": ancienne, "quoi": "note_pdf"})))
chk("une note d'avant (l'option « empilement » d'alors, l'avis d'ouverture) fait sa note, nappes empilées",
    r["ok"] and base64.b64decode(r["fichier"]["base64"])[:5] == b"%PDF-", r.get("erreur"))
captures = os.environ.get("NOTE_DALLE_CAPTURES")
if captures:                                          # les PDF des cas, à regarder (scratchpad de l'agent)
    os.makedirs(captures, exist_ok=True)
    for nom in ("deux dalles : CSP puis prédalle", nom_s3, "Prédalle · peau de 4 cm, base Ø16 qui en sort",
                "CSP · acier excessif (Ø40/100 en bas)", "CSP · trois renforts sur une face (n barres, barres, treillis)"):
        with open(os.path.join(captures, re.sub(r"[^\w]+", "_", nom)[:60] + ".pdf"), "wb") as f:
            f.write(N.pdf(CAS[nom], {"nom_projet": nom}))

# ---------------------------------------------------------------- 7. la traduction = l'inverse de la lecture d'un fichier Streamlit
chk("les types de couche de l'écran Streamlit : une source (EN.LIBELLES_TYPES), l'inverse de la lecture (FI.TYPES_ECRAN)",
    {v: k for k, v in EN.LIBELLES_TYPES.items()} == FI.TYPES_ECRAN, (EN.LIBELLES_TYPES, FI.TYPES_ECRAN))
ecarts = []
for nom, e in CAS.items():
    for di, d in enumerate(e["dalles"]):
        dalles, vals = N.valeurs(d, 1)
        lu = FI.lire({"dalles": dalles, "values": vals})
        if lu["remarques"]:
            ecarts.append((nom, "remarques", lu["remarques"][:2]))
        a, b = P._calculer(d), MO.verifier(lu["dalles"][0])
        for sa, sb in zip(a["sections"], b["sections"]):
            for k, x in sb["grandeurs"].items():
                y = sa["grandeurs"][k]
                if (isinstance(x, float) and not proche(x, y)) or (not isinstance(x, float) and x != y):
                    ecarts.append((nom, sa["nom"], k, x, y))
            if sa["statut"] != sb["statut"]:
                ecarts.append((nom, sa["nom"], "statut"))
chk("aller-retour : relues par la lecture d'un fichier de l'écran Streamlit (beton/dalle/fichier.lire), les valeurs "
    "traduites redonnent chaque grandeur du site (sans remarque)", not ecarts, ecarts[:3])

# ---------------------------------------------------------------- 8. MUTATIONS : chaque complément est gardé
def mutation(nom, appliquer, controle):
    """Applique ``appliquer()`` (qui rend la fonction de retour arrière), joue ``controle()`` (vrai = le contrôle passe) :
    la mutation doit le faire rougir."""
    retour = appliquer()
    try:
        try:
            passe = controle()
        except ValueError:
            passe = False
    finally:
        retour()
    chk(f"mutation · {nom} : le contrôle rougit", not passe)


def remplace(mod, nom, valeur):
    vrai = getattr(mod, nom)
    setattr(mod, nom, valeur)
    return lambda: setattr(mod, nom, vrai)


def decale(d, did=1):
    dalles, vals = VRAI_VALEURS(d, did)
    vals[f"dal{did}_sec1_dist_axe_inf_y_c1"] = repr(float(vals[f"dal{did}_sec1_dist_axe_inf_y_c1"].replace(",", ".")) + 0.5)
    return dalles, vals


VRAI_VALEURS = N.valeurs
nom_m = "CSP · M_y,inf = 28 (treillis par défaut)"
mutation("une hauteur utile décalée (axe inf. Y + 0,5 cm dans la traduction) → parité et note refusée",
         lambda: remplace(N, "valeurs", decale),
         lambda: not ecarts_parite(CAS[nom_m]) and bool(N.resultats(CAS[nom_m])))
mutation("une dalle oubliée par le générateur (la dernière planche perdue) → la note est refusée",
         lambda: remplace(EPD, "_collecter_resultats", (lambda vrai: lambda ds, v, b: vrai(ds, v, b)[:-1])(EPD._collecter_resultats)),
         lambda: bool(N.resultats(CAS["CSP · trois sections"])))


def axes_non_empiles(d, s):
    return {f: [c["dist"] if c.get("dist") is not None else POS._auto_moteur(d, c, j, f) for j, c in enumerate(s["faces"][f]["couches"])]
            for f in EN.FACES}


mutation("l'axe empilé non transmis (l'automatique du moteur, nappes au même niveau) → parité et note refusée",
         lambda: remplace(N.POS, "axes", axes_non_empiles),
         lambda: not ecarts_parite(CAS[nom_s3]) and bool(N.resultats(CAS[nom_s3])))
mutation("S5 retiré (le lit de contact réduit à la base principale : la nappe secondaire au travers du renfort Ø25) → le calcul, la "
         "planche, la coupe et le PDF de la note S5 disent 4,5 / 15,5 cm / 430 mm²",
         lambda: remplace(POS, "lit_contact", lambda d, s, cote: [0]), lambda: s5_bout_en_bout()[0])
mutation("l'état de l'écran retiré (« Vérifié » sans effort)",
         lambda: remplace(N, "etat_label", lambda rs: "Vérifié"),
         lambda: planche("CSP neuve (sans effort)")[0]["etat"] == N.ETAT_INCOMPLET)
mutation("les niveaux du calcul retirés de la coupe", lambda: remplace(N, "_niveaux_du_calcul", lambda coupe, R_: None),
         lambda: coupe_juste(nom_s3))
mutation("le verdict de l'acier excessif retiré", lambda: remplace(N, "_verdict_site", lambda v_pdf, v, rs: None),
         lambda: "acier excessif" in planche("CSP · acier excessif (Ø40/100 en bas)")[0]["verifs"][1]["verdicts"][0]["texte"])
mutation("les observations retirées", lambda: remplace(N, "observations", lambda d, r, i: []),
         lambda: "Non vérifiés" in " ".join(N.observations_de(planche("Prédalle · neuve sollicitée")[0])))
mutation("le contrôle de la note sans effet (N.ecarts vide) sur une traduction fausse → la parité du test le voit encore",
         lambda: (lambda r1, r2: lambda: (r1(), r2()))(remplace(N, "ecarts", lambda R_, rs: []), remplace(N, "valeurs", decale)),
         lambda: not ecarts_parite(CAS[nom_m]))

# ---------------------------------------------------------------- 9. comme dans le NAVIGATEUR (et ce que la note charge de plus que celle de la poutre)
SONDE = r"""
import builtins, json, os, sys
RACINE, STUB = {racine!r}, {stub!r}
sys.path.insert(0, STUB); sys.path.insert(1, RACINE); os.chdir(RACINE)
class Refus:
    def find_spec(self, name, path=None, target=None):
        racine = name.split(".")[0]
        if racine in ("fontTools", "_rl_accel", "rl_accel", "streamlit", "numpy", "matplotlib") or name == "reportlab.lib._rl_accel":
            raise ImportError("refusé dans le navigateur : " + name)
sys.meta_path.insert(0, Refus())
lus = set(); _open = builtins.open
def espion(f, *a, **k):
    try: lus.add(os.path.abspath(f))
    except Exception: pass
    return _open(f, *a, **k)
builtins.open = espion
QUOI = {quoi!r}
if QUOI == "poutre":
    from api_web import poutre as P, note_poutre as NP
    etat = P.nouvel_etat()
    for ch, v in ((["sections", 0, "M_inf"], 120), (["sections", 0, "V"], 80), (["sections", 0, "inf", "+"], 1), (["sections", 0, "etriers", "+"], "epingle"),
                  (["sections", "+"], 1), (["sections", 1, "M_sup"], 90), (["sections", 0, "laterales", "n"], 1)):
        P.modifier(etat, ch, v)
    P.gerer_elements(etat, "dupliquer")
    o = NP.fichier(etat, {{"nom": "relevé"}})
else:
    from api_web import dalle as P, note_dalle as N
    etat = P.nouvel_etat()
    for ch, v in ((["sections", 0, "My_inf"], 28), (["sections", 0, "Mx_sup"], 12), (["sections", 0, "V"], 60), (["enrobage"], 0),
                  (["sections", 0, "faces", "inf_x", "couches", "+"], 1), (["sections", 0, "faces", "inf_x", "couches", 1, "type"], "barres"),
                  (["sections", 0, "nappes", "sup", "treillis"], "8/10/100/150"), (["sections", "+"], 1), (["sections", 1, "My_sup"], 30)):
        assert P.modifier(etat, ch, v) is None, ch
    P.gerer_elements(etat, "ajouter")
    for ch, v in ((["type"], "predalle"), (["sections", 0, "My_inf"], 22), (["sections", 0, "faces", "inf_y", "couches", "+"], 1)):
        assert P.modifier(etat, ch, v) is None, ch
    o = N.fichier(etat, {{"nom": "relevé", "chantier": "Essai", "indice": "B"}})
import base64, PIL, reportlab
assert base64.b64decode(o["base64"])[:5] == b"%PDF-"
base = os.path.dirname(os.path.dirname(os.path.abspath(reportlab.__file__)))
mods = {{os.path.abspath(m.__file__) for m in list(sys.modules.values()) if getattr(m, "__file__", None)}}
rl = sorted(os.path.relpath(f, base) for f in mods | lus if f.startswith(os.path.join(base, "reportlab") + os.sep) and not f.endswith(".pyc"))
depot = sorted(os.path.relpath(f, RACINE) for f in mods | lus if f.startswith(RACINE + os.sep) and not f.startswith(STUB) and "__pycache__" not in f)
polices = sorted(os.path.basename(f) for f in lus if f.lower().endswith((".ttf", ".otf", ".afm", ".pfb")))
print(json.dumps({{"pdf": True, "pil": PIL.__file__, "fonttools": "fontTools" in sys.modules, "reportlab": rl, "depot": depot,
                  "polices": polices, "interdits": sorted(m for m in sys.modules if m.split(".")[0] in ("streamlit", "numpy", "matplotlib", "fontTools"))}}))
"""
sys.path.insert(0, os.path.join(RACINE, "site", "web"))
import construire as K  # noqa: E402


def sonde(quoi):
    tmp_ = tempfile.mkdtemp()
    for f_ in K.NOTE_POLICES:
        shutil.copy(os.path.join(RACINE, "ndc_pdf", "fonts", f_), tmp_)
    p_ = subprocess.run([sys.executable, "-c", SONDE.format(racine=RACINE, stub=K.NOTE_SOURCES, quoi=quoi)],
                        capture_output=True, text=True, cwd=RACINE, env={**os.environ, "NDC_FONT_DIR": tmp_})
    shutil.rmtree(tmp_, ignore_errors=True)
    try:
        return json.loads(p_.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        return {"erreur": p_.stderr[-800:]}


s_d, s_p = sonde("dalle"), sonde("poutre")
chk("comme dans le navigateur (fontTools, accélérateurs C, streamlit, numpy, matplotlib refusés ; Pillow factice ; les seules "
    "polices du paquet) : la note de la dalle se fait, CSP et prédalle", s_d.get("pdf") is True and s_d.get("fonttools") is False
    and s_d.get("pil", "").startswith(K.NOTE_SOURCES) and not s_d.get("interdits") and set(s_d.get("polices", [])) <= set(K.NOTE_POLICES),
    s_d.get("erreur") or s_d)
en_plus = sorted(set(s_d.get("depot", [])) - set(s_p.get("depot", [])))
hors_paquets = [f_ for f_ in en_plus if f_.split(os.sep)[0] not in K.PAQUETS_ENTIERS + ("ndc_pdf",)]
rl_plus = sorted(set(s_d.get("reportlab", [])) - set(s_p.get("reportlab", [])))
print("     · fichiers du dépôt chargés en plus de la note de la poutre, hors des paquets entiers du moteur :", hors_paquets)
print("     · fichiers de reportlab en plus :", rl_plus)
print("     · polices lues :", s_d.get("polices"))
chk("ce que la note de la dalle charge de plus que celle de la poutre, hors des paquets du moteur : exactement "
    "modules/export_pdf_dalle.py et modules/treillis.py (à ajouter au paquet de la note), aucun fichier de reportlab de plus",
    hors_paquets == ["modules/export_pdf_dalle.py", "modules/treillis.py"] and not rl_plus, (hors_paquets, rl_plus))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:200])
sys.exit(1 if KO else 0)

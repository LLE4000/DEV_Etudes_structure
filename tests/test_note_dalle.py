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
from api_web import note_dalle_planche as NP  # noqa: E402
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
    "CSP · d utile ≤ 0 au-delà de 100 % (moment sur la face)": cas((F("inf_y") + [0, "dist"], "21"), (S0 + ["My_inf"], 10)),
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
    # (DAL5b : un treillis unique a ses fils X soudés DANS la peau — plus de croisement ; en barres, la base X reste SUR la
    #  prédalle, la règle du moteur, et y croise le renfort Y)
    "Prédalle · couches croisées sur la prédalle (renfort Y, base X)": cas(
        (["type"], "predalle"), (S0 + ["nappes", "inf", "mode"], "barres"), (F("inf_y") + [0, "d"], 10), (F("inf_y") + [0, "esp"], 100),
        (F("inf_x") + [0, "d"], 10), (F("inf_x") + [0, "esp"], 100), (F("inf_y") + ["+"], 1), (S0 + ["My_inf"], 30), (S0 + ["Mx_inf"], 10)),
    # (DAL6) la base X d'une prédalle en barres, posée SUR la prédalle (h_p + Ø/2 = 6,5), dimensionne la hauteur
    "Prédalle · barres, la base X posée sur la prédalle dimensionne": cas(
        (["type"], "predalle"), (S0 + ["nappes", "inf", "mode"], "barres"), (S0 + ["Mx_inf"], 30), (S0 + ["My_inf"], 5)),
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


def formules(v):
    """Le texte des lignes d'une vérification : libellés, expressions (et leurs formes courtes), lignes en gras."""
    out = []
    for it in v["items"]:
        for x in it[1:]:
            out += [str(y[1]) for y in x] if isinstance(x, list) else [str(x)]
    return "\n".join(out)


def textes_planche(sec):
    """Tout le texte d'une planche : l'en-tête compact, les sollicitations, les vérifications et leurs verdicts."""
    out = [t for case in sec["entete"]["cases"] for t in case] + [sec["entete"]["note"]]
    so = sec["blocs"].sollicitations
    out += [" ".join(ligne) for ligne in so["lignes"]] + [so["V"]]
    for v in sec["verifs"]:
        out += [str(v["titre"]), formules(v)] + [k["texte"] for k in v["verdicts"]]
    return "\n".join(out)


TITRES = {"hauteur": NP.TITRE_HAUTEUR, "minmax": NP.TITRE_MINMAX, "tau": NP.TITRE_TRANCHANT, "!": NP.TITRE_AVERTISSEMENTS}


def verif(sec, cle):
    """La vérification d'une planche par sa clé : « hauteur », « minmax », « tau », « ! » ou une face (« inf_y »…)."""
    return next((v for v in sec["verifs"] if v.get("cle") == cle or v["titre"] == TITRES.get(cle)), None)


def res(x, nd, unite):
    """Un résultat de formule de la note : « \\res{16,5 \\u{cm}} »."""
    return rf"\res{{{ND.fn(x, nd)} \u{{{unite}}}}}"


MM2 = "mm}^{2"
manques, n_chiffres = [], 0
for nom, e in CAS.items():
    secs, pls = lire_planches(e)
    for sec, pl in zip(secs, pls):
        rs, g = pl["site"], pl["site"]["grandeurs"]
        par_cle = {v["cle"]: v for v in rs["verifications"]}
        p = rs["principale"]
        if [v["titre"] for v in sec["verifs"][:2]] != [NP.TITRE_HAUTEUR, NP.TITRE_MINMAX]:
            manques.append((nom, rs["nom"], "ordre 1-2", [v["titre"] for v in sec["verifs"][:2]]))
        # 1 la hauteur : h_u,min (si un moment) et h_min au dixième
        v1, v2 = verif(sec, "hauteur"), verif(sec, "minmax")
        attendus = [(v1, res(g["h_min"], 1, "cm"))] + ([(v1, res(g["hu_min"], 1, "cm"))] if g["M_hauteur"] > 0 else [])
        # 2 les sections minimale et maximale, UNE fois
        attendus += [(v2, res(g[k], 0, MM2)) for k in ("As_min_ec", "As_min_plancher", "As_min_base", "As_max")]
        # 3 à 6 : chaque nappe, dans l'ordre de l'écran
        for v_pdf, f in zip(sec["verifs"][2:6], MO.ordre_faces(p)):
            if v_pdf.get("cle") != f:
                manques.append((nom, rs["nom"], f, "ordre", v_pdf.get("cle")))
                continue
            attendus += [(v_pdf, res(g[f"d_{f}"], 1, "cm")), (v_pdf, res(g[f"As_req_{f}"], 0, MM2)),
                         (v_pdf, res(g[f"As_min_{f}"], 0, MM2)), (v_pdf, f"{ND.fn(g[f'As_pm_{f}'], 0)} mm²/m")]
            verdict = v_pdf["verdicts"][0]["texte"].replace(C.NBSP, " ")
            vv = par_cle[f]
            if vv["etat"] == MO.OK:
                attendus_v = [f"{ND.fn(g[f'As_{f}'], 0)} mm² posés", " ≥ ", f"{ND.fn(g[f'besoin_{f}'], 0)} mm² requis",
                              f"taux {C.pc(C.pct(vv['taux']))} %"]
            elif vv["taux"] > 1 and "d_utile_negatif" not in vv["motifs"]:
                attendus_v = [f"{ND.fn(g[f'As_{f}'], 0)} mm² posés", " < ", f"{ND.fn(g[f'besoin_{f}'], 0)} mm² requis"]
            else:
                attendus_v = []                      # acier excessif, d ≤ 0 (quel que soit le taux) : contrôlés plus bas
            manques += [(nom, rs["nom"], f, t, verdict) for t in attendus_v if t not in verdict]
            if (v_pdf["verdicts"][0]["etat"] == "ok") != (vv["etat"] == MO.OK):
                manques.append((nom, rs["nom"], f, "couleur du verdict"))
            n_chiffres += len(attendus_v)
        if g["V"] > 0:
            v7 = verif(sec, "tau")
            attendus += [(v7, res(g["tau"], 2, "N/mm}^{2")), (v7, res(g["tau_adm"], 2, "N/mm}^{2"))]
        elif verif(sec, "tau") is not None:
            manques.append((nom, rs["nom"], "τ sans V"))
        for v_pdf, t in attendus:
            n_chiffres += 1
            if v_pdf is None or t not in formules(v_pdf):
                manques.append((nom, rs["nom"], v_pdf and v_pdf["titre"], t))
        so = sec["blocs"].sollicitations
        lu = {"Mx_inf": so["lignes"][0][1], "Mx_sup": so["lignes"][0][2], "My_inf": so["lignes"][1][1], "My_sup": so["lignes"][1][2],
              "V": so["V"]}
        for k, x in lu.items():
            n_chiffres += 1
            if x != ND.fn(rs[k], 1):
                manques.append((nom, rs["nom"], k, x, rs[k]))
chk(f"le texte de chaque planche écrit les chiffres du moteur du site ({n_chiffres} chiffres : h_u,min, h_min, les sections "
    "minimale et maximale communes, d, A_s,req, A_s,min, A_s/m, A_s, A_s,néc et le taux, τ, τ_adm, les efforts), les nappes "
    "dans l'ordre de l'écran", not manques, manques[:4])

# l'écran affiche les mêmes chiffres (la ligne de chaque nappe : A_s,néc et A_s, et l'état du bloc), aux mêmes arrondis
ecran, ecarts_arrondi, n_ecran = [], [], 0
ETATS_ECRAN = {"Conforme": "Vérifié", "Non conforme": "Non vérifié", "Données incomplètes": N.ETAT_INCOMPLET}
BLOC = re.compile(r'<section class="et-bloc (\w+)" data-section="(\d+)">.*?<span class="e">([^<]+)</span>', re.S)
for nom, e in CAS.items():
    for i in range(len(e["dalles"])):
        ee = copy.deepcopy(e)
        ee["actif"] = i
        res_ = P.resultat(ee)
        html_ = res_["resultat"].replace(C.FINE, " ")
        secs = N.sections(ee)
        debut = sum(len(d["sections"]) for d in e["dalles"][:i])
        etats_ecran = [x[2] for x in BLOC.findall(res_["resultat"])]
        for k, s in enumerate(P._calculer(e["dalles"][i])["sections"]):
            sec = secs[debut + k]
            if ETATS_ECRAN.get(etats_ecran[k]) != sec["etat"]:
                ecran.append((nom, s["nom"], etats_ecran[k], sec["etat"]))
            for v in s["verifications"]:
                if v["cle"] not in EN.FACES or P._sous_limite_non_conforme(v) or "d_utile_negatif" in v["motifs"]:
                    continue                      # acier excessif, d ≤ 0 (quel que soit le taux) : contrôlés plus bas
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


# ---------------------------------------------------------------- 3. la HAUTEUR détaillée, l'en-tête, sans répétition (DAL6, point 8)
def planche(nom, k=0):
    secs, pls = lire_planches(CAS[nom])
    return secs[k], pls[k]


def nombres(expr):
    """Les nombres d'une expression de la note (« 3 + 0,5 + 0,5 = 4,0 » → [3.0, 0.5, 0.5, 4.0])."""
    return [float(x.replace(",", ".")) for x in re.findall(r"(?<![\w{])-?\d+(?:,\d+)?", expr)]


def decompositions(sec):
    """(somme lue, résultat lu) de chaque y_G décomposé de la hauteur : « y_G = c + Ø/2 = 3 + 0,5 = \\res{3,5 …} »."""
    out = []
    for it in verif(sec, "hauteur")["items"]:
        if it[0] == "f" and it[2].startswith("y_{G} = ") and r"\sum" not in it[2]:
            valeurs = it[2].split(" = ")[-2]
            out.append((round(sum(nombres(valeurs)), 6), nombres(it[2].split(r"\res{")[1])[0]))
    return out


# la hauteur : le moment et sa face, h_u,min en symboles PUIS en valeurs, l'enrobage mécanique, h_min ≤ h — point 8
sec, pl = planche("CSP · moments sur les deux faces et les deux directions, V")
v1 = verif(sec, "hauteur")
f_ = [it for it in v1["items"] if it[0] == "f"]
textes_v1 = formules(v1)
chk("hauteur minimale DÉTAILLÉE (M_y,inf = 35 kN·m, C30/37, B500) : le moment dimensionnant et sa face, h_u,min en symboles "
    "(√(M / (α_b·b·μ))) puis en valeurs (√(35·10⁶ / (12,96 · 1000 · 0,1709))) = 12,6 cm, α_b et μ dits, l'enrobage mécanique "
    "décomposé (y_G = c + Ø/2 = 3 + 0,5 = 3,5 cm), puis h_min = 12,6 + 3,5 = 16,1 cm ≤ h = 20 cm, et le taux (80 %)",
    v1["items"][0][:4] == ("v", "Moment dimensionnant", "M_{y,inf}", "35,0")
    and f_[0][2] == r"h_{u,min} = \sqrt{\frac{M}{\alpha_{b} \cdot b \cdot \mu}}"
    and f_[1][2] == r"= \sqrt{\frac{35 \cdot 10^{6}}{12,96 \cdot 1000 \cdot 0,1709}} = \res{12,6 \u{cm}}"
    and "Face inférieure Y : la plus exigeante" in textes_v1 and r"$\alpha_{b}$ = 12,96 N/mm²" in textes_v1
    and r"$\mu$ = 0,1709" in textes_v1
    and r"y_{G} = c + \frac{\phi}{2} = 3 + 0,5 = \res{3,5 \u{cm}}" in textes_v1
    and r"h_{min} = h_{u,min} + y_{G} = 12,6 + 3,5 = \res{16,1 \u{cm}} \le h = 20 \u{cm}" in textes_v1
    and v1["verdicts"][0]["texte"].replace(C.NBSP, " ") == "Épaisseur 20 cm ≥ hauteur minimale 16,1 cm — taux 80 %",
    textes_v1[:400])
# chaque décomposition de y_G écrite redonne le y_G du moteur (la note n'écrit jamais une somme fausse) — tous les cas
faux, n_dec, sortes = [], 0, set()
for nom, e in CAS.items():
    for sec in N.sections(e):
        for somme, lu in decompositions(sec):
            n_dec += 1
            if abs(somme - lu) > 0.051:
                faux.append((nom, sec["section"], somme, lu))
        sortes |= {("appui" if "e_{appui}" in it[2] else "jeu" if "+ jeu" in it[2] else "hp" if "h_{p} +" in it[2] else "c")
                   for it in verif(sec, "hauteur")["items"] if it[0] == "f" and it[2].startswith("y_{G} = ") and r"\sum" not in it[2]}
chk(f"chaque enrobage mécanique décomposé dans la hauteur redonne y_G ({n_dec} décompositions : enrobage + demi-Ø, + jeu, la "
    "couche porteuse + demi-Ø, ou la prédalle + demi-Ø)", not faux and n_dec >= 20 and sortes == {"c", "jeu", "appui", "hp"},
    (faux[:3], sortes))
t_ = formules(verif(planche("deux dalles : CSP puis prédalle", 1)[0], "hauteur"))
t2_ = formules(verif(planche("Prédalle · barres, la base X posée sur la prédalle dimensionne")[0], "hauteur"))
chk("… une face secondaire dimensionnante : posée sur la principale, y_G = e_appui + Ø_appui/2 + Ø/2 = 3,5 + 0,5 + 0,5 = 4,5 cm "
    "(« posée sur la base Y ») ; posée sur la prédalle (Ø12, son demi-diamètre arrondi au 0,5 cm supérieur), y_G = h_p + Ø/2 = 6 + 1 "
    "= 7,0 cm",
    r"y_{G} = e_{appui} + \frac{\phi_{appui}}{2} + \frac{\phi}{2} = 3,5 + 0,5 + 0,5 = \res{4,5 \u{cm}}" in t_
    and "posée sur la base Y" in t_ and r"y_{G} = h_{p} + \frac{\phi}{2} = 6 + 1 = \res{7,0 \u{cm}}" in t2_
    and "posée sur la prédalle" in t2_ and "(Ø 12, arrondi au 0,5 cm supérieur)" in t2_, (t_[-300:], t2_[-300:]))
sec, pl = planche("CSP · trois renforts sur une face (n barres, barres, treillis)")
v3 = verif(sec, "inf_y")
chk("… une face à plusieurs couches : y_G = Σ A_s,i·e_i / Σ A_s,i dans sa nappe (au-delà de trois couches, les couches en "
    "texte), et la hauteur y renvoie (« calculé en 3 ») au lieu de la répéter",
    r"y_{G} = \frac{\sum A_{s,i} \cdot e_{i}}{\sum A_{s,i}}" in formules(v3) and "Les 4 couches" in formules(v3)
    and "calculé en 3" in formules(verif(sec, "hauteur")) and r"\sum" not in formules(verif(sec, "hauteur")), formules(v3)[:300])
sec, pl = planche("CSP neuve (sans effort)")
chk("sans aucun moment, la hauteur le dit (h_u,min = 0) et compare encore h_min à h",
    "Aucun moment appliqué" in formules(verif(sec, "hauteur")) and r"h_{min} = h_{u,min} + y_{G} = 0,0 + " in formules(verif(sec, "hauteur")), "")

# l'EN-TÊTE compact : une fois sous le titre, les données de la dalle, du béton, de l'acier, et ce qui n'est pas vérifié
mauvais = []
for nom, e in CAS.items():
    secs = N.sections(e)
    k = 0
    for d in e["dalles"]:
        for s_ in P._calculer(d)["sections"]:
            sec, k = secs[k], k + 1
            g = s_["grandeurs"]
            ent = sec["entete"]
            titres = [c_[0] for c_ in ent["cases"]]
            texte = " ".join(c_[1] for c_ in ent["cases"])
            attendus = [f"$b$ = {C.nb(g['b'])} cm", f"$h$ = {C.nb(g['h'])} cm", f"$c$ = {C.nb(g['enrobage_calcul'])} cm",
                        f"principale {g['principale'].upper()}", f"$f_{{ck}}$ = {ND.fn(g['fck'], 0)}",
                        f"$f_{{ctm}}$ = {ND.fn(g['fctm'], 2)}", f"= {ND.fn(g['fyd'], 1)} N/mm²",
                        rf"$\alpha_{{b}}$ = {ND.fn(g['alpha_b'], 2)}", rf"$\mu$ = {ND.fn(g['mu'], 4)}"]
            attendus += [f"$h_{{p}}$ = {C.nb(g['h_pre'])} cm"] if P.est_predalle(d) else []
            attendus += ["(0 saisi)"] if not g["enrobage"] else []
            attendus += [f"jeu {C.nb(g['jeu_premier_lit'])} cm"] if g["jeu_premier_lit"] else []
            if (titres != [("Prédalle" if P.est_predalle(d) else "Dalle pleine"), f"Béton {g['beton']}", f"Acier B{int(g['fyk'])}"]
                    or ent["note"] != P.non_verifies(d) or any(a not in texte for a in attendus)):
                mauvais.append((nom, titres, ent["note"][:30], [a for a in attendus if a not in texte]))
chk("l'EN-TÊTE compact de chaque planche (point 8) : le type (« Dalle pleine », « Prédalle »), b, h, h_p, c (« 0 saisi »), le "
    "jeu, la direction principale ; le béton (f_ck, f_ctm, α_b) ; l'acier (f_yd = f_yk/γ_s, μ) ; et « Non vérifié(e)s : … » "
    "en une phrase, celle de l'écran (api_web/dalle.non_verifies)", not mauvais, mauvais[:2])
# « sans les répéter inutilement à chaque calcul » : f_ctm et les sections minimale et maximale UNE fois par planche
rep = []
for nom, e in CAS.items():
    for sec, pl in zip(*lire_planches(e)):
        g = pl["site"]["grandeurs"]
        tout = "\n".join(formules(v) for v in sec["verifs"])
        for x in (rf"\frac{{{ND.fn(g['fctm'], 2)}}}", res(g["As_max"], 0, MM2), "A_{s,min,EC} = ", "A_{s,max} = "):
            if tout.count(x) != 1:
                rep.append((nom, sec["section"], x, tout.count(x)))
        if [t for t, _ in sec["blocs"]] != [NP.TITRE_SOLLICITATIONS]:
            rep.append((nom, "panneau", [t for t, _ in sec["blocs"]]))
chk("sans répétition (point 8) : f_ctm, A_s,min,EC et A_s,max s'écrivent UNE fois par planche (les sections communes), plus "
    "dans chaque nappe ; le panneau de gauche n'a plus DIMENSIONS ni MATÉRIAUX (l'en-tête les dit), seulement les sollicitations",
    not rep, rep[:3])
sec, pl = planche("CSP · treillis par nappe 8/10/100/150 en bas, 6/6/150/150 en haut")
t = textes_planche(sec)
chk("les armatures POSÉES sont un résultat (point 8) : « Armatures posées : Treillis 8/10/100/150 — 503 mm²/m » en Y, sa "
    "transposée 10/8/150/100 (524 mm²/m) en X ; le verdict dit posé ≥ requis et le taux",
    "Armatures posées : Treillis 8/10/100/150 — 503 mm²/m" in t and "Armatures posées : Treillis 10/8/150/100 — 524 mm²/m" in t
    and "On prend" not in t, t[:200])
sec, pl = planche("CSP · B400, γs 1,15, C20/25, bande de 50 cm")
chk("… une bande de 50 cm : le verdict dit les mm² posés SUR LA BANDE (A_s = A_s/m · b)",
    "393 mm² posés sur la bande de 50 cm" in verif(sec, "inf_y")["verdicts"][0]["texte"].replace(C.NBSP, " "),
    verif(sec, "inf_y")["verdicts"][0]["texte"])

# ---------------------------------------------------------------- 4. ce que le site ajoute : empilement, coupe, verdict, état, avertissements
nom_s3 = "CSP · treillis + renforts, renfort secondaire empilé (S3)"
d_s3 = CAS[nom_s3]["dalles"][0]
sec, pl = planche(nom_s3)
axes = [c["e"] for c in pl["R"]["dirs"]["x"]["geo_inf"]["couches"]]
axes_y = [c["e"] for c in pl["R"]["dirs"]["y"]["geo_inf"]["couches"]]
chk("S3 et DAL5b : la planche compte les axes du site — les fils X du treillis soudés sur les fils Y (3,5 + 0,5 + 0,5 = 4,5), le renfort X "
    "(S3) dans leur plan (3,5 + 0,5 + 1,0 = 5,0) ; le renfort Y que « Treillis + renforts » ajoute posé SUR le treillis, sur le renfort X "
    "(6,0 + 1,0 = 7,0) — avant DAL5b : 5,5 et 6,0, la nappe X soulevée par le renfort Y resté dans le plan des fils Y (4,0)",
    axes == POS.axes(d_s3, d_s3["sections"][0])["inf_x"] == [4.5, 5.0]
    and axes_y == POS.axes(d_s3, d_s3["sections"][0])["inf_y"] == [3.5, 7.0], (axes, axes_y))
# S5 de bout en bout : un renfort principal 3 Ø25 sous la nappe secondaire — le calcul, la planche, la coupe et le PDF disent la même
# chose (la nappe X sur le renfort Y : axe 6,5, d = 13,5 cm), et pas la base seule (4,5 : d = 15,5). DAL5b : S5 vaut pour des BARRES
# (Ø10/100 : le Ø et la section du treillis 10/10/100/100) ; le treillis, lui, pose son renfort Y sur lui (plus bas, DAL5b de bout en bout)
nom_s5 = "CSP · barres Ø10/100 + renfort principal 3 Ø25 (S5)"
e_s5 = cas((S0 + ["nappes", "inf", "mode"], "barres"), (F("inf_y") + [0, "d"], 10), (F("inf_y") + [0, "esp"], 100),
           (F("inf_x") + [0, "d"], 10), (F("inf_x") + [0, "esp"], 100), (F("inf_y") + ["+"], 1), (F("inf_y") + [1, "d"], 25),
           (S0 + ["My_inf"], 30), (S0 + ["Mx_inf"], 20))


def pdftexte(etat, nom_):
    tmp_ = tempfile.mkdtemp()
    with open(os.path.join(tmp_, "n.pdf"), "wb") as fh:
        fh.write(N.pdf(etat, {"nom_projet": nom_}))
    txt_ = subprocess.run(["pdftotext", "-layout", os.path.join(tmp_, "n.pdf"), "-"], capture_output=True, text=True).stdout
    shutil.rmtree(tmp_, ignore_errors=True)
    return txt_


def s5_bout_en_bout():
    """Le calcul, la planche, la coupe de l'écran et le PDF (pdftotext) de la note S5 : (passe, ce qu'on a lu)."""
    r_ = P._calculer(e_s5["dalles"][0])
    g_ = r_["sections"][0]["grandeurs"]
    secs_, pls_ = lire_planches(e_s5)
    txt_ = formules(verif(secs_[0], "inf_x"))
    pdf_ = pdftexte(e_s5, nom_s5)
    leg_ = re.findall(r"X · Ø10/100 · axe à \d+ · sur [^<]+", P.resultat(copy.deepcopy(e_s5))["vues"][0]["svg"])
    e_ = r_["sections"][0]["faces"]["inf_x"]["couches"][0]["e"]
    passe = (e_ == 6.5 and g_["d_inf_x"] == 13.5 and round(g_["As_req_inf_x"]) == 494
             and [c["e"] for c in pls_[0]["R"]["dirs"]["x"]["geo_inf"]["couches"]] == [6.5] and not ecarts_parite(e_s5)
             and res(13.5, 1, "cm") in txt_ and res(494, 0, MM2) in txt_
             and res(15.5, 1, "cm") not in txt_ and res(430, 0, MM2) not in txt_
             and leg_ == ["X · Ø10/100 · axe à 65 · sur renfort Y"]          # la légende commune (DAL5) : une fois
             and "13,5 cm" in pdf_ and "494 mm" in pdf_)
    return passe, (e_, g_["d_inf_x"], round(g_["As_req_inf_x"], 1), leg_, txt_[:200])


passe5, lu5 = s5_bout_en_bout()
chk("S5 de bout en bout (renfort principal 3 Ø25 en inf. Y, M_x,inf = 20) : le calcul pose la base inf. X sur le renfort (axe 6,5, d = 13,5 cm, "
    "A_s,req = 494 mm²) ; la planche compte la couche à 6,5 et écrit d = 13,5 cm et 494 mm² (et non 15,5 cm, 430 mm² de la base seule) ; "
    "la légende commune des coupes dit « X · Ø10/100 · axe à 65 · sur renfort Y » ; pdftotext lit 13,5 cm et 494 mm dans le PDF", passe5, lu5)
# DAL5b de bout en bout : le même renfort 3 Ø25 sur un TREILLIS (la note neuve en « Treillis + renforts ») — posé SUR le treillis
# (4,5 + 0,5 + 1,5 = 6,5), les fils X soudés restent à 4,5 : le calcul, la planche, la coupe et le PDF disent la même chose
e_t5 = cas((S0 + ["nappes", "inf", "mode"], "treillis_renforts"), (F("inf_y") + [1, "d"], 25), (S0 + ["My_inf"], 30), (S0 + ["Mx_inf"], 20))
r_t5 = P._calculer(e_t5["dalles"][0])
g_t5 = r_t5["sections"][0]["grandeurs"]
secs_t5, pls_t5 = lire_planches(e_t5)
txt_t5 = formules(verif(secs_t5[0], "inf_y"))
pdf_t5 = pdftexte(e_t5, "DAL5b")
sv_t5 = P.resultat(copy.deepcopy(e_t5))["vues"][0]["svg"]
d_y5 = ND.fn(g_t5["d_inf_y"], 1)
chk(f"DAL5b de bout en bout (3 Ø25 en inf. Y sur le treillis, M_y,inf = 30) : le calcul pose le renfort à 6,5 (sur les fils X, 4,5) — "
    f"d_y = {d_y5} cm, A_s,req = {round(g_t5['As_req_inf_y'])} mm² ; d_x = 15,5 cm (les fils X ne montent plus) ; la planche compte les "
    "couches Y à 3,5 et 6,5 ; la formule écrit d_y ; la légende dit « R1 Y · 3 Ø25 · axe à 65 · sur le treillis » ; pdftotext lit d_y",
    [c["e"] for c in r_t5["sections"][0]["faces"]["inf_y"]["couches"]] == [3.5, 6.5] and r_t5["sections"][0]["faces"]["inf_x"]["couches"][0]["e"] == 4.5
    and g_t5["d_inf_x"] == 15.5 and [c["e"] for c in pls_t5[0]["R"]["dirs"]["y"]["geo_inf"]["couches"]] == [3.5, 6.5] and not ecarts_parite(e_t5)
    and rf"\res{{{d_y5} \u{{cm}}}}" in txt_t5 and "R1 Y · 3 Ø25 · axe à 65 · sur le treillis" in sv_t5
    and "axes : Y à 35 · X à 45" in sv_t5 and f"{d_y5} cm" in pdf_t5,
    (g_t5["d_inf_y"], g_t5["As_req_inf_y"], re.findall(r"R1 Y[^<]+", sv_t5), txt_t5[:160]))


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
sec, pl = planche(nom_s3)
brut = ND.construire_sections_dalle(N.resultats(CAS[nom_s3]))[0]
chk("… ce que le générateur seul ne fait pas : il écartait le renfort de sa base pour le dessin (le cas S3 le montre)",
    any(a != b for a, b in niveaux_coupe(brut, pl["R"])), niveaux_coupe(brut, pl["R"]))
sch = sec["coupe"]["schemas"]
chk("les titres des schémas sont ceux de l'écran (« Direction Y · principale », « Direction X ») et, dans le schéma secondaire, "
    "les fils du treillis ne répètent pas sa désignation (« Inf. X : fils du treillis », sa section au mètre gardée) — DAL5a",
    [x["titre"] for x in sch] == ["Direction Y · principale", "Direction X"] and sch[1]["labs_inf"][0][0] == "Inf. X : fils du treillis"
    and sch[1]["labs_inf"][0][1] == f"{ND.fn(pl['R']['dirs']['x']['geo_inf']['couches'][0]['As_pm'], 0)} mm²/m"
    and sch[0]["labs_inf"][0][0] == "Inf. Y : 10/10/100/100", ([x["titre"] for x in sch], sch[1]["labs_inf"][:1]))
sec, pl = planche("CSP · gros Ø (Ø25/150 en bas, Ø20/200 en haut), h = 30")
chk("… des BARRES dans les deux directions : chaque schéma garde sa désignation", "fils du treillis" not in str(sec["coupe"]), "")

sec, pl = planche("CSP · acier excessif (Ø40/100 en bas)")
vd = verif(sec, "inf_y")["verdicts"][0]
vb = ND.construire_sections_dalle(N.resultats(CAS["CSP · acier excessif (Ø40/100 en bas)"]))[0]["verifs"][1]["verdicts"][0]
chk("acier excessif : le verdict dit A_s > A_s,max (le générateur seul écrirait « A_s < requise », faux)",
    vd["etat"] == "ko" and "12566 mm² > section d'armature maximale : 8000 mm² (acier excessif)" in vd["texte"].replace(C.NBSP, " ")
    and "<" in vb["texte"], (vd, vb["texte"]))
sec, pl = planche("CSP · d utile ≤ 0 (axe saisi au-delà de h)")
vd = verif(sec, "inf_y")["verdicts"][0]
chk("d utile ≤ 0 : le verdict dit d = −1,0 cm ≤ 0 (position des couches incompatible), en rouge",
    vd["etat"] == "ko" and "d = -1,0 cm ≤ 0" in vd["texte"].replace(C.NBSP, " "), vd)
sec, pl = planche("CSP · d utile ≤ 0 au-delà de 100 % (moment sur la face)")
vd = verif(sec, "inf_y")["verdicts"][0]
chk("d utile ≤ 0 AU-DELÀ de 100 % (un moment sur la face) : le verdict dit encore d = −1,0 cm ≤ 0, et aucun avertissement ne le répète",
    vd["etat"] == "ko" and "d = -1,0 cm ≤ 0" in vd["texte"].replace(C.NBSP, " ")
    and not any("d utile ≤ 0" in o for o in N.avertissements_de(sec)), (vd, N.avertissements_de(sec)))

sec, pl = planche("CSP neuve (sans effort)")
chk("une section sans aucun effort n'est jamais « Vérifié » : « Données incomplètes » (l'écran aussi)",
    sec["etat"] == N.ETAT_INCOMPLET and pl["R"]["etat_global"] == "ok", sec["etat"])
sec, pl = planche("CSP · hauteur insuffisante et τ > τ_adm,I")
chk("hauteur insuffisante et τ > τ_adm,I : « Non vérifié », verdicts rouges (hauteur, τ)",
    sec["etat"] == "Non vérifié" and verif(sec, "hauteur")["verdicts"][0]["etat"] == "ko" and verif(sec, "tau")["verdicts"][0]["etat"] == "ko",
    sec["etat"])
sec, pl = planche("Prédalle · neuve sollicitée")
chk("prédalle : l'en-tête dit la peau (« Prédalle », h_p = 6 cm), la coupe la teinte (6 cm de prédalle, 14 coulés en place)",
    sec["entete"]["cases"][0][0] == "Prédalle" and "(prédalle $h_{p}$ = 6 cm)" in sec["entete"]["cases"][0][1]
    and sec["coupe"]["h_pre"] == 60.0 and "coulé en place : 14 cm" in sec["coupe"]["h_pre_label"], sec["entete"]["cases"][0])

# les AVERTISSEMENTS de l'écran : en fin de planche (« ! »), seulement s'il y en a — le bloc « Observations » n'existe plus
AVERT = {
    "Prédalle · peau de 4 cm, base Ø16 qui en sort": ["sort de la peau de 4,0 cm"],
    "Prédalle · couches croisées sur la prédalle (renfort Y, base X)": ["à des niveaux trop proches pour se croiser"],
    "CSP · enrobage nul (lu 3 cm) et jeu premier lit 0,5": ["Enrobage nul : comme l'écran Streamlit, le calcul pose les couches avec un enrobage de 3 cm."],
    "CSP · nappes X et Y au même niveau (axe saisi)": ["les nappes X et Y inférieures sont calculées au même niveau", "Rendre l'axe saisi à l'automatique"],
}
faux = []
for nom, attendus in AVERT.items():
    sec, pl = planche(nom)
    av = " | ".join(N.avertissements_de(sec))
    faux += [(nom, a, av[:120]) for a in attendus if a not in av]
    if "<" in av or "&" in av:
        faux.append((nom, "balisage", av[:120]))
    if sec["verifs"][-1]["num"] != NP.NUM_AVERTISSEMENTS or sec["verifs"][-1]["verdicts"]:
        faux.append((nom, "pas en fin de planche, ou un verdict"))
sans = [(n, s_["section"]) for n in CAS if n not in AVERT for s_ in N.sections(CAS[n]) if N.avertissements_de(s_)]
chk("les avertissements de l'écran (rares : axe saisi, peau, croisement, enrobage nul) s'écrivent en fin de planche, seulement s'il "
    "y en a, en texte, sans verdict (l'état de la planche reste celui de l'écran) ; aucun autre cas n'en a ; plus de bloc « Observations »",
    not faux and not sans and not hasattr(N, "observations") and not hasattr(N, "TITRE_OBSERVATIONS"), (faux[:2], sans[:3]))
# ce qui n'est pas vérifié : une source pour l'écran et la note (api_web/dalle.non_verifies, la fin de texte_perimetre)
r_p = P.resultat(copy.deepcopy(CAS["Prédalle · neuve sollicitée"]))
_d_csp = CAS["CSP · M_y,inf = 28 (treillis par défaut)"]
r_c = P.resultat(copy.deepcopy(_d_csp))
_p0 = CAS["Prédalle · neuve sollicitée"]["dalles"][0]
chk("le périmètre de la prédalle, et celui de la dalle CSP (flèche, fissuration non vérifiées), sont le texte de l'écran "
    "(api_web/dalle.texte_perimetre, une source pour les deux) — entier dans les fenêtres de calcul ; sa dernière phrase "
    "(« Non vérifié(e)s : … ») sous le résultat et dans l'en-tête de chaque planche de la note",
    P.texte_perimetre(_p0) in r_p["volets"] and P.non_verifies(_p0) in r_p["resultat"] and P.texte_perimetre(_p0).endswith(P.non_verifies(_p0))
    and planche("Prédalle · neuve sollicitée")[0]["entete"]["note"] == P.non_verifies(_p0)
    and P.texte_perimetre(_d_csp["dalles"][0]) == P.PERIMETRE_CSP and "la flèche et la fissuration" in P.PERIMETRE_CSP
    and P.PERIMETRE_CSP in r_c["volets"] and P.NON_VERIFIEES_CSP in r_c["resultat"]
    and planche("CSP · M_y,inf = 28 (treillis par défaut)")[0]["entete"]["note"] == P.NON_VERIFIEES_CSP, "")


# ---------------------------------------------------------------- 5. le DOCUMENT : chaque planche tient, en deux colonnes, rien ne dépasse
def construire(etat):
    fd, chemin = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    try:
        return N.document(etat, {}, chemin)
    finally:
        os.remove(chemin)


debordes, trois = [], []
for nom, e in CAS.items():
    d_ = construire(e)
    if d_.warnings:
        debordes.append((nom, d_.warnings))
    if d_.colonnes != 2:
        trois.append(nom)
chk("chaque planche de chaque cas tient sur sa page (rien ne déborde)", not debordes, debordes[:2])
chk("… en DEUX colonnes, toutes (l'en-tête compact, les sections communes et les nappes sans libellés y font la place)",
    not trois, trois)
# MESURER CE QUE L'ŒIL VOIT : aucune ligne ne dépasse sa colonne en LARGEUR (le kit réduit une formule jusqu'à min_form, une
# ligne en gras jusqu'à 6 pt, puis la laisse déborder sur la colonne voisine) — la largeur des colonnes du kit, liée à
# NP.LARGEUR_COLONNE (la note s'en sert pour replier la ligne des armatures posées)
from ndc_pdf import mathx as MX  # noqa: E402
from ndc_pdf import styles as STY  # noqa: E402
from ndc_pdf.fonts import register_all, string_width  # noqa: E402
register_all()


def largeur_colonne(n_cols):
    S_ = N._style(n_cols)
    W_ = STY.A4L[0]
    return (W_ - S_.margin - (S_.margin + S_.coupe_w + S_.gap) - S_.col_gap * (n_cols - 1)) / n_cols


def trop_larges(sec, n_cols):
    """Les lignes d'une planche plus larges que leur colonne, à leur plus petit corps : formules (min_form), lignes en gras (6 pt)."""
    S_ = N._style(n_cols)
    gw, ms_ = largeur_colonne(n_cols), N._style(n_cols).ms()
    out = []
    for v in sec["verifs"]:
        for it in v["items"]:
            if it[0] == "f" and MX.measure(MX.parse(it[2]), ms_, S_.min_form).w > gw + 0.5:
                out.append((sec["section"], v["titre"], it[2][:50]))
            elif it[0] == "t" and string_width(it[1], S_.f_v, 6.0) > gw + 0.5:
                out.append((sec["section"], v["titre"], it[1][:50]))
    return out


larges = [(nom,) + x for nom, e in CAS.items() for sec in N.sections(e) for x in trop_larges(sec, 2)]
chk(f"aucune formule ni ligne en gras ne dépasse sa colonne en largeur (deux colonnes : {largeur_colonne(2):.0f} pt), y compris "
    "y_G à quatre couches et les armatures posées à trois renforts (repliées)",
    not larges and abs(largeur_colonne(2) - NP.LARGEUR_COLONNE[2]) < 1 and abs(largeur_colonne(3) - NP.LARGEUR_COLONNE[3]) < 1,
    (larges[:3], largeur_colonne(2), largeur_colonne(3)))
# MESURER CE QUE L'ŒIL VOIT : la coupe garde la place de ses étiquettes — aucun texte de la coupe n'en chevauche un autre (on
# espionne chaque texte écrit par la coupe, mesuré dans sa police et son corps ; les cotes verticales, écrites dans un repère
# tourné, sont à part), et sa hauteur reste au-dessus de COUPE_MIN (mesuré le 09/11/2026). Balayage : épaisseurs de 12 à
# 60 cm, prédalle et CSP, une coupe à un schéma (directions identiques) et à deux.
import ndc_pdf.section as SEC  # noqa: E402


def chevauchements_coupe(etat):
    boites, hauteurs, vrai_txt, vrai_ds = [], [], SEC.draw_text, STY.draw_section

    def espion(c, x, y, txt, font, size, color=None, align="left", **k):
        lw = vrai_txt(c, x, y, txt, font, size, color, align, **k)
        if boites and boites[-1][0] is not c:              # un nouveau document (le repli reconstruit la note)
            boites.clear()
        if abs(x) > 1.0:                                   # (0, y) : une cote verticale, dans un repère tourné
            g = x - (lw if align == "right" else (lw / 2 if align == "center" else 0))
            boites.append((c, txt, g, g + lw, y - 0.22 * size, y + 0.78 * size))
        return lw

    def espion_ds(c, x, y, w, h, coupe, *a, **k):
        hauteurs.append((h, coupe))
        return vrai_ds(c, x, y, w, h, coupe, *a, **k)
    SEC.draw_text, STY.draw_section = espion, espion_ds
    try:
        d_ = construire(etat)
    finally:
        SEC.draw_text, STY.draw_section = vrai_txt, vrai_ds
    out = [(a[1], b[1]) for i, a in enumerate(boites) for b in boites[i + 1:]
           if a[2] < b[3] - 0.5 and b[2] < a[3] - 0.5 and a[4] < b[5] - 0.5 and b[4] < a[5] - 0.5]
    basses = [round(h_) for h_, co in hauteurs if h_ < N.coupe_min({"coupe": co})]
    return out, d_, basses


def balayage():
    """Les planches (CSP, prédalle ; h de 12 à 60 cm ; un et deux schémas) dont la coupe se chevauche, déborde ou passe sous
    COUPE_MIN : (fautes, nombre d'essais)."""
    fautes, n_ = [], 0
    for type_ in ("coulee", "predalle"):
        for h in (12, 20, 30, 45, 60):
            base_ = cas((["type"], type_), (["h"], h), (S0 + ["My_inf"], 20), (S0 + ["Mx_sup"], 6), (S0 + ["V"], 30),
                        (F("sup_x") + ["+"], 1))
            n_ += 1
            ch_, d_, basses = chevauchements_coupe(base_)
            if ch_ or d_.warnings or basses:
                fautes.append((type_, h, ch_[:2], d_.warnings, basses))
    une = cas((F("inf_x") + [0, "dist"], "3,5"), (F("sup_x") + [0, "dist"], "3,5"), (S0 + ["My_inf"], 20))   # X et Y au même niveau
    n_ += 1
    ch_, d_, basses = chevauchements_coupe(une)
    if ch_ or d_.warnings or basses or len(N.sections(une)[0]["coupe"]["schemas"]) != 1:
        fautes.append(("un schéma", 20, ch_[:2], d_.warnings, basses))
    return fautes, n_


faux, n_essais = balayage()
chk(f"la coupe : aucun de ses textes n'en chevauche un autre, sa hauteur reste au-dessus de COUPE_MIN, rien ne déborde ({n_essais} "
    "planches : h de 12 à 60 cm, CSP et prédalle, un ou deux schémas)", not faux, faux[:3])
vrai_h = N.COUPE_H
N.COUPE_H = 120.0
try:
    faux_m, _ = balayage()
finally:
    N.COUPE_H = vrai_h
chk("mutation · COUPE_H trop petit : la coupe tassée sous COUPE_MIN, ses étiquettes se chevauchent — le contrôle rougit", bool(faux_m), faux_m[:1])
# le dernier recours : une planche trop longue (des avertissements nombreux) passe SEULE en trois colonnes — ses titres
# raccourcis et ses formules dans leur forme courte y tiennent, en hauteur comme en largeur
long_ = cas((["type"], "predalle"), (["h"], 40), (F("inf_y") + ["+"], 1), (S0 + ["My_inf"], 30), (S0 + ["Mx_sup"], 8),
            (S0 + ["V"], 40), (["sections", "+"], 1), (["sections", 1, "My_inf"], 12))
vrai_av = N.avertissements
N.avertissements = lambda d, r, i: (["Avertissement d'essai, assez long pour tenir sur trois lignes de la colonne de calcul, "
                                     "en fin de planche, après l'effort tranchant."] * 8 if i == 0 else [])
vrai_build = N._style


def espion_style(n_cols):
    s_ = vrai_build(n_cols)
    vrai_planche = s_.planche

    def planche_(d, doc, sec):
        if sec.get("n_cols") == 3:
            gw = largeur_colonne(3)
            titres_l.extend(v["titre"] for v in sec["verifs"] if string_width(v["titre"], s_.f_h, 7.0) > gw - (s_.s_tit * 1.25 + 8))
            vus.append(sec["section"])
            larges3.extend(trop_larges(sec, 3))
        return vrai_planche(d, doc, sec)
    s_.planche = planche_
    return s_


titres_l, vus, larges3 = [], [], []
N._style = espion_style
try:
    d_ = construire(long_)
finally:
    N.avertissements, N._style = vrai_av, vrai_build
chk("une planche trop longue : elle seule en trois colonnes (l'autre section en deux), chaque titre et chaque ligne y tiennent, "
    "rien ne déborde", not d_.warnings and d_.trois_colonnes == 1 and vus and not titres_l and not larges3 and set(vus) == {"Section A"},
    (d_.warnings, d_.trois_colonnes, vus, titres_l, larges3[:2]))


# ---------------------------------------------------------------- 6. le PDF, lu par pdftotext ; le routeur
e = CAS["deux dalles : CSP puis prédalle"]
octets = N.pdf(e, {"nom_projet": "Chantier Essai", "partie": "Dalles"})
n_pages = octets.count(b"/Type /Page\n") + octets.count(b"/Type /Page ") + octets.count(b"/Type /Page>")
chk("la note est un PDF : la garde et une planche par section (1 + 3 pages)", octets[:5] == b"%PDF-" and n_pages == 4, (octets[:8], n_pages))
tmp = tempfile.mkdtemp()
with open(os.path.join(tmp, "n.pdf"), "wb") as f:
    f.write(octets)
txt = subprocess.run(["pdftotext", "-layout", os.path.join(tmp, "n.pdf"), "-"], capture_output=True, text=True).stdout.replace("\xa0", " ")
shutil.rmtree(tmp, ignore_errors=True)
d0, d1 = e["dalles"]
g0 = P._calculer(d0)["sections"][0]["grandeurs"]
g1 = P._calculer(d1)["sections"][0]["grandeurs"]
attendus = [f"{d0['nom']} — Section A", f"{d1['nom']} — Section A", f"{d1['nom']} — Section B", "Chantier Essai",
            f"{ND.fn(g0['As_req_inf_y'], 0)} mm", f"{ND.fn(g1['d_inf_x'], 1)} cm", "DALLE PLEINE", "PRÉDALLE", "BÉTON C30/37",
            "ACIER B500", "Non vérifiées : la flèche et la fissuration.", "Non vérifiés : la phase provisoire", "Hauteur minimale",
            "Hauteur utile minimale", "Enrobage mécanique", "Sections d'acier minimale et maximale",
            "Armatures posées : Treillis 10/10/100/100 — 785 mm²/m",
            f"taux {C.pc(C.pct(next(v for v in P._calculer(d0)['sections'][0]['verifications'] if v['cle'] == 'hauteur')['taux']))} %",
            "VÉRIFIÉ"]
manque = [a for a in attendus if a not in txt]
chk("pdftotext lit dans la note le nom de chaque dalle et de chaque section, le chantier, des chiffres de l'écran (A_s,req, d), "
    "l'en-tête compact (le type, le béton, l'acier, ce qui n'est pas vérifié), la hauteur détaillée, les sections communes, les "
    "armatures posées et leur taux", not manque, manque)
chk("… et plus de bloc « Observations » (ni DIMENSIONS, ni MATÉRIAUX : l'en-tête les dit), ni « On prend »",
    not any(x in txt for x in ("OBSERVATIONS", "Observations", "DIMENSIONS", "MATÉRIAUX", "On prend")),
    [x for x in ("OBSERVATIONS", "Observations", "DIMENSIONS", "MATÉRIAUX", "On prend") if x in txt])

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
mutation("le verdict de l'acier excessif retiré (la comparaison générique de la nappe)",
         lambda: remplace(NP, "verdict_nappe", lambda v, rs: dict(etat="ko", texte="12566 mm² posés ≥ 707 mm² requis")),
         lambda: "acier excessif" in verif(planche("CSP · acier excessif (Ø40/100 en bas)")[0], "inf_y")["verdicts"][0]["texte"])
mutation("les avertissements retirés", lambda: remplace(N, "avertissements", lambda d, r, i: []),
         lambda: "sort de la peau" in " ".join(N.avertissements_de(planche("Prédalle · peau de 4 cm, base Ø16 qui en sort")[0])))
mutation("la phrase « Non vérifiés » retirée de l'en-tête", lambda: remplace(NP, "entete", (lambda vrai: lambda d, rs: dict(vrai(d, rs), note=""))(NP.entete)),
         lambda: planche("Prédalle · neuve sollicitée")[0]["entete"]["note"].startswith("Non vérifiés : la phase provisoire"))
mutation("la décomposition de y_G écrite sans la vérifier (la tolérance infinie : une somme fausse s'écrirait)",
         lambda: remplace(NP, "TOLERANCE_AXE", 1e9),
         lambda: all(abs(a - b) <= 0.051 for nom_ in CAS for s_ in N.sections(CAS[nom_]) for a, b in decompositions(s_)))
mutation("les sections minimale et maximale répétées dans chaque nappe (la nappe rend aussi A_s,max)",
         lambda: remplace(NP, "verif_nappe", (lambda vrai: lambda d, s, rs, v, num: dict(vrai(d, s, rs, v, num), items=vrai(d, s, rs, v, num)["items"]
                                                                                       + [("f", "", r"A_{s,max} = 0,04 \cdot b \cdot h")]))(NP.verif_nappe)),
         lambda: all("\n".join(formules(v) for v in sec_["verifs"]).count("A_{s,max} = ") == 1 for sec_ in N.sections(CAS[nom_m])))
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

# -*- coding: utf-8 -*-
"""Générateur SAF — fondation de grue : l'ÉCRAN du site (api_web/saf.py, saf_saisie.py, saf_resultat.py, dessin_saf.py),
l'EXPORT du SAF, le lien avec le module de raideur du sol (api_web/sol.page_resultats) et les grues enregistrées (le carnet).
Cahier des charges § 3, § 7, § 8, § 9, les essais B1 à B5 côté écran ; lot 5 (version 4, 15/11/2026) : U1 à U11, S0, l'équilibre
(T18) à l'écran, les efforts SCIA (E1 à E4). La note PDF : tests/test_saf_grue_note.py.

Ce que la suite prouve :
  1. L'ÉTAT — une note neuve aux valeurs du bureau, les réactions vides ; la relecture ; les fondations de la note.
  2. LA SAISIE — chaque chemin écrit ce qu'il dit ; un refus rend la note intacte, avec sa raison ; le contour se
     dessine sans déplacer la grue ; le bord pris pour repère des cotes.
  3. L'ÉCRAN DIT LE MOTEUR — la contrainte, le tableau par k, les positions, les contrôles sont les nombres du moteur ;
     le plan porte ce que les gestes règlent ; ce qui ne va pas se voit où il est.
  4. B1 à B5 — avec la vraie bibliothèque des châssis, à l'écran et dans la note.
  5. « PROPOSER », le lien avec la raideur du sol, les grues enregistrées.
  6. L'EXPORT — LE fichier SAF (celui du moteur, à l'octet) ; l'ancienne archive d'une note de plusieurs fondations.
  7. MUTATIONS — arbre miroir (api_web copié, le reste du dépôt lié), contrôles rapides.

    /usr/bin/python3.12 tests/test_api_web_saf.py [--rapide] [--racine dossier]
"""
import base64
import copy
import hashlib
import html as H
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(ICI)
ARGS = sys.argv[1:]
RACINE_MOTEUR = ARGS[ARGS.index("--racine") + 1] if "--racine" in ARGS else RACINE
RAPIDE = "--rapide" in ARGS
os.chdir(RACINE_MOTEUR)
sys.path.insert(0, RACINE_MOTEUR)

import numpy as np  # noqa: E402

from api_web import commun as C  # noqa: E402
from api_web import registre as REG  # noqa: E402
from api_web import routeur  # noqa: E402
from api_web import saf as M  # noqa: E402
from api_web import saf_resultat as SR  # noqa: E402
from api_web import saf_saisie as SS  # noqa: E402
from api_web import sol as SOL  # noqa: E402
from grues.chassis import acces as BIB  # noqa: E402
from saf.fondation_grue import ecriture as EC  # noqa: E402
from saf.fondation_grue import efforts as EF  # noqa: E402
from saf.fondation_grue import projet as PR  # noqa: E402
from saf.fondation_grue import proposition as PP  # noqa: E402
from saf.fondation_grue import scia as SCM  # noqa: E402
from saf.fondation_grue import verification as V  # noqa: E402

OK, KO = [], []
INFOS = {"nom": "Grue G1", "chantier": "Chantier A", "indice": "B", "date": "14/11/2026"}
L_T7 = [[0, 0], [12, 0], [12, 7], [7, 7], [7, 12], [0, 12]]


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{str(info)[:600]}]" if info != "" and not cond else ""))


def q(**k):
    """Une requête au routeur, comme l'écran."""
    return json.loads(routeur.traiter(json.dumps(dict(module="saf", **k))))


def mod(etat, chemin, valeur, **k):
    r = q(action="modifier", etat=etat, chemin=[str(c) for c in chemin], valeur=valeur, **k)
    assert r["ok"], r.get("erreur")
    return r


def txt(h):
    """Un fragment de l'écran en texte (sans balises, espaces simples)."""
    return " ".join(H.unescape(re.sub(r"<[^>]+>", " ", h)).replace(" ", " ").replace(" ", " ").split())


def f0(r):
    return r["etat"]["fondations"][0]


def pdf_texte(octets):
    return " ".join(subprocess.run(["pdftotext", "-", "-"], input=octets, capture_output=True, check=True).stdout.decode("utf-8")
                    .replace(" ", " ").replace(" ", " ").split())


def fr(x, nd=1):
    return C.fr(x, nd).replace(" ", " ").replace(" ", " ")


# ------------------------------------------------------------------ les contrôles RAPIDES (rejoués par les mutations)
def rapides():
    r0 = q(action="nouveau")
    el = f0(r0)
    # (lot 5, § 4 et § 12 : maille 0,20 m, bandes de 0,50 m, kg/cm², enrobage 5 cm, treillis CHOISIS Ø12 e 15 en bas et en haut)
    chk("une note neuve : une fondation « F1 » aux valeurs du bureau — 8,00 × 8,00 × 0,50 m, C30/37, k = 2 ; 5 ; 10 ; 20 ; 30 (cochés SCIA : 2 ; 5 ; 10), "
        "châssis saisi 6 × 6 m, plaques 0,50 × 0,50, sans plot, bandes 0,50 m, maille 0,20 m, limite 0,5 kg/cm², unité kg/cm², enrobage 5 cm, "
        "treillis Ø12 e 15 en bas et en haut, renforts Ø auto, ancrage 50 Ø ; les réactions, les valeurs SCIA et les textes du projet VIDES ; le "
        "carnet n'est pas dans la note",
        r0["ok"] and el["nom"] == "F1" and (el["dalle"]["Lx"], el["dalle"]["Ly"], el["dalle"]["h"], el["dalle"]["beton"]) == (8.0, 8.0, 0.5, "C30/37")
        and [(x["k"], x["scia"]) for x in el["sol"]["k"]] == [(2.0, True), (5.0, True), (10.0, True), (20.0, False), (30.0, False)]
        and el["grue"]["chassis"] == "saisie" and (el["grue"]["appuis"], el["grue"]["rect"]) == ("rectangle", {"a": 0.5, "b": 0.5, "angle": 0})
        and el["grue"]["plots"]["type"] == "aucun" and el["limite"] == 0.5 and el["bandes"]["largeur"] == 0.5 and el["options"]["maille"] == 0.2
        and el["options"]["unite"] == "kgcm2" and el["armatures"]["inf"] == {"x": {"d": 12, "e": 150}, "y": {"d": 12, "e": 150}, "xy": False}
        and el["armatures"]["sup"] == el["armatures"]["inf"] and (el["armatures"]["renfort"], el["armatures"]["ancrage"]) == ("auto", 50.0)
        and all(el["scia"][c] is None for c in PR.CHAMPS_SCIA)
        and el["reactions"] == [{"nom": "", "P": [None] * 4}] and set(el["projet"].values()) == {""} and "carnet" not in r0["etat"], el)
    chk("sans réaction : le poids propre seul (σ = 25·h = 12,5 kPa = 0,13 kg/cm², l'unité par défaut), la note est « Données incomplètes », le "
        "bloc « Grue et charges » dit « réactions à saisir », et l'export est éteint, avec sa raison écrite dessous — rien en fenêtre surgissante",
        r0["resume"]["niveau"] == "n" and "poids propre seul" in r0["resume"]["gouvernant"] and "0,13\u00a0kg/cm²" in r0["resume"]["gouvernant"]
        and "à saisir" in txt(r0["donnees"]) and re.search(r'<button type="button" class="btn primary sf-exp-b" disabled[ >]', r0["resultat"]) is not None
        and 'data-telecharger="saf"' not in r0["resultat"]
        and "Aucun jeu de réactions complet" in txt(r0["resultat"]), (r0["resume"], txt(r0["resultat"])[-300:]))
    # U1 (lot 5, § 4) : QUATRE blocs comme le module « Poutres béton armé », AUCUN sous-titre à l'intérieur, un seul ouvert ; les champs
    # gardés trouvent leur ligne, ceux du § 2 « On retire » ont disparu
    chemins = set(re.findall(r'data-chemin="([^"]+)"', r0["donnees"]))
    garde = {"projet/grue", "dalle/forme", "dalle/Lx", "dalle/Ly", "dalle/h", "dalle/beton", "sol/k/0/k", "sol/k/0/scia", "sol/k/+",
             "grue/a", "grue/b", "grue/appuis", "grue/rect/a", "grue/rect/b", "grue/rect/angle", "grue/plots/type", "grue/ex", "grue/ey",
             "grue/beta", "grue/dmin", "reactions/0/nom", "reactions/0/P/0", "reactions/+", "bandes/largeur", "limite", "options/maille",
             "options/anneau", "options/c1x", "options/c1y", "options/c2x", "options/c2y", "options/rond", "options/aretes", "options/nonlinear",
             "options/h_min", "options/h_max", "options/h_pas", "options/unite", "carnet/garder", "armatures/enrobage", "armatures/inf/x/d",
             "armatures/inf/x/e", "armatures/inf/xy", "armatures/sup/x/d", "armatures/sup/x/e", "armatures/sup/xy", "armatures/sup/e10",
             "armatures/renfort", "armatures/ancrage", "scia/sym"}
    retires = {"armatures/treillis_inf", "armatures/treillis_sup", "options/torsion", "options/recouvrement", "options/l_max",
               "options/gamma_s", "options/jeu", "options/arrondi"}
    titres = re.findall(r'<span class="tt">([^<]+)</span>', r0["donnees"])
    chk("U1 la colonne de gauche : 4 titres de blocs (1 Dalle et sol, 2 Grue et charges, 3 Ferraillage, 4 Options rangé à part), AUCUN "
        "sous-titre à l'intérieur, un seul bloc ouvert ; les champs gardés y sont, ceux du § 2 (treillis automatique, torsion, recouvrement, "
        "longueur des barres, options du moteur poutre) ont disparu ; le projet et l'indice restent dans la barre du haut",
        re.findall(r'data-sec="([a-z]+)"', r0["donnees"]) == ["dalle", "grue", "armatures", "options"]
        and re.findall(r'data-sec="([a-z]+)" data-secondaire', r0["donnees"]) == ["options"] and r0["donnees"].count('class="sec open"') == 1
        and titres == ["Dalle et sol", "Grue et charges", "Ferraillage", "Options"] and 'class="grp"' not in r0["donnees"]
        and garde <= chemins and not (retires & chemins) and "projet/projet" not in chemins and "projet/indice" not in chemins,
        (titres, sorted(garde - chemins), sorted(retires & chemins), r0["donnees"].count('class="grp"')))
    type_ = M._etat_type()
    r = q(action="calculer", etat=type_)
    el = f0(r)
    calc = M.calculer(PR.lire(el))
    par_k = {x["k"]: x for x in calc["par_k"]}
    gov = max((par_k[k] for k in (2.0, 5.0, 10.0)), key=lambda x: x["sigma_max"])
    res = txt(r["resultat"])
    # U2 (lot 5, § 5.2) : les cartes du module poutre — la ligne d'étapes, SOL, ÉQUILIBRE, CONTRÔLES, ARMATURES, pas de grand chiffre
    chk("U2 la dalle type : l'écran dit le MOTEUR dans la carte SOL — « σ max = 1,12 kg/cm² > 0,50 kg/cm² · cas 3.a · k = 10 », le taux "
        f"{C.pct(gov['sigma_max'] / (0.5 * V.KG_CM2))} % ; la carte est « Non conforme » (rouge) ; aucun grand chiffre ; la ligne d'étapes en tête ; "
        "les cartes SOL, ÉQUILIBRE, CONTRÔLES, ARMATURES et leurs pastilles d'état ; le pied « Méthode du bureau · moteur Dalles » et sa version",
        f"σ max = {fr(gov['sigma_max'] / V.KG_CM2, 2)} kg/cm² > 0,50 kg/cm² · cas {gov['cas_max']} · k = 10" in res and gov["cas_max"] == "3.a"
        and fr(gov["sigma_max"] / V.KG_CM2, 2) == "1,12" and r["resume"]["niveau"] == "r"
        and r["resume"]["taux"] == C.pct(gov["sigma_max"] / (0.5 * V.KG_CM2))
        and 'chip xl' not in r["resultat"] and 'class="verdict' not in r["resultat"] and '<section class="et-bloc nok" data-theme="0">' in r["resultat"]
        and "Sol Non conforme" in res and "Méthode du bureau · moteur Dalles v2.2" in res
        and re.findall(r'<span class="n">([^<]+)</span><span class="e">', r["resultat"]) == ["Sol", "Équilibre", "Contrôles", "Armatures"]
        and re.findall(r'<li class="[a-z]*"><span class="n">\d</span><span class="t">([^<]+)</span>', r["resultat"])
        == ["Données", "Vérification", "SAF", "Efforts SCIA", "Note"], (res[:300], r["resume"]))
    lignes = re.findall(r'<tr class="([a-z ]+)"><td><button[^>]*data-valeur="([0-9.]+)"[^>]*>[^<]*</button></td><td class="n"><span class="chip ([a-z])">([^<]+)</span></td>'
                        r'<td class="n">([^<]+)</td><td class="n">([^<]+)</td>.*?<input type="checkbox" data-agir="1" data-chemin="sol/k/(\d)/scia"( checked)?',
                        r["resultat"], re.S)
    attendu = [(str(k), "r", fr(par_k[k]["sigma_max"] / V.KG_CM2, 2), fr(par_k[k]["w_max"]), fr(par_k[k]["decolle"]) + " %", str(i), " checked" if i < 3 else "")
               for i, k in enumerate((2.0, 5.0, 10.0, 20.0, 30.0))]
    chk("le tableau par k (dépliable dans la carte SOL) : k | σ max (kg/cm²) | tassement max | surface décollée | case SCIA — les cinq k, les "
        "nombres du moteur, chaque ligne à la couleur de sa contrainte ; la ligne du k gouvernant est marquée ; les cases règlent « sol/k/i/scia »",
        [(a[1], a[2], a[3], a[4], a[5].replace(" ", " "), a[6], a[7]) for a in lignes] == attendu
        and [a[0] for a in lignes].count("nok vu") == 1 and lignes[2][0] == "nok vu", lignes)
    sm = V.sigma_moyen(calc["modele"])
    chk("σ moyen (un repère, dans la carte SOL) : (poids de la dalle + réactions du jeu le plus lourd) ÷ aire = (800 + 1 580) ÷ 64 = 37,2 kPa = "
        "0,38 kg/cm² ; le tassement max et la surface décollée ; les contrôles : appuis dans la dalle, distance au bord 0,75 m ≥ 0,50 m ; "
        "« 16 cas » (le tableau des positions dans sa fenêtre)",
        abs(sm["sigma"] - 37.1875) < 1e-9 and "σ moyen repère (800 + 1 580) ÷ 64,0 m² = 0,38 kg/cm²" in res and "Distance au bord OK 0,75 m ≥ 0,50 m" in res
        and "Tassement max valeur w max = " in res and "Surface décollée valeur" in res
        and "Appuis dans la dalle OK" in res and "Cas générés 16 cas 16 positions" in res and 'id="ck-s2"' in r["volets"], res[300:900])
    chk("les contrôles sont des LIGNES DE L'ÉTABLI (C.ligne_verif : « el vc », l'état en un mot) — celles des autres modules — et chacune ouvre "
        "le détail des contrôles (fenêtre s1)",
        r["resultat"].count('<div class="el vc" data-calcul="s1" title="Voir le détail">') == 2 and 'class="sf-ctl' not in r["resultat"]
        and r["resultat"].count('<span class="et-mot p">OK</span>') >= 2, re.findall(r'<div class="el[^>]*>', r["resultat"]))
    svg = r["vues"][0]["svg"]
    glisser = json.loads(H.unescape(re.search(r'<g class="sf-chassis" data-plan-glisser="([^"]+)"', svg).group(1)))
    tourner = json.loads(H.unescape(re.search(r'data-plan-tourner="([^"]+)"', svg).group(1)))
    chk("le plan se manipule : le châssis porte ce que son déplacement règle (grue/position, e_x et e_y de départ, la grille 5 cm, les accroches — "
        "le centre et les axes de la dalle, ses sommets et les milieux de ses bords, « une plaque à la distance mini » de chaque bord : ± 0,25 m) ; "
        "la poignée de rotation, ses accroches 0, 45, 90° ; les cotes L_x et L_y se tapent ; P1 à P4 ; les bandes BX1 à BY2",
        glisser["chemin"] == "grue/position" and (glisser["x"], glisser["y"], glisser["pas"], glisser["ech"]) == (0.0, 0.0, 0.05, 100.0)
        and {0.0, -4.0, 4.0, -0.25, 0.25} <= set(glisser["ax"]) and {0.0, -0.25, 0.25} <= set(glisser["ay"])
        and tourner["chemin"] == "grue/beta" and {0.0, 45.0, 90.0} <= set(tourner["accroches"]) and tourner["cible"] == ".sf-chassis"
        and all(f'"chemin":"{c}"' in H.unescape(svg) for c in ("dalle/Lx", "dalle/Ly")) and svg.count('class="sf-appui"') == 4
        and all(f">{t}</text>" in svg for t in ("P1", "P2", "P3", "P4", "BX1", "BX2", "BY1", "BY2")), (glisser, tourner))
    png = base64.b64decode(re.search(r'href="data:image/png;base64,([^"]+)"', svg).group(1))
    ent = r["vues"][0]["entete"]
    menus = re.findall(r'<select data-agir="1" data-chemin="(vue/[a-z]+)"[^>]*>(.*?)</select>', ent, re.S)
    premiers = {c: re.findall(r'<option value="([^"]*)"( selected)?>([^<]+)</option>', o)[0] for c, o in menus}
    chk("U8 / U3 la barre du centre, sur UNE ligne : la carte (Contrainte, Tassement, Armatures, Sans carte), le cas et le k (« Enveloppe des "
        "cas », « Enveloppe des k » en tête et choisis) — plus d'onglets ; la carte, une image PNG découpée au contour, est l'ENVELOPPE des 16 cas "
        "et des 3 k cochés SCIA : son maximum est celui du moteur, pointé en kg/cm²",
        png[:8] == b"\x89PNG\r\n\x1a\n" and f"σ max {fr(gov['sigma_max'] / V.KG_CM2, 2)} kg/cm²" in svg and 'clip-path="url(#saf_dalle)"' in svg
        and ent.startswith('<div class="et-ent sf-ent sf-barre">') and "<br" not in ent and 'class="l1"' not in ent
        and "vue/onglet" not in ent and [c for c, _ in menus] == ["vue/carte", "vue/cas", "vue/k"]
        and premiers["vue/cas"] == ("", " selected", "Enveloppe des cas") and premiers["vue/k"] == ("", " selected", "Enveloppe des k")
        and re.findall(r'<option value="([a-z]+)"', dict(menus)["vue/carte"]) == ["sigma", "w", "armatures", "aucune"], (len(png), premiers))
    # U3 : l'enveloppe montre les QUATRE coins pareils — les 8 positions du jeu 3 donnent toutes la même contrainte, deux par coin
    ch3 = SR.champ(PR.lire(el), calc, SR.synthese(PR.lire(el), calc))
    d3 = ch3["dalle"]
    sig = ch3["valeurs"].reshape(d3.ny + 1, d3.nnx)
    coins = [float(sig[:d3.ny // 2 + 1, :d3.nx // 2 + 1].max()), float(sig[:d3.ny // 2 + 1, d3.nx // 2:].max()),
             float(sig[d3.ny // 2:, :d3.nx // 2 + 1].max()), float(sig[d3.ny // 2:, d3.nx // 2:].max())]
    chk("U3 la carte en enveloppe par défaut : sur la dalle type, les quatre quarts de la dalle montrent la MÊME contrainte maximale, "
        "110,3 kPa avec la maille de 0,20 m (deux positions par coin) ; une position seule (3.a) ne charge qu'un coin",
        max(coins) - min(coins) < 1e-6 * max(coins) and fr(max(coins)) == "110,3" and d3.a == 0.2 and ch3["maxi"] == max(coins)
        and len({round(c, 6) for c in [float(x) for x in (lambda s3: [s3[:d3.ny // 2 + 1, :d3.nx // 2 + 1].max(), s3[:d3.ny // 2 + 1, d3.nx // 2:].max(),
                                                                   s3[d3.ny // 2:, :d3.nx // 2 + 1].max(), s3[d3.ny // 2:, d3.nx // 2:].max()])(
            SR.enveloppe_champs([x for x in calc["par_k"] if x["k"] == 10.0], "3.a")[0].reshape(d3.ny + 1, d3.nnx))]}) > 1, coins)
    # la carte « Armatures » (lot 5, § 5.1) : le principe sur le plan, « Bas · Haut » à la place des listes cas et k
    ra_ = mod(type_, ["vue", "carte"], "armatures")
    ent_a, svg_a = ra_["vues"][0]["entete"], ra_["vues"][0]["svg"]
    fer_t = M.armatures(PR.lire(el), calc)
    bas = [x for x in fer_t["renforts"] if x["face"] == "inf"]
    rh_ = mod(ra_["etat"], ["vue", "face"], "sup")
    haut = [x for x in fer_t["renforts"] if x["face"] == "sup"]
    chk(f"la carte Armatures, Bas : les zones rouges, les {len(bas)} renforts du bas (bandes et locaux) avec leur repère et leurs barres "
        "(« R1 · 2 Ø12 »), la petite légende ; la barre du centre : la carte puis « Bas · Haut », sans les listes cas et k ; Haut : les renforts "
        "du haut ; une face inconnue est refusée",
        '<g class="sf-armatures">' in svg_a and svg_a.count('class="sf-renfort"') == len(bas) > 0 and 'class="sf-rouge"' in svg_a
        and f">{bas[0]['rep']} · {bas[0]['n']} Ø{bas[0]['d']}</text>" in svg_a and 'class="sf-arm-leg"' in svg_a
        and re.findall(r'data-chemin="vue/face" data-valeur="([a-z]+)"', ent_a) == ["inf", "sup"] and "vue/cas" not in ent_a and "vue/k" not in ent_a
        and rh_["vues"][0]["svg"].count('class="sf-renfort"') == len(haut) and f"{haut[0]['rep']} · " in rh_["vues"][0]["svg"]
        and "bas ou haut" in (q(action="modifier", etat=type_, chemin=["vue", "face"], valeur="milieu")["message"] or "").lower(),
        (len(bas), len(haut), svg_a.count('class="sf-renfort"')))
    rmx = q(action="modifier", etat=type_, chemin=["vue", "carte"], valeur="m_inf_x")
    chk("la carte « Moments » du lot 4 n'existe plus : la demander est refusé avec sa raison, la note reste intacte",
        rmx["message"] == "Carte : contrainte, tassement, armatures ou sans carte" and rmx["etat"] == type_, rmx["message"])
    # la carte ARMATURES de la colonne, U4 « Non vérifiés »
    res_a = txt(r["resultat"])
    chk("la carte ARMATURES : les treillis choisis (« bas Ø12 e 15 · haut Ø12 e 15 »), pour chaque direction (« BX1 = BX2 ») la flexion (m, "
        "m_Rd du treillis, les renforts), la hauteur de béton (d_min ≤ d du moteur Dalles), l'effort tranchant (τ ≤ τ_adm,I) ; les zones rouges ; "
        "« Écart module / SCIA : valeurs SCIA à saisir » ; sa fenêtre de calcul",
        "Treillis choisi bas Ø12 e 15 · haut Ø12 e 15" in res_a and "Flexion BX1 = BX2" in res_a and "Flexion BY1 = BY2" in res_a
        and re.search(r"Hauteur BX1 = BX2 \d+ % d min = [\d,]+ ≤ d = 43,0 cm", res_a) is not None
        and re.search(r"Effort tranchant BX1 = BX2 \d+ % τ = [\d,]+ ≤ τ adm,I = [\d,]+ N/mm² · V à d = 94,0 kN", res_a) is not None and "Zones rouges · bas" in res_a and "valeurs SCIA à saisir" in res_a
        and 'id="ck-a0"' in r["volets"], res_a[res_a.find("Armatures"):][:600])
    chk("U4 la ligne « Non vérifiés » : le poinçonnement (le moteur Dalles ne le vérifie pas), la fissuration, les efforts horizontaux et les "
        "moments aux pieds de la grue, la torsion, les ancrages et les recouvrements (le principe n'est pas un plan d'exécution)",
        "Non vérifiés : le poinçonnement, la fissuration, les efforts horizontaux et les moments aux pieds de la grue, la torsion, les ancrages et "
        "les recouvrements (le principe n'est pas un plan d'exécution)." in res_a, res_a[res_a.find("Non vérifiés"):][:300])
    rc8 = mod(type_, ["grue", "plots", "type"], "coule")
    res8 = txt(rc8["resultat"])
    chk("U4 plots coulés en place : « Non vérifiés » nomme le ferraillage des plots coulés en place",
        "Non vérifiés : le poinçonnement, le ferraillage des plots coulés en place, la fissuration" in res8, res8[res8.find("Non vérifiés"):][:200])
    # U6 : l'unité des contraintes de sol, partout ensemble
    rk, rm = mod(type_, ["options", "unite"], "kpa"), mod(type_, ["options", "unite"], "mpa")
    tk, tm = txt(rk["resultat"]), txt(rm["resultat"])
    chk("U6 l'unité : dalle type, k = 10, maille 0,20 — 1,12 kg/cm² par défaut, 110,3 kPa, 0,110 MPa ; la carte (légende et valeur pointée), "
        "la carte SOL, le tableau par k, σ moyen et le champ de la limite changent ENSEMBLE, sans unité mélangée",
        "σ max = 110,3 kPa > 49,0 kPa" in tk and "σ max kPa" in tk and "= 37,2 kPa" in tk and "kg/cm²" not in tk + rk["vues"][0]["svg"]
        and "σ max 110,3 kPa" in rk["vues"][0]["svg"] and "σ (kPa)" in rk["vues"][0]["svg"]
        and "σ max = 0,110 MPa > 0,049 MPa" in tm and "MPa" in rm["vues"][0]["svg"] and "kg/cm²" not in tm
        and re.search(r'data-chemin="limite"[^>]*value="49,033"', rk["donnees"]) is not None
        and re.search(r'data-chemin="limite"[^>]*value="0,5"', r["donnees"]) is not None and "kPa" in txt(rk["donnees"]),
        (tk[:220], tm[:160]))
    rl = mod(rk["etat"], ["limite"], "60")
    chk("U6 la limite se saisit dans l'unité choisie : 60 kPa = 0,6118 kg/cm² gardés dans la note, relus « 60 » en kPa",
        abs(f0(rl)["limite"] - 60 / V.KG_CM2) < 1e-12 and re.search(r'data-chemin="limite"[^>]*value="60"', rl["donnees"]) is not None, f0(rl)["limite"])
    # U7 : le dégradé à plusieurs couleurs, la limite en trait noir sur la légende et en ligne noire sur le plan — la barre de la
    # légende, de bas en haut : un bleu foncé, puis du vert, du jaune, jusqu'au rouge
    barre = re.findall(r'<rect x="[-\d.]+" y="([-\d.]+)" width="[-\d.]+" height="[-\d.]+" fill="#([0-9a-f]{6})"/>', svg)
    couleurs = [tuple(int(c[i:i + 2], 16) for i in (0, 2, 4)) for _, c in sorted(barre, key=lambda x: -float(x[0]))][:32]
    chk("U7 la carte : un dégradé à plusieurs couleurs (« turbo », du bleu foncé au rouge) de 0 au maximum ; un trait NOIR à 0,50 kg/cm² sur la "
        "légende (« 0,50 limite »), une ligne NOIRE σ = 0,50 kg/cm² sur le plan avec son étiquette ; le tassement : le même dégradé, sans ligne",
        '<g class="sf-iso"' in svg and 'stroke="#000000"' in svg and ">0,50 kg/cm²</text>" in svg and ">0,50 limite</text>" in svg
        and 'class="sf-lim"' in svg and len(couleurs) >= 24 and couleurs[0][2] > couleurs[0][0] and max(couleurs[0]) < 100 and couleurs[-1][0] > 3 * couleurs[-1][2]
        and any(c[1] > 200 and c[0] < 120 for c in couleurs) and any(c[0] > 200 and c[1] > 180 and c[2] < 100 for c in couleurs)
        and '<g class="sf-iso"' not in mod(type_, ["vue", "carte"], "w")["vues"][0]["svg"], svg.count("sf-iso"))
    # (l'échelle monte jusqu'à la limite quand la contrainte reste dessous : le poids propre seul, 0,13 kg/cm² sous 0,50 — le trait
    # de la limite est EN HAUT de la barre, pas au-dessus d'elle)
    s0 = r0["vues"][0]["svg"]
    cadre = re.search(r'<rect x="[-\d.]+" y="([-\d.]+)" width="[-\d.]+" height="([-\d.]+)" fill="none" stroke="#8793a1"', s0)
    lim0 = re.search(r'<line class="sf-lim" x1="[-\d.]+" y1="([-\d.]+)"', s0)
    chk("U7 sans réaction (0,13 kg/cm² sous la limite de 0,50) : l'échelle va de 0 à la LIMITE — son trait noir est en haut de la barre",
        bool(cadre and lim0) and abs(float(lim0.group(1)) - float(cadre.group(1))) < 0.6 and ">0,50 limite</text>" in s0,
        (cadre and cadre.groups(), lim0 and lim0.group(1)))
    # ---- les plots : les dimensions par défaut de chaque sorte (le propriétaire, 15/11/2026 : préfabriqués 2,50 × 2,50 × 0,35 m ;
    # coulés en place 2,00 × 2,00 × 0,35 m, le cahier du lot 4 § 8.2) — une dimension saisie à la main est gardée
    pf = mod(type_, ["grue", "plots", "type"], "prefabrique")
    pc = mod(pf["etat"], ["grue", "plots", "type"], "coule")
    pm = mod(mod(pf["etat"], ["grue", "plots", "a"], "3")["etat"], ["grue", "plots", "type"], "coule")
    dims = lambda r_: tuple(f0(r_)["grue"]["plots"][c] for c in ("a", "b", "h"))
    chk("les plots préfabriqués : 2,50 × 2,50 × 0,35 m par défaut ; coulés en place : 2,00 × 2,00 × 0,35 m ; une dimension saisie (a = 3,00 m) "
        "reste quand on change de sorte",
        dims(pf) == (2.5, 2.5, 0.35) and dims(pc) == (2.0, 2.0, 0.35) and dims(pm) == (3.0, 2.5, 0.35), (dims(pf), dims(pc), dims(pm)))
    # ---- la saisie
    r1 = mod(type_, ["grue", "position"], [0.25, 0.25])
    r2 = mod(type_, ["sol", "k", 1, "k"], "2")
    r3 = mod(type_, ["dalle", "Lx"], "abc")
    chk("déplacer le châssis (le geste rend [e_x, e_y]) : la grue est à +0,25 ; +0,25 et le maximum change ; une saisie refusée laisse la note "
        "INTACTE et dit pourquoi — deux k égaux, un nombre illisible",
        (f0(r1)["grue"]["ex"], f0(r1)["grue"]["ey"]) == (0.25, 0.25) and r1["resume"]["taux"] != r["resume"]["taux"] and r1["message"] is None
        and r2["message"] == "Deux valeurs de k sont égales" and r2["etat"] == type_ and "illisible" in r3["message"] and r3["etat"] == type_,
        (r2["message"], r3["message"]))
    long58 = "Élément de rive très long du bâtiment B, file 12 à 18, N+3"
    rn, rn2 = mod(type_, ["nom"], long58), mod(type_, ["nom"], long58 + " — nord")
    chk("le nom d'une fondation : 60 caractères au plus, la borne des autres modules du site — un nom de 58 caractères est gardé EN ENTIER (il "
        "cède à l'écran, pas dans la note) ; au-delà, la saisie est refusée avec son compte et la note reste intacte",
        len(long58) == 58 and f0(rn)["nom"] == long58 and rn["message"] is None
        and "60 caractères au plus (65 saisis)" in (rn2["message"] or "") and rn2["etat"] == type_, (f0(rn)["nom"], rn2["message"]))
    # ---- le nom de chaque bande : à gauche ou en bas de la dalle, jamais à droite (la légende) ni en haut — quel que soit
    # l'angle du châssis (tourné de 90° ou de 180°, le sens des bandes s'inverse)
    def noms_bandes(rep):
        svg = rep["vues"][0]["svg"]
        g = re.search(r'<g class="sf-bandes">(.*?)</g>', svg, re.S).group(1)
        return [(float(x), float(y), n) for x, y, n in re.findall(r'<text x="([-\d.]+)" y="([-\d.]+)"(?:(?!fill="none")[^>])*>(B[XY]\d)</text>', g)]
    places, d12 = {}, mod(mod(type_, ["dalle", "Lx"], "12")["etat"], ["dalle", "Ly"], "12")["etat"]      # (12 × 12 : le châssis tourné y tient)
    for beta in ("0", "45", "90", "180", "-90"):
        nb = noms_bandes(mod(d12, ["grue", "beta"], beta))
        # (jamais à droite ; un châssis droit : à gauche ou en bas — à 45°, le bout gauche d'une diagonale peut être en haut)
        places[beta] = [(n, round(x), round(y)) for x, y, n in nb if x > 1199 or (beta != "45" and not (x < 1 or y > 1199))]
        places[beta + " (nombre)"] = len(nb)
    chk("le nom de chaque bande s'écrit hors de la dalle, JAMAIS à droite (la légende des couleurs) — châssis tourné de 0°, 45°, 90°, 180° "
        "et −90° : le sens des bandes s'inverse, leur nom reste à gauche ou en bas (à 45°, le bout gauche d'une diagonale peut être en haut)",
        all(places[b_ + " (nombre)"] == 4 and not places[b_] for b_ in ("0", "45", "90", "180", "-90")), places)
    # ---- le cadre du plan quand une surface d'appui sort de la dalle (le calcul est impossible, la carte n'est plus peinte)
    cadre = lambda rep: [float(x) for x in re.search(r'viewBox="([^"]+)"', rep["vues"][0]["svg"]).group(1).split()]
    r_out = mod(type_, ["grue", "position"], [1.05, 0.0])
    r_sans = mod(r_out["etat"], ["vue", "carte"], "aucune")
    chk("une surface d'appui sortie de la dalle (calcul impossible, plus de carte) : le plan GARDE la place de la légende — son cadre ne rétrécit "
        "pas, la dalle ne saute pas de côté à l'instant où l'on veut ramener la grue ; « Sans carte », lui, rend cette place au dessin",
        "Calcul impossible" in txt(r_out["resultat"]) and "<image" not in r_out["vues"][0]["svg"] and "<image" in r["vues"][0]["svg"]
        and cadre(r_out)[2] >= cadre(r)[2] and cadre(r_sans)[2] < cadre(r_out)[2] - 50, (cadre(r), cadre(r_out), cadre(r_sans)))
    rc = mod(type_, ["dalle", "forme"], "contour")
    rc2 = mod(rc["etat"], ["dalle", "sommet", 0], [-2.0, 0.0])
    rc3 = mod(rc2["etat"], ["dalle", "sommet", "+"], {"x": 5.0, "y": -1.0})
    g2 = f0(rc2)
    chk("le contour part du rectangle (4 sommets) ; tirer le sommet 1 de 2 m vers la gauche agrandit la dalle SANS déplacer la grue — le contour "
        "revient à l'origine de sa boîte (0 ; 0)…(10 ; 0), et e_x suit (+1,00 : le centre de la dalle a reculé de 1 m) ; un clic hors de la dalle "
        "ajoute un sommet sur le bord le plus proche ; un contour garde au moins 3 sommets",
        f0(rc)["dalle"]["sommets"] == [[0.0, 0.0], [8.0, 0.0], [8.0, 8.0], [0.0, 8.0]]
        and g2["dalle"]["sommets"] == [[0.0, 0.0], [10.0, 0.0], [10.0, 8.0], [2.0, 8.0]] and (g2["grue"]["ex"], g2["grue"]["ey"]) == (1.0, 0.0)
        and f0(rc3)["dalle"]["sommets"][1] == [5.0, 0.0] and len(f0(rc3)["dalle"]["sommets"]) == 5
        and "au moins 3 sommets" in mod(mod(rc["etat"], ["dalle", "sommets", 0, "-"], 1)["etat"], ["dalle", "sommets", 0, "-"], 1)["message"],
        (g2["dalle"]["sommets"], g2["grue"]["ex"], f0(rc3)["dalle"]["sommets"]))
    rb = mod(type_, ["grue", "repere"], 1)
    rb2 = mod(rb["etat"], ["grue", "cote", "perp"], "3,5")
    rb3 = mod(rb2["etat"], ["grue", "cote", "long"], "3")
    chk("un bord touché devient le REPÈRE des cotes du châssis (le bord droit) : son trait, deux cotes qui se tapent — à 3,50 m du bord, la grue "
        "est à e_x = +0,50 ; à 3,00 m du début du bord, e_y = −1,00 ; le même bord touché de nouveau rend les axes de la dalle",
        f0(rb)["grue"]["repere"] == 1 and "sf-bord-choisi" in rb["vues"][0]["svg"] and "grue/cote/perp" in rb["vues"][0]["svg"]
        and "Cotes au bord 2 de la dalle" in txt(rb["donnees"]) and (f0(rb2)["grue"]["ex"], f0(rb2)["grue"]["ey"]) == (0.5, 0.0)
        and (f0(rb3)["grue"]["ex"], f0(rb3)["grue"]["ey"]) == (0.5, -1.0) and f0(mod(rb["etat"], ["grue", "repere"], 1))["grue"]["repere"] is None,
        (f0(rb2)["grue"], f0(rb3)["grue"]))
    rp = mod(type_, ["grue", "position"], [3.2, 0.0])
    chk("une plaque qui sort de la dalle se voit OÙ ELLE EST : dessinée en rouge, « Calcul impossible — Une surface d'appui sort de la dalle », "
        "le contrôle en rouge, l'export éteint avec sa raison",
        rp["resume"]["niveau"] == "x" and 'class="sf-appui"' in rp["vues"][0]["svg"] and rp["vues"][0]["svg"].count('stroke="#dc2626"') >= 2
        and "Calcul impossible" in txt(rp["resultat"]) and "Une surface d'appui sort de la dalle" in txt(rp["resultat"])
        and 'data-telecharger="saf"' not in rp["resultat"], txt(rp["resultat"])[:300])
    rv = mod(mod(type_, ["vue", "cas"], "2.b")["etat"], ["vue", "k"], "30")
    champ = SR.champ(PR.lire(f0(rv)), M.calculer(PR.lire(f0(rv))), SR.synthese(PR.lire(f0(rv)), M.calculer(PR.lire(f0(rv)))))
    c2b = next(c for c in par_k[30.0]["cas"] if c["nom"] == "2.b")
    chk("la position et le k affichés se choisissent dans les listes (« Enveloppe » en tête) : la carte est celle du cas 2.b à k = 30, son "
        "maximum pointé ; le résultat de la note ne change pas (le calcul n'est pas refait)",
        abs(champ["maxi"] - c2b["sigma_max"]) < 1e-9 and abs(float(champ["valeurs"].max()) - c2b["sigma_max"]) < 1e-9
        and f"σ max {fr(c2b['sigma_max'] / V.KG_CM2, 2)} kg/cm²" in rv["vues"][0]["svg"] and rv["resume"] == r["resume"]
        and mod(type_, ["vue", "carte"], "aucune")["vues"][0]["svg"].count("<image") == 0
        and "w max" in mod(type_, ["vue", "carte"], "w")["vues"][0]["svg"], champ["maxi"])
    # ---- la bibliothèque des châssis (B1 à B5, côté écran)
    sel = q(action="selecteur", etat=type_, role="chassis")
    noms = re.findall(r'data-choix="([^"]+)"', sel["html"])
    sel_w = q(action="selecteur", etat=type_, role="chassis", familles=["Wolffkran"])
    chk("le sélecteur des châssis : « Saisie » puis les 21 châssis de la bibliothèque, par constructeur (les puces Liebherr, Wolffkran) ; la "
        "recherche (data-cherche) ; le filtre d'un constructeur",
        noms[0] == "Saisie" and len(noms) == 22 and 'data-cherche="1"' in sel["html"]
        and re.findall(r'class="fchip on" data-famille="([^"]+)"', sel["html"]) == ["Liebherr", "Wolffkran"]
        and len(re.findall(r'data-choix="([^"]+)"', sel_w["html"])) == 15 and "Liebherr Châssis 132 HC – 4,6 m × 4,6 m" in H.unescape(sel["html"]), noms[:4])
    r132 = q(action="choisir", etat=mod(type_, ["dalle", "Lx"], "9")["etat"], role="chassis", nom="Liebherr Châssis 132 HC – 4,6 m × 4,6 m")
    d132 = txt(r132["donnees"])
    chk("B1 choisir le 132 HC : appuis « Réels » d'office — l'écran lit 0,58 × 0,40 m à 45°, aire 0,2320 m², a × b 4,60 × 4,60 m, la note de la "
        "bibliothèque sur la partie centrale (dans l'aide ⓘ du châssis) ; quatre surfaces d'appui sur le plan",
        f0(r132)["grue"]["chassis"] == "liebherr-132hc-chassis-4600" and f0(r132)["grue"]["appuis"] == "reels" and "0,58 × 0,40 m à 45° · 0,2320 m²" in d132
        and "a × b 4,60 × 4,60 m" in d132 and r132["vues"][0]["svg"].count('class="sf-appui"') == 4
        and BIB.chassis("liebherr-132hc-chassis-4600")["fondation"]["note"][:40] in H.unescape(r132["donnees"]), d132[d132.find("Châssis"):][:400])
    r132r = mod(r132["etat"], ["grue", "appuis"], "rectangle")
    chk("B5 132 HC, appuis « Rectangle » d'un clic : 0,50 × 0,50 m à 45° par défaut (l'orientation de la platine), marqués « saisie » ; « Réels » "
        "rend 0,58 × 0,40 m à 45° ; choisir un autre châssis de la bibliothèque rend ses appuis réels",
        "0,50 × 0,50 m à 45° · 0,2500 m² · saisie" in txt(r132r["donnees"]) and 'class="sf-saisi"' in r132r["donnees"]
        and f0(q(action="choisir", etat=r132r["etat"], role="chassis", nom="Liebherr Châssis 200 HC – 6,0 m × 6,0 m"))["grue"]["appuis"] == "reels"
        and "0,58 × 0,40 m à 45°" in txt(mod(r132r["etat"], ["grue", "appuis"], "reels")["donnees"]), txt(r132r["donnees"])[300:700])
    r290 = q(action="choisir", etat=type_, role="chassis", nom="Liebherr Châssis 290 HC – 6,0 m × 6,0 m – pieds à vis")
    r290b = mod(r290["etat"], ["options", "rond"], "48")
    pts48 = re.findall(r'<polygon class="sf-appui" points="([^"]+)"', r290b["vues"][0]["svg"])
    chk("B2 290 HC sur pieds à vis : « Ø 0,50 m → carré 0,443 m » par défaut ; l'option « 48 côtés » dessine des polygones de 48 sommets, aire 0,1958 m²",
        "Ø 0,50 m → carré 0,443 m" in txt(r290["donnees"]) and "Ø 0,50 m en 48 côtés · 0,1958 m²" in txt(r290b["donnees"])
        and len(pts48) == 4 and all(len(p.split()) == 48 for p in pts48), txt(r290b["donnees"])[300:600])
    avant = hashlib.sha256(open(BIB.FICHIER, "rb").read()).hexdigest()
    rkrv = mod(q(action="choisir", etat=type_, role="chassis", nom="Wolffkran Kreuzrahmen KRV 10-60 – appuis à 6,0 m")["etat"], ["grue", "appuis"], "reels")
    chk("B3 Wolff KRV 10-60 : appuis « Rectangle » d'office (« Réels » est verrouillé et sans effet), la pastille « platine non indiquée », "
        "0,50 × 0,50 m marqués « saisie » ; la bibliothèque n'a pas changé",
        "platine non indiquée" in txt(rkrv["donnees"]) and 'class="sf-pastille"' in rkrv["donnees"] and "0,50 × 0,50 m · 0,2500 m² · saisie" in txt(rkrv["donnees"])
        and re.search(r'<div class="seg full locked">.*?data-chemin="grue/appuis"', rkrv["donnees"], re.S) is not None
        and hashlib.sha256(open(BIB.FICHIER, "rb").read()).hexdigest() == avant, txt(rkrv["donnees"])[300:700])
    r200 = q(action="choisir", etat=type_, role="chassis", nom="Liebherr Châssis 200 HC – 6,0 m × 6,0 m")
    r200r = mod(r200["etat"], ["grue", "appuis"], "rectangle")
    r200r = mod(r200r["etat"], ["grue", "rect", "angle"], "0")
    p1 = [tuple(float(v) for v in p.split(",")) for p in re.findall(r'<polygon class="sf-appui" points="([^"]+)"', r200r["vues"][0]["svg"])[0].split()]
    chk("B4 un châssis de la bibliothèque (200 HC, 6 × 6 m), β = 0, en rectangle 0,50 × 0,50 à 0° : P1 est en haut à gauche du plan ; mêmes cas, "
        "mêmes contraintes et même tableau par k que le même châssis saisi à la main",
        max(x for x, _ in p1) < 400 and max(y for _, y in p1) < 400
        and re.findall(r'<tr class="[a-z ]+"><td>.*?</tr>', r200r["resultat"], re.S) == re.findall(r'<tr class="[a-z ]+"><td>.*?</tr>', r["resultat"], re.S)
        and r200r["resume"]["taux"] == r["resume"]["taux"], p1)
    # ---- Proposer
    p5 = q(action="selecteur", etat=type_, role="proposer:0.500")
    li = PP.pour_epaisseur(PR.lire(f0(r)), 0.5)
    chk("« Proposer », une épaisseur par requête (une LECTURE : la barre de progression avance, la note ne change pas) : pour h = 0,50, la ligne "
        "du moteur — la plus petite dalle sous la limite pour TOUS les k cochés",
        p5["ok"] and "etat" not in p5 and f"{fr(li['Lx'], 2)} × {fr(li['Ly'], 2)} m" in txt(p5["html"]) and fr(li["sigma_max"] / V.KG_CM2, 2) in txt(p5["html"])
        and fr(li["volume"]) in txt(p5["html"]) and "disabled" in p5["html"], txt(p5["html"]))
    rg = mod(type_, ["proposition", "garder"], 1)
    chk("« Proposer » n'est gardé dans la note que COMPLET : avec une seule épaisseur faite, la demande est refusée ; le bouton porte la suite "
        "des épaisseurs (0,30 à 1,20 m par 0,10 m : dix) et ce qui la clôt",
        "pas complète" in (rg["message"] or "") and f0(rg)["proposition"] is None
        and json.loads(H.unescape(re.search(r'data-suite="([^"]+)"', r["resultat"]).group(1))) ==
        {"role": "proposer", "etapes": [f"{h:.3f}" for h in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2)], "fin": "proposition/garder",
         "sortie": "#sf-prop-t"}, rg["message"])
    perimee = copy.deepcopy(type_)
    perimee["fondations"][0]["proposition"] = {"signature": M._signature_proposition(PR.lire(type_["fondations"][0])),
                                               "lignes": [{"h": 0.5, "d": 1.0, "sigma_max": 48.0, "Lx": 10.0, "Ly": 10.0, "sommets": None, "volume": 50.0}]}
    chk("une proposition gardée dans la note vaut pour SES données : changer la vue ou le nom la garde ; changer une réaction la retire (elle "
        "proposerait une dalle calculée pour d'autres charges)",
        f0(mod(perimee, ["vue", "carte"], "w"))["proposition"] is not None and f0(mod(perimee, ["nom"], "F9"))["proposition"] is not None
        and f0(mod(perimee, ["reactions", 0, "P", 0], "360"))["proposition"] is None and f0(mod(perimee, ["limite"], "0,6"))["proposition"] is None,
        f0(mod(perimee, ["reactions", 0, "P", 0], "360"))["proposition"])
    # ---- le lien avec la raideur du sol, le carnet
    sol = SOL._etat_deux_sondages()
    page = SOL.page_resultats(sol)
    ecran_sol = txt(SOL.resultat(SOL.normaliser(sol))["resultat"])
    rr = mod(type_, ["sol", "reprise"], {"nom": "Sol essai", "chantier": "Chantier A", "etat": sol})
    chk("« Reprendre » une note de raideur du sol (§ 9) : la page de résultats que le module « Rigidité du sol » rend porte ses valeurs — chacune "
        "se lit dans son écran — ; les k de la fondation deviennent ceux de la note (cochés SCIA), la note est nommée dans la rubrique, et sa page "
        "est gardée dans le projet pour l'annexe",
        page["k"] and all(li[2].replace(" ", " ") in ecran_sol for b in page["blocs"] for li in b["lignes"])
        and [x["k"] for x in f0(rr)["sol"]["k"]] == sorted(set(page["k"])) and all(x["scia"] for x in f0(rr)["sol"]["k"])
        and "Repris de Sol essai" in txt(rr["donnees"]) and f0(rr)["sol"]["reprise"]["page"]["blocs"] == page["blocs"], (page["k"], f0(rr)["sol"]["k"]))
    rk = mod(mod(mod(type_, ["vue", "k"], "30")["etat"], ["vue", "carte"], "w")["etat"], ["sol", "reprise"],
             {"nom": "Sol essai", "chantier": "Chantier A", "etat": sol})
    svg_k = rk["vues"][0]["svg"]
    chk("« Reprendre » pendant que la vue montre k = 30, que la note de sol n'a pas : la vue revient à l'enveloppe et la carte du tassement "
        "se peint encore (une vue sur un k disparu ne peignait plus AUCUNE carte — le harnais navigateur l'a trouvé)",
        30.0 not in [x["k"] for x in f0(rk)["sol"]["k"]] and f0(rk)["vue"]["k"] is None and "<image" in svg_k and "w max" in svg_k,
        (f0(rk)["vue"], [x["k"] for x in f0(rk)["sol"]["k"]]))
    abaque = copy.deepcopy(SOL.nouvel_etat())
    abaque["fondations"][0]["mode"] = "abaque"
    ra = mod(type_, ["sol", "reprise"], {"nom": "Abaque", "etat": abaque})
    chk("une note de sol qui ne donne pas de k (l'abaque des sols) est refusée avec sa raison ; les k de la fondation ne bougent pas",
        "ne donne pas de valeur de k" in (ra["message"] or "") and ra["etat"] == type_, ra["message"])
    c1 = mod(mod(type_, ["projet", "grue"], "125EC-B6")["etat"], ["carnet", "garder"], 1, carnet=[])
    c2 = mod(M.nouvel_etat(), ["carnet", "prendre"], 0, carnet=c1["carnet"])
    c3 = mod(M.nouvel_etat(), ["carnet", 0, "-"], 1, carnet=c1["carnet"])
    chk("les grues enregistrées (le carnet) servent d'une note à l'autre : « Enregistrer cette grue » la rend dans la réponse — son nom (le champ "
        "Grue), son châssis, ses appuis, ses plots, ses réactions —, jamais dans la note ; une AUTRE note la reprend d'un clic ; on la retire ; un "
        "carnet abîmé est écarté sans erreur",
        c1["message"] == "Grue « 125EC-B6 » enregistrée" and [x["nom"] for x in c1["carnet"]] == ["125EC-B6"] and "carnet" not in c1["etat"]
        and c1["carnet"][0]["reactions"][2]["P"] == [135.0, 100.0, 800.0, 135.0] and f0(c2)["reactions"] == f0(r)["reactions"]
        and f0(c2)["projet"]["grue"] == "125EC-B6" and c2["resume"]["taux"] == r["resume"]["taux"] and c3["carnet"] == []
        and "125EC-B6" in txt(c2["donnees"]) and REG.carnets() == ["saf"]
        and q(action="calculer", etat=type_, carnet=[{"nom": "x", "grue": {"chassis": "inconnu"}}, 3, None])["carnet"] == [], c1["message"])
    # ---- l'export
    exp = q(action="telecharger", etat=type_, quoi="saf", infos=INFOS)
    octets_saf = base64.b64decode(exp["fichier"]["base64"]) if exp["ok"] else b""
    date = M._date(INFOS)
    ep = M.projet_de_la_note(PR.lire(f0(r)), INFOS)
    feuille = zipfile.ZipFile(io.BytesIO(octets_saf)).read("xl/sharedStrings.xml") if octets_saf else b""
    chk("S0 « Exporter le SAF » donne UN SEUL fichier — « <Projet>.xlsx » (le projet de la barre du haut : « Chantier A »), écrit avec le PREMIER "
        "k coché « SCIA » (dalle type : 2 MN/m³), à l'OCTET celui du moteur ; sa feuille LLE_DONNEE porte le tableau de tous les k ; les "
        "boutons « Exporter le SAF » et « Imprimer la note » ; « Joindre les notes SCIA » à côté",
        exp["ok"] and exp["fichier"]["nom"] == "Chantier A.xlsx" and exp["fichier"]["type"] == M.TYPE_XLSX
        and octets_saf == EC.fichier(ep, 2.0, date, calc["par_k"], "Grue G1") and b"CAS DIMENSIONNANT" in feuille and b"2 MN/m\xc2\xb3" in feuille
        and ep["projet"]["projet"] == "Chantier A" and ep["projet"]["indice"] == "B"
        and "Exporter le SAF" in txt(r["resultat"]) and "Imprimer la note" in txt(r["resultat"]) and 'data-joindre="note_pdf"' in r["resultat"]
        and 'data-telecharger="saf" data-apres="etapes/saf"' in r["resultat"] and 'data-telecharger="note_pdf" data-apres="etapes/note"' in r["resultat"],
        (exp.get("erreur"), exp.get("fichier", {}).get("nom")))
    vide = q(action="telecharger", etat=M.nouvel_etat(), quoi="export", infos=INFOS)
    sans_k = q(action="telecharger", etat=mod(mod(mod(type_, ["sol", "k", 0, "scia"], False)["etat"], ["sol", "k", 1, "scia"], False)["etat"], ["sol", "k", 2, "scia"], False)["etat"],
               quoi="saf", infos=INFOS)
    pres = mod(type_, ["grue", "dmin"], "1")
    chk("les contrôles avant export : sans jeu de réactions, ou sans k coché, l'export est REFUSÉ avec sa raison ; une plaque sous la distance "
        "mini n'empêche pas l'export — un avertissement, à l'écran, sous le bouton",
        not vide["ok"] and "aucun jeu de réactions complet" in vide["erreur"] and not sans_k["ok"] and "aucun k coché" in sans_k["erreur"]
        and 'data-telecharger="saf" data-apres="etapes/saf"' in pres["resultat"] and "sous la distance mini de 1,00 m" in txt(pres["resultat"])
        and "Distance au bord À vérifier 0,75 m < 1,00 m" in txt(pres["resultat"]), (vide.get("erreur"), sans_k.get("erreur")))
    lot5()


# ------------------------------------------------------------------ le lot 5 (15/11/2026) : l'équilibre, Proposer, les étapes, SCIA
def _etapes(rep):
    """La ligne d'étapes de la réponse : [(étape, faite)]."""
    return [(t_, c_ == "fait") for c_, t_ in re.findall(r'<li class="([a-z]*)"><span class="n">\d</span><span class="t">([^<]+)</span>', rep["resultat"])]


def lot5():
    """Le lot 5, côté écran (cahier des charges version 4, § 1, § 5.2, § 8, § 9) : chaque contrôle rougit sans ce qu'il
    garde (les mutations le prouvent)."""
    print("== lot 5 : l'équilibre, Proposer, les étapes, le collage, les efforts SCIA")
    type_ = M._etat_type()
    # T18 : la carte ÉQUILIBRE (§ 5.2) — la résultante du sol = les charges, au cas affiché
    r1 = mod(type_, ["vue", "cas"], "1.a")
    pl2 = mod(mod(mod(type_, ["grue", "plots", "type"], "prefabrique")["etat"], ["grue", "plots", "a"], "2")["etat"], ["grue", "plots", "b"], "2")
    pf = mod(pl2["etat"], ["vue", "cas"], "1.a")
    pc = mod(pf["etat"], ["grue", "plots", "type"], "coule")
    chk("T18 la carte ÉQUILIBRE, dalle type, cas 1.a (jeu 1) : « Résultante du sol 2 380,0 kN · Charges : grue 1 580,0 + dalle 800,0 + plots "
        "0,0 » sans plot ; 2 520,0 kN avec plots préfabriqués ou coulés 2,00 × 2,00 × 0,35 m (1 580,0 + 800,0 + 140,0) ; écart 0,00 % ; l'écart "
        "le plus grand de tous les cas et de tous les k",
        "Équilibre Conforme Cas 1.a · k = 10 0,00 % Résultante du sol 2 380,0 kN · Charges : grue 1 580,0 + dalle 800,0 + plots 0,0 = 2 380,0 kN" in txt(r1["resultat"])
        and all("Résultante du sol 2 520,0 kN · Charges : grue 1 580,0 + dalle 800,0 + plots 140,0 = 2 520,0 kN · Écart 0,00 %" in txt(x["resultat"])
                for x in (pf, pc))
        and "Écart le plus grand OK sur 80 cas et 5 k : 0,00 % < 0,1 %" in txt(r1["resultat"]), txt(pf["resultat"])[:400])
    chk("T18 sans cas choisi (l'enveloppe) : la carte ÉQUILIBRE dit le cas de σ max (3.a, k = 10 : 1 170,0 + 800,0 = 1 970,0 kN)",
        "Cas 3.a · k = 10 0,00 % Résultante du sol 1 970,0 kN · Charges : grue 1 170,0 + dalle 800,0" in txt(q(action="calculer", etat=type_)["resultat"]), "")
    # U9 (§ 9) : Proposer — un clic sur une ligne APPLIQUE Lx, Ly et h et recalcule tout ; « Annuler » revient en arrière
    e = copy.deepcopy(type_)
    for h in PR.epaisseurs(PR.lire(e["fondations"][0])):
        assert q(action="selecteur", etat=e, role=f"proposer:{h:.3f}")["ok"]
    rg = mod(e, ["proposition", "garder"], 1)
    lignes = f0(rg)["proposition"]["lignes"]
    best = PP.meilleure(lignes)
    ra = mod(rg["etat"], ["proposition", "appliquer"], best)
    rb = mod(ra["etat"], ["proposition", "annuler"], 1)
    da, db = f0(ra)["dalle"], f0(rb)["dalle"]
    chk(f"U9 « Proposer » : un clic sur la ligne au plus petit volume (h = {fr(lignes[best]['h'], 2)} m, {fr(lignes[best]['Lx'], 2)} × "
        f"{fr(lignes[best]['Ly'], 2)} m) l'APPLIQUE — Lx, Ly et h changent, tout est recalculé (la contrainte tient) ; un bouton « Annuler » "
        "apparaît ; « Annuler » rend 8,00 × 8,00 × 0,50 m et le résultat d'avant, et le bouton disparaît ; sans proposition appliquée, "
        "« Annuler » est refusé",
        (da["h"], da["Lx"], da["Ly"]) == (lignes[best]["h"], lignes[best]["Lx"], lignes[best]["Ly"])
        and SR.synthese(PR.lire(f0(ra)), M.calculer(PR.lire(f0(ra))))["niveau"] in ("p", "f")
        and 'data-chemin="proposition/annuler"' in ra["resultat"] and "Annuler" in txt(ra["resultat"])
        and (db["h"], db["Lx"], db["Ly"]) == (0.5, 8.0, 8.0) and rb["resume"] == q(action="calculer", etat=type_)["resume"]
        and 'data-chemin="proposition/annuler"' not in rb["resultat"]
        and "rien à annuler" in (q(action="modifier", etat=rb["etat"], chemin=["proposition", "annuler"], valeur=1)["message"] or "").lower(),
        (lignes[best], da, db))
    chk("A4 / U9 « Proposer » ne regarde que le SOL : à h = 0,30 m le moteur Dalles dit que la hauteur de béton ne suffit pas — le résumé de la "
        "note le dit (« Hauteur de béton insuffisante : épaissir la dalle »)",
        lignes[best]["h"] == 0.3 and ra["resume"]["niveau"] == "r" and "Hauteur de béton insuffisante" in ra["resume"]["gouvernant"]
        and "Hauteur de béton insuffisante : épaissir la dalle" in txt(ra["resultat"]), ra["resume"])
    # U10 (§ 1) : la ligne d'étapes se coche au bon moment
    n0 = _etapes(q(action="nouveau"))
    e0 = _etapes(q(action="calculer", etat=type_))
    rs = mod(type_, ["etapes", "saf"], "1")
    rv = mod(rs["etat"], ["vue", "carte"], "w")
    rsc = mod(mod(mod(rv["etat"], ["scia", "bx_m"], "75")["etat"], ["scia", "bx_v"], "90")["etat"], ["scia", "mx"], "-130")
    rsc_part = mod(rv["etat"], ["scia", "bx_m"], "75")
    rn = mod(rsc["etat"], ["etapes", "note"], "1")
    rr = mod(rn["etat"], ["reactions", 0, "P", 0], "360")
    chk("U10 la ligne d'étapes (1 Données · 2 Vérification · 3 SAF · 4 Efforts SCIA · 5 Note) : une note neuve n'a rien de coché ; la dalle "
        "type coche 1 et 2 ; « Exporter le SAF » coche 3 (changer la vue la garde) ; le tableau SCIA à moitié rempli ne coche pas 4, rempli il "
        "le coche (et le SAF reste coché : il ne dépend pas des valeurs SCIA) ; « Imprimer la note » coche 5 ; changer une réaction décoche 3 et "
        "5 (le SAF et la note sont à refaire)",
        [c for _, c in n0] == [False] * 5 and [t_ for t_, _ in e0] == ["Données", "Vérification", "SAF", "Efforts SCIA", "Note"]
        and [c for _, c in e0] == [True, True, False, False, False] and [c for _, c in _etapes(rs)] == [True, True, True, False, False]
        and [c for _, c in _etapes(rv)] == [True, True, True, False, False] and [c for _, c in _etapes(rsc_part)] == [True, True, True, False, False]
        and [c for _, c in _etapes(rsc)] == [True, True, True, True, False] and [c for _, c in _etapes(rn)] == [True] * 5
        and [c for _, c in _etapes(rr)] == [True, True, False, True, False], (_etapes(rs), _etapes(rsc), _etapes(rn), _etapes(rr)))
    # le collage d'un tableau Excel dans les réactions (§ 4, bloc 2)
    rp = q(action="modifier", etat=M.nouvel_etat(), chemin=["reactions", "coller", "0", "0"], valeur="350\t440\t440\t350\r\n150\t150\t550\t550\n135\t100\t800\t135\n")
    rp2 = q(action="modifier", etat=type_, chemin=["reactions", "coller", "1", "2"], valeur="600;610")
    chk("coller un tableau copié d'Excel dans un champ de réaction : une ligne par jeu, les valeurs séparées par des tabulations (ou « ; ») ; "
        "les jeux manquants sont ajoutés, « 3 jeux collés » ; collé dans P3 du jeu 2, il remplit P3 et P4 de ce jeu ; chaque champ porte son "
        "chemin de collage",
        rp["ok"] and rp["message"] == "3 jeux collés" and [x["P"] for x in f0(rp)["reactions"]] == [[350.0, 440.0, 440.0, 350.0], [150.0, 150.0, 550.0, 550.0], [135.0, 100.0, 800.0, 135.0]]
        and f0(rp2)["reactions"][1]["P"] == [150.0, 150.0, 600.0, 610.0] and rp2["message"] == "1 jeu collé"
        and 'data-coller="reactions/coller/2/3"' in q(action="calculer", etat=type_)["donnees"], (rp.get("message"), rp2.get("message")))
    # E1 à E4 (§ 8) : le tableau des efforts SCIA
    r = q(action="calculer", etat=type_)
    el = PR.lire(f0(r))
    calc = M.calculer(el)
    ef = M.efforts(el, calc)
    mo = SCM.module(ef, calc["par_k"])
    champs = dict(re.findall(r'<input data-chemin="(scia/[a-z_/0-9]+)"[^>]*placeholder="([^"]*)"', r["donnees"]))
    bx = [b for b in ef["bandes"] if b["nom"].startswith("BX")]
    chk("E1 le tableau SCIA (bloc 3), dalle type : la case « x = y » cochée d'office (dalle et châssis carrés, centrés) et TROIS champs — M+ le "
        "plus fort des bandes, V à d le plus fort, m le plus négatif de la carte — avec en GRIS la valeur du module (T14 : 71,6 kN·m ; 94,0 kN ; "
        "−140,8 kN·m/m) ; σ max de chaque k coché, en gris aussi ; décochée : six champs",
        'data-chemin="scia/sym" data-sorte="case" checked' in r["donnees"]
        and list(champs) == ["scia/bx_m", "scia/bx_v", "scia/mx", "scia/sigma/2", "scia/sigma/5", "scia/sigma/10"]
        and (champs["scia/bx_m"], champs["scia/bx_v"], champs["scia/mx"]) == (fr(mo["bx_m"]), fr(mo["bx_v"]), fr(mo["mx"]))
        and fr(mo["bx_m"]) == fr(max(b["M_plus"] for b in bx)) and fr(mo["mx"]) == fr(ef["extremes"]["sup_x"])
        and champs["scia/sigma/10"] == fr(next(x for x in calc["par_k"] if x["k"] == 10.0)["sigma_max"] / V.KG_CM2, 2)
        and re.findall(r'data-chemin="(scia/[a-z_]+)"', mod(type_, ["scia", "sym"], False)["donnees"])
        == ["scia/sym", "scia/bx_m", "scia/by_m", "scia/bx_v", "scia/by_v", "scia/mx", "scia/my"], champs)
    r2 = mod(type_, ["scia", "bx_m"], "80")
    r3 = mod(r2["etat"], ["scia", "bx_m"], "")
    chk("E2 un champ vide reste VIDE : effacé, la valeur SCIA redevient « non saisie » (None, jamais la valeur grise) ; le principe garde les "
        "moments du module (r = 1) ; la carte dit « valeurs SCIA à saisir »",
        f0(r2)["scia"]["bx_m"] == 80.0 and f0(r3)["scia"]["bx_m"] is None
        and M.armatures(PR.lire(f0(r3)), calc)["echelle"] == {"inf_x": 1.0, "inf_y": 1.0, "sup_x": 1.0, "sup_y": 1.0}
        and "Écart module / SCIA à saisir valeurs SCIA à saisir" in txt(r3["resultat"])
        and re.search(r'data-chemin="scia/bx_m"[^>]*value=""', r3["donnees"]) is not None, f0(r3)["scia"])
    d107 = mod(mod(type_, ["dalle", "Lx"], "10,7")["etat"], ["dalle", "Ly"], "10,7")["etat"]
    el7 = PR.lire(f0(q(action="calculer", etat=d107)))
    calc7 = M.calculer(el7)
    m7 = SCM.module(M.efforts(el7, calc7), calc7["par_k"])
    r7 = mod(d107, ["scia", "bx_m"], repr(round(m7["bx_m"] * 1.10, 6)))
    fer7 = M.armatures(PR.lire(f0(r7)), calc7)
    r8 = mod(d107, ["scia", "bx_m"], repr(round(m7["bx_m"] * 1.30, 6)))
    chk("E3 / E4 dalle 10,70 × 10,70 sans plot : taper en bas 1,10 × la valeur grise donne r = 1,10 — « Écart module / SCIA concordent M+ r = "
        "1,10 », l'échelle des moments du bas en x et en y (x = y) ; 1,30 × : une pastille ORANGE avec l'écart en % (« 30,0 % »)",
        abs(fer7["echelle"]["inf_x"] - 1.10) < 1e-6 and abs(fer7["echelle"]["inf_y"] - 1.10) < 1e-4 and fer7["echelle"]["sup_x"] == 1.0
        and "Écart module / SCIA concordent M+ r = 1,10" in txt(r7["resultat"])
        and re.search(r'<b>Écart module / SCIA</b>.*?<span class="et-mot v">30,0 %</span>', r8["resultat"], re.S) is not None
        and "M+ r = 1,30" in txt(r8["resultat"]), (fer7["echelle"], txt(r8["resultat"])[txt(r8["resultat"]).find("Écart"):][:120]))
    # U11 : le recalcul de la dalle type en 2 à 3 s au plus (§ 3), le changement d'un treillis sans relancer le sol (A2)
    import time
    t0 = time.perf_counter()
    rq = mod(type_, ["dalle", "h"], "0,47")
    duree = time.perf_counter() - t0
    t0 = time.perf_counter()
    rt = mod(rq["etat"], ["armatures", "inf", "x", "d"], "16")
    duree_t = time.perf_counter() - t0
    fer_q, fer_t = M.armatures(PR.lire(f0(rq)), M.calculer(PR.lire(f0(rq)))), M.armatures(PR.lire(f0(rt)), M.calculer(PR.lire(f0(rt))))
    chk(f"U11 / A2 le recalcul de la dalle type (h changé : sol, efforts, principe) prend {duree:.2f} s (≤ 3 s) ; changer un treillis "
        f"{duree_t:.2f} s (≤ 1 s : le sol n'est pas recalculé) et le principe le suit — « bas Ø16 e 15 », le m_Rd du bas en x du moteur "
        f"Dalles ({fr(fer_q['res']['mRd']['inf_x'])} → {fr(fer_t['res']['mRd']['inf_x'])} kN·m/m) écrit dans la carte",
        duree <= 3.0 and duree_t <= 1.0 and "Treillis choisi bas Ø16 e 15 · haut Ø12 e 15" in txt(rt["resultat"])
        and fer_t["res"]["mRd"]["inf_x"] > 1.5 * fer_q["res"]["mRd"]["inf_x"]
        and f"m Rd = {fr(fer_t['res']['mRd']['inf_x'])} kN·m/m" in txt(rt["resultat"]), (duree, duree_t))
def complets():
    print("== 6. la note PDF (en entier : tests/test_saf_grue_note.py) et l'archive")
    etat = M._avec(projet={"projet": "Essai 2026", "grue": "132 HC", "indice": "A", "creation": "14/11/2026"}, dalle={"Lx": 9.0, "Ly": 9.0},
                   grue={"chassis": "liebherr-132hc-chassis-4600", "plots": {"type": "coule", "a": 2.0, "b": 2.0, "h": 0.35}})
    etat = mod(etat, ["sol", "reprise"], {"nom": "Sol essai", "chantier": "Chantier A", "etat": SOL._etat_deux_sondages()})["etat"]
    r = q(action="calculer", etat=etat)
    el = PR.lire(f0(r))
    exp = q(action="telecharger", etat=etat, quoi="export", infos=INFOS)
    z = zipfile.ZipFile(io.BytesIO(base64.b64decode(exp["fichier"]["base64"]))) if exp["ok"] else None
    noms = z.namelist() if z else []
    k = el["sol"]["k"][0]["k"]
    chk("l'archive (une note de plusieurs fondations, l'ancien export) : le SAF — un seul, écrit avec le premier k coché, ici le k repris de "
        "la note de sol — et la note de calcul PDF",
        exp["ok"] and noms == ["Chantier A.xlsx", "Note de calcul - Grue G1.pdf"] and z.read(noms[1])[:5] == b"%PDF-"
        and EC.k_du_fichier(el) == k, (exp.get("erreur"), noms))
    note = q(action="telecharger", etat=etat, quoi="note_pdf", infos=INFOS)
    chk("« Imprimer la note » rend la note de calcul PDF, « Note de calcul - <note>.pdf »",
        note["ok"] and base64.b64decode(note["fichier"]["base64"])[:5] == b"%PDF-" and note["fichier"]["nom"] == "Note de calcul - Grue G1.pdf",
        note.get("erreur"))
    deux = copy.deepcopy(etat)
    M.gerer_elements(deux, "ajouter")
    deux = q(action="calculer", etat=deux)["etat"]
    rd = q(action="telecharger", etat=deux, quoi="export", infos=INFOS)
    chk("une note de DEUX fondations dont l'une n'a pas de réaction : l'export est refusé et nomme la fondation qui l'empêche ; la note PDF, elle, "
        "sort (la fondation sans réaction y est dite « Données incomplètes », poids propre seul)",
        not rd["ok"] and "« F2 » ne s'exporte pas" in rd["erreur"]
        and "Poids propre seul" in pdf_texte(base64.b64decode(q(action="telecharger", etat=deux, quoi="note_pdf", infos=INFOS)["fichier"]["base64"])),
        rd.get("erreur"))
    print("== 5 bis. « Proposer » en entier, puis une ligne appliquée")
    e = M._etat_type()
    for h in PR.epaisseurs(PR.lire(e["fondations"][0])):
        rp = q(action="selecteur", etat=e, role=f"proposer:{h:.3f}")
        assert rp["ok"], rp
    rg = mod(e, ["proposition", "garder"], 1)
    lignes = f0(rg)["proposition"]["lignes"]
    best = PP.meilleure(lignes)
    ra = mod(rg["etat"], ["proposition", "appliquer"], best)
    chk("« Proposer » complet (dix épaisseurs) se garde dans la note : le tableau h | dalle | σ max | béton, la ligne au plus petit volume mise en "
        "évidence ; un clic l'APPLIQUE — l'épaisseur et les dimensions de la dalle sont celles de la ligne, et la contrainte tient (≤ 49,0 kPa) pour "
        "tous les k cochés ; le tableau reste (d'autres lignes s'essaient), la ligne appliquée y est marquée ; « Proposer » ne regarde que le SOL : "
        "à h = 0,30 m la hauteur de béton ne suffit pas (moteur Dalles), et le résumé de la note (sol ET armatures) le dit",
        rg["message"] is None and len(lignes) == 10 and rg["resultat"].count('class="best') == 1
        and (f0(ra)["dalle"]["h"], f0(ra)["dalle"]["Lx"], f0(ra)["dalle"]["Ly"]) == (lignes[best]["h"], lignes[best]["Lx"], lignes[best]["Ly"])
        and SR.synthese(PR.lire(f0(ra)), M.calculer(PR.lire(f0(ra))))["niveau"] in ("p", "f") and f0(ra)["proposition"] is not None
        and 'class="best cur"' in ra["resultat"] and ra["resume"]["niveau"] == "r" and "Hauteur de béton insuffisante" in ra["resume"]["gouvernant"],
        (lignes[best], ra["resume"]))
    rx = mod(ra["etat"], ["reactions", 0, "P", 0], "360")
    chk("une proposition calculée pour d'AUTRES données n'en est plus une : changer une réaction la retire de la note",
        f0(rx)["proposition"] is None and "sf-tp" not in rx["resultat"], f0(rx)["proposition"])
    print("== 1 bis. les fondations de la note, la relecture, le parcours du module")
    e = M._etat_type()
    M.gerer_elements(e, "dupliquer")
    M.gerer_elements(e, "ajouter")
    r3 = q(action="calculer", etat=e)
    chk("une note de trois fondations (copier, ajouter) : F1, F2 (la copie, ouverte après elle), F3 neuve et ouverte ; la note est « Données "
        "incomplètes » tant que F3 n'a pas de réaction, puis son résumé nomme la plus chargée ; supprimer rend deux fondations, jamais zéro",
        [f["nom"] for f in r3["etat"]["fondations"]] == ["F1", "F2", "F3"] and r3["etat"]["actif"] == 2 and r3["resume"]["niveau"] == "n"
        and r3["resume"]["gouvernant"].startswith("F3 · poids propre seul") and q(action="supprimer", etat=r3["etat"], index=2)["resume"]["gouvernant"].startswith("F1 · σ")
        and len(q(action="supprimer", etat=r3["etat"], index=2)["etat"]["fondations"]) == 2
        and len(q(action="supprimer", etat=M.nouvel_etat(), index=0)["etat"]["fondations"]) == 1, r3["resume"])
    abime = copy.deepcopy(M._etat_type())
    abime["fondations"][0]["dalle"]["h"] = "épais"
    ra = q(action="calculer", etat=abime)
    chk("une note abîmée (un fichier .stab retouché) est refusée avec sa raison, jamais relue à moitié ; une note relue est la même (idempotent)",
        not ra["ok"] and "épaisseur" in ra["erreur"] and M.normaliser(M.normaliser(M._etat_type())) == M.normaliser(M._etat_type()), ra.get("erreur"))
    from api_web import parcours as P
    journal = P.parcourir("saf", routeur.traiter)
    rates = [(d, e_) for d, ok, e_ in journal if not ok]
    chk(f"le parcours du module (api_web/parcours.py : chaque choix du cadre Données, puis ses requêtes propres — {len(journal)} requêtes) : aucune "
        "n'échoue", not rates and len(journal) > 60, rates[:4])


MUTATIONS = [
    ("saf_resultat.py", '''    retenus = [r for r in par_k if r["k"] in scia] or par_k''', '''    retenus = par_k''', "le maximum pris sur tous les k, cochés ou non"),
    ("saf_resultat.py", '''    taux = gov["sigma_max"] / limite''', '''    taux = gov["sigma_max"] / (limite * 2)''', "le taux face à une limite doublée"),
    ("saf_resultat.py", '''        niv = C.niveau(r["sigma_max"] / s["limite"])
        vu = s["r_vue"] is r''', '''        niv = "p"
        vu = s["r_vue"] is r''', "le tableau par k toujours vert"),
    ("saf_resultat.py", '''            sig = s_.copy() if sig is None else np.maximum(sig, s_)''', '''            sig = s_.copy() if sig is None else sig''',
     "la carte d'un seul cas au lieu de l'enveloppe"),
    ("saf_resultat.py", '''        ks = [r for r in par_k if r["k"] in scia] or list(par_k)''', '''        ks = list(par_k)''',
     "l'enveloppe qui prend aussi les k non cochés « SCIA »"),
    ("dessin_saf.py", '''        out.append(_iso(v, champ, ident))''', '''        pass''', "la ligne noire σ = limite oubliée sur le plan"),
    ("saf.py", '''                p_["a"], p_["b"], p_["h"] = PR.PLOTS_DEFAUT[valeur]''', '''                pass''',
     "les plots préfabriqués gardent les dimensions des plots coulés (2,00 m au lieu de 2,50 m)"),
    ("dessin_saf.py", '''        maxi = max(maxi, float(champ["limite"]))''', '''        pass''', "l'échelle qui ne monte pas jusqu'à la limite"),
    ("saf_saisie.py", '''    return f"{sol_nb(el, kpa)}{C.NBSP}{unite(el)[0]}"''', '''    return f"{C.fr(kpa, 1)}{C.NBSP}{unite(el)[0]}"''',
     "une contrainte écrite en kPa sous le nom d'une autre unité"),
    ("saf.py", '''        el["limite"] = _nombre(valeur, "la limite") * SS.unite(el)[1] / V.KG_CM2''', '''        el["limite"] = _nombre(valeur, "la limite")''',
     "la limite saisie en kPa gardée comme des kg/cm²"),
    ("saf.py", '''        g["ex"], g["ey"] = round(gx - (x0 + x1) / 2, 3), round(gy - (y0 + y1) / 2, 3)''', '''        pass''', "la grue déplacée par un sommet tiré"),
    ("saf.py", '''    if relu.get("proposition") and relu["proposition"].get("signature") != _signature_proposition(relu):
        relu["proposition"] = None''', '''    pass''', "une proposition périmée gardée"),
    ("saf.py", '''    if not m["cas"]:
        bloque.append("aucun jeu de réactions complet")''', '''    pass''', "l'export permis sans réaction"),
    ("saf.py", '''        g["chassis"], g["appuis"] = ident, "reels"''', '''        g["chassis"] = ident''', "un châssis de la bibliothèque choisi sans ses appuis réels"),
    ("saf.py", '''            g["repere"] = None if i == g.get("repere") else i''', '''            g["repere"] = i''', "le repère au bord qui ne se relâche plus"),
    ("saf.py", '''    el["sol"]["k"] = [{"k": k, "scia": True} for k in ks]''', '''    el["sol"]["k"] = [{"k": k, "scia": False} for k in ks]''', "les k repris non cochés SCIA"),
    ("saf.py", '''    return [(EC.nom_fichier(ep, nom_note), EC.fichier(ep, k, date, calc["par_k"], nom_note))]''',
     '''    return [(EC.nom_fichier(ep, nom_note) + str(x), EC.fichier(ep, x, date, calc["par_k"], nom_note)) for x in PR.k_scia(el)]''',
     "un fichier SAF par k coché (la règle d'avant le lot 4)"),
    ("saf_saisie.py", '''    reels_possible = bool(ch["id"]) and not ch["impose"]''', '''    reels_possible = True''', "« Réels » offert sans platine dans la bibliothèque"),
    ("dessin_saf.py", '''        bon = ctl["plaques_dedans"][i] and ctl["distances"][i] >= g["dmin"] - G.TOLERANCE
        coul, fond = (APPUI, APPUI_FOND) if bon else (ROUGE, ROUGE_FOND)''', '''        coul, fond = (APPUI, APPUI_FOND)''',
     "une plaque hors de la dalle dessinée en orange"),
    ("dessin_saf.py", '''        ax.update((round(x0 + g["dmin"] + (mx - px0) - cx, 4), round(x1 - g["dmin"] - (px1 - mx) - cx, 4)))''', '''        pass''',
     "l'accroche « plaque à la distance mini » perdue"),
    ("sol.py", '''                lignes.append([x["nom"], f"w = {_plage(x['w_bas'], x['w_haut'], 1)} mm", f"{_plage_courte(x['k_bas'], x['k_haut'])} MN/m³"])''',
     '''                lignes.append([x["nom"], f"w = {_plage(x['w_bas'], x['w_haut'], 1)} mm", f"{_plage_courte(x['k_bas'] * 2, x['k_haut'] * 2)} MN/m³"])''',
     "la page reprise du sol qui ne dit plus les k de son écran"),
    ("saf.py", '''        if len(nom) > PR.NOM_MAX:''', '''        if len(nom) > 10 ** 6:''', "un nom trop long coupé en silence au lieu d'être refusé"),
    ("saf_resultat.py", '''        rows += C.ligne_verif(nom, texte, None, niv, calcul="s1", etat_txt=mot)''',
     '''        rows += C.ligne_verif(nom, texte, None, niv, etat_txt=mot)''', "une ligne de contrôle qui n'ouvre plus son détail"),
    ("saf.py", '''legende=el["vue"]["carte"] != "aucune")''', '''legende=False)''', "la place de la légende perdue quand le calcul est impossible"),
    ("dessin_saf.py", '''            (x1, y1), (x2, y2) = (x2, y2), (x1, y1)
        ux, uy''', '''            pass
        ux, uy''', "le nom d'une bande écrit sous la légende, châssis tourné"),
    ("saf.py", '''    synth_note = [synthese_note(f, calc if i == etat["actif"] else calculer(f), x) for i, (f, x) in enumerate(zip(etat["fondations"], synth))]''',
     "    synth_note = synth", "le résumé de la note ne dit que le sol (pas la hauteur de béton)"),
]
# (lot 5, 15/11/2026)
MUTATIONS += [
    ("saf.py", '''            if valeur not in PR.CARTES:
                raise ValueError("carte : contrainte, tassement, armatures ou sans carte")''', '''            pass''', "une carte inconnue (« Moments » du lot 4) acceptée"),
    ("saf.py", '''            if valeur not in PR.FACES_VUE:
                raise ValueError("face : bas ou haut")''', '''            pass''', "une face inconnue acceptée"),
    ("saf.py", '''    hors = {"vue", "etapes", "proposition"} | ({"scia"} if etape == "saf" else set())''',
     '''    hors = {"etapes", "proposition"} | ({"scia"} if etape == "saf" else set())''', "changer la vue décoche « SAF »"),
    ("saf.py", '''    hors = {"vue", "etapes", "proposition"} | ({"scia"} if etape == "saf" else set())''',
     '''    hors = {"vue", "etapes", "proposition"}''', "taper une valeur SCIA décoche « SAF »"),
    ("saf_resultat.py", '''("SAF", d1 and faites_saf)''', '''("SAF", d1)''', "l'étape « SAF » cochée sans export"),
    ("saf_resultat.py", '''("Efforts SCIA", bool(sc and sc["complet"]))''', '''("Efforts SCIA", bool(sc and sc["saisi"]))''',
     "l'étape « Efforts SCIA » cochée avec un tableau à moitié rempli"),
    ("saf_resultat.py", '''    plots_ = sum(gamma * pl["hauteur"] * G.aire(q) for q in pl["polys"]) if pl else 0.0''', '''    plots_ = 0.0''',
     "l'équilibre qui oublie le poids des plots"),
    ("saf.py", '''        d["h"] = li["h"]
        if li.get("sommets"):''', '''        if li.get("sommets"):''', "« Proposer » appliqué sans l'épaisseur"),
    ("saf.py", '''        d["forme"], d["Lx"], d["Ly"], d["h"] = av["forme"], av["Lx"], av["Ly"], av["h"]''', '''        d["forme"] = av["forme"]''',
     "« Annuler » qui ne rend pas les dimensions"),
    ("saf.py", '''        cases = [c.strip() for c in (l_.split("\\t") if "\\t" in l_ else l_.split(";"))]''', '''        cases = [c.strip() for c in l_.split(";")]''',
     "le collage d'Excel qui ne lit pas les tabulations"),
    ("saf_saisie.py", '''f'<input data-coller="reactions/coller/{i}/{p}" ', 1)''', '''f'<input ', 1)''', "les champs de réaction sans collage"),
    ("saf.py", '''        sc[ch[0]] = None if vide else _nombre(valeur, "la valeur SCIA")''', '''        sc[ch[0]] = 0.0 if vide else _nombre(valeur, "la valeur SCIA")''',
     "un champ SCIA effacé gardé à 0 (au lieu de vide)"),
    ("saf_armatures.py", '''None, "v" if alerte else "p", etat_txt''', '''None, "p", etat_txt''', "l'écart module / SCIA hors de 15 % sans pastille orange"),
    ("saf_armatures.py", '''auto=gris), aides[cle]''', '''auto=""), aides[cle]''', "le tableau SCIA sans les valeurs grises du module"),
    ("saf.py", '''        html_res, volets = SR.colonne(el, calc, s, SA.carte(el, fer, ef, sc), fer["non_verifies"], etapes_=etp)''',
     '''        html_res, volets = SR.colonne(el, calc, s, SA.carte(el, fer, ef, sc), etapes_=etp)''', "« Non vérifiés » sans le poinçonnement"),
    ("dessin_armatures.py", '''    for r in fer["renforts"]:
        if r["face"] != face:
            continue
        cs = _coins(r)
        bande''', '''    for r in fer["renforts"]:
        cs = _coins(r)
        bande''', "la carte Armatures du bas qui dessine aussi les renforts du haut"),
    ("saf.py", '''    cle = json.dumps([_signature_calcul(el), PR.k_scia(el), el["bandes"]["largeur"], el["armatures"], el["scia"], el["dalle"]["beton"]],''',
     '''    cle = json.dumps([_signature_calcul(el), PR.k_scia(el), el["bandes"]["largeur"], el["scia"], el["dalle"]["beton"]],''',
     "un treillis changé qui ne refait pas le principe (cache)"),
]
MUTATIONS = [m for m in MUTATIONS if m[2] is not None]
TEMOIN = ("saf.py", 'MODULE = "saf"', 'MODULE = "saf"')


def muter(fichier, ancien, nouveau):
    """Les contrôles rapides sur un arbre miroir : api_web copié (et muté), le reste du dépôt lié."""
    with tempfile.TemporaryDirectory() as tmp:
        for nom in os.listdir(RACINE):
            if nom in (".git", ".temoin_tests", "__pycache__", "node_modules"):
                continue
            if nom == "api_web":
                shutil.copytree(os.path.join(RACINE, nom), os.path.join(tmp, nom), ignore=shutil.ignore_patterns("__pycache__"))
            else:
                os.symlink(os.path.join(RACINE, nom), os.path.join(tmp, nom))
        cible = os.path.join(tmp, "api_web", fichier)
        src = open(cible, encoding="utf-8").read()
        if src.count(ancien) != 1:
            return False, None
        open(cible, "w", encoding="utf-8").write(src.replace(ancien, nouveau))
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--rapide", "--racine", tmp], capture_output=True, text=True, cwd=RACINE)
        return True, p.returncode


def main():
    print("== 1 à 5. l'état, la saisie, l'écran, la bibliothèque, Proposer, le sol, le carnet, l'export (contrôles rapides)")
    rapides()
    if RAPIDE:
        print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
        return 1 if KO else 0
    complets()
    print("== 7. mutations")
    trouve, code = muter(*TEMOIN)
    chk("témoin : l'écran recopié sans rien changer passe les contrôles rapides", trouve and code == 0, code)
    # (les mutations se jouent EN PARALLÈLE — chacune sur son arbre miroir ; MUTATIONS_PARALLELES, 3 par défaut)
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(os.environ.get("MUTATIONS_PARALLELES", "3")))) as pool:
        joues = list(pool.map(lambda m_: muter(*m_[:3]), MUTATIONS))
    for (fichier, ancien, nouveau, nom), (trouve, code) in zip(MUTATIONS, joues):
        chk(f"mutation « {nom} » ({fichier}) : un contrôle rougit", trouve and code == 1,
            "texte absent ou répété" if not trouve else f"code {code} : AUCUN contrôle ne rougit")
    print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
    for nom, info in KO:
        print("   -", nom[:170], "|", str(info)[:400])
    return 1 if KO else 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""Générateur SAF — fondation de grue : l'ÉCRAN du site (api_web/saf.py, saf_saisie.py, saf_resultat.py, dessin_saf.py),
la NOTE PDF (note_saf.py), l'EXPORT en un clic, le lien avec le module de raideur du sol (api_web/sol.page_resultats) et
les grues enregistrées (le carnet). Cahier des charges § 3, § 7, § 8, § 9 et les essais B1 à B5 côté écran.

Ce que la suite prouve :
  1. L'ÉTAT — une note neuve aux valeurs du bureau, les réactions vides ; la relecture ; les fondations de la note.
  2. LA SAISIE — chaque chemin écrit ce qu'il dit ; un refus rend la note intacte, avec sa raison ; le contour se
     dessine sans déplacer la grue ; le bord pris pour repère des cotes.
  3. L'ÉCRAN DIT LE MOTEUR — la contrainte, le tableau par k, les positions, les contrôles sont les nombres du moteur ;
     le plan porte ce que les gestes règlent ; ce qui ne va pas se voit où il est.
  4. B1 à B5 — avec la vraie bibliothèque des châssis, à l'écran et dans la note.
  5. « PROPOSER », le lien avec la raideur du sol, les grues enregistrées.
  6. L'EXPORT et la NOTE — l'archive, ses fichiers SAF (ceux du moteur, à l'octet), la note qui dit les chiffres de
     l'écran, l'annexe.
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

from api_web import commun as C  # noqa: E402
from api_web import registre as REG  # noqa: E402
from api_web import routeur  # noqa: E402
from api_web import saf as M  # noqa: E402
from api_web import saf_resultat as SR  # noqa: E402
from api_web import saf_saisie as SS  # noqa: E402
from api_web import sol as SOL  # noqa: E402
from grues.chassis import acces as BIB  # noqa: E402
from saf.fondation_grue import ecriture as EC  # noqa: E402
from saf.fondation_grue import projet as PR  # noqa: E402
from saf.fondation_grue import proposition as PP  # noqa: E402
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
    # (lot 4 : maille 0,20 m, bandes de 0,50 m, contraintes de sol en kg/cm², enrobage 5 cm, treillis automatiques — le harnais suit)
    chk("une note neuve : une fondation « F1 » aux valeurs du bureau — 8,00 × 8,00 × 0,50 m, C30/37, k = 2 ; 5 ; 10 ; 20 ; 30 (cochés SCIA : 2 ; 5 ; 10), "
        "châssis saisi 6 × 6 m, plaques 0,50 × 0,50, sans plot, bandes 0,50 m, maille 0,20 m, limite 0,5 kg/cm², unité kg/cm², enrobage 5 cm, "
        "treillis automatiques ; les réactions et les textes du projet VIDES ; le carnet n'est pas dans la note",
        r0["ok"] and el["nom"] == "F1" and (el["dalle"]["Lx"], el["dalle"]["Ly"], el["dalle"]["h"], el["dalle"]["beton"]) == (8.0, 8.0, 0.5, "C30/37")
        and [(x["k"], x["scia"]) for x in el["sol"]["k"]] == [(2.0, True), (5.0, True), (10.0, True), (20.0, False), (30.0, False)]
        and el["grue"]["chassis"] == "saisie" and (el["grue"]["appuis"], el["grue"]["rect"]) == ("rectangle", {"a": 0.5, "b": 0.5, "angle": 0})
        and el["grue"]["plots"]["type"] == "aucun" and el["limite"] == 0.5 and el["bandes"]["largeur"] == 0.5 and el["options"]["maille"] == 0.2
        and el["options"]["unite"] == "kgcm2" and el["armatures"] == {"enrobage": 0.05, "treillis_inf": "auto", "treillis_sup": "auto"}
        and el["reactions"] == [{"nom": "", "P": [None] * 4}] and set(el["projet"].values()) == {""} and "carnet" not in r0["etat"], el)
    chk("sans réaction : le poids propre seul (σ = 25·h = 12,5 kPa = 0,13 kg/cm², l'unité par défaut), la note est « Données incomplètes », le "
        "bloc « Grue et charges » dit « réactions à saisir », et l'export est éteint, avec sa raison écrite dessous — rien en fenêtre surgissante",
        r0["resume"]["niveau"] == "n" and "poids propre seul" in r0["resume"]["gouvernant"] and "0,13\u00a0kg/cm²" in r0["resume"]["gouvernant"]
        and "à saisir" in txt(r0["donnees"]) and re.search(r'<button type="button" class="btn primary sf-exp-b" disabled[ >]', r0["resultat"]) is not None
        and "data-telecharger" not in r0["resultat"]
        and "Aucun jeu de réactions complet" in txt(r0["resultat"]), (r0["resume"], txt(r0["resultat"])[-300:]))
    # U1 (lot 4, § 2.1) : QUATRE blocs comme le module « Poutres béton armé », et TOUS les anciens champs y sont (le projet et l'indice
    # passent dans la barre du haut : ils ne sont plus des champs de la colonne de gauche)
    chemins = set(re.findall(r'data-chemin="([^"]+)"', r0["donnees"]))
    anciens = {"projet/grue", "dalle/forme", "dalle/Lx", "dalle/Ly", "dalle/h", "dalle/beton", "sol/k/0/k", "sol/k/0/scia", "sol/k/+",
               "grue/a", "grue/b", "grue/appuis", "grue/rect/a", "grue/rect/b", "grue/rect/angle", "grue/plots/type", "grue/ex", "grue/ey",
               "grue/beta", "grue/dmin", "reactions/0/nom", "reactions/0/P/0", "reactions/+", "bandes/largeur", "limite", "options/maille",
               "options/anneau", "options/c1x", "options/c1y", "options/c2x", "options/c2y", "options/rond", "options/aretes", "options/nonlinear",
               "options/h_min", "options/h_max", "options/h_pas", "carnet/garder"}
    nouveaux = {"armatures/enrobage", "armatures/treillis_inf", "armatures/treillis_sup", "options/unite", "options/torsion",
                "options/recouvrement", "options/l_max", "options/gamma_s", "options/jeu", "options/arrondi"}
    titres = re.findall(r'<span class="tt">([^<]+)</span>', r0["donnees"])
    chk("U1 la colonne de gauche en QUATRE blocs numérotés, dépliables, avec leur résumé : 1 Dalle et sol, 2 Grue et charges, 3 Armatures, "
        "4 Options (rangé à part) ; tous les anciens champs sont présents, et ceux du lot 4 (enrobage, treillis, unité, torsion, recouvrement, "
        "longueur des barres, options du moteur poutre)",
        re.findall(r'data-sec="([a-z]+)"', r0["donnees"]) == ["dalle", "grue", "armatures", "options"]
        and re.findall(r'data-sec="([a-z]+)" data-secondaire', r0["donnees"]) == ["options"]
        and titres == ["Dalle et sol", "Grue et charges", "Armatures", "Options"] and anciens <= chemins and nouveaux <= chemins
        and "projet/projet" not in chemins and "projet/indice" not in chemins,
        (titres, sorted(anciens - chemins), sorted(nouveaux - chemins)))
    type_ = M._etat_type()
    r = q(action="calculer", etat=type_)
    el = f0(r)
    calc = M.calculer(PR.lire(el))
    par_k = {x["k"]: x for x in calc["par_k"]}
    gov = max((par_k[k] for k in (2.0, 5.0, 10.0)), key=lambda x: x["sigma_max"])
    res = txt(r["resultat"])
    # U2 (lot 4, § 2.2) : les cartes du module poutre — plus de grand rectangle (« chip xl »), plus de verdict en très gros
    chk("U2 la dalle type : l'écran dit le MOTEUR dans la carte SOL — « σ max = 1,12 kg/cm² > 0,50 kg/cm² · cas 3.a · k = 10 », le taux "
        f"{C.pct(gov['sigma_max'] / (0.5 * V.KG_CM2))} % ; la carte est « Non conforme » (rouge) ; aucun grand rectangle de contrainte ; le pied "
        "« Méthode du bureau · moteur poutre » et la version du moteur",
        f"σ max = {fr(gov['sigma_max'] / V.KG_CM2, 2)} kg/cm² > 0,50 kg/cm² · cas {gov['cas_max']} · k = 10" in res and gov["cas_max"] == "3.a"
        and fr(gov["sigma_max"] / V.KG_CM2, 2) == "1,12" and r["resume"]["niveau"] == "r"
        and r["resume"]["taux"] == C.pct(gov["sigma_max"] / (0.5 * V.KG_CM2))
        and 'chip xl' not in r["resultat"] and 'class="verdict' not in r["resultat"] and '<section class="et-bloc nok" data-theme="0">' in r["resultat"]
        and "Sol Non conforme" in res and "Méthode du bureau · moteur poutre v2.40" in res, (res[:300], r["resume"]))
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
    chk("σ moyen (un repère, dans la carte SOL) : (poids de la dalle + réactions du jeu le plus lourd) ÷ aire = (800 + 1 580) / 64 = 37,2 kPa = "
        "0,38 kg/cm² ; le tassement max et la surface décollée ; les contrôles : appuis dans la dalle, distance au bord 0,750 m ≥ 0,50 m, "
        "équilibre ; « 16 cas générés », la liste repliée",
        abs(sm["sigma"] - 37.1875) < 1e-9 and "σ moyen = (800 + 1 580) / 64,0 m² = 0,38 kg/cm²" in res and "Distance au bord OK 0,750 m ≥ 0,50 m" in res
        and "Tassement max info w max = " in res and "Surface décollée info" in res
        and "Appuis dans la dalle OK" in res and "Équilibre OK écart 0,0000 %" in res and "16 cas générés" in res
        and '<details class="sf-pos" data-plan-volet="positions">' in r["resultat"] and r["resultat"].count('class="sf-cas') == 17, res[300:700])
    chk("les contrôles sont des LIGNES DE L'ÉTABLI (C.ligne_verif : « el vc », l'état en un mot) — celles des autres modules — et chacune ouvre "
        "le détail des contrôles (fenêtre s1)",
        r["resultat"].count('<div class="el vc" data-calcul="s1" title="Voir le détail">') == 3 and 'class="sf-ctl' not in r["resultat"]
        and r["resultat"].count('<span class="et-mot p">OK</span>') == 3, re.findall(r'<div class="el[^>]*>', r["resultat"]))
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
    chk("U8 / U3 la barre du centre, sur UNE ligne : l'onglet « Plan », puis trois listes — la carte (Contrainte, Tassement, Sans carte), le "
        "cas et le k, « Enveloppe » en tête et choisie ; la carte, une image PNG découpée au contour, est l'ENVELOPPE des 16 cas et des 3 k "
        "cochés SCIA : son maximum est celui du moteur, pointé en kg/cm²",
        png[:8] == b"\x89PNG\r\n\x1a\n" and f"σ max {fr(gov['sigma_max'] / V.KG_CM2, 2)} kg/cm²" in svg and 'clip-path="url(#saf_dalle)"' in svg
        and ent.startswith('<div class="et-ent sf-ent sf-barre">') and "<br" not in ent and 'class="l1"' not in ent
        and re.findall(r'data-chemin="vue/onglet" data-valeur="([a-z]+)"', ent) == ["plan"]
        and [c for c, _ in menus] == ["vue/carte", "vue/cas", "vue/k"] and premiers["vue/cas"] == ("", " selected", "Enveloppe")
        and premiers["vue/k"] == ("", " selected", "Enveloppe")
        and re.findall(r'<option value="([a-z]+)"', dict(menus)["vue/carte"]) == ["sigma", "w", "aucune"], (len(png), premiers))
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
        and "data-telecharger" not in rp["resultat"], txt(rp["resultat"])[:300])
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
    chk("B1 choisir le 132 HC : appuis « Réels » d'office — l'écran lit 0,58 × 0,40 m à 45°, aire 0,2320 m², les entraxes 4,60 × 4,60 m, la note de la "
        "bibliothèque sur la partie centrale ; quatre surfaces d'appui sur le plan",
        f0(r132)["grue"]["chassis"] == "liebherr-132hc-chassis-4600" and f0(r132)["grue"]["appuis"] == "reels" and "0,58 × 0,40 m à 45° · 0,2320 m²" in d132
        and "Entraxes 4,60 × 4,60 m" in d132 and r132["vues"][0]["svg"].count('class="sf-appui"') == 4
        and BIB.chassis("liebherr-132hc-chassis-4600")["fondation"]["note"][:40] in d132, d132[d132.find("Châssis"):][:400])
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
    z = zipfile.ZipFile(io.BytesIO(base64.b64decode(exp["fichier"]["base64"])))
    date = M._date(INFOS)
    ep = M.projet_de_la_note(PR.lire(f0(r)), INFOS)
    attendus = {"Chantier A.xlsx": EC.fichier(ep, 2.0, date, calc["par_k"], "Grue G1")}
    feuille = zipfile.ZipFile(io.BytesIO(z.read("Chantier A.xlsx"))).read("xl/sharedStrings.xml") if "Chantier A.xlsx" in z.namelist() else b""
    chk("S0 l'export : UNE archive « SAF - <note>.zip » — UN SEUL fichier SAF (lot 4, § 7), « <Projet>.xlsx » (le projet de la barre du haut : "
        "« Chantier A »), écrit avec le PREMIER k coché « SCIA » (dalle type : 2 MN/m³), à l'OCTET celui du moteur ; sa feuille LLE_DONNEE porte "
        "le tableau de tous les k et le cas dimensionnant ; le projet et l'indice y sont ceux de la barre du haut ; le bouton dit « Exporter · SAF + note »",
        exp["ok"] and exp["fichier"]["nom"] == "SAF - Grue G1.zip" and exp["fichier"]["type"] == "application/zip"
        and z.namelist() == ["Chantier A.xlsx"] and all(z.read(n) == o for n, o in attendus.items()) and b"CAS DIMENSIONNANT" in feuille
        and b"2 MN/m\xc2\xb3" in feuille and ep["projet"]["projet"] == "Chantier A" and ep["projet"]["indice"] == "B"
        and "Exporter · SAF + note" in txt(r["resultat"]), (exp.get("erreur"), z.namelist()))
    vide = q(action="telecharger", etat=M.nouvel_etat(), quoi="export", infos=INFOS)
    sans_k = q(action="telecharger", etat=mod(mod(mod(type_, ["sol", "k", 0, "scia"], False)["etat"], ["sol", "k", 1, "scia"], False)["etat"], ["sol", "k", 2, "scia"], False)["etat"],
               quoi="saf", infos=INFOS)
    pres = mod(type_, ["grue", "dmin"], "1")
    chk("les contrôles avant export : sans jeu de réactions, ou sans k coché, l'export est REFUSÉ avec sa raison ; une plaque sous la distance "
        "mini n'empêche pas l'export — un avertissement, à l'écran, sous le bouton",
        not vide["ok"] and "aucun jeu de réactions complet" in vide["erreur"] and not sans_k["ok"] and "aucun k coché" in sans_k["erreur"]
        and 'data-telecharger="export" data-avec-note="1"' in pres["resultat"] and "sous la distance mini de 1,00 m" in txt(pres["resultat"])
        and "Distance au bord Attention 0,750 m < 1,00 m" in txt(pres["resultat"]), (vide.get("erreur"), sans_k.get("erreur")))


def complets():
    print("== 6. la note PDF et l'export complet")
    etat = M._avec(projet={"projet": "Essai 2026", "grue": "132 HC", "indice": "A", "creation": "14/11/2026"}, dalle={"Lx": 9.0, "Ly": 9.0},
                   grue={"chassis": "liebherr-132hc-chassis-4600", "plots": {"type": "coule", "a": 2.0, "b": 2.0, "h": 0.35}})
    etat = mod(etat, ["sol", "reprise"], {"nom": "Sol essai", "chantier": "Chantier A", "etat": SOL._etat_deux_sondages()})["etat"]
    r = q(action="calculer", etat=etat)
    el = PR.lire(f0(r))
    calc = M.calculer(el)
    s = SR.synthese(el, calc)
    exp = q(action="telecharger", etat=etat, quoi="export", infos=INFOS)
    z = zipfile.ZipFile(io.BytesIO(base64.b64decode(exp["fichier"]["base64"]))) if exp["ok"] else None
    noms = z.namelist() if z else []
    k = el["sol"]["k"][0]["k"]
    chk("S0 l'export en un clic : le SAF (un seul, écrit avec le premier k coché — ici le k repris de la note de sol) ET la note de calcul PDF, "
        "dans une archive",
        exp["ok"] and noms == ["Chantier A.xlsx", "Note de calcul - Grue G1.pdf"] and z.read(noms[1])[:5] == b"%PDF-"
        and EC.k_du_fichier(el) == k, (exp.get("erreur"), noms))
    note = q(action="telecharger", etat=etat, quoi="note_pdf", infos=INFOS)
    octets = base64.b64decode(note["fichier"]["base64"])
    t = pdf_texte(octets)
    n_pages = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", "-"], input=octets, capture_output=True).stdout.decode()).group(1))
    pire = next(x for x in calc["par_k"] if x["k"] == s["k"])
    chk("la note PDF, à la mise en page des notes du site : la garde (le chantier, le nom de la note, l'indice, la date ; un sommaire de trois "
        "planches) puis « Plan et données », « Vérification rapide », « Annexe — raideur du sol » : quatre pages",
        note["ok"] and n_pages == 4 and all(x in t for x in ("Note de calcul", "Chantier A", "Grue G1", "14/11/2026", "F1 — Plan et données",
                                                              "F1 — Vérification rapide", "F1 — Annexe — raideur du sol")), (n_pages, t[:300]))
    chk("la note dit LES DONNÉES : le projet et la grue, le châssis par son nom ET son id dans la bibliothèque, les appuis réels 0,58 × 0,40 m à 45° "
        "et leur aire, les plots coulés en place 2,00 × 2,00 × 0,35 m, les trois jeux, le béton et E, k, la limite — un champ vide ne s'imprime pas, "
        "aucun libellé n'est coupé (« … »)",
        all(x in t for x in ("Essai 2026", "132 HC", "Liebherr", "liebherr-132hc-chassis-4600", "réels (la bibliothèque)", "rectangle 0,58 × 0,40 m à 45°", "0,2320 m²",
                             "coulés en place", "2,00 × 2,00 × 0,35 m", "32 800", "Contrainte de contact admise", "0,50 kg/cm²", "350", "800",
                             "Le fichier SAF est écrit avec k = ", "Les k à passer dans SCIA"))
        and "…" not in t, t[300:900])
    # (lot 4, § 2.4 : la note parle l'unité de l'écran — kg/cm² par défaut —, jamais deux unités mélangées ; la carte est l'enveloppe)
    chk(f"la note dit LES CHIFFRES DE L'ÉCRAN, dans l'unité choisie : σ max = {fr(s['sigma'] / V.KG_CM2, 2)} kg/cm² > 0,50 kg/cm² au cas {s['cas']}, "
        "le tableau par k (σ max, tassement, décollé), les seize cas générés (cas, LC, jeu, P1 à P4), les contrôles ; la carte en enveloppe ; "
        "aucun kPa",
        f"σ max = {fr(s['sigma'] / V.KG_CM2, 2)} kg/cm² > 0,50 kg/cm² · cas {s['cas']}, k = {C.nb(s['k'])} MN/m³" in t
        and "kPa" not in t.split("REPRISE")[0]
        and "ENVELOPPE DES CAS" in t
        and f"{fr(pire['w_max'], 2)}" in t and t.count("LC17") >= 1 and "CAS DE CHARGE GÉNÉRÉS : 16" in t and "3.h" in t
        and "Les quatre surfaces d'appui sont dans la dalle." in t and "Équilibre" in t, t[900:1600])
    b = calc["modele"]["bandes"][0]
    chk("la note porte les BANDES d'intégration et leurs coordonnées (début, fin, largeur), et l'ANNEXE : la page de résultats du module "
        "« Rigidité du sol » — la note reprise, sa fondation, sa méthode, ses lignes « À encoder dans SCIA » et « Zonage de la dalle »",
        all(x in t for x in ("BX1", "BY2", f"{fr(b['debut'][0], 3)} ; {fr(b['debut'][1], 3)}", "Sol essai", "Sondage CPT", "À encoder dans SCIA",
                             "Zonage de la dalle", el["sol"]["reprise"]["page"]["blocs"][1]["lignes"][0][2].replace(" ", " "))), t[-700:])
    krv = M._avec(grue={"chassis": "wolff-krv-10-60-a60"})
    tk = pdf_texte(base64.b64decode(q(action="telecharger", etat=krv, quoi="note_pdf", infos=INFOS)["fichier"]["base64"]))
    chk("B3 dans la note : Wolff KRV 10-60 — appuis « rectangle (saisie) », « Platine non indiquée dans la bibliothèque : les dimensions de la "
        "surface d'appui sont saisies », 0,50 × 0,50 m ; aucun libellé coupé (« … ») ; sans note de sol reprise, pas d'annexe",
        "Appuis rectangle (saisie)" in tk and "Platine non indiquée dans la bibliothèque : les dimensions de la surface d'appui sont saisies." in tk
        and "0,50 × 0,50 m" in tk and "Annexe" not in tk and "…" not in tk and "…" not in t, tk[300:900])
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
        "tous les k cochés ; le tableau reste (d'autres lignes s'essaient), la ligne appliquée y est marquée",
        rg["message"] is None and len(lignes) == 10 and rg["resultat"].count('class="best') == 1
        and (f0(ra)["dalle"]["h"], f0(ra)["dalle"]["Lx"], f0(ra)["dalle"]["Ly"]) == (lignes[best]["h"], lignes[best]["Lx"], lignes[best]["Ly"])
        and ra["resume"]["niveau"] in ("p", "f") and f0(ra)["proposition"] is not None and 'class="best cur"' in ra["resultat"],
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
    ("saf.py", '''        out["carnet"] = mod.lire_carnet(etat.pop("carnet", []))''', None, None),
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
]
MUTATIONS += [
    ("saf.py", '''        if len(nom) > PR.NOM_MAX:''', '''        if len(nom) > 10 ** 6:''', "un nom trop long coupé en silence au lieu d'être refusé"),
    ("saf_resultat.py", '''        rows += C.ligne_verif(nom, texte, None, niv, calcul="s1", etat_txt="OK" if bon else ("Non" if grave else "Attention"))''',
     '''        rows += C.ligne_verif(nom, texte, None, niv, etat_txt="OK" if bon else ("Non" if grave else "Attention"))''',
     "une ligne de contrôle qui n'ouvre plus son détail"),
]
MUTATIONS += [
    ("saf.py", '''legende=el["vue"]["carte"] != "aucune")''', '''legende=False)''', "la place de la légende perdue quand le calcul est impossible"),
    ("dessin_saf.py", '''            (x1, y1), (x2, y2) = (x2, y2), (x1, y1)
        ux, uy''', '''            pass
        ux, uy''', "le nom d'une bande écrit sous la légende, châssis tourné"),
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
    for fichier, ancien, nouveau, nom in MUTATIONS:
        trouve, code = muter(fichier, ancien, nouveau)
        chk(f"mutation « {nom} » ({fichier}) : un contrôle rougit", trouve and code == 1,
            "texte absent ou répété" if not trouve else f"code {code} : AUCUN contrôle ne rougit")
    print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
    for nom, info in KO:
        print("   -", nom[:170], "|", str(info)[:400])
    return 1 if KO else 0


if __name__ == "__main__":
    sys.exit(main())

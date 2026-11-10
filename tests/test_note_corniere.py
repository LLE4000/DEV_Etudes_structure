# -*- coding: utf-8 -*-
"""La NOTE PDF du module « Cornières d'appui » sur le site (acier/corniere/note.py, api_web/note_corniere.py — le lot
écran y arrive par api_web/corniere.telecharger), révision § 9 du 10/11/2026 au soir, vérifiée en Python ordinaire,
SANS l'écran du module : l'état de la note du site est construit par le moteur (nouvelle cornière + modifications),
{"version": 2, "cornieres": [...]}.

- TOUTE la note en un PDF valide : la page de garde du site (projet, partie, date, indice, sommaire avec l'état limite)
  puis deux pages par cornière (trois avec une page de suite), dans l'ordre de la liste, sur 17 états : A historique,
  A équilibre, B avec Y = 85 et Y = 45, ailes inégales (150 puis 100 horizontal), petite (L 40×40×4), grosse
  (L 250×250×35, équilibre, quatre ancrages), deux cornières dans une note (A avec largeur d'appui, B retournée), la
  garde (ancrage bas), une note v1 relue, l'ELU (R = 60 saisie à l'ELU), une poutre en A (160/120) et une en B, un
  boulon traversant à deux rondelles, un entraxe saisi (80), une note ELS + ELU ;
- PARITÉ avec le moteur de l'écran (``moteur.verifier``), cornière par cornière, par pdftotext : le nom, la
  configuration, le mode et l'ÉTAT LIMITE (bandeau et sommaire), le statut et le taux max, chaque taux des
  vérifications, M1 à M4, H, N et V par ancrage (ELS et ELU à l'ELS ; à l'ELU les seules N_ELU et V_ELU, pas d'ELS
  inventé), chaque RÉSULTAT des formules écrites ; les HYPOTHÈSES COMPACTES (la limite σ et τ avec son symbole —
  « σ_adm = f_y/1,5 = 236,7 N/mm² », « f_y/γ_M0 = 355,0 N/mm² » —, dix lignes au plus, un budget de caractères) ; les
  lignes de CHARGE ET ANCRAGE (R et son état limite, L, X ± dx, a, R_ELU à l'ELS ; n × type Ø d, profondeur,
  rondelles, Y ± dy, entraxe et e₁, support voile ou poutre avec ses distances aux bords) ; la PAGE 2 : le résumé de
  méthode (trois lignes), les trois vues et leur légende, les limites (entières ou abrégées), les références, un
  budget de caractères hors cartouche ; les MAILLES COLORÉES du champ de von Mises dans le PDF (pymupdf : les couleurs
  de remplissage de la page 2 sont celles du dégradé du moteur, maille par maille ; sans pymupdf, le flux de contenu) ;
- chaque FORMULE de la note (acier/corniere/formules.py) ré-évaluée deux fois, sur tous les blocs de tous les états —
  dont les gabarits de l'ELU (sigma_lim_ELU, tau_lim_ELU, N_ELU_direct, V_ELU_direct) ;
- l'IDENTIFICATION : la note du site (chantier et nom PIÉGÉS — accents, guillemets, « & », tiret long —, indice « C »,
  date) dans la garde et dans l'en-tête de chaque page, jamais « indice 0 » ni « indice A » ; le nom du fichier ;
- la PAGE DE SUITE : un bloc de la page 1 qui ne tient pas passe en page 3, la colonne d'origine le dit, le sommaire
  compte la page ; la PAGE 2 ne part JAMAIS en page de suite (des avertissements en masse : trois lignes, pas de refus) ;
- les REFUS : une cornière hors domaine, une note sans cornière ;
- COMME DANS LE NAVIGATEUR (sonde calquée sur celle du paquet de la note, site/web/construire._SONDE_NOTE) : sans
  fontTools ni accélérateurs C, avec le Pillow factice et les seules polices du paquet, la note se fait, son texte est
  celui de CPython, et les fichiers du dépôt qu'elle charge en plus de l'écran sont ceux que ``note_corniere.FICHIERS``
  déclare ;
- MUTATIONS (chacune doit faire rougir un contrôle) : une cornière oubliée, une donnée non transmise au moteur (la
  tolérance dy), le mode mal nommé, un gabarit de formule faux, l'indice de la note perdu, l'η max du bandeau pris sur
  une autre vérification, l'état limite non imprimé, la page 2 sans sa troisième vue, les hypothèses en prose longue,
  les rondelles non imprimées.

    python3 tests/test_note_corniere.py
"""
import base64
import copy
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import types
import zipfile
import zlib

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)
sys.path.insert(0, os.path.join(RACINE, "tests"))

import note_cas_outils as O  # noqa: E402
from acier.corniere import entree as E  # noqa: E402
from acier.corniere import formules as FO  # noqa: E402
from acier.corniere import methode as MT  # noqa: E402
from acier.corniere import moteur as MO  # noqa: E402
from acier.corniere import note as NT  # noqa: E402
from acier.corniere import schemas as SC  # noqa: E402
from acier.formats import F, f0  # noqa: E402
from api_web import commun as C  # noqa: E402
from api_web import note_corniere as NC  # noqa: E402

try:
    import pymupdf  # noqa: E402
except ImportError:  # pragma: no cover — le flux de contenu prend le relais
    pymupdf = None

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:400]}"))


def corniere(**kw):
    c = MO.nouvelle_corniere()
    c.update(kw)
    return c


def etat(*cs):
    return {"version": 2, "cornieres": [copy.deepcopy(c) for c in cs], "actif": 0}


ETATS = {
    "A historique (cornière neuve)": etat(corniere()),
    "A équilibre": etat(corniere(nom="Équilibre", mode="equilibre")),
    "B, Y = 85": etat(corniere(nom="Retournée", configuration="B")),
    "B, Y = 45 (cas de contrôle retourné, sans jeu)": etat(corniere(nom="Retournée basse", configuration="B", Y=45, dy=0, dx=0, R=42.76)),
    "inégale L 150×100×12, 150 horizontal": etat(corniere(nom="Inégale 150", profil="L 150x100x12", Y=65)),
    "inégale L 150×100×12, 100 horizontal": etat(corniere(nom="Inégale 100", profil="L 150x100x12", aile_horizontale="B", Y=115, X=60, dx=15)),
    "petite L 40×40×4": etat(corniere(nom="Petite", profil="L 40x40x4", X=25, dx=5, Y=28, dy=2, d=8, R=4, L=200, n=2)),
    "grosse L 250×250×35, équilibre, quatre ancrages": etat(corniere(nom="Grosse", profil="L 250x250x35", X=150, dx=30, Y=180, dy=15, d=24, R=200, L=600,
                                                                 n=4, mode="equilibre")),
    "deux cornières (A avec a = 100, B avec u_V = 0 et c = 10)": etat(corniere(nom="Appui A", a=100),
                                                                     corniere(nom="Appui B", configuration="B", uV=0, yc=10, dx=0, dy=0)),
    "garde : ancrage bas, cas de contrôle, R_ELU saisie": etat(corniere(nom="Garde", Y=40, dy=0, dx=0, R=42.76, R_ELU=68.9)),
    "note v1 relue (modèle lame), saisie libre": etat({"nom": "Ancienne", "standard": False, "A": 100, "B": 100, "t": 10, "V": 20, "e": 50}),
    # révision § 9 : l'état limite, le support, l'ancrage
    "ELU, R = 60 saisie à l'ELU (A historique)": etat(corniere(nom="ELU 60", etat_limite="ELU", R=60)),
    "poutre A (160 au-dessus, 120 au-dessous)": etat(corniere(nom="Poutre A", support="poutre", support_haut=160, support_bas=120)),
    "poutre B (Y = 45, sans jeu dy)": etat(corniere(nom="Poutre B", configuration="B", support="poutre", support_haut=160, support_bas=150, Y=45, dy=0)),
    "boulon traversant, deux rondelles, épaisseur 200": etat(corniere(nom="Traversant", type_ancrage="traversant", rondelles=2, profondeur=200)),
    "entraxe saisi 80": etat(corniere(nom="Entraxe 80", entraxe=80)),
    "ELS + ELU dans une note": etat(corniere(nom="Mixte ELS"), corniere(nom="Mixte ELU", etat_limite="ELU", R=60, mode="equilibre")),
}
NOM = "Façade « Est » & l'atelier \"Nord\""
CHANTIER = "Résidence d'Été & Cie — « Lot 2 »"
INFOS = {"nom": NOM, "chantier": CHANTIER, "indice": "C", "date": "03/04/2026"}
INTERDITS = ("indice 0", "indice A", "INDICE 0", "INDICE A")
TETE = "Note de calcul · 03/04/2026 · indice C"
TITRES_P2 = ("LARGEUR EFFICACE ET ÉLÉVATION", "MOMENTS SUR LA CORNIÈRE", "CONTRAINTES DE VON MISES")
# les budgets de caractères (sans blancs), fixés d'après la mesure sur les 17 états (le plus grand observé + ≈ 15 %) :
# le bloc Hypothèses de la page 1 (entre son titre et « CHARGE ET ANCRAGE »), la page 2 hors cartouche. Les mutations
# « hypothèses en prose longue » (≈ 1 500 caractères) les dépassent.
SEUIL_HYPOTHESES = 760
SEUIL_PAGE2 = 3300


def norm(t):
    return O.norm(t)


def plat(t):
    return re.sub(r"\s+", "", t)


def plat_sym(t):
    """Sans blanc ni « _ », « $ », « { », « } » : un symbole indicé (R_ELU, $σ_{eq}$) sort de pdftotext en « RELU »,
    « σeq » ; la virgule française de la note (``note.francais``) appliquée au texte attendu."""
    return plat(NT.francais(t).replace("_", "").replace("$", "").replace("{", "").replace("}", ""))


def pages_brutes(octets):
    """Le texte de chaque page dans l'ordre du dessin (pdftotext -raw : une formule reste d'un tenant)."""
    return O.pages(octets, brut=True)


def debut_pages(tx, nom):
    """L'indice de la page 1 d'une cornière (son bandeau : « Cornière d'appui — <nom> » et « η max »), puis le nombre
    de pages qu'elle occupe (2, ou 3 avec une page de suite)."""
    for i, t in enumerate(tx):
        n = norm(t)
        if f"{NT.TITRE} — {nom}" in n and "η max" in n:
            k = 2
            if i + 2 < len(tx) and "SUITE DE" in norm(tx[i + 2]) and nom in norm(tx[i + 2]):
                k = 3
            return i, k
    return None, 0


def nombre(x, nd):
    return F(x, nd)


def hypotheses_plat(p1):
    """Le bloc Hypothèses de la page 1, sans blancs (entre son titre et « CHARGE ET ANCRAGE »)."""
    t = plat(p1)
    i, j = t.find("HYPOTHÈSES"), t.find("CHARGEETANCRAGE")
    return t[i:j] if 0 <= i < j else t


def page2_hors_cartouche(p2):
    """La page 2 sans blancs, sans le cartouche (dessiné en dernier : son premier libellé est « CORNIÈRE »)."""
    t = plat(p2)
    i = t.rfind("CORNIÈRE")
    return t[:i] if i > 0 else t


def _hex(rgb):
    return "#%02X%02X%02X" % tuple(int(round(x * 255)) for x in rgb)


def couleurs_attendues(res):
    """Les couleurs que la scène des contraintes donne aux mailles (le plus grand des taux des quatre nœuds — l'échelle
    recopiée des décisions dans ``schemas.couleur_taux``), les deux ailes et le coin."""
    out = set()
    for cle in ("horizontale", "verticale", "coin"):
        TX = res["maillage"][cle]["taux"]
        for i in range(len(TX) - 1):
            for j in range(len(TX[0]) - 1):
                out.add(SC.couleur_taux(max(TX[i][j], TX[i + 1][j], TX[i][j + 1], TX[i + 1][j + 1])).upper())
    return out


def nb_mailles(res):
    """Le nombre de mailles du champ de von Mises que le moteur donne (2 ailes × pas en s × niveaux ζ : 768 avec un congé,
    576 sans — une saisie libre à r = 0 — plus les 64 du coin)."""
    mz = res["maillage"]
    return sum((len(mz[a]["s"]) - 1) * (len(mz[a]["zeta"]) - 1) for a in ("horizontale", "verticale")) + (len(mz["coin"]["zeta"]) - 1) ** 2


def _colore(h):
    r, g, b = int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
    return max(r, g, b) - min(r, g, b) > 12


def fills_page(octets, i_page):
    """(nombre de remplissages colorés — hors blanc, noir et gris —, couleurs de remplissage) d'une page : pymupdf lit les
    chemins de la page, un par maille. Sans lui, la page est RENDUE (pdftoppm, PPM lu sans bibliothèque) et ce sont les
    couleurs des pixels qui sont comptées (None pour le nombre : l'anticrénelage ne compte pas des mailles)."""
    if pymupdf is not None:
        doc = pymupdf.open(stream=octets, filetype="pdf")
        out = [_hex(dr["fill"]) for dr in doc[i_page].get_drawings() if dr.get("fill") is not None]
        return sum(1 for h in out if _colore(h)), set(out)
    p = subprocess.run(["pdftoppm", "-r", "60", "-f", str(i_page + 1), "-l", str(i_page + 1), "-singlefile", "-", "-"], input=octets,
                       capture_output=True, check=True)
    tete, _, px = p.stdout.partition(b"255\n")
    couleurs = {"#%02X%02X%02X" % (px[i], px[i + 1], px[i + 2]) for i in range(0, len(px) - 2, 3)}
    return None, couleurs


def ecarts_parite(e, octets):
    """Les écarts entre la note et le moteur de l'écran, cornière par cornière (voir l'en-tête)."""
    tx = pages_brutes(octets)
    out = []
    attendu_pages = 1
    debuts = []
    for c in e["cornieres"]:
        res = MO.verifier(c)
        calc = NT.Calcul(c)
        nom = res["corniere"]["nom"]
        mat, g, anc, ef = res["materiau"], res["geometrie"], res["ancrages"], res["efforts"]
        etat_l = mat["etat_limite"]
        i, k = debut_pages(tx, nom)
        if i is None:
            out.append(f"« {nom} » : aucune page de calcul à son nom")
            continue
        debuts.append(i)
        attendu_pages += k
        p1, p2 = norm(tx[i]), norm(tx[i + 1])
        p12 = p1 + " " + p2 + (" " + norm(tx[i + 2]) if k == 3 else "")
        s1, s2, s12 = plat_sym(p1), plat_sym(p2), plat_sym(p12)
        # le bandeau : statut, η max, gouvernante, combinaison, configuration, mode et état limite — lus SUR LA LIGNE du
        # bandeau (le récapitulatif répète « η max … », et un bandeau qui imprimerait un autre taux s'y cachait —
        # relecture adverse du 10/11/2026 : la mutation « η max du bandeau pris sur S1 » survivait)
        gv = res["gouvernant"]
        bandeau = (f"{NT.STATUTS[res['statut']][0]} η max {NT._pct(res['taux_max'])} — gouverne : {gv['nom']} · {gv['combinaison']['libelle']} "
                   f"Configuration {res['configuration']} · {E.LIBELLES_MODE[res['mode']]} · {etat_l}")
        if bandeau not in p1:
            out.append(f"« {nom} » p. 1 : bandeau « {bandeau} » absent")
        # la garde : la ligne du sommaire de CETTE cornière porte sa désignation, sa configuration, son mode, son état
        # limite, son η max et son statut
        sommaire = (f"{res['section']['designation']} · configuration {res['configuration']} ({E.LIBELLES_CONFIGURATION[res['configuration']]}) · "
                    f"{E.LIBELLES_MODE[res['mode']]} · {etat_l} · η max {NT._pct(res['taux_max'])} {NT.STATUTS[res['statut']][0]}")
        if sommaire not in re.sub(r"\s+", " ", norm(tx[0])):
            out.append(f"« {nom} » garde : ligne du sommaire « {sommaire} » absente")
        # chaque vérification : son taux ; les moments M1 à M4
        for v in res["verifications"]:
            if not O.present(NT._pct(v["taux"]), p1):
                out.append(f"« {nom} » p. 1 : taux {v['cle']} {NT._pct(v['taux'])} absent")
            if v["cle"] in ("S1", "S2", "S3", "S4") and not O.present(nombre(abs(v["M"]), 3), p1):
                out.append(f"« {nom} » p. 1 : M de {v['cle']} = {nombre(abs(v['M']), 3)} absent")
        # H, N et V par ancrage : ELS et ELU à l'ELS ; à l'ELU les seules N_ELU et V_ELU, et rien qui dise « ELS »
        for lab, val in (("H", ef["H"]), ("N_ELS", anc["N_ELS"]), ("V_ELS", anc["V_ELS"]), ("N_ELU", anc["N_ELU"]), ("V_ELU", anc["V_ELU"])):
            if val is not None and not O.present(nombre(val, 2), p1):
                out.append(f"« {nom} » p. 1 : {lab} = {nombre(val, 2)} absent")
        if etat_l == E.ELU:
            for mot in ("N_ELS", "V_ELS", "R_ELU =", "σ_adm =", "contraintes admissibles"):
                if plat_sym(mot) in s12:
                    out.append(f"« {nom} » (ELU) : « {mot} » imprimé — rien n'est inventé pour l'ELS")
            if plat("ELU, PAR ANCRAGE — R SAISIE À L'ELU") not in plat(p1):
                out.append(f"« {nom} » (ELU) p. 1 : le groupe « ELU, par ancrage — R saisie à l'ELU » manque")
        else:
            if plat("ELS, PAR ANCRAGE") not in plat(p1) or plat_sym("N_ELU") not in s1:
                out.append(f"« {nom} » (ELS) p. 1 : les groupes ELS puis ELU des ancrages manquent")
        # les résultats des formules écrites : la chaîne de chaque section, la coupe gouvernante, les ancrages
        subs = []
        for cle in ("S1", "S2", "S3", "S4", "nette"):
            b = FO.bloc(res, cle)
            if b is not None:
                subs += FO.chaine(b)
        subs += FO.coupe(FO.bloc(res, res["gouvernant"]["cle"])) + FO.subs(FO.bloc(res, "ancrages"))
        for s in subs:
            r = FO.fmt(s.res, s.nd) if s.nd != -1 else NT._pct(s.res)
            if not O.present(r, p1):
                out.append(f"« {nom} » p. 1 : résultat « {FO.rendre(s)} » absent")
        # les HYPOTHÈSES compactes : dix lignes au plus, le budget de caractères, l'état limite avec sa limite et son symbole
        hyp = calc.hypotheses()
        if len(hyp) > 10:
            out.append(f"« {nom} » p. 1 : {len(hyp)} lignes d'hypothèses (dix au plus)")
        n_hyp = len(hypotheses_plat(p1))
        if n_hyp > SEUIL_HYPOTHESES:
            out.append(f"« {nom} » p. 1 : hypothèses trop longues ({n_hyp} caractères, budget {SEUIL_HYPOTHESES})")
        if etat_l == E.ELS:
            attendus = [f"σ_adm = f_y/{f0(mat['coef'])} = {F(mat['sigma_adm'], 1)} N/mm²", f"τ_adm = σ_adm/√3 = {F(mat['tau_adm'], 1)} N/mm²"]
        else:
            attendus = [f"f_y/γ_M0 = {F(mat['sigma_adm'], 1)} N/mm²", f"f_y/(√3·γ_M0) = {F(mat['tau_adm'], 1)} N/mm²", "γ_M0 = 1,0"]
        for x in attendus:
            if plat_sym(x) not in s1:
                out.append(f"« {nom} » p. 1 : hypothèse de l'état limite « {x} » absente")
        if plat(f"ÉTAT LIMITE {etat_l} ·") not in plat(p1):
            out.append(f"« {nom} » p. 1 : la ligne « État limite  {etat_l} » manque")
        # CHARGE ET ANCRAGE : chaque grandeur sur sa ligne, dans l'état limite de la saisie
        c_ = res["corniere"]
        jx = f" ± {f0(c_['dx'])}" if c_["dx"] > 0 else ""
        jy = f" ± {f0(c_['dy'])}" if c_["dy"] > 0 else ""
        lignes = [f"R = {NT._kn(ef['R'])} à l'{etat_l}", f"L = {f0(g['L'])} mm (longueur reprise", f"X = {f0(c_['X'])}{jx} mm du nu du mur",
                  f"Y = {f0(c_['Y'])}{jy} mm de la face extérieure",
                  f"{anc['n']} × {E.LIBELLES_TYPE_ANCRAGE[g['type_ancrage']]} Ø{f0(anc['d'])} (trou d_0 = {f0(anc['d0'])} mm)",
                  f"s = {f0(anc['entraxe'])} mm ({'auto : L/n' if g['entraxe_auto'] else 'saisi'}) · e_1 = {f0(anc['distance_bout'])} mm"]
        lignes.append(f"a = {f0(g['a'])} mm (largeur d'appui" if g["a"] is not None else "a = L (la charge sur toute la tranche)")
        if g["type_ancrage"] == "traversant":
            lignes.append(f"épaisseur traversée {f0(g['profondeur'])} mm")
        else:
            lignes.append(f"h_ef = {f0(g['profondeur'])} mm")
        rd = g["rondelle"]
        if g["rondelles"] == 0:
            lignes.append("RONDELLES sans rondelle")
        elif g["rondelles"] == 1:
            lignes.append(f"une rondelle ISO 7089 sous l'écrou (Ø{f0(rd['normale'][0])} × {f0(rd['normale'][1])})")
        else:
            lignes += [f"rondelle large ISO 7093-1 contre l'aile (Ø{f0(rd['large'][0])} × {f0(rd['large'][1])})",
                       f"ISO 7089 sous l'écrou (Ø{f0(rd['normale'][0])} × {f0(rd['normale'][1])})"]
        if g["support"] == E.POUTRE:
            b_ = g["bords_poutre"]
            lignes += [f"poutre en béton : {f0(g['support_haut'])} mm au-dessus et {f0(g['support_bas'])} mm au-dessous",
                       f"{f0(b_['haut'])} mm du bord supérieur, {f0(b_['bas'])} mm du bord inférieur"]
        else:
            lignes.append("SUPPORT voile en béton")
        if etat_l == E.ELS:
            lignes.append(f"R_ELU = 1,5·R = {NT._kn(ef['R_ELU'])} (auto" if ef["R_ELU_auto"] else f"R_ELU = {NT._kn(ef['R_ELU'])} (saisie)")
        for x in lignes:
            if plat_sym(x) not in s1:
                out.append(f"« {nom} » p. 1 : ligne de charge / ancrage « {x} » absente")
        # la page 2 : le titre, le résumé de méthode (trois lignes), les trois vues et leurs légendes, les limites
        # (entières ou abrégées), les références (entières), les dessins, le budget de caractères
        if f"DÉTAILS ET MÉTHODE — {nom}" not in p2:
            out.append(f"« {nom} » p. 2 : le titre manque")
        for ligne in MT.resume(res):
            if plat_sym(ligne[:50]) not in s2:
                out.append(f"« {nom} » p. 2 : ligne du résumé « {ligne[:50]} » absente")
        for t in TITRES_P2:
            if plat(t) not in plat(p2):
                out.append(f"« {nom} » p. 2 : la colonne « {t} » manque")
        for cle in ("elevation", "moments", "contraintes"):
            leg = calc.legende_vue(cle)
            if plat_sym(leg[:45]) not in s2:
                out.append(f"« {nom} » p. 2 : légende de la vue {cle} « {leg[:45]} » absente")
        for a in res["avertissements"]:
            if plat_sym(a["message"][:60]) not in s12 and plat_sym(NT.abreger(a["message"])) not in s12:
                out.append(f"« {nom} » p. 2 : avertissement « {a['message'][:40]} » absent (entier ou abrégé)")
        for r in (MT.REFERENCES[0][:40], MT.REFERENCES[-1][-30:]):
            if plat_sym(r) not in s2:
                out.append(f"« {nom} » p. 2 : la référence « {r} » manque")
        for x in (f"R = {NT._kn(ef['R']).replace(',00', '')}", f"L = {f0(g['L'])}"):
            if x not in p12:
                out.append(f"« {nom} » : le texte du dessin « {x} » manque")
        n_p2 = len(page2_hors_cartouche(p2))
        if n_p2 > SEUIL_PAGE2:
            out.append(f"« {nom} » p. 2 : trop de texte ({n_p2} caractères hors cartouche, budget {SEUIL_PAGE2})")
        # les mailles colorées du champ de von Mises, dans le PDF : les couleurs de l'échelle, maille par maille, le coin compris
        # (pymupdf : un chemin rempli par maille — au moins 90 % du compte du moteur, et au plus 3 % des couleurs attendues
        # absentes ; rendu : au moins quarante couleurs froides — bleu, cyan, vert — et le rouge de la barre)
        n_col, vues = fills_page(octets, i + 1)
        attendues = couleurs_attendues(res)
        if n_col is not None:
            manquent = attendues - vues
            if n_col < 0.9 * nb_mailles(res) or len(manquent) > 0.03 * len(attendues) or SC.ROUGE_TAUX.upper() not in vues:
                out.append(f"« {nom} » p. 2 : mailles colorées — {n_col} remplissages colorés pour {nb_mailles(res)} mailles, {len(manquent)}/{len(attendues)} "
                           f"couleurs de l'échelle absentes, rouge {SC.ROUGE_TAUX} {'présent' if SC.ROUGE_TAUX.upper() in vues else 'absent'}")
        else:
            froides = [h for h in vues if _colore(h) and int(h[5:7], 16) > int(h[1:3], 16)]
            if len(froides) < 40 or SC.ROUGE_TAUX.upper() not in vues:
                out.append(f"« {nom} » p. 2 : mailles colorées (rendu) — {len(froides)} couleurs froides distinctes, rouge {'présent' if SC.ROUGE_TAUX.upper() in vues else 'absent'}")
    if len(tx) != attendu_pages:
        out.append(f"{len(tx)} pages pour {attendu_pages} attendues (garde + pages de chaque cornière)")
    # le sommaire de la garde : les pages annoncées sont celles où chaque cornière commence
    annonces = [int(x) for x in re.findall(r"page (\d+)", tx[0])]
    if annonces != [i + 1 for i in debuts]:
        out.append(f"sommaire : pages annoncées {annonces}, pages réelles {[i + 1 for i in debuts]}")
    return out


def ecarts_identification(octets, infos=INFOS):
    tx = pages_brutes(octets)
    out = []
    tout = norm(" ".join(tx))
    out += [f"« {x} » imprimé" for x in INTERDITS if x in tout]
    g = plat(tx[0])
    for lab, val in (("PROJET", infos["chantier"]), ("PARTIE", infos["nom"]), ("DATE", infos["date"]), ("INDICE", infos["indice"])):
        if plat(lab + val) not in g:
            out.append(f"garde : {lab} {val} absent")
    for i, t in enumerate(tx[1:]):
        if f"{infos['date']} · indice {infos['indice']}" not in norm(t):
            out.append(f"page {i + 2} : « {infos['date']} · indice {infos['indice']} » absent de l'en-tête")
    return out


# ---------------------------------------------------------------- 1. la note de chaque état, la parité
NOTES, DUREES, MESURES = {}, {}, {}
for nom_e, e in ETATS.items():
    t0 = time.time()
    f = NC.fichier(copy.deepcopy(e), INFOS)
    DUREES[nom_e] = time.time() - t0
    octets = base64.b64decode(f["base64"])
    NOTES[nom_e] = octets
    n = len(e["cornieres"])
    tx = pages_brutes(octets)
    chk(f"{nom_e} : un PDF ({len(tx)} pages, {len(octets) // 1000} ko, {DUREES[nom_e]:.2f} s), nommé d'après la note, rien qui déborde",
        octets[:5] == b"%PDF-" and len(tx) >= 1 + 2 * n and f["nom"] == f"Note de calcul - {C.nom_fichier(NOM, '')}.pdf" and f["type"] == "application/pdf"
        and not NT.derniers_avertissements and C.refus_fichier(f) is None, (octets[:8], len(tx), f["nom"], NT.derniers_avertissements))
    ec = ecarts_parite(e, octets)
    chk(f"{nom_e} : parité avec le moteur — nom, statut, η max, état limite, taux, M1 à M4, H, N et V, formules, hypothèses compactes, charge et "
        "ancrage, page 2 (résumé, trois vues, légendes, limites, références, budget), mailles colorées", not ec, ec[:5])
    ec = ecarts_identification(octets)
    chk(f"{nom_e} : l'identification de la note du site dans la garde et l'en-tête de chaque page", not ec, ec[:4])
    for c in e["cornieres"]:
        i, k = debut_pages(tx, MO.verifier(c)["corniere"]["nom"])
        if i is not None:
            MESURES[(nom_e, i)] = (len(hypotheses_plat(norm(tx[i]))), len(page2_hors_cartouche(norm(tx[i + 1]))), fills_page(octets, i + 1)[0] or 0)
chk("au moins seize états, dont deux notes à deux cornières et une note qui mêle ELS et ELU",
    len(ETATS) >= 16 and sum(1 for e in ETATS.values() if len(e["cornieres"]) == 2) >= 2
    and any({c.get("etat_limite", "ELS") for c in e["cornieres"]} == {"ELS", "ELU"} for e in ETATS.values()))
print(f"--   durée de génération : {min(DUREES.values()):.2f} à {max(DUREES.values()):.2f} s par note (CPython, {len(ETATS)} notes)")
print(f"--   hypothèses (caractères sans blancs) : {min(m[0] for m in MESURES.values())} à {max(m[0] for m in MESURES.values())} (budget {SEUIL_HYPOTHESES}) ; "
      f"page 2 hors cartouche : {min(m[1] for m in MESURES.values())} à {max(m[1] for m in MESURES.values())} (budget {SEUIL_PAGE2}) ; "
      f"remplissages colorés de la page 2 : {min(m[2] for m in MESURES.values())} à {max(m[2] for m in MESURES.values())} (≥ 90 % des mailles du moteur) — "
      f"{'pymupdf' if pymupdf else 'flux de contenu'}")
tailles = {nom_e: (NC.pdf(copy.deepcopy(e), INFOS), list(NT.dernieres_variantes))[1] for nom_e, e in ETATS.items()}
print(f"--   tailles retenues : {sorted(set(v for vs in tailles.values() for v in vs))} — {sum(1 for vs in tailles.values() if 'serrée' in vs)} note(s) serrée(s)")
chk("aucune note ne passe en page de suite sur les états d'essai ; la page 2 n'a rien à reporter (BLOCS ne porte que la page 1)",
    all(len(pages_brutes(o)) == 1 + 2 * len(e["cornieres"]) for (nom_e, e), o in zip(ETATS.items(), NOTES.values()))
    and all(cle[0] == 1 for cle in NT.BLOCS))

# le contenu de la note v1 relue : l'avis de relecture est imprimé (attention), le nom et la section gardés
tv1 = norm(" ".join(pages_brutes(NOTES["note v1 relue (modèle lame), saisie libre"])))
chk("une note v1 relue : l'avis « note d'avant le 10/11/2026, modèle lame » est imprimé dans les limites, le nom et la saisie libre 100 × 100 × 10 gardés",
    "modèle lame" in tv1 and "Ancienne" in tv1 and "100 × 100 × 10" in tv1 and "saisie libre" in tv1, tv1[:300])
# la garde (ancrage bas) : l'avertissement rouge avec ses deux pourcentages — jamais abrégé
res_g = MO.verifier(ETATS["garde : ancrage bas, cas de contrôle, R_ELU saisie"]["cornieres"][0])
tg = norm(" ".join(pages_brutes(NOTES["garde : ancrage bas, cas de contrôle, R_ELU saisie"])))
chk("la garde : l'avertissement rouge « équilibre P1 plus sévère » avec ses deux pourcentages est imprimé ; R_ELU saisie (68,9 kN), pas « auto »",
    res_g["garde"]["depasse"] and f"{100 * res_g['garde']['taux_equilibre_P1']:.1f} %".replace(".", ",") in tg
    and f"{100 * res_g['garde']['taux_historique']:.1f} %".replace(".", ",") in tg and "plus sévère" in tg and "68,9" in tg
    and plat_sym("R_ELU = 1,5·R") not in plat_sym(tg),
    [x for x in tg.split(". ") if "sévère" in x or "68,9" in x][:3])
# la configuration B : la section nette, la pression sur le support, le mode imposé
tb = norm(" ".join(pages_brutes(NOTES["B, Y = 85"])))
res_b = MO.verifier(ETATS["B, Y = 85"]["cornieres"][0])
chk("la configuration B : la section nette (b_net, la face du trou), la pression sur le support, l'avertissement du mode imposé",
    "nette au trou (dessus)" in tb and "SECTIONNETTE" in plat(tb) and F(res_b["ancrages"]["pression_support"], 1) in tb
    and "Pression du bout libre sur le support" in tb and "formules du classeur ne s'appliquent pas" in tb and "(info)" not in tb,
    [x for x in tb.split(" · ") if "nette" in x][:3])
# l'ELU : la conclusion dit le critère et γ_M0, la coupe et le cartouche disent l'état limite
te = norm(" ".join(pages_brutes(NOTES["ELU, R = 60 saisie à l'ELU (A historique)"])))
chk("l'ELU : la conclusion dit « à l'ELU, critère de von Mises (EN 1993-1-1 § 6.2.1(5)), γ_M0 = 1,0 » ; la légende de la coupe dit « R la réaction ELU » "
    "et « critère de von Mises » ; le cartouche « ELU : f_y/γ_M0 = 355,0 » ; la formule du taux écrit la limite f_y/γ_M0",
    plat_sym("à l'ELU, critère de von Mises (EN 1993-1-1 § 6.2.1(5)), γ_M0 = 1,0") in plat_sym(te) and "R la réaction ELU" in te
    and plat_sym("ELU : f_y/γ_M0 = 355,0") in plat_sym(te) and plat_sym("η = σ_éq/ f_y/γ_M0 = 341,5/355") in plat_sym(te)
    and plat_sym("R_ELU") not in plat_sym(te), [x for x in te.split(" · ") if "ELU" in x][:4])
# la note ELS + ELU : chaque page dit son état limite, la garde reste générale
tm = pages_brutes(NOTES["ELS + ELU dans une note"])
chk("une note ELS + ELU : la page de la cornière ELS dit « R la réaction ELS », celle de l'ELU « R la réaction ELU », la légende de la garde reste "
    "générale et tient sur sa ligne",
    "R la réaction ELS" in norm(tm[1]) and "R la réaction ELU" in norm(tm[3]) and "ELS : σadm = fy/coef ; ELU : fy/γM0" in norm(tm[0])
    and NT.string_width(NT.LEGENDE_GARDE, "Carlito", 7.4) <= 595 - 2 * 52, [norm(t)[-200:] for t in tm[:4:2]])
# le résumé de méthode tient en trois lignes sur chaque état ; les limites en trois lignes au plus
doc_meta = {"date": "x", "indice": "x"}
trop = []
for nom_e, e in ETATS.items():
    for c in e["cornieres"]:
        m = NT.Methode(NT.Calcul(c), doc_meta)
        n_res = len(m.lignes_resume(842 - 2 * NT.M)[1])
        n_lim = len(m.lignes_limites(842 - 2 * NT.M - NT.LAB_W)[1])
        if n_res > MT.LIGNES_RESUME or n_lim > NT.LIGNES_LIMITES:
            trop.append((nom_e, n_res, n_lim))
chk("page 2 : le résumé de méthode tient en trois lignes, les limites en trois lignes au plus, sur chaque cornière des 17 états", not trop, trop)
chk("abreger : « Chevilles non vérifiées : efforts… » → « Chevilles non vérifiées » ; un message court reste entier (sans son point)",
    NT.abreger("Chevilles non vérifiées : efforts N et V par ancrage fournis pour Hilti PROFIS, ELS et ELU séparés.") == "Chevilles non vérifiées"
    and NT.abreger("Effet de levier non couvert si les chevilles sont très raides (> ≈ 250 kN/mm).") == "Effet de levier non couvert si les chevilles sont très raides"
    and NT.abreger("Un avis court.") == "Un avis court")

# ---------------------------------------------------------------- 2. chaque formule ré-évaluée
n_sub, ecarts = 0, []
for nom_e, e in ETATS.items():
    for c in e["cornieres"]:
        res = MO.verifier(c)
        for b, subs in FO.toutes(res):
            for s in subs:
                n_sub += 1
                ok1, v1 = FO.verifier(s)
                ok2, v2 = FO.exact(s)
                if not (ok1 and ok2):
                    ecarts.append((nom_e, b["cle"], FO.rendre(s), v1, v2))
chk(f"{n_sub} formules (tous les blocs de {sum(len(e['cornieres']) for e in ETATS.values())} cornières) ré-évaluées : les nombres imprimés redonnent le "
    "résultat du moteur (arrondi d'affichage) et les termes exacts le redonnent à 1e-9", not ecarts and n_sub >= 1200, ecarts[:4])
cles_moteur = {f["cle"] for e in ETATS.values() for c in e["cornieres"] for b in MO.verifier(c)["calcul"] for f in b["formules"]}
chk("chaque clé de formule que le moteur nomme a son gabarit, et chaque gabarit sert au moins une fois (dont les quatre de l'ELU)",
    cles_moteur <= set(FO.GABARITS) and set(FO.GABARITS) <= cles_moteur
    and {"sigma_lim_ELU", "tau_lim_ELU", "N_ELU_direct", "V_ELU_direct"} <= cles_moteur,
    (sorted(cles_moteur - set(FO.GABARITS)), sorted(set(FO.GABARITS) - cles_moteur)))
res0 = MO.verifier(ETATS["A historique (cornière neuve)"]["cornieres"][0])
chk("en mode historique, les cellules du classeur sont imprimées avec les étiquettes (C19 C35, C55 C25 C23 C56 C57, C82 C83…)",
    all(x in norm(pages_brutes(NOTES["A historique (cornière neuve)"])[1]) for x in ("C19 C35", "C55 C25 C23 C56 C57", "C78 C82 C83")),
    norm(pages_brutes(NOTES["A historique (cornière neuve)"])[1])[:300])
chk("la section gouvernante est écrite « M3 », pas « M(s) » ; le taux « η »", any(s.nom == "M3" for s in FO.chaine(FO.bloc(res0, "S3")))
    and any(s.nom == "η" for s in FO.coupe(FO.bloc(res0, "S3"))) and not any(s.nom in FO.TAUX_DETAIL for s in FO.coupe(FO.bloc(res0, "S3"))))
res_elu = MO.verifier(ETATS["ELU, R = 60 saisie à l'ELU (A historique)"]["cornieres"][0])
s_eta = next(s for s in FO.renommer(FO.coupe(FO.bloc(res_elu, "S3")), NT.Calcul(ETATS["ELU, R = 60 saisie à l'ELU (A historique)"]["cornieres"][0]).symboles_limites())
             if s.nom == "η")
chk("formules.renommer : à l'ELU le terme « σ_adm » du taux s'écrit « (f_y/γ_M0) », le résultat et le gabarit ne bougent pas",
    "(f_y/γ_M0)" in FO.rendre(s_eta) and FO.exact(s_eta)[0] and FO.verifier(s_eta)[0], FO.rendre(s_eta))

# ---------------------------------------------------------------- 3. l'identification, le nom du fichier, la correspondance
e2 = ETATS["deux cornières (A avec a = 100, B avec u_V = 0 et c = 10)"]
chk("infos_doc : la correspondance de la poutre (projet = chantier, partie = nom, indice, date)",
    NC.infos_doc(e2, INFOS) == {"nom_projet": CHANTIER, "partie": NOM, "indice": "C", "date": "03/04/2026"})
chk("sans chantier : projet = le nom de la note, partie = « Cornières d'appui » ; sans infos : le nom de la cornière seule, indice A",
    NC.infos_doc(e2, {"nom": "Terrasse"}) == {"nom_projet": "Terrasse", "partie": NC.TITRE, "indice": "A"}
    and NC.infos_doc(ETATS["A équilibre"], {}) == {"nom_projet": "Équilibre", "partie": NC.TITRE, "indice": "A"}
    and NC.infos_doc(e2, {}) == {"nom_projet": NC.TITRE, "partie": NC.TITRE, "indice": "A"})
f_sans = NC.fichier(copy.deepcopy(ETATS["A équilibre"]), {})
t_sans = norm(" ".join(pages_brutes(base64.b64decode(f_sans["base64"]))))
chk("sans infos : le fichier porte le nom de la cornière, la garde dit « indice A » et la date du jour",
    f_sans["nom"] == "Note de calcul - Équilibre.pdf" and "indice A" in t_sans and time.strftime("%d/%m/%Y") in t_sans, (f_sans["nom"], t_sans[:200]))
f_dang = NC.fichier(copy.deepcopy(ETATS["A équilibre"]), {"nom": "Note <b>/x"})
chk("un nom de note dangereux donne un nom de fichier sûr (celui de commun.nom_fichier)",
    f_dang["nom"] == C.nom_fichier("Note de calcul - Note <b>/x", "pdf") and "<" not in f_dang["nom"] and "/" not in f_dang["nom"]
    and C.refus_fichier(f_dang) is None, f_dang["nom"])

# ---------------------------------------------------------------- 4. la page de suite (page 1), la page 2 qui ne se reporte jamais, les refus
vrai_recap = NT.Note.recapitulatif


def recapitulatif_bavard(self, d, fr):
    """Le récapitulatif suivi de vingt-quatre lignes : la colonne 3 de la page 1 déborde dans les deux tailles, le bloc
    passe en page de suite (où il tient)."""
    vrai_recap(self, d, fr)
    for i in range(24):
        self.rp(d, fr, f"Ligne de remplissage {i + 1} pour faire déborder la colonne du récapitulatif.", size=self.s_txt * 0.95, col=self.S.mut)


NT.Note.recapitulatif = recapitulatif_bavard
try:
    octets = NC.pdf(copy.deepcopy(e2), INFOS)
    reports = NT.derniers_reports
finally:
    NT.Note.recapitulatif = vrai_recap
tx = pages_brutes(octets)
chk("un bloc de la page 1 qui ne tient pas (récapitulatif allongé) passe en PAGE DE SUITE : 1 + 3 + 3 pages, « Suite p. 3 : récapitulatif » en page 1, "
    "« SUITE DE LA PAGE 1 » en page 3 avec le récapitulatif, le sommaire compte les pages (2 et 5), rien ne déborde, la note n'est pas refusée",
    len(tx) == 7 and "Suite p. 3" in norm(tx[1]) and "récapitulatif" in norm(tx[1]) and "SUITE DE LA PAGE 1" in norm(tx[3]) and "RÉCAPITULATIF" in plat(tx[3])
    and [int(x) for x in re.findall(r"page (\d+)", tx[0])] == [2, 5] and not NT.derniers_avertissements and reports
    and all(cle[0] == 1 for rep in reports for cle, _ in rep) and ecarts_parite(e2, octets) == [],
    (len(tx), reports, NT.derniers_avertissements, [t[:80] for t in tx[1:4]]))

vrai_init = NT.Calcul.__init__


def init_bavard(self, c):
    """Le moteur rend douze avertissements de plus : la page 2 les tient en trois lignes (abrégés, puis tronqués), sans
    page de suite ni refus."""
    vrai_init(self, c)
    self.res["avertissements"] = list(self.res["avertissements"]) + [
        {"code": f"essai_{i}", "niveau": MO.INFO, "message": f"Avertissement d'essai numéro {i} : une phrase assez longue pour occuper de la place (sans conséquence)."}
        for i in range(12)]


NT.Calcul.__init__ = init_bavard
try:
    octets = NC.pdf(copy.deepcopy(ETATS["A historique (cornière neuve)"]), INFOS)
    reports = NT.derniers_reports
finally:
    NT.Calcul.__init__ = vrai_init
tx = pages_brutes(octets)
chk("la page 2 ne passe JAMAIS en page de suite : avec douze avertissements de plus, trois pages, rien ne déborde, aucun report, les limites tronquées « … »",
    len(tx) == 3 and not NT.derniers_avertissements and reports == () and "…" in tx[2], (len(tx), reports, NT.derniers_avertissements))
try:
    NC.fichier(etat(corniere(nom="Trop loin", X=130)), INFOS)
    r = "aucune erreur"
except ValueError as err:
    r = str(err)
chk("une cornière hors domaine est refusée avec son nom et le message du moteur, jamais une note fausse",
    "« Trop loin »" in r and "au-delà de l'aile" in r, r)
try:
    NC.fichier({"version": 2, "cornieres": [], "actif": 0}, INFOS)
    r = "aucune erreur"
except ValueError as err:
    r = str(err)
chk("une note sans cornière est refusée", "aucune cornière" in r, r)
chk("generer_note accepte aussi une simple liste de cornières", NT.generer_note([corniere()], {"indice": "B"})[:5] == b"%PDF-")

# ---------------------------------------------------------------- 5. les mutations
e_mut = ETATS["deux cornières (A avec a = 100, B avec u_V = 0 et c = 10)"]
e_a = ETATS["A historique (cornière neuve)"]
vrai_poser = NT.poser
compte = []


def poser_oublieux(d, calc, doc_meta, k, reports=()):
    compte.append(1)
    return [] if len(compte) == 2 else vrai_poser(d, calc, doc_meta, k, reports)


NT.poser = poser_oublieux
try:
    mut = NC.pdf(copy.deepcopy(e_mut), INFOS)
finally:
    NT.poser = vrai_poser
chk("mutation « une cornière oubliée » : la parité rougit", ecarts_parite(e_mut, mut) != [])


def init_sans_dy(self, c):
    c = dict(c)
    c["dy"] = 0.0
    vrai_init(self, c)


NT.Calcul.__init__ = init_sans_dy
try:
    mut = NC.pdf(copy.deepcopy(e_a), INFOS)
finally:
    NT.Calcul.__init__ = vrai_init
chk("mutation « une donnée non transmise au moteur » (la tolérance dy oubliée) : la parité rougit", ecarts_parite(e_a, mut) != [])

vrai_E = NT.E
NT.E = types.SimpleNamespace(**{k: v for k, v in vars(E).items() if not k.startswith("__")})
NT.E.LIBELLES_MODE = {E.HISTORIQUE: "méthode du classeur", E.EQUILIBRE: "équilibre"}
try:
    mut = NC.pdf(copy.deepcopy(e_a), INFOS)
finally:
    NT.E = vrai_E
chk("mutation « le mode mal nommé » : la parité rougit", ecarts_parite(e_a, mut) != [])

vrai_gab = FO.GABARITS["Xe"]
FO.GABARITS["Xe"] = (("X", "r", "e"), "X - r", 1)
try:
    s_mut = next(s for s in FO.chaine(FO.bloc(res0, "S1")) if s.nom == "Xe")
    rouge = not FO.exact(s_mut)[0] and not FO.verifier(s_mut)[0]
finally:
    FO.GABARITS["Xe"] = vrai_gab
chk("mutation « un gabarit faux » (Xe = X − r) : la ré-évaluation rougit", rouge)

vrai_infos = NC.infos_doc
NC.infos_doc = lambda etat_, infos: {k: v for k, v in vrai_infos(etat_, infos).items() if k != "indice"}
try:
    mut = NC.pdf(copy.deepcopy(e_a), INFOS)
finally:
    NC.infos_doc = vrai_infos
chk("mutation « l'indice de la note perdu » : l'identification rougit (« indice 0 » imprimé)", ecarts_identification(mut) != [])

vrai_bandeau = NT.Note.bandeau


def bandeau_faux(self, d):
    """Le bandeau imprime l'η de la PREMIÈRE vérification (S1) à la place de l'η max ; le récapitulatif et la garde, eux,
    restent justes — la parité doit le voir sur la ligne du bandeau, pas « quelque part dans la page »."""
    res = self.C.res
    vrai = res["taux_max"]
    res["taux_max"] = res["verifications"][0]["taux"]
    try:
        return vrai_bandeau(self, d)
    finally:
        res["taux_max"] = vrai


NT.Note.bandeau = bandeau_faux
try:
    mut = NC.pdf(copy.deepcopy(e_a), INFOS)
finally:
    NT.Note.bandeau = vrai_bandeau
ec_bandeau = ecarts_parite(e_a, mut)
chk("mutation « l'η max du bandeau pris sur S1 » (le récapitulatif répète un « η max » juste) : la parité rougit, et sur la ligne du bandeau",
    ec_bandeau != [] and all("bandeau" in x for x in ec_bandeau), ec_bandeau[:3])

# --- révision § 9 : l'état limite non imprimé, la page 2 sans sa troisième vue, les hypothèses en prose, les rondelles oubliées
vrai_hyp = NT.Calcul.hypotheses
e_elu = ETATS["ELU, R = 60 saisie à l'ELU (A historique)"]
NT.Calcul.hypotheses = lambda self: [(lab, val) for lab, val in vrai_hyp(self) if lab != "État limite"]
try:
    mut_a, mut_u = NC.pdf(copy.deepcopy(e_a), INFOS), NC.pdf(copy.deepcopy(e_elu), INFOS)
finally:
    NT.Calcul.hypotheses = vrai_hyp
ec_a, ec_u = ecarts_parite(e_a, mut_a), ecarts_parite(e_elu, mut_u)
chk("mutation « l'état limite non imprimé » (la ligne des hypothèses retirée, à l'ELS et à l'ELU) : la parité rougit sur l'hypothèse de l'état limite",
    ec_a and ec_u and all("état limite" in x.lower() for x in ec_a + ec_u), (ec_a[:2], ec_u[:2]))

vrai_colonnes = NT.Methode.COLONNES
NT.Methode.COLONNES = vrai_colonnes[:2]
try:
    mut = NC.pdf(copy.deepcopy(e_a), INFOS)
finally:
    NT.Methode.COLONNES = vrai_colonnes
ec = ecarts_parite(e_a, mut)
chk("mutation « la page 2 sans sa troisième vue » (les contraintes de von Mises) : la parité rougit — la colonne manque, les mailles colorées aussi",
    ec and any("CONTRAINTES DE VON MISES" in x for x in ec) and any("mailles colorées" in x for x in ec), ec[:3])

PROSE = [("Configuration", "A — aile horizontale EN BAS : la charge pose sur l'aile horizontale, l'aile verticale monte le long du mur, l'ancrage est "
                           "au-dessus de l'angle ; le talon s'appuie sur le mur, l'aile verticale est tendue"),
         ("Méthode", "aux contraintes admissibles, efforts ELS non majorés : σ_adm = f_y/1,5 = 236,7 N/mm² · τ_adm = σ_adm/√3 = 136,6 N/mm² · "
                     "σ_eq = √(σ² + 3τ²) ≤ σ_adm — ce n'est pas une justification EC3 aux états limites"),
         ("Mode", "méthode historique, feuille P8 du classeur : sections avec le pivot e/2 (YH = Y − e/2 = 77,5 mm), τ = R/(L·t) dans les deux ailes, "
                  "traction d'ancrage sur le bras Y − jeu = 75 mm (C76 = 10 mm) · maxima du congé en information"),
         ("Tranche", "courte : L = 300 mm est la largeur de toutes les sections ; 3 ancrage(s) se partagent R et la traction · largeur d'appui "
                     "a = 100 mm : L_eff = min(L ; a + 2·Xe) = 194 mm sur l'aile horizontale (σ × 1,546)")]
NT.Calcul.hypotheses = lambda self: vrai_hyp(self) + PROSE
try:
    mut = NC.pdf(copy.deepcopy(e_a), INFOS)
finally:
    NT.Calcul.hypotheses = vrai_hyp
ec = ecarts_parite(e_a, mut)
chk("mutation « les hypothèses en prose longue » (les phrases d'avant ajoutées) : la parité rougit sur le budget de caractères du bloc",
    ec and all("hypothèses trop longues" in x for x in ec), ec[:3])

vrai_charge = NT.Calcul.charge_et_ancrage
NT.Calcul.charge_et_ancrage = lambda self: [(lab, val) for lab, val in vrai_charge(self) if lab != "Rondelles"]
try:
    mut = NC.pdf(copy.deepcopy(ETATS["boulon traversant, deux rondelles, épaisseur 200"]), INFOS)
finally:
    NT.Calcul.charge_et_ancrage = vrai_charge
ec = ecarts_parite(ETATS["boulon traversant, deux rondelles, épaisseur 200"], mut)
chk("mutation « les rondelles non imprimées » : la parité rougit sur les deux rondelles (ISO 7093-1 contre l'aile, ISO 7089 sous l'écrou)",
    len(ec) == 2 and all("ISO 70" in x for x in ec), ec[:3])

# ---------------------------------------------------------------- 6. comme dans le navigateur
sys.path.insert(0, os.path.join(RACINE, "site", "web"))
try:
    import construire as K  # noqa: E402
finally:
    sys.path.pop(0)
SONDE = r"""
import builtins, json, os, sys
RACINE, STUB, REFUS = {racine!r}, {stub!r}, {refus!r}
sys.path.insert(0, STUB); sys.path.insert(1, RACINE); os.chdir(RACINE)
class Refus:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in REFUS or name == "reportlab.lib._rl_accel":
            raise ImportError("refusé dans le navigateur : " + name)
sys.meta_path.insert(0, Refus())
lus = set(); _open = builtins.open
def espion(f, *a, **k):
    try: lus.add(os.path.abspath(f))
    except Exception: pass
    return _open(f, *a, **k)
builtins.open = espion
def depot():
    mods = {{os.path.abspath(m.__file__) for m in list(sys.modules.values()) if getattr(m, "__file__", None)}}
    return {{os.path.relpath(f, RACINE) for f in mods | lus if f.startswith(RACINE + os.sep) and not f.endswith(".pyc")}}
from acier.corniere import moteur as MO
from api_web import dessin_corniere as DC
with _open({etats!r}, encoding="utf-8") as f:
    etats = json.load(f)
for e in etats:
    for c in e["cornieres"]:
        DC.vues(MO.verifier(c))
ecran = depot()
numpy_ecran = "numpy" in sys.modules
from api_web import note_corniere as NC
sorties = [NC.fichier(e, {infos!r})["base64"] for e in etats]
note = depot()
import PIL
print(json.dumps({{"note": sorted(f for f in note - ecran if not f.endswith(".ttf")), "numpy_ecran": numpy_ecran,
                   "numpy": "numpy" in sys.modules, "polices": sorted(os.path.basename(f) for f in lus if f.endswith(".ttf")),
                   "pil": PIL.__file__, "fonttools": "fontTools" in sys.modules,
                   "rl_accel": any(m.split(".")[-1] == "_rl_accel" for m in sys.modules), "sorties": sorties}}))
"""


def sonde(etats, infos):
    dossier = tempfile.mkdtemp()
    try:
        for f in K.NOTE_POLICES:
            shutil.copy(os.path.join(RACINE, "ndc_pdf", "fonts", f), dossier)
        ch = os.path.join(dossier, "etats.json")
        with open(ch, "w", encoding="utf-8") as fh:
            json.dump(etats, fh, ensure_ascii=False)
        code = SONDE.format(racine=RACINE, stub=O.STUB, refus=K.NOTE_REFUS, etats=ch, infos=infos)
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=RACINE, env={**os.environ, "NDC_FONT_DIR": dossier})
    finally:
        shutil.rmtree(dossier, ignore_errors=True)
    try:
        out = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        return {"erreur": p.stderr[-1500:]}
    out["sorties"] = [base64.b64decode(s) for s in out["sorties"]]
    return out


e_nav = [ETATS["A historique (cornière neuve)"], ETATS["deux cornières (A avec a = 100, B avec u_V = 0 et c = 10)"], ETATS["petite L 40×40×4"],
         ETATS["grosse L 250×250×35, équilibre, quatre ancrages"], ETATS["ELS + ELU dans une note"], ETATS["poutre A (160 au-dessus, 120 au-dessous)"]]
infos_nav = {"nom": "relevé", "chantier": "C", "indice": "B", "date": "01/01/2026"}
s = sonde(e_nav, infos_nav)
chk("navigateur : sans fontTools ni accélérateurs C, avec le Pillow factice et les seules polices du paquet, la note se fait (dont ELS + ELU, poutre)",
    "erreur" not in s and s["fonttools"] is False and s["rl_accel"] is False and s["pil"].startswith(O.STUB), s.get("erreur") or {k: v for k, v in s.items() if k != "sorties"})
if "erreur" not in s:
    cpy = [O.pages(NC.pdf(copy.deepcopy(x), infos_nav)) for x in e_nav]
    chk("navigateur : le texte de chaque page est celui de la note faite en CPython", [O.pages(o) for o in s["sorties"]] == cpy)
    pol = sorted({p for o in s["sorties"] for p in O.polices(o)})
    noms_paquet = {f[:-4] for f in K.NOTE_POLICES}
    chk(f"navigateur : les polices de la note sont celles du paquet (et Helvetica, police standard non embarquée) : {pol}",
        set(pol) <= noms_paquet | {"Helvetica"} and set(s["polices"]) <= set(K.NOTE_POLICES), (pol, s["polices"]))
    chk("navigateur : numpy est déjà chargé par l'écran du module (le catalogue des cornières) — la note ne l'ajoute pas", s["numpy_ecran"] is True)
    with open(os.path.join(RACINE, "site", "web", "moteur", "manifeste.json"), encoding="utf-8") as fh:
        moteur = set(json.load(fh)["fichiers"])
    # déjà dans le paquet de la note de la poutre (ndc_pdf, le Pillow factice) ou dans un paquet du site empaqueté entier
    # (api_web : construire.PAQUETS_ENTIERS — le manifeste du moteur est celui de la dernière construction)
    deja = {f for f in s["note"] if f.startswith(("ndc_pdf/", "site/note_pdf/")) or f.split("/")[0] in K.PAQUETS_ENTIERS}
    hors = sorted(set(s["note"]) - moteur - deja)
    print(f"--   fichiers du dépôt chargés par la note, hors écran, paquet du moteur et paquet de la poutre : {hors}")
    chk("navigateur : ces fichiers sont ceux que note_corniere.FICHIERS déclare, et tous les déclarés sont chargés (le paquet de la note les reçoit)",
        set(hors) <= set(NC.FICHIERS) and all(f in s["note"] for f in NC.FICHIERS), (hors, [f for f in NC.FICHIERS if f not in s["note"]]))
if "corniere" in K.NOTE_PDF:
    octets_zip, _ = K.paquet_note()
    noms = set(zipfile.ZipFile(io.BytesIO(octets_zip)).namelist())
    chk("paquet de la note (note.zip) : il porte les fichiers que la note des cornières déclare", all(f in noms for f in NC.FICHIERS),
        [f for f in NC.FICHIERS if f not in noms])
else:
    print("--   paquet de la note : « corniere » n'est pas encore dans construire.NOTE_PDF — contrôle en attente de l'intégration")

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

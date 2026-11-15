# -*- coding: utf-8 -*-
"""Générateur SAF — fondation de grue : la NOTE DE CALCUL PDF (api_web/note_saf.py, dessin_efforts_saf.py). Cahier des charges
du lot 5, version 4 (15/11/2026), § 10 et les essais NC1 à NC8 (sous-lot 5d) ; A3 dans la note.

Ce que la suite prouve :
  1. NC1, NC2 — la garde en deux parties (une seule pastille, plus de « F1 — »), la bande grise sur la seule page Données,
     aucun titre seul en bas d'une colonne, aucune colonne vide au milieu de la note, plus de Coupes ni de Bordereau, plus les
     phrases du lot 4 (« estimation », « Torsion »).
  2. NC3, NC4 — les charges en kN/m² des lignes FF du SAF (dalle type, plots), les combinaisons de la feuille du SAF.
  3. NC5, NC6 — le sol (par k, tous les cas pour chaque k coché), les efforts (les cas de T14, le treillis sur les
     diagrammes, les deux cartes avec la ligne à m_Rd) ; le texte des dessins : 7 pt au moins une fois imprimés.
  4. NC7, NC8 — la comparaison avec SCIA (saisie, partielle, écart), les notes SCIA jointes à la fin du PDF ; une note
     illisible le dit (« avis »).
  5. A3, les données — les barres de chaque renfort, la bibliothèque des châssis (B3), l'annexe sol, deux fondations.
  6. MUTATIONS — arbre miroir (api_web copié, le reste du dépôt lié), contrôles rapides.

    /usr/bin/python3.12 tests/test_saf_grue_note.py [--rapide] [--racine dossier]
"""
import base64
import copy
import io
import json
import re
import os
import shutil
import subprocess
import sys
import tempfile
import time

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(ICI)
ARGS = sys.argv[1:]
RACINE_MOTEUR = ARGS[ARGS.index("--racine") + 1] if "--racine" in ARGS else RACINE
RAPIDE = "--rapide" in ARGS
os.chdir(RACINE_MOTEUR)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, RACINE_MOTEUR)

import fitz  # noqa: E402  (les tests seulement : relire le PDF, ses polices, ses dessins)

from api_web import commun as C  # noqa: E402
from api_web import routeur  # noqa: E402
from api_web import saf as M  # noqa: E402
from api_web import sol as SOL  # noqa: E402
from saf.fondation_grue import ecriture as EC  # noqa: E402
from saf.fondation_grue import projet as PR  # noqa: E402
from saf.fondation_grue import scia as SCM  # noqa: E402
from saf.fondation_grue import verification as V  # noqa: E402

OK, KO = [], []
INFOS = {"nom": "Grue G1", "chantier": "Chantier A", "indice": "B", "date": "15/11/2026"}
PANNEAU = (0xF3 / 255, 0xF4 / 255, 0xF7 / 255)            # la bande grise (Encre.panel)


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{str(info)[:500]}]" if info != "" and not cond else ""))


def q(**k):
    return json.loads(routeur.traiter(json.dumps(dict(module="saf", **k))))


def mod(etat, chemin, valeur):
    r = q(action="modifier", etat=etat, chemin=[str(c) for c in chemin], valeur=valeur)
    assert r["ok"], r.get("erreur")
    return r["etat"]


def propre(t):
    return " ".join(t.replace(" ", " ").replace(" ", " ").split())


class Note:
    """Une note imprimée : ses octets, son avis, le texte de chaque page, le document fitz."""

    def __init__(self, etat, infos=INFOS):
        from api_web import note_saf as NS
        r = q(action="telecharger", etat=etat, quoi="note_pdf", infos=infos)
        assert r["ok"], r.get("erreur")
        self.octets = base64.b64decode(r["fichier"]["base64"])
        self.avis = r.get("avis")
        self.nom = r["fichier"]["nom"]
        self.doc = fitz.open(stream=self.octets, filetype="pdf")
        self.pages = [propre(p.get_text()) for p in self.doc]
        self.texte = " ".join(self.pages)
        self.figures = list(NS.FIGURES)

    def page_de(self, mot):
        return next((i for i, t in enumerate(self.pages) if mot in t), None)


def fr(x, nd=1):
    return propre(C.fr(x, nd))


def titres_seuls(note, garde=True):
    """Les titres (les titres numérotés et ceux des chapitres : Poppins, ~9,4 à 9,8 pt) sans rien dessous dans leur
    colonne : [(page, titre)]."""
    out = []
    for ip, p in enumerate(note.doc):
        if ip == 0 and garde:
            continue
        spans = [s for b in p.get_text("dict")["blocks"] for li in b.get("lines", []) for s in li["spans"] if s["text"].strip()]
        H = p.rect.height
        for s in spans:
            if "Poppins" in s["font"] and 8.9 <= s["size"] <= 9.9 and s["bbox"][1] > 60:
                x0, y1 = s["bbox"][0], s["bbox"][3]
                dessous = [t for t in spans if t["bbox"][1] > y1 + 1 and abs(t["bbox"][0] - x0) < 70 and t["bbox"][1] < H - 20]
                if not dessous:
                    out.append((ip + 1, s["text"]))
    return out


def bandes_grises(note):
    """Les pages qui portent la bande grise (un grand rectangle couleur panneau, hors de la garde)."""
    out = []
    for ip, p in enumerate(note.doc):
        if ip == 0:
            continue
        for dr in p.get_drawings():
            f = dr.get("fill")
            r = dr.get("rect")
            if f and r and all(abs(a - b) < 0.01 for a, b in zip(f, PANNEAU)) and r.width > 200 and r.height > 300:
                out.append(ip + 1)
                break
    return out


def colonnes_vides(note):
    """Les pages (sauf la garde et la dernière) dont une moitié n'a rien (la page Données : la bande et sa colonne large)."""
    out = []
    for ip, p in enumerate(note.doc):
        if ip == 0 or ip == len(note.doc) - 1:
            continue
        W = p.rect.width
        spans = [s for b in p.get_text("dict")["blocks"] for li in b.get("lines", []) for s in li["spans"] if s["text"].strip() and s["bbox"][1] > 80]
        dessins = [d_["rect"] for d_ in p.get_drawings() if d_["rect"].y0 > 80]
        gauche = any(s["bbox"][0] < W / 2 - 20 for s in spans) or any(r.x1 < W / 2 for r in dessins)
        droite = any(s["bbox"][0] > W / 2 for s in spans) or any(r.x0 > W / 2 for r in dessins)
        if not (gauche and droite):
            out.append(ip + 1)
    return out


def chevauchements(note):
    """Les mots qui en recouvrent un autre (plus d'1 pt dans les deux sens), le même mot peint deux fois au même point mis
    à part : [(page, mot, mot)] — ce que l'œil voit : deux textes l'un sur l'autre ne se lisent pas (revue du 15/11/2026)."""
    import itertools
    out = []
    for ip, p in enumerate(note.doc):
        ws = [w for w in p.get_text("words") if w[4].strip()]
        for a_, b_ in itertools.combinations(ws, 2):
            ix = min(a_[2], b_[2]) - max(a_[0], b_[0])
            iy = min(a_[3], b_[3]) - max(a_[1], b_[1])
            if ix > 1.0 and iy > 1.0 and not (a_[4] == b_[4] and abs(a_[0] - b_[0]) < 0.01 and abs(a_[1] - b_[1]) < 0.01):
                out.append((ip + 1, a_[4], b_[4]))
    return out


def titres_sans_dessin(note):
    """Les titres « BX1 — M(s) et V(s) » dont le diagramme M(s) n'est pas sur la même page : [(page, bande)]."""
    out = []
    for ip, t in enumerate(note.pages):
        for b in ("BX1", "BX2", "BY1", "BY2"):
            if f"{b} — M(s) et V(s)" in t and f"{b} — M(s) et V(s) (suite)" not in t and f"M(s) [kN·m] · BANDE {b}" not in t:
                out.append((ip + 1, b))
    return out


def droite_donnees(note):
    """Jusqu'où va le texte de la page Données (fraction de la largeur) : une colonne large va au bord droit."""
    p = note.doc[1]
    xs = [w[2] for w in p.get_text("words") if w[1] > 100]
    return max(xs) / p.rect.width if xs else 0.0


class _Flux:
    """Une planche d'essai en flux de chapitres (sans calcul) : ``n`` lignes de remplissage, un chapitre, puis un tableau."""

    def __init__(self, n):
        from api_web import note_saf as NS
        vs = [{"titre": "Remplissage", "items": [("pr", f"Ligne {i} du remplissage.") for i in range(n)], "verdicts": []},
              {"chapitre": "Chapitre suivant"},
              {"titre": "Un tableau", "items": [("tab", [("A", 120.0, "left"), ("B", 120.0, "right")],
                                                 [[("lab", f"ligne {i}"), ("num", str(i))] for i in range(14)])], "verdicts": []}]
        sec = NS._numeroter({"poutre": "Essai", "section": "Flux", "beton": "", "acier": "", "etat": "", "couleur": "ok", "bande": False,
                             "n_cols": 2, "verifs": vs})
        b = io.BytesIO()
        d = NS.Planche().build(b, sections=[sec], doc={"titre": "Essai", "bureau": "Essai", "date": "", "indice": "A"}, garde=False)
        d.save()
        self.doc = fitz.open(stream=b.getvalue(), filetype="pdf")


def pdf_essai(lignes, pages=2):
    """Une « note SCIA » d'essai (reportlab), ``pages`` pages."""
    from reportlab.pdfgen import canvas
    b = io.BytesIO()
    c = canvas.Canvas(b)
    for i in range(pages):
        c.drawString(72, 720, f"{lignes} — page {i + 1}")
        c.showPage()
    c.save()
    return b.getvalue()


# ------------------------------------------------------------------ les contrôles RAPIDES (rejoués par les mutations)
def rapides():
    type_ = M._etat_type()
    el = PR.lire(type_["fondations"][0])
    calc = M.calculer(el)
    n = Note(type_)
    print("== 1. NC1, NC2 : la garde, la mise en page")
    g = n.pages[0]
    chk("NC1 la garde : Bureau, Note de calcul, Projet, Partie, Date, Indice ; l'objet « Fondation de grue — dalle 8,00 × 8,00 × 0,50 m — C30/37 » ; "
        "le sommaire en DEUX parties — « Vérification complémentaire — calcul indépendant de SCIA » avec UNE pastille (l'état global) et ses "
        "chapitres en petit, sans pastille ; « Annexe — Note SCIA » à joindre ; la légende des couleurs",
        all(x in g for x in ("Note de calcul", "Chantier A", "Grue G1", "15/11/2026", "Fondation de grue — dalle 8,00 × 8,00 × 0,50 m — C30/37",
                             "01 Vérification complémentaire — calcul indépendant de SCIA NON CONFORME pages 2 à",
                             "Données page 2", "Charges et combinaisons page 3", "Sol page", "Efforts page", "Principe d'armatures page",
                             "Comparaison avec SCIA page", "02 Annexe — Note SCIA — à joindre", "Vert : sous la limite"))
        and g.count("NON CONFORME") == 1 and "F1 —" not in n.texte, g[:700])
    chk("NC2 la bande grise seulement sur la page Données (le plan coté et les données clés) ; les autres pages prennent toute la largeur",
        bandes_grises(n) == [2] and "Fondation de grue — Données" in n.pages[1], bandes_grises(n))
    seuls = [(n_, t_) for n_ in range(22, 40) for t_ in titres_seuls(_Flux(n_), garde=False)]
    chk("NC2 un titre de chapitre ne reste jamais seul en bas d'une colonne : une planche d'essai, le chapitre posé à chaque hauteur de "
        "la colonne (18 remplissages) — il part avec son premier titre et le début de son tableau", not seuls, seuls[:3])
    chk("NC2 aucun titre seul en bas d'une colonne ; aucune colonne vide au milieu de la note (un chapitre court partage la page avec le "
        "suivant) ; plus de pages Coupes ni Bordereau ; plus « une estimation… », « Torsion : oui »",
        not titres_seuls(n) and not colonnes_vides(n) and all(x not in n.texte for x in ("Coupes", "Bordereau", "estimation", "Torsion : oui", "COUPE"))
        and "Fondation de grue — Charges et combinaisons · Sol" in n.pages[2], (titres_seuls(n), colonnes_vides(n)))
    print("== 2. NC3, NC4 : les charges et les combinaisons (les feuilles du SAF)")
    t = n.texte
    chk("NC3 dalle type : LC1 = 25 × 0,50 = 12,50 kN/m² ; le cas 1.a (LC2, jeu 1) : 350 kN → 1 400 kN/m², 440 → 1 760, 440 → 1 760, 350 → 1 400 "
        "(0,50 × 0,50 m = 0,250 m²) — les valeurs des lignes FF du SAF ; les 16 cas ; la position de chaque surface (X, Y du SAF)",
        "SF1 sur la dalle D1 : 25 × h 25 × 0,50 = 12,50 kN/m²" in t and "1.a LC2 1 350 1 400 440 1 760 440 1 760 350 1 400" in t
        and "3.h LC17 3" in t and "0,250 m²" in t and "P1 1,000 ; 7,000 0.75;1.25;1.25;0.75;0.75 7.25;7.25;6.75;6.75;7.25" in t, t[t.find("LC1 —"):][:500])
    pf = mod(mod(mod(type_, ["grue", "plots", "type"], "prefabrique"), ["grue", "plots", "a"], "2"), ["grue", "plots", "b"], "2")
    npf = Note(pf)
    chk("NC3 avec plots préfabriqués 2,00 × 2,00 × 0,35 : 243,056 ; 305,556 ; 305,556 ; 243,056 kN/m² sur le carré de diffusion de 1,440 m² ; "
        "les plots : FP1 à FP4 = 25 × 0,35 = 8,75 kN/m² (les valeurs du SAF, S1b)",
        "1.a LC2 1 350 243,056 440 305,556 440 305,556 350 243,056" in npf.texte and "carré de diffusion 1,20 × 1,20 m = 1,440 m²" in npf.texte
        and "FP1 à FP4 sur chaque plot : 25 × hauteur 25 × 0,35 = 8,75 kN/m²" in npf.texte, npf.texte[npf.texte.find("LC1 —"):][:600])
    fe = EC.feuilles(el, 2.0)
    combis = next(f for f in fe if f["nom"] == "StructuralLoadCombination")["lignes"][1:]
    manque = [li[0] for li in combis if f"{li[0]} linéaire LC1 × 1,00" + (f" + {li[10]} × 1,00" if li[10] else "") not in t]
    chk(f"NC4 les combinaisons : les {len(combis)} de la feuille StructuralLoadCombination du SAF, mêmes noms, même contenu (ELS-Car (auto)1 = "
        "LC1 ; ELS-Car (auto)n = LC1 + LCn, coefficients 1,00, linéaires) ; le sol pour SCIA : C1z, C1x, C1y, C2x, C2y de chaque k",
        len(combis) == 17 and not manque and "2 2 0,2 0,2 0,2 0,2 dans le SAF" in t and "10 10 1 1 1 1 à mettre dans SCIA" in t, manque[:3])
    print("== 3. NC5, NC6 : le sol, les efforts, la taille du texte des dessins")
    par_k = {r["k"]: r for r in calc["par_k"]}
    lignes_k = [f"{C.nb(k)} {fr(par_k[k]['sigma_max'] / V.KG_CM2, 2)} {par_k[k]['cas_max']} {fr(par_k[k]['w_max'], 2)} {fr(par_k[k]['decolle'], 1)} %"
                for k in (2.0, 5.0, 10.0, 20.0, 30.0)]
    lc = {c["nom"]: c["lc"] for c in calc["modele"]["cas"]}
    tous = [f"{c['nom']} {lc[c['nom']]} " + " ".join(f"{fr(par_k[k]['cas'][i]['sigma_max'] / V.KG_CM2, 2)} {fr(par_k[k]['cas'][i]['w_max'], 1)}" for k in (2.0, 5.0, 10.0))
            for i, c in enumerate(par_k[2.0]["cas"])]
    chk("NC5 le sol : trois lignes d'hypothèses ; la carte enveloppe (la limite) ; le tableau par k (σ max et son cas, tassement, décollé — les "
        "valeurs du module, U3/U6) ; le tableau de TOUS les cas pour chaque k coché (σ max et tassement) ; σ moyen ; l'équilibre (2 380 → le cas "
        "dimensionnant 3.a : 1 970,0 kN) et l'écart le plus grand",
        all(x in t for x in lignes_k) and all(x in t for x in tous) and "CONTRAINTE DE CONTACT — ENVELOPPE DES CAS · k = 2 ; 5 ; 10 MN/m³" in t
        and "Calcul indépendant de SCIA" in t and "σ moyen = (dalle 800 + réactions 1 580) kN ÷ 64,0 m² = 0,38 kg/cm²" in t
        and "Résultante du sol 1 970,0 kN · Charges : grue 1 170,0 + dalle 800,0 + plots 0,0 = 1 970,0 kN" in t
        and "Écart le plus grand sur 80 cas et 5 k : 0,00 % < 0,1 %" in t, ([x for x in lignes_k + tous if x not in t][:3]))
    ef = M.efforts(el, calc)
    t14 = [f"{b['nom']} M+ {fr(b['M_plus'])} kN·m {fr(b['M_plus'] / b['largeur'])} kN·m/m {b['cas_M_plus'][0]} · k = {C.nb(b['cas_M_plus'][1])}" for b in ef["bandes"]]
    t14 += [f"{b['nom']} V à d {fr(b['V_d'])} kN {fr(b['V_d'] / b['largeur'])} kN/m {b['cas_V_d'][0]} · k = {C.nb(b['cas_V_d'][1])}" for b in ef["bandes"]]
    t14 += [f"{b['nom']} M− {fr(b['M_moins'])} kN·m {fr(b['M_moins'] / b['largeur'])} kN·m/m {b['cas_M_moins'][0]} · k = {C.nb(b['cas_M_moins'][1])}"
            for b in ef["bandes"]]
    chk("NC6 les efforts : les diagrammes M(s) et V(s) des quatre bandes, le treillis du bas et du haut (m_Rd × 0,50 : 48,6 en x, 47,5 en y) en "
        "trait interrompu, les abscisses de V à d ; le tableau des bandes avec le cas et le k de chaque valeur — M+, M− et V à d (T14 : 3.b · k = 2…) ; deux cartes "
        "enveloppes des moments au point avec la ligne noire à m_Rd",
        all(f"M(s) [kN·m] · BANDE {b} DE 0,50 m" in t and f"V(s) [kN] · BANDE {b}" in t for b in ("BX1", "BX2", "BY1", "BY2"))
        and "treillis bas 48,6" in t and "treillis haut -48,6" in t and "treillis bas 47,5" in t and all(x in t for x in t14)
        and "EN BAS : LE PLUS GRAND DE mx ET my · LIGNE NOIRE : mRd = 95,0 kN·m/m" in t and "EN HAUT : LE PLUS NÉGATIF DE mx ET my" in t
        and "m haut min -140,8 kN·m/m" in t, [x for x in t14 if x not in t][:3])
    petits = [(nm, round(tt, 2)) for nm, tt in n.figures if tt < 7.0 - 1e-6]
    chk(f"§ 10.3 les dessins se lisent imprimés : les {len(n.figures)} dessins de la note (plan coté, carte de contrainte, 8 diagrammes, 2 cartes "
        "des moments, 2 plans du principe ; lot 6 : la légende et les 2 diagrammes de couverture du calcul des bandes) ont un texte de 7 pt "
        "au moins une fois posés",
        len(n.figures) == 18 and sum(nm.startswith("Couverture") for nm, _ in n.figures) == 3 and not petits, petits or n.figures)
    print("== 4. NC7, NC8 : SCIA")
    chk("NC7 sans valeur SCIA : « Valeurs SCIA non saisies », le tableau garde les valeurs du module (r = 1)",
        "Valeurs SCIA non saisies" in t and "Bandes BX : M+ le plus fort (x = y) 71,6 kN·m SCIA non saisi" in t, t[t.find("Module / SCIA"):][:300])
    mo = SCM.module(ef, calc["par_k"])
    sc = mod(mod(mod(type_, ["scia", "bx_m"], repr(round(mo["bx_m"] * 1.05, 4))), ["scia", "bx_v"], repr(round(mo["bx_v"] * 0.97, 4))),
             ["scia", "mx"], repr(round(mo["mx"] * 1.02, 4)))
    nsc = Note(sc)
    part = Note(mod(type_, ["scia", "bx_m"], repr(round(mo["bx_m"] * 1.30, 4))))
    chk("NC7 valeurs saisies : le tableau complet (module, SCIA, rapport, écart, état) et « Les deux calculs concordent à 5,0 % près » ; une "
        "valeur à 1,30 × : « Écarts à vérifier : +30,0 % … » et « saisies en partie » ; un champ vide reste « SCIA non saisi » (E2)",
        "Les deux calculs concordent à 5,0 % près." in nsc.texte and "1,05 5,0 % concorde" in nsc.texte and "0,97 -3,0 % concorde" in nsc.texte
        and "Valeurs SCIA non saisies" not in nsc.texte and "Écarts à vérifier : +30,0 % sur bandes BX : M+ le plus fort" in part.texte
        and "SCIA non saisi" in part.texte and "saisies en partie" in part.texte, nsc.texte[nsc.texte.find("Module / SCIA"):][:500])
    annexe = pdf_essai("NOTE SCIA ESSAI k = 2", 2)
    na = Note(type_, dict(INFOS, annexes=[{"nom": "scia k2.pdf", "base64": base64.b64encode(annexe).decode()}]))
    nb_ = Note(type_, dict(INFOS, annexes=[{"nom": "faux.pdf", "base64": base64.b64encode(b"pas un pdf").decode()}]))
    chk("NC8 les notes SCIA jointes vont à la FIN du PDF (ses 2 pages telles quelles) ; la garde le dit (« jointe : 1 PDF — scia k2.pdf », ses "
        "pages) ; un fichier illisible n'empêche pas la note : l'écran le dit (« Notes SCIA non jointes : faux.pdf … »), la garde dit « à joindre »",
        len(na.doc) == len(n.doc) + 2 and "NOTE SCIA ESSAI k = 2 — page 1" in na.pages[-2] and "NOTE SCIA ESSAI k = 2 — page 2" in na.pages[-1]
        and f"02 Annexe — Note SCIA pages {len(n.doc) + 1} à {len(n.doc) + 2} jointe : 1 PDF — scia k2.pdf" in na.pages[0] and na.avis is None
        and len(nb_.doc) == len(n.doc) and (nb_.avis or "").startswith("Notes SCIA non jointes : faux.pdf") and "à joindre" in nb_.pages[0],
        (len(na.doc), len(n.doc), na.pages[0][-300:], nb_.avis))
    print("== 5. A3, les données")
    e7 = mod(mod(type_, ["dalle", "Lx"], "10,7"), ["dalle", "Ly"], "10,7")
    n7 = Note(e7)
    el7 = PR.lire(e7["fondations"][0])
    fer7 = M.armatures(el7, M.calculer(el7))
    lignes_r = [f"{r['rep']} {'bas' if r['face'] == 'inf' else 'haut'} {r['sens']} {r['n']} Ø{r['d']} {fr(r['longueur'], 2)} m {fr(r['largeur'], 2)} m"
                for r in fer7["renforts"]]
    chk(f"A3 dalle 10,70 × 10,70 × 0,50 sans plot : le tableau des renforts de la note montre les mêmes barres que la carte de l'écran (les "
        f"{len(lignes_r)} renforts : repère, face, sens, barres, longueur, largeur) ; « Principe d'armatures, pas un plan d'exécution. »",
        all(x in n7.texte for x in lignes_r) and "Principe d'armatures, pas un plan d'exécution." in n7.texte
        and "U de fermeture : Ø12 e 15 cm sur tout le pourtour (42,8 m)." in n7.texte, [x for x in lignes_r if x not in n7.texte][:3])
    revue_15_11(type_, n, npf, n7, ef, mo)


def revue_15_11(type_, n, npf, n7, ef, mo):
    """Les défauts de la note trouvés par la relecture indépendante du 15/11/2026 (chacun a sa mutation)."""
    print("== 6. revue du 15/11/2026 : chevauchements, titres, colonnes, SCIA, hors du moteur Dalles")
    t30 = mod(mod(mod(mod(mod(mod(type_, ["dalle", "Lx"], "13"), ["dalle", "Ly"], "13"), ["grue", "beta"], "30"), ["grue", "plots", "type"], "prefabrique"),
                  ["grue", "plots", "a"], "2"), ["grue", "plots", "b"], "2")
    n30 = Note(t30)
    e5 = copy.deepcopy(type_)
    e5["fondations"][0]["sol"]["k"] = [{"k": k, "scia": True} for k in (2.0, 5.0, 10.0, 20.0, 30.0)]
    n5 = Note(e5)
    ch = {nom: chevauchements(x) for nom, x in (("type", n), ("plots", npf), ("10,70", n7), ("tourné 30°", n30), ("cinq k", n5))}
    chk("revue N1 · aucun mot n'en recouvre un autre (dalle type, plots, 10,70, châssis tourné de 30°, cinq k cochés) : les repères x et y "
        "d'un même pied, le liseré des diagrammes, « P4 » et la cote 0,75, la valeur pointée et le plot, les en-têtes « k = … » du tableau "
        "de tous les cas",
        not any(ch.values()), {k_: v_[:3] for k_, v_ in ch.items() if v_})
    chk("revue N2 · le titre « BX1 — M(s) et V(s) » part avec son diagramme (dalle type, 10,70) ; la page Données tient en une colonne large "
        "(son texte va au bord droit) ; le carré de diffusion d'un châssis tourné : 1,20 × 1,20 m (ses côtés, pas sa boîte)",
        not titres_sans_dessin(n) and not titres_sans_dessin(n7) and droite_donnees(n7) > 0.9
        and "carré de diffusion 1,20 × 1,20 m = 1,440 m²" in n30.texte,
        (titres_sans_dessin(n), titres_sans_dessin(n7), round(droite_donnees(n7), 3), n30.texte[n30.texte.find("Surface de charge"):][:120]))
    c45 = Note(mod(type_, ["dalle", "beton"], "C45/55"))
    chk("revue N3 · béton C45/55 (hors du moteur Dalles) : le chapitre Efforts reste (diagrammes, tableau des bandes, cartes), sans trait "
        "m_Rd, et le dit ; les armatures « non vérifiées »",
        all(x in c45.texte for x in ("M(s) [kN·m] · BANDE BX1", "Tableau des bandes", "Moments au point", "Treillis non vérifié",
                                     "Armatures non vérifiées")) and "treillis bas" not in c45.texte, c45.texte[:300])
    r_ = mod(mod(type_, ["scia", "bx_m"], repr(round(mo["bx_m"] * 1.10, 4))), ["scia", "mx"], repr(round(mo["mx"] * 1.20, 4)))
    nr = Note(r_)
    ex = ef["extremes"]
    chk("revue N4 · SCIA saisi (r = 1,10 en bas, 1,20 en haut) : les valeurs écrites sous les cartes des moments sont celles des cartes "
        "(× le rapport SCIA) et le disent",
        f"mx max = {fr(ex['inf_x'] * 1.10)} kN·m/m" in nr.texte and f"mx min = {fr(ex['sup_x'] * 1.20)} kN·m/m" in nr.texte
        and "Valeurs du module × le rapport SCIA, comme les cartes." in nr.texte, nr.texte[nr.texte.find("mx max"):][:200])
    sig = mod(type_, ["scia", "sigma", "2"], "0,60")
    nsig = Note(sig)
    nsig2 = Note(mod(mod(mod(sig, ["scia", "bx_m"], repr(round(mo["bx_m"] * 1.02, 4))), ["scia", "bx_v"], repr(round(mo["bx_v"], 4))),
                     ["scia", "mx"], repr(round(mo["mx"] * 0.99, 4))))
    chk("revue N5 · σ max SCIA seul saisi (0,60 kg/cm² pour k = 2) : la comparaison n'est plus « non saisie », la ligne σ dit « à vérifier » ; avec "
        "les moments qui concordent : « Écarts à vérifier … σ max, k = 2 », jamais « concordent »",
        "Valeurs SCIA non saisies" not in nsig.texte and "Écarts à vérifier" in nsig.texte and "sur σ max, k = 2 MN/m³" in nsig.texte
        and re.search(r"σ max, k = 2 MN/m³ \S+ kg/cm² 0,60 kg/cm² \S+ \S+ % à vérifier", nsig.texte) is not None
        and "Écarts à vérifier" in nsig2.texte and "concordent" not in nsig2.texte, nsig2.texte[nsig2.texte.find("Module / SCIA"):][:400])
    vi = M._avec(dalle={"Lx": 7.4, "Ly": 7.4}, grue={"a": 3.8, "b": 3.8, "plots": {"type": "coule", "a": 3.0, "b": 3.0, "h": 0.35}})
    nvi = Note(mod(vi, ["scia", "bx_v"], "120"))
    chk("revue N6 · V à d illisible (dalle 7,40, châssis 3,80, plots coulés 3,00) et V SCIA saisi : la note s'imprime, la ligne dit « SCIA seul »",
        "SCIA seul" in nvi.texte, nvi.texte[nvi.texte.find("Module / SCIA"):][:300])
    vieux = mod(mod(mod(type_, ["scia", "bx_m"], "75"), ["scia", "bx_v"], "90"), ["scia", "mx"], "-130")
    nv = Note(mod(vieux, ["dalle", "h"], "0,45"))
    chk("revue N7 · valeurs SCIA tapées puis h changé : la note dit qu'elles sont d'un autre modèle, non utilisées",
        SCM.PERIME in nv.texte, nv.texte[nv.texte.find("Module / SCIA"):][:300])
    # V(s) avec son signe (15/11/2026) : V max au-dessus de l'axe, V min en dessous ; « V à d » sur la courbe où il est lu
    from api_web import dessin_efforts_saf as DE
    el7 = PR.lire(mod(mod(type_, ["dalle", "Lx"], "10,7"), ["dalle", "Ly"], "10,7")["fondations"][0])
    ef7 = M.efforts(el7, M.calculer(el7))
    svg = DE.tranchant(ef7["bandes"][0], 380.0, 130.0)
    lignes_ = re.findall(r'<polyline points="([^"]+)"', svg)
    axe = re.findall(r'<line x1="40.0" y1="([0-9.]+)" x2="[0-9.]+" y2="\1" stroke="#15181f"', svg)
    ys = [[float(p_.split(",")[1]) for p_ in li.split()] for li in lignes_]
    y0 = float(axe[0]) if axe else None
    # (V max monte bien au-dessus de l'axe — loin des pieds, l'enveloppe peut le frôler —, V min descend bien en dessous)
    dessus = y0 is not None and len(ys) == 2 and min(ys[0]) < y0 - 30 and max(ys[0]) < y0 + 10
    dessous = y0 is not None and len(ys) == 2 and max(ys[1]) > y0 + 30 and min(ys[1]) > y0 - 10
    chk("V(s) avec son signe, dalle 10,70, BX1 : V max tracé au-dessus de l'axe, V min en dessous (la courbe passe du positif au "
        "négatif sous chaque pied) ; « V à d 84,1 (168,2/m) » écrit sur la courbe ; la note le dit aussi",
        dessus and dessous and "V à d 84,1 (168,2/m)" in svg and "V à d 84,1 (168,2/m)" in n7.texte,
        (y0, [(min(y), max(y)) for y in ys], "V à d 84,1" in svg))
    lot6_bandes(type_, n7)


def lot6_bandes(type_, n7):
    """Lot 6 (15/11/2026) : le CALCUL DES BANDES — la fenêtre de calcul de l'écran (une par direction, ouverte par les lignes de
    la carte ARMATURES), le diagramme de couverture de M(s), le chapitre « Calcul des bandes » de la note."""
    import numpy as np
    from api_web import dessin_efforts_saf as DE
    print("== 6 bis. lot 6 : le calcul des bandes (écran et note)")
    e7 = mod(mod(type_, ["dalle", "Lx"], "10,7"), ["dalle", "Ly"], "10,7")
    r = q(action="calculer", etat=e7)
    vol, res = propre(r["volets"]), r["resultat"]
    i = vol.find('id="ck-bd-x"')
    bx = vol[i:vol.find('id="ck-bd-y"')] if i >= 0 else ""
    lignes_bx = re.findall(r'<td class="nm">(.*?)<span>', bx)
    chk("L6-a l'écran, dalle 10,70 : la fenêtre « Bandes BX1 = BX2 · section de calcul » (une pour les deux bandes pareilles) et celle des "
        "bandes BY ; ses lignes Hauteur, Inférieure X, Supérieure X, Effort tranchant (724 mm² ≤ A_s = 829 mm² ; V à d = 84,1 kN, au dixième) ; "
        "son diagramme de couverture (« R1 4 Ø12 : 107,0 ») ; les lignes Flexion, Hauteur et Effort tranchant de la carte l'ouvrent",
        'id="ck-bd-y"' in vol and "Bandes BX1 = BX2 · section de calcul" in bx and lignes_bx == ["Hauteur", "Inférieure X", "Supérieure X", "Effort tranchant"]
        and "724 mm²" in bx and "829 mm²" in bx and "V à d = 84,1 kN · sans étriers" in bx and 'class="sf-couv"><svg' in bx
        and "R1 4 Ø12 : 107,0" in bx and "R2 4 Ø12 : 107,0" in bx
        and res.count('data-calcul="bd-x"') == 3 and res.count('data-calcul="bd-y"') == 3, (lignes_bx, res.count('data-calcul="bd-x"'), bx[:300]))
    # la couverture : le treillis, puis le treillis + le renfort, atteint à un ancrage de chaque extrémité (au bord : tout de suite)
    r_ = {"rep": "R1", "s0": 1.0, "s1": 4.0, "ancrage": 0.6, "mRd_pose": 200.0, "bord0": False, "bord1": False}
    pts = DE._resistance(6.0, 100.0, [r_])
    xs, vs = [p_[0] for p_ in pts], [p_[1] for p_ in pts]
    pb = DE._resistance(6.0, 100.0, [dict(r_, s0=0.0, bord0=True)])
    lu = [round(float(np.interp(x, xs, vs)), 6) for x in (0.5, 1.3, 1.6, 2.5, 3.7, 4.5)]
    chk("L6-b le diagramme de couverture : 100 (le treillis) hors du renfort, une rampe sur l'ancrage (150 à mi-ancrage), 200 (treillis + "
        "renfort) au-delà ; un renfort arrêté au bord de la dalle reprend tout dès le bord",
        lu == [100.0, 150.0, 200.0, 200.0, 150.0, 100.0] and round(float(np.interp(1e-6, [p_[0] for p_ in pb], [p_[1] for p_ in pb])), 3) == 200.0,
        (lu, pb[:4]))
    chk("L6-c la note, dalle 10,70 : le chapitre « Calcul des bandes » (au sommaire), un bloc par direction (« Bande BX1 = BX2 — section 50 "
        "× 50 cm », « Bande BY1 = BY2 … »), le diagramme de couverture, les vérifications du moteur Dalles avec leurs formules et leurs "
        "verdicts (724 mm² ≤ A_s = 829 mm², 87 %)",
        "Calcul des bandes" in n7.pages[0] and "Bande BX1 = BX2 — section 50 × 50 cm" in n7.texte and "Bande BY1 = BY2 — section 50 × 50 cm" in n7.texte
        and "M(s) [kN·m] ET CE QUE REPRENNENT LES ARMATURES · BANDE BX1" in n7.texte and "R1 4 Ø12 : 107,0" in n7.texte
        and "724 mm² ≤As = 829 mm² 87 %" in n7.texte and "0,016 × 37 / 1,05 = 0,56" in n7.texte
        and "93,37 × 10" in n7.texte, n7.texte[n7.texte.find("CALCUL DES BANDES"):][:600])


def complets():
    print("== 6. les données de la note : la bibliothèque, l'annexe sol, deux fondations")
    etat = M._avec(projet={"projet": "Essai 2026", "grue": "132 HC", "indice": "A", "creation": "14/11/2026"}, dalle={"Lx": 9.0, "Ly": 9.0},
                   grue={"chassis": "liebherr-132hc-chassis-4600", "plots": {"type": "coule", "a": 2.0, "b": 2.0, "h": 0.35}})
    etat = mod(etat, ["sol", "reprise"], {"nom": "Sol essai", "chantier": "Chantier A", "etat": SOL._etat_deux_sondages()})
    n = Note(etat)
    el = PR.lire(etat["fondations"][0])
    t = n.texte
    chk("la note dit LES DONNÉES : le projet et la grue (« grue 132 HC » dans l'objet), le châssis par son nom ET son id, les appuis réels "
        "0,58 × 0,40 m à 45° et leur aire, les plots coulés 2,00 × 2,00 × 0,35 m, les jeux, E, k, la limite, le volume de béton ; aucun libellé "
        "coupé (« … ») ; plots coulés : « Non vérifiés » nomme leur ferraillage",
        all(x in t for x in ("Essai 2026", "— grue 132 HC", "Liebherr", "liebherr-132hc-chassis-4600", "réels (la bibliothèque)", "rectangle 0,58 × 0,40 m à 45°",
                             "0,2320 m²", "coulés en place", "2,00 × 2,00 × 0,35 m", "32 800", "Contrainte de contact admise", "0,50 kg/cm²",
                             "Plots coulés en place 4 × 2,00 × 2,00 × 0,35 = 5,60 m³", "le ferraillage des plots coulés en place"))
        and "…" not in t, t[:400])
    chk("l'annexe sol : la page de résultats du module « Rigidité du sol » — la note reprise, sa méthode, « À encoder dans SCIA », « Zonage de "
        "la dalle » — en chapitre « Annexe sol », au sommaire",
        all(x in t for x in ("ANNEXE SOL", "Sol essai", "Sondage CPT", "À encoder dans SCIA", "Zonage de la dalle",
                             propre(el["sol"]["reprise"]["page"]["blocs"][1]["lignes"][0][2])))
        and "Annexe sol page" in n.pages[0], t[-600:])
    krv = M._avec(grue={"chassis": "wolff-krv-10-60-a60"})
    tk = Note(krv).texte
    chk("B3 dans la note : Wolff KRV 10-60 — appuis « rectangle (saisie) », « Platine non indiquée dans la bibliothèque : les dimensions de la "
        "surface d'appui sont saisies », 0,50 × 0,50 m ; sans note de sol reprise, pas d'annexe",
        "Appuis rectangle (saisie)" in tk and "Platine non indiquée dans la bibliothèque : les dimensions de la surface d'appui sont saisies." in tk
        and "0,50 × 0,50 m" in tk and "ANNEXE SOL" not in tk and "…" not in tk, tk[300:800])
    deux = copy.deepcopy(M._etat_type())
    M.gerer_elements(deux, "ajouter")
    deux = q(action="calculer", etat=deux)["etat"]
    nd = Note(deux)
    chk("une note de DEUX fondations dont l'une n'a pas de réaction : chaque fondation a ses pages (« Fondation de grue F1 — … », « F2 »), la "
        "seconde dit « Poids propre seul » ; le sommaire nomme les chapitres de chacune",
        "Poids propre seul" in nd.texte and "Fondation de grue F2 — Données" in nd.texte and "F1 · Données page 2" in nd.pages[0]
        and "F2 · Données page" in nd.pages[0], nd.pages[0][:600])


MUTATIONS = [
    ("note_saf.py", '''    sec = dict(F, section="Données", bande=True, coupe_w=300, n_cols=1)''', '''    sec = dict(F, section="Données", bande=False, coupe_w=300, n_cols=1)''',
     "la page Données sans sa bande grise"),
    ("note_saf.py", '''            li += [("num", C.nb(c["R"][p_])), ("num", _q(-row[iq]) if row else "—")]''',
     '''            li += [("num", C.nb(c["R"][p_])), ("num", C.nb(c["R"][p_]))]''', "q = R au lieu de la valeur FF du SAF"),
    ("note_saf.py", '''                    parts.append(f"{lc} × {_n(f_, 2)}")''', '''                    parts.append(f"{lc}")''', "les combinaisons sans leurs coefficients"),
    ("note_saf.py", '''        verifs.append({"titre": "Tous les cas, pour chaque k coché « SCIA »", "items": items, "verdicts": []})''',
     '''        verifs.append({"titre": "Tous les cas, pour chaque k coché « SCIA »", "items": items[:1], "verdicts": []})''', "le tableau de tous les cas vide"),
    ("note_saf.py", '''TEXTE_MINI = 7.0''', '''TEXTE_MINI = 2.0''', "des dessins au texte trop petit pour être lu imprimé"),
    ("note_saf.py", '''        verifs.append({"titre": "Module / SCIA", "items": [("t", "Valeurs SCIA non saisies")], "verdicts": []})''',
     '''        verifs.append({"titre": "Module / SCIA", "items": [("t", "—")], "verdicts": []})''', None),
    ("note_saf.py", '''            items.append(("t", "Valeurs SCIA non saisies"))''', '''            pass''', "« Valeurs SCIA non saisies » tu"),
    ("note_saf.py", '''            octets = PDF.reunir([octets] + [a[1] for a in annexes], signets=signets, titre=nom_note(etat, infos), auteur="")''',
     '''            pass''', "les notes SCIA jointes oubliées"),
    ("note_saf.py", '''    p2 = {"titre": PARTIE_2, "etat": "", "couleur": style.acc,''', '''    p2 = {"titre": PARTIE_2, "etat": mot, "couleur": style.acc,''',
     "une pastille sur l'annexe aussi (deux pastilles sur la garde)"),
    ("note_saf.py", '''                besoin = min(max(besoin, 80.0), self.hauteur_colonne())''', '''                besoin = 0.0''',
     "un titre de chapitre laissé seul en bas d'une colonne"),
    ("note_saf.py", '''            besoin = max(58.0, S.title_gap + S.s_tit + self.hauteur_premier(fr, v))''', '''            besoin = 0.0''',
     "un titre laissé seul en bas d'une colonne"),
    ("dessin_efforts_saf.py", '''    for v, mot in traits:''', '''    for v, mot in ():''', "les diagrammes sans le treillis"),
    # (revue du 15/11/2026)
    ("dessin_efforts_saf.py", '''font-weight="600" text-anchor="{ancre}" '
            f'fill="#ffffff" stroke="#ffffff"''', '''text-anchor="{ancre}" '
            f'fill="none" stroke="#ffffff"''', "le liseré des diagrammes peint en noir (une ombre décalée)"),
    ("dessin_armatures.py", '''        if r["sens"] == "x":
            cx, cy = min(xs) + 0.22 * (max(xs) - min(xs)), sum(ys) / 4''', '''        if True:
            cx, cy = sum(xs) / 4, sum(ys) / 4''', "les repères x et y d'un même pied l'un sur l'autre"),
    ("dessin_saf.py", '''    carte_note = not interactif and (champ is not None or armatures is not None)''', '''    carte_note = False''',
     "la valeur pointée, le plot et les cotes plaque–bord écrits sur les cartes de la note"),
    ("dessin_saf.py", ''' + 1.8 * t / ECH''', ''' + 1.0 * t / ECH''', "« P4 » collé à la plaque, sous la cote 0,75"),
    ("note_saf.py", '''K_PAR_TABLEAU = 3 ''', '''K_PAR_TABLEAU = 9 ''', "cinq k dans un seul tableau (les en-têtes se chevauchent)"),
    ("note_saf.py", '''            if it[0] == "fig" and it[2] is None and any(''', '''            if False and it[0] == "fig" and it[2] is None and any(''',
     "un titre gardé avec la seule légende (son diagramme sur la page suivante)"),
    ("note_saf.py", '''        a_, b_ = math.dist(s0[0], s0[1]), math.dist(s0[1], s0[2])''',
     '''        a_, b_ = max(p[0] for p in s0) - min(p[0] for p in s0), max(p[1] for p in s0) - min(p[1] for p in s0)''', "la boîte d'un carré tourné prise pour son côté"),
    ("note_saf.py", '''{_n(ex[c] * ept[c], 1)} kN·m/m''', '''{_n(ex[c], 1)} kN·m/m''', "les valeurs sous les cartes sans le rapport SCIA"),
    ("note_saf.py", '''            chapitres.append(page_efforts(el, calc, ef, fer, F, style))''',
     '''            chapitres.append(page_efforts(el, calc, ef, fer, F, style)) if not fer.get("hors_domaine") else None''', "le chapitre Efforts retiré hors du moteur Dalles"),
    ("note_saf.py", '''            elif li["r"] is None:''', '''            elif False:''', "la note qui plante sur une valeur SCIA sans valeur du module"),
    ("note_saf.py", '''("eta", "à vérifier" if x["alerte"] else "concorde", not x["alerte"])''', '''("eta", "concorde", True)''', "un écart sur σ max dit « concorde »"),
    ("note_saf.py", '''        if sc.get("perime"):''', '''        if False:''', "des valeurs SCIA périmées dites « non saisies »"),
    ("note_saf.py", '''        a_, b_ = math.dist(s0[0], s0[1]), math.dist(s0[1], s0[2])''', '''        a_, b_ = math.dist(s0[0], s0[1]), math.dist(s0[1], s0[2])''', None),
    ("dessin_efforts_saf.py", '''    vp, vn = np.asarray(b["Vmax"], float), np.asarray(b["Vmin"], float)''',
     '''    vp, vn = np.asarray(b["Vabs"], float), np.zeros(len(b["s"]))''', "V(s) tracé en valeur absolue"),
    # (lot 6 : le calcul des bandes)
    ("saf_armatures.py", '''        ck = ident_bandes(sens) if fer.get("sections") else "a0"''', '''        ck = "a0"''', "les lignes des bandes qui n'ouvrent pas leur calcul"),
    ("saf.py", '''        volets += SA.fenetre(el, fer, ef) + SB.fenetre(el, fer, ef)''', '''        volets += SA.fenetre(el, fer, ef)''', "la fenêtre du calcul des bandes oubliée"),
    ("saf_bandes.py", '''            paires = [(g["nom"], max(ss, key=lambda s: s["taux_max"]))]''', '''            paires = [(s["bande"], s) for s in ss]''',
     "deux bandes pareilles écrites deux fois"),
    ("saf_bandes.py", '''                    li = dict(li, sous=f"V{N}à{N}d{N}={N}{C.fr(s['V'], 1)}{N}kN · sans étriers")''', '''                    pass''',
     "V à d écrit à la précision du calcul"),
    ("dessin_efforts_saf.py", '''            k = min(1.0, 1.0 if r["bord0"] else (x - a) / an, 1.0 if r["bord1"] else (z - x) / an)''', '''            k = 1.0''',
     "la couverture sans la rampe de l'ancrage"),
    ("note_saf.py", '''            if fer.get("sections"):
                chapitres.append(page_bandes(el, calc, ef, fer, F, style))''', '''            pass''', "le chapitre « Calcul des bandes » oublié"),
]
MUTATIONS = [m_ for m_ in MUTATIONS if m_[3] is not None]
TEMOIN = ("note_saf.py", 'TITRE = "Fondation de grue"', 'TITRE = "Fondation de grue"')


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
    t0 = time.time()
    rapides()
    print(f"   (contrôles rapides : {time.time() - t0:.1f} s)")
    if RAPIDE:
        print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
        return 1 if KO else 0
    complets()
    print("== 7. mutations")
    trouve, code = muter(*TEMOIN)
    chk("témoin : la note recopiée sans rien changer passe les contrôles rapides", trouve and code == 0, code)
    # (les mutations se jouent EN PARALLÈLE — chacune sur son arbre miroir ; MUTATIONS_PARALLELES, 3 par défaut)
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(os.environ.get("MUTATIONS_PARALLELES", "3")))) as pool:
        joues = list(pool.map(lambda m_: muter(*m_[:3]), MUTATIONS))
    for (fichier, ancien, nouveau, nom), (trouve, code) in zip(MUTATIONS, joues):
        chk(f"mutation « {nom} » ({fichier}) : un contrôle rougit", trouve and code == 1,
            "texte absent ou répété" if not trouve else f"code {code} : AUCUN contrôle ne rougit")
    print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
    for nom, info in KO:
        print("   -", nom[:160], "|", str(info)[:300])
    return 1 if KO else 0


if __name__ == "__main__":
    sys.exit(main())

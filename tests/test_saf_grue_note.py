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
    """Les pages (sauf la garde, la page Données et la dernière) dont une des deux colonnes n'a rien."""
    out = []
    for ip, p in enumerate(note.doc):
        if ip in (0, 1) or ip == len(note.doc) - 1:
            continue
        W = p.rect.width
        spans = [s for b in p.get_text("dict")["blocks"] for li in b.get("lines", []) for s in li["spans"] if s["text"].strip() and s["bbox"][1] > 80]
        dessins = [d_["rect"] for d_ in p.get_drawings() if d_["rect"].y0 > 80]
        gauche = any(s["bbox"][0] < W / 2 - 20 for s in spans) or any(r.x1 < W / 2 for r in dessins)
        droite = any(s["bbox"][0] > W / 2 for s in spans) or any(r.x0 > W / 2 for r in dessins)
        if not (gauche and droite):
            out.append(ip + 1)
    return out


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
    chk("NC6 les efforts : les diagrammes M(s) et V(s) des quatre bandes, le treillis du bas et du haut (m_Rd × 0,50 : 48,6 en x, 47,5 en y) en "
        "trait interrompu, les abscisses de V à d ; le tableau des bandes avec le cas et le k de chaque valeur (T14 : 3.b · k = 2…) ; deux cartes "
        "enveloppes des moments au point avec la ligne noire à m_Rd",
        all(f"M(s) [kN·m] · BANDE {b} DE 0,50 m" in t and f"V(s) [kN] · BANDE {b}" in t for b in ("BX1", "BX2", "BY1", "BY2"))
        and "treillis bas 48,6" in t and "treillis haut -48,6" in t and "treillis bas 47,5" in t and all(x in t for x in t14)
        and "EN BAS : LE PLUS GRAND DE mx ET my · LIGNE NOIRE : mRd = 95,0 kN·m/m" in t and "EN HAUT : LE PLUS NÉGATIF DE mx ET my" in t
        and "m haut min -140,8 kN·m/m" in t, [x for x in t14 if x not in t][:3])
    petits = [(nm, round(tt, 2)) for nm, tt in n.figures if tt < 7.0 - 1e-6]
    chk(f"§ 10.3 les dessins se lisent imprimés : les {len(n.figures)} dessins de la note (plan coté, carte de contrainte, 8 diagrammes, 2 cartes "
        "des moments, 2 plans du principe) ont un texte de 7 pt au moins une fois posés",
        len(n.figures) == 15 and not petits, petits or n.figures)
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
    ("note_saf.py", '''    sec = dict(F, section="Données", bande=True, coupe_w=300)''', '''    sec = dict(F, section="Données", bande=False, coupe_w=300)''',
     "la page Données sans sa bande grise"),
    ("note_saf.py", '''            li += [("num", C.nb(c["R"][p_])), ("num", _q(-row[iq]) if row else "—")]''',
     '''            li += [("num", C.nb(c["R"][p_])), ("num", C.nb(c["R"][p_]))]''', "q = R au lieu de la valeur FF du SAF"),
    ("note_saf.py", '''                    parts.append(f"{lc} × {_n(f_, 2)}")''', '''                    parts.append(f"{lc}")''', "les combinaisons sans leurs coefficients"),
    ("note_saf.py", '''        verifs.append({"titre": "Tous les cas, pour chaque k coché « SCIA »", "items": [''',
     '''        verifs.append({"titre": "Tous les cas, pour chaque k coché « SCIA »", "items": [] and [''', "le tableau de tous les cas vide"),
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
    ("dessin_efforts_saf.py", '''    for v, mot in ((tb, "treillis bas"), (th, "treillis haut")):''', '''    for v, mot in ():''', "les diagrammes sans le treillis"),
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

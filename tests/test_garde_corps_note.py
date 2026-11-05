# -*- coding: utf-8 -*-
"""Note de calcul « Garde-corps » (acier/garde_corps/note.py, version 3 : symboles composés,
formules par groupe, peu de prose). Chaque garantie rougit si on la retire :

  1. FORME : DEUX pages A4 paysage, aucun débordement de colonne, sur le cas par défaut et
     sept variantes (L avec lisses intermédiaires et main courante continue, U à entraxes
     saisis et fin au mur avec un plat et des boulons, murale, Eurocode soudé sur plat avec
     panneau et catégorie C5, scellé sans remplissage, cumul en mode P, section libre) ; moins
     de 2,5 s par note.
  2. PAGE 1, LES HYPOTHÈSES D'ABORD : bandeau (statut, η max, l'élément gouvernant, son
     critère et son cas de charge, la méthode choisie) ; le bloc HYPOTHÈSES en lignes
     « libellé · valeur » (garde-corps, hauteur, montants, main courante, barreaux, lisse basse,
     acier, méthode, limites, fixation) ; les QUATRE CHARGES — le symbole et la valeur sur la
     première ligne, le nom DESSOUS, l'origine en dessous (Tab. 6.12, NBN B 03-004, usage du
     bureau) — et la règle des cas DISTINCTS.
  3. LES SYMBOLES SONT COMPOSÉS : aucun « q_h », « W_el », « s_trib », « σ_adm » brut en page 1 ;
     le texte extrait les lit « qh », « Wel,y », « strib », « σadm » (indices posés en vrai).
  4. LE CALCUL ÉLÉMENT PAR ÉLÉMENT, dans l'ordre de lecture : 1 · MONTANT (pastille η, cas
     gouvernant et autres cas, données, puis CHARGES → SOLLICITATIONS → CONTRAINTES → FLÈCHE →
     TAUX, chaque taux sous sa contrainte), 2 · MAIN COURANTE, 3 · BARREAUX, 4 · ANCRAGE
     (soudure, ancrages — efforts, résistances et taux, platine en flexion) ; le résultat de
     chaque formule est CALÉ À DROITE de la colonne ; le RÉCAPITULATIF ; la CONCLUSION avec le
     cas gouvernant et la masse ; identification ; l'ancienne prose a disparu.
  5. DESSINS peints depuis les MÊMES scènes que l'écran : élévation (« M1 », « L = 3600 »),
     coupe A-A, plan (« L₁ ») en page 1 ; platine en plan et coupe sur la fixation en page 2 ;
     le cartouche porte l'échelle et les limites composées.
  6. PAGE 2 : DÉTAILS ET MÉTHODE — les montants un à un avec leur cas, la section et la
     BIBLIOTHÈQUE (le plus léger qui tient), la soudure (vue et formules du module Soudures),
     les sections de ``methode.py`` de la méthode choisie (un fragment de chaque, pas celui de
     l'autre méthode), les références, le cartouche.
  7. VARIANTES : Eurocode (γ_Q, f_y/γ_M0, efforts de calcul, soudé sur plat, fournisseur) ;
     murale (console C…, R_v = q_v·s_trib, M = R_v·e, cas 2) ; main courante continue (k_M) ;
     U (en U, 1,20 m + 0,90 m, fin ancrée dans le mur, angle, boulons M12, F_t,Rd/γ_F, virgules
     françaises dans la note du moteur) ; scellé (sous-titre sans « sur montants » doublé) ;
     cumul en mode P (cas 1 + 2, P = 1,2 kN) ; section libre (Caisson, C8).

Lancement : python3 tests/test_garde_corps_note.py (depuis la racine).
"""
import os
import sys
import time

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

import pymupdf  # noqa: E402

from acier.garde_corps import etat as ET  # noqa: E402
from acier.garde_corps import fixation as FX  # noqa: E402
from acier.garde_corps import methode as MT  # noqa: E402
from acier.garde_corps import note as NT  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{info}]" if info and not cond else ""))


def texte(pdf: bytes, page: int = 0) -> str:
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    return doc[page].get_text()


def mots(pdf: bytes, page: int = 0):
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    return doc[page].get_text("words")


def norm(s: str) -> str:
    return " ".join(s.replace(" ", " ").split())


def serre(s: str) -> str:
    return norm(s).replace(" ", "")


def plat(s: str) -> str:
    """Comme ``serre``, les indices aplatis : « q_h » du texte source se lit « qh » une fois composé."""
    return serre(s).replace("_", "")


def cas(nom="Garde-corps 1", type_="barreaudage", **kw):
    c = ET.nouveau_cas(nom, type_)
    for k, v in kw.items():
        c[k] = {**c[k], **v} if isinstance(v, dict) else v
    return c


ET_ = ET.nouvel_etat()
INFOS = {"nom_projet": "Résidence des Tilleuls", "partie": "Balcons", "indice": "B"}

# ================================================================
print("\n=== 1. Forme : deux pages, aucun débordement, douze cas ===")
CAS = [("défaut", cas()),
       ("L lisses continue", cas("L", "lisses", geometrie={"forme": "L"}, montant={"section": {"nom": "RHS 60x40x3"}}, lisse={"appuis": "continue"})),
       ("U plat boulons", cas("U", "barreaudage", geometrie={"forme": "U", "fin": "mur", "troncons": [{"L": 2100, "s_max": 1200, "entraxes": [1200, 900]}, {"L": 1500, "s_max": 1000, "entraxes": []},
                                                                                                      {"L": 2000, "s_max": 1000, "entraxes": []}]},
                              montant={"section": {"nom": "Plat 60x12"}}, fixation={"ancrage": {**FX.fixation_defaut()["ancrage"], "type": "boulon"}})),
       ("murale", cas("mur", "murale")),
       ("eurocode soudé C5", cas("EC", "panneau", methode={"mode": "eurocode"}, fixation={"type": "soudee"}, charges={"categorie": "C5"})),
       ("scellé", cas("scellé", "main_courante", fixation={"type": "scellee"})),
       ("cumul P", cas("P", charges={"mode": "P", "P": 1.2, "cumul": True})),
       ("section libre", cas("libre", montant={"section": {"famille": "CAISSON", "dims": {"h": 60, "b": 40, "t": 3}}})),
       ("latéral", cas("latéral", fixation={"type": "laterale"})),
       ("latéral L plat boulons", cas("latéral L", geometrie={"forme": "L"}, montant={"section": {"nom": "Plat 60x12"}},
                                      fixation={"type": "laterale", "platine": {"bp": 200.0, "lp": 160.0, "tp": 10.0, "d_h": 60.0},
                                                "ancrage": {**FX.fixation_defaut()["ancrage"], "type": "boulon", "ex": 140.0, "ey": 90.0}})),
       ("sans montant", cas("loggia", geometrie={"debut": "mur", "fin": "mur", "troncons": [{"L": 2500, "s_max": 3000, "entraxes": []}, {"L": 2400, "s_max": 1000, "entraxes": []}, {"L": 3600, "s_max": 1000, "entraxes": []}]})),
       ("rampant industriel", cas("escalier", geometrie={"pente": 32.0}, charges={"categorie": "ind"}, methode={"fleche_mode": "mm", "fleche_val": 30.0}))]
PDF = {}
for nom, c in CAS:
    t0 = time.time()
    pdf = NT.generer_note(ET_, c, INFOS)
    dt = time.time() - t0
    PDF[nom] = pdf
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    chk(f"{nom} : 2 pages A4 paysage, sans débordement (variante {NT.derniere_variante}), {dt:.1f} s",
        doc.page_count == 2 and all(abs(p.rect.width - 841.9) < 1 and abs(p.rect.height - 595.3) < 1 for p in doc) and not NT.derniers_avertissements and dt < (4.5 if nom == "défaut" else 2.5),
        f"pages {doc.page_count} ; avertissements {NT.derniers_avertissements} ; {dt:.1f} s")

# ================================================================
print("\n=== 2. Page 1 : les hypothèses d'abord ===")
p1 = norm(texte(PDF["défaut"], 0))
s1 = serre(texte(PDF["défaut"], 0))
chk("bandeau : CONFORME, η max 64 %, gouverne Montant principal M2 · RHS 50x30x3 · flèche · cas 1 · qh seule, « Vérification à l'ELS » (jamais « du bureau »)",
    "CONFORME" in s1 and "ηmax64%" in s1 and "Montant principal M2" in p1 and "RHS 50x30x3" in p1 and "flèche" in p1 and "cas 1 · qh seule" in p1 and "Vérification à l'ELS" in p1
    and "du bureau" not in p1.replace("usage du bureau", ""))
chk("titre : « Garde-corps — barreaudage vertical sur montants »", "barreaudage vertical sur montants" in p1)
chk("HYPOTHÈSES en tête de la première colonne, avant les charges et les dessins", "HYPOTHÈSES" in s1 and "CHARGES—" in s1 and "DESSINS—" in s1
    and s1.index("HYPOTHÈSES") < s1.index("CHARGES—") < s1.index("DESSINS—"))
chk("hypothèses en lignes « libellé · valeur » : GARDE-CORPS (droit, 3,60 m en 4 travées de 0,90 m, 5 montants), HAUTEUR H = 1100 mm, MONTANTS RHS 50x30x3",
    "GARDE-CORPSBarreaudagevertical·droit:3,60men4travéesde0,90m·5montants" in s1 and "HAUTEURH=1100mmau-dessusdelaplatine" in s1 and "MONTANTSRHS50x30x3" in s1)
chk("… MAIN COURANTE (portée 900 mm, sur deux appuis), BARREAUX (Plat 30x8, eb = 110 mm, portée 950 mm déduite de H, hbas), LISSE BASSE non calculée",
    "MAINCOURANTERHS50x30x3·portée900mm,surdeuxappuis(unetravée)" in s1 and "BARREAUXPlat30x8·eb=110mm·portée950mm,déduitedeH,hbas=100mmetdessections" in s1
    and "LISSEBASSERHS50x25x2.5(froid)(noncalculée)" in s1)
chk("… ACIER (S235, fy = 235, fu = 360, E = 210 000), VÉRIFICATION (à l'ELS, charges de service sans coefficient, σ ≤ σadm = 160, τ ≤ τadm = 90, flèche ≤ L/100, k = 100, γF = 1,5) — pas de ligne « méthode »",
    "ACIERS235·fy=235N/mm²·fu=360N/mm²·E=210000N/mm²" in s1 and "MÉTHODE" not in s1
    and "VÉRIFICATIONàl'ELS,chargesdeservicesanscoefficient:σ≤σadm=160N/mm²·τ≤τadm=90N/mm²·flèche≤L/100·soudurek=100kg/(mm·cm)·ancrages:résistancesdecalcul/γF=1,5" in s1)
chk("… FIXATION : platine boulonnée, cordon a = 4 mm, platine 150 × 150 × 10, 2 × 2 chevilles M12, ex = 100, ey = 100",
    "FIXATIONplatineboulonnée·soudurecordond'anglea=4mm·platine150×150×10mm·2×2chevillesM12(NRd9kN,VRd12kN),rangéesàex=100mm,ey=100mm" in s1)
chk("tableau des QUATRE CHARGES : le symbole et la valeur sur la première ligne, le nom DESSOUS, l'origine en dessous",
    "qh 0,50 kN/m charge linéaire horizontale sur la main courante A · habitation — Tab. 6.12 EN 1991-1-1, valeur recommandée" in p1
    and "qv 0,50 kN/m charge linéaire verticale sur la main courante usage du bureau (NBN B 03-004 : Qkv, à confirmer)" in p1
    and "Qr 0,50 kN charge ponctuelle horizontale sur le remplissage NBN B 03-004 : Qkh,2" in p1
    and "qr 0,50 kN/m² pression horizontale répartie sur le remplissage usage du bureau" in p1 and "quatrecasdecharge" in s1)
chk("la règle des cas : les quatre charges agissent SÉPARÉMENT, chaque élément est vérifié sous chaque cas, le pire gouverne",
    "agissent SÉPARÉMENT" in p1 and "le pire gouverne" in p1)

# ================================================================
print("\n=== 3. Les symboles sont composés ===")
def spans(pdf: bytes, page: int, taille_min: float):
    """Les textes de la page composés à ``taille_min`` points ou plus (les cotes des dessins sont plus petites)."""
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    return [sp["text"] for b in doc[page].get_text("dict")["blocks"] for l in b.get("lines", []) for sp in l["spans"] if sp["size"] >= taille_min]


p2 = norm(texte(PDF["défaut"], 1))
s2 = serre(texte(PDF["défaut"], 1))
bruts1 = [t for t in spans(PDF["défaut"], 0, 5.0) if "_" in t]
chk("aucun symbole brut (« q_h », « W_el », « s_trib », « σ_adm », « e_x »…) dans le texte de la page 1 — seules les cotes des dessins, plus petites, gardent leur notation",
    not bruts1, str(bruts1)[:200])
chk("… ni dans la prose de méthode de la page 2 : « M/W_el », « ±e_x/2 », « b_eff·t_p² », « A_v », « N_t » sont composés jusque dans les mots",
    all(k not in p2 for k in ("W_el", "e_x/2", "b_eff", "A_v", "N_t", "l_p ;", "γ_F", "h_p/2")))
chk("les indices sont posés en vrai : « P1 = qh·strib », « σ = M1/Wel,y », « ηf = f/fadm »", "P1 = qh·strib" in p1 and "σ = M1/Wel,y" in p1 and "ηf = f/fadm" in p1)

# ================================================================
print("\n=== 4. Page 1 : le calcul élément par élément, dans l'ordre de lecture ===")
chk("1 · MONTANT M2 — RHS 50x30x3, pastille η 64 %, « Gouverne : cas 1 · qh seule · autres cas : cas 2 · qv seule (1 %) », données H et strib",
    "η64%1·MONTANTM2—RHS50x30x3" in s1 and "Gouverne : cas 1 · qh seule · autres cas : cas 2 · qv seule (1 %)" in p1 and "H = 1100 mm · strib = 900 mm · console encastrée en pied" in p1)
i_m = s1.index("1·MONTANTM2")
i_l = s1.index("2·MAINCOURANTE")
bloc_m = s1[i_m:i_l]
chk("… ses groupes dans l'ordre CHARGES → SOLLICITATIONS → CONTRAINTES → FLÈCHE → TAUX",
    all(k in bloc_m for k in ("CHARGES", "SOLLICITATIONS", "CONTRAINTES", "FLÈCHE", "TAUX"))
    and bloc_m.index("CHARGES") < bloc_m.index("SOLLICITATIONS") < bloc_m.index("CONTRAINTES") < bloc_m.index("FLÈCHE") < bloc_m.index("TAUX"))
chk("… formules développées : P1 = qh·strib = 0,5 × 900/10³ = 0,45 kN, M1 = P1·H = 0,495 kN·m, σ = 91,2 N/mm², f = P1·H³/(3·E·Iy) = 7,01 mm, η = max(ησ, ητ, ηf) = 63,7 %",
    "P1 = qh·strib = 0,5 × 900/103 = 0,45 kN" in p1 and "M1 = P1·H = 0,45 × 1100/103 = 0,495 kN·m" in p1 and "σ = M1/Wel,y = 0,495 × 106/5425 = 91,2 N/mm²" in p1
    and "f = P1·H3/ 3·E·Iy = 0,45 × 103 × 11003/ 3 × 210000 × 135622 = 7,01 mm" in p1 and "η = max ησ, ητ, ηf = max 0,5703, 0,0184, 0,6373 = 63,7 %" in p1)
chk("… chaque taux SOUS sa contrainte : σ, ησ = σ/σadm = 91,2/160, τ, ητ, puis f, fadm, ηf, puis η",
    bloc_m.index("σ=M1/Wel,y") < bloc_m.index("ησ=σ/σadm=91,2/160=57,0%") < bloc_m.index("τ=V/Av") < bloc_m.index("ητ=τ/τadm") < bloc_m.index("f=P1·H3")
    < bloc_m.index("fadm=L/x=1100/100=11mm") < bloc_m.index("ηf=f/fadm=7,01/11=63,7%") < bloc_m.index("η=maxησ,ητ,ηf"))
w1 = mots(PDF["défaut"], 0)
y_p1 = next((w[1] for w in w1 if w[4] == "P1"), None)
x_res = max((w[2] for w in w1 if y_p1 is not None and abs(w[1] - y_p1) < 2.5 and w[4] == "kN"), default=0.0)
y_eta = next((w[1] for w in w1 if w[4] == "ησ"), None)
x_eta = max((w[2] for w in w1 if y_eta is not None and abs(w[1] - y_eta) < 2.5 and w[4] == "%"), default=0.0)
chk("… la valeur « = 0,45 kN » suit le calcul sur la ligne (x < 480 pt) ; seul le taux « = 57,0 % » est calé à droite de la colonne 2 (x ≥ 555 pt)",
    y_p1 is not None and 0 < x_res < 480.0 and y_eta is not None and x_eta >= 555.0, f"x1 valeur = {x_res:.0f}, x1 taux = {x_eta:.0f}")
chk("2 · MAIN COURANTE — RHS 50x30x3, η 8 %, L = 900 mm sur deux appuis, Mh = qh·L²/8, σ = Mh/Wel,z, f = 5·qh·L⁴/(384·E·Iz)",
    "η8%2·MAINCOURANTE—RHS50x30x3" in s1 and "L = 900 mm (l'entraxe le plus grand) · sur deux appuis (une travée)" in p1 and "Mh = qh·L2/8 = 0,5 × 9002/8/106 = 0,051 kN·m" in p1
    and "σ = Mh/Wel,z" in p1 and "L4/ 384·E·Iz" in p1)
chk("3 · BARREAUX — Plat 30x8, η 62 %, L = 950 mm, eb = 110 mm, M = Qr·L/4, f = Qr·L³/(48·E·I)", "η62%3·BARREAUX—Plat30x8" in s1 and "L = 950 mm · eb = 110 mm · sur deux appuis" in p1
    and "M = Qr·L/4 = 0,5 × 950/4/103 = 0,1187 kN·m" in p1 and "L3/ 48·E·I" in p1)
chk("4 · ANCRAGE — platine boulonnée, η 59 %, le cas et les efforts en pied (virgules françaises), SOUDURE — C6 du module Soudures, η = |f|/Fadm = 59,4 %",
    "η59%4·ANCRAGE—platineboulonnée" in s1 and "Gouverne : cas 1 · qh seule · en pied du montant : N = 0,00 kN, V = 0,45 kN, M = 0,495 kN·m (efforts de service)" in p1
    and "SOUDURE—C6DUMODULESOUDURES" in s1 and "η = |f|/Fadm = 233,2/392,4 = 59,4 %" in p1)
i_a = s1.index("4·ANCRAGE")
bloc_a = s1[i_a:]
chk("… puis ANCRAGES — EFFORTS (z, Z, Ft = Z/n, Fv), ANCRAGES — RÉSISTANCES ET TAUX (NRd/γF, ηinter, ηt), PLATINE EN FLEXION (m, beff, Mp, σp, treq, ηp), dans cet ordre",
    bloc_a.index("SOUDURE—") < bloc_a.index("ANCRAGES—EFFORTS") < bloc_a.index("ANCRAGES—RÉSISTANCESETTAUX") < bloc_a.index("PLATINEENFLEXION")
    and all(k in p1 for k in ("z = ex/2 + hp/2 = 100/2 + 50/2 = 75 mm", "Ft = Z/n = 6,6/2 = 3,3 kN", "NRd/γF = 6 kN", "σp = 6·Mp/ beff·t2p = 6 × 0,1351 × 106/ 150 × 102 = 54,1 N/mm²", "treq = 6·Mp/ beff·σlim")))
chk("l'ordre 1, 2, 3, 4 est celui de la lecture", s1.index("1·MONTANT") < s1.index("2·MAINCOURANTE") < s1.index("3·BARREAUX") < s1.index("4·ANCRAGE"))
chk("RÉCAPITULATIF : montant, main courante, barreaux, soudure, chevilles (sans la parenthèse des résistances), platine, chacun avec son cas", "RÉCAPITULATIF" in s1
    and all(k in p1 for k in ("Main courante · RHS 50x30x3", "Barreaux · Plat 30x8", "Soudure · cordon d'angle a = 4 mm", "Ancrages · 2 × 2 chevilles M12", "Platine 150 × 150 × 10 · flexion"))
    and "Chevilles / boulons" not in p1 and p1.count("qh seule") >= 7 and "3 · Qr seule" in p1)
chk("CONCLUSION : taux maximal 64 %, le cas gouvernant, porte les charges, masse indicative", "CONCLUSION" in s1 and "Taux maximal 64 %" in p1 and "cas 1 · qh seule" in p1
    and "porte les charges" in p1 and "Masse indicative" in p1)
chk("l'ancienne prose a disparu : plus de « Autres cas : », de phrase « Console encastrée en pied, hauteur », de paragraphe de notations ni de « Platine rigide »",
    "Autres cas :" not in p1 and "Console encastrée en pied, hauteur" not in p1 and "traction de la rangée de chevilles tendue" not in p1 and "Platine rigide" not in p1 and "CRITÈRES" not in s1)
chk("identification : projet, partie, indice B", "Résidence des Tilleuls" in p1 and "Balcons" in p1 and "indice B" in p1)

# ================================================================
print("\n=== 5. Dessins peints ===")
chk("page 1 : élévation (M1, L = 3600, H = 1100), coupe A-A, vue en plan (L₁)", all(k in p1 for k in ("M1", "L = 3600", "H = 1100", "Coupe A-A", "Vue en plan")) and "L₁" in p1)
chk("page 2 : platine en plan (e_x = 100, l_p = 150, b_p = 150) et coupe sur la fixation (a 4, t_p 10, h_p 50)",
    all(k in p2 for k in ("e_x = 100", "l_p = 150", "b_p = 150", "a 4", "t_p 10", "h_p 50")))
chk("le cartouche porte l'échelle de l'élévation (1:n), la date, l'indice, le type, la méthode et ses limites composées (σadm 160 · τadm 90, fy 235)",
    "1:" in p2 and "indice B" in p2 and "Barreaudage vertical" in p2 and "ÉCHELLEDEL'ÉLÉVATION" in s2 and "σadm 160 · τadm 90" in p2 and "fy 235 N/mm² · fu 360" in p2)

# ================================================================
print("\n=== 6. Page 2 : détails et méthode ===")
chk("titre DÉTAILS ET MÉTHODE, références", "DÉTAILSETMÉTHODE" in s2 and "Références" in p2 and "EN 1991-1-1" in p2)
chk("MONTANTS UN À UN : M1 à M5, strib 450 / 900, P 0,23 / 0,45, cas 1 · qh seule, η 32 % / 64 %", "MONTANTSUNÀUN" in s2 and "M5" in p2 and "450" in p2 and "0,45" in p2
    and "32 %" in p2 and "64 %" in p2 and "1 · qh seule" in p2)
chk("SECTION ET BIBLIOTHÈQUE : le plus léger qui tient (CHS 48.3x2.5), la section retenue, kg/m", "BIBLIOTHÈQUE" in s2 and "CHS 48.3x2.5" in p2 and "le plus léger qui tient" in p2
    and "RHS 50x30x3" in p2 and "kg/m" in p2)
chk("la soudure : vue peinte (« RHS 50x30x3 sur platine ») et formules du module Soudures par groupe (GROUPE DE CORDONS, POINT GOUVERNANT, CRITÈRE : Aw, |f|, τadm = k·g/10)",
    "SOUDURESURLAPLATINE" in s2 and "sur platine" in p2 and s2.index("GROUPEDECORDONS") < s2.index("POINTGOUVERNANT") < s2.index("CRITÈRE")
    and "Aw = 2·a·leff" in p2 and "|f|" in p2 and "τadm = k·g/10" in p2)
els = MT.sections("els", "platine", False)
chk("les sections du cas par défaut : Modèle, Vérification à l'ELS, Ancrage, Hors périmètre — ni Eurocode ni Ancrage latéral",
    [s["titre"] for s in els] == ["Modèle", "Vérification à l'ELS", "Ancrage", "Hors périmètre"], str([s["titre"] for s in els]))
for sec in els:
    frag = plat(sec["paragraphes"][0][:70])
    chk(f"section « {sec['titre']} » : un fragment mot pour mot (symboles composés)", frag in plat(p2), frag[:50])
frag_ec = plat(MT.SECTIONS[2]["paragraphes"][0][:70])
chk("la section Eurocode n'est PAS imprimée en méthode ELS", frag_ec not in plat(p2))
chk("ni la section « Ancrage latéral »", plat([s for s in MT.SECTIONS if s["titre"] == "Ancrage latéral"][0]["paragraphes"][0][:70]) not in plat(p2))

# ================================================================
print("\n=== 7. Variantes ===")
pE = norm(texte(PDF["eurocode soudé C5"], 0)) + norm(texte(PDF["eurocode soudé C5"], 1))
chk("Eurocode : « vérification à l'ELU selon l'Eurocode », γQ, fy/γM0, catégorie C5 à 3 kN/m, soudé sur plat, panneau à justifier par le fournisseur, pas de 4 · ANCRAGE chevilles",
    "à l'ELU selon l'Eurocode" in pE and "γQ" in pE and "fy/γM0" in pE and "3,00 kN/m" in pE and "soudé sur un plat" in pE and "fournisseur" in pE and "3·PANNEAU" in serre(pE)
    and "Ft = Z/n = " not in pE and "σlim = fy/γM0 = 235/1 = 235 N/mm²" in pE)
chk("Eurocode : la section « Méthode ELS » n'est pas imprimée, la section « Eurocode » l'est", plat(MT.SECTIONS[1]["paragraphes"][0][:70]) not in plat(pE)
    and plat(MT.SECTIONS[2]["paragraphes"][0][:70]) in plat(pE))
pM = norm(texte(PDF["murale"], 0))
chk("murale : titre « main courante murale sur consoles », 1 · CONSOLE C…, Rv = qv·strib, M = Rv·e, cas 2 · qv seule, 2 · MAIN COURANTE",
    "main courante murale sur consoles" in pM and "1·CONSOLEC" in serre(pM) and "Rv = qv·strib" in pM and "M = Rv·e" in pM and "cas 2 · qv seule" in pM and "2·MAINCOURANTE" in serre(pM))
pL = norm(texte(PDF["L lisses continue"], 0))
chk("L, lisses, main courante continue : « en L », angle, 3 · LISSES INTERMÉDIAIRES, Mh = kM·qh·L², kM = 0,107, continue sur plusieurs appuis",
    "en L" in pL and "angle" in pL and "3·LISSESINTERMÉDIAIRES" in serre(pL) and "Mh = kM·qh·L2 = 0,107" in pL and "kM = 0,107" in pL and "continue sur plusieurs appuis" in pL)
pU = norm(texte(PDF["U plat boulons"], 0))
chk("U à entraxes saisis : « en U », travées de 1,20 m + 0,90 m, fin ancrée dans le mur, montant d'angle, boulons M12 8.8, Ft,Rd/γF, NON CONFORME, la note du moteur en virgules",
    "en U" in pU and "1,20 m + 0,90 m" in pU and "fin ancrée dans le mur" in pU and "angle" in pU and "M12" in pU and "Ft,Rd/γF" in pU and "NONCONFORME" in serre(pU)
    and "il reprend H = 0,25 kN et V = 0,25 kN" in pU)
pS = norm(texte(PDF["scellé"], 0))
chk("scellé, main courante seule : sous-titre « main courante seule sur montants » (pas « sur montants sur montants »), la fixation n'est pas vérifiée, sans remplissage, pas de 3 · ni de Ft",
    "main courante seule sur montants" in pS and "sur montants sur montants" not in pS and "n'est pas vérifiée" in pS and "sans remplissage" in pS
    and "3·BARREAUX" not in serre(pS) and "Ft = Z/n" not in pS)
pP = norm(texte(PDF["cumul P"], 0))
chk("cumul en mode P : P = 1,20 kN (saisie directe), cas 1 + 2 · qh et qv cumulées, σ = M1/Wel,y + N/A", "1,20 kN" in pP and "saisie directe" in pP
    and "cas 1 + 2 · qh et qv cumulées" in pP and "σ = M1/Wel,y + N/A" in pP)
pLib = norm(texte(PDF["section libre"], 0)) + norm(texte(PDF["section libre"], 1))
chk("section libre : « Caisson 60x40x3 », soudure C8", "Caisson 60x40x3" in pLib and "C8" in pLib)
pLat, sLat = norm(texte(PDF["latéral"], 0)), serre(texte(PDF["latéral"], 0))
chk("platine latérale : hypothèses (H au-dessus de la dalle, descente dh = 50, console de 1150 ; platine latérale 150 × 150 × 10, soudure le long de la platine, "
    "rangées à ex vertical, colonnes à ey), montant sur H = 1150, « 4 · ANCRAGE — platine latérale en nez de dalle », arrachement P, z = ex = 100 mm, Z = M/z + P/2, "
    "m = ey/2 − b/2 − …, tp = 10 mm, « Platine latérale 150 × 150 × 10 » au récapitulatif",
    "descente dh = 50 mm" in pLat and "console de 1150 mm" in pLat and "platine latérale 150 × 150 × 10 mm" in pLat and "le long de la platine" in pLat
    and "(vertical), colonnes à ey = 100 mm" in pLat and "H = 1150 mm (1100 au-dessus de la dalle + descente dh = 50)" in pLat
    and "4·ANCRAGE—platinelatéraleennezdedalle" in sLat and "P = 0,45 kN (arrachement)" in pLat and "z = ex = 100 mm" in pLat and "Z = M/z + P/2" in pLat
    and "m=ey/2−b/2−0,8·a·2" in sLat and "tp = 10 mm" in pLat and "Platine latérale 150 × 150 × 10 · flexion" in pLat
    and "Ancrages · 2 × 2 chevilles M12" in pLat)
pLat2, sLat2 = norm(texte(PDF["latéral"], 1)), serre(texte(PDF["latéral"], 1))
chk("… page 2 : « DÉTAIL DE L'ANCRAGE — platine latérale : vue de face, coupe horizontale », les deux dessins (« 4 chevilles M12 · vue de face », « Coupe horizontale sur la fixation »), "
    "la soudure C8 par ses formules (deux cordons de 142 : Aw = 2·a·leff) et pas de dessin de soudure C8 ; la méthode « Ancrage latéral » (z = ex, Z = M/z + P/2) "
    "à la place de « Ancrage » (z = ex/2 + hp/2)",
    "DÉTAILDEL'ANCRAGE—platinelatérale:vuedeface,coupehorizontale" in sLat2 and "4 chevilles M12 · vue de face" in pLat2 and "Coupe horizontale sur la fixation" in pLat2
    and "Aw = 2·a·leff = 2 × 4 × 142" in pLat2 and "cordon gauche" not in pLat2
    and "ANCRAGELATÉRAL" in sLat2 and "z = ex, Z = M/z + P/2" in pLat2 and "z = ex/2 + hp/2" not in pLat2)
pSM, sSM = norm(texte(PDF["sans montant"], 0)), serre(texte(PDF["sans montant"], 0))
chk("sans montant : « aucun : la main courante porte de mur à mur », 1 · MAIN COURANTE porteuse sur L = 2500, 2 · BARREAUX, 3 · ANCRAGE — main courante ancrée dans les murs "
    "avec les réactions H et V de chaque mur, conclusion « ancrages aux murs à justifier »",
    "aucun : la main courante porte de mur à mur" in pSM and "1·MAINCOURANTE—RHS50x30x3" in sSM and "L = 2500 mm de mur à mur" in pSM and "2·BARREAUX" in sSM
    and "3·ANCRAGE—maincouranteancréedanslesmurs" in sSM and "mur de début : H = 0,62 kN" in pSM and "mur de fin : H = 0,62 kN" in pSM
    and "V = 0,62 kN (vertical" in pSM and "ancrages aux murs à justifier" in pSM and "1·MONTANT" not in sSM)
pSM2 = norm(texte(PDF["sans montant"], 1))
chk("sans montant : titre « barreaudage vertical de mur à mur » (pas « sur montants »), pas de ligne NOTE qui répète les réactions des murs, "
    "page 2 sans section « Ancrage » (rien n'est fixé)",
    "Garde-corps — barreaudage vertical de mur à mur" in pSM and "sur montants" not in pSM and pSM.count("Ancrage de la main courante dans le mur") == 0
    and "Platine rigide" not in pSM2 and "ANCRAGE" not in serre(texte(PDF["sans montant"], 1)).replace("DÉTAILDEL'ANCRAGE", ""))
pRp, sRp = norm(texte(PDF["rampant industriel"], 0)), serre(texte(PDF["rampant industriel"], 0))
chk("rampant industriel : « rampant, pente 32° (entraxes mesurés suivant la pente) », qh = 0,30 kN/m d'origine EN ISO 14122-3 à confirmer, flèche ≤ 30 mm (fadm = 30 mm), « Élévation · rampant »",
    "rampant, pente 32° (entraxes mesurés suivant la pente)" in pRp and "qh 0,30 kN/m charge linéaire horizontale sur la main courante plateforme industrielle (machines) — EN ISO 14122-3, 0,3 kN/m à confirmer" in pRp
    and "flèche≤30mm" in sRp and "fadm = 30 mm" in pRp and "Élévation · rampant" in pRp)

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

# -*- coding: utf-8 -*-
"""Note de calcul « Garde-corps » (acier/garde_corps/note.py). Chaque garantie rougit si
on la retire :

  1. FORME : DEUX pages A4 paysage, aucun débordement de colonne, sur le cas par défaut et
     sept variantes (L avec lisses intermédiaires, U à entraxes saisis et fin au mur avec un
     plat et des boulons, système mur, Eurocode soudé sur plat avec panneau et catégorie C5,
     scellé, mode P, section libre) ; moins de 2 s par note.
  2. CONTENU p.1 : statut, η max, l'élément gouvernant et son critère, la méthode choisie,
     la ligne de géométrie (forme, travées, montants, sections), les charges avec leur
     origine (Tab. 6.12, usage du bureau, NBN B 03-004), la méthode (σ_adm, τ_adm, flèche),
     tableau des CHARGES, des MONTANTS (abscisse, s_trib, P, η), des ÉLÉMENTS, la
     VÉRIFICATION DU MONTANT comme à la main (P₁ = q_h·s_trib, M₁ = P₁·H, σ, f, η), la
     FIXATION (soudure η, Z, F_t, m, σ_p, t_req), la CONCLUSION et la masse.
  3. DESSINS peints depuis les MÊMES scènes que l'écran : « M1 », « L = 3600 », « b_p 150 »
     dans le texte du PDF ; le cartouche porte l'échelle.
  4. PAGE 2 : MÉTHODE ET DÉTAIL, les sections de ``methode.py`` de la méthode choisie (un
     fragment de chaque section mot pour mot, pas celui de l'autre méthode), la soudure (vue
     et formules du module Soudures), le détail de la lisse et du remplissage, la
     BIBLIOTHÈQUE (le plus léger qui tient), les références, le cartouche.
  5. EUROCODE : « γ_Q » et « f_y/γ_M0 » dans la note, « efforts de calcul » ; système mur :
     « console », « R_v », « N_t ».

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


def norm(s: str) -> str:
    return " ".join(s.replace(" ", " ").split())


def serre(s: str) -> str:
    return norm(s).replace(" ", "")


def cas(nom="Garde-corps 1", **kw):
    c = ET.nouveau_cas(nom)
    for k, v in kw.items():
        c[k] = {**c[k], **v} if isinstance(v, dict) else v
    return c


ET_ = ET.nouvel_etat()
INFOS = {"nom_projet": "Résidence des Tilleuls", "partie": "Balcons", "indice": "B"}

# ================================================================
print("\n=== 1. Forme : deux pages, aucun débordement, huit cas ===")
CAS = [("défaut", cas()),
       ("L lisses", cas("L", geometrie={"forme": "L"}, montant={"section": {"nom": "RHS 60x40x3"}}, remplissage={"type": "lisses"})),
       ("U plat boulons", cas("U", geometrie={"forme": "U", "fin": "mur", "troncons": [{"L": 2100, "s_max": 1200, "entraxes": [1200, 900]}, {"L": 1500, "s_max": 1000, "entraxes": []},
                                                                                        {"L": 2000, "s_max": 1000, "entraxes": []}]},
                              montant={"section": {"nom": "Plat 60x12"}}, fixation={"ancrage": {**FX.fixation_defaut()["ancrage"], "type": "boulon"}})),
       ("mur", cas("mur", geometrie={"systeme": "mur"}, remplissage={"type": "aucun"})),
       ("eurocode soudé C5", cas("EC", methode={"mode": "eurocode"}, fixation={"type": "soudee"}, remplissage={"type": "panneau"}, charges={"categorie": "C5"})),
       ("scellé", cas("scellé", fixation={"type": "scellee"})),
       ("mode P", cas("P", charges={"mode": "P", "P": 1.2})),
       ("section libre", cas("libre", montant={"section": {"famille": "CAISSON", "dims": {"h": 60, "b": 40, "t": 3}}}))]
PDF = {}
for nom, c in CAS:
    t0 = time.time()
    pdf = NT.generer_note(ET_, c, INFOS)
    dt = time.time() - t0
    PDF[nom] = pdf
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    chk(f"{nom} : 2 pages A4 paysage, sans débordement (variante {NT.derniere_variante}), {dt:.1f} s",
        doc.page_count == 2 and all(abs(p.rect.width - 841.9) < 1 and abs(p.rect.height - 595.3) < 1 for p in doc) and not NT.derniers_avertissements and dt < 2.5,
        f"pages {doc.page_count} ; avertissements {NT.derniers_avertissements} ; {dt:.1f} s")

# ================================================================
print("\n=== 2. Contenu de la page 1 ===")
p1 = norm(texte(PDF["défaut"], 0))
s1 = serre(texte(PDF["défaut"], 0))
chk("bandeau : CONFORME, η max 75 %, gouverne Montant M2 · CHS 48.3x3.2 · flèche, Méthode ELS du bureau",
    "CONFORME" in s1 and "ηmax75%" in s1 and "Montant M2" in p1 and "CHS 48.3x3.2" in p1 and "flèche" in p1 and "Méthode ELS du bureau" in p1)
chk("ligne de géométrie : droit, 3,60 m en 4 travées de 0,90 m, 5 montants, H = 1100, lisse haute CHS 42.4x3.2, barreaux Plat 30x8 à 110 mm",
    "3,60 m en 4 travées de 0,90 m" in p1 and "5 montants" in p1 and "H = 1100 mm" in p1 and "CHS 42.4x3.2" in p1 and "Plat 30x8 à 110 mm" in p1)
chk("les charges et leur origine : q_h = 0,5 kN/m catégorie A, Tab. 6.12 valeur recommandée, ANB, q_v, remplissage q_r et Q_r",
    "q_h = 0,5 kN/m" in p1 and "catégorie A" in p1 and "Tab. 6.12" in p1 and "ANB" in p1 and "q_v = 0,5 kN/m" in p1 and "Q_r = 0,5 kN" in p1)
chk("la méthode : σ_adm = 160, τ_adm = 90, flèche ≤ L/100, acier S235, k = 100", "σ_adm = 160 N/mm²" in p1 and "τ_adm = 90 N/mm²" in p1 and "L/100" in p1 and "S235" in p1 and "k = 100" in p1)
chk("tableau CHARGES avec les origines (habitation, usage du bureau, NBN B 03-004)", "CHARGES" in s1 and "habitation" in p1 and "usage du bureau" in p1 and "NBN B 03-004" in p1)
chk("tableau MONTANTS : M1 à M5, s_trib 450 / 900, P 0,23 / 0,45, η 37 % / 75 %", "MONTANTS" in s1 and "M5" in p1 and "450" in p1 and "0,45" in p1 and "37 %" in p1 and "75 %" in p1)
chk("tableau ÉLÉMENTS : montant, lisse haute, barreaux, soudure, boulons / chevilles, platine", all(k in p1 for k in ("Lisse haute", "Barreaux", "Soudure", "Boulons / chevilles", "Platine")))
chk("VÉRIFICATION DU MONTANT comme à la main : P₁ = q_h·s_trib, M₁ = P₁·H, σ = M/W_el + N/A = … = 104,2 N/mm², f = … = 8,21 mm, η = max",
    "VÉRIFICATIONDUMONTANTM2" in s1 and "P₁ = q_h·s_trib" in p1 and "M₁ = P₁·H" in p1 and "104,2 N/mm²" in p1 and "8,21 mm" in p1 and "η = max(η_σ, η_τ, η_f)" in p1)
chk("FIXATION : soudure η = 71 %, Z, F_t = Z/n, N_Rd/γ_F, m, σ_p, t_req", "FIXATION" in s1 and "η = 71 %" in p1 and "Z = max(0" in p1 and "F_t = Z/n" in p1 and "N_Rd/γ_F" in p1
    and "σ_p = 6·M_p/(b_eff·t_p²)" in p1 and "t_req" in p1)
chk("CONCLUSION : taux maximal 75 %, porte les charges, masse indicative", "CONCLUSION" in s1 and "Taux maximal 75 %" in p1 and "porte les charges" in p1 and "Masse indicative" in p1)
chk("identification : projet, partie, indice B, date", "Résidence des Tilleuls" in p1 and "Balcons" in p1 and "indice B" in p1)

# ================================================================
print("\n=== 3. Dessins peints ===")
chk("les textes des scènes sont dans le PDF : M1, L = 3600, H = 1100, b_p = 150, e_x = 100, q_h = 0,50 kN/m ⊙",
    all(k in p1 for k in ("M1", "L = 3600", "H = 1100", "b_p = 150", "e_x = 100", "q_h = 0,50 kN/m")) and "⊙" in p1)
p2 = norm(texte(PDF["défaut"], 1))
s2 = serre(texte(PDF["défaut"], 1))
chk("le cartouche porte l'échelle de l'élévation (1:n), la date, l'indice, le garde-corps, la méthode", "1:" in p2 and "indice B" in p2 and "montants verticaux sur platine" in p2
    and "ÉCHELLEDEL'ÉLÉVATION" in s2 and "σ_adm 160" in p2)

# ================================================================
print("\n=== 4. Page 2 : méthode et détail ===")
chk("titre MÉTHODE ET DÉTAIL, références", "MÉTHODEETDÉTAIL" in s2 and "Références" in p2 and "EN 1991-1-1" in p2)
els = [s for s in MT.SECTIONS if not s["titre"].startswith("Eurocode")]
for sec in els:
    frag = serre(sec["paragraphes"][0][:70])
    chk(f"section « {sec['titre']} » : un fragment mot pour mot", frag in s2, frag[:50])
frag_ec = serre(MT.SECTIONS[3]["paragraphes"][0][:70])
chk("la section Eurocode n'est PAS imprimée en méthode ELS", frag_ec not in s2)
chk("la soudure : vue peinte (texte de la scène « CHS 48.3x3.2 sur platine ») et formules du module Soudures (A_w, |f|, τ_adm, η)",
    "SOUDURESURLAPLATINE" in s2 and "sur platine" in p2 and "A_w" in p2 and "|f|" in p2 and "τ_adm = k·g/10" in p2)
chk("détail des autres éléments : la lisse (M_h = q_h·L²/8), le remplissage (M = Q_r·L/4)", "M_h = q_h·L²/8" in p2 and "M = Q_r·L/4" in p2)
chk("BIBLIOTHÈQUE : le plus léger qui tient (CHS 48.3x2.5), la section retenue en gras", "BIBLIOTHÈQUE" in s2 and "CHS 48.3x2.5" in p2 and "Le plus léger qui tient" in p2)
chk("la section de l'élément principal est dessinée (Ø 48,3 · kg/m)", "SECTIONDEL'ÉLÉMENTPRINCIPAL" in s2 and "kg/m" in p2)

# ================================================================
print("\n=== 5. Variantes ===")
pE = norm(texte(PDF["eurocode soudé C5"], 0)) + norm(texte(PDF["eurocode soudé C5"], 1))
chk("Eurocode : efforts de calcul, γ_Q, f_y/γ_M0, catégorie C5 à 3 kN/m, soudé sur plat, panneau à justifier par le fournisseur",
    "efforts de calcul" in pE and "γ_Q" in pE and "f_y/γ_M0" in pE and "q_h = 3 kN/m" in pE and "soudé sur un plat" in pE and "fournisseur" in pE)
chk("Eurocode : la section « Méthode ELS » n'est pas imprimée, la section « Eurocode » l'est", serre(MT.SECTIONS[2]["paragraphes"][0][:70]) not in serre(pE)
    and serre(MT.SECTIONS[3]["paragraphes"][0][:70]) in serre(pE))
pM = norm(texte(PDF["mur"], 0))
chk("système mur : lisse ancrée dans un mur, consoles C1 …, R_v = q_v·s_trib, M = R_v·e, N_t (platine tirée)",
    "lisse ancrée dans un mur" in pM and "C1" in pM and "R_v = q_v·s_trib" in pM and "M = R_v·e" in pM and "N_t" in pM)
pU = norm(texte(PDF["U plat boulons"], 0))
chk("U à entraxes saisis : « en U », travées de 1,20 m + 0,90 m, fin ancrée dans le mur, montant d'angle, boulons M12 8.8, F_t,Rd/γ_F",
    "en U" in pU and "1,20 m + 0,90 m" in pU and "fin ancrée dans le mur" in pU and "angle" in pU and "M12" in pU and "F_t,Rd/γ_F" in pU)
pS = norm(texte(PDF["scellé"], 0))
chk("scellé : la fixation n'est pas vérifiée, pas de tableau des boulons", "scellé" in pS.lower() and "F_t = Z/n" not in pS)
pP = norm(texte(PDF["mode P"], 0))
chk("mode P : P = 1,2 kN par montant (saisie directe)", "P = 1,2 kN" in pP and "saisie directe" in pP)
pLib = norm(texte(PDF["section libre"], 0)) + norm(texte(PDF["section libre"], 1))
chk("section libre : « Caisson 60x40x3 », non standard, soudure C8", "Caisson 60x40x3" in pLib and "C8" in pLib)

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

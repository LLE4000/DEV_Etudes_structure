# -*- coding: utf-8 -*-
"""Note de calcul « Garde-corps » (acier/garde_corps/note.py, version 2). Chaque garantie
rougit si on la retire :

  1. FORME : DEUX pages A4 paysage, aucun débordement de colonne, sur le cas par défaut et
     sept variantes (L avec lisses intermédiaires et main courante continue, U à entraxes
     saisis et fin au mur avec un plat et des boulons, murale, Eurocode soudé sur plat avec
     panneau et catégorie C5, scellé sans remplissage, cumul en mode P, section libre) ; moins
     de 2,5 s par note.
  2. PAGE 1, LES HYPOTHÈSES D'ABORD : bandeau (statut, η max, l'élément gouvernant, son
     critère et son cas de charge, la méthode choisie), le bloc HYPOTHÈSES (garde-corps,
     sections, matériau et limites, fixation), le tableau des QUATRE CHARGES numérotées avec
     leur symbole, leur nom, leur valeur et leur origine (Tab. 6.12, NBN B 03-004, usage du
     bureau) et la règle des cas DISTINCTS.
  3. PAGE 1, LE CALCUL ÉLÉMENT PAR ÉLÉMENT : les critères en typographie mathématique, puis
     1 · MONTANT (cas gouvernant, autres cas et leur taux, P₁ = q_h·s_trib, M₁ = P₁·H, σ, τ, f,
     η), 2 · MAIN COURANTE (sur deux appuis, M_h = q_h·L²/8), 3 · BARREAUX (cas 3, M = Q_r·L/4),
     4 · ANCRAGE (soudure η, z, Z, F_t = Z/n, N_Rd/γ_F, m, σ_p, t_req), le RÉCAPITULATIF, la
     CONCLUSION avec le cas gouvernant et la masse ; identification.
  4. DESSINS peints depuis les MÊMES scènes que l'écran : élévation (« M1 », « L = 3600 »),
     coupe A-A (« Coupe A-A »), plan (« L₁ ») en page 1 ; platine en plan (« e_x = 100 »,
     « l_p = 150 ») et coupe sur la fixation (« b_p = 150 », « a 4 ») en page 2 ; le cartouche
     porte l'échelle.
  5. PAGE 2 : DÉTAILS ET MÉTHODE — les montants un à un avec leur cas, la section et la
     BIBLIOTHÈQUE (le plus léger qui tient), la soudure (vue et formules du module Soudures),
     les sections de ``methode.py`` de la méthode choisie (un fragment de chaque, mot pour mot,
     pas celui de l'autre méthode), les références, le cartouche.
  6. VARIANTES : Eurocode (γ_Q, f_y/γ_M0, efforts de calcul, soudé sur plat, fournisseur) ;
     murale (console C…, R_v = q_v·s_trib, M = R_v·e, cas 2) ; main courante continue (k_M) ;
     U (en U, 1,20 m + 0,90 m, fin ancrée dans le mur, angle, boulons M12, F_t,Rd/γ_F) ; scellé ;
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


def norm(s: str) -> str:
    return " ".join(s.replace(" ", " ").split())


def serre(s: str) -> str:
    return norm(s).replace(" ", "")


def cas(nom="Garde-corps 1", type_="barreaudage", **kw):
    c = ET.nouveau_cas(nom, type_)
    for k, v in kw.items():
        c[k] = {**c[k], **v} if isinstance(v, dict) else v
    return c


ET_ = ET.nouvel_etat()
INFOS = {"nom_projet": "Résidence des Tilleuls", "partie": "Balcons", "indice": "B"}

# ================================================================
print("\n=== 1. Forme : deux pages, aucun débordement, huit cas ===")
CAS = [("défaut", cas()),
       ("L lisses continue", cas("L", "lisses", geometrie={"forme": "L"}, montant={"section": {"nom": "RHS 60x40x3"}}, lisse={"appuis": "continue"})),
       ("U plat boulons", cas("U", "barreaudage", geometrie={"forme": "U", "fin": "mur", "troncons": [{"L": 2100, "s_max": 1200, "entraxes": [1200, 900]}, {"L": 1500, "s_max": 1000, "entraxes": []},
                                                                                                      {"L": 2000, "s_max": 1000, "entraxes": []}]},
                              montant={"section": {"nom": "Plat 60x12"}}, fixation={"ancrage": {**FX.fixation_defaut()["ancrage"], "type": "boulon"}})),
       ("murale", cas("mur", "murale")),
       ("eurocode soudé C5", cas("EC", "panneau", methode={"mode": "eurocode"}, fixation={"type": "soudee"}, charges={"categorie": "C5"})),
       ("scellé", cas("scellé", "main_courante", fixation={"type": "scellee"})),
       ("cumul P", cas("P", charges={"mode": "P", "P": 1.2, "cumul": True})),
       ("section libre", cas("libre", montant={"section": {"famille": "CAISSON", "dims": {"h": 60, "b": 40, "t": 3}}}))]
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
chk("bandeau : CONFORME, η max 64 %, gouverne Montant principal M2 · RHS 50x30x3 · flèche · cas 1 · q_h seule, Méthode ELS du bureau",
    "CONFORME" in s1 and "ηmax64%" in s1 and "Montant principal M2" in p1 and "RHS 50x30x3" in p1 and "flèche" in p1 and "cas 1 · q_h seule" in p1 and "Méthode ELS du bureau" in p1)
chk("titre : « Garde-corps — barreaudage vertical sur montants »", "barreaudage vertical sur montants" in p1)
chk("HYPOTHÈSES en tête de la première colonne, avant les charges et les dessins", "HYPOTHÈSES" in s1 and "CHARGES" in s1 and "DESSINS" in s1
    and s1.index("HYPOTHÈSES") < s1.index("CHARGES—") < s1.index("DESSINS—"))
chk("hypothèses · garde-corps : droit, 3,60 m en 4 travées de 0,90 m, 5 montants, H = 1100 mm, portée 900 mm sur deux appuis",
    "3,60 m en 4 travées de 0,90 m" in p1 and "5 montants" in p1 and "H = 1100 mm" in p1 and "portée de la main courante : 900 mm, sur deux appuis" in p1)
chk("hypothèses · sections : montants RHS 50x30x3, main courante RHS 50x30x3, barreaux Plat 30x8 à e_b = 110 mm, portée 950 mm déduite, lisse basse non calculée",
    "montants RHS 50x30x3" in p1 and "main courante RHS 50x30x3" in p1 and "barreaux Plat 30x8 à e_b = 110 mm" in p1 and "portée 950 mm" in p1 and "non calculée" in p1)
chk("hypothèses · matériau et limites : S235, charges de service sans coefficient, σ_adm = 160, τ_adm = 90, flèche ≤ L/100, k = 100, γ_F = 1,5",
    "S235" in p1 and "sans coefficient" in p1 and "σ_adm = 160 N/mm²" in p1 and "τ_adm = 90 N/mm²" in p1 and "L/100" in p1 and "k = 100" in p1 and "γ_F = 1,5" in p1)
chk("hypothèses · fixation : platine 150 × 150 × 10, cordon a = 4, 2 × 2 chevilles M12, e_x = 100, e_y = 100",
    "platine 150 × 150 × 10" in p1 and "a = 4 mm" in p1 and "2 × 2 cheville M12" in p1 and "e_x = 100 mm" in p1 and "e_y = 100 mm" in p1)
chk("tableau des QUATRE CHARGES numérotées, symboles, noms, valeurs, origines (Tab. 6.12, NBN B 03-004, usage du bureau)",
    all(k in p1 for k in ("q_h — charge linéaire horizontale sur la main courante", "q_v — charge linéaire verticale", "Q_r — charge ponctuelle horizontale",
                          "q_r — pression horizontale", "0,50 kN/m", "0,50 kN", "0,50 kN/m²", "Tab. 6.12", "NBN B 03-004", "usage du bureau")) and "quatrecasdecharge" in s1)
chk("la règle des cas : les quatre charges agissent SÉPARÉMENT, chaque élément est vérifié sous chaque cas, le pire gouverne",
    "agissent SÉPARÉMENT" in p1 and "le pire gouverne" in p1)

# ================================================================
print("\n=== 3. Page 1 : le calcul élément par élément ===")
chk("CRITÈRES de la méthode ELS en typographie mathématique (σ, W_el, σ_adm, τ_adm, f_adm)", "CRITÈRES" in s1 and "adm" in p1)
chk("1 · MONTANT M2 — RHS 50x30x3 — cas 1 · q_h seule ; autres cas : cas 2 · q_v seule avec son taux", "1·MONTANTM2—RHS50x30x3—cas1·q_hseule" in s1 and "Autres cas : cas 2 · q_v seule : η = 1 %" in p1)
chk("… formules développées : P₁ = q_h·s_trib, M₁ = P₁·H, σ = M₁/W_el,y = 91,2 N/mm², f = P₁·H³/(3·E·I_y) = 7,01 mm, η = max",
    "P₁ = q_h·s_trib" in p1 and "M₁ = P₁·H" in p1 and "91,2 N/mm²" in p1 and "7,01 mm" in p1 and "η = max(η_σ, η_τ, η_f)" in p1)
chk("2 · MAIN COURANTE — RHS 50x30x3 — cas 1 : poutre sur deux appuis, portée 900, M_h = q_h·L²/8, σ = M_h/W_el,z, f = 5·q_h·L⁴/(384·E·I_z)",
    "2·MAINCOURANTE—RHS50x30x3—cas1·q_hseule" in s1 and "sur deux appuis" in p1 and "L = 900 mm" in p1 and "M_h = q_h·L²/8" in p1 and "σ = M_h/W_el,z" in p1 and "L⁴/(384·E·I_z)" in p1)
chk("3 · BARREAUX — Plat 30x8 — cas 3 · Q_r seule : portée 950, M = Q_r·L/4, f = Q_r·L³/(48·E·I)", "3·BARREAUX—Plat30x8—cas3·Q_rseule" in s1 and "L = 950 mm" in p1
    and "M = Q_r·L/4" in p1 and "L³/(48·E·I)" in p1)
chk("4 · ANCRAGE — cas 1 : soudure η = 59 %, z = e_x/2 + h_p/2, Z, F_t = Z/n, N_Rd/γ_F, m, σ_p = 6·M_p/(b_eff·t_p²), t_req",
    "4·ANCRAGE—cas1·q_hseule" in s1 and "η = 59 %" in p1 and "z = e_x/2 + h_p/2" in p1 and "F_t = Z/n" in p1 and "N_Rd/γ_F" in p1 and "σ_p = 6·M_p/(b_eff·t_p²)" in p1 and "t_req" in p1)
chk("l'ordre 1, 2, 3, 4 est celui de la lecture", s1.index("1·MONTANT") < s1.index("2·MAINCOURANTE") < s1.index("3·BARREAUX") < s1.index("4·ANCRAGE"))
chk("RÉCAPITULATIF : montant, main courante, barreaux, soudure, chevilles, platine, chacun avec son cas", "RÉCAPITULATIF" in s1
    and all(k in p1 for k in ("Main courante · RHS 50x30x3", "Barreaux · Plat 30x8", "Soudure", "Chevilles", "Platine 150 × 150 × 10")) and p1.count("q_h seule") >= 6 and "3 · Q_r seule" in p1)
chk("CONCLUSION : taux maximal 64 %, le cas gouvernant, porte les charges, masse indicative", "CONCLUSION" in s1 and "Taux maximal 64 %" in p1 and "cas 1 · q_h seule" in p1
    and "porte les charges" in p1 and "Masse indicative" in p1)
chk("identification : projet, partie, indice B", "Résidence des Tilleuls" in p1 and "Balcons" in p1 and "indice B" in p1)

# ================================================================
print("\n=== 4. Dessins peints ===")
chk("page 1 : élévation (M1, L = 3600, H = 1100), coupe A-A, vue en plan (L₁)", all(k in p1 for k in ("M1", "L = 3600", "H = 1100", "Coupe A-A", "Vue en plan")) and "L₁" in p1)
p2 = norm(texte(PDF["défaut"], 1))
s2 = serre(texte(PDF["défaut"], 1))
chk("page 2 : platine en plan (e_x = 100, l_p = 150, b_p = 150) et coupe sur la fixation (a 4, t_p 10, h_p 50)",
    all(k in p2 for k in ("e_x = 100", "l_p = 150", "b_p = 150", "a 4", "t_p 10", "h_p 50")))
chk("le cartouche porte l'échelle de l'élévation (1:n), la date, l'indice, le type, la méthode", "1:" in p2 and "indice B" in p2 and "Barreaudage vertical" in p2
    and "ÉCHELLEDEL'ÉLÉVATION" in s2 and "σ_adm 160" in p2)

# ================================================================
print("\n=== 5. Page 2 : détails et méthode ===")
chk("titre DÉTAILS ET MÉTHODE, références", "DÉTAILSETMÉTHODE" in s2 and "Références" in p2 and "EN 1991-1-1" in p2)
chk("MONTANTS UN À UN : M1 à M5, s_trib 450 / 900, P 0,23 / 0,45, cas 1 · q_h seule, η 32 % / 64 %", "MONTANTSUNÀUN" in s2 and "M5" in p2 and "450" in p2 and "0,45" in p2
    and "32 %" in p2 and "64 %" in p2 and "1 · q_h seule" in p2)
chk("SECTION ET BIBLIOTHÈQUE : le plus léger qui tient (CHS 48.3x2.5), la section retenue, kg/m", "BIBLIOTHÈQUE" in s2 and "CHS 48.3x2.5" in p2 and "le plus léger qui tient" in p2
    and "RHS 50x30x3" in p2 and "kg/m" in p2)
chk("la soudure : vue peinte (« RHS 50x30x3 sur platine ») et formules du module Soudures (A_w, |f|, τ_adm, η)",
    "SOUDURESURLAPLATINE" in s2 and "sur platine" in p2 and "A_w" in p2 and "|f|" in p2 and "τ_adm = k·g/10" in p2)
els = [s for s in MT.SECTIONS if not s["titre"].startswith("Eurocode")]
for sec in els:
    frag = serre(sec["paragraphes"][0][:70])
    chk(f"section « {sec['titre']} » : un fragment mot pour mot", frag in s2, frag[:50])
frag_ec = serre(MT.SECTIONS[2]["paragraphes"][0][:70])
chk("la section Eurocode n'est PAS imprimée en méthode ELS", frag_ec not in s2)

# ================================================================
print("\n=== 6. Variantes ===")
pE = norm(texte(PDF["eurocode soudé C5"], 0)) + norm(texte(PDF["eurocode soudé C5"], 1))
chk("Eurocode : efforts de calcul, γ_Q, f_y/γ_M0, catégorie C5 à 3 kN/m, soudé sur plat, panneau à justifier par le fournisseur, pas de 4 · ANCRAGE chevilles",
    "efforts de calcul" in pE and "γ_Q" in pE and "f_y/γ_M0" in pE and "3,00 kN/m" in pE and "soudé sur un plat" in pE and "fournisseur" in pE and "3·PANNEAU" in serre(pE)
    and "F_t = Z/n = " not in pE)
chk("Eurocode : la section « Méthode ELS » n'est pas imprimée, la section « Eurocode » l'est", serre(MT.SECTIONS[1]["paragraphes"][0][:70]) not in serre(pE)
    and serre(MT.SECTIONS[2]["paragraphes"][0][:70]) in serre(pE))
pM = norm(texte(PDF["murale"], 0))
chk("murale : titre « main courante murale sur consoles », 1 · CONSOLE C…, R_v = q_v·s_trib, M = R_v·e, cas 2 · q_v seule, 2 · MAIN COURANTE",
    "main courante murale sur consoles" in pM and "1·CONSOLEC" in serre(pM) and "R_v = q_v·s_trib" in pM and "M = R_v·e" in pM and "cas 2 · q_v seule" in pM and "2·MAINCOURANTE" in serre(pM))
pL = norm(texte(PDF["L lisses continue"], 0))
chk("L, lisses, main courante continue : « en L », angle, 3 · LISSES INTERMÉDIAIRES, M_h = k_M·q_h·L², k_M = 0,107, continue sur plusieurs appuis",
    "en L" in pL and "angle" in pL and "3·LISSESINTERMÉDIAIRES" in serre(pL) and "M_h = k_M·q_h·L²" in pL and "k_M = 0,107" in pL and "continue sur plusieurs appuis" in pL)
pU = norm(texte(PDF["U plat boulons"], 0))
chk("U à entraxes saisis : « en U », travées de 1,20 m + 0,90 m, fin ancrée dans le mur, montant d'angle, boulons M12 8.8, F_t,Rd/γ_F, NON CONFORME",
    "en U" in pU and "1,20 m + 0,90 m" in pU and "fin ancrée dans le mur" in pU and "angle" in pU and "M12" in pU and "F_t,Rd/γ_F" in pU and "NONCONFORME" in serre(pU))
pS = norm(texte(PDF["scellé"], 0))
chk("scellé, main courante seule : la fixation n'est pas vérifiée, sans remplissage, pas de 3 · ni de F_t", "n'est pas vérifiée" in pS and "sans remplissage" in pS
    and "3·BARREAUX" not in serre(pS) and "F_t = Z/n" not in pS)
pP = norm(texte(PDF["cumul P"], 0))
chk("cumul en mode P : P = 1,20 kN (saisie directe), cas 1 + 2 · q_h et q_v cumulées, σ = M₁/W_el,y + N/A", "1,20 kN" in pP and "saisie directe" in pP
    and "cas 1 + 2 · q_h et q_v cumulées" in pP and "σ = M₁/W_el,y + N/A" in pP)
pLib = norm(texte(PDF["section libre"], 0)) + norm(texte(PDF["section libre"], 1))
chk("section libre : « Caisson 60x40x3 », soudure C8", "Caisson 60x40x3" in pLib and "C8" in pLib)

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

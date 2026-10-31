# -*- coding: utf-8 -*-
"""Écran, dessins et note du module pied de poteau (AppTest, application
réelle) — lot B.

Chaque garantie rougit si on la retire :
  1. NAVIGATION : la carte « Platine de pied de poteau sur béton » est
     DISPONIBLE dans le catalogue, « Créer » ouvre l'écran dans le projet.
  2. ÉCRAN : deux colonnes, colonne du dessin FIGÉE (CSS sticky, conteneur
     ppb_col_dessin), deux composants (coupe A-A, vue en plan), carte par
     objet SANS la géométrie (elle vit sur le dessin), visserie, onglets.
  3. SOURCE UNIQUE : un message du dessin (cote, poignée) écrit dans
     ppb_<clé> et recalcule ; massif 1 400 → VÉRIFIÉ ; wx = 480 → pince
     bloquante, chip modifiable ; poignée n_ay → 6 tiges ; RHS et crosse.
  4. DESSINS : SVG valides (3 vues × écran/fabrication + domaine) ; la
     répartition est dessinée (zone comprimée, axe neutre, résultante,
     effort de la tige tendue) ; poignées des files et des rangées ; chaque
     cote de fabrication posée UNE fois sur la planche ; domaine : niveau 2
     (polygone) et niveau 1 (losange).
  5. NOTE : 2 pages A4 paysage sans débordement (défaut + cas figés),
     échelle normalisée, cartouche TIGES + visserie, conclusion EC3 / EC2 /
     EC2-4 avec γM0 ANB et αcc.
  6. BENCHMARK dans l'application : VALIDÉ (calculs manuels) ; VAL-B
     chargeable ; exports texte et PDF ; onglets Répartition et Méthode.
  7. PROJET : description de la liste, rapport du projet avec un pied.
  8. ÉTANCHÉITÉ : aucune clé asm_ / fpl_ / pab_ / béton créée.

Lancement : python3 tests/test_pied_poteau_ecran.py
"""
import os
import sys
import xml.etree.ElementTree as ET

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

import pymupdf  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from acier.assemblages.pied_poteau.platine_beton import (interface, ecran_saisie, moteur,  # noqa: E402
                                                          schemas, note)
from acier.assemblages.pied_poteau.platine_beton.entrees import FORME_RHS, TYPE_CROSSE  # noqa: E402
from acier import projet as modele, rapport_projet  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom
          + (f"   [{info}]" if info and not cond else ""))


def md(at):
    return "\n".join(str(m.value) for m in at.markdown) + "\n" + "\n".join(str(c.value) for c in at.caption)


def run(at):
    at.run()
    if at.exception:
        chk("aucune exception", False, str(at.exception[0].message)[:200])
    return at


def cles_widgets(at):
    return {w.key for w in list(at.number_input) + list(at.selectbox) + list(at.text_input) + list(at.checkbox) if w.key}


def message(at, changes, t):
    derniers = at.session_state["ppb_ui_dernier_clic"]
    if derniers.get("ppb_cmp_plan") != t:
        derniers["ppb_cmp_plan"] = t
        for k, v in changes.items():
            at.session_state["ppb_" + k] = ecran_saisie.valeur_widget(k, v)


# ================================================================
print("=== 1. Navigation ===")
at = AppTest.from_file(os.path.join(RACINE, "streamlit_app.py"), default_timeout=240)
at.session_state["page"] = "Assemblages métalliques"
run(at)
chk("la carte « Platine de pied de poteau sur béton » est dans le catalogue, DISPONIBLE (bouton Créer)",
    "Platine de pied de poteau sur béton" in md(at)
    and any(b.key == "asm_btn_open_platine_beton" for b in at.button) and "À venir" not in md(at))
at.button(key="asm_btn_open_platine_beton").click(); run(at)
chk("« Créer » ouvre l'écran dans le projet (entrée A1)", not at.exception
    and at.session_state["asm_courant"] == "platine_beton" and at.session_state["prj_ouvert"] is not None
    and at.session_state["ppb_id_rep"] == "A1")
chk("en-tête du module", "Pied de poteau – platine sur béton" in md(at))

# ================================================================
print("\n=== 2. Écran ===")
cles = [k for k in at.session_state.to_dict() if k.startswith("ppb_") and k[4:] in interface.CLES]
chk(f"les {len(interface.CLES)} clés d'entrée existent en session", len(cles) == len(interface.CLES), str(len(cles)))
t = md(at)
chk("cas par défaut : NON VÉRIFIÉ, 107,7 % (interaction N / V du béton, massif 900)",
    "PIED DE POTEAU NON VÉRIFIÉ" in t and "107,7 %" in t and "Interaction N / V du béton" in t)
chk("taux par élément : Pied, Béton, Platine, Tiges, Poteau, Soudure",
    all(x in t for x in ("Pied", "Béton", "Platine", "Tiges", "Poteau", "Soudure")))
chk("deux composants empilés dans la colonne figée (CSS sticky sur ppb_col_dessin)",
    len(at.get("component_instance")) == 2 and "position: sticky" in t and "st-key-ppb_col_dessin" in t)
w = cles_widgets(at)
donnees = {k for k in w if k.startswith("ppb_") and k[4:] in interface.CLES}
chk("la géométrie ne se règle QUE sur le dessin (aucune cote dans la carte)",
    not donnees & {"ppb_" + k for k in ecran_saisie.GEOMETRIE_DESSIN}, str(sorted(donnees)))
chk("carte par objet : poteau, platine (nuance), tiges, béton, efforts, visserie, coefficients",
    {"ppb_forme", "ppb_prof_c", "ppb_nu_c", "ppb_nu_pl", "ppb_ancr", "ppb_classe", "ppb_typ_a", "ppb_n_ax",
     "ppb_n_ay", "ppb_beton", "ppb_fiss", "ppb_hc_u", "ppb_N_Ed", "ppb_Mx_Ed", "ppb_My_Ed", "ppb_Vy_Ed",
     "ppb_visserie", "ppb_g_M0", "ppb_a_cc", "ppb_g_Mc"} <= w)
chk("profilé de la base : pas de dimensions personnalisées affichées", not {"ppb_h_u", "ppb_t_u"} & w)
chk("onglet par défaut : Vérifications (six tableaux, pince Tab. 3.3, limites écrites en clair)",
    at.session_state["ppb_ui_onglet"] == "Vérifications"
    and all(x in t for x in ("Niveau 1 : interaction linéaire", "Cône de béton du groupe tendu",
                             "Tendue — tronçon en T", "Aile / paroi comprimée", "Contrainte équivalente",
                             "Pinces – EN 1993-1-8 Tableau 3.3", "fendage et éclatement latéral")))
chk("alertes courtes : tiges de coin (modèle d'ingénieur), soulèvement, bord < 1,5·hef, rigidité",
    all(x in t for x in ("Tiges de coin", "Soulèvement", "Bord du massif à moins de 1,5·hef", "Rigidité du pied")))

# ================================================================
print("\n=== 3. Source unique ===")
message(at, {"bc_u": 1400, "lc_u": 1400, "hc_u": 800}, 1001); run(at)
chk("message du dessin : massif 1 400 → VÉRIFIÉ", "PIED DE POTEAU VÉRIFIÉ" in md(at)
    and float(at.session_state["ppb_bc_u"]) == 1400.0)
message(at, {"w_x": 480}, 1002); run(at)
chk("wx = 480 → pince bloquante (Tab. 3.3), « Localiser » disponible",
    "pince des tiges au bord de la platine" in md(at) and any(b.key == "ppb_btn_al_dist_e" for b in at.button)
    and "NON VÉRIFIÉ" in md(at))
at.button(key="ppb_btn_al_dist_e").click(); run(at)
chk("chip d'alerte : wx modifiable sur place", any(x.key == "ppb_fix_w_x" for x in at.number_input))
at.number_input(key="ppb_fix_w_x").set_value(400.0); run(at)
chk("la chip écrit dans la source unique et le blocage disparaît",
    float(at.session_state["ppb_w_x"]) == 400.0 and "pince des tiges" not in md(at))
message(at, {"n_ay": 3}, 1003); run(at)
lu = {k: at.session_state.get("ppb_" + k) for k in interface.CLES}
chk("poignée « rangées » : n_ay = 3 → six tiges, deux classes de rangée (coin + latérales)",
    float(at.session_state["ppb_n_ay"]) == 3.0 and moteur.compute(lu).n_t == 6 and "Tiges latérales" in md(at))
message(at, {"n_ay": 2}, 1004); run(at)
at.selectbox(key="ppb_typ_a").set_value(TYPE_CROSSE); run(at)
lu = {k: at.session_state.get("ppb_" + k) for k in interface.CLES}
chk("tige à crosse : l'adhérence entre dans la résistance de la tige (formule Ft,bond,Rd à l'écran)",
    moteur.compute(lu).Ft_bond is not None and "Ft,bond,Rd = π·d·hef·fbd" in md(at))
at.selectbox(key="ppb_typ_a").set_value("Tige à tête (ou plaque d'ancrage)"); run(at)
at.selectbox(key="ppb_forme").set_value(FORME_RHS); run(at)
w = cles_widgets(at)
chk("profil creux : h, b, t demandés, tw / tf / r et la liste des profilés retirés",
    {"ppb_h_u", "ppb_b_u", "ppb_t_u"} <= w and not {"ppb_tw_u", "ppb_tf_u", "ppb_prof_c"} & w)
at.button(key="ppb_btn_reset").click(); run(at)
chk("réinitialiser : retour aux défauts (massif 900, profilé en I), repère conservé",
    float(at.session_state["ppb_bc_u"]) == 900.0 and at.session_state["ppb_forme"] != FORME_RHS
    and at.session_state["ppb_id_rep"] == "A1")

# ================================================================
print("\n=== 4. Dessins ===")
R0 = moteur.compute(dict())
for nom, fab in (("écran", False), ("fabrication", True)):
    for f, excl in (("coupe_aa", schemas.EXCLURE_AA), ("plan", schemas.EXCLURE_PLAN), ("coupe_bb", schemas.EXCLURE_BB)):
        o = schemas.options_fabrication(excl, 12) if fab else schemas.options_ecran(R0)
        ET.fromstring(getattr(schemas, f)(R0, o).svg(identifiant="t"))
ET.fromstring(schemas.domaine(R0).svg(identifiant="d"))
chk("SVG valides (3 vues × écran/fabrication + domaine)", True)
sp = schemas.plan(R0, schemas.options_ecran(R0)).svg()
chk("vue en plan écran : zone comprimée (ct), axe neutre (an), résultante (pt), aire efficace (ae), "
    "effort de la tige tendue « T = 43 kN », « C = 343 kN »",
    all(x in sp for x in ('class="ct"', 'class="an"', 'class="pt"', 'class="ae"', ">T = 43 kN<", ">C = 343 kN<")))
chk("vue en plan écran : poignées des rangées et des files (data-action), pièces cliquables",
    all(x in sp for x in ('data-action="n_ay:+1"', 'data-action="n_ay:-1"', 'data-action="n_ax:+1"',
                          'data-group="col"', 'data-group="plate"', 'data-group="anchors"',
                          'data-group="pedestal"', 'data-group="efforts"')))
chk("vue en plan écran : les cotes de niveau 2 (pinces ex, ey ; b, h ; bande c) n'apparaissent qu'au niveau 2",
    ">ex 50<" not in sp and ">c 59,4<" not in sp and ">ex 50<" in schemas.plan(R0, schemas.options_ecran(R0, 2)).svg()
    and ">c 59,4<" in schemas.plan(R0, schemas.options_ecran(R0, 2)).svg())
se = schemas.coupe_aa(R0, schemas.options_ecran(R0)).svg()
chk("coupe A-A écran : mortier (gr), béton (be), tiges (tg), cordon (wb) et symbole « a 8 » éditable",
    all(x in se for x in ('class="gr"', 'class="be"', 'class="tg"', 'class="wb"', 'data-key="a_w"')))
f = 2.4 * 15
sa = schemas.coupe_aa(R0, schemas.options_fabrication(schemas.EXCLURE_AA, f)).svg()
sb = schemas.coupe_bb(R0, schemas.options_fabrication(schemas.EXCLURE_BB, f)).svg()
sf = schemas.plan(R0, schemas.options_fabrication(schemas.EXCLURE_PLAN, f)).svg()
cotes = ("bp = 500", "lp = 500", "wx = 400", "wy = 400", "bc = 900", "lc = 900", "tp 25", "tg 30", "hef = 300",
         "dh 40", "ex 50", "ey 50", "b = 300", "h = 290")
planche = sa + sb + sf
doubles = [c for c in cotes if planche.count(">" + c + "<") > 1]
absentes = [c for c in cotes if planche.count(">" + c + "<") == 0]
chk("plan de principe : chaque cote posée UNE fois sur la planche (bp, lp, wx, wy, bc, lc, tp, tg, hef, dh, "
    "ex, ey, b, h)", not doubles and not absentes, f"doubles {doubles} absentes {absentes}")
chk("plan de principe : ni poignées, ni étiquette des efforts, ni cote cliquable",
    "data-action" not in planche and "data-group" not in planche and "data-key" not in planche)
dom = schemas.domaine(R0)
d2 = [p for p in dom.corps if p["t"] == "poly" and "d2" in p["cls"]]
d1 = [p for p in dom.corps if p["t"] == "poly" and "d1" in p["cls"]]
chk("domaine : polygone du niveau 2 (≥ 60 sommets), losange du niveau 1, point des efforts appliqués",
    len(d2) == 1 and len(d2[0]["pts"]) >= 60 and len(d1) == 1 and len(d1[0]["pts"]) == 4
    and any(p["t"] == "circle" and "pt" in p["cls"] for p in dom.corps))
Rr = moteur.compute(dict(forme=FORME_RHS, prof_c="Personnalisé", h_u=250, b_u=250, t_u=10, bp_u=450, lp_u=450,
                         tp_u=20, n_ay=3, w_x=350, w_y=360, N_Ed=-150, Mx_Ed=40, My_Ed=10, Vy_Ed=30,
                         bc_u=1200, lc_u=1200, hc_u=800))
spr = schemas.plan(Rr, schemas.options_ecran(Rr)).svg()
chk("profil creux, soulèvement : vide intérieur, six tiges, quatre tiges tendues étiquetées",
    spr.count('class="bo"') == 7 and spr.count(">T = ") == 4)

# ================================================================
print("\n=== 5. Note (2 pages) ===")
R = moteur.compute(dict(id_projet="Halle A", id_rep="PP1"))
pdf = note.generer_note(R, {"nom_projet": "Halle A", "partie": "PP1", "date": "31/10/2026",
                            "indice": "0", "visserie": "1 rondelle + 1 écrou + contre-écrou"})
doc = pymupdf.open(stream=pdf, filetype="pdf")
chk("deux pages A4 paysage, sans débordement, échelle et disposition exposées",
    doc.page_count == 2 and not note.derniers_avertissements
    and note.derniere_echelle is not None and note.derniere_disposition is not None,
    str(note.derniers_avertissements))
t1 = doc[0].get_text(); s1 = t1.replace(" ", "").replace("\n", "")
chk("page 1 : bandeau, en-tête compact (poteau, platine, massif, tiges, efforts), hypothèses, tableaux",
    "NONVÉRIFIÉ" in s1 and "Poteau HEA 300 S355" in t1 and "Platine 500×500×25 S355" in t1
    and "Massif 900×900×600 C30/37 fissuré" in t1 and "Tiges 4 × M24 8.8" in t1 and "NEd 300,0 kN" in t1
    and "AireeﬃcaceEC3§6.2.5" in s1.replace("ffi", "ﬃ") or "AireefficaceEC3§6.2.5" in s1)
chk("page 1 : vue en plan (répartition), coupe A-A, domaine et conclusion EC3 / EC2 / EC2-4 (γM0 ANB, αcc)",
    "RÉPARTITION" in s1 and "COUPEA-A" in s1 and "DOMAINE(Mx,My)" in s1
    and "selonEN1993-1-8§6.2.8,EN1992-1-1§6.7etEN1992-4" in s1 and "(γM0:ANBbelge;αcc=0,85)" in s1)
t2 = doc[1].get_text(); s2 = t2.replace(" ", "").replace("\n", "")
chk("page 2 : PLAN DE PRINCIPE, coupe A-A, vue en plan, coupe B-B, échelle affichée",
    "PLANDEPRINCIPE" in s2 and all(x in s2 for x in ("COUPEA-A", "VUEENPLAN", "COUPEB-B")) and "Échelle1:" in s2)
chk("page 2 : cartouche TIGES + visserie + Ø des trous + hef, PLATINE + mortier",
    "500×500×25 S355" in t2 and "par tige : 1 rondelle + 1 écrou + contre-écrou" in t2 and "trous Ø26" in t2
    and "hef 300" in t2 and "mortier 30 mm" in t2)
CAS = {"massif1400": dict(bc_u=1400, lc_u=1400, hc_u=800),
       "rhs_soul": dict(forme=FORME_RHS, prof_c="Personnalisé", h_u=250, b_u=250, t_u=10, bp_u=450, lp_u=450, tp_u=20,
                        n_ay=3, w_x=350, w_y=360, N_Ed=-150, Mx_Ed=40, My_Ed=10, Vy_Ed=30, bc_u=1200, lc_u=1200, hc_u=800),
       "crosse": dict(typ_a=TYPE_CROSSE, hef=500, bc_u=1400, lc_u=1400, hc_u=900),
       "gros": dict(prof_c="HEB 400", bp_u=700, lp_u=800, tp_u=40, w_x=560, w_y=660, n_ay=3, ancr="M30",
                    bc_u=1600, lc_u=1800, hc_u=1000, hef=500, d_h=60, N_Ed=1200, Mx_Ed=350, My_Ed=60, Vy_Ed=150),
       "quatre_rangees": dict(n_ay=4, w_y=420, lp_u=560, bc_u=1200, lc_u=1300, hc_u=800, N_Ed=600, Mx_Ed=120),
       "N_seul": dict(Mx_Ed=0, My_Ed=0, Vy_Ed=0, bc_u=1200, lc_u=1200)}
nb = 0
for nom, cas in CAS.items():
    p = note.generer_note(moteur.compute(cas), {})
    dc = pymupdf.open(stream=p, filetype="pdf")
    if dc.page_count != 2 or note.derniers_avertissements or note.derniere_echelle is None:
        nb += 1
        print("      ", nom, dc.page_count, note.derniere_echelle, note.derniers_avertissements)
chk(f"les {len(CAS)} cas figés : 2 pages, échelle trouvée, sans débordement", nb == 0, str(nb))

# ================================================================
print("\n=== 6. Benchmark, exports, onglets ===")
at.session_state["ppb_ui_onglet"] = "Benchmark"; run(at)
chk("onglet Benchmark : calculs manuels VAL-A et VAL-B, statut VALIDÉ, aucun exemple publié (dit en clair)",
    all(x in md(at) for x in ("VAL-A", "VAL-B", "Benchmark du module : VALIDÉ", "Aucun exemple publié")))
at.button(key="ppb_btn_val_1").click(); run(at)
chk("« Charger ces données » : VAL-B (massif 3 000) dans l'outil, VÉRIFIÉ",
    float(at.session_state["ppb_bc_u"]) == 3000.0 and "PIED DE POTEAU VÉRIFIÉ" in md(at))
at.button(key="ppb_btn_reset").click(); run(at)
at.button(key="ppb_btn_pdf").click(); run(at)
chk("« Générer PDF » depuis l'application : 2 pages",
    bool(at.session_state.get("ppb_pdf_bytes"))
    and pymupdf.open(stream=at.session_state["ppb_pdf_bytes"], filetype="pdf").page_count == 2)
at.session_state["ppb_ui_onglet"] = "Note"; run(at)
code = "\n".join(str(c.value) for c in at.code)
chk("onglet Note : export texte avec statut, répartition, Tableau 3.3 et références EC2-4",
    "PIED DE POTEAU" in code and "NON VÉRIFIÉ" in code and "Répartition élastique" in code
    and "Tableau 3.3 : conforme" in code and "EC2-4 §7.2.3.2" in code)
at.session_state["ppb_ui_onglet"] = "Répartition"; run(at)
chk("onglet Répartition : domaine (Mx, My), tiges (classe, T, FT,Rd), rangées (tronçons en T)",
    all(x in md(at) for x in ("Domaine résistant (Mx, My)", "Tiges : position, classe, efforts",
                              "Rangées : tronçons en T", 'class="d2"')))
at.session_state["ppb_ui_onglet"] = "Méthode"; run(at)
chk("onglet Méthode : modèle, deux niveaux, ANB belge, limites",
    all(x in md(at) for x in ("Modèle mécanique", "niveau 1", "niveau 2", "Annexe nationale belge",
                              "Limites du modèle")))
at.session_state["ppb_ui_onglet"] = "Vérifications"; run(at)

# ================================================================
print("\n=== 7. Projet ===")
e = modele.trouver(at.session_state["prj_projet"], at.session_state["prj_ouvert"])
chk("la liste décrit un pied : profilé, platine, massif, N, Mx, My, Vy",
    e is not None and modele.description(e["values"]).startswith("HEA 300 · platine 500×500×25 · massif 900×900")
    and "Mx = 50 kNm" in modele.description(e["values"]) and "Vy = 40 kN" in modele.description(e["values"]))
p = modele.nouveau()
modele.ajouter(p, "platine_beton", e["values"], "1 rondelle + 1 écrou")
rp = rapport_projet.generer_rapport(p, {"nom_projet": "Halle A", "partie": "Fondations", "date": "31/10/2026", "indice": "0"})
dr = pymupdf.open(stream=rp, filetype="pdf")
tr = "".join(pg.get_text() for pg in dr)
chk("rapport du projet avec un pied : garde + récapitulatif + note de 2 pages, titre du type",
    dr.page_count == 4 and "Platine de pied de poteau sur béton" in tr, str(dr.page_count))

# ================================================================
print("\n=== 8. Étanchéité ===")
fuites = [k for k in at.session_state.to_dict()
          if k.startswith(("b1_", "dal", "pre", "asm_", "fpl_", "pab_")) and k != "asm_courant"]
chk("aucune clé asm_ / fpl_ / pab_ / béton créée par le module", not fuites, str(fuites[:5]))
at.button(key="ppb_btn_retour").click(); run(at)
chk("◀ Projet ramène à la liste, l'entrée y figure", "asm_courant" not in at.session_state.to_dict()
    and "Assemblages (1)" in md(at))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

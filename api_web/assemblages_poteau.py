# -*- coding: utf-8 -*-
"""La famille POUTRE–POTEAU du module « Assemblages métalliques » du site : plat d'âme soudé, doubles cornières,
platine d'about partielle — sur l'AILE d'un poteau (cas a) ou dans son ÂME (cas b). Fichier d'AIDE (sans MODULE) :
il décrit les trois configurations à la carte commune (api_web/assemblages.py) — leurs sections de données, leurs
libellés courts, leurs pièces, leurs formules, leurs dessins, leur prédimensionnement.

Les moteurs sont ceux du dépôt (acier/assemblages/poutre_poteau/<config>/ : entrees, moteur, synthese, formules,
notation, texte, methode, benchmark, schemas), importés SANS leurs fichiers d'écran. Les textes de l'écran Streamlit
qu'on ne peut pas importer (ils vivent dans ecran_resultats.py, qui charge streamlit) sont des COPIES NOMMÉES, liées
à l'original par tests/test_api_web_assemblages_poteau.py : ``PERIMETRE_AILE_*``, les « paramètres retenus »
(``_parametres_*``), le « détail de la solution retenue par défaut » (``detail_predim_*``) et la règle de choix de
l'onglet Prédim (``REGLE_*``). Les limites de validité (« non revérifiés ») ne sont PAS des copies : elles viennent
de ``methode.limites(ame)``, la source unique que l'écran Streamlit lit aussi.

Ce qui distingue la famille de la famille poutre–poutre, et que ce fichier porte :
- le CAS D'APPUI (aile / âme), en tête de la section « Assemblage » : il commande toute la mécanique (ce que le
  poteau vérifie, la recoupe des semelles et le jeu latéral de la poutre qui entre entre les ailes) ;
- la pièce porteuse est un POTEAU (section « Poteau porteur », sélecteur des profilés « Poteau ») ;
- le « sans objet » de l'aile (plat d'âme et cornières sur l'aile : aucune vérification locale de l'aile n'est
  demandée par les guides) dit dans la vue Méthode, comme dans le repli « Périmètre » de Streamlit ;
- trois vues : élévation, vue en plan (celles de l'écran Streamlit) et coupe A-A (celle de la note de calcul).
"""
from acier.assemblages.poutre_poteau.doubles_cornieres_poteau import (benchmark as BDC, entrees as EDC,
                                                                      formules as FDC, methode as MEDC, moteur as MDC,
                                                                      notation as NDC, schemas as SDC, synthese as YDC,
                                                                      texte as TDC)
from acier.assemblages.poutre_poteau.plat_ame_poteau import (benchmark as BPA, entrees as EPA, formules as FPA,
                                                             methode as MEPA, moteur as MPA, notation as NPA,
                                                             schemas as SPA, synthese as YPA, texte as TPA)
from acier.assemblages.poutre_poteau.platine_about_poteau import (benchmark as BPB, entrees as EPB, formules as FPB,
                                                                  methode as MEPB, moteur as MPB, notation as NPB,
                                                                  schemas as SPB, synthese as YPB, texte as TPB)
from acier.bibliotheques import ORI_P, ORI_S, PERSO
from acier.formats import F, pct
from acier.js import N, js_str, mx

from . import assemblages as A
from . import assemblages_poutre as PP
from . import commun as C

# ------------------------------------------------------------------ copies nommées de l'écran Streamlit
# (ecran_resultats.PERIMETRE_AILE du plat d'âme et des cornières ; le test les relit dans le module d'écran)
PERIMETRE_AILE_PA = ("Aile du poteau : l'attache est dans le plan de l'âme du poteau ; VEd et VEd·z y "
                     "cheminent en plan. Aucune vérification locale de l'aile n'est demandée par "
                     "MSB P5 §3.2 ni par SCI P358 ; la section du poteau se vérifie dans son propre "
                     "calcul (EN 1993-1-1 §6.2).")
PERIMETRE_AILE_DC = ("Aile du poteau : l'attache est centrée sur l'âme du poteau (deux files symétriques "
                     "à ±p3/2) et vit dans son plan ; VEd et VEd·z y cheminent en plan. Sous l'effort "
                     "tranchant seul, le côté poteau n'apporte que la pression diamétrale de l'aile (ou "
                     "les cordons) : aucun cisaillement local de l'aile n'est demandé par MSB P5 §4.2 ni "
                     "par SCI P358. La section du poteau se vérifie dans son propre calcul "
                     "(EN 1993-1-1 §6.2).")

# ------------------------------------------------------------------ libellés courts (symbole, libellé)
# Le libellé LONG et l'aide de chaque champ restent ceux de l'écran Streamlit (entrees.GROUPES, « i » au survol).
LIBELLES = PP.libelles(
    appui=("", "Appui"), prof_P=("", "Profilé"), prof_S=("", "Profilé"),
    g_h=("g<sub>h</sub>", "jeu à l'about"), rec_S=("r<sub>c</sub>", "recoupe"),
    j_lat=("j", "jeu latéral"), lt_ok=("", "Poutre portée maintenue au déversement"),
    p3_u=("p<sub>3</sub>", "entraxe des files"), a_w=("a", "gorge (double)"),
    z_C=("z<sub>p</sub>", "sous le dessus"), orient=("", "Grande aile"))
# Les options : « Aile du poteau » et « Âme du poteau » ne tiennent pas dans leur segment au format tablette (mesuré :
# la colonne Données déborde et défile de côté) ; « Sur l'aile » et « Dans l'âme » y tiennent et se lisent avec le
# libellé « Appui ». La grande aile d'une cornière inégale se dit du côté où elle porte (« Grande aile sur poutre
# principale » = côté poteau ici).
AFFICHAGE = dict(PP.AFFICHAGE, appui={EPA.APPUI_AILE: "Sur l'aile", EPA.APPUI_AME: "Dans l'âme"},
                 orient={ORI_P: "côté poteau", ORI_S: "côté portée"})
SEGMENTS = PP.SEGMENTS | {"appui"}
NOMS_ACIERS = {"P": "poteau", "S": "poutre portée"}


# ------------------------------------------------------------------ outils communs aux trois
def aile(u) -> bool:
    return u.get("appui") != EPA.APPUI_AME


def resume_assemblage(cfg, u):
    m = PP.AFFICHAGE["mode_calc"].get(u.get("mode_calc"), "")
    return f"{C.esc(cfg.court)} · {'aile' if aile(u) else 'âme'} · {m.lower()}"


def resume_poteau(u):
    return f"{C.esc(PP.profil(u, 'P'))} · {C.esc(u.get('nu_P') or '')}"


def resume_portee(u):
    s = f"{C.esc(PP.profil(u, 'S'))} · {C.esc(u.get('nu_S') or '')}"
    if not aile(u) and N(u.get("rec_S")):
        s += f" · r<sub>c</sub> {C.nb(N(u.get('rec_S')))}"
    return s


def items_assemblage(u, *suite):
    """La section 1 : le repère, le mode, le CAS D'APPUI (il commande toute la mécanique), puis ce que la
    configuration y ajoute et les règles du prédimensionnement."""
    return ["id_rep", "mode_calc", "appui",
            ("aide", "Dans l'âme, la poutre entre entre les ailes du poteau : recoupe r<sub>c</sub> et jeu j dans la "
                     "section Poutre portée.")] + list(suite) + PP.predim_items(u)


def items_poteau():
    return ["prof_P", "nu_P", "hP_u", "bP_u", "twP_u", "tfP_u", "rP_u"]


def items_portee(*position):
    """La poutre portée et sa position ; en cas (b), la recoupe des semelles et le jeu latéral (entrees.champ_visible)."""
    return ["prof_S", "nu_S", "hS_u", "bS_u", "twS_u", "tfS_u", "rS_u", ("grp", "Position")] + list(position) + ["rec_S", "j_lat"]


def items_grugeage():
    return ["d_nt", "d_nb", "l_n", "r_n", "lt_ok",
            ("aide", "Un poteau n'oblige pas à gruger : 0 = semelle non grugée.")]


def sans_objet_aile(texte, debut):
    """Ce que l'écran Streamlit ajoute au « sans objet » de son repli Périmètre, sur l'aile seulement."""
    return lambda R: [debut + texte] if R.aile else []


def vues_poteau(S):
    """Les vues du moteur : élévation et vue en plan (celles de l'écran Streamlit, avec ses options, sans
    l'interactivité : les cotes se saisissent dans les Données), puis la coupe A-A de la note de calcul."""
    def vues(R, niveau, hl):
        opt = S.options_ecran(R, niveau, hl)
        opt.interactive, opt.poignees, opt.editables, opt.locked = False, False, niveau >= 1, None
        u = R.u
        cas = "sur l'aile du poteau" if R.aile else "dans l'âme du poteau, entre les ailes"
        return [("elevation", "Élévation", "Élévation", S.elevation(R, opt), f"{u.prof_S} {cas}"),
                ("plan", "Vue en plan", "Plan", S.plan(R, opt), "coupe horizontale au niveau des boulons"),
                ("coupe", "Coupe A-A", "Coupe A-A", S.coupe_aa(R, opt), "regard le long de la poutre portée")]
    return vues


def piece_poteau(R, u):
    """Le poteau d'une ligne du résultat : son profilé (il ouvre le sélecteur) et le cas d'appui."""
    t, pk = PP.piece_profil(u, "P")
    return f"{t} · {'aile' if R.aile else 'âme'}", pk


# ================================================================== plat d'âme soudé sur poteau
def _sections_pa(u, R, el):
    cfg = A.CONFIGS["plat_ame_poteau"]
    return [
        ("as", "Assemblage", resume_assemblage(cfg, u), items_assemblage(u), True),
        ("po", "Poteau porteur", resume_poteau(u), items_poteau(), True),
        ("pt", "Poutre portée", resume_portee(u), items_portee("g_h"), True),
        ("gr", "Grugeage", PP.resume_grugeage(u), items_grugeage(), False),
        ("pl", "Plat d'âme", f"{C.nb(N(u.get('hp_u')))} × {C.nb(N(u.get('bp_u')))} × {C.nb(N(u.get('tp_u')))} · a {C.nb(N(u.get('a_w')))}",
         ["hp_u", "tp_u", "bp_u", "z_C", "nu_pl", ("grp", "Soudure sur le poteau"), "a_w"], False),
        ("bo", "Boulons", f"{C.nb(N(u.get('n1_u')))} × {u.get('n2_u')} {u.get('boulon_u')} {u.get('classe')}",
         PP.boulons_items(u) + [("grp", "Groupe · plat et âme de la portée (1 plan)"), "n1_u", "n2_u", "p1_u", "p2_u",
                                "e1_u", "e2b_u"], True),
        ("ef", "Efforts ELU", PP.resume_efforts(u, ("V_Ed", "N_Ed", "M_Ed")), ["V_Ed", "N_Ed", "M_Ed", "nat_N"], True),
        ("co", "Coefficients et modèle", f"γ<sub>M1</sub> {C.nb(N(u.get('g_M1')))} · ANB",
         PP.coefficients("g_M0", "g_M1", "g_M2", "g_M2n", "g_M3", "g_M3s", "g_Mu", ("grp", "Modèle"), "opt_blf", "expo"), False),
        ("ca", "Cartouche et affichage", C.esc(u.get("id_red") or "rédacteur à saisir"), A.section_cartouche(cfg, u, el), False),
    ]


def _parametres_pa(R):
    """Les paramètres retenus de l'écran Streamlit (plat_ame_poteau/ecran_resultats._parametres_retenus) — copie
    nommée."""
    u = R.u
    g = [["Cas d'appui", str(R.appui) + (" — attache dans le plan de l'âme du poteau" if R.aile
                                         else " — attache perpendiculaire à l'âme du poteau")],
         ["Boulon retenu", R.boulon + " – d0 = " + F(R.d_0, 0) + " mm"],
         ["Plat", js_str(R.h_p) + " × " + js_str(R.b_p) + " × " + js_str(R.t_p) + " mm – " + str(u.nu_pl)],
         ["Excentricité z : face d'appui du poteau → centre du groupe", F(R.zeff, 1) + " mm"],
         ["Moment d'excentricité MS = VEd·z + |MEd|", F(R.M_S, 2) + " kNm"],
         ["Modèle", R.mod],
         ["Inertie polaire du groupe", F(R.Ip, 0) + " mm²"],
         ["Plat court / long (limite tp/0,15 = " + F(R.t_p / 0.15, 1) + " mm)",
          ("LONG (déversement vérifié)" if R.long_p else "court")],
         ["Pince e2 du plat (bord libre)", F(R.e2_p, 1) + " mm"],
         ["Pince e1,b dans l'âme de la poutre portée", F(R.e1b_S, 1) + " mm"],
         ["Pince basse du plat", F(R.e1bot, 1) + " mm"],
         ["Gorge de pleine résistance recommandée", F(R.a_plein, 1) + " mm (a = " + F(R.a_w, 0) + " mm)"],
         ["Moment apporté au POTEAU VEd·e_col (information, hors verdict)",
          F(R.M_col, 2) + " kNm avec e_col = " + F(R.e_col, 1) + " mm"]]
    if R.ame:
        g += [["Largeur droite de l'âme entre congés hP − 2·tfP − 2·rP", F(R.libre_droit, 0) + " mm"],
              ["Passage de la semelle entre ailes hP − 2·tfP − 2·j", F(R.libre_ailes, 0) + " mm (bS = "
               + F(R.b_S, 0) + " mm → " + ("passe" if R.passe_S else "NE PASSE PAS") + ")"],
              ["Recoupe minimale des semelles (bP − twP)/2 + j − gh", F(R.rec_min, 1) + " mm (recoupe "
               + F(N(u.rec_S), 0) + " mm)"],
              ["Variante du cas (b)", "b1 : about au-delà des bouts d'aile (gh ≥ " + F(R.gh_b1, 1) + " mm)"
               if R.b1 else ("b0 : la poutre passe sans recoupe" if R.passe_S and N(u.rec_S) == 0
                             else "b2 : semelles recoupées")]]
    else:
        g += [["Poteau porteur", "aucune vérification locale de l'aile sous VEd seul (voir Méthode)"]]
    return g


def detail_predim_pa(p):
    """Le détail de l'onglet Prédim (plat_ame_poteau/ecran_resultats.onglet_predim) : le texte est celui du plat
    d'âme poutre–poutre, le test le relit dans l'écran poutre–poteau."""
    return PP.detail_predim_pa(p)


REGLE_PA = ("Règle de choix par défaut : le plus petit nombre de rangées ; à égalité, le plus petit "
            "diamètre. L'épaisseur du plat est plafonnée à 0,5·d (ductilité) ; la gorge proposée est la "
            "pleine résistance. Le poteau n'intervient pas dans la proposition : la hauteur disponible "
            "et les taux sont commandés par la poutre portée et le plat. Seule la vérification "
            "détaillée fait foi.")


def _piece_pa(R, u, grp):
    if grp == "Boulons":
        return PP.boulons_texte(R, u, f"{js_str(R.n_1)} × {js_str(R.n_2)}"), ("" if R.pred else "boulon")
    if grp == "Plat d'âme":
        return C.esc(f"{js_str(R.h_p)} × {js_str(R.b_p)} × {js_str(R.t_p)} mm"), ""
    if grp == "Poutre portée":
        return PP.piece_profil(u, "S")
    if grp == "Poteau porteur":
        return piece_poteau(R, u)
    return C.esc(f"gorge a = {js_str(R.a_w)} mm"), ""


A.enregistrer(A.Config(
    cle="plat_ame_poteau", E=EPA, calculer=MPA.compute, S=YPA, N=NPA, T=TPA, methode=MEPA.METHODE, bench=BPA,
    vues=vues_poteau(SPA), sections=_sections_pa, libelles=LIBELLES, affichage=AFFICHAGE, segments=SEGMENTS,
    formules=lambda R, c: A.lignes_texte(R, c, FPA.textes), parametres=_parametres_pa, piece=_piece_pa,
    non_reverifies=lambda R: tuple(MEPA.limites(R.ame)), dimensions=PP.dimensions(EPA),
    sans_objet=sans_objet_aile(PERIMETRE_AILE_PA, "toutes les vérifications locales de l'aile du poteau : "),
    noms_aciers=NOMS_ACIERS, titres_profils={"prof_S": "Poutre portée · profilé", "prof_P": "Poteau · profilé"},
    tableaux_boulons=lambda R: [("groupe", R.tab)],
    predim={"apply": MPA.apply_solution, "label": MPA.solution_label,
            "nom": lambda r: f"{js_str(r.n)} {r.b['n']} · plat {js_str(r.hp)}×{js_str(r.prop.tp)}",
            "col": lambda r: r.hp, "unite_col": "h<sub>p</sub> mm",
            "colonnes": [("variante", lambda r: r.var), ("h<sub>p</sub> (mm)", lambda r: js_str(r.hp)),
                         ("η boulons", lambda r: pct(r.e.bS)), ("η p. diam. âme", lambda r: pct(r.e.pdS))],
            "detail": detail_predim_pa, "regle": REGLE_PA,
            "geo": lambda r: f"e1 {r.e1} / p1 {r.p1} / e2 {r.e2} mm"}))


# ================================================================== doubles cornières sur poteau
def _sections_dc(u, R, el):
    cfg = A.CONFIGS["doubles_cornieres_poteau"]
    bolt_S, bolt_P = u.get("fix_S") == EDC.BOULONNEE, u.get("fix_P") == EDC.BOULONNEE
    groupes = []
    if bolt_P:
        groupes += [("grp", "Groupe P · ailes A sur le poteau (1 plan, par cornière)"), "n1P_u", "n2P_u", "p1P_u", "p2_P",
                    "e1P_u", "gA_u"]
    if bolt_S:
        groupes += [("grp", "Groupe S · âme de la poutre portée (2 plans)"), "n1S_u", "n2S_u", "p1S_u", "p2_S", "e1S_u",
                    "e2b_u"]
    secs = [
        ("as", "Assemblage", resume_assemblage(cfg, u),
         items_assemblage(u, ("grp", "Fixation des cornières"), "fix_P", "fix_S",
                          ("aide", "Ailes A sur le poteau, ailes B sur l'âme de la poutre portée. Soudées des deux "
                                   "côtés : assemblage peu déformable, à réserver à des cas particuliers.")), True),
        ("po", "Poteau porteur", resume_poteau(u), items_poteau(), True),
        ("pt", "Poutre portée", resume_portee(u), items_portee("g_h"), True),
        ("gr", "Grugeage", PP.resume_grugeage(u), items_grugeage(), False),
        ("cn", "Cornières", f"2 {C.esc(u.get('corn_u'))} · h<sub>c</sub> {C.nb(N(u.get('LC_u')))}",
         ["corn_u", "nu_C", "orient", "k1_u", "k2_u", "kt_u", "kr_u", "LC_u", "z_C"], False),
    ]
    if bolt_S or bolt_P:
        secs.append(("bo", "Boulons", f"{u.get('boulon_u')} {u.get('classe')}", PP.boulons_items(u) + groupes, True))
    if not (bolt_S and bolt_P):
        secs.append(("so", "Soudures", "cordons d'angle", ["a_P", "lh_P", "a_S", "lh_S"], True))
    secs += [
        ("ef", "Efforts ELU", PP.resume_efforts(u), ["V_Ed", "N_Ed", "H_Ed", "M_Ed", "nat_N"], True),
        ("co", "Coefficients et modèle", "γ<sub>M</sub> · options",
         PP.coefficients("g_M0", "g_M2", "g_M2n", "g_M3", "g_M3s", "g_Mu", ("grp", "Modèle"), "opt_exc", "k_rot",
                         "opt_blf", "expo"), False),
        ("ca", "Cartouche et affichage", C.esc(u.get("id_red") or "rédacteur à saisir"),
         A.section_cartouche(cfg, u, el, avec_visserie=bolt_S or bolt_P), False),
    ]
    return secs


def _parametres_dc(R):
    """Les paramètres retenus de l'écran Streamlit (doubles_cornieres_poteau/ecran_resultats._parametres_retenus)."""
    u = R.u
    g = [["Cas d'appui", str(R.appui) + (" — attache dans le plan de l'âme du poteau" if R.aile
                                         else " — attache perpendiculaire à l'âme du poteau")],
         ["Boulon retenu", R.boulon + " – d0 = " + F(R.d_0, 0) + " mm"],
         ["Cornières", R.corn_txt + " – aile A " + F(R.b_A, 0) + ", aile B " + F(R.b_B, 0)
          + ", tc " + F(R.t_C, 0) + ", rc " + F(R.r_C, 0) + " mm"],
         ["Hauteur des cornières hc", F(R.L_C, 0) + " mm"
          + ("" if R.rec_L else " (hc < 0,6·h : recommandation MSB Part 5 §4.1 non respectée)")],
         ["Excentricité z : face d'appui du poteau → centre du groupe S "
          + ("" if R.bolt_S else "(cordons)"), F(R.zeff, 1) + " mm"],
         ["Moment d'excentricité MS = VEd·z + |MEd|", F(R.M_S, 2) + " kNm"],
         ["Modèle du groupe S", R.mod_S], ["Modèle du groupe P", R.mod_P],
         ["Moment par groupe P (option)", F(R.M_P, 2) + " kNm"],
         ["Inertie polaire du groupe S", F(R.Ip_S, 0) + " mm²"],
         ["Gauge p3 = tw,S + 2·gA (entraxe des deux files sur le poteau)", F(R.p_3, 1) + " mm"],
         ["Pince e1,b dans l'âme de la poutre portée", F(R.e1b_S, 1) + " mm"],
         ["Pinces e2 : aile B / aile A", F(R.e2a_S, 1) + " / " + F(R.e2a_P, 1) + " mm"],
         ["Pinces basses e1' : groupe S / groupe P", F(R.e1bot_S, 1) + " / " + F(R.e1bot_P, 1) + " mm"],
         ["Trusquinage gA minimal (congé de la cornière) tc + rc + d0/2", F(R.gA_min, 1) + " mm"],
         ["Moment apporté au POTEAU VEd·e_col (information, hors verdict)",
          F(R.M_col, 2) + " kNm avec e_col = " + F(R.e_col, 1) + " mm"]]
    if R.ame:
        g += [["Largeur droite de l'âme entre congés hP − 2·tfP − 2·rP", F(R.libre_droit, 0) + " mm"],
              ["Demi-largeur droite (hP − 2·tfP)/2 − rP", F(R.demi_droit, 1) + " mm"],
              ["Longueur d'aile A maximale (demi-largeur − tw,S/2)", F(R.bA_max, 1)
               + " mm (aile A " + F(R.b_A, 0) + " mm)"],
              ["Gauge p3 maximal (l'écrou dégage le congé)", F(R.p3_max_ame, 1) + " mm"],
              ["Passage de la semelle entre ailes hP − 2·tfP − 2·j", F(R.libre_ailes, 0) + " mm (bS = "
               + F(R.b_S, 0) + " mm → " + ("passe" if R.passe_S else "NE PASSE PAS") + ")"],
              ["Recoupe minimale des semelles (bP − twP)/2 + j − gh", F(R.rec_min, 1) + " mm (recoupe "
               + F(N(u.rec_S), 0) + " mm)"]]
    else:
        g += [["Gauge p3 minimal sur l'aile (twP + 2·rP + dw)", F(R.p3_congeP, 1)
               + " mm — gauges usuels 90 / 100 / 120 / 140 mm"],
              ["Pince au bout d'aile du poteau e = (bP − p3)/2", F(R.e_P_fl, 1) + " mm"],
              ["Tronçon en T de l'aile du poteau : m / n / Σleff",
               F(R.m_P, 2) + " / " + F(R.n_Pf, 2) + " / " + F(R.leff1_P, 0) + " mm (Tab. 6.4)"]]
    return g


def detail_predim_dc(p):
    """Le détail de l'onglet Prédim (doubles_cornieres_poteau/ecran_resultats.onglet_predim) — copie nommée."""
    return [["Boulon proposé", p.boulon], ["Rangées n1 (1 file, groupes S et P)", js_str(p.n)],
            ["e1 / p1 / e2 = e2,b", f"{p.e1} / {p.p1} / {p.e2} mm"],
            ["Hauteur des cornières hc = 2·e1 + (n1 − 1)·p1",
             f"{p.LC} mm (hauteur disponible " + F(p.hav, 0) + " mm)"],
            ["Épaisseur nécessaire au taux cible", F(p.treq, 2) + " mm"],
            ["Cornière proposée", "2 × L" + js_str(p.leg) + "x" + js_str(p.leg) + "x" + js_str(p.tC)
             + " (aile ≥ " + F(p.legreq, 0) + " mm)"],
            ["Trusquinage gA / gauge p3", js_str(p.gA) + " / " + F(p.p3, 0) + " mm"],
            ["Taux estimé (boulons, pressions diamétrales de l'âme et du poteau)", pct(p.eta, 1)],
            ["État", p.msg]]


REGLE_DC = ("Règle de choix par défaut : le plus petit nombre de rangées ; à égalité, le plus petit "
            "diamètre. Le POTEAU n'entre dans la proposition que par la pression diamétrale de son "
            "aile ou de son âme, par le gauge minimal p3 ≥ twP + 2·rP + dw (cas a) et — en cas (b) — "
            "par le plafond de l'aile A et du gauge. Seule la vérification détaillée fait foi.")


def _piece_dc(R, u, grp):
    if grp == "Boulons":
        g = [f"S {js_str(R.n1_S)}×{js_str(R.n2_S)}" if R.bolt_S else "", f"P {js_str(R.n1_P)}×{js_str(R.n2_P)}" if R.bolt_P else ""]
        return C.esc(f"{R.boulon} {u.get('classe')} · " + " · ".join(x for x in g if x)), ("" if R.pred else "boulon")
    if grp == "Cornières":
        return C.esc(f"2 {R.corn_txt} · hc {js_str(R.L_C)}"), ("" if R.pred else "corniere")
    if grp == "Poutre portée":
        return PP.piece_profil(u, "S")
    if grp == "Poteau porteur":
        return piece_poteau(R, u)
    g = [f"a<sub>A</sub> = {C.nb(N(u.get('a_P')))}" if not R.bolt_P else "",
         f"a<sub>B</sub> = {C.nb(N(u.get('a_S')))}" if not R.bolt_S else ""]
    return " · ".join(x for x in g if x) + " mm", ""


LEGENDE_AILES = "Aile A sur le poteau (groupe P) · aile B sur la poutre portée (groupe S)."


def matrice_cornieres(R):
    """La matrice « Cornières — aile A / aile B » de l'onglet Vérifications (synthese.matrice_cornieres) : une ligne
    par vérification, l'aile A (sur le POTEAU), l'aile B (sur la poutre portée) et la plus sollicitée ; la légende
    est celle de l'écran poutre–poteau (``LEGENDE_AILES``)."""
    ls = YDC.matrice_cornieres(R)
    if not ls:
        return ""
    lignes = []
    for x in ls:
        refs = YDC.refs_fusionnees([c["ref"] for c in (x["A"], x["B"]) if c])
        m = x["max"]["c"]
        lignes.append([f"<b>{C.esc(x['lab'])}</b>", PP._cellule_aile(x["A"]), PP._cellule_aile(x["B"]),
                       C.chip(A.taux_c(m, 1), A.niveau_c(m)), C.esc(refs)])
    return ('<div class="ck-g asm-large"><div class="ck-t">Cornières — aile A / aile B</div>'
            + A.tableau(["Vérification", "Aile A : E<sub>d</sub> / R<sub>d</sub> · η", "Aile B : E<sub>d</sub> / R<sub>d</sub> · η",
                         "max", "Réf."], lignes, droite=(1, 2, 3), classe="asm-tab asm-mat")
            + f'<div class="asm-note">{C.esc(LEGENDE_AILES)}</div></div>')


A.enregistrer(A.Config(
    cle="doubles_cornieres_poteau", E=EDC, calculer=MDC.compute, S=YDC, N=NDC, T=TDC, methode=MEDC.METHODE, bench=BDC,
    vues=vues_poteau(SDC), sections=_sections_dc, libelles=dict(LIBELLES, z_C=PP.LIBELLES["z_C@dc"]),
    affichage=AFFICHAGE, segments=SEGMENTS, formules=lambda R, c: A.lignes_texte(R, c, FDC.textes),
    parametres=_parametres_dc, piece=_piece_dc, non_reverifies=lambda R: tuple(MEDC.limites(R.ame)),
    dimensions=PP.dimensions(EDC, corniere=True),
    sans_objet=sans_objet_aile(PERIMETRE_AILE_DC, "cisaillement local de l'aile du poteau : "),
    noms_aciers=NOMS_ACIERS, titres_profils={"prof_S": "Poutre portée · profilé", "prof_P": "Poteau · profilé"},
    regle_33="e ≥ 1,2·d<sub>0</sub> ; p<sub>1</sub> ≥ 2,2·d<sub>0</sub> ; p<sub>2</sub> et p<sub>3</sub> ≥ 2,4·d<sub>0</sub>",
    tableaux_boulons=lambda R: ([("groupe S", R.tabS)] if R.bolt_S else []) + ([("groupe P (une cornière)", R.tabP)] if R.bolt_P and R.M_P > 0 else []),
    selecteurs={"corniere": {"rendre": PP._sel_corniere, "choisir": A._choisir_liste("corn_u", "cornieres")}},
    blocs=lambda R, grp: matrice_cornieres(R) if grp == "Cornières" else "",
    predim={"apply": MDC.apply_solution, "label": MDC.solution_label,
            "nom": lambda r: f"{js_str(r.n)} {r.b['n']} · {MDC.nom_cornière_proposee(r)}",
            "col": lambda r: r.Lc, "unite_col": "h<sub>c</sub> mm",
            "colonnes": [("h<sub>c</sub> (mm)", lambda r: js_str(r.Lc)), ("Cornière", MDC.nom_cornière_proposee),
                         ("η boulons S", lambda r: pct(r.e.bS)), ("η p. diam. âme", lambda r: pct(r.e.pdS)),
                         ("η boulons P", lambda r: pct(r.e.bP)), ("η p. diam. poteau", lambda r: pct(r.e.pdP))],
            "detail": detail_predim_dc, "regle": REGLE_DC,
            "geo": lambda r: f"e1 {r.e1} / p1 {r.p1} / e2 {r.e2} / gA {r.prop.gA} mm · p3 {js_str(r.prop.p3)} mm"}))


# ================================================================== platine d'about partielle sur poteau
def _sections_pb(u, R, el):
    cfg = A.CONFIGS["platine_about_poteau"]
    return [
        ("as", "Assemblage", resume_assemblage(cfg, u), items_assemblage(u), True),
        ("po", "Poteau porteur", resume_poteau(u), items_poteau(), True),
        ("pt", "Poutre portée", resume_portee(u), items_portee(), True),
        ("gr", "Grugeage", PP.resume_grugeage(u), items_grugeage(), False),
        ("pl", "Platine d'about", f"{C.nb(N(u.get('hp_u')))} × {C.nb(N(u.get('bp_u')))} × {C.nb(N(u.get('tp_u')))} · a {C.nb(N(u.get('a_w')))}",
         ["hp_u", "tp_u", "bp_u", "z_C", "nu_pl", ("grp", "Soudure sur l'âme de la portée"), "a_w"], False),
        ("bo", "Boulons", f"2 × {C.nb(N(u.get('n1_u')))} {u.get('boulon_u')} {u.get('classe')}",
         PP.boulons_items(u) + [("grp", "Groupe · 2 files sur le poteau, 1 plan"), "n1_u", "p1_u", "e1_u", "p3_u",
                                ("aide", "Les deux files sont symétriques par rapport à l'âme du poteau ; l'écrou "
                                         "dégage son congé : p<sub>3</sub> ≥ t<sub>w,P</sub> + 2·r<sub>P</sub> + d<sub>w</sub>.")],
         True),
        ("ef", "Efforts ELU", PP.resume_efforts(u, ("V_Ed", "N_Ed")), ["V_Ed", "N_Ed", "nat_N"], True),
        ("co", "Coefficients et modèle", "γ<sub>M</sub> · options",
         PP.coefficients("g_M0", "g_M2", "g_M2n", "g_Mu", "g_M3", "g_M3s", ("grp", "Modèle"), "opt_blf", "expo"), False),
        ("ca", "Cartouche et affichage", C.esc(u.get("id_red") or "rédacteur à saisir"), A.section_cartouche(cfg, u, el), False),
    ]


def _parametres_pb(R):
    """Les paramètres retenus de l'écran Streamlit (platine_about_poteau/ecran_resultats._parametres_retenus)."""
    u = R.u
    g = [["Cas d'appui", str(R.appui) + (" — les deux files sont symétriques par rapport à l'âme du poteau"
                                         if R.aile else " — la platine porte sur l'âme, entre les ailes")],
         ["Boulon retenu", R.boulon + " – d0 = " + F(R.d_0, 0) + " mm – " + js_str(R.n_1) + " rangées × 2 files"],
         ["Platine", js_str(R.h_p) + " × " + js_str(R.b_p) + " × " + js_str(R.t_p) + " mm – " + str(u.nu_pl)],
         ["Modèle", R.mod],
         ["Groupe de boulons : branche retenue", R.branche_g + " = " + F(R.FRd_g, 1) + " kN"],
         ["Effort par boulon Fv,Ed = VEd/n", F(R.Fv_Ed, 2) + " kN"],
         ["Pince latérale e2 = (bp − p3)/2", F(R.e_2, 1) + " mm"],
         ["Pinces verticales e1 (haut) / e1' (bas)", F(R.e_1, 1) + " / " + F(R.e1bot, 1) + " mm"],
         ["Entraxe minimal des files (rondelle hors cordon)", F(R.p3_min, 1) + " mm (p3 = " + F(R.p_3, 0) + " mm)"],
         ["hp ≥ 1,36·p3 (flexion de la platine négligeable)",
          ("oui" if R.hp_136 else "non") + " → k bloc = " + F(R.k_bloc, 1)],
         ["Ductilité : tp,max = 0,36·d·√(fub/fy)", F(R.tp_max, 1) + " mm (tp = " + F(R.t_p, 0) + " mm)"],
         ["Gorge de pleine résistance recommandée",
          F(R.k_plein, 3) + "·tw = " + F(R.a_plein, 1) + " mm (a = " + F(R.a_w, 1) + " mm)"],
         ["Pression diamétrale du poteau : t / αb / k1",
          F(R.t_appui, 1) + " mm (" + ("aile" if R.aile else "âme") + ") / " + F(R.ab_P, 3) + " / "
          + F(R.k1_P, 2) + " → Fb,Rd = " + F(R.Fb_P, 1) + " kN par boulon"],
         ["Moment apporté au POTEAU VEd·e_col (information, hors verdict)",
          F(R.M_col, 2) + " kNm avec e_col = " + F(R.e_col, 1) + " mm"]]
    if R.aile:
        g += [["Gauge minimal du poteau (l'écrou dégage le congé) twP + 2·rP + dw",
               F(R.p3_congeP, 1) + " mm (p3 = " + F(R.p_3, 0) + " mm)"],
              ["Pince au bout d'aile e = (bP − p3)/2", F(R.e_P, 1) + " mm"]]
    else:
        g += [["Largeur droite de l'âme entre congés hP − 2·tfP − 2·rP", F(R.libre_droit, 0) + " mm"],
              ["Largeur maximale de la platine (entre congés, jeux compris)",
               F(R.bp_maxP, 0) + " mm (bp = " + F(R.b_p, 0) + " mm)"],
              ["Dégagement de l'écrou p3/2 + dw/2 ≤ (hP − 2·tfP)/2 − rP",
               F(R.p_3 / 2 + R.d_w / 2, 1) + " ≤ " + F(R.demi_droit, 1) + " mm (p3 maximal "
               + F(R.p3_maxP, 0) + " mm)"],
              ["Passage de la semelle entre ailes hP − 2·tfP − 2·j", F(R.libre_ailes, 0) + " mm (bS = "
               + F(R.b_S, 0) + " mm → " + ("passe" if R.passe_S else "NE PASSE PAS") + ")"],
              ["Recoupe minimale des semelles (bP − twP)/2 + j − tp", F(R.rec_min, 1) + " mm (recoupe "
               + F(N(u.rec_S), 0) + " mm)"]]
    if N(u.N_Ed) > 0:
        g += [["Tronçon en T de la PLATINE : m / n / ew",
               F(R.m_T, 1) + " / " + F(R.nT, 1) + " / " + F(R.ew_T, 2) + " mm — Σleff = "
               + F(R.leff_T, 0) + " mm → FT,Rd = " + F(R.FT_Rd, 1) + " kN"]]
        if R.aile:
            g += [["Tronçon en T de l'AILE DU POTEAU : m / n / ew",
                   F(R.m_fP, 1) + " / " + F(R.n_fP, 1) + " / " + F(R.ew_fP, 2) + " mm — Σleff = "
                   + F(R.leff_fP, 0) + " mm (Tab. 6.4) → FT,Rd = " + F(R.FT_Rd_P, 1) + " kN"],
                  ["Rapport aile du poteau / platine",
                   F(R.FT_Rd_P / R.FT_Rd, 2) if R.FT_Rd > 0 else "—"],
                  ["Âme du poteau en traction transversale (ω = 1, beff = Σleff)",
                   F(R.Ft_wc, 1) + " kN"]]
        else:
            g += [["Flexion HORS PLAN de l'âme du poteau",
                   "NON VÉRIFIÉE — non couverte par l'EN 1993-1-8 pour une âme non raidie (voir Méthode)"]]
    return g


def detail_predim_pb(p):
    """Le détail de l'onglet Prédim (platine_about_poteau/ecran_resultats.onglet_predim) — copie nommée."""
    return [["Boulon proposé", p.boulon], ["Rangées n1 (2 files)", js_str(p.n)],
            ["e1 / p1 / e2", f"{p.e1} / {p.p1} / {p.e2} mm"],
            ["Entraxe des files p3 (≥ 90 mm ; rondelle hors cordon ; écrou hors congé du poteau)", f"{p.p3} mm"],
            ["Hauteur de la platine hp = max(2·e1 + (n1 − 1)·p1 ; 0,6·h)",
             f"{p.hp} mm (hauteur disponible " + F(p.hav, 0) + " mm)"],
            ["Épaisseur nécessaire au taux cible", F(p.treq, 2) + " mm"],
            ["Platine proposée", js_str(p.hp) + " × " + js_str(p.bp) + " × " + js_str(p.tp)
             + " mm – gorge a = " + js_str(p.a) + " mm"],
            ["Taux estimé (boulons, âme portée, poteau)", pct(p.eta, 1)], ["État", p.msg]]


REGLE_PB = ("Règle de choix par défaut : le plus petit nombre de rangées ; à égalité, le plus petit "
            "diamètre. L'épaisseur de la platine reste entre 8 mm et la limite de ductilité ; la gorge "
            "proposée est la pleine résistance du guide. Le poteau n'intervient que par le GAUGE minimal "
            "(l'écrou dégage son congé), la pression diamétrale et — en cas (b) — le cisaillement local "
            "de son âme. Seule la vérification détaillée fait foi.")


def _piece_pb(R, u, grp):
    if grp == "Boulons":
        return PP.boulons_texte(R, u, f"2 × {js_str(R.n_1)}"), ("" if R.pred else "boulon")
    if grp == "Platine d'about":
        return C.esc(f"{js_str(R.h_p)} × {js_str(R.b_p)} × {js_str(R.t_p)} mm"), ""
    if grp == "Poutre portée":
        return PP.piece_profil(u, "S")
    if grp == "Poteau porteur":
        return piece_poteau(R, u)
    return C.esc(f"gorge a = {js_str(R.a_w)} mm"), ""


def _eta_poteau_pb(r, ame):
    """« η poteau » d'une ligne du prédimensionnement, comme la colonne de l'onglet Prédim : pression diamétrale, et
    en cas (b) le cisaillement local de l'âme."""
    return pct(mx(r.e.pdP, r.e.vlP) if ame else r.e.pdP)


def _decimales_bench_pb(r):
    """Les décimales de l'onglet Benchmark de la platine sur poteau : trois pour un rapport sans unité."""
    nd = 3 if r["unit"] == "-" else (1 if abs(r["ref"]) < 10 else 0)
    return nd, nd + 1


A.enregistrer(A.Config(
    cle="platine_about_poteau", nom_plat="platine", E=EPB, calculer=MPB.compute, S=YPB, N=NPB, T=TPB,
    methode=MEPB.METHODE, bench=BPB, vues=vues_poteau(SPB), sections=_sections_pb, libelles=LIBELLES,
    affichage=AFFICHAGE, segments=SEGMENTS, formules=lambda R, c: A.lignes_texte(R, c, FPB.textes),
    parametres=_parametres_pb, piece=_piece_pb, non_reverifies=lambda R: tuple(MEPB.limites(R.ame)),
    dimensions=PP.dimensions(EPB), noms_aciers=NOMS_ACIERS,
    titres_profils={"prof_S": "Poutre portée · profilé", "prof_P": "Poteau · profilé"},
    regle_33="e ≥ 1,2·d<sub>0</sub> ; p<sub>1</sub> ≥ 2,2·d<sub>0</sub> ; p<sub>3</sub> (p<sub>2</sub> du tableau) ≥ 2,4·d<sub>0</sub>",
    decimales_bench=_decimales_bench_pb,
    predim={"apply": MPB.apply_solution, "label": MPB.solution_label,
            "nom": lambda r: f"2×{js_str(r.n)} {r.b['n']} · {js_str(r.hp)}×{js_str(r.prop.bp)}×{js_str(r.prop.tp)}",
            "col": lambda r: r.hp, "unite_col": "h<sub>p</sub> mm",
            "colonnes": lambda R: [("h<sub>p</sub> (mm)", lambda r: js_str(r.hp)), ("η boulons", lambda r: pct(r.e.bg)),
                                   ("η âme portée", lambda r: pct(r.e.vwS)),
                                   ("η poteau", lambda r: _eta_poteau_pb(r, R.ame))],
            "detail": detail_predim_pb, "regle": REGLE_PB,
            "geo": lambda r: f"e1 {r.e1} / p1 {r.p1} / p3 {r.p3} mm"}))

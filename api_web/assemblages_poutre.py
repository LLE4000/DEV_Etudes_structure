# -*- coding: utf-8 -*-
"""La famille POUTRE–POUTRE du module « Assemblages métalliques » du site : plat d'âme soudé, doubles cornières
d'âme, platine d'about (hauteur partielle). Fichier d'AIDE (sans MODULE) : il décrit les trois configurations à la
carte commune (api_web/assemblages.py) — leurs sections de données, leurs libellés courts, leurs pièces, leurs
formules, leurs dessins, leur prédimensionnement.

Les moteurs sont ceux du dépôt (acier/assemblages/poutre_poutre/<config>/ : entrees, moteur, synthese, formules,
notation, texte, methode, benchmark, schemas), importés SANS leurs fichiers d'écran. Les quelques textes de l'écran
Streamlit qu'on ne peut pas importer (ils vivent dans ecran_resultats.py, qui charge streamlit) sont des COPIES
NOMMÉES, liées à l'original par tests/test_api_web_assemblages.py : ``NON_REVERIFIES_*`` et les « paramètres
retenus ».
"""
from acier.assemblages.poutre_poutre.doubles_cornieres import (benchmark as BDC, entrees as EDC, formules as FDC,
                                                              methode as MEDC, moteur as MDC, notation as NDC,
                                                              schemas as SDC, synthese as YDC, texte as TDC)
from acier.assemblages.poutre_poutre.plat_ame import (benchmark as BPA, entrees as EPA, formules as FPA, methode as MEPA,
                                                     moteur as MPA, notation as NPA, schemas as SPA, synthese as YPA,
                                                     texte as TPA)
from acier.assemblages.poutre_poutre.platine_about import (benchmark as BPB, entrees as EPB, formules as FPB,
                                                          methode as MEPB, moteur as MPB, notation as NPB, schemas as SPB,
                                                          synthese as YPB, texte as TPB)
from acier.bibliotheques import DB, ORI_P, ORI_S, PERSO, find
from acier.formats import F, pct
from acier.js import N, js_str

from . import assemblages as A
from . import commun as C

# ------------------------------------------------------------------ copies nommées de l'écran Streamlit
# (ecran_resultats.NON_REVERIFIES de chaque configuration ; le test les relit dans le source et les compare)
NON_REVERIFIES_PA = ("flexion HORS PLAN de l'âme porteuse sous VEd·z et NEd (attache d'un seul côté — négligée par "
                     "les guides pour les détails courants)",
                     "cisaillement global, flexion et déversement de la poutre principale",
                     "torsion induite", "efforts d'arrachement accidentels (tying)",
                     "deux poutres en vis-à-vis sur la même âme")
NON_REVERIFIES_DC = ("cisaillement global, flexion et déversement de la poutre principale",
                     "torsion induite par une attache d'un seul côté", "flexion hors plan de l'âme sous NEd",
                     "deux poutres en vis-à-vis partageant les mêmes boulons")
NON_REVERIFIES_PB = ("flexion TRANSVERSALE de l'âme porteuse sous la traction des boulons (NEd, tying)",
                     "capacité de rotation au-delà des règles de détail",
                     "cisaillement global, flexion et déversement de la poutre principale",
                     "torsion induite", "deux poutres en vis-à-vis sur la même âme")


# ------------------------------------------------------------------ libellés courts (symbole, libellé)
# Le libellé LONG et l'aide de chaque champ restent ceux de l'écran Streamlit (entrees.GROUPES) : ils s'affichent
# au survol du « i » ; le cadre Données porte une forme courte, qui tient sur sa ligne.
def _s(x):
    return x


LIBELLES = {
    "id_projet": ("", "Projet"), "id_rep": ("", "Repère"), "id_red": ("", "Rédacteur"), "id_date": ("", "Date"),
    "mode_calc": ("", "Mode"),
    "prof_P": ("", "Profilé"), "nu_P": ("", "Nuance"), "prof_S": ("", "Profilé"), "nu_S": ("", "Nuance"),
    "hP_u": ("h", "hauteur"), "bP_u": ("b", "largeur"), "twP_u": ("t<sub>w</sub>", "âme"), "tfP_u": ("t<sub>f</sub>", "semelle"),
    "rP_u": ("r", "congé"),
    "hS_u": ("h", "hauteur"), "bS_u": ("b", "largeur"), "twS_u": ("t<sub>w</sub>", "âme"), "tfS_u": ("t<sub>f</sub>", "semelle"),
    "rS_u": ("r", "congé"),
    "d_top": ("Δz", "décalage"), "g_h": ("g<sub>h</sub>", "jeu à l'about"),
    "d_nt": ("d<sub>c,sup</sub>", "en haut"), "d_nb": ("d<sub>c,inf</sub>", "en bas"), "l_n": ("c", "longueur"),
    "r_n": ("r", "rayon (exécution)"), "lt_ok": ("", "Secondaire maintenue au déversement"),
    "hp_u": ("h<sub>p</sub>", "hauteur"), "tp_u": ("t<sub>p</sub>", "épaisseur"), "bp_u": ("b<sub>p</sub>", "largeur"),
    "z_C": ("z<sub>p</sub>", "sous le dessus"), "nu_pl": ("", "Nuance"), "a_w": ("a", "gorge (double)"),
    "boulon_u": ("", "Boulon"), "classe": ("", "Classe"), "trou": ("", "Trou"), "d0_u": ("d<sub>0</sub>", "trou imposé"),
    "filet": ("", "Filetage dans le plan de cisaillement"), "cat": ("", "Catégorie"),
    "mu_s": ("μ", "frottement"), "k_s": ("k<sub>s</sub>", "trou (Tab. 3.6)"), "k_ser": ("", "ELS / ELU"),
    "n1_u": ("n<sub>1</sub>", "rangées"), "n2_u": ("n<sub>2</sub>", "files"), "p1_u": ("p<sub>1</sub>", "entraxe vertical"),
    "p2_u": ("p<sub>2</sub>", "entraxe horizontal"), "e1_u": ("e<sub>1</sub>", "pince haute"),
    "e2b_u": ("e<sub>2,b</sub>", "about → 1re file"), "p3_u": ("p<sub>3</sub>", "entraxe des files"),
    "V_Ed": ("V<sub>Ed</sub>", "tranchant"), "N_Ed": ("N<sub>Ed</sub>", "normal"),
    "M_Ed": ("M<sub>Ed</sub>", "moment"), "H_Ed": ("H<sub>Ed</sub>", "horizontal"), "nat_N": ("", "Nature"),
    "g_M0": ("γ<sub>M0</sub>", "sections"), "g_M1": ("γ<sub>M1</sub>", "instabilités"),
    "g_M2": ("γ<sub>M2</sub>", "boulons"), "g_M2n": ("γ<sub>M2</sub>", "sections nettes"),
    "g_M3": ("γ<sub>M3</sub>", "ELU"), "g_M3s": ("γ<sub>M3,ser</sub>", "ELS"),
    "g_Mu": ("γ<sub>Mu</sub>", "tying"),
    "opt_blf": ("", "Appliquer β<sub>Lf</sub> (joints longs)"), "expo": ("", "Assemblage exposé aux intempéries"),
    "fy_u": ("f<sub>y</sub>", "limite élastique"), "fu_u": ("f<sub>u</sub>", "résistance"), "bw_u": ("β<sub>w</sub>", "corrélation"),
    "eta_c": ("η", "taux cible"), "k_e1": ("e<sub>1</sub>", "= k · d<sub>0</sub>"), "k_p1": ("p<sub>1</sub>", "= k · d<sub>0</sub>"),
    "k_e2": ("e<sub>2</sub>", "= k · d<sub>0</sub>"), "pd_dmin": ("", "Ø minimal"), "pd_dmax": ("", "Ø maximal"),
    # doubles cornières
    "fix_P": ("", "Ailes A"), "fix_S": ("", "Ailes B"), "corn_u": ("", "Cornière"), "nu_C": ("", "Nuance"),
    "orient": ("", "Grande aile"), "k1_u": ("", "grande aile"), "k2_u": ("", "petite aile"), "kt_u": ("t<sub>c</sub>", "épaisseur"),
    "kr_u": ("r<sub>c</sub>", "congé"), "LC_u": ("h<sub>c</sub>", "hauteur"), "z_C@dc": ("z<sub>c</sub>", "sous le dessus"),
    "n1S_u": ("n<sub>1</sub>", "rangées"), "n2S_u": ("n<sub>2</sub>", "files"), "p1S_u": ("p<sub>1</sub>", "entraxe vertical"),
    "p2_S": ("p<sub>2</sub>", "entraxe horizontal"), "e1S_u": ("e<sub>1</sub>", "pince haute"),
    "n1P_u": ("n<sub>1</sub>", "rangées"), "n2P_u": ("n<sub>2</sub>", "files"), "p1P_u": ("p<sub>1</sub>", "entraxe vertical"),
    "p2_P": ("p<sub>2</sub>", "entraxe horizontal"), "e1P_u": ("e<sub>1</sub>", "pince haute"),
    "gA_u": ("g<sub>A</sub>", "trusquinage"),
    "a_P": ("a<sub>A</sub>", "gorge ailes A"), "lh_P": ("ℓ<sub>h,A</sub>", "retours ailes A"),
    "a_S": ("a<sub>B</sub>", "gorge ailes B"), "lh_S": ("ℓ<sub>h,B</sub>", "retours ailes B"),
    "opt_exc": ("", "Excentricité g<sub>A</sub> reprise par le groupe P"), "k_rot": ("k<sub>rot</sub>", "groupe P"),
}
# « Prédimensionnement » ne tient pas dans un segment de la colonne Données (mesuré : 145 px pour 150 px offerts,
# texte coupé) : le segment dit « Prédim. », comme l'onglet Prédim de l'écran Streamlit ; l'aide dit le mot entier.
AFFICHAGE = {"mode_calc": {EPA.MODE_VERIF: "Vérification", EPA.MODE_PREDIM: "Prédim."},
             "orient": {ORI_P: "côté principale", ORI_S: "côté secondaire"},
             "nat_N": {EPB.NAT_ELU: "ELU", EPB.NAT_TYING: "Tying"}}
SEGMENTS = frozenset({"mode_calc", "fix_P", "fix_S", "n2_u", "n2S_u", "n2P_u", "trou", "cat", "nat_N", "orient"})


def libelles(**surcharges):
    d = dict(LIBELLES)
    d.update(surcharges)
    return d


# ------------------------------------------------------------------ outils communs aux trois
def dimensions(E, corniere=False):
    """Les valeurs PROVISOIRES des dimensions vides (``ecran_saisie.DIMENSIONS`` : celles des profilés par défaut,
    sans le congé ; la cornière par défaut pour les doubles cornières)."""
    d = E.defaults()
    out = {}
    for X in ("P", "S"):
        p = find(DB["profils"], d["prof_" + X])
        out.update({f"h{X}_u": p["h"], f"b{X}_u": p["b"], f"tw{X}_u": p["tw"], f"tf{X}_u": p["tf"]})
    if corniere:
        c = find(DB["cornieres"], d["corn_u"])
        out.update({"k1_u": c["a1"], "k2_u": c["a2"], "kt_u": c["t"]})
    return out


def vues_poutre(S):
    """Les trois vues du moteur (élévation, plan, vue de droite) avec les options de l'écran Streamlit, sans
    l'interactivité (le site ne modifie pas une cote sur le dessin : les cotes se saisissent dans les Données)."""
    def vues(R, niveau, hl):
        opt = S.options_ecran(R, niveau, hl)
        opt.interactive, opt.poignees, opt.editables, opt.locked = False, False, niveau >= 1, None
        u = R.u
        return [("elevation", "Élévation", "Élévation", S.elevation(R, opt), "rotule à la face de l'âme porteuse"),
                ("plan", "Vue en plan", "Plan", S.plan(R, opt), f"{u.prof_S} sur {u.prof_P}"),
                ("droite", "Vue de droite", "Droite", S.vue_droite(R, opt), "âme de la poutre porteuse")]
    return vues


def profil(u, X):
    return "personnalisé" if u.get("prof_" + X) == PERSO else str(u.get("prof_" + X))


def resume_poutres(u):
    return f"{profil(u, 'S')} sur {profil(u, 'P')}"


def resume_grugeage(u):
    if not (N(u.get("d_nt")) or N(u.get("d_nb"))):
        return "non grugée"
    return f"d<sub>c</sub> {C.nb(N(u.get('d_nt')))} / {C.nb(N(u.get('d_nb')))} · c {C.nb(N(u.get('l_n')))}"


def resume_efforts(u, cles=("V_Ed", "N_Ed", "M_Ed", "H_Ed")):
    noms = {"V_Ed": "V", "N_Ed": "N", "M_Ed": "M", "H_Ed": "H"}
    out = [f"{noms[k]} {C.nb(N(u.get(k)))}" for k in cles if k in u and (k == "V_Ed" or N(u.get(k)))]
    return " · ".join(out) + " kN"


def resume_assemblage(cfg, u):
    m = AFFICHAGE["mode_calc"].get(u.get("mode_calc"), "")
    return f"{C.esc(cfg.court)} · {m.lower()}"


def predim_items(u):
    if u.get("mode_calc") == EPA.MODE_VERIF:
        return []
    return [("grp", "Prédimensionnement", "Les champs verrouillés sont pilotés par la proposition ; la ligne « Proposition » du "
                                          "résultat la recopie dans les données et repasse en vérification."),
            "eta_c", "pd_dmin", "pd_dmax", "k_e1", "k_p1", "k_e2"]


def boulons_items(u):
    return ["boulon_u", "classe", "trou", "cat", "filet"] + A.d0_items(u) + ["mu_s", "k_s", "k_ser"]


def coefficients(*cles):
    return list(cles) + [("grp", "Acier personnalisé"), "fy_u", "fu_u", "bw_u"]


def piece_profil(u, X):
    return C.esc(profil(u, X).replace("personnalisé", "Personnalisé")), "prof_" + X


def boulons_texte(R, u, n):
    return C.esc(f"{n} {R.boulon} · {u.get('classe')}")


# ------------------------------------------------------------------ l'onglet Prédim : copies nommées
# Le tableau « Détail de la solution retenue par défaut » et la légende de l'onglet Prédim de chaque écran
# Streamlit (ecran_resultats.onglet_predim) ; le test les relit en faisant tourner onglet_predim avec un faux « st ».
def detail_predim_pa(p):
    return [["Boulon proposé", p.boulon], ["Rangées n1 (1 file)", js_str(p.n)],
            ["e1 / p1 / e2 = e2,b", f"{p.e1} / {p.p1} / {p.e2} mm"],
            ["Hauteur du plat hp = 2·e1 + (n1 − 1)·p1", f"{p.hp} mm (hauteur disponible " + F(p.hav, 0) + " mm)"],
            ["Épaisseur nécessaire au taux cible", F(p.treq, 2) + " mm"],
            ["Plat proposé", js_str(p.hp) + " × " + js_str(p.bp) + " × " + js_str(p.tp) + " mm – gorge a = " + js_str(p.a) + " mm"],
            ["Taux estimé (boulons et pression diamétrale de l'âme)", pct(p.eta, 1)], ["État", p.msg]]


def detail_predim_dc(p):
    return [["Boulon proposé", p.boulon], ["Rangées n1 (groupes S et P, 1 file)", p.n],
            ["e1 / p1 / e2 = e2,b", f"{p.e1} / {p.p1} / {p.e2} mm"],
            ["Hauteur des cornières hc = 2·e1 + (n1 − 1)·p1", f"{p.LC} mm (hauteur disponible " + F(p.hav, 0) + " mm)"],
            ["Épaisseur nécessaire au taux cible", F(p.treq, 2) + " mm"],
            ["Cornière proposée", f"L{p.leg}x{p.leg}x{p.tC} – trusquinage gA = {p.gA} mm"],
            ["Taux estimé (boulons et pressions diamétrales des âmes)", pct(p.eta, 1)], ["État", p.msg]]


def detail_predim_pb(p):
    return [["Boulon proposé", p.boulon], ["Rangées n1 (2 files)", js_str(p.n)],
            ["e1 / p1 / e2", f"{p.e1} / {p.p1} / {p.e2} mm"],
            ["Entraxe des files p3 (≥ 90 mm, rondelle hors cordon)", f"{p.p3} mm"],
            ["Hauteur de la platine hp = max(2·e1 + (n1 − 1)·p1 ; 0,6·h)", f"{p.hp} mm (hauteur disponible " + F(p.hav, 0) + " mm)"],
            ["Épaisseur nécessaire au taux cible", F(p.treq, 2) + " mm"],
            ["Platine proposée", js_str(p.hp) + " × " + js_str(p.bp) + " × " + js_str(p.tp) + " mm – gorge a = " + js_str(p.a) + " mm"],
            ["Taux estimé (boulons, âme portée, âme porteuse)", pct(p.eta, 1)], ["État", p.msg]]


# la légende des calculs manuels de l'onglet Benchmark des doubles cornières (sa colonne de contrôle est un classeur)
LEGENDE_MANUEL_DC = ("Colonne « manuel » : calcul indépendant hors moteur (intégration numérique pour les cordons). "
                     "Colonne « Excel » : classeur recalculé avec les mêmes données.", "Excel")

REGLE_PA = ("Règle de choix par défaut : le plus petit nombre de rangées ; à égalité, le plus petit diamètre. L'épaisseur du "
            "plat est plafonnée à 0,5·d (ductilité) ; la gorge proposée est la pleine résistance. Seule la vérification "
            "détaillée fait foi.")
REGLE_DC = ("Règle de choix par défaut : le plus petit nombre de rangées ; à égalité, le plus petit diamètre. MEd est ajouté "
            "à VEd·z ; les tractions dues à HEd et MEd ne sont prises en compte que dans la vérification détaillée.")
REGLE_PB = ("Règle de choix par défaut : le plus petit nombre de rangées ; à égalité, le plus petit diamètre. L'épaisseur de "
            "la platine reste entre 8 mm et la limite de ductilité ; la gorge proposée est la pleine résistance du guide. "
            "Seule la vérification détaillée fait foi.")


# ================================================================== plat d'âme soudé
def _sections_pa(u, R, el):
    cfg = A.CONFIGS["plat_ame"]
    rp = R or {}
    return [
        ("as", "Assemblage", resume_assemblage(cfg, u), ["id_rep", "mode_calc"] + predim_items(u), True),
        ("pt", "Poutres", resume_poutres(u),
         [("grp", "Principale (porteuse)"), "prof_P", "nu_P", "hP_u", "bP_u", "twP_u", "tfP_u", "rP_u",
          ("grp", "Secondaire (portée)"), "prof_S", "nu_S", "hS_u", "bS_u", "twS_u", "tfS_u", "rS_u",
          ("grp", "Position"), "d_top", "g_h"], True),
        ("gr", "Grugeage", resume_grugeage(u), ["d_nt", "d_nb", "l_n", "r_n", "lt_ok"], False),
        ("pl", "Plat d'âme", f"{C.nb(N(u.get('hp_u')))} × {C.nb(N(u.get('bp_u')))} × {C.nb(N(u.get('tp_u')))} · a {C.nb(N(u.get('a_w')))}",
         ["hp_u", "tp_u", "bp_u", "z_C", "nu_pl", ("grp", "Soudure sur l'âme porteuse"), "a_w"], False),
        ("bo", "Boulons", f"{C.nb(N(u.get('n1_u')))} × {u.get('n2_u')} {u.get('boulon_u')} {u.get('classe')}",
         boulons_items(u) + [("grp", "Groupe · plat et âme de la portée")] + ["n1_u", "n2_u", "p1_u", "p2_u", "e1_u", "e2b_u"], True),
        ("ef", "Efforts ELU", resume_efforts(u, ("V_Ed", "N_Ed", "M_Ed")), ["V_Ed", "N_Ed", "M_Ed"], True),
        ("co", "Coefficients et modèle", f"γ<sub>M1</sub> {C.nb(N(u.get('g_M1')))} · ANB",
         coefficients("g_M0", "g_M1", "g_M2", "g_M2n", "g_M3", "g_M3s", ("grp", "Modèle"), "opt_blf", "expo"), False),
        ("ca", "Cartouche et affichage", C.esc(u.get("id_red") or "rédacteur à saisir"), A.section_cartouche(cfg, u, el), False),
    ]


def _parametres_pa(R):
    """Les paramètres retenus de l'écran Streamlit (plat_ame/ecran_resultats._parametres_retenus) — copie nommée."""
    return [["Boulon retenu", R.boulon + " – d0 = " + F(R.d_0, 0) + " mm"],
            ["Plat", js_str(R.h_p) + " × " + js_str(R.b_p) + " × " + js_str(R.t_p) + " mm – " + str(R.u.nu_pl)],
            ["Excentricité z : face de l'âme porteuse → centre du groupe", F(R.zeff, 1) + " mm"],
            ["Moment d'excentricité MS = VEd·z + |MEd|", F(R.M_S, 2) + " kNm"],
            ["Modèle", R.mod],
            ["Inertie polaire du groupe", F(R.Ip, 0) + " mm²"],
            ["Plat court / long (limite tp/0,15 = " + F(R.t_p / 0.15, 1) + " mm)",
             ("LONG (déversement vérifié)" if R.long_p else "court")],
            ["Pince e2 du plat (bord libre)", F(R.e2_p, 1) + " mm"],
            ["Pince e1,b dans l'âme secondaire", F(R.e1b_S, 1) + " mm"],
            ["Pince basse du plat", F(R.e1bot, 1) + " mm"],
            ["Gorge de pleine résistance recommandée", F(R.a_plein, 1) + " mm (a = " + F(R.a_w, 0) + " mm)"]]


def _piece_pa(R, u, grp):
    if grp == "Boulons":
        return boulons_texte(R, u, f"{js_str(R.n_1)} × {js_str(R.n_2)}"), ("" if R.pred else "boulon")
    if grp == "Plat d'âme":
        return C.esc(f"{js_str(R.h_p)} × {js_str(R.b_p)} × {js_str(R.t_p)} mm"), ""
    if grp == "Poutre secondaire":
        return piece_profil(u, "S")
    if grp == "Poutre principale":
        return piece_profil(u, "P")
    return C.esc(f"gorge a = {js_str(R.a_w)} mm"), ""


A.enregistrer(A.Config(
    cle="plat_ame", E=EPA, calculer=MPA.compute, S=YPA, N=NPA, T=TPA, methode=MEPA.METHODE, bench=BPA,
    vues=vues_poutre(SPA), sections=_sections_pa, libelles=libelles(), affichage=AFFICHAGE, segments=SEGMENTS,
    formules=lambda R, c: A.lignes_texte(R, c, FPA.textes), parametres=_parametres_pa, piece=_piece_pa,
    non_reverifies=NON_REVERIFIES_PA, dimensions=dimensions(EPA),
    tableaux_boulons=lambda R: [("groupe", R.tab)],
    predim={"apply": MPA.apply_solution, "label": MPA.solution_label,
            "nom": lambda r: f"{js_str(r.n)} {r.b['n']} · plat {js_str(r.hp)}×{js_str(r.prop.tp)}",
            "col": lambda r: r.hp, "unite_col": "h<sub>p</sub> mm",
            "colonnes": [("variante", lambda r: r.var), ("h<sub>p</sub> (mm)", lambda r: js_str(r.hp)),
                         ("η boulons", lambda r: pct(r.e.bS)), ("η p. diam. âme", lambda r: pct(r.e.pdS))],
            "detail": detail_predim_pa, "regle": REGLE_PA,
            "geo": lambda r: f"e1 {r.e1} / p1 {r.p1} / e2 {r.e2} mm"}))


# ================================================================== doubles cornières
def _sections_dc(u, R, el):
    cfg = A.CONFIGS["doubles_cornieres"]
    bolt_S, bolt_P = u.get("fix_S") == EDC.BOULONNEE, u.get("fix_P") == EDC.BOULONNEE
    groupes = []
    if bolt_S:
        groupes += [("grp", "Groupe S · âme de la secondaire (2 plans)"), "n1S_u", "n2S_u", "p1S_u", "p2_S", "e1S_u", "e2b_u"]
    if bolt_P:
        groupes += [("grp", "Groupe P · âme de la principale (par cornière)"), "n1P_u", "n2P_u", "p1P_u", "p2_P", "e1P_u", "gA_u"]
    secs = [
        ("as", "Assemblage", resume_assemblage(cfg, u),
         ["id_rep", "mode_calc",
          ("grp", "Fixation des cornières", "Ailes A sur l'âme de la principale, ailes B sur l'âme de la secondaire. Soudées des "
                                            "deux côtés : assemblage peu déformable, à réserver à des cas particuliers."),
          "fix_P", "fix_S"] + predim_items(u), True),
        ("pt", "Poutres", resume_poutres(u),
         [("grp", "Principale (porteuse)"), "prof_P", "nu_P", "hP_u", "bP_u", "twP_u", "tfP_u", "rP_u",
          ("grp", "Secondaire (portée)"), "prof_S", "nu_S", "hS_u", "bS_u", "twS_u", "tfS_u", "rS_u",
          ("grp", "Position"), "d_top", "g_h"], True),
        ("gr", "Grugeage", resume_grugeage(u), ["d_nt", "d_nb", "l_n", "r_n", "lt_ok"], False),
        ("cn", "Cornières", f"2 {C.esc(u.get('corn_u'))} · h<sub>c</sub> {C.nb(N(u.get('LC_u')))}",
         ["corn_u", "nu_C", "orient", "k1_u", "k2_u", "kt_u", "kr_u", "LC_u", "z_C"], False),
    ]
    if bolt_S or bolt_P:
        secs.append(("bo", "Boulons", f"{u.get('boulon_u')} {u.get('classe')}", boulons_items(u) + groupes, True))
    else:
        # les deux ailes soudées : l'écran Streamlit garde son bloc BOULONS (entrees.champ_visible), et le moteur les
        # lit encore — « Surdimensionné » sans d0 bloque le verdict (alerte d0_manq) ; cachés, on ne pouvait plus le
        # corriger. La section reste, repliée, avec ce qu'elle veut dire.
        secs.append(("bo", "Boulons", "sans objet (ailes soudées)",
                     boulons_items(u) + [("aide", "Les deux ailes sont soudées : aucun boulon n'est vérifié. Ces champs "
                                                 "restent ceux de l'écran Streamlit (un trou « Surdimensionné » sans "
                                                 "d<sub>0</sub> y bloque encore le verdict).")], False))
    if not (bolt_S and bolt_P):
        secs.append(("so", "Soudures", "cordons d'angle", ["a_P", "lh_P", "a_S", "lh_S"], True))
    secs += [
        ("ef", "Efforts ELU", resume_efforts(u), ["V_Ed", "N_Ed", "H_Ed", "M_Ed"], True),
        ("co", "Coefficients et modèle", "γ<sub>M</sub> · options",
         coefficients("g_M0", "g_M2", "g_M2n", "g_M3", "g_M3s", ("grp", "Modèle"), "opt_exc", "k_rot", "opt_blf", "expo"), False),
        ("ca", "Cartouche et affichage", C.esc(u.get("id_red") or "rédacteur à saisir"),
         A.section_cartouche(cfg, u, el, avec_visserie=bolt_S or bolt_P), False),
    ]
    return secs


def _parametres_dc(R):
    """Les paramètres retenus de l'écran Streamlit (doubles_cornieres/ecran_resultats._parametres_retenus)."""
    return [["Boulon retenu", R.boulon + " – d0 = " + F(R.d_0, 0) + " mm"],
            ["Cornières", R.corn_txt + " – aile A " + F(R.b_A, 0) + ", aile B " + F(R.b_B, 0) + ", tc " + F(R.t_C, 0) + ", rc " + F(R.r_C, 0) + " mm"],
            ["Hauteur des cornières hc", F(R.L_C, 0) + " mm" + ("" if R.rec_L else " (hc < 0,6·h : recommandation MSB Part 5 §4.1 non respectée)")],
            ["Excentricité z : face de l'âme porteuse → centre du groupe S " + ("" if R.bolt_S else "(cordons)"), F(R.zeff, 1) + " mm"],
            ["Moment d'excentricité MS = VEd·z + |MEd|", F(R.M_S, 2) + " kNm"],
            ["Modèle du groupe S", R.mod_S], ["Modèle du groupe P", R.mod_P],
            ["Moment par groupe P (option)", F(R.M_P, 2) + " kNm"],
            ["Inertie polaire du groupe S", F(R.Ip_S, 0) + " mm²"],
            ["Entraxe p3 entre les files des deux cornières", F(R.p_3, 1) + " mm"],
            ["Pince e1,b dans l'âme secondaire", F(R.e1b_S, 1) + " mm"],
            ["Pinces e2 : aile B / aile A", F(R.e2a_S, 1) + " / " + F(R.e2a_P, 1) + " mm"],
            ["Pinces basses e1 : groupe S / groupe P", F(R.e1bot_S, 1) + " / " + F(R.e1bot_P, 1) + " mm"]]


def _piece_dc(R, u, grp):
    if grp == "Boulons":
        g = [f"S {js_str(R.n1_S)}×{js_str(R.n2_S)}" if R.bolt_S else "", f"P {js_str(R.n1_P)}×{js_str(R.n2_P)}" if R.bolt_P else ""]
        return C.esc(f"{R.boulon} {u.get('classe')} · " + " · ".join(x for x in g if x)), ("" if R.pred else "boulon")
    if grp == "Cornières":
        return C.esc(f"2 {R.corn_txt} · hc {js_str(R.L_C)}"), ("" if R.pred else "corniere")
    if grp == "Poutre secondaire":
        return piece_profil(u, "S")
    if grp == "Poutre principale":
        return piece_profil(u, "P")
    g = [f"a<sub>A</sub> = {C.nb(N(u.get('a_P')))}" if not R.bolt_P else "", f"a<sub>B</sub> = {C.nb(N(u.get('a_S')))}" if not R.bolt_S else ""]
    return " · ".join(x for x in g if x) + " mm", ""


def _cellule_aile(x):
    """« Ed / Rd unité · η » d'une aile (ecran_resultats._cell_aile de l'écran Streamlit), η en pastille de couleur."""
    if x is None:
        return "—"
    c = x["c"]
    un = f" {C.esc(x['unit'])}" if x["unit"] else ""
    return f'{C.esc(x["Ed"])} / {C.esc(x["Rd"])}{un} {C.chip(A.taux_c(c, 1), A.niveau_c(c))}'


def matrice_cornieres(R, Y):
    """La matrice « Cornières — aile A / aile B » de l'onglet Vérifications (synthese.matrice_cornieres) : une ligne
    par vérification, l'aile A (sur la principale), l'aile B (sur la secondaire) et la plus sollicitée."""
    ls = Y.matrice_cornieres(R)
    if not ls:
        return ""
    lignes = []
    for x in ls:
        refs = Y.refs_fusionnees([c["ref"] for c in (x["A"], x["B"]) if c])
        m = x["max"]["c"]
        lignes.append([f"<b>{C.esc(x['lab'])}</b>", _cellule_aile(x["A"]), _cellule_aile(x["B"]),
                       C.chip(A.taux_c(m, 1), A.niveau_c(m)), C.esc(refs)])
    return ('<div class="ck-g asm-large"><div class="ck-t">Cornières — aile A / aile B</div>'
            + A.tableau(["Vérification", "Aile A : E<sub>d</sub> / R<sub>d</sub> · η", "Aile B : E<sub>d</sub> / R<sub>d</sub> · η",
                         "max", "Réf."], lignes, droite=(1, 2, 3), classe="asm-tab asm-mat")
            + '<div class="asm-note">Aile A sur la poutre principale (groupe P) · aile B sur la poutre secondaire (groupe S).</div></div>')


def _sel_corniere(cfg, u, familles=None):
    """Les cornières du catalogue des assemblages, triées par grandeur (ailes puis épaisseur), avec le taux de
    l'assemblage pour chacune ; l'optimal est la plus petite qui tient."""
    rows = []
    for c in sorted(DB["cornieres"], key=lambda c: (c["a1"], c["a2"], c["t"])):
        R = A._avec(cfg, u, corn_u=c["n"])
        rows.append(A._ligne_sel(c["n"], C.nb(c["a1"]), R, c["n"] == u.get("corn_u"),
                                 f"{c['n']} · ailes {C.nb(c['a1'])} × {C.nb(c['a2'])} · t {C.nb(c['t'])} · r {C.nb(c['r'])} mm"))
    A._marquer_optimal(rows)
    return C.selecteur("corniere", "Cornières · EN 10056-1", "triées par aile puis épaisseur · un clic retient la cornière", "", rows,
                       A._debut(rows), "aile mm", etoile="la plus petite qui tient")


A.enregistrer(A.Config(
    cle="doubles_cornieres", E=EDC, calculer=MDC.compute, S=YDC, N=NDC, T=TDC, methode=MEDC.METHODE, bench=BDC,
    vues=vues_poutre(SDC), sections=_sections_dc, libelles=libelles(z_C=LIBELLES["z_C@dc"]), affichage=AFFICHAGE,
    segments=SEGMENTS, formules=lambda R, c: A.lignes_sub(R, c, FDC), parametres=_parametres_dc, piece=_piece_dc,
    non_reverifies=NON_REVERIFIES_DC, dimensions=dimensions(EDC, corniere=True),
    tableaux_boulons=lambda R: ([("groupe S", R.tabS)] if R.bolt_S else []) + ([("groupe P (une cornière)", R.tabP)] if R.bolt_P and R.M_P > 0 else []),
    selecteurs={"corniere": {"rendre": _sel_corniere, "choisir": A._choisir_liste("corn_u", "cornieres")}},
    blocs=lambda R, grp: matrice_cornieres(R, YDC) if grp == "Cornières" else "", legende_manuel=LEGENDE_MANUEL_DC,
    predim={"apply": MDC.apply_solution, "label": MDC.solution_label,
            "nom": lambda r: f"{js_str(r.n)} {r.b['n']} · {MDC.nom_cornière_proposee(r)}",
            "col": lambda r: r.Lc, "unite_col": "h<sub>c</sub> mm",
            "colonnes": [("h<sub>c</sub> (mm)", lambda r: js_str(r.Lc)), ("Cornière", MDC.nom_cornière_proposee),
                         ("η boulons S", lambda r: pct(r.e.bS)), ("η p. diam. âme S", lambda r: pct(r.e.pdS)),
                         ("η boulons P", lambda r: pct(r.e.bP)), ("η p. diam. âme P", lambda r: pct(r.e.pdP))],
            "detail": detail_predim_dc, "regle": REGLE_DC,
            "geo": lambda r: f"e1 {r.e1} / p1 {r.p1} / e2 {r.e2} / gA {r.prop.gA} mm"}))


# ================================================================== platine d'about
def _sections_pb(u, R, el):
    cfg = A.CONFIGS["platine_about"]
    return [
        ("as", "Assemblage", resume_assemblage(cfg, u), ["id_rep", "mode_calc"] + predim_items(u), True),
        ("pt", "Poutres", resume_poutres(u),
         [("grp", "Principale (porteuse)"), "prof_P", "nu_P", "hP_u", "bP_u", "twP_u", "tfP_u", "rP_u",
          ("grp", "Secondaire (portée)"), "prof_S", "nu_S", "hS_u", "bS_u", "twS_u", "tfS_u", "rS_u",
          ("grp", "Position"), "d_top"], True),
        ("gr", "Grugeage", resume_grugeage(u), ["d_nt", "d_nb", "l_n", "r_n", "lt_ok"], False),
        ("pl", "Platine d'about", f"{C.nb(N(u.get('hp_u')))} × {C.nb(N(u.get('bp_u')))} × {C.nb(N(u.get('tp_u')))} · a {C.nb(N(u.get('a_w')))}",
         ["hp_u", "tp_u", "bp_u", "z_C", "nu_pl", ("grp", "Soudure sur l'âme de la portée"), "a_w"], False),
        ("bo", "Boulons", f"2 × {C.nb(N(u.get('n1_u')))} {u.get('boulon_u')} {u.get('classe')}",
         boulons_items(u) + [("grp", "Groupe · 2 files, 1 plan")] + ["n1_u", "p1_u", "e1_u", "p3_u"], True),
        ("ef", "Efforts ELU", resume_efforts(u, ("V_Ed", "N_Ed")), ["V_Ed", "N_Ed", "nat_N"], True),
        ("co", "Coefficients et modèle", "γ<sub>M</sub> · options",
         coefficients("g_M0", "g_M2", "g_M2n", "g_Mu", "g_M3", "g_M3s", ("grp", "Modèle"), "opt_blf", "expo"), False),
        ("ca", "Cartouche et affichage", C.esc(u.get("id_red") or "rédacteur à saisir"), A.section_cartouche(cfg, u, el), False),
    ]


def _parametres_pb(R):
    """Les paramètres retenus de l'écran Streamlit (platine_about/ecran_resultats._parametres_retenus)."""
    g = [["Boulon retenu", R.boulon + " – d0 = " + F(R.d_0, 0) + " mm – " + js_str(R.n_1) + " rangées × 2 files"],
         ["Platine", js_str(R.h_p) + " × " + js_str(R.b_p) + " × " + js_str(R.t_p) + " mm – " + str(R.u.nu_pl)],
         ["Modèle", R.mod],
         ["Groupe de boulons : branche retenue", R.branche_g + " = " + F(R.FRd_g, 1) + " kN"],
         ["Effort par boulon Fv,Ed = VEd/n", F(R.Fv_Ed, 2) + " kN"],
         ["Pince latérale e2 = (bp − p3)/2", F(R.e_2, 1) + " mm"],
         ["Pinces verticales e1 (haut) / e1' (bas)", F(R.e_1, 1) + " / " + F(R.e1bot, 1) + " mm"],
         ["Entraxe minimal des files (rondelle hors cordon)", F(R.p3_min, 1) + " mm (p3 = " + F(R.p_3, 0) + " mm)"],
         ["hp ≥ 1,36·p3 (flexion de la platine négligeable)", ("oui" if R.hp_136 else "non") + " → k bloc = " + F(R.k_bloc, 1)],
         ["Ductilité : tp,max = 0,36·d·√(fub/fy)", F(R.tp_max, 1) + " mm (tp = " + F(R.t_p, 0) + " mm)"],
         ["Gorge de pleine résistance recommandée", F(R.k_plein, 3) + "·tw = " + F(R.a_plein, 1) + " mm (a = " + F(R.a_w, 1) + " mm)"]]
    if N(R.u.N_Ed) > 0:
        g += [["Tronçon en T : m / n / ew", F(R.m_T, 1) + " / " + F(R.nT, 1) + " / " + F(R.ew_T, 2) + " mm"],
              ["Tronçon en T : Σleff", F(R.leff_T, 0) + " mm"]]
    return g


def _piece_pb(R, u, grp):
    if grp == "Boulons":
        return boulons_texte(R, u, f"2 × {js_str(R.n_1)}"), ("" if R.pred else "boulon")
    if grp == "Platine":
        return C.esc(f"{js_str(R.h_p)} × {js_str(R.b_p)} × {js_str(R.t_p)} mm"), ""
    if grp == "Poutre secondaire":
        return piece_profil(u, "S")
    if grp == "Poutre principale":
        return piece_profil(u, "P")
    return C.esc(f"gorge a = {js_str(R.a_w)} mm"), ""


A.enregistrer(A.Config(
    cle="platine_about", nom_plat="platine", E=EPB, calculer=MPB.compute, S=YPB, N=NPB, T=TPB, methode=MEPB.METHODE, bench=BPB,
    vues=vues_poutre(SPB), sections=_sections_pb, libelles=libelles(z_C=("z<sub>p</sub>", "sous le dessus")),
    affichage=AFFICHAGE, segments=SEGMENTS, formules=lambda R, c: A.lignes_texte(R, c, FPB.textes),
    parametres=_parametres_pb, piece=_piece_pb, non_reverifies=NON_REVERIFIES_PB, dimensions=dimensions(EPB),
    predim={"apply": MPB.apply_solution, "label": MPB.solution_label,
            "nom": lambda r: f"2×{js_str(r.n)} {r.b['n']} · {js_str(r.hp)}×{js_str(r.prop.bp)}×{js_str(r.prop.tp)}",
            "col": lambda r: r.hp, "unite_col": "h<sub>p</sub> mm",
            "colonnes": [("h<sub>p</sub> (mm)", lambda r: js_str(r.hp)), ("η boulons", lambda r: pct(r.e.bv)),
                         ("η âme portée", lambda r: pct(r.e.vwS)), ("η p. diam. P", lambda r: pct(r.e.pdP))],
            "detail": detail_predim_pb, "regle": REGLE_PB,
            "geo": lambda r: f"e1 {r.e1} / p1 {r.p1} / p3 {r.p3} mm"}))

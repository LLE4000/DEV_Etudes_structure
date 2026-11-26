# -*- coding: utf-8 -*-
"""La famille POUTRE–POUTRE du module « Assemblages métalliques » du site : plat d'âme soudé, doubles cornières
d'âme, platine d'about (hauteur partielle). Fichier d'AIDE (sans MODULE) : il décrit les trois configurations à la
carte commune (api_web/assemblages.py) — leurs sections de données, leurs libellés courts, leurs pièces, leurs
formules, leurs dessins, les mots de leurs propositions.

Les moteurs sont ceux du dépôt (acier/assemblages/poutre_poutre/<config>/ : entrees, moteur, synthese, formules,
notation, benchmark, schemas), importés SANS leurs fichiers d'écran. Les quelques textes de l'écran Streamlit qu'on ne
peut pas importer (ils vivent dans ecran_resultats.py, qui charge streamlit) sont des COPIES NOMMÉES, liées à
l'original par tests/test_api_web_assemblages.py : ``NON_REVERIFIES_*`` et les « paramètres retenus ».

Les rubriques du cadre Données (revue du 25/11/2026) : Assemblage · Poutres (les deux poutres, leur position ET le
grugeage : « comme ça on définit un peu la géométrie ») · la pièce d'attache · Boulons · [Soudures] · Efforts ELU ·
Options.
"""
from acier.assemblages.poutre_poutre.doubles_cornieres import (benchmark as BDC, entrees as EDC, formules as FDC,
                                                              moteur as MDC, notation as NDC, schemas as SDC,
                                                              synthese as YDC)
from acier.assemblages.poutre_poutre.plat_ame import (benchmark as BPA, entrees as EPA, formules as FPA, moteur as MPA,
                                                     notation as NPA, schemas as SPA, synthese as YPA)
from acier.assemblages.poutre_poutre.platine_about import (benchmark as BPB, entrees as EPB, formules as FPB,
                                                          moteur as MPB, notation as NPB, schemas as SPB,
                                                          synthese as YPB)
from acier.bibliotheques import DB, ORI_P, ORI_S, PERSO, find
from acier.formats import F, nb
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
LIBELLES = {
    "id_projet": ("", "Projet"), "id_rep": ("", "Repère"), "id_red": ("", "Rédacteur"), "id_date": ("", "Date"),
    "prof_P": ("", "Profilé"), "nu_P": ("", "Nuance"), "prof_S": ("", "Profilé"), "nu_S": ("", "Nuance"),
    "hP_u": ("h", "hauteur"), "bP_u": ("b", "largeur"), "twP_u": ("t<sub>w</sub>", "âme"), "tfP_u": ("t<sub>f</sub>", "semelle"),
    "rP_u": ("r", "congé"),
    "hS_u": ("h", "hauteur"), "bS_u": ("b", "largeur"), "twS_u": ("t<sub>w</sub>", "âme"), "tfS_u": ("t<sub>f</sub>", "semelle"),
    "rS_u": ("r", "congé"),
    "d_top": ("Δz", "décalage des dessus"), "g_h": ("g<sub>h</sub>", "jeu à l'about"),
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
    "k_e2": ("e<sub>2</sub>", "= k · d<sub>0</sub>"), "pd_dmin": ("", "Ø min"), "pd_dmax": ("", "Ø max"),
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
AFFICHAGE = {"orient": {ORI_P: "côté principale", ORI_S: "côté secondaire"},
             "nat_N": {EPB.NAT_ELU: "ELU", EPB.NAT_TYING: "Tying"}}
SEGMENTS = frozenset({"fix_P", "fix_S", "n2_u", "n2S_u", "n2P_u", "trou", "cat", "nat_N", "orient"})


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
    """Les trois vues du moteur (élévation, plan, vue de droite) avec les options de l'écran Streamlit, interactivité
    comprise : le kit écrit sur chaque objet ce qu'il règle (cote, pièce, poignée d'un groupe de boulons), et
    ``assemblages.vues`` le traduit pour le site (``dessin_assemblages.manipulable``). Aucune cote verrouillée :
    l'écran du site est toujours en vérification."""
    def vues(R, niveau, hl):
        opt = S.options_ecran(R, niveau, hl)
        opt.interactive, opt.poignees, opt.editables, opt.locked = True, True, niveau >= 1, None
        u = R.u
        return [("elevation", "Élévation", "Élévation", S.elevation(R, opt), "rotule à la face de l'âme porteuse"),
                ("plan", "Vue en plan", "Plan", S.plan(R, opt), f"{u.prof_S} sur {u.prof_P}"),
                ("droite", "Vue de droite", "Droite", S.vue_droite(R, opt), "âme de la poutre porteuse")]
    return vues


def profil(u, X):
    return "personnalisé" if u.get("prof_" + X) == PERSO else str(u.get("prof_" + X))


def grugee(u) -> bool:
    return bool(N(u.get("d_nt")) or N(u.get("d_nb")))


def resume_poutres(u):
    """« HEA 300 sur HEA 400 · grugée » : les deux poutres, et le grugeage quand il y en a un (texte)."""
    return f"{profil(u, 'S')} sur {profil(u, 'P')}" + (" · grugée" if grugee(u) else "")


def resume_efforts(u, cles=("V_Ed", "N_Ed", "M_Ed", "H_Ed")):
    noms = {"V_Ed": "V", "N_Ed": "N", "M_Ed": "M", "H_Ed": "H"}
    out = [f"{noms[k]} {C.nb(N(u.get(k)))}" for k in cles if k in u and (k == "V_Ed" or N(u.get(k)))]
    return " · ".join(out) + " kN"


def boulons_items(u):
    return [("duo", "boulon_u", "classe"), "trou", "cat", "filet"] + A.d0_items(u) + [("duo", "mu_s", "k_s"), "k_ser"]


def items_poutres(*position):
    """Les deux poutres, chacune avec son acier sur UNE ligne (« Principale HEA 400 · S355 ») ; leurs dimensions quand
    le profilé est « Personnalisé », deux par ligne ; la position ; puis le GRUGEAGE de la secondaire (le propriétaire,
    25/11/2026 : « poutre et grugeage, ça pourrait être regroupé ensemble […] comme ça on définit un peu la géométrie »)."""
    return ([("profil", "Principale", "prof_P", "nu_P"), ("duo", "hP_u", "bP_u"), ("duo", "twP_u", "tfP_u"), "rP_u",
             ("profil", "Secondaire", "prof_S", "nu_S"), ("duo", "hS_u", "bS_u"), ("duo", "twS_u", "tfS_u"), "rS_u"]
            + list(position) + [("grp", "Grugeage de la secondaire", AIDE_GRUGEAGE)] + GRUGEAGE)


GRUGEAGE = [("duo", "d_nt", "d_nb"), ("duo", "l_n", "r_n"), "lt_ok"]
AIDE_GRUGEAGE = "Profondeur en haut et en bas, longueur c, rayon d'exécution. 0 = semelle non grugée."


def coefficients(*cles):
    """Les premiers items de la rubrique « Options » : les coefficients partiels, le modèle, l'acier personnalisé."""
    return [("grp", "Coefficients partiels")] + list(cles) + [("grp", "Acier personnalisé"), "fy_u", "fu_u", "bw_u"]


def piece_profil(u, X):
    return C.esc(profil(u, X).replace("personnalisé", "Personnalisé")), "prof_" + X


def boulons_texte(R, u, n):
    return C.esc(f"{n} {R.boulon} · {u.get('classe')}")


# ------------------------------------------------------------------ les propositions : leurs mots
# Une proposition du prédimensionnement (acier/assemblages/propositions.py) se dit depuis SES saisies : en quelques mots
# dans la liste (``nom_*``), puis sa géométrie sous la liste (``detail_*``). Rien n'est recalculé ici.
def _i(v, k) -> str:
    return js_str(int(N(v.get(k))))


def nom_plat(v) -> str:
    """« 3 M20 · plat 190×80×8 » ; deux files : « 3×2 M20 · … »."""
    files = "×2" if int(N(v.get("n2_u"))) == 2 else ""
    return f"{_i(v, 'n1_u')}{files} {v.get('boulon_u')} · plat {nb(N(v.get('hp_u')))}×{nb(N(v.get('bp_u')))}×{nb(N(v.get('tp_u')))}"


def detail_plat(v) -> str:
    p2 = f" / p2 {nb(N(v.get('p2_u')))}" if int(N(v.get("n2_u"))) == 2 else ""
    return (f"{_i(v, 'n1_u')} rangées × {_i(v, 'n2_u')} file{'s' if int(N(v.get('n2_u'))) > 1 else ''} {v.get('boulon_u')} "
            f"{v.get('classe')} · plat {nb(N(v.get('hp_u')))} × {nb(N(v.get('bp_u')))} × {nb(N(v.get('tp_u')))} mm · "
            f"gorge a {nb(N(v.get('a_w')))} · e1 {nb(N(v.get('e1_u')))} / p1 {nb(N(v.get('p1_u')))} / e2 {nb(N(v.get('e2b_u')))}{p2} mm")


def nom_cornieres(v) -> str:
    """« 3 M20 · L100x100x8 » : les rangées du groupe BOULONNÉ (celui de l'âme portée quand les deux le sont ; deux
    files : « 3×2 M20 · … ») ; sans aucun boulon — les deux ailes soudées — : « L100x100x8 · hc 190 · soudées » (un nom
    qui parlerait de boulons qui n'existent pas serait faux)."""
    bS, bP = v.get("fix_S") == EDC.BOULONNEE, v.get("fix_P") == EDC.BOULONNEE
    if not (bS or bP):
        return f"{v.get('corn_u')} · hc {nb(N(v.get('LC_u')))} · soudées"
    X = "S" if bS else "P"
    files = "×2" if int(N(v.get("n2" + X + "_u"))) == 2 else ""
    return f"{_i(v, 'n1' + X + '_u')}{files} {v.get('boulon_u')} · {v.get('corn_u')}"


def detail_cornieres(v) -> str:
    """La géométrie d'une proposition : chaque aile (son groupe de boulons, ou sa gorge quand elle est soudée), le
    boulon, la cornière, sa longueur ; les pinces et le trusquinage seulement là où il y a des boulons."""
    bS, bP = v.get("fix_S") == EDC.BOULONNEE, v.get("fix_P") == EDC.BOULONNEE
    g = [f"groupe S {_i(v, 'n1S_u')} × {_i(v, 'n2S_u')}" if bS else f"ailes B soudées (a {nb(N(v.get('a_S')))})",
         f"groupe P {_i(v, 'n1P_u')} × {_i(v, 'n2P_u')} par cornière" if bP else f"ailes A soudées (a {nb(N(v.get('a_P')))})"]
    inegale = f" (grande aile {AFFICHAGE['orient'].get(v.get('orient'), '')})" if _inegale(v) else ""
    boulon = f" · {v.get('boulon_u')} {v.get('classe')}" if (bS or bP) else ""
    pinces = (f" · e1 {nb(N(v.get('e1S_u')))} / p1 {nb(N(v.get('p1S_u')))} / e2 {nb(N(v.get('e2b_u')))}" if (bS or bP) else "")
    trusquinage = f" · gA {nb(N(v.get('gA_u')))}" if bP else ""
    return f"{' · '.join(g)}{boulon} · 2 {v.get('corn_u')}{inegale} · hc {nb(N(v.get('LC_u')))}{pinces}{trusquinage} mm"


def _inegale(v) -> bool:
    try:
        c = find(DB["cornieres"], v.get("corn_u"))
    except (ValueError, KeyError, TypeError):
        return False
    return c["a1"] != c["a2"]


def nom_platine(v) -> str:
    """« 2×3 M20 · platine 240×175×8 »."""
    return f"2×{_i(v, 'n1_u')} {v.get('boulon_u')} · platine {nb(N(v.get('hp_u')))}×{nb(N(v.get('bp_u')))}×{nb(N(v.get('tp_u')))}"


def detail_platine(v) -> str:
    return (f"2 files × {_i(v, 'n1_u')} rangées {v.get('boulon_u')} {v.get('classe')} · platine {nb(N(v.get('hp_u')))} × "
            f"{nb(N(v.get('bp_u')))} × {nb(N(v.get('tp_u')))} mm · gorge a {nb(N(v.get('a_w')))} · e1 {nb(N(v.get('e1_u')))} / "
            f"p1 {nb(N(v.get('p1_u')))} / p3 {nb(N(v.get('p3_u')))} mm")


# ================================================================== plat d'âme soudé
def _sections_pa(u, R, el):
    cfg = A.CONFIGS["plat_ame"]
    return [
        ("as", "Assemblage", C.esc(cfg.court), ["id_rep"], True),
        ("pt", "Poutres", resume_poutres(u), items_poutres(("duo", "d_top", "g_h")), True),
        ("pl", "Plat d'âme", f"{C.nb(N(u.get('hp_u')))} × {C.nb(N(u.get('bp_u')))} × {C.nb(N(u.get('tp_u')))} · a {C.nb(N(u.get('a_w')))}",
         [("duo", "hp_u", "bp_u"), ("duo", "tp_u", "z_C"), ("duo", "nu_pl", "a_w")], False),
        ("bo", "Boulons", f"{C.nb(N(u.get('n1_u')))} × {u.get('n2_u')} {u.get('boulon_u')} {u.get('classe')}",
         boulons_items(u) + [("grp", "Groupe · plat et âme de la portée"), ("duo", "n1_u", "n2_u"), ("duo", "p1_u", "p2_u"),
                             ("duo", "e1_u", "e2b_u")], True),
        ("ef", "Efforts ELU", resume_efforts(u, ("V_Ed", "N_Ed", "M_Ed")), [("duo", "V_Ed", "N_Ed"), "M_Ed"], True),
        A.section_options(cfg, u, el, coefficients("g_M0", "g_M1", "g_M2", "g_M2n", "g_M3", "g_M3s", ("grp", "Modèle"),
                                                   "opt_blf", "expo")),
    ]


def _parametres_pa(R):
    """Les paramètres retenus de l'écran Streamlit (plat_ame/ecran_resultats._parametres_retenus) — copie nommée."""
    return [["Boulon retenu", R.boulon + " – d0 = " + F(R.d_0, 0) + " mm"],
            ["Plat", nb(R.h_p) + " × " + nb(R.b_p) + " × " + nb(R.t_p) + " mm – " + str(R.u.nu_pl)],
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
        return boulons_texte(R, u, f"{js_str(R.n_1)} × {js_str(R.n_2)}"), "boulon"
    if grp == "Plat d'âme":
        return C.esc(f"{nb(R.h_p)} × {nb(R.b_p)} × {nb(R.t_p)} mm"), ""
    if grp == "Poutre secondaire":
        return piece_profil(u, "S")
    if grp == "Poutre principale":
        return piece_profil(u, "P")
    return C.esc(f"gorge a = {nb(R.a_w)} mm"), ""


A.enregistrer(A.Config(
    cle="plat_ame", E=EPA, calculer=MPA.compute, S=YPA, N=NPA, bench=BPA,
    vues=vues_poutre(SPA), sections=_sections_pa, libelles=libelles(), affichage=AFFICHAGE, segments=SEGMENTS,
    formules=lambda R, c: A.lignes_texte(R, c, FPA.textes), parametres=_parametres_pa, piece=_piece_pa,
    non_reverifies=NON_REVERIFIES_PA, dimensions=dimensions(EPA),
    tableaux_boulons=lambda R: [("groupe", R.tab)],
    predim={"moteur": MPA, "genre": "plat", "nom": nom_plat, "detail": detail_plat}))


# ================================================================== doubles cornières
FIXATION = ("grp", "Fixation des cornières", "Ailes A sur l'âme de la principale, ailes B sur l'âme de la secondaire. Soudées des "
                                             "deux côtés : assemblage peu déformable, à réserver à des cas particuliers.")


def _sections_dc(u, R, el):
    cfg = A.CONFIGS["doubles_cornieres"]
    bolt_S, bolt_P = u.get("fix_S") == EDC.BOULONNEE, u.get("fix_P") == EDC.BOULONNEE
    groupes = []
    if bolt_S:
        groupes += [("grp", "Groupe S · âme de la secondaire (2 plans)"), ("duo", "n1S_u", "n2S_u"), ("duo", "p1S_u", "p2_S"),
                    ("duo", "e1S_u", "e2b_u")]
    if bolt_P:
        groupes += [("grp", "Groupe P · âme de la principale (par cornière)"), ("duo", "n1P_u", "n2P_u"), ("duo", "p1P_u", "p2_P"),
                    ("duo", "e1P_u", "gA_u")]
    secs = [
        ("as", "Assemblage", C.esc(cfg.court), ["id_rep", FIXATION, "fix_P", "fix_S"], True),
        ("pt", "Poutres", resume_poutres(u), items_poutres(("duo", "d_top", "g_h")), True),
        ("cn", "Cornières", f"2 {C.esc(u.get('corn_u'))} · h<sub>c</sub> {C.nb(N(u.get('LC_u')))}",
         [("profil", "Cornière", "corn_u", "nu_C"), "orient", ("duo", "k1_u", "k2_u"), ("duo", "kt_u", "kr_u"), ("duo", "LC_u", "z_C")], False),
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
        secs.append(("so", "Soudures", "cordons d'angle", [("duo", "a_P", "lh_P"), ("duo", "a_S", "lh_S")], True))
    secs += [
        ("ef", "Efforts ELU", resume_efforts(u), [("duo", "V_Ed", "N_Ed"), ("duo", "H_Ed", "M_Ed")], True),
        A.section_options(cfg, u, el, coefficients("g_M0", "g_M2", "g_M2n", "g_M3", "g_M3s", ("grp", "Modèle"), "opt_exc",
                                                   "k_rot", "opt_blf", "expo"), avec_visserie=bolt_S or bolt_P),
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
        return C.esc(f"{R.boulon} {u.get('classe')} · " + " · ".join(x for x in g if x)), "boulon"
    if grp == "Cornières":
        return C.esc(f"2 {R.corn_txt} · hc {nb(R.L_C)}"), "corniere"
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
    cle="doubles_cornieres", E=EDC, calculer=MDC.compute, S=YDC, N=NDC, bench=BDC,
    vues=vues_poutre(SDC), sections=_sections_dc, libelles=libelles(z_C=LIBELLES["z_C@dc"]), affichage=AFFICHAGE,
    segments=SEGMENTS, formules=lambda R, c: A.lignes_sub(R, c, FDC), parametres=_parametres_dc, piece=_piece_dc,
    non_reverifies=NON_REVERIFIES_DC, dimensions=dimensions(EDC, corniere=True),
    tableaux_boulons=lambda R: ([("groupe S", R.tabS)] if R.bolt_S else []) + ([("groupe P (une cornière)", R.tabP)] if R.bolt_P and R.M_P > 0 else []),
    selecteurs={"corniere": {"rendre": _sel_corniere, "choisir": A._choisir_liste("corn_u", "cornieres")}},
    blocs=lambda R, grp: matrice_cornieres(R, YDC) if grp == "Cornières" else "",
    predim={"moteur": MDC, "genre": "cornieres", "nom": nom_cornieres, "detail": detail_cornieres}))


# ================================================================== platine d'about
def _sections_pb(u, R, el):
    cfg = A.CONFIGS["platine_about"]
    return [
        ("as", "Assemblage", C.esc(cfg.court), ["id_rep"], True),
        ("pt", "Poutres", resume_poutres(u), items_poutres("d_top"), True),
        ("pl", "Platine d'about", f"{C.nb(N(u.get('hp_u')))} × {C.nb(N(u.get('bp_u')))} × {C.nb(N(u.get('tp_u')))} · a {C.nb(N(u.get('a_w')))}",
         [("duo", "hp_u", "bp_u"), ("duo", "tp_u", "z_C"), ("duo", "nu_pl", "a_w")], False),
        ("bo", "Boulons", f"2 × {C.nb(N(u.get('n1_u')))} {u.get('boulon_u')} {u.get('classe')}",
         boulons_items(u) + [("grp", "Groupe · 2 files, 1 plan"), ("duo", "n1_u", "p3_u"), ("duo", "p1_u", "e1_u")], True),
        ("ef", "Efforts ELU", resume_efforts(u, ("V_Ed", "N_Ed")), [("duo", "V_Ed", "N_Ed"), "nat_N"], True),
        A.section_options(cfg, u, el, coefficients("g_M0", "g_M2", "g_M2n", "g_Mu", "g_M3", "g_M3s", ("grp", "Modèle"),
                                                   "opt_blf", "expo")),
    ]


def _parametres_pb(R):
    """Les paramètres retenus de l'écran Streamlit (platine_about/ecran_resultats._parametres_retenus)."""
    g = [["Boulon retenu", R.boulon + " – d0 = " + F(R.d_0, 0) + " mm – " + js_str(R.n_1) + " rangées × 2 files"],
         ["Platine", nb(R.h_p) + " × " + nb(R.b_p) + " × " + nb(R.t_p) + " mm – " + str(R.u.nu_pl)],
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
        return boulons_texte(R, u, f"2 × {js_str(R.n_1)}"), "boulon"
    if grp == "Platine":
        return C.esc(f"{nb(R.h_p)} × {nb(R.b_p)} × {nb(R.t_p)} mm"), ""
    if grp == "Poutre secondaire":
        return piece_profil(u, "S")
    if grp == "Poutre principale":
        return piece_profil(u, "P")
    return C.esc(f"gorge a = {nb(R.a_w)} mm"), ""


A.enregistrer(A.Config(
    cle="platine_about", nom_plat="platine", E=EPB, calculer=MPB.compute, S=YPB, N=NPB, bench=BPB,
    vues=vues_poutre(SPB), sections=_sections_pb, libelles=libelles(z_C=("z<sub>p</sub>", "sous le dessus")),
    affichage=AFFICHAGE, segments=SEGMENTS, formules=lambda R, c: A.lignes_texte(R, c, FPB.textes),
    parametres=_parametres_pb, piece=_piece_pb, non_reverifies=NON_REVERIFIES_PB, dimensions=dimensions(EPB),
    predim={"moteur": MPB, "genre": "platine", "nom": nom_platine, "detail": detail_platine}))

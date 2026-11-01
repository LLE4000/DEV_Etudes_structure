# -*- coding: utf-8 -*-
"""Résultats à l'écran : bandeau d'état d'une ligne, alertes courtes, dessins
cliquables (cotes, poignées + / −, fenêtres d'objet), et les cinq onglets —
Vérifications (tableaux par élément, formules et substitutions, Tableau 3.3),
Prédim, Note, Benchmark, Méthode.

Tout est lu dans ``R`` (moteur) et ``B`` (benchmark) ; rien n'est recalculé.

Charte du 31/10/2026 : les fenêtres d'objet, le panneau Cotes et les chips
d'alerte lisent ``LIBELLE``, ``AIDES`` et ``AFFICHAGE`` de la carte (une forme
par clé) ; l'onglet Vérifications range la légende, le périmètre et le
Tableau 3.3 dans des replis (rien ne se perd, tout est dans ``md(at)``).
"""
import streamlit as st

from acier.js import N, js_str
from acier.formats import F, pct
from acier.assemblages import carte as kit, ui_commun
from acier.assemblages.ui_commun import (bloc_statut_ligne, bloc_alerte, badge_nature,
                                         tableau_md, tableau_html, taux_html, verdict_html, Html, _esc,
                                         C_TRAIT, legende_natures, expander_perimetre, tableau_33_replie)
from . import schemas, texte, synthese, formules
from acier.bibliotheques import PERSO
from .notation import ec, ligne_alerte, ref_courte, LEGENDE_REFERENCES
from .entrees import CHAMPS, CLES
from .moteur import apply_solution, solution_label, nom_cornière_proposee
from .benchmark import SRC, NOCOVER, BENCH
from .methode import METHODE
from .ecran_saisie import D, PREFIXE, K, LIBELLE, AIDES
from .schemas import CLE_PAR_COTE

ONGLETS = ("Vérifications", "Prédim", "Note", "Benchmark", "Méthode")
NIVEAUX_ECRAN = ("Vue simple", "Édition", "Cotations complètes")
CLES_TETE_DESSIN = ("ui_niveau", "ui_composant")      # réglages d'écran en tête de la colonne dessin
NATURES = ("EC", "COMP", "INT")
NON_REVERIFIES = ("cisaillement global, flexion et déversement de la poutre principale",
                  "torsion induite par une attache d'un seul côté", "flexion hors plan de l'âme sous NEd",
                  "deux poutres en vis-à-vis partageant les mêmes boulons")
CAPTION_USAGE = ("Touche une cote pour la modifier, touche une pièce (poutre, cornière, boulon, cordon, VEd) pour "
                 "ses paramètres ; glisse la poutre portée (gh, Δz).")


# ------------------------------------------------------------------ bandeau
def bandeau(R):
    """Une ligne : statut, taux maximal, vérification dimensionnante ; puis
    les taux par élément."""
    ok = R.verified
    titre = R.statut + (" – sous réserve des alertes" if ok and (R.reserve or R.alerts) else "")
    # la dimensionnante nommée comme dans la note (synthese.court), pas le libellé long du moteur
    comp = "— " + pct(R.eta_max, 1) + (" · " + synthese.court(R.gov) if R.gov else "")
    if not R.geo_ok:
        comp += " · géométrie non valide"
    if not R.dist_ok:
        comp += " · pinces ou entraxes non conformes"
    bloc_statut_ligne("ok" if ok else "nok", titre, comp,
                      [(nom, eta, okg) for nom, eta, okg, _ in synthese.taux_par_element(R)])
    if R.pred:
        bloc_alerte("Mode prédimensionnement : " + R.boulon + ", " + js_str(R.n1_S) + " rangées, " + R.corn_txt
                    + ", hc = " + F(R.L_C, 0) + " mm. " + R.pd.msg, False)


# ------------------------------------------------------------------ alertes
def alertes(R, u):
    """Les alertes, forme commune des cinq écrans (``ui_commun.alertes``) ;
    sélectionnée, les champs en cause modifiables sur place (``kit.pastilles``)."""
    ui_commun.alertes(PREFIXE, R, ligne_alerte, lambda a: kit.pastilles(D, a, u, R.pred), texte=ec)


# ------------------------------------------------------------------ dessins
def _panneaux(R, u):
    """Les fenêtres d'objet du composant : un clic sur une pièce du dessin
    ouvre TOUS ses paramètres — une VUE des mêmes clés (libellés LIBELLE,
    aides AIDES, options AFFICHAGE), titres courts, aucun doublon (a_P / a_S
    seulement dans leur fenêtre de cordon) ; la fenêtre Poutre secondaire
    porte aussi gh, Δz, dc,sup, dc,inf et c. En prédimensionnement, les
    champs pilotés (◆) sont désactivés."""
    pred = R.pred

    def ch(k):
        return kit.champ_fenetre(D, k, u, pred)

    def perso(X):
        return [ch(f"{d}{X}_u") for d in ("h", "b", "tw", "tf", "r")]

    g = {}
    if R.bolt_S:
        g["S"] = dict(titre="Groupe S",
                      champs=[ch("n1S_u"), ch("n2S_u"), ch("p1S_u")]
                      + ([ch("p2_S")] if R.n2_S > 1 else [])
                      + [ch("e1S_u"), ch("e2b_u")],
                      note="Âme de la poutre secondaire, deux plans de cisaillement.")
    if R.bolt_P:
        g["P"] = dict(titre="Groupe P",
                      champs=[ch("n1P_u"), ch("n2P_u"), ch("p1P_u")]
                      + ([ch("p2_P")] if R.n2_P > 1 else [])
                      + [ch("e1P_u"), ch("gA_u")],
                      note="Âme de la poutre principale, un plan, par cornière.")
    if R.bolt_S or R.bolt_P:
        g["bolts"] = dict(titre="Boulons",
                          champs=[ch("boulon_u"), ch("classe"), ch("trou"), ch("cat"), ch("filet")]
                          + ([ch("mu_s"), ch("k_s")] if u.get("cat") != "A" else [])
                          + ([ch("k_ser")] if u.get("cat") == "B" else []))
    g["beamS"] = dict(titre="Poutre secondaire",
                      champs=[ch("prof_S"), ch("nu_S")]
                      + (perso("S") if u.get("prof_S") == PERSO else [])
                      + [ch("lt_ok"), ch("r_n"), ch("g_h"), ch("d_top"), ch("d_nt"), ch("d_nb"), ch("l_n")],
                      note="gh, Δz, dc,sup, dc,inf et c : aussi des cotes sur le dessin — la poutre se déplace "
                           "aussi à la souris (gh, Δz).")
    g["beamP"] = dict(titre="Poutre principale",
                      champs=[ch("prof_P"), ch("nu_P")]
                      + (perso("P") if u.get("prof_P") == PERSO else []))
    g["cleat"] = dict(titre="Cornières",
                      champs=[ch("corn_u"), ch("nu_C"), ch("orient")]
                      + ([ch("k1_u"), ch("k2_u"), ch("kt_u"), ch("kr_u")] if u.get("corn_u") == PERSO else [])
                      + [ch("fix_P"), ch("fix_S")],
                      note="Deux cornières identiques ; hauteur hc et position zc : cotes sur le dessin.")
    if not R.bolt_S:
        g["weldS"] = dict(titre="Cordons B", champs=[ch("a_S"), ch("lh_S"), ch("fix_S")],
                          note="Ailes B sur l'âme portée.")
    if not R.bolt_P:
        g["weldP"] = dict(titre="Cordons A", champs=[ch("a_P"), ch("lh_P"), ch("fix_P")],
                          note="Ailes A sur l'âme porteuse.")
    g["efforts"] = dict(titre="Efforts", champs=[ch("V_Ed"), ch("N_Ed"), ch("H_Ed"), ch("M_Ed")])
    return g


def _alerte_courante(R):
    sel = st.session_state.get("asm_ui_alerte")
    for a in R.alerts:
        if a.id == sel:
            return dict(dims=set(a.dims), elems=set(a.elems))
    return None


def dessins(R, u):
    """La colonne du dessin : en tête le niveau de cotation et la case
    « Dessin interactif », puis élévation et vue en plan EMPILÉES — la
    colonne est figée à l'écran (CSS sticky, interface.py). Tout se règle
    sur le dessin — une cote se touche, une pièce (poutre, cornière, boulons,
    cordons, efforts) s'ouvre au clic avec tous ses paramètres, la poutre
    portée se déplace à la souris (jeu gh, décalage Δz). Renvoie
    ``(e, p, hl)`` pour le panneau des cotes (repli)."""
    with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="center", key="asm_tete_dessin"):
        niveau = st.radio(LIBELLE["ui_niveau"], [0, 1, 2], format_func=lambda i: NIVEAUX_ECRAN[i],
                          horizontal=True, key="asm_ui_niveau", label_visibility="collapsed")
        st.checkbox(LIBELLE["ui_composant"], key="asm_ui_composant", help=AIDES["ui_composant"])
    interactif = bool(st.session_state.get("asm_ui_composant", True))
    hl = _alerte_courante(R)
    opt = schemas.options_ecran(R, niveau, hl)
    opt.editables = niveau >= 1
    e = schemas.elevation(R, opt); p = schemas.plan(R, opt)
    valeurs = {k: st.session_state.get(K(k)) for k in CLES}
    groupes = _panneaux(R, u)
    for d, cle in ((e, "asm_cmp_elev"), (p, "asm_cmp_plan")):
        if interactif:
            from acier.composants.svg_cliquable import svg_cliquable
            kit.traiter_clic(D, svg_cliquable(d.svg(identifiant=cle), valeurs, key=cle, groupes=groupes), cle, u)
        else:
            st.markdown(d.svg(identifiant=cle), unsafe_allow_html=True)
    if interactif and not R.pred:
        st.caption(CAPTION_USAGE)       # en prédim, le bandeau ocre et les ◆ suffisent (P9)
    return e, p, hl


def panneau_cotes(R, u, e, p, hl):
    """Repli (tête de la colonne de droite, dessin interactif décoché) : une
    saisie par cote (``kit.panneau_cotes``) — les cotes du dessin, celles de
    l'Édition en Vue simple (point 15 : le panneau était vide), puis la
    géométrie visible qu'il n'émet pas (rien ne se perd)."""
    if not (schemas.cotes_modifiables(e) or schemas.cotes_modifiables(p)):
        opt = schemas.options_ecran(R, 1, hl)
        opt.editables = True
        e, p = schemas.elevation(R, opt), schemas.plan(R, opt)
    emises = [k for k, _s, _i in schemas.cotes_modifiables(e) + schemas.cotes_modifiables(p)]
    chauds = {CLE_PAR_COTE[d] for d in (hl["dims"] if hl else ()) if d in CLE_PAR_COTE}
    kit.panneau_cotes(D, kit.cotes_du_panneau(D, u, emises, list(CLE_PAR_COTE.values())), u, chauds)


# ------------------------------------------------------ onglet vérifications
def _cellule_verif(l):
    return Html(f"{l['lab']}{badge_nature(l['nat'])}")


def _table_simple(R, cle):
    ls = synthese.lignes(R, cle)
    if not ls:
        return False
    st.markdown(f"##### {synthese.titre_table(cle)}")
    lignes = [[_cellule_verif(l), l["Ed"], l["Rd"] + (" " + l["unit"] if l["unit"] else ""),
               taux_html(l["eta"], l["ok"], l["c"] is R.gov), l["ref"]] for l in ls]
    st.markdown(tableau_html(["Vérification", "Ed", "Rd", "η", "Réf."], lignes, droite=(1, 2, 3),
                             largeurs=["34%", "12%", "14%", "9%", None]), unsafe_allow_html=True)
    return True


def _cell_aile(x, gov):
    """« Ed / Rd unité · η » : une ligne sur PC ; au téléphone la cellule ne se coupe qu'après « / » et
    avant « · » (espaces insécables ailleurs) — un nombre ne se coupe jamais."""
    if x is None:
        return "—"
    rd = x["Rd"] + ("&nbsp;" + _esc(x["unit"]) if x["unit"] else "")
    return Html(f"{x['Ed']}&nbsp;/ {rd} ·&nbsp;{taux_html(x['eta'], x['ok'], x['c'] is gov)}")


# La matrice tient dans la colonne du téléphone (358 px à 390) au lieu de défiler : cellules resserrées et
# en-têtes repliés sous 1181 px — la table serrée du kit (tableau_html(serre=1), classe tab-serre de
# carte.css_kit ; mesuré le 31/10/2026 : 430 px au plus serré avec les marges du kit, 309 px ainsi ; à 1024
# la table passe de 699 à 519 px de haut). Le défilement (defilant=True) reste le filet d'un écran plus étroit.


def _table_cornieres(R):
    """La matrice aile A · aile B · max, défilante au téléphone
    (tableau_html(defilant=True)) ; la légende des ailes en caption."""
    ls = synthese.matrice_cornieres(R)
    if not ls:
        return
    st.markdown("##### Cornières — aile A / aile B")
    lignes = []
    for l in ls:
        refs = synthese.refs_fusionnees([x["ref"] for x in (l["A"], l["B"]) if x])
        nat = (l["A"] or l["B"])["nat"]
        lignes.append([Html(f"{l['lab']}{badge_nature(nat)}"), _cell_aile(l["A"], R.gov), _cell_aile(l["B"], R.gov),
                       taux_html(l["max"]["eta"], l["max"]["ok"], l["max"]["c"] is R.gov), refs])
    st.markdown(tableau_html(["Vérification", "Aile A : Ed / Rd · η", "Aile B : Ed / Rd · η", "max", "Réf."], lignes,
                             droite=(1, 2, 3), largeurs=["26%", "22%", "22%", "8%", None], defilant=True, serre=1),
                unsafe_allow_html=True)
    st.caption("Aile A sur la poutre principale (groupe P) · aile B sur la poutre secondaire (groupe S).")


def _table_boulons_efforts(t, lab):
    with st.expander("Répartition élastique par boulon – " + lab):
        st.markdown(tableau_md(["Rangée / file", "x", "y", "Fx", "Fz", "F [kN]"],
                               [[f"{b.i} / {b.j}", F(b.x, 1), F(b.y, 1), F(b.fx, 2), F(b.fz, 2), F(b.f, 2)] for b in t],
                               droite=(1, 2, 3, 4, 5)))


def _formules(R):
    with st.expander("Formules et substitutions numériques"):
        st.caption("Chaque substitution est composée avec les nombres du moteur et vérifiée par un test qui la "
                   "recalcule. Grandeurs par cornière, par boulon ou par cordon comme indiqué.")
        for grp in ("Boulons", "Cornières", "Poutre secondaire", "Poutre principale", "Soudures"):
            on = [c for c in R.checks if c.grp == grp and c.active]
            if not on:
                continue
            st.markdown(f"**{grp}**")
            for c in on:
                lignes = formules.textes(R, c)
                corps = "<br>".join(_esc(l) for l in lignes)
                st.markdown(
                    f"<div style='margin:2px 0 8px 0;padding:4px 10px;border-left:3px solid "
                    f"{C_TRAIT['ok'] if c.ok else C_TRAIT['nok']};font-size:0.88em;'>"
                    f"<b>{_esc(synthese.court(c))}</b>{badge_nature(c.nat)} "
                    f"<span style='color:{C_TRAIT['mut']};'>{_esc(ref_courte(c.ref))}</span><br>"
                    f"<span style='font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:0.95em;'>{corps}</span></div>",
                    unsafe_allow_html=True)


def _tableau_33(R):
    """Le Tableau 3.3 replié : verdict dans le titre (ouvert d'office dès une
    ligne NON OK), lignes nommées par le moteur (x.lab), boutons Localiser,
    règles en caption."""
    def rendre():
        if not R.dist:
            st.caption("Sans objet (aucun boulon).")
            return
        lignes = [[x.lab, F(x.val, 1), F(x.min, 1), "—" if x.max is None else F(x.max, 0), verdict_html(x.ok)]
                  for x in R.dist]
        st.markdown(tableau_html(["Distance", "Valeur", "Min", "Max", ""], lignes, droite=(1, 2, 3),
                                 largeurs=[None, "10%", "10%", "10%", "10%"]), unsafe_allow_html=True)
        nok = [x for x in R.dist if not x.ok]
        if nok:
            cols = st.columns(min(4, len(nok)))
            for i, x in enumerate(nok):
                with cols[i % len(cols)]:
                    st.button("Localiser : " + x.lab.split(" (")[0], key=f"asm_btn_dist_{x.id}", width="stretch",
                              on_click=ui_commun.selectionner, args=(PREFIXE, x.alert))
        st.caption("e ≥ 1,2·d0 ; p1 ≥ 2,2·d0 ; p2 ≥ 2,4·d0 ; p ≤ min(14t ; 200 mm) ; e ≤ 4t + 40 mm si exposé.")
    tableau_33_replie(R.dist_ok, rendre)


def _parametres_retenus(R):
    g = [["Boulon retenu", R.boulon + " – d0 = " + F(R.d_0, 0) + " mm"],
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
    # titre court (charte §10) ; « et excentricités » : les lignes z, MS et MP le disent
    with st.expander("Paramètres retenus"):
        st.markdown(tableau_md(["Paramètre", "Valeur"], g))


def onglet_verifications(R):
    """Les tables par élément d'abord ; après la dernière, la légende des
    natures (une ligne + « Légende et références »), puis les replis :
    répartition par boulon, périmètre, formules, Tableau 3.3, paramètres."""
    _table_simple(R, "boulons")
    _table_cornieres(R)
    _table_simple(R, "portee")
    _table_simple(R, "porteuse")
    _table_simple(R, "cordons")
    legende_natures(NATURES, LEGENDE_REFERENCES)
    if R.bolt_S:
        _table_boulons_efforts(R.tabS, "groupe S")
    if R.bolt_P and R.M_P > 0:
        _table_boulons_efforts(R.tabP, "groupe P (une cornière)")
    expander_perimetre(NON_REVERIFIES, [synthese.court(c) for c in R.checks if not c.active])
    _formules(R)
    _tableau_33(R)
    _parametres_retenus(R)


# ------------------------------------------------------------ onglet prédim
def _appliquer(i, ecrire):
    R = st.session_state.get("_asm_R")
    r = R.pd.rows[i]
    if not r.prop:
        return
    ecrire(apply_solution({k: st.session_state.get(K(k)) for k in CHAMPS}, r))
    st.session_state["asm_ui_alerte"] = None
    st.session_state["asm_ui_onglet"] = ONGLETS[0]
    st.session_state["_asm_toast"] = "Solution appliquée aux données et à la géométrie : " + solution_label(r)


def onglet_predim(R, u, ecrire):
    p = R.pd
    rows = [r for r in p.rows if r.geom]
    sols = sorted([r for r in rows if r.eta <= 1],
                  key=lambda r: (0 if r.ok else 1, r.score if r.ok else r.eta))[:9]
    # une seule caption d'usage : le mode actif est dit par le bandeau (P9)
    st.caption("Touche « Appliquer » pour recopier une solution dans les données et la géométrie ; elle devient "
               "modifiable comme une saisie. Seule la vérification détaillée fait foi.")
    st.markdown("#### Solutions proposées")
    if sols:
        for i in range(0, len(sols), 3):
            cols = st.columns(3)
            for col, r in zip(cols, sols[i:i + 3]):
                with col:
                    with st.container(border=True):
                        retenue = r is p.pick
                        st.markdown(f"**{r.b['n']} – {r.n} rangées**  \n{nom_cornière_proposee(r)}, hc {r.Lc} mm")
                        st.caption(f"e1 {r.e1} / p1 {r.p1} / e2 {r.e2} / gA {r.prop.gA} mm  \ntaux estimé "
                                   + pct(r.eta) + (" – retenue par défaut" if retenue else
                                                    ("" if r.ok else " – au-dessus du taux cible de " + pct(N(u.get("eta_c"))))))
                        st.button("Appliquer", key=f"asm_btn_sol_{p.rows.index(r)}", width="stretch",
                                  type="primary" if retenue else "secondary",
                                  on_click=_appliquer, args=(p.rows.index(r), ecrire))
    else:
        bloc_alerte(p.msg, True)
    st.markdown("#### Détail de la solution retenue par défaut")
    st.markdown(tableau_md(["", ""], [
        ["Boulon proposé", p.boulon], ["Rangées n1 (groupes S et P, 1 file)", p.n],
        ["e1 / p1 / e2 = e2,b", f"{p.e1} / {p.p1} / {p.e2} mm"],
        ["Hauteur des cornières hc = 2·e1 + (n1 − 1)·p1", f"{p.LC} mm (hauteur disponible " + F(p.hav, 0) + " mm)"],
        ["Épaisseur nécessaire au taux cible", F(p.treq, 2) + " mm"],
        ["Cornière proposée", f"L{p.leg}x{p.leg}x{p.tC} – trusquinage gA = {p.gA} mm"],
        ["Taux estimé (boulons et pressions diamétrales des âmes)", pct(p.eta, 1)], ["État", p.msg]]))
    st.markdown("#### Épaisseur nécessaire par vérification")
    st.markdown(tableau_md(["Vérification", "Épaisseur"], [[q[0], F(q[1], 2) + " mm"] for q in p.reqs], droite=(1,)))
    st.markdown("#### Toutes les combinaisons géométriquement possibles")
    h = st.columns([1, 0.6, 0.8, 1.6, 1, 1, 1, 1, 1, 1.3])
    for c, t in zip(h, ("Boulon", "n1", "hc", "Cornière", "η boulons S", "η p. diam. âme S", "η boulons P",
                        "η p. diam. âme P", "η max", "")):
        c.markdown(f"**{t}**" if t else "")
    for r in rows:
        c = st.columns([1, 0.6, 0.8, 1.6, 1, 1, 1, 1, 1, 1.3], vertical_alignment="center")
        fort = "**" if r is p.pick else ""
        vals = (r.b["n"], js_str(r.n), js_str(r.Lc), nom_cornière_proposee(r), pct(r.e.bS), pct(r.e.pdS),
                pct(r.e.bP), pct(r.e.pdP), pct(r.eta))
        for cc, v in zip(c[:9], vals):
            cc.markdown(f"{fort}{v}{fort}")
        with c[9]:
            st.button("Appliquer" + ("" if r.ok else " ⚠"), key=f"asm_btn_row_{p.rows.index(r)}",
                      width="stretch", on_click=_appliquer, args=(p.rows.index(r), ecrire),
                      help=None if r.ok else "taux cible dépassé")
    st.caption("Règle de choix par défaut : le plus petit nombre de rangées ; à égalité, le plus petit "
               "diamètre. MEd est ajouté à VEd·z ; les tractions dues à HEd et MEd ne sont prises en "
               "compte que dans la vérification détaillée.")


# -------------------------------------------------------------- onglet note
def onglet_note(R, u, generer_detaille):
    """Export texte et rapport détaillé (la note de calcul de deux pages est
    celle du bouton « 📄 PDF » de la barre d'outils)."""
    c1, c2, c3 = st.columns([1.2, 1.2, 2.6], vertical_alignment="center")
    with c1:
        if st.button("📄 Rapport détaillé (3 pages)", width="stretch", key="asm_btn_pdf_detail"):
            generer_detaille()
        if st.session_state.get("asm_pdf_detail_bytes"):
            st.download_button("⬇️ Télécharger le rapport détaillé", data=st.session_state["asm_pdf_detail_bytes"],
                               file_name=texte.nom_fichier(u, "_detail.pdf"), mime="application/pdf",
                               width="stretch", key="asm_btn_pdf_detail_dl")
    complet = c2.checkbox("Texte : toutes les vérifications", key="asm_ui_txt_complet")
    t = texte.construire_texte(R, complet)
    with c2:
        st.download_button("⬇️ Exporter le texte (.txt)", data=("﻿" + t).encode("utf-8"),
                           file_name=texte.nom_fichier(u, ".txt"), mime="text/plain;charset=utf-8",
                           width="stretch", key="asm_btn_txt")
    c3.caption("Note de calcul (2 pages, A4 paysage — synthèse + plan de principe) : bouton « 📄 PDF » de la "
               "barre d'outils. Rapport détaillé : garde, synthèse et développement de chaque vérification. Le "
               "texte ci-dessous se copie avec l'icône en haut à droite du cadre.")
    st.code(t, language=None)


# --------------------------------------------------------- onglet benchmark
def _charger_benchmark(i, ecrire):
    """Callback : les données de l'exemple publié i dans l'outil."""
    from .entrees import defaults
    d = defaults(); d.update(BENCH[i]["inp"]); ecrire(d)
    st.session_state["asm_ui_alerte"] = None
    st.session_state["asm_ui_onglet"] = ONGLETS[0]
    st.session_state["_asm_toast"] = BENCH[i]["id"] + " chargé : voir Vérifications"


def onglet_benchmark(B, ecrire):
    from acier.assemblages.ui_commun import bloc_statut
    bloc_statut("Benchmark du module : " + B["statut"],
                ["Ces tableaux sont recalculés par le moteur de l'outil à chaque ouverture ; ils ne dépendent pas de tes données."],
                "ok" if B["ok"] else "nok")
    st.caption("Source : " + SRC["titre"] + ". " + SRC["org"] + ", " + SRC["annee"] + ". Norme : " + SRC["norme"] + ". " + SRC["url"])
    st.caption("Seuils : écart ≤ 1 % OK ; 1 à 3 % à expliquer ; > 3 % à vérifier. Aucune formule n'a été ajustée pour coller à la source.")
    for i, b in enumerate(B["bench"]):
        st.markdown(f"#### {b['id']}")
        st.caption(b["titre"])
        c1, c2 = st.columns([4, 1.4], vertical_alignment="center")
        c1.caption("Données : " + b["data"])
        with c2:
            st.button("Charger ces données dans l'outil", key=f"asm_btn_bm_{i}", width="stretch",
                      on_click=_charger_benchmark, args=(i, ecrire))
        lignes = []
        for r in b["rows"]:
            lab = r["lab"] + ((" — " + r["exp"]) if (r["exp"] and abs(r["ec"]) > 0.003) else "")
            # espaces et tirets insécables : une valeur ne se coupe pas dans le tableau
            lignes.append([lab, r["page"].replace("-", "‑").replace(" ", " "),
                           F(r["ref"], 1 if r["ref"] < 10 else 0) + " " + r["unit"],
                           F(r["val"], 2 if r["ref"] < 10 else 1), F(r["ec"] * 100, 2) + " %", r["st"]])
        st.markdown(tableau_md(["Vérification", "Page", "Référence", "Outil", "Écart", "Statut"], lignes, droite=(2, 3, 4)))
    st.markdown("#### Vérifications sans exemple publié consulté")
    for x in NOCOVER:
        st.markdown(f"- **{x[0]}** – {x[1]}")
    for c in B["valid"]:
        st.markdown(f"#### {c['id']} – calcul manuel indépendant")
        st.caption(c["titre"] + ". Colonne « manuel » : calcul indépendant hors moteur (intégration numérique "
                   "pour les cordons). Colonne « Excel » : classeur recalculé avec les mêmes données.")
        lignes = []
        for r in c["rows"]:
            n = 4 if abs(r["hand"]) < 10 else (2 if abs(r["hand"]) < 1000 else 0)
            lignes.append([r["lab"], F(r["hand"], n) + ("" if r["unit"] == "-" else " " + r["unit"]),
                           F(r["xls"], n), F(r["val"], n), F(r["ec"] * 100, 3) + " %", r["st"]])
        st.markdown(tableau_md(["Grandeur", "Manuel", "Excel", "Outil", "Écart", ""], lignes, droite=(1, 2, 3, 4)))


def onglet_methode():
    st.markdown(METHODE)


# ------------------------------------------------------------------- onglets
def onglets(R, B, u, ecrire, generer_detaille):
    """La barre d'onglets et l'onglet courant. ``ecrire(u)`` écrit un jeu
    complet de saisies dans la source unique (solution, benchmark)."""
    st.session_state["_asm_R"] = R
    st.segmented_control("Onglet", list(ONGLETS), key="asm_ui_onglet", label_visibility="collapsed", wrap=True)
    onglet = st.session_state.get("asm_ui_onglet") or ONGLETS[0]
    if onglet == ONGLETS[1]:
        onglet_predim(R, u, ecrire)
    elif onglet == ONGLETS[2]:
        onglet_note(R, u, generer_detaille)
    elif onglet == ONGLETS[3]:
        onglet_benchmark(B, ecrire)
    elif onglet == ONGLETS[4]:
        onglet_methode()
    else:
        onglet_verifications(R)

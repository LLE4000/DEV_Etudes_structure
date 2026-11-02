# -*- coding: utf-8 -*-
"""Résultats à l'écran des doubles cornières sur poteau : bandeau d'une ligne,
alertes courtes, dessins cliquables (colonne figée), onglets Vérifications ·
Prédim · Note · Benchmark · Méthode. Tout est lu dans ``R`` (moteur) et
``B`` (benchmark).

Charte du 31/10/2026 : les fenêtres d'objet, le panneau Cotes et les chips
d'alerte lisent ``LIBELLE`` et ``AIDES`` de la carte (une forme par clé) ;
l'onglet Vérifications range la légende, le périmètre et le Tableau 3.3
dans des replis (rien ne se perd, tout est dans ``md(at)``).

Le groupe « Poteau porteur » n'est jamais vide quand les cornières sont
BOULONNÉES sur le poteau (pression diamétrale de l'aile ou de l'âme) ; la
bascule ``appui`` change ce qu'il porte : cisaillement local de l'âme en
cas (b), tronçon en T de l'aile et âme en traction transversale en cas (a)
sous boulons tendus."""
import streamlit as st

from acier.js import N, js_str
from acier.formats import F, pct
from acier.assemblages import carte as kit, ui_commun
from acier.assemblages.ui_commun import (bloc_statut_ligne, bloc_alerte, badge_nature,
                                         tableau_md, tableau_html, taux_html, verdict_html, Html, _esc,
                                         C_TRAIT, legende_natures, expander_perimetre, tableau_33_replie)
from acier.bibliotheques import PERSO
from . import schemas, texte, synthese, formules, methode
from .notation import ligne_alerte, ref_courte, LEGENDE_REFERENCES
from .entrees import CHAMPS, CLES
from .moteur import apply_solution, solution_label, nom_cornière_proposee
from .benchmark import SRC, NOCOVER
from .methode import METHODE
from .ecran_saisie import D, PREFIXE, K, LIBELLE, AIDES
from .schemas import CLE_PAR_COTE

ONGLETS = ("Vérifications", "Prédim", "Note", "Benchmark", "Méthode")
NIVEAUX_ECRAN = ("Vue simple", "Édition", "Cotations complètes")
CLES_TETE_DESSIN = ("ui_niveau", "ui_composant")      # réglages d'écran en tête de la colonne dessin
NATURES = ("EC", "COMP", "INT")
# Les limites de validité ont UNE source : methode.limites(ame) — l'onglet
# Méthode les écrit en liste, le repli « Périmètre » les reprend mot pour mot.
NON_REVERIFIES = methode.limites(True)
PERIMETRE_AILE = ("Aile du poteau : l'attache est centrée sur l'âme du poteau (deux files symétriques "
                  "à ±p3/2) et vit dans son plan ; VEd et VEd·z y cheminent en plan. Sous l'effort "
                  "tranchant seul, le côté poteau n'apporte que la pression diamétrale de l'aile (ou "
                  "les cordons) : aucun cisaillement local de l'aile n'est demandé par MSB P5 §4.2 ni "
                  "par SCI P358. La section du poteau se vérifie dans son propre calcul "
                  "(EN 1993-1-1 §6.2).")
CAPTION_USAGE = ("Touche une cote pour la modifier, touche une pièce (poteau, poutre, cornière, boulon, "
                 "cordon, VEd) pour ses paramètres ; glisse la poutre portée (gh).")


# ------------------------------------------------------------------ bandeau
def bandeau(R):
    ok = R.verified
    titre = R.statut + (" – sous réserve des alertes" if ok and (R.reserve or R.alerts) else "")
    comp = "— " + pct(R.eta_max, 1) + (" · " + synthese.court(R.gov) if R.gov else "")
    if not R.geo_ok:
        comp += " · géométrie non valide"
    if not R.dist_ok:
        comp += " · pinces ou entraxes non conformes"
    bloc_statut_ligne("ok" if ok else "nok", titre, comp,
                      [(nom, eta, okg) for nom, eta, okg, _ in synthese.taux_par_element(R)])
    if R.pred:
        bloc_alerte("Mode prédimensionnement : " + R.boulon + ", " + js_str(R.n1_S) + " rangées, "
                    + R.corn_txt + ", hc = " + F(R.L_C, 0) + " mm. " + R.pd.msg, False)


# ------------------------------------------------------------------ alertes
def alertes(R, u):
    """Les alertes, forme commune des écrans (``ui_commun.alertes``) ;
    sélectionnée, les champs en cause modifiables sur place (``kit.pastilles``)."""
    ui_commun.alertes(PREFIXE, R, ligne_alerte, lambda a: kit.pastilles(D, a, u, R.pred))


# ------------------------------------------------------------------ dessins
def _panneaux(R, u):
    """Les fenêtres d'objet du composant : un clic sur une pièce ouvre TOUS
    ses paramètres — une VUE des mêmes clés (libellés LIBELLE, aides AIDES),
    titres courts, aucun doublon (a_P / a_S ne restent que dans leur fenêtre
    de cordon)."""
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
                      note="Âme de la poutre portée, deux plans de cisaillement.")
    if R.bolt_P:
        g["P"] = dict(titre="Groupe P",
                      champs=[ch("n1P_u"), ch("n2P_u"), ch("p1P_u")]
                      + ([ch("p2_P")] if R.n2_P > 1 else [])
                      + [ch("e1P_u"), ch("gA_u")],
                      note="Poteau, un plan par cornière ; gauge p3 = tw,S + 2·gA = " + F(R.p_3, 1)
                           + " mm (minimum " + F(R.p3_congeP, 1) + " mm sur l'aile).")
    if R.bolt_S or R.bolt_P:
        g["bolts"] = dict(titre="Boulons",
                          champs=[ch("boulon_u"), ch("classe"), ch("trou"), ch("cat"), ch("filet")]
                          + ([ch("mu_s"), ch("k_s")] if u.get("cat") != "A" else [])
                          + ([ch("k_ser")] if u.get("cat") == "B" else []))
    g["beamS"] = dict(titre="Poutre portée",
                      champs=[ch("prof_S"), ch("nu_S")]
                      + (perso("S") if u.get("prof_S") == PERSO else [])
                      + [ch("lt_ok"), ch("r_n"), ch("g_h"), ch("d_nt"), ch("d_nb"), ch("l_n")]
                      + ([ch("rec_S"), ch("j_lat")] if R.ame else []),
                      note="gh, dc,sup, dc,inf, c" + (", rc et j" if R.ame else "")
                           + " : aussi des cotes sur le dessin — la poutre se déplace aussi à la souris.")
    g["col"] = dict(titre="Poteau",
                    champs=[ch("appui"), ch("prof_P"), ch("nu_P")]
                    + (perso("P") if u.get("prof_P") == PERSO else []),
                    note=("Cas (b) : les cornières et la poutre tiennent entre les ailes — largeur droite "
                          "entre congés " + F(R.libre_droit, 0) + " mm, aile A maximale "
                          + F(R.bA_max, 1) + " mm, passage de la semelle " + F(R.libre_ailes, 0)
                          + " mm.") if R.ame else
                         ("Cas (a) : l'attache est centrée sur l'âme du poteau ; gauge minimal "
                          "twP + 2·rP + dw = " + F(R.p3_congeP, 1) + " mm, pince au bout d'aile "
                          + F(R.e_P_fl, 1) + " mm."))
    g["colFl"] = g["col"]
    g["colWeb"] = g["col"]
    g["recut"] = g["beamS"]
    g["cleat"] = dict(titre="Cornières",
                      champs=[ch("corn_u"), ch("nu_C"), ch("orient")]
                      + ([ch("k1_u"), ch("k2_u"), ch("kt_u"), ch("kr_u")] if u.get("corn_u") == PERSO else [])
                      + [ch("fix_P"), ch("fix_S")],
                      note="Deux cornières identiques ; hauteur hc et position zc : cotes sur le dessin.")
    if not R.bolt_S:
        g["weldS"] = dict(titre="Cordons B", champs=[ch("a_S"), ch("lh_S"), ch("fix_S")],
                          note="Ailes B sur l'âme de la poutre portée.")
    if not R.bolt_P:
        g["weldP"] = dict(titre="Cordons A", champs=[ch("a_P"), ch("lh_P"), ch("fix_P")],
                          note="Ailes A sur l'aile ou l'âme du poteau.")
    g["efforts"] = dict(titre="Efforts",
                        champs=[ch("V_Ed"), ch("N_Ed"), ch("H_Ed"), ch("M_Ed")]
                        + ([ch("nat_N")] if N(u.get("N_Ed")) > 0 else []))
    return g


def _alerte_courante(R):
    sel = st.session_state.get("dcc_ui_alerte")
    for a in R.alerts:
        if a.id == sel:
            return dict(dims=set(a.dims), elems=set(a.elems))
    return None


def dessins(R, u):
    """La colonne du dessin : en tête le niveau de cotation et la case
    « Dessin interactif », puis élévation et VUE EN PLAN empilées (colonne
    figée par interface.py). Renvoie ``(e, p, hl)`` pour le repli."""
    with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="center",
                      key="dcc_tete_dessin"):
        niveau = st.radio(LIBELLE["ui_niveau"], [0, 1, 2], format_func=lambda i: NIVEAUX_ECRAN[i],
                          horizontal=True, key="dcc_ui_niveau", label_visibility="collapsed")
        st.checkbox(LIBELLE["ui_composant"], key="dcc_ui_composant", help=AIDES["ui_composant"])
    interactif = bool(st.session_state.get("dcc_ui_composant", True))
    hl = _alerte_courante(R)
    opt = schemas.options_ecran(R, niveau, hl)
    opt.editables = niveau >= 1
    e = schemas.elevation(R, opt); p = schemas.plan(R, opt)
    valeurs = {k: st.session_state.get(K(k)) for k in CLES}
    groupes = _panneaux(R, u)
    for d, cle in ((e, "dcc_cmp_elev"), (p, "dcc_cmp_plan")):
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
    l'Édition en Vue simple, puis la géométrie visible qu'il n'émet pas."""
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
    """« Ed / Rd unité · η » : une ligne sur PC ; au téléphone la cellule ne se
    coupe qu'après « / » et avant « · » (espaces insécables ailleurs)."""
    if x is None:
        return "—"
    rd = x["Rd"] + ("&nbsp;" + _esc(x["unit"]) if x["unit"] else "")
    return Html(f"{x['Ed']}&nbsp;/ {rd} ·&nbsp;{taux_html(x['eta'], x['ok'], x['c'] is gov)}")


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
        lignes.append([Html(f"{l['lab']}{badge_nature(nat)}"), _cell_aile(l["A"], R.gov),
                       _cell_aile(l["B"], R.gov),
                       taux_html(l["max"]["eta"], l["max"]["ok"], l["max"]["c"] is R.gov), refs])
    st.markdown(tableau_html(["Vérification", "Aile A : Ed / Rd · η", "Aile B : Ed / Rd · η", "max", "Réf."],
                             lignes, droite=(1, 2, 3), largeurs=["26%", "22%", "22%", "8%", None],
                             defilant=True, serre=1), unsafe_allow_html=True)
    st.caption("Aile A sur le poteau (groupe P) · aile B sur la poutre portée (groupe S).")


def _table_boulons_efforts(t, lab):
    with st.expander("Répartition élastique par boulon – " + lab):
        st.markdown(tableau_md(["Rangée / file", "x", "y", "Fx", "Fz", "F [kN]"],
                               [[f"{b.i} / {b.j}", F(b.x, 1), F(b.y, 1), F(b.fx, 2), F(b.fz, 2), F(b.f, 2)]
                                for b in t], droite=(1, 2, 3, 4, 5)))


def _formules(R):
    with st.expander("Formules et substitutions numériques"):
        st.caption("Les valeurs introduites sont composées par le moteur ; la ligne du taux est vérifiée "
                   "par un test qui la recalcule. Grandeurs par cornière, par boulon ou par cordon comme "
                   "indiqué.")
        for grp in ("Boulons", "Cornières", "Poutre portée", "Poteau porteur", "Soudures"):
            on = [c for c in R.checks if c.grp == grp and c.active]
            if not on:
                continue
            st.markdown(f"**{grp}**")
            for c in on:
                corps = "<br>".join(_esc(l) for l in formules.textes(R, c))
                st.markdown(
                    f"<div style='margin:2px 0 8px 0;padding:4px 10px;border-left:3px solid "
                    f"{C_TRAIT['ok'] if c.ok else C_TRAIT['nok']};font-size:0.88em;'>"
                    f"<b>{_esc(synthese.court(c))}</b>{badge_nature(c.nat)} "
                    f"<span style='color:{C_TRAIT['mut']};'>{_esc(ref_courte(c.ref))}</span><br>"
                    f"<span style='font-family:ui-monospace,SFMono-Regular,Menlo,monospace;"
                    f"font-size:0.95em;'>{corps}</span></div>",
                    unsafe_allow_html=True)


def _tableau_33(R):
    """Le Tableau 3.3 replié : verdict dans le titre, lignes nommées par le
    moteur (x.lab), boutons Localiser, règles en caption."""
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
                    st.button("Localiser : " + x.lab.split(" (")[0], key=f"dcc_btn_dist_{x.id}",
                              width="stretch", on_click=ui_commun.selectionner, args=(PREFIXE, x.alert))
        st.caption("e ≥ 1,2·d0 ; p1 ≥ 2,2·d0 ; p2 et p3 ≥ 2,4·d0 ; p ≤ min(14t ; 200 mm) ; "
                   "e ≤ 4t + 40 mm si exposé.")
    tableau_33_replie(R.dist_ok, rendre)


def _parametres_retenus(R):
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
    with st.expander("Paramètres retenus"):
        st.markdown(tableau_md(["Paramètre", "Valeur"], g))


def onglet_verifications(R):
    _table_simple(R, "boulons")
    _table_cornieres(R)
    _table_simple(R, "portee")
    _table_simple(R, "poteau")
    _table_simple(R, "cordons")
    # la légende après la DERNIÈRE table, avant les replis (charte §10)
    legende_natures(NATURES, LEGENDE_REFERENCES)
    if R.bolt_S:
        _table_boulons_efforts(R.tabS, "groupe S")
    if R.bolt_P and R.M_P > 0:
        _table_boulons_efforts(R.tabP, "groupe P (une cornière)")
    sans_objet = [synthese.court(c) for c in R.checks if not c.active]
    if R.aile:
        sans_objet.append("cisaillement local de l'aile du poteau : " + PERIMETRE_AILE)
    expander_perimetre(methode.limites(R.ame), sans_objet)
    _formules(R)
    _tableau_33(R)
    _parametres_retenus(R)


# ------------------------------------------------------------ onglet prédim
def _appliquer(i, ecrire):
    R = st.session_state.get("_dcc_R")
    r = R.pd.rows[i]
    if not r.prop:
        return
    ecrire(apply_solution({k: st.session_state.get(K(k)) for k in CHAMPS}, r))
    st.session_state["dcc_ui_alerte"] = None
    st.session_state["dcc_ui_onglet"] = ONGLETS[0]
    st.session_state["_dcc_toast"] = "Solution appliquée aux données et à la géométrie : " + solution_label(r)


def onglet_predim(R, u, ecrire):
    p = R.pd
    rows = [r for r in p.rows if r.geom]
    sols = sorted([r for r in rows if r.eta <= 1],
                  key=lambda r: (0 if r.ok else 1, r.score if r.ok else r.eta))[:9]
    st.caption("Touche « Appliquer » pour recopier une solution dans les données et la géométrie ; elle "
               "devient modifiable comme une saisie. Seule la vérification détaillée fait foi.")
    st.markdown("#### Solutions proposées")
    if sols:
        for i in range(0, len(sols), 3):
            cols = st.columns(3)
            for col, r in zip(cols, sols[i:i + 3]):
                with col:
                    with st.container(border=True):
                        retenue = r is p.pick
                        st.markdown(f"**{r.b['n']} – {r.n} rangées**  \n{nom_cornière_proposee(r)}, "
                                    f"hc {r.Lc} mm")
                        st.caption(f"e1 {r.e1} / p1 {r.p1} / e2 {r.e2} / gA {r.prop.gA} mm  \n"
                                   f"p3 {r.prop.p3} mm · taux estimé " + pct(r.eta)
                                   + (" – retenue par défaut" if retenue else
                                      ("" if r.ok else " – au-dessus du taux cible de "
                                       + pct(N(u.get("eta_c"))))))
                        st.button("Appliquer", key=f"dcc_btn_sol_{p.rows.index(r)}", width="stretch",
                                  type="primary" if retenue else "secondary",
                                  on_click=_appliquer, args=(p.rows.index(r), ecrire))
    else:
        bloc_alerte(p.msg, True)
    st.markdown("#### Détail de la solution retenue par défaut")
    st.markdown(tableau_md(["", ""], [
        ["Boulon proposé", p.boulon], ["Rangées n1 (1 file, groupes S et P)", js_str(p.n)],
        ["e1 / p1 / e2 = e2,b", f"{p.e1} / {p.p1} / {p.e2} mm"],
        ["Hauteur des cornières hc = 2·e1 + (n1 − 1)·p1",
         f"{p.LC} mm (hauteur disponible " + F(p.hav, 0) + " mm)"],
        ["Épaisseur nécessaire au taux cible", F(p.treq, 2) + " mm"],
        ["Cornière proposée", "2 × L" + js_str(p.leg) + "x" + js_str(p.leg) + "x" + js_str(p.tC)
         + " (aile ≥ " + F(p.legreq, 0) + " mm)"],
        ["Trusquinage gA / gauge p3", js_str(p.gA) + " / " + F(p.p3, 0) + " mm"],
        ["Taux estimé (boulons, pressions diamétrales de l'âme et du poteau)", pct(p.eta, 1)],
        ["État", p.msg]]))
    st.markdown("#### Épaisseur nécessaire par vérification")
    st.markdown(tableau_md(["Vérification", "Épaisseur"], [[q[0], F(q[1], 2) + " mm"] for q in p.reqs],
                           droite=(1,)))
    st.markdown("#### Toutes les combinaisons géométriquement possibles")
    h = st.columns([1, 0.6, 0.8, 1, 1, 1, 1, 1, 1.3])
    for c, t in zip(h, ("Boulon", "n1", "hc", "η boulons S", "η p. diam. âme", "η boulons P",
                        "η p. diam. poteau", "η max", "")):
        c.markdown(f"**{t}**" if t else "")
    for r in rows:
        c = st.columns([1, 0.6, 0.8, 1, 1, 1, 1, 1, 1.3], vertical_alignment="center")
        fort = "**" if r is p.pick else ""
        vals = (r.b["n"], js_str(r.n), js_str(r.Lc), pct(r.e.bS), pct(r.e.pdS), pct(r.e.bP),
                pct(r.e.pdP), pct(r.eta))
        for cc, v in zip(c[:8], vals):
            cc.markdown(f"{fort}{v}{fort}")
        with c[8]:
            st.button("Appliquer" + ("" if r.ok else " ⚠"), key=f"dcc_btn_row_{p.rows.index(r)}",
                      width="stretch", on_click=_appliquer, args=(p.rows.index(r), ecrire),
                      help=None if r.ok else "taux cible dépassé")
    st.caption("Règle de choix par défaut : le plus petit nombre de rangées ; à égalité, le plus petit "
               "diamètre. Le POTEAU n'entre dans la proposition que par la pression diamétrale de son "
               "aile ou de son âme, par le gauge minimal p3 ≥ twP + 2·rP + dw (cas a) et — en cas (b) — "
               "par le plafond de l'aile A et du gauge. Seule la vérification détaillée fait foi.")


# -------------------------------------------------------------- onglet note
def onglet_note(R, u):
    c1, c2 = st.columns([1.6, 2.6], vertical_alignment="center")
    complet = c1.checkbox("Texte : toutes les vérifications", key="dcc_ui_txt_complet")
    t = texte.construire_texte(R, complet)
    with c1:
        st.download_button("⬇️ Exporter le texte (.txt)", data=("﻿" + t).encode("utf-8"),
                           file_name=texte.nom_fichier(u, ".txt"), mime="text/plain;charset=utf-8",
                           width="stretch", key="dcc_btn_txt")
    c2.caption("Note de calcul (2 pages, A4 paysage — synthèse + plan de principe) : bouton "
               "« 📄 PDF » de la barre d'outils. Le texte ci-dessous se copie avec l'icône "
               "en haut à droite du cadre.")
    st.code(t, language=None)


# --------------------------------------------------------- onglet benchmark
def _charger_benchmark(inp, ecrire):
    from .entrees import defaults
    d = defaults(); d.update(inp); ecrire(d)
    st.session_state["dcc_ui_alerte"] = None
    st.session_state["dcc_ui_onglet"] = ONGLETS[0]
    st.session_state["_dcc_toast"] = "Exemple chargé : voir Vérifications"


def onglet_benchmark(B, ecrire):
    from acier.assemblages.ui_commun import bloc_statut
    bloc_statut("Benchmark du module : " + B["statut"],
                ["Ces tableaux sont recalculés par le moteur à chaque ouverture ; ils ne dépendent pas "
                 "des données saisies."],
                "ok" if B["ok"] else "nok")
    st.caption("Source : " + SRC["titre"] + ". " + SRC["org"] + ", " + SRC["annee"] + ". Norme : "
               + SRC["norme"] + ". " + SRC["url"])
    st.caption("Seuils : écart ≤ 1 % OK ; 1 à 3 % à expliquer ; > 3 % à vérifier. Aucune formule n'a été "
               "ajustée pour coller à la source.")
    from .benchmark import BENCH
    for i, b in enumerate(B["bench"]):
        st.markdown(f"#### {b['id']}")
        st.caption(b["titre"])
        c1, c2 = st.columns([4, 1.4], vertical_alignment="center")
        c1.caption("Données : " + b["data"])
        with c2:
            st.button("Charger ces données dans l'outil", key=f"dcc_btn_bm_{i}", width="stretch",
                      on_click=_charger_benchmark, args=(BENCH[i]["inp"], ecrire))
        lignes = []
        for r in b["rows"]:
            lab = r["lab"] + ((" — " + r["exp"]) if (r["exp"] and abs(r["ec"]) > 0.003) else "")
            lignes.append([lab, r["page"].replace("-", "‑").replace(" ", " "),
                           F(r["ref"], 1 if r["ref"] < 10 else 0) + " " + r["unit"],
                           F(r["val"], 2 if r["ref"] < 10 else 1), F(r["ec"] * 100, 2) + " %", r["st"]])
        st.markdown(tableau_md(["Vérification", "Page", "Référence", "Outil", "Écart", "Statut"], lignes,
                               droite=(2, 3, 4)))
    st.markdown("#### Vérifications sans exemple publié consulté")
    for x in NOCOVER:
        st.markdown(f"- **{x[0]}** – {x[1]}")
    for c in B["valid"]:
        st.markdown(f"#### {c['id']} – calcul manuel indépendant")
        st.caption(c["titre"] + ". Colonne « manuel » : calcul indépendant hors moteur ; colonne "
                   "« contrôle » : même valeur recomposée à la main (arrondie).")
        lignes = []
        for r in c["rows"]:
            n = 4 if abs(r["hand"]) < 10 else (2 if abs(r["hand"]) < 1000 else 0)
            lignes.append([r["lab"], F(r["hand"], n) + ("" if r["unit"] == "-" else " " + r["unit"]),
                           F(r["xls"], n), F(r["val"], n), F(r["ec"] * 100, 3) + " %", r["st"]])
        st.markdown(tableau_md(["Grandeur", "Manuel", "Contrôle", "Outil", "Écart", ""], lignes,
                               droite=(1, 2, 3, 4)))


def onglet_methode():
    st.markdown(METHODE)


# ------------------------------------------------------------------- onglets
def onglets(R, B, u, ecrire):
    st.session_state["_dcc_R"] = R
    st.segmented_control("Onglet", list(ONGLETS), key="dcc_ui_onglet", label_visibility="collapsed",
                         wrap=True)
    onglet = st.session_state.get("dcc_ui_onglet") or ONGLETS[0]
    if onglet == ONGLETS[1]:
        onglet_predim(R, u, ecrire)
    elif onglet == ONGLETS[2]:
        onglet_note(R, u)
    elif onglet == ONGLETS[3]:
        onglet_benchmark(B, ecrire)
    elif onglet == ONGLETS[4]:
        onglet_methode()
    else:
        onglet_verifications(R)

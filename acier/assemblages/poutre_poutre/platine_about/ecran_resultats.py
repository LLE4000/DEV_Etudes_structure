# -*- coding: utf-8 -*-
"""Résultats à l'écran de la platine d'about : bandeau d'une ligne, alertes
courtes, dessins cliquables (colonne figée), onglets Vérifications · Prédim ·
Note · Benchmark · Méthode. Tout est lu dans ``R`` (moteur) et ``B``
(benchmark).

Charte du 31/10/2026 : les fenêtres d'objet, le panneau Cotes et les chips
d'alerte lisent ``LIBELLE`` et ``AIDES`` de la carte (une forme par clé) ;
l'onglet Vérifications range la légende, le périmètre et le Tableau 3.3
dans des replis (rien ne se perd, tout est dans ``md(at)``)."""
import streamlit as st

from acier.js import N, js_str
from acier.formats import F, pct
from acier.assemblages.ui_commun import (bloc_statut_ligne, bloc_alerte_ligne, bloc_alerte, badge_nature,
                                         tableau_md, tableau_html, taux_html, verdict_html, Html, _esc,
                                         C_TRAIT, legende_natures, expander_perimetre, tableau_33_replie)
from acier.bibliotheques import PERSO
from . import schemas, texte, synthese, formules
from .notation import ligne_alerte, ref_courte, LEGENDE_REFERENCES
from .entrees import CHAMPS, CLES, champ_visible, PILOTEES_PAR_PREDIM, traction
from .moteur import apply_solution, solution_label
from .benchmark import SRC, NOCOVER
from .methode import METHODE
from .ecran_saisie import K, valeur_widget, PAS, LIBELLE, AIDES, AFFICHAGE

ONGLETS = ("Vérifications", "Prédim", "Note", "Benchmark", "Méthode")
NIVEAUX_ECRAN = ("Vue simple", "Édition", "Cotations complètes")
CLES_TETE_DESSIN = ("ui_niveau", "ui_composant")      # réglages d'écran en tête de la colonne dessin
NATURES = ("EC", "COMP", "INT")
NON_REVERIFIES = ("flexion TRANSVERSALE de l'âme porteuse sous la traction des boulons (NEd, tying)",
                  "capacité de rotation au-delà des règles de détail",
                  "cisaillement global, flexion et déversement de la poutre principale",
                  "torsion induite", "deux poutres en vis-à-vis sur la même âme")
CAPTION_USAGE = ("Touche une cote pour la modifier, touche une pièce (poutre, platine, boulon, cordon, VEd) pour "
                 "ses paramètres ; glisse la poutre portée (Δz).")


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
        bloc_alerte("Mode prédimensionnement : " + R.boulon + ", " + js_str(R.n_1) + " rangées × 2, platine "
                    + js_str(R.h_p) + " × " + js_str(R.b_p) + " × " + js_str(R.t_p) + ". " + R.pd.msg, False)


# ------------------------------------------------------------------ alertes
def _selectionner(ident, a_geo):
    cur = st.session_state.get("pab_ui_alerte")
    st.session_state["pab_ui_alerte"] = None if cur == ident else ident


def _ecrire_depuis(cle_widget, k):
    st.session_state[K(k)] = valeur_widget(k, st.session_state.get(cle_widget))


def _mini_champ(k, u, prefixe, disabled=False):
    """La chip d'une alerte : le même champ, modifiable sur place (libellé
    LIBELLE, options AFFICHAGE) ; écrit dans la source unique."""
    f = CHAMPS[k]
    cle = prefixe + k
    st.session_state[cle] = valeur_widget(k, st.session_state.get(K(k)))
    lab = LIBELLE.get(k, k)
    if f["t"] == "s":
        aff = AFFICHAGE.get(k)
        st.selectbox(lab, f["o"], key=cle, on_change=_ecrire_depuis, args=(cle, k), disabled=disabled,
                     format_func=(lambda o, a=aff: a.get(o, o)) if aff else str)
    elif f["t"] == "x":
        st.text_input(lab, key=cle, on_change=_ecrire_depuis, args=(cle, k), disabled=disabled)
    else:
        st.number_input(lab, key=cle, step=PAS.get(k, 1.0), format="%g",
                        on_change=_ecrire_depuis, args=(cle, k), disabled=disabled)


def alertes(R, u):
    sel = st.session_state.get("pab_ui_alerte")
    ids = [a.id for a in R.alerts]
    if sel and sel not in ids:
        st.session_state["pab_ui_alerte"] = None
        sel = None
        st.toast("Blocage levé : géométrie valide" if (R.geo_ok and R.dist_ok)
                 else "Ce blocage est levé ; il en reste d'autres")
    for a in R.alerts:
        est_sel = sel == a.id
        titre, detail = ligne_alerte(a, R)
        c1, c2 = st.columns([7, 1], vertical_alignment="center", gap="small")
        with c1:
            bloc_alerte_ligne(titre, detail, a.block, est_sel)
        with c2:
            st.button("Masquer" if est_sel else "Localiser", key=f"pab_btn_al_{a.id}",
                      width="stretch", on_click=_selectionner,
                      args=(a.id, bool(a.dims or a.elems)))
        if est_sel:
            with st.container(border=True):
                st.markdown(a.why if a.why else a.msg)
                if R.pred and any(k in PILOTEES_PAR_PREDIM for k in a.fields):
                    st.caption("Mode prédimensionnement : les champs marqués ◆ sont pilotés par la "
                               "proposition (onglet Prédim, ou mode Vérification pour les modifier).")
                visibles = [k for k in a.fields if champ_visible(u, k)]
                if visibles:
                    st.caption("Champs en cause — modifiables ici :")
                    cols = st.columns(min(4, len(visibles)))
                    for i, k in enumerate(visibles):
                        with cols[i % len(cols)]:
                            _mini_champ(k, u, "pab_fix_", disabled=R.pred and k in PILOTEES_PAR_PREDIM)
                if a.dims or a.elems:
                    st.caption("Sur le dessin : cotes et éléments en cause en rouge.")


# ------------------------------------------------------------------ dessins
def _traiter_clic(retour, cle_cmp):
    if not retour or not isinstance(retour, dict):
        return
    derniers = st.session_state.setdefault("pab_ui_dernier_clic", {})
    if derniers.get(cle_cmp) == retour.get("t"):
        return
    derniers[cle_cmp] = retour.get("t")
    changes = retour.get("changes")
    if changes is None and retour.get("key"):
        changes = {retour["key"]: retour.get("value")}
    ecrits = []
    for k, v in (changes or {}).items():
        if k in CHAMPS:
            st.session_state[K(k)] = valeur_widget(k, v)
            ecrits.append(LIBELLE.get(k, k) + " = " + js_str(valeur_widget(k, v)))
    if ecrits:
        st.toast(" ; ".join(ecrits) + " : géométrie et vérifications recalculées")
        st.rerun()


def _panneaux(R, u):
    """Les fenêtres d'objet du composant : un clic sur une pièce ouvre TOUS
    ses paramètres — une VUE des mêmes clés (libellés LIBELLE, aides AIDES),
    titres courts, aucun doublon (a_w ne reste que dans Soudure) ; la fenêtre
    Poutre secondaire porte aussi Δz, dc,sup, dc,inf et c."""
    pred = R.pred

    def ch(k):
        f = CHAMPS[k]
        aff = AFFICHAGE.get(k)
        return dict(key=k, lab=LIBELLE[k], type=f["t"], options=f.get("o") or [], step=PAS.get(k, 1.0),
                    dis=bool(pred and k in PILOTEES_PAR_PREDIM), help=AIDES.get(k),
                    affichage=aff or None)

    def perso(X):
        return [ch(f"{d}{X}_u") for d in ("h", "b", "tw", "tf", "r")]

    g = {}
    g["G"] = dict(titre="Groupe", champs=[ch("n1_u"), ch("p1_u"), ch("e1_u"), ch("p3_u")],
                  note="Deux files symétriques par rapport à l'âme de la poutre portée.")
    g["bolts"] = dict(titre="Boulons",
                      champs=[ch("boulon_u"), ch("classe"), ch("trou"), ch("cat"), ch("filet")]
                      + ([ch("mu_s"), ch("k_s")] if u.get("cat") != "A" else [])
                      + ([ch("k_ser")] if u.get("cat") == "B" else []))
    g["beamS"] = dict(titre="Poutre secondaire",
                      champs=[ch("prof_S"), ch("nu_S")]
                      + (perso("S") if u.get("prof_S") == PERSO else [])
                      + [ch("lt_ok"), ch("r_n"), ch("d_top"), ch("d_nt"), ch("d_nb"), ch("l_n")],
                      note="Δz, dc,sup, dc,inf et c : aussi des cotes sur le dessin — la poutre se déplace aussi "
                           "verticalement à la souris.")
    g["beamP"] = dict(titre="Poutre principale",
                      champs=[ch("prof_P"), ch("nu_P")]
                      + (perso("P") if u.get("prof_P") == PERSO else []))
    g["plate"] = dict(titre="Platine",
                      champs=[ch("hp_u"), ch("tp_u"), ch("bp_u"), ch("z_C"), ch("nu_pl")],
                      note="hp, bp, tp et zp : aussi des cotes sur le dessin ; la gorge a : fenêtre Soudure.")
    g["weld"] = dict(titre="Soudure", champs=[ch("a_w")])
    # la nature de NEd n'a de sens qu'en traction : même règle que la carte (entrees.traction)
    g["efforts"] = dict(titre="Efforts", champs=[ch("V_Ed"), ch("N_Ed")] + ([ch("nat_N")] if traction(u) else []))
    return g


def _alerte_courante(R):
    sel = st.session_state.get("pab_ui_alerte")
    for a in R.alerts:
        if a.id == sel:
            return dict(dims=set(a.dims), elems=set(a.elems))
    return None


def dessins(R, u):
    """La colonne du dessin : en tête le niveau de cotation et la case
    « Dessin interactif », puis élévation et vue en plan empilées (colonne
    figée par interface.py). Renvoie ``(e, p, hl)`` pour le repli."""
    with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="center", key="pab_tete_dessin"):
        niveau = st.radio(LIBELLE["ui_niveau"], [0, 1, 2], format_func=lambda i: NIVEAUX_ECRAN[i],
                          horizontal=True, key="pab_ui_niveau", label_visibility="collapsed")
        st.checkbox(LIBELLE["ui_composant"], key="pab_ui_composant", help=AIDES["ui_composant"])
    interactif = bool(st.session_state.get("pab_ui_composant", True))
    hl = _alerte_courante(R)
    opt = schemas.options_ecran(R, niveau, hl)
    opt.editables = niveau >= 1
    e = schemas.elevation(R, opt); p = schemas.plan(R, opt)
    valeurs = {k: st.session_state.get(K(k)) for k in CLES}
    groupes = _panneaux(R, u)
    for d, cle in ((e, "pab_cmp_elev"), (p, "pab_cmp_plan")):
        if interactif:
            from acier.composants.svg_cliquable import svg_cliquable
            _traiter_clic(svg_cliquable(d.svg(identifiant=cle), valeurs, key=cle, groupes=groupes), cle)
        else:
            st.markdown(d.svg(identifiant=cle), unsafe_allow_html=True)
    if interactif and not R.pred:
        st.caption(CAPTION_USAGE)       # en prédim, le bandeau ocre et les ◆ suffisent (P9)
    return e, p, hl


def panneau_cotes(e, p, hl):
    """Repli (tête de la colonne de droite) : une saisie par cote, libellé
    LIBELLE, aide AIDES."""
    cotes = []
    vus = set()
    for k, sym, ident in schemas.cotes_modifiables(e) + schemas.cotes_modifiables(p):
        if k not in vus:
            vus.add(k); cotes.append((k, sym, ident))
    with st.expander("Cotes", expanded=True):
        if not cotes:
            st.caption("Aucune cote modifiable dans cette configuration.")
        chauds = set(hl["dims"]) if hl else set()
        for i in range(0, len(cotes), 4):
            cols = st.columns(4)
            for col, (k, sym, ident) in zip(cols, cotes[i:i + 4]):
                with col:
                    cle = "pab_cote_" + k
                    st.session_state[cle] = valeur_widget(k, st.session_state.get(K(k)))
                    st.number_input(("🔴 " if ident in chauds else "") + LIBELLE.get(k, sym + " (mm)"), key=cle,
                                    step=PAS.get(k, 1.0), format="%g",
                                    on_change=_ecrire_depuis, args=(cle, k), help=AIDES.get(k))


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


def _formules(R):
    with st.expander("Formules et substitutions numériques"):
        st.caption("Les valeurs introduites sont composées par le moteur ; la ligne du taux est vérifiée "
                   "par un test qui la recalcule.")
        for grp in ("Boulons", "Platine", "Poutre secondaire", "Poutre principale", "Soudure"):
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
                    f"<span style='font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:0.95em;'>{corps}</span></div>",
                    unsafe_allow_html=True)


def _tableau_33(R):
    """Le Tableau 3.3 replié : verdict dans le titre, lignes nommées par le
    moteur (x.lab), boutons Localiser, règles en caption."""
    def rendre():
        if not R.dist:
            st.caption("Sans objet.")
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
                    st.button("Localiser : " + x.lab.split(" (")[0], key=f"pab_btn_dist_{x.id}", width="stretch",
                              on_click=_selectionner, args=(x.alert, True))
        st.caption("e ≥ 1,2·d0 ; p1 ≥ 2,2·d0 ; p3 (p2 du tableau) ≥ 2,4·d0 ; p ≤ min(14t ; 200 mm) ; "
                   "e ≤ 4t + 40 mm si exposé.")
    tableau_33_replie(R.dist_ok, rendre)


def _parametres_retenus(R):
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
    with st.expander("Paramètres retenus"):
        st.markdown(tableau_md(["Paramètre", "Valeur"], g))


def onglet_verifications(R):
    for cle in ("boulons", "platine", "portee", "porteuse", "soudure"):
        _table_simple(R, cle)
    legende_natures(NATURES, LEGENDE_REFERENCES)
    expander_perimetre(NON_REVERIFIES, [synthese.court(c) for c in R.checks if not c.active])
    _formules(R)
    _tableau_33(R)
    _parametres_retenus(R)


# ------------------------------------------------------------ onglet prédim
def _appliquer(i, ecrire):
    R = st.session_state.get("_pab_R")
    r = R.pd.rows[i]
    if not r.prop:
        return
    ecrire(apply_solution({k: st.session_state.get(K(k)) for k in CHAMPS}, r))
    st.session_state["pab_ui_alerte"] = None
    st.session_state["pab_ui_onglet"] = ONGLETS[0]
    st.session_state["_pab_toast"] = "Solution appliquée aux données et à la géométrie : " + solution_label(r)


def onglet_predim(R, u, ecrire):
    p = R.pd
    rows = [r for r in p.rows if r.geom]
    sols = sorted([r for r in rows if r.eta <= 1],
                  key=lambda r: (0 if r.ok else 1, r.score if r.ok else r.eta))[:9]
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
                        st.markdown(f"**{r.b['n']} – {r.n} rangées × 2**  \nplatine {r.hp} × "
                                    + js_str(r.prop.bp if r.prop else 0) + " × "
                                    + js_str(r.prop.tp if r.prop else 0) + " mm")
                        st.caption(f"e1 {r.e1} / p1 {r.p1} / p3 {r.p3} mm  \ntaux estimé " + pct(r.eta)
                                   + (" – retenue par défaut" if retenue else
                                      ("" if r.ok else " – au-dessus du taux cible de " + pct(N(u.get("eta_c"))))))
                        st.button("Appliquer", key=f"pab_btn_sol_{p.rows.index(r)}", width="stretch",
                                  type="primary" if retenue else "secondary",
                                  on_click=_appliquer, args=(p.rows.index(r), ecrire))
    else:
        bloc_alerte(p.msg, True)
    st.markdown("#### Détail de la solution retenue par défaut")
    st.markdown(tableau_md(["", ""], [
        ["Boulon proposé", p.boulon], ["Rangées n1 (2 files)", js_str(p.n)],
        ["e1 / p1 / e2", f"{p.e1} / {p.p1} / {p.e2} mm"],
        ["Entraxe des files p3 (≥ 90 mm, rondelle hors cordon)", f"{p.p3} mm"],
        ["Hauteur de la platine hp = max(2·e1 + (n1 − 1)·p1 ; 0,6·h)",
         f"{p.hp} mm (hauteur disponible " + F(p.hav, 0) + " mm)"],
        ["Épaisseur nécessaire au taux cible", F(p.treq, 2) + " mm"],
        ["Platine proposée", js_str(p.hp) + " × " + js_str(p.bp) + " × " + js_str(p.tp) + " mm – gorge a = " + js_str(p.a) + " mm"],
        ["Taux estimé (boulons, âme portée, âme porteuse)", pct(p.eta, 1)], ["État", p.msg]]))
    st.markdown("#### Épaisseur nécessaire par vérification")
    st.markdown(tableau_md(["Vérification", "Épaisseur"], [[q[0], F(q[1], 2) + " mm"] for q in p.reqs], droite=(1,)))
    st.markdown("#### Toutes les combinaisons géométriquement possibles")
    h = st.columns([1, 0.6, 0.8, 1, 1, 1, 1, 1.3])
    for c, t in zip(h, ("Boulon", "n1", "hp", "η boulons", "η âme portée", "η p. diam. P", "η max", "")):
        c.markdown(f"**{t}**" if t else "")
    for r in rows:
        c = st.columns([1, 0.6, 0.8, 1, 1, 1, 1, 1.3], vertical_alignment="center")
        fort = "**" if r is p.pick else ""
        vals = (r.b["n"], js_str(r.n), js_str(r.hp), pct(r.e.bv), pct(r.e.vwS), pct(r.e.pdP), pct(r.eta))
        for cc, v in zip(c[:7], vals):
            cc.markdown(f"{fort}{v}{fort}")
        with c[7]:
            st.button("Appliquer" + ("" if r.ok else " ⚠"), key=f"pab_btn_row_{p.rows.index(r)}",
                      width="stretch", on_click=_appliquer, args=(p.rows.index(r), ecrire),
                      help=None if r.ok else "taux cible dépassé")
    st.caption("Règle de choix par défaut : le plus petit nombre de rangées ; à égalité, le plus petit "
               "diamètre. L'épaisseur de la platine reste entre 8 mm et la limite de ductilité ; la gorge "
               "proposée est la pleine résistance du guide. Seule la vérification détaillée fait foi.")


# -------------------------------------------------------------- onglet note
def onglet_note(R, u):
    c1, c2 = st.columns([1.6, 2.6], vertical_alignment="center")
    complet = c1.checkbox("Texte : toutes les vérifications", key="pab_ui_txt_complet")
    t = texte.construire_texte(R, complet)
    with c1:
        st.download_button("⬇️ Exporter le texte (.txt)", data=("﻿" + t).encode("utf-8"),
                           file_name=texte.nom_fichier(u, ".txt"), mime="text/plain;charset=utf-8",
                           width="stretch", key="pab_btn_txt")
    c2.caption("Note de calcul (2 pages, A4 paysage — synthèse + plan de principe) : bouton "
               "« 📄 PDF » de la barre d'outils. Le texte ci-dessous se copie avec l'icône "
               "en haut à droite du cadre.")
    st.code(t, language=None)


# --------------------------------------------------------- onglet benchmark
def _charger_benchmark(inp, ecrire):
    from .entrees import defaults
    d = defaults(); d.update(inp); ecrire(d)
    st.session_state["pab_ui_alerte"] = None
    st.session_state["pab_ui_onglet"] = ONGLETS[0]
    st.session_state["_pab_toast"] = "Exemple chargé : voir Vérifications"


def _nd(x):
    """Décimales d'affichage d'une valeur de référence."""
    return 2 if abs(x) < 1 else (1 if abs(x) < 10 else 0)


def onglet_benchmark(B, ecrire):
    from acier.assemblages.ui_commun import bloc_statut
    bloc_statut("Benchmark du module : " + B["statut"],
                ["Ces tableaux sont recalculés par le moteur à chaque ouverture ; ils ne dépendent pas des données saisies."],
                "ok" if B["ok"] else "nok")
    st.caption("Source : " + SRC["titre"] + ". " + SRC["org"] + ", " + SRC["annee"] + ". Norme : " + SRC["norme"] + ". " + SRC["url"])
    st.caption("Seuils : écart ≤ 1 % OK ; 1 à 3 % à expliquer ; > 3 % à vérifier. Aucune formule n'a été ajustée pour coller à la source.")
    from .benchmark import BENCH
    for i, b in enumerate(B["bench"]):
        st.markdown(f"#### {b['id']}")
        st.caption(b["titre"])
        c1, c2 = st.columns([4, 1.4], vertical_alignment="center")
        c1.caption("Données : " + b["data"])
        with c2:
            st.button("Charger ces données dans l'outil", key=f"pab_btn_bm_{i}", width="stretch",
                      on_click=_charger_benchmark, args=(BENCH[i]["inp"], ecrire))
        lignes = []
        for r in b["rows"]:
            lab = r["lab"] + ((" — " + r["exp"]) if (r["exp"] and abs(r["ec"]) > 0.003) else "")
            n = _nd(r["ref"])
            lignes.append([lab, r["page"].replace("-", "‑").replace(" ", " "),
                           F(r["ref"], n) + ("" if r["unit"] == "-" else " " + r["unit"]),
                           F(r["val"], n + 1), F(r["ec"] * 100, 2) + " %", r["st"]])
        st.markdown(tableau_md(["Vérification", "Réf.", "Référence", "Outil", "Écart", "Statut"], lignes, droite=(2, 3, 4)))
    st.markdown("#### Vérifications sans exemple publié consulté")
    for x in NOCOVER:
        st.markdown(f"- **{x[0]}** – {x[1]}")
    for c in B["valid"]:
        st.markdown(f"#### {c['id']} – calcul manuel indépendant")
        st.caption(c["titre"] + ". Colonne « manuel » : calcul indépendant hors moteur ; colonne « contrôle » : "
                   "même valeur recomposée à la main (arrondie).")
        lignes = []
        for r in c["rows"]:
            n = 4 if abs(r["hand"]) < 10 else (2 if abs(r["hand"]) < 1000 else 0)
            lignes.append([r["lab"], F(r["hand"], n) + ("" if r["unit"] == "-" else " " + r["unit"]),
                           F(r["xls"], n), F(r["val"], n), F(r["ec"] * 100, 3) + " %", r["st"]])
        st.markdown(tableau_md(["Grandeur", "Manuel", "Contrôle", "Outil", "Écart", ""], lignes, droite=(1, 2, 3, 4)))


def onglet_methode():
    st.markdown(METHODE)


# ------------------------------------------------------------------- onglets
def onglets(R, B, u, ecrire):
    st.session_state["_pab_R"] = R
    st.segmented_control("Onglet", list(ONGLETS), key="pab_ui_onglet", label_visibility="collapsed", wrap=True)
    onglet = st.session_state.get("pab_ui_onglet") or ONGLETS[0]
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

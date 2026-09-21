# -*- coding: utf-8 -*-
"""Colonne de résultats : bandeau d'état, alertes cliquables, et les six
onglets — Schéma et géométrie (dessins cliquables + panneau des cotes),
Vérifications, Prédim, Rapport et export, Benchmark, Méthode.

Tout est lu dans ``R`` (moteur) et ``B`` (benchmark) ; rien n'est recalculé.
"""
import streamlit as st

from acier.js import N, js_str
from acier.formats import F, pct
from acier.assemblages.ui_commun import (bloc_verification, bloc_statut, bloc_alerte,
                                         badge_nature, tableau_md)
from . import schemas, texte
from .entrees import (COURT, UNITE, CHAMPS, champ_visible, mode_predim, PILOTEES_PAR_PREDIM)
from .moteur import GROUPES_VERIF, apply_solution, solution_label, nom_cornière_proposee
from .benchmark import SRC, NOCOVER, BENCH
from .methode import METHODE
from .ecran_saisie import K, valeur_widget, PAS

ONGLETS = ("Schéma et géométrie", "Vérifications", "Prédim", "Rapport et export",
           "Benchmark", "Méthode")


# ------------------------------------------------------------------ bandeau
def bandeau(R, B):
    ok = R.verified
    titre = R.statut + (" – sous réserve des alertes" if ok and (R.reserve or R.alerts) else "")
    l1 = "Taux maximal " + pct(R.eta_max, 1) + (" : " + R.gov.label if R.gov else "")
    if not R.geo_ok:
        l1 += " | géométrie non valide"
    if not R.dist_ok:
        l1 += " | pinces ou entraxes non conformes"
    bloc_statut(titre, [l1], "ok" if ok else "nok", "Benchmark du module : " + B["statut"])
    if R.pred:
        bloc_alerte("Mode prédimensionnement : " + R.boulon + ", " + js_str(R.n1_S) + " rangées, " + R.corn_txt
                    + ", Lc = " + F(R.L_C, 0) + " mm. " + R.pd.msg, False)


# ------------------------------------------------------------------ alertes
def _selectionner(ident, a_geo):
    cur = st.session_state.get("asm_ui_alerte")
    st.session_state["asm_ui_alerte"] = None if cur == ident else ident
    if st.session_state["asm_ui_alerte"] and a_geo:
        st.session_state["asm_ui_onglet"] = ONGLETS[0]


def _ecrire_depuis(cle_widget, k):
    """Callback : recopie la valeur d'un widget secondaire (chip d'alerte,
    panneau des cotes) dans la clé de session du champ (source unique)."""
    st.session_state[K(k)] = valeur_widget(k, st.session_state.get(cle_widget))


def _mini_champ(k, u, prefixe, disabled=False):
    """Un champ « en cause » modifiable sur place (clé secondaire, recopiée
    dans la source unique à chaque changement)."""
    f = CHAMPS[k]
    cle = prefixe + k
    st.session_state[cle] = valeur_widget(k, st.session_state.get(K(k)))
    lab = COURT.get(k, f["l"]) + (UNITE.get(k, "") and " (" + UNITE[k].strip() + ")")
    if f["t"] == "s":
        st.selectbox(lab, f["o"], key=cle, on_change=_ecrire_depuis, args=(cle, k), disabled=disabled)
    elif f["t"] == "x":
        st.text_input(lab, key=cle, on_change=_ecrire_depuis, args=(cle, k), disabled=disabled)
    else:
        st.number_input(lab, key=cle, step=PAS.get(k, 1.0), format="%g",
                        on_change=_ecrire_depuis, args=(cle, k), disabled=disabled)


def alertes(R, u):
    sel = st.session_state.get("asm_ui_alerte")
    ids = [a.id for a in R.alerts]
    if sel and sel not in ids:
        st.session_state["asm_ui_alerte"] = None
        sel = None
        st.toast("Blocage levé : géométrie valide" if (R.geo_ok and R.dist_ok)
                 else "Ce blocage est levé ; il en reste d'autres")
    for a in R.alerts:
        est_sel = sel == a.id
        c1, c2 = st.columns([6, 1.1], vertical_alignment="center")
        with c1:
            bloc_alerte(a.msg, a.block, est_sel)
        with c2:
            st.button("Masquer" if est_sel else "Localiser", key=f"asm_btn_al_{a.id}",
                      use_container_width=True, on_click=_selectionner,
                      args=(a.id, bool(a.dims or a.elems)))
        if est_sel:
            with st.container(border=True):
                if a.why:
                    st.markdown(a.why)
                if R.pred and any(k in PILOTEES_PAR_PREDIM for k in a.fields):
                    st.caption("Mode prédimensionnement : les champs marqués ◆ sont pilotés par la "
                               "proposition. Applique une solution de l'onglet Prédim, ou passe en "
                               "mode VÉRIFICATION, pour les modifier.")
                visibles = [k for k in a.fields if champ_visible(u, k)]
                if visibles:
                    st.caption("Champs en cause — modifiables ici :")
                    cols = st.columns(min(3, len(visibles)))
                    for i, k in enumerate(visibles):
                        with cols[i % len(cols)]:
                            _mini_champ(k, u, "asm_fix_", disabled=R.pred and k in PILOTEES_PAR_PREDIM)
                if a.dims or a.elems:
                    st.caption("Sur le dessin : cotes et éléments en cause en rouge (onglet Schéma et géométrie).")


# ------------------------------------------------------ onglet géométrie
def _traiter_clic(retour, cle_cmp):
    """Un clic validé sur une cote du dessin → source unique → recalcul."""
    if not retour or not isinstance(retour, dict):
        return
    derniers = st.session_state.setdefault("asm_ui_dernier_clic", {})
    if derniers.get(cle_cmp) == retour.get("t"):
        return
    derniers[cle_cmp] = retour.get("t")
    k = retour.get("key")
    if k in CHAMPS:
        st.session_state[K(k)] = valeur_widget(k, retour.get("value"))
        st.toast(f"{retour.get('sym', k)} = {js_str(valeur_widget(k, retour.get('value')))} mm : "
                 "géométrie et vérifications recalculées")
        st.rerun()


def _dessins(R, u):
    niveau = st.radio("Niveau de cotation", [0, 1, 2], format_func=lambda i: schemas.NIVEAUX[i],
                      horizontal=True, key="asm_ui_niveau", label_visibility="collapsed")
    hl = None
    sel = st.session_state.get("asm_ui_alerte")
    for a in R.alerts:
        if a.id == sel:
            hl = dict(dims=set(a.dims), elems=set(a.elems))
    opt = schemas.options_ecran(R, niveau, hl)
    e = schemas.elevation(R, opt); p = schemas.plan(R, opt)
    st.caption("Mode prédimensionnement : les cotes pilotées par la proposition (◆) ne se modifient pas ici."
               if R.pred else
               "Cotes sur fond jaune : touche la valeur, saisis la nouvelle, valide avec Entrée. "
               "Le formulaire et les vérifications suivent. En violet italique : grandeurs calculées, non modifiables.")
    interactif = bool(st.session_state.get("asm_ui_composant", True))
    valeurs = {k: st.session_state.get(K(k)) for k in schemas.CLE_PAR_COTE.values()}
    c1, c2 = st.columns([3, 2])
    for col, d, cle in ((c1, e, "asm_cmp_elev"), (c2, p, "asm_cmp_plan")):
        with col:
            if interactif:
                from acier.composants.svg_cliquable import svg_cliquable
                _traiter_clic(svg_cliquable(d.svg(identifiant=cle), valeurs, key=cle), cle)
            else:
                st.markdown(d.svg(identifiant=cle), unsafe_allow_html=True)
            st.caption("Élévation à l'échelle, cotes en mm." if d is e else "Vue en plan")
    # panneau des cotes : la même source, une saisie par cote modifiable
    cotes = []
    vus = set()
    for k, sym, ident in schemas.cotes_modifiables(e) + schemas.cotes_modifiables(p):
        if k not in vus:
            vus.add(k); cotes.append((k, sym, ident))
    # cotes présentes sur le dessin mais non éditables (verrouillées en prédim)
    with st.expander("Cotes", expanded=not interactif):
        if not cotes:
            st.caption("Aucune cote modifiable dans cette configuration.")
        chauds = set(hl["dims"]) if hl else set()
        for i in range(0, len(cotes), 4):
            cols = st.columns(4)
            for col, (k, sym, ident) in zip(cols, cotes[i:i + 4]):
                with col:
                    cle = "asm_cote_" + k
                    st.session_state[cle] = valeur_widget(k, st.session_state.get(K(k)))
                    # libellé court non ambigu (« e1 groupe S » / « e1 groupe P »)
                    st.number_input(("🔴 " if ident in chauds else "") + f"{COURT.get(k, sym)} (mm)", key=cle,
                                    step=PAS.get(k, 1.0), format="%g",
                                    on_change=_ecrire_depuis, args=(cle, k), help=CHAMPS[k]["l"])


def _parametres_retenus(R):
    g = [["Boulon retenu", R.boulon + " – d0 = " + F(R.d_0, 0) + " mm"],
         ["Cornières", R.corn_txt + " – aile A " + F(R.b_A, 0) + ", aile B " + F(R.b_B, 0) + ", tc " + F(R.t_C, 0) + ", rc " + F(R.r_C, 0) + " mm"],
         ["Longueur des cornières Lc", F(R.L_C, 0) + " mm" + ("" if R.rec_L else " (Lc < 0,6·h : recommandation MSB Part 5 §4.1 non respectée)")],
         ["Excentricité z : face de l'âme porteuse → centre du groupe S " + ("" if R.bolt_S else "(cordons)"), F(R.zeff, 1) + " mm"],
         ["Moment d'excentricité MS = VEd·z + |MEd|", F(R.M_S, 2) + " kNm"],
         ["Modèle du groupe S", R.mod_S], ["Modèle du groupe P", R.mod_P],
         ["Moment par groupe P (option)", F(R.M_P, 2) + " kNm"],
         ["Inertie polaire du groupe S", F(R.Ip_S, 0) + " mm²"],
         ["Entraxe p3 entre les files des deux cornières", F(R.p_3, 1) + " mm"],
         ["Pince e1,b dans l'âme secondaire", F(R.e1b_S, 1) + " mm"],
         ["Pinces e2 : aile B / aile A", F(R.e2a_S, 1) + " / " + F(R.e2a_P, 1) + " mm"],
         ["Pinces basses e1 : groupe S / groupe P", F(R.e1bot_S, 1) + " / " + F(R.e1bot_P, 1) + " mm"]]
    st.markdown("#### Paramètres retenus et excentricités")
    st.markdown(tableau_md(["Paramètre", "Valeur"], g))


def _tableau_33(R):
    st.markdown("#### Pinces et entraxes – EN 1993-1-8 Tableau 3.3")
    if not R.dist:
        st.caption("Sans objet (aucun boulon).")
        return
    h = st.columns([3.2, 1, 1, 1, 1.6])
    for c, t in zip(h, ("Distance", "Valeur", "Min", "Max", "Statut")):
        c.markdown(f"**{t}**")
    for x in R.dist:
        c = st.columns([3.2, 1, 1, 1, 1.6], vertical_alignment="center")
        c[0].markdown(f"{x.lab}<br><span style='color:#6b7280;font-size:0.85em'>{x.rule}</span>", unsafe_allow_html=True)
        c[1].markdown(F(x.val, 1)); c[2].markdown(F(x.min, 1)); c[3].markdown("—" if x.max is None else F(x.max, 0))
        with c[4]:
            if x.ok:
                st.markdown("<span style='color:#2E7D46;font-weight:700'>OK</span>", unsafe_allow_html=True)
            else:
                st.button("NON OK – localiser", key=f"asm_btn_dist_{x.id}", use_container_width=True,
                          on_click=_selectionner, args=(x.alert, True))


def onglet_geometrie(R, u):
    _dessins(R, u)
    _parametres_retenus(R)
    _tableau_33(R)


# ------------------------------------------------------ onglet vérifications
def _table_boulons(t, lab):
    with st.expander("Répartition élastique par boulon – " + lab):
        st.markdown(tableau_md(["Rangée / file", "x", "y", "Fx", "Fz", "F [kN]"],
                               [[f"{b.i} / {b.j}", F(b.x, 1), F(b.y, 1), F(b.fx, 2), F(b.fz, 2), F(b.f, 2)] for b in t],
                               droite=(1, 2, 3, 4, 5)))


def onglet_verifications(R):
    st.markdown(badge_nature("EC") + " formule directe de l'Eurocode " + badge_nature("COMP")
                + " modèle complémentaire reconnu (MSB Part 5, ECCS n°126, SCI P358) " + badge_nature("INT")
                + " interprétation de l'outil. Déplie une ligne pour voir la formule, les valeurs et la référence.",
                unsafe_allow_html=True)
    for g in GROUPES_VERIF:
        a = [c for c in R.checks if c.grp == g]
        on = [c for c in a if c.active]; off = [c for c in a if not c.active]
        if not on:
            continue
        st.markdown(f"#### {g}")
        for c in on:
            bloc_verification(c)
            with st.expander("Formule, valeurs et référence"):
                st.markdown(f"*{c.formula}*")
                if c.vals:
                    st.markdown(c.vals)
                nd = 3 if c.unit == "-" else 1
                st.markdown("Sollicitation = " + F(c.Ed, 3) + " ; résistance = " + F(c.Rd, 3)
                            + ("" if c.unit == "-" else " " + c.unit) + " ; η = " + pct(c.eta, 1))
                st.markdown(f"<span style='color:#6b7280'>Réf. : {c.ref}</span> {badge_nature(c.nat)}",
                            unsafe_allow_html=True)
        if g == "Boulons":
            if R.bolt_S:
                _table_boulons(R.tabS, "groupe S")
            if R.bolt_P and R.M_P > 0:
                _table_boulons(R.tabP, "groupe P (une cornière)")
        if g == "Poutre principale":
            st.caption("Non revérifiés ici : cisaillement global, flexion et déversement de la poutre "
                       "principale ; torsion induite par une attache d'un seul côté ; flexion hors plan de "
                       "l'âme sous NEd ; deux poutres en vis-à-vis partageant les mêmes boulons.")
        if off:
            st.caption("Sans objet dans cette configuration : " + " ; ".join(c.label for c in off) + ".")


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
    st.caption(("Mode PRÉDIMENSIONNEMENT actif : la solution retenue par défaut pilote le calcul ("
                + R.statut + ", taux maximal " + pct(R.eta_max, 1) + ")." if R.pred else "Mode VÉRIFICATION actif.")
               + " Touche une solution pour la recopier dans les données et la géométrie : elle devient "
               "alors modifiable comme une saisie. Seule la vérification détaillée fait foi.")
    st.markdown("#### Solutions proposées")
    if sols:
        for i in range(0, len(sols), 3):
            cols = st.columns(3)
            for col, r in zip(cols, sols[i:i + 3]):
                with col:
                    with st.container(border=True):
                        retenue = r is p.pick
                        st.markdown(f"**{r.b['n']} – {r.n} rangées**  \n{nom_cornière_proposee(r)}, Lc {r.Lc} mm")
                        st.caption(f"e1 {r.e1} / p1 {r.p1} / e2 {r.e2} / gA {r.prop.gA} mm  \ntaux estimé "
                                   + pct(r.eta) + (" – retenue par défaut" if retenue else
                                                    ("" if r.ok else " – au-dessus du taux cible de " + pct(N(u.get("eta_c"))))))
                        st.button("Appliquer", key=f"asm_btn_sol_{p.rows.index(r)}", use_container_width=True,
                                  type="primary" if retenue else "secondary",
                                  on_click=_appliquer, args=(p.rows.index(r), ecrire))
    else:
        bloc_alerte(p.msg, True)
    st.markdown("#### Détail de la solution retenue par défaut")
    st.markdown(tableau_md(["", ""], [
        ["Boulon proposé", p.boulon], ["Rangées n1 (groupes S et P, 1 file)", p.n],
        ["e1 / p1 / e2 = e2,b", f"{p.e1} / {p.p1} / {p.e2} mm"],
        ["Longueur des cornières Lc = 2·e1 + (n1 − 1)·p1", f"{p.LC} mm (hauteur disponible " + F(p.hav, 0) + " mm)"],
        ["Épaisseur nécessaire au taux cible", F(p.treq, 2) + " mm"],
        ["Cornière proposée", f"L{p.leg}x{p.leg}x{p.tC} – trusquinage gA = {p.gA} mm"],
        ["Taux estimé (boulons et pressions diamétrales des âmes)", pct(p.eta, 1)], ["État", p.msg]]))
    st.markdown("#### Épaisseur nécessaire par vérification")
    st.markdown(tableau_md(["Vérification", "Épaisseur"], [[q[0], F(q[1], 2) + " mm"] for q in p.reqs], droite=(1,)))
    st.markdown("#### Toutes les combinaisons géométriquement possibles")
    h = st.columns([1, 0.6, 0.8, 1.6, 1, 1, 1, 1, 1, 1.3])
    for c, t in zip(h, ("Boulon", "n1", "Lc", "Cornière", "η boulons S", "η p. diam. âme S", "η boulons P",
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
                      use_container_width=True, on_click=_appliquer, args=(p.rows.index(r), ecrire),
                      help=None if r.ok else "taux cible dépassé")
    st.caption("Règle de choix par défaut : le plus petit nombre de rangées ; à égalité, le plus petit "
               "diamètre. MEd est ajouté à VEd·z ; les tractions dues à HEd et MEd ne sont prises en "
               "compte que dans la vérification détaillée.")


# ------------------------------------------------------------ onglet export
def onglet_export(R, u):
    complet = st.checkbox("toutes les vérifications dans le texte", key="asm_ui_txt_complet")
    t = texte.construire_texte(R, complet)
    c1, c2 = st.columns([1, 2])
    with c1:
        st.download_button("⬇️ Exporter le rapport .txt", data=("﻿" + t).encode("utf-8"),
                           file_name=texte.nom_fichier(u, ".txt"), mime="text/plain;charset=utf-8",
                           use_container_width=True, key="asm_btn_txt")
    with c2:
        st.caption("Le rapport complet PDF (note de calcul A4 paysage) : bouton « 📄 Générer PDF » de la "
                   "barre d'outils. Le texte ci-dessous se copie avec l'icône en haut à droite du cadre.")
    st.code(t, language=None)


# --------------------------------------------------------- onglet benchmark
def _charger_benchmark(i, ecrire):
    """Callback : les données de l'exemple publié i dans l'outil."""
    from .entrees import defaults
    d = defaults(); d.update(BENCH[i]["inp"]); ecrire(d)
    st.session_state["asm_ui_alerte"] = None
    st.session_state["asm_ui_onglet"] = ONGLETS[1]
    st.session_state["_asm_toast"] = BENCH[i]["id"] + " chargé : voir Vérifications"


def onglet_benchmark(B, ecrire):
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
            st.button("Charger ces données dans l'outil", key=f"asm_btn_bm_{i}", use_container_width=True,
                      on_click=_charger_benchmark, args=(i, ecrire))
        lignes = []
        for r in b["rows"]:
            lab = r["lab"] + ((" — " + r["exp"]) if (r["exp"] and abs(r["ec"]) > 0.003) else "")
            # espaces et tirets insécables : une valeur ne se coupe pas dans le tableau
            lignes.append([lab, r["page"].replace("-", "\u2011").replace(" ", "\u00a0"),
                           F(r["ref"], 1 if r["ref"] < 10 else 0) + "\u00a0" + r["unit"],
                           F(r["val"], 2 if r["ref"] < 10 else 1), F(r["ec"] * 100, 2) + "\u00a0%", r["st"]])
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
            lignes.append([r["lab"], F(r["hand"], n) + ("" if r["unit"] == "-" else "\u00a0" + r["unit"]),
                           F(r["xls"], n), F(r["val"], n), F(r["ec"] * 100, 3) + "\u00a0%", r["st"]])
        st.markdown(tableau_md(["Grandeur", "Manuel", "Excel", "Outil", "Écart", ""], lignes, droite=(1, 2, 3, 4)))


def onglet_methode():
    st.markdown(METHODE)


# ------------------------------------------------------------------- render
def render(R, B, u, ecrire):
    """La colonne de droite entière. ``ecrire(u)`` écrit un jeu complet de
    saisies dans la source unique (application d'une solution, benchmark)."""
    st.session_state["_asm_R"] = R
    bandeau(R, B)
    alertes(R, u)
    seg = getattr(st, "segmented_control", None)
    if seg:
        seg("Onglet", list(ONGLETS), key="asm_ui_onglet", label_visibility="collapsed")
    else:
        st.radio("Onglet", list(ONGLETS), key="asm_ui_onglet", horizontal=True, label_visibility="collapsed")
    onglet = st.session_state.get("asm_ui_onglet") or ONGLETS[0]
    if onglet == ONGLETS[0]:
        onglet_geometrie(R, u)
    elif onglet == ONGLETS[1]:
        onglet_verifications(R)
    elif onglet == ONGLETS[2]:
        onglet_predim(R, u, ecrire)
    elif onglet == ONGLETS[3]:
        onglet_export(R, u)
    elif onglet == ONGLETS[4]:
        onglet_benchmark(B, ecrire)
    else:
        onglet_methode()

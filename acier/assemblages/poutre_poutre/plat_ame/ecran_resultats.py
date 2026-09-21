# -*- coding: utf-8 -*-
"""Résultats à l'écran du plat d'âme : bandeau d'une ligne, alertes courtes,
dessins cliquables (colonne figée), onglets Vérifications · Prédim · Note ·
Benchmark · Méthode. Tout est lu dans ``R`` (moteur) et ``B`` (benchmark)."""
import streamlit as st

from acier.js import N, js_str
from acier.formats import F, pct
from acier.assemblages.ui_commun import (bloc_statut_ligne, bloc_alerte_ligne, bloc_alerte, badge_nature,
                                         tableau_md, tableau_html, taux_html, Html)
from acier.bibliotheques import PERSO
from . import schemas, texte, synthese, formules
from .notation import ligne_alerte, ref_courte, LEGENDE_REFERENCES
from .entrees import COURT, UNITE, CHAMPS, CLES, champ_visible, PILOTEES_PAR_PREDIM
from .moteur import apply_solution, solution_label
from .benchmark import SRC, NOCOVER
from .methode import METHODE
from .ecran_saisie import K, valeur_widget, PAS

ONGLETS = ("Vérifications", "Prédim", "Note", "Benchmark", "Méthode")
NIVEAUX_ECRAN = ("Vue simple", "Édition", "Cotations complètes")


# ------------------------------------------------------------------ bandeau
def bandeau(R):
    ok = R.verified
    titre = R.statut + (" – sous réserve des alertes" if ok and (R.reserve or R.alerts) else "")
    comp = "— " + pct(R.eta_max, 1) + (" · " + R.gov.label if R.gov else "")
    if not R.geo_ok:
        comp += " · géométrie non valide"
    if not R.dist_ok:
        comp += " · pinces ou entraxes non conformes"
    bloc_statut_ligne("ok" if ok else "nok", titre, comp,
                      [(nom, eta, okg) for nom, eta, okg, _ in synthese.taux_par_element(R)])
    if R.pred:
        bloc_alerte("Mode prédimensionnement : " + R.boulon + ", " + js_str(R.n_1) + " rangées, plat "
                    + js_str(R.h_p) + " × " + js_str(R.t_p) + ". " + R.pd.msg, False)


# ------------------------------------------------------------------ alertes
def _selectionner(ident, a_geo):
    cur = st.session_state.get("fpl_ui_alerte")
    st.session_state["fpl_ui_alerte"] = None if cur == ident else ident


def _ecrire_depuis(cle_widget, k):
    st.session_state[K(k)] = valeur_widget(k, st.session_state.get(cle_widget))


def _mini_champ(k, u, prefixe, disabled=False):
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
    sel = st.session_state.get("fpl_ui_alerte")
    ids = [a.id for a in R.alerts]
    if sel and sel not in ids:
        st.session_state["fpl_ui_alerte"] = None
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
            st.button("Masquer" if est_sel else "Localiser", key=f"fpl_btn_al_{a.id}",
                      use_container_width=True, on_click=_selectionner,
                      args=(a.id, bool(a.dims or a.elems)))
        if est_sel:
            with st.container(border=True):
                st.markdown(a.why if a.why else a.msg)
                if R.pred and any(k in PILOTEES_PAR_PREDIM for k in a.fields):
                    st.caption("Mode prédimensionnement : les champs marqués ◆ sont pilotés par la "
                               "proposition (onglet Prédim, ou mode VÉRIFICATION pour les modifier).")
                visibles = [k for k in a.fields if champ_visible(u, k)]
                if visibles:
                    st.caption("Champs en cause — modifiables ici :")
                    cols = st.columns(min(4, len(visibles)))
                    for i, k in enumerate(visibles):
                        with cols[i % len(cols)]:
                            _mini_champ(k, u, "fpl_fix_", disabled=R.pred and k in PILOTEES_PAR_PREDIM)
                if a.dims or a.elems:
                    st.caption("Sur le dessin : cotes et éléments en cause en rouge.")


# ------------------------------------------------------------------ dessins
def _traiter_clic(retour, cle_cmp):
    if not retour or not isinstance(retour, dict):
        return
    derniers = st.session_state.setdefault("fpl_ui_dernier_clic", {})
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
            ecrits.append(COURT.get(k, k) + " = " + js_str(valeur_widget(k, v)))
    if ecrits:
        st.toast(" ; ".join(ecrits) + " : géométrie et vérifications recalculées")
        st.rerun()


def _panneaux(R, u):
    """Les fenêtres d'objet du composant : un clic sur une pièce ouvre TOUS
    ses paramètres."""
    pred = R.pred

    def ch(k, lab):
        f = CHAMPS[k]
        return dict(key=k, lab=lab, type=f["t"], options=f.get("o") or [], step=PAS.get(k, 1.0),
                    dis=bool(pred and k in PILOTEES_PAR_PREDIM))

    def perso(X):
        return [ch(f"{d}{X}_u", d + " (mm)") for d in ("h", "b", "tw", "tf", "r")]

    g = {}
    g["G"] = dict(titre=schemas.GROUPE_BOULONS["titre"],
                  champs=[ch("n1_u", "Rangées n1"), ch("n2_u", "Files n2"), ch("p1_u", "Entraxe p1 (mm)")]
                  + ([ch("p2_u", "Entraxe p2 (mm)")] if R.n_2 > 1 else [])
                  + [ch("e1_u", "Pince e1 (mm)"), ch("e2b_u", "Pince e2,b (mm)")])
    g["bolts"] = dict(titre="Boulons",
                      champs=[ch("boulon_u", "Diamètre"), ch("classe", "Classe"), ch("trou", "Type de trou"),
                              ch("cat", "Catégorie"), ch("filet", "Filetage cisaillé")]
                      + ([ch("mu_s", "Frottement μ"), ch("k_s", "ks")] if u.get("cat") != "A" else [])
                      + ([ch("k_ser", "ELS / ELU")] if u.get("cat") == "B" else []))
    g["beamS"] = dict(titre="Poutre secondaire (portée)",
                      champs=[ch("prof_S", "Profilé"), ch("nu_S", "Nuance")]
                      + (perso("S") if u.get("prof_S") == PERSO else [])
                      + [ch("lt_ok", "Maintenue au déversement"), ch("r_n", "Rayon du grugeage (mm)")],
                      note="Grugeage, jeu et décalage : cotes dc,sup, dc,inf, c, gh et Δz sur le dessin — "
                           "la poutre se déplace aussi à la souris.")
    g["beamP"] = dict(titre="Poutre principale (porteuse)",
                      champs=[ch("prof_P", "Profilé"), ch("nu_P", "Nuance")]
                      + (perso("P") if u.get("prof_P") == PERSO else []))
    g["plate"] = dict(titre="Plat d'âme",
                      champs=[ch("hp_u", "Hauteur hp (mm)"), ch("tp_u", "Épaisseur tp (mm)"),
                              ch("bp_u", "Largeur bp (mm)"), ch("z_C", "Position zp (mm)"),
                              ch("nu_pl", "Nuance"), ch("a_w", "Gorge a (mm)")],
                      note="hp, bp, zp et la gorge a : aussi des cotes sur le dessin.")
    g["weld"] = dict(titre="Soudure (double cordon)", champs=[ch("a_w", "Gorge a (mm)")])
    g["efforts"] = dict(titre="Efforts de calcul ELU",
                        champs=[ch("V_Ed", "VEd (kN)"), ch("N_Ed", "NEd (kN)"), ch("M_Ed", "MEd (kNm)")])
    return g


def _alerte_courante(R):
    sel = st.session_state.get("fpl_ui_alerte")
    for a in R.alerts:
        if a.id == sel:
            return dict(dims=set(a.dims), elems=set(a.elems))
    return None


def dessins(R, u):
    """La colonne du dessin : élévation puis vue en plan, empilées (colonne
    figée par interface.py). Renvoie ``(e, p, hl)`` pour le repli."""
    interactif = bool(st.session_state.get("fpl_ui_composant", True))
    niveau = st.radio("Niveau de cotation", [0, 1, 2], format_func=lambda i: NIVEAUX_ECRAN[i],
                      horizontal=True, key="fpl_ui_niveau", label_visibility="collapsed")
    hl = _alerte_courante(R)
    opt = schemas.options_ecran(R, niveau, hl)
    opt.editables = niveau >= 1
    e = schemas.elevation(R, opt); p = schemas.plan(R, opt)
    valeurs = {k: st.session_state.get(K(k)) for k in CLES}
    groupes = _panneaux(R, u)
    for d, cle in ((e, "fpl_cmp_elev"), (p, "fpl_cmp_plan")):
        if interactif:
            from acier.composants.svg_cliquable import svg_cliquable
            _traiter_clic(svg_cliquable(d.svg(identifiant=cle), valeurs, key=cle, groupes=groupes), cle)
        else:
            st.markdown(d.svg(identifiant=cle), unsafe_allow_html=True)
    if R.pred:
        st.caption("◆ pilotés par la proposition : onglet Prédim pour appliquer une solution.")
    elif interactif:
        st.caption("Touche une cote pour la modifier · touche une pièce (poutre, plat, boulon, cordon, VEd) "
                   "pour tous ses paramètres · glisse la poutre portée pour régler le jeu gh et le décalage Δz.")
    return e, p, hl


def panneau_cotes(e, p, hl):
    """Repli (tête de la colonne de droite) : une saisie par cote."""
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
                    cle = "fpl_cote_" + k
                    st.session_state[cle] = valeur_widget(k, st.session_state.get(K(k)))
                    st.number_input(("🔴 " if ident in chauds else "") + f"{COURT.get(k, sym)} (mm)", key=cle,
                                    step=PAS.get(k, 1.0), format="%g",
                                    on_change=_ecrire_depuis, args=(cle, k), help=CHAMPS[k]["l"])


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


def _table_boulons_efforts(t, lab):
    with st.expander("Répartition élastique par boulon – " + lab):
        st.markdown(tableau_md(["Rangée / file", "x", "y", "Fx", "Fz", "F [kN]"],
                               [[f"{b.i} / {b.j}", F(b.x, 1), F(b.y, 1), F(b.fx, 2), F(b.fz, 2), F(b.f, 2)] for b in t],
                               droite=(1, 2, 3, 4, 5)))


def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _formules(R):
    with st.expander("Formules et substitutions numériques"):
        st.caption("Les valeurs introduites sont composées par le moteur ; la ligne du taux est vérifiée "
                   "par un test qui la recalcule.")
        for grp in ("Boulons", "Plat d'âme", "Poutre secondaire", "Poutre principale", "Soudure"):
            on = [c for c in R.checks if c.grp == grp and c.active]
            if not on:
                continue
            st.markdown(f"**{grp}**")
            for c in on:
                corps = "<br>".join(_esc(l) for l in formules.textes(R, c))
                st.markdown(
                    f"<div style='margin:2px 0 8px 0;padding:4px 10px;border-left:3px solid "
                    f"{'#2E7D46' if c.ok else '#9C3341'};font-size:0.88em;'>"
                    f"<b>{_esc(synthese.court(c))}</b>{badge_nature(c.nat)} "
                    f"<span style='color:#6E7480;'>{_esc(c.ref)}</span><br>"
                    f"<span style='font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:0.95em;'>{corps}</span></div>",
                    unsafe_allow_html=True)


def _tableau_33(R):
    st.markdown("##### Pinces et entraxes – EN 1993-1-8 Tableau 3.3")
    if not R.dist:
        st.caption("Sans objet.")
        return
    lignes = []
    for x in R.dist:
        lignes.append([x.lab, F(x.val, 1), F(x.min, 1), "—" if x.max is None else F(x.max, 0),
                       Html("<span style='color:#2E7D46;font-weight:700'>OK</span>" if x.ok
                            else "<span style='color:#9C3341;font-weight:700'>NON OK</span>")])
    st.markdown(tableau_html(["Distance", "Valeur", "Min", "Max", ""], lignes, droite=(1, 2, 3),
                             largeurs=[None, "10%", "10%", "10%", "10%"]), unsafe_allow_html=True)
    nok = [x for x in R.dist if not x.ok]
    if nok:
        cols = st.columns(min(4, len(nok)))
        for i, x in enumerate(nok):
            with cols[i % len(cols)]:
                st.button("Localiser : " + x.lab.split(" (")[0], key=f"fpl_btn_dist_{x.id}", use_container_width=True,
                          on_click=_selectionner, args=(x.alert, True))
    st.caption("e ≥ 1,2·d0 ; p1 ≥ 2,2·d0 ; p2 ≥ 2,4·d0 ; p ≤ min(14t ; 200 mm) ; e ≤ 4t + 40 mm si exposé.")


def _parametres_retenus(R):
    g = [["Boulon retenu", R.boulon + " – d0 = " + F(R.d_0, 0) + " mm"],
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
    with st.expander("Paramètres retenus et excentricités"):
        st.markdown(tableau_md(["Paramètre", "Valeur"], g))


def onglet_verifications(R):
    st.markdown(badge_nature("EC") + " formule directe de l'Eurocode " + badge_nature("COMP")
                + " modèle complémentaire reconnu (MSB Part 5 §3, SCI P358, ECCS n°126, BS 5950-1) "
                + badge_nature("INT") + " interprétation de l'outil · " + LEGENDE_REFERENCES, unsafe_allow_html=True)
    for cle in ("boulons", "plat", "portee", "porteuse", "soudure"):
        _table_simple(R, cle)
    _table_boulons_efforts(R.tab, "groupe")
    st.caption("Non revérifiés ici : flexion HORS PLAN de l'âme porteuse sous VEd·z et NEd (attache d'un "
               "seul côté — négligée par les guides pour les détails courants) ; cisaillement global, flexion "
               "et déversement de la poutre principale ; torsion induite ; efforts d'arrachement accidentels "
               "(tying) ; deux poutres en vis-à-vis sur la même âme.")
    off = [c for c in R.checks if not c.active]
    if off:
        st.caption("Sans objet dans cette configuration : " + " ; ".join(synthese.court(c) for c in off) + ".")
    _formules(R)
    _tableau_33(R)
    _parametres_retenus(R)


# ------------------------------------------------------------ onglet prédim
def _appliquer(i, ecrire):
    R = st.session_state.get("_fpl_R")
    r = R.pd.rows[i]
    if not r.prop:
        return
    ecrire(apply_solution({k: st.session_state.get(K(k)) for k in CHAMPS}, r))
    st.session_state["fpl_ui_alerte"] = None
    st.session_state["fpl_ui_onglet"] = ONGLETS[0]
    st.session_state["_fpl_toast"] = "Solution appliquée aux données et à la géométrie : " + solution_label(r)


def onglet_predim(R, u, ecrire):
    p = R.pd
    rows = [r for r in p.rows if r.geom]
    sols = sorted([r for r in rows if r.eta <= 1],
                  key=lambda r: (0 if r.ok else 1, r.score if r.ok else r.eta))[:9]
    st.caption(("Mode PRÉDIMENSIONNEMENT actif : la solution retenue par défaut pilote le calcul ("
                + R.statut + ", taux maximal " + pct(R.eta_max, 1) + ")." if R.pred else "Mode VÉRIFICATION actif.")
               + " Touche une solution pour la recopier dans les données et la géométrie. "
               "Seule la vérification détaillée fait foi.")
    st.markdown("#### Solutions proposées")
    if sols:
        for i in range(0, len(sols), 3):
            cols = st.columns(3)
            for col, r in zip(cols, sols[i:i + 3]):
                with col:
                    with st.container(border=True):
                        retenue = r is p.pick
                        st.markdown(f"**{r.b['n']} – {r.n} rangées** ({r.var})  \nplat {r.hp} × "
                                    + js_str(r.prop.tp if r.prop else 0) + " mm")
                        st.caption(f"e1 {r.e1} / p1 {r.p1} / e2 {r.e2} mm  \ntaux estimé " + pct(r.eta)
                                   + (" – retenue par défaut" if retenue else
                                      ("" if r.ok else " – au-dessus du taux cible de " + pct(N(u.get("eta_c"))))))
                        st.button("Appliquer", key=f"fpl_btn_sol_{p.rows.index(r)}", use_container_width=True,
                                  type="primary" if retenue else "secondary",
                                  on_click=_appliquer, args=(p.rows.index(r), ecrire))
    else:
        bloc_alerte(p.msg, True)
    st.markdown("#### Détail de la solution retenue par défaut")
    st.markdown(tableau_md(["", ""], [
        ["Boulon proposé", p.boulon], ["Rangées n1 (1 file)", js_str(p.n)],
        ["e1 / p1 / e2 = e2,b", f"{p.e1} / {p.p1} / {p.e2} mm"],
        ["Hauteur du plat hp = 2·e1 + (n1 − 1)·p1", f"{p.hp} mm (hauteur disponible " + F(p.hav, 0) + " mm)"],
        ["Épaisseur nécessaire au taux cible", F(p.treq, 2) + " mm"],
        ["Plat proposé", js_str(p.hp) + " × " + js_str(p.bp) + " × " + js_str(p.tp) + " mm – gorge a = " + js_str(p.a) + " mm"],
        ["Taux estimé (boulons et pression diamétrale de l'âme)", pct(p.eta, 1)], ["État", p.msg]]))
    st.markdown("#### Épaisseur nécessaire par vérification")
    st.markdown(tableau_md(["Vérification", "Épaisseur"], [[q[0], F(q[1], 2) + " mm"] for q in p.reqs], droite=(1,)))
    st.markdown("#### Toutes les combinaisons géométriquement possibles")
    h = st.columns([1, 0.6, 0.9, 0.8, 1, 1, 1, 1.3])
    for c, t in zip(h, ("Boulon", "n1", "variante", "hp", "η boulons", "η p. diam. âme", "η max", "")):
        c.markdown(f"**{t}**" if t else "")
    for r in rows:
        c = st.columns([1, 0.6, 0.9, 0.8, 1, 1, 1, 1.3], vertical_alignment="center")
        fort = "**" if r is p.pick else ""
        vals = (r.b["n"], js_str(r.n), r.var, js_str(r.hp), pct(r.e.bS), pct(r.e.pdS), pct(r.eta))
        for cc, v in zip(c[:7], vals):
            cc.markdown(f"{fort}{v}{fort}")
        with c[7]:
            st.button("Appliquer" + ("" if r.ok else " ⚠"), key=f"fpl_btn_row_{p.rows.index(r)}",
                      use_container_width=True, on_click=_appliquer, args=(p.rows.index(r), ecrire),
                      help=None if r.ok else "taux cible dépassé")
    st.caption("Règle de choix par défaut : le plus petit nombre de rangées ; à égalité, le plus petit "
               "diamètre. L'épaisseur du plat est plafonnée à 0,5·d (ductilité) ; la gorge proposée est la "
               "pleine résistance. Seule la vérification détaillée fait foi.")


# -------------------------------------------------------------- onglet note
def onglet_note(R, u):
    c1, c2 = st.columns([1.6, 2.6], vertical_alignment="center")
    complet = c1.checkbox("Texte : toutes les vérifications", key="fpl_ui_txt_complet")
    t = texte.construire_texte(R, complet)
    with c1:
        st.download_button("⬇️ Exporter le texte (.txt)", data=("﻿" + t).encode("utf-8"),
                           file_name=texte.nom_fichier(u, ".txt"), mime="text/plain;charset=utf-8",
                           use_container_width=True, key="fpl_btn_txt")
    c2.caption("Note de calcul (2 pages, A4 paysage — synthèse + plan de principe) : bouton "
               "« 📄 Générer PDF » de la barre d'outils. Le texte ci-dessous se copie avec l'icône "
               "en haut à droite du cadre.")
    st.code(t, language=None)


# --------------------------------------------------------- onglet benchmark
def _charger_benchmark(inp, ecrire):
    from .entrees import defaults
    d = defaults(); d.update(inp); ecrire(d)
    st.session_state["fpl_ui_alerte"] = None
    st.session_state["fpl_ui_onglet"] = ONGLETS[0]
    st.session_state["_fpl_toast"] = "Exemple chargé : voir Vérifications"


def onglet_benchmark(B, ecrire):
    from acier.assemblages.ui_commun import bloc_statut
    bloc_statut("Benchmark du module : " + B["statut"],
                ["Ces tableaux sont recalculés par le moteur à chaque ouverture ; ils ne dépendent pas de tes données."],
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
            st.button("Charger ces données dans l'outil", key=f"fpl_btn_bm_{i}", use_container_width=True,
                      on_click=_charger_benchmark, args=(BENCH[i]["inp"], ecrire))
        lignes = []
        for r in b["rows"]:
            lab = r["lab"] + ((" — " + r["exp"]) if (r["exp"] and abs(r["ec"]) > 0.003) else "")
            lignes.append([lab, r["page"].replace("-", "‑").replace(" ", " "),
                           F(r["ref"], 1 if r["ref"] < 10 else 0) + " " + r["unit"],
                           F(r["val"], 2 if r["ref"] < 10 else 1), F(r["ec"] * 100, 2) + " %", r["st"]])
        st.markdown(tableau_md(["Vérification", "Page", "Référence", "Outil", "Écart", "Statut"], lignes, droite=(2, 3, 4)))
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
    st.session_state["_fpl_R"] = R
    seg = getattr(st, "segmented_control", None)
    if seg:
        seg("Onglet", list(ONGLETS), key="fpl_ui_onglet", label_visibility="collapsed")
    else:
        st.radio("Onglet", list(ONGLETS), key="fpl_ui_onglet", horizontal=True, label_visibility="collapsed")
    onglet = st.session_state.get("fpl_ui_onglet") or ONGLETS[0]
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

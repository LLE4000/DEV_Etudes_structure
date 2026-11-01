# -*- coding: utf-8 -*-
"""Résultats à l'écran du pied de poteau : bandeau d'une ligne, alertes
courtes, dessins cliquables (colonne figée : coupe A-A puis vue en plan avec
la répartition), onglets Vérifications · Répartition · Note · Benchmark ·
Méthode. Tout est lu dans ``R`` (moteur) et ``B`` (benchmark).

Charte du 31/10/2026 : les fenêtres d'objet, le panneau Cotes et les chips
d'alerte lisent ``LIBELLE`` (et ``LIBELLE_MODE``), ``AIDES`` et
``AFFICHAGE`` de la carte (une forme par clé) ; l'onglet Vérifications range
la légende, le périmètre et le Tableau 3.3 dans des replis (rien ne se perd,
tout est dans ``md(at)``). La tête de la colonne dessin et le panneau Cotes
servent aussi le calcul manuel (manuel/ecran.py)."""
import streamlit as st

from acier.js import N, js_str
from acier.formats import F, pct
from acier.assemblages.ui_commun import (bloc_statut_ligne, bloc_alerte_ligne, badge_nature,
                                         tableau_md, tableau_html, taux_html, verdict_html, Html, _esc, C_TRAIT,
                                         legende_natures, expander_perimetre, tableau_33_replie)
from . import schemas, texte, synthese, formules
from .notation import ligne_alerte, LEGENDE_REFERENCES
from .entrees import CHAMPS, CLES, champ_visible, ancrage_txt
from .moteur import GROUPES_VERIF
from .benchmark import SRC, NOCOVER
from .methode import METHODE
from .ecran_saisie import (K, valeur_widget, PAS, LIBELLE, AIDES, AFFICHAGE, COTES_ORDRE, libelle, description)
from .manuel.schemas import CLE_PAR_COTE as CLE_PAR_COTE_MANUEL

ONGLETS = ("Vérifications", "Répartition", "Note", "Benchmark", "Méthode")
NIVEAUX_ECRAN = ("Vue simple", "Édition", "Cotations complètes")
CLES_TETE_DESSIN = ("ui_niveau", "ui_composant")      # réglages d'écran en tête de la colonne dessin
NATURES = ("EC", "EC + INT", "INT")
TEXTES_NATURES = {"EC + INT": "Eurocode et interprétation nommée",
                  "INT": "modèle d'ingénieur (niveau 2, tiges de coin ou latérales)"}
NON_REVERIFIES = ("fendage et éclatement latéral du massif (EC2-4 §7.2.1.7-8, signalés en alerte)",
                  "tiges post-installées, raidisseurs, bêche", "flambement et classe de section du poteau",
                  "poinçonnement du massif au-delà de l'EN 1992-1-1 §6.7")
CAPTION_LEGENDE = "Vue en plan : aire efficace (bleu), zone comprimée (rouge), axe neutre, C et T des tiges."
CAPTION_USAGE = CAPTION_LEGENDE + " Touche une cote ou une pièce (poteau, platine, tiges, massif, mortier, efforts)."
# Cote du dessin (identifiant de la cote) ← clé : pour mettre en rouge une saisie du panneau Cotes
_COTE_DE = {v: k for k, v in list(schemas.CLE_PAR_COTE.items()) + list(CLE_PAR_COTE_MANUEL.items())}


# ------------------------------------------------------------------ bandeau
def bandeau(R):
    ok = R.verified
    titre = R.statut + (" – sous réserve des alertes" if ok and (R.reserve or R.alerts) else "")
    # la dimensionnante nommée comme dans la note (synthese.court), plus le libellé du moteur (88 caractères)
    comp = "— " + pct(R.eta_max, 1) + (" · " + synthese.court(R.gov) if R.gov else "")
    if not R.geo_ok:
        comp += " · géométrie non valide"
    if not R.dist_ok:
        comp += " · pinces non conformes"
    bloc_statut_ligne("ok" if ok else "nok", titre, comp,
                      [(nom, eta, okg) for nom, eta, okg, _ in synthese.taux_par_element(R)])


def information(R):
    """La ligne d'information sous le bandeau : la vérification d'information
    la plus sollicitée, hors verdict."""
    pire = synthese.pire_information(R)
    if pire is not None:
        st.caption("Béton d'ancrage (EN 1992-4, en information, hors verdict) : " + synthese.court(pire) + " à "
                   + pct(pire.eta, 1) + (" — non satisfait" if not pire.ok else " — satisfait")
                   + ". Périmètre « EC3 + EC2-4 » en tête de la carte pour le compter dans le verdict.")


# ------------------------------------------------- la platine, à la main
def _lig(lab, val, fort=False):
    return [Html(f"<b>{_esc(lab)}</b>" if fort else _esc(lab)), Html(f"<b>{_esc(val)}</b>" if fort else _esc(val))]


def bloc_platine(R):
    """L'épaisseur de la platine comme on la calcule à la main : de chaque
    côté le moment pris, sa résistance, l'épaisseur qu'il demande ; en tête
    l'épaisseur requise, le plus grand des quatre critères."""
    P = R.plaque
    u = R.u
    c = R.ck["tpq"]
    st.markdown("##### Épaisseur de la platine — le calcul à la main")
    if not c.active:
        st.caption("Géométrie non valide : pas d'épaisseur requise.")
        return
    crit = {"c": "console comprimée", "N": "appui N ≤ fjd·Aeff", "M": "flexion composée (niveau 1)", "T": "tronçon en T"}
    pire = max(("c", "N", "M", "T"), key=lambda k: P["t_" + k])
    req = "∞ (aucune épaisseur ne suffit : agrandir la platine ou les tiges)" if c.Ed == float("inf") else F(c.Ed, 1) + " mm"
    bloc_statut_ligne("ok" if c.ok else "nok",
                      "Épaisseur requise " + req + (" ≤ " if c.ok else " > ") + "tp = " + F(u.tp_u, 0) + " mm",
                      "— " + pct(c.eta, 1) + " · critère : " + crit[pire] + " · fy = " + F(R.fy_p, 0) + " MPa (épaisseur retenue)",
                      [(crit[k], P["t_" + k] / u.tp_u if u.tp_u else 0.0, P["t_" + k] <= u.tp_u + 1e-9)
                       for k in ("c", "N", "M", "T")])
    g1, g2 = st.columns(2, gap="medium")
    with g1:
        st.markdown("**Côté comprimé — console au nu de l'aile**")
        rows = [_lig("Pression sous la platine (répartition élastique)",
                     "p max = " + F(P["p_max"], 1) + " MPa ≤ fjd = " + F(R.fjd, 1) + " MPa"),
                _lig("Console mobilisée", "bande c = tp·√(fy/(3·fjd·γM0)) = " + F(P["c"], 0) + " mm (EC3 §6.2.5) — débord "
                     "géométrique " + F(P["L_geo"], 0) + " mm"),
                _lig("Moment pris (" + P["ligne"] + ")", "mEd = ∫p·s ds sur c = " + F(P["m_Ed"], 1) + " kNm/m", True),
                _lig("Résistance élastique", "mRd = tp²·fy/(6·γM0) = " + F(P["m_Rd_el"], 1) + " kNm/m → "
                     + pct(R.ck["pc"].eta, 0) if R.ck["pc"].active else "—"),
                _lig("Épaisseur pour ce moment", "tc = √(6·mEd·γM0/fy) = " + F(P["t_c"], 1) + " mm"),
                _lig("Appui : N ≤ fjd·Aeff(c)", "tN = " + F(P["t_N"], 1) + " mm" if P["t_N"] != float("inf") else "tN = ∞"),
                _lig("Flexion composée : niveau 1 ≤ 1", "tM = " + F(P["t_M"], 1) + " mm" if P["t_M"] != float("inf") else "tM = ∞")]
        st.markdown(tableau_html(["", ""], rows, largeurs=["42%", None]), unsafe_allow_html=True)
    with g2:
        st.markdown("**Côté tendu — tronçon en T de la rangée la plus sollicitée**")
        T = P["tendu"]
        if T and T["T"] > 1e-9:
            rows = [_lig("Rangée tendue", T["cle"] + " (" + T["classe"] + ") : T = " + F(T["T"], 1) + " kN"),
                    _lig("Bras au nu de l'aile (cordon déduit)", "m = e − 0,8·a·√2 = " + F(T["m"], 1) + " mm"),
                    _lig("Moment pris", "M = T·m = " + F(T["T"], 1) + " × " + F(T["m"], 1) + " = " + F(T["M"], 2) + " kNm", True),
                    _lig("Longueur efficace", "Σleff,1 = " + F(T["leff1"], 0) + " mm"
                         + (" (plafond de coin 0,25·b par tige)" if T["classe"] == "coin" else "")),
                    _lig("Résistance plastique", "Mpl,1 = 0,25·Σleff,1·tp²·fy/γM0 = " + F(T["Mpl1"], 2) + " kNm"),
                    _lig("Effet de levier", ("Lb = " + F(T["Lb"], 0) + (" ≤ " if T["levier"] else " > ") + "Lb* = "
                         + F(min(T["Lb_star"], 9999), 0) + " mm : " + ("avec" if T["levier"] else "sans")
                         + " → FT,Rd = " + ("4" if T["levier"] else "2") + "·Mpl,1/m… = " + F(T["FT_Rd"], 0)
                         + " kN (mode " + str(T["mode"]) + ")")),
                    _lig("Épaisseur pour ce moment", "tp ≥ √(" + ("1" if T["levier"] else "2") + "·T·m·γM0/(Σleff,1·fy)) = "
                         + F(T["t_formule"], 1) + " mm ; tous modes : tT = " + F(P["t_T"], 1) + " mm")]
        else:
            rows = [_lig("Rangée tendue", "aucune tige tendue : la platine ne fléchit pas en traction"),
                    _lig("Épaisseur pour ce côté", "tT = " + F(P["t_T"], 1) + " mm (minimum du balayage)")]
        st.markdown(tableau_html(["", ""], rows, largeurs=["42%", None]), unsafe_allow_html=True)
    NV = P["navier"]
    base = ("Comparaison manuelle — Navier sur TOUTE la platine (repère, pas un critère) : σ = N/A ± 6·Mx/(bp·lp²) ± "
            "6·My/(lp·bp²) = " + F(NV["sN"], 2) + " ± " + F(NV["sMx"], 2) + " ± " + F(NV["sMy"], 2) + " MPa")
    if NV["applicable"]:
        st.caption(base + " → σ max = " + F(NV["s_max"], 2) + " MPa, débord L = " + F(NV["L"], 0) + " mm, m = σ max·L²/2 = "
                   + F(NV["m"], 1) + " kNm/m → t = √(6·m·γM0/fy) = " + F(NV["t"], 1) + " mm (pression uniforme sur "
                   "toute la platine : plus de bras, moins d'appui que l'aire efficace).")
    else:
        st.caption(base + " → σ min = " + F(NV["s_min"], 2) + " MPa " + ("< 0 : soulèvement, Navier ne s'applique pas"
                   if u.N_Ed > 0 else ": pas de compression") + " — la répartition sans traction ci-dessus fait foi.")


# ------------------------------------------------------------------ alertes
def _selectionner(ident, a_geo):
    cur = st.session_state.get("ppb_ui_alerte")
    st.session_state["ppb_ui_alerte"] = None if cur == ident else ident


def _ecrire_depuis(cle_widget, k):
    st.session_state[K(k)] = valeur_widget(k, st.session_state.get(cle_widget))


def _mini_champ(k, u, prefixe, disabled=False):
    """La chip d'une alerte : le même champ, modifiable sur place (libellé
    LIBELLE du mode, options AFFICHAGE) ; écrit dans la source unique."""
    f = CHAMPS[k]
    cle = prefixe + k
    st.session_state[cle] = valeur_widget(k, st.session_state.get(K(k)))
    lab = libelle(k, u)
    if f["t"] == "s":
        aff = AFFICHAGE.get(k)
        st.selectbox(lab, f["o"], key=cle, on_change=_ecrire_depuis, args=(cle, k), disabled=disabled,
                     format_func=(lambda o, a=aff: a.get(o, o)) if aff else str)
    elif f["t"] == "x":
        st.text_input(lab, key=cle, on_change=_ecrire_depuis, args=(cle, k), disabled=disabled)
    else:
        st.number_input(lab, key=cle, step=PAS.get(k, 1.0), format="%g",
                        on_change=_ecrire_depuis, args=(cle, k), disabled=disabled)


def _alerte(a, R, u, sel):
    """Une alerte : sa ligne courte et son bouton Localiser sur UNE rangée ;
    sélectionnée, l'explication et les champs en cause, modifiables sur place."""
    est_sel = sel == a.id
    titre, detail = ligne_alerte(a, R)
    # L'alerte et son bouton sur une rangée à toutes les largeurs : en colonnes, le bouton passait sous
    # l'alerte au téléphone (une rangée de plus par alerte).
    with st.container(horizontal=True, wrap=False, gap="small", vertical_alignment="center", key=f"ppb_al_{a.id}"):
        bloc_alerte_ligne(titre, detail, a.block, est_sel)
        st.button("Masquer" if est_sel else "Localiser", key=f"ppb_btn_al_{a.id}",
                  width="content", on_click=_selectionner, args=(a.id, bool(a.dims or a.elems)))
    if est_sel:
        with st.container(border=True):
            st.markdown(a.why if a.why else a.msg)
            visibles = [k for k in a.fields if champ_visible(u, k)]
            if visibles:
                st.caption("Champs en cause — modifiables ici :")
                cols = st.columns(min(4, len(visibles)))
                for i, k in enumerate(visibles):
                    with cols[i % len(cols)]:
                        _mini_champ(k, u, "ppb_fix_", disabled=bool(st.session_state.get("prj_verrou")))
            if a.dims or a.elems:
                st.caption("Sur le dessin : cotes et éléments en cause en rouge.")


def titre_remarques(remarques, R):
    """« ⚠️ 2 remarques — Tiges de coin · Soulèvement » : le titre du repli des
    alertes d'information (le titre court de chacune, avant « : », « — » ou
    « ( » ; le titre entier est dans le repli)."""
    import re
    noms = [re.split(r" : | — | \(", ligne_alerte(a, R)[0])[0] for a in remarques]
    return "⚠️ " + str(len(remarques)) + " remarques — " + " · ".join(noms)


def alertes(R, u):
    """Les alertes : une ligne par alerte BLOQUANTE ; les alertes d'information
    (attention, modèle d'ingénieur, soulèvement, rigidité…) regroupées dans un
    repli « ⚠️ n remarques — … » dès qu'il y en a deux, ouvert quand l'une
    d'elles est localisée. Rien ne se perd (même texte, mêmes boutons) ; le
    premier écran du téléphone garde le bandeau ET le dessin (cinq alertes
    d'information au cas par défaut de la vérification complète : 425 px)."""
    sel = st.session_state.get("ppb_ui_alerte")
    ids = [a.id for a in R.alerts]
    if sel and sel not in ids:
        st.session_state["ppb_ui_alerte"] = None
        sel = None
        st.toast("Blocage levé : géométrie valide" if (R.geo_ok and R.dist_ok)
                 else "Ce blocage est levé ; il en reste d'autres")
    remarques = [a for a in R.alerts if not a.block]
    for a in R.alerts:
        if a.block or len(remarques) == 1:
            _alerte(a, R, u, sel)
    if len(remarques) > 1:
        with st.expander(titre_remarques(remarques, R), expanded=any(a.id == sel for a in remarques)):
            for a in remarques:
                _alerte(a, R, u, sel)


# ------------------------------------------------------------------ dessins
def _traiter_clic(retour, cle_cmp, u):
    if not retour or not isinstance(retour, dict):
        return
    derniers = st.session_state.setdefault("ppb_ui_dernier_clic", {})
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
            ecrits.append(libelle(k, u) + " = " + js_str(valeur_widget(k, v)))
    if ecrits:
        st.toast(" ; ".join(ecrits) + " : géométrie et vérifications recalculées")
        st.rerun()


def _panneaux(R, u):
    """Les fenêtres d'objet du composant (vérification complète) : un clic sur
    une pièce ouvre TOUS ses paramètres — une VUE des mêmes clés (libellés
    LIBELLE / LIBELLE_MODE, aides AIDES, options AFFICHAGE), titres courts,
    aucun doublon : a_w ne reste que dans Soudure, αcc dans les avancés, le
    périmètre dans l'en-tête de la carte ; la fenêtre Massif porte tg quand il
    est nul (le mortier n'est alors pas dessiné : amendement 2)."""
    def ch(k):
        f = CHAMPS[k]
        return dict(key=k, lab=libelle(k, u), type=f["t"], options=f.get("o") or [], step=PAS.get(k, 1.0),
                    dis=False, help=AIDES.get(k), affichage=AFFICHAGE.get(k) or None)

    perso = [ch(k) for k in ("h_u", "b_u", "tw_u", "tf_u", "r_u", "t_u") if champ_visible(u, k)]
    tg_nul = (N(u.get("t_g")) or 0.0) <= 0
    g = {}
    g["col"] = dict(titre="Poteau",
                    champs=[ch("forme")] + ([ch("prof_c")] if champ_visible(u, "prof_c") else []) + [ch("nu_c")]
                    + perso + [ch("L_c"), ch("contrev")],
                    note="Lc et le contreventement ne servent qu'à la classification de rigidité (information).")
    g["plate"] = dict(titre="Platine", champs=[ch("bp_u"), ch("lp_u"), ch("tp_u"), ch("nu_pl")],
                      note="bp, lp et tp : aussi des cotes sur le dessin ; la gorge a : fenêtre Soudure.")
    g["anchors"] = dict(titre="Tiges",
                        champs=[ch("n_ax"), ch("n_ay"), ch("w_x"), ch("w_y"), ch("ancr"), ch("classe"), ch("typ_a"),
                                ch("hef")]
                        + ([ch("d_h")] if champ_visible(u, "d_h") else [])
                        + ([ch(k) for k in ("eta_tau", "eta_k1", "eta_nrks", "eta_vrks")] if champ_visible(u, "eta_tau") else []),
                        note="Grille centrée sur l'axe du poteau : wx et wy sont les entraxes TOTAUX.")
    g["pedestal"] = dict(titre="Massif", champs=[ch("beton"), ch("fiss"), ch("bc_u"), ch("lc_u"), ch("hc_u")]
                         + ([ch("t_g")] if tg_nul else []))
    g["grout"] = dict(titre="Mortier", champs=[ch("t_g"), ch("f_g")],
                      note="βj = 2/3 si tg ≤ 0,2·min(bp, lp) et fg ≥ 0,2·fck (EC3 §6.2.5(7)).")
    g["weld"] = dict(titre="Soudure", champs=[ch("a_w")])
    g["efforts"] = dict(titre="Efforts", champs=[ch(k) for k in ("N_Ed", "Mx_Ed", "My_Ed", "Vx_Ed", "Vy_Ed", "T_Ed")],
                        note="Mx comprime le côté y > 0 (haut de la vue en plan) ; My comprime le côté x > 0.")
    return g


def _alerte_courante(R):
    sel = st.session_state.get("ppb_ui_alerte")
    for a in R.alerts:
        if a.id == sel:
            return dict(dims=set(a.dims), elems=set(a.elems))
    return None


def tete_dessin():
    """La tête de la colonne dessin (les deux modes) : niveau de cotation et
    case « Dessin interactif » (déplacée des paramètres avancés : elle
    gouverne cette colonne). Renvoie le niveau."""
    with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="center", key="ppb_tete_dessin"):
        niveau = st.radio(LIBELLE["ui_niveau"], [0, 1, 2], format_func=lambda i: NIVEAUX_ECRAN[i],
                          horizontal=True, key="ppb_ui_niveau", label_visibility="collapsed")
        st.checkbox(LIBELLE["ui_composant"], key="ppb_ui_composant", help=AIDES["ui_composant"])
    return niveau


def dessins(R, u):
    """La colonne du dessin : en tête le niveau de cotation et la case
    « Dessin interactif », puis coupe A-A et vue en plan (répartition),
    empilées (colonne figée par interface.py). Renvoie ``(e, p, hl)``."""
    niveau = tete_dessin()
    interactif = bool(st.session_state.get("ppb_ui_composant", True))
    hl = _alerte_courante(R)
    opt = schemas.options_ecran(R, niveau, hl)
    opt.editables = niveau >= 1
    e = schemas.coupe_aa(R, opt); p = schemas.plan(R, opt)
    valeurs = {k: st.session_state.get(K(k)) for k in CLES}
    groupes = _panneaux(R, u)
    for d, cle in ((e, "ppb_cmp_elev"), (p, "ppb_cmp_plan")):
        if interactif:
            from acier.composants.svg_cliquable import svg_cliquable
            _traiter_clic(svg_cliquable(d.svg(identifiant=cle), valeurs, key=cle, groupes=groupes), cle, u)
        else:
            st.markdown(d.svg(identifiant=cle), unsafe_allow_html=True)
    st.caption(CAPTION_USAGE if interactif else CAPTION_LEGENDE)
    return e, p, hl


def cotes_panneau(u):
    """Les clés du panneau Cotes (dessin fixe), dans les deux modes :
    GEOMETRIE_DESSIN ∩ champ_visible — pas seulement les cotes que le dessin
    émet (tg nul, hef et dh sans boulon restent atteignables)."""
    return [k for k in COTES_ORDRE if champ_visible(u, k)]


def panneau_cotes(u, hl):
    """Repli (tête de la colonne de droite) : une saisie par cote, libellé
    LIBELLE, aide AIDES."""
    cotes = cotes_panneau(u)
    with st.expander("Cotes", expanded=True):
        chauds = set(hl["dims"]) if hl else set()
        for i in range(0, len(cotes), 4):
            cols = st.columns(4)
            for col, k in zip(cols, cotes[i:i + 4]):
                with col:
                    cle = "ppb_cote_" + k
                    st.session_state[cle] = valeur_widget(k, st.session_state.get(K(k)))
                    st.number_input(("🔴 " if _COTE_DE.get(k) in chauds else "") + libelle(k, u), key=cle,
                                    step=PAS.get(k, 1.0), format="%g",
                                    on_change=_ecrire_depuis, args=(cle, k), help=description(u).AIDES.get(k))


# ------------------------------------------------------ onglet vérifications
def _cellule_verif(l):
    return Html(f"{l['lab']}{badge_nature(l['nat'])}")


def _table_simple(R, cle):
    ls = synthese.lignes(R, cle)
    if not ls:
        return False
    st.markdown(f"##### {synthese.titre_table(cle, R)}")
    lignes = [[_cellule_verif(l), l["Ed"], l["Rd"] + (" " + l["unit"] if l["unit"] else ""),
               taux_html(l["eta"], l["ok"], l["c"] is R.gov), l["ref"]] for l in ls]
    st.markdown(tableau_html(["Vérification", "Ed", "Rd", "η", "Réf."], lignes, droite=(1, 2, 3),
                             largeurs=["34%", "12%", "14%", "9%", None]), unsafe_allow_html=True)
    return True


def _formules(R):
    with st.expander("Formules et substitutions numériques"):
        st.caption("Les valeurs introduites sont composées par le moteur ; la ligne du taux est vérifiée "
                   "par un test qui la recalcule.")
        for grp in GROUPES_VERIF:
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
                    f"<span style='color:{C_TRAIT['mut']};'>{_esc(c.ref)}</span><br>"
                    f"<span style='font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:0.95em;'>{corps}</span></div>",
                    unsafe_allow_html=True)


def _tableau_33(R):
    """Le Tableau 3.3 replié (les deux modes) : verdict dans le titre de
    l'expander « Pinces et entraxes (Tab. 3.3) : conforme », ouvert d'office
    dès une ligne NON OK ; lignes nommées par le moteur (x.lab), boutons
    Localiser, règles en caption."""
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
                    st.button("Localiser : " + x.lab.split(" (")[0], key=f"ppb_btn_dist_{x.id}", width="stretch",
                              on_click=_selectionner, args=(x.alert, True))
        st.caption("e ≥ 1,2·d0 au bord de la platine (Tableau 3.3) ; les distances aux bords du MASSIF relèvent de "
                   "l'EN 1992-4 (cône, bord, fendage).")
    tableau_33_replie(R.dist_ok, rendre)


def _parametres_retenus(R):
    u = R.u
    g = [["Périmètre", "EN 1993-1-8 seule — béton d'ancrage (EN 1992-4) en information" if R.perim_ec3
          else "Complet — EN 1993-1-8 + EN 1992-4"],
         ["Poteau", R.nom_c + " " + str(u.nu_c) + " – fy = " + F(R.fy_c, 0) + " MPa"],
         ["Platine", js_str(N(u.bp_u)) + " × " + js_str(N(u.lp_u)) + " × " + js_str(N(u.tp_u)) + " mm " + str(u.nu_pl)
          + " – fy = " + F(R.fy_p, 0) + " MPa (épaisseur)"],
         ["Tiges", js_str(R.n_t) + " × " + str(u.ancr) + " " + str(u.classe) + " – d0 = " + F(R.d0, 0) + " mm, As = "
          + F(R.As, 0) + " mm² – " + ancrage_txt(u) + (" mm" if champ_visible(u, "d_h") else "")],
         ["Béton", "fck = " + F(R.fck, 0) + " MPa, fcd = αcc·fck/γc = " + F(R.fcd, 1) + " MPa, Ecm = "
          + F(R.E_cm / 1e3, 1) + " GPa, " + ("fissuré" if R.fiss else "non fissuré")],
         ["Appui : fjd = βj·fcd·√(Ac1/Ac0)", F(R.fjd, 1) + " MPa (βj = " + F(R.beta_j, 3) + ", √(Ac1/Ac0) = "
          + F(R.ratio_c, 2) + ")"],
         ["Bande c et aire efficace", "c = " + F(R.c, 1) + " mm ; Aeff = " + F(R.A_eff / 100, 0) + " cm² ; NRd = "
          + F(R.N_Rd / 1e3, 0) + " kN"],
         ["Raideurs de la répartition", "kc = " + F(R.k_c, 1) + " N/mm³ ; kb = " + F(R.k_b / 1e3, 0) + " kN/mm"],
         ["Répartition élastique", "C = " + F(R.C, 1) + " kN en (" + F(R.xC, 0) + " ; " + F(R.yC, 0) + ") mm ; pmax = "
          + F(R.pmax, 1) + " MPa ; zone comprimée " + F(R.A_c / 100, 0) + " cm² ; ΣT = " + F(R.T_tot, 1) + " kN"],
         ["Longueur d'allongement Lb = 8d + tg + tp + 0,6d", F(R.Lb, 0) + " mm"],
         ["Résistance d'une tige Ft,anc = min(EC3 ; EC2-4" + (" ; adhérence" if R.Ft_bond is not None else "") + ")",
          F(R.Ft_anc / 1e3, 1) + " kN"],
         ["Cordons : point le plus sollicité", str(R.cordons.get("point", "—")) + " – L = " + F(R.cordons.get("L", 0), 0) + " mm"]]
    if R.rig:
        g.append(["Rigidité Sj,ini / seuil rigide (§5.2.2.5)", F(R.Sj / 1e3, 1) + " / " + F(R.Sj_seuil / 1e3, 1)
                  + " MNm/rad → " + ("RIGIDE" if R.rig["rigide"] else "SEMI-RIGIDE") + " (information)"])
    with st.expander("Paramètres retenus"):
        st.markdown(tableau_md(["Paramètre", "Valeur"], g))


def onglet_verifications(R):
    """La platine à la main en tête, les tables par élément ; puis la légende
    des natures (une ligne + « Légende et références »), le périmètre, les
    formules, le Tableau 3.3 et les paramètres retenus, repliés."""
    bloc_platine(R)
    for cle in synthese.ORDRE:
        _table_simple(R, cle)
    legende_natures(NATURES, LEGENDE_REFERENCES, textes=TEXTES_NATURES)
    expander_perimetre(NON_REVERIFIES, [synthese.court(c) for c in R.checks if not c.active])
    _formules(R)
    _tableau_33(R)
    _parametres_retenus(R)


# ------------------------------------------------------- onglet répartition
def onglet_repartition(R):
    st.caption("La répartition élastique (platine rigide, lit sans traction, tiges tendues) donne l'effort de "
               "chaque tige — c'est elle qui alimente les vérifications d'EN 1992-4. Le domaine (Mx, My) compare "
               "le niveau 1 (verdict) et le niveau 2 (information).")
    c1, c2 = st.columns([1, 1.1], gap="medium")
    with c1:
        st.markdown("##### Domaine résistant (Mx, My) à N = " + F(R.u.N_Ed, 0) + " kN")
        st.markdown(schemas.domaine(R).svg(identifiant="ppb_dom"), unsafe_allow_html=True)
        lignes = []
        for nom, lab in (("x+", "Mx > 0 (comprime y > 0)"), ("x-", "Mx < 0"), ("y+", "My > 0 (comprime x > 0)"),
                         ("y-", "My < 0")):
            lignes.append([lab, F(R.MRd1.get(nom, 0), 1), str(R.branche1.get(nom, "—")), F(R.MRd2.get(nom, 0), 1)])
        st.markdown(tableau_html(["Direction", "Niveau 1 (kNm)", "Branche", "Niveau 2 (kNm)"], lignes,
                                 droite=(1, 3), largeurs=[None, "20%", "26%", "20%"]), unsafe_allow_html=True)
        st.caption("Niveau 1 : " + pct(R.eta1, 1) + " (verdict) · niveau 2 : " + pct(R.eta2, 1)
                   + " (information, M,Rd exact dans la direction appliquée = " + F(R.M_R2, 1) + " kNm).")
    with c2:
        st.markdown("##### Tiges : position, classe, efforts")
        lignes = []
        for t in R.tiges:
            tendue = t["T"] > 1e-6
            lignes.append([F(t["x"], 0) + " ; " + F(t["y"], 0), t["classe"], t.get("rangee", "—"),
                           F(t["T"], 1) if tendue else "0", F(t["FT_Rd"] / 1e3, 1),
                           taux_html(t["T"] / (t["FT_Rd"] / 1e3), t["T"] <= t["FT_Rd"] / 1e3 + 1e-9) if tendue and t["FT_Rd"] > 0 else "—",
                           F(t["V_ec3"], 1) + " / " + F(t["V_en"], 1)])
        st.markdown(tableau_html(["x ; y (mm)", "Classe", "Rangée", "T (kN)", "FT,Rd (kN)", "η", "V EC3 / EC2-4 (kN)"],
                                 lignes, droite=(3, 4, 5, 6)), unsafe_allow_html=True)
        st.caption("Classes : ext = au-delà d'une paroi (Tab. 6.6) ; coin = au-delà de l'about d'aile ; lat = entre "
                   "les niveaux d'aile ; int = entre les ailes (Tab. 6.4). FT,Rd : résistance par tige de sa rangée "
                   "(tronçon en T, acier compris).")
        st.markdown("##### Rangées : tronçons en T")
        lignes = []
        for r in R.rangees:
            lignes.append([r["cle"], r["classe"], F(r["m"], 1), F(r["leff1"], 0) + " / " + F(r["leff2"], 0),
                           F(r["Lb"], 0) + " / " + ("∞" if r["Lb_star"] == float("inf") else F(r["Lb_star"], 0)),
                           str(r["mode"]), F(r["FT_Rd"] / 1e3, 1)])
        st.markdown(tableau_html(["Rangée", "Classe", "m (mm)", "Σleff,1 / Σleff,2 (mm)", "Lb / Lb* (mm)", "Mode",
                                  "FT,Rd (kN)"], lignes, droite=(2, 3, 4, 6)), unsafe_allow_html=True)


# -------------------------------------------------------------- onglet note
def onglet_note(R, u):
    c1, c2 = st.columns([1.6, 2.6], vertical_alignment="center")
    complet = c1.checkbox("Texte : toutes les vérifications", key="ppb_ui_txt_complet")
    t = texte.construire_texte(R, complet)
    with c1:
        st.download_button("⬇️ Exporter le texte (.txt)", data=("﻿" + t).encode("utf-8"),
                           file_name=texte.nom_fichier(u, ".txt"), mime="text/plain;charset=utf-8",
                           width="stretch", key="ppb_btn_txt")
    c2.caption("Note de calcul (2 pages, A4 paysage — synthèse + plan de principe) : bouton "
               "« 📄 PDF » de la barre d'outils. Le texte ci-dessous se copie avec l'icône "
               "en haut à droite du cadre.")
    st.code(t, language=None)


# --------------------------------------------------------- onglet benchmark
def _charger(inp, ecrire):
    from .entrees import defaults, MODE_COMPLET
    d = defaults(); d.update(inp); d["mode"] = MODE_COMPLET; ecrire(d)       # les cas VAL-A / VAL-B sont ceux de la vérification complète
    st.session_state["ppb_ui_alerte"] = None
    st.session_state["ppb_ui_onglet"] = ONGLETS[0]
    st.session_state["_ppb_toast"] = "Données chargées : voir Vérifications"


def _nd(x):
    return 2 if abs(x) < 1 else (1 if abs(x) < 10 else 0)


def onglet_benchmark(B, ecrire):
    from acier.assemblages.ui_commun import bloc_statut
    from .benchmark import BENCH, VALID
    bloc_statut("Benchmark du module : " + B["statut"],
                ["Ces tableaux sont recalculés par le moteur à chaque ouverture ; ils ne dépendent pas de tes données."],
                "ok" if B["ok"] else "nok")
    st.caption("Source : " + SRC["titre"] + ". " + SRC["org"] + ". Norme : " + SRC["norme"] + ".")
    if not B["bench"]:
        st.caption("Aucun exemple publié n'est rejoué en V1 (lot C : MSB Part 5 pied articulé, SCI P398 pied "
                   "encastré, avec le document sous les yeux). Seuil des calculs manuels : écart ≤ 0,5 %.")
    for i, b in enumerate(B["bench"]):
        st.markdown(f"#### {b['id']}")
        st.caption(b["titre"])
        c1, c2 = st.columns([4, 1.4], vertical_alignment="center")
        c1.caption("Données : " + b["data"])
        with c2:
            st.button("Charger ces données dans l'outil", key=f"ppb_btn_bm_{i}", width="stretch",
                      on_click=_charger, args=(BENCH[i]["inp"], ecrire))
        lignes = []
        for r in b["rows"]:
            n = _nd(r["ref"])
            lignes.append([r["lab"], r["page"], F(r["ref"], n) + ("" if r["unit"] == "-" else " " + r["unit"]),
                           F(r["val"], n + 1), F(r["ec"] * 100, 2) + " %", r["st"]])
        st.markdown(tableau_md(["Vérification", "Réf.", "Référence", "Outil", "Écart", "Statut"], lignes, droite=(2, 3, 4)))
    for f in B.get("fiches", []):
        st.markdown(f"#### {f['id']} – fiche à compléter")
        st.caption(f["titre"] + " — " + f.get("source", "") + ". " + f.get("data", ""))
    st.markdown("#### Vérifications sans exemple publié consulté")
    for x in NOCOVER:
        st.markdown(f"- **{x[0]}** – {x[1]}")
    V = B.get("croisee")
    if V:
        st.markdown(f"#### Validation croisée indépendante du {V['date']} — {V['configurations']} configurations, "
                    f"{V['directions']} directions")
        st.caption("Hors moteur (numpy + scipy) : grille de ressorts pour la répartition élastique, programme linéaire "
                   "(HiGHS) pour le domaine résistant — le domaine exact des champs admissibles bornés. Rapport : "
                   + V["rapport"] + ".")
        st.markdown(tableau_md(["Grandeur", "Écart maximal", "Tolérance"], [list(e) for e in V["ecarts"]], droite=(1, 2)))
        import os
        dossier = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "docs", "validation",
                               "figures")
        for nom in V["figures"]:
            chemin = os.path.join(dossier, nom)
            if os.path.exists(chemin):
                st.image(chemin, width="stretch")
    for i, c in enumerate(B["valid"]):
        st.markdown(f"#### {c['id']} – calcul manuel indépendant")
        c1, c2 = st.columns([4, 1.4], vertical_alignment="center")
        c1.caption(c["titre"] + ". Colonne « manuel » : calcul indépendant hors moteur ; colonne « contrôle » : "
                   "même valeur recomposée à la main (arrondie).")
        with c2:
            st.button("Charger ces données dans l'outil", key=f"ppb_btn_val_{i}", width="stretch",
                      on_click=_charger, args=(VALID[i]["inp"], ecrire))
        lignes = []
        for r in c["rows"]:
            n = 4 if abs(r["main"]) < 10 else (2 if abs(r["main"]) < 1000 else 0)
            lignes.append([r["lab"], F(r["main"], n) + ("" if r["unit"] == "-" else " " + r["unit"]),
                           F(r["ref"], n), F(r["val"], n), F(r["ec"] * 100, 3) + " %", "OK" if r["ok"] else "À CONTRÔLER"])
        st.markdown(tableau_md(["Grandeur", "Manuel", "Contrôle", "Outil", "Écart", ""], lignes, droite=(1, 2, 3, 4)))


def onglet_methode():
    st.markdown(METHODE)


# ------------------------------------------------------------------- onglets
def onglets(R, B, u, ecrire):
    st.session_state["_ppb_R"] = R
    st.segmented_control("Onglet", list(ONGLETS), key="ppb_ui_onglet", label_visibility="collapsed", wrap=True)
    onglet = st.session_state.get("ppb_ui_onglet") or ONGLETS[0]
    if onglet == ONGLETS[1]:
        onglet_repartition(R)
    elif onglet == ONGLETS[2]:
        onglet_note(R, u)
    elif onglet == ONGLETS[3]:
        onglet_benchmark(B, ecrire)
    elif onglet == ONGLETS[4]:
        onglet_methode()
    else:
        onglet_verifications(R)

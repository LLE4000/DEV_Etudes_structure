# -*- coding: utf-8 -*-
"""Résultats à l'écran du pied de poteau : bandeau d'une ligne, alertes
courtes, dessins cliquables (colonne figée : coupe A-A puis vue en plan avec
la répartition), onglets Vérifications · Répartition · Note · Benchmark ·
Méthode. Tout est lu dans ``R`` (moteur) et ``B`` (benchmark)."""
import streamlit as st

from acier.js import N, js_str
from acier.formats import F, pct
from acier.assemblages.ui_commun import (bloc_statut_ligne, bloc_alerte_ligne, badge_nature,
                                         tableau_md, tableau_html, taux_html, Html)
from acier.bibliotheques import PERSO
from . import schemas, texte, synthese, formules
from .notation import ligne_alerte, LEGENDE_REFERENCES
from .entrees import COURT, UNITE, CHAMPS, CLES, champ_visible, forme_RHS, ancrage_txt
from .moteur import GROUPES_VERIF
from .benchmark import SRC, NOCOVER
from .methode import METHODE
from .ecran_saisie import K, valeur_widget, PAS

ONGLETS = ("Vérifications", "Répartition", "Note", "Benchmark", "Méthode")
NIVEAUX_ECRAN = ("Vue simple", "Édition", "Cotations complètes")


# ------------------------------------------------------------------ bandeau
def bandeau(R):
    ok = R.verified
    titre = R.statut + (" – sous réserve des alertes" if ok and (R.reserve or R.alerts) else "")
    comp = "— " + pct(R.eta_max, 1) + (" · " + R.gov.label if R.gov else "")
    if not R.geo_ok:
        comp += " · géométrie non valide"
    if not R.dist_ok:
        comp += " · pinces non conformes"
    bloc_statut_ligne("ok" if ok else "nok", titre, comp,
                      [(nom, eta, okg) for nom, eta, okg, _ in synthese.taux_par_element(R)])
    pire = synthese.pire_information(R)
    if pire is not None:
        st.caption("Béton d'ancrage (EN 1992-4, en information, hors verdict) : " + synthese.court(pire) + " à "
                   + pct(pire.eta, 1) + (" — non satisfait" if not pire.ok else " — satisfait")
                   + ". Périmètre « Complet » dans la carte pour le compter dans le verdict.")


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
    sel = st.session_state.get("ppb_ui_alerte")
    ids = [a.id for a in R.alerts]
    if sel and sel not in ids:
        st.session_state["ppb_ui_alerte"] = None
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
            st.button("Masquer" if est_sel else "Localiser", key=f"ppb_btn_al_{a.id}",
                      use_container_width=True, on_click=_selectionner,
                      args=(a.id, bool(a.dims or a.elems)))
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


# ------------------------------------------------------------------ dessins
def _traiter_clic(retour, cle_cmp):
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
            ecrits.append(COURT.get(k, k) + " = " + js_str(valeur_widget(k, v)))
    if ecrits:
        st.toast(" ; ".join(ecrits) + " : géométrie et vérifications recalculées")
        st.rerun()


def _panneaux(R, u):
    """Les fenêtres d'objet du composant : un clic sur une pièce ouvre TOUS
    ses paramètres."""
    def ch(k, lab):
        f = CHAMPS[k]
        return dict(key=k, lab=lab, type=f["t"], options=f.get("o") or [], step=PAS.get(k, 1.0), dis=False)

    perso = [ch(k, COURT.get(k, k) + " (mm)") for k in ("h_u", "b_u", "tw_u", "tf_u", "r_u", "t_u")
             if champ_visible(u, k)]
    g = {}
    g["col"] = dict(titre="Poteau",
                    champs=[ch("forme", "Forme"), ch("prof_c", "Profilé"), ch("nu_c", "Nuance")] + perso
                    + [ch("L_c", "Hauteur Lc (mm)"), ch("contrev", "Ossature contreventée")],
                    note="Lc et le contreventement ne servent qu'à la classification de rigidité (information).")
    g["plate"] = dict(titre="Platine",
                      champs=[ch("bp_u", "bp — largeur selon x (mm)"), ch("lp_u", "lp — longueur selon y (mm)"),
                              ch("tp_u", "tp — épaisseur (mm)"), ch("nu_pl", "Nuance"), ch("a_w", "Gorge a (mm)")],
                      note="bp, lp, tp et la gorge a : aussi des cotes sur le dessin.")
    g["anchors"] = dict(titre="Tiges d'ancrage",
                        champs=[ch("n_ax", "Files (selon x)"), ch("n_ay", "Rangées (selon y)"),
                                ch("w_x", "wx — entraxe des files (mm)"), ch("w_y", "wy — entraxe des rangées (mm)"),
                                ch("ancr", "Tige"), ch("classe", "Classe"), ch("typ_a", "Ancrage"),
                                ch("hef", "hef (mm)")]
                        + ([ch("d_h", "dh — tête (mm)")] if champ_visible(u, "d_h") else [])
                        + ([ch(k, COURT[k]) for k in ("eta_tau", "eta_k1", "eta_nrks", "eta_vrks")] if champ_visible(u, "eta_tau") else []),
                        note="Grille centrée sur l'axe du poteau : wx et wy sont les entraxes TOTAUX.")
    g["pedestal"] = dict(titre="Massif de béton",
                         champs=[ch("beton", "Classe"), ch("fiss", "Fissuré"), ch("bc_u", "bc — selon x (mm)"),
                                 ch("lc_u", "lc — selon y (mm)"), ch("hc_u", "hc — hauteur (mm)"), ch("a_cc", "αcc")])
    g["grout"] = dict(titre="Mortier de scellement",
                      champs=[ch("t_g", "Épaisseur tg (mm)"), ch("f_g", "Résistance fg (MPa)")],
                      note="βj = 2/3 si tg ≤ 0,2·min(bp, lp) et fg ≥ 0,2·fck (EC3 §6.2.5(7)).")
    g["weld"] = dict(titre="Cordons poteau / platine", champs=[ch("a_w", "Gorge a (mm)")])
    g["efforts"] = dict(titre="Efforts de calcul ELU (section de pied)",
                        champs=[ch("N_Ed", "NEd (kN, compression > 0)"), ch("Mx_Ed", "Mx,Ed (kNm)"),
                                ch("My_Ed", "My,Ed (kNm)"), ch("Vx_Ed", "Vx,Ed (kN)"), ch("Vy_Ed", "Vy,Ed (kN)"),
                                ch("T_Ed", "TEd (kNm)"), ch("perim", "Périmètre de la vérification")],
                        note="Mx comprime le côté y > 0 (haut de la vue en plan) ; My comprime le côté x > 0.")
    return g


def _alerte_courante(R):
    sel = st.session_state.get("ppb_ui_alerte")
    for a in R.alerts:
        if a.id == sel:
            return dict(dims=set(a.dims), elems=set(a.elems))
    return None


def dessins(R, u):
    """La colonne du dessin : coupe A-A puis vue en plan (répartition),
    empilées (colonne figée par interface.py). Renvoie ``(e, p, hl)``."""
    interactif = bool(st.session_state.get("ppb_ui_composant", True))
    niveau = st.radio("Niveau de cotation", [0, 1, 2], format_func=lambda i: NIVEAUX_ECRAN[i],
                      horizontal=True, key="ppb_ui_niveau", label_visibility="collapsed")
    hl = _alerte_courante(R)
    opt = schemas.options_ecran(R, niveau, hl)
    opt.editables = niveau >= 1
    e = schemas.coupe_aa(R, opt); p = schemas.plan(R, opt)
    valeurs = {k: st.session_state.get(K(k)) for k in CLES}
    groupes = _panneaux(R, u)
    for d, cle in ((e, "ppb_cmp_elev"), (p, "ppb_cmp_plan")):
        if interactif:
            from acier.composants.svg_cliquable import svg_cliquable
            _traiter_clic(svg_cliquable(d.svg(identifiant=cle), valeurs, key=cle, groupes=groupes), cle)
        else:
            st.markdown(d.svg(identifiant=cle), unsafe_allow_html=True)
    st.caption("Vue en plan : aire efficace (bleu), zone comprimée de la répartition élastique (rouge), axe "
               "neutre, résultante C et effort T de chaque tige tendue."
               + (" Touche une cote pour la modifier · touche une pièce (poteau, platine, tiges, massif, "
                  "mortier, efforts) pour tous ses paramètres." if interactif else ""))
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
                    cle = "ppb_cote_" + k
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
    st.markdown(f"##### {synthese.titre_table(cle, R)}")
    lignes = [[_cellule_verif(l), l["Ed"], l["Rd"] + (" " + l["unit"] if l["unit"] else ""),
               taux_html(l["eta"], l["ok"], l["c"] is R.gov), l["ref"]] for l in ls]
    st.markdown(tableau_html(["Vérification", "Ed", "Rd", "η", "Réf."], lignes, droite=(1, 2, 3),
                             largeurs=["34%", "12%", "14%", "9%", None]), unsafe_allow_html=True)
    return True


def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


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
                    f"{'#2E7D46' if c.ok else '#9C3341'};font-size:0.88em;'>"
                    f"<b>{_esc(synthese.court(c))}</b>{badge_nature(c.nat)} "
                    f"<span style='color:#6E7480;'>{_esc(c.ref)}</span><br>"
                    f"<span style='font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:0.95em;'>{corps}</span></div>",
                    unsafe_allow_html=True)


def _tableau_33(R):
    st.markdown("##### Pinces – EN 1993-1-8 Tableau 3.3")
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
                st.button("Localiser : " + x.lab.split(" (")[0], key=f"ppb_btn_dist_{x.id}", use_container_width=True,
                          on_click=_selectionner, args=(x.alert, True))
    st.caption("e ≥ 1,2·d0 au bord de la platine (Tableau 3.3) ; les distances aux bords du MASSIF relèvent de "
               "l'EN 1992-4 (cône, bord, fendage).")


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
    bloc_platine(R)
    st.markdown(badge_nature("EC") + " formule directe de l'Eurocode " + badge_nature("EC + INT")
                + " Eurocode et interprétation nommée " + badge_nature("INT")
                + " modèle d'ingénieur (niveau 2, tiges de coin ou latérales) · " + LEGENDE_REFERENCES,
                unsafe_allow_html=True)
    for cle in synthese.ORDRE:
        _table_simple(R, cle)
    st.caption("Non revérifiés ici : fendage et éclatement latéral du massif (EC2-4 §7.2.1.7-8, signalés en "
               "alerte) ; tiges post-installées, raidisseurs, bêche ; flambement et classe de section du poteau ; "
               "poinçonnement du massif au-delà de l'EN 1992-1-1 §6.7.")
    off = [c for c in R.checks if not c.active]
    if off:
        st.caption("Sans objet dans cette configuration : " + " ; ".join(synthese.court(c) for c in off) + ".")
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
                           use_container_width=True, key="ppb_btn_txt")
    c2.caption("Note de calcul (2 pages, A4 paysage — synthèse + plan de principe) : bouton "
               "« 📄 Générer PDF » de la barre d'outils. Le texte ci-dessous se copie avec l'icône "
               "en haut à droite du cadre.")
    st.code(t, language=None)


# --------------------------------------------------------- onglet benchmark
def _charger(inp, ecrire):
    from .entrees import defaults
    d = defaults(); d.update(inp); ecrire(d)
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
            st.button("Charger ces données dans l'outil", key=f"ppb_btn_bm_{i}", use_container_width=True,
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
                st.image(chemin, use_container_width=True)
    for i, c in enumerate(B["valid"]):
        st.markdown(f"#### {c['id']} – calcul manuel indépendant")
        c1, c2 = st.columns([4, 1.4], vertical_alignment="center")
        c1.caption(c["titre"] + ". Colonne « manuel » : calcul indépendant hors moteur ; colonne « contrôle » : "
                   "même valeur recomposée à la main (arrondie).")
        with c2:
            st.button("Charger ces données dans l'outil", key=f"ppb_btn_val_{i}", use_container_width=True,
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
    seg = getattr(st, "segmented_control", None)
    if seg:
        seg("Onglet", list(ONGLETS), key="ppb_ui_onglet", label_visibility="collapsed")
    else:
        st.radio("Onglet", list(ONGLETS), key="ppb_ui_onglet", horizontal=True, label_visibility="collapsed")
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

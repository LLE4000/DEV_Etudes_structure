# -*- coding: utf-8 -*-
"""L'écran du CALCUL MANUEL de la platine : la carte (colonne de droite —
mode, boulons, éléments soudés, raidisseurs, appui, efforts, critère) et
les résultats (bandeau, alertes, dessins cliquables, onglets Vérifications ·
Efforts des tiges · Note · Benchmark · Méthode).

SOURCE UNIQUE : les clés ``ppb_<clé>`` du module ; les listes ``ppb_tiges``,
``ppb_elems``, ``ppb_raid`` sont la source des boulons, éléments et
raidisseurs — les widgets ligne par ligne et les glissements du dessin y
écrivent (clés dérivées « t3_u » : distance au bord gauche, « t3_v » : au
bord haut). La bascule de mode reprend la disposition (conversion.py).
"""
import streamlit as st

from acier.js import N, js_str
from acier.formats import F, pct
from acier.assemblages.ui_commun import (bloc_statut_ligne, badge_nature, tableau_md, tableau_html, taux_html, Html,
                                         titre_bloc)
from acier.bibliotheques import PERSO
from ..entrees import (CHAMPS, COURT, CLES, champ_visible, TYPES_ELEM, TYPE_I, TYPE_RHS, TYPE_CHS, TYPE_PLAT,
                       TYPE_ROND, TYPE_RECT, DIRS_RAID, PROFILS, element_defaut, raidisseur_defaut, MODE_COMPLET,
                       MODE_MANUEL)
from ..ecran_saisie import K, valeur_widget, PAS, LIBELLES, _lignes, _contient_chaud, ligne_visserie, avances, identification
from .. import ecran_resultats as ER
from ..notation import LEGENDE_REFERENCES
from .. import formules
from . import schemas, synthese, texte
from . import elements as EL
from .methode import METHODE
from .moteur import GROUPES_VERIF
from .conversion import vers_manuel, vers_complet, disposition_reguliere

ONGLETS = ("Vérifications", "Efforts des tiges", "Note", "Benchmark", "Méthode")
NIVEAUX_ECRAN = ("Vue simple", "Édition", "Cotations complètes")
DIMS_PAR_TYPE = {TYPE_I: ("h", "b", "tw", "tf", "r"), TYPE_RHS: ("h", "b", "t"), TYPE_CHS: ("d", "t"),
                 TYPE_PLAT: ("b", "t"), TYPE_ROND: ("d",), TYPE_RECT: ("b", "h")}
LIB_DIM = {"h": "h", "b": "b", "tw": "tw", "tf": "tf", "r": "r", "t": "t", "d": "Ø"}


def _verrou():
    return bool(st.session_state.get("prj_verrou", False))


# ------------------------------------------------------------ bascule de mode
def changer_mode(lire, ecrire):
    """on_change du sélecteur de mode : la disposition est reprise d'un mode à
    l'autre ; vers la vérification complète seulement si elle est régulière."""
    u = lire()
    prec = st.session_state.get("ppb_ui_mode_prec")
    nouveau = st.session_state.get(K("mode"))
    if nouveau == prec:
        return
    if nouveau == MODE_COMPLET:
        v = vers_complet(dict(u, mode=MODE_MANUEL))
        if v is None:
            raison = disposition_reguliere(u)["raison"]
            st.session_state[K("mode")] = MODE_MANUEL
            st.session_state["_ppb_toast"] = "Vérification complète impossible : " + raison + "."
            return
        ecrire(v)
        st.session_state["_ppb_toast"] = "Disposition reprise dans la vérification complète"
    else:
        ecrire(vers_manuel(dict(u, mode=MODE_COMPLET)))
        st.session_state["_ppb_toast"] = "Grille et profilé repris dans le calcul manuel"
    st.session_state["ppb_ui_mode_prec"] = nouveau
    st.session_state["ppb_ui_alerte"] = None


def bloc_mode(u, chauds, lire, ecrire):
    """Le bloc MODE de la carte (les deux écrans) : le sélecteur de mode, et
    en mode manuel le critère, l'aire d'appui, le point d'application."""
    st.session_state["ppb_ui_mode_prec"] = st.session_state.get(K("mode"))
    cols = st.columns([0.62, 3], gap="small", vertical_alignment="bottom")
    with cols[0]:
        titre_bloc(("🔴 " if "mode" in chauds else "") + "MODE")
    with cols[1]:
        f = CHAMPS["mode"]
        st.session_state[K("mode")] = valeur_widget("mode", st.session_state.get(K("mode")))
        st.selectbox(LIBELLES.get("mode", f["l"]), f["o"], key=K("mode"), help=f.get("h"), disabled=_verrou(),
                     on_change=changer_mode, args=(lire, ecrire), label_visibility="collapsed")
    if u.get("mode") != MODE_COMPLET:
        _lignes([(["crit", "appui"], None), (["xN", "yN"], None)], u, chauds)
        reg = disposition_reguliere(u)
        st.caption("Vérification complète EN 1993-1-8 + EN 1992-4 : "
                   + ("disponible pour cette disposition (grille régulière autour d'un seul profilé)." if reg["ok"]
                      else "indisponible — " + reg["raison"] + "."))


# --------------------------------------------------------------- les listes
def _liste(cle, etat=None):
    etat = st.session_state if etat is None else etat
    lst = etat.get(K(cle))
    if not isinstance(lst, list):
        lst = []
        etat[K(cle)] = lst
    return lst


def _ecrire_champ_liste(cle, i, champ, wkey, conv=float):
    """on_change d'un widget de ligne : la valeur va dans la liste source."""
    lst = _liste(cle)
    if i < len(lst):
        v = st.session_state.get(wkey)
        try:
            lst[i][champ] = conv(v) if conv is not str else str(v)
        except (TypeError, ValueError):
            pass
        st.session_state[K(cle)] = lst


def _supprimer(cle, i):
    lst = _liste(cle)
    if i < len(lst):
        del lst[i]
        st.session_state[K(cle)] = lst
        st.session_state["_ppb_toast"] = "Ligne supprimée"


def _ajouter(cle, objet):
    lst = _liste(cle)
    lst.append(objet)
    st.session_state[K(cle)] = lst


def _ajouter_boulon(u):
    lp = N(u.get("lp_u")) or 500.0
    tiges = _liste("tiges")
    # au bord bas, à 60 mm, à droite du dernier boulon posé sur cette ligne
    y = -(lp / 2 - 60.0)
    xs = [float(N(t.get("x")) or 0) for t in tiges if abs(float(N(t.get("y")) or 0) - y) < 1e-6]
    x = (max(xs) + 80.0) if xs else 0.0
    _ajouter("tiges", dict(x=x, y=y))


def _poser_grille():
    nx_, ny_ = int(st.session_state.get("ppb_ui_gx") or 2), int(st.session_state.get("ppb_ui_gy") or 2)
    wx, wy = float(st.session_state.get("ppb_ui_gwx") or 400), float(st.session_state.get("ppb_ui_gwy") or 400)
    from .. import geometrie as GE
    st.session_state[K("tiges")] = [dict(x=float(x), y=float(y)) for (x, y, _i, _j) in GE.positions_tiges(nx_, ny_, wx, wy)]
    st.session_state["_ppb_toast"] = f"Grille {nx_} × {ny_} posée"


def _num(label, wkey, valeur, cle, i, champ, step=1.0, aide=None, visible=False):
    st.session_state[wkey] = float(valeur) if valeur is not None else 0.0
    st.number_input(label, key=wkey, step=step, format="%g", on_change=_ecrire_champ_liste, args=(cle, i, champ, wkey),
                    disabled=_verrou(), help=aide, label_visibility="visible" if visible else "collapsed")


def _sel(label, wkey, valeur, options, cle, i, champ, visible=False):
    st.session_state[wkey] = valeur if valeur in options else options[0]
    st.selectbox(label, options, key=wkey, on_change=_ecrire_champ_liste, args=(cle, i, champ, wkey, str),
                 disabled=_verrou(), label_visibility="visible" if visible else "collapsed")


def _etiq(txt, fort=False):
    st.markdown(f"<div style='font-size:0.8em;color:{'#33415C' if fort else '#6E7480'};line-height:1.2;"
                f"padding:0 0 6px 0;{'font-weight:700;' if fort else ''}'>{txt}</div>", unsafe_allow_html=True)


def _entete(titre, labels, poids):
    """Une ligne d'en-tête : le titre du bloc dans la colonne de gauche, les
    libellés des colonnes de la table (les lignes ont leurs étiquettes masquées)."""
    cols = st.columns([0.62] + poids, gap="small", vertical_alignment="bottom")
    with cols[0]:
        if titre:
            titre_bloc(titre)
    for c, lab in zip(cols[1:], labels):
        with c:
            if lab:
                _etiq(lab)
    return cols


def _ligne(poids):
    return st.columns([0.62] + poids, gap="small", vertical_alignment="center")


def _poubelle(key, cle, i, aide):
    st.button("🗑️", key=key, on_click=_supprimer, args=(cle, i), disabled=_verrou(), help=aide, use_container_width=True)


def bloc_boulons(u, chauds):
    tiges = _liste("tiges")
    chaud = _contient_chaud([(["ancr", "classe", "typ_a", "hef", "d_h"], None)], chauds, u) or "anchors" in chauds
    _lignes([(["ancr", "classe", "typ_a"], None), (["hef", "d_h"], None)], u, chauds, ("🔴 " if chaud else "") + "BOULONS")
    P = [1, 1, 0.45]
    _entete("", ["x (mm)", "y (mm)", ""], P)
    for i, t in enumerate(tiges):
        cols = _ligne(P)
        with cols[0]:
            _etiq(f"n° {i + 1}", True)
        with cols[1]:
            _num("x (mm)", f"ppb_t{i}_x", N(t.get("x")), "tiges", i, "x", 5.0)
        with cols[2]:
            _num("y (mm)", f"ppb_t{i}_y", N(t.get("y")), "tiges", i, "y", 5.0)
        with cols[3]:
            _poubelle(f"ppb_btn_tdel_{i}", "tiges", i, "Retirer ce boulon")
    cols = _ligne([1, 1.4])
    with cols[1]:
        st.button("➕ Boulon", key="ppb_btn_tadd", on_click=_ajouter_boulon, args=(u,), disabled=_verrou(),
                  use_container_width=True, help="Un boulon de plus (puis le déplacer sur le dessin ou taper x, y)")
    with cols[2]:
        with st.popover("Poser une grille…", use_container_width=True, disabled=_verrou()):
            for k, v in (("ppb_ui_gx", 2), ("ppb_ui_gy", 2), ("ppb_ui_gwx", 400.0), ("ppb_ui_gwy", 400.0)):
                st.session_state.setdefault(k, v)
            g1, g2 = st.columns(2)
            with g1:
                st.number_input("Files (x)", key="ppb_ui_gx", min_value=1, step=1)
                st.number_input("wx (mm)", key="ppb_ui_gwx", step=10.0, format="%g")
            with g2:
                st.number_input("Rangées (y)", key="ppb_ui_gy", min_value=1, step=1)
                st.number_input("wy (mm)", key="ppb_ui_gwy", step=10.0, format="%g")
            st.button("Poser la grille (remplace les boulons)", key="ppb_btn_grille", on_click=_poser_grille,
                      type="primary", use_container_width=True)
    st.caption("Repère : origine au centre de la platine, x vers la droite, y vers le haut de la vue en plan. "
               "Chaque boulon se déplace aussi à la souris sur le dessin.")


def bloc_elements(u, chauds):
    elems = _liste("elems")
    P = [1.7, 1.1, 0.75, 0.75, 0.45]
    _entete(("🔴 " if "col" in chauds else "") + "ÉLÉMENTS SOUDÉS", ["Type", "Profilé · rotation", "x (mm)", "y (mm)", ""], P)
    for i, e in enumerate(elems):
        e = dict(element_defaut(), **{k: v for k, v in e.items() if v is not None})
        t = e.get("type") if e.get("type") in TYPES_ELEM else TYPE_I
        cols = _ligne(P)
        with cols[0]:
            _etiq(f"É{i + 1}", True)
        with cols[1]:
            _sel("Type", f"ppb_e{i}_type", t, TYPES_ELEM, "elems", i, "type")
        with cols[2]:
            if t == TYPE_I:
                _sel("Profilé", f"ppb_e{i}_prof", e.get("prof") or "HEA 300", PROFILS, "elems", i, "prof")
            else:
                _sel("Rotation", f"ppb_e{i}_rot", str(int(N(e.get("rot")) or 0)), ["0", "90"], "elems", i, "rot")
        with cols[3]:
            _num("x (mm)", f"ppb_e{i}_x", N(e.get("x")), "elems", i, "x", 5.0)
        with cols[4]:
            _num("y (mm)", f"ppb_e{i}_y", N(e.get("y")), "elems", i, "y", 5.0)
        with cols[5]:
            _poubelle(f"ppb_btn_edel_{i}", "elems", i, "Retirer cet élément")
        perso = t != TYPE_I or (e.get("prof") or "HEA 300") == PERSO
        dims = list(DIMS_PAR_TYPE[t]) if perso else []
        extra = ["rot"] if t == TYPE_I else []
        if dims or extra:
            n = len(dims) + len(extra)
            cols = st.columns([0.62] + [1] * n + [1] * max(0, 5 - n), gap="small", vertical_alignment="bottom")
            k = 1
            for d in dims:
                with cols[k]:
                    _num(LIB_DIM[d] + " (mm)", f"ppb_e{i}_{d}", N(e.get(d)), "elems", i, d, 1.0, visible=True)
                k += 1
            for d in extra:
                with cols[k]:
                    _sel("Rotation (°)", f"ppb_e{i}_rot", str(int(N(e.get("rot")) or 0)), ["0", "90"], "elems", i, "rot",
                         visible=True)
                k += 1
    cols = _ligne([1.7, 2.6])
    with cols[1]:
        st.button("➕ Élément soudé", key="ppb_btn_eadd", on_click=_ajouter, args=("elems", element_defaut()),
                  disabled=_verrou(), use_container_width=True, help="Profilé en I, tube, plat, rond ou rectangle de plus, à l'axe")


def bloc_raidisseurs(u, chauds):
    raid = _liste("raid")
    P = [0.8, 0.8, 0.8, 0.8, 0.7, 0.7, 0.45]
    _entete(("🔴 " if "raid" in chauds else "") + "RAIDISSEURS",
            ["x (mm)", "y (mm)", "Sens", "L (mm)", "t (mm)", "h (mm)", ""] if raid else [], P)
    for i, r in enumerate(raid):
        r = dict(raidisseur_defaut(), **{k: v for k, v in r.items() if v is not None})
        cols = _ligne(P)
        with cols[0]:
            _etiq(f"R{i + 1}", True)
        with cols[1]:
            _num("x (mm)", f"ppb_r{i}_x", N(r.get("x")), "raid", i, "x", 5.0)
        with cols[2]:
            _num("y (mm)", f"ppb_r{i}_y", N(r.get("y")), "raid", i, "y", 5.0)
        with cols[3]:
            _sel("Sens", f"ppb_r{i}_dir", str(r.get("dir") or "+x").replace("-", "−"), DIRS_RAID, "raid", i, "dir")
        with cols[4]:
            _num("L (mm)", f"ppb_r{i}_L", N(r.get("L")), "raid", i, "L", 5.0)
        with cols[5]:
            _num("t (mm)", f"ppb_r{i}_t", N(r.get("t")), "raid", i, "t", 1.0)
        with cols[6]:
            _num("h (mm)", f"ppb_r{i}_h", N(r.get("h")), "raid", i, "h", 10.0, "Hauteur du raidisseur (dessin des coupes)")
        with cols[7]:
            _poubelle(f"ppb_btn_rdel_{i}", "raid", i, "Retirer ce raidisseur")
    cols = _ligne([1.7, 2.6])
    with cols[1]:
        st.button("➕ Raidisseur", key="ppb_btn_radd", on_click=_ajouter, args=("raid", raidisseur_defaut()),
                  disabled=_verrou(), use_container_width=True,
                  help="Un plat vertical soudé sur la platine, parti de (x ; y) dans un sens")


def carte(u, chauds, lire, ecrire):
    """La carte complète du mode manuel."""
    with st.container(gap=None):
        bloc_mode(u, chauds, lire, ecrire)
        _lignes([(["nu_pl"], None)], u, chauds, ("🔴 " if "nu_pl" in chauds else "") + "PLATINE")
        bloc_boulons(u, chauds)
        ligne_visserie(u)
        bloc_elements(u, chauds)
        bloc_raidisseurs(u, chauds)
        _lignes([(["beton", "hc_u"], None)], u, chauds, ("🔴 " if _contient_chaud([(["beton", "hc_u"], None)], chauds, u) else "") + "BÉTON")
        _lignes([(["N_Ed", "Mx_Ed", "My_Ed"], None), (["Vx_Ed", "Vy_Ed", "T_Ed"], None)], u, chauds,
                ("🔴 " if _contient_chaud([(["N_Ed", "Mx_Ed", "My_Ed", "Vx_Ed", "Vy_Ed", "T_Ed"], None)], chauds, u) else "")
                + "EFFORTS ELU")
    avances(u, chauds)
    identification(u, chauds)


# ------------------------------------------------------------------ bandeau
def bandeau(R):
    ok = R.verified
    titre = R.statut + (" – sous réserve des alertes" if ok and (R.reserve or R.alerts) else "")
    P = R.plaque
    req = "∞" if P["t_req"] == float("inf") else F(P["t_req"], 1)
    comp = ("— épaisseur requise " + req + " mm" + (" ≤ " if ok else " > ") + "tp = " + F(R.u.tp_u, 0) + " mm · "
            + pct(R.eta_max, 1) + (" · " + R.gov.label.split(" (")[0] if R.gov else ""))
    if not R.geo_ok:
        comp += " · géométrie non valide"
    if not R.dist_ok:
        comp += " · pinces non conformes"
    bloc_statut_ligne("ok" if ok else "nok", titre, comp,
                      [(nom, eta, okg) for nom, eta, okg, _ in synthese.taux_par_element(R)])
    pire = synthese.pire_information(R)
    if pire is not None:
        st.caption("En information, hors verdict : " + synthese.court(pire) + " à " + pct(pire.eta, 1)
                   + (" — non satisfait" if not pire.ok else " — satisfait")
                   + ". Le verdict porte sur la platine ; les tiges et le béton relèvent du logiciel d'ancrage ou de "
                   "la vérification complète.")


# ------------------------------------------------------------------ dessins
def _valeurs(u, R):
    """Les valeurs des clés d'entrée ET des clés dérivées des pièces déplaçables."""
    v = {k: st.session_state.get(K(k)) for k in CLES}
    bp, lp = N(u.get("bp_u")) or 0.0, N(u.get("lp_u")) or 0.0
    for t in R.tiges:
        v[f"t{t['i']}_u"], v[f"t{t['i']}_v"] = round(t["x"] + bp / 2), round(lp / 2 - t["y"])
        v[f"t{t['i']}_x"], v[f"t{t['i']}_y"] = t["x"], t["y"]
    for i, e in enumerate(R.elems):
        v[f"e{i}_u"], v[f"e{i}_v"] = round(e["x"] + bp / 2), round(lp / 2 - e["y"])
        v[f"e{i}_x"], v[f"e{i}_y"], v[f"e{i}_rot"], v[f"e{i}_type"], v[f"e{i}_prof"] = e["x"], e["y"], str(e["rot"]), e["type"], e["prof"]
        for d in ("h", "b", "tw", "tf", "r", "t", "d"):
            v[f"e{i}_{d}"] = e[d]
    for i, r in enumerate(R.raid):
        v[f"r{i}_u"], v[f"r{i}_v"] = round(r["x"] + bp / 2), round(lp / 2 - r["y"])
        v[f"r{i}_x"], v[f"r{i}_y"], v[f"r{i}_dir"], v[f"r{i}_L"], v[f"r{i}_t"], v[f"r{i}_h"] = r["x"], r["y"], r["dir"], r["L"], r["t"], r["h"]
    return v


def _appliquer_derivee(k, v, u, etat=None):
    """Une clé dérivée (« t3_u », « e0_x », « r1_dir »…) → la liste source
    (``etat`` : l'état de session, celui de Streamlit par défaut). True si écrit."""
    import re
    etat = st.session_state if etat is None else etat
    m = re.match(r"^([ter])(\d+)_([A-Za-z]+)$", k)
    if not m:
        return False
    cle = {"t": "tiges", "e": "elems", "r": "raid"}[m.group(1)]
    i, champ = int(m.group(2)), m.group(3)
    lst = _liste(cle, etat)
    if i >= len(lst):
        return False
    bp, lp = N(u.get("bp_u")) or 0.0, N(u.get("lp_u")) or 0.0
    if champ == "u":
        lst[i]["x"] = float(v) - bp / 2
    elif champ == "v":
        lst[i]["y"] = lp / 2 - float(v)
    elif champ in ("type", "prof", "dir"):
        lst[i][champ] = str(v)
    elif champ == "rot":
        lst[i]["rot"] = int(float(v))
    else:
        lst[i][champ] = float(v)
    etat[K(cle)] = lst
    return True


def _traiter_clic(retour, cle_cmp, u):
    if not retour or not isinstance(retour, dict):
        return
    derniers = st.session_state.setdefault("ppb_ui_dernier_clic", {})
    if derniers.get(cle_cmp) == retour.get("t"):
        return
    derniers[cle_cmp] = retour.get("t")
    changes = retour.get("changes") or {}
    ecrits = []
    for k, v in changes.items():
        if k in CHAMPS:
            st.session_state[K(k)] = valeur_widget(k, v)
            ecrits.append(COURT.get(k, k) + " = " + js_str(valeur_widget(k, v)))
        elif _appliquer_derivee(k, v, u):
            ecrits.append(k.split("_")[0] + " déplacé" if k.endswith(("_u", "_v")) else k + " = " + js_str(v))
    if ecrits:
        st.toast(" ; ".join(sorted(set(ecrits))) + " : recalculé")
        st.rerun()


def _panneaux(R, u):
    """Les fenêtres d'objet du composant."""
    def ch(k, lab):
        f = CHAMPS[k]
        return dict(key=k, lab=lab, type=f["t"], options=f.get("o") or [], step=PAS.get(k, 1.0), dis=False)

    def nb(k, lab, step=5.0):
        return dict(key=k, lab=lab, type="n", options=[], step=step, dis=False)

    def se(k, lab, options):
        return dict(key=k, lab=lab, type="s", options=list(options), step=1, dis=False)
    g = {}
    g["plate"] = dict(titre="Platine", champs=[ch("bp_u", "bp — largeur selon x (mm)"), ch("lp_u", "lp — longueur selon y (mm)"),
                                               ch("tp_u", "tp — épaisseur (mm)"), ch("nu_pl", "Nuance"), ch("a_w", "Gorge a (mm)")])
    g["pedestal"] = dict(titre="Massif de béton", champs=[ch("beton", "Classe"), ch("bc_u", "bc — selon x (mm)"),
                                                           ch("lc_u", "lc — selon y (mm)"), ch("hc_u", "hc — hauteur (mm)"), ch("a_cc", "αcc")])
    g["grout"] = dict(titre="Mortier de scellement", champs=[ch("t_g", "Épaisseur tg (mm)"), ch("f_g", "Résistance fg (MPa)")])
    g["weld"] = dict(titre="Cordons élément / platine", champs=[ch("a_w", "Gorge a (mm)")],
                     note="La gorge ne sert qu'au bras de levier (0,8·a·√2 déduit) : les cordons ne sont pas vérifiés ici.")
    g["anchors"] = dict(titre="Boulons", champs=[ch("ancr", "Boulon"), ch("classe", "Classe"), ch("typ_a", "Ancrage"), ch("hef", "hef (mm)")]
                        + ([ch("d_h", "dh — tête (mm)")] if champ_visible(u, "d_h") else []))
    g["efforts"] = dict(titre="Efforts de calcul ELU",
                        champs=[ch("N_Ed", "NEd (kN, compression > 0)"), ch("Mx_Ed", "Mx,Ed (kNm)"), ch("My_Ed", "My,Ed (kNm)"),
                                ch("Vx_Ed", "Vx,Ed (kN)"), ch("Vy_Ed", "Vy,Ed (kN)"), ch("T_Ed", "TEd (kNm)"),
                                ch("xN", "xN — point d'application (mm)"), ch("yN", "yN (mm)")],
                        note="Mx comprime le côté y > 0 (haut de la vue en plan) ; My comprime x > 0 ; efforts en (xN ; yN).")
    for t in R.tiges:
        g[f"bolt{t['i']}"] = dict(titre=f"Boulon n° {t['i'] + 1}", champs=[nb(f"t{t['i']}_x", "x (mm)"), nb(f"t{t['i']}_y", "y (mm)")],
                                  note="Se déplace aussi à la souris. Pour le retirer : 🗑️ dans la carte.")
    for i, e in enumerate(R.elems):
        champs = [se(f"e{i}_type", "Type", TYPES_ELEM), nb(f"e{i}_x", "x (mm)"), nb(f"e{i}_y", "y (mm)"), se(f"e{i}_rot", "Rotation (°)", ["0", "90"])]
        if e["type"] == TYPE_I:
            champs.insert(1, se(f"e{i}_prof", "Profilé", PROFILS))
        for d in DIMS_PAR_TYPE[e["type"]]:
            if e["type"] == TYPE_I and e["prof"] != PERSO:
                break
            champs.append(nb(f"e{i}_{d}", LIB_DIM[d] + " (mm)", 1.0))
        g[f"elem{i}"] = dict(titre=f"Élément {i + 1} — {EL.nom(e)}", champs=champs, note="Se déplace aussi à la souris.")
    for i, r in enumerate(R.raid):
        g[f"raid{i}"] = dict(titre=f"Raidisseur {i + 1}", champs=[nb(f"r{i}_x", "x (mm)"), nb(f"r{i}_y", "y (mm)"),
                                                                 se(f"r{i}_dir", "Sens", DIRS_RAID), nb(f"r{i}_L", "L (mm)"),
                                                                 nb(f"r{i}_t", "t (mm)", 1.0), nb(f"r{i}_h", "h (mm)", 10.0)])
    return g


def dessins(R, u):
    """La colonne du dessin : coupe A-A puis vue en plan. Renvoie ``(e, p, hl)``."""
    interactif = bool(st.session_state.get("ppb_ui_composant", True))
    niveau = st.radio("Niveau de cotation", [0, 1, 2], format_func=lambda i: NIVEAUX_ECRAN[i],
                      horizontal=True, key="ppb_ui_niveau", label_visibility="collapsed")
    hl = ER._alerte_courante(R)
    opt = schemas.options_ecran(R, niveau, hl)
    opt.editables = niveau >= 1
    e, p = schemas.coupe_aa(R, opt), schemas.plan(R, opt)
    valeurs = _valeurs(u, R)
    groupes = _panneaux(R, u)
    for d, cle in ((e, "ppb_cmp_elev"), (p, "ppb_cmp_plan")):
        if interactif:
            from acier.composants.svg_cliquable import svg_cliquable
            _traiter_clic(svg_cliquable(d.svg(identifiant=cle), valeurs, key=cle, groupes=groupes), cle, u)
        else:
            st.markdown(d.svg(identifiant=cle), unsafe_allow_html=True)
    st.caption("Vue en plan : aire d'appui (bleu), zone comprimée (rouge), résultante C, axe neutre, effort T et "
               "bras de levier m de chaque boulon tendu jusqu'à son appui, bande comprimée la plus sollicitée (mEd)."
               + (" Glisser un boulon, un élément ou un raidisseur pour le déplacer ; toucher une pièce pour ses "
                  "paramètres ; toucher une cote pour la modifier." if interactif else ""))
    return e, p, hl


# ------------------------------------------------- vérifications à la main
def _lig(lab, val, fort=False):
    return [Html(f"<b>{ER._esc(lab)}</b>" if fort else ER._esc(lab)), Html(f"<b>{ER._esc(val)}</b>" if fort else ER._esc(val))]


def bloc_main(R):
    """Le calcul à la main : l'épaisseur requise et ses quatre critères, puis
    boulon par boulon, face par face, la bande comprimée et les trous."""
    P = R.plaque
    u = R.u
    c = R.ck["tpq"]
    st.markdown("##### Épaisseur de la platine — le calcul à la main")
    if not c.active:
        st.caption("Géométrie non valide ou équilibre introuvable : pas d'épaisseur requise.")
        return
    crit = {"T": "tronçon en T des boulons", "S": "section au nu d'une paroi", "c": "bande comprimée", "b": "pression diamétrale"}

    def mm(v):
        return "∞" if v == float("inf") else F(v, 1)
    req = "∞ (aucune épaisseur ne suffit)" if c.Ed == float("inf") else F(c.Ed, 1) + " mm"
    bloc_statut_ligne("ok" if c.ok else "nok",
                      "Épaisseur requise " + req + (" ≤ " if c.ok else " > ") + "tp = " + F(u.tp_u, 0) + " mm",
                      "— " + pct(c.eta, 1) + " · critère : " + crit[P["pire"]] + " · " + P["critere"] + " · fy = " + F(R.fy_p, 0) + " MPa",
                      [(crit[k], P["t_" + k] / u.tp_u if u.tp_u else 0.0, P["t_" + k] <= u.tp_u + 1e-9) for k in ("T", "S", "c", "b")])
    tendus = [t for t in R.tiges if t["T"] > 1e-9]
    g1, g2 = st.columns(2, gap="medium")
    with g1:
        st.markdown("**Côté tendu — boulon par boulon (tronçon en T)**")
        if tendus:
            lignes = []
            for t in tendus:
                lignes.append([js_str(t["i"] + 1), F(t["T"], 1), t["appui"]["face"].replace("élément ", "É"),
                               F(t["m"], 1) + (" (angle)" if t["coin"] else ""), F(t["M"], 2), F(t["leff1"], 0),
                               F(t["FT_Rd"], 0) + " (m" + str(t["mode_ts"]).split(" ")[0] + ")",
                               taux_html(t["eta"], t["eta"] <= 1 + 1e-9), F(t["t_pl"], 1) + " / " + F(t["t_el"], 1)])
            st.markdown(tableau_html(["n°", "T (kN)", "Appui", "m (mm)", "T·m (kNm)", "leff (mm)", "FT,Rd (kN)", "η", "tp plast. / élast."],
                                     lignes, droite=(1, 3, 4, 5, 6, 7, 8)), unsafe_allow_html=True)
            pt = R.pire_tige
            st.caption(f"Boulon {pt['i'] + 1} : Lb = {F(R.Lb, 0)} / Lb* = {F(min(pt['Lb_star'], 9999), 0)} mm → "
                       + ("avec" if pt["levier"] else "sans") + " effet de levier, mode " + str(pt["mode_ts"])
                       + f" ; Mpl,1 = 0,25·leff·tp²·fy/γM0 = {F(pt['Mpl1'], 2)} kNm ; σ Von Mises de la bande (2·leff) = {F(pt['svm'], 0)} MPa.")
        else:
            st.caption("Aucun boulon tendu : la platine ne fléchit pas en traction.")
        st.markdown("**Côté tendu — section au nu de chaque paroi**")
        if R.sections:
            lignes = []
            for s in R.sections:
                lignes.append([s["nom"].replace("élément ", "É").replace("raidisseur ", "R"),
                               ", ".join(str(i + 1) for i in s["idx"]), ", ".join(F(m, 0) for m in s["m"]),
                               F(s["M"], 2), F(s["b"], 0), F(s["Mpl"], 2), taux_html(s["eta"], s["eta"] <= 1 + 1e-9),
                               F(s["t_pl"], 1) + " / " + F(s["t_el"], 1)])
            st.markdown(tableau_html(["Face", "Boulons", "m⊥ (mm)", "ΣT·m (kNm)", "b (mm)", "Mpl (kNm)", "η", "tp plast. / élast."],
                                     lignes, droite=(2, 3, 4, 5, 6, 7)), unsafe_allow_html=True)
            st.caption("b = min(platine ; L + 2·m) : la face de longueur L et la diffusion à 45° sur le bras.")
        else:
            st.caption("Aucune section : pas de boulon tendu devant une paroi.")
    with g2:
        st.markdown("**Côté comprimé — la bande la plus sollicitée**")
        cs = R.console
        rows = [_lig("Pression sous la platine (répartition rigide)", "p max = " + F(R.pmax, 2) + " MPa ≤ fjd = " + F(R.fjd, 1) + " MPa"),
                _lig("Aire d'appui", ("toute la platine" if R.plein else "bande c = tp·√(fy/(3·fjd·γM0)) = " + F(R.c, 0) + " mm autour des parois")),
                _lig("Bande", cs["nom"].replace("élément ", "É") + " : p au nu = " + F(cs["p0"], 2) + " MPa, console de " + F(cs["L"], 0) + " mm"),
                _lig("Moment pris", "mEd = ∫p·t dt = " + F(cs["m_Ed"], 2) + " kNm/m", True),
                _lig("Résistance élastique", "mRd = tp²·fy/(6·γM0) = " + F(cs["m_Rd"], 2) + " kNm/m → " + (pct(R.ck["pc"].eta, 0) if R.ck["pc"].active else "—")),
                _lig("Épaisseur pour ce moment", "tc = √(6·mEd·γM0/fy) = " + F(cs["t"], 1) + " mm (à tp donné) ; balayage : " + mm(P["t_c"]) + " mm")]
        st.markdown(tableau_html(["", ""], rows, largeurs=["38%", None]), unsafe_allow_html=True)
        st.markdown("**Trous — pression diamétrale (Tab. 3.4)**")
        pb = R.pire_appui
        if pb and pb["Fv"] > 1e-9:
            rows = [_lig("Boulon le plus cisaillé", f"n° {pb['i'] + 1} : Fv = {F(pb['Fv'], 1)} kN (V partagé, torsion, frottement déduit)"),
                    _lig("Pinces et entraxes dans le sens de l'effort", f"e1 = {F(pb['e1'], 0)}, p1 = {('∞' if pb['p1'] == float('inf') or pb['p1'] > 1e8 else F(pb['p1'], 0))}, e2 = {F(pb['e2'], 0)}, "
                         f"p2 = {('∞' if pb['p2'] == float('inf') or pb['p2'] > 1e8 else F(pb['p2'], 0))} mm"),
                    _lig("Fb,Rd = k1·αb·fu·d·tp/γM2", f"αb = {F(pb['ab'], 2)}, k1 = {F(pb['k1'], 2)} → {F(pb['Fb'], 1)} kN → tb = {F(pb['t_b'], 1)} mm", True)]
        else:
            rows = [_lig("Cisaillement par boulon", "nul (frottement Cf,d·N = " + F(R.Ff_Rd / 1e3, 0) + " kN ≥ V) : pas de pression diamétrale")]
        st.markdown(tableau_html(["", ""], rows, largeurs=["38%", None]), unsafe_allow_html=True)
    st.caption("Chaque épaisseur « balayage » est la plus petite qui satisfait son critère, la répartition étant "
               "recalculée à chaque épaisseur essayée ; « à tp donné » : l'épaisseur que demande le moment calculé "
               "avec l'épaisseur retenue.")


def _parametres_retenus(R):
    u = R.u
    g = [["Mode", "Calcul manuel — critère " + R.plaque["critere"] + " ; appui : " + ("toute la platine" if R.plein else "bande c")],
         ["Platine", js_str(N(u.bp_u)) + " × " + js_str(N(u.lp_u)) + " × " + js_str(N(u.tp_u)) + " mm " + str(u.nu_pl)
          + " – fy = " + F(R.fy_p, 0) + " MPa (épaisseur), fu = " + F(R.fu_p, 0)],
         ["Éléments soudés", " + ".join(f"{EL.nom(e)} en ({F(e['x'], 0)} ; {F(e['y'], 0)})" + (" tourné 90°" if e["rot"] else "") for e in R.elems) or "—"],
         ["Raidisseurs", ", ".join(f"({F(r['x'], 0)} ; {F(r['y'], 0)}) {r['dir']} L {F(r['L'], 0)} t {F(r['t'], 0)}" for r in R.raid) or "—"],
         ["Boulons", js_str(R.n_t) + " × " + str(u.ancr) + " " + str(u.classe) + " – d0 = " + F(R.d0, 0) + " mm, As = " + F(R.As, 0)
          + " mm² – " + ", ".join(f"({F(t['x'], 0)} ; {F(t['y'], 0)})" for t in R.tiges)],
         ["Béton", "fck = " + F(R.fck, 0) + " MPa, fcd = " + F(R.fcd, 1) + " MPa, fjd = βj·fcd·√(Ac1/Ac0) = " + F(R.fjd, 1) + " MPa"],
         ["Bras : cordon déduit", "0,8·a·√2 = " + F(R.cord, 1) + " mm"],
         ["Raideurs de la répartition", "kc = " + F(R.k_c, 1) + " N/mm³ ; kb = " + F(R.k_b / 1e3, 0) + " kN/mm (" + ("avec" if R.levier else "sans") + " effet de levier)"],
         ["Efforts au point (" + F(R.xN, 0) + " ; " + F(R.yN, 0) + ")", "Mx = " + F(R.Mx_o, 1) + " kNm, My = " + F(R.My_o, 1) + " kNm à l'axe de la platine"],
         ["Longueur d'allongement Lb = 8d + tg + tp + 0,6d", F(R.Lb, 0) + " mm"],
         ["Résistance d'un boulon (acier)", "Ft,Rd = " + F(R.Ft_EC3 / 1e3, 1) + " kN ; NRd,s = " + F(R.N_Rd_s / 1e3, 1) + " kN ; F2,vb,Rd = " + F(R.F2vb / 1e3, 1) + " kN"]]
    with st.expander("Paramètres retenus"):
        st.markdown(tableau_md(["Paramètre", "Valeur"], g))


def _formules(R):
    with st.expander("Formules et substitutions numériques"):
        for grp in GROUPES_VERIF:
            on = [c for c in R.checks if c.grp == grp and c.active]
            if not on:
                continue
            st.markdown(f"**{grp}**")
            for c in on:
                corps = "<br>".join(ER._esc(l) for l in formules.textes(R, c))
                st.markdown(
                    f"<div style='margin:2px 0 8px 0;padding:4px 10px;border-left:3px solid "
                    f"{'#2E7D46' if c.ok else '#9C3341'};font-size:0.88em;'>"
                    f"<b>{ER._esc(synthese.court(c))}</b>{badge_nature(c.nat)} "
                    f"<span style='color:#6E7480;'>{ER._esc(c.ref)}</span><br>"
                    f"<span style='font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:0.95em;'>{corps}</span></div>",
                    unsafe_allow_html=True)


def _table_simple(R, cle):
    ls = synthese.lignes(R, cle)
    if not ls:
        return False
    st.markdown(f"##### {synthese.titre_table(cle, R)}")
    lignes = [[Html(f"{l['lab']}{badge_nature(l['nat'])}"), l["Ed"], l["Rd"] + (" " + l["unit"] if l["unit"] else ""),
               taux_html(l["eta"], l["ok"], l["c"] is R.gov), l["ref"]] for l in ls]
    st.markdown(tableau_html(["Vérification", "Ed", "Rd", "η", "Réf."], lignes, droite=(1, 2, 3),
                             largeurs=["34%", "12%", "14%", "9%", None]), unsafe_allow_html=True)
    return True


def onglet_verifications(R):
    bloc_main(R)
    st.markdown(badge_nature("EC") + " formule directe de l'Eurocode " + badge_nature("EC + INT")
                + " Eurocode et interprétation nommée " + badge_nature("INT")
                + " modèle d'ingénieur (sections au nu, bandes, angles) · " + LEGENDE_REFERENCES, unsafe_allow_html=True)
    for cle in synthese.ORDRE:
        _table_simple(R, cle)
    st.caption("Non vérifiés ici : cordons, raidisseurs eux-mêmes, poteau, béton d'ancrage (cône, arrachement, bord, "
               "pry-out) — logiciel d'ancrage ou vérification complète.")
    off = [c for c in R.checks if not c.active]
    if off:
        st.caption("Sans objet dans cette configuration : " + " ; ".join(synthese.court(c) for c in off) + ".")
    _formules(R)
    ER._tableau_33(R)
    _parametres_retenus(R)


def onglet_efforts(R):
    st.caption("Les efforts des tiges viennent de la platine rigide (lit de ressorts sans traction, tiges tendues) — "
               "le modèle des logiciels d'ancrage : ce sont ces valeurs que l'on compare à Spit ou Hilti. "
               + ("Appui sur toute la platine, comme eux." if R.plein else "Appui sur la bande c (EN 1993-1-8) : un peu plus "
                  "de pression et de traction que sur toute la platine."))
    c1, c2 = st.columns([1.2, 1], gap="medium")
    with c1:
        st.markdown("##### Boulons : position, effort, appui")
        lignes = []
        for t in R.tiges:
            tendu = t["T"] > 1e-6
            lignes.append([js_str(t["i"] + 1), F(t["x"], 0) + " ; " + F(t["y"], 0), F(t["T"], 1) if tendu else "0",
                           taux_html(t["T"] / (R.Ft_anc / 1e3), t["T"] <= R.Ft_anc / 1e3 + 1e-9) if tendu else "—",
                           F(t["V_ec3"], 1) + " / " + F(t["V_en"], 1),
                           t["appui"]["face"].replace("élément ", "É") if tendu else "—", F(t["m"], 0) if tendu else "—"])
        st.markdown(tableau_html(["n°", "x ; y (mm)", "T (kN)", "T/Ft,Rd", "V EC3 / brut (kN)", "Appui", "m (mm)"], lignes,
                                 droite=(2, 3, 4, 6)), unsafe_allow_html=True)
        st.caption("V EC3 : frottement Cf,d·N déduit ; brut : sans frottement (valeur des logiciels d'ancrage).")
    with c2:
        st.markdown("##### Répartition")
        g = [["Résultante comprimée C", F(R.C, 1) + " kN en (" + F(R.xC, 0) + " ; " + F(R.yC, 0) + ") mm"],
             ["Pression maximale", F(R.pmax, 2) + " MPa (fjd = " + F(R.fjd, 1) + ")"],
             ["Zone comprimée", F(R.A_c / 100, 0) + " cm² sur " + F(R.A_eff / 100, 0) + " cm² d'appui"],
             ["Traction totale ΣT", F(R.T_tot, 1) + " kN (T max = " + F(R.T_max, 1) + " kN)"],
             ["Raideurs", "kc = " + F(R.k_c, 1) + " N/mm³ ; kb = " + F(R.k_b / 1e3, 0) + " kN/mm"],
             ["Torsion autour du centre du groupe", F(R.T_G, 2) + " kNm (centre en " + F(R.gx, 0) + " ; " + F(R.gy, 0) + ")"]]
        st.markdown(tableau_md(["Grandeur", "Valeur"], g))


def onglet_note(R, u):
    c1, c2 = st.columns([1.6, 2.6], vertical_alignment="center")
    complet = c1.checkbox("Texte : toutes les vérifications", key="ppb_ui_txt_complet")
    t = texte.construire_texte(R, complet)
    from ..texte import nom_fichier
    with c1:
        st.download_button("⬇️ Exporter le texte (.txt)", data=("﻿" + t).encode("utf-8"),
                           file_name=nom_fichier(u, ".txt"), mime="text/plain;charset=utf-8",
                           use_container_width=True, key="ppb_btn_txt")
    c2.caption("Note de calcul (2 pages, A4 paysage — le calcul à la main + le plan de principe à l'échelle) : "
               "bouton « 📄 Générer PDF » de la barre d'outils.")
    st.code(t, language=None)


def onglet_benchmark(ecrire):
    from .benchmark import run_bench_manuel, CAS
    B = run_bench_manuel()
    from acier.assemblages.ui_commun import bloc_statut
    bloc_statut("Benchmark du calcul manuel : " + B["statut"],
                ["Chaque cas est recalculé à la main dans tests/test_platine_manuelle.py (formule par formule) et rejoué ici."],
                "ok" if B["ok"] else "nok")
    for i, c in enumerate(B["cas"]):
        st.markdown(f"#### {c['id']} – {c['titre']}")
        k1, k2 = st.columns([4, 1.4], vertical_alignment="center")
        k1.caption(c["data"])
        with k2:
            st.button("Charger ces données dans l'outil", key=f"ppb_btn_bm_m{i}", use_container_width=True,
                      on_click=_charger, args=(CAS[i]["inp"], ecrire))
        # Un écart qui s'afficherait « ±0,000 % » perd son signe (pas de « -0,000 % »).
        lignes = [[r["lab"], F(r["main"], 3) + ("" if r["unit"] == "-" else " " + r["unit"]), F(r["val"], 3),
                   F((r["ec"] if abs(r["ec"]) >= 5e-6 else 0.0) * 100, 3) + " %", "OK" if r["ok"] else "À CONTRÔLER"]
                  for r in c["rows"]]
        st.markdown(tableau_md(["Grandeur", "Manuel", "Outil", "Écart", ""], lignes, droite=(1, 2, 3)))


def _charger(inp, ecrire):
    from ..entrees import defaults
    d = defaults()
    d.update(inp)
    ecrire(d)
    st.session_state["ppb_ui_alerte"] = None
    st.session_state["ppb_ui_onglet"] = ONGLETS[0]
    st.session_state["_ppb_toast"] = "Données chargées : voir Vérifications"


def onglets(R, u, ecrire):
    st.session_state["_ppb_R"] = R
    seg = getattr(st, "segmented_control", None)
    if seg:
        seg("Onglet", list(ONGLETS), key="ppb_ui_onglet", label_visibility="collapsed")
    else:
        st.radio("Onglet", list(ONGLETS), key="ppb_ui_onglet", horizontal=True, label_visibility="collapsed")
    onglet = st.session_state.get("ppb_ui_onglet") or ONGLETS[0]
    if onglet not in ONGLETS:
        onglet = ONGLETS[0]
    if onglet == ONGLETS[1]:
        onglet_efforts(R)
    elif onglet == ONGLETS[2]:
        onglet_note(R, u)
    elif onglet == ONGLETS[3]:
        onglet_benchmark(ecrire)
    elif onglet == ONGLETS[4]:
        st.markdown(METHODE)
    else:
        onglet_verifications(R)

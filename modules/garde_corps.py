# -*- coding: utf-8 -*-
"""Écran Streamlit du module « Garde-corps » (version 2).

Le moteur, l'état, les dessins et la note vivent dans ``acier/garde_corps`` (sans
Streamlit : ``tests/test_garde_corps_moteur.py`` le garde) ; ce fichier n'est que la vue.

Organisation : barre d'outils du cas ; le SWITCH de méthode — « Vérification à l'ELS » par
défaut | « Eurocode (ELU) » ; à gauche les dessins de dessinateur (élévation, coupe A-A, plan,
platine en plan, coupe sur la fixation, section) ; à droite la carte RÉSULTAT puis cinq
onglets à libellé fixe : Garde-corps (le TYPE en menu déroulant, la forme en plan, les
tronçons, H, le remplissage), Charges (les quatre charges NOMMÉES, en cas distincts ; le
cumul en option), Sections (pour chaque élément : la section, et le TABLEAU des profilés
compatibles — un clic sur une ligne retient la section), Ancrage (platine, soudure, chevilles
ou boulons, console au mur), Détail (efforts, formules, paramètres, identification, méthode).
Une seule source : l'état ``st.session_state["gc"]`` (``acier/garde_corps/etat.py``) ; les
widgets (clés ``gc_w_``) en sont des vues, réinitialisées dès que l'état change par ailleurs
(``modules/liaison.py``). Le cas par défaut est un garde-corps complet et conforme : zéro
clic pour un premier résultat, un clic pour la note.
"""
import copy
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from acier.garde_corps import bibliotheque as BI
from acier.garde_corps import charges as CH
from acier.garde_corps import etat as ET
from acier.garde_corps import fixation as FX
from acier.garde_corps import formules as FO
from acier.garde_corps import geometrie as GE
from acier.garde_corps import membres as ME
from acier.garde_corps import methode as MT
from acier.garde_corps import note as NT
from acier.garde_corps import schemas as SC
from acier.garde_corps import types as TY
from acier.profiles import familles as PF
from acier.soudures import schemas as SCS
from modules import liaison as LI

P = "gc_"
REGISTRE: LI.Registre = {}
ETAPES = ["1 · Garde-corps", "2 · Charges", "3 · Sections", "4 · Ancrage", "5 · Détail"]
MODES = {ME.ELS: "Vérification à l'ELS", ME.EUROCODE: "Eurocode (ELU)"}
LIBRE = "— section libre (dimensions) —"
DIMS_LIBRES = {"TUBE": {"d": 48.3, "t": 3.2}, "CAISSON": {"h": 60.0, "b": 40.0, "t": 3.0}, "PLAT": {"h": 50.0, "b": 10.0}, "ROND": {"d": 12.0}}
# Les familles proposées dans les tableaux de profilés, et celles cochées par défaut pour chaque rôle (RHS pour les montants et la main courante)
FAMILLES_LIB = {"RHS": "RHS", "SHS": "SHS", "CHS": "CHS", "plats": "plats", "ronds": "ronds"}
FAMILLES_DEFAUT = {"montant": ["RHS"], "lisse": ["RHS"], "barreau": ["plats", "ronds"], "lisse_inter": ["RHS"], "console": ["plats"]}
GROUPES_SECTIONS = {"montant": ["creux", "froid", "plats", "ronds"], "lisse": ["creux", "froid", "plats", "ronds"], "barreau": ["plats", "ronds", "creux", "froid"],
                    "lisse_inter": ["creux", "froid", "plats", "ronds"], "console": ["plats", "creux", "froid", "ronds"]}
AIDE_ROLE = {"montant": "La dimension h est dans le sens de la charge horizontale (axe fort) ; un montant d'angle rectangulaire est vérifié dans les deux orientations.",
             "lisse": "h vertical : l'axe fort porte q_v, l'axe faible q_h.", "barreau": "h dans le sens de la charge (perpendiculaire au garde-corps).",
             "lisse_inter": "h dans le sens de la charge (perpendiculaire au garde-corps), portée = l'entraxe des montants.",
             "console": "Plat vertical (h vertical) soudé sur la platine au mur ; longueur e dans l'onglet Ancrage."}
CSS = """<style>
.gc-res{border:1px solid #e5e7eb;border-radius:10px;padding:12px 14px;background:#fff;margin-bottom:8px}
.gc-res .t{font-size:11px;letter-spacing:.08em;color:#6b7280;font-weight:600}
.gc-pill{float:right;font-size:12px;border-radius:999px;padding:3px 10px;font-weight:600}
.gc-pill.nok{background:#fee2e2;color:#b91c1c}.gc-pill.ok{background:#dcfce7;color:#15803d}.gc-pill.att{background:#ffedd5;color:#c2410c}.gc-pill.ref{background:#f3f4f6;color:#374151}
.gc-eta{font-size:26px;font-weight:700;margin:2px 0 0;line-height:1.1}
.gc-phrase{font-size:13.5px;margin:4px 0 6px}
.gc-row{display:grid;grid-template-columns:190px 1fr 56px;align-items:center;gap:10px;font-size:12.5px;margin:5px 0}
.gc-row.sel span:first-child{font-weight:700}
.gc-trk{height:6px;background:#f1f3f5;border-radius:3px;overflow:hidden}.gc-trk b{display:block;height:100%}
.gc-v{text-align:right;font-variant-numeric:tabular-nums;font-weight:600}
.gc-note{font-size:12px;color:#6b7280;margin-top:6px}
.gc-tab{width:100%;border-collapse:collapse;font-size:12.5px}.gc-tab th{text-align:left;font-size:10.5px;letter-spacing:.05em;color:#6b7280;padding:4px 6px;border-bottom:1px solid #e5e7eb}
.gc-tab td{padding:4px 6px;border-bottom:1px solid #f3f4f6;vertical-align:top}.gc-tab td.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.gc-tab tr.g td{font-weight:700;background:#fff7ed}.gc-tab tr.cur td{font-weight:700;background:#eff6ff}
.gc-r{color:#b91c1c;font-weight:700}.gc-o{color:#c2410c;font-weight:700}.gc-g{color:#15803d;font-weight:700}.gc-i{color:#6b7280}
.gc-leg{font-size:11.5px;color:#6b7280}
.gc-switch{font-size:12.5px;color:#374151;padding-top:6px}
.gc-ch{display:grid;grid-template-columns:34px 1fr;gap:8px;align-items:start;font-size:13px;margin-top:10px}
.gc-ch .n{font-size:18px;font-weight:700;color:#33415c;line-height:1.1}.gc-ch .s{font-weight:700}.gc-ch .o{font-size:11.5px;color:#6b7280}
.gc-role{font-size:14px;font-weight:700;margin:14px 0 2px;padding-top:8px;border-top:1px solid #e5e7eb}
.gc-elt{font-size:12.5px;color:#374151}
.gc-opt{font-size:13px;color:#1f2937;padding:4px 0}
</style>"""


# ---------------------------------------------------------------- état et vues des widgets
def _etat() -> Dict:
    if "gc" not in st.session_state:
        st.session_state["gc"] = ET.nouvel_etat()
    return st.session_state["gc"]


def _reset_widgets():
    LI.synchroniser(P, REGISTRE, ET.cas_actif(_etat()))


def _chemin(d: Dict, chemin):
    for c in chemin[:-1]:
        d = d[c]
    return d, chemin[-1]


def _ecrire(chemin, cle, conv=None):
    v = st.session_state.get(cle)
    if conv is not None:
        v = conv(v)
    d, k = _chemin(ET.cas_actif(_etat()), chemin)
    d[k] = v


def _cle(chemin) -> str:
    return P + "w_" + "_".join(str(c) for c in chemin)


def _num(label, chemin, cas, step=1.0, fmt="%.1f", min_value=None, max_value=None, help=None, width=None, disabled=False):
    d, k = _chemin(cas, chemin)
    val = d.get(k)
    cle = _cle(chemin)
    entier = isinstance(step, int)
    conv = (lambda v: None if v is None else (int(v) if entier else float(v)))
    v0 = (int(val) if entier else float(val)) if val is not None else (0 if entier else 0.0)
    LI.enregistrer(REGISTRE, cle, chemin, (lambda v: (0 if entier else 0.0) if v is None else (int(v) if entier else float(v))))
    kw = {}
    if min_value is not None:
        kw["min_value"] = min_value
    if max_value is not None:
        kw["max_value"] = max_value
    if width is not None:
        kw["width"] = width
    st.number_input(label, value=v0, step=step, format="%d" if entier else fmt, key=cle, on_change=_ecrire, args=(chemin, cle, conv), help=help,
                    disabled=disabled, **kw)


def _texte_w(label, chemin, cas, help=None, width=None):
    d, k = _chemin(cas, chemin)
    cle = _cle(chemin)
    LI.enregistrer(REGISTRE, cle, chemin, lambda v: v or "")
    kw = {"width": width} if width is not None else {}
    st.text_input(label, value=d.get(k) or "", key=cle, on_change=_ecrire, args=(chemin, cle, None), help=help, **kw)


def _choix(label, chemin, cas, options, affichage=None, help=None, segmente=False, width=None, on_change=None):
    d, k = _chemin(cas, chemin)
    cle = _cle(chemin)
    cur = d.get(k)
    if cur not in options:
        cur = options[0]
    LI.enregistrer(REGISTRE, cle, chemin, None, options)
    fmt = (lambda o: affichage.get(o, str(o))) if affichage else (lambda o: str(o))
    kw = {"width": width} if width is not None else {}

    def _on():
        _ecrire(chemin, cle, None)
        if on_change:
            on_change()
    if segmente:
        st.segmented_control(label, options, default=cur, format_func=fmt, key=cle, on_change=_on, help=help, selection_mode="single", **kw)
    else:
        st.selectbox(label, options, index=options.index(cur), format_func=fmt, key=cle, on_change=_on, help=help, **kw)
    return d.get(k)


def _case(label, chemin, cas, help=None):
    d, k = _chemin(cas, chemin)
    cle = _cle(chemin)
    LI.enregistrer(REGISTRE, cle, chemin, bool)
    st.checkbox(label, value=bool(d.get(k)), key=cle, on_change=_ecrire, args=(chemin, cle, bool), help=help)


def _changer(toast: str = ""):
    LI.demander(P)
    if toast:
        st.toast(toast)
    st.rerun()


def _fmt(x, nd=1):
    if x is None:
        return "—"
    return f"{x:,.{nd}f}".replace(",", " ").replace(".", ",")


def _cls(t):
    return "gc-r" if t > 1 else ("gc-o" if t > 0.85 else "gc-g")


def _noms_sections(groupes: Optional[List[str]]) -> List[str]:
    """Les noms des sections standard des groupes, dans l'ordre du catalogue (famille, taille)."""
    return [s["nom"] for s in BI.standards().values() if groupes is None or BI.groupe_de(s) in groupes]


def _section_w(label, chemin, cas, groupes=None, help=None, width=None):
    """Le choix d'une section : la liste des sections standard des groupes (recherche en tapant),
    ou « section libre » avec sa forme et ses dimensions. L'état garde la spécification."""
    d, k = _chemin(cas, chemin)
    spec = d.get(k) or {}
    cle = _cle(chemin)
    noms = _noms_sections(groupes)
    cur = spec.get("nom")
    options = ([cur] if cur and cur not in noms else []) + noms + [LIBRE]
    valeur = cur if cur else LIBRE
    LI.enregistrer(REGISTRE, cle, chemin, lambda sp: (sp or {}).get("nom") or LIBRE, options)

    def _on():
        v = st.session_state[cle]
        dd, kk = _chemin(ET.cas_actif(_etat()), chemin)
        if v == LIBRE:
            if (dd.get(kk) or {}).get("nom") or not dd.get(kk):
                dd[kk] = {"famille": "TUBE", "dims": dict(DIMS_LIBRES["TUBE"])}
        else:
            dd[kk] = {"nom": v}
    kw = {"width": width} if width is not None else {}
    st.selectbox(label, options, index=options.index(valeur), key=cle, on_change=_on, help=help, **kw)
    if not spec.get("nom"):
        cle_f = _cle(chemin + ("famille",))
        fam = spec.get("famille") if spec.get("famille") in BI.NON_STANDARD else "TUBE"
        LI.enregistrer(REGISTRE, cle_f, chemin + ("famille",), None, list(BI.NON_STANDARD))

        def _fam():
            dd, kk = _chemin(ET.cas_actif(_etat()), chemin)
            f = st.session_state[cle_f]
            dd[kk] = {"famille": f, "dims": dict(DIMS_LIBRES[f])}
            _reset_widgets()
        with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=P + "c_libre_" + "_".join(str(c) for c in chemin)):
            st.selectbox("Forme", list(BI.NON_STANDARD), index=list(BI.NON_STANDARD).index(fam), format_func=lambda f: BI.NON_STANDARD[f][0], key=cle_f,
                         on_change=_fam, width=180)
            for dim in BI.NON_STANDARD[fam][1]:
                _num(PF.LIBELLE_DIM[dim], chemin + ("dims", dim), cas, 0.5, "%.1f", 0.5, width=120)
        st.caption("Section non standard : ses propriétés sont calculées de ses dimensions par la même fonction que le catalogue "
                   "(h dans le sens de l'effort = axe fort).")


def _entraxes_w(i: int, cas):
    """Les entraxes saisis d'un tronçon : un texte « 1200 900 » → la liste (vide : répartition régulière)."""
    chemin = ("geometrie", "troncons", i, "entraxes")
    d, k = _chemin(cas, chemin)
    cle = _cle(chemin)
    LI.enregistrer(REGISTRE, cle, chemin, lambda lst: " ".join(f"{float(e):g}" for e in (lst or [])))

    def _on():
        txt = (st.session_state.get(cle) or "").replace(";", " ").replace(",", " ")
        vals = []
        for mot in txt.split():
            try:
                v = float(mot)
            except ValueError:
                continue
            if v > 0:
                vals.append(v)
        dd, kk = _chemin(ET.cas_actif(_etat()), chemin)
        dd[kk] = vals
    st.text_input("Entraxes saisis (mm, facultatif)", value=" ".join(f"{float(e):g}" for e in (d.get(k) or [])), key=cle, on_change=_on, width=220,
                  help="Par exemple « 1200 900 » : un poteau central à 1,20 m du premier et 0,90 m du dernier. Vide : la longueur est répartie "
                       "régulièrement à l'entraxe maximal.")


# ---------------------------------------------------------------- en-tête, barre d'outils, switch
def _entete(etat: Dict, cas: Dict):
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("## Garde-corps")
    st.caption("Dimensionnement d'un garde-corps acier — type en menu (barreaudage, lisses, panneau, main courante seule, main courante murale), "
               "forme en plan droite, en L ou en U, entraxes libres, quatre charges nommées (EN 1991-1-1 Tab. 6.12, NBN B 03-004) en cas distincts, "
               "montant principal, main courante sur deux appuis ou continue, barreaux, ancrage (soudure, chevilles ou boulons, platine sur la dalle ou "
               "en nez de dalle) · vérification à l'ELS par défaut, Eurocode en option · version 3.0")
    with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="center", key=P + "barre"):
        if st.button("🏠 Accueil", key=P + "btn_home", width="content"):
            st.session_state.retour_accueil_demande = True
            st.rerun()
        noms = [c["nom"] for c in etat["cas"]]

        def _sel():
            etat["actif"] = noms.index(st.session_state[P + "sel_cas"])
            _reset_widgets()
        st.session_state[P + "sel_cas"] = noms[etat["actif"]]
        st.selectbox("Cas", noms, key=P + "sel_cas", on_change=_sel, label_visibility="collapsed", width=240)
        if st.button("＋ Cas", key=P + "btn_new", width="content", help="Nouveau garde-corps (le cas par défaut)"):
            n = len(etat["cas"]) + 1
            c = ET.nouveau_cas(f"Garde-corps {n}")
            c["methode"] = copy.deepcopy(cas["methode"])
            etat["cas"].append(c)
            etat["actif"] = len(etat["cas"]) - 1
            _changer("Nouveau cas")
        if st.button("⧉", key=P + "btn_copy", width="content", help="Copier ce cas"):
            d = copy.deepcopy(cas)
            d["nom"] = f"{cas['nom']} (copie)"
            etat["cas"].append(d)
            etat["actif"] = len(etat["cas"]) - 1
            _changer("Cas copié")
        if len(etat["cas"]) > 1 and st.button("🗑️", key=P + "btn_del", width="content", help="Supprimer ce cas"):
            etat["cas"].pop(etat["actif"])
            etat["actif"] = max(0, etat["actif"] - 1)
            _changer("Cas supprimé")
        st.download_button("💾 Enregistrer", data=ET.vers_json(etat).encode("utf-8"), file_name=f"garde_corps_{cas['nom']}.json",
                           mime="application/json", key=P + "btn_save", width="content", help="Tous les cas (JSON)")
        if st.button("📂 Ouvrir", key=P + "btn_open", width="content"):
            st.session_state[P + "ouvrir"] = not st.session_state.get(P + "ouvrir", False)
        if st.button("📄 Note", key=P + "btn_note", width="content", help="Note de calcul PDF en deux pages : hypothèses, dessins, montant, main courante, "
                                                                        "remplissage, ancrage avec les formules développées ; détails, soudure, méthode"):
            ident = etat.get("identification") or {}
            st.session_state[P + "note"] = (cas["nom"], NT.generer_note(etat, cas, {"nom_projet": ident.get("projet", ""), "partie": ident.get("partie", ""),
                                                                                   "indice": ident.get("indice", "0")}))
        _texte_w("Nom du cas", ("nom",), cas, width=260)
    note = st.session_state.get(P + "note")
    if note and note[0] == cas["nom"]:
        st.download_button("⬇️ Télécharger la note (PDF, 2 pages)", data=note[1], file_name=f"note_garde_corps_{cas['nom']}.pdf",
                           mime="application/pdf", key=P + "dl_note", width="content")
    if st.session_state.get(P + "ouvrir"):
        up = st.file_uploader("Fichier JSON du module", type=["json"], key=P + "uploader", label_visibility="collapsed")
        if up is not None:
            try:
                st.session_state["gc"] = ET.depuis_json(up.read().decode("utf-8-sig"))
                st.session_state[P + "ouvrir"] = False
                _changer("Fichier chargé")
            except Exception as e:  # noqa: BLE001
                st.error(f"Fichier invalide : {e}")


def _switch(cas: Dict):
    c1, c2 = st.columns([1.25, 2], vertical_alignment="center")
    with c1:
        _choix("Méthode", ("methode", "mode"), cas, [ME.ELS, ME.EUROCODE], MODES, segmente=True)
    m = cas["methode"]
    fl = f"L/{m.get('fleche_val', 100):g}" if m.get("fleche_mode", "L/x") == "L/x" else f"{m.get('fleche_val', 0):g} mm"
    with c2:
        if m["mode"] == ME.ELS:
            lim = ME.limites(m, ET.acier(m)["fy"])
            st.markdown(f'<div class="gc-switch"><b>Charges de service</b>, sans coefficient, contraintes admissibles dérivées de la nuance — '
                        f'σ = M/W + N/A ≤ σ_adm = f_y/{lim["coef"]:g} = <b>{_fmt(lim["sigma"])} N/mm²</b>, τ = V/A_v ≤ τ_adm = σ_adm/√3 = '
                        f'<b>{_fmt(lim["tau"])} N/mm²</b>, σ_eq = √(σ² + 3τ²) ≤ σ_adm, flèche ≤ <b>{fl}</b> ; soudure '
                        f'à l\'ELS (k = {m.get("k", 100):g} kg/(mm·cm)), chevilles et boulons à leur résistance de calcul / γ_F = {m.get("gamma_F", 1.5):g} '
                        f'(paramètres dans l\'onglet Détail).</div>'.replace("f_y/1.5", "f_y/1,5"), unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="gc-switch"><b>Efforts de calcul (ELU)</b> : charges × γ_Q = {m.get("gamma_Q", 1.5):g}, σ ≤ f_y/γ_M0 (élastique, W_el), '
                        f'τ ≤ f_y/(√3·γ_M0), flèche ≤ {fl} sous charges de service ; soudure EN 1993-1-8 directionnelle, boulons Tab. 3.4, chevilles à '
                        f'leurs valeurs de calcul ETA (paramètres dans l\'onglet Détail).</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- colonne de gauche : dessins
def _colonne_dessin(cas: Dict, res: Dict):
    vues = {k: (t, d) for k, t, d in SC.vues(res)}
    if not vues:
        with st.container(key=P + "dessin"):
            st.warning("Calcul impossible — " + (res.get("messages") or ["données manquantes"])[0])
        return

    def bloc(cle: str, ident: str, legende: str = ""):
        if cle not in vues:
            return
        t, d = vues[cle]
        with st.container(key=P + f"dessin_{ident}"):
            st.markdown(f'<div class="gc-leg" style="margin:2px 0"><b>{t.upper()}</b>{" · " + legende if legende else ""}</div>' + SC.svg(d, "gc_v_" + ident),
                        unsafe_allow_html=True)
    with st.container(key=P + "dessin"):
        t, d = vues["elevation"]
        st.markdown(f'<div class="gc-leg" style="margin:2px 0"><b>{t.upper()}</b> · {res["type_libelle"]} · {res["geometrie"]["phrase"]}</div>'
                    + SC.svg(d, "gc_v0"), unsafe_allow_html=True)
    c1, c2 = st.columns([1.25, 1.75], gap="small")
    with c1:
        bloc("coupe", "coupe")
    with c2:
        bloc("plan", "plan")
        bloc("section", "section", "élément principal")
    c3, c4 = st.columns([1, 1.1], gap="small")
    with c3:
        bloc("platine", "platine", "vue de face (platine latérale, console au mur) ou de dessus (platine sur la dalle)")
    with c4:
        bloc("pied", "pied", "coupe à grande échelle, efforts du cas gouvernant")
    st.markdown('<div class="gc-leg">Dessins à l\'échelle, en mm : pièces au trait fort, sections coupées hachurées, cordons en ocre à leur taille a·√2, '
                'chevilles en gris (tige cachée en traits interrompus), ◉ l\'élément gouvernant et son taux, flèches rouges dans le sens des charges '
                '(⊙ vers soi), A-A le plan de la coupe.</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- colonne de droite : résultat
def _carte_resultat(cas: Dict, res: Dict):
    stat = res["statut"]
    pill = {ET.OK: ("ok", "● Conforme"), ET.ATTENTION: ("att", "● Conforme, attention"), ET.NOK: ("nok", "● Non conforme"),
            ET.REFUS: ("ref", "● Calcul impossible")}[stat]
    g = res.get("gouvernant")
    if stat == ET.REFUS or not g:
        eta = "—"
        phrase = (res["messages"] or ["Données manquantes"])[0]
    else:
        t = res["taux_max"]
        col = "#b91c1c" if t > 1 else ("#c2410c" if t > 0.85 else "#15803d")
        eta = f'<span style="color:{col}">η = {t * 100:.0f} %</span>'
        crit = ME.critere_texte(g["critere"]) if g["cle"] not in ("soudure", "ancrage", "platine") else g["critere"]
        cas_g = (next((e for e in res["elements"] if e["cle"] == g["cle"]), {}).get("detail") or {}).get("cas_charge")
        phrase = f"<b>{g['libelle']}</b> — {crit}" + (f" · {CH.CAS_LIBELLES.get(cas_g, cas_g)}" if cas_g else "")
    m = cas["methode"]
    ligne_m = f"{res['type_libelle']} · {MODES[m['mode']]} · {CH.phrase_cumul(res['charges'])}"
    barres = ""
    for e in res.get("elements", []):
        t = e["taux"]
        col = "#b91c1c" if t > 1 else ("#c2410c" if t > 0.85 else "#2563eb")
        sel = g and e["cle"] == g["cle"]
        lab = e["libelle"].split(" · ")[0]
        barres += (f'<div class="gc-row{" sel" if sel else ""}"><span>{lab}{" ◀" if sel else ""}</span>'
                   f'<div class="gc-trk"><b style="width:{min(t, 1) * 100:.0f}%;background:{col}"></b></div>'
                   f'<span class="gc-v" style="color:{col}">{t * 100:.0f} %</span></div>')
    if res.get("masse"):
        barres += f'<div class="gc-note">Masse indicative de l\'acier : {_fmt(res["masse"], 0)} kg · {res["geometrie"]["phrase"]}</div>'
    st.markdown(f'<div class="gc-res"><span class="gc-pill {pill[0]}">{pill[1]}</span><div class="t">RÉSULTAT</div>'
                f'<div class="gc-eta">{eta}</div><div class="gc-phrase">{phrase}</div><div class="gc-note">{ligne_m}</div>{barres}</div>',
                unsafe_allow_html=True)
    for msg in res.get("messages", []):
        if msg.startswith("Refus") or "DÉPASSÉ" in msg:
            st.warning(msg)
        else:
            st.caption("• " + msg)


# ---------------------------------------------------------------- onglet 1 : garde-corps
def _etape_garde_corps(cas: Dict, res: Dict):
    g = cas["geometrie"]

    def _type():
        c = ET.cas_actif(_etat())
        TY.appliquer_type(c, c.get("type"))
        _reset_widgets()
    typ = _choix("Type de garde-corps", ("type",), cas, list(TY.TYPES), {k: v["libelle"] for k, v in TY.TYPES.items()}, on_change=_type,
                 help="Le type fixe le système (montants verticaux sur platine, ou main courante portée par des consoles au mur) et le remplissage ; "
                      "tout le reste se règle ensuite.")
    st.caption(TY.TYPES[typ]["description"])
    with st.container(horizontal=True, wrap=True, gap="medium", vertical_alignment="bottom", key=P + "c_sys"):
        _choix("Forme en plan", ("geometrie", "forme"), cas, list(GE.FORMES), GE.FORMES, segmente=True, width="content",
               help="Un, deux ou trois tronçons droits ; un montant d'angle reçoit une demi-travée de chaque tronçon dans deux directions.")
        _num("H — main courante au-dessus de la dalle (mm)", ("geometrie", "H"), cas, 10.0, "%.0f", 300.0, width=250,
             help="Bras de levier de la charge horizontale : hauteur de l'axe de la main courante au-dessus de la platine (ou du dessus de dalle pour une "
                  "platine latérale : le montant est alors vérifié sur H + d_h). La charge s'applique au plus à 1,20 m (EN 1991-1-1 §6.4).")
        _num("Pente α du rampant (°, 0 : horizontal)", ("geometrie", "pente"), cas, 1.0, "%.0f", 0.0, max_value=60.0, width=230,
             help="Garde-corps d'escalier : les entraxes se mesurent le long de la main courante, les montants restent verticaux ; la pente incline "
                  "l'élévation et se lit dans les hypothèses, elle ne change pas les formules.")
    n = GE.N_TRONCONS.get(g.get("forme", "droit"), 1)
    for i in range(n):
        with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=P + f"c_tr{i}"):
            st.markdown(f"**Tronçon {i + 1}**")
            _num("Longueur L (mm)", ("geometrie", "troncons", i, "L"), cas, 100.0, "%.0f", 100.0, width=150)
            _num("Entraxe maxi (mm)", ("geometrie", "troncons", i, "s_max"), cas, 50.0, "%.0f", 200.0, width=150,
                 help="La longueur est répartie en travées égales ne dépassant pas cet entraxe — sauf si des entraxes sont saisis à côté.")
            _entraxes_w(i, cas)
    with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=P + "c_ext"):
        _choix("Début", ("geometrie", "debut"), cas, list(GE.EXTREMITES), GE.EXTREMITES, width=240,
               help="Un montant d'extrémité, ou la main courante ancrée dans le mur : le mur reprend la demi-travée. Les deux bouts ancrés et une "
                    "seule travée : garde-corps SANS montant, la main courante porte de mur à mur (loggia).")
        _choix("Fin", ("geometrie", "fin"), cas, list(GE.EXTREMITES), GE.EXTREMITES, width=240)
    rp = cas["remplissage"]
    rpt = rp.get("type")
    if rpt == "barreaux":
        st.markdown("**Barreaudage** — barreaux verticaux entre la lisse basse et la main courante ; leur portée se déduit de H, de h_bas et des sections")
        with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=P + "c_rp"):
            _num("Entraxe des barreaux e_b (mm)", ("remplissage", "entraxe"), cas, 5.0, "%.0f", 20.0, width=220,
                 help="Un barreau reprend la pression q_r × e_b (cas 4) ou la charge ponctuelle Q_r à mi-portée (cas 3) ; le pire gouverne.")
            _num("h_bas — axe de la lisse basse au-dessus de la platine (mm)", ("remplissage", "h_bas"), cas, 10.0, "%.0f", 0.0, width=330)
        r = res.get("remplissage") or {}
        if r.get("L"):
            st.caption(f"Portée des barreaux déduite : {_fmt(r['L'], 0)} mm (entre le dessus de la lisse basse et le dessous de la main courante) ; "
                       f"sections des barreaux et de la lisse basse dans l'onglet Sections.")
    elif rpt == "lisses":
        st.markdown("**Lisses intermédiaires** — horizontales, entre montants, sous la main courante")
        with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=P + "c_rp"):
            _num("Nombre de lisses intermédiaires", ("remplissage", "n_lisses"), cas, 1, min_value=1, width=220)
            _num("Entraxe des lisses e_l (mm)", ("remplissage", "entraxe_lisses"), cas, 10.0, "%.0f", 50.0, width=220,
                 help="Chaque lisse reprend q_r × e_l, ou Q_r à mi-portée, sur la portée entre montants.")
    elif rpt == "panneau":
        st.markdown("**Panneau** (verre, tôle, câbles) — entre la lisse basse et la main courante")
        with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=P + "c_rp"):
            _num("h_bas — axe de la lisse basse au-dessus de la platine (mm)", ("remplissage", "h_bas"), cas, 10.0, "%.0f", 0.0, width=330)
        st.caption("La résistance du panneau sous q_r et Q_r est à justifier par le fournisseur ; la note le dit.")
    else:
        st.caption("Main courante seule, sans remplissage (rampe, lisse de protection).")
    geo = res.get("geometrie") or {}
    if geo.get("montants"):
        nom = "console" if g.get("systeme") == "mur" else "montant"
        lignes = ""
        for m in geo["montants"]:
            lignes += (f"<tr><td><b>{m['nom'] if nom == 'montant' else m['nom'].replace('M', 'C')}</b></td><td class='n'>{_fmt(m['x'], 0)}</td>"
                       f"<td class='n'>{_fmt(m['trib1'], 0)}</td><td class='n'>{_fmt(m['trib2'], 0) if m['coin'] else '—'}</td><td>{'angle' if m['coin'] else ''}</td></tr>")
        st.markdown(f"<table class='gc-tab'><tr><th>{nom.capitalize()}</th><th class='n'>x (mm)</th><th class='n'>s_trib (mm)</th><th class='n'>s_trib,2 (mm)</th><th></th></tr>"
                    + lignes + "</table>", unsafe_allow_html=True)
        st.caption(f"{geo['phrase']} · portée de la main courante : l'entraxe le plus grand, {_fmt(geo['s_max'], 0)} mm.")
    elif geo.get("sans_montant"):
        st.info(f"Sans montant : la main courante porte de mur à mur sur {_fmt(geo['L_total'], 0)} mm (les deux bouts ancrés, une seule travée) ; "
                "la fixation n'a pas d'objet, les réactions aux murs sont données dans le résultat. Pour des montants, raccourcir l'entraxe maximal.")


# ---------------------------------------------------------------- onglet 2 : charges
def _ligne_charge(n: int, sym: str, nom: str, origine: str):
    st.markdown(f'<div class="gc-ch"><div class="n">{n}</div><div><span class="s">{sym}</span> — {nom}<br><span class="o">{origine}</span></div></div>',
                unsafe_allow_html=True)


def _etape_charges(cas: Dict, res: Dict):
    ch = cas["charges"]
    cats = [c[0] for c in CH.CATEGORIES]
    libs = {c[0]: (c[1] + (f" — {c[2]:g} kN/m" if c[2] is not None else "")) for c in CH.CATEGORIES}

    def _cat():
        cas["charges"].update(CH.normaliser(cas["charges"]))
        _reset_widgets()
    st.markdown("**Les quatre charges** — chacune agit seule : quatre cas de charge distincts (les charges de service d'un garde-corps ne se cumulent pas) ; "
                "le cumul de q_h et q_v est une option, du côté sûr.")
    _ligne_charge(1, "q_h", "charge linéaire horizontale sur la main courante, à la hauteur H", CH.CHARGES[0]["norme"])
    with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=P + "c_ch1"):
        _choix("Catégorie d'usage (Tab. 6.12, valeurs recommandées)", ("charges", "categorie"), cas, cats, libs, on_change=_cat, width=420,
               help="La charge linéaire horizontale sur la main courante selon l'usage ; l'annexe nationale belge peut fixer d'autres valeurs : "
                    "« valeur libre » les accepte. La note imprime la valeur retenue et son origine.")
        _num("q_h (kN/m)", ("charges", "qh"), cas, 0.1, "%.2f", 0.0, disabled=(ch.get("categorie") != "perso"), width=140)
    _ligne_charge(2, "q_v", "charge linéaire verticale sur la main courante", CH.CHARGES[1]["norme"])
    with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=P + "c_ch2"):
        _num("q_v (kN/m)", ("charges", "qv"), cas, 0.1, "%.2f", 0.0, width=140, help=CH.SOURCE_QV)
    _ligne_charge(3, "Q_r", "charge ponctuelle horizontale sur le remplissage (barreau, lisse, panneau)", CH.CHARGES[2]["norme"])
    with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=P + "c_ch3"):
        _num("Q_r (kN)", ("charges", "Qr"), cas, 0.1, "%.2f", 0.0, width=140, help=CH.SOURCE_QR_PONCT)
    _ligne_charge(4, "q_r", "pression horizontale répartie sur le remplissage", CH.CHARGES[3]["norme"])
    with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=P + "c_ch4"):
        _num("q_r (kN/m²)", ("charges", "qr"), cas, 0.1, "%.2f", 0.0, width=140, help=CH.SOURCE_QR)
    _case("Cumuler q_h et q_v (cas 1 + 2, côté sûr) — par défaut chaque charge agit seule", ("charges", "cumul"), cas,
          help="Le cumul fait agir la charge horizontale et la charge verticale ensemble sur le montant, la main courante, la soudure et la platine. "
               "Les charges 3 et 4 restent toujours des cas séparés.")
    with st.expander("Options : charge P par montant saisie directement, charge ponctuelle Q sur la main courante", expanded=(ch.get("mode") == "P" or ch.get("Q", 0) > 0)):
        with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=P + "c_ch5"):
            cleP = P + "w_charges_mode"
            LI.enregistrer(REGISTRE, cleP, ("charges", "mode"), lambda v: v == "P")

            def _mode():
                cas["charges"]["mode"] = "P" if st.session_state[cleP] else "q"
            st.checkbox("Saisir directement la charge ponctuelle P qui revient à chaque montant", value=(ch.get("mode") == "P"), key=cleP, on_change=_mode,
                        help="À la place de q_h × longueur tributaire ; la main courante reçoit alors l'équivalent P / entraxe.")
            if ch.get("mode") == "P":
                _num("P (kN)", ("charges", "P"), cas, 0.1, "%.2f", 0.0, width=130)
        with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=P + "c_ch6"):
            _num("Q — charge ponctuelle horizontale sur la main courante (kN, 0 : sans)", ("charges", "Q"), cas, 0.1, "%.2f", 0.0, width=380,
                 help="Elle s'ajoute à q_h sur la main courante (« somme ») ou forme un cas séparé (« max »).")
            _choix("Avec q_h", ("methode", "comb"), cas, ["somme", "max"], {"somme": "somme (Q + q_h)", "max": "max (cas séparés)"}, width=200)
    st.caption(CH.phrase(res["charges"]) + " · " + res["limites"]["nature"] + ".")


# ---------------------------------------------------------------- onglet 3 : sections et tableaux de profilés
def _groupes_de(fams, froid: bool) -> List[str]:
    g = set()
    if any(f in ("RHS", "SHS", "CHS") for f in fams):
        g.add("creux")
        if froid:
            g.add("froid")
    if "plats" in fams:
        g.add("plats")
    if "ronds" in fams:
        g.add("ronds")
    return sorted(g)


def _de_famille(sec: Dict, fams) -> bool:
    fam = sec["famille"].replace("-F", "")
    if fam in ("RHS", "SHS", "CHS"):
        return fam in fams
    if fam == "PLAT":
        return "plats" in fams
    if fam == "ROND":
        return "ronds" in fams
    return True


def _chemin_role(role: str):
    return {"montant": ("montant", "section"), "lisse": ("lisse", "section"), "barreau": ("remplissage", "section"),
            "lisse_inter": ("remplissage", "section_lisses"), "console": ("fixation", "console", "section")}[role]


def _tableau_role(cas: Dict, res: Dict, role: str, masse_max: float, seuls: bool, froid: bool):
    """Le tableau des profilés compatibles pour un rôle : chaque section standard des familles
    cochées, vérifiée dans le rôle ; un CLIC sur une ligne retient la section."""
    lib = ET.ROLES[role]
    fams = st.pills("Familles", list(FAMILLES_LIB), default=FAMILLES_DEFAUT[role], format_func=lambda f: FAMILLES_LIB[f], key=P + f"lib_{role}_fam",
                    selection_mode="multi", label_visibility="collapsed")
    if not fams:
        st.caption("Choisir au moins une famille.")
        return
    cands = [c for c in ET.candidats(cas, role, _groupes_de(fams, froid), float(masse_max)) if _de_famille(c["section"], fams)]
    best = ET.plus_leger(cands)
    cur = (ET.spec_du_role(cas, role) or {}).get("nom")
    nom_best = best["section"]["nom"] if best is not None else None
    if best is not None:
        # l'OPTIMAL — la section la plus légère des familles cochées qui tient dans ce rôle — est proposé à chaque choix de profilé
        retenu = nom_best == cur
        with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="center", key=P + f"c_opt_{role}"):
            st.markdown(f'<div class="gc-opt">★ Optimal pour {lib} : <b>{nom_best}</b> · {best["section"]["masse"]:.2f} kg/m · η = {best["taux"] * 100:.0f} % '
                        f'({ME.critere_texte(best["critere"])})'.replace(".", ",").replace(nom_best.replace(".", ","), nom_best)
                        + (' · <span class="gc-ok">retenu</span>' if retenu else '') + '</div>', unsafe_allow_html=True)
            if st.button("Prendre l'optimal", key=P + f"btn_best_{role}", width="content", disabled=retenu,
                         help="Retient la section la plus légère qui tient, parmi les familles cochées et sous la masse maximale."):
                ET.appliquer_section(cas, best["section"], role)
                _changer(f"{nom_best} retenu pour {lib}")
    else:
        st.warning("Aucune section de la sélection ne tient : élargir les familles ou la masse maximale.")
    montres = [c for c in cands if not seuls or c["taux"] <= 1.0][:40]
    if not montres:
        st.caption("Aucune ligne à montrer.")
        return
    df = pd.DataFrame([{"Section": c["section"]["nom"], "dimensions (mm)": BI.dimensions_texte(c["section"]), "kg/m": round(c["masse"], 2),
                        "η σ_eq": int(round(min(c["taux_eq"], 9.99) * 100)), "η flèche": int(round(min(c["taux_fleche"], 9.99) * 100)),
                        "η": int(round(min(c["taux"], 9.99) * 100)),
                        "retenue": ("◀ " if c["section"]["nom"] == cur else "") + ("★" if c["section"]["nom"] == nom_best else "")} for c in montres])
    # largeurs en pixels : le nom complet de la section se lit, et les sept colonnes tiennent dans la colonne de droite d'un PC
    cfg = {"Section": st.column_config.TextColumn("Section", width=125), "dimensions (mm)": st.column_config.TextColumn("dimensions (mm)", width=92),
           "kg/m": st.column_config.NumberColumn("kg/m", format="%.2f", width=52),
           "η σ_eq": st.column_config.NumberColumn("η σ_eq", format="%d %%", width=58), "η flèche": st.column_config.NumberColumn("η flèche", format="%d %%", width=62),
           "η": st.column_config.ProgressColumn("η", min_value=0, max_value=150, format="%d %%", width=110),
           "retenue": st.column_config.TextColumn("", width=36)}
    ev = st.dataframe(df, hide_index=True, column_config=cfg, key=P + f"tab_{role}_{LI.generation(P)}", on_select="rerun", selection_mode="single-row",
                      height=min(300, 35 * (len(df) + 1) + 12), width="stretch")
    rows = list(ev.selection.rows) if ev is not None and hasattr(ev, "selection") else []
    if rows:
        sec = montres[rows[0]]["section"]
        if sec["nom"] != cur:
            ET.appliquer_section(cas, sec, role)
            _changer(f"{sec['nom']} retenu pour {lib}")
    st.caption(f"Un clic sur une ligne retient la section pour {lib} · {len(cands)} sections essayées par masse croissante ({len(montres)} montrées), chacune "
               f"vérifiée dans ce rôle (σ, τ, σ_eq, flèche) ; ★ l'optimal, le plus léger qui tient ; la fixation est vérifiée pour la section retenue.")


def _bloc_role(cas: Dict, res: Dict, role: str, masse_max: float, seuls: bool, froid: bool):
    lib = ET.ROLES[role]
    st.markdown(f'<div class="gc-role">{lib.capitalize()}</div>', unsafe_allow_html=True)
    with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=P + f"c_sec_{role}"):
        _section_w("Section", _chemin_role(role), cas, GROUPES_SECTIONS[role], help=AIDE_ROLE[role], width=300)
        if role == "lisse":
            _choix("Modèle", ("lisse", "appuis"), cas, list(ME.APPUIS), {"deux": "poutre sur deux appuis", "continue": "poutre continue"}, segmente=True,
                   help="Sur deux appuis (par défaut : les lisses sont rarement continues d'un montant à l'autre) : M = qL²/8 ; continue sur n travées : "
                        "coefficients des tables (0,125 / 0,100 / 0,107 qL²).")
        if role == "barreau":
            _section_w("Lisse basse (non calculée)", ("remplissage", "section_basse"), cas, ["creux", "froid", "plats", "ronds"], width=240,
                       help="Elle fixe la portée des barreaux et n'est pas vérifiée.")
    e = next((e for e in res.get("elements", []) if e["cle"] == role), None)
    if e:
        d = e["detail"] or {}
        crit = ME.critere_texte(e["critere"])
        st.markdown(f'<div class="gc-elt">{e["libelle"]} → <span class="{_cls(e["taux"])}">η = {e["taux"] * 100:.0f} %</span> ({crit}'
                    f'{" · " + CH.CAS_LIBELLES.get(d.get("cas_charge"), d.get("cas_charge")) if d.get("cas_charge") else ""})</div>', unsafe_allow_html=True)
    _tableau_role(cas, res, role, masse_max, seuls, froid)


def _etape_sections(cas: Dict, res: Dict):
    gen = LI.generation(P)
    for k in [k for k in list(st.session_state) if isinstance(k, str) and k.startswith(P + "tab_") and not k.endswith(f"_{gen}")]:
        del st.session_state[k]
    st.markdown("**Pour chaque élément** : sa section, puis le tableau des profilés compatibles — un clic sur une ligne la retient.")
    with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=P + "c_lib"):
        masse_max = st.number_input("Masse maxi des tableaux (kg/m)", value=15.0, step=1.0, min_value=1.0, key=P + "lib_masse", width=200)
        seuls = st.checkbox("Seulement ceux qui tiennent", value=False, key=P + "lib_ok")
        froid = st.checkbox("Y compris les tubes formés à froid", value=False, key=P + "lib_froid")
    for role in ET.roles_du_cas(cas):
        _bloc_role(cas, res, role, masse_max, seuls, froid)
    sec = res.get("section_montant")
    if sec:
        st.caption(f"Élément principal : {BI.libelle(sec)} · {BI.dimensions_texte(sec)} mm · A = {_fmt(sec['A'], 2)} cm² · I_y = {_fmt(sec['Iy'], 2)} cm⁴ · "
                   f"W_el,y = {_fmt(sec['Wel_y'], 2)} cm³ · I_z = {_fmt(sec['Iz'], 2)} cm⁴ · W_el,z = {_fmt(sec['Wel_z'], 2)} cm³ · A_v,z = {_fmt(sec['Avz'], 2)} cm²")


# ---------------------------------------------------------------- onglet 4 : ancrage
def _etape_ancrage(cas: Dict, res: Dict):
    fx = cas["fixation"]
    g = cas["geometrie"]
    typ = _choix("Fixation", ("fixation", "type"), cas, list(FX.TYPES), FX.TYPES, segmente=True,
                 help="Platine boulonnée : le montant est soudé sur sa platine posée sur la dalle, fixée par chevilles ou boulons ; platine latérale : la "
                      "platine est verticale, en applique sur le nez de dalle, le montant soudé le long d'elle ; soudé : le montant est soudé directement "
                      "sur un plat existant (la soudure seule est vérifiée) ; scellé : rien n'est vérifié.")
    if g.get("systeme") == "mur":
        with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=P + "c_cons"):
            _num("Console : longueur e du mur à l'axe de la main courante (mm)", ("fixation", "console", "e"), cas, 10.0, "%.0f", 20.0, width=380,
                 help="Bras de levier de la charge verticale sur la console et sur ses chevilles ; la section de la console se choisit dans l'onglet Sections.")
    if (res.get("geometrie") or {}).get("sans_montant"):
        st.info("Sans montant : rien à fixer — la main courante est ancrée dans les murs (réactions dans le résultat).")
        return
    if typ == "scellee":
        st.caption("Montant scellé dans le béton ou dans un fourreau : la fixation n'est pas vérifiée ici.")
        return
    lat = typ == FX.LATERALE and g.get("systeme") == "montants"
    st.markdown("**Soudure du montant le long de la platine** — deux cordons longitudinaux de la hauteur b_p, vérifiés par le module Soudures (C8)" if lat
                else "**Soudure sur la platine** — vérifiée par le module Soudures")
    with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=P + "c_soud"):
        jon = _choix("Jonction", ("fixation", "soudure", "jonction"), cas, ["angle", "partielle", "pleine"],
                     {"angle": "cordon d'angle", "partielle": "pénétration partielle", "pleine": "pleine pénétration"}, segmente=True)
        if jon == "angle":
            _num("Gorge a (mm)", ("fixation", "soudure", "a"), cas, 0.5, "%.1f", 1.0, width=140)
        elif jon == "partielle":
            _num("Pénétration p (mm)", ("fixation", "soudure", "p"), cas, 0.5, "%.1f", 1.0, width=160)
            _num("Renfort a_r (mm, 0 : sans)", ("fixation", "soudure", "renfort"), cas, 0.5, "%.1f", 0.0, width=180)
        _num("Épaisseur de la platine t_p (mm)", ("fixation", "platine", "tp"), cas, 1.0, "%.0f", 3.0, width=220)
    s = (res.get("fixation") or {}).get("soudure")
    if s:
        st.markdown(f"Soudure : <span class='{_cls(s['taux'])}'>η = {s['taux'] * 100:.0f} %</span> — {s['critere']}, {s['position']} · "
                    f"{CH.CAS_LIBELLES.get(s.get('cas_charge'), '')} · {s['efforts_txt']}", unsafe_allow_html=True)
        vues = SCS.vues(s["res"])
        if vues:
            with st.expander("Dessins de la soudure (module Soudures)", expanded=False):
                cols = st.columns(min(2, len(vues)), gap="small")
                for i, (cle, titre, d) in enumerate(vues[:2]):
                    with cols[i]:
                        with st.container(key=P + f"soud_{i}"):
                            st.markdown(f'<div class="gc-leg"><b>{titre.upper()}</b></div>' + SCS.svg(d, f"gc_s{i}"), unsafe_allow_html=True)
    if typ not in ("platine", FX.LATERALE):
        return
    if lat:
        st.markdown("**Platine latérale et ancrage** — platine verticale en applique sur le nez de dalle, deux rangées horizontales de chevilles (ou boulons) "
                    "à ±e_x/2 du centre, la haute tendue et la basse en appui ; colonnes de part et d'autre du montant à e_y")
        with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=P + "c_pl"):
            _num("d_h — descente du dessus de dalle au haut de la platine (mm)", ("fixation", "platine", "d_h"), cas, 5.0, "%.0f", 0.0, width=330,
                 help="Le montant est vérifié sur H + d_h ; le moment à la platine vaut P·(H + d_h + b_p/2).")
            _num("b_p — hauteur de la platine (mm)", ("fixation", "platine", "bp"), cas, 5.0, "%.0f", 40.0, width=220)
            _num("l_p — largeur de la platine (mm)", ("fixation", "platine", "lp"), cas, 5.0, "%.0f", 40.0, width=220)
            _num("e_x — entraxe vertical des deux rangées (mm)", ("fixation", "ancrage", "ex"), cas, 5.0, "%.0f", 10.0, width=270, help="Le bras de levier z du couple d'ancrages.")
            _num("Chevilles par rangée (2 ou 4, par paires)", ("fixation", "ancrage", "n_par_rangee"), cas, 2, min_value=2, max_value=4, width=240)
            _num("e_y — entraxe horizontal des colonnes (mm)", ("fixation", "ancrage", "ey"), cas, 5.0, "%.0f", 10.0, width=270,
                 help="De part et d'autre du montant ; doit dépasser sa largeur b plus un trou.")
    else:
        st.markdown("**Platine et ancrage** — platine rigide, deux rangées de chevilles (ou boulons) de part et d'autre du montant, à ±e_x/2")
        with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=P + "c_pl"):
            _num("b_p — dans le sens de l'effort (mm)", ("fixation", "platine", "bp"), cas, 5.0, "%.0f", 40.0, width=230)
            _num("l_p — largeur (mm)", ("fixation", "platine", "lp"), cas, 5.0, "%.0f", 40.0, width=170)
            _num("e_x — entraxe des deux rangées (mm)", ("fixation", "ancrage", "ex"), cas, 5.0, "%.0f", 10.0, width=250, help="Dans le sens de l'effort ; doit dépasser la dimension h_p du montant.")
            _num("Chevilles par rangée", ("fixation", "ancrage", "n_par_rangee"), cas, 1, min_value=1, max_value=4, width=160)
            _num("e_y — entraxe dans la rangée (mm)", ("fixation", "ancrage", "ey"), cas, 5.0, "%.0f", 10.0, width=230)
    with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=P + "c_anc"):
        ta = _choix("Ancrage", ("fixation", "ancrage", "type"), cas, list(FX.ANCRAGES), FX.ANCRAGES, segmente=True)
        if ta == "boulon":
            _choix("Boulon", ("fixation", "ancrage", "boulon"), cas, FX.boulons(), width=120)
            _choix("Classe", ("fixation", "ancrage", "classe"), cas, FX.classes(), width=110)
        else:
            _texte_w("Cheville (désignation)", ("fixation", "ancrage", "nom"), cas, width=220)
            _num("N_Rd par cheville (kN)", ("fixation", "ancrage", "NRd"), cas, 0.5, "%.1f", 0.1, width=170, help="Valeur de calcul en traction de l'ETA (logiciel du fabricant), pour le béton et la profondeur du projet.")
            _num("V_Rd par cheville (kN)", ("fixation", "ancrage", "VRd"), cas, 0.5, "%.1f", 0.1, width=170, help="Valeur de calcul en cisaillement de l'ETA.")
    pl = (res.get("fixation") or {}).get("platine")
    if pl and pl.get("taux_max") is not None:
        arr = f" · arrachement P = {_fmt(pl['P'] / 1e3, 2)} kN" if pl.get("mode") == FX.LATERALE else ""
        st.markdown(f"{CH.CAS_LIBELLES.get(pl.get('cas_charge'), '')}{arr} · bras de levier z = {_fmt(pl['z'])} mm · traction de la rangée Z = {_fmt(pl['Z'] / 1e3, 2)} kN → "
                    f"F_t = {_fmt(pl['Ft'] / 1e3, 2)} kN par cheville (≤ {_fmt(pl['FtRd'] / 1e3, 2)}) : <span class='{_cls(pl['taux_t'])}'>{pl['taux_t'] * 100:.0f} %</span> · "
                    f"F_v = {_fmt(pl['Fv'] / 1e3, 2)} kN (≤ {_fmt(pl['FvRd'] / 1e3, 2)}) : <span class='{_cls(pl['taux_v'])}'>{pl['taux_v'] * 100:.0f} %</span> · interaction "
                    f"<span class='{_cls(pl['inter'])}'>{pl['inter'] * 100:.0f} %</span><br>Platine : bras m = {_fmt(pl['m'])} mm, b_eff = {_fmt(pl['b_eff'], 0)} mm, "
                    f"σ_p = {_fmt(pl['sigma_p'])} N/mm² ≤ {_fmt(pl['lim_sigma'], 0)} : <span class='{_cls(pl['taux_platine'])}'>{pl['taux_platine'] * 100:.0f} %</span> "
                    f"— épaisseur requise {_fmt(pl['tp_req'])} mm (t_p = {_fmt(pl['tp'], 0)})", unsafe_allow_html=True)
    elif pl:
        for m in pl.get("messages", []):
            st.warning(m)


# ---------------------------------------------------------------- onglet 5 : détail
def _etape_detail(etat: Dict, cas: Dict, res: Dict):
    if res["statut"] == ET.REFUS and not res.get("elements"):
        st.warning((res["messages"] or ["Calcul impossible"])[0])
    lignes = ""
    g = res.get("gouvernant") or {}
    for e in res.get("elements", []):
        d = e["detail"]
        cls = " class='g'" if e["cle"] == g.get("cle") else ""
        cas_c = CH.CAS_LIBELLES.get(d.get("cas_charge"), d.get("cas_charge") or "") if isinstance(d, dict) else ""
        if e["cle"] in ("montant", "lisse", "barreau", "lisse_inter", "console"):
            M = max(abs(d.get("M1", 0.0)), abs(d.get("Mv", 0.0)), abs(d.get("Mh", 0.0)))
            lignes += (f"<tr{cls}><td>{e['libelle']}</td><td>{cas_c}</td><td class='n'>{_fmt(M / 1e6, 3)}</td><td class='n'>{_fmt(max(abs(d.get('V1', 0.0)), abs(d.get('V2', 0.0))) / 1e3, 2)}</td>"
                       f"<td class='n'>{_fmt(d['sigma'])}</td><td class='n'>{_fmt(d['tau'], 2)}</td><td class='n'>{_fmt(d['fleche'], 2)} / {_fmt(d['fleche_lim'], 1)}</td>"
                       f"<td class='n {_cls(e['taux'])}'>{e['taux'] * 100:.0f} %</td><td>{ME.critere_texte(e['critere'])}</td></tr>")
        else:
            lignes += (f"<tr{cls}><td>{e['libelle']}</td><td>{cas_c}</td><td class='n'>—</td><td class='n'>—</td><td class='n'>—</td><td class='n'>—</td><td class='n'>—</td>"
                       f"<td class='n {_cls(e['taux'])}'>{e['taux'] * 100:.0f} %</td><td>{e['critere']}</td></tr>")
    st.markdown("<table class='gc-tab'><tr><th>Élément</th><th>cas gouvernant</th><th class='n'>M (kN·m)</th><th class='n'>V (kN)</th><th class='n'>σ (N/mm²)</th><th class='n'>τ</th>"
                "<th class='n'>flèche / adm. (mm)</th><th class='n'>η</th><th>critère</th></tr>" + lignes + "</table>", unsafe_allow_html=True)
    liste = res.get("montants") or res.get("consoles") or []
    if liste:
        lignes = ""
        for v in liste:
            lignes += (f"<tr{' class=g' if v['nom'] == ((res.get('montant') or res.get('console') or {}).get('nom')) else ''}><td><b>{v['nom']}</b>{' · angle' if v['coin'] else ''}</td>"
                       f"<td class='n'>{_fmt(v['x'], 0)}</td><td class='n'>{_fmt(v['trib1'] + v['trib2'], 0)}</td><td class='n'>{_fmt(v['P1'] / 1e3, 2)}"
                       f"{' + ' + _fmt(v['P2'] / 1e3, 2) if v['P2'] > 0 else ''}</td><td class='n'>{_fmt(v['N'] / 1e3, 2)}</td><td class='n'>{_fmt(v['sigma'])}</td>"
                       f"<td class='n'>{_fmt(v['fleche'], 2)}</td><td class='n {_cls(v['taux_max'])}'>{v['taux_max'] * 100:.0f} %</td></tr>")
        st.markdown("<table class='gc-tab'><tr><th>Montant / console</th><th class='n'>x (mm)</th><th class='n'>s_trib</th><th class='n'>P (kN)</th><th class='n'>N (kN)</th>"
                    "<th class='n'>σ</th><th class='n'>flèche</th><th class='n'>η</th></tr>" + lignes + "</table>", unsafe_allow_html=True)
    fo = FO.toutes(res)
    with st.expander("Les formules, écrites comme à la main (la note les imprime) — " + FO.TITRES[FO.cle_methode(res)], expanded=False):
        for titre, cle in (("1 · Montant principal", "montant"), ("Console", "console"), ("2 · Main courante", "lisse"), ("3 · Remplissage", "barre"), ("4 · Platine et ancrage", "platine")):
            b = fo.get(cle)
            if b and b.get("subs"):
                # les groupes dans l'ordre de lecture : charges, sollicitations, contraintes, flèche, taux (la note les imprime de même)
                st.markdown(f"**{titre}** — {b['cas']}" + (f" (autres cas : {b['autres']})" if b.get("autres") else "") + "  \n"
                            + "  \n".join(f"*{lib}*  \n" + "  \n".join(FO.rendre(s) for s in subs) for lib, subs in b["groupes"]))
        sou = fo["soudure"]
        if sou["critere"]:
            st.markdown("**Soudure**  \n" + "  \n".join(FO.rendre(s) for s in sou["groupe"] + sou["point"] + sou["critere"]))
        st.caption("Chaque ligne est une substitution numérique testée contre le moteur (tests/test_garde_corps_formules.py).")
    with st.expander("Paramètres de la méthode (limites, flèche, acier, coefficients)", expanded=False):
        m = cas["methode"]
        with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=P + "c_meth1"):
            _choix("Nuance", ("methode", "nuance"), cas, list(ET.nuances()), width=140)
            _num("E (N/mm²)", ("methode", "E"), cas, 1000.0, "%.0f", 100000.0, width=150)
            _choix("Flèche admissible", ("methode", "fleche_mode"), cas, ["L/x", "mm"], {"L/x": "L / x", "mm": "valeur en mm"}, width=150)
            _num("x, ou la valeur (mm)", ("methode", "fleche_val"), cas, 10.0, "%.0f", 1.0, width=170,
                 help="L/100 par défaut (pratique des garde-corps ; l'ancien module proposait L/200).")
        with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=P + "c_meth2"):
            if m["mode"] == ME.ELS:
                _num("γ_adm (σ_adm = f_y/γ_adm, τ_adm = σ_adm/√3)", ("methode", "gamma_adm"), cas, 0.05, "%.2f", 1.0, width=290,
                     help="Les contraintes admissibles se déduisent de la nuance : σ_adm = f_y/1,5 (156,7 N/mm² pour le S235), τ_adm = σ_adm/√3 (90,5), "
                          "et la contrainte équivalente σ_eq = √(σ² + 3τ²) ≤ σ_adm.")
                _num("k cordons (kg/(mm·cm))", ("methode", "k"), cas, 5.0, "%.0f", 1.0, width=190)
                _num("γ_F (résistances de calcul → service)", ("methode", "gamma_F"), cas, 0.05, "%.2f", 1.0, width=250)
            else:
                _num("γ_Q", ("methode", "gamma_Q"), cas, 0.05, "%.2f", 1.0, width=120)
                _num("γ_M0", ("methode", "gamma_M0"), cas, 0.05, "%.2f", 1.0, width=120)
                _num("γ_M2 (boulons, cordons)", ("methode", "gamma_M2"), cas, 0.05, "%.2f", 1.0, width=200)
                _choix("Critère des cordons", ("methode", "ec"), cas, ["directionnelle", "simplifiee"], {"directionnelle": "directionnelle", "simplifiee": "simplifiée"}, width=180)
    with st.expander("Identification de la note (projet, partie, indice)", expanded=False):
        ident = etat.setdefault("identification", {"projet": "", "partie": "", "indice": "0"})
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            st.text_input("Projet", value=ident.get("projet", ""), key=P + "id_projet", on_change=lambda: ident.__setitem__("projet", st.session_state[P + "id_projet"]))
        with c2:
            st.text_input("Partie d'ouvrage", value=ident.get("partie", ""), key=P + "id_partie", on_change=lambda: ident.__setitem__("partie", st.session_state[P + "id_partie"]))
        with c3:
            st.text_input("Indice", value=ident.get("indice", "0"), key=P + "id_indice", on_change=lambda: ident.__setitem__("indice", st.session_state[P + "id_indice"]))
    with st.expander("Méthode et sources (les mêmes textes que la page 2 de la note)", expanded=False):
        for sec in MT.sections(cas["methode"].get("mode", ME.ELS), cas["fixation"].get("type", "platine"), bool((res.get("geometrie") or {}).get("sans_montant"))):
            st.markdown(f"**{sec['titre']}**")
            for p in sec["paragraphes"]:
                st.markdown(p)
        st.markdown("**Références** — " + " · ".join(MT.REFERENCES))


# ---------------------------------------------------------------- page
def show():
    etat = _etat()
    cas = ET.cas_actif(etat)
    LI.au_rendu(P, REGISTRE, cas)
    _entete(etat, cas)
    _switch(cas)
    res = ET.verifier(cas)
    col_g, col_d = st.columns([1.15, 1], gap="large")
    with col_g:
        _colonne_dessin(cas, res)
    with col_d:
        _carte_resultat(cas, res)
        onglets = st.tabs(ETAPES)
        with onglets[0]:
            _etape_garde_corps(cas, res)
        with onglets[1]:
            _etape_charges(cas, res)
        with onglets[2]:
            _etape_sections(cas, res)
        with onglets[3]:
            _etape_ancrage(cas, res)
        with onglets[4]:
            _etape_detail(etat, cas, res)

# -*- coding: utf-8 -*-
"""Écran Streamlit du module « Garde-corps ».

Le moteur, l'état, les dessins et la note vivent dans ``acier/garde_corps`` (sans
Streamlit : ``tests/test_garde_corps_moteur.py`` le garde) ; ce fichier n'est que la vue.

Organisation (étude docs/acier/GARDE_CORPS_ETUDE.md) : barre d'outils du cas ; le SWITCH de
méthode — « Méthode ELS (bureau) » | « Eurocode (ELU) » ; à gauche les dessins (élévation
développée, vue en plan, détail de la fixation, section) ; à droite la carte RÉSULTAT (le
taux de chaque élément) puis six onglets à libellé fixe : Garde-corps, Charges, Sections
(avec la BIBLIOTHÈQUE : le plus léger qui tient, en un clic), Remplissage, Fixation, Détail.
Une seule source : l'état ``st.session_state["gc"]`` (``acier/garde_corps/etat.py``) ; les
widgets (clés ``gc_w_``) en sont des vues, réinitialisées dès que l'état change par ailleurs
(``modules/liaison.py``). Le cas par défaut est un garde-corps complet et conforme : zéro
clic pour un premier résultat, un clic pour la note.
"""
import copy
import json
from typing import Dict, List, Optional

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
from acier.profiles import familles as PF
from acier.soudures import schemas as SCS
from modules import liaison as LI

P = "gc_"
REGISTRE: LI.Registre = {}
ETAPES = ["1 · Garde-corps", "2 · Charges", "3 · Sections", "4 · Remplissage", "5 · Fixation", "6 · Détail"]
MODES = {ME.ELS: "Méthode ELS (bureau)", ME.EUROCODE: "Eurocode (ELU)"}
LIBRE = "— section libre (dimensions) —"
GROUPES_LIB = {"creux": "tubes (CHS, SHS, RHS)", "froid": "tubes formés à froid", "plats": "plats", "ronds": "ronds"}
DIMS_LIBRES = {"TUBE": {"d": 48.3, "t": 3.2}, "CAISSON": {"h": 60.0, "b": 40.0, "t": 3.0}, "PLAT": {"h": 50.0, "b": 10.0}, "ROND": {"d": 12.0}}
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


def _section_w(label, chemin, cas, groupes=None, help=None):
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
    st.selectbox(label, options, index=options.index(valeur), key=cle, on_change=_on, help=help)
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
    st.caption("Dimensionnement d'un garde-corps — montants verticaux sur platine ou lisse ancrée dans un mur, forme en plan droite, en L "
               "ou en U, entraxes libres, charges EN 1991-1-1 Tab. 6.12 par catégorie d'usage, remplissage, fixation (soudure, chevilles ou "
               "boulons, platine) · formules du bureau (σ = M/W, τ = V/A_v, flèche) ou Eurocode · bibliothèque des profilés creux et des "
               "plats · version 2.0")
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
        if st.button("📄 Note", key=P + "btn_note", width="content", help="Note de calcul PDF en deux pages : calcul (dessins, charges, éléments, fixation, formules), méthode et détail"):
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
            st.markdown(f'<div class="gc-switch"><b>Efforts de service</b>, sans coefficient : les vieilles formules du bureau — σ = M/W + N/A ≤ '
                        f'<b>{m.get("sigma_adm", 160):g} N/mm²</b>, τ = V/A_v ≤ <b>{m.get("tau_adm", 90):g} N/mm²</b>, flèche ≤ <b>{fl}</b> ; soudure '
                        f'par la méthode ELS (k = {m.get("k", 100):g}), chevilles et boulons à leur résistance de calcul / γ_F = {m.get("gamma_F", 1.5):g} '
                        f'(paramètres dans l\'onglet Détail).</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="gc-switch"><b>Efforts de calcul (ELU)</b> : charges × γ_Q = {m.get("gamma_Q", 1.5):g}, σ ≤ f_y/γ_M0 (élastique, W_el), '
                        f'τ ≤ f_y/(√3·γ_M0), flèche ≤ {fl} sous efforts de service ; soudure EN 1993-1-8 directionnelle, boulons Tab. 3.4, chevilles à '
                        f'leurs valeurs de calcul ETA (paramètres dans l\'onglet Détail).</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- colonne de gauche : dessins
def _colonne_dessin(cas: Dict, res: Dict):
    vues = SC.vues(res)
    with st.container(key=P + "dessin"):
        if not vues:
            st.warning("Calcul impossible — " + (res.get("messages") or ["données manquantes"])[0])
        else:
            cle, titre, d = vues[0]
            st.markdown(f'<div class="gc-leg" style="margin:2px 0"><b>{titre.upper()}</b> · {res["geometrie"]["phrase"]}</div>' + SC.svg(d, "gc_v0"),
                        unsafe_allow_html=True)
    if len(vues) > 1:
        cols = st.columns(len(vues) - 1, gap="small")
        for i, (cle, titre, d) in enumerate(vues[1:], 1):
            with cols[i - 1]:
                with st.container(key=P + f"dessin_{i}"):
                    st.markdown(f'<div class="gc-leg" style="margin:2px 0"><b>{titre.upper()}</b></div>' + SC.svg(d, f"gc_v{i}"), unsafe_allow_html=True)
    st.markdown('<div class="gc-leg">Pièces au trait fort, sections hachurées, cordons en ocre à leur taille a·√2, chevilles en gris, ◉ l\'élément '
                'gouvernant et son taux, flèches rouges dans le sens des charges (⊙ vers soi), cotes en mm.</div>', unsafe_allow_html=True)


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
        phrase = f"<b>{g['libelle']}</b> — {ME.critere_texte(g['critere']) if g['cle'] not in ('soudure', 'ancrage', 'platine') else g['critere']}"
    m = cas["methode"]
    ch = res["charges"]
    ligne_m = (f"{MODES[m['mode']]} · {CH.phrase(ch)}")
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
    c1, c2 = st.columns([1.1, 1])
    with c1:
        _choix("Système", ("geometrie", "systeme"), cas, list(GE.SYSTEMES), GE.SYSTEMES, segmente=True,
               help="Montants verticaux encastrés en pied (platine) portant la lisse haute, ou une lisse horizontale (main courante) "
                    "portée par des consoles ancrées dans un mur, aux mêmes entraxes.")
    with c2:
        _choix("Forme en plan", ("geometrie", "forme"), cas, list(GE.FORMES), GE.FORMES, segmente=True,
               help="Un, deux ou trois tronçons droits ; un montant d'angle reçoit une demi-travée de chaque tronçon dans deux directions.")
    n = GE.N_TRONCONS.get(g.get("forme", "droit"), 1)
    for i in range(n):
        with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=P + f"c_tr{i}"):
            st.markdown(f"**Tronçon {i + 1}**")
            _num("Longueur L (mm)", ("geometrie", "troncons", i, "L"), cas, 100.0, "%.0f", 100.0, width=150)
            _num("Entraxe maxi (mm)", ("geometrie", "troncons", i, "s_max"), cas, 50.0, "%.0f", 200.0, width=150,
                 help="La longueur est répartie en travées égales ne dépassant pas cet entraxe — sauf si des entraxes sont saisis à côté.")
            _entraxes_w(i, cas)
    with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=P + "c_ext"):
        _num("H — main courante au-dessus de la fixation (mm)", ("geometrie", "H"), cas, 10.0, "%.0f", 300.0, width=260,
             help="Bras de levier de la charge horizontale : hauteur de la lisse haute au-dessus de la platine (fixation en nez de dalle : "
                  "ajouter la descente). La charge s'applique au plus à 1,20 m (EN 1991-1-1 §6.4).")
        _choix("Début", ("geometrie", "debut"), cas, list(GE.EXTREMITES), GE.EXTREMITES, width=240,
               help="Un montant d'extrémité, ou la lisse ancrée dans le mur : le mur reprend la demi-travée.")
        _choix("Fin", ("geometrie", "fin"), cas, list(GE.EXTREMITES), GE.EXTREMITES, width=240)
    if g.get("systeme") == "mur":
        st.caption("Les consoles (section, longueur e du mur à l'axe de la lisse) se règlent dans l'onglet Fixation.")
    geo = res.get("geometrie") or {}
    if geo.get("montants"):
        nom = "console" if g.get("systeme") == "mur" else "montant"
        lignes = ""
        for m in geo["montants"]:
            lignes += (f"<tr><td><b>{m['nom'] if nom == 'montant' else m['nom'].replace('M', 'C')}</b></td><td class='n'>{_fmt(m['x'], 0)}</td>"
                       f"<td class='n'>{_fmt(m['trib1'], 0)}</td><td class='n'>{_fmt(m['trib2'], 0) if m['coin'] else '—'}</td><td>{'angle' if m['coin'] else ''}</td></tr>")
        st.markdown(f"<table class='gc-tab'><tr><th>{nom.capitalize()}</th><th class='n'>x (mm)</th><th class='n'>s_trib (mm)</th><th class='n'>s_trib,2 (mm)</th><th></th></tr>"
                    + lignes + "</table>", unsafe_allow_html=True)
        st.caption(f"{geo['phrase']} · portée de la lisse : l'entraxe le plus grand, {_fmt(geo['s_max'], 0)} mm.")


# ---------------------------------------------------------------- onglet 2 : charges
def _etape_charges(cas: Dict, res: Dict):
    ch = cas["charges"]
    cats = [c[0] for c in CH.CATEGORIES]
    libs = {c[0]: (c[1] + (f" — {c[2]:g} kN/m" if c[2] is not None else "")) for c in CH.CATEGORIES}

    def _cat():
        cas["charges"].update(CH.normaliser(cas["charges"]))
        _reset_widgets()
    c1, c2 = st.columns([1.6, 1])
    with c1:
        _choix("Catégorie d'usage (EN 1991-1-1 Tab. 6.12, valeurs recommandées)", ("charges", "categorie"), cas, cats, libs, on_change=_cat,
               help="La charge linéaire horizontale sur la lisse haute selon l'usage ; l'annexe nationale belge peut fixer d'autres valeurs : "
                    "« valeur libre » les accepte. La note imprime la valeur retenue et son origine.")
    with c2:
        _num("q_h — charge horizontale (kN/m)", ("charges", "qh"), cas, 0.1, "%.2f", 0.0, disabled=(ch.get("categorie") != "perso"))
    with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=P + "c_ch1"):
        cleP = P + "w_charges_mode"
        LI.enregistrer(REGISTRE, cleP, ("charges", "mode"), lambda v: v == "P")

        def _mode():
            cas["charges"]["mode"] = "P" if st.session_state[cleP] else "q"
        st.checkbox("Saisir directement la charge ponctuelle P qui revient à chaque montant", value=(ch.get("mode") == "P"), key=cleP, on_change=_mode,
                    help="À la place de q_h × longueur tributaire ; la lisse reçoit alors l'équivalent P / entraxe.")
        if ch.get("mode") == "P":
            _num("P (kN)", ("charges", "P"), cas, 0.1, "%.2f", 0.0, width=130)
    with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=P + "c_ch2"):
        _num("q_v — charge verticale sur la main courante (kN/m)", ("charges", "qv"), cas, 0.1, "%.2f", 0.0, width=300, help=CH.SOURCE_QV)
        _num("Q — charge ponctuelle horizontale sur la lisse (kN)", ("charges", "Q"), cas, 0.1, "%.2f", 0.0, width=300,
             help="En option (0 : sans). Elle s'ajoute à q_h sur la lisse (« somme ») ou forme un cas séparé (« max »).")
        _choix("Avec q_h", ("methode", "comb"), cas, ["somme", "max"], {"somme": "somme (Q + q_h)", "max": "max (cas séparés)"}, width=200)
    with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=P + "c_ch3"):
        _num("q_r — pression sur le remplissage (kN/m²)", ("charges", "qr"), cas, 0.1, "%.2f", 0.0, width=300, help=CH.SOURCE_QR)
        _num("Q_r — charge ponctuelle sur le remplissage (kN)", ("charges", "Qr"), cas, 0.1, "%.2f", 0.0, width=300, help=CH.SOURCE_QR_PONCT)
    st.caption(CH.phrase(res["charges"]) + " · " + res["limites"]["nature"] + ".")


# ---------------------------------------------------------------- onglet 3 : sections et bibliothèque
def _bibliotheque(cas: Dict, res: Dict):
    cle_p = ET.element_principal(cas)
    lib = "montant" if cle_p == "montant" else "lisse (élément principal)"
    st.markdown(f"**Bibliothèque — le {lib} qui tient**")
    with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=P + "c_lib"):
        groupes = st.pills("Familles", list(GROUPES_LIB), default=["creux"], format_func=lambda g: GROUPES_LIB[g], key=P + "lib_groupes", selection_mode="multi")
        masse_max = st.number_input("Masse maxi (kg/m)", value=15.0, step=1.0, min_value=1.0, key=P + "lib_masse", width=140)
        seuls = st.checkbox("Seulement ceux qui tiennent", value=False, key=P + "lib_ok")
    if not groupes:
        st.caption("Choisir au moins une famille.")
        return
    cands = ET.candidats(cas, list(groupes), float(masse_max))
    best = ET.plus_leger(cands)
    cur = (res.get("section_montant") or {}).get("nom")
    if best is not None:
        if st.button(f"Prendre le plus léger qui tient : {best['section']['nom']} ({best['section']['masse']:.2f} kg/m, η = {best['taux'] * 100:.0f} %)".replace(".", ","),
                     key=P + "btn_best", width="content", disabled=(best["section"]["nom"] == cur)):
            ET.appliquer_section(cas, best["section"])
            _changer(f"{best['section']['nom']} retenu")
    else:
        st.warning("Aucune section de la sélection ne tient : élargir les familles ou la masse maximale.")
    montres = [c for c in cands if not seuls or c["taux"] <= 1.0][:30]
    lignes = ""
    for c in montres:
        s = c["section"]
        cls = " cur" if s["nom"] == cur else ""
        lignes += (f"<tr class='{cls}'><td>{s['nom']}</td><td>{BI.dimensions_texte(s)}</td><td class='n'>{_fmt(s['masse'], 2)}</td>"
                   f"<td class='n {_cls(c['taux_sigma'])}'>{c['taux_sigma'] * 100:.0f} %</td><td class='n {_cls(c['taux_fleche'])}'>{c['taux_fleche'] * 100:.0f} %</td>"
                   f"<td class='n {_cls(c['taux'])}'>{c['taux'] * 100:.0f} %</td></tr>")
    st.markdown("<table class='gc-tab'><tr><th>Section</th><th>dimensions</th><th class='n'>kg/m</th><th class='n'>η σ</th><th class='n'>η flèche</th><th class='n'>η</th></tr>"
                + lignes + "</table>", unsafe_allow_html=True)
    st.caption(f"{len(cands)} sections essayées par masse croissante ({len(montres)} montrées) : chacune vérifiée comme {lib} du cas "
               f"({ME.critere_texte('sigma')}, τ, flèche) ; la fixation est vérifiée pour la section retenue.")


def _etape_sections(cas: Dict, res: Dict):
    g = cas["geometrie"]
    if g.get("systeme") == "mur":
        _section_w("Lisse (élément principal, portée = entraxe des consoles)", ("lisse", "section"), cas, ["creux", "froid", "plats", "ronds"],
                   help="h vertical : l'axe fort porte la charge verticale, l'axe faible la charge horizontale.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            _section_w("Montant", ("montant", "section"), cas, ["creux", "froid", "plats", "ronds"],
                       help="La dimension h est dans le sens de la charge horizontale (axe fort) ; un montant d'angle rectangulaire est vérifié dans les deux orientations.")
        with c2:
            _section_w("Lisse haute (main courante)", ("lisse", "section"), cas, ["creux", "froid", "plats", "ronds"],
                       help="h vertical : l'axe fort porte q_v, l'axe faible q_h.")
    sec = res.get("section_montant")
    if sec:
        st.caption(f"{BI.libelle(sec)} · {BI.dimensions_texte(sec)} mm · A = {_fmt(sec['A'], 2)} cm² · I_y = {_fmt(sec['Iy'], 2)} cm⁴ · W_el,y = {_fmt(sec['Wel_y'], 2)} cm³ · "
                   f"I_z = {_fmt(sec['Iz'], 2)} cm⁴ · W_el,z = {_fmt(sec['Wel_z'], 2)} cm³ · A_v,z = {_fmt(sec['Avz'], 2)} cm²")
    _bibliotheque(cas, res)


# ---------------------------------------------------------------- onglet 4 : remplissage
def _etape_remplissage(cas: Dict, res: Dict):
    rp = cas["remplissage"]
    typ = _choix("Remplissage", ("remplissage", "type"), cas, list(ET.REMPLISSAGES), ET.REMPLISSAGES, segmente=True)
    if typ == "barreaux":
        with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=P + "c_rp"):
            _num("Entraxe des barreaux (mm)", ("remplissage", "entraxe"), cas, 5.0, "%.0f", 20.0, width=200, help="Le barreau reprend q_r × entraxe, ou Q_r à mi-portée (deux cas, le pire).")
            _num("Portée du barreau L (mm)", ("remplissage", "L"), cas, 10.0, "%.0f", 50.0, width=200, help="Entre la lisse basse et la lisse haute (appuis simples).")
        _section_w("Section des barreaux", ("remplissage", "section"), cas, ["plats", "ronds", "creux", "froid"], help="h dans le sens de la charge (perpendiculaire au garde-corps).")
    elif typ == "lisses":
        with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=P + "c_rp"):
            _num("Nombre de lisses intermédiaires", ("remplissage", "n_lisses"), cas, 1, min_value=1, width=200)
            _num("Entraxe des lisses (mm)", ("remplissage", "entraxe_lisses"), cas, 10.0, "%.0f", 50.0, width=200, help="Chaque lisse reprend q_r × entraxe, ou Q_r à mi-portée, sur la portée entre montants.")
        _section_w("Section des lisses intermédiaires", ("remplissage", "section_lisses"), cas, ["creux", "froid", "plats", "ronds"])
    elif typ == "panneau":
        st.caption("Remplissage en panneau (verre, tôle) : sa résistance sous q_r et Q_r est à justifier par le fournisseur ; la note le dit.")
    else:
        st.caption("Main courante seule (rampe d'escalier, lisse de protection).")
    v = (res.get("remplissage") or {}).get("verif")
    if v:
        st.caption(f"{res['remplissage']['libelle']} : cas gouvernant « {'pression q_r × e' if v['cas'] == 'q' else 'charge ponctuelle Q_r'} » — "
                   f"M = {_fmt(v['M1'] / 1e6, 4)} kN·m, σ = {_fmt(v['sigma'])} N/mm², flèche {_fmt(v['fleche'], 2)} mm ≤ {_fmt(v['fleche_lim'], 1)} → η = {v['taux_max'] * 100:.0f} %")


# ---------------------------------------------------------------- onglet 5 : fixation
def _etape_fixation(cas: Dict, res: Dict):
    fx = cas["fixation"]
    g = cas["geometrie"]
    typ = _choix("Fixation", ("fixation", "type"), cas, list(FX.TYPES), FX.TYPES, segmente=True,
                 help="Platine boulonnée : le montant est soudé sur sa platine, fixée par chevilles ou boulons ; soudé : le montant est soudé "
                      "directement sur un plat existant (la soudure seule est vérifiée) ; scellé : rien n'est vérifié.")
    if g.get("systeme") == "mur":
        c1, c2 = st.columns(2)
        with c1:
            _section_w("Section de la console", ("fixation", "console", "section"), cas, ["plats", "creux", "froid", "ronds"],
                       help="Plat vertical (h vertical) soudé sur la platine au mur ; longueur e dans l'onglet Garde-corps.")
        with c2:
            _num("Longueur e de la console (mm)", ("fixation", "console", "e"), cas, 10.0, "%.0f", 20.0,
                 help="Du mur à l'axe de la lisse : bras de levier de la charge verticale sur la console et sur ses chevilles.")
    if typ == "scellee":
        st.caption("Montant scellé dans le béton ou dans un fourreau : la fixation n'est pas vérifiée ici.")
        return
    st.markdown("**Soudure sur la platine** — vérifiée par le module Soudures")
    soud = fx["soudure"]
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
        col = "gc-r" if s["taux"] > 1 else ("gc-o" if s["taux"] > 0.85 else "gc-g")
        st.markdown(f"Soudure : <span class='{col}'>η = {s['taux'] * 100:.0f} %</span> — {s['critere']}, {s['position']} · {s['efforts_txt']}", unsafe_allow_html=True)
        vues = SCS.vues(s["res"])
        if vues:
            cols = st.columns(min(2, len(vues)), gap="small")
            for i, (cle, titre, d) in enumerate(vues[:2]):
                with cols[i]:
                    with st.container(key=P + f"soud_{i}"):
                        st.markdown(f'<div class="gc-leg"><b>{titre.upper()}</b></div>' + SCS.svg(d, f"gc_s{i}"), unsafe_allow_html=True)
    if typ != "platine":
        return
    st.markdown("**Platine et ancrage** — platine rigide, deux rangées de boulons de part et d'autre du montant")
    with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=P + "c_pl"):
        _num("b_p — dans le sens de l'effort (mm)", ("fixation", "platine", "bp"), cas, 5.0, "%.0f", 40.0, width=230)
        _num("l_p — largeur (mm)", ("fixation", "platine", "lp"), cas, 5.0, "%.0f", 40.0, width=170)
        _num("e_x — entraxe des deux rangées (mm)", ("fixation", "ancrage", "ex"), cas, 5.0, "%.0f", 10.0, width=250, help="Dans le sens de l'effort ; doit dépasser la dimension h_p du montant.")
        _num("Boulons par rangée", ("fixation", "ancrage", "n_par_rangee"), cas, 1, min_value=1, max_value=4, width=150)
        _num("e_y — entraxe dans la rangée (mm)", ("fixation", "ancrage", "ey"), cas, 5.0, "%.0f", 10.0, width=230)
    anc = fx["ancrage"]
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
        st.markdown(f"Bras de levier z = {_fmt(pl['z'])} mm · traction de la rangée Z = {_fmt(pl['Z'] / 1e3, 2)} kN → F_t = {_fmt(pl['Ft'] / 1e3, 2)} kN par boulon "
                    f"(≤ {_fmt(pl['FtRd'] / 1e3, 2)}) : <span class='{_cls(pl['taux_t'])}'>{pl['taux_t'] * 100:.0f} %</span> · F_v = {_fmt(pl['Fv'] / 1e3, 2)} kN "
                    f"(≤ {_fmt(pl['FvRd'] / 1e3, 2)}) : <span class='{_cls(pl['taux_v'])}'>{pl['taux_v'] * 100:.0f} %</span> · interaction "
                    f"<span class='{_cls(pl['inter'])}'>{pl['inter'] * 100:.0f} %</span><br>Platine : bras m = {_fmt(pl['m'])} mm, b_eff = {_fmt(pl['b_eff'], 0)} mm, "
                    f"σ_p = {_fmt(pl['sigma_p'])} N/mm² ≤ {_fmt(pl['lim_sigma'], 0)} : <span class='{_cls(pl['taux_platine'])}'>{pl['taux_platine'] * 100:.0f} %</span> "
                    f"— épaisseur requise {_fmt(pl['tp_req'])} mm (t_p = {_fmt(pl['tp'], 0)})", unsafe_allow_html=True)
    elif pl:
        for m in pl.get("messages", []):
            st.warning(m)


# ---------------------------------------------------------------- onglet 6 : détail
def _etape_detail(etat: Dict, cas: Dict, res: Dict):
    if res["statut"] == ET.REFUS and not res.get("elements"):
        st.warning((res["messages"] or ["Calcul impossible"])[0])
    lignes = ""
    g = res.get("gouvernant") or {}
    for e in res.get("elements", []):
        d = e["detail"]
        cls = " class='g'" if e["cle"] == g.get("cle") else ""
        if e["cle"] in ("montant", "lisse", "barreau", "lisse_inter", "console"):
            M = max(abs(d.get("M1", 0.0)), abs(d.get("Mv", 0.0)), abs(d.get("Mh", 0.0)))
            lignes += (f"<tr{cls}><td>{e['libelle']}</td><td class='n'>{_fmt(M / 1e6, 3)}</td><td class='n'>{_fmt(max(abs(d.get('V1', 0.0)), abs(d.get('V2', 0.0))) / 1e3, 2)}</td>"
                       f"<td class='n'>{_fmt(d['sigma'])}</td><td class='n'>{_fmt(d['tau'], 2)}</td><td class='n'>{_fmt(d['fleche'], 2)} / {_fmt(d['fleche_lim'], 1)}</td>"
                       f"<td class='n {_cls(e['taux'])}'>{e['taux'] * 100:.0f} %</td><td>{ME.critere_texte(e['critere'])}</td></tr>")
        else:
            lignes += (f"<tr{cls}><td>{e['libelle']}</td><td class='n'>—</td><td class='n'>—</td><td class='n'>—</td><td class='n'>—</td><td class='n'>—</td>"
                       f"<td class='n {_cls(e['taux'])}'>{e['taux'] * 100:.0f} %</td><td>{e['critere']}</td></tr>")
    st.markdown("<table class='gc-tab'><tr><th>Élément</th><th class='n'>M (kN·m)</th><th class='n'>V (kN)</th><th class='n'>σ (N/mm²)</th><th class='n'>τ</th>"
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
        for titre, cle in (("Montant", "montant"), ("Console", "console"), ("Lisse", "lisse"), ("Remplissage", "barre"), ("Platine et ancrage", "platine")):
            if fo[cle]:
                st.markdown(f"**{titre}**  \n" + "  \n".join(FO.rendre(s) for s in fo[cle]))
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
                _num("σ_adm (N/mm²)", ("methode", "sigma_adm"), cas, 5.0, "%.0f", 10.0, width=150)
                _num("τ_adm (N/mm²)", ("methode", "tau_adm"), cas, 5.0, "%.0f", 5.0, width=150)
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
        for sec in MT.SECTIONS:
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
            _etape_remplissage(cas, res)
        with onglets[4]:
            _etape_fixation(cas, res)
        with onglets[5]:
            _etape_detail(etat, cas, res)

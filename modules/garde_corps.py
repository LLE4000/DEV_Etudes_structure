# -*- coding: utf-8 -*-
"""Écran Streamlit du module « Garde-corps » (version 3.2 : la disposition refondue).

Le moteur, l'état, les dessins et la note vivent dans ``acier/garde_corps`` (sans
Streamlit : ``tests/test_garde_corps_moteur.py`` le garde) ; ce fichier n'est que la vue.

Organisation — PLUS D'ONGLETS : barre d'outils du cas ; le SWITCH de méthode (« Vérification
à l'ELS » par défaut | « Eurocode (ELU) ») avec ses paramètres en popover ; puis deux colonnes.
À GAUCHE ce que le calcul PRODUIT : l'ÉLÉVATION, toujours ; UN SEUL dessin de détail à la fois,
au choix (coupe A-A, plan, platine, fixation, section) ; la carte RÉSULTAT (statut, η, l'élément
gouvernant, son critère et son cas) qui porte UNE LIGNE PAR ÉLÉMENT — son taux en barre et, pour
chaque élément dimensionné, SA SECTION à choisir sur place et le bouton « ★ optimal » qui nomme le
plus léger qui tient et ouvre en popover le tableau des profilés compatibles (un clic retient).
Un CODE COULEUR des taux, le même partout (pastilles, barres, cellules du tableau) : rouge au-dessus
de 100 %, vert foncé de 95 à 100 %, vert pâle en dessous — les trois statuts du moteur.
À DROITE ce qu'on SAISIT, en trois cartes toujours visibles : DÉFINITION (le type en menu, la
forme, H, la pente, les tronçons, les extrémités, le remplissage), CHARGES (les quatre charges
nommées en cas distincts, le cumul, les options), FIXATION (platine, latérale, soudée, scellée ;
console, soudure, platine, ancrage, et le résumé chiffré). En BAS, repliés : les efforts, les
formules, l'identification de la note, la méthode. Au téléphone les colonnes s'empilent :
dessins, résultat, puis la saisie.
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
             "console": "Plat vertical (h vertical) soudé sur la platine au mur ; longueur e dans la carte Fixation."}
# Les libellés courts du sélecteur de fixation (les libellés complets, FX.TYPES, vont dans la note et l'aide)
FIXATIONS_COURTES = {"platine": "platine sur la dalle", "laterale": "platine latérale (nez de dalle)", "soudee": "soudé sur plat existant", "scellee": "scellé (non vérifié)"}
# Le CODE COULEUR des taux, tranché par le bureau le 06/11/2026 : ROUGE au-dessus de 100 %, VERT FONCÉ de 95 à 100 %, VERT PÂLE en dessous.
# Les trois niveaux SONT les trois statuts du moteur (membres.statut, seuils SEUIL_ATTENTION / SEUIL_NOK : source unique) ; ici, les couleurs.
NIVEAU = {ME.NOK: "r", ME.ATTENTION: "f", ME.OK: "p"}
COULEUR_BARRE = {"r": "#dc2626", "f": "#15803d", "p": "#86efac"}
COULEUR_CELLULE = {"r": ("#dc2626", "#ffffff"), "f": ("#15803d", "#ffffff"), "p": ("#dcfce7", "#166534")}       # (fond, texte) des cellules du tableau
CODE_COULEUR = (f"rouge au-dessus de {ME.SEUIL_NOK * 100:.0f} %, vert foncé de {ME.SEUIL_ATTENTION * 100:.0f} à {ME.SEUIL_NOK * 100:.0f} %, "
                "vert pâle en dessous")
# Le dessin de détail : un seul à la fois, choisi par un sélecteur ; libellé court du sélecteur, légende sous le dessin
VUES_DETAIL = {"coupe": "Coupe A-A", "plan": "Plan", "platine": "Platine", "pied": "Fixation", "section": "Section"}
LEGENDES_DETAIL = {"coupe": "", "plan": "", "section": "élément principal", "pied": "coupe à grande échelle, efforts du cas gouvernant",
                   "platine": "vue de face (platine latérale, console au mur) ou de dessus (platine sur la dalle)"}
CSS = """<style>
.gc-res{padding:0 0 4px}
.gc-res .t,.gc-kt{font-size:11px;letter-spacing:.08em;color:#6b7280;font-weight:600}
.gc-kt{margin:0 0 2px}
.gc-pill{float:right;font-size:12px;border-radius:999px;padding:3px 10px;font-weight:600}
.gc-pill.nok{background:#dc2626;color:#fff}.gc-pill.ok{background:#dcfce7;color:#166534}.gc-pill.att{background:#15803d;color:#fff}.gc-pill.ref{background:#f3f4f6;color:#374151}
.gc-eta{font-size:26px;font-weight:700;margin:4px 0 0;line-height:1.1}
.gc-chip{display:inline-block;min-width:46px;text-align:center;border-radius:6px;padding:1px 6px;font-weight:700;font-variant-numeric:tabular-nums;font-size:12px;line-height:16px}
.gc-chip.r{background:#dc2626;color:#fff}.gc-chip.f{background:#15803d;color:#fff}.gc-chip.p{background:#dcfce7;color:#166534}
.gc-chip.big{font-size:22px;line-height:30px;padding:2px 12px;border-radius:8px;min-width:0}
.gc-phrase{font-size:13.5px;margin:4px 0 6px}
.gc-row{display:flex;flex-wrap:wrap;align-items:center;column-gap:10px;row-gap:3px;font-size:12.5px;margin:14px 0 4px}
.gc-row>span:first-child{flex:1 1 230px;min-width:0}.gc-row .gc-trk{flex:0 0 150px}.gc-row .gc-v{flex:0 0 56px}
.gc-row.first{margin-top:8px}
.gc-trk{height:6px;background:#f1f3f5;border-radius:3px;overflow:hidden}.gc-trk b{display:block;height:100%}
.gc-v{text-align:right;font-variant-numeric:tabular-nums;font-weight:600}
.st-key-gc_k_res [data-testid="stMarkdownContainer"]{margin-bottom:0!important}
.gc-note{font-size:12px;color:#6b7280;margin-top:6px}
.gc-tab{width:100%;border-collapse:collapse;font-size:12.5px}.gc-tab th{text-align:left;font-size:10.5px;letter-spacing:.05em;color:#6b7280;padding:4px 6px;border-bottom:1px solid #e5e7eb}
.gc-tab td{padding:4px 6px;border-bottom:1px solid #f3f4f6;vertical-align:top}.gc-tab td.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.gc-tab tr.g td{font-weight:700;background:#fff7ed}.gc-tab tr.cur td{font-weight:700;background:#eff6ff}
.gc-g{color:#15803d;font-weight:700}.gc-i{color:#6b7280;font-weight:400}
.gc-leg{font-size:11.5px;color:#6b7280}
.gc-switch{font-size:12.5px;color:#374151}
.gc-ch{display:grid;grid-template-columns:repeat(auto-fit,minmax(125px,1fr));gap:4px 12px;font-size:12px;margin:4px 0 2px;color:#374151}
.gc-ch .n{font-size:15px;font-weight:700;color:#33415c;margin-right:6px}.gc-ch .s{font-weight:700}.gc-ch .o{font-size:11px;color:#6b7280}
.gc-opt{font-size:13px;color:#1f2937;padding:2px 0}
.gc-fix{font-size:12.5px;color:#374151;margin-top:2px}
.st-key-gc_dessin2 svg{max-height:380px}
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


def _choix(label, chemin, cas, options, affichage=None, help=None, segmente=False, width=None, on_change=None, label_visibility="visible"):
    """Un choix dans l'état : menu déroulant, ou segments (qui passent à la ligne sur un écran étroit)."""
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
        st.segmented_control(label, options, default=cur, format_func=fmt, key=cle, on_change=_on, help=help, selection_mode="single",
                             label_visibility=label_visibility, wrap=True, **kw)
    else:
        st.selectbox(label, options, index=options.index(cur), format_func=fmt, key=cle, on_change=_on, help=help, label_visibility=label_visibility, **kw)
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


def _niveau(t) -> str:
    """Le niveau du code couleur d'un taux : « r » rouge, « f » vert foncé, « p » vert pâle — par le statut du moteur."""
    return NIVEAU.get(ME.statut(t), "p")


def _chip(t, big: bool = False) -> str:
    """Un taux en pastille colorée (« 115 % » rouge, « 97 % » vert foncé, « 64 % » vert pâle) ; ``big`` : le η de la carte."""
    return f'<span class="gc-chip {_niveau(t)}{" big" if big else ""}">{"η = " if big else ""}{t * 100:.0f} %</span>'


def _couleur(t) -> str:
    """La teinte de la barre d'un taux."""
    return COULEUR_BARRE[_niveau(t)]


def _style_taux(v) -> str:
    """Le style d'une cellule de taux (en % entier) du tableau des profilés — le même code couleur."""
    try:
        bg, fg = COULEUR_CELLULE[_niveau(float(v) / 100.0)]
    except (TypeError, ValueError):
        return ""
    return f"background-color: {bg}; color: {fg}; font-weight: 700"


def _noms_sections(groupes: Optional[List[str]]) -> List[str]:
    """Les noms des sections standard des groupes, dans l'ordre du catalogue (famille, taille)."""
    return [s["nom"] for s in BI.standards().values() if groupes is None or BI.groupe_de(s) in groupes]


def _section_w(label, chemin, cas, groupes=None, help=None, width=None, label_visibility="visible", dims=True) -> Dict:
    """Le choix d'une section : la liste des sections standard des groupes (recherche en tapant),
    ou « section libre » avec sa forme et ses dimensions (``_dims_libres_w`` — à la suite si
    ``dims``, sinon là où l'appelant le décide). L'état garde la spécification ; rend-la."""
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
    st.selectbox(label, options, index=options.index(valeur), key=cle, on_change=_on, help=help, label_visibility=label_visibility, **kw)
    if dims and not spec.get("nom"):
        _dims_libres_w(chemin, cas)
    return spec


def _dims_libres_w(chemin, cas):
    """La forme et les dimensions d'une section libre (sous le sélecteur qui l'a choisie)."""
    d, k = _chemin(cas, chemin)
    spec = d.get(k) or {}
    if spec.get("nom"):
        return
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
                     on_change=_fam, width=160)
        for dim in BI.NON_STANDARD[fam][1]:
            _num(PF.LIBELLE_DIM[dim], chemin + ("dims", dim), cas, 0.5, "%.1f", 0.5, width=110)
    st.caption("Section non standard : ses propriétés sont calculées de ses dimensions par la même fonction que le catalogue (h dans le sens de l'effort = axe fort).")


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
    st.text_input("Entraxes saisis (mm, facultatif)", value=" ".join(f"{float(e):g}" for e in (d.get(k) or [])), key=cle, on_change=_on, width=210,
                  help="Par exemple « 1200 900 » : un poteau central à 1,20 m du premier et 0,90 m du dernier. Vide : la longueur est répartie "
                       "régulièrement à l'entraxe maximal.")


def _titre_carte(texte: str):
    st.markdown(f'<div class="gc-kt">{texte}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- en-tête, barre d'outils, switch
def _entete(etat: Dict, cas: Dict):
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("## Garde-corps")
    st.caption("Dimensionnement d'un garde-corps acier — type en menu (barreaudage, lisses, panneau, main courante seule, main courante murale), "
               "forme en plan droite, en L ou en U, entraxes libres, quatre charges nommées (EN 1991-1-1 Tab. 6.12, NBN B 03-004) en cas distincts, "
               "montant principal, main courante sur deux appuis ou continue, barreaux, ancrage (soudure, chevilles ou boulons, platine sur la dalle ou "
               "en nez de dalle) · vérification à l'ELS par défaut, Eurocode en option · version 3.2")
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
                                                                        "remplissage, ancrage avec les formules développées ; détails, soudure, méthode "
                                                                        "(une page de suite quand une colonne ne tient pas)"):
            ident = etat.get("identification") or {}
            st.session_state[P + "note"] = (cas["nom"], NT.generer_note(etat, cas, {"nom_projet": ident.get("projet", ""), "partie": ident.get("partie", ""),
                                                                                   "indice": ident.get("indice", "0")}))
        _texte_w("Nom du cas", ("nom",), cas, width=260)
    note = st.session_state.get(P + "note")
    if note and note[0] == cas["nom"]:
        st.download_button("⬇️ Télécharger la note (PDF)", data=note[1], file_name=f"note_garde_corps_{cas['nom']}.pdf",
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


def _parametres(cas: Dict):
    """Les paramètres de la méthode (nuance, E, flèche, coefficients) — dans le popover du switch."""
    m = cas["methode"]
    with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=P + "c_meth1"):
        _choix("Nuance", ("methode", "nuance"), cas, list(ET.nuances()), width=130)
        _num("E (N/mm²)", ("methode", "E"), cas, 1000.0, "%.0f", 100000.0, width=140)
        _choix("Flèche admissible", ("methode", "fleche_mode"), cas, ["L/x", "mm"], {"L/x": "L / x", "mm": "valeur en mm"}, width=150)
        _num("x, ou la valeur (mm)", ("methode", "fleche_val"), cas, 10.0, "%.0f", 1.0, width=160,
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


def _switch(cas: Dict):
    with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="center", key=P + "c_switch"):
        _choix("Méthode", ("methode", "mode"), cas, [ME.ELS, ME.EUROCODE], MODES, segmente=True, width="content", label_visibility="collapsed")
        with st.popover("Paramètres", key=P + "pop_meth", width="content", help="Nuance, module E, flèche admissible, coefficients de la méthode"):
            _parametres(cas)
    m = cas["methode"]
    fl = f"L/{m.get('fleche_val', 100):g}" if m.get("fleche_mode", "L/x") == "L/x" else f"{m.get('fleche_val', 0):g} mm"
    if m["mode"] == ME.ELS:
        lim = ME.limites(m, ET.acier(m)["fy"])
        st.markdown(f'<div class="gc-switch"><b>Charges de service</b>, sans coefficient, contraintes admissibles dérivées de la nuance — '
                    f'σ = M/W + N/A ≤ σ_adm = f_y/{lim["coef"]:g} = <b>{_fmt(lim["sigma"])} N/mm²</b>, τ = V/A_v ≤ τ_adm = σ_adm/√3 = '
                    f'<b>{_fmt(lim["tau"])} N/mm²</b>, σ_eq = √(σ² + 3τ²) ≤ σ_adm, flèche ≤ <b>{fl}</b> ; soudure '
                    f'à l\'ELS (k = {m.get("k", 100):g} kg/(mm·cm)), chevilles et boulons à leur résistance de calcul / γ_F = {m.get("gamma_F", 1.5):g} '
                    f'(bouton Paramètres).</div>'.replace("f_y/1.5", "f_y/1,5"), unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="gc-switch"><b>Efforts de calcul (ELU)</b> : charges × γ_Q = {m.get("gamma_Q", 1.5):g}, σ ≤ f_y/γ_M0 (élastique, W_el), '
                    f'τ ≤ f_y/(√3·γ_M0), flèche ≤ {fl} sous charges de service ; soudure EN 1993-1-8 directionnelle, boulons Tab. 3.4, chevilles à '
                    f'leurs valeurs de calcul ETA (bouton Paramètres).</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- colonne de gauche : les dessins
def _dessins(cas: Dict, res: Dict):
    """L'élévation, toujours ; puis UN dessin de détail au choix (sélecteur ``gc_vue``)."""
    vues = {k: (t, d) for k, t, d in SC.vues(res)}
    if not vues:
        with st.container(key=P + "dessin"):
            st.warning("Calcul impossible — " + (res.get("messages") or ["données manquantes"])[0])
        return
    t, d = vues["elevation"]
    with st.container(key=P + "dessin"):
        st.markdown(f'<div class="gc-leg" style="margin:2px 0"><b>{t.upper()}</b> · {res["type_libelle"]} · {res["geometrie"]["phrase"]}</div>'
                    + SC.svg(d, "gc_v0"), unsafe_allow_html=True)
    opts = [k for k in VUES_DETAIL if k in vues]
    cle = P + "vue"
    if st.session_state.get(cle) not in opts:
        st.session_state[cle] = opts[0]
    vue = st.segmented_control("Dessin de détail", opts, format_func=lambda k: VUES_DETAIL[k], key=cle, selection_mode="single", required=True,
                               label_visibility="collapsed", width="content", wrap=True, help="Un seul dessin de détail à la fois ; la note les imprime tous.")
    if vue not in vues:
        vue = opts[0]
    t, d = vues[vue]
    leg = LEGENDES_DETAIL.get(vue, "")
    with st.container(key=P + "dessin2"):
        st.markdown(f'<div class="gc-leg" style="margin:2px 0"><b>{t.upper()}</b>{" · " + leg if leg else ""}</div>' + SC.svg(d, "gc_v1"),
                    unsafe_allow_html=True)
    st.markdown('<div class="gc-leg">Dessins à l\'échelle, en mm : pièces au trait fort, sections coupées hachurées, cordons en ocre à leur taille a·√2, '
                'chevilles en gris (tige cachée en traits interrompus), ◉ l\'élément gouvernant et son taux, flèches rouges dans le sens des charges '
                '(⊙ vers soi), A-A le plan de la coupe.</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- colonne de gauche : résultat et éléments
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


def _popover_profils(cas: Dict, res: Dict, role: str):
    """Le bouton « ★ optimal » d'un élément : il NOMME le plus léger qui tient (familles cochées,
    masse maximale) et ouvre le tableau des profilés compatibles — un CLIC sur une ligne retient la
    section. Les réglages vivent dans le popover ; on les lit avant pour écrire le libellé."""
    lib = ET.ROLES[role]
    k_fam, k_m, k_ok, k_fr = P + f"lib_{role}_fam", P + f"lib_{role}_masse", P + f"lib_{role}_ok", P + f"lib_{role}_froid"
    fams = st.session_state.get(k_fam, FAMILLES_DEFAUT[role]) or []
    masse_max = float(st.session_state.get(k_m, 15.0) or 15.0)
    seuls = bool(st.session_state.get(k_ok, False))
    froid = bool(st.session_state.get(k_fr, False))
    cands = [c for c in ET.candidats(cas, role, _groupes_de(fams, froid), masse_max) if _de_famille(c["section"], fams)] if fams else []
    best = ET.plus_leger(cands)
    cur = (ET.spec_du_role(cas, role) or {}).get("nom")
    nom_best = best["section"]["nom"] if best is not None else None
    retenu = nom_best is not None and nom_best == cur
    if best is not None:
        label = "★ optimal retenu" if retenu else f"★ {nom_best} · {best['taux'] * 100:.0f} %"
    else:
        label = "★ aucun ne tient" if cands else "★ profilés"
    with st.popover(label, key=P + f"pop_{role}", width="content",
                    help=f"Le plus léger qui tient pour {lib} parmi les familles cochées ; ouvrir pour le tableau des profilés compatibles"):
        st.markdown(f'<div class="gc-opt"><b>{lib.capitalize()}</b> — chaque section standard des familles cochées, vérifiée dans ce rôle '
                    f'(σ, τ, σ_eq, flèche), par masse croissante ; un clic sur une ligne la retient.</div>', unsafe_allow_html=True)
        with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=P + f"c_lib_{role}"):
            st.pills("Familles", list(FAMILLES_LIB), default=FAMILLES_DEFAUT[role], format_func=lambda f: FAMILLES_LIB[f], key=k_fam, selection_mode="multi")
            st.number_input("Masse maxi (kg/m)", value=15.0, step=1.0, min_value=1.0, key=k_m, width=130)
            st.checkbox("Seulement ceux qui tiennent", value=False, key=k_ok)
            st.checkbox("Tubes formés à froid", value=False, key=k_fr)
        if not fams:
            st.caption("Choisir au moins une famille.")
            return
        if best is not None:
            with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="center", key=P + f"c_opt_{role}"):
                st.markdown(f'<div class="gc-opt">★ Optimal pour {lib} : <b>{nom_best}</b> · {_fmt(best["section"]["masse"], 2)} kg/m · η {_chip(best["taux"])} '
                            f'({ME.critere_texte(best["critere"])})' + (' · <span class="gc-g">retenu</span>' if retenu else '') + '</div>', unsafe_allow_html=True)
                if st.button("Prendre l'optimal", key=P + f"btn_best_{role}", width="content", disabled=retenu,
                             help="Retient la section la plus légère qui tient, parmi les familles cochées et sous la masse maximale."):
                    ET.appliquer_section(cas, best["section"], role)
                    _changer(f"{nom_best} retenu pour {lib}")
        elif cands:
            st.warning("Aucune section de la sélection ne tient : élargir les familles ou la masse maximale.")
        montres = [c for c in cands if not seuls or c["taux"] <= 1.0][:40]
        if not montres:
            st.caption("Aucune ligne à montrer.")
            return
        df = pd.DataFrame([{"Section": c["section"]["nom"], "dimensions (mm)": BI.dimensions_texte(c["section"]), "kg/m": round(c["masse"], 2),
                            "η σ_eq": int(round(min(c["taux_eq"], 9.99) * 100)), "η flèche": int(round(min(c["taux_fleche"], 9.99) * 100)),
                            "η": int(round(min(c["taux"], 9.99) * 100)),
                            "retenue": ("◀ " if c["section"]["nom"] == cur else "") + ("★" if c["section"]["nom"] == nom_best else "")} for c in montres])
        # largeurs en pixels : le nom complet de la section se lit, et les sept colonnes tiennent dans le popover ;
        # les trois colonnes de taux sont des cellules colorées par le code couleur (Styler pandas → st.dataframe)
        cfg = {"Section": st.column_config.TextColumn("Section", width=125), "dimensions (mm)": st.column_config.TextColumn("dimensions (mm)", width=92),
               "kg/m": st.column_config.NumberColumn("kg/m", format="%.2f", width=52),
               "η σ_eq": st.column_config.NumberColumn("η σ_eq", format="%d %%", width=62), "η flèche": st.column_config.NumberColumn("η flèche", format="%d %%", width=66),
               "η": st.column_config.NumberColumn("η", format="%d %%", width=62), "retenue": st.column_config.TextColumn("", width=36)}
        colonnes_taux = ["η σ_eq", "η flèche", "η"]
        ev = st.dataframe(df.style.map(_style_taux, subset=colonnes_taux), hide_index=True, column_config=cfg, key=P + f"tab_{role}_{LI.generation(P)}",
                          on_select="rerun", selection_mode="single-row", height=min(300, 35 * (len(df) + 1) + 12), width=560)
        rows = list(ev.selection.rows) if ev is not None and hasattr(ev, "selection") else []
        if rows:
            sec = montres[rows[0]]["section"]
            if sec["nom"] != cur:
                ET.appliquer_section(cas, sec, role)
                _changer(f"{sec['nom']} retenu pour {lib}")
        st.caption(f"{len(cands)} sections essayées par masse croissante ({len(montres)} montrées) ; ◀ la section retenue, ★ l'optimal ; "
                   f"taux : {CODE_COULEUR} ; la fixation est vérifiée pour la section retenue.")


def _ligne_html(e: Dict, gouv: bool, first: bool = False) -> str:
    """La ligne d'un élément : son nom en gras, le reste de son libellé, le critère et le cas en gris, la barre η."""
    parts = e["libelle"].split(" · ")
    d = e["detail"] if isinstance(e.get("detail"), dict) else {}
    cas_c = CH.CAS_LIBELLES.get(d.get("cas_charge"), d.get("cas_charge")) if d.get("cas_charge") else ""
    dimensionne = e["cle"] in ET.ROLES
    crit = ME.critere_texte(e["critere"]) if dimensionne else e["critere"]
    infos = parts[2:] if dimensionne else parts[1:]     # pour un élément dimensionné, la section (parts[1]) se lit dans le sélecteur
    t = e["taux"]
    col = _couleur(t)
    gris = " · ".join(infos + [crit] + ([cas_c] if cas_c else []))
    return (f'<div class="gc-row{" sel" if gouv else ""}{" first" if first else ""}"><span><b>{parts[0]}</b>{" ◀" if gouv else ""} '
            f'<span class="gc-i">{gris}</span></span><div class="gc-trk"><b style="width:{min(t, 1) * 100:.0f}%;background:{col}"></b></div>'
            f'<span class="gc-v">{_chip(t)}</span></div>')


def _ligne_role(cas: Dict, res: Dict, role: str, e: Optional[Dict], gouv: bool, first: bool):
    """Un élément dimensionné : sa ligne (nom, critère, barre), puis sa SECTION à choisir, le
    modèle d'appuis pour la main courante, et le bouton « ★ optimal »."""
    if e is not None:
        st.markdown(_ligne_html(e, gouv, first), unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="gc-row{" first" if first else ""}"><span><b>{ET.LIBELLES[role]}</b> <span class="gc-i">non vérifié</span></span><div></div><span></span></div>',
                    unsafe_allow_html=True)
    with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="center", key=P + f"c_el_{role}"):
        spec = _section_w("Section", _chemin_role(role), cas, GROUPES_SECTIONS[role], help=AIDE_ROLE[role], width=200, label_visibility="collapsed", dims=False)
        if role == "lisse":
            _choix("Modèle", ("lisse", "appuis"), cas, list(ME.APPUIS), {"deux": "2 appuis", "continue": "continue"}, segmente=True, width="content",
                   label_visibility="collapsed",
                   help="Sur deux appuis (par défaut : les lisses sont rarement continues d'un montant à l'autre) : M = qL²/8 ; continue sur n travées : "
                        "coefficients des tables (0,125 / 0,100 / 0,107 qL²).")
        _popover_profils(cas, res, role)
    if not spec.get("nom"):
        _dims_libres_w(_chemin_role(role), cas)


def _resultat(cas: Dict, res: Dict):
    """La carte RÉSULTAT : le statut, η, l'élément gouvernant ; puis une ligne par élément, la
    section des éléments dimensionnés à choisir sur place."""
    gen = LI.generation(P)
    for k in [k for k in list(st.session_state) if isinstance(k, str) and k.startswith(P + "tab_") and not k.endswith(f"_{gen}")]:
        del st.session_state[k]
    stat = res["statut"]
    pill = {ET.OK: ("ok", "● Conforme"), ET.ATTENTION: ("att", "● Conforme, attention"), ET.NOK: ("nok", "● Non conforme"),
            ET.REFUS: ("ref", "● Calcul impossible")}[stat]
    g = res.get("gouvernant")
    if stat == ET.REFUS or not g:
        eta = "—"
        phrase = (res["messages"] or ["Données manquantes"])[0]
    else:
        eta = _chip(res["taux_max"], big=True)
        crit = ME.critere_texte(g["critere"]) if g["cle"] not in ("soudure", "ancrage", "platine") else g["critere"]
        cas_g = (next((e for e in res["elements"] if e["cle"] == g["cle"]), {}).get("detail") or {}).get("cas_charge")
        phrase = f"<b>{g['libelle']}</b> — {crit}" + (f" · {CH.CAS_LIBELLES.get(cas_g, cas_g)}" if cas_g else "")
    m = cas["methode"]
    ligne_m = f"{res['type_libelle']} · {MODES[m['mode']]} · {CH.phrase_cumul(res['charges'])}"
    with st.container(border=True, gap=None, key=P + "k_res"):
        st.markdown(f'<div class="gc-res"><span class="gc-pill {pill[0]}">{pill[1]}</span><div class="t">RÉSULTAT</div>'
                    f'<div class="gc-eta">{eta}</div><div class="gc-phrase">{phrase}</div><div class="gc-note">{ligne_m}</div></div>', unsafe_allow_html=True)
        elements = {e["cle"]: e for e in res.get("elements", [])}
        roles = ET.roles_du_cas(cas)
        st.markdown('<div class="gc-kt" style="margin-top:12px">ÉLÉMENTS — la section se choisit sur place, ★ nomme le plus léger qui tient et ouvre le tableau</div>',
                    unsafe_allow_html=True)
        for i, role in enumerate(roles):
            _ligne_role(cas, res, role, elements.get(role), bool(g and g["cle"] == role), i == 0)
        autres = "".join(_ligne_html(e, bool(g and g["cle"] == e["cle"])) for e in res.get("elements", []) if e["cle"] not in roles)
        if res.get("masse"):
            autres += f'<div class="gc-note" style="margin-top:12px">Masse indicative de l\'acier : {_fmt(res["masse"], 0)} kg · {res["geometrie"]["phrase"]}</div>'
        autres += f'<div class="gc-note">Code couleur des taux : {CODE_COULEUR}.</div>'
        st.markdown(autres, unsafe_allow_html=True)
    for msg in res.get("messages", []):
        if msg.startswith("Refus") or "DÉPASSÉ" in msg:
            st.warning(msg)
        else:
            st.caption("• " + msg)


# ---------------------------------------------------------------- colonne de droite : définition
def _carte_definition(cas: Dict, res: Dict):
    g = cas["geometrie"]

    def _type():
        c = ET.cas_actif(_etat())
        TY.appliquer_type(c, c.get("type"))
        _reset_widgets()
    with st.container(border=True, key=P + "k_def"):
        _titre_carte("DÉFINITION")
        typ = _choix("Type de garde-corps", ("type",), cas, list(TY.TYPES), {k: v["libelle"] for k, v in TY.TYPES.items()}, on_change=_type,
                     help="Le type fixe le système (montants verticaux sur platine, ou main courante portée par des consoles au mur) et le remplissage ; "
                          "tout le reste se règle ensuite.")
        st.caption(TY.TYPES[typ]["description"])
        with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=P + "c_sys"):
            _choix("Forme en plan", ("geometrie", "forme"), cas, list(GE.FORMES), GE.FORMES, segmente=True, width="content",
                   help="Un, deux ou trois tronçons droits ; un montant d'angle reçoit une demi-travée de chaque tronçon dans deux directions.")
            _num("H, main courante (mm)", ("geometrie", "H"), cas, 10.0, "%.0f", 300.0, width=150,
                 help="Bras de levier de la charge horizontale : hauteur de l'axe de la main courante au-dessus de la platine (ou du dessus de dalle pour une "
                      "platine latérale : le montant est alors vérifié sur H + d_h). La charge s'applique au plus à 1,20 m (EN 1991-1-1 §6.4).")
            _num("Pente α (°)", ("geometrie", "pente"), cas, 1.0, "%.0f", 0.0, max_value=60.0, width=110,
                 help="Garde-corps d'escalier (0 : horizontal) : les entraxes se mesurent le long de la main courante, les montants restent verticaux ; "
                      "la pente incline l'élévation et se lit dans les hypothèses, elle ne change pas les formules.")
        n = GE.N_TRONCONS.get(g.get("forme", "droit"), 1)
        for i in range(n):
            with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=P + f"c_tr{i}"):
                _num(f"Tronçon {i + 1} : L (mm)", ("geometrie", "troncons", i, "L"), cas, 100.0, "%.0f", 100.0, width=150)
                _num("Entraxe maxi (mm)", ("geometrie", "troncons", i, "s_max"), cas, 50.0, "%.0f", 200.0, width=150,
                     help="La longueur est répartie en travées égales ne dépassant pas cet entraxe — sauf si des entraxes sont saisis à côté.")
                _entraxes_w(i, cas)
        with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=P + "c_ext"):
            _choix("Début", ("geometrie", "debut"), cas, list(GE.EXTREMITES), GE.EXTREMITES, width=220,
                   help="Un montant d'extrémité, ou la main courante ancrée dans le mur : le mur reprend la demi-travée. Les deux bouts ancrés et une "
                        "seule travée : garde-corps SANS montant, la main courante porte de mur à mur (loggia).")
            _choix("Fin", ("geometrie", "fin"), cas, list(GE.EXTREMITES), GE.EXTREMITES, width=220)
        rp = cas["remplissage"]
        rpt = rp.get("type")
        if rpt == "barreaux":
            st.markdown("**Barreaudage** — barreaux verticaux entre la lisse basse et la main courante")
            with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=P + "c_rp"):
                _num("Entraxe e_b (mm)", ("remplissage", "entraxe"), cas, 5.0, "%.0f", 20.0, width=140,
                     help="Un barreau reprend la pression q_r × e_b (cas 4) ou la charge ponctuelle Q_r à mi-portée (cas 3) ; le pire gouverne.")
                _num("h_bas (mm)", ("remplissage", "h_bas"), cas, 10.0, "%.0f", 0.0, width=120, help="Axe de la lisse basse au-dessus de la platine.")
                _section_w("Lisse basse (non calculée)", ("remplissage", "section_basse"), cas, ["creux", "froid", "plats", "ronds"], width=220, dims=False,
                           help="Elle fixe la portée des barreaux et n'est pas vérifiée.")
            _dims_libres_w(("remplissage", "section_basse"), cas)
            r = res.get("remplissage") or {}
            if r.get("L"):
                st.caption(f"Portée des barreaux déduite : {_fmt(r['L'], 0)} mm (entre le dessus de la lisse basse et le dessous de la main courante) ; "
                           f"la section des barreaux se choisit dans le résultat.")
        elif rpt == "lisses":
            st.markdown("**Lisses intermédiaires** — horizontales, entre montants, sous la main courante")
            with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=P + "c_rp"):
                _num("Nombre de lisses", ("remplissage", "n_lisses"), cas, 1, min_value=1, width=140)
                _num("Entraxe e_l (mm)", ("remplissage", "entraxe_lisses"), cas, 10.0, "%.0f", 50.0, width=140,
                     help="Chaque lisse reprend q_r × e_l, ou Q_r à mi-portée, sur la portée entre montants.")
        elif rpt == "panneau":
            st.markdown("**Panneau** (verre, tôle, câbles) — entre la lisse basse et la main courante")
            with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=P + "c_rp"):
                _num("h_bas (mm)", ("remplissage", "h_bas"), cas, 10.0, "%.0f", 0.0, width=120, help="Axe de la lisse basse au-dessus de la platine.")
            st.caption("La résistance du panneau sous q_r et Q_r est à justifier par le fournisseur ; la note le dit.")
        else:
            st.caption("Main courante seule, sans remplissage (rampe, lisse de protection).")
        geo = res.get("geometrie") or {}
        if geo.get("sans_montant"):
            st.info(f"Sans montant : la main courante porte de mur à mur sur {_fmt(geo['L_total'], 0)} mm (les deux bouts ancrés, une seule travée) ; "
                    "la fixation n'a pas d'objet, les réactions aux murs sont données dans le résultat. Pour des montants, raccourcir l'entraxe maximal.")


# ---------------------------------------------------------------- colonne de droite : charges
def _carte_charges(cas: Dict, res: Dict):
    ch = cas["charges"]
    cats = [c[0] for c in CH.CATEGORIES]
    libs = {c[0]: (c[1] + (f" — {c[2]:g} kN/m" if c[2] is not None else "")) for c in CH.CATEGORIES}

    def _cat():
        cas["charges"].update(CH.normaliser(cas["charges"]))
        _reset_widgets()
    # les quatre charges, en cellules courtes (le nom et l'origine complets sont dans l'aide des champs et dans la note)
    cellules = (("q_h", "horizontale sur la main courante, à H", "EN 1991-1-1 Tab. 6.12 (catégorie)"), ("q_v", "verticale sur la main courante", "usage du bureau"),
                ("Q_r", "ponctuelle sur le remplissage", "NBN B 03-004 (Q_kh,2)"), ("q_r", "pression sur le remplissage", "usage du bureau"))
    with st.container(border=True, key=P + "k_ch"):
        st.markdown('<div class="gc-kt">CHARGES <span style="letter-spacing:0;font-weight:400">— quatre cas distincts, chaque charge agit seule ; '
                    'le cumul de q_h et q_v est une option, du côté sûr</span></div><div class="gc-ch">'
                    + "".join(f'<div><span class="n">{i + 1}</span><span class="s">{s}</span> — {nom}<br><span class="o">{org}</span></div>'
                              for i, (s, nom, org) in enumerate(cellules)) + "</div>", unsafe_allow_html=True)
        with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=P + "c_ch1"):
            _choix("Catégorie d'usage (Tab. 6.12)", ("charges", "categorie"), cas, cats, libs, on_change=_cat, width=300,
                   help="La charge linéaire horizontale sur la main courante selon l'usage ; l'annexe nationale belge peut fixer d'autres valeurs : "
                        "« valeur libre » les accepte. La note imprime la valeur retenue et son origine.")
            _num("q_h (kN/m)", ("charges", "qh"), cas, 0.1, "%.2f", 0.0, disabled=(ch.get("categorie") != "perso"), width=110)
            _num("q_v (kN/m)", ("charges", "qv"), cas, 0.1, "%.2f", 0.0, width=110, help=CH.SOURCE_QV)
            _num("Q_r (kN)", ("charges", "Qr"), cas, 0.1, "%.2f", 0.0, width=100, help=CH.SOURCE_QR_PONCT)
            _num("q_r (kN/m²)", ("charges", "qr"), cas, 0.1, "%.2f", 0.0, width=110, help=CH.SOURCE_QR)
        with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="center", key=P + "c_ch2"):
            _case("Cumuler q_h et q_v (cas 1 + 2, côté sûr)", ("charges", "cumul"), cas,
                  help="Le cumul fait agir la charge horizontale et la charge verticale ensemble sur le montant, la main courante, la soudure et la platine. "
                       "Les charges 3 et 4 restent toujours des cas séparés. Par défaut chaque charge agit seule.")
            with st.popover("Options", key=P + "pop_ch", width="content", help="Charge P par montant saisie directement ; charge ponctuelle Q sur la main courante"):
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


# ---------------------------------------------------------------- colonne de droite : fixation
def _resume_fixation(res: Dict, lat: bool) -> str:
    """Une ligne : soudure, ancrages (F_t, F_v, arrachement en latéral), platine (σ_p, t_req face à t_p) avec leurs taux."""
    fix = res.get("fixation") or {}
    s, pl = fix.get("soudure"), fix.get("platine")
    parts = []
    if s:
        parts.append(f"soudure {_chip(s['taux'])} ({s['critere']}, {s['position']}, {CH.CAS_LIBELLES.get(s.get('cas_charge'), '')})")
    if pl and pl.get("taux_max") is not None:
        arr = f", arrachement P = {_fmt(pl['P'] / 1e3, 2)} kN" if lat else ""
        parts.append(f"ancrages {_chip(pl['taux_boulons'])} ({CH.CAS_LIBELLES.get(pl.get('cas_charge'), '')}{arr} : "
                     f"z = {_fmt(pl['z'])} mm, Z = {_fmt(pl['Z'] / 1e3, 2)} kN, F_t = {_fmt(pl['Ft'] / 1e3, 2)} kN ≤ {_fmt(pl['FtRd'] / 1e3, 2)}, "
                     f"F_v = {_fmt(pl['Fv'] / 1e3, 2)} kN ≤ {_fmt(pl['FvRd'] / 1e3, 2)}, interaction {pl['inter'] * 100:.0f} %)")
        parts.append(f"platine {_chip(pl['taux_platine'])} (m = {_fmt(pl['m'])} mm, b_eff = {_fmt(pl['b_eff'], 0)} mm, "
                     f"σ_p = {_fmt(pl['sigma_p'])} ≤ {_fmt(pl['lim_sigma'], 0)} N/mm², épaisseur requise {_fmt(pl['tp_req'])} mm face à t_p = {_fmt(pl['tp'], 0)})")
    return " · ".join(parts)


def _carte_fixation(cas: Dict, res: Dict):
    fx = cas["fixation"]
    g = cas["geometrie"]
    with st.container(border=True, key=P + "k_fix"):
        _titre_carte("FIXATION")
        typ = _choix("Fixation", ("fixation", "type"), cas, list(FX.TYPES), FIXATIONS_COURTES, segmente=True, label_visibility="collapsed",
                     help="Platine boulonnée : le montant est soudé sur sa platine posée sur la dalle, fixée par chevilles ou boulons ; platine latérale : la "
                          "platine est verticale, en applique sur le nez de dalle, le montant soudé le long d'elle ; soudé : le montant est soudé directement "
                          "sur un plat existant (la soudure seule est vérifiée) ; scellé : rien n'est vérifié.")
        if g.get("systeme") == "mur":
            with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=P + "c_cons"):
                _num("Console : e du mur à l'axe de la main courante (mm)", ("fixation", "console", "e"), cas, 10.0, "%.0f", 20.0, width=330,
                     help="Bras de levier de la charge verticale sur la console et sur ses chevilles ; la section de la console se choisit dans le résultat.")
        if (res.get("geometrie") or {}).get("sans_montant"):
            st.info("Sans montant : rien à fixer — la main courante est ancrée dans les murs (réactions dans le résultat).")
            return
        if typ == "scellee":
            st.caption("Montant scellé dans le béton ou dans un fourreau : la fixation n'est pas vérifiée ici.")
            return
        lat = typ == FX.LATERALE and g.get("systeme") == "montants"
        st.markdown("**Soudure du montant le long de la platine** — deux cordons longitudinaux de la hauteur b_p (module Soudures, C8)" if lat
                    else "**Soudure sur la platine** — vérifiée par le module Soudures")
        s = (res.get("fixation") or {}).get("soudure")
        with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=P + "c_soud"):
            jon = _choix("Jonction", ("fixation", "soudure", "jonction"), cas, ["angle", "partielle", "pleine"],
                         {"angle": "cordon d'angle", "partielle": "pénétration partielle", "pleine": "pleine pénétration"}, segmente=True)
            if jon == "angle":
                _num("Gorge a (mm)", ("fixation", "soudure", "a"), cas, 0.5, "%.1f", 1.0, width=110)
            elif jon == "partielle":
                _num("Pénétration p (mm)", ("fixation", "soudure", "p"), cas, 0.5, "%.1f", 1.0, width=140)
                _num("Renfort a_r (mm)", ("fixation", "soudure", "renfort"), cas, 0.5, "%.1f", 0.0, width=130, help="0 : sans renfort.")
            _num("t_p, platine (mm)", ("fixation", "platine", "tp"), cas, 1.0, "%.0f", 3.0, width=130)
            vues = SCS.vues(s["res"]) if s else []
            if vues:
                with st.popover("Dessins de la soudure", key=P + "pop_soud", width="content", help="Les vues du module Soudures pour ce groupe de cordons"):
                    cols = st.columns(min(2, len(vues)), gap="small")
                    for i, (cle, titre, d) in enumerate(vues[:2]):
                        with cols[i]:
                            with st.container(key=P + f"soud_{i}"):
                                st.markdown(f'<div class="gc-leg"><b>{titre.upper()}</b></div>' + SCS.svg(d, f"gc_s{i}"), unsafe_allow_html=True)
        if typ in ("platine", FX.LATERALE):
            if lat:
                st.markdown("**Platine latérale et ancrage** — platine verticale en applique sur le nez de dalle, deux rangées horizontales de chevilles "
                            "(ou boulons) à ±e_x/2 du centre, la haute tendue et la basse en appui ; colonnes de part et d'autre du montant à e_y")
                with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=P + "c_pl"):
                    _num("d_h — descente (mm)", ("fixation", "platine", "d_h"), cas, 5.0, "%.0f", 0.0, width=140,
                         help="Du dessus de dalle au haut de la platine. Le montant est vérifié sur H + d_h ; le moment à la platine vaut P·(H + d_h + b_p/2).")
                    _num("b_p — hauteur (mm)", ("fixation", "platine", "bp"), cas, 5.0, "%.0f", 40.0, width=140)
                    _num("l_p — largeur (mm)", ("fixation", "platine", "lp"), cas, 5.0, "%.0f", 40.0, width=140)
                    _num("e_x — entraxe vertical (mm)", ("fixation", "ancrage", "ex"), cas, 5.0, "%.0f", 10.0, width=180, help="Entre les deux rangées : le bras de levier z du couple d'ancrages.")
                    _num("Par rangée (2 ou 4)", ("fixation", "ancrage", "n_par_rangee"), cas, 2, min_value=2, max_value=4, width=140, help="Chevilles par rangée, par paires.")
                    _num("e_y — entraxe horizontal (mm)", ("fixation", "ancrage", "ey"), cas, 5.0, "%.0f", 10.0, width=190,
                         help="Entre les colonnes, de part et d'autre du montant ; doit dépasser sa largeur b plus un trou.")
            else:
                st.markdown("**Platine et ancrage** — platine rigide, deux rangées de chevilles (ou boulons) de part et d'autre du montant, à ±e_x/2")
                with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=P + "c_pl"):
                    _num("b_p — sens de l'effort (mm)", ("fixation", "platine", "bp"), cas, 5.0, "%.0f", 40.0, width=170)
                    _num("l_p — largeur (mm)", ("fixation", "platine", "lp"), cas, 5.0, "%.0f", 40.0, width=140)
                    _num("e_x — entraxe des rangées (mm)", ("fixation", "ancrage", "ex"), cas, 5.0, "%.0f", 10.0, width=190, help="Dans le sens de l'effort ; doit dépasser la dimension h_p du montant.")
                    _num("Par rangée", ("fixation", "ancrage", "n_par_rangee"), cas, 1, min_value=1, max_value=4, width=110, help="Chevilles par rangée.")
                    _num("e_y — dans la rangée (mm)", ("fixation", "ancrage", "ey"), cas, 5.0, "%.0f", 10.0, width=170)
            with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=P + "c_anc"):
                ta = _choix("Ancrage", ("fixation", "ancrage", "type"), cas, list(FX.ANCRAGES), FX.ANCRAGES, segmente=True)
                if ta == "boulon":
                    _choix("Boulon", ("fixation", "ancrage", "boulon"), cas, FX.boulons(), width=110)
                    _choix("Classe", ("fixation", "ancrage", "classe"), cas, FX.classes(), width=100)
                else:
                    _texte_w("Cheville (désignation)", ("fixation", "ancrage", "nom"), cas, width=180)
                    _num("N_Rd par cheville (kN)", ("fixation", "ancrage", "NRd"), cas, 0.5, "%.1f", 0.1, width=160,
                         help="Valeur de calcul en traction de l'ETA (logiciel du fabricant), pour le béton et la profondeur du projet.")
                    _num("V_Rd par cheville (kN)", ("fixation", "ancrage", "VRd"), cas, 0.5, "%.1f", 0.1, width=160, help="Valeur de calcul en cisaillement de l'ETA.")
        resume = _resume_fixation(res, lat)
        if resume:
            st.markdown(f'<div class="gc-fix">{resume}</div>', unsafe_allow_html=True)
        pl = (res.get("fixation") or {}).get("platine")
        if pl and pl.get("taux_max") is None:
            for msg in pl.get("messages", []):
                st.warning(msg)


# ---------------------------------------------------------------- en bas : le détail, replié
def _detail(etat: Dict, cas: Dict, res: Dict):
    _titre_carte("DÉTAIL")
    with st.expander("Efforts et taux — éléments avec leur cas gouvernant, montants / consoles un à un", expanded=False):
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
                           f"<td class='n'>{_chip(e['taux'])}</td><td>{ME.critere_texte(e['critere'])}</td></tr>")
            else:
                lignes += (f"<tr{cls}><td>{e['libelle']}</td><td>{cas_c}</td><td class='n'>—</td><td class='n'>—</td><td class='n'>—</td><td class='n'>—</td><td class='n'>—</td>"
                           f"<td class='n'>{_chip(e['taux'])}</td><td>{e['critere']}</td></tr>")
        st.markdown("<table class='gc-tab'><tr><th>Élément</th><th>cas gouvernant</th><th class='n'>M (kN·m)</th><th class='n'>V (kN)</th><th class='n'>σ (N/mm²)</th><th class='n'>τ</th>"
                    "<th class='n'>flèche / adm. (mm)</th><th class='n'>η</th><th>critère</th></tr>" + lignes + "</table>", unsafe_allow_html=True)
        liste = res.get("montants") or res.get("consoles") or []
        if liste:
            lignes = ""
            for v in liste:
                lignes += (f"<tr{' class=g' if v['nom'] == ((res.get('montant') or res.get('console') or {}).get('nom')) else ''}><td><b>{v['nom']}</b>{' · angle' if v['coin'] else ''}</td>"
                           f"<td class='n'>{_fmt(v['x'], 0)}</td><td class='n'>{_fmt(v['trib1'] + v['trib2'], 0)}</td><td class='n'>{_fmt(v['P1'] / 1e3, 2)}"
                           f"{' + ' + _fmt(v['P2'] / 1e3, 2) if v['P2'] > 0 else ''}</td><td class='n'>{_fmt(v['N'] / 1e3, 2)}</td><td class='n'>{_fmt(v['sigma'])}</td>"
                           f"<td class='n'>{_fmt(v['fleche'], 2)}</td><td class='n'>{_chip(v['taux_max'])}</td></tr>")
            st.markdown("<table class='gc-tab'><tr><th>Montant / console</th><th class='n'>x (mm)</th><th class='n'>s_trib (mm)</th><th class='n'>P (kN)</th><th class='n'>N (kN)</th>"
                        "<th class='n'>σ</th><th class='n'>flèche</th><th class='n'>η</th></tr>" + lignes + "</table>", unsafe_allow_html=True)
            geo = res.get("geometrie") or {}
            st.caption(f"{geo.get('phrase', '')} · portée de la main courante : l'entraxe le plus grand, {_fmt(geo.get('s_max'), 0)} mm · "
                       "s_trib : la longueur tributaire (les deux directions à un angle).")
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
    col_g, col_d = st.columns([1.1, 1], gap="large")
    with col_g:
        _dessins(cas, res)
        _resultat(cas, res)
    with col_d:
        _carte_definition(cas, res)
        _carte_charges(cas, res)
        _carte_fixation(cas, res)
    _detail(etat, cas, res)

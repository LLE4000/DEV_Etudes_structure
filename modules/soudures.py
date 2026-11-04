# -*- coding: utf-8 -*-
"""Écran Streamlit du module « Soudures ».

Le moteur, l'état et les dessins vivent dans ``acier/soudures`` (sans
Streamlit : ``tests/test_soudures_moteur.py`` le garde) ; ce fichier n'est
que la vue.

Organisation (étude docs/acier/SOUDURES_ETUDE.md §5) : barre d'outils du
cas ; le SWITCH de méthode bien visible — « Eurocode (ELU) » |
« Méthode ELS (bureau) » ; à gauche la configuration (C1 … C8), le dessin
technique, l'encart « la gorge » et la carte des taux ; à droite la carte
RÉSULTAT puis cinq onglets à libellé fixe : Configuration, Cordons, Efforts,
Méthode, Détail. Une seule source : l'état ``st.session_state["sdr"]``
(``acier/soudures/etat.py``) ; les widgets (clés ``sdr_w_``) en sont des
vues, réinitialisées dès que l'état change par ailleurs.
"""
import copy
import json
from typing import Dict, List

import streamlit as st

from acier.soudures import configurations as CF
from acier.soudures import criteres as K
from acier.soudures import dessin as DS
from acier.soudures import etat as ET
from acier.soudures import methode as ME
from acier.soudures import note as NT
from modules import liaison as LI

P = "sdr_"
REGISTRE: LI.Registre = {}       # clé de widget → chemin dans le cas (modules/liaison.py)
ETAPES = ["1 · Configuration", "2 · Cordons", "3 · Efforts", "4 · Méthode", "5 · Détail"]
MODES = {K.EUROCODE: "Eurocode (ELU)", K.ELS: "Méthode ELS (bureau)"}
CRITERES = {K.DIRECTIONNELLE: "directionnelle (§4.5.3.2)", K.SIMPLIFIEE: "simplifiée (§4.5.3.3)"}
COTES_C4 = {"gauche": "gauche", "droite": "droite", "bas": "bas", "haut": "haut"}
CSS = """<style>
.sdr-res{border:1px solid #e5e7eb;border-radius:10px;padding:12px 14px;background:#fff;margin-bottom:8px}
.sdr-res .t{font-size:11px;letter-spacing:.08em;color:#6b7280;font-weight:600}
.sdr-pill{float:right;font-size:12px;border-radius:999px;padding:3px 10px;font-weight:600}
.sdr-pill.nok{background:#fee2e2;color:#b91c1c}.sdr-pill.ok{background:#dcfce7;color:#15803d}.sdr-pill.att{background:#ffedd5;color:#c2410c}.sdr-pill.ref{background:#f3f4f6;color:#374151}
.sdr-eta{font-size:26px;font-weight:700;margin:2px 0 0;line-height:1.1}
.sdr-phrase{font-size:13.5px;margin:4px 0 6px}
.sdr-row{display:grid;grid-template-columns:150px 1fr 56px;align-items:center;gap:10px;font-size:12.5px;margin:5px 0}
.sdr-row.sel span:first-child{font-weight:700}
.sdr-trk{height:6px;background:#f1f3f5;border-radius:3px;overflow:hidden}.sdr-trk b{display:block;height:100%}
.sdr-v{text-align:right;font-variant-numeric:tabular-nums;font-weight:600}
.sdr-note{font-size:12px;color:#6b7280;margin-top:6px}
.sdr-tab{width:100%;border-collapse:collapse;font-size:12.5px}.sdr-tab th{text-align:left;font-size:10.5px;letter-spacing:.05em;color:#6b7280;padding:4px 6px;border-bottom:1px solid #e5e7eb}
.sdr-tab td{padding:4px 6px;border-bottom:1px solid #f3f4f6;vertical-align:top}.sdr-tab td.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.sdr-tab tr.g td{font-weight:700;background:#fff7ed}
.sdr-r{color:#b91c1c;font-weight:700}.sdr-o{color:#c2410c;font-weight:700}.sdr-g{color:#15803d;font-weight:700}.sdr-i{color:#6b7280}
.sdr-leg{font-size:11.5px;color:#6b7280}
.sdr-switch{font-size:12.5px;color:#374151;padding-top:6px}
</style>"""


# ---------------------------------------------------------------- état et vues des widgets
def _etat() -> Dict:
    if "sdr" not in st.session_state:
        st.session_state["sdr"] = ET.nouvel_etat()
    return st.session_state["sdr"]


def _reset_widgets():
    """Depuis un callback : les widgets prennent les valeurs de l'état (modules/liaison.py)."""
    LI.synchroniser(P, REGISTRE, ET.cas_actif(_etat()))


def _chemin(d: Dict, chemin):
    for c in chemin[:-1]:
        d = d[c]
    return d, chemin[-1]


def _ecrire(chemin, cle, conv=None):
    """Callback : écrit la valeur du widget ``cle`` dans le cas actif, au chemin donné."""
    v = st.session_state.get(cle)
    if conv is not None:
        v = conv(v)
    d, k = _chemin(ET.cas_actif(_etat()), chemin)
    d[k] = v


def _cle(chemin) -> str:
    return P + "w_" + "_".join(str(c) for c in chemin)


def _num(label, chemin, cas, step=1.0, fmt="%.1f", min_value=None, max_value=None, help=None, width=None):
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
    st.number_input(label, value=v0, step=step, format="%d" if entier else fmt, key=cle, on_change=_ecrire, args=(chemin, cle, conv), help=help, **kw)


def _texte_w(label, chemin, cas, help=None, width=None):
    d, k = _chemin(cas, chemin)
    cle = _cle(chemin)
    LI.enregistrer(REGISTRE, cle, chemin, lambda v: v or "")
    kw = {"width": width} if width is not None else {}
    st.text_input(label, value=d.get(k) or "", key=cle, on_change=_ecrire, args=(chemin, cle, None), help=help, **kw)


def _choix(label, chemin, cas, options, affichage=None, help=None, segmente=False, width=None):
    d, k = _chemin(cas, chemin)
    cle = _cle(chemin)
    cur = d.get(k)
    if cur not in options:
        cur = options[0]
    LI.enregistrer(REGISTRE, cle, chemin, None, options)
    fmt = (lambda o: affichage.get(o, str(o))) if affichage else (lambda o: str(o))
    kw = {"width": width} if width is not None else {}
    if segmente:
        st.segmented_control(label, options, default=cur, format_func=fmt, key=cle, on_change=_ecrire, args=(chemin, cle, None),
                             help=help, selection_mode="single", **kw)
    else:
        st.selectbox(label, options, index=options.index(cur), format_func=fmt, key=cle, on_change=_ecrire, args=(chemin, cle, None), help=help, **kw)


def _case(label, chemin, cas, help=None):
    d, k = _chemin(cas, chemin)
    cle = _cle(chemin)
    LI.enregistrer(REGISTRE, cle, chemin, bool)
    st.checkbox(label, value=bool(d.get(k)), key=cle, on_change=_ecrire, args=(chemin, cle, bool), help=help)


def _troncons(cas, cle_p="troncons", label="Soudure discontinue (par tronçons)"):
    """Case « par tronçons » puis n et l ; dans l'état : params[cle_p] = {"n": 0 | k, "l": mm}."""
    tr = cas["params"].setdefault(cle_p, dict(CF.CONTINU))
    cle_on = P + "w_" + cle_p + "_on"

    def _on():
        tr["n"] = max(1, int(tr.get("n") or 0)) if st.session_state[cle_on] else 0

    LI.enregistrer(REGISTRE, cle_on, ("params", cle_p, "n"), lambda n: int(n or 0) >= 1)
    with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=P + "c_" + cle_p):
        st.checkbox(label, value=int(tr.get("n") or 0) >= 1, key=cle_on, on_change=_on,
                    help="Au lieu d'un cordon continu : n tronçons de l mm, aux deux bouts et régulièrement entre (pourtour : "
                         "régulièrement, le premier en haut). EN 1993-1-8 §4.3.2.2 : espacement ≤ min(200 mm ; 12 t en compression ; 16 t en traction).")
        if int(tr.get("n") or 0) >= 1:
            _num("Tronçons", ("params", cle_p, "n"), cas, 1, min_value=1, width=120)
            _num("Longueur d'un tronçon (mm)", ("params", cle_p, "l"), cas, 5.0, "%.0f", min_value=1.0, width=200)


def _changer(toast: str = ""):
    """Après un changement de l'état hors callback : synchronisation au rendu suivant, relance."""
    LI.demander(P)
    if toast:
        st.toast(toast)
    st.rerun()


def _fmt(x, nd=1):
    return f"{x:,.{nd}f}".replace(",", " ").replace(".", ",")


# ---------------------------------------------------------------- en-tête, barre d'outils, switch
def _entete(etat: Dict, cas: Dict):
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("## Soudures")
    st.caption("Vérification des cordons de soudure — groupe de cordons en répartition élastique, EN 1993-1-8 §4 "
               "(méthode directionnelle ou simplifiée, efforts de calcul) ou méthode ELS du bureau (efforts de service, "
               "k kg par mm de gorge et par cm de cordon) · recouvrement, T, console, gousset, platine de profilé, tube, "
               "bout à bout, cordons libres · version 1.0")
    with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="center", key=P + "barre"):
        if st.button("🏠 Accueil", key=P + "btn_home", width="content"):
            st.session_state.retour_accueil_demande = True
            st.rerun()
        noms = [c["nom"] for c in etat["cas"]]

        def _sel():
            etat["actif"] = noms.index(st.session_state[P + "sel_cas"])
            _reset_widgets()
        st.session_state[P + "sel_cas"] = noms[etat["actif"]]     # le sélecteur SUIT l'état
        st.selectbox("Cas", noms, key=P + "sel_cas", on_change=_sel, label_visibility="collapsed", width=240)
        if st.button("＋ Cas", key=P + "btn_new", width="content", help="Nouveau cas (console fléchie par défaut)"):
            n = len(etat["cas"]) + 1
            c = ET.nouveau_cas("C3", f"Cas {n} · Console fléchie")
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
        st.download_button("💾 Enregistrer", data=ET.vers_json(etat).encode("utf-8"), file_name=f"soudures_{cas['nom']}.json",
                           mime="application/json", key=P + "btn_save", width="content", help="Tous les cas (JSON)")
        if st.button("📂 Ouvrir", key=P + "btn_open", width="content"):
            st.session_state[P + "ouvrir"] = not st.session_state.get(P + "ouvrir", False)
        if st.button("📄 Note", key=P + "btn_note", width="content", help="Note de calcul PDF en deux pages : calcul (dessin, gorge, cordons, efforts, vérifications), méthode et détail"):
            ident = etat.get("identification") or {}
            st.session_state[P + "note"] = (cas["nom"], NT.generer_note(etat, cas, {"nom_projet": ident.get("projet", ""), "partie": ident.get("partie", ""),
                                                                                    "indice": ident.get("indice", "0")}))
        _texte_w("Nom du cas", ("nom",), cas, width=260)
    note = st.session_state.get(P + "note")
    if note and note[0] == cas["nom"]:
        st.download_button("⬇️ Télécharger la note (PDF, 2 pages)", data=note[1], file_name=f"note_soudures_{cas['nom']}.pdf",
                           mime="application/pdf", key=P + "dl_note", width="content")
    if st.session_state.get(P + "ouvrir"):
        up = st.file_uploader("Fichier JSON du module", type=["json"], key=P + "uploader", label_visibility="collapsed")
        if up is not None:
            try:
                st.session_state["sdr"] = ET.depuis_json(up.read().decode("utf-8-sig"))
                st.session_state[P + "ouvrir"] = False
                _changer("Fichier chargé")
            except Exception as e:  # noqa: BLE001
                st.error(f"Fichier invalide : {e}")


def _switch(cas: Dict):
    """Le switch de méthode, bien visible, et une ligne qui dit ce que chaque position change."""
    c1, c2 = st.columns([1, 2.2], vertical_alignment="center")
    with c1:
        _choix("Méthode", ("methode", "mode"), cas, [K.EUROCODE, K.ELS], MODES, segmente=True)
    m = cas["methode"]
    with c2:
        if m["mode"] == K.ELS:
            kq = K.k_equivalent_eurocode(ET.acier_cordon(cas["params"])["fu"], ET.acier_cordon(cas["params"])["bw"], m["gamma_M2"], m["gamma_F"], m["g"])
            st.markdown(f'<div class="sdr-switch"><b>Efforts de service</b>, sans coefficient : résultante par mm de cordon ≤ k·a avec '
                        f'<b>k = {m["k"]:g} kg/(mm·cm)</b> = {K.tau_adm_bureau(m["k"], m["g"]):.1f} N/mm² sur la gorge — '
                        f'l\'Eurocode donnerait f_vw,d/γ_F = {kq["tau_N_mm2"]:.1f} N/mm² ({kq["k_kg_mm_cm"]:.0f} kg/(mm·cm)) pour le '
                        f'{ET.acier_cordon(cas["params"])["nuance_ref"]} (onglet Méthode).</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="sdr-switch"><b>Efforts de calcul (ELU)</b> : EN 1993-1-8 §4.5.3, méthode {CRITERES[m["ec"]].split(" (")[0]}, '
                        f'fu et βw de l\'acier de référence, γM2 = {m["gamma_M2"]:g}. Le taux selon la méthode ELS du bureau est affiché '
                        f'à côté, avec les mêmes efforts, à titre indicatif.</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- colonne de gauche : configuration et dessins
def _colonne_dessin(etat: Dict, cas: Dict, res: Dict):
    def _cfg():
        v = st.session_state.get(P + "w_config")
        if v and v != cas["config"]:
            ET.changer_config(cas, v)
            _reset_widgets()
    LI.enregistrer(REGISTRE, P + "w_config", ("config",), None, list(CF.CONFIGS))
    st.radio("Configuration", list(CF.CONFIGS), format_func=lambda c: f"{c} · {CF.CONFIGS[c]['libelle']}", index=list(CF.CONFIGS).index(cas["config"]),
             key=P + "w_config", on_change=_cfg, horizontal=True, label_visibility="collapsed")
    with st.container(key=P + "dessin"):
        st.markdown(DS.vue_principale(res), unsafe_allow_html=True)
    c1, c2 = st.columns([1.3, 1], gap="small")
    with c1:
        with st.container(key=P + "gorge"):
            st.markdown(DS.encart_gorge(res), unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="sdr-leg" style="margin:4px 0 2px"><b>CARTE DES TAUX</b> · par cordon, selon la méthode choisie</div>', unsafe_allow_html=True)
        st.markdown(DS.carte_taux(res), unsafe_allow_html=True)
    st.markdown(f'<div class="sdr-leg">{DS.legende()}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- colonne de droite : résultat
def _carte_resultat(cas: Dict, res: Dict):
    stat = res["statut"]
    pill = {ET.OK: ("ok", "● Conforme"), ET.ATTENTION: ("att", "● Conforme, attention"), ET.NOK: ("nok", "● Non conforme"),
            ET.REFUS: ("ref", "● Calcul impossible")}[stat]
    m = res["methode"]
    g = res.get("gouvernant")
    if stat == ET.REFUS:
        eta = "—"
        phrase = (res["messages"] or ["Données manquantes"])[0]
    else:
        t = res["taux_max"]
        col = "#b91c1c" if t > 1 else ("#c2410c" if t > 0.85 else "#15803d")
        eta = f'<span style="color:{col}">η = {t * 100:.0f} %</span>'
        if res.get("pleine"):
            phrase = f"Pleine pénétration : la pièce gouverne — {g['critere']}"
        else:
            ch = next((c["chaine"] for c in res["cordons"] if c["nom"] == g["cordon"]), g["cordon"])
            phrase = f"<b>{ch}</b>, {g['position']} — {g['critere']}"
    if m.mode == K.ELS:
        ligne_m = f"Méthode ELS du bureau · k = {m.k:g} kg/(mm·cm) → {K.tau_adm_bureau(m.k, m.g):.1f} N/mm² · efforts de service"
    else:
        ligne_m = (f"Eurocode {CRITERES[m.ec].split(' (')[0]} · acier de référence {res['acier']['nuance_ref']} (fu {res['acier']['fu']:g}, "
                   f"βw {res['acier']['bw']:g}) · γM2 {m.gamma_M2:g} · efforts de calcul")
    barres = ""
    comp = res.get("comparaison") or {}
    if comp and not res.get("pleine"):
        kq = res.get("k_equivalent") or {}
        lignes = [("directionnelle", "EC3 directionnelle", m.mode == K.EUROCODE and m.ec == K.DIRECTIONNELLE),
                  ("simplifiee", "EC3 simplifiée", m.mode == K.EUROCODE and m.ec == K.SIMPLIFIEE),
                  ("bureau", f"ELS bureau k = {m.k:g}", m.mode == K.ELS),
                  ("bureau_equiv", f"ELS k Eurocode = {kq.get('k_kg_mm_cm', 0):.0f}", False)]
        for cle, lab, sel in lignes:
            t = comp.get(cle)
            if t is None:
                continue
            col = "#b91c1c" if t > 1 else ("#c2410c" if t > 0.85 else "#2563eb")
            barres += (f'<div class="sdr-row{" sel" if sel else ""}"><span>{lab}{" ◀" if sel else ""}</span>'
                       f'<div class="sdr-trk"><b style="width:{min(t, 1) * 100:.0f}%;background:{col}"></b></div>'
                       f'<span class="sdr-v" style="color:{col}">{t * 100:.0f} %</span></div>')
        barres += ('<div class="sdr-note">Les quatre lignes appliquent chaque méthode aux MÊMES efforts saisis : la méthode choisie (◀) '
                   'décide du statut ; les autres sont indicatives (l\'Eurocode attend des efforts de calcul, le bureau des efforts de service).</div>')
    st.markdown(f'<div class="sdr-res"><span class="sdr-pill {pill[0]}">{pill[1]}</span><div class="t">RÉSULTAT</div>'
                f'<div class="sdr-eta">{eta}</div><div class="sdr-phrase">{phrase}</div><div class="sdr-note">{ligne_m}</div>{barres}</div>',
                unsafe_allow_html=True)
    for msg in res.get("messages", []):
        if msg.startswith("Refus") or "DÉPASSÉ" in msg:
            st.warning(msg)
        elif "ne compte pas" in msg or "ne tiennent pas" in msg:
            st.warning(msg)
        else:
            st.caption("• " + msg)


# ---------------------------------------------------------------- onglet 1 : configuration
def _profils(groupe: str) -> List[str]:
    from acier.profiles import catalogue as CAT
    from acier.profiles import familles as F
    return [n for n, r in CAT.par_nom().items() if F.FAMILLES.get(r["famille"], {}).get("groupe") == groupe]


def _nuances(cas):
    nz = list(ET.nuances())
    c1, c2 = st.columns(2)
    with c1:
        _choix("Nuance de la pièce A (support)", ("params", "nuance_A"), cas, nz, help="fu et βw de référence : la pièce la plus faible (Tab. 4.1)")
    with c2:
        _choix("Nuance de la pièce B (rapportée)", ("params", "nuance_B"), cas, nz)


def _etape_configuration(etat: Dict, cas: Dict, res: Dict):
    cfg = cas["config"]
    st.caption(f"**{cfg} · {CF.CONFIGS[cfg]['libelle']}** — {CF.CONFIGS[cfg]['description']}")
    if cfg == "C1":
        c1, c2, c3 = st.columns(3)
        with c1:
            _num("Largeur b du plat B (mm)", ("params", "b"), cas, 5.0, "%.0f", 10.0)
            _num("Épaisseur t_B (mm)", ("params", "tB"), cas, 1.0, "%.0f", 1.0)
        with c2:
            _num("Recouvrement L (mm)", ("params", "L_rec"), cas, 5.0, "%.0f", 10.0)
            _num("Épaisseur t_A (mm)", ("params", "tA"), cas, 1.0, "%.0f", 1.0)
        with c3:
            _num("Gorge a (mm)", ("params", "a"), cas, 0.5, "%.1f", 1.0)
            _case("Retours d'angle exécutés", ("params", "retours"), cas, help="Sans retours : longueur efficace l − 2a (§4.5.1)")
        c1, c2 = st.columns(2)
        with c1:
            _case("Cordons latéraux (le long de N)", ("params", "lateraux"), cas)
        with c2:
            _choix("Cordons frontaux (en travers)", ("params", "frontaux"), cas, [0, 1, 2], {0: "aucun", 1: "un (bout du plat B)", 2: "deux (bouts de B et de A)"})
        _troncons(cas, "troncons", "Cordons latéraux par tronçons")
    elif cfg == "C2":
        c1, c2, c3 = st.columns(3)
        with c1:
            _num("Longueur de la jonction l (mm)", ("params", "l"), cas, 5.0, "%.0f", 10.0)
            _num("Épaisseur t_B de la pièce debout (mm)", ("params", "tB"), cas, 1.0, "%.0f", 1.0)
        with c2:
            _num("Épaisseur t_A du support (mm)", ("params", "tA"), cas, 1.0, "%.0f", 1.0)
            _num("Gorge a (mm)", ("params", "a"), cas, 0.5, "%.1f", 1.0)
        with c3:
            _case("Deux cordons (un de chaque côté)", ("params", "deux_cordons"), cas)
            _case("Retours d'angle exécutés", ("params", "retours"), cas)
        _troncons(cas)
    elif cfg == "C3":
        c1, c2, c3 = st.columns(3)
        with c1:
            _num("Hauteur h du plat (mm)", ("params", "h"), cas, 5.0, "%.0f", 10.0)
            _num("Épaisseur t du plat (mm)", ("params", "t"), cas, 1.0, "%.0f", 1.0)
        with c2:
            _num("Excentricité e de V (mm)", ("params", "e"), cas, 5.0, "%.0f", 0.0)
            _num("Gorge a (mm)", ("params", "a"), cas, 0.5, "%.1f", 1.0)
        with c3:
            _case("Deux cordons (un de chaque côté)", ("params", "deux_cordons"), cas)
            _case("Retours d'angle exécutés", ("params", "retours"), cas)
        _troncons(cas)
    elif cfg == "C4":
        c1, c2, c3 = st.columns(3)
        with c1:
            _num("Longueur b_x du plat (mm)", ("params", "bx"), cas, 5.0, "%.0f", 10.0)
            _num("Hauteur b_y du plat (mm)", ("params", "by"), cas, 5.0, "%.0f", 10.0)
        with c2:
            _num("Épaisseur t (mm)", ("params", "t"), cas, 1.0, "%.0f", 1.0)
            _num("Gorge a (mm)", ("params", "a"), cas, 0.5, "%.1f", 1.0)
        with c3:
            _num("x du point d'application (mm)", ("params", "xV"), cas, 5.0, "%.0f")
            _num("y du point d'application (mm)", ("params", "yV"), cas, 5.0, "%.0f")
        cur = [c for c in cas["params"].get("cotes", []) if c in COTES_C4]

        def _cotes():
            cas["params"]["cotes"] = list(st.session_state[P + "w_cotes"] or [])
        LI.enregistrer(REGISTRE, P + "w_cotes", ("params", "cotes"), lambda v: [c for c in (v or []) if c in COTES_C4])
        st.pills("Côtés soudés", list(COTES_C4), default=cur, key=P + "w_cotes", on_change=_cotes, selection_mode="multi")
        _case("Retours d'angle exécutés", ("params", "retours"), cas)
        _troncons(cas)
    elif cfg == "C5":
        c1, c2, c3 = st.columns(3)
        with c1:
            _choix("Profilé du catalogue", ("params", "profil"), cas, _profils("I/H"))
            _choix("Semelles soudées", ("params", "semelles"), cas, ["deux faces", "extérieur"], {"deux faces": "deux faces (ext. + int.)", "extérieur": "face extérieure seule"})
        with c2:
            _num("Gorge des semelles a_f (mm)", ("params", "a_f"), cas, 0.5, "%.1f", 1.0)
            _case("Âme soudée (deux cordons)", ("params", "ame"), cas)
        with c3:
            _num("Gorge de l'âme a_w (mm)", ("params", "a_w"), cas, 0.5, "%.1f", 1.0)
            _case("Retours d'angle exécutés", ("params", "retours"), cas)
        _troncons(cas, "troncons_semelles", "Semelles par tronçons")
        _troncons(cas, "troncons_ame", "Âme par tronçons")
    elif cfg == "C6":
        c1, c2 = st.columns(2)
        with c1:
            _choix("Tube du catalogue", ("params", "profil"), cas, _profils("creux"))
        with c2:
            _num("Gorge a (mm)", ("params", "a"), cas, 0.5, "%.1f", 1.0)
        _troncons(cas, "troncons", "Pourtour par tronçons (pas de cordon partout)")
    elif cfg == "C7":
        c1, c2, c3 = st.columns(3)
        with c1:
            _num("Largeur b (mm)", ("params", "b"), cas, 5.0, "%.0f", 10.0)
            _num("Épaisseur t (mm)", ("params", "t"), cas, 1.0, "%.0f", 1.0)
        with c2:
            _choix("Pénétration", ("params", "penetration"), cas, ["pleine", "partielle"], {"pleine": "pleine (la pièce gouverne)", "partielle": "partielle (gorge = pénétration p)"}, segmente=True)
        with c3:
            _num("Pénétration p par face (mm)", ("params", "p"), cas, 0.5, "%.1f", 1.0)
            _case("Soudée des deux faces", ("params", "deux_faces"), cas)
    else:
        _elements_libres(cas)
    _nuances(cas)


def elements_depuis_lignes(lignes: List[Dict]) -> List[Dict]:
    """Les lignes du tableau (cellules vides = NaN) → les éléments de l'état,
    complétés par les défauts de leur type ; une ligne vide est un segment par défaut."""
    out = []
    for row in lignes:
        typ = row.get("type") if row.get("type") in ("segment", "arc") else "segment"
        base = dict(CF.ELEMENT_ARC if typ == "arc" else CF.ELEMENT_SEGMENT)
        el = {"type": typ, "nom": (row.get("nom") if isinstance(row.get("nom"), str) else "") or ""}
        for k in ("x1", "y1", "x2", "y2", "xc", "yc", "r", "ang1", "ang2", "a"):
            v = row.get(k)
            el[k] = float(v) if v is not None and v == v else float(base.get(k, 0.0))
        el["cote"] = int(row.get("cote")) if row.get("cote") in (1, -1, 1.0, -1.0) else 1
        el["retours"] = False
        n, lt = row.get("n"), row.get("l")
        el["troncons"] = {"n": int(n) if n is not None and n == n else 0, "l": float(lt) if lt is not None and lt == lt else 60.0}
        out.append(el)
    return out


def _elements_libres(cas: Dict):
    """C8 : le tableau des éléments (segments et arcs) — on calcule une soudure sur n'importe quoi."""
    import pandas as pd
    st.caption("Chaque ligne est un **segment** (P1 → P2) ou un **arc** (centre, rayon, angles en degrés, sens trigonométrique ; "
               "0 → 360 = cercle). Côté : la pièce rapportée est à gauche (+1) ou à droite (−1) du sens de parcours — "
               "pour un cercle, +1 = la pièce est à l'intérieur. Tronçons n = 0 : élément continu.")
    cols = ["type", "nom", "x1", "y1", "x2", "y2", "xc", "yc", "r", "ang1", "ang2", "a", "cote", "n", "l"]
    lignes = []
    for el in CF.elements_c8(cas["params"]):
        tr = el.get("troncons") or {}
        lignes.append({"type": el["type"], "nom": el["nom"], "x1": el.get("x1"), "y1": el.get("y1"), "x2": el.get("x2"), "y2": el.get("y2"),
                       "xc": el.get("xc"), "yc": el.get("yc"), "r": el.get("r"), "ang1": el.get("ang1"), "ang2": el.get("ang2"),
                       "a": el.get("a", 5.0), "cote": int(el.get("cote", 1)), "n": int(tr.get("n", 0) or 0), "l": float(tr.get("l", 60.0) or 60.0)})
    df = pd.DataFrame(lignes, columns=cols)
    ed = st.data_editor(df, num_rows="dynamic", hide_index=True, key=P + f"w_elements_{LI.generation(P)}", width="stretch",
                        column_config={"type": st.column_config.SelectboxColumn("type", options=["segment", "arc"], required=True),
                                       "nom": st.column_config.TextColumn("nom"),
                                       "x1": st.column_config.NumberColumn("x1", format="%.0f"), "y1": st.column_config.NumberColumn("y1", format="%.0f"),
                                       "x2": st.column_config.NumberColumn("x2", format="%.0f"), "y2": st.column_config.NumberColumn("y2", format="%.0f"),
                                       "xc": st.column_config.NumberColumn("xc", format="%.0f"), "yc": st.column_config.NumberColumn("yc", format="%.0f"),
                                       "r": st.column_config.NumberColumn("r", format="%.0f"),
                                       "ang1": st.column_config.NumberColumn("ang1 °", format="%.0f"), "ang2": st.column_config.NumberColumn("ang2 °", format="%.0f"),
                                       "a": st.column_config.NumberColumn("a (mm)", format="%.1f", min_value=1.0),
                                       "cote": st.column_config.SelectboxColumn("côté", options=[1, -1], required=True),
                                       "n": st.column_config.NumberColumn("tronçons n", format="%d", min_value=0, step=1),
                                       "l": st.column_config.NumberColumn("l tronçon", format="%.0f", min_value=1.0)})
    nouveaux = elements_depuis_lignes(ed.to_dict("records"))
    if nouveaux != cas["params"].get("elements"):
        cas["params"]["elements"] = nouveaux
        cas["params"].pop("cordons", None)
        st.rerun()
    c1, c2, c3 = st.columns(3)
    with c1:
        _num("x du point de référence R (mm)", ("params", "xR"), cas, 5.0, "%.0f")
    with c2:
        _num("y de R (mm)", ("params", "yR"), cas, 5.0, "%.0f")
    with c3:
        _num("z de R (mm, hors du plan)", ("params", "zR"), cas, 5.0, "%.0f", help="Une force dans le plan appliquée à z ≠ 0 crée un moment de flexion (transport en G)")


# ---------------------------------------------------------------- onglet 2 : cordons
def _etape_cordons(cas: Dict, res: Dict):
    if res["statut"] == ET.REFUS and not res.get("chaines"):
        st.warning((res["messages"] or ["Calcul impossible"])[0])
        return
    if res.get("pleine"):
        st.caption("Pleine pénétration : aucun cordon à vérifier, la pièce la plus faible gouverne (§4.7.1).")
        return
    pr = res.get("proprietes") or {}
    st.caption(f"A_w = {_fmt(pr.get('A_w', 0), 0)} mm² · G = ({_fmt(pr.get('xG', 0), 1)} ; {_fmt(pr.get('yG', 0), 1)}) · "
               f"I_x = {_fmt(pr.get('Ix', 0) / 1e4, 1)} cm⁴ · I_y = {_fmt(pr.get('Iy', 0) / 1e4, 1)} cm⁴ · I_xy = {_fmt(pr.get('Ixy', 0) / 1e4, 1)} cm⁴ · "
               f"J = {_fmt(pr.get('J', 0) / 1e4, 1)} cm⁴ · {pr.get('n_cordons', 0)} segment(s) de calcul · βLw = {res.get('beta_Lw', 1.0):.3f}")
    lignes = ""
    for c in res.get("chaines", []):
        nat = "pénétration partielle" if c["bout_a_bout"] else ("pourtour" if c["ferme"] else ("chaîne (l − 2a)" if c["continu"] else ("retours (l entière)" if c["retours"] else "l − 2a")))
        lignes += (f"<tr><td><b>{c['nom']}</b></td><td class='n'>{_fmt(c['a'], 1)}</td><td class='n'>{_fmt(c['l'], 0)}</td><td class='n'>{_fmt(c['l_eff'], 0)}</td>"
                   f"<td>{nat}</td><td class='n'>{_fmt(c['f_max'], 0)}</td><td class='n {_cls(c['taux'])}'>{c['taux'] * 100:.0f} %</td><td>{c['position']}</td></tr>")
    st.markdown("<table class='sdr-tab'><tr><th>Cordon</th><th class='n'>a (mm)</th><th class='n'>l (mm)</th><th class='n'>l_eff (mm)</th><th>longueur efficace</th>"
                "<th class='n'>|f| max (N/mm)</th><th class='n'>η</th><th>au point</th></tr>" + lignes + "</table>", unsafe_allow_html=True)
    if res.get("cordons_exclus"):
        noms = sorted({c["chaine"] for c in res["cordons_exclus"]})
        st.caption("Ne comptent pas (gorge < 3 mm ou l − 2a < max(30 mm ; 6a)) : " + ", ".join(noms))


# ---------------------------------------------------------------- onglet 3 : efforts
def _etape_efforts(cas: Dict, res: Dict):
    m = cas["methode"]
    if m["mode"] == K.ELS:
        st.caption("**Efforts de service** (méthode ELS du bureau : sans coefficient de charge), dans le sens du dessin ; une valeur négative retourne la flèche.")
    else:
        st.caption("**Efforts de calcul à l'ELU** (γG·G + γQ·Q), dans le sens du dessin ; une valeur négative retourne la flèche.")
    champs = CF.EFFORTS[cas["config"]]
    cols = st.columns(3)
    for i, (cle, lib, unite, aide) in enumerate(champs):
        with cols[i % 3]:
            _num(lib, ("efforts", cle), cas, 1.0 if unite == "kN" else 0.5, "%.1f", help=aide)
    e = res.get("efforts")
    if e is not None:
        st.caption(f"Au point de référence ({_fmt(e.reference[0], 0)} ; {_fmt(e.reference[1], 0)} ; {_fmt(e.reference[2], 0)}) mm dans le plan de joint : "
                   f"V_x = {_fmt(e.Vx / 1e3)} kN, V_y = {_fmt(e.Vy / 1e3)} kN, N = {_fmt(e.N / 1e3)} kN, "
                   f"M_x = {_fmt(e.Mx / 1e6, 2)} kN·m, M_y = {_fmt(e.My / 1e6, 2)} kN·m, M_z = {_fmt(e.Mz / 1e6, 2)} kN·m")


# ---------------------------------------------------------------- onglet 4 : méthode
def _etape_methode(etat: Dict, cas: Dict, res: Dict):
    m = cas["methode"]
    acier = ET.acier_cordon(cas["params"])
    st.markdown("**Eurocode (EN 1993-1-8 §4.5.3)**")
    c1, c2 = st.columns(2)
    with c1:
        _choix("Critère", ("methode", "ec"), cas, [K.DIRECTIONNELLE, K.SIMPLIFIEE], CRITERES, segmente=True,
               help="Directionnelle : √(σ⊥² + 3(τ⊥² + τ∥²)) ≤ fu/(βw·γM2) et σ⊥ ≤ 0,9·fu/γM2 ; simplifiée : |f| ≤ f_vw,d·a, toujours du côté sûr")
    with c2:
        _num("γM2", ("methode", "gamma_M2"), cas, 0.05, "%.2f", 1.0, help="1,25 : NBN EN 1993-1-8 ANB")
    d = K.directionnelle(0, 0, 0, acier["fu"], acier["bw"], m["gamma_M2"])
    s = K.simplifiee(0, 1.0, acier["fu"], acier["bw"], m["gamma_M2"])
    st.caption(f"Acier de référence {acier['nuance_ref']} (la pièce la plus faible) : fu = {acier['fu']:g} N/mm², βw = {acier['bw']:g} → "
               f"fu/(βw·γM2) = {d['lim_c']:.1f} N/mm², 0,9·fu/γM2 = {d['lim_perp']:.1f} N/mm², f_vw,d = {s['f_vw_d']:.1f} N/mm².")
    st.markdown("**Méthode ELS du bureau**")
    c1, c2, c3 = st.columns(3)
    with c1:
        _num("k (kg par mm de gorge et par cm de cordon)", ("methode", "k"), cas, 5.0, "%.1f", 1.0)
    with c2:
        _choix("g (N par kg)", ("methode", "g"), cas, [9.81, 10.0], {9.81: "9,81 N/kg", 10.0: "10 N/kg"})
    with c3:
        _num("γ_F (ELU/ELS) pour l'équivalent Eurocode", ("methode", "gamma_F"), cas, 0.05, "%.2f", 1.0)
    kq = K.k_equivalent_eurocode(acier["fu"], acier["bw"], m["gamma_M2"], m["gamma_F"], m["g"])
    with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="center", key=P + "c_kq"):
        st.markdown(f"k = {m['k']:g} kg/(mm·cm) = **{K.tau_adm_bureau(m['k'], m['g']):.1f} N/mm²** sur la gorge · l'Eurocode au même niveau de service "
                    f"donne f_vw,d/γ_F = **{kq['tau_N_mm2']:.1f} N/mm² = {kq['k_kg_mm_cm']:.0f} kg/(mm·cm)** pour le {acier['nuance_ref']}")
        if st.button(f"Prendre la valeur Eurocode ({kq['k_kg_mm_cm']:.0f})", key=P + "btn_kq", width="content"):
            cas["methode"]["k"] = round(kq["k_kg_mm_cm"], 1)
            _changer(f"k = {kq['k_kg_mm_cm']:.0f} kg/(mm·cm)")
        if st.button("Revenir à 100", key=P + "btn_k100", width="content"):
            cas["methode"]["k"] = K.K_BUREAU
            _changer("k = 100 kg/(mm·cm)")
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
        for sec in ME.SECTIONS:
            st.markdown(f"**{sec['titre']}**")
            for p in sec["paragraphes"]:
                st.markdown(p)
        st.markdown("**Références** — " + " · ".join(ME.REFERENCES))


# ---------------------------------------------------------------- onglet 5 : détail
def _etape_detail(cas: Dict, res: Dict):
    if res["statut"] == ET.REFUS:
        st.warning((res["messages"] or ["Calcul impossible"])[0])
        return
    if res.get("pleine"):
        pc = res["piece"]
        st.caption(f"Pièce {cas['params']['b']:g} × {cas['params']['t']:g} mm : σ = {_fmt(pc['sigma'])} N/mm², τ = {_fmt(pc['tau'])} N/mm², "
                   f"σ_c = {_fmt(pc['sigma_c'])} ≤ {_fmt(pc['lim'])} N/mm² → η = {pc['taux'] * 100:.0f} %")
        return
    g = res["gouvernant"]
    r = res.get("resistances") or {}
    st.caption(f"Par cordon et par point (début, milieu, fin de la longueur efficace) : force par unité de longueur |f|, composantes sur la gorge, "
               f"σ_c = √(σ⊥² + 3(τ⊥² + τ∥²)), et le taux selon chaque méthode. Limites : fu/(βw·γM2) = {r.get('lim_c', 0):.1f}, "
               f"0,9·fu/γM2 = {r.get('lim_perp', 0):.1f}, f_vw,d = {r.get('f_vw_d', 0):.1f} N/mm² · bureau {r.get('tau_adm_bureau', 0):.1f} N/mm².")
    lignes = ""
    chaines = {c["nom"]: c["chaine"] for c in res["cordons"]}
    for p in res["points"]:
        gouv = p is g
        lignes += (f"<tr class='{'g' if gouv else ''}'><td>{chaines.get(p['cordon'], p['cordon'])}</td><td>{p['position']}</td>"
                   f"<td class='n'>{_fmt(p['f'], 0)}</td><td class='n'>{_fmt(p['sigma_perp'])}</td><td class='n'>{_fmt(p['tau_perp'])}</td>"
                   f"<td class='n'>{_fmt(p['tau_par'])}</td><td class='n'>{_fmt(p['sigma_c'])}</td>"
                   f"<td class='n {_cls(p['taux_directionnelle'])}'>{p['taux_directionnelle'] * 100:.0f} %</td>"
                   f"<td class='n {_cls(p['taux_simplifiee'])}'>{p['taux_simplifiee'] * 100:.0f} %</td>"
                   f"<td class='n {_cls(p['taux_bureau'])}'>{p['taux_bureau'] * 100:.0f} %</td>"
                   f"<td class='n {_cls(p['taux'])}'>{p['taux'] * 100:.0f} %</td></tr>")
    st.markdown("<table class='sdr-tab'><tr><th>Cordon</th><th>point</th><th class='n'>|f| N/mm</th><th class='n'>σ⊥</th><th class='n'>τ⊥</th><th class='n'>τ∥</th>"
                "<th class='n'>σ_c</th><th class='n'>η dir.</th><th class='n'>η simpl.</th><th class='n'>η ELS</th><th class='n'>η choisie</th></tr>"
                + lignes + "</table>", unsafe_allow_html=True)


def _cls(t):
    return "sdr-r" if t > 1 else ("sdr-o" if t > 0.85 else "sdr-g")


# ---------------------------------------------------------------- page
def show():
    etat = _etat()
    cas = ET.cas_actif(etat)
    LI.au_rendu(P, REGISTRE, cas)          # une synchronisation demandée hors callback se fait ici, avant tout widget
    _entete(etat, cas)
    _switch(cas)
    res = ET.verifier(cas)
    col_g, col_d = st.columns([1.15, 1], gap="large")
    with col_g:
        _colonne_dessin(etat, cas, res)
    with col_d:
        _carte_resultat(cas, res)
        onglets = st.tabs(ETAPES)
        with onglets[0]:
            _etape_configuration(etat, cas, res)
        with onglets[1]:
            _etape_cordons(cas, res)
        with onglets[2]:
            _etape_efforts(cas, res)
        with onglets[3]:
            _etape_methode(etat, cas, res)
        with onglets[4]:
            _etape_detail(cas, res)

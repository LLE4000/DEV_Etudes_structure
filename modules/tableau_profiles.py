# -*- coding: utf-8 -*-
"""Catalogue des profilés métalliques — la vue (Streamlit).

Tout ce qui est calculé ou décidé vit dans ``acier/profiles`` (catalogue,
familles, géométrie, dessin) ; ce fichier n'est qu'une page : familles en
pastilles, recherche, filtres numériques, tri, colonnes au choix,
préférences gardées par défaut, tableau cliquable, fiche du profilé avec
son dessin, comparaison, profilé personnalisé (non standard), export CSV.
Une seule arborescence pour PC, tablette et téléphone (R1) : les colonnes
se replient d'elles-mêmes sous 640 px.
"""
import csv
import io
import json
from typing import Dict, List

import pandas as pd
import streamlit as st

from acier.profiles import catalogue as CAT
from acier.profiles import dessin as DS
from acier.profiles import familles as F
from modules import navigation

P = "prf_"
VERSION = "1.0"
N_FILTRES = 3
CSS = """<style>
[class*="st-key-prf_"] [data-testid="stExpander"] details{border-radius:10px}
.prf-fiche{border:1px solid #e5e7eb;border-radius:12px;padding:12px 14px;background:#fff}
.prf-fiche .t{font-size:11px;letter-spacing:.08em;color:#6b7280;font-weight:600}
.prf-fiche h3{margin:2px 0 0;font-size:20px}
.prf-tab{width:100%;border-collapse:collapse;font-size:12.5px}
.prf-tab th{text-align:left;font-size:10.5px;letter-spacing:.05em;color:#6b7280;padding:4px 6px;border-bottom:1px solid #e5e7eb;font-weight:600}
.prf-tab td{padding:4px 6px;border-bottom:1px solid #f3f4f6;vertical-align:top}
.prf-tab td.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.prf-tab td.g{color:#6b7280;font-size:10.5px;letter-spacing:.05em;padding-top:9px;font-weight:600;border-bottom:none}
.prf-note{font-size:12px;color:#6b7280}
.prf-chip{display:inline-block;border-radius:999px;padding:2px 9px;font-size:11.5px;background:#eef2ff;color:#3730a3;margin-right:4px}
</style>"""


# ---------------------------------------------------------------- état
def _init():
    """Session initialisée depuis les préférences enregistrées (défauts du bureau)."""
    if P + "init" in st.session_state:
        return
    prefs = CAT.preferences()
    st.session_state[P + "init"] = True
    st.session_state[P + "familles"] = list(prefs["familles"])
    st.session_state[P + "colonnes"] = list(prefs["colonnes"])
    st.session_state[P + "tri"] = prefs["tri"]
    st.session_state[P + "decroissant"] = bool(prefs["decroissant"])
    st.session_state[P + "courants"] = bool(prefs["courants_seulement"])
    st.session_state[P + "filtres"] = [{"cle": "", "min": None, "max": None} for _ in range(N_FILTRES)]
    st.session_state[P + "recherche"] = ""
    st.session_state[P + "fiche"] = None
    st.session_state[P + "persos"] = []
    st.session_state[P + "compare"] = []


def _prefs_courantes() -> Dict:
    return {"colonnes": st.session_state[P + "colonnes"], "familles": st.session_state[P + "familles"],
            "tri": st.session_state[P + "tri"], "decroissant": st.session_state[P + "decroissant"],
            "courants_seulement": st.session_state[P + "courants"], "decimales_fixes": None}


def _sync_pills(groupes):
    """Les pastilles de familles sont des VUES de la liste d'état : on les
    règle avant de les créer, et leur changement réécrit la liste."""
    for groupe, fams in groupes:
        cle = P + "w_fam_" + groupe
        if cle not in st.session_state:
            st.session_state[cle] = [f for f in fams if f in st.session_state[P + "familles"]]


def _lire_pills(groupes):
    sel = []
    for groupe, fams in groupes:
        sel += [f for f in fams if f in st.session_state.get(P + "w_fam_" + groupe, [])]
    st.session_state[P + "familles"] = sel


def _reset_pills(groupes, familles):
    st.session_state[P + "familles"] = list(familles)
    for groupe, fams in groupes:
        st.session_state[P + "w_fam_" + groupe] = [f for f in fams if f in familles]
    for g in CAT.GROUPES_COLONNES:
        st.session_state.pop(P + "w_col_" + g, None)


# ---------------------------------------------------------------- tableau
def _lignes_tableau(liste: List[Dict], colonnes: List[str]) -> pd.DataFrame:
    lignes = []
    for p in liste:
        l = {"Profilé": p["nom"], "Famille": p["famille"]}
        for c in colonnes:
            v = p.get(c)
            l[_entete(c)] = None if v is None else round(float(v), CAT.DECIMALES.get(c, 2))
        lignes.append(l)
    return pd.DataFrame(lignes)


def _entete(c: str) -> str:
    u = CAT.UNITE.get(c, "")
    return f"{CAT.LIBELLE[c]} ({u})" if u else CAT.LIBELLE[c]


def _csv(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=";")
    w.writerow(df.columns)
    for _, row in df.iterrows():
        w.writerow([("" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v).replace(".", ",")) for v in row])
    return buf.getvalue().encode("utf-8-sig")


def _table_fiche(p: Dict) -> str:
    """Toutes les propriétés d'un profilé, par groupe, en HTML."""
    lignes = []
    for g in CAT.GROUPES_COLONNES:
        cles = [c for c in CAT.colonnes_du_groupe(g) if p.get(c) is not None]
        if not cles:
            continue
        lignes.append(f"<tr><td class='g' colspan='3'>{g.upper()}</td></tr>")
        for c in cles:
            aide = CAT.AIDE_COLONNES.get(c, "")
            lignes.append(f"<tr><td title=\"{aide}\">{CAT.LIBELLE[c]}</td><td class='n'>{CAT.formater(p[c], c)}</td>"
                          f"<td class='prf-note'>{CAT.UNITE.get(c, '')}</td></tr>")
    return "<table class='prf-tab'>" + "".join(lignes) + "</table>"


def _table_comparaison(profils: List[Dict], colonnes: List[str]) -> str:
    en_tete = "".join(f"<th class='n'>{p['nom']}</th>" for p in profils)
    lignes = []
    for c in colonnes:
        if all(p.get(c) is None for p in profils):
            continue
        cellules = "".join(f"<td class='n'>{CAT.formater(p.get(c), c)}</td>" for p in profils)
        lignes.append(f"<tr><td>{_entete(c)}</td>{cellules}</tr>")
    return f"<table class='prf-tab'><tr><th>Propriété</th>{en_tete}</tr>" + "".join(lignes) + "</table>"


# ---------------------------------------------------------------- profilé personnalisé
def _nom_perso(fam: str, dims: Dict[str, float]) -> str:
    f = F.FAMILLES[fam]
    return f"{fam} " + "x".join(f"{float(dims[k]):g}" for k in f["dims"])


def _perso():
    """Un profilé défini par ses dimensions (PRS, caisson, tube, plat, rond) :
    calculé par le MÊME moteur que le catalogue, puis ajouté au tableau."""
    fams = [f for f, d in F.FAMILLES.items() if d.get("non_standard")]
    fam = st.selectbox("Forme", fams, format_func=lambda f: F.FAMILLES[f]["libelle"], key=P + "perso_fam")
    f = F.FAMILLES[fam]
    st.caption(f["description"])
    defauts = {"h": 400.0, "b": 200.0, "tw": 8.0, "tf": 12.0, "t": 6.0, "d": 100.0}
    with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=P + "perso_dims"):
        dims = {k: st.number_input(F.LIBELLE_DIM[k], min_value=0.1, value=defauts[k], step=0.5, key=P + f"perso_{fam}_{k}", width=130)
                for k in f["dims"]}
        ajouter = st.button("Calculer et ajouter au tableau", key=P + "perso_btn", width="content")
    if ajouter:
        try:
            if fam in ("PRS",) and 2 * dims["tf"] >= dims["h"]:
                raise ValueError("h doit dépasser 2·tf")
            if fam in ("CAISSON", "TUBE") and 2 * dims["t"] >= min(v for k, v in dims.items() if k in ("h", "b", "d")):
                raise ValueError("t doit être inférieure à la demi-dimension")
            p = F.calculer(fam, dims, torsion_exacte=False)
            p.update({"nom": _nom_perso(fam, dims), "famille": fam, "courant": False, "dims": dict(dims), "perso": True})
            persos = [q for q in st.session_state[P + "persos"] if q["nom"] != p["nom"]] + [p]
            st.session_state[P + "persos"] = persos
            st.session_state[P + "fiche"] = p["nom"]
            st.session_state[P + "fiche_suivre"] = p["nom"]           # la liste déroulante suit l'ajout (au rendu suivant :
            st.rerun()                                                 # sa clé ne se modifie pas après sa création)
        except ValueError as e:
            st.error(f"Dimensions impossibles : {e}")


# ---------------------------------------------------------------- page
def show():
    _init()
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("## Catalogue des profilés")
    st.caption("Profilés laminés, creux et reconstitués — dimensions normalisées, propriétés calculées depuis le contour exact "
               f"· {len(CAT.profils())} profilés · version {VERSION}")
    groupes = CAT.groupes_presents()
    toutes = [f for _, fams in groupes for f in fams]

    # ---- barre d'outils
    with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="center", key=P + "barre"):
        navigation.bouton_accueil(P + "btn_home", width="content")
        if st.button("💾 Garder par défaut", key=P + "btn_save", width="content",
                     help="Enregistre familles, colonnes, tri et « courants seulement » comme réglage par défaut de la page."):
            CAT.enregistrer_preferences(_prefs_courantes())
            st.toast("Réglages gardés par défaut")
        if st.button("↩ Rétablir", key=P + "btn_reset", width="content", help="Revient aux défauts livrés et les enregistre."):
            CAT.retablir_preferences()
            prefs = CAT.preferences()
            _reset_pills(groupes, prefs["familles"])
            st.session_state[P + "colonnes"] = list(prefs["colonnes"])
            st.session_state[P + "tri"] = prefs["tri"]
            st.session_state[P + "decroissant"] = False
            st.session_state[P + "courants"] = False
            st.session_state[P + "filtres"] = [{"cle": "", "min": None, "max": None} for _ in range(N_FILTRES)]
            for k in [k for k in st.session_state if k.startswith(P + "w_")]:
                del st.session_state[k]
            st.rerun()
        st.toggle("Séries courantes seulement", key=P + "courants",
                  help="IPE, HE A/B/M, UPE, UPN, cornières, tubes ; masque IPE A/O, IPN, HE AA, HL, HD, HP.")
        if st.button("Toutes les familles", key=P + "btn_tout", width="content"):
            _reset_pills(groupes, toutes)
            st.rerun()
        if st.button("Aucune", key=P + "btn_rien", width="content"):
            _reset_pills(groupes, [])
            st.rerun()

    # ---- familles : une rangée de pastilles par groupe
    _sync_pills(groupes)
    comptes = dict(CAT.familles_presentes())
    with st.container(horizontal=True, wrap=True, gap="medium", vertical_alignment="top", key=P + "fams"):
        for groupe, fams in groupes:
            st.pills(groupe, fams, selection_mode="multi", key=P + "w_fam_" + groupe,
                     format_func=lambda f: f"{f} · {comptes.get(f, 0)}", width="content")
    _lire_pills(groupes)
    familles = st.session_state[P + "familles"]

    # ---- recherche, tri, filtres
    cles_num = [c[0] for c in CAT.COLONNES]
    with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=P + "recherche_l"):
        st.text_input("Rechercher", key=P + "recherche", placeholder="IPE 200 · 100x100 · HEB…", width=230)
        st.selectbox("Trier par", [""] + cles_num, key=P + "tri", format_func=lambda c: "ordre du catalogue" if c == "" else _entete(c), width=200)
        st.toggle("Décroissant", key=P + "decroissant")
    with st.expander("Filtres numériques — par exemple Iy ≥ 1 500 cm⁴, masse ≤ 30 kg/m", expanded=any(f["cle"] for f in st.session_state[P + "filtres"])):
        filtres = st.session_state[P + "filtres"]
        for i in range(N_FILTRES):
            with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=P + f"filtre_{i}"):
                cle = st.selectbox("Propriété", [""] + cles_num, key=P + f"w_f_cle_{i}",
                                   index=([""] + cles_num).index(filtres[i]["cle"]) if filtres[i]["cle"] in cles_num else 0,
                                   format_func=lambda c: "—" if c == "" else _entete(c), width=200)
                lo = st.number_input("≥ (min)", value=filtres[i]["min"], key=P + f"w_f_min_{i}", width=150, placeholder="min")
                hi = st.number_input("≤ (max)", value=filtres[i]["max"], key=P + f"w_f_max_{i}", width=150, placeholder="max")
                filtres[i] = {"cle": cle, "min": lo, "max": hi}
        if st.button("Effacer les filtres", key=P + "btn_filtres_rien", width="content"):
            st.session_state[P + "filtres"] = [{"cle": "", "min": None, "max": None} for _ in range(N_FILTRES)]
            for k in [k for k in st.session_state if k.startswith(P + "w_f_")]:
                del st.session_state[k]
            st.rerun()
    bornes = {f["cle"]: (f["min"], f["max"]) for f in st.session_state[P + "filtres"] if f["cle"]}

    # ---- colonnes
    with st.expander(f"Colonnes affichées — {len(st.session_state[P + 'colonnes'])} sur {len(cles_num)}"):
        choix = []
        for g in CAT.GROUPES_COLONNES:
            opts = CAT.colonnes_du_groupe(g)
            cle = P + "w_col_" + g
            if cle not in st.session_state:
                st.session_state[cle] = [c for c in opts if c in st.session_state[P + "colonnes"]]
            choix += st.pills(g, opts, selection_mode="multi", key=cle, format_func=_entete, width="content") or []
        st.session_state[P + "colonnes"] = [c for c in cles_num if c in choix]
        st.caption("Les colonnes vides pour les familles affichées sont masquées d'elles-mêmes. "
                   "Survoler l'en-tête d'une fiche donne la définition.")
    colonnes = st.session_state[P + "colonnes"]

    # ---- la liste
    base = CAT.filtrer(CAT.profils(), familles, st.session_state[P + "recherche"], bornes, st.session_state[P + "courants"])
    persos = CAT.filtrer(st.session_state[P + "persos"], None, st.session_state[P + "recherche"], bornes, False)
    liste = base + persos
    tri = st.session_state[P + "tri"]
    liste = CAT.trier(liste, tri, st.session_state[P + "decroissant"]) if tri else liste
    cols_vis = CAT.colonnes_utiles(liste, colonnes)
    noms = [p["nom"] for p in liste]
    if st.session_state[P + "fiche"] not in noms:
        st.session_state[P + "fiche"] = noms[0] if noms else None

    gauche, droite = st.columns([1.55, 1], gap="medium")
    with gauche:
        with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="center", key=P + "entete_table"):
            st.markdown(f"**{len(liste)} profilé{'s' if len(liste) > 1 else ''}**"
                        + (f" · {len(familles)} famille{'s' if len(familles) > 1 else ''}" if familles else "")
                        + (f" · {len(persos)} personnalisé{'s' if len(persos) > 1 else ''}" if persos else ""))
            df = _lignes_tableau(liste, cols_vis)
            st.download_button("⬇️ CSV", _csv(df), "profiles.csv", "text/csv", key=P + "btn_csv", width="content",
                               help="Le tableau affiché, séparateur ; et virgule décimale.")
        if not liste:
            st.info("Aucun profilé : élargir les familles, la recherche ou les filtres.")
        else:
            config = {"Profilé": st.column_config.TextColumn("Profilé", pinned=True, width="medium"),
                      "Famille": st.column_config.TextColumn("Famille", width="small")}
            for c in cols_vis:
                config[_entete(c)] = st.column_config.NumberColumn(_entete(c), format="localized", help=CAT.AIDE_COLONNES.get(c))
            ev = st.dataframe(df, hide_index=True, column_config=config, key=P + "table", on_select="rerun",
                              selection_mode="single-row", height=min(760, 35 * (len(df) + 1) + 12), width="stretch")
            sel = list(ev.selection.rows) if ev is not None and hasattr(ev, "selection") else []
            if sel:
                st.session_state[P + "fiche"] = noms[sel[0]]
                st.session_state[P + "w_fiche"] = noms[sel[0]]        # la liste déroulante suit le clic
            st.caption("Clic sur une ligne : la fiche à droite · clic sur un en-tête : tri de l'affichage · ⋮ : recherche et plein écran.")

    with droite:
        if noms:
            suivre = st.session_state.pop(P + "fiche_suivre", None)
            if suivre in noms:
                st.session_state[P + "w_fiche"] = suivre
            if st.session_state.get(P + "w_fiche") not in noms:
                st.session_state[P + "w_fiche"] = st.session_state[P + "fiche"]

            def _sel_fiche():
                st.session_state[P + "fiche"] = st.session_state[P + "w_fiche"]
            st.selectbox("Fiche du profilé", noms, key=P + "w_fiche", on_change=_sel_fiche)
            p = next(q for q in liste if q["nom"] == st.session_state[P + "fiche"])
            f = F.FAMILLES[p["famille"]]
            chips = f"<span class='prf-chip'>{f['libelle']}</span><span class='prf-chip'>{f['norme']}</span>" \
                    + ("<span class='prf-chip'>personnalisé</span>" if p.get("perso") else "")
            st.markdown(f"<div class='prf-fiche'><div class='t'>PROFILÉ</div><h3>{p['nom']}</h3>{chips}</div>", unsafe_allow_html=True)
            try:
                st.markdown(DS.svg_section(p["famille"], p["dims"], 260, p), unsafe_allow_html=True)
            except Exception as e:  # noqa: BLE001 — la fiche reste lisible sans dessin
                st.caption(f"dessin indisponible : {e}")
            st.markdown(_table_fiche(p), unsafe_allow_html=True)
            if p.get("methode_It"):
                st.caption(f"It : {p['methode_It']}.")
            st.download_button("⬇️ Fiche (JSON)", json.dumps(p, ensure_ascii=False, indent=1).encode("utf-8"),
                               f"{p['nom'].replace(' ', '_')}.json", "application/json", key=P + "btn_json", width="content")
        with st.expander("Comparer des profilés"):
            choix = st.multiselect("Profilés à comparer (jusqu'à 5)", noms, key=P + "compare", max_selections=5)
            if len(choix) >= 2:
                st.markdown(_table_comparaison([next(q for q in liste if q["nom"] == n) for n in choix], cols_vis or CAT.DEFAUT_COLONNES),
                            unsafe_allow_html=True)
            else:
                st.caption("Choisir au moins deux profilés du tableau.")
        with st.expander("Profilé personnalisé (non standard)"):
            _perso()

    with st.expander("Méthode et sources"):
        m = CAT.meta()
        st.markdown(f"**Source unique** : `acier/donnees/profiles.json` (version {m['version']}, {m['date']}, {m['nombre']} profilés). "
                    f"Conventions : {m['conventions']}.")
        st.markdown("**Dimensions** :\n" + "\n".join(f"- {s}" for s in m["sources_dimensions"]))
        st.markdown(f"**Calcul** : {m['calcul']}. **Contrôle croisé** : {m['controle']}.")
        st.markdown("**Définitions** :\n" + "\n".join(f"- **{CAT.LIBELLE[c]}** — {t}" for c, t in CAT.AIDE_COLONNES.items()))
        st.markdown("**Familles** :\n" + "\n".join(f"- **{f}** — {d['libelle']} · {d['norme']} · {d['description']}" for f, d in F.FAMILLES.items()))

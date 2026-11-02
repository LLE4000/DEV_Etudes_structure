# -*- coding: utf-8 -*-
"""Écran Streamlit du module « Charges sur plancher » (lot 2).

Le moteur, l'état et les dessins vivent dans ``plancher/charges`` (sans
Streamlit : ``tests/test_charges_plancher_moteur.py`` le garde) ; ce fichier
n'est que la vue.

Organisation (étude § 7.2, maquette « digeste ») : barre d'outils du cas ;
à gauche le PLAN cliquable et déplaçable (composant ``svg_cliquable``), la
coupe sous l'appui sélectionné et le balayage ; à droite le RÉSULTAT en
une phrase avec ses barres et ses propositions, puis quatre étapes
en onglets : Plancher, Charges, Répartiteur, Hypothèses ; en bas, le détail
des vérifications. Une seule source : l'état ``st.session_state["chp"]``
(``plancher/charges/etat.py``) ; les widgets des étapes en sont des vues (clés ``chp_w_``,
réinitialisées dès que l'état change par ailleurs).
"""
import json
from typing import Dict, List

import streamlit as st

from plancher.charges import etat as ET
from plancher.charges import verification as V
from plancher.charges import dessin as DS
from plancher.charges import symboles as SY
from plancher.charges import bibliotheque as BIB
from plancher.charges import resistances as RS

P = "chp_"
VUES = ["Plan", "Coupe", "Balayage"]
ETAPES = ["1 · Plancher", "2 · Charges", "3 · Répartiteur", "4 · Hypothèses"]
TYPES = {"hourdis": "Hourdis", "dalle": "Dalle BA", "predalle": "Prédalle"}
ETATS_H = {"poses": "Posés", "clavetes": "Clavetés", "chapes": "Chapés"}
FAMILLES = [("flexion", "Flexion"), ("tranchant", "Tranchant"), ("local", "Local"), ("plateau", "Plateaux"),
            ("filiere", "Filières"), ("joint", "Joints")]
CSS = """<style>
[class*="st-key-chp_"] [data-testid="stExpander"] details{border-radius:10px}
.chp-res{border:1px solid #e5e7eb;border-radius:10px;padding:12px 14px;background:#fff;margin-bottom:8px}
.chp-res .t{font-size:11px;letter-spacing:.08em;color:#6b7280;font-weight:600}
.chp-pill{float:right;font-size:12px;border-radius:999px;padding:3px 10px;font-weight:600}
.chp-pill.nok{background:#fee2e2;color:#b91c1c}.chp-pill.ok{background:#dcfce7;color:#15803d}.chp-pill.att{background:#ffedd5;color:#c2410c}.chp-pill.ref{background:#f3f4f6;color:#374151}
.chp-row{display:grid;grid-template-columns:96px 1fr 56px;align-items:center;gap:10px;font-size:13px;margin:6px 0}
.chp-trk{height:6px;background:#f1f3f5;border-radius:3px;overflow:hidden}.chp-trk b{display:block;height:100%}
.chp-v{text-align:right;font-variant-numeric:tabular-nums;font-weight:600}
.chp-phrase{font-size:14px;margin:4px 0 6px}
.chp-note{font-size:12px;color:#6b7280}
.chp-tab{width:100%;border-collapse:collapse;font-size:12.5px}.chp-tab th{text-align:left;font-size:10.5px;letter-spacing:.05em;color:#6b7280;padding:4px 6px;border-bottom:1px solid #e5e7eb}
.chp-tab td{padding:5px 6px;border-bottom:1px solid #f3f4f6;vertical-align:top}.chp-tab td.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.chp-r{color:#b91c1c;font-weight:700}.chp-o{color:#c2410c;font-weight:700}.chp-g{color:#15803d;font-weight:700}.chp-i{color:#6b7280}
.chp-leg{font-size:11.5px;color:#6b7280}
</style>"""


# ---------------------------------------------------------------- état et vues des widgets
def _etat() -> Dict:
    if "chp" not in st.session_state:
        st.session_state["chp"] = ET.nouvel_etat()
    return st.session_state["chp"]


def _reset_widgets():
    """Les widgets des étapes relisent l'état au prochain rendu."""
    for k in [k for k in st.session_state if k.startswith(P + "w_")]:
        del st.session_state[k]


def _chemin(d: Dict, chemin):
    for c in chemin[:-1]:
        d = d[c]
    return d, chemin[-1]


def _ecrire(chemin, cle, conv=None):
    """Callback : écrit la valeur du widget ``cle`` dans l'état au chemin donné."""
    v = st.session_state.get(cle)
    if conv is not None:
        v = conv(v)
    d, k = _chemin(ET.cas_actif(_etat()), chemin)
    d[k] = v


def _num(label, chemin, cas, step=0.1, fmt="%.2f", min_value=None, nul_vaut_none=False, help=None, aide_vide="—"):
    """number_input lié à ``chemin`` du cas ; ``nul_vaut_none`` : 0 = non renseigné (None)."""
    d, k = _chemin(cas, chemin)
    val = d.get(k)
    cle = P + "w_" + "_".join(str(c) for c in chemin)
    if nul_vaut_none:
        conv = (lambda v: None if v in (None, 0, 0.0) else float(v))
        v0 = 0.0 if val is None else float(val)
    else:
        conv = (lambda v: None if v is None else (int(v) if isinstance(step, int) else float(v)))
        v0 = val if val is not None else 0
    kw = {} if min_value is None else {"min_value": min_value}
    st.number_input(label, value=v0, step=step, format=fmt if not isinstance(step, int) else "%d", key=cle,
                    on_change=_ecrire, args=(chemin, cle, conv), help=help, **kw)


def _texte_w(label, chemin, cas, help=None):
    d, k = _chemin(cas, chemin)
    cle = P + "w_" + "_".join(str(c) for c in chemin)
    st.text_input(label, value=d.get(k) or "", key=cle, on_change=_ecrire, args=(chemin, cle, None), help=help)


def _choix(label, chemin, cas, options, affichage=None, help=None, segmente=False):
    d, k = _chemin(cas, chemin)
    cle = P + "w_" + "_".join(str(c) for c in chemin)
    cur = d.get(k)
    if cur not in options:
        cur = options[0]
    fmt = (lambda o: affichage.get(o, str(o))) if affichage else (lambda o: str(o))
    if segmente:
        st.segmented_control(label, options, default=cur, format_func=fmt, key=cle, on_change=_ecrire, args=(chemin, cle, None),
                             help=help, selection_mode="single")
    else:
        st.selectbox(label, options, index=options.index(cur), format_func=fmt, key=cle, on_change=_ecrire, args=(chemin, cle, None), help=help)


def _case(label, chemin, cas, help=None):
    d, k = _chemin(cas, chemin)
    cle = P + "w_" + "_".join(str(c) for c in chemin)
    st.checkbox(label, value=bool(d.get(k)), key=cle, on_change=_ecrire, args=(chemin, cle, bool), help=help)


def _changer(toast: str = ""):
    """Après un changement de l'état hors widgets : widgets relus, relance."""
    _reset_widgets()
    if toast:
        st.toast(toast)
    st.rerun()


def _fmt(x, nd=2):
    return f"{x:,.{nd}f}".replace(",", " ").replace(".", ",")


# ---------------------------------------------------------------- en-tête et barre d'outils
def _entete(etat: Dict, cas: Dict):
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("## Charges sur plancher")
    st.caption("Charges temporaires de chantier sur hourdis, dalle coulée en place ou prédalle — vérification du plancher, "
               "des plateaux et des filières · version 1.0 (lot 2)")
    with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="center", key=P + "barre"):
        if st.button("🏠 Accueil", key=P + "btn_home", width="content"):
            st.session_state.retour_accueil_demande = True
            st.rerun()
        noms = [c["nom"] for c in etat["cas"]]
        def _sel():
            etat["actif"] = noms.index(st.session_state[P + "sel_cas"])
            _reset_widgets()
        # le sélecteur SUIT l'état : après « ＋ Cas » ou « ⧉ », le cas actif a changé sans lui
        # (défaut trouvé par le harnais le 02/11/2026 : il affichait encore C1, cas C2 actif)
        st.session_state[P + "sel_cas"] = noms[etat["actif"]]
        st.selectbox("Cas", noms, key=P + "sel_cas", on_change=_sel, label_visibility="collapsed", width=220)
        if st.button("＋ Cas", key=P + "btn_new", width="content", help="Nouveau cas (même bibliothèque)"):
            etat["cas"].append(ET.nouveau_cas(f"C{len(etat['cas']) + 1}"))
            etat["actif"] = len(etat["cas"]) - 1
            _changer("Nouveau cas")
        if st.button("⧉", key=P + "btn_copy", width="content", help="Copier ce cas"):
            import copy
            d = copy.deepcopy(cas)
            d["nom"] = f"{cas['nom']} (copie)"
            etat["cas"].append(d)
            etat["actif"] = len(etat["cas"]) - 1
            _changer("Cas copié")
        if len(etat["cas"]) > 1 and st.button("🗑️", key=P + "btn_del", width="content", help="Supprimer ce cas"):
            etat["cas"].pop(etat["actif"])
            etat["actif"] = max(0, etat["actif"] - 1)
            _changer("Cas supprimé")
        st.download_button("💾 Enregistrer", data=ET.texte_json(etat).encode("utf-8"), file_name=f"charges_plancher_{cas['nom']}.json",
                           mime="application/json", key=P + "btn_save", width="content", help="Tous les cas et les fiches ajoutées (JSON)")
        if st.button("📂 Ouvrir", key=P + "btn_open", width="content"):
            st.session_state[P + "ouvrir"] = not st.session_state.get(P + "ouvrir", False)
        _texte_w("Nom du cas", ("nom",), cas)
    if st.session_state.get(P + "ouvrir"):
        up = st.file_uploader("Fichier JSON du module", type=["json"], key=P + "uploader", label_visibility="collapsed")
        if up is not None:
            try:
                st.session_state["chp"] = ET.charger(json.loads(up.read().decode("utf-8-sig")))
                st.session_state[P + "ouvrir"] = False
                _changer("Fichier chargé")
            except Exception as e:  # noqa: BLE001
                st.error(f"Fichier invalide : {e}")


# ---------------------------------------------------------------- colonne du plan
def _traiter(etat, cas, retour, index):
    if not retour or not isinstance(retour, dict):
        return
    derniers = st.session_state.setdefault(P + "dernier_clic", {})
    if derniers.get("plan") == retour.get("t"):
        return
    derniers["plan"] = retour.get("t")
    faits = ET.appliquer_changements(etat, cas, retour.get("changes") or {}, index)
    if faits:
        _changer(" ; ".join(faits))


def _colonne_plan(etat, cas, res, plancher, index):
    with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="center", key=P + "tete_plan"):
        vue = st.segmented_control("Vue", VUES, default=cas.get("vue", "Plan") if cas.get("vue") in VUES else "Plan",
                                   key=P + "vue", label_visibility="collapsed")
        cas["vue"] = vue or "Plan"
        opt = cas.setdefault("options", {})
        for k, lab in (("cotes", "Cotes"), ("aires", "Aires efficaces"), ("taux", "Taux par élément")):
            opt[k] = st.checkbox(lab, value=opt.get(k, True), key=P + "opt_" + k)
    ctx = ET.contexte_dessin(etat, cas, res, plancher, index)
    svg, valeurs, groupes = DS.plan(ctx)
    if cas["vue"] == "Balayage":
        _balayage(etat, cas, res, plancher, index, svg)
        return
    if cas["vue"] == "Coupe":
        st.markdown(DS.coupe(_ctx_coupe(etat, cas, res, plancher, index)), unsafe_allow_html=True)
        st.caption("Coupe transversale sous l'appui sélectionné : éléments voisins, couche de compression, plateau, semelle, charge, aire efficace.")
        return
    from acier.composants.svg_cliquable import svg_cliquable
    with st.container(key=P + "col_dessin"):
        retour = svg_cliquable(svg, valeurs, key=P + "plan", groupes=groupes)
    _traiter(etat, cas, retour, index)
    st.markdown('<div class="chp-leg">■ semelle &nbsp; ▨ plateau &nbsp; ┄ aire efficace &nbsp; ▬ filière &nbsp; '
                '<span style="background:#ffedd5;padding:0 6px">85–100 %</span> <span style="background:#fee2e2;padding:0 6px">&gt; 100 %</span> '
                '&nbsp;·&nbsp; glisser un engin, un plateau, une filière ; clic : réglages ; cote L : clic pour saisir</div>',
                unsafe_allow_html=True)
    with st.expander("Coupe sous l'appui sélectionné", expanded=True):
        st.markdown(DS.coupe(_ctx_coupe(etat, cas, res, plancher, index)), unsafe_allow_html=True)


def _ctx_coupe(etat, cas, res, plancher, index) -> Dict:
    """Contexte de la coupe : l'appui sélectionné, sinon l'appui de l'élément gouvernant, sinon le premier."""
    sel = cas.get("selection") or ""
    ref = sel.split(":", 1)[1] if sel.startswith("rep:") else None
    emps = index["empreintes"]
    if ref not in emps:
        ref = next(iter(emps), None)
    pl = cas["plancher"]
    patches = {p["ref"]: p for p in res.get("patches", [])}
    e = emps.get(ref)
    el = plancher.element_de(e.y) if e else None
    els = plancher.elements()
    i = el.index if el else 0
    voisins = (els[i - 1].nom if i > 0 else "", els[i].nom if els else "", els[i + 1].nom if i + 1 < len(els) else "")
    taux = {x["nom"]: x["taux"] for x in res.get("elements", [])}
    pt = patches.get(ref, {})
    plat = index["plateaux"].get(ref)
    chape = 0.0
    if pl["type"] == "hourdis" and pl["hourdis"].get("chape", {}).get("oui"):
        chape = float(pl["hourdis"]["chape"]["epaisseur"])
    return {"type": "hourdis" if pl["type"] == "hourdis" else "dalle", "h": float(pl[pl["type"]]["h"]), "chape": chape,
            "b_element": els[0].largeur if els else 1.2, "voisins": voisins, "nom_element": voisins[1],
            "plateau": {"A": plat.A, "t": plat.t} if plat else None, "semelle": e.a if e else 0.15,
            "P": (e.P * e.phi) if e else 0.0, "a_eff": pt.get("a") if pt else None, "taux": taux.get(voisins[1], 0.0) or 0.0,
            "joints_predalle": pl["type"] == "predalle"}


def _balayage(etat, cas, res, plancher, index, svg):
    engins = [c["ref"] for c in cas["charges"] if c["type"] == "engin"]
    if not engins:
        st.info("Le balayage déplace un engin le long de la portée : ajoutez d'abord un engin (étape 2).")
        st.markdown(svg, unsafe_allow_html=True)
        return
    ref = st.selectbox("Engin balayé", engins, key=P + "bal_ref")
    _, charges, reps, params, _ = ET.construire(etat, cas)
    if res["statut"] == "refus":
        st.warning(res["message"])
        return
    b = V.balayer(plancher, charges, reps, ref, params)
    x_act = next(c["x"] for c in cas["charges"] if c["ref"] == ref)
    st.markdown(DS.balayage(b["points"], b["pire"], x_act, plancher.L), unsafe_allow_html=True)
    if b["x_admissibles"]:
        xa = b["x_admissibles"]
        st.caption(f"Positions admissibles du centre de l'engin : de {_fmt(min(xa))} à {_fmt(max(xa))} m par pas de 10 cm "
                   f"({len(xa)} positions sur {len(b['points'])}).")
    else:
        st.caption("Aucune position admissible le long de la portée avec les charges et répartiteurs actuels.")
    st.markdown(svg, unsafe_allow_html=True)


# ---------------------------------------------------------------- colonne de droite : résultat et étapes
def _carte_resultat(etat, cas, res, plancher, index):
    stat = res["statut"]
    pill = {"ok": ("ok", "● Conforme"), "attention": ("att", "● Conforme, attention"), "nok": ("nok", "● Non conforme"),
            "refus": ("ref", "● Données manquantes")}[stat]
    if stat == "refus":
        phrase = res["message"]
    elif res["gouverne"]:
        g = res["gouverne"]
        phrase = f"<b>{g['famille'].capitalize()}</b> · {g['objet']} — {g['taux'] * 100:.0f} %"
    else:
        phrase = "Aucune charge posée."
    barres = ""
    for fam, lab in FAMILLES:
        ts = [v["taux"] for v in res.get("verifications", []) if v["famille"] == fam and v["taux"] is not None and v["statut"] in ("ok", "attention", "nok")]
        if not ts:
            continue
        t = max(ts)
        col = "#b91c1c" if t > 1 else ("#c2410c" if t >= 0.85 else "#2563eb")
        barres += (f'<div class="chp-row"><span>{lab}</span><div class="chp-trk"><b style="width:{min(t, 1) * 100:.0f}%;background:{col}"></b></div>'
                   f'<span class="chp-v" style="color:{col}">{t * 100:.0f} %</span></div>')
    nv = [v for v in res.get("verifications", []) if v["statut"] == "non_verifie"]
    note = f'<div class="chp-note">{len(nv)} ligne(s) non vérifiée(s) : paramètre à transcrire ou donnée absente — voir le détail.</div>' if nv else ""
    st.markdown(f'<div class="chp-res"><span class="chp-pill {pill[0]}">{pill[1]}</span><div class="t">RÉSULTAT</div>'
                f'<div class="chp-phrase">{phrase}</div>{barres}{note}</div>', unsafe_allow_html=True)
    # propositions
    if stat == "nok":
        with st.container(horizontal=True, wrap=True, gap="small", key=P + "props"):
            if st.button("💡 Proposer une filière sous les appuis en défaut", key=P + "btn_prop", width="content"):
                _proposer(etat, cas, res, plancher, index)
            if any(c["type"] == "engin" for c in cas["charges"]) and st.button("↔ Balayer la position", key=P + "btn_bal", width="content"):
                cas["vue"] = "Balayage"
                st.session_state[P + "vue"] = "Balayage"
                st.rerun()
    prop = st.session_state.get(P + "proposition")
    if prop and prop.get("cas") == cas["nom"]:
        if prop["trouve"]:
            montes = ", ".join(f"{m['nom']} {m['avant'] * 100:.0f} → {m['apres'] * 100:.0f} %" for m in prop["taux_montes"]) or "aucun"
            st.info(f"Proposition : {prop['texte']} → pire taux du système {prop['taux_max'] * 100:.0f} % "
                    f"({prop['gouverne']['objet']}). Taux montés chez les voisins : {montes}.")
            if st.button("Appliquer la proposition", key=P + "btn_appliquer"):
                for f in prop["filieres"]:
                    cas.setdefault("filieres", []).append(f)
                for r, rp in prop["repartiteurs"].items():
                    cas["repartiteurs"][r] = rp
                st.session_state.pop(P + "proposition", None)
                _changer("Filière(s) appliquée(s)")
        else:
            st.warning(f"Aucune filière du catalogue ne ramène tout le système sous 100 % ({len(prop['essais'])} essais).")


def _proposer(etat, cas, res, plancher, index):
    """Une filière en travers par appui en défaut (flexion ou tranchant de son élément)."""
    _, charges, reps, params, _ = ET.construire(etat, cas)
    defauts = {e["nom"] for e in res["elements"] if e["taux"] > 1.0}
    groupes = []
    for ref, e in index["empreintes"].items():
        el = plancher.element_de(e.y)
        if el and el.nom in defauts and ref not in cas["repartiteurs"] or (el and el.nom in defauts and cas["repartiteurs"].get(ref, {}).get("type") == "plateau"):
            groupes.append([ref])
    if not groupes:
        st.session_state[P + "proposition"] = {"cas": cas["nom"], "trouve": False, "essais": []}
        return
    cands = [(n, 210000e3 * BIB.profils()[n]["Iy"] * 1e-8, BIB.profils()[n]["Wel"]) for n in BIB.candidats_filieres(3.6)]
    d = V.dimensionner_filiere(plancher, charges, reps, groupes, cands,
                               lambda n, l, x, y, refs: BIB.filiere_profil(n, l, x, y, refs=refs), params)
    if not d["trouve"]:
        st.session_state[P + "proposition"] = {"cas": cas["nom"], "trouve": False, "essais": d["essais"]}
        return
    fils, rep = [], {}
    for fil, refs in zip(d["filieres"], groupes):
        fid = ET.ref_libre(cas, "F") + f"p{len(fils)}"
        nom = fil.nom.split(" · ")[0]
        fils.append({"id": fid, "nom": nom, "longueur": round(fil.longueur, 3), "orientation": "y", "x": round(fil.x, 3), "y": round(fil.y, 3)})
        for r in refs:
            rep[r] = {"type": "filiere", "id": fid}
    st.session_state[P + "proposition"] = {"cas": cas["nom"], "trouve": True, "filieres": fils, "repartiteurs": rep,
                                           "taux_max": d["taux_max"], "gouverne": d["gouverne"], "taux_montes": d["taux_montes"],
                                           "texte": f"{fils[0]['nom']} × {_fmt(fils[0]['longueur'])} m sous " + ", ".join(g[0] for g in groupes)}


def _resume_plancher(cas, plancher) -> str:
    pl = cas["plancher"]
    if pl["type"] == "hourdis":
        h = pl["hourdis"]
        ch = f" + couche de compression {h['chape']['epaisseur'] * 1e3:.0f} mm" if h.get("chape", {}).get("oui") else ""
        typ = (h.get("type_cat") or "").split(" · ")[0] or "Hourdis"
        if not typ.lower().startswith("hourdis"):
            typ = "Hourdis " + typ
        return (f"{typ}{ch} · L {_fmt(pl['L'])} m · {h['n_elements']} × {_fmt(h['b_element'])} m · "
                f"{ETATS_H.get(h.get('etat'), '')}")
    d = pl[pl["type"]]
    return f"{TYPES[pl['type']]} · L {_fmt(pl['L'])} m · largeur {_fmt(d['largeur'])} m · h {d['h'] * 1e3:.0f} mm · {d['beton']}"


def _etape_plancher(etat, cas, plancher):
    pl = cas["plancher"]
    st.caption(_resume_plancher(cas, plancher))
    _choix("Type", ("plancher", "type"), cas, list(TYPES), TYPES, segmente=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        _num("Portée L (m)", ("plancher", "L"), cas, 0.1, min_value=0.5)
    with c2:
        _choix("Appuis", ("plancher", "sens"), cas, [1, 2], {1: "A–B gauche / droite", 2: "A–B haut / bas"})
    with c3:
        _num("Charges présentes (kN/m²)", ("plancher", "q_presentes"), cas, 0.5, min_value=0.0, help="Stocks, personnel… (EN 1991-1-6 §4.11), en plus du poids propre")
    if pl["type"] == "hourdis":
        _etape_hourdis(etat, cas)
    else:
        _etape_dalle(cas, pl["type"])


def _etape_hourdis(etat, cas):
    h = cas["plancher"]["hourdis"]
    types = ET.types_hourdis(etat)
    noms = list(types)
    cur = h.get("type_cat") if h.get("type_cat") in noms else noms[0]
    def _type():
        ET.appliquer_type_hourdis(cas, types[st.session_state[P + "w_type_cat"]])
        _reset_widgets()
    st.selectbox("Type (catalogue ou fiche)", noms, index=noms.index(cur), key=P + "w_type_cat", on_change=_type)
    c1, c2, c3 = st.columns(3)
    with c1:
        _num("Éléments", ("plancher", "hourdis", "n_elements"), cas, 1, min_value=1)
        _num("M_Rd (kN·m / él.)", ("plancher", "hourdis", "M_Rd"), cas, 1.0, "%.1f", 0.0)
        _num("I (m⁴ / él.)", ("plancher", "hourdis", "I"), cas, 0.0001, "%.5f", nul_vaut_none=True, help="Inertie du fabricant ; 0 = estimation nommée (0,6·b·h³/12), raideur relative seulement")
    with c2:
        _num("Largeur d'élément (m)", ("plancher", "hourdis", "b_element"), cas, 0.1)
        _num("V_Rd (kN / él.)", ("plancher", "hourdis", "V_Rd"), cas, 1.0, "%.1f", 0.0)
        _num("Membrure p_adm (N/mm²)", ("plancher", "hourdis", "p_loc_adm"), cas, 0.05, nul_vaut_none=True, help="Pression locale admissible du fabricant (P-5) ; 0 = non renseignée")
    with c3:
        _num("Hauteur h (m)", ("plancher", "hourdis", "h"), cas, 0.01)
        _choix("Nature des résistances", ("plancher", "hourdis", "nature"), cas, ["Rd", "service"], {"Rd": "Rd (calcul)", "service": "service"},
               help="P-1 / P-2 : Rd (M_Rd, V_Rd du catalogue) → efforts à l'ELU, γG·G + γQ·Q ; service (charge admissible du catalogue) → coefficients 1")
        _num("g (kN/m²)", ("plancher", "hourdis", "g"), cas, 0.1)
    _choix("État", ("plancher", "hourdis", "etat"), cas, list(ETATS_H), ETATS_H, segmente=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        _case("Couche de compression", ("plancher", "hourdis", "chape", "oui"), cas)
        _num("M_cr (kN·m / él.)", ("plancher", "hourdis", "M_cr"), cas, 1.0, "%.1f", nul_vaut_none=True, help="Moment de fissuration du fabricant (P-11), information")
    with c2:
        _num("Épaisseur chape (m)", ("plancher", "hourdis", "chape", "epaisseur"), cas, 0.01)
        _num("M_Rd avec chape", ("plancher", "hourdis", "M_Rd_chape"), cas, 1.0, "%.1f", nul_vaut_none=True)
    with c3:
        _num("fck(t) chape (MPa)", ("plancher", "hourdis", "chape", "fck_t"), cas, 1.0, "%.1f", help="Résistance à l'âge du calcul (module Age béton)")
        _num("V_Rd avec chape", ("plancher", "hourdis", "V_Rd_chape"), cas, 1.0, "%.1f", nul_vaut_none=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        _case("Chape collaborante", ("plancher", "hourdis", "chape", "collaborante"), cas, help="Surface rugueuse, joints remplis, selon le fabricant (P-17)")
    with c2:
        _num("Classe exigée (MPa)", ("plancher", "hourdis", "chape", "fck_exigee"), cas, 1.0, "%.1f", nul_vaut_none=True, help="Classe que le fabricant exige pour ses valeurs avec chape ; 0 = inconnue → valeurs nues (P-17)")
    with c3:
        _parts_annexe_c(cas)
    _texte_w("Source du catalogue", ("plancher", "hourdis", "source"), cas)
    # P-1 : un catalogue transcrit sert de défaut, le bureau peut toujours saisir ses valeurs — et les
    # garder comme fiche de ce fichier (sélectionnable ci-dessus, enregistrée avec le cas)
    with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=P + "fiche_h"):
        nom_f = st.text_input("Garder ces valeurs comme fiche du catalogue", key=P + "fiche_h_nom",
                              placeholder="ex. ERGON SP 200 · fiche 2026", width=340)
        if st.button("＋ Fiche de hourdis", key=P + "btn_fiche_h", width="content",
                     help="Largeur, hauteur, g, E, I, M_Rd, V_Rd, nature, valeurs avec chape, M_cr, p_adm et source deviennent "
                          "une fiche choisie dans « Type (catalogue ou fiche) » et enregistrée avec le fichier du cas"):
            if not (nom_f or "").strip():
                st.toast("Donner un nom à la fiche")
            else:
                ET.ajouter_fiche_hourdis(etat, cas, nom_f.strip())
                _changer(f"Fiche « {nom_f.strip()} » ajoutée au catalogue de ce fichier")


def _parts_annexe_c(cas):
    h = cas["plancher"]["hourdis"]
    cur = h.get("parts_annexe_C") or {}
    txt0 = ", ".join(f"{k}:{v}" for k, v in cur.items())
    cle = P + "w_parts_C"
    def _maj():
        s = st.session_state[cle].strip()
        parts = {}
        for morceau in s.replace(";", ",").split(","):
            if ":" in morceau:
                k, v = morceau.split(":")
                parts[int(k.strip())] = float(v.strip().replace(",", "."))
        h["parts_annexe_C"] = parts or None
    st.text_input("Parts annexe C (0:part, 1:voisin…)", value=txt0, key=cle, on_change=_maj,
                  help="Figure C.1 de l'EN 1168 transcrite par le bureau ; vide = l'élément chargé porte tout")


def _etape_dalle(cas, typ):
    base = ("plancher", typ)
    c1, c2, c3 = st.columns(3)
    with c1:
        _num("Épaisseur h (m)", base + ("h",), cas, 0.01)
        _num("As inf. (mm²/m)", base + ("As_inf",), cas, 10.0, "%.0f", 0.0)
        _num("Revêtement h1 (m)", base + ("h1",), cas, 0.01, min_value=0.0)
    with c2:
        _num("Hauteur utile d (m)", base + ("d",), cas, 0.01)
        _num("fyk (MPa)", base + ("fyk",), cas, 50.0, "%.0f")
        _num("Largeur (m)", base + ("largeur",), cas, 0.1, min_value=0.5)
    with c3:
        _choix("Béton", base + ("beton",), cas, list(RS.BETON))
        _num("fck(t) (MPa, 0 = classe)", base + ("fck_t",), cas, 1.0, "%.1f", nul_vaut_none=True, help="Résistance à l'âge du calcul : classe arrondie vers le bas")
        _num("γs", base + ("gamma_s",), cas, 0.05)
    _case("Armature transversale ≥ 20 % (largeur efficace)", base + ("armature_transversale_ok",), cas)
    if typ == "predalle":
        c1, c2, c3 = st.columns(3)
        with c1:
            _num("Largeur de prédalle (m)", base + ("b_predalle",), cas, 0.1)
        with c2:
            _case("Joints armés (treillis de chape continu)", base + ("joints_armes",), cas)
            _case("Chape durcie", base + ("chape_durcie",), cas, help="Sans chape durcie, aucune charge d'engin ni d'étai (P-7)")
        with c3:
            _num("Étais en place : entraxe (m, 0 = non)", base + ("portee_etayee",), cas, 0.1, nul_vaut_none=True, help="Portée réduite ; report sur l'étage inférieur non modélisé (P-8)")


def _etape_charges(etat, cas, res, index):
    n = len(cas["charges"])
    tot = sum(sum((c.get("P") or {}).values()) if c["type"] == "engin" else c.get("P", 0.0) for c in cas["charges"])
    st.caption(f"{n} objet(s) · {_fmt(tot, 0)} kN au total")
    with st.container(horizontal=True, wrap=True, gap="small", key=P + "add"):
        if st.button("＋ Ajouter depuis la bibliothèque", key=P + "btn_lib", width="content"):
            _dialogue_bibliotheque(etat, cas)
        if st.button("＋ Étançon", key=P + "btn_et", width="content"):
            ET.ajouter_charge(etat, cas, "ponctuel"); _changer("Étançon posé au centre : glisser pour déplacer")
        if st.button("＋ File d'étais", key=P + "btn_fi", width="content"):
            ET.ajouter_charge(etat, cas, "lineaire"); _changer("Charge linéaire posée au centre")
        if st.button("＋ Stock", key=P + "btn_st", width="content"):
            ET.ajouter_charge(etat, cas, "surfacique"); _changer("Charge surfacique posée au centre")
    for c in cas["charges"]:
        cols = st.columns([5, 2, 1.2, 0.8])
        with cols[0]:
            if c["type"] == "engin":
                desc = f"**{c['ref']}** · {c['gabarit'].split(' · ')[0]} — x {_fmt(c['x'])} · y {_fmt(c['y'])} m · {c.get('angle', 0)}°"
            elif c["type"] == "ponctuel":
                desc = f"**{c['ref']}** · étançon {_fmt(c['P'], 0)} kN, semelle {c['a'] * 1e3:.0f} × {c['b'] * 1e3:.0f} — x {_fmt(c['x'])} · y {_fmt(c['y'])} m"
            elif c["type"] == "lineaire":
                desc = f"**{c['ref']}** · linéaire {_fmt(c['p'], 1)} kN/m sur {_fmt(c['longueur'])} m — x {_fmt(c['x'])} · y {_fmt(c['y'])} m"
            else:
                desc = f"**{c['ref']}** · surfacique {_fmt(c['q'], 1)} kN/m² sur {_fmt(c['lx'])} × {_fmt(c['ly'])} m"
            st.markdown(desc)
        with cols[1]:
            if c["type"] == "engin":
                st.caption(f"{_fmt(sum((c.get('P') or {}).values()), 0)} kN")
        with cols[2]:
            if st.button("Sélect.", key=P + "sel_" + c["ref"], width="content"):
                cas["selection"] = ("engin:" if c["type"] == "engin" else "charge:") + c["ref"]
                _changer()
        with cols[3]:
            if st.button("🗑️", key=P + "del_" + c["ref"], width="content"):
                ET._action_charge(cas, c, "Supprimer")
                _changer(f"{c['ref']} supprimé")
    # charges par appui de l'engin sélectionné
    sel = cas.get("selection") or ""
    ref = sel.split(":", 1)[1] if sel.startswith(("engin:", "rep:")) else None
    if ref:
        ref = ref.split(".")[0]
    eng = next((c for c in cas["charges"] if c["type"] == "engin" and c["ref"] == ref), None)
    if eng:
        st.markdown(f"**Appuis de {eng['ref']}** — charge par appui (kN, tableau de stabilité du constructeur, pour l'opération)")
        gab = ET.gabarits(etat)[eng["gabarit"]]
        cols = st.columns(min(4, max(1, len(gab["appuis"]))))
        for i, ap in enumerate(gab["appuis"]):
            with cols[i % len(cols)]:
                _num(f"{ap['nom']} (kN)", ("charges", cas["charges"].index(eng), "P", ap["nom"]), cas, 1.0, "%.1f", 0.0)
        taux = {e["nom"]: e for e in res.get("elements", [])}
        lignes = []
        for e in index["engins"][eng["ref"]].empreintes():
            el = next((x for x in res.get("elements", []) if x["y0"] <= e.y < x["y1"]), None)
            rp = cas["repartiteurs"].get(e.ref)
            rtxt = "—" if not rp else (rp["nom"] if rp["type"] == "plateau" else f"filière {rp['id']}")
            t = el["taux"] if el else None
            cls = "chp-r" if t and t > 1 else ("chp-o" if t and t >= 0.85 else "chp-g")
            lignes.append(f"<tr><td><b>{e.ref.split('.')[-1]}</b></td><td class='n'>{_fmt(e.P * e.phi, 0)} kN</td><td>{rtxt}</td>"
                          f"<td>{el['nom'] if el else '—'}</td><td class='n {cls}'>{(t * 100):.0f} %</td></tr>" if t is not None else
                          f"<tr><td><b>{e.ref.split('.')[-1]}</b></td><td class='n'>{_fmt(e.P * e.phi, 0)} kN</td><td>{rtxt}</td><td>—</td><td class='n'>—</td></tr>")
        st.markdown("<table class='chp-tab'><tr><th>Appui</th><th class='n'>Charge</th><th>Répartiteur</th><th>Élément</th><th class='n'>Taux</th></tr>"
                    + "".join(lignes) + "</table>", unsafe_allow_html=True)


@st.dialog("Bibliothèque — ajouter une charge", width="large")
def _dialogue_bibliotheque(etat, cas):
    onglets = st.tabs(["Engins", "Nouvelle fiche d'engin"])
    with onglets[0]:
        for nom, g in ET.gabarits(etat).items():
            c1, c2, c3 = st.columns([1.2, 4, 1.2])
            with c1:
                st.markdown(SY.pictogramme(g), unsafe_allow_html=True)
            with c2:
                st.markdown(f"**{nom}**")
                st.caption((g.get("source", "") or "")[:220])
            with c3:
                if st.button("Poser", key=P + "pose_" + nom):
                    ET.ajouter_engin(etat, cas, nom)
                    _changer(f"{nom.split(' · ')[0]} posé au centre : glisser pour déplacer, clic pour les charges par appui")
    with onglets[1]:
        with st.form(P + "fiche_engin"):
            nom = st.text_input("Nom (modèle, constructeur)")
            typ = st.selectbox("Type d'appuis", ["pieds", "roues", "chenilles"], format_func=lambda t: {"pieds": "stabilisateurs", "roues": "roues", "chenilles": "chenilles + stabilisateurs"}[t])
            c1, c2, c3 = st.columns(3)
            with c1:
                lng = st.number_input("Longueur châssis (m)", value=5.0, step=0.1)
            with c2:
                lrg = st.number_input("Largeur châssis (m)", value=2.4, step=0.1)
            with c3:
                fl = st.number_input("Flèche en projection (m)", value=3.0, step=0.1)
            st.markdown("Appuis : position depuis le centre (x vers l'avant, y vers la droite), empreinte, charge maximale, φ")
            import pandas as pd
            df = st.data_editor(pd.DataFrame([
                {"nom": "P1", "dx": 2.3, "dy": -2.3, "a": 0.30, "b": 0.30, "P": 0.0, "phi": 1.0},
                {"nom": "P2", "dx": 2.3, "dy": 2.3, "a": 0.30, "b": 0.30, "P": 0.0, "phi": 1.0},
                {"nom": "P3", "dx": -2.3, "dy": -2.3, "a": 0.30, "b": 0.30, "P": 0.0, "phi": 1.0},
                {"nom": "P4", "dx": -2.3, "dy": 2.3, "a": 0.30, "b": 0.30, "P": 0.0, "phi": 1.0}]), num_rows="dynamic", key=P + "fiche_appuis")
            src = st.text_input("Source (document, date, page)")
            if st.form_submit_button("Enregistrer la fiche"):
                if not nom.strip():
                    st.error("Un nom est nécessaire.")
                else:
                    etat["fiches"]["engins"].append({
                        "nom": nom.strip(), "type_appuis": typ, "generique": False, "source": src,
                        "silhouette": {"longueur": float(lng), "largeur": float(lrg), "fleche": float(fl)},
                        "appuis": [{"nom": str(r["nom"]), "dx": float(r["dx"]), "dy": float(r["dy"]), "a": float(r["a"]), "b": float(r["b"]),
                                    "P": float(r["P"]), "phi": float(r["phi"])} for _, r in df.iterrows()]})
                    _changer(f"Fiche « {nom.strip()} » ajoutée (enregistrée avec le fichier du cas)")


def _etape_repartiteur(etat, cas, index):
    refs = list(index["empreintes"].keys())
    sel = cas.get("selection") or ""
    ref_sel = sel.split(":", 1)[1] if sel.startswith("rep:") else None
    rp = cas["repartiteurs"].get(ref_sel) if ref_sel else None
    titre = (f"Sous {ref_sel} : " + ("aucun" if not rp else (rp["nom"] if rp["type"] == "plateau" else f"filière {rp['id']}"))
             if ref_sel else "Choisir un appui (clic sur une semelle du plan, ou ci-dessous)")
    st.caption(titre)
    if not refs:
        st.caption("Posez d'abord un engin ou un étançon (étape 2).")
    else:
        if ref_sel not in refs:
            ref_sel = refs[0]
        # Ces trois widgets MIROIRS suivent l'état (sélection faite sur le plan, plateau posé par la
        # fenêtre d'une semelle) : leur valeur est écrite depuis l'état AVANT leur création, et le
        # changement de l'utilisateur passe par on_change. Un widget laissé à sa valeur mémorisée
        # réécrivait l'état à l'envers (plateau posé sur le plan → remis à « aucun » au rendu
        # suivant) — défaut trouvé par le harnais le 02/11/2026.
        def _sel_rep():
            cas["selection"] = "rep:" + st.session_state[P + "sel_rep"]
            _reset_widgets()
        st.session_state[P + "sel_rep"] = ref_sel
        st.selectbox("Appui", refs, key=P + "sel_rep", on_change=_sel_rep)
        rp = cas["repartiteurs"].get(ref_sel)
        typ = rp["type"] if rp else "aucun"
        cle_t = P + "rep_type_" + ref_sel
        def _rep_type():
            v = st.session_state.get(cle_t)
            if v and v != typ:
                faits = ET.regler_repartiteur(etat, cas, ref_sel, "type", v)
                _reset_widgets()
                if faits:
                    st.toast(" ; ".join(faits))
        st.session_state[cle_t] = typ
        st.segmented_control("Type", ["aucun", "plateau", "filiere"], key=cle_t, on_change=_rep_type,
                             format_func=lambda t: {"aucun": "Aucun", "plateau": "Plateau", "filiere": "Filière"}[t],
                             label_visibility="collapsed")
        dispo = ET.repartiteurs_disponibles(etat)
        if typ == "plateau":
            noms = [n for n, d in dispo.items() if d.get("type") == "plateau"]
            cur = rp["nom"] if rp["nom"] in noms else noms[0]
            def _el():
                ET.regler_repartiteur(etat, cas, ref_sel, "element", st.session_state[P + "rep_el"])
                _reset_widgets()
            st.session_state[P + "rep_el"] = cur
            st.selectbox("Plateau", noms, key=P + "rep_el", on_change=_el)
            d = dispo[cur]
            st.caption(f"{d['A']:.0f} × {d['Bp']:.0f} × {d['t']:.0f} mm · f = {d.get('f') or '—'} MPa · {d.get('source', '')[:120]}")
            c1, c2, c3 = st.columns(3)
            with c1:
                _num("Décalage x (m)", ("repartiteurs", ref_sel, "dx"), cas, 0.05)
            with c2:
                _num("Décalage y (m)", ("repartiteurs", ref_sel, "dy"), cas, 0.05)
            with c3:
                if st.button("Idem tous les appuis de l'engin", key=P + "rep_idem"):
                    ET.regler_repartiteur(etat, cas, ref_sel, "idem", "oui")
                    _changer("Même plateau sous tous les appuis de l'engin")
        elif typ == "filiere":
            f = next((f for f in cas.get("filieres", []) if f["id"] == rp["id"]), None)
            if f:
                noms = [n for n, d in dispo.items() if d.get("type") == "filiere"]
                i = cas["filieres"].index(f)
                cur = f["nom"] if f["nom"] in noms else noms[0]
                _choix("Élément", ("filieres", i, "nom"), cas, noms)
                c1, c2, c3 = st.columns(3)
                with c1:
                    _num("Longueur (m)", ("filieres", i, "longueur"), cas, 0.1, min_value=0.1)
                with c2:
                    _choix("Orientation", ("filieres", i, "orientation"), cas, ["y", "x"], {"y": "en travers des éléments", "x": "dans le sens porteur"})
                with c3:
                    autres = [r for r in refs if r != ref_sel]
                    lies = [r for r, x in cas["repartiteurs"].items() if x.get("id") == f["id"] and r != ref_sel]
                    choix = st.multiselect("Autres appuis portés", autres, default=[r for r in lies if r in autres], key=P + "fil_refs")
                    if set(choix) != set(lies):
                        for r in autres:
                            if r in choix:
                                cas["repartiteurs"][r] = {"type": "filiere", "id": f["id"]}
                            elif r in lies:
                                cas["repartiteurs"].pop(r, None)
                        _changer("Appuis de la filière mis à jour")
                d = dispo.get(cur, {})
                st.caption(f"{d.get('origine', '')} · {d.get('source', '')[:140]}")
    _nouvel_element(etat)


def _nouvel_element(etat):
    with st.expander("Nouvel élément de répartition (plateau, profilé, madrier sur champ…)", expanded=False):
        with st.form(P + "fiche_rep"):
            typ = st.selectbox("Type", ["plateau", "filiere"], format_func=lambda t: {"plateau": "Plateau (posé sous un appui)", "filiere": "Filière (poutre sur plusieurs éléments)"}[t])
            nom = st.text_input("Nom", placeholder="ex. Plateau azobé 800 × 800 × 80, Madrier C24 75 × 225 sur champ")
            c1, c2, c3 = st.columns(3)
            with c1:
                mat = st.selectbox("Matériau", ["S235", "S355", "Bois C24", "Azobé", "OSB", "Autre"])
                f = st.number_input("f (MPa) : fy/γM0 ou f_m,d", value=0.0, step=1.0, help="0 = à renseigner : l'élément ne peut pas être calculé sans sa résistance (source obligatoire)")
            with c2:
                E = st.number_input("E (MPa)", value=210000.0, step=1000.0, help="Filière : raideur EI ; plateau : sans effet")
                masse = st.number_input("Masse (kg ou kg/m)", value=0.0, step=1.0)
            with c3:
                src = st.text_input("Source (fiche, norme, date)")
            st.markdown("Dimensions (mm)")
            c1, c2, c3 = st.columns(3)
            with c1:
                A = st.number_input("Plateau A · ou filière : b (largeur)", value=600.0, step=10.0)
            with c2:
                B = st.number_input("Plateau B · ou filière : h (hauteur sur champ)", value=600.0, step=10.0)
            with c3:
                t = st.number_input("Plateau : épaisseur t", value=20.0, step=1.0)
            prof = st.selectbox("Filière : profilé du catalogue (sinon rectangle b × h)", ["— rectangle b × h —"] + list(BIB.profils()))
            if st.form_submit_button("Enregistrer l'élément"):
                if not nom.strip():
                    st.error("Un nom est nécessaire.")
                else:
                    fy = {"S235": 235.0, "S355": 355.0}.get(mat)
                    fiche = {"nom": nom.strip(), "type": typ, "materiau": mat, "f": (f or fy or None), "E": float(E), "masse": float(masse), "source": src}
                    if typ == "plateau":
                        fiche.update({"t": float(t), "A": float(A), "Bp": float(B)})
                    else:
                        fiche["section"] = {"profil": prof} if prof != "— rectangle b × h —" else {"b": float(A), "h": float(B)}
                    etat["fiches"]["repartiteurs"].append(fiche)
                    _changer(f"Élément « {nom.strip()} » ajouté à la bibliothèque du cas")


def _etape_hypotheses(cas, params, res):
    st.caption(f"ELU quand la résistance est une valeur de calcul : γG {params.gamma_G:g} sur les poids propres, "
               f"γQ {params.gamma_Q:g} sur les charges de chantier (coefficients 1 avec une valeur admissible de service) · "
               f"tranchant : {params.part_tranchant_element:.0%} de la réaction d'une charge concentrée sur l'élément chargé · "
               + ("k Heft 240 transcrit" if params.k_flexion_heft240 is not None else "b_m = t_y (k à transcrire)"))
    c1, c2, c3 = st.columns(3)
    with c1:
        _num("γG sur les poids propres (P-2)", ("params", "gamma_G"), cas, 0.05, help="Hourdis, chape, dalle : 1,35 à l'ELU")
        _num("γQ sur les charges de chantier (P-2)", ("params", "gamma_Q"), cas, 0.05,
             help="Charges présentes, engins (× φ), étais, stocks : 1,5 à l'ELU ; aussi sur la charge des effets locaux, de la filière et du plateau")
    with c2:
        _num("Part du tranchant d'une charge concentrée sur l'élément chargé (P-3)", ("params", "part_tranchant_element"), cas, 0.05,
             help="1 = la réaction d'appui reste entière sur l'élément qui porte la charge, sans partage par les joints (prudent) ; "
                  "l'annexe C de l'EN 1168 répartit la flexion, pas le tranchant près de l'appui")
        _num("k Heft 240 flexion (0 = non transcrit, P-4)", ("params", "k_flexion_heft240"), cas, 0.1, nul_vaut_none=True)
    with c3:
        _num("Joint : résistance (kN/m, 0 = non vérifié, P-15)", ("params", "joint_adm_kN_m"), cas, 1.0, nul_vaut_none=True)
        _num("Joint : longueur mobilisée (m)", ("params", "joint_longueur_m"), cas, 0.1, nul_vaut_none=True)
        _num("αcc (ANB à confirmer)", ("params", "alpha_cc"), cas, 0.05)
    for h in res.get("hypotheses", []):
        st.caption("• " + h)


def _detail(res):
    with st.expander("Détail des vérifications par élément", expanded=False):
        if res["statut"] == "refus":
            st.warning(res["message"])
            return
        if res.get("elements"):
            lignes = "".join(
                f"<tr><td><b>{e['nom']}</b></td><td class='n'>{_fmt(e['M_max'], 1)}</td><td class='n'>{_fmt(e['M_adm'], 1)}</td>"
                f"<td class='n {_cls(e['taux_M'])}'>{e['taux_M'] * 100:.0f} %</td><td class='n'>{_fmt(e['V_max'], 1)}</td><td class='n'>{_fmt(e['V_adm'], 1)}</td>"
                f"<td class='n {_cls(e['taux_V'])}'>{e['taux_V'] * 100:.0f} %</td><td>{e['gouverne']}</td></tr>" for e in res["elements"])
            st.markdown("<table class='chp-tab'><tr><th>Élément</th><th class='n'>M</th><th class='n'>M_adm</th><th class='n'>η_M</th>"
                        "<th class='n'>V</th><th class='n'>V_adm</th><th class='n'>η_V</th><th>gouverne</th></tr>" + lignes + "</table>",
                        unsafe_allow_html=True)
        lignes = ""
        for v in res.get("verifications", []):
            if v["famille"] in ("flexion", "tranchant") and res.get("elements"):
                continue
            t = v["taux"]
            cls = {"info": "chp-i", "non_verifie": "chp-o", "nok": "chp-r", "attention": "chp-o", "ok": "chp-g"}.get(v["statut"], "")
            ttxt = f"{t * 100:.0f} %" if t is not None else {"info": "info", "non_verifie": "non vérifié"}.get(v["statut"], "—")
            lignes += (f"<tr><td>{v['famille']}</td><td>{v['objet']}</td><td class='n'>{_val(v['E'])}</td><td class='n'>{_val(v['R'])}</td>"
                       f"<td class='n {cls}'>{ttxt}</td><td class='chp-i'>{v['methode']}</td></tr>")
        st.markdown("<table class='chp-tab'><tr><th>Famille</th><th>Objet</th><th class='n'>E</th><th class='n'>R</th><th class='n'>η</th><th>Méthode</th></tr>"
                    + lignes + "</table>", unsafe_allow_html=True)


def _cls(t):
    return "chp-r" if t > 1 else ("chp-o" if t >= 0.85 else "chp-g")


def _val(x):
    if x is None:
        return "—"
    if isinstance(x, (int, float)):
        return _fmt(x, 2 if abs(x) < 10 else 1)
    return str(x)


# ---------------------------------------------------------------- page
def show():
    etat = _etat()
    cas = ET.cas_actif(etat)
    _entete(etat, cas)
    try:
        plancher, charges, reps, params, index = ET.construire(etat, cas)
    except (ValueError, KeyError) as e:
        st.error(f"Donnée manquante : {e}")
        return
    res = V.verifier(plancher, charges, reps, params)
    col_g, col_d = st.columns([1.15, 1], gap="large")
    with col_g:
        _colonne_plan(etat, cas, res, plancher, index)
    with col_d:
        _carte_resultat(etat, cas, res, plancher, index)
        # Quatre étapes en ONGLETS à libellé fixe : un expander dont le titre résume son contenu se
        # referme à chaque saisie qui change ce titre (Streamlit le remonte), ce qui coupait la saisie
        # des charges par appui (02/11/2026). Le résumé vit en tête de chaque onglet.
        onglets = st.tabs(ETAPES)
        with onglets[0]:
            _etape_plancher(etat, cas, plancher)
        with onglets[1]:
            _etape_charges(etat, cas, res, index)
        with onglets[2]:
            _etape_repartiteur(etat, cas, index)
        with onglets[3]:
            _etape_hypotheses(cas, params, res)
    _detail(res)

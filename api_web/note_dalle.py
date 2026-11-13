# -*- coding: utf-8 -*-
"""La NOTE DE CALCUL PDF du module « Dalles béton armé » sur le site — les deux types, Dalle CSP et Prédalle + dalle CSP
(décision de l'ingénieur du 09/11/2026, S4 : la note se fait SUR LE SITE et dit les chiffres du site).

Elle n'est pas réécrite : c'est le générateur VALIDÉ de l'écran Streamlit (``modules/export_pdf_dalle.py`` v2, mise en
page ``ndc_pdf``) qui la produit, dans le navigateur (Pyodide), à partir des valeurs que l'écran Streamlit aurait en
mémoire pour la même dalle (les clés « dal<i>_… », « dal<i>_sec<j>_… » de ``st.session_state``). Ce fichier TRADUIT
l'état du site (``api_web/dalle.py``) en ces valeurs (``valeurs``), appelle ``export_pdf_dalle._collecter_resultats`` —
qui refait le calcul avec ``_compute_section``, le calcul du PDF Streamlit — puis complète chaque planche de ce que le
site sait en plus.

LES AXES. Le site pose les nappes l'une sur l'autre (api_web/dalle_positions.py : la direction secondaire, base et
renforts, sur la principale ; la règle de pose de la prédalle) ; le générateur, lui, poserait X et Y au même niveau
(``_auto_dist_couche``). Chaque couche reçoit donc son axe EFFECTIF du site (``POS.axes``) comme une distance SAISIE
(« dist_auto » faux, « dist_axe » = l'axe, écrit sans arrondi) — ce que l'écran Prédalle de Streamlit faisait déjà pour
sa note (``modules/predalle._valeurs_pour_export``, qui figeait l'axe à 0,01 cm près). Le centre de gravité imposé passe
par « ycdg » (« ycdg_auto » vrai sinon). Le calcul de la planche est alors celui de l'écran : mêmes formules, mêmes
axes (``tests/test_note_dalle.py`` le prouve cas par cas, et ``ecarts`` le revérifie à chaque note : un écart lève,
la note ne sort pas).

CE QUE LE SITE ÉCRIT LUI-MÊME (le générateur donne la coupe et les chiffres ; la planche est celle du site) :
- le CONTENU de la planche (lot DAL6, demande du propriétaire du 13/11/2026, point 8 — ``api_web/note_dalle_planche``) :
  l'en-tête compact sous le titre (la dalle, le béton, l'acier, « Non vérifié(e)s : … » en une phrase), les
  sollicitations en tableau, la hauteur minimale DÉTAILLÉE, les sections minimale et maximale communes calculées une
  fois, les nappes avec leur armature posée en résultat, l'effort tranchant — composés des grandeurs que le moteur du
  site nomme. Le bloc « Observations » du panneau de gauche n'existe plus : les avertissements de l'écran (rares)
  s'écrivent en fin de planche, seulement s'il y en a ;
- l'ÉTAT : « Données incomplètes » pour une section sans aucun effort (jamais « Vérifié »), « Non vérifié » / « Vérifié »
  sinon — l'état de l'écran (api_web/plaque_compact.etat_bloc) ;
- les NIVEAUX DESSINÉS de la coupe : ceux du calcul (le générateur écartait deux axes trop proches pour le dessin —
  ``_coupe_dalle_depuis_R._niveaux`` — et dessinait un renfort, posé dans le plan de sa base, plus haut qu'il n'est
  compté) ; chaque couche est dessinée à l'axe qui entre dans d, comme sur la coupe de l'écran ; les titres des
  schémas sont ceux de l'écran (« Direction Y · principale », « Direction X ») et, dans le schéma de la direction
  secondaire, les fils d'un treillis unique ne répètent pas sa désignation (« fils du treillis », DAL5a).

Plusieurs dalles : une note, une planche par section, dans l'ordre des dalles. Chaque dalle est traduite à part (γs et
le jeu sont ceux de la note sur le site, mais une note relue pourrait en porter de différents), puis les planches sont
réunies dans un seul document.
"""
import base64
import copy
import html
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from beton.dalle import entree as EN
from beton.dalle import moteur as MO

from . import dalle_positions as POS

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ETAT_INCOMPLET = "Données incomplètes"
PARTIE = "Dalles en béton armé"
# la tolérance de ``ecarts`` (relative) : les deux calculs font les mêmes opérations sur les mêmes flottants
TOLERANCE = 1e-9
# la plus petite coupe à deux schémas dont les textes ne se chevauchent pas — MESURÉ le 09/11/2026 (en réduisant la coupe
# de 5 en 5 pt jusqu'au premier chevauchement) : 180 à 200 pt pour h = 10 à 15 cm, 225 (20 cm), 240 (30 cm), 245 (40 cm),
# 260 (50 cm), 280 (60 cm), CSP comme prédalle ; COUPE_MIN = (a, b) par nombre de schémas : a + b·h (h en cm), au-dessus
# de chaque mesure d'au moins 10 pt. Depuis DAL6 (13/11/2026) le panneau de gauche ne porte plus que la coupe et les
# sollicitations : la coupe garde toute sa hauteur (``Style.coupe_h``), au-dessus de COUPE_MIN pour h ≤ 60 cm —
# tests/test_note_dalle.py le revérifie (h de 12 à 60 cm, aucun texte de la coupe n'en chevauche un autre).
COUPE_MIN = {1: (110.0, 0.75), 2: (205.0, 1.5)}
# la hauteur demandée pour la coupe (pt) : la place que l'en-tête compact et le tableau des sollicitations laissent dans le
# panneau de gauche — le kit la borne à ce qui reste au-dessus de la réserve du panneau (``styles.planche``)
COUPE_H = 400.0


def _beton() -> Dict:
    """Les classes de béton : le fichier que lisent le moteur du site et l'écran Streamlit (``beton_classes.json``)."""
    with open(os.path.join(RACINE, "beton_classes.json"), encoding="utf-8") as f:
        return json.load(f)


def _axe(x: float) -> str:
    """Un axe écrit pour le champ « Dist. axe » de l'écran, SANS arrondi (le générateur le relit par float) : la note
    calcule sur l'axe même du site, au bit près."""
    return repr(float(x)).replace(".", ",")


# ------------------------------------------------------------------ l'état du site → les valeurs de l'écran Streamlit
def valeurs(d: Dict, did: int = 1) -> Tuple[List[Dict], Dict]:
    """(dalles, values) de l'écran Streamlit pour UNE dalle du site (normalisée, avec ses clés « type » et « verrou ») —
    ``did`` : son numéro dans la note. Ce que ``generer_rapport_pdf`` / ``_collecter_resultats`` attendent : la liste
    ``[{"id", "nom", "sections": [{"id", "nom"}]}]`` et les valeurs de ``st.session_state``.

    Chaque couche reçoit son axe EFFECTIF du site (``POS.axes`` : l'axe saisi, sinon l'automatique du site, nappes
    posées l'une sur l'autre) comme une distance saisie ; un centre de gravité saisi passe par « ycdg ». Une prédalle
    passe sa peau (« h_pre » : la coupe et les DIMENSIONS de la planche la disent)."""
    def kd(base):
        return f"dal{did}_{base}"
    v = {"gamma_s": d["gamma_s"], "jeu_enrobage_cm": d["jeu_premier_lit"],
         f"meta_dalle_nom_{did}": d["nom"], kd("beton"): d["beton"], kd("fyk"): d["fyk"], kd("b"): d["b"],
         kd("h"): d["h"], kd("enrobage_beton"): d["enrobage"], kd("dir_principale"): d["dir_principale"]}
    if POS.type_dalle(d) == "predalle":
        v[kd("h_pre")] = d["h_pre"]
    sections = []
    for j, s in enumerate(d["sections"]):
        sid = j + 1

        def ks(base):
            return f"dal{did}_sec{sid}_{base}"
        sections.append({"id": sid, "nom": s["nom"]})
        v[f"meta_dal{did}_nom_{sid}"] = s["nom"]
        for k in EN.MOMENTS + ("V",):
            v[ks(k)] = s[k]
        axes = POS.axes(d, s)
        for f in EN.FACES:
            face = s["faces"][f]
            v[ks(f"ncouches_{f}")] = len(face["couches"])
            for i, (c, e) in enumerate(zip(face["couches"], axes[f]), start=1):
                v[ks(f"arm_type_{f}_c{i}")] = EN.LIBELLES_TYPES[c["type"]]
                v[ks(f"treillis_{f}_c{i}")] = c["treillis"]
                v[ks(f"ø_barres_{f}_c{i}")] = c["d"]
                v[ks(f"esp_barres_{f}_c{i}")] = c["esp"]
                v[ks(f"n_barres_{f}_c{i}")] = c["n"]
                v[ks(f"dist_auto_{f}_c{i}")] = False
                v[ks(f"dist_axe_{f}_c{i}")] = _axe(e)
            v[ks(f"ycdg_auto_{f}")] = face["yG"] is None
            v[ks(f"ycdg_{f}")] = "" if face["yG"] is None else _axe(face["yG"])
    return [{"id": did, "nom": d["nom"], "sections": sections}], v


# ------------------------------------------------------------------ ce que l'écran dit, en texte (pour la planche)
def _texte(fragment: str) -> str:
    """Le texte d'un fragment HTML de l'écran (un message, une aide), sans balises ni entités : la planche l'écrit tel
    que l'écran le montre."""
    t = re.sub(r"<[^>]+>", "", fragment)
    return " ".join(html.unescape(t).split())


def _messages(fragment: str) -> List[str]:
    """Les messages d'un fragment de l'écran (``C.message`` : un ``<div class="msg…">`` chacun), en texte."""
    return [_texte(m) for m in re.findall(r'<div class="msg[^"]*">(.*?)</div>', fragment, flags=re.S)]


def avertissements(d: Dict, r: Dict, i: int) -> List[str]:
    """Les AVERTISSEMENTS que l'écran affiche sous la section ``i`` (ceux du type : api_web/dalle._avertissements_de) ;
    sur la première planche de la dalle, ceux de la dalle entière (sans section) — les textes de l'écran, sans balisage.
    Ce qui n'est pas vérifié n'en est plus (DAL6) : une phrase dans l'en-tête de chaque planche
    (``note_dalle_planche.entete``, ``api_web/dalle.non_verifies``)."""
    from . import dalle as DL
    out = _messages(DL._avertissements_de(d, r, i))
    if i == 0:                           # ce qui vaut pour toute la dalle : une fois, sur sa première planche
        out += _messages(DL._avertissements_de(d, r))
    return out


def etat_label(rs: Dict) -> str:
    """L'état de la planche, celui de l'en-tête du bloc de l'écran (api_web/plaque_compact.etat_bloc) dans les mots de la
    note : « Non vérifié » d'abord (une non-conformité sans effort n'est jamais cachée), « Données incomplètes » sans
    aucun effort, « Vérifié » sinon."""
    from ndc_pdf import data as ND
    if rs["statut"] == MO.NOK:
        return ND.ETAT_LABELS["nok"]
    if not rs["sollicitee"]:
        return ETAT_INCOMPLET
    return ND.ETAT_LABELS["ok"]


# ------------------------------------------------------------------ le contrôle : la planche dit-elle l'écran ?
def _proche(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) <= TOLERANCE * max(1.0, abs(float(a)), abs(float(b)))


def ecarts(R: Dict, rs: Dict) -> List[Tuple]:
    """Les grandeurs où la planche (``R`` : ``export_pdf_dalle._compute_section``) diffère de l'écran (``rs`` : la
    section du moteur du site) — vide quand la note dit l'écran : hauteur (h_u,min, h_min, état), chaque nappe (axe de
    chaque couche, y_G, d, A_s,req, A_s,min, A_s posé, état), τ et τ_adm (et leur état), l'état de la section."""
    g = rs["grandeurs"]
    out = []

    def cmp(nom, a, b):
        if not _proche(a, b):
            out.append((nom, a, b))
    cmp("hu_min", R["hmin"], g["hu_min"])
    cmp("h_min", R["h_min_dalle"], g["h_min"])
    if R["etat_h"] != g["etat_hauteur"]:
        out.append(("etat_hauteur", R["etat_h"], g["etat_hauteur"]))
    cmp("As_max", R["As_max"], g["As_max"])
    for f in EN.FACES:
        cote, dk = f.split("_")
        D = R["dirs"][dk]
        geo = D[f"geo_{cote}"]
        if len(geo["couches"]) != len(rs["faces"][f]["couches"]):
            out.append((f"{f} couches", len(geo["couches"]), len(rs["faces"][f]["couches"])))
            continue
        for j, (c, c_s) in enumerate(zip(geo["couches"], rs["faces"][f]["couches"])):
            cmp(f"{f} axe {j}", c["e"], c_s["e"])
            cmp(f"{f} As/m {j}", c["As_pm"], c_s["As_pm"])
        cmp(f"yG_{f}", geo["e_cdg"], g[f"yG_{f}"])
        cmp(f"d_{f}", D["di" if cote == "inf" else "ds"], g[f"d_{f}"])
        cmp(f"M_{f}", D[f"M_{cote}"], g[f"M_{f}"])
        cmp(f"As_req_{f}", D[f"As_req_{cote}"], g[f"As_req_{f}"])
        cmp(f"As_min_{f}", D[f"As_min_{cote}"], g[f"As_min_{f}"])
        cmp(f"As_{f}", D[f"As_{cote}"], g[f"As_{f}"])
        if D[f"etat_{cote}"] != g[f"etat_{f}"]:
            out.append((f"etat_{f}", D[f"etat_{cote}"], g[f"etat_{f}"]))
    Sh = R["shear"]
    if g["V"] > 0:
        if Sh is None:
            out.append(("tau", None, g["tau"]))
        else:
            cmp("tau", Sh["tau"], g["tau"])
            cmp("tau_adm", Sh["tau_adm"], g["tau_adm"])
            if Sh["etat_tau"] != g["etat_tau"]:
                out.append(("etat_tau", Sh["etat_tau"], g["etat_tau"]))
    elif Sh is not None:
        out.append(("tau sans V", Sh["tau"], 0.0))
    if R["etat_global"] != rs["statut"]:
        out.append(("statut", R["etat_global"], rs["statut"]))
    return out


# ------------------------------------------------------------------ les planches
def resultats(etat: Dict) -> List[Dict]:
    """Les planches de la note (les payloads de ``ndc_pdf.data.construire_sections_dalle``) : celles du PDF Streamlit,
    une par section de chaque dalle, complétées de ce que l'écran dit (« site » : la section calculée par le moteur du
    site, l'état, les avertissements ; « etat_dalle », « indice » : la dalle du site et le rang de sa section). Lève
    ValueError si une planche ne dit pas les chiffres de l'écran (``ecarts``) : la note ne sort pas."""
    from modules import export_pdf_dalle as EPD
    from . import dalle as DL
    beton = _beton()
    out = []
    for di, d in enumerate(etat["dalles"]):
        r = DL._calculer(d)
        dalles, vals = valeurs(d, di + 1)
        planches = EPD._collecter_resultats(dalles, vals, beton)
        if len(planches) != len(r["sections"]):
            raise ValueError(f"dalle « {d['nom']} » : {len(planches)} planche(s) pour {len(r['sections'])} section(s)")
        for i, (pl, rs) in enumerate(zip(planches, r["sections"])):
            e = ecarts(pl["R"], rs)
            if e:
                nom, a, b = e[0]
                raise ValueError(f"La note PDF ne dirait pas les chiffres de l'écran pour « {d['nom']} · {rs['titre']} » "
                                 f"({nom} : {a!r} dans la note, {b!r} à l'écran) : elle n'est pas produite.")
            pl["site"] = rs
            pl["etat_dalle"], pl["indice"] = d, i
            pl["etat_label"] = etat_label(rs)
            pl["avertissements"] = avertissements(d, r, i)
            out.append(pl)
    return out


def _niveaux_du_calcul(coupe: Dict, R: Dict) -> None:
    """Les niveaux dessinés de la coupe = les axes du calcul (mm, depuis la face), couche par couche — dans l'ordre des
    schémas du générateur (la direction principale, puis la secondaire ; une seule quand elles sont identiques)."""
    p = R["principale"]
    s = MO.direction_secondaire(p)
    montrees = [p, s][:len(coupe["schemas"])]
    for sch, dk in zip(coupe["schemas"], montrees):
        o = MO.direction_secondaire(dk)
        for cle, direction in (("points", dk), ("filants", o)):
            for face in ("inf", "sup"):
                geo = R["dirs"][direction][f"geo_{face}"]["couches"]
                for c_dessin, c in zip(sch[f"{cle}_{face}"], geo):
                    c_dessin["e"] = float(c["e"]) * 10.0


class Panneau(list):
    """Le panneau de gauche d'une planche, sous la coupe : les SOLLICITATIONS de la section en tableau
    (``note_dalle_planche.sollicitations``). Pour le kit, une liste d'un bloc de quatre lignes (sa réserve de place sous
    la coupe : ``styles.planche``) ; ``_style`` le dessine (``_ecrire_sollicitations``). DIMENSIONS et MATÉRIAUX n'y sont
    plus : l'en-tête compact de la planche les dit une fois (DAL6)."""
    def __init__(self, sollicitations: Dict):
        super().__init__([(sollicitations["titre"], [None] * 4)])
        self.sollicitations = sollicitations


def _ecrire_sollicitations(d, S, x: float, y: float, w: float, so: Dict) -> float:
    """Le tableau des sollicitations sous la coupe, à partir de ``y`` : un titre comme les blocs du kit, puis une ligne par
    direction (M⁺ · inf. tendue, M⁻ · sup. tendue) et l'effort tranchant — ``kit.table``, textes riches. Rend le bas."""
    from ndc_pdf.kit import Frame, mix, table
    fr = Frame(x, y, w, 0)
    fr.down(9)
    d.t(fr.x, fr.y, so["titre"].upper(), S.f_v, 6.3, S.acc, track=1.15)
    fr.down(3)
    d.line(fr.x, fr.y, fr.x1, fr.y, mix(S.rule, S.ink, 0.12), 0.5)
    fr.down(11)
    largeurs = (0.30 * w, 0.35 * w, 0.35 * w)
    cols = [(ent, lg, "left" if k == 0 else "right") for k, (ent, lg) in enumerate(zip(so["entetes"], largeurs))]
    lignes = [[("lab", dk), ("num", f"{m_inf} kNm"), ("num", f"{m_sup} kNm")] for dk, m_inf, m_sup in so["lignes"]]
    lignes.append([("lab", "Effort tranchant $V$"), ("num", f"{so['V']} kN"), ("lab", "")])
    return table(d, fr, cols, lignes, S, 7.4, 11.0, s_ent=6.0, ms=S._ms_col)


def _style(n_cols: int):
    """Le style de la note : celui du générateur (``export_pdf._style_ndc`` : la palette, le repli en trois colonnes),
    dont le panneau de gauche, sous la coupe, dit les sollicitations en tableau (``Panneau``). Hériter du style est
    l'extension du kit : le kit ne change que par ses extensions rétro-compatibles (l'en-tête compact, ``Style.entete``)."""
    from modules import export_pdf as EP
    base = EP._style_ndc(n_cols)

    class StyleNote(type(base)):
        def _data(self, d, ms, x, y, w, blocs, *a, **k):
            so = getattr(blocs, "sollicitations", None)
            if so is None:
                return super()._data(d, ms, x, y, w, blocs, *a, **k)
            return _ecrire_sollicitations(d, self, x, y, w, so)
    s = StyleNote()
    s.n_cols = base.n_cols
    return s


def coupe_min(sec: Dict) -> float:
    """La plus petite hauteur (pt) que la coupe d'une planche peut prendre sans que ses textes se chevauchent :
    COUPE_MIN de son nombre de schémas, pour son épaisseur."""
    a, b = COUPE_MIN[min(2, max(1, len(sec["coupe"]["schemas"])))]
    return a + b * float(sec["coupe"]["h"]) / 10.0


# les titres des vérifications d'une planche en TROIS colonnes, dits plus court pour y tenir (la colonne fait 143 pt) :
# rien ne se perd — le titre garde la face, la direction et son rôle
TITRES_COURTS = (("Armatures inférieures", "Armatures inf."), ("Armatures supérieures", "Armatures sup."),
                 ("Sections d'acier minimale et maximale", "Acier minimal et maximal"))


def forme_courte(expr: str) -> str:
    """La forme courte d'une formule « symbole = expression en symboles = substitution = résultat » : sans l'expression en
    symboles (« A_s,req = 35·10⁶ / (333,3 · 0,9 · 165) = 707 mm² ») — pour une planche en trois colonnes ; une formule
    plus courte, ou qui commence par « = », reste telle quelle."""
    parts = expr.split(" = ")
    if len(parts) >= 4 and parts[0].strip():
        return " = ".join([parts[0]] + parts[2:])
    return expr


def _trois_colonnes(sec: Dict) -> None:
    """Une planche qui déborde en deux colonnes passe, seule, en trois (``sec["n_cols"]``, l'extension du kit), ses
    titres raccourcis (TITRES_COURTS), les formules dans leur forme courte (le 4ᵉ élément d'une ligne « f » quand
    ``note_dalle_planche`` en donne une, sinon ``forme_courte``), les lignes en gras dans la leur (le 3ᵉ élément d'une ligne
    « t ») — la colonne fait 143 pt."""
    sec["n_cols"] = 3
    for v in sec["verifs"]:
        for long_, court in TITRES_COURTS:
            v["titre"] = v["titre"].replace(long_, court)
        items = []
        for it in v["items"]:
            if it[0] == "f":
                items.append((it[0], it[1], it[3] if len(it) > 3 else forme_courte(it[2])))
            elif it[0] == "t" and len(it) > 2:      # une ligne en gras : sa forme en trois colonnes (une ligne, ou repliée)
                items += [("t", it[2])] if isinstance(it[2], str) else list(it[2])
            else:
                items.append(it)
        v["items"] = items


def _titres_coupe(coupe: Dict, principale: str) -> None:
    """Les titres des schémas de la coupe, ceux de l'écran (DAL5a) : « Direction Y · principale », « Direction X » ; un
    seul schéma (directions identiques) : « Directions Y et X (identiques) · principale Y »."""
    p, s = principale.upper(), MO.direction_secondaire(principale).upper()
    sch = coupe["schemas"]
    if len(sch) == 1:
        sch[0]["titre"] = f"Directions {p} et {s} (identiques) · principale {p}"
    else:
        sch[0]["titre"], sch[1]["titre"] = f"Direction {p} · principale", f"Direction {s}"


def _treillis_une_fois(coupe: Dict, d: Dict, s: Dict, principale: str) -> None:
    """Dans le schéma de la direction SECONDAIRE, les fils d'un treillis unique ne répètent pas sa désignation, que le
    schéma principal donne déjà (point 4 du propriétaire, DAL5a sur l'écran) : « Inf. X : fils du treillis », avec la
    section de ses fils (mm²/m), qui reste propre à la direction."""
    if len(coupe["schemas"]) < 2:
        return
    sec_ = coupe["schemas"][1]
    q = MO.direction_secondaire(principale).upper()
    for cote in ("inf", "sup"):
        labs = sec_.get(f"labs_{cote}") or []
        if labs and POS.treillis(d, s, cote):
            labs[0] = (f"{'Inf.' if cote == 'inf' else 'Sup.'} {q} : fils du treillis", labs[0][1])


def sections(etat: Dict) -> List[Dict]:
    """Les planches mises en forme : la coupe du générateur (``construire_sections_dalle`` : ses niveaux remplacés par
    ceux du calcul, ses titres par ceux de l'écran, le treillis nommé une fois) ; le CONTENU du site
    (``note_dalle_planche``) — l'en-tête compact, les sollicitations en tableau, les vérifications détaillées, les
    avertissements s'il y en a ; l'état de l'écran."""
    from ndc_pdf import data as ND
    from . import note_dalle_planche as NP
    planches = resultats(etat)
    out = ND.construire_sections_dalle(planches)
    for sec, pl in zip(out, planches):
        rs, d, i = pl["site"], pl["etat_dalle"], pl["indice"]
        p = pl["R"]["principale"]
        sec["etat"] = pl["etat_label"]
        _niveaux_du_calcul(sec["coupe"], pl["R"])
        _titres_coupe(sec["coupe"], p)
        _treillis_une_fois(sec["coupe"], d, d["sections"][i], p)
        sec["entete"] = NP.entete(d, rs)
        sec["blocs"] = Panneau(NP.sollicitations(rs))
        sec["coupe_h"] = COUPE_H
        sec["verifs"] = NP.verifications(d, i, rs, pl["avertissements"])
    return out


def avertissements_de(sec: Dict) -> List[str]:
    """Les avertissements écrits sur une planche (le bloc « ! » de fin de planche), en texte."""
    from . import note_dalle_planche as NP
    return [it[1] for v in sec.get("verifs", []) if v.get("titre") == NP.TITRE_AVERTISSEMENTS for it in v["items"]
            if it[0] == "p"]


# ------------------------------------------------------------------ le document
def document(etat: Dict, infos: Optional[Dict], chemin: str):
    """Le document construit (pas encore écrit) : la garde et les planches, comme ``generer_rapport_pdf`` (même
    construction, même repli en trois colonnes si une planche déborde). ``d.warnings`` : ce qui déborde encore ;
    ``d.colonnes`` : 2, ou 3 quand une planche au moins a dû passer en trois colonnes ; ``d.trois_colonnes`` : celles
    qui sont, seules, en trois colonnes."""
    from ndc_pdf import data as ND
    secs = sections(etat)
    doc_meta = ND.construire_doc(infos or {}, date_defaut=datetime.today().strftime("%d/%m/%Y"))
    d = _style(2).build(chemin, sections=secs, doc=doc_meta)
    d.colonnes = 2
    trois = [sec for sec in secs if any(w.startswith(f"{sec['poutre']} — {sec['section']} :") for w in d.warnings)]
    if trois:                            # la planche qui déborde (seule) en trois colonnes, ses titres raccourcis
        for sec in trois:
            _trois_colonnes(sec)
        d = _style(2).build(chemin, sections=secs, doc=doc_meta)
        d.colonnes = 3
    if d.warnings:                       # le repli du générateur : toute la note en trois colonnes
        d = _style(3).build(chemin, sections=secs, doc=doc_meta)
        d.colonnes = 3
    d.trois_colonnes = len(trois)
    return d


def pdf(etat: Dict, infos: Optional[Dict] = None) -> bytes:
    """La note PDF (octets) : garde A4 portrait + une planche A4 paysage par section. Une planche qui déborde encore en
    trois colonnes est REFUSÉE avec un message : ce qui sort de la page serait perdu sans un mot — jamais une note
    tronquée."""
    import tempfile
    fd, chemin = tempfile.mkstemp(suffix=".pdf", prefix="note_dalle_")
    os.close(fd)
    try:
        d = document(etat, infos, chemin)
        if d.warnings:
            raise ValueError("La note ne tient pas sur sa planche (" + " ; ".join(d.warnings) + ") : elle ne sort pas "
                             "tronquée.")
        d.save()
        with open(chemin, "rb") as f:
            return f.read()
    finally:
        try:
            os.remove(chemin)
        except OSError:
            pass


def fichier(etat: Dict, infos: Optional[Dict] = None) -> Dict:
    """Le fichier à télécharger : {"nom", "type": "application/pdf", "base64"} (le routeur le contrôle). ``infos`` :
    {nom, chantier, indice, date} de la note du site → la page de garde (projet, partie, indice, date)."""
    infos = dict(infos or {})
    nom = str(infos.get("nom") or "").strip() or (etat["dalles"][0].get("nom") or "Dalles")
    chantier = str(infos.get("chantier") or "").strip()
    doc = {"nom_projet": chantier or nom, "partie": nom if chantier else PARTIE,
           "indice": str(infos.get("indice") or "A"), "date": str(infos.get("date") or "")}
    octets = pdf(copy.deepcopy(etat), doc)
    from . import commun as C
    return {"nom": C.nom_fichier(f"Note de calcul - {nom}", "pdf"), "type": "application/pdf",
            "base64": base64.b64encode(octets).decode("ascii")}

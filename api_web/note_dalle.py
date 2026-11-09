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

CE QUE LE SITE AJOUTE À LA PLANCHE (le générateur ne le sait pas) :
- l'ÉTAT : « Données incomplètes » pour une section sans aucun effort (jamais « Vérifié »), « Non vérifié » / « Vérifié »
  sinon — l'état de l'écran (api_web/plaque_compact.etat_bloc) ;
- les NIVEAUX DESSINÉS de la coupe : ceux du calcul (le générateur écartait deux axes trop proches pour le dessin —
  ``_coupe_dalle_depuis_R._niveaux`` — et dessinait un renfort, posé dans le plan de sa base, plus haut qu'il n'est
  compté) ; chaque couche est dessinée à l'axe qui entre dans d, comme sur la coupe de l'écran ;
- le VERDICT d'une nappe non conforme alors que A_s ≥ A_s,néc : acier excessif (A_s > A_s,max) ou d utile ≤ 0 — le
  générateur y écrirait « A_s < A_s,requise », qui est faux ;
- les OBSERVATIONS : les avertissements que l'écran affiche sous la section (enrobage nul lu 3 cm, nappes au même
  niveau par un axe saisi, base hors de la prédalle, couches croisées sur la prédalle, d utile ≤ 0) et, une fois par
  dalle, le périmètre d'une prédalle (ce qui n'est PAS vérifié) — les textes mêmes de l'écran, sans leur balisage ;
  écrites sous les données du panneau de gauche (la coupe cède, s'il le faut, le juste nécessaire sans descendre sous
  ``coupe_min``), sinon en fin de planche (la planche, seule, en trois colonnes si elle déborde : ``document``).

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
TITRE_OBSERVATIONS = "Observations"
# les OBSERVATIONS s'écrivent sous les données du panneau de gauche ; quand la place y manque, la coupe cède le juste
# nécessaire, sans descendre sous ``coupe_min`` ; au-delà, elles passent en fin de planche (``_observations_en_colonnes``).
# MESURÉ le 09/11/2026 (en réduisant la coupe de 5 en 5 pt jusqu'au premier chevauchement de deux de ses textes) : la plus
# petite coupe à deux schémas sans chevauchement fait 180 à 200 pt pour h = 10 à 15 cm, 225 (20 cm), 240 (30 cm), 245
# (40 cm), 260 (50 cm), 280 (60 cm), CSP comme prédalle — sous elle, l'étiquette de la nappe supérieure monte sur la
# ligne du titre de son schéma (« filants : dir. Y »). COUPE_MIN = (a, b) par nombre de schémas : a + b·h (h en cm),
# au-dessus de chaque mesure d'au moins 10 pt ; tests/test_note_dalle.py le revérifie (h de 12 à 60 cm).
COUPE_MIN = {1: (110.0, 0.75), 2: (205.0, 1.5)}
TETE_OBSERVATIONS = 17.0      # pt : le titre « OBSERVATIONS » et son filet, au-dessus du texte
ECART_OBSERVATIONS = 2.5      # pt : entre deux observations
INTERLIGNE = 1.32             # l'interligne des paragraphes du kit (Doc.para)
MARGE_PANNEAU = 2.0           # pt : la dernière ligne au-dessus de la marge de la page (le panneau descend à marge − 4)
CLE_HC = "hauteur_coupe"      # la hauteur de la coupe à la dernière construction (pt)
# la place des étiquettes à droite de la coupe : celle que ndc_pdf/styles.planche donne à draw_section (label_w=96) —
# le dessinateur des planches qui ont des observations dessine la même coupe (tests/test_note_dalle.py le compare)
LARGEUR_ETIQUETTES = 96


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


def observations(d: Dict, r: Dict, i: int) -> List[str]:
    """Ce que l'écran affiche sous la section ``i`` en plus des vérifications : ses avertissements (ceux du type :
    api_web/dalle._avertissements_de) ; sur la première planche de la dalle, ceux de la dalle entière (sans section)
    puis le périmètre du type (ce qui n'est pas vérifié : la prédalle, et depuis le 10/11/2026 la dalle coulée en place,
    api_web/dalle.texte_perimetre) — l'écran le dit une fois, sous le résultat de la dalle. Les textes de l'écran, sans
    balisage."""
    from . import dalle as DL
    out = _messages(DL._avertissements_de(d, r, i))
    if i == 0:                           # ce qui vaut pour toute la dalle : une fois, sur sa première planche
        out += _messages(DL._avertissements_de(d, r))
        perimetre = DL.texte_perimetre(d)
        if perimetre:
            out.append(_texte(perimetre))
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
    site, l'état, les observations). Lève ValueError si une planche ne dit pas les chiffres de l'écran (``ecarts``) :
    la note ne sort pas."""
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
            pl["etat_label"] = etat_label(rs)
            pl["observations"] = observations(d, r, i)
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


def _verdict_site(v_pdf: Dict, v: Dict, rs: Dict) -> None:
    """Le verdict d'une nappe que le moteur déclare non conforme alors que A_s couvre A_s,néc (acier excessif, d ≤ 0) :
    la comparaison qui la rend non conforme, comme l'écran (api_web/dalle._verdict)."""
    from ndc_pdf import data as ND
    if v["etat"] != MO.NOK or v["taux"] > 1:
        return
    cote, dk = v["cle"].split("_")
    face = "inférieure" if cote == "inf" else "supérieure"
    if "acier_excessif" in v["motifs"]:
        txt = (f"Section d'armature {face} {dk.upper()} : {ND.fn(v['limite'], 0)} mm² > section d'armature maximale : "
               f"{ND.fn(v['As_max'], 0)} mm² (acier excessif)")
    elif "d_utile_negatif" in v["motifs"]:
        d_u = rs["grandeurs"]["d_" + v["cle"]]
        txt = (f"Hauteur utile {face} {dk.upper()} : d = {ND.fn(d_u, 1)} cm ≤ 0 — position des couches incompatible avec "
               "l'épaisseur")
    else:
        return
    v_pdf["verdicts"] = [dict(etat="ko", texte=ND._unites_insecables(txt))]


class _Blocs(list):
    """Les blocs de données d'une planche (``sec["blocs"]`` : DIMENSIONS, MATÉRIAUX, SOLLICITATIONS), qui portent en
    plus les OBSERVATIONS de l'écran à écrire dessous (``observations``) — et, après une construction, la place qui leur
    a manqué (``manque``, pt). Une liste ordinaire pour le kit : ``_data`` la parcourt comme avant."""
    def __init__(self, blocs=(), observations=()):
        super().__init__(blocs)
        self.observations = list(observations)
        self.manque = 0.0


def _hauteur_observations(d, S, textes: List[str], largeur: float) -> float:
    """La hauteur (pt) des observations repliées dans ``largeur`` : le titre, puis chaque texte."""
    return TETE_OBSERVATIONS + sum(len(d.wrap(t, S.f_b, S.s_lab, largeur)) * S.s_lab * INTERLIGNE + ECART_OBSERVATIONS
                                   for t in textes)


def _ecrire_observations(d, S, x: float, haut: float, w: float, textes: List[str]) -> None:
    """Les observations sous les données du panneau de gauche, à partir de ``haut`` : un titre comme « DIMENSIONS »,
    puis chaque texte replié dans la largeur du panneau."""
    from ndc_pdf.kit import Frame, mix
    fr = Frame(x, haut, w, 0)
    fr.down(12)
    d.t(fr.x, fr.y, TITRE_OBSERVATIONS.upper(), S.f_v, 6.3, S.acc, track=1.15)
    fr.down(3)
    d.line(fr.x, fr.y, fr.x1, fr.y, mix(S.rule, S.ink, 0.12), 0.5)
    for t in textes:
        fr.down(ECART_OBSERVATIONS)
        d.para(fr, t, S.f_b, S.s_lab, S.ink, lead=INTERLIGNE)


def _style(n_cols: int):
    """Le style de la note : celui du générateur (``export_pdf._style_ndc`` : la palette, le repli en trois colonnes),
    dont les DONNÉES du panneau de gauche (``_data``) sont suivies des observations de l'écran, quand la planche en a
    (``_Blocs.observations``) et que la place, sous les données, les contient ; sinon la place qui manque est notée
    (``_Blocs.manque``) et rien n'est écrit — ``document`` fait alors de la place (la coupe moins haute) ou reporte les
    observations en fin de planche. Hériter du style est l'extension du kit : ni le kit ni le générateur ne changent."""
    from modules import export_pdf as EP
    base = EP._style_ndc(n_cols)

    class StyleNote(type(base)):
        def _data(self, d, ms, x, y, w, blocs, *a, **k):
            bas = super()._data(d, ms, x, y, w, blocs, *a, **k)
            textes = getattr(blocs, "observations", None)
            if textes:
                besoin = _hauteur_observations(d, self, textes, w)
                place = bas - (self.margin + MARGE_PANNEAU)
                blocs.manque = max(0.0, besoin - place)
                if not blocs.manque:
                    _ecrire_observations(d, self, x, bas, w, textes)
            return bas
    s = StyleNote()
    s.n_cols = base.n_cols
    return s


def _hauteur_coupe(sec: Dict):
    """Le dessinateur du panneau de gauche (l'extension ``sec["dessin"]`` du kit) d'une planche qui a des observations :
    la coupe telle que le kit la dessine (``draw_section``, ``LARGEUR_ETIQUETTES``), sa hauteur notée (``CLE_HC``) —
    pour la réduire, s'il le faut, du juste nécessaire."""
    def dessin(d, x, y, w, h, S):
        from ndc_pdf import styles as STY
        sec[CLE_HC] = h
        STY.draw_section(d.c, x, y, w, h, sec["coupe"], S.sec_style(), label_w=LARGEUR_ETIQUETTES)
    return dessin


def coupe_min(sec: Dict) -> float:
    """La plus petite hauteur (pt) que la coupe d'une planche peut prendre sans que ses textes se chevauchent :
    COUPE_MIN de son nombre de schémas, pour son épaisseur."""
    a, b = COUPE_MIN[min(2, max(1, len(sec["coupe"]["schemas"])))]
    return a + b * float(sec["coupe"]["h"]) / 10.0


# les titres des vérifications d'une planche en TROIS colonnes : ceux du générateur n'y tiennent pas (« Armatures
# supérieures — direction principale » fait 160 pt au plus petit corps du kit, la colonne 123) ; dits plus court, sans
# rien perdre (le premier élément de chaque vérification nomme encore la face et la direction : « Moment supérieur Y »)
TITRES_COURTS = (("Armatures inférieures", "Armatures inf."), ("Armatures supérieures", "Armatures sup."),
                 (" — direction principale", " — principale"), (" — direction secondaire", " — secondaire"))


def _trois_colonnes(sec: Dict) -> None:
    """Une planche qui déborde en deux colonnes passe, seule, en trois (``sec["n_cols"]``, l'extension du kit), ses
    titres raccourcis (TITRES_COURTS)."""
    sec["n_cols"] = 3
    for v in sec["verifs"]:
        for long_, court in TITRES_COURTS:
            v["titre"] = v["titre"].replace(long_, court)


def _observations_en_colonnes(sec: Dict) -> None:
    """Le repli : les observations en fin de planche, une vérification de plus (du texte, sans verdict) ; la coupe
    reprend sa hauteur."""
    textes = list(sec["blocs"].observations)
    sec["blocs"].observations = []
    sec.pop("coupe_h", None)
    sec["verifs"].append(dict(num=len(sec["verifs"]) + 1, titre=TITRE_OBSERVATIONS,
                              items=[("p", t) for t in textes], verdicts=[]))


def sections(etat: Dict) -> List[Dict]:
    """Les planches mises en forme (``construire_sections_dalle``), complétées : l'état de l'écran, les niveaux du
    calcul sur la coupe, le verdict d'une nappe non conforme sous 100 %, les observations de l'écran (sous les données
    du panneau de gauche : ``sec["blocs"].observations``)."""
    from ndc_pdf import data as ND
    planches = resultats(etat)
    out = ND.construire_sections_dalle(planches)
    for sec, pl in zip(out, planches):
        rs = pl["site"]
        sec["etat"] = pl["etat_label"]
        _niveaux_du_calcul(sec["coupe"], pl["R"])
        # les vérifications 2 à 5 de la planche : les quatre nappes dans l'ordre du générateur (principale puis
        # secondaire, en bas puis en haut) — celui de l'écran, MO.ordre_faces
        par_cle = {v["cle"]: v for v in rs["verifications"]}
        for v_pdf, f in zip(sec["verifs"][1:5], MO.ordre_faces(pl["R"]["principale"])):
            _verdict_site(v_pdf, par_cle[f], rs)
        sec["blocs"] = _Blocs(sec["blocs"])
        if pl["observations"]:
            sec["blocs"].observations = list(pl["observations"])
            sec["dessin"] = _hauteur_coupe(sec)
    return out


def observations_de(sec: Dict) -> List[str]:
    """Les observations d'une planche, où qu'elles soient écrites (sous les données, ou en fin de planche)."""
    textes = list(getattr(sec.get("blocs"), "observations", []) or [])
    for v in sec.get("verifs", []):
        if v.get("titre") == TITRE_OBSERVATIONS:
            textes += [it[1] for it in v["items"] if it[0] == "p"]
    return textes


# ------------------------------------------------------------------ le document
def document(etat: Dict, infos: Optional[Dict], chemin: str):
    """Le document construit (pas encore écrit) : la garde et les planches, comme ``generer_rapport_pdf`` (même
    construction, même repli en trois colonnes si une planche déborde). Les observations d'une planche s'écrivent sous
    ses données ; quand la place manque, la coupe cède le juste nécessaire tant qu'elle garde COUPE_MIN (ses étiquettes
    ne se chevauchent pas), sinon elles passent en fin de planche. ``d.warnings`` : ce qui déborde encore ;
    ``d.colonnes`` : 2, ou 3 quand une planche au moins a dû passer en trois colonnes ; ``d.replis`` : les planches
    dont les observations sont en fin de planche ; ``d.coupes_reduites`` : celles dont la coupe a cédé de la place ;
    ``d.trois_colonnes`` : celles qui sont, seules, en trois colonnes."""
    from ndc_pdf import data as ND
    secs = sections(etat)
    doc_meta = ND.construire_doc(infos or {}, date_defaut=datetime.today().strftime("%d/%m/%Y"))
    d = _style(2).build(chemin, sections=secs, doc=doc_meta)
    reduites, replis = 0, 0
    for sec in secs:
        manque = sec["blocs"].manque
        if not manque:
            continue
        if sec[CLE_HC] - manque - 1.0 >= coupe_min(sec):
            sec["coupe_h"] = sec[CLE_HC] - manque - 1.0
            reduites += 1
        else:
            _observations_en_colonnes(sec)
            replis += 1
    if reduites or replis:
        d = _style(2).build(chemin, sections=secs, doc=doc_meta)
        for sec in secs:                 # (une coupe réduite fait toujours la place : par sûreté, le repli)
            if sec["blocs"].manque and sec["blocs"].observations:
                _observations_en_colonnes(sec)
                replis += 1
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
    d.replis, d.coupes_reduites, d.trois_colonnes = replis, reduites, len(trois)
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

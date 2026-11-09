# -*- coding: utf-8 -*-
"""Les NIVEAUX réels des deux directions d'une dalle (module « Dalles béton armé », api_web/dalle.py) : l'axe
« automatique DU SITE » d'une couche, la copie de la dalle que le moteur lit, les axes effectifs pour l'écran et le
dessin — sans toucher aux moteurs ``beton/dalle`` et ``beton/predalle`` (docs/site/DALLES_FUSION.md, § 4).

POURQUOI. Les deux directions d'une nappe se croisent : leurs axes sont à des hauteurs différentes. Les moteurs, extraits
des écrans Streamlit, posent les deux bases d'une même face AU MÊME NIVEAU (``distance_auto`` ne lit pas l'autre
direction) et le disent par l'avertissement « nappes_confondues ». Le moteur honore une position IMPOSÉE (``dist``) :
le site calcule donc l'axe automatique de la couche telle qu'elle est posée, et l'écrit dans la copie envoyée au moteur
quand il diffère de l'automatique du moteur. L'état garde ``dist = None`` (automatique) ; le résultat du moteur porte
l'axe effectivement utilisé (``couches[j]["e"]``), que l'écran et le dessin lisent.

LES NAPPES SONT TOUJOURS EMPILÉES (décision de l'ingénieur du 09/11/2026 : plus d'option — anciennes notes, notes de
l'ancien module Prédalles et fichiers Streamlit importés compris ; la clé d'état ``empilement`` d'avant ne pilote plus
rien : api_web/dalle.normaliser la relit sans effet).

LA RÈGLE (``axe_auto``), avec la convention d'arrondi du moteur — demi-diamètre arrondi au 0,5 cm supérieur,
``MO_D.arrondi_demi_cm(Ø / 20)`` (Ø en mm) :
  · direction PRINCIPALE, base et renforts : l'automatique du moteur (dalle : enrobage + ⌈Ø/2⌉ + jeu ; prédalle : sa
    règle de pose, ``beton/predalle/moteur.pose``) — la nappe la plus proche du parement ;
  · direction SECONDAIRE, faces inférieures ET supérieures d'une dalle coulée, faces SUPÉRIEURES d'une prédalle : posée
    SUR la nappe principale, les barres en contact — sa BASE comme ses RENFORTS (S3, décision de l'ingénieur du
    09/11/2026 : un renfort se pose sur SA nappe, entre ou à côté des barres de base de sa direction, dans leur lit) —,
    et elle repose sur la couche principale LA PLUS HAUTE qu'elle croise (S5, relecture du 09/11/2026 : un renfort
    principal plus épais que sa base, posé dans le lit de la base, porte la nappe secondaire ; la poser sur la base
    seule faisait traverser les barres) :
        ``max(e_k + ⌈Ø_k/2⌉) + ⌈Ø/2⌉``, le max sur les couches k du LIT DE CONTACT (ci-dessous)
    (e_k : l'axe EFFECTIF de la couche principale k, saisi s'il l'est, sinon l'automatique du moteur — la nappe
    secondaire repose sur la principale là où elle est ; Ø_k : ses fils porteurs ou ses barres, Øl pour un treillis ;
    Ø : la couche posée) ; à égalité, la base l'emporte (la formule d'avant S5, au flottant près). La mesure est depuis
    le parement de la face (dessus pour les supérieures) ;
  · prédalle, faces INFÉRIEURES : la règle du moteur, conservée telle quelle (base principale DANS la peau, base
    secondaire et TOUS les renforts inférieurs SUR la prédalle, h_p + ⌈Ø/2⌉ : les couches secondaires y sont déjà dans
    le lit de leur base ; un renfort principal posé sur la prédalle y croise la base secondaire au même niveau, ce que le
    moteur DIT — « croisement_sur_predalle » — et que la question 11 de l'ingénieur, « empiler les couches basses de la
    prédalle », tranchera : S5 n'y touche pas).
  Un axe SAISI (``dist``) n'est jamais déplacé.
  Le renfort secondaire prend la formule de sa base parce que les deux moteurs posent un renfort dans le MÊME LIT que la
  base de sa face (``distance_auto`` ne dépend pas du rang, hors faces inférieures d'une prédalle) : son décalage
  relatif à la base est nul, il le reste une fois empilé (tests/test_api_web_dalle_positions.py le vérifie). Sens de la
  sécurité : le renfort monte, son d utile baisse, A_s,req augmente ; de même pour S5, la nappe secondaire ne descend
  jamais (le max contient la base).

LE LIT DE CONTACT (S5, ``lit_contact``) — les couches principales que la nappe secondaire croise, et sur lesquelles elle
repose : en convention du moteur, le DESSOUS d'une couche est e − ⌈Ø/2⌉ (vers le parement), son DESSUS e + ⌈Ø/2⌉. Le lit
part de la BASE principale ; une autre couche principale en fait partie si son dessous ne dépasse pas le dessus du lit
(elle le touche ou le chevauche : la nappe secondaire, qui repose au moins sur ce dessus, ne peut pas passer sous elle),
et le dessus du lit monte alors au sien — jusqu'à ce que plus rien n'entre (un renfort qui chevauche un autre renfort
du lit en est aussi). TOLÉRANCE : ``TOL_LIT`` = 1e-6 cm, les arrondis des flottants, rien de plus — une couche dont le
dessous est plus haut que le dessus du lit, même de 1 mm, est un AUTRE lit (un 2e lit principal posé sur la nappe
secondaire ou sur des cales) : la nappe secondaire ne s'y appuie pas. Pourquoi aucune marge : un 2e lit posé sur la
nappe secondaire a son dessous, en convention du moteur, au moins 0,05 cm au-dessus du dessus du lit, même placé aux
Ø réels (Ø en mm entiers, ⌈Ø/2⌉ dépasse Ø/20 de 0,45 cm au plus, fil de 5 mm au moins : e − ⌈Ø/2⌉ ≥ dessus +
Ø_s/10 − 0,45 ≥ dessus + 0,05) ; une marge positive l'avalerait.
Avec les axes automatiques, le lit contient TOUTES les couches principales (le moteur les pose toutes au même dessous :
enrobage + jeu) ; seule une couche SAISIE peut en sortir. Une couche principale AUTOMATIQUE est du lit quoi qu'il arrive
(relecture S5 du 09/11/2026 : sous une base SAISIE plus bas que l'automatique — Ø8 saisie à 3,4 avec un jeu de 1 cm,
dessus 3,9 —, un renfort 3 Ø20 automatique, posé sur les cales à 4,0, sortait du lit ; la nappe secondaire, posée sur la
base à 4,4, le traversait sans rien dire, A_s,req 427,4 au lieu de 493,8 mm²).
  Cas limites (tests/test_api_web_dalle_positions.py (g)) : un renfort principal SAISI dans un 2e lit (dessous au-dessus
  du lit) ne porte rien ; plusieurs renforts : le plus haut du lit porte, la base à égalité ; une base principale saisie
  plus haut que son renfort : la base porte ; un renfort saisi qui chevauche le lit sans être à l'automatique en est ;
  un renfort principal de même ⌈Ø/2⌉ que la base (Ø16 sur Ø20) : rien ne change. Ce que la règle ne dit pas : une
  couche principale SAISIE dont le dessous tombe DANS l'épaisseur de la nappe secondaire (au-dessus du lit, sous le
  dessus de la secondaire) n'est pas du lit et rien ne le signale — l'axe saisi est celui qu'on a voulu (écrit dans
  docs/site/DALLES_FUSION.md § 4.3, révision S5).

EXEMPLE (dalle coulée, c = 3 cm, jeu 0, principale Y, bases Ø12 en Y et treillis 10/10/100/100 en X) :
  inf. Y (principale) : 3 + ⌈1,2/2⌉₀,₅ = 3 + 1,0 = 4,0 cm ;
  inf. X (secondaire) : 4,0 + 1,0 + ⌈1,0/2⌉₀,₅ = 4,0 + 1,0 + 0,5 = 5,5 cm ;
  sup. Y : 4,0 cm ; sup. X : 5,5 cm (depuis le dessus) ; un renfort 3 Ø12 en inf. Y (principale) : 3 + 1,0 = 4,0 cm
  (règle du moteur) ;
  S3 — un renfort 3 Ø12 en inf. X (secondaire) : 4,0 + 1,0 + 1,0 = 6,0 cm, sur la nappe Y comme la base X (avant le
  09/11/2026 : 3 + 1,0 = 4,0, au niveau de la base Y, sous sa propre nappe) ; un renfort Ø25 : 4,0 + 1,0 + 1,5 = 6,5 cm.
  S5 — la base inf. Y en treillis 10/10/100/100 (Ø10 : 3 + 0,5 = 3,5) et un renfort 3 Ø25 en inf. Y (3 + 1,5 = 4,5, le
  même dessous 3,0, dans le lit) : la base inf. X repose sur le renfort, max(3,5 + 0,5 ; 4,5 + 1,5) + 0,5 = 6,5 cm
  (sur la base seule : 3,5 + 0,5 + 0,5 = 4,5) ; le même renfort saisi à 9 cm (dessous 7,5 > 4,0) : un 2e lit, 4,5.
  Même dalle en PRÉDALLE (h_pre = 6 cm) : inf. Y 4,0 (dans la peau), inf. X 6 + 0,5 = 6,5 (sur la prédalle, règle du
  moteur), renfort inf. Y 6 + 1,0 = 7,0 et renfort inf. X 6 + 1,0 = 7,0 (règle du moteur) ; sup. Y 4,0, sup. X 5,5
  (empilées), un renfort 3 Ø16 en sup. X : 4,0 + 1,0 + 1,0 = 6,0 (S3).

API
---
    axe_auto(d, s, f, j) -> float          l'axe automatique DU SITE (cm, depuis le parement) de la couche j de la face f
    lit_contact(d, s, cote) -> [rangs]     les couches principales de la face (inf | sup) que la nappe secondaire croise
    appui(d, s, cote) -> dict              celle sur laquelle elle repose : {couche, e, d, demi, dessus, lit}
    nom_appui(d, s, cote, court) -> str    la même, en mots : « la base Y », « le renfort Y », « le renfort R2 Y » ;
                                           court (légendes) : « la base Y », « renfort Y »
    empilee(d, s, f, j) -> bool            la couche est-elle posée par le site sur la nappe principale (axe du site ≠ moteur)
    axes(d, s) -> {face: [axe effectif…]}  imposé, sinon l'automatique du site — pour le dessin et l'écran
    pour_moteur(d) -> dict                 la dalle telle que le moteur la lit : sans les clés du site, ``dist`` renseigné
                                           là où l'automatique du site diffère de celui du moteur ; ``d`` n'est pas modifié
    yG_calcule(d, s, f) / h_utile(d, s, f) le centre de gravité calculé et la hauteur utile avec les axes effectifs
    annoter(r, d) -> r                     le résultat du moteur relu avec la notion d'automatique DU SITE (auto, e_auto,
                                           e_auto_moteur, empilee, et pour une couche empilée appui, sur et
                                           sur_court) — ce que l'écran et les dessins lisent
    candidats_base(d, i, f, familles)      le sélecteur de la nappe de base, chaque candidat posé à SON axe empilé (les
                                           couches secondaires posées sur lui le suivent)
    moteur(d), type_dalle(d), principale(d)

``d`` est une dalle du SITE (normalisée par le moteur de son type, plus ``verrou`` et ``type`` ∈ {coulee, predalle} —
absent : coulee) ; ``s`` une de ses sections ; ``f`` une face (« inf_x », « inf_y », « sup_x »,
« sup_y ») ; ``j`` le rang de la couche (0 : la base). Rien de normatif n'est calculé ici : les axes du moteur sont
ceux du moteur, l'empilement est une addition de demi-diamètres arrondis par la fonction du moteur, et la hauteur utile
est lue dans ``beton/dalle/moteur._face`` sur des couches aux axes figés.
"""
import copy
from typing import Dict, List, Optional, Sequence

from beton.dalle import entree as EN_D
from beton.dalle import moteur as MO_D
from beton.predalle import moteur as MO_P

FACES = EN_D.FACES
TYPES = ("coulee", "predalle")
TYPE_DEFAUT = "coulee"
CLES_SITE = frozenset({"verrou", "type"})     # ce que le site ajoute à la dalle du moteur (= dalle.CLES_SITE, lié par un test)
TOL_LIT = 1e-6                                # cm : le contact d'une couche avec le lit, aux arrondis des flottants près (S5)


# ------------------------------------------------------------------ lire la dalle du site
def type_dalle(d: Dict) -> str:
    """« coulee » ou « predalle » (absent ou inconnu : coulee)."""
    t = d.get("type")
    return t if t in TYPES else TYPE_DEFAUT


def moteur(d: Dict):
    """Le moteur du type : ``beton.dalle.moteur`` ou ``beton.predalle.moteur``."""
    return MO_P if type_dalle(d) == "predalle" else MO_D


def principale(d: Dict) -> str:
    """La direction principale, en minuscule (« x » ou « y »), quelle que soit la casse stockée (le moteur garde « X »/« Y »)."""
    return str(d.get("dir_principale") or EN_D.DEFAUTS_DALLE["dir_principale"]).strip().lower()


def _auto_moteur(d: Dict, c: Dict, j: int, f: str) -> float:
    """L'axe automatique DU MOTEUR du type (cm) : la règle de l'écran Streamlit, inchangée."""
    enrobage = d.get("enrobage", EN_D.DEFAUTS_DALLE["enrobage"])
    jeu = d.get("jeu_premier_lit", EN_D.DEFAUTS_DALLE["jeu_premier_lit"])
    if type_dalle(d) == "predalle":
        return MO_P.distance_auto(c, j, f, d.get("dir_principale", EN_D.DEFAUTS_DALLE["dir_principale"]), enrobage, jeu, d.get("h_pre"))
    return MO_D.distance_auto(c, enrobage, jeu)


def _demi(c: Dict) -> float:
    """Le demi-diamètre de la couche, arrondi au 0,5 cm supérieur — la convention du moteur (Ø en mm, /20 → cm)."""
    return MO_D.arrondi_demi_cm(MO_D.couche_diametre(c) / 20.0)


# ------------------------------------------------------------------ l'axe automatique du site
def axe_auto(d: Dict, s: Dict, f: str, j: int) -> float:
    """L'axe automatique DU SITE (cm, depuis le parement de la face) de la couche ``j`` de la face ``f`` de la section
    ``s`` — la règle et l'exemple chiffré sont en tête du fichier. Direction principale (base et renforts), face
    inférieure d'une prédalle : exactement ``distance_auto`` du moteur du type. Direction secondaire, sa base ET ses
    renforts (S3) : posée sur la nappe principale de la même face, sur la couche la plus haute de son lit de contact
    (S5) — le dessus de cette couche (son axe effectif + ⌈Ø_p/2⌉) + ⌈Ø/2⌉."""
    c = s["faces"][f]["couches"][j]
    cote, dk = f.split("_")
    if dk == principale(d) or (cote == "inf" and type_dalle(d) == "predalle"):
        return _auto_moteur(d, c, j, f)
    return appui(d, s, cote)["dessus"] + _demi(c)


def _couches_principales(d: Dict, s: Dict, cote: str) -> List[tuple]:
    """(rang, axe EFFECTIF, ⌈Ø/2⌉) de chaque couche de la direction principale de la face ``cote`` (inf | sup), base
    d'abord : l'axe saisi s'il l'est, sinon l'automatique du moteur (une couche principale n'est jamais empilée) — là
    où la nappe secondaire la croise."""
    f = f"{cote}_{principale(d)}"
    return [(j, c["dist"] if c.get("dist") is not None else _auto_moteur(d, c, j, f), _demi(c))
            for j, c in enumerate(s["faces"][f]["couches"])]


def lit_contact(d: Dict, s: Dict, cote: str) -> List[int]:
    """Les rangs des couches principales de la face ``cote`` dans le LIT DE CONTACT (en tête du fichier) : la base, toute
    couche AUTOMATIQUE (posée sur les cales du premier lit, enrobage + jeu : elle ne peut pas être un 2e lit), puis toute
    couche dont le dessous (e − ⌈Ø/2⌉) ne dépasse pas le dessus du lit (à ``TOL_LIT`` près), le dessus du lit montant à
    chaque couche admise. Triés."""
    pts = _couches_principales(d, s, cote)
    auto = [c.get("dist") is None for c in s["faces"][f"{cote}_{principale(d)}"]["couches"]]
    dans, reste = [pts[0]], pts[1:]
    dessus = pts[0][1] + pts[0][2]
    entre = True
    while entre:
        entre = False
        for p in list(reste):
            if auto[p[0]] or p[1] - p[2] <= dessus + TOL_LIT:
                dans.append(p)
                reste.remove(p)
                dessus = max(dessus, p[1] + p[2])
                entre = True
    return sorted(p[0] for p in dans)


def appui(d: Dict, s: Dict, cote: str) -> Dict:
    """La couche principale sur laquelle repose la nappe secondaire de la face ``cote`` (S5) : la plus HAUTE du lit de
    contact (le plus grand dessus e + ⌈Ø/2⌉), la base à égalité (à ``TOL_LIT`` près) — ``{"couche": rang, "e": axe
    effectif, "d": Ø (mm, Øl d'un treillis), "demi": ⌈Ø/2⌉, "dessus": e + ⌈Ø/2⌉, "lit": rangs du lit}``. Le dessus est
    calculé de la couche retenue elle-même (axe + demi) : la base retenue rend, au flottant près, la formule d'avant S5."""
    pts = {p[0]: p for p in _couches_principales(d, s, cote)}
    lit = lit_contact(d, s, cote)
    haut = max(pts[j][1] + pts[j][2] for j in lit)
    k = next(j for j in lit if pts[j][1] + pts[j][2] >= haut - TOL_LIT)
    c = s["faces"][f"{cote}_{principale(d)}"]["couches"][k]
    return {"couche": k, "e": pts[k][1], "d": MO_D.couche_diametre(c), "demi": pts[k][2], "dessus": pts[k][1] + pts[k][2],
            "lit": lit}


def nom_appui(d: Dict, s: Dict, cote: str, court: bool = False) -> str:
    """Ce sur quoi repose la nappe secondaire de la face ``cote``, en mots (formules, pose de la couche) : « la base Y »,
    « le renfort Y » (la face principale n'en porte qu'un) ou « le renfort R2 Y » (le repère de la couche à l'écran,
    api_web/plaque_compact.couche). ``court`` : la forme des LÉGENDES des coupes — « renfort Y » pour tout renfort (sans
    article ni repère ; « la base Y » inchangé) : une ligne de légende doit tenir dans sa colonne, « sur le renfort Y »
    débordait sur la colonne des supérieures (capture S5 du 09/11/2026) et « sur renfort R2 Y » l'aurait fait aussi ;
    « sur renfort Y » a la longueur de « sur la base Y ». Le repère reste dans la forme longue (formule, pose)."""
    k = appui(d, s, cote)["couche"]
    dk = principale(d).upper()
    if k == 0:
        return f"la base {dk}"
    if court:
        return f"renfort {dk}"
    n = len(s["faces"][f"{cote}_{principale(d)}"]["couches"]) - 1
    return f"le renfort {dk}" if n == 1 else f"le renfort R{k} {dk}"


def empilee(d: Dict, s: Dict, f: str, j: int) -> bool:
    """La couche est-elle posée PAR LE SITE sur la nappe principale (base ou renfort de la direction secondaire) :
    automatique (``dist`` vide) et axe du site différent de l'automatique du moteur."""
    c = s["faces"][f]["couches"][j]
    return c.get("dist") is None and axe_auto(d, s, f, j) != _auto_moteur(d, c, j, f)


def axes(d: Dict, s: Dict) -> Dict[str, List[float]]:
    """{face: [axe EFFECTIF de chaque couche]} : l'axe imposé, sinon l'automatique du site — ce que le moteur utilisera."""
    return {f: [c["dist"] if c.get("dist") is not None else axe_auto(d, s, f, j) for j, c in enumerate(s["faces"][f]["couches"])]
            for f in FACES}


# ------------------------------------------------------------------ la dalle que le moteur lit
def pour_moteur(d: Dict) -> Dict:
    """La copie de la dalle telle que le MOTEUR la lit : sans les clés du site (verrou, type) ; pour chaque couche
    automatique dont l'axe du site diffère de l'automatique du moteur (les couches de la direction secondaire posées sur
    la principale), ``dist`` prend l'axe du site (le moteur honore une position imposée) ; les autres couches sont
    inchangées. ``d`` n'est jamais modifié."""
    out = {k: copy.deepcopy(v) for k, v in d.items() if k not in CLES_SITE}
    for s_d, s_m in zip(d.get("sections") or [], out.get("sections") or []):
        for f in FACES:
            for j, (c_d, c_m) in enumerate(zip(s_d["faces"][f]["couches"], s_m["faces"][f]["couches"])):
                if c_d.get("dist") is None:
                    a = axe_auto(d, s_d, f, j)
                    if a != _auto_moteur(d, c_d, j, f):
                        c_m["dist"] = a
    return out


def _face_figee(d: Dict, s: Dict, f: str) -> Dict:
    """La géométrie de la face par le moteur des dalles (``_face`` : Σ As, y_G), ses couches aux axes EFFECTIFS."""
    face = s["faces"][f]
    couches = [dict(c, dist=e) for c, e in zip(face["couches"], axes(d, s)[f])]
    return MO_D._face(couches, face.get("yG"), d["b"], d.get("enrobage", EN_D.DEFAUTS_DALLE["enrobage"]),
                      d.get("jeu_premier_lit", EN_D.DEFAUTS_DALLE["jeu_premier_lit"]))


def yG_calcule(d: Dict, s: Dict, f: str) -> float:
    """Le centre de gravité CALCULÉ de la face (cm), Σ As·e / Σ As sur les axes effectifs — ce que le champ « y_G » montre
    quand il est vide."""
    return _face_figee(d, s, f)["yG_calcule"]


def h_utile(d: Dict, s: Dict, f: str) -> float:
    """La hauteur utile de la face (cm) : h − y_G, le y_G imposé s'il l'est, sinon calculé sur les axes effectifs."""
    return float(d["h"]) - _face_figee(d, s, f)["yG"]


# ------------------------------------------------------------------ le résultat relu par le site
def annoter(r: Dict, d: Dict) -> Dict:
    """Le résultat du moteur (``moteur(d).verifier(pour_moteur(d))``) relu avec la notion d'automatique DU SITE, en place :
    pour chaque couche, ``auto`` = l'axe est automatique dans l'ÉTAT (le moteur, lui, a reçu un axe imposé pour une base
    empilée), ``e_auto`` = l'automatique du site (ce que le champ vide affiche : « auto · 5,5 »), ``e_auto_moteur`` =
    l'automatique du moteur (gardé), ``empilee`` = posée par le site sur la nappe principale ; une couche empilée reçoit
    aussi ``appui`` (le rang de la couche principale qui la porte, S5), ``sur`` (la même en mots : ``nom_appui``, ce que
    disent la formule et la pose) et ``sur_court`` (sa forme courte, ce que disent les légendes des coupes). Les formules
    « e » d'une prédalle reçoivent ``auto``, ``empilee`` et, empilée, ``appui``,
    ``sur`` et les termes ``e_p``, ``d_p``, ``demi_p`` (la couche principale qui porte : la base, ou le renfort plus
    haut qu'elle, S5), ``d_s``, ``demi_s`` (la couche posée, base ou renfort de la direction secondaire ; le terme
    « saisi » du moteur reste : la formule se lit encore sans rien d'autre). Rend ``r``."""
    for s_d, s_r in zip(d.get("sections") or [], r["sections"]):
        for f in FACES:
            for j, (c_d, c_r) in enumerate(zip(s_d["faces"][f]["couches"], s_r["faces"][f]["couches"])):
                c_r["e_auto_moteur"] = _auto_moteur(d, c_d, j, f)
                c_r["e_auto"] = axe_auto(d, s_d, f, j)
                c_r["auto"] = c_d.get("dist") is None
                c_r["empilee"] = c_r["auto"] and c_r["e_auto"] != c_r["e_auto_moteur"]
                if c_r["empilee"]:
                    c_r["appui"] = appui(d, s_d, f.split("_")[0])["couche"]
                    c_r["sur"] = nom_appui(d, s_d, f.split("_")[0])
                    c_r["sur_court"] = nom_appui(d, s_d, f.split("_")[0], court=True)
        for v in s_r["verifications"]:
            if v["cle"] not in FACES:
                continue
            for fo in v["formules"]:
                if fo.get("cle") != "e" or "couche" not in fo:
                    continue
                c_r = s_r["faces"][v["cle"]]["couches"][fo["couche"]]
                fo["auto"], fo["empilee"] = c_r["auto"], c_r["empilee"]
                if c_r["empilee"]:
                    a = appui(d, s_d, v["cle"].split("_")[0])
                    fo["appui"], fo["sur"] = c_r["appui"], c_r["sur"]
                    fo["termes"] = fo["termes"] + [
                        {"symbole": "e_p", "valeur": a["e"], "unite": "cm"},
                        {"symbole": "d_p", "valeur": a["d"], "unite": "mm"},
                        {"symbole": "demi_p", "valeur": a["demi"], "unite": "cm"},
                        {"symbole": "d_s", "valeur": c_r["d"], "unite": "mm"},
                        {"symbole": "demi_s", "valeur": _demi(s_d["faces"][v["cle"]]["couches"][fo["couche"]]), "unite": "cm"}]
    return r


# ------------------------------------------------------------------ le sélecteur de la nappe de base
def candidats_base(d: Dict, i: int, f: str, familles: Optional[Sequence[str]] = None) -> List[Dict]:
    """Le sélecteur de la nappe de BASE d'une face (``candidats_base`` du moteur du type), les niveaux du site respectés :
    le Ø d'un candidat déplace l'axe empilé — le sien (base secondaire) ou celui des couches secondaires posées sur lui
    (base principale : la base ET les renforts de la direction secondaire, S3 — tant qu'elle est la plus haute de son
    lit : sous un renfort principal plus haut, ce renfort les porte et un candidat plus mince ne les fait pas
    descendre, S5) —, que le moteur ne sait pas recalculer sur un axe imposé ; la dalle est donc recomposée candidat
    par candidat, et chacune calculée par le moteur (``verifier(pour_moteur(essai))``). Mêmes clés que le moteur (plus
    « e », l'axe de la couche), même tri."""
    MO = moteur(d)
    if f not in FACES:
        raise ValueError(f"face : {' '.join(FACES)} attendu (reçu {f!r})")
    sections = d.get("sections") or []
    if not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < len(sections):
        raise ValueError(f"section n° {i!r} introuvable ({len(sections)} section(s))")
    fams = tuple(x for x in (familles or MO_D.FAMILLES) if x in MO_D.FAMILLES) or MO_D.FAMILLES
    base = sections[i]["faces"][f]["couches"][0]
    out = []
    for cand in MO._liste_candidats(fams):
        essai = copy.deepcopy(d)
        c1 = essai["sections"][i]["faces"][f]["couches"][0]
        c1.update(cand)
        s = MO.verifier(pour_moteur(essai))["sections"][i]
        g, geo = s["grandeurs"], s["faces"][f]["couches"][0]
        retenue = base["type"] == cand["type"] and (base["treillis"] == cand["treillis"] if cand["type"] == "treillis"
                                                    else (base["d"], base["esp"]) == (cand["d"], cand["esp"]))
        out.append({"nom": MO_D.candidat_nom(c1), "famille": cand["type"], "type": cand["type"], "treillis": c1["treillis"],
                    "d": c1["d"], "esp": c1["esp"], "As_pm": geo["As_pm"], "As": g[f"As_{f}"], "e": geo["e"], "yG": g[f"yG_{f}"],
                    "taux": g[f"besoin_{f}"] / g[f"As_{f}"], "etat": g[f"etat_{f}"], "ok": g[f"etat_{f}"] == MO_D.OK,
                    "statut_section": s["statut"], "retenue": retenue, "optimal": False})
    out.sort(key=lambda x: (x["As_pm"], MO_D.FAMILLES.index(x["famille"]), x["d"]))
    optimal = next((x for x in out if x["ok"]), None)
    if optimal is not None:
        optimal["optimal"] = True
    return out

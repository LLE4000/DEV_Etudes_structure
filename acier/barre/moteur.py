# -*- coding: utf-8 -*-
"""LA BARRE VÉRIFIÉE par ses deux méthodes — l'historique du bureau (NBN B 51-001, charges de service) et
l'Eurocode 3 (ELU) —, depuis ses données d'écran (acier/barre/entree.py). docs/acier/BARRES_ACIER.md § 9.

DEUX MODES :

« efforts » (le mode des classeurs) — les efforts sont donnés : ceux de service vont à la méthode historique, ceux de
calcul à l'Eurocode. Une vérification par méthode.

« chargee » — la barre a ses appuis, ses cas de charge, ses maintiens ; le module fait la statique (statique.py) et les
combinaisons (combinaisons.py), puis cherche, vérification par vérification, l'occurrence la plus DÉFAVORABLE :

    Eurocode 3 : pour chaque combinaison (6.10) —
        la SECTION dans chaque section à vérifier (appuis, ruptures de chargement, extremum du moment de chaque
        travée), avec les efforts qui s'y trouvent ENSEMBLE, des deux côtés d'un saut ;
        la BARRE par tronçon de déversement : flambement sous N, déversement sous le diagramme réel du tronçon
        (M_cr numérique, charges à leur niveau), flexion composée avec le plus grand moment du tronçon.
    Méthode historique : pour chaque combinaison de service (G + ΣQ) — dans chaque section à vérifier et aux deux
        bouts de chaque tronçon, avec la longueur de déversement L_c de son tronçon (à une frontière : le côté gauche
        avec le tronçon de gauche, le côté droit avec celui de droite).

L'enveloppe garde, pour chaque vérification, où elle a été trouvée (la combinaison, l'abscisse ou le tronçon) et les
formules de CETTE occurrence : la note imprime ce qui a décidé.

Ce que la barre chargée ne fait pas (v1) : la flexion d'axe faible et les charges horizontales (le mode « efforts »
les prend) ; la flèche n'est donnée qu'à titre d'information.
"""
import math
from typing import Dict, List, Optional, Sequence, Tuple


from acier import materiaux as MA

from . import ame as AME
from . import combinaisons as CO
from . import comparaison as CP
from . import ec3 as EC
from . import ec3_deversement as DV
from . import ec3_section as ES
from . import entree as EN
from . import historique as HI
from . import mcr as MC
from . import parametres as PA
from . import prs as PR
from . import resultat as RE
from . import section as SE
from . import statique as ST
from .enveloppe import IDS_BARRE, ORDRE, Enveloppe          # noqa: F401 — ORDRE, relu par les tests

POINTS_DIAGRAMME = 40            # points par travée des enveloppes rendues pour le dessin
BALAYAGE_TRONCON = 24            # points par tronçon pour le plus grand moment
EPS = 1e-9
SAUT = 1e-9                      # l'écart relatif entre les lectures de gauche et de droite qui fait un saut de diagramme


def _refusee(methode: str, norme: str, message: str) -> Dict:
    """Le résultat d'une méthode qui REFUSE la barre (une épaisseur au-delà de 80 mm (domaine du module), une section sans courbe…) :
    « non couvert », avec le motif — l'autre méthode reste rendue."""
    J = RE.Journal(methode, norme)
    J.verif("E-SECTION" if methode == "ec3" else "H-SIGMA", "Vérification impossible", norme, "", None, None, "", [], statut=RE.NON_COUVERT,
            messages=[J.alerte("refus", "attention", message)])
    r = J.rendre()
    r.update({"combinaison": None, "combinaisons": [], "a_confirmer": [], "refus": message})     # (la comparaison le lit : comparaison.py)
    return r


def _fy_historique(nuance: str, s: Dict) -> Dict:
    """La limite élastique de la méthode ELS : la NOMINALE de la nuance (235, 275, 355 N/mm²), quelle que soit
    l'épaisseur, comme les classeurs du bureau (brief v12 § 6.1, § 13.2 point 5, § 18.3) — l'Eurocode 3 garde la table
    selon l'épaisseur. Au-delà de 80 mm, la méthode se calcule (« hors domaine », brief § 19.1 point 5). Le DOMAINE de
    la doc (AE235 à AE355) se lit sur la nuance : elle est passée avec f_y. {"fy", "nuance", "fy_source", "fy_ref"}."""
    return {"fy": MA.nominale(nuance), "nuance": nuance, "fy_source": HI.SOURCE_NOMINALE, "fy_ref": HI.FY_NOMINALE}


def _deversement_historique(b: Dict, Ld: Optional[float], bout_libre: bool = False, niveau: Optional[str] = None) -> Dict:
    """Le déversement de la méthode ELS : actif ou non, L_d (mm ; None : la longueur de l'axe faible), C_d, la variante
    (classeurs par défaut), le facteur 5,1 de la doc (profil fermé), une classe imposée (None : le tableau 12), le niveau
    des charges (le message de C_d > 1 sous une charge sur la semelle haute : ``niveau`` « haut » dès qu'UNE charge y est
    posée — lot C ; à défaut celui de la barre), et une console à bout libre (hors du domaine de la formule des
    classeurs : historique.verifier le dit)."""
    h = b["historique"]
    return {"actif": h["deversement"], "Lc": Ld, "Cd": h["Cd"], "variante": h["variante"], "facteur_doc": h["facteur_doc"],
            "classe": h["classe_d"], "niveau": niveau or b["niveau"], "bout_libre": bout_libre}


def _C_historique(mode: str, forme: str, psi: float) -> Dict:
    """Le coefficient C d'un axe : 1 (« unite », les classeurs) ou, selon la doc p. 20 et 21 (« norme ») : 0,85 sous des
    charges appliquées sur la barre, √(0,3·(1 + ψ²) + 0,4·ψ) sous des moments d'extrémité seuls."""
    if mode == "unite":
        return {"mode": "unite"}
    if forme in (DV.REPARTI, DV.PONCTUEL, DV.GENERAL):
        return {"mode": "charges"}
    return {"mode": "moments", "psi": 1.0 if forme == DV.CONSTANT else psi}


# une CONSOLE (encastrée – libre) : u = 2,1, la valeur RECOMMANDÉE du tableau d'aide (doc p. 16, brief § 13.3 ; 2 en
# théorie) — sur tout axe où son bout est libre, la longueur de flambement PROPOSÉE (champ L vide) est 2,1 × sa longueur ;
# u reste le multiplicateur saisi (1 par défaut : les notes enregistrées portent u = 1)
U_CONSOLE = 2.1
CONSOLE_LZ = "Console à bout libre : L_cr,z = 2,1·L, maintiens intermédiaires non comptés — À VÉRIFIER."


def _bout_libre(b: Dict, trs: Optional[List[Dict]]) -> bool:
    """Une console dont le bout n'est tenu par aucun maintien complet : son dernier tronçon finit LIBRE (statique.troncons)."""
    return b["modele"] == "console" and bool(trs) and trs[-1]["appuis"][1] == MC.LIBRE


def _longueurs(b: Dict, trs: Optional[List[Dict]]) -> Dict:
    """Les longueurs de flambement (mm) et leur ORIGINE (« portee », « troncon », « console », « saisie ») : celles
    données, sinon la plus grande travée (axe fort) et le plus long tronçon entre sections tenues (axe faible). Une
    CONSOLE flambe sur U_CONSOLE = 2,1 fois sa longueur (brief § 13.3, « encastrée – libre », valeur recommandée) autour
    de l'axe fort, et autour de l'axe faible tant que son bout est libre : il se déplace latéralement comme dans le plan.
    Un maintien complet au bout en fait une section tenue : la plus grande distance entre sections tenues (côté sûr). Des
    maintiens intermédiaires seuls ne raccourcissent rien : la longueur exacte est plus courte (3,77 m pour une console
    de 3 m tenue à mi-longueur, au lieu de 6,30) mais la partie libre ne se prend pas isolément sans justification —
    2,1·L, hypothèse prudente, À VÉRIFIER (CONSOLE_LZ : dite une fois, et seulement sous compression ; l'ingénieur peut
    saisir la longueur). Un DÉBORD (porte-à-faux d'une barre sur appuis, lot C) sur un axe où son bout est libre (y
    toujours ; z sauf maintien complet au bout) : la longueur d'Euler de la barre ENTIÈRE sur ses points tenus (``_L_debord``,
    origine « debord ») quand elle donne plus que les travées — l'appui n'est pas un encastrement : la travée voisine tourne
    avec le débord (4 m + 2,5 m : 7,89 m, et non 2,1 × 2,5 = 5,25 m ; contre-vérification du lot C)."""
    out = {}
    for a in ("y", "z"):
        L, origine = b["flambement"][a]["L"], "saisie"
        if L is None:
            if a == "y":
                L, origine = max(b["travees"]), "portee"
                if b["modele"] == "console":
                    L, origine = round(U_CONSOLE * L, 9), "console"        # une console flambe sur 2,1 fois sa longueur (§ 13.3)
            else:
                L = max(t["x2"] - t["x1"] for t in trs if MC.LIBRE not in t["appuis"]) if any(
                    MC.LIBRE not in t["appuis"] for t in trs) else 0.0
                origine = "portee" if abs(L - max(b["travees"])) < EPS else "troncon"
                if _bout_libre(b, trs):
                    L, origine = round(U_CONSOLE * sum(b["travees"]), 9), "console"    # bout libre : 2,1·L, même avec des maintiens
            Ld = _L_debord(b, trs, a)                        # un débord à bout libre : la barre entière, si elle l'emporte
            if Ld is not None and Ld > L + EPS:
                L, origine = Ld, "debord"
        out[a] = {"L": L * 1e3, "u": b["flambement"][a]["u"], "origine": origine}
    return out


def _debords_libres(b: Dict, trs: Optional[List[Dict]], a: str) -> List[float]:
    """Les débords (m) dont le bout est LIBRE pour le flambement d'axe ``a`` : y toujours (aucun appui au bout) ; z sauf
    maintien complet au bout (statique.troncons). Une console n'en a pas (sa règle : U_CONSOLE)."""
    if b["modele"] == "console" or not trs:
        return []
    g, d = (b.get("debords") or (0.0, 0.0))[:2]
    return [x for k, x in ((0, g), (-1, d)) if x > EPS and (a == "y" or trs[k]["appuis"][0 if k == 0 else 1] == MC.LIBRE)]


def _tenus(b: Dict, trs: List[Dict], a: str) -> Tuple[float, ...]:
    """Les abscisses tenues en déplacement pour le flambement d'axe ``a`` : y — les appuis ; z — les bouts des tronçons
    qui ne sont pas libres (les appuis en fourche ou encastrés, les maintiens complets)."""
    if a == "y":
        return tuple(ST.appuis_x(b))
    return tuple(sorted({t["x1"] for t in trs if t["appuis"][0] != MC.LIBRE} | {t["x2"] for t in trs if t["appuis"][1] != MC.LIBRE}))


def _L_debord(b: Dict, trs: Optional[List[Dict]], a: str) -> Optional[float]:
    """La longueur de flambement d'axe ``a`` (m) d'une barre dont un DÉBORD a son bout libre : la plus grande de la
    longueur d'Euler de la barre entière sur ses points tenus (statique.longueur_euler : E·I et N constants, l'appui
    articulé — la travée voisine tourne avec le débord) et de 2,1 × chaque débord (13.3, la console encastrée, valeur
    recommandée) ; None sans débord libre. Elle sert aussi de L_d au tronçon du débord dans la méthode ELS (axe z)."""
    libres = _debords_libres(b, trs, a)
    if not libres:
        return None
    return max([round(U_CONSOLE * x, 9) for x in libres] + [ST.longueur_euler(round(ST.longueur(b), 9), _tenus(b, trs, a))])


# un débord à bout libre : dit UNE fois (la méthode ELS, qui lit L_d ; à défaut l'Eurocode 3 comprimé)
MOT_DEBORD = "Débord libre : L_cr et L_d de la barre entière (Euler, E·I et N constants), déversement hors domaine — À VÉRIFIER."


def _avis_debord(b: Dict, trs: Optional[List[Dict]]) -> List[Dict]:
    """L'alerte À VÉRIFIER d'une barre à débord libre (``_L_debord``)."""
    if _debords_libres(b, trs, "y") or _debords_libres(b, trs, "z"):
        return [{"code": "debord_libre", "niveau": "attention", "message": MOT_DEBORD}]
    return []


def _avis_longueurs(b: Dict, trs: Optional[List[Dict]]) -> List[Dict]:
    """L'alerte À VÉRIFIER d'une console à bout libre dont L_cr,z automatique ignore des maintiens complets intermédiaires
    (_chargee la donne à la première méthode COMPRIMÉE : sans N > 0, L_cr,z ne change aucun résultat)."""
    if b["flambement"]["z"]["L"] is None and _bout_libre(b, trs) and len(trs) > 1:
        return [{"code": "console_Lz", "niveau": "attention", "message": CONSOLE_LZ}]
    return []


def _longueurs_rendues(b: Dict, axes: Dict) -> Dict:
    """Les longueurs de flambement PRISES, en mètres : {"y": {"L", "u", "auto", "origine"}, "z": …} ; « auto » quand la
    barre ne la donne pas (l'écran l'écrit dans le champ vide) ; « origine » : « portee » (la plus grande travée),
    « troncon » (le plus long tronçon entre sections tenues), « console » (2,1 × la longueur d'une console, brief § 13.3),
    « debord » (la longueur d'Euler de la barre entière avec son débord libre, ``_L_debord``), « saisie » (lot C, C1 point 14 : l'écran écrit « auto · 6,3 », la note « L_cr,z = 2,1 × 3,00 m »)."""
    return {a: {"L": axes[a]["L"] / 1e3, "u": axes[a]["u"], "auto": b["flambement"][a]["L"] is None, "origine": axes[a]["origine"]}
            for a in ("y", "z")}


def _axes_historique(b: Dict, axes: Dict) -> Dict:
    """Les axes de la méthode ELS : ceux de la barre, avec la COURBE du fascicule imposée par axe (``flambement.{y,z}.courbe``
    ; None : celle de la norme — un profil creux formé à froid de plus de 6 mm reste « hors domaine », historique.domaine).
    L'Eurocode 3 garde la courbe de son tableau 6.2 : la courbe saisie est celle du classeur."""
    return {a: dict(axes[a], courbe=b["flambement"][a].get("courbe")) for a in ("y", "z")}


def poids_propre_imposes(position: str, q: float, L: float, angle: Optional[float] = None) -> Dict:
    """Le poids propre d'une barre aux EFFORTS IMPOSÉS, selon sa POSITION (brief § 4.2, § 6.2 point 7) — fonction pure :
    q (kN/m) sur L (m) ; {"N" (kN, compression), "M" (kN·m), "V" (kN)} : horizontale, en flexion (q·L²/8, q·L/2) ;
    verticale, dans N (q·L) ; inclinée d'un angle α sur l'horizontale, q·cos α en flexion et q·L·sin α dans N. Les
    interrupteurs de la barre (``poids_propre_efforts`` : « dans P », « en flexion ») choisissent ce qui est ajouté."""
    fN, fM = ST.repartition(position, angle)
    return {"N": fN * q * L, "M": fM * q * L * L / 8.0, "V": fM * q * L / 2.0}


# le poids propre ajouté aux efforts saisis : la note le dit en une ligne (À VÉRIFIER : compté deux fois s'ils le contiennent)
MOT_POIDS_PROPRE_IMPOSES = "Poids propre ajouté aux efforts saisis (× γ_G,sup à l'ELU) : compté deux fois s'ils le contiennent (À VÉRIFIER)."


def _efforts_avec_poids(b: Dict, s: Dict, gamma: float) -> Tuple[Dict, Dict, bool]:
    """(efforts de service, efforts de calcul, poids propre ajouté ?) : ceux saisis, plus le poids propre que les interrupteurs
    demandent (``poids_propre_imposes`` ; longueur : la plus grande des deux longueurs de flambement, À VÉRIFIER) — tel quel
    en service, × γ_G,sup à l'ELU ; en flexion, il augmente |M| (son sens n'est pas connu : prudent)."""
    sv, el = dict(b["efforts"]["service"]), dict(b["efforts"]["elu"])
    sw = b.get("poids_propre_efforts") or {}
    if not (sw.get("N") or sw.get("M")):
        return sv, el, False
    # les INTERRUPTEURS décident, comme le classeur : « dans P » → P += q·L (la part d'un poteau) ; « en flexion » → M += q·L²/8,
    # V += q·L/2 (celle d'une poutre) — la barre aux efforts imposés ne garde pas sa position (entree.py : position et angle sont des
    # clés de la barre chargée)
    L = max(b["flambement"]["y"]["L"], b["flambement"]["z"]["L"])
    g = ST.poids_propre(s)
    dans_P, flexion = poids_propre_imposes("verticale", g, L), poids_propre_imposes("horizontale", g, L)
    for e, f in ((sv, 1.0), (el, gamma)):
        if sw.get("N"):
            e["N"] += f * dans_P["N"]
        if sw.get("M"):
            e["My"] = math.copysign(abs(e["My"]) + f * flexion["M"], e["My"] or 1.0)
            e["Vz"] = math.copysign(abs(e["Vz"]) + f * flexion["V"], e["Vz"] or 1.0)
    return sv, el, True


def section_de(b: Dict) -> Dict:
    """La section de la barre relue : un PRS (ses plats, sa nuance — acier/barre/prs.py) ou un profilé du catalogue."""
    if b.get("prs"):
        return PR.section_prs(**b["prs"], nuance=b["nuance"])
    return SE.du_catalogue(b["profil"])


# ------------------------------------------------------------------ efforts imposés
def _efforts_imposes(b: Dict, s: Dict) -> Dict:
    axes = _longueurs(b, None)
    Lc = b["Lc"]
    methodes = {}
    P = PA.jeu(b["parametres"])
    service, elu, avec_pp = _efforts_avec_poids(b, s, P["gamma_G_sup"])     # (lot C) le poids propre des interrupteurs

    def historique() -> Dict:
        # (un PRS aux bords des semelles non saisis : historique.verifier bloque le verdict et calcule pour information en larges
        # plats — la seule règle ; intégration du lot B, point 20)
        e, h = service, b["historique"]
        return HI.verifier({"section": s, **_fy_historique(b["nuance"], s), "s": h["s"], "P": e["N"] * 1e3, "My": e["My"] * 1e6, "Mz": e["Mz"] * 1e6,
                         "Vz": e["Vz"] * 1e3, "Vy": e["Vy"] * 1e3, "axes": _axes_historique(b, axes), "tolerance": h["tolerance"],
                         "C": {a: _C_historique(h["C"], b["diagramme"][a]["forme"], b["diagramme"][a]["psi"]) for a in ("y", "z")},
                         "deversement": _deversement_historique(b, None if Lc is None else Lc * 1e3)})

    def ec3() -> Dict:
        e = elu
        r = EC.verifier({"section": s, "nuance": b["nuance"], "NEd": e["N"] * 1e3, "MyEd": e["My"] * 1e6, "MzEd": e["Mz"] * 1e6,
                         "VzEd": e["Vz"] * 1e3, "VyEd": e["Vy"] * 1e3, "axes": axes, "parametres": b["parametres"], "voilement": False,
                         "flexion": {a: {"diagramme": b["diagramme"][a]["forme"], "psi": b["diagramme"][a]["psi"]} for a in ("y", "z")},
                         "deversement": {"Lc": None if Lc is None else Lc * 1e3, "niveau": b["niveau"]}, "ouvrage": b["ouvrage"]})
        r["a_confirmer"] = r["grandeurs"].get("a_confirmer", [])
        return r

    for nom, calcul, norme in (("historique", historique, HI.NORME), ("ec3", ec3, EC.NORME)):     # TOUJOURS les deux (brief § 0, § 9)
        try:
            methodes[nom] = calcul()
            if avec_pp:
                methodes[nom]["hypotheses"].append(MOT_POIDS_PROPRE_IMPOSES)
        except RE.Refus as e:
            methodes[nom] = _refusee(nom, norme, str(e))
    return {"statique": None, "methodes": methodes, "ame": AME.efforts(dict(b, efforts={"service": service, "elu": elu}), s),
            "longueurs": _longueurs_rendues(b, axes)}


# ------------------------------------------------------------------ barre chargée
def _moment_max(comb: ST.Combinaison, tr: Dict, xs: Sequence[float]) -> float:
    """Le plus grand |M| de la combinaison sur le tronçon (kN·m) : balayage et sections à vérifier."""
    return abs(_moment_extreme(comb, tr, xs)[0])


def _moment_extreme(comb: ST.Combinaison, tr: Dict, xs: Sequence[float]) -> Tuple[float, float]:
    """(le moment de plus grande valeur absolue du tronçon, AVEC son signe (kN·m ; M > 0 comprime la semelle haute), son
    abscisse (m)) — la semelle comprimée du tronçon s'y lit (``semelle_comprimee``)."""
    a, b = tr["x1"], tr["x2"]
    pts = [a + (b - a) * i / BALAYAGE_TRONCON for i in range(BALAYAGE_TRONCON + 1)] + [x for x in xs if a < x < b]
    m, xm = 0.0, a
    for x in pts:
        e = comb.lire(x)
        for v in ((e["M_g"],) if x > a + EPS else ()) + ((e["M_d"],) if x < b - EPS else ()):
            if abs(v) > abs(m):
                m, xm = v, x
    return m, xm


def plans_moyens(s: Dict) -> Tuple[float, float]:
    """Les cotes des plans moyens des semelles haute et basse, depuis le centre de GRAVITÉ (mm, z vers le haut)."""
    if s.get("prs"):
        p = s["prs"]
        return s["h"] - p["t1"] / 2.0 - s["z_g"], p["t2"] / 2.0 - s["z_g"]
    demi = ((s.get("h") or 0.0) - (s.get("tf") or s.get("t") or 0.0)) / 2.0
    return demi, -demi


def semelle_comprimee(s: Dict, N: float, M: float) -> Tuple[str, bool]:
    """(la semelle comprimée — « haut » ou « bas » —, est-elle vraiment comprimée ?) sous N (kN, > 0 en compression) et M
    (kN·m) : celle dont σ_x = N/A + M·z/I_y, au plan moyen de la semelle, est la plus comprimée (brief § 18.2, § 8.2) ;
    aucune comprimée (une traction qui l'emporte) : le signe de M (prudent, À VÉRIFIER — MOT_SEMELLE_TRACTION)."""
    zh, zb = plans_moyens(s)
    sh = N * 1e3 / s["A"] + M * 1e6 * zh / s["Iy"]
    sb = N * 1e3 / s["A"] + M * 1e6 * zb / s["Iy"]
    if max(sh, sb) > 0.0:
        return ("haut" if sh >= sb else "bas"), True
    return ("haut" if M >= 0.0 else "bas"), False


MOT_SEMELLE_TRACTION = "Aucune semelle comprimée (traction) : la semelle comprimée suit le signe de M (À VÉRIFIER)."
MOT_POIDS_PROPRE = "Poids propre au centre de gravité."
NIVEAUX_MOTS = {"haut": "sur la semelle haute", "centre": "au centre de cisaillement", "bas": "sous la semelle basse"}
NIVEAUX_COURTS = {"haut": "semelle haute", "centre": "centre de cisaillement", "bas": "semelle basse"}


def _kc_tableau(b: Dict, cas: Sequence[Dict], facteurs: Sequence[float], tr: Dict, C: Optional[ST.Combinaison] = None) -> Optional[float]:
    """k_c du tableau 6.6 quand le tronçon EST un de ses schémas : un tronçon entre deux fourches, sans maintien intérieur
    ni charge dedans — moment linéaire (ec3_deversement.kc_lineaire, ψ lu sur la combinaison ``C``) ; ou une portée seule
    (deux appuis simples, deux encastrements, un encastrement et un appui), tronçon = la portée, sous des charges uniformes
    sur toute la portée ou des charges à mi-portée seulement (ec3_deversement.kc_portee), sans débord. Sinon None (k_c = 1)."""
    L = sum(b["travees"])
    charges = [ch for f, c in zip(facteurs, cas) if f for ch in c.get("charges") or []]
    if C is not None and not tr["maintiens"] and tuple(tr["appuis"]) == (MC.FOURCHE, MC.FOURCHE):
        k = DV.kc_lineaire(charges, tr["x1"], tr["x2"], C.lire(tr["x1"])["M_d"], C.lire(tr["x2"])["M_g"])
        if k is not None:
            return k
    if b["modele"] not in DV.MODELES_KC or tr["maintiens"] or tr["x2"] - tr["x1"] < L - EPS or any(b.get("debords") or ()):
        return None
    return DV.kc_portee(b["modele"], charges, L)


def _Lc_historique(comb: ST.Combinaison, tr: Dict, xs: Sequence[float], impose: Optional[float] = None) -> float:
    """La longueur de déversement de la méthode historique sur le tronçon (m) : la plus grande distance entre deux
    points où la semelle COMPRIMÉE est tenue. Un maintien d'une seule semelle ne compte que si elle est comprimée sur
    tout le tronçon (moment d'un seul signe). Un tronçon à bout LIBRE (``impose``, m — ``_libres``) : le bout libre n'est
    pas une section tenue — la longueur de flambement de son bout libre, maintiens intermédiaires non comptés : hypothèse
    prudente, À VÉRIFIER (intégration du lot B : le bout libre comptait comme un maintien, L_d = la longueur de la console)."""
    if impose is not None:
        return impose
    a, b = tr["x1"], tr["x2"]
    pts = [a + (b - a) * i / BALAYAGE_TRONCON for i in range(1, BALAYAGE_TRONCON)]
    Ms = [comb.M(x) for x in pts]
    haut, bas = all(m >= -EPS for m in Ms), all(m <= EPS for m in Ms)
    tenus = [a, b]
    for x_mm, z in tr["maintiens"]:
        if (z > 0 and haut) or (z < 0 and bas):
            tenus.append(a + x_mm / 1e3)
    tenus.sort()
    return max(v - u for u, v in zip(tenus, tenus[1:]))


def _cotes_historique(e: Dict, x: float, Lcs: Sequence[Tuple[Dict, float]]) -> List[Tuple[Tuple[Dict, float], Tuple[float, float]]]:
    """Les couples (M, V) de la section lue ``e`` (abscisse ``x``), chacun avec le tronçon (et son L_d) qui le vérifie dans
    la méthode ELS. À la FRONTIÈRE de deux tronçons, le côté gauche se vérifie avec le tronçon de gauche et le côté droit
    avec celui de droite, même quand le moment est le même des deux côtés : leurs L_d diffèrent (sinon le taux dépend du
    sens de dessin de la poutre). Ailleurs, le tronçon de la section, pour ses deux côtés (``ST.cotes``)."""
    gauche = next((t for t in Lcs if t[0]["x1"] + EPS < x <= t[0]["x2"] + EPS), None)
    droite = next((t for t in Lcs if t[0]["x1"] - EPS <= x < t[0]["x2"] - EPS), None)
    if gauche is not None and droite is not None and gauche is not droite:
        return [(gauche, (e["M_g"], e["V_g"])), (droite, (e["M_d"], e["V_d"]))]
    t = gauche or droite or Lcs[-1]
    return [(t, mv) for mv in ST.cotes(e, EPS)]


def _segments(b: Dict) -> List[Tuple[float, float]]:
    """Les morceaux de la barre (m) : le débord gauche, chaque travée, le débord droit."""
    g, d = (b.get("debords") or (0.0, 0.0))[:2]
    segs = [(0.0, g)] if g > EPS else []
    x = g
    for Lt in b["travees"]:
        segs.append((x, x + Lt))
        x += Lt
    if d > EPS:
        segs.append((x, x + d))
    return segs


def _travee_de(b: Dict, tr: Dict) -> Optional[Tuple[float, float]]:
    """La TRAVÉE (entre deux appuis) qui porte le tronçon — celle de la formule générale de C_mi,0, la travée extraite
    du brief § 7.8 ; None pour un débord ou une console."""
    if b["modele"] == "console":
        return None
    ax = ST.appuis_x(b)
    milieu = 0.5 * (tr["x1"] + tr["x2"])
    for u, v in zip(ax, ax[1:]):
        if u - EPS <= milieu <= v + EPS:
            return u, v
    return None


def _niveaux_charges(b: Dict) -> List[str]:
    """Les niveaux d'application des charges SAISIES (le poids propre à part : au centre de gravité), dans l'ordre haut,
    centre, bas."""
    vus = {ch.get("niveau") or b["niveau"] for c in b["cas"] for ch in c.get("charges") or [] if ch.get("type") != "C" and ch.get("v")}
    return [n for n in ("haut", "centre", "bas") if n in vus] or [b["niveau"]]


def _libres(b: Dict, trs: List[Dict]) -> Dict[int, float]:
    """Les tronçons d'un bout LIBRE — une console, un débord — et leur L_d imposée (m), maintiens intermédiaires non
    comptés : une console, 2,1 × sa longueur (13.3) ; un débord, la longueur de flambement d'axe z de la barre entière
    (``_L_debord`` : l'appui n'est pas un encastrement). À VÉRIFIER (hors du domaine de la formule des classeurs)."""
    out: Dict[int, float] = {}
    if not trs:
        return out
    if _bout_libre(b, trs):
        return {i: round(U_CONSOLE * sum(b["travees"]), 9) for i in range(len(trs))}
    g, d = (b.get("debords") or (0.0, 0.0))[:2]
    Lb = ST.longueur(b)
    Ld = _L_debord(b, trs, "z")
    for i, tr in enumerate(trs):
        if g > EPS and trs[0]["appuis"][0] == MC.LIBRE and tr["x2"] <= g + EPS:
            out[i] = Ld
        if d > EPS and trs[-1]["appuis"][1] == MC.LIBRE and tr["x1"] >= Lb - d - EPS:
            out[i] = Ld
    return out


def _chargee(b: Dict, s: Dict) -> Dict:
    P = PA.jeu(b["parametres"])
    cas = ST.cas_complets(b, s)
    EI = P["E"] * s["Iy"] / 1e9                                # kN·m²
    sols = ST.resoudre(b, cas, EI)
    N_cas = [ST.N_du_cas(c) for c in cas]
    pts = ST.ruptures(sols)
    trs = ST.troncons(b, s)                                    # flambement, L_d de la méthode ELS : TOUS les maintiens
    trs_mcr = ST.troncons(b, s, pour_mcr=True)                 # M_cr : sans les maintiens « négligés dans M_cr » (brief § 7.7)
    axes = _longueurs(b, trs)
    zg = SE.cote(s, b["niveau"])                               # (la cote de la barre : celle d'une charge sans niveau)

    def zg_de(ch: Dict) -> float:
        """La cote d'une charge (mm, au-dessus du centre de cisaillement) : SON niveau, à défaut celui de la barre ; le poids
        propre au centre de GRAVITÉ (z_g = −z_s ; 0 pour un I bisymétrique)."""
        return ST.cote_charge(b, s, ch)

    Lt = ST.longueur(b)
    g_, d_ = (b.get("debords") or (0.0, 0.0))[:2]
    bouts = set(ST.appuis_x(b)) | ({0.0} if g_ <= EPS else set()) | ({round(Lt, 9)} if d_ <= EPS else set())
    grille = sorted({round(x, 9) for x in pts} | {round(x0 + (x1 - x0) * i / POINTS_DIAGRAMME, 9) for x0, x1 in _segments(b)
                                                  for i in range(POINTS_DIAGRAMME + 1)})
    statique: Dict = {"L": Lt, "x": grille, "troncons": [{"x1": t["x1"], "x2": t["x2"], "appuis": list(t["appuis"]),
                                                           "maintiens": [[t["x1"] + x / 1e3, z] for x, z in t["maintiens"]]} for t in trs],
                      "cas": [{"nom": c["nom"], "nature": c["nature"], "N": c.get("N", 0.0),
                               "reactions": [{"nom": a["nom"], "x": a["x"], "R": a["R"], "M": a.get("M", 0.0)} for a in sol.appuis]}
                              for c, sol in zip(cas, sols)], "enveloppes": {}}
    methodes = {}
    comprimee: Dict[str, bool] = {}          # une combinaison de la méthode a-t-elle N > 0 ? (alors elle lit L_cr,z)
    racines: set = set()
    negliges = [m for m in b.get("maintiens") or [] if m.get("neglige")]
    # la variante (0,4 ; 0,75) de l'ANB:2018, 6.3.2.3(1) : demandée quand la barre a des maintiens, TOUS négligés dans M_cr ;
    # ec3_deversement.couple_anb l'admet ou la refuse (hors bâtiment, maintiens comptés…) et le dit
    variante = bool(b.get("maintiens")) and len(negliges) == len(b["maintiens"])
    niveaux = _niveaux_charges(b)
    traction = [False]

    def enveloppe_diagrammes(nom: str, combs: List[ST.Combinaison]) -> None:
        """Les enveloppes de M et de V, avec LEURS abscisses : là où un diagramme saute (un appui, une charge ponctuelle,
        un couple), l'abscisse y est deux fois — la lecture de gauche, puis celle de droite — pour que le saut se trace
        vertical ; aux deux bouts de la barre, seule la lecture du côté de la barre compte (dehors, il n'y a rien)."""
        xs, M_max, M_min, V_max, V_min = [], [], [], [], []
        for x in grille:
            es = [c.lire(x) for c in combs]
            cotes = ("d",) if x <= EPS else (("g",) if x >= Lt - EPS else ("g", "d"))
            if len(cotes) == 2 and all(abs(e[k + "_g"] - e[k + "_d"]) <= SAUT * max(1.0, abs(e[k + "_g"]), abs(e[k + "_d"]))
                                       for e in es for k in ("M", "V")):
                cotes = ("d",)                                       # diagrammes continus ici : un seul point
            for cote in cotes:
                xs.append(x)
                M_max.append(max(e["M_" + cote] for e in es))
                M_min.append(min(e["M_" + cote] for e in es))
                V_max.append(max(e["V_" + cote] for e in es))
                V_min.append(min(e["V_" + cote] for e in es))
        statique["enveloppes"][nom] = {"x": xs, "M_max": M_max, "M_min": M_min, "V_max": V_max, "V_min": V_min}

    def ec3() -> Dict:
        env = Enveloppe(EC.METHODE, EC.NORME)
        combs = []
        for co in CO.elu(cas, P, b["combinaisons"]):
            C = ST.Combinaison(co["nom"], sols, co["facteurs"], N_cas)
            combs.append(C)
            xs = ST.sections(b, C, pts, racines)
            for x in xs:                                         # la SECTION, là où les efforts sont ensemble (N de la section)
                for M, V, N in ST.cotes_n(C, x, EPS):
                    if abs(M) < EPS and abs(V) < EPS and abs(N) < EPS:
                        continue
                    r = ES.verifier({"section": s, "nuance": b["nuance"], "NEd": N * 1e3, "MyEd": M * 1e6, "VzEd": V * 1e3,
                                     "parametres": b["parametres"], "voilement": False})
                    env.verser(r, {"combinaison": co["nom"], "x": x})
            for i, tr in enumerate(trs_mcr):                     # la BARRE, tronçon par tronçon (ceux de M_cr)
                Mmax = _moment_max(C, tr, xs)
                if Mmax < EPS and C.N <= EPS:
                    continue
                rep, pon = ST.charges_du_troncon(cas, co["facteurs"], tr, zg_de)
                a = tr["x1"]
                troncon = {"L": (tr["x2"] - a) * 1e3, "moment": (lambda x, C=C, a=a: C.M(a + x / 1e3) * 1e6), "reparties": rep,
                           "ponctuelles": pon, "appuis": tr["appuis"], "maintiens": tr["maintiens"], "M_max": Mmax * 1e6,
                           "entre_appuis": round(tr["x1"], 9) in bouts and round(tr["x2"], 9) in bouts}   # (la variante du couple le lit)
                flexion = {"diagramme": DV.CONSTANT}
                tv = _travee_de(b, tr)
                if tv is not None and Mmax > EPS:
                    # tableau A.2 (D.1 ANB), formule générale : la flèche et la portée de la TRAVÉE du tronçon (la travée extraite,
                    # brief § 7.8) ; le M est celui du tronçon, jamais plus grand que celui de la travée : C_mi,0 du côté sûr
                    fl = max(abs(C.lire(x)["f"] or 0.0) for x in grille if tv[0] - EPS <= x <= tv[1] + EPS)
                    flexion = {"diagramme": DV.GENERAL, "delta": fl * 1e3, "L": (tv[1] - tv[0]) * 1e3}
                kc = _kc_tableau(b, cas, co["facteurs"], tr, C)
                zs = {c_[-1] for c_ in (*rep, *pon)}
                dev = {"troncon": troncon, "kc": kc, "zg": zs.pop() if len(zs) == 1 else zg}
                dans = sum(1 for m in negliges if tr["x1"] - EPS <= m["x"] <= tr["x2"] + EPS)
                if variante:
                    dev.update(couple="variante", maintiens_negliges=dans)
                if PR.est_prs(s) and b["modele"] == "appuis" and kc is not None:
                    # le diagramme d'un cas des tableaux (annexe E ANB) : la FORME des charges de la portée, lue sur elles
                    # (intégration du lot B : déduite de k_c par une égalité de flottants)
                    dev["diagramme"] = DV.forme_portee([ch for f_, c_ in zip(co["facteurs"], cas) if f_ for ch in c_.get("charges") or []],
                                                       sum(b["travees"]))
                M_ex, x_ex = _moment_extreme(C, tr, xs)
                semelle, vraie = semelle_comprimee(s, C.N_en(x_ex), M_ex)
                traction[0] = traction[0] or (not vraie and PR.monosymetrique(s))
                barre_ec3 = {"section": s, "nuance": b["nuance"], "NEd": C.N * 1e3, "MyEd": Mmax * 1e6, "axes": axes,
                             "flexion": {"y": flexion}, "parametres": b["parametres"], "voilement": False, "console": b["modele"] == "console",
                             "deversement": dev, "ouvrage": b["ouvrage"]}
                if PR.monosymetrique(s) and semelle == "bas":
                    barre_ec3["MyEd"] = -barre_ec3["MyEd"]       # la semelle BASSE est comprimée (PRS monosymétrique)
                r = EC.verifier(barre_ec3)
                env.verser(r, {"combinaison": co["nom"], "troncon": [tr["x1"], tr["x2"]]}, prefixes=IDS_BARRE)
        enveloppe_diagrammes("ec3", combs)
        comprimee["ec3"] = any(C.N > EPS for C in combs)
        r = env.rendre()
        enc = [f"appui {chr(65 + i)}" for i, _ in enumerate(ST.appuis_x(b)) if b["modele"] != "console"
               and ST.torsion_appui(b, i) == MC.ENCASTRE]
        appuis_t = ("appuis en fourche (déplacement latéral et torsion empêchés)" if not enc else
                    f"{', '.join(enc)} encastré{'s' if len(enc) > 1 else ''} en torsion, les autres appuis en fourche")
        if len(niveaux) == 1:
            r["hypotheses"].append(f"Charges appliquées {NIVEAUX_MOTS[niveaux[0]]} ; {appuis_t}.")
        else:                                    # une hypothèse par fait (120 caractères au plus)
            r["hypotheses"].append("Charges appliquées à plusieurs hauteurs : " + ", ".join(NIVEAUX_COURTS[n] for n in niveaux) + ".")
            r["hypotheses"].append(appuis_t[:1].upper() + appuis_t[1:] + ".")
        if any(ch.get("niveau") == ST.GRAVITE for c in cas for ch in c.get("charges") or []):
            r["hypotheses"].append(MOT_POIDS_PROPRE)
        if traction[0]:
            r["hypotheses"].append(MOT_SEMELLE_TRACTION)
        return r

    def historique() -> Dict:
        env = Enveloppe(HI.METHODE, HI.NORME)
        h = b["historique"]
        fy = _fy_historique(b["nuance"], s)
        libres = _libres(b, trs)                         # une console, un débord : le tronçon du bout libre et sa longueur (m)
        axes_h = _axes_historique(b, axes)
        combs = []
        fleche = {"valeur": 0.0, "x": None, "combinaison": None}
        hors_libre = []                                  # un tronçon à bout libre hors domaine (H-DEV) : le verdict le garde
        # le message de C_d > 1 sous une charge sur la semelle haute (brief § 6.4) : dès qu'UNE charge y est posée
        niveau_els = "haut" if "haut" in niveaux else niveaux[0]
        for co in CO.service(cas, b["combinaisons"]):
            C = ST.Combinaison(co["nom"], sols, co["facteurs"], N_cas)
            combs.append(C)
            xs = ST.sections(b, C, pts)
            Lcs = [(tr, _Lc_historique(C, tr, xs, libres.get(i))) for i, tr in enumerate(trs)]
            # les BOUTS de chaque tronçon sont des sections vérifiées : un maintien qui n'est ni un appui ni un point de
            # charge borne deux tronçons, et le plus grand moment d'un tronçon peut tomber à son bout
            for x in sorted(set(xs) | {tr[k] for tr in trs for k in ("x1", "x2")}):
                for (tr, Lc), (M, V) in _cotes_historique(C.lire(x), x, Lcs):
                    if abs(M) < EPS and abs(V) < EPS and abs(C.N) < EPS:
                        continue
                    semelle, _ = semelle_comprimee(s, C.N, M)
                    dv = dict(_deversement_historique(b, Lc * 1e3, trs.index(tr) in libres, niveau_els), semelle=semelle)
                    r = HI.verifier({"section": s, **fy, "s": h["s"], "P": C.N * 1e3, "My": M * 1e6, "Vz": V * 1e3, "axes": axes_h,
                                     "C": {"y": _C_historique(h["C"], DV.GENERAL, 1.0)}, "tolerance": h["tolerance"],
                                     "deversement": dv})
                    env.verser(r, {"combinaison": co["nom"], "x": x, "troncon": [tr["x1"], tr["x2"]]})
                    if trs.index(tr) in libres and any(v["id"] == "H-DEV" and v["statut"] == RE.HORS_DOMAINE for v in r["verifications"]):
                        hors_libre.append(tr)
            for x in grille:                                     # la flèche de service, à titre d'information
                f = C.lire(x)["f"]
                if f is not None and abs(f) > abs(fleche["valeur"]):
                    fleche = {"valeur": f, "x": x, "combinaison": co["nom"]}
        enveloppe_diagrammes("historique", combs)
        comprimee["historique"] = any(C.N > EPS for C in combs)
        portee = max([max(b["travees"]) * (2.0 if b["modele"] == "console" else 1.0)] + [2.0 * x for x in (g_, d_) if x > EPS])
        fleche["valeur"] *= 1e3                                  # mm
        fleche["rapport"] = portee * 1e3 / abs(fleche["valeur"]) if fleche["valeur"] else None
        statique["fleche"] = fleche
        res = env.rendre()
        if hors_libre and res["statut"] == RE.OK:
            # (contre-vérification du lot C) le déversement d'un débord hors domaine ne se perd pas quand la travée gouverne
            res["statut"] = RE.HORS_DOMAINE
        return res

    avis = _avis_longueurs(b, trs)               # une donnée commune : dite UNE fois, sous la première méthode qui la lit
    avis_d = _avis_debord(b, trs)                # (la méthode ELS lit L_d du débord ; l'Eurocode 3, L_cr sous compression)
    for nom, calcul, norme in (("historique", historique, HI.NORME), ("ec3", ec3, EC.NORME)):     # TOUJOURS les deux (brief § 0, § 9)
        try:
            methodes[nom] = calcul()
            if comprimee[nom]:
                methodes[nom]["alertes"] += avis
                avis = []                        # (l'autre méthode lit la même longueur : pas de redite)
            if avis_d and (nom == "historique" or comprimee[nom]):
                methodes[nom]["alertes"] += avis_d
                avis_d = []
        except RE.Refus as e:
            methodes[nom] = _refusee(nom, norme, str(e))
    # les sections critiques pour le dessin : les points demandés et les racines du moment (combinaisons ELU)
    statique["sections"] = sorted([{"x": x, "genre": "point"} for x in b.get("points") or []]
                                  + [{"x": x, "genre": "racine"} for x in sorted(racines)], key=lambda p_: p_["x"])
    return {"statique": statique, "methodes": methodes, "ame": AME.chargee(b, s, cas, sols, N_cas, pts),
            "longueurs": _longueurs_rendues(b, axes)}


def verifier(barre: Dict) -> Dict:
    """La barre vérifiée : {"barre" (relue), "profil", "masse" (kg/m), "poids_propre" (kN/m : celui que la barre chargée
    compte — acier/barre/statique.poids_propre ; l'écran le lit, il ne le recalcule pas), "statique" (None en efforts imposés),
    "longueurs" (de flambement, celles qui sont prises), "methodes": {"historique": …, "ec3": …} (la forme de
    acier/barre/resultat.py ; en barre chargée, chaque vérification dit sa combinaison et son lieu), "ame" (la même
    forme : l'âme selon l'EN 1993-1-5, toujours calculée, avec la décision de chaque raidisseur — acier/barre/ame.py),
    "statut" (le plus défavorable des deux méthodes et de l'âme : nok, bloqué, non couvert, hors domaine, ok),
    "comparaison" (acier/barre/comparaison.py)}. Les DEUX méthodes sont toujours calculées (brief § 0 et § 9) :
    ``barre["methodes"]`` (les anciennes notes) ne décide plus du calcul ; ce qu'une note IMPRIME est la clé de la note
    ``impression`` (api_web/profil.py)."""
    b = EN.normaliser(barre)
    s = section_de(b)
    out = _efforts_imposes(b, s) if b["mode"] == "efforts" else _chargee(b, s)
    ec = out["methodes"].get("ec3")
    if ec and s.get("forme") == "I" and not AME.AM.est_soude(s):       # brief § 8.3, point 8 : un laminé dont le CISAILLEMENT
        # gouverne — V_pl,Rd ne passe pas (le remède dit), ou E-Vz a le plus fort taux de l'Eurocode 3 et de l'âme (À VÉRIFIER :
        # « gouverne » lu ainsi) — un raidisseur n'y peut rien ; sans force concentrée sur l'âme (aucun lieu), pas de question de
        # raidisseur : rien n'est dit tant qu'il passe
        autres = [x["taux"] for r_ in (ec, out["ame"]) for x in r_["verifications"] if x["id"] != "E-Vz" and x["taux"] is not None]
        for v in ec["verifications"]:
            if v["id"] == "E-Vz" and v["statut"] == RE.NOK:
                v["messages"].append(AME.AM.MOT_CISAILLEMENT)
            elif v["id"] == "E-Vz" and out["ame"]["lieux"] and v["taux"] is not None and v["taux"] >= max(autres, default=0.0):
                v["messages"].append(AME.AM.MOT_CISAILLEMENT_GOUVERNE)
    statuts = [r["statut"] for r in out["methodes"].values()] + [out["ame"]["statut"]]
    statut = RE.pire_statut(statuts)
    out.update({"barre": b, "profil": s["nom"], "masse": s.get("masse"), "poids_propre": ST.poids_propre(s), "statut": statut,
                "comparaison": CP.comparer(out["methodes"], out["ame"])})
    return out

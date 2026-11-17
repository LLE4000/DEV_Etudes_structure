# -*- coding: utf-8 -*-
"""Le DÉVERSEMENT de la méthode ELS (brief v12 § 6.4, § 18.6) : ce que les classeurs « Déversement » du bureau calculent,
avec leurs valeurs, et la variante du fascicule (doc p. 19). Fonctions pures, N et mm ; la vérification qui les emploie
est acier/barre/historique.py.

PAR DÉFAUT, LA MÉTHODE DES CLASSEURS (NBN B 51-002, § 4.3.3.5.3, tableaux 11 et 12, que les classeurs citent) :

    λ_e,d = 3,9·√(l·h / (b·t_f·C_d))      l : distance entre maintiens latéraux de la semelle comprimée (L_d, distincte
                                          de L_cr,y et L_cr,z) ; h : hauteur ; b, t_f : semelle COMPRIMÉE ; C_d : 1 par défaut
    λ̄ = λ_e,d / λ_E                       (λ_E avec la f_y nominale)
    Φ = 1 + n·(λ̄ − 0,2) + λ̄²     φ_d = [Φ − √(Φ² − 4·λ̄²)] / (2·λ̄²)      φ_d = 1 si λ̄ ≤ 0,2
    σ_adm,d = f_y / s × φ_d                et la vérification σ = M/W ≤ σ_adm,d

Le 3,9 : l'élancement équivalent du fascicule (doc p. 18, λ_e,d² = π²·E·W/M_cr), avec la seule torsion de Saint-Venant
et deux semelles égales, donne √(3π·√2,6) = 3,899 ; le module garde 3,9, comme les classeurs.

n (tableau 11) : a 0,206 ; b 0,339 ; c 0,489 ; d 0,756. La classe (tableau 12, ligne « axe faible », f_y ≤ 430 N/mm²) :
I ou H laminé — h/b > 1,2 : b jusqu'à t_f = 40 mm, c jusqu'à 80 mm ; h/b ≤ 1,2 : c jusqu'à 80 mm ; au-delà, d (pour
h/b > 1,2 le tableau n'a pas de ligne : d, choix prudent, note 3 du tableau 11) ; I soudé (PRS) — semelles oxycoupées :
c jusqu'à 40 mm, d au-delà ; larges plats : d ; U : c.

PROFILS FERMÉS (RHS, caissons ; doc p. 17-18, gauchissement négligé) : M_cr = (π/L_d)·√(E·I_z·G·I_t), G = E/2,6, d'où
λ_ed = √(π·√(E/G)·L_d·W_el,y / √(I_z·I_t)) ; le facteur π·√2,6 = 5,0657 (la doc l'arrondit à 5,1 : variante), la courbe
du fascicule. Tube rond ou carré : pas de déversement (φ_ed = 1, règle complémentaire, EC3 § 6.3.2.1(2) : I_y = I_z).

LA VARIANTE DU FASCICULE (doc p. 19), éteinte par défaut : la semelle comprimée et un sixième de l'âme, vus comme une
barre comprimée — I_d = t_f·b³/12, A_d = b·t_f + h_w·t_w/6, λ_ed = L_d/√(I_d/A_d), courbe de l'axe faible.

LES CORRECTIONS DES CLASSEURS (brief § 6.4 et 6.7) sont NOMMÉES ici (``CORRECTIONS``) : chaque fonction dit celles qui
jouent sur le cas calculé ; la vérification les imprime (« Correction du classeur : … »). Les choix du bureau (f_y
nominale, s, C_d, courbes et classes) n'en sont pas.
"""
import math
from typing import Dict, List, Optional

from . import section as SE
from .resultat import Refus

VARIANTES = ("classeurs", "fascicule")      # la méthode des classeurs (défaut) ; la variante du fascicule p. 19
K_CLASSEURS = 3.9                           # λ_e,d = 3,9·√(l·h/(b·t_f·C_d)) (classeurs ; doc p. 18 : 3,899)
N_TABLEAU_11 = {"a": 0.206, "b": 0.339, "c": 0.489, "d": 0.756}     # NBN B 51-002, tableau 11
CLASSES = tuple(N_TABLEAU_11)
PALIER = 0.2                                # φ_d = 1 si λ̄ ≤ 0,2 (NBN B 51-002, § 4.3.3.5.3)
TF_40, TF_80 = 40.0, 80.0                   # mm : les seuils d'épaisseur du tableau 12
TF_EUROCODE = 100.0                         # mm : le seuil que la feuille « Calcul automatique » prenait (correction 3)
HB_TABLEAU_12 = 1.2                         # h/b : la limite du tableau 12
FY_CLASSEURS = 235.0                        # N/mm² : la f_y que la feuille S355 de DEVERSEMENT_CAD prenait (correction 4)
# la correction 4 n'est connue que pour la feuille S355 (brief § 6.4) : elle n'est dite que pour f_y = 355 N/mm². À VÉRIFIER :
# une feuille S275 des classeurs anciens, s'il y en a une, prenait-elle aussi 157 N/mm² ?
FY_FEUILLE_S355 = 355.0
# À VÉRIFIER : le brief ne borne pas C_d ; 1 (moment uniforme, le cas le plus défavorable) à 3 (l'ordre des C₁ des
# diagrammes courants) est une borne de saisie du module, pas une valeur des classeurs
CD_MIN, CD_MAX = 1.0, 3.0
# la colonne lue au tableau 12 : f_y ≤ 430 N/mm² (brief § 6.4). À VÉRIFIER : la colonne f_y > 430 n'est pas dans le brief ;
# au-delà, la classe de la colonne ≤ 430 est gardée (hors domaine de toute façon : nuance hors AE235 à AE355) et l'hypothèse
# le dit
FY_COLONNE_12 = 430.0
E_SUR_G = 2.6                               # E/G = 2·(1 + ν), ν = 0,3 (doc p. 18)
FACTEUR_DOC = 5.1                           # la doc arrondit π·√2,6 = 5,0657 à 5,1 (variante, brief § 6.4)
SOUDES = ("soude_oxycoupe", "soude_larges_plats")
CAISSONS = ("caisson", "caisson_recuit")
# les I que la règle des classeurs ne couvre pas, nommés pour le message « à fournir » (jamais la clé interne)
PROFILS_A_FOURNIR = {"recuit": "un I recuit", "lamine_semelles_additionnelles": "un I à semelles additionnelles"}

# les VRAIES erreurs des classeurs que le module corrige (brief § 6.2, 6.4, 6.7) : identifiant → phrase d'une ligne
# (« {…} » : la valeur du cas). « une_courbe » est un garde-fou : il ne joue sur aucun cas calculé (la saisie l'impose).
CORRECTIONS = {
    "pcr_u": "P_cr avec (u·L)² ; le classeur Flambement oubliait u.",
    "phi_min": "φ_min dans la flexion composée (ici l'axe fort) ; le classeur prenait toujours l'axe faible.",
    "une_courbe": "une seule courbe par axe ; le classeur rendait φ = 1 sans choix et additionnait α et β pour deux choix.",
    "cisaillement_abs": "|V_y| + |V_z| ; le classeur additionnait les valeurs signées, qui se compensaient.",
    "phid_plafond": "φ_d = 1 pour λ̄ ≤ 0,2 (NBN B 51-002) ; prolongée, la formule des classeurs donne {prolonge}.",
    "seuils_epaisseur": "classe pour 40 < t_f ≤ 80 mm (tableau 12) ; les tests d'épaisseur des classeurs n'en donnaient aucune.",
    "seuils_eurocode": "seuils de 40 et 80 mm du tableau 12 ; la feuille « Calcul automatique » prenait 100 mm (Eurocode).",
    "fy_nuance": "σ_adm,d avec la f_y de la nuance ({fy} N/mm²) ; la feuille S355 de DEVERSEMENT_CAD prenait 157 N/mm² (S235).",
    "classe_type": "classe du type de profilé ({classe}) ; DEVERSEMENT_CAD donnait celle d'un I laminé ({classe_lamine}).",
    "axe_faible": "classe de la ligne « axe faible » seule (doc p. 19) ; les classeurs montraient aussi la ligne « axe fort ».",
}
# les messages de 6.4, sans changer les valeurs
MESSAGE_CD = ("La formule des classeurs ne tient pas compte de la hauteur de la charge : avec une charge sur la semelle haute, "
              "gardez C_d = 1.")
MESSAGE_PRS = "Formule prudente : elle néglige le gauchissement, qui compte beaucoup pour une poutre haute. Voir l'Eurocode 3."
MESSAGE_3_9 = "3,9 = √(3π·√2,6) arrondi (doc p. 18 : torsion de Saint-Venant seule, deux semelles égales)."
MESSAGE_CLASSES = "Classes du flambement (fascicule) et du déversement (tableau 12) différentes, comme dans les classeurs."
MESSAGE_COLONNE = "classe lue pour f_y ≤ 430 N/mm²"
MESSAGE_FERME = "facteur π·√2,6 = 5,0657 (5,1 dans la doc)."              # (après « L_d = … m ; »)


def texte_correction(cle: str, **valeurs) -> str:
    """La phrase d'une correction, avec les valeurs du cas — déjà écrites (``resultat.fr``) : aucun second arrondi ici."""
    t = CORRECTIONS[cle]
    for k, v in valeurs.items():
        t = t.replace("{" + k + "}", str(v))
    return t


class AFournir(Refus):
    """Une classe du tableau 12 que la règle des classeurs ne donne pas : « valeur à fournir » (brief § 1, règle 1) — le
    déversement est BLOQUÉ, le reste de la méthode se calcule."""


# ------------------------------------------------------------------ la géométrie de la semelle comprimée
def est_prs(s: Dict) -> bool:
    """Un I soudé (PRS) : les bords de ses semelles décident ses courbes et sa classe (acier/barre/prs.py)."""
    return s.get("forme") == "I" and (isinstance(s.get("prs"), dict) or s.get("fabrication") in SOUDES)


def geometrie(s: Dict, semelle: str = "haut") -> Dict:
    """{"h", "b", "tf", "hw", "tw"} (mm) d'un I, avec la semelle COMPRIMÉE (« haut » ou « bas ») : un PRS lit ses plats
    (``s["prs"]`` : b1 × t1 en haut, b2 × t2 en bas, âme h_w × t_w) ; un laminé, ses dimensions (h_w = h − 2·t_f)."""
    p = s.get("prs") if isinstance(s.get("prs"), dict) else None
    if p:
        b, tf = (p["b1"], p["t1"]) if semelle == "haut" else (p["b2"], p["t2"])
        h = s.get("h") or p["t1"] + p["hw"] + p["t2"]
        return {"h": float(h), "b": float(b), "tf": float(tf), "hw": float(p["hw"]), "tw": float(p["tw"])}
    return {"h": s.get("h"), "b": s.get("b"), "tf": s.get("tf"), "hw": SE.hauteur_ame(s), "tw": s.get("tw")}


# ------------------------------------------------------------------ φ_d (NBN B 51-002)
def phi_d_prolonge(lambda_bar: float, classe: str) -> float:
    """La formule des classeurs, sans le palier : [Φ − √(Φ² − 4·λ̄²)] / (2·λ̄²), Φ = 1 + n·(λ̄ − 0,2) + λ̄² (λ̄ > 0)."""
    n = N_TABLEAU_11[classe]
    Phi = 1.0 + n * (lambda_bar - PALIER) + lambda_bar * lambda_bar
    return (Phi - math.sqrt(Phi * Phi - 4.0 * lambda_bar * lambda_bar)) / (2.0 * lambda_bar * lambda_bar)


def phi_d(lambda_bar: float, classe: str) -> float:
    """φ_d de la NBN B 51-002 (§ 4.3.3.5.3) : 1 si λ̄ ≤ 0,2, sinon la formule (correction 1 : prolongée sous 0,2, elle
    dépasse 1 — jusqu'à 1,04 en classe a et 1,18 en classe d)."""
    if lambda_bar <= PALIER:
        return 1.0
    return phi_d_prolonge(lambda_bar, classe)


# ------------------------------------------------------------------ la classe (tableau 12, ligne « axe faible »)
def _classe_lamine(h: float, b: float, tf: float) -> str:
    if h / b > HB_TABLEAU_12:
        return "b" if tf <= TF_40 else "c" if tf <= TF_80 else "d"
    return "c" if tf <= TF_80 else "d"


def classe(s: Dict, h: float, b: float, tf: float) -> Dict:
    """La classe de déversement proposée d'office (tableau 12, ligne « axe faible », f_y ≤ 430 N/mm²), avec t_f de la
    semelle comprimée : {"classe", "regle", "classe_lamine" (celle d'un I laminé de mêmes h, b, t_f), "corrections"}.
    ``AFournir`` (un Refus) quand la règle des classeurs ne couvre pas le profil (valeur à fournir : une classe saisie)."""
    forme, fab = s.get("forme"), s.get("fabrication")
    lam = _classe_lamine(h, b, tf)
    corr: List[str] = []
    if forme == "U":
        cl, regle = "c", "U"
    elif forme == "I" and fab == "lamine":
        cl = lam
        # la règle, COURTE (une ligne « Déversement » étiquetée reste sous 110 caractères — intégration du lot B) : les seuls
        # critères qui décident (h/b ≤ 1,2 : c jusqu'à 80 mm, sans seuil à 40) ; « laminé » est le cas du profil, pas une raison
        elance = h / b > HB_TABLEAU_12
        ep = ("t_f > 80 mm" if tf > TF_80 else ("t_f ≤ 40 mm" if tf <= TF_40 else "40 < t_f ≤ 80 mm") if elance else "t_f ≤ 80 mm")
        regle = f"h/b {'>' if elance else '≤'} 1,2, {ep}"
        if (h / b > HB_TABLEAU_12 and TF_40 < tf <= TF_80) or tf == TF_80:
            corr.append("seuils_epaisseur")        # « 4 < tf < 8 » (cm) toujours faux ; « tf < 8 » sans tf = 8 (correction 2)
        if TF_80 < tf <= TF_EUROCODE:
            corr.append("seuils_eurocode")          # 100 mm de la feuille « Calcul automatique » (correction 3)
    elif forme == "I" and fab in SOUDES:
        if fab == "soude_oxycoupe":
            cl, regle = ("c", "oxycoupé, t_f ≤ 40 mm") if tf <= TF_40 else ("d", "oxycoupé, t_f > 40 mm")
        else:
            cl, regle = "d", "larges plats"
    else:
        raise AFournir(f"Classe de déversement à fournir (NBN B 51-002, tableau 12) pour {PROFILS_A_FOURNIR.get(fab, 'ce profil')} — "
                       "flexion non jugée.")
    if forme != "I" or fab != "lamine":
        if cl != lam:
            corr.append("classe_type")              # DEVERSEMENT_CAD : toujours la classe d'un I laminé (correction 5)
    if forme == "I" and not (fab == "lamine" and tf > TF_80):
        # À VÉRIFIER : la ligne « axe fort » du tableau 12 n'est pas dans le brief ; la correction 6 est dite pour tout I, sauf
        # un laminé de plus de 80 mm, où la ligne « axe fort » donne sans doute d aussi (la correction n'y jouerait pas)
        corr.append("axe_faible")
    return {"classe": cl, "regle": regle, "classe_lamine": lam, "corrections": corr}


# ------------------------------------------------------------------ les trois formules
def classeurs(s: Dict, Ld: float, lambda_E: float, Cd: float = 1.0, semelle: str = "haut", classe_imposee: Optional[str] = None) -> Dict:
    """La méthode des classeurs « Déversement » pour la longueur ``Ld`` (mm) entre maintiens de la semelle comprimée :
    {"cas": "classeurs", "lambda" (λ_e,d), "lambda_bar", "classe", "classe_norme", "regle", "n", "Phi", "phi" (φ_d),
    "phi_prolonge", "h", "b", "tf", "Lc" (L_d), "Cd", "semelle", "corrections"} (sans la correction 4, qui dépend de f_y)."""
    if not CD_MIN <= Cd <= CD_MAX:
        raise Refus(f"le coefficient de moment C_d du déversement doit être entre {CD_MIN:g} et {CD_MAX:g} : {Cd!r}")
    g = geometrie(s, semelle)
    if not g["h"] or not g["b"] or not g["tf"]:
        raise Refus("h, b et t_f manquent : le déversement des classeurs en dépend")
    lam = K_CLASSEURS * math.sqrt(Ld * g["h"] / (g["b"] * g["tf"] * Cd))
    lb = lam / lambda_E
    if classe_imposee is not None and classe_imposee not in CLASSES:
        raise Refus(f"classe de déversement : a, b, c ou d — reçu {classe_imposee!r}")
    try:                    # la classe du tableau 12, même quand une autre est imposée (l'alerte la compare)
        cl = classe(s, g["h"], g["b"], g["tf"])
    except AFournir:
        if classe_imposee is None:
            raise
        cl = None
    k = classe_imposee or cl["classe"]
    n = N_TABLEAU_11[k]
    Phi = 1.0 + n * (lb - PALIER) + lb * lb
    prolonge = phi_d_prolonge(lb, k)
    corr = list(cl["corrections"]) if (cl and classe_imposee is None) else []      # une classe imposée : aucune correction de classe
    if lb < PALIER and prolonge > 1.0:
        corr.insert(0, "phid_plafond")
    return {"cas": "classeurs", "lambda": lam, "lambda_bar": lb, "classe": k, "classe_norme": cl["classe"] if cl else None,
            "classe_lamine": cl["classe_lamine"] if cl else None, "regle": "classe imposée" if classe_imposee else cl["regle"], "n": n, "Phi": Phi,
            "phi": phi_d(lb, k), "phi_prolonge": prolonge, "h": g["h"], "b": g["b"], "tf": g["tf"], "Lc": Ld, "Cd": Cd, "semelle": semelle,
            "corrections": corr}


def fascicule(s: Dict, Ld: float, lambda_E: float, semelle: str = "haut") -> Dict:
    """La variante du fascicule (doc p. 19) : {"cas": "I", "Id", "Ad", "rd", "hw", "b", "tf", "tw", "lambda", "lambda_bar",
    "Lc"} — φ_ed se lit sur la courbe de l'axe faible du fascicule (acier/barre/historique.phi). Un PRS prend sa semelle
    comprimée et son âme (la doc établit I_d et A_d pour deux semelles égales)."""
    g = geometrie(s, semelle)
    if g["hw"] is None or not g["b"] or not g["tw"] or not g["tf"]:
        raise Refus("h, b, t_w et t_f manquent : le déversement d'un I en dépend")
    Id = g["tf"] * g["b"] ** 3 / 12.0
    Ad = g["b"] * g["tf"] + g["hw"] * g["tw"] / 6.0
    rd = math.sqrt(Id / Ad)
    lam = Ld / rd
    return {"cas": "I", "Id": Id, "Ad": Ad, "rd": rd, "hw": g["hw"], "b": g["b"], "tf": g["tf"], "tw": g["tw"], "lambda": lam,
            "lambda_bar": lam / lambda_E, "Lc": Ld, "semelle": semelle}


def ferme(s: Dict, Ld: float, lambda_E: float, facteur: Optional[float] = None, e_sur_g: float = E_SUR_G) -> Dict:
    """Un profil fermé rectangulaire ou un caisson (doc p. 17-18) : λ_ed = √(F·L_d·W_el,y/√(I_z·I_t)), F = π·√(E/G)
    (5,0657 avec E/G = 2,6 ; ``facteur`` : celui de la doc, 5,1) — {"cas": "ferme", "lambda", "lambda_bar", "facteur",
    "Lc"} ; φ_ed se lit sur la courbe du fascicule."""
    if not s.get("It") or not s.get("Wel_y") or not s.get("Iz"):
        raise Refus("I_t, I_z et W_el,y manquent : le déversement d'un profil fermé en dépend")
    F = math.pi * math.sqrt(e_sur_g) if facteur is None else facteur
    lam = math.sqrt(F * Ld * s["Wel_y"] / math.sqrt(s["Iz"] * s["It"]))
    return {"cas": "ferme", "lambda": lam, "lambda_bar": lam / lambda_E, "facteur": F, "Lc": Ld}

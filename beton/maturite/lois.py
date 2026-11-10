# -*- coding: utf-8 -*-
"""Les LOIS du module « Maturité du béton » : les tables des normes — chacune UNE fois, sa source à côté — et les
formules, en ``math`` pur (ni numpy, ni rien d'autre que la bibliothèque standard).

    f_cm(t) = β_cc(t_T)·f_cm                 f_cm = f_ck + 8                    EN 1992-1-1:2004 §3.1.2(6), (3.1)
    β_cc(t_T) = exp{ s·[1 − √(28/t_T)] }                                         (3.2)
    f_ck(t) = f_cm(t) − 8  pour 3 < t_T < 28 j ;  f_ck pour t_T ≥ 28 j           §3.1.2(5)
    t_T = Σ Δt_i · k(T_i)                    l'âge équivalent à 20 °C

Ce fichier porte :
- les CLASSES de béton (f_ck, f_ck,cube) ;
- le DURCISSEMENT — le coefficient s — par le ciment (R, N, S de l'EC2) ou par le béton (rapide, moyen, lent de la
  NBN EN 206, écrit sur le bon de livraison ; ou le rapport r connu) : ``durcissement`` ;
- les trois LOIS DE TEMPÉRATURE k(T) : la NBN B 15-400 (« âge fictif », par défaut), l'EC2 (B.10), Arrhenius à énergie
  d'activation saisie : ``k_T`` ;
- la résistance à un âge équivalent (``beta_cc``, ``fcm_t``, ``fck_t``) et son inverse en forme fermée (``t_T_requis``) ;
- l'INTÉGRATION de l'âge équivalent sur un profil journalier (``age_equivalent``) et son inverse exact (``age_reel``),
  l'heure de fin de bétonnage comprise ; la prolongation au-delà du relevé (``temperature_prolongee``) ; le calage de s
  sur un essai (``s_cale``).

Le temps : t = 0 à la FIN DU BÉTONNAGE. La température est constante par jour civil ; le jour du bétonnage (jour 0) ne
compte que pour la fraction qui en reste, (24 − heure)/24. t_T(t) est donc linéaire par morceaux : il s'intègre et
s'inverse exactement, sans dichotomie.

Unités : °C, jours, MPa, kJ/mol. Méthode, sources et preuves : ``docs/beton/MATURITE_MOTEUR.md``.
"""
import math

# ------------------------------------------------------------------ les classes de béton
# EN 1992-1-1:2004, tableau 3.1 (et NBN EN 206, classes de résistance à la compression) : f_ck sur cylindre, f_ck,cube
# sur cube de 150 mm, en MPa. Le rapport f_ck/f_ck,cube (0,78 à 0,83) ramène un cube au cylindre.
CLASSES = {"C16/20": (16.0, 20.0), "C20/25": (20.0, 25.0), "C25/30": (25.0, 30.0), "C30/37": (30.0, 37.0),
           "C35/45": (35.0, 45.0), "C40/50": (40.0, 50.0), "C45/55": (45.0, 55.0), "C50/60": (50.0, 60.0)}
DELTA_FCM = 8.0            # f_cm = f_ck + 8 MPa — EN 1992-1-1:2004, tableau 3.1
AGE_REFERENCE_J = 28.0     # l'âge de référence de (3.2), en jours
SEUIL_INDICATIF_J = 3.0    # §3.1.2(5) : f_ck(t) = f_cm(t) − 8 vaut pour 3 < t < 28 j ; en deçà, « à confirmer par essais »


def fck_classe(classe: str) -> float:
    """f_ck (MPa) d'une classe du tableau ; ValueError si elle n'y est pas."""
    if classe not in CLASSES:
        raise ValueError(f"classe : {classe!r} n'est pas une classe du module ({', '.join(CLASSES)})")
    return CLASSES[classe][0]


def rapport_cube(classe: str) -> float:
    """f_ck/f_ck,cube de la classe : ce qui ramène la résistance d'un cube de 150 mm à celle du cylindre."""
    fck, cube = CLASSES[classe]
    return fck / cube


# ------------------------------------------------------------------ le durcissement : le coefficient s
# EN 1992-1-1:2004 §3.1.2(6) : s selon la classe de RÉSISTANCE DU CIMENT, et les ciments de chaque classe
CIMENTS = {"R": {"s": 0.20, "ciments": ("CEM 42,5 R", "CEM 52,5 N", "CEM 52,5 R")},
           "N": {"s": 0.25, "ciments": ("CEM 32,5 R", "CEM 42,5 N")},
           "S": {"s": 0.38, "ciments": ("CEM 32,5 N",)}}
# NBN EN 206 (§ 7.2, tableau « Développement de la résistance du béton à 20 °C » — tableau 16 de l'édition 2013, 12 de
# l'EN 206-1:2000 ; repris par l'EN 13670:2009, annexe F) : la classe écrite sur le BON DE LIVRAISON, par le rapport
# r = f_cm,2j / f_cm,28j. (borne basse comprise, borne haute exclue ; None : pas de borne haute). Le moteur prend s à
# la BORNE BASSE de la classe : le durcissement le plus lent qu'elle admet, du côté sûr. (« très lent », r < 0,15,
# n'est pas proposé.)
BETONS = {"rapide": (0.50, None), "moyen": (0.30, 0.50), "lent": (0.15, 0.30)}
AGE_RAPPORT_J = 2.0        # r compare la résistance moyenne à 2 jours à celle à 28 jours
R_MIN, R_MAX = 0.15, 0.70  # le domaine du rapport r saisi (« r connu ») : de la borne du béton lent à un béton très rapide
S_CALE_MIN, S_CALE_MAX = 0.10, 0.80   # le domaine d'un s calé sur un essai : hors de lui, le calage est refusé
MODES_DURCISSEMENT = ("ciment", "beton", "r")


def s_de_r(r: float) -> float:
    """s tel que β_cc(2 j) = r : (3.2) écrite à 2 jours, r = exp{s·(1 − √14)} → s = −ln r / (√14 − 1)."""
    if not 0.0 < r < 1.0:
        raise ValueError(f"r : {r!r} doit être strictement entre 0 et 1")
    return -math.log(r) / (math.sqrt(AGE_REFERENCE_J / AGE_RAPPORT_J) - 1.0)


def r_de_s(s: float) -> float:
    """r = f_cm,2j / f_cm,28j que suppose un coefficient s : β_cc(2 j) = exp{s·(1 − √14)}."""
    return math.exp(s * (1.0 - math.sqrt(AGE_REFERENCE_J / AGE_RAPPORT_J)))


def durcissement(choix: dict) -> dict:
    """Le durcissement choisi → {"mode", "cle", "s", "r", "libelle", "source", "detail"} : s retenu, le rapport r qu'il
    suppose, les mots pour le dire. ``choix`` : {"mode": "ciment" | "beton" | "r", "ciment": "R" | "N" | "S",
    "beton": "rapide" | "moyen" | "lent", "r": 0,15 à 0,70}."""
    mode = choix["mode"]
    if mode == "ciment":
        c = choix["ciment"]
        s = CIMENTS[c]["s"]
        return {"mode": mode, "cle": c, "s": s, "r": r_de_s(s), "libelle": f"ciment de classe {c}",
                "source": "EN 1992-1-1 §3.1.2(6)", "detail": " · ".join(CIMENTS[c]["ciments"])}
    if mode == "beton":
        b = choix["beton"]
        bas, haut = BETONS[b]
        return {"mode": mode, "cle": b, "s": s_de_r(bas), "r": bas, "libelle": f"béton à durcissement {b}",
                "source": "NBN EN 206 (développement de la résistance), s à la borne basse de la classe",
                "detail": f"r ≥ {bas:.2f}".replace(".", ",") if haut is None else f"{bas:.2f} ≤ r < {haut:.2f}".replace(".", ",")}
    if mode == "r":
        r = float(choix["r"])
        return {"mode": mode, "cle": "r", "s": s_de_r(r), "r": r, "libelle": "rapport r connu",
                "source": "rapport r = f_cm,2j / f_cm,28j donné par la centrale ou par des essais", "detail": ""}
    raise ValueError(f"durcissement/mode : {mode!r} ({', '.join(MODES_DURCISSEMENT)})")


# ------------------------------------------------------------------ la température : le coefficient k(T)
T_REFERENCE_C = 20.0       # la température de référence : k = 1, un jour vaut un jour
# NBN B 15-400 (complément belge à la NBN EN 13670), « âge fictif » : le coefficient k selon la température moyenne du
# béton sur 24 h — k = 1 à 20 °C et au-dessus, 0,8 à 15 °C, 0,6 à 10 °C, 0,45 à 5 °C, 0,3 à 0 °C, 0,15 à −5 °C,
# interpolé linéairement (l'exemple de la norme : trois jours à 5, 7 et 5 °C valent 1,41 j). Valeurs relevées sur deux
# sources concordantes (projet d'annexe devenu NBN B 15-400), à contrôler sur l'édition 2024.
K_NBN_NORME = ((-5.0, 0.15), (0.0, 0.30), (5.0, 0.45), (10.0, 0.60), (15.0, 0.80), (20.0, 1.00))
# DEUX CHOIX DU MOTEUR, qui ne sont pas dans la norme : sous −5 °C, la droite de −5 à 0 °C (0,03 par degré) est
# prolongée jusqu'à k = 0 à −10 °C, et k = 0 en dessous ; au-dessus de 20 °C, k reste à 1,00 (aucun bonus compté).
T_NUL_NBN = -10.0
K_NBN = ((T_NUL_NBN, 0.0),) + K_NBN_NORME
# EN 1992-1-1:2004 annexe B, (B.10) : t_T = Σ Δt_i·exp{−[4000/(273 + T) − 13,65]}, T de 0 à 80 °C. « 13,65 » y est
# l'arrondi de 4000/293 = 13,652 : le moteur prend la référence exacte, k = 1 à 20 °C. Hors du domaine (sous 0 °C) : k = 0.
EA_SUR_R_EC2 = 4000.0      # K
T_KELVIN = 273.0           # (B.10) écrit 273 + T
T_MIN_EC2, T_MAX_EC2 = 0.0, 80.0
R_GAZ = 8.314              # J/(mol·K), la constante des gaz parfaits
EA_MIN, EA_MAX = 20.0, 60.0                      # kJ/mol, le domaine de l'énergie d'activation saisie
EA_EC2_KJ = EA_SUR_R_EC2 * R_GAZ / 1000.0        # 33,256 kJ/mol : l'énergie d'activation que (B.10) suppose
LOIS = {"nbn": {"libelle": "NBN B 15-400", "source": "NBN B 15-400 (complément belge à la NBN EN 13670), âge fictif"},
        "ec2": {"libelle": "EC2 (B.10)", "source": "EN 1992-1-1 annexe B, (B.10)"},
        "perso": {"libelle": "Arrhenius, E_a saisie", "source": "loi d'Arrhenius, énergie d'activation saisie"}}


def k_nbn(T: float) -> float:
    """k de la NBN B 15-400 : interpolation LINÉAIRE entre les points du tableau, 1,00 au-dessus de 20 °C, la droite
    prolongée jusqu'à 0 à −10 °C, 0 en dessous."""
    if T <= K_NBN[0][0]:
        return 0.0
    if T >= K_NBN[-1][0]:
        return K_NBN[-1][1]
    for (Ta, ka), (Tb, kb) in zip(K_NBN, K_NBN[1:]):
        if T <= Tb:
            return ka + (kb - ka) * (T - Ta) / (Tb - Ta)
    return K_NBN[-1][1]                                           # pragma: no cover — la boucle couvre tout l'intervalle


def segment_nbn(T: float):
    """Les deux points du tableau qui encadrent T (pour écrire l'interpolation) : ((T_a, k_a), (T_b, k_b)) ; None hors
    du tableau (k = 0 sous −10 °C, k = 1 au-dessus de 20 °C)."""
    if T <= K_NBN[0][0] or T >= K_NBN[-1][0]:
        return None
    for a, b in zip(K_NBN, K_NBN[1:]):
        if T <= b[0]:
            return a, b
    return None                                                   # pragma: no cover


def k_arrhenius(T: float, Ea_kJ: float) -> float:
    """k d'Arrhenius : exp{−(E_a/R)·[1/(273 + T) − 1/293]}, R = 8,314 J/(mol·K) ; 0 sous 0 °C (hors domaine)."""
    if T < T_MIN_EC2:
        return 0.0
    if T > T_MAX_EC2:
        raise ValueError(f"température : {T:g} °C dépasse le domaine de la loi ({T_MAX_EC2:g} °C)")
    return math.exp(-(Ea_kJ * 1000.0 / R_GAZ) * (1.0 / (T_KELVIN + T) - 1.0 / (T_KELVIN + T_REFERENCE_C)))


def k_ec2(T: float) -> float:
    """k de l'EC2, (B.10) : exp{−4000·[1/(273 + T) − 1/293]} pour 0 ≤ T ≤ 80 °C ; 0 sous 0 °C (hors domaine)."""
    if T < T_MIN_EC2:
        return 0.0
    if T > T_MAX_EC2:
        raise ValueError(f"température : {T:g} °C dépasse le domaine de (B.10) ({T_MAX_EC2:g} °C)")
    return math.exp(-EA_SUR_R_EC2 * (1.0 / (T_KELVIN + T) - 1.0 / (T_KELVIN + T_REFERENCE_C)))


def k_T(T: float, loi: dict) -> float:
    """Le coefficient k(T) de la loi choisie : ``loi`` = {"mode": "nbn" | "ec2" | "perso", "Ea_kJ": …}."""
    mode = loi["mode"]
    if mode == "nbn":
        return k_nbn(T)
    if mode == "ec2":
        return k_ec2(T)
    if mode == "perso":
        return k_arrhenius(T, loi["Ea_kJ"])
    raise ValueError(f"loi/mode : {mode!r} ({', '.join(LOIS)})")


# ------------------------------------------------------------------ la résistance à un âge équivalent
BASES = ("fck", "fcm")     # ce à quoi une cible se compare : la résistance caractéristique f_ck(t), ou la moyenne f_cm(t)


def beta_cc(t_T: float, s: float) -> float:
    """β_cc(t_T) = exp{s·[1 − √(28/t_T)]} — (3.2), telle qu'écrite ; 0 à t_T = 0."""
    if t_T <= 0.0:
        return 0.0
    return math.exp(s * (1.0 - math.sqrt(AGE_REFERENCE_J / t_T)))


def beta_retenu(t_T: float, s: float) -> float:
    """β_cc borné à 1 : AUCUN gain n'est compté au-delà de 28 jours équivalents (f_ck(t) = f_ck pour t ≥ 28 j, §3.1.2(5) ;
    le moteur applique la même borne à la moyenne — une échéance de chantier ne s'appuie pas sur le gain d'après 28 j)."""
    return min(beta_cc(t_T, s), 1.0)


def fcm_t(t_T: float, fck: float, s: float) -> float:
    """f_cm(t) = β_cc(t_T)·f_cm, f_cm = f_ck + 8 — (3.1) ; plafonnée à f_cm au-delà de 28 jours équivalents."""
    return beta_retenu(t_T, s) * (fck + DELTA_FCM)


def fck_t(t_T: float, fck: float, s: float) -> float:
    """f_ck(t) = f_cm(t) − 8 — §3.1.2(5) —, jamais négative : max(0 ; …) ; f_ck au-delà de 28 jours équivalents.
    Avant 3 jours équivalents la valeur est INDICATIVE (``indicatif``)."""
    return max(0.0, fcm_t(t_T, fck, s) - DELTA_FCM)


def resistance(t_T: float, base: str, fck: float, s: float) -> float:
    """La résistance à laquelle une cible se compare : f_ck(t) (base « fck ») ou f_cm(t) (base « fcm »)."""
    if base == "fck":
        return fck_t(t_T, fck, s)
    if base == "fcm":
        return fcm_t(t_T, fck, s)
    raise ValueError(f"base : {base!r} ({', '.join(BASES)})")


def indicatif(t_T: float) -> bool:
    """t_T ≤ 3 j : hors du domaine où l'EC2 donne f_ck(t) = f_cm(t) − 8 — la valeur est à confirmer par essais."""
    return t_T <= SEUIL_INDICATIF_J


def beta_requis(cible: float, base: str, fck: float) -> float:
    """β_cc qu'il faut pour atteindre la cible : (cible + 8)/f_cm si elle se compare à la caractéristique, cible/f_cm si
    elle se compare à la moyenne."""
    fcm = fck + DELTA_FCM
    if base == "fck":
        return (cible + DELTA_FCM) / fcm
    if base == "fcm":
        return cible / fcm
    raise ValueError(f"base : {base!r} ({', '.join(BASES)})")


def t_T_requis(cible: float, base: str, fck: float, s: float):
    """L'âge équivalent où la cible est atteinte, en FORME FERMÉE : t_T = 28/(1 − ln β_req/s)², β_req = ``beta_requis``.
    None quand la cible dépasse ce que la classe donne à 28 jours (β_req > 1) : non atteignable."""
    if cible <= 0.0:
        return 0.0
    b = beta_requis(cible, base, fck)
    if b > 1.0 + 1e-12:
        return None
    if b >= 1.0:
        return AGE_REFERENCE_J
    return AGE_REFERENCE_J / (1.0 - math.log(b) / s) ** 2


# ------------------------------------------------------------------ l'âge équivalent sur un profil journalier
HEURES_PAR_JOUR = 24.0
JOURS_PROLONGATION = 5     # au-delà du dernier jour connu : la moyenne des cinq derniers jours (comme la météo)


def fraction_premier_jour(heure: float) -> float:
    """La fraction du jour du bétonnage qui compte : (24 − heure)/24 — la fin du bétonnage à 16 h laisse un tiers de jour."""
    return (HEURES_PAR_JOUR - heure) / HEURES_PAR_JOUR


def age_equivalent(t: float, ks, k_fin: float, f0: float) -> float:
    """t_T(t) = Σ Δt_i·k_i, l'âge équivalent à l'âge réel t (jours depuis la fin du bétonnage). ``ks`` : le coefficient
    k de chaque jour connu, le jour du bétonnage d'abord ; ``k_fin`` : celui des jours d'après (la prolongation, ou la
    température constante) ; ``f0`` : la fraction du premier jour (``fraction_premier_jour``)."""
    if t <= 0.0:
        return 0.0
    total, debut = 0.0, 0.0
    for j, k in enumerate(ks):
        duree = f0 if j == 0 else 1.0
        if t <= debut + duree:
            return total + k * (t - debut)
        total += k * duree
        debut += duree
    return total + k_fin * (t - debut)


def age_reel(t_T: float, ks, k_fin: float, f0: float):
    """L'INVERSE exact de ``age_equivalent`` : le plus petit âge réel t où l'âge équivalent atteint ``t_T`` ; None s'il
    ne l'atteint jamais (k = 0 jusqu'au bout : le gel, sous une loi qui ne compte rien sous 0 °C)."""
    if t_T <= 0.0:
        return 0.0
    total, debut = 0.0, 0.0
    for j, k in enumerate(ks):
        duree = f0 if j == 0 else 1.0
        if k > 0.0 and total + k * duree >= t_T:
            return debut + (t_T - total) / k
        total += k * duree
        debut += duree
    if k_fin > 0.0:
        return debut + (t_T - total) / k_fin
    return None


def segment(t: float, ks, k_fin: float, f0: float) -> dict:
    """Le jour où tombe l'âge réel t : {"jour", "debut" (âge réel au début du jour), "t_T_debut" (âge équivalent
    cumulé à ce début), "k"} — de quoi écrire t_T(t) = t_T,début + k·(t − début) et son inverse."""
    total, debut = 0.0, 0.0
    for j, k in enumerate(ks):
        duree = f0 if j == 0 else 1.0
        if t <= debut + duree:
            return {"jour": j, "debut": debut, "t_T_debut": total, "k": k}
        total += k * duree
        debut += duree
    if not ks:                                                    # une température constante : un seul morceau, depuis t = 0
        return {"jour": 0, "debut": 0.0, "t_T_debut": 0.0, "k": k_fin}
    m = max(0, math.ceil(t - debut - 1e-12) - 1)                  # les jours d'après, tous au même k (un jour chacun)
    return {"jour": len(ks) + m, "debut": debut + m, "t_T_debut": total + k_fin * m, "k": k_fin}


def segment_inverse(t_T: float, ks, k_fin: float, f0: float):
    """Le jour où l'âge équivalent atteint ``t_T`` (même forme que ``segment``) ; None s'il ne l'atteint jamais."""
    t = age_reel(t_T, ks, k_fin, f0)
    if t is None:
        return None
    total, debut = 0.0, 0.0
    for j, k in enumerate(ks):
        duree = f0 if j == 0 else 1.0
        if k > 0.0 and total + k * duree >= t_T:
            return {"jour": j, "debut": debut, "t_T_debut": total, "k": k}
        total += k * duree
        debut += duree
    return {"jour": len(ks), "debut": debut, "t_T_debut": total, "k": k_fin}


def temperature_prolongee(temps) -> float:
    """La température des jours d'APRÈS le relevé : la moyenne des cinq derniers jours connus (ceux qui portent une
    température ; à défaut, de tous ceux qui en portent une). ``temps`` : °C, ``nan`` ou None pour un jour sans donnée."""
    def valides(xs):
        return [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    v = valides(list(temps)[-JOURS_PROLONGATION:]) or valides(temps)
    if not v:
        raise ValueError("aucune température pour prolonger le relevé")
    return sum(v) / len(v)


# ------------------------------------------------------------------ le calage de s sur un essai
def s_cale(f_mes: float, fck: float, t_T: float) -> float:
    """s tel que la courbe passe par la mesure : f_mes = f_cm·exp{s·[1 − √(28/t_T)]} →
    s = ln(f_mes/f_cm) / (1 − √(28/t_T)). ValueError quand il n'existe pas (t_T ≤ 0, ou t_T ≥ 28 j : le dénominateur
    s'annule puis change de signe — une mesure à 28 jours ne dit rien du durcissement)."""
    if f_mes <= 0.0:
        raise ValueError("la résistance mesurée doit être positive")
    if t_T <= 0.0:
        raise ValueError("l'âge équivalent de l'essai est nul (aucun durcissement compté à cette température)")
    if t_T >= AGE_REFERENCE_J:
        raise ValueError("l'essai a 28 jours équivalents ou plus : il ne renseigne pas le durcissement")
    return math.log(f_mes / (fck + DELTA_FCM)) / (1.0 - math.sqrt(AGE_REFERENCE_J / t_T))

# -*- coding: utf-8 -*-
# =============================================================
#  sol_theorie.py — Noyau de calcul géotechnique (aucune dépendance
#  à Streamlit : ce module est testable seul).
#  VERSION 1.0
#
#  Il remplace le modèle « ressorts en série » 1/k = Σ hᵢ/Eᵢ de
#  raideur_sol v4.0, qui supposait la contrainte UNIFORME sur toute la
#  hauteur et sous-estimait k d'un facteur 2 à 2,5.
#
#  Méthode retenue — la seule qui définit k par ce que k signifie :
#      1. diffusion des contraintes    Δσ(z) = q · I(z)   [Boussinesq/Newmark]
#      2. tassement par tranches       w = Σ Δσᵢ · hᵢ / Mᵢ
#      3. coefficient de réaction      k = q / w
#      4. profondeur d'influence       calculée : Δσ ≤ crit · σ'v0
#
#  VALIDATION DE PREMIER PRINCIPE (voir tests/test_sol_theorie.py) :
#    · I_centre(z→0) = 1,000000  et  I_coin(z→0) = 0,250000
#    · champ lointain identique à la charge ponctuelle 3/(2π z²)
#    · ∫ I_centre(z) dz = 1,1198·B, à comparer au facteur élastique
#      publié Is = 1,12 pour un carré souple au centre — concordance à
#      4 chiffres significatifs contre une source indépendante.
#
#  Conventions d'unités, valables PARTOUT dans ce module :
#      longueurs        m
#      contraintes      kPa
#      modules          MPa   (M, E)
#      qc               MPa
#      fs               kPa
#      k                kN/m³ (converti en MN/m³ pour l'affichage)
#      poids volumique  kN/m³
# =============================================================

import math

PA = 101.325            # kPa — pression atmosphérique de référence
GAMMA_W = 9.81          # kN/m³ — poids volumique de l'eau


# =============================================================
#  1. DIFFUSION DES CONTRAINTES (Boussinesq / Newmark)
# =============================================================
def I_coin(B: float, L: float, z: float) -> float:
    """
    Facteur d'influence de la contrainte verticale sous le COIN d'un
    rectangle B×L uniformément chargé, à la profondeur z (Newmark).

    Δσ_z = q · I_coin

    Le terme en arctangente change de branche quand m²n² > m²+n²+1 :
    atan2 gère ce basculement de quadrant automatiquement, là où un
    atan simple renverrait une valeur négative fausse en profondeur.
    C'est le piège classique de cette formule.
    """
    if z <= 0:
        return 0.25
    if B <= 0 or L <= 0:
        return 0.0
    m = B / z
    n = L / z
    m2, n2 = m * m, n * n
    s = m2 + n2 + 1.0
    r = math.sqrt(s)
    t1 = (2.0 * m * n * r / (s + m2 * n2)) * ((m2 + n2 + 2.0) / s)
    t2 = math.atan2(2.0 * m * n * r, s - m2 * n2)
    return (t1 + t2) / (4.0 * math.pi)


# Positions de calcul sous une semelle/un radier B×L.
# Découpage en rectangles élémentaires ayant le point pour coin commun.
POSITIONS = {
    "centre": "Centre",
    "bord_long": "Milieu du grand côté",
    "bord_court": "Milieu du petit côté",
    "angle": "Angle",
}


def I_position(B: float, L: float, z: float, position: str = "centre") -> float:
    """Facteur d'influence sous un point remarquable de la fondation."""
    if z <= 0:
        return 1.0 if position != "angle" else 0.25
    if position == "centre":
        return 4.0 * I_coin(B / 2.0, L / 2.0, z)
    if position == "bord_court":          # milieu du côté de longueur B
        return 2.0 * I_coin(B / 2.0, L, z)
    if position == "bord_long":           # milieu du côté de longueur L
        return 2.0 * I_coin(B, L / 2.0, z)
    if position == "angle":
        return I_coin(B, L, z)
    raise ValueError(f"Position inconnue : {position}")


def delta_sigma(q_kPa: float, B: float, L: float, z: float,
                position: str = "centre") -> float:
    """Contrainte verticale apportée par la fondation à la profondeur z."""
    return q_kPa * I_position(B, L, z, position)


# =============================================================
#  2. CONTRAINTES EN PLACE
# =============================================================
def contraintes_en_place(couches, z: float, nappe_m=None):
    """
    (σv0 total, σ'v0 effectif) en kPa à la profondeur z sous le TN.

    couches : [{"h": m, "gamma": kN/m³, ...}] du haut vers le bas.
    Sous la nappe, on déjauge : γ' = γ − γw.
    """
    sv, prof = 0.0, 0.0
    reste = z
    for c in couches:
        if reste <= 0:
            break
        h = min(c.get("h", 0.0), reste)
        if h <= 0:
            continue
        g = float(c.get("gamma", 19.0) or 19.0)
        sv += g * h
        prof += h
        reste -= h
    if reste > 0 and couches:                     # au-delà du profil saisi
        g = float(couches[-1].get("gamma", 19.0) or 19.0)
        sv += g * reste
    u = 0.0
    if nappe_m is not None and z > nappe_m:
        u = (z - nappe_m) * GAMMA_W
    return sv, max(sv - u, 1.0)


# =============================================================
#  3. TASSEMENT ET COEFFICIENT DE RÉACTION
# =============================================================
def tassement(couches, B: float, L: float, q_kPa: float, D: float = 0.0,
              nappe_m=None, critere: float = 0.20, dz: float = 0.05,
              position: str = "centre", cle_module: str = "M",
              q_net: bool = True, z_max_mult: float = 6.0):
    """
    Tassement par la méthode des tranches, avec diffusion des contraintes.

        w = Σ Δσ(zᵢ) · dz / M(zᵢ)

    couches   : [{"h", "gamma", "M"}] à partir du TERRAIN NATUREL
    D         : profondeur d'assise sous le TN (les couches au-dessus de D
                ne participent pas — elles sont excavées ou hors charge)
    q_kPa     : pression de contact SOUS la fondation
    q_net     : si True, la contrainte utile est q − σv0(D) (déchargement
                dû à l'excavation). C'est le cas usuel en fondation.
    critere   : arrêt quand Δσ ≤ critere · σ'v0 (0,20 usuel ; 0,10 strict)
    position  : point de calcul (centre / bord / angle)

    Retourne un dict complet, exploitable pour l'affichage ET le graphique.
    """
    if B <= 0 or L <= 0 or q_kPa <= 0:
        return {"w_m": 0.0, "w_mm": 0.0, "k_kNm3": 0.0, "k_MNm3": 0.0,
                "z_influence": 0.0, "tranches": [], "q_net_kPa": 0.0,
                "convergence": "charge ou géométrie nulle"}

    sv0_D, _ = contraintes_en_place(couches, D, nappe_m)
    q_util = max(q_kPa - sv0_D, 0.0) if q_net else q_kPa
    if q_util <= 0:
        return {"w_m": 0.0, "w_mm": 0.0, "k_kNm3": 0.0, "k_MNm3": 0.0,
                "z_influence": 0.0, "tranches": [], "q_net_kPa": 0.0,
                "convergence": "pression nette nulle (q ≤ poids des terres excavées)"}

    H_profil = sum(c.get("h", 0.0) for c in couches)
    z_plafond = min(max(H_profil - D, 0.0), z_max_mult * max(B, 1e-6))
    if z_plafond <= 0:
        return {"w_m": 0.0, "w_mm": 0.0, "k_kNm3": 0.0, "k_MNm3": 0.0,
                "z_influence": 0.0, "tranches": [], "q_net_kPa": q_util,
                "convergence": "aucune couche sous le niveau d'assise"}

    def module_a(z_sous_tn):
        cum = 0.0
        for c in couches:
            h = c.get("h", 0.0)
            if z_sous_tn <= cum + h:
                return c.get(cle_module), c
            cum += h
        return (couches[-1].get(cle_module), couches[-1]) if couches else (None, None)

    w = 0.0
    z = 0.0                       # profondeur SOUS L'ASSISE
    z_infl = None
    tranches = []
    motif = "profil épuisé"
    sans_module = 0.0

    while z < z_plafond - 1e-12:
        pas = min(dz, z_plafond - z)
        zm = z + pas / 2.0                       # milieu de tranche
        z_tn = D + zm                            # profondeur sous le TN
        ds = delta_sigma(q_util, B, L, zm, position)
        _, svp0 = contraintes_en_place(couches, z_tn, nappe_m)

        if svp0 > 0 and ds <= critere * svp0:
            z_infl = zm
            motif = f"Δσ ≤ {critere:.0%} · σ'v0"
            break

        M, c = module_a(z_tn)
        if M is None or M <= 0:
            sans_module += pas
            z += pas
            continue

        dw = ds * pas / (M * 1000.0)             # M MPa -> kPa
        w += dw
        tranches.append({
            "z_sous_assise": zm, "z_sous_tn": z_tn, "dz": pas,
            "delta_sigma": ds, "sigma_v0_eff": svp0, "M": M,
            "dw_mm": dw * 1000.0, "couche": c.get("nom") if c else None,
            "ratio": ds / svp0 if svp0 > 0 else None,
        })
        z += pas

    if z_infl is None:
        z_infl = z
        if z >= z_plafond - 1e-9:
            motif = ("profil saisi épuisé" if H_profil - D <= z_max_mult * B
                     else f"plafond {z_max_mult:g}·B atteint")

    w_m = w
    k_kN = (q_util / w_m) if w_m > 1e-12 else 0.0
    return {
        "w_m": w_m, "w_mm": w_m * 1000.0,
        "k_kNm3": k_kN, "k_MNm3": k_kN / 1000.0,
        "z_influence": z_infl, "convergence": motif,
        "tranches": tranches, "q_net_kPa": q_util,
        "sigma_v0_assise": sv0_D, "position": position,
        "h_sans_module": sans_module,
    }


def k_zone(couches, B, L, q_kPa, D=0.0, nappe_m=None, critere=0.20,
           dz=0.05, cle_module="M", q_net=True):
    """
    k aux quatre points remarquables de la fondation.

    Sous une charge uniforme, le centre tasse plus que les bords (la
    contrainte s'y superpose davantage) : k y est donc plus FAIBLE. Cette
    variation est exactement ce que produit un calcul itératif de type
    Soilin. À défaut d'une licence Soilin, ces valeurs permettent
    d'encoder plusieurs zones de sol sous une même dalle dans SCIA.
    """
    out = {}
    for pos in POSITIONS:
        out[pos] = tassement(couches, B, L, q_kPa, D=D, nappe_m=nappe_m,
                             critere=critere, dz=dz, position=pos,
                             cle_module=cle_module, q_net=q_net)
    return out


# =============================================================
#  4. AUTRES THÉORIES — pour recoupement, jamais comme résultat principal
# =============================================================
def k_elastique(E_MPa: float, B: float, nu: float = 0.30,
                Is: float = 0.88) -> float:
    """
    Semelle sur massif semi-infini élastique : w = q·B·(1−ν²)·Is/E.
    Is ≈ 0,88 pour un carré RIGIDE, 1,12 pour un carré souple au centre.
    Retourne k en MN/m³. Sol homogène uniquement.
    """
    if E_MPa <= 0 or B <= 0 or nu >= 0.5:
        return 0.0
    return E_MPa / (B * (1.0 - nu ** 2) * Is)


def k_vesic(E_MPa: float, B: float, nu: float = 0.30,
            EI_fondation=None) -> float:
    """
    Vesić (1961). Le terme en racine douzième vaut 0,9 à 1,1 dans les cas
    courants : la formule se réduit alors à k ≈ E/(B(1−ν²)), d'où le
    repli quand la raideur de la fondation n'est pas fournie.
    """
    if E_MPa <= 0 or B <= 0 or nu >= 0.5:
        return 0.0
    base = E_MPa / (B * (1.0 - nu ** 2))
    if not EI_fondation or EI_fondation <= 0:
        return base
    ratio = (E_MPa * 1000.0) * (B ** 4) / EI_fondation
    return 0.65 * (ratio ** (1.0 / 12.0)) * base


def k_terzaghi_taille(k_plaque_MNm3: float, B: float, L=None,
                      nature: str = "sable") -> float:
    """
    Terzaghi (1955) : passage du k mesuré à la PLAQUE de 0,30 m au k
    d'une fondation de largeur B.

        sable  : k = k_plaque · ((B + 0,30)/(2B))²
        argile : k = k_plaque · (0,30/B)

    Correction de forme rectangulaire pour le sable : ·(1 + 0,5·B/L)/1,5.
    Sans cette correction, les valeurs tabulées surestiment k d'un
    facteur 3 à 30 selon la largeur.
    """
    if k_plaque_MNm3 <= 0 or B <= 0:
        return 0.0
    if nature == "sable":
        k = k_plaque_MNm3 * ((B + 0.30) / (2.0 * B)) ** 2
        if L and L > 0:
            k *= (1.0 + 0.5 * min(B / L, 1.0)) / 1.5
        return k
    return k_plaque_MNm3 * (0.30 / B)


def module_oedometrique(E_MPa: float, nu: float = 0.30) -> float:
    """M = E·(1−ν)/((1+ν)(1−2ν)). Diverge quand ν → 0,5 (incompressible)."""
    if E_MPa <= 0 or nu >= 0.5 or nu < 0:
        return 0.0
    return E_MPa * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))


def module_young(M_MPa: float, nu: float = 0.30) -> float:
    """Réciproque de module_oedometrique."""
    if M_MPa <= 0 or nu >= 0.5 or nu < 0:
        return 0.0
    return M_MPa * (1.0 + nu) * (1.0 - 2.0 * nu) / (1.0 - nu)


# =============================================================
#  5. CLASSIFICATION CPT — Robertson
# =============================================================
CN_MAX = 1.7        # plafond du facteur de normalisation (pa/σ'v0)^n


def ic_robertson(qt_kPa: float, fs_kPa: float, sv0_kPa: float,
                 svp0_kPa: float, n_max_iter: int = 30, tol: float = 1e-5,
                 cn_max: float = CN_MAX):
    """
    Indice de comportement Ic, avec exposant n déterminé par itération.

        Qtn = ((qt − σv0)/pa) · CN,   CN = (pa/σ'v0)^n  plafonné à cn_max
        Fr  = fs/(qt − σv0) · 100 %
        Ic  = √[(3,47 − log₁₀Qtn)² + (log₁₀Fr + 1,22)²]
        n   = 0,381·Ic + 0,05·(σ'v0/pa) − 0,15,  plafonné à 1,0

    Le PLAFOND CN = 1,7 est essentiel près de la surface : sous la nappe
    à faible profondeur, σ'v0 devient très petit et (pa/σ'v0)^n explose,
    ce qui fait basculer à tort un sol organique vers une classe limoneuse.
    Vérifié contre groundhog : sans plafond, une tourbe à 2 m sous nappe
    donne Ic = 2,79 (limon argileux, faux) ; avec plafond, Ic = 3,41
    (sol organique, correct).

    Retourne (Ic, Qtn, Fr, n, converge).
    """
    qnet = qt_kPa - sv0_kPa
    # Un frottement nul ou négatif n'est pas une mesure de frottement très
    # faible : c'est une absence de donnée (capteur, colonne manquante).
    # Classer dessus reviendrait à inventer un Fr plancher et à produire un
    # Ic extrême sans le dire. On refuse de classer.
    if qnet <= 0 or fs_kPa is None or fs_kPa <= 0:
        return (None, None, None, None, False)
    svp0 = max(svp0_kPa, 1.0)
    Fr = max(fs_kPa / qnet * 100.0, 1e-3)

    def _qtn(n_exp):
        cn = (PA / svp0) ** n_exp
        if cn_max is not None:
            cn = min(cn, cn_max)
        return max((qnet / PA) * cn, 1e-6)

    n = 1.0
    Ic = 2.5
    converge = False
    for _ in range(n_max_iter):
        Qtn = _qtn(n)
        Ic_new = math.sqrt((3.47 - math.log10(Qtn)) ** 2 +
                           (math.log10(Fr) + 1.22) ** 2)
        n_new = max(0.0, min(1.0, 0.381 * Ic_new + 0.05 * (svp0 / PA) - 0.15))
        if abs(Ic_new - Ic) < tol and abs(n_new - n) < tol:
            Ic, n, converge = Ic_new, n_new, True
            break
        Ic, n = Ic_new, n_new
    return (Ic, _qtn(n), Fr, n, converge)


# Bornes de Ic et libellés (zones SBTn de Robertson).
SBT_ZONES = [
    (1.31, 7, "Sable graveleux à sable dense"),
    (2.05, 6, "Sable propre à sable limoneux"),
    (2.60, 5, "Sable limoneux à limon sableux"),
    (2.95, 4, "Limon argileux à argile limoneuse"),
    (3.60, 3, "Argile silteuse à argile"),
    (99.0, 2, "Sol organique, tourbe"),
]


def sbt_from_ic(Ic):
    """(numéro de zone, libellé) d'après Ic."""
    if Ic is None:
        return (None, "Indéterminé")
    for borne, zone, libelle in SBT_ZONES:
        if Ic < borne:
            return (zone, libelle)
    return (2, "Sol organique, tourbe")


def est_grenu(Ic) -> bool:
    """Ic < 2,60 : comportement plutôt sableux (drainé)."""
    return Ic is not None and Ic < 2.60


# --- Corrélations vers le module de déformation ------------------
#
#  Formule de Robertson (2009) pour le module OEDOMÉTRIQUE :
#
#      M = α_M · (qt − σv0)
#      Ic > 2,2 :  α_M = Qt  si Qt ≤ 14   sinon  α_M = 14
#      Ic ≤ 2,2 :  α_M = 0,0188 · 10^(0,55·Ic + 1,68)
#      avec Qt = (qt − σv0)/σ'v0     ← Qt SIMPLE, PAS Qtn normalisé
#
#  Ces trois points sont vérifiés contre l'implémentation open source
#  groundhog (Université de Gand) : coefficient 0,0188 (et non 0,03 que
#  l'on rencontre dans la littérature secondaire), Qt et non Qtn, seuil
#  à Ic = 2,2 (et non 2,6 qui sépare les classes SBT).
#
#  IMPORTANT — les corrélations CPT → module divergent d'un facteur 2 à 4
#  entre auteurs pour un même sol. On ne renvoie donc JAMAIS une valeur
#  unique mais un ENCADREMENT :
#    · sols grenus : Robertson (haut) recoupé par Schmertmann E'=2,5·qc (bas)
#    · sols fins   : Robertson (haut), avec une marge de 40 % vers le bas
#      (Robertson signale lui-même une surestimation sur certaines argiles).
ALPHA_M_PLAFOND = 14.0
ALPHA_M_A, ALPHA_M_B, ALPHA_M_C = 0.0188, 0.55, 1.68
IC_SEUIL_MODULE = 2.2
SCHMERTMANN_CARRE = 2.5
MARGE_FINS = 0.60


def alpha_M_robertson(Ic, Qt):
    """Multiplicateur α_M de la corrélation du module oedométrique."""
    if Ic is None:
        return None
    if Ic > IC_SEUIL_MODULE:
        if Qt is None:
            return None
        return min(Qt, ALPHA_M_PLAFOND)
    return ALPHA_M_A * 10 ** (ALPHA_M_B * Ic + ALPHA_M_C)


def modules_depuis_cpt(Ic, qt_kPa, sv0_kPa, svp0_kPa, qc_MPa, nu=0.30, Qtn=None):
    """
    (M_bas, M_haut) en MPa — module OEDOMÉTRIQUE encadré.
    Retourne (None, None) si la classification n'a pas abouti.

    Sols fins : la littérature écrit α_M = min(Qtn ; 14) tandis que
    l'implémentation de référence groundhog utilise Qt = (qt−σv0)/σ'v0.
    Les deux coïncident partout où le plafond de normalisation CN = 1,7
    n'est pas actif — c'est-à-dire dès que σ'v0 > 60 kPa environ. Près de
    la surface et sous la nappe, ils divergent fortement (jusqu'à un
    facteur 3 sur une tourbe à 2 m). Là où deux conventions également
    défendables ne s'accordent pas, on ne tranche pas : on ÉLARGIT
    l'encadrement. Ailleurs il reste inchangé, et la borne haute continue
    de coïncider exactement avec groundhog.
    """
    if Ic is None:
        return (None, None)
    qnet_kPa = max(qt_kPa - sv0_kPa, 1.0)
    qnet_MPa = qnet_kPa / 1000.0
    Qt = qnet_kPa / max(svp0_kPa, 1.0)

    if Ic <= IC_SEUIL_MODULE:
        # Sol grenu : Robertson recoupé par Schmertmann (E' = 2,5·qc,
        # module de YOUNG, converti en module oedométrique).
        aM = alpha_M_robertson(Ic, Qt)
        if aM is None:
            return (None, None)
        M_robertson = aM * qnet_MPa
        M_schmertmann = module_oedometrique(SCHMERTMANN_CARRE * max(qc_MPa, 0.0), nu)
        bas, haut = sorted((M_robertson, M_schmertmann))
        return (bas, haut)

    # Sol fin : les deux conventions de normalisation servent de bornes.
    candidats = [min(Qt, ALPHA_M_PLAFOND)]
    if Qtn is not None and Qtn > 0:
        candidats.append(min(Qtn, ALPHA_M_PLAFOND))
    M_haut = max(candidats) * qnet_MPa
    M_bas = MARGE_FINS * min(candidats) * qnet_MPa
    return (M_bas, M_haut)


# =============================================================
#  6. DÉCOUPAGE AUTOMATIQUE EN COUCHES
# =============================================================
def profil_depuis_cpt(points, nappe_m=None, gamma_defaut=19.0,
                      seuil_ic=0.25, h_min=0.50, nu=0.30, qc_refus=None):
    """
    Transforme un sondage brut [(z, qc_MPa, fs_kPa)] en profil de couches
    homogènes, avec classification et modules encadrés.

    Le poids volumique est estimé à partir du comportement (Robertson) ;
    il n'intervient que dans σ'v0, donc son influence sur k est faible.

    qc_refus : au-delà, on considère un refus de pointe (rocher) et on
    ne corrèle plus qc -> module. None pour désactiver.
    """
    if not points:
        return []

    # --- passe 1 : contraintes, Ic, modules point par point ---
    lignes = []
    sv = 0.0
    z_prec = 0.0
    for (z, qc, fs) in points:
        h = max(z - z_prec, 0.0)
        gam = gamma_defaut
        sv += gam * h
        u = max(0.0, z - nappe_m) * GAMMA_W if nappe_m is not None else 0.0
        svp = max(sv - u, 1.0)
        qt_kPa = qc * 1000.0
        Ic = Qtn = None
        if fs is not None and fs > 0:
            Ic, Qtn, Fr, n, _ = ic_robertson(qt_kPa, fs, sv, svp)
        Mb, Mh = modules_depuis_cpt(Ic, qt_kPa, sv, svp, qc, nu, Qtn=Qtn) \
            if Ic is not None else (None, None)
        refus = (qc_refus is not None and qc >= qc_refus)
        lignes.append({"z": z, "qc": qc, "fs": fs, "Ic": Ic, "Qtn": Qtn,
                       "sv0": sv, "svp0": svp, "M_bas": Mb, "M_haut": Mh,
                       "gamma": gam, "refus": refus})
        z_prec = z

    # --- passe 2 : regroupement sur la stabilité de Ic ---
    def clef(l):
        return l["Ic"] if l["Ic"] is not None else -1.0

    groupes = [[lignes[0]]]
    for l in lignes[1:]:
        ref = groupes[-1][-1]
        saut = abs(clef(l) - clef(ref)) > seuil_ic
        epais = (l["z"] - groupes[-1][0]["z"]) >= h_min
        if saut and epais:
            groupes.append([l])
        else:
            groupes[-1].append(l)

    # fusion des groupes trop minces avec le voisin le plus proche en Ic
    fusion = True
    while fusion and len(groupes) > 1:
        fusion = False
        for i, g in enumerate(groupes):
            if (g[-1]["z"] - g[0]["z"]) >= h_min:
                continue
            voisins = []
            if i > 0:
                voisins.append((abs(_moy(groupes[i - 1], "Ic") - _moy(g, "Ic")), i - 1))
            if i < len(groupes) - 1:
                voisins.append((abs(_moy(groupes[i + 1], "Ic") - _moy(g, "Ic")), i + 1))
            if not voisins:
                continue
            _, j = min(voisins)
            groupes[min(i, j)] = groupes[i] + groupes[j] if i < j else groupes[j] + groupes[i]
            del groupes[max(i, j)]
            fusion = True
            break

    # --- passe 3 : synthèse par couche ---
    couches = []
    for g in groupes:
        z0 = couches[-1]["z1"] if couches else 0.0
        z1 = g[-1]["z"]
        h = max(z1 - z0, 1e-6)
        Ic_m = _moy(g, "Ic")
        zone, libelle = sbt_from_ic(Ic_m)
        couches.append({
            "z0": z0, "z1": z1, "h": h,
            "Ic": Ic_m, "zone": zone, "sbt": libelle,
            "qc": _moy(g, "qc"),
            "fs": _moy(g, "fs"),
            "M_bas": _harm(g, "M_bas"), "M_haut": _harm(g, "M_haut"),
            "gamma": gamma_depuis_ic(Ic_m, _moy(g, "qc")),
            "refus": any(l["refus"] for l in g),
            "n_points": len(g),
        })
    return couches


def _moy(groupe, cle):
    vals = [l[cle] for l in groupe if l.get(cle) is not None]
    return (sum(vals) / len(vals)) if vals else None


def _harm(groupe, cle):
    """Moyenne HARMONIQUE : c'est la compressibilité (1/M) qui s'additionne."""
    vals = [l[cle] for l in groupe if l.get(cle) is not None and l[cle] > 0]
    if not vals:
        return None
    return len(vals) / sum(1.0 / v for v in vals)


def gamma_depuis_ic(Ic, qc_MPa=None):
    """
    Poids volumique estimé d'après le comportement. Valeurs prudentes et
    usuelles ; n'intervient que dans σ'v0 (donc dans la profondeur
    d'influence), pas directement dans le tassement.
    """
    if Ic is None:
        return 19.0
    if Ic < 1.31:
        return 21.0
    if Ic < 2.05:
        return 20.0
    if Ic < 2.60:
        return 19.5
    if Ic < 2.95:
        return 19.0
    if Ic < 3.60:
        return 18.0
    return 14.0


# =============================================================
#  7. PRÉPARATION DU PROFIL POUR LE CALCUL DE TASSEMENT
# =============================================================
def couches_pour_calcul(couches, cle="M_bas"):
    """Adapte un profil issu du CPT au format attendu par tassement()."""
    out = []
    for c in couches:
        out.append({
            "h": c.get("h", 0.0),
            "gamma": c.get("gamma", 19.0),
            "M": c.get(cle),
            "nom": c.get("sbt") or c.get("type") or "—",
        })
    return out


def bilan_k(couches, B, L, q_kPa, D=0.0, nappe_m=None, critere=0.20,
            dz=0.05, q_net=True):
    """
    Calcul complet : k encadré (bornes basse et haute des modules) aux
    quatre positions de la fondation. C'est la fonction de haut niveau
    appelée par l'interface.
    """
    prof_bas = couches_pour_calcul(couches, "M_bas")
    prof_haut = couches_pour_calcul(couches, "M_haut")

    res = {"positions": {}, "B": B, "L": L, "q_kPa": q_kPa, "D": D,
           "critere": critere}
    for pos in POSITIONS:
        # module BAS  -> tassement grand -> k FAIBLE
        rb = tassement(prof_bas, B, L, q_kPa, D=D, nappe_m=nappe_m,
                       critere=critere, dz=dz, position=pos, q_net=q_net)
        rh = tassement(prof_haut, B, L, q_kPa, D=D, nappe_m=nappe_m,
                       critere=critere, dz=dz, position=pos, q_net=q_net)
        res["positions"][pos] = {
            "k_bas_MNm3": min(rb["k_MNm3"], rh["k_MNm3"]),
            "k_haut_MNm3": max(rb["k_MNm3"], rh["k_MNm3"]),
            "w_bas_mm": min(rb["w_mm"], rh["w_mm"]),
            "w_haut_mm": max(rb["w_mm"], rh["w_mm"]),
            "z_influence": max(rb["z_influence"], rh["z_influence"]),
            "detail_bas": rb, "detail_haut": rh,
        }
    c = res["positions"]["centre"]
    res["k_bas_MNm3"] = c["k_bas_MNm3"]
    res["k_haut_MNm3"] = c["k_haut_MNm3"]
    res["w_bas_mm"] = c["w_bas_mm"]
    res["w_haut_mm"] = c["w_haut_mm"]
    res["z_influence"] = c["z_influence"]
    res["q_net_kPa"] = c["detail_bas"]["q_net_kPa"]
    return res

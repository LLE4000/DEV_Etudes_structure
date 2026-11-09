# -*- coding: utf-8 -*-
"""Longueurs d'ANCRAGE et de RECOUVREMENT des barres d'armature — EN 1992-1-1:2004 §8.4.2 à 8.4.4 et §8.7.3, avec les
coefficients partiels de l'annexe belge (γ_c = 1,5, γ_s = 1,15). Rien n'est recopié d'un tableau : chaque longueur est
calculée, et ``tests/test_bibliotheque_moteur.py`` la confronte aux tableaux 24 et 25 de la NIT 260 (CSTC, 2017), qui
sont une approximation arrondie au centimètre de ce même calcul (B500, bonnes conditions d'adhérence). La NIT a
arrondi au plus près et pris f_ctk,0,05 à sa manière (elle se dit elle-même approximative) : le moteur suit la relation
analytique du tableau 3.1 (0,7 · f_ctm) et arrondit au centimètre SUPÉRIEUR pour le plan. Écart mesuré le 08/11/2026
sur les 154 cases : la case arrondie s'écarte de la NIT d'au plus 20 mm jusqu'à Ø 32 et 50 mm à Ø 50 ; la longueur
calculée d'au plus 1,5 % une fois l'arrondi de la NIT (5 mm) retiré — le test le borne.

Le calcul (chaque étape est un TERME nommé de ``calculer``) ::

    f_ctd     = α_ct · f_ctk,0,05 / γ_c              §3.1.6(2)P, f_ctk,0,05 limité à C60/75 pour l'adhérence (§8.4.2(2))
    f_bd      = 2,25 · η1 · η2 · f_ctd               §8.4.2(2) : η1 = 1,0 (bonnes conditions) ou 0,7 ; η2 = 1,0 (Ø ≤ 32)
                                                      ou (132 − Ø)/100
    σ_sd      = taux · f_yd,  f_yd = f_yk / γ_s      la contrainte de calcul de la barre à l'origine de l'ancrage
    l_b,rqd   = (Ø/4) · σ_sd / f_bd                  §8.4.3(2)
    l_bd      = α1 α2 α3 α4 α5 · l_b,rqd ≥ l_b,min   §8.4.4(1), α2·α3·α5 ≥ 0,7
    l_b,min   = max(0,3 l_b,rqd ; 10 Ø ; 100 mm)     en traction ; max(0,6 l_b,rqd ; 10 Ø ; 100 mm) en compression
    l_0       = α1 α2 α3 α5 α6 · l_b,rqd ≥ l_0,min   §8.7.3(1), α6 = (ρ1/25)^0,5 borné à [1,0 ; 1,5] (tableau 8.3)
    l_0,min   = max(0,3 α6 l_b,rqd ; 15 Ø ; 200 mm)

Les α (tableau 8.2) : α1 forme de la barre (0,7 pour un crochet, un coude ou une boucle tendus si c_d > 3 Ø) ;
α2 enrobage (1 − 0,15 (c_d − Ø)/Ø barre droite, 1 − 0,15 (c_d − 3 Ø)/Ø barre courbe, dans [0,7 ; 1,0]) ;
α3 confinement par des armatures transversales (1 − K λ, dans [0,7 ; 1,0]) ; α4 barre transversale soudée (0,7) ;
α5 pression transversale (1 − 0,04 p, dans [0,7 ; 1,0]). En compression, α1 = α2 = α3 = α5 = 1,0.

Domaine : Ø 6 à 50 mm ; au-delà de 32 mm les barres sont de « gros diamètre » (§8.8 : recouvrement non admis, sauf
dans une section d'au moins 1 m ou sous 80 % de la résistance de calcul — note 4 du tableau 25 de la NIT 260) ; la
NIT elle-même dit ses tableaux approximatifs au-delà de 32 mm. Précontrainte, paquets de barres : hors du moteur.
"""
import math

from beton.donnees import ec2 as EC

SOURCE = "EN 1992-1-1:2004 §8.4.2–8.4.4 (ancrage), §8.7.3 (recouvrement), §8.8 (Ø > 32 mm) ; NBN EN 1992-1-1 ANB (γ_c 1,5, γ_s 1,15)"
CONTROLE = "NIT 260 (CSTC, mars 2017), tableaux 24 et 25 — B500, bonnes conditions d'adhérence, arrondi au cm"

GAMMA_C = 1.5                 # tableau 2.1N, ELU durable et transitoire
GAMMA_S = EC.GAMMA_S          # 1,15
ALPHA_CT = 1.0                # §3.1.6(2)P, valeur recommandée
FCK_LIMITE_FBD = 60.0         # §8.4.2(2) : f_ctk,0,05 limité à la valeur de C60/75 pour f_bd
PHI_ETA2 = 32.0               # §8.4.2(2) : η2 = 1,0 jusqu'à Ø 32, (132 − Ø)/100 au-delà
PHI_GRANDE = 32               # §8.8 : « gros diamètre » au-delà
ETA1 = {"bonnes": 1.0, "mediocres": 0.7}
CONDITIONS = (("bonnes", "bonnes conditions d'adhérence"), ("mediocres", "autres conditions (η1 = 0,7 : × 1/0,7 ≈ 1,43)"))
SOLLICITATIONS = (("traction", "barre tendue"), ("compression", "barre comprimée"))
FORMES = (("droit", "barre droite"), ("courbe", "crochet, coude ou boucle"))
DIAMETRES = (6, 8, 10, 12, 14, 16, 20, 25, 32, 40, 50)                       # les lignes des tableaux de la NIT 260
CLASSES = ("C20/25", "C25/30", "C30/37", "C35/45", "C40/50", "C45/55", "C50/60")   # leurs colonnes
NUANCES = (400, 500)
ALPHA_MIN = 0.7               # borne basse de α2, α3, α5 et de leur produit
ALPHA6_MIN, ALPHA6_MAX = 1.0, 1.5
LB_MIN = {"traction": (0.3, 10.0, 100.0), "compression": (0.6, 10.0, 100.0)}   # (part de l_b,rqd, × Ø, mm)
L0_MIN = (0.3, 15.0, 200.0)                                                     # (part de α6·l_b,rqd, × Ø, mm)
K_CONFINEMENT = (0.0, 0.05, 0.1)   # figure 8.4 : barre hors d'un cadre 0 ; sur la branche droite 0,05 ; à l'angle 0,1

DEFAUTS = {"phi": 16, "classe": "C30/37", "fyk": 500, "conditions": "bonnes", "sollicitation": "traction",
           "forme": "droit", "c_d": None, "taux": 1.0, "K": 0.0, "lam": 0.0, "soude": False, "p": 0.0, "rho1": 100.0}


def fck_de(classe: str) -> float:
    """« C30/37 » → 30 (MPa). Une classe mal écrite est refusée."""
    try:
        a, b = str(classe).upper().lstrip("C").split("/")
        fck, cube = float(a), float(b)
    except (ValueError, AttributeError):
        raise ValueError(f"classe de béton illisible « {classe} » (attendu : C30/37)")
    if not 12 <= fck <= 90 or cube <= fck:
        raise ValueError(f"classe de béton hors du tableau 3.1 « {classe} »")
    return fck


def _nombre(x, nom, mini=None, maxi=None) -> float:
    if isinstance(x, bool):
        raise ValueError(f"{nom} : un nombre est attendu")
    try:
        v = float(str(x).replace(",", ".")) if not isinstance(x, (int, float)) else float(x)
    except ValueError:
        raise ValueError(f"{nom} : nombre illisible « {x} »")
    if not math.isfinite(v):
        raise ValueError(f"{nom} : nombre illisible « {x} »")
    if mini is not None and v < mini:
        raise ValueError(f"{nom} : au moins {_g(mini)}")
    if maxi is not None and v > maxi:
        raise ValueError(f"{nom} : au plus {_g(maxi)}")
    return v


def _choix(x, nom, permis):
    if x not in permis:
        raise ValueError(f"{nom} : « {x} » n'est pas permis ({', '.join(str(p) for p in permis)})")
    return x


def normaliser(params: dict) -> dict:
    """Les paramètres relus strictement (chaque refus en français) ; les absents prennent leur défaut."""
    p = {**DEFAUTS, **(params or {})}
    out = {
        "phi": _nombre(p["phi"], "Ø", 6, 50),
        "classe": str(p["classe"]),
        "fyk": int(_choix(int(_nombre(p["fyk"], "f_yk", 400, 600)), "f_yk", NUANCES)),
        "conditions": _choix(p["conditions"], "conditions d'adhérence", tuple(k for k, _ in CONDITIONS)),
        "sollicitation": _choix(p["sollicitation"], "sollicitation", tuple(k for k, _ in SOLLICITATIONS)),
        "forme": _choix(p["forme"], "forme de l'ancrage", tuple(k for k, _ in FORMES)),
        "c_d": None if p["c_d"] in (None, "") else _nombre(p["c_d"], "c_d", 0, 1000),
        "taux": _nombre(p["taux"], "σ_sd / f_yd", 0.0, 1.0),
        "K": _nombre(p["K"], "K", 0, 0.1),
        "lam": _nombre(p["lam"], "λ", 0, 100),
        "soude": bool(p["soude"]) if not isinstance(p["soude"], str) else p["soude"].lower() in ("1", "true", "oui", "on"),
        "p": _nombre(p["p"], "pression transversale p", 0, 100),
        "rho1": _nombre(p["rho1"], "proportion de barres recouvertes", 0, 100),
    }
    fck_de(out["classe"])
    return out


# ============================================================ le calcul, terme par terme
def adherence(fck: float, phi: float, conditions: str) -> dict:
    """f_ctk,0,05 (limité à C60/75), f_ctd, η1, η2 et f_bd (MPa) — §3.1.6, §8.4.2."""
    fck_bd = min(fck, FCK_LIMITE_FBD)
    fctk = EC.fctk_005(fck_bd)
    fctd = ALPHA_CT * fctk / GAMMA_C
    eta1 = ETA1[conditions]
    eta2 = 1.0 if phi <= PHI_ETA2 else (132.0 - phi) / 100.0
    return {"fck_bd": fck_bd, "fctk_005": fctk, "fctd": fctd, "eta1": eta1, "eta2": eta2, "fbd": 2.25 * eta1 * eta2 * fctd}


def alphas(p: dict) -> dict:
    """Les coefficients α1 à α5 du tableau 8.2 pour les paramètres ``p`` (normalisés), et leur produit borné."""
    phi, cd = p["phi"], p["c_d"]
    tendue = p["sollicitation"] == "traction"
    if not tendue:
        a1 = a2 = a3 = a5 = 1.0
    else:
        courbe = p["forme"] == "courbe"
        a1 = 0.7 if (courbe and cd is not None and cd > 3 * phi) else 1.0
        if cd is None:
            a2 = 1.0
        else:
            a2 = 1.0 - 0.15 * ((cd - 3 * phi) if courbe else (cd - phi)) / phi
            a2 = min(1.0, max(ALPHA_MIN, a2))
        a3 = min(1.0, max(ALPHA_MIN, 1.0 - p["K"] * p["lam"]))
        a5 = min(1.0, max(ALPHA_MIN, 1.0 - 0.04 * p["p"]))
    a4 = 0.7 if p["soude"] else 1.0
    prod235 = max(ALPHA_MIN, a2 * a3 * a5)                     # §8.4.4(1) : α2·α3·α5 ≥ 0,7
    return {"alpha1": a1, "alpha2": a2, "alpha3": a3, "alpha4": a4, "alpha5": a5, "produit_235": prod235}


def alpha6(rho1: float) -> float:
    """α6 = (ρ1/25)^0,5 dans [1,0 ; 1,5] — §8.7.3(1), tableau 8.3 (< 25 % : 1,0 ; 33 % : 1,15 ; 50 % : 1,4 ; > 50 % : 1,5)."""
    return min(ALPHA6_MAX, max(ALPHA6_MIN, math.sqrt(max(rho1, 0.0) / 25.0)))


def calculer(params: dict) -> dict:
    """Tout le calcul : ``entree`` (paramètres normalisés), ``adherence``, ``sigma_sd``, ``lb_rqd``, ``alphas``,
    ``lbd`` (``brut``, ``min``, ``valeur``), ``l0`` (``alpha6``, ``brut``, ``min``, ``valeur``), ``termes`` (la
    justification, dans l'ordre), ``avertissements`` ([{code, message}]). Longueurs en mm, non arrondies."""
    p = normaliser(params)
    fck = fck_de(p["classe"])
    ad = adherence(fck, p["phi"], p["conditions"])
    fyd = p["fyk"] / GAMMA_S
    sigma = p["taux"] * fyd
    lb_rqd = p["phi"] / 4.0 * sigma / ad["fbd"]
    al = alphas(p)
    part, kphi, plancher = LB_MIN[p["sollicitation"]]
    lbd_brut = al["alpha1"] * al["produit_235"] * al["alpha4"] * lb_rqd
    lbd_min = max(part * lb_rqd, kphi * p["phi"], plancher)
    lbd = max(lbd_brut, lbd_min)
    a6 = alpha6(p["rho1"])
    l0_brut = al["alpha1"] * al["produit_235"] * a6 * lb_rqd
    l0_min = max(L0_MIN[0] * a6 * lb_rqd, L0_MIN[1] * p["phi"], L0_MIN[2])
    l0 = max(l0_brut, l0_min)
    res = {"entree": p, "fck": fck, "adherence": ad, "fyd": fyd, "sigma_sd": sigma, "lb_rqd": lb_rqd, "alphas": al,
           "lbd": {"brut": lbd_brut, "min": lbd_min, "valeur": lbd, "gouverne_min": lbd_min > lbd_brut},
           "l0": {"alpha6": a6, "brut": l0_brut, "min": l0_min, "valeur": l0, "gouverne_min": l0_min > l0_brut}}
    res["termes"] = termes(res)
    res["avertissements"] = avertissements(res)
    return res


def _g(x) -> str:
    """Un nombre dans une règle, en français (« 1,15 »)."""
    return f"{x:g}".replace(".", ",")


def termes(res: dict) -> list:
    """La justification : [{cle, symbole, nom, valeur, unite, regle}] dans l'ordre du calcul."""
    p, ad, al = res["entree"], res["adherence"], res["alphas"]
    phi = p["phi"]
    out = []

    def t(cle, symbole, nom, valeur, unite, regle):
        out.append({"cle": cle, "symbole": symbole, "nom": nom, "valeur": valeur, "unite": unite, "regle": regle})

    lim = f" (limité à C60/75 : f_ck = {_g(ad['fck_bd'])})" if ad["fck_bd"] < res["fck"] else ""
    t("fctk", "f_ctk,0,05", "traction caractéristique, fractile 5 %", ad["fctk_005"], "MPa", f"0,7 · f_ctm — tableau 3.1{lim}")
    t("fctd", "f_ctd", "traction de calcul", ad["fctd"], "MPa", f"α_ct · f_ctk,0,05 / γ_c = {_g(ALPHA_CT)} · f_ctk,0,05 / {_g(GAMMA_C)} — §3.1.6(2)P")
    t("eta1", "η1", "conditions d'adhérence", ad["eta1"], "", "1,0 bonnes conditions, 0,7 autres — §8.4.2(2), figure 8.2")
    t("eta2", "η2", "diamètre", ad["eta2"], "", "1,0 pour Ø ≤ 32 mm, (132 − Ø)/100 au-delà — §8.4.2(2)")
    t("fbd", "f_bd", "contrainte d'adhérence de calcul", ad["fbd"], "MPa", "2,25 · η1 · η2 · f_ctd — §8.4.2(2), éq. (8.2)")
    t("fyd", "f_yd", "limite d'élasticité de calcul", res["fyd"], "MPa", f"f_yk / γ_s = {p['fyk']} / {_g(GAMMA_S)} — §3.2.7")
    t("sigma", "σ_sd", "contrainte de la barre à l'origine de l'ancrage", res["sigma_sd"], "MPa",
      f"{_g(p['taux'])} · f_yd" + (" (barre pleinement sollicitée)" if p["taux"] >= 1 else " (taux σ_sd / f_yd, ou A_s,req / A_s,prov)") + " — §8.4.3(2)")
    t("lb_rqd", "l_b,rqd", "longueur d'ancrage de référence", res["lb_rqd"], "mm", "(Ø/4) · σ_sd / f_bd — §8.4.3(2), éq. (8.3)")
    forme = dict(FORMES)[p["forme"]]
    cd = "c_d non précisé" if p["c_d"] is None else f"c_d = {_g(p['c_d'])} mm"
    t("alpha1", "α1", "forme de la barre", al["alpha1"], "", f"{forme}, {cd} — tableau 8.2 (0,7 si crochet, coude ou boucle tendus avec c_d > 3 Ø)")
    t("alpha2", "α2", "enrobage", al["alpha2"], "", f"{cd} — tableau 8.2, borné à [0,7 ; 1,0]")
    t("alpha3", "α3", "confinement par armatures transversales", al["alpha3"], "", f"1 − K·λ, K = {_g(p['K'])}, λ = {_g(p['lam'])} — tableau 8.2, figure 8.4")
    t("alpha4", "α4", "barre transversale soudée", al["alpha4"], "", "0,7 si une barre transversale soudée (§8.6) — tableau 8.2")
    t("alpha5", "α5", "pression transversale", al["alpha5"], "", f"1 − 0,04 · p, p = {_g(p['p'])} MPa — tableau 8.2, borné à [0,7 ; 1,0]")
    if al["produit_235"] > al["alpha2"] * al["alpha3"] * al["alpha5"] + 1e-12:
        t("prod", "α2·α3·α5", "produit borné", al["produit_235"], "", "≥ 0,7 — §8.4.4(1), éq. (8.4)")
    part, kphi, pl = LB_MIN[p["sollicitation"]]
    lb = res["lbd"]
    t("lbd_brut", "α1…α5 · l_b,rqd", "longueur d'ancrage avant minimum", lb["brut"], "mm", "α1 · α2 · α3 · α4 · α5 · l_b,rqd — éq. (8.4)")
    t("lbd_min", "l_b,min", "longueur d'ancrage minimale", lb["min"], "mm", f"max({_g(part)} l_b,rqd ; {_g(kphi)} Ø ; {_g(pl)} mm) — §8.4.4(1), éq. (8.6){'' if p['sollicitation'] == 'traction' else ' (8.7)'}")
    t("lbd", "l_bd", "LONGUEUR D'ANCRAGE DE CALCUL", lb["valeur"], "mm", "max(α1…α5 · l_b,rqd ; l_b,min) — §8.4.4(1)" + (" ; le minimum gouverne" if lb["gouverne_min"] else ""))
    l0 = res["l0"]
    t("alpha6", "α6", "proportion de barres recouvertes dans la section", l0["alpha6"], "", f"(ρ1/25)^0,5 dans [1,0 ; 1,5], ρ1 = {_g(p['rho1'])} % — §8.7.3(1), tableau 8.3")
    t("l0_brut", "α1 α2 α3 α5 α6 · l_b,rqd", "longueur de recouvrement avant minimum", l0["brut"], "mm", "éq. (8.10)")
    t("l0_min", "l_0,min", "longueur de recouvrement minimale", l0["min"], "mm", f"max({_g(L0_MIN[0])} α6 l_b,rqd ; {_g(L0_MIN[1])} Ø ; {_g(L0_MIN[2])} mm) — éq. (8.11)")
    t("l0", "l_0", "LONGUEUR DE RECOUVREMENT DE CALCUL", l0["valeur"], "mm", "max(α1 α2 α3 α5 α6 · l_b,rqd ; l_0,min) — §8.7.3(1)" + (" ; le minimum gouverne" if l0["gouverne_min"] else ""))
    return out


def avertissements(res: dict) -> list:
    p = res["entree"]
    out = []
    if p["phi"] > PHI_GRANDE:
        out.append({"code": "gros_diametre", "message": (
            f"Ø {_g(p['phi'])} > 32 mm : barre de gros diamètre (§8.8) — recouvrement non admis, sauf dans une section "
            "d'au moins 1 m ou sous 80 % de la résistance de calcul ; armatures de confinement en zone d'ancrage.")})
    if p["forme"] == "courbe" and p["c_d"] is None and p["sollicitation"] == "traction":
        out.append({"code": "cd_inconnu", "message": "c_d non précisé : α1 = α2 = 1,0 (aucune réduction) — préciser c_d pour en profiter."})
    if p["conditions"] == "mediocres":
        out.append({"code": "adherence_mediocre", "message": (
            "Conditions d'adhérence « autres » (η1 = 0,7) : barres inclinées de moins de 45°, à plus de 250 mm du fond "
            "d'un élément de plus de 600 mm, ou dans les 300 mm supérieurs — figure 8.2.")})
    if res["lbd"]["gouverne_min"]:
        out.append({"code": "lbd_min", "message": "La longueur d'ancrage minimale gouverne (barre peu sollicitée ou réductions fortes)."})
    return out


# ============================================================ les tableaux de l'écran (calculés)
def arrondi_cm(mm: float) -> int:
    """Une longueur arrondie au centimètre SUPÉRIEUR (mm), pour un plan : la valeur sûre (434,8 → 440 ; 480,0 → 480)."""
    return int(math.ceil(mm / 10.0 - 1e-9)) * 10


def tableau(grandeur: str = "lbd", conditions: str = "bonnes", fyk: int = 500, rho1: float = 100.0) -> dict:
    """Le tableau Ø × classe de ``grandeur`` (« lbd » ou « l0 ») : barre tendue pleinement sollicitée, droite, sans
    réduction (α1 à α5 = 1), ``rho1`` % recouvert pour l_0. ``cellules[Ø][classe]`` = {mm, cm (arrondi supérieur),
    k (en Ø)} ; ``coefficients[classe]`` = l / Ø pour Ø ≤ 32."""
    if grandeur not in ("lbd", "l0"):
        raise ValueError("grandeur : « lbd » ou « l0 »")
    cellules, coefs = {}, {}
    for phi in DIAMETRES:
        cellules[phi] = {}
        for cl in CLASSES:
            r = calculer({"phi": phi, "classe": cl, "fyk": fyk, "conditions": conditions, "rho1": rho1})
            v = r[grandeur]["valeur"]
            cellules[phi][cl] = {"mm": v, "cm": arrondi_cm(v), "k": v / phi}
            if phi == 16:
                coefs[cl] = v / phi
    return {"grandeur": grandeur, "conditions": conditions, "fyk": fyk, "rho1": rho1, "diametres": list(DIAMETRES),
            "classes": list(CLASSES), "cellules": cellules, "coefficients": coefs, "source": SOURCE, "controle": CONTROLE}

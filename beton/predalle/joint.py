# -*- coding: utf-8 -*-
"""Prédalle en béton armé — le JOINT entre deux prédalles et ses BARRES DE COUTURE (lot 5 ; décisions du propriétaire
du 13/11/2026, docs/site/DALLES_FUSION.md § 10).

LE JOINT. Les prédalles portent dans la direction PRINCIPALE ; deux prédalles voisines se touchent par un joint
parallèle à cette direction. Au joint, la peau est coupée : la base SECONDAIRE, posée dans la peau en usine, s'y
interrompt. La continuité de la direction secondaire est rendue par des BARRES DE COUTURE posées SUR les prédalles, au
chantier, à cheval sur le joint ; elles recouvrent la base secondaire de chaque côté.

La direction secondaire est donc VÉRIFIÉE DEUX FOIS : en pleine prédalle (la face inférieure secondaire du moteur des
dalles, base dans la peau) et AU JOINT (ici) — les MÊMES formules (beton/dalle/moteur.py, une seule copie), d'autres
armatures : les barres de couture à la place de la base secondaire, posées sur la prédalle, les renforts secondaires
gardés tels quels.

LES RÈGLES
- la couture d'une section, ``section["couture"]`` : ``None`` — AUTOMATIQUE, la base secondaire inférieure de la
  section reprise en barres (un treillis Øl/Øt/el/et donne ses porteurs : Øl tous les el, le Ø relevé au premier Ø du
  moteur qui ne soit pas plus petit, si le fil n'en est pas un ; des barres Ø/e ou « n barres » sont reprises telles
  quelles) — ou ``{"d": Ø, "esp": e}``, des barres Ø/e saisies ;
- leur position RÉELLE : sur la prédalle, h_p + Ø/2 (beton/position.py) — POSÉES SUR les renforts principaux inférieurs
  qui reposent eux-mêmes sur la prédalle (la direction secondaire sur la principale, la règle du site : leur d en est
  plus petit, le côté sûr), h_p + Ø_principal + Ø/2 ; l'axe de calcul, ⌈réel + jeu de pose⌉ au pas d'arrondi, comme
  toute couche ;
- au joint, la face inférieure secondaire = la couture + les renforts secondaires (à leur position) ; un y_G IMPOSÉ à
  la face décrit la pleine prédalle : au joint, y_G est calculé ;
- la HAUTEUR : le joint est une famille de plus de h_min = h_u,min(M) + y_G ; la plus exigeante est retenue (la ligne
  « Hauteur » le dit : « famille joint X ») ;
- le calcul de PLEINE PRÉDALLE (celui de l'écran Streamlit et de la note PDF, que leurs tests de parité comparent) est
  GARDÉ : les grandeurs du moteur des dalles ne changent pas (``h_min``, ``etat_hauteur``… restent celles de la pleine
  prédalle ; le joint a ses clés, ``h_min_joint``, ``etat_joint``…) et ``section["pleine_predalle"]`` /
  ``r["pleine_predalle"]`` gardent le statut, le taux et le gouvernant d'avant le joint, et la vérification de hauteur ;
- la LONGUEUR de chaque barre : la longueur de RECOUVREMENT l₀ (EN 1992-1-1 §8.7.3, beton/bibliotheque/ancrages :
  bonnes conditions d'adhérence, barre droite tendue, σ_sd = f_yd, ρ1 = 100 % des barres recouvertes dans la même
  section, d'où α6 = 1,5), ARRONDIE au 5 cm SUPÉRIEUR de chaque côté du joint (Ø10, C30/37, B500 : l₀ = 536,1 mm →
  55 cm, 110 cm en tout) — le propriétaire vérifiera la norme (« la longueur de recouvrement par défaut »).

API
---
    couche_couture(section, secondaire) -> dict      la couture de la section, couche du moteur des dalles (sans axe)
    reglage_couture(section, secondaire, b) -> dict  la même dite en barres Ø/e, {d, esp} (ce que l'écran règle)
    normaliser_couture(x, chemin) -> None | dict     la saisie contrôlée (ValueError en français)
    lire_pas(x, chemin) -> float                     le pas d'arrondi de la longueur, contrôlé (l'un de PAS_COUTURE)
    longueur(d, classe, fyk, pas) -> dict            l₀ (mm, non arrondie), le côté arrondi au pas (cm), le total (cm), les
                                                     termes
    joints(p, figee, r, formule_axe) -> None         ajoute au calcul ``r`` de la prédalle ``p`` (normalisée ; ``figee`` :
                                                     la dalle que le moteur des dalles a calculée ; ``formule_axe`` : la
                                                     formule de l'axe d'une couche, celle du moteur prédalle) la
                                                     vérification du joint de chaque section, la hauteur du joint, les
                                                     statuts et le gouvernant recalculés, les avertissements du joint
"""
import copy
import math

from beton import position as P
from beton import treillis as TR
from beton.bibliotheque import ancrages as AN
from beton.dalle import entree as ED
from beton.dalle import moteur as MD

CLE = "joint"
# la longueur de chaque côté arrondie au PAS supérieur : 5 cm par défaut (décision du propriétaire du 13/11/2026 :
# « Ø10 : 53,6 → 55 cm, 110 cm en tout ») ; depuis le lot 6 (14/11/2026), une OPTION de la note (« pas_couture »,
# beton/predalle/entree.py) — le gabarit du bureau peut la fixer à 10 cm (« au 5 cm près ou à 10 cm près ») ou à 1 cm
PAS_COUTURE = (1.0, 5.0, 10.0)
PAS_LONGUEUR = 5.0
RHO1 = 100.0                 # % : toutes les barres de couture recouvertes dans la même section (α6 = 1,5)
TOL = 1e-6                   # cm : un renfort principal « repose sur la prédalle » quand son dessous y est, à TOL près
CLES_COUTURE = ("d", "esp")


def titre(secondaire: str) -> str:
    return f"Joint des prédalles — dir. {secondaire.upper()} (barres de couture)"


# ============================================================ la couture d'une section
def _diametre_au_moins(d: float) -> int:
    """Le premier Ø du moteur qui ne soit pas plus petit que ``d`` (un fil de treillis de 5 mm donne des barres Ø6)."""
    return next((x for x in ED.DIAMETRES if x >= d - 1e-9), ED.DIAMETRES[-1])


def couche_couture(section: dict, secondaire: str) -> dict:
    """La couture de la section, couche du moteur des dalles (``dist`` : None — joints() la pose sur la prédalle) :
    saisie (barres Ø/e), sinon AUTOMATIQUE — la base secondaire inférieure reprise en barres (en-tête du module)."""
    c = section.get("couture")
    if c is not None:
        return {"type": "barres", "treillis": TR.TREILLIS_DEFAUT, "d": int(c["d"]), "esp": int(c["esp"]), "n": 3, "dist": None}
    base = section["faces"][f"inf_{secondaire}"]["couches"][0]
    if base["type"] == "treillis":
        t = TR.parse_designation(base["treillis"])
        return {"type": "barres", "treillis": base["treillis"], "d": _diametre_au_moins(t[0]), "esp": int(t[2]),
                "n": base["n"], "dist": None}
    return {"type": base["type"], "treillis": base["treillis"], "d": base["d"], "esp": base["esp"], "n": base["n"], "dist": None}


def reglage_couture(section: dict, secondaire: str, b: float) -> dict:
    """La couture de la section dite en barres Ø/e — ce que l'écran règle : la saisie, ou l'automatique (« n barres » :
    l'espacement de n barres dans la bande b, arrondi au mm et borné aux espacements admis — l'écran l'affiche ; le calcul
    garde les n barres)."""
    c = couche_couture(section, secondaire)
    esp = c["esp"]
    if c["type"] == "n_barres":
        esp = min(ED.ESP_MAX, max(ED.ESP_MIN, int(round(float(b) * 10.0 / max(1, c["n"])))))
    return {"d": int(c["d"]), "esp": int(esp)}


def normaliser_couture(x, chemin: str = "couture"):
    """La couture saisie, contrôlée : ``None`` (automatique) ou ``{"d", "esp"}`` — un Ø du moteur, un espacement entier
    de 25 à 500 mm. Une autre clé, un autre type : ValueError (le chemin du champ dans le message)."""
    if x is None:
        return None
    if not isinstance(x, dict):
        raise ValueError(f"{chemin} : null (automatique) ou {{\"d\": Ø, \"esp\": e}} attendu (reçu {x!r})")
    inconnues = sorted(str(k) for k in x if k not in CLES_COUTURE)
    if inconnues:
        raise ValueError(f"{chemin} : clé(s) inconnue(s) {', '.join(inconnues)} — clés admises : d, esp")
    d, esp = x.get("d"), x.get("esp")
    if isinstance(d, bool) or not isinstance(d, (int, float)) or d not in ED.DIAMETRES:
        raise ValueError(f"{chemin}.d : un Ø parmi {', '.join(str(v) for v in ED.DIAMETRES)} attendu (reçu {d!r})")
    if (isinstance(esp, bool) or not isinstance(esp, (int, float)) or esp != esp or esp != int(esp)
            or not ED.ESP_MIN <= esp <= ED.ESP_MAX):
        raise ValueError(f"{chemin}.esp : un espacement entier de {ED.ESP_MIN} à {ED.ESP_MAX} mm attendu (reçu {esp!r})")
    return {"d": int(d), "esp": int(esp)}


# ============================================================ la longueur des barres
def lire_pas(x, chemin: str = "pas_couture") -> float:
    """Le pas d'arrondi de la longueur des barres de couture (cm), CONTRÔLÉ : l'un de PAS_COUTURE (1, 5 ou 10 cm) ; un
    nombre (pas un booléen), sinon ValueError avec le chemin du champ — jamais de substitution silencieuse."""
    vs = [f"{v:g}" for v in PAS_COUTURE]
    permis = f"{', '.join(vs[:-1])} ou {vs[-1]}"
    if isinstance(x, bool) or not isinstance(x, (int, float)) or x != x or float(x) not in PAS_COUTURE:
        raise ValueError(f"{chemin} : le pas d'arrondi de la longueur des barres de couture vaut {permis} cm (reçu {x!r})")
    return float(x)


def longueur(d: float, classe: str, fyk: int, pas: float = PAS_LONGUEUR) -> dict:
    """La longueur d'une barre de couture : le recouvrement l₀ (mm, beton/bibliotheque/ancrages, non arrondi), le côté
    arrondi au PAS supérieur (cm ; 5 cm par défaut, ``pas_couture`` de la note), le total des deux côtés (cm), les termes
    du calcul et ses avertissements."""
    pas = lire_pas(pas, "pas")
    res = AN.calculer({"phi": float(d), "classe": classe, "fyk": int(fyk), "rho1": RHO1})
    l0 = res["l0"]["valeur"]
    cote = math.ceil(l0 / 10.0 / pas - 1e-9) * pas
    return {"l0": l0, "cote": cote, "total": 2.0 * cote, "pas": pas, "lb_rqd": res["lb_rqd"], "fbd": res["adherence"]["fbd"],
            "sigma_sd": res["sigma_sd"], "alpha6": res["l0"]["alpha6"], "brut": res["l0"]["brut"], "min": res["l0"]["min"],
            "gouverne_min": res["l0"]["gouverne_min"], "rho1": RHO1, "avertissements": res["avertissements"]}


def formules_longueur(d: float, lg: dict) -> list:
    """Les trois formules de la longueur, chacune recalculable depuis ses seuls termes : l_b,rqd = Ø/4 · σ_sd / f_bd ;
    l₀ = max(α6 · l_b,rqd ; l₀,min), l₀,min = max(0,3 · α6 · l_b,rqd ; 15 Ø ; 200 mm) — les α1, α2, α3, α5 valent 1 (barre
    droite, enrobage non compté, ni cadre ni pression) ; le côté = ⌈l₀⌉ au pas (cm), le total = 2 côtés."""
    return [_f("lb_rqd", [_t("phi", float(d), "mm"), _t("sigma_sd", lg["sigma_sd"], "N/mm²"), _t("fbd", lg["fbd"], "N/mm²")],
               lg["lb_rqd"], "mm"),
            _f("l0", [_t("phi", float(d), "mm"), _t("alpha6", lg["alpha6"]), _t("lb_rqd", lg["lb_rqd"], "mm"), _t("l0_min", lg["min"], "mm")],
               lg["l0"], "mm"),
            _f("l_couture", [_t("l0", lg["l0"], "mm"), _t("pas", lg["pas"], "cm")], lg["cote"], "cm", total=lg["total"])]


def _t(symbole: str, valeur, unite: str = "") -> dict:
    return {"symbole": symbole, "valeur": valeur, "unite": unite}


def _f(cle: str, termes: list, resultat, unite: str, **plus) -> dict:
    out = {"cle": cle, "termes": termes, "resultat": resultat, "unite": unite}
    out.update(plus)
    return out


# ============================================================ la vérification du joint
def joints(p: dict, figee: dict, r: dict, formule_axe) -> None:
    """Ajoute à ``r`` la vérification du JOINT de chaque section (en-tête du module) : la clé « joint », placée après la
    face inférieure secondaire ; la hauteur du joint quand elle gouverne ; ``section["joint"]`` (la couture, sa
    géométrie, sa longueur) ; les statuts, taux et gouvernants de chaque section et de la prédalle recalculés — ceux de
    la pleine prédalle gardés à part ; les avertissements du joint."""
    pr = p["dir_principale"].lower()
    sec = MD.direction_secondaire(pr)
    f = f"inf_{sec}"
    jd = copy.deepcopy(figee)
    coutures, termes = [], []
    for s_j, s_p, s_r in zip(jd["sections"], p["sections"], r["sections"]):
        c = couche_couture(s_p, sec)
        t = termes_pose(s_r["faces"][f"inf_{pr}"]["couches"], p["h_pre"], MD.couche_diametre(c), pr)
        c["dist"] = sum(v for _, v in t)                # sur la prédalle (h_p + Ø/2), ou sur le renfort principal qui y repose
        face = s_j["faces"][f]
        face["couches"] = [c] + face["couches"][1:]
        face["yG"] = None
        coutures.append(c)
        termes.append(t)
    r["pleine_predalle"] = {"statut": r["statut"], "taux_max": r["taux_max"], "gouvernant": dict(r["gouvernant"])}
    rj = MD.verifier(jd)
    for i, (s, sj, s_p, c, t) in enumerate(zip(r["sections"], rj["sections"], p["sections"], coutures, termes)):
        g, gj = s["grandeurs"], sj["grandeurs"]
        kh = next(n for n, x in enumerate(s["verifications"]) if x["cle"] == "hauteur")
        s["pleine_predalle"] = {"statut": s["statut"], "taux_max": s["taux_max"], "gouvernant": s["gouvernant"],
                                "hauteur": s["verifications"][kh]}
        geo = sj["faces"][f]
        c0 = geo["couches"][0]
        c0.update(auto=True, pose="sur_predalle", couture=True, reel_auto=c0["e_reel"], e_auto=c0["e"],
                  detail=P.detail(t, p["jeu_premier_lit"], p["arrondi"]), sur=len(t) > 2)
        # les renforts : les mêmes couches, aux mêmes positions qu'en pleine prédalle — leur lecture par le moteur prédalle
        for x, x_p in zip(geo["couches"][1:], s["faces"][f]["couches"][1:]):
            x.update({k: x_p[k] for k in ("auto", "reel_auto", "e_auto", "pose", "detail") if k in x_p})
        lg = longueur(c0["d"], p["beton"], p["fyk"], p.get("pas_couture", PAS_LONGUEUR))
        vf = next(v for v in sj["verifications"] if v["cle"] == f)
        axes = [dict(formule_axe(x, j), couture=j == 0) for j, x in enumerate(geo["couches"])]
        v = dict(vf, cle=CLE, titre=titre(sec), formules=axes + vf["formules"] + formules_longueur(c0["d"], lg), face="inf",
                 direction=sec, principale=False, joint=True)
        s["joint"] = {"face": f, "direction": sec, "auto": s_p.get("couture") is None, "couture": c0, "geometrie": geo,
                      "As": gj[f"As_{f}"], "As_pm": geo["As_pm"], "yG": gj[f"yG_{f}"], "d": gj[f"d_{f}"],
                      "As_req": gj[f"As_req_{f}"], "As_min": gj[f"As_min_{f}"], "besoin": gj[f"besoin_{f}"],
                      "etat": gj[f"etat_{f}"], "longueur": lg}
        k = next(n for n, x in enumerate(s["verifications"]) if x["cle"] == f)
        s["verifications"].insert(k + 1, v)
        # la HAUTEUR du joint : sa famille (le moment de la face, y_G du joint) — la ligne « Hauteur » la prend quand elle
        # excède celle de la pleine prédalle ; les grandeurs de la pleine prédalle ne changent pas
        M_j = gj[f"M_{f}"]
        hu_j = MD._hu_min(M_j, gj["alpha_b"], gj["b"], gj["mu"])
        h_j = hu_j + gj[f"yG_{f}"]
        g.update({"M_joint": M_j, "yG_joint": gj[f"yG_{f}"], "d_joint": gj[f"d_{f}"], "As_joint": gj[f"As_{f}"],
                  "As_req_joint": gj[f"As_req_{f}"], "As_min_joint": gj[f"As_min_{f}"], "besoin_joint": gj[f"besoin_{f}"],
                  "etat_joint": gj[f"etat_{f}"], "hu_min_joint": hu_j, "h_min_joint": h_j,
                  "etat_hauteur_joint": MD.OK if h_j <= g["h"] else MD.NOK})
        if M_j > 0 and h_j > s["verifications"][kh]["valeur"] + 1e-12:
            hv = s["verifications"][kh]
            s["verifications"][kh] = dict(hv, valeur=h_j, taux=h_j / hv["limite"], etat=g["etat_hauteur_joint"],
                                          motifs=[] if h_j <= g["h"] else ["hauteur_insuffisante"], face=CLE, joint=True,
                                          formules=[_f("hu_min", [_t("M", M_j, "kN·m"), _t("alpha_b", gj["alpha_b"], "N/mm²"),
                                                                 _t("b", gj["b"] * 10.0, "mm"), _t("mu", gj["mu"])], hu_j, "cm"),
                                                    _f("h_min", [_t("hu_min", hu_j, "cm"), _t("yG", gj[f"yG_{f}"], "cm")], h_j, "cm")])
        s["statut"] = MD.fusion_statuts(*(x["etat"] for x in s["verifications"]))
        gouv = max(s["verifications"], key=lambda x: x["taux"])
        s["taux_max"], s["gouvernant"] = gouv["taux"], gouv["cle"]
    i_g, v_g = max(((i, v) for i, sec_ in enumerate(r["sections"]) for v in sec_["verifications"]), key=lambda x: x[1]["taux"])
    r["taux_max"] = v_g["taux"]
    r["statut"] = MD.fusion_statuts(*(s["statut"] for s in r["sections"]))
    r["gouvernant"] = {"section": i_g, "nom": r["sections"][i_g]["nom"], "cle": v_g["cle"], "titre": v_g["titre"], "taux": v_g["taux"]}
    r["avertissements"] += _avertissements(p, r)


def termes_pose(principales: list, h_pre: float, d: float, pr: str) -> list:
    """La position réelle de la couture en termes : [(« h_p », h_p), (« Ø/2 », Ø/20)] sur la prédalle ; posée sur le
    renfort principal inférieur le plus épais de ceux qui reposent sur la prédalle (son dessous à h_p, ``TOL`` près), [(«
    h_p », h_p), (« Ø_Y », Ø_r/10), (« Ø/2 », Ø/20)]. ``principales`` : les couches de la face inférieure principale (le
    calcul de pleine prédalle : positions réelles, Ø) ; la base, dans la peau, n'en est pas."""
    sur = [x["d"] for k, x in enumerate(principales) if k and abs(x["e_reel"] - x["d"] / 20.0 - float(h_pre)) <= TOL]
    if not sur:
        return [("h_p", float(h_pre)), ("Ø/2", float(d) / 20.0)]
    return [("h_p", float(h_pre)), (f"Ø_{pr.upper()}", max(sur) / 10.0), ("Ø/2", float(d) / 20.0)]


def _fr(x: float, nd: int = 1) -> str:
    return f"{x:.{nd}f}".replace(".", ",")


def _avertissements(p: dict, r: dict) -> list:
    """Ce que le joint ajoute : la couture hors de l'épaisseur (d ≤ 0), une barre de gros diamètre (§8.8 : le
    recouvrement n'y est pas admis sans précaution). (Un renfort principal posé sur la prédalle ne la croise pas : elle
    est posée sur lui — ``termes_pose``.)"""
    out = []
    for i, s in enumerate(r["sections"]):
        j = s["joint"]
        c0 = j["couture"]
        v = next(x for x in s["verifications"] if x["cle"] == CLE)
        if "d_utile_negatif" in v["motifs"]:
            out.append({"code": "joint_d_utile_negatif", "section": i,
                        "message": f"{s['titre']} · joint des prédalles : les barres de couture, posées sur la prédalle à "
                                   f"{_fr(c0['e_reel'])} cm, sont hors de l'épaisseur (d utile ≤ 0)."})
        for a in j["longueur"]["avertissements"]:
            out.append({"code": f"joint_{a['code']}", "section": i, "message": f"{s['titre']} · barres de couture : {a['message']}"})
    return out

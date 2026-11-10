# -*- coding: utf-8 -*-
"""Poutre bois — le CALCUL selon l'Eurocode 5 : une pièce de bois massif ou lamellé-collé, de section rectangulaire,
sur deux appuis simples, sous charge uniformément répartie. Refait le 10/11/2026 sur la norme
(docs/site/LOT_Q_QUATRE_MODULES.md § 1 ; la méthode, les sources et les preuves : docs/bois/POUTRE_BOIS_MOTEUR.md).

Bibliothèque standard seule (math, re). Les valeurs de norme viennent de ``donnees`` (une source chacune), la saisie
de ``entree`` (relecture stricte). Nombres non arrondis ; sortie JSON stricte.

    from bois.poutre import moteur as M
    r = M.verifier(M.nouvelle_poutre())          # r["taux_max"], r["statut"], r["gouvernant"], r["verifications"]…

API
---
    nouvelle_poutre(nom="P1") -> dict           la poutre neuve (entree.nouvelle_poutre)
    normaliser(poutre) -> dict                  copie complète et contrôlée ; ValueError hors domaine (entree.normaliser)
    verifier(poutre) -> dict                    le calcul complet (ci-dessous)
    candidats_sections(poutre) -> list          les sections standard du TYPE du bois, chacune recalculée, par aire
    optimal(poutre) -> dict | None              la plus légère qui tient

UNITÉS — la saisie : mm (section, appui), m (portée, entraxes), kN/m ou kN/m² (charges). Le calcul : N et mm, avec
1 kN/m = 1 N/mm (l'ancien écran multipliait encore par 1 000 : ses flèches étaient 1 000 fois trop grandes). Les
résultats : N/mm² (contraintes), kN et kN·m (efforts), mm (flèches), kg (masses).

LA MÉTHODE (EN 1995-1-1:2004 + A1:2008 + A2:2014, annexe belge)
----------
Charges : g = G + g_pp (poids propre ρ_mean·b·h·9,81, compté par défaut), q = Q ; saisie surfacique : G = g_s·e,
Q = q_s·e (e : l'entraxe des pièces).

ELU (NBN EN 1990, 6.10) — DEUX combinaisons, chacune avec SON k_mod (§3.1.3(2) : celui de l'action de plus courte
durée) : « G seule », p_d = γ_G·g avec k_mod permanent ; « G + Q », p_d = γ_G·g + γ_Q·q avec le k_mod de la charge
variable (absente quand q = 0). M_d = p_d·L²/8, V_d = p_d·L/2 (à l'axe de l'appui, sans la réduction du §6.1.7(3) :
côté sûr). Chaque vérification retient la combinaison au taux le plus fort et la NOMME.

  flexion (§6.1.6, §6.3.3)        σ_m,d = M_d/W ≤ k_crit·f_m,d ; f_m,d = k_mod·k_h·k_sys·f_m,k/γ_M
      k_h (§3.2(3), §3.3(3))      massif : min((150/h)^0,2 ; 1,3) si h < 150 mm ; lamellé : min((600/h)^0,1 ; 1,1) si h < 600 mm ; 1 sinon
      déversement                 continu : k_crit = 1 ; aux appuis : l_ef = 0,9·L + 2·h ; entretoises : l_ef = min(e_m ; 0,9·L) + 2·h
                                  σ_m,crit = 0,78·b²·E_0,05/(h·l_ef) ; λ_rel,m = √(f_m,k/σ_m,crit)
                                  k_crit = 1 (λ ≤ 0,75) ; 1,56 − 0,75·λ (≤ 1,4) ; 1/λ² au-delà
  cisaillement (§6.1.7)           τ_d = 1,5·V_d/(k_cr·b·h) ≤ f_v,d = k_mod·k_sys·f_v,k/γ_M ; k_cr = 0,67
  compression à l'appui (§6.1.5)  σ_c,90,d = V_d/(b·l_ef,a) ≤ k_c,90·f_c,90,d ; f_c,90,d = k_mod·f_c,90,k/γ_M
                                  l_ef,a = a + min(30 ; a ; l_1/2) + min(30 ; a ; débord) ; l_1 = L − a
                                  k_c,90 = 1,5 (massif), 1,75 (lamellé, a ≤ 400 mm) si l_1 ≥ 2·h ; 1,0 sinon

ELS (§2.2.3, §7.2), combinaison caractéristique, E_0,mean et G_mean :

  w(p) = 5·p·L⁴/(384·E·I) + 1,2·p·L²/(8·G·A)            flexion + effort tranchant, écrits séparément
  w_inst = w(g) + w(q) ≤ L/n_inst
  w_fin = w(g)·(1 + k_def) + w(q)·(1 + ψ₂·k_def) ; w_net,fin = w_fin − w_c ≤ L/n_fin

SORTIE de ``verifier`` ::

    {"reference": …, "poutre": {…} (normalisée),
     "materiau": {"classe", "type": "massif"|"lamelle", "libelle_type", "norme", "fm_k", "ft0_k", "fc0_k", "fc90_k",
                  "fv_k", "E0_mean", "E0_05", "G_mean", "rho_k", "rho_mean"},
     "coefficients": {"service", "libelle_service", "gamma_M", "gamma_G", "gamma_Q", "kdef", "kcr", "ksys" (1 ou 1,1),
                      "systeme" (bool), "kh", "kmod_permanent", "kmod_variable", "nature", "libelle_nature",
                      "duree", "libelle_duree", "psi2", "nature_libre" (bool)},
     "geometrie": {"b", "h" (mm), "standard", "section" (« 75x225 » ou None), "designation" (« 75 × 225 », à la virgule),
                   "A" (mm²), "W" (mm³), "I" (mm⁴), "L" (m), "L_mm", "elancement" (h/b), "longueur" (mm : L + a + 2·débord)},
     "charges": {"mode", "G", "Q" (kN/m, hors poids propre), "g_surf", "q_surf" (kN/m²), "entraxe" (m) — None en
                 saisie linéaire —, "poids_propre" (kN/m, toujours calculé), "poids_propre_compte" (bool),
                 "g", "q" (kN/m : ce que le calcul applique)},
     "combinaisons": [<combinaison>, …]  — « G » toujours, « GQ » quand q > 0,
     "combinaison_elu": "G"|"GQ"  — celle du taux ELU le plus fort (les diagrammes V et M),
     "efforts": {"combinaison", "nom", "p_d" (kN/m), "M_d" (kN·m), "V_d" (kN), "R" (kN, la réaction d'appui = V_d)},
     "deversement": {"type", "libelle", "l_ef" (mm, None en continu), "entraxe" (mm, entretoises), "sigma_crit",
                     "lambda_rel" (None en continu), "kcrit", "branche": "continu"|"palier"|"lineaire"|"elance"},
     "appui": {"verifie" (bool), "a", "debord", "l1", "cote_travee", "cote_about", "l_ef" (mm), "kc90",
               "kc90_plein" (la valeur de la norme pour ce bois), "raison": None|"distance"|"longueur"},
     "fleches": {"w_g_m", "w_g_v", "w_g", "w_q_m", "w_q_v", "w_q", "w_inst", "n_inst", "lim_inst", "w_fin", "w_c",
                 "w_net_fin", "n_fin", "lim_fin" (mm)},
     "verifications": [<verification>, …]  — flexion, cisaillement, appui (si vérifié), fleche_inst, fleche_fin,
     "taux_max", "statut": "ok"|"nok", "gouvernant": {"cle", "nom", "groupe", "taux", "combinaison"},
     "taux_elu", "taux_els",
     "calcul": [<bloc>, …], "hypotheses": [str, …], "avertissements": [{"code", "niveau", "message"}, …],
     "non_couvert": [str, …], "masse_lineique" (kg/m), "masse" (kg, sur la longueur de la pièce)}

  <combinaison> = {"cle": "G"|"GQ", "nom": "G seule"|"G + Q", "expression": "γ_G·g"|"γ_G·g + γ_Q·q", "duree",
                   "libelle_duree", "kmod", "p_d" (kN/m), "M_d" (kN·m), "V_d" (kN)}
  <verification> = {"cle", "nom", "groupe": "ELU"|"ELS", "article", "enonce" (« σ_m,d = M_d / W ≤ k_crit·f_m,d »),
                    "symbole", "symbole_limite", "valeur", "limite", "unite", "taux", "etat": "ok"|"nok",
                    "combinaison" (la clé de celle qui gouverne ; None à l'ELS), "detail": [{"combinaison", "nom",
                    "duree", "kmod", "resistance" (f_d), "valeur", "limite", "taux", "etat"}, …] (une ligne par
                    combinaison ; vide à l'ELS), "blocs": [clés des blocs de ``calcul`` qui y mènent], "formules":
                    [<formule>, …] (les mêmes, à la suite)}
  <bloc> = {"cle", "titre", "combinaison" (ou None), "formules": [<formule>, …]}
  <formule> = {"cle", "symbole", "expression" (« γ_G·g + γ_Q·q »), "gabarit" (« {γ_G} × {g} + {γ_Q} × {q} », ou None
               pour une valeur LUE dans un tableau ou fixée par une condition : ``expression`` dit alors d'où elle
               vient), "termes": [{"symbole", "valeur", "unite"}], "resultat", "unite"}
     Le RÉSULTAT d'une formule se recalcule à partir de son gabarit et de ses seuls termes (le test le fait, par un
     interpréteur indépendant) : c'est « symbole = expression = valeurs = résultat », comme une note faite à la main.
     L'écran remplace chaque {terme} par sa valeur mise en forme ; il n'écrit aucune formule.
"""
import math
import re
from typing import Dict, List, Optional

from . import donnees as D
from . import entree as E
from .entree import normaliser, nouvelle_poutre  # noqa: F401 — l'API

OK, NOK = "ok", "nok"
ELU, ELS = "ELU", "ELS"
INFORMATION, ATTENTION = "information", "attention"
REFERENCE = D.REFERENCE

_ACCOLADES = re.compile(r"\{([^{}]+)\}")


# ------------------------------------------------------------------ les formules : « symbole = expression = valeurs = résultat »
def expression(gabarit: str) -> str:
    """L'expression symbolique d'un gabarit : « {γ_G} × {g} + {γ_Q} × {q} » → « γ_G·g + γ_Q·q »."""
    return _ACCOLADES.sub(r"\1", gabarit).replace(" × ", "·")


def _c(x: float) -> str:
    """Une CONSTANTE de la norme écrite dans un gabarit (0,78 ; 1,56 ; 384) : virgule décimale, sans zéro inutile.
    Elle vient de ``donnees`` : le texte de la formule et le calcul ont la même source."""
    return f"{x:.6g}".replace(".", ",")


def _n(x: float, nd: int = 2) -> str:
    """Un nombre dans un message ou une hypothèse : ``nd`` décimales, virgule décimale."""
    return f"{x:.{nd}f}".replace(".", ",")


def _t(symbole: str, valeur, unite: str = "") -> Dict:
    return {"symbole": symbole, "valeur": valeur, "unite": unite}


def _fx(cle: str, symbole: str, gabarit: str, termes: List[Dict], resultat, unite: str = "") -> Dict:
    """Une formule CALCULÉE : son gabarit (chaque terme entre accolades), ses termes, son résultat."""
    return {"cle": cle, "symbole": symbole, "expression": expression(gabarit), "gabarit": gabarit, "termes": termes,
            "resultat": resultat, "unite": unite}


def _lu(cle: str, symbole: str, origine: str, resultat, unite: str = "", termes: Optional[List[Dict]] = None) -> Dict:
    """Une valeur LUE (un tableau de la norme) ou FIXÉE par une condition : pas de gabarit, ``origine`` dit d'où
    elle vient."""
    return {"cle": cle, "symbole": symbole, "expression": origine, "gabarit": None, "termes": termes or [],
            "resultat": resultat, "unite": unite}


def _taux(valeur: float, limite: float) -> float:
    return valeur / limite


def _etat(taux: float) -> str:
    """Conforme jusqu'à 100 % compris ; au-delà, non conforme (le seuil du code couleur du bureau)."""
    return NOK if taux > 1.0 else OK


# ------------------------------------------------------------------ les coefficients
def k_h(typ: str, h: float) -> float:
    """k_h (§3.2(3) massif, §3.3(3) lamellé-collé) : la majoration de f_m,k sous la hauteur de référence."""
    r = D.K_H[typ]
    if h >= r["h_ref"]:
        return 1.0
    return min((r["h_ref"] / h) ** r["exposant"], r["plafond"])


def longueur_efficace(maintien: Dict, L_mm: float, h: float) -> Optional[float]:
    """l_ef du déversement (tableau 6.1), en mm ; None quand la fibre comprimée est maintenue sur toute la longueur."""
    if maintien["type"] == "continu":
        return None
    if maintien["type"] == "appuis":
        return D.LEF_CHARGE_REPARTIE * L_mm + D.LEF_FIBRE_COMPRIMEE * h
    return min(maintien["entraxe"] * 1000.0, D.LEF_CHARGE_REPARTIE * L_mm) + D.LEF_FIBRE_COMPRIMEE * h


def k_crit(lambda_rel: float) -> float:
    """k_crit (6.34) en fonction de l'élancement relatif de flexion."""
    if lambda_rel <= D.LAMBDA_PALIER:
        return 1.0
    if lambda_rel <= D.LAMBDA_ELANCE:
        return D.K_CRIT_ORIGINE - D.K_CRIT_PENTE * lambda_rel
    return 1.0 / lambda_rel ** 2


def _branche(lambda_rel: float) -> str:
    return "palier" if lambda_rel <= D.LAMBDA_PALIER else ("lineaire" if lambda_rel <= D.LAMBDA_ELANCE else "elance")


def fleche_flexion(p: float, L_mm: float, E_mod: float, inertie: float) -> float:
    """La flèche de FLEXION à mi-portée sous la charge répartie p (N/mm = kN/m), en mm : 5·p·L⁴/(384·E·I)."""
    return D.FLECHE_NUM * p * L_mm ** 4 / (D.FLECHE_DEN * E_mod * inertie)


def fleche_tranchant(p: float, L_mm: float, G_mod: float, aire: float) -> float:
    """La flèche d'EFFORT TRANCHANT à mi-portée sous p (N/mm), en mm : 1,2·p·L²/(8·G·A)."""
    return D.FORME_TRANCHANT * p * L_mm ** 2 / (8.0 * G_mod * aire)


# ------------------------------------------------------------------ le calcul (les nombres)
def _calculer(p: Dict) -> Dict:
    """Toutes les grandeurs d'une poutre NORMALISÉE, à plat (le contexte que ``verifier`` met en forme)."""
    mat = D.CLASSES[p["classe"]]
    typ = mat["type"]
    b, h, L = p["b"], p["h"], p["L"]
    L_mm = L * 1000.0
    A = b * h
    W = b * h ** 2 / 6.0
    inertie = b * h ** 3 / 12.0

    # les charges (kN/m = N/mm)
    ch = p["charges"]
    surfacique = ch["mode"] == "surfacique"
    G = ch["g"] * ch["entraxe"] if surfacique else ch["G"]
    Q = ch["q"] * ch["entraxe"] if surfacique else ch["Q"]
    g_pp = mat["rho_mean"] * b * h * D.G_PESANTEUR / 1e9
    g = G + g_pp if ch["poids_propre"] else G
    q = Q

    # les coefficients
    service = p["service"]
    gamma_M = D.GAMMA_M[typ]
    kdef = D.KDEF[service]
    ksys = D.K_SYS if p["systeme"] else 1.0
    kh = k_h(typ, h)
    duree, psi2 = ch["duree"], ch["psi2"]

    # le déversement (indépendant de la combinaison)
    l_ef = longueur_efficace(p["maintien"], L_mm, h)
    if l_ef is None:
        sigma_crit = lambda_rel = None
        kcrit, branche = 1.0, "continu"
    else:
        sigma_crit = D.COEF_SIGMA_CRIT * b ** 2 * mat["E0_05"] / (h * l_ef)
        lambda_rel = math.sqrt(mat["fm_k"] / sigma_crit)
        kcrit, branche = k_crit(lambda_rel), _branche(lambda_rel)

    # l'appui (géométrie ; indépendant de la combinaison)
    ap = p["appui"]
    a, debord = ap["a"], ap["debord"]
    l1 = L_mm - a
    cote_travee = min(D.MAJORATION_APPUI, a, l1 / 2.0)
    cote_about = min(D.MAJORATION_APPUI, a, debord)
    l_ef_a = a + cote_travee + cote_about
    kc90_plein = D.K_C90[typ]
    raison = None
    if l1 < D.L1_SUR_H_MIN * h:
        raison = "distance"
    elif typ in D.A_MAX_K_C90 and a > D.A_MAX_K_C90[typ]:
        raison = "longueur"
    kc90 = D.K_C90_DEFAUT if raison else kc90_plein

    # les deux combinaisons ELU, chacune avec son k_mod
    combs = [{"cle": "G", "nom": "G seule", "duree": "permanent", "p_d": p["gamma_G"] * g}]
    if q > 0:
        combs.append({"cle": "GQ", "nom": "G + Q", "duree": duree, "p_d": p["gamma_G"] * g + p["gamma_Q"] * q})
    for c in combs:
        c["kmod"] = D.KMOD[service][c["duree"]]
        c["M_d"] = c["p_d"] * L ** 2 / 8.0                       # kN·m
        c["V_d"] = c["p_d"] * L / 2.0                            # kN
        # flexion
        c["fm_d"] = c["kmod"] * kh * ksys * mat["fm_k"] / gamma_M
        c["sigma_m"] = c["M_d"] * 1e6 / W
        c["lim_m"] = kcrit * c["fm_d"]
        c["taux_m"] = _taux(c["sigma_m"], c["lim_m"])
        # cisaillement
        c["fv_d"] = c["kmod"] * ksys * mat["fv_k"] / gamma_M
        c["tau"] = D.COEF_TAU_RECTANGLE * c["V_d"] * 1e3 / (D.K_CR * b * h)
        c["taux_v"] = _taux(c["tau"], c["fv_d"])
        # compression à l'appui
        c["fc90_d"] = c["kmod"] * mat["fc90_k"] / gamma_M
        c["sigma_c90"] = c["V_d"] * 1e3 / (b * l_ef_a)
        c["lim_c90"] = kc90 * c["fc90_d"]
        c["taux_c90"] = _taux(c["sigma_c90"], c["lim_c90"])

    # les flèches (ELS caractéristique)
    w_g_m = fleche_flexion(g, L_mm, mat["E0_mean"], inertie)
    w_g_v = fleche_tranchant(g, L_mm, mat["G_mean"], A)
    w_q_m = fleche_flexion(q, L_mm, mat["E0_mean"], inertie)
    w_q_v = fleche_tranchant(q, L_mm, mat["G_mean"], A)
    w_g, w_q = w_g_m + w_g_v, w_q_m + w_q_v
    w_inst = w_g + w_q
    lim_inst = L_mm / p["lim_inst"]
    w_fin = w_g * (1.0 + kdef) + w_q * (1.0 + psi2 * kdef)
    w_c = p["contre_fleche"]
    w_net_fin = w_fin - w_c
    lim_fin = L_mm / p["lim_fin"]
    taux_inst = _taux(w_inst, lim_inst)
    taux_fin = _taux(max(w_net_fin, 0.0), lim_fin)

    masse_lineique = mat["rho_mean"] * A / 1e6                   # kg/m
    longueur = L_mm + a + 2.0 * debord                           # mm : la pièce, d'about à about
    return dict(mat=mat, typ=typ, b=b, h=h, L=L, L_mm=L_mm, A=A, W=W, I=inertie, surfacique=surfacique, G=G, Q=Q, g_pp=g_pp,
                g=g, q=q, service=service, gamma_M=gamma_M, kdef=kdef, ksys=ksys, kh=kh, duree=duree, psi2=psi2, l_ef=l_ef,
                sigma_crit=sigma_crit, lambda_rel=lambda_rel, kcrit=kcrit, branche=branche, a=a, debord=debord, l1=l1,
                cote_travee=cote_travee, cote_about=cote_about, l_ef_a=l_ef_a, kc90=kc90, kc90_plein=kc90_plein, raison=raison,
                combs=combs, w_g_m=w_g_m, w_g_v=w_g_v, w_q_m=w_q_m, w_q_v=w_q_v, w_g=w_g, w_q=w_q, w_inst=w_inst,
                lim_inst=lim_inst, w_fin=w_fin, w_c=w_c, w_net_fin=w_net_fin, lim_fin=lim_fin, taux_inst=taux_inst,
                taux_fin=taux_fin, masse_lineique=masse_lineique, longueur=longueur, masse=masse_lineique * longueur / 1000.0)


def _lignes_verifiees(p: Dict, x: Dict) -> List[Dict]:
    """Les vérifications d'un contexte : (clé, groupe, taux, combinaison gouvernante), dans l'ordre de la note."""
    out = []
    cles = [("flexion", "taux_m"), ("cisaillement", "taux_v")] + ([("appui", "taux_c90")] if p["appui"]["verifier"] else [])
    for cle, k in cles:
        c = max(x["combs"], key=lambda c: c[k])
        out.append({"cle": cle, "groupe": ELU, "taux": c[k], "combinaison": c["cle"]})
    out.append({"cle": "fleche_inst", "groupe": ELS, "taux": x["taux_inst"], "combinaison": None})
    out.append({"cle": "fleche_fin", "groupe": ELS, "taux": x["taux_fin"], "combinaison": None})
    return out


NOMS = {"flexion": "Flexion", "cisaillement": "Cisaillement", "appui": "Compression à l'appui",
        "fleche_inst": "Flèche instantanée", "fleche_fin": "Flèche finale nette"}


# ------------------------------------------------------------------ le détail : les blocs de formules
def _blocs(p: Dict, x: Dict) -> List[Dict]:
    """Le détail du calcul, bloc par bloc : chaque formule avec son gabarit, ses termes et son résultat."""
    mat, typ = x["mat"], x["typ"]
    b, h, L, L_mm = x["b"], x["h"], x["L"], x["L_mm"]
    ch = p["charges"]
    blocs = []

    def bloc(cle, titre, formules, combinaison=None):
        blocs.append({"cle": cle, "titre": titre, "combinaison": combinaison, "formules": formules})

    # ---- les charges
    f = []
    if x["surfacique"]:
        f.append(_fx("G", "G", "{g_s} × {e}", [_t("g_s", ch["g"], "kN/m²"), _t("e", ch["entraxe"], "m")], x["G"], "kN/m"))
        f.append(_fx("Q", "Q", "{q_s} × {e}", [_t("q_s", ch["q"], "kN/m²"), _t("e", ch["entraxe"], "m")], x["Q"], "kN/m"))
    if ch["poids_propre"]:
        f.append(_fx("g_pp", "g_pp", f"{{ρ_mean}} × {{b}} × {{h}} × {_c(D.G_PESANTEUR)} / 10⁹",
                     [_t("ρ_mean", mat["rho_mean"], "kg/m³"), _t("b", b, "mm"), _t("h", h, "mm")], x["g_pp"], "kN/m"))
        f.append(_fx("g", "g", "{G} + {g_pp}", [_t("G", x["G"], "kN/m"), _t("g_pp", x["g_pp"], "kN/m")], x["g"], "kN/m"))
    else:
        f.append(_fx("g", "g", "{G}", [_t("G", x["G"], "kN/m")], x["g"], "kN/m"))
    f.append(_fx("q", "q", "{Q}", [_t("Q", x["Q"], "kN/m")], x["q"], "kN/m"))
    bloc("charges", "Charges", f)

    # ---- les combinaisons ELU
    for c in x["combs"]:
        gab = "{γ_G} × {g}" if c["cle"] == "G" else "{γ_G} × {g} + {γ_Q} × {q}"
        termes = [_t("γ_G", p["gamma_G"]), _t("g", x["g"], "kN/m")]
        if c["cle"] == "GQ":
            termes += [_t("γ_Q", p["gamma_Q"]), _t("q", x["q"], "kN/m")]
        bloc(f"combinaison_{c['cle']}", f"Combinaison {c['nom']}", [
            _lu("kmod", "k_mod", f"tableau 3.1 · classe de service {x['service']} · {D.LIBELLES_DUREE[c['duree']]}", c["kmod"]),
            _fx("p_d", "p_d", gab, termes, c["p_d"], "kN/m"),
            _fx("M_d", "M_d", "{p_d} × {L}² / 8", [_t("p_d", c["p_d"], "kN/m"), _t("L", L, "m")], c["M_d"], "kN·m"),
            _fx("V_d", "V_d", "{p_d} × {L} / 2", [_t("p_d", c["p_d"], "kN/m"), _t("L", L, "m")], c["V_d"], "kN"),
        ], c["cle"])

    # ---- la section
    r = D.K_H[typ]
    if h < r["h_ref"]:
        f_kh = _fx("kh", "k_h", f"min(({_c(r['h_ref'])} / {{h}})^{_c(r['exposant'])} ; {_c(r['plafond'])})", [_t("h", h, "mm")], x["kh"])
    else:
        f_kh = _lu("kh", "k_h", f"h ≥ {_c(r['h_ref'])} mm : pas de majoration ({r['article']})", x["kh"], termes=[_t("h", h, "mm")])
    bloc("section", "Section", [
        _fx("A", "A", "{b} × {h}", [_t("b", b, "mm"), _t("h", h, "mm")], x["A"], "mm²"),
        _fx("W", "W", "{b} × {h}² / 6", [_t("b", b, "mm"), _t("h", h, "mm")], x["W"], "mm³"),
        _fx("I", "I", "{b} × {h}³ / 12", [_t("b", b, "mm"), _t("h", h, "mm")], x["I"], "mm⁴"),
        f_kh,
    ])

    # ---- le déversement
    if x["l_ef"] is None:
        f = [_lu("kcrit", "k_crit", "fibre comprimée maintenue sur toute la longueur, appuis maintenus en torsion (§6.3.3(5))", x["kcrit"])]
    else:
        if p["maintien"]["type"] == "appuis":
            f_lef = _fx("l_ef", "l_ef", f"{_c(D.LEF_CHARGE_REPARTIE)} × {{L}} + {_c(D.LEF_FIBRE_COMPRIMEE)} × {{h}}",
                        [_t("L", L_mm, "mm"), _t("h", h, "mm")], x["l_ef"], "mm")
        else:
            f_lef = _fx("l_ef", "l_ef", f"min({{e_m}} ; {_c(D.LEF_CHARGE_REPARTIE)} × {{L}}) + {_c(D.LEF_FIBRE_COMPRIMEE)} × {{h}}",
                        [_t("e_m", p["maintien"]["entraxe"] * 1000.0, "mm"), _t("L", L_mm, "mm"), _t("h", h, "mm")], x["l_ef"], "mm")
        f = [f_lef,
             _fx("sigma_crit", "σ_m,crit", f"{_c(D.COEF_SIGMA_CRIT)} × {{b}}² × {{E_0,05}} / ({{h}} × {{l_ef}})",
                 [_t("b", b, "mm"), _t("E_0,05", mat["E0_05"], "N/mm²"), _t("h", h, "mm"), _t("l_ef", x["l_ef"], "mm")], x["sigma_crit"], "N/mm²"),
             _fx("lambda_rel", "λ_rel,m", "√({f_m,k} / {σ_m,crit})",
                 [_t("f_m,k", mat["fm_k"], "N/mm²"), _t("σ_m,crit", x["sigma_crit"], "N/mm²")], x["lambda_rel"])]
        lam = [_t("λ_rel,m", x["lambda_rel"])]
        if x["branche"] == "palier":
            f.append(_lu("kcrit", "k_crit", f"λ_rel,m ≤ {_c(D.LAMBDA_PALIER)} : pas de réduction (6.34)", x["kcrit"], termes=lam))
        elif x["branche"] == "lineaire":
            f.append(_fx("kcrit", "k_crit", f"{_c(D.K_CRIT_ORIGINE)} − {_c(D.K_CRIT_PENTE)} × {{λ_rel,m}}", lam, x["kcrit"]))
        else:
            f.append(_fx("kcrit", "k_crit", "1 / {λ_rel,m}²", lam, x["kcrit"]))
    bloc("deversement", "Déversement", f)

    # ---- la flexion, le cisaillement, par combinaison
    for c in x["combs"]:
        gab = "{k_mod} × {k_h} × {k_sys} × {f_m,k} / {γ_M}" if p["systeme"] else "{k_mod} × {k_h} × {f_m,k} / {γ_M}"
        termes = [_t("k_mod", c["kmod"]), _t("k_h", x["kh"])] + ([_t("k_sys", x["ksys"])] if p["systeme"] else []) \
            + [_t("f_m,k", mat["fm_k"], "N/mm²"), _t("γ_M", x["gamma_M"])]
        bloc(f"flexion_{c['cle']}", "Flexion", [
            _fx("fm_d", "f_m,d", gab, termes, c["fm_d"], "N/mm²"),
            _fx("sigma_m", "σ_m,d", "{M_d} × 10⁶ / {W}", [_t("M_d", c["M_d"], "kN·m"), _t("W", x["W"], "mm³")], c["sigma_m"], "N/mm²"),
            _fx("taux", "η", "{σ_m,d} / ({k_crit} × {f_m,d})",
                [_t("σ_m,d", c["sigma_m"], "N/mm²"), _t("k_crit", x["kcrit"]), _t("f_m,d", c["fm_d"], "N/mm²")], c["taux_m"]),
        ], c["cle"])
    for c in x["combs"]:
        gab = "{k_mod} × {k_sys} × {f_v,k} / {γ_M}" if p["systeme"] else "{k_mod} × {f_v,k} / {γ_M}"
        termes = [_t("k_mod", c["kmod"])] + ([_t("k_sys", x["ksys"])] if p["systeme"] else []) \
            + [_t("f_v,k", mat["fv_k"], "N/mm²"), _t("γ_M", x["gamma_M"])]
        bloc(f"cisaillement_{c['cle']}", "Cisaillement", [
            _fx("fv_d", "f_v,d", gab, termes, c["fv_d"], "N/mm²"),
            _fx("tau", "τ_d", f"{_c(D.COEF_TAU_RECTANGLE)} × {{V_d}} × 10³ / ({{k_cr}} × {{b}} × {{h}})",
                [_t("V_d", c["V_d"], "kN"), _t("k_cr", D.K_CR), _t("b", b, "mm"), _t("h", h, "mm")], c["tau"], "N/mm²"),
            _fx("taux", "η", "{τ_d} / {f_v,d}", [_t("τ_d", c["tau"], "N/mm²"), _t("f_v,d", c["fv_d"], "N/mm²")], c["taux_v"]),
        ], c["cle"])

    # ---- la compression à l'appui
    if p["appui"]["verifier"]:
        deux_h = D.L1_SUR_H_MIN * h
        if x["raison"] == "distance":
            origine = f"appuis rapprochés, l_1 < {_c(D.L1_SUR_H_MIN)}·h = {_n(deux_h, 0)} mm : pas de majoration (§6.1.5(2))"
        elif x["raison"] == "longueur":
            origine = f"lamellé-collé, a > {_c(D.A_MAX_K_C90[typ])} mm : pas de majoration (§6.1.5(4))"
        else:
            origine = (f"{D.COURTS_TYPE[typ]} résineux sur appuis discrets, l_1 ≥ {_c(D.L1_SUR_H_MIN)}·h = {_n(deux_h, 0)} mm"
                       + (f", a ≤ {_c(D.A_MAX_K_C90[typ])} mm" if typ in D.A_MAX_K_C90 else "") + " (§6.1.5(4))")
        m = _c(D.MAJORATION_APPUI)
        bloc("appui", "Appui", [
            _fx("l1", "l_1", "{L} − {a}", [_t("L", L_mm, "mm"), _t("a", x["a"], "mm")], x["l1"], "mm"),
            _fx("l_ef_a", "l_ef,a", f"{{a}} + min({m} ; {{a}} ; {{l_1}} / 2) + min({m} ; {{a}} ; {{débord}})",
                [_t("a", x["a"], "mm"), _t("l_1", x["l1"], "mm"), _t("débord", x["debord"], "mm")], x["l_ef_a"], "mm"),
            _lu("kc90", "k_c,90", origine, x["kc90"], termes=[_t("l_1", x["l1"], "mm"), _t("h", h, "mm")]),
        ])
        for c in x["combs"]:
            bloc(f"appui_{c['cle']}", "Compression à l'appui", [
                _fx("fc90_d", "f_c,90,d", "{k_mod} × {f_c,90,k} / {γ_M}",
                    [_t("k_mod", c["kmod"]), _t("f_c,90,k", mat["fc90_k"], "N/mm²"), _t("γ_M", x["gamma_M"])], c["fc90_d"], "N/mm²"),
                _fx("sigma_c90", "σ_c,90,d", "{V_d} × 10³ / ({b} × {l_ef,a})",
                    [_t("V_d", c["V_d"], "kN"), _t("b", b, "mm"), _t("l_ef,a", x["l_ef_a"], "mm")], c["sigma_c90"], "N/mm²"),
                _fx("taux", "η", "{σ_c,90,d} / ({k_c,90} × {f_c,90,d})",
                    [_t("σ_c,90,d", c["sigma_c90"], "N/mm²"), _t("k_c,90", x["kc90"]), _t("f_c,90,d", c["fc90_d"], "N/mm²")], c["taux_c90"]),
            ], c["cle"])

    # ---- les flèches
    gab_m = f"{_c(D.FLECHE_NUM)} × {{p}} × {{L}}⁴ / ({_c(D.FLECHE_DEN)} × {{E_0,mean}} × {{I}})"
    gab_v = f"{_c(D.FORME_TRANCHANT)} × {{p}} × {{L}}² / (8 × {{G_mean}} × {{A}})"

    def composantes(nom, charge, w_m, w_v, w):
        gm, gv = gab_m.replace("{p}", "{" + nom + "}"), gab_v.replace("{p}", "{" + nom + "}")
        return [_fx(f"w_{nom}_m", f"w_{nom},m", gm, [_t(nom, charge, "N/mm"), _t("L", L_mm, "mm"), _t("E_0,mean", mat["E0_mean"], "N/mm²"),
                                                    _t("I", x["I"], "mm⁴")], w_m, "mm"),
                _fx(f"w_{nom}_v", f"w_{nom},v", gv, [_t(nom, charge, "N/mm"), _t("L", L_mm, "mm"), _t("G_mean", mat["G_mean"], "N/mm²"),
                                                    _t("A", x["A"], "mm²")], w_v, "mm"),
                _fx(f"w_{nom}", f"w_{nom}", "{w_" + nom + ",m} + {w_" + nom + ",v}",
                    [_t(f"w_{nom},m", w_m, "mm"), _t(f"w_{nom},v", w_v, "mm")], w, "mm")]

    bloc("fleche_inst", "Flèche instantanée",
         composantes("g", x["g"], x["w_g_m"], x["w_g_v"], x["w_g"]) + composantes("q", x["q"], x["w_q_m"], x["w_q_v"], x["w_q"]) + [
             _fx("w_inst", "w_inst", "{w_g} + {w_q}", [_t("w_g", x["w_g"], "mm"), _t("w_q", x["w_q"], "mm")], x["w_inst"], "mm"),
             _fx("lim_inst", "w_lim", "{L} / {n}", [_t("L", L_mm, "mm"), _t("n", p["lim_inst"])], x["lim_inst"], "mm"),
             _fx("taux", "η", "{w_inst} / {w_lim}", [_t("w_inst", x["w_inst"], "mm"), _t("w_lim", x["lim_inst"], "mm")], x["taux_inst"]),
         ])
    if ch["nature"] == D.NATURE_LIBRE:
        origine_psi = "saisi (nature « autre »)"
    else:
        origine_psi = f"NBN EN 1990 ANB · {D.LIBELLES_NATURE[ch['nature']]}"
    f = [_lu("kdef", "k_def", f"tableau 3.2 · classe de service {x['service']}", x["kdef"]),
         _lu("psi2", "ψ_2", origine_psi, x["psi2"]),
         _fx("w_fin", "w_fin", "{w_g} × (1 + {k_def}) + {w_q} × (1 + {ψ_2} × {k_def})",
             [_t("w_g", x["w_g"], "mm"), _t("k_def", x["kdef"]), _t("w_q", x["w_q"], "mm"), _t("ψ_2", x["psi2"])], x["w_fin"], "mm"),
         _fx("w_net_fin", "w_net,fin", "{w_fin} − {w_c}", [_t("w_fin", x["w_fin"], "mm"), _t("w_c", x["w_c"], "mm")], x["w_net_fin"], "mm"),
         _fx("lim_fin", "w_lim", "{L} / {n}", [_t("L", L_mm, "mm"), _t("n", p["lim_fin"])], x["lim_fin"], "mm")]
    if x["w_net_fin"] >= 0:
        f.append(_fx("taux", "η", "{w_net,fin} / {w_lim}", [_t("w_net,fin", x["w_net_fin"], "mm"), _t("w_lim", x["lim_fin"], "mm")], x["taux_fin"]))
    else:
        f.append(_lu("taux", "η", "w_net,fin < 0 : la poutre reste cambrée, le taux est compté nul", x["taux_fin"],
                     termes=[_t("w_net,fin", x["w_net_fin"], "mm")]))
    bloc("fleche_fin", "Flèche finale nette", f)
    return blocs


# ------------------------------------------------------------------ les avertissements, les hypothèses
def _avertissements(p: Dict, x: Dict) -> List[Dict]:
    out = []

    def dire(code, niveau, message):
        out.append({"code": code, "niveau": niveau, "message": message})

    ch, ap = p["charges"], p["appui"]
    if x["g"] == 0 and x["q"] == 0:
        dire("sans_charge", ATTENTION, "Aucune charge : G = Q = 0 et le poids propre n'est pas compté ; tous les taux sont nuls.")
    if not ch["poids_propre"]:
        dire("poids_propre", ATTENTION, f"Poids propre non compté ({_n(x['g_pp'], 3)} kN/m) : il doit être compris dans G.")
    if ap["verifier"] and x["raison"] == "distance":
        dire("kc90_distance", ATTENTION,
             f"Appuis rapprochés : la distance libre l_1 = L − a = {_n(x['l1'], 0)} mm est inférieure à 2·h = {_n(D.L1_SUR_H_MIN * x['h'], 0)} mm ; "
             f"k_c,90 est ramené de {_n(x['kc90_plein'], 2)} à {_n(x['kc90'], 1)} (EN 1995-1-1 §6.1.5).")
    if ap["verifier"] and x["raison"] == "longueur":
        dire("kc90_longueur", ATTENTION,
             f"Lamellé-collé : la longueur d'appui a = {_n(x['a'], 0)} mm dépasse {_n(D.A_MAX_K_C90[x['typ']], 0)} mm ; "
             f"k_c,90 est ramené de {_n(x['kc90_plein'], 2)} à {_n(x['kc90'], 1)} (EN 1995-1-1 §6.1.5(4)).")
    if not ap["verifier"]:
        dire("appui_non_verifie", ATTENTION, "La compression à l'appui n'est pas vérifiée : elle a été retirée de la note.")
    if x["w_net_fin"] < 0:
        dire("contre_fleche", ATTENTION,
             f"La contre-flèche ({_n(x['w_c'], 1)} mm) dépasse la flèche finale ({_n(x['w_fin'], 1)} mm) : la poutre reste cambrée de "
             f"{_n(-x['w_net_fin'], 1)} mm, le taux de la flèche finale nette est compté nul.")
    if p["systeme"] and x["surfacique"] and ch["entraxe"] > D.ENTRAXE_SYSTEME:
        dire("systeme_entraxe", ATTENTION,
             f"Effet système k_sys = {_n(D.K_SYS, 1)} avec un entraxe de {_n(ch['entraxe'], 2)} m : il suppose des pièces semblables, "
             f"rapprochées ({_n(D.ENTRAXE_SYSTEME, 2)} m environ) et liées par un platelage continu qui répartit les charges "
             f"(EN 1995-1-1 §6.6) ; à justifier.")
    return out


def _hypotheses(p: Dict, x: Dict) -> List[str]:
    out = ["Poutre de section rectangulaire sur deux appuis simples, charge uniformément répartie ; la portée L est comptée "
           "entre axes des appuis.",
           "ELU, NBN EN 1990 (6.10) : deux combinaisons, γ_G·g seule avec k_mod permanent, et γ_G·g + γ_Q·q avec le k_mod de la "
           "charge variable (EN 1995-1-1 §3.1.3(2)) ; chaque vérification retient la plus défavorable.",
           "Effort tranchant pris à l'axe de l'appui, sans la réduction du §6.1.7(3) : côté sûr."]
    typ = p["maintien"]["type"]
    if typ == "continu":
        out.append("Déversement : fibre comprimée maintenue sur toute la longueur (plancher ou platelage fixé) et appuis maintenus "
                   "en torsion : k_crit = 1 (§6.3.3(5)).")
    elif typ == "appuis":
        out.append("Déversement : maintien latéral aux appuis seulement, appuis maintenus en torsion, charge appliquée sur la fibre "
                   f"comprimée : l_ef = {_c(D.LEF_CHARGE_REPARTIE)}·L + {_c(D.LEF_FIBRE_COMPRIMEE)}·h (tableau 6.1).")
    else:
        out.append("Déversement : entretoises qui maintiennent la fibre comprimée, appuis maintenus en torsion ; moment pris constant "
                   f"entre deux entretoises (côté sûr) : l_ef = min(e_m ; {_c(D.LEF_CHARGE_REPARTIE)}·L) + {_c(D.LEF_FIBRE_COMPRIMEE)}·h "
                   "(tableau 6.1).")
    if p["appui"]["verifier"]:
        out.append("Appui : la réaction V_d sur la largeur b et la longueur efficace l_ef,a (§6.1.5) ; la charge portée par les "
                   "abouts au-delà de l'axe de l'appui est négligée.")
    out.append("Flèches en combinaison caractéristique, avec E_0,mean et G_mean ; la part d'effort tranchant est comptée "
               f"(section rectangulaire : facteur {_c(D.FORME_TRANCHANT)}) ; fluage par k_def et ψ₂ (§2.2.3).")
    if p["systeme"]:
        out.append(f"Effet système k_sys = {_n(D.K_SYS, 1)} sur f_m,d et f_v,d (§6.6) : pièces semblables, régulièrement espacées, liées "
                   "par un platelage continu qui répartit les charges.")
    return out


# ------------------------------------------------------------------ le calcul complet
ENONCES = {
    "flexion": ("§6.1.6, §6.3.3", "σ_m,d = M_d / W ≤ k_crit·f_m,d", "σ_m,d", "k_crit·f_m,d", "N/mm²"),
    "cisaillement": ("§6.1.7", f"τ_d = {_c(D.COEF_TAU_RECTANGLE)}·V_d / (k_cr·b·h) ≤ f_v,d", "τ_d", "f_v,d", "N/mm²"),
    "appui": ("§6.1.5", "σ_c,90,d = V_d / (b·l_ef,a) ≤ k_c,90·f_c,90,d", "σ_c,90,d", "k_c,90·f_c,90,d", "N/mm²"),
    "fleche_inst": ("§7.2", "w_inst = w_g + w_q ≤ L / n", "w_inst", "w_lim", "mm"),
    "fleche_fin": ("§2.2.3, §7.2", "w_net,fin = w_fin − w_c ≤ L / n", "w_net,fin", "w_lim", "mm"),
}
# les grandeurs de chaque vérification ELU dans une combinaison : (résistance de calcul, valeur, limite, taux)
_ELU = {"flexion": ("fm_d", "sigma_m", "lim_m", "taux_m"), "cisaillement": ("fv_d", "tau", "fv_d", "taux_v"),
        "appui": ("fc90_d", "sigma_c90", "lim_c90", "taux_c90")}


def verifier(poutre: Dict) -> Dict:
    """Le calcul complet d'une poutre (le format est en tête du fichier)."""
    p = normaliser(poutre)
    x = _calculer(p)
    mat, typ, ch = x["mat"], x["typ"], p["charges"]
    blocs = _blocs(p, x)
    par_cle = {b["cle"]: b for b in blocs}

    def formules(cles):
        return [f for k in cles for f in par_cle[k]["formules"]]

    verifs = []
    for li in _lignes_verifiees(p, x):
        cle = li["cle"]
        article, enonce, symbole, symbole_limite, unite = ENONCES[cle]
        v = {"cle": cle, "nom": NOMS[cle], "groupe": li["groupe"], "article": article, "enonce": enonce, "symbole": symbole,
             "symbole_limite": symbole_limite, "unite": unite, "combinaison": li["combinaison"], "detail": []}
        if li["groupe"] == ELU:
            k_res, k_val, k_lim, k_taux = _ELU[cle]
            for c in x["combs"]:
                v["detail"].append({"combinaison": c["cle"], "nom": c["nom"], "duree": c["duree"], "kmod": c["kmod"],
                                    "resistance": c[k_res], "valeur": c[k_val], "limite": c[k_lim], "taux": c[k_taux],
                                    "etat": _etat(c[k_taux])})
            g = next(d for d in v["detail"] if d["combinaison"] == li["combinaison"])
            v.update(valeur=g["valeur"], limite=g["limite"], taux=g["taux"])
            k = li["combinaison"]
            v["blocs"] = {"flexion": ["charges", f"combinaison_{k}", "section", "deversement", f"flexion_{k}"],
                          "cisaillement": ["charges", f"combinaison_{k}", "section", f"cisaillement_{k}"],
                          "appui": ["charges", f"combinaison_{k}", "appui", f"appui_{k}"]}[cle]
        elif cle == "fleche_inst":
            v.update(valeur=x["w_inst"], limite=x["lim_inst"], taux=x["taux_inst"], blocs=["charges", "section", "fleche_inst"])
        else:
            v.update(valeur=x["w_net_fin"], limite=x["lim_fin"], taux=x["taux_fin"], blocs=["charges", "section", "fleche_inst", "fleche_fin"])
        v["etat"] = _etat(v["taux"])
        v["formules"] = formules(v["blocs"])
        verifs.append(v)

    gov = max(verifs, key=lambda v: v["taux"])
    elu = [v for v in verifs if v["groupe"] == ELU]
    gov_elu = max(elu, key=lambda v: v["taux"])
    c_elu = next(c for c in x["combs"] if c["cle"] == gov_elu["combinaison"])
    lib_m = D.LIBELLES_MAINTIEN[p["maintien"]["type"]]
    return {
        "reference": REFERENCE, "poutre": p,
        "materiau": dict({"classe": p["classe"], "type": typ, "libelle_type": D.LIBELLES_TYPE[typ], "norme": mat["norme"]},
                         **{k: mat[k] for k in D.PROPRIETES}),
        "coefficients": {"service": x["service"], "libelle_service": D.LIBELLES_SERVICE[x["service"]], "gamma_M": x["gamma_M"],
                         "gamma_G": p["gamma_G"], "gamma_Q": p["gamma_Q"], "kdef": x["kdef"], "kcr": D.K_CR, "ksys": x["ksys"],
                         "systeme": p["systeme"], "kh": x["kh"], "kmod_permanent": D.KMOD[x["service"]]["permanent"],
                         "kmod_variable": D.KMOD[x["service"]][x["duree"]], "nature": ch["nature"],
                         "libelle_nature": D.LIBELLES_NATURE[ch["nature"]], "duree": x["duree"],
                         "libelle_duree": D.LIBELLES_DUREE[x["duree"]], "psi2": x["psi2"], "nature_libre": ch["nature"] == D.NATURE_LIBRE},
        "geometrie": {"b": x["b"], "h": x["h"], "standard": p["standard"], "section": p["section"] if p["standard"] else None,
                      "designation": f"{_c(x['b'])} × {_c(x['h'])}", "A": x["A"], "W": x["W"], "I": x["I"], "L": x["L"],
                      "L_mm": x["L_mm"], "elancement": x["h"] / x["b"], "longueur": x["longueur"]},
        "charges": {"mode": ch["mode"], "G": x["G"], "Q": x["Q"], "g_surf": ch["g"] if x["surfacique"] else None,
                    "q_surf": ch["q"] if x["surfacique"] else None, "entraxe": ch["entraxe"] if x["surfacique"] else None,
                    "poids_propre": x["g_pp"], "poids_propre_compte": ch["poids_propre"], "g": x["g"], "q": x["q"]},
        "combinaisons": [{"cle": c["cle"], "nom": c["nom"], "expression": "γ_G·g" if c["cle"] == "G" else "γ_G·g + γ_Q·q",
                          "duree": c["duree"], "libelle_duree": D.LIBELLES_DUREE[c["duree"]], "kmod": c["kmod"], "p_d": c["p_d"],
                          "M_d": c["M_d"], "V_d": c["V_d"]} for c in x["combs"]],
        "combinaison_elu": c_elu["cle"],
        "efforts": {"combinaison": c_elu["cle"], "nom": c_elu["nom"], "p_d": c_elu["p_d"], "M_d": c_elu["M_d"], "V_d": c_elu["V_d"],
                    "R": c_elu["V_d"]},
        "deversement": {"type": p["maintien"]["type"], "libelle": lib_m, "l_ef": x["l_ef"],
                        "entraxe": p["maintien"]["entraxe"] * 1000.0 if p["maintien"]["type"] == "entretoises" else None,
                        "sigma_crit": x["sigma_crit"], "lambda_rel": x["lambda_rel"], "kcrit": x["kcrit"], "branche": x["branche"]},
        "appui": {"verifie": p["appui"]["verifier"], "a": x["a"], "debord": x["debord"], "l1": x["l1"], "cote_travee": x["cote_travee"],
                  "cote_about": x["cote_about"], "l_ef": x["l_ef_a"], "kc90": x["kc90"], "kc90_plein": x["kc90_plein"], "raison": x["raison"]},
        "fleches": {k: x[k] for k in ("w_g_m", "w_g_v", "w_g", "w_q_m", "w_q_v", "w_q", "w_inst", "lim_inst", "w_fin", "w_c",
                                      "w_net_fin", "lim_fin")} | {"n_inst": p["lim_inst"], "n_fin": p["lim_fin"]},
        "verifications": verifs,
        "taux_max": gov["taux"],
        "statut": NOK if any(v["etat"] == NOK for v in verifs) else OK,
        "gouvernant": {"cle": gov["cle"], "nom": gov["nom"], "groupe": gov["groupe"], "taux": gov["taux"], "combinaison": gov["combinaison"]},
        "taux_elu": gov_elu["taux"],
        "taux_els": max(v["taux"] for v in verifs if v["groupe"] == ELS),
        "calcul": blocs,
        "hypotheses": _hypotheses(p, x),
        "avertissements": _avertissements(p, x),
        "non_couvert": list(D.NON_COUVERT),
        "masse_lineique": x["masse_lineique"],
        "masse": x["masse"],
    }


# ------------------------------------------------------------------ le sélecteur de sections
def candidats_sections(poutre: Dict) -> List[Dict]:
    """Les sections standard du TYPE du bois de la poutre, chacune recalculée avec le reste de la poutre (son poids
    propre, son k_h, son déversement, son appui) : ``{nom (« 75x225 »), designation, b, h, A, W, I, masse (kg/m), taux
    (le plus fort), gouvernant (clé), nom_gouvernant, combinaison, ok, retenue, optimal}``, triées par aire
    croissante — donc par masse (à aire égale, la plus étroite d'abord). ``retenue`` : la section standard de la
    poutre (aucune quand elle est sur mesure) ; ``optimal`` : la plus légère qui tient."""
    p = normaliser(poutre)
    typ = E.type_classe(p["classe"])
    out = []
    for s in E.sections(typ):
        q = E.poutre_avec_section(p, s["nom"])
        q = normaliser(q)
        x = _calculer(q)
        lignes = _lignes_verifiees(q, x)
        gov = max(lignes, key=lambda li: li["taux"])
        out.append({"nom": s["nom"], "designation": E.designation(s["nom"]), "b": s["b"], "h": s["h"], "A": x["A"], "W": x["W"],
                    "I": x["I"], "masse": x["masse_lineique"], "taux": gov["taux"], "gouvernant": gov["cle"],
                    "nom_gouvernant": NOMS[gov["cle"]], "combinaison": gov["combinaison"],
                    "ok": all(_etat(li["taux"]) == OK for li in lignes),
                    "retenue": p["standard"] and s["nom"] == p["section"], "optimal": False})
    out.sort(key=lambda c: (c["A"], c["b"]))
    premier = next((c for c in out if c["ok"]), None)
    if premier is not None:
        premier["optimal"] = True
    return out


def optimal(poutre: Dict) -> Optional[Dict]:
    """La section standard la plus légère qui tient (None si aucune ne tient)."""
    return next((c for c in candidats_sections(poutre) if c["optimal"]), None)

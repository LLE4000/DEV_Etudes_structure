# -*- coding: utf-8 -*-
"""Les CAS DE CHARGE d'une barre et leurs COMBINAISONS (brief v12 § 5 ; docs/acier/BARRES_ACIER.md § 5.8).

Un cas : {"nom", "nature", "charges", "N", "courte_duree", "psi0"}. Natures : « permanent » (G) ; les charges
d'exploitation par catégorie, A à H (les fluides et matières stockés : E) ; la neige, le vent, la température ; les
actions d'exécution et les machines, dont ψ₀ est SAISI, sans défaut (tableau A1.1 ANB, notes 1 et 2). Les γ et les ψ
viennent de la table des paramètres (acier/barre/parametres.py). Pas de « tassements ».

EUROCODE (NBN EN 1990 ANB:2021, classe CC2) — pour chaque ARRANGEMENT des cas variables (chacun présent ou absent : la
disposition défavorable, travée par travée), chaque JEU de γ_G (γ_G,sup ou γ_G,inf pour chaque source permanente) et
chaque nature EN TÊTE parmi les présentes :

    (6.10)    Σ γ_G·G + γ_Q·Q_1 + Σ γ_Q·ψ₀,i·Q_i            γ_G,sup = 1,35 ; γ_G,inf = 1,00 ; γ_Q = 1,5
    (6.10a)   Σ γ_G·G + Σ γ_Q·ψ₀,i·Q_i                     toutes à ψ₀ ; Q_1, l'accompagnement principal, à son ψ₀,₁
    (6.10b)   Σ ξ·γ_G,sup·G + γ_Q·Q_1 + Σ γ_Q·ψ₀,i·Q_i     ξ = 0,85 (1,00 en géotechnique) ; γ_G,inf = 1,00

Expression « 6.10 » (le défaut) ou « 6.10ab » : (6.10a) ET (6.10b), versées ensemble — l'enveloppe garde la plus
défavorable, jamais la plus favorable. Sans action variable, (6.10b) vaut ξ fois (6.10a) sur les G défavorables : elle
n'est pas versée. Les cas d'une même nature sont UNE action disposée librement : en tête, ils reçoivent tous γ_Q
((6.10), (6.10b)) ou γ_Q·ψ₀,₁ ((6.10a)).

γ_G PAR SOURCE — À VÉRIFIER : chaque cas permanent est pris comme une source indépendante (le côté sûr ; le texte de
l'EN 1990, non lu, donne un seul γ aux actions d'une même source) : un G défavorable sur une travée, un G favorable sur
l'autre. Deux cas de MÊME répartition (charges proportionnelles, aux mêmes abscisses et niveaux) forment une seule
source : leurs γ mêlés donneraient les mêmes efforts, multipliés par un facteur compris entre les deux extrêmes. Pour n
sources, le nombre de combinaisons est multiplié par 2^(n − 1) ; au-delà de COMBINAISONS_MAX, la barre est REFUSÉE
(« au plus k cas permanents… ») — jamais une combinaison abandonnée en silence.

RÈGLE BELGE DE COURTE DURÉE (tableau A1.1 ANB, note 3 ; mise en œuvre RESTREINTE du brief § 5, un choix du module) :
seules la neige et le vent reçoivent ψ₀ = 0,3 — derrière l'autre (d'office), ou derrière une action déclarée « de
courte durée » (``cas["courte_duree"]``, tous les cas présents de sa nature) ; « derrière » : en tête dans (6.10) et
(6.10b), accompagnement principal dans (6.10a). Option ``regle_courte_duree`` False : ψ₀ du tableau partout. Tout autre
cas garde le ψ₀ du tableau ; quand deux actions de courte durée suivent une action de longue durée, la combinaison porte
« limite » (le module n'applique pas toute la note 3 : ``hypotheses`` le dit).

MÉTHODE ELS (charges de service) — pour chaque arrangement : G + ΣQ, tout à 100 % (décision du bureau du 16/11/2026,
brief § 13.2 point 5).
"""
import math
from itertools import combinations, product
from typing import Dict, List, Optional, Sequence

from . import parametres as PA
from .resultat import Refus, fixe, fr

PERMANENT = "permanent"
SAISI = "saisi"                 # la « clé ψ » d'une nature dont ψ₀ est saisi cas par cas : aucun défaut
# nature → (libellé, clé du triplet ψ dans la table des paramètres ; SAISI : ψ₀ du cas)
NATURES = {
    PERMANENT: ("Permanente (G)", None),
    "exploitation_A": ("Exploitation A — habitation", "psi_A"), "exploitation_B": ("Exploitation B — bureaux", "psi_B"),
    "exploitation_C": ("Exploitation C — lieux de réunion", "psi_C"), "exploitation_D": ("Exploitation D — commerces", "psi_D"),
    "exploitation_E": ("Exploitation E — stockage, fluides", "psi_E"),
    "exploitation_F": ("Exploitation F — véhicules ≤ 30 kN", "psi_F"),
    "exploitation_G": ("Exploitation G — véhicules de 30 à 160 kN", "psi_G"), "exploitation_H": ("Exploitation H — toitures", "psi_H"),
    "neige": ("Neige", "psi_neige"), "vent": ("Vent", "psi_vent"), "temperature": ("Température", "psi_temperature"),
    "execution": ("Exécution — actions pendant les travaux", SAISI), "machines": ("Machines", SAISI),
}
PSI0_SAISI = tuple(n for n, (_, k) in NATURES.items() if k == SAISI)
# l'aide du champ ψ₀ (l'écran la lit ; une grue ou un camion de chantier n'est pas une catégorie F ou G, brief § 5)
AIDE_PSI0 = {"execution": "ψ₀ à saisir : 1,0 au tableau A1.1 ANB (ψ₂ = 0,2), à fixer au cas par cas (note 2).",
             "machines": "ψ₀ à saisir, au cas par cas (tableau A1.1 ANB, note 1) : aucune catégorie par défaut."}
COURTES = ("neige", "vent")     # de courte durée d'office, l'une derrière l'autre (règle belge, brief § 5)
VARIABLES_MAX = 4               # NATURES variables au plus ; le nombre de cas est borné par entree.CAS_MAX
COMBINAISONS_MAX = 512          # combinaisons ELU au plus (la mesure : docs/acier/BARRES_ACIER.md § 5.8)
DEFAUT = {"expression": "6.10", "geotechnique": False, "regle_courte_duree": True}
SUFFIXES = {"6.10": "", "6.10a": " (6.10a)", "6.10b": " (6.10b)"}
HYPOTHESE_SERVICE = "Méthode ELS : G + ΣQ, toutes les charges à 100 % (décision du bureau du 16/11/2026, § 13.2 point 5)."


class AFournir(Refus):
    """ψ₀ d'une action d'exécution ou de machines non saisi, alors qu'il sert (le cas accompagne une autre nature, ou
    l'expression est (6.10a)) : le calcul est BLOQUÉ, valeur à fournir — jamais un ψ₀ deviné. ``cas`` : leurs noms."""

    def __init__(self, noms: Sequence[str]):
        self.cas = list(noms)
        super().__init__(f"ψ₀ à fournir (au cas par cas, tableau A1.1 ANB) : {', '.join(f'« {n} »' for n in self.cas)}")


def _fr(x: float) -> str:
    return fixe(x, 2).rstrip("0").rstrip(".").replace(".", ",")


def _options(options: Optional[Dict]) -> Dict:
    return dict(DEFAUT, **(options or {}))


def _nom(cas: Sequence[Dict], f: Sequence[float], perm: Sequence[int], suffixe: str = "") -> str:
    """« 1,35·G + 1,5·Q » : plusieurs cas permanents au MÊME γ s'écrivent « G » (chacun sinon) ; un facteur 1 ne s'écrit
    pas ; les permanents d'abord ; ``suffixe`` : « (6.10a) » ou « (6.10b) »."""
    var = [i for i in range(len(cas)) if i not in perm]
    if len(perm) > 1 and len({round(f[i], 12) for i in perm}) == 1 and all(cas[i]["nom"] != "G" for i in var):
        termes = [(f[perm[0]], "G")]
    else:
        termes = [(f[i], cas[i]["nom"]) for i in perm]
    termes += [(f[i], cas[i]["nom"]) for i in var]
    return " + ".join(nom if abs(x - 1.0) < 1e-12 else f"{_fr(x)}·{nom}" for x, nom in termes if x != 0) + suffixe


def _trier(cas: Sequence[Dict]):
    if not cas:
        raise Refus("la barre n'a aucun cas de charge")
    perm, var = [], []
    for i, c in enumerate(cas):
        if c.get("nature") not in NATURES:
            raise Refus(f"cas « {c.get('nom')} » : nature inconnue ({c.get('nature')!r})")
        (perm if c["nature"] == PERMANENT else var).append(i)
    n = len({cas[i]["nature"] for i in var})
    if n > VARIABLES_MAX:
        raise Refus(f"au plus {VARIABLES_MAX} natures d'actions variables (la barre en a {n})")
    return perm, var


def _arrangements(var: Sequence[int]):
    for k in range(len(var) + 1):
        for s in combinations(var, k):
            yield s


# ------------------------------------------------------------------ les sources permanentes
def _repartition(c: Dict) -> Optional[Dict]:
    """La répartition d'un cas : {(genre, abscisses, sens, niveau…): valeur}, N et N par tronçon compris ; None quand elle
    n'est pas connue (un cas sans « charges ») — il est alors une source à lui seul."""
    if "charges" not in c:
        return None
    r: Dict = {}

    def ajouter(k, v):
        r[k] = r.get(k, 0.0) + float(v)

    for ch in c.get("charges") or []:
        bout = ch.get("b")
        ajouter((ch.get("type"), round(float(ch.get("a", 0.0)), 9), None if bout is None else round(float(bout), 9), ch.get("sens"),
                 ch.get("niveau"), ch.get("introduction"), ch.get("ss"), ch.get("contact")), ch.get("v", 0.0))
    ajouter(("N",), c.get("N") or 0.0)
    for t in c.get("N_troncons") or []:
        ajouter(("N_troncon", round(float(t["x2"]), 9)), t["N"])
    return {k: v for k, v in r.items() if v != 0.0}


def _proportionnelles(r1: Optional[Dict], r2: Optional[Dict]) -> bool:
    """Deux répartitions connues, aux mêmes clés, dans un même rapport k > 0."""
    if r1 is None or r2 is None or not r1 or set(r1) != set(r2):
        return False
    k0 = next(iter(r1))
    k = r2[k0] / r1[k0]
    return k > 0 and all(abs(r2[x] - k * r1[x]) <= 1e-9 * max(1.0, abs(r2[x])) for x in r1)


def sources(cas: Sequence[Dict], perm: Optional[Sequence[int]] = None) -> List[List[int]]:
    """Les SOURCES permanentes : des listes d'indices de cas. Chaque cas permanent est une source (À VÉRIFIER), sauf
    deux cas de même répartition (proportionnels), qui en font une ; un cas sans charge rejoint la première."""
    perm = [i for i, c in enumerate(cas) if c.get("nature") == PERMANENT] if perm is None else perm
    out: List[List[int]] = []
    vides = []
    for i in perm:
        r = _repartition(cas[i])
        if r == {}:
            vides.append(i)
            continue
        for src in out:
            if _proportionnelles(_repartition(cas[src[0]]), r):
                src.append(i)
                break
        else:
            out.append([i])
    if vides:
        out = [sorted(out[0] + vides)] + out[1:] if out else [vides]
    return out


# ------------------------------------------------------------------ ψ₀ et la règle de courte durée
def _manquants(cas: Sequence[Dict], var: Sequence[int], o: Dict) -> List[str]:
    """Les cas d'exécution ou de machines sans ψ₀ alors qu'il SERT : en (6.10a), toujours ; en (6.10) et (6.10b),
    quand le cas accompagne une autre nature (il y en a une parmi les cas variables)."""
    natures = {cas[i]["nature"] for i in var}
    return [cas[i]["nom"] for i in var if NATURES[cas[i]["nature"]][1] == SAISI and cas[i].get("psi0") is None
            and (o["expression"] == "6.10ab" or len(natures) > 1)]


def _psi0(c: Dict, P: Dict) -> float:
    """ψ₀ du tableau A1.1 ANB (la table des paramètres) ; ψ₀ SAISI pour l'exécution et les machines (lu seulement pour
    elles : une autre nature garde le tableau)."""
    cle = NATURES[c["nature"]][1]
    return float(c["psi0"]) if cle == SAISI else P[cle][0]


def _courte(cas: Sequence[Dict], s: Sequence[int], nature: str) -> bool:
    """Une nature de COURTE DURÉE dans l'arrangement ``s`` : la neige et le vent d'office ; une autre si TOUS ses cas
    présents sont déclarés « de courte durée »."""
    return nature in COURTES or all(cas[i].get("courte_duree") is True for i in s if cas[i]["nature"] == nature)


def _generer(cas: Sequence[Dict], P: Dict, o: Dict, perm: Sequence[int], var: Sequence[int], srcs: List[List[int]]) -> List[Dict]:
    ab = o["expression"] == "6.10ab"
    xi = P["xi_geotechnique"] if o["geotechnique"] else P["xi"]
    regle = o["regle_courte_duree"]
    jeux = list(product((True, False), repeat=len(srcs)))
    out, vus = [], set()
    for expr in (("6.10a", "6.10b") if ab else ("6.10",)):
        g_sup = P["gamma_G_sup"] * (xi if expr == "6.10b" else 1.0)
        for jeu in jeux:
            gam = {i: (g_sup if sup else P["gamma_G_inf"]) for src, sup in zip(srcs, jeu) for i in src}
            for s in _arrangements(var):
                if not s and ((jeu and not any(jeu)) or expr == "6.10b"):
                    continue                    # G seul : 1,00·G = 1,35·G / 1,35 ; (6.10b) = ξ fois (6.10a) sur les G défavorables
                for tete in sorted({cas[i]["nature"] for i in s}) or [None]:
                    f = [0.0] * len(cas)
                    for i, g in gam.items():
                        f[i] = g
                    psi0, courtes, cles = {}, [], set()
                    tete_courte = tete is not None and _courte(cas, s, tete)
                    for i in s:
                        n = cas[i]["nature"]
                        if n == tete and expr != "6.10a":
                            f[i] = P["gamma_Q"]
                            continue
                        p = _psi0(cas[i], P)
                        if NATURES[n][1] != SAISI:
                            cles.add(NATURES[n][1])
                        if regle and n != tete and n in COURTES and tete_courte:
                            p = P["psi0_courte_duree"]
                            courtes.append(cas[i]["nom"])
                        psi0[cas[i]["nom"]] = p
                        f[i] = P["gamma_Q"] * p
                    suivantes = {cas[i]["nature"] for i in s if cas[i]["nature"] != tete and _courte(cas, s, cas[i]["nature"])}
                    limite = bool(regle and tete is not None and not tete_courte and len(suivantes) >= 2 and suivantes & set(COURTES))
                    cle_f = tuple(round(x, 12) for x in f)
                    if cle_f in vus or not any(f):
                        continue
                    vus.add(cle_f)
                    out.append({"nom": _nom(cas, f, perm, SUFFIXES[expr]), "facteurs": f, "expression": expr,
                                "dominante": None if expr == "6.10a" else tete, "principale": tete if expr == "6.10a" else None,
                                "psi": sorted(cles), "psi0": psi0, "courte_duree": courtes, "limite": limite,
                                "gamma_G": {cas[i]["nom"]: ("sup" if g > P["gamma_G_inf"] else "inf") for i, g in gam.items()}})
    return out


def _budget(n: int, n_une: int, n_src: int) -> None:
    """Le BUDGET des γ_G par source : au-delà de COMBINAISONS_MAX (ou du nombre qu'UNE source donne, toujours admis : les
    cas et les natures ont leurs propres bornes), refus « au plus k cas permanents » — k sources en donnent environ
    n_une·2^(k − 1)."""
    plafond = max(COMBINAISONS_MAX, n_une)
    if n <= plafond:
        return
    k = max(1, min(n_src - 1, int(math.floor(math.log2(2 * plafond / max(1, n_une))))))
    quoi = "un seul cas permanent (ou des cas de même répartition)" if k == 1 else f"au plus {k} cas permanents de répartitions différentes"
    raise Refus(f"{n} combinaisons, {plafond} au plus : {quoi}")


def elu(cas: Sequence[Dict], P: Dict, options: Optional[Dict] = None) -> List[Dict]:
    """Les combinaisons fondamentales : [{"nom", "facteurs": [γ de chaque cas], "expression" (« 6.10 » | « 6.10a » |
    « 6.10b »), "dominante" (la nature en tête ; None en (6.10a)), "principale" (en (6.10a)), "psi": [clés ψ du tableau
    employées], "psi0": {cas: ψ₀ employé}, "courte_duree": [cas à ψ₀ = 0,3], "limite" (la règle de courte durée n'est
    pas appliquée en entier), "gamma_G": {cas permanent: « sup » | « inf »}}]. ``options`` : la clé « combinaisons » de
    la barre (acier/barre/entree.py) — expression, géotechnique, règle de courte durée ; None : les défauts. Refus
    (AFournir, un Refus) : un ψ₀ saisi qui manque et sert ; Refus : trop de natures, trop de combinaisons."""
    o = _options(options)
    perm, var = _trier(cas)
    manque = _manquants(cas, var, o)
    if manque:
        raise AFournir(manque)
    srcs = sources(cas, perm)
    out = _generer(cas, P, o, perm, var, srcs)
    if len(out) > COMBINAISONS_MAX:
        _budget(len(out), len(_generer(cas, P, o, perm, var, [sum(srcs, [])] if srcs else [])), len(srcs))
    return out


def controler(cas: Sequence[Dict], options: Optional[Dict] = None, P: Optional[Dict] = None) -> int:
    """Le nombre de combinaisons ELU de ces cas (l'écran peut le demander à la saisie) ; Refus comme ``elu`` — sauf le
    ψ₀ à fournir, qui BLOQUE le calcul sans refuser la barre."""
    o = _options(options)
    P = PA.jeu() if P is None else P
    provisoires = [dict(c, psi0=1.0) if NATURES.get(c.get("nature"), ("", None))[1] == SAISI and c.get("psi0") is None else c for c in cas]
    return len(elu(provisoires, P, o))


def service(cas: Sequence[Dict], options: Optional[Dict] = None) -> List[Dict]:
    """Les combinaisons de service de la méthode ELS : G + ΣQ à 100 %, pour chaque arrangement des cas variables
    (``options`` n'y change rien : la décision du bureau, § 13.2 point 5)."""
    perm, var = _trier(cas)
    out = []
    for s in _arrangements(var):
        f = [0.0] * len(cas)
        for i in list(perm) + list(s):
            f[i] = 1.0
        if any(f):
            out.append({"nom": _nom(cas, f, perm), "facteurs": f, "dominante": None, "psi": []})
    return out


# ------------------------------------------------------------------ les hypothèses (une ligne chacune ; la note les imprime)
def _lignes(tete: str, items: Sequence[str], largeur: int = 120) -> List[str]:
    """« tête : a ; b ; c. » en lignes de ``largeur`` caractères au plus (la suite sans tête)."""
    out, cur = [], tete + " : "
    for k, it in enumerate(items):
        sep = "" if cur.endswith(": ") or not cur else " ; "
        if cur and len(cur) + len(sep) + len(it) + 1 > largeur:
            out.append(cur.rstrip() + " ;")
            cur, sep = "", ""
        cur += sep + it
    out.append(cur + ".")
    return out


def _libelle(nature: str) -> str:
    """« exploitation B », « neige », « exécution »… (le libellé court de la nature, dans une phrase)."""
    t = NATURES[nature][0].split(" — ")[0]
    return t[0].lower() + t[1:]


def hypotheses(options: Optional[Dict], cas: Sequence[Dict], P: Optional[Dict] = None) -> List[str]:
    """Les lignes de la note sur les combinaisons ELU (une chacune, 120 caractères au plus) : l'expression et CC2, les γ
    (et ξ), les ψ₀ employés, la règle de courte durée (appliquée ou non), les actions déclarées de courte durée, sa
    limite, γ_G par source permanente, ce que « G » désigne dans un nom."""
    o = _options(options)
    P = PA.jeu() if P is None else P
    ab = o["expression"] == "6.10ab"
    out = ["Combinaisons ELU : (6.10a) et (6.10b), la plus défavorable retenue ; NBN EN 1990 ANB:2021, classe CC2." if ab
           else "Combinaisons ELU : (6.10), chaque action variable en tête tour à tour ; NBN EN 1990 ANB:2021, classe CC2."]
    out.append(f"γ_G,sup = {fr(P['gamma_G_sup'], 2)} ; γ_G,inf = {fr(P['gamma_G_inf'], 2)} ; γ_Q = {_fr(P['gamma_Q'])} "
               "(tableau A1.2(B) ANB) ; une action variable favorable : 0.")
    if ab:
        xi = P["xi_geotechnique"] if o["geotechnique"] else P["xi"]
        out.append(f"(6.10b) : ξ = {fr(xi, 2)}{' (application géotechnique)' if o['geotechnique'] else ''} ; "
                   f"ξ·γ_G,sup = {fixe(xi * P['gamma_G_sup'], 4).rstrip('0').rstrip('.').replace('.', ',')} ; γ_G,inf = {fr(P['gamma_G_inf'], 2)}.")
    try:
        perm, var = _trier(cas)
    except Refus as e:
        return out + [f"Combinaisons refusées : {e}."]
    items, vus = [], set()
    for i in var:
        n = cas[i]["nature"]
        if NATURES[n][1] == SAISI:
            v = cas[i].get("psi0")
            items.append(f"« {cas[i]['nom']} » ({_libelle(n)}) " + ("à fournir" if v is None else f"{_fr(v)} saisi"))
        elif n not in vus:
            vus.add(n)
            items.append(f"{_libelle(n)} {_fr(P[NATURES[n][1]][0])}")
    if items:
        out += _lignes("ψ₀ (tableau A1.1 ANB)", items)
    try:
        combs = elu(cas, P, o)
    except Refus as e:
        return out + [f"Combinaisons {'bloquées' if isinstance(e, AFournir) else 'refusées'} : {e}."]
    if any(cas[i]["nature"] in COURTES for i in var):
        if not o["regle_courte_duree"]:
            out.append("Règle de courte durée désactivée : ψ₀ du tableau A1.1 ANB partout.")
        elif any(c["courte_duree"] for c in combs):
            out.append(f"Règle de courte durée appliquée (A1.1 ANB, note 3) : ψ₀ = {_fr(P['psi0_courte_duree'])}, "
                       "neige ou vent derrière l'autre ou une action déclarée.")
        else:
            out.append("Règle de courte durée (tableau A1.1 ANB, note 3) : sans effet ici, aucune action de courte durée en tête.")
        declares = [cas[i]["nom"] for i in var if cas[i].get("courte_duree") is True and cas[i]["nature"] not in COURTES]
        if declares and o["regle_courte_duree"]:
            out += _lignes("Déclarées de courte durée", [f"« {n} »" for n in declares])
        if any(c["limite"] for c in combs):
            out.append("Limite : la note 3 n'est pas appliquée en entier — neige et vent derrière une action de longue durée : ψ₀ du tableau.")
    srcs = sources(cas, perm)
    if len(srcs) > 1:
        out += _lignes("γ_G,sup ou γ_G,inf par source permanente (À VÉRIFIER)", [" + ".join(f"« {cas[i]['nom']} »" for i in s) for s in srcs])
    if len(perm) > 1 and all(cas[i]["nom"] != "G" for i in var):
        out += _lignes("G dans un nom, les cas permanents au même γ_G", [" + ".join(cas[i]["nom"] for i in perm)])
    return out

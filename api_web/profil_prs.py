# -*- coding: utf-8 -*-
"""Barres en acier (module « Choix de profilé ») — le PRS À L'ÉCRAN (brief v12 § 18.2, 18.3, 18.4, 18.7) : sa saisie,
sa bibliothèque, le tableau de ses propriétés. Le moteur des PRS est ``acier/barre/prs.py`` : ce fichier ne CALCULE
rien — il lit la section que ``prs.section_prs`` calcule, CHOISIT ce qu'il en montre et l'ÉCRIT.

    LA SAISIE (la rubrique « Barre », sous « Profilé ») : « Catalogue · PRS » ; un PRS se saisit par ses trois plats, en
    trois lignes « b × t » (semelle haute b₁ × t₁ ; âme h_w × t_w, et la gorge a des cordons ; semelle basse b₂ × t₂),
    puis les bords des semelles (oxycoupés · larges plats : la méthode ELS en tire ses courbes ; vides, son verdict est
    « bloqué »). La case « section soudée équivalente » (§ 6.3.2.3) n'est offerte qu'à un PRS BISYMÉTRIQUE (brief § 18.5,
    § 13.2 point 10) : elle écrit ``parametres["prs_soudee_equivalente"]`` ; un PRS qui devient monosymétrique la perd.
    Les chemins : « prs » (catalogue ∣ prs), « prs/b1 » … « prs/t2 », « prs/a » (vide : s = 0), « prs/bords »,
    « prs/soudee » ; « carnet/garder », « carnet/retirer » (la bibliothèque).

    LA BIBLIOTHÈQUE (brief § 18.2) : UN stockage, le CARNET du module (api_web/routeur.py : il voyage avec chaque requête,
    l'écran le garde dans ce navigateur, js/carnet.js) — des enregistrements au format du Catalogue
    (``prs.vers_bibliotheque`` : dimensions, nuance, bords ; AUCUNE propriété), relus par ``prs.de_bibliotheque`` (les
    propriétés toujours RECALCULÉES). La barre garde une COPIE de ses plats (clé « prs ») : la note se suffit.
    À VÉRIFIER : la page Catalogue garde ses profilés personnalisés dans SA note et ne lit pas ce carnet en v1.

    LES PROPRIÉTÉS (brief § 18.7) : ``proprietes(s)`` — [(symbole, valeur écrite, unité, origine)] — le MÊME tableau à
    l'écran (la vue « Section », api_web/dessin_profil.section_prs) et dans la note (le panneau, bloc C5). Recalculées
    à chaque fois, jamais stockées.
"""
from typing import Dict, List, Optional, Sequence, Tuple

from acier import materiaux as MA
from acier.barre import ec3_section as ES
from acier.barre import prs as PR
from acier.barre import section as SE
from acier.barre.resultat import Refus
from acier.profiles import catalogue as CAT

from . import commun as C
from . import profil_mots as M

ROLE = "profil"
N = C.NBSP
SOUDEE = "prs_soudee_equivalente"          # la clé de la table des paramètres (acier/barre/parametres.py), DÉCISION BUREAU n° 10
CARNET_MAX = 40                            # PRS dans la bibliothèque de ce navigateur
CHOIX = (("catalogue", "Catalogue"), ("prs", "PRS"))
BORDS = (("oxycoupes", "oxycoupés"), ("larges_plats", "larges plats"))
# (les clés de BORDS sont celles du moteur, prs.BORDS : tests/test_api_web_profil_prs.py les lie)
# les trois lignes de la saisie : (libellé, ses symboles, largeur, épaisseur, l'aide au survol)
LIGNES = (("semelle haute", "b<sub>1</sub> × t<sub>1</sub>", "b1", "t1",
           "b₁ × t₁ : largeur × épaisseur de la semelle haute (celle du dessus, barre posée), mm"),
          ("âme", "h<sub>w</sub> × t<sub>w</sub> · a", "hw", "tw",
           "h_w × t_w : hauteur entre les semelles × épaisseur de l'âme, mm ; a : gorge des cordons âme-semelles (vide : s = 0)"),
          ("semelle basse", "b<sub>2</sub> × t<sub>2</sub>", "b2", "t2", "b₂ × t₂ : largeur × épaisseur de la semelle basse, mm"))
AIDE_BORDS = "Oxycoupés ou larges plats (doc p. 4) : les courbes de la méthode ELS. Vides : son verdict est bloqué."
AIDE_SOUDEE = ("§ 6.3.2.3 au lieu du § 6.3.2.2 pour le déversement : une déclaration de l'ingénieur, imprimée dans la note "
               "(PRS bisymétrique seulement).")
REFUS_SOUDEE = "« section soudée équivalente » : réservée à un PRS bisymétrique (semelles égales)"


# ------------------------------------------------------------------ la section, lue dans le moteur
def section(el: Dict) -> Dict:
    """La section du PRS de la barre, CALCULÉE par le moteur (``prs.section_prs``) avec la nuance de la barre."""
    return PR.section_prs(**{k: el["prs"].get(k) for k in PR.DIMS + ("a", "bords", "nom")}, nuance=el["nuance"])


def _cat_i() -> List[Dict]:
    """Les profilés en I du catalogue que le module vérifie (IPE, HEA, HEB…)."""
    return [p for p in CAT.profils() if SE.FAMILLES.get(p["famille"], ("",))[0] == "I"]


def plats_de(nom: str) -> Dict:
    """Les plats d'un PRS NEUF, ceux du profilé en I du catalogue ``nom`` (un IPE 300 : 150 × 10,7 / 278,6 × 7,1 /
    150 × 10,7) — un tube ou un nom inconnu : ceux de l'IPE 300 ; gorge et bords NON SAISIS (jamais supposés)."""
    rec = CAT.par_nom().get(nom) or {}
    if not (rec.get("b") and rec.get("tf") and rec.get("h") and rec.get("tw")):
        rec = CAT.par_nom()["IPE 300"]
    b, tf, tw = (round(float(rec[k]), 3) for k in ("b", "tf", "tw"))
    return {"b1": b, "t1": tf, "hw": round(float(rec["h"]) - 2.0 * tf, 3), "tw": tw, "b2": b, "t2": tf, "a": None, "bords": None, "nom": None}


def proche_catalogue(s: Dict) -> str:
    """Le profilé en I du catalogue le plus proche d'un PRS (h, puis b) : celui que la barre retrouve en revenant au
    catalogue (le sélecteur en propose d'autres)."""
    return min(_cat_i(), key=lambda p: (abs(float(p["h"]) - s["h"]) + abs(float(p["b"]) - max(s["prs"]["b1"], s["prs"]["b2"])), p.get("masse") or 0.0))["nom"]


def norme(s: Dict) -> str:
    """La norme produit des plats (« EN 10025-2:2004 ») : celle de la table des aciers (acier/materiaux.py)."""
    return MA.ligne(s["nuance"], ES.produit(s))["norme"]


# ------------------------------------------------------------------ les propriétés (brief § 18.7)
def proprietes(s: Dict) -> List[Tuple[str, str, str, str]]:
    """Le tableau des propriétés d'un PRS : [(symbole, valeur écrite, unité, origine)] — A, z_g, I_y, I_z, W_el,y (aux
    deux fibres d'un PRS monosymétrique), W_pl,y, I_t, I_w ; z_s, z_j et ψ_f d'un PRS monosymétrique (semelle haute
    comprimée) ; f_y de chaque plat et f_y de calcul. Tout est lu dans la section du moteur ; rien n'est stocké."""
    p, mono, n = s["prs"], s["monosymetrique"], M.n
    rangs = [("A", n(s["A"] / 1e2, 2), "cm²", "trois plats, cordons non comptés"),
             ("z_g", n(s["z_g"], 1), "mm", "depuis la face inférieure"),
             ("I_y", n(s["Iy"] / 1e4, 1), "cm⁴", "trois plats"),
             ("I_z", n(s["Iz"] / 1e4, 1), "cm⁴", "trois plats")]
    if mono:
        rangs += [("W_el,y,haut", n(s["Wel_y_haut"] / 1e3, 1), "cm³", "fibre haute : I_y / (h − z_g)"),
                  ("W_el,y,bas", n(s["Wel_y_bas"] / 1e3, 1), "cm³", "fibre basse : I_y / z_g")]
    else:
        rangs.append(("W_el,y", n(s["Wel_y"] / 1e3, 1), "cm³", "I_y / (h/2)"))
    rangs += [("W_pl,y", n(s["Wpl_y"] / 1e3, 1), "cm³", f"axe neutre plastique à {n(s['z_pl'], 1)} mm"),
              ("I_t", n(s["It"] / 1e4, 2), "cm⁴", "Σ (b·t³/3)·(1 − 0,63·t/b), côté sûr"),
              ("I_w", n(s["Iw"] / 1e6, 0), "cm⁶", "h_s²·I₁·I₂/(I₁ + I₂), semelles seules")]
    if mono:
        zj = PR.zj_psi(s, PR.HAUT)
        rangs += [("z_s", n(s["z_s"], 1), "mm", "C au-dessus de G (semelles seules)"),
                  ("z_j", n(zj["z_j"], 2), "mm", "annexe E ANB, semelle haute comprimée"),
                  ("ψ_f", n(zj["psi_f"], 4), "", "annexe E ANB, semelle haute comprimée")]
    fy = ES.limites_parois(s["nuance"], s)
    tab = ES.tableau_fy(s["nuance"], s)
    for cle, sym, t, ep in (("semelle_haut", "f_y,f1", "t₁", p["t1"]), ("ame", "f_y,w", "t_w", p["tw"]), ("semelle_bas", "f_y,f2", "t₂", p["t2"])):
        rangs.append((sym, n(fy[cle], 0), "N/mm²", f"{t} = {PR.mm(ep)} mm, {tab}"))
    rangs.append(("f_y", n(ES.limite_elastique(s["nuance"], s), 0), "N/mm²", "la plus petite des trois (calcul)"))
    return rangs


def fiche(el: Dict) -> Dict:
    """La fiche du PRS de la barre, à la forme de ``profil.fiche_profil`` : {"famille", "dims", "dimensions" et
    "proprietes" [(symbole, valeur écrite, unité)], "tableau" (``proprietes``, avec l'origine), "sous"}."""
    s = section(el)
    p = s["prs"]
    dims = [("b_1", PR.mm(p["b1"]), "mm"), ("t_1", PR.mm(p["t1"]), "mm"), ("h_w", PR.mm(p["hw"]), "mm"), ("t_w", PR.mm(p["tw"]), "mm"),
            ("b_2", PR.mm(p["b2"]), "mm"), ("t_2", PR.mm(p["t2"]), "mm"), ("h", PR.mm(s["h"]), "mm")]
    dims.append(("a", PR.mm(p["a"]), "mm") if p["a"] is not None else ("a", "non saisie", ""))
    tab = proprietes(s)
    return {"famille": PR.FAMILLE, "dims": dict(p), "dimensions": dims, "proprietes": [(a, b, c) for a, b, c, _ in tab], "tableau": tab,
            "sous": sous_titre(s)}


def sous_titre(s: Dict) -> str:
    return f"PRS {'monosymétrique' if s['monosymetrique'] else 'bisymétrique'} · tôles {norme(s)}"


# ------------------------------------------------------------------ la bibliothèque : le carnet du module
def lire_carnet(carnet) -> List[Dict]:
    """Le carnet reçu de l'écran, RELU : des PRS au format du Catalogue (``prs.vers_bibliotheque``), chacun recalculé par
    ``prs.de_bibliotheque`` — ce qui ne se relit pas est écarté, un nom en double aussi (le premier reste)."""
    out, vus = [], set()
    for e in carnet if isinstance(carnet, list) else []:
        try:
            s = PR.de_bibliotheque(e)
        except (Refus, ValueError, TypeError, KeyError, AttributeError):
            continue
        if s["nom"] not in vus:
            vus.add(s["nom"])
            out.append(PR.vers_bibliotheque(s))
    return out[:CARNET_MAX]


def enregistrement(el: Dict) -> Optional[Dict]:
    """L'enregistrement qu'aurait le PRS de la barre dans la bibliothèque (``prs.vers_bibliotheque`` : plats, gorge, bords,
    nuance) ; None pour un profilé du catalogue ou un PRS que le moteur refuse."""
    if not el.get("prs"):
        return None
    try:
        return PR.vers_bibliotheque(section(el))
    except (Refus, ValueError, TypeError, KeyError):
        return None


def dans_carnet(el: Dict, carnet: Sequence[Dict]) -> bool:
    """Le PRS de la barre est-il dans la bibliothèque — le MÊME enregistrement, et non le seul nom : le nom ne porte ni la
    nuance, ni la gorge, ni les bords (intégration du lot C : deux PRS aux mêmes plats, en S235 et en S355, étaient pris
    l'un pour l'autre — « Retirer » au lieu d'« Enregistrer », et le sélecteur cochait l'autre, avec son taux)."""
    e = enregistrement(el)
    return e is not None and any(x == e for x in lire_carnet(list(carnet or [])))


def garder(el: Dict, carnet) -> Tuple[List[Dict], str]:
    """(la bibliothèque, le message) : le PRS de la barre y est enregistré (un PRS de même nom y est remplacé)."""
    if not el.get("prs"):
        raise ValueError("seul un PRS s'enregistre dans la bibliothèque (un profilé du catalogue y est déjà)")
    e = PR.vers_bibliotheque(section(el))
    reste = [x for x in lire_carnet(carnet) if x["nom"] != e["nom"]]
    if len(reste) >= CARNET_MAX:
        raise ValueError(f"au plus {CARNET_MAX} PRS dans la bibliothèque : en retirer un d'abord")
    return sorted(reste + [e], key=lambda x: x["nom"]), f"{e['nom']} enregistré dans la bibliothèque (ce navigateur)"


def retirer(el: Dict, carnet) -> Tuple[List[Dict], str]:
    """(la bibliothèque, le message) : le PRS de la barre en est retiré (la barre le garde)."""
    lu = lire_carnet(carnet)
    reste = [x for x in lu if x["nom"] != el.get("profil")]
    if len(reste) == len(lu):
        raise ValueError(f"{el.get('profil')} n'est pas dans la bibliothèque")
    return reste, f"{el['profil']} retiré de la bibliothèque"


def prendre(el: Dict, entree: Dict) -> None:
    """La barre prend le PRS d'un enregistrement de la bibliothèque : une COPIE de ses plats (clé « prs », la note se
    suffit) et sa nuance (une seule nuance par PRS : celle de l'enregistrement passe devant celle de la barre)."""
    s, el["prs"] = plats_entree(entree, el["nuance"])
    el["nuance"] = s["nuance"]
    if s["monosymetrique"]:
        el["parametres"].pop(SOUDEE, None)


# ------------------------------------------------------------------ la saisie : les chemins « prs/… »
def _nombre(valeur, quoi: str) -> float:
    x = C.lire_nombre(valeur)
    if x is None:
        raise ValueError(f"{quoi} : « {valeur} » n'est pas un nombre")
    return x


def appliquer(el: Dict, cles: List[str], valeur) -> Optional[str]:
    """Écrit une saisie du PRS dans ``el`` (une copie : profil.modifier la relit, une saisie impossible est refusée en
    une ligne par ``prs.section_prs``). KeyError : le champ n'existe pas ; ValueError : la valeur est refusée."""
    if cles == ["prs"]:
        if valeur not in dict(CHOIX):
            raise ValueError(f"profilé : « catalogue » ou « prs » — reçu « {valeur} »")
        if valeur == "prs" and not el.get("prs"):
            el["prs"] = plats_de(el["profil"])
        elif valeur == "catalogue" and el.get("prs"):
            el["profil"] = proche_catalogue(section(el))
            el["prs"] = None
            el["parametres"].pop(SOUDEE, None)
        return None
    if len(cles) != 2 or not el.get("prs"):
        raise KeyError("/".join(cles))
    k, p = cles[1], el["prs"]
    if k in PR.DIMS:
        p[k] = _nombre(valeur, {"b1": "b₁", "t1": "t₁", "hw": "h_w", "tw": "t_w", "b2": "b₂", "t2": "t₂"}[k])
        p["nom"] = None                               # (le nom suit les plats : « PRS 600x300x15x30 »)
        if (p["b1"], p["t1"]) != (p["b2"], p["t2"]):
            el["parametres"].pop(SOUDEE, None)        # (jamais pour un monosymétrique)
    elif k == "a":
        p["a"] = None if valeur is None or (isinstance(valeur, str) and not valeur.strip()) else _nombre(valeur, "gorge a")
    elif k == "bords":
        if valeur not in dict(BORDS) and valeur not in ("", None):
            raise ValueError(f"bords des semelles : oxycoupés ou larges plats — reçu « {valeur} »")
        p["bords"] = valeur or None
    elif k == "soudee":
        oui = C.lire_bool(valeur)
        if oui is None:
            raise ValueError(f"case attendue (oui ou non), reçu « {valeur} »")
        if oui and (p["b1"], p["t1"]) != (p["b2"], p["t2"]):
            raise ValueError(REFUS_SOUDEE)
        if oui:
            el["parametres"][SOUDEE] = True
        else:
            el["parametres"].pop(SOUDEE, None)
    else:
        raise KeyError("/".join(cles))
    return None


# ------------------------------------------------------------------ la rubrique « Barre » : le profilé
def _ligne_plat(libelle: str, sym: str, kb: str, kt: str, aide: str, p: Dict) -> str:
    """Une ligne « b × t » : le libellé (l'aide dans son icône), la largeur, l'épaisseur — et, sur la ligne de l'âme, la gorge."""
    gorge = ""
    if kb == "hw":
        gorge = (f'<span class="pf-x">a</span>{C.num(["prs", "a"], p.get("a"), "", dec=3, mini=0, auto="s = 0")}')
    # (intégration du lot C, R1) l'aide dans l'icône .aide[title] : au doigt, etabli.js n'ouvre qu'elle (un title sur le label ne se lit qu'au survol)
    return (f'<div class="pf-plat{" ame" if gorge else ""}"><label><span>{libelle}</span> <span class="pf-sym">{sym}</span>'
            f'<span class="aide" title="{C.esc(aide)}">{C.icon("info")}</span></label>'
            f'{C.num(["prs", kb], p[kb], "", dec=3, mini=0)}<span class="pf-x">×</span>{C.num(["prs", kt], p[kt], "", dec=3, mini=0)}{gorge}</div>')


def bouton(el: Dict, r: Dict) -> str:
    """Le bouton du profilé : son nom et sa masse (ρ·A pour un PRS, ``prs.MASSE_VOLUMIQUE``) ; il ouvre le sélecteur
    — la bibliothèque des PRS pour un PRS (sans ★ : elle n'a pas de « plus léger qui tient »)."""
    masse = f"{M.n(r['masse'], 1)}{N}kg/m" if r.get("masse") else ""
    titre = ("Choisir un PRS de la bibliothèque (ce navigateur)" if el.get("prs") else
             "Choisir le profilé dans la liste (★ : le plus léger qui tient)")
    return (f'<button type="button" class="pf-prof" data-pk="{ROLE}" title="{titre}">'
            f'<b>{C.esc(el["profil"])}</b><span>{masse}</span>{C.icon("chevron-down")}</button>')


def rubrique(el: Dict, r: Dict, carnet: Optional[Sequence[Dict]] = None) -> str:
    """Sous « Acier » : « Catalogue · PRS » ; pour un PRS, ses trois lignes de plats, les bords, la case « section soudée
    équivalente » (bisymétrique seulement), h et la norme produit en une ligne d'aide ; puis le bouton du profilé, et
    pour un PRS, l'enregistrer dans la bibliothèque (ou l'en retirer)."""
    p = el.get("prs")
    s = C.segments(["prs"], "prs" if p else "catalogue", CHOIX)
    if not p:
        return s + bouton(el, r)
    sec = section(el)
    s += "".join(_ligne_plat(*x, p) for x in LIGNES)
    s += C.ligne("Bords", C.segments(["prs", "bords"], p.get("bords"), BORDS), aide=AIDE_BORDS, full=True)
    if not sec["monosymetrique"]:
        s += C.case(["prs", "soudee"], bool(el["parametres"].get(SOUDEE)), "Section soudée équivalente (§ 6.3.2.3)")
    s += C.aide(f"h = t₁ + h<sub>w</sub> + t₂ = {PR.mm(sec['h'])}{N}mm · tôles {norme(sec)} · cotes en mm")
    s += bouton(el, r)
    if dans_carnet(el, carnet or []):
        s += (f'<button type="button" class="btn sm pf-biblio" data-chemin="carnet/retirer" data-valeur="1" '
              f'title="Retirer ce PRS de la bibliothèque de ce navigateur (la barre le garde)">{C.icon("trash-2")}Retirer de la bibliothèque</button>')
    else:
        s += (f'<button type="button" class="btn sm pf-biblio" data-chemin="carnet/garder" data-valeur="1" '
              f'title="Garder ce PRS pour une autre barre, une autre note (ce navigateur)">{C.icon("plus")}Enregistrer dans la bibliothèque</button>')
    return s


# ------------------------------------------------------------------ le sélecteur : la bibliothèque des PRS
def ligne_selecteur(s: Dict, taux: Optional[float]) -> str:
    """Ce que le sélecteur écrit d'un PRS survolé : ses plats, son aire, son inertie, son module plastique, son taux."""
    p = s["prs"]
    eta = f" · η = {C.fr(taux, 3)}" if taux is not None else ""
    return (f"<b>{C.esc(s['nom'])}</b> · {PR.mm(p['b1'])} × {PR.mm(p['t1'])} / {PR.mm(p['hw'])} × {PR.mm(p['tw'])} / "
            f"{PR.mm(p['b2'])} × {PR.mm(p['t2'])} mm · {C.esc(s['nuance'])} · A = {M.n(s['A'] / 1e2, 2)} cm² · "
            f"I<sub>y</sub> = {M.n(s['Iy'] / 1e4, 1)} cm⁴{eta}")


def plats_entree(entree: Dict, nuance: str) -> Tuple[Dict, Dict]:
    """(la section, les plats au format de la barre) d'un enregistrement de la bibliothèque."""
    s = PR.de_bibliotheque(entree, nuance)
    return s, {**{k: s["prs"][k] for k in PR.DIMS + ("a",)}, "bords": s["bords"], "nom": s["nom"]}

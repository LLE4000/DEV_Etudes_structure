# -*- coding: utf-8 -*-
"""Le matériau d'une PIÈCE d'assemblage : f_y et f_u selon SON épaisseur, β_w.

NBN EN 1993-1-1 ANB:2018, 3.2.1(1) : « L'option a est normative » — f_y et f_u sont ceux de la norme de produit,
selon l'épaisseur nominale (tableaux 3.3 ANB et 3.4 ANB). Les valeurs se lisent dans la table unique du site
(``acier/materiaux.py``, par ``acier.bibliotheques.acier``) ; ce fichier n'en porte AUCUNE. Il dit seulement QUELLE
épaisseur décide, et il pose le résultat sur le calcul ``R`` des sept moteurs d'assemblage (les six attaches
poutre–poutre et poutre–poteau, le pied de poteau).

Quelle épaisseur décide (règle commune du site, du côté sûr) :
- une pièce simple (plat d'âme, platine, cornière) : SON épaisseur ;
- un profilé en I ou H vu comme un tout (poutre portée, poutre porteuse, poteau) : sa paroi la plus épaisse,
  max(t_f, t_w), donc la plus petite f_y. L'EN 1993-1-1 3.2.1 ne dit pas quelle épaisseur retenir pour une
  section : c'est une convention du site, la même que les barres et le flambement.
f_u suit la même épaisseur (constante de 3 à 100 mm au tableau 3.3 ANB).

« Personnalisé » : f_y, f_u et β_w saisis par l'utilisateur, tels quels (inchangé).

Le DOMAINE d'une règle écrite pour une nuance (poutre grugée : limites S275 / S355 du guide) se lit sur la NUANCE
(``nominale``), jamais sur la f_y réduite par l'épaisseur : une S275 de 20 mm (f_y = 265) reste une S275.

Un BENCHMARK (exemple publié, calcul à la main) et la PARITÉ avec le corrigé de référence donnent au moteur LEUR
matériau par le paramètre ``materiau`` de ``compute`` : une fonction (nom, t) → {"fy", "fu", "bw"} qui remplace la
table du site le temps d'un calcul — une validation de formules, pas une note. ``fixe`` en fabrique une.

Un calcul hors des tableaux (pièce plus épaisse que la dernière ligne) n'est pas lancé : ``MA.HorsTable`` (une
ValueError), qui nomme la pièce, la nuance et la borne — l'écran l'affiche en « Calcul impossible ».
"""
from acier import bibliotheques as BI
from acier import materiaux as MA
from acier.bibliotheques import DB, PERSO, find

# Les épaisseurs que le prédimensionnement propose pour un plat, une platine ou une cornière (DB["epais"], 8 à 15 mm) :
# toutes dans la première bande du tableau 3.3 ANB (t ≤ 16 mm), donc une seule f_y. Le prédimensionnement lit
# l'acier à la PLUS ÉPAISSE d'entre elles (la plus petite f_y si la liste changeait) ; la vérification relit ensuite
# l'acier à l'épaisseur retenue.
T_PREDIM = float(max(DB["epais"]))
# Une épaisseur nulle ou absente (saisie incomplète, que le moteur signale par ses alertes) : la pièce n'a pas de
# résistance et son acier n'y change rien ; on lit la première bande où f_u est constante (≥ 3 mm), sans lever.
T_SANS_EPAISSEUR = 3.0


def table_du_site(nom, t):
    """L'acier ``nom`` pour une pièce laminée d'épaisseur ``t`` (mm) : la table unique du site."""
    return BI.acier(nom, t)


def t_profil(tf, tw):
    """L'épaisseur qui décide d'un profilé en I ou H vu comme un tout : sa paroi la plus épaisse (convention du
    site, du côté sûr : la plus petite f_y)."""
    return max(tf, tw)


def t_ame(tf, tw):
    """L'épaisseur qui décide d'une exigence de SUR-RÉSISTANCE portée par l'ÂME d'un profilé (gorge de pleine
    résistance a/t_w = β_w·γ_M2·f_y/(√2·f_u), MSB P5 §2.2) : l'âme à SA propre épaisseur, t_w. Une f_y plus BASSE y
    rendrait l'exigence MOINS sévère : la règle « plus petite f_y » (``t_profil``, max(t_f, t_w)) vaut pour les
    RÉSISTANCES, pas pour une exigence de sur-résistance ou de ductilité. ``tf`` est reçu pour que l'appel nomme le
    profilé entier (et qu'une mutation puisse rejouer l'ancienne règle)."""
    return tw


def lire(nom, t, perso=None, materiau=None, piece=""):
    """Le matériau d'une pièce : {"fy", "fu", "bw", "t", "nominale", "source"}.

    ``nom`` : la nuance de la bibliothèque, ou ``PERSO`` (alors ``perso`` = {"fy", "fu", "bw"} saisis) ; ``t`` :
    l'épaisseur qui décide (mm) ; ``materiau`` : la fonction (nom, t) → acier qui remplace la table du site
    (benchmark, parité) ; ``piece`` : son nom, pour le refus hors tableau. ``nominale`` est le nombre du nom de la
    nuance (S275 → 275), pour un contrôle de domaine ; en « Personnalisé », la f_y saisie."""
    if nom == PERSO:
        p = perso or {}
        return {"fy": p.get("fy"), "fu": p.get("fu"), "bw": p.get("bw"), "t": t, "nominale": p.get("fy"),
                "source": "saisie (Personnalisé)"}
    t_lu = t if t > 0 else T_SANS_EPAISSEUR
    try:
        a = (materiau or table_du_site)(nom, t_lu)
    except MA.HorsTable as e:
        raise MA.HorsTable(f"{piece} : {e}" if piece else str(e)) from None
    return {"fy": a["fy"], "fu": a["fu"], "bw": a["bw"], "t": t, "nominale": MA.nominale(nom),
            "source": (MA.source(nom, t_lu) if materiau is None else "matériau de l'exemple")}


def poser(R, X, nom, t, perso=None, materiau=None, cle=None, piece=""):
    """Pose le matériau de la pièce ``X`` sur le calcul : ``R.fy_X``, ``R.fu_X``, ``R.bw_X`` (les clés du moteur de
    référence) et ``R.materiau[X]`` (épaisseur, nuance, valeur nominale, source, clé de saisie — un dictionnaire, hors
    des scalaires comparés au corrigé)."""
    return enregistrer(R, X, lire(nom, t, perso, materiau, piece), nom, cle, piece)


def enregistrer(R, X, m, nom, cle=None, piece=""):
    """Pose sur ``R`` un matériau déjà lu (``lire``, ou la règle propre d'un moteur pour son « Personnalisé »)."""
    R["fy_" + X] = m["fy"]
    R["fu_" + X] = m["fu"]
    R["bw_" + X] = m["bw"]
    if not isinstance(R.get("materiau"), dict):
        R["materiau"] = {}
    R["materiau"][X] = dict(m, nom=nom, cle=cle, piece=piece)
    return m


def nominale(R, X):
    """La valeur nominale de la nuance de la pièce ``X`` (contrôle de domaine) ; à défaut, sa f_y."""
    m = (R.get("materiau") or {}).get(X)
    return m["nominale"] if m else R["fy_" + X]


def fixe(valeurs):
    """Un matériau FIXÉ par nuance, sans épaisseur : celui d'un exemple publié ou d'un calcul à la main (benchmark),
    ou celui du corrigé de référence (parité). ``valeurs`` : {nom: (fy, fu)} ; β_w de la bibliothèque. Une nuance
    absente lève (aucune valeur n'est inventée)."""
    def lire_fixe(nom, t):
        if nom not in valeurs:
            raise ValueError(f"nuance {nom!r} absente du matériau de l'exemple")
        fy, fu = valeurs[nom]
        return {"n": nom, "fy": float(fy), "fu": float(fu), "bw": find(DB["aciers"], nom)["bw"]}
    return lire_fixe


# Le matériau des exemples PUBLIÉS et des calculs À LA MAIN des onglets Benchmark (validation de formules, pas des
# notes) : celui de leur source, l'EN 1993-1-1 tableau 3.1 (t ≤ 40 mm) — S275 du guide (f_y = 275, f_u = 430),
# S355 des calculs à la main (f_y = 355, f_u = 510), sans épaisseur.
EXEMPLES = fixe({"S275": (275.0, 430.0), "S355": (355.0, 510.0)})


def texte_nuances():
    """Le texte d'aide des nuances (aide à l'écran et rubrique Méthode des attaches) : la règle belge, et les valeurs
    du S355 LUES dans la table (aucun nombre recopié)."""
    fy16, fy40, fu = MA.fy("S355", 16), MA.fy("S355", 40), MA.fu("S355", 10)
    return (f"fy et fu selon l'épaisseur de chaque pièce : NBN EN 1993-1-1 ANB:2018 3.2.1(1) (option a), "
            f"tableau 3.3 ANB (EN 10025-2) ; S355 N/NL et S460 N/NL : tableau 3.4 ANB. Plat, platine, cornière : son "
            f"épaisseur ; profilé : sa paroi la plus épaisse, max(tf, tw). S355 : fy = {fy16:g} MPa jusqu'à 16 mm, "
            f"{fy40:g} MPa de 16 à 40 mm ; fu = {fu:g} MPa de 3 à 100 mm. La ligne « S355 (fu 490) » suit le S355 "
            f"belge (fu = {fu:g} MPa). « Personnalisé » : fy, fu et βw saisis.")

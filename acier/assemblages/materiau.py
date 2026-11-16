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
from acier.assemblages import mecanique as MEC
from acier.bibliotheques import DB, PERSO, find
from acier.formats import F

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
        raise MA.HorsTable(refus_hors_table(nom, t_lu, piece, e)) from None
    return {"fy": a["fy"], "fu": a["fu"], "bw": a["bw"], "t": t, "nominale": MA.nominale(nom),
            "source": (MA.source(nom, t_lu) if materiau is None else "matériau de l'exemple")}


def refus_hors_table(nom, t, piece, e):
    """La phrase d'un calcul refusé par la table (``e`` : son ``MA.HorsTable``), telle que l'écran l'écrit sous
    « Calcul impossible » : la pièce, la nuance, le produit, l'épaisseur et la borne du tableau de l'ANB — « plat :
    S355 laminé de 300 mm, au-delà du tableau 3.3 ANB (250 mm) ». Lue sur la table (``MA.ligne``,
    ``MA.epaisseur_max``), aucune borne recopiée ; un autre refus (nuance inconnue…) garde la phrase de la table."""
    tete = f"{piece} : " if piece else ""
    try:
        sup = MA.epaisseur_max(nom)
        if t > sup:
            return (f"{tete}{nom} {MA.LIBELLES_PRODUIT['lamine']} de {t:g} mm, au-delà du tableau "
                    f"{MA.ligne(nom)['tableau']} ({sup:g} mm)")
    except MA.HorsTable:
        pass
    return f"{tete}{e}"


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
# … et ce que l'onglet Benchmark en dit, cas par cas (``texte_exemple``) : (source de ces valeurs, ce qu'en fait une note)
SOURCE_EXEMPLES = ("EN 1993-1-1 tab. 3.1, sans épaisseur", "une note prend le tableau 3.3 ANB")
# un cas rejoué sur la table du site (pied de poteau, vérification complète) : le matériau d'une note
SOURCE_TABLE = ("tableau 3.3 ANB, selon l'épaisseur de chaque pièce", "le matériau d'une note")


def texte_exemple(R, source=SOURCE_EXEMPLES):
    """« matériau de l'exemple : S355, f_y 355 / f_u 510 (EN 1993-1-1 tab. 3.1, sans épaisseur) — une note prend le
    tableau 3.3 ANB » : le matériau qu'un cas de l'onglet Benchmark a REÇU, lu sur son calcul (``R.materiau``, pièce
    par pièce ; aucun nombre recopié), d'où viennent ces valeurs et ce qu'en fait une note (``source``). Un benchmark
    valide des FORMULES sur le matériau de son exemple : le lecteur doit savoir qu'une note, avec les mêmes données,
    prend la table du site. Une nuance qui revient avec deux matériaux (deux épaisseurs) nomme ses pièces, un
    « Personnalisé » aussi ; un « Personnalisé » dit « valeurs saisies » et vient EN TÊTE, pour que la source entre
    parenthèses ne soit attribuée qu'aux pièces qui la tiennent (« « Personnalisé » (cornières), valeurs saisies f_y
    430 / f_u 430 ; S275, f_y 275 / f_u 430 (EN 1993-1-1 tab. 3.1, sans épaisseur) — une note prend… »)."""
    groupes = {}
    for m in (R.get("materiau") or {}).values():
        nom = "« Personnalisé »" if m.get("nom") == PERSO else m.get("nom")
        pieces = groupes.setdefault((nom, m.get("fy"), m.get("fu")), [])
        if m.get("piece") and m["piece"] not in pieces:
            pieces.append(m["piece"])
    noms = [n for n, _, _ in groupes]
    parts, saisis = [], []
    for (nom, fy, fu), pieces in groupes.items():
        qui = f" ({', '.join(pieces)})" if pieces and (noms.count(nom) > 1 or nom == "« Personnalisé »") else ""
        if nom == "« Personnalisé »":
            # un « Personnalisé », ce sont les valeurs SAISIES : la source de la table (``source``) ne le vise pas
            saisis.append(f"{nom}{qui}, valeurs saisies f_y {fy:g} / f_u {fu:g}")
        else:
            parts.append(f"{nom}{qui}, f_y {fy:g} / f_u {fu:g}")
    origine, note = source
    if not parts:
        return f"matériau de l'exemple : {' ; '.join(saisis)}"
    return f"matériau de l'exemple : {''.join(s + ' ; ' for s in saisis)}{' ; '.join(parts)} ({origine}) — {note}"


def avec_exemple(texte, R, source=SOURCE_EXEMPLES):
    """``texte`` (titre ou ligne de données d'un cas de l'onglet Benchmark) suivi de ``texte_exemple``."""
    t = (texte or "").rstrip()
    phrase = texte_exemple(R, source)
    if not t:
        return phrase[0].upper() + phrase[1:]
    if t.endswith("."):
        return f"{t} {phrase[0].upper()}{phrase[1:]}"
    return f"{t} — {phrase}"


def texte_nuances():
    """Le texte d'aide des nuances (aide à l'écran et rubrique Méthode des attaches) : la règle belge, et les valeurs
    du S355 LUES dans la table (aucun nombre recopié)."""
    fy16, fy40, fu = MA.fy("S355", 16), MA.fy("S355", 40), MA.fu("S355", 10)
    return (f"fy et fu selon l'épaisseur de chaque pièce : NBN EN 1993-1-1 ANB:2018 3.2.1(1) (option a), "
            f"tableau 3.3 ANB (EN 10025-2) ; S355 N/NL et S460 N/NL : tableau 3.4 ANB. Plat, platine, cornière : son "
            f"épaisseur ; profilé : sa paroi la plus épaisse, max(tf, tw). S355 : fy = {fy16:g} MPa jusqu'à 16 mm, "
            f"{fy40:g} MPa de 16 à 40 mm ; fu = {fu:g} MPa de 3 à 100 mm. La ligne « S355 (fu 490) » suit le S355 "
            f"belge (fu = {fu:g} MPa). « Personnalisé » : fy, fu et βw saisis.")


# Les nuances dont l'aide du champ « Gorge a » des platines d'about dit le coefficient de pleine résistance : celles
# du guide (MSB P5 §2.2)
NUANCES_PLEINE_RESISTANCE = ("S235", "S275", "S355")


def coefs_pleine_resistance(gM2, gM0=1.0, nuances=NUANCES_PLEINE_RESISTANCE):
    """(t_w max, [(nuance, a/t_w)]) : le coefficient de la gorge de PLEINE RÉSISTANCE d'une âme soudée par un double
    cordon, a/t_w = β_w·γ_M2·f_y/(√2·f_u·γ_M0) (MSB P5 §2.2), CALCULÉ par la fonction de l'avis a_rec
    (``mecanique.coef_pleine_resistance`` ; la copie de ``platine_about`` lui est tenue égale par
    ``test_assemblages_mecanique``) sur la table du site, la poutre portée et la platine de la même nuance. f_y et f_u
    de la PREMIÈRE bande du tableau 3.3 ANB (``MA.bornes``) : le coefficient ne vaut que pour une âme t_w ≤ t_w max
    (16 mm) ; au-delà, f_y baisse et l'avis du moteur, qui lit l'âme à SON épaisseur (``t_ame``), le dit. Aucun
    coefficient recopié : le guide en tire 0,46 / 0,48 / 0,55 avec les f_u de l'EN 1993-1-1 (tableau 3.1)."""
    t_max = min(MA.bornes(n)[0][0] for n in nuances)
    out = []
    for n in nuances:
        a = table_du_site(n, t_max)
        out.append((n, MEC.coef_pleine_resistance(a["fy"], a["fu"], a["bw"], gM2, gM0)))
    return t_max, out


def texte_pleine_resistance(gM2, gM0=1.0):
    """« 0,46 / 0,50 / 0,60·tw en S235 / S275 / S355 (tw ≤ 16 mm, tableau 3.3 ANB) » : la gorge de pleine résistance
    que l'aide du champ « Gorge a » des platines d'about recommande (écran du site, carte, rubrique Méthode), avec les
    coefficients de ``coefs_pleine_resistance`` écrits comme l'avis a_rec (deux décimales) et la condition d'épaisseur.
    ``gM2``, ``gM0`` : ceux d'une note neuve (``entrees.defaults``)."""
    t_max, ks = coefs_pleine_resistance(gM2, gM0)
    tableaux = sorted({MA.ligne(n)["tableau"] for n, _ in ks})
    return (f"{' / '.join(F(k, 2) for _, k in ks)}·tw en {' / '.join(n for n, _ in ks)} "
            f"(tw ≤ {t_max:g} mm, tableau {', '.join(tableaux)})")

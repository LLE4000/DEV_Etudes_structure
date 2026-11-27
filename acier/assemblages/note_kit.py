# -*- coding: utf-8 -*-
"""Le kit des notes PDF des assemblages : le bloc HYPOTHÈSES empaqueté, et la note SANS les taux.

Règle (charte du 31/10/2026, §11) : au plus ``n_max`` lignes de ``larg``
caractères, composées de SEGMENTS écrits par le module — jamais coupés ;
quand ils ne tiennent pas, les formes courtes sont prises du dernier
segment au premier ; s'ils ne tiennent toujours pas, ``ValueError`` (le
test rougit : c'est le texte qu'il faut revoir, pas la règle).

La note SANS les taux (demande du propriétaire, 25/11/2026 : « afficher les pourcentages de résultats dans la note de
calcul ou pas. […] Si je dimensionne tout, je n'ai pas envie qu'on voie en un clin d'œil que j'ai dimensionné tout à
60 %. […] Soit on garde la note de calcul actuelle, soit les pourcentages sont masqués ») : ``infos["sans_taux"]``,
recopié dans ``doc_meta``. La note garde alors tout ce qui PROUVE — Ed, Rd, le verdict de chaque vérification, celui de
l'assemblage, la vérification dimensionnante — et n'écrit plus aucun pourcentage : ni « η max », ni les taux par
élément du bandeau, ni la colonne η (elle dit « OK » ou « NON OK »), ni le « = 80,3 % » qui finit une formule
(``masquer_taux``). Sans l'option, pas un octet ne change.
"""

import itertools
import re

from acier.formats import F

_MINUSCULES = "abcdefghijklmnopqrstuvwxyzàâäéèêëîïôöùûüç"

# Les renvois aux guides : un § SANS indication est de l'EN 1993-1-8 (légende du pied de la note :
# « EC3 = EN 1993-1-8 sauf indication ») ; un § du guide MSB Part 5 porte donc TOUJOURS « MSB P5 »
# (relecture du 01/11/2026 : « (§4.2.4) » nu, à côté de « (§4.5.1) » de l'EN, se lisait comme l'EN).
MSB = "MSB P5 "

# ------------------------------------------------------------------ la note SANS les taux
SANS_TAUX = "sans_taux"                     # la clé de ``infos`` (l'écran du site) et de ``doc_meta`` (les notes)


def sans_taux(doc):
    """La note s'écrit-elle SANS les taux ? (``doc`` : le ``doc_meta`` d'une note, ou les ``infos`` qu'elle reçoit.)"""
    return bool((doc or {}).get(SANS_TAUX))


def verdict(ok):
    """Ce que la colonne η d'un tableau dit à la place d'un taux."""
    return "OK" if ok else "NON OK"


# « η = 66,70 kN / 83,07 kN = 80,3 % ≤ 100 % » : la ligne qui finit une vérification, dans les six moteurs
_ETA_COMPAREE = re.compile(r"η = (?P<ed>[^=;]+?) / (?P<rd>[^=;]+?) = (?:>\s*)?[-−]?\d+(?:[.,]\d+)?\s*%\s*(?P<op>≤|>)\s*100\s*%")
# « … = 27,0 % », « → 85 % », « (85 %) » : un taux en fin d'égalité ou entre parenthèses (les lignes substituées des doubles
# cornières, les tableaux « calcul à la main » du pied de poteau)
_TAUX_FINAL = re.compile(r"\s*(?:=|→)\s*(?:>\s*)?[-−]?\d+(?:[.,]\d+)?\s*%|\s*\((?:>\s*)?[-−]?\d+(?:[.,]\d+)?\s*%\)")


# « béton d'ancrage à 107,7 %, hors périmètre » : un taux dit dans une phrase (le motif « à vérifier » du pied de poteau)
_TAUX_DIT = re.compile(r"\s+à\s+(?:>\s*)?[-−]?\d+(?:[.,]\d+)?\s*%")


def masquer_taux(texte):
    """Un texte de la note SANS ses taux : « η = 66,70 kN / 83,07 kN = 80,3 % ≤ 100 % » devient « 66,70 kN ≤ 83,07 kN »
    (la comparaison reste, le pourcentage part) ; ailleurs, le « = 27,0 % », le « → 85 % », le « (85 %) » ou le
    « à 107,7 % » est retiré. Un SEUIL (« au-delà de 100 % ») n'est pas un taux : il reste."""
    t = _ETA_COMPAREE.sub(lambda m: m.group("ed").strip() + " " + m.group("op") + " " + m.group("rd").strip(), texte)
    return _TAUX_DIT.sub("", _TAUX_FINAL.sub("", t))


def coefficients(tete, g_M2, g_M2n, tete_net=None):
    """Les formes du segment des coefficients partiels (trois modules
    poutre-poutre), de la plus longue à la plus courte : ``tete``
    (« γM0 1,00 · γM2 1,25 ») suivie de ce que couvre γM2 ; quand γM2,net
    diffère, il est écrit à part, et ``tete_net`` remplace ``tete`` dans la
    forme courte (platine d'about : sans « (ANB) »)."""
    if abs(g_M2 - g_M2n) > 1e-9:
        net = " · γM2,net " + F(g_M2n, 2)
        return (tete + " (boulons, soudures)" + net + " (nettes)", (tete_net or tete) + net)
    return (tete + " (boulons, soudures, sections nettes)", tete + " (boulons, soudures, nettes)", tete)


def _capitale(s):
    """Capitale initiale d'une ligne — lettre latine seulement (« γM0 »
    reste tel quel)."""
    return s[0].upper() + s[1:] if s and s[0] in _MINUSCULES else s


def _emballer(textes, larg):
    """Glouton : un segment rejoint la ligne courante (« ; ») s'il tient
    avec le point final, sinon il ouvre la suivante. None si un segment seul
    ne tient pas."""
    lignes, cur = [], ""
    for t in textes:
        if len(t) + 1 > larg:
            return None
        cand = t if not cur else cur + " ; " + t
        if len(cand) + 1 <= larg:
            cur = cand
        else:
            lignes.append(_capitale(cur) + ".")
            cur = t
    if cur:
        lignes.append(_capitale(cur) + ".")
    return lignes


def empaqueter(segments, n_max=3, larg=110, une_fois=None):
    """``segments`` : chaînes, ou tuples de formes de la plus longue à la plus
    courte. Renvoie les lignes (≤ ``n_max``, chacune ≤ ``larg``).

    Le choix des formes : le PREMIER jeu, dans l'ordre lexicographique des
    indices de forme, dont l'emballage tient — les segments du début gardent
    leur forme longue tant que c'est possible, les formes courtes sont prises
    du dernier au premier. (Avant le 01/11/2026, un compteur sans remise à
    zéro : un segment raccourci laissait les suivants à leur forme la plus
    courte, et quatre cas figés ne trouvaient plus de solution dès que les
    renvois disaient « MSB P5 ».)

    ``une_fois`` (ex. ``MSB`` = « MSB P5 ») : un jeu où ce texte figure au
    plus une fois par ligne passe avant les autres ; à défaut, le premier
    jeu qui tient (le renvoi n'est jamais retiré, il peut alors figurer deux
    fois sur une ligne)."""
    formes = []
    for s in segments:
        fs = [s] if isinstance(s, str) else [f for f in s if f]
        if fs:
            formes.append(fs)
    bornes = []
    for fs in formes:                       # un segment qui ne tient pas seul prend d'abord sa forme qui tient
        i = 0
        while i < len(fs) and len(fs[i]) + 1 > larg:
            i += 1
        if i == len(fs):
            raise ValueError("segment trop long pour " + str(larg) + " caractères : " + fs[-1][:40])
        bornes.append(i)
    premier = None
    for choix in itertools.product(*[range(b, len(fs)) for b, fs in zip(bornes, formes)]):
        lignes = _emballer([fs[i] for fs, i in zip(formes, choix)], larg)
        if lignes is None or len(lignes) > n_max:
            continue
        if not une_fois or all(l.count(une_fois.strip()) <= 1 for l in lignes):
            return lignes
        premier = premier or lignes
    if premier is not None:
        return premier
    raise ValueError("les hypothèses ne tiennent pas en " + str(n_max) + " lignes de " + str(larg))


LIGNES_DE_SECOURS = 6


def empaqueter_au_mieux(segments, n_max=3, larg=110, une_fois=None, n_secours=LIGNES_DE_SECOURS):
    """``empaqueter`` qui ne REFUSE plus la note : ``n_max`` lignes tant que
    les formes courtes y tiennent ; sinon une ligne de plus, jusqu'à
    ``n_secours``. La colonne des dessins s'ajuste au nombre de lignes
    (``colonne_dessins`` de chaque note).

    Avant le lot de clôture du 27/11/2026 (défaut D1 du dossier de contrôle),
    ``empaqueter`` levait « les hypothèses ne tiennent pas en 3 lignes » dès
    que plusieurs cas se cumulaient — grugeage et chaînage, grugeage et
    boulons précontraints, appui dans l'âme… — et l'utilisateur n'obtenait
    pas sa note, alors que le calcul était fait."""
    for n in range(n_max, n_secours + 1):
        try:
            return empaqueter(segments, n_max=n, larg=larg, une_fois=une_fois)
        except ValueError:
            if n >= n_secours:
                raise


def segment_maintien(maintenue):
    """Le segment des HYPOTHÈSES que le calcul SUPPOSE sans le vérifier (lot de clôture du 27/11/2026, L19 ; arbitrage
    A1 : « reporter le maintien réel au déversement dans les hypothèses ») : la poutre grugée maintenue au déversement
    (``maintenue`` : poutre grugée, saisie « Oui »). La stabilité du grugeage (stN, stD) ne vaut que sous cette
    hypothèse ; « Non » rend « À VÉRIFIER ». Placé en DERNIER dans le bloc : c'est lui que le kit raccourcit d'abord,
    les coefficients gardent leur forme longue et la note tient dans les trois lignes de la charte. La seconde poutre
    éventuelle (âme porteuse chargée des deux côtés) n'y est pas : elle ne tiendrait pas dans la charte, la méthode la
    déclare hors périmètre. None s'il n'y a rien à dire."""
    if not maintenue:
        return None
    return ("poutre grugée maintenue au déversement (donnée de la saisie)", "maintenue au déversement (saisie)",
            "maintenue au déversement")


MM = "\u00a0mm"           # l'espace INSÉCABLE colle le nombre à son unité (``ndc_pdf.kit.Doc.wrap`` ne coupe pas dessus)


def phrase_articulation(R):
    """La JUSTIFICATION de l'articulation, écrite au verdict de la note quand le détail éprouvé est tenu (lot de
    clôture du 27/11/2026, arbitrage A1 : « reporter la justification de l'articulation dans le verdict de la note
    révisée ») : le moteur nomme les conditions et leurs bornes — platine d'about : ``rec_tp`` (tp ≤ ``tp_max`` =
    0,36·d·√(fub/fy)), ``rec_a`` (a ≥ ``a_detail``, 0,8 × pleine résistance) ; plat d'âme : ``rec_2d`` (la plus petite
    pince ``pince_2d`` ≥ 2d), ``rec_a`` (a ≥ ``a_plein``), ``rec_boul`` (8.8 non précontraints) —, la note ne compose
    que la phrase. Quand une condition manque, le statut « À VÉRIFIER (articulation à justifier : …) » le dit déjà :
    "" (rien à ajouter) ; "" aussi pour un module qui ne nomme pas ces conditions (doubles cornières). La phrase est
    COURTE à dessein : la conclusion grandit vers le haut, sur la place des blocs de formules FACULTATIFS de la colonne
    (mesuré sur 192 notes, exemples du dossier et grille d'options : 43 la portent, aucune ne change de disposition ni
    ne déborde ; sur cinq notes du plat sur poteau aux options cumulées, elle prend la place du dernier bloc facultatif —
    jamais d'une formule obligatoire)."""
    if R.get("rec_tp") is not None and R.get("tp_max") is not None:                       # platine d'about
        if not (R.rec_tp and R.rec_a):
            return ""
        return ("articulation : détail éprouvé tenu (tp = " + F(R.t_p, 0) + " ≤ " + F(R.tp_max, 1) + MM + " ; a = "
                + F(R.a_w, 0) + " ≥ " + F(R.a_detail, 1) + MM + ")")
    if R.get("rec_2d") is not None and R.get("pince_2d") is not None:                     # plat d'âme
        if not (R.rec_2d and R.rec_a and R.get("rec_boul")):
            return ""
        from acier.assemblages.sci import N_2D             # (import local : sci ne dépend pas du kit des notes)
        return ("articulation : détail éprouvé tenu (" + R.pince_2d[0] + " = " + F(R.pince_2d[1], 0) + " ≥ 2d = "
                + F(N_2D * R.d_b, 0) + MM + " ; a = " + F(R.a_w, 0) + " ≥ " + F(R.a_plein, 1) + MM + " ; 8.8 non précontraints)")
    return ""

# -*- coding: utf-8 -*-
"""Les PROPOSITIONS du prédimensionnement, VÉRIFIÉES par le calcul complet (25/11/2026).

Demande du propriétaire : « Les propositions, ça doit nous proposer des propositions qui sont correctes. La seule raison
pour laquelle ça n'irait pas, ce serait que les profilés seraient pas assez suffisants. Et qu'on devrait changer de
profilé si on a 0 solution. […] Ça pourrait épaissir ma platine, ça pourrait rajouter des boulons. »

Le prédimensionnement des six moteurs (``moteur.predim``) ESTIME : deux ou trois taux par combinaison (boulons, pression
diamétrale des âmes), une seule file de boulons, des pinces « de détail ». Sa proposition pouvait donc être dite
« complète » et l'assemblage rester non vérifié — ou ne rien proposer quand la hauteur manquait d'un millimètre. Ce
fichier ne remplace AUCUNE formule : il CHERCHE des dispositions, et laisse le calcul complet (``moteur.compute``) JUGER
chacune. Une proposition est donc, par construction, un assemblage que le moteur déclare vérifié.

La recherche (``chercher``) :

1. une GRAINE par boulon et par nombre de rangées : la combinaison du prédimensionnement du moteur, sur TOUT le
   catalogue des boulons, avec ses pinces de détail (k·d0 des règles) quand elles tiennent dans la hauteur disponible,
   sinon RESSERRÉES juste ce qu'il faut (jamais sous les minima du Tableau 3.3) ;
2. chaque graine est recopiée par ``moteur.apply_solution`` (boulons, rangées, pièce d'attache, gorge — jamais un
   effort, un profilé, un acier, un niveau ni le grugeage), sa géométrie posée (``_ecrire_*``), puis CALCULÉE ;
3. si elle ne passe pas, elle est RETOUCHÉE d'après ce que le calcul lui reproche, puis recalculée : la gorge portée à
   la pleine résistance, la pièce épaissie (dans la limite de ductilité ; une cornière du CATALOGUE, jamais une
   épaisseur qui n'existe pas), l'entraxe des rangées étiré dans la hauteur disponible, une DEUXIÈME FILE de boulons ;
4. par diamètre, on garde la disposition qui a le MOINS de rangées et tient le taux cible — sur UNE file d'abord
   (rajouter une rangée avant de doubler les files), sur deux files ensuite ; à défaut, la moins chargée de celles que le
   moteur vérifie. Les diamètres de la plage choisie d'abord ; les autres seulement si la plage ne donne rien ;
5. une disposition retenue que le moteur vérifie « sous réserve » d'un avis qu'elle peut LEVER elle-même l'est : un plat
   d'âme plus court que la hauteur recommandée par les guides (l'avis « hp_rec » du moteur) est allongé — le plus petit
   entraxe qui fait tomber l'avis, dans la hauteur disponible et sous l'entraxe maximal que le moteur nomme ; et s'il
   n'y suffit pas, une ou deux rangées de plus sont essayées. C'est le moteur qui dit quand l'avis tombe.

Sans aucune solution, ``blocage`` nomme la vérification qui reste au-delà de 100 % dans le meilleur candidat : toutes
les dispositions ayant été essayées, ce sont les PROFILÉS (ou le grugeage, la position de l'attache) qui ne suffisent
pas. Et quand AUCUN candidat n'a même une géométrie valide (une semelle qui heurte la poutre porteuse, un grugeage trop
court : des données que la recherche ne touche pas), ``geometrie`` rend le calcul du candidat qui a le moins d'alertes
bloquantes : c'est elles qu'il faut lever d'abord.

Trois genres d'attache, parce que leurs saisies diffèrent (``GENRES``) : le plat d'âme (une ou deux files, un plat), les
doubles cornières (deux groupes de boulons, une cornière du catalogue) et la platine d'about (deux files, une platine).

Référence : aucune règle de calcul ici — EN 1993-1-8 Tableau 3.3 pour les pinces minimales et Tableau 3.4 pour
l'entraxe qui sature α_b (p1 = 3,75·d0), que le moteur vérifie de toute façon.
Origine de la copie : AUCUNE — composant NEUF.
"""
import math

from acier.assemblages import mecanique as MEC
from acier.bibliotheques import DB, ORI_P, ORI_S
from acier.js import AttrDict, INF, N, ceil5, mx

# le nombre de calculs complets d'une recherche, au plus (un calcul : 1 à 3 ms hors navigateur, cinq à dix fois plus
# dedans) ; au-delà la recherche s'arrête et le dit (``complet`` faux)
BUDGET = 420
# les retouches d'un candidat, au plus (gorge, quatre épaisseurs, entraxe, deux files par groupe)
RETOUCHES = 9
# les crans d'une réserve qu'on lève, au plus (un plat allongé de 5 mm d'entraxe par cran) ; et les rangées de plus
# qu'on essaie quand les crans n'y suffisent pas
LEVEES = 24
RANGEES_DE_PLUS = 2
# les pinces MINIMALES du Tableau 3.3 (e ≥ 1,2·d0 ; p1 ≥ 2,2·d0) : le plancher d'une disposition resserrée
K_MINI = {"k_e1": 1.2, "k_p1": 2.2, "k_e2": 1.2}
# l'entraxe p1 au-delà duquel α_b ne gagne plus rien (Tableau 3.4 : p1/(3·d0) − 1/4 = 1) : le plafond d'un entraxe étiré
K_P1_SATURE = 3.75
# l'entraxe horizontal p2 d'une deuxième file : k_p1·d0 des règles, jamais sous 2,4·d0 (Tableau 3.3)
K_P2_MINI = 2.4
MODE_VERIF = "VÉRIFICATION"


def _suivante(valeurs, x, maxi=INF):
    """La plus petite valeur de ``valeurs`` strictement au-dessus de ``x`` et ≤ ``maxi`` ; None s'il n'y en a pas."""
    for v in sorted(valeurs):
        if x + 1e-9 < v <= maxi + 1e-9:
            return v
    return None


def _a_alerte(R, ident):
    return any(a.id == ident for a in R.alerts)


def _groupes(S):
    """(pièce d'attache, poutre portée, pièce porteuse, soudures) : les groupes de la synthèse du moteur, dans l'ordre
    de ``synthese.ELEMENTS`` (après « Boulons »)."""
    return tuple(S.ELEMENTS[i][0] for i in (1, 2, 3, 4))


def _p2(u, r):
    return ceil5(mx(K_P2_MINI, N(u.get("k_p1"))) * r.b["d0"])


def _detail(u, b):
    """(e1, p1, e2) de détail d'un boulon : k·d0 des règles, au multiple de 5 supérieur (la règle des six moteurs)."""
    d0 = b["d0"]
    return ceil5(N(u.get("k_e1")) * d0), ceil5(N(u.get("k_p1")) * d0), ceil5(N(u.get("k_e2")) * d0)


def _hauteur(e1, p1, n):
    return 2 * e1 + (n - 1) * p1


def _pression_diametrale(R):
    """Le calcul reproche-t-il une pression diamétrale (ou les boulons) ? C'est ce qu'un entraxe plus grand soulage."""
    g = R.get("gov")
    return g is not None and (g.key.startswith("pd") or g.grp == "Boulons")


# ============================================================================================ plat d'âme
def _ecrire_plat(M, base, r, s):
    v = M.apply_solution(base, r)
    p2 = _p2(base, r)
    v.update(e1_u=s.e1, p1_u=s.p1, e2b_u=s.e2, hp_u=_hauteur(s.e1, s.p1, r.n), n2_u=s.fS,
             bp_u=ceil5(N(base.get("g_h")) + 2 * s.e2 + (p2 if s.fS == 2 else 0)))
    if s.fS == 2:
        v["p2_u"] = p2
    if s.t is not None:
        v["tp_u"] = s.t
    if s.a is not None:
        v["a_w"] = s.a
    return v


def _retoucher_plat(S, v, R, r, s, deux):
    """Ce que le calcul reproche au plat d'âme, corrigé d'un cran dans l'état ``s`` : le mot de la retouche, ou None
    (``deux`` : la deuxième file est permise)."""
    piece, portee, _porteuse, soudure = _groupes(S)
    grp = R.gov.grp if R.get("gov") is not None else ""
    a = MEC.gorge_proposee(R.a_plein)
    if (grp == soudure or _a_alerte(R, "a_rec")) and a > N(v["a_w"]):
        s.a = a
        return "gorge"
    if grp == piece:
        t = _suivante(DB["epais"], N(v["tp_u"]), r.prop.tmax)
        if t is not None:
            s.t, s.a = t, None                    # la gorge de pleine résistance suit l'épaisseur (retouche suivante)
            return "épaisseur"
    if _etirer(R, r, s):
        return "entraxe"
    if deux and grp in ("Boulons", portee, piece) and s.fS == 1:
        s.fS = 2
        return "deux files"
    return None


# la RÉSERVE que le plat d'âme sait lever lui-même : l'avis « hp_rec » du moteur — un plat plus court que la hauteur
# recommandée par les guides pour maintenir la poutre en torsion. Le SEUIL n'est pas écrit ici : c'est le moteur qui
# émet l'avis, et qui cesse de l'émettre.
RESERVE_PLAT = "hp_rec"


def _plafond_p1(R):
    """L'entraxe p1 MAXIMAL que le moteur admet : la ligne de son Tableau 3.3 qui porte un maximum et dont la valeur est
    l'entraxe des rangées (« Entraxe p1 », max = min(14·t ; 200 mm)) ; sans elle, pas de plafond."""
    for x in (R.get("dist") or []):
        if x.max is not None and abs(x.val - R.p_1) < 1e-9:
            return x.max
    return INF


def _lever_plat(r, s):
    """Un cran vers la levée de ``RESERVE_PLAT`` : 5 mm d'entraxe de plus, tant que les rangées tiennent dans la hauteur
    disponible et que l'entraxe reste sous son plafond (``s.plafond``). Vrai si le cran est pris."""
    if r.n < 2 or s.p1 + 5 > s.plafond + 1e-9 or _hauteur(s.e1, s.p1 + 5, r.n) > s.limite + 1e-9:
        return False
    s.p1 += 5
    return True


# ============================================================================================ platine d'about
def _ecrire_platine(M, base, r, s):
    v = M.apply_solution(base, r)
    v.update(e1_u=s.e1, p1_u=s.p1, hp_u=mx(r.hp, _hauteur(s.e1, s.p1, r.n)), bp_u=r.p3 + 2 * s.e2)
    if s.t is not None:
        v["tp_u"] = s.t
    if s.a is not None:
        v["a_w"] = s.a
    return v


def _retoucher_platine(S, v, R, r, s, deux):
    """Ce que le calcul reproche à la platine d'about, corrigé d'un cran : le mot de la retouche, ou None (une platine a
    toujours deux files : ``deux`` est sans objet)."""
    piece, _portee, _porteuse, soudure = _groupes(S)
    grp = R.gov.grp if R.get("gov") is not None else ""
    a = MEC.gorge_proposee(R.a_plein)
    if (grp == soudure or _a_alerte(R, "a_rec")) and a > N(v["a_w"]):
        s.a = a
        return "gorge"
    if grp == piece:
        t = _suivante(DB["epais"], N(v["tp_u"]), R.get("tp_max", INF))
        if t is not None:
            s.t = t
            return "épaisseur"
    if _etirer(R, r, s):
        return "entraxe"
    return None


# ============================================================================================ doubles cornières
# les vérifications du CÔTÉ PORTEUR (ailes A, groupe P) : la deuxième file se pose sur le groupe que le calcul nomme
_COTE_P = ("bv_P", "bt_P", "bi_P", "gl_P", "pdA", "cgA", "cnA", "cbA", "flA", "tsA", "pdP", "vlP", "vnP", "tfP", "twtP",
           "ppP", "wP", "waP", "wlP", "wtP")


def _boulonne(v, X):
    return str(v.get("fix_" + X, "")).lower().startswith("boulon")


def corniere_du_catalogue(v, r, s):
    """La cornière la plus LÉGÈRE du catalogue qui porte la disposition ``s`` : épaisseur ≥ celle que le
    prédimensionnement du moteur estime (ou celle de la retouche), aile A ≥ trusquinage + pince (+ p2 pour deux files),
    aile B ≥ jeu + deux pinces (+ p2). Une cornière inégale se pose dans le sens qui convient. Rend {corn_u, orient,
    gA_u} — le trusquinage dégage le congé de LA cornière retenue, jamais sous celui que le moteur a posé (un poteau y
    met son gauge minimal) — ou None si aucune cornière du catalogue ne convient. Jamais une cornière « personnalisée » :
    une épaisseur qui n'existe pas n'est pas une proposition."""
    dw, p2 = r.b["dw"], _p2(v, r)
    t_min = s.t if s.t is not None else r.prop.tC
    besoin_B = (N(v.get("g_h")) + 2 * s.e2 + (p2 if s.fS == 2 else 0)) if _boulonne(v, "S") else 0
    mieux = None
    for c in DB["cornieres"]:
        if c["t"] < t_min - 1e-9:
            continue
        gA = mx(r.prop.gA, ceil5(c["t"] + c["r"] + dw / 2 + 2))
        besoin_A = (gA + s.e2 + (p2 if s.fP == 2 else 0)) if _boulonne(v, "P") else 0
        for orient, aile_A, aile_B in ((ORI_P, c["a1"], c["a2"]), (ORI_S, c["a2"], c["a1"])):
            if aile_A + 1e-9 < besoin_A or aile_B + 1e-9 < besoin_B:
                continue
            cle = ((c["a1"] + c["a2"] - c["t"]) * c["t"], c["t"])      # l'aire, puis l'épaisseur
            if mieux is None or cle < mieux[0]:
                mieux = (cle, dict(corn_u=c["n"], orient=orient, gA_u=gA))
            break                                  # (ailes égales : un seul sens)
    return mieux[1] if mieux else None


def _ecrire_cornieres(M, base, r, s):
    v = M.apply_solution(base, r)
    c = corniere_du_catalogue(v, r, s)
    if c is None:
        return None
    p2 = _p2(base, r)
    v.update(c, e1S_u=s.e1, p1S_u=s.p1, e2b_u=s.e2, e1P_u=s.e1, p1P_u=s.p1, LC_u=_hauteur(s.e1, s.p1, r.n),
             n2S_u=s.fS, n2P_u=s.fP)
    for k in ("k1_u", "k2_u", "kt_u", "kr_u"):     # (les dimensions d'une cornière personnalisée : sans objet)
        v[k] = base.get(k)
    if s.fS == 2:
        v["p2_S"] = p2
    if s.fP == 2:
        v["p2_P"] = p2
    for X, a in s.gorges.items():
        v["a_" + X] = a
    return v


def _retoucher_cornieres(S, v, R, r, s, deux):
    """Ce que le calcul reproche aux doubles cornières, corrigé d'un cran : le mot de la retouche, ou None (``deux`` : la
    deuxième file est permise, sur le groupe que le calcul nomme)."""
    piece, portee, porteuse, soudure = _groupes(S)
    g = R.get("gov")
    grp, cle = (g.grp, g.key) if g is not None else ("", "")
    cote = "P" if cle in _COTE_P else "S"
    if grp == soudure and not _boulonne(v, cote):
        a = N(v.get("a_" + cote)) + 1              # la gorge du cordon en cause, un millimètre de plus, sous 0,7·tc
        if a <= 0.7 * R.t_C + 1e-9:
            s.gorges[cote] = a
            return "gorge"
    if grp == piece:
        t = _suivante(sorted({c["t"] for c in DB["cornieres"]}), R.t_C)
        if t is not None:
            s.t = t
            return "épaisseur"
    if _etirer(R, r, s):
        return "entraxe"
    if deux and grp in ("Boulons", portee, porteuse, piece) and _boulonne(v, cote):
        if cote == "S" and s.fS == 1:
            s.fS = 2
            return "deux files"
        if cote == "P" and s.fP == 1:
            s.fP = 2
            return "deux files"
    return None


def _etirer(R, r, s):
    """L'entraxe des rangées étiré dans la hauteur de la graine, jusqu'à saturer α_b (3,75·d0) — quand le calcul
    reproche une pression diamétrale ou les boulons, et une seule fois."""
    if s.etire or r.n < 2 or not _pression_diametrale(R):
        return False
    s.etire = True
    libre = s.limite - _hauteur(s.e1, s.p1, r.n)
    p1 = min(ceil5(K_P1_SATURE * r.b["d0"]), s.p1 + 5 * math.floor(libre / (5 * (r.n - 1)) + 1e-9))
    if p1 <= s.p1:
        return False
    s.p1 = p1
    return True


def _signature_cornieres(v):
    """Ce qui distingue deux propositions de cornières pour le MOTEUR quand aucune aile n'est boulonnée — le diamètre et
    les rangées des boulons n'y jouent pas : la cornière, son sens, sa longueur et ses gorges. None dès qu'une aile est
    boulonnée (chaque diamètre est alors une proposition à part entière)."""
    if _boulonne(v, "S") or _boulonne(v, "P"):
        return None
    return (v.get("corn_u"), v.get("orient"), N(v.get("LC_u")), N(v.get("a_S")), N(v.get("a_P")))


def _boulons_plat(v):
    return int(N(v["n1_u"])) * int(N(v["n2_u"]))


def _boulons_platine(v):
    return 2 * int(N(v["n1_u"]))


def _boulons_cornieres(v):
    return ((int(N(v["n1S_u"])) * int(N(v["n2S_u"])) if _boulonne(v, "S") else 0)
            + (2 * int(N(v["n1P_u"])) * int(N(v["n2P_u"])) if _boulonne(v, "P") else 0))


# ``limite`` : la hauteur que les rangées peuvent occuper — la hauteur disponible de l'âme portée (plat, cornières) ;
# pour une platine, celle que le moteur lui a donnée (hp ≥ 0,6·h : les rangées s'y écartent sans l'agrandir)
# ``files`` : le genre admet-il une deuxième file (la platine en a toujours deux)
# ``reserve`` / ``lever`` / ``plafond`` : l'avis non bloquant que le genre sait lever lui-même, le cran qui y mène et le
# plafond de ce cran lu dans le calcul (le plat trop court : 5 mm d'entraxe, sous l'entraxe maximal), ou None
# ``signature`` : ce qui distingue deux propositions pour le moteur quand le diamètre n'y joue pas (cornières sans
# aucun boulon), ou None
GENRES = {
    "plat": AttrDict(ecrire=_ecrire_plat, retoucher=_retoucher_plat, boulons=_boulons_plat, limite=lambda r, hav: hav,
                     files=True, reserve=RESERVE_PLAT, lever=_lever_plat, plafond=_plafond_p1, signature=None),
    "platine": AttrDict(ecrire=_ecrire_platine, retoucher=_retoucher_platine, boulons=_boulons_platine,
                        limite=lambda r, hav: r.hp, files=False, reserve=None, lever=None, plafond=None, signature=None),
    "cornieres": AttrDict(ecrire=_ecrire_cornieres, retoucher=_retoucher_cornieres, boulons=_boulons_cornieres,
                          limite=lambda r, hav: hav, files=True, reserve=None, lever=None, plafond=None,
                          signature=_signature_cornieres),
}


# ============================================================================================ la recherche
def _graines(M, base):
    """Une GRAINE par (boulon, rangées) : la ligne du prédimensionnement du moteur qui tient dans la hauteur — celle
    des pinces de détail d'abord, sinon celle des pinces minimales. {(i, n): (ligne, de_detail)}, et la hauteur
    disponible. Une ligne porte sa pièce d'attache (``prop``)."""
    noms = [b["n"] for b in DB["boulons"]]
    large = dict(base, pd_dmin=noms[0], pd_dmax=noms[-1])
    out, hav = {}, None
    for detail, surcharge in ((True, {}), (False, K_MINI)):
        try:
            pd = M.compute(dict(large, **surcharge)).pd
        except Exception:  # noqa: BLE001 — des saisies que le moteur ne calcule pas : aucune graine de ce jeu
            continue
        hav = pd.hav if hav is None else hav
        for r in pd.rows:
            if r.geom and r.prop and (r.i, r.n) not in out:
                out[(r.i, r.n)] = (r, detail)
    return out, hav


def _etat_initial(G, base, r, de_detail, hav):
    """L'état d'une graine : ses pinces (celles du moteur ; une graine aux pinces minimales reprend la pince e2 de
    détail, qui ne coûte aucune hauteur, puis son entraxe et sa pince e1 remontent vers le détail tant que la hauteur
    le permet), une file, l'épaisseur et la gorge du moteur."""
    s = AttrDict(e1=r.e1, p1=r.p1, e2=r.e2, fS=1, fP=1, t=None, a=None, gorges={}, etire=False, limite=G.limite(r, hav),
                 plafond=INF)
    if not de_detail:
        e1d, p1d, e2d = _detail(base, r.b)
        s.e2 = mx(s.e2, e2d)
        while s.p1 + 5 <= p1d and _hauteur(s.e1, s.p1 + 5, r.n) <= s.limite + 1e-9:
            s.p1 += 5
        while s.e1 + 5 <= e1d and _hauteur(s.e1 + 5, s.p1, r.n) <= s.limite + 1e-9:
            s.e1 += 5
    return s


def _reserves(R, changees):
    """Les avis NON bloquants qui portent sur ce que la proposition règle (un champ qu'elle écrit) : la ductilité, une
    hauteur sous 0,6·h… — pas ceux qui tiennent aux données gardées (un moment, une traction)."""
    return [a for a in R.alerts if not a.block and set(a.fields) & changees]


def chercher(M, S, u, genre, nb_max=6, budget=BUDGET):
    """Les propositions VÉRIFIÉES pour les saisies ``u`` (un dictionnaire, comme pour ``M.compute``).

    ``M`` : le moteur (``compute``, ``apply_solution``) ; ``S`` : sa synthèse (``ELEMENTS``) ; ``genre`` : « plat »,
    « cornieres » ou « platine ».

    Rend ``solutions`` — au plus ``nb_max``, une par diamètre, la plus simple d'abord, chacune {u (saisies complètes), R
    (son calcul), boulon, i (rang du diamètre), n (rangées), boulons (nombre), eta, cible (taux cible tenu), plage
    (diamètre dans la plage choisie), deux_files, reserves (avis non bloquants), retouches (mots)} —, ``blocage`` (sans solution : le meilleur candidat non
    vérifié {u, R, gov} ; None si rien ne tient dans la hauteur ou si aucun candidat n'a une géométrie valide),
    ``geometrie`` (sans solution ni blocage : le calcul du candidat qui a le moins d'alertes bloquantes, ou None),
    ``graines``, ``essais`` (calculs complets), ``complet`` (faux si le budget a arrêté la recherche), ``cible`` et
    ``hav`` (la hauteur disponible)."""
    G = GENRES[genre]
    base = dict(u, mode_calc=MODE_VERIF)
    cible = N(base.get("eta_c")) or 0.9
    noms = [b["n"] for b in DB["boulons"]]
    bornes = sorted(noms.index(x) if x in noms else k for x, k in ((base.get("pd_dmin"), 0), (base.get("pd_dmax"), len(noms) - 1)))
    graines, hav = _graines(M, base)
    essais = [2]                                  # les deux calculs qui ont donné les graines
    faits = {}                                    # les dispositions déjà calculées (la passe à deux files repart des mêmes)
    invalide = [None]                             # le calcul à la géométrie non valide qui a le moins d'alertes bloquantes

    def calculer(v):
        cle = tuple(sorted((k, str(x)) for k, x in v.items()))
        if cle not in faits:
            essais[0] += 1
            try:
                faits[cle] = M.compute(v)
            except Exception:  # noqa: BLE001 — une disposition que le moteur ne calcule pas n'est pas une proposition
                faits[cle] = None
        return faits[cle]

    def aboutir(r, de_detail, deux):
        """La graine écrite, calculée, retouchée tant que le calcul lui reproche quelque chose : (saisies, R, mots) du
        meilleur état rencontré — vérifié et le moins chargé ; à défaut le dernier calculé."""
        s = _etat_initial(G, base, r, de_detail, hav)
        mots, garde, s_garde = [], None, None
        for _ in range(RETOUCHES + 1):
            v = G.ecrire(M, base, r, s)
            R = calculer(v) if v is not None else None
            if R is None:
                break
            if garde is None or not garde[1].verified or (R.verified and R.eta_max < garde[1].eta_max):
                garde, s_garde = (v, R, list(mots)), AttrDict(s, gorges=dict(s.gorges))
            if (R.verified and R.eta_max <= cible + 1e-9) or not (R.geo_ok and R.dist_ok) or essais[0] >= budget:
                break
            mot = G.retoucher(S, v, R, r, s, deux)
            if mot is None:
                break
            if mot not in mots:
                mots.append(mot)
        # la disposition gardée tient le taux cible « sous réserve » d'un avis qu'elle peut lever elle-même (étape 5) : le
        # plus PETIT nombre de crans qui fait tomber l'avis (il tombe au-delà d'un seuil : dichotomie), le moteur la
        # vérifiant toujours au taux cible ; adoptée alors, sinon laissée telle quelle
        if (garde is not None and G.reserve is not None and garde[1].verified and garde[1].eta_max <= cible + 1e-9
                and _a_alerte(garde[1], G.reserve)):
            s_garde.plafond = G.plafond(garde[1])

            def essai(k):
                t = AttrDict(s_garde, gorges=dict(s_garde.gorges))
                for _ in range(k):
                    G.lever(r, t)
                v2 = G.ecrire(M, base, r, t)
                R2 = calculer(v2) if (v2 is not None and essais[0] < budget) else None
                return (v2, R2) if (R2 is not None and R2.verified and R2.eta_max <= cible + 1e-9
                                    and not _a_alerte(R2, G.reserve)) else None

            t, crans = AttrDict(s_garde, gorges=dict(s_garde.gorges)), 0
            while crans < LEVEES and G.lever(r, t):
                crans += 1
            mieux = essai(crans) if crans else None
            bas, haut = 1, crans - 1
            while mieux is not None and bas <= haut:
                k = (bas + haut) // 2
                e = essai(k)
                if e is not None:
                    mieux, haut = e, k - 1
                else:
                    bas = k + 1
            if mieux is not None:
                garde = (mieux[0], mieux[1], garde[2] + ["hauteur"])
        return garde

    def rang(x):
        """Le rang d'une disposition parmi celles d'un diamètre : au taux cible d'abord ; puis le moins de réserves ;
        puis, au taux cible, le moins de boulons — sinon la moins chargée."""
        return (x.cible, -len(x.reserves), -x.boulons if x.cible else -x.eta)

    def passe(dans_plage):
        """{diamètre: sa disposition} et le meilleur refus, pour les diamètres de la plage (ou hors d'elle)."""
        trouve, refus = {}, None
        for i in sorted({k[0] for k in graines if (bornes[0] <= k[0] <= bornes[1]) == dans_plage}):
            for deux in ((False, True) if G.files else (False,)):
                if i in trouve and trouve[i].cible:
                    break                          # ce diamètre a sa disposition au taux cible
                de_plus = 0                        # les rangées essayées au-delà de celle qui tient le taux cible
                for n in sorted(k[1] for k in graines if k[0] == i):
                    if essais[0] >= budget:
                        break
                    r, de_detail = graines[(i, n)]
                    etat = aboutir(r, de_detail, deux)
                    if etat is None:
                        continue
                    v, R, mots = etat
                    if not R.verified:
                        if R.geo_ok and R.dist_ok and math.isfinite(R.eta_max) and (refus is None or R.eta_max < refus.R.eta_max):
                            refus = AttrDict(u=v, R=R, gov=R.get("gov"))
                        elif not (R.geo_ok and R.dist_ok):
                            n_bloc = sum(1 for a in R.alerts if a.block)
                            if invalide[0] is None or n_bloc < invalide[0][0]:
                                invalide[0] = (n_bloc, R)
                        continue
                    changees = {k for k in v if v.get(k) != base.get(k)}
                    sol = AttrDict(u=v, R=R, boulon=r.b["n"], i=i, n=n, boulons=G.boulons(v), eta=R.eta_max,
                                   cible=bool(R.eta_max <= cible + 1e-9), plage=dans_plage, deux_files="deux files" in mots,
                                   reserves=_reserves(R, changees), retouches=mots)
                    if i not in trouve or rang(sol) > rang(trouve[i]):
                        trouve[i] = sol
                    if sol.cible:
                        # le moins de rangées qui tient le taux cible — sauf si la réserve que le genre sait lever y
                        # reste (un plat encore trop court) : une ou deux rangées de plus sont essayées
                        if G.reserve is None or not _a_alerte(R, G.reserve) or de_plus >= RANGEES_DE_PLUS:
                            break
                        de_plus += 1
        return trouve, refus

    trouve, refus = passe(True)
    if not trouve:                                # rien dans la plage de diamètres choisie : le reste du catalogue
        hors, refus_hors = passe(False)
        trouve = hors
        if refus is None or (refus_hors is not None and refus_hors.R.eta_max < refus.R.eta_max):
            refus = refus_hors
    # la plus simple d'abord : au taux cible, sans réserve, sur une file, le moins de boulons, le plus petit diamètre
    sols = sorted(trouve.values(), key=lambda x: (not x.cible, len(x.reserves), x.deux_files, x.boulons, x.i))
    # deux dispositions que RIEN ne distingue pour le moteur n'en font qu'une — des cornières soudées des deux côtés,
    # trouvées par deux diamètres de boulons qui n'existent pas : la mieux classée reste
    if G.signature is not None:
        vues, uniques = set(), []
        for x in sols:
            cle = G.signature(x.u)
            if cle is None or cle not in vues:
                uniques.append(x)
            if cle is not None:
                vues.add(cle)
        sols = uniques
    return AttrDict(solutions=sols[:nb_max], blocage=None if sols else refus,
                    geometrie=invalide[0][1] if (not sols and refus is None and invalide[0] is not None) else None,
                    graines=len(graines), essais=essais[0], complet=bool(essais[0] < budget), cible=cible, hav=hav)

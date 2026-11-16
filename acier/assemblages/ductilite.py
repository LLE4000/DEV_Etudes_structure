# -*- coding: utf-8 -*-
"""Ductilité d'une attache par PLAT D'ÂME (poutre–poutre et poutre–poteau) : la règle, son avis et le plafond du
prédimensionnement — une seule source pour les deux moteurs (``plat_ame`` et ``plat_ame_poteau``).

SCI P358, 5.3, point (3) : la rotation de l'attache vient de l'ovalisation des trous ; elle est assurée si le plat
OU l'âme de la poutre portée a une épaisseur au plus égale à k·d (d : diamètre des boulons) — l'un des deux suffit.
k selon la NUANCE de la pièce : 0,50 en S275, 0,42 en S355. Règle du site (brief v12 § 19.1 point 3, réponse du
propriétaire du 16/11/2026) : 0,50 aussi en S235 ; au-delà du S355, la règle n'est pas couverte, et l'avis le dit.

C'est une RECOMMANDATION, jamais un verdict : l'avis n'est pas bloquant et ne change pas le statut de l'attache.
Il ajoute, quand le plat est d'une nuance S355 (pas un acier « Personnalisé »), la phrase du propriétaire sur l'étude du SCI (New Steel
Construction, mai 2018) : un plat de 10 mm en S355 avec des M20 reste admis si la géométrie standard du P358 est
respectée.

La NUANCE se lit sur sa valeur nominale (``materiau.nominale`` : S355 → 355, jamais la f_y réduite par
l'épaisseur) ; un acier « Personnalisé » n'a pas de nuance : sa f_y saisie en tient lieu, et une f_y entre deux
nuances prend le k de la nuance supérieure (côté sûr — À VÉRIFIER, la règle ne parle que des nuances).
"""
from acier.formats import F, f0, f_ecart
from acier.assemblages import materiau as MT
from acier.bibliotheques import PERSO

# (borne supérieure incluse de la valeur nominale, k) — SCI P358 §5.3 (3) ; S235 : la valeur du S275 (règle du site)
K_NUANCE = ((275.0, 0.50), (355.0, 0.42))
# le plafond du prédimensionnement pour un plat de nuance non couverte (au-delà du S355) : le plus petit k connu
# (À VÉRIFIER : la règle n'est pas publiée pour ces nuances ; l'avis de la vérification le dit)
K_MIN = min(k for _, k in K_NUANCE)
# la phrase du propriétaire (brief v12 § 19.1 point 3), mot pour mot
ETUDE_SCI_2018 = ("un plat de 10 mm en S355 avec des M20 reste admis si la géométrie standard du P358 est respectée "
                  "(étude SCI, New Steel Construction, mai 2018)")
REF = "SCI P358 §5.3 (3)"


def k_nuance(fy_nominale):
    """k de la règle pour une pièce de nuance ``fy_nominale`` (N/mm²) : 0,50 jusqu'au S275, 0,42 jusqu'au S355,
    None au-delà (règle non couverte)."""
    if fy_nominale is None:
        return None
    for borne, k in K_NUANCE:
        if fy_nominale <= borne + 1e-9:
            return k
    return None


def evaluer(tp, tw, d, fyn_plat, fyn_poutre):
    """La règle : {"k_p", "k_w", "ok_p", "ok_w", "ok", "couvert"}. ``ok`` : tp ≤ k_p·d OU tw ≤ k_w·d ;
    ``couvert`` : la nuance d'au moins une des deux pièces est couverte (≤ S355)."""
    k_p, k_w = k_nuance(fyn_plat), k_nuance(fyn_poutre)
    ok_p = bool(k_p is not None and tp <= k_p * d + 1e-9)
    ok_w = bool(k_w is not None and tw <= k_w * d + 1e-9)
    return {"k_p": k_p, "k_w": k_w, "ok_p": ok_p, "ok_w": ok_w, "ok": ok_p or ok_w,
            "couvert": k_p is not None or k_w is not None}


def poser(R, d):
    """Pose la règle sur le calcul ``R`` d'un plat d'âme (plat « L », poutre portée « S ») : ``R.duct`` et
    ``R.rec_tp`` (la ductilité est assurée)."""
    R.duct = evaluer(R.t_p, R.tw_S, d, MT.nominale(R, "L"), MT.nominale(R, "S"))
    R.rec_tp = R.duct["ok"]
    return R.duct


def tp_max(tw, d, fyn_plat, fyn_poutre):
    """L'épaisseur de plat la plus forte que la règle admet (prédimensionnement) : aucune borne (inf) si l'âme de la
    poutre portée suffit (tw ≤ k_w·d) ; sinon k_p·d ; un plat de nuance non couverte prend ``K_MIN``."""
    r = evaluer(0.0, tw, d, fyn_plat, fyn_poutre)
    if r["ok_w"]:
        return float("inf")
    return (r["k_p"] if r["k_p"] is not None else K_MIN) * d


def admet_un_plat(tw, d, fyn_plat, fyn_poutre, t_min):
    """Le boulon de diamètre ``d`` admet-il au moins un plat de la liste du prédimensionnement (``t_min`` : sa plus petite
    épaisseur) sous la règle ? Non pour un M16 en S355 dont l'âme portée dépasse 0,42·16 = 6,7 mm : 6,7 < 8 mm."""
    return tp_max(tw, d, fyn_plat, fyn_poutre) >= t_min - 1e-9


# le prédimensionnement range APRÈS les autres (score augmenté de ce rang, sous celui d'une ligne au-delà de la cible)
# un boulon qui n'admet aucun plat de la liste : la proposition suit la règle tant qu'un autre diamètre la tient
RANG_NON_DUCTILE = 5000


def msg_predim(treq, tp, tmax, d):
    """Le message du prédimensionnement quand la proposition ne tient pas la règle, None sinon : la plus petite épaisseur
    de la liste au-delà de k·d (« 8 mm > 0,42·d = 6,7 mm »), ou l'épaisseur nécessaire au-delà de k·d."""
    if tp > tmax + 1e-9:
        return (f"Plus petite épaisseur proposée {f0(tp)} mm > {F(tmax / d, 2)}·d = {f_ecart(tmax, tp)} mm (ductilité, "
                f"{REF}) : augmenter le diamètre.")
    if treq > tmax + 1e-9:
        return ("Épaisseur nécessaire > " + F(tmax / d, 2) + "·d (ductilité, " + REF + ") : augmenter le nombre de "
                "rangées, le diamètre ou la nuance du plat.")
    return None


def _nuance(R, X):
    """Le nom de la nuance d'une pièce (« S355 ») ; « Personnalisé » : « f_y 300 »."""
    m = (R.get("materiau") or {}).get(X) or {}
    if m.get("nom") == PERSO:
        return "f_y " + f0(m.get("fy"))
    return str(m.get("nom") or "")


def _terme(sym, t, k, d, qui, k_dit=None):
    """« tp = 10 mm > 0,42·d = 8,4 mm (plat S355) » — « tw = 8,5 mm > 8,4 mm (poutre S355) » quand le même k vient
    d'être écrit (``k_dit``) —, ou « poutre S460 N/NL : non couverte au-delà du S355 ». L'épaisseur s'écrit avec la
    décimale qui la départage de k·d (``f_ecart`` : « 8,43 mm > 8,4 mm », jamais « 8,4 mm > 8,4 mm »)."""
    if k is None:
        return f"{qui} : non couverte au-delà du S355"
    borne = f"{f0(k * d)} mm" if k == k_dit else f"{F(k, 2)}·d = {f0(k * d)} mm"
    return f"{sym} = {f_ecart(t, k * d)} mm > {borne} ({qui})"


def _k_ecrits():
    """« 0,50 en S235 et S275, 0,42 en S355 » : les k lus dans ``K_NUANCE`` (l'aide et l'avis « non couverte »)."""
    return f"{F(K_NUANCE[0][1], 2)} en S235 et S275, {F(K_NUANCE[1][1], 2)} en S355"


def _plat_s355(R):
    """Le plat est-il d'une nuance S355 (la phrase de l'étude du SCI ne parle que d'elle) ? Un acier « Personnalisé »
    n'a pas de nuance, même si sa f_y le range avec le S355 pour k."""
    m = (R.get("materiau") or {}).get("L") or {}
    return m.get("nom") != PERSO and MT.nominale(R, "L") == K_NUANCE[-1][0]


def alertes(R, d):
    """Les avis de la règle, au format des listes d'alertes des deux moteurs (id, c, b, m, f, d, e, w) :
    « tp_duct » (ductilité non assurée) ou « duct_nc » (règle non couverte : les deux nuances au-delà du S355). Non
    bloquants : une recommandation."""
    r = R.duct
    if r["ok"]:
        return []
    qui_p, qui_w = "plat " + _nuance(R, "L"), "poutre " + _nuance(R, "S")
    champs = ["tp_u", "boulon_u", "nu_pl", "nu_S"]
    if not r["couvert"]:
        return [dict(id="duct_nc", c=True, b=0,
                     m="Ductilité de l'attache : règle " + REF + " non couverte au-delà du S355 — à justifier "
                       "(recommandation).",
                     f=champs, d=["tp"], e=["plate"],
                     w=qui_p + ", " + qui_w + " : k connu jusqu'au S355 seulement (" + _k_ecrits() + ").")]
    w = _terme("tp", R.t_p, r["k_p"], d, qui_p) + " ; " + _terme("tw", R.tw_S, r["k_w"], d, qui_w, r["k_p"])
    if _plat_s355(R):
        w += " ; " + ETUDE_SCI_2018
    return [dict(id="tp_duct", c=True, b=0,
                 m="Ductilité de l'attache non assurée : le plat et l'âme de la poutre portée dépassent k·d ("
                   + REF + ") — recommandation, pas un verdict.",
                 f=champs, d=["tp"], e=["plate"], w=w + ".")]


def texte_aide():
    """L'aide du champ tp (écran et carte) : la règle en une ligne, k lus dans ``K_NUANCE``."""
    return f"Ductilité ({REF}) : tp ≤ k·d du boulon, ou tw de la poutre portée ≤ k·d ; k = {_k_ecrits()}, non couvert au-delà."

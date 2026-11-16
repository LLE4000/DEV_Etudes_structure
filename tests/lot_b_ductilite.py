# -*- coding: utf-8 -*-
"""Contrôles PARTAGÉS du lot B (brief v12 § 19.1 point 3) pour les deux plats d'âme (poutre–poutre et poutre–poteau) :
la ductilité de l'attache, SCI P358 §5.3 (3).

Règle du site : ductilité assurée si tp ≤ k·d (k selon la nuance du PLAT) OU si tw ≤ k·d (k selon la nuance de la
POUTRE PORTÉE) ; k = 0,50 en S235 et S275, 0,42 en S355 ; au-delà du S355 : règle non couverte, et l'avis le dit. Une
recommandation, jamais un verdict. Le prédimensionnement suit la même règle.

Ce fichier n'est pas une suite (pas de « test_ » devant son nom) : chaque suite d'un moteur l'appelle avec son moteur,
puis rejoue les MÊMES contrôles sous chaque mutation (``MUTATIONS``) et exige qu'ils rougissent.
"""
from acier.bibliotheques import PERSO
from acier.assemblages import ductilite as DU

# la phrase du propriétaire (brief v12 § 19.1 point 3), copie NOMMÉE : le contrôle la cherche mot pour mot
PHRASE_SCI = ("un plat de 10 mm en S355 avec des M20 reste admis si la géométrie standard du P358 est respectée "
              "(étude SCI, New Steel Construction, mai 2018)")
PREDIM = "PRÉDIMENSIONNEMENT"


def _avis(R, ident):
    return next((a for a in R.alerts if a.id == ident), None)


def controles(compute, ame_mince):
    """[(nom, condition, info)] ; ``ame_mince`` : un profilé de poutre portée dont l'âme est sous 0,42·20 = 8,4 mm."""
    out = []
    R = compute({})                                                     # plat 10 S355, M20, âme S355 > 8,4 mm
    a = _avis(R, "tp_duct")
    out.append(("note neuve (plat 10 mm S355, M20, âme portée S355 de plus de 8,4 mm) : k = 0,42 des deux côtés, "
                "ductilité non assurée", R.duct["k_p"] == 0.42 and R.duct["k_w"] == 0.42 and not R.rec_tp,
                str(R.duct)))
    out.append(("… avis « tp_duct » NON bloquant, l'attache reste vérifiée (une recommandation, pas un verdict)",
                a is not None and not a.block and R.verified, f"{a and a.block} / {R.verified}"))
    out.append(("… l'avis chiffre les deux pièces : « tp = 10 mm > 0,42·d = 8,4 mm (plat S355) » et l'âme portée",
                a is not None and "tp = 10 mm > 0,42·d = 8,4 mm (plat S355)" in a.why
                and f"tw = {str(R.tw_S).replace('.', ',')} mm > 8,4 mm (poutre S355)" in a.why, a and a.why))
    out.append(("… et ajoute la phrase de l'étude SCI (plat en S355)", a is not None and PHRASE_SCI in a.why,
                a and a.why))
    R = compute(dict(nu_pl="S275", nu_S="S275"))
    out.append(("S275 / S275, plat 10 mm, M20 : 10 ≤ 0,50·20 → ductilité assurée, aucun avis",
                R.rec_tp and not _avis(R, "tp_duct") and R.duct["k_p"] == 0.50, str(R.duct)))
    R = compute(dict(nu_pl="S235", nu_S="S235"))
    out.append(("S235 : k = 0,50 (la valeur du S275, règle du site)", R.duct["k_p"] == 0.50 and R.rec_tp,
                str(R.duct)))
    R = compute(dict(nu_pl="S275", nu_S="S275", tp_u=12))
    out.append(("S275, plat 12 mm > 10 mm mais âme portée ≤ 10 mm : l'un des deux suffit, aucun avis",
                R.rec_tp and R.duct["ok_w"] and not R.duct["ok_p"] and not _avis(R, "tp_duct"), str(R.duct)))
    R = compute(dict(prof_S=ame_mince, tp_u=12))
    out.append((f"S355, plat 12 mm, poutre {ame_mince} (tw = {R.tw_S:g} ≤ 8,4 mm) : l'âme suffit, aucun avis",
                R.rec_tp and R.duct["ok_w"] and not _avis(R, "tp_duct"), str(R.duct)))
    R = compute(dict(nu_pl="S275", nu_S="S355"))
    a = _avis(R, "tp_duct")
    out.append(("plat S275 de 10 mm (≤ 0,50·d) sur poutre S355 : assuré par le plat",
                R.rec_tp and R.duct["ok_p"] and a is None, str(R.duct)))
    R = compute(dict(nu_pl="S275", nu_S="S355", tp_u=12))
    a = _avis(R, "tp_duct")
    out.append(("plat S275 de 12 mm sur poutre S355 : avis, sans la phrase de l'étude SCI (plat hors S355)",
                a is not None and PHRASE_SCI not in a.why and "0,50·d = 10 mm (plat S275)" in a.why, a and a.why))
    R = compute(dict(nu_pl="S460 N/NL", nu_S="S460 N/NL", tp_u=12))
    a = _avis(R, "duct_nc")
    out.append(("S460 N/NL / S460 N/NL : règle non couverte au-delà du S355 — avis « duct_nc », non bloquant",
                a is not None and not a.block and not _avis(R, "tp_duct") and "S355" in a.msg
                and not R.duct["couvert"], a and (a.msg + " | " + a.why)))
    R = compute(dict(nu_pl="S460 N/NL", tp_u=12))
    a = _avis(R, "tp_duct")
    out.append(("plat S460 N/NL sur poutre S355 trop épaisse : avis, le plat « non couvert au-delà du S355 »",
                a is not None and "plat S460 N/NL : non couverte au-delà du S355" in a.why, a and a.why))
    R = compute(dict(nu_pl=PERSO, fy_u=300, fu_u=430, bw_u=0.85))
    out.append(("plat « Personnalisé » f_y 300 (entre S275 et S355) : k = 0,42, celui de la nuance supérieure "
                "(côté sûr, À VÉRIFIER)", R.duct["k_p"] == 0.42, str(R.duct)))
    # le prédimensionnement suit la même règle : plafond k·d du plat, sauf si l'âme portée suffit (tw ≤ k·d) — le
    # plafond attendu est recalculé ici avec les k ÉCRITS (0,42 / 0,50 : copie nommée de la règle du brief)
    k_ecrit = {"S355": 0.42, "S275": 0.50}
    plafonds = []
    for prof in ("HEA 400", "IPE 500", ame_mince):
        for nu, k in k_ecrit.items():
            Rp = compute(dict(mode_calc=PREDIM, prof_S=prof, nu_pl=nu, nu_S=nu))
            pk = Rp.pd.pick
            d = pk.b["d"]
            att = float("inf") if Rp.tw_S <= k * d + 1e-9 else k * d
            plafonds.append((prof, nu, pk.b["n"], att, pk.prop.tmax, Rp.pd.tp))
    out.append(("prédimensionnement : plafond de l'épaisseur = k·d du plat (0,42 en S355, 0,50 en S275), aucun si "
                "l'âme portée suffit — 6 cas (HEA 400, IPE 500, âme mince ; S355 et S275)",
                all(a == t or abs(a - t) < 1e-9 for _, _, _, a, t, _ in plafonds), str(plafonds)))
    out.append(("… HEA 400 S355 (tw = 11 mm) : plafond 0,42·d effectivement appliqué, épaisseur proposée sous lui",
                any(p == "HEA 400" and nu == "S355" and t != float("inf") and abs(t - 0.42 * int(b[1:])) < 1e-9
                    and tp <= t + 1e-9 for p, nu, b, _, t, tp in plafonds), str(plafonds)))
    return out


# ---------------------------------------------------------------- les mutations : chacune remet un défaut
def _remplacer(nom, valeur):
    def mut(on):
        if on:
            mut.avant = getattr(DU, nom)
            setattr(DU, nom, valeur)
        else:
            setattr(DU, nom, mut.avant)
    return mut


def _plat_seul(on):
    """Le plat seul décide (l'âme de la poutre portée ignorée) : la règle d'avant le lot B."""
    if on:
        _plat_seul.avant = DU.evaluer

        def ev(tp, tw, d, fp, fw):
            r = _plat_seul.avant(tp, tw, d, fp, fw)
            return dict(r, ok_w=False, ok=r["ok_p"])
        DU.evaluer = ev
    else:
        DU.evaluer = _plat_seul.avant


def _bloquant(on):
    """L'avis devient un verdict (bloquant)."""
    if on:
        _bloquant.avant = DU.alertes
        DU.alertes = lambda R, d: [dict(a, b=1) for a in _bloquant.avant(R, d)]
    else:
        DU.alertes = _bloquant.avant


MUTATIONS = [
    ("k = 0,50 pour toutes les nuances (la règle d'avant le lot B)", _remplacer("K_NUANCE", ((1e9, 0.50),))),
    ("le plat seul décide (l'âme portée ignorée)", _plat_seul),
    ("au-delà du S355 pris comme le S355 (règle supposée couverte)", _remplacer("K_NUANCE", ((275.0, 0.50), (1e9, 0.42)))),
    ("phrase de l'étude SCI retirée", _remplacer("ETUDE_SCI_2018", "")),
    ("plafond du prédimensionnement à 0,5·d, quelle que soit la nuance", _remplacer("tp_max", lambda tw, d, a, b: 0.5 * d)),
    ("l'avis de ductilité devient bloquant (un verdict)", _bloquant),
]


def jouer(chk, compute, ame_mince):
    """Joue les contrôles, puis chaque mutation (ses contrôles doivent rougir) ; ``chk`` : la fonction de la suite."""
    for nom, cond, info in controles(compute, ame_mince):
        chk(nom, cond, info)
    for nom, mut in MUTATIONS:
        mut(True)
        try:
            rouges = [n for n, c, _ in controles(compute, ame_mince) if not c]
        except Exception as e:  # noqa: BLE001 — une mutation qui fait planter le calcul a aussi été vue
            rouges = [f"exception {type(e).__name__}"]
        finally:
            mut(False)
        chk(f"mutation « {nom} » : les contrôles de la ductilité rougissent ({len(rouges)})", bool(rouges), str(rouges))

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
    # contre-vérification du bloc : l'épaisseur écrite avec la décimale qui la départage de k·d (jamais « 8,4 mm >
    # 8,4 mm ») ; la phrase de l'étude SCI pour un plat de NUANCE S355 seulement (pas un « Personnalisé » de f_y 300)
    R = compute(dict(tp_u=8.43))
    a = _avis(R, "tp_duct")
    out.append(("plat de 8,43 mm : « tp = 8,43 mm > 0,42·d = 8,4 mm » (et non « 8,4 mm > 8,4 mm »)",
                a is not None and "tp = 8,43 mm > 0,42·d = 8,4 mm (plat S355)" in a.why, a and a.why))
    R = compute(dict(nu_pl=PERSO, fy_u=300, fu_u=430, bw_u=0.85))
    a = _avis(R, "tp_duct")
    out.append(("plat « Personnalisé » f_y 300 : avis sans la phrase de l'étude SCI (pas une nuance S355)",
                a is not None and PHRASE_SCI not in a.why and "(plat f_y 300)" in a.why, a and a.why))
    # l'avis « non couverte » écrit les k LUS dans K_NUANCE (source unique) : un k changé y apparaît
    vrai_k = DU.K_NUANCE
    DU.K_NUANCE = ((275.0, 0.50), (355.0, 0.40))
    try:
        a = _avis(compute(dict(nu_pl="S460 N/NL", nu_S="S460 N/NL", tp_u=12)), "duct_nc")
    finally:
        DU.K_NUANCE = vrai_k
    out.append(("l'avis « non couverte » lit ses k dans K_NUANCE (un k de 0,40 y paraît), jamais une copie écrite",
                a is not None and "0,50 en S235 et S275, 0,40 en S355" in a.why, a and a.why))
    # le prédimensionnement à PETITE charge (V_Ed = 30 kN, plage M16 à M24, S355) : 0,42·16 = 6,7 mm < 8 mm, la plus
    # petite épaisseur de la liste — le M16 n'admet aucun plat : la proposition prend le diamètre suivant, ductile
    Rp = compute(dict(mode_calc=PREDIM, V_Ed=30))
    pk = Rp.pd.pick
    out.append(("prédimensionnement, V_Ed = 30 kN, S355 : le M16 (6,7 mm < 8 mm) cède au diamètre suivant, plat sous "
                "k·d, « Proposition complète »",
                pk.b["d"] > 16 and Rp.pd.tp <= pk.prop.tmax + 1e-9 and Rp.pd.msg == "Proposition complète",
                f"{pk.b['n']} tp {Rp.pd.tp} tmax {pk.prop.tmax} | {Rp.pd.msg}"))
    Rp = compute(dict(mode_calc=PREDIM, V_Ed=30, pd_dmin="M16", pd_dmax="M16"))
    out.append(("… plage réduite au M16 : « Plus petite épaisseur proposée 8 mm > 0,42·d = 6,7 mm (ductilité, …) », "
                "jamais « Proposition complète »",
                Rp.pd.msg.startswith("Plus petite épaisseur proposée 8 mm > 0,42·d = 6,7 mm (ductilité, " + DU.REF + ")"),
                Rp.pd.msg))
    return out


def controles_note(compute, generer_note):
    """[(nom, condition, info)] de la note PDF : l'avis long (plat S355 de 12,5 mm, poutre portée S460 N/NL) garde la
    phrase de l'étude SCI ENTIÈRE (deux lignes), sans « … »."""
    import pymupdf
    R = compute(dict(nu_S="S460 N/NL", tp_u=12.5))
    t = " ".join(pymupdf.open(stream=generer_note(R, {}), filetype="pdf")[0].get_text().split())
    return [("note PDF : l'avis long (poutre S460 N/NL, plat de 12,5 mm) garde la phrase de l'étude SCI entière, sans « … »",
             PHRASE_SCI in t, t[t.find("Ductilit"):][:400])]


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


def _f0_seul(on):
    """L'épaisseur écrite par f0 seul (« 8,4 mm > 8,4 mm »)."""
    if on:
        _f0_seul.avant = DU.f_ecart
        DU.f_ecart = lambda x, ref: DU.f0(x)
    else:
        DU.f_ecart = _f0_seul.avant


MUTATIONS = [
    ("k = 0,50 pour toutes les nuances (la règle d'avant le lot B)", _remplacer("K_NUANCE", ((1e9, 0.50),))),
    ("épaisseur écrite par f0 seul (« 8,4 mm > 8,4 mm »)", _f0_seul),
    ("phrase SCI posée dès que k = 0,42 (un « Personnalisé » de f_y 300 compris)",
     _remplacer("_plat_s355", lambda R: R.duct["k_p"] == 0.42)),
    ("k de l'avis « non couverte » recopiés en dur", _remplacer("_k_ecrits", lambda: "0,50 en S235 et S275, 0,42 en S355")),
    ("prédimensionnement : le M16 non ductile gardé au premier rang", _remplacer("RANG_NON_DUCTILE", 0)),
    ("prédimensionnement : « Proposition complète » malgré un plat au-delà de k·d", _remplacer("msg_predim", lambda *a: None)),
    ("le plat seul décide (l'âme portée ignorée)", _plat_seul),
    ("au-delà du S355 pris comme le S355 (règle supposée couverte)", _remplacer("K_NUANCE", ((275.0, 0.50), (1e9, 0.42)))),
    ("phrase de l'étude SCI retirée", _remplacer("ETUDE_SCI_2018", "")),
    ("plafond du prédimensionnement à 0,5·d, quelle que soit la nuance", _remplacer("tp_max", lambda tw, d, a, b: 0.5 * d)),
    ("l'avis de ductilité devient bloquant (un verdict)", _bloquant),
]


def jouer(chk, compute, ame_mince, generer_note=None):
    """Joue les contrôles, puis chaque mutation (ses contrôles doivent rougir) ; ``chk`` : la fonction de la suite ;
    ``generer_note`` : la note PDF du moteur (son contrôle et sa mutation, joués une fois)."""
    for nom, cond, info in controles(compute, ame_mince):
        chk(nom, cond, info)
    if generer_note is not None:
        for nom, cond, info in controles_note(compute, generer_note):
            chk(nom, cond, info)
        from acier.assemblages.poutre_poutre.plat_ame import note as NP
        vrai = NP._deux_lignes
        NP._deux_lignes = lambda d, txt, police, taille, largeur: (d.fit(txt, police, taille, largeur), "")
        try:
            rouges = [n for n, c, _ in controles_note(compute, generer_note) if not c]
        finally:
            NP._deux_lignes = vrai
        chk(f"mutation « avis coupé par “…” dans la note » : le contrôle de la note rougit ({len(rouges)})", bool(rouges),
            str(rouges))
    for nom, mut in MUTATIONS:
        mut(True)
        try:
            rouges = [n for n, c, _ in controles(compute, ame_mince) if not c]
        except Exception as e:  # noqa: BLE001 — une mutation qui fait planter le calcul a aussi été vue
            rouges = [f"exception {type(e).__name__}"]
        finally:
            mut(False)
        chk(f"mutation « {nom} » : les contrôles de la ductilité rougissent ({len(rouges)})", bool(rouges), str(rouges))

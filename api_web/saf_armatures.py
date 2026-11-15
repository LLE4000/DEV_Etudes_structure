# -*- coding: utf-8 -*-
"""Les ARMATURES du « Générateur SAF — Fondation de grue » (lot 5, cahier des charges du 15/11/2026, § 5.2, § 7, § 8) :
la carte ARMATURES de la colonne de droite et sa fenêtre de calcul, le tableau des efforts SCIA du bloc 3 et son aide
« Dans SCIA ». Les PHRASES (``treillis``, ``ligne_flexion``, ``ligne_hauteur``, ``ligne_tranchant``, ``barres``…) sont
écrites une fois, ici : l'écran et la note les lisent (une seule source).

Ce fichier ne calcule rien : il met en mots le principe d'armatures (``saf.fondation_grue.armatures.calculer``) et le
tableau module / SCIA (``saf.fondation_grue.scia.rapports``)."""
from typing import Dict, List, Optional, Tuple

from saf.fondation_grue import scia as SC

from . import commun as C
from . import etabli_compact as E

N = C.NBSP
ETATS = {"ok": ("ok", "Conforme"), "nok": ("nok", "Non conforme"), "v": ("warn", "À vérifier")}
SENS = (("x", "BX"), ("y", "BY"))
FACES = (("inf", "Bas"), ("sup", "Haut"))


# ------------------------------------------------------------------ les phrases (écran et note)
def un_treillis(t: Dict) -> str:
    """« Ø12 e 15 » (Ø en mm, espacement en cm)."""
    return f"Ø{t['d']}{N}e{N}{C.nb(t['e'] / 10)}"


def treillis_face(el: Dict, face: str) -> str:
    """Le treillis d'une face : « Ø12 e 15 » (le même en x et en y) ou « x Ø12 e 15 · y Ø10 e 15 »."""
    f = el["armatures"][face]
    if f["x"] == f["y"]:
        return un_treillis(f["x"])
    return f"x {un_treillis(f['x'])} · y {un_treillis(f['y'])}"


def treillis(el: Dict) -> str:
    """« bas Ø12 e 15 · haut Ø12 e 15 »."""
    return f"bas {treillis_face(el, 'inf')} · haut {treillis_face(el, 'sup')}"


def barres(r: Dict) -> str:
    """« 4 Ø12 » ; « treillis » sans renfort."""
    return f"{r['n']}{N}Ø{r['d']}" if r["n"] else "treillis"


def reperes(liste: List[Dict]) -> str:
    """« R1–R4 » (une suite) ; « R1, R3 » ; « R2 »."""
    nums = [int(r["rep"][1:]) for r in liste]
    if not nums:
        return ""
    if nums == list(range(nums[0], nums[-1] + 1)) and len(nums) > 2:
        return f"R{nums[0]}–R{nums[-1]}"
    return ", ".join(f"R{n}" for n in nums)


def _pareils(liste: List[Dict]) -> bool:
    return len({(r["n"], r["d"]) for r in liste}) == 1


def ligne_barres(liste: List[Dict]) -> str:
    """« R1–R4 : 4 Ø12 » ; des renforts différents : « R1 4 Ø12, R2 5 Ø12 »."""
    if not liste:
        return "treillis suffisant"
    if _pareils(liste):
        return f"{reperes(liste)}{N}: {barres(liste[0])}"
    return ", ".join(f"{r['rep']} {barres(r)}" for r in liste)


def m_txt(x: float) -> str:
    return f"{C.fr(x, 1)}{N}kN·m/m"


def ligne_flexion(m: float, mrd: float, liste: List[Dict]) -> str:
    """La flexion d'une direction : « m = 186,7 &gt; m_Rd = 97,3 kN·m/m · R1–R4 : 4 Ø12 »."""
    signe = "&gt;" if m > mrd + 1e-9 else "≤"
    return f"m{N}={N}{C.fr(m, 1)} {signe} m<sub>Rd</sub>{N}={N}{m_txt(mrd)} · {ligne_barres(liste)}"


def ligne_acier(r: Dict) -> str:
    """« A_s,néc = 1 448 mm²/m · treillis 754 · renfort 694 mm²/m »."""
    return (f"A<sub>s,néc</sub>{N}={N}{C.fr(r['besoin'], 0)} · treillis {C.fr(r['As_tr'], 0)} · renfort "
            f"{C.fr(r['renfort_pm'], 0)}{N}mm²/m")


def ligne_hauteur(h: Dict) -> str:
    """La hauteur de béton (§ 7.2) : « d_min = 29,0 ≤ d = 43,0 cm » (le moteur Dalles : h_min = d_min + y_G ≤ h)."""
    signe = "≤" if h["etat"] == "ok" else "&gt;"
    return f"d<sub>min</sub>{N}={N}{C.fr(h['hu_min'], 1)} {signe} d{N}={N}{C.fr(h['d'], 1)}{N}cm"


def ligne_tranchant(t: Dict) -> Tuple[str, Optional[float], str, str]:
    """(la comparaison, le taux ou None, le niveau, l'état en mots) de l'effort tranchant d'une bande."""
    if t["etat"] == "non_verifie":
        return "V à d ne se lit pas (zones chargées trop proches)", None, "v", "non vérifié"
    signe = "≤" if t["etat"] == "ok" else "&gt;"
    src = " (SCIA)" if t.get("scia") else ""
    txt = (f"τ{N}={N}{C.fr(t['tau'], 2)} {signe} τ<sub>adm,I</sub>{N}={N}{C.fr(t['tau_adm'], 2)}{N}N/mm² · "
           f"V{N}à{N}d{N}={N}{C.fr(t['V'], 1)}{N}kN{src}")
    return txt, t["taux"], C.niveau(t["taux"]), ""


def groupe_bandes(fer: Dict, ef: Dict, sens: str) -> Dict:
    """Les bandes d'une direction (BX1, BX2 ou BY1, BY2), dites ensemble quand elles sont pareilles (« BX1 = BX2 », § 10.5) :
    {"nom", "bandes", "m_inf", "m_sup" (kN·m/m, le plus fort), "renforts_inf", "renforts_sup", "tranchant" (le plus
    défavorable), "hauteur"}."""
    pref = "BX" if sens == "x" else "BY"
    bds = [b for b in ef["bandes"] if b["nom"].startswith(pref)]
    ech = fer.get("echelle") or {}
    m_inf = [max(b["Mmax"][~b["dans_plot"]].max() if (~b["dans_plot"]).any() else 0.0, 0.0) / b["largeur"] * ech.get(f"inf_{sens}", 1.0) for b in bds]
    m_sup = [max(-b["Mmin"][~b["dans_plot"]].min() if (~b["dans_plot"]).any() else 0.0, 0.0) / b["largeur"] * ech.get(f"sup_{sens}", 1.0) for b in bds]
    tr = [fer["tranchant"][b["nom"]] for b in bds]
    pareil = (len(bds) == 2 and abs(m_inf[0] - m_inf[1]) < 0.05 and abs(m_sup[0] - m_sup[1]) < 0.05
              and all(t["V"] is not None for t in tr) and abs(tr[0]["V"] - tr[1]["V"]) < 0.05)
    nom = f"{bds[0]['nom']} = {bds[1]['nom']}" if pareil else " · ".join(b["nom"] for b in bds)
    noms = {b["nom"] for b in bds}
    ri = [r for r in fer["renforts"] if r["type"] == "bande" and r["bande"] in noms and r["face"] == "inf"]
    rs = [r for r in fer["renforts"] if r["type"] == "bande" and r["bande"] in noms and r["face"] == "sup"]
    lus = [t for t in tr if t["etat"] != "non_verifie"]
    pire_t = max(lus, key=lambda t: t["taux"]) if lus else tr[0]
    cands = [(r["h_min"], r["hu_min"], r["yG"]) for r in ri + rs]
    if cands:
        h_min, hu, yG = max(cands)
        hh = {"h_min": h_min, "hu_min": hu, "yG": yG, "d": fer["hauteur"]["h"] - yG,
              "etat": "ok" if h_min <= fer["hauteur"]["h"] + 1e-9 else "nok"}
        hh["taux"] = hu / max(hh["d"], 1e-9)
    else:
        hh = None
    return {"nom": nom, "bandes": bds, "m_inf": max(m_inf, default=0.0), "m_sup": max(m_sup, default=0.0),
            "renforts_inf": ri, "renforts_sup": rs, "tranchant": pire_t, "hauteur": hh}


def zones_txt(fer: Dict, face: str) -> str:
    """« x 4,08 m² · y 4,40 m² · R9–R24 : 4 Ø12 » ; « aucune »."""
    z = [(s, fer["zones"][f"{face}_{s}"]) for s in ("x", "y")]
    if not any(v["aire"] > 0 for _, v in z):
        return "aucune"
    loc = [r for r in fer["renforts"] if r["type"] == "local" and r["face"] == face]
    parts = [f"{s} {C.fr(v['aire'], 2)}{N}m²" for s, v in z if v["aire"] > 0]
    return " · ".join(parts) + (f" · {ligne_barres(loc)}" if loc else "")


# ------------------------------------------------------------------ la carte ARMATURES
def carte(el: Dict, fer: Dict, ef: Dict, sc: Optional[Dict]) -> str:
    """La carte ARMATURES (§ 5.2) : les treillis choisis ; pour chaque direction, la flexion (le moment par mètre, ce que
    reprend le treillis, les renforts), la hauteur de béton (d et d mini du moteur Dalles), l'effort tranchant ; les zones
    rouges et leurs renforts locaux ; l'écart module / SCIA ; les alertes (« Hauteur de béton insuffisante »…)."""
    if fer.get("hors_domaine"):
        rows = C.ligne_verif("Armatures", C.esc(fer["hors_domaine"]), None, "v", etat_txt="non vérifié")
        return E.bloc(3, "Armatures", "warn", "À vérifier", "", rows, theme=True, calcul="a0")
    res = fer["res"]
    rows = C.ligne_verif("Treillis", C.esc(treillis(el)), None, "c", calcul="a0", etat_txt="choisi")
    for sens, _ in SENS:
        g = groupe_bandes(fer, ef, sens)
        mrd_i, mrd_s = fer["mRd_bandes"][f"inf_{sens}"], fer["mRd_bandes"][f"sup_{sens}"]
        flex = (f"bas {ligne_flexion(g['m_inf'], mrd_i, g['renforts_inf'])}<br>haut {ligne_flexion(g['m_sup'], mrd_s, g['renforts_sup'])}")
        rows += C.ligne_verif(f"Flexion {g['nom']}", flex, None, "c", calcul="a0",
                              etat_txt="renforts" if g["renforts_inf"] or g["renforts_sup"] else "treillis")
        if g["hauteur"] is not None:
            rows += C.ligne_verif(f"Hauteur {g['nom']}", ligne_hauteur(g["hauteur"]), C.pct(g["hauteur"]["taux"]),
                                  C.niveau(g["hauteur"]["taux"]) if g["hauteur"]["etat"] == "ok" else "r", calcul="a0")
        txt, taux, niv, mot = ligne_tranchant(g["tranchant"])
        rows += C.ligne_verif(f"Effort tranchant {g['nom']}", txt, C.pct(taux) if taux is not None else None,
                              niv if taux is None or g["tranchant"]["etat"] == "ok" else "r", calcul="a0", etat_txt=mot)
    h = fer["hauteur"]
    if not any(r["type"] == "bande" for r in fer["renforts"]):
        rows += C.ligne_verif("Hauteur de béton", ligne_hauteur(h), C.pct(h["taux"]), C.niveau(h["taux"]) if h["etat"] == "ok" else "r", calcul="a0")
    for face, mot in FACES:
        z = zones_txt(fer, face)
        rows += C.ligne_verif(f"Zones rouges · {mot.lower()}", z, None, "c", calcul="a0",
                              etat_txt="renforts locaux" if z != "aucune" else "aucune")
    if sc is not None:
        rows += ligne_scia(sc)
    pied = "".join(C.message(C.esc(t), n == "r") for t, n in fer["alertes"])
    cls, mot = ETATS[fer["etat"]]
    return E.bloc(3, "Armatures", cls, mot, "", rows, pied, theme=True, calcul="a0")


def ligne_scia(sc: Dict) -> str:
    """La ligne « Écart module / SCIA » de la carte (§ 8.2) : les rapports r saisis ; une pastille orange hors de 0,85 à
    1,15 ; « à saisir » tant que le tableau est vide (étape 4)."""
    if not sc["saisi"]:
        return C.ligne_verif("Écart module / SCIA", "valeurs SCIA à saisir (bloc 3)", None, "c", etat_txt="à saisir")
    vus = [li for li in sc["lignes"] if li["r"] is not None and li["cle"] in sc["a_taper"]]
    txt = " · ".join(f"{_court(li['cle'], sc['sym'])} r{N}={N}{C.fr(li['r'], 2)}" for li in vus)
    pire = max((li["ecart"] for li in vus), default=0.0)
    alerte = any(li["alerte"] for li in vus)
    return C.ligne_verif("Écart module / SCIA", txt, None, "v" if alerte else "p", etat_txt=f"{C.fr(pire * 100, 1)}{N}%" if alerte else "concordent")


def _court(cle: str, sym: bool) -> str:
    return {"bx_m": "M+ x" if not sym else "M+", "by_m": "M+ y", "bx_v": "V x" if not sym else "V", "by_v": "V y",
            "mx": "m<sub>x</sub>" if not sym else "m", "my": "m<sub>y</sub>"}[cle]


# ------------------------------------------------------------------ la fenêtre de calcul
def fenetre(el: Dict, fer: Dict, ef: Dict) -> str:
    """La fenêtre « a0 » : ce que reprend chaque treillis (moteur Dalles), chaque renfort (repère, face, sens, m_dim,
    A_s,néc, renfort, barres, dimensions), l'effort tranchant de chaque bande."""
    if fer.get("hors_domaine"):
        return ""
    res = fer["res"]
    lignes = []
    for cle in ("inf_x", "inf_y", "sup_x", "sup_y"):
        f, s = cle.split("_")
        lignes.append({"nom": f"Treillis {'bas' if f == 'inf' else 'haut'} {s}", "sous": un_treillis(el["armatures"][f][s]),
                       "formule": f"A<sub>s</sub>{N}={N}{C.fr(res['As'][cle], 0)}{N}mm²/m · d{N}={N}{C.fr(res['d'][cle], 1)}{N}cm",
                       "valeur": m_txt(res["mRd"][cle]), "limite": "m<sub>Rd</sub>", "taux": None, "niveau": "c", "etat_txt": "moteur Dalles"})
    for r in fer["renforts"]:
        lignes.append({"nom": r["rep"], "sous": f"{'bande ' + r['bande'] if r['type'] == 'bande' else 'zone rouge'} · {'bas' if r['face'] == 'inf' else 'haut'} {r['sens']}",
                       "formule": f"{ligne_acier(r)} · {C.fr(r['longueur'], 2)} × {C.fr(r['largeur'], 2)}{N}m",
                       "valeur": m_txt(r["m_dim"]), "limite": barres(r), "taux": None, "niveau": "v" if r["serre"] else "c",
                       "etat_txt": "serré" if r["serre"] else "posé"})
    for nom, t in fer["tranchant"].items():
        txt, taux, niv, mot = ligne_tranchant(t)
        lignes.append({"nom": f"V {nom}", "formule": txt, "valeur": f"{C.fr(t['v'], 1)}{N}kN/m" if t["v"] is not None else "—",
                       "limite": "", "taux": C.pct(taux) if taux is not None else None, "niveau": niv if t["etat"] != "nok" else "r",
                       "etat_txt": mot})
    hyps = [f"Moteur du module Dalles · méthode du bureau, ELS · f<sub>yd</sub>{N}={N}f<sub>yk</sub>/γ<sub>s</sub>{N}={N}{C.fr(res['fyd'], 1)}{N}N/mm²",
            f"Lits : x contre l'enrobage ({C.nb(el['armatures']['enrobage'] * 100)}{N}cm), y posé dessus · jeu de pose {C.nb(res['jeu'])}{N}cm, arrondi {C.nb(res['arrondi'])}{N}cm",
            f"Renfort = A<sub>s</sub>(m<sub>dim</sub>) − A<sub>s</sub> treillis · n = ⌈renfort × largeur ÷ aire d'une barre⌉ · ancrage {C.nb(el['armatures']['ancrage'])}{N}Ø",
            "Pas de torsion, pas d'armatures d'effort tranchant · principe d'armatures, pas un plan d'exécution"]
    return C.fenetre_calcul("a0", "Principe d'armatures", "renforts des bandes, zones rouges, renforts locaux", hyps, lignes, "",
                            colonnes=("Élément", "Calcul", "m", "Barres", "État"))


# ------------------------------------------------------------------ le tableau des efforts SCIA (bloc 3) et son aide
def tableau_scia(el: Dict, sc: Optional[Dict], ef: Optional[Dict], unite_sol) -> str:
    """Le tableau du § 8.1 : une ligne par valeur à recopier, la valeur du module en GRIS dans le champ (on tape la
    valeur SCIA par-dessus ; un champ vide reste vide), la case « x = y » ; σ max de chaque k coché (facultatif)."""
    sym = SC.sym(el)
    mod = SC.module(ef) if ef is not None else {c: None for c, _, _ in SC.LIGNES}
    out = C.case(["scia", "sym"], sym, "x = y")
    libelles = {"bx_m": ("M<sup>+</sup> BX", "M<sup>+</sup>"), "by_m": ("M<sup>+</sup> BY", ""), "bx_v": ("V à d BX", "V à d"),
                "by_v": ("V à d BY", ""), "mx": ("m<sub>x</sub> min", "m min"), "my": ("m<sub>y</sub> min", "")}
    aides = {"bx_m": "Bandes BX : M+ le plus fort (kN·m, sur la largeur de la bande)",
             "by_m": "Bandes BY : M+ le plus fort (kN·m, sur la largeur de la bande)",
             "bx_v": "Bandes BX : V à d le plus fort (kN)", "by_v": "Bandes BY : V à d le plus fort (kN)",
             "mx": "Dalle : m_x le plus négatif de la carte 2D (kN·m/m)", "my": "Dalle : m_y le plus négatif de la carte 2D (kN·m/m)"}
    unites = {"bx_m": "kN·m", "by_m": "kN·m", "bx_v": "kN", "by_v": "kN", "mx": "kN·m/m", "my": "kN·m/m"}
    paires = []
    for cle, _, _ in SC.LIGNES:
        if sym and cle in SC.JUMELLES:
            continue
        lib = libelles[cle][1 if sym else 0]
        gris = C.fr(mod[cle], 1) if mod[cle] is not None else ""
        paires.append((lib, C.num(["scia", cle], el["scia"][cle], unites[cle], dec=1, auto=gris), aides[cle]))
    for i in range(0, len(paires), 2):
        out += C.ligne2(*paires[i:i + 2])
    sym_, div, nd = unite_sol
    ks = [x for x in el["sol"]["k"] if x["scia"]]
    if ks and sc is not None:
        pk = []
        for x, s in zip(ks, sc["sigma"]):
            v = el["scia"]["sigma"].get(SC.cle_k(x["k"]))
            gris = C.fr(s["module"] / div, nd) if s["module"] is not None else ""
            pk.append((f"σ k{N}{C.nb(x['k'])}", C.num(["scia", "sigma", SC.cle_k(x["k"])], None if v is None else round(v / div, nd + 3),
                                                       sym_, dec=nd + 1, auto=gris), "σ max SCIA de ce k (facultatif : comparaison seulement)"))
        for i in range(0, len(pk), 2):
            out += C.ligne2(*pk[i:i + 2])
    return out


def aide_scia(el: Dict, m: Dict, ef: Optional[Dict], fer: Optional[Dict]) -> str:
    """L'aide repliée « Dans SCIA » (§ 8.3), en lignes courtes, avec les valeurs du projet."""
    def pt(p):
        return f"({C.fr(p[0], 2)}{N};{N}{C.fr(p[1], 2)})"
    bandes = " · ".join(f"{b['nom']} {pt(b['debut'])} → {pt(b['fin'])}" for b in m["bandes"])
    lg = C.fr(el["bandes"]["largeur"], 2)
    pos = ""
    if ef is not None and ef["bandes"]:
        b = ef["bandes"][0]
        xs = []
        for s1, s2 in b["charge"]:
            xs += [s1 - ef["d_v"], s2 + ef["d_v"]]
        p0, u = b["p0"], b["u"]
        lus = [x for x in xs if -1e-9 <= x <= b["L"] + 1e-9]
        coord = "x" if abs(u[0]) >= abs(u[1]) else "y"
        pos = f"sur {b['nom']}, à {coord} = " + " ; ".join(C.fr((p0[0] if coord == 'x' else p0[1]) + x * (u[0] if coord == 'x' else u[1]), 2) for x in lus) + f"{N}m"
    mrd = f"{C.fr(min(fer['res']['mRd'].values()), 1)}{N}kN·m/m" if fer and not fer.get("hors_domaine") else "m_Rd du treillis"
    etapes = ["Ouvrir le SAF, lancer le calcul.",
              f"Créer 4 bandes d'intégration de {lg}{N}m : {bandes}. L'option « arêtes internes » met leurs axes dans le modèle.",
              "Lire M+ le plus fort des bandes x et des bandes y.",
              f"Lire V à d : {pos}." if pos else "Lire V à d à d du nu des zones chargées.",
              "Lire m<sub>x</sub> et m<sub>y</sub> les plus négatifs sur la carte 2D (efforts de base).",
              f"Contrôle : carte m<sub>x</sub> (puis m<sub>y</sub>), limite de l'échelle à {mrd} ; hors des bandes, ce qui dépasse = les zones rouges.",
              "Imprimer la note SCIA en PDF, pour la joindre."]
    corps = "".join(f"<li>{t}</li>" for t in etapes)
    return (f'<details class="sf-aide" data-plan-volet="aide-scia"><summary><b>Dans SCIA</b>{C.icon("chevron-down")}</summary>'
            f'<ol>{corps}</ol></details>')

# -*- coding: utf-8 -*-
"""Fondation de grue — la SECTION DE CALCUL de chaque bande (lot 6, 15/11/2026 : « une note de calcul au droit des
bandes, aux appuis »). numpy seul, par ``armatures``.

Le moteur du module Dalles (``beton/dalle/moteur.verifier``, sans le modifier) calcule la bande comme UNE SECTION b × h :
b la largeur de la bande (50 ou 100 cm), h celle du principe (``armatures.h_moteur`` : au cm inférieur), ses moments
M⁺ et M⁻ (kN·m sur la largeur de la bande) et V à d (kN), avec ce que le principe y a posé — le treillis de chaque face
et le renfort de la bande (« n barres » dans la bande, à l'axe du treillis de sa direction : ``armatures``, § 7.3). Ses
formules sont celles du module Dalles, écrites par lui (``verifications`` → ``formules``) : l'écran et la note les
mettent en mots par le module Dalles (``api_web/dalle._formule``), une seule écriture.

La section CONTRÔLE le principe. Le principe dimensionne chaque face SEULE (§ 7.4 : n = ⌈(A_s(m_dim) − A_s du treillis) ×
largeur ÷ aire d'une barre⌉) ; la section applique toute la méthode du moteur Dalles, dont le minimum 0,25·A_s,req de la
face OPPOSÉE de la même direction et le maximum 0,04·b·h. Une face que la section dit non conforme est une alerte rouge
du principe (``alerte``) : le principe ne l'aurait pas vue. Sur la dalle de contrôle (10,70 × 10,70 × 0,50), rien ne
change : la face opposée n'y gouverne nulle part.

Ce que la section prend, face par face :
  - avec un ou des renforts de bande : celui qui GOUVERNE (le plus fort A_s,néc ÷ A_s posée ; le premier à égalité), son
    moment m_dim (le plus fort du tronçon, × l'échelle SCIA) et ses barres ;
  - sans renfort : le plus fort moment de la bande sur cette face (hors des plots coulés, × l'échelle SCIA) — au plus ce que
    reprend le treillis.
Les deux faces sont dans la même section : M⁺ (sous un pied) et M⁻ (entre les pieds) ne sont pas au même endroit de la
bande ; les prendre ensemble est du côté de la sécurité — c'est l'usage d'une section du module Dalles.
h n'est pas toujours un nombre entier de cm : d est lu au cm inférieur (``h_moteur``), A_s,min au cm SUPÉRIEUR comme le
principe (``armatures.as_min``) — la section le dit (``plafond``) et le compte dans son verdict.

    bande(el, fer, ef, bd)  -> la section de calcul d'une bande (ou {"bande", "hors": raison})
    bandes(el, fer, ef)     -> [la section de chaque bande], dans l'ordre des bandes
    alerte(sections)        -> les alertes que la section ajoute au principe (texte, "r")
"""
from typing import Dict, List

from beton import position as POS
from beton.dalle import entree as DE

from . import armatures as AR

CLE_MOMENT = {"inf_x": "Mx_inf", "sup_x": "Mx_sup", "inf_y": "My_inf", "sup_y": "My_sup"}
FACES = ("inf", "sup")
FACE_TXT = {"inf": "inférieure", "sup": "supérieure"}


def pose_pm(r: Dict) -> float:
    """La section POSÉE d'un renfort de bande, au mètre (mm²/m) : le treillis + n barres sur la largeur."""
    return r["As_tr"] + r["n"] * AR.aire_barre(r["d"]) / r["largeur"]


def _gouverne(renforts: List[Dict]) -> Dict:
    """Le renfort qui gouverne une face : le plus fort A_s,néc ÷ A_s posée (le premier à égalité)."""
    return max(renforts, key=lambda r: r["besoin"] / pose_pm(r))


def _moment_bande(fer: Dict, bd: Dict, face: str) -> float:
    """Le plus fort moment de la bande sur une face (kN·m/m, ≥ 0), hors des plots coulés, × l'échelle SCIA — celle du
    principe : en bas celle du NOM de la bande, en haut celle du sens de ses barres (``armatures.principe``)."""
    ech = fer.get("echelle") or {}
    hors = ~bd["dans_plot"]
    if face == "inf":
        e_ = ech.get(f"inf_{AR._sens_bande(bd['nom'])}", 1.0)
        v = bd["Mmax"][hors]
    else:
        e_ = ech.get(f"sup_{AR.axe_bande(bd)}", 1.0)
        v = -bd["Mmin"][hors]
    return max(float(v.max()) if v.size else 0.0, 0.0) / bd["largeur"] * e_


def _couches(el: Dict, faces: Dict) -> Dict:
    """Les quatre faces de la section : le treillis (``armatures._couches``) et, sur la face d'un renfort, ses n barres à
    l'axe du treillis de sa direction (la position RÉELLE du treillis : le calcul y ajoute le même jeu et le même arrondi,
    l'axe de calcul est le sien)."""
    out = AR._couches(el)
    c = el["armatures"]["enrobage"] * 100.0
    for f in faces.values():
        r = f["renfort"]
        if r is None or not r["n"]:
            continue
        base = out[f["cle"]]["couches"][0]
        dist = base["dist"] if base["dist"] is not None else POS.sur_parement(c, base["d"])
        out[f["cle"]]["couches"].append({"type": "n_barres", "d": r["d"], "n": r["n"], "dist": round(dist, 6)})
    return out


def bande(el: Dict, fer: Dict, ef: Dict, bd: Dict) -> Dict:
    """La section de calcul d'une bande ``bd`` (``ef["bandes"]``) par le moteur Dalles — voir l'en-tête. Rend
    {"bande", "sens" (x | y : ses barres), "b" (cm), "h" (cm), "largeur" (m),
     "faces": {"inf" | "sup": {"cle" (la face de la section), "renfort" (celui qui gouverne, ou None), "renforts" (ceux de
              la bande sur cette face), "m" (kN·m/m), "M" (kN·m)}},
     "V" (kN, ou None : V à d ne se lit pas), "scia_V" (bool), "entree" (l'entrée du moteur), "resultat" (sa sortie),
     "section" (la section calculée), "verifications" (hauteur, les deux faces, τ : celles que la note écrit),
     "plafond" (None, ou {"h", "As_min": {cle: mm²}} : A_s,min au cm supérieur ; la vérification de la face dit alors ce
     besoin), "besoin" {cle: mm²}, "taux" {cle: besoin ÷ A_s}, "taux_max" (de toutes les vérifications), "etat" ("ok" |
     "nok"), "nok" (les faces non conformes), "causes" {cle: ``cause``}}
    — ou {"bande", "hors": raison} quand le moteur Dalles ne peut pas la calculer."""
    nom = bd["nom"]
    hd = fer.get("hors_domaine")
    if hd:
        return {"bande": nom, "hors": hd}
    largeur = bd["largeur"]
    b_cm = int(round(largeur * 100))
    if abs(b_cm - largeur * 100) > 1e-6 or not DE.B_MIN <= b_cm <= DE.B_MAX:
        return {"bande": nom, "hors": f"largeur de bande {largeur:g} m hors du moteur Dalles".replace(".", ",")}
    ax = AR.axe_bande(bd)
    faces = {}
    for face in FACES:
        cle = f"{face}_{ax}"
        rs = [r for r in fer["renforts"] if r["type"] == "bande" and r["bande"] == nom and r["face"] == face]
        if rs:
            r = _gouverne(rs)
            if r["n"] > DE.N_MAX:
                return {"bande": nom, "hors": f"{r['n']} barres dans la bande : plus que le moteur Dalles n'en compte ({DE.N_MAX})"}
            cs, m = r["cle_section"], r["m_dim"]
        else:
            r = None
            cs, m = AR._cle_section(el, cle, fer["tourne"], fer["res"]["mRd"]), _moment_bande(fer, bd, face)
        faces[face] = {"cle": cs, "renfort": r, "renforts": rs, "m": m, "M": m * largeur}
    t = fer["tranchant"][nom]
    V = t["V"]
    s = {"nom": nom, "Mx_inf": 0.0, "Mx_sup": 0.0, "My_inf": 0.0, "My_sup": 0.0, "V": float(V) if V is not None else 0.0,
         "faces": _couches(el, faces)}
    for f in faces.values():
        s[CLE_MOMENT[f["cle"]]] = max(0.0, float(f["M"]))
    h = AR.h_moteur(el)
    entree = {"nom": f"Fondation de grue — bande {nom}", "b": b_cm, "h": h, "beton": el["dalle"]["beton"],
              "enrobage": round(el["armatures"]["enrobage"] * 100.0, 6), "dir_principale": faces["inf"]["cle"][-1].upper(),
              "sections": [s]}
    res = AR._moteur(entree)
    sec = res["sections"][0]
    cles = [faces["inf"]["cle"], faces["sup"]["cle"]]
    verifs = [v for v in sec["verifications"] if v["cle"] in ("hauteur", "tau", *cles)]
    # A_s,min au cm SUPÉRIEUR de h (comme le principe) : la section au cm inférieur l'écrit, la section au cm supérieur le donne
    plafond = None
    besoin = {v["cle"]: v["valeur"] for v in verifs if v["cle"] in cles}
    hc = AR.h_plafond(el)
    if hc != h:
        sec_c = AR._moteur(dict(entree, h=hc))["sections"][0]
        plafond = {"h": hc, "As_min": {c: sec_c["grandeurs"][f"As_min_{c}"] for c in cles}}
        besoin = {c: max(besoin[c], plafond["As_min"][c]) for c in cles}
        # (la vérification d'une face dit alors le besoin au cm supérieur : sa valeur, son taux, son état — la ligne
        #  « A_s,min au cm supérieur » de l'écran et de la note l'écrit)
        out_v = []
        for v in verifs:
            c = v["cle"]
            if c in cles and besoin[c] > v["valeur"]:
                insuf = besoin[c] > v["limite"]
                v = dict(v, valeur=besoin[c], taux=besoin[c] / v["limite"], etat="nok" if insuf else v["etat"],
                         motifs=list(v["motifs"]) + (["acier_insuffisant"] if insuf and "acier_insuffisant" not in v["motifs"] else []))
            out_v.append(v)
        verifs = out_v
    g = sec["grandeurs"]
    taux, nok, causes = {}, [], {}
    for v in verifs:
        if v["cle"] in cles:
            c = v["cle"]
            taux[c] = v["taux"]
            if v["etat"] != "ok":
                nok.append(c)
                causes[c] = cause(g, c, v)
    etat = "nok" if any(v["etat"] != "ok" for v in verifs) else "ok"
    return {"bande": nom, "sens": ax, "b": b_cm, "h": h, "largeur": largeur, "faces": faces, "V": V, "scia_V": bool(t.get("scia")),
            "entree": entree, "resultat": res, "section": sec, "verifications": verifs, "plafond": plafond, "besoin": besoin,
            "taux": taux, "taux_max": max(v["taux"] for v in verifs), "etat": etat, "nok": nok, "causes": causes,
            "couverture": couverture(fer, bd, faces)}


def m_rd_pose(fer: Dict, r: Dict) -> float:
    """Ce que reprend la section POSÉE d'un renfort de bande (kN·m/m) : A_s,req étant linéaire en M (moteur Dalles),
    m_Rd = A_s posée ÷ A_s,req(1 kN·m/m) — le même rapport que ``armatures.resistances`` pour le treillis seul."""
    cs = r["cle_section"]
    return pose_pm(r) * fer["res"]["mRd"][cs] / fer["res"]["As"][cs]


def couverture(fer: Dict, bd: Dict, faces: Dict) -> Dict:
    """Ce que les armatures de la bande reprennent le long de s, face par face (le diagramme de couverture de M(s)) :
    {"inf" | "sup": {"mRd" (kN·m/m : le treillis, celui que le principe compare, ``mRd_bandes``), "renforts": [{"rep", "s0",
    "s1" (m, le long de la bande), "ancrage" (m), "mRd_pose" (kN·m/m : le treillis + le renfort), "bord0", "bord1" (le
    renfort s'arrête au bord de la dalle : pas d'ancrage à y dessiner)}]}}. Le renfort ne reprend tout qu'à un ancrage de
    son extrémité : c'est pourquoi le principe le prolonge de l'ancrage de chaque côté de la zone où M dépasse le treillis."""
    out = {}
    for face, f in faces.items():
        cle = f"{face}_{AR.axe_bande(bd)}"
        rs = [{"rep": r["rep"], "s0": r["s0"], "s1": r["s1"], "ancrage": r["ancrage"], "mRd_pose": m_rd_pose(fer, r),
               "bord0": r["s0"] <= 1e-9, "bord1": r["s1"] >= bd["L"] - 1e-9} for r in f["renforts"]]
        out[face] = {"mRd": fer["mRd_bandes"][cle], "renforts": sorted(rs, key=lambda x: x["s0"])}
    return out


def cause(g: Dict, c: str, v: Dict) -> str:
    """Pourquoi une face de la section n'est pas conforme : « excessif » (A_s > 0,04·b·h), « opposee » (le minimum
    0,25·A_s,req de la face opposée gouverne et n'est pas tenu), « renfort » (A_s < A_s,req : le renfort ne couvre pas le
    moment), « minimum » (A_s,min de la dalle — 0,26·f_ctm/f_yk·b·h, 0,0013·b·h, au cm supérieur de h — non tenu : le
    principe le dit déjà, ``armatures.calculer`` « Treillis sous le minimum »), « geometrie » (d ≤ 0)."""
    As = v["limite"]
    if "acier_excessif" in v["motifs"]:
        return "excessif"
    if "d_utile_negatif" in v["motifs"]:
        return "geometrie"
    if As < g[f"As_req_{c}"]:
        return "renfort"
    if As < g[f"As_min_opposee_{c}"] and g[f"As_min_opposee_{c}"] > g["As_min_base"]:
        return "opposee"
    return "minimum"


def bandes(el: Dict, fer: Dict, ef: Dict) -> List[Dict]:
    """La section de calcul de chaque bande (``bande``), dans l'ordre des bandes."""
    return [bande(el, fer, ef, bd) for bd in ef["bandes"]]


MOTS_CAUSE = {"excessif": "plus que 0,04·b·h d'acier", "opposee": "sous le minimum 0,25·A_s,req de la face opposée",
              "renfort": "le renfort ne couvre pas le moment", "geometrie": "d ≤ 0"}


def alerte(sections: List[Dict]) -> List[tuple]:
    """Les alertes que la section AJOUTE au principe : une face que le moteur Dalles dit non conforme pour une raison que le
    principe ne voit pas (``cause`` : la face opposée, le maximum…). La hauteur, l'effort tranchant et le minimum de la dalle
    sont déjà des alertes du principe : elles ne sont pas redites (la section les écrit, en rouge, dans son détail)."""
    out = []
    for s in sections:
        if s.get("hors"):
            continue
        for c in s["nok"]:
            k = s["causes"][c]
            if k == "minimum":
                continue
            out.append((f"Bande {s['bande']}, armature {FACE_TXT[c.split('_')[0]]} {c[-1].upper()} : {MOTS_CAUSE[k]} "
                        "(moteur Dalles) — renforcer ou épaissir", "r"))
    return out

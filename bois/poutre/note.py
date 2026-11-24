# -*- coding: utf-8 -*-
"""Poutre bois — la NOTE DE CALCUL PDF : la page de garde du site, puis UNE planche A4 paysage par poutre, sur le kit
``ndc_pdf`` (la planche des notes Poutre et Dalle : le panneau de gauche — la section dessinée, les données —, deux
colonnes de calcul en flux continu, une page de suite quand elles débordent).

RIEN n'est calculé ici (R3) : la planche lit ``moteur.verifier`` — ses blocs de formules (symbole, gabarit, termes,
résultat) deviennent des substitutions « symbole = formule = valeurs = résultat » ; ses vérifications, des bandes de
verdict avec leur taux ; ses hypothèses, quelques lignes grises en fin de planche. Sobre, sans prose (le propriétaire,
24/11/2026) : la section et la qualité du bois en titre, les données au panneau, le calcul en colonnes.

    generer_note(etat_ou_poutres, infos) -> bytes      la note entière ; ValueError si une poutre est hors domaine ou si une
                                                       planche ne tient pas, même poursuivie (jamais une note tronquée)
    sections(etat_ou_poutres) -> [sec, …]               les planches, telles que le kit les lit (tests)
"""
import io
import os
import re
import tempfile
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence

from acier.garde_corps import membres as ME   # les seuils du code couleur des taux (la source unique du site : commun.niveau)
from acier.js import to_fixed
from ndc_pdf import data as ndc_data
from ndc_pdf import riche as RI
from ndc_pdf.styles import Encre

from . import donnees as D
from . import moteur as MO

TITRE_NOTE = "Note de calcul — Poutres bois"
FINE = " "                 # l'espace fine insécable entre les milliers
NBSP = " "
_ACCOLADES = re.compile(r"\{([^{}]+)\}")
_MILLIERS = re.compile(r"(?<![0-9,])([0-9]{4,})(?![0-9])")
# les décimales d'un RÉSULTAT par unité (celles de l'écran, api_web/bois.py) ; sans unité, un coefficient : trois
DECIMALES = {"N/mm²": 2, "kN": 2, "kN·m": 2, "kN/m": 3, "kN/m²": 2, "mm²": 0, "mm³": 0, "mm⁴": 0, "kg/m": 2, "kg": 1, "m": 2, "N/mm": 3,
             "kg/m³": 0}
ETATS = {ME.OK: "Vérifié", ME.ATTENTION: "À la limite", ME.NOK: "Non vérifié"}
COUL = {ME.OK: "ok", ME.ATTENTION: "att", ME.NOK: "ko"}
# la largeur d'une colonne de calcul de la planche (A4 paysage, marges, panneau et gouttières du kit) : les tableaux la lisent
COL_W = (841.89 - 2 * Encre.margin - Encre.coupe_w - Encre.gap - Encre.col_gap) / Encre.n_cols


# ------------------------------------------------------------------ les nombres
def _fixe(x: float, nd: int) -> str:
    """``x`` avec ``nd`` décimales, à la française, les milliers séparés d'une espace fine ; l'arrondi est celui du
    site (demi vers l'extérieur sur la valeur binaire exacte : acier.js.to_fixed, le même que l'écran)."""
    s = to_fixed(float(x), nd)
    if s.startswith("-") and float(s) == 0:
        s = s[1:]
    ent, _, dec = s.partition(".")
    signe, ent = ("−", ent[1:]) if ent.startswith("-") else ("", ent)
    if len(ent) > 3:                               # « 4 000 », comme l'écran (api_web.commun.fr)
        ent = f"{int(ent):,}".replace(",", FINE)
    return signe + ent + ("," + dec if dec else "")


def _nd(x: float, nd: int, sig: int = 2) -> int:
    """Assez de décimales pour garder ``sig`` chiffres significatifs aux petites valeurs (0,0345 kN·m, pas « 0,03 »)."""
    a = abs(x)
    while a and nd < 6 and a < 10 ** (sig - 1 - nd):
        nd += 1
    return nd


def _fmt(x: Optional[float], unite: str = "") -> str:
    """Un RÉSULTAT à la précision d'usage de l'écran (api_web/bois._n) : contraintes et efforts au centième, charges au
    millième, flèches au centième de millimètre (les longueurs en mm entières), coefficients à trois décimales."""
    if x is None:
        return "—"
    if unite == "mm":
        return _fixe(x, _nd(x, 2)) if abs(x) < 100 else _fixe(x, 0 if float(x).is_integer() else 1)
    if unite == "":
        return _fixe(x, _nd(x, 3))
    nd = DECIMALES.get(unite, 2)
    return _fixe(x, _nd(x, nd) if nd else 0)


def _val(x: float) -> str:
    """Une VALEUR substituée dans une formule (api_web/bois._v) : 75 reste 75, 1,35 reste 1,35 ; une grandeur calculée
    longue est arrondie à quatre chiffres significatifs au moins ; un nombre négatif est mis entre parenthèses."""
    a = abs(x)
    if a >= 1000:
        s = _fixe(x, 0 if a >= 1e5 or float(x).is_integer() else 1)
    elif a >= 100:
        s = _fixe(x, 1)
    elif a >= 10:
        s = _fixe(x, 2)
    elif a >= 1:
        s = _fixe(x, 3)
    else:
        s = _fixe(x, _nd(x, 4, 4))
    s = s.rstrip("0").rstrip(",") if "," in s else s
    return f"({s})" if x < 0 else s


def _pct(t: Optional[float]) -> str:
    """Un taux au dixième : « 45,0 % »."""
    return "—" if t is None else _fixe(100.0 * t, 1) + NBSP + "%"


def _qte(x: Optional[float], unite: str) -> str:
    """Un résultat suivi de son unité (insécable)."""
    s = _fmt(x, unite)
    return f"{s}{NBSP}{unite}" if unite and s != "—" else s


# ------------------------------------------------------------------ les formules du moteur → la planche
def _tex(s: str) -> str:
    """La notation du moteur (« σ_m,d », « 10⁶ », « √(…) », « × ») en mini-LaTeX du kit."""
    return RI.tex(s.replace(" × ", r" \cdot ").replace("×", r"\cdot"))


def _gabarit(gab: str, rendre: Callable[[str], str]) -> str:
    """Le gabarit d'une formule (« {γ_G} × {g} + {γ_Q} × {q} ») où chaque terme est rendu par ``rendre``."""
    return _tex(_ACCOLADES.sub(lambda m: rendre(m.group(1)), gab))


def _item(f: Dict):
    """L'élément de planche d'une formule du moteur : une substitution (« e ») quand elle a un gabarit, une ligne
    clé-valeur (« v », d'où la valeur vient) quand elle est lue dans un tableau ou fixée par une condition ; rien pour
    le taux (le verdict le porte)."""
    if f["cle"] == "taux":
        return None
    sym = RI.tex(f["symbole"])
    if f["gabarit"] is None:                      # une valeur lue : « k_h = 1,000 — h ≥ 150 mm : pas de majoration (§3.2(3)) »
        return ("pr", f"${sym}$ = {_qte(f['resultat'], f['unite'])} — " + RI.auto(f["expression"]))
    termes = {t["symbole"]: t["valeur"] for t in f["termes"]}
    expr = _gabarit(f["gabarit"], lambda t: t)
    vals = _gabarit(f["gabarit"], lambda t: _val(termes[t]))
    res = _fmt(f["resultat"], f["unite"]) + (r" \u{" + f["unite"] + "}" if f["unite"] else "")
    return ("e", None, [sym, expr, vals], res)


def _items(r: Dict, cles: Sequence[str], sauf: Sequence[str] = ()) -> List[tuple]:
    """Les éléments des blocs ``cles`` du détail du calcul, dans l'ordre du moteur, moins les formules ``sauf``."""
    par_cle = {b["cle"]: b for b in r["calcul"]}
    out = []
    for k in cles:
        for f in par_cle[k]["formules"]:
            if f["cle"] in sauf:
                continue
            it = _item(f)
            if it is not None:
                out.append(it)
    return out


def _verdict(r: Dict, v: Dict) -> Dict:
    """La bande de verdict d'une vérification : ce qu'elle compare, sa couleur (les seuils du site), son taux à droite."""
    u = v["unite"]
    signe = "≤" if v["etat"] == MO.OK else ">"
    if v["groupe"] == MO.ELS:
        n = r["fleches"]["n_inst" if v["cle"] == "fleche_inst" else "n_fin"]
        texte = f"${RI.tex(v['symbole'])}$ = {_qte(v['valeur'], u)} {signe} L/{n} = {_qte(v['limite'], u)}"
    else:
        texte = f"${RI.tex(v['symbole'])}$ = {_qte(v['valeur'], u)} {signe} ${RI.tex(v['symbole_limite'])}$ = {_qte(v['limite'], u)}"
    return {"texte": texte, "etat": COUL[ME.statut(v["taux"])], "cale": _pct(v["taux"]), "riche": True}


def _comb(r: Dict, cle: str) -> Dict:
    return next(c for c in r["combinaisons"] if c["cle"] == cle)


# ------------------------------------------------------------------ le panneau : le dessin et les données
def _dessin(r: Dict) -> Callable:
    """Le dessinateur du panneau : la section b × h à l'échelle (fil du bois ou lamelles), cotée ; à sa droite, le schéma de
    la poutre — appuis, charge répartie, portée, maintien latéral."""
    geo, mat, dv = r["geometrie"], r["materiau"], r["deversement"]
    b, h, L = geo["b"], geo["h"], geo["L"]

    def dessin(d, x, y, w, hc, S):
        # ---- la section, à gauche (un tiers de la largeur)
        ws, hs = w * 0.34, hc - 26
        k = min(ws / b, hs / h) * 0.86
        bw, bh = b * k, h * k
        x0, y0 = x + (ws - bw) / 2, y + 14 + (hs - bh) / 2
        d.box(x0, y0, bw, bh, fill="#F6EBD7", stroke=S.ink, lw=0.9)
        if mat["type"] == D.LAMELLE:
            n = max(2, round(h / D.EPAISSEUR_LAMELLE))
            for i in range(1, n):
                yy = y0 + bh * i / n
                d.line(x0, yy, x0 + bw, yy, S.mut, 0.4)
        else:
            for i in range(1, 4):                     # le fil du bois : trois veines
                xx = x0 + bw * i / 4
                d.line(xx, y0 + 2, xx, y0 + bh - 2, "#C9B48F", 0.5)
        d.t(x0 + bw / 2, y0 + bh + 5, f"b = {_fixe(b, 0 if b.is_integer() else 1)}", S.f_b, 6.2, S.mut, "center")
        d.t(x0 - 4, y0 + bh / 2 - 2, f"h = {_fixe(h, 0 if h.is_integer() else 1)}", S.f_b, 6.2, S.mut, "right")
        d.t(x + ws / 2, y + 2, "section · mm", S.f_v, 5.4, S.mut, "center", track=1.0)
        # ---- le schéma, à droite
        xs0, xs1 = x + ws + 14, x + w - 6
        ym = y + hc * 0.46
        d.line(xs0, ym, xs1, ym, S.ink, 1.6)
        for xa in (xs0, xs1):                         # les appuis : deux triangles
            d.c.setFillColor(d.c._fillColorObj if False else __import__("reportlab.lib.colors", fromlist=["HexColor"]).HexColor(S.ink))
            p = d.c.beginPath()
            p.moveTo(xa, ym - 1)
            p.lineTo(xa - 5, ym - 9)
            p.lineTo(xa + 5, ym - 9)
            p.close()
            d.c.drawPath(p, stroke=0, fill=1)
        n_fl = 9                                      # la charge répartie : des flèches régulières
        for i in range(n_fl):
            xa = xs0 + (xs1 - xs0) * i / (n_fl - 1)
            d.line(xa, ym + 22, xa, ym + 5, S.acc, 0.6)
            d.line(xa - 2, ym + 8, xa, ym + 4, S.acc, 0.6)
            d.line(xa + 2, ym + 8, xa, ym + 4, S.acc, 0.6)
        d.line(xs0, ym + 22, xs1, ym + 22, S.acc, 0.7)
        d.t((xs0 + xs1) / 2, ym + 26, "g + q", S.f_b, 6.4, S.acc, "center")
        d.line(xs0, ym - 18, xs1, ym - 18, S.mut, 0.5)
        d.line(xs0, ym - 21, xs0, ym - 15, S.mut, 0.5)
        d.line(xs1, ym - 21, xs1, ym - 15, S.mut, 0.5)
        d.t((xs0 + xs1) / 2, ym - 28, f"L = {_fixe(L, 2)} m", S.f_b, 6.4, S.mut, "center")
        maintien = {"continu": "fibre comprimée maintenue en continu", "appuis": "maintien latéral aux appuis seulement",
                    "entretoises": f"entretoises tous les {_fixe(dv['entraxe'] / 1000.0, 2)} m" if dv.get("entraxe") else "entretoises"}[dv["type"]]
        d.t((xs0 + xs1) / 2, y + 2, d.fit(maintien, S.f_b, 5.6, xs1 - xs0), S.f_b, 5.6, S.mut, "center")
    return dessin


def _blocs(r: Dict) -> List[tuple]:
    """Les quatre blocs de données du panneau : la poutre, le bois, les charges, les coefficients — (libellé, symbole en
    mini-LaTeX, valeur, unité)."""
    p, geo, mat, ch, co, ap, fl = r["poutre"], r["geometrie"], r["materiau"], r["charges"], r["coefficients"], r["appui"], r["fleches"]
    maintien = {"continu": "continu", "appuis": "aux appuis", "entretoises": "entretoises"}[p["maintien"]["type"]]
    poutre = [("Section", "b × h", geo["designation"], "mm"), ("Portée", "L", _fixe(geo["L"], 2), "m"),
              ("Appui", "a", _fixe(ap["a"], 0), "mm"), ("Maintien", "", maintien, ""),
              ("Classe de service", "", str(co["service"]), ""), ("Limites de flèche", "", f"L/{fl['n_inst']} · L/{fl['n_fin']}", "")]
    essence = mat["libelle_essence"] if mat["essence_dite"] else D.COURTS_TYPE[mat["type"]]
    bois = [("Essence", "", essence, ""), ("Classe", "", mat["classe"], "")]
    if mat["qualites"]:                             # la première qualité de classement visuel qui mène à la classe (la règle française)
        bois.append(("Qualité visuelle", "", mat["qualites"][0], ""))
    bois += [("Flexion", RI.tex("f_m,k"), _fixe(mat["fm_k"], 1), "N/mm²"),
            ("Cisaillement", RI.tex("f_v,k"), _fixe(mat["fv_k"], 1), "N/mm²"), ("Compression ⊥", RI.tex("f_c,90,k"), _fixe(mat["fc90_k"], 1), "N/mm²"),
            ("Module", RI.tex("E_0,mean"), _fixe(mat["E0_mean"], 0), "N/mm²"), ("", RI.tex("E_0,05"), _fixe(mat["E0_05"], 0), "N/mm²"),
            ("", RI.tex("G_mean"), _fixe(mat["G_mean"], 0), "N/mm²"), ("Masse volumique", RI.tex("ρ_mean"), _fixe(mat["rho_mean"], 0), "kg/m³")]
    if ch["mode"] == "surfacique":
        charges = [("Permanente", RI.tex("g_s"), _fmt(ch["g_surf"], "kN/m²"), "kN/m²"), ("Variable", RI.tex("q_s"), _fmt(ch["q_surf"], "kN/m²"), "kN/m²"),
                   ("Entraxe", "e", _fixe(ch["entraxe"], 2), "m")]
    else:
        charges = [("Permanente", "G", _fmt(ch["G"], "kN/m"), "kN/m"), ("Variable", "Q", _fmt(ch["Q"], "kN/m"), "kN/m")]
    charges += [("Poids propre", RI.tex("g_pp"), _fmt(ch["poids_propre"], "kN/m") + ("" if ch["poids_propre_compte"] else " (non compté)"), "kN/m"),
                ("Nature de Q", "", co["libelle_nature"], ""), ("Durée", "", co["libelle_duree"], ""), ("Quasi permanent", RI.tex("ψ_2"), _fixe(co["psi2"], 2), "")]
    coefs = [("Permanent", RI.tex("k_mod"), _fixe(co["kmod_permanent"], 2), ""), (co["libelle_duree"].capitalize(), RI.tex("k_mod"), _fixe(co["kmod_variable"], 2), ""),
             ("Fluage", RI.tex("k_def"), _fixe(co["kdef"], 2), ""), ("Hauteur", RI.tex("k_h"), _fmt(co["kh"]), ""),
             ("Déversement", RI.tex("k_crit"), _fmt(r["deversement"]["kcrit"]), ""), ("Appui", RI.tex("k_c,90"), _fmt(ap["kc90"]), ""),
             ("Matériau", RI.tex("γ_M"), _fixe(co["gamma_M"], 2), ""), ("Charges", RI.tex("γ_G · γ_Q"), f"{_fixe(co['gamma_G'], 2)} · {_fixe(co['gamma_Q'], 2)}", "")]
    if co["systeme"]:
        coefs.append(("Effet système", RI.tex("k_sys"), _fixe(co["ksys"], 2), ""))
    return [("POUTRE", poutre), ("BOIS", bois), ("CHARGES", charges), ("COEFFICIENTS", coefs)]


# ------------------------------------------------------------------ les vérifications (les colonnes)
def _verifs(r: Dict) -> List[Dict]:
    """Les vérifications de la planche, dans l'ordre de la note : charges et combinaisons, flexion (et déversement),
    cisaillement, compression à l'appui, flèches, hypothèses."""
    par_cle = {v["cle"]: v for v in r["verifications"]}
    k_elu = r["combinaison_elu"]
    out = []

    def verif(titre, items, verdicts=()):
        # le kit ne dessine une bande de verdict QUE là où un élément (« k », i) la place : sans emplacement donné, chaque
        # verdict ferme la vérification (tests/test_note_bois.py a trouvé les verdicts ELU absents de la planche, 24/11/2026)
        items = list(items)
        if verdicts and not any(it[0] == "k" for it in items):
            items += [("k", i) for i in range(len(verdicts))]
        out.append({"num": len(out) + 1, "titre": titre, "items": items, "verdicts": list(verdicts)})

    # 1 · les charges et les combinaisons : g (et g_pp), le tableau des combinaisons, les formules de celle qui gouverne l'ELU
    items = _items(r, ["charges"], sauf=("q",) if not r["charges"]["mode"] == "surfacique" else ())
    cols = [("Combinaison", 0.36 * COL_W, "left"), ("$k_mod$", 0.14 * COL_W, "right"), ("$p_d$ (kN/m)", 0.17 * COL_W, "right"),
            ("$M_d$ (kN·m)", 0.17 * COL_W, "right"), ("$V_d$ (kN)", 0.16 * COL_W, "right")]
    lignes = [[("lab", c["nom"], c["cle"] == k_elu), ("num", _fixe(c["kmod"], 2)), ("num", _fmt(c["p_d"], "kN/m")), ("num", _fmt(c["M_d"], "kN·m")),
               ("num", _fmt(c["V_d"], "kN"))] for c in r["combinaisons"]]
    items.append(("tab", cols, lignes, f"ELU, NBN EN 1990 (6.10) ; k_mod de l'action la plus courte (§3.1.3(2)) ; en gras, la combinaison qui gouverne l'ELU."))
    items += _items(r, [f"combinaison_{k_elu}"], sauf=("kmod",))
    verif("Charges et combinaisons", items)
    # 2 · la flexion, avec le déversement
    v = par_cle["flexion"]
    k = v["combinaison"]
    items = _items(r, ["section"], sauf=("A", "I")) + _items(r, ["deversement"]) + _items(r, [f"flexion_{k}"])
    if k != k_elu:
        items.insert(0, ("pr", f"Gouvernée par la combinaison {_comb(r, k)['nom']} ($k_mod$ = {_fixe(_comb(r, k)['kmod'], 2)})."))
    verif("Flexion et déversement — §6.1.6, §6.3.3", items, [_verdict(r, v)])
    # 3 · le cisaillement
    v = par_cle["cisaillement"]
    k = v["combinaison"]
    items = _items(r, [f"cisaillement_{k}"])
    if k != k_elu:
        items.insert(0, ("pr", f"Gouverné par la combinaison {_comb(r, k)['nom']} ($k_mod$ = {_fixe(_comb(r, k)['kmod'], 2)})."))
    verif("Cisaillement — §6.1.7", items, [_verdict(r, v)])
    # 4 · la compression à l'appui
    if "appui" in par_cle:
        v = par_cle["appui"]
        k = v["combinaison"]
        items = _items(r, ["appui", f"appui_{k}"])
        if k != k_elu:
            items.insert(0, ("pr", f"Gouvernée par la combinaison {_comb(r, k)['nom']} ($k_mod$ = {_fixe(_comb(r, k)['kmod'], 2)})."))
        verif("Compression à l'appui — §6.1.5", items, [_verdict(r, v)])
    else:
        verif("Compression à l'appui — §6.1.5", [("pr", "Non vérifiée : retirée de la note (la réaction d'appui est à vérifier par ailleurs).")])
    # 5 · les flèches : la formule de chaque part (flexion, effort tranchant) écrite une fois, pour g ; pour q, la somme
    # seulement ; les limites L/n sont dans les verdicts ; w_net,fin ne s'écrit qu'avec une contre-flèche
    sauf_fin = ("psi2", "lim_fin") + (("w_net_fin",) if r["fleches"]["w_c"] == 0 else ())
    items = _items(r, ["fleche_inst"], sauf=("w_q_m", "w_q_v", "lim_inst")) + [("k", 0)] + _items(r, ["fleche_fin"], sauf=sauf_fin) + [("k", 1)]
    verif("Flèches — §2.2.3, §7.2", items, [_verdict(r, par_cle["fleche_inst"]), _verdict(r, par_cle["fleche_fin"])])
    # 6 · les hypothèses PROPRES à cette poutre (le maintien, l'appui, un débord, un feuillu, l'effet système), ce qui n'est
    # pas couvert, les normes — des lignes grises ; les hypothèses générales du moteur (la poutre sur deux appuis, les deux
    # combinaisons, l'effort tranchant à l'axe, les flèches, l'essence écrite) sont dites par la planche elle-même
    items = [("pr", RI.auto(h)) for h in r["hypotheses"] if not h.startswith(HYPOTHESES_GENERALES)]
    items.append(("pr", "Non couvert : " + " ; ".join(r["non_couvert"]) + "."))
    items.append(("pr", RI.auto(r["reference"] + ".")))
    verif("Hypothèses", items)
    return out


HYPOTHESES_GENERALES = ("Poutre de section rectangulaire", "ELU, NBN EN 1990 (6.10)", "Effort tranchant pris à l'axe", "Flèches en combinaison",
                        "L'essence (")


def _etat(r: Dict) -> str:
    return ETATS[ME.statut(r["taux_max"])] if r["statut"] == MO.OK else ETATS[ME.NOK]


def section(poutre: Dict) -> Dict:
    """La planche d'UNE poutre, telle que le kit la lit (``ndc_pdf.styles.Style.planche``) : le titre, les matériaux, l'état,
    le dessin et les données du panneau, les vérifications."""
    r = MO.verifier(poutre)
    geo, mat = r["geometrie"], r["materiau"]
    qualite = mat["classe"] + (f" {mat['essence_dite']}" if mat["essence_dite"] else "")
    blocs = _blocs(r)
    # la place à garder sous la figure : les deux piles de blocs (gauche : 1 et 3, droite : 2 et 4), 13 pt la ligne
    reserve = 26 + 13 * max(sum(1 + len(rows) for rows in (blocs[0][1], blocs[2][1])), sum(1 + len(rows) for rows in (blocs[1][1], blocs[3][1])))
    # à droite du titre, et sur la ligne du sommaire : le type de bois, sa norme, le taux de travail le plus fort (le « η » du
    # verdict de l'écran, au dixième ; le chip de l'écran l'arrondit à l'unité)
    return {"poutre": poutre.get("nom") or "Poutre", "section": f"{geo['designation']} mm · {qualite} · L = {_fixe(geo['L'], 2)} m",
            "beton": mat["libelle_type"], "acier": f"{mat['norme']} · η max {_pct(r['taux_max'])}", "etat": _etat(r), "titre_coupe": "SECTION ET SCHÉMA",
            "dessin": _dessin(r), "coupe_h": 150, "reserve": reserve, "blocs": blocs, "verifs": _verifs(r), "taux_max": r["taux_max"],
            "statut": r["statut"]}


def poutres_de(etat_ou_poutres) -> List[Dict]:
    """Les poutres d'une note : l'état du site ({"poutres": [...]}) ou une liste ; ValueError s'il n'y en a aucune."""
    liste = etat_ou_poutres.get("poutres") if isinstance(etat_ou_poutres, dict) else etat_ou_poutres
    if not isinstance(liste, list) or not liste:
        raise ValueError("La note n'a aucune poutre : rien à imprimer.")
    return list(liste)


def sections(etat_ou_poutres) -> List[Dict]:
    """Les planches de toutes les poutres de la note, dans l'ordre ; une poutre hors domaine arrête la note avec le message
    du moteur (jamais un chiffre faux imprimé)."""
    out = []
    for i, p in enumerate(poutres_de(etat_ou_poutres)):
        try:
            out.append(section(p))
        except ValueError as err:
            nom = (p.get("nom") if isinstance(p, dict) else None) or f"n° {i + 1}"
            raise ValueError(f"La note n'est pas préparée : poutre « {nom} », {err}") from err
    return out


class Planche(Encre):
    """Le style des notes du site (palette Encre), la planche poursuivie sur une page suivante quand ses colonnes
    débordent — jamais une troisième colonne (le propriétaire, 14/11/2026)."""
    planche_suite = True


def generer_note(etat_ou_poutres, infos: Optional[Dict] = None, chemin: Optional[str] = None) -> bytes:
    """La note (bytes PDF) : la garde du site (le sommaire : une ligne par poutre, sa section et sa qualité, son état),
    puis une planche par poutre. ``infos`` : nom_projet, partie, date, indice. Ce qui déborderait encore (le panneau, une
    formule plus haute qu'une colonne) est REFUSÉ avec un message : jamais une note tronquée."""
    infos = infos or {}
    doc_meta = ndc_data.construire_doc(infos, date_defaut=datetime.today().strftime("%d/%m/%Y"))
    doc_meta["titre"] = TITRE_NOTE
    secs = sections(etat_ou_poutres)
    fichier = chemin
    if fichier is None:
        fd, fichier = tempfile.mkstemp(suffix=".pdf", prefix="note_bois_")
        os.close(fd)
    try:
        d = Planche().build(fichier, sections=secs, doc=doc_meta)
        if d.warnings:
            raise ValueError("La note ne tient pas sur sa planche (" + " ; ".join(d.warnings) + ") : elle ne sort pas tronquée. "
                             "À signaler au bureau ; l'écran garde tout le calcul.")
        d.save()
        with open(fichier, "rb") as fh:
            return fh.read()
    finally:
        if chemin is None:
            try:
                os.remove(fichier)
            except OSError:
                pass

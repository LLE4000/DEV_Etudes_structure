# -*- coding: utf-8 -*-
"""Les planches du FERRAILLAGE de la note PDF « Fondation de grue » (lot 4d, § 6.4), à la suite des planches qui
existent, dans la même mise en page (``note_saf.Planche``) :

1. « Efforts » — les diagrammes M(s) et V(s) en enveloppe des quatre bandes, avec leurs valeurs extrêmes ;
2. « Ferraillage » — le plan ferraillé ; le tableau des treillis, des renforts, des U et des cages, avec les contrôles
   du moteur poutre (comme la carte du module poutre) ;
3. « Coupes » — A-A et B-B ;
4. « Bordereau » — le bordereau des aciers et les quantités ;
et la ligne « Non vérifiés ».

Ce fichier ne calcule rien : il met en planches le ferraillage de l'écran (``saf.ferraillage``) — les mêmes chiffres.
"""
from typing import Dict, List

import numpy as np

from saf.fondation_grue import ferraillage as FE

from . import commun as C
from . import dessin_ferraillage as DF
from . import dessin_saf as DS

N = C.NBSP


def _n(x: float, nd: int = 1) -> str:
    return C.fr(x, nd)


# ------------------------------------------------------------------ les diagrammes M(s), V(s)
def diagrammes(ef: Dict) -> str:
    """Les diagrammes M(s) (le plus grand et le plus petit, kNm, intégrés sur la bande) et |V|(s) (kN) de l'enveloppe,
    une ligne par bande, avec les valeurs extrêmes ; s le long de la bande (m)."""
    W, Hc, gx, gy = 470.0, 120.0, 60.0, 46.0
    out = []
    for i, b in enumerate(ef["bandes"]):
        y0 = i * (Hc + gy) + 30.0
        for j, (titre, courbes, unite) in enumerate(((f"{b['nom']} — M(s)", [(b["Mmax"], DF.INF), (b["Mmin"], DF.SUP)], "kNm"),
                                                    (f"{b['nom']} — |V|(s)", [(b["Vabs"], DS.ENCRE)], "kN"))):
            x0 = j * (W + gx)
            s = b["s"]
            vals = np.concatenate([c for c, _ in courbes])
            vmax, vmin = max(float(vals.max()), 0.0), min(float(vals.min()), 0.0)
            echy = Hc / max(vmax - vmin, 1e-9)
            X = lambda x: x0 + x / max(s[-1], 1e-9) * W          # noqa: E731
            Y = lambda v: y0 + (vmax - v) * echy                 # noqa: E731
            out.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{W:.1f}" height="{Hc:.1f}" fill="none" stroke="{DS.SCHEMA}" stroke-width="0.8"/>')
            out.append(f'<line x1="{x0:.1f}" y1="{Y(0):.1f}" x2="{x0 + W:.1f}" y2="{Y(0):.1f}" stroke="{DS.SCHEMA}" stroke-width="0.8"/>')
            for c, col in courbes:
                pts = " ".join(f"{X(a):.1f},{Y(v):.1f}" for a, v in zip(s, c))
                out.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="1.6"/>')
            out.append(DS._texte(x0, y0 - 8, titre + f" ({unite})", 13, DS.ENCRE, "start", gras=True, halo=False))
            if j == 0:
                ip, im = int(np.argmax(b["Mmax"])), int(np.argmin(b["Mmin"]))
                out.append(DS._texte(X(s[ip]), Y(b["Mmax"][ip]) - 4, f"M⁺ {_n(b['M_plus'])}", 11, DF.INF, "middle"))
                out.append(DS._texte(X(s[im]), Y(b["Mmin"][im]) + 13, f"M⁻ {_n(b['M_moins'])}", 11, DF.SUP, "middle"))
            else:
                out.append(DS._texte(x0 + W - 4, y0 + 14, f"V à d (d du moteur) : voir le ferraillage · max {_n(float(b['Vabs'].max()))}", 11,
                                     DS.ENCRE, "end"))
            out.append(DS._texte(x0, y0 + Hc + 13, "0", 10, DS.SCHEMA, "start", halo=False))
            out.append(DS._texte(x0 + W, y0 + Hc + 13, f"s = {_n(float(s[-1]), 2)} m", 10, DS.SCHEMA, "end", halo=False))
    h = len(ef["bandes"]) * (Hc + gy) + 30.0
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="-10 0 {2 * W + gx + 20:.1f} {h:.1f}" role="img" aria-label="Diagrammes des efforts" '
            f'class="df-diagrammes">{"".join(out)}</svg>')


# ------------------------------------------------------------------ les tableaux
def _tab_bandes(fer: Dict, ef: Dict, largeur: float) -> Dict:
    w0 = 34.0
    w = (largeur - w0) / 4
    cols = [("BANDE", w0, "left"), ("M⁺ [kNm]", w, "right"), ("M⁻ [kNm]", w, "right"), ("V à d [kN]", w, "right"), ("PAR m", w, "right")]
    lignes = []
    for b in ef["bandes"]:
        t = fer["tranchant"][b["nom"]]
        lignes.append([("lab", b["nom"], True), ("num", _n(b["M_plus"])), ("num", _n(b["M_moins"])), ("num", _n(t["V"])),
                       ("lab", f"÷ {_n(b['largeur'], 2)} m")])
    return {"titre": "Efforts des bandes (enveloppe)", "items": [
        ("pr", "M et V intégrés sur la largeur EXACTE de la bande (bords interpolés), en enveloppe de toutes les positions de tous les jeux et des k "
               "cochés « SCIA » ; m > 0 : fibre inférieure tendue. Valeurs par mètre = valeur intégrée ÷ largeur de la bande. V à d : le plus grand "
               "|V(s)| à d du bord des zones chargées, d étant celui du moteur poutre."),
        ("tab", cols, lignes)], "verdicts": []}


def _tab_ferraillage(fer: Dict, ef: Dict, largeur: float) -> Dict:
    w0 = 44.0
    w1 = largeur - w0 - 40.0
    cols = [("ÉLÉMENT", w0, "left"), ("FERRAILLAGE ET CONTRÔLE DU MOTEUR POUTRE", w1, "left"), ("TAUX", 40.0, "right")]
    lignes = []
    for cote, nom in (("inf", "Treillis inf."), ("sup", "Treillis sup.")):
        f = fer["faces"][cote]
        n_loc = sum(1 for r in f["locaux"] if not r["impossible"])
        t = max(float((cv["besoin"][cv["controle"]] / cv["As_place"][cv["controle"]]).max()) if cv["controle"].any() else 0.0
                for cv in f["couverture"].values())
        lignes.append([("lab", nom, True), ("lab", f"{FE.libelle_treillis(f['code'])} (x et y), {'automatique' if f['auto'] else 'imposé'}"
                                                    f"{f', + {n_loc} renfort(s) local(aux)' if n_loc else ''} — A_s = {_n(f['As_pm'], 0)} mm²/m"),
                       ("num", f"{C.pct(t)} %")])
    for cote in ("inf", "sup"):
        for nom, bf in fer["faces"][cote]["bandes"].items():
            r = bf["renfort"]
            reps = [ln["rep"] for ln in fer["bordereau"]["lignes"] if ln["famille"] == "R" and ln.get("bande") == nom and ln.get("cote") == cote]
            txt = (f"{cote}. : A_s,néc = {_n(bf['retenu']['necessaire_pm'], 0)} ≤ {_n(bf['As_place_pm'], 0)} mm²/m — "
                   + (f"{reps[0]} : {r['n']} Ø{r['d']}, L = {_n(r['L'], 2)} m" if r else "treillis seul"))
            lignes.append([("lab", nom, True), ("lab", txt), ("num", f"{C.pct(bf['taux'])} %")])
    for nom, t in fer["tranchant"].items():
        if t["tau"] is None:
            continue
        txt = f"effort tranchant : τ = {_n(t['tau'], 3)} ≤ τ_adm,{t['bande']} = {_n(t['tau_lim'], 3)} N/mm² (V à d = {_n(t['V'])} kN, d = {_n(t['d_v'] * 100)} cm)"
        if t["C"]:
            txt += f" — C Ø{t['C']['d']} e = {_n(t['C']['s'])} cm"
        lignes.append([("lab", nom, True), ("lab", txt), ("num", f"{C.pct(t['taux'])} %")])
    for nom, hb in fer["hauteur"].items():
        lignes.append([("lab", "Zone courante" if nom == "courante" else nom, True), ("lab", f"hauteur : h_min = {_n(hb['h_min'])} ≤ h = {_n(hb['h'], 0)} cm"),
                       ("num", f"{C.pct(hb['taux'])} %")])
    u = fer["U"]
    lignes.append([("lab", "U1", True), ("lab", f"U de fermeture Ø{u['d']} e = {_n(u['e'], 0)} cm, {u['n']} U (âme {_n(u['ame'], 2)} m, branches {_n(u['branche'], 2)} m)"),
                   ("lab", "")])
    for cg in fer["cages"][:1]:
        lignes.append([("lab", "Plots", True), ("lab", f"cage : C Ø{cg['d']} e = {_n(cg['e'], 0)} cm sur les 4 faces, 2 U en tête ; brins verticaux sous le pied : "
                                                      "à dimensionner"), ("lab", "")])
    for titre, texte, niv in fer["lignes"]:
        lignes.append([("lab", titre, True), ("lab", texte), ("lab", "non traité" if niv == "nok" else "non vérifié")])
    return {"titre": "Ferraillage", "items": [
        ("pr", "Méthode du bureau, ELS : chaque bande est une poutre de largeur b (section h + plot au droit d'un plot coulé, face inférieure) ; "
               "treillis et renforts locaux par mètre, avec les moments de calcul m ± |mxy| ; l'acier demandé est celui du moteur du module "
               "« Poutres béton armé »."),
        ("tab", cols, lignes)], "verdicts": [{"etat": "ok" if fer["ok"] else "ko",
                                              "texte": "Ferraillage conforme" if fer["ok"] else "Ferraillage non conforme"}]}


def forme_texte(ln: Dict) -> str:
    """La forme d'une barre en mots, avec ses cotes (m) : « droite 7,90 », « U 0,60 / 0,40 / 0,60 »."""
    cotes = " / ".join(_n(c, 2) for c in ln["cotes"])
    return {"droite": "droite", "U": "U", "C": "C"}[ln["forme"]] + " " + cotes


def _tab_bordereau(fer: Dict, largeur: float) -> List[Dict]:
    bd = fer["bordereau"]
    w = [26.0, 18.0, 0.0, 24.0, 28.0, 30.0, 30.0]
    w[2] = largeur - sum(w)
    cols = [("REP.", w[0], "left"), ("Ø", w[1], "right"), ("FORME [m]", w[2], "left"), ("NOMBRE", w[3], "right"), ("L [m]", w[4], "right"),
            ("L TOT. [m]", w[5], "right"), ("POIDS [kg]", w[6], "right")]
    lignes = [[("lab", ln["rep"], True), ("num", str(ln["d"])), ("lab", forme_texte(ln)), ("num", str(ln["n"])), ("num", _n(ln["L"], 2)),
               ("num", _n(ln["Lt"], 1)), ("num", _n(ln["poids"], 1))] for ln in bd["lignes"]]
    tot = [[("lab", f"Ø{d}", True), ("num", _n(p, 1)), ("lab", "kg")] for d, p in bd["par_d"].items()]
    wq = (largeur - 40.0) / 2
    cols2 = [("TOTAL", 40.0, "left"), ("", wq, "right"), ("", wq, "left")]
    tot += [[("lab", "Acier", True), ("num", _n(bd["total"], 1)), ("lab", "kg")],
            [("lab", "Béton", True), ("num", _n(bd["beton"], 2)), ("lab", "m³ (dalle + plots coulés)")],
            [("lab", "Ratio", True), ("num", _n(bd["ratio"], 1)), ("lab", "kg/m³")]]
    return [{"titre": "Bordereau des aciers", "items": [
        ("pr", "T1 à T4 : treillis inférieur x, inférieur y, supérieur x, supérieur y (barres filantes) ; R : renforts de bandes ; L : renforts locaux ; "
               "U : U de fermeture ; C : cages des plots ; V : C d'effort tranchant. Longueur d'une barre façonnée = somme des cotes ; barres regroupées "
               "par longueur au 5 cm supérieur ; poids = π·Ø²/4 × 7 850 kg/m³."),
        ("tab", cols, lignes)], "verdicts": []},
        {"titre": "Quantités", "items": [("tab", cols2, tot)], "verdicts": []}]


# ------------------------------------------------------------------ les planches
def planches(el: Dict, calc: Dict, s: Dict, style, fer: Dict, ef: Dict, planche, non_verifies: str) -> List[Dict]:
    """Les quatre planches du ferraillage (``planche`` : ``note_saf._planche``)."""
    largeur = style.largeur_colonne()
    m = calc["modele"]
    nv = {"titre": "Non vérifiés", "items": [("pr", non_verifies)], "verdicts": []}
    blocs_b = [("BANDES", [("Largeur", "b", _n(ef["bandes"][0]["largeur"], 2), "m"), ("Maille", "", _n(m["maille"], 2), "m")]),
               ("MOTEUR", [("Méthode", "", "bureau, ELS", ""), ("Torsion", "", "oui" if ef["torsion"] else "non", "")])]
    out = [planche(el, s, "Efforts", "DIAGRAMMES M(s) ET V(s) — ENVELOPPE", diagrammes(ef), blocs_b, [_tab_bandes(fer, ef, largeur)], style)]
    bd = fer["bordereau"]
    blocs_f = [("TREILLIS", [("Inférieur", "", FE.libelle_treillis(fer["faces"]["inf"]["code"]), ""),
                             ("Supérieur", "", FE.libelle_treillis(fer["faces"]["sup"]["code"]), "")]),
               ("QUANTITÉS", [("Acier", "", _n(bd["total"], 0), "kg"), ("Béton", "", _n(bd["beton"], 1), "m³"), ("Ratio", "", _n(bd["ratio"], 0), "kg/m³")])]
    out.append(planche(el, s, "Ferraillage", "PLAN FERRAILLÉ — COTES EN m", DF.plan(el, m, fer, ef, ident="note_fer", carte="vecteur"), blocs_f,
                       [_tab_ferraillage(fer, ef, largeur), nv], style))
    blocs_c = [("COUPES", [("A-A", "", "le long de BX1", ""), ("B-B", "", "le long de BY1", "")]),
               ("ENROBAGE", [("Enrobage", "c", _n(fer["enrobage"] * 100, 1), "cm"), ("Hauteur", "h", _n(fer["h"], 2), "m")])]
    out.append(planche(el, s, "Coupes", "COUPES A-A ET B-B — À L'ÉCHELLE", DF.coupes(el, m, fer, ef), blocs_c,
                       [{"titre": "Lecture des coupes", "items": [("pr", "Traits : les barres dans le sens de la coupe ; points : les barres de l'autre "
                                                                         "sens ; en bleu la face inférieure, en rouge la face supérieure ; U aux deux bouts ; "
                                                                         "mêmes repères que le bordereau.")], "verdicts": []}], style))
    out.append(planche(el, s, "Bordereau", "BORDEREAU DES ACIERS", None, blocs_f[1:], _tab_bordereau(fer, largeur), style))
    return out

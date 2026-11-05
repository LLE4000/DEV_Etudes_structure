# -*- coding: utf-8 -*-
"""Note de calcul « Soudures » — DEUX pages A4 paysage, sur le kit ``ndc_pdf``
des notes de l'application (palette Encre, mêmes marges et mêmes polices que
les notes d'assemblage et de plancher).

Page 1 · CALCUL : bandeau (statut, η max, cordon et point gouvernants,
critère, la méthode CHOISIE, la configuration, les efforts et la méthode en
deux lignes), puis trois colonnes — les SCHÉMAS TECHNIQUES tels qu'à l'écran
(les scènes de ``schemas.py`` peintes en vectoriel par le peintre du kit des
assemblages, source unique) avec l'encart LA GORGE et les HYPOTHÈSES ; le
tableau des CORDONS et celui des EFFORTS avec les propriétés du groupe ; le
tableau des VÉRIFICATIONS par point, la VÉRIFICATION AU POINT GOUVERNANT
écrite comme à la main (le critère de la méthode choisie en typographie
mathématique, puis les substitutions numériques de ``formules.py`` : groupe,
point, critère — testées contre le moteur) et la CONCLUSION. La note
n'imprime QUE la méthode choisie ; la comparaison des quatre lectures reste
à l'écran (carte RÉSULTAT, onglet Détail).

Page 2 · MÉTHODE ET DÉTAIL : les sections de ``methode.py`` de la méthode
choisie (mêmes textes que l'onglet Méthode), la CARTE DES TAUX de tous les
cordons, le détail complet par point s'il ne tenait pas en page 1, les
références, et un cartouche (cas, configuration, méthode, acier, date ·
indice, échelle du dessin).

``derniers_avertissements`` : débordements de colonne (vide si tout tient) ;
la note est d'abord composée en taille normale, puis en taille réduite si une
colonne déborde (variante « serrée »), comme les autres notes.
"""
import math
import os
import tempfile
from datetime import datetime
from typing import Dict, List, Optional

from reportlab.lib.pagesizes import A4, landscape

from ndc_pdf import data as ndc_data
from ndc_pdf.kit import Doc, Frame, chip, formula, mix
from ndc_pdf.styles import Encre
from ndc_pdf.svg import peindre_svg
from acier.assemblages.poutre_poutre.doubles_cornieres.rapport import peindre as peindre_scene   # peintre des scènes du kit

from . import configurations as CF
from . import criteres as K
from . import dessin as DS
from . import etat as ET
from . import formules as FO
from . import methode as ME
from . import schemas as SC

A4L = landscape(A4)
M, BAS, GAP = 22, 40, 12
LARG = (318, 212)                     # colonnes 1 et 2 de la page 1 ; la 3e prend le reste
S_TAB_H, S_DATA = 5.6, 7.0
LEAD_TAB = 8.8
TITRE = "Soudures — cordons d'angle et bout à bout"
STATUTS = {ET.OK: ("CONFORME", "ok"), ET.ATTENTION: ("CONFORME · ATTENTION", "att"), ET.NOK: ("NON CONFORME", "ko"),
           ET.REFUS: ("CALCUL IMPOSSIBLE", "mut")}
MODES = {K.EUROCODE: "Eurocode (ELU)", K.ELS: "Méthode ELS du bureau"}
MODES_PHRASE = {K.EUROCODE: "l'Eurocode (efforts de calcul, ELU)", K.ELS: "la méthode ELS du bureau (efforts de service)"}
LEGENDE = ("Dessins : pièces au trait fort, sections hachurées, cordons en bande ocre à leur taille a·√2 (symbole EN 22553), "
           "◉ point gouvernant et son taux, flèches rouges dans le sens des efforts saisis (⊙ vers soi, ⊗ vers le fond, ⇒ vecteur moment), cotes en mm. "
           "Répartition élastique des six composantes sur les gorges ; f évaluée au début, au milieu et à la fin de la longueur efficace de chaque cordon.")
NOTATIONS = ("f : force par unité de longueur sur le cordon (N/mm) · a : gorge · l_eff : longueur efficace (l − 2a sans retours d'angle) · "
             "σ⊥, τ⊥, τ∥ : composantes sur la gorge (EN 1993-1-8 fig. 4.5) · σ_c = √(σ⊥² + 3(τ⊥² + τ∥²)) · η : taux · "
             "k : kg par mm de gorge et par cm de cordon (méthode ELS du bureau).")
VARIANTES = ((1.0, "normale"), (0.92, "serrée"))
N_POINTS_P1 = 14


def _fmt(x, nd=1) -> str:
    if x is None:
        return "—"
    if isinstance(x, float) and (math.isinf(x) or math.isnan(x)):
        return "∞"
    return f"{x:,.{nd}f}".replace(",", " ").replace(".", ",")


def _g(v) -> str:
    """Un nombre « tel quel » (sans zéros inutiles), virgule décimale."""
    return f"{float(v):g}".replace(".", ",")


def _pct(t) -> str:
    if t is None:
        return "—"
    return f"{t * 100:.0f} %"


def _tr(p: Dict, cle: str = "troncons") -> str:
    t = p.get(cle) or {}
    n = int(t.get("n", 0) or 0)
    return f"par {n} tronçon{'s' if n > 1 else ''} de {_g(float(t.get('l', 0)))} mm" if n >= 1 else "continu"


def _jonction(p: Dict, cle: str = "jonction", cle_a: str = "a", cle_p: str = "p", cle_r: str = "renfort", cle_tr: str = "troncons") -> str:
    """Le type de jonction en clair : « cordons d'angle a 5 continu », « pénétration partielle p 6 + renfort a_r 3 … »,
    « pleine pénétration (la soudure vaut la pièce) »."""
    jon = p.get(cle, "angle")
    if jon == "pleine":
        return "pleine pénétration (la soudure vaut la pièce)"
    if jon == "partielle":
        r = float(p.get(cle_r, 0) or 0)
        return f"pénétration partielle p {_g(p.get(cle_p, 0))}" + (f" + renfort a_r {_g(r)} (gorge {_g(float(p.get(cle_p, 0)) + r)})" if r > 0 else "") + f" {_tr(p, cle_tr)}"
    return f"cordons d'angle a {_g(p[cle_a])} {_tr(p, cle_tr)}"


class Calcul:
    """Tout ce que la note imprime, calculé une fois : moteur, dessins, textes."""

    def __init__(self, etat: Dict, cas: Dict):
        self.etat, self.cas = etat, cas
        self.res = ET.verifier(cas)
        self.vues = SC.vues(self.res, SC.options_schemas(lvl=1))      # les scènes du kit, cotations principales
        self.svg_gorge = DS.encart_gorge(self.res)
        self.acier = self.res["acier"]
        self.meth = self.res["methode"]

    # --- lignes de données de l'en-tête
    def ligne_configuration(self) -> str:
        p, cfg = self.cas["params"], self.cas["config"]
        lib = CF.CONFIGS[cfg]["libelle"]
        if cfg == "C1":
            nf = int(p.get("frontaux", 1))
            s = (f"{lib} : plat B {_g(p['b'])} × {_g(p['tB'])} sur plat A t {_g(p['tA'])}, recouvrement {_g(p['L_rec'])} mm, "
                 f"{'cordons latéraux ' + _tr(p) if p.get('lateraux', True) else 'sans cordons latéraux'}"
                 f"{', ' + str(nf) + ' frontal' + ('s' if nf > 1 else '') if nf else ''}, a {_g(p['a'])}")
        elif cfg == "C2":
            s = (f"{lib} : pièce debout t {_g(p['tB'])} sur support t {_g(p['tA'])}, jonction l {_g(p['l'])} mm, "
                 f"{'deux faces' if p.get('deux_cordons', True) else 'une face'}, {_jonction(p)}")
        elif cfg == "C3":
            pot = p.get("poteau") or "plat"
            support = "un plat" if pot == "plat" else f"l'aile d'un {pot}"
            s = (f"{lib} : plat {_g(p['h'])} × {_g(p['t'])} sur {support}, V à e {_g(p['e'])} mm de la face, "
                 f"{'deux faces' if p.get('deux_cordons', True) else 'une face'}, {_jonction(p)}")
        elif cfg == "C4":
            s = (f"{lib} : plat {_g(p['bx'])} × {_g(p['by'])} × {_g(p['t'])}, côtés soudés {', '.join(p.get('cotes', []))}, a {_g(p['a'])} {_tr(p)}, "
                 f"effort en ({_g(p.get('xV', 0))} ; {_g(p.get('yV', 0))})")
        elif cfg == "C5":
            s = (f"{lib} : {p['profil']} sur platine t {_g(p.get('tp', 15))}, semelles {p.get('semelles', 'deux faces')} "
                 f"{_jonction(p, 'jonction_semelles', 'a_f', 'p_f', 'renfort_f', 'troncons_semelles')}, "
                 f"âme {_jonction(p, 'jonction_ame', 'a_w', 'p_w', 'renfort_w', 'troncons_ame') if p.get('ame', True) else 'non soudée'}")
        elif cfg == "C6":
            s = f"{lib} : {p['profil']} sur platine t {_g(p.get('tp', 15))}, pourtour {_jonction(p)}"
        elif cfg == "C7":
            s = (f"{lib} : plats {_g(p['b'])} × {_g(p['t'])}, " + ("pleine pénétration" if p.get("penetration", "pleine") == "pleine"
                 else f"pénétration partielle p {_g(p['p'])} {'des deux faces' if p.get('deux_faces', True) else 'd une face'}"))
        else:
            els = CF.elements_c8(p)
            n_s = sum(1 for e in els if e["type"] == "segment")
            s = (f"{lib} : {len(els)} élément(s) ({n_s} segment(s), {len(els) - n_s} arc(s)), "
                 f"référence R ({_g(p.get('xR', 0))} ; {_g(p.get('yR', 0))} ; {_g(p.get('zR', 0))})")
        ret = " · retours d'angle exécutés" if p.get("retours") and cfg not in ("C6", "C7", "C8") else ""
        return (s + ret + f" · acier A {p.get('nuance_A', 'S235')} / B {p.get('nuance_B', 'S235')} → référence {self.acier['nuance_ref']} "
                f"(fu {_g(self.acier['fu'])}, βw {_g(self.acier['bw'])})")

    def ligne_efforts(self) -> str:
        e = self.cas["efforts"]
        morceaux = [f"{lib.split(' (')[0]} = {_fmt(float(e.get(cle, 0) or 0), 1)} {unite}" for cle, lib, unite, _ in CF.EFFORTS[self.cas["config"]]]
        nature = "efforts de service (sans coefficient)" if self.meth.mode == K.ELS else "efforts de calcul (ELU)"
        return "Efforts — " + ", ".join(morceaux) + f" — {nature}, dans le sens du dessin"

    def ligne_methode(self) -> str:
        m, r = self.meth, self.res.get("resistances") or {}
        if m.mode == K.ELS:
            kq = self.res.get("k_equivalent") or {}
            return (f"Méthode ELS du bureau : résultante |f| ≤ k·a avec k = {_g(m.k)} kg/(mm·cm) = {_fmt(K.tau_adm_bureau(m.k, m.g))} N/mm² sur la gorge "
                    f"(g = {_g(m.g)} N/kg) · équivalent Eurocode au même niveau de service f_vw,d/γ_F = {_fmt(kq.get('tau_N_mm2', 0))} N/mm² = "
                    f"{kq.get('k_kg_mm_cm', 0):.0f} kg/(mm·cm) (γ_F = {_g(m.gamma_F)})")
        crit = "directionnelle (§4.5.3.2)" if m.ec == K.DIRECTIONNELLE else "simplifiée (§4.5.3.3)"
        return (f"Eurocode EN 1993-1-8 §4.5.3, méthode {crit}, γM2 = {_g(m.gamma_M2)} : fu/(βw·γM2) = {_fmt(r.get('lim_c', 0))}, "
                f"0,9·fu/γM2 = {_fmt(r.get('lim_perp', 0))}, f_vw,d = {_fmt(r.get('f_vw_d', 0))} N/mm²" if r else
                f"Eurocode EN 1993-1-8 §4.5.3, méthode {crit}, γM2 = {_g(m.gamma_M2)}")

    def formules(self) -> Dict:
        """Les substitutions de la méthode CHOISIE au point gouvernant (``formules.py``) : groupe, point, critère."""
        return FO.toutes(self.res)

    def methode_point(self) -> str:
        g = self.res.get("gouvernant") or {}
        return FO.methode_du_point(self.res, g) if g else ("els" if self.meth.mode == K.ELS else self.meth.ec)


class Note:
    """Page 1 · CALCUL."""

    def __init__(self, calc: Calcul, doc_meta: Dict, k: float = 1.0):
        self.C, self.doc, self.k = calc, doc_meta, k
        self.S = Encre()
        self.warnings: List[str] = []
        self.s_tab = 6.4 * k
        self.s_hyp = 6.1 * k
        self.lead = LEAD_TAB * k
        self.echelle = None
        self.points_montres = 0

    # --- briques
    def titre(self, d, fr, txt):
        S = self.S
        fr.down(10)
        d.t(fr.x, fr.y, d.fit(txt, S.f_v, 6.2, fr.w - 1.3 * len(txt)), S.f_v, 6.2, S.acc, track=1.3)
        fr.down(3)
        d.line(fr.x, fr.y, fr.x1, fr.y, mix(S.rule, S.ink, 0.12), 0.5)

    def table(self, d, fr, titre, cols, lignes, note=None):
        """``cols`` : (entête, largeur, alignement) ; ``lignes`` : cellules (style, texte[, ok[, gras]]) —
        styles lab (retour à la ligne), ref (petit, gris), num (droite), eta (droite, coloré selon ok)."""
        S, st, lead = self.S, self.s_tab, self.lead
        self.titre(d, fr, titre)
        fr.down(8)
        xs, x = [], fr.x
        for ent, w, al in cols:
            xs.append(x)
            d.t(x + (w - 3 if al == "right" else 0), fr.y, ent, S.f_b, S_TAB_H, S.mut, al)
            x += w
        fr.down(2.5)
        d.line(fr.x, fr.y, fr.x1, fr.y, S.acc, 0.5)
        for cells in lignes:
            rendus, n_max = [], 1
            for (ent, w, al), cell in zip(cols, cells):
                style, txt = cell[0], cell[1]
                if style in ("lab", "ref"):
                    size = st * 0.87 if style == "ref" else st
                    ls = d.wrap(txt, S.f_b, size, w - 4)
                    n_max = max(n_max, len(ls))
                    rendus.append((style, ls, cell))
                else:
                    rendus.append((style, [txt], cell))
            h = lead * n_max + 1.2
            fr.down(lead - 1.8)
            y = fr.y
            for (ent, w, al), (style, ls, cell), xx in zip(cols, rendus, xs):
                if style == "lab":
                    f = "Carlito-Bold" if (len(cell) > 2 and cell[2]) else S.f_b
                    for i, l in enumerate(ls):
                        d.t(xx, y - i * lead, l, f, st, S.ink)
                elif style == "ref":
                    for i, l in enumerate(ls):
                        d.t(xx, y - i * lead, l, S.f_b, st * 0.87, S.mut)
                elif style == "num":
                    gras = len(cell) > 2 and cell[2]
                    d.t(xx + w - 3, y, ls[0], "Carlito-Bold" if gras else S.f_b, st, S.ink, "right")
                elif style == "eta":
                    ok = cell[2]
                    gras = len(cell) > 3 and cell[3]
                    d.t(xx + w - 3, y, ls[0], "Carlito-Bold" if (gras or not ok) else S.f_b, st, S.ok if ok else S.ko, "right")
            fr.down(h - (lead - 1.8))
            d.line(fr.x, fr.y + 1.5, fr.x1, fr.y + 1.5, mix(S.rule, S.ink, 0.06), 0.4)
        if note:
            fr.down(st * 1.3)
            d.t(fr.x, fr.y, d.fit(note, S.f_b, st * 0.9, fr.w), S.f_b, st * 0.9, S.mut)
        fr.down(2)

    # --- bandeau
    def bandeau(self, d):
        S, C, doc = self.S, self.C, self.doc
        res, cas = C.res, C.cas
        W, H = d.W, d.H
        y = H - M - 4
        d.t(M, y, doc.get("bureau", ""), S.f_v, 8.0, S.acc, track=1.4)
        d.t(W - M, y, "Note de calcul · " + doc.get("date", "") + " · indice " + str(doc.get("indice", "")), S.f_b, 8, S.mut, "right")
        y -= 15
        d.t(M, y, f"{TITRE} — {cas['config']} · {CF.CONFIGS[cas['config']]['libelle']}", S.f_h, 12.5, S.ink)
        ident = " · ".join(str(x) for x in (doc.get("projet"), doc.get("partie"), "cas " + cas["nom"]) if x)
        d.t(W - M, y, d.fit(ident, S.f_b, 8, 300), S.f_b, 8, S.ink, "right")
        y -= 8
        d.line(M, y, W - M, y, S.ink, 1.0)
        y -= 15
        libelle, cle = STATUTS[res["statut"]]
        col = {"ok": S.ok, "att": S.att, "ko": S.ko, "mut": S.mut}[cle]
        w = chip(d, M, y - 3.2, libelle, S.f_v, 6.6, "#FFFFFF", col)
        x = M + w + 8
        if res["statut"] == ET.REFUS:
            d.t(x, y, d.fit((res["messages"] or ["—"])[0], S.f_b, 8, W - M - x), S.f_b, 8, S.ink)
        else:
            g = res["gouvernant"]
            t1 = "η max " + _pct(res["taux_max"])
            d.t(x, y, t1, "Carlito-Bold", 8.6, col)
            x += d.w(t1, "Carlito-Bold", 8.6) + 5
            ch = next((c["chaine"] for c in res["cordons"] if c["nom"] == g["cordon"]), g["cordon"])
            t2 = f"— gouverne : {ch}, {g['position']} · {g['critere']}"
            t3 = FO.TITRES[C.methode_point()]
            wt = d.w(t3, "Carlito-Bold", 7.6)
            xr = W - M - wt
            d.t(x, y, d.fit(t2, S.f_b, 8, xr - x - 6), S.f_b, 8, S.ink)
            d.t(xr, y, t3, "Carlito-Bold", 7.6, S.acc)
        y -= 12
        d.t(M, y, d.fit(C.ligne_configuration(), S.f_b, S_DATA, W - 2 * M), S.f_b, S_DATA, S.ink)
        y -= S_DATA * 1.4
        d.t(M, y, d.fit(C.ligne_efforts(), S.f_b, S_DATA, W - 2 * M), S.f_b, S_DATA, S.ink)
        y -= S_DATA * 1.4
        d.t(M, y, d.fit(C.ligne_methode(), S.f_b, S_DATA, W - 2 * M), S.f_b, S_DATA, S.ink)
        y -= 6
        d.line(M, y, W - M, y, mix(S.rule, S.ink, 0.3), 0.6)
        return y - 4

    # --- colonne 1 : dessin, gorge, hypothèses
    def colonne_dessin(self, d, fr):
        S, C = self.S, self.C
        lignes = []
        textes = [m for m in C.res.get("messages", []) if not m.startswith("Refus")]
        textes.append("Longueur efficace l_eff = l − 2a sans retours d'angle (§4.5.1) ; un cordon ne compte que si l_eff ≥ max(30 mm ; 6a) et a ≥ 3 mm (§4.5.2).")
        if C.res.get("beta_Lw", 1.0) >= 1.0 and C.cas["config"] == "C1":
            textes.append("Assemblage par recouvrement : βLw = 1 (L_j ≤ 150·a).")
        for h in textes:
            lignes.extend(d.wrap("• " + h, S.f_b, self.s_hyp, fr.w))
        h_hyp = 13 + len(lignes) * self.s_hyp * 1.3 + 4
        h_gorge = 132
        dispo = fr.room() - h_hyp - 13 - 13 - 6
        vues = C.vues
        self.titre(d, fr, "DESSINS — " + (" · ".join(t.lower() for _, t, _ in vues) if vues else "calcul impossible"))
        if vues:
            # la vue principale sur la largeur, les autres côte à côte dessous ; hauteurs au prorata des viewBox, bornées
            _, _, d0 = vues[0]
            h0 = min(max(90.0, fr.w * d0.viewbox[3] / d0.viewbox[2]), 0.62 * (dispo - h_gorge))
            fr.down(h0)
            self.echelle = peindre_scene(d, d0, fr.x, fr.y, fr.w, h0 - 2)
            if len(vues) > 1:
                n = len(vues) - 1
                wv = (fr.w - 4 * (n - 1)) / n
                h1 = min(max(60.0, max(wv * dv.viewbox[3] / dv.viewbox[2] for _, _, dv in vues[1:])), dispo - h_gorge - h0 - 6)
                fr.down(h1 + 3)
                for i, (_, _, dv) in enumerate(vues[1:]):
                    peindre_scene(d, dv, fr.x + i * (wv + 4), fr.y, wv, h1 - 2)
        else:
            fr.down(20)
            d.t(fr.x, fr.y + 6, (C.res.get("messages") or ["Calcul impossible"])[0][:90], S.f_b, self.s_hyp, S.ko)
        self.titre(d, fr, "LA GORGE AU POINT GOUVERNANT — EN 1993-1-8 fig. 4.5")
        fr.down(h_gorge)
        peindre_svg(d.c, C.svg_gorge, fr.x + (fr.w - h_gorge * 320.0 / 330.0) / 2, fr.y, h_gorge * 320.0 / 330.0, h_gorge - 2)
        self.titre(d, fr, "HYPOTHÈSES")
        fr.down(3)
        for l in lignes:
            fr.down(self.s_hyp * 1.3)
            d.t(fr.x, fr.y, l, S.f_b, self.s_hyp, S.mut)

    # --- colonne 2 : cordons, efforts, groupe
    def table_cordons(self, d, fr):
        C = self.C
        res = C.res
        rows = []
        g = res.get("gouvernant") or {}
        g_ch = next((c["chaine"] for c in res.get("cordons", []) if c["nom"] == g.get("cordon")), None)
        chaines = sorted(res.get("chaines", []), key=lambda c: -c["taux"])
        montres = chaines[:16]
        for c in montres:
            gouv = c["nom"] == g_ch
            rows.append([("lab", c["nom"] + (" (pleine pén.)" if c.get("pleine") else ""), gouv), ("num", _fmt(c["a"], 1)),
                         ("num", _fmt(c["l_eff"], 0)), ("num", _fmt(c["f_max"], 0)), ("eta", _pct(c["taux"]), c["taux"] <= 1.0, gouv), ("ref", c["position"])])
        for c in res.get("cordons_exclus", [])[:4]:
            rows.append([("ref", c["chaine"]), ("num", _fmt(c["a"], 1)), ("num", "—"), ("num", "—"), ("ref", "exclu"), ("ref", "ne compte pas")])
        if not rows:
            rows = [[("ref", "aucun cordon utile"), ("ref", ""), ("ref", ""), ("ref", ""), ("ref", ""), ("ref", "")]]
        note = None
        if len(chaines) > 16:
            reste = chaines[16:]
            note = f"+ {len(reste)} cordon(s) non montrés, tous ≤ {_pct(max(c['taux'] for c in reste))} (carte complète en page 2)"
        w = fr.w
        cols = [("CORDON", w - 26 - 30 - 36 - 28 - 34, "left"), ("a", 26, "right"), ("l_eff", 30, "right"), ("|f| N/mm", 36, "right"),
                ("η", 28, "right"), ("POINT", 34, "left")]
        self.table(d, fr, "CORDONS — gorge, longueur efficace, force maximale, taux", cols, rows, note)

    def table_efforts(self, d, fr):
        C = self.C
        e = C.cas["efforts"]
        rows = []
        for cle, lib, unite, aide in CF.EFFORTS[C.cas["config"]]:
            v = float(e.get(cle, 0) or 0)
            rows.append([("lab", lib.split(" (")[0], abs(v) > 0), ("num", _fmt(v, 1) + " " + unite), ("ref", aide)])
        w = fr.w
        cols = [("EFFORT", 34, "left"), ("VALEUR", 46, "right"), ("SENS", w - 80, "left")]
        nature = "service" if C.meth.mode == K.ELS else "calcul (ELU)"
        self.table(d, fr, f"EFFORTS — valeurs de {nature}, dans le sens du dessin", cols, rows)
        pr = C.res.get("proprietes") or {}
        if pr:
            ef = C.res.get("efforts")
            txt = (f"Groupe : A_w {_fmt(pr.get('A_w', 0), 0)} mm², G ({_fmt(pr.get('xG', 0), 1)} ; {_fmt(pr.get('yG', 0), 1)}), "
                   f"I_x {_fmt(pr.get('Ix', 0) / 1e4, 1)}, I_y {_fmt(pr.get('Iy', 0) / 1e4, 1)}, I_xy {_fmt(pr.get('Ixy', 0) / 1e4, 1)}, "
                   f"J {_fmt(pr.get('J', 0) / 1e4, 1)} cm⁴ · βLw {_fmt(C.res.get('beta_Lw', 1.0), 3)}")
            if ef is not None:
                txt += (f" · au point de référence ({_fmt(ef.reference[0], 0)} ; {_fmt(ef.reference[1], 0)} ; {_fmt(ef.reference[2], 0)}) : "
                        f"V_x {_fmt(ef.Vx / 1e3)} kN, V_y {_fmt(ef.Vy / 1e3)} kN, N {_fmt(ef.N / 1e3)} kN, M_x {_fmt(ef.Mx / 1e6, 2)}, "
                        f"M_y {_fmt(ef.My / 1e6, 2)}, M_z {_fmt(ef.Mz / 1e6, 2)} kN·m")
            for l in d.wrap(txt, self.S.f_b, self.s_hyp, fr.w):
                fr.down(self.s_hyp * 1.3)
                d.t(fr.x, fr.y, l, self.S.f_b, self.s_hyp, self.S.mut)

    # --- colonne 3 : vérifications par point, comparaison, conclusion
    def lignes_points(self, tous: bool = False):
        """Les points à imprimer : tous s'ils tiennent (≤ 14 en page 1, ≤ 36 en page 2), sinon le
        pire point de chaque cordon (un pourtour ou un tronçon courbe est découpé en segments de
        calcul : trois points par segment n'apprennent rien de plus que le pire du cordon)."""
        res = self.C.res
        pts = sorted(res.get("points", []), key=lambda p: -p["taux"])
        chaines = {c["nom"]: c["chaine"] for c in res.get("cordons", [])}
        limite = 36 if tous else N_POINTS_P1
        if len(pts) <= limite:
            return pts, chaines, ""
        pires, vus = [], set()
        for p in pts:
            ch = chaines.get(p["cordon"], p["cordon"])
            if ch not in vus:
                vus.add(ch)
                pires.append(p)
        if tous:
            return pires[:40], chaines, f"le pire point de chacun des {len(pires)} cordons ({len(pts)} points calculés)"
        if len(pires) <= limite:
            return pires, chaines, f"le pire point de chaque cordon ({len(pts)} points calculés) — détail en page 2"
        return pires[:limite], chaines, f"les {limite} cordons les plus chargés sur {len(pires)} ({len(pts)} points calculés) — suite en page 2"

    def table_points(self, d, fr, tous: bool = False):
        C, S = self.C, self.S
        res = C.res
        if not res.get("points"):
            return
        g = res["gouvernant"]
        montres, chaines, note = self.lignes_points(tous)
        self.points_montres = len(montres)
        rows = []
        for p in montres:
            gouv = p is g
            rows.append([("lab", chaines.get(p["cordon"], p["cordon"]), gouv), ("ref", p["position"]), ("num", _fmt(p["f"], 0)),
                         ("num", _fmt(p["sigma_perp"], 0)), ("num", _fmt(p["tau_perp"], 0)), ("num", _fmt(p["tau_par"], 0)),
                         ("num", _fmt(p["sigma_c"], 0)), ("eta", _pct(p["taux"]), p["taux"] <= 1.0, gouv)])
        w = fr.w
        cols = [("CORDON", w - 30 - 30 - 26 - 26 - 26 - 30 - 28, "left"), ("POINT", 30, "left"), ("|f|", 30, "right"), ("σ⊥", 26, "right"),
                ("τ⊥", 26, "right"), ("τ∥", 26, "right"), ("σ_c", 30, "right"), ("η", 28, "right")]
        self.table(d, fr, "VÉRIFICATIONS PAR POINT — N/mm et N/mm², taux selon la méthode choisie", cols, rows, note or None)

    def verification(self, d, fr):
        """La vérification au point gouvernant, écrite comme à la main : le critère de la
        méthode choisie en typographie mathématique, puis les substitutions numériques
        (groupe, point, critère) — chacune testée contre le moteur (``formules.py``)."""
        C, S = self.C, self.S
        fo = C.formules()
        if not fo["critere"]:
            return
        cle = C.methode_point()
        self.titre(d, fr, "VÉRIFICATION AU POINT GOUVERNANT — " + FO.TITRES[cle].split(" (")[0])
        fr.down(2)
        formula(d, fr, "", FO.CRITERES_MATH[cle], S.ms(), 7.6 * self.k, S.mut, S.f_b, 5.8, lead=1.0, min_size=5.5)
        for titre, cles in (("Groupe", "groupe"), ("Point gouvernant", "point"), ("Critère", "critere")):
            subs = fo[cles]
            if not subs:
                continue
            fr.down(self.s_hyp * 1.5)
            d.t(fr.x, fr.y, titre.upper(), S.f_v, 5.4, S.mut, track=1.1)
            for sub in subs:
                for l in d.wrap(FO.rendre(sub), S.f_b, self.s_hyp, fr.w - 4):
                    fr.down(self.s_hyp * 1.3)
                    d.t(fr.x + 4, fr.y, l, S.f_b, self.s_hyp, S.ink)
        fr.down(2)

    def conclusion(self, d, x, w):
        S, C = self.S, self.C
        res = C.res
        libelle, cle = STATUTS[res["statut"]]
        col = {"ok": S.ok, "att": S.att, "ko": S.ko, "mut": S.mut}[cle]
        if res["statut"] == ET.REFUS:
            txt = (res["messages"] or ["Calcul impossible."])[0]
        else:
            g = res["gouvernant"]
            ch = next((c["chaine"] for c in res["cordons"] if c["nom"] == g["cordon"]), g["cordon"])
            ou = f"{ch}, {g['position']}" + (" (pleine pénétration : la pièce gouverne)" if g.get("pleine") else "")
            if g.get("pleine"):
                porte = ("La soudure vaut la pièce : sa section porte les efforts saisis." if res["statut"] != ET.NOK else
                         "La section de la pièce ne porte pas les efforts saisis : épaisseur, longueur ou réduction d'effort à étudier.")
            else:
                porte = ("Les cordons portent les efforts saisis." if res["statut"] != ET.NOK else
                         "Les cordons ne portent pas les efforts saisis en l'état : gorge, longueur, cordons supplémentaires ou réduction d'effort à étudier.")
            txt = f"Taux maximal {_pct(res['taux_max'])} sur {ou}, selon {MODES_PHRASE[C.meth.mode]}. " + porte
            if any("DÉPASSÉ" in m for m in res.get("messages", [])):
                txt += " L'espacement des tronçons dépasse la limite de l'EN 1993-1-8 §4.3.2.2 : à revoir."
        lignes = d.wrap(txt, S.f_b, 6.6, w - 16)
        h = 13 + 9 + len(lignes) * 6.6 * 1.3 + 8
        y0 = BAS
        d.box(x, y0, w, h, fill=mix(S.panel, "#FFFFFF", 0.3), stroke=col, lw=0.8, r=3)
        d.t(x + 8, y0 + h - 11, "CONCLUSION", S.f_v, 6.2, col, track=1.3)
        yy = y0 + h - 11 - 9
        for l in lignes:
            d.t(x + 8, yy, l, S.f_b, 6.6, S.ink)
            yy -= 6.6 * 1.3
        return h

    def construire(self, d):
        S = self.S
        W = d.W
        top = self.bandeau(d)
        w1, w2 = LARG
        w3 = W - 2 * M - w1 - w2 - 2 * GAP
        x1, x2, x3 = M, M + w1 + GAP, M + w1 + GAP + w2 + GAP
        for x in (x2, x3):
            d.line(x - GAP / 2, top, x - GAP / 2, BAS, S.rule, 0.45)
        f1, f2, f3 = Frame(x1, top, w1, BAS), Frame(x2, top, w2, BAS), Frame(x3, top, w3, BAS)
        self.colonne_dessin(d, f1)
        self.table_cordons(d, f2)
        self.table_efforts(d, f2)
        hc = self.conclusion(d, x3, w3)
        f3.bottom = BAS + hc + 6
        self.table_points(d, f3)
        self.verification(d, f3)
        d.t(M, 24, d.fit(LEGENDE, S.f_b, 5.6, W - 2 * M), S.f_b, 5.6, S.mut)
        d.t(M, 16.5, d.fit(NOTATIONS, S.f_b, 5.6, W - 2 * M), S.f_b, 5.6, S.mut)
        for f in (f1, f2, f3):
            if f.overflow() > 1.0:
                self.warnings.append(f"page 1, colonne {f.x:.0f} : {f.overflow():.0f} pt")
        return self.warnings


CART_H = 40


class Methode:
    """Page 2 · MÉTHODE ET DÉTAIL."""

    def __init__(self, calc: Calcul, doc_meta: Dict, k: float = 1.0, echelle: Optional[float] = None, points_p1: int = 0):
        self.C, self.doc, self.k = calc, doc_meta, k
        self.S = Encre()
        self.warnings: List[str] = []
        self.s = 6.3 * k
        self.echelle = echelle
        self.points_p1 = points_p1
        self.note1 = Note(calc, doc_meta, k)

    def titre(self, d, fr, txt):
        S = self.S
        fr.down(11)
        d.t(fr.x, fr.y, d.fit(txt, S.f_v, 6.2, fr.w - 1.3 * len(txt)), S.f_v, 6.2, S.acc, track=1.3)
        fr.down(3)
        d.line(fr.x, fr.y, fr.x1, fr.y, mix(S.rule, S.ink, 0.12), 0.5)
        fr.down(2)

    def paragraphe(self, d, fr, txt, col=None, gras=False):
        S = self.S
        for l in d.wrap(txt, "Carlito-Bold" if gras else S.f_b, self.s, fr.w):
            fr.down(self.s * 1.32)
            d.t(fr.x, fr.y, l, "Carlito-Bold" if gras else S.f_b, self.s, col or S.ink)
        fr.down(self.s * 0.5)

    def section(self, d, fr, sec):
        self.titre(d, fr, sec["titre"].upper())
        for p in sec["paragraphes"]:
            self.paragraphe(d, fr, p)

    def carte_taux(self, d, fr):
        C, S = self.C, self.S
        ch = sorted(C.res.get("chaines", []), key=lambda c: -c["taux"])
        if not ch:
            return
        self.titre(d, fr, "CARTE DES TAUX — tous les cordons")
        larg = fr.w
        for c in ch:
            fr.down(self.s * 1.45)
            col = S.ok if c["taux"] <= 0.7 else (S.att if c["taux"] <= 1.0 else S.ko)
            d.t(fr.x, fr.y, d.fit(f"{c['nom']} · a{_g(c['a'])} · {_fmt(c['l_eff'], 0)} mm", S.f_b, self.s * 0.95, larg * 0.52), S.f_b, self.s * 0.95, S.ink)
            x0 = fr.x + larg * 0.55
            wb = larg * 0.33
            d.box(x0, fr.y - 0.5, wb, 3.6, fill=mix(S.rule, "#FFFFFF", 0.4))
            d.box(x0, fr.y - 0.5, wb * min(1.0, c["taux"]), 3.6, fill=col)
            d.t(fr.x + larg, fr.y, _pct(c["taux"]), "Carlito-Bold", self.s * 0.95, col, "right")
        fr.down(3)

    def cartouche(self, d):
        S, C, doc = self.S, self.C, self.doc
        W = d.W
        x0, y0, w, h = M, M, W - 2 * M, CART_H
        d.box(x0, y0, w, h, stroke=S.ink, lw=0.9)
        cas = C.cas
        ech = "—"
        if self.echelle:
            ech = "1:" + f"{2.8346 / self.echelle:.0f}"
        ident = " · ".join(str(x) for x in (doc.get("projet"), doc.get("partie")) if x)
        m = C.meth
        meth = MODES[m.mode] + (f" · k = {_g(m.k)} kg/(mm·cm)" if m.mode == K.ELS else (" · directionnelle" if m.ec == K.DIRECTIONNELLE else " · simplifiée"))
        cases = [("CAS", cas["nom"], ident or "—", 0.18),
                 ("CONFIGURATION", f"{cas['config']} · {CF.CONFIGS[cas['config']]['libelle']}", d.fit(C.ligne_configuration().split(" : ", 1)[-1], S.f_b, 6.2, w * 0.26 - 12), 0.28),
                 ("MÉTHODE", meth, f"γM2 {_g(m.gamma_M2)} · g {_g(m.g)} · γ_F {_g(m.gamma_F)}", 0.20),
                 ("ACIER", f"référence {C.acier['nuance_ref']}", f"fu {_g(C.acier['fu'])} N/mm² · βw {_g(C.acier['bw'])}", 0.12),
                 ("DATE · INDICE", str(doc.get("date", "")), "indice " + str(doc.get("indice", "")), 0.11),
                 ("ÉCHELLE DU DESSIN", ech, "A4 paysage", 0.11)]
        x = x0
        for i, (lab, l1, l2, part) in enumerate(cases):
            cw = w * part
            if i:
                d.line(x, y0 + 2, x, y0 + h - 2, S.rule, 0.6)
            d.t(x + 6, y0 + h - 10, lab, S.f_v, 5.4, S.mut, track=1.2)
            d.t(x + 6, y0 + h - 21, d.fit(str(l1), "Carlito-Bold", 6.6, cw - 12), "Carlito-Bold", 6.6, S.ink)
            if l2:
                d.t(x + 6, y0 + h - 32, d.fit(str(l2), S.f_b, 6.2, cw - 12), S.f_b, 6.2, S.ink)
            x += cw

    def construire(self, d):
        S = self.S
        W, H = d.W, d.H
        y = H - M - 4
        d.t(M, y, self.doc.get("bureau", ""), S.f_v, 8.0, S.acc, track=1.4)
        d.t(W - M, y, TITRE + " · cas " + self.C.cas["nom"], S.f_b, 7.6, S.mut, "right")
        y -= 16
        d.t(M, y, "MÉTHODE ET DÉTAIL", S.f_h, 13.5, S.ink)
        d.t(W - M, y, d.fit("Références : " + " · ".join(ME.REFERENCES), S.f_b, 6.2, W - 2 * M - 150), S.f_b, 6.2, S.mut, "right")
        y -= 7
        d.line(M, y, W - M, y, S.ink, 1.0)
        top = y - 6
        bas = M + CART_H + 10
        wc = (W - 2 * M - 2 * GAP) / 3.0
        frames = [Frame(M + i * (wc + GAP), top, wc, bas) for i in range(3)]
        for i in (1, 2):
            d.line(frames[i].x - GAP / 2, top, frames[i].x - GAP / 2, bas, S.rule, 0.45)
        # la méthode CHOISIE seulement : le texte de l'autre méthode n'est pas imprimé
        els = self.C.meth.mode == K.ELS
        secs = [s for s in ME.SECTIONS if not (s["titre"].startswith("Eurocode") and els) and not (s["titre"].startswith("Méthode ELS") and not els)]
        for sec in secs[:2]:
            self.section(d, frames[0], sec)
        for sec in secs[2:3]:
            self.section(d, frames[1], sec)
        for sec in secs[3:]:
            self.section(d, frames[2], sec)
        self.carte_taux(d, frames[2])
        pts = self.C.res.get("points") or []
        if len(pts) > self.points_p1:
            self.note1.s_tab, self.note1.lead = 5.6 * self.k, 7.6 * self.k
            self.note1.table_points(d, frames[2], tous=True)
        self.cartouche(d)
        for f in frames:
            if f.overflow() > 1.0:
                self.warnings.append(f"page 2, colonne {f.x:.0f} : {f.overflow():.0f} pt")
        return self.warnings


def _essai(calc, doc_meta, k, chemin):
    d = Doc(chemin, A4L, title=doc_meta.get("titre", "Note de calcul"))
    d.new_page(A4L)
    n1 = Note(calc, doc_meta, k)
    w = list(n1.construire(d))
    d.new_page(A4L)
    n2 = Methode(calc, doc_meta, k, n1.echelle, n1.points_montres)
    w += n2.construire(d)
    d.save()
    return w


def generer_note(etat: Dict, cas: Optional[Dict] = None, infos: Optional[Dict] = None, chemin: Optional[str] = None) -> bytes:
    """La note (bytes PDF) : page 1 de calcul, page 2 de méthode et détail.
    ``infos`` : nom_projet, partie, date, indice."""
    global derniers_avertissements, derniere_variante
    cas = cas or ET.cas_actif(etat)
    infos = infos or {}
    doc_meta = ndc_data.construire_doc(infos, date_defaut=datetime.today().strftime("%d/%m/%Y"))
    doc_meta["titre"] = "Note de calcul — soudures"
    calc = Calcul(etat, cas)
    if chemin is None:
        fd, chemin = tempfile.mkstemp(suffix=".pdf", prefix="note_soudures_")
        os.close(fd)
    warn, retenue = [], VARIANTES[-1][1]
    for k, nom in VARIANTES:
        warn = _essai(calc, doc_meta, k, chemin)
        retenue = nom
        if not warn:
            break
    derniers_avertissements = warn
    derniere_variante = retenue
    with open(chemin, "rb") as fh:
        return fh.read()


derniers_avertissements: List[str] = []
derniere_variante: Optional[str] = None

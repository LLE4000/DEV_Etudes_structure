# -*- coding: utf-8 -*-
"""Note de calcul « Garde-corps » (version 2) — DEUX pages A4 paysage, sur le kit ``ndc_pdf``
des notes de l'application (palette Encre, mêmes marges et mêmes polices que les notes
d'assemblage, de plancher et de soudures).

Page 1 · CALCUL, les HYPOTHÈSES d'abord et peu de prose :
- bandeau : statut, η max, l'élément gouvernant, son critère et son cas de charge, la méthode ;
- colonne 1 : HYPOTHÈSES (garde-corps, sections, acier et limites, fixation), le tableau des
  QUATRE CHARGES nommées avec leur origine et la règle des cas (distincts, ou cumul q_h + q_v),
  les DESSINS (élévation, coupe A-A, vue en plan) peints depuis les mêmes scènes que l'écran ;
- colonne 2 : le critère de la méthode en typographie mathématique, puis 1 · MONTANT PRINCIPAL
  (ou la console au mur) et 2 · MAIN COURANTE, chacun avec son cas de charge gouvernant, les
  autres cas et leur taux, et les formules DÉVELOPPÉES (substitutions numériques de
  ``formules.py``, testées contre le moteur), le RÉCAPITULATIF des éléments ;
- colonne 3 : 3 · BARREAUX (ou lisses intermédiaires, ou le panneau), 4 · ANCRAGE (soudure,
  chevilles ou boulons, platine) avec leurs formules, la CONCLUSION.

Page 2 · DÉTAILS ET MÉTHODE : la platine en plan et la coupe sur la fixation, le tableau des
MONTANTS (ou des consoles), la SOUDURE (vue et formules du module Soudures), la section de
l'élément principal et la BIBLIOTHÈQUE, les textes de MÉTHODE (``methode.py``) de la méthode
choisie, les références, le cartouche.

``derniers_avertissements`` : débordements de colonne (vide si tout tient) ; la note est
d'abord composée en taille normale, puis en taille réduite si une colonne déborde.
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
from acier.assemblages.poutre_poutre.doubles_cornieres.rapport import peindre as peindre_scene   # peintre des scènes du kit
from acier.soudures import formules as FOS
from acier.soudures import schemas as SCS

from . import bibliotheque as BI
from . import charges as CH
from . import etat as ET
from . import fixation as FX
from . import formules as FO
from . import membres as ME
from . import methode as MT
from . import schemas as SC
from . import types as TY

A4L = landscape(A4)
M, BAS, GAP = 22, 40, 12
LARG = (286, 246)
S_TAB_H, S_DATA = 5.6, 7.0
LEAD_TAB = 8.8
TITRE = "Garde-corps"
STATUTS = {ET.OK: ("CONFORME", "ok"), ET.ATTENTION: ("CONFORME · ATTENTION", "att"), ET.NOK: ("NON CONFORME", "ko"), ET.REFUS: ("CALCUL IMPOSSIBLE", "mut")}
MODES = {ME.ELS: "Méthode ELS du bureau", ME.EUROCODE: "Eurocode (ELU)"}
MODES_PHRASE = {ME.ELS: "la méthode ELS du bureau (charges de service, contraintes admissibles)", ME.EUROCODE: "l'Eurocode (efforts de calcul)"}
LEGENDE = ("Dessins à l'échelle, en mm : pièces au trait fort, sections coupées hachurées, cordons en ocre à leur taille a·√2, chevilles en gris, "
           "◉ l'élément gouvernant et son taux, flèches rouges dans le sens des charges (⊙ vers soi), A-A le plan de la coupe. Chaque montant reçoit "
           "la charge de sa longueur tributaire ; un montant d'angle, une demi-travée de chaque tronçon dans deux directions.")
NOTATIONS = ("P₁ : charge horizontale en tête du montant (q_h × s_trib) · N : charge verticale (q_v × s_trib) · M₁ = P₁·H · W_el : module élastique "
             "(h dans le sens de l'effort) · A_v : aire de cisaillement (EN 1993-1-1 §6.2.6, entière pour un plein) · f : flèche · η : taux · "
             "Z : traction de la rangée de chevilles tendue · m : bras de la platine en console.")
VARIANTES = ((1.0, "normale"), (0.92, "serrée"))
TITRES_ROLE = {"montant": "MONTANT PRINCIPAL", "console": "CONSOLE", "lisse": "MAIN COURANTE", "barreau": "BARREAUX", "lisse_inter": "LISSES INTERMÉDIAIRES"}


def _fmt(x, nd=1) -> str:
    if x is None:
        return "—"
    if isinstance(x, float) and (math.isinf(x) or math.isnan(x)):
        return "∞"
    return f"{x:,.{nd}f}".replace(",", " ").replace(".", ",")


def _g(v) -> str:
    return f"{float(v):g}".replace(".", ",")


def _pct(t) -> str:
    if t is None:
        return "—"
    return f"{t * 100:.0f} %"


class Calcul:
    """Tout ce que la note imprime, calculé une fois : moteur, dessins, textes."""

    def __init__(self, etat: Dict, cas: Dict):
        self.etat, self.cas = etat, ET.normaliser_cas(cas)
        self.res = ET.verifier(self.cas)
        vues = SC.vues(self.res, SC.options_schemas(lvl=1))
        self.vues = {cle: d for cle, _, d in vues}
        self.titres = {cle: t for cle, t, _ in vues}
        if "coupe" in self.vues:
            # la coupe A-A, à côté de l'élévation, reçoit une police plus grande : elle est peinte petite
            opt = SC.options_schemas(lvl=1)
            opt.fs_force = 44.0
            self.vues["coupe"] = SC.coupe(self.res, opt)
        self.meth = self.res["methode"]
        self.acier = self.res["acier"]
        self.fo = FO.toutes(self.res)
        s = (self.res.get("fixation") or {}).get("soudure")
        self.vues_soudure = SCS.vues(s["res"], SCS.options_schemas(lvl=1)) if s else []

    def systeme(self) -> str:
        return self.cas["geometrie"].get("systeme", "montants")

    def type_libelle(self) -> str:
        return TY.TYPES[TY.type_de(self.cas)]["libelle"]

    def sous_titre(self) -> str:
        return self.type_libelle().lower() + ("" if self.systeme() == "mur" else " sur montants")

    # --- les hypothèses, en lignes « libellé : texte »
    def hyp_garde_corps(self) -> str:
        geo = self.res["geometrie"]
        g = self.cas["geometrie"]
        s = f"{self.type_libelle()} · {geo['phrase']} · H = {_g(geo['H'])} mm" + (" (au-dessus de la platine)" if self.systeme() == "montants" else " (au-dessus du sol)")
        if self.systeme() == "mur":
            c = self.res.get("console") or {}
            s += f" · consoles à e = {_g(c.get('e', 0))} mm du mur"
        s += f" · portée de la main courante : {_g(geo['s_max'])} mm, {ME.APPUIS[self.cas['lisse'].get('appuis', 'deux')]}"
        return s

    def hyp_sections(self) -> str:
        res = self.res
        parts = []
        if self.systeme() == "montants":
            parts.append(f"montants {(res.get('section_montant') or {}).get('nom', '—')}")
        else:
            c = res.get("console") or {}
            parts.append(f"consoles {(c.get('section') or {}).get('nom', '—')}")
        li = (res.get("lisse") or {}).get("section") or {}
        parts.append(f"main courante {li.get('nom', '—')}")
        rp = res.get("remplissage") or {}
        if rp.get("type") == "barreaux" and rp.get("section"):
            parts.append(f"barreaux {rp['section']['nom']} à e_b = {_g(rp['entraxe'])} mm, portée {_g(round(rp['L']))} mm (déduite de H, h_bas = {_g(rp['h_bas'])} mm et des sections)")
            if rp.get("section_basse"):
                parts.append(f"lisse basse {rp['section_basse']['nom']} (non calculée)")
        elif rp.get("type") == "lisses" and rp.get("section"):
            parts.append(f"{rp.get('n', 1)} lisses intermédiaires {rp['section']['nom']} à e_l = {_g(rp['entraxe'])} mm")
        elif rp.get("type") == "panneau":
            parts.append("panneau (verre, tôle) justifié par son fournisseur" + (f" · lisse basse {rp['section_basse']['nom']}" if rp.get("section_basse") else ""))
        else:
            parts.append("sans remplissage")
        return " · ".join(parts)

    def hyp_methode(self) -> str:
        m, lim = self.meth, self.res["limites"]
        fl = f"L/{_g(m.get('fleche_val', 100))}" if m.get("fleche_mode", "L/x") == "L/x" else f"{_g(m.get('fleche_val', 0))} mm"
        if m["mode"] == ME.ELS:
            return (f"acier {self.acier['nuance']} · méthode ELS du bureau : charges de service sans coefficient, σ = M/W_el + N/A ≤ σ_adm = {_g(lim['sigma'])} N/mm², "
                    f"τ = V/A_v ≤ τ_adm = {_g(lim['tau'])} N/mm², flèche ≤ {fl} · soudure k = {_g(m.get('k', 100))} kg/(mm·cm) · résistances de calcul des ancrages / γ_F = {_g(m.get('gamma_F', 1.5))}")
        return (f"acier {self.acier['nuance']} (f_y = {_g(self.acier['fy'])}) · Eurocode : charges × γ_Q = {_g(m.get('gamma_Q', 1.5))}, σ ≤ f_y/γ_M0 = {_fmt(lim['sigma'])} N/mm² (élastique, W_el), "
                f"τ ≤ f_y/(√3·γ_M0) = {_fmt(lim['tau'])}, flèche ≤ {fl} sous charges de service · soudure EN 1993-1-8 {m.get('ec', 'directionnelle')}, γ_M2 = {_g(m.get('gamma_M2', 1.25))}")

    def hyp_fixation(self) -> str:
        fx = self.cas["fixation"]
        typ = fx.get("type", "platine")
        s = FX.TYPES[typ]
        if typ == "scellee":
            return s + " : la fixation n'est pas vérifiée"
        pl = fx["platine"]
        s += f" : soudure {FX.texte_soudure(fx['soudure'])}"
        if typ == "platine":
            s += (f" · platine {_g(pl['bp'])} × {_g(pl['lp'])} × {_g(pl['tp'])} mm · {FX.texte_ancrage(fx['ancrage'])}, rangées à e_x = {_g(fx['ancrage']['ex'])} mm, "
                  f"e_y = {_g(fx['ancrage']['ey'])} mm")
        return s

    def ligne_cas(self) -> str:
        ch = self.res["charges"]
        if ch.get("cumul"):
            return "Cas de charge : q_h et q_v agissent ensemble (cas 1 + 2, côté sûr) ; Q_r et q_r en cas séparés (3 et 4) ; chaque élément est vérifié sous chaque cas, le pire gouverne."
        return "Cas de charge : les quatre charges agissent SÉPARÉMENT (cas 1 à 4) — les charges de service d'un garde-corps ne se cumulent pas ; chaque élément est vérifié sous chaque cas, le pire gouverne."


class Note:
    """Page 1 · CALCUL."""

    def __init__(self, calc: Calcul, doc_meta: Dict, k: float = 1.0):
        self.C, self.doc, self.k = calc, doc_meta, k
        self.S = Encre()
        self.warnings: List[str] = []
        self.s_tab = 6.3 * k
        self.s_hyp = 6.1 * k
        self.lead = LEAD_TAB * k
        self.echelle = None

    # --- briques
    def titre(self, d, fr, txt):
        """Un titre de bloc en petites capitales espacées ; il se resserre avant de se tronquer."""
        S = self.S
        fr.down(10)
        size, track = 6.2, 1.3
        while size > 5.0 and d.w(txt, S.f_v, size) + track * len(txt) > fr.w:
            size -= 0.2
            track -= 0.1
        d.t(fr.x, fr.y, d.fit(txt, S.f_v, size, fr.w - track * len(txt)), S.f_v, size, S.acc, track=track)
        fr.down(3)
        d.line(fr.x, fr.y, fr.x1, fr.y, mix(S.rule, S.ink, 0.12), 0.5)

    def table(self, d, fr, titre, cols, lignes, note=None):
        S, st, lead = self.S, self.s_tab, self.lead
        if titre:
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
            for l in d.wrap(note, S.f_b, st * 0.9, fr.w)[:3]:
                d.t(fr.x, fr.y, l, S.f_b, st * 0.9, S.mut)
                fr.down(st * 1.2)
            fr.down(-st * 1.2)
        fr.down(2)

    def paragraphe(self, d, fr, txt, size=None, col=None, gras=False, indent=0):
        S = self.S
        size = size or self.s_hyp
        for l in d.wrap(txt, "Carlito-Bold" if gras else S.f_b, size, fr.w - indent):
            fr.down(size * 1.3)
            d.t(fr.x + indent, fr.y, l, "Carlito-Bold" if gras else S.f_b, size, col or S.ink)

    def ligne_kv(self, d, fr, label, txt, col=None):
        """« LABEL  texte… » : le libellé en gras, le texte à la suite puis en retrait."""
        S = self.S
        size = self.s_hyp
        wl = d.w(label, "Carlito-Bold", size) + 4
        mots = txt.split(" ")
        lignes, cur, w_dispo = [], "", fr.w - wl
        for mot in mots:
            t = (cur + " " + mot).strip()
            if d.w(t, S.f_b, size) <= w_dispo or not cur:
                cur = t
            else:
                lignes.append(cur)
                cur = mot
                w_dispo = fr.w - 6
        if cur:
            lignes.append(cur)
        fr.down(size * 1.32)
        d.t(fr.x, fr.y, label, "Carlito-Bold", size, S.ink)
        d.t(fr.x + wl, fr.y, lignes[0], S.f_b, size, col or S.ink)
        for l in lignes[1:]:
            fr.down(size * 1.32)
            d.t(fr.x + 6, fr.y, l, S.f_b, size, col or S.ink)

    def substitutions(self, d, fr, titre, subs, indent=4):
        S = self.S
        if not subs:
            return
        if titre:
            fr.down(self.s_hyp * 1.5)
            d.t(fr.x, fr.y, titre.upper(), S.f_v, 5.4, S.mut, track=1.1)
        for sub in subs:
            for l in d.wrap(FO.rendre(sub), S.f_b, self.s_hyp, fr.w - indent):
                fr.down(self.s_hyp * 1.3)
                d.t(fr.x + indent, fr.y, l, S.f_b, self.s_hyp, S.ink)

    def bloc(self, d, fr, titre, bloc, sous_titre=None):
        """Un élément : « n · ÉLÉMENT — section · cas gouvernant », les autres cas et leur taux, les formules développées."""
        if not bloc or not bloc.get("subs"):
            return
        S = self.S
        self.titre(d, fr, titre + (" — " + bloc["cas"] if bloc.get("cas") else ""))
        if sous_titre:
            fr.down(1)
            self.paragraphe(d, fr, sous_titre, size=self.s_hyp * 0.95, col=S.mut)
        if bloc.get("autres"):
            fr.down(1)
            self.paragraphe(d, fr, "Autres cas : " + bloc["autres"] + ".", size=self.s_hyp * 0.95, col=S.mut)
        self.substitutions(d, fr, None, bloc["subs"])
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
        d.t(M, y, f"{TITRE} — {C.sous_titre()}", S.f_h, 12.5, S.ink)
        ident = " · ".join(str(x) for x in (doc.get("projet"), doc.get("partie"), "cas " + cas["nom"]) if x)
        d.t(W - M, y, d.fit(ident, S.f_b, 8, 300), S.f_b, 8, S.ink, "right")
        y -= 8
        d.line(M, y, W - M, y, S.ink, 1.0)
        y -= 15
        libelle, cle = STATUTS[res["statut"]]
        col = {"ok": S.ok, "att": S.att, "ko": S.ko, "mut": S.mut}[cle]
        w = chip(d, M, y - 3.2, libelle, S.f_v, 6.6, "#FFFFFF", col)
        x = M + w + 8
        g = res.get("gouvernant")
        if res["statut"] == ET.REFUS or not g:
            d.t(x, y, d.fit((res["messages"] or ["—"])[0], S.f_b, 8, W - M - x), S.f_b, 8, S.ink)
        else:
            t1 = "η max " + _pct(res["taux_max"])
            d.t(x, y, t1, "Carlito-Bold", 8.6, col)
            x += d.w(t1, "Carlito-Bold", 8.6) + 5
            crit = ME.critere_texte(g["critere"]) if g["cle"] not in ("soudure", "ancrage", "platine") else g["critere"]
            cas_g = (next((e for e in res["elements"] if e["cle"] == g["cle"]), {}).get("detail") or {}).get("cas_charge")
            t2 = f"— gouverne : {g['libelle']} · {crit}" + (f" · {CH.CAS_LIBELLES.get(cas_g, cas_g)}" if cas_g else "")
            t3 = FO.TITRES[FO.cle_methode(res)]
            wt = d.w(t3, "Carlito-Bold", 7.6)
            xr = W - M - wt
            d.t(x, y, d.fit(t2, S.f_b, 8, xr - x - 6), S.f_b, 8, S.ink)
            d.t(xr, y, t3, "Carlito-Bold", 7.6, S.acc)
        y -= 6
        d.line(M, y, W - M, y, mix(S.rule, S.ink, 0.3), 0.6)
        return y - 4

    # --- colonne 1 : hypothèses, charges, dessins
    def hypotheses(self, d, fr):
        C = self.C
        self.titre(d, fr, "HYPOTHÈSES")
        fr.down(2)
        self.ligne_kv(d, fr, "Garde-corps", C.hyp_garde_corps())
        self.ligne_kv(d, fr, "Sections", C.hyp_sections())
        self.ligne_kv(d, fr, "Matériau et limites", C.hyp_methode())
        self.ligne_kv(d, fr, "Fixation", C.hyp_fixation())
        for m in [m for m in C.res.get("messages", []) if not m.startswith("Refus")][:2]:
            self.ligne_kv(d, fr, "Note", m, col=self.S.mut)
        fr.down(2)

    def table_charges(self, d, fr):
        C = self.C
        rows = []
        for c in C.res["charges_tableau"]:
            rows.append([("num", str(c["n"]), True), ("lab", f"{c['symbole']} — {c['nom']}", c["n"] == 1), ("num", _fmt(c["valeur"], 2) + " " + c["unite"], True), ("ref", c["norme"])])
        ch = C.res["charges"]
        if ch.get("Q", 0) > 0:
            rows.append([("num", "+"), ("lab", "Q — charge ponctuelle horizontale sur la main courante"), ("num", _fmt(ch["Q"], 2) + " kN"), ("ref", f"option, {C.meth.get('comb', 'somme')} avec q_h")])
        w = fr.w
        cols = [("N°", 12, "right"), ("CHARGE", w - 12 - 48 - 92, "left"), ("VALEUR", 48, "right"), ("ORIGINE", 92, "left")]
        self.table(d, fr, "CHARGES — " + C.res["limites"]["nature"] + ", quatre cas de charge", cols, rows, C.ligne_cas())

    def dessins(self, d, fr):
        S, C = self.S, self.C
        vues = C.vues
        self.titre(d, fr, "DESSINS — élévation, coupe A-A, plan (détails p. 2)")
        dispo = fr.room() - 6
        if not vues:
            fr.down(20)
            d.t(fr.x, fr.y + 6, (C.res.get("messages") or ["Calcul impossible"])[0][:90], S.f_b, self.s_hyp, S.ko)
            return
        # l'élévation et la coupe A-A côte à côte (comme sur une planche), la vue en plan dessous
        d0, dc = vues["elevation"], vues.get("coupe")
        w0 = 0.72 * fr.w if dc is not None else fr.w
        h0 = min(max(70.0, w0 * d0.viewbox[3] / d0.viewbox[2], (fr.w - w0 - 4) * dc.viewbox[3] / dc.viewbox[2] if dc is not None else 0.0), 0.66 * dispo)
        fr.down(h0 + 3)
        self.echelle = peindre_scene(d, d0, fr.x, fr.y, w0, h0)
        if dc is not None:
            peindre_scene(d, dc, fr.x + w0 + 4, fr.y, fr.w - w0 - 4, h0)
        dp = vues.get("plan")
        if dp is not None:
            h1 = min(max(40.0, fr.w * dp.viewbox[3] / dp.viewbox[2]), dispo - h0 - 10)
            fr.down(h1 + 3)
            peindre_scene(d, dp, fr.x, fr.y, fr.w, h1 - 2)

    # --- colonnes 2 et 3 : les éléments un à un, avec leurs formules
    def critere(self, d, fr):
        S, C = self.S, self.C
        cle = FO.cle_methode(C.res)
        self.titre(d, fr, "CRITÈRES — " + FO.TITRES[cle].split(" (")[0])
        fr.down(2)
        formula(d, fr, "", FO.CRITERES_MATH[cle], S.ms(), 7.0 * self.k, S.mut, S.f_b, 5.8, lead=1.0, min_size=5.0)

    def element_principal(self, d, fr):
        C = self.C
        res = C.res
        if C.systeme() == "montants" and res.get("montant"):
            m = res["montant"]
            self.bloc(d, fr, f"1 · MONTANT {m['nom']} — {m['section']['nom']}", C.fo["montant"],
                      f"Console encastrée en pied, hauteur H = {_g(m['verif']['H'])} mm, longueur tributaire s_trib = {_g(m['verif']['trib1'] + m['verif']['trib2'])} mm"
                      + (" (montant d'angle, deux directions)" if m["verif"]["coin"] else "") + ".")
        elif res.get("console"):
            c = res["console"]
            self.bloc(d, fr, f"1 · CONSOLE {c['nom']} — {c['section']['nom']}", C.fo["console"],
                      f"Console encastrée dans le mur, longueur e = {_g(c['e'])} mm, longueur tributaire s_trib = {_g(c['verif']['trib1'] + c['verif']['trib2'])} mm.")

    def main_courante(self, d, fr):
        C = self.C
        li = C.res.get("lisse")
        if not li:
            return
        v = li["verif"]
        mod = ME.APPUIS[li["appuis"]] if li["appuis"] != "continue" or li["n_travees"] >= 2 else ME.APPUIS["deux"]
        self.bloc(d, fr, f"2 · MAIN COURANTE — {li['section']['nom']}", C.fo["lisse"],
                  f"Poutre {mod}, portée L = {_g(v['L'])} mm (l'entraxe le plus grand) ; q_h fléchit l'axe faible (W_el,z), q_v l'axe fort (W_el,y)"
                  + (f" ; coefficients des tables pour {li['n_travees']} travées : k_M = {_g(v['kM'])}, k_V = {_g(v['kV'])}, k_f = {_g(v['kf'])}" if v.get("kM") else "") + ".")

    def remplissage(self, d, fr):
        C = self.C
        rp = C.res.get("remplissage") or {}
        if rp.get("verif"):
            v = rp["verif"]
            nom = "BARREAUX" if rp["type"] == "barreaux" else "LISSES INTERMÉDIAIRES"
            ent = f"e_b = {_g(rp['entraxe'])} mm" if rp["type"] == "barreaux" else f"e_l = {_g(rp['entraxe'])} mm"
            self.bloc(d, fr, f"3 · {nom} — {rp['section']['nom']}", C.fo["barre"],
                      f"Poutre sur deux appuis, portée L = {_g(round(v['L']))} mm, {ent} : Q_r à mi-portée (cas 3) ou q_r × entraxe (cas 4), le pire gouverne.")
        elif rp.get("type") == "panneau":
            self.titre(d, fr, "3 · PANNEAU")
            fr.down(1)
            self.paragraphe(d, fr, f"Remplissage en panneau (verre, tôle) : sa résistance sous Q_r = {_g(C.res['charges']['Qr'])} kN et q_r = {_g(C.res['charges']['qr'])} kN/m² "
                                   "est à justifier par le fournisseur ; la lisse basse n'est pas calculée.")
            fr.down(2)

    def ancrage(self, d, fr):
        S, C = self.S, self.C
        fx = C.res.get("fixation") or {}
        if not fx.get("soudure"):
            if (C.cas["fixation"].get("type") == "scellee"):
                self.titre(d, fr, "4 · ANCRAGE")
                fr.down(1)
                self.paragraphe(d, fr, "Montant scellé : la fixation n'est pas vérifiée (scellement à justifier séparément).")
            return
        s = fx["soudure"]
        pl = fx.get("platine")
        typ = C.cas["fixation"].get("type", "platine")
        self.titre(d, fr, "4 · ANCRAGE" + (" — " + CH.CAS_LIBELLES.get(pl.get("cas_charge"), "") if pl and pl.get("cas_charge") else ""))
        fr.down(2)
        self.paragraphe(d, fr, f"Soudure ({s['res']['config']} du module Soudures, {CH.CAS_LIBELLES.get(s.get('cas_charge'), '')}) : η = {_pct(s['taux'])}, "
                               f"{s['critere']}, {s['position']} — {s['efforts_txt']} ; détail en page 2.")
        if pl and pl.get("taux_max") is not None:
            self.paragraphe(d, fr, f"Platine rigide {_g(pl['bp'])} × {_g(pl['lp'])} × {_g(pl['tp'])}, {pl['n_total']} {pl.get('nom_ancrage') or 'chevilles'} en deux rangées "
                                   "à ±e_x/2 : la rangée tendue reprend Z, l'appui se fait au nu comprimé du montant.", size=self.s_hyp * 0.95, col=S.mut)
            self.substitutions(d, fr, None, C.fo["platine"]["subs"])
        elif pl:
            for m in pl.get("messages", []):
                self.paragraphe(d, fr, m, col=S.ko)
        fr.down(2)

    def recapitulatif(self, d, fr):
        C = self.C
        res = C.res
        g = res.get("gouvernant") or {}
        rows = []
        for e in res.get("elements", []):
            dd = e["detail"] if isinstance(e["detail"], dict) else {}
            gouv = e["cle"] == g.get("cle")
            cas_c = CH.CAS_LIBELLES.get(dd.get("cas_charge"), dd.get("cas_charge") or "—")
            crit = ME.critere_texte(e["critere"]) if e["cle"] in ("montant", "lisse", "barreau", "lisse_inter", "console") else e["critere"]
            rows.append([("lab", e["libelle"].split(" · ")[0] + (" · " + e["libelle"].split(" · ")[1] if " · " in e["libelle"] else ""), gouv),
                         ("ref", cas_c.replace("cas ", "")), ("eta", _pct(e["taux"]), e["taux"] <= 1.0, gouv), ("ref", crit[:24])])
        if not rows:
            return
        w = fr.w
        cols = [("ÉLÉMENT", w - 58 - 28 - 54, "left"), ("CAS", 58, "left"), ("η", 28, "right"), ("CRITÈRE", 54, "left")]
        self.table(d, fr, "RÉCAPITULATIF — éléments, cas gouvernants, taux", cols, rows)

    def conclusion(self, d, x, w):
        S, C = self.S, self.C
        res = C.res
        libelle, cle = STATUTS[res["statut"]]
        col = {"ok": S.ok, "att": S.att, "ko": S.ko, "mut": S.mut}[cle]
        g = res.get("gouvernant")
        if res["statut"] == ET.REFUS or not g:
            txt = (res["messages"] or ["Calcul impossible."])[0]
        else:
            crit = ME.critere_texte(g["critere"]) if g["cle"] not in ("soudure", "ancrage", "platine") else g["critere"]
            cas_g = (next((e for e in res["elements"] if e["cle"] == g["cle"]), {}).get("detail") or {}).get("cas_charge")
            elts = "montant principal, main courante, remplissage et ancrage" if C.systeme() == "montants" else "main courante, consoles et ancrage"
            porte = (f"Le garde-corps porte les charges retenues : {elts} vérifiés." if res["statut"] != ET.NOK else
                     "Le garde-corps ne porte pas les charges retenues en l'état : section, entraxe, hauteur ou ancrage à revoir (les tableaux de l'écran donnent le plus léger qui tient).")
            txt = (f"Taux maximal {_pct(res['taux_max'])} sur {g['libelle']} ({crit}" + (f", {CH.CAS_LIBELLES.get(cas_g, cas_g)}" if cas_g else "") + f"), selon {MODES_PHRASE[C.meth['mode']]}. "
                   + porte)
            if res.get("masse"):
                txt += f" Masse indicative de l'acier : {_fmt(res['masse'], 0)} kg."
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
        self.hypotheses(d, f1)
        self.table_charges(d, f1)
        self.dessins(d, f1)
        self.critere(d, f2)
        self.element_principal(d, f2)
        self.main_courante(d, f2)
        self.recapitulatif(d, f2)
        hc = self.conclusion(d, x3, w3)
        f3.bottom = BAS + hc + 6
        self.remplissage(d, f3)
        self.ancrage(d, f3)
        d.t(M, 24, d.fit(LEGENDE, S.f_b, 5.6, W - 2 * M), S.f_b, 5.6, S.mut)
        d.t(M, 16.5, d.fit(NOTATIONS, S.f_b, 5.6, W - 2 * M), S.f_b, 5.6, S.mut)
        for f in (f1, f2, f3):
            if f.overflow() > 1.0:
                self.warnings.append(f"page 1, colonne {f.x:.0f} : {f.overflow():.0f} pt")
        return self.warnings


CART_H = 40


class Methode:
    """Page 2 · DÉTAILS ET MÉTHODE."""

    def __init__(self, calc: Calcul, doc_meta: Dict, k: float = 1.0, echelle: Optional[float] = None):
        self.C, self.doc, self.k = calc, doc_meta, k
        self.S = Encre()
        self.warnings: List[str] = []
        self.s = 6.3 * k
        self.echelle = echelle
        self.note1 = Note(calc, doc_meta, k)

    def titre(self, d, fr, txt):
        fr.down(1)
        self.note1.titre(d, fr, txt)
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

    def details_fixation(self, d, fr):
        C = self.C
        vues = [(k, C.vues[k]) for k in ("platine", "pied") if k in C.vues]
        if not vues:
            return
        self.titre(d, fr, "DÉTAIL DE L'ANCRAGE — " + ", ".join(C.titres[k].lower().replace("coupe sur la fixation", "coupe") for k, _ in vues))
        parts = {"platine": 0.46, "pied": 0.54} if len(vues) == 2 else {vues[0][0]: 1.0}
        larg = {k: (fr.w - 4 * (len(vues) - 1)) * parts[k] for k, _ in vues}
        h = min(max(110.0, max(larg[k] * dv.viewbox[3] / dv.viewbox[2] for k, dv in vues)), 190.0)
        fr.down(h + 3)
        x = fr.x
        for k, dv in vues:
            peindre_scene(d, dv, x, fr.y, larg[k], h - 2)
            x += larg[k] + 4

    def table_montants(self, d, fr):
        C = self.C
        res = C.res
        liste = res.get("montants") or res.get("consoles") or []
        if not liste:
            return
        gouv = (res.get("montant") or res.get("console") or {}).get("nom")
        rows = []
        for v in liste[:16]:
            g = v["nom"] == gouv
            rows.append([("lab", v["nom"] + (" · angle" if v["coin"] else ""), g), ("num", _fmt(v["x"] / 1000, 2)), ("num", _fmt(v["trib1"] + v["trib2"], 0)),
                         ("num", _fmt(v["P1"] / 1e3, 2) + (" + " + _fmt(v["P2"] / 1e3, 2) if v["P2"] > 0 else "")), ("num", _fmt(v["N"] / 1e3, 2)),
                         ("ref", CH.CAS_LIBELLES.get(v.get("cas_charge"), "").replace("cas ", "")), ("eta", _pct(v["taux_max"]), v["taux_max"] <= 1.0, g)])
        note = f"+ {len(liste) - 16} non montrés" if len(liste) > 16 else None
        w = fr.w
        nom = "CONSOLE" if C.systeme() == "mur" else "MONTANT"
        cols = [(nom, w - 30 - 32 - 44 - 30 - 46 - 28, "left"), ("x (m)", 30, "right"), ("s_trib", 32, "right"), ("P (kN)", 44, "right"), ("N (kN)", 30, "right"), ("CAS", 46, "left"), ("η", 28, "right")]
        self.note1.s_tab, self.note1.lead = 5.8 * self.k, 7.8 * self.k
        self.note1.table(d, fr, f"{nom}S UN À UN — x, s_trib (mm), charges, cas, taux", cols, rows, note)

    def soudure(self, d, fr):
        C, S = self.C, self.S
        s = (C.res.get("fixation") or {}).get("soudure")
        if not s:
            return
        g = s["res"].get("gouvernant") or {}
        self.titre(d, fr, "SOUDURE SUR LA PLATINE — " + (FOS.TITRES[FOS.methode_du_point(s["res"], g)].split(" (")[0] if g else "module Soudures"))
        if C.vues_soudure:
            _, _, dv = C.vues_soudure[0]
            h = min(max(80.0, fr.w * 0.8 * dv.viewbox[3] / dv.viewbox[2]), 130.0)
            fr.down(h + 2)
            peindre_scene(d, dv, fr.x + 0.1 * fr.w, fr.y, fr.w * 0.8, h)
        fo = C.fo["soudure"]
        self.note1.s_hyp = self.s * 0.97
        for titre, cle in (("Groupe", "groupe"), ("Point gouvernant", "point"), ("Critère", "critere")):
            self.note1.substitutions(d, fr, titre, fo[cle])
        fr.down(3)

    def section_et_bibliotheque(self, d, fr):
        C = self.C
        dv = C.vues.get("section")
        role = "lisse" if C.systeme() == "mur" else "montant"
        self.titre(d, fr, "SECTION DE L'ÉLÉMENT PRINCIPAL ET BIBLIOTHÈQUE")
        if dv is not None:
            h = min(max(55.0, fr.w * 0.4 * dv.viewbox[3] / dv.viewbox[2]), 80.0)
            fr.down(h + 2)
            peindre_scene(d, dv, fr.x + fr.w * 0.3, fr.y, fr.w * 0.4, h)
        sec_cur = C.res.get("section_montant") or {}
        groupe = BI.groupe_de(sec_cur) if sec_cur else "creux"
        cands = ET.candidats(C.cas, role, [groupe], 15.0)
        if not cands:
            return
        best = ET.plus_leger(cands)
        cur = sec_cur.get("nom")
        montres = [c for c in cands if c["taux"] <= 1.6][:9]
        if best and best not in montres:
            montres.append(best)
        rows = [[("lab", c["section"]["nom"], c["section"]["nom"] == cur), ("num", _fmt(c["section"]["masse"], 2)), ("eta", _pct(c["taux"]), c["taux"] <= 1.0, c["section"]["nom"] == cur)]
                for c in montres]
        w = fr.w
        cols = [("SECTION", w - 36 - 30, "left"), ("kg/m", 36, "right"), ("η", 30, "right")]
        self.note1.s_tab, self.note1.lead = 5.8 * self.k, 7.8 * self.k
        self.note1.table(d, fr, None, cols, rows,
                         f"{BI.GROUPES.get(groupe, groupe)} essayés comme {ET.ROLES[role]}, par masse croissante · le plus léger qui tient : {best['section']['nom']} (η {_pct(best['taux'])}) ; "
                         "la section retenue est en gras." if best else "Aucune section ≤ 15 kg/m ne tient.")

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
        meth = MODES[m["mode"]] + (f" · σ_adm {_g(m.get('sigma_adm', 160))} · τ_adm {_g(m.get('tau_adm', 90))}" if m["mode"] == ME.ELS else f" · γ_Q {_g(m.get('gamma_Q', 1.5))} · γ_M0 {_g(m.get('gamma_M0', 1))}")
        cases = [("CAS", cas["nom"], ident or "—", 0.18),
                 ("GARDE-CORPS", C.type_libelle(), d.fit(C.res["geometrie"]["phrase"], S.f_b, 6.2, w * 0.28 - 12), 0.28),
                 ("MÉTHODE", meth, "flèche ≤ " + (f"L/{_g(m.get('fleche_val', 100))}" if m.get("fleche_mode", "L/x") == "L/x" else f"{_g(m.get('fleche_val', 0))} mm"), 0.20),
                 ("ACIER", C.acier["nuance"], f"f_y {_g(C.acier['fy'])} N/mm² · f_u {_g(C.acier['fu'])}", 0.12),
                 ("DATE · INDICE", str(doc.get("date", "")), "indice " + str(doc.get("indice", "")), 0.11),
                 ("ÉCHELLE DE L'ÉLÉVATION", ech, "A4 paysage", 0.11)]
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
        d.t(M, y, "DÉTAILS ET MÉTHODE", S.f_h, 13.5, S.ink)
        d.t(W - M, y, d.fit("Références : " + " · ".join(MT.REFERENCES), S.f_b, 6.2, W - 2 * M - 150), S.f_b, 6.2, S.mut, "right")
        y -= 7
        d.line(M, y, W - M, y, S.ink, 1.0)
        top = y - 6
        bas = M + CART_H + 10
        wc = (W - 2 * M - 2 * GAP) / 3.0
        frames = [Frame(M + i * (wc + GAP), top, wc, bas) for i in range(3)]
        for i in (1, 2):
            d.line(frames[i].x - GAP / 2, top, frames[i].x - GAP / 2, bas, S.rule, 0.45)
        self.details_fixation(d, frames[0])
        self.table_montants(d, frames[0])
        self.section_et_bibliotheque(d, frames[0])
        self.soudure(d, frames[1])
        els = self.C.meth["mode"] == ME.ELS
        secs = [s for s in MT.SECTIONS if not (s["titre"].startswith("Eurocode") and els) and not (s["titre"].startswith("Méthode ELS") and not els)]
        for sec in secs:
            self.section(d, frames[2], sec)
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
    n2 = Methode(calc, doc_meta, k, n1.echelle)
    w += n2.construire(d)
    d.save()
    return w


def generer_note(etat: Dict, cas: Optional[Dict] = None, infos: Optional[Dict] = None, chemin: Optional[str] = None) -> bytes:
    """La note (bytes PDF) : page 1 de calcul, page 2 de détails et méthode. ``infos`` : nom_projet, partie, date, indice."""
    global derniers_avertissements, derniere_variante
    cas = cas or ET.cas_actif(etat)
    infos = infos or {}
    doc_meta = ndc_data.construire_doc(infos, date_defaut=datetime.today().strftime("%d/%m/%Y"))
    doc_meta["titre"] = "Note de calcul — garde-corps"
    calc = Calcul(etat, cas)
    if chemin is None:
        fd, chemin = tempfile.mkstemp(suffix=".pdf", prefix="note_garde_corps_")
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

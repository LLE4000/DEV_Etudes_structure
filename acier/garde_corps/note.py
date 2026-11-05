# -*- coding: utf-8 -*-
"""Note de calcul « Garde-corps » — DEUX pages A4 paysage, sur le kit ``ndc_pdf`` des
notes de l'application (palette Encre, mêmes marges et mêmes polices que les notes
d'assemblage, de plancher et de soudures).

Page 1 · CALCUL : bandeau (statut, η max, élément gouvernant et critère, la méthode
CHOISIE, la géométrie, les charges et leur origine, la méthode en trois lignes), puis trois
colonnes — les DESSINS tels qu'à l'écran (élévation développée, vue en plan, détail de la
fixation : les scènes de ``schemas.py`` peintes en vectoriel par le peintre du kit des
assemblages, source unique) et les HYPOTHÈSES ; le tableau des CHARGES, celui des MONTANTS
(ou des consoles) avec leur longueur tributaire, celui des ÉLÉMENTS ; la VÉRIFICATION DE
L'ÉLÉMENT PRINCIPAL écrite comme à la main (le critère de la méthode choisie en typographie
mathématique, puis les substitutions numériques de ``formules.py``, testées contre le
moteur), la FIXATION (soudure, boulons ou chevilles, platine) et la CONCLUSION.

Page 2 · MÉTHODE ET DÉTAIL : les sections de ``methode.py`` de la méthode choisie (mêmes
textes que l'onglet Détail), la soudure (sa vue du plan de joint peinte, ses formules du
module Soudures), le détail de la lisse et du remplissage, la BIBLIOTHÈQUE (les sections
essayées), la section de l'élément principal, les références et un cartouche.

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
from . import geometrie as GE
from . import membres as ME
from . import methode as MT
from . import schemas as SC

A4L = landscape(A4)
M, BAS, GAP = 22, 40, 12
LARG = (318, 212)
S_TAB_H, S_DATA = 5.6, 7.0
LEAD_TAB = 8.8
TITRE = "Garde-corps"
STATUTS = {ET.OK: ("CONFORME", "ok"), ET.ATTENTION: ("CONFORME · ATTENTION", "att"), ET.NOK: ("NON CONFORME", "ko"), ET.REFUS: ("CALCUL IMPOSSIBLE", "mut")}
MODES = {ME.ELS: "Méthode ELS du bureau", ME.EUROCODE: "Eurocode (ELU)"}
MODES_PHRASE = {ME.ELS: "la méthode ELS du bureau (efforts de service, contraintes admissibles)", ME.EUROCODE: "l'Eurocode (efforts de calcul)"}
LEGENDE = ("Dessins : pièces au trait fort, sections hachurées, cordons en ocre à leur taille a·√2, chevilles en gris, ◉ l'élément gouvernant et son "
           "taux, flèches rouges dans le sens des charges (⊙ vers soi), cotes en mm. Chaque montant reçoit la charge de sa longueur tributaire ; un "
           "montant d'angle, une demi-travée de chaque tronçon dans deux directions.")
NOTATIONS = ("P : charge horizontale en tête du montant (q_h × s_trib) · N : charge verticale (q_v × s_trib) · M = P·H · W_el : module élastique "
             "(h dans le sens de l'effort) · A_v : aire de cisaillement (EN 1993-1-1 §6.2.6, entière pour un plein) · f : flèche · η : taux · "
             "Z : traction de la rangée de boulons tendue · m : bras de la platine en console.")
VARIANTES = ((1.0, "normale"), (0.92, "serrée"))


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
        self.vues = {cle: d for cle, _, d in SC.vues(self.res, SC.options_schemas(lvl=1))}
        self.titres = {cle: t for cle, t, _ in SC.vues(self.res, SC.options_schemas(lvl=1))}
        self.meth = self.res["methode"]
        self.acier = self.res["acier"]
        self.fo = FO.toutes(self.res)
        s = (self.res.get("fixation") or {}).get("soudure")
        self.vues_soudure = SCS.vues(s["res"], SCS.options_schemas(lvl=1)) if s else []

    def systeme(self) -> str:
        return self.cas["geometrie"].get("systeme", "montants")

    def sous_titre(self) -> str:
        g = self.cas["geometrie"]
        return "lisse ancrée dans un mur (consoles)" if self.systeme() == "mur" else "montants verticaux sur platine"

    def ligne_geometrie(self) -> str:
        geo = self.res["geometrie"]
        sec = self.res.get("section_montant") or {}
        li = (self.res.get("lisse") or {}).get("section") or {}
        rp = self.res.get("remplissage") or {}
        s = f"Géométrie — {geo['phrase']} · H = {_g(geo['H'])} mm"
        if self.systeme() == "montants":
            s += f" · montants {sec.get('nom', '—')} · lisse haute {li.get('nom', '—')}"
        else:
            c = self.res.get("console") or {}
            s += f" · lisse {li.get('nom', '—')} · consoles {(c.get('section') or {}).get('nom', '—')}, e = {_g(c.get('e', 0))} mm"
        if rp.get("section"):
            s += f" · {rp['libelle']} {rp['section']['nom']}" + (f" à {_g(rp['entraxe'])} mm" if rp.get("entraxe") else "")
        elif rp:
            s += f" · remplissage : {rp['libelle']}"
        return s

    def ligne_charges(self) -> str:
        return "Charges — " + CH.phrase(self.res["charges"]) + " · " + self.res["limites"]["nature"]

    def ligne_methode(self) -> str:
        m, lim = self.meth, self.res["limites"]
        fl = f"L/{_g(m.get('fleche_val', 100))}" if m.get("fleche_mode", "L/x") == "L/x" else f"{_g(m.get('fleche_val', 0))} mm"
        if m["mode"] == ME.ELS:
            return (f"Méthode ELS du bureau : σ = M/W_el + N/A ≤ σ_adm = {_g(lim['sigma'])} N/mm², τ = V/A_v ≤ τ_adm = {_g(lim['tau'])} N/mm², flèche ≤ {fl} · "
                    f"acier {self.acier['nuance']} · soudure k = {_g(m.get('k', 100))} kg/(mm·cm) · résistances de calcul des ancrages / γ_F = {_g(m.get('gamma_F', 1.5))}")
        return (f"Eurocode : charges × γ_Q = {_g(m.get('gamma_Q', 1.5))}, σ ≤ f_y/γ_M0 = {_fmt(lim['sigma'])} N/mm² (élastique, W_el), τ ≤ f_y/(√3·γ_M0) = {_fmt(lim['tau'])}, "
                f"flèche ≤ {fl} sous efforts de service · acier {self.acier['nuance']} (f_y = {_g(self.acier['fy'])}) · soudure EN 1993-1-8 {m.get('ec', 'directionnelle')}, γ_M2 = {_g(m.get('gamma_M2', 1.25))}")

    def ligne_fixation(self) -> str:
        fx = self.cas["fixation"]
        r = self.res.get("fixation") or {}
        typ = fx.get("type", "platine")
        s = "Fixation — " + FX.TYPES[typ]
        if typ == "scellee":
            return s
        pl = fx["platine"]
        s += f" : soudure {FX.texte_soudure(fx['soudure'])}"
        if typ == "platine":
            s += (f" · platine {_g(pl['bp'])} × {_g(pl['lp'])} × {_g(pl['tp'])} · {FX.texte_ancrage(fx['ancrage'])}, rangées à e_x = {_g(fx['ancrage']['ex'])} mm")
        return s


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
        self.formules_p1: List[str] = []

    # --- briques
    def titre(self, d, fr, txt):
        S = self.S
        fr.down(10)
        d.t(fr.x, fr.y, d.fit(txt, S.f_v, 6.2, fr.w - 1.3 * len(txt)), S.f_v, 6.2, S.acc, track=1.3)
        fr.down(3)
        d.line(fr.x, fr.y, fr.x1, fr.y, mix(S.rule, S.ink, 0.12), 0.5)

    def table(self, d, fr, titre, cols, lignes, note=None):
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

    def paragraphe(self, d, fr, txt, size=None, col=None, gras=False):
        S = self.S
        size = size or self.s_hyp
        for l in d.wrap(txt, "Carlito-Bold" if gras else S.f_b, size, fr.w):
            fr.down(size * 1.3)
            d.t(fr.x, fr.y, l, "Carlito-Bold" if gras else S.f_b, size, col or S.ink)

    def substitutions(self, d, fr, titre, subs, indent=4):
        S = self.S
        if not subs:
            return
        fr.down(self.s_hyp * 1.5)
        d.t(fr.x, fr.y, titre.upper(), S.f_v, 5.4, S.mut, track=1.1)
        for sub in subs:
            for l in d.wrap(FO.rendre(sub), S.f_b, self.s_hyp, fr.w - indent):
                fr.down(self.s_hyp * 1.3)
                d.t(fr.x + indent, fr.y, l, S.f_b, self.s_hyp, S.ink)

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
            t2 = f"— gouverne : {g['libelle']} · {crit}"
            t3 = FO.TITRES[FO.cle_methode(res)]
            wt = d.w(t3, "Carlito-Bold", 7.6)
            xr = W - M - wt
            d.t(x, y, d.fit(t2, S.f_b, 8, xr - x - 6), S.f_b, 8, S.ink)
            d.t(xr, y, t3, "Carlito-Bold", 7.6, S.acc)
        y -= 12
        for txt in (C.ligne_geometrie(), C.ligne_charges(), C.ligne_methode()):
            d.t(M, y, d.fit(txt, S.f_b, S_DATA, W - 2 * M), S.f_b, S_DATA, S.ink)
            y -= S_DATA * 1.4
        y -= 2
        d.line(M, y, W - M, y, mix(S.rule, S.ink, 0.3), 0.6)
        return y - 4

    # --- colonne 1 : dessins, hypothèses
    def colonne_dessin(self, d, fr):
        S, C = self.S, self.C
        lignes = []
        textes = [m for m in C.res.get("messages", []) if not m.startswith("Refus")]
        textes.append("Montant : console encastrée en pied chargée en tête ; lisse : poutre sur deux appuis entre montants ; barreau ou lisse "
                      "intermédiaire : poutre sur deux appuis sous q_r × entraxe ou Q_r à mi-portée (le pire des deux cas).")
        textes.append(C.ligne_fixation())
        for h in textes:
            lignes.extend(d.wrap("• " + h, S.f_b, self.s_hyp, fr.w))
        h_hyp = 13 + len(lignes) * self.s_hyp * 1.3 + 4
        dispo = fr.room() - h_hyp - 13 - 8
        vues = C.vues
        self.titre(d, fr, "DESSINS — " + " · ".join(t.lower() for k, t in C.titres.items() if k != "section"))
        if vues:
            d0 = vues["elevation"]
            h0 = min(max(80.0, fr.w * d0.viewbox[3] / d0.viewbox[2]), 0.5 * dispo)
            fr.down(h0 + 2)
            self.echelle = peindre_scene(d, d0, fr.x, fr.y, fr.w, h0)
            autres = [(k, vues[k]) for k in ("plan", "detail") if k in vues]
            if autres:
                parts = {"plan": 0.42, "detail": 0.58} if len(autres) == 2 else {autres[0][0]: 1.0}
                larg = {k: (fr.w - 4 * (len(autres) - 1)) * parts[k] for k, _ in autres}
                h1 = min(max(70.0, max(larg[k] * dv.viewbox[3] / dv.viewbox[2] for k, dv in autres)), dispo - h0 - 8)
                fr.down(h1 + 3)
                x = fr.x
                for k, dv in autres:
                    peindre_scene(d, dv, x, fr.y, larg[k], h1 - 2)
                    x += larg[k] + 4
        else:
            fr.down(20)
            d.t(fr.x, fr.y + 6, (C.res.get("messages") or ["Calcul impossible"])[0][:90], S.f_b, self.s_hyp, S.ko)
        self.titre(d, fr, "HYPOTHÈSES")
        fr.down(3)
        for l in lignes:
            fr.down(self.s_hyp * 1.3)
            d.t(fr.x, fr.y, l, S.f_b, self.s_hyp, S.mut)

    # --- colonne 2 : charges, montants, éléments
    def table_charges(self, d, fr):
        C = self.C
        ch = C.res["charges"]
        rows = []
        if ch["mode"] == "P":
            rows.append([("lab", "P par montant", True), ("num", _fmt(ch["P"], 2) + " kN"), ("ref", "saisie directe")])
        else:
            rows.append([("lab", "q_h horizontale", True), ("num", _fmt(ch["qh"], 2) + " kN/m"), ("ref", CH.origine_qh(ch).split(" — ")[0])])
        rows.append([("lab", "q_v verticale"), ("num", _fmt(ch["qv"], 2) + " kN/m"), ("ref", "main courante, usage du bureau")])
        if ch.get("Q", 0) > 0:
            rows.append([("lab", "Q ponctuelle"), ("num", _fmt(ch["Q"], 2) + " kN"), ("ref", f"sur la lisse, {C.meth.get('comb', 'somme')} avec q_h")])
        rows.append([("lab", "q_r remplissage"), ("num", _fmt(ch["qr"], 2) + " kN/m²"), ("ref", "pression, usage du bureau")])
        rows.append([("lab", "Q_r remplissage"), ("num", _fmt(ch["Qr"], 2) + " kN"), ("ref", "NBN B 03-004 (Q_kh,2)")])
        w = fr.w
        cols = [("CHARGE", 58, "left"), ("VALEUR", 50, "right"), ("ORIGINE", w - 108, "left")]
        self.table(d, fr, "CHARGES — " + C.res["limites"]["nature"], cols, rows,
                   "Tab. 6.12 : valeurs recommandées de l'EN 1991-1-1 ; l'ANB belge peut en fixer d'autres." if ch["mode"] != "P" and ch["categorie"] != "perso" else None)

    def table_montants(self, d, fr):
        C = self.C
        res = C.res
        liste = res.get("montants") or res.get("consoles") or []
        if not liste:
            return
        gouv = (res.get("montant") or res.get("console") or {}).get("nom")
        rows = []
        for v in liste[:14]:
            g = v["nom"] == gouv
            rows.append([("lab", v["nom"] + (" · angle" if v["coin"] else ""), g), ("num", _fmt(v["x"] / 1000, 2)), ("num", _fmt(v["trib1"] + v["trib2"], 0)),
                         ("num", _fmt(v["P1"] / 1e3, 2) + (" + " + _fmt(v["P2"] / 1e3, 2) if v["P2"] > 0 else "")), ("eta", _pct(v["taux_max"]), v["taux_max"] <= 1.0, g)])
        note = f"+ {len(liste) - 14} non montrés" if len(liste) > 14 else None
        w = fr.w
        nom = "CONSOLE" if C.systeme() == "mur" else "MONTANT"
        cols = [(nom, w - 34 - 34 - 46 - 30, "left"), ("x (m)", 34, "right"), ("s_trib", 34, "right"), ("P (kN)", 46, "right"), ("η", 30, "right")]
        self.table(d, fr, f"{nom}S — abscisse, longueur tributaire (mm), charge horizontale, taux", cols, rows, note)

    def table_elements(self, d, fr):
        C = self.C
        res = C.res
        g = res.get("gouvernant") or {}
        rows = []
        for e in res.get("elements", []):
            dd = e["detail"]
            gouv = e["cle"] == g.get("cle")
            if e["cle"] in ("montant", "lisse", "barreau", "lisse_inter", "console"):
                rows.append([("lab", e["libelle"].split(" · ")[0] + " · " + e["libelle"].split(" · ")[1] if " · " in e["libelle"] else e["libelle"], gouv),
                             ("num", _fmt(dd["sigma"], 0)), ("num", _fmt(dd["fleche"], 1) + "/" + _fmt(dd["fleche_lim"], 0)), ("eta", _pct(e["taux"]), e["taux"] <= 1.0, gouv),
                             ("ref", ME.critere_texte(e["critere"]))])
            else:
                rows.append([("lab", e["libelle"].split(" · ")[0], gouv), ("num", "—"), ("num", "—"), ("eta", _pct(e["taux"]), e["taux"] <= 1.0, gouv), ("ref", e["critere"][:22])])
        if not rows:
            return
        w = fr.w
        cols = [("ÉLÉMENT", w - 30 - 40 - 28 - 46, "left"), ("σ", 30, "right"), ("f / adm", 40, "right"), ("η", 28, "right"), ("CRITÈRE", 46, "left")]
        self.table(d, fr, "ÉLÉMENTS — contrainte (N/mm²), flèche (mm), taux", cols, rows)

    # --- colonne 3 : vérification de l'élément principal, fixation, conclusion
    def verification(self, d, fr):
        C, S = self.S, self.C
        S, C = self.S, self.C
        fo = C.fo
        cle = FO.cle_methode(C.res)
        princ = fo["montant"] or fo["console"]
        if not princ:
            return
        nom = "DU MONTANT " + (C.res.get("montant") or {}).get("nom", "") if C.systeme() == "montants" else "DE LA CONSOLE " + (C.res.get("console") or {}).get("nom", "")
        self.titre(d, fr, f"VÉRIFICATION {nom} — " + FO.TITRES[cle].split(" (")[0])
        fr.down(2)
        formula(d, fr, "", FO.CRITERES_MATH[cle], S.ms(), 7.0 * self.k, S.mut, S.f_b, 5.8, lead=1.0, min_size=5.0)
        self.substitutions(d, fr, "Efforts et contraintes", princ)
        self.formules_p1.append("montant" if fo["montant"] else "console")

    def fixation(self, d, fr):
        S, C = self.S, self.C
        fx = C.res.get("fixation") or {}
        if not fx.get("soudure"):
            return
        self.titre(d, fr, "FIXATION — soudure, ancrage, platine")
        s = fx["soudure"]
        txt = (f"Soudure ({s['res']['config']} du module Soudures) : η = {_pct(s['taux'])}, {s['critere']}, {s['position']} — {s['efforts_txt']} ; détail en page 2.")
        fr.down(2)
        self.paragraphe(d, fr, txt)
        pl = fx.get("platine")
        if pl and pl.get("taux_max") is not None:
            self.substitutions(d, fr, "Boulons ou chevilles, platine", C.fo["platine"])
            self.formules_p1.append("platine")
        elif pl:
            for m in pl.get("messages", []):
                self.paragraphe(d, fr, m, col=S.ko)

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
            porte = ("Le garde-corps porte les charges retenues : montants, lisse, remplissage et fixation vérifiés." if res["statut"] != ET.NOK else
                     "Le garde-corps ne porte pas les charges retenues en l'état : section, entraxe, hauteur ou fixation à revoir (la bibliothèque donne le plus léger qui tient).")
            txt = f"Taux maximal {_pct(res['taux_max'])} sur {g['libelle']} ({crit}), selon {MODES_PHRASE[C.meth['mode']]}. " + porte
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
        self.colonne_dessin(d, f1)
        self.table_charges(d, f2)
        self.table_montants(d, f2)
        self.table_elements(d, f2)
        hc = self.conclusion(d, x3, w3)
        f3.bottom = BAS + hc + 6
        self.verification(d, f3)
        self.fixation(d, f3)
        d.t(M, 24, d.fit(LEGENDE, S.f_b, 5.6, W - 2 * M), S.f_b, 5.6, S.mut)
        d.t(M, 16.5, d.fit(NOTATIONS, S.f_b, 5.6, W - 2 * M), S.f_b, 5.6, S.mut)
        for f in (f1, f2, f3):
            if f.overflow() > 1.0:
                self.warnings.append(f"page 1, colonne {f.x:.0f} : {f.overflow():.0f} pt")
        return self.warnings


CART_H = 40


class Methode:
    """Page 2 · MÉTHODE ET DÉTAIL."""

    def __init__(self, calc: Calcul, doc_meta: Dict, k: float = 1.0, echelle: Optional[float] = None, formules_p1=()):
        self.C, self.doc, self.k = calc, doc_meta, k
        self.S = Encre()
        self.warnings: List[str] = []
        self.s = 6.3 * k
        self.echelle = echelle
        self.formules_p1 = set(formules_p1)
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

    def soudure(self, d, fr):
        C, S = self.C, self.S
        s = (C.res.get("fixation") or {}).get("soudure")
        if not s:
            return
        self.titre(d, fr, "SOUDURE SUR LA PLATINE — " + FOS.TITRES[FOS.methode_du_point(s["res"], s["res"].get("gouvernant") or {})].split(" (")[0] if s["res"].get("gouvernant") else "SOUDURE")
        if C.vues_soudure:
            _, _, dv = C.vues_soudure[0]
            h = min(max(90.0, fr.w * dv.viewbox[3] / dv.viewbox[2]), 150.0)
            fr.down(h + 2)
            peindre_scene(d, dv, fr.x, fr.y, fr.w, h)
        fo = C.fo["soudure"]
        self.note1.s_hyp = self.s * 0.97
        for titre, cle in (("Groupe", "groupe"), ("Point gouvernant", "point"), ("Critère", "critere")):
            self.note1.substitutions(d, fr, titre, fo[cle])
        fr.down(3)

    def autres_formules(self, d, fr):
        C = self.C
        fo = C.fo
        blocs = [("La lisse", "lisse"), ("Le remplissage", "barre")]
        if "platine" not in self.formules_p1:
            blocs.append(("Platine et ancrage", "platine"))
        if "montant" not in self.formules_p1 and fo["montant"]:
            blocs.insert(0, ("Le montant", "montant"))
        if "console" not in self.formules_p1 and fo["console"]:
            blocs.insert(0, ("La console", "console"))
        if not any(fo[c] for _, c in blocs):
            return
        self.titre(d, fr, "DÉTAIL DES AUTRES ÉLÉMENTS — comme à la main")
        self.note1.s_hyp = self.s * 0.97
        for titre, cle in blocs:
            self.note1.substitutions(d, fr, titre, fo[cle])
        fr.down(3)

    def bibliotheque(self, d, fr):
        C, S = self.C, self.S
        sec_cur = C.res.get("section_montant") or {}
        groupe = BI.groupe_de(sec_cur) if sec_cur else "creux"
        cands = ET.candidats(C.cas, [groupe], 15.0)
        if not cands:
            return
        best = ET.plus_leger(cands)
        cur = sec_cur.get("nom")
        montres = [c for c in cands if c["taux"] <= 1.6][:12]
        if best and best not in montres:
            montres.append(best)
        rows = []
        for c in montres:
            s = c["section"]
            rows.append([("lab", s["nom"], s["nom"] == cur), ("num", _fmt(s["masse"], 2)), ("eta", _pct(c["taux"]), c["taux"] <= 1.0, s["nom"] == cur)])
        w = fr.w
        cols = [("SECTION", w - 36 - 30, "left"), ("kg/m", 36, "right"), ("η", 30, "right")]
        lib = "montant" if C.systeme() == "montants" else "lisse"
        self.note1.s_tab, self.note1.lead = 5.8 * self.k, 7.8 * self.k
        self.note1.table(d, fr, f"BIBLIOTHÈQUE — {BI.GROUPES.get(groupe, groupe)} essayés comme {lib}, par masse croissante", cols, rows,
                         f"Le plus léger qui tient : {best['section']['nom']} (η {_pct(best['taux'])}) ; la section retenue est en gras." if best else "Aucune section ≤ 15 kg/m ne tient.")

    def section_vue(self, d, fr):
        C = self.C
        dv = C.vues.get("section")
        if dv is None:
            return
        self.titre(d, fr, "SECTION DE L'ÉLÉMENT PRINCIPAL")
        h = min(max(70.0, fr.w * 0.55 * dv.viewbox[3] / dv.viewbox[2]), 110.0)
        fr.down(h + 2)
        peindre_scene(d, dv, fr.x + fr.w * 0.2, fr.y, fr.w * 0.6, h)

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
                 ("GARDE-CORPS", C.sous_titre(), d.fit(C.res["geometrie"]["phrase"], S.f_b, 6.2, w * 0.28 - 12), 0.28),
                 ("MÉTHODE", meth, f"flèche ≤ " + (f"L/{_g(m.get('fleche_val', 100))}" if m.get("fleche_mode", "L/x") == "L/x" else f"{_g(m.get('fleche_val', 0))} mm"), 0.20),
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
        d.t(M, y, "MÉTHODE ET DÉTAIL", S.f_h, 13.5, S.ink)
        d.t(W - M, y, d.fit("Références : " + " · ".join(MT.REFERENCES), S.f_b, 6.2, W - 2 * M - 150), S.f_b, 6.2, S.mut, "right")
        y -= 7
        d.line(M, y, W - M, y, S.ink, 1.0)
        top = y - 6
        bas = M + CART_H + 10
        wc = (W - 2 * M - 2 * GAP) / 3.0
        frames = [Frame(M + i * (wc + GAP), top, wc, bas) for i in range(3)]
        for i in (1, 2):
            d.line(frames[i].x - GAP / 2, top, frames[i].x - GAP / 2, bas, S.rule, 0.45)
        els = self.C.meth["mode"] == ME.ELS
        secs = [s for s in MT.SECTIONS if not (s["titre"].startswith("Eurocode") and els) and not (s["titre"].startswith("Méthode ELS") and not els)]
        for sec in secs[:2]:
            self.section(d, frames[0], sec)
        self.autres_formules(d, frames[0])
        for sec in secs[2:]:
            self.section(d, frames[1], sec)
        self.bibliotheque(d, frames[1])
        self.soudure(d, frames[2])
        self.section_vue(d, frames[2])
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
    n2 = Methode(calc, doc_meta, k, n1.echelle, n1.formules_p1)
    w += n2.construire(d)
    d.save()
    return w


def generer_note(etat: Dict, cas: Optional[Dict] = None, infos: Optional[Dict] = None, chemin: Optional[str] = None) -> bytes:
    """La note (bytes PDF) : page 1 de calcul, page 2 de méthode et détail. ``infos`` : nom_projet, partie, date, indice."""
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

# -*- coding: utf-8 -*-
"""Note de calcul « Garde-corps » (version 3) — DEUX pages A4 paysage, sur le kit ``ndc_pdf`` des
notes de l'application (palette Encre, mêmes marges et mêmes polices que les notes d'assemblage,
de plancher et de soudures), avec les symboles COMPOSÉS (``ndc_pdf.riche`` : q_h, W_el,y, σ_adm en
vraie typographie, dans les hypothèses, les charges, les formules, les tableaux) et peu de prose.

Page 1 · CALCUL, les hypothèses d'abord :
- bandeau : statut, η max, l'élément gouvernant, son critère et son cas de charge, la méthode ;
- colonne 1 : HYPOTHÈSES en lignes « libellé · valeur » (garde-corps, hauteur, sections, acier,
  méthode, limites, fixation), les QUATRE CHARGES — le symbole et la valeur sur la première ligne,
  le nom et l'origine dessous —, la règle des cas, les DESSINS (élévation, coupe A-A, vue en
  plan) peints depuis les mêmes scènes que l'écran ;
- colonnes 2 et 3 : 1 · MONTANT (ou la console au mur), 2 · MAIN COURANTE, 3 · BARREAUX (ou lisses,
  ou panneau), 4 · ANCRAGE (soudure, chevilles ou boulons, platine) — chaque élément porte sa
  pastille η, son cas gouvernant et les autres cas, une ligne de données, puis ses formules
  DÉVELOPPÉES par GROUPE dans l'ordre de lecture (charges → sollicitations → contraintes → flèche
  → taux ; ``formules.py``, testées contre le moteur) avec le résultat calé à droite ; le
  RÉCAPITULATIF (barres de taux) ; la CONCLUSION.

Page 2 · DÉTAILS ET MÉTHODE : la platine en plan et la coupe sur la fixation, le tableau des
MONTANTS (ou des consoles), la section de l'élément principal et la BIBLIOTHÈQUE, la SOUDURE (vue
et formules du module Soudures), les textes de MÉTHODE (``methode.py``) de la méthode choisie,
les références, le cartouche.

``derniers_avertissements`` : débordements de colonne (vide si tout tient) ; la note est
d'abord composée en taille normale, puis en taille réduite si une colonne déborde.
"""
import math
import os
import re
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from reportlab.lib.pagesizes import A4, landscape

from ndc_pdf import data as ndc_data
from ndc_pdf import riche as RI
from ndc_pdf.kit import Doc, Frame, chip, hx, mix
from ndc_pdf.mathx import MathStyle
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
S_TAB_H = 5.6
LEAD_TAB = 8.8
LAB_W = 58
TITRE = "Garde-corps"
STATUTS = {ET.OK: ("CONFORME", "ok"), ET.ATTENTION: ("CONFORME · ATTENTION", "att"), ET.NOK: ("NON CONFORME", "ko"), ET.REFUS: ("CALCUL IMPOSSIBLE", "mut")}
MODES = {ME.ELS: "Méthode ELS du bureau", ME.EUROCODE: "Eurocode (ELU)"}
MODES_PHRASE = {ME.ELS: "méthode ELS du bureau (charges de service, contraintes admissibles)", ME.EUROCODE: "Eurocode (efforts de calcul)"}
LEGENDE = ("Dessins à l'échelle, cotes en mm · ◉ l'élément gouvernant et son taux · flèches rouges : les charges (⊙ vers soi) · A-A : plan de la coupe · "
           "cordons en ocre à leur taille a·√2, chevilles en gris · chaque montant reçoit la charge de sa longueur tributaire.")
VARIANTES = ((1.0, "normale"), (0.92, "serrée"))
FIXATIONS = {"platine": "platine boulonnée", "soudee": "montant soudé sur un plat existant", "scellee": "montant scellé"}


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


def _cas(lbl: Optional[str]) -> str:
    """Un libellé de cas de charge en texte riche (« cas 1 · $q_h$ seule »)."""
    return RI.auto(lbl or "")


def _efforts(txt: str) -> str:
    """« N = 0,00 kN, V = 0,45 kN, M = 0,495 kN·m (…) » avec ses symboles composés."""
    return re.sub(r"\b([NVM]) = ", r"$\1$ = ", txt)


def _sans_parentheses(txt: str) -> str:
    return re.sub(r"\s*\([^)]*\)", "", txt).strip()


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
        lib = self.type_libelle().lower()
        return lib + ("" if self.systeme() == "mur" or " sur " in lib else " sur montants")

    def fleche_adm(self) -> str:
        m = self.meth
        return f"$L$/{_g(m.get('fleche_val', 100))}" if m.get("fleche_mode", "L/x") == "L/x" else f"{_g(m.get('fleche_val', 0))} mm"

    # --- les hypothèses, en lignes « libellé · valeur » (valeur en texte riche)
    def hypotheses(self) -> List[Tuple[str, str]]:
        res, cas, geo = self.res, self.cas, self.res["geometrie"]
        L = [("Garde-corps", f"{self.type_libelle()} · {geo['phrase']}")]
        if self.systeme() == "mur":
            c = res.get("console") or {}
            L.append(("Hauteur", f"$H$ = {_g(geo['H'])} mm au-dessus du sol · consoles à $e$ = {_g(c.get('e', 0))} mm du mur"))
            L.append(("Consoles", (c.get("section") or {}).get("nom", "—")))
        else:
            L.append(("Hauteur", f"$H$ = {_g(geo['H'])} mm au-dessus de la platine"))
            L.append(("Montants", (res.get("section_montant") or {}).get("nom", "—")))
        li = (res.get("lisse") or {}).get("section") or {}
        L.append(("Main courante", f"{li.get('nom', '—')} · portée {_g(geo['s_max'])} mm, {ME.APPUIS[cas['lisse'].get('appuis', 'deux')]}"))
        rp = res.get("remplissage") or {}
        if rp.get("type") == "barreaux" and rp.get("section"):
            L.append(("Barreaux", f"{rp['section']['nom']} · $e_b$ = {_g(rp['entraxe'])} mm · portée {_g(round(rp['L']))} mm, déduite de $H$, $h_{{bas}}$ = {_g(rp['h_bas'])} mm et des sections"))
            if rp.get("section_basse"):
                L.append(("Lisse basse", f"{rp['section_basse']['nom']} (non calculée)"))
        elif rp.get("type") == "lisses" and rp.get("section"):
            L.append(("Lisses", f"{rp.get('n', 1)} lisses intermédiaires {rp['section']['nom']} · $e_l$ = {_g(rp['entraxe'])} mm"))
        elif rp.get("type") == "panneau":
            L.append(("Panneau", "verre ou tôle, justifié par son fournisseur" + (f" · lisse basse {rp['section_basse']['nom']}" if rp.get("section_basse") else "")))
        else:
            L.append(("Remplissage", "sans remplissage"))
        ac, m, lim = self.acier, self.meth, res["limites"]
        L.append(("Acier", f"{ac['nuance']} · $f_y$ = {_g(ac['fy'])} N/mm² · $f_u$ = {_g(ac['fu'])} N/mm² · $E$ = {_fmt(float(m.get('E') or ME.E_ACIER), 0)} N/mm²"))
        fl = self.fleche_adm()
        if m["mode"] == ME.ELS:
            L.append(("Méthode", "ELS du bureau : charges de service sans coefficient, contraintes admissibles"))
            L.append(("Limites", f"$σ_{{adm}}$ = {_g(lim['sigma'])} N/mm² · $τ_{{adm}}$ = {_g(lim['tau'])} N/mm² · flèche ≤ {fl} · soudure $k$ = {_g(m.get('k', 100))} kg/(mm·cm) · "
                                 f"ancrages : résistances de calcul / $γ_F$ = {_g(m.get('gamma_F', 1.5))}"))
        else:
            L.append(("Méthode", f"Eurocode : efforts de calcul, charges × $γ_Q$ = {_g(m.get('gamma_Q', 1.5))} · vérification élastique ($W_{{el}}$)"))
            L.append(("Limites", f"$σ$ ≤ $f_y/γ_{{M0}}$ = {_fmt(lim['sigma'])} N/mm² · $τ$ ≤ $f_y/(\\sqrt{{3}}·γ_{{M0}})$ = {_fmt(lim['tau'])} N/mm² · flèche ≤ {fl} sous charges de service · "
                                 f"soudure EN 1993-1-8 {m.get('ec', 'directionnelle')}, $γ_{{M2}}$ = {_g(m.get('gamma_M2', 1.25))}"))
        L.append(("Fixation", self.hyp_fixation()))
        for msg in [x for x in res.get("messages", []) if not x.startswith("Refus")][:2]:
            L.append(("Note", RI.auto(msg)))
        return L

    def hyp_fixation(self) -> str:
        fx = self.cas["fixation"]
        typ = fx.get("type", "platine")
        if typ == "scellee":
            return "montant scellé : la fixation n'est pas vérifiée"
        s = f"{FIXATIONS[typ]} · soudure {RI.auto(FX.texte_soudure(fx['soudure']))}"
        if typ == "platine":
            pl, anc = fx["platine"], fx["ancrage"]
            s += f" · platine {_g(pl['bp'])} × {_g(pl['lp'])} × {_g(pl['tp'])} mm · {RI.auto(FX.texte_ancrage(anc))}, rangées à $e_x$ = {_g(anc['ex'])} mm, $e_y$ = {_g(anc['ey'])} mm"
        return s

    def ligne_cas(self) -> str:
        if self.res["charges"].get("cumul"):
            return "$q_h$ et $q_v$ agissent ensemble (cas 1 + 2, côté sûr) ; $Q_r$ et $q_r$ en cas séparés (3 et 4) ; chaque élément est vérifié sous chaque cas, le pire gouverne."
        return "Les quatre charges agissent SÉPARÉMENT (cas 1 à 4) : chaque élément est vérifié sous chaque cas, le pire gouverne."

    def charges_lignes(self) -> List[Dict]:
        rows = list(self.res["charges_tableau"])
        ch = self.res["charges"]
        if ch.get("Q", 0) > 0:
            rows.append({"n": "+", "symbole": "Q", "nom": "charge ponctuelle horizontale sur la main courante", "valeur": ch["Q"], "unite": "kN",
                         "norme": f"option, {self.meth.get('comb', 'somme')} avec q_h"})
        return rows

    def taux_de(self, cle: str) -> Optional[float]:
        for e in self.res.get("elements", []):
            if e["cle"] == cle:
                return e["taux"]
        return None


class Note:
    """Page 1 · CALCUL."""

    def __init__(self, calc: Calcul, doc_meta: Dict, k: float = 1.0):
        self.C, self.doc, self.k = calc, doc_meta, k
        self.S = Encre()
        self.warnings: List[str] = []
        self.s_txt = 6.1 * k
        self.s_tab = 6.2 * k
        self.lead = LEAD_TAB * k
        self.echelle = None
        self._ms: Dict = {}

    # --- composition
    def ms(self, col=None, op=None, res=None) -> MathStyle:
        """Le style mathématique d'un texte : symboles en Carlito italique, mots et nombres en romain,
        opérateurs dans la couleur ``op`` (grise dans les formules), résultat en gras ``res``."""
        S = self.S
        key = (str(col), str(op), str(res))
        st = self._ms.get(key)
        if st is None:
            st = MathStyle(var="Carlito-Italic", rom="Carlito", num="Carlito", op="Carlito", ink=hx(col or S.ink), accent=hx(S.acc), rule=hx(col or S.ink),
                           res="Carlito-Bold", res_color=hx(res or S.acc), op_color=hx(op) if op else None, tight=0.88)
            self._ms[key] = st
        return st

    def rt(self, d, x, y, s, size, col=None, font=None, align="left", width=None):
        col = col or self.S.ink
        return RI.texte(d, x, y, s, font or self.S.f_b, size, col, self.ms(col), align, width)

    def rp(self, d, fr, s, size=None, col=None, font=None, indent=0, lead=1.32, max_lignes=None):
        col = col or self.S.ink
        return RI.paragraphe(d, fr, s, font or self.S.f_b, size or self.s_txt, col, self.ms(col), lead, indent, max_lignes)

    # --- briques
    def titre(self, d, fr, txt, taux=None):
        """Un titre de bloc en petites capitales espacées, sa pastille η à droite ; il se resserre avant de se tronquer."""
        S = self.S
        fr.down(10)
        wc = 0.0
        if taux is not None:
            lab = "η " + _pct(taux)
            wc = d.w(lab, "Carlito-Bold", 5.8) + 10
            chip(d, fr.x1 - wc, fr.y - 2.5, lab, "Carlito-Bold", 5.8, "#FFFFFF", S.ok if taux <= 1.0 else S.ko, pad=5, h=8.4)
        avail = fr.w - (wc + 6 if wc else 0.0)
        size, track = 6.2, 1.3
        while size > 5.0 and d.w(txt, S.f_v, size) + track * len(txt) > avail:
            size -= 0.2
            track -= 0.1
        d.t(fr.x, fr.y, d.fit(txt, S.f_v, size, avail - track * len(txt)), S.f_v, size, S.acc, track=track)
        fr.down(3.2)
        d.line(fr.x, fr.y, fr.x1, fr.y, mix(S.rule, S.ink, 0.12), 0.5)

    def etiquette(self, d, fr, txt):
        """Le libellé d'un groupe de formules, en petites capitales grises."""
        fr.down(self.s_txt * 1.6)
        d.t(fr.x, fr.y, txt.upper(), self.S.f_v, 4.9 * self.k, self.S.mut, track=1.0)

    def ligne_hyp(self, d, fr, label, val, col=None):
        """« LIBELLÉ   valeur » : le libellé en petites capitales à gauche, la valeur composée à droite, repliée."""
        S = self.S
        size = self.s_txt
        col = col or S.ink
        lab_w = LAB_W * self.k
        lignes = RI.envelopper(val, S.f_b, size, self.ms(col), fr.w - lab_w)
        fr.down(size * 1.42)
        d.t(fr.x, fr.y + 0.2, label.upper(), S.f_v, 4.9 * self.k, S.mut, track=0.9)
        for i, l in enumerate(lignes):
            if i:
                fr.down(size * 1.3)
            RI.dessiner(d, fr.x + lab_w, fr.y, l, S.f_b, size, col, self.ms(col))

    def barre(self, d, x, y, w, taux, col):
        """Une barre de taux : la piste vaut 100 %, le remplissage s'arrête au taux (plein au-delà)."""
        S = self.S
        d.box(x, y, w, 2.6, fill=mix(S.rule, "#FFFFFF", 0.35), r=1.3)
        d.box(x, y, max(2.6, w * min(taux, 1.0)), 2.6, fill=col, r=1.3)

    def table(self, d, fr, titre, cols, lignes, note=None):
        """Un tableau : cellules ``(style, texte riche, …)`` — lab (gras si retenu), ref (gris), num (à droite, gras),
        eta (couleur du verdict, gras si gouverne, barre de taux si un taux est donné)."""
        S, st, lead = self.S, self.s_tab, self.lead
        if titre:
            self.titre(d, fr, titre)
        fr.down(8)
        xs, x = [], fr.x
        for ent, w, al in cols:
            xs.append(x)
            self.rt(d, x + (w - 3 if al == "right" else 0), fr.y, ent, S_TAB_H, S.mut, align=al)
            x += w
        fr.down(2.5)
        d.line(fr.x, fr.y, fr.x1, fr.y, S.acc, 0.5)
        for cells in lignes:
            rendus, n_max = [], 1
            for (ent, w, al), cell in zip(cols, cells):
                style, txt = cell[0], cell[1]
                if style in ("lab", "ref"):
                    size = st * 0.87 if style == "ref" else st
                    f = "Carlito-Bold" if (style == "lab" and len(cell) > 2 and cell[2]) else S.f_b
                    col = S.ink if style == "lab" else S.mut
                    ls = RI.envelopper(txt, f, size, self.ms(col), w - 4)
                    n_max = max(n_max, len(ls))
                    rendus.append((style, ls, cell, f, size, col))
                else:
                    rendus.append((style, [txt], cell, S.f_b, st, S.ink))
            h = lead * n_max + 1.2
            fr.down(lead - 1.8)
            y = fr.y
            for (ent, w, al), (style, ls, cell, f, size, col), xx in zip(cols, rendus, xs):
                if style in ("lab", "ref"):
                    for i, l in enumerate(ls):
                        RI.dessiner(d, xx, y - i * lead, l, f, size, col, self.ms(col))
                elif style == "num":
                    gras = len(cell) > 2 and cell[2]
                    d.t(xx + w - 3, y, ls[0], "Carlito-Bold" if gras else S.f_b, st, S.ink, "right")
                elif style == "eta":
                    ok = cell[2]
                    gras = len(cell) > 3 and cell[3]
                    colr = S.ok if ok else S.ko
                    fnt = "Carlito-Bold" if (gras or not ok) else S.f_b
                    d.t(xx + w - 3, y, ls[0], fnt, st, colr, "right")
                    if len(cell) > 4 and cell[4] is not None:
                        wb = w - 3 - d.w(ls[0], fnt, st) - 10
                        if wb >= 12:
                            self.barre(d, xx + 2, y + 0.8, wb, cell[4], colr)
            fr.down(h - (lead - 1.8))
            d.line(fr.x, fr.y + 1.5, fr.x1, fr.y + 1.5, mix(S.rule, S.ink, 0.06), 0.4)
        if note:
            fr.down(st * 0.3)
            self.rp(d, fr, note, size=st * 0.9, col=S.mut, lead=1.24, max_lignes=3)
        fr.down(2)

    def formules(self, d, fr, groupes, indent=4):
        """Les formules par groupe : l'étiquette du groupe, puis chaque substitution « nom = littéral =
        numérique » avec son résultat calé à droite (les taux η en vert ou en rouge)."""
        S = self.S
        size = self.s_txt
        ms = self.ms(S.ink, op=S.mut)
        for lib, subs in groupes:
            if not subs:
                continue
            if lib:
                self.etiquette(d, fr, lib)
            for sub in subs:
                parts, res = FO.morceaux(sub)
                col_res = None
                if sub.nom.startswith("η") and sub.res is not None and math.isfinite(sub.res):
                    col_res = S.ok if sub.res <= 1.0 else S.ko
                RI.formule_calee(d, fr, [RI.tex(p) for p in parts], self.res_tex(sub, res), ms, size, ms_res=self.ms(S.ink, op=S.mut, res=col_res),
                                 indent=indent, retrait=10, lead=1.38, min_size=4.8)

    @staticmethod
    def res_tex(sub, res: str) -> str:
        """Le résultat d'une substitution en mini-LaTeX accentué : « \\res{0,45 \\t{kN}} »."""
        if sub.nd == -1 or not sub.unit:
            return r"\res{" + RI.tex(res) + "}"
        return r"\res{" + RI.tex(FO.fmt(sub.res, sub.nd)) + r" \t{" + sub.unit + "}}"

    def bloc(self, d, fr, titre, bloc, taux=None, donnees=None):
        """Un élément : titre et pastille η, le cas qui gouverne et les autres cas, la ligne de données, les formules par groupe."""
        if not bloc or not bloc.get("subs"):
            return
        S = self.S
        self.titre(d, fr, titre, taux)
        fr.down(0.5)
        if bloc.get("cas"):
            autres = ", ".join(f"{_cas(lib)} ({_pct(t)})" for lib, t in bloc.get("cas_taux", []))
            self.rp(d, fr, f"Gouverne : {_cas(bloc['cas'])}" + (f" · autres cas : {autres}" if autres else ""), size=self.s_txt * 0.95, col=S.mut)
        if donnees:
            self.rp(d, fr, donnees, size=self.s_txt * 0.95, col=S.mut)
        self.formules(d, fr, bloc["groupes"])
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
            t2 = f"— gouverne : {RI.auto(g['libelle'])} · {RI.auto(crit)}" + (f" · {_cas(CH.CAS_LIBELLES.get(cas_g, cas_g))}" if cas_g else "")
            t3 = FO.TITRES[FO.cle_methode(res)]
            wt = d.w(t3, "Carlito-Bold", 7.6)
            xr = W - M - wt
            self.rt(d, x, y, t2, 8, S.ink, width=xr - x - 6)
            d.t(xr, y, t3, "Carlito-Bold", 7.6, S.acc)
        y -= 6
        d.line(M, y, W - M, y, mix(S.rule, S.ink, 0.3), 0.6)
        return y - 4

    # --- colonne 1 : hypothèses, charges, dessins
    def hypotheses(self, d, fr):
        self.titre(d, fr, "HYPOTHÈSES")
        fr.down(1)
        for label, val in self.C.hypotheses():
            self.ligne_hyp(d, fr, label, val, self.S.mut if label == "Note" else None)
        fr.down(2)

    def table_charges(self, d, fr):
        """Les charges, une à une : le symbole et la valeur sur la première ligne, le nom dessous, l'origine en gris."""
        S, C = self.S, self.C
        st = self.s_tab
        self.titre(d, fr, "CHARGES — " + C.res["limites"]["nature"].split(" (")[0] + ", quatre cas de charge")
        fr.down(8)
        x_sym = fr.x + 14
        d.t(fr.x, fr.y, "N°", S.f_b, S_TAB_H, S.mut)
        d.t(x_sym, fr.y, "CHARGE", S.f_b, S_TAB_H, S.mut)
        d.t(fr.x1 - 3, fr.y, "VALEUR", S.f_b, S_TAB_H, S.mut, "right")
        fr.down(2.5)
        d.line(fr.x, fr.y, fr.x1, fr.y, S.acc, 0.5)
        for c in C.charges_lignes():
            fr.down(st * 1.8)
            d.t(fr.x + 8, fr.y, str(c["n"]), "Carlito-Bold", st, S.mut, "right")
            self.rt(d, x_sym, fr.y, "$" + RI.tex(c["symbole"]) + "$", st * 1.2, S.ink)
            d.t(fr.x1 - 3, fr.y, _fmt(c["valeur"], 2) + " " + c["unite"], "Carlito-Bold", st, S.ink, "right")
            self.rp(d, fr, c["nom"], size=st * 0.95, indent=14, lead=1.3, max_lignes=2)
            self.rp(d, fr, RI.auto(c["norme"]), size=st * 0.86, col=S.mut, indent=14, lead=1.26, max_lignes=2)
            fr.down(2.4)
            d.line(fr.x, fr.y, fr.x1, fr.y, mix(S.rule, S.ink, 0.06), 0.4)
        fr.down(st * 0.4)
        self.rp(d, fr, C.ligne_cas(), size=st * 0.9, col=S.mut, lead=1.24)
        fr.down(2)

    def dessins(self, d, fr):
        S, C = self.S, self.C
        vues = C.vues
        self.titre(d, fr, "DESSINS — élévation, coupe A-A, plan (détails p. 2)")
        dispo = fr.room() - 6
        if not vues:
            fr.down(20)
            d.t(fr.x, fr.y + 6, (C.res.get("messages") or ["Calcul impossible"])[0][:90], S.f_b, self.s_txt, S.ko)
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
    def element_principal(self, d, fr):
        C = self.C
        res = C.res
        if C.systeme() == "montants" and res.get("montant"):
            m = res["montant"]
            v = m["verif"]
            self.bloc(d, fr, f"1 · MONTANT {m['nom']} — {m['section']['nom']}", C.fo["montant"], C.taux_de("montant"),
                      f"$H$ = {_g(v['H'])} mm · $s_{{trib}}$ = {_g(v['trib1'] + v['trib2'])} mm · console encastrée en pied" + (" · montant d'angle : deux directions" if v["coin"] else ""))
        elif res.get("console"):
            c = res["console"]
            v = c["verif"]
            self.bloc(d, fr, f"1 · CONSOLE {c['nom']} — {c['section']['nom']}", C.fo["console"], C.taux_de("console"),
                      f"$e$ = {_g(c['e'])} mm du mur · $s_{{trib}}$ = {_g(v['trib1'] + v['trib2'])} mm · encastrée dans le mur")

    def main_courante(self, d, fr):
        C = self.C
        li = C.res.get("lisse")
        if not li:
            return
        v = li["verif"]
        cont = li["appuis"] == "continue" and li["n_travees"] >= 2
        donnees = f"$L$ = {_g(v['L'])} mm (l'entraxe le plus grand) · {ME.APPUIS['continue'] if cont else ME.APPUIS['deux']}"
        if cont and v.get("kM"):
            donnees += f" · {li['n_travees']} travées : $k_M$ = {_g(v['kM'])}, $k_V$ = {_g(v['kV'])}, $k_f$ = {_g(v['kf'])}"
        self.bloc(d, fr, f"2 · MAIN COURANTE — {li['section']['nom']}", C.fo["lisse"], C.taux_de("lisse"), donnees)

    def remplissage(self, d, fr):
        C = self.C
        rp = C.res.get("remplissage") or {}
        if rp.get("verif"):
            v = rp["verif"]
            barreaux = rp["type"] == "barreaux"
            nom = "BARREAUX" if barreaux else "LISSES INTERMÉDIAIRES"
            ent = ("$e_b$" if barreaux else "$e_l$") + f" = {_g(rp['entraxe'])} mm"
            self.bloc(d, fr, f"3 · {nom} — {rp['section']['nom']}", C.fo["barre"], C.taux_de("barreau" if barreaux else "lisse_inter"),
                      f"$L$ = {_g(round(v['L']))} mm · {ent} · sur deux appuis · $Q_r$ à mi-portée (cas 3) ou $q_r$ × entraxe (cas 4)")
        elif rp.get("type") == "panneau":
            self.titre(d, fr, "3 · PANNEAU")
            fr.down(1)
            self.rp(d, fr, f"Remplissage en panneau (verre, tôle) : sa résistance sous $Q_r$ = {_g(C.res['charges']['Qr'])} kN et $q_r$ = {_g(C.res['charges']['qr'])} kN/m² "
                           "est à justifier par le fournisseur ; la lisse basse n'est pas calculée.")
            fr.down(2)

    def ancrage(self, d, fr):
        S, C = self.S, self.C
        fx = C.res.get("fixation") or {}
        s = fx.get("soudure")
        if not s:
            if C.cas["fixation"].get("type") == "scellee":
                self.titre(d, fr, "4 · ANCRAGE — montant scellé")
                fr.down(1)
                self.rp(d, fr, "Montant scellé : la fixation n'est pas vérifiée (scellement à justifier séparément).")
            return
        pl = fx.get("platine")
        typ = C.cas["fixation"].get("type", "platine")
        cas_g = (pl or {}).get("cas_charge") or s.get("cas_charge")
        self.titre(d, fr, "4 · ANCRAGE — " + FIXATIONS.get(typ, typ), fx.get("taux_max"))
        fr.down(0.5)
        self.rp(d, fr, f"Gouverne : {_cas(CH.CAS_LIBELLES.get(cas_g, cas_g))} · en pied du montant : {_efforts(s['efforts_txt'])}", size=self.s_txt * 0.95, col=S.mut)
        fos = C.fo["soudure"]
        groupes = [(f"Soudure — {s['res']['config']} du module Soudures, {s['position']}, détail p. 2", (fos["point"][:1] if fos["point"] else []) + fos["critere"])]
        if pl and pl.get("taux_max") is not None:
            groupes += C.fo["platine"]["groupes"]
        self.formules(d, fr, groupes)
        if pl and pl.get("taux_max") is None:
            for m in pl.get("messages", []):
                self.rp(d, fr, m, col=S.ko)
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
            lib = [_sans_parentheses(x) for x in e["libelle"].split(" · ")]
            if e["cle"] == "ancrage" and len(lib) > 1:
                lib[0] = "Chevilles" if "cheville" in lib[1] else "Boulons"
            rows.append([("lab", RI.auto(" · ".join(lib[:2])), gouv), ("ref", _cas(cas_c).replace("cas ", "")), ("eta", _pct(e["taux"]), e["taux"] <= 1.0, gouv, e["taux"]),
                         ("ref", RI.auto(crit[:24]))])
        if not rows:
            return
        w = fr.w
        cols = [("ÉLÉMENT", w - 48 - 44 - 46, "left"), ("CAS", 48, "left"), ("η", 44, "right"), ("CRITÈRE", 46, "left")]
        self.s_tab = 6.0 * self.k
        self.table(d, fr, "RÉCAPITULATIF — éléments, cas gouvernants, taux", cols, rows)
        self.s_tab = 6.2 * self.k

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
            elts = "montant, main courante, remplissage et ancrage" if C.systeme() == "montants" else "main courante, consoles et ancrage"
            porte = (f"Le garde-corps porte les charges retenues : {elts} vérifiés." if res["statut"] != ET.NOK else
                     "Le garde-corps ne porte pas les charges retenues en l'état : section, entraxe, hauteur ou ancrage à revoir (les tableaux de l'écran donnent le plus léger qui tient).")
            txt = (f"Taux maximal {_pct(res['taux_max'])} sur {RI.auto(g['libelle'])} ({RI.auto(crit)}" + (f", {_cas(CH.CAS_LIBELLES.get(cas_g, cas_g))}" if cas_g else "")
                   + f"), {MODES_PHRASE[C.meth['mode']]}. " + porte)
            if res.get("masse"):
                txt += f" Masse indicative de l'acier : {_fmt(res['masse'], 0)} kg."
        lignes = RI.envelopper(txt, S.f_b, 6.6, self.ms(S.ink), w - 16)
        h = 13 + 9 + len(lignes) * 6.6 * 1.3 + 8
        y0 = BAS
        d.box(x, y0, w, h, fill=mix(S.panel, "#FFFFFF", 0.3), stroke=col, lw=0.8, r=3)
        d.t(x + 8, y0 + h - 11, "CONCLUSION", S.f_v, 6.2, col, track=1.3)
        yy = y0 + h - 11 - 9
        for l in lignes:
            RI.dessiner(d, x + 8, yy, l, S.f_b, 6.6, S.ink, self.ms(S.ink))
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
        self.element_principal(d, f2)
        self.main_courante(d, f2)
        self.recapitulatif(d, f2)
        hc = self.conclusion(d, x3, w3)
        f3.bottom = BAS + hc + 6
        self.remplissage(d, f3)
        self.ancrage(d, f3)
        d.t(M, 20, d.fit(LEGENDE, S.f_b, 5.4, W - 2 * M), S.f_b, 5.4, S.mut)
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

    def paragraphe(self, d, fr, txt, col=None):
        self.note1.rp(d, fr, RI.auto(txt), size=self.s, col=col)
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
                         ("ref", _cas(CH.CAS_LIBELLES.get(v.get("cas_charge"), "")).replace("cas ", "")), ("eta", _pct(v["taux_max"]), v["taux_max"] <= 1.0, g)])
        note = f"+ {len(liste) - 16} non montrés" if len(liste) > 16 else None
        w = fr.w
        nom = "CONSOLE" if C.systeme() == "mur" else "MONTANT"
        cols = [(nom, w - 30 - 34 - 44 - 30 - 46 - 28, "left"), ("$x$ (m)", 30, "right"), ("$s_{trib}$ (mm)", 34, "right"), ("$P$ (kN)", 44, "right"), ("$N$ (kN)", 30, "right"),
                ("CAS", 46, "left"), ("η", 28, "right")]
        self.note1.s_tab, self.note1.lead = 5.8 * self.k, 7.8 * self.k
        self.note1.table(d, fr, f"{nom}S UN À UN — abscisse, longueur tributaire, charges, cas, taux", cols, rows, note)

    def soudure(self, d, fr):
        C = self.C
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
        self.note1.s_txt = self.s * 0.97
        self.note1.formules(d, fr, [("Groupe de cordons", fo["groupe"]), ("Point gouvernant", fo["point"]), ("Critère", fo["critere"])])
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
        rows = [[("lab", c["section"]["nom"], c["section"]["nom"] == cur), ("num", _fmt(c["section"]["masse"], 2)), ("eta", _pct(c["taux"]), c["taux"] <= 1.0, c["section"]["nom"] == cur, c["taux"])]
                for c in montres]
        w = fr.w
        cols = [("SECTION", w - 36 - 64, "left"), ("kg/m", 36, "right"), ("η", 64, "right")]
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
        meth = MODES[m["mode"]] + (f" · $σ_{{adm}}$ {_g(m.get('sigma_adm', 160))} · $τ_{{adm}}$ {_g(m.get('tau_adm', 90))}" if m["mode"] == ME.ELS
                                   else f" · $γ_Q$ {_g(m.get('gamma_Q', 1.5))} · $γ_{{M0}}$ {_g(m.get('gamma_M0', 1))}")
        cases = [("CAS", cas["nom"], ident or "—", 0.18),
                 ("GARDE-CORPS", C.type_libelle(), C.res["geometrie"]["phrase"], 0.28),
                 ("MÉTHODE", meth, "flèche ≤ " + C.fleche_adm(), 0.20),
                 ("ACIER", C.acier["nuance"], f"$f_y$ {_g(C.acier['fy'])} N/mm² · $f_u$ {_g(C.acier['fu'])}", 0.12),
                 ("DATE · INDICE", str(doc.get("date", "")), "indice " + str(doc.get("indice", "")), 0.11),
                 ("ÉCHELLE DE L'ÉLÉVATION", ech, "A4 paysage", 0.11)]
        x = x0
        for i, (lab, l1, l2, part) in enumerate(cases):
            cw = w * part
            if i:
                d.line(x, y0 + 2, x, y0 + h - 2, S.rule, 0.6)
            d.t(x + 6, y0 + h - 10, lab, S.f_v, 5.4, S.mut, track=1.2)
            self.note1.rt(d, x + 6, y0 + h - 21, str(l1), 6.6, S.ink, font="Carlito-Bold", width=cw - 12)
            if l2:
                self.note1.rt(d, x + 6, y0 + h - 32, str(l2), 6.2, S.ink, width=cw - 12)
            x += cw

    def construire(self, d):
        S = self.S
        W, H = d.W, d.H
        y = H - M - 4
        d.t(M, y, self.doc.get("bureau", ""), S.f_v, 8.0, S.acc, track=1.4)
        d.t(W - M, y, TITRE + " · cas " + self.C.cas["nom"], S.f_b, 7.6, S.mut, "right")
        y -= 16
        d.t(M, y, "DÉTAILS ET MÉTHODE", S.f_h, 13.5, S.ink)
        self.note1.rt(d, W - M, y, "Références : " + " · ".join(RI.auto(r) for r in MT.REFERENCES), 6.2, S.mut, align="right", width=W - 2 * M - 150)
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

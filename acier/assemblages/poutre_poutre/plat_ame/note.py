# -*- coding: utf-8 -*-
"""Note de calcul du plat d'âme — DEUX pages A4 paysage, la même charte que
le module doubles cornières (palette Encre, bandeau statut, en-tête compact
en trois lignes, trois colonnes, pied en deux lignes) :

- page 1 : bandeau (statut, η max, dimensionnante, taux par élément,
  alertes), en-tête (objets · attaches · efforts), dessins + hypothèses,
  tableaux par élément (Ed / Rd / η / réf.), Tableau 3.3, formules des
  vérifications essentielles, conclusion ;
- page 2 : « PLAN DE PRINCIPE » — trois vues à la même échelle normalisée,
  héritée du module doubles cornières (PlanPrincipe), avec les vues et le
  cartouche de CET assemblage.
"""
import os
import tempfile
from datetime import datetime

from reportlab.lib.pagesizes import A4, landscape

from ndc_pdf.kit import Doc, Frame, chip, mix
from ndc_pdf.styles import Encre
from ndc_pdf import data as ndc_data
from acier.formats import F, pct
from acier.js import N, js_str
from acier.bibliotheques import PERSO
from . import schemas, synthese, formules
from .entrees import VISSERIE_DEFAUT
from .notation import ligne_alerte, LEGENDE_REFERENCES
from ..doubles_cornieres.rapport import peindre
from ..doubles_cornieres.note import PlanPrincipe, TEXTE_MM

A4L = landscape(A4)
M = 22
BAS = 40
GAP = 12
LARG = (210, 262, None)
S_TAB, S_TAB_H, S_REF, S_FORM, S_DATA = 6.5, 5.6, 5.6, 6.0, 7.0
LEAD_TAB = 8.8

TITRE = "Assemblage poutre–poutre — plat d'âme soudé (fin plate)"
NOTATIONS_COURTES = ("Notations : EN 1993-1-8 Tab. 3.3 (e1, p1, e2, p2) ; hp, bp, tp : plat ; zp : position du "
                     "plat ; e2,b : pince à l'about ; c, dc : grugeage ; gh, Δz : jeu et décalage ; a : gorge "
                     "(cordon double) ; z : excentricité (face d'appui → centre du groupe).")
VARIANTES = ("toutes", "eta")


def _prof(R, X):
    u = R.u
    if u["prof_" + X] == PERSO:
        return "h " + F(R["h_" + X], 0) + " b " + F(R["b_" + X], 0) + " tw " + F(R["tw_" + X], 1) + " tf " + F(R["tf_" + X], 1)
    return u["prof_" + X]


def lignes_entete(R):
    """L'en-tête compact : objets, attaches, efforts — pas de géométrie
    détaillée (elle est cotée sur le plan de principe)."""
    u = R.u
    grugee = bool(N(u.d_nt) or N(u.d_nb))
    l1 = ("Principale " + _prof(R, "P") + " " + u.nu_P
          + "    |    Secondaire " + _prof(R, "S") + " " + u.nu_S + (" · grugée" if grugee else "")
          + "    |    Plat " + js_str(R.h_p) + "×" + js_str(R.b_p) + "×" + js_str(R.t_p) + " " + u.nu_pl)
    l2 = ("Boulons " + R.boulon + " " + u.classe + " cat. " + u.cat + " · trous Ø" + F(R.d_0, 0)
          + (" surdim." if u.trou != "Normal" else "")
          + "    |    Groupe " + js_str(R.n_1) + " × " + js_str(R.n_2) + " (1 plan)"
          + "    |    Soudure double a " + F(R.a_w, 0))
    l3 = ("VEd " + F(u.V_Ed, 1) + " kN    |    NEd " + F(u.N_Ed, 1) + " kN    |    MEd "
          + F(u.M_Ed, 2) + " kNm")
    return l1, l2, l3


def lignes_hypotheses(R):
    u = R.u
    h = ["Articulé — rotule à la face de l'âme porteuse ; le groupe de boulons reprend VEd·z (MSB P5 §3.2).",
         "z = " + F(R.zeff, 1) + " mm ; MS = VEd·z + |MEd| = " + F(R.M_S, 2) + " kNm ; Ip = " + F(R.Ip, 0) + " mm².",
         "Plat " + ("COURT (z ≤ tp/0,15 : pas de déversement)" if not R.long_p
                    else "LONG (z > tp/0,15 : déversement vérifié, courbe BS 5950-1)") + "."]
    if R.cas_g > 0:
        h.append("Grugeage : flexion de la section réduite (MSB P5 §4.2.4).")
    h.append("γM0 " + F(N(u.g_M0), 2) + " · γM1 " + F(N(u.g_M1), 2) + " (ANB) · γM2 " + F(N(u.g_M2), 2)
             + " · nettes " + F(N(u.g_M2n), 2) + ".")
    h.append("Cordons : deux cordons verticaux pleine hauteur (§4.5.3.3).")
    return h


def _priorite(R):
    act = [c for c in R.checks if c.active]
    ordre = []
    if R.gov:
        ordre.append(R.gov)
    ordre += [c for c in act if not c.ok]
    ordre += [pire for _, _, _, pire in synthese.taux_par_element(R)]
    ordre += [c for c in act if c.ess]
    vus = []
    for c in ordre:
        if c not in vus:
            vus.append(c)
    return vus


def _obligatoires(R):
    n = (1 if R.gov else 0) + sum(1 for c in R.checks if c.active and not c.ok) + len(synthese.taux_par_element(R))
    return _priorite(R)[:n]


# ------------------------------------------------------------- page 1
class Note:
    """Compose la page 1 ; ``warnings`` = débordements (vide si tout tient)."""

    def __init__(self, R, doc_meta, variante):
        self.R = R; self.doc = doc_meta; self.mode_lignes = variante
        self.S = Encre(); self.warnings = []; self.formules_imprimees = []

    def titre(self, d, fr, txt):
        S = self.S
        fr.down(10)
        d.t(fr.x, fr.y, txt, S.f_v, 6.2, S.acc, track=1.3)
        fr.down(3)
        d.line(fr.x, fr.y, fr.x1, fr.y, mix(S.rule, S.ink, 0.12), 0.5)

    def table(self, d, fr, titre, cols, lignes):
        S = self.S
        self.titre(d, fr, titre)
        fr.down(8)
        xs = []
        x = fr.x
        for ent, w, al in cols:
            xs.append(x)
            d.t(x + (w - 3 if al == "right" else 0), fr.y, ent.upper() if ent != "η" else ent, S.f_b, S_TAB_H, S.mut, al)
            x += w
        fr.down(2.5)
        d.line(fr.x, fr.y, fr.x1, fr.y, S.acc, 0.5)
        for cells in lignes:
            rendus = []
            n_max = 1
            for (ent, w, al), cell in zip(cols, cells):
                style, txt = cell[0], cell[1]
                if style in ("lab", "ref"):
                    size = S_REF if style == "ref" else S_TAB
                    ls = d.wrap(txt, S.f_b, size, w - 4)
                    n_max = max(n_max, len(ls))
                    rendus.append((style, ls, cell))
                else:
                    rendus.append((style, [txt], cell))
            h = LEAD_TAB * n_max + 1.2
            fr.down(LEAD_TAB - 1.8)
            y = fr.y
            for (ent, w, al), (style, ls, cell), xx in zip(cols, rendus, xs):
                if style == "lab":
                    f = S.f_b if not (len(cell) > 2 and cell[2] == "gov") else "Carlito-Bold"
                    for i, l in enumerate(ls):
                        d.t(xx, y - i * LEAD_TAB, l, f, S_TAB, S.ink)
                elif style == "ref":
                    for i, l in enumerate(ls):
                        d.t(xx, y - i * LEAD_TAB, l, S.f_b, S_REF, S.mut)
                elif style == "num":
                    d.t(xx + w - 3, y, ls[0], S.f_b, S_TAB, S.ink, "right")
                elif style == "eta":
                    ok, gov = cell[2], (len(cell) > 3 and cell[3])
                    d.t(xx + w - 3, y, ls[0], "Carlito-Bold" if (gov or not ok) else S.f_b, S_TAB,
                        S.ok if ok else S.ko, "right")
                elif style == "st":
                    ok = cell[2]
                    d.t(xx + w - 3, y, ls[0], "Carlito-Bold", S_TAB, S.ok if ok else S.ko, "right")
            fr.down(h - (LEAD_TAB - 1.8))
            d.line(fr.x, fr.y + 1.5, fr.x1, fr.y + 1.5, mix(S.rule, S.ink, 0.06), 0.4)
        fr.down(2)

    def bandeau(self, d):
        S, R, doc = self.S, self.R, self.doc
        W, H = d.W, d.H
        y = H - M - 4
        d.t(M, y, doc.get("bureau", ""), S.f_v, 8.0, S.acc, track=1.4)
        d.t(W - M, y, "Note de calcul · " + doc.get("date", "") + " · indice " + str(doc.get("indice", "")), S.f_b, 8, S.mut, "right")
        y -= 15
        d.t(M, y, TITRE, S.f_h, 12.5, S.ink)
        u = R.u
        ident = " · ".join(str(x) for x in (doc.get("projet") or u.id_projet, doc.get("partie") or u.id_rep, u.id_red) if x)
        if ident:
            d.t(W - M, y, ident, S.f_b, 8, S.ink, "right")
        y -= 8
        d.line(M, y, W - M, y, S.ink, 1.0)
        y -= 15
        etat_ok = R.verified
        w = chip(d, M, y - 3.2, "VÉRIFIÉ" if etat_ok else "NON VÉRIFIÉ", S.f_v, 6.6, "#FFFFFF", S.ok if etat_ok else S.ko)
        x = M + w + 8
        t1 = "η max " + pct(R.eta_max, 1)
        d.t(x, y, t1, "Carlito-Bold", 8.6, S.ok if etat_ok else S.ko)
        x += d.w(t1, "Carlito-Bold", 8.6) + 5
        from .notation import ref_courte
        t2 = ("— dimensionnant : " + synthese.court(R.gov) + " (" + ref_courte(R.gov.ref) + ")") if R.gov else ""
        if not R.geo_ok:
            t2 += " · géométrie non valide"
        if not R.dist_ok:
            t2 += " · Tableau 3.3 non conforme"
        if R.verified and (R.reserve or R.alerts):
            t2 += " · sous réserve des points signalés"
        pieces = []
        for nom, eta, ok, _ in synthese.taux_par_element(R):
            pieces.append((nom + " ", S.f_b, S.ink)); pieces.append((pct(eta, 0), "Carlito-Bold", S.ok if ok else S.ko))
            pieces.append(("   ", S.f_b, S.ink))
        wt = sum(d.w(t, f, 8) for t, f, _ in pieces)
        xr = W - M - wt
        d.t(x, y, d.fit(t2, S.f_b, 8, xr - x - 6), S.f_b, 8, S.ink)
        for t, f, col in pieces:
            d.t(xr, y, t, f, 8, col)
            xr += d.w(t, f, 8)
        al = list(R.alerts)
        for a in al[:3]:
            y -= 10
            titre, det = ligne_alerte(a, R)
            col = S.ko if a.block else S.att
            d.box(M + 1, y - 0.5, 4.2, 4.2, fill=col)
            txt = titre + (" : " + det if det else "")
            d.t(M + 9, y, d.fit(txt, S.f_b, 6.8, W - 2 * M - 12), S.f_b, 6.8, S.ink)
        if len(al) > 3:
            y -= 9
            d.t(M + 9, y, "+ " + str(len(al) - 3) + " autre(s) point(s) signalé(s).", S.f_b, 6.4, S.mut)
        y -= 12
        l1, l2, l3 = lignes_entete(R)
        d.t(M, y, d.fit(l1, S.f_b, S_DATA, W - 2 * M), S.f_b, S_DATA, S.ink)
        y -= S_DATA * 1.35
        d.t(M, y, d.fit(l2, S.f_b, S_DATA, W - 2 * M), S.f_b, S_DATA, S.ink)
        y -= S_DATA * 1.45
        d.t(M, y, l3, "Carlito-Bold", S_DATA, S.ink)
        y -= 6
        d.line(M, y, W - M, y, mix(S.rule, S.ink, 0.3), 0.6)
        return y - 4

    def colonne_dessins(self, d, fr):
        S, R = self.S, self.R
        hyp = lignes_hypotheses(R)
        lignes = []
        for h in hyp:
            lignes.extend(d.wrap(h, S.f_b, 6.2, fr.w))
        h_hyp = 13 + len(lignes) * 6.2 * 1.3 + 4
        dispo = fr.room() - h_hyp - 2 * 13 - 6
        he = dispo * 0.58; hp = dispo - he
        opt = schemas.options_rapport()
        self.titre(d, fr, "ÉLÉVATION (mm)")
        fr.down(he)
        peindre(d, schemas.elevation(R, opt), fr.x, fr.y, fr.w, he - 2)
        self.titre(d, fr, "VUE EN PLAN")
        fr.down(hp)
        peindre(d, schemas.plan(R, opt), fr.x, fr.y, fr.w, hp - 2)
        self.titre(d, fr, "HYPOTHÈSES")
        fr.down(3)
        for l in lignes:
            fr.down(6.2 * 1.3)
            d.t(fr.x, fr.y, l, S.f_b, 6.2, S.mut)

    def table_simple(self, d, fr, cle, largeurs):
        R = self.R
        ls = synthese.lignes(R, cle)
        if not ls:
            return
        cols = [("Vérification", largeurs[0], "left"), ("Ed", largeurs[1], "right"), ("Rd", largeurs[2], "right"),
                ("η", largeurs[3], "right"), ("Réf.", largeurs[4], "left")]
        lignes = []
        for l in ls:
            gov = l["c"] is R.gov
            lignes.append([("lab", l["lab"], "gov" if gov else ""),
                           ("num", l["Ed"]), ("num", l["Rd"] + (" " + l["unit"] if l["unit"] else "")),
                           ("eta", l["pct"], l["ok"], gov), ("ref", l["ref"])])
        self.table(d, fr, synthese.titre_table(cle).upper(), cols, lignes)

    def table_pinces(self, d, fr):
        R = self.R
        if not R.dist:
            return
        w = fr.w
        cols = [("Distance (Tab. 3.3)", w - 128, "left"), ("Valeur", 32, "right"), ("Min", 32, "right"),
                ("Max", 32, "right"), ("", 32, "right")]
        lignes = [[("lab", x.lab), ("num", F(x.val, 1)), ("num", F(x.min, 1)),
                   ("num", "—" if x.max is None else F(x.max, 0)), ("st", "OK" if x.ok else "NON OK", x.ok)] for x in R.dist]
        self.table(d, fr, "PINCES ET ENTRAXES — EN 1993-1-8 TAB. 3.3 (mm)", cols, lignes)

    def conclusion(self, d, x, w):
        S, R = self.S, self.R
        txt = (R.statut + " à l'ELU selon EN 1993-1-8 et EN 1993-1-1 (γM1 : ANB belge)"
               + (" — sous réserve des points signalés." if R.verified and (R.reserve or R.alerts) else "."))
        col = S.ok if R.verified else S.ko
        lines = d.wrap(txt, S.f_b, 7.2, w - 15)
        h = len(lines) * 7.2 * 1.28 + 7
        y0 = BAS
        d.box(x, y0, w, h, fill=mix(col, "#FFFFFF", 0.90))
        d.box(x, y0 + 1.5, 2.0, h - 3, fill=col)
        for i, ln in enumerate(lines):
            d.t(x + 9, y0 + h - 7.2 * (1.02 + i * 1.28), ln, S.f_b, 7.2, S.ink)
        return h

    def _bloc_formules(self, d, c, largeur):
        S, R = self.S, self.R
        textes = formules.textes(R, c)
        if self.mode_lignes == "eta" and len(textes) > 1:
            textes = [textes[0], textes[-1]]
        bloc = []
        for t in textes:
            bloc.extend(d.wrap(t, S.f_b, S_FORM, largeur - 6))
        return bloc, 9 + len(bloc) * S_FORM * 1.25 + 3

    def _simuler(self, d, frames, retenues, hauteurs):
        R = self.R
        rooms = [max(0.0, f.room()) for f in frames]
        rooms[0] -= 16
        i = 0
        places, manq = [], []
        for c in [c for c in R.checks if c in retenues]:
            h = hauteurs[(c.key, i)]
            while h > rooms[i] and i + 1 < len(frames):
                i += 1
                h = hauteurs[(c.key, i)]
            if h > rooms[i]:
                manq.append(c)
            else:
                rooms[i] -= h
                places.append(c)
        return places, manq

    def formules(self, d, frames):
        S, R = self.S, self.R
        cand = _priorite(R)
        if not frames or not cand:
            return
        oblig = _obligatoires(R)
        hauteurs = {(c.key, i): self._bloc_formules(d, c, f.w)[1] for c in cand for i, f in enumerate(frames)}
        dispo = sum(max(0.0, f.room()) for f in frames) - 16 - 8
        retenues = []
        total = 0.0
        for c in cand:
            h = max(hauteurs[(c.key, i)] for i in range(len(frames)))
            if total + h <= dispo:
                retenues.append(c); total += h
        while True:
            places, manq = self._simuler(d, frames, retenues, hauteurs)
            if not any(c in oblig for c in manq):
                break
            facultatives = [c for c in retenues if c not in oblig]
            if not facultatives:
                break
            retenues.remove(facultatives[-1])
        for c in manq:
            if c in oblig:
                self.warnings.append("formules : " + c.key + " ne tient pas")
        i = 0
        fr = frames[0]
        self.titre(d, fr, "FORMULES ET SUBSTITUTIONS NUMÉRIQUES")
        fr.down(2)
        from .notation import ref_courte
        for c in places:
            bloc, h = self._bloc_formules(d, c, fr.w)
            while not fr.fits(h) and i + 1 < len(frames):
                i += 1
                fr = frames[i]
            if not fr.fits(h):
                continue
            self.formules_imprimees.append(c.key)
            fr.down(9)
            d.box(fr.x, fr.y - 1.4, 1.6, 6.4, fill=S.ok if c.ok else S.ko)
            lab = synthese.court(c) + ("" if c.nat == "EC" else " [" + c.nat + "]")
            d.t(fr.x + 5, fr.y, lab, "Carlito-Bold", 6.3, S.ink)
            wl = d.w(lab, "Carlito-Bold", 6.3)
            d.t(fr.x + 5 + wl + 4, fr.y, d.fit("— " + ref_courte(c.ref), S.f_b, S_REF, fr.w - wl - 12), S.f_b, S_REF, S.mut)
            for l in bloc:
                fr.down(S_FORM * 1.25)
                d.t(fr.x + 5, fr.y, l, S.f_b, S_FORM, S.ink)
            fr.down(3)

    def construire(self, d):
        S = self.S
        W, H = d.W, d.H
        top = self.bandeau(d)
        w1, w2 = LARG[0], LARG[1]
        w3 = W - 2 * M - w1 - w2 - 2 * GAP
        x1, x2, x3 = M, M + w1 + GAP, M + w1 + GAP + w2 + GAP
        for x in (x2, x3):
            d.line(x - GAP / 2, top, x - GAP / 2, BAS, S.rule, 0.45)
        f1 = Frame(x1, top, w1, BAS)
        f2 = Frame(x2, top, w2, BAS)
        f3 = Frame(x3, top, w3, BAS)
        self.colonne_dessins(d, f1)
        self.table_simple(d, f2, "boulons", (92, 34, 46, 26, w2 - 198))
        self.table_simple(d, f2, "plat", (92, 34, 46, 26, w2 - 198))
        self.table_simple(d, f2, "soudure", (92, 34, 46, 26, w2 - 198))
        self.table_simple(d, f3, "portee", (98, 36, 50, 28, w3 - 212))
        self.table_simple(d, f3, "porteuse", (98, 36, 50, 28, w3 - 212))
        self.table_pinces(d, f3)
        hc = self.conclusion(d, x3, w3)
        f3.bottom = BAS + hc + 6
        self.formules(d, [f2, f3])
        d.t(M, 24, d.fit(LEGENDE_REFERENCES, S.f_b, 5.6, W - 2 * M), S.f_b, 5.6, S.mut)
        d.t(M, 16.5, d.fit(NOTATIONS_COURTES, S.f_b, 5.6, W - 2 * M), S.f_b, 5.6, S.mut)
        for f in (f1, f2, f3):
            if f.overflow() > 1.0:
                self.warnings.append(f"colonne {f.x:.0f} : {f.overflow():.0f} pt")
        return self.warnings


# ------------------------------------------------- page 2 : plan de principe
class PlanPrincipeFP(PlanPrincipe):
    """Le plan de principe du plat d'âme : mêmes échelles normalisées, mêmes
    dispositions et même cartouche que le module doubles cornières — seules
    les VUES et les cases du cartouche sont propres à cet assemblage."""

    def __init__(self, R, doc_meta):
        super().__init__(R, doc_meta)
        self.titre = TITRE

    def _vues(self, dnm):
        f = TEXTE_MM * dnm
        R = self.R
        return [("ÉLÉVATION", schemas.elevation(R, schemas.options_fabrication(schemas.EXCLURE_ELEVATION, f))),
                ("VUE EN PLAN", schemas.plan(R, schemas.options_fabrication(schemas.EXCLURE_PLAN, f))),
                ("VUE DE DROITE", schemas.vue_droite(R, schemas.options_fabrication(schemas.EXCLURE_DROITE, f)))]

    def _cartouche(self, d, dnm):
        from ..doubles_cornieres.note import _fmt_echelle, CART_H
        S, R, doc = self.S, self.R, self.doc
        u = R.u
        W = d.W
        x0, y0, w, h = M, M, W - 2 * M, CART_H
        d.box(x0, y0, w, h, stroke=S.ink, lw=0.9)
        fix = ("Boulons " + R.boulon + " " + u.classe + " – trous Ø" + F(R.d_0, 0)
               + " – " + js_str(R.n_1) + " × " + js_str(R.n_2) + " · soudure double a " + F(R.a_w, 0))
        vis = "par boulon : " + (str(doc.get("visserie") or "").strip() or VISSERIE_DEFAUT)
        ident = " · ".join(str(x) for x in (doc.get("projet"), doc.get("partie")) if x)
        cases = [("ASSEMBLAGE", "Poutre–poutre – plat d'âme soudé", ident or "—", 0.185),
                 ("POUTRES", "P : " + _prof(R, "P") + " " + u.nu_P, "S : " + _prof(R, "S") + " " + u.nu_S, 0.165),
                 ("PLAT", js_str(R.h_p) + "×" + js_str(R.b_p) + "×" + js_str(R.t_p) + " " + u.nu_pl,
                  "zp " + F(N(u.z_C), 0) + " mm", 0.165),
                 ("FIXATIONS", fix, vis, 0.27),
                 ("DATE · INDICE", str(doc.get("date", "")), "indice " + str(doc.get("indice", "")), 0.10),
                 ("ÉCHELLE", _fmt_echelle(dnm), "A4 paysage", 0.115)]
        x = x0
        for i, (lab, l1, l2, part) in enumerate(cases):
            cw = w * part
            if i:
                d.line(x, y0 + 2, x, y0 + h - 2, S.rule, 0.6)
            d.t(x + 6, y0 + h - 10, lab, S.f_v, 5.4, S.mut, track=1.2)
            gras = S.f_h if lab == "ÉCHELLE" else "Carlito-Bold"
            d.t(x + 6, y0 + h - 21, d.fit(str(l1), gras, 7.2 if lab == "ÉCHELLE" else 6.6, cw - 12),
                gras, 7.2 if lab == "ÉCHELLE" else 6.6, S.ink)
            if l2:
                d.t(x + 6, y0 + h - 32, d.fit(str(l2), S.f_b, 6.2, cw - 12), S.f_b, 6.2, S.ink)
            x += cw


def _essai(R, doc_meta, variante, chemin):
    d = Doc(chemin, A4L, title=doc_meta.get("titre", "Note de calcul"))
    d.new_page(A4L)
    n = Note(R, doc_meta, variante)
    w = n.construire(d)
    d.save()
    return list(w)


def generer_note(R, infos=None, chemin=None):
    """La note (bytes) : page 1 de calcul, page 2 « PLAN DE PRINCIPE »."""
    global derniers_avertissements, derniere_variante, derniere_echelle, derniere_disposition
    infos = infos or {}
    doc_meta = ndc_data.construire_doc(infos, date_defaut=datetime.today().strftime("%d/%m/%Y"))
    doc_meta["titre"] = "Note de calcul"
    doc_meta["visserie"] = infos.get("visserie", "")
    if chemin is None:
        fd, chemin = tempfile.mkstemp(suffix=".pdf", prefix="note_plat_ame_")
        os.close(fd)
    retenue = VARIANTES[-1]
    warn = []
    for v in VARIANTES:
        warn = _essai(R, doc_meta, v, chemin)
        retenue = v
        if not warn:
            break
    d = Doc(chemin, A4L, title=doc_meta.get("titre", "Note de calcul"))
    d.new_page(A4L)
    n1 = Note(R, doc_meta, retenue)
    warn = list(n1.construire(d))
    d.new_page(A4L)
    pp = PlanPrincipeFP(R, doc_meta)
    warn += pp.construire(d)
    d.save()
    derniere_variante = retenue
    derniers_avertissements = warn
    derniere_echelle = getattr(pp, "echelle", None)
    derniere_disposition = getattr(pp, "disposition", None)
    with open(chemin, "rb") as fh:
        return fh.read()


derniers_avertissements = []
derniere_variante = None
derniere_echelle = None
derniere_disposition = None

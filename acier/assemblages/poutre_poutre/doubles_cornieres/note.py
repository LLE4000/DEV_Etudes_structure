# -*- coding: utf-8 -*-
"""Note de calcul d'UNE page A4 paysage — la note du bouton « 📄 Générer PDF ».

Même charte que les notes béton (``ndc_pdf``, palette 01_encre, Poppins /
Carlito, chip d'état, encadré de verdict), mêmes primitives ; le corps de
texte n'est jamais réduit sous celui des notes béton (6,0 pt). La page se lit
à trois niveaux :

- 5 secondes  : le bandeau — statut, taux maximal, vérification
                dimensionnante, taux par élément, alertes ;
- 30 secondes : les tableaux par élément physique — boulons, cornières
                (matrice aile A · aile B · max), poutre portée, poutre
                porteuse, cordons, pinces et entraxes (Tableau 3.3) ;
- contrôle    : formules et substitutions numériques (``formules.py``,
                testées), hypothèses, dessins cotés.

Disposition : bandeau + une ligne de données, puis trois colonnes — dessins
et hypothèses ; boulons, cornières, cordons ; poutres, pinces, conclusion —
les formules occupant la place restante en flux. Une boucle d'ajustement
réduit le nombre de formules imprimées (toutes les essentielles → une par
élément → la dimensionnante et les non vérifiées), jamais les tableaux ni
le corps de texte. Le rapport détaillé (``rapport.py``, trois pages) reste
disponible dans l'onglet Note.

Rien n'est recalculé ici : tout vient de ``R``.
"""
import os
import tempfile
from datetime import datetime

from reportlab.lib.pagesizes import A4, landscape

from ndc_pdf.kit import Doc, Frame, chip, mix
from ndc_pdf.styles import Encre
from ndc_pdf import data as ndc_data
from acier.formats import F, pct
from acier.js import N
from acier.bibliotheques import PERSO
from acier.js import js_str, mx
from . import schemas, synthese, formules
from .entrees import VISSERIE_DEFAUT
from .notation import ec, ligne_alerte, LEGENDE_REFERENCES
from .rapport import peindre, peindre_echelle

A4L = landscape(A4)
M = 22                     # marge
BAS = 40                   # bas du corps (au-dessus du pied de page)
GAP = 12
LARG = (210, 262, None)    # colonnes 1 et 2 ; la 3ᵉ prend le reste
S_TAB, S_TAB_H, S_REF, S_FORM, S_DATA = 6.5, 5.6, 5.6, 6.0, 7.0
LEAD_TAB = 8.8

TITRE = "Assemblage poutre–poutre — doubles cornières d'âme"
NOTATIONS_COURTES = ("Notations : EN 1993-1-8 Tab. 3.3 (e1, p1, e2, p2) ; hc, zc : hauteur et position des "
                     "cornières ; c, dc : grugeage ; gh, Δz : jeu et décalage ; gA, p3 : groupe P ; a : gorge ; "
                     "z : excentricité.")
PINCES_COURT = {"S_e1c": "S – e1 cornière (mini)", "S_e2c": "S – e2 cornière, aile B", "S_e1b": "S – e1,b âme portée",
                "S_he": "S – he âme portée", "S_e2b": "S – e2,b âme portée", "S_p1": "S – p1", "S_p2": "S – p2",
                "P_e1c": "P – e1 cornière (mini)", "P_e2c": "P – e2 cornière, aile A", "P_p1": "P – p1", "P_p2": "P – p2"}

# Variantes d'ajustement des formules : lignes imprimées par vérification —
# toutes ; la première (résistance) et la dernière (taux) ; le taux seul. La
# sélection des vérifications est gloutonne par priorité (voir ``_priorite``).
VARIANTES = ("toutes", "extremes", "eta")


# ------------------------------------------------------------- contenu
def _prof(R, X):
    u = R.u
    if u["prof_" + X] == PERSO:
        return "h " + F(R["h_" + X], 0) + " b " + F(R["b_" + X], 0) + " tw " + F(R["tw_" + X], 1) + " tf " + F(R["tf_" + X], 1)
    return u["prof_" + X]


def lignes_entete(R):
    """L'en-tête compact : deux lignes d'objets, puis la ligne des efforts.
    Pas de géométrie détaillée ici — elle est cotée sur le plan de principe
    (page 2)."""
    u = R.u
    grugee = bool(N(u.d_nt) or N(u.d_nb))
    l1 = ("Principale " + _prof(R, "P") + " " + u.nu_P
          + "    |    Secondaire " + _prof(R, "S") + " " + u.nu_S + (" · grugée" if grugee else "")
          + "    |    Cornières 2 × " + R.corn_txt + " " + u.nu_C)
    if R.bolt_S or R.bolt_P:
        b = ("Boulons " + R.boulon + " " + u.classe + " cat. " + u.cat + " · trous Ø" + F(R.d_0, 0)
             + (" surdim." if u.trou != "Normal" else ""))
    else:
        b = "Cornières soudées"
    l2 = (b + "    |    Groupe S " + (js_str(R.n1_S) + " × " + js_str(R.n2_S) if R.bolt_S
                                      else "soudé, a " + F(N(u.a_S), 0))
          + "    |    Groupe P " + ("2 × (" + js_str(R.n1_P) + " × " + js_str(R.n2_P) + ")" if R.bolt_P
                                    else "soudé, a " + F(N(u.a_P), 0)))
    l3 = ("VEd " + F(u.V_Ed, 1) + " kN    |    NEd " + F(u.N_Ed, 1) + " kN    |    HEd "
          + F(u.H_Ed, 1) + " kN    |    MEd " + F(u.M_Ed, 2) + " kNm")
    return l1, l2, l3


def lignes_hypotheses(R):
    """Les hypothèses : une par ligne, courtes, références abrégées."""
    u = R.u
    h = ["Articulé — rotule à la face de l'âme porteuse (MSB P5 §4.2.1.1).",
         "z = " + F(R.zeff, 1) + " mm ; MS = VEd·z + |MEd| = " + F(R.M_S, 2) + " kNm"
         + (" ; Ip,S = " + F(R.Ip_S, 0) + " mm²" if R.bolt_S else "") + "."]
    if R.bolt_P:
        h.append("Groupe P : " + ("MP = " + F(R.M_P, 2) + " kNm par cornière" if R.M_P > 0
                                  else "cisaillement centré, " + F(N(u.k_rot), 2) + "·Fv,Rd")
                 + " (MSB P5 §4.2.1.2).")
    elif R.M_P > 0:
        h.append("Cordons A : MP = " + F(R.M_P, 2) + " kNm par cornière.")
    if R.cas_g > 0:
        h.append("Grugeage : flexion de la section réduite (MSB P5 §4.2.4).")
    h.append("γM0 " + F(N(u.g_M0), 2) + " · γM2 " + F(N(u.g_M2), 2) + " · nettes " + F(N(u.g_M2n), 2)
             + (" · γM3 " + F(N(u.g_M3), 2) if u.cat != "A" else "") + ".")
    if not (R.bolt_S and R.bolt_P):
        h.append("Cordons : Leff = longueur totale (EC3 §4.5.1).")
    return h


def _priorite(R):
    """Les vérifications candidates aux formules, par priorité : la
    dimensionnante, les non vérifiées, la plus sollicitée de chaque élément,
    puis les autres essentielles (ordre du moteur)."""
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
    """Ce qui doit tenir : dimensionnante, non vérifiées, une par élément."""
    n = (1 if R.gov else 0) + sum(1 for c in R.checks if c.active and not c.ok) + len(synthese.taux_par_element(R))
    return _priorite(R)[:n]


# ------------------------------------------------------------- dessin
class Note:
    """Compose la page ; ``warnings`` = débordements (vide si tout tient)."""

    def __init__(self, R, doc_meta, variante):
        self.R = R; self.doc = doc_meta; self.mode_lignes = variante
        self.S = Encre(); self.warnings = []; self.formules_imprimees = []

    # --- briques
    def titre(self, d, fr, txt):
        S = self.S
        fr.down(10)
        d.t(fr.x, fr.y, txt, S.f_v, 6.2, S.acc, track=1.3)
        fr.down(3)
        d.line(fr.x, fr.y, fr.x1, fr.y, mix(S.rule, S.ink, 0.12), 0.5)

    def table(self, d, fr, titre, cols, lignes):
        """``cols`` = [(entête, largeur, alignement)] ; ``lignes`` = listes
        de cellules ``(style, texte[, couleur])`` — style : lab, num, ref,
        eta ; une cellule texte se replie (lab, ref) ou se cale (num)."""
        S = self.S
        self.titre(d, fr, titre)
        fr.down(8)
        xs = []
        x = fr.x
        for ent, w, al in cols:
            xs.append(x)
            # capitales latines seulement : « η » reste « η »
            d.t(x + (w - 3 if al == "right" else 0), fr.y, ent.upper() if ent != "η" else ent, S.f_b, S_TAB_H, S.mut, al)
            x += w
        fr.down(2.5)
        d.line(fr.x, fr.y, fr.x1, fr.y, S.acc, 0.5)
        for cells in lignes:
            # hauteur de la ligne : la cellule la plus repliée
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
                elif style == "txt":
                    d.t(xx, y, d.fit(ls[0], S.f_b, S_TAB, w - 4), S.f_b, S_TAB, S.ink)
            fr.down(h - (LEAD_TAB - 1.8))
            d.line(fr.x, fr.y + 1.5, fr.x1, fr.y + 1.5, mix(S.rule, S.ink, 0.06), 0.4)
        fr.down(2)

    # --- bandeau
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
        # statut · taux maximal · dimensionnante ··· taux par élément
        y -= 15
        etat_ok = R.verified
        w = chip(d, M, y - 3.2, "VÉRIFIÉ" if etat_ok else "NON VÉRIFIÉ", S.f_v, 6.6, "#FFFFFF", S.ok if etat_ok else S.ko)
        x = M + w + 8
        t1 = "η max " + pct(R.eta_max, 1)
        d.t(x, y, t1, "Carlito-Bold", 8.6, S.ok if etat_ok else S.ko)
        x += d.w(t1, "Carlito-Bold", 8.6) + 5
        t2 = ("— dimensionnant : " + synthese.court(R.gov) + " (" + synthese.lignes_ref(R.gov) + ")") if R.gov else ""
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
        # alertes (trois au plus)
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
            d.t(M + 9, y, "+ " + str(len(al) - 3) + " autre(s) point(s) signalé(s) — voir le rapport détaillé.", S.f_b, 6.4, S.mut)
        # en-tête compact : deux lignes d'objets, puis la ligne des efforts
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

    # --- colonne 1 : dessins et hypothèses
    def colonne_dessins(self, d, fr):
        # les dessins sont rendus SANS cartouche (options_rapport) : ses
        # lignes répéteraient la ligne de données et les hypothèses
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

    # --- tableaux
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

    def table_cornieres(self, d, fr):
        """La matrice aile A · aile B · max ; les références, communes aux
        deux ailes, sur une ligne sous la matrice."""
        S, R = self.S, self.R
        ls = synthese.matrice_cornieres(R)
        if not ls:
            return
        w = fr.w
        cols = [("Vérification", w - 176, "left"), ("Aile A : Ed / Rd", 52, "right"), ("η", 24, "right"),
                ("Aile B : Ed / Rd", 52, "right"), ("η", 24, "right"), ("max", 24, "right")]
        lignes = []
        refs = []
        for l in ls:
            unite = (l["A"] or l["B"])["unit"]
            cells = [("lab", l["lab"] + (" [" + unite + "]" if unite else ""), "gov" if l["max"]["c"] is R.gov else "")]
            for x in (l["A"], l["B"]):
                if x:
                    cells.append(("num", x["Ed"] + " / " + x["Rd"]))
                    cells.append(("eta", x["pct"], x["ok"], x["c"] is R.gov))
                else:
                    cells.append(("num", "—")); cells.append(("num", ""))
            cells.append(("eta", l["max"]["pct"], l["max"]["ok"], l["max"]["c"] is R.gov))
            lignes.append(cells)
            refs.append(l["lab"][0].lower() + l["lab"][1:] + " " + synthese.refs_fusionnees([x["ref"] for x in (l["A"], l["B"]) if x]))
        self.table(d, fr, "CORNIÈRES (AILE A : PRINCIPALE · AILE B : SECONDAIRE)", cols, lignes)
        for ln in d.wrap("Réf. : " + " ; ".join(refs) + ".", S.f_b, S_REF, w):
            fr.down(S_REF * 1.25)
            d.t(fr.x, fr.y, ln, S.f_b, S_REF, S.mut)
        fr.down(2)

    def table_pinces(self, d, fr):
        R = self.R
        if not R.dist:
            return
        w = fr.w
        cols = [("Distance (Tab. 3.3)", w - 128, "left"), ("Valeur", 32, "right"), ("Min", 32, "right"),
                ("Max", 32, "right"), ("", 32, "right")]
        lignes = [[("lab", PINCES_COURT.get(x.id, x.lab)), ("num", F(x.val, 1)), ("num", F(x.min, 1)),
                   ("num", "—" if x.max is None else F(x.max, 0)), ("st", "OK" if x.ok else "NON OK", x.ok)] for x in R.dist]
        self.table(d, fr, "PINCES ET ENTRAXES — EN 1993-1-8 TAB. 3.3 (mm)", cols, lignes)

    # --- conclusion
    def conclusion(self, d, x, w):
        S, R = self.S, self.R
        # le taux maximal et la dimensionnante sont déjà dans le bandeau :
        # la conclusion ne les répète pas
        txt = (R.statut + " à l'ELU selon EN 1993-1-8 et EN 1993-1-1"
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

    # --- formules en flux
    def _bloc_formules(self, d, c, largeur):
        """Les lignes repliées d'une vérification et la hauteur du bloc."""
        S, R = self.S, self.R
        L = formules.lignes(R, c)
        if self.mode_lignes == "extremes" and len(L) > 2:
            L = [L[0], L[-1]]
        elif self.mode_lignes == "eta" and len(L) > 1:
            L = [L[-1]]
        textes = [formules.rendre(s) for ligne in L for s in ligne] if L else formules.textes(R, c)
        bloc = []
        for t in textes:
            bloc.extend(d.wrap(t, S.f_b, S_FORM, largeur - 6))
        return bloc, 9 + len(bloc) * S_FORM * 1.25 + 3

    def _simuler(self, d, frames, retenues, hauteurs):
        """Place les blocs retenus dans l'ordre du moteur, colonne après
        colonne : ``(placés, manquants)``."""
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
        """Sélection gloutonne par priorité dans la place restante (simulée
        colonne par colonne), puis impression dans l'ordre du moteur ; ce qui
        ne tient pas renvoie au rapport détaillé. Les obligatoires
        (dimensionnante, non vérifiées, une par élément) qui ne tiennent
        pas sont des débordements."""
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
            # une obligatoire ne tient pas : on retire la moins prioritaire des facultatives
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
            ref = synthese.lignes_ref(c)
            d.t(fr.x + 5 + wl + 4, fr.y, d.fit("— " + ref, S.f_b, S_REF, fr.w - wl - 12), S.f_b, S_REF, S.mut)
            for l in bloc:
                fr.down(S_FORM * 1.25)
                d.t(fr.x + 5, fr.y, l, S.f_b, S_FORM, S.ink)
            fr.down(3)

    # --- page
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
        self.table_cornieres(d, f2)
        self.table_simple(d, f2, "cordons", (92, 34, 46, 26, w2 - 198))
        self.table_simple(d, f3, "portee", (98, 36, 50, 28, w3 - 212))
        self.table_simple(d, f3, "porteuse", (98, 36, 50, 28, w3 - 212))
        self.table_pinces(d, f3)
        hc = self.conclusion(d, x3, w3)
        f3.bottom = BAS + hc + 6
        self.formules(d, [f2, f3])
        # pied de page : références et notations, une ligne chacune
        d.t(M, 24, d.fit(LEGENDE_REFERENCES, S.f_b, 5.6, W - 2 * M), S.f_b, 5.6, S.mut)
        d.t(M, 16.5, d.fit(NOTATIONS_COURTES, S.f_b, 5.6, W - 2 * M), S.f_b, 5.6, S.mut)
        for f in (f1, f2, f3):
            if f.overflow() > 1.0:
                self.warnings.append(f"colonne {f.x:.0f} : {f.overflow():.0f} pt")
        return self.warnings


# ------------------------------------------------- page 2 : plan de principe
# Échelles normalisées testées, de la plus grande à la plus petite ; la
# première où une disposition tient est retenue (jamais « trop petit pour
# être sûr que ça rentre » : on descend seulement si ça ne tient pas)
ECHELLES = (1, 2, 2.5, 5, 10, 15, 20, 25, 30, 40, 50)
MM = 72 / 25.4                # points par millimètre à l'échelle 1:1
TITRE_VUE = 13                # bande de titre au-dessus de chaque vue
GAP_VUES = 9
CART_H = 44                   # cartouche compact en pied de page
TEXTE_MM = 2.4                # hauteur du texte des cotes SUR LE PAPIER —
#                               constante quelle que soit l'échelle : les
#                               vues sont reconstruites pour chaque échelle
#                               candidate avec fs = TEXTE_MM × dénominateur


def _fmt_echelle(dnm):
    return "1:" + (js_str(int(dnm)) if dnm == int(dnm) else F(dnm, 1))


class PlanPrincipe:
    """La page « PLAN DE PRINCIPE » : élévation, vue en plan et vue de
    droite, toutes à la MÊME échelle normalisée, choisie automatiquement
    comme la plus grande qui fait tenir les trois vues cotées ; la
    disposition est choisie parmi plusieurs pour remplir la feuille."""

    def __init__(self, R, doc_meta):
        self.R = R; self.doc = doc_meta; self.S = Encre(); self.warnings = []
        self.titre = TITRE                # un autre assemblage met le sien
        self.vues = self._vues(5)

    def _vues(self, dnm):
        """Les trois vues pour le dénominateur ``dnm`` : la police des cotes
        est imposée pour que le texte imprimé fasse TEXTE_MM sur les trois
        vues, quelle que soit l'échelle."""
        f = TEXTE_MM * dnm
        R = self.R
        return [("ÉLÉVATION", schemas.elevation(R, schemas.options_fabrication(schemas.EXCLURE_ELEVATION, f))),
                ("VUE EN PLAN", schemas.plan(R, schemas.options_fabrication(schemas.EXCLURE_PLAN, f))),
                ("VUE DE DROITE", schemas.vue_droite(R, schemas.options_fabrication(fs_force=f)))]

    # --- choix de l'échelle et de la disposition
    def _dispositions(self, t, zw, zh):
        """Les dispositions candidates pour les tailles ``t[i] = (w, h)``
        (titre compris) : liste de ``(nom, [(i, x, y), …])`` en coordonnées
        zone (origine en haut à gauche), ou None si ça ne tient pas."""
        (w0, h0), (w1, h1), (w2, h2) = t
        g = GAP_VUES
        out = []

        def centre(nom, wt, ht, slots):
            if wt <= zw and ht <= zh:
                dx, dy = (zw - wt) / 2, (zh - ht) / 2
                out.append((nom, [(i, x + dx, y + dy) for i, x, y in slots]))

        cw = mx(w0, w1)
        centre("élévation et plan à gauche, droite à droite", cw + g + w2, mx(h0 + g + h1, h2),
               [(0, (cw - w0) / 2, 0), (1, (cw - w1) / 2, h0 + g), (2, cw + g, (mx(h0 + g + h1, h2) - h2) / 2)])
        cw = mx(w1, w2)
        centre("élévation à gauche, plan et droite à droite", w0 + g + cw, mx(h0, h1 + g + h2),
               [(0, 0, (mx(h0, h1 + g + h2) - h0) / 2), (1, w0 + g + (cw - w1) / 2, 0), (2, w0 + g + (cw - w2) / 2, h1 + g)])
        cw = mx(w0, w2)
        centre("élévation et droite à gauche, plan à droite", cw + g + w1, mx(h0 + g + h2, h1),
               [(0, (cw - w0) / 2, 0), (2, (cw - w2) / 2, h0 + g), (1, cw + g, (mx(h0 + g + h2, h1) - h1) / 2)])
        lb = mx(h1, h2)
        centre("élévation en tête, plan et droite dessous", mx(w0, w1 + g + w2), h0 + g + lb,
               [(0, (mx(w0, w1 + g + w2) - w0) / 2, 0), (1, 0, h0 + g + (lb - h1) / 2), (2, w1 + g, h0 + g + (lb - h2) / 2)])
        hb = mx(h0, h1, h2)
        centre("trois vues côte à côte", w0 + g + w1 + g + w2, hb,
               [(0, 0, (hb - h0) / 2), (1, w0 + g, (hb - h1) / 2), (2, w0 + g + w1, (hb - h2) / 2)])
        return out

    def choisir(self, zw, zh):
        """``(dénominateur, échelle pt/mm, nom, slots)`` — la plus grande
        échelle normalisée qui tient (vues reconstruites à chaque essai,
        texte imprimé constant), puis la disposition qui remplit le mieux la
        feuille (aire du rectangle englobant)."""
        for dnm in ECHELLES:
            self.vues = self._vues(dnm)
            s = MM / dnm
            t = [(v.viewbox[2] * s, v.viewbox[3] * s + TITRE_VUE) for _, v in self.vues]
            cand = self._dispositions(t, zw, zh)
            if cand:
                def aire(c):
                    xs = [x for i, x, y in c[1]] + [x + t[i][0] for i, x, y in c[1]]
                    ys = [y for i, x, y in c[1]] + [y + t[i][1] for i, x, y in c[1]]
                    return (max(xs) - min(xs)) * (max(ys) - min(ys))
                nom, slots = max(cand, key=aire)
                return dnm, s, nom, slots, t
        return None

    # --- rendu
    def construire(self, d):
        S, R, doc = self.S, self.R, self.doc
        W, H = d.W, d.H
        y = H - M - 4
        d.t(M, y, doc.get("bureau", ""), S.f_v, 8.0, S.acc, track=1.4)
        d.t(W - M, y, self.titre, S.f_b, 7.6, S.mut, "right")
        y -= 16
        d.t(M, y, "PLAN DE PRINCIPE", S.f_h, 13.5, S.ink)
        y -= 7
        d.line(M, y, W - M, y, S.ink, 1.0)
        zone_y1 = M + CART_H + 8
        zone_h = y - 8 - zone_y1
        zone_w = W - 2 * M
        choix = self.choisir(zone_w, zone_h)
        if choix is None:                       # ne devrait pas arriver (1:50)
            self.warnings.append("plan de principe : aucune échelle normalisée ne tient")
            return self.warnings
        dnm, s, nom, slots, t = choix
        self.echelle = dnm; self.disposition = nom
        d.t(W - M, y + 9, "Échelle " + _fmt_echelle(dnm), S.f_h, 10.5, S.acc, "right")
        for i, x, yy in slots:
            titre, vue = self.vues[i]
            x0 = M + x
            y_top = zone_y1 + zone_h - yy       # haut du bloc (canevas : origine en bas)
            d.t(x0 + t[i][0] / 2, y_top - 8, titre, S.f_v, 6.4, S.acc, "center", track=1.3)
            peindre_echelle(d, vue, x0, y_top - t[i][1], t[i][0], t[i][1] - TITRE_VUE, s)
        self._cartouche(d, dnm)
        return self.warnings

    def _cartouche(self, d, dnm):
        """Cartouche compact : assemblage, poutres, cornières, boulons ou
        cordons, date · indice, échelle."""
        S, R, doc = self.S, self.R, self.doc
        u = R.u
        W = d.W
        x0, y0, w, h = M, M, W - 2 * M, CART_H
        d.box(x0, y0, w, h, stroke=S.ink, lw=0.9)
        if R.bolt_S or R.bolt_P:
            fix = ("Boulons " + R.boulon + " " + u.classe + " – trous Ø" + F(R.d_0, 0)
                   + " – S " + js_str(R.n1_S) + " × " + js_str(R.n2_S) + " · P 2 × (" + js_str(R.n1_P) + " × " + js_str(R.n2_P) + ")"
                   if R.bolt_S and R.bolt_P else
                   "Boulons " + R.boulon + " " + u.classe + " – trous Ø" + F(R.d_0, 0))
            if not R.bolt_S:
                fix += " · ailes B soudées a " + F(N(u.a_S), 0)
            if not R.bolt_P:
                fix += " · ailes A soudées a " + F(N(u.a_P), 0)
        else:
            fix = "Soudures : ailes A a " + F(N(u.a_P), 0) + " · ailes B a " + F(N(u.a_S), 0) + " (retours " + F(N(u.lh_S), 0) + ")"
        # visserie (rondelles, écrous) : annotation de fabrication saisie à
        # l'écran, une par boulon — seulement s'il y a des boulons
        vis = ("par boulon : " + (str(doc.get("visserie") or "").strip() or VISSERIE_DEFAUT)
               if (R.bolt_S or R.bolt_P) else "")
        ident = " · ".join(str(x) for x in (doc.get("projet"), doc.get("partie")) if x)
        cases = [("ASSEMBLAGE", "Poutre–poutre – doubles cornières d'âme", ident or "—", 0.185),
                 ("POUTRES", "P : " + _prof(R, "P") + " " + u.nu_P, "S : " + _prof(R, "S") + " " + u.nu_S, 0.165),
                 ("CORNIÈRES", "2 × " + R.corn_txt + " " + u.nu_C, "hc " + F(R.L_C, 0) + " · zc " + F(N(u.z_C), 0) + " mm", 0.165),
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
    """La note (bytes) : page 1 de calcul (la première variante de formules
    sans débordement est retenue), page 2 « PLAN DE PRINCIPE » (trois vues à
    la même échelle normalisée, la plus grande qui tient)."""
    global derniers_avertissements, derniere_variante, derniere_echelle, derniere_disposition
    infos = infos or {}
    doc_meta = ndc_data.construire_doc(infos, date_defaut=datetime.today().strftime("%d/%m/%Y"))
    doc_meta["titre"] = "Note de calcul"
    doc_meta["visserie"] = infos.get("visserie", "")
    if chemin is None:
        fd, chemin = tempfile.mkstemp(suffix=".pdf", prefix="note_assemblage_")
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
    pp = PlanPrincipe(R, doc_meta)
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

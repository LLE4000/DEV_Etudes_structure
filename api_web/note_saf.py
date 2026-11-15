# -*- coding: utf-8 -*-
"""La NOTE DE CALCUL PDF du « Générateur SAF — Fondation de grue » (api_web/saf.py) — lot 5, cahier des charges du
15/11/2026, § 10 : une VÉRIFICATION COMPLÉMENTAIRE DE SCIA, PAR UN CALCUL INDÉPENDANT (ce n'est pas le même moteur). Elle
donne tout ce qu'il faut pour comparer les deux : mêmes cas, mêmes combinaisons, mêmes charges en kN/m², mêmes bandes, et
un tableau module / SCIA ; la note SCIA est jointe en annexe.

Les pages (§ 10.2), le kit des notes du site (``ndc_pdf.styles``) :
  1. la garde (A4 portrait) — un sommaire en DEUX parties : la vérification complémentaire (une seule pastille, l'état
     global ; ses chapitres en petit, sans pastille) et l'annexe « Note SCIA » (jointe ou à joindre) ;
  2. Données — la SEULE page avec la bande grise : le plan coté et les données clés à gauche ; la dalle et le sol, la grue,
     les réactions, les bandes, le volume de béton ;
  3. Charges et combinaisons — LC1, les cas (R et q = R ÷ aire de la surface de charge : les lignes FF du SAF), les surfaces
     de charge, les combinaisons ELS-Car (auto)n, le sol pour SCIA (C1z, C1x, C1y, C2x, C2y par k) — LUS DANS LES FEUILLES
     DU SAF (``ecriture.feuilles``) : une seule source ;
  4. Sol — trois lignes d'hypothèses, la carte enveloppe, le tableau par k, le tableau de TOUS les cas pour chaque k coché,
     σ moyen, l'équilibre ;
  5. Efforts — les diagrammes enveloppes M(s) et V(s) des quatre bandes (le treillis en trait interrompu), le tableau des
     bandes (le cas et le k de chaque valeur), les deux cartes enveloppes des moments au point avec la ligne à m_Rd ;
  6. Principe d'armatures — les plans du bas et du haut, le tableau des renforts, les contrôles du moteur Dalles (une
     ligne par direction quand la dalle est symétrique), « Principe d'armatures, pas un plan d'exécution. » ;
  7. Comparaison avec SCIA (puis l'annexe sol quand les k viennent du module de raideur du sol) ;
  Annexe — les notes SCIA jointes (``pdf_reunion.reunir``), à la fin du PDF.

Sans bande grise, une page prend toute la largeur ; un chapitre trop long continue sur la page suivante ; un titre reste
avec son tableau ; un tableau trop long se poursuit dans la colonne suivante, son en-tête redit. Diagrammes et cartes :
texte de 7 pt au moins une fois imprimés (§ 10.3) — chaque dessin est posé à la taille qui le garantit.

Ce fichier ne calcule rien : le calcul est celui de l'écran (``saf.calculer``, ``saf.efforts``, ``saf.armatures``,
``saf.scia``), les phrases celles de l'écran (``saf_resultat``, ``saf_armatures``), les dessins ceux de ``dessin_saf``,
``dessin_armatures`` et ``dessin_efforts_saf``.

    sections(etat) -> [planche]        les planches (les dictionnaires que ``Planche.planche`` dessine)
    pdf(etat, infos) -> (octets, avis) infos = {nom, chantier, indice, date, annexes: [{nom, base64}]}
    fichier(etat, infos) -> {"nom", "type": "application/pdf", "base64"[, "avis"]}
"""
import base64
import io
import os
import re
import tempfile
import xml.etree.ElementTree as ET_
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from ndc_pdf import data as ndc_data
from ndc_pdf.kit import Doc, Frame, chip, mix, table
from ndc_pdf.styles import Encre
from ndc_pdf.svg import peindre_svg
from saf.fondation_grue import ecriture as EC
from saf.fondation_grue import efforts as EF
from saf.fondation_grue import geometrie as G
from saf.fondation_grue import projet as PR
from saf.fondation_grue import scia as SCM
from saf.fondation_grue import verification as V

from . import commun as C
from . import dessin_efforts_saf as DE
from . import dessin_saf as DS
from . import pdf_reunion as PDF
from . import saf_armatures as SA
from . import saf_resultat as SR
from . import saf_saisie as SS

TITRE = "Fondation de grue"
PARTIE_1 = "Vérification complémentaire — calcul indépendant de SCIA"
PARTIE_2 = "Annexe — Note SCIA"
MARGE_DESSIN = 0.0
TEXTE_MINI = 7.0                  # pt : le plus petit texte d'un diagramme ou d'une carte, imprimé (§ 10.3)
COULEURS = {"p": "ok", "f": "att", "r": "ko", "n": "att", "x": "ko", "v": "att"}
MOTS = {"p": "Conforme", "f": "Conforme, à la limite", "r": "Non conforme", "n": "Données incomplètes", "x": "Calcul impossible",
        "v": "À vérifier"}
LEGENDE = "Vert : sous la limite   ·   Ocre : à la limite (95 à 100 %), ou à compléter   ·   Rouge : au-dessus de la limite"
ORIGINES = {"mesure": "≈ ", "calcule": "calc. "}
PLOTS = {"aucun": "aucun", "prefabrique": "préfabriqués", "coule": "coulés en place"}
FACES = {"inf": "bas", "sup": "haut"}
N = " "


def _n(x: float, nd: int = 2) -> str:
    return C.fr(x, nd).replace(C.NBSP, " ").replace(C.FINE, " ")


def _q(x: float) -> str:
    """Une charge répartie comme le SAF l'écrit (au millième), sans zéros inutiles : « 1 400 », « 243,056 », « 8,75 »."""
    r = round(float(x), 3)
    nd = 0 if r == int(r) else len(repr(r).split(".")[1])
    return _n(r, min(nd, 3))


def _sol(el: Dict, kpa: float, unite: bool = True) -> str:
    """Une contrainte de sol dans l'unité choisie (la même que l'écran, jamais deux unités mélangées)."""
    t = SS.sol(el, kpa) if unite else SS.sol_nb(el, kpa)
    return t.replace(C.NBSP, " ").replace(C.FINE, " ")


def _texte(h: str) -> str:
    """Un texte de l'écran (HTML court) en texte de la note."""
    import html as H
    return H.unescape(re.sub(r"<br\s*/?>", " · ", re.sub(r"<(?!br)[^>]+>", "", h))).replace(C.NBSP, " ").replace(C.FINE, " ")


def _v(label: str, valeur: str, unite: str = "", symbole: str = "") -> tuple:
    return ("v", label, symbole, valeur, unite)


# ------------------------------------------------------------------ la planche : sans bande grise, des figures, des tableaux longs
class Planche(Encre):
    """La planche des notes du site, pour cette note : elle se poursuit sur une page suivante quand elle déborde ; sans
    bande grise (``sec["bande"]`` faux), ses colonnes prennent toute la largeur ; des FIGURES (« fig ») dans le flux ; un
    titre ne reste jamais seul en bas d'une colonne ; un tableau trop long se poursuit dans la colonne suivante, son en-tête
    redit ; la garde en deux parties (§ 10.4). Le texte des tableaux et des paragraphes : 7 pt."""
    planche_suite = True
    LEGENDE_GARDE = LEGENDE
    s_tab, lead_tab = 7.0, 9.2
    s_lab = 7.0
    S_ENT = 6.4                    # l'en-tête des tableaux

    def coul_etat(self, sec):
        return {"ok": self.ok, "att": self.att, "ko": self.ko}[sec.get("couleur", "ok")]

    def largeur_colonne(self, sec: Optional[Dict] = None) -> float:
        sec = sec or {}
        n = sec.get("n_cols", self.n_cols)
        if sec.get("bande", True):
            zx = self.margin + sec.get("coupe_w", self.coupe_w) + self.gap
        else:
            zx = self.margin
        return (self.page[0] - self.margin - zx - self.col_gap * (n - 1)) / n

    def hauteur_colonne(self) -> float:
        return self.page[1] - 2 * self.margin - 40.0

    def _page_planche(self, d, doc, sec, ms, suite=False):
        if sec.get("bande", True):
            return super()._page_planche(d, doc, sec, ms, suite)
        S = self
        W, H = d.W, d.H
        m = S.margin
        y = H - m - 2
        d.t(m, y, doc.get("bureau", ""), S.f_v, 8.4, S.ink)
        d.t(W - m, y, f"{doc.get('date', '')} · indice {doc.get('indice', '')}", S.f_b, 8, S.mut, "right")
        wc = 0
        if sec.get("etat"):
            wc = chip(d, W - m, y - 20, sec["etat"].upper(), S.f_v, 6.6, "#FFFFFF", self.coul_etat(sec), align="right")
        mat = f"{sec.get('beton', '')} / {sec.get('acier', '')}".strip(" /")
        wm = 0
        if mat:
            d.t(W - m - wc - 8, y - 16, mat, S.f_v, 8, S.ink, "right")
            wm = d.w(mat, S.f_v, 8) + 16
        # (le titre de la page — les chapitres qu'elle porte — s'écrit quand la page est finie : ``_ecrire_entete``)
        self._entete = (m, y - 18, W - 2 * m - wc - wm - 8)
        d.line(m, y - 28, W - m, y - 28, S.ink, 1.0)
        top = y - 38
        n_cols = sec.get("n_cols", S.n_cols)
        gw = (W - 2 * m - S.col_gap * (n_cols - 1)) / n_cols
        cols = [Frame(m + i * (gw + S.col_gap), top, gw, m) for i in range(n_cols)]
        for fr in cols[1:]:
            d.line(fr.x - S.col_gap / 2, top + 4, fr.x - S.col_gap / 2, m + 2, S.rule, 0.45)
        return Frame(m, top, 0, m), cols

    # -------------------------------------------------- le flux
    def _place(self, d, fr, v, besoin, cols, suivante):
        """La place pour ``besoin`` pt : la colonne suivante, ou la page suivante (avec le rappel « (suite) »)."""
        if not fr.fits(besoin):
            if cols.index(fr) + 1 < len(cols):
                fr = cols[cols.index(fr) + 1]
                self._suite(d, fr, v)
            elif suivante is not None and not fr.fits(besoin - self.debord_admis):
                fr = suivante()
                self._suite(d, fr, v)
        return fr

    def _hauteur_tab(self, fr, cols_, lignes, note=None) -> float:
        brouillon = Doc(io.BytesIO(), self.page)
        brouillon.new_page(self.page)
        f0 = Frame(fr.x, 1.0e6, fr.w, -1.0e9)
        table(brouillon, f0, cols_, lignes, self, self.s_tab, self.lead_tab, s_ent=self.S_ENT, ms=self._ms_col, note=note)
        return 8.0 + f0.top - f0.min_y

    def _hauteur_fig(self, fr, it) -> float:
        w, h = _taille_fig(it[1], fr.w, it[3] if len(it) > 3 else None)
        return h + (self.s_lab * 1.6 if it[2] else 0.0) + 6.0

    def hauteur_premier(self, fr, v) -> float:
        """La hauteur du premier élément d'une vérification (pour garder son titre avec lui) : un tableau (son en-tête et
        trois lignes au moins), une figure."""
        if not v or not v["items"]:
            return 0.0
        it = v["items"][0]
        if it[0] == "tab":
            return self._hauteur_tab(fr, it[1], it[2][:3])
        if it[0] == "fig":
            return self._hauteur_fig(fr, it)
        return 24.0

    CHAPITRE = 12.0               # pt : le titre d'un chapitre dans le flux

    def _ecrire_entete(self, d, sec, chapitres: List[str], suite: bool):
        """Le titre d'une page sans bande : « Fondation de grue — Sol · Efforts » (les chapitres qu'elle porte)."""
        x, y, w = self._entete
        ch = list(chapitres) or [sec["section"]]
        if suite:
            ch[0] += " (suite)"
        t = f"{sec['poutre']} — {' · '.join(ch)}"
        taille = 14.0
        if d.w(t, self.f_h, taille) > w > 0:
            taille = max(9.0, taille * w / d.w(t, self.f_h, taille))
        d.t(x, y, t, self.f_h, taille, self.ink)

    def _chapitre(self, d, fr, nom: str):
        fr.down(self.CHAPITRE + 6.0)
        d.t(fr.x, fr.y, nom.upper(), self.f_v, self.CHAPITRE * 0.78, self.acc, track=1.2)
        fr.down(4.0)
        d.line(fr.x, fr.y, fr.x1, fr.y, self.acc, 0.9)
        fr.down(2.0)

    def planche(self, d, doc, sec):
        """La planche ; sans bande grise, un FLUX DE CHAPITRES (§ 10.2 : un chapitre trop long continue sur la page suivante,
        un chapitre court partage la page avec le suivant) — ``sec["_chapitres"]`` : [(chapitre, page de la planche)]."""
        S = self
        ms = S.ms()
        bande = sec.get("bande", True)
        C_, cols = self._page_planche(d, doc, sec, ms)
        etat = {"page": 1, "sur_page": [], "suite": False, "en_cours": None}
        sec["_chapitres"] = []
        suivante = None
        if S.planche_suite:
            def suivante():
                if not bande:
                    self._ecrire_entete(d, sec, etat["sur_page"], etat["suite"])
                d.new_page(S.page)
                etat["page"] += 1
                etat["suite"] = True
                etat["sur_page"] = [etat["en_cours"]] if etat["en_cours"] else []
                cols[:] = self._page_planche(d, doc, sec, ms, suite=True)[1]
                return cols[0]
        fr = cols[0]
        vs = sec.get("verifs", [])
        for i, v in enumerate(vs):
            if v.get("chapitre"):
                suivant = next((x for x in vs[i + 1:] if not x.get("chapitre")), None)
                besoin = self.CHAPITRE + 12.0 + (S.title_gap + S.s_tit + self.hauteur_premier(fr, suivant) if suivant else 0.0)
                besoin = min(max(besoin, 80.0), self.hauteur_colonne())
                if not fr.fits(besoin):
                    if cols.index(fr) + 1 < len(cols):
                        fr = cols[cols.index(fr) + 1]
                    elif suivante is not None:
                        etat["en_cours"] = None
                        fr = suivante()
                etat["en_cours"] = v["chapitre"]
                if not etat["sur_page"]:
                    etat["suite"] = False
                etat["sur_page"].append(v["chapitre"])
                sec["_chapitres"].append((v["chapitre"], etat["page"]))
                self._chapitre(d, fr, v["chapitre"])
                continue
            besoin = max(58.0, S.title_gap + S.s_tit + self.hauteur_premier(fr, v))
            besoin = min(besoin, self.hauteur_colonne())
            if not fr.fits(besoin):
                if cols.index(fr) + 1 < len(cols):
                    fr = cols[cols.index(fr) + 1]
                elif suivante is not None:
                    fr = suivante()
            self._titre(d, fr, v)
            fr = self._items(d, fr, v, ms, cols, suivante)
        if not bande:
            self._ecrire_entete(d, sec, etat["sur_page"], etat["suite"])
        nom = f"{sec['poutre']} — {sec['section']}"
        for f in [C_] + cols:
            if f.overflow() > S.debord_admis:
                d.warnings.append(f"{nom} : {f.overflow():.0f} pt")
        return d

    def _items(self, d, fr, v, ms, cols, suivante=None):
        paquet = []

        def vider(fr_):
            if paquet:
                fr_ = super(Planche, self)._items(d, fr_, dict(v, items=list(paquet)), ms, cols, suivante)
                paquet.clear()
            return fr_
        for it in v["items"]:
            if it[0] == "fig":
                fr = vider(fr)
                fr = self._figure(d, fr, v, it, cols, suivante)
            elif it[0] == "tab":
                fr = vider(fr)
                fr = self._tableau(d, fr, v, it, cols, suivante)
            else:
                paquet.append(it)
        return vider(fr)

    def _figure(self, d, fr, v, it, cols, suivante):
        """Une figure (« fig », svg, légende ou None[, hauteur maxi]) à la largeur de la colonne, centrée."""
        svg, leg = it[1], it[2]
        fr = self._place(d, fr, v, self._hauteur_fig(fr, it), cols, suivante)
        w, h = _taille_fig(svg, fr.w, it[3] if len(it) > 3 else None)
        fr.down(4.0)
        if leg:
            fr.down(self.s_lab * 1.1)
            d.t(fr.x, fr.y, leg, self.f_v, self.s_lab, self.acc, track=0.6)
            fr.down(self.s_lab * 0.5)
        fr.down(h)
        peindre_svg(d.c, svg, fr.x + (fr.w - w) / 2, fr.y, w, h, marge=MARGE_DESSIN, riche=True)
        fr.down(2.0)
        return fr

    def _tableau(self, d, fr, v, it, cols, suivante):
        """Un tableau ; trop long pour la place qui reste, il se poursuit dans la colonne suivante, son en-tête redit."""
        cols_, lignes = it[1], list(it[2])
        note = it[3] if len(it) > 3 else None
        while True:
            h_tout = self._hauteur_tab(fr, cols_, lignes, note)
            if fr.fits(h_tout) or self._hauteur_tab(fr, cols_, lignes[:1]) > self.hauteur_colonne():
                fr.down(8.0)
                table(d, fr, cols_, lignes, self, self.s_tab, self.lead_tab, s_ent=self.S_ENT, ms=self._ms_col, note=note)
                return fr
            # combien de lignes tiennent ici (trois au moins, sinon la colonne suivante)
            n = 0
            for k in range(1, len(lignes)):
                if fr.fits(self._hauteur_tab(fr, cols_, lignes[:k])):
                    n = k
                else:
                    break
            if n >= 3:
                fr.down(8.0)
                table(d, fr, cols_, lignes[:n], self, self.s_tab, self.lead_tab, s_ent=self.S_ENT, ms=self._ms_col)
                lignes = lignes[n:]
            fr = self._place(d, fr, v, 1.0e9, cols, suivante) if not fr.fits(1.0e9) else fr

    # -------------------------------------------------- la garde (§ 10.4)
    def garde(self, d, doc, sections, fiches=None):
        """La garde : le bandeau, le cartouche (projet, partie, date, indice, et l'objet : « Fondation de grue — dalle … —
        C30/37 », la grue) ; le sommaire en deux parties ; la légende des couleurs."""
        S = self
        W, H = d.W, d.H
        m = 52
        g = doc.get("garde_saf") or {}
        d.box(0, H - 132, W, 132, fill=S.panel)
        y = H - 52
        d.t(m, y, doc.get("bureau", ""), S.f_v, 8.6, S.acc, track=1.6)
        d.t(m, y - 30, doc.get("titre", "Note de calcul"), S.f_h, 26, S.ink)
        d.line(m, H - 132, W - m, H - 132, S.acc, 1.2)
        fr = Frame(m, H - 132, W - 2 * m, m)
        for lab, val in (("PROJET", doc.get("projet", "")), ("PARTIE", doc.get("partie", "")), ("DATE", doc.get("date", "")),
                         ("INDICE", str(doc.get("indice", ""))), ("OBJET", g.get("objet", ""))):
            fr.down(21)
            d.t(fr.x, fr.y, lab, S.f_v, 7.0, S.mut, track=1.3)
            d.t(fr.x + 108, fr.y, d.fit(val, S.f_b, 10, fr.w - 108), S.f_b, 10, S.ink)
            d.line(fr.x + 108, fr.y - 5, fr.x1, fr.y - 5, mix(S.rule, S.ink, 0.10), 0.5)
        fr.down(40)
        d.t(fr.x, fr.y, "SOMMAIRE", S.f_v, 7.6, S.acc, track=1.6)
        fr.down(5)
        d.line(fr.x, fr.y, fr.x1, fr.y, S.acc, 0.9)
        for i, partie in enumerate(g.get("parties") or [], 1):
            fr.down(24)
            d.t(fr.x, fr.y, f"{i:02d}", S.f_h, 10.5, mix(S.acc, "#FFFFFF", 0.55))
            d.t(fr.x + 26, fr.y, partie["titre"], S.f_h, 11, S.ink)
            wc = 0.0
            if partie.get("etat"):
                wc = chip(d, fr.x1, fr.y - 2, partie["etat"].upper(), S.f_v, 6.4, "#FFFFFF", partie["couleur"], align="right") + 10.0
            d.t(fr.x1 - wc, fr.y, partie["pages"], S.f_b, 7.6, S.mut, "right")
            if partie.get("sous"):
                fr.down(12)
                d.t(fr.x + 26, fr.y, d.fit(partie["sous"], S.f_b, 8.2, fr.w - 120), S.f_b, 8.2, S.mut)
            for ch in partie.get("chapitres") or []:
                fr.down(11)
                d.t(fr.x + 38, fr.y, d.fit(ch[0], S.f_b, 7.6, fr.w - 140), S.f_b, 7.6, S.mut)
                d.t(fr.x1, fr.y, ch[1], S.f_b, 7.6, S.mut, "right")
            fr.down(10)
            d.line(fr.x, fr.y, fr.x1, fr.y, mix(S.rule, S.ink, 0.06), 0.5)
        d.t(m, 46, LEGENDE, S.f_b, 7.4, S.mut)
        return d


def _vb(svg: str) -> Tuple[float, float]:
    vb = [float(t) for t in ET_.fromstring(svg).get("viewBox").split()]
    return vb[2], vb[3]


def _taille_fig(svg: str, largeur: float, hmax: Optional[float]) -> Tuple[float, float]:
    """(largeur, hauteur) d'une figure posée dans la colonne : toute sa largeur, sans dépasser ``hmax``."""
    vw, vh = _vb(svg)
    e = largeur / vw
    if hmax and vh * e > hmax:
        e = hmax / vh
    return vw * e, vh * e


_TAILLES = re.compile(r'font-size="([\d.]+)"')


def texte_mini(svg: str, largeur: float, hmax: Optional[float]) -> float:
    """Le plus petit texte de la figure une fois posée (pt)."""
    vw, _ = _vb(svg)
    w, _ = _taille_fig(svg, largeur, hmax)
    tailles = [float(t) for t in _TAILLES.findall(svg)]
    return min(tailles) * w / vw if tailles else 99.0


# les figures posées dans la note et leur plus petit texte (pt) — les tests le lisent (§ 10.3 : 7 pt au moins)
FIGURES: List[Tuple[str, float]] = []


def lisible(fabrique, largeur: float, hmax: Optional[float], nom: str) -> str:
    """Un plan (``dessin_saf.plan``) dont le texte fait 7 pt au moins une fois posé : son texte est grossi (``texte``)
    jusqu'à ce que ce soit vrai — le dessin garde sa géométrie."""
    f = 1.0
    svg = fabrique(f)
    for _ in range(6):
        t = texte_mini(svg, largeur, hmax)
        if t >= TEXTE_MINI - 1e-6:
            break
        f *= TEXTE_MINI / t * 1.02
        svg = fabrique(f)
    FIGURES.append((nom, texte_mini(svg, largeur, hmax)))
    return svg


def _figure_en_pt(svg: str, nom: str, largeur: float) -> str:
    FIGURES.append((nom, texte_mini(svg, largeur, None)))
    return svg


# ------------------------------------------------------------------ 2. Données (la bande grise)
def _origine(platine: Optional[Dict], champ: str, texte: str) -> str:
    return ORIGINES.get(((platine or {}).get("origines") or {}).get(champ), "") + texte


def appuis_texte(ch: Dict) -> List[Tuple[str, str]]:
    """Les appuis, en lignes (libellé, valeur) : réels ou rectangle, la forme, les dimensions avec leur origine (« ≈ » :
    mesurée ; « calc. » : calculée), l'aire ; « saisie » quand les dimensions le sont."""
    p = ch["platine"]
    out = [("Appuis", "rectangle (saisie)" if ch["appuis"] == "rectangle" else "réels (la bibliothèque)")]
    if ch["appuis"] == "rectangle":
        out.append(("Surface d'appui", ch["dims"]))
    elif p["forme"] == "rectangle":
        out.append(("Surface d'appui", f"rectangle {_origine(p, 'a', _n(p['a']))} × {_origine(p, 'b', _n(p['b']))} m" + (" à 45°" if ch["angle"] == 45 else "")))
    elif p["forme"] == "cercle":
        out.append(("Surface d'appui", f"ronde Ø {_origine(p, 'diametre', _n(p['diametre']))} m — "
                    + ("polygone à 48 côtés" if ch["forme"] == "cercle" else f"carré de même aire, côté {_origine(p, 'cote_carre_meme_aire', _n(p['cote_carre'], 3))} m")))
    else:
        out.append(("Surface d'appui", f"octogone, {_origine(p, 'cote_sur_plats', _n(p['cote_sur_plats']))} m sur plats"))
    if p and p["excentrement"]:
        out.append(("Excentrement radial", f"{_origine(p, 'excentrement_radial', _n(p['excentrement'] * 1000, 0))} mm"))
    out.append(("Aire d'une surface", f"{_n(ch['aire'], 4)} m²"))
    return out


def _dims(el: Dict) -> str:
    return _texte(SS.dimensions(el))


def page_donnees(el: Dict, calc: Dict, F: Dict, style: Planche) -> Dict:
    m = calc["modele"]
    ch, g, d, p = m["chassis"], el["grue"], el["dalle"], el["projet"]
    sec = dict(F, section="Données", bande=True, coupe_w=300)
    verifs = []
    items = [_v(lab, p[k]) for k, lab in (("projet", "Projet"), ("grue", "Grue"), ("creation", "Date de création"),
                                          ("mise_a_jour", "Date de mise à jour"), ("indice", "Indice")) if p.get(k)]
    if items:
        verifs.append({"titre": "Projet", "items": items, "verdicts": []})
    items = [_v("Dalle", "rectangle" if d["forme"] == "rectangle" else f"contour de {len(m['contour'])} sommets"),
             _v("Dimensions", f"{_n(d['Lx'])} × {_n(d['Ly'])}" if d["forme"] == "rectangle" else f"aire {_n(G.aire(m['contour']), 2)}",
                "m" if d["forme"] == "rectangle" else "m²"),
             _v("Épaisseur", _n(d["h"]), "m", "h"), _v(f"Béton {d['beton']}", _n(m["E"], 0), "MPa", "E"),
             _v("Raideurs de sol", " ; ".join(C.nb(x["k"]) for x in el["sol"]["k"]), "MN/m³", "k"),
             _v("Cochées « SCIA » (l'enveloppe)", " ; ".join(C.nb(k) for k in PR.k_scia(el)) or "aucune", "MN/m³" if PR.k_scia(el) else ""),
             _v("Contrainte de contact admise", _sol(el, SS.limite_kpa(el), False), SS.unite(el)[0])]
    if d["forme"] == "contour":
        items.append(("pr", "Sommets (x ; y) en m : " + " · ".join(f"({C.nb(round(x, 3))} ; {C.nb(round(y, 3))})" for x, y in m["contour"]) + "."))
    verifs.append({"titre": "Dalle et sol", "items": items, "verdicts": []})
    items = [_v("Châssis", ch["nom"] if ch["id"] else "saisie")]
    if ch["id"]:
        items += [_v("Constructeur", ch["constructeur"]), _v("Id dans la bibliothèque", ch["id"])]
    items.append(_v("Entraxes des appuis", f"{_n(ch['a'])} × {_n(ch['b'])}", "m", "a × b"))
    items += [_v(lab, val) for lab, val in appuis_texte(ch)]
    if ch["impose"]:
        items.append(("pr", "Platine non indiquée dans la bibliothèque : les dimensions de la surface d'appui sont saisies."))
    if ch["note_fondation"]:
        items.append(("pr", f"Note de la bibliothèque : {ch['note_fondation']}"))
    pl = g["plots"]
    items.append(_v("Plots sous les pieds", PLOTS[pl["type"]]))
    if pl["type"] != "aucun":
        items.append(_v("Plots : a × b × hauteur", f"{_n(pl['a'])} × {_n(pl['b'])} × {_n(pl['h'])}", "m"))
    items += [_v("Position du centre", f"{_n(g['ex'])} ; {_n(g['ey'])}", "m", "e_x ; e_y"), _v("Angle", C.nb(round(g["beta"], 2), 2), "°", "β"),
              _v("Distance mini plaque–bord", _n(g["dmin"]), "m")]
    verifs.append({"titre": "Grue : châssis, appuis, plots", "items": items, "verdicts": []})
    largeur = style.largeur_colonne(sec)
    wp = 36.0
    cols = [("JEU", largeur - 4 * wp, "left")] + [(f"P{i + 1} [kN]", wp, "right") for i in range(4)]
    lignes = [[("lab", f"{i + 1}" + (f" — {j['nom']}" if j["nom"] else ""), False)] + [("num", "" if v is None else C.nb(v)) for v in j["P"]]
              for i, j in enumerate(el["reactions"])]
    verifs.append({"titre": "Réactions par jeu", "items": [
        ("pr", "P1 en haut à gauche, P2 en haut à droite, P3 en bas à droite, P4 en bas à gauche : sens horaire vu du dessus, châssis non tourné."),
        ("tab", cols, lignes)], "verdicts": []})
    wn, wl = 30.0, 34.0
    wc = (largeur - wn - wl) / 2
    cols = [("BANDE", wn, "left"), ("DÉBUT (x ; y) [m]", wc, "left"), ("FIN (x ; y) [m]", wc, "left"), ("LARGEUR", wl, "right")]
    lignes = [[("lab", b["nom"], True), ("lab", f"{_n(b['debut'][0])} ; {_n(b['debut'][1])}"), ("lab", f"{_n(b['fin'][0])} ; {_n(b['fin'][1])}"),
               ("num", f"{_n(b['largeur'])} m")] for b in m["bandes"]]
    verifs.append({"titre": "Bandes d'intégration", "items": [
        ("pr", "Par les centres des surfaces d'appui, le long des axes du châssis : BX1 par P1 et P2, BX2 par P4 et P3, BY1 par P1 et P4, "
               "BY2 par P2 et P3. Dans SCIA : dalle D1, « Constant symmetric »."), ("tab", cols, lignes)], "verdicts": []})
    vol = G.aire(m["contour"]) * d["h"]
    items = [_v("Dalle", f"{_n(G.aire(m['contour']), 2)} m² × {_n(d['h'])} m = {_n(vol, 2)}", "m³")]
    if pl["type"] == "coule" and m["plots"]:
        vp = sum(G.aire(q) for q in m["plots"]["polys"]) * m["plots"]["hauteur"]
        items.append(_v("Plots coulés en place", f"{len(m['plots']['polys'])} × {_n(pl['a'])} × {_n(pl['b'])} × {_n(pl['h'])} = {_n(vp, 2)}", "m³"))
        items.append(_v("Total", _n(vol + vp, 2), "m³"))
    verifs.append({"titre": "Volume de béton", "items": items, "verdicts": []})
    w_fig = sec["coupe_w"]
    svg = lisible(lambda f: DS.plan(el, m, None, interactif=False, ident="note_plan", texte=f), w_fig, 300.0, "Données : plan coté")
    blocs = [("DALLE", [("Dalle", "", _dims(el), ""), ("Béton", "", d["beton"], "")]),
             ("CHÂSSIS", [("Entraxes", "", f"{_n(ch['a'])} × {_n(ch['b'])}", "m"), ("Appuis", "", _texte(ch["dims"]), ""),
                          ("Position", "", f"{_n(g['ex'])} ; {_n(g['ey'])}", "m"), ("Angle", "β", C.nb(round(g["beta"], 2), 2), "°")])]
    w, h = _taille_fig(svg, w_fig, 300.0)

    def dessin(d_, x, y, ww, hh, S):
        peindre_svg(d_.c, svg, x + (ww - w) / 2, y, w, h, marge=0.0, riche=True)
    sec.update(titre_coupe="PLAN COTÉ — COTES EN m", blocs=blocs, verifs=verifs, dessin=dessin, coupe_h=h + 2.0,
               reserve=20.0 + 7.6 * 1.42 * max(len(r) for _t, r in blocs) + 6.0)
    return sec


# ------------------------------------------------------------------ 3. Charges et combinaisons (les feuilles du SAF)
def _feuille(feuilles: List[Dict], nom: str) -> Tuple[List[str], List[List]]:
    f = next((x for x in feuilles if x["nom"] == nom), None)
    if f is None:
        return [], []
    return f["lignes"][0], f["lignes"][1:]


def _coords(txt: str) -> List[float]:
    return [float(v) for v in str(txt or "").split(";") if v.strip()]


def page_charges(el: Dict, calc: Dict, F: Dict, style: Planche) -> Dict:
    m = calc["modele"]
    sec = dict(F, section="Charges et combinaisons", bande=False, n_cols=2)
    largeur = style.largeur_colonne(sec)
    ks = PR.k_scia(el)
    k_saf = ks[0] if ks else el["sol"]["k"][0]["k"]
    fe = EC.feuilles(el, k_saf)
    verifs = []
    # LC1 : le poids propre (SF1) et le poids des plots (FP)
    c_sf, l_sf = _feuille(fe, "StructuralSurfaceAction")
    c_ff, l_ff = _feuille(fe, "StructuralSurfaceActionFree")
    sf = l_sf[0][c_sf.index("Value [kN/m2]")] if l_sf else None
    items = [_v("SF1 sur la dalle D1 : 25 × h", f"25 × {_n(el['dalle']['h'])} = {_n(-sf, 2)}", "kN/m²")] if sf is not None else []
    fp = [li for li in l_ff if str(li[0]).startswith("FP")]
    if fp:
        q = -fp[0][c_ff.index("q [kN/m2]")]
        items.append(_v(f"{fp[0][0]} à {fp[-1][0]} sur chaque plot : 25 × hauteur", f"25 × {_n(m['plots']['hauteur'])} = {_q(q)}", "kN/m²"))
    verifs.append({"titre": "LC1 — poids propre", "items": items or [("pr", "—")], "verdicts": []})
    # les cas : R (kN) et q = R ÷ aire de la surface de charge (kN/m²) — les lignes FF du SAF
    ff = {li[0]: li for li in l_ff if str(li[0]).startswith("FF")}
    iq = c_ff.index("q [kN/m2]") if c_ff else 0
    wc, wl, wj = 24.0, 30.0, 20.0
    wp = (largeur - wc - wl - wj) / 8
    cols = [("CAS", wc, "left"), ("LC", wl, "left"), ("JEU", wj, "left")]
    for i in range(4):
        cols += [(f"R{i + 1} [kN]", wp, "right"), (f"q{i + 1} [kN/m²]", wp, "right")]
    lignes = []
    for i_cas, c in enumerate(m["cas"]):
        li = [("lab", c["nom"], True), ("lab", c["lc"]), ("lab", str(c["jeu_saisi"]))]
        for p_ in range(4):
            row = ff.get(f"FF{4 * i_cas + p_ + 1}")
            li += [("num", C.nb(c["R"][p_])), ("num", _q(-row[iq]) if row else "—")]
        lignes.append(li)
    surfaces = V.surfaces_chargees(m)
    s0 = surfaces[0]
    xs, ys = [p[0] for p in s0], [p[1] for p in s0]
    a_, b_ = max(xs) - min(xs), max(ys) - min(ys)
    if m["plots"]:
        quoi = (f"carré de diffusion {_n(a_, 2)} × {_n(b_, 2)} m = {_n(G.aire(s0), 3)} m² (sous un plot : une plaque de 0,50 × 0,50 m diffusée "
                f"à 45° sur la hauteur du plot, sans dépasser le plot, § 6)")
    else:
        quoi = f"la surface d'appui, {_texte(m['chassis']['dims'])} = {_n(G.aire(s0), 3)} m²" if el["grue"]["appuis"] == "rectangle" \
            else f"la surface d'appui, {_n(G.aire(s0), 4)} m²"
    items = [("pr", f"{len(m['cas'])} cas (chaque jeu dans toutes les positions de la grue : rotations de 90° et miroirs, doublons retirés). "
                    "R1 à R4 : la réaction de chaque pied ; q = R ÷ aire de la surface de charge : les valeurs des lignes FF du SAF (charge "
                    "descendante)."), ("tab", cols, lignes)]
    if not m["cas"]:
        items = [("pr", "Aucun jeu de réactions complet : aucun cas de charge.")]
    verifs.append({"titre": "Cas de charge : réactions et charges réparties", "items": items, "verdicts": []})
    # les surfaces de charge et leur position (les lignes FF du premier cas)
    if m["cas"]:
        ix, iy = c_ff.index("Coordinate X [m]"), c_ff.index("Coordinate Y [m]")
        cols = [("PIED", 24.0, "left"), ("CENTRE (x ; y) [m]", 70.0, "left"), ("X DU SAF [m]", (largeur - 94) / 2, "left"),
                ("Y DU SAF [m]", (largeur - 94) / 2, "left")]
        lignes = []
        for p_ in range(4):
            row = ff.get(f"FF{p_ + 1}")
            cx, cy = G.centre_gravite(surfaces[p_])
            lignes.append([("lab", f"P{p_ + 1}", True), ("lab", f"{_n(cx, 3)} ; {_n(cy, 3)}"), ("lab", str(row[ix]) if row else ""),
                           ("lab", str(row[iy]) if row else "")])
        verifs.append({"titre": "Surfaces de charge", "items": [("pr", f"Surface de charge : {quoi}."), ("tab", cols, lignes)], "verdicts": []})
    # les combinaisons (feuille StructuralLoadCombination)
    c_co, l_co = _feuille(fe, "StructuralLoadCombination")
    if l_co:
        i_n, i_t = c_co.index("Name"), c_co.index("Type")
        cols = [("COMBINAISON", 84.0, "left"), ("TYPE", 50.0, "left"), ("CONTENU", largeur - 134.0, "left")]
        lignes = []
        for li in l_co:
            parts = []
            for j in (1, 2):
                lc = li[c_co.index(f"Load Case name {j}")]
                if lc:
                    f_ = (li[c_co.index(f"Load Factor {j}")] or 1) * (li[c_co.index(f"Multiplier {j}")] or 1)
                    parts.append(f"{lc} × {_n(f_, 2)}")
            lignes.append([("lab", str(li[i_n]), True), ("lab", "non linéaire" if li[i_t] == "Nonlinear" else "linéaire"), ("lab", " + ".join(parts))])
        verifs.append({"titre": "Combinaisons (les noms du SAF)", "items": [
            ("pr", "ELS caractéristiques (« SLS ») : ELS-Car (auto)1 = LC1, puis LC1 + LCn pour chaque cas ; coefficients 1,00."),
            ("tab", cols, lignes)], "verdicts": []})
    # le sol pour SCIA : chaque k
    cols = [("k", 30.0, "left"), ("C1z [MN/m³]", 48.0, "right"), ("C1x [MN/m³]", 48.0, "right"), ("C1y [MN/m³]", 48.0, "right"),
            ("C2x [MN/m]", 46.0, "right"), ("C2y [MN/m]", 46.0, "right"), ("", largeur - 266.0, "left")]
    lignes = []
    for x in el["sol"]["k"]:
        c_so, l_so = _feuille(EC.feuilles(el, x["k"]) if x["k"] != k_saf else fe, "StructuralSurfaceConnection")
        r_ = l_so[0]
        lignes.append([("lab", C.nb(x["k"]), x["k"] == k_saf)] + [("num", C.nb(r_[c_so.index(c)])) for c in
                                                                   ("C1z [MN/m3]", "C1x [MN/m3]", "C1y [MN/m3]", "C2x [MN/m]", "C2y [MN/m]")]
                      + [("lab", "dans le SAF" if x["k"] == k_saf else ("à mettre dans SCIA" if x["scia"] else "non coché « SCIA »"))])
    verifs.append({"titre": "Sol pour SCIA", "items": [
        ("pr", f"Le SAF porte k = {C.nb(k_saf)} MN/m³ (le premier coché « SCIA ») ; les autres k cochés sont à mettre dans SCIA (SS1 sous D1, SS2 "
               "sous l'anneau D2)."), ("tab", cols, lignes)], "verdicts": []})
    sec["verifs"] = verifs
    return sec


# ------------------------------------------------------------------ 4. Sol
def _verdict_sol(el: Dict, s: Dict) -> Dict:
    if s.get("r_vue") is None:
        return {"texte": s["phrase"], "etat": "ko"}
    signe = "≤" if s["taux"] <= 1 else ">"
    quoi = "Poids propre seul" if s["pp"] else MOTS[s["niveau"]]
    ou = "les réactions sont à saisir" if s["pp"] else f"cas {s['cas']}, k = {C.nb(s['k'])} MN/m³"
    return {"texte": f"{quoi} — σ max = {_sol(el, s['sigma'])} {signe} {_sol(el, s['limite'])} · {ou}", "etat": COULEURS[s["niveau"]],
            "cale": f"{C.pct(s['taux'])} %".replace(C.NBSP, " ")}


def page_sol(el: Dict, calc: Dict, s: Dict, F: Dict, style: Planche) -> Dict:
    m = calc["modele"]
    sec = dict(F, section="Sol", bande=False, n_cols=2)
    largeur = style.largeur_colonne(sec)
    o = el["options"]
    verifs = [{"titre": "Hypothèses", "items": [
        ("pr", f"Plaque mince sur sol élastique (Winkler, C2 = k × {C.nb(o['c2x'])}), sol en compression seule (les zones qui se soulèvent sont "
               f"libérées), maille {_n(m['maille'])} m."),
        ("pr", "Charges ELS caractéristiques : poids propre, plots et réactions de la grue — chaque jeu dans toutes ses positions."),
        ("pr", "Calcul indépendant de SCIA (un autre moteur) : une vérification complémentaire.")], "verdicts": []}]
    if s.get("r_vue") is None:
        verifs.append({"titre": "Contrainte de contact", "items": [("k", 0)], "verdicts": [_verdict_sol(el, s)]})
        sec["verifs"] = verifs
        return sec
    ks = PR.k_scia(el) or [x["k"] for x in el["sol"]["k"]]
    el_env = dict(el, vue=dict(el["vue"], carte="sigma", cas=None, k=None))
    champ = SR.champ(el_env, calc, s)
    svg = lisible(lambda f: DS.plan(el, m, champ, interactif=False, carte="vecteur", ident="note_carte", texte=f), largeur, 330.0,
                  "Sol : carte de contrainte")
    titre = ("CONTRAINTE DE CONTACT — POIDS PROPRE" if s["pp"] else "CONTRAINTE DE CONTACT — ENVELOPPE DES CAS") + \
        f" · k = {' ; '.join(C.nb(k) for k in ks)} MN/m³"
    verifs.append({"titre": "Carte de la contrainte de contact", "items": [("fig", svg, titre, 330.0), ("k", 0)], "verdicts": [_verdict_sol(el, s)]})
    scia = set(PR.k_scia(el))
    w = (40.0, 50.0, 40.0, 44.0, 40.0)
    cols = [("k [MN/m³]", w[0], "left"), (f"σ MAX [{SS.unite(el)[0]}]", w[1], "right"), ("CAS", largeur - sum(w), "left"), ("w MAX [mm]", w[2], "right"),
            ("DÉCOLLÉ", w[3], "right"), ("SCIA", w[4], "right")]
    lignes = []
    for r in calc["par_k"]:
        gov = r["k"] == s["k"]
        lignes.append([("lab", C.nb(r["k"]), gov), ("eta", _sol(el, r["sigma_max"], False), r["sigma_max"] <= s["limite"], gov), ("lab", r["cas_max"]),
                       ("num", _n(r["w_max"], 2)), ("num", f"{_n(r['decolle'], 1)} %"), ("num", "oui" if r["k"] in scia else "—")])
    verifs.append({"titre": "Par raideur de sol", "items": [("tab", cols, lignes)], "verdicts": []})
    # tous les cas, pour chaque k coché
    if not s["pp"]:
        retenus = [r for r in calc["par_k"] if r["k"] in set(ks)]
        wc, wl = 30.0, 32.0
        wk = (largeur - wc - wl) / (2 * len(retenus))
        cols = [("CAS", wc, "left"), ("LC", wl, "left")]
        for r in retenus:
            cols += [(f"σ [{SS.unite(el)[0]}] k = {C.nb(r['k'])}", wk, "right"), (f"w [mm] k = {C.nb(r['k'])}", wk, "right")]
        lignes = []
        lc = {c["nom"]: c["lc"] for c in m["cas"]}
        for i, c in enumerate(retenus[0]["cas"]):
            li = [("lab", c["nom"], False), ("lab", lc.get(c["nom"], ""))]
            for r in retenus:
                cc = r["cas"][i]
                li += [("eta", _sol(el, cc["sigma_max"], False), cc["sigma_max"] <= s["limite"], cc["nom"] == r["cas_max"]), ("num", _n(cc["w_max"], 1))]
            lignes.append(li)
        verifs.append({"titre": "Tous les cas, pour chaque k coché « SCIA »", "items": [
            ("pr", f"σ max à la couleur de la limite ({_sol(el, s['limite'])}) ; en gras, le cas le plus défavorable de chaque k."),
            ("tab", cols, lignes)], "verdicts": []})
    # σ moyen ; l'équilibre ; les contrôles
    sm = V.sigma_moyen(m)
    q = SR.equilibre_cas(el, calc, s)
    items = [("pr", f"σ moyen = (dalle {_n(sm['dalle'], 0)}" + (f" + plots {_n(sm['plots'], 0)}" if sm["plots"] else "")
                    + (f" + réactions {_n(sm['reactions'], 0)}" if sm["reactions"] else "") + f") kN ÷ {_n(sm['aire'], 1)} m² = {_sol(el, sm['sigma'])} — un repère.")]
    verdicts = []
    if q is not None:
        bon = q["ecart_max"] <= SR.ECART_EQUILIBRE
        verdicts.append({"texte": f"Équilibre, cas {q['cas']} · k = {C.nb(q['k'])} MN/m³ : " + _texte(SR.phrase_equilibre(q)), "etat": "ok" if bon else "ko"})
        verdicts.append({"texte": f"Écart le plus grand sur {q['n_cas']} cas et {q['n_k']} k : {_n(q['ecart_max'] * 100, 2)} % "
                                  f"{'<' if bon else '≥'} 0,1 %", "etat": "ok" if bon else "ko"})
    ctl, g = m["controles"], el["grue"]
    dedans = all(ctl["plaques_dedans"])
    verdicts.append({"texte": "Les quatre surfaces d'appui sont dans la dalle." if dedans else "Une surface d'appui sort de la dalle.", "etat": "ok" if dedans else "ko"})
    if dedans:
        verdicts.append({"texte": f"Distance mini plaque–bord : {_n(ctl['distance'], 2)} m {'≥' if ctl['distance_ok'] else '<'} {_n(g['dmin'])} m"
                                  + ("." if ctl["distance_ok"] else " — sous le minimum."), "etat": "ok" if ctl["distance_ok"] else "att"})
    if m["plots"]:
        bon = ctl["plots_ok"]
        verdicts.append({"texte": "Les plots sont dans la dalle, ne se chevauchent pas, et chaque surface d'appui est sur son plot." if bon
                         else "Un plot sort de la dalle, en chevauche un autre, ou ne porte pas toute sa surface d'appui.", "etat": "ok" if bon else "ko"})
    items += [("k", i) for i in range(len(verdicts))]
    verifs.append({"titre": "σ moyen, équilibre, contrôles", "items": items, "verdicts": verdicts})
    sec["verifs"] = verifs
    return sec


# ------------------------------------------------------------------ 5. Efforts
def _cas_k(ck) -> str:
    return f"{ck[0]} · k = {C.nb(ck[1])}" if ck else "—"


def _champ_moments(calc: Dict, ef: Dict, fer: Dict, face: str) -> Dict:
    """La carte enveloppe des moments au point (le champ de ``dessin_saf.plan``) : en bas, le plus grand de m_x et m_y ;
    en haut, le plus négatif — × l'échelle SCIA de chaque direction ; la ligne noire à m_Rd (le plus faible des deux
    directions de la face)."""
    import numpy as np
    d = calc["par_k"][0]["dalle"]
    ech = fer.get("echelle") or {}
    nx, ny = np.nan_to_num(ef["noeuds"][f"{face}_x"]), np.nan_to_num(ef["noeuds"][f"{face}_y"])
    if face == "inf":
        val = np.maximum(np.maximum(nx * ech.get("inf_x", 1.0), ny * ech.get("inf_y", 1.0)), 0.0).ravel()
    else:
        val = np.maximum(np.maximum(-nx * ech.get("sup_x", 1.0), -ny * ech.get("sup_y", 1.0)), 0.0).ravel()
    val = np.where(d.on0, val, 0.0)
    n = V.premier_argmax(np.where(d.on0, val, -np.inf))
    mrd = min(fer["res"]["mRd"][f"{face}_x"], fer["res"]["mRd"][f"{face}_y"])
    return {"mode": "m", "dalle": d, "valeurs": val, "decolle": np.zeros(len(val), bool), "maxi": float(val[n]),
            "x": d.x0 + (n % d.nnx) * d.a, "y": d.y0 + (n // d.nnx) * d.b, "limite": mrd, "unite": ("kN·m/m", 1.0, 1),
            "symbole": "m bas" if face == "inf" else "m haut", "mot": "max" if face == "inf" else "min", "signe": 1.0 if face == "inf" else -1.0}


def page_efforts(el: Dict, calc: Dict, ef: Dict, fer: Dict, F: Dict, style: Planche) -> Dict:
    m = calc["modele"]
    sec = dict(F, section="Efforts", bande=False, n_cols=2)
    largeur = style.largeur_colonne(sec)
    ech = fer.get("echelle") or {}
    mrd_b = fer["mRd_bandes"]
    verifs = []
    leg = _figure_en_pt(DE.legende(largeur), "Efforts : légende", largeur)
    rapport = any(abs(ech.get(c, 1.0) - 1.0) > 1e-9 for c in ech)
    intro = [("pr", f"Enveloppe de tous les cas et des k cochés « SCIA » ({' ; '.join(C.nb(k) for k in ef['ks'])} MN/m³), intégrée sur la largeur de "
                    f"chaque bande ({_n(ef['largeur'])} m), sans torsion ; V se lit à d = h − 6 cm = {_n(ef['d_v'])} m du bord des zones chargées."
                    + (" Les moments sont ceux du module × le rapport SCIA ÷ module de leur face (§ 8.2)." if rapport else "")), ("fig", leg, None)]
    for i, b in enumerate(ef["bandes"]):
        s_ = "x" if b["nom"].startswith("BX") else "y"
        svg_m = _figure_en_pt(DE.moments(b, mrd_b[f"inf_{s_}"], mrd_b[f"sup_{s_}"], ech.get(f"inf_{s_}", 1.0), ech.get(f"sup_{s_}", 1.0), largeur, 150.0),
                              f"Efforts : M(s) {b['nom']}", largeur)
        svg_v = _figure_en_pt(DE.tranchant(b, largeur, 105.0), f"Efforts : V(s) {b['nom']}", largeur)
        verifs.append({"titre": f"{b['nom']} — M(s) et V(s)", "items": (intro if i == 0 else [])
                       + [("fig", svg_m, f"M(s) [kN·m] · BANDE {b['nom']} DE {_n(b['largeur'])} m"),
                          ("fig", svg_v, f"V(s) [kN] · BANDE {b['nom']}")], "verdicts": []})
    w = (34.0, 40.0, 52.0, 52.0)
    cols = [("BANDE", w[0], "left"), ("GRANDEUR", w[1], "left"), ("VALEUR", w[2], "right"), ("PAR MÈTRE", w[3], "right"), ("CAS · k", largeur - sum(w), "left")]
    lignes = []
    for b in ef["bandes"]:
        for nom, cle, unite, cas in (("M+", "M_plus", "kN·m", "cas_M_plus"), ("M−", "M_moins", "kN·m", "cas_M_moins"), ("V à d", "V_d", "kN", "cas_V_d")):
            v = b[cle]
            lignes.append([("lab", b["nom"], nom == "M+"), ("lab", nom), ("num", f"{_n(v, 1)} {unite}" if v is not None else "ne se lit pas"),
                           ("num", f"{_n(v / b['largeur'], 1)} {unite}/m" if v is not None else "—"), ("lab", _cas_k(b[cas]))])
    verifs.append({"titre": "Tableau des bandes (T14 : le cas et le k de chaque valeur)", "items": [
        ("pr", "Valeurs du module, hors des plots coulés en place (au nu du plot)."), ("tab", cols, lignes)], "verdicts": []})
    items = []
    for face in ("inf", "sup"):
        champ = _champ_moments(calc, ef, fer, face)
        svg = lisible(lambda f, ch=champ: DS.plan(el, m, ch, interactif=False, carte="vecteur", ident=f"note_m_{face}", texte=f), largeur, 330.0,
                      f"Efforts : carte des moments {FACES[face]}")
        titre = ("EN BAS : LE PLUS GRAND DE mx ET my" if face == "inf" else "EN HAUT : LE PLUS NÉGATIF DE mx ET my") + \
            f" · LIGNE NOIRE : mRd = {_n(champ['limite'], 1)} kN·m/m"
        items.append(("fig", svg, titre, 330.0))
    ex, ce = ef["extremes"], ef["cas_extremes"]
    items.append(("pr", " · ".join(f"m{c[-1]} {'max' if c.startswith('inf') else 'min'} = {_n(ex[c], 1)} kN·m/m ({_cas_k(ce.get(c))})" for c in
                                   ("inf_x", "inf_y", "sup_x", "sup_y")) + "."))
    verifs.append({"titre": "Moments au point : cartes enveloppes", "items": items, "verdicts": []})
    sec["verifs"] = verifs
    return sec


# ------------------------------------------------------------------ 6. Principe d'armatures
def page_principe(el: Dict, calc: Dict, ef: Dict, fer: Dict, F: Dict, style: Planche) -> Dict:
    m = calc["modele"]
    sec = dict(F, section="Principe d'armatures", bande=False, n_cols=2)
    largeur = style.largeur_colonne(sec)
    verifs = []
    if fer.get("hors_domaine"):
        verifs.append({"titre": "Armatures", "items": [("k", 0)], "verdicts": [{"texte": f"Armatures non vérifiées : {fer['hors_domaine']}.", "etat": "att"}]})
        sec["verifs"] = verifs
        return sec
    for face in ("inf", "sup"):
        svg = lisible(lambda f, fa=face: DS.plan(el, m, None, interactif=False, ident=f"note_a_{fa}", texte=f,
                                                armatures={"ef": ef, "fer": fer, "face": fa, "court": True}), largeur, 420.0,
                      f"Principe : plan {FACES[face]}")
        verifs.append({"titre": f"Plan du {FACES[face]} — treillis {SA.treillis_face(el, face).replace(C.NBSP, ' ')}", "items": [
            ("fig", svg, f"{FACES[face].upper()} : RENFORTS DES BANDES (REPÈRE), ZONES ROUGES ET RENFORTS LOCAUX (TIRETS)", 420.0)], "verdicts": []})
    w = (28.0, 28.0, 26.0, 48.0, 44.0, 44.0)
    cols = [("REPÈRE", w[0], "left"), ("FACE", w[1], "left"), ("SENS", w[2], "left"), ("BARRES", w[3], "left"), ("LONGUEUR", w[4], "right"),
            ("LARGEUR", w[5], "right"), ("POSITION (x ; y) [m]", largeur - sum(w), "left")]
    lignes = [[("lab", r["rep"], True), ("lab", FACES[r["face"]]), ("lab", r["sens"]), ("lab", SA.barres(r).replace(C.NBSP, " ")),
               ("num", f"{_n(r['longueur'])} m"), ("num", f"{_n(r['largeur'])} m"),
               ("lab", f"x {_n(r['x0'])} à {_n(r['x1'])} · y {_n(r['y0'])} à {_n(r['y1'])}" + (f" ({r['bande']})" if r["bande"] else " (local)"))]
              for r in fer["renforts"]]
    items = [("pr", f"Renforts des bandes (sur toute la largeur de la bande) et renforts locaux des zones rouges ; ancrage {C.nb(el['armatures']['ancrage'])} Ø de "
                    "chaque côté ; n = ⌈(As(m dim) − As du treillis) × largeur ÷ aire d'une barre⌉, As() du moteur Dalles.")]
    items.append(("tab", cols, lignes) if lignes else ("pr", "Aucun renfort : les treillis suffisent partout."))
    verifs.append({"titre": "Renforts", "items": items, "verdicts": []})
    res = fer["res"]
    items, verdicts = [], []
    for cle in ("inf_x", "inf_y", "sup_x", "sup_y"):
        f_, s_ = cle.split("_")
        items.append(_v(f"Treillis {FACES[f_]} {s_} {SA.un_treillis(el['armatures'][f_][s_]).replace(C.NBSP, ' ')}",
                        f"{_n(res['mRd'][cle], 1)}", "kN·m/m", "m_Rd"))
        items.append(("pr", f"d = {_n(res['d'][cle], 1)} cm · As = {_n(res['As'][cle], 0)} mm²/m (moteur Dalles)."))

    def ver(texte, etat):
        items.append(("k", len(verdicts)))
        verdicts.append({"texte": texte, "etat": etat})
    for s_, _ in SA.SENS:
        g = SA.groupe_bandes(fer, ef, s_)
        mrd_i, mrd_s = fer["mRd_bandes"][f"inf_{s_}"], fer["mRd_bandes"][f"sup_{s_}"]
        ver(f"Flexion {g['nom']} : bas {_texte(SA.ligne_flexion(g['m_inf'], mrd_i, g['renforts_inf']))} ; haut "
            f"{_texte(SA.ligne_flexion(g['m_sup'], mrd_s, g['renforts_sup']))}", "ok")
        if g["hauteur"] is not None:
            ver(f"Hauteur de béton {g['nom']} : {_texte(SA.ligne_hauteur(g['hauteur']))} ({C.pct(g['hauteur']['taux'])} %)",
                "ok" if g["hauteur"]["etat"] == "ok" else "ko")
        txt, taux, niv, mot = SA.ligne_tranchant(g["tranchant"])
        ver(f"Effort tranchant {g['nom']} : {_texte(txt)}" + (f" ({C.pct(taux)} %)" if taux is not None else " — non vérifié"),
            "att" if taux is None else ("ok" if g["tranchant"]["etat"] == "ok" else "ko"))
    if not any(r["type"] == "bande" for r in fer["renforts"]):
        h = fer["hauteur"]
        ver(f"Hauteur de béton : {_texte(SA.ligne_hauteur(h))} ({C.pct(h['taux'])} %)", "ok" if h["etat"] == "ok" else "ko")
    for t, n in fer["alertes"]:
        ver(t, "ko" if n == "r" else "att")
    ver("Poinçonnement : non vérifié (le moteur Dalles ne le vérifie pas).", "att")
    items.append(("pr", fer["u"] + "."))
    items.append(("pr", SR.non_verifies(el, calc, fer["non_verifies"])))
    items.append(("t", "Principe d'armatures, pas un plan d'exécution."))
    verifs.append({"titre": "Contrôles du moteur Dalles", "items": items, "verdicts": verdicts})
    sec["verifs"] = verifs
    return sec


# ------------------------------------------------------------------ 7. Comparaison avec SCIA, l'annexe sol
def page_scia(el: Dict, calc: Dict, sc: Optional[Dict], F: Dict, style: Planche, s: Dict) -> Dict:
    sec = dict(F, section="Comparaison avec SCIA", bande=False, n_cols=2)
    largeur = style.largeur_colonne(sec)
    verifs = []
    if sc is None:
        verifs.append({"titre": "Module / SCIA", "items": [("t", "Valeurs SCIA non saisies")], "verdicts": []})
    else:
        w = (54.0, 54.0, 40.0, 40.0, 44.0)
        cols = [("GRANDEUR", largeur - sum(w), "left"), ("MODULE", w[0], "right"), ("SCIA", w[1], "right"), ("RAPPORT", w[2], "right"),
                ("ÉCART", w[3], "right"), ("ÉTAT", w[4], "right")]
        lignes = []
        for li in sc["lignes"]:
            if li["cle"] not in sc["a_taper"]:
                continue
            lib = li["libelle"] + (" (x = y)" if sc["sym"] and li["cle"] in SCM.JUMELLES.values() else "")
            mod = f"{_n(li['module'], 1)} {li['unite']}" if li["module"] is not None else "—"
            if li["scia"] is None:
                lignes.append([("lab", lib), ("num", mod), ("num", "SCIA non saisi"), ("num", "—"), ("num", "—"), ("num", "r = 1")])
            else:
                lignes.append([("lab", lib), ("num", mod), ("num", f"{_n(li['scia'], 1)} {li['unite']}"), ("num", _n(li["r"], 2)),
                               ("num", f"{_n((li['r'] - 1) * 100, 1)} %"), ("eta", "à vérifier" if li["alerte"] else "concorde", not li["alerte"])])
        for x in sc["sigma"]:
            mod = _sol(el, x["module"]) if x["module"] is not None else "—"
            if x["scia"] is None:
                lignes.append([("lab", f"σ max, k = {C.nb(x['k'])} MN/m³"), ("num", mod), ("num", "SCIA non saisi"), ("num", "—"), ("num", "—"), ("num", "—")])
            else:
                e = abs(x["r"] - 1.0) if x["r"] is not None else 0.0
                lignes.append([("lab", f"σ max, k = {C.nb(x['k'])} MN/m³"), ("num", mod), ("num", _sol(el, x["scia"])), ("num", _n(x["r"], 2) if x["r"] else "—"),
                               ("num", f"{_n((x['r'] - 1) * 100, 1)} %" if x["r"] else "—"), ("eta", "à vérifier" if e > SCM.ECART else "concorde", e <= SCM.ECART)])
        items = [("pr", "r = SCIA ÷ module. SCIA donne le niveau, le module la forme : les moments du module sont multipliés par r de leur face et le "
                        "principe d'armatures est refait ; V SCIA va directement au moteur Dalles ; σ max : comparaison seule. Un champ vide reste vide "
                        "(r = 1 : la valeur du module est gardée).")]
        if not sc["saisi"]:
            items.append(("t", "Valeurs SCIA non saisies"))
            items.append(("tab", cols, lignes))
            verdicts = []
        else:
            items.append(("tab", cols, lignes))
            vus = [li for li in sc["lignes"] if li["r"] is not None and li["cle"] in sc["a_taper"]] + [x for x in sc["sigma"] if x.get("r")]
            pire = max((abs(x["r"] - 1.0) for x in vus), default=0.0)
            if sc["alertes"]:
                verdicts = [{"texte": "Écarts à vérifier : " + " ; ".join(a.replace("Écart module / SCIA : ", "") for a in sc["alertes"]) + ".", "etat": "att"}]
            else:
                verdicts = [{"texte": f"Les deux calculs concordent à {_n(pire * 100, 1)} % près.", "etat": "ok"}]
            if not sc["complet"]:
                verdicts.append({"texte": "Valeurs SCIA saisies en partie : les lignes « SCIA non saisi » gardent la valeur du module.", "etat": "att"})
            items += [("k", i) for i in range(len(verdicts))]
        verifs.append({"titre": "Module / SCIA", "items": items, "verdicts": verdicts})
    # l'annexe : la page de résultats du module de raideur du sol (quand les k en viennent)
    r = el["sol"].get("reprise")
    page = (r or {}).get("page")
    if page:
        items = [("pr", f"Les k de la fondation viennent de la note « {r.get('note') or 'sans nom'} »" + (f" ({r['chantier']})" if r.get("chantier") else "")
                        + f" — {page.get('fondation') or ''} · {page.get('methode') or ''} · {str(page.get('resume') or '')}.")]
        if page.get("phrase"):
            items.append(("pr", str(page["phrase"]) + "."))
        verifs.append({"chapitre": "Annexe sol"})
        verifs.append({"titre": "La note de raideur du sol reprise", "items": items, "verdicts": []})
        for b in page.get("blocs") or []:
            wv = 96.0
            cols = [("", (largeur - wv) * 0.42, "left"), ("", (largeur - wv) * 0.58, "left"), ("k", wv, "right")]
            lignes = [[("lab", str(li[0]), False), ("lab", str(li[1])), ("num", str(li[2]), True)] for li in b.get("lignes") or []]
            verifs.append({"titre": str(b.get("titre") or ""), "items": ([("pr", str(b["sous"]) + ".")] if b.get("sous") else []) + [("tab", cols, lignes)],
                           "verdicts": []})
        if page.get("messages"):
            verifs.append({"titre": "Remarques du module « Rigidité du sol »", "items": [("pr", str(t)) for t in page["messages"]], "verdicts": []})
    sec["verifs"] = verifs
    return sec


# ------------------------------------------------------------------ le document
def _numeroter(sec: Dict) -> Dict:
    i = 0
    for v in sec["verifs"]:
        if not v.get("chapitre"):
            i += 1
            v["num"] = i
    if not sec.get("bande", True):
        sec.setdefault("dessin", lambda d, x, y, w, h, S: None)
    return sec


def _pastille(etat: Dict) -> Tuple[str, str]:
    """(le mot, la couleur) de l'état global de la note — le résumé de l'écran."""
    from . import saf as M
    synth = [SR.synthese(f, M.calculer(f)) for f in etat["fondations"]]
    sn = [M.synthese_note(f, M.calculer(f), x) for f, x in zip(etat["fondations"], synth)]
    r = M.resume_note(etat, sn)
    return r["statut"], COULEURS.get(r["niveau"], "att")


def sections(etat: Dict, style: Optional[Planche] = None) -> List[Dict]:
    """Les planches de la note, fondation par fondation : Données, Charges et combinaisons, Sol, Efforts, Principe
    d'armatures, Comparaison avec SCIA (et l'annexe sol). Une fondation sans calcul n'a que Données et Sol."""
    from . import saf as M                                        # (le calcul est celui de l'écran)
    style = style or Planche()
    del FIGURES[:]
    mot, couleur = _pastille(etat)
    plusieurs = len(etat["fondations"]) > 1
    out = []
    for el in etat["fondations"]:
        calc = M.calculer(el)
        s = SR.synthese(el, calc)
        F = {"poutre": TITRE + (f" {el.get('nom') or ''}".rstrip() if plusieurs else ""), "beton": f"{_dims(el)} · {el['dalle']['beton']}",
             "acier": "", "etat": mot, "couleur": couleur, "fondation": el.get("nom") or ""}
        chapitres = []
        if calc["par_k"] is not None and calc["modele"]["cas"]:
            chapitres.append(page_charges(el, calc, F, style))
        chapitres.append(page_sol(el, calc, s, F, style))
        fer = M.armatures(el, calc) if s.get("r_vue") is not None and not s.get("pp") else None
        if fer is not None:
            ef = M.efforts(el, calc)
            sc = SCM.rapports(el, ef, calc["par_k"])
            if not fer.get("hors_domaine"):
                chapitres.append(page_efforts(el, calc, ef, fer, F, style))
            chapitres.append(page_principe(el, calc, ef, fer, F, style))
            chapitres.append(page_scia(el, calc, sc, F, style, s))
        elif el["sol"].get("reprise"):
            chapitres.append(page_scia(el, calc, None, F, style, s))
        flux = dict(F, section="Vérification", bande=False, n_cols=2, verifs=[])
        for ch in chapitres:
            flux["verifs"].append({"chapitre": ch["section"]})
            flux["verifs"] += ch["verifs"]
        out += [_numeroter(page_donnees(el, calc, F, style)), _numeroter(flux)]
    return out


def nom_note(etat: Dict, infos: Optional[Dict]) -> str:
    """Le nom de la note : celui que la note du site porte ; à défaut, le projet de la fondation quand elle est seule,
    son nom, le titre du module sinon."""
    nom = str((infos or {}).get("nom") or "").strip()
    if nom:
        return nom
    fs = etat.get("fondations") or []
    if len(fs) == 1:
        return str(fs[0]["projet"].get("projet") or fs[0].get("nom") or "").strip() or TITRE
    return TITRE


def infos_doc(etat: Dict, infos: Optional[Dict]) -> Dict:
    """Les « infos » de la garde depuis celles de la note du site (projet = le chantier, à défaut le nom de la note ;
    partie = le nom de la note quand il y a un chantier, sinon le titre)."""
    infos = infos or {}
    nom, chantier = nom_note(etat, infos), str(infos.get("chantier") or "").strip()
    out = {"nom_projet": chantier or nom, "partie": nom if chantier else TITRE, "indice": str(infos.get("indice") or "").strip() or "A"}
    if str(infos.get("date") or "").strip():
        out["date"] = str(infos.get("date")).strip()
    return out


def objet(etat: Dict) -> str:
    """La ligne d'objet de la garde : « Fondation de grue — dalle 8,00 × 8,00 × 0,50 m — C30/37 — grue 132 HC »."""
    out = []
    for el in etat["fondations"]:
        t = f"{TITRE} — dalle {_dims(el)} — {el['dalle']['beton']}"
        if el["projet"].get("grue"):
            t += f" — grue {el['projet']['grue']}"
        out.append(t)
    return " ; ".join(out)


def lire_annexes(infos: Optional[Dict]) -> Tuple[List[Tuple[str, bytes, int]], List[str]]:
    """Les notes SCIA jointes (``infos["annexes"]`` : [{nom, base64}]) : [(nom, octets, pages)], et ce qui n'a pas pu être
    joint (le nom et la raison)."""
    ok, refus = [], []
    for a in (infos or {}).get("annexes") or []:
        nom = str((a or {}).get("nom") or "note SCIA.pdf")[:120]
        try:
            octets = base64.b64decode(str(a.get("base64") or ""), validate=False)
            if octets[:5] != b"%PDF-":
                raise PDF.PdfIllisible("ce n'est pas un PDF")
            n = PDF.nombre_de_pages(octets)
            ok.append((nom, octets, n))
        except Exception as e:                                   # noqa: BLE001 — un PDF illisible n'empêche pas la note
            refus.append(f"{nom} ({str(e)[:80]})")
    return ok, refus


def _compter(style: Planche, doc_meta: Dict, secs: List[Dict]) -> List[int]:
    return [style.pages_planche(doc_meta, sec) for sec in secs]


def document(etat: Dict, infos: Optional[Dict], chemin: str, annexes: Optional[List[Tuple[str, bytes, int]]] = None):
    """Le document construit (pas encore écrit) : la garde et les planches. ``d.warnings`` : ce qui déborde encore."""
    from . import saf as M
    etat = M.normaliser(etat)
    style = Planche()
    doc_meta = ndc_data.construire_doc(infos_doc(etat, infos), date_defaut=datetime.today().strftime("%d/%m/%Y"))
    secs = sections(etat, style)
    pages = _compter(style, doc_meta, secs)
    mot, couleur = _pastille(etat)
    debut, chapitres = 2, []
    for sec, n in zip(secs, pages):
        pre = f"{sec['fondation']} · " if len(etat["fondations"]) > 1 else ""
        if sec.get("bande", True):
            chapitres.append((pre + sec["section"], f"page {debut}" if n == 1 else f"pages {debut} à {debut + n - 1}"))
        else:
            for nom, p in sec.get("_chapitres") or []:
                chapitres.append((pre + nom, f"page {debut + p - 1}"))
        debut += n
    fin = debut - 1
    annexes = annexes or []
    n_ann = sum(a[2] for a in annexes)
    p2 = {"titre": PARTIE_2, "etat": "", "couleur": style.acc,
          "sous": (f"jointe : {len(annexes)} PDF — " + " · ".join(a[0] for a in annexes)) if annexes else "à joindre (« Joindre les notes SCIA »)",
          "pages": (f"pages {fin + 1} à {fin + n_ann}" if n_ann > 1 else f"page {fin + 1}") if annexes else "—"}
    doc_meta["garde_saf"] = {"objet": objet(etat), "parties": [
        {"titre": PARTIE_1, "etat": mot, "couleur": {"ok": style.ok, "att": style.att, "ko": style.ko}[couleur], "pages": f"pages 2 à {fin}",
         "chapitres": chapitres}, p2]}
    return style.build(chemin, sections=secs, doc=doc_meta), fin


def pdf(etat: Dict, infos: Optional[Dict] = None) -> Tuple[bytes, Optional[str]]:
    """La note PDF (octets) et, s'il y a lieu, l'avis à dire à l'écran (une note SCIA qui n'a pas pu être jointe). Ce qui
    déborderait encore est REFUSÉ avec un message : jamais une note tronquée. Les notes SCIA jointes vont à la fin
    (``pdf_reunion.reunir`` : leurs pages telles quelles), un signet par partie."""
    annexes, refus = lire_annexes(infos)
    fd, chemin = tempfile.mkstemp(suffix=".pdf", prefix="note_saf_")
    os.close(fd)
    try:
        d, fin = document(etat, infos, chemin, annexes)
        if d.warnings:
            raise ValueError("La note ne tient pas sur sa planche (" + " ; ".join(d.warnings) + ") : elle ne sort pas tronquée. À signaler "
                             "au bureau ; l'écran garde tout le calcul.")
        d.save()
        with open(chemin, "rb") as f:
            octets = f.read()
    finally:
        try:
            os.remove(chemin)
        except OSError:
            pass
    avis = None
    if annexes:
        try:
            signets = [(PARTIE_1, 1)]
            p = fin + 1
            for nom, _o, n in annexes:
                signets.append((f"{PARTIE_2} — {nom}", p))
                p += n
            octets = PDF.reunir([octets] + [a[1] for a in annexes], signets=signets, titre=nom_note(etat, infos), auteur="")
        except Exception as e:                                   # noqa: BLE001
            refus.append(f"assemblage ({str(e)[:80]})")
            octets_seuls, _ = pdf(etat, dict(infos or {}, annexes=[]))
            return octets_seuls, "Notes SCIA non jointes : " + " ; ".join(refus)
    if refus:
        avis = "Notes SCIA non jointes : " + " ; ".join(refus)
    return octets, avis


def fichier(etat: Dict, infos: Optional[Dict] = None) -> Dict:
    """Le fichier à télécharger : {"nom", "type": "application/pdf", "base64"[, "avis"]} (le routeur le contrôle)."""
    infos = dict(infos or {})
    octets, avis = pdf(etat, infos)
    out = {"nom": C.nom_fichier(f"Note de calcul - {nom_note(etat, infos)}", "pdf"), "type": "application/pdf",
           "base64": base64.b64encode(octets).decode("ascii")}
    if avis:
        out["avis"] = avis[:300]
    return out

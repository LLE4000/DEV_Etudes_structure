# -*- coding: utf-8 -*-
"""Note de calcul « Cornière d'appui » — sur le kit ``ndc_pdf`` des notes du site (palette Encre, mêmes marges et mêmes
polices que les notes du garde-corps, des soudures et des assemblages), symboles COMPOSÉS (``ndc_pdf.riche``), peu de
prose. Décisions : ``docs/acier/CORNIERE_DECISIONS.md`` § 5 ; ce qui est livré : ``docs/acier/CORNIERE_MOTEUR.md`` § 11.

UNE NOTE = la page de garde du site (A4 portrait : projet, partie, date, indice, sommaire — ``ndc_pdf.styles.Style.
garde_lignes``, la garde des notes Poutre, Dalle et Assemblages) puis, PAR CORNIÈRE, deux pages A4 paysage :

Page 1 · CALCUL —
- bandeau : statut, η max, la section gouvernante et la combinaison de jeux qui la gouverne, la configuration et le mode ;
- colonne 1 : HYPOTHÈSES en lignes « libellé · valeur » (cornière, configuration, acier, méthode aux contraintes
  admissibles avec le coefficient affiché et σ_adm, τ_adm, le mode — pivot, frottement nul —, les jeux, la tranche),
  la CHARGE (ELS et ELU) et l'ANCRAGE, la COUPE cotée peinte depuis les scènes partagées avec l'écran
  (``acier.corniere.schemas``, peintre de scène du kit) ;
- colonnes 2 et 3 : 1 · GÉOMÉTRIE DES SECTIONS (W, A, largeur efficace, section nette), 2 · EFFORTS ET BRAS DE LEVIER
  (la chaîne de chaque section, avec la combinaison qui la gouverne et, en mode historique, les cellules du classeur),
  3 · VÉRIFICATION DES SECTIONS (le tableau σ, τ, σ_éq, η de S1 à S4, des maxima du congé et de la section nette, puis
  les formules de la section gouvernante), 4 · ANCRAGES (N et V par ancrage, ELS et ELU, positions, informations) ;
  le RÉCAPITULATIF (barres de taux, combinaisons) ; la CONCLUSION.
  Chaque formule est écrite « symbole = expression = valeurs = résultat » (``formules.py`` : les gabarits des formules
  que le moteur nomme, ré-évalués par les tests).

Page 2 · DÉTAILS ET MÉTHODE — les vues MOMENTS et CONTRAINTES (les trois positions de la charge et l'enveloppe, la
bande de tolérance de l'ancrage, le maximum du congé) et l'ÉLÉVATION (la tranche, les ancrages, la diffusion de la
charge), peintes depuis les mêmes scènes ; les textes de MÉTHODE du mode et de la configuration (``methode.py`` : ce
qui change quand on retourne la cornière, les jeux et l'enveloppe), les LIMITES (les avertissements du moteur, tels
quels), les RÉFÉRENCES, le cartouche.

DÉBORDEMENT — comme le garde-corps : la note est d'abord composée en taille normale, puis serrée si une colonne
déborde ; si une colonne déborde encore, les blocs du bas de la colonne passent, un à un, sur une PAGE DE SUITE
(``Suite``, ``mise_en_page``), jusqu'à ce que tout tienne — jamais une note tronquée. Une note qui ne tiendrait toujours
pas (un seul bloc plus haut qu'une colonne) est refusée avec un message, ce qui ne s'est pas vu sur les états d'essai.

``generer_note(etat_ou_cornieres, infos, chemin=None) -> bytes`` : l'état de la note du site ({"version": 2,
"cornieres": [...]}) ou une liste de cornières ; ``infos`` = {nom_projet, partie, indice, date} (la correspondance avec
la note du site est dans api_web/note_corniere.py). ``derniers_avertissements`` : les débordements (vides si tout tient).
"""
import io
import math
import os
import re
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from reportlab.lib.pagesizes import A4, landscape

from acier.assemblages.poutre_poutre.doubles_cornieres.rapport import peindre as peindre_scene   # peintre des scènes du kit
from acier.formats import F, f0
from acier.garde_corps import membres as ME   # les seuils du code couleur des taux (la source unique du site : commun.niveau)
from ndc_pdf import data as ndc_data
from ndc_pdf import kit as KIT
from ndc_pdf import riche as RI
from ndc_pdf.kit import Doc, Frame, chip, hx, mix
from ndc_pdf.mathx import MathStyle
from ndc_pdf.styles import Encre

from . import entree as E
from . import formules as FO
from . import methode as MT
from . import moteur as MO
from . import schemas as SC

A4L, A4P = landscape(A4), A4
M, BAS, GAP = 22, 40, 12
LARG = (286, 246)
S_TAB_H = 5.6
LEAD_TAB = 8.8
LAB_W = 58
CART_H = 40
PART_VUES = 0.46                 # la part de la hauteur de la page 2 donnée à la rangée des trois vues ; le reste aux textes
_DECIMALE = re.compile(r"(?<=\d)\.(?=\d)")
TEXTE_DESSIN = 5.6               # pt : la taille du texte des dessins à l'échelle (coupe, élévation) dans la note
TITRE = "Cornière d'appui"
TITRE_NOTE = "Note de calcul — Cornières d'appui"
STATUTS = {MO.OK: ("CONFORME", "ok"), MO.NOK: ("NON CONFORME", "ko")}
VERT_PALE = "#D9ECDF"            # le fond d'une pastille η sous le seuil d'attention (code couleur du bureau, 06/11/2026)
VARIANTES = ((1.0, "normale"), (0.92, "serrée"))
LEGENDE = ("Coupe à l'échelle, cotes en mm · sections 1 à 4 en orange · R la réaction ELS (positions X ± dx en tirets) · H la traction "
           "totale des ancrages · C la compression du mur · η = σéq/σadm · chaque grandeur garde la combinaison de jeux la plus défavorable (enveloppe).")
LEGENDE_GARDE = "Vert : conforme   ·   Rouge : non conforme   —   méthode aux contraintes admissibles, efforts ELS, η = σéq/σadm"
# les blocs de chaque colonne, (page, rang) → méthodes de la page, dans l'ordre de lecture : la composition des pages ET
# l'ordre dans lequel les blocs du bas d'une colonne qui déborde passent en page de suite
BLOCS = {(1, 0): ("hypotheses", "charges", "coupe"),
         (1, 1): ("geometrie", "efforts", "sections"),
         (1, 2): ("ancrages", "recapitulatif"),
         (2, 0): ("vue_moments",), (2, 1): ("vue_contraintes",), (2, 2): ("vue_elevation",),
         (2, 3): ("methode",), (2, 4): ("retournement",), (2, 5): ("limites", "references")}
NOMS_BLOCS = {"hypotheses": "hypothèses", "charges": "charge et ancrage", "coupe": "coupe", "geometrie": "géométrie des sections",
              "efforts": "efforts et bras de levier", "sections": "vérification des sections", "ancrages": "ancrages",
              "recapitulatif": "récapitulatif", "vue_moments": "vue des moments", "vue_contraintes": "vue des contraintes",
              "vue_elevation": "élévation", "methode": "modèle et méthode", "retournement": "retournement, jeux", "limites": "limites",
              "references": "références"}
TITRES_PAGES = {1: "calcul", 2: "détails et méthode"}
# le nom d'une vérification dans le récapitulatif (le moteur nomme « Section 1 · aile horizontale, pied du congé ») et son
# nom ABRÉGÉ dans le tableau des sections (une ligne par vérification)
NOMS_COURTS = {"S1": "S1 · aile horizontale, pied du congé", "S2": "S2 · aile horizontale, nu de l'aile verticale",
               "S3": "S3 · aile verticale, pied du congé", "S4": "S4 · aile verticale, nu de l'aile horizontale",
               "max_h": "maximum du congé, aile horizontale", "max_v": "maximum du congé, aile verticale",
               "nette": "section nette au trou"}
NOMS_TABLE = {"S1": "S1 · aile h., pied du congé", "S2": "S2 · aile h., nu de l'aile v.", "S3": "S3 · aile v., pied du congé",
              "S4": "S4 · aile v., nu de l'aile h.", "max_h": "max. du congé, aile h.", "max_v": "max. du congé, aile v.",
              "nette": "nette au trou"}
ANCRAGES_ELS = ("Mmax_j", "bras_A", "y_a", "bras_B", "H_eq", "N_ELS_hist", "N_ELS_eq", "V_ELS")
ANCRAGES_ELU = ("R_ELU_auto", "rapport_ELU", "ELU")


class Debord(str):
    """Un débordement de colonne : le texte (« page 1, colonne 320 : 28 pt ») et la colonne qui déborde, ``colonne`` =
    (page, rang) — ce que ``mise_en_page`` lit pour savoir quel bloc passer en page de suite."""

    def __new__(cls, page: int, rang: int, x: float, pt: float):
        s = str.__new__(cls, f"page {page}, colonne {x:.0f} : {pt:.0f} pt")
        s.colonne = (page, rang)
        return s


def _etat_page(page):
    """L'état d'une page (et de la page 1 qu'une page 2 emprunte pour ses briques) : ses attributs, copiés."""
    return [(o, dict(vars(o))) for o in (page, getattr(page, "note1", None)) if o is not None]


def _remettre(etat):
    for o, attrs in etat:
        o.__dict__.clear()
        o.__dict__.update(attrs)


def _colonne(page, d, fr, cle):
    """Les blocs de la colonne ``cle`` posés dans ``fr`` ; ceux que la page de suite prend (les derniers) y partent."""
    suite = page.suite
    reportes = []
    for nom in BLOCS[cle]:
        if suite is not None and suite.prend(cle, nom):
            reportes.append(nom)
        else:
            getattr(page, nom)(d, fr)
    if reportes:
        suite.reporter(d, page, cle, reportes, fr)


# ------------------------------------------------------------------ nombres
def _fmt(x, nd=1) -> str:
    if x is None:
        return "—"
    if isinstance(x, float) and (math.isinf(x) or math.isnan(x)):
        return "∞"
    return f"{x:,.{nd}f}".replace(",", " ").replace(".", ",")


def _pct(t) -> str:
    """Un taux comme l'écran l'écrit dans les vues : une décimale (« 95,8 % »)."""
    if t is None:
        return "—"
    return F(100.0 * t, 1).replace("-", "−", 1) + " %"


def _kn(x) -> str:
    return F(x, 2).replace("-", "−", 1) + " kN"


def _knm(x) -> str:
    return F(x, 3).replace("-", "−", 1)


def _mm(x) -> str:
    return f0(x)


def francais(txt: str) -> str:
    """Un texte du moteur avec ses nombres à la virgule française : les messages du moteur écrivent certains nombres
    avec le point (« 52.50 kN ») — la note les FORMATE, elle ne les recalcule pas."""
    return _DECIMALE.sub(",", str(txt))


# ------------------------------------------------------------------ le calcul d'une cornière
class Calcul:
    """Tout ce que la note imprime d'une cornière, calculé une fois : le résultat du moteur, les dessins (à la demande,
    par boîte)."""

    def __init__(self, corniere: Dict):
        self.c = MO.normaliser(corniere)
        self.res = MO.verifier(self.c)
        self._vues: Dict = {}

    def nom(self) -> str:
        return str(self.c.get("nom") or "Cornière")

    def vue(self, cle: str, w: float, h: float, texte: float = TEXTE_DESSIN):
        """Le dessin ``cle`` pour une boîte (w, h) en points : les vues à l'échelle (coupe, élévation) sont reconstruites
        jusqu'à ce que leur texte se lise à ``texte`` pt ; les diagrammes, mis en page en tailles de texte, prennent la
        taille de la boîte. ValueError quand la vue ne se dessine pas (``schemas``)."""
        key = (cle, round(w), round(h), texte)
        if key not in self._vues:
            fn = SC.FONCTIONS[cle]
            if cle in ("coupe", "elevation"):
                d0 = fn(self.res, None, False)
                self._vues[key] = SC.ajuster(lambda fs: fn(self.res, fs, False), (w, h), texte, d0.fs)
            else:
                self._vues[key] = fn(self.res, None, False)
        return self._vues[key]

    def hypotheses(self) -> List[Tuple[str, str]]:
        """Les lignes « libellé · valeur » de la colonne 1 (valeur en texte riche)."""
        res, c = self.res, self.c
        sec, g, mat, cfg, mode = res["section"], res["geometrie"], res["materiau"], res["configuration"], res["mode"]
        L: List[Tuple[str, str]] = []
        if sec["source"] == "catalogue":
            piece = f"{sec['designation']} du catalogue ({sec['norme']}), {_fmt(sec['masse'], 1)} kg/m"
        else:
            piece = f"{sec['designation']} mm, saisie libre ({_fmt(sec['masse'], 1)} kg/m calculés)"
        L.append(("Cornière", f"{piece} · aile horizontale {_mm(sec['larg'])} mm (aile {sec['aile_horizontale']} du nom), aile verticale "
                              f"{_mm(sec['haut'])} mm · $e$ = {_mm(g['e'])} mm · congé $r$ = {_mm(g['r'])} mm"))
        if cfg == "A":
            conf = ("A — aile horizontale EN BAS : la charge pose sur l'aile horizontale, l'aile verticale monte le long du mur, "
                    "l'ancrage est au-dessus de l'angle ; le talon s'appuie sur le mur, l'aile verticale est tendue")
        else:
            conf = ("B — aile horizontale EN HAUT (retournée) : l'aile verticale descend le long du mur, l'ancrage est au-dessous de "
                    "l'angle ; la cornière pivote autour du bout libre, l'aile verticale est comprimée, le trou est dans le moment maximal")
        L.append(("Configuration", conf))
        L.append(("Acier", f"{mat['nuance']} · $f_y$ = {_mm(mat['fy'])} N/mm² · coefficient {f0(mat['coef'])} sur $f_y$"))
        L.append(("Méthode", f"aux contraintes admissibles, efforts ELS non majorés : $σ_{{adm}}$ = $f_y$/{f0(mat['coef'])} = {_fmt(mat['sigma_adm'])} N/mm² · "
                             f"$τ_{{adm}}$ = $σ_{{adm}}$/√3 = {_fmt(mat['tau_adm'])} N/mm² · $σ_{{eq}}$ = √(σ² + 3τ²) ≤ $σ_{{adm}}$ — "
                             "ce n'est pas une justification EC3 aux états limites"))
        if mode == E.HISTORIQUE:
            L.append(("Mode", f"méthode historique, feuille P8 du classeur : sections avec le pivot $e$/2 ($YH$ = $Y$ − $e$/2 = {_mm(c['Y'] - g['e'] / 2)} mm), "
                              f"$τ$ = $R$/($L$·$t$) dans les deux ailes, traction d'ancrage sur le bras $Y$ − jeu = {_mm(c['Y'] - c['jeu_ancrage'])} mm "
                              f"(C76 = {_mm(c['jeu_ancrage'])} mm) · maxima du congé en information"))
        elif cfg == "A":
            L.append(("Mode", f"équilibre d'un seul corps libre, frottement nul au mur : compression du mur à $y_c$ = {_mm(g['yc'])} mm du dessous, "
                              f"cisaillement d'ancrage à $u_V$ = {_mm(g['uV'])} mm du nu du mur ; $N$ = +$R$ et $V$ = $H$ dans l'aile verticale ; "
                              "le maximum le long du congé entre dans le verdict"))
        else:
            L.append(("Mode", f"équilibre de la cornière retournée (imposé en B), frottement nul : contact au bout libre à $c$ = {_mm(g['yc'])} mm, "
                              f"cisaillement d'ancrage à $u_V$ = {_mm(g['uV'])} mm ; moment constant $R$·($X′$ − $e$/2) de l'ancrage à l'angle, "
                              "$N$ = −$R$ ; section nette au trou"))
        if c["dx"] > 0 or c["dy"] > 0:
            L.append(("Jeux", f"charge à $X$ − {_mm(c['dx'])}, $X$, $X$ + {_mm(c['dx'])} ; ancrage à $Y$ − {_mm(c['dy'])}, $Y$, $Y$ + {_mm(c['dy'])} : "
                              f"{len(res['combinaisons'])} combinaison(s), chaque grandeur garde la plus défavorable (enveloppe)"))
        else:
            L.append(("Jeux", "aucun ($dx$ = $dy$ = 0) : une seule combinaison, la charge à $X$ et l'ancrage à $Y$"))
        tranche = (f"courte : $L$ = {_mm(g['L'])} mm est la largeur de toutes les sections ; {g['n']} ancrage(s) se partagent $R$ et la traction")
        if g["a"] is not None:
            tranche += (f" · largeur d'appui $a$ = {_mm(g['a'])} mm : $L_{{eff}}$ = min($L$ ; $a$ + 2·$Xe$) = {_fmt(g['L_eff'])} mm sur l'aile "
                        f"horizontale (σ × {_fmt(g['facteur_L_eff'], 3)})")
        L.append(("Tranche", tranche))
        if res["mode_demande"] != mode:
            L.append(("Note", "mode historique demandé : sans effet en configuration B, où les formules du classeur ne s'appliquent pas"))
        return L

    def charge_et_ancrage(self) -> List[Tuple[str, str]]:
        res, c = self.res, self.c
        g, ef, anc = res["geometrie"], res["efforts"], res["ancrages"]
        jx = f" ± {_mm(c['dx'])}" if c["dx"] > 0 else ""
        jy = f" ± {_mm(c['dy'])}" if c["dy"] > 0 else ""
        a = f"$a$ = {_mm(g['a'])} mm" if g["a"] is not None else "$a$ = $L$ (toute la tranche)"
        elu = (f"$R_{{ELU}}$ = 1,5·$R$ = {_kn(ef['R_ELU'])} (auto : à remplacer par la combinaison réelle 1,35·G + 1,5·Q)" if ef["R_ELU_auto"]
               else f"$R_{{ELU}}$ = {_kn(ef['R_ELU'])} (saisie)")
        L = [("Charge", f"$R$ = {_kn(ef['R'])} à l'ELS, posée sur l'aile horizontale à $X$ = {_mm(c['X'])}{jx} mm du nu du mur, largeur d'appui {a} · {elu}"),
             ("Ancrage", f"{anc['n']} × Ø{_mm(anc['d'])} (trou $d_0$ = {_mm(anc['d0'])} mm) à $Y$ = {_mm(c['Y'])}{jy} mm de la face extérieure de l'aile "
                         f"horizontale, entraxe {_mm(g['L'] / anc['n'])} mm, le premier à {_mm(g['L'] / (2 * anc['n']))} mm du bord · chevilles non vérifiées "
                         "(efforts fournis pour Hilti PROFIS)")]
        return L


# ------------------------------------------------------------------ page 1 · calcul
class Note:
    """Page 1 · CALCUL."""

    def __init__(self, calc: Calcul, doc_meta: Dict, k: float = 1.0, suite: Optional["Suite"] = None):
        self.C, self.doc, self.k = calc, doc_meta, k
        self.suite = suite
        self.S = Encre()
        self.warnings: List[str] = []
        self.s_txt = 6.1 * k
        self.s_tab = 6.0 * k
        self.lead = LEAD_TAB * k
        self.echelle = None
        self._ms: Dict = {}
        self.n_bloc = 0

    def numero(self) -> str:
        self.n_bloc += 1
        return f"{self.n_bloc} · "

    # --- composition
    def ms(self, col=None, op=None, res=None) -> MathStyle:
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
    def couleurs_eta(self, taux) -> Tuple[str, str]:
        """(texte, fond) d'une pastille η — le code couleur des taux du site (seuils ``membres.statut``) : rouge au-dessus de
        100 %, vert foncé de 95 à 100 %, vert pâle (texte vert) en dessous."""
        S = self.S
        s = ME.statut(taux)
        if s == ME.NOK:
            return "#FFFFFF", S.ko
        if s == ME.ATTENTION:
            return "#FFFFFF", S.ok
        return S.ok, VERT_PALE

    def titre(self, d, fr, txt, taux=None):
        """Un titre de bloc en petites capitales espacées, sa pastille η à droite ; il se resserre avant de se tronquer."""
        S = self.S
        fr.down(10)
        wc = 0.0
        if taux is not None:
            lab = "η " + _pct(taux)
            wc = d.w(lab, "Carlito-Bold", 5.8) + 10
            fg, bg = self.couleurs_eta(taux)
            chip(d, fr.x1 - wc, fr.y - 2.5, lab, "Carlito-Bold", 5.8, fg, bg, pad=5, h=8.4)
        avail = fr.w - (wc + 6 if wc else 0.0)
        size, track = 6.2, 1.3
        while size > 5.0 and d.w(txt, S.f_v, size) + track * len(txt) > avail:
            size -= 0.2
            track -= 0.1
        d.t(fr.x, fr.y, d.fit(txt, S.f_v, size, avail - track * len(txt)), S.f_v, size, S.acc, track=track)
        fr.down(3.2)
        d.line(fr.x, fr.y, fr.x1, fr.y, mix(S.rule, S.ink, 0.12), 0.5)

    def etiquette(self, d, fr, txt):
        """Le libellé d'un groupe de formules, en petites capitales grises (tronqué à la colonne)."""
        fr.down(self.s_txt * 1.6)
        size, track = 4.9 * self.k, 0.6
        d.t(fr.x, fr.y, d.fit(txt.upper(), self.S.f_v, size, fr.w - track * len(txt)), self.S.f_v, size, self.S.mut, track=track)

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
        if titre:
            self.titre(d, fr, titre)
        fr.down(8)
        KIT.table(d, fr, cols, lignes, self.S, self.s_tab, self.lead, S_TAB_H, ms=self.ms, note=note, barre=self.barre)

    def formules(self, d, fr, groupes, indent=4):
        """Les formules par groupe : l'étiquette du groupe, puis chaque substitution « nom = littéral = numérique » avec
        son résultat calé à droite (les taux η en vert ou en rouge)."""
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
                                 indent=indent, retrait=10, lead=1.38, min_size=4.8, cale="droite" if col_res is not None else "suite")

    @staticmethod
    def res_tex(sub, res: str) -> str:
        if sub.nd == -1 or not sub.unit:
            return r"\res{" + RI.tex(res) + "}"
        return r"\res{" + RI.tex(FO.fmt(sub.res, sub.nd)) + r" \t{" + sub.unit + "}}"

    # --- bandeau
    def bandeau(self, d):
        S, C, doc = self.S, self.C, self.doc
        res = C.res
        W, H = d.W, d.H
        y = H - M - 4
        d.t(M, y, doc.get("bureau", ""), S.f_v, 8.0, S.acc, track=1.4)
        d.t(W - M, y, "Note de calcul · " + doc.get("date", "") + " · indice " + str(doc.get("indice", "")), S.f_b, 8, S.mut, "right")
        y -= 15
        d.t(M, y, d.fit(f"{TITRE} — {C.nom()}", S.f_h, 12.5, W - 2 * M - 310), S.f_h, 12.5, S.ink)
        ident = " · ".join(str(x) for x in (doc.get("projet"), doc.get("partie")) if x)
        d.t(W - M, y, d.fit(ident, S.f_b, 8, 300), S.f_b, 8, S.ink, "right")
        y -= 8
        d.line(M, y, W - M, y, S.ink, 1.0)
        y -= 15
        libelle, cle = STATUTS[res["statut"]]
        col = S.ok if cle == "ok" else S.ko
        w = chip(d, M, y - 3.2, libelle, S.f_v, 6.6, "#FFFFFF", col)
        x = M + w + 8
        t1 = "η max " + _pct(res["taux_max"])
        d.t(x, y, t1, "Carlito-Bold", 8.6, col)
        x += d.w(t1, "Carlito-Bold", 8.6) + 5
        g = res["gouvernant"]
        t2 = f"— gouverne : {g['nom']} · {g['combinaison']['libelle']}"
        t3 = f"Configuration {res['configuration']} · {E.LIBELLES_MODE[res['mode']]}"
        wt = d.w(t3, "Carlito-Bold", 7.6)
        xr = W - M - wt
        self.rt(d, x, y, t2, 8, S.ink, width=xr - x - 6)
        d.t(xr, y, t3, "Carlito-Bold", 7.6, S.acc)
        y -= 6
        d.line(M, y, W - M, y, mix(S.rule, S.ink, 0.3), 0.6)
        return y - 4

    # --- colonne 1 : hypothèses, charge et ancrage, coupe
    def hypotheses(self, d, fr):
        self.titre(d, fr, "HYPOTHÈSES")
        fr.down(1)
        for label, val in self.C.hypotheses():
            self.ligne_hyp(d, fr, label, val, self.S.mut if label == "Note" else None)
        fr.down(2)

    def charges(self, d, fr):
        self.titre(d, fr, "CHARGE ET ANCRAGE — ELS et ELU")
        fr.down(1)
        for label, val in self.C.charge_et_ancrage():
            self.ligne_hyp(d, fr, label, val)
        fr.down(2)

    def coupe(self, d, fr):
        """La coupe cotée (le schéma statique), à la place qui reste dans la colonne ; sa légende dessous."""
        S, C = self.S, self.C
        self.titre(d, fr, "COUPE — schéma statique coté (moments, contraintes et élévation p. 2)")
        s_leg = 5.2 * self.k
        lignes = RI.envelopper(RI.auto(SC.legende(C.res, "coupe")), S.f_b, s_leg, self.ms(S.mut), fr.w)[:3]
        h_leg = len(lignes) * s_leg * 1.25 + 3
        h = max(60.0, fr.room() - h_leg - 6)
        fr.down(h + 3)
        try:
            dv = C.vue("coupe", fr.w, h)
        except (ValueError, ArithmeticError):
            # une coupe qu'on ne sait pas dessiner (proportions extrêmes) : le cadre qui le dit, le calcul est rendu
            d.box(fr.x, fr.y, fr.w, h, stroke=S.mut, lw=0.6, r=4)
            d.t(fr.x + fr.w / 2, fr.y + h / 2 + 4, "Dessin impossible à cette échelle", S.f_v, 7.0, S.acc, "center")
            d.t(fr.x + fr.w / 2, fr.y + h / 2 - 6, "les dimensions saisies sont trop disproportionnées — le calcul reste valable", S.f_b, 5.6, S.mut, "center")
        else:
            self.echelle = peindre_scene(d, dv, fr.x, fr.y, fr.w, h)
        for l in lignes:
            fr.down(s_leg * 1.25)
            RI.dessiner(d, fr.x, fr.y, l, S.f_b, s_leg, S.mut, self.ms(S.mut))

    # --- colonnes 2 et 3
    def _lib(self, cle: str, v: Dict, b: Optional[Dict], cles=None) -> str:
        """L'étiquette d'un groupe de formules : la vérification, la combinaison qui la gouverne et, en mode historique,
        les cellules du classeur."""
        hist = self.C.res["mode"] == E.HISTORIQUE
        nom = cle if cle.startswith("S") else NOMS_COURTS.get(cle, cle)
        if cle == "nette":
            nom += f" (face {v.get('face', '')})"
        txt = f"{nom} — {v['combinaison']['libelle']}"
        if hist and b is not None:
            cel = FO.cellules(b, cles)
            if cel:
                txt += " · " + " ".join(cel)
        return txt

    def geometrie(self, d, fr):
        S, C = self.S, self.C
        res = C.res
        g = res["geometrie"]
        self.titre(d, fr, f"{self.numero()}GÉOMÉTRIE DES SECTIONS")
        fr.down(0.5)
        secs = {s["cle"]: s for s in g["sections"]}
        txt = (f"$s$ : distance au dos de l'autre aile · S1 (aile horizontale) et S3 (aile verticale) au pied du congé : $s$ = $e$ + $r$ = "
               f"{_mm(secs['S1']['s'])} mm, $t$ = $e$ = {_mm(secs['S1']['t'])} mm · S2 et S4 au nu de l'autre aile : $s$ = $e$ = {_mm(secs['S2']['s'])} mm, "
               f"$t$ = $e$ + $r$ = {_mm(secs['S2']['t'])} mm · largeur des coupes $b$ = $L$ = {_mm(g['L'])} mm"
               + (f", $L_{{eff}}$ = {_fmt(g['L_eff'])} mm sur l'aile horizontale" if g["a"] is not None else ""))
        self.rp(d, fr, txt, size=self.s_txt * 0.95, col=S.mut)
        # W (et A) des quatre sections, groupées quand (b, t) sont identiques
        groupes: List = []
        for cle in ("S1", "S2", "S3", "S4"):
            b = FO.bloc(res, cle)
            W = next(s for s in FO.coupe(b) if s.nom == "W")
            A = next((s for s in FO.coupe(b) if s.nom == "A"), None)
            key = FO.cle_identique(W)
            for gr in groupes:
                if gr[0] == key:
                    gr[1].append(cle)
                    break
            else:
                groupes.append([key, [cle], W, A])
        subs = []
        for _key, cles, W, A in groupes:
            idx = "".join("₁₂₃₄"[int(c[1]) - 1] for c in cles)
            noms = " = ".join(f"W{'₁₂₃₄'[int(c[1]) - 1]}" for c in cles)
            subs.append(FO.Sub(noms, W.template, W.vals, W.res, W.unit, W.nd))
            if A is not None:
                subs.append(FO.Sub(" = ".join(f"A{'₁₂₃₄'[int(c[1]) - 1]}" for c in cles), A.template, A.vals, A.res, A.unit, A.nd))
            del idx
        gr = [("Modules et aires des coupes", subs)]
        b_eff = FO.bloc(res, "largeur_efficace")
        if b_eff is not None:
            gr.append((f"Largeur efficace, diffusion à 45° — {b_eff['combinaison']['libelle']}", FO.subs(b_eff)))
        b_net = FO.bloc(res, "nette")
        if b_net is not None:
            gr.append(("Section nette au droit des trous (aile verticale)", [s for s in FO.subs(b_net) if s.nom in ("d0", "b_net")]))
        self.formules(d, fr, gr)
        fr.down(2)

    def efforts(self, d, fr):
        S, C = self.S, self.C
        res = C.res
        ef, g = res["efforts"], res["geometrie"]
        hist = res["mode"] == E.HISTORIQUE
        self.titre(d, fr, f"{self.numero()}EFFORTS ET BRAS DE LEVIER")
        fr.down(0.5)
        txt = (f"$R$ = {_kn(ef['R'])} · $M_{{max}}$ = $R$·$X′$ = {_knm(ef['Mmax'])} kN·m (enveloppe) · traction totale des ancrages $H$ = "
               f"{_kn(ef['H'])} = compression du mur $C$, bras {_mm(ef['bras_H'])} mm ({ef['combinaison_H']['libelle']})")
        if res["configuration"] == "B":
            txt += f" · moment constant de l'ancrage à l'angle $R$·($X′$ − $e$/2) = {_knm(ef['M_angle'])} kN·m"
        if hist:
            txt += " · bras nominaux du classeur, le jeu Xj dans les moments, Yj dans ρ"
        self.rp(d, fr, txt, size=self.s_txt * 0.95, col=S.mut)
        vus = set()
        groupes = []
        for v in res["verifications"]:
            if v["cle"] not in ("S1", "S2", "S3", "S4", "nette"):
                continue
            b = FO.bloc(res, v["cle"])
            subs = []
            for s in FO.chaine(b):
                key = FO.cle_identique(s)
                if key in vus:
                    continue
                vus.add(key)
                subs.append(s)
            groupes.append((self._lib(v["cle"], v, b, FO.CHAINE), subs))
        self.formules(d, fr, groupes)
        fr.down(2)

    def sections(self, d, fr):
        S, C = self.S, self.C
        res = C.res
        hist = res["mode"] == E.HISTORIQUE
        gov = res["gouvernant"]["cle"]
        self.titre(d, fr, f"{self.numero()}VÉRIFICATION DES SECTIONS", res["taux_max"])
        fr.down(0.5)
        self.rp(d, fr, f"$σ$ = |M|/W + |N|/A · $τ$ = {'R/(L·t)' if hist else '|V|/A'} · $σ_{{eq}}$ = √(σ² + 3τ²) · η = $σ_{{eq}}$/$σ_{{adm}}$ "
                       f"($σ_{{adm}}$ = {_fmt(res['materiau']['sigma_adm'])} N/mm²) · contraintes en N/mm², $M$ en kN·m, $s$ et $t$ en mm",
                size=self.s_txt * 0.95, col=S.mut)
        rows = []
        for v in res["verifications"]:
            g_ = v["cle"] == gov
            lab = NOMS_TABLE[v["cle"]] + (f" ({v['face']})" if v["cle"] == "nette" else "")
            st = f"{_mm(v['s'])} / {F(v['t'], 2 if v.get('dans_conge') else 0).replace(',00', '')}"
            if v["verdict"]:
                eta = ("eta", _pct(v["taux"]), v["ok"], g_, None)
            else:
                eta = ("ref", _pct(v["taux"]) + " (info)")
            rows.append([("lab", lab, g_), ("num", st), ("num", _knm(v["M"])), ("num", _fmt(v["sigma"])), ("num", _fmt(v["tau"])),
                         ("num", _fmt(v["sigma_eq"])), eta])
        w = fr.w
        cols = [("SECTION", w - 36 - 32 - 28 - 24 - 28 - 28, "left"), ("$s$ / $t$", 36, "right"), ("$M$", 32, "right"), ("$σ$", 28, "right"),
                ("$τ$", 24, "right"), ("$σ_{eq}$", 28, "right"), ("η", 28, "right")]
        mh, mv = res["maxima"]["horizontale"], res["maxima"]["verticale"]

        def ou(v):
            return f"{_fmt(v['recul'], 2)} mm avant le pied du congé" if v["dans_conge"] else "au pied du congé"
        if hist:
            note = (f"(info) en information, hors verdict : la méthode historique rend le classeur ; le maximum du congé vaut {_pct(mh['taux'])} sur l'aile "
                    f"horizontale ({ou(mh)}, {'+' if mh['ecart'] >= 0 else ''}{_fmt(100 * mh['ecart'])} % sur S1) et {_pct(mv['taux'])} sur la verticale ({ou(mv)}).")
        else:
            note = (f"Le maximum du congé entre dans le verdict : aile horizontale {ou(mh)} ({'+' if mh['ecart'] >= 0 else ''}{_fmt(100 * mh['ecart'])} % sur S1), "
                    f"aile verticale {ou(mv)} ({'+' if mv['ecart'] >= 0 else ''}{_fmt(100 * mv['ecart'])} % sur S3).")
        self.table(d, fr, None, cols, rows, note=note)
        vg = next(v for v in res["verifications"] if v["cle"] == gov)
        bg = FO.bloc(res, gov)
        self.formules(d, fr, [("Gouvernante : " + self._lib(gov, vg, bg, FO.COUPE), FO.coupe(bg))])
        fr.down(2)

    def ancrages(self, d, fr):
        S, C = self.S, self.C
        res = C.res
        anc, c = res["ancrages"], C.c
        hist = res["mode"] == E.HISTORIQUE
        b = FO.bloc(res, "ancrages")
        self.titre(d, fr, f"{self.numero()}ANCRAGES — efforts par ancrage (chevilles non vérifiées)")
        fr.down(0.5)
        z = ", ".join(_mm(x) for x in anc["positions_z"])
        txt = (f"{anc['n']} × Ø{_mm(anc['d'])} (trou $d_0$ = {_mm(anc['d0'])} mm) à $z$ = {z} mm du bord de la tranche, à $Y$ = {_mm(c['Y'])} mm · "
               f"bras de la traction {_mm(anc['bras'])} mm · gouverne : {anc['combinaison_N']['libelle']}")
        if hist:
            txt += " · " + " ".join(FO.cellules(b))
        self.rp(d, fr, txt, size=self.s_txt * 0.95, col=S.mut)
        paires = list(zip(b["formules"], FO.subs(b)))
        els = [s for f, s in paires if f["cle"] in ANCRAGES_ELS]
        elu = [s for f, s in paires if f["cle"] in ANCRAGES_ELU]
        info = [s for f, s in paires if f["cle"] not in ANCRAGES_ELS + ANCRAGES_ELU]
        lib_elu = "ELU — R_ELU = 1,5·R (auto)" if anc["R_ELU_auto"] else "ELU — R_ELU saisie"
        self.formules(d, fr, [("ELS, par ancrage", els), (lib_elu, elu), ("Information, sans limite", info)])
        deja = {"pression_diametrale", "arrachement", "pression_support"}
        for i in res["informations"]:
            if i["cle"] in deja:
                continue
            val = _pct(i["valeur"]) if i["cle"] == "autre_mode" else f"{_fmt(i['valeur'], 2 if i['unite'] == 'kN' else 1)} {i['unite']}".strip()
            cb = f" ({i['combinaison']['libelle']})" if i.get("combinaison") else ""
            self.rp(d, fr, f"{i['nom']} : {val}{cb} — {RI.auto(francais(i['texte']))}", size=self.s_txt * 0.95, col=S.mut, lead=1.26)
        fr.down(2)

    def recapitulatif(self, d, fr):
        S, C = self.S, self.C
        res = C.res
        gov = res["gouvernant"]["cle"]
        rows = []
        for v in res["verifications"]:
            g_ = v["cle"] == gov
            lab = NOMS_COURTS[v["cle"]] + (f" (face {v['face']})" if v["cle"] == "nette" else "")
            if v["verdict"]:
                eta = ("eta", _pct(v["taux"]), v["ok"], g_, v["taux"])
            else:
                eta = ("ref", _pct(v["taux"]) + " (information)")
            rows.append([("lab", lab, g_), ("ref", v["combinaison"]["libelle"]), eta])
        w = fr.w
        cols = [("VÉRIFICATION", w - 94 - 54, "left"), ("COMBINAISON", 94, "left"), ("η", 54, "right")]
        note = f"η max {_pct(res['taux_max'])} : {res['gouvernant']['nom']}"
        if res.get("comparaison"):
            cp = res["comparaison"]
            note += f" · en mode {E.LIBELLES_MODE[cp['mode']]} : {_pct(cp['taux_max'])} ({cp['gouvernant']['nom']}), en information"
        self.s_tab = 5.8 * self.k
        self.table(d, fr, "RÉCAPITULATIF — vérifications, combinaisons gouvernantes, taux", cols, rows, note)
        self.s_tab = 6.0 * self.k

    def conclusion(self, d, x, w):
        S, C = self.S, self.C
        res = C.res
        libelle, cle = STATUTS[res["statut"]]
        col = S.ok if cle == "ok" else S.ko
        g, anc, sec = res["gouvernant"], res["ancrages"], res["section"]
        txt = (f"Taux maximal {_pct(res['taux_max'])} — {g['nom'].lower()}, {g['combinaison']['libelle']} — méthode aux contraintes admissibles "
               f"(efforts ELS, coefficient {f0(res['materiau']['coef'])} sur $f_y$), configuration {res['configuration']}, {E.LIBELLES_MODE[res['mode']]}. ")
        if res["statut"] == MO.OK:
            txt += (f"La cornière {sec['designation']} porte la réaction $R$ = {_kn(res['efforts']['R'])} sur {_mm(res['geometrie']['L'])} mm. ")
        else:
            txt += (f"La cornière {sec['designation']} ne porte pas la réaction $R$ = {_kn(res['efforts']['R'])} en l'état : section, position de la charge "
                    "ou de l'ancrage à revoir (le sélecteur de l'écran donne la plus légère qui tient). ")
        txt += (f"Efforts à donner au calcul des chevilles, par ancrage : $N_{{ELS}}$ = {_kn(anc['N_ELS'])}, $V_{{ELS}}$ = {_kn(anc['V_ELS'])} ; "
                f"$N_{{ELU}}$ = {_kn(anc['N_ELU'])}, $V_{{ELU}}$ = {_kn(anc['V_ELU'])} (chevilles non vérifiées ici).")
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
        _colonne(self, d, f1, (1, 0))
        _colonne(self, d, f2, (1, 1))
        hc = self.conclusion(d, x3, w3)
        f3.bottom = BAS + hc + 6
        _colonne(self, d, f3, (1, 2))
        d.t(M, 20, d.fit(LEGENDE, S.f_b, 5.4, W - 2 * M), S.f_b, 5.4, S.mut)
        for rang, f in enumerate((f1, f2, f3)):
            if f.overflow() > 1.0:
                self.warnings.append(Debord(1, rang, f.x, f.overflow()))
        return self.warnings


# ------------------------------------------------------------------ page 2 · détails et méthode
class Methode:
    """Page 2 · DÉTAILS ET MÉTHODE : les trois vues (moments, contraintes, élévation) sur la rangée du haut, les textes
    (modèle et méthode ; retournement, jeux ; limites et références) sur trois colonnes dessous, le cartouche."""

    def __init__(self, calc: Calcul, doc_meta: Dict, k: float = 1.0, echelle: Optional[float] = None, suite: Optional["Suite"] = None):
        self.C, self.doc, self.k = calc, doc_meta, k
        self.suite = suite
        self.S = Encre()
        self.warnings: List[str] = []
        self.s = 6.3 * k
        self.echelle = echelle
        self.echelle_elevation = None
        self.note1 = Note(calc, doc_meta, k)

    def titre(self, d, fr, txt):
        fr.down(1)
        self.note1.titre(d, fr, txt)
        fr.down(2)

    def paragraphe(self, d, fr, txt, col=None, size=None, lead=1.32):
        self.note1.rp(d, fr, RI.auto(txt), size=size or self.s, col=col, lead=lead)
        fr.down(self.s * 0.45)

    def _vue(self, d, fr, cle: str, titre: str, texte: float = TEXTE_DESSIN) -> None:
        """Une vue des schémas dans sa cellule : le titre, le dessin à la place qui reste, la légende dessous. Une vue
        qu'on ne sait pas dessiner (proportions extrêmes) est remplacée par la phrase qui le dit : le calcul est rendu."""
        S, C = self.S, self.C
        self.titre(d, fr, titre)
        s_leg = 5.0 * self.k
        lignes = RI.envelopper(RI.auto(SC.legende(C.res, cle)), S.f_b, s_leg, self.note1.ms(S.mut), fr.w)[:3]
        h_leg = len(lignes) * s_leg * 1.25 + 3
        h = max(50.0, fr.room() - h_leg - 6)
        fr.down(h + 2)
        try:
            dv = C.vue(cle, fr.w, h, texte)
        except (ValueError, ArithmeticError):
            d.box(fr.x, fr.y, fr.w, h, stroke=S.mut, lw=0.6, r=4)
            d.t(fr.x + fr.w / 2, fr.y + h / 2 + 4, "Dessin impossible à cette échelle", S.f_v, 7.0, S.acc, "center")
            d.t(fr.x + fr.w / 2, fr.y + h / 2 - 6, "les dimensions saisies sont trop disproportionnées — le calcul reste valable", S.f_b, 5.6, S.mut, "center")
        else:
            ech = peindre_scene(d, dv, fr.x, fr.y, fr.w, h)
            if cle == "elevation":
                self.echelle_elevation = ech
        for l in lignes:
            fr.down(s_leg * 1.25)
            RI.dessiner(d, fr.x, fr.y, l, S.f_b, s_leg, S.mut, self.note1.ms(S.mut))

    def vue_moments(self, d, fr):
        self._vue(d, fr, "moments", "MOMENTS — une courbe par position de la charge, l'enveloppe en gras")

    def vue_contraintes(self, d, fr):
        self._vue(d, fr, "contraintes", "CONTRAINTES — σéq et taux le long des ailes, le maximum")

    def vue_elevation(self, d, fr):
        self._vue(d, fr, "elevation", "ÉLÉVATION — la tranche L, les ancrages, la charge diffusée")

    def _sections(self):
        res = self.C.res
        g = res["geometrie"]
        return MT.sections(res["mode"], res["configuration"], jeux=g["dx"] > 0 or g["dy"] > 0, largeur_efficace=g["a"] is not None)

    def _textes(self, d, fr, secs):
        for sec in secs:
            self.titre(d, fr, sec["titre"].upper())
            for p in sec["paragraphes"]:
                self.paragraphe(d, fr, p)

    def methode(self, d, fr):
        """Le modèle et le mode de calcul (``methode.py``, les premières sections)."""
        self._textes(d, fr, self._sections()[:MT.PARTAGE])

    def retournement(self, d, fr):
        """Ce qui change quand on retourne la cornière, les jeux et l'enveloppe, la largeur efficace (les suivantes)."""
        self._textes(d, fr, self._sections()[MT.PARTAGE:])

    def limites(self, d, fr):
        """Les avertissements du moteur, tels quels : rouge, attention, information."""
        S = self.S
        self.titre(d, fr, "LIMITES ET AVERTISSEMENTS")
        couleurs = {MO.ROUGE: S.ko, MO.ATTENTION: S.att, MO.INFO: S.mut}
        for a in self.C.res["avertissements"]:
            col = couleurs.get(a["niveau"], S.mut)
            prefixe = {MO.ROUGE: "▲ ", MO.ATTENTION: "▲ ", MO.INFO: "• "}.get(a["niveau"], "• ")
            self.paragraphe(d, fr, prefixe + francais(a["message"]), col=col if a["niveau"] != MO.INFO else S.ink, lead=1.28)

    def references(self, d, fr):
        self.titre(d, fr, "RÉFÉRENCES")
        for r in MT.REFERENCES:
            self.paragraphe(d, fr, "• " + r, col=self.S.mut, size=self.s * 0.95, lead=1.26)

    def cartouche(self, d):
        S, C, doc = self.S, self.C, self.doc
        res = C.res
        W = d.W
        x0, y0, w, h = M, M, W - 2 * M, CART_H
        d.box(x0, y0, w, h, stroke=S.ink, lw=0.9)
        ech = "—"
        if self.echelle:
            ech = "1:" + f"{2.8346 / self.echelle:.0f}"
        ident = " · ".join(str(x) for x in (doc.get("projet"), doc.get("partie")) if x)
        mat = res["materiau"]
        cases = [("CORNIÈRE", C.nom(), ident or "—", 0.20),
                 ("SECTION", res["section"]["designation"], f"aile horizontale {_mm(res['section']['larg'])} · verticale {_mm(res['section']['haut'])} · $L$ = {_mm(res['geometrie']['L'])} mm", 0.22),
                 ("CONFIGURATION · MODE", f"{res['configuration']} — {E.LIBELLES_CONFIGURATION[res['configuration']]}", E.LIBELLES_MODE[res["mode"]], 0.22),
                 ("ACIER", mat["nuance"], f"$f_y$ {_mm(mat['fy'])} N/mm² · $σ_{{adm}}$ = $f_y$/{f0(mat['coef'])} = {_fmt(mat['sigma_adm'])}", 0.14),
                 ("DATE · INDICE", str(doc.get("date", "")), "indice " + str(doc.get("indice", "")), 0.11),
                 ("ÉCHELLE DE LA COUPE", ech, "A4 paysage", 0.11)]
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
        S, C = self.S, self.C
        W, H = d.W, d.H
        y = H - M - 4
        d.t(M, y, self.doc.get("bureau", ""), S.f_v, 8.0, S.acc, track=1.4)
        d.t(W - M, y, "Note de calcul · " + self.doc.get("date", "") + " · indice " + str(self.doc.get("indice", "")), S.f_b, 8, S.mut, "right")
        y -= 16
        d.t(M, y, d.fit(f"DÉTAILS ET MÉTHODE — {C.nom()}", S.f_h, 13.5, W - 2 * M - 220), S.f_h, 13.5, S.ink)
        d.t(W - M, y, f"{TITRE} · configuration {C.res['configuration']} · {E.LIBELLES_MODE[C.res['mode']]}", S.f_b, 7.6, S.mut, "right")
        y -= 7
        d.line(M, y, W - M, y, S.ink, 1.0)
        top = y - 6
        bas = M + CART_H + 10
        h_vues = (top - bas - GAP) * PART_VUES
        wc = (W - 2 * M - 2 * GAP) / 3.0
        milieu = top - h_vues - GAP
        xs = [M + i * (wc + GAP) for i in range(3)]
        vues = [Frame(x, top, wc, top - h_vues) for x in xs]
        textes = [Frame(x, milieu, wc, bas) for x in xs]
        frames = vues + textes
        for x in xs[1:]:
            d.line(x - GAP / 2, top, x - GAP / 2, bas, S.rule, 0.45)
        d.line(M, milieu + GAP / 2, W - M, milieu + GAP / 2, S.rule, 0.45)
        for rang, f in enumerate(frames):
            _colonne(self, d, f, (2, rang))
        self.cartouche(d)
        for rang, f in enumerate(frames):
            if f.overflow() > 1.0:
                self.warnings.append(Debord(2, rang, f.x, f.overflow()))
        return self.warnings


# ------------------------------------------------------------------ page de suite
class Suite:
    """Page 3 · SUITE : les blocs du bas d'une colonne des pages 1 et 2 qui ne tiennent pas, même en taille réduite
    (le mécanisme de la note du garde-corps : ``reports`` = les blocs qui partent, ``((page, rang), nom)``, choisis par
    ``mise_en_page``). La page de suite a trois colonnes ; chaque colonne d'origine qui reporte en reçoit une (une
    quatrième s'empile dans la troisième), sous son titre « SUITE DE LA PAGE 1 · CALCUL ». Un bloc reporté est d'abord
    dessiné AU MOMENT où sa page l'aurait dessiné, sur un brouillon et dans sa colonne de suite (sa numérotation et sa
    hauteur sont celles qu'il aurait eues), puis redessiné en page 3 avec l'état de sa page d'alors : le même dessin."""

    def __init__(self, calc: Calcul, doc_meta: Dict, k: float, reports, W: float, H: float):
        self.C, self.doc, self.k = calc, doc_meta, k
        self.reports = set(reports)
        self.S = Encre()
        self.W, self.H = W, H
        self.top = H - M - 4 - 16 - 7 - 6           # l'en-tête de la page 2 (Methode.construire)
        self.bas = M + CART_H + 10
        self.wc = (W - 2 * M - 2 * GAP) / 3.0
        self.cadres: Dict = {}
        self.ordre: List = []
        self.titres: List = []
        self.blocs: List = []
        self.pages: set = set()
        self.brouillon = Doc(io.BytesIO(), (W, H))
        self.warnings: List[str] = []

    def prend(self, cle, nom) -> bool:
        return (cle, nom) in self.reports

    def cadre(self, cle) -> Frame:
        if cle not in self.cadres:
            if len(self.ordre) < 3:
                i = len(self.ordre)
                fr = Frame(M + i * (self.wc + GAP), self.top, self.wc, self.bas)
                self.ordre.append(fr)
            else:
                fr = self.ordre[-1]
            self.cadres[cle] = fr
        return self.cadres[cle]

    def reporter(self, d, page, cle, noms, fr_origine):
        S = self.S
        fr = self.cadre(cle)
        poses = []
        for nom in noms:
            y0, min0 = fr.y, fr.min_y
            titre = not poses
            if titre:
                fr.down(12)
            yb = fr.y
            etat = _etat_page(page)
            getattr(page, nom)(self.brouillon, fr)
            if fr.y == yb:
                fr.y, fr.min_y = y0, min0
                continue
            if titre:
                self.titres.append((fr.x, y0 - 8, f"SUITE DE LA PAGE {cle[0]} · {TITRES_PAGES[cle[0]].upper()}"))
                self.pages.add(cle[0])
            self.blocs.append((page, nom, etat, fr, yb))
            poses.append(nom)
        if poses:
            fr_origine.down(6.1 * self.k * 1.9)
            txt = "Suite p. 3 : " + ", ".join(NOMS_BLOCS[n] for n in poses) + "."
            d.t(fr_origine.x, fr_origine.y, d.fit(txt, S.f_b, 6.2 * self.k, fr_origine.w), S.f_b, 6.2 * self.k, S.acc)

    def utile(self) -> bool:
        return bool(self.blocs)

    def construire(self, d, echelle: Optional[float]):
        S, C = self.S, self.C
        W, H = d.W, d.H
        y = H - M - 4
        d.t(M, y, self.doc.get("bureau", ""), S.f_v, 8.0, S.acc, track=1.4)
        d.t(W - M, y, "Note de calcul · " + self.doc.get("date", "") + " · indice " + str(self.doc.get("indice", "")), S.f_b, 8, S.mut, "right")
        y -= 16
        titre = f"SUITE DE LA PAGE {min(self.pages)}" if len(self.pages) == 1 else "SUITE DES PAGES 1 ET 2"
        d.t(M, y, d.fit(f"{titre} — {C.nom()}", S.f_h, 13.5, W - 2 * M - 160), S.f_h, 13.5, S.ink)
        d.t(W - M, y, TITRE, S.f_b, 7.6, S.mut, "right")
        y -= 7
        d.line(M, y, W - M, y, S.ink, 1.0)
        for fr in self.ordre[1:]:
            d.line(fr.x - GAP / 2, self.top, fr.x - GAP / 2, self.bas, S.rule, 0.45)
        for x, yt, txt in self.titres:
            d.t(x, yt, txt, S.f_v, 5.6, S.mut, track=1.1)
        for page, nom, etat, fr, yb in self.blocs:
            maintenant = _etat_page(page)
            _remettre(etat)
            f = Frame(fr.x, fr.top, fr.w, fr.bottom)
            f.y = f.min_y = yb
            getattr(page, nom)(d, f)
            _remettre(maintenant)
        Methode(C, self.doc, self.k, echelle).cartouche(d)
        for rang, fr in enumerate(self.ordre):
            if fr.overflow() > 1.0:
                self.warnings.append(Debord(3, rang, fr.x, fr.overflow()))
        return self.warnings


# ------------------------------------------------------------------ la mise en page d'une cornière
def poser(d, calc: Calcul, doc_meta: Dict, k: float, reports=()) -> List[str]:
    """Les pages d'UNE cornière posées dans le document ``d`` (sans l'ouvrir ni l'enregistrer) : page 1, page 2, et la
    page de suite quand des blocs y partent. Rend les débordements."""
    suite = Suite(calc, doc_meta, k, reports, *A4L) if reports else None
    d.new_page(A4L)
    n1 = Note(calc, doc_meta, k, suite)
    w = list(n1.construire(d))
    d.new_page(A4L)
    n2 = Methode(calc, doc_meta, k, n1.echelle, suite)
    w += n2.construire(d)
    if suite is not None and suite.utile():
        d.new_page(A4L)
        w += suite.construire(d, n1.echelle)
    return w


def _essai(calc: Calcul, doc_meta: Dict, k: float, chemin: str, reports=()) -> Tuple[List[str], int]:
    """(débordements, nombre de pages) d'un essai de mise en page écrit dans ``chemin``."""
    d = Doc(chemin, A4L, title=doc_meta.get("titre", TITRE_NOTE))
    w = poser(d, calc, doc_meta, k, reports)
    n = d.save()
    return w, n


def mise_en_page(calc: Calcul, doc_meta: Dict, chemin: str) -> Tuple[float, str, Tuple, List[str], int]:
    """(k, nom de la taille, reports, débordements, pages) : la taille normale, puis la réduite ; si une colonne déborde
    encore en taille réduite, le DERNIER bloc de chaque colonne qui déborde passe en page de suite, et ainsi de suite
    jusqu'à ce que tout tienne (ou qu'il n'y ait plus rien à reporter : les débordements restent, et le disent)."""
    k, nom = VARIANTES[-1]
    warn: List[str] = []
    n = 2
    for k, nom in VARIANTES:
        warn, n = _essai(calc, doc_meta, k, chemin)
        if not warn:
            return k, nom, (), warn, n
    reports: List = []
    while warn:
        nouveaux = []
        for w in warn:
            cle = getattr(w, "colonne", None)
            if cle not in BLOCS:
                continue
            restent = [b for b in BLOCS[cle] if (cle, b) not in reports]
            if restent and (cle, restent[-1]) not in nouveaux:
                nouveaux.append((cle, restent[-1]))
        if not nouveaux:
            break
        reports += nouveaux
        warn, n = _essai(calc, doc_meta, k, chemin, tuple(reports))
    return k, nom, tuple(reports), warn, n


# ------------------------------------------------------------------ la note entière
def cornieres_de(etat_ou_cornieres) -> List[Dict]:
    """Les cornières d'une note : l'état du site ({"cornieres": [...]}) ou une liste ; ValueError s'il n'y en a aucune."""
    if isinstance(etat_ou_cornieres, dict):
        liste = etat_ou_cornieres.get("cornieres")
    else:
        liste = etat_ou_cornieres
    if not isinstance(liste, list) or not liste:
        raise ValueError("La note n'a aucune cornière : rien à imprimer.")
    return list(liste)


def _refus(nom: str, warn: List[str]) -> ValueError:
    return ValueError(f"La note de la cornière « {nom} » ne tient pas dans sa mise en page, même avec une page de suite ("
                      + " ; ".join(warn) + ") : elle ne sort pas tronquée. À signaler au bureau ; l'écran garde tout le calcul.")


def generer_note(etat_ou_cornieres, infos: Optional[Dict] = None, chemin: Optional[str] = None) -> bytes:
    """La note (bytes PDF) : la page de garde du site (sommaire : une ligne par cornière, son statut, sa page), puis les
    pages de chaque cornière (deux, trois avec une page de suite). ``infos`` : nom_projet, partie, date, indice. Une
    cornière hors domaine arrête la note avec le message du moteur (jamais un chiffre faux imprimé)."""
    global derniers_avertissements, derniere_variante, derniers_reports
    infos = infos or {}
    doc_meta = ndc_data.construire_doc(infos, date_defaut=datetime.today().strftime("%d/%m/%Y"))
    doc_meta["titre"] = TITRE_NOTE
    calcs = []
    for i, c in enumerate(cornieres_de(etat_ou_cornieres)):
        try:
            calcs.append(Calcul(c))
        except ValueError as err:
            nom = (c.get("nom") if isinstance(c, dict) else None) or f"n° {i + 1}"
            raise ValueError(f"La note n'est pas préparée : cornière « {nom} », {err}") from err
    fichiers = []
    try:
        fd, brouillon = tempfile.mkstemp(suffix=".pdf", prefix="note_corniere_brouillon_")
        os.close(fd)
        fichiers.append(brouillon)
        if chemin is None:
            fd, chemin = tempfile.mkstemp(suffix=".pdf", prefix="note_corniere_")
            os.close(fd)
            fichiers.append(chemin)
        dispositions = [mise_en_page(calc, doc_meta, brouillon) for calc in calcs]
        warn_tous: List[str] = []
        variantes, reports_tous = [], []
        for calc, (k, nom, reports, warn, n) in zip(calcs, dispositions):
            if warn:
                raise _refus(calc.nom(), warn)
            variantes.append(nom)
            reports_tous.append(reports)
        S = Encre()
        d = Doc(chemin, A4P, title=TITRE_NOTE)
        d.new_page(A4P)
        lignes = []
        for calc, (k, nom, reports, warn, n) in zip(calcs, dispositions):
            res = calc.res
            lignes.append(dict(titre=calc.nom(),
                               sous=f"{res['section']['designation']} · configuration {res['configuration']} ({E.LIBELLES_CONFIGURATION[res['configuration']]}) · "
                                    f"{E.LIBELLES_MODE[res['mode']]} · η max {_pct(res['taux_max'])}",
                               etat=STATUTS[res["statut"]][0], couleur=S.ok if res["statut"] == MO.OK else S.ko, pages=n))
        S.garde_lignes(d, doc_meta, lignes, legende=LEGENDE_GARDE)
        for calc, (k, nom, reports, warn, n) in zip(calcs, dispositions):
            warn_tous += poser(d, calc, doc_meta, k, reports)
        d.save()
        if warn_tous:
            raise _refus(calcs[0].nom(), warn_tous)
        derniers_avertissements = list(warn_tous)
        derniere_variante = variantes[-1] if len(variantes) == 1 else None
        dernieres_variantes = list(variantes)
        derniers_reports = tuple(reports_tous[0]) if len(reports_tous) == 1 else tuple(reports_tous)
        globals()["dernieres_variantes"] = dernieres_variantes
        with open(chemin, "rb") as fh:
            return fh.read()
    finally:
        for f in fichiers:
            try:
                os.remove(f)
            except OSError:
                pass


derniers_avertissements: List[str] = []
derniere_variante: Optional[str] = None          # la taille retenue quand la note n'a qu'une cornière
dernieres_variantes: List[str] = []              # la taille retenue pour chaque cornière de la dernière note
derniers_reports: Tuple = ()                     # les blocs passés en page de suite par la dernière note

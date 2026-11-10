# -*- coding: utf-8 -*-
"""Note de calcul « Cornière d'appui » — sur le kit ``ndc_pdf`` des notes du site (palette Encre, mêmes marges et mêmes
polices que les notes du garde-corps, des soudures et des assemblages), symboles COMPOSÉS (``ndc_pdf.riche``), peu de
prose. Décisions : ``docs/acier/CORNIERE_DECISIONS.md`` § 5 et § 9 (révision du 10/11/2026 au soir) ; ce qui est livré :
``docs/acier/CORNIERE_MOTEUR.md`` § 12.

UNE NOTE = la page de garde du site (A4 portrait : projet, partie, date, indice, sommaire — ``ndc_pdf.styles.Style.
garde_lignes``, la garde des notes Poutre, Dalle et Assemblages ; une ligne par cornière avec son état limite) puis,
PAR CORNIÈRE, deux pages A4 paysage :

Page 1 · CALCUL —
- bandeau : statut, η max, la section gouvernante et la combinaison de jeux qui la gouverne, la configuration, le mode
  et l'état limite ;
- colonne 1 : HYPOTHÈSES en lignes « libellé · valeur » COURTES (dix au plus : cornière, configuration et mode, acier,
  l'état limite avec sa limite σ et τ et son symbole — « σ_adm = f_y/1,5 = 236,7 N/mm² » à l'ELS, « f_y/γ_M0 = 355 N/mm² »
  à l'ELU —, le pivot ou u_V et y_c du mode, les jeux dx et dy, la tranche L et L_eff), sans les phrases du moteur ;
  la CHARGE ET L'ANCRAGE en lignes séparées « grandeur : valeur unité » (R et son état limite, L, X ± dx, a, R_ELU à
  l'ELS ; n × type Ø d, profondeur, rondelles, Y ± dy, entraxe s et e₁, support — voile, ou poutre avec ses étendues et
  les distances aux bords) ; la COUPE cotée peinte depuis les scènes partagées avec l'écran (``acier.corniere.schemas``,
  peintre de scène du kit), sa légende dit l'état limite ;
- colonnes 2 et 3 : 1 · GÉOMÉTRIE DES SECTIONS (W, A, largeur efficace, section nette), 2 · EFFORTS ET BRAS DE LEVIER
  (la chaîne de chaque section, avec la combinaison qui la gouverne et, en mode historique, les cellules du classeur),
  3 · VÉRIFICATION DES SECTIONS (le tableau σ, τ, σ_éq, η de S1 à S4, des maxima du congé et de la section nette, puis
  les formules de la section gouvernante ; la limite porte le symbole de l'état limite), 4 · ANCRAGES (N et V par
  ancrage : ELS puis ELU à l'ELS ; à l'ELU les seules N_ELU = H/n et V_ELU = R/n, rien d'inventé ; positions,
  informations) ; le RÉCAPITULATIF (barres de taux, combinaisons) ; la CONCLUSION (à l'ELU : « critère de von Mises,
  γ_M0 = 1,0 »). Chaque formule est écrite « symbole = expression = valeurs = résultat » (``formules.py`` : les gabarits
  des formules que le moteur nomme, ré-évalués par les tests).

Page 2 · DÉTAILS ET MÉTHODE — une page DOMINÉE PAR LES DESSINS : sous le titre, le RÉSUMÉ de méthode en trois lignes au
plus (``methode.resume`` : le modèle et l'état limite, le mode et la configuration, les jeux et l'enveloppe) ; puis
trois colonnes égales, chacune un titre court, la vue aussi grande que la colonne le permet (à l'échelle, le texte des
dessins à ``TEXTE_DESSIN`` pt par ``schemas.ajuster``) et une légende courte qui dit l'état limite — à gauche LARGEUR
EFFICACE ET ÉLÉVATION (scène ``elevation``), au centre MOMENTS SUR LA CORNIÈRE (``moments``), à droite CONTRAINTES DE
VON MISES (``contraintes``, le maillage coloré par le taux avec sa barre de couleur, le dégradé vert → orange → rouge
rendu tel quel par le peintre du kit) ; sous la rangée, les LIMITES (les avertissements du moteur, trois lignes au plus,
abrégés si besoin : ``abreger``), la ligne des RÉFÉRENCES, le cartouche. Cette page ne passe JAMAIS en page de suite :
ce qui se réduit quand la place manque, c'est la taille des légendes et des textes, pas le dessin.

DÉBORDEMENT — comme le garde-corps : la note est d'abord composée en taille normale, puis serrée si une colonne de la
page 1 déborde ; si elle déborde encore, les blocs du bas de la colonne passent, un à un, sur une PAGE DE SUITE
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
from ndc_pdf.fonts import string_width
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
_DECIMALE = re.compile(r"(?<=\d)\.(?=\d)")
_COUPURE = re.compile(r" \(| : ")
TEXTE_DESSIN = 5.6               # pt : la taille du texte des dessins à l'échelle (les quatre vues) dans la note
LIGNES_LIMITES = 3               # page 2 : les avertissements du moteur tiennent en trois lignes au plus (abrégés si besoin)
LIGNES_LEGENDE = 2               # page 2 : la légende d'une vue, deux lignes au plus (sa taille se réduit avant le dessin)
LIGNES_REFERENCES = 2            # page 2 : la ligne des références (deux si la police ne peut plus descendre)
TITRE = "Cornière d'appui"
TITRE_NOTE = "Note de calcul — Cornières d'appui"
STATUTS = {MO.OK: ("CONFORME", "ok"), MO.NOK: ("NON CONFORME", "ko")}
VERT_PALE = "#D9ECDF"            # le fond d'une pastille η sous le seuil d'attention (code couleur du bureau, 06/11/2026)
VARIANTES = ((1.0, "normale"), (0.92, "serrée"))
# la légende de la garde reste GÉNÉRALE : une note peut mêler des cornières à l'ELS et à l'ELU, chacune le dit sur sa page
LEGENDE_GARDE = "Vert : conforme · Rouge : non conforme — η = σéq/σlim ; l'état limite de chaque cornière (ELS : σadm = fy/coef ; ELU : fy/γM0) est dit sur sa page"
# les blocs de chaque colonne de la PAGE 1, (page, rang) → méthodes de la page, dans l'ordre de lecture : la composition
# ET l'ordre dans lequel les blocs du bas d'une colonne qui déborde passent en page de suite. La page 2 n'y est pas : elle
# ne se reporte jamais (ses textes se réduisent, ses dessins prennent la place qui reste)
BLOCS = {(1, 0): ("hypotheses", "charges", "coupe"),
         (1, 1): ("geometrie", "efforts", "sections"),
         (1, 2): ("ancrages", "recapitulatif")}
NOMS_BLOCS = {"hypotheses": "hypothèses", "charges": "charge et ancrage", "coupe": "coupe", "geometrie": "géométrie des sections",
              "efforts": "efforts et bras de levier", "sections": "vérification des sections", "ancrages": "ancrages",
              "recapitulatif": "récapitulatif"}
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
# les formules du bloc Ancrages du moteur, par famille : la chaîne qui mène à la traction H (bras, Mmax), les efforts ELS
# par ancrage, les efforts ELU déduits de l'ELS (R_ELU, ρ), les efforts ELU directs (R saisie à l'ELU) ; le reste est de
# l'information sans limite (pression diamétrale, arrachement, pression sur le support)
ANCRAGES_CHAINE = ("Mmax_j", "H_hist", "bras_A", "y_a", "bras_B", "H_eq")
ANCRAGES_ELS = ("N_ELS_hist", "N_ELS_eq", "V_ELS")
ANCRAGES_ELU = ("R_ELU_auto", "rapport_ELU", "ELU")
ANCRAGES_ELU_DIRECT = ("N_ELU_direct", "V_ELU_direct")


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
    if x is None:
        return "—"
    return F(x, 2).replace("-", "−", 1) + " kN"


def _knm(x) -> str:
    return F(x, 3).replace("-", "−", 1)


def _mm(x) -> str:
    return f0(x)


def francais(txt: str) -> str:
    """Un texte du moteur avec ses nombres à la virgule française : les messages du moteur écrivent certains nombres
    avec le point (« 52.50 kN ») — la note les FORMATE, elle ne les recalcule pas."""
    return _DECIMALE.sub(",", str(txt))


def abreger(message: str) -> str:
    """Un avertissement du moteur ABRÉGÉ à sa proposition principale : ce qui précède la première parenthèse ou le
    premier « : » (« Chevilles non vérifiées : efforts N et V… » → « Chevilles non vérifiées »), quand il en reste
    assez pour être compris ; le point final tombe. La page 2 ne s'en sert que si les trois lignes des limites ne
    suffisent pas aux messages entiers."""
    m = _COUPURE.search(message)
    if m and m.start() >= 20:
        return message[:m.start()].rstrip(" .")
    return message.rstrip(".")


def legende_coupe(res: Dict) -> str:
    """La légende du bas de la page 1 : les conventions de la coupe et l'ÉTAT LIMITE de la cornière (une note peut mêler
    des cornières à l'ELS et à l'ELU : la légende est celle de la page)."""
    mat = res["materiau"]
    etat = mat["etat_limite"]
    if etat == E.ELS:
        lim = f"η = σéq/σadm avec σadm = fy/{f0(mat['coef'])} (contraintes admissibles, efforts ELS)"
    else:
        lim = f"η = σéq/(fy/γM0), γM0 = {F(mat['gamma_M0'], 1)} (critère de von Mises, réaction saisie à l'ELU)"
    return (f"Coupe à l'échelle, cotes en mm · sections 1 à 4 en orange · R la réaction {etat} (positions X ± dx en tirets) · H la traction "
            f"totale des ancrages · C la compression du mur · {lim} · chaque grandeur garde la combinaison de jeux la plus défavorable (enveloppe).")


# ------------------------------------------------------------------ texte riche coloré par morceaux (les limites de la page 2)
def _flux(items, font, size, ms_de, width):
    """Les lignes d'un texte riche COLORÉ par morceaux : ``items`` = [(texte riche, couleur)], posés à la suite et
    repliés sur ``width`` ; chaque ligne est une liste de (jeton, couleur). ``ms_de(couleur)`` : le style des symboles."""
    sp = string_width(" ", font, size)
    lignes, cur, wcur = [], [], 0.0
    for txt, col in items:
        js = RI.jetons(txt)
        if js:
            js[-1][2] = True                      # un espace après chaque morceau
        for j in js:
            w = RI.largeur_ligne([j], font, size, ms_de(col))
            if cur and wcur + w > width + 0.01:
                lignes.append(cur)
                cur, wcur = [], 0.0
            cur.append((j, col))
            wcur += w + (sp if j[2] else 0.0)
    if cur:
        lignes.append(cur)
    return lignes


def _tronquer(lignes, n, font, size, ms_de, width):
    """Les ``n`` premières lignes d'un flux, la dernière terminée par « … » quand il en restait (jamais vu sur les états
    d'essai : les trois lignes des limites suffisent aux messages abrégés)."""
    if len(lignes) <= n:
        return lignes
    out = lignes[:n]
    der = list(out[-1])
    col = der[-1][1]
    pts = ([False, "…", False], col)
    while der and RI.largeur_ligne([j for j, _ in der + [pts]], font, size, ms_de(col)) > width:
        der.pop()
    out[-1] = der + [pts]
    return out


def _dessiner_flux(d, x, y, ligne, font, size, ms_de):
    """Une ligne d'un flux : les jetons consécutifs de même couleur dessinés ensemble (``riche.dessiner``)."""
    sp = string_width(" ", font, size)
    i = 0
    while i < len(ligne):
        col = ligne[i][1]
        j = i
        while j < len(ligne) and ligne[j][1] == col:
            j += 1
        sous = [jt for jt, _ in ligne[i:j]]
        x += RI.dessiner(d, x, y, sous, font, size, col, ms_de(col))
        if sous[-1][2] and j < len(ligne):
            x += sp
        i = j


# ------------------------------------------------------------------ le calcul d'une cornière
class Calcul:
    """Tout ce que la note imprime d'une cornière, calculé une fois : le résultat du moteur, les dessins (à la demande,
    par boîte), les lignes des hypothèses et de la charge, les légendes."""

    def __init__(self, corniere: Dict):
        self.c = MO.normaliser(corniere)
        self.res = MO.verifier(self.c)
        self._vues: Dict = {}

    def nom(self) -> str:
        return str(self.c.get("nom") or "Cornière")

    def els(self) -> bool:
        return self.res["materiau"]["etat_limite"] == E.ELS

    def vue(self, cle: str, w: float, h: float, texte: float = TEXTE_DESSIN):
        """Le dessin ``cle`` pour une boîte (w, h) en points : les quatre vues sont à l'échelle, chacune est reconstruite
        (``schemas.ajuster``) jusqu'à ce que son texte se lise à ``texte`` pt dans la boîte. ValueError quand la vue ne
        se dessine pas (``schemas``)."""
        key = (cle, round(w), round(h), texte)
        if key not in self._vues:
            fn = SC.FONCTIONS[cle]
            d0 = fn(self.res, None, False)
            self._vues[key] = SC.ajuster(lambda fs: fn(self.res, fs, False), (w, h), texte, d0.fs)
        return self._vues[key]

    def hypotheses(self) -> List[Tuple[str, str]]:
        """Les lignes « libellé · valeur » du bloc HYPOTHÈSES (valeur en texte riche), dix au plus, COURTES — § 9.6 : la
        cornière, la configuration et le mode, l'acier, l'état limite avec ses limites et son symbole, le pivot ou u_V
        et y_c du mode, les jeux, la tranche. Les phrases du moteur (``res["hypotheses"]``) ne sont pas recopiées."""
        res, c = self.res, self.c
        sec, g, mat, cfg, mode = res["section"], res["geometrie"], res["materiau"], res["configuration"], res["mode"]
        L: List[Tuple[str, str]] = []
        if sec["source"] == "catalogue":
            piece = f"{sec['designation']} du catalogue ({sec['norme']}), {_fmt(sec['masse'], 1)} kg/m"
        else:
            piece = f"{sec['designation']} mm, saisie libre ({_fmt(sec['masse'], 1)} kg/m)"
        L.append(("Cornière", f"{piece} · aile horizontale {_mm(sec['larg'])} (aile {sec['aile_horizontale']} du nom), verticale {_mm(sec['haut'])} · "
                              f"$e$ = {_mm(g['e'])} · $r$ = {_mm(g['r'])} mm"))
        L.append(("Configuration", f"{cfg} — {E.LIBELLES_CONFIGURATION[cfg]} · {E.LIBELLES_MODE[mode]}"))
        L.append(("Acier", f"{mat['nuance']} · $f_y$ = {_mm(mat['fy'])} N/mm²"))
        if mat["etat_limite"] == E.ELS:
            L.append(("État limite", f"ELS · $σ_{{adm}}$ = $f_y$/{f0(mat['coef'])} = {_fmt(mat['sigma_adm'])} N/mm² · $τ_{{adm}}$ = $σ_{{adm}}$/√3 = "
                                     f"{_fmt(mat['tau_adm'])} N/mm² · $σ_{{eq}}$ = √(σ² + 3τ²) ≤ $σ_{{adm}}$ (efforts ELS)"))
        else:
            L.append(("État limite", f"ELU · $f_y$/$γ_{{M0}}$ = {_fmt(mat['sigma_adm'])} N/mm² · $f_y$/(√3·$γ_{{M0}}$) = {_fmt(mat['tau_adm'])} N/mm² · "
                                     f"$γ_{{M0}}$ = {F(mat['gamma_M0'], 1)} · $σ_{{eq}}$ = √(σ² + 3τ²) ≤ $f_y$/$γ_{{M0}}$ (von Mises, EN 1993-1-1 § 6.2.1(5))"))
        if mode == E.HISTORIQUE:
            L.append(("Mode", f"pivot $e$/2 : $YH$ = $Y$ − $e$/2 = {_mm(c['Y'] - g['e'] / 2)} mm · $τ$ = $R$/($L$·$t$) dans les deux ailes · "
                              f"bras de la traction $Y$ − jeu = {_mm(c['Y'] - c['jeu_ancrage'])} mm (C76 = {_mm(c['jeu_ancrage'])})"))
        elif cfg == "A":
            L.append(("Mode", f"équilibre, frottement nul au mur · pivot $y_c$ = {_mm(g['yc'])} mm du dessous · $u_V$ = {_mm(g['uV'])} mm du nu du mur · "
                              "$N$ = +$R$ dans l'aile verticale"))
        else:
            L.append(("Mode", f"équilibre (imposé en B) · contact au bout libre $c$ = {_mm(g['yc'])} mm · $u_V$ = {_mm(g['uV'])} mm · "
                              "$N$ = −$R$ dans l'aile verticale · section nette au trou"))
        if c["dx"] > 0 or c["dy"] > 0:
            L.append(("Jeux", f"$dx$ = {_mm(c['dx'])} mm, $dy$ = {_mm(c['dy'])} mm · {len(res['combinaisons'])} combinaisons, chaque grandeur garde la "
                              "plus défavorable (enveloppe)"))
        else:
            L.append(("Jeux", "aucun ($dx$ = $dy$ = 0) : une seule combinaison, la charge à $X$ et l'ancrage à $Y$"))
        tranche = f"$L$ = {_mm(g['L'])} mm, largeur de toutes les sections · {g['n']} ancrage{'s' if g['n'] > 1 else ''}"
        if g["a"] is not None:
            tranche += f" · $a$ = {_mm(g['a'])} mm : $L_{{eff}}$ = min($L$ ; $a$ + 2·$Xe$) = {_fmt(g['L_eff'])} mm (σ × {_fmt(g['facteur_L_eff'], 3)})"
        L.append(("Tranche", tranche))
        if res["mode_demande"] != mode:
            L.append(("Note", "mode historique demandé : sans effet en configuration B"))
        return L

    def charge_et_ancrage(self) -> List[Tuple[str, str]]:
        """Les lignes « grandeur : valeur unité » du bloc CHARGE ET ANCRAGE (§ 9.6) : R et son état limite, L, X ± dx, a,
        R_ELU (à l'ELS seulement : auto ou saisie) ; puis n × type Ø d, la profondeur, les rondelles, Y ± dy, l'entraxe
        et e₁ (auto ou saisi), le support."""
        res, c = self.res, self.c
        g, ef, anc, mat = res["geometrie"], res["efforts"], res["ancrages"], res["materiau"]
        etat = mat["etat_limite"]
        jx = f" ± {_mm(c['dx'])}" if c["dx"] > 0 else ""
        jy = f" ± {_mm(c['dy'])}" if c["dy"] > 0 else ""
        L = [("Charge", f"$R$ = {_kn(ef['R'])} à l'{etat}" + (" (réaction saisie à l'ELU)" if etat == E.ELU else " (réaction de service)")),
             ("Longueur L", f"$L$ = {_mm(g['L'])} mm (longueur reprise, la tranche)"),
             ("Position X", f"$X$ = {_mm(c['X'])}{jx} mm du nu du mur, sur l'aile horizontale"),
             ("Appui a", f"$a$ = {_mm(g['a'])} mm (largeur d'appui de la charge)" if g["a"] is not None else "$a$ = $L$ (la charge sur toute la tranche)")]
        if etat == E.ELS:
            L.append(("Réaction ELU", f"$R_{{ELU}}$ = 1,5·$R$ = {_kn(ef['R_ELU'])} (auto : à remplacer par la combinaison réelle)" if ef["R_ELU_auto"]
                      else f"$R_{{ELU}}$ = {_kn(ef['R_ELU'])} (saisie)"))
        L.append(("Ancrage", f"{anc['n']} × {E.LIBELLES_TYPE_ANCRAGE[g['type_ancrage']]} Ø{_mm(anc['d'])} (trou $d_0$ = {_mm(anc['d0'])} mm)"))
        if g["type_ancrage"] == "traversant":
            L.append(("Profondeur", f"épaisseur traversée {_mm(g['profondeur'])} mm (écrou et rondelle sur la face arrière)"))
        else:
            L.append(("Profondeur", f"$h_{{ef}}$ = {_mm(g['profondeur'])} mm dans le support"))
        rd = g["rondelle"]
        n_, l_ = rd["normale"], rd["large"]
        if g["rondelles"] == 0:
            rondelles = E.LIBELLES_RONDELLES[0]
        elif g["rondelles"] == 1:
            rondelles = f"{E.LIBELLES_RONDELLES[1]} (Ø{_mm(n_[0])} × {_mm(n_[1])})"
        else:
            rondelles = (f"rondelle large ISO 7093-1 contre l'aile (Ø{_mm(l_[0])} × {_mm(l_[1])}) et rondelle ISO 7089 sous l'écrou "
                         f"(Ø{_mm(n_[0])} × {_mm(n_[1])})")
        L.append(("Rondelles", rondelles))
        L.append(("Position Y", f"$Y$ = {_mm(c['Y'])}{jy} mm de la face extérieure de l'aile horizontale"))
        L.append(("Entraxe", f"$s$ = {_mm(anc['entraxe'])} mm ({'auto : L/n' if g['entraxe_auto'] else 'saisi'}) · $e_1$ = {_mm(anc['distance_bout'])} mm au bout"))
        if g["support"] == E.POUTRE:
            b = g["bords_poutre"]
            support = (f"{E.LIBELLES_SUPPORT[g['support']]} : {_mm(g['support_haut'])} mm au-dessus et {_mm(g['support_bas'])} mm au-dessous de la face "
                       f"extérieure de l'aile horizontale · axe d'ancrage à {_mm(b['haut'])} mm du bord supérieur, {_mm(b['bas'])} mm du bord inférieur")
        else:
            support = f"{E.LIBELLES_SUPPORT[g['support']]} (sans limite)"
        L.append(("Support", support))
        return L

    def symboles_limites(self) -> Dict[str, str]:
        """Les symboles des limites dans les formules imprimées : le moteur nomme les termes « σ_adm » et « τ_adm » (le
        NOM des limites, § 9.5) ; à l'ELU la note écrit le symbole de l'état limite (``materiau.symbole_sigma``,
        ``symbole_tau``), entre parenthèses pour rester lisible dans un quotient."""
        mat = self.res["materiau"]
        if self.els():
            return {}
        return {"σ_adm": f"({mat['symbole_sigma']})", "τ_adm": f"({mat['symbole_tau']})"}

    def legende_vue(self, cle: str) -> str:
        """La légende COURTE d'une vue de la page 2 (texte riche) : l'état limite, ce que montre la vue, ses conventions —
        composée depuis le résultat du moteur, comme ``schemas.legende`` dont elle est la forme brève."""
        res = self.res
        g, mat, anc, cfg, mode = res["geometrie"], res["materiau"], res["ancrages"], res["configuration"], res["mode"]
        etat = mat["etat_limite"]
        if cle == "elevation":
            a = f"$a$ = {_mm(g['a'])}" if g["a"] is not None else "$a$ = $L$"
            coupe = " · tranche coupée au dessin, cotes vraies" if g["L"] > SC.COUPURE_TRANCHE * g["haut"] else ""
            return (f"{etat} : $N$ et $V$ par ancrage (enveloppe) · la tranche $L$ = {_mm(g['L'])} vue de face, {g['n']} ancrage{'s' if g['n'] > 1 else ''} "
                    f"à l'entraxe $s$ = {_mm(anc['entraxe'])}, $e_1$ = {_mm(anc['distance_bout'])} au bout · l'aile horizontale rabattue : la charge "
                    f"({a}) et sa diffusion à 45° jusqu'à la section 1, $L_{{eff}}$ = {_mm(g['L_eff'])}{coupe}")
        if cle == "moments":
            signe = "$M$ > 0 : face du congé tendue" if cfg == "A" else "$M$ < 0 (face du congé comprimée), constant de l'ancrage à l'angle"
            return (f"configuration {cfg} · {E.LIBELLES_MODE[mode]} · {etat} : $R$ = {_kn(res['efforts']['R'])} · le diagramme de chaque aile porté hors "
                    f"de l'angle depuis son axe ($M$ = 0) : une courbe par position de la charge (extrêmes en trait fin), l'enveloppe en gras et "
                    f"remplie, $M_1$ à $M_4$ et $M_{{max}}$ · {signe}")
        sym = "$σ_{adm}$" if etat == E.ELS else "$f_y$/$γ_{M0}$"
        return (f"{etat} · taux $σ_{{eq}}$ / {sym} = {_fmt(mat['sigma_adm'])} N/mm² par maille, enveloppe des positions · ◉ le maximum de chaque aile · "
                f"vert sous 100 %, orange de 100 à 105 %, rouge au-delà · contraintes de poutre σ = M·ζ/W + N/A, τ = V/A — pas un calcul par éléments finis")


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

    def symbole_sigma(self) -> str:
        """Le symbole composé de la limite de l'état limite : σ_adm à l'ELS, f_y/γ_M0 à l'ELU (``materiau.symbole_sigma``)."""
        return "$σ_{adm}$" if self.C.els() else "$f_y$/$γ_{M0}$"

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
        t3 = f"Configuration {res['configuration']} · {E.LIBELLES_MODE[res['mode']]} · {res['materiau']['etat_limite']}"
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
        self.titre(d, fr, "CHARGE ET ANCRAGE")
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
            noms = " = ".join(f"W{'₁₂₃₄'[int(c[1]) - 1]}" for c in cles)
            subs.append(FO.Sub(noms, W.template, W.vals, W.res, W.unit, W.nd))
            if A is not None:
                subs.append(FO.Sub(" = ".join(f"A{'₁₂₃₄'[int(c[1]) - 1]}" for c in cles), A.template, A.vals, A.res, A.unit, A.nd))
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
        sym = self.symbole_sigma()
        quot = sym if C.els() else f"({sym})"           # le quotient reste lisible : η = σ_eq/(f_y/γ_M0)
        self.titre(d, fr, f"{self.numero()}VÉRIFICATION DES SECTIONS", res["taux_max"])
        fr.down(0.5)
        self.rp(d, fr, f"$σ$ = |M|/W + |N|/A · $τ$ = {'R/(L·t)' if hist else '|V|/A'} · $σ_{{eq}}$ = √(σ² + 3τ²) · η = $σ_{{eq}}$/{quot} "
                       f"({sym} = {_fmt(res['materiau']['sigma_adm'])} N/mm², {res['materiau']['etat_limite']}) · contraintes en N/mm², $M$ en kN·m, $s$ et $t$ en mm",
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
        self.formules(d, fr, [("Gouvernante : " + self._lib(gov, vg, bg, FO.COUPE), FO.renommer(FO.coupe(bg), C.symboles_limites()))])
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
        if C.els():
            # à l'ELS : la chaîne et les efforts ELS par ancrage, puis les efforts ELU déduits (R_ELU auto ou saisie)
            els = [s for f, s in paires if f["cle"] in ANCRAGES_CHAINE + ANCRAGES_ELS]
            elu = [s for f, s in paires if f["cle"] in ANCRAGES_ELU]
            info = [s for f, s in paires if f["cle"] not in ANCRAGES_CHAINE + ANCRAGES_ELS + ANCRAGES_ELU]
            lib_elu = "ELU — R_ELU = 1,5·R (auto)" if anc["R_ELU_auto"] else "ELU — R_ELU saisie"
            groupes = [("ELS, par ancrage", els), (lib_elu, elu), ("Information, sans limite", info)]
        else:
            # à l'ELU : la chaîne et les seules N_ELU = H/n, V_ELU = R/n — rien n'est inventé pour l'ELS
            elu = [s for f, s in paires if f["cle"] in ANCRAGES_CHAINE + ANCRAGES_ELU_DIRECT]
            info = [s for f, s in paires if f["cle"] not in ANCRAGES_CHAINE + ANCRAGES_ELU_DIRECT]
            groupes = [("ELU, par ancrage — R saisie à l'ELU, pas d'efforts ELS", elu), ("Information, sans limite", info)]
        self.formules(d, fr, groupes)
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
        g, anc, sec, mat = res["gouvernant"], res["ancrages"], res["section"], res["materiau"]
        if C.els():
            methode = f"méthode aux contraintes admissibles (efforts ELS, coefficient {f0(mat['coef'])} sur $f_y$)"
        else:
            methode = f"à l'ELU, critère de von Mises (EN 1993-1-1 § 6.2.1(5)), $γ_{{M0}}$ = {F(mat['gamma_M0'], 1)}"
        txt = (f"Taux maximal {_pct(res['taux_max'])} — {g['nom'].lower()}, {g['combinaison']['libelle']} — {methode}, "
               f"configuration {res['configuration']}, {E.LIBELLES_MODE[res['mode']]}. ")
        if res["statut"] == MO.OK:
            txt += (f"La cornière {sec['designation']} porte la réaction $R$ = {_kn(res['efforts']['R'])} ({mat['etat_limite']}) sur {_mm(res['geometrie']['L'])} mm. ")
        else:
            txt += (f"La cornière {sec['designation']} ne porte pas la réaction $R$ = {_kn(res['efforts']['R'])} ({mat['etat_limite']}) en l'état : section, "
                    "position de la charge ou de l'ancrage à revoir (le sélecteur de l'écran donne la plus légère qui tient). ")
        if C.els():
            txt += (f"Efforts à donner au calcul des chevilles, par ancrage : $N_{{ELS}}$ = {_kn(anc['N_ELS'])}, $V_{{ELS}}$ = {_kn(anc['V_ELS'])} ; "
                    f"$N_{{ELU}}$ = {_kn(anc['N_ELU'])}, $V_{{ELU}}$ = {_kn(anc['V_ELU'])} (chevilles non vérifiées ici).")
        else:
            txt += (f"Efforts à donner au calcul des chevilles, par ancrage, à l'ELU : $N_{{ELU}}$ = {_kn(anc['N_ELU'])}, $V_{{ELU}}$ = {_kn(anc['V_ELU'])} "
                    "(R saisie à l'ELU : pas d'efforts ELS ; chevilles non vérifiées ici).")
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
        # la légende de la page (les conventions de la coupe, l'état limite) : une ligne, deux si elle ne tient pas
        lignes = RI.envelopper(legende_coupe(self.C.res), S.f_b, 5.4, self.ms(S.mut), W - 2 * M)[:2]
        for i, l in enumerate(lignes):
            RI.dessiner(d, M, 20 + (len(lignes) - 1 - i) * 7, l, S.f_b, 5.4, S.mut, self.ms(S.mut))
        for rang, f in enumerate((f1, f2, f3)):
            if f.overflow() > 1.0:
                self.warnings.append(Debord(1, rang, f.x, f.overflow()))
        return self.warnings


# ------------------------------------------------------------------ page 2 · détails et méthode
class Methode:
    """Page 2 · DÉTAILS ET MÉTHODE, dominée par les dessins : le résumé de méthode (trois lignes au plus) sous le titre ;
    trois colonnes égales — élévation, moments, contraintes —, chacune un titre court, la vue aussi grande que la colonne
    le permet et une légende courte ; sous la rangée, les limites (trois lignes au plus), la ligne des références, le
    cartouche. Rien de cette page ne part en page de suite : les textes se réduisent, le dessin garde sa place."""

    COLONNES = (("elevation", "LARGEUR EFFICACE ET ÉLÉVATION"), ("moments", "MOMENTS SUR LA CORNIÈRE"), ("contraintes", "CONTRAINTES DE VON MISES"))

    def __init__(self, calc: Calcul, doc_meta: Dict, k: float = 1.0, echelle: Optional[float] = None, suite: Optional["Suite"] = None):
        self.C, self.doc, self.k = calc, doc_meta, k
        self.suite = suite
        self.S = Encre()
        self.warnings: List[str] = []
        self.s = 6.0 * k                 # le résumé de méthode
        self.s_lim = 5.6 * k             # les limites
        self.s_ref = 5.2 * k             # les références
        self.s_leg = 5.0 * k             # la légende d'une vue
        self.echelle = echelle
        self.echelle_elevation = None
        self.note1 = Note(calc, doc_meta, k)

    # --- le résumé de méthode, sous le titre
    def lignes_resume(self, width: float):
        """(taille, [lignes de jetons]) du résumé : trois lignes au plus — la taille descend (jusqu'à 5,2 pt) avant
        qu'une quatrième ligne ne soit admise ; rien n'est tronqué."""
        S = self.S
        ms = self.note1.ms(S.ink)
        textes = [RI.auto(t) for t in MT.resume(self.C.res)]
        size = self.s
        while True:
            lignes = [l for t in textes for l in RI.envelopper(t, S.f_b, size, ms, width)]
            if len(lignes) <= MT.LIGNES_RESUME or size <= 5.2 * self.k + 1e-9:
                return size, lignes
            size -= 0.2

    def resume(self, d, fr):
        S = self.S
        size, lignes = self.lignes_resume(fr.w)
        for l in lignes:
            fr.down(size * 1.3)
            RI.dessiner(d, fr.x, fr.y, l, S.f_b, size, S.ink, self.note1.ms(S.ink))

    # --- les textes du bas : limites et références, mesurés avant d'être posés
    def lignes_limites(self, width: float):
        """(taille, lignes du flux coloré) des avertissements du moteur : entiers quand ils tiennent en trois lignes ;
        sinon ceux d'information abrégés, puis ceux d'attention, puis la taille réduite, puis — jamais vu — tronqués.
        Un avertissement rouge n'est jamais abrégé."""
        S = self.S
        couleurs = {MO.ROUGE: S.ko, MO.ATTENTION: S.att, MO.INFO: S.ink}
        prefixes = {MO.ROUGE: "▲ ", MO.ATTENTION: "▲ ", MO.INFO: "• "}
        avs = self.C.res["avertissements"]

        def items(abr):
            return [(prefixes.get(a["niveau"], "• ") + RI.auto(francais(abreger(a["message"]) if a["niveau"] in abr else a["message"])),
                     couleurs.get(a["niveau"], S.ink)) for a in avs]
        essais = [(self.s_lim, ()), (self.s_lim, (MO.INFO,)), (self.s_lim, (MO.INFO, MO.ATTENTION)), (5.0 * self.k, (MO.INFO, MO.ATTENTION))]
        size, lignes = self.s_lim, []
        for size, abr in essais:
            lignes = _flux(items(abr), S.f_b, size, self.note1.ms, width)
            if len(lignes) <= LIGNES_LIMITES:
                return size, lignes
        return size, _tronquer(lignes, LIGNES_LIMITES, S.f_b, size, self.note1.ms, width)

    def lignes_references(self, width: float):
        """(taille, lignes) de la ligne des références (deux lignes au plus, la taille descend d'abord)."""
        S = self.S
        txt = RI.auto(" · ".join(MT.REFERENCES))
        size = self.s_ref
        while True:
            lignes = _flux([(txt, S.mut)], S.f_b, size, self.note1.ms, width)
            if len(lignes) <= LIGNES_REFERENCES or size <= 4.6 * self.k + 1e-9:
                return size, _tronquer(lignes, LIGNES_REFERENCES, S.f_b, size, self.note1.ms, width)
            size -= 0.2

    def hauteur_textes(self, width: float) -> float:
        s1, l1 = self.lignes_limites(width)
        s2, l2 = self.lignes_references(width)
        return 6 + len(l1) * s1 * 1.28 + 5 + len(l2) * s2 * 1.28 + 2

    def textes(self, d, fr):
        """Les deux rangées du bas : « LIMITES » puis « RÉFÉRENCES », le libellé en petites capitales à gauche."""
        S = self.S
        lab_w = LAB_W * self.k
        width = fr.w - lab_w
        fr.down(6)
        for lab, (size, lignes) in (("LIMITES", self.lignes_limites(width)), ("RÉFÉRENCES", self.lignes_references(width))):
            for i, l in enumerate(lignes):
                fr.down(size * 1.28)
                if i == 0:
                    d.t(fr.x, fr.y + 0.2, lab, S.f_v, 4.9 * self.k, S.mut, track=0.9)
                _dessiner_flux(d, fr.x + lab_w, fr.y, l, S.f_b, size, self.note1.ms)
            fr.down(5)

    # --- une colonne : le titre, la vue, la légende
    H_TITRE = 10 + 3.2 + 3            # ce que ``Note.titre`` et le blanc qui le suit prennent dans la colonne

    def preparer(self, cle: str, w: float, h_dispo: float) -> Dict:
        """Ce qu'une colonne va poser, mesuré avant : la légende (deux lignes au plus — sa taille descend jusqu'à 4,2 pt
        avant qu'une troisième soit admise), le dessin reconstruit pour la place (``Calcul.vue``) et la hauteur qu'il
        réclame à la largeur de la colonne (la place qui reste au plus), la hauteur totale de la colonne. Une vue qu'on ne
        sait pas dessiner (proportions extrêmes) donne ``dessin`` None : le cadre qui le dit sera posé à sa place."""
        S, C = self.S, self.C
        ms = self.note1.ms(S.mut)
        txt = C.legende_vue(cle)
        s_leg = self.s_leg
        lignes = RI.envelopper(txt, S.f_b, s_leg, ms, w)
        while len(lignes) > LIGNES_LEGENDE and s_leg > 4.2 * self.k + 1e-9:
            s_leg -= 0.2
            lignes = RI.envelopper(txt, S.f_b, s_leg, ms, w)
        lignes = lignes[:LIGNES_LEGENDE + 1]
        h_leg = len(lignes) * s_leg * 1.25 + 3
        h_max = max(40.0, h_dispo - self.H_TITRE - h_leg - 4)
        try:
            dv = C.vue(cle, w, h_max)
        except (ValueError, ArithmeticError):
            dv, h = None, min(h_max, 70.0)
        else:
            _, _, vw, vh = dv.viewbox
            h = min(h_max, (w - 6.0) * vh / vw + 6.0)   # la hauteur que le dessin réclame à cette largeur (marge du peintre : 3 pt)
        return {"dessin": dv, "h": h, "s_leg": s_leg, "lignes": lignes, "h_col": self.H_TITRE + h + 2 + h_leg}

    def vue(self, d, fr, cle: str, titre: str, prep: Dict) -> None:
        """Une vue dans sa colonne : le titre, le dessin aussi grand que la colonne le permet, la légende dessous."""
        S = self.S
        self.note1.titre(d, fr, titre)
        fr.down(3)
        dv, h = prep["dessin"], prep["h"]
        fr.down(h)
        if dv is None:
            d.box(fr.x, fr.y, fr.w, h, stroke=S.mut, lw=0.6, r=4)
            d.t(fr.x + fr.w / 2, fr.y + h / 2 + 4, "Dessin impossible à cette échelle", S.f_v, 7.0, S.acc, "center")
            d.t(fr.x + fr.w / 2, fr.y + h / 2 - 6, "les dimensions saisies sont trop disproportionnées — le calcul reste valable", S.f_b, 5.6, S.mut, "center")
        else:
            ech = peindre_scene(d, dv, fr.x, fr.y, fr.w, h)
            if cle == "elevation":
                self.echelle_elevation = ech
        fr.down(2)
        for l in prep["lignes"]:
            fr.down(prep["s_leg"] * 1.25)
            RI.dessiner(d, fr.x, fr.y, l, S.f_b, prep["s_leg"], S.mut, self.note1.ms(S.mut))

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
        if C.els():
            acier = f"$f_y$ {_mm(mat['fy'])} · ELS : $σ_{{adm}}$ = $f_y$/{f0(mat['coef'])} = {_fmt(mat['sigma_adm'])} N/mm²"
        else:
            acier = f"$f_y$ {_mm(mat['fy'])} · ELU : $f_y$/$γ_{{M0}}$ = {_fmt(mat['sigma_adm'])} N/mm², $γ_{{M0}}$ = {F(mat['gamma_M0'], 1)}"
        cases = [("CORNIÈRE", C.nom(), ident or "—", 0.20),
                 ("SECTION", res["section"]["designation"], f"aile horizontale {_mm(res['section']['larg'])} · verticale {_mm(res['section']['haut'])} · $L$ = {_mm(res['geometrie']['L'])} mm", 0.21),
                 ("CONFIGURATION · MODE", f"{res['configuration']} — {E.LIBELLES_CONFIGURATION[res['configuration']]}", E.LIBELLES_MODE[res["mode"]], 0.20),
                 ("ACIER · ÉTAT LIMITE", mat["nuance"], acier, 0.19),
                 ("DATE · INDICE", str(doc.get("date", "")), "indice " + str(doc.get("indice", "")), 0.10),
                 ("ÉCHELLE", ech, "coupe · A4 paysage", 0.10)]
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
        res = C.res
        W, H = d.W, d.H
        y = H - M - 4
        d.t(M, y, self.doc.get("bureau", ""), S.f_v, 8.0, S.acc, track=1.4)
        d.t(W - M, y, "Note de calcul · " + self.doc.get("date", "") + " · indice " + str(self.doc.get("indice", "")), S.f_b, 8, S.mut, "right")
        y -= 16
        d.t(M, y, d.fit(f"DÉTAILS ET MÉTHODE — {C.nom()}", S.f_h, 13.5, W - 2 * M - 260), S.f_h, 13.5, S.ink)
        d.t(W - M, y, f"{TITRE} · configuration {res['configuration']} · {E.LIBELLES_MODE[res['mode']]} · {res['materiau']['etat_limite']}", S.f_b, 7.6, S.mut, "right")
        y -= 7
        d.line(M, y, W - M, y, S.ink, 1.0)
        top = y - 6
        bas = M + CART_H + 10
        # le résumé de méthode sous le titre
        fr = Frame(M, top, W - 2 * M, bas)
        fr.down(1)
        self.resume(d, fr)
        top_vues = fr.y - 9
        # les textes du bas, mesurés d'abord : la rangée des vues peut prendre tout ce qui reste au-dessus d'eux ; elle
        # prend la hauteur de sa plus haute colonne (les dessins sont limités par la largeur des colonnes), et les textes
        # la suivent directement — le blanc, s'il en reste, est au-dessus du cartouche, comme sur une planche
        h_textes = self.hauteur_textes(W - 2 * M - LAB_W * self.k)
        wc = (W - 2 * M - 2 * GAP) / 3.0
        xs = [M + i * (wc + GAP) for i in range(3)]
        preps = [self.preparer(cle, wc, top_vues - (bas + h_textes + GAP)) for cle, _ in self.COLONNES]
        bas_vues = max(bas + h_textes + GAP, top_vues - max(p["h_col"] for p in preps) - 2)
        for x in xs[1:]:
            d.line(x - GAP / 2, top_vues, x - GAP / 2, bas_vues + GAP / 2, S.rule, 0.45)
        d.line(M, bas_vues + GAP / 2, W - M, bas_vues + GAP / 2, S.rule, 0.45)
        for rang, (x, (cle, titre), prep) in enumerate(zip(xs, self.COLONNES, preps)):
            f = Frame(x, top_vues, wc, bas_vues)
            self.vue(d, f, cle, titre, prep)
            if f.overflow() > 1.0:
                self.warnings.append(Debord(2, rang, f.x, f.overflow()))
        ft = Frame(M, bas_vues, W - 2 * M, bas)
        self.textes(d, ft)
        if ft.overflow() > 1.0:
            self.warnings.append(Debord(2, 3, ft.x, ft.overflow()))
        self.cartouche(d)
        return self.warnings


# ------------------------------------------------------------------ page de suite
class Suite:
    """Page 3 · SUITE : les blocs du bas d'une colonne de la page 1 qui ne tiennent pas, même en taille réduite (le
    mécanisme de la note du garde-corps : ``reports`` = les blocs qui partent, ``((page, rang), nom)``, choisis par
    ``mise_en_page``). La page de suite a trois colonnes ; chaque colonne d'origine qui reporte en reçoit une (une
    quatrième s'empile dans la troisième), sous son titre « SUITE DE LA PAGE 1 · CALCUL ». Un bloc reporté est d'abord
    dessiné AU MOMENT où sa page l'aurait dessiné, sur un brouillon et dans sa colonne de suite (sa numérotation et sa
    hauteur sont celles qu'il aurait eues), puis redessiné en page 3 avec l'état de sa page d'alors : le même dessin.
    La page 2 ne reporte jamais (``BLOCS``)."""

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
    """(k, nom de la taille, reports, débordements, pages) : la taille normale, puis la réduite ; si une colonne de la
    page 1 déborde encore en taille réduite, le DERNIER bloc de chaque colonne qui déborde passe en page de suite, et
    ainsi de suite jusqu'à ce que tout tienne (ou qu'il n'y ait plus rien à reporter : les débordements restent, et le
    disent). La page 2 n'a rien à reporter : elle ne déborde pas (ses textes se réduisent)."""
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
    """La note (bytes PDF) : la page de garde du site (sommaire : une ligne par cornière, son statut, son état limite, sa
    page), puis les pages de chaque cornière (deux, trois avec une page de suite). ``infos`` : nom_projet, partie, date,
    indice. Une cornière hors domaine arrête la note avec le message du moteur (jamais un chiffre faux imprimé)."""
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
                                    f"{E.LIBELLES_MODE[res['mode']]} · {res['materiau']['etat_limite']} · η max {_pct(res['taux_max'])}",
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

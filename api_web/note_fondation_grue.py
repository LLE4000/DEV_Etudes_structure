# -*- coding: utf-8 -*-
"""La NOTE DE CALCUL PDF du module « Fondations de grues » sur le site — le générateur et sa plomberie.

Elle a la forme des notes Poutre et Dalle (le kit ``ndc_pdf`` : palette Encre, mêmes polices, la même page de garde) et
elle dit LES CHIFFRES DE L'ÉCRAN : elle ne calcule rien. Tout vient de ``beton.fondation_grue.moteur.verifier`` (le
résultat que l'écran affiche), les formules sont écrites par ``beton.fondation_grue.ecriture`` (la même écriture que
les fenêtres de calcul), les dessins sont les scènes de ``beton.fondation_grue.schemas`` (celles de l'écran, peintes
telles quelles), les mots et les nombres sont ceux de l'écran (``api_web.fondation_grue`` : une source).

UNE NOTE = la page de garde (A4 portrait : projet, partie, date, indice, sommaire — une ligne par fondation) puis, PAR
FONDATION, des pages A4 paysage :

  · CALCUL (une page, ou plusieurs) — deux colonnes qui se remplissent dans l'ordre de lecture : les hypothèses (la
    méthode du classeur, les matériaux et leurs contraintes), les données saisies AVEC LEUR CELLULE du classeur
    (géométrie, tolérances, charges de la grue), puis un bloc par partie du classeur — ses sollicitations, chaque
    vérification avec ses formules « symbole = expression = valeurs = résultat » et la cellule dont chacune reproduit la
    valeur, son verdict (l'état que le CLASSEUR donne : OK, LIMITE, A VERIFIER, PAS OK), le cisaillement et sa zone —,
    les contrôles de géométrie, les ÉCARTS relevés dans le classeur (les deux chiffres, jamais corrigés), le
    récapitulatif, ce que la note ne couvre pas, la conclusion ;
  · DIAGRAMMES M ET V — pour chaque poutre et chaque semelle : la poutre sur deux appuis dans chaque position de
    tolérance, une courbe par position, l'enveloppe en trait fort, la valeur que le classeur retient en tireté ; et le
    tableau des positions (portée, réactions, moments, efforts tranchants) ;
  · PLAN DE PRINCIPE (la dernière page) — la vue en plan à une échelle normalisée, les pièces, les armatures choisies
    (ce qui part vers RBAR), ce que le calcul ne fixe pas, le cartouche. Ni plan de coffrage, ni plan d'armatures.

MISE EN PAGE — les blocs du calcul sont MESURÉS puis posés : un bloc qui ne tient plus dans la colonne passe à la
suivante, puis à la page suivante. Rien n'est jamais tronqué ; la taille du texte est la plus grande (``VARIANTES``) qui
n'ajoute pas de page. Un bloc plus haut qu'une colonne entière — jamais vu — fait refuser la note avec un message.

    generer_note(etat_ou_fondations, infos, chemin=None) -> bytes      infos = {nom_projet, partie, indice, date}
    fichier(etat, infos) -> {"nom", "type": "application/pdf", "base64"}     (le contrat de api_web/note_poutre.fichier)

L'IDENTIFICATION est TOUJOURS celle de la note du site (R2) : projet = le chantier, à défaut le nom de la note ; partie =
le nom de la note quand il y a un chantier, sinon le titre du module ; indice ; date de la page.
"""
import base64
import copy
import html
import io
import os
import re
import tempfile
from datetime import datetime
from functools import partial
from typing import Dict, List, Optional, Tuple

from reportlab.lib.pagesizes import A4, landscape

from beton.fondation_grue import ecarts as EC
from beton.fondation_grue import ecriture as W
from beton.fondation_grue import entree as EN
from beton.fondation_grue import export as X
from beton.fondation_grue import moteur as MO
from beton.fondation_grue import schemas as SC
from ndc_pdf import data as ndc_data
from ndc_pdf import riche as RI
from ndc_pdf.kit import Doc, Frame, chip, hx, mix
from ndc_pdf.mathx import MathStyle
from ndc_pdf.styles import Encre

from . import commun as C
from . import fondation_grue as FG

A4L, A4P = landscape(A4), A4
M, GAP = 22, 16                 # marges de la page, gouttière entre deux colonnes
BAS = 32                        # le bas des colonnes : dessous, la ligne de référence et le numéro de page
CART_H = 40                     # le cartouche de la dernière page
LAB_W = 62                      # la colonne des libellés d'une ligne « LIBELLÉ   valeur »
TITRE = "Fondation de grue"
TITRE_MODULE = "Fondations de grues"
TITRE_NOTE = "Note de calcul — Fondations de grues"
# les tailles du texte (voir ``mise_en_page``) : le nombre de pages de calcul est celui de la taille NORMALE ; la taille
# retenue est la plus grande qui tient dans ce nombre de pages (une dernière page presque vide ne sert à personne) ; la
# taille « serrée » ne sert que si un bloc ne tient pas dans une colonne en taille normale
NORMALE, SERREE = 1.0, 0.93
VARIANTES = ((1.30, "très grande"), (1.22, "grande"), (1.15, "assez grande"), (1.08, "moyenne"), (NORMALE, "normale"), (SERREE, "serrée"))
STATUTS = {MO.OK: "CONFORME", MO.WARN: "À VÉRIFIER", MO.NOK: "NON CONFORME"}
LEGENDE_GARDE = ("Vert : conforme (OK du classeur) · Ocre : à vérifier (LIMITE, A VERIFIER, cisaillement en zone 3) · "
                 "Rouge : non conforme (PAS OK)")
# les échelles normalisées du plan de principe (1:n) : la plus grande qui tient dans la page
ECHELLES = (10, 20, 25, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500, 750, 1000)
PT_PAR_MM = 72.0 / 25.4
ALIGNEMENTS = {"start": "left", "middle": "center", "end": "right"}

# LES DONNÉES SAISIES, par schéma : [(titre de la ligne, [(libellé — texte riche —, chemin de la saisie, unité)])]. Chaque
# donnée s'imprime avec la CELLULE du classeur où elle se saisit (``entree.SAISIES``) ; les libellés sont ceux de l'écran.
# tests/test_note_fondation_grue.py exige que toute saisie de ``entree.SAISIES`` soit dans la note.
DONNEES = {
    EN.SEMELLES8: (
        ("Châssis", (("$L_c$", "chassis", "cm"), ("a", "dist_a", "cm"), ("b", "dist_b", "cm"), ("enrobage c", "enrobage", "mm"))),
        ("Poutre support", (("b", "support/b", "cm"), ("h", "support/h", "cm"))),
        ("Semelle", (("b", "semelle/b", "cm"), ("h", "semelle/h", "cm"), ("longueur $L_s$", "semelle/L", "cm"), ("appui c", "c_appui", "cm"))),
        ("Pieux", (("Ø", "pieux/D", "cm"), ("entraxe s", "pieux/entraxe", "cm"), ("capacité", "pieux/capacite", "T"))),
        ("Tolérances", (("poutre support e", "e_support", "cm"), ("semelle e", "e_semelle", "cm"))),
        ("Étriers", (("part de V comptée, poutre support", "support/part_V", "%"),)),
    ),
    EN.CADRE4: (
        ("Châssis", (("L", "chassis", "cm"),)),
        ("Poutre", (("b", "poutre/b", "cm"), ("h", "poutre/h", "cm"))),
        ("Pieux", (("Ø", "pieux/D", "cm"), ("capacité", "pieux/capacite", "kN"))),
        ("Tolérance", (("pieux e", "e", "cm"),)),
        ("Étriers", (("part de V comptée", "poutre/part_V", "%"),)),
    ),
    EN.DIAGONALES8: (
        ("Châssis", (("L", "chassis", "cm"),)),
        ("Poutre secondaire", (("b", "secondaire/b", "cm"), ("h", "secondaire/h", "cm"))),
        ("Poutre primaire", (("b", "primaire/b", "cm"), ("h", "primaire/h", "cm"), ("longueur $L_p$", "primaire/L", "cm"))),
        ("Pieux", (("Ø", "pieux/D", "cm"), ("entraxe s", "pieux/entraxe", "cm"))),
        ("Tolérances", (("poutre secondaire e", "e_secondaire", "cm"), ("poutre primaire e", "e_primaire", "cm"))),
        ("Étriers", (("part de V comptée, poutre secondaire", "secondaire/part_V", "%"),)),
    ),
}
CHARGES = (("$H_s$", "charges/H_s", "kN"), ("$H_{hs}$", "charges/H_hs", "kN"), ("$M_s$", "charges/M_s", "kN·m"))
TEXTES_FONDATION = (("", "projet"), ("grue", "grue"), ("flèche", "fleche"), ("hauteur sous crochet", "hauteur_crochet"),
                    ("niveau du béton", "niveau_beton"), ("châssis", "dimensions_chassis"))
ORIENTATIONS = "A/B, B/A, B/C, C/B, C/D, D/C, D/A, A/D"
METHODE_TOLERANCES = (
    "Chaque poutre et chaque semelle est une poutre sur deux appuis. Pour chaque position de tolérance (un pieu exécuté à côté de sa place, "
    "le châssis posé décalé), les appuis et les charges sont placés là où cette position les met : chaque position donne sa courbe de moment "
    "et sa courbe d'effort tranchant. L'enveloppe est en trait fort ; la valeur que le classeur retient est repérée par un tireté.")


class Debord(str):
    """Un débordement : le texte qui le dit (« page « plan de principe », colonne 430 : 28 pt »)."""

    def __new__(cls, page: str, x: float, pt: float):
        return str.__new__(cls, f"page « {page} », colonne {x:.0f} : {pt:.0f} pt")


# ------------------------------------------------------------------ les mots et les nombres : ceux de l'écran
def _texte(x) -> str:
    return str(x or "").strip()


def _clair(s: str) -> str:
    """Un texte que l'écran rend en HTML échappé (« &gt; », « &amp; »), remis en clair pour la note."""
    return html.unescape(s)


def comparaison(v: Dict) -> str:
    """« néc. 2 458 ≤ posé 2 513 mm² » : la phrase de l'écran (``FG._comparaison``), en clair."""
    return _clair(FG._comparaison(v, etat=False))


def taux_bloc(b: Dict) -> Optional[float]:
    taux = [v["taux"] for v in b["verifications"] if v["taux"] is not None]
    return max(taux) if taux else None


def efforts(r: Dict, b: Dict) -> str:
    """La pièce d'un bloc et les sollicitations que le classeur retient pour elle (la ligne d'efforts de l'écran)."""
    s = b["piece"]
    el = next((e for e in r["elements"] if e["cle"] == b["cle"]), None)
    if el is None:
        return s
    d = el["dimensionnant"]
    for sym, cle, unite in (("$M_{max}$", "M_max", "kN·m"), ("$M_{min}$", "M_min", "kN·m"), ("$V_{max}$", "V_max", "kN")):
        if d.get(cle) is not None:
            s += f" · {sym} = {FG._nu(d[cle], unite)}"
    return s


def cellule_de(f: Dict, chemin: str) -> str:
    return EN.SAISIES[f["type"]].get(chemin, "")


def nom_formule(nom: str) -> str:
    """Le nom d'une formule en mini-LaTeX : son symbole composé, et ce qui le qualifie (« — B.b (CAS 1.b-B/A) »,
    « (CAS A), t = 0,60 ») en texte droit — le compositeur écrirait en italique des lettres qui ne sont pas des grandeurs."""
    for sep in (" — ", " ("):
        if sep in nom:
            sym, _, reste = nom.partition(sep)
            reste = (sep + reste).strip()
            return RI.tex(sym) + r" \t{ " + reste.replace("{", "(").replace("}", ")") + "}"
    return RI.tex(nom)


# ------------------------------------------------------------------ le peintre des scènes de l'écran
def peindre(d: Doc, scene, x: float, y: float, w: float, h: float, echelle: Optional[float] = None, haut: bool = False) -> float:
    """Peint la scène (``schemas.Scene`` : les primitives du dessin de l'écran) dans la boîte (x, y, w, h) — origine en
    bas à gauche —, centrée (ou calée en haut avec ``haut``), à l'échelle ``echelle`` (points par unité de la scène) ou à
    la plus grande qui tient. Tout est à l'échelle : les traits (0,25 pt au moins), les tirets, les textes. Rend l'échelle."""
    cx, cy, cw, ch = scene.cadre
    s = echelle or min(w / cw, h / ch)
    ox = x + (w - cw * s) / 2.0 - cx * s
    oy = (y + h if haut else y + h - (h - ch * s) / 2.0) + cy * s
    c = d.c

    def P(px, py):
        return ox + px * s, oy - py * s

    def trait(couleur, ep, tirets):
        c.setStrokeColor(hx(couleur))
        c.setLineWidth(max(0.25, ep * s))
        if tirets:
            c.setDash([max(0.6, float(v) * s) for v in str(tirets).replace(",", " ").split()])
        else:
            c.setDash()

    c.saveState()
    c.setLineJoin(1)
    for it in scene.items:
        k = it[0]
        if k == "ligne":
            _, x1, y1, x2, y2, couleur, ep, tirets = it
            trait(couleur, ep, tirets)
            c.line(*P(x1, y1), *P(x2, y2))
        elif k == "poly":
            _, pts, couleur, ep, rempli, ferme, tirets, opacite = it
            p = c.beginPath()
            p.moveTo(*P(*pts[0]))
            for pt in pts[1:]:
                p.lineTo(*P(*pt))
            if ferme:
                p.close()
            if rempli:
                c.setFillColor(hx(rempli))
                c.setFillAlpha(float(opacite) if opacite is not None else 1.0)
            if couleur:
                trait(couleur, ep, tirets)
            c.drawPath(p, stroke=1 if couleur else 0, fill=1 if rempli else 0)
            c.setFillAlpha(1.0)
        elif k == "cercle":
            _, px, py, r, couleur, ep, rempli = it
            if rempli:
                c.setFillColor(hx(rempli))
            trait(couleur, ep, None)
            c.circle(*P(px, py), r * s, stroke=1, fill=1 if rempli else 0)
        else:
            _, tx, ty, txt, taille, ancre, couleur, gras, angle = it
            police = "Carlito-Bold" if gras else "Carlito"
            X_, Y_ = P(tx, ty)
            if angle:
                c.saveState()
                c.translate(X_, Y_)
                c.rotate(-angle)
                d.t(0, 0, txt, police, taille * s, couleur, ALIGNEMENTS[ancre])
                c.restoreState()
            else:
                d.t(X_, Y_, txt, police, taille * s, couleur, ALIGNEMENTS[ancre])
    c.setDash()
    c.restoreState()
    return s


def echelle_plan(scene, w: float, h: float) -> Tuple[int, float]:
    """(n, points par mm de la scène) : la plus grande échelle normalisée 1:n à laquelle le plan (en mm) tient dans w × h."""
    _, _, cw, ch = scene.cadre
    for n in ECHELLES:
        s = PT_PAR_MM / n
        if cw * s <= w and ch * s <= h:
            return n, s
    n = ECHELLES[-1]
    return n, PT_PAR_MM / n


# ------------------------------------------------------------------ ce qu'une fondation fait imprimer
class Calcul:
    """Une fondation, calculée une fois : le résultat du moteur (celui de l'écran) et ses vues."""

    def __init__(self, fondation: Dict):
        self.res = MO.verifier(fondation)
        self.f = self.res["fondation"]
        self.vues = {cle: (titre, scene, legende) for cle, titre, _court, _ech, scene, legende in SC.vues(self.res)}
        self.controle = X.controle(self.res)

    def nom(self) -> str:
        return _texte(self.f.get("nom")) or TITRE

    def sous_titre(self) -> str:
        m = self.res["materiaux"]
        return f"{EN.COURTS_TYPE[self.f['type']]} · {m['beton']} · {m['acier']}"

    def gouvernant(self) -> Tuple[Dict, Dict]:
        """(la vérification qui gouverne, son bloc) ; ValueError si le moteur n'en nomme aucune (jamais vu : chaque
        schéma a des vérifications chiffrées) — la note ne s'imprime pas sans son taux de tête."""
        g = self.res["gouvernant"]
        if not g:
            raise ValueError("aucune vérification chiffrée : rien ne gouverne")
        v = next(x for x in self.res["verifications"] if x["cle"] == g["cle"])
        return v, next(b for b in self.res["blocs"] if b["cle"] == v["bloc"])


class Note:
    """Les briques et les blocs d'une fondation, à la taille ``k``."""

    def __init__(self, calc: Calcul, doc_meta: Dict, k: float = 1.0):
        self.C, self.doc, self.k = calc, doc_meta, k
        self.S = Encre()
        self.s_txt = 6.3 * k
        self.s_tab = 6.2 * k
        self.lead = 8.9 * k
        self._ms: Dict = {}

    # --- composition
    def ms(self, col=None, op=None, res=None) -> MathStyle:
        S = self.S
        key = (str(col), str(op), str(res))
        st = self._ms.get(key)
        if st is None:
            st = MathStyle(var="Carlito-Italic", rom="Carlito", num="Carlito", op="Carlito", ink=hx(col or S.ink), accent=hx(S.acc),
                           rule=hx(col or S.ink), res="Carlito-Bold", res_color=hx(res or S.acc), op_color=hx(op) if op else None, tight=0.88)
            self._ms[key] = st
        return st

    def couleur(self, etat: str) -> str:
        """La couleur d'un état du moteur : vert (OK du classeur), ocre (LIMITE, A VERIFIER, zone 3), rouge (PAS OK)."""
        S = self.S
        return {MO.OK: S.ok, MO.WARN: S.att, MO.NOK: S.ko}.get(etat, S.mut)

    def rt(self, d, x, y, s, size, col=None, font=None, align="left", width=None):
        col = col or self.S.ink
        return RI.texte(d, x, y, s, font or self.S.f_b, size, col, self.ms(col), align, width)

    def rp(self, d, fr, s, size=None, col=None, font=None, indent=0, lead=1.3, max_lignes=None):
        col = col or self.S.ink
        return RI.paragraphe(d, fr, s, font or self.S.f_b, size or self.s_txt, col, self.ms(col), lead, indent, max_lignes)

    # --- briques
    def titre(self, d, fr, txt, pastille=None, couleur=None):
        """Un titre de bloc en petites capitales espacées ; à droite, sa pastille (« À VÉRIFIER · 97,8 % »)."""
        S = self.S
        fr.down(11 * self.k)
        wc = 0.0
        if pastille:
            wc = d.w(pastille, "Carlito-Bold", 5.8) + 10
            chip(d, fr.x1 - wc, fr.y - 2.5, pastille, "Carlito-Bold", 5.8, "#FFFFFF", couleur or S.acc, pad=5, h=8.4)
        avail = fr.w - (wc + 6 if wc else 0.0)
        size, track = 6.4 * self.k, 1.2
        while size > 5.0 and d.w(txt, S.f_v, size) + track * len(txt) > avail:
            size -= 0.2
            track = max(0.3, track - 0.1)
        d.t(fr.x, fr.y, d.fit(txt, S.f_v, size, avail - track * len(txt)), S.f_v, size, S.acc, track=track)
        fr.down(3.2)
        d.line(fr.x, fr.y, fr.x1, fr.y, mix(S.rule, S.ink, 0.12), 0.5)

    def etiquette(self, d, fr, txt, droite=""):
        """Le libellé d'un groupe de formules, en petites capitales grises ; à droite, ce qu'il désigne (« 8Ø20 »)."""
        S = self.S
        fr.down(self.s_txt * 1.65)
        size, track = 5.0 * self.k, 0.6
        wd = 0.0
        if droite:
            wd = d.w(droite, "Carlito-Bold", self.s_txt) + 6
            d.t(fr.x1, fr.y, droite, "Carlito-Bold", self.s_txt, S.ink, "right")
        d.t(fr.x, fr.y, d.fit(txt.upper(), S.f_v, size, fr.w - wd - track * len(txt)), S.f_v, size, S.mut, track=track)

    def largeur_libelles(self, d, libelles) -> float:
        """La colonne des libellés d'un bloc de lignes : celle du kit, élargie au plus long libellé du bloc (« POUTRE
        SECONDAIRE » ne doit pas toucher sa valeur)."""
        size, track = 4.9 * self.k, 0.8
        return max([LAB_W * self.k] + [d.w(lab.upper(), self.S.f_v, size) + track * len(lab) + 8 for lab in libelles])

    def ligne(self, d, fr, label, val, col=None, lab_w=None):
        """« LIBELLÉ   valeur » : le libellé en petites capitales à gauche, la valeur (texte riche) à droite, repliée."""
        S = self.S
        size = self.s_txt
        col = col or S.ink
        lab_w = lab_w or LAB_W * self.k
        lignes = RI.envelopper(val, S.f_b, size, self.ms(col), fr.w - lab_w)
        fr.down(size * 1.42)
        d.t(fr.x, fr.y + 0.2, label.upper(), S.f_v, 4.9 * self.k, S.mut, track=0.8)
        for i, l in enumerate(lignes):
            if i:
                fr.down(size * 1.28)
            RI.dessiner(d, fr.x + lab_w, fr.y, l, S.f_b, size, col, self.ms(col))

    def ligne_donnees(self, d, fr, label, items, lab_w=None):
        """« LIBELLÉ   a = 25 cm ᶜ³⁵ · b = 25 cm ᶜ³⁷ » : des données saisies, chacune suivie de SA CELLULE du classeur
        en petit et en gris ; la ligne se replie entre deux données. ``items`` = [(texte riche, cellule)]."""
        S = self.S
        size, s_cel = self.s_txt, self.s_txt * 0.74
        lab_w = lab_w or LAB_W * self.k
        ms = self.ms(S.ink)
        x0, x1 = fr.x + lab_w, fr.x1
        fr.down(size * 1.42)
        d.t(fr.x, fr.y + 0.2, label.upper(), S.f_v, 4.9 * self.k, S.mut, track=0.8)
        x = x0
        for i, (txt, cel) in enumerate(items):
            wt = RI.largeur(txt, S.f_b, size, ms)
            wcel = d.w(cel, S.f_b, s_cel) + 2.5 if cel else 0.0
            sep = d.w(" · ", S.f_b, size) if x > x0 else 0.0
            if x > x0 and x + sep + wt + wcel > x1 + 0.01:
                fr.down(size * 1.28)
                x, sep = x0, 0.0
            if sep:
                d.t(x, fr.y, " · ", S.f_b, size, S.mut)
                x += sep
            RI.dessiner(d, x, fr.y, RI.jetons(txt), S.f_b, size, S.ink, ms)
            x += wt
            if cel:
                d.t(x + 2.0, fr.y + size * 0.28, cel, S.f_b, s_cel, S.mut)
                x += wcel

    def formule(self, d, fr, fx, indent=4):
        """Une formule du classeur : « symbole = expression = valeurs = résultat », repliée sur les « = » ; au bout de sa
        dernière ligne, calée à droite, la CELLULE dont elle reproduit la valeur."""
        S = self.S
        size = self.s_txt
        parts, res = W.morceaux(fx)
        cel = fx.get("cellule") or ""
        w_cel = d.w(cel, S.f_b, size * 0.8) + 6 if cel else 0.0
        col_res = None
        f2 = Frame(fr.x, fr.top, fr.w - w_cel, fr.bottom)
        f2.y, f2.min_y = fr.y, fr.min_y
        ms = self.ms(S.ink, op=S.mut)
        RI.formule_calee(d, f2, [nom_formule(parts[0])] + [RI.tex(p) for p in parts[1:]], r"\res{" + RI.tex(res) + "}", ms, size,
                         ms_res=self.ms(S.ink, op=S.mut, res=col_res), indent=indent, retrait=10, lead=1.38, min_size=4.8, cale="suite")
        fr.y, fr.min_y = f2.y, min(fr.min_y, f2.min_y)
        if cel:
            d.t(fr.x1, fr.y, cel, S.f_b, size * 0.8, S.mut, "right")

    def bande(self, d, fr, txt, etat, droite=""):
        """Le verdict d'une vérification : une bande teintée à la couleur de l'état (vert, ocre, rouge), la comparaison à
        gauche, le taux et l'état du classeur à droite."""
        S = self.S
        col = self.couleur(etat)
        size = self.s_txt
        wd = d.w(droite, "Carlito-Bold", size) + 8 if droite else 0.0
        lignes = RI.envelopper(txt, S.f_b, size, self.ms(S.ink), fr.w - 12 - wd)
        h = len(lignes) * size * 1.28 + 4.2
        fr.down(2.0)
        fr.down(h)
        d.box(fr.x, fr.y, fr.w, h, fill=mix(col, "#FFFFFF", 0.88))
        d.box(fr.x, fr.y, 2.0, h, fill=col)
        yy = fr.y + h - size * 1.05 - 1.2
        for l in lignes:
            RI.dessiner(d, fr.x + 7, yy, l, S.f_b, size, S.ink, self.ms(S.ink))
            yy -= size * 1.28
        if droite:
            d.t(fr.x1 - 5, fr.y + h - size * 1.05 - 1.2, droite, "Carlito-Bold", size, col, "right")
        fr.down(1.5)

    def tableau(self, d, fr, cols, lignes, note=None):
        """Un tableau simple : ``cols`` = [(en-tête, largeur, alignement)] ; ``lignes`` = des cellules (texte riche,
        couleur ou None, gras) — un texte « left » se replie dans sa colonne."""
        S = self.S
        st, lead = self.s_tab, self.lead
        xs, x = [], fr.x
        fr.down(8 * self.k)
        for ent, w, al in cols:
            xs.append(x)
            RI.texte(d, x + (w - 3 if al == "right" else 0), fr.y, ent, S.f_b, 5.6 * self.k, S.mut, self.ms(S.mut), al)
            x += w
        fr.down(2.5)
        d.line(fr.x, fr.y, fr.x1, fr.y, S.acc, 0.5)
        for cells in lignes:
            rendus, n_max = [], 1
            for (ent, w, al), cell in zip(cols, cells):
                txt, col, gras = (cell + (None, False))[:3] if isinstance(cell, tuple) else (cell, None, False)
                col = col or S.ink
                f = "Carlito-Bold" if gras else S.f_b
                ls = RI.envelopper(txt, f, st, self.ms(col), w - 4) if al == "left" else [RI.jetons(txt)]
                n_max = max(n_max, len(ls))
                rendus.append((ls, col, f))
            fr.down(lead - 1.8)
            y = fr.y
            for (ent, w, al), (ls, col, f), xx in zip(cols, rendus, xs):
                for i, l in enumerate(ls):
                    RI.dessiner(d, xx + (w - 3 if al == "right" else 0), y - i * lead, l, f, st, col, self.ms(col), al)
            fr.down(lead * (n_max - 1) + 3.0)
            d.line(fr.x, fr.y + 1.5, fr.x1, fr.y + 1.5, mix(S.rule, S.ink, 0.06), 0.4)
        if note:
            fr.down(st * 0.3)
            RI.paragraphe(d, fr, note, S.f_b, st * 0.92, S.mut, self.ms(S.mut), 1.24)
        fr.down(2)

    # --- les blocs du calcul, dans l'ordre de lecture
    def hypotheses(self, d, fr):
        r, f = self.C.res, self.C.f
        m = r["materiaux"]
        z = " / ".join(C.fr(x, 2) for x in m["tau_zones"])
        self.titre(d, fr, "HYPOTHÈSES")
        fr.down(1)
        lw = self.largeur_libelles(d, ("Méthode", "Schéma", "Béton", "Acier", "Cisaillement", "Chantier"))
        self.ligne(d, fr, "Méthode", f"contraintes admissibles — celle du classeur du bureau « Fondation grue tour sur pieux », fiche {r['classeur']}", lab_w=lw)
        schema = EN.LIBELLES_TYPE[f["type"]]
        if f["type"] == EN.SEMELLES8:
            schema += f" · côté b : {EN.LIBELLES_COTE_B[f['cote_b']]}"
        self.ligne(d, fr, "Schéma", schema, lab_w=lw)
        self.ligne_donnees(d, fr, "Béton", [(f"{m['beton']}", cellule_de(f, "beton")), (f"$R'_{{wk}}$ = {FG._g(m['Rwk'])} N/mm²", ""),
                                            (f"$σ'_b$ = {C.fr(m['sigma_b'], 2)} N/mm²", "")], lw)
        self.ligne_donnees(d, fr, "Acier", [(f"{m['acier']}", cellule_de(f, "acier")), (f"$σ_{{adm}}$ = {C.fr(m['sigma_adm'], 2)} N/mm²", ""),
                                            (f"μ = {C.fr(m['mu'], 4)}", "")], lw)
        self.ligne(d, fr, "Cisaillement", f"zones 1 / 2 / 3 jusqu'à {z} N/mm² (au-delà de la zone 3 : non conforme)", lab_w=lw)
        textes = [(f"{lib} {_texte(f.get(cle))}".strip(), cellule_de(f, cle)) for lib, cle in TEXTES_FONDATION if _texte(f.get(cle))]
        if textes:
            self.ligne_donnees(d, fr, "Chantier", textes, lw)
        fr.down(2)

    def geometrie(self, d, fr):
        f = self.C.f
        self.titre(d, fr, "DONNÉES — géométrie et tolérances d'exécution (en petit : la cellule du classeur)")
        fr.down(1)
        lw = self.largeur_libelles(d, [label for label, _champs in DONNEES[f["type"]]])
        for label, champs in DONNEES[f["type"]]:
            self.ligne_donnees(d, fr, label, [(f"{lib} = {FG._g(EN.lire(f, ch))} {u}", cellule_de(f, ch)) for lib, ch, u in champs], lw)
        fr.down(2)

    def charges(self, d, fr):
        S, f = self.S, self.C.f
        ch = f["charges"]
        self.titre(d, fr, "CHARGES DE LA GRUE")
        if f["type"] == EN.SEMELLES8:
            fr.down(1)
            w = fr.w
            wn = 30.0 * self.k
            cols = [("CAS", w - 4 * wn - 54 - 62, "left"), ("ÉTAT", 54, "left")] + [(p + " (kN)", wn, "right") for p in EN.PIEDS] + [("CELLULES", 62, "right")]
            lignes = []
            for i, cas in enumerate(ch["cas"]):
                cels = [cellule_de(f, f"charges/cas/{i}/{p}") for p in EN.PIEDS]
                cel = f"{cels[0]}:{cels[-1]}" if all(cels) else "—"
                lignes.append([(cas["nom"], None, True), (EN.LIBELLES_ETAT_CAS[cas["etat"]], S.mut)] + [FG._g(cas[p]) for p in EN.PIEDS] + [(cel, S.mut)])
            self.tableau(d, fr, cols, lignes,
                         note=f"Réactions verticales de la grue par pied (A, B, C, D en tournant). Chaque cas est essayé dans les huit orientations "
                              f"de la grue ({ORIENTATIONS}) et dans chaque position de tolérance.")
        else:
            fr.down(1)
            self.ligne_donnees(d, fr, "Par pied", [(f"P = {FG._g(ch['P'])} kN", cellule_de(f, "charges/P"))])
        self.ligne_donnees(d, fr, "Horizontal", [(f"{lib} = {FG._g(EN.lire(f, c))} {u}", cellule_de(f, c)) for lib, c, u in CHARGES])
        fr.down(2)

    def bloc_tete(self, b, i, d, fr):
        """L'en-tête d'un bloc du classeur : son numéro et son nom, son état et son taux, la pièce et les sollicitations
        que le classeur retient pour elle ; ses formules de sollicitation suivent, une par bloc de mise en page."""
        S, r = self.S, self.C.res
        t = taux_bloc(b)
        pastille = STATUTS[b["etat"]] + (f" · {FG._pct(t)}" if t is not None else "")
        self.titre(d, fr, f"{i} · {b['nom'].upper()}", pastille, self.couleur(b["etat"]))
        fr.down(0.5)
        self.rp(d, fr, efforts(r, b), size=self.s_txt * 0.96, col=S.mut)
        if b["sollicitations"]:
            self.etiquette(d, fr, "Sollicitations")

    def sollicitation(self, fx, dernier, d, fr):
        """Une sollicitation d'un bloc (une formule) ; la dernière est suivie d'un petit blanc."""
        self.formule(d, fr, fx)
        if dernier:
            fr.down(1)

    def verification(self, v, d, fr):
        """Une vérification : son nom et sa pièce, ses formules, puis son verdict — l'état que le classeur lui donne."""
        self.etiquette(d, fr, v["nom"], v["piece"])
        for fx in v["formules"]:
            self.formule(d, fr, fx)
        droite = (f"{FG._pct(v['taux'])} · " if v["taux"] is not None else "") + f"{v['etat_classeur']} ({v['cellule_etat']})"
        self.bande(d, fr, comparaison(v), v["etat"], droite)
        fr.down(1)

    def bloc_infos(self, b, d, fr):
        """Ce que le classeur calcule sans le comparer, ou prescrit : le cisaillement et sa zone, les valeurs de passage,
        les prescriptions ; et les écarts relevés dans ce bloc (leur détail est au bloc « Écarts »)."""
        S = self.S
        for info in b["informations"]:
            if info.get("formule"):
                self.etiquette(d, fr, "Cisaillement")
                self.formule(d, fr, info["formule"])
                self.bande(d, fr, _texte(info.get("texte")), info["etat"], _texte(info.get("zone")).capitalize())
        autres = [i for i in b["informations"] if not i.get("formule")]
        if autres:
            self.etiquette(d, fr, "Informations du classeur")
            for info in autres:
                txt = RI.auto(info["nom"])
                if info.get("valeur") is not None:
                    txt += f" = {FG._nu(info['valeur'], info.get('unite', ''))}"
                if info.get("texte"):
                    txt += f" · {info['texte']}"
                cel = _texte(info.get("cellule"))
                f2 = Frame(fr.x, fr.top, fr.w - 22, fr.bottom)
                f2.y, f2.min_y = fr.y, fr.min_y
                self.rp(d, f2, txt, size=self.s_txt * 0.96, col=S.mut, indent=4, lead=1.28)
                fr.y, fr.min_y = f2.y, min(fr.min_y, f2.min_y)
                if cel:
                    d.t(fr.x1, fr.y, cel, S.f_b, self.s_txt * 0.8, S.mut, "right")
        if b.get("ecarts"):
            codes = ", ".join(dict.fromkeys(e["code"] for e in b["ecarts"]))
            fr.down(self.s_txt * 0.3)
            self.rp(d, fr, f"Écart{'s' if len(b['ecarts']) > 1 else ''} relevé{'s' if len(b['ecarts']) > 1 else ''} dans ce bloc : {codes} — détail au bloc « Écarts ».",
                    size=self.s_txt * 0.96, col=S.att, indent=4)
        fr.down(2)

    def controles(self, d, fr):
        S, r = self.S, self.C.res
        ko = sum(1 for c in r["controles"] if not c["ok"])
        self.titre(d, fr, "CONTRÔLES DE GÉOMÉTRIE — les conditions que le classeur vérifie sur les dimensions",
                   f"{ko} NON RESPECTÉ{'S' if ko > 1 else ''}" if ko else "RESPECTÉS", S.ko if ko else S.ok)
        w = fr.w
        cols = [("CONDITION", w - 96 - 34 - 40, "left"), ("VALEURS", 96, "right"), ("CELLULE", 34, "right"), ("ÉTAT", 40, "right")]
        lignes = []
        for c in r["controles"]:
            signe = "≥" if " ≥ " in c["texte"] else "≤"
            vu = signe if c["ok"] else ("<" if signe == "≥" else ">")
            lignes.append([c["texte"], f"{FG._g(c['valeur'])} {vu} {FG._g(c['limite'])} {c['unite']}", (_texte(c.get("cellule")), S.mut),
                           ("OK" if c["ok"] else "NON", S.ok if c["ok"] else S.ko, True)])
        self.tableau(d, fr, cols, lignes)

    def ecarts_tete(self, d, fr):
        S, r = self.S, self.C.res
        self.titre(d, fr, "ÉCARTS RELEVÉS DANS LE CLASSEUR", f"{len(r['ecarts'])} À CONNAÎTRE", S.att)
        fr.down(0.5)
        self.rp(d, fr, "Le module calcule comme le classeur : aucun de ces écarts n'est corrigé. « Classeur » est la valeur de la fiche ; "
                       "« redémontré », celle que donne la même vérification refaite à partir de la statique (la poutre sur deux appuis).",
                size=self.s_txt * 0.96, col=S.mut)

    def ecart(self, e, d, fr):
        S, r = self.S, self.C.res
        grave = e["change_verdict"] or FG.NIVEAUX_GRAVITE.get(e["gravite"], "c") == "v"
        col = S.ko if e["change_verdict"] else (S.att if grave else S.mut)
        mot = "change le verdict" if e["change_verdict"] else EC.LIBELLES_GRAVITE.get(e["gravite"], "écart")
        fr.down(self.s_txt * 1.7)
        wd = d.w(mot, "Carlito-Bold", self.s_txt * 0.92) + 6
        d.t(fr.x1, fr.y, mot, "Carlito-Bold", self.s_txt * 0.92, col, "right")
        self.rt(d, fr.x, fr.y, f"{e['code']} · {e['titre']}", self.s_txt, S.ink, font="Carlito-Bold", width=fr.w - wd)
        bloc = next((b["nom"] for b in r["blocs"] if b["cle"] == e["bloc"]), "")
        u = f" {e.get('unite') or ''}".rstrip()
        faits = [x for x in (bloc, f"cellule {e['cellule']}" if e.get("cellule") else "",
                             f"classeur {_clair(FG._val_ecart(e['classeur']))}{u}" if e.get("classeur") is not None else "",
                             f"redémontré {_clair(FG._val_ecart(e['redemontre']))}{u}" if e.get("redemontre") is not None else "") if x]
        self.rp(d, fr, " · ".join(faits), size=self.s_txt * 0.96, col=S.ink, indent=4)
        self.rp(d, fr, RI.auto(e["message"]), size=self.s_txt * 0.96, col=S.mut, indent=4, lead=1.26)
        fr.down(1.5)

    def recapitulatif(self, d, fr):
        S, r = self.S, self.C.res
        gov = r["gouvernant"]["cle"] if r["gouvernant"] else None
        self.titre(d, fr, "RÉCAPITULATIF — chaque vérification, l'état que le classeur lui donne, son taux")
        w = fr.w
        cols = [("VÉRIFICATION", w - 150 - 62 - 36, "left"), ("COMPARAISON", 150, "left"), ("ÉTAT DU CLASSEUR", 62, "right"), ("TAUX", 36, "right")]
        noms = {b["cle"]: FG._nom_bloc(b) for b in r["blocs"]}
        lignes = []
        for v in r["verifications"]:
            col = self.couleur(v["etat"])
            g = v["cle"] == gov
            lignes.append([(f"{noms[v['bloc']]} · {FG._court(v).lower()}", None, g), (comparaison(v), S.mut),
                           (f"{v['etat_classeur']} ({v['cellule_etat']})", col, v["etat"] != MO.OK),
                           (FG._pct(v["taux"]) if v["taux"] is not None else "—", col, g or v["etat"] != MO.OK)])
        for b in r["blocs"]:
            for i in b["informations"]:
                if i.get("zone"):
                    col = self.couleur(i["etat"])
                    lignes.append([f"{noms[b['cle']]} · cisaillement", (f"τ = {FG._nu(i['valeur'], i.get('unite', ''))}", S.mut),
                                   (f"{_texte(i['zone']).capitalize()} ({i['cellule']})", col, i["etat"] != MO.OK), ("—", S.mut)])
        note = None
        if gov:
            v, b = self.C.gouvernant()
            note = f"En gras, la vérification qui gouverne : {FG._nom_bloc(b).lower()}, {FG._court(v).lower()} — {FG._pct(r['taux_max'])}."
        self.tableau(d, fr, cols, lignes, note)

    def limites(self, d, fr):
        S = self.S
        self.titre(d, fr, "CE QUE CETTE NOTE NE COUVRE PAS")
        fr.down(1)
        self.rp(d, fr, "Les classeurs — donc cette note — ne calculent pas : " + " ; ".join(MO.NON_COUVERT) + ".", size=self.s_txt * 0.96, col=S.mut)
        fr.down(2)

    def texte_conclusion(self) -> str:
        r = self.C.res
        v, b = self.C.gouvernant()
        txt = (f"Taux maximal {FG._pct(r['taux_max'])} — {FG._nom_bloc(b).lower()}, {FG._court(v).lower()} ({comparaison(v)}) ; "
               f"état du classeur : {v['etat_classeur']}. ")
        regard = FG._a_regarder(r)
        if r["statut"] == MO.NOK:
            txt += "La fondation n'est PAS conforme en l'état : " + " ; ".join(regard) + ". "
        elif regard:
            txt += "À regarder : " + " ; ".join(regard) + ". "
        else:
            txt += "Toutes les vérifications du classeur sont « OK ». "
        ko = [c["texte"] for c in r["controles"] if not c["ok"]]
        if ko:
            txt += ("Contrôles de géométrie non respectés : " if len(ko) > 1 else "Contrôle de géométrie non respecté : ") + " ; ".join(ko) + ". "
        if r["ecarts"]:
            n, s = len(r["ecarts"]), "s" if len(r["ecarts"]) > 1 else ""
            txt += f"{n} écart{s} relevé{s} dans le classeur ({codes_ecarts(r['ecarts'])}) : signalé{s}, non corrigé{s}. "
        return txt + f"Méthode aux contraintes admissibles ; formules, unités et arrondis : ceux des {r['reference']}."

    def conclusion(self, d, fr):
        S, r = self.S, self.C.res
        col = self.couleur(r["statut"])
        size = 6.6 * self.k
        lignes = RI.envelopper(self.texte_conclusion(), S.f_b, size, self.ms(S.ink), fr.w - 16)
        h = 13 + 8 + len(lignes) * size * 1.3 + 6
        fr.down(8)
        fr.down(h)
        d.box(fr.x, fr.y, fr.w, h, fill=mix(S.panel, "#FFFFFF", 0.3), stroke=col, lw=0.8, r=3)
        d.t(fr.x + 8, fr.y + h - 11, "CONCLUSION", S.f_v, 6.2, col, track=1.3)
        yy = fr.y + h - 11 - 9
        for l in lignes:
            RI.dessiner(d, fr.x + 8, yy, l, S.f_b, size, S.ink, self.ms(S.ink))
            yy -= size * 1.3

    def unites(self) -> List[Tuple[str, bool, object]]:
        """Les blocs du calcul, dans l'ordre : (clé, « ne se sépare pas du suivant », fonction (d, fr))."""
        r = self.C.res
        u = [("hypotheses", False, self.hypotheses), ("geometrie", False, self.geometrie), ("charges", False, self.charges)]
        for i, b in enumerate(r["blocs"], 1):
            u.append((f"{b['cle']}.tete", True, partial(self.bloc_tete, b, i)))
            n = len(b["sollicitations"])
            u += [(f"{b['cle']}.s{k}", False, partial(self.sollicitation, fx, k == n - 1)) for k, fx in enumerate(b["sollicitations"])]
            u += [(v["cle"], False, partial(self.verification, v)) for v in b["verifications"]]
            if b["informations"] or b.get("ecarts"):
                u.append((f"{b['cle']}.infos", False, partial(self.bloc_infos, b)))
        if r["controles"]:
            u.append(("controles", False, self.controles))
        if r["ecarts"]:
            u.append(("ecarts", True, self.ecarts_tete))
            u += [(f"ecart.{k}", False, partial(self.ecart, e)) for k, e in enumerate(r["ecarts"])]
        return u + [("recapitulatif", False, self.recapitulatif), ("limites", True, self.limites), ("conclusion", False, self.conclusion)]

    # --- l'habillage des pages
    def entete(self, d, titre: str, droite: str = "") -> float:
        """Le haut d'une page : le bureau, « Note de calcul · date · indice », le titre et l'identification ; rend l'y du
        filet sous le titre."""
        S, doc = self.S, self.doc
        W_, H = d.W, d.H
        y = H - M - 4
        d.t(M, y, doc.get("bureau", ""), S.f_v, 8.0, S.acc, track=1.4)
        d.t(W_ - M, y, "Note de calcul · " + doc.get("date", "") + " · indice " + str(doc.get("indice", "")), S.f_b, 8, S.mut, "right")
        y -= 15
        ident = droite or " · ".join(str(x) for x in (doc.get("projet"), doc.get("partie")) if x)
        wi = min(300.0, d.w(ident, S.f_b, 8))
        d.t(M, y, d.fit(titre, S.f_h, 12.5, W_ - 2 * M - wi - 14), S.f_h, 12.5, S.ink)
        d.t(W_ - M, y, d.fit(ident, S.f_b, 8, 300), S.f_b, 8, S.ink, "right")
        y -= 8
        d.line(M, y, W_ - M, y, S.ink, 1.0)
        return y

    def bandeau(self, d, y: float) -> float:
        """Sous le titre de la première page : le statut, η max, ce qui gouverne, le schéma et les matériaux."""
        S, r = self.S, self.C.res
        W_ = d.W
        col = self.couleur(r["statut"])
        y -= 15
        w = chip(d, M, y - 3.2, STATUTS[r["statut"]], S.f_v, 6.6, "#FFFFFF", col)
        x = M + w + 8
        t1 = "η max " + FG._pct(r["taux_max"])
        d.t(x, y, t1, "Carlito-Bold", 8.6, col)
        x += d.w(t1, "Carlito-Bold", 8.6) + 5
        v, b = self.C.gouvernant()
        t2 = f"— gouverne : {FG._nom_bloc(b).lower()}, {FG._court(v).lower()} · {comparaison(v)} · état du classeur : {v['etat_classeur']}"
        t3 = self.C.sous_titre()
        wt = d.w(t3, "Carlito-Bold", 7.6)
        xr = W_ - M - wt
        self.rt(d, x, y, t2, 8, S.ink, width=xr - x - 8)
        d.t(xr, y, t3, "Carlito-Bold", 7.6, S.acc)
        y -= 6
        d.line(M, y, W_ - M, y, mix(S.rule, S.ink, 0.3), 0.6)
        return y

    def pied(self, d, texte: str = ""):
        """Le bas d'une page : la ligne de référence à gauche, le numéro de la page à droite."""
        S = self.S
        txt = texte or (f"Formules, unités et arrondis : ceux des {self.C.res['reference']}. Chaque formule porte, à droite, la cellule de la fiche "
                        "dont elle reproduit la valeur.")
        for i, l in enumerate(RI.envelopper(txt, S.f_b, 5.4, self.ms(S.mut), d.W - 2 * M - 40)[:2]):
            RI.dessiner(d, M, 20 - i * 6.6, l, S.f_b, 5.4, S.mut, self.ms(S.mut))
        d.t(d.W - M, 20, f"page {d.n}", S.f_b, 6.2, S.mut, "right")

    def colonnes(self, d, top: float) -> List[Frame]:
        w = (d.W - 2 * M - GAP) / 2.0
        return [Frame(M, top, w, BAS), Frame(M + w + GAP, top, w, BAS)]

    def hauts(self, d) -> Tuple[float, float]:
        """(haut des colonnes de la première page de calcul, haut des suivantes) : sans rien dessiner."""
        y = d.H - M - 4 - 15 - 8
        return y - 15 - 6 - 4, y - 6

    # --- les pages
    def page_calcul(self, d, numero: int, n_pages: int, blocs_par_colonne) -> List[str]:
        S = self.S
        suite = f" — calcul, page {numero} sur {n_pages}" if n_pages > 1 else ""
        y = self.entete(d, f"{TITRE} — {self.C.nom()}{suite}")
        top = self.bandeau(d, y) - 4 if numero == 1 else y - 6
        cadres = self.colonnes(d, top)
        d.line(cadres[1].x - GAP / 2, top, cadres[1].x - GAP / 2, BAS, S.rule, 0.45)
        warn = []
        for fr, blocs in zip(cadres, blocs_par_colonne):
            for fn in blocs:
                fn(d, fr)
            if fr.overflow() > 1.0:
                warn.append(Debord("calcul", fr.x, fr.overflow()))
        self.pied(d)
        return warn

    def page_diagrammes(self, d, cles: List[str]) -> List[str]:
        """Les diagrammes M et V d'un ou deux éléments : la scène de l'écran, puis le tableau de ses positions de
        tolérance. Deux éléments : deux colonnes égales, la méthode en tête de page. Un seul : le dessin prend la grande
        colonne de gauche ; la méthode et le tableau, celle de droite."""
        S, r = self.S, self.C.res
        y = self.entete(d, f"DIAGRAMMES M ET V — {self.C.nom()}", f"{TITRE} · {self.C.sous_titre()}")
        seul = len(cles) == 1
        if seul:
            top = y - 6
            wg = (d.W - 2 * M - GAP) * 0.58
            cadres = [Frame(M, top, wg, BAS), Frame(M + wg + GAP, top, d.W - 2 * M - GAP - wg, BAS)]
        else:
            fr = Frame(M, y - 5, d.W - 2 * M, BAS)
            self.rp(d, fr, METHODE_TOLERANCES, size=6.6, col=S.ink, lead=1.3)
            top = fr.y - 7
            cadres = self.colonnes(d, top)
        d.line(cadres[1].x - GAP / 2, top, cadres[1].x - GAP / 2, BAS, S.rule, 0.45)
        warn, elements = [], []
        for i, cle in enumerate(cles):
            el = next(e for e in r["elements"] if e["cle"] == cle)
            elements.append(el)
            titre, scene, _legende = self.C.vues["mv_" + cle]
            fr = cadres[i]
            self.titre(d, fr, titre.upper())
            fr.down(3)
            h_table = 0.0 if seul else self.hauteur_positions(el, fr.w) + 4
            _, _, cw, ch = scene.cadre
            s = min(fr.w / cw, (fr.room() - h_table) / ch)
            h = ch * s
            fr.down(h)
            peindre(d, scene, fr.x, fr.y, fr.w, h, echelle=s)
            if seul:
                ft = cadres[1]
                self.titre(d, ft, "LES TOLÉRANCES D'EXÉCUTION — une poutre sur deux appuis")
                ft.down(1)
                self.rp(d, ft, METHODE_TOLERANCES, size=6.6, col=S.ink, lead=1.32)
                ft.down(4)
                self.titre(d, ft, f"{el['nom'].upper()} — positions de tolérance")
                self.positions(d, ft, el)
            else:
                self.positions(d, fr, el)
        warn += [Debord("diagrammes", f_.x, f_.overflow()) for f_ in cadres if f_.overflow() > 1.0]
        self.pied(d, SC.legende_mv(elements) + " La statique est écrite indépendamment de la transcription des classeurs ; les tests "
                     "vérifient que les deux donnent les mêmes réactions, moments et efforts tranchants.")
        return warn

    def lignes_positions(self, el):
        """(colonnes, lignes, note) du tableau des positions de tolérance d'un élément : la portée, les réactions, les
        extremums de M et de V de chaque position ; puis l'enveloppe et ce que le classeur retient."""
        S = self.S
        support = any("max" in p for p in el["positions"])
        wn = 42.0

        def val(v, u):
            return FG._n(v, u) if v is not None and abs(v) > 5e-7 else "—"

        cols = [("POSITION", 40, "left"), ("PORTÉE (m)", 36, "right"), ("$R_A$ (kN)", wn, "right"), ("$R_B$ (kN)", wn, "right"),
                ("$M_{max}$ (kN·m)", wn + 6, "right"), ("$M_{min}$ (kN·m)", wn + 6, "right"), ("$V_{max}$ (kN)", wn, "right")]
        if support:
            cols.append(("COLONNE (M max)", 0, "right"))
        lignes = []
        for p in el["positions"]:
            if "max" in p:                     # la poutre support : les maximums des colonnes du grand tableau de cette position
                mx = p["max"]
                vals = [mx["RA"]["valeur"], mx["RB"]["valeur"], mx["M_max"]["valeur"], None, mx["V_max"]["valeur"]]
            else:
                e = p["extremes"]
                vals = [p["RA"], p["RB"], e["M_max"]["valeur"], e["M_min"]["valeur"], e["V_max"]["valeur"]]
            ligne = [(p["cle"], SC.COULEURS.get(p["cle"], S.ink), True), C.fr(p["portee"], 2)] + [val(v, u) for v, u in zip(vals, ("kN", "kN", "kN·m", "kN·m", "kN"))]
            if support:
                ligne.append((FG._colonne(p["max"]["M_max"]), S.mut))
            lignes.append(ligne)
        ex, dim = el["extremes"], el["dimensionnant"]
        fin_ = [("trait fort", S.mut)] if support else []
        lignes.append([("enveloppe", None, True), "", "", "", (val(ex["M_max"]["valeur"], "kN·m"), None, True), (val(ex["M_min"]["valeur"], "kN·m"), None, True),
                       (val(max(abs(ex["V_max"]["valeur"]), abs(ex["V_min"]["valeur"])), "kN"), None, True)] + fin_)
        fin_ = [("tireté", S.mut)] if support else []
        lignes.append([("classeur", S.acc, True), "", "", ""] + [(val(dim.get(k), u), S.acc, True) for k, u in (("M_max", "kN·m"), ("M_min", "kN·m"), ("V_max", "kN"))] + fin_)
        note = ("« enveloppe » : le maximum de toutes les positions (la courbe en trait fort) ; « classeur » : ce que le classeur retient pour dimensionner "
                "(le tireté)." + (" Poutre support : chaque position est calculée pour chaque cas de la grue dans les huit orientations ; ses valeurs sont "
                                   "les maximums de ses colonnes ; « colonne » : celle du grand tableau du classeur qui donne M max." if support else ""))
        return cols, lignes, note

    def positions(self, d, fr, el):
        cols, lignes, note = self.lignes_positions(el)
        fixe = sum(w for _e, w, _a in cols)
        cols = [(e, w if w else max(40.0, fr.w - fixe), a) for e, w, a in cols]
        if not any(w == 0 for _e, w, _a in self.lignes_positions(el)[0]):       # pas de colonne élastique : la première prend le reste
            cols[0] = (cols[0][0], cols[0][1] + max(0.0, fr.w - fixe), cols[0][2])
        self.tableau(d, fr, cols, lignes, note)

    def hauteur_positions(self, el, largeur: float) -> float:
        """La hauteur que le tableau des positions prendra dans une colonne de ``largeur`` (dessiné sur un brouillon)."""
        brouillon = Doc(io.BytesIO(), A4L)
        brouillon.new_page(A4L)
        fr = Frame(M, 1.0e6, largeur, -1.0e9)
        self.positions(brouillon, fr, el)
        return fr.top - fr.min_y

    def page_plan(self, d) -> List[str]:
        """La dernière page : le plan de principe à une échelle normalisée, les pièces, les armatures choisies, ce que
        le calcul ne fixe pas, le cartouche."""
        S, r = self.S, self.C.res
        y = self.entete(d, f"PLAN DE PRINCIPE — {self.C.nom()}", f"{TITRE} · {self.C.sous_titre()}")
        top, bas = y - 6, M + CART_H + 10
        w_plan = (d.W - 2 * M - GAP) * 0.56
        fp = Frame(M, top, w_plan, bas)
        fd = Frame(M + w_plan + GAP, top, d.W - 2 * M - GAP - w_plan, bas)
        d.line(fd.x - GAP / 2, top, fd.x - GAP / 2, bas, S.rule, 0.45)
        titre, scene, legende = self.C.vues["plan"]
        self.titre(d, fp, "VUE EN PLAN — ni plan de coffrage, ni plan d'armatures")
        s_leg = 5.4
        lignes = RI.envelopper(legende, S.f_b, s_leg, self.ms(S.mut), fp.w)[:3]
        h = fp.room() - len(lignes) * s_leg * 1.25 - 8
        n, s = echelle_plan(scene, fp.w, h)
        fp.down(3)
        fp.down(h)
        peindre(d, scene, fp.x, fp.y, fp.w, h, echelle=s)
        fp.down(2)
        for l in lignes:
            fp.down(s_leg * 1.25)
            RI.dessiner(d, fp.x, fp.y, l, S.f_b, s_leg, S.mut, self.ms(S.mut))
        # à droite : les pièces, les armatures choisies, ce que le calcul ne fixe pas
        self.titre(d, fd, "PIÈCES")
        w = fd.w
        cols = [("PIÈCE", w - 34 - 186, "left"), ("NOMBRE", 34, "right"), ("DIMENSIONS", 186, "right")]
        self.tableau(d, fd, cols, [[(p["nom"], None, True), str(p["nombre"]), dimensions(p)] for p in r["pieces"]])
        self.titre(d, fd, "ARMATURES CHOISIES — ce que la note retient et que l'export transmet à RBAR")
        cols = [("ÉLÉMENT · POSTE", w - 116 - 54 - 46, "left"), ("ARMATURE", 116, "left"), ("CELLULES", 54, "right"), ("ÉTAT", 46, "right")]
        mots = {MO.OK: "OK", MO.WARN: "à vérifier", MO.NOK: "PAS OK"}
        lignes = []
        for l in self.C.controle["lignes"]:
            col = self.couleur(l["etat"]) if l["etat"] in mots else S.mut
            lignes.append([f"{l['element']} · {l['poste'].lower()}", (l["valeur"], None, True), (plage(l.get("cellules")), S.mut),
                           (mots.get(l["etat"], "prescription"), col, l["etat"] in (MO.WARN, MO.NOK))])
        self.tableau(d, fd, cols, lignes)
        manques = list(self.C.controle["manques"])
        if manques:
            self.titre(d, fd, "CE QUE LE CALCUL NE FIXE PAS — à compléter par le dessinateur")
            fd.down(1)
            for m in manques:
                self.rp(d, fd, "• " + RI.auto(m), size=self.s_txt * 0.96, col=S.mut, lead=1.26)
        warn = [Debord("plan de principe", f_.x, f_.overflow()) for f_ in (fp, fd) if f_.overflow() > 1.0]
        self.cartouche(d, f"1:{n}")
        return warn

    def cartouche(self, d, echelle: str):
        S, C_, doc = self.S, self.C, self.doc
        f, m = C_.f, C_.res["materiaux"]
        W_ = d.W
        x0, y0, w, h = M, M, W_ - 2 * M, CART_H
        d.box(x0, y0, w, h, stroke=S.ink, lw=0.9)
        ident = " · ".join(str(x) for x in (doc.get("projet"), doc.get("partie")) if x)
        pieux = next(p for p in C_.res["pieces"] if p["cle"] == "pieux")
        cases = [("FONDATION", C_.nom(), ident or "—", 0.22),
                 ("SCHÉMA", EN.COURTS_TYPE[f["type"]], f"{pieux['nombre']} pieux Ø{FG._g(pieux['D'])} cm", 0.17),
                 ("MATÉRIAUX", f"{m['beton']} · {m['acier']}", f"$σ'_b$ = {C.fr(m['sigma_b'], 2)} · $σ_{{adm}}$ = {C.fr(m['sigma_adm'], 2)} N/mm²", 0.23),
                 ("CHARGES · TOLÉRANCES", FG._resume_charges(f), FG._resume_tolerances(f), 0.18),
                 ("DATE · INDICE", str(doc.get("date", "")), "indice " + str(doc.get("indice", "")), 0.10),
                 ("ÉCHELLE", echelle, "plan · A4 paysage", 0.10)]
        x = x0
        for i, (lab, l1, l2, part) in enumerate(cases):
            cw = w * part
            if i:
                d.line(x, y0 + 2, x, y0 + h - 2, S.rule, 0.6)
            d.t(x + 6, y0 + h - 10, lab, S.f_v, 5.4, S.mut, track=1.2)
            self.rt(d, x + 6, y0 + h - 21, str(l1), 6.6, S.ink, font="Carlito-Bold", width=cw - 12)
            if l2:
                self.rt(d, x + 6, y0 + h - 32, str(l2), 6.2, S.ink, width=cw - 12)
            x += cw
        d.t(W_ - M, M - 9, f"page {d.n}", S.f_b, 6.2, S.mut, "right")


def plage(cellules: Optional[str]) -> str:
    """« G129, G130, G131, G132 » → « G129:G132 » : des cellules qui se suivent dans une colonne, en plage ; « — » sans cellule."""
    cs = [c.strip() for c in (cellules or "").split(",") if c.strip()]
    if not cs:
        return "—"
    m = [re.fullmatch(r"([A-Z]+)(\d+)", c) for c in cs]
    if len(cs) > 1 and all(m) and len({x.group(1) for x in m}) == 1 and [int(x.group(2)) for x in m] == list(range(int(m[0].group(2)), int(m[0].group(2)) + len(cs))):
        return f"{cs[0]}:{cs[-1]}"
    return ", ".join(cs)


def codes_ecarts(ecarts) -> str:
    """Les codes des écarts, dans l'ordre où ils viennent ; un code relevé plusieurs fois le dit (« E7 × 2 ») : le
    compte annoncé se retrouve dans la liste."""
    compte = {}
    for e in ecarts:
        compte[e["code"]] = compte.get(e["code"], 0) + 1
    return ", ".join(c if k == 1 else f"{c} × {k}" for c, k in compte.items())


def dimensions(p: Dict) -> str:
    """Les dimensions d'une pièce (``moteur`` : « pieces »), en clair."""
    g = FG._g
    if p["cle"] == "pieux":
        s = f"Ø {g(p['D'])} cm"
        if p.get("entraxe"):
            s += f" · entraxe {g(p['entraxe'])} cm"
        if p.get("capacite"):
            s += f" · capacité {g(p['capacite'])} {p.get('unite_capacite', '')}".rstrip()
        return s
    s = f"{g(p['b'])} × {g(p['h'])} cm"
    if p.get("longueur"):
        s += f" · longueur {g(p['longueur'])} cm"
    if p.get("portee"):
        s += f" · portée {g(p['portee'])} cm"
    return s


# ------------------------------------------------------------------ la mise en page d'une fondation
def _hauteurs(note: Note, unites, largeur: float) -> List[float]:
    """La hauteur de chaque bloc du calcul dans une colonne de ``largeur`` : il est dessiné sur un brouillon."""
    brouillon = Doc(io.BytesIO(), A4L)
    brouillon.new_page(A4L)
    out = []
    for _cle, _lie, fn in unites:
        fr = Frame(M, 1.0e6, largeur, -1.0e9)
        fn(brouillon, fr)
        out.append(fr.top - fr.min_y)
    return out


def repartir(hauteurs: List[float], lies: List[bool], h1: float, h2: float, n_cols: int = 2) -> Optional[List[List[List[int]]]]:
    """Les blocs répartis dans les colonnes, dans l'ordre : [page][colonne] → indices. ``h1`` : la hauteur d'une colonne
    de la première page ; ``h2`` : celle des suivantes. Un bloc « lié » ne reste pas en bas d'une colonne sans le
    suivant : il part avec lui (et avec toute la chaîne des blocs liés qui le suivent, jusqu'au premier qui ne l'est
    pas). None quand un bloc est plus haut qu'une colonne entière."""
    pages: List[List[List[int]]] = [[[]]]
    reste = h1
    for i, h in enumerate(hauteurs):
        besoin, j = h, i
        while lies[j] and j + 1 < len(hauteurs):
            j += 1
            besoin += hauteurs[j]
        if pages[-1][-1] and besoin > reste:
            if len(pages[-1]) < n_cols:
                pages[-1].append([])
            else:
                pages.append([[]])
            reste = h1 if len(pages) == 1 else h2
        if h > reste + 0.5:
            return None
        pages[-1][-1].append(i)
        reste -= h
    return pages


def mise_en_page(calc: Calcul, doc_meta: Dict) -> Tuple[float, str, List[List[List[int]]]]:
    """(k, nom de la taille, répartition) : le nombre de pages de calcul est celui de la taille NORMALE (à défaut, de
    la serrée) ; la taille retenue est la plus GRANDE qui tient dans ce nombre de pages. ValueError quand un bloc ne
    tient dans aucune colonne, même en taille serrée."""
    brouillon = Doc(io.BytesIO(), A4L)
    largeur = (A4L[0] - 2 * M - GAP) / 2.0

    def essai(k: float):
        note = Note(calc, doc_meta, k)
        h1, h2 = (top - BAS for top in note.hauts(brouillon))
        unites = note.unites()
        return repartir(_hauteurs(note, unites, largeur), [lie for _c, lie, _f in unites], h1, h2)

    reference = essai(NORMALE)
    if reference is None:
        serree = essai(SERREE)
        if serree is None:
            raise ValueError(f"La note de la fondation « {calc.nom()} » ne tient pas dans sa mise en page (un bloc du calcul est plus haut "
                             "qu'une colonne) : elle ne sort pas tronquée. À signaler au bureau ; l'écran garde tout le calcul.")
        return SERREE, "serrée", serree
    for k, nom in VARIANTES:
        if k <= NORMALE:
            break
        rep = essai(k)
        if rep is not None and len(rep) == len(reference):
            return k, nom, rep
    return NORMALE, "normale", reference


def poser(d: Doc, calc: Calcul, doc_meta: Dict, k: float, repartition) -> List[str]:
    """Les pages d'UNE fondation posées dans le document : le calcul, les diagrammes, le plan de principe. Rend les
    débordements (aucun, par construction, pour les pages de calcul)."""
    note = Note(calc, doc_meta, k)
    unites = note.unites()
    warn: List[str] = []
    for n, page in enumerate(repartition, 1):
        d.new_page(A4L)
        warn += note.page_calcul(d, n, len(repartition), [[unites[i][2] for i in col] for col in page])
    fixe = Note(calc, doc_meta, NORMALE)          # les pages à disposition fixe : toujours à la taille normale
    cles = [e["cle"] for e in calc.res["elements"]]
    for i in range(0, len(cles), 2):
        d.new_page(A4L)
        warn += fixe.page_diagrammes(d, cles[i:i + 2])
    d.new_page(A4L)
    warn += fixe.page_plan(d)
    return warn


def pages_de(calc: Calcul, repartition) -> int:
    return len(repartition) + (len(calc.res["elements"]) + 1) // 2 + 1


# ------------------------------------------------------------------ la note entière
def fondations_de(etat_ou_fondations) -> List[Dict]:
    """Les fondations d'une note : l'état du site ({"fondations": [...]}) ou une liste ; ValueError s'il n'y en a aucune."""
    liste = etat_ou_fondations.get("fondations") if isinstance(etat_ou_fondations, dict) else etat_ou_fondations
    if not isinstance(liste, list) or not liste:
        raise ValueError("La note n'a aucune fondation : rien à imprimer.")
    return list(liste)


def generer_note(etat_ou_fondations, infos: Optional[Dict] = None, chemin: Optional[str] = None) -> bytes:
    """La note (octets PDF) : la page de garde du site (une ligne par fondation : son schéma, son taux, ce qui gouverne,
    son statut, sa page), puis les pages de chaque fondation. ``infos`` : nom_projet, partie, date, indice. Une fondation
    hors domaine arrête la note avec le message du moteur — jamais un chiffre faux imprimé."""
    global derniers_avertissements, dernieres_variantes
    doc_meta = ndc_data.construire_doc(infos or {}, date_defaut=datetime.today().strftime("%d/%m/%Y"))
    doc_meta["titre"] = TITRE_NOTE
    calcs = []
    for i, f in enumerate(fondations_de(etat_ou_fondations)):
        try:
            calc = Calcul(f)
            calc.gouvernant()
            calcs.append(calc)
        except (ValueError, KeyError, TypeError) as err:
            nom = (f.get("nom") if isinstance(f, dict) else None) or f"n° {i + 1}"
            raise ValueError(f"La note n'est pas préparée : fondation « {nom} », {err}") from err
    dispositions = [mise_en_page(calc, doc_meta) for calc in calcs]
    S = Encre()
    a_effacer = None
    if chemin is None:
        fd, chemin = tempfile.mkstemp(suffix=".pdf", prefix="note_fondation_grue_")
        os.close(fd)
        a_effacer = chemin
    try:
        d = Doc(chemin, A4P, title=TITRE_NOTE)
        d.new_page(A4P)
        lignes = []
        for calc, (k, nom, rep) in zip(calcs, dispositions):
            r = calc.res
            v, b = calc.gouvernant()
            lignes.append(dict(titre=calc.nom(),
                               sous=f"{calc.sous_titre()} · η max {FG._pct(r['taux_max'])} · gouverne : {FG._nom_bloc(b).lower()}, {FG._court(v).lower()}",
                               etat=STATUTS[r["statut"]], couleur={MO.OK: S.ok, MO.WARN: S.att, MO.NOK: S.ko}[r["statut"]], pages=pages_de(calc, rep)))
        S.garde_lignes(d, doc_meta, lignes, legende=LEGENDE_GARDE)
        warn: List[str] = []
        for calc, (k, nom, rep) in zip(calcs, dispositions):
            warn += poser(d, calc, doc_meta, k, rep)
        d.save()
        if warn:
            raise ValueError("La note ne tient pas dans sa mise en page (" + " ; ".join(warn) + ") : elle ne sort pas tronquée. À signaler au "
                             "bureau ; l'écran garde tout le calcul.")
        derniers_avertissements = list(warn)
        dernieres_variantes = [nom for _k, nom, _rep in dispositions]
        with open(chemin, "rb") as fh:
            return fh.read()
    finally:
        if a_effacer:
            try:
                os.remove(a_effacer)
            except OSError:
                pass


derniers_avertissements: List[str] = []
dernieres_variantes: List[str] = []              # la taille du texte retenue pour chaque fondation de la dernière note


# ------------------------------------------------------------------ la plomberie : la note du site → le fichier
def nom_note(etat: Dict, infos: Optional[Dict]) -> str:
    """Le nom de la note : celui que la note du site porte ; à défaut, le nom de la fondation quand elle est seule, le
    titre du module sinon."""
    nom = _texte((infos or {}).get("nom"))
    if nom:
        return nom
    fs = etat.get("fondations") or []
    if len(fs) == 1 and isinstance(fs[0], dict):
        return _texte(fs[0].get("nom")) or TITRE_MODULE
    return TITRE_MODULE


def infos_doc(etat: Dict, infos: Optional[Dict]) -> Dict:
    """Les « infos » du générateur ({"nom_projet", "partie", "indice", "date"}) depuis celles de la note du site : la
    correspondance de la note de la poutre (api_web/note_poutre.py)."""
    infos = infos or {}
    nom, chantier = nom_note(etat, infos), _texte(infos.get("chantier"))
    out = {"nom_projet": chantier or nom, "partie": nom if chantier else TITRE_MODULE, "indice": _texte(infos.get("indice")) or "A"}
    if _texte(infos.get("date")):
        out["date"] = _texte(infos.get("date"))
    return out


def pdf(etat: Dict, infos: Optional[Dict] = None) -> bytes:
    """La note PDF (octets) de toute la note : la garde, puis les pages de chaque fondation, dans l'ordre de la liste."""
    return generer_note(etat, infos_doc(etat, infos))


def fichier(etat: Dict, infos: Optional[Dict] = None) -> Dict:
    """Le fichier à télécharger : {"nom", "type": "application/pdf", "base64"} (le routeur le contrôle)."""
    infos = dict(infos or {})
    octets = pdf(copy.deepcopy(etat), infos)
    return {"nom": C.nom_fichier(f"Note de calcul - {nom_note(etat, infos)}", "pdf"), "type": "application/pdf",
            "base64": base64.b64encode(octets).decode("ascii")}

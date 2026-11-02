# -*- coding: utf-8 -*-
"""Note de calcul de la platine d'about — DEUX pages A4 paysage, la même
charte que les deux autres assemblages de la famille (palette Encre,
bandeau statut, en-tête compact en trois lignes, trois colonnes, pied en
deux lignes) :

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

from ndc_pdf.kit import Doc
from ndc_pdf import data as ndc_data
from acier.formats import F
from acier.js import N, js_str
from acier.bibliotheques import PERSO
from acier.assemblages.note_kit import empaqueter, coefficients, MSB
from . import schemas, synthese
from .entrees import VISSERIE_DEFAUT
from .notation import LEGENDE_REFERENCES, ligne_alerte, ref_courte, NOTATIONS as NOTATIONS_COURTES
from ..plat_ame import note as _np
from ..doubles_cornieres.note import PlanPrincipe, TEXTE_MM

A4L = landscape(A4)
M = _np.M

TITRE = "Assemblage poutre–poutre — platine d'about de hauteur partielle"
VARIANTES = _np.VARIANTES


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
          + "    |    Platine " + js_str(R.h_p) + "×" + js_str(R.b_p) + "×" + js_str(R.t_p) + " " + u.nu_pl)
    l2 = ("Boulons " + R.boulon + " " + u.classe + " cat. " + u.cat + " · trous Ø" + F(R.d_0, 0)
          + (" surdim." if u.trou != "Normal" else "")
          + "    |    Groupe " + js_str(R.n_1) + " × 2 (p3 " + F(R.p_3, 0) + ", 1 plan)"
          + "    |    Soudure double a " + F(R.a_w, 1).replace(",0", ""))
    l3 = ("VEd " + F(u.V_Ed, 1) + " kN    |    NEd " + F(u.N_Ed, 1) + " kN"
          + ((" (" + str(u.nat_N) + ")") if N(u.N_Ed) > 0 else ""))
    return l1, l2, l3


def segments_hypotheses(R):
    """Les SEGMENTS du bloc HYPOTHÈSES (charte du 31/10/2026, §11) : modèle
    articulé, rupture de bloc de la platine (k), grugeage, nature de NEd,
    cordons et pleine résistance, coefficients — chaque segment écrit en
    entier, avec sa forme courte quand il en a une ; le kit
    (``note_kit.empaqueter``) les range en ≤ 3 lignes de ≤ 110 caractères.
    Retirés parce que déjà sur la page : la branche du groupe, Fv,Rd / Fb,Rd
    (table Boulons, formules) et Σleff (formule du tronçon en T)."""
    u = R.u
    signe = "≥" if R.hp_136 else "<"
    k = F(R.k_bloc, 1)
    seg = ["articulé : rotule à la face de l'âme porteuse (MSB P5 §2.2) ; groupe centré, VEd/n par boulon",
           ("platine : hp " + signe + " 1,36·p3 → rupture de bloc, k = " + k,
            "platine hp " + signe + " 1,36·p3 → k = " + k)]
    if R.cas_g > 0:
        seg.append(("grugeage : section réduite, bras tp + c (MSB P5 §4.2.4)", "grugeage : bras tp + c (MSB P5 §4.2.4)"))
    tying = bool(R.tying and N(u.N_Ed) > 0)
    if N(u.N_Ed) > 0:
        seg.append(("tying : force de chaînage seule, fu/γMu (MSB P5 §2.3)", "tying : chaînage seul, fu/γMu (MSB P5 §2.3)",
                    "tying : fu/γMu (MSB P5 §2.3)")
                   if tying else ("NEd ELU combiné avec VEd (tronçon en T §6.2.4)", "NEd ELU avec VEd (§6.2.4)"))
    plein = F(R.k_plein, 2) + "·tw = " + F(R.a_plein, 1) + " mm"
    seg.append(("cordons sur hp (§4.5.3.3), pleine résistance " + plein, "cordons hp, pleine résistance " + plein))
    suffixe = (" · γM3 " + F(N(u.g_M3), 2) if u.cat != "A" else "") + (" · γMu " + F(N(u.g_Mu), 2) if tying else "")
    g = " · γM2 " + F(N(u.g_M2), 2)
    seg.append(tuple(f + suffixe for f in coefficients("γM0 " + F(N(u.g_M0), 2) + " (ANB)" + g, N(u.g_M2), N(u.g_M2n),
                                                       tete_net="γM0 " + F(N(u.g_M0), 2) + g)))
    return seg


def lignes_hypotheses(R):
    """≤ 3 lignes de ≤ 110 caractères, « MSB P5 » une fois par ligne si possible (ValueError sinon : le test rougit)."""
    return empaqueter(segments_hypotheses(R), n_max=3, larg=110, une_fois=MSB)


class Note(_np.Note):
    """Page 1 : la mise en page de la famille (bandeau, colonnes, tableaux,
    formules, conclusion — classe du plat d'âme) avec le titre, la synthèse,
    les schémas, l'en-tête, les hypothèses et les tableaux de CET
    assemblage."""
    TITRE_NOTE = TITRE
    SYN = synthese
    SCH = schemas
    COL2 = ("boulons", "platine", "soudure")
    COL3 = ("portee", "porteuse")
    NORME = "(γM0 : ANB belge)"
    LEGENDE = LEGENDE_REFERENCES
    # elle héritait du ``ref_courte`` du plat d'âme par le repli ``REF or``,
    # retiré ; le sien est mot pour mot le même — elle le NOMME.
    REF = staticmethod(ref_courte)
    NOTATIONS = NOTATIONS_COURTES

    def entete(self, R):
        return lignes_entete(R)

    def hypotheses(self, R):
        return lignes_hypotheses(R)

    def alerte(self, a, R):
        return ligne_alerte(a, R)


# ------------------------------------------------- page 2 : plan de principe
class PlanPrincipePA(PlanPrincipe):
    """Le plan de principe de la platine d'about : mêmes échelles
    normalisées, mêmes dispositions et même cartouche que les deux autres
    assemblages — seules les VUES et les cases du cartouche sont propres à
    cet assemblage."""

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
               + " – " + js_str(R.n_1) + " × 2 (p3 " + F(R.p_3, 0) + ") · soudure double a "
               + F(R.a_w, 1).replace(",0", ""))
        vis = "par boulon : " + (str(doc.get("visserie") or "").strip() or VISSERIE_DEFAUT)
        # Le rédacteur saisi (id_red) est porté au cartouche — un champ saisi et jamais lu était
        # un défaut (charte du 31/10/2026, §6 et amendement 3) ; un champ vide n'est pas imprimé.
        ident = " · ".join(str(x) for x in (doc.get("projet"), doc.get("partie"), u.id_red) if x)
        cases = [("ASSEMBLAGE", "Poutre–poutre – platine d'about", ident or "—", 0.185),
                 ("POUTRES", "P : " + _prof(R, "P") + " " + u.nu_P, "S : " + _prof(R, "S") + " " + u.nu_S, 0.165),
                 ("PLATINE", js_str(R.h_p) + "×" + js_str(R.b_p) + "×" + js_str(R.t_p) + " " + u.nu_pl,
                  "zp " + F(N(u.z_C), 0) + " mm · zt " + F(R.zt_pl, 0) + " mm", 0.165),
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
        fd, chemin = tempfile.mkstemp(suffix=".pdf", prefix="note_platine_about_")
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
    pp = PlanPrincipePA(R, doc_meta)
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

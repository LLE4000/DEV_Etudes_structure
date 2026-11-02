# -*- coding: utf-8 -*-
"""Note de calcul des doubles cornières sur poteau — DEUX pages A4 paysage,
par SOUS-CLASSEMENT, pas par recopie (`POUTRE_POTEAU.md` §10.8) :

- la page 1 est celle de la famille : ``plat_ame.note.Note`` a été écrite
  pour cela (attributs de classe TITRE_NOTE, SYN, SCH, COL2, COL3, NORME,
  LEGENDE, NOTATIONS, REF et méthodes ``entete``, ``hypotheses``,
  ``alerte``) ; ``colonne_dessins`` empile déjà ÉLÉVATION + VUE EN PLAN (la
  coupe A-A ne va qu'en page 2). Seuls le titre, les tables des colonnes 2
  et 3, l'en-tête compact et le bloc HYPOTHÈSES sont propres à ce module ;
- la page 2 « PLAN DE PRINCIPE » est celle du module doubles cornières
  poutre–poutre (``PlanPrincipe`` : échelles normalisées, dispositions,
  cartouche), avec les trois vues de CET assemblage et un cartouche dont la
  case des profilés devient « POTEAU · POUTRE » et celle de l'assemblage
  porte le CAS D'APPUI.
"""
import os
import tempfile
from datetime import datetime

from reportlab.lib.pagesizes import A4, landscape

from ndc_pdf.kit import Doc
from ndc_pdf import data as ndc_data
from acier.formats import F
from acier.js import N, js_str
from . import schemas, synthese
from .entrees import VISSERIE_DEFAUT
from acier.assemblages.note_kit import empaqueter, coefficients, MSB
from .notation import ligne_alerte, ref_courte, LEGENDE_REFERENCES, NOTATIONS as NOTATIONS_COURTES
from ...poutre_poutre.plat_ame.note import Note as NoteFamille, _prof, M
from ...poutre_poutre.doubles_cornieres.note import PlanPrincipe, TEXTE_MM, CART_H, _fmt_echelle

A4L = landscape(A4)

TITRE = "Assemblage poutre–poteau — doubles cornières d'âme"
VARIANTES = ("toutes", "eta")


def lignes_entete(R):
    """L'en-tête compact : objets, attaches, efforts — pas de géométrie
    détaillée (elle est cotée sur le plan de principe)."""
    u = R.u
    grugee = bool(N(u.d_nt) or N(u.d_nb))
    l1 = ("Poteau " + _prof(R, "P") + " " + u.nu_P + " · appui : " + str(R.appui).lower()
          + "    |    Poutre portée " + _prof(R, "S") + " " + u.nu_S + (" · grugée" if grugee else "")
          + (" · semelles recoupées" if R.ame and N(u.rec_S) > 0 else "")
          + "    |    Cornières 2 × " + R.corn_txt + " " + u.nu_C + " · hc " + js_str(R.L_C))
    l2 = ("Boulons " + R.boulon + " " + u.classe + " cat. " + u.cat + " · trous Ø" + F(R.d_0, 0)
          + (" surdim." if u.trou != "Normal" else "")
          + "    |    Groupe S " + js_str(R.n1_S) + " × " + js_str(R.n2_S) + " (2 plans) · "
          + str(u.fix_S).lower()
          + "    |    Groupe P " + js_str(R.n1_P) + " × " + js_str(R.n2_P) + " par cornière · "
          + str(u.fix_P).lower() + " · p3 " + F(R.p_3, 1))
    l3 = ("VEd " + F(u.V_Ed, 1) + " kN    |    NEd " + F(u.N_Ed, 1) + " kN"
          + ((" (" + str(u.nat_N) + ")") if u.N_Ed > 0 else "")
          + "    |    HEd " + F(u.H_Ed, 1) + " kN    |    MEd " + F(u.M_Ed, 2) + " kNm")
    return l1, l2, l3


def segments_hypotheses(R):
    """Les SEGMENTS du bloc HYPOTHÈSES (charte du 31/10/2026, §11) : modèle et
    excentricité, CAS D'APPUI, gauge, grugeage, tying, moment apporté au
    poteau, fixations, coefficients — chaque segment écrit en entier, avec ses
    formes courtes ; le kit (``note_kit.empaqueter``) les range en ≤ 3 lignes
    de ≤ 110 caractères.

    Gardés parce qu'ils ne vivent nulle part ailleurs sur la page : z, le
    moment repris, le cas d'appui, le gauge p3 et le moment VEd·e apporté au
    POTEAU (que le projeteur doit reporter à son calcul d'élément — §6.3)."""
    u = R.u
    face = "l'âme" if R.ame else "l'aile"
    seg = [("articulé : rotule à la face de " + face + " du poteau (" + MSB + "§4.2.1.1)",
            "articulé : rotule à la face de " + face + " (" + MSB + "§4.2.1.1)",
            "articulé : rotule à la face de " + face + " du poteau")]
    # le CAS D'APPUI vient tout de suite après : le kit raccourcit du dernier
    # segment vers le premier, et c'est lui qu'il faut garder en entier
    if R.ame:
        # la flexion hors plan de l'âme est AUSSI dite par l'alerte « ame_hp »
        # imprimée sous le bandeau : la forme courte peut donc la lâcher
        seg.append(("appui dans l'âme : cisaillement local un seul côté, flexion hors plan NON vérifiée",
                    "appui dans l'âme : cisaillement local un seul côté",
                    "appui dans l'âme (cisaillement local)",
                    "appui dans l'âme du poteau"))
    else:
        seg.append(("appui sur l'aile, attache dans le plan de l'âme du poteau",
                    "appui sur l'aile (plan de l'âme du poteau)",
                    "appui sur l'aile"))
    seg.append(("z = " + F(R.zeff, 1) + " mm, "
                + ("le groupe S reprend" if R.bolt_S else "les cordons B reprennent")
                + " VEd·z + |MEd|", "z = " + F(R.zeff, 1) + " mm"))
    if R.bolt_P:
        seg.append(("groupe P centré, k " + F(N(u.k_rot), 2) + " sur Fv,Rd (" + MSB + "§4.2.1.2)",
                    "groupe P centré, k " + F(N(u.k_rot), 2) + " sur Fv,Rd",
                    "groupe P centré (k " + F(N(u.k_rot), 2) + ")"))
        if R.ame:
            seg.append(("gauge p3 " + F(R.p_3, 1) + " mm, aile A dans la partie droite de l'âme",
                        "gauge p3 " + F(R.p_3, 1) + " mm"))
        else:
            seg.append(("gauge p3 " + F(R.p_3, 1) + " mm ≥ twP + 2·rP + dw = "
                        + F(R.p3_congeP, 1) + " mm",
                        "gauge p3 " + F(R.p_3, 1) + " ≥ " + F(R.p3_congeP, 1) + " mm",
                        "gauge p3 " + F(R.p_3, 1) + " mm"))
    if not R.bolt_P or not R.bolt_S:
        seg.append(("ailes " + ("A et B" if not (R.bolt_P or R.bolt_S) else
                                ("A" if not R.bolt_P else "B"))
                    + " soudées, cordons pleine hauteur (§4.5.3.3)",
                    "ailes " + ("A et B" if not (R.bolt_P or R.bolt_S) else
                                ("A" if not R.bolt_P else "B")) + " soudées (§4.5.3.3)"))
    if R.cas_g > 0:
        seg.append(("grugeage : flexion de la section réduite (" + MSB + "§4.2.4)",
                    "grugeage : section réduite (" + MSB + "§4.2.4)"))
    if R.tying:
        seg.append(("tying : force de chaînage seule, fu/γMu (" + MSB + "§2.3)",
                    "tying : NEd seul, fu/γMu (" + MSB + "§2.3)",
                    "tying : NEd seul, γMu " + F(N(u.g_Mu), 2)))
    seg.append(("au poteau : VEd·e = " + F(R.M_col, 1) + " kNm à reporter au calcul d'élément",
                "au poteau VEd·e = " + F(R.M_col, 1) + " kNm",
                "au poteau VEd·e " + F(R.M_col, 1) + " kNm"))
    suffixe = " · γM3 " + F(N(u.g_M3), 2) if u.cat != "A" else ""
    g = "γM0 " + F(N(u.g_M0), 2) + " (ANB) · γM2 " + F(N(u.g_M2), 2)
    seg.append(tuple(f + suffixe for f in coefficients(g, N(u.g_M2), N(u.g_M2n))))
    return seg


def lignes_hypotheses(R):
    """≤ 3 lignes de ≤ 110 caractères, « MSB P5 » une fois par ligne si possible
    (ValueError sinon : le test rougit)."""
    return empaqueter(segments_hypotheses(R), n_max=3, larg=110, une_fois=MSB)


# ------------------------------------------------------------- page 1
class Note(NoteFamille):
    """La page 1 de la famille, avec les tables, l'en-tête et les hypothèses
    de CET assemblage. Les tables vides ne s'impriment pas (``table_simple``
    sort si ``lignes`` est vide) : les cordons n'y sont que si une fixation
    est soudée, le groupe « Poteau porteur » que si les ailes A sont
    boulonnées ou l'appui dans l'âme."""
    TITRE_NOTE = TITRE
    SYN = synthese
    SCH = schemas
    COL2 = ("boulons", "cornieres")
    COL3 = ("portee", "poteau", "cordons")
    NORME = "(γM0 : ANB belge)"
    LEGENDE = LEGENDE_REFERENCES
    NOTATIONS = NOTATIONS_COURTES
    REF = staticmethod(ref_courte)

    def entete(self, R):
        return lignes_entete(R)

    def hypotheses(self, R):
        return lignes_hypotheses(R)

    def alerte(self, a, R):
        return ligne_alerte(a, R)


# ------------------------------------------------- page 2 : plan de principe
class PlanPrincipePP(PlanPrincipe):
    """Le plan de principe des doubles cornières sur poteau : mêmes échelles
    normalisées, mêmes dispositions et même cartouche que la famille —
    seules les VUES et les cases du cartouche sont propres à cet assemblage.
    La case « POUTRES » devient « POTEAU · POUTRE » et la case
    « ASSEMBLAGE » porte le cas d'appui."""

    def __init__(self, R, doc_meta):
        super().__init__(R, doc_meta)
        self.titre = TITRE

    def _vues(self, dnm):
        f = TEXTE_MM * dnm
        R = self.R
        return [("ÉLÉVATION", schemas.elevation(R, schemas.options_fabrication(schemas.EXCLURE_ELEVATION, f))),
                ("VUE EN PLAN", schemas.plan(R, schemas.options_fabrication(schemas.EXCLURE_PLAN, f))),
                ("COUPE A-A", schemas.coupe_aa(R, schemas.options_fabrication(schemas.EXCLURE_COUPE, f)))]

    def _cartouche(self, d, dnm):
        S, R, doc = self.S, self.R, self.doc
        u = R.u
        W = d.W
        x0, y0, w, h = M, M, W - 2 * M, CART_H
        d.box(x0, y0, w, h, stroke=S.ink, lw=0.9)
        fix = []
        if R.bolt_S or R.bolt_P:
            fix.append("Boulons " + R.boulon + " " + u.classe + " – trous Ø" + F(R.d_0, 0))
        if R.bolt_S:
            fix.append("S : " + js_str(R.n1_S) + " × " + js_str(R.n2_S))
        else:
            fix.append("S soudé a " + F(N(u.a_S), 0))
        if R.bolt_P:
            fix.append("P : 2 × (" + js_str(R.n1_P) + " × " + js_str(R.n2_P) + ")")
        else:
            fix.append("P soudé a " + F(N(u.a_P), 0))
        vis = (("par boulon : " + (str(doc.get("visserie") or "").strip() or VISSERIE_DEFAUT))
               if (R.bolt_S or R.bolt_P) else "cornières soudées des deux côtés")
        ident = " · ".join(str(x) for x in (doc.get("projet"), doc.get("partie")) if x)
        cases = [("ASSEMBLAGE", "Poutre–poteau – doubles cornières",
                  "appui : " + str(R.appui).lower() + (" · " + ident if ident else ""), 0.225),
                 ("POTEAU · POUTRE", "P : " + _prof(R, "P") + " " + u.nu_P,
                  "S : " + _prof(R, "S") + " " + u.nu_S, 0.165),
                 ("CORNIÈRES", "2 × " + R.corn_txt + " " + u.nu_C,
                  "hc " + F(R.L_C, 0) + " · zc " + F(N(u.z_C), 0) + " mm"
                  + (" · recoupe " + F(N(u.rec_S), 0) if R.ame and N(u.rec_S) > 0 else ""), 0.155),
                 ("FIXATIONS", " – ".join(fix), vis, 0.245),
                 ("DATE · INDICE", str(doc.get("date", "")), "indice " + str(doc.get("indice", "")), 0.095),
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
        fd, chemin = tempfile.mkstemp(suffix=".pdf", prefix="note_doubles_cornieres_poteau_")
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
    pp = PlanPrincipePP(R, doc_meta)
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

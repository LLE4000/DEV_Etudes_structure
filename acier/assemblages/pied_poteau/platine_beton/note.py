# -*- coding: utf-8 -*-
"""Note de calcul du pied de poteau — DEUX pages A4 paysage, la charte de la
famille (palette Encre, bandeau statut, en-tête compact en trois lignes,
trois colonnes, pied en deux lignes) :

- page 1 : bandeau (statut, η max, dimensionnante, taux par élément,
  alertes), en-tête (poteau · platine · massif ; tiges · mortier · cordons ;
  efforts), colonne 1 = vue en plan avec la RÉPARTITION, coupe A-A, domaine
  (Mx, My) et hypothèses, colonnes 2 et 3 = tableaux par élément (Ed / Rd /
  η / réf.), pince de la platine, formules des vérifications essentielles,
  conclusion ;
- page 2 : « PLAN DE PRINCIPE » — coupe A-A, vue en plan et coupe B-B à la
  même échelle normalisée (PlanPrincipe du module doubles cornières), avec
  le cartouche de CET assemblage.
"""
import os
import tempfile
from datetime import datetime

from reportlab.lib.pagesizes import A4, landscape

from ndc_pdf.kit import Doc, mix
from ndc_pdf import data as ndc_data
from acier.formats import F, pct
from acier.js import N, js_str
from . import schemas, synthese
from .entrees import VISSERIE_DEFAUT, OUI, ancrage_txt
from .notation import LEGENDE_REFERENCES, ligne_alerte, ref_courte
from ...poutre_poutre.plat_ame import note as _np
from ...poutre_poutre.doubles_cornieres.note import PlanPrincipe, TEXTE_MM, _fmt_echelle, CART_H
from ...poutre_poutre.doubles_cornieres.rapport import peindre

A4L = landscape(A4)
M = _np.M
BAS = _np.BAS

TITRE = "Pied de poteau — platine sur béton (N + Mx + My)"
NOTATIONS_COURTES = ("Notations : bp, lp, tp : platine ; wx, wy : entraxes des files et des rangées ; hef, dh : "
                     "ancrage ; tg : mortier ; bc, lc, hc : massif ; c : bande efficace (EC3 §6.2.5) ; fjd : "
                     "résistance d'appui ; C, pmax : résultante et pression de la répartition ; T : effort d'une tige.")
VARIANTES = _np.VARIANTES


def lignes_entete(R):
    """L'en-tête compact : objets, attaches, efforts — la géométrie détaillée
    est cotée sur le plan de principe."""
    u = R.u
    l1 = ("Poteau " + R.nom_c + " " + str(u.nu_c)
          + "    |    Platine " + js_str(N(u.bp_u)) + "×" + js_str(N(u.lp_u)) + "×" + js_str(N(u.tp_u)) + " " + str(u.nu_pl)
          + "    |    Massif " + js_str(N(u.bc_u)) + "×" + js_str(N(u.lc_u)) + "×" + js_str(N(u.hc_u)) + " "
          + str(u.beton) + (" fissuré" if u.fiss == OUI else " non fissuré"))
    l2 = ("Tiges " + js_str(R.n_t) + " × " + str(u.ancr) + " " + str(u.classe) + " (" + js_str(int(u.n_ax)) + " files × "
          + js_str(int(u.n_ay)) + " rangées, " + js_str(N(u.w_x)) + " × " + js_str(N(u.w_y)) + ") · "
          + ancrage_txt(u, True) + " · hef " + F(u.hef, 0)
          + "    |    Mortier " + F(u.t_g, 0) + " mm, " + F(u.f_g, 0) + " MPa"
          + "    |    Cordons a " + F(u.a_w, 1).replace(",0", ""))
    l3 = ("NEd " + F(u.N_Ed, 1) + " kN    |    Mx,Ed " + F(u.Mx_Ed, 1) + " kNm    |    My,Ed " + F(u.My_Ed, 1)
          + " kNm    |    Vx,Ed " + F(u.Vx_Ed, 1) + " kN    |    Vy,Ed " + F(u.Vy_Ed, 1) + " kN"
          + ("    |    TEd " + F(u.T_Ed, 2) + " kNm" if N(u.T_Ed) else ""))
    return l1, l2, l3


def lignes_hypotheses(R):
    u = R.u
    h = [("Périmètre EN 1993-1-8 : platine, tiges, poteau, cordons, appui fjd ; béton d'ancrage EN 1992-4 en information."
          if R.perim_ec3 else "Périmètre complet : EN 1993-1-8 et EN 1992-4 (béton d'ancrage dans le verdict)."),
         "Aire efficace EC3 §6.2.5 : fjd = " + F(R.fjd, 1) + " MPa (βj = 2/3, √(Ac1/Ac0) = " + F(R.ratio_c, 2)
         + "), c = " + F(R.c, 0) + " mm, Aeff = " + F(R.A_eff / 100, 0) + " cm².",
         "Répartition élastique (platine rigide, lit sans traction, tiges tendues) : C = " + F(R.C, 0) + " kN, pmax = "
         + F(R.pmax, 1) + " MPa, ΣT = " + F(R.T_tot, 0) + " kN.",
         "Verdict au niveau 1 (Tab. 6.7, interaction linéaire : " + F(R.eta1 * 100, 0) + " %) ; niveau 2 exact : "
         + F(R.eta2 * 100, 0) + " %, information.",
         "Tiges coulées en place (EC2-4, ψM,N = 1) ; γM0 " + F(N(u.g_M0), 2) + " (ANB) · γM2 " + F(N(u.g_M2), 2)
         + " · γc " + F(N(u.g_c), 2) + " (αcc " + F(N(u.a_cc), 2) + ") · γMc " + F(N(u.g_Mc), 2) + "."]
    if R.rig:
        h.append("Rigidité Sj,ini = " + F(R.Sj / 1e3, 0) + " MNm/rad, seuil " + F(R.Sj_seuil / 1e3, 0) + " → pied "
                 + ("RIGIDE" if R.rig["rigide"] else "SEMI-RIGIDE") + " (information).")
    return h


class Note(_np.Note):
    """Page 1 : la mise en page de la famille (bandeau, colonnes, tableaux,
    formules, conclusion — classe du plat d'âme) avec le titre, la synthèse,
    les schémas, l'en-tête, les hypothèses, les tableaux et les références
    courtes de CET assemblage."""
    TITRE_NOTE = TITRE
    SYN = synthese
    SCH = schemas
    COL2 = ("pied", "platine", "beton")
    COL3 = ("ancrages", "tiges", "poteau", "soudure")
    NORME = "(γM0 : ANB belge ; αcc = 0,85)"
    LEGENDE = LEGENDE_REFERENCES
    NOTATIONS = NOTATIONS_COURTES
    REF = staticmethod(ref_courte)

    def entete(self, R):
        return lignes_entete(R)

    def hypotheses(self, R):
        return lignes_hypotheses(R)

    def alerte(self, a, R):
        return ligne_alerte(a, R)

    def table_simple(self, d, fr, cle, largeurs):
        """Comme la famille, avec le titre qui dit « information » pour les
        ancrages hors périmètre, et le calcul à la main sous la platine."""
        R = self.R
        ls = self.SYN.lignes(R, cle)
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
        self.table(d, fr, self.SYN.titre_table(cle, R).upper(), cols, lignes)
        if cle == "platine":
            self.table_platine_main(d, fr)

    def table_platine_main(self, d, fr):
        """Le calcul de la platine à la main : le moment pris de chaque côté,
        l'épaisseur qu'il demande, l'épaisseur requise."""
        R = self.R
        P = R.plaque
        u = R.u
        if not R.ck["tpq"].active:
            return
        T = P.get("tendu")

        def mm(v):
            return "∞" if v == float("inf") else F(v, 1)
        w = fr.w
        cols = [("Côté", 62, "left"), ("Le moment pris, sa résistance, l'épaisseur qu'il demande", w - 62, "left")]
        lignes = [[("lab", "Console comprimée"),
                   ("lab", "p max " + F(P["p_max"], 1) + " MPa sur c = " + F(P["c"], 0) + " mm (" + P["ligne"]
                    + ") : mEd = ∫p·s ds = " + F(P["m_Ed"], 1) + " ≤ tp²·fy/(6·γM0) = " + F(P["m_Rd_el"], 1)
                    + " kNm/m → tc = " + mm(P["t_c"]) + " mm ; appui tN = " + mm(P["t_N"]) + " ; niveau 1 tM = "
                    + mm(P["t_M"]) + " mm")]]
        if T and T["T"] > 1e-9:
            lignes.append([("lab", "Tronçon en T"),
                           ("lab", "rangée " + T["cle"] + " : M = T·m = " + F(T["T"], 1) + " × " + F(T["m"], 1) + " = "
                            + F(T["M"], 2) + " kNm ; Σleff,1 = " + F(T["leff1"], 0) + " mm, Mpl,1 = " + F(T["Mpl1"], 2)
                            + " kNm, " + ("avec" if T["levier"] else "sans") + " effet de levier → FT,Rd = "
                            + F(T["FT_Rd"], 0) + " kN → tT = " + mm(P["t_T"]) + " mm")])
        lignes.append([("lab", "Épaisseur requise", "gov"),
                       ("lab", "max(tc ; tN ; tM ; tT) = " + mm(P["t_req"]) + " mm pour tp = " + F(u.tp_u, 0) + " mm ("
                        + pct(R.ck["tpq"].eta, 0) + ")", "gov")])
        NV = P["navier"]
        if NV["applicable"]:
            lignes.append([("lab", "Repère Navier"),
                           ("lab", "platine entière : σ max = " + F(NV["s_max"], 2) + " MPa, L = " + F(NV["L"], 0)
                            + " mm, m = σ·L²/2 = " + F(NV["m"], 1) + " kNm/m → t = " + F(NV["t"], 1) + " mm")])
        self.table(d, fr, "ÉPAISSEUR DE LA PLATINE — LE CALCUL À LA MAIN", cols, lignes)

    def colonne_dessins(self, d, fr):
        """Colonne 1 : la vue en plan avec la répartition (le dessin qui
        explique le résultat), la coupe A-A, le domaine (Mx, My), puis les
        hypothèses."""
        S, R = self.S, self.R
        hyp = self.hypotheses(R)
        lignes = []
        for h in hyp:
            lignes.extend(d.wrap(h, S.f_b, 6.2, fr.w))
        h_hyp = 13 + len(lignes) * 6.2 * 1.3 + 4
        dispo = fr.room() - h_hyp - 3 * 13 - 9
        hp = dispo * 0.46
        he = dispo * 0.28
        hd = dispo - hp - he
        opt = schemas.options_rapport()
        self.titre(d, fr, "VUE EN PLAN — RÉPARTITION (mm)")
        fr.down(hp)
        peindre(d, schemas.plan(R, opt), fr.x, fr.y, fr.w, hp - 2)
        self.titre(d, fr, "COUPE A-A")
        fr.down(he)
        peindre(d, schemas.coupe_aa(R, opt), fr.x, fr.y, fr.w, he - 2)
        self.titre(d, fr, "DOMAINE (Mx, My) À N = " + F(R.u.N_Ed, 0) + " kN")
        fr.down(hd)
        peindre(d, schemas.domaine(R), fr.x, fr.y, fr.w, hd - 2)
        self.titre(d, fr, "HYPOTHÈSES")
        fr.down(3)
        for l in lignes:
            fr.down(6.2 * 1.3)
            d.t(fr.x, fr.y, l, S.f_b, 6.2, S.mut)

    def conclusion(self, d, x, w):
        S, R = self.S, self.R
        statut = "PIED DE POTEAU VÉRIFIÉ" if R.verified else "PIED DE POTEAU NON VÉRIFIÉ"
        if R.perim_ec3:
            txt = (statut + " à l'ELU selon EN 1993-1-8 §6.2.5 et §6.2.8 " + self.NORME + " — le béton d'ancrage "
                   "(EN 1992-4) est donné en information et ne fait pas partie du verdict"
                   + (" ; sous réserve des points signalés." if R.verified and (R.reserve or R.alerts) else "."))
        else:
            txt = (statut + " à l'ELU selon EN 1993-1-8 §6.2.8, EN 1992-1-1 §6.7 et EN 1992-4 " + self.NORME
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


# ------------------------------------------------- page 2 : plan de principe
class PlanPrincipePB(PlanPrincipe):
    """Le plan de principe du pied : mêmes échelles normalisées, mêmes
    dispositions et même cartouche que les assemblages poutre–poutre — seules
    les VUES (coupe A-A, vue en plan, coupe B-B) et les cases du cartouche
    sont propres à cet assemblage."""

    def __init__(self, R, doc_meta):
        super().__init__(R, doc_meta)
        self.titre = TITRE

    def _vues(self, dnm):
        f = TEXTE_MM * dnm
        R = self.R
        return [("COUPE A-A", schemas.coupe_aa(R, schemas.options_fabrication(schemas.EXCLURE_AA, f))),
                ("VUE EN PLAN", schemas.plan(R, schemas.options_fabrication(schemas.EXCLURE_PLAN, f))),
                ("COUPE B-B", schemas.coupe_bb(R, schemas.options_fabrication(schemas.EXCLURE_BB, f)))]

    def _cartouche(self, d, dnm):
        S, R, doc = self.S, self.R, self.doc
        u = R.u
        W = d.W
        x0, y0, w, h = M, M, W - 2 * M, CART_H
        d.box(x0, y0, w, h, stroke=S.ink, lw=0.9)
        fix = ("Tiges " + js_str(R.n_t) + " × " + str(u.ancr) + " " + str(u.classe) + " – trous Ø" + F(R.d0, 0)
               + " – " + js_str(int(u.n_ax)) + " × " + js_str(int(u.n_ay)) + " (" + js_str(N(u.w_x)) + " × "
               + js_str(N(u.w_y)) + ") · " + ancrage_txt(u, True) + " · hef " + F(u.hef, 0))
        vis = "par tige : " + (str(doc.get("visserie") or "").strip() or VISSERIE_DEFAUT)
        ident = " · ".join(str(x) for x in (doc.get("projet"), doc.get("partie")) if x)
        cases = [("ASSEMBLAGE", "Pied de poteau – platine sur béton", ident or "—", 0.185),
                 ("POTEAU · MASSIF", R.nom_c + " " + str(u.nu_c),
                  "massif " + js_str(N(u.bc_u)) + "×" + js_str(N(u.lc_u)) + "×" + js_str(N(u.hc_u)) + " " + str(u.beton), 0.165),
                 ("PLATINE", js_str(N(u.bp_u)) + "×" + js_str(N(u.lp_u)) + "×" + js_str(N(u.tp_u)) + " " + str(u.nu_pl),
                  "mortier " + F(u.t_g, 0) + " mm · cordons a " + F(u.a_w, 1).replace(",0", ""), 0.165),
                 ("TIGES D'ANCRAGE", fix, vis, 0.27),
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
    """La note (bytes) : page 1 de calcul, page 2 « PLAN DE PRINCIPE » — celle
    du mode de ``R`` (calcul manuel : ``manuel.note``)."""
    global derniers_avertissements, derniere_variante, derniere_echelle, derniere_disposition
    if getattr(R, "mode", "") == "manuel":
        from .manuel import note as _nm
        octets = _nm.generer_note(R, infos, chemin)
        derniers_avertissements, derniere_variante = _nm.derniers_avertissements, _nm.derniere_variante
        derniere_echelle, derniere_disposition = _nm.derniere_echelle, _nm.derniere_disposition
        return octets
    infos = infos or {}
    doc_meta = ndc_data.construire_doc(infos, date_defaut=datetime.today().strftime("%d/%m/%Y"))
    doc_meta["titre"] = "Note de calcul"
    doc_meta["visserie"] = infos.get("visserie", "")
    if chemin is None:
        fd, chemin = tempfile.mkstemp(suffix=".pdf", prefix="note_pied_poteau_")
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
    pp = PlanPrincipePB(R, doc_meta)
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

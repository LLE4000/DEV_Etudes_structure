# -*- coding: utf-8 -*-
"""Note de calcul du CALCUL MANUEL de la platine — DEUX pages A4 paysage, la
charte de la famille (classe de mise en page du plat d'âme) :

- page 1 : bandeau (statut, épaisseur requise, taux par élément, alertes),
  en-tête (platine · éléments soudés · massif ; boulons · mortier · cordons ;
  efforts), colonne 1 = vue en plan (répartition, bras de levier, bande
  comprimée), coupe A-A, hypothèses ; colonne 2 = platine (épaisseur requise
  et ses quatre critères) suivie du CALCUL À LA MAIN (boulon par boulon,
  section par section, bande comprimée, trous, épaisseur requise), trous ;
  colonne 3 = tiges et appui en information, pinces, formules, conclusion ;
- page 2 : « PLAN DE PRINCIPE » — coupe A-A, vue en plan avec la chaîne des
  perçages depuis les bords, coupe B-B, à la même échelle normalisée, avec le
  cartouche de CET assemblage.
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
from . import elements as EL
from ..entrees import VISSERIE_DEFAUT, ancrage_txt
from ..notation import LEGENDE_REFERENCES, ligne_alerte, ref_courte
from ....poutre_poutre.plat_ame import note as _np
from ....poutre_poutre.doubles_cornieres.note import PlanPrincipe, TEXTE_MM, _fmt_echelle, CART_H
from ....poutre_poutre.doubles_cornieres.rapport import peindre

A4L = landscape(A4)
M = _np.M
BAS = _np.BAS

TITRE = "Platine sur béton — calcul manuel de la platine"
NOTATIONS_COURTES = ("Notations : bp, lp, tp : platine ; (x ; y) : repère au centre de la platine ; m : bras de levier "
                     "jusqu'à l'appui, cordon déduit ; leff : longueur efficace du tronçon en T ; b : largeur de la section "
                     "au nu (L + 2·m) ; c : bande efficace (EC3 §6.2.5) ; fjd : résistance d'appui ; C, pmax : résultante et "
                     "pression ; T : effort d'un boulon ; mEd : moment de la bande comprimée.")
# toutes : six boulons et quatre sections, formules complètes ; eta : trois boulons, deux sections, formules
# réduites à la ligne du taux ; mini : le boulon et la section les plus sollicités seulement
VARIANTES = ("toutes", "eta", "mini")
MAX_LIGNES = {"toutes": (6, 4), "eta": (3, 2), "mini": (1, 1)}


def lignes_entete(R):
    u = R.u
    l1 = ("Platine " + js_str(N(u.bp_u)) + "×" + js_str(N(u.lp_u)) + "×" + js_str(N(u.tp_u)) + " " + str(u.nu_pl)
          + "    |    " + " + ".join(EL.nom(e) + " en (" + F(e["x"], 0) + " ; " + F(e["y"], 0) + ")" + (" tourné" if e["rot"] else "")
                                     for e in R.elems)
          + ("    |    " + js_str(len(R.raid)) + " raidisseur(s)" if R.raid else "")
          + "    |    Massif " + js_str(N(u.bc_u)) + "×" + js_str(N(u.lc_u)) + "×" + js_str(N(u.hc_u)) + " " + str(u.beton))
    l2 = ("Boulons " + js_str(R.n_t) + " × " + str(u.ancr) + " " + str(u.classe) + " : "
          + ", ".join("(" + F(t["x"], 0) + " ; " + F(t["y"], 0) + ")" for t in R.tiges)
          + " · " + ancrage_txt(u, True) + " · hef " + F(u.hef, 0)
          + "    |    Mortier " + F(u.t_g, 0) + " mm    |    Cordons a " + F(u.a_w, 1).replace(",0", ""))
    l3 = ("NEd " + F(u.N_Ed, 1) + " kN    |    Mx,Ed " + F(u.Mx_Ed, 1) + " kNm    |    My,Ed " + F(u.My_Ed, 1)
          + " kNm    |    Vx,Ed " + F(u.Vx_Ed, 1) + " kN    |    Vy,Ed " + F(u.Vy_Ed, 1) + " kN"
          + ("    |    TEd " + F(u.T_Ed, 2) + " kNm" if N(u.T_Ed) else "")
          + ("    |    en (" + F(R.xN, 0) + " ; " + F(R.yN, 0) + ")" if abs(R.xN) + abs(R.yN) > 1e-9 else ""))
    return l1, l2, l3


def lignes_hypotheses(R):
    u = R.u
    h = ["Calcul manuel de la platine — critère " + R.plaque["critere"] + " ; appui : "
         + ("toute la platine (platine rigide)." if R.plein else "bande c = " + F(R.c, 0) + " mm autour des parois (EC3 §6.2.5)."),
         "Efforts des tiges par la platine rigide (lit sans traction, tiges tendues) : C = " + F(R.C, 0) + " kN, pmax = "
         + F(R.pmax, 1) + " MPa (fjd = " + F(R.fjd, 1) + "), ΣT = " + F(R.T_tot, 0) + " kN ; kc = " + F(R.k_c, 0)
         + " N/mm³, kb = " + F(R.k_b / 1e3, 0) + " kN/mm.",
         "Bras m jusqu'à l'appui le plus proche, cordon 0,8·a·√2 = " + F(R.cord, 1) + " mm déduit ; leff par boulon "
         "(Tab. 6.6) plafonnée à L/2 + m (m dans un angle) ; section au nu b = min(platine ; L + 2m).",
         "Côté comprimé : chaque paroi reprend en console la pression qui lui est la plus proche ; bande la plus sollicitée.",
         "Tiges (acier) et appui (fjd) en information ; γM0 " + F(N(u.g_M0), 2) + " (ANB) · γM2 " + F(N(u.g_M2), 2)
         + " · γc " + F(N(u.g_c), 2) + " (αcc " + F(N(u.a_cc), 2) + ")."]
    return h


class Note(_np.Note):
    TITRE_NOTE = TITRE
    SYN = synthese
    SCH = schemas
    COL2 = ("platine", "trous")
    COL3 = ("tiges", "appui")
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

    def table_simple(self, d, fr, cle, largeurs):
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
            self.table_main(d, fr)

    def table_main(self, d, fr):
        """Le calcul à la main : boulon par boulon, section par section, la
        bande comprimée, les trous, l'épaisseur requise."""
        R = self.R
        P = R.plaque
        u = R.u
        if not R.ck["tpq"].active:
            return

        def mm(v):
            return "∞" if v == float("inf") else F(v, 1)
        w = fr.w
        court = self.mode_lignes != "toutes"
        n_b, n_s = MAX_LIGNES.get(self.mode_lignes, (6, 4))
        cols = [("Pas", 58, "left"), ("Le moment pris, sa résistance, l'épaisseur qu'il demande", w - 58, "left")]
        lignes = []
        tendus = sorted([t for t in R.tiges if t["T"] > 1e-9], key=lambda t: -t["eta"])
        for t in tendus[:n_b]:
            txt = ("T = " + F(t["T"], 1) + " kN ; appui " + t["appui"]["face"].replace("élément ", "É") + " ; m = "
                   + F(t["m"], 1) + " mm" + (" (angle)" if t["coin"] else "") + " ; T·m = " + F(t["M"], 2) + " kNm ; leff = "
                   + F(t["leff1"], 0) + " mm")
            if not court:
                txt += (", Mpl = " + F(t["Mpl1"], 2) + " kNm, " + ("avec" if t["levier"] else "sans") + " effet de levier")
            txt += (" → FT,Rd = " + F(t["FT_Rd"], 0) + " kN (mode " + str(t["mode_ts"]).split(" ")[0] + ") → "
                    + pct(t["T"] / t["FT_Rd"] if t["FT_Rd"] > 0 else 9.99, 0) + " ; tp plast. " + F(t["t_pl"], 1)
                    + " / élast. " + F(t["t_el"], 1) + " mm")
            lignes.append([("lab", "Boulon " + js_str(t["i"] + 1)), ("lab", txt)])
        if len(tendus) > n_b:
            lignes.append([("lab", "…"), ("lab", js_str(len(tendus) - n_b) + " autre(s) boulon(s) tendu(s), moins sollicité(s) — export texte.")])
        secs = sorted(R.sections, key=lambda s: -s["eta"])
        for s in secs[:n_s]:
            txt = (s["nom"].replace("élément ", "É").replace("raidisseur ", "R") + " : boulons "
                   + ", ".join(str(i + 1) for i in s["idx"]) + ", m⊥ " + ", ".join(F(m, 0) for m in s["m"])
                   + " mm ; M = ΣT·m = " + F(s["M"], 2) + " kNm ; b = " + ("" if court else "min(platine ; L + 2m) = ") + F(s["b"], 0)
                   + " mm ; Mpl = " + F(s["Mpl"], 2) + " kNm → " + pct(s["eta"], 0) + " ; tp plast. " + F(s["t_pl"], 1)
                   + " / élast. " + F(s["t_el"], 1) + " mm")
            lignes.append([("lab", "Section"), ("lab", txt)])
        if len(secs) > n_s:
            lignes.append([("lab", "…"), ("lab", js_str(len(secs) - n_s) + " autre(s) section(s) — export texte.")])
        cs = R.console
        if cs["m_Ed"] > 1e-9:
            lignes.append([("lab", "Bande comprimée"),
                           ("lab", cs["nom"].replace("élément ", "É") + " : p au nu " + F(cs["p0"], 2) + " MPa, console de "
                            + F(cs["L"], 0) + " mm ; mEd = ∫p·t dt = " + F(cs["m_Ed"], 2) + " ≤ tp²·fy/(6·γM0) = "
                            + F(cs["m_Rd"], 2) + " kNm/m → tc = " + F(cs["t"], 1) + " mm")])
        pb = R.pire_appui
        if pb and pb["Fv"] > 1e-9:
            lignes.append([("lab", "Trous"),
                           ("lab", "boulon " + js_str(pb["i"] + 1) + " : Fv = " + F(pb["Fv"], 1) + " kN ; αb = " + F(pb["ab"], 2)
                            + ", k1 = " + F(pb["k1"], 2) + " ; Fb,Rd = k1·αb·fu·d·tp/γM2 = " + F(pb["Fb"], 0) + " kN → tb = "
                            + F(pb["t_b"], 1) + " mm")])
        lignes.append([("lab", "Épaisseur requise", "gov"),
                       ("lab", "max(tT " + mm(P["t_T"]) + " ; tS " + mm(P["t_S"]) + " ; tc " + mm(P["t_c"]) + " ; tb "
                        + mm(P["t_b"]) + ") = " + mm(P["t_req"]) + " mm pour tp = " + F(u.tp_u, 0) + " mm ("
                        + pct(R.ck["tpq"].eta, 0) + ") — répartition recalculée à chaque épaisseur", "gov")])
        self.table(d, fr, "LE CALCUL À LA MAIN", cols, lignes)

    def _bloc_formules(self, d, c, largeur):
        """Formules réduites à la ligne du taux dans les variantes courtes."""
        if self.mode_lignes == "mini":
            self.mode_lignes = "eta"
            try:
                return super()._bloc_formules(d, c, largeur)
            finally:
                self.mode_lignes = "mini"
        return super()._bloc_formules(d, c, largeur)

    def colonne_dessins(self, d, fr):
        S, R = self.S, self.R
        hyp = self.hypotheses(R)
        lignes = []
        for h in hyp:
            lignes.extend(d.wrap(h, S.f_b, 6.2, fr.w))
        h_hyp = 13 + len(lignes) * 6.2 * 1.3 + 4
        dispo = fr.room() - h_hyp - 2 * 13 - 6
        hp = dispo * 0.60
        he = dispo - hp
        opt = schemas.options_rapport()
        self.titre(d, fr, "VUE EN PLAN — RÉPARTITION, BRAS (mm)")
        fr.down(hp)
        peindre(d, schemas.plan(R, opt), fr.x, fr.y, fr.w, hp - 2)
        self.titre(d, fr, "COUPE A-A")
        fr.down(he)
        peindre(d, schemas.coupe_aa(R, opt), fr.x, fr.y, fr.w, he - 2)
        self.titre(d, fr, "HYPOTHÈSES")
        fr.down(3)
        for l in lignes:
            fr.down(6.2 * 1.3)
            d.t(fr.x, fr.y, l, S.f_b, 6.2, S.mut)

    def conclusion(self, d, x, w):
        S, R = self.S, self.R
        P = R.plaque
        req = "∞" if P["t_req"] == float("inf") else F(P["t_req"], 1)
        txt = (("PLATINE VÉRIFIÉE" if R.verified else "PLATINE NON VÉRIFIÉE") + " (calcul manuel) : épaisseur requise " + req
               + " mm" + (" ≤ " if R.verified else " > ") + "tp = " + F(R.u.tp_u, 0) + " mm, selon EN 1993-1-8 §6.2.4 "
               "(tronçons en T), §6.2.5 (bande comprimée) et Tableau 3.4 " + self.NORME + " — tiges et béton d'ancrage en "
               "information" + (" ; sous réserve des points signalés." if R.verified and (R.reserve or R.alerts) else "."))
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
class PlanPrincipeMan(PlanPrincipe):
    def __init__(self, R, doc_meta):
        super().__init__(R, doc_meta)
        self.titre = TITRE

    def _vues(self, dnm):
        f = TEXTE_MM * dnm
        R = self.R
        return [("COUPE A-A", schemas.coupe_aa(R, schemas.options_fabrication(R, schemas.EXCLURE_AA, f))),
                ("VUE EN PLAN", schemas.plan(R, schemas.options_fabrication(R, schemas.EXCLURE_PLAN, f))),
                ("COUPE B-B", schemas.coupe_bb(R, schemas.options_fabrication(R, schemas.EXCLURE_BB, f)))]

    def _cartouche(self, d, dnm):
        S, R, doc = self.S, self.R, self.doc
        u = R.u
        W = d.W
        x0, y0, w, h = M, M, W - 2 * M, CART_H
        d.box(x0, y0, w, h, stroke=S.ink, lw=0.9)
        fix = ("Boulons " + js_str(R.n_t) + " × " + str(u.ancr) + " " + str(u.classe) + " – trous Ø" + F(R.d0, 0)
               + " – perçages cotés depuis les bords (vue en plan) · " + ancrage_txt(u, True) + " · hef " + F(u.hef, 0))
        vis = "par boulon : " + (str(doc.get("visserie") or "").strip() or VISSERIE_DEFAUT)
        ident = " · ".join(str(x) for x in (doc.get("projet"), doc.get("partie")) if x)
        elems = " + ".join(EL.nom(e) + " (" + F(e["x"], 0) + " ; " + F(e["y"], 0) + ")" for e in R.elems)
        raid = (js_str(len(R.raid)) + " raidisseur(s) " + ", ".join(F(r["L"], 0) + "×" + F(r["t"], 0) for r in R.raid)) if R.raid \
            else "sans raidisseur"
        cases = [("ASSEMBLAGE", "Platine sur béton – calcul manuel", ident or "—", 0.17),
                 ("ÉLÉMENTS SOUDÉS", elems, raid, 0.20),
                 ("PLATINE · MASSIF", js_str(N(u.bp_u)) + "×" + js_str(N(u.lp_u)) + "×" + js_str(N(u.tp_u)) + " " + str(u.nu_pl),
                  "massif " + js_str(N(u.bc_u)) + "×" + js_str(N(u.lc_u)) + "×" + js_str(N(u.hc_u)) + " " + str(u.beton)
                  + " · mortier " + F(u.t_g, 0) + " · a " + F(u.a_w, 1).replace(",0", ""), 0.19),
                 ("BOULONS", fix, vis, 0.225),
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
    """La note (bytes) : page 1 du calcul manuel, page 2 « PLAN DE PRINCIPE »."""
    global derniers_avertissements, derniere_variante, derniere_echelle, derniere_disposition
    infos = infos or {}
    doc_meta = ndc_data.construire_doc(infos, date_defaut=datetime.today().strftime("%d/%m/%Y"))
    doc_meta["titre"] = "Note de calcul"
    doc_meta["visserie"] = infos.get("visserie", "")
    if chemin is None:
        fd, chemin = tempfile.mkstemp(suffix=".pdf", prefix="note_platine_manuelle_")
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
    pp = PlanPrincipeMan(R, doc_meta)
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

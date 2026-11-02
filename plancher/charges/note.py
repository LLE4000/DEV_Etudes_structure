# -*- coding: utf-8 -*-
"""Note de calcul « Charges sur plancher » — DEUX pages A4 paysage, sur le
kit ``ndc_pdf`` des notes de l'application (palette Encre, mêmes marges et
mêmes polices que les notes d'assemblage).

Page 1 · CALCUL : bandeau (statut, taux maximal, famille et objet
gouvernants, taux par famille, deux lignes de données), puis trois
colonnes — le PLAN tel qu'à l'écran (peint depuis le même SVG, source
unique), la COUPE sous l'appui sélectionné et les HYPOTHÈSES du moteur ;
le tableau des ÉLÉMENTS (M_Ed, M_Rd, η_M, V_Ed, V_Rd, η_V) et celui des
CHARGES ; le tableau des VÉRIFICATIONS (plateaux, filières, effets locaux,
joints, fissuration, lignes d'hypothèse) et la CONCLUSION.

Page 2 · MÉTHODE ET DÉTAIL : les sections de ``methode.py`` (mêmes textes
que l'onglet Méthode), la BIBLIOTHÈQUE utilisée avec ses sources, le
BALAYAGE de position s'il y a un engin, ce qui reste à transcrire, et un
cartouche (cas, plancher, charges, répartiteurs, date · indice, échelle du
plan).

``warnings`` : débordements de colonne (vide si tout tient) ; la note est
d'abord composée en taille normale, puis en taille réduite si une colonne
déborde (variante « serrée »), comme les notes d'assemblage.
"""
import math
import os
import tempfile
from datetime import datetime
from typing import Dict, List, Optional

from reportlab.lib.pagesizes import A4, landscape

from ndc_pdf import data as ndc_data
from ndc_pdf.kit import Doc, Frame, chip, mix
from ndc_pdf.styles import Encre

from . import dessin as DS
from . import etat as ET
from . import methode as ME
from . import verification as V
from .peintre import peindre_svg

A4L = landscape(A4)
M, BAS, GAP = 22, 40, 12
LARG = (284, 232)                     # colonnes 1 et 2 de la page 1 ; la 3e prend le reste (282 pt)
S_TAB_H, S_DATA = 5.6, 7.0
LEAD_TAB = 8.8
TITRE = "Charges de chantier sur plancher"
TYPES = {"hourdis": "Hourdis", "dalle": "Dalle coulée en place", "predalle": "Prédalle"}
STATUTS = {"ok": ("CONFORME", "ok"), "attention": ("CONFORME · ATTENTION", "att"), "nok": ("NON CONFORME", "ko"),
           "refus": ("DONNÉES MANQUANTES", "mut")}
FAMILLES = [("flexion", "Flexion"), ("tranchant", "Tranchant"), ("local", "Local"), ("plateau", "Plateaux"),
            ("filiere", "Filières"), ("joint", "Joints")]
LEGENDE = ("Flexion et tranchant : efforts de calcul γG·G + γQ·Q contre les résistances de calcul du catalogue (nature Rd), "
           "ou efforts de service contre les valeurs admissibles (nature service) — P-2 ; tranchant 100 % sur l'élément chargé — P-3.")
NOTATIONS = ("M_Ed, V_Ed : efforts sollicitants · M_Rd, V_Rd : résistances · η : taux · E, R : effet et résistance d'une ligne · "
             "P : charge de service par appui, φ : coefficient dynamique · a, b : semelle · c : bande plastique (EN 1993-1-8 §6.2.5).")
VARIANTES = ((1.0, "normale"), (0.92, "serrée"))


def _fmt(x, nd=2) -> str:
    if x is None:
        return "—"
    if isinstance(x, float) and (math.isinf(x) or math.isnan(x)):
        return "∞"
    return f"{x:,.{nd}f}".replace(",", " ").replace(".", ",")


def _pct(t) -> str:
    if t is None:
        return "—"
    if isinstance(t, float) and (math.isinf(t) or math.isnan(t)):
        return "∞"
    return f"{t * 100:.0f} %"


def _val(x) -> str:
    if x is None:
        return "—"
    if isinstance(x, (int, float)):
        return _fmt(x, 2 if abs(x) < 10 else 1)
    return str(x)


class Calcul:
    """Tout ce que la note imprime, calculé une fois : moteur, dessins, textes."""

    def __init__(self, etat: Dict, cas: Dict):
        self.etat, self.cas = etat, cas
        self.plancher, self.charges, self.reps, self.params, self.index = ET.construire(etat, cas)
        self.res = V.verifier(self.plancher, self.charges, self.reps, self.params)
        self.ctx = ET.contexte_dessin(etat, cas, self.res, self.plancher, self.index)
        self.ctx["impression"] = True             # sans poignées ni invites de clic, sans halo de sélection
        self.svg_plan = DS.plan(self.ctx)[0]
        self.svg_coupe = DS.coupe(ET.contexte_coupe(etat, cas, self.res, self.plancher, self.index)) if self.index["empreintes"] else ""
        self.svg_balayage = ""
        engins = [c["ref"] for c in cas["charges"] if c["type"] == "engin"]
        if engins and self.res["statut"] != "refus":
            b = V.balayer(self.plancher, self.charges, self.reps, engins[0], self.params)
            x_act = next(c["x"] for c in cas["charges"] if c["ref"] == engins[0])
            self.svg_balayage = DS.balayage(b["points"], b["pire"], x_act, self.plancher.L)
            self.balayage = b
        self.taux_familles = {}
        for fam, _ in FAMILLES:
            ts = [v["taux"] for v in self.res.get("verifications", []) if v["famille"] == fam and v["taux"] is not None
                  and v["statut"] in ("ok", "attention", "nok")]
            if ts:
                self.taux_familles[fam] = max(ts)

    # --- lignes de données de l'en-tête
    def ligne_plancher(self) -> str:
        pl, p = self.cas["plancher"], self.plancher
        if pl["type"] == "hourdis":
            h = pl["hourdis"]
            ch = (f" + couche de compression {h['chape']['epaisseur'] * 1e3:.0f} mm (fck(t) {h['chape']['fck_t']:g} MPa)"
                  if h.get("chape", {}).get("oui") else "")
            typ = (h.get("type_cat") or "").split(" · ")[0] or "Hourdis"
            if not typ.lower().startswith("hourdis"):
                typ = "Hourdis " + typ
            return (f"{typ}{ch} · L {_fmt(p.L)} m · {h['n_elements']} × {_fmt(h['b_element'])} m · "
                    f"h {h['h'] * 1e3:.0f} mm · {h.get('etat', '')} · M_Rd {_fmt(h['M_Rd'], 1)} kN·m, V_Rd {_fmt(h['V_Rd'], 1)} kN par élément ({h.get('nature', 'Rd')}) · "
                    f"g {_fmt(h['g'], 2)} + présentes {_fmt(pl['q_presentes'], 2)} kN/m²")
        d = pl[pl["type"]]
        extra = f" · prédalle {_fmt(d['b_predalle'])} m, joints {'armés' if d.get('joints_armes') else 'non armés'}" if pl["type"] == "predalle" else ""
        return (f"{TYPES[pl['type']]} · L {_fmt(p.L)} m · largeur {_fmt(d['largeur'])} m · h {d['h'] * 1e3:.0f} mm, d {d['d'] * 1e3:.0f} mm · "
                f"{d['beton']}" + (f" (fck(t) {d['fck_t']:g} MPa)" if d.get("fck_t") else "") + f" · As {_fmt(d['As_inf'], 0)} mm²/m, fyk {_fmt(d['fyk'], 0)} MPa"
                + extra + f" · présentes {_fmt(pl['q_presentes'], 2)} kN/m²")

    def ligne_charges(self) -> str:
        morceaux = []
        for c in self.cas["charges"]:
            if c["type"] == "engin":
                P = c.get("P") or {}
                morceaux.append(f"{c['ref']} {c['gabarit'].split(' · ')[0]} ({', '.join(f'{k} {_fmt(v, 0)}' for k, v in P.items())} kN) en x {_fmt(c['x'])}, y {_fmt(c['y'])} m, {c.get('angle', 0)}°")
            elif c["type"] == "ponctuel":
                morceaux.append(f"{c['ref']} étançon {_fmt(c['P'], 0)} kN, semelle {c['a'] * 1e3:.0f} × {c['b'] * 1e3:.0f} mm en x {_fmt(c['x'])}, y {_fmt(c['y'])} m")
            elif c["type"] == "lineaire":
                morceaux.append(f"{c['ref']} file d'étais {_fmt(c['p'], 1)} kN/m sur {_fmt(c['longueur'])} m en x {_fmt(c['x'])}, y {_fmt(c['y'])} m")
            else:
                morceaux.append(f"{c['ref']} stock {_fmt(c['q'], 1)} kN/m² sur {_fmt(c['lx'])} × {_fmt(c['ly'])} m")
        reps = []
        for ref, rp in self.cas["repartiteurs"].items():
            reps.append(f"{ref} : " + (rp["nom"].split(" · ")[0] if rp["type"] == "plateau" else f"filière {rp['id']}"))
        for f in self.cas.get("filieres", []):
            reps.append(f"{f['id']} = {f['nom']} × {_fmt(f['longueur'])} m")
        return ("Charges : " + (" ; ".join(morceaux) if morceaux else "aucune")
                + "   |   Répartiteurs : " + (" ; ".join(reps) if reps else "aucun"))


_ORDRE = {"nok": 4, "non_verifie": 3, "attention": 2, "ok": 1, "info": 0}


def _grouper(verifs: List[Dict]) -> List:
    """Une ligne par (famille, objet, charge d'origine) : les empreintes d'une
    charge linéaire ou surfacique discrétisée (« S1.17 »), ou les éléments
    sous une filière (« F1 · HEB 140→H2 »), donnent UNE ligne, la pire —
    sinon un stock de 32 empreintes écrivait 96 lignes d'effets locaux."""
    groupes: Dict = {}
    ordre: List = []
    for v in verifs:
        tete, _, reste = v["objet"].partition(" sous ")
        nom_groupe = v["objet"]
        if reste:
            ref = reste
            while True:                                 # S1.3.7 → S1 ; F1.4 → F1
                base = ref.rsplit(".", 1)
                if len(base) == 2 and base[1].isdigit():
                    ref = base[0]
                else:
                    break
            if "→" in ref:
                ref = ref.split("→")[0]
            cle = (v["famille"], tete, ref)
            nom_groupe = f"{tete} sous {ref}"
        else:
            # sans « sous », chaque ligne est une vérification distincte (flexion d'une filière ET son
            # ressort qui tire, par exemple) : jamais fusionnée
            cle = (v["famille"], v["objet"], v["methode"], len(ordre))
        if cle not in groupes:
            groupes[cle] = [v, 1]
            ordre.append(cle)
        else:
            g = groupes[cle]
            g[1] += 1
            a, b = g[0], v
            pire = (_ORDRE.get(b["statut"], 0), b["taux"] if b["taux"] is not None else -1.0) > \
                   (_ORDRE.get(a["statut"], 0), a["taux"] if a["taux"] is not None else -1.0)
            g[0] = dict(b if pire else a, objet=nom_groupe)
    return [(groupes[c][0], groupes[c][1]) for c in ordre]


class Note:
    """Page 1 · CALCUL."""

    def __init__(self, calc: Calcul, doc_meta: Dict, k: float = 1.0):
        self.C, self.doc, self.k = calc, doc_meta, k
        self.S = Encre()
        self.warnings: List[str] = []
        self.s_tab = 6.5 * k
        self.s_hyp = 6.2 * k
        self.lead = LEAD_TAB * k
        self.echelle_plan = None

    # --- briques
    def titre(self, d, fr, txt):
        S = self.S
        fr.down(10)
        d.t(fr.x, fr.y, txt, S.f_v, 6.2, S.acc, track=1.3)
        fr.down(3)
        d.line(fr.x, fr.y, fr.x1, fr.y, mix(S.rule, S.ink, 0.12), 0.5)

    def table(self, d, fr, titre, cols, lignes, note=None):
        """``cols`` : (entête, largeur, alignement) ; ``lignes`` : cellules
        (style, texte[, ok]) — styles lab (retour à la ligne), ref (petit,
        gris), num (droite), eta (droite, coloré selon ok), st (statut)."""
        S, st, lead = self.S, self.s_tab, self.lead
        self.titre(d, fr, titre)
        fr.down(8)
        xs, x = [], fr.x
        for ent, w, al in cols:
            xs.append(x)
            d.t(x + (w - 3 if al == "right" else 0), fr.y, ent, S.f_b, S_TAB_H, S.mut, al)
            x += w
        fr.down(2.5)
        d.line(fr.x, fr.y, fr.x1, fr.y, S.acc, 0.5)
        for cells in lignes:
            rendus, n_max = [], 1
            for (ent, w, al), cell in zip(cols, cells):
                style, txt = cell[0], cell[1]
                if style in ("lab", "ref"):
                    size = st * 0.87 if style == "ref" else st
                    ls = d.wrap(txt, S.f_b, size, w - 4)
                    n_max = max(n_max, len(ls))
                    rendus.append((style, ls, cell))
                else:
                    rendus.append((style, [txt], cell))
            h = lead * n_max + 1.2
            fr.down(lead - 1.8)
            y = fr.y
            for (ent, w, al), (style, ls, cell), xx in zip(cols, rendus, xs):
                if style == "lab":
                    f = "Carlito-Bold" if (len(cell) > 2 and cell[2]) else S.f_b
                    for i, l in enumerate(ls):
                        d.t(xx, y - i * lead, l, f, st, S.ink)
                elif style == "ref":
                    for i, l in enumerate(ls):
                        d.t(xx, y - i * lead, l, S.f_b, st * 0.87, S.mut)
                elif style == "num":
                    d.t(xx + w - 3, y, ls[0], S.f_b, st, S.ink, "right")
                elif style == "eta":
                    ok = cell[2]
                    gras = len(cell) > 3 and cell[3]
                    d.t(xx + w - 3, y, ls[0], "Carlito-Bold" if (gras or not ok) else S.f_b, st,
                        S.ok if ok else S.ko, "right")
                elif style == "st":
                    col = {"ok": S.ok, "att": S.att, "ko": S.ko, "mut": S.mut}[cell[2]]
                    d.t(xx + w - 3, y, ls[0], "Carlito-Bold", st, col, "right")
            fr.down(h - (lead - 1.8))
            d.line(fr.x, fr.y + 1.5, fr.x1, fr.y + 1.5, mix(S.rule, S.ink, 0.06), 0.4)
        if note:
            fr.down(st * 1.3)
            d.t(fr.x, fr.y, d.fit(note, S.f_b, st * 0.9, fr.w), S.f_b, st * 0.9, S.mut)
        fr.down(2)

    # --- bandeau
    def bandeau(self, d):
        S, C, doc = self.S, self.C, self.doc
        res, cas = C.res, C.cas
        W, H = d.W, d.H
        y = H - M - 4
        d.t(M, y, doc.get("bureau", ""), S.f_v, 8.0, S.acc, track=1.4)
        d.t(W - M, y, "Note de calcul · " + doc.get("date", "") + " · indice " + str(doc.get("indice", "")), S.f_b, 8, S.mut, "right")
        y -= 15
        d.t(M, y, TITRE + " — " + TYPES[cas["plancher"]["type"]].lower(), S.f_h, 12.5, S.ink)
        ident = " · ".join(str(x) for x in (doc.get("projet"), doc.get("partie"), "cas " + cas["nom"]) if x)
        d.t(W - M, y, ident, S.f_b, 8, S.ink, "right")
        y -= 8
        d.line(M, y, W - M, y, S.ink, 1.0)
        y -= 15
        libelle, cle = STATUTS[res["statut"]]
        col = {"ok": S.ok, "att": S.att, "ko": S.ko, "mut": S.mut}[cle]
        w = chip(d, M, y - 3.2, libelle, S.f_v, 6.6, "#FFFFFF", col)
        x = M + w + 8
        if res["statut"] == "refus":
            d.t(x, y, d.fit(res["message"], S.f_b, 8, W - M - x), S.f_b, 8, S.ink)
        else:
            t1 = "η max " + _pct(res["taux_max"])
            d.t(x, y, t1, "Carlito-Bold", 8.6, col)
            x += d.w(t1, "Carlito-Bold", 8.6) + 5
            g = res.get("gouverne")
            t2 = f"— gouverne : {g['famille']} · {g['objet']}" if g else "— aucune charge"
            pieces = []
            for fam, lab in FAMILLES:
                if fam in C.taux_familles:
                    t = C.taux_familles[fam]
                    pieces.append((lab + " ", S.f_b, S.ink))
                    pieces.append((_pct(t), "Carlito-Bold", S.ok if t <= 1.0 else S.ko))
                    pieces.append(("   ", S.f_b, S.ink))
            wt = sum(d.w(t, f, 8) for t, f, _ in pieces)
            xr = W - M - wt
            d.t(x, y, d.fit(t2, S.f_b, 8, xr - x - 6), S.f_b, 8, S.ink)
            for t, f, cc in pieces:
                d.t(xr, y, t, f, 8, cc)
                xr += d.w(t, f, 8)
        nv = res.get("non_verifies") or []
        if nv:
            y -= 10
            d.box(M + 1, y - 0.5, 4.2, 4.2, fill=S.att)
            objets = [v["objet"] + (f" (×{n})" if n > 1 else "") for v, n in _grouper(nv)]
            d.t(M + 9, y, d.fit(f"{len(nv)} ligne(s) non vérifiée(s) : paramètre à transcrire ou donnée absente — "
                                 + " ; ".join(objets), S.f_b, 6.8, W - 2 * M - 12), S.f_b, 6.8, S.ink)
        y -= 12
        d.t(M, y, d.fit(C.ligne_plancher(), S.f_b, S_DATA, W - 2 * M), S.f_b, S_DATA, S.ink)
        y -= S_DATA * 1.4
        d.t(M, y, d.fit(C.ligne_charges(), S.f_b, S_DATA, W - 2 * M), S.f_b, S_DATA, S.ink)
        y -= 6
        d.line(M, y, W - M, y, mix(S.rule, S.ink, 0.3), 0.6)
        return y - 4

    # --- colonne 1 : plan, coupe, hypothèses
    def colonne_plan(self, d, fr):
        S, C = self.S, self.C
        lignes = []
        textes = list(C.res.get("hypotheses", []))
        for v in C.res.get("verifications", []):          # notes du moteur (P-4, P-8, P-16, chape…), une fois chacune
            if v["famille"] == "hypothese" and v["methode"] not in textes:
                textes.append(v["methode"])
        for h in textes:
            lignes.extend(d.wrap("• " + h, S.f_b, self.s_hyp, fr.w))
        h_hyp = 13 + len(lignes) * self.s_hyp * 1.3 + 4
        h_coupe = 78 if C.svg_coupe else 0
        dispo = fr.room() - h_hyp - 13 - (13 if C.svg_coupe else 0) - 6
        hp = max(80.0, dispo - h_coupe)
        self.titre(d, fr, "PLAN — appuis A et B, taux par élément")
        fr.down(hp)
        self.echelle_plan = peindre_svg(d.c, C.svg_plan, fr.x, fr.y, fr.w, hp - 2)
        if C.svg_coupe:
            self.titre(d, fr, "COUPE SOUS L'APPUI SÉLECTIONNÉ")
            fr.down(h_coupe)
            peindre_svg(d.c, C.svg_coupe, fr.x, fr.y, fr.w, h_coupe - 2)
        self.titre(d, fr, "HYPOTHÈSES")
        fr.down(3)
        for l in lignes:
            fr.down(self.s_hyp * 1.3)
            d.t(fr.x, fr.y, l, S.f_b, self.s_hyp, S.mut)

    # --- colonne 2 : éléments et charges
    def table_elements(self, d, fr):
        C = self.C
        els = C.res.get("elements", [])
        if not els:
            return
        unite = "kN·m/m" if C.plancher.type != "hourdis" else "kN·m"
        uv = "kN/m" if C.plancher.type != "hourdis" else "kN"
        gouv = C.res.get("gouverne") or {}
        rows = []
        tri = sorted(els, key=lambda e: -(e["taux"] if e["taux"] is not None and not math.isinf(e["taux"]) else 1e9))
        montres = tri[:12] if len(els) > 13 else els
        if len(els) > 13:
            montres = sorted(montres, key=lambda e: e["index"])
        for e in montres:
            g = e["nom"] == gouv.get("objet")
            rows.append([("lab", e["nom"], g), ("num", _fmt(e["M_max"], 1)), ("num", _fmt(e["M_adm"], 1)),
                         ("eta", _pct(e["taux_M"]), e["taux_M"] <= 1.0, g and e["gouverne"] == "flexion"),
                         ("num", _fmt(e["V_max"], 1)), ("num", _fmt(e["V_adm"], 1)),
                         ("eta", _pct(e["taux_V"]), e["taux_V"] <= 1.0, g and e["gouverne"] == "tranchant")])
        note = None
        if len(els) > 13:
            reste = [e for e in els if e not in montres]
            note = f"+ {len(reste)} élément(s) non montrés, tous ≤ {_pct(max(e['taux'] for e in reste))}"
        w = fr.w
        cols = [("ÉLÉMENT", 42, "left"), (f"M_Ed {unite}", 38, "right"), ("M_Rd", 32, "right"), ("η", 26, "right"),
                (f"V_Ed {uv}", 36, "right"), ("V_Rd", 32, "right"), ("η", w - 206, "right")]
        titre = "ÉLÉMENTS — flexion et tranchant" + (" (bandes d'un mètre)" if C.plancher.type != "hourdis" else "")
        self.table(d, fr, titre, cols, rows, note)

    def table_charges(self, d, fr):
        C = self.C
        rows = []
        patches = {p["ref"]: p for p in C.res.get("patches", [])}
        for c in C.cas["charges"]:
            if c["type"] == "engin":
                en = C.index["engins"][c["ref"]]
                rows.append([("lab", c["ref"], True), ("ref", c["gabarit"].split(" · ")[0] + f" · {c.get('angle', 0)}°"),
                             ("num", f"{_fmt(c['x'])} · {_fmt(c['y'])}"), ("num", _fmt(sum((c.get('P') or {}).values()), 0)), ("ref", "")])
                for e in en.empreintes():
                    rp = C.cas["repartiteurs"].get(e.ref)
                    rtxt = "—" if not rp else (rp["nom"].split(" · ")[0] if rp["type"] == "plateau" else f"filière {rp['id']}")
                    rows.append([("ref", "  " + e.ref.split(".")[-1]), ("ref", f"semelle {e.a * 1e3:.0f} × {e.b * 1e3:.0f}, φ {e.phi:g}"),
                                 ("num", f"{_fmt(e.x)} · {_fmt(e.y)}"), ("num", _fmt(e.P, 0)), ("ref", rtxt)])
            elif c["type"] == "ponctuel":
                rp = C.cas["repartiteurs"].get(c["ref"])
                rtxt = "—" if not rp else (rp["nom"].split(" · ")[0] if rp["type"] == "plateau" else f"filière {rp['id']}")
                rows.append([("lab", c["ref"], True), ("ref", f"étançon, semelle {c['a'] * 1e3:.0f} × {c['b'] * 1e3:.0f}, φ {c.get('phi', 1.0):g}"),
                             ("num", f"{_fmt(c['x'])} · {_fmt(c['y'])}"), ("num", _fmt(c["P"], 0)), ("ref", rtxt)])
            elif c["type"] == "lineaire":
                rows.append([("lab", c["ref"], True), ("ref", f"file d'étais {_fmt(c['p'], 1)} kN/m × {_fmt(c['longueur'])} m, {'en travers' if c.get('orientation', 'y') == 'y' else 'dans le sens porteur'}"),
                             ("num", f"{_fmt(c['x'])} · {_fmt(c['y'])}"), ("num", _fmt(c["p"] * c["longueur"], 0)), ("ref", "—")])
            else:
                rows.append([("lab", c["ref"], True), ("ref", f"stock {_fmt(c['q'], 1)} kN/m² sur {_fmt(c['lx'])} × {_fmt(c['ly'])} m"),
                             ("num", f"{_fmt(c['x0'])} · {_fmt(c['y0'])}"), ("num", _fmt(c["q"] * c["lx"] * c["ly"], 0)), ("ref", "—")])
        if not rows:
            rows = [[("ref", "aucune charge posée"), ("ref", ""), ("ref", ""), ("ref", ""), ("ref", "")]]
        w = fr.w
        cols = [("RÉF", 30, "left"), ("CHARGE", w - 30 - 46 - 34 - 56, "left"), ("x · y (m)", 46, "right"), ("P (kN)", 34, "right"),
                ("RÉPARTITEUR", 56, "left")]
        self.table(d, fr, "CHARGES — valeurs de service", cols, rows)

    # --- colonne 3 : vérifications et conclusion
    def table_verifs(self, d, fr):
        C, S = self.C, self.S
        rows = []
        # les lignes d'hypothèse (P-4, P-8, P-16, notes de chape…) vont au bloc HYPOTHÈSES de la colonne 1
        for v, n in _grouper([v for v in C.res.get("verifications", [])
                              if v["famille"] != "hypothese" and not (v["famille"] in ("flexion", "tranchant") and C.plancher.type == "hourdis")]):
            fam = {"flexion": "flexion", "tranchant": "tranchant", "local": "local", "plateau": "plateau", "filiere": "filière",
                   "joint": "joint", "fissuration": "fissuration"}.get(v["famille"], v["famille"])
            if n > 1:
                fam += f" ×{n}"
            u = v.get("unite", "")
            e_ = (_val(v["E"]) + (" " + u if u and v["E"] is not None else "")) if v["E"] is not None else "—"
            r_ = (_val(v["R"]) + (" " + u if u and v["R"] is not None else "")) if v["R"] is not None else "—"
            t = v["taux"]
            if v["statut"] in ("ok", "attention", "nok") and t is not None:
                der = ("eta", _pct(t), t <= 1.0, v["statut"] == "nok")
            else:
                der = ("st", {"info": "info", "non_verifie": "non vérifié"}.get(v["statut"], v["statut"]),
                       "mut" if v["statut"] == "info" else "att")
            objet = v["objet"] + (" (la pire)" if n > 1 else "")
            rows.append([("ref", fam), ("lab", objet), ("num", e_), ("num", r_), der])
        if not rows:
            rows = [[("ref", "—"), ("ref", "aucune ligne"), ("ref", ""), ("ref", ""), ("ref", "")]]
        w = fr.w
        cols = [("FAMILLE", 38, "left"), ("OBJET", w - 38 - 40 - 40 - 32, "left"), ("E", 40, "right"), ("R", 40, "right"), ("η", 32, "right")]
        self.table(d, fr, "VÉRIFICATIONS — répartiteurs, effets locaux, joints, information", cols, rows)

    def conclusion(self, d, x, w):
        S, C = self.S, self.C
        res = C.res
        libelle, cle = STATUTS[res["statut"]]
        col = {"ok": S.ok, "att": S.att, "ko": S.ko, "mut": S.mut}[cle]
        if res["statut"] == "refus":
            txt = res["message"]
        elif res.get("gouverne"):
            g = res["gouverne"]
            txt = (f"Taux maximal {_pct(res['taux_max'])} en {g['famille']} sur {g['objet']}. "
                   + ("Le plancher porte les charges posées, avec les répartiteurs décrits." if res["statut"] != "nok" else
                      "Le plancher ne porte pas les charges posées en l'état : répartiteur, filière, déplacement de l'engin ou "
                      "réduction de charge à étudier."))
        else:
            txt = "Aucune charge posée."
        nv = res.get("non_verifies") or []
        if nv:
            txt += f" {len(nv)} ligne(s) restent non vérifiées (paramètre à transcrire ou donnée absente) : la conformité est sous cette réserve."
        lignes = d.wrap(txt, S.f_b, 6.6, w - 16)
        h = 13 + 9 + len(lignes) * 6.6 * 1.3 + 8
        y0 = BAS
        d.box(x, y0, w, h, fill=mix(S.panel, "#FFFFFF", 0.3), stroke=col, lw=0.8, r=3)
        d.t(x + 8, y0 + h - 11, "CONCLUSION", S.f_v, 6.2, col, track=1.3)
        yy = y0 + h - 11 - 9
        for l in lignes:
            d.t(x + 8, yy, l, S.f_b, 6.6, S.ink)
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
        self.colonne_plan(d, f1)
        self.table_elements(d, f2)
        self.table_charges(d, f2)
        hc = self.conclusion(d, x3, w3)
        f3.bottom = BAS + hc + 6
        self.table_verifs(d, f3)
        d.t(M, 24, d.fit(LEGENDE, S.f_b, 5.6, W - 2 * M), S.f_b, 5.6, S.mut)
        d.t(M, 16.5, d.fit(NOTATIONS, S.f_b, 5.6, W - 2 * M), S.f_b, 5.6, S.mut)
        for f in (f1, f2, f3):
            if f.overflow() > 1.0:
                self.warnings.append(f"page 1, colonne {f.x:.0f} : {f.overflow():.0f} pt")
        return self.warnings


CART_H = 40


class Methode:
    """Page 2 · MÉTHODE ET DÉTAIL."""

    def __init__(self, calc: Calcul, doc_meta: Dict, k: float = 1.0, echelle_plan: Optional[float] = None):
        self.C, self.doc, self.k = calc, doc_meta, k
        self.S = Encre()
        self.warnings: List[str] = []
        self.s = 6.3 * k
        self.echelle_plan = echelle_plan

    def titre(self, d, fr, txt):
        S = self.S
        fr.down(11)
        d.t(fr.x, fr.y, txt, S.f_v, 6.2, S.acc, track=1.3)
        fr.down(3)
        d.line(fr.x, fr.y, fr.x1, fr.y, mix(S.rule, S.ink, 0.12), 0.5)
        fr.down(2)

    def paragraphe(self, d, fr, txt, col=None, gras=False):
        S = self.S
        for l in d.wrap(txt, "Carlito-Bold" if gras else S.f_b, self.s, fr.w):
            fr.down(self.s * 1.32)
            d.t(fr.x, fr.y, l, "Carlito-Bold" if gras else S.f_b, self.s, col or S.ink)
        fr.down(self.s * 0.5)

    def section(self, d, fr, sec):
        self.titre(d, fr, sec["titre"].upper())
        for p in sec["paragraphes"]:
            self.paragraphe(d, fr, p)

    def bibliotheque(self, d, fr):
        C, S = self.C, self.S
        self.titre(d, fr, "BIBLIOTHÈQUE UTILISÉE ET SOURCES")
        gabs = ET.gabarits(C.etat)
        pl = C.cas["plancher"]
        if pl["type"] == "hourdis":
            h = pl["hourdis"]
            self.paragraphe(d, fr, f"Hourdis : {h.get('type_cat', '')} — source : {h.get('source') or 'à renseigner'}.")
        vus = set()
        for c in C.cas["charges"]:
            if c["type"] == "engin" and c["gabarit"] not in vus:
                vus.add(c["gabarit"])
                g = gabs[c["gabarit"]]
                app = " ; ".join(f"{a['nom']} ({_fmt(a['dx'])} ; {_fmt(a['dy'])}) m, {a['a'] * 1e3:.0f} × {a['b'] * 1e3:.0f} mm, φ {a.get('phi', 1.0):g}" for a in g["appuis"])
                self.paragraphe(d, fr, f"Engin {g['nom']} ({'gabarit générique' if g.get('generique') else 'fiche'}) — appuis : {app}. Source : {g.get('source') or '—'}. "
                                       "Les charges par appui sont celles saisies pour l'opération (tableau de stabilité du constructeur).")
        dispo = ET.repartiteurs_disponibles(C.etat)
        for ref, rp in C.cas["repartiteurs"].items():
            if rp["type"] == "plateau" and rp["nom"] not in vus:
                vus.add(rp["nom"])
                dd = dispo.get(rp["nom"], {})
                self.paragraphe(d, fr, f"Plateau {rp['nom']} : {dd.get('A', 0):.0f} × {dd.get('Bp', 0):.0f} × {dd.get('t', 0):.0f} mm, f {dd.get('f') or '—'} MPa, "
                                       f"{dd.get('materiau', '')} — source : {dd.get('source') or '—'}.")
        for f in C.cas.get("filieres", []):
            dd = dispo.get(f["nom"], {})
            sec = dd.get("section", {})
            desc = f"profilé {sec['profil']}" if sec.get("profil") else f"rectangle {sec.get('b', 0):.0f} × {sec.get('h', 0):.0f} mm sur champ"
            self.paragraphe(d, fr, f"Filière {f['id']} : {f['nom']} ({desc}), E {dd.get('E', 0):.0f} MPa, f {dd.get('f') or '—'} MPa, "
                                   f"{_fmt(f['longueur'])} m — source : {dd.get('source') or '—'}.")

    def balayage(self, d, fr):
        C = self.C
        if not C.svg_balayage:
            return
        self.titre(d, fr, "BALAYAGE DE POSITION DE L'ENGIN")
        h = fr.w * 150.0 / 900.0 + 4
        fr.down(h)
        peindre_svg(d.c, C.svg_balayage, fr.x, fr.y, fr.w, h - 2)
        b = C.balayage
        xa = b["x_admissibles"]
        txt = (f"Positions admissibles du centre de l'engin : de {_fmt(min(xa))} à {_fmt(max(xa))} m ({len(xa)} sur {len(b['points'])}, pas 0,10 m)."
               if xa else "Aucune position admissible le long de la portée avec les charges et répartiteurs actuels.")
        self.paragraphe(d, fr, txt)

    def cartouche(self, d):
        S, C, doc = self.S, self.C, self.doc
        W = d.W
        x0, y0, w, h = M, M, W - 2 * M, CART_H
        d.box(x0, y0, w, h, stroke=S.ink, lw=0.9)
        cas, pl = C.cas, C.cas["plancher"]
        n_eng = sum(1 for c in cas["charges"] if c["type"] == "engin")
        n_aut = len(cas["charges"]) - n_eng
        n_pl = sum(1 for r in cas["repartiteurs"].values() if r["type"] == "plateau")
        n_fi = len(cas.get("filieres", []))
        ech = "—"
        if self.echelle_plan:
            ech = "1:" + f"{2.8346 / self.echelle_plan:.0f}"
        ident = " · ".join(str(x) for x in (doc.get("projet"), doc.get("partie")) if x)
        cases = [("CAS", cas["nom"], ident or "—", 0.16),
                 ("PLANCHER", TYPES[pl["type"]] + " · L " + _fmt(C.plancher.L) + " m", "largeur " + _fmt(C.plancher.B) + " m", 0.20),
                 ("CHARGES", f"{n_eng} engin(s), {n_aut} autre(s)", C.ligne_charges()[:70], 0.26),
                 ("RÉPARTITEURS", f"{n_pl} plateau(x), {n_fi} filière(s)", "", 0.15),
                 ("DATE · INDICE", str(doc.get("date", "")), "indice " + str(doc.get("indice", "")), 0.12),
                 ("ÉCHELLE DU PLAN", ech, "A4 paysage", 0.11)]
        x = x0
        for i, (lab, l1, l2, part) in enumerate(cases):
            cw = w * part
            if i:
                d.line(x, y0 + 2, x, y0 + h - 2, S.rule, 0.6)
            d.t(x + 6, y0 + h - 10, lab, S.f_v, 5.4, S.mut, track=1.2)
            d.t(x + 6, y0 + h - 21, d.fit(str(l1), "Carlito-Bold", 6.6, cw - 12), "Carlito-Bold", 6.6, S.ink)
            if l2:
                d.t(x + 6, y0 + h - 32, d.fit(str(l2), S.f_b, 6.2, cw - 12), S.f_b, 6.2, S.ink)
            x += cw

    def construire(self, d):
        S = self.S
        W, H = d.W, d.H
        y = H - M - 4
        d.t(M, y, self.doc.get("bureau", ""), S.f_v, 8.0, S.acc, track=1.4)
        d.t(W - M, y, TITRE + " · cas " + self.C.cas["nom"], S.f_b, 7.6, S.mut, "right")
        y -= 16
        d.t(M, y, "MÉTHODE ET DÉTAIL", S.f_h, 13.5, S.ink)
        d.t(W - M, y, "Références : " + " · ".join(ME.REFERENCES), S.f_b, 6.2, S.mut, "right")
        y -= 7
        d.line(M, y, W - M, y, S.ink, 1.0)
        top = y - 6
        bas = M + CART_H + 10
        wc = (W - 2 * M - 2 * GAP) / 3.0
        frames = [Frame(M + i * (wc + GAP), top, wc, bas) for i in range(3)]
        for i in (1, 2):
            d.line(frames[i].x - GAP / 2, top, frames[i].x - GAP / 2, bas, S.rule, 0.45)
        secs = ME.SECTIONS
        for sec in secs[:2]:
            self.section(d, frames[0], sec)
        for sec in secs[2:4]:
            self.section(d, frames[1], sec)
        for sec in secs[4:]:
            self.section(d, frames[2], sec)
        self.bibliotheque(d, frames[2])
        self.balayage(d, frames[2])
        self.titre(d, frames[2], "À TRANSCRIRE AVANT UN USAGE EN PRODUCTION")
        self.paragraphe(d, frames[2], " ".join(ME.A_TRANSCRIRE), S.mut)
        self.cartouche(d)
        for f in frames:
            if f.overflow() > 1.0:
                self.warnings.append(f"page 2, colonne {f.x:.0f} : {f.overflow():.0f} pt")
        return self.warnings


def _essai(calc, doc_meta, k, chemin):
    d = Doc(chemin, A4L, title=doc_meta.get("titre", "Note de calcul"))
    d.new_page(A4L)
    n1 = Note(calc, doc_meta, k)
    w = list(n1.construire(d))
    d.new_page(A4L)
    n2 = Methode(calc, doc_meta, k, n1.echelle_plan)
    w += n2.construire(d)
    d.save()
    return w


def generer_note(etat: Dict, cas: Optional[Dict] = None, infos: Optional[Dict] = None, chemin: Optional[str] = None) -> bytes:
    """La note (bytes PDF) : page 1 de calcul, page 2 de méthode et détail.
    ``infos`` : nom_projet, partie, date, indice."""
    global derniers_avertissements, derniere_variante
    cas = cas or ET.cas_actif(etat)
    infos = infos or {}
    doc_meta = ndc_data.construire_doc(infos, date_defaut=datetime.today().strftime("%d/%m/%Y"))
    doc_meta["titre"] = "Note de calcul — charges de chantier sur plancher"
    calc = Calcul(etat, cas)
    if chemin is None:
        fd, chemin = tempfile.mkstemp(suffix=".pdf", prefix="note_plancher_")
        os.close(fd)
    warn, retenue = [], VARIANTES[-1][1]
    for k, nom in VARIANTES:
        warn = _essai(calc, doc_meta, k, chemin)
        retenue = nom
        if not warn:
            break
    derniers_avertissements = warn
    derniere_variante = retenue
    with open(chemin, "rb") as fh:
        return fh.read()


derniers_avertissements: List[str] = []
derniere_variante: Optional[str] = None

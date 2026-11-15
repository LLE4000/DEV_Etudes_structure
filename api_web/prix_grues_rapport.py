# -*- coding: utf-8 -*-
"""Le RAPPORT PDF du module « Prix des grues » (cahier § 11.1, § 11.2) — la même boîte à outils que les notes de calcul du
site (ndc_pdf : reportlab, les polices du site), prix HTVA :

- le rapport d'UNE grue (1 à 2 pages) : l'en-tête (projet, repère, grue, loueur, catalogue, dates, date d'impression,
  écusson), l'encadré « Ce qui est calculé » de sa logique, les hypothèses (mois, jours facturés, jours du grutier,
  heures par jour, châssis, conduite, fondation, prix du kWh avec son année), le tableau détaillé (une ligne par poste
  retenu : poste, quantité, unité, prix, quantité facturée, total, source et écusson), les sous-totaux matériel et
  main-d'œuvre, le total, puis les prix forcés avec leur justification, les prix manquants et les points d'attention ;
- le rapport du PROJET : une page de synthèse (une ligne par grue : repère, grue, loueur, période, total, écusson ; le
  total du projet), puis le rapport de chaque grue.

L'écusson est dans l'en-tête, la légende des écussons en bas de chaque page (lisible imprimée en noir et blanc : chaque
écusson a son pictogramme). Rien n'est calculé ici : ``grues.prix.calcul`` rend tout."""
import datetime as dt
import os
import tempfile
from typing import Dict, List, Optional, Sequence

from grues.prix import base as B
from grues.prix import calcul as K
from grues.prix import ecussons as E
from grues.prix import valeurs as V
from ndc_pdf.kit import Doc, Frame, hx, mix

from . import prix_grues_commun as PC
from . import prix_grues_textes as TX

A4 = (595.28, 841.89)
MARGE = 38.0
F, FB = "Carlito", "Carlito-Bold"
INK, MUT, RULE, ACC, FOND = "#1b2333", "#5d6678", "#d9dee7", "#2f54d6", "#f3f5f9"
COULEURS = {E.A_JOUR: "#1f8a4c", E.PAS_A_JOUR: "#c0392b", E.A_VERIFIER: "#c46a00", E.MANQUANT: "#c0392b", E.CORRIGE: "#2563c9",
            E.DEMANDE: "#5b6f8f"}
ORANGE = "#e67e22"
COLONNES = (("Poste", 178, "left"), ("Qté", 40, "right"), ("Unité", 40, "left"), ("Prix", 56, "right"), ("Facturé", 54, "right"),
            ("Total", 66, "right"), ("Source", 85, "left"))


def _txt(s) -> str:
    """Un texte de l'écran pour le PDF : les espaces fines deviennent des espaces insécables (la police les a)."""
    return str(s if s is not None else "").replace(" ", " ")


# ------------------------------------------------------------------ l'écusson dessiné
def ecusson(d: Doc, x: float, y: float, taille: float, cle: str) -> float:
    """Le bouclier (24 unités ramenées à ``taille`` points) et son pictogramme ; (x, y) : son coin bas gauche. Rend sa largeur."""
    c = d.c
    k = taille / 24.0
    col = COULEURS.get(cle, MUT)

    def P(u, v):
        return x + u * k, y + (24 - v) * k
    c.saveState()
    c.setStrokeColor(hx(col))
    c.setFillColor(mix("#ffffff", col, 0.16))
    c.setLineWidth(1.6 * k)
    c.setLineJoin(1)
    c.setLineCap(1)
    p = c.beginPath()
    p.moveTo(*P(12, 2.2))
    p.lineTo(*P(19.6, 5))
    p.lineTo(*P(19.6, 11.1))
    p.curveTo(*P(19.6, 16), *P(16.4, 19.7), *P(12, 21.8))
    p.curveTo(*P(7.6, 19.7), *P(4.4, 16), *P(4.4, 11.1))
    p.lineTo(*P(4.4, 5))
    p.close()
    c.drawPath(p, stroke=1, fill=1)
    c.setLineWidth(1.9 * k)

    def trait(*pts):
        q = c.beginPath()
        q.moveTo(*P(*pts[0]))
        for pt in pts[1:]:
            q.lineTo(*P(*pt))
        c.drawPath(q, stroke=1, fill=0)
    if cle == E.A_JOUR:
        trait((8.6, 12.1), (11, 14.5), (15.6, 9.6))
    elif cle == E.PAS_A_JOUR:
        cx, cy = P(12, 12)
        c.circle(cx, cy, 4.1 * k, stroke=1, fill=0)
        trait((12, 9.6), (12, 12), (13.7, 13.1))
    elif cle == E.A_VERIFIER:
        trait((12, 8.2), (12, 12.8))
        cx, cy = P(12, 15.6)
        c.setFillColor(hx(col))
        c.circle(cx, cy, 1.1 * k, stroke=0, fill=1)
    elif cle == E.MANQUANT:
        d.t(x + 12 * k, y + 7.2 * k, "?", FB, 11.5 * k, col, "center")
    elif cle == E.CORRIGE:
        trait((9, 15.3), (9.6, 12.7), (13.9, 8.4), (15.9, 10.4), (11.6, 14.7), (9, 15.3))
        trait((8.4, 16.2), (11.7, 16.2))
    elif cle == E.DEMANDE:
        trait((8, 9.2), (16, 9.2), (16, 15), (8, 15), (8, 9.2))
        trait((8.3, 9.6), (12, 12.4), (15.7, 9.6))
    c.restoreState()
    return taille


def ecusson_mot(d: Doc, x: float, y: float, cle: str, taille: float = 12, size: float = 8.5, raison: str = "") -> float:
    """L'écusson, son mot (et sa raison) ; rend la largeur dessinée."""
    if not cle:
        return 0.0
    ecusson(d, x, y - taille * 0.28, taille, cle)
    mot = E.LIBELLES.get(cle, cle)
    d.t(x + taille + 3, y, mot, FB, size, COULEURS.get(cle, MUT))
    w = taille + 3 + d.w(mot, FB, size)
    if raison:
        d.t(x + w + 5, y, _txt(raison), F, size, MUT)
        w += 5 + d.w(_txt(raison), F, size)
    return w


# ------------------------------------------------------------------ la page
class Page:
    """Une page du rapport : son en-tête, son pied (la légende des écussons, le numéro), une colonne de composition."""

    def __init__(self, d: Doc, titre: str, sous: str, cle: Optional[str], raison: str):
        self.d, self.titre, self.sous, self.cle, self.raison = d, titre, sous, cle, raison
        self.fr = None
        self.nouvelle()

    def nouvelle(self):
        d = self.d
        d.new_page(A4)
        W, H = A4
        d.box(0, H - 62, W, 62, fill=FOND)
        d.t(MARGE, H - 30, _txt(self.titre), FB, 15, INK)
        d.t(MARGE, H - 47, _txt(self.sous), F, 9, MUT)
        if self.cle:
            w = d.w(E.LIBELLES.get(self.cle, ""), FB, 9) + 18
            ecusson_mot(d, W - MARGE - w, H - 30, self.cle, 14, 9)
            if self.raison:
                d.t(W - MARGE, H - 47, _txt(self.raison), F, 8.5, MUT, "right")
        # le pied : la légende des écussons et la page
        y = 26
        d.line(MARGE, y + 14, W - MARGE, y + 14, RULE, 0.5)
        x = MARGE
        for k in (E.A_JOUR, E.A_VERIFIER, E.PAS_A_JOUR, E.MANQUANT, E.CORRIGE):
            x += ecusson_mot(d, x, y, k, 9, 7) + 12
        d.t(W - MARGE, y, f"Prix HTVA · page {d.n}", F, 7.5, MUT, "right")
        self.fr = Frame(MARGE, H - 76, W - 2 * MARGE, 50)

    def place(self, h: float) -> bool:
        """Assez de place pour ``h`` points ; sinon une nouvelle page."""
        if self.fr.y - h < self.fr.bottom:
            self.nouvelle()
            return False
        return True


def _titre_bloc(pg: Page, t: str):
    pg.place(22)
    pg.fr.down(14)
    pg.d.t(pg.fr.x, pg.fr.y, t.upper(), FB, 8, ACC, track=0.6)
    pg.fr.down(4)


def _kv_deux(pg: Page, paires: Sequence):
    """Des paires (libellé, valeur) sur deux colonnes."""
    d = pg.d
    for i in range(0, len(paires), 2):
        pg.place(13)
        fr = pg.fr
        w = fr.w / 2
        fr.down(12)
        for j, (k, v) in enumerate(paires[i:i + 2]):
            x = fr.x + j * w
            d.t(x, fr.y, _txt(k), F, 8.5, MUT)
            d.t(x + w - 10, fr.y, d.fit(_txt(v), FB, 8.5, w * 0.62), FB, 8.5, INK, "right")


def _encadre(pg: Page, titre: str, lignes: List[str]):
    d = pg.d
    larg = pg.fr.w - 16
    corps = [l2 for l in lignes for l2 in d.wrap("• " + _txt(l), F, 8.3, larg)]
    h = 16 + 11 * len(corps) + 6
    pg.place(h + 6)
    fr = pg.fr
    fr.down(6)
    d.box(fr.x, fr.y - h, fr.w, h, fill="#f7f9fd", stroke=RULE, lw=0.6, r=4)
    y = fr.y - 13
    d.t(fr.x + 8, y, _txt(titre), FB, 9, INK)
    for l in corps:
        y -= 11
        d.t(fr.x + 8, y, l, F, 8.3, INK)
    fr.down(h)


def _entete_tableau(pg: Page):
    d, fr = pg.d, pg.fr      # (appelée juste après un saut de page : pg.fr est la nouvelle colonne)
    fr.down(12)
    x = fr.x
    for nom, w, al in COLONNES:
        d.t(x + (w - 3 if al == "right" else 0), fr.y, nom, FB, 7.5, MUT, al)
        x += w
    fr.down(4)
    d.line(fr.x, fr.y, fr.x1, fr.y, ACC, 0.6)


def _cellules(ln: Dict) -> List[str]:
    o = ln["origine"] or {}
    n = V.nature(o.get("valeur"))
    if ln["force"]:
        prix = PC.nombre(ln["force"]["prix"], 3)
    elif n == V.COMPRIS:
        prix = "compris"
    else:
        prix = PC.nombre(ln["prix"], 3) if ln["prix"] is not None else "—"
    q = PC.nombre(ln["quantite"], 2) + (f" {ln['quantite_unite']}" if ln["quantite_unite"] else "")
    if ln["unite"] in ("%", "coef"):
        fac = PC.euros(ln["facturee"], 2)
    elif ln["facturee_unite"] == "" and ln["facturee"] == 1:
        fac = ""
    else:
        fac = f"{PC.nombre(ln['facturee'], 2)} {ln['facturee_unite']}".strip()
    tot = "manquant" if ln["manquant"] else PC.euros(ln["total"], 2)
    return [_txt(ln["libelle"]), _txt(q), _txt(ln["unite"]), _txt(prix), _txt(fac), _txt(tot)]


def _source(ln: Dict) -> tuple:
    """(écusson, texte) de la colonne Source."""
    if ln["force"]:
        return None, "forcé"
    if ln["manquant"]:
        return E.MANQUANT, ln["manquant_raison"]
    o = ln["origine"] or {}
    if ln["code"] == "FOND":
        return ln["ecusson"], (ln["fondation"] or {}).get("source") or ""
    return ln["ecusson"], "corrigé" if o.get("corrige") else (str(o.get("annee")) if o.get("annee") else "année ?")


def _ligne(pg: Page, ln: Dict):
    d = pg.d
    cel = _cellules(ln)
    l0 = d.wrap(cel[0], F, 8.3, COLONNES[0][1] - 6) or [""]
    det = d.wrap(_txt(ln.get("detail") or ""), F, 6.8, COLONNES[0][1] - 6) if ln.get("detail") else []
    h = 11 * len(l0) + 8.5 * len(det) + 3
    if not pg.place(h + 2):
        _entete_tableau(pg)
    fr = pg.fr
    fr.down(10.5)
    y = fr.y
    if ln["force"]:
        d.box(fr.x - 2, y - h + 8.5, fr.w + 4, h, fill=mix("#ffffff", ORANGE, 0.10))
    if ln["manquant"]:
        d.box(fr.x - 2, y - h + 8.5, fr.w + 4, h, fill=mix("#ffffff", "#c0392b", 0.08))
    for i, l in enumerate(l0):
        d.t(fr.x, y - 11 * i, l, F, 8.3, INK)
    for i, l in enumerate(det):
        d.t(fr.x, y - 11 * len(l0) - 8.5 * i + 2, l, F, 6.8, MUT)
    x = fr.x + COLONNES[0][1]
    for (nom, w, al), v in zip(COLONNES[1:6], cel[1:]):
        gras = nom == "Total"
        col = "#c0392b" if (gras and ln["manquant"]) else (ORANGE if (nom == "Prix" and ln["force"]) else INK)
        d.t(x + (w - 3 if al == "right" else 0), y, v, FB if gras or (nom == "Prix" and ln["force"]) else F, 8.3, col, al)
        x += w
    ec, t = _source(ln)
    if ec:
        ecusson(d, x, y - 2.4, 9.5, ec)
    else:
        d.box(x, y - 2.2, 9.5, 9.5, fill=ORANGE, r=2)
        d.t(x + 4.75, y, "F", FB, 7, "#ffffff", "center")
    d.t(x + 12, y, d.fit(_txt(t), F, 7.5, COLONNES[6][1] - 14), F, 7.5, MUT)
    fr.down(h - 10.5)
    d.line(fr.x, fr.y + 1, fr.x1, fr.y + 1, RULE, 0.35)


def _groupe(pg: Page, libelle: str, total: float):
    d = pg.d
    if not pg.place(30):
        _entete_tableau(pg)
    fr = pg.fr
    fr.down(13)
    d.box(fr.x - 2, fr.y - 3.5, fr.w + 4, 13, fill=FOND)
    d.t(fr.x, fr.y, _txt(libelle), FB, 8.3, INK)
    x = fr.x + sum(c[1] for c in COLONNES[:6])
    d.t(x - 3, fr.y, _txt(PC.euros(total)), FB, 8.3, INK, "right")
    fr.down(2)


def _liste(pg: Page, titre: str, items: List[str], couleur: str = INK):
    if not items:
        return
    d = pg.d
    _titre_bloc(pg, f"{titre} ({len(items)})")
    for it in items:
        for l in d.wrap("• " + _txt(it), F, 8.3, pg.fr.w):
            pg.place(11)
            pg.fr.down(11)
            d.t(pg.fr.x, pg.fr.y, l, F, 8.3, couleur)


def _periode_txt(r: Dict) -> str:
    p = r["periode"]
    if r["logique"] == "MA":
        return f"{PC.nombre(p.get('jours'), 1)} j"
    if p.get("debut") and p.get("fin"):
        return f"{_jj(p['debut'])} → {_jj(p['fin'])} · {PC.nombre(p.get('mois'), 1)} mois"
    return f"{PC.nombre(p.get('mois'), 2)} mois"


def _jj(iso: str) -> str:
    a, m, j = iso.split("-")
    return f"{j}/{m}/{a}"


def page_grue(d: Doc, base: Dict, g: Dict, r: Dict, projet: str, date_impr: str):
    titre = f"Prix d'une grue · {g.get('repere') or ''}"
    sous = f"{projet} · {r.get('modele') or ''} · {r.get('loueur') or ''} · imprimé le {date_impr}"
    pg = Page(d, titre, sous, r["ecusson"], r["ecusson_raison"])
    _kv_deux(pg, [("Projet", projet), ("Repère", g.get("repere") or ""), ("Grue", r.get("modele") or ""), ("Loueur", r.get("loueur") or ""),
                  ("Catalogue", r.get("catalogue") or ""), ("Période", _periode_txt(r))])
    contrib = None
    p = B.poste(base, "CONTRIB_ENV")
    if p and r.get("catalogue"):
        x = V.montant((p.get("prix_par_cas") or {}).get(r["catalogue"]))
        contrib = f"{PC.nombre(x, 2)} %" if x is not None else None
    _encadre(pg, f"Ce qui est calculé · {TX.NOMS.get(r['logique'], '')}", TX.ce_qui_est_calcule(r["logique"], contrib))
    _titre_bloc(pg, "Hypothèses")
    _kv_deux(pg, r["hypotheses"])
    _titre_bloc(pg, "Détail du prix")
    _entete_tableau(pg)
    lignes = {ln["code"]: ln for ln in r["lignes"]}
    for gr in r["groupes"]:
        ret = [lignes[c] for c in gr["lignes"] if lignes[c]["retenu"]]
        if not ret:
            continue
        _groupe(pg, gr["libelle"], gr["total"])
        for ln in ret:
            _ligne(pg, ln)
    # les totaux
    pg.place(64)
    fr = pg.fr
    fr.down(16)
    xr = fr.x + sum(c[1] for c in COLONNES[:6]) - 3
    for k, v in (("Matériel", r["materiel"]), ("Main-d'œuvre", r["main_oeuvre"])):
        d.t(xr - 120, fr.y, k, F, 9, MUT)
        d.t(xr, fr.y, _txt(PC.euros(v)), FB, 9, INK, "right")
        fr.down(12)
    fr.down(6)
    d.line(xr - 200, fr.y + 12, xr + 2, fr.y + 12, INK, 0.8)
    d.t(xr - 200, fr.y, "Total HTVA", FB, 11, INK)
    d.t(xr, fr.y, _txt(PC.euros(r["total_arrondi"])), FB, 14, "#c0392b" if r["manquants"] else INK, "right")
    if r["manquants"]:
        n = len(r["manquants"])
        fr.down(13)
        d.t(xr, fr.y, f"INCOMPLET – {n} prix manquant{'s' if n > 1 else ''}", FB, 9, "#c0392b", "right")
    _liste(pg, "Prix forcés", [f"{x['libelle']} : {PC.prix_unitaire(x['prix'])} — {x['justification'] or 'SANS JUSTIFICATION'}" for x in r["forces"]], ORANGE)
    _liste(pg, "Prix manquants", [f"{x['libelle']} ({x['raison']})" for x in r["manquants"]], "#c0392b")
    _liste(pg, "Points d'attention", [a["texte"] for a in r["attention"]])


def page_projet(d: Doc, etat: Dict, calculs: List[Dict], projet: str, date_impr: str):
    ecs = [E.pour_ensemble(r["ecusson"], None, dt.date.today()) if r["ecusson"] != E.MANQUANT else E.MANQUANT for r in calculs]
    manq = sum(len(r["manquants"]) for r in calculs)
    cle = E.MANQUANT if manq else E.agreger(ecs, vide=E.A_VERIFIER)
    pg = Page(d, f"Prix des grues · {projet}", f"Rapport du projet · {len(calculs)} grue{'s' if len(calculs) > 1 else ''} · imprimé le {date_impr}",
              cle, f"{manq} prix manquant{'s' if manq > 1 else ''}" if manq else "")
    fr = pg.fr
    cols = (("Repère", 50, "left"), ("Grue", 150, "left"), ("Loueur", 70, "left"), ("Période", 120, "left"), ("Total", 70, "right"),
            ("Écusson", 59, "left"))
    fr.down(18)
    x = fr.x
    for nom, w, al in cols:
        d.t(x + (w - 3 if al == "right" else 0), fr.y, nom, FB, 7.5, MUT, al)
        x += w
    fr.down(4)
    d.line(fr.x, fr.y, fr.x1, fr.y, ACC, 0.6)
    for g, r in zip(etat["grues"], calculs):
        fr.down(13)
        vals = [g.get("repere") or "", r.get("modele") or "—", r.get("loueur") or "—", _periode_txt(r) if r["pret"] else "—",
                PC.euros(r["total_arrondi"]) if r["pret"] else "—"]
        x = fr.x
        for (nom, w, al), v in zip(cols[:5], vals):
            d.t(x + (w - 3 if al == "right" else 0), fr.y, d.fit(_txt(v), FB if nom == "Total" else F, 8.3, w - 6), FB if nom == "Total" else F, 8.3, INK, al)
            x += w
        if r["pret"]:
            ecusson_mot(d, x, fr.y, r["ecusson"], 9, 7.5)
        fr.down(3)
        d.line(fr.x, fr.y, fr.x1, fr.y, RULE, 0.35)
    fr.down(20)
    tot = sum(r["total_arrondi"] for r in calculs if r["pret"])
    xr = fr.x + sum(c[1] for c in cols[:5]) - 3
    d.t(xr - 200, fr.y, "Total du projet HTVA", FB, 11, INK)
    d.t(xr, fr.y, _txt(PC.euros(tot)), FB, 14, INK, "right")
    if manq:
        fr.down(13)
        d.t(xr, fr.y, f"INCOMPLET – {manq} prix manquant{'s' if manq > 1 else ''}", FB, 9, "#c0392b", "right")


def pdf(base: Dict, etat: Dict, infos: Optional[Dict], projet_entier: bool, aujourdhui: dt.date) -> bytes:
    """Le rapport (octets) : la grue ouverte, ou le projet entier (la synthèse puis chaque grue calculée)."""
    infos = infos or {}
    projet = str(infos.get("chantier") or infos.get("nom") or "Projet").strip() or "Projet"
    date_impr = str(infos.get("date") or "").strip() or aujourdhui.strftime("%d/%m/%Y")
    calculs = [K.calculer(base, g, aujourdhui) for g in etat["grues"]]
    choisies = list(range(len(etat["grues"]))) if projet_entier else [etat["actif"]]
    if not any(calculs[i]["pret"] for i in choisies):
        raise ValueError("choisir d'abord une grue (le rapport n'a rien à imprimer)")
    fd, chemin = tempfile.mkstemp(suffix=".pdf", prefix="prix_grues_")
    os.close(fd)
    try:
        d = Doc(chemin, A4, title=f"Prix des grues - {projet}")
        if projet_entier:
            page_projet(d, etat, calculs, projet, date_impr)
        for i in choisies:
            if calculs[i]["pret"]:
                page_grue(d, base, etat["grues"][i], calculs[i], projet, date_impr)
        d.save()
        with open(chemin, "rb") as f:
            return f.read()
    finally:
        try:
            os.remove(chemin)
        except OSError:
            pass

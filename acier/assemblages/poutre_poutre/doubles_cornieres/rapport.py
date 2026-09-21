# -*- coding: utf-8 -*-
"""Note de calcul PDF de l'assemblage — même système, même charte que les
notes béton (``ndc_pdf``, palette 01_encre) : page de garde portrait avec le
cartouche PROJET / PARTIE / DATE / INDICE et le sommaire, puis des planches
A4 paysage.

- Planche 1 « Synthèse » : à gauche les deux dessins cotés (le même
  générateur que l'écran, peint en vectoriel) et les données ; à droite les
  hypothèses et excentricités, chaque groupe de vérifications (une ligne par
  vérification active : sollicitation / résistance · taux · statut, verdict
  du groupe) et la synthèse ;
- Planche 2 « Développement » : à gauche synthèse, paramètres retenus,
  efforts par boulon, pinces et entraxes ; à droite les vérifications
  essentielles développées (formule, valeurs introduites, sollicitation /
  résistance, taux, référence et nature).

Une boucle d'ajustement (2 puis 3 colonnes, puis moins de détail) garantit
que rien ne déborde, sans jamais réduire le corps de texte sous celui des
notes béton. Le rapport lit les résultats du moteur : aucune valeur n'est
recalculée ici.
"""
import os
import tempfile
from datetime import datetime

from reportlab.lib.colors import HexColor, Color

from ndc_pdf.fonts import draw_text
from ndc_pdf.styles import Encre
from ndc_pdf import data as ndc_data
from acier.formats import F, pct
from acier.js import N
from acier.bibliotheques import PERSO
from . import schemas, texte
from .moteur import GROUPES_VERIF

TITRE_ASSEMBLAGE = "Assemblage poutre–poutre"
SOUS_TITRE = "Doubles cornières d'âme"

# Libellés courts des vérifications (colonne étroite de la planche)
COURT = {
    "bv_S": "Boulons S – cisaillement", "bv_P": "Boulons P – cisaillement",
    "bt_P": "Boulons P – traction", "bi_P": "Boulons P – interaction V + T",
    "gl_S": "Boulons S – glissement", "gl_P": "Boulons P – glissement",
    "pdB": "Ailes B – pression diamétrale", "pdA": "Ailes A – pression diamétrale",
    "cgB": "Ailes B – cisaillement brut", "cnB": "Ailes B – cisaillement net",
    "cbB": "Ailes B – rupture de bloc", "flB": "Ailes B – flexion dans le plan",
    "cgA": "Ailes A – cisaillement brut", "cnA": "Ailes A – cisaillement net",
    "cbA": "Ailes A – rupture de bloc", "flA": "Ailes A – flexion talon–file",
    "tsA": "Ailes A – tronçon en T", "tnB": "Ailes B – traction nette",
    "tbB": "Ailes B – bloc en traction",
    "pdS": "Âme S – pression diamétrale", "vgS": "Âme S – cisaillement brut",
    "vnS": "Âme S – cisaillement net", "vbS": "Âme S – rupture de bloc",
    "mN": "Grugeage – flexion + V", "m2": "Grugeage – 2e file",
    "stN": "Grugeage – stabilité (ln)", "stD": "Grugeage – stabilité (dn)",
    "tnS": "Âme S – traction nette", "tbS": "Âme S – bloc en traction",
    "pdP": "Âme P – pression diamétrale", "vlP": "Âme P – cisaillement local brut",
    "vnP": "Âme P – cisaillement local net",
    "wS": "Cordons B – effort critique", "waS": "Cordons B – gorge minimale",
    "wlS": "Cordons B – longueur minimale", "wP": "Cordons A – effort critique",
    "waP": "Cordons A – gorge minimale", "wlP": "Cordons A – longueur minimale",
}


def _court(c):
    return COURT.get(c.key, c.label)


# ===================================================================
#  Peintre ReportLab de la scène (la même que le SVG)
# ===================================================================
def _couleur(hexa, alpha=1.0):
    c = HexColor(hexa)
    return Color(c.red, c.green, c.blue, alpha=alpha) if alpha < 1 else c


def _police(st):
    if st["bold"]:
        return "Carlito-Bold"
    if st["italic"]:
        return "Carlito-Italic"
    return "Carlito"


def _appliquer_style(c, st, s):
    """Remplissage et trait d'après le style résolu ; retourne (fill, stroke)."""
    fill = st["fill"] != "none"
    stroke = st["stroke"] != "none"
    if fill:
        c.setFillColor(_couleur(st["fill"], st["fo"]))
    if stroke:
        c.setStrokeColor(_couleur(st["stroke"]))
        c.setLineWidth(max(0.25, st["sw"] * 0.45))
        if st["dash"]:
            c.setDash([st["dash"][0] * 0.6, st["dash"][1] * 0.6])
        else:
            c.setDash()
    return fill, stroke


def _peindre_prim(c, p, ctx, s, X, Y, palette):
    t = p["t"]
    if t == "g":
        sous = tuple(ctx) + tuple(p["cls"])
        if "x" in p:
            c.saveState()
            c.translate(X(p["x"]), Y(p["y"]))
            if p.get("rot"):
                c.rotate(90)          # rotate(-90) en repère y vers le bas
            for e in p["enfants"]:
                _peindre_prim(c, e, sous, s, lambda px: px * s, lambda py: -py * s, palette)
            c.restoreState()
        else:
            for e in p["enfants"]:
                _peindre_prim(c, e, sous, s, X, Y, palette)
        return
    st = schemas.style_de(p["cls"], ctx, palette)
    if t == "text":
        x = X(p["x"]) if p.get("x") is not None else X(0)
        y = Y(p["y"]) if p.get("y") is not None else Y(0)
        align = "center" if p.get("anchor") == "middle" or p.get("x") is None else "left"
        draw_text(c, x, y, p["txt"], _police(st), max(3.0, p["size"] * s), _couleur(st["tfill"]), align)
        return
    fill, stroke = _appliquer_style(c, st, s)
    if not fill and not stroke:
        return
    if t == "rect":
        x1, y1 = X(p["x"]), Y(p["y"] + p["h"])
        w, h = p["w"] * s, p["h"] * s
        if p.get("rx"):
            c.roundRect(x1, y1, w, h, p["rx"] * s, stroke=int(stroke), fill=int(fill))
        else:
            c.rect(x1, y1, w, h, stroke=int(stroke), fill=int(fill))
    elif t == "poly":
        pth = c.beginPath()
        for i, (x, y) in enumerate(p["pts"]):
            (pth.moveTo if i == 0 else pth.lineTo)(X(x), Y(y))
        pth.close()
        c.drawPath(pth, stroke=int(stroke), fill=int(fill))
    elif t == "path":
        pth = c.beginPath()
        for l in p["lignes"]:
            for i, (x, y) in enumerate(l):
                (pth.moveTo if i == 0 else pth.lineTo)(X(x), Y(y))
        c.drawPath(pth, stroke=int(stroke), fill=0)
    elif t == "circle":
        c.circle(X(p["cx"]), Y(p["cy"]), p["r"] * s, stroke=int(stroke), fill=int(fill))
    c.setDash()


def peindre(d, dessin, x, y, w, h, palette=None, marge=3):
    """Peint une vue dans le rectangle ``(x, y, w, h)`` du canevas (origine
    en bas à gauche), à l'échelle et centrée."""
    palette = palette or schemas.PALETTE
    c = d.c
    vx, vy, vw, vh = dessin.viewbox
    s = min((w - 2 * marge) / vw, (h - 2 * marge) / vh)
    ox = x + (w - vw * s) / 2 - vx * s
    oy = y + h - (h - vh * s) / 2 + vy * s

    def X(px):
        return ox + px * s

    def Y(py):
        return oy - py * s

    c.saveState()
    for p in dessin.corps:
        _peindre_prim(c, p, (), s, X, Y, palette)
    for p in dessin.cotes:
        _peindre_prim(c, p, (), s, X, Y, palette)
    for p in dessin.cartouche:
        _peindre_prim(c, p, ("cart",), s, X, Y, palette)
    c.restoreState()
    return s


def dessinateur(R):
    """Le dessinateur d'une planche : élévation (60 %) et plan (40 %) côte à
    côte, cotations principales, sans interaction."""
    opt = schemas.options_rapport()
    e = schemas.elevation(R, opt)
    p = schemas.plan(R, opt)

    def dessin(d, x, y, w, h, style):
        we = w * 0.6
        peindre(d, e, x, y, we - 4, h)
        peindre(d, p, x + we + 4, y, w - we - 4, h)
    return dessin


# ===================================================================
#  Contenu : blocs et vérifications
# ===================================================================
def _prof_txt(R, X):
    u = R.u
    if u["prof_" + X] == PERSO:
        return "h " + F(R["h_" + X], 0) + " b " + F(R["b_" + X], 0) + " tw " + F(R["tw_" + X], 1) + " tf " + F(R["tf_" + X], 1)
    return u["prof_" + X]


def _blocs_synthese(R):
    u = R.u
    nt, nb = N(u.d_nt), N(u.d_nb)
    profs = [("Principale", None, _prof_txt(R, "P"), ""),
             ("Secondaire", None, _prof_txt(R, "S"), ""),
             ("Nuances", None, f"{u.nu_P} / {u.nu_S} / {u.nu_C}", "")]
    if nt or nb:
        profs.append(("Grugeage", None, ("sup. " + F(nt, 0) if nt else "") + (" / " if nt and nb else "")
                      + ("inf. " + F(nb, 0) if nb else "") + " × " + F(N(u.l_n), 0), "mm"))
    else:
        profs.append(("Grugeage", None, "aucun", ""))
    profs.append(("Jeu / décalage", None, F(N(u.g_h), 0) + " / " + F(N(u.d_top), 0), "mm"))
    att = [("Cornières", None, "2 × " + R.corn_txt, ""),
           ("Longueur", "L_c", F(R.L_C, 0), "mm"),
           ("Dessus des cornières", "z_c", F(N(u.z_C), 0), "mm")]
    if R.bolt_S or R.bolt_P:
        att.append(("Boulons", None, f"{R.boulon} cl. {u.classe} cat. {u.cat} – d0 {F(R.d_0, 0)}", "mm"))
    if R.bolt_S:
        att.append(("Groupe S", None, f"{R.n1_S} × {R.n2_S} – p1 {F(R.p1_S, 0)} / e1 {F(R.e1_S, 0)}", "mm"))
    else:
        att.append(("Cordons ailes B", None, "a " + F(N(u.a_S), 0) + ", retours " + F(N(u.lh_S), 0), "mm"))
    if R.bolt_P:
        att.append(("Groupe P", None, f"2 × ({R.n1_P} × {R.n2_P}) – gA {F(R.g_A, 0)}", "mm"))
    else:
        att.append(("Cordons ailes A", None, "a " + F(N(u.a_P), 0) + ", retours " + F(N(u.lh_P), 0), "mm"))
    eff = [(None, "V_{Ed}", F(u.V_Ed, 1), "kN"), (None, "N_{Ed}", F(u.N_Ed, 1), "kN"),
           (None, "H_{Ed}", F(u.H_Ed, 1), "kN"), (None, "M_{Ed}", F(u.M_Ed, 2), "kNm")]
    blocs = [("PROFILÉS", profs), ("ATTACHE", att), ("EFFORTS ELU", eff)]
    ident = [(lab, None, str(val), "") for lab, val in (("Projet", u.id_projet), ("Repère", u.id_rep),
                                                          ("Rédacteur", u.id_red), ("Date", u.id_date)) if val]
    if ident:
        blocs.append(("IDENTIFICATION", ident))
    return blocs


def _pire(checks):
    """La vérification au taux le plus élevé (première en cas d'égalité)."""
    pire = None
    for c in checks:
        if pire is None or c.eta > pire.eta:
            pire = c
    return pire


def _verifs_synthese(R):
    u = R.u
    verifs = []
    n = 1
    hyp = texte.lignes_hypotheses(R)
    items = [("v", "Excentricité de calcul", "z", F(R.zeff, 1), "mm"),
             ("v", "Moment d'excentricité", "M_S", F(R.M_S, 2), "kNm")]
    if R.M_P > 0:
        items.append(("v", "Moment par groupe P (option)", "M_P", F(R.M_P, 2), "kNm"))
    if R.bolt_S:
        items.append(("v", "Inertie polaire du groupe S", "I_p", F(R.Ip_S, 0), "mm²"))
    items += [("p", l) for l in hyp[:4]]
    items.append(("p", hyp[5]))
    if not R.bolt_S or not R.bolt_P:
        items.append(("p", "Cordons : longueur efficace prise égale à la longueur totale (cordon vertical "
                           "Lc + retours), hypothèse de cordons pleins sur toute leur longueur – EN 1993-1-8 §4.5.1."))
    verifs.append(dict(num=n, titre="Hypothèses et excentricités", items=items, verdicts=[]))
    n += 1
    for g in GROUPES_VERIF:
        on = [c for c in R.checks if c.grp == g and c.active]
        if not on:
            continue
        items = []
        for c in on:
            nd = 3 if c.unit == "-" else 1
            val = (F(c.Ed, nd) + " / " + F(c.Rd, nd) + ("" if c.unit == "-" else " " + c.unit)
                   + " · " + pct(c.eta, 1) + ("" if c.ok else " NON OK"))
            items.append(("v", _court(c), None, val, ""))
        pire = _pire(on)
        ok = all(c.ok for c in on)
        items.append(("k", 0))
        verdicts = [dict(etat="ok" if ok else "ko",
                         texte=(g + " : vérifié" if ok else g + " : NON VÉRIFIÉ")
                         + " – taux maximal " + pct(pire.eta, 1) + " (" + _court(pire) + ")")]
        verifs.append(dict(num=n, titre=g, items=items, verdicts=verdicts))
        n += 1
    items = [("v", "Vérification dimensionnante", None, _court(R.gov) if R.gov else "—", ""),
             ("v", "Taux maximal", None, pct(R.eta_max, 1), ""),
             ("v", "Géométrie / Tableau 3.3", None,
              ("valide" if R.geo_ok else "NON VALIDE") + " / " + ("conforme" if R.dist_ok else "NON CONFORME"), "")]
    for a in R.alerts:
        items.append(("p", ("Bloquant : " if a.block else "Attention : ") + a.msg))
    items.append(("k", 0))
    verdicts = [dict(etat="ok" if R.verified else "ko",
                     texte=R.statut + (" – sous réserve des points signalés" if R.verified and (R.reserve or R.alerts) else "")
                     + " – ELU, EN 1993-1-8 et EN 1993-1-1")]
    verifs.append(dict(num=n, titre="Synthèse", items=items, verdicts=verdicts))
    return verifs


def _blocs_developpement(R):
    u = R.u
    synth = [("Statut", None, "Vérifié" if R.verified else "Non vérifié", ""),
             ("Taux maximal", None, pct(R.eta_max, 1), ""),
             ("Géométrie", None, "valide" if R.geo_ok else "NON VALIDE", ""),
             ("Tableau 3.3", None, "conforme" if R.dist_ok else "NON CONFORME", "")]
    params = [("Boulon retenu", None, R.boulon + " – d0 " + F(R.d_0, 0), "mm"),
              ("Cornières", None, R.corn_txt, ""),
              ("Ailes A / B, tc, rc", None, F(R.b_A, 0) + " / " + F(R.b_B, 0) + ", " + F(R.t_C, 0) + ", " + F(R.r_C, 0), "mm"),
              ("Longueur des cornières", "L_c", F(R.L_C, 0) + ("" if R.rec_L else " (< 0,6 h)"), "mm"),
              ("Excentricité", "z", F(R.zeff, 1), "mm"),
              ("Moment d'excentricité", "M_S", F(R.M_S, 2), "kNm"),
              ("Moment par groupe P", "M_P", F(R.M_P, 2), "kNm"),
              ("Inertie polaire groupe S", "I_p", F(R.Ip_S, 0), "mm²"),
              ("Entraxe des files", "p_3", F(R.p_3, 1), "mm"),
              ("Pince e1,b (âme S)", None, F(R.e1b_S, 1), "mm"),
              ("Pinces e2 aile B / A", None, F(R.e2a_S, 1) + " / " + F(R.e2a_P, 1), "mm"),
              ("Pinces basses S / P", None, F(R.e1bot_S, 1) + " / " + F(R.e1bot_P, 1), "mm")]
    # disposition de _data : blocs pairs en colonne étroite, impairs en
    # colonne large — les libellés longs (paramètres, pinces) vont à droite
    blocs = [("SYNTHÈSE", synth), ("PARAMÈTRES RETENUS", params)]
    if R.bolt_S:
        blocs.append(("EFFORTS PAR BOULON – GROUPE S",
                      [(f"Rangée {b.i} / file {b.j}", None, F(b.f, 1), "kN") for b in R.tabS[:12]]))
    else:
        blocs.append(("CONTRÔLES", [
            ("Alertes bloquantes", None, str(sum(1 for a in R.alerts if a.block)), ""),
            ("Alertes informatives", None, str(sum(1 for a in R.alerts if not a.block)), ""),
            ("Pinces non conformes", None, str(sum(1 for x in R.dist if not x.ok)), "")]))
    if R.dist:
        pinces = []
        for x in R.dist:
            lab = x.lab.split(" (")[0]
            val = F(x.val, 1) + " ≥ " + F(x.min, 1) + (" ≤ " + F(x.max, 0) if x.max is not None else "")
            pinces.append((lab + ("" if x.ok else " – NON OK"), None, val, "mm"))
        blocs.append(("PINCES ET ENTRAXES – TABLEAU 3.3", pinces))
    return blocs


def _verifs_developpement(R, detail):
    """Les vérifications essentielles développées. ``detail`` :
    « complet » (formule + valeurs), « sans_vals » (formule seule, valeurs
    pour la dimensionnante), « formules » (formule seule), « compact »
    (une ligne par vérification)."""
    ess = [c for c in R.checks if c.active and (c.ess or not c.ok)]
    verifs = []
    if detail == "compact":
        items = []
        for c in ess:
            nd = 3 if c.unit == "-" else 1
            items.append(("v", _court(c), None, F(c.Ed, nd) + " / " + F(c.Rd, nd)
                          + ("" if c.unit == "-" else " " + c.unit) + " · " + pct(c.eta, 1)
                          + " · " + ("OK" if c.ok else "NON OK"), ""))
        return [dict(num=1, titre="Vérifications essentielles", items=items, verdicts=[])]
    for i, c in enumerate(ess, start=1):
        nd = 3 if c.unit == "-" else 1
        items = [("p", c.formula)]
        if c.vals and (detail == "complet" or (detail == "sans_vals" and R.gov is c)):
            items.append(("p", c.vals))
        items.append(("v", "Sollicitation / résistance", None,
                      F(c.Ed, nd) + " / " + F(c.Rd, nd), "" if c.unit == "-" else c.unit))
        items.append(("k", 0))
        verifs.append(dict(num=i, titre=_court(c), items=items,
                           verdicts=[dict(etat="ok" if c.ok else "ko",
                                          texte="η = " + pct(c.eta, 1) + " – " + ("OK" if c.ok else "NON OK")
                                          + " · " + c.ref + " [" + c.nat + "]")]))
    return verifs


def _etat(R):
    return "Vérifié" if R.verified else "Non vérifié"


def _materiaux(R):
    u = R.u
    txt = u.nu_S if u.nu_S == u.nu_P else f"{u.nu_P} / {u.nu_S}"
    if R.bolt_S or R.bolt_P:
        txt += f" · {R.boulon} {u.classe}"
    return txt


def section_synthese(R, n_cols=2):
    return dict(poutre=TITRE_ASSEMBLAGE, section=SOUS_TITRE + " – synthèse", beton="", acier=_materiaux(R),
                etat=_etat(R), titre_coupe="ÉLÉVATION ET VUE EN PLAN", dessin=dessinateur(R),
                coupe_h=290, blocs=_blocs_synthese(R), verifs=_verifs_synthese(R), n_cols=n_cols)


def section_developpement(R, n_cols=2, detail="complet"):
    return dict(poutre=TITRE_ASSEMBLAGE, section=SOUS_TITRE + " – développement", beton="", acier=_materiaux(R),
                etat=_etat(R), titre_coupe="PARAMÈTRES ET CONTRÔLES", blocs=_blocs_developpement(R),
                verifs=_verifs_developpement(R, detail), n_cols=n_cols)


# Variantes d'ajustement de chaque planche, de la plus détaillée à la plus
# compacte — le corps de texte n'est jamais réduit
VARIANTES_SYNTHESE = (2, 3)
VARIANTES_DEVELOPPEMENT = ((2, "complet"), (3, "complet"), (3, "sans_vals"), (3, "formules"), (3, "compact"))


def _essai(sections_, doc_meta):
    """Construit une note d'essai (sans garde) et rend ses avertissements."""
    fd, chemin = tempfile.mkstemp(suffix=".pdf", prefix="essai_assemblage_")
    os.close(fd)
    try:
        d = Encre().build(chemin, sections=sections_, doc=doc_meta, garde=False)
        d.save()
        return list(d.warnings)
    finally:
        try:
            os.remove(chemin)
        except OSError:
            pass


def generer_pdf(R, infos=None, garde=True, chemin=None):
    """La note (bytes). Chaque planche est ajustée séparément : la synthèse
    en 2 puis 3 colonnes ; le développement en 2 puis 3 colonnes, puis avec
    moins de détail. La première variante sans débordement est retenue."""
    global derniers_avertissements, derniere_variante
    infos = infos or {}
    doc_meta = ndc_data.construire_doc(infos, date_defaut=datetime.today().strftime("%d/%m/%Y"))
    doc_meta["titre"] = "Note de calcul"
    n1 = VARIANTES_SYNTHESE[-1]
    for n in VARIANTES_SYNTHESE:
        if not _essai([section_synthese(R, n)], doc_meta):
            n1 = n
            break
    n2, detail = VARIANTES_DEVELOPPEMENT[-1]
    for n, det in VARIANTES_DEVELOPPEMENT:
        if not _essai([section_developpement(R, n, det)], doc_meta):
            n2, detail = n, det
            break
    if chemin is None:
        fd, chemin = tempfile.mkstemp(suffix=".pdf", prefix="note_assemblage_")
        os.close(fd)
    d = Encre().build(chemin, sections=[section_synthese(R, n1), section_developpement(R, n2, detail)],
                      doc=doc_meta, garde=garde)
    d.save()
    derniere_variante = (n1, n2, detail)
    derniers_avertissements = list(d.warnings)
    with open(chemin, "rb") as fh:
        return fh.read()


derniers_avertissements = []
derniere_variante = None

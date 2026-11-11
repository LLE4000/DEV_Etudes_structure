# -*- coding: utf-8 -*-
"""Les dessins et les vues en tableau du module « Charges sur plancher » sur le site.

Les DESSINS sont ceux du moteur, à l'échelle (plancher/charges/dessin.py : le plan, la coupe sous un appui, le bandeau
du balayage — les mêmes que l'écran d'origine et la note PDF). Le site les HABILLE seulement :

- le plan est rendu en mode « impression » (sans poignées ni invites de clic : sur le site, les positions se saisissent
  dans le cadre Données) ; les infobulles et attributs d'interaction du composant cliquable sont retirés ;
- tout texte venu de l'état (noms d'engins, d'appuis, de plateaux, de filières : un fichier peut en apporter
  n'importe lesquels) est ÉCHAPPÉ avant d'entrer dans le dessin — le moteur les écrit tels quels ;
- les identifiants internes du SVG (motifs de hachure, flèches de cote) reçoivent un préfixe par vue : plusieurs
  dessins vivent dans la même page ;
- le code couleur est celui du site (rouge au-delà de 100 %, vert foncé de 95 à 100 %) : l'orange de l'écran
  d'origine (seuil d'attention 85 %) devient le vert foncé du bureau, au seuil de 95 % (C.niveau).

Les VUES EN TABLEAU (méthode, bibliothèque, détail des vérifications) sont des blocs HTML que la toile affiche à la
place d'un dessin ; leurs styles sont dans site/web/css/modules/plancher.css.
"""
import html
import re
from typing import Dict, List, Optional

from plancher.charges import bibliotheque as BIB
from plancher.charges import dessin as DS
from plancher.charges import methode as MET
from plancher.charges import symboles as SY

from . import commun as C

ORANGE, ORANGE_F = DS.ORANGE, DS.ORANGE_F
VERT_FONCE, VERT_PALE = "#15803d", "#dcfce7"          # --att et --ok-bg de la charte
SEUILS_SITE = (0.95, 1.0)                              # le code couleur du site : vert foncé dès 95 %, rouge au-delà de 100 %
_TITRE = re.compile(r"<title>.*?</title>", re.S)
_INTERACTION = re.compile(r'\s(?:class="ed(?: [a-z]+)?"|data-(?:group|drag|drag2|dsym|dsym2|key|sym|action)="[^"]*")')
_RACINE = re.compile(r"<svg ([^>]*?)>", re.S)


def echapper(x):
    """Une copie de ``x`` dont chaque chaîne (clés comprises) est échappée pour le HTML : les noms venus d'un fichier
    entrent ainsi dans le SVG comme du texte. Les valeurs de logique du moteur (« hourdis », « ponctuel », « x »…)
    n'ont aucun caractère spécial : l'échappement les laisse intactes."""
    if isinstance(x, str):
        return html.escape(x, quote=True)
    if isinstance(x, dict):
        return {echapper(k) if isinstance(k, str) else k: echapper(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(echapper(v) for v in x)
    return x


def _habiller(svg: str, prefixe: str, ancien: str, titre: str) -> str:
    """Retire l'interaction, préfixe les identifiants, passe au code couleur du site et pose un titre accessible."""
    s = _TITRE.sub("", svg)
    s = _INTERACTION.sub("", s)
    s = s.replace(DS.STYLE, "<style>text{font-family:Inter,system-ui,sans-serif}</style>")
    s = s.replace(f'id="{ancien}', f'id="{prefixe}_').replace(f"url(#{ancien}", f"url(#{prefixe}_")
    s = s.replace(ORANGE_F, VERT_PALE).replace(ORANGE, VERT_FONCE)
    s = s.replace(ORANGE_F.upper(), VERT_PALE).replace(ORANGE.upper(), VERT_FONCE)

    def racine(m):
        attrs = re.sub(r'\s*(?:width|height|style)="[^"]*"', "", m.group(1))
        return f'<svg {attrs} role="img" aria-label="{C.esc(titre)}">'
    return _RACINE.sub(racine, s, count=1)


def _rayon_etiquette(ctx: Dict) -> None:
    """Le nom d'un engin s'écrit sous son symbole, à ``rayon_etiquette`` (mm écran) de son centre ; etat.contexte_dessin
    le place à |dy| + 900 mm, ce qui le faisait chevaucher les étiquettes « P1 · 45 kN » des appuis du bas (capture du
    07/11/2026). Le site le descend sous la plus basse de ces étiquettes, avec la taille de texte du plan (k)."""
    v = DS.Vue(ctx["L"], ctx["B"], ctx.get("sens", 1))
    k = max(1.0, v.largeur_svg / DS.LARGEUR_REF)
    for en in ctx.get("engins", []):
        _, cy = v.s(en["x"], en["y"])
        bas = 0.0
        for ap in en.get("empreintes", []):
            _, y = v.s(ap["x"], ap["y"])
            _, h = v.dims(ap["a"], ap["b"])
            bas = max(bas, y - cy + max(h / 2, 300) + 120 + 340 * k)       # bas de l'étiquette de l'appui (dessin._etiquette)
        en["rayon_etiquette"] = max(en.get("rayon_etiquette", 1600), bas + 60 + DS.FS_S * k)


def plan(ctx: Dict, prefixe: str) -> str:
    """Le plan du plancher à l'échelle (mm), statique : appuis, éléments teintés par leur taux, engins à l'échelle de
    leur fiche, semelles et charges par appui, plateaux et aires efficaces, filières, charges réparties, cotes."""
    c = echapper(dict(ctx))
    c["impression"] = True
    c["seuils"] = SEUILS_SITE
    _rayon_etiquette(c)
    svg, _, _ = DS.plan(c)
    return _habiller(_legende_plan(svg, c), prefixe, "chp_", "Plan du plancher")


def _legende_plan(svg: str, ctx: Dict) -> str:
    """La légende des symboles du plan (celle que l'écran d'origine écrit sous son plan : semelle, plateau, aire efficace,
    filière, couleurs des éléments), dessinée avec les MÊMES symboles, sur une ligne de plus au bas du dessin."""
    v = DS.Vue(ctx["L"], ctx["B"], ctx.get("sens", 1))
    k = max(1.0, v.largeur_svg / DS.LARGEUR_REF)
    y_appui = DS.OY + v.H + (280 if v.sens == 1 else 1000) + 290 + 330 * k        # la ligne « ● appui » de dessin.plan
    y = y_appui + 380 * k
    fs, s = DS.FS_S * k, 170 * k
    ns = DS.NS
    items = [(f'<rect x="{{x}}" y="{{y}}" width="{s:.0f}" height="{s:.0f}" fill="{DS.NOIR}"/>', s, "semelle"),
             (f'<rect x="{{x}}" y="{{y}}" width="{s:.0f}" height="{s:.0f}" fill="url(#chp_hc)" stroke="{DS.NOIR}" stroke-width="1" {ns}/>', s, "plateau"),
             (f'<rect x="{{x}}" y="{{y}}" width="{s:.0f}" height="{s:.0f}" fill="none" stroke="{DS.BLEU}" stroke-width="1" stroke-dasharray="40 28" {ns}/>',
              s, "aire efficace"),
             (f'<rect x="{{x}}" y="{{y}}" width="{2.2 * s:.0f}" height="{0.55 * s:.0f}" fill="#cbd5e1" stroke="{DS.NOIR}" stroke-width="1" {ns}/>',
              2.2 * s, "filière"),
             (f'<rect x="{{x}}" y="{{y}}" width="{1.6 * s:.0f}" height="{s:.0f}" fill="{DS.ORANGE_F}" stroke="{DS.ORANGE}" stroke-width="1" {ns}/>',
              1.6 * s, "élément 95–100 %"),
             (f'<rect x="{{x}}" y="{{y}}" width="{1.6 * s:.0f}" height="{s:.0f}" fill="{DS.ROUGE_F}" stroke="{DS.ROUGE}" stroke-width="1" {ns}/>',
              1.6 * s, "&gt; 100 %")]
    gap, esp = 90 * k, 330 * k
    total = sum(w + gap + len(t.replace("&gt;", ">")) * fs * 0.56 + esp for _, w, t in items) - esp
    sc = min(1.0, (v.largeur_svg - 600) / total)
    x = max(300.0, DS.OX if DS.OX + total * sc <= v.largeur_svg - 300 else 300.0)
    out = [f'<g transform="translate({x:.0f} {y:.0f}) scale({sc:.4f})">']
    cx = 0.0
    for forme, w, t in items:
        h = s if "filière" not in t else 0.55 * s
        out.append(forme.format(x=f"{cx:.0f}", y=f"{-0.36 * fs - h / 2:.0f}"))
        cx += w + gap
        out.append(f'<text x="{cx:.0f}" y="0" font-size="{fs:.0f}" fill="{DS.GRIS}">{t}</text>')
        cx += len(t.replace("&gt;", ">")) * fs * 0.56 + esp
    out.append("</g>")
    h_neuf = y + 260 * k
    svg = re.sub(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', lambda m: f'viewBox="0 0 {m.group(1)} {max(float(m.group(2)), h_neuf):.0f}"', svg, count=1)
    return svg.replace("</svg>", "".join(out) + "</svg>", 1)


def coupe(ctx: Dict, prefixe: str) -> str:
    """La coupe transversale sous un appui : éléments voisins, couche de compression, plateau, semelle, charge,
    diffusion à 45° et aire efficace."""
    return _habiller(DS.coupe(echapper(dict(ctx))), prefixe, "cpc_", "Coupe sous l'appui")


def bandeau(points: List[Dict], pire: Optional[Dict], x_actuel: float, L: float, prefixe: str) -> str:
    """Le bandeau du balayage : taux du plancher entier selon la position de l'engin, position actuelle, la pire."""
    return _habiller(DS.balayage(points, pire, x_actuel, L), prefixe, "bal_", "Balayage de la position de l'engin")


def pictogramme(gabarit: Dict, w: int = 84, h: int = 54) -> str:
    """La vignette d'un engin (symbole du moteur, à l'échelle de sa fiche)."""
    return SY.pictogramme(echapper(gabarit), w, h)


# ------------------------------------------------------------------ les vues en tableau
def vue_methode(gabarits: Optional[Dict[str, Dict]] = None, dispo: Optional[Dict[str, Dict]] = None,
                hourdis: Optional[Dict[str, Dict]] = None) -> str:
    """La méthode : les sections de plancher/charges/methode.py (les MÊMES textes que l'onglet Méthode de l'écran
    d'origine et la page 2 de la note), les références, ce qui reste à transcrire, puis — comme dans l'onglet Méthode
    de l'écran d'origine — la bibliothèque livrée et ajoutée avec ses sources."""
    out = '<div class="chp-doc chp-large"><p class="chp-chap">Ce que l\'outil vérifie, avec la décision du bureau que chaque règle applique (P-n de l\'étude).</p>'
    for sec in MET.SECTIONS:
        out += f'<h4>{C.esc(sec["titre"])}</h4>' + "".join(f"<p>{C.symb_math(_fr_txt(p))}</p>" for p in sec["paragraphes"])
    out += f'<h4>Références</h4><p>{C.esc(" · ".join(MET.REFERENCES))}</p>'
    out += '<h4>À transcrire avant un usage en production</h4><ul>' + "".join(f"<li>{C.esc(_fr_txt(x))}</li>" for x in MET.A_TRANSCRIRE) + "</ul>"
    if gabarits is not None:
        out += "<h4>Bibliothèque et sources</h4>" + _tableau_bibliotheque(gabarits, dispo or {}, hourdis or {})
    return out + "</div>"


def _tableau_bibliotheque(gabarits: Dict[str, Dict], dispo: Dict[str, Dict], hourdis: Dict[str, Dict]) -> str:
    """Les données de bibliothèque (engins, plateaux, hourdis, profilés de filière) et leurs SOURCES — le tableau de
    l'onglet Méthode de l'écran d'origine, avec la vignette de chaque engin."""
    lignes = []
    for nom, g in gabarits.items():
        app = ", ".join(f"{a['nom']} ({C.nb(a['dx'])} ; {C.nb(a['dy'])}) m {C.fr(a['a'] * 1e3, 0)} × {C.fr(a['b'] * 1e3, 0)}" for a in g["appuis"])
        typ = f"engin · {g.get('type_appuis', '')}" + (" · générique" if g.get("generique") else " · fiche du fichier")
        lignes.append((f'<span class="chp-pic">{pictogramme(g, 60, 38)}</span>', nom, typ, app, g.get("source") or "—"))
    for nom, d in dispo.items():
        if d.get("type") == "plateau":
            f = f"f = {C.nb(d['f'])} MPa" if d.get("f") else "f à renseigner"
            lignes.append(("", nom, f"plateau · {d.get('origine', '')}", f"{C.fr(d.get('A', 0), 0)} × {C.fr(d.get('Bp', 0), 0)} × {C.fr(d.get('t', 0), 0)} mm · {f}",
                           d.get("source") or "—"))
        elif d.get("origine") == "fiche":
            sec = d.get("section", {})
            txt = sec.get("profil") or f"rectangle {C.nb(sec.get('b'))} × {C.nb(sec.get('h'))} mm sur champ"
            lignes.append(("", nom, "filière · fiche du fichier", f"{txt} · {d.get('materiau', '')}", d.get("source") or "—"))
    for nom, d in hourdis.items():
        if d.get("catalogue"):                 # une variante d'un catalogue de fabricant : une ligne par catalogue, plus bas
            continue
        lignes.append(("", nom, "hourdis" + (" · générique" if d.get("generique", True) else " · fiche du fichier"),
                       f"M_Rd {C.nb(d.get('M_Rd') or 0)} · V_Rd {C.nb(d.get('V_Rd') or 0)} ({d.get('nature', 'Rd')})", d.get("source") or "—"))
    for k in BIB.resume_catalogues(hourdis):
        lignes.append(("", k["titre"], "hourdis · catalogue du fabricant",
                       f"{len(k['profils'])} profils ({k['profils'][0]} à {k['profils'][-1]}), {k['variantes']} variantes · M_Rd "
                       f"{C.nb(k['M_min'])} à {C.nb(k['M_max'])} kN·m · V_Rd {C.nb(k['V_min'])} à {C.nb(k['V_max'])} kN par élément ({k['nature']})",
                       f"{k['document']} — transcrit dans {k['fichier']}"))
    lignes.append(("", "Profilés", "filières · catalogue", "IPE, HEA, HEB (I_y, W_el, masse)",
                   "catalogue des profilés de l'application (acier/donnees/profiles.json)"))
    corps = "".join(f'<tr><td class="chp-pc">{pic}</td><td><b>{C.esc(nom)}</b><span>{C.esc(typ)}</span></td><td>{C.symb_math(contenu)}</td>'
                    f'<td class="chp-src">{C.esc(src)}</td></tr>' for pic, nom, typ, contenu, src in lignes)
    return (f'<p class="chp-meth">Les fiches livrées sont GÉNÉRIQUES et nommées comme telles (P-13) : '
            f'chaque valeur porte sa source ; une charge par appui absente vaut 0 et bloque le calcul.</p>'
            f'<table class="vtab chp-tab"><thead><tr><th></th><th>Fiche</th><th>Contenu</th><th>Source</th></tr></thead><tbody>{corps}</tbody></table>')


def vue_balayage(b: Dict, x_actuel: float, L: float, legende: str, plan_svg: str, prefixe: str) -> str:
    """Le balayage : le bandeau du taux selon la position de l'engin, la phrase des positions admissibles, puis le plan
    (comme l'écran d'origine, qui montre le plan sous le bandeau)."""
    return (f'<div class="chp-vue chp-bal"><div class="chp-bal-b">{bandeau(b["points"], b["pire"], x_actuel, L, prefixe)}</div>'
            f'<p class="chp-leg">{C.esc(legende)}</p><div class="chp-bal-p">{plan_svg}</div></div>')


_DECIMALE = re.compile(r"(= ?-?\d+)\.(\d+)")


def _fr_txt(s) -> str:
    """Un texte du moteur à la française : « y = 3.74 m » → « y = 3,74 m » (les références « E1.P2 » restent)."""
    return re.sub(r"(\d)%", "\\1\u00a0%", _DECIMALE.sub(r"\1,\2", str(s)))


FAMILLES_TXT = {"flexion": "Flexion", "tranchant": "Tranchant", "local": "Local", "plateau": "Plateau", "filiere": "Filière",
                "joint": "Joint", "fissuration": "Fissuration", "hypothese": "Hypothèse"}
STATUTS_TXT = {"info": "info", "non_verifie": "non vérifié", "nok": "non conforme"}


def _puce(t: Optional[float], statut: str = "ok") -> str:
    if t is None or statut not in ("ok", "attention", "nok"):
        txt = (f"{C.pc(C.pct(t))}{C.NBSP}%" if t is not None else "—") if statut == "info" else STATUTS_TXT.get(statut, "—")
        return f'<span class="chip {"r" if statut == "nok" else "c"}">{C.esc(txt)}</span>'
    niv = "r" if statut == "nok" else C.niveau(t)
    return C.chip(C.pct(t), niv)


def _val(x) -> str:
    if x is None:
        return "—"
    if isinstance(x, (int, float)):
        return "∞" if x == float("inf") else C.fr(x, 2 if abs(x) < 10 else 1)
    return C.esc(x)


def vue_detail(res: Dict, typ: str) -> str:
    """Le détail des vérifications (l'expander « Détail des vérifications par élément » de l'écran d'origine) : un
    tableau par élément (moment, tranchant, ce qui gouverne), puis chaque autre vérification avec sa méthode."""
    out = '<div class="chp-vue chp-doc chp-large">'
    if res.get("elements"):
        u = ("kN·m", "kN") if typ == "hourdis" else ("kN·m/m", "kN/m")
        lignes = "".join(
            f'<tr><td class="nm">{C.esc(e["nom"])}</td><td class="n">{C.fr(e["M_max"], 1)}</td><td class="n">{C.fr(e["M_adm"], 1)}</td>'
            f'<td class="n">{_puce(e["taux_M"])}</td><td class="n">{C.fr(e["V_max"], 1)}</td><td class="n">{C.fr(e["V_adm"], 1)}</td>'
            f'<td class="n">{_puce(e["taux_V"])}</td><td>{C.esc(e["gouverne"])}</td></tr>' for e in res["elements"])
        out += (f'<h4>Par {"élément" if typ == "hourdis" else "bande de 1 m"}</h4><table class="vtab chp-tab"><thead><tr><th>Élément</th>'
                f'<th class="n">M ({u[0]})</th><th class="n">M<sub>adm</sub></th><th class="n">η<sub>M</sub></th><th class="n">V ({u[1]})</th>'
                f'<th class="n">V<sub>adm</sub></th><th class="n">η<sub>V</sub></th><th>gouverne</th></tr></thead><tbody>{lignes}</tbody></table>')
    autres = [v for v in res.get("verifications", []) if not (v["famille"] in ("flexion", "tranchant") and res.get("elements") and typ == "hourdis")]
    ordre = list(dict.fromkeys(["flexion", "tranchant", "fissuration", "local", "plateau", "filiere", "joint", "hypothese"] + [v["famille"] for v in autres]))
    for fam in ordre:
        vs = [v for v in autres if v["famille"] == fam]
        if not vs:
            continue
        out += f'<h4>{FAMILLES_TXT.get(fam, C.esc(fam))} · {len(vs)}</h4>'
        if fam == "hypothese":
            out += "<ul>" + "".join(f"<li>{C.symb_math(_fr_txt(v['methode']))}</li>" for v in vs) + "</ul>"
            continue
        for m in dict.fromkeys(v["methode"] for v in vs):
            out += f'<p class="chp-meth"><b>Méthode</b> · {C.symb_math(_fr_txt(m))}</p>'
        lignes = ""
        for v in vs:
            u = f' {v["unite"]}' if v.get("unite") and v["E"] is not None else ""
            lignes += (f'<tr><td class="nm">{C.esc(_fr_txt(v["objet"]))}</td><td class="n">{_val(v["E"])}{C.esc(u)}</td><td class="n">{_val(v["R"])}{C.esc(u) if v["R"] is not None else ""}</td>'
                       f'<td class="n">{_puce(v["taux"], v["statut"])}</td></tr>')
        out += (f'<table class="vtab chp-tab"><thead><tr><th>Objet</th><th class="n">E (effet)</th><th class="n">R (résistance)</th>'
                f'<th class="n">η</th></tr></thead><tbody>{lignes}</tbody></table>')
    return out + "</div>"

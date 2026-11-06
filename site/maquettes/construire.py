# -*- coding: utf-8 -*-
"""Maquettes du site « Stab Méthodes » : assemble les pages statiques (accueil, projets, module
garde-corps) à partir des COMPOSANTS de la charte (charte.css) et des DONNÉES RÉELLES du moteur
(donnees.json, extraites par extraire_donnees.py). Les fonctions de ce fichier préfigurent les
composants React du site : barre de navigation, barre du haut, champ, pastille, carte, établi.
Chaque page rend (titre, corps, script) : page() en fait un fichier, apercu.py les réunit en une page.

Lancement : python3 site/maquettes/construire.py (depuis la racine) → site/maquettes/pages/*.html
"""
import html
import json
import os
import re

ICI = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(ICI, "donnees.json"), encoding="utf-8"))
NBSP = " "


# ---------------------------------------------------------------- briques
def icon(nom: str, cls: str = "i") -> str:
    """Une icône Lucide (licence ISC, icones/LICENSE), sans taille fixe : la charte la dimensionne."""
    src = open(os.path.join(ICI, "icones", nom + ".svg"), encoding="utf-8").read()
    src = re.sub(r"<!--.*?-->", "", src, flags=re.S).strip()
    src = re.sub(r'class="[^"]*"', f'class="{cls}"', src, count=1)
    return re.sub(r'\s(width|height)="24"', "", src)


def glyph(trace: str) -> str:
    return (f'<svg class="i" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" '
            f'stroke-linecap="round" stroke-linejoin="round">{trace}</svg>')


# les pictogrammes des modules, dessinés pour la charte (24 × 24, trait 1,6)
MOD = {
    "poutre": glyph('<rect x="3" y="8.5" width="18" height="5.5" rx="1"/><path d="M6 14l-2.2 4h4.4zM18 14l-2.2 4h4.4z"/>'
                    '<circle cx="7.5" cy="12" r=".7" fill="currentColor"/><circle cx="12" cy="12" r=".7" fill="currentColor"/><circle cx="16.5" cy="12" r=".7" fill="currentColor"/>'),
    "dalle": glyph('<path d="M3 12.5 8 8h13l-5 4.5z"/><path d="M3 12.5V15h13v-2.5M16 15l5-4.5V8"/>'),
    "predalle": glyph('<path d="M3 12.5 8 8h13l-5 4.5z"/><path d="M3 12.5V15h13v-2.5M16 15l5-4.5V8M8.3 12.5l4.6-4.5M12.2 12.5l4.6-4.5"/>'),
    "plancher": glyph('<path d="M3 17h18M5 20l2-3M9 20l2-3M13 20l2-3M17 20l2-3"/><path d="M7 4v8M5 10l2 2 2-2M12 4v8M10 10l2 2 2-2M17 4v8M15 10l2 2 2-2"/>'),
    "taux": glyph('<rect x="3" y="5" width="18" height="14" rx="1.5"/><circle cx="7" cy="15.5" r="1.1" fill="currentColor"/><circle cx="10.3" cy="15.5" r="1.1" fill="currentColor"/>'
                  '<circle cx="13.7" cy="15.5" r="1.1" fill="currentColor"/><circle cx="17" cy="15.5" r="1.1" fill="currentColor"/>'),
    "enrobage": glyph('<rect x="4" y="4" width="16" height="16" rx="1.5"/><circle cx="12" cy="10" r="2.6"/><path d="M12 12.6V20M10.5 18.5h3"/>'),
    "age": glyph('<circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3 2"/>'),
    "armatures": glyph('<rect x="3" y="4.5" width="18" height="15" rx="1.5"/><path d="M3 9.5h18M9 9.5v10M15 9.5v10"/>'),
    "profil": glyph('<path d="M5 4h14v3h-5.6v10H19v3H5v-3h5.6V7H5z"/>'),
    "catalogue": glyph('<rect x="3.5" y="4" width="17" height="16" rx="1.5"/><path d="M8 8h3.4M9.7 8v8M8 16h3.4M14 8.5h3M14 12h3M14 15.5h3"/>'),
    "corniere": glyph('<path d="M7 4v16h12v-3H10V4z"/>'),
    "flambement": glyph('<path d="M8 3.5h8M8 20.5h8"/><path d="M12 3.5c4.2 5 4.2 12 0 17"/><path d="M12 3.5v0M12 20.5"/>'),
    "assemblages": glyph('<rect x="4" y="5" width="16" height="14" rx="1.5"/><circle cx="8.3" cy="9.3" r="1.3"/><circle cx="15.7" cy="9.3" r="1.3"/>'
                         '<circle cx="8.3" cy="14.7" r="1.3"/><circle cx="15.7" cy="14.7" r="1.3"/>'),
    "soudures": glyph('<path d="M3 16h11M14 16l6-7.5"/><path d="M6.5 16v-4.5l4.5 4.5" fill="currentColor" fill-opacity=".18"/>'),
    "garde_corps": glyph('<path d="M3 6.5h18M3 16.5h18M3 20h18"/><path d="M5 6.5V20M19 6.5V20M9.5 6.5v10M12 6.5v10M14.5 6.5v10"/>'),
    "bois": glyph('<rect x="3" y="8" width="18" height="8" rx="1"/><path d="M5.5 11c2.5-1.2 5 1.2 7.5 0s4.5-1.2 5.5 0M5.5 13.5c2.5 1.2 5-1.2 7.5 0"/>'),
    "sol": glyph('<rect x="6.5" y="4" width="11" height="4" rx=".8"/><path d="M8.5 8l-1.2 1.4 2.4 1.4-2.4 1.4 2.4 1.4-1.2 1.4M15.5 8l-1.2 1.4 2.4 1.4-2.4 1.4 2.4 1.4-1.2 1.4"/>'
                  '<path d="M3 17h18M5 20l2.5-3M10 20l2.5-3M15 20l2.5-3"/>'),
}
LOGO = ('<svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M5 6h14M5 18h14M12 6v12"/><path d="M8.5 6v0M15.5 18" /></svg>')

FAMILLES = [
    ("beton", "Béton", [("poutre", "Poutre", "Flexion, effort tranchant, flèche et armatures"),
                        ("dalle", "Dalle", "Dalle pleine sur appuis, moments et armatures"),
                        ("predalle", "Prédalle", "Prédalle et dalle de compression"),
                        ("plancher", "Charges sur plancher", "Engins et charges concentrées"),
                        ("taux", "Taux d'armature", "Sections d'acier et pourcentages"),
                        ("enrobage", "Enrobage", "Enrobage nominal, EN 1992-1-1"),
                        ("age", "Âge du béton", "Résistance au jeune âge"),
                        ("armatures", "Tableau armatures", "Diamètres, sections et masses")]),
    ("acier", "Acier", [("profil", "Choix profilé", "Le plus léger qui tient"),
                        ("catalogue", "Catalogue profilés", "IPE, HEA, RHS… filtrés"),
                        ("corniere", "Cornière", "Traction et compression"),
                        ("flambement", "Flambement", "Barres comprimées, EN 1993-1-1"),
                        ("assemblages", "Assemblages", "Platines, cornières, plats d'âme"),
                        ("soudures", "Soudures", "Groupes de cordons, ELS et EC3")]),
    ("autres", "Autres", [("garde_corps", "Garde-corps", "Montants, main courante, ancrage"),
                          ("bois", "Poutre bois", "Flexion et flèche, EN 1995"),
                          ("sol", "Rigidité du sol", "Raideur et coefficients de ressort")]),
]


def chip(taux, niveau, xl=False) -> str:
    if xl:
        return f'<span class="chip xl {niveau}"><small>η</small>{taux}{NBSP}%</span>'
    return f'<span class="chip {niveau}">{taux}{NBSP}%</span>'


def legende(croissant: bool = False) -> str:
    """La légende du code couleur, bâtie sur les seuils lus dans le moteur (donnees.json) : aucune valeur recopiée ici."""
    a, n = D["seuils"]["attention"], D["seuils"]["nok"]
    puces = [f'<span class="chip r">&gt;{NBSP}{n}{NBSP}%</span>', f'<span class="chip f">{a}–{n}{NBSP}%</span>',
             f'<span class="chip p">&lt;{NBSP}{a}{NBSP}%</span>']
    return "".join(reversed(puces) if croissant else puces)


def statut(niveau) -> str:
    lib = {"p": "Conforme", "f": "Conforme, attention", "r": "Non conforme"}[niveau]
    return f'<span class="status {niveau}"><span class="dot"></span>{lib}</span>'


def esc(s) -> str:
    return html.escape(str(s), quote=False)


def symb(s) -> str:
    """Le texte échappé, ses symboles composés en vrai : q_h → q<sub>h</sub> (comme la note PDF)."""
    return re.sub(r"\b([qQστ])_([a-zA-Z]+)\b", r"\1<sub>\2</sub>", esc(s))


def court(s: str) -> str:
    """Les libellés du moteur, raccourcis pour une ligne de résultat."""
    return (s.replace(", sur deux appuis (une travée)", " · 2 appuis").replace("continue sur plusieurs appuis", "continue")
            .replace(" seule", "").replace("σ équivalente", "σ éq."))


def kg(x) -> str:
    return f"{x:.2f}".replace(".", ",")


# ---------------------------------------------------------------- la navigation
FICHE = ' data-fiche="1"'  # la carte qui ouvre la fiche du rapport, au téléphone


def go(cible: str) -> str:
    """Le lien vers une autre page de la maquette (vide : pas de lien)."""
    return f' data-go="{cible}"' if cible else ""


def theme_btn() -> str:
    """Le bouton qui bascule le thème clair / sombre (les maquettes le montrent ; le site le rangera dans les réglages)."""
    return f'<a class="btn icon ghost" data-theme-toggle="1" title="Thème clair ou sombre">{icon("sun-moon")}</a>'


def nav(actif: str) -> str:
    def item(ic, lib, cle, count=""):
        c = f'<span class="count">{count}</span>' if count else ""
        go = f' data-go="{cle}"' if cle in ("accueil", "projets") else ""
        return f'<a class="navitem{" on" if cle == actif else ""}"{go}>{icon(ic)}<span>{lib}</span>{c}</a>'
    fams = ""
    for cle, nom, mods in FAMILLES:
        ouvert = cle == "beton"
        fams += (f'<a class="navitem"><span class="fam modic {cle}" style="width:20px;height:20px;border-radius:6px">{MOD[mods[0][0]]}</span>'
                 f'<span>{nom}</span><span class="count">{len(mods)}</span>{icon("chevron-down" if ouvert else "chevron-right")}</a>')
        if ouvert:
            fams += '<div class="navsub">' + "".join(f'<a class="navitem"><span>{n}</span></a>' for _, n, _ in mods[:5]) + "</div>"
    return (f'<nav class="nav"><div class="brand"><div class="logo">{LOGO}</div><div class="wordmark">Stab Méthodes<span>Bureau d\'études</span></div>{theme_btn()}</div>'
            f'<div class="searchbtn">{icon("search")}<span>Rechercher</span><kbd>Ctrl K</kbd></div>'
            + item("house", "Accueil", "accueil") + item("folder", "Projets", "projets", "4") + item("clock", "Récents", "recents")
            + '<div class="navlabel">Modules</div>' + fams
            + f'<div class="navfoot"><div class="storage">Stockage · 38 Mo sur 1 Go<div class="bar"><b></b></div></div>'
              f'<div class="me"><span class="avatar">LL</span><div class="who"><b>Mon compte</b><span>stab-methodes.be</span></div>'
              f'<span class="spacer"></span>{icon("settings")}</div></div></nav>')


def rail(actif: str) -> str:
    def it(svg, on=False, go=""):
        g = f' data-go="{go}"' if go else ""
        return f'<a class="navitem{" on" if on else ""}"{g}>{svg}</a>'
    return (f'<nav class="nav"><div class="brand"><div class="logo">{LOGO}</div></div>'
            + it(icon("house"), go="accueil") + it(icon("folder"), actif == "projets", "projets") + it(icon("clock")) + '<div class="sep"></div>'
            + it(MOD["poutre"]) + it(MOD["profil"]) + it(MOD["garde_corps"], actif == "garde_corps", "garde-corps") + '<div class="sep"></div>'
            + f'<div style="margin-top:auto;display:flex;flex-direction:column;align-items:center;gap:10px">'
              f'<a class="navitem" data-theme-toggle="1" title="Thème clair ou sombre">{icon("sun-moon")}</a>{it(icon("settings"))}'
              f'<span class="avatar">LL</span></div></nav>')


def mnav(actif: str) -> str:
    """Les onglets du bas du téléphone (accueil et projets) : les mêmes entrées que la navigation du PC."""
    def it(ic, lib, cle, go=""):
        g = f' data-go="{go}"' if go else ""
        return f'<a class="{"on" if cle == actif else ""}"{g}>{icon(ic)}<span>{lib}</span></a>'
    return ('<nav class="mnav">' + it("house", "Accueil", "accueil", "accueil") + it("folder", "Projets", "projets", "projets")
            + it("clock", "Récents", "recents") + it("layout-grid", "Modules", "modules", "accueil") + "</nav>")


# le script commun : thème demandé par l'adresse, bouton de thème, liens entre les pages ; ALLER est remplacé par l'aperçu
# en une seule page (les vues y sont des ancres)
SCRIPT_COMMUN = """
(function(){
  var q=new URLSearchParams(location.search), root=document.documentElement;
  if(q.get('theme')==='dark') root.dataset.theme='dark';
  function sombre(){return root.dataset.theme==='dark'||(!root.dataset.theme&&matchMedia('(prefers-color-scheme: dark)').matches);}
  window.ALLER=window.ALLER||function(p){location.href=p+'.html'+(root.dataset.theme==='dark'?'?theme=dark':'');};
  document.addEventListener('click',function(ev){
    var g=ev.target.closest('[data-go]'); if(g){ev.preventDefault();window.ALLER(g.dataset.go);return;}
    var t=ev.target.closest('[data-theme-toggle]'); if(t){root.dataset.theme=sombre()?'light':'dark';}
  });
})();"""


def page(titre: str, corps: str, script: str = "") -> str:
    return (f'<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{esc(titre)} · Stab Méthodes</title><link rel="stylesheet" href="../charte.css"></head><body>{corps}'
            f'<script>{SCRIPT_COMMUN}{script}</script></body></html>')


# ---------------------------------------------------------------- l'accueil
def accueil() -> tuple:
    gc = D["gc"]
    vue = {v["cle"]: v for v in gc["vues"]}
    cartes = [
        (vue["elevation"]["mini"], "Garde-corps terrasse R+1", "Réalex", "C", "il y a 12 min", gc["eta"], gc["niveau"]),
        (D["soudure"]["svg"], "Gousset de contreventement", "Hall logistique", "A", "hier", D["soudure"]["eta"], D["soudure"]["niveau"]),
        (D["gc_lat"]["svg"], "Platine latérale nez de dalle", "Résidence Les Tilleuls", "B", "3 nov.", D["gc_lat"]["eta"], D["gc_lat"]["niveau"]),
    ]
    reprendre = "".join(
        f'<a class="card rcard"{go("garde-corps" if i == 0 else "")}><div class="thumb">{svg}{chip(e, n)}</div><div class="body"><div class="t">{esc(t)}</div>'
        f'<div class="m">{icon("folder")}<span>{esc(ch)}</span><span>·</span><span class="indice" style="height:18px;min-width:18px;font-size:10.5px">{ind}</span>'
        f'<span>·</span><span>{quand}</span></div></div></a>' for i, (svg, t, ch, ind, quand, e, n) in enumerate(cartes))
    familles = ""
    for cle, nom, mods in FAMILLES:
        tuiles = "".join(f'<a class="tile"{go("garde-corps" if k == "garde_corps" else "")}><span class="modic {cle}">{MOD[k]}</span>'
                         f'<div><b>{esc(n)}</b><span>{esc(s)}</span></div><span class="chev">{icon("chevron-right")}</span></a>'
                         for k, n, s in mods)
        familles += (f'<div class="fam {cle}"><div class="famhead"><span class="pill {cle}">{nom}</span><span class="muted">{len(mods)} modules</span></div>'
                     f'<div class="tiles">{tuiles}</div></div>')
    # au téléphone, une famille à la fois : le même choix que les trois blocs du PC
    choix = '<div class="seg full mfam">' + "".join(
        f'<button data-fam="{cle}" class="{"on" if cle == "beton" else ""}">{nom} · {len(mods)}</button>' for cle, nom, mods in FAMILLES) + "</div>"
    familles = f'{choix}<div class="fams" data-fam="beton">{familles}</div>'
    sites = [("Réalex", "15 rapports · 4 dossiers", (12, 2, 1)), ("Résidence Les Tilleuls", "8 rapports · Namur", (8, 0, 0)),
             ("École Saint-Joseph", "11 rapports · Liège", (9, 1, 1)), ("Hall logistique", "6 rapports · Charleroi", (5, 1, 0))]
    lst = "".join(f'<div class="site"{go("projets" if n == "Réalex" else "")}><span class="ic">{icon("building-2")}</span><div><b>{n}</b><span>{s}</span></div><div class="chips">'
                  + (f'<span class="chip p">{a}</span>' if a else "") + (f'<span class="chip f">{b}</span>' if b else "") + (f'<span class="chip r">{c}</span>' if c else "")
                  + "</div></div>" for n, s, (a, b, c) in sites)
    feed = [("SM", "a2", "Sophie a diffusé l'<b>indice B</b> de <b>Garde-corps mezzanine</b>", "il y a 2 h"),
            ("TD", "a3", "Thomas a créé <b>Platine d'about P3</b> dans Hall logistique", "hier, 16:40"),
            ("LL", "", "Vous avez exporté <b>Dalle R+1 · zone B</b> en .stab", "hier, 11:05")]
    act = "".join(f'<div class="feed"><span class="avatar sm {c}">{i}</span><div>{t}<span class="when">{w}</span></div></div>' for i, c, t, w in feed)
    mhead = (f'<header class="mhead"><div class="logo">{LOGO}</div><span class="ttl">Stab Méthodes</span><span class="spacer"></span>'
             f'{theme_btn()}<a class="btn icon ghost" title="Notifications">{icon("bell")}</a><span class="avatar">LL</span></header>')
    corps = (f'<div class="app">{nav("accueil")}<main class="main">{mhead}<div class="page">'
             f'<div class="hello"><div><div class="h1">Bonjour</div><div class="sub">Vendredi 6 novembre · 3 rapports en cours, 1 à diffuser</div></div>'
             f'<span class="spacer"></span><div class="bigsearch">{icon("search")}<span>Rechercher un module, un chantier, une note…</span><span class="kbd">Ctrl K</span></div>'
             f'<a class="btn primary big" title="Nouveau calcul">{icon("plus")}<span class="lbl">Nouveau calcul</span></a></div>'
             f'<div><div class="sechead"><div class="h2">Reprendre</div><a class="btn ghost sm">Tout voir {icon("arrow-up-right")}</a></div>'
             f'<div class="resume">{reprendre}</div><div class="sechead"><div class="h2">Modules</div></div>{familles}</div>'
             f'<aside class="side"><div class="card"><div class="sechead"><div class="h3">Chantiers actifs</div><a class="btn ghost sm">{icon("plus")}</a></div>{lst}'
             f'<div class="legend" style="margin-top:12px">{legende(True)}<span>nombre de rapports</span></div></div>'
             f'<div class="card"><div class="sechead"><div class="h3">Activité du bureau</div></div>{act}</div></aside>'
             f'</div>{mnav("accueil")}</main></div>')
    script = """
(function(){
  var fams=document.querySelector('.fams');
  document.querySelectorAll('.mfam button').forEach(function(b){b.addEventListener('click',function(){
    fams.dataset.fam=b.dataset.fam;
    document.querySelectorAll('.mfam button').forEach(function(x){x.classList.toggle('on',x===b);});
  });});
})();"""
    return "Accueil", corps, script


# ---------------------------------------------------------------- les projets
def projets() -> tuple:
    gc = D["gc"]
    elev = {v["cle"]: v for v in gc["vues"]}["elevation"]["svg"]
    noeuds = (f'<div class="field text">{icon("search")}<input value="" placeholder="Filtrer les chantiers"></div>'
              f'<a class="node">{icon("layout-grid")}Tous les chantiers<span class="count">4</span></a>'
              f'<a class="node chantier">{icon("folder-open")}Réalex<span class="count">15</span></a>'
              f'<a class="node l1">{icon("folder")}Étude de la dalle<span class="count">4</span></a>'
              f'<a class="node l1 on">{icon("folder-open")}Garde-corps<span class="count">5</span></a>'
              f'<a class="node l1">{icon("folder")}Assemblages métalliques<span class="count">4</span></a>'
              f'<a class="node l1">{icon("folder")}Fondations<span class="count">2</span></a>'
              f'<a class="node chantier">{icon("folder")}Résidence Les Tilleuls<span class="count">8</span></a>'
              f'<a class="node chantier">{icon("folder")}École Saint-Joseph<span class="count">11</span></a>'
              f'<a class="node chantier">{icon("folder")}Hall logistique<span class="count">6</span></a>'
              f'<div class="navlabel">Archives</div><a class="node">{icon("folder")}2025<span class="count">37</span></a>'
              f'<a class="btn ghost sm" style="margin-top:auto;justify-content:flex-start">{icon("plus")}Nouveau chantier</a>')
    lignes = [("Garde-corps terrasse R+1", "barreaudage · droit 3,60 m", gc["eta"], gc["niveau"], "C", "draft", "Brouillon", "il y a 12 min", "LL", "", True),
              ("Garde-corps mezzanine", f"barreaudage · H 1{NBSP}345 mm", D["gc_haut"]["eta"], D["gc_haut"]["niveau"], "B", "pub", "Diffusé", "hier", "SM", "a2", False),
              ("Garde-corps coursive en L", "lisses horizontales · en L", D["gc_L"]["eta"], D["gc_L"]["niveau"], "B", "pub", "Diffusé", "3 nov.", "LL", "", False),
              ("Garde-corps tribune", "flèche admissible L/200", D["gc_l200"]["eta"], D["gc_l200"]["niveau"], "A", "draft", "Brouillon", "3 nov.", "TD", "a3", False),
              ("Garde-corps escalier principal", "rampant · pente 32°", D["gc_rampant"]["eta"], D["gc_rampant"]["niveau"], "B", "pub", "Diffusé", "28 oct.", "LL", "", False)]
    rows = "".join(
        f'<tr class="{"sel" if sel else ""}"{go("garde-corps" if sel else "")}><td><span class="mod"><span class="modic autres">{MOD["garde_corps"]}</span><span><b style="font-weight:650">{esc(n)}</b>'
        f'<span class="muted" style="display:block;font-size:11.5px">{esc(s)}</span></span></span></td><td>{chip(e, nv)}</td>'
        f'<td><span class="indice">{ind}</span></td><td><span class="tag {tc}"><span class="dot"></span>{tl}</span></td>'
        f'<td class="muted">{q}</td><td><span class="avatar sm {ac}">{av}</span></td><td class="n">{icon("ellipsis")}</td></tr>'
        for n, s, e, nv, ind, tc, tl, q, av, ac, sel in lignes)
    table = (f'<div class="card tablecard"><table class="table"><thead><tr><th>Rapport</th><th>Résultat</th><th>Indice</th><th>État</th><th>Modifié</th><th>Par</th><th></th></tr></thead>'
             f'<tbody>{rows}</tbody></table></div>')
    # au téléphone : les mêmes rapports en cartes ; la carte du rapport retenu ouvre sa fiche (indices compris)
    cartes = "".join(
        f'<a class="mrow"{FICHE if sel else ""}><span class="modic autres">{MOD["garde_corps"]}</span>'
        f'<div class="mt"><b>{esc(n)}</b><span>{esc(s)}</span><div class="mm"><span class="indice">{ind}</span>'
        f'<span class="tag {tc}"><span class="dot"></span>{tl}</span><span>{q}</span><span class="avatar sm {ac}">{av}</span></div></div>{chip(e, nv)}</a>'
        for n, s, e, nv, ind, tc, tl, q, av, ac, sel in lignes)
    cartes = f'<div class="mlist">{cartes}</div>'
    versions = [("C", True, "Brouillon", "draft", "Montant RHS 50x30x3 au lieu de RHS 50x30x4", "aujourd'hui, 14:32", "LL", "", False),
                ("B", False, "Diffusé", "pub", "Catégorie A, entraxe maximal 1,00 m", "28 oct. 2026", "SM", "a2", True),
                ("A", False, "Diffusé", "pub", "Première émission", "14 oct. 2026", "LL", "", True)]
    tl = "".join(
        f'<div class="ver{" cur" if cur else ""}"><span class="indice">{i}</span><div><b>Indice {i}<span class="tag {tc}"><span class="dot"></span>{t}</span></b>'
        f'<p>{esc(p)}</p><div class="when"><span class="avatar sm {ac}">{av}</span>{w}</div></div>'
        + (f'<a class="btn sm">{icon("file-text")}PDF</a>' if pdf else f'<a class="btn ghost sm">{icon("eye")}</a>') + "</div>"
        for i, cur, t, tc, p, w, av, ac, pdf in versions)
    tiroir = (f'<aside class="drawer"><span class="grab"></span><a class="btn icon ghost dclose" data-fermer-fiche="1" title="Fermer">{icon("x")}</a>'
              f'<div class="thumb">{elev}</div>'
              f'<div><div class="h2">Garde-corps terrasse R+1</div><div class="muted" style="margin-top:3px">Réalex / Garde-corps</div></div>'
              f'<div style="display:flex;align-items:center;gap:8px">{statut(gc["niveau"])}{chip(gc["eta"], gc["niveau"])}<span class="muted" style="font-size:12px">{esc(gc["gouvernant"].split(" · ")[0])} · {esc(gc["critere"])}</span></div>'
              f'<dl class="facts"><dt>Type</dt><dd>{esc(gc["type"])}</dd><dt>Géométrie</dt><dd>droit 3,60 m · 5 montants</dd><dt>Montants</dt><dd>RHS 50x30x3</dd>'
              f'<dt>Fixation</dt><dd>platine 150 × 150 × 10</dd><dt>Méthode</dt><dd>vérification ELS · S235</dd><dt>Masse</dt><dd>{gc["masse"]}{NBSP}kg</dd></dl>'
              f'<div style="display:flex;gap:8px"><a class="btn primary" style="flex:1;justify-content:center"{go("garde-corps")}>{icon("arrow-up-right")}Ouvrir</a>'
              f'<a class="btn">{icon("file-down")}Note PDF</a><a class="btn icon">{icon("download")}</a></div>'
              f'<div><div class="kicker" style="margin-bottom:4px">Indices</div><div class="timeline">{tl}</div></div></aside>')
    top = (f'<header class="top"><div class="crumbs">{icon("folder")}<span>Projets</span>{icon("chevron-right")}<span>Réalex</span>{icon("chevron-right")}<b>Garde-corps</b></div>'
           f'<span class="spacer"></span><a class="btn">{icon("upload")}Importer .stab</a><a class="btn primary">{icon("plus")}Nouveau rapport</a></header>')
    liste = (f'<section class="list"><div class="head"><div><div class="kicker">Réalex · chantier</div><div class="h1" style="margin-top:4px">Garde-corps</div>'
             f'<div class="muted" style="margin-top:4px">5 rapports · 3 diffusés · mis à jour il y a 12 min</div></div><span class="spacer"></span>'
             f'<div class="seg"><button class="on">Liste</button><button>Vignettes</button></div></div>{table}{cartes}'
             f'<div class="legend" style="margin-top:12px"><span>Résultat :</span>{legende(True)}</div></section>')
    mhead = (f'<header class="mhead"><div class="logo">{LOGO}</div><span class="ttl">Projets</span><span class="spacer"></span>'
             f'<a class="btn icon" title="Importer .stab">{icon("upload")}</a><a class="btn icon primary" title="Nouveau rapport">{icon("plus")}</a></header>')
    corps = (f'<div class="app rail">{rail("projets")}<main class="main">{top}{mhead}<div class="proj"><aside class="tree">{noeuds}</aside>{liste}'
             f'<div class="fscrim" data-fermer-fiche="1"></div>{tiroir}</div>{mnav("projets")}</main></div>')
    script = """
(function(){
  var proj=document.querySelector('.proj'), q=new URLSearchParams(location.search);
  document.querySelectorAll('[data-fiche]').forEach(function(e){e.addEventListener('click',function(){proj.classList.add('show-fiche');});});
  document.querySelectorAll('[data-fermer-fiche]').forEach(function(e){e.addEventListener('click',function(){proj.classList.remove('show-fiche');});});
  if(q.get('fiche')) proj.classList.add('show-fiche');
  // l'aperçu en une page rappelle cette fonction quand la vue Projets s'affiche (cachée, elle n'a pas de mesures)
  window.RECENTRER=function(){
    var on=document.querySelector('.tree .node.on'), t=document.querySelector('.tree');
    if(on&&matchMedia('(max-width: 700px)').matches) t.scrollLeft+=on.getBoundingClientRect().left-t.getBoundingClientRect().left-16;
  };
  window.RECENTRER();
})();"""
    return "Projets", corps, script


# ---------------------------------------------------------------- le module garde-corps
def ligne(label, champ, sym="", aide=False, full=False) -> str:
    s = f'<span class="sym">{sym}</span>' if sym else ""
    a = icon("info") if aide else ""
    return f'<div class="row{" full" if full else ""}"><label>{s}<span>{label}</span>{a}</label>{champ}</div>'


def num(v, u, locked=False, focus=False) -> str:
    cls = " locked" if locked else (" focus" if focus else "")
    lk = icon("lock") if locked else ""
    return f'<div class="field{cls}">{lk}<input value="{v}"><span class="u">{u}</span></div>'


def sel(v) -> str:
    return f'<div class="field select"><span class="v">{esc(v)}</span>{icon("chevron-down")}</div>'


def seg(opts, on, full=True) -> str:
    return f'<div class="seg{" full" if full else ""}">' + "".join(f'<button class="{"on" if o == on else ""}">{o}</button>' for o in opts) + "</div>"


def section(n, titre, resume, contenu="", ouvert=False) -> str:
    ch = icon("chevron-down" if ouvert else "chevron-right")
    corps = f'<div class="rows">{contenu}</div>' if ouvert else ""
    return (f'<div class="sec{" open" if ouvert else ""}"><div class="sechd"><span class="n">{n}</span><span class="tt">{titre}</span>'
            f'<span class="sum">{"" if ouvert else esc(resume)}</span>{ch}</div>{corps}</div>')


def module() -> tuple:
    gc = D["gc"]
    vues = gc["vues"]
    courts = {"elevation": "Élévation", "coupe": "Coupe A-A", "plan": "Plan", "platine": "Platine", "pied": "Fixation", "section": "Section"}
    legendes = {"elevation": f'{esc(gc["type"])} · {esc(gc["phrase"])}', "coupe": "dans l'axe d'un montant", "plan": "vue de dessus",
                "platine": "platine sur la dalle, vue de dessus", "pied": "coupe à grande échelle, efforts du cas gouvernant", "section": "élément principal"}
    # le panneau des données
    s1 = (ligne("Type", sel("Barreaudage vertical"), full=True) + ligne("Forme", seg(["Droit", "En L", "En U"], "Droit"), full=True)
          + ligne("Hauteur", num(f"1{NBSP}100", "mm", focus=True), "H", True) + ligne("Pente", num("0", "°"), "α"))
    s4 = (ligne("Catégorie", sel("A · habitation — 0,5 kN/m"), full=True) + '<div class="grp">Sur la main courante</div>'
          + ligne("horizontale", num("0,50", "kN/m", locked=True), "q<sub>h</sub>", True)
          + ligne("verticale", num("0,50", "kN/m"), "q<sub>v</sub>") + '<div class="grp">Sur le remplissage</div>'
          + ligne("ponctuelle", num("0,50", "kN"), "Q<sub>r</sub>")
          + ligne("pression", num("0,50", "kN/m²"), "q<sub>r</sub>")
          + f'<div class="row"><label><span>Cumuler q<sub>h</sub> et q<sub>v</sub></span></label><div style="display:flex;justify-content:flex-end"><span class="switch"></span></div></div>'
          + '<div class="hint">Quatre cas distincts : <b>chaque charge agit seule</b>. q<sub>h</sub> suit la catégorie, EN 1991-1-1 Tab. 6.12.</div>')
    donnees = (f'<div class="phead"><span class="h3">Données</span>{seg(["ELS", "Eurocode"], "ELS", full=False)}</div>'
               + section(1, "Garde-corps", "", s1, True)
               + section(2, "Tronçons", "3,60 m · 4 × 0,90 m")
               + section(3, "Remplissage", "barreaux · e 110 mm")
               + section(4, "Charges", "", s4, True)
               + section(5, "Fixation", "platine · 4 chevilles M12")
               + section(6, "Méthode", "S235 · flèche L/100"))
    # le dessin
    tabs = "".join(f'<button data-vue="{v["cle"]}" class="{"on" if v["cle"] == "elevation" else ""}">{courts.get(v["cle"], v["titre"])}</button>' for v in vues)
    feuilles = "".join(
        f'<div class="view{" on" if v["cle"] == "elevation" else ""}" data-vue="{v["cle"]}"><div class="sheet"><div class="stitle"><b>{esc(v["titre"].upper())}</b>'
        f'<span>{legendes.get(v["cle"], "")}</span><span class="scale">mm · à l\'échelle</span></div><div class="draw">{v["svg"]}</div>'
        f'<div class="cart"><div><span>Rapport</span><b>Garde-corps terrasse R+1</b></div><div><span>Chantier</span><b>Réalex</b></div>'
        f'<div><span>Indice</span><b>C</b></div><div><span>Date</span><b>06/11/2026</b></div><div class="mark"><i></i>Stab Méthodes</div></div></div></div>'
        for v in vues)
    minis = "".join(f'<div class="mini{" on" if v["cle"] == "elevation" else ""}" data-vue="{v["cle"]}"><div class="pic">{v["mini"]}</div>'
                    f'<span>{courts.get(v["cle"], v["titre"])}</span></div>' for v in vues)
    dessin = (f'<section class="canvas"><div class="ctool"><div class="seg">{tabs}</div><div class="zoom"><button>{icon("minus")}</button><span>100 %</span>'
              f'<button>{icon("plus")}</button><button>{icon("maximize-2")}</button></div></div>'
              f'<div class="stage">{feuilles}</div><div class="strip">{minis}</div>{profils(gc)}</section>')
    # le résultat
    opt = gc["optimaux"]
    els = ""
    for e in gc["elements"]:
        parts = [court(x) for x in (e["desc"], e["critere"], e["cas"]) if x]
        desc = " · ".join(p for i, p in enumerate(parts) if p not in parts[:i])
        l3 = ""
        if e["cle"] in opt:
            o = opt[e["cle"]]
            l3 = (f'<div class="l3">{sel(e["section"])}</div><a class="optline" data-profils="1" title="Le plus léger qui tient">{icon("star")}'
                  f'<span class="lab">Optimal</span><b>{esc(o["nom"])}</b>{chip(o["taux"], o["niveau"])}<span class="go">Profilés{icon("chevron-right")}</span></a>')
        gouv = e["nom"] == gc["gouvernant"].split(" · ")[0]
        els += (f'<div class="el{" gov" if gouv else ""}"><div class="l1"><b>{esc(e["nom"])}</b>{chip(e["taux"], e["niveau"])}</div>'
                f'<div class="l2">{symb(desc)}</div><div class="bar"><b class="{e["niveau"]}" style="width:{min(e["taux"], 100)}%"></b></div>{l3}</div>')
    resultat = (f'<div class="phead"><span class="h3">Résultat</span>{statut(gc["niveau"])}</div>'
                f'<div class="verdict"><div class="big">{chip(gc["eta"], gc["niveau"], xl=True)}<div class="gov"><b>{esc(gc["gouvernant"].split(" · ")[0])}</b>'
                f'{esc(gc["critere"])} · {symb(gc["elements"][0]["cas"])}</div></div>'
                f'<div class="meta">{icon("circle-check")}Vérification ELS · quatre cas distincts · S235</div></div>'
                f'<div class="els">{els}</div><div class="foot"><div class="mass"><span>Masse indicative de l\'acier</span><b>{gc["masse"]}{NBSP}kg</b></div>'
                f'<div class="legend">{legende()}</div></div>')
    top = (f'<header class="top"><div class="crumbs">{icon("folder")}<span{go("projets")}>Réalex</span>{icon("chevron-right")}<span{go("projets")}>Garde-corps</span>{icon("chevron-right")}'
           f'<b>Garde-corps terrasse R+1</b></div><span class="indice">C</span><span class="tag draft"><span class="dot"></span>Brouillon</span>'
           f'<span class="spacer"></span><span class="saved">{icon("circle-check")}Enregistré il y a 2 min</span>'
           f'<span class="avstack"><span class="avatar sm">LL</span><span class="avatar sm a2">SM</span></span>'
           f'<a class="btn ghost">{icon("history")}Indices</a><a class="btn">{icon("download")}.stab</a><a class="btn primary">{icon("file-down")}Note PDF</a></header>')
    mtop = (f'<header class="mtop"><a class="btn icon ghost"{go("projets")} title="Retour aux projets">{icon("arrow-left")}</a><div class="t"><b>Garde-corps terrasse R+1</b><span>Réalex · indice C · brouillon</span></div>'
            f'{chip(gc["eta"], gc["niveau"])}</header>')
    mtabs = f'<div class="mtabs"><div class="seg full"><button data-tab="donnees">Données</button><button data-tab="dessin">Dessin</button><button data-tab="resultat">Résultat</button></div></div>'
    mbar = f'<div class="mbar"><a class="btn">{icon("history")}Indices</a><a class="btn primary">{icon("file-down")}Note PDF</a></div>'
    corps = (f'<div class="app rail">{rail("garde_corps")}<main class="main">{top}{mtop}{mtabs}'
             f'<div class="bench" data-tab="donnees"><section class="panel left">{donnees}</section>{dessin}<section class="panel right">{resultat}</section></div>{mbar}</main></div>')
    script = """
(function(){
  var q=new URLSearchParams(location.search), bench=document.querySelector('.bench');
  function vue(k){document.querySelectorAll('[data-vue]').forEach(function(e){e.classList.toggle('on',e.dataset.vue===k);});}
  document.querySelectorAll('[data-vue]').forEach(function(e){e.addEventListener('click',function(){vue(e.dataset.vue);});});
  function tab(t){bench.dataset.tab=t;document.querySelectorAll('.mtabs button').forEach(function(b){b.classList.toggle('on',b.dataset.tab===t);});}
  document.querySelectorAll('.mtabs button').forEach(function(b){b.addEventListener('click',function(){tab(b.dataset.tab);});});
  document.querySelectorAll('[data-profils]').forEach(function(b){b.addEventListener('click',function(){document.querySelector('.canvas').classList.add('show-profils');});});
  document.querySelectorAll('[data-fermer]').forEach(function(b){b.addEventListener('click',function(){document.querySelector('.canvas').classList.remove('show-profils');});});
  tab(q.get('tab')||'donnees'); if(q.get('vue')) vue(q.get('vue'));
  if(q.get('profils')) document.querySelector('.canvas').classList.add('show-profils');
})();"""
    return "Garde-corps terrasse R+1", corps, script


def profils(gc) -> str:
    """La feuille des profilés du montant : familles, réglages, l'optimal, le tableau aux cellules colorées."""
    lignes = ""
    for r in gc["table_montant"]:
        cls = "ret" if r["retenue"] else ""
        etoile = f'<span class="star">{icon("star")}</span>' if r["optimal"] else ""
        marque = '<span class="tag draft" style="height:20px">retenue</span>' if r["retenue"] else ""
        lignes += (f'<tr class="{cls}"><td><b style="font-weight:600">{esc(r["nom"])}</b>{etoile}{marque}</td>'
                   f'<td class="n">{kg(r["kgm"])}</td><td class="eta">{chip(r["eq"], r["n_eq"])}</td><td class="eta">{chip(r["fl"], r["n_fl"])}</td>'
                   f'<td class="eta">{chip(r["eta"], r["n_eta"])}</td></tr>')
    best = next((r for r in gc["table_montant"] if r["optimal"]), None)
    bandeau = (f'<div class="optimal">{icon("star")}<span><b>Optimal : {esc(best["nom"])}</b> · {kg(best["kgm"])} kg/m · η {best["eta"]}{NBSP}% · le plus léger qui tient</span>'
               f'<a class="btn primary sm">Prendre l\'optimal</a></div>') if best else ""
    return (f'<div class="scrim" data-fermer="1"></div><div class="sheetR"><div class="sh"><span class="modic autres">{MOD["garde_corps"]}</span>'
            f'<div><div class="h2">Profilés pour le montant principal</div><div class="muted" style="font-size:12px">chaque section vérifiée dans ce rôle · un clic la retient</div></div>'
            f'<a class="btn icon ghost" data-fermer="1">{icon("x")}</a></div>'
            f'<div class="filters"><span class="fchip on">RHS</span><span class="fchip">SHS</span><span class="fchip">CHS</span><span class="fchip">Plats</span><span class="fchip">Ronds</span>'
            f'<span class="spacer"></span><span class="fchip on">{icon("check")}Formés à froid</span><span class="fchip">Seulement ceux qui tiennent</span>'
            f'<span class="fchip">≤ 15 kg/m</span></div>{bandeau}'
            f'<div class="ptable"><table class="table"><thead><tr><th>Section</th><th class="n">kg/m</th><th>η σ<sub>eq</sub></th><th>η flèche</th><th>η</th></tr></thead>'
            f'<tbody>{lignes}</tbody></table></div></div>')


def main():
    sortie = os.path.join(ICI, "pages")
    os.makedirs(sortie, exist_ok=True)
    for nom, fn in (("accueil", accueil), ("projets", projets), ("garde-corps", module)):
        with open(os.path.join(sortie, nom + ".html"), "w", encoding="utf-8") as f:
            f.write(page(*fn()))
        print(nom, os.path.getsize(os.path.join(sortie, nom + ".html")) // 1024, "ko")


if __name__ == "__main__":
    main()

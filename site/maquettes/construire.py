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
    # l'aide-mémoire : une fiche de données (lignes de texte) et la section d'une barre
    "donnees_beton": glyph('<path d="M6 3.5h9l3.5 3.5v13.5H6z"/><path d="M15 3.5V7h3.5M8.8 10.5h6.4M8.8 13.5h3.6"/>'
                           '<circle cx="14.6" cy="16.6" r="1.9" fill="currentColor" fill-opacity=".18"/>'),
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

# les modules du bureau : le titre seul dit ce que fait le module (retour du bureau, 06/11/2026)
FAMILLES = [
    ("beton", "Béton", [("poutre", "Poutres béton armé"), ("dalle", "Dalles béton armé"), ("predalle", "Prédalles"),
                        ("plancher", "Charges sur plancher"), ("taux", "Taux d'armature"), ("enrobage", "Enrobage"),
                        ("age", "Maturité du béton"), ("armatures", "Tableau des armatures"),
                        ("donnees_beton", "Données béton")]),
    ("acier", "Acier", [("profil", "Choix de profilé"), ("catalogue", "Catalogue des profilés"), ("corniere", "Cornières"),
                        ("flambement", "Flambement"), ("assemblages", "Assemblages métalliques"), ("soudures", "Soudures")]),
    ("autres", "Autres", [("garde_corps", "Garde-corps"), ("bois", "Poutres bois"), ("sol", "Rigidité du sol")]),
]
# les modules MASQUÉS (docs/site/BIBLIOTHEQUES.md § 2.1) : des consultations que la Bibliothèque Béton armé remplace.
# Leurs pages restent servies (une note existante s'ouvre encore, js/modules.js les connaît), mais l'accueil, le volet
# « Modules » et la fenêtre « Nouvelle note » ne les listent plus ; leur retrait définitif se décidera avec le bureau.
MASQUES = ("donnees_beton", "armatures", "enrobage")
# les modules validés par un administrateur — POUR L'EXEMPLE : la maquette montre la pastille, le bureau décidera
VALIDES = {"poutre": ("Chef de service", "12/10/2026"), "dalle": ("Chef de service", "12/10/2026"),
           "enrobage": ("Vous", "20/10/2026"), "armatures": ("Vous", "20/10/2026"), "catalogue": ("Vous", "03/11/2026")}


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


def pastille_validee(cle: str) -> str:
    """La petite pastille d'un module validé par un administrateur (rien si le module n'est pas validé)."""
    if cle not in VALIDES:
        return ""
    qui, quand = VALIDES[cle]
    return f'<span class="ok" title="Module validé · {qui} · {quand}">{icon("badge-check")}</span>'


def statut(niveau) -> str:
    lib = {"p": "Conforme", "f": "Conforme, attention", "r": "Non conforme", "c": "À compléter"}[niveau]
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


# les pages des modules que la maquette montre ; les autres tuiles ne mènent nulle part
PAGES_MODULES = {"garde_corps": "garde-corps", "poutre": "poutre"}


def rail(actif: str) -> str:
    """La barre d'icônes : cinq fonctions seulement, chacune nommée par une bulle au survol. « Modules » ouvre le volet des
    modules (la seconde navigation) ; il est allumé quand on travaille dans un module."""
    def it(ic, nom, cle, attr=""):
        return f'<a class="navitem{" on" if cle == actif else ""}" data-tip="{nom}"{attr}>{icon(ic)}</a>'
    return (f'<nav class="nav"><div class="brand"><div class="logo">{LOGO}</div></div>'
            + it("house", "Accueil", "accueil", go("accueil")) + it("folder", "Projets", "projets", go("projets"))
            + it("layout-grid", "Modules", "modules", ' data-modules="1"') + it("users", "Équipe", "equipe", go("equipe"))
            + it("settings", "Paramètres", "parametres", go("parametres"))
            + f'<div style="margin-top:auto;display:flex;flex-direction:column;align-items:center;gap:10px">'
              f'<a class="navitem" data-theme-toggle="1" data-tip="Thème clair ou sombre">{icon("sun-moon")}</a>'
              f'<span class="avatar" title="Mon compte">LL</span></div></nav>' + modpanel())


def modpanel() -> str:
    """Le volet des modules, classés par famille ; la pastille dit qu'un administrateur a validé le module."""
    corps = ""
    for cle, nom, mods in FAMILLES:
        corps += f'<div class="navlabel">{nom}</div>' + "".join(
            f'<a class="navitem"{go(PAGES_MODULES.get(k, ""))}><span class="modic {cle}">{MOD[k]}</span><span>{esc(n)}</span>{pastille_validee(k)}</a>'
            for k, n in mods)
    return (f'<aside class="modpanel"><div class="mp-h"><div class="h2">Modules</div><span class="okleg">{icon("badge-check")}validé</span>'
            f'<a class="btn icon ghost" data-modules="1" title="Fermer">{icon("x")}</a></div>{corps}</aside>')


def mnav(actif: str) -> str:
    """Les onglets du bas du téléphone : les cinq entrées de la barre du PC, dans le même ordre."""
    def it(ic, lib, cle, go=""):
        g = f' data-go="{go}"' if go else ""
        return f'<a class="{"on" if cle == actif else ""}"{g}>{icon(ic)}<span>{lib}</span></a>'
    return ('<nav class="mnav">' + it("house", "Accueil", "accueil", "accueil") + it("folder", "Projets", "projets", "projets")
            + it("layout-grid", "Modules", "modules", "accueil") + it("users", "Équipe", "equipe", "equipe")
            + it("settings", "Paramètres", "parametres", "parametres") + "</nav>")


# le script commun : thème demandé par l'adresse, bouton de thème, liens entre les pages ; ALLER est remplacé par l'aperçu
# en une seule page (les vues y sont des ancres)
SCRIPT_COMMUN = """
(function(){
  var q=new URLSearchParams(location.search), root=document.documentElement;
  if(q.get('theme')==='dark') root.dataset.theme='dark';
  function sombre(){return root.dataset.theme==='dark'||(!root.dataset.theme&&matchMedia('(prefers-color-scheme: dark)').matches);}
  window.ALLER=window.ALLER||function(p){location.href=p+'.html'+(root.dataset.theme==='dark'?'?theme=dark':'');};
  var corps=document.body;
  document.addEventListener('click',function(ev){
    var g=ev.target.closest('[data-go]'); if(g){ev.preventDefault();corps.classList.remove('show-modules','show-nouvelle');window.ALLER(g.dataset.go);return;}
    var t=ev.target.closest('[data-theme-toggle]'); if(t){root.dataset.theme=sombre()?'light':'dark';}
    var m=ev.target.closest('[data-modules]'); if(m){corps.classList.toggle('show-modules');return;}
    if(!ev.target.closest('.modpanel')) corps.classList.remove('show-modules');
    var n=ev.target.closest('[data-nouvelle]'); if(n){corps.classList.add('show-nouvelle');return;}
    if(ev.target.closest('[data-fermer-modal]')||ev.target.classList.contains('modal')) corps.classList.remove('show-nouvelle');
    var o=ev.target.closest('.mopt'); if(o){document.querySelectorAll('.mopt.on').forEach(function(x){x.classList.remove('on');});o.classList.add('on');}
  });
  if(q.get('modules')) corps.classList.add('show-modules');
  if(q.get('nouvelle')) corps.classList.add('show-nouvelle');
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
        tuiles = "".join(f'<a class="tile"{go(PAGES_MODULES.get(k, ""))}><span class="modic {cle}">{MOD[k]}</span>'
                         f'<div><b>{esc(n)}</b></div>{pastille_validee(k)}<span class="chev">{icon("chevron-right")}</span></a>'
                         for k, n in mods)
        familles += (f'<div class="fam {cle}"><div class="famhead"><span class="pill {cle}">{nom}</span><span class="muted">{len(mods)} modules</span></div>'
                     f'<div class="tiles">{tuiles}</div></div>')
    # au téléphone, une famille à la fois : le même choix que les trois blocs du PC
    choix = '<div class="seg full mfam">' + "".join(
        f'<button data-fam="{cle}" class="{"on" if cle == "beton" else ""}">{nom} · {len(mods)}</button>' for cle, nom, mods in FAMILLES) + "</div>"
    familles = f'{choix}<div class="fams" data-fam="beton">{familles}</div>'
    sites = [("Réalex", "4 dossiers", 15), ("Résidence Les Tilleuls", "2 dossiers", 8), ("École Saint-Joseph", "3 dossiers", 11), ("Hall logistique", "2 dossiers", 6)]
    lst = "".join(f'<div class="site"{go("projets" if n == "Réalex" else "")}><span class="ic">{icon("building-2")}</span><div><b>{n}</b><span>{s}</span></div>'
                  f'<span class="nb">{r} rapports</span></div>' for n, s, r in sites)
    feed = [("SM", "a2", "Sophie a diffusé l'<b>indice B</b> de <b>Garde-corps mezzanine</b>", "il y a 2 h"),
            ("TD", "a3", "Thomas a créé <b>Platine d'about P3</b> dans Hall logistique", "hier, 16:40"),
            ("LL", "", "Vous avez exporté <b>Dalle R+1 · zone B</b> en .stab", "hier, 11:05")]
    act = "".join(f'<div class="feed"><span class="avatar sm {c}">{i}</span><div>{t}<span class="when">{w}</span></div></div>' for i, c, t, w in feed)
    mhead = (f'<header class="mhead"><div class="logo">{LOGO}</div><span class="ttl">Stab Méthodes</span><span class="spacer"></span>'
             f'{theme_btn()}<a class="btn icon ghost" title="Notifications">{icon("bell")}</a><span class="avatar">LL</span></header>')
    corps = (f'<div class="app rail">{rail("accueil")}<main class="main">{mhead}<div class="page">'
             f'<div class="hello"><div><div class="h1">Bonjour</div><div class="sub">Vendredi 6 novembre · 3 rapports en cours, 1 à diffuser</div></div>'
             f'<span class="spacer"></span><div class="bigsearch">{icon("search")}<span>Rechercher un module, un chantier, une note…</span><span class="kbd">Ctrl K</span></div>'
             f'<a class="btn primary big" data-nouvelle="1" title="Nouvelle note">{icon("plus")}<span class="lbl">Nouvelle note</span></a></div>'
             f'<div><div class="sechead"><div class="h2">Reprendre</div><a class="btn ghost sm">Tout voir {icon("arrow-up-right")}</a></div>'
             f'<div class="resume">{reprendre}</div><div class="sechead"><div class="h2">Modules</div><span class="okleg">{icon("badge-check")}validé par un administrateur</span></div>{familles}</div>'
             f'<aside class="side"><div class="card"><div class="sechead"><div class="h3">Chantiers actifs</div><a class="btn ghost sm">{icon("plus")}</a></div>{lst}</div>'
             f'<div class="card"><div class="sechead"><div class="h3">Activité du bureau</div></div>{act}</div></aside>'
             f'</div>{mnav("accueil")}</main></div>{nouvelle_note()}')
    script = """
(function(){
  var fams=document.querySelector('.fams');
  document.querySelectorAll('.mfam button').forEach(function(b){b.addEventListener('click',function(){
    fams.dataset.fam=b.dataset.fam;
    document.querySelectorAll('.mfam button').forEach(function(x){x.classList.toggle('on',x===b);});
  });});
})();"""
    return "Accueil", corps, script


def nouvelle_note() -> str:
    """« Nouvelle note » : on choisit où la ranger (chantier, dossier), on la nomme, on choisit le module — un seul écran."""
    opts = ""
    for cle, nom, mods in FAMILLES:
        opts += f'<div class="mlabel">{nom}</div>' + "".join(
            f'<div class="mopt{" on" if k == "poutre" else ""}"><span class="modic {cle}">{MOD[k]}</span><span>{esc(n)}</span>{pastille_validee(k)}</div>'
            for k, n in mods)
    return (f'<div class="modal"><div class="mcard"><div class="mh"><span class="h2">Nouvelle note</span>'
            f'<a class="btn icon ghost" data-fermer-modal="1" title="Fermer">{icon("x")}</a></div><div class="mb">'
            f'<div class="mrow3"><label>Chantier{sel("Réalex")}</label><label>Dossier{sel("Étude de la dalle")}</label>'
            f'<label>Nom de la note<div class="field text"><input value="Poutres R+1"></div></label></div>'
            f'<div class="mgrid">{opts}</div></div>'
            f'<div class="mf"><a class="btn" data-fermer-modal="1">Annuler</a><a class="btn primary"{go("poutre")}>{icon("plus")}Créer la note</a></div></div></div>')


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
                ("A", False, "Old", "old", "Première émission · remplacée par l'indice B", "14 oct. 2026", "LL", "", True)]
    # un indice « old » s'imprime grisé, sous un tampon OLD : l'aperçu est la première page de la VRAIE note
    tampon = f'<div class="oldprev" title="Indice old : la note s\'imprime grisée, avec le filigrane OLD"><img src="{D["note_page"]}" alt="Note, indice A"><span>OLD</span></div>'
    tl = "".join(
        f'<div class="ver{" cur" if cur else ""}{" old" if tc == "old" else ""}"><span class="indice">{i}</span><div><b>Indice {i}<span class="tag {tc}"><span class="dot"></span>{t}</span></b>'
        f'<p>{esc(p)}</p><div class="when"><span class="avatar sm {ac}">{av}</span>{w}</div>{tampon if tc == "old" else ""}</div>'
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


# ---------------------------------------------------------------- l'équipe
def equipe() -> tuple:
    """Qui entre (les adresses invitées), qui fait quoi (administrateur, membre), quels modules sont validés.
    Les personnes sont des EXEMPLES ; « Vous » est l'administrateur qui invite."""
    gens = [("LL", "", "Vous", "votre adresse", True, True, "en ligne"),
            ("CS", "a4", "Chef de service", "chef.service@exemple.be", True, True, "hier"),
            ("SM", "a2", "Sophie M.", "sophie.m@exemple.be", False, True, "il y a 2 h"),
            ("TD", "a3", "Thomas D.", "thomas.d@exemple.be", False, True, "hier"),
            ("", "", "nouveau.collegue@exemple.be", "invitation envoyée le 3 nov.", False, False, "—")]
    lignes = ""
    for ini, ac, nom, mail, admin, actif, vu in gens:
        av = f'<span class="avatar {ac}">{ini}</span>' if ini else f'<span class="avatar" style="background:var(--ink-4)">{icon("mail")}</span>'
        role = (f'<span class="role admin">{icon("shield-check")}Administrateur</span>' if admin else '<span class="role membre">Membre</span>')
        etat = '<span class="etat"><span class="dot"></span>Actif</span>' if actif else '<span class="etat inv"><span class="dot"></span>Invité</span>'
        act = '<a class="btn sm">Renvoyer</a>' if not actif else f'<span class="muted">{icon("ellipsis")}</span>'
        lignes += (f'<tr><td><div class="who2">{av}<div><b>{esc(nom)}</b><span>{esc(mail)}</span></div></div></td><td>{role}</td><td>{etat}</td>'
                   f'<td class="muted">{vu}</td><td class="n">{act}</td></tr>')
    membres = (f'<div class="card flush"><div class="sechead"><div class="h3">Membres · 4 actifs, 1 invitation</div></div><div class="tscroll"><table class="table">'
               f'<thead><tr><th>Personne</th><th>Rôle</th><th>État</th><th>Vu</th><th></th></tr></thead><tbody>{lignes}</tbody></table></div></div>')
    invite = (f'<div class="card"><div class="h3">Inviter une adresse e-mail</div><div class="invrow"><div class="field text">{icon("mail")}'
              f'<input value="" placeholder="prenom.nom@bureau.be"></div>{seg(["Membre", "Administrateur"], "Membre", full=False)}'
              f'<a class="btn primary">{icon("user-plus")}Envoyer l\'invitation</a></div>'
              f'<div class="hint">Seules les adresses invitées entrent sur le site. La personne reçoit un lien, choisit son mot de passe, et c\'est tout.</div></div>')
    mods = [(k, n) for _, _, ms in FAMILLES for k, n in ms]
    vrows = ""
    for k, n in mods:
        if k in VALIDES:
            qui, quand = VALIDES[k]
            etat, act = f'<span class="valid ok">{icon("badge-check")}Validé</span>', f'<span class="muted">{qui} · {quand}</span>'
        else:
            etat, act = f'<span class="valid non">{icon("shield-check")}À valider</span>', '<a class="btn sm">Valider</a>'
        cle = next(c for c, _, ms in FAMILLES if any(k == x for x, _ in ms))
        vrows += f'<tr><td><span class="mod"><span class="modic {cle}">{MOD[k]}</span><b style="font-weight:600">{esc(n)}</b></span></td><td>{etat}</td><td class="n">{act}</td></tr>'
    validation = (f'<div class="card flush"><div class="sechead"><div class="h3">Validation des modules</div><span class="okleg">{icon("badge-check")}visible de tous</span></div>'
                  f'<div class="tscroll"><table class="table"><thead><tr><th>Module</th><th>État</th><th class="n">Par · le</th></tr></thead><tbody>{vrows}</tbody></table></div></div>')
    droits = (f'<aside class="card"><div class="h3">Qui peut faire quoi</div><div class="rights">'
              f'<div><span class="ic">{icon("shield-check")}</span><div><b>Administrateur</b>Ouvre et modifie toutes les notes, invite et retire des personnes, nomme d\'autres administrateurs, valide les modules.</div></div>'
              f'<div><span class="ic">{icon("user")}</span><div><b>Membre</b>Crée ses notes et seul lui les modifie. Il lit les notes des chantiers.</div></div>'
              f'<div><span class="ic">{icon("share-2")}</span><div><b>Partager</b>L\'auteur d\'une note ou d\'un dossier peut autoriser un collègue à la modifier.</div></div>'
              f'<div><span class="ic">{icon("folder")}</span><div><b>Dossier personnel</b>Visible par son propriétaire et les administrateurs ; il l\'ouvre à qui il veut.</div></div>'
              f'<div><span class="ic">{icon("badge-check")}</span><div><b>Module validé</b>Pastille posée par un administrateur. Une nouvelle version du module repasse « à valider ».</div></div>'
              f'</div></aside>')
    top = (f'<header class="top"><div class="crumbs">{icon("users")}<b>Équipe du bureau</b></div><span class="spacer"></span>'
           f'<span class="muted" style="font-size:12px">Réservé aux administrateurs</span></header>')
    mhead = f'<header class="mhead"><div class="logo">{LOGO}</div><span class="ttl">Équipe</span><span class="spacer"></span></header>'
    corps = (f'<div class="app rail">{rail("equipe")}<main class="main">{top}{mhead}<div class="team"><section class="tmain">{invite}{membres}{validation}</section>'
             f'{droits}</div>{mnav("equipe")}</main></div>')
    return "Équipe", corps, ""


# ---------------------------------------------------------------- le module garde-corps
def ligne(label, champ, sym="", aide=False, full=False, src="") -> str:
    """Une ligne de saisie ; ``src`` dit d'où vient la valeur : « bureau » (la valeur des Paramètres, l'état normal : rien
    ne s'affiche) ou « modifié:<valeur du bureau> » (changée pour cette note seulement : le champ porte « modifié » et la
    valeur du bureau reste écrite en dessous)."""
    s = f'<span class="sym">{sym}</span>' if sym else ""
    a = icon("info") if aide else ""
    tag, origine = "", ""
    if src.startswith("modifié:"):
        bureau = src.split(":", 1)[1]
        tag = '<span class="src mod" title="Changé pour cette note seulement">modifié</span>'
        champ = f'<div class="avec-bureau">{champ}<span class="was">bureau : {bureau}</span></div>'
        origine = ' data-src="modifie"'
    elif src:
        origine = ' data-src="bureau"'
    return f'<div class="row{" full" if full else ""}"{origine}><label>{s}<span>{label}</span>{a}{tag}</label>{champ}</div>'


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


def symb_math(s: str) -> str:
    """Un symbole du moteur écrit comme dans la note : q_h → q<sub>h</sub>, W_el,y → W<sub>el,y</sub> (le texte est échappé)."""
    return re.sub(r"([A-Za-zσητγ])_([A-Za-z0-9éèàùστη]+(?:,[A-Za-z0-9éèàù]+)*)", r"\1<sub>\2</sub>", esc(s))


def pc(t) -> str:
    """Un taux tel qu'il s'affiche : entier, ou une décimale à la française."""
    return f"{t:.1f}".replace(".", ",") if isinstance(t, float) and t != int(t) else f"{int(t)}"


def nb(x, dec: int = 3) -> str:
    """Une valeur (coefficient, dimension, résistance) écrite à la française, SANS autre arrondi que ``dec`` décimales :
    1,25 reste 1,25 et 12,96 reste 12,96. ``pc`` est réservé aux taux — il arrondit à une décimale et ferait de
    γ_M2 = 1,25 un « 1,2 ». Le chiffre affiché est relu et comparé à la valeur : un écart arrête la construction."""
    s = f"{x:.{dec}f}".rstrip("0").rstrip(".")
    assert abs(float(s) - x) <= 0.5 * 10 ** -dec + 1e-12, (x, s)
    return s.replace(".", ",")


def ligne_res(nom: str, piece: str, taux, niv: str, calcul: str = "", pk: str = "", gov: bool = False, leger: str = "") -> str:
    """Une vérification du résultat, sur DEUX lignes : le nom et la pièce, puis la jauge et le taux. La ligne entière ouvre
    le calcul détaillé ; la pièce, quand elle se choisit (profilé, armatures), ouvre le sélecteur. Composant commun à
    tous les modules (``piece`` est déjà du HTML)."""
    etoile = f'<span class="leger" title="{esc(leger)}">{icon("star")}</span>' if leger else ""
    if pk:
        p = f'<button class="prof" data-pk="{pk}" title="Choisir">{etoile}<span>{piece}</span>{icon("chevron-down")}</button>'
    else:
        p = f'<span class="prof"><span>{piece}</span></span>'
    ouvre = f' data-calcul="{calcul}" title="Voir le calcul"' if calcul else ""
    txt = f"{pc(taux)}{NBSP}%" if taux is not None else "à compléter"
    return (f'<div class="el{" gov" if gov else ""}"{ouvre}><div class="l1"><b>{esc(nom)}</b>{p}</div>'
            f'<div class="l2"><div class="bar"><b class="{niv}" style="width:{min(taux or 0, 100)}%"></b></div><span class="pct {niv}">{txt}</span></div></div>')


def lignes_fx(lignes: list) -> str:
    """Les lignes « symbole = formule = valeurs = résultat » d'un groupe (les taux en pastille de couleur)."""
    out = ""
    for li in lignes:
        parts = li["parts"]
        h = f'<span class="s">{symb_math(parts[0])}</span>' + "".join(f'<span class="eq">=</span><span>{symb_math(p)}</span>' for p in parts[1:])
        res = f'<span class="chip {li["niveau"]}">{esc(li["res"])}</span>' if li["niveau"] else f'<span class="r">{esc(li["res"])}</span>'
        out += f'<div class="fx">{h}<span class="eq">=</span>{res}</div>'
    return out


def fenetre_calcul(ident: str, titre: str, sous_titre: str, hyps: list, lignes: list, detail: str, pied: str = "") -> str:
    """La grande fenêtre de calcul, LA MÊME pour tous les modules : les hypothèses, un tableau (vérification, formule,
    valeur, limite, taux aligné à droite), puis le détail des formules sur deux colonnes."""
    hyp = "".join(f"<span>{h}</span>" for h in hyps)
    rows = "".join(
        f'<tr class="{"gov" if li.get("gov") else ""}"><td class="nm">{li["nom"]}{("<span>" + li["sous"] + "</span>") if li.get("sous") else ""}</td>'
        f'<td class="fo">{li["formule"]}</td><td class="n v">{li["valeur"]}</td><td class="n">{li["limite"]}</td>'
        f'<td class="n">{chip(pc(li["taux"]), li["niveau"])}</td></tr>' for li in lignes)
    table = (f'<table class="vtab"><thead><tr><th>Vérification</th><th class="fo">Formule</th><th class="n">Valeur</th><th class="n">Limite</th>'
             f'<th class="n">Taux</th></tr></thead><tbody>{rows}</tbody></table>')
    return (f'<div class="flyout calcul" id="ck-{ident}"><div class="fl-h"><div><b>{titre}</b><span>{sous_titre}</span></div>'
            f'<button class="pcoll" data-fermer="1" title="Fermer">{icon("x")}</button></div><div class="ck-hyp">{hyp}</div>'
            f'<div class="ck-body">{table}<div class="ck-sub">Le détail du calcul</div><div class="ck-cols">{detail}</div>{pied}</div></div>')


def selecteur(ident: str, titre: str, sous_titre: str, filtres: str, rows: list, debut: int, unite: str) -> str:
    """Le sélecteur, LE MÊME pour les profilés et les armatures : trié par ordre de grandeur, le retenu (✓) et l'optimal
    (★, le plus léger qui tient) marqués, la molette fait défiler, un clic choisit ; sous la liste, les propriétés de la
    ligne survolée (norme, sections, inerties) disent d'où viennent les écarts de taux."""
    lignes = "".join(
        f'<button class="pk-row{" cur" if r["cur"] else ""}{" opt" if r["opt"] else ""}" data-nom="{esc(r["nom"])}" data-prop="{esc(r.get("prop", ""))}">'
        f'<span class="mk">{icon("check") if r["cur"] else (icon("star") if r["opt"] else "")}</span>'
        f'<span class="nm">{esc(r["nom"])}</span><span class="kg">{r["col"]}</span>{chip(pc(r["taux"]), r["niveau"])}</button>' for r in rows)
    prop0 = next((r.get("prop", "") for r in rows if r["cur"]), "")
    return (f'<div class="flyout picker" id="pk-{ident}" data-debut="{debut}"><div class="fl-h"><div><b>{titre}</b><span>{sous_titre}</span></div>'
            f'<button class="pcoll" data-fermer="1" title="Fermer">{icon("x")}</button></div><div class="filters">{filtres}</div>'
            f'<div class="pk-list">{lignes}</div><div class="pk-prop">{prop0}</div>'
            f'<div class="fl-foot"><span>{icon("check")}retenu</span><span>{icon("star")}le plus léger qui tient</span><span>{unite} · taux</span></div></div>')


# le script commun à tous les établis : vues du dessin, onglets du téléphone, cadres repliables, volets (sélecteur et
# grande fenêtre de calcul) ; ?vue= ?tab= ?replie=1 ?profils=1 ?calcul=<élément>|1 rejouent un état pour les captures
SCRIPT_ETABLI = """
(function(){
var q=new URLSearchParams(location.search), tel=matchMedia('(max-width: 700px)');
// chaque établi est câblé pour lui-même : l'aperçu en une page en porte plusieurs (garde-corps, poutre)
document.querySelectorAll('.bench').forEach(function(bench){
  var main=bench.closest('.main')||document;
  function vue(k){bench.querySelectorAll('[data-vue]').forEach(function(e){e.classList.toggle('on',e.dataset.vue===k);});}
  bench.querySelectorAll('.ctool [data-vue]').forEach(function(e){e.addEventListener('click',function(){vue(e.dataset.vue);});});
  function tab(t){bench.dataset.tab=t;main.querySelectorAll('.mtabs button').forEach(function(b){b.classList.toggle('on',b.dataset.tab===t);});}
  main.querySelectorAll('.mtabs button').forEach(function(b){b.addEventListener('click',function(){tab(b.dataset.tab);});});
  // les cadres se replient pour rendre la largeur au dessin
  bench.querySelectorAll('[data-coll]').forEach(function(b){b.addEventListener('click',function(){bench.classList.toggle(b.dataset.coll==='g'?'sans-g':'sans-d');});});
  // les volets : le sélecteur à hauteur de sa ligne, la grande fenêtre de calcul en haut ; au téléphone, ils montent du bas
  function fermer(){bench.querySelectorAll('.flyout.on').forEach(function(f){f.classList.remove('on');});bench.querySelectorAll('.el.sel').forEach(function(e){e.classList.remove('sel');});bench.classList.remove('fly');}
  function ouvrir(id,ancre){
    fermer(); var f=document.getElementById(id); if(!f) return;
    f.classList.add('on'); bench.classList.add('fly');
    if(ancre&&ancre.classList.contains('el')) ancre.classList.add('sel');
    if(!tel.matches&&ancre&&!f.classList.contains('calcul')){var b=bench.getBoundingClientRect(),a=ancre.getBoundingClientRect();
      f.style.top=Math.max(10,Math.min(a.top-b.top-12,b.height-f.offsetHeight-10))+'px';}
    var l=f.querySelector('.pk-list'); if(l){var r=l.querySelector('.pk-row'); l.scrollTop=(r?r.offsetHeight:34)*(+f.dataset.debut||0);}
  }
  bench.querySelectorAll('[data-pk]').forEach(function(b){b.addEventListener('click',function(ev){ev.stopPropagation();ouvrir('pk-'+b.dataset.pk,b.closest('.el')||b);});});
  // la ligne ENTIÈRE ouvre le calcul détaillé
  bench.querySelectorAll('[data-calcul]').forEach(function(b){b.addEventListener('click',function(ev){ev.stopPropagation();ouvrir('ck-'+b.dataset.calcul,b);});});
  bench.querySelectorAll('[data-fermer]').forEach(function(b){b.addEventListener('click',fermer);});
  document.addEventListener('click',function(ev){if(!ev.target.closest('.flyout,[data-pk],[data-calcul]')) fermer();});
  // le sélecteur : survoler une ligne montre ses propriétés, cliquer la retient (le site relance le calcul)
  bench.querySelectorAll('.picker').forEach(function(p){var pr=p.querySelector('.pk-prop');p.querySelectorAll('.pk-row').forEach(function(r){
    r.addEventListener('mouseenter',function(){if(pr) pr.innerHTML=r.dataset.prop;});
    r.addEventListener('click',function(){p.querySelectorAll('.pk-row.cur').forEach(function(x){x.classList.remove('cur');});r.classList.add('cur');
      var b=bench.querySelector('.prof[data-pk="'+p.id.slice(3)+'"] span'); if(b) b.textContent=r.dataset.nom;});
  });});
  tab(q.get('tab')||'donnees'); if(q.get('vue')) vue(q.get('vue'));
  if(q.get('replie')) bench.classList.add('sans-g','sans-d');
  if(q.get('profils')) ouvrir('pk-'+bench.dataset.pk0,bench.querySelector('.prof[data-pk="'+bench.dataset.pk0+'"]').closest('.el'));
  var c=q.get('calcul'); if(c){c=(c==='1'?bench.dataset.ck0:c); ouvrir('ck-'+c,bench.querySelector('.el[data-calcul="'+c+'"]'));}
});
})();"""


def feuilles_dessin(vues: list, defaut: str, cartouche: str) -> str:
    """La toile : les onglets des vues (sans vignettes en double), la feuille avec son cartouche, le zoom."""
    tabs = "".join(f'<button data-vue="{v["cle"]}" class="{"on" if v["cle"] == defaut else ""}">{v["court"]}</button>' for v in vues)
    feuilles = "".join(
        f'<div class="view{" on" if v["cle"] == defaut else ""}" data-vue="{v["cle"]}"><div class="sheet"><div class="stitle"><b>{esc(v["titre"].upper())}</b>'
        f'<span>{v.get("legende", "")}</span><span class="scale">mm · à l\'échelle</span></div><div class="draw">{v["svg"]}</div>{cartouche}</div></div>'
        for v in vues)
    return (f'<section class="canvas"><div class="ctool"><div class="seg">{tabs}</div><div class="zoom"><button>{icon("minus")}</button><span>100 %</span>'
            f'<button>{icon("plus")}</button><button>{icon("maximize-2")}</button></div></div><div class="stage">{feuilles}</div></section>')


def cartouche(rapport: str, chantier: str, indice: str) -> str:
    return (f'<div class="cart"><div><span>Rapport</span><b>{rapport}</b></div><div><span>Chantier</span><b>{chantier}</b></div>'
            f'<div><span>Indice</span><b>{indice}</b></div><div><span>Date</span><b>06/11/2026</b></div><div class="mark"><i></i>Stab Méthodes</div></div>')


def barre_haut(chemin: list, titre: str, indice: str, etat: tuple, module_cle: str) -> str:
    """La barre du haut d'un module : le fil d'Ariane (chaque partie cliquable), l'indice et son statut, la pastille de
    validation du module, puis les actions : indices, .stab, imprimer, note PDF."""
    fil = "".join(f'<span{go(cible)}>{esc(nom)}</span>{icon("chevron-right")}' for nom, cible in chemin)
    if module_cle in VALIDES:
        qui, quand = VALIDES[module_cle]
        val = f'<span class="valid ok" title="Validé par {qui} le {quand}">{icon("badge-check")}Module validé</span>'
    else:
        val = f'<span class="valid non" title="Pas encore validé par un administrateur">{icon("shield-check")}Module à valider</span>'
    return (f'<header class="top"><div class="crumbs">{icon("folder")}{fil}<b>{esc(titre)}</b></div><span class="indice">{indice}</span>'
            f'<span class="tag {etat[0]} menu" title="Brouillon, diffusé ou old"><span class="dot"></span>{etat[1]}{icon("chevron-down")}</span>{val}'
            f'<span class="spacer"></span><span class="saved">{icon("circle-check")}<span class="txt">Enregistré il y a 2 min</span></span>'
            f'<span class="avstack"><span class="avatar sm">LL</span><span class="avatar sm a2">SM</span></span>'
            f'<a class="btn ghost">{icon("history")}Indices</a><a class="btn">{icon("download")}.stab</a>'
            f'<a class="btn" title="Imprimer la note">{icon("printer")}Imprimer</a><a class="btn primary">{icon("file-down")}Note PDF</a></header>')


def elements_rapport(items: list, ajout: str) -> str:
    """« Dans ce rapport » : les éléments de la note, de PLUSIEURS modules, groupés par module — on passe de l'un à
    l'autre comme d'un calque à l'autre. ``items`` : [(module, [(nom, détail, taux|None, niveau, actif, cible)])]."""
    corps, n = "", 0
    for mod_cle, liste in items:
        fam = next(c for c, _, ms in FAMILLES if any(k == mod_cle for k, _ in ms))
        nom_mod = next(nm for c, _, ms in FAMILLES for k, nm in ms if k == mod_cle)
        corps += f'<div class="nl-mod">{esc(nom_mod)}</div>'
        for nom, det, taux, niv, actif, cible in liste:
            n += 1
            t = f"{pc(taux)}{NBSP}%" if taux is not None else "à compléter"
            corps += (f'<div class="ni{" on" if actif else ""}"{go(cible)}><span class="modic {fam}">{MOD[mod_cle]}</span>'
                      f'<span class="t">{esc(nom)} <span>· {esc(det)}</span></span><span class="pct {niv}">{t}</span></div>')
    return (f'<div class="notelist"><div class="nl-h"><span class="kicker">Dans ce rapport · {n}</span></div>{corps}'
            f'<div class="nl-add">{icon("plus")}{ajout}</div></div>')


def etabli(rail_actif: str, top: str, mtop: str, donnees: str, dessin: str, resultat: str, volets: str, pk0: str, ck0: str) -> str:
    mtabs = f'<div class="mtabs"><div class="seg full"><button data-tab="donnees">Données</button><button data-tab="dessin">Dessin</button><button data-tab="resultat">Résultat</button></div></div>'
    mbar = f'<div class="mbar"><a class="btn">{icon("history")}Indices</a><a class="btn primary">{icon("file-down")}Note PDF</a></div>'
    return (f'<div class="app rail">{rail(rail_actif)}<main class="main">{top}{mtop}{mtabs}'
            f'<div class="bench" data-tab="donnees" data-pk0="{pk0}" data-ck0="{ck0}">{dessin}<section class="panel left">{donnees}</section>'
            f'<section class="panel right">{resultat}</section>{volets}</div>{mbar}</main></div>')


def module() -> tuple:
    gc = D["gc"]
    dft = D["gc_defauts"]
    nuance = dft["nuance"]
    courts = {"elevation": "Élévation", "coupe": "Coupe A-A", "plan": "Plan", "platine": "Platine", "pied": "Fixation", "section": "Section"}
    legendes = {"elevation": f'{esc(gc["type"])} · {esc(gc["phrase"])}', "coupe": "dans l'axe d'un montant", "plan": "vue de dessus",
                "platine": "platine sur la dalle, vue de dessus", "pied": "coupe à grande échelle, efforts du cas gouvernant", "section": "élément principal"}
    vues = [dict(v, court=courts.get(v["cle"], v["titre"]), legende=legendes.get(v["cle"], "")) for v in gc["vues"]]
    # les éléments du rapport (chacun vérifié par le moteur)
    items = [("garde_corps", [("G1 Façade", "barreaudage", gc["eta"], gc["niveau"], True, ""),
                              ("G2 Nez de dalle", "platine latérale", D["gc_lat"]["eta"], D["gc_lat"]["niveau"], False, ""),
                              ("G3 Retour en L", "lisses", D["gc_L"]["eta"], D["gc_L"]["niveau"], False, "")])]
    # les données
    s1 = (ligne("Type", sel("Barreaudage vertical"), full=True) + ligne("Forme", seg(["Droit", "En L", "En U"], "Droit"), full=True)
          + ligne("Hauteur", num(f"1{NBSP}100", "mm", focus=True), "H", True) + ligne("Pente", num("0", "°"), "α"))
    s4 = (ligne("Catégorie", sel("A · habitation — 0,5 kN/m"), full=True) + '<div class="grp">Sur la main courante</div>'
          + ligne("horizontale", num("0,50", "kN/m", locked=True), "q<sub>h</sub>", True)
          + ligne("verticale", num("0,50", "kN/m"), "q<sub>v</sub>") + '<div class="grp">Sur le remplissage</div>'
          + ligne("ponctuelle", num("0,50", "kN"), "Q<sub>r</sub>")
          + ligne("pression", num("0,50", "kN/m²"), "q<sub>r</sub>")
          + f'<div class="row"><label><span>Cumuler q<sub>h</sub> et q<sub>v</sub></span></label><div style="display:flex;justify-content:flex-end"><span class="switch"></span></div></div>'
          + '<div class="hint">Quatre cas distincts : <b>chaque charge agit seule</b>. q<sub>h</sub> suit la catégorie, EN 1991-1-1 Tab. 6.12.</div>')
    donnees = (f'<div class="phead"><span class="h3">Données</span>{seg(["ELS", "Eurocode"], "ELS", full=False)}'
               f'<button class="pcoll" data-coll="g" title="Replier les données">{icon("panel-left")}</button></div>'
               f'<div class="pbody">{elements_rapport(items, "Ajouter un élément")}'
               + section(1, "Garde-corps", "", s1, True)
               + section(2, "Tronçons", "3,60 m · 4 × 0,90 m")
               + section(3, "Remplissage", "barreaux · e 110 mm")
               + section(4, "Charges", "", s4, True)
               + section(5, "Fixation", "platine · 4 chevilles M12")
               + section(6, "Méthode", f"{nuance} · flèche L/{int(dft['fleche_val'])} · bureau") + "</div>")
    dessin = feuilles_dessin(vues, "elevation", cartouche("Garde-corps terrasse R+1", "Réalex", "C"))
    # le résultat : une ligne par vérification, sur deux lignes ; la ligne entière ouvre le calcul
    gouv = gc["gouvernant"].split(" · ")[0]
    cal = {"montant": "montant", "lisse": "lisse", "barreau": "barreau", "soudure": "soudure", "ancrage": "platine", "platine": "platine"}
    rows = ""
    for e in gc["elements"]:
        # la nuance est la même pour tous les éléments : elle est dans « Méthode » et en tête du calcul, pas sur chaque ligne
        nom, piece = e["nom"], e["section"]
        if piece:
            pass
        elif nom.startswith("Platine "):
            nom, piece = "Platine", nom[len("Platine "):]
        else:
            piece = e["desc"].replace("cordon d'angle ", "gorge ").replace(" chevilles", "")
        o = gc["optimaux"].get(e["cle"])
        leger = f"Plus léger qui tient : {o['nom']} · {o['taux']} %" if o and not o["retenu"] else ""
        rows += ligne_res(nom, esc(piece), e["taux"], e["niveau"], calcul=cal.get(e["cle"], ""),
                          pk=e["cle"] if e["cle"] in gc["selecteurs"] else "", gov=e["nom"] == gouv, leger=leger)
    resultat = (f'<div class="phead"><span class="h3">Résultat</span>{statut(gc["niveau"])}'
                f'<button class="pcoll" data-coll="d" title="Replier le résultat">{icon("panel-right")}</button></div>'
                f'<div class="pbody"><div class="verdict">{chip(gc["eta"], gc["niveau"], xl=True)}<div class="gov"><b>{esc(gouv)}</b>'
                f'{esc(gc["critere"])} · {symb(gc["elements"][0]["cas"])}</div></div><div class="els">{rows}</div></div>'
                f'<div class="pfoot"><div class="mass"><span>Masse indicative de l\'acier</span><b>{gc["masse"]}{NBSP}kg</b></div>'
                f'<div class="legend">{legende()}</div></div>')
    # les volets : un sélecteur par élément à profilé, une grande fenêtre de calcul par élément calculé
    noms = {e["cle"]: e["nom"] for e in gc["elements"]}
    secs = {e["cle"]: e["section"] for e in gc["elements"]}
    familles = {"montant": ["RHS", "SHS", "CHS"], "lisse": ["RHS", "SHS", "CHS"], "barreau": ["Plats", "Ronds", "Carrés"]}
    volets = ""
    for k, lst in gc["selecteurs"].items():
        i_o = next((i for i, r in enumerate(lst) if r["optimal"]), 0)
        filtres = "".join(f'<span class="fchip{" on" if j == 0 or (k == "barreau" and j == 1) else ""}">{f}</span>' for j, f in enumerate(familles[k]))
        filtres += f'<span class="fchip{" on" if k == "montant" else ""}">{icon("check") if k == "montant" else ""}Formés à froid</span>'
        rows_pk = [{"nom": r["nom"], "col": kg(r["kgm"]), "taux": r["eta"], "niveau": r["niveau"], "cur": r["retenue"], "opt": r["optimal"],
                    "prop": r["prop"]} for r in lst]
        volets += selecteur(k, f"{noms.get(k, k)} · profilé", "triés par masse · la molette fait défiler", filtres, rows_pk, max(0, i_o - 4), "kg/m")
    hyps = [f"Acier <b>{nuance}</b> · f<sub>y</sub> = {pc(gc['fy'])} N/mm²", f"Méthode <b>ELS</b> · σ<sub>adm</sub> = f<sub>y</sub> / {pc(dft['gamma_adm'])}",
            f"Flèche ≤ <b>L/{int(dft['fleche_val'])}</b>"]
    titres = {"montant": f"{noms['montant']} · {secs['montant']}", "lisse": f"{noms['lisse']} · {secs['lisse']}",
              "barreau": f"{noms['barreau']} · {secs['barreau']}", "platine": "Fixation · platine et ancrages", "soudure": "Soudure du montant sur la platine"}
    for k, c in gc["calculs"].items():
        lignes = [{"nom": r["verif"], "formule": (f"{symb_math(r['sym'])} = {symb_math(r['formule'])}" if r["formule"] else symb_math(r["sym"])),
                   "valeur": r["valeur"], "limite": (f"{symb_math(r['sym_lim'])} = " if r["sym_lim"] else "") + r["limite"],
                   "taux": r["taux"], "niveau": r["niveau"], "gov": i == 0} for i, r in enumerate(gc["tableau"].get(k, []))]
        detail = "".join(f'<div class="ck-g"><div class="ck-t">{esc(g["titre"])}</div>{lignes_fx(g["lignes"])}</div>' for g in c["groupes"])
        pied = f'<div class="ck-autres">Autres cas : {symb_math(c["autres"])}</div>' if c["autres"] else ""
        sous = f"{symb_math(c['cas'])} · les formules de la note" if c["cas"] else "les formules de la note"
        volets += fenetre_calcul(k, titres.get(k, k), sous, hyps, lignes, detail, pied)
    top = barre_haut([("Réalex", "projets"), ("Garde-corps", "projets")], "Garde-corps terrasse R+1", "C", ("draft", "Brouillon"), "garde_corps")
    mtop = (f'<header class="mtop"><a class="btn icon ghost"{go("projets")} title="Retour aux projets">{icon("arrow-left")}</a><div class="t"><b>Garde-corps terrasse R+1</b><span>Réalex · indice C · brouillon</span></div>'
            f'{chip(gc["eta"], gc["niveau"])}</header>')
    corps = etabli("modules", top, mtop, donnees, dessin, resultat, volets, "montant", "montant")
    return "Garde-corps terrasse R+1", corps, SCRIPT_ETABLI


def toutes_les_pages() -> list:
    """Les pages de la maquette, dans l'ordre : (nom de fichier, fonction). Les pages de module vivent dans leur propre
    fichier et importent les composants d'ici : on les charge ici seulement, pour éviter l'import circulaire."""
    import page_parametres
    import page_poutre
    return [("accueil", accueil), ("projets", projets), ("garde-corps", module), ("poutre", page_poutre.poutre),
            ("equipe", equipe), ("parametres", page_parametres.parametres)]


def main():
    sortie = os.path.join(ICI, "pages")
    os.makedirs(sortie, exist_ok=True)
    for nom, fn in toutes_les_pages():
        with open(os.path.join(sortie, nom + ".html"), "w", encoding="utf-8") as f:
            f.write(page(*fn()))
        print(nom, os.path.getsize(os.path.join(sortie, nom + ".html")) // 1024, "ko")


if __name__ == "__main__":
    main()

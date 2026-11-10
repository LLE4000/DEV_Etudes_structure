# -*- coding: utf-8 -*-
"""Les briques partagées de l'établi compact (api_web/etabli_compact.py) : la colonne de gauche COMPACTE des dalles et
des poutres (docs/site/DALLES_FUSION.md § 1, revue avec l'ingénieur le 09/11/2026), sans rien changer à l'arborescence
que d'autres modules emploient encore.

- ``arbre(…, compact=False)``, ``tete_resultats(nom, dims)`` et ``menu(items, titre)`` rendent EXACTEMENT le HTML
  d'avant : un gel (l'empreinte du rendu de quatre arborescences et de deux têtes, relevée sur le code du 09/11/2026
  avant la fusion — commit bdb4efa) et le menu « ⋯ » écrit en toutes lettres ;
- ``arbre(…, compact=True)`` (``arbre_compact``) : ni en-tête « LISTE (n) », ni arborescence ; UNE LIGNE ``.et-elsel``
  (que js/etabli.js pose dans l'en-tête du panneau) : le NOM de l'ouvert en champ (verrouillé : en lecture, un cadenas),
  la LISTE des éléments en menu (chacun ``data-activer``, l'ouvert coché, puis « Ajouter … »), COPIER et SUPPRIMER
  l'ouvert (éteint quand il est seul ; sa confirmation nomme l'élément) ; puis les rubriques ;
- les attributs rendus sont ceux que js/etabli.js lit (agirElement, la ligne de l'élément) : deux copies, liées ici ;
- un nom piégé (balise, guillemet) ne produit ni balise ni attribut ;
- ``tete_resultats`` n'a plus de champ : un titre qui se lit ;
- cinq MUTATIONS (en mémoire, le dépôt intact) : une altération du module fait rougir le contrôle qui la garde.

    python3 tests/test_etabli_compact.py
"""
import contextlib
import hashlib
import inspect
import os
import re
import sys
import textwrap
from html.parser import HTMLParser

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

from api_web import commun as C  # noqa: E402
from api_web import etabli_compact as E  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:300]}"))


class _Balises(HTMLParser):
    """Les balises ouvrantes d'un fragment, leurs attributs DÉCODÉS (ce que le navigateur lirait)."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.balises = []

    def handle_starttag(self, tag, attrs):
        self.balises.append((tag, dict(attrs)))

    handle_startendtag = handle_starttag


def balises(h):
    p = _Balises()
    p.feed(h)
    p.close()
    return p.balises


# ------------------------------------------------------------------ 1. le gel : ce qui ne devait pas bouger
EXEMPLES = [
    ("POUTRE", [("V1", "· 25 × 50 cm", False)], 0, '<div class="sec">corps</div>', "poutre"),
    ("POUTRE", [("V1", "· 25 × 50 cm", False), ("V2 <b>", "· 30 × 60 cm", False), ("V3", "· 20 × 40 cm", False)], 1, "<p>x</p>", "poutre"),
    ("DALLE", [("Dalle 1", "· 100 × 20 cm", True), ("Dalle 2", "· 100 × 22 cm", False)], 0, "<p>y</p>", "dalle"),
    ("PREDALLE", [("Prédalle « 1 »", "· 100 × 22 cm", False)], 0, "", "predalle"),
]
GEL_AVANT = "bb495636a31fd4edc3d7bd5e1e9472307f3874a8162627bed9e8b626640c030b"


def ctl_gel():
    rendus = [E.arbre(getattr(E, g), els, a, corps, marque) for g, els, a, corps, marque in EXEMPLES]
    rendus += [E.tete_resultats("V1", "25 × 50 cm · C30/37 · B500"), E.tete_resultats("Dalle &amp; 1", "x")]
    empreinte = hashlib.sha256("\x1e".join(rendus).encode()).hexdigest()
    return empreinte == GEL_AVANT, empreinte


chk("arbre(compact=False) et tete_resultats(nom, dims) rendent le HTML d'avant la fusion (gel du 09/11/2026, à l'octet)", *ctl_gel())
_avant = [E.arbre(getattr(E, g), els, a, corps, marque) for g, els, a, corps, marque in EXEMPLES]
chk("… l'arbre d'avant garde son en-tête « POUTRES (n) », sa ligne d'élément et « Ajouter une poutre »",
    all('class="et-arbre-h"' in r and 'class="et-pt-add" data-ajouter="1"' in r for r in _avant)
    and 'class="et-pt-nom"' in _avant[0] and len(re.findall(r'<div class="et-pt(?: on)?"', _avant[0])) == 1)


def ctl_menu():
    """menu(items, titre), sans les deux arguments ajoutés le 09/11/2026 : le HTML d'avant, écrit en toutes lettres."""
    m = E.menu("<i>x</i>", "Actions « a » & b")
    attendu = ('<div class="et-mn"><button type="button" class="et-mn-b" data-menu="1" aria-haspopup="menu" aria-expanded="false" '
               'title="Actions « a » &amp; b">' + C.icon("ellipsis") + '</button><div class="et-mn-l" role="menu" hidden><i>x</i></div></div>')
    return m == attendu, m


chk("menu(items, titre) sans « bouton » ni « classe » : le menu « ⋯ » d'avant, à l'octet", *ctl_menu())
_m2 = E.menu("<i>x</i>", "t", "<b>B</b>", "et-elt-liste")
chk("menu(…, bouton, classe) : le contenu du bouton remplace « ⋯ », la classe s'ajoute au menu, le reste ne change pas",
    _m2 == ('<div class="et-mn et-elt-liste"><button type="button" class="et-mn-b" data-menu="1" aria-haspopup="menu" aria-expanded="false" '
            'title="t"><b>B</b></button><div class="et-mn-l" role="menu" hidden><i>x</i></div></div>'), _m2)

# ------------------------------------------------------------------ 2. la ligne de l'élément : l'anatomie
SQUELETTE = re.compile(r'<div class="et-arbre et-arbre-c" data-(?P<marque>\w+)="(?P<actif>\d+)"><div class="et-elsel" data-elements="(?P<n>\d+)">'
                       r'(?P<ligne>.*?<button type="button" class="et-elt-b danger"[^>]*>.*?</button>)</div>(?P<corps>.*)</div>', re.S)
LIGNE = re.compile(r'(?P<champ><input [^>]*>|<span class="et-elt-nom lu"[^>]*>.*?</span></span>)'
                   r'(?P<liste><div class="et-mn et-elt-liste">.*</div></div>)'
                   r'(?P<copier><button type="button" class="et-elt-b" [^>]*>.*?</button>)'
                   r'(?P<supprimer><button type="button" class="et-elt-b danger" [^>]*>.*?</button>)', re.S)
BOUTON_LISTE = re.compile(r'<div class="et-mn et-elt-liste"><button type="button" class="et-mn-b" data-menu="1" aria-haspopup="menu" '
                          r'aria-expanded="false" title="([^"]*)">(.*?)</button><div class="et-mn-l" role="menu" hidden>(.*)</div></div>', re.S)
ITEM = re.compile(r'<button type="button" role="menuitemradio" aria-checked="(true|false)" class="et-elt-i( on)?" data-activer="(\d+)" '
                  r'title="([^"]*)">(.*?)<span class="nom">(.*?)</span><span class="dim">(.*?)</span></button>', re.S)


def anatomie(h):
    """Les morceaux de l'arbre compact (ou None) : la marque, le rang, le nombre, la ligne découpée, la liste, le corps."""
    s = SQUELETTE.fullmatch(h)
    if not s:
        return None
    li = LIGNE.fullmatch(s["ligne"])
    if not li:
        return None
    b = BOUTON_LISTE.fullmatch(li["liste"])
    if not b:
        return None
    items_html = b.group(3)
    items = ITEM.findall(items_html)
    reste = ITEM.sub("", items_html)
    return {"marque": s["marque"], "actif": s["actif"], "n": s["n"], "corps": s["corps"], "champ": li["champ"], "copier": li["copier"],
            "supprimer": li["supprimer"], "titre_liste": b.group(1), "bouton": b.group(2), "items": items, "apres_items": reste}


UN = [("Dalle 1", "· 100 × 20 cm", False)]
CORPS = '<div class="sec" data-sec="dl">corps</div>'
AJOUTER_DALLE = '<button type="button" role="menuitem" class="" data-ajouter="1">' + C.icon("plus") + "Ajouter une dalle</button>"


def ctl_un():
    """Un élément : la ligne entière, morceau par morceau."""
    h = E.arbre(E.DALLE, UN, 0, CORPS, "dalle", compact=True)
    a = anatomie(h)
    if a is None:
        return False, h[:300]
    champ = balises(a["champ"])[0]
    return (a["marque"] == "dalle" and a["actif"] == "0" and a["n"] == "1" and a["corps"] == CORPS
            and champ[0] == "input" and champ[1].get("class") == "et-elt-nom" and champ[1].get("data-chemin") == "nom"
            and champ[1].get("data-sorte") == "texte" and champ[1].get("value") == "Dalle 1"
            and a["bouton"] == C.icon("chevron-down") and a["titre_liste"] == "Dalles de la note : en ouvrir une autre, en ajouter une"
            and a["items"] == [("true", " on", "0", "Ouvert : Dalle 1 · 100 × 20 cm", C.icon("check"), "Dalle 1", "· 100 × 20 cm")]
            and a["apres_items"] == AJOUTER_DALLE
            and a["copier"] == ('<button type="button" class="et-elt-b" data-dupliquer-index="0" title="Copier la dalle (la copie s\'ajoute juste '
                                'après)" aria-label="Copier la dalle">' + C.icon("copy") + "</button>"), a)


def ctl_un_supprimer():
    """Un élément : Supprimer est là, ÉTEINT (une note garde au moins un élément), et le dit."""
    a = anatomie(E.arbre(E.DALLE, UN, 0, CORPS, "dalle", compact=True))
    if a is None:
        return False, "anatomie illisible"
    b = balises(a["supprimer"])[0][1]
    return (b.get("data-supprimer") == "0" and "disabled" in b and b.get("title") == "Une note garde au moins une dalle"
            and b.get("data-confirmer") == "Supprimer la dalle « Dalle 1 » de la note ?" and b.get("aria-label") == "Supprimer la dalle"), b


chk("compact, un élément : la ligne .et-elsel (data-elements=\"1\") puis les rubriques, dans .et-arbre-c marqué data-dalle=\"0\" ; le NOM en "
    "champ (data-chemin=\"nom\", data-sorte=\"texte\") ; la LISTE en menu (un chevron, sans compte ; la dalle ouverte cochée, "
    "data-activer=\"0\", puis « Ajouter une dalle », data-ajouter) ; COPIER (data-dupliquer-index=\"0\")", *ctl_un())
chk("compact, un élément : SUPPRIMER présent mais éteint (disabled, « Une note garde au moins une dalle »), sa confirmation prête",
    *ctl_un_supprimer())
_h1 = E.arbre(E.DALLE, UN, 0, CORPS, "dalle", compact=True)
chk("compact : plus rien de l'ancien écran — ni en-tête « DALLES (n) », ni ligne d'arborescence, ni « Ajouter » en bas, ni segments, "
    "ni menu « ⋯ », ni « Renommer », ni marque .seul / .et-seul",
    not any(x in _h1 for x in ("et-arbre-h", "et-pt", 'class="seg"', "seul", "data-renommer", C.icon("ellipsis"), "Renommer")), _h1[:300])
chk("compact : arbre(…, compact=True) est arbre_compact(…) — une seule fabrique de la ligne",
    _h1 == E.arbre_compact(E.DALLE, UN, 0, CORPS, "dalle"))

# ------------------------------------------------------------------ 3. deux éléments, trois avec un verrou
DEUX = [("Dalle 1", "· 100 × 20 cm", False), ("Dalle 2", "· 100 × 22 cm", False)]


def ctl_deux():
    """Deux éléments, le second ouvert : le compte « 2 » sur le bouton de la liste ; l'ouvert = actif (coché, son nom
    dans le champ, copier et supprimer le visent) ; supprimer ALLUMÉ, sa confirmation nomme l'élément."""
    a = anatomie(E.arbre(E.DALLE, DEUX, 1, "<p>c</p>", "dalle", compact=True))
    if a is None:
        return False, "anatomie illisible"
    sup = balises(a["supprimer"])[0][1]
    return (a["actif"] == "1" and a["n"] == "2" and a["bouton"] == C.icon("chevron-down") + '<span class="n">2</span>'
            and [(c, on, i) for c, on, i, *_ in a["items"]] == [("false", "", "0"), ("true", " on", "1")]
            and [x[4] for x in a["items"]] == ['<span class="ic"></span>', C.icon("check")]
            and [x[3] for x in a["items"]] == ["Ouvrir cette dalle : Dalle 1 · 100 × 20 cm", "Ouvert : Dalle 2 · 100 × 22 cm"]
            and balises(a["champ"])[0][1].get("value") == "Dalle 2" and 'data-dupliquer-index="1"' in a["copier"]
            and sup.get("data-supprimer") == "1" and "disabled" not in sup and sup.get("title") == "Supprimer la dalle"
            and sup.get("data-confirmer") == "Supprimer la dalle « Dalle 2 » de la note ?" and a["corps"] == "<p>c</p>"), a


chk("compact, deux éléments : le compte « 2 » sur le bouton de la liste ; la seconde ouverte cochée (aria-checked, .on), la première "
    "« Ouvrir cette dalle » ; le champ, Copier et Supprimer visent l'ouverte ; Supprimer ALLUMÉ, sa confirmation nomme « Dalle 2 »", *ctl_deux())
TROIS = [("Dalle 1", "· 100 × 20 cm", False), ("Dalle <2>", "· 100 × 22 cm", True), ("Dalle 3", "· 120 × 25 cm", False)]
_a3 = anatomie(_h3 := E.arbre(E.DALLE, TROIS, 1, "<p>corps</p>", "dalle", compact=True))
chk("compact, l'ouverte VERROUILLÉE : son nom se lit (span.et-elt-nom.lu, un cadenas devant, aucun champ nulle part) ; le cadenas aussi "
    "dans la liste, sur elle seule ; copier et supprimer restent (le verrou garde les données, pas l'élément)",
    _a3 is not None and "<input" not in _h3 and 'data-chemin="nom"' not in _h3
    and _a3["champ"].startswith('<span class="et-elt-nom lu" title="La dalle est verrouillée')
    and _a3["champ"].endswith(C.icon("lock") + "<span>Dalle &lt;2&gt;</span></span>")
    and [x[4] for x in _a3["items"]] == ['<span class="ic"></span>', C.icon("check") + C.icon("lock"), '<span class="ic"></span>']
    and _a3["bouton"].endswith('<span class="n">3</span>') and 'data-dupliquer-index="1"' in _a3["copier"]
    and "disabled" not in balises(_a3["supprimer"])[0][1], _a3)
chk("compact : « corps » (les rubriques) vient après la ligne, une seule fois", _h3.count("<p>corps</p>") == 1
    and _h3.index("et-elsel") < _h3.index("<p>corps</p>") and _h3.endswith("<p>corps</p></div>"))
_hors = [anatomie(E.arbre(E.DALLE, TROIS, r_, "", "dalle", compact=True)) for r_ in (9, -3)]
chk("compact : un rang hors liste est ramené dans la liste (jamais d'exception) : 9 → la dernière, −3 → la première, et c'est elle qui est cochée",
    all(_hors) and [x["actif"] for x in _hors] == ["2", "0"]
    and [[i for c, _, i, *_ in x["items"] if c == "true"] for x in _hors] == [["2"], ["0"]], _hors)
_ap = anatomie(_hp := E.arbre(E.POUTRE, [("V1", "· 25 × 50 cm", False)], 0, "", "poutre", compact=True))
chk("compact, la poutre : les mots de son genre (data-poutre, « Nom de cette poutre », « Poutres de la note », « Ajouter une poutre », "
    "« Copier la poutre », « Supprimer la poutre « V1 » de la note ? », « Une note garde au moins une poutre »)",
    _ap is not None and _ap["marque"] == "poutre" and 'aria-label="Nom de cette poutre"' in _ap["champ"]
    and _ap["titre_liste"].startswith("Poutres de la note") and _ap["apres_items"].endswith("Ajouter une poutre</button>")
    and 'aria-label="Copier la poutre"' in _ap["copier"]
    and balises(_ap["supprimer"])[0][1].get("data-confirmer") == "Supprimer la poutre « V1 » de la note ?"
    and balises(_ap["supprimer"])[0][1].get("title") == "Une note garde au moins une poutre", _hp[:300])

# ------------------------------------------------------------------ 4. les attributs : ceux que l'écran lit (deux copies, liées)
ATTRS = {"data-elements", "data-chemin", "data-sorte", "data-menu", "data-activer", "data-ajouter", "data-dupliquer-index",
         "data-supprimer", "data-confirmer"}
_lus = {a for a in re.findall(r'\s(data-[a-z-]+)=', E.arbre(E.DALLE, TROIS, 0, "", "dalle", compact=True) + E.arbre(E.DALLE, UN, 0, "", "dalle", compact=True))
        if a != "data-dalle"}
chk("compact : les attributs data-* rendus sont exactement ceux de la ligne (elements, chemin, sorte, menu, activer, ajouter, "
    "dupliquer-index, supprimer, confirmer) — plus de data-renommer", _lus == ATTRS, _lus ^ ATTRS)
_js = open(os.path.join(RACINE, "site", "web", "js", "etabli.js"), encoding="utf-8").read()
# (lot S4 : l'établi est un écran montable, ses fonctions sont dans ouvrirEtabli — la fin de agirElement est l'accolade
#  à SA propre indentation ; ses écouteurs portent le signal du montage)
_agir = re.search(r"\n( *)function agirElement\(ev\) \{.*?\n\1\}", _js, re.S)
chk("js/etabli.js lit ces attributs-là : agirElement (supprimer + sa confirmation, activer, ajouter, copier) — et la ligne de l'élément "
    "posée dans l'en-tête lui passe ses clics",
    _agir is not None and all(x in _agir.group(0) for x in ('closest("[data-supprimer]")', "dataset.confirmer", 'closest("[data-activer]")',
                                                             'closest("[data-ajouter]")', 'closest("[data-dupliquer-index]")'))
    and re.search(r'ELT\.addEventListener\("click", \(ev\) => \{ agirElement\(ev\); \}(, \{ signal \})?\)', _js) is not None)

# ------------------------------------------------------------------ 5. les noms piégés
PIEGES = ['<img src=x onerror=alert(1)>', 'x" onfocus="alert(1)', "Dalle & 'q'"]


def ctl_pieges():
    """Chaque nom piégé, ouvert puis fermé, déverrouillé puis verrouillé : aucune balise <img>, aucun attribut « on… » ;
    le navigateur relit le nom EXACT dans le champ et dans la confirmation."""
    fautes = []
    for v in (False, True):
        els = [(n, "· 1 × 2 cm", v) for n in PIEGES]
        for actif in range(len(PIEGES)):
            h = E.arbre(E.DALLE, els, actif, "", "dalle", compact=True)
            bs = balises(h)
            if "<img" in h or any(t == "img" for t, _ in bs) or any(k.startswith("on") for _, at in bs for k in at):
                fautes.append(("balise ou attribut", v, actif))
            champ = [at for t, at in bs if t == "input" and at.get("class") == "et-elt-nom"]
            if (champ and champ[0].get("value") != PIEGES[actif]) or (bool(champ) == v):
                fautes.append(("champ", v, actif, champ))
            conf = [at.get("data-confirmer") for _, at in bs if "data-supprimer" in at]
            if conf != [f"Supprimer la dalle « {PIEGES[actif]} » de la note ?"]:
                fautes.append(("confirmation", v, actif, conf))
            if h.count(f'<span class="nom">{C.esc(PIEGES[actif])}</span>') != 1:
                fautes.append(("liste", v, actif))
    return not fautes, fautes


chk("compact : un nom piégé (« <img src=x onerror=alert(1)> », un guillemet qui fermerait l'attribut, « & ') ne produit ni balise ni "
    "attribut — échappé dans le champ, la liste, les titres et la confirmation ; le navigateur relit le nom exact", *ctl_pieges())

# ------------------------------------------------------------------ 6. la tête des résultats : un titre, plus de champ
t0 = E.tete_resultats("Dalle &amp; 1", "100 × 20 cm · C30/37 · B500")
chk("tete_resultats(nom, dims) : le titre d'avant (<b>nom</b>, <span>dims</span>), sans champ — le nom se modifie à UN endroit, la ligne de l'élément",
    t0 == '<div class="et-rp"><b>Dalle &amp; 1</b><span>100 × 20 cm · C30/37 · B500</span></div>', t0)
chk("tete_resultats : plus de paramètre « chemin_nom » ni « verrou » (la tête des résultats ne porte plus jamais le champ)",
    list(inspect.signature(E.tete_resultats).parameters) == ["nom", "dims"], inspect.signature(E.tete_resultats))

# ------------------------------------------------------------------ 7. les mutations : une altération du module fait rougir un contrôle
print("\n=== 6 bis. Le rangement du 10/11/2026 : les styles de l'ANCIENNE arborescence, retirés ===")
# css/etabli.css n'est chargé que par les pages des établis compacts (construire.ETABLI_COMPACT) ; chacun de ces modules
# rend sa colonne COMPACTE, en dur (``E.arbre(…, compact=True)``) ; la seule à rendre encore l'ancienne arborescence est
# api_web/predalle.py, dont la page REDIRIGE vers les dalles (construire.REDIRECTIONS) et ne charge pas ce fichier. Les
# règles .et-pt…, .et-arbre-h n'avaient donc plus rien à habiller : retirées de css/etabli.css et css/modules/dalle.css
# (la pastille .et-pt-e et la couleur --et-pt-bg, employées, restent). Ce contrôle garde les trois prémisses.
sys.path.insert(0, os.path.join(RACINE, "site", "web"))
import importlib  # noqa: E402
with contextlib.redirect_stdout(open(os.devnull, "w")):
    CO = importlib.import_module("construire")
_ANC = re.compile(r"\.et-pt(?!-e\b)[-\w]*|\.et-arbre-h")


def _css(chemin):
    with open(os.path.join(RACINE, "site", "web", chemin), encoding="utf-8") as f:
        return re.sub(r"/\*.*?\*/", "", f.read(), flags=re.S)


def ctl_css_ancienne():
    # (l'en-tête « JUSTIFICATIONS (n) » du bloc des fiches porte encore la classe .et-arbre-h : ses règles, rendues à lui
    # seul sous « .et-fiches > », ne sont pas celles de l'ancienne arborescence — contrôle suivant)
    return [(c, m) for c in ("css/etabli.css", "css/modules/dalle.css", "css/modules/poutre.css")
            if os.path.exists(os.path.join(RACINE, "site", "web", c))
            for m in _ANC.findall(re.sub(r"\.et-fiches\s*>\s*\.et-arbre-h", "", _css(c)))]


chk("css/etabli.css et les feuilles des établis compacts ne portent plus aucune règle de l'ancienne arborescence "
    "(.et-pt…, .et-arbre-h) ; la pastille .et-pt-e reste", not ctl_css_ancienne() and ".et-pt-e" in _css("css/etabli.css"),
    ctl_css_ancienne())
# (le premier retrait avait emporté, avec « .et-pt-add », la règle à plusieurs sélecteurs qui le portait : « + Lit »,
# « + Section » retombaient à 26 px au téléphone — le harnais « formats » l'a trouvé ; la règle est gardée ici)
chk("le retrait n'a emporté aucune règle vivante : au téléphone, « + Lit », « Retirer », « Large », « Justifier » restent à 44 px",
    '[data-etabli="compact"] .et-add, [data-etabli="compact"] .et-ret, [data-etabli="compact"] .et-large, '
    '[data-etabli="compact"] .et-just { height: 44px; font-size: 13px; }' in _css("css/etabli.css"))
# (relecture du 10/11/2026) … ni le style d'un fragment VIVANT : le bloc « Justifications » des fiches de la bibliothèque
# (api_web/fiches.bloc, poutre) a pour en-tête un .et-arbre-h — son style était parti avec l'ancienne arborescence, et
# l'œil lisait « Justifications1 » en texte courant (ni capitales, ni pastille du compte). Chaque classe de l'établi
# (« et-… ») que le bloc rend a sa règle dans css/etabli.css, et l'en-tête a la sienne : capitales, et le compte en pastille.
FI = importlib.import_module("api_web.fiches")
_bloc = FI.bloc([{"titre": "Enrobage poutre XC1 C30/37", "symbole": "c_nom", "texte": "", "valeur": 26, "unite": "mm"}])
_cls_bloc = sorted({k for a in re.findall(r'class="([^"]+)"', _bloc) for k in a.split() if k.startswith("et-")})
_css_et = _css("css/etabli.css")
_sans_regle = [k for k in _cls_bloc if not re.search(r"\." + re.escape(k) + r"(?![-\w])[^{}]*\{", _css_et)]
_tete = re.search(r'<div class="et-fiches"><div class="([^"]+)"><span>Justifications</span><span class="n">1</span></div>', _bloc)
chk("le bloc « Justifications » des fiches (fiches.bloc) garde son style : chaque classe « et-… » qu'il rend a sa règle, "
    "son en-tête s'écrit en capitales et son compte en pastille (« JUSTIFICATIONS 1 », pas « Justifications1 »)",
    _cls_bloc and not _sans_regle and _tete is not None
    and re.search(r"\.et-fiches\s*>\s*\." + _tete.group(1) + r"\s*\{[^}]*display:\s*flex[^}]*text-transform:\s*uppercase", _css_et) is not None
    and re.search(r"\.et-fiches\s*>\s*\." + _tete.group(1) + r"\s+\.n\s*\{[^}]*background:", _css_et) is not None,
    {"classes": _cls_bloc, "sans_regle": _sans_regle, "tete": _tete and _tete.group(1)})
_src = {cle: inspect.getsource(importlib.import_module(f"api_web.{cle}").arbre) for cle in CO.ETABLI_COMPACT}
chk("chaque établi compact (construire.ETABLI_COMPACT) rend sa colonne compacte, en dur — et charge seul css/etabli.css ; "
    "la page de l'ancien module Prédalles redirige",
    set(CO.ETABLI_COMPACT) <= set(CO.ARBRE_COMPACT) and all("compact=True" in s_ for s_ in _src.values())
    and "predalle" in CO.REDIRECTIONS and "predalle" not in CO.ETABLI_COMPACT, _src)

print("\n=== 7. Les mutations (en mémoire : le dépôt n'est pas touché) ===")


def mutant(fonction, avant, apres):
    """Une copie de ``fonction`` dont la source a ``avant`` (présent une fois) remplacé par ``apres``."""
    src = textwrap.dedent(inspect.getsource(fonction))
    assert src.count(avant) == 1, (fonction.__name__, avant, src.count(avant))
    espace = dict(vars(E))
    exec(compile(src.replace(avant, apres), f"<mutant {fonction.__name__}>", "exec"), espace)
    return espace[fonction.__name__]


@contextlib.contextmanager
def remplace(nom, valeur):
    vrai = getattr(E, nom)
    setattr(E, nom, valeur)
    try:
        yield
    finally:
        setattr(E, nom, vrai)


def muter(titre, fonction, avant, apres, controle):
    """Le contrôle est vert, ROUGE sous la mutation, vert de nouveau une fois le module remis."""
    m = mutant(fonction, avant, apres)
    vert_avant = controle()[0]
    with remplace(fonction.__name__, m):
        try:
            pendant = controle()[0]
        except Exception as e:  # noqa: BLE001 — une mutation peut faire lever : le contrôle n'est pas vert
            pendant = False
            print("       (mutation : exception", type(e).__name__, ")")
    vert_apres = controle()[0]
    chk(f"mutation « {titre} » : le contrôle rougit, puis reverdit", vert_avant and not pendant and vert_apres, (vert_avant, pendant, vert_apres))


muter("Supprimer allumé quand l'élément est seul", E.arbre_compact, '{" disabled" if seul else ""}', '{""}', ctl_un_supprimer)
muter("la liste sans data-activer (on ne peut plus ouvrir un autre élément)", E.arbre_compact, 'data-activer="{i}" ', "", ctl_deux)
muter("le nom de l'ouvert non échappé (le champ, la confirmation)", E.arbre_compact, "nom = C.esc(nom_txt)", "nom = nom_txt", ctl_pieges)
muter("les noms de la liste non échappés", E.arbre_compact, '<span class="nom">{C.esc(n)}</span>', '<span class="nom">{n}</span>', ctl_pieges)
muter("menu() sans « bouton » ne rend plus « ⋯ »", E.menu, 'C.icon("ellipsis")', 'C.icon("chevron-down")', ctl_menu)

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:200])
sys.exit(1 if KO else 0)

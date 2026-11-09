# -*- coding: utf-8 -*-
"""Les briques partagées de l'établi compact (api_web/etabli_compact.py) : ce que la fusion des dalles (docs/site/
DALLES_FUSION.md § 1, lot FUS5) a ajouté SANS rien changer à la poutre.

- ``arbre(…, compact=False)`` et ``tete_resultats(nom, dims)`` rendent EXACTEMENT le HTML d'avant : un gel (l'empreinte
  du rendu de quatre arborescences et de deux têtes, relevée sur le code du 09/11/2026 avant la fusion — commit
  bdb4efa) — la poutre ne bouge pas tant que l'ingénieur n'a pas validé ;
- ``arbre(…, compact=True)`` : ni en-tête « LISTE (n) », ni ligne d'élément ; un élément → les rubriques en tête et le
  menu « ⋯ » (Ajouter, Renommer, Copier ; Supprimer éteint) ; plusieurs → le sélecteur (.seg, data-activer, l'ouvert
  marqué), le compte, le menu de l'ouvert (Supprimer allumé) ; les attributs que js/etabli.js lit sont ceux d'avant ;
- ``tete_resultats(…, chemin_nom)`` : le nom en champ data-chemin="nom" data-sorte="texte" (le même mécanisme que la
  ligne d'arborescence), verrouillé → en lecture, un cadenas devant.

    python3 tests/test_etabli_compact.py
"""
import hashlib
import os
import re
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

from api_web import etabli_compact as E  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:300]}"))


# 1. le gel : la poutre (compact=False) et la tête sans chemin rendent le HTML d'avant la fusion, à l'octet
EXEMPLES = [
    ("POUTRE", [("V1", "· 25 × 50 cm", False)], 0, '<div class="sec">corps</div>', "poutre"),
    ("POUTRE", [("V1", "· 25 × 50 cm", False), ("V2 <b>", "· 30 × 60 cm", False), ("V3", "· 20 × 40 cm", False)], 1, "<p>x</p>", "poutre"),
    ("DALLE", [("Dalle 1", "· 100 × 20 cm", True), ("Dalle 2", "· 100 × 22 cm", False)], 0, "<p>y</p>", "dalle"),
    ("PREDALLE", [("Prédalle « 1 »", "· 100 × 22 cm", False)], 0, "", "predalle"),
]
GEL_AVANT = "bb495636a31fd4edc3d7bd5e1e9472307f3874a8162627bed9e8b626640c030b"
rendus = [E.arbre(getattr(E, g), els, a, corps, marque) for g, els, a, corps, marque in EXEMPLES]
rendus += [E.tete_resultats("V1", "25 × 50 cm · C30/37 · B500"), E.tete_resultats("Dalle &amp; 1", "x")]
empreinte = hashlib.sha256("\x1e".join(rendus).encode()).hexdigest()
chk("arbre(compact=False) et tete_resultats(nom, dims) rendent le HTML d'avant la fusion (gel du 09/11/2026, à l'octet)",
    empreinte == GEL_AVANT, empreinte)
chk("… l'arbre d'avant garde son en-tête « POUTRES (n) », sa ligne d'élément et « Ajouter une poutre »",
    all('class="et-arbre-h"' in r and 'class="et-pt-add" data-ajouter="1"' in r for r in rendus[:4])
    and 'class="et-pt-nom"' in rendus[0] and len(re.findall(r'<div class="et-pt(?: on)?"', rendus[0])) == 1)

# 2. l'arbre COMPACT, un seul élément
UN = [("Dalle 1", "· 100 × 20 cm", False)]
h1 = E.arbre(E.DALLE, UN, 0, '<div class="sec" data-sec="dl">corps</div>', "dalle", compact=True)
chk("compact, un élément : ni en-tête « DALLES (1) », ni ligne « Dalle 1 », ni champ « nom » à gauche, ni « Ajouter » en bas",
    "et-arbre-h" not in h1 and "et-pt-l" not in h1 and 'data-chemin="nom"' not in h1 and "et-pt-add" not in h1 and 'data-activer=' not in h1, h1)
chk("compact, un élément : les rubriques suivent directement le menu discret (.et-elsel.seul), dans .et-arbre-c.et-seul marqué data-dalle=\"0\"",
    re.match(r'<div class="et-arbre et-arbre-c et-seul" data-dalle="0"><div class="et-elsel seul"><div class="et-mn">.*?</div></div></div><div class="sec" data-sec="dl">corps</div></div>$',
             h1, re.S) is not None, h1[:200])
menu1 = re.findall(r'<button type="button" role="menuitem" class="([^"]*)" ([^>]*?)>', h1)
chk("compact, un élément : le menu ⋯ offre Ajouter (data-ajouter, l'action existante), Renommer, Copier (data-dupliquer-index), Supprimer ÉTEINT (jamais le dernier)",
    [(c, re.sub(r' data-confirmer="[^"]*"', "", a)) for c, a in menu1]
    == [("", 'data-ajouter="1"'), ("", 'data-renommer="0"'), ("", 'data-dupliquer-index="0"'), ("danger", 'data-supprimer="0" disabled')], menu1)

# 3. plusieurs éléments : le sélecteur
TROIS = [("Dalle 1", "· 100 × 20 cm", False), ("Dalle <2>", "· 100 × 22 cm", True), ("Dalle 3", "· 120 × 25 cm", False)]
h3 = E.arbre(E.DALLE, TROIS, 1, "<p>corps</p>", "dalle", compact=True)
segs = re.findall(r'<button type="button" class="([^"]*)" data-activer="(\d)" aria-pressed="(\w+)"', h3)
chk("compact, trois éléments : un sélecteur .seg d'un segment par élément (data-activer=\"i\", l'ouvert marqué on / aria-pressed)",
    'class="et-elsel"><div class="seg" role="group"' in h3 and segs == [("", "0", "false"), ("on", "1", "true"), ("", "2", "false")], segs)
chk("compact : le nom est échappé dans le segment, le verrou y met un cadenas ; le compte est écrit ; l'arbre porte le rang de l'ouvert",
    "<2>" not in h3 and h3.count("Dalle &lt;2&gt;") >= 2 and h3.count('<span class="nom">') == 3
    and '<span class="n">3</span>' in h3 and 'data-dalle="1"' in h3
    and re.search(r'data-activer="1"[^>]*><svg[^>]*>.*?<rect.*?</svg><span class="nom">Dalle &lt;2&gt;</span>', h3, re.S)
    and not re.search(r'data-activer="0"[^>]*><svg', h3), h3[:300])
menu3 = re.findall(r'<button type="button" role="menuitem" class="([^"]*)" ([^>]*?)>', h3)
chk("compact, trois éléments : le menu ⋯ agit sur l'OUVERT (copier 1, supprimer 1 avec confirmation, Renommer éteint : verrouillé) ; Supprimer allumé",
    [(c, a) for c, a in menu3] == [("", 'data-ajouter="1"'), ("", 'data-renommer="1" disabled'), ("", 'data-dupliquer-index="1"'),
                                   ("danger", 'data-supprimer="1" data-confirmer="Supprimer la dalle « Dalle &lt;2&gt; » de la note ?"')], menu3)
chk("compact : « corps » (les rubriques) vient après le sélecteur, une seule fois", h3.count("<p>corps</p>") == 1 and h3.index("et-elsel") < h3.index("<p>corps</p>"))
chk("compact : un rang hors liste est ramené dans la liste (jamais d'exception)", 'data-dalle="2"' in E.arbre(E.DALLE, TROIS, 9, "", "dalle", compact=True))
# les attributs que js/etabli.js lit (agirElement, renommer) sont ceux de l'arbre d'avant : les mêmes noms, pas un de plus
ATTRS = {"data-activer", "data-ajouter", "data-dupliquer-index", "data-supprimer", "data-confirmer", "data-renommer", "data-menu"}
lus = {a for a in re.findall(r'\s(data-[a-z-]+)=', E.arbre(E.DALLE, TROIS, 1, "", "dalle", compact=True) + E.arbre(E.DALLE, UN, 0, "", "dalle", compact=True))
       if a != "data-dalle"}
chk("compact : les attributs data-* lus par js/etabli.js sont ceux de l'arbre d'avant (activer, ajouter, dupliquer-index, supprimer, confirmer, renommer, menu)",
    lus == ATTRS, lus ^ ATTRS)

# 4. la tête des résultats : le nom en champ
t0 = E.tete_resultats("Dalle &amp; 1", "100 × 20 cm · C30/37 · B500")
t1 = E.tete_resultats("Dalle &amp; 1", "100 × 20 cm · C30/37 · B500", "nom")
t1b = E.tete_resultats("Dalle 1", "d", ["nom"])
t2 = E.tete_resultats("Dalle 1", "100 × 20 cm", "nom", verrou=True)
chk("tete_resultats sans chemin : le HTML d'avant (<b>nom</b>, <span>dims</span>)", t0 == '<div class="et-rp"><b>Dalle &amp; 1</b><span>100 × 20 cm · C30/37 · B500</span></div>', t0)
chk("tete_resultats(chemin « nom ») : un champ data-chemin=\"nom\" data-sorte=\"texte\" (le mécanisme de la ligne d'arborescence), la valeur échappée telle quelle, les dimensions à côté",
    t1.startswith('<div class="et-rp et-rp-c"><input data-chemin="nom" data-sorte="texte" value="Dalle &amp; 1" class="et-rp-nom" size="10"')
    and t1.endswith('<span>100 × 20 cm · C30/37 · B500</span></div>') and "<b>" not in t1, t1)
chk("tete_resultats : le chemin s'écrit aussi en séquence (comme les autres champs)", 'data-chemin="nom"' in t1b)
chk("tete_resultats verrouillé : le nom se lit (pas de champ), un cadenas devant, dans la même tête", "<input" not in t2 and t2.startswith('<div class="et-rp et-rp-c"><b><svg')
    and t2.endswith('Dalle 1</b><span>100 × 20 cm</span></div>'), t2)

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:200])
sys.exit(1 if KO else 0)

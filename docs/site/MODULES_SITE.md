# Ajouter un module au site « Stab Méthodes »

Ce guide est le contrat de tout module du site (`site/web`). Deux modules le tiennent déjà et servent de modèles :
**garde-corps** (`api_web/garde_corps.py`, moteur `acier/garde_corps`) et **poutres béton armé** (`api_web/poutre.py`,
moteur `beton/poutre`, extrait de l'écran Streamlit `modules/poutre.py` avec ses tests de parité
`tests/test_poutre_moteur.py` et sa doc `docs/beton/POUTRE_MOTEUR.md`). Les lire avant d'écrire.

## 1. Le principe

Le site est statique. Le moteur Python du dépôt tourne **dans le navigateur** (Pyodide 0.29.5, Python 3.13). L'écran
(`site/web/js/etabli.js`) ne calcule rien : il envoie des requêtes JSON au routeur (`api_web/routeur.py`), qui appelle
le module, et pose les **fragments HTML** que le module rend. Les chiffres sont ceux de Streamlit parce que c'est le
même calcul.

Bibliothèques permises sur le chemin du calcul : **la bibliothèque standard et numpy**. Refusées (absentes du
navigateur) : streamlit, pandas, matplotlib, scipy, reportlab, PIL, fitz. Le relevé du paquet
(`site/web/construire.py`) les refuse et échoue si un chemin les charge.

## 2. Les fichiers d'un module — et SEULEMENT ceux-là

| Fichier | Rôle |
|---|---|
| `api_web/<clé>.py` | le module du site : déclare `MODULE = "<clé>"` et `PAGE = "<adresse>"` (il entre alors au registre, `api_web/registre.py`, sans rien modifier d'autre) |
| le moteur | s'il existe déjà sans Streamlit (`acier/soudures`, `acier/assemblages/...`, `plancher/charges`, `acier/profiles`) : l'importer, **sans** ses fichiers d'écran (`interface.py`, `ecran_*.py`, `note.py` importent streamlit/reportlab). S'il vit dans l'écran Streamlit (`modules/<x>.py`) : l'**extraire** dans un paquet de calcul (`beton/<x>/`, `bois/<x>/`, `sol/`, `flambement/`, `acier/<x>/`), comme `beton/poutre` |
| `api_web/dessin_<clé>.py` (ou dans le paquet du moteur) | les dessins SVG, à l'échelle |
| `tests/test_<x>_moteur.py` | la **parité** du moteur extrait avec l'écran Streamlit (faux `st`, comme `tests/test_poutre_moteur.py`), des contrôles à la main, des mutations qui rougissent |
| `tests/test_api_web_<clé>.py` | ce que l'écran du site affiche = ce que le moteur calcule ; saisies, refus, effets de bord, sélecteurs, échappement |
| `docs/<domaine>/<X>_MOTEUR.md` | la méthode, d'où vient chaque grandeur, la preuve de parité, ce qui n'est pas couvert, les défauts trouvés dans le module Streamlit |
| `site/web/css/modules/<clé>.css` (facultatif) | une feuille de style propre au module (chargée par son établi seulement) |

**Ne pas modifier** les fichiers partagés : `api_web/commun.py`, `routeur.py`, `registre.py`, `parcours.py`,
`icones.py`, `site/web/js/*`, `site/web/css/site.css`, `charte.css`, `construire.py`, `verifier_site.js`,
`tests/test_api_web_modules.py`, `tests/test_site_web.py`, ni `modules/*.py` (l'écran Streamlit reste le secours,
inchangé). Un besoin commun se signale dans le rapport final (« Besoin commun »), il ne se bricole pas.

Ne pas lancer `python3 site/web/construire.py` **sans** `--dossier` (site/web est reconstruit à l'intégration), ne pas
faire de commit ni de push : l'intégrateur s'en charge.

## 3. L'interface d'un module (`api_web/<clé>.py`)

```python
MODULE = "<clé>"          # la clé des familles de l'accueil (site/maquettes/construire.py, FAMILLES)
PAGE = "<adresse>"        # « charges-plancher » → établi charges-plancher.html (servi sans « .html »)
PARCOURS = [...]          # facultatif : requêtes de plus pour le parcours (api_web/parcours.py)

def nouvel_etat() -> dict                         # la note neuve (les défauts de l'écran Streamlit)
def normaliser(etat) -> dict                      # relit un état reçu (fichier .stab, navigateur) ; ValueError claire si abîmé
def modifier(etat, chemin, valeur) -> str | None  # une saisie ; chemin = data-chemin coupé aux « / » ; message si refusée (l'état reste intact)
def gerer_elements(etat, action, index)           # « activer », « ajouter », « dupliquer », « supprimer » (jamais le dernier)
def choisir_section(etat, role, nom)              # la ligne choisie dans un sélecteur
def selecteur(etat, role, familles, froid) -> str # le HTML d'un sélecteur (C.selecteur)
def resultat(etat) -> dict                        # voir ci-dessous
```

`resultat(etat)` rend : `resultat` (HTML du cadre Résultat), `volets` (fenêtres de calcul `C.fenetre_calcul`),
`vues` (`[{cle, titre, court, svg, legende}]` — titre, court et légende sont du TEXTE, la page les échappe ; `svg` est
du HTML posé dans la toile : un dessin `<svg …>`, ou un bloc `<div …>` quand la vue est un tableau — catalogue,
tableau des armatures), `phrase` (HTML, sous le titre de la première vue), `statut_html` (`C.statut(niveau)`), `resume`
(`{taux, niveau, statut, gouvernant}` — `taux` arrondi par `C.pct` ou None ; `niveau` parmi `p f r c x i` ; `i` =
« Consultation » pour un module sans vérification : tableau, catalogue, courbe), `donnees` (HTML du cadre Données),
`elements` (HTML de la liste des éléments de la note, comme garde-corps et poutre).

Une note contient **plusieurs éléments** (plusieurs dalles, plusieurs cornières…) : copier le schéma de
`api_web/poutre.py` (`{"version": 1, "<éléments>": [...], "actif": 0}`, `gerer_elements`, `elements`, résumé de la
note = son élément le plus chargé).

## 4. Les briques d'écran (`api_web/commun.py`) — toujours elles, jamais du HTML refait à la main

- **Cadre Données** : `C.section(n, titre, résumé, corps, ouvert, clé)` (sections numérotées, repliables) ;
  `C.ligne(libellé, champ, symbole, aide, full)` ; champs `C.num(chemin, valeur, unité, dec, entier, desactive, mini,
  maxi)`, `C.texte`, `C.choix` (menu), `C.segments` (2 à 4 options), `C.case`, `C.zone_texte` (plusieurs lignes : un
  tableau collé), `C.fichier` (le moteur reçoit `{"nom", "texte"}`), `C.aide`, `C.message`. Ajouter/retirer : boutons
  `class="addrow"` / `class="moins"` avec `data-chemin=".../+"` ou `".../<i>/-"` (voir `api_web/poutre.py`).
- **Résultat** : un **verdict** (`C.chip(taux, niveau, xl=True)` + la vérification gouvernante et sa comparaison
  « valeur ≤ limite »), puis une ligne par vérification `C.ligne_res(nom, pièce, taux, niveau, calcul=, pk=, gov=)`
  (deux lignes : nom et pièce, puis jauge et taux), groupées si utile (`<div class="elgrp">`), les avertissements
  (`C.message`), le pied (`pfoot` : méthode + `C.legende()`).
- **Fenêtre de calcul** : `C.fenetre_calcul(id, titre, sous_titre, hypothèses, lignes, détail)` — hypothèses en
  pastilles, tableau vérification/formule/valeur/limite/taux, puis le **détail des formules** `C.lignes_fx` :
  « symbole = expression = valeurs substituées = résultat », comme une note faite à la main.
- **Sélecteur** : `C.selecteur(...)` — trié par ordre de grandeur, le retenu (✓) et l'optimal (★, le plus léger qui
  tient) marqués.
- **Couleur** : `C.niveau(taux)` (rouge > 100 %, vert foncé 95–100 %, vert pâle < 95 % — les seuils du moteur,
  `ME.statut`) ; rouge aussi quand le moteur déclare non conforme sous 100 %. `C.pct` arrondit **sans changer de
  bande**.
- **Nombres** : `C.nb` (exact, virgule, espace fine des milliers) pour les données ; `C.fr(x, nd)` pour les résultats
  (mm² entiers, cm au mm, contraintes au centième…). Unités toujours écrites ; indices en `<sub>`.
- **Icônes** : `C.icon(nom)` (Lucide, `api_web/icones.py`).

## 5. Les règles

1. **Parité fonctionnelle avec Streamlit** : chaque champ, option, résultat, avertissement et dessin de l'écran
   Streamlit existe sur le site (l'organisation peut et doit être meilleure). La note PDF est hors périmètre (le bouton
   est grisé). Ce qui manque est écrit dans le rapport et dans la doc du moteur.
2. **Le calcul est celui de Streamlit** : on extrait, on ne réécrit pas. Les défauts trouvés dans le module Streamlit
   sont **reproduits** (parité) et **écrits** (doc, rapport), surtout s'ils sont non conservatifs — on ne les corrige
   pas en silence. Exception : une saisie que Streamlit ramène en silence à une valeur fausse (négatif → 0…) est
   **refusée** sur le site avec un message.
3. **Rien de neuf sans le dire** : un module qui n'a pas de calcul dans Streamlit (flambement) porte un avertissement
   visible « nouveau calcul, à valider par le bureau », et ses tests reproduisent des exemples publiés.
4. **Sécurité** : tout texte venu de l'état est échappé (`C.esc`) dans le HTML ; aucun `<script>`, aucun `on…=`,
   aucune ressource externe (politique de sécurité du site).
5. **Beau et lisible** : la charte des maquettes (`site/maquettes`) ; des dessins de dessinateur (traits, cotes,
   hachures, échelle) comme `acier/garde_corps/schemas.py` et `api_web/dessin_poutre.py` ; les nombres alignés ; rien
   de coupé ; le français du bureau (pas d'anglicisme, pas de jargon de développeur).
6. **Noms en français** (comme le dépôt), commentaires et docstrings en français.

## 6. Se vérifier avant de rendre

```bash
python3.12 tests/test_api_web_modules.py <clé>   # le CONTRAT commun : réponse complète, champs reliés, parcours sans erreur…
python3.12 tests/test_api_web_<clé>.py           # les tests propres
python3.12 tests/test_<x>_moteur.py              # la parité du moteur
# dans le vrai navigateur, sur une copie à soi (d'autres modules se construisent en même temps) :
python3.12 site/web/construire.py --dossier /tmp/site_<clé> --modules <clé>
python3.12 site/web/outils/serveur_local.py <port libre, 8800-8899> --dossier /tmp/site_<clé> &
node site/web/verifier_site.js http://127.0.0.1:<port> /tmp/captures_<clé> module:<clé>
```

Puis **regarder les captures** (`/tmp/captures_<clé>/site_<clé>*.png`) : le dessin est-il lisible et beau, le
résultat clair, rien de coupé ? Corriger tant que ce n'est pas le cas.

## 7. Le rapport final

Fichiers créés ; ce qui est couvert et ce qui ne l'est pas (par rapport à Streamlit) ; défauts trouvés dans le module
Streamlit ; résultats des tests (nombres OK/échecs) et du contrôle navigateur ; besoins communs éventuels.

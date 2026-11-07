# Catalogue des profilés — le moteur sans Streamlit et le module du site

> **Défauts NON CONSERVATIFS de l'écran Streamlit (reproduits par parité, écrits à l'écran du site)**
>
> 1. **I_t d'un profilé personnalisé PRS = Σ b·t³/3 (parois minces)** : il surestime la raideur de torsion des plats
>    épais — mesuré contre Saint-Venant (différences finies, même contour, grille convergée) : **+2 %** pour un
>    PRS 400x200x8x12, **+10 %** pour un 200x150x15x30. Un I_t trop grand donne un moment critique de déversement trop
>    grand. Le catalogue laminé, lui, est calculé par Saint-Venant. (`CO.reserve_torsion`, D11.)
> 2. **A_v,z d'un PRS = formule des laminés, t_w·(h − t_f)** : l'EN 1993-1-1 §6.2.6(3)(d) donne η·h_w·t_w pour un
>    profilé soudé ; avec **η = 1,0**, la valeur affichée la dépasse de t_f/h_w (**+18 %** pour un PRS
>    300x300x20x40 : 52,0 cm² contre 44,0). Avec η = 1,2 elle reste en dessous tant que t_f ≤ h/7. À trancher par
>    le bureau selon le η retenu. (`CO.reserve_cisaillement`, D12.)
> 3. **Deux filtres sur la même propriété : seul le dernier compte, sans le dire** (« I_y ≥ 10 000 » puis
>    « I_y ≤ 5 000 » ne garde que la seconde borne). Le site refuse ce cas. (D1.)
>
> Les deux premiers sont dans le moteur de sections partagé (`acier/profiles/familles.py`, choix de l'écran pour les
> profilés « non standard ») : ils ne sont pas corrigés ici (parité), ils sont écrits au résultat du site pour tout
> PRS, et mesurés par `tests/test_catalogue_moteur.py`.

*Livré le 07/11/2026. Moteur `acier/profiles/consultation.py` : bibliothèque standard + le paquet `acier/profiles`
(numpy pour le seul calcul d'un profilé personnalisé), Python 3.12 et 3.13 (donc Pyodide 0.29.5). Extrait de l'écran
Streamlit `modules/tableau_profiles.py` v1.0, qui reste la **référence** ; la preuve de parité est
`tests/test_catalogue_moteur.py`. Le catalogue lui-même (sources, calcul des propriétés, contrôle croisé) est décrit
dans `docs/acier/PROFILES.md`.*

## 1 · Ce qui est livré

| Pièce | Rôle |
|---|---|
| `acier/profiles/consultation.py` | ce que la page fait du catalogue, extrait mot pour mot de l'écran : réglages d'ouverture et défauts livrés, familles et colonnes remises dans leur ordre, liste affichée (familles, recherche, bornes, séries courantes, profilés personnalisés, tri), compteur, lignes du tableau, fiche, comparaison, profilé personnalisé (contrôles, nom, calcul), texte de l'export CSV, fiche JSON, « Méthode et sources » |
| `api_web/catalogue.py` | le module du site (`MODULE = "catalogue"`, `PAGE = "catalogue-profiles"`) : une note de **plusieurs recherches**, rendue en fragments HTML |
| `api_web/dessin_catalogue.py` | les dessins à l'échelle : la section cotée, la vignette, la comparaison des sections à la même échelle |
| `site/web/css/modules/catalogue.css` | le tableau (en-têtes et noms figés, lignes alternées, colonne triée), la comparaison, la fiche, le cadre Données |
| `tests/test_catalogue_moteur.py` | la parité avec l'écran réel (AppTest), le bout à bout site ↔ écran, les mutations, les mesures des réserves, la pureté — ≈ 7 min |
| `tests/test_api_web_catalogue.py` | ce que le site affiche = ce que le moteur calcule ; saisies, refus, sélecteurs, formules, dessins, échappement, notes abîmées — 3 s |

Le calcul des propriétés n'est PAS ici : il est dans `acier/profiles/familles.py` (`calculer`), le même pour le
catalogue livré (`acier/donnees/profiles.json`) et pour le profilé personnalisé. `consultation.py` ne fait que
choisir, filtrer, trier, arrondir et écrire, exactement comme l'écran.

## 2 · L'API du moteur

```python
from acier.profiles import consultation as CO

CO.reglages_initiaux() -> dict          # préférences enregistrées + défauts (familles, colonnes, tri, decroissant, courants, filtres)
CO.reglages_livres() -> dict            # « ↩ Rétablir » (sans écrire le fichier : voir défaut D5)
CO.familles_ordonnees(choix) / CO.colonnes_ordonnees(choix)   # l'ordre des pastilles de l'écran
CO.liste(familles, recherche, filtres, courants, persos, tri, decroissant) -> [profil]
CO.n_persos_affiches(recherche, filtres, persos) ; CO.compteur(n, familles, n_persos) -> "**900 profilés** · 11 familles"
CO.colonnes_visibles(liste, colonnes) ; CO.lignes_tableau(liste, colonnes) -> [{"Profilé", "Famille", "Iy (cm⁴)": 80.1, …}]
CO.texte_tableau(lignes, ";") -> str    # le CSV de l'écran, sans BOM ; "\t" pour coller dans Excel
CO.fiche_retenue(fiche, noms) ; CO.compares_retenus(compare, noms)
CO.fiche(p) -> [(groupe, [(clé, libellé, valeur écrite, unité, définition)])] ; CO.chips_fiche(p) ; CO.methode_it(p)
CO.comparaison(profils, colonnes) ; CO.colonnes_comparaison(colonnes_visibles)
CO.formes_perso() ; CO.dims_defaut(forme) ; CO.nom_perso(forme, dims) ; CO.controle_ecran(forme, dims)
CO.calculer_perso(forme, dims) -> profil ; CO.ajouter_perso(persos, p)
CO.refus_dimension(k, x) ; CO.refus_du_site(forme, dims)   # ce que le SITE refuse en plus (non fini, < 0,1 mm, > 10 m)
CO.avertissements_perso(forme, dims)    # ce que le SITE écrit en plus (PRS t_w ≥ b)
CO.reserve_torsion(p) ; CO.reserve_cisaillement(p)       # les réserves NON CONSERVATIVES d'un PRS, écrites au résultat
CO.fiche_json(p) ; CO.nom_fichier_fiche(p) ; CO.legende_page() ; CO.methode_et_sources()
```

Un **filtre** est `{"cle": "Iy", "min": 1500.0, "max": None}` (trois filtres, comme l'écran) ; un **profilé** est
l'enregistrement à plat du catalogue (`CAT.profils()`), ou celui d'un profilé personnalisé (`perso: True`).

## 3 · D'où vient chaque chose de l'écran

| L'écran Streamlit (`modules/tableau_profiles.py`) | Le moteur (`consultation.py`) | Le site (`api_web/catalogue.py`) |
|---|---|---|
| `_init` (préférences), « ↩ Rétablir » | `reglages_initiaux`, `reglages_livres` | recherche neuve ; bouton « Rétablir les réglages livrés » (section 4) |
| `_lire_pills`, `[c for c in cles_num if c in choix]` | `familles_ordonnees`, `colonnes_ordonnees` | cases des sections 2 et 4 |
| `bornes = {f["cle"]: (min, max) …}` | `bornes` | section 3 (trois filtres) |
| `base + persos`, `CAT.trier` | `liste`, `n_persos_affiches` | vue « Tableau » |
| `st.markdown("**n profilés** · …")` | `compteur` | phrase de la vue, légende |
| `_lignes_tableau` (arrondi par colonne) | `lignes_tableau` | cellules du tableau (`C.fr`, décimales fixes) |
| `_csv` | `texte_tableau` | section 7, texte à copier |
| `if fiche not in noms: noms[0]` | `fiche_retenue` | ligne « Profilé » du résultat, sélecteur |
| `st.multiselect(..., max_selections=5)` | `compares_retenus`, `MAX_COMPARES` | ligne « Comparer », section 5, vue « Comparaison » |
| `_table_fiche`, puces, « It : … » | `fiche`, `chips_fiche`, `methode_it` | cadre Résultat (en-tête, toutes les propriétés) |
| `_table_comparaison` | `comparaison`, `colonnes_comparaison` | tableau de la vue « Comparaison » |
| `_perso`, `_nom_perso` | `controle_ecran`, `nom_perso`, `calculer_perso` | section 6 |
| « ⬇️ Fiche (JSON) » | `fiche_json`, `nom_fichier_fiche` | section 7, texte à copier |
| « Méthode et sources » | `methode_et_sources` | fenêtre « Méthode et sources » |
| `DS.svg_section` | — (contour : `familles.contours`) | vue « Section » (`dessin_catalogue.section`) |

## 4 · La preuve de parité (`tests/test_catalogue_moteur.py`)

L'écran **réel** est exécuté par Streamlit (AppTest, sans navigateur) : 36 états tirés au hasard (graine 20261107) et
des cas ciblés, posés geste par geste (pastilles, recherche, trois filtres, séries courantes, tri, colonnes,
personnalisés, choix de la fiche, comparaison, autre recherche, boutons « Toutes », « Aucune », « Effacer les
filtres », « Garder par défaut », « Rétablir »). Après CHAQUE exécution (236), ce que l'écran montre est comparé à ce que
le module calcule depuis les mêmes réglages :

- la liste (noms, ordre), les colonnes et **chaque cellule** — 1 893 010 cellules, comparées par empreinte SHA-256 du
  JSON (les flottants en `repr`, donc au bit près) ;
- le compteur, le message « Aucun profilé », la légende du titre, l'aide des colonnes, le titre « n sur 39 » ;
- l'état qui suit les clics : ordre des familles et des colonnes, fiche qui passe au premier (et ne revient pas),
  comparaison qui perd les profilés sortis ;
- la fiche : l'enregistrement dessiné, l'en-tête et la table de toutes les propriétés (le HTML de l'écran recomposé
  depuis le module, au caractère près), la légende I_t ; une fiche de chaque famille du catalogue et de chaque forme ;
- la comparaison (HTML au caractère près) et son aide ;
- les deux fichiers téléchargés : le CSV octet pour octet (BOM compris) et la fiche JSON avec son nom ;
- les profilés personnalisés : chaque grandeur au bit près, nom, place, messages d'erreur — aux bornes exactes
  (2·t_f = h, 2·t = min(h, b), 2·t = d refusés ; un demi-millimètre de moins, calculés) ;
- « Méthode et sources », phrase par phrase.

Puis **22 mutations** du module (le premier filtre gagne, persos filtrés par familles, tri avant persos, compteur
sans pluriel, arrondi retiré, colonnes vides gardées, fiche qui revient, comparaison qui garde, aide perdue, pastille
« personnalisé » perdue, CSV « corrigé », JSON indenté à 2, nom à une décimale, « > » au lieu de « ≥ », ordre des
clics, Rétablir qui garde « décroissant »…) : **chacune rougit**. Enfin, la **pureté** : le module et le site se
chargent et calculent dans un sous-processus où `sys.meta_path` refuse streamlit, pandas, matplotlib, scipy, reportlab,
PIL et fitz.

**Bout à bout (relecture du 07/11/2026)** : la chaîne écran → moteur → HTML était prouvée en deux maillons ; elle l'est
aussi d'un seul tenant. Dix-huit états posés geste par geste des DEUX côtés (l'écran par AppTest, le site par les
requêtes du navigateur à `api_web/routeur.traiter`) — ouverture, recherche et tri décroissant, cornières filtrées avec
leurs colonnes, CHS au plus léger, séries courantes, min > max, aucune famille + PRS, caisson, tube à décimale, plat au
minimum exact (0,1 mm), PRS t_w > b, comparaisons de 3 et de 5 profilés mêlés, borne pile égale à la valeur exacte,
HD (noms à point) sur 39 colonnes, U triés par y_s, formés à froid sous trois filtres — et ce que le SITE affiche (le
texte de chaque cellule du tableau relu, la fiche, la comparaison, le compteur, le texte d'export comparé au CSV et au
JSON téléchargés, l'ordre du sélecteur) est comparé à ce que l'écran MONTRE, sans passer par `consultation.py` :
**42 416 valeurs, 0 écart**. Trois mutations de la couche du site (une décimale perdue, une ligne de fiche perdue, une
colonne vide affichée) le font rougir. Les deux réserves non conservatives (I_t et A_v,z des PRS) y sont mesurées.

Résultat au 07/11/2026 (relecture) : **66 OK, 0 échec** (236 exécutions de l'écran, 1 893 010 cellules ; bout à bout
42 416 valeurs ; ≈ 7 min). Module du site : `tests/test_api_web_catalogue.py` **127 OK** ; contrat commun **17 OK** ;
contrôle navigateur (`verifier_site.js module:catalogue`) **23 OK**, aucune erreur JavaScript.

Essai rapide : `CAT_N_CAS=4 python3.12 tests/test_catalogue_moteur.py` (les contrôles de couverture et quatre mutations
demandent les 36 cas : ils rougissent en essai rapide).

## 5 · Le module du site

- **Plusieurs recherches dans une note** (« Poutres R+1 », « Poteaux ») : chacune garde ses familles, sa recherche,
  ses filtres, son tri, ses colonnes, ses profilés personnalisés, sa comparaison et sa fiche ; « Copier » reprend les
  réglages de la recherche ouverte. Résumé de la note : niveau « i » (Consultation), sans taux.
- **Données**, sept sections numérotées : 1 Recherche et tri · 2 Familles (cases par groupe, avec leur nombre ; séries
  courantes ; toutes, aucune) · 3 Filtres numériques (trois, sur deux lignes ; chaque borne est FACULTATIVE : effacer
  le nombre retire la borne, comme le champ de l'écran qu'on vide ; ★ le plus léger qui satisfait les filtres et le
  bouton qui l'ouvre) · 4 Colonnes du tableau (par groupe, symboles en indices et unités ; nombre de lignes ;
  rétablir) · 5 Comparaison (l'aide « Choisir au moins deux profilés du tableau » de l'écran sous deux) · 6 Profilé
  personnalisé, ouvert d'emblée (PRS, caisson, tube, plat, rond ; dimensions de 0,1 mm à 10 m ; avertissement PRS
  t_w ≥ b) · 7 Export (texte à copier).
- **Toile** : la vue « Tableau » (en-têtes avec symboles et unités, noms et en-têtes figés au défilement, lignes
  alternées, colonne triée en tête et surlignée, profilé ouvert surligné et toujours visible même au-delà des lignes
  affichées, comparés numérotés, personnalisés marqués) ; la vue « Section » (contour exact à l'échelle, hachures,
  axes y-y et z-z par G, cotes h, b ou Ø, t_w, t_f, t, r, centre de cisaillement S d'un U, axes principaux u, v d'une
  cornière) ; la vue « Comparaison » dès deux profilés (sections à la même échelle, propriétés côte à côte).
- **Résultat** : l'en-tête (vignette, nom, famille, masse, aire, pastilles), la ligne « Profilé » (sélecteur trié dans
  l'ordre du tableau, familles en pastilles, retenu ✓, plus léger ★ ; elle ouvre la fenêtre de calcul), « Comparer »
  (sélecteur : un clic ajoute, un clic sur un profilé coché ✓ le retire), « Méthode et sources », la méthode de I_t,
  les réserves d'un PRS (I_t, A_v,z), l'étoile, toutes les propriétés groupe par groupe (la propriété filtrée
  marquée). Un dessin qui échouerait laisse la fiche lisible (« Dessin indisponible », comme l'écran).
- **Fenêtre de calcul** du profilé : les formules du moteur de sections, écrites « symbole = formule = valeurs =
  résultat » (aire, masse, élancements c/t, aires de cisaillement, modules, rayons de giration, axes principaux d'une
  cornière, torsion, gauchissement). Les valeurs substituées portent quatre chiffres significatifs au moins et les
  négatifs sont entre parenthèses : `tests/test_api_web_catalogue.py` réévalue les **13 361** formules écrites (1 419
  profilés + 5 personnalisés) ; chacune redonne la valeur du catalogue à 0,12 % près au pire (arrondi des valeurs
  substituées). Les formules de W_pl, de I_t laminé et de I_w d'un U (intégrations du contour) sont nommées, non
  développées.
- Ce que le site présente autrement que l'écran, sans rien changer aux nombres : la colonne de tri vient juste après
  le nom (et s'affiche même si elle n'est pas cochée) ; « Famille » est masquée sous 1 360 px (le nom la porte) ; au
  plus 100, 200 (défaut), 500 lignes ou toutes ; le texte d'export est en tabulations (les mêmes cellules que le CSV).

## 6 · Inventaire de l'écran Streamlit, coché

| Élément de l'écran | Site |
|---|---|
| titre, légende (nombre de profilés, version) | ✓ pied du résultat, fenêtre « Méthode et sources » |
| 🏠 Accueil | ✓ (rail du site) |
| 💾 Garder par défaut | ✗ pas de fichier partagé dans le navigateur ; chaque note garde ses réglages, « Copier » les reprend — besoin commun |
| ↩ Rétablir | ✓ (sans écrire de fichier) |
| Séries courantes seulement (+ aide) | ✓ section 2 |
| Toutes les familles, Aucune | ✓ section 2 |
| pastilles des familles par groupe, avec leur nombre | ✓ cases par groupe, groupes nommés en clair |
| Rechercher, Trier par, Décroissant | ✓ section 1 |
| trois filtres numériques (propriété, ≥ min, ≤ max), Effacer les filtres | ✓ section 3 (une propriété filtrée deux fois est refusée : D1) |
| Colonnes affichées — n sur 39 (pastilles par groupe, aide) | ✓ section 4 |
| compteur « n profilés · n familles · n personnalisés » | ✓ phrase et légende de la vue |
| ⬇️ CSV | ◐ texte à copier (mêmes cellules) ; le téléchargement attend une brique commune |
| tableau (Profilé figé, Famille, nombres localisés, aide au survol des en-têtes) | ✓ vue « Tableau » |
| clic sur une ligne → fiche | ◐ par le sélecteur « Profilé » (et le bouton ★) ; le clic dans la toile attend une brique commune |
| clic sur un en-tête → tri de l'affichage ; ⋮ recherche, plein écran | ◐ « Trier par » ; replier Données et Résultat |
| « Aucun profilé : élargir… » | ✓ avec la cause (aucune famille, recherche, min > max…) |
| Fiche du profilé (liste déroulante) | ✓ sélecteur (✓ retenu, ★ le plus léger) |
| en-tête de la fiche (PROFILÉ, nom, libellé, norme, personnalisé) | ✓ |
| dessin de la section (axes, cotes h, b, épaisseurs en légende) | ✓ plus complet (t_w, t_f, r cotés ; G, S ; u, v) |
| toutes les propriétés par groupe, définition au survol | ✓ |
| « It : méthode » | ✓ |
| ⬇️ Fiche (JSON) | ◐ texte à copier |
| Comparer des profilés (jusqu'à 5, tableau, « au moins deux ») | ✓ + sections à la même échelle |
| Profilé personnalisé (forme, description, dimensions ≥ 0,1 mm, bouton, erreurs, fiche qui suit) | ✓ |
| Méthode et sources | ✓ |

## 7 · Défauts constatés dans l'écran Streamlit (reproduits, écrits)

- **D1 · Deux filtres sur la même propriété : seul le dernier compte, sans le dire.** « I_y ≥ 10 000 » puis
  « I_y ≤ 5 000 » ne garde que la seconde borne (le dictionnaire des bornes écrase la première). Le moteur reproduit
  (`bornes`) ; le site **refuse** de filtrer deux fois la même propriété (message : régler ses deux bornes sur la même
  ligne). Non conservatif pour qui croit avoir posé les deux bornes.
- **D2 · L'export CSV réécrit les NOMS.** Le point devient virgule dans toutes les cellules, noms compris :
  « HD 260x54.1 » s'exporte « HD 260x54,1 », qui n'est plus le nom du catalogue (456 profilés concernés). Reproduit
  dans le texte d'export (parité octet pour octet).
- **D3 · PRS à âme au moins aussi épaisse que les semelles sont larges (t_w ≥ b) : calculé sans un mot.** L'aire et
  les inerties restent justes (le contour est l'union des trois plats), mais ce n'est plus un I : c_f/t_f devient
  négatif (PRS 400x200x300x12 : −4,17) et A_v,z (formule d'un I) est sans objet. Le site calcule pareil et l'écrit
  (avertissement avant le calcul, ligne rouge au résultat).
- **D4 · Le titre « Colonnes affichées — n sur 39 » a une exécution de retard** : il est écrit avant que l'écran relise
  ses pastilles. Le site n'a pas ce retard.
- **D5 · « Garder par défaut » et « Rétablir » écrivent le fichier des préférences pour TOUS les utilisateurs du
  serveur** ; « Rétablir » ne vide pas la recherche. Le site n'écrit rien (chaque note garde ses réglages) ; il
  reproduit la recherche gardée.
- **D6 · La fiche et la comparaison ne reviennent pas.** Un filtre qui chasse la fiche la fait passer au premier
  profilé ; le filtre retiré, elle n'y revient pas. Un profilé comparé qui sort du tableau quitte la comparaison pour
  de bon. Reproduit.
- **D7 · Les bornes portent sur la valeur exacte, le tableau montre l'arrondi** : « I_y ≤ 80,1 » écarte l'IPE 80 que
  le tableau affiche « 80,1 » (I_y = 80,14 cm⁴). Reproduit ; l'aide du site le dit.
- **D8 · Les profilés personnalisés ignorent les familles et les séries courantes** (« Aucune » famille les montre
  encore), mais pas la recherche ni les bornes. Reproduit.
- **D9 · Une dimension sous 0,1 mm** est remontée en silence au minimum par le champ de l'écran ; le site la
  **refuse** avec un message (règle des saisies ramenées en silence).
- **D10 · Une dimension démesurée fait tomber l'écran** : h = 1e300 mm lève `OverflowError` dans le calcul, que
  `_perso` n'attrape pas (il n'attrape que `ValueError`) ; la page affiche une trace. Le site refuse toute dimension
  au-delà de 10 m (`CO.DIM_MAX`) ou non finie, à la saisie comme dans un fichier .stab.
- **D11 · I_t d'un PRS par Σ b·t³/3 : NON CONSERVATIF** (voir l'encadré en tête) : +2 % à +10 % mesurés.
- **D12 · A_v,z d'un PRS par la formule des laminés : NON CONSERVATIF avec η = 1,0** (voir l'encadré en tête).

Défauts trouvés et corrigés dans le SITE à la relecture du 07/11/2026 : une note .stab portant `NaN` ou `Infinity`
(bornes, dimensions) faisait tomber la réponse (« AssertionError ») ou calculait un « PRS infx200x8x12 » d'aire NaN —
refusée maintenant avec un message ; « familles » ou « colonnes » qui ne sont pas des listes, une comparaison en
double — refusées ou dédoublonnées ; un chemin de saisie inconnu rendait le jargon Python (« list index out of
range », « invalid literal for int() ») — il rend « Saisie inconnue » ; la borne d'un filtre ne pouvait pas être
effacée seule (l'établi refuse un nombre vide) ; un clic sur un profilé déjà comparé ne faisait rien ; le titre de la
vue Section s'écrivait « L 100X100X10 » en capitales (« × » maintenant) ; les noms sous les sections de la
comparaison tombaient à 9 px dès quatre profilés ; l'avertissement PRS t_w ≥ b écrivait « c_f/t_f » sans indices.
Chacun est gardé par `tests/test_api_web_catalogue.py` (section 8).

Défaut trouvé et corrigé dans le SITE (pas dans Streamlit, qui n'a pas de fenêtre de calcul) : un I_yz négatif
s'écrivait « −42,2² » (lu −(42,2²)) et les petites cornières recevaient des valeurs substituées trop arrondies
(I_y = 0,4 cm⁴) ; la formule écrite de I_v d'une L 100x65x8 aurait donné 81,6 cm⁴ au lieu de 24,8 cm⁴. Le test des
13 361 formules le garde.

## 8 · Ce qui n'est pas couvert

- le téléchargement direct de `profiles.csv` et de la fiche `.json` (texte à copier à la place) ;
- le clic dans le tableau (une ligne → la fiche, un en-tête → le tri) ;
- « Garder par défaut » (préférences communes à toutes les notes) ;
- au téléphone, l'établi commun n'affiche que l'onglet Données (pas d'onglet pour passer au tableau ni au résultat) ;
- I_t exact (Saint-Venant) et A_v,z de profilé soudé pour un PRS : le moteur de sections partagé ne les calcule pas
  ainsi (réserves écrites, à trancher par le bureau).

Tous relèvent d'une brique commune du site (§ 9) ; le calcul, lui, est entier.

## 9 · Besoins communs (signalés, non bricolés)

1. **Téléchargement d'un fichier produit par le moteur** (CSV, JSON) : une brique `C.telecharger(nom, type, texte)` et
   son câblage dans `etabli.js`.
2. **Clic dans une vue de la toile** (ligne d'un tableau → `choisir`, en-tête → `modifier` du tri) : `etabli.js` ne
   câble que les cadres Données et Résultat.
3. **`C.ligne_res` sans taux** (ligne d'information, sans jauge) : le module retire la seconde ligne du HTML de la brique.
4. **`C.fenetre_calcul` sans tableau de vérification** (un catalogue ne vérifie rien) : le module retire le tableau vide.
5. **`C.selecteur` sans pastille de taux** (la colonne de droite libre) : le module retire la pastille.
6. **Un champ numérique facultatif** (`C.num(..., facultatif=True)`) : `etabli.js` refuse un nombre vide (« Valeur
   refusée »). En attendant, le module rend les bornes des filtres par `C.num` puis les envoie comme TEXTE
   (`data-sorte="texte"`, lu et contrôlé par le module) : une borne vidée tombe. Conséquence pour le contrôle commun
   (`verifier_site.js` saisit dans le premier champ `data-sorte="nombre"` VISIBLE) : la section 6 est ouverte d'emblée.
7. ~~L'échelle « mm · à l'échelle » sur un tableau~~ : réglé par l'établi commun (le contrôle « jamais écrit sur un
   tableau » passe) ; la règle de la feuille du module reste, sans effet.
8. **L'établi au téléphone** : pas d'onglets Données / Dessin / Résultat (`page_module` sans `.mtabs`).
9. **Préférences d'affichage d'un module** (réglages par défaut gardés dans le navigateur, hors note).
10. ~~Nom de note mal encodé dans `verifier_site.js`~~ : réglé (le contrôle lit les noms « accents intacts »).

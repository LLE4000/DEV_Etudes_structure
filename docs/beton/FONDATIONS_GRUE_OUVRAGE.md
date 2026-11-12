# Fondations de grues — le fichier « rbar.ouvrage » et les options de conception

*Écrit le 12/11/2026. Le site écrit, depuis la note de calcul, le fichier que RBAR dessine (plan de coffrage, plan
d'armatures, bordereau). Code : `beton/fondation_grue/ouvrage.py` (le fichier), `beton/fondation_grue/conception.py`
(les options) ; écran : `api_web/fondation_grue.py` ; preuves : `tests/test_fondation_grue_ouvrage.py`, et la mesure
`docs/site/mesures/2026-11-12_fichier_rbar_lu_par_rbar.md` (le fichier passé dans le vrai lecteur de RBAR).*

**Quand le propriétaire demande « qu'est-ce qu'on doit donner comme valeur ? » : la réponse est au § 5.**

## 1. Ce que c'est

- **Le format est celui de RBAR**, pas celui du site : « rbar.ouvrage », version 1 (dépôt RBAR,
  `docs/reference/ouvrages/regles/format-du-fichier.md`, ajouts 0.4.27 à 0.4.35), plus les quatre ajouts demandés
  par la conversation RBAR le 12/11/2026 : `feuilles.echelles`, `cartouche`, `revisions`, la charge calculée des
  pieux (sa clé n'existe pas encore : en attendant, un encadré « Pieux »).
- **RBAR ne sait pas ce qu'est une fondation de grue** (exigence n° 1 du brief, `FONDATIONS_GRUE_RBAR.md` § 3) :
  les noms, les titres, les textes sont dans le fichier ; RBAR dessine des poutres, des pieux, des tracés.
- Il remplace, pour RBAR, l'ancien fichier `stab.fondation_grue` (« Modèles de poutre pour RBAR ») : celui-ci reste
  téléchargeable, inchangé, pour l'import des modèles dans la palette de RBAR.

## 2. D'où vient chaque valeur

Rien n'est recalculé par l'export, rien n'est inventé. Ce qu'aucune source ne donne reste **absent** : RBAR le prend
dans le standard du poste, ou ne le dessine pas, et le dit.

| Clé du fichier | Source |
|---|---|
| `modelesPoutre` : lits `bas`, `haut`, `basSuivants`, `hautSuivants` | les armatures **choisies dans la note**, celles que le classeur vérifie (un second groupe = un lit suivant) |
| `modelesPoutre` : `etrier` (Ø, pas, brins) et ses `zones` | les étriers de la note (extrémités et partie centrale) ; la longueur des zones : option |
| `modelesPoutre` : `lat` (Ø) | la prescription du classeur (2Ø12) ; le nombre de rangées, les retours, le recouvrement : options |
| `modelesPoutre` : `cages` (Ø) | le renfort au droit des pieux de la note ; les cadres, l'emprise, la jambe : options |
| `modelesPoutre` : `scellements` (nombre, Ø) | les perfos de la note (3Ø25 par plaque à Greenwood) ; longueurs, remarque, motif de pose : options |
| retours des barres (`retour_mm`) | option (longueur fixe, ou arrêté sous la face opposée) |
| `enrobage`, `materiaux` | la note |
| `typesPieux` | le Ø et la capacité du pieu de la note (« Pieu Ø53 — 75 T (ELS) ») |
| `pieux` | la géométrie en plan du site (la même que l'écran et le DXF) ; le recépage : option |
| `poutres` | sections de la note, bouts de chaque poutre (géométrie du site, jusqu'à la face extérieure des semelles) ; désignations, coupes, niveau du dessus : options |
| `poutres[].origines` | le poste et les **cellules du classeur** de chaque armature (« F1 · Poutre support châssis · Armatures inférieures · classeur G129:G132 ») — RBAR les écrit en REMARQUES du bordereau |
| `encadres` | Matériaux (la note) ; Remarques de pose (option) ; **Pieux** : le pieu le plus chargé (service, sans coefficient) et la capacité ; **Grue** : ses données et la plus grande charge par pied |
| `traces` | le châssis de la grue (son carré et ses diagonales), au nom de la grue |
| `feuilles` | la table « mise en page par châssis » (§ 4) |
| `cartouche` | la note : le chantier (= le projet), son nom (= le titre), sa date |
| `revisions` | l'indice courant de la note et sa date |

## 3. Le bouton « Exporter »

Un seul bouton, en bas de la colonne des résultats, le même sur PC, tablette et téléphone (composant commun
`etabli_compact.exporter`). Son menu :

1. **Contrôle avant export** — ce qui part (chaque valeur, et d'où elle vient), ce qui est à regarder, ce qui manque,
   ce qui bloque ;
2. **DWG pour AutoCAD** — étape 1 : le fichier « rbar.ouvrage » ; dans AutoCAD, commande `RBAROUVRAGE`, puis choisir
   ce fichier. Éteinte quand l'envoi est bloqué, avec la raison ;
3. **Plan de principe (.dxf)**, **Armatures (.csv)**, **Cas de la grue (.csv)** — inchangés ;
4. **Modèles de poutre pour RBAR (.json)** — l'ancien fichier, inchangé.

**On garde toutes ces lignes pour l'instant** (décision du propriétaire du 12/11/2026, soir : la conversation RBAR
travaille en parallèle, rien d'utile ne doit disparaître). À terme : trois exports — DWG, PDF, Excel, ce dernier
remplissant un classeur type dont les cases à remplir sont colorées (`FONDATIONS_GRUE.md` § 9.2).

**« Plans PDF » n'apparaît pas encore** : il attend le moteur de plans de RBAR en WebAssembly (dans le navigateur, sans
réseau, chaque fichier sous 25 Mo — la limite de Cloudflare Pages). L'état « en cours… » viendra avec lui, et un
aperçu des deux feuilles au centre de l'écran (`FONDATIONS_GRUE.md` § 9.3) ; l'étape 2 du DWG (le DWG fait sans
AutoCAD), plus tard.

**Ce qui bloque l'envoi** (le fichier n'est jamais écrit, la ligne est éteinte, la raison est dite) : une vérification
du classeur PAS OK ; des étriers de Ø différents aux extrémités et au centre (un modèle de RBAR n'a qu'un Ø
d'étrier) ; des zones d'étriers serrés qui ne tiennent pas dans la poutre ; un retour qui ne tient pas dans la hauteur ;
un schéma que le fichier ne décrit pas encore (§ 6).

## 4. Les options de conception

Rubrique 7 de l'écran, pour le schéma 8 pieux sur semelles. Elles décrivent **le plan** ; elles ne changent **aucun**
résultat du calcul (vérifié : `tests/test_fondation_grue_ouvrage.py`, § 3). Elles vivent **dans la note** : la même
note exporte le même plan sur n'importe quel poste (des réglages gardés dans un seul navigateur feraient sortir deux
plans différents de la même note ; des réglages communs au bureau viendront avec les comptes).

**Les valeurs par défaut sont celles de Greenwood** (décision du propriétaire du 12/11/2026) : celles que la
conversation RBAR a relevées sur le plan de référence (GWW VAL COF 001, indice A) pour son fichier d'exemple.

| Option | Défaut (Greenwood) | Ce qu'elle fait |
|---|---|---|
| Mise en page des plans | selon le châssis | la ligne de la table ci-dessous ; un châssis absent prend la ligne du châssis immédiatement plus grand ; au-delà de 10 m, RBAR choisit. On peut aussi imposer une ligne, ou laisser RBAR choisir |
| Retours (poutre support, semelle) | longueur fixe : 50 cm, 60 cm | ou « arrêté sous la face opposée » (la pratique du bureau, § 5) ; 0 : barres droites |
| Arrêt sous la face opposée | 10 cm | retour = h − enrobage − Ø étrier − arrêt (la barre est dans les étriers) ; l'écran dit le retour obtenu ; un retour qui ne tient pas dans les étriers bloque l'export |
| Retours des barres latérales | 50 cm, 80 cm | longueur fixe |
| Zone d'étriers serrés (poutre support) | 2 m à chaque about | les étriers des extrémités de la note, puis ceux de la partie centrale |
| Rangées de barres latérales par face | 2 et 2 | 0 : aucune |
| Recouvrement des latérales au milieu | oui | |
| Coupes | support : au milieu de la zone serrée et au milieu ; semelle : au droit d'un pieu | ou au milieu, ou aucune |
| Cages de renfort au droit des pieux | 5 + 5 cadres, 70 × 70 cm, jambe 65 cm | le Ø vient de la note |
| Barres scellées (perfos) | 60 cm dont 40 scellés, « scellement chimique, en seconde phase » | nombre et Ø de la note ; posées en triangle sous chaque pied |
| Niveaux | dessus −0,50 m, recépage −1,20 m | vidés : non écrits sur le plan |
| Textes du plan | « Implantation », « Poutres transversales », « Poutres longitudinales », « Châssis de la grue », trois remarques de pose | vidés : absents (RBAR nomme alors les familles) |

**Règles fixes** (tirées de Greenwood, pas des options : rien ne les fait varier aujourd'hui) : un retour à chaque
about ; étriers de la poutre support en cadres qui se chevauchent (2 pour 4 brins, 3 pour 6) ; ceux de la semelle en
deux cadres pour 4 brins ; 2 brins : un étrier pleine largeur ; jamais de barres de montage ; poutres vues d'en haut
(trait plein) ; perfos : trois par plaque, deux à 42 cm du pied à ± 15 cm de l'axe, une à 11 cm sur l'axe.

**La table « mise en page par châssis »** (message de RBAR du 12/11/2026, valeurs du PDF d'essai, format A0, pas de
1/30) — `conception.MISES_EN_PAGE` :

| Châssis | Format | Plan | Détails | Familles | Coupes | Croisements | Implantation |
|---|---|---|---|---|---|---|---|
| 4,50 m | A0 | 1/20 | 1/10 | 1/20 | 1/20 | 1/5 | 1/50 |
| 4,60 m | A0 | 1/20 | 1/10 | 1/20 | 1/20 | 1/5 | 1/50 |
| 6,00 m | A0 | 1/20 | 1/10 | 1/20 | 1/10 | 1/10 | 1/50 |
| 8,00 m | A0 | 1/25 | 1/10 | 1/25 | 1/10 | 1/10 | 1/50 |
| 10,00 m | A0 | 1/50 | 1/20 | 1/20 | 1/10 | 1/10 | 1/100 |

## 5. Les questions sur les options de conception — les valeurs à donner

Ce sont les valeurs que le propriétaire doit fixer. Chacune ira, selon sa réponse, à l'un de trois endroits : **une
option** (on change sa valeur par défaut dans `conception.py`), **le classeur Excel** (la valeur se calcule, puis le
moteur la reprend), ou **le code dur** (une règle fixe, plus une option).

| # | Question | Aujourd'hui | Ce qu'il faut décider |
|---|---|---|---|
| Q1 | **Retours des barres inférieures et supérieures** | Greenwood : longueur fixe 50 cm (poutre support), 60 cm (semelle). La pratique du bureau dite le 12/11 — le retour monte et s'arrête à 10 cm sous la face supérieure — existe : « Arrêté sous la face opposée », arrêt 10 cm (Greenwood : 54,8 cm et 59,8 cm) | le mode par défaut ; l'arrêt (10 cm pour toutes les hauteurs ?) ; la mesure se prend-elle au **bout** du retour (c'est ainsi aujourd'hui, Ø étrier déduit) ? même règle vers le bas pour les barres supérieures ? arrondi (au cm, aux 5 cm) ? |
| Q2 | **Retours des barres latérales** | 50 cm et 80 cm (Greenwood) | une longueur fixe, ou une règle (une longueur d'ancrage, la largeur de la poutre…) ? |
| Q3 | **Longueur des zones d'étriers serrés** | 2 m à chaque about (Greenwood) ; le classeur donne un pas « aux extrémités » et un pas « en partie centrale », sans dire où l'on passe de l'un à l'autre | une longueur fixe, ou une règle (jusqu'au pied de la grue plus une hauteur ? un quart de la portée ?) — le classeur peut-il la calculer ? |
| Q4 | **Nombre de rangées de barres latérales** | 2 par face ; le classeur prescrit « 2Ø12 tous les 30 à 40 cm » sans en fixer le nombre | un nombre fixe, ou tiré de la hauteur (une rangée tous les 30 à 40 cm) ? |
| Q5 | **Cages de renfort : passage des brins aux cadres** | le classeur vérifie des BRINS (16Ø12, dont 5 de base, à Greenwood) ; le plan dessine des CADRES par sens : 5 + 5 (chaque cadre = 2 U), 70 × 70 cm, jambe 65 cm | la règle de passage brins → cadres ; l'emprise (70 × 70 : Ø du pieu plus quoi ?) ; la jambe (h − 2 × 5 cm ?) — le contrôle le rappelle à chaque export tant que ce n'est pas tranché |
| Q6 | **Barres scellées (perfos)** | 60 cm dont 40 scellés, « scellement chimique, en seconde phase », trois par plaque en triangle | les longueurs (fixes, ou une règle : produit, ATE) ; la remarque ; le motif de pose pour 6 ou 9 perfos par plaque (aujourd'hui : non posées, le contrôle le dit) |
| Q7 | **Niveaux** | dessus −0,50 m, recépage −1,20 m (Greenwood) ; le contrôle demande de les vérifier tant qu'ils valent ceux de Greenwood | ce sont des données du chantier : deviennent-elles des saisies de la note (rubrique 1 ou 3), sans valeur par défaut ? |
| Q8 | **Coupes** | support : au milieu de la zone serrée et au milieu ; semelle : au droit d'un pieu | confirmer |
| Q9 | **Dispositions d'étriers** (règle fixe) | support : cadres qui se chevauchent ; semelle : deux cadres pour 4 brins | confirmer ; et pour un nombre impair de brins (aujourd'hui : RBAR prend son standard, le contrôle le dit) |
| Q10 | **Textes du plan** | « Implantation », « Poutres transversales », « Poutres longitudinales », « Châssis de la grue », trois remarques de pose | le vocabulaire du bureau |
| Q11 | **Mise en page** | la table du 12/11 ; châssis absent : la ligne immédiatement plus grande ; au-delà de 10 m : RBAR choisit | les valeurs définitives de la table ; la règle d'un châssis absent ; où la modifier (aujourd'hui dans le code, `conception.MISES_EN_PAGE` ; plus tard un réglage du bureau) |
| Q12 | **Cartouche** | le site écrit le chantier, le nom de la note, la date | d'où viennent le numéro de projet, le bâtiment, la phase, la discipline, « vérifié par », les intervenants ? « dessiné par » = la personne connectée ? |

## 6. Les limites — ce que le fichier ne fait pas encore

- **Seul le schéma « 8 pieux — 2 poutres support sur 4 semelles », avec des semelles des deux côtés, s'écrit**
  (Greenwood, BASE). Les variantes 4 pieux et 8 pieux primaires, et la paroi de pieux sécants (le format de RBAR n'a ni
  paroi ni massif en version 1), ne l'ont pas : la ligne « DWG pour AutoCAD » est éteinte et dit pourquoi.
- **La charge horizontale des pieux n'est pas calculée** : le classeur ne répartit pas l'effort horizontal entre les
  pieux. La charge verticale part dans un encadré (une seule valeur : le pieu le plus chargé) tant que RBAR n'a pas
  sa clé.
- **L'historique des indices n'est pas gardé** : `revisions` ne porte que l'indice courant.
- **Le cartouche est incomplet** (§ 5, Q12).
- **`feuilles.echelles`, `cartouche`, `revisions`** sont lus par RBAR en notes (« clé non lue ») jusqu'à ce qu'il les
  ait faits — en cours de son côté. Aujourd'hui, RBAR choisit ses échelles ; pour Greenwood, son choix est celui de la
  table (vérifié, mesure du 12/11).
- **Le site ne fait pas le DWG** : le dessinateur ouvre AutoCAD avec RBAR et lance `RBAROUVRAGE`. Les plans PDF
  attendent le moteur de RBAR (§ 3).
- **Le visuel des plans** est celui de RBAR (il sera revu en une fois, plus tard, par décision du propriétaire).

## 7. Ce qui le prouve

- `tests/test_fondation_grue_ouvrage.py` : le fichier Greenwood du site **est** l'exemple de RBAR, à ses écarts voulus
  près (listés dans le test, chacun avec sa raison et à sa valeur) ; les règles du format de RBAR recopiées en
  contrôles, tenues par les fichiers aux options poussées à leurs bornes et par le classeur BASE, chaque contre-exemple
  vu ; chaque option change le fichier là où elle le dit, et rien du calcul ; la normalisation ; le contrôle et les
  blocages ; des mutations du fichier et des options, toutes vues.
- `tests/test_api_web_fondation_grue.py` : l'écran (la rubrique 7, le menu « Exporter », ses lignes éteintes) ;
  `tests/test_etabli_compact.py` : le composant « Exporter » du kit ; `tests/test_reference_refonte.py` : la photo des
  fondations de grues, trois cas pour les options (le fichier « rbar.ouvrage » y est photographié).
- Le harnais du navigateur, filtre `exporter` (`site/web/verifier_site.js`) : au PC, en tablette portrait et au
  téléphone, le menu ouvert est mesuré tel que l'œil le voit — entier dans l'écran et dans la colonne des résultats,
  chaque ligne atteinte d'un bord à l'autre —, « DWG pour AutoCAD » télécharge vraiment le fichier (son cartouche dit la
  note et son chantier), le menu se referme, « Contrôle avant export » ouvre sa fenêtre. Il a trouvé deux défauts, corrigés
  le 12/11/2026 : le menu restait ouvert après un téléchargement ; plus large que la colonne, il y perdait ses icônes au
  PC, et la colonne courte de la tablette portrait le coupait (les menus restent désormais dans leur colonne, `js/compact.js`).
- La mesure du 12/11/2026 : le fichier passé dans le **vrai lecteur** de RBAR et dessiné par son noyau — aucun refus,
  les deux feuilles produites (`docs/site/mesures/2026-11-12_fichier_rbar_lu_par_rbar.md`).

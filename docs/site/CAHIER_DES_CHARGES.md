# Site « Stab Méthodes » — cahier des charges

Ce que le bureau a demandé pour le site (stab-methodes.be), et les réponses retenues. Les maquettes qui le
montrent sont dans `site/maquettes/` (voir son README). Mis à jour le 06/11/2026 (relecture des maquettes :
navigation, paramètres, résultat sur deux lignes, fenêtre de calcul, rapports multi-modules, poutres béton armé).

**Refonte de l'établi (07/11/2026)** : spécification retenue, en cours de réalisation — `docs/site/REFONTE_ETABLI.md`.
Ses décisions (D1 à D9 : ouvrir un calcul n'écrit rien, « Chantiers », navigation à libellés, panneaux accrochés…)
réécriront ici les § 1 bis, § 5 et § 8 au fil des lots qui les réalisent ; d'ici là, ces paragraphes décrivent le
site tel qu'il est.

## 1 · Principes

- Le site remplace Streamlit comme écran principal ; **Streamlit reste en secours**, sur le même moteur.
- **Le calcul est celui du moteur Python du dépôt**, exécuté dans le navigateur : les mêmes chiffres, les mêmes
  formules, la même note PDF que Streamlit.
- **Gratuit** : le site sur Cloudflare Pages, les données sur Supabase, le domaine chez OVH.
- **Le PC d'abord.** Le téléphone n'est pas prioritaire : il doit fonctionner, sans plus d'effort. Depuis le
  07/11/2026, l'établi est **complet** au téléphone et à la tablette (§ 10 bis) : les mêmes fonctions qu'au PC, une
  disposition propre à chaque format.
- Premier module : le garde-corps (il fixe le style) ; ensuite le menu Béton, en commençant par les poutres
  (maquette faite : `site/maquettes/pages/poutre.html`). **Avant le site**, le calcul des poutres doit sortir de
  l'écran Streamlit (`modules/poutre.py` mêle calcul et écran) pour devenir un moteur aux formules nommées et
  testées, comme celui du garde-corps : le site ne fait que l'appeler.
- **Homogénéité** : tous les modules sont bâtis avec les mêmes composants (cadres, lignes de résultat, fenêtre de
  calcul, sélecteur) ; un module ne réinvente pas un geste qu'un autre a déjà.

## 1 bis · Navigation

- Une barre de **cinq icônes** : Accueil, Projets, Modules, Équipe, Paramètres ; chacune nommée par une bulle au
  survol. Rien d'autre dans la barre.
- **Deux navigations qui ne se mélangent pas** : « Projets » mène aux chantiers et à leurs dossiers ; « Modules »
  ouvre le volet des modules, classés par famille (Béton, Acier, Autres).
- Le **fil d'Ariane** (chantier › dossier › note) est cliquable.
- Au téléphone : les **cinq mêmes entrées**, en bas de l'écran, dans le même ordre.
- **Nouvelle note** : on choisit le chantier, le dossier, le nom, puis le module ; l'établi du module s'ouvre.

## 2 · Accès

- Le site est **privé**. Seules les **adresses e-mail invitées** par un administrateur y entrent.
- La personne invitée reçoit un lien et **choisit son mot de passe**. Rien d'autre à faire.
- Un administrateur peut retirer une personne ; elle n'entre plus, ses notes restent au bureau.

## 3 · Rôles et droits

| | Administrateur | Membre |
|---|---|---|
| Lire les notes des chantiers | oui | oui |
| Modifier une note | **toutes** | les siennes, et celles qu'on lui partage |
| Inviter, retirer, nommer administrateur | oui | non |
| Valider un module | oui | non |

- **Plusieurs administrateurs** (par exemple le chef de service et l'auteur du site).
- **Par défaut, seul l'auteur d'une note la modifie** (et les administrateurs).
- **Partager** : l'auteur d'une note ou d'un dossier peut autoriser un collègue à la modifier.
- **Dossier personnel** : visible par son propriétaire et les administrateurs ; il l'ouvre à qui il veut, en lecture
  ou en modification.

## 4 · Validation des modules

- Un administrateur pose une **petite pastille « validé »** sur un module dont il a relu le principe (avec son nom
  et la date). Elle se voit sur l'accueil et dans le module ; « à valider » sinon.
- La pastille porte la **version du module** : une nouvelle version du module repasse « à valider ».
- La pastille dit que la **méthode** est validée ; chaque note reste à relire par son auteur.

## 5 · Projets, notes et indices

- Chantier › dossiers › rapports. Chaque rapport garde ses **indices A, B, C…** avec leur historique.
- Statut d'un indice : **Brouillon**, **Diffusé**, **Old** (remplacé).
- Un indice **Old s'imprime grisé, avec le filigrane OLD** (tampon en travers de chaque page).
- Une note peut contenir **plusieurs éléments** (plusieurs garde-corps, plusieurs poutres…) : la liste est en tête
  des données, on passe de l'un à l'autre en un clic, on en ajoute un en un clic.
- Export et import d'un rapport en fichier **`.stab`**.

## 5 bis · Module, élément, rapport

- Un **module** est un type de calcul (Poutres béton armé, Garde-corps…). Un **élément** est une pièce calculée
  (la poutre V1, le garde-corps G1). Un **rapport** (la note) rassemble des éléments, **de plusieurs modules** :
  5 poutres, 3 dalles et 2 prédalles dans le même rapport.
- Le rapport imprimé a **une seule page de garde**, un **sommaire**, une **pagination continue** et un seul gabarit
  **A4 paysage** ; ses indices (A, B, C…) restent ceux du rapport.

## 5 ter · Paramètres

- La page **Paramètres** porte les **valeurs du bureau** (unités, format des notes, indice old, impression), les
  **matériaux par défaut** (béton, armatures, acier des profilés, E) et les **valeurs de chaque module** (poutres :
  γ_s, enrobage, jeux, armatures de montage ; garde-corps : méthode, σ_adm = f_y / 1,5, flèche, γ_M0, γ_M2).
- **Seuls les administrateurs** changent les valeurs du bureau.
- Une note peut changer une valeur **pour elle seule** : le champ porte « modifié » et la **valeur du bureau reste
  écrite dessous** ; les valeurs du bureau ne changent pas. Une valeur venue du bureau ne porte pas d'étiquette.
- Les valeurs s'affichent **telles quelles** : γ_M2 = 1,25 (jamais arrondi comme un taux).

## 6 · Impression

- Le bouton **Imprimer** ouvre la note prête à imprimer ; le navigateur **retient la dernière imprimante** utilisée.
- Un site ne peut pas choisir une imprimante tout seul (sécurité des navigateurs). Pour imprimer **sans aucune
  fenêtre**, un raccourci Chrome avec l'option `--kiosk-printing` envoie directement à l'imprimante par défaut de
  Windows — un réglage par PC, facultatif.

## 7 · Ne jamais perdre une note

- **Enregistrement automatique** à chaque modification (« Enregistré il y a 2 min »).
- **Historique** : chaque version enregistrée peut être rouverte ; on revient en arrière en un clic.
- **Corbeille** de 30 jours pour une note supprimée.
- **Sauvegarde quotidienne** de toute la base, et export `.stab` à tout moment.
- Si un calcul plante, la note garde ses **données** : on la rouvre, on corrige, on relance.
- **Un échec d'enregistrement se voit, à toutes les largeurs** (07/11/2026, refonte de l'établi, lot 0 bis) : l'indicateur
  de la barre du haut — au téléphone, la 2ᵉ ligne de l'en-tête — passe au rouge (« Non enregistré : stockage plein ») et
  y reste jusqu'à la prochaine écriture réussie ; un message, dit une fois, offre « Exporter .stab ». Avant, le texte
  était caché sous 1 360 px : l'échec y était muet.
  Au téléphone (07/11/2026, relecture du lot 0 bis), la 2ᵉ ligne tient entre le retour et la pastille de la note : elle
  écrit « ⚠ Non enregistré » seul, en rouge, entier ; la raison (« stockage plein », « mémoire illisible ») est dans le
  message, au survol, et en tête du menu ⋯, dont l'entrée rouvre le message. Mesuré avant : « Non enregistré : stockage
  plein » passait sous la pastille « À compléter » du plancher à 390 px, et « … mémoire illisible » sous toute pastille
  à 360 px. Au pire (pastille « Calcul impossible », 320 px), le texte finit par « … », jamais sous la pastille.
- **Une mémoire des notes illisible n'est jamais écrasée** (07/11/2026, lot 0 bis) : abîmée, elle n'a aucune note
  lisible, mais elle n'est pas « vide ». Avant toute écriture, son texte brut est copié à part, tel quel, une fois
  (`stab.notes.v1.illisible.<heure>`) ; si la copie échoue, rien n'est écrit et l'écran dit « Non enregistré : la
  mémoire des notes est illisible ». L'accueil le signale et rend ce texte en fichier. Avant, la première écriture
  remplaçait la mémoire par la seule note ouverte (30 notes → 1, mesuré).

## 8 · L'écran d'un module

- **La toile du dessin occupe toute la largeur.** Les données (à gauche) et le résultat (à droite) flottent dessus,
  dans deux cadres ; on les **replie** pour donner toute la place au dessin.
- **Résultat** : une vérification par ligne, **sur deux lignes** — le nom et la pièce (profilé, armatures), puis la
  jauge et le taux ; les lignes se groupent par élément ou par section ; la ligne qui gouverne est marquée.
- **La ligne entière ouvre le calcul** : une grande fenêtre (la toile moins la colonne Résultat, environ 75 % de la
  largeur), avec en tête les hypothèses (matériaux, méthode), puis un tableau **Vérification · Formule · Valeur ·
  Limite · Taux** (valeurs alignées à droite), puis le détail (formule = valeurs = résultat). Au téléphone, elle monte
  du bas sur toute la largeur, sans la colonne Formule.
- **La pièce ouvre le sélecteur** (profilés, dispositions d'armatures) : trié par masse ou par section d'acier ; le
  retenu (✓), l'optimal (★ le plus léger qui tient), quatre au-dessus et au-dessous ; la molette fait défiler, un clic
  choisit ; au survol, les propriétés (formé à froid EN 10219 ou fini à chaud EN 10210, A, I, W). Quand un plus léger
  tient, une ★ le signale sur la ligne du résultat.
- **Statuts** : Conforme, Non conforme, **À compléter** — un élément dont les données manquent n'est jamais
  « conforme ».
- **Méthode** : ELS (méthode du bureau, σ_adm = f_y / 1,5 = 235 / 1,5 = 156,7 N/mm² en S235) ou Eurocode (γ_M0,
  γ_M2) ; c'est un choix de la note, les deux ne se mélangent pas.
- Code couleur des taux : rouge au-dessus de 100 %, vert foncé de 95 à 100 %, vert pâle en dessous. Le taux
  **affiché** garde la couleur de sa bande : 94,98 % s'affiche 94 % (vert pâle), jamais « 95 % » en vert pâle.
- La toile a un **fond uni** ; les deux cadres se replient ; le dessin passe en plein écran.

## 9 · Noms des modules

Le titre seul, sans explication : il dit ce que fait le module.

- **Béton** : Poutres béton armé · Dalles béton armé · Prédalles · Charges sur plancher · Taux d'armature ·
  Enrobage · Maturité du béton · Tableau des armatures
  *(09/11/2026 : « Prédalles » a rejoint « Dalles béton armé » — une seule carte, le type « Prédalle + béton coulé en
  place » se choisit dans la dalle ; l'ancienne page redirige vers celle des dalles : `docs/site/DALLES_FUSION.md`.)*
- **Acier** : Choix de profilé · Catalogue des profilés · Cornières · Flambement · Assemblages métalliques · Soudures
- **Autres** : Garde-corps · Poutres bois · Rigidité du sol

D'autres changements de noms suivront ; cette liste est la référence.

## 10 · Première version du vrai site (06/11/2026)

- **`site/web/`** : l'accueil, le garde-corps et les poutres béton armé, avec de **vrais calculs**. Le PC d'abord.
- **Le calcul tourne dans le navigateur** : le moteur Python du dépôt, exécuté par Pyodide (Python 3.13 en
  WebAssembly). Mêmes chiffres que Streamlit, parce que c'est le même code. Aucun serveur de calcul à payer.
- **Rien n'est chargé ailleurs** : le cœur Python et numpy sont déposés dans le site, avec leurs empreintes
  vérifiées (celles du paquet officiel) ; le paquet du moteur porte la sienne, vérifiée avant chaque exécution. Le
  calcul n'a besoin d'aucun réseau. Seule exception, de **données** et à la demande : la météo d'un lieu (§ 10 ter).
- **Les notes** s'enregistrent dans le navigateur, le fichier **.stab** les emporte ; une note contient plusieurs
  éléments (plusieurs garde-corps, plusieurs poutres). Les comptes, les projets partagés et les droits viendront
  avec Supabase (§ 2 et 3).
- **Un fichier .stab reçu ne peut rien exécuter** : à l'import, seules les données sont gardées (noms, chantier,
  état) ; le dessin et le résumé sont refaits par le moteur. Tout nom (note, poutre, section, garde-corps) s'affiche
  comme du texte. Vérifié dans le vrai navigateur avec des fichiers piégés.
- **En ligne** : Cloudflare Pages relié au dépôt GitHub (dossier `site/web`, sans construction), le domaine
  stab-methodes.be, et Cloudflare Access pour le garder privé (les adresses invitées reçoivent un code par e-mail)
  en attendant les comptes. Les étapes : `site/web/README.md`.
- **Projets, équipe, journal d'activité** (11/11/2026) : les pages Projets et Équipe existent, l'accueil les porte —
  un projet est le nom d'un chantier, avec ses sous-projets ; chacun a ses initiales (1 à 5) dans une pastille, qui
  signent ses notes et ses lignes du journal. Tout est enregistré **dans le navigateur**, comme les notes : le partage
  entre les postes, les invitations et les rôles viendront avec les comptes (`docs/site/PROJETS_EQUIPE.md`).
- **Reste à faire** : les comptes (le partage des projets et de l'équipe entre les postes, les droits), les paramètres
  du bureau, les indices et le rapport assemblé d'un projet.

## 10 bis · Téléphone et tablette (07/11/2026)

Les relecteurs des modules ont trouvé l'établi inutilisable hors du PC : au téléphone, seul le cadre Données était
atteignable (aucun onglet) ; en tablette portrait, les deux cadres flottants écrasaient la toile à 62 px. Les réponses
retenues (gabarit commun : `site/web/construire.py`, `css/site.css`, `js/etabli.js`, `js/toile.js`) :

- **Une seule page, trois dispositions** ; les fonctions, les champs et les résultats sont les mêmes partout.
  - **PC et petit portable** (1 200 px et plus) : la maquette — la toile sur toute la largeur, les deux cadres
    flottants qu'on replie.
  - **Tablette portrait** (701 à 1 199 px) : la toile en haut sur toute la largeur (au moins 500 × 300 px offerts au
    dessin), les données et le résultat côte à côte dessous ; chaque cadre se replie.
  - **Tablette paysage** (701 à 1 199 px : 1 024 × 768, 1 180 × 820) : la toile à droite sur toute la hauteur (au
    moins 500 × 400 px), les données au-dessus du résultat à gauche. Les cadres flottants laissaient 450 px au dessin
    à 1 180 px.
  - **Téléphone** (700 px et moins) : le dessin des maquettes — en-tête compact (retour, nom de la note, taux, menu des
    actions : ouvrir, exporter, imprimer, thème, modules), onglets Données · Dessin · Résultat toujours visibles, barre
    du bas ; la fenêtre de calcul et le sélecteur montent du bas. Champs en 16 px (iOS n'agrandit plus la page), cibles
    de 36 px au moins, aucun défilement de côté. L'accueil a son en-tête et les cinq entrées en bas (§ 1 bis).
- **La toile** : zoom (− / + / ajuster, Ctrl + molette, pincer au doigt ; le dessin agrandi glisse) et **plein écran**
  de la vue (§ 8) ; la légende de **chaque** vue se lit ; un tableau défile dans sa feuille, aligné en haut à gauche ;
  un module sans vue affiche « Pas de dessin pour cette note. » (et non « le calcul est impossible »).
- **Corrections de la charte, faites dans la feuille du site** (la charte reste la copie des maquettes) : le cartouche
  ne se chevauche plus à ≤ 1 024 px ; dans « Dans cette note », c'est le nom long qui se coupe, jamais les dimensions ;
  les lettres grecques d'un en-tête en capitales gardent leur casse (η ne devient plus « H ») ; la grande pastille du
  verdict ne se comprime plus.
- **Un fichier propre au module** (export du JSON de l'écran Streamlit…) : bouton `data-telecharger`, action
  `telecharger` du routeur, fichier rendu par le moteur et enregistré par le navigateur sous un nom nettoyé (ni
  chemin, ni « .. », ni caractère invisible). Vérifié de bout en bout dans Chromium (filtre « telecharger ») : du
  bouton rendu par la brique au fichier enregistré, au PC et au téléphone, refus compris.
- **Le démarrage** : numpy n'est chargé que pour les modules qui s'en servent (garde-corps, choix de profilé, soudures),
  relevés par la construction du site ; un import imprévu le fait charger à la demande ; le paquet du moteur arrive
  pendant que Python démarre. Mesures ci-dessous.
- **Vérifié dans le vrai navigateur** : `site/web/verifier_site.js`, filtre « formats » (chaque module, quatre formats,
  ce que l'œil voit : tailles lues, visibilité, débordements, noms coupés).

## 10 ter · La météo du lieu (07/11/2026)

Demande de l'ingénieur : l'écran Streamlit « Âge / maturité du béton » allait chercher les températures du lieu du
chantier (Open-Meteo) pour calculer la maturité ; le site ne le faisait pas, sa politique de sécurité interdisant toute
connexion extérieure. Réponse retenue :

- **La météo est une IMPORTATION de données, à la demande** — jamais du code. Le calcul reste possible hors ligne
  (température constante, relevé saisi) ; une météo récupérée est **gardée dans la note** (le lieu trouvé, les dates, le
  jour et l'heure, les réponses du service), qui se recalcule ensuite **sans aucun appel**, et se transmet dans le .stab.
- **La politique de sécurité s'ouvre à trois adresses, et à elles seules** : `connect-src 'self'` plus le géocodage,
  l'archive et les prévisions d'Open-Meteo (gratuit, sans clé). Rien d'autre n'en vient (ni script, ni feuille, ni
  image) ; ce qui est reçu est relu par le moteur (des dates et des nombres) et tout nom renvoyé s'affiche comme du texte.
- **Le moteur décide, le navigateur transporte** : le moteur fait le plan des appels (la logique de l'écran Streamlit,
  prouvée au bit près), le navigateur les fait et rend les réponses brutes, le moteur les lit. Une brique commune,
  réutilisable par d'autres modules : `C.bouton_meteo` (`docs/site/MODULES_SITE.md` § 4).
- **Dans le module Maturité** : une troisième source de cure, « Météo du lieu », à côté de « Constante » et « Relevé » ;
  le lieu, les dates, le bouton ; le badge de Streamlit (jours mesurés, prévus, prolongés), la date de récupération, les
  températures reçues visibles (vue « Météo du lieu », tableau jour par jour) et **modifiables** ensuite ; le détail
  horaire à la demande. Les refus (pas de réseau, lieu introuvable, service en panne, dates hors de portée, réponse
  dont les jours ne sont pas ceux demandés, réponse de plus de 5 Mo, autre coulage ouvert pendant la récupération)
  arrivent en message.
- Le détail : `site/web/README.md` (la météo du lieu), `docs/beton/MATURITE_MOTEUR.md` § 4 bis.

**Mesure du démarrage (07/11/2026, Chromium de Playwright, serveur local, PC à 4 cœurs).** Du début de la navigation
au premier résultat affiché ; le site d'avant (copie figée) et celui d'après servis tour à tour sur le même port,
trois tours, une ouverture à cache vide puis trois à cache plein par tour ; médianes.

| Établi | Avant, cache plein | Après, cache plein | Avant, cache vide | Après, cache vide |
|---|---|---|---|---|
| Poutres béton armé (sans numpy) | 2,46 s | **2,14 s** (− 0,31 s, − 13 %) | 2,61 s | 2,19 s |
| Soudures (avec numpy) | 3,10 s | 3,16 s (inchangé, au bruit près) | 3,08 s | 3,21 s |

D'où vient le temps (poutre, cache plein) : le démarrage de Python lui-même (Pyodide : WebAssembly et bibliothèque
standard) ≈ 1,8 s ; numpy ≈ 0,34 s (épargné désormais à quatre modules : poutres, dalles, taux d'armature, tableau des
armatures) ; le paquet du moteur ≈ 40 ms ; le premier calcul ≈ 0,1 s (0,7 s pour soudures, qui importe numpy). Les
« ~6 s » relevés par les relecteurs l'étaient sur une machine chargée (le même parcours y mesurait 3,5 à 7,5 s). Ce qui
reste à gagner est dans Pyodide (un instantané de la mémoire après démarrage, par exemple) : non fait.


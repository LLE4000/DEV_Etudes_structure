# Bibliothèques du site : architecture proposée (08/11/2026)

- **Statut** : **proposition à trancher par l'ingénieur** (CLAUDE.md, liste de contrôle : une proposition part en
  document). Les lots BIB1 à BIB5 ci-dessous se font dans l'ordre, chacun fermé comme un lot M de
  `docs/site/REFONTE_ETABLI.md` (révision du 08/11/2026) : calcul inchangé ou regel gardé, tests, harnais aux trois
  formats, captures regardées, relecture de l'ingénieur sur le site.
- **Demande de l'ingénieur** (08/11/2026, « recadrage de l'organisation du site ») : distinguer les **Calculs** (des
  dimensionnements et des notes) des **Bibliothèques** (des références techniques et des outils rapides) ; une
  Bibliothèque Béton armé en une seule interface (menu compact à gauche, les tableaux sur presque toute la largeur,
  chaque rubrique avec un tableau complet à sélection et une fiche rapide) ; une Bibliothèque Acier (le catalogue des
  profilés) ; dans chaque calcul, un bouton discret « Bibliothèque » qui ouvre un grand panneau sans quitter la note,
  prérempli par la note, sans jamais modifier le calcul sans action explicite ; « Inclure dans la note » (fiche
  justificative compacte ou extrait de tableau), repris dans le PDF ; les extraits transmis de la NIT 260 (tableaux
  20, 21, 24, 25) comparés à ce que le site contient. Moins de modules visibles, « un seul menu » pour ancrages et
  recouvrements, des tableaux qui se voient d'un coup, sans défilement. Priorité : la Bibliothèque Béton armé et son
  intégration dans Poutres BA.

## 1. Ce qui existe (inventaire du 08/11/2026)

Dix-huit modules, tous présentés comme des calculs sur l'accueil (`site/maquettes/construire.py`, `FAMILLES`). Quatre
sont en fait des consultations (statut « Consultation », sans verdict) :

| Module (page) | Ce qu'il fait | Sa source de données | Tests |
|---|---|---|---|
| **Données béton** (`donnees-beton`) | cinq rubriques déjà : Classes de béton, Aciers d'armature, Tableau des armatures, Enrobages (durabilité, feu), Ancrages et recouvrements — **cette dernière vide** (« Tableaux du bureau à venir ») ; établi à trois colonnes, note de « consultations » | `beton_classes.json` (5 classes, valeurs du bureau α_b, μ, τ_adm), `beton/donnees/ec2.py` (tableau 3.1 : f_cm, f_ctm, f_ctk, E_cm), `beton/poutre/moteur.materiaux` (f_yd, γ_s 1,5 bureau et 1,15 EC2), `beton/armatures/tableau.py`, `beton/enrobage`, `beton/donnees/ancrages_recouvrements.json` (vide) | 97 contrôles écran, 25 moteur (tableau 3.1 de C12/15 à C90/105) |
| **Tableau des armatures** (`tableau-armatures`) | sections A_s pour Ø × n barres et par mètre, masse, « la plus économique » pour un A_s requis, repères multiples | `beton/armatures/tableau.py` : l'image « Tableau armature.png » du bureau recalculée (Ø 6 à 40, n 1 à 10, espacements 15/30/35/40 cm, ρ = 7 860 kg/m³), parité avec l'image prouvée aux pixels | 84 écran, 81 moteur |
| **Enrobage** (`enrobage`) | c_nom d'un élément (poutre, poteau, dalle, voile) : durabilité (c_min,b, c_min,dur, Δc_dev) et feu (EN 1992-1-2, méthode tabulée) ; éléments multiples | `beton/enrobage/donnees.py`, **copie de l'écran Streamlit `modules/enrobage.py` v2.0**, avec ses écarts à la norme D1 à D11 (`docs/beton/ENROBAGE_MOTEUR.md` §6), reproduits exprès et tenus par un test de parité | 53 écran (300 éléments), 34 moteur (parité Streamlit, 14 mutations) |
| **Catalogue des profilés** (`catalogue-profiles`) | tableau des profilés laminés (familles, recherche, filtres, tri, colonnes), fiche d'un profilé avec ses formules, comparaison (5), profilés personnalisés (PRS, caisson, tube, plat, rond), export en texte à copier | `acier/donnees/profiles.json` (consommé aussi par profil, flambement, cornière, garde-corps, soudures, plancher) ; `acier/profiles/consultation.py`, extrait de l'écran Streamlit | parité AppTest, 20 mutations |

**Ce qui manque pour la bibliothèque demandée** :
- **ancrages et recouvrements** : aucun calcul nulle part (le seul f_bd du dépôt sert aux tiges à crosse des pieds de
  poteau, `acier/assemblages/pied_poteau/platine_beton/composants.py`) ;
- **classes d'exposition belges** (EI, EE1-4, ES1-4, EA1-3 de la NBN B 15-001) : absentes ; la NIT 260 n'est citée
  nulle part ;
- **classe structurale** : seulement choisie à la main (S1 à S6) ; les modifications du tableau 4.3N de l'EC2
  (= tableau 21 de la NIT 260 : durée 100 ans, classe de résistance, dalle, contrôle qualité) ne sont pas calculées ;
- **aciers** : f_yk, f_yd, E_s seulement ; ni k, ni ε_uk, ni les classes de ductilité A/B/C ;
- dans le PDF (`ndc_pdf/`) : aucune brique de tableau (deux copies locales dans les notes garde-corps et soudures),
  aucun bloc « fiche justificative ».

Le champ « c enrobage » des modules poutre, dalle et prédalle est une simple saisie : rien ne le relie au module
Enrobage.

## 2. L'architecture proposée

### 2.1 L'accueil

- Les chantiers, dossiers et notes : inchangés.
- Les familles **Béton**, **Acier**, **Autres** restent ; chacune ne montre plus que ses **calculs**. Les tuiles de
  Données béton, Tableau des armatures, Enrobage et Catalogue quittent la liste.
- La **bibliothèque** de la famille (Béton armé, Acier) est une entrée **distincte** dans le bloc de sa famille : une
  carte large à part, avec l'icône d'un livre et un style propre (contour, teinte), placée en tête de ligne du bloc,
  avec le libellé de ses rubriques (« Bétons · Aciers · Sections · Enrobages · Ancrages » — les titres du menu ; le
  feu et les recouvrements sont dans les sous-titres de la page). Elle se distingue d'une tuile de calcul au premier
  regard, et reste à côté des calculs de sa famille. *(Forme remplacée le 09/11/2026 : voir la révision ci-dessous.)*
- **Révision du 09/11/2026 — la carte ÉTROITE et HAUTE, les modules à sa droite** (demande de l'ingénieur : « je la
  ferais tenir sur 2 en hauteur. Elle serait beaucoup moins large et elle serait plus haute. Et les modules seraient à
  droite. Donc à chaque fois, il y a la bibliothèque, puis les modules qui sont à droite »). Ce qui est fait :
  - **PC et tablette.** Le bloc d'une famille qui a une bibliothèque est `.tiles.fam-b` : la grille des tuiles de la
    charte (mêmes colonnes que partout : quatre au-dessus de 1 360 px, trois en dessous), dont la carte prend la
    **première colonne** — la largeur d'une tuile, à GAUCHE — et la **hauteur du bloc de ses modules** ; les modules
    de la famille sont dans leur propre grille (`.fam-b > .tiles`), posée en **sous-grille à partir de la deuxième
    colonne** : ils restent à DROITE quel que soit leur nombre, aucun ne passe sous la carte. La carte ne se mesure
    pas (`container-type: size`) : c'est le bloc des modules qui fait la hauteur de la rangée, la carte la prend
    (plancher : deux rangées de tuiles). Mesuré le 09/11/2026 : 234 × 120 px à 1 440 × 900 (les cinq modules du Béton
    sur trois colonnes : deux rangées, 3 + 2) ; 268 × 184 à 1 280, 351 × 184 à 1 180, 299 × 184 à 1 024, 231 × 184 à 820
    (sur deux colonnes : trois rangées, 2 + 2 + 1). Relu le même jour à 1 920 × 1 080, 1 366 × 768, 1 361 et 1 360
    (la bascule de quatre à trois colonnes), 1 100, 960, 768, 701 et 700 : la même forme, rien ne déborde.
    *Révision du 10/11/2026 (petits chantiers, demande de l'ingénieur) : de 701 à 1 360 px, la forme du PC — QUATRE
    colonnes pour toutes les familles (`css/site.css`, la charte n'en donnait que trois), la carte sur la première, les
    cinq modules du Béton et de l'Acier sur deux rangées, la carte à leur hauteur (120 px, mesurés à 701, 820, 1 024,
    1 180, 1 280 et 1 360 ; elle faisait 184 px, trois rangées) ; la carte, plus étroite (141 à 261 px), pose l'icône
    AU-DESSUS du titre et ses rubriques à la suite, séparées par « · » (sous 900 px, l'icône et les marges se resserrent,
    l'icône des tuiles passe à 30 px : « d'armature » se coupait à 701). Au-dessus de 1 360 px et au téléphone, rien ne
    change. Contrôle : `verifier_site.js`, filtre « bibliotheque » (deux rangées, 120 px, l'icône au-dessus du titre de
    701 à 1 360, à gauche à 1 440 ; un format 701 px ajouté).*
  - **La carte met la hauteur à profit** : l'icône du livre et « Bibliothèque » / « Béton armé » en tête (le titre sur
    deux lignes, jamais tronqué), puis ses rubriques **en liste**, chacune avec l'icône de son menu (les titres et les
    icônes de `RUBRIQUES`, une source). Sa mise en page suit la hauteur qu'elle reçoit (requête de conteneur) : à deux
    rangées (PC large), la liste se range sur deux colonnes (Bétons, Aciers, Sections | Enrobages, Ancrages) ; à trois
    rangées et plus, une seule colonne. Toute la carte reste le lien vers la page de la bibliothèque.
  - **Les familles sans bibliothèque** (Autres ; l'Acier jusqu'à BIB5) gardent leur grille, **alignée sur la marge
    gauche** : leurs tuiles tombent sur les MÊMES lignes de colonnes que la carte et les modules du bloc à
    bibliothèque (la carte occupe une colonne de la grille commune). Choisi plutôt que de décaler ces familles d'une
    colonne vers la droite : un seul rythme vertical pour toute la section (l'œil suit des colonnes continues d'une
    famille à l'autre), pas de colonne vide à gauche qui se lirait comme une carte manquante, et pas de rangée de
    plus pour l'Acier (à trois colonnes — tablette, petit PC — ses six modules tiennent sur deux rangées ; décalés
    d'une colonne, ils en prendraient trois).
  - **L'Acier aura sa bibliothèque (BIB5) sans rien recoder** : le gabarit lit `_bibliotheques_de(fam)` ; une
    bibliothèque déclarée dans `construire.BIBLIOTHEQUES` sur la famille Acier prend la même forme (contrôlé par
    `tests/test_site_web.py` avec une bibliothèque d'essai). Une seule bibliothèque par famille : la construction
    refuse la seconde (le gabarit n'a qu'une première colonne). À l'écran, `verifier_site.js` pose dans la page (à
    1 440 × 900) une carte Acier — et une sur « Autres », trois modules — sous cette forme et exige les mêmes contrôles
    que pour le Béton. Une famille de PEU de modules (une seule rangée, plus basse que le plancher de la carte) garde
    des tuiles à la hauteur d'une tuile, en haut du bloc (`align-content: start` ; relecture du 09/11/2026 : sans
    cela, la rangée unique s'étirait à 120 px). *Fait la nuit du 09 au 10/11/2026 (BIB5) : la vraie carte
    « Bibliothèque Acier » (Profilés, Nuances, Boulons) a pris cette forme sans une ligne de gabarit ; les essais
    (`test_site_web`, `verifier_site.js`) portent désormais sur « Autres », la seule famille sans bibliothèque.*
  - **Téléphone** : inchangé à l'œil — la carte en tête de la liste de sa famille, pleine largeur, l'icône à gauche,
    « Bibliothèque Béton armé » sur une ligne, les rubriques à la suite (« Bétons · Aciers · … », les « · » posés par
    la feuille). La même carte, rendue une fois ; seule la feuille diffère (R1).
  - **Les contrôles** : `tests/test_site_web.py` § 4 ter (la forme rendue : un bloc par famille avec exactement ses
    modules listés, la carte à gauche puis la grille des modules pour une famille à bibliothèque, la grille seule pour
    les autres, les cinq rubriques et leurs icônes dans l'ordre du menu, la bibliothèque d'essai sur l'Acier, la
    seconde refusée, la règle de sous-grille dans la feuille) ; `verifier_site.js`, filtre « bibliotheque » (à
    1 440 × 900, 1 280 × 800, 1 180 × 820, 1 024 × 768, 820 × 1 180, iPhone 13, et en thème sombre au PC et au
    téléphone : la carte au plus large comme une tuile, sur toute la hauteur du bloc de ses modules, à gauche de chacun,
    aucun sous elle, les modules sur toutes les colonnes suivantes ; les mêmes pistes de colonnes pour le bloc de chaque
    famille, LUES SUR LA GRILLE (relecture du 09/11/2026 : déduites des tuiles d'une famille sans bibliothèque, elles
    n'étaient plus que trois à 1 440 dès que « Autres », trois modules, restait seule — le contrôle aurait rougi sur la
    forme juste le jour de BIB5) ; le
    titre et les rubriques visibles dans la carte, aucun coupé ; au téléphone la carte pleine largeur en tête ; la carte
    distincte d'une tuile). Quatre mutations les font rougir (sous-grille retirée, ancienne forme rendue, carte qui ne
    prend plus la hauteur, règle du téléphone retirée).
- Le volet « Modules » du rail et le menu du téléphone listent **Bibliothèques** comme un groupe à part.
- Les pages des anciens modules restent **servies** (une note existante s'ouvre encore ; une note neuve se crée
  encore depuis l'adresse de la page), mais ne sont plus listées ; la liste nommée `MASQUES`, à côté de `FAMILLES`
  dans `site/maquettes/construire.py`, le dit et les tests la lisent (`donnees_beton`, `armatures`, `enrobage` ; le
  catalogue `catalogue` depuis BIB5). `js/modules.js` les garde avec le drapeau `masque`, pour que l'accueil
  (« Reprendre ») et le harnais les connaissent. Leur retrait définitif se décidera quand la bibliothèque aura remplacé
  chaque fonction (§ 3).

### 2.2 La Bibliothèque Béton armé — une page, cinq rubriques

Page `bibliotheque-beton` (module `api_web/biblio_beton.py`), **sans les trois colonnes de l'établi** : un gabarit
propre (`construire.page_bibliotheque`) :

- à gauche, un **menu compact** de cinq entrées (56 px de haut, icône et titre court) ; au téléphone, une barre de
  segments en haut ;
- à droite, une **barre de paramètres** sur une ligne (les entrées de la fiche rapide), puis **le tableau**, sur
  toute la largeur, dimensionné pour tenir dans l'écran sans défilement à 1 440 × 900 et sur tablette ;
- la **fiche rapide n'est pas un second écran** : ses paramètres **allument la case** du tableau et une bande sous le
  tableau donne la valeur retenue, en gros, avec « Pourquoi ? » (les lignes de justification, les formules, la
  référence) à la demande, et « Inclure dans la note » quand la bibliothèque est ouverte depuis une note (§ 2.4) ;
- un clic sur une case du tableau fait l'inverse : il règle les paramètres et montre la même bande ;
- très peu de texte permanent : les conditions d'application et les sources sont dans « Pourquoi ? » et dans une ligne
  de source sous le tableau (norme, édition, tableau ou paragraphe).

| # | Rubrique | Tableau(x) | Paramètres de la fiche rapide → valeur | Moteur |
|---|---|---|---|---|
| 1 | **Bétons** | classes C12/15 à C90/105 × f_ck, f_ck,cube, f_cm, f_ctm, f_ctk,0,05, f_ctk,0,95, E_cm, ε_c2, ε_cu2, f_cd ; les valeurs du bureau (α_b, μ_B500, μ_B400, τ_adm) pour les cinq classes qui en ont | classe, γ_c (1,5), α_cc → la fiche de la classe | `beton/donnees/ec2.py` (existant, à compléter des ε et de f_cd), `beton_classes.json` |
| 2 | **Aciers d'armature** | B400, B500 (A, B, C) × f_yk, f_yd (γ_s 1,15 EC2 et 1,5 bureau), E_s, ε_yd, k, ε_uk, classe de ductilité ; les treillis soudés (désignation, Ø, mailles, section par mètre) | nuance, γ_s → la fiche ; treillis → sa section | `beton/poutre/moteur.materiaux`, `beton/treillis.py` (existants) ; k, ε_uk (EN 1992-1-1 annexe C, à ajouter) |
| 3 | **Sections et masses** | Ø × n barres (1 à 10) : A_s et masse ; Ø × espacement : A_s par mètre | A_s requis (barres ou nappe) → les dispositions qui suffisent, la plus économique ; ou Ø et n (ou e) → A_s | `beton/armatures` (existant, inchangé) |
| 4 | **Enrobages et feu** | c_min,dur : classe structurale S1-S6 × exposition (EC2, avec le libellé belge de la NIT 260) ; les modifications de classe structurale ; feu : distance à l'axe a et b_min par durée R, pour poutre (tableau 5.5) et dalle (tableau 5.8) — poteau et voile dits « non transcrits » tant que l'ingénieur n'a pas relu les tableaux 5.2a et 5.4 (§ 4, Feu) | élément, exposition, classe de béton, durée d'utilisation (50/100 ans), contrôle qualité, Ø barre, Ø étrier, Δc_dev, feu (R, b_min), coulé contre le sol → **c_nom** et sa chaîne (c_min,b, classe structurale calculée, c_min,dur, c_min, Δc_dev, a, c_feu) | **nouveau** `beton/bibliotheque/enrobage.py` aux valeurs de la norme (§ 4) ; l'ancien `beton/enrobage` garde l'écran Streamlit — **10/11/2026 (Lot Q)** : DEUX ONGLETS, « Calcul » (par défaut : le détail dessiné à l'échelle, la feuille de calcul terme par terme, c_nom allumé) et « Tableaux » (4.4N et le tableau du feu ENTIER de l'élément : 5.2a, 5.3/5.4, 5.5/5.6, 5.8/5.9, case allumée et cliquable, état de vérification dessous) ; les quatre éléments au feu, poteau au tableau 5.2a par défaut ou à la formule (5.7) ; paramètres partagés dans l'adresse (`onglet=`, l'ancien `vue=` ouvre « Tableaux ») — `api_web/biblio_enrobages.py`, `api_web/dessin_biblio_enrobage.py`, moteur du feu `beton/bibliotheque/feu.py` |
| 5 | **Ancrages et recouvrements** (un seul menu) | l_bd et l_0 : Ø (6 à 50) × classe de béton (C20/25 à C50/60), B500, bonnes conditions, σ_sd = f_yd ; un sélecteur bonnes / médiocres conditions ; l_0 pour un recouvrement de plus de 50 % | Ø, classe, conditions d'adhérence, forme de l'ancrage (droit, crochet/coude/boucle avec c_d), barre transversale soudée, taux σ_sd/f_yd (ou A_s,req / A_s,prov), proportion recouverte, pression transversale → **l_bd**, **l_0**, leurs minimums, les facteurs α | **nouveau** `beton/bibliotheque/ancrages.py` (EN 1992-1-1 §8.4.2-8.4.4, §8.7.3, γ_c 1,5, γ_s 1,15 ; validé contre la NIT 260, § 4) |

Les tableaux sont **calculés** par les moteurs, jamais recopiés (R3) ; la NIT 260 sert de **contrôle** dans les
tests, et de référence citée dans « Pourquoi ? ». Un paramètre hors du domaine d'un tableau le dit (Ø > 32 mm :
approximation ; recouvrement de Ø 40 et 50 sous conditions ; XA et XF non couvertes ; précontrainte exclue).

**Contrat technique.** Une seule fonction de rendu, `biblio_beton.consulter(rubrique, params) → {menu, barre,
tableau, bande, pourquoi, sources, fiche}`, appelée par le routeur (action `consulter`). La page la demande au
chargement et à chaque changement de paramètre ; le panneau d'une note (§ 2.4) appelle la **même** fonction avec les
paramètres de la note : un seul HTML, deux hôtes. La page ne crée pas de note (une consultation n'en est pas une) ;
ses paramètres vivent dans l'adresse (`?rubrique=enrobages&expo=XC1…`) : une adresse se partage.

### 2.3 La Bibliothèque Acier

La page du catalogue devient l'entrée « Bibliothèque Acier » (même adresse, même moteur, mêmes notes), avec le
gabarit de § 2.2 : à gauche le menu (**Profilés laminés** aujourd'hui ; places prévues pour nuances d'acier,
boulons, soudures), à droite le tableau sur toute la largeur. Les outils qui dimensionnent (cornière, flambement,
profil, assemblages, soudures) restent des calculs.

**Révision de la nuit du 09 au 10/11/2026 — BIB5 fait, autrement que prévu sur un point.** La Bibliothèque Acier est
une page NOUVELLE (`bibliotheque-acier`, `api_web/biblio_acier.py`), au gabarit de la béton armé (`construire.
page_bibliotheque`, `js/biblio.js`, `css/bibliotheque.css`) — et non la page du catalogue renommée : l'ancienne page
`catalogue-profiles` garde son adresse, son moteur et ses notes (masquée de l'accueil comme Données béton, une ancienne
note s'y rouvre : contrôlé dans le vrai navigateur). Raison : une consultation n'est pas une note (§ 2.2), et le
module du catalogue en est une (recherches multiples, profilés personnalisés) ; garder les deux évite toute migration
de format. Trois rubriques :

- **Profilés** — le catalogue entier, sur la largeur : familles en pastilles (groupe par groupe, avec leur nombre),
  recherche (« ipe200 », « hea 3 »), séries courantes, tri (sélecteur, ou clic sur un en-tête ; un second clic
  renverse), lignes (100, 200, 500, toutes ; le profilé ouvert toujours montré), menu « Colonnes » (39 propriétés en
  pastilles, gardé ouvert d'un clic à l'autre), filtres numériques (une ligne libre de plus à chaque filtre posé, trois
  au plus ; « Effacer les filtres »). Un clic sur une ligne ouvre le profilé : sa ligne s'allume, la bande donne sa
  vignette, son nom, sa masse, sa famille et sa norme, ses dimensions, A, I_y, W_pl,y, I_z ; avec un filtre posé, « le
  plus léger qui satisfait les filtres » (★ dans le tableau et la bande, « Ouvrir … »). « Fiche du profilé » (le volet
  « Pourquoi ? ») : la section à l'échelle (le dessin du catalogue), puis TOUTES ses propriétés groupe par groupe,
  chacune avec la formule du catalogue, valeurs substituées, ou sa définition ; la méthode de I_t ; les sources. Le
  tableau est grand : il DÉFILE dans son cadre (l'en-tête et la colonne des noms restent posés), la bande reste dans
  l'écran. Rien n'est refait : liste, colonnes, cellules, plus léger, raisons d'un tableau vide, formules sont ceux du
  module du site « Catalogue des profilés » (`api_web/catalogue.py`, tiré mot pour mot de l'écran Streamlit), les
  nombres ceux de `acier/donnees/profiles.json`.
- **Nuances** — les aciers de construction tels que les moteurs d'assemblage, de soudure et de garde-corps les lisent
  (`acier/donnees/bibliotheques.json` : f_y, f_u, β_w), f_y de 40 à 80 mm et ε par le moteur du flambement
  (`acier/flambement/normes.py`, tableau 3.1), f_vw,d par `acier/resistances.fvw_d` avec le γ_M2 de
  `acier/soudures/criteres.py`. Bande : f_y selon l'épaisseur (t ≤ 40 mm ; 40 < t ≤ 80 mm, avec l'avertissement que
  f_u n'est connu du dépôt que pour t ≤ 40 mm). *Relecture du 10/11/2026 : à 40 < t ≤ 80 mm, l'en-tête de f_u (et
  de ε), la phrase de la bande et la ligne f_vw,d de la justification et de la fiche disent « t ≤ 40 mm » — f_vw,d est
  calculé avec le f_u de t ≤ 40 mm, plus grand que celui du tableau 3.1 au-delà (S355 : 510 contre 470 MPa, f_vw,d
  +8,5 %) ; il ne se lisait pas comme tel.*
- **Boulons** — les boulons et leurs classes tels que les assemblages les lisent (bibliotheques.json : d, d_0, A,
  A_s, d_m, d_w ; f_yb, f_ub, α_v) ; F_t,Rd et F_v,Rd (plan de cisaillement dans le filetage, A_s) par
  `acier/resistances.Ft_Rd` / `Fv_Rd`, les fonctions des moteurs d'assemblage. Bande : F_v,Rd (vue « Boulons »), f_ub
  (vue « Classes »).

Aucune valeur n'est retapée dans la page (R2, R3) : `tests/test_api_web_bibliotheque_acier.py` relit les fichiers À
PART et compare, et quatre mutations montrent que la page suit ses sources. Les DEUX copies de f_y (t ≤ 40 mm) qui se
rencontrent dans la rubrique « Nuances » — `bibliotheques.json` et `normes.NUANCES` — sont NOMMÉES et liées par ce même
test. Deux autres copies existent et ne sont pas liées (relevé, rien de modifié : moteurs hors du lot) :
`acier/choix_profil/moteur.FY_EPAIS` (f_y de 40 à 80 mm, 3 nuances) et `plancher/charges/operations.FY_MATERIAU`
(« la table de l'écran d'origine, telle quelle »).

*Ce qui reste (BIB5), et ce qui manque pour d'autres rubriques* : le bouton « Bibliothèque » des calculs acier
(« Choix de profilé » d'abord, § 2.4) n'est PAS fait — l'établi de ce module n'est pas compact, il faudrait poser le
bouton et le panneau dans `construire.page_module` hors de `compact()`, et `bibliotheque_prefill` dans
`api_web/profil.py` : un lot à part, avec sa relecture ; les profilés personnalisés, la comparaison et l'export en
texte du catalogue ne sont que dans l'ancienne page (masquée, ouvrable par l'adresse et ses notes) — les porter dans la
bibliothèque, ou décider de les laisser au calcul, est à trancher. Rubriques futures, sans valeur dans le dépôt
aujourd'hui (rien n'a été inventé) : **soudures** (épaisseurs de gorge minimales et maximales, longueurs utiles : le
moteur des soudures ne les tabule pas) ; **f_u au-delà de 40 mm** (tableau 3.1, à transcrire depuis le texte) ;
**trous et pinces** (e_1, e_2, p_1, p_2 minimaux : lus par chaque moteur d'assemblage dans ses propres règles, aucune
table commune) ; **ancrages et chevilles** (valeurs ETA du pied de poteau, propres à chaque produit) ; **tôles et plats
du commerce** (`bibliotheques.json` n'a que cinq épaisseurs et huit ailes de cornière : une table du bureau serait à
fournir).

### 2.4 Le bouton « Bibliothèque » de chaque calcul

- Dans le ruban du haut (`construire.compact`, avant « Enregistrer comme note… »), un bouton discret
  `data-action="bibliotheque"` (icône livre, libellé caché sous 1 100 px comme les autres) ; dans le menu ⋯ du
  téléphone, l'entrée « Bibliothèque ». Un module l'a quand il sait préremplir sa bibliothèque
  (`api_web/<cle>.bibliotheque_prefill`, `construire._bibliotheque_du_module`) : la poutre aujourd'hui ; dalle et
  prédalle quand elles l'auront (le panneau et le script sont prêts pour les trois).
- Il ouvre un **grand panneau** (`#et-bibliotheque`, fenêtre modale comme `#et-rangement` mais de
  `min(1240px, 100vw − 32px)` et presque toute la hauteur ; au téléphone, une feuille plein écran) qui contient le
  **même gabarit** que la page de la bibliothèque (`construire._gabarit_bibliotheque`, câblé par le même
  `js/biblio.js` ; `js/biblio_note.js` porte ce qui est propre à la note). Le routeur (action « bibliotheque ») rend la
  rubrique avec les **paramètres connus de la note** — pour la poutre : l'élément « poutre », le béton, le plus gros Ø
  des lits et des cadres, b ; les ancrages, bétons, aciers et sections reçoivent les leurs — et ouvre les **Enrobages**
  d'abord. Les paramètres changés à l'écran priment ; à chaque ouverture, le panneau repart de la note.
- Rien du calcul ne change tant qu'on ne clique pas « Utiliser cette valeur » : le routeur nomme la **cible**
  (`bibliotheque_cibles` : le chemin `modifier`, la valeur, le libellé — c_nom en cm dans « c enrobage », la classe
  dans « Béton », la nuance dans « Acier ») ; le bouton dit ce qu'il écrira (« Utiliser cette valeur · c = 2,6 cm »,
  l'ancienne valeur au survol) et reste **grisé avec la raison** quand le module ne connaît pas la valeur (une classe
  hors des tableaux du bureau) ou qu'aucun champ ne la reçoit (ancrages, sections). Après l'écriture, le panneau se
  relit depuis la note (la case allumée suit) ; Échap et « Fermer » reviennent à la note telle quelle, le focus au bouton.

### 2.5 « Inclure dans la note » et le PDF

- Deux insertions, depuis la bande de valeur du panneau : la **fiche justificative** (valeur retenue, paramètres,
  lignes de justification avec leur règle, sources) et l'**extrait de tableau** (le tableau entier de la rubrique,
  la ligne et la case retenues marquées, avec l'en-tête et la source) — la même fiche figée, la sorte dit ce qu'on
  met en avant.
- Elles s'écrivent dans l'état de la note : `etat.fiches = [ {id, sorte: "fiche" | "extrait", bibliotheque, rubrique,
  titre, symbole, valeur, texte, unite, phrase, params, lignes: [{symbole, nom, valeur, texte, unite, regle}], sources,
  avertissements, extrait: {titre, entetes, lignes, case, source}, date, moteur} ]` — **une copie figée** de tout ce
  qu'il faut pour relire la fiche sans le moteur (traçabilité des notes anciennes ; c'est une copie d'un run gelé au
  sens de R2, comme D-013 : `texte` est la valeur telle que la bande l'affichait, `moteur` nomme la bibliothèque et sa
  version). `api_web/fiches.py` (partagé par poutre, dalle, prédalle) la **relit strictement** (clé inconnue, sorte,
  extrait non rectangulaire, case hors du tableau, textes bornés, 24 fiches au plus : refusé avec un message, l'état
  intact) ; la clé `fiches` est **absente** tant que la note n'en a aucune — une note ancienne reste identique à
  l'octet, le gel (`test_reference_refonte`) n'a pas bougé. Le routeur sert l'action « fiches » (ajouter, retirer,
  monter, descendre) comme « modifier » : l'état, les fragments, un message ; « Annuler » la défait.
- Dans l'établi compact, sous l'arborescence, le bloc **« Justifications (n) »** de la note : une ligne par fiche
  (titre, valeur), menu ⋯ (monter, descendre, retirer) ; un clic ouvre au centre la vue « J1 » (clé `fiche-0` — jamais
  « s0 », que js/compact.js lit comme une section) : titre, valeur en grand, phrase des paramètres, lignes, extrait
  (la case allumée), sources, date. Rien de tout cela tant que la note n'a aucune fiche (très peu de texte permanent).
- Dans la note PDF (`ndc_pdf/styles.Style.justifications`, `build(fiches=…)`) : après les planches, des pages A4
  **paysage**, même charte, « Justifications de la bibliothèque », annoncées au sommaire de la garde ; **deux colonnes**,
  chaque fiche commence dans une colonne neuve et se poursuit dans la suivante (« J2 (suite) ») — le flux des planches,
  jamais une fiche tronquée ; l'extrait est un tableau (la ligne et la case retenues en gras) qui tient entier dans sa
  colonne (douze colonnes du tableau 3.1 comprises). La brique **`ndc_pdf.kit.table`** porte le corps des tableaux :
  les notes garde-corps et soudures l'appellent (leurs copies locales sont retirées), les fiches aussi — une source.
- Le gabarit du PDF n'est pas refait : la page de garde, le cartouche et les planches restent ; sans fiche, le document
  est celui d'avant.

## 3. Correspondance ancien → nouveau

| Aujourd'hui | Demain | Données, calculs, notes |
|---|---|---|
| Données béton · Classes de béton | Bibliothèque Béton armé · 1 Bétons | mêmes sources ; les ε et f_cd en plus |
| Données béton · Aciers d'armature | · 2 Aciers d'armature | mêmes sources ; k, ε_uk, ductilité, treillis en plus |
| Données béton · Tableau des armatures, **Tableau des armatures** (module) | · 3 Sections et masses (tableau + fiche rapide « A_s requis → dispositions ») | `beton/armatures` inchangé ; les notes « repères » existantes s'ouvrent sur l'ancienne page |
| Données béton · Enrobages, **Enrobage** (module) | · 4 Enrobages et feu (tableaux + fiche rapide = le calcul complet de c_nom) | nouveau moteur aux valeurs de la norme (§ 4) ; l'ancien module reste servi avec son moteur de parité Streamlit |
| Données béton · Ancrages (vide) | · 5 Ancrages et recouvrements (calculés) | nouveau moteur ; le format « tableaux du bureau » (`ancrages_recouvrements.json`) reste disponible pour un tableau maison |
| **Catalogue des profilés** | Bibliothèque Acier · Profilés (BIB5 : page `bibliotheque-acier` ; + Nuances, Boulons) | module inchangé, masqué, sa page et ses notes servies ; la bibliothèque lit sa liste et ses formules (une source) ; personnalisés, comparaison, export : seulement dans l'ancienne page |
| Maturité du béton, Taux d'armature, Charges sur plancher, Poutres, Dalles, Prédalles | Calculs · Béton | inchangés |
| Choix de profilé, Cornières, Flambement, Assemblages, Soudures | Calculs · Acier | inchangés |
| Garde-corps, Poutres bois, Rigidité du sol | Calculs · Autres | inchangés |
| — | bouton « Bibliothèque » dans chaque calcul, « Inclure dans la note », fiches dans le PDF | nouveau |

## 4. Les extraits de la NIT 260 (CSTC, mars 2017) comparés au site

Les valeurs ci-dessous sont celles des images transmises ; aucune mise en page de la publication n'est reproduite : les
tableaux du site sont les nôtres, calculés, et citent la source.

**Tableau 20 — c_min,dur selon la classe structurale et l'exposition.** Il est identique au tableau 4.4N de
l'EN 1992-1-1 (sans la colonne X0) : S4 → 15 / 25 / 30 / 35 / 40 / 45 mm pour XC1 / XC2-XC3 / XC4 / XD1-XS1 /
XD2-XS2 / XD3-XS3. Le site (`beton/enrobage/donnees.py`, copie de l'écran Streamlit) écrit **+ 5 mm** en X0 et XC1
de S3 à S6 (écart D4, côté sûr ; S4/XC1 = 20 au lieu de 15). Le tableau 20 donne aussi la **correspondance belge** :
XC1 ↔ EI ; XC2, XC3 ↔ EE1, EE2 ; XC4 ↔ EE3 ; XD1, XS1 ↔ ES2 ; XD2, XS2 ↔ ES1, ES3 ; XD3, XS3 ↔ EE4, ES4. Notes :
XA et EA → la plus grande valeur exigée pour les autres classes ; précontrainte + 10 mm (+ 5 mm sur les cases
astérisquées 10*).

**Tableau 21 — classe structurale.** Départ S4 ; durée d'utilisation de 100 ans : + 2 classes ; classe de résistance
≥ C30/37 (XC1), ≥ C35/45 (XC2-XC3), ≥ C40/50 (XC4, XD1, XD2, XS1), ≥ C45/55 (XD3, XS2, XS3) : − 1 ; élément
assimilable à une dalle (faces d'éléments plans attaquées dans une seule direction, jamais les faces latérales) : − 1 ;
maîtrise particulière de la qualité (préfabrication sous les conditions de la NBN EN 13369 §6) : − 1 ; cumulables.
C'est le tableau 4.3N de l'EC2. **Le site ne le calcule pas** (classe choisie à la main, avec une proposition
par exposition, D3, non conservative sous S4). L'exemple de la NIT (dalle C30/37, XC1/EI, treillis Ø12 : S4 → S2,
c_min,dur = 10, c_min = 12, c_nom = 22 mm) devient un cas de test.

**Tableau 24 — l_bd (B500, bonnes conditions d'adhérence)** et **tableau 25 — l_0 (recouvrement de plus de
50 %)**, Ø 6 à 50 × C20/25 à C50/60. Recalculés par l'EN 1992-1-1 (§8.4.2 : f_bd = 2,25 η1 η2 f_ctd, f_ctd = f_ctk,0,05
/ γ_c, γ_c = 1,5 ; §8.4.3 : l_b,rqd = Ø/4 · σ_sd / f_bd avec σ_sd = f_yd = 500/1,15 ; η2 = (132 − Ø)/100 au-delà
de 32 mm ; §8.7.3 : l_0 = α6 · l_b,rqd, α6 = 1,5 pour plus de 50 % recouvert), ils se retrouvent **à l'arrondi
au centimètre près**. Mesuré le 08/11/2026 par `tests/test_bibliotheque_moteur.py` sur les 154 cases : la case du
moteur (arrondie au cm **supérieur**) s'écarte de la NIT (arrondie au plus près) d'au plus **2 cm jusqu'à Ø 32**
(C35/45 Ø 32 : 1 040 pour 1 020 ; C30/37 Ø 32, l_0 : 1 720 pour 1 740) et de **3 cm (l_bd) à 5 cm (l_0) à Ø 50**
(C30/37 : 2 180 pour 2 210, 3 270 pour 3 320) ; la longueur calculée reste à **1,5 %** de la NIT une fois son arrondi
retiré. La NIT a pris f_ctk,0,05 à sa manière (C30/37 : 2,0 MPa imprimé, là où 0,7 · f_ctm donne 2,03) ; le moteur suit
la relation analytique du tableau 3.1, comme `beton/donnees/ec2.py` ; les coefficients ≈ 47 Ø … ≈ 25 Ø de la NIT sont
ceux de ce calcul. La NIT elle-même note que ses valeurs sont des approximations valables
pour Ø ≤ 32 mm, arrondies au centimètre. Conditions médiocres : × 1,43 (η1 = 0,7) ; réductions : taux de
contrainte, 0,7 pour un ancrage courbe tendu avec c_d > 3 Ø, 0,7 pour une barre transversale soudée ; jamais sous
10 Ø ni 100 mm ; recouvrement jamais sous 15 Ø ni 200 mm ; Ø 40 et 50 recouverts seulement si la section fait plus
d'un mètre ou si l'acier est limité à 80 %. Tout cela devient le moteur de la rubrique 5 et ses tests.

**Normes et éditions** (à confirmer par l'ingénieur sur les textes papier — cette session n'a pas accès au web) :
EN 1992-1-1:2004 + AC:2010 avec l'annexe belge NBN EN 1992-1-1 ANB:2010 (γ_c 1,5, γ_s 1,15, Δc_dev 10 mm, classe
structurale S4 pour 50 ans ; α_cc 0,85 est écrit « ANB à confirmer » dans le code) ; EN 1992-1-2:2004 + ANB pour
le feu ; NBN B 15-001:2018 pour les classes d'environnement belges ; NIT 260 (2017), elle-même fondée sur ces
textes. La seconde génération (EN 1992-1-1:2023) change la méthode d'ancrage (tableaux de l_bd) ; elle n'est pas
encore applicable en Belgique sans son annexe nationale : la bibliothèque le dira dans sa ligne de source.

**Feu.** Le moteur actuel reproduit des valeurs de l'écran Streamlit qui s'écartent de l'EN 1992-1-2 (D1 : poteaux
lus dans la colonne μ_fi = 0,2 et non 0,7 ; D2 : voiles R180/R240), **non conservatives** et signalées en rouge à
l'écran. La bibliothèque prend les valeurs de la norme : les **poutres** (tableau 5.5, avec a_sd = a + 10 mm pour les
barres d'angle d'un lit unique) et les **dalles** (tableau 5.8) sont transcrites, identiques à celles de l'écran
Streamlit ; les **poteaux** (tableau 5.2a) et les **voiles** (tableau 5.4) **ne le sont pas** (`FEU_A_TRANSCRIRE`) :
leurs valeurs à l'écran Streamlit sont douteuses et cette session n'a pas le texte — la rubrique le dit et ne donne
aucune valeur, jusqu'à ce que l'ingénieur transcrive les deux tableaux depuis l'EN 1992-1-2 (μ_fi = 0,7).

**Feu — fait le 10/11/2026 (Lot Q, docs/site/LOT_Q_QUATRE_MODULES.md § 4).** Les tableaux 5.2a (poteaux, méthode A),
5.3 et 5.4 (voiles), 5.5 et 5.6 (poutres), 5.8 et 5.9 (dalles) sont transcrits dans `beton/bibliotheque/feu.py`, leur
SEULE source, recoupés sur deux rapports de sources indépendants (aucune case contredite) ; chaque tableau porte son
état de vérification, que l'écran écrit dessous. `FEU_A_TRANSCRIRE` a disparu : la rubrique donne les poteaux
(R 90, μ_fi = 0,7 : 350/53) et les voiles (REI 180, μ_fi = 0,7, une face : 210/50). Le poteau se lit au tableau 5.2a de
l'EN PAR DÉFAUT (le plus sévère) ou, au choix, par la formule (5.7) avec α_cc = 0,85, avec la mention de l'annexe belge
et la case du tableau en comparaison. Rien n'est interpolé ; une dimension sous le minimum, ou une case vide (R 240 à
μ_fi = 0,7), est « non atteinte par la méthode tabulée ». La rubrique a deux onglets, « Calcul » et « Tableaux » (§ 2.2) ;
les anciens écrans « Enrobage » et « Données béton » (masqués) lisent les mêmes tableaux et renvoient à
« Bibliothèque › Enrobages › Calcul ». Preuves : `tests/test_bibliotheque_moteur.py`, `tests/test_api_web_bibliotheque.py`
(chaque case affichée = le moteur, la case allumée, l'adresse relue), harnais filtre « bibliotheque ».

## 5. Les lots

| Lot | Contenu | Fermeture |
|---|---|---|
| **BIB1** moteur — **fait le 08/11/2026** | `beton/bibliotheque/` : `ancrages.py` (§8.4, §8.7), `enrobage.py` (4.4N, 4.3N, feu aux valeurs de la norme pour poutres et dalles, libellés belges), `aciers.py` (k, ε_uk, ductilité), les ε et f_cd des bétons dans `beton/donnees/ec2.py` | `tests/test_bibliotheque_moteur.py` (114 contrôles) : l'exemple p. 49 de la NIT, les tableaux 24 et 25 à 2 cm (Ø ≤ 32) et 5 cm (Ø 50), le tableau 20, le tableau 21 sur ses extrêmes, les déformations du tableau 3.1 ; 15 mutations ; aucun module existant touché |
| **BIB2** page — **fait le 08/11/2026** | `api_web/biblio_beton.py` (hors du registre des modules : action « consulter » du routeur, `consulter(rubrique, params, hote)`), `construire.page_bibliotheque` (page `bibliotheque-beton`), `css/bibliotheque.css`, `js/bibliotheque.js` : les cinq rubriques, tableau à sélection (chaque case porte les paramètres qu'un clic règle), barre de paramètres, bande de valeur, « Pourquoi ? » en volet, l'adresse qui garde l'état ; trois icônes Lucide ajoutées (book-open, anchor, flame) | `tests/test_api_web_bibliotheque.py` (101 contrôles : valeurs = moteurs, chaque case allume sa case, refus doux, échappement, routeur, page, mutations) ; `verifier_site.js` filtre « bibliotheque » (100 contrôles : 1 440 × 900, 1 180 × 820, 820 × 1 180, 390 × 844 — le tableau se voit d'un coup sur PC et tablette, la case allumée visible, les gestes, l'adresse rouverte) ; captures regardées ; **relecture de l'ingénieur attendue** |
| **BIB3** accueil — **fait le 09/11/2026** | `site/maquettes/construire.py` : `MASQUES` (donnees_beton, armatures, enrobage) ; `construire.py` : la carte « Bibliothèque Béton armé » (`.tile.biblio`, icône livre, rubriques en sous-titre) en tête du bloc Béton, le volet « Modules » (rail du PC, onglet du téléphone) commence par le groupe **Bibliothèques**, les masqués quittent les tuiles, le volet et « Nouvelle note » ; `js/modules.js` les garde avec `masque: true` (pages servies, notes existantes ouvrables, note neuve par l'adresse) ; `css/site.css` `.tile.biblio` | `tests/test_site_web.py` § 4 ter (carte en tête du bloc, rubriques, drapeau `masque`, volet, options, pages masquées toujours écrites) ; `verifier_site.js` accueil (registre visible, masqués absents, carte en tête, volet) et le repli par l'adresse pour un module masqué (`nouvelle()`, `neuve()` — le harnais SUIT : la fenêtre « Nouvelle note » ne les propose plus) ; captures accueil PC et téléphone regardées |
| **BIB4** la poutre — **fait le 09/11/2026** | `api_web/fiches.py` (relecture stricte, gestion, bloc, vues), `routeur` (actions « bibliotheque », « fiches »), `poutre.py` (`bibliotheque_prefill`, `bibliotheque_cibles`, `fiches`, `normaliser`), `biblio_beton._fiche` (texte, phrase, moteur), `ndc_pdf/kit.table` + `styles.justifications` (deux colonnes, suite), `note_poutre`, `construire` (bouton, menu ⋯, panneau = gabarit partagé), `js/biblio.js` (câblage partagé page + panneau), `js/biblio_note.js`, `css` ; dalle et prédalle : au prochain lot (le panneau et le script les attendent) | `tests/test_api_web_fiches.py` (44 contrôles : préremplissage, cibles possibles ou non, utiliser = modifier, fiches ajouter/monter/descendre/retirer, clé absente, 12 fiches abîmées refusées, échappement, PDF lu par pdftotext, brique table, 3 mutations) ; `test_site_web` (203 : bouton, panneau, identifiants, rien ailleurs) ; gel inchangé (239) ; notes garde-corps et soudures inchangées ; harnais « pilote » (panneau prérempli, utiliser, inclure, J1, PDF avec la fiche, .stab, retirer), « formats:poutre » (feuille plein écran au téléphone), « bibliotheque » ; captures regardées |
| **BIB5** acier — **fait la nuit du 09 au 10/11/2026** (révision § 2.3), sauf le bouton des calculs acier | `api_web/biblio_acier.py` (Profilés, Nuances, Boulons ; briques et relecture des paramètres de `biblio_beton`), `routeur.BIBLIOTHEQUES["acier"]`, `construire.BIBLIOTHEQUES["acier"]` (la carte de l'accueil, à gauche du bloc Acier, et le volet « Modules » sans rien recoder), `MASQUES` + `catalogue` (maquettes et site), `js/biblio.js` (une ligne, un en-tête, un bouton de la bande règlent ; une pastille règle le paramètre de son bloc ; un menu `data-garde` reste ouvert ; le tableau garde son défilement dans une rubrique), `css/bibliotheque.css` (le grand tableau qui défile dans son cadre ; et la hauteur de la grille, `flex: 1 1 0` + rangée `minmax(0, 1fr)` : sans elles la page des profilés faisait 6 868 px de haut, la bande hors de l'écran — muet pour le béton dont les tableaux tiennent) | `tests/test_api_web_bibliotheque_acier.py` (121 contrôles depuis la relecture — f_u et f_vw,d disent leur domaine à 40 < t ≤ 80 mm, aucun « < » nu dans un fragment — ; à la livraison 118 : le tableau = celui du module du catalogue cellule par cellule, les nombres = `profiles.json` relu à part, le plus léger, la fiche, chaque ligne / en-tête / pastille, refus doux, recherche échappée, nuances et boulons relus à part et recalculés, f_y de `bibliotheques.json` = `normes.NUANCES` (R2), routeur, page ; 6 mutations) ; `test_site_web` (254 : la carte Acier et ses rubriques, le catalogue masqué, la bibliothèque d'essai passée sur « Autres », le paquet seul rend chaque rubrique de chaque bibliothèque) ; `test_api_web_bibliotheque` (101, la bibliothèque « inconnue » est devenue « bois ») ; harnais « bibliotheque » (203 : l'accueil aux huit formats avec les deux cartes, l'Acier aux quatre formats, ses gestes au PC, au doigt au téléphone, une ancienne note du catalogue rouverte ; mutation de la hauteur : 3 rouges) ; captures regardées |

Ordre : BIB1 → BIB2 → BIB3 (court, pour que la bibliothèque se trouve) → BIB4 → BIB5, puis les autres modules de
calcul reçoivent le bouton à mesure qu'ils passent sur l'établi de référence (lots M).

## 6. Ce que l'ingénieur tranche

1. **Les écarts de l'écran Streamlit** (D1 à D4) ne passent pas dans la bibliothèque : elle suit la norme et la
   NIT 260. L'ancien module Enrobage reste servi tel quel, masqué, jusqu'à ce que la bibliothèque soit validée ; il
   sera alors retiré avec son moteur de parité (une seule vérité, R2).
2. **L'entrée bibliothèque dans le bloc de sa famille** (carte distincte) plutôt qu'une section « Bibliothèques »
   séparée en bas de page — les deux se font ; la première garde la bibliothèque à côté de ses calculs.
3. **Les anciennes pages** (Données béton, Tableau des armatures, Enrobage) restent servies pour les notes existantes
   jusqu'au retrait ; aucune note n'est convertie.
4. **α_cc** (0,85 ou 1,0 pour f_cd) et **γ_s du bureau** (1,5, à côté du 1,15 de l'EC2) : la bibliothèque montre les
   deux lectures, nommées ; le calcul des modules ne change pas.

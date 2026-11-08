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
  avec le libellé de ses rubriques (« Bétons · Aciers · Sections · Enrobages et feu · Ancrages »). Elle se distingue
  d'une tuile de calcul au premier regard, et reste à côté des calculs de sa famille.
- Le volet « Modules » du rail et le menu du téléphone listent **Bibliothèques** comme un groupe à part.
- Les pages des quatre anciens modules restent **servies** (une note existante s'ouvre encore), mais ne sont plus
  listées ; la liste nommée `MASQUES` de `FAMILLES` le dit et les tests la lisent. Leur retrait définitif se décidera
  quand la bibliothèque aura remplacé chaque fonction (§ 3).

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
| 4 | **Enrobages et feu** | c_min,dur : classe structurale S1-S6 × exposition (EC2, avec le libellé belge de la NIT 260) ; les modifications de classe structurale ; feu : distance à l'axe a et b_min par durée R, pour poutre (tableau 5.5) et dalle (tableau 5.8) — poteau et voile dits « non transcrits » tant que l'ingénieur n'a pas relu les tableaux 5.2a et 5.4 (§ 4, Feu) | élément, exposition, classe de béton, durée d'utilisation (50/100 ans), contrôle qualité, Ø barre, Ø étrier, Δc_dev, feu (R, b_min), coulé contre le sol → **c_nom** et sa chaîne (c_min,b, classe structurale calculée, c_min,dur, c_min, Δc_dev, a, c_feu) | **nouveau** `beton/bibliotheque/enrobage.py` aux valeurs de la norme (§ 4) ; l'ancien `beton/enrobage` garde l'écran Streamlit |
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

### 2.4 Le bouton « Bibliothèque » de chaque calcul

- Dans le ruban du haut (`construire.page_module`, à côté de « Ouvrir »), un bouton discret `data-action="bibliotheque"`
  (icône livre, libellé caché sous 1 100 px comme les autres) ; dans le menu ⋯ du téléphone, l'entrée « Bibliothèque ».
- Il ouvre un **grand panneau** (`#et-bibliotheque`, fenêtre modale comme `#et-rangement` mais de
  `min(1240px, 100vw − 32px)` et presque toute la hauteur ; au téléphone, une feuille plein écran) qui contient la
  bibliothèque de la famille du module, rendue par `consulter` avec les **paramètres connus de la note** : pour la
  poutre, le béton, les Ø des barres et de l'étrier, l'enrobage saisi, l'élément « poutre », b ; pour les dalles, idem
  avec « dalle ».
- Rien du calcul ne change tant qu'on ne clique pas « Utiliser cette valeur » (qui écrit, par exemple, c_nom dans le
  champ « c enrobage » de la poutre, par le chemin `modifier` ordinaire, et le dit par un message) ; « Fermer »
  revient à la note telle quelle.

### 2.5 « Inclure dans la note » et le PDF

- Deux insertions, depuis la bande de valeur du panneau : la **fiche justificative** (valeur retenue, paramètres,
  lignes de justification, formules, référence) et l'**extrait de tableau** (la ligne et la colonne de la case, la
  case marquée, avec l'en-tête et la source).
- Elles s'écrivent dans l'état de la note : `etat.fiches = [ {id, rubrique, sorte: "fiche" | "extrait", titre,
  params, valeur, lignes, formules, sources, extrait, date, moteur} ]` — **une copie figée** de tout ce qu'il faut
  pour relire la fiche sans le moteur (traçabilité des notes anciennes ; c'est une copie d'un run gelé au sens de R2,
  comme D-013). Le précédent est `justifs` des sections de la poutre : clé ajoutée aux clés admises, absente dans une
  note ancienne (qui reste identique à l'octet), relue strictement.
- Dans l'établi compact, sous l'arborescence, un bloc **« Justifications (n) »** de la note : une ligne par fiche
  (titre, valeur), menu ⋯ (retirer, monter, descendre) ; au centre, une vue « Justification » montre la fiche choisie.
- Dans la note PDF : après les planches, des pages A4 **paysage**, même charte (`ndc_pdf/styles.py`), « Justifications
  de la bibliothèque » : deux ou trois fiches par page (bloc clé-valeur, formules en `riche`, source en pied), un
  extrait de tableau par fiche quand il y en a un. Il faut pour cela **une brique `table` dans `ndc_pdf/kit.py`**,
  tirée des deux copies locales des notes garde-corps et soudures (qui la reprendront : une source).
- Le gabarit du PDF n'est pas refait : la page de garde, le cartouche et les planches restent.

## 3. Correspondance ancien → nouveau

| Aujourd'hui | Demain | Données, calculs, notes |
|---|---|---|
| Données béton · Classes de béton | Bibliothèque Béton armé · 1 Bétons | mêmes sources ; les ε et f_cd en plus |
| Données béton · Aciers d'armature | · 2 Aciers d'armature | mêmes sources ; k, ε_uk, ductilité, treillis en plus |
| Données béton · Tableau des armatures, **Tableau des armatures** (module) | · 3 Sections et masses (tableau + fiche rapide « A_s requis → dispositions ») | `beton/armatures` inchangé ; les notes « repères » existantes s'ouvrent sur l'ancienne page |
| Données béton · Enrobages, **Enrobage** (module) | · 4 Enrobages et feu (tableaux + fiche rapide = le calcul complet de c_nom) | nouveau moteur aux valeurs de la norme (§ 4) ; l'ancien module reste servi avec son moteur de parité Streamlit |
| Données béton · Ancrages (vide) | · 5 Ancrages et recouvrements (calculés) | nouveau moteur ; le format « tableaux du bureau » (`ancrages_recouvrements.json`) reste disponible pour un tableau maison |
| **Catalogue des profilés** | Bibliothèque Acier · Profilés laminés | module inchangé, gabarit de bibliothèque |
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

## 5. Les lots

| Lot | Contenu | Fermeture |
|---|---|---|
| **BIB1** moteur — **fait le 08/11/2026** | `beton/bibliotheque/` : `ancrages.py` (§8.4, §8.7), `enrobage.py` (4.4N, 4.3N, feu aux valeurs de la norme pour poutres et dalles, libellés belges), `aciers.py` (k, ε_uk, ductilité), les ε et f_cd des bétons dans `beton/donnees/ec2.py` | `tests/test_bibliotheque_moteur.py` (114 contrôles) : l'exemple p. 49 de la NIT, les tableaux 24 et 25 à 2 cm (Ø ≤ 32) et 5 cm (Ø 50), le tableau 20, le tableau 21 sur ses extrêmes, les déformations du tableau 3.1 ; 15 mutations ; aucun module existant touché |
| **BIB2** page | `api_web/biblio_beton.py`, `construire.page_bibliotheque`, `css/bibliotheque.css`, `js/bibliotheque.js` : les cinq rubriques, tableau à sélection, barre de paramètres, bande de valeur, « Pourquoi ? » | tests Python ; `verifier_site.js` filtre « bibliotheque » (chaque rubrique, chaque case, aucun défilement à 1 440 × 900 et 1 180 × 820, au doigt) ; captures PC et tablette regardées ; relecture de l'ingénieur |
| **BIB3** accueil | Calculs et bibliothèques, `MASQUES`, rail et menu du téléphone | tests `test_site_web`, harnais accueil ; captures |
| **BIB4** la poutre | bouton et panneau « Bibliothèque », préremplissage, « Utiliser cette valeur », « Inclure dans la note », `etat.fiches`, bloc « Justifications », vue, PDF (brique `table` du kit) ; puis dalle et prédalle (même établi) | gel de la poutre : regel accepté par la garde (champ `fiches` ajouté, note ancienne identique) ; tests note PDF ; harnais « pilote » ; captures |
| **BIB5** acier | Bibliothèque Acier (catalogue sous le gabarit), bouton « Bibliothèque » des calculs acier | tests catalogue inchangés (moteur intact) ; harnais ; captures |

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

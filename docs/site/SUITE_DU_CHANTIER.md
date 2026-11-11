# La suite du chantier du site — plan priorisé

*Établi le 09/11/2026 au soir, à la demande de l'ingénieur (« dis-moi c'est quoi la suite… je vais te lancer pendant
la nuit »), à partir d'une relecture de `docs/site/REFONTE_ETABLI.md`, `BIBLIOTHEQUES.md`, `DALLES_FUSION.md`,
`DEFAUTS_STREAMLIT.md`, des docs des moteurs et de l'historique git ; les points de sécurité ont été revérifiés dans le
code. Ce fichier est un PLAN : il se met à jour à chaque lot (une ligne faite se raye avec son commit), il ne remplace
ni les docs des modules ni les tests.*

## Décisions de l'ingénieur du 09/11/2026 (soir, réponse à ce plan)

- **S1** (cisaillement de la poutre au lit 1) : on NE CHANGE PAS. C'est la pratique du bureau (hauteur de la barre moins
  l'enrobage). Gardé en REMARQUE, à revoir peut-être plus tard ; pas de message rouge.
- **S2** (choix de profilé : V–M, flambement) : plus tard.
- **S3** (renfort de la direction secondaire) : corriger — il se pose sur sa nappe, comme la base.
- **Nappes empilées** : plus d'option. Les nappes se posent TOUJOURS l'une sur l'autre, anciennes notes et fichiers
  Streamlit importés compris (la comparaison avec les anciennes valeurs Streamlit n'a plus cours).
- **S4** : la note PDF de la dalle se fait SUR LE SITE et dit les chiffres du site.
- **Notes PDF** : les notes PDF qui existent (Streamlit) passent sur le site pour tous les modules qui en ont une.
- **Accueil** : la carte de la bibliothèque devient étroite et haute (deux rangées), les modules de la famille à sa
  droite.
- Les autres questions (§ 4) restent en suspens.

## Fait la nuit du 09 au 10/11/2026 (intégration, branche `integration-nuit-0911`, sur 55f83c0)

- ~~S3 et « plus d'option » des nappes~~ : 7d7da99 (nappes toujours l'une sur l'autre, anciennes notes et imports Streamlit
  compris, avis d'ouverture ; renfort secondaire sur sa nappe), regel de la photo « dalle » à part : 1264711 (aucun
  chiffre gelé ne bouge), relecture : 0641c23.
- ~~S4, la note PDF de la dalle sur le site, aux chiffres du site~~ : 53be686 (api_web/note_dalle.py), branchée par
  l'intégration f319a7a.
- ~~Les notes PDF de Streamlit sur le site~~ : garde-corps, soudures, charges sur plancher fee5966 (relecture 10fdbb7) ;
  assemblages 397aef7 (relecture 70d3f23) ; branchées par l'intégration f319a7a (bouton sur TOUTE page, compacte ou
  non ; paquet de la note relevé sur une note riche de chaque module ; numpy avant la note) et contrôlées dans le vrai
  navigateur aux trois formats ab0c057 (harnais, filtre « note_pdf »).
- ~~Accueil : la bibliothèque étroite et haute, les modules à sa droite~~ : d5bbf71, relecture 030c82d.
- Site reconstruit (pages, paquets du moteur et de la note) : f93bdb8. Référence (gel) : 239 OK, seul regel celui de la
  photo « dalle » (1264711, raison : décision de l'ingénieur du 09/11).

## Fait ensuite, deuxième vague de la nuit (intégration, même branche `integration-nuit-0911`, sur 915cba7)

- ~~S5, la nappe secondaire sur un renfort principal plus épais~~ : c33dfd1 (la couche secondaire, base et renforts S3,
  repose sur la couche principale la plus HAUTE de son lit de contact : `api_web/dalle_positions.lit_contact`, axe
  saisi jamais déplacé ; moteurs `beton/` non touchés), relecture e81dda3 (une couche principale AUTOMATIQUE est
  toujours du lit : sans elle, sous une base principale saisie plus bas que son axe automatique, le renfort sortait du
  lit et la nappe le traversait encore, A_s,req 13,4 % trop bas). Mesures, A_s,req secondaire avant → après : 3 Ø12 sur
  treillis Ø10, 430,1 → 459,8 mm² (+6,9 %) ; 3 Ø25, 430,1 → 493,8 (+14,8 %) ; CSP h 25 avec 4 Ø20, 577,2 → 607,2
  (+5,2 %, la nappe passe NON CONFORME) ; deux renforts +7,1 % ; prédalle, faces supérieures +6,1 %. A_s,req ne baisse
  jamais (513 couples base × renfort ; 3 000 dalles aléatoires de la relecture) et aucune face principale ne bouge ;
  aucun chiffre gelé ne bouge (239 OK, sans regel). La relecture l'a dit « bloquant » pour un défaut OUVERT et
  PRÉEXISTANT (S6, § 1) ; intégré quand même : S6 date de FUS3/FUS4, S5 ne l'introduit pas et ne l'aggrave pas (chiffres
  principaux identiques avant et après, secondaire jamais plus bas) — S5 est strictement plus sûr. La prédalle aux
  faces inférieures n'est pas empilée (question 11, inchangée).
- ~~Notes PDF : le garde-corps qui refusait sa note, l'indice à deux sources~~ : f9e911a (la grille de 240 états du
  garde-corps figée AVANT de toucher au générateur), df6a8ea (l'identification imprimée est TOUJOURS celle de la note du
  site, six modules ; une PAGE DE SUITE pour le garde-corps qui déborde), regel distinct des photos « soudures » et
  « plancher » fb7508a (les trois champs d'identification quittent l'écran ; aucun chiffre ne bouge), relecture 4431348
  (la page de suite passe enfin par les conditions du navigateur : relevé du paquet de la note et sonde). Les 154 notes
  qui tenaient en deux pages donnent les mêmes octets PDF ; les 86 refusées sortent en trois pages, rien de perdu.
- ~~Bibliothèque Acier (BIB5)~~ : 6ce57e7 (rubriques Profilés, Nuances, Boulons ; aucune valeur recopiée ; le Catalogue
  des profilés est masqué de l'accueil, sa page et ses notes restent servies), relecture b60804d (Nuances, 40 < t ≤ 80 mm :
  f_u et f_vw,d disent leur domaine t ≤ 40 mm — f_vw,d s'affichait 8,5 % trop haut pour une paroi S355 épaisse, sans le
  dire ; « < » échappé).
- Site reconstruit (pages dont `bibliotheque-acier.html`, paquets du moteur et de la note) : 6ae211b. Référence (gel) :
  239 OK ; seul regel de la vague : fb7508a (identification, aucun chiffre). Sur cet état : `python3.12 lancer_tests.py
  --tout` 117/117 suites vertes, 9 034 OK, 0 échec (589 s sur 4 processus) ; harnais COMPLET sans filtre (port 8771)
  2 317 OK, 0 KO (992 s), après les filtres touchés (nappes, module:dalle, formats:dalle, module:predalle, note_pdf,
  telecharger, bibliotheque, pilote, accueil : 624 OK, 0 KO). Captures regardées : accueil PC, tablette, téléphone
  avec les deux bibliothèques ; Bibliothèque Acier (profilés PC et téléphone, nuances tablette) ; coupe S5 PC et
  téléphone (nappe X au contact au-dessus du renfort Y Ø20, « sur renfort Y ») ; première page de deux garde-corps
  autrefois refusés (U + cumul + montant soudé, ELS ; panneau L + cumul + platine, Eurocode) et leur page 3, et la
  note de trois garde-corps téléchargée dans le navigateur ; une note de soudures d'indice C (avant : « indice 0 »).

## Fait ensuite, le lot des petits chantiers (même branche `integration-nuit-0911`, sur 06299b2, 10/11/2026)

Demande de l'ingénieur : « fais-les tous en un seul lot ». Un commit par chantier (ou groupe), chacun avec son test
vu rouge sans sa correction ; moteurs et générateurs de note non touchés.

- ~~1 à 4 — ouvrir un calcul n'écrit rien (TOUS les modules) ; rouvrir une note sans la modifier ne change pas sa date ;
  l'indice réel au lieu de « Brouillon » (« Non enregistrée » pour la provisoire) ; `?note=` vers une note absente~~ :
  eb0a647 (carte de l'accueil = lien ; « Enregistrer comme note… » sur toute page, aux trois formats ; « Nouvelle note »
  reste le chemin nommé ; une note absente le dit avec « Importer .stab », rien n'est créé ; filtre « ouverture » du
  harnais). Décision écrite : `REFONTE_ETABLI.md`, révision du 10/11/2026.
- ~~5 à 7 — dalle : nom de plus de 60 caractères coupé à la relecture avec un avis ; la dalle CSP dit ce qui n'est pas
  vérifié (flèche, fissuration), à l'écran et dans les observations de la note PDF ; l'avis d'ouverture d'une ancienne
  dalle vu au téléphone dès l'ouverture~~ : 410a00e, regel de la photo « dalle » b5581c1 (texte du périmètre, aucun
  chiffre).
- ~~8 — accueil de 701 à 1 360 px : la carte de la bibliothèque sur deux rangées, l'icône au-dessus du titre~~ : eeb29f5
  (captures 820, 1 024, 1 180, 1 280, 1 440 regardées).
- ~~9 — 44 px au doigt, tablette et téléphone~~ : bc112de (le harnais « formats » a trouvé trente rouges au premier
  passage — segments pleins coupés, nom de la dalle à la tablette paysage, sélecteur hors écran, « Ouvrir » à 42 px —
  corrigés avant le commit ; voir son message).
- ~~10 — `X-Robots-Tag: noindex, nofollow` sur tout le site~~ : 91378c3 (`_headers` ; `serveur_local.py` l'applique —
  il lit `_headers` au démarrage : un serveur local déjà lancé doit être relancé pour le servir).
- 11 — rangement : ~~les styles de l'ancienne arborescence~~ (e043d15 : `.et-pt…`, `.et-arbre-h` retirés de
  `css/etabli.css` et `css/modules/dalle.css` ; vérifié d'abord : les établis compacts ne la rendent plus ; le retrait
  avait emporté une règle vivante — « + Lit », « + Section » à 26 px au téléphone —, le harnais « formats » l'a trouvé,
  rétablie et gardée par un test : 950aefc) ;
  ~~la précision sur les tubes formés à froid~~ (daa5bcf : `docs/acier/PROFILES.md` § 7, la largeur plate
  (b − 2·r_o)/t, plus petite que (b − 3t)/t de 1 à 3 sur les 233 tubes EN 10219 ; 7, 9, 29 et 27 tubes changent de classe
  en S235, S275, S355, S460 ; la doc et le test se tiennent). NON FAIT, faute de preuve ou de cible : `renommer()`,
  `[data-renommer]`, `etabli_compact.arbre(compact=False)` et le repli de la poutre de `js/compact.js` RESTENT — le module
  Prédalles (enregistré, servi par le routeur, gelé, testé) rend encore l'ancienne arborescence ; « la copie en double
  dans le code de la dalle » et « le test manquant des bulles d'aide » sont INTROUVABLES dans `docs/site/*.md` (ni ces
  mots, ni l'historique git, ni les autres copies du dépôt) : rien n'a été inventé (question 14).
- ~~12 — pied de poteau, « Ancrages … information 108 % » sous « Conforme »~~ : 2ca6264 et 5a958dc, regels de la
  photo « assemblages » 3283eff et c497ba2. Ce n'est PAS un défaut de calcul ni de verdict (§ 1, examiné) : c'est
  le périmètre « EC3 seul ». Une ligne d'information ne porte plus de taux (« hors verdict », gris, sans jauge) ; le béton
  d'ancrage dit « hors périmètre » (le mot plus long, « hors périmètre « EC3 seul » », se coupait au PC : vu sur la
  capture) ; le chiffre reste au dixième dans la fenêtre de la pièce et dans la phrase d'information (« à 107,7 % — non
  satisfait »). Captures du cas à 454 % regardées au PC, à la tablette et au téléphone.
- Site reconstruit (pages, paquet du moteur) : 42f4df2. Référence (gel) : 239 OK ; regels du lot : b5581c1
  (dalle), 3283eff et c497ba2 (assemblages), affichage seul, aucun chiffre. Suites ciblées (sur le dernier état des sources) : test_site_web 274,
  test_api_web_modules 363, test_api_web_dalle 136, test_api_web_dalles_fusion 78, test_note_dalle 43,
  test_etabli_compact 26, test_api_web_poutre 111, test_api_web_predalle 103, test_api_web_assemblages 160,
  test_api_web_assemblages_pied 93, test_profiles_catalogue 29, test_reference_refonte 239 — 0 échec. Harnais (port
  8771, filtres) : pilote 62, ouverture 92, enregistrement 130, bibliotheque 215, module:dalle 57, module:assemblages
  25, telecharger 24, note_pdf 104, formats 1 614 OK et 6 KO (les « + Lit » à 26 px, ci-dessus : corrigés, puis
  formats:poutre, formats:dalle, formats:predalle, module:assemblages et pilote rejoués sur la construction finale,
  0 KO). `lancer_tests.py --tout` et le harnais complet ne sont PAS rejoués (le coordinateur). Le serveur local du
  port 8771 n'envoie `X-Robots-Tag` qu'une fois relancé (il lit `_headers` au démarrage).
- Relecture adverse du lot (10/11/2026) : suites ciblées et mutations annoncées rejouées (toutes rouges sans leur
  correction, vertes avec ; les comptes du lot retrouvés) ; parcours propres (un établi ordinaire ouvert depuis sa
  carte, modifié, quitté par l'accueil, rouvert, rangé : la note est là, la valeur aussi ; `?note=` absent ; une dalle
  au nom de 72 caractères ; l'accueil à 820, 1 024, 1 280, 1 440 et au téléphone). Deux défauts d'AFFICHAGE trouvés et
  corrigés, aucun chiffre touché : ~~l'en-tête « JUSTIFICATIONS (n) » des fiches de la poutre sans style (parti avec
  l'ancienne arborescence, e043d15)~~ : 1b83725 ; ~~à la tablette, le nom de la note serré à 110 px (1 px pour la
  poutre à 1 180 px)~~ : 3a8277a (REFONTE_ETABLI.md, révision « relecture »). Restent OUVERTS, préexistants :
  une saisie suivie AUSSITÔT d'un départ de la page (sans Entrée, sans attendre le calcul) est perdue (mesuré sur les
  soudures) — l'écriture suit la réponse du moteur, avant comme après le lot, et rien ne garde la sortie ; au PC à
  1 280 px, la poutre coupe son nom (103 px vus pour 124 ; 110 avant le lot) ; à la tablette, le lien « maison » du
  fil fait 14 px de haut (le rail porte la même cible à 44).

## Fait ensuite, le module Cornières refondu (même branche `integration-nuit-0911`, sur 030757a, 10/11/2026)

Demande de l'ingénieur (10/11) : reprendre son classeur Excel « Dimensionnement cornières » comme référence, les
colonnes organisées comme la poutre (données, dessins, résultats), la note PDF, la cornière vers le bas (son classeur)
ET vers le haut (modèle mécanique reconstruit, pas un simple retournement), et ses jeux par trois positions de la
charge. Décisions écrites AVANT le code : `docs/acier/CORNIERE_DECISIONS.md` (221c9ed, révisée f286a79 : l'enveloppe des
jeux, l'idée de l'ingénieur généralisée — charge à X ± dx, ancrage à Y ± dy, neuf combinaisons, chaque grandeur garde
la plus défavorable et dit laquelle gouverne ; quatre vues utiles, la vue « section seule » supprimée). Quatre lots en
chaîne (moteur → schémas → écran ‖ note → intégration), puis deux relectures adverses ; l'ancien modèle « lame »
(`acier/corniere` d'avant, l'écran Streamlit) n'est plus reproduit par le site (`DEFAUTS_STREAMLIT.md`, lignes Cornières).

- ~~Moteur~~ : 7524b77 — `acier/corniere/{entree,statique,moteur}.py` sur la feuille P8 (deux configurations A/B, deux
  modes historique/équilibre, enveloppe des 1/3/9 combinaisons, S1–S4, maxima du congé, section nette en B, ancrages ELS
  et ELU séparés, L_eff = min(L ; a + 2·Xe), diagrammes continus, formules avec les cellules du classeur, comparaison des
  modes, garde P1). Le cas de contrôle (L 120×120×15, S355, 42,76 kN, X = 75, Y = 85) rend M1…M4 et 88,8621 % en section 3
  à 1e-9 ; les 93 formules de P8 codées telles quelles dans le test face au moteur sur 240 saisies.
- ~~Schémas~~ : dcffa9e (kit de dessin : `data-sel`, styles) et 5f76a9e — `acier/corniere/schemas.py` : coupe cotée à
  l'échelle (trois positions de la charge, ancrage ± dy, pivot ou contact, bras du mode, sections 1–4), moments (trois
  courbes et l'enveloppe par aile), contraintes (t(s), σ_éq, taux, le maximum réel dans le congé repéré), élévation
  (ancrages, efforts, largeur d'appui, L_eff) ; `api_web/dessin_corniere.py` en couche mince.
- ~~Écran~~ : e6c3894 — `api_web/corniere.py` réécrit sur l'établi compact (état version 2 `{"cornieres", "actif"}`,
  quatre rubriques, chaque champ une pièce du dessin, verdict et trois blocs thématiques Sections / Ancrages /
  Comparaison, cinq fenêtres de calcul depuis les seuls termes du moteur, sélecteur des 233 cornières avec l'optimale,
  note v1 relue avec un avis) ; regel distinct fb5141e (raison dans les décisions § 7) ; filtre « corniere » du harnais.
- ~~Note PDF~~ : 68e4795 — `acier/corniere/{note,formules,methode}.py`, `api_web/note_corniere.py` : garde du site puis
  deux pages par cornière (calcul avec la coupe cotée, les quatre sections, les ancrages ELS/ELU, le récapitulatif ;
  détails et méthode avec les trois vues), page de suite au besoin ; parité pdftotext ↔ moteur, 1 017 formules
  ré-évaluées.
- ~~Intégration~~ : 9a34095 (`NOTE_PDF`, note riche, sept modules comptés, le harnais lit η à la précision du chip),
  0216684 (générés).
- ~~Relecture adverse de la mécanique~~ (recalcul indépendant : 3 971 égalités à 1e-9, enveloppe monotone sur 8
  statiques × 7 pas, catalogue 233 × A/B × 2 modes sans plantage, garde à Y = 43,547 mm, note sur quatre états neufs,
  mutations rejouées physiquement) : 1f5eddc (la parité de la note lit la ligne du bandeau — la mutation « η du bandeau
  pris sur S1 » survivait). Ses deux points ouverts corrigés ensuite : d78b6d6 (les textes du moteur à la virgule ;
  plus d'arrachement vers le bord en B, où l'effort du boulon va vers l'angle) et regel c5d2bee.
- ~~Relecture adverse de l'écran~~ (harnais six filtres, mesures dans les SVG, captures huit cas × quatre formats, clair et
  sombre) : 4c3b78c (le titre d'un bloc n'est plus mangé par un état d'en-tête trop long ; le nom du mode coupé) et
  regel fb049f5. Interrompue par un redémarrage du conteneur à la dernière étape ; son travail (modifications non
  commitées dans son arbre) a été relu et commité par le coordinateur.
- Dates : 8145944 (les lots sont du 10/11, pas du 11/11).
- Reste à confirmer par l'ingénieur à l'écran : dx = 20 et dy = 10 mm par défaut, R_ELU = 1,5·R comme valeur initiale,
  contact de compression au talon (y_c = 0). À trancher (CORNIERE_MOTEUR.md § 8) : en B le moment constant vaut
  R·(X′ − e/2) (les décisions écrivent R·(X′ − u_V), égal au défaut u_V = e/2) ; les maxima du congé restent hors verdict
  en historique (sinon le cas de contrôle rend 90,15 % et non 88,86 %) ; S420 absent des nuances du site ; une note v1
  relue pose ses défauts au prorata des ailes.
- Ouverts, affichage : les textes secondaires des diagrammes sous 13 px (graduations 10 px au PC, 7 à 9 px à la
  tablette portrait, le zoom de la toile les rattrape) ; le chip du verdict arrondit η à l'unité quand le bloc et la note
  donnent la décimale (brique partagée, tous les modules) ; dans la vue Contraintes, la 2ᵉ ligne du bloc du maximum peut
  toucher la ligne de base de « S1 : … ».
- Non fait, prévu par la demande : l'étape 4 (cornière longue, plusieurs ancrages, charges localisées, répartition
  justifiée, diagramme longitudinal), le maillage coloré (contraintes, déformée) et les cornières PRS (architecture
  prévue : `section()` isole la géométrie).
- Vérification finale (coordinateur) : site reconstruit ; `lancer_tests.py --tout` 119/119 suites, 9 254 OK, 0 échec ;
  harnais complet (port 8771) 2 588 OK et 1 KO, CLASSÉ « le harnais a trouvé » : le contrôle « l'état verrouillé des
  briques » charge `api_web.commun` avec le python3 du SYSTÈME (3.11) et le lot écran y avait écrit une f-string à
  antislash dans l'expression (Python ≥ 3.12 seulement ; Pyodide 3.13 et les suites 3.12 ne la voyaient pas) —
  corrigé 42ad8cc (sortie HTML identique à l'octet près, verrou dans `test_site_web.py` : 277 OK), puis formats:poutre
  128 OK et formats:corniere 117 OK rejoués ; suites touchées rejouées (commun 121, api_web_corniere 137, référence 244,
  modules 364). Captures regardées : coupe A et B au PC, vues Moments et Contraintes, coupe B au téléphone, note PDF
  (5 pages). Générés : 15fe358, e0dea86.

## Fait ensuite, la révision § 9 du module Cornières (même branche `integration-nuit-0911`, sur 703092f, 10/11/2026 au soir)

Demande de l'ingénieur (10/11 au soir, « en une seule intervention ») : la page 2 de la note en trois colonnes
graphiques (élévation et largeur efficace ; les moments dessinés SUR la cornière avec les trois courbes et l'enveloppe ;
un maillage de von Mises coloré, vert → orange à 100 % → rouge au-delà de 105 %), les paramètres des ancrages (nombre,
entraxe, type, profondeur, rondelles), le support (voile ou poutre avec ses limites, dessiné à l'échelle avec coupures),
les jeux par défaut dx = ±10 et dy = ±5, le choix ELS/ELU dans les hypothèses ; puis la page 1 resserrée (hypothèses
compactes, charge et ancrage en lignes) et la colonne de gauche de l'écran (volet de détails pour le catalogue, champs
visibles en saisie libre). Décisions écrites AVANT le code : `docs/acier/CORNIERE_DECISIONS.md` § 9 ; le détail dans
`docs/acier/CORNIERE_MOTEUR.md` § 14 (et § 10, 11, 12 révisés par les lots).

- ~~Moteur~~ : a41118d — `etat_limite` ELS/ELU (en ELU : f_y/γ_M0 avec γ_M0 = 1,0, τ = f_y/(√3·γ_M0), von Mises
  EN 1993-1-1 § 6.2.1(5), les chevilles reçoivent H/n et R/n, plus de 1,5 nulle part ; invariant prouvé taux(ELS, R) =
  taux(ELU, coef·R)), ancrages (entraxe vide = L/n, positions, refus aux minima de l'EN 1993-1-8 tableau 3.3, type,
  profondeur, rondelles ISO 7089 / 7093-1), support voile/poutre (refus si l'ancrage ou le contact sort de la poutre,
  distances aux bords), champ de von Mises (σ = M·ζ/W + N/A, τ = V/A, enveloppe maille par maille des combinaisons, le
  modèle du classeur et non des éléments finis), jeux 10/5 ; 115 contrôles, 35 mutations.
- ~~Schémas~~ : 923576f (kit : `style_prim`, six classes) et 82aae43 — coupe avec le support, l'ancrage (manchon,
  scellement hachuré, traversant avec écrou), la profondeur, les rondelles ; moments SUR la cornière (axes M = 0, trois
  courbes et enveloppe remplie, M1…M4) ; contraintes = maillage 48 × 8 par aile coloré par le taux, barre de couleur,
  maximum ◉ ; élévation avec les positions des ancrages (e₁ · s · … · e₁) et les coupures ; 193 contrôles.
- ~~Écran~~ : 5c70c4f, 2ea8561 (regel), c8a59aa, e37b5ac (γ_M0 « 1,0 », placeholders courts), e5031f3 (regel) — volet
  « Détails » du catalogue, ELS/ELU, entraxe, type, profondeur, rondelles, support ; 199 contrôles, 24 états, 23
  mutations ; harnais `module:corniere` 85 OK.
- ~~Note PDF~~ : 1e9df24, dc22572, 18ca8cc — page 2 DOMINÉE par les dessins (résumé de méthode en trois lignes, puis trois
  colonnes égales : élévation et largeur efficace, moments sur la cornière, contraintes de von Mises avec sa barre ;
  limites ≤ 3 lignes, références, cartouche ; 2 400–2 700 caractères hors cartouche au lieu de ≈ 6 500 ; jamais de page de
  suite), page 1 compacte (hypothèses en 7–8 lignes « libellé · valeur », charge et ancrage en 11 lignes, ELU : symboles,
  N_ELU/V_ELU seuls, conclusion « critère de von Mises, γ_M0 = 1,0 »), garde avec l'état limite ; le peintre du kit cale les
  textes à ancre `end` à droite (additif) ; 90 contrôles (17 états, 1 551 formules ré-évaluées, 10 mutations, sonde
  navigateur) ; intégration 5cb2f4e (légende des contraintes à la nouvelle échelle, le coin compté, trois cornières dans
  `construire.NOTES_RELEVE`).
- ~~Révision § 9.8~~ : f70b393 (regel f8ac262) — les remarques de l'ingénieur devant les captures : trois zéros à Y ± dy sur
  l'aile verticale (les courbes par position de l'ancrage), remplissage jusqu'à Mmax et Mmax écrit à gauche, échelle de
  couleurs bleu → orange → rouge, coin maillé (`maillage["coin"]`), barre graduée en N/mm² et en %, menu déroulant des
  233 cornières.
- ~~Révision § 9.9~~ : bb436b1 — la seconde remarque, devant les captures de § 9.8 : la barre de couleur arrêtée au maximum
  du maillage (sa valeur écrite ; 50/100/105 % seulement sous le maximum et sans superposition), les repères 1 à 4 hors
  du dessin (pointillé léger `sl` jusqu'au contour), le maillage deux fois plus fin (96 × 12 par aile, 12 × 12 dans le
  coin), le maillage dessiné DÉFORMÉ par la déformée élastique de la combinaison gouvernante calculée par le moteur
  (`maillage["deformee"]` : poutre, appuis du schéma statique, bloc rigide ; amplifiée à 10 % de haut, facteur écrit,
  contour d'origine en tirets) ; le modèle et ses limites écrits en § 9.9 des décisions.
- ~~Intégration~~ : 5cb2f4e, db4c08e (lot note : légende à la nouvelle échelle, coin compté, trois cornières dans la note
  riche, site reconstruit) ; après § 9.9 : 9fe3d3a (site reconstruit, photo de référence regelée — la légende de la vue des
  contraintes et les nombres du dessin, affichage seul). Sur l'arbre réuni : moteur 118, schémas 203, note 90, kit 52,
  module web 201, référence 21 OK ; harnais `module:corniere` 85, `formats:corniere` 117, `note_pdf` 119, `pilote` 63 OK,
  0 KO ; suite complète 119/119 suites vertes, 9 471 OK, 0 échec (706 s sur 4 processus).
- À confirmer par l'ingénieur à l'écran : γ_M0 = 1,0 (ANB) ; la profondeur 100 mm et une rondelle par défaut ; R_ELU =
  1,5·R comme valeur initiale en ELS ; les points de la section précédente (contact au talon, S420, moment en B).
- Non fait, et pourquoi : la déformée n'était d'abord pas faite (« seulement si disponible et justifiée ») — demandée
  expressément ensuite, elle l'est, avec son modèle écrit (§ 9.9 ; pas de critère de flèche) ; le dimensionnement automatique et
  les autres modules (hors demande) ; les ancrages restent sur une ligne horizontale (l'étape 4 — cornière longue,
  plusieurs lignes, charges localisées — reste à faire).

## Fait ensuite, le lot S4 : une application, un moteur (même branche `integration-nuit-0911`, sur 2fd6dd6, 10/11/2026)

Demande de l'ingénieur (10/11, après la mesure 7c38a02 qui montrait un redémarrage complet de Python à chaque écran) :
une application unique, navigation sans rechargement, chaque module dans ses fichiers, un moteur Python partagé qui
reste actif, modules chargés à la demande, ouverture préparée à l'approche d'une tuile, l'interface tout de suite sans
écran de chargement plein écran, les calculs inchangés et prouvés identiques. Conception écrite AVANT le code :
`REFONTE_ETABLI.md`, S4-D1 à S4-D10 (007c634) et C1 à C14 (ba7dcc6) ; la réalisation et ses écarts : même fichier,
« Le lot S4 réalisé ».

- ~~Registre paresseux~~ : 007c634 (`api_web/registre.module(clé)` ne lit que son fichier, une fois ; mêmes refus que
  `cles()`).
- ~~Construction, aperçus, version du site~~ (A1), ~~moteur partagé~~ (A2), ~~application et écrans montables~~ (A3),
  ~~harnais~~ (A4a), ~~filtre « demarrage »~~ (A4b) : réunis 0ea2647, 0911ba6, 0311c29, a2cd22c, 959ff87, 62f895b ;
  harnais à l'intégration bbbbcf4, a02512e ; une adresse en « .html » dans l'application 60a13a8.
- ~~Revue adverse~~ (5 angles, 40 constats, 38 confirmés, aucun écart de calcul) et ses corrections en deux vagues :
  moteur 6204e69, c077a9b ; navigation, fusion d8a11e5 ; écrans, fusion e5c43cc ; preuves, fusion 0c521f2.
- Mesure « après » : `mesures/2026-11-10_demarrage_apres.md`.
- Non fait, et pourquoi : la recette Safari (iPad, iPhone) se fait à la main, elle ne se prouve pas dans Chromium ;
  annuler/rétablir ne restaure pas ses piles si le moteur refuse l'état rendu (hors de la revue) ; l'instantané mémoire
  de Pyodide reste hors périmètre (D8).
- Après la mise en ligne (10/11 au soir) : une tablette Android restait sur « Chargement… » sans un mot (l'application
  n'avait monté aucun écran ; le téléphone, lui, marchait). Cause NON établie à ce jour (session de connexion expirée ?
  fichiers de deux versions pendant le déploiement ?). Ajouté : le FILET du démarrage — chaque page porte un message,
  montré par une animation CSS au bout de 12 s si js/app.js n'a monté aucun écran (body[data-montage] le cache), avec
  « Recharger la page » ; sans script ni fichier à charger. Gardé par `tests/test_site_web.py` et `demarrage:secours`
  (trois formats ; deux mutations rougissent : la règle qui le cache retirée, l'animation retirée).

## Fait ensuite, le lot Q : quatre modules refaits sur la norme — trois livraisons (même branche, sur 7bc200d, 10 et 11/11/2026)

Demande de l'ingénieur du 10/11/2026 : « fais bois, feu, taux, maturité ; vérifie les formules, tout n'était pas juste ;
pas grave si on n'a pas exactement la même chose que sur Streamlit ». La conception est écrite AVANT le code, sources
croisées : `docs/site/LOT_Q_QUATRE_MODULES.md` (c6ddb33, e5c30bd, 5b15dfc, d53f89c). Un module à la fois ; cette
première livraison en porte deux, et ce qui les relie aux poutres.

- ~~Poutres bois refait sur l'Eurocode 5~~ (flexion, cisaillement, appui, déversement, flèche instantanée et finale ;
  deux combinaisons ELU avec leur k_mod) : moteur cafb4b8, écran 5f5d319, 63e87e2, gel ae9b130, 0010900, docs 6007417.
  99 + 112 contrôles, 77 mutations. Trois cas recalculés à la main par l'intégrateur (note neuve, G seule qui gouverne,
  déversement) : mêmes nombres.
- ~~Taux d'armature refait~~ (un quantitatif qui ne sous-estime pas : barres bornées aux abouts, étrier de fermeture,
  talons, recouvrements, bordereau par repère et par diamètre, CSV) : moteur de80274, dessins 68b5eda, écran 1e51a8e,
  4b2c8e6, gel 9f87a02, docs f4253d2. 30 + 68 contrôles, 63 mutations. La poutre type : 85,8 kg/m³ (33 cadres de
  1,72 m, 88,25 kg, + 5 %), recalculée à la main.
- ~~Poutres béton armé : l'option « Taux d'armature »~~ (éteinte par défaut ; allumée, une longueur par section ; le
  taux ne va dans la note PDF que sur demande) : 2f4ee30, 1268b12, ec06859. Parité avec l'écran Streamlit rejoué sur
  99 sections.
- ~~« Détailler dans Taux d'armature »~~ (une poutre dimensionnée part dans le module détaillé, préremplie) : 4897194,
  494d9ad, contrôles 4002d34, 32a5c31, 938a664. Mécanique générale : `docs/site/MODULES_SITE.md` § 4 sexies.

Deuxième livraison, le même soir :

- ~~« Enrobages » dans la Bibliothèque, en deux onglets~~ (« Calcul » : le dessin du détail et la feuille terme par
  terme, durabilité puis feu ; « Tableaux » : le tableau entier de l'EN 1992-1-2 § 5, la case de la situation allumée ;
  poteaux au tableau 5.2a par défaut, formule (5.7) au choix avec la valeur du tableau en comparaison) : moteur e19bf47,
  anciens écrans corrigés c218093, état de vérification 375be08, gels b841f15, écran 625de00, harnais 74b05df, docs
  f42c55a. 231 + 135 contrôles ; les distances fausses d'hier (colonne μ_fi décalée des poteaux, étrier retiré d'une
  dalle, R 240 inventé) sont refusées par des mutations. Trois cas recalculés à la main par l'intégrateur (poutre R 90,
  poteau R 90 au tableau puis à la formule, dalle R 120) : mêmes nombres.

Troisième livraison, le 11/11/2026 — **le lot Q est complet** :

- ~~Maturité du béton refait sur la norme~~ (une DATE par objectif, à l'heure près, avec sa variante prudente ; le
  durcissement lu sur le bon de livraison ou sur le ciment ; la loi de température belge par défaut ; le gel avant
  5 MPa ; l'essai sur éprouvettes et le calage de la courbe ; le sélecteur de classe ; plus de numpy) : moteur 2e8b941,
  dessins 407b5ca, page compacte 1f40ddb, écran fdc9700, feuille 64acd83, cas et gel d0606ab, aide de la loi 9c28f0f,
  docs c96bad7, relecture des captures 70c9ed8 et 8c4aa2b. 208 + 81 contrôles, 46 + 7 mutations, dix-huit
  contre-épreuves des retouches. Trois coulages recalculés par l'intégrateur par une AUTRE méthode que le moteur (intégration numérique au
  1/20 000 de jour et dichotomie, face à la forme fermée) : hiver à 3 °C en béton rapide, relevé avec gel en loi
  (B.10), essai sur cube calé — mêmes nombres. Les trois défauts non conservatifs de l'ancien écran sont refusés par des
  tests (le seuil à la température MOYENNE : annoncé à 4,8 j, atteint à 7,23 j ; le premier jour compté en entier ;
  l'essai lu comme une résistance caractéristique).
- **Un invariant de la conception s'est révélé faux, et c'est écrit** (`LOT_Q_QUATRE_MODULES.md` § 2.6,
  `docs/beton/MATURITE_MOTEUR.md` § 3) : « la loi belge jamais plus rapide que (B.10) sous 20 °C » ne tient que de 0 à
  12,2 °C ; au-delà le tableau belge dépasse (B.10) de 1,5 % au plus. L'aide de l'écran disait « la plus prudente par
  temps froid » : elle dit maintenant ce qui est vrai (plus prudente que (B.10) ; un ciment au laitier durcit plus
  lentement au froid que le tableau).
- **Un défaut de franchise trouvé en jouant un cas d'hiver réaliste**, corrigé dans le moteur : pour un objectif DATÉ,
  l'échéance affichée (« 21 MPa atteints J + 28,4 ») pouvait tomber après le relevé — donc reposer sur des températures
  prolongées — sans que rien le dise (seul le jour prévu était regardé). Toute échéance affichée, nominale ou prudente,
  est maintenant signalée ; deux contrôles et deux mutations le tiennent (13bdcdc).
- **Ce que l'œil voit, mesuré** : la seconde ligne d'un résultat ne se replie pas ; sa place, lue au navigateur aux six
  formats, est de 313 px au PC (59 caractères). Trois lignes du module y étaient ou pouvaient y être coupées (« Jours »
  à quatre sources, l'essai sur cube, le gel en variante prudente) : raccourcies, et tenues par un contrôle. « Gel avant
  5 MPa » dit la date du premier gel et sur quoi il est jugé (minimale horaire, ou moyenne du jour).
- ~~Harnais « modules »~~ : le premier nombre d'un cadre Données peut être dans une rubrique repliée — le harnais suit
  (16f2c48) ; `module:age` s'arrêtait à 18 contrôles, il en joue 26.

À voir avec l'ingénieur après essai : au téléphone, l'onglet « Calcul » de la rubrique « Enrobages » ramène la page sur
la ligne c_nom après chaque
réglage (la règle de la bibliothèque : la case allumée se voit) ; avec quinze paramètres au-dessus, une bande de
résultat fixe en bas d'écran et aucun saut serait peut-être plus commode. La feuille de la formule (5.7), dix-huit
termes, défile dans son cadre au PC et à la tablette (c_nom reste visible dans la bande).

**À confirmer par l'ingénieur** (rien de bloquant, chaque choix est écrit et modifiable dans l'écran) :

1. Bois — la neige est classée « court terme » (k_mod 0,9) ; le texte de l'annexe belge n'a pas pu être lu ; le choix
   plus sévère serait « moyen terme » (k_mod 0,8).
2. Bois — limites de flèche par défaut L/300 (instantanée) et L/250 (finale nette) : ce sont les fourchettes de
   l'EN 1995-1-1, tableau 7.2 ; l'annexe belge n'a pas été trouvée.
3. Bois — k_cr = 0,67 pour le lamellé-collé aussi (le plus sévère des deux lectures).
4. Feu — dix cases (R 180 des tableaux 5.5 et 5.6, REI 30 et REI 180 du 5.9) ne sont recoupées que par du code
   publié : à relire sur la norme papier.
5. Maturité — le coulage d'ouverture est un « béton à durcissement moyen » (s = 0,439 : J + 10,7 pour 70 % de f_ck à
   20 °C) et non le ciment N de l'EC2 (J + 6,5) : le béton courant ici (CEM III/A 42,5 N LA) est « moyen » sur les
   bons de livraison, et le défaut doit être du côté sûr.
6. Maturité — les six valeurs du tableau belge (NBN B 15-400, « âge fictif ») viennent de deux sources concordantes :
   à relire sur l'édition 2024. Le prolongement jusqu'à 0 à −10 °C et l'absence de bonus au-dessus de 20 °C sont des
   choix du module, dits à l'écran.
7. Maturité — les délais minimaux de décoffrage par type de coffrage et les durées de cure de la norme belge ne sont
   PAS dans le module (leurs valeurs n'ont pas pu être relues) : à ajouter si le bureau fournit le texte.
8. Maturité — pas encore de note PDF (le bouton reste « bientôt ») : le résultat se lit à l'écran et s'imprime.
9. Maturité — sans détail horaire, le gel est jugé sur la MOYENNE du jour (c'est dit sur la ligne) : une nuit peut geler
   sous une moyenne de + 1 °C. Faut-il une colonne « T min » dans le relevé saisi, pour juger le gel sur elle ?

## Fait ensuite, le catalogue des hourdis Ergon 2018 (même branche, sur d9fdb49, 11/11/2026)

Demande de l'ingénieur du 11/11/2026, avec le guide technique Ergon 2018 : « on peut prendre leur hourdis […] le MRd et
le VRd […] les dimensions […] ce sera Ergon, mais on mettra aussi des autres marques plus tard ». Référence :
`docs/beton/HOURDIS.md`.

- ~~Le catalogue transcrit~~ (7 profils SP 150 à SPZ 500, 53 variantes : M_Rd, V_Rd,c, M_Rd avec 50 mm de couche
  C30/37 ; dimensions, poids, section, mise en œuvre, coupes relevées) : f170478. Trois lectures du guide qui s'accordent
  (0 écart), trois erreurs du guide nommées, une empreinte qui garde les valeurs ; 87 contrôles, 17 mutations.
- ~~Charges sur plancher : les 53 types au menu~~, groupés par profil, valeurs reprises et modifiables ; P-17 révisée
  (M et V avec chape jugés séparément, épaisseur exigée) — Ergon ne publiant pas V_Rd,c avec chape, le tranchant reste
  jugé sur la valeur nue : 68ff13a. Le gel du plancher a été regelé avec sa raison (menu et vue Méthode ; aucun
  résultat changé).
- ~~La rubrique « Hourdis » de la Bibliothèque Béton armé~~ (coupe cotée, caractéristiques, tableau des variantes, la
  première qui suffit ★ pour M_Ed et V_Ed donnés, la Gamme et le profil le plus léger qui suffit) : 6522bc8, site
  630c92d ; harnais « bibliotheque » 263 contrôles aux quatre formats (la tablette avait rougi : le cadre défilait —
  d'où la coupe et les caractéristiques à gauche, le tableau à droite).
- ~~Les conditions du fabricant lues dans le catalogue~~ (feu, classe de conséquence, environnement, chevilles : elles
  étaient écrites dans l'écran, fausses pour la marque suivante) : 881b2aa, trois mutations qui rougissent.

**À confirmer par l'ingénieur** (rien de bloquant) :

1. Un cas neuf de Charges sur plancher démarre toujours sur la fiche générique « à renseigner » : faut-il qu'il démarre
   sur un type Ergon (lequel) ?
2. V_Rd,c avec couche collaborante n'est pas publié par Ergon : le tranchant est jugé sur la valeur nue (prudent). Si le
   bureau obtient la valeur du fabricant (« sur demande »), elle s'ajoute au catalogue et le moteur la prend.
3. M_cr, pression locale admissible sur la membrure et courbes de performance (charge utile selon la portée) ne sont pas
   publiés ou pas transcrits : la membrure reste « répartiteur exigé » ou « non vérifiée » (P-5). Les courbes sont-elles
   utiles (elles se transcriraient point par point) ?
4. La sélection de la bibliothèque prend M_Ed et V_Ed **par élément** (1,2 m), comme le guide : un moment par mètre se
   multiplie par 1,2. Faut-il une saisie par mètre en plus ?

## 0. « L'audit » : ce que c'est, ce qui en reste

- C'est l'étude INTERNE du 07/11/2026 sur l'établi (le rangement des calculs) et le démarrage
  (`REFONTE_ETABLI.md`, parties A « jury » et B « spécification », 13 lots). Ce n'est pas un contrôle réglementaire.
- Revue avec l'ingénieur le 08/11 (`REFONTE_ETABLI.md`, « Révision du 08/11/2026 ») : l'écran de la poutre est devenu
  le modèle, une grande partie du plan d'origine est abandonnée.
- Ses deux défauts de PERTE DE DONNÉES (échec d'enregistrement muet ; mémoire des notes écrasée) sont corrigés et figés
  par des tests.
- Il n'en reste qu'une OBLIGATION permanente : le GEL des calculs (`tests/test_reference_refonte.py`, 239 contrôles).
  Une retouche d'écran ne change pas un chiffre sans une raison écrite (`--geler --accepter="raison"`).
- Le reste (passer les autres modules sur l'établi compact, le démarrage, les chantiers) est de l'ergonomie souhaitée,
  pas une obligation.
- Le relevé de SÉCURITÉ qui compte vraiment est `DEFAUTS_STREAMLIT.md` (défauts des écrans Streamlit reproduits par
  les moteurs) : toujours valable.

## 1. D'abord : la sécurité des calculs (défauts non conservatifs, muets à l'écran)

| # | Défaut | Où | Taille | Décision ? |
|---|---|---|---|---|
| S1 | Poutre BA : le cisaillement prend d = h − min(lit 1 inf., lit 1 sup.), pas la hauteur utile des aciers tendus ; avec plusieurs lits, s_max et s_th sortent trop grands (+5,8 % sur le cas d'essai « audit C-1 ») | `beton/poutre/moteur.py:216` | — | **REMARQUE** — décision du 09/11 : on NE CHANGE PAS (pratique du bureau : hauteur de la barre moins l'enrobage), pas de message rouge ; à revoir peut-être plus tard |
| S2 | Choix de profilé : pas d'interaction V–M ; flambement sans moment et une seule courbe pour les deux axes : un poteau fléchi peut sortir « conforme » | `acier/choix_profil/moteur.py:315-373` | message : petit | **plus tard** (décision du 09/11) |
| ~~S3~~ | ~~Dalle, nappes empilées : un RENFORT de la direction secondaire reste au niveau de la base principale (sous sa propre nappe) : ≈ −2 % sur A_s,req mesuré, plus avec de gros Ø~~ | `api_web/dalle_positions.py` | — | **fait** : 7d7da99 (+2,0 à +6,4 % sur A_s,req, sens de la sécurité) |
| ~~S4~~ | ~~Note PDF de la dalle : elle se fait dans Streamlit, qui pose X et Y au même niveau sans le dire~~ | `api_web/note_dalle.py` | — | **fait** : 53be686 + f319a7a — la note se fait sur le site, aux axes du site, contrôlée contre son moteur à chaque note |
| ~~S5~~ | ~~Dalle (relecture du 09/11, PRÉEXISTANT depuis FUS4, ni introduit ni aggravé par la nuit) : la nappe SECONDAIRE (base, et renfort S3) se pose sur la BASE principale alors qu'elle croise un RENFORT principal plus épais, posé dans le lit de la base — les barres se traversent dans le calcul et la coupe, rien ne le dit : A_s,req secondaire sous-estimé de 6,9 % dès le mode par défaut « Treillis + renforts » (3 Ø12), 14,8 % avec un renfort Ø25~~ | `api_web/dalle_positions.axe_auto` | — | **fait** : c33dfd1 + relecture e81dda3 (la nappe secondaire repose sur la couche principale la plus haute de son lit de contact ; +5,2 à +14,8 % sur A_s,req secondaire, jamais en baisse ; aucun chiffre gelé ne bouge) |
| S6 | Dalle, mode par défaut « Treillis + renforts » (relecture de S5, 09/11, PRÉEXISTANT depuis FUS3/FUS4, ni introduit ni aggravé par S5) : un treillis soudé est UNE pièce, mais le calcul et la coupe montent ses fils X sur le renfort principal plus épais et laissent ses fils Y au parement — 2,0 à 3,0 cm d'écart d'axes, contre 1,0 pour des fils soudés. Si le treillis monte entier, A_s,req PRINCIPAL est sous-estimé de 2,9 à 4,6 % (renfort Ø12 : 611,7 au lieu de 638,9 mm² ; Ø25 : 631,0 au lieu de 660,0) ; si c'est le renfort qui se pose sur le treillis (2e lit), c'est son axe qui est trop bas. Aucune géométrie physique ne donne les axes du modèle dès que Ø renfort > Ø fil | `api_web/dalle_positions` (direction principale : `pour_moteur`, `annoter`, sélecteur, gel) | moyen, puis regel | **oui** : la pose des renforts par rapport à un treillis soudé (sous, dans ou sur le treillis) |

**Examiné le 10/11/2026, pas un défaut de calcul** : pied de poteau, périmètre « EC3 seul » (la valeur par défaut du module,
Lot B bis), « Ancrages … information 108 % » sous « Conforme ». Le béton d'ancrage EN 1992-4 y est calculé EN INFORMATION
et n'entre pas dans le verdict, comme dans l'écran Streamlit (`synthese.info`, `moteur.info_keys`) ; le moteur le dit
(« non satisfait — hors périmètre, en information »). Mesuré : l'exemple publié VAL-A, 107,7 % (interaction N/V) ;
hef 100 mm, N = −50 kN, Mx = 10 kNm, 454 % sous « vérifié » — et NON CONFORME au périmètre « EC3 + EC2-4 ». L'affichage
est corrigé (2ca6264) ; le choix du périmètre par défaut est une question (14).

Puis, déjà SIGNALÉS à l'écran mais en attente d'arbitrage : ~~anciennes notes de dalle et fichiers importés non empilés
(avertissement affiché)~~ (réglé : 7d7da99, toujours empilées, un avis le dit à l'ouverture) ; défauts Streamlit marqués en rouge — bois (G seul, k_cr, flèche finale, appui, EN 338), feu
des poteaux et voiles (ancien module Enrobage, a trop faible de 15 à 25 mm), taux d'armature, maturité (décoffrage trop
tôt), cornière (longitudinale = transversale), catalogue (PRS), sol (k), plancher, flambement, choix de profilé (D1, D2,
D3, D5, D7), prédalle (couches basses croisées). Règle : Streamlit d'abord, puis le site suit.

## 2. Ensuite : les fonctions demandées

- ~~Rejouer TOUTES les suites (`python3 lancer_tests.py --tout`) avant la prochaine publication~~ : fait le 09/11 sur
  l'intégration (f93bdb8) — 114/114 suites vertes, 8 863 OK, 0 échec (581 s sur 4 processus) ; harnais complet sans
  filtre : 2 234 OK, 0 KO (941 s). Refait après la deuxième vague (6ae211b) : 117/117 suites, 9 034 OK, 0 échec
  (589 s) ; harnais complet 2 317 OK, 0 KO (992 s). À refaire avant chaque publication qui en doute.
- Site privé : Cloudflare Access sur stab-methodes.be ET sur l'adresse .pages.dev (à faire par l'ingénieur dans
  Cloudflare ; le 07/11, .pages.dev répondait sans mot de passe).
- ~~Dalle : un nom de plus de 60 caractères fait refuser toute une ancienne note → le couper avec un message
  (petit)~~ : 410a00e (coupé à la relecture, l'avis le dit ; la saisie garde son refus).
- ~~Un lien `?note=` vers une note absente ouvre une note vide sans rien dire → message clair et « Importer .stab »
  (petit)~~ : eb0a647 (rien n'est créé ; une note d'un autre module s'ouvre par sa page).
- ~~Dalle coulée : dire ce qui n'est pas vérifié (flèche, fissuration), comme la prédalle (petit)~~ : 410a00e (même
  composant, mêmes places, et dans les observations de la note PDF) ; regel de la photo « dalle » b5581c1.
- Bibliothèque Béton armé dans la note de dalle (BIB4 bis : préremplissage, « Utiliser cette valeur », fiches) (moyen).
- Relecture de la Bibliothèque Béton armé par l'ingénieur, puis retrait de l'ancien module Enrobage (petit).
- Les 11 autres calculs sur l'établi compact, un module à la fois, relu par l'ingénieur à chaque fois : flambement,
  profil, bois, maturité ; puis plancher, sol, taux ; puis assemblages ; enfin garde-corps, soudures, cornière (gros).
- ~~Notes PDF des autres modules sur le site~~ — **sur le site** (le générateur validé de Streamlit, inchangé, aux chiffres
  du site) : poutre, dalle (CSP et prédalle), garde-corps, soudures, charges sur plancher, assemblages (toute la note :
  garde, récapitulatif, deux pages par assemblage). **Existe dans Streamlit, pas encore sur le site** : la rigidité du
  sol (`modules/rigidite_sol.py`, `_rapport_pdf` : reportlab + figures matplotlib, écrites DANS l'écran Streamlit — à
  extraire de l'écran, figures sans matplotlib ; non fait cette nuit). **N'existent pas dans Streamlit** : bois,
  maturité, taux d'armature, flambement, choix de profilé, cornière, catalogue — à CONCEVOIR avec le bureau (rien n'a
  été inventé cette nuit).
- Notes PDF, écarts relevés par la relecture du 09/11 : ~~le garde-corps REFUSE sa note quand le générateur validé
  déborde (86 des 240 états d'une grille)~~ (réglé : une page de suite, df6a8ea, relecture 4431348) ; ~~l'INDICE a deux
  sources pour les soudures et le plancher (le PDF écrivait « indice 0 » sous une note B)~~ (réglé : toujours celui de
  la note du site, six modules, df6a8ea ; regel fb7508a). Restent (à trancher, § 4) : arrondi d'un taux en bord de
  bande ; seuil d'attention à 85 % dans les notes des soudures et du plancher (et 70 % pour le tableau des cordons des
  soudures) contre 95 % à l'écran (mesuré : MODULES_SITE.md § 4 quinquies, DEFAUTS_STREAMLIT.md) ; quand la géométrie
  gouverne, la page d'un assemblage garde le η du moteur ; soudures : 2 états sur 56 d'une grille (C6 en quatre
  tronçons, Eurocode) restent refusés (ce générateur n'a pas de page de suite ; le message dit la vraie cause) ; dans
  une note de plusieurs garde-corps, « Suite p. 3 » renvoie à la 3e page DU garde-corps, pas du PDF (aucun numéro de
  page imprimé ; même convention que « détail p. 2 ») ; assemblages : les champs « Projet » et « Date » du cartouche d'un
  assemblage restent à l'écran du site et ne s'impriment jamais (ils servent encore à la « note texte » de l'écran).
- ~~Bibliothèque Acier (BIB5) : le catalogue sur le modèle de la Bibliothèque béton~~ : fait, 6ce57e7 + relecture
  b60804d. Reste : le bouton « Bibliothèque » sur « Choix de profilé » (son établi n'est pas compact : bouton et panneau
  hors de `compact()`, `bibliotheque_prefill` dans `api_web/profil.py` — BIBLIOTHEQUES.md) ; les profilés personnalisés
  (PRS), la comparaison et l'export texte ne sont que dans l'ancienne page du catalogue, masquée (les porter ou les
  laisser : § 4) ; au téléphone, le menu « Colonnes » (feuille fixe) couvre son propre bouton et le toucher qui le
  referme déclenche aussi la commande touchée (ergonomie, aucun chiffre) ; le texte des règles écrit en toutes lettres
  le 0,9 de F_t,Rd et le 235 de ε (les valeurs viennent du moteur, ces deux copies ne sont liées par aucun test) ; deux
  autres copies de f_y non liées (`choix_profil` FY_EPAIS, plancher FY_MATERIAU) ; `biblio_beton._champ` met des
  segments dans un `<label>` (un clic sur le libellé active le premier bouton ; corrigé côté acier seulement).
- Chantiers et dossiers (nouvel écran, MIGRATION du format de la mémoire des notes : rien sans l'accord) (gros).
- **Hourdis — les réflexions de l'ingénieur du 11/11/2026 (« pas à faire tout de suite »)**, avec l'avis donné le même
  jour ; ordre proposé : 1, 2, 3, puis 4 sur document du fabricant.
  1. *Les autres marques* (Prefer, et une autre à nommer) : conseillé. Un fichier et une ligne de `CATALOGUES` chacune,
     transcrites comme Ergon (trois lectures, empreinte, mesure datée) ; il faut leurs guides techniques en PDF, édition
     la plus récente. Les conditions propres à chaque marque (béton, feu, classe de conséquence) restent affichées : une
     comparaison entre marques se lit avec elles (petit à moyen par marque).
  2. *Les découpes* : les hourdis rétrécis sont « sciés longitudinalement dans l'axe de l'alvéole » (p. 26) et leurs
     largeurs sont publiées (SP 200 : 520 / 700 / 890 / 1 080 mm), déjà écrites dans la rubrique ; les montrer sur la
     coupe. À comprendre d'abord : les largeurs publiées ne tombent pas pile sur les axes d'alvéoles (520 pour un axe à
     504 mm du bord : joint compris ?) — à demander ou à déduire du guide, jamais à inventer (petit).
  3. *Un module « choix de hourdis »* (après Charges sur plancher) : portée, appuis, charges permanentes et variables,
     couche ou non → pour chaque profil la variante la moins armée qui suffit, puis le plus léger. L'ELU se juge sur les
     tableaux M_Rd / V_Rd,c déjà transcrits (le poids propre de CHAQUE profil compté) ; les déformations ne s'en
     déduisent pas (la contre-flèche de précontrainte n'est pas publiée) : relever aussi les COURBES DE PERFORMANCE du
     guide (vectorielles dans le PDF, lisibles comme les coupes), et retenir la plus défavorable des deux lectures. Se
     conçoit par écrit avant le code (moyen à gros).
  4. *Les torons sur la coupe* : déconseillé tant que leur position n'est pas publiée — le guide donne nombre et Ø, pas
     l'implantation, et dit (p. 28) « Vérifiez toujours la disposition des torons avant de forer ! Demandez à chaque fois
     un schéma d'implantation » : un toron dessiné à une place supposée servirait à décider où forer. À faire sur le
     schéma d'implantation du fabricant ; en attendant, une mention sous la coupe (petit).

## 3. Plus tard : le confort

Démarrage plus rapide (moteur chargé une fois) ; ~~l'indice réel (A, B…) au lieu de « Brouillon »~~ (eb0a647 ;
l'indice ne se CHANGE pas encore à l'écran, question 14) ; ~~rouvrir une note sans changer sa date~~ (eb0a647) ;
~~boutons au doigt de 44 px~~ (bc112de) ; modifier les données sur le dessin (à trancher) ; ce qui dépend des
comptes en ligne (équipe, droits, statut validé/diffusé, rapport multi-modules, historique — choix de Supabase) ;
vérification finale d'ensemble ; petit rangement du code (en partie : e043d15, daa5bcf — le reste, question 14).

## 4. Les questions à trancher par l'ingénieur

1. ~~Poutre, cisaillement (S1)~~ — tranché le 09/11 : on ne change pas (remarque gardée, § 1).
2. ~~PDF de la dalle (S4) : sur le site maintenant ?~~ — tranché : sur le site (fait). Reste : corriger aussi Streamlit ?
3. ~~Anciennes notes de dalle et imports Streamlit~~ — tranché : toujours empilées, plus d'option (fait).
4. Ordre des défauts Streamlit signalés (proposition : bois, maturité, taux d'abord).
5. Feu des poteaux et voiles : transcription des tableaux 5.2a et 5.4 de l'EN 1992-1-2 depuis le texte papier.
6. Bibliothèque Béton armé : validée ? α_cc (0,85 ou 1,0) et γ_s du bureau (1,5) ?
7. Chantiers et dossiers (migration de format) : on y va ?
8. Comptes en ligne : Supabase, et quand ?
9. Ordre des modules à passer sur l'établi compact.
10. Cornière : les 10 questions sur le classeur Excel.
11. Plus petits : jeu entre lits par défaut (1 cm ou 2,1 cm) ; espacement maximal et 20 % secondaire de l'EC2 pour les
    dalles ; empiler les couches basses de la prédalle ; la branche du dépôt partagé à mettre à jour.
12. Nouvelles (relectures du 09/11) : ~~S5 (nappe secondaire sur un renfort principal plus épais, § 1)~~ (fait,
    c33dfd1) ; ~~garde-corps dont la note déborde~~ (fait : une page de suite dans le générateur, df6a8ea — la mise en
    page est à confirmer, question 13) ; ~~l'indice unique d'une note~~ (fait : celui de la note du site, df6a8ea — à
    confirmer, question 13) ; ~~l'avis d'ouverture d'une ancienne dalle, au téléphone, n'est que dans l'onglet
    Résultat~~ (410a00e : dit à l'ouverture, même message, même composant) ; ~~accueil : la carte de la bibliothèque fait
    trois rangées de haut de 701 à 1 360 px (les cinq modules du Béton sur deux colonnes), et son icône est à gauche du
    titre~~ (eeb29f5 : deux rangées, l'icône au-dessus du titre, les deux bibliothèques) ; le périmètre de la prédalle
    sur chaque planche de la note ?
13. Nouvelles (deuxième vague de la nuit du 09/11) :
    - **S6** (§ 1, sécurité, préexistant) : la pose des renforts principaux par rapport à un treillis soudé — sous, dans
      ou sur le treillis ? Puis la direction principale suit (calcul, coupe, regel).
    - Dalle, notes enregistrées entre la mise en ligne du 09/11 et S5 (état en version 3) avec un renfort principal plus
      épais que la base : elles se rouvrent avec les chiffres de S5, plus sûrs, mais SANS avis (430,1 → 459,8 mm² sur la
      note par défaut avec « + Renfort » ; une nappe conforme peut passer NON CONFORME sans un mot). Un avis demande une
      version 4 de l'état (une marque de format) : rien sans accord.
    - Dalle : une couche principale SAISIE dont le dessous tombe dans l'épaisseur de la nappe secondaire n'est pas du
      lit de contact, et rien ne le signale (un avertissement serait possible).
    - Décisions prises cette nuit sans l'ingénieur, à confirmer : les champs « Identification de la note » quittent
      l'écran du site des soudures et du plancher (sans effet depuis l'indice unique ; l'état les garde pour
      Streamlit ; regel fb7508a, séparable) ; la page de suite du garde-corps est UNE page 3 « SUITE DES PAGES 1 ET 2 »,
      une colonne par colonne d'origine, le cartouche de la page 2 ; assemblages : indice par défaut « A » (était
      « 0 »), projet « Assemblages » sans nom ni chantier ; la Bibliothèque Acier est une page NOUVELLE
      (`bibliotheque-acier`), le Catalogue des profilés masqué et non renommé (une consultation n'est pas une note : pas
      de migration).
    - Seuils d'attention des notes (85 %, 70 % pour les cordons) et de l'écran (95 %, GC10) : aligner les notes, ou le
      dire dans la note ? (générateurs validés.)
    - « Suite p. 3 » dans une note de plusieurs garde-corps (numéro du garde-corps, pas du PDF) ; « Projet » et « Date »
      d'un assemblage à l'écran, jamais imprimés ; soudures C6 en quatre tronçons à l'Eurocode, encore refusées.
    - Catalogue : les profilés personnalisés (PRS), la comparaison et l'export — dans la Bibliothèque Acier, ou laissés
      dans l'ancienne page masquée ? Au téléphone, un bouton « Fermer » dans la feuille « Colonnes » ?
14. Nouvelles (lot des petits chantiers, 10/11/2026) :
    - Pied de poteau : le périmètre par défaut « EC3 seul » laisse « Conforme » un pied dont le béton d'ancrage ne
      tient pas (VAL-A : 107,7 % ; un cas d'essai à 454 %) — l'écran le dit désormais sans ambiguïté (« hors périmètre »,
      « hors verdict », phrase d'information « non satisfait »). Garder ce défaut, ou passer à « EC3 + EC2-4 » par défaut
      (Streamlit d'abord, puis le site suit) ?
    - L'INDICE d'une note se lit à l'écran mais ne s'y change pas (il vient d'un `.stab` importé, « A » sinon) : un champ
      « Indice » dans « Classer la note… » ?
    - Le module Prédalles (page redirigée vers les dalles depuis la fusion) est encore enregistré, gelé et testé, et rend
      l'ancienne arborescence : le retirer permettrait de retirer `arbre(compact=False)`, `renommer()`,
      `[data-renommer]` et le repli de `js/compact.js` (avec ses 9 photos du gel et ses tests) — on le retire ?
    - « La copie en double dans le code de la dalle » et « le test manquant des bulles d'aide » : où étaient-ils écrits ?
      Introuvables dans `docs/site/*.md` (candidat possible, non touché : `_lecture` et `_formulaire_verrouille`, en
      deux exemplaires dans `api_web/dalle.py` et `api_web/predalle.py` — REFONTE_ETABLI.md, M1) ; le contrôle des
      bulles ⓘ au doigt existe dans le harnais (filtre « doigt »).
    - En ligne : vérifier que Cloudflare Pages sert bien `X-Robots-Tag: noindex, nofollow` (`curl -I`) après la mise en
      ligne.

## 5. Ce qui se fait sans attendre de décision (session de nuit)

~~S3 (renfort empilé, avec test et regel motivé)~~ (fait, 7d7da99) ; ~~les messages rouges de S1 et S2~~ (S1 : pas de
message, décision du 09/11 ; S2 : plus tard) ; ~~le nom
long coupé ; le lien vers une note absente ; « non vérifié » de la dalle coulée~~ (le lot des petits chantiers) ; BIB4 bis ; puis le premier module de
la liste des 11 (sans le publier comme « fini » avant la relecture de l'ingénieur) ; enfin `lancer_tests.py --tout`.

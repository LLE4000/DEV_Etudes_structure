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

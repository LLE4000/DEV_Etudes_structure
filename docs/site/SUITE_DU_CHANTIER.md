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
| S5 | Dalle (relecture du 09/11, PRÉEXISTANT depuis FUS4, ni introduit ni aggravé par la nuit) : la nappe SECONDAIRE (base, et renfort S3) se pose sur la BASE principale alors qu'elle croise un RENFORT principal plus épais, posé dans le lit de la base — les barres se traversent dans le calcul et la coupe, rien ne le dit : A_s,req secondaire sous-estimé de 6,9 % dès le mode par défaut « Treillis + renforts » (3 Ø12), 14,8 % avec un renfort Ø25 | `api_web/dalle_positions.axe_auto` (le moteur ne compare que les couches de base) | petit à moyen, puis regel | **oui** : poser la nappe secondaire sur la couche principale la plus épaisse qu'elle croise, ou avertir |

Puis, déjà SIGNALÉS à l'écran mais en attente d'arbitrage : ~~anciennes notes de dalle et fichiers importés non empilés
(avertissement affiché)~~ (réglé : 7d7da99, toujours empilées, un avis le dit à l'ouverture) ; défauts Streamlit marqués en rouge — bois (G seul, k_cr, flèche finale, appui, EN 338), feu
des poteaux et voiles (ancien module Enrobage, a trop faible de 15 à 25 mm), taux d'armature, maturité (décoffrage trop
tôt), cornière (longitudinale = transversale), catalogue (PRS), sol (k), plancher, flambement, choix de profilé (D1, D2,
D3, D5, D7), prédalle (couches basses croisées). Règle : Streamlit d'abord, puis le site suit.

## 2. Ensuite : les fonctions demandées

- ~~Rejouer TOUTES les suites (`python3 lancer_tests.py --tout`) avant la prochaine publication~~ : fait le 09/11 sur
  l'intégration (f93bdb8) — 114/114 suites vertes, 8 863 OK, 0 échec (581 s sur 4 processus) ; harnais complet sans
  filtre : 2 234 OK, 0 KO (941 s). À refaire avant chaque publication qui en doute.
- Site privé : Cloudflare Access sur stab-methodes.be ET sur l'adresse .pages.dev (à faire par l'ingénieur dans
  Cloudflare ; le 07/11, .pages.dev répondait sans mot de passe).
- Dalle : un nom de plus de 60 caractères fait refuser toute une ancienne note → le couper avec un message (petit).
- Un lien `?note=` vers une note absente ouvre une note vide sans rien dire → message clair et « Importer .stab »
  (petit).
- Dalle coulée : dire ce qui n'est pas vérifié (flèche, fissuration), comme la prédalle (petit).
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
- Notes PDF, écarts relevés par la relecture du 09/11 (à trancher, § 4) : le garde-corps REFUSE sa note quand le
  générateur validé déborde (86 des 240 états d'une grille : formes L/U avec « Cumuler qh et qv » sauf fixation scellée,
  L/U à montant soudé) ; l'INDICE a deux sources pour les soudures et le plancher (le PDF écrit « indice 0 » quand la
  note du site dit B — vu dans le vrai navigateur le 09/11) ; arrondi d'un taux en bord de bande et seuil d'attention à
  85 % dans les notes contre 95 % à l'écran (DEFAUTS_STREAMLIT.md) ; quand la géométrie gouverne, la page d'un
  assemblage garde le η du moteur.
- Bibliothèque Acier (BIB5) : le catalogue sur le modèle de la Bibliothèque béton (moyen).
- Chantiers et dossiers (nouvel écran, MIGRATION du format de la mémoire des notes : rien sans l'accord) (gros).

## 3. Plus tard : le confort

Démarrage plus rapide (moteur chargé une fois) ; l'indice réel (A, B…) au lieu de « Brouillon » ; rouvrir une note sans
changer sa date ; boutons au doigt de 44 px ; modifier les données sur le dessin (à trancher) ; ce qui dépend des
comptes en ligne (équipe, droits, statut validé/diffusé, rapport multi-modules, historique — choix de Supabase) ;
vérification finale d'ensemble ; petit rangement du code.

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
12. Nouvelles (relectures du 09/11) : S5 (nappe secondaire sur un renfort principal plus épais, § 1) ; garde-corps
    dont la note déborde (mise en page du générateur validé, ou l'accepter) ; l'indice unique d'une note (celui de la
    note du site, ou le champ de l'écran des soudures et du plancher) ; l'avis d'ouverture d'une ancienne dalle, au
    téléphone, n'est que dans l'onglet Résultat ; accueil : la carte de la bibliothèque fait trois rangées de haut de 701
    à 1 360 px (les cinq modules du Béton sur deux colonnes), et son icône est à gauche du titre ; le périmètre de la
    prédalle sur chaque planche de la note ?

## 5. Ce qui se fait sans attendre de décision (session de nuit)

~~S3 (renfort empilé, avec test et regel motivé)~~ (fait, 7d7da99) ; ~~les messages rouges de S1 et S2~~ (S1 : pas de
message, décision du 09/11 ; S2 : plus tard) ; le nom
long coupé ; le lien vers une note absente ; « non vérifié » de la dalle coulée ; BIB4 bis ; puis le premier module de
la liste des 11 (sans le publier comme « fini » avant la relecture de l'ingénieur) ; enfin `lancer_tests.py --tout`.

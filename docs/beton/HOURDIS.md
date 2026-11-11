# Hourdis alvéolés — les catalogues des fabricants (Ergon 2018)

Demande de l'ingénieur (11/11/2026), avec le guide technique Ergon 2018 : « on peut prendre leur hourdis. Il y a
toujours le MRd et le VRd qui sont indiqués […] tu as tous les tableaux […] les dimensions des hourdis qu'on peut
reprendre. Et donc là, ce sera Ergon, mais on mettra aussi des autres marques plus tard. »

Ce qui est fait : le catalogue Ergon 2018 est **transcrit une fois** (7 profils, 53 variantes), dans un fichier qui fait
foi ; deux écrans le lisent — le menu des types de **Charges sur plancher** et la rubrique **« Hourdis »** de la
Bibliothèque Béton armé. Une autre marque, c'est un fichier de plus et une ligne de `CATALOGUES` (§ 7).

## 1 · Les fichiers

| Fichier | Rôle |
|---|---|
| `beton/donnees/hourdis/ergon_2018.json` | **le catalogue** (source unique) : en-tête `meta` (fabricant, édition, document, unités, conditions d'emploi, couche collaborante, notation des torons, ce qui n'est pas publié, écarts du guide, pages) et `profils` (dimensions, poids, section, mise en œuvre, variantes, cotes, dessin relevé) |
| `beton/hourdis/catalogue.py` | **le lecteur** (bibliothèque standard seulement) : catalogues, profils, variantes, torons, fiche du module Charges sur plancher, sélection (`verifier`, `gamme`) |
| `tools/transcrire_hourdis_ergon.py` | la **transcription** (hors chemin de production) : trois lectures du PDF qui doivent s'accorder (§ 4) ; écrit le JSON et le rapport daté |
| `docs/validation/hourdis_ergon_transcription_2026-11-11.md` | la **mesure datée** de la transcription : 0 écart de relecture, relevé des coupes à 1,8 % au plus, trois erreurs du guide |
| `plancher/charges/bibliotheque.py` | les 53 types Ergon ajoutés aux types de hourdis du module Charges sur plancher (`types_hourdis`, `groupes_types_hourdis`, `resume_catalogues`) |
| `api_web/biblio_hourdis.py`, `api_web/dessin_biblio_hourdis.py` | la rubrique « Hourdis » de la Bibliothèque Béton armé et sa coupe cotée |
| `tests/test_hourdis_catalogue.py` | le catalogue (cohérence, empreinte, relues, lecteur, sélection, pureté, mutations) |
| `tests/test_api_web_bibliotheque_hourdis.py` | la rubrique (chaque case = le catalogue, la coupe, la sélection, l'adresse, la fiche, mutations) |
| `tests/test_charges_plancher_moteur.py`, `tests/test_api_web_plancher.py` | les types Ergon dans Charges sur plancher (P-17 révisée, menu, valeurs reprises, épaisseur exigée) |

## 2 · Ce que dit le guide : les conditions d'emploi

| Condition | Valeur | Page |
|---|---|---|
| béton des hourdis | C55/67 (E_cm = 38 214 MPa par le tableau 3.1 de l'EN 1992-1-1) | 13, 20 |
| classe de conséquence | CC2 | 8 |
| environnement | EI ou XC1 (autres exigences sur demande) | 13 |
| feu | Rf 60 (Rf 90 ou Rf 120 à la demande) | 13, 20 |
| valeurs | **de calcul, à l'ELU, par élément** : M_Ed ≤ M_Rd et V_Ed ≤ V_Rd,c, M_Ed et V_Ed calculés avec le poids propre de l'élément et la charge utile pondérés | 29 à 35 |
| largeur | module 1 200 mm (joint compris), élément réel 1 196 mm | 20 |
| poids propre | avec joints, en kN/m et kN/m² | 20 |
| couche collaborante | ≥ 50 mm au point le plus haut du hourdis, béton C30/37, treillis minimum 6×6×150×150 DE 500S ; les tableaux donnent **M_Rd avec 50 mm de couche, pas V_Rd,c** | 23 |
| joints | béton C30/37, granulats ≤ 8 mm, aucune circulation avant durcissement | 25 |
| notation | SP 200-4-2X : 4 torons Ø 12,5 et 2 torons Ø 9,3 (indice X) ; indice Y : Ø 15,2 (SPZ 500) | 29, 35 |

| Profil | Page | h (mm) | g (kN/m², joints compris) | Variantes | M_Rd (kN·m) | V_Rd,c (kN) | Appui min. béton/acier · maçonnerie (mm) |
|---|---|---|---|---|---|---|---|
| SP 150 | 29 | 150 | 2,45 | 5 | 44 – 77 | 95 – 101 | 80 · 100 |
| SP 200 | 30 | 200 | 2,78 | 6 | 51 – 134 | 89 – 96 | 80 · 120 |
| SPN 200 | 31 | 200 | 2,92 | 7 | 76 – 191 | 101 – 115 | 80 · 120 |
| SP 265 | 32 | 265 | 3,83 | 9 | 106 – 279 | 137 – 145 | 80 · 120 |
| SP 320 | 33 | 320 | 4,12 | 10 | 132 – 418 | 168 – 185 | 130 · 150 |
| SP 400 | 34 | 400 | 4,94 | 9 | 251 – 635 | 223 – 234 | 130 · 150 |
| SPZ 500 | 35 | 500 | 6,30 | 7 | 497 – 1 006 | 342 – 396 | 130 · 200 |

Les libellés courts que l'écran affiche (« Rf 60 », « CC2 », « EI ou XC1 », « M8 à M12, simple traction » pour les
chevilles dans l'alvéole) sont des champs du catalogue (`meta.*.court`), jamais écrits dans l'écran : une autre marque a
les siens (contrôlé par une mutation, § 8).

## 3 · Ce que le guide ne publie pas — et ce que l'application en fait

| Non publié | Conséquence |
|---|---|
| **V_Rd,c avec couche collaborante** | Charges sur plancher juge le tranchant sur **V_Rd,c nu** (la plus faible des deux : P-17 révisée, § 6.1) ; la bibliothèque le dit dans la bande et dans « Pourquoi ? » |
| moment de fissuration M_cr | la ligne « fissuration » de Charges sur plancher n'apparaît pas (information, P-11) |
| pression locale admissible sur la membrure supérieure | la membrure est « non vérifiée » sous un répartiteur efficace, « répartiteur exigé » sinon (P-5) |
| courbes de performance (charge utile admissible selon la portée, p. 21-22 et 29-35) | graphiques : non transcrits |

## 4 · La transcription : trois lectures, une empreinte

`python3 tools/transcrire_hourdis_ergon.py <ERGON2018_GuideTechnique_FR.pdf>` (PyMuPDF requis ; ni l'outil ni le PDF
n'entrent dans le chemin de production, le PDF n'est pas dans le dépôt) :

1. **la transcription à la main**, page par page (`TRANSCRIPTION` dans l'outil) — c'est elle qui est livrée ;
2. **la relecture du texte du PDF** : caractéristiques (p. 20), les sept tableaux M_Rd / V_Rd,c / M_Rd avec couche
   (ordre des lignes compris), poids à transporter, joints, cotes des coupes (p. 29-35), réservations (p. 24), appuis
   (p. 25), rétrécis (p. 26), chevilles (p. 28). Tout écart arrête l'outil ; **0 écart** le 11/11/2026 ;
3. **le relevé vectoriel des coupes** (contour gris, alvéoles blanches, à l'échelle de la cote 1 196) : l'aire,
   l'inertie et les modules recalculés sur le relevé sont confrontés aux valeurs publiées — 1,8 % au plus (SPN 200), les
   autres profils à 0,4 %. Le relevé est livré comme **dessin** de la coupe ; les valeurs de calcul restent celles du
   fabricant.

**Trois erreurs du guide lui-même**, nommées dans `meta.ecarts` (la valeur livrée est celle que les deux autres
lectures confirment) : la chaîne horizontale du SP 265 (p. 32, un 222 au bord au lieu de 154) ; la coupe du SP 400 de
la page 20 (218 et un 270 répété ; la page 34 dit 216 et 193) ; les joints du SPZ 500 (18,2 l/m² p. 35, 16,6 p. 20 :
les deux sont livrés).

**L'empreinte.** `tests/test_hourdis_catalogue.py` garde l'empreinte (sha256, 16 caractères) des valeurs de calcul et
des conditions qui les accompagnent (`EMPREINTE_ERGON`). Une valeur ne se corrige pas à la main dans le JSON : on corrige
la transcription, on relance l'outil sur le PDF, on reporte la nouvelle empreinte — et la mesure datée du jour le dit.
Quatorze valeurs relues dans le guide sont écrites une seconde fois dans le test (copie nommée, liée au fichier).

## 5 · Le lecteur `beton/hourdis/catalogue.py`

| Fonction | Rend |
|---|---|
| `catalogues()` · `catalogue(cle)` · `titre(cat)` | les catalogues du dossier, dans l'ordre de `CATALOGUES` ; « Ergon 2018 » |
| `profil(cat, nom)` · `variante(cat, nom)` · `torons(cat, nom)` | un profil ; (profil, variante) ; les torons d'une désignation (nombre, Ø) par la notation du catalogue |
| `inertie_element(p)` · `largeur_element(cat)` · `E_cm(cat)` | I_x en m⁴ ; 1,2 m (le module) ; E_cm du béton des hourdis en MPa (tableau 3.1) |
| `nom_type(cat, v)` · `chercher_type(nom)` | « SP 200-4-2X · Ergon 2018 » et l'inverse |
| `fiche_plancher(cle, nom)` · `fiches_plancher()` | la fiche d'un type pour Charges sur plancher (§ 6.1) ; les 53 |
| `verifier(p, M_Ed, V_Ed, avec_chape)` | variante par variante : ce qui suffit à M_Ed et V_Ed (par élément, ELU, poids propre compris) ; la **première** qui suffit = la moins armée (le tableau est rangé par M_Rd croissant) ; M_Rd avec couche seulement si demandé et publié ; V_Rd,c toujours nu |
| `gamme(cat, M_Ed, V_Ed, avec_chape)` | profil par profil : bornes de M_Rd et V_Rd,c, la première variante qui suffit ; le profil le plus **léger** (kN/m²) qui suffit |

Le lecteur n'importe que la bibliothèque standard (contrôlé dans un sous-processus qui refuse numpy, scipy, fitz et
streamlit). Il est expédié avec le moteur du site (le paquet de `construire.py` l'emporte, avec le JSON).

## 6 · Où le catalogue sert

### 6.1 Charges sur plancher

- Le menu « Type » du hourdis propose la **fiche générique à renseigner** (hors groupe, toujours en tête), puis les
  53 types Ergon **groupés par profil** (« Ergon 2018 · SP 150 » … « Ergon 2018 · SPZ 500 »), puis les fiches du
  fichier. Choisir un type reprend ses valeurs : b = 1,2 m, h, g (poids avec joints), E = E_cm du C55/67, I = I_x,
  M_Rd, V_Rd (nature Rd), M_Rd avec chape, épaisseur exigée 50 mm, classe exigée C30/37 (f_ck 30 MPa), source page par
  page. Les valeurs reprises restent modifiables ; revenir au type générique les efface.
- **P-17 révisée le 11/11/2026** : les valeurs « avec chape » de M et de V sont admises **séparément**, à trois
  conditions — chape collaborante, f_ck(t) de la chape ≥ la classe exigée, épaisseur ≥ l'épaisseur exigée par le
  fabricant (`Chape.valeurs_avec_chape_admises(e_min)`, `motif_refus`). Le catalogue Ergon ne publie que M avec chape :
  le moteur juge **M_Rd avec chape et V_Rd,c nu**, et l'hypothèse le dit (« M_Rd catalogue avec chape, V_Rd nu (valeur
  avec chape non publiée) ») ; une chape de 40 mm ou à 25 MPa garde les valeurs nues, et la note dit pourquoi.
- Un cas neuf démarre toujours sur le type générique « à renseigner » (point à confirmer, `docs/site/SUITE_DU_CHANTIER.md`).
- L'onglet Méthode (Streamlit) et la vue Méthode du site listent le catalogue (une ligne : profils, variantes, bornes de
  M_Rd et V_Rd,c, document) ; les anciens fichiers de cas s'ouvrent (le champ `e_chape` manquant est complété à vide).

### 6.2 La Bibliothèque Béton armé · rubrique « Hourdis »

Deux vues sur les mêmes paramètres (adresse : `?rubrique=hourdis&vue=…&fabricant=…&profil=…&variante=…`) :

- **Profil** (par défaut, SP 200) : la **coupe cotée** — le tracé relevé dans le guide, les cotes **imprimées par le
  fabricant** (chaîne entre axes d'alvéoles, largeur 1 196, chaîne verticale, hauteur, dimension des alvéoles ; rien
  n'est mesuré sur le tracé pour être coté) —, les caractéristiques (dimensions, poids, section, modules, appui
  minimal, transport et joints, hourdis rétrécis, réservations maximales, chevilles dans l'alvéole) et le **tableau des
  variantes** : M_Rd, V_Rd,c, M_Rd avec couche, les torons de chacune, la case choisie allumée.
- **Gamme** : les sept profils côte à côte (h, g, A, I_x, W_sup, W_inf, bornes de M_Rd et V_Rd,c).
- **Efforts facultatifs** M_Ed et V_Ed (par élément, ELU, poids propre compris) et la case « Avec couche
  collaborante » : les cases qui suffisent s'allument en vert, la première variante qui suffit est marquée ★ ; dans la
  Gamme, la première variante de chaque profil et le profil le plus léger qui suffit ★ — avec l'avertissement que le
  poids propre du profil retenu change M_Ed et V_Ed. Le jugement est celui du lecteur (`verifier`, `gamme`) : l'écran
  choisit, compose et formate.
- La bande, « Pourquoi ? » (les termes, la page, les conditions), les sources page par page, et la **fiche** que l'on
  ajoute à une note comme celles des autres rubriques (avec l'extrait du tableau).

## 7 · Ajouter un fabricant

1. Transcrire son guide dans `beton/donnees/hourdis/<fabricant>_<édition>.json`, **au même schéma** (le JSON Ergon est
   le modèle : `meta` complet, y compris les libellés `court`, la nature des valeurs — `Rd` ou `service` —, la couche
   collaborante et la notation des torons ; un profil par entrée, avec ses pages). Écrire l'outil de transcription à
   trois lectures sur le modèle de `tools/transcrire_hourdis_ergon.py`, et verser sa mesure datée dans
   `docs/validation/`.
2. Ajouter le fichier à `CATALOGUES` dans `beton/hourdis/catalogue.py` (le test du dossier exige que chaque fichier y
   soit nommé, et réciproquement).
3. Dans `tests/test_hourdis_catalogue.py` : le nouveau catalogue passe **automatiquement** les contrôles de cohérence
   (`defauts`, appliqués à chaque catalogue du dossier) ; lui ajouter son empreinte et ses valeurs relues. Une notation
   des désignations différente de celle d'Ergon se lit dans `torons` (et dans la copie du test) : elle s'écrit, elle ne
   se contourne pas.
4. Rien à recoder ailleurs : le menu de Charges sur plancher prend un groupe par profil (« <Fabricant> <édition> ·
   <profil> »), la rubrique « Hourdis » ajoute le fabricant à son menu « Fabricant », la vue Méthode une ligne.

## 8 · Les contrôles

- `tests/test_hourdis_catalogue.py` (87) : le dossier (`CATALOGUES` = les fichiers) ; la cohérence de chaque catalogue
  (chaînes de cotes = b et h, alvéoles, poids ÷ 1,2, section du dessin à 2 % du publié, centres d'alvéoles à 5 mm des
  cotes, M_Rd strictement croissant et aire des torons croissante dans chaque profil, M_Rd avec chape > M_Rd,
  désignations, libellés courts) ; les écarts du guide ; l'empreinte ; quatorze valeurs relues ; la fiche de Charges
  sur plancher ; la sélection (`verifier`, `gamme`) ; la pureté ; **17 mutations** du catalogue qui rougissent.
- `tests/test_api_web_bibliotheque_hourdis.py` (100) : chaque case de chaque profil = le catalogue ; la coupe (cotes du
  fabricant, légende) ; les caractéristiques ; la sélection ★ ; la Gamme ; les paramètres et refus doux ; la fiche ;
  mutations (une valeur du catalogue changée → la bande suit ; le lecteur qui ne désigne plus → l'étoile s'éteint ; les
  libellés du fabricant changés → l'écran suit).
- `tests/test_charges_plancher_moteur.py` et `tests/test_api_web_plancher.py` (§ « Catalogue Ergon 2018 ») : P-17
  révisée (M avec chape et V nu, épaisseur, classe), le menu (54 options, 7 groupes), les valeurs reprises, la chape à
  40 mm ou 25 MPa, la fiche gardée, le retour au générique, la vue Méthode, un ancien fichier.
- Le harnais du site, filtre « bibliotheque » : la rubrique aux quatre formats (PC, tablette paysage et portrait,
  téléphone) — le cadre ne défile pas au PC et à la tablette, la case allumée se voit, chaque case rallume la sienne.

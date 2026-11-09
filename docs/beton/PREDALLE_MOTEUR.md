# Prédalle béton armé — le moteur sans Streamlit et le module du site

*Livré le 07/11/2026. Moteur `beton/predalle/` : bibliothèque standard seule, Python 3.12 et 3.13 (donc Pyodide 0.29.5).
Extrait de l'écran Streamlit `modules/predalle.py` v1.1, qui reste la **référence** ; la preuve de parité est
`tests/test_predalle_moteur.py`. Le module du site est `api_web/predalle.py` (dessins : `api_web/dessin_predalle.py`,
feuille de style : `site/web/css/modules/predalle.css`, tests : `tests/test_api_web_predalle.py`) ; son écran est
l'établi compact des trois modules béton armé, aux places de la poutre (`api_web/plaque_compact.py`,
`api_web/etabli_compact.py` ; `docs/site/REFONTE_ETABLI.md`, « Un seul établi pour les trois modules béton armé »).*

> **À LIRE AVANT D'EMPLOYER LE MODULE — défauts NON CONSERVATIFS de l'écran Streamlit, reproduits (parité) et signalés
> sur le site** (détail §8) :
>
> 1. **Couches croisées calculées au même niveau** (§8, défaut 4) : la base secondaire et les renforts inférieurs de la
>    direction principale sont tous posés à h_pre + ⌈Ø/2⌉ alors qu'ils se croisent ; de même les deux bases
>    supérieures X et Y à c + ⌈Ø/2⌉ + j. La couche qui passe dessus a en réalité un d PLUS PETIT que celui calculé
>    (exemple : renfort Y 3 Ø12 à 7,0 cm et base X à 6,5 cm, l'un des deux est en vrai à au moins 7,6 cm). Le site le
>    dit à chaque calcul (avertissements « croisement sur la prédalle » et « nappes confondues ») ; **saisir l'axe** de
>    la couche du dessus corrige.
> 2. **La peau ramenée en silence** (§8, défaut 1) : l'écran réduit h_pre sans le dire quand l'épaisseur totale baisse
>    ou à l'ouverture d'un fichier ; les couches posées SUR la prédalle descendent avec elle, leur d grandit. Le site
>    refuse la saisie et signale le fichier.
> 3. **Ce que l'écran ne vérifie pas du tout** (§7) : phase provisoire (prédalle seule, étaiement, levage),
>    cisaillement à l'interface prédalle / béton coulé en place (EC2 §6.2.5), flèche, fissuration. Un « conforme » du
>    module ne couvre QUE la section finale en flexion simple et τ ≤ τ_adm,I.

## 1 · Ce qui est livré

| Pièce | Rôle |
|---|---|
| `beton/predalle/entree.py` | la peau préfabriquée (`h_pre`, bornes de l'écran : 3 cm à min(15 ; h − 1)), les défauts de l'écran prédalle (« Prédalle 1 », h = 22, peau 6), la normalisation ; le reste est contrôlé par le moteur des dalles |
| `beton/predalle/moteur.py` | la RÈGLE PRÉDALLE des axes automatiques, le calcul (celui du moteur des dalles sur les axes figés), les avertissements propres à la prédalle, le sélecteur de la nappe de base — et le format (entrée, sortie) dans sa docstring |
| `beton/predalle/fichier.py` | la lecture d'un fichier de l'écran (« 💾 Enregistrer » : `predalle_ba.json`, migration v1 comprise), avec ce que l'écran y remplace en silence |
| `tests/test_predalle_moteur.py` | données, contrôles à la main, parité (calcul, couches, écran, note PDF, fichier), couverture, sélecteur, fichiers, format, pureté, **mutations** |
| `api_web/predalle.py`, `api_web/dessin_predalle.py`, `site/web/css/modules/predalle.css` | le module du site : saisie, résultat, fenêtres de calcul, sélecteur, coupes et plan à l'échelle |
| `api_web/plaque_compact.py`, `api_web/etabli_compact.py` | l'établi compact partagé avec les dalles (et, pour les briques, la poutre) : arborescence, rubriques, nappes et couches, en-tête des vues, un bloc par section |
| `tests/test_api_web_predalle.py` | ce que le site affiche = ce que le moteur calcule ; saisies, refus, effets de bord, sélecteur, fichier Streamlit, avertissements, dessins, échappement |

## 2 · Ce qu'est l'écran prédalle — et pourquoi le moteur emploie celui des dalles

L'en-tête de `modules/predalle.py` le dit : c'est une **copie adaptée** de l'écran Dalle (`modules/dalle.py`), « AUCUNE
formule nouvelle » ; ce qui change est « la GÉOMÉTRIE DES COUCHES, rien d'autre ». Le code le confirme : comparés ligne
à ligne, les deux fichiers ne diffèrent que par l'espace de clés (`pre…` au lieu de `dal…`), les défauts (h = 22),
le champ « Prédalle (cm) » (`h_pre`) et ses bornes, et **une** fonction de calcul : `_auto_dist_couche`, l'axe
AUTOMATIQUE d'une couche. `_dimensionnement_compute_states` (toutes les formules) est identique mot pour mot.

Le moteur fait donc exactement ce que l'écran fait pour sa propre note PDF (`_valeurs_pour_export`) : il **fige** l'axe
effectif de chaque couche — l'automatique de la prédalle, ou l'axe saisi — puis confie le calcul au moteur des dalles
(`beton/dalle`, extrait de `modules/dalle.py` et prouvé contre lui, `docs/beton/DALLE_MOTEUR.md`). Il n'emploie que son
API **publique** (le test le contrôle). Une seule copie des formules ; mais la parité de la prédalle est prouvée
**directement** contre `modules/predalle.py` (le test rejoue l'écran prédalle, pas l'écran dalle) : si l'une des deux
copies Streamlit change, le test rougit.

## 3 · La règle prédalle (la seule différence de calcul)

`distance_auto(couche, rang, face, principale, enrobage, jeu, h_pre)` — mêmes opérations, même ordre que l'écran :

| Couche | `pose` | Axe automatique (cm) | Exemple (c = 3, h_pre = 6) |
|---|---|---|---|
| base (couche 1) inférieure de la direction **principale** | `predalle` | c + ⌈Ø/2⌉₀,₅ — **sans** jeu premier lit (fond de moule) | Ø10 : 3 + 0,5 = 3,5 ; Ø20 : 3 + 1 = 4,0 |
| toute autre couche inférieure (base secondaire, tous les renforts) | `sur_predalle` | h_pre + ⌈Ø/2⌉₀,₅ | Ø10 : 6 + 0,5 = 6,5 ; Ø12 : 6 + 1 = 7,0 |
| couche supérieure | `superieure` | c + ⌈Ø/2⌉₀,₅ + jeu premier lit (la règle de la dalle) | Ø10, jeu 0 : 3,5 |

⌈x⌉₀,₅ : arrondi au 0,5 cm **supérieur** de Ø/2 (Ø en mm, /20 → cm). Un enrobage nul est lu 3,0 cm (« or 3.0 ») et une
peau nulle 6,0 cm (« or 6.0 », inatteignable : h_pre ≥ 3). Un axe saisi (colonne « Dist. axe ») remplace l'automatique ;
un centre de gravité saisi (« CDG ») remplace Σ As·e / Σ As. Chaque direction garde ainsi SA hauteur utile : d principal
grand, d secondaire réduit de la peau.

Tout le reste — h_u,min, h_min = h_u,min + y_G de la famille la plus exigeante (audit C-1), A_s,req = M / (f_yd·0,9·d),
A_s,min = max(0,26·f_ctm/f_yk·b·h ; 0,0013·b·h ; 0,25·A_s,req opposée de la même direction), A_s,max = 0,04·b·h,
τ = V / (0,75·b·h) ≤ τ_adm,I (pas d'étriers) — est la méthode des dalles, décrite §3 et §4 de `DALLE_MOTEUR.md`.

## 4 · L'API

```python
from beton.predalle import moteur as M

M.nouvelle_predalle() -> dict        # « Prédalle 1 », bande 100 × 22 cm dont 6 cm de peau, C30/37, B500, enrobage 3, Y, γs 1,5
M.nouvelle_section(nom="A") / M.nouvelle_couche(base=True) / M.nom_section_suivant(predalle)   # ceux des dalles
M.normaliser(predalle) -> dict       # complétée et contrôlée ; ValueError hors du domaine de l'écran (la peau d'abord)
M.h_pre_max(h) -> float              # _h_pre_max : max(3, min(15, h − 1))
M.distance_auto(c, rang, face, principale, enrobage, jeu, h_pre) -> float
M.pose(face, rang, principale) -> "predalle" | "sur_predalle" | "superieure"
M.verifier(predalle) -> dict         # le calcul de toutes les sections
M.candidats_base(predalle, i_section, face, familles=None) -> list   # l'axe de chaque candidat recalculé par la règle prédalle

from beton.predalle import fichier
fichier.lire(payload) -> {"predalles": [...], "gamma_s", "jeu_premier_lit", "remarques": [...]}
```

**Entrée** : celle du moteur des dalles (`DALLE_MOTEUR.md` §2) plus `"h_pre"` (cm). **Sortie** de `verifier` : celle du
moteur des dalles, avec `"reference": "modules/predalle.py v1.1"`, `"predalle"` (l'entrée normalisée, à la place de
`"dalle"`), et :

- dans les grandeurs de chaque section : `h_pre` (la peau) et `h_coule` (h − h_pre, le béton coulé en place) ;
- dans chaque couche de chaque face : `auto` (axe automatique ?), `e_auto` (l'automatique DE LA PRÉDALLE, même pour un
  axe saisi) et `pose` (`predalle`, `sur_predalle`, `superieure`) ;
- en tête des `formules` de chaque vérification de face, une formule `e` par couche (ses termes : `c` ou `h_pre`,
  `demi_d`, `d`, `jeu` ; ou `saisi`), recalculable depuis ses seuls termes (le test le fait) ;
- deux avertissements de plus que le moteur des dalles (absents de l'écran Streamlit, aucun chiffre n'en change) :
  `base_hors_predalle` (la base principale, posée dans la peau, en sort : axe + Ø/2 > h_pre) et
  `croisement_sur_predalle` (deux couches inférieures croisées, posées toutes deux sur la peau au même niveau ; le cas
  base contre base reste l'avertissement `nappes_confondues` des dalles).

## 5 · La preuve de parité (`tests/test_predalle_moteur.py`)

Chaque note est chargée dans l'**état de l'écran** `modules/predalle.py` piloté sans navigateur (vrai `st.session_state`,
ses fonctions de défauts, de migration et de synchronisation des champs), puis comparée au moteur :

1. toutes les clés de `_dimensionnement_compute_states`, direction par direction, et ses états ;
2. chaque couche : `_couche_as_per_m`, `_dist_couche_eff`, `_auto_dist_couche` (LA règle prédalle), `_couche_label`,
   `_couche_diam_mm`, le drapeau « automatique » ; chaque face : `_layers_geometry` avec et sans CDG saisi ;
3. ce que l'écran AFFICHE : `render_dimensionnement_right` exécuté avec un faux `st` qui enregistre, et un espion sur
   `open_bloc_left_right` — titre, texte de droite, état, **pourcentage exact** de chaque bloc, lignes et infobulles ;
4. la note PDF : ce que l'écran lui envoie (`_valeurs_pour_export`, axes figés au centième) relu par
   `modules/export_pdf_dalle._compute_section` ;
5. le fichier : ce que « 💾 Enregistrer » écrit (`_build_save_payload`), relu par `beton/predalle/fichier.py`.

**Résultats (07/11/2026, 152 contrôles au vert)** : 72 notes (80 prédalles, 115 sections : les cas de `tests/test_predalle.py`, la peau à ses
bornes, chaque béton, B400, γs 1 / 1,15 / 2, jeu premier lit, enrobage nul, bandes de 20 à 500 cm, h de 5 à 100 cm,
axes et CDG saisis, quatre couches, acier excessif, τ > τ_adm,I, sans moment, plusieurs sections et prédalles, 30 tirages
au sort à graine fixe, deux bornes EXACTES τ = τ_adm,I et h = h_min construites avec l'écran) ; **20 750 comparaisons
numériques, 99,4 % identiques au bit près, écart relatif maximal 1,7·10⁻¹⁶** ; les 72 situations de couverture atteintes
(chaque état, motif, terme de A_s,min, type, pose, avertissement…) ; le sélecteur repassé par l'écran nappe par nappe
(7 faces × 40 nappes) ; 7 fichiers de l'écran (migration v1, peau ramenée, peau nulle, fichier du module Dalle,
remplacements muets) ; 24 refus ; 5 322 formules recalculées depuis leurs seuls termes (dont 651 axes de couche) ; pureté sous
Python 3.12 (streamlit, numpy, reportlab, `modules/` refusés par `sys.meta_path`).

**Mutations** (18, toutes rougissent) : la base principale avec le jeu premier lit ; toute couche inférieure sur la
peau ; les renforts principaux dans la peau ; « sur la peau » avec la règle de la dalle, sans l'arrondi, avec Ø entier ;
la direction principale ignorée ; la peau lue 6 cm ; l'axe saisi écrasé ; le jeu oublié en haut ; l'enrobage nul lu 0 —
et, injectées dans le moteur des dalles qu'il emploie : bras de levier 0,85, A_s,min avec la face elle-même, τ_adm,II,
hauteur par le plus grand moment (avant l'audit C-1), A_s ≤ A_s,max oublié, « n barres » sans la largeur de bande, y_G
saisi ignoré. Un témoin (le moteur rechargé sans altération) reste vert.

## 6 · Le module du site : tout ce que l'écran Streamlit offre

| Écran Streamlit | Site |
|---|---|
| plusieurs prédalles ; 📋 copier, 🗑️ supprimer (jamais la 1) ; nom | l'arborescence de la note : « Ajouter une prédalle », ouvrir ; le menu ⋯ de chacune : renommer, copier (insérée après elle), supprimer (jamais la dernière) |
| 🔒 verrouiller une prédalle | « Verrouiller la prédalle » : lecture seule, aucun sélecteur |
| classe de béton, B400 / B500, largeur de bande, épaisseur, **Prédalle (cm)**, enrobage, direction principale | section 1 « Prédalle » : les mêmes champs, le coulé en place (h − h_p) affiché, la peau bornée et refusée hors bornes |
| sections : ➕ ajouter, 📋 copier, 🗑️ supprimer, nom ; Mx inf / Mx sup / My inf / My sup / V max | section 2 : une carte par section (tableau X / Y × inf. / sup., V), copier, retirer, ajouter |
| tableau des couches par nappe (quatre nappes) : type Treillis / Barres / n barres, treillis ou Ø, espacement ou nombre, Dist. axe (auto ou saisie), CDG (auto ou saisi), ＋ renfort (4 couches), 🗑️ | une section « Armatures » par section : les mêmes champs, la POSE de chaque couche dite (dans / sur la prédalle, depuis le dessus) avec sa règle, « ↺ » pour revenir à l'automatique |
| bandeaux Hauteur, quatre nappes (principale d'abord), Effort tranchant si V > 0 : état, %, A_s,req / A_s,min / A_s,max / A_s fourni et leurs infobulles, « d utile ≤ 0 » | résultat : verdict, une ligne par vérification (taux, couleur, nappe), groupées par section ; fenêtre de calcul par section : hypothèses, tableau, formules substituées (axes compris) |
| icônes d'état de chaque section et de chaque prédalle | taux et couleur de chaque section (carte, groupe du résultat) et de chaque prédalle (liste de la note) |
| ⚙️ γs, jeu premier lit (communs à la note) | section « Méthode » : γs et jeu communs à toutes les prédalles de la note |
| 💾 Enregistrer / 📂 Ouvrir (JSON) | le .stab du site ; et « Ouvrir un fichier Streamlit » (predalle_ba.json) avec ce que l'écran remplaçait en silence |
| 🔄 Réinitialiser, 🏠 Accueil, infos projet (nom, partie, date, indice) | nouvelle note, accueil, et le cartouche de la note du site (nom, chantier, indice, date) |
| (rien) | sélecteur de la nappe de base trié par section au mètre (✓ retenue, ★ la plus petite qui tient, axe recalculé) ; étoile sur chaque ligne ; coupes et plan à l'échelle ; avertissements ; verdict qui nomme la vérification NON conforme ; périmètre vérifié écrit sous le résultat et dans chaque fenêtre de calcul ; pictogramme de pose de chaque couche |

## 7 · Ce qui n'est pas couvert

- **La note PDF** (`modules/export_pdf_dalle`, bouton « Générer PDF ») : hors périmètre du site (bouton grisé). Sa
  partie calcul est comparée par le test de parité.
- **Les unités d'affichage** (⚙️ « Unité de longueur » cm/mm, « Unité d'armature » mm²/cm²) : le site écrit les
  longueurs en cm et les sections en mm² (les défauts de l'écran) ; aucun chiffre n'en dépend.
- **Ce que l'écran Streamlit ne fait pas, le site non plus** — à savoir avant d'employer le module pour une vraie
  prédalle : la **phase provisoire** (prédalle seule, poids du béton frais, portée entre étais, levage), les
  **raidisseurs** (treillis de rigidité), l'**étaiement**, le **cisaillement à l'interface** prédalle / béton coulé en
  place (EC2 §6.2.5), la fissuration et la flèche, le sens de portée unique des prédalles (l'écran calcule les deux
  directions comme une dalle). Le moteur ne les invente pas (« rien de neuf sans le dire ») ; ils relèvent d'un module à
  écrire et à valider par le bureau.

## 8 · Défauts constatés dans l'écran Streamlit (le moteur reproduit, le site les dit)

1. **La peau est ramenée en silence** dans [3 ; min(15 ; h − 1)] (`_ensure_defaults_for_dalle`, audit I-5) : réduire
   l'épaisseur, ou ouvrir un fichier, change la peau sans un mot — **non conservatif** : la peau calculée est plus mince
   que la peau réelle, les couches posées sur elle descendent et leur d grandit. *Site* : une peau hors bornes, ou une épaisseur qui ne
   la contiendrait plus, est **refusée** avec un message ; un fichier qui en porte une est repris comme l'écran, et c'est
   dit.
2. **Une peau nulle dans un fichier** reste 0 dans l'état de l'écran mais est calculée 6 cm (« or 6.0 ») ; le champ
   (min 3) refuse alors de s'afficher. *Site* : relevé à la reprise du fichier.
3. **Documentation contradictoire de la règle des axes** : l'en-tête du fichier dit la base principale posée à
   « enrobage + demi-Ø + jeu premier lit (règle Dalle inchangée) », le code (retour du bureau du 01/09) la pose SANS jeu ;
   l'infobulle de la colonne « Dist. axe » dit « enrobage + demi-Ø + jeu premier lit (renfort dans le plan du treillis) »
   et celle du « Jeu premier lit » « compté dans la distance d'axe des couches » — faux pour toutes les couches
   inférieures d'une prédalle. *Site* : la règle juste est écrite sur chaque couche et dans l'aide du jeu.
4. **Couches croisées calculées au même niveau sur la peau** (non conservatif) : la base secondaire et les renforts de
   la direction principale sont tous posés à h_pre + ⌈Ø/2⌉ alors qu'ils se croisent ; celle du dessus a en réalité un d
   plus petit (cas 2 de `tests/test_predalle.py` : renfort Y à 7,0 cm, base X à 6,5 cm). Même défaut, hérité de la
   dalle, pour les bases supérieures X et Y. *Site* : avertissements `croisement_sur_predalle` et `nappes_confondues` ;
   saisir l'axe corrige. *Révision du 09/11/2026 (tranché, `docs/site/DALLES_FUSION.md` § 4) : avec l'option de note
   « nappes empilées » (vraie pour une note neuve, fausse pour une note d'avant ou un fichier importé), le site pose les
   bases SUPÉRIEURES secondaires sur les principales par `dist` ; les faces inférieures gardent la règle de pose du
   moteur (base principale dans la peau, autre base sur la peau) ; le moteur ne change pas.*
5. **La base principale peut sortir de la peau** sans contrôle : peau de 3 cm et treillis Ø12 → axe 4,0 cm, dessus des
   fils à 4,6 cm. *Site* : avertissement `base_hors_predalle` (grave).
6. **Un fichier du module Dalle ouvert dans l'écran Prédalle** (`payload.get("predalles", payload.get("dalles"))`)
   donne une note neuve aux noms de la dalle, sans aucune de ses valeurs, sans un mot. *Site* : relevé.
7. Hérités de l'écran Dalle (`DALLE_MOTEUR.md` §7) : moment négatif lu 0, espacement et nombre de barres bornés en
   silence, type ou Ø inconnus ramenés au défaut, axe ou CDG illisibles lus « automatique », enrobage nul lu 3 cm.
   *Site* : refusés (saisie) ou relevés (fichier) ; l'enrobage nul est calculé 3 cm comme l'écran, et c'est dit.
8. **Aucune borne contre l'absurde** : un moment de 10³⁰⁰ kN·m, un axe ou un jeu premier lit de plusieurs mètres sont
   calculés (taux de 10³⁰⁰ %). *Site* : bornes PROPRES AU SITE, qui ne refusent que l'absurde — sollicitation au plus
   100 000 kN·m ou kN, longueur saisie (axe, y_G, jeu) au plus 100 cm, nom au plus 60 caractères — à la saisie, dans
   un .stab et dans un fichier Streamlit.
9. **Une prédalle non conforme peut afficher une vérification conforme comme la plus chargée** : acier excessif
   (A_s > A_s,max) et d utile ≤ 0 sont non conformes sous 100 % ; l'écran n'a pas de verdict global, il colore le
   bloc. *Site* : le verdict, le résumé de la note, le taux de chaque section et de chaque prédalle nomment la
   vérification NON conforme la plus chargée (`gouvernante()` : un CHOIX parmi les vérifications du moteur, rien n'est
   recalculé) ; la ligne dit pourquoi elle est rouge sous 100 %.

## 9 · La relecture du 07/11/2026 (bout à bout, robustesse)

- **Bout à bout** (`tests/test_api_web_predalle.py` §11) : 31 notes (les cas courants, les bornes de la peau, chaque
  type de couche, axes et y_G saisis, B400, γs 1,15 et 2, C25/30 à C40/50, enrobage nul et 20 cm, acier excessif,
  τ > τ_adm,I, τ = τ_adm,I À LA LIMITE (100 %, vert foncé) et un cheveu au-dessus, 5 m × 100 cm, 5 cm, plusieurs
  sections et prédalles, 8 tirages à graine fixe) — 233 vérifications. Ce que le SITE affiche (lignes du résultat,
  fenêtres de calcul, liste de la note, passés par `api_web/routeur.traiter`) est comparé à ce que l'écran Streamlit
  `modules/predalle.py` AFFICHE (`render_dimensionnement_right` rejoué avec un faux `st`) : ordre et nombre des
  vérifications, taux (le pourcentage exact de l'écran arrondi par `C.pct`, sans quitter sa bande de couleur), état,
  A_s,req, A_s,min, A_s,max, A_s fourni (au mètre et sur la bande), d, h_u,min, h_min, y_G de la famille qui gouverne,
  τ et τ_adm,I, couleur de chaque section et de chaque prédalle. Aucun écart. Trois mutations du site (axe de la base
  avec le jeu, A_s arrondi de 0,2 %, taux de 0,6 %) font rougir la comparaison.
- **Robustesse** (§10) : saisies absurdes, chemins inexistants (« champ introuvable », plus de message Python brut),
  verrou lu « false », note .stab abîmée refusée (« predalles » qui n'est pas une liste n'est plus remplacé en silence
  par une note neuve), nom de section vide relu, fichier Streamlit aux valeurs absurdes refusé.
- **Écran** (captures PC, tablette, téléphone, thème sombre, sur une note de trois prédalles dont une non conforme
  sous 100 %) : le PÉRIMÈTRE vérifié est écrit sous le résultat et au pied de chaque fenêtre de calcul (section finale
  seulement ; phase provisoire, étaiement, interface, flèche, fissuration non vérifiés) ; la pose de chaque couche a un
  pictogramme (une coupe en miniature : barre dans la peau, sur la peau, sous le dessus) au lieu d'un carré coloré qui
  se lisait comme une case à cocher ; deux couches distantes de 6,5 et 7,0 cm ne sont plus dites « au même niveau »
  mais « à des niveaux trop proches pour se croiser » (le moteur et le site) ; résumés de sections repliées raccourcis
  (le détail au survol) et légende du plan raccourcie : plus rien de coupé au format PC.
- **Reste à l'enveloppe commune** (signalé, non bricolé) : au téléphone, aucun onglet n'atteint le Résultat ni le
  dessin ; en tablette portrait, la toile centrale étroite rend les coupes petites.

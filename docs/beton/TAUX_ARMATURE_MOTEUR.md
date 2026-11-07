# Taux d'armature — le moteur sans Streamlit et le module du site

*Livré le 07/11/2026. Moteur `beton/taux/` : bibliothèque standard seule, Python 3.12 et 3.13 (donc Pyodide 0.29.5).
Extrait de l'écran Streamlit `modules/taux_armature.py` (« Poutre BA – Modélisation et taux d'armature »), qui reste la
**référence** et le secours, inchangé ; la preuve de parité est `tests/test_taux_moteur.py`. Le module du site est
`api_web/taux.py` (adresse `taux-armature`), ses dessins `api_web/dessin_taux.py`, ses tests `tests/test_api_web_taux.py`.*

## ⚠ Défauts NON CONSERVATIFS de l'écran Streamlit — à lire d'abord

Un quantitatif est « non conservatif » quand il **sous-estime** une quantité à commander. L'écran Streamlit le fait dans
les cas suivants ; le moteur les **reproduit** (parité) et le site les **dit** (avertissement dans le résultat) :

| | Ce qui est sous-estimé | Quand | Ce que le site affiche |
|---|---|---|---|
| **D9** | l'acier des étriers : **un étrier de moins** dans une zone | L / s tombe juste mais le flottant est un peu au-dessous de l'entier : `int(2,9999999999999996) + 1` = 3 au lieu de 4 (L = 1 m, zone 0–30 %, s = 10 cm ; aussi L = 1 m, 0–70 % à 10 cm ; L = 2 m, 0–30 % à 20 cm…). **Fréquent** : mesuré le 07/11/2026 sur 39 690 zones ordinaires (L de 2 à 12 m par 0,5 m, De et À par pas de 5 %, s de 5 à 30 cm), **2 012 zones (5,1 %)** perdent un étrier | avertissement **grave** dans le résultat, quotient écrit en entier dans la fenêtre de la zone |
| **D2** | le **volume de béton** et son poids (le taux en kg/m³ est, lui, surestimé) | forme « Rectangulaire + deux talons » : les deux talons sont ignorés par le calcul | message **grave** dans les données et le résultat, talons en fantôme sur la coupe |
| **D11** | l'acier des **cadres de talon** | forme à talon avec des étriers : seul le contour de l'âme est compté | avertissement « les cadres ou épingles du talon ne sont pas comptés » |
| **D5** | l'acier des **retours** | un retour ajouté vaut 0 cm tant que sa hauteur n'est pas saisie (la hauteur proposée à l'ajout est perdue) | avertissement « retour de 0 cm » |
| **D13** | une famille ou une zone entière | « À » ≤ « De », ou zone sans type : elle pèse 0 **sans un mot** | avertissement **grave**, bilan rouge sous la zone |

Aucun de ces défauts n'est corrigé en silence (règle 2 de `docs/site/MODULES_SITE.md`) : les chiffres du site sont ceux
de l'écran, et ces messages disent où ils sont faux.

Le module est un **quantitatif** : il ne vérifie rien. Il donne le volume de béton d'une poutre, la masse de ses armatures
(familles de barres longitudinales, zones d'étriers), majorée d'un pourcentage global, et leur rapport : le **taux
d'armature en kg/m³**. Sur le site, la note est donc une « Consultation » (pas de taux de travail, pas de couleur de
conformité).

## 1 · Ce qui est livré

| Pièce | Rôle |
|---|---|
| `beton/taux/entree.py` | les listes, libellés, bornes et défauts de l'écran ; la normalisation stricte d'une poutre (ValueError hors du domaine) |
| `beton/taux/geometrie.py` | la section (âme, talons), son aire, son polygone, la position de chaque barre (`bar_centers`) |
| `beton/taux/moteur.py` | longueurs, étriers, quantités, taux, couleurs et codes des familles, avertissements, candidats de Ø, note entière — et le **format complet** dans sa docstring |
| `beton/taux/saisie.py` | les effets de bord de l'écran à la saisie (ajouts, forme, portée en mètres, retours, types d'étriers) |
| `tests/test_taux_moteur.py` | parité avec l'écran rejoué, contrôles à la main, couverture, effets de bord, constats AppTest, mutations, format, pureté (127 contrôles, 5 s) |
| `api_web/taux.py` | le module du site : état de la note (plusieurs poutres), saisie, résultat, fenêtres de calcul, sélecteur de Ø |
| `api_web/dessin_taux.py` | coupe en SVG à l'échelle (mm) ; élévation à l'échelle des longueurs, hauteurs amplifiées au-delà d'un élancement dessiné de 6 (écrit sur le dessin) ; tableau récapitulatif en HTML |
| `site/web/css/modules/taux.css` | la feuille propre au module (chiffre du taux, synthèse, cartes, pastilles, vue « Tableau ») |
| `tests/test_api_web_taux.py` | l'écran du site = le moteur = ce que l'écran Streamlit écrirait ; saisies, refus, effets de bord, sélecteur, dessins (dont : aucune écriture sur une autre), échappement (150 contrôles) |

Chaque fonction de `geometrie.py` et de `moteur.py` reprend l'expression du module **dans le même ordre d'opérations** :
les flottants sont identiques **au bit près** (6 469 comparaisons sur 60 poutres, 100 %).

## 2 · L'API

```python
from beton.taux import moteur as M, saisie as SA

M.nouvelle_poutre() -> dict          # l'écran au premier affichage : 30 × 60 cm, L 6 m, c 3 cm, B500A, majoration 5 %, aucune armature
M.normaliser(poutre) -> dict         # complétée (clé absente → défaut de l'écran) et contrôlée ; ValueError avec le chemin du champ
M.calculer(poutre) -> dict           # le quantitatif (sortie ci-dessous)
M.candidats_diametre(poutre, i)      # les dix Ø de l'écran pour la famille i, la poutre recalculée pour chacun
M.synthese_note([res, …]) -> dict    # la note entière : Σ volumes, Σ masses majorées, leur rapport (ajout du site)
SA.ajouter_famille(p, "principal" | "renfort" | "lateral" | "talon"), SA.ajouter_etrier(p), SA.ajouter_type(p),
SA.changer_forme(p, forme), SA.changer_portee(p, L), SA.longueur_en_metres(p, f, "de" | "a", m), SA.changer_retours(f, r),
SA.supprimer_type(p, i), SA.renommer_type(p, i, nom)
```

**Entrée** (une poutre) : `nom`, `forme` (`rectangulaire`, `talon_gauche`, `talon_droit`, `deux_talons`), `enrobage` (cm),
`L` (m), `b`, `h` (cm), `talon_gauche` et `talon_droit` (`b`, `h` en cm), `longueurs` (`pourcent` | `metre`), `acier`
(`B500A`, `B500B`, `B500C`), `majoration` (%), `familles` (`role`, `position`, `n`, `d`, `retours`, `h_retour`, `de`, `a`),
`types_etriers` (`nom`, `d`), `zones` (`nom`, `de`, `a`, `s`, `type`).

**Sortie** de `calculer` (JSON strict, nombres non arrondis) :

- `geometrie` : `b`, `h`, `c`, `L`, `bg`, `hg`, `bd`, `hd` (talons CALCULÉS), `aire_cm2`, `polygone`, `talons_ignores` ;
- `familles`, une par famille, dans l'ordre : `index`, `code` (INF1, SUP1, LAT1…), `role`, `position`, `libelle_role`,
  `libelle_position`, `groupe` (bas, haut, cote), `rang`, `couleur` (le dégradé de l'écran), `libelle` (« 3Ø16 »,
  « 2 × 2Ø12 »), `n`, `n_eff`, `d`, `retours`, `h_retour`, `de`, `a` (%), `de_m`, `a_m` (m), `lit` (rang du lit à sa
  position), `L_active`, `L_tot` (m), `m_lin` (kg/m), `volume` (m³), `masse_majoree` (kg, la colonne « kg tot »), `part`
  (du volume d'acier), `apport` (masse majorée / Vc, kg/m³ — ajout du site), `As` (mm²), `barres` (les centres dessinés,
  cm), `formules` ;
- `etriers`, une ligne par zone calculée : zone, type, `n`, `L_un` (un étrier), `L_tot`, `m_lin`, `volume`,
  `masse_majoree`, `part`, `apport`, `quotient` (L_zone / s avant troncature), `formules` ; `zones_ignorees` ;
- `quantites` : `aire_cm2`, `Vc` (m³), `masse_beton` (kg), `V_longitudinales`, `V_etriers`, `V_acier` (m³), `masse_acier`,
  `kg_m3` (sans majoration : calculés par l'écran, non affichés), `majoration`, `masse_acier_majoree`, `kg_m3_majore`
  (les trois chiffres de l'écran), `par_categorie` (m³ : principal, renfort, latéral, étriers), `masses_categories`
  (kg majorés), `formules` ;
- `avertissements` : `code`, `message`, `grave` — ce que l'écran tait ; ils ne changent aucun chiffre.

## 3 · La méthode du module (reproduite telle quelle)

| Grandeur | Expression du module (fonction) |
|---|---|
| aire de la section | A_c = b·h + b_t,g·h_t,g + b_t,d·h_t,d (cm², `area_cm2`) — talons lus par `build_geometry` |
| volume de béton | V_c = A_c / 10 000 · L (m³) ; poids du béton = V_c · 2 500 kg/m³ |
| longueur active | L_active = L · (À − De) / 100, À et De bornés à 0–100 % ; 0 si À ≤ De (`active_length_m`) |
| barres effectives | n_eff = n, ou 2·n pour « Latérales 2 côtés » (`effective_n_bars`) |
| acier d'une famille | L_tot = L_active · n_eff + retours · h_retour / 100 · n_eff ; 0 si L_active = 0 (`total_bar_length_m`) |
| masse linéique | m_lin = π · (Ø / 2 000)² · 7 850 kg/m (`steel_linear_mass_kg_m`) |
| un étrier | L_étrier = (2 · ((b − 2c) + (h − 2c)) · 10 + 2 · 8 · Ø) / 1 000 m : le contour de l'ÂME à l'enrobage + deux crochets de 8 Ø (`stirrup_length_m`) |
| étriers d'une zone | n = int(L_zone / s) + 1, **tronqué** (`compute_stirrups_for_zones`) ; L_tot = n · L_étrier |
| volume d'acier | V = L_tot · π · (Ø / 2 000)², sommé familles + zones (`compute_quantities`) |
| masse et taux | M_s = V_s · 7 850 ; M_s,maj = M_s · (1 + maj / 100) ; taux = M_s,maj / V_c (`show`) |
| colonne « kg tot » | L_tot · m_lin · (1 + maj / 100), ligne par ligne (`show`) |
| position des barres | lits : axe à c + Ø/2 de la face, lits espacés de Ø + 2 cm, barres de x = −b/2 + c à b/2 − c (une seule : au milieu) ; latérales : n + 2 barres de c à h − c, extrêmes retirées (`bar_centers`) |
| couleurs, codes | dégradé #ffcccc → #b30000 (bas), #cce0ff → #0033cc (haut), #666666 (latérales) ; INF1, SUP1, LAT1 numérotés par groupe dans l'ordre de la liste |

La « qualité d'acier » (B500A/B/C) n'entre dans **aucun** calcul (comme l'écran) ; le site l'écrit sur la note.

## 4 · La preuve de parité (`tests/test_taux_moteur.py`, 127 contrôles au vert)

1. **Données lues dans le module** : libellés des positions et rôles, menus (formes, Ø, retours, aciers), bornes des 17
   champs à bornes fixes (lues dans l'arbre syntaxique de `modules/taux_armature.py`), majoration par défaut, masses
   volumiques.
2. **À la main** : trois poutres recalculées (1,080 m³, 14 étriers de 1,688 m, 29,7 kg et 27,5 kg/m³ — les chiffres
   affichés par l'écran ; une section à talon de 2 300 cm² ; les latérales à mi-hauteur et aux quarts).
3. **L'écran ENTIER rejoué** (`show`) sur **60 poutres** (30 choisies, 30 tirées au sort, graine fixe) avec un faux
   `st` qui enregistre (colonnes et conteneurs compris) et des espions sur `build_geometry`, `build_rebar_families`,
   `compute_stirrups_for_zones`, `compute_quantities`, `draw_section` : géométrie, polygone, aire, **chaque barre
   dessinée** (x, y, Ø, famille, couleur), codes, longueurs, étriers, volumes par catégorie, masses, taux — **6 469
   comparaisons numériques, toutes identiques au bit près** — puis les **chaînes affichées** : les trois métriques, la
   synthèse, chaque cellule du tableau récapitulatif. Le dessin matplotlib de l'écran tourne sur 4 cas.
4. **Couverture** : chaque forme, position, rôle, Ø, nombre de retours, mode de saisie, rang de lit, zone ignorée,
   avertissement, majoration 0 et 50 % est atteint.
5. **Effets de bord** de `saisie.py` contre l'écran : forme changée (6 transitions), « Retours » = 0, type supprimé
   (3 cas), type renommé, boutons d'ajout, portée changée en mètres.
6. **Constats sur l'écran réel** (AppTest de Streamlit) : chaque défaut D1–D8 ci-dessous est MONTRÉ ; si l'écran est
   corrigé, le constat rougit.
7. **Mutations** : 14 altérations plausibles du moteur (masse volumique, latérales comptées une fois, retours oubliés,
   crochets de 10 Ø, arrondi au lieu de troncature, deux talons comptés, pas des lits, couleurs, codes, rangs de lit,
   borne À ≤ De…) font chacune rougir la parité (de 1 à 60 cas).
8. **Format** : JSON strict, normalisation idempotente, 39 saisies hors domaine refusées par ValueError, **1 652
   formules** « symbole = valeurs = résultat » recalculées à partir de leurs seuls termes, chaque grandeur de sortie
   nommée ici et dans la docstring.
9. **Pureté** : le moteur se charge et calcule dans un sous-processus où `sys.meta_path` refuse streamlit, pandas, numpy,
   matplotlib… (Python 3.12 ; 3.13 rejoué quand il est présent).

## 5 · Les défauts de l'écran Streamlit (reproduits par le moteur, ÉCRITS ici et montrés sur le site)

| | Défaut | Effet | Sur le site |
|---|---|---|---|
| **D1** | la synthèse affiche « Poids béton (≈{CONCRETE_DENSITY_KG_M3:.0f} kg/m³) » (chaîne sans `f`) | libellé illisible, chiffre juste | libellé « Poids du béton », 2 500 kg/m³ dans les hypothèses |
| **D2** | « Rectangulaire + deux talons » ne contient ni « gauche » ni « droit » : `build_geometry` calcule la section **sans talon** alors que l'écran montre quatre champs de talon | volume et poids du béton **sous-estimés**, taux en kg/m³ **surestimé** (côté sûr pour un prix d'acier, faux pour un volume de béton) | reproduit ; message grave dans les données et dans le résultat ; talons en fantôme sur la coupe |
| **D3** | « deux talons » + « + Renfort talon » (ou une position de talon) : l'écran s'arrête sur « Talon gauche non défini. » | écran bloqué | refusé avec ce message ; bouton et positions de talon non proposés pour cette forme ; passer à « deux talons » avec une famille sur un talon est refusé |
| **D4** | talon de 0 cm (valeur affichée par défaut) lu `0 or 10.0` = 10 cm (`or 15.0` en hauteur) | le calcul prend 10 × 15 cm alors que le champ affiche 0 | la note neuve porte 10 × 15 (ce que l'écran calcule) ; un talon de 0 est **refusé** avec un message |
| **D5** | `add_family` propose une hauteur de retour h − 2c (b − 2c pour une latérale), aussitôt remise à 0 par `edit_family` (« Retours » = 0) ; passer à 1 retour garde 0 | un retour ajouté pèse 0 tant qu'on n'a pas saisi sa hauteur | reproduit ; avertissement « retour de 0 cm » |
| **D6** | supprimer une famille décale les clés de widgets (par rang) : la suivante hérite des valeurs de la supprimée | la note change sans le dire | sans objet (le site désigne les champs par leur chemin) |
| **D7** | le menu « Section » perd le premier choix qui suit un changement (il faut choisir deux fois) | saisie hésitante | sans objet |
| **D8** | en mètres, une portée changée garde les mètres saisis s'ils tiennent dans la nouvelle portée, sinon reprend le pourcentage | les longueurs changent selon le cas | reproduit (`changer_portee`) et **dit** : les familles qui gardent leur pourcentage sont nommées |
| **D9** | la troncature `int(L_zone / s) + 1` sur des flottants : L = 1 m, 0–30 %, s = 10 cm → 0,3 / 0,1 = 2,9999999999999996 → **3** étriers au lieu de 4 | un étrier de moins (non conservatif sur le quantitatif) | reproduit ; avertissement grave, quotient écrit en entier dans la fenêtre de calcul |
| **D10** | deux zones qui se recouvrent sont comptées deux fois ; une partie de portée sans zone n'a pas d'étrier | double compte ou oubli, sans un mot | reproduit ; avertissements « se recouvrent » et « sans étriers » ; chaîne des zones sur l'élévation |
| **D11** | la longueur d'un étrier ignore les talons (contour de l'âme seule) et prend le contour à l'enrobage, sans le Ø | étriers de talon non comptés | reproduit ; écrit sur la coupe (« contour compté : l'âme à l'enrobage ») et avertissement « les cadres ou épingles du talon ne sont pas comptés » (forme à talon avec étriers) |
| **D12** | « Ø (mm) » d'un étrier écrit `int(phi)` dans le tableau : 7,5 s'affiche 7 | affichage seulement (le calcul prend 7,5) | le site écrit 7,5 |
| **D13** | une zone sans type, ou dont À ≤ De, est ignorée en silence ; une famille À ≤ De pèse 0 en silence | quantités manquantes sans un mot | reproduit ; avertissements graves et bilan rouge sous la zone |
| **D14** | deux types d'étriers du même nom : le calcul prend le DERNIER, le menu des zones le premier | ambiguïté | refusé (impossible à l'écran, qui ne crée qu'un type) |
| **D15** | 2·c ≥ b ou 2·c ≥ h : la longueur d'un étrier devient négative | masse négative | refusé |

Les barres sont placées axes à c (et non c + Ø étrier + Ø/2) des faces latérales : les barres d'angle sont centrées sur
le contour d'étrier compté. C'est la convention de dessin du module (aucun effet sur les quantités) ; la coupe du site
la reproduit.

## 6 · Le module du site : tout ce que l'écran Streamlit offre, coché

| Écran Streamlit | Site |
|---|---|
| « Saisie des longueurs de lit » (% / m) | ✓ segments « % de L / mètres » (section 2) ; les champs « de », « à » passent en m |
| Section (4 formes), enrobage, portée, largeur, hauteur | ✓ section 1, mêmes bornes (refus hors bornes) |
| talons gauche / droit (largeur, hauteur) selon la forme | ✓ affichés selon la forme ; deux talons : message D2 |
| boutons + Lit principal, + Renfort local, + Latéral, + Renfort talon (grisé sans talon), + Étrier | ✓ « Lit principal », « Renfort local », « Barres latérales », « Renfort de talon » (si talon), « Zone d'étriers » (crée E1 s'il n'y a aucun type), et « Type d'étrier » (ajout du site) |
| carte d'une famille : pastille de couleur, rôle – code, poubelle | ✓ pastille, code, rôle, masse de la famille, ✕ |
| position (menu selon rôle et forme), n barres (1–20), Ø (10 valeurs), retours (0–2), hauteur de retour (« — » sans retour), De, À | ✓ mêmes listes et bornes ; hauteur verrouillée à 0 sans retour |
| familles groupées : lits principaux, renforts (y compris talons), latérales | ✓ mêmes groupes |
| types d'étriers : nom, Ø (6–16), poubelle | ✓ nom, Ø en menu (6 à 16 ; une valeur hors pas, 7,5, reste proposée), ✕ |
| zones : nom, De, À, espacement (2–100 cm), type ; « Crée au moins un type… » | ✓ + bilan de la zone (n × Ø, kg) ; message si aucun type |
| Qualité d'acier, Majoration globale (%) | ✓ section 4 « Quantitatif » |
| métriques : Volume béton, Masse acier, Taux d'armature | ✓ verdict (taux en kg/m³) + synthèse |
| Synthèse : masse majorée, volume, poids béton, taux | ✓ (+ taux sans majoration, calculé par l'écran mais non affiché) |
| Tableau récapitulatif (couleur, code, position, n, Ø, L active, L tot, kg/m, kg tot), « Ajoute des armatures… » | ✓ vue « Tableau », mêmes colonnes et précisions, ligne Total (ajout) ; message si vide |
| « Section générale (tous lits) » (matplotlib) | ✓ vue « Coupe » : béton gris, barres aux positions et couleurs de l'écran, contour d'étrier compté, cotes, légende |

**Ajouts du site** (rien de neuf dans le calcul : des projections de ce que le moteur calcule) : plusieurs poutres par
note, résumé et vue « Note » (taux de la note) ; vue « Élévation » (longueurs à l'échelle, hauteurs amplifiées et dites au-delà d'un élancement de 6 ; barres de De à À, étriers à leur espacement, chaîne
des zones, barres développées) ; part de chaque famille et zone dans l'acier et apport au taux ; fenêtres de calcul avec
formules substituées ; sélecteur de Ø par famille (taux de la poutre et part de la famille pour chaque Ø, la retenue ✓ ;
**pas d'optimal ★** : un quantitatif n'a pas de critère « qui tient ») ; avertissements de ce que l'écran tait.

**Non couvert** : la note PDF (hors périmètre du site) ; les clés de widgets de l'écran (D6, D7 : sans objet) ; le
« type » d'étrier (`full`, `inner`, …) que l'écran garde sans l'utiliser. Aucune borne normative (A_s,min, A_s,max,
ratios usuels) : l'écran n'en a pas, le site n'en invente pas.

## 7 · Les règles de saisie du site

- une saisie hors du domaine de l'écran est **refusée** avec un message et la note reste intacte (jamais ramenée en
  silence : talon de 0, Ø non proposé, 21 barres, 2,5 barres, espacement de 1 cm, enrobage qui ne laisse pas de place…) ;
- réécrire la valeur affichée par un champ ne change rien (un pourcentage saisi en mètres garde ses décimales) ;
- les effets de bord de l'écran sont reproduits (`beton/taux/saisie.py`) et **dits** quand l'écran les tait (familles
  ramenées sur l'âme, zones réaffectées, pourcentages gardés en mètres).

## 8 · Relecture indépendante du 07/11/2026

Ce qui a été contrôlé, et comment :

- **Chiffres** : 25 notes construites **par le routeur**, saisie par saisie comme la page les envoie (forme, talons,
  mètres, portée changée, types supprimés, troncature, deux talons, 0,5 m et 60 m, 200 × 300, 12 × 10, Ø6 à Ø40,
  majorations 0 et 50 %…), puis l'écran Streamlit rejoué sur le même état : taux, Vc, masse, poids du béton et chaque
  cellule du tableau récapitulatif **identiques** (25 sur 25). D8 (portée changée en mètres) est prouvé contre le vrai
  Streamlit par AppTest (§4.6).
- **Robustesse** : 17 champs × 16 saisies absurdes (négatif, zéro, texte, 1e9, nan, inf, vide, null, booléen, liste,
  objet, « 0x10 », texte piégé…), 27 notes `.stab` abîmées, noms piégés dans la poutre, les zones et les types : refus
  lisibles, note intacte, aucun élément actif dans les fragments (lu par un analyseur HTML).
- **Écran** : PC, tablette, téléphone, sur une note remplie de deux et trois poutres (talon gauche, talon droit, deux
  talons, troncature), chaque vue et chaque fenêtre regardée.

Ce qui a été corrigé :

| Défaut trouvé | Correction | Test qui rougit si on la retire |
|---|---|---|
| coupe « deux talons » : la note « talons non comptés (défaut D2) » était écrite **sur** la cote de l'âme | la note passe sous la chaîne de cotes | « aucune écriture sur une autre » (coupe et élévation, chaque cas) |
| verdict au téléphone : « 1,380 » et « m³ » sur deux lignes ; au PC « 5 % » seul sur une ligne | deux lignes fixes, chaque nombre lié à son unité par une espace insécable | « le verdict ne sépare jamais un nombre de son unité » |
| sélecteur : le « Ø » du sous-titre seul sur sa ligne | espace insécable | « le Ø du sous-titre reste avec son mot » |
| D11 (cadres de talon non comptés) dit seulement dans la légende de la coupe | avertissement du moteur `etriers_talon` (aucun chiffre ne change) | « D11 dit » ; couverture de la parité |
| `.stab` dont `actif` n'est pas un nombre : erreur Python en anglais (« invalid literal for int() ») | refus en français | « actif = … : refusé en français » |
| tableau au téléphone : la position coupée (« Armatures inférieures… ») | la position se lit sur deux lignes | (feuille de style ; vu sur capture) |
| élévation d'une poutre courante (6 m × 60 cm) : une bande de 65 px, étriers à 11 px | hauteurs amplifiées dès un élancement de 6 (×2 ici), écrit sur le dessin | « élancement : hauteurs ×1 jusqu'à L/h = 6… » |
| défauts non conservatifs écrits dans le tableau du §5 seulement | écrits **en tête** de ce document, avec la fréquence de D9 mesurée (5,1 % des zones ordinaires) | — |

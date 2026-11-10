# Taux d'armature — le moteur et le module du site

*Refait le 10/11/2026 (Lot Q) sur la conception `docs/site/LOT_Q_QUATRE_MODULES.md` § 3. Moteur `beton/taux/` :
bibliothèque standard seule, Python 3.12 et 3.13 (donc Pyodide). Il ne reproduit PLUS l'écran Streamlit
`modules/taux_armature.py` (dont les défauts restent relevés dans `docs/site/DEFAUTS_STREAMLIT.md`) : il compte l'acier
d'une poutre selon la méthode ci-dessous, prouvée à la main. Module du site : `api_web/taux.py` (adresse
`taux-armature`), dessins `api_web/dessin_taux.py`, feuille `site/web/css/modules/taux.css`.*

Le module est un **QUANTITATIF** : il ne vérifie rien. Il donne la masse d'acier à commander d'une poutre, son
**bordereau** (un repère par ligne de barres et par ligne d'étriers, la synthèse par diamètre), le **taux en kg/m³**
(masse majorée / volume de béton) et le **ratio en kg par mètre** de poutre. La note du site est une « Consultation ».
Un chiffre faux par défaut, c'est de l'acier qui manque sur chantier : **dans le doute sur une longueur, le moteur prend
le côté qui commande le plus d'acier**, et la règle est écrite (ici, dans le pied du résultat et dans les fenêtres).

## 1 · Les fichiers

| Fichier | Rôle |
|---|---|
| `beton/taux/entree.py` | la poutre TYPE d'une note neuve ; les listes (formes, familles, Ø) ; les bornes ; `normaliser` (strict : `ValueError` en français qui nomme le champ, jamais une valeur ramenée à une borne) |
| `beton/taux/geometrie.py` | la section (âme et talons BAS, à l'extérieur de l'âme), son aire, son polygone ; le niveau des lits ; la position des barres, des cadres, des épingles et du cadre de talon pour la coupe |
| `beton/taux/moteur.py` | `calculer(poutre)` : les longueurs, `compter_etriers` (LA règle des étriers), le bordereau, les synthèses, le taux, les avertissements ; `synthese_note`, `bordereau_csv` — la MÉTHODE et le FORMAT de sortie dans sa docstring |
| `beton/taux/saisie.py` | les gestes de l'écran : ajouter une ligne de barres, un type, une ligne, une zone (la suite) ; la limite de deux zones jointives qui suit ; la longueur qui fait suivre la moitié droite ; le type employé qui ne se retire pas |
| `beton/taux/depuis_poutre.py` | `convertir(poutre_ba, longueurs, date)` : recevoir une poutre dimensionnée dans « Poutres béton armé » (§ 6.2 de la conception) |
| `api_web/taux.py` | l'écran (établi compact) : rubriques, résultats, fenêtres de calcul, `telecharger(etat, "bordereau")` |
| `api_web/dessin_taux.py` | les vues : élévation, coupe(s), bordereau ; deux variantes par dessin (`.tx-large`, `.tx-etroit`) |

## 2 · La poutre

`{nom, forme: rectangulaire | talon_gauche | talon_droit | deux_talons, b, h (cm), L (m), enrobage (cm),
talon_gauche {b, h}, talon_droit {b, h} (cm), barres [...], types_etriers [...], zones [...], majoration (%),
recouvrements {actifs, l_com (m), k (× Ø)}, rho_beton, rho_acier (kg/m³), origine: null | {module, nom, date}}`.

- une **ligne de barres** : `{famille: inf | sup | peau, lit (le rang sur sa face ; sans objet pour la peau), n (par
  face pour la peau), d (mm), de, a (m depuis l'about gauche ; null = jusqu'à l'about ; les deux null = barre FILANTE,
  sinon un RENFORT), retours: 0 | 1 | 2, l_retour (cm ; null = automatique)}` ;
- un **type d'étriers** : `{nom, lignes: [{forme: cadre | cadre_interieur | epingle, d (mm), largeur (cm, cadre
  intérieur)}], cadre_talon}` — ce qu'on pose à chaque cours ; une **zone** : `{de, a (m), s (cm), type}` — où, et à
  quel pas ;
- Ø des barres : 6, 8, 10, 12, 14, 16, 20, 25, 28, 32, 40 ; des étriers : 6 à 20. Bornes larges mais réelles
  (b, h de 10 à 300 cm ; L de 0,5 à 60 m ; n de 1 à 40 ; s de 5 à 50 cm ; l'enrobage laisse un cadre).
- **Défaut à l'ouverture** (la poutre type, § 3.5) : 30 × 60 cm, L = 6,00 m, c = 3 cm ; 3 Ø 20 inférieures filantes,
  renfort 2 Ø 16 de 1,20 à 4,80 m (second lit), 2 Ø 12 supérieures filantes ; cadres Ø 8 au pas de 15 cm sur 1,50 m à
  chaque about, 25 cm en travée ; majoration 5 %.

## 3 · La méthode (et ses sources)

- **Béton** : A_c = b·h + Σ b_t·h_t (les talons de la forme) ; V_c = A_c·L ; M_c = ρ_c·V_c (ρ_c = 2 500 kg/m³).
- **Masse linéique** : m = π·Ø²/4·ρ_s, ρ_s = 7 850 kg/m³ — les masses nominales de l'**EN 10080** à moins de 0,5 %
  (Ø 8 : 0,395 ; Ø 12 : 0,888 ; Ø 16 : 1,578 ; Ø 20 : 2,466 ; Ø 25 : 3,853 ; Ø 32 : 6,313 kg/m).
- **Une ligne de barres** : x₁ = max(de ; c), x₂ = min(à ; L − c) — bornée aux abouts à l'enrobage, une filante
  mesure **L − 2c** ; l = (x₂ − x₁) + r·l_r + n_rec·l_rec. Un retour automatique vaut toute la hauteur entre enrobages
  (h − 2c ; b − 2c pour la peau) : le côté sûr. Barres de peau : n par face, deux faces.
- **Recouvrements** (option, active par défaut) : au-delà de la longueur commerciale l_com (12 m), n_rec =
  ⌈(l₀ − l_com)/(l_com − l_rec)⌉ recouvrements de l_rec = k·Ø (k = 50). **Écart voulu avec la conception**, qui écrivait
  ⌈l₀/l_com⌉ − 1 : cette formule en oublie un quand l₀ approche d'un multiple de l_com (l₀ = 23,9 m, Ø 20 : deux barres
  de 12 m qui se recouvrent de 1 m ne couvrent que 23 m) ; la formule retenue est celle de la POSE (chaque recouvrement
  mange l_rec sur la barre suivante), prouvée par simulation dans `tests/test_taux_moteur.py` § 2.
- **Étriers** (longueur développée, sans déduction des cintrages — côté sûr de quelques centimètres) :
  cadre 2·[(b − 2c) + (h − 2c)] + 2·l_r ; cadre intérieur 2·[w + (h − 2c)] + 2·l_r ; épingle (h − 2c) + 2·l_r ;
  cadre de talon 2·[(b_tot − 2c) + (h_t − 2c)] + 2·l_r ; **l_r = max(10·Ø ; 70 mm)** — **EN 1992-1-1:2004 §8.5(2),
  figure 8.5** (ancrage d'une armature transversale par un coude à 90° ; le crochet ne demanderait que 5 Ø ≥ 50 mm :
  10 Ø est le côté sûr).
- **Nombre de cours** d'une zone l_z au pas s : ⌈l_z/s − 10⁻⁹⌉ (le 10⁻⁹ absorbe les flottants : 0,3/0,1 =
  2,9999999999999996 compte 3 intervalles), **plus UN cours de fermeture par suite de zones jointives** — la pose réelle
  « un étrier à chaque bout de la suite, un tous les s au plus », un seul étrier à la jonction de deux zones. Écrit une
  fois, dans `moteur.compter_etriers`. Zones qui se recouvrent : refusées (les deux zones nommées).
- **Quantités** : masse par repère, par famille (inférieures, supérieures, renforts, peau, étriers), par diamètre ;
  M_s ; M_s,maj = M_s·(1 + majoration) ; **taux = M_s,maj / V_c** (et sans majoration) ; ratio = M_s,maj / L.
- **Le niveau des lits** (coupe et élévation seulement, aucune longueur n'en dépend) : le premier lit sur le cadre
  (axe à c + Ø_cadre + Ø/2), le suivant au-dessus à l'espacement libre minimal de l'EN 1992-1-1 §8.2(2).

## 4 · Recevoir une poutre de « Poutres béton armé » (`depuis_poutre.convertir`)

Chaque section de longueur l_i devient un tronçon [x_i ; x_i + l_i] (L = Σ l_i) ; pour chaque lit (face, rang) de même
Ø partout : les barres COMMUNES à toutes les sections deviennent filantes, le surplus de chaque section un renfort sur
son tronçon (tronçons voisins de même surplus réunis) ; un Ø qui change donne des barres par tronçon ; les barres
latérales de même ; chaque composition d'étriers distincte devient un type (cadre de toute la largeur → cadre ; cadre
entre deux barres → cadre intérieur de la largeur que la poutre calcule ; épingle), chaque section une zone à son pas.
La largeur d'un cadre intérieur et la position des barres sont APPELÉES dans `beton/poutre/geometrie.py`. Sans
longueur de section : refus, qui dit où la saisir. L'`origine` (module, nom, date) est gardée ; l'écran l'écrit dans
l'en-tête de chaque vue (« Reçue de la poutre V1, le 10/11/2026 ») et en tête du résultat.

## 5 · L'écran (`api_web/taux.py`)

- **Rubriques** : 1 Poutre (la forme en pictogrammes ; b × h ; L et c ; les talons de la forme) · 2 Armatures
  longitudinales (une carte compacte par ligne, rangée sous sa face : repère, n, Ø, « filant · L1 » ou « 1,20 →
  4,80 m » ; dépliée : famille, lit, de, à — vides = filant —, retours et leur longueur ; « + Lit inférieur »,
  « + Lit supérieur », « + Renfort », « + Barres de peau ») · 3 Étriers (les types : « E1 · cadre Ø 8 + … · 33 cours »,
  leurs lignes, « + Ligne », « cadre de talon » ; les zones : « Z1 · 0,00 → 1,50 m · e 15 · E1 · 10 étriers », leur
  quotient ; « + Zone » propose la suite) · 4 Hypothèses (majoration, recouvrements, masses volumiques).
- **Résultats** : le verdict de valeur (taux majoré, la masse majorée pour le volume, le taux sans majoration) ;
  « Acier par famille » et « Par diamètre » (masse, part en jauge GRISE — une part, pas un taux de travail) ; « Béton » ;
  « Note entière » dès deux poutres (`synthese_note`) ; les avertissements du moteur ; la méthode ; au pied, le ratio
  au mètre et « Bordereau CSV ».
- **Fenêtres** : repères et taux, par diamètre, béton, étriers par zone — chaque ligne « symbole = expression =
  valeurs = résultat » écrite à partir des seuls termes du moteur.
- Ce qui SUIT une saisie (la zone jointive d'une limite déplacée, la moitié droite quand L change) est DIT au-dessus du
  résultat. Une note de l'ancien module (version 1) se rouvre : noms gardés, poutres types, avis dit une fois.

## 6 · Ce qui n'est pas couvert (dit, jamais tu)

- aucune vérification (ni section d'acier requise, ni espacements, ni ancrages) : c'est le rôle de « Poutres béton
  armé » ; les épingles inutiles, les types inemployés, une portion sans étriers sont des AVERTISSEMENTS ;
- les ligatures et les chutes ne sont pas détaillées : une majoration globale (5 % par défaut) ;
- les crochets des barres longitudinales, les coupleurs, les aciers d'attente, les barres en attente des appuis ;
- un cadre plus long que la longueur commerciale est dit (avertissement), ses recouvrements ne sont pas comptés ;
- une poutre droite, de section constante sur toute sa longueur (pas d'inertie variable, pas de biais) ; une poutre
  reçue dont les sections changent de b ou de h n'existe pas (la poutre BA a une seule section de béton).

## 7 · Les preuves

| Suite | Contrôles | Ce qu'elle prouve |
|---|---|---|
| `tests/test_taux_moteur.py` | 30 contrôles, 48 mutations rejetées | la poutre type à la main ; chaque formule ; le compte des étriers sur une grille contre la pose simulée (jamais un de moins ; L = 1 m, zone 0–0,3 m, s = 10 cm : 4) ; talons ; recouvrements ; bordereau = Σ repères = Σ Ø = total ; CSV relu ; refus ; gestes ; `depuis_poutre` (trois poutres, invariant au mètre) ; géométrie ; JSON strict ; bibliothèque standard seule |
| `tests/test_api_web_taux.py` | 67 contrôles, 15 mutations vues rouges | l'écran = le moteur sur dix états (chaque nombre affiché) ; chaque formule des fenêtres réévaluée à partir des valeurs écrites ; refus ; gestes et ce qui suit ; contrat de sélection ; échappement ; CSV ; bout en bout depuis le routeur (« transmettre ») |
| `tests/test_api_web_modules.py taux` | 23 | le contrat commun des modules (parcours, champs, fichiers, éléments) |
| `tests/test_reference_refonte.py taux` | 11 cas | la photo gelée (regel accepté le 10/11/2026) |

**La poutre type, à la main** : 3 Ø 20 × 5,94 m = 17,82 m × 2,466 = 43,95 kg ; 2 Ø 16 × 3,60 m = 7,20 m × 1,578 =
11,36 kg ; 2 Ø 12 × 5,94 m = 11,88 m × 0,888 = 10,55 kg ; cadres : 2·(0,24 + 0,54) + 2 × 0,08 = 1,72 m, 10 + 12 + 10
+ 1 = 33 cours, 56,76 m × 0,395 = 22,40 kg ; M_s = 88,25 kg ; × 1,05 = 92,67 kg ; V_c = 0,30 × 0,60 × 6 = 1,080 m³ ;
**taux = 85,8 kg/m³** (81,7 sans majoration) ; 15,4 kg/m.

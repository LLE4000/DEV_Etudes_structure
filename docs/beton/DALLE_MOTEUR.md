# Dalle béton armé — le moteur sans Streamlit

> **⚠ Défaut NON CONSERVATIF de l'écran Streamlit, reproduit (parité) et signalé à l'écran du site.** Les nappes X et
> Y d'une même face sont calculées **au même niveau** : l'axe automatique ne dépend que de l'enrobage et du Ø de la
> couche, si bien que les deux treillis inférieurs par défaut sont tous deux à 3,5 cm du parement. Posées l'une sur
> l'autre, la nappe du dessus est au moins (Ø_x + Ø_y)/2 = 1,0 cm plus loin : sur une dalle de 20 cm, son d réel est
> 15,5 cm et non 16,5 cm — **A_s,req sous-estimé de 6,5 %** (16,5 / 15,5) et **h_min sous-estimé de 1 cm**. Le moteur
> rend l'avertissement `nappes_confondues` ; le site l'affiche sous le résultat de chaque section concernée et invite à
> saisir l'axe de la nappe du dessus. Le bureau doit trancher (§ 7, défaut 1).
>
> **Révision du 09/11/2026 — tranché par le bureau (chantier `docs/site/DALLES_FUSION.md`, § 4).** Le SITE pose la
> direction secondaire d'une nappe SUR la principale (axe = axe principal + ⌈Ø_p/2⌉₀,₅ + ⌈Ø_s/2⌉₀,₅, les barres en
> contact, la convention d'arrondi du moteur), par le canal `dist` que le moteur honore déjà (`api_web/dalle_positions.py`)
> — le moteur lui-même ne change pas (parité Streamlit intacte). Option de NOTE « nappes empilées » : vraie pour une
> note neuve, fausse pour une note d'avant ou un fichier Streamlit importé (leurs chiffres ne bougent pas) ; l'ingénieur
> peut la cocher sur une ancienne note. Empilée, la nappe n'est plus « confondue » et l'avertissement s'éteint de
> lui-même ; mesuré sur le cas « dalle 22 cm réaliste » : d_inf,x 18,5 → 17,5 cm, A_s,req 270,3 → 285,7 mm²/m (+5,7 %).

*Livré le 07/11/2026. Moteur `beton/dalle/` et base des treillis `beton/treillis.py` : bibliothèque standard seule,
Python 3.12 et 3.13 (donc Pyodide 0.29.5). Extrait de l'écran Streamlit `modules/dalle.py` v2.2 (dalle
BIDIRECTIONNELLE), qui reste la **référence** ; la preuve de parité est `tests/test_dalle_moteur.py`. Le module du site
qui l'emploie est `api_web/dalle.py` (dessins : `api_web/dessin_dalle.py`, tests : `tests/test_api_web_dalle.py`) ;
son écran est l'établi compact des trois modules béton armé, aux places de la poutre (`api_web/plaque_compact.py`,
`api_web/etabli_compact.py` ; `docs/site/REFONTE_ETABLI.md`, « Un seul établi pour les trois modules béton armé »).*

## 1 · Ce qui est livré

| Pièce | Rôle |
|---|---|
| `beton/treillis.py` | la base des treillis soudés (Øl/Øt/el/et), copie TENUE de `modules/treillis.py` (l'écran reste inchangé) : mêmes treillis, mêmes fonctions, mêmes résultats au bit près ; plus `est_canonique` et `fils`. Le module Prédalles l'emploiera |
| `beton/dalle/entree.py` | les listes, bornes et défauts de l'écran, les classes de béton (`beton_classes.json`, lu depuis le paquet), la normalisation stricte d'une dalle |
| `beton/dalle/moteur.py` | les couches, le centre de gravité, le calcul d'une section, les vérifications dans l'ordre de l'écran, les avertissements, le sélecteur de la nappe de base — et le **format complet** (entrée, sortie) dans sa docstring |
| `beton/dalle/fichier.py` | la lecture d'un fichier de l'écran (« 💾 Enregistrer » : `dalle_ba.json`, migration v1 → v2 comprise), avec ce que l'écran y remplace en silence |
| `tests/test_dalle_moteur.py` | données, contrôles à la main, parité (calcul, couches, écran, note PDF, fichier), couverture, sélecteur, fichiers, format, pureté, **mutations** (169 contrôles, 5 s) |

Chaque fonction reprend l'expression de l'écran **dans le même ordre d'opérations** : les flottants sont identiques au
bit près (99,6 % des comparaisons ; le reste est relu dans les pourcentages de l'écran, à 2·10⁻¹⁶ près).

## 2 · L'API

```python
from beton.dalle import moteur as M

M.classes_beton() -> dict                       # beton_classes.json (copie)
M.nouvelle_dalle() -> dict                      # « Dalle 1 », bande 100 × 20 cm, C30/37, B500, enrobage 3, principale Y, γs 1,5
M.nouvelle_section(nom="A") -> dict             # sans sollicitation, un treillis 10/10/100/100 sur chacune des quatre nappes
M.nouvelle_couche(base=True) -> dict            # base : treillis 10/10/100/100 ; renfort (bouton ＋) : 3 Ø12 « n barres »
M.nom_section_suivant(dalle) -> str             # la première lettre libre (A, B, …, Z, AA…)
M.normaliser(dalle) -> dict                     # complétée et contrôlée ; ValueError hors du domaine de l'écran
M.verifier(dalle) -> dict                       # le calcul de toutes les sections
M.candidats_base(dalle, i_section, face, familles=None) -> list   # face = "inf_x" | "sup_x" | "inf_y" | "sup_y"

from beton.dalle import fichier
fichier.lire(payload) -> {"dalles": [...], "gamma_s", "jeu_premier_lit", "remarques": [...]}
```

**Entrée** (cm, kN, kN·m par bande ; Ø et espacements en mm ; une clé absente ou `null` prend le défaut de l'écran) :

```json
{"nom": "D1", "b": 100, "h": 20, "beton": "C30/37", "fyk": 500, "enrobage": 3.0, "dir_principale": "Y",
 "gamma_s": 1.5, "jeu_premier_lit": 0.0,
 "sections": [{"nom": "A", "Mx_inf": 28, "Mx_sup": 12, "My_inf": 35, "My_sup": 9, "V": 60,
               "faces": {"inf_y": {"couches": [{"type": "treillis", "treillis": "10/10/100/100"},
                                               {"type": "n_barres", "n": 3, "d": 12, "dist": 8.0}], "yG": null},
                         "inf_x": {}, "sup_x": {}, "sup_y": {}}}]}
```

Une couche garde ses quatre réglages (treillis, Ø, espacement, nombre) quel que soit son type, comme l'écran : seul le
type dit lequel compte. `dist` (axe imposé, cm) et `yG` (centre de gravité imposé) valent `null` tant que rien n'est
saisi : le calcul prend alors l'automatique.

**Sortie** de `verifier` (JSON strict, nombres non arrondis) : `reference`, `dalle` (normalisée), `materiaux`,
`sections` (chacune : `titre`, sollicitations, `principale`, `statut`, `taux_max`, `gouvernant`, `grandeurs`,
`faces` — la géométrie de chaque couche —, `verifications` dans l'ordre de l'écran), `taux_max`, `statut`,
`gouvernant` (`section`, `nom`, `cle`, `titre`, `taux`), `avertissements` (`enrobage_nul`, `d_utile_negatif`,
`nappes_confondues`). Chaque vérification porte `valeur`, `limite`, `taux` = valeur / limite (le pourcentage du
bandeau de l'écran ÷ 100), `etat`, `motifs` et ses `formules` (`termes` → `resultat`), recalculables depuis leurs seuls
termes (le test le fait pour les 5 478 formules des cas).

## 3 · La méthode du bureau (celle de l'écran, formules de la poutre appliquées à une bande)

Unités : b, h, axes en cm ; Ø et espacements en mm ; M en kN·m et V en kN **par bande** de largeur b ; aires en mm².

1. **Matériaux.** fyd = fyk / γs (γs « coefficient acier ELS », 1,5 par défaut, commun à toutes les dalles de la note) ;
   fctm = 0,30·fck^(2/3) ; α_b et μ (μ_a500 ou μ_a400) lus dans `beton_classes.json` ;
   τ_adm,I = 0,016·fck,cube / 1,05.
2. **Couches.** Treillis : As/m = π·Øl²/4 × 1000 / el (la nomenclature est la seule source) ; barres Ø/e :
   π·Ø²/4 × 1000 / e ; « n barres » : n·π·Ø²/4 / (b/100) — n barres posées dans la bande. Les couches d'une nappe
   **s'additionnent** ; As de la nappe = Σ As/m × b/100.
3. **Axe d'une couche** (automatique) : enrobage + ½ Ø arrondi au 0,5 cm **supérieur** + jeu premier lit (Ø porteur
   pour un treillis) ; un enrobage nul est lu 3,0 cm. Un axe saisi le remplace.
4. **Centre de gravité et hauteur utile.** y_G = Σ As/m·e / Σ As/m (ou la valeur saisie) ; d = h − y_G ; d ≤ 0 rend
   la nappe « nok » ; les formules prennent max(d ; 0,1 cm).
5. **Hauteur.** h_u,min(M) = √(M·10⁶ / (α_b·b·μ)) (b en mm) ; CHAQUE famille sollicitée (inf. X, sup. X, inf. Y,
   sup. Y) demande h_u,min(M) + son y_G, et la **plus exigeante** est retenue (audit C-1 du 15/09/2026 ; à égalité,
   l'ordre inf. X, sup. X, inf. Y, sup. Y) ; sans moment, h_min = y_G de la nappe inférieure X. Conforme si h_min ≤ h ;
   taux h_min / h.
6. **Flexion, nappe par nappe, direction par direction.** As,req = M·10⁶ / (fyd·0,9·d) ; As,min = max(0,26·fctm/fyk·b·h ;
   0,0013·b·h ; 0,25·As,req de la face **opposée de la même direction**) — h et non d, comme la poutre ;
   As,max = 0,04·b·h. Conforme si d > 0, As ≥ max(As,req ; As,min) et As ≤ As,max ; taux max(As,req ; As,min) / As.
7. **Effort tranchant** (v2.1) : une dalle ne reçoit pas d'étriers. τ = V·10³ / (0,75·b·h) ≤ τ_adm,I (le seuil
   « pas besoin d'étriers ») ; vérifié seulement si V > 0 ; taux τ / τ_adm,I.
8. **Statuts.** Section : « nok » si une vérification l'est ; dalle : la même fusion sur ses sections. Toutes les
   comparaisons sont larges (≤) : une valeur pile à la limite est conforme (bornes exactes prouvées).
9. **Ordre** (audit I-3) : hauteur, inf. principale, inf. secondaire, sup. principale, sup. secondaire, tranchant. La
   direction principale (Y par défaut) n'entre dans **aucune** formule : elle ordonne l'affichage, la note, les dessins.

## 4 · D'où vient chaque grandeur

`grandeurs` d'une section → `modules/dalle.py` (`r` = le dict rendu par `_dimensionnement_compute_states`, `D` =
`r["dirs"][dk]` ; `<f>` = `inf_x`, `sup_x`, `inf_y`, `sup_y`).

| Grandeur | Sens | Dans l'écran |
|---|---|---|
| `beton`, `fck`, `fck_cube`, `alpha_b`, `mu` | classe et données de `beton_classes.json` | `r["beton"]`, `beton_data[beton]`, `r["alpha_b"]`, `r["mu_val"]` |
| `fctm` | 0,30·fck^(2/3) | `r["fctm"]` |
| `fyk`, `gamma_s`, `fyd` | nuance, coefficient acier, fyk / γs | `r["fyk"]`, `r["gamma_s"]`, `r["fyd"]` |
| `tau_1`, `tau_2`, `tau_4` | τ_adm,I, II, IV (seul I est employé) | `r["tau_1"]`, `r["tau_2"]`, `r["tau_4"]` |
| `b`, `h` | largeur de bande, épaisseur (cm) | `r["b"]`, `r["h"]` |
| `enrobage`, `enrobage_calcul` | saisi ; lu par le calcul (0 → 3,0) | `KD("enrobage_beton")`, `_auto_dist_couche` (« or 3.0 ») |
| `jeu_premier_lit` | jeu ajouté à l'axe automatique | `jeu_enrobage_cm` (paramètre global) |
| `Mx_inf`, `Mx_sup`, `My_inf`, `My_sup`, `V` | sollicitations de la section | `KS("Mx_inf")`… ; `r["V_val"]` |
| `principale` | direction principale (« x » / « y ») | `r["principale"]` |
| `As_pm_<f>` | Σ As/m des couches | `D["As_inf_pm"]`, `D["As_sup_pm"]` |
| `As_<f>` | As de la nappe sur la bande | `D["As_inf_total"]`, `D["As_sup_total"]` |
| `yG_calcule_<f>` | Σ As·e / Σ As | `_layers_geometry(…, use_manual=False)` |
| `yG_<f>` | y_G retenu (saisi ou calculé) | `D["e_cdg_inf"]`, `D["e_cdg_sup"]` |
| `yG_impose_<f>` | y_G saisi ? | `_ycdg_manual` |
| `d_<f>` | h − y_G | `D["d_utile_inf"]`, `D["d_utile_sup"]` |
| `d_calc_<f>` | max(d ; 0,1) | `d_calc_inf`, `d_calc_sup` (locales) |
| `geometrie_<f>_ok` | d > 0 | `D["geom_inf_ok"]`, `D["geom_sup_ok"]` |
| `M_<f>` | le moment de la nappe | `D["M_inf_val"]`, `D["M_sup_val"]` |
| `As_req_<f>` | M / (fyd·0,9·d) | `D["As_formule_*"]` = `D["As_req_*_final"]` |
| `As_min_ec`, `As_min_plancher`, `As_min_base` | 0,26·fctm/fyk·b·h ; 0,0013·b·h ; leur max | `r["As_min_ec"]`, `r["As_min_plancher"]`, `As_min_base` (local) |
| `As_min_opposee_<f>` | 0,25·As,req de la face opposée, même direction | `0.25 * As_formule_*` (local) |
| `As_min_<f>` | As,min retenu | `D["As_min_inf_eff"]`, `D["As_min_sup_eff"]` |
| `As_min_terme_<f>` | le terme qui gouverne (`ec`, `plancher`, `opposee`) | nommé par le moteur |
| `besoin_<f>` | max(As,req ; As,min) | `besoin_dim` (`_render_face_armatures`) |
| `etat_<f>` | état de la nappe | `D["etat_inf"]`, `D["etat_sup"]` |
| `As_max` | 0,04·b·h | `r["As_max"]` |
| `face_hauteur`, `M_hauteur`, `yG_hauteur` | la famille la plus exigeante | `r["M_max"]`, `r["e_cdg_gov"]` (la face est nommée par le moteur) |
| `hu_min`, `h_min`, `etat_hauteur` | h_u,min ; h_u,min + y_G ; état | `r["hmin_calc"]`, `r["h_min_dalle"]`, `r["etat_h"]` |
| `etat_inf`, `etat_sup` | fusion des deux directions (les bandeaux de synthèse) | `r["etat_inf"]`, `r["etat_sup"]` |
| `tau`, `tau_adm`, `etat_tau` | τ ; τ_adm,I ; état | `r["tau"]`, `r["tau_adm"]`, `r["etat_tau"]` |

Chaque couche (`faces[<f>]["couches"]`) : `As_pm` = `_couche_as_per_m`, `e` = `_dist_couche_eff`, `e_auto` =
`_auto_dist_couche`, `libelle` = `_couche_label`, `d` = `_couche_diam_mm`, `valeur` (légende de la coupe) =
`_couche_valeur` de la note PDF.

## 5 · La preuve de parité

`tests/test_dalle_moteur.py` charge chaque note dans l'état de l'écran (`st.session_state`, clés `KD` / `KS`), passe
par le début d'un run (`_ensure_global_defaults`, `_init_dalles_if_needed` : défauts, migrations, coercitions) et les
synchronisations des champs « Dist. axe » et « CDG », puis compare au moteur :

- **le calcul** : TOUTES les clés de `_dimensionnement_compute_states` et de ses `dirs` (une clé ajoutée à l'écran
  fait rougir le test tant qu'elle n'est pas comparée) ; chaque couche (`_couche_as_per_m`, `_dist_couche_eff`,
  `_auto_dist_couche`, `_couche_label`, `_couche_diam_mm`) ; chaque nappe (`_layers_geometry`, avec et sans y_G saisi) ;
- **l'écran** : `render_dimensionnement_right` est exécuté avec un faux `st` qui enregistre, et un espion sur
  `open_bloc_left_right` lit le titre, le texte de droite, l'état et le POURCENTAGE EXACT (flottant) de chaque bloc ;
  les deux lignes de la hauteur, la ligne de τ, les lignes Aₛ,req / Aₛ,min / Aₛ,max / Aₛ fourni ET leurs infobulles
  (formules et valeurs), le message « d utile ≤ 0 », les lignes du tableau des couches, l'icône de chaque section et de
  chaque dalle ;
- **la note PDF** : `modules/export_pdf_dalle._compute_section`, seconde copie du calcul dans l'écran ;
- **le fichier** : ce que « 💾 Enregistrer » écrit (`_build_save_payload`), relu par `beton/dalle/fichier.py`,
  redonne la note.

**79 notes, 88 dalles, 135 sections** : 47 cas écrits (défauts, bidirectionnelle complète, principale X, renforts en
« n barres » posés à l'automatique et à 8 cm, audit C-1, barres partout, les cinq bétons, B400, γs 1,0 / 1,15 / 1,5 /
2,0, jeux, enrobages 0 à 20 cm, bandes de 20 à 500 cm, épaisseurs de 5 à 100 cm, τ au-delà, acier insuffisant et
excessif, chaque terme de As,min, y_G saisis, axes saisis au-delà de h, V seul, égalités, chaque famille qui gouverne
la hauteur, treillis hors de la base, espacements et nombres aux bornes, valeurs non entières, plusieurs dalles), 24
tirages à graine fixe, 6 notes tirées à deux dalles et 2 **bornes exactes** construites avec l'écran (τ = τ_adm,I et
h = h_min au flottant près : les comparaisons sont « ≤ » et non « < »). 25 145 comparaisons numériques, 25 043
identiques au bit près, écart maximal 1,7·10⁻¹⁶ (tolérance 10⁻⁹). Un contrôle de **couverture** exige que les cas
atteignent 63 situations (chaque état, chaque motif, chaque terme de As,min, chaque type de couche en base et en
renfort, 1 à 4 couches, chaque famille qui gouverne la hauteur, chaque avertissement, les bornes…).

Aussi : quatre cas recalculés **à la main** (formules en N et mm : les chiffres de `tests/test_dalle_bidir.py`,
707 / 566 / 182 / 242 mm², y_G = 4,86 cm, l'audit C-1 19,6 cm), la base des treillis comparée **fonction par
fonction** à `modules/treillis.py`, le **sélecteur** (6 nappes × 40 candidats repassés par l'écran), les **fichiers**
de l'écran (migration v1, saisies remplacées en silence et relevées, refus), le **format** (JSON strict, 37 saisies
hors domaine refusées, formules recalculées depuis leurs termes, les 53 grandeurs nommées ici et dans la docstring) et
la **pureté** : un sous-processus dont `sys.meta_path` refuse streamlit, pandas, reportlab, numpy, matplotlib, scipy et
`modules/` charge le moteur, calcule les 88 dalles, le sélecteur et un fichier, et ne charge que la bibliothèque
standard — sous Python 3.12 et 3.13, résultats identiques.

**Mutations** (section 9 du test) : 25 altérations de la source du moteur, rechargée sous un autre nom et comparée à ce
que l'écran a rendu — coefficients (0,9 ; 0,25 ; 0,26 ; 0,0013 ; 0,04 ; 0,75 ; exposant de fctm ; fyd), bornes « ≤ »
changées en « < » (τ, hauteur), τ_adm,II au lieu de I, la face opposée de As,min, As ≤ As,max oublié, la règle de
l'audit C-1, b en cm dans h_u,min, l'arrondi du demi-Ø, le jeu, l'enrobage nul, l'axe et le y_G saisis ignorés, la
largeur de bande ignorée (« n barres », As de la nappe), le plancher de d, l'ordre de l'écran, la direction
principale : **chacune fait rougir au moins une note**.

## 6 · Ce que le moteur ne couvre pas (et le site non plus)

- la **note de calcul PDF** (`modules/export_pdf_dalle.py`) : le bouton est grisé sur le site ;
- l'écran Streamlit : **unités d'affichage** (cm / mm, mm² / cm² : le site écrit toujours cm et mm², comme la note),
  **informations du projet** (le site a les siennes : note, chantier, indice), **Réinitialiser** (le site : nouvelle
  note) ;
- les **surépaisseurs** (annoncées « non couvertes » par l'écran lui-même) ;
- l'écran ne vérifie **ni l'espacement maximal des barres** (EC2 §9.3.1.1(3) : 3h ≤ 400 mm en principale, 3,5h ≤ 450 mm
  en secondaire) **ni la section minimale de la direction secondaire** (20 % de la principale, §9.3.1.1(2)) : la
  direction principale n'entre dans aucune formule. Le moteur reproduit l'écran ; ces deux contrôles sont à décider
  par le bureau (ils seraient du calcul NOUVEAU, à valider).

## 7 · Défauts constatés dans l'écran Streamlit (non corrigés : le moteur reproduit, le site les dit)

1. **Nappes X et Y d'une même face calculées au même niveau** — **non conservatif.** L'axe automatique ne dépend que
   de l'enrobage et du Ø de la couche : les nappes inférieures X et Y (deux treillis 10/10/100/100 par défaut) sont
   toutes deux à 3,5 cm, alors que croisées l'une passe sur l'autre (au moins (Øx + Øy)/2 = 1,0 cm plus haut). La nappe
   du dessus a en réalité d = 15,5 cm et non 16,5 : l'As,req réel est 6,5 % plus grand que celui calculé (16,5 / 15,5), et
   h_min est sous-estimé de 1 cm. Le moteur reproduit et
   rend l'avertissement `nappes_confondues` ; le site l'affiche et propose de saisir l'axe. *Révision du 09/11/2026 :
   tranché — le site empile la direction secondaire sur la principale par `dist` (option de note « nappes empilées »,
   vraie pour une note neuve ; voir l'encadré en tête et `docs/site/DALLES_FUSION.md` § 4) ; le moteur reste tel quel.*
2. **Saisies ramenées en silence.** Un moment ou un effort tranchant saisi négatif devient 0 (`max(0, …)` au début de
   chaque run : la nappe passe « ok ») ; un axe (« Dist. axe ») ou un centre de gravité (« CDG ») illisible ou nul est lu
   « automatique » ; une désignation de treillis illisible (ancien fichier) est lue 0 mm²/m avec un Ø de 10 mm ; à
   l'ouverture d'un fichier, une classe de béton inconnue devient C30/37, une nuance autre que 400/500 devient 500, un
   type ou un Ø inconnu le défaut, un espacement ou un nombre hors bornes est borné. Le moteur et le site **refusent**
   ces saisies avec un message ; la lecture d'un fichier les **relève** (`remarques`).
3. **Enrobage nul lu 3,0 cm.** Le champ accepte 0 (`min_value=0.0`), `_auto_dist_couche` lit `enrobage or 3.0` : le
   champ affiche 0, le calcul et la note PDF prennent 3 cm. Reproduit, avec l'avertissement `enrobage_nul`.
4. **Le calcul est écrit deux fois** : `_dimensionnement_compute_states` (écran) et `_compute_section` (note PDF). Elles
   sont aujourd'hui identiques (le test le prouve à chaque passage) ; une correction dans l'une seulement les ferait
   diverger.
5. **Unités cm² partielles** : en « cm² », seul le bandeau d'une nappe change d'unité ; les lignes Aₛ,req, Aₛ,min,
   Aₛ,max et Aₛ fourni restent en mm².
6. Ni « ajouter une dalle » (seulement « copier »), ni suppression de la dalle 1 ou de la section 1 : le site permet
   d'ajouter, copier et supprimer n'importe quel élément (jamais le dernier).
7. **Aucune borne haute** sur les sollicitations (champ texte), le jeu premier lit, l'axe d'une couche ni le centre de
   gravité : 10³⁰⁰ kN·m donne un taux de 10¹⁵⁰ %, 10³⁰⁸ un « inf ». Le site refuse l'absurde (ci-dessous, § 8).

## 8 · Le module du site : ce que la relecture indépendante du 07/11/2026 a vérifié et changé

**Vérifié.** Les chiffres AFFICHÉS par le site (routeur → fragments HTML) contre ceux que l'écran Streamlit AFFICHE
(`render_dimensionnement_right` rejoué avec un faux `st`), sur 23 notes variées (défauts, bidirectionnelle, principale
X, renfort posé à 8 cm, audit C-1, barres partout, B400 et γs 1,15, enrobage nul, bandes de 20 et 500 cm, h = 5 et
100 cm, τ dépassé, **τ et A_s pile à la limite**, acier excessif, A_s,min de la face opposée, y_G saisis, quatre couches,
décimales et jeu, trois sections et deux dalles, V seul, C50/60) : 1 664 comparaisons (h_u,min, h_min, y_G, h,
A_s,req, A_s,min, A_s,max, A_s, A_s/m, τ, τ_adm,I, titres, états, couleurs, pourcentages), **aucun écart** hors la règle
d'arrondi du site qui ne change jamais de bande de couleur (`C.pct`).

**Changé dans le module du site** (le moteur et ses chiffres sont inchangés ; `tests/test_api_web_dalle.py`, § 12) :

- **Le verdict ne nomme plus une vérification conforme quand la dalle ne l'est pas.** Le moteur nomme la plus chargée ;
  une nappe peut être non conforme SOUS 100 % (acier excessif A_s > A_s,max, d utile ≤ 0) pendant qu'une autre, conforme,
  est plus chargée : le verdict affichait alors « hauteur 59 % ≤ … » sur fond rouge. La règle du site (`gouvernante`,
  un CHOIX parmi les vérifications du moteur) : dans une dalle non conforme, la plus chargée des non conformes, avec la
  comparaison qui la rend non conforme (« A_s = 50 265 mm² > A_s,max = 8 000 mm² · acier excessif ») ; le résumé de la
  note, la liste des dalles, les sections et la ligne gouvernante du tableau suivent la même règle. Une ligne non
  conforme sous 100 % dit pourquoi (« Ø40/25 · acier excessif »).
- **Bornes propres au site, contre l'absurde** (l'écran n'en a pas) : une sollicitation au plus 100 000 kN·m ou kN (la
  plus forte bande du domaine, 5 m × 100 cm en C50/60, porte moins de 20 000 kN·m et 4 000 kN) ; le jeu premier lit,
  l'axe d'une couche et un centre de gravité au plus 100 cm (l'épaisseur maximale). Refus avec un message à la saisie,
  erreur claire à la relecture d'un .stab. Le verrou ne lit plus « false » comme vrai ; un chemin introuvable est dit en
  français ; « dalles » qui n'est pas une liste est une note abîmée (refusée) et non une note neuve.
- **La lecture d'un fichier de l'écran** suit `_load_from_payload` quand « dalles » n'est pas une liste (la dalle 1
  neuve) au lieu de lever une erreur de Python (`tests/test_dalle_moteur.py`, § 6).
- **Les dessins** : la hauteur utile cotée dans chaque coupe est celle de la face TENDUE de la direction (la supérieure
  sur appui) et non toujours l'inférieure ; la légende écrit « axe à 35 » (le « e = » se confondait avec l'espacement
  des barres Ø/e).
- **Les formules** écrivent y_G et d au 1/100 de cm (« 13,1 + 3,8 », et non « 13,1 + 3,7951 ») ; les comparaisons du
  verdict ne se coupent plus au milieu d'un membre ; une couche de treillis dit ses fils porteurs « Ø10/100 » ; le champ
  de l'axe d'une couche est nommé « axe », suivi de « auto » ou du bouton « saisi ↺ » (retour à l'automatique).

**Non couvert, écrit ici :** les unités d'affichage de l'écran (cm / mm, mm² / cm² : partielles dans l'écran lui-même,
défaut 5) — le site écrit toujours cm et mm², comme la note PDF ; la note PDF (bouton grisé).

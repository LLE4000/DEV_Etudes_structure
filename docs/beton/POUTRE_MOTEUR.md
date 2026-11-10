# Poutre béton armé — le moteur sans Streamlit

*Livré le 06/11/2026. Moteur `beton/poutre/` : bibliothèque standard seule, Python 3.12 et 3.13
(donc Pyodide 0.29.5). Extrait de l'écran Streamlit `modules/poutre.py` v2.40 (APP_VERSION 1.2.41),
qui reste la **référence** ; la preuve de parité est `tests/test_poutre_moteur.py`.*

## 1 · Ce qui est livré

| Pièce | Rôle |
|---|---|
| `beton/poutre/entree.py` | les listes et les valeurs par défaut du module, les classes de béton (`beton_classes.json`, lu depuis le paquet : `beton/poutre/../../beton_classes.json`), la normalisation stricte d'une poutre |
| `beton/poutre/geometrie.py` | la position de chaque lit (distance parement → axe), le centre de gravité y_G d'une face, les aires, la section des étriers |
| `beton/poutre/moteur.py` | le calcul d'une section, les cinq vérifications de l'écran, l'agrégation, le sélecteur d'armatures — et le **format complet** (entrée, sortie) dans sa docstring |
| `tests/test_poutre_moteur.py` | parité avec le module, contrôles à la main, couverture, sélecteur, format, pureté (122 contrôles, 1,5 s) |

Chaque fonction de `geometrie.py` et de `moteur.py` reprend l'expression du module **dans le même
ordre d'opérations** : les flottants sont identiques au bit près (98 % des comparaisons ; le reste
est relu dans les pourcentages de l'écran, à 3·10⁻¹⁶ près).

## 2 · L'API

```python
from beton.poutre import moteur as M

M.classes_beton() -> dict                       # beton_classes.json (copie)
M.nouvelle_poutre() -> dict                     # « Poutre 1 », 20 × 40, C30/37, B500, γs 1,5, une section « A »
M.nouvelle_section(nom="A") -> dict             # 2Ø16 / 2Ø16, un étrier Ø10, pas 30 cm, sans sollicitation
M.nom_section_suivant(poutre) -> str            # la première lettre libre (A, B, …, Z, AA…)
M.normaliser(poutre) -> dict                    # complétée et contrôlée ; ValueError hors du domaine de l'écran
M.verifier(poutre) -> dict                      # le calcul de toutes les sections
M.candidats_armatures(poutre, nom_section, face) -> list   # face = "inf" | "sup"
```

**Entrée** (cm, kN, kN·m ; Ø en mm ; une clé absente ou `null` prend le défaut du module) :

```json
{"nom": "V1", "b": 30, "h": 50, "beton": "C30/37", "fyk": 500, "gamma_s": 1.5,
 "enrobage": 3.0, "jeu_premier_lit": 0.0, "jeu_entre_lits": 1.5,
 "sections": [{"nom": "A", "M_inf": 122, "M_sup": 0, "V": 60,
               "inf": [{"n": 3, "d": 20}], "sup": [{"n": 2, "d": 16}],
               "etriers": [{"type": "etrier", "d": 8}], "pas": 20,
               "yG_inf": null, "yG_sup": null}]}
```

`M_inf` tend la face inférieure (travée), `M_sup` la face supérieure (appui), en valeur absolue.
Lits : du parement vers le cœur, 1 à 4 par face, Ø parmi 6 8 10 12 16 20 25 32 40. Étriers : une
ligne = un étrier (`"etrier"`, 2 brins) ou une épingle (`"epingle"`, 1 brin), Ø parmi 6 8 10 12 ;
les libellés du module (« Étrier », « Épingle ») sont admis. `yG_inf` / `yG_sup` : le champ
« CDG (cm) » de l'écran (null = calculé).

**Sortie de `verifier`** (JSON strict, nombres non arrondis — le site formate) : `reference`,
`poutre` (normalisée), `materiaux`, `sections`, `taux_max`, `statut` (`"ok"`, `"warn"`, `"nok"` :
les états du module), `gouvernant` (`section`, `cle`, `nom`, `taux`), `avertissements`. Une section
de la sortie, abrégée (la section A ci-dessus) :

```json
{"nom": "A", "M_inf": 122.0, "M_sup": 0.0, "V": 60.0, "statut": "ok",
 "taux_max": 0.9588594102326534, "gouvernant": "inf",
 "grandeurs": {"fyd": 333.3333333333333, "alpha_b": 12.96, "mu": 0.1709, "fctm": 2.896468153816889,
               "fck_cube": 37.0, "d_inf": 45.0, "d_sup": 45.0, "yG_inf": 5.0, "yG_sup": 5.0,
               "hu_min": 42.849499727073365, "h_min": 47.849499727073365,
               "As_req_inf": 903.7037037037037, "As_min_ec": 225.92451599771738, "As_min_plancher": 195.0,
               "As_min_opposee_sup": 225.92592592592592, "As_min_sup": 225.92592592592592,
               "As_min_terme_sup": "opposee", "As_max": 6000.0, "As_inf": 942.4777960769379,
               "tau": 0.5333333333333333, "tau_1": 0.5638095238095238, "bande_tau": "I",
               "Ast": 100.53096491487338, "s_th": 25.132741228718345, "s_max": 30.0,
               "s_lim": 25.132741228718345, "s_th_tronque": 25.1, "…": "70 grandeurs en tout"},
 "verifications": [
   {"cle": "inf", "nom": "Armatures inférieures", "valeur": 903.7037037037037,
    "limite": 942.4777960769379, "unite": "mm²", "taux": 0.9588594102326534, "etat": "ok",
    "motifs": [], "As_max": 6000.0, "libelle": "3Ø20",
    "formules": [{"cle": "As_req", "termes": [{"symbole": "M", "valeur": 122.0, "unite": "kN·m"},
                                              {"symbole": "fyd", "valeur": 333.3333333333333, "unite": "N/mm²"},
                                              {"symbole": "k", "valeur": 0.9, "unite": ""},
                                              {"symbole": "d", "valeur": 45.0, "unite": "cm"}],
                  "resultat": 903.7037037037037, "unite": "mm²"},
                 {"cle": "As_min", "…": "…"}, {"cle": "besoin", "…": "…"}, {"cle": "As_max", "…": "…"}]},
   {"cle": "tau", "nom": "Vérification de l'effort tranchant", "valeur": 0.5333333333333333,
    "limite": 0.5638095238095238, "taux": 0.945945945945946, "etat": "ok", "bande": "I",
    "besoin": "pas_d_etriers", "etat_bande": "ok", "…": "…"},
   {"cle": "hauteur", "…": "…"}, {"cle": "sup", "…": "…"}, {"cle": "pas", "…": "…"}],
 "geometrie": {"b": 30.0, "h": 50.0, "enrobage": 3.0, "enrobage_calcul": 3.0,
               "inf": {"lits": [{"n": 3, "d": 20, "distance": 5.0, "As": 942.4777960769379}],
                       "As": 942.4777960769379, "yG": 5.0, "yG_calcule": 5.0, "libelle": "3Ø20"},
               "sup": {"…": "…"},
               "etriers": {"lignes": [{"type": "etrier", "d": 8, "brins": 2}], "d_max": 8.0,
                           "pas": 20.0, "Ast": 100.53096491487338, "resume": "Étrier Ø8"}}}
```

Les vérifications viennent dans l'ordre des blocs de l'écran : `hauteur`, `inf`, `sup`, puis `tau`
et `pas` si V > 0. `taux` = `valeur` / `limite` est le pourcentage du bandeau divisé par 100.
`formules` donne, pour chaque ligne « formule = valeurs = résultat », les termes dans l'ordre de la
formule avec leur unité : le site écrit la phrase, il ne recalcule rien (le test le prouve en
retrouvant chaque résultat à partir de ses seuls termes). `motifs` dit ce qui rend un état non
conforme : `hauteur_insuffisante`, `d_utile_negatif`, `acier_insuffisant`, `acier_excessif`
(As > As,max), `barres_inclinees` (état warn), `cisaillement_excessif`, `pas_trop_grand`.

**Sortie de `candidats_armatures`** : les 30 dispositions d'UN lit (n de 2 à 6, Ø 10 12 16 20 25 32)
pour la face demandée ; chacune remplace les lits de la face et la section est recalculée (le Ø
déplace le lit, donc d), triées par aire : `n`, `d`, `nom` (« 3Ø20 »), `aire`, `distance`, `yG`,
`taux` (besoin / aire), `etat`, `ok`, `statut_section`, `retenue` (la disposition actuelle),
`optimal` (la plus petite aire « ok »).

**Domaine.** `normaliser` refuse (ValueError, avec le chemin du champ) tout ce que l'écran ne permet
pas : clé inconnue, Ø non proposé (Ø14 par exemple), nuance autre que 400 / 500, classe inconnue,
plus de 4 lits, aucune barre, nombre négatif ou infini, pas ou y_G nul. Le module, lui, remplace
sans rien dire quand il relit un fichier (Ø → 16, nuance → 500, classe → C30/37, Ø d'étrier → 10,
lits ramenés à 4) : le moteur ne reproduit pas ces substitutions, il refuse.

## 3 · La méthode du bureau

Unités du module : b, h, distances en cm ; Ø en mm ; M en kN·m ; V en kN ; aires en mm².

1. **Matériaux.** fyd = fyk / γs (γs = « coefficient acier ELS », 1,5 par défaut) ;
   fctm = 0,30·fck^(2/3) ; α_b et μ (μ_a500 ou μ_a400 selon la nuance) lus dans
   `beton_classes.json` ; τ_adm,I = 0,016·fck,cube/1,05, τ_adm,II = 0,032·fck,cube/1,05,
   τ_adm,IV = 0,064·fck,cube/1,05.
2. **Position des lits** (automatique) : lit 1 = c + arr½(Ø_étrier/10 + Ø₁/20) + jeu premier lit ;
   lit i = lit(i−1) + arr½(Ø(i−1)/20) + jeu entre lits + arr½(Øᵢ/20) — arr½ : arrondi au 0,5 cm
   supérieur ; Ø_étrier = le plus grand Ø des étriers et épingles ; un enrobage c nul est lu 3,0 cm.
3. **Centre de gravité et hauteurs utiles.** y_G = Σ Asᵢ·eᵢ / Σ Asᵢ (ou la valeur imposée) ;
   flexion : d = h − y_G de la face ; cisaillement : d_v = h − min(lit 1 inf. ; lit 1 sup.).
   d ≤ 0 rend la vérification « nok » ; les formules prennent max(d ; 0,1 cm).
4. **Hauteur.** h_u,min(M) = √(M·10⁶ / (α_b·b·μ)) (b en mm) ; parmi les faces sollicitées, la
   famille la PLUS EXIGEANTE est celle de h_u,min(M) + y_G le plus grand (à égalité, l'inférieure) ;
   h_min = h_u,min + y_G ≤ h ; sans moment, h_min = y_G inférieur. Taux h_min / h.
5. **Flexion.** As,req = M / (fyd·0,9·d) ; As,min = max(0,26·fctm/fyk·b·h ; 0,0013·b·h ;
   0,25·As,req de la face opposée) — h et non d, « en cohérence avec le rapport PDF » ;
   As,max = 0,04·b·h. Conforme si d > 0, As ≥ max(As,req ; As,min) et As ≤ As,max.
   Taux max(As,req ; As,min) / As.
6. **Effort tranchant.** τ = V / (0,75·b·h) ; bande I (τ ≤ τ_adm,I : ok, pas besoin d'étriers),
   II (≤ τ_adm,II : ok, étriers), IV (≤ τ_adm,IV : warn, barres inclinées et étriers), au-delà
   (nok, inacceptable). Taux τ / τ_adm de la bande atteinte ; d_v ≤ 0 → nok.
7. **Étriers.** A_st = Σ brins·π·Ø²/4 ; s_th = A_st·fyd·d_v / V ; s_max = min(0,75·d_v ; 30 cm) ;
   s_lim = min(s_th ; s_max). Conforme si pas ≤ s_lim (et d_v > 0). Taux pas / s_lim. L'écran
   affiche s_th et s_max **tronqués** au mm inférieur (recopiables sans changer le verdict) ; la
   comparaison porte sur la valeur exacte.
8. **Statuts.** Section : « nok » si une vérification l'est, sinon « warn » si une l'est, sinon
   « ok » ; poutre : la même fusion sur les sections. Sans effort tranchant, τ et pas sont « ok »
   et ne sont pas affichés. Toutes les comparaisons sont larges (≤) : une valeur pile à la limite
   est conforme.

## 4 · D'où vient chaque grandeur

`grandeurs` d'une section → `modules/poutre.py` (`r` = le dict rendu par
`_dimensionnement_compute_states`).

| Grandeur | Sens | Dans le module |
|---|---|---|
| `beton`, `fck`, `fck_cube`, `alpha_b`, `mu` | classe et données de `beton_classes.json` | `r["beton"]`, `beton_data[beton]`, `r["alpha_b"]`, `r["mu_val"]` |
| `fctm` | 0,30·fck^(2/3) | `r["fctm"]` |
| `fyk`, `gamma_s`, `fyd` | nuance, coefficient acier, fyk/γs | `_get_fyk_and_mu_ref`, `_get_gamma_s`, `r["fyd"]` |
| `tau_1`, `tau_2`, `tau_4` | τ_adm,I, II, IV | `r["tau_1"]`, `r["tau_2"]`, `r["tau_4"]` |
| `b`, `h` | largeur, hauteur (cm) | `r["b"]`, `r["h"]` |
| `enrobage`, `enrobage_calcul` | saisi ; lu par le calcul (0 → 3,0) | `KB("enrobage_beton")`, `_auto_dist_lit` (« or 3.0 ») |
| `d_etrier` | Ø max des étriers (mm) | `_stirrup_diam_mm` |
| `M_inf`, `M_sup`, `V` | sollicitations | `r["M_inf_val"]`, `r["M_sup_val"]`, `r["V_val"]` |
| `dist_lit1_inf`, `dist_lit1_sup` | axe du lit 1 (cm) | `r["dist_l1_inf"]`, `r["dist_l1_sup"]`, `_get_dist_lit` |
| `yG_calcule_inf`, `yG_calcule_sup` | Σ As·e / Σ As | `_layers_geometry(…, use_manual=False)` |
| `yG_inf`, `yG_sup` | y_G retenu (imposé ou calculé) | `r["e_cdg_inf"]`, `r["e_cdg_sup"]` |
| `yG_impose_inf`, `yG_impose_sup` | y_G imposé ? | `_ycdg_manual`, `ycdg_auto_*` |
| `As_inf`, `As_sup` | As réel de la face | `r["As_inf_total"]`, `r["As_sup_total"]` |
| `d_inf`, `d_sup`, `d_cisaillement` | h − y_G ; h − min(lit 1) | `r["d_utile_inf"]`, `r["d_utile_sup"]`, `r["d_utile_shear"]` |
| `d_calc_inf`, `d_calc_sup`, `d_calc_cisaillement` | max(d ; 0,1) | `d_calc_inf`, `d_calc_sup`, `d_calc_shear` (locales) |
| `geometrie_inf_ok`, `geometrie_sup_ok`, `geometrie_cisaillement_ok` | d > 0 | `r["geom_inf_ok"]`, `r["geom_sup_ok"]`, `r["geom_shear_ok"]` |
| `face_hauteur`, `M_hauteur`, `yG_hauteur` | famille la plus exigeante | `r["M_max"]`, `r["e_cdg_gov"]` (la face est nommée par le moteur) |
| `hu_min`, `h_min` | h_u,min ; h_u,min + y_G | `r["hmin_calc"]`, `r["h_min_poutre"]` |
| `As_req_inf`, `As_req_sup` | M / (fyd·0,9·d) | `r["As_formule_*"]` = `r["As_req_*_final"]` |
| `As_min_ec`, `As_min_plancher`, `As_min_base` | 0,26·fctm/fyk·b·h ; 0,0013·b·h ; leur max | `r["As_min_ec"]`, `r["As_min_plancher"]`, `r["As_min_formula"]` |
| `As_min_opposee_inf`, `As_min_opposee_sup` | 0,25·As,req de la face opposée | `0.25 * As_formule_*` (local) |
| `As_min_inf`, `As_min_sup` | As,min retenu | `r["As_min_inf_eff"]`, `r["As_min_sup_eff"]` |
| `As_min_terme_inf`, `As_min_terme_sup` | le terme qui gouverne (`ec`, `plancher`, `opposee`) | nommé par le moteur |
| `besoin_inf`, `besoin_sup` | max(As,req ; As,min) | `besoin_dim` (`_render_face_armatures`) |
| `As_max` | 0,04·b·h | `r["As_max"]` |
| `tau`, `bande_tau`, `tau_lim`, `besoin_tau` | τ et sa bande | **code d'écran** : `_bloc_pas`, `_shear_need_text` |
| `Ast`, `pas` | section des étriers ; pas choisi | `_shear_lines_total_Ast_mm2`, `KS("shear_pas")` |
| `s_th`, `s_max`, `s_lim` | pas théorique, maximal, limite | **code d'écran** : `_bloc_pas` (`pas_th`, `s_max`, `pas_lim`) ; état : `_pas_state` |
| `s_th_tronque`, `s_max_tronque`, `s_lim_tronque` | tronqués au mm inférieur | `pas_th_aff`, `s_max_aff` (v2.41) |

Les états : `etat_h`, `etat_inf`, `etat_sup`, `etat_tau`, `etat_pas`, `etat_global` de `r` ;
`etat_bande` (la couleur du bandeau τ) vient de `_shear_need_text`. La géométrie du dessin :
`_get_dist_lit` pour chaque lit, `_shear_lines_summary` pour le résumé des étriers.

## 5 · La preuve de parité

`tests/test_poutre_moteur.py` charge chaque poutre dans l'état du module (`st.session_state`, clés
`KB` / `KS`), passe par le début d'un run de l'écran (`_ensure_global_defaults`,
`_init_beams_if_needed` : défauts, migrations, coercitions), puis compare au moteur :

- **le calcul** : TOUTES les clés de `_dimensionnement_compute_states` (49, dont les six états ; une
  clé ajoutée au module fait rougir le test tant qu'elle n'est pas comparée), chaque
  `_get_dist_lit`, `_layers_geometry` (y_G calculé), `_shear_lines_total_Ast_mm2`,
  `_stirrup_diam_mm`, `_shear_lines_summary` ;
- **l'écran** : `render_dimensionnement_right` est exécuté avec un faux `st` qui enregistre au lieu
  d'afficher, et un espion sur `open_bloc_left_right` lit le titre, le texte de droite, l'état et le
  POURCENTAGE EXACT (flottant) de chaque bloc. τ et s_lim, que le module ne calcule que dans son
  code d'écran, en sont tirés ; la ligne « τ = … ≤ τ_adm_… → … » donne la bande et le besoin ;
  s_th et s_max tronqués, la ligne « hᵤ,min + d₁ = … » et l'icône d'état de chaque section et de la
  poutre sont relus aussi.

**75 poutres, 99 sections** : 46 cas écrits (défauts, travée seule, appui seul, les deux, moments
nuls, V nul, 1 à 4 lits, Ø 6 à 40, B400 / B500, les cinq bétons, γs 1,15 / 1,35 / 1,5, enrobages 0 à
5 cm, jeux 0 à 2,5 cm, étriers + épingles, y_G imposés, géométrie impossible, sections qui ne
tiennent pas), 24 tirages à graine fixe et 5 **bornes exactes** construites avec le module (pas =
s_lim, h = h_min, τ = τ_adm,I / II / IV au flottant près : elles prouvent que les comparaisons sont
« ≤ » et non « < »). 6 375 comparaisons numériques, 6 266 identiques au bit près, écart maximal
2,75·10⁻¹⁶ (tolérance 10⁻⁹) : les autres sont relues dans les pourcentages de l'écran (×100, ÷100).
Un contrôle de **couverture** exige que les cas atteignent 82 situations (chaque bande de τ, chaque
état, chaque motif, chaque terme de As,min, chaque borne…).

Aussi : deux sections recalculées **à la main** (formules écrites en N et mm dans le test), le
**sélecteur** (6 faces × 30 dispositions repassées par le module), le **format** (JSON strict,
26 saisies refusées, avertissement d'enrobage nul, les 2 019 formules recalculées depuis leurs seuls
termes) et la **pureté** : un sous-processus dont `sys.meta_path` refuse streamlit, pandas, reportlab,
numpy, matplotlib, scipy et pymupdf charge le moteur, calcule les 75 poutres et ne charge que la
bibliothèque standard — sous Python 3.12 et 3.13, avec des résultats identiques.

**Chaque contrôle rougit si l'on retire ce qu'il protège** : 40 mutations volontaires (06/11/2026)
font toutes rougir le test — coefficients et formules (0,9 ; 0,25 ; 0,75 ; 30 cm ; exposant de
fctm ; μ_a400), arrondi ½ cm, jeux, Ø d'étrier, « ≤ » changé en « < » (hauteur, bandes de τ, pas),
famille gouvernante, d du cisaillement, contrôles de d ≤ 0 (flexion, τ, pas), As,max, brins de
l'épingle, fusion des statuts, y_G imposé, enrobage nul, troncature, sélecteur (n, tri, optimal),
import de streamlit, valeurs par défaut, normalisation (moment négatif, classe inconnue), termes
des formules, taux arrondi, gouvernant, doc incomplète, bornes exactes retirées. Une mutation
témoin (renommer un cas) reste verte. Deux trous trouvés ainsi ont été bouchés : avec d ≤ 0, les
formules prennent d = 0,1 cm (s_max = 0,075 cm, As,req immense) et refusent déjà presque tout ; deux
cas (pas de 0,05 cm, face sans moment) isolent les contrôles de d.

## 6 · Ce que le moteur ne couvre pas

- le **taux d'armature** (kg/m³ par section et pour la poutre, longueurs de section, majoration,
  retour d'étrier, arrondi) et la **masse** d'acier ;
- les **armatures technologiques** (barres de peau du dessin PDF) et la **position** « de barre X
  à barre Y » des étriers et épingles (dessin seulement, sans effet sur le calcul) ;
- la **note de calcul PDF** (`modules/export_pdf.py`, qui porte sa propre copie du calcul) ;
- la **zone réduite** (effort tranchant réduit, supprimé du module) et le **dimensionnement des
  barres inclinées** : la bande IV est nommée, rien n'est calculé ;
- l'**ancienne saisie manuelle** des distances de lit (override, retiré de l'écran) et les
  **migrations** des anciens fichiers (second lit, cadres multiples, « 3 brins », M_sup optionnel,
  statut « Validé ») ;
- l'écran : unités d'affichage, verrouillage, informations du projet, enregistrement JSON au format
  du module.

## 7 · Défauts constatés dans le module (non corrigés : le moteur reproduit)

1. **Enrobage nul lu 3,0 cm.** L'écran accepte 0 cm (`min_value=0.0`) mais `_auto_dist_lit` lit
   `float(enrobage or 3.0)`. Reproduit, avec l'avertissement `enrobage_nul` (cas « enrobage nul »).
2. **Bandeau τ sans le contrôle de d.** `_bloc_pas` recalcule l'état de τ par `_shear_need_text`
   sans `geom_shear_ok` : avec d_v ≤ 0 (h = 5 cm, lit 1 à 5,0 cm), le bandeau τ est vert alors que
   la section est rouge. Le moteur rend les deux (`etat` et `etat_bande`).
3. **« ≤ » écrit même quand τ dépasse τ_adm,IV** : « τ = 2.67 N/mm² ≤ τ_adm_IV = 2.26 N/mm² → Pas
   acceptable » (et τ y est écrit avec un point décimal).
4. **d du cisaillement au lit 1** : d_v = h − min(lit 1 inf. ; lit 1 sup.) et non la hauteur utile
   des aciers tendus ; avec plusieurs lits, s_th et s_max sont surestimés (**non conservatif**) :
   cas « audit C-1 », d_v = 55,0 cm contre d = 52,0 cm en flexion, s_th + 5,8 %.
5. **Saisies ramenées sans avertissement** au début de chaque run (`_sync_float_raw_keys`,
   `max(0.0, …)`) : un moment saisi négatif devient 0 (la face passe « ok » : non conservatif), un
   pas saisi 0 ou négatif est vérifié à 30 cm (`or 30.0`) alors que le champ affiche 0. Le moteur
   refuse ces saisies.
6. **Commentaire périmé** dans `_dimensionnement_compute_states` : « γs paramétrable, défaut 1.15 »,
   le défaut est 1,5 ; l'en-tête dit « VERSION 2.40 » pour un code qui porte des évolutions v2.41.
7. `beton_classes.json` porte un `tau_lim` par classe que le module ne lit pas (τ_adm est recalculé
   depuis fck,cube) : une seconde valeur, non tenue.

## Le taux d'armature simplifié (option, 10/11/2026)

`beton/poutre/taux.py` — l'option « Taux d'armature » de l'écran Streamlit (`_taux_armature_section`,
`_section_poids_vol`, `_beam_totals`, v2.39+), extraite et branchée sur la géométrie que le moteur calcule déjà pour le
dessin (`<section>["geometrie"]` : les lits, l'emprise de chaque cadre, les armatures latérales). C'est un RATIO
d'avant-métré au mètre courant de poutre, pas un bordereau : le module « Taux d'armature » du site détaille
(`docs/beton/TAUX_ARMATURE_MOTEUR.md`).

**Entrée** (deux clés facultatives ; absentes, la poutre normalisée et la sortie de `verifier` sont celles d'avant, à
l'octet) :

- `poutre["taux"] = {"actif": bool, "majoration": % (0 à 50, défaut 5), "retour": cm (0 à 50, défaut 10),
  "arrondi": kg/m³ entier (1 à 50, défaut 5)}` — éteinte aux réglages par défaut, la clé disparaît à la normalisation ;
  éteinte avec un réglage changé, elle est gardée (le réglage n'est pas perdu) ;
- `section["longueur"]` (m, > 0, ≤ 200) — la longueur de poutre que la section représente ; absente : la section n'entre
  pas dans le taux de la poutre.

**Calcul**, au mètre courant :

| Armature | Compté | Longueur d'une pièce |
|---|---|---|
| lits inférieurs et supérieurs | n barres | 1 m |
| cadre | 100 / pas par mètre | 2·(w + h − 2c) + 2·retour — w : b − 2c s'il entoure toute la largeur, sinon l'emprise des barres entourées, étrier compris |
| épingle | 100 / pas par mètre | (h − 2c) + 2·retour |
| latérales (de peau) | 2·n (n par face, automatiques ou imposées) | 1 m |

masse linéique = 7 850 kg/m³ · π·Ø²/4 ; taux brut = masse · (1 + majoration) / (b·h·1 m) ; taux = le taux brut arrondi au
multiple SUPÉRIEUR de « arrondi ». Poutre : Σ(masse majorée_i · l_i) / Σ(b·h · l_i) sur les sections qui ont une longueur.

**Sortie** : `verifier(p)["taux_armature"]` (seulement si l'option est allumée) = `{"reglages", "sections": [{"nom",
"longueur", "lignes": [{"cle", "position", "famille", "n", "d", "aire", "masse_lineique", "l_barre", "l_totale", "masse",
"part"[, "formule"]}], "masse", "majoration", "masse_majoree", "volume", "taux_brut", "arrondi", "taux", "retour", "pas",
"cours"}], "poutre": {"longueur", "masse", "volume", "taux_brut", "taux", "comptees", "sans_longueur"} | None}`.

**Ce qui n'est pas compté** : les ancrages, les recouvrements, les arrêts de barres, les abouts, les crochets des barres
longitudinales — la majoration les couvre forfaitairement.

**Écarts avec l'écran Streamlit** : aucun sur ce que les deux écrans savent décrire. Le site compte en plus ce que son
ferraillage détaillé permet (une répartition imposée des barres du lit 1 déplace l'emprise d'un cadre intérieur ; des
latérales imposées remplacent les automatiques) ; une épingle du site est verticale (elle n'a qu'une position), là où
l'écran Streamlit comptait aussi une épingle « horizontale » entre deux barres.

**Preuves** : `tests/test_poutre_taux.py` (41 contrôles : l'exemple 30 × 60 à deux sections recalculé à la main —
A : 15,34 kg/m → 90 kg/m³ ; B : 19,42 kg/m → 115 kg/m³ ; poutre 4 m + 2 m : 100 kg/m³ —, les invariants, l'option, l'écran
comparé au calcul à la main, la note PDF, 17 mutations du moteur et 7 de l'écran, toutes rejetées) ;
`tests/test_poutre_moteur.py` § 9 (la PARITÉ : 99 sections et 60 poutres, trois jeux de réglages, égales à l'option du
module rejouée, à 1e-9 près) ; `tests/reference_refonte` (deux cas gelés).

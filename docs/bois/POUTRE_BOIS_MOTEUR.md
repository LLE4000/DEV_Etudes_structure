# Poutre bois — le moteur sans Streamlit, et le module du site

*Livré le 07/11/2026. Moteur `bois/poutre/` : bibliothèque standard seule (ni numpy, ni Streamlit), Python 3.12 et
3.13 (donc Pyodide 0.29.5). Extrait de l'écran Streamlit `modules/poutre_bois.py`, qui reste la **référence** et le
secours (inchangé) ; la preuve de parité est `tests/test_poutre_bois_moteur.py`. Module du site : `api_web/bois.py`
(clé `bois`, établi `poutre-bois`), dessins `api_web/dessin_bois.py`, tests `tests/test_api_web_bois.py`.*

> **À lire d'abord — l'écran Streamlit calcule des flèches 1 000 fois trop grandes** (écart `fleche_unite`, § 6.1).
> Le moteur le reproduit (c'est la règle de parité), le site l'écrit en rouge sous le résultat avec la valeur corrigée
> de l'unité, à titre d'information. Tant que le module n'est pas corrigé, **aucune poutre réaliste ne passe à l'ELS**
> et le verdict de la note est toujours gouverné par une flèche. Décision du bureau attendue.
>
> **Les défauts NON CONSERVATIFS de l'écran Streamlit** (reproduits, pas corrigés ; chacun a son code dans
> `moteur.ECARTS` et son paragraphe au § 6) — à trancher par le bureau, dans `modules/poutre_bois.py` d'abord :
>
> | Code | Défaut | Ordre de grandeur |
> |---|---|---|
> | `combinaison_G` (§ 6.5) | seule γ_G·G + γ_Q·Q est vérifiée, avec le k_mod de la durée choisie ; **γ_G·G seule avec k_mod permanent n'est jamais faite** | 63 × 200 C24, L = 3 m, G = 3, Q = 0,5 kN/m, moyen terme : l'écran dit flexion **92 % (conforme)**, la combinaison G seule donne **104 %** |
> | `cisaillement_kcr` (§ 6.4) | τ sans k_cr = 0,67 | τ sous-estimé d'un facteur **1,49** |
> | `fleche_finale` (§ 6.2) | w_fin = (1 + k_def)·w(G + ψ₂Q) : il manque (1 − ψ₂)·w_inst,Q | ψ₂ = 0,3 : 70 % de la flèche instantanée de Q oubliée |
> | `appui_els` (§ 6.3) | compression à l'appui sous G + Q **non pondérées**, comparée à f_c,90,d | facteur 1,35 à 1,5 manquant (k_c,90 et longueur efficace omis, en sens inverse) |
> | `poids_propre` (§ 6.7) | le poids propre n'est jamais compté | faible (0,02 à 0,1 kN/m) |
> | base des classes (§ 6.9) | f_c,90,k de C22 à C40 et f_v,k de C35, C40 plus élevés qu'EN 338:2016 (relevé de mémoire, à confirmer) | jusqu'à + 40 % sur f_c,90,k (C40) |
>
> Le site les rappelle au pied de chaque fenêtre de calcul ; `combinaison_G` est en plus chiffrée à titre indicatif
> et dite en tête des messages du résultat (en rouge au-delà de 100 %) quand elle gouverne et atteint 95 %.

## 1 · Ce qui est livré

| Pièce | Rôle |
|---|---|
| `bois/poutre/donnees.py` | la base des classes (`TIMBER_BDD`), les sections standard (`SECTIONS_STD`), γ_M, k_mod, k_def, les options des menus de l'écran, les défauts et bornes de ses neuf champs numériques — des COPIES nommées, relues dans le module et dans son écran par le test |
| `bois/poutre/entree.py` | la poutre neuve (l'écran au premier affichage) et la normalisation stricte (`ValueError` en français hors du domaine de l'écran) |
| `bois/poutre/moteur.py` | le calcul de l'écran mot pour mot, les cinq vérifications, les formules du détail (termes et résultats), les écarts EC5 nommés (`ECARTS`), le sélecteur de sections — et le **format complet** dans sa docstring |
| `bois/poutre/diagrammes.py` | les courbes V(x), M(x), w(x) des dessins, portées par les maxima du calcul |
| `tests/test_poutre_bois_moteur.py` | données, contrôles à la main (dont la combinaison G seule), parité au bit près sur 770 poutres, couverture, 13 mutations, sélecteur, format, courbes, pureté (60 contrôles, 2 s) |
| `api_web/bois.py`, `api_web/dessin_bois.py`, `site/web/css/modules/bois.css` | le module du site, ses dessins SVG, sa feuille de style |
| `tests/test_api_web_bois.py` | ce que l'écran du site affiche = ce que le moteur calcule ; saisies, refus, sélecteur, éléments, dessins, échappement, et ce que la relecture a corrigé (85 contrôles) |

## 2 · L'API

```python
from bois.poutre import moteur as M

M.nouvelle_poutre() -> dict                 # C24, SC 1, « long », section 38x150, L 4 m, G 2, Q 3, γ 1,35/1,5, ψ₂ 0,3, L/300, L/200
M.normaliser(poutre) -> dict                # complétée et contrôlée ; ValueError hors du domaine de l'écran
M.verifier(poutre) -> dict                  # le calcul (format : docstring de bois/poutre/moteur.py)
M.candidats_sections(poutre) -> list        # les 11 sections standard calculées, triées par aire (= masse)
M.optimal(poutre) -> dict | None            # la plus légère qui tient
M.ECARTS                                    # les écarts du module avec l'EC5 : code → (texte, sens)
```

**Entrée** (mm, m, kN/m ; une clé absente ou `null` prend le défaut de l'écran) :

```json
{"nom": "P1", "classe": "C24", "service": 1, "duree": "long", "standard": true, "section": "45x220",
 "b": 80.0, "h": 240.0, "L": 4.0, "G": 2.0, "Q": 3.0, "gamma_G": 1.35, "gamma_Q": 1.5, "psi2": 0.3,
 "lim_inst": "L/300", "lim_fin": "L/200", "appui": false, "a": 60.0}
```

`b`, `h` sont la section **sur mesure** : l'écran les garde même quand la bascule « Section standard » est allumée
(ils reviennent quand on l'éteint) ; le site fait de même. `a` est gardée même quand l'option d'appui est éteinte (le
champ de l'écran est alors grisé, pas vidé).

**Domaine** (les bornes des `st.number_input` de l'écran, fermées) : b 30–300 mm, h 80–500 mm, L 0,5–12 m, G et Q
0–50 kN/m, γ_G et γ_Q 1–2, ψ₂ 0,1–1, a 30–200 mm ; classe parmi C14…C40, service 1/2/3 (entier), durée parmi
`permanent long moyen court instant`, section parmi les onze, limites parmi `L/300 L/350 L/400 L/200` et
`L/200 L/250 L/300`. Hors domaine : `ValueError` (« La portée L doit rester entre 0,5 et 12 m, comme dans l'écran
Streamlit (reçu 0,2) ») ; le site la rend en message et garde la poutre intacte — jamais de valeur ramenée en silence.
Les champs du site ne portent **pas** de bornes locales (`data-min`, `data-max`) : l'établi refuserait alors la saisie
lui-même avec un message muet sur le domaine (« Valeur refusée : « 0,2 » ») et laisserait la valeur refusée dans le
champ ; c'est le moteur qui refuse, dit le domaine, et le champ revient à la valeur retenue.

**Indicatif** (`indicatif`, sans effet sur les vérifications ni sur la parité) : la masse et le poids propre (ρ_mean·A),
les deux flèches sans l'écart d'unité (`w_inst_unite`, `w_fin_unite` = flèches de l'écran / 1 000), et la
combinaison γ_G·G seule avec k_mod permanent (`combinaison_G` : q, k_mod, σ_m,Ed, f_m,d, τ_Ed, f_v,d et les deux taux ;
None quand G = 0 ou que la durée choisie est déjà « permanent »).

## 3 · La méthode, grandeur par grandeur

Poutre sur deux appuis, charge uniforme, section rectangulaire de bois massif. Chaque expression est celle du module,
**dans le même ordre d'opérations** (c'est ce qui donne l'identité au bit près).

| Grandeur (sortie `grandeurs`) | Expression de l'écran | Unité |
|---|---|---|
| `A`, `I`, `W` | `b·h`, `b·(h³)/12`, `I/(h/2)` (`sect_rect`) | mm², mm⁴, mm³ |
| `kh` | `min(max((150/max(h, 1))^0,2 ; 0,6) ; 1,3)` (`kh_depth`) | — |
| `kmod`, `kdef` | `KMOD[service][durée]`, `KDEF[service]` | — |
| `fm_d` | `kmod·fm_k·kh/γ_M` | MPa |
| `fv_d` | `kmod·fv_k/γ_M` | MPa |
| `fc90_d` | `kmod·fc90_k/γ_M` | MPa |
| `q_ELU` | `γ_G·G + γ_Q·Q` | kN/m |
| `M_Ed`, `V_Ed` | `q_ELU·L²/8`, `q_ELU·L/2` | kN·m, kN |
| `q_inst`, `q_qp` | `G + Q`, `G + ψ₂·Q` | kN/m |
| `sigma_m_Ed` | `M_Ed·10⁶/W` | MPa |
| `tau_Ed` | `1,5·(V_Ed·1000)/(b·h)` (`shear_tau_rect`) | MPa |
| `w_inst` | `5·(q_inst·1000)·(L·1000)⁴/(384·E0_mean·I)` (`defl_simply`) | « mm » (× 1 000, § 6.1) |
| `w_qp`, `w_fin` | `defl_simply(q_qp…)`, `(1 + kdef)·w_qp` | idem |
| `L_mm`, `lim_inst_mm`, `lim_fin_mm` | `L·1000`, `L_mm/300…`, `L_mm/200…` (le nombre lu après « L/ ») | mm |
| `R_appui` | `(q_inst·L)/2` — **non pondérée** (option d'appui) | kN |
| `sigma_c90_Ed` | `R_appui·1000/(b·a)` | MPa |
| `u_flex`, `u_shear`, `u_c90`, `u_wi`, `u_wf` | valeur / limite (`np.nan` si la limite est nulle, ce que le domaine n'atteint pas ; le moteur rend alors `None`) | — |

État d'une vérification : `ok` si u ≤ 1,0, `nok` sinon — le `st.success` / `st.error` de l'écran ; un taux de 100 %
exactement est conforme (un cas le prouve, et une mutation `<` le fait rougir). Le site colore avec `C.niveau`
(rouge > 100 %, vert foncé 95–100 %, vert pâle < 95 %).

## 4 · La preuve de parité (`tests/test_poutre_bois_moteur.py`)

1. **Données** : `TIMBER_BDD`, `SECTIONS_STD` (valeurs ET types : les dimensions standard sont des entiers),
   `GAMMA_M`, `KMOD`, `KDEF` égaux à ceux du module ; l'écran est exécuté avec un faux `st` qui enregistre chaque
   widget : les six menus (options, ordre, défaut), les deux bascules, les neuf champs numériques (défaut, min, max),
   le champ `a` grisé tant que l'appui est éteint — tout est comparé à `donnees.py`.
2. **À la main** : 63 × 200 C24, SC 2, moyen terme, appui 80 mm, L/400 et L/250 ; 80 × 120 sur mesure, C30, SC 3,
   instantanée (k_h > 1). Recalculés en N et en mm sans le moteur.
3. **Parité** sur **770 poutres** (chaque classe × service × durée ; chaque section × chaque couple de limites ;
   quarante sections sur mesure autour de h = 150 mm ; chaque champ à ses deux bornes ; un taux exactement égal à
   100 % ; 420 tirages, graine 20261107) : l'écran est **exécuté** (`show()` avec un faux `st` qui sert les saisies par
   libellé ; ses variables locales sont relevées à son retour) et comparé au moteur : **20 020 grandeurs identiques au
   bit près, valeur et type** (écart maximal : 0), et **11 186 lignes affichées identiques au texte près** (les quatre
   métriques, les lignes ELU et ELS, chaque pastille verte ou rouge, la légende des propriétés), reconstruites à partir
   des seules valeurs du moteur.
4. **Couverture** : chaque vérification conforme ET non conforme ; k_h > 1, = 1 et < 1 ; chaque option de chaque menu.
5. **Mutations** (le moteur altéré dans son texte, rechargé, rejoué) : γ_M, k_h « corrigé » selon l'EC5, τ avec k_cr,
   appui à l'ELU, flèche finale de l'EC5, unité de la flèche « corrigée », ordre des opérations de f_m,d, W = b·h²/6,
   k_def, limite finale, ψ₂ oublié, « ≤ » devenu « < » : **les douze rougissent**. Corriger un défaut du module
   dans le moteur casse donc la parité, visiblement. Une treizième (la combinaison G seule calculée avec le k_mod
   long terme) fait rougir son contrôle à la main : elle est hors parité, puisque indicative.
6. **Sélecteur** : chaque section candidate repassée par l'écran (taux maximal, aire), tri, retenue, plus légère qui
   tient.
7. **Format** : JSON strict, normalisation idempotente, 36 saisies hors domaine refusées avec un message ; « formule =
   valeurs = résultat » : plus de 16 000 formules du détail se recalculent à partir de leurs seuls termes.
8. **Courbes** : V(0) = V_Ed, M(L/2) = M_Ed, w(L/2) = w_inst et w_fin du calcul.
9. **Pureté** : dans un sous-processus où `sys.meta_path` refuse streamlit, numpy, pandas…, le moteur se charge et
   rend les mêmes résultats, sous Python 3.12 et 3.13.

## 5 · Le site : ce que l'écran Streamlit offre, et où il est

| Écran Streamlit | Site (`api_web/bois.py`) |
|---|---|
| Classe de résistance (10) | § 1 « Poutre », menu « Classe », propriétés de la base sous le menu |
| Classe de service (radio 1/2/3) | § 1, segments « 1 / 2 / 3 » de la ligne « Service », avec ce que la classe désigne et k_mod, k_def, γ_M |
| Durée dominante (5) | § 1, menu « Durée » (permanente, long, moyen, court terme, instantanée), sa période dans l'aide |
| Bascule « Section standard » | § 2, segments « Standard / Sur mesure » |
| Section (mm), 11 | § 2, menu « b × h » **et** le sélecteur de la ligne « Flexion » : trié par masse, retenue ✓, plus légère qui tient ★, propriétés A, I, W et la vérification qui gouverne |
| Largeur b, hauteur h (sur mesure) | § 2, champs b, h (bornes de l'écran) |
| Portée L | § 2 ; A, I, W, k_h affichés dessous |
| G, Q, γ_G, γ_Q, ψ₂ | § 3 « Charges » ; q_ELU, G + Q, G + ψ₂Q dessous |
| Limite instantanée (4), finale (3) | § 4 « Flèches admissibles », segments |
| Bascule « Compression ⟂ fil à l'appui », largeur a (grisée sinon) | § 5 « Appuis », case et champ a (grisé de même) |
| Métriques : section, portée, E0,mean, ρ | « chiffres clés » sous les vérifications (avec M_Ed et V_Ed) |
| ELU : M_Ed, V_Ed ; σ_m,Ed et f_m,d ; flexion % ; τ_Ed et f_v,d ; cisaillement % ; σ_c,90,Ed et f_c,90,d ; compression % | lignes « Flexion », « Cisaillement », « Compression à l'appui » (groupe ELU), fenêtre de calcul ELU |
| ELS : w et limite, taux, instantanée et finale | lignes « Flèche instantanée », « Flèche finale » (groupe ELS), fenêtre de calcul ELS |
| Légende : fm,k, fv,k, fc,90,k, E0,mean, Gmean, k_mod (SC, durée), k_def, k_h, γ_M | aide du § 1 et du § 2, pastilles d'hypothèses des fenêtres |
| « Hypothèses : poutre simplement appuyée, charge uniforme… Ajuste la BDD et les facteurs selon ton Annexe Nationale » | message sous le résultat (deux appuis, charge uniforme, base et coefficients à ajuster selon l'Annexe nationale) ; les écarts connus au pied des fenêtres |
| — (pas de dessin) | **nouveau, affichage seul** : schéma statique et efforts (charges G/Q, appuis, a, réactions, V, M), déformée (limites reportées), section (cernes, σ_m et τ face à f_m,d et f_v,d). La portée est toujours à l'échelle ; la hauteur de la poutre (au plus 72 unités) et la largeur d'appui (au plus 18 % de la portée) sont réduites quand une poutre haute ou un appui large sur une portée courte écraserait la feuille — le dessin le dit en rouge |
| — | **nouveau, indicatif** : la combinaison γ_G·G seule (§ 6.5), chiffrée au pied du calcul ELU, dite en tête des messages quand elle gouverne et atteint 95 % |
| — (une poutre) | plusieurs poutres par note (ajouter, copier, supprimer) ; résumé = la plus chargée |

Affichage : les taux au-delà de 999 % sont plafonnés à 999 % dans les pastilles et les jauges (comme le sélecteur du
garde-corps) ; la comparaison exacte est dans le verdict, la valeur exacte du taux dans le détail du calcul (« η =
… = 1 063 264 % »). Les nombres sont ceux du moteur, à la précision de l'écran au moins (contraintes et efforts au
centième, flèches au dixième).

**Non couvert** : le bouton « Accueil » (le site a sa navigation) ; la note PDF (l'écran n'en a pas ; bouton grisé) ;
modifier la base des classes ou les coefficients (l'écran ne le permet pas non plus : ce sont des constantes du code).

## 6 · Défauts trouvés dans l'écran Streamlit (reproduits, pas corrigés)

Chacun a son **code** dans `moteur.ECARTS` ; le site rappelle ceux qui jouent au pied de la fenêtre de calcul.

### 6.1 `fleche_unite` — les flèches sont 1 000 fois trop grandes (côté sûr, mais l'ELS est faussé)

`defl_simply` calcule `5·(q·1000)·(L·1000)⁴/(384·E·I)` : q est en kN/m, soit **déjà** en N/mm ; le facteur 1 000 le
transforme en N/m dans une formule en millimètres. Exemple, défauts de l'écran (38 × 150, C24, L = 4 m, G + Q =
5 kN/m) : l'écran affiche w_inst = 141 768,6 mm (taux 1 063 264 %) ; la flèche réelle est 141,8 mm (taux 1 063 %).
Une 75 × 225 sous 0,5 + 0,75 kN/m sur 3,5 m : 3 118,9 mm affichés pour 3,1 mm réels (L/300 = 11,7 mm : elle passe
en réalité). Conséquence : **les deux vérifications de flèche échouent pour toute poutre réaliste** ; le sélecteur
n'a presque jamais de « plus légère qui tient ». Une mutation du test (la flèche avec q en N/mm) rougit la parité.

### 6.2 `fleche_finale` — la flèche finale n'est pas celle de l'EC5 (non conservatif)

L'écran : w_fin = (1 + k_def)·w(G + ψ₂Q). L'EC5 §2.2.3 : w_fin = w_inst,G·(1 + k_def) + w_inst,Q·(1 + ψ₂·k_def). La
différence vaut (1 − ψ₂)·w_inst,Q : avec ψ₂ = 0,3, la part de Q est comptée à 30 % au lieu de 100 % (+ fluage). Non
conservatif dès que Q ≠ 0 et ψ₂ < 1.

### 6.3 `appui_els` — la compression à l'appui sous charges non pondérées (non conservatif à l'ELU)

R = (G + Q)·L/2 est comparée à f_c,90,d (une résistance de **calcul**) : il manque γ_G et γ_Q (facteur 1,35 à 1,5).
En sens inverse, k_c,90 (≥ 1) et la longueur efficace (a prolongée de 30 mm au plus de chaque côté, EC5 §6.1.5) sont omis (côté sûr). Le bilan
dépend du cas ; le défaut de pondération domine pour les appuis courts.

### 6.4 `cisaillement_kcr` — τ sans k_cr (non conservatif)

τ = 1,5·V/(b·h) avec la largeur pleine ; l'EC5 (amendement A1, §6.1.7) prend b_ef = k_cr·b, k_cr = 0,67 pour le bois
massif : τ est sous-estimé d'un facteur 1,49.

### 6.5 `combinaison_G` — la combinaison γ_G·G seule n'est pas vérifiée (non conservatif)

*Relevé à la relecture du 07/11/2026.* L'écran ne fait qu'une combinaison ELU, γ_G·G + γ_Q·Q, avec le k_mod de la
durée « dominante » choisie. L'EC5 §3.1.3 (2) prend pour une combinaison le k_mod de l'action de plus courte durée,
mais **chaque combinaison** de l'EN 1990 se vérifie avec le sien : γ_G·G seule, avec k_mod **permanent**, gouverne
dès que γ_G·G / k_mod,perm > (γ_G·G + γ_Q·Q) / k_mod, c'est-à-dire dès que G domine Q. Exemple (63 × 200 C24, SC 1,
L = 3 m, G = 3, Q = 0,5 kN/m, moyen terme, k_mod 0,8) : l'écran dit la flexion **conforme à 92 %** ; 1,35·G avec
k_mod 0,6 donne **104 %**. Avec G = 4, Q = 1, court terme : 118 % à l'écran, 138 % en G seule.

Le moteur la **chiffre à titre indicatif** (`indicatif.combinaison_G` : flexion et cisaillement, mêmes formules que
l'écran, k_mod permanent) **sans** l'ajouter aux vérifications : le verdict, les taux et la parité restent ceux de
l'écran. Le site la cite au pied du calcul ELU (avec ses chiffres) dès qu'elle n'est pas couverte (G > 0 et durée
autre que « permanent »), et la dit **en tête des messages sous le résultat** quand elle gouverne la résistance et
atteint la bande d'attention (95 %) : en rouge au-delà de 100 %, en message ordinaire entre 95 et 100 %. La correction (une seconde ligne « G seule » dans les vérifications)
appartient au bureau, dans `modules/poutre_bois.py` d'abord.

### 6.6 `kh_reduit` — k_h < 1 au-dessus de 150 mm (côté sûr)

L'EC5 §3.2(3) n'applique k_h qu'aux hauteurs inférieures à 150 mm (k_h = 1 au-delà) ; l'écran prend (150/h)^0,2 pour
toute hauteur : f_m,d est réduite de 5,6 % pour h = 200 mm, de 11,7 % pour h = 280 mm.

### 6.7 `poids_propre` — le poids propre n'est pas compté (non conservatif, faible)

ρ_mean est affiché mais n'entre dans aucune charge. Le site donne l'ordre de grandeur (ρ_mean·g·A, indicatif) et
rappelle de l'inclure dans G.

### 6.8 `deversement` — ce qui n'est pas vérifié

Ni le déversement (k_crit, EC5 §6.3.3 : décisif pour les poutres hautes et minces non tenues), ni la part de
l'effort tranchant dans la flèche (G_mean est dans la base mais inutilisé). La combinaison γ_G·G seule avec k_mod
permanent, elle, a son propre code (§ 6.5).

### 6.9 La base des classes (« valeurs usuelles EN 338 – à ajuster si besoin »)

Elle ne suit pas une édition unique d'EN 338. Rapprochée d'EN 338:2016 (tableau 1 — **relevé de mémoire, à confirmer
sur le texte de la norme**), les valeurs **utilisées par le calcul** qui seraient plus favorables que la norme :
f_c,90,k de C22 à C40 (2,5 / 2,7 / 2,9 / 3,0 / 3,4 / 3,8 MPa contre 2,4 / 2,5 / 2,5 / 2,7 / 2,7 / 2,7) et f_v,k de
C35 et C40 (4,5 et 5,0 MPa contre 4,0). En sens inverse, f_v,k de C14 à C22 (2,0 à 3,0 MPa) est plus faible que les
3,0 à 3,8 MPa d'EN 338:2009/2016 (côté sûr). E0,mean et f_m,k concordent. ρ_mean diffère pour C20, C27, C35, C40
(sans effet sur le calcul). À trancher par le bureau, dans `modules/poutre_bois.py` d'abord, puis dans `donnees.py`
(le test exige qu'ils restent égaux).

### 6.10 Ergonomie et affichage

- La section proposée à l'ouverture est la **première** de la liste (38 × 150 : le menu n'a pas d'index), qui ne tient
  pas sous les charges par défaut (flexion 782 %). Le site reprend ce défaut (parité de la note neuve).
- `f"{u*100:.0f}%"` arrondit 100,4 % en « 100% » affiché en rouge ; le site arrondit sans changer de bande (101 %).
- La métrique « Section (mm) » tronque une section sur mesure (`int(80,6)` → 80) ; le site écrit la valeur saisie.
- ψ₂ ne descend pas sous 0,1 (une charge sans part quasi permanente, ψ₂ = 0, n'est pas saisissable) ; le site garde
  le domaine de l'écran et le dit.

## 7 · Besoins communs relevés (signalés, pas bricolés)

- La mention « mm · à l'échelle » de l'en-tête des vues est fixe (`js/etabli.js`) : pour un diagramme (V, M, w, σ, τ),
  seule l'abscisse est à l'échelle. Une légende d'échelle par vue (champ `echelle` de la vue) serait plus juste.
- `C.chip` / `C.pc` n'écrivent pas les milliers : un taux de 1 063 264 % ne peut s'afficher lisiblement dans une
  pastille, d'où le plafond à 999 % (même choix que le sélecteur du garde-corps). Un plafond commun, écrit « > 999 % »,
  serait plus honnête.
- `C.symb_math` ne met pas en indice après ψ (ψ_2 reste tel quel) : le module écrit ψ₂.
- Les bornes locales d'un champ (`C.num(..., mini, maxi)` → `data-min`, `data-max`) font refuser la saisie par
  l'établi (`js/etabli.js`, `valider`) avec un message muet sur le domaine (« Valeur refusée : « 0,2 » »), et la
  valeur refusée reste affichée dans le champ alors que la note garde l'ancienne. Le module n'en pose donc pas : le
  moteur refuse et dit le domaine. Un message commun « entre 0,5 et 12 m » (les bornes sont connues de la page) et un
  champ remis à sa valeur feraient mieux pour tous les modules.
- Format tablette : la toile est étroite et le schéma statique (dessin en largeur) y devient petit ; dans le cartouche,
  « Note » et « Chantier » se chevauchent. Mise en page commune.

## 8 · Relecture indépendante du 07/11/2026

Ce qui a été contrôlé, et ce qui a changé.

- **Chiffres** : 20 poutres (défauts de l'écran, appui coché, sur mesure à décimales, k_h > 1, taux exactement
  100 %, sans charge, toutes bornes hautes, toutes bornes basses, chaque limite, la bande 95–100 %, 100,1–100,49 %,
  94,5–94,99 %…) : l'écran Streamlit rejoué par un faux `st` indépendant de celui des tests, comparé aux **fragments
  HTML** que rend `api_web/routeur.traiter` (pas au moteur) : métriques, M_Ed, V_Ed, σ, f, τ, flèches, limites,
  87 pastilles (taux et couleur) — **aucun écart** à l'arrondi d'affichage près.
- **Défaut non conservatif de plus** : `combinaison_G` (§ 6.5), ajouté au moteur (indicatif), au site et à la tête de
  ce document.
- **Saisie** : les bornes locales des champs retirées (§ 2) ; le message de refus dit le domaine en clair.
- **Résultat** : la mention de l'Annexe nationale de l'écran Streamlit, qui manquait ; les messages sans jargon
  (« reproduit pour la parité » → « le site reprend ses chiffres tels quels ») ; les flèches corrigées de l'unité
  viennent du moteur (`indicatif`), le site ne calcule plus rien ; les petits résultats gardent deux chiffres
  significatifs (M_Ed = 0,0022 kN·m, et non « 0,00 » sous un diagramme tracé en entier).
- **Dessins** : poutre haute ou appui large sur portée courte réduits et dits (la feuille était écrasée par une
  poutre de 520 unités) ; « 0 » au lieu de « −0,00 » ; étiquettes posées sur un trait détourées de blanc (L/300 et
  L/200 étaient barrées par l'axe) ; indices dans la liste des écarts (k<sub>cr</sub>, w<sub>inst,Q</sub>…).
- **Robustesse** : un rang actif infini ou illisible (« abc ») dans une note ou une requête est relu (poutre 1) au lieu
  de lever une erreur technique.

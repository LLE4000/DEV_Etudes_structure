# Barres en acier — Méthode ELS (NBN B 51-001) et Eurocode 3

*Ouvert le 14/11/2026. Ce document est la SPÉCIFICATION VÉRIFIÉE du chantier « barres acier » : il remplace le brief
reçu ce jour-là (rédigé en conversation à partir des classeurs du bureau), qu'il reprend, corrige et complète. Il est
tenu à jour lot par lot.*

*16/11/2026 : le brief version 11 change plusieurs points de ce document, dont quatre où le module en service n'est pas
conservatif. L'inventaire, les écarts et le plan sont dans `BARRES_ACIER_V11.md` ; ce document sera mis à jour lot par
lot à mesure des corrections.*

*16/11/2026, lot A (deuxième passe) : la méthode du bureau se NOMME « Méthode ELS (NBN B 51-001) » — à l'écran, dans
la note, dans ce document — et « ELS » seul là où la place manque ; sa clé interne reste « historique » (état, notes
enregistrées, `historique.py`, `test_barre_historique.py`). Les phrases que le lot A a rendues fausses (matériau, couple
(λ̄_LT,0 ; β), η, A_v,y, poids propre, âme) sont corrigées en place.*

**Demande du propriétaire (14/11/2026).** Vérifier un profilé métallique — poutre, poteau, poutre-poteau — avec ses
instabilités (flambement, déversement, voilement local, raidisseurs), par DEUX méthodes calculées ensemble : la méthode
ELS du bureau (ancienne norme belge NBN B 51-001, contraintes admissibles, charges de service) et l'Eurocode 3
(NBN EN 1993-1-1 et 1993-1-5, annexes belges). À l'impression : l'une, l'autre ou les deux. Dans le module existant
« Choix de profilé ». « Quand mon supérieur lit, il retrouve sa méthode. » Écran simple, messages courts, raidisseurs et
appuis ajoutables.

## 1. Les sources, et ce qui n'est PAS dans le dépôt

| Source | Ce que c'est | Où |
|---|---|---|
| Six classeurs « Flambement » du bureau | quatre récents de même gabarit (HEB 200, HEA 120, CHS 168,3, tube Ø610) et deux anciens (FLAMBECBD, FLAMBEBAD) | reçus en conversation — **pas dans le dépôt** |
| Documentation théorique du bureau | 26 pages numérisées : « 4. Instabilité élastique » (compression simple, flexion composée plane, déversement, flexion composée déviée, exemples) ; citée ici « doc p. X » | reçue en conversation — **pas dans le dépôt** |
| Brief du 14/11/2026 | le cahier des charges d'origine | reçu en conversation ; ce document le remplace |
| EN 1993-1-1, EN 1993-1-5, ANB belges | les textes | **absents** : voir § 8 |

**Pourquoi pas dans le dépôt** : ce dépôt est PUBLIC. Les classeurs et la documentation sont des documents du bureau
(et un cours annoté) ; les normes sont sous droits. Ce document garde ce qu'il faut pour travailler sans eux : chaque
formule avec sa page, chaque donnée et chaque résultat des classeurs (§ 6), figés par `tests/test_barre_historique.py`.

## 2. Le deuxième avis : ce qui a été vérifié, et comment

Tout a été RECALCULÉ de façon indépendante (les formules de la documentation, écrites à neuf — pas celles des
classeurs), puis comparé.

| Objet | Contrôle | Résultat |
|---|---|---|
| Les 108 valeurs du tableau φ (doc p. 5) | la formule de la doc p. 3, de λ̄ = 0,05 à 1,80, courbes a, b, c | **108 sur 108** au millième |
| Les 4 classeurs récents | chaque cellule de formule relue ; recalcul depuis les seules données | écart relatif ≤ 5·10⁻¹⁵ |
| Les 2 anciens classeurs | idem | écart relatif ≤ 4·10⁻¹⁵ |
| Les 5 exemples de la doc (p. 6, 8, 14, 23, 25) | recalcul sans arrondi intermédiaire | conformes à l'arrondi de la doc près (§ 6.3) |
| χ de l'EN 1993-1-1 (20 valeurs du brief) | recalcul | **20 sur 20** |
| Le moment critique M_cr (7 cas R1 du brief) | un solveur par éléments finis écrit à neuf (numpy/scipy, hors moteur), convergence vérifiée | **7 sur 7** au centième de kN·m |
| Le déversement R3 (2 niveaux de charge × 2 jeux de paramètres) | idem | conformes |
| L'âme sous charge concentrée R4 (5 cas) | recalcul | **5 sur 5** |
| Les formules EN 1993-1-1 et 1993-1-5 du brief | relues une à une, de mémoire du texte et par dérivation quand c'est possible | deux erreurs de recopie (E1, E2), une valeur à pointer (E3) — § 3 |

**Conclusion.** Les classeurs calculent juste ce qu'ils calculent ; leurs défauts sont ceux que le brief avait vus (§ 4.3).
Le brief est fiable ; trois corrections et quelques précisions ci-dessous.

## 3. Les écarts relevés

| # | Où | Écart | Effet | Décision |
|---|---|---|---|---|
| E1 | brief § 7.7, facteur k_c (tableau 6.6) | charge concentrée à mi-portée : **0,86**, pas 0,90 (0,90 est la charge répartie sur poutre bi-encastrée). Contrôle : k_c ≈ 1/√C₁ → 1/√1,348 = 0,861 ; 1/√1,127 = 0,942 | f, donc χ_LT,mod : ≈ 2 % | corrigé |
| E2 | brief § 7.8, tableau A.2 | les deux cas sont INVERSÉS. La formule générale du même tableau le prouve : charge répartie π²·(5/384)·8 − 1 = +0,028 → C_m,0 = 1 **+ 0,03**·N_Ed/N_cr ; charge centrale π²/12 − 1 = −0,178 → 1 **− 0,18**·N_Ed/N_cr | k_yy, k_zz : quelques % | corrigé |
| E3 | brief § 7.1, f_u du S355 | tableau 3.1 de l'édition 2005 (EN 10025-2) : 355 / **510** (t ≤ 40 mm), 335 / 470 ; 490 est la valeur du S355 N/NL (et de l'édition 2022) | aucun en v1 (f_u n'entre dans aucune vérification retenue) | lot A : le tableau 3.1 n'est plus lu — f_u et f_y selon l'épaisseur, option a de l'ANB:2018 3.2.1(1), tableau 3.3 ANB : S355, f_u = 470 de 3 à 100 mm (§ 5.1) |
| E4 | brief § 11.2, HEB 200 en compression | N_b,z,Rd = 570,1 kN suppose λ = L/i avec i = 5,07 cm (l'arrondi du classeur). Avec N_cr = π²·E·I_z/L² (la définition de l'EC3, I_z = 2003 cm⁴) : λ̄ = 1,5139, χ = 0,3101, **569,1 kN** | 0,2 % | les deux valeurs sont dans le test ; le moteur suit la définition de l'EC3 |
| E5 | brief § 4.1 | « tableau 13.2 » : c'est le tableau 13.1 (doc p. 16) | — | corrigé |
| E6 | brief § 6.4, « tubes : φ_ed = 1 » | la doc (p. 17-18) donne une formule pour les profils FERMÉS : λ_ed = √(π·√(E/G)·ℓ·W_x/√(I_y·J_T)) (« 5,1 », exact 5,066), « généralement proche de l'unité ». Exact pour un tube rond ou carré (pas de déversement) ; pour un tube RECTANGULAIRE fléchi autour de l'axe fort, la formule de la doc s'applique | ≤ 2 % | tube rond ou carré : 1 ; rectangulaire : la formule de la doc, avec la courbe du tube |
| E7 | doc p. 18 et 19 | deux coquilles dans les formules INTERMÉDIAIRES du moment critique des profils ouverts : (h_a/2ℓ)² pour (h_a/2ℓ²)² (p. 18) ; π⁴ pour π² au premier terme (p. 19) | aucun : la méthode n'emploie que les formules (3) et (4) de la p. 19 | noté |
| E8 | doc p. 20, dispense « C/φ_ed = 1 si P/(A·φ) ≤ 0,15·σ_adm » | annotée « FAUX » dans la marge ; l'exemple de la p. 23 l'applique pourtant (à 0,17·σ_adm). Sans elle, cet exemple (HEA 200, 8 m, sans maintien) donne φ_ed = 0,307 et σ = 422 N/mm² au lieu de 130 | majeur pour une poutre non maintenue | la dispense n'est PAS reprise (brief) ; voir § 4.5 et la question Q4 |
| E9 | classeurs, poids propre | 78,6 kN/m³ × A (g = 10) ; la doc p. 8 prend 7,85 t/m³ × 9,81 = 77,0 | 2 % sur un terme mineur | lot A : 78,6 kN/m³ × A (brief H13), la valeur nommée par le moteur (`statique.POIDS_VOLUMIQUE`) ; le test des classeurs reprend leur saisie |

Données des classeurs à signaler (non corrigées dans les tests, § 6) : A = 2520 mm² pour le CHS 168,3 × 6 (catalogue :
3059) ; « section ailes » du HEA 120 = une seule aile (960 au lieu de 1920 mm²) ; W = 3506 cm³ pour le tube Ø610 × 12,5
(I/(D/2) = 3435 : le classeur divise par le rayon MOYEN) ; valeurs d'axe faible dans les cases « fort » de FLAMBECBD ;
f_y = 450 N/mm² pour le tube Ø610 (hors des nuances AE235 à AE355 de la doc).

## 4. La méthode ELS (NBN B 51-001, contraintes admissibles)

Charges de SERVICE, non pondérées (le « cumul I » de la doc : poids propre + exploitation + neige…) ; la sécurité est
portée par s = 1,5 : σ_adm = f_y / s. Unités internes : N, mm.

### 4.1 Les formules

Pour chaque axe (y fort, z faible ; la doc écrit x pour le fort, y pour le faible) :

| N° | Grandeur | Formule | Référence |
|---|---|---|---|
| H1 | Élancement limite | λ_E = π·√(E / f_y), E = 210 000 N/mm² | doc p. 2 et 5 |
| H2 | Élancement | λ = u·L / i | doc p. 1 |
| H3 | Élancement réduit | λ̄ = λ / λ_E | doc p. 3 |
| H4 | Termes | A′ = 0,5 + α·λ̄² ; B′ = β·λ̄² | doc p. 3 |
| H5 | Coefficient de flambement | φ = 1 / (A′ + √(A′² − B′)) | doc p. 3 |
| H6 | Charge admissible | P_adm = A·φ·f_y / s | doc p. 2 et 6 |
| H7 | Charge critique | P_cr = π²·E·I / (u·L)² | doc p. 1, 23 et 25 |
| H8 | Facteur du second ordre | PE = 1 − s·P / P_cr, doit rester > 0 | doc p. 11 |
| H9 | Flexion composée plane | σ = P/(A·φ_min) + C·M/(φ_ed·W·PE) ≤ f_y / s ; φ_ed = φ_d des classeurs (§ 4.5) | doc p. 20, formule (5) |
| H10 | Flexion composée déviée | σ = P/(A·φ_min) + C_y·M_y/(φ_ed·W_y·PE_y) + C_z·M_z/(W_z·PE_z) ≤ f_y / s | doc p. 26 |
| H11 | Cisaillement | τ_z = \|V_z\| / A_âme ; τ_y = \|V_y\| / A_ailes ; τ = τ_y + τ_z (PRS : A_âme = h_w·t_w, A_ailes = b₁·t₁ + b₂·t₂) | classeurs |
| H12 | Contrainte équivalente | σ_eq = √(σ² + 3·τ²) ≤ f_y / s | classeurs |
| H13 | Poids propre | q = 78,6 kN/m³ × A ; M = q·L²/8 ; V = q·L/2 (poutre sur deux appuis) | classeurs (E9) |

Courbes (doc p. 3) : a : α = 0,514, β = 0,795 ; b : α = 0,554, β = 0,738 ; c : α = 0,532, β = 0,377.

φ_min = min(φ_y ; φ_z) : la doc demande φ(λ_max), « le plan de flambement le plus défavorable » (p. 20 et 26) — le plus
petit φ, puisque la courbe change avec l'axe. Conséquence dite par la doc : le flambement en compression simple dans
l'autre plan n'a plus à être vérifié à part.

### 4.2 Le choix de la courbe (doc p. 4)

| Profil | Axe faible | Axe fort |
|---|---|---|
| I ou H laminé, h/b > 1,2 | b | a |
| I ou H laminé, h/b ≤ 1,2 | c | b |
| I ou H soudé, semelles oxycoupées | b | a |
| I ou H soudé, semelles en larges plats | c | b |
| I ou H laminé avec semelles additionnelles | a | b |
| I ou H recuit | b | a |
| Tube rond ou carré fini à chaud, ou recuit | a | a |
| Profil creux fini à froid, paroi ≤ 6 mm (lecture de la numérisation : « ≤ » ou « < ») | c | c |
| Caisson soudé non recuit | b | b |
| Caisson soudé recuit | a | a |
| Té, U, cornière | c | c |

**U et cornières (lot C, C1b ; brief v12 § 3).** En efforts imposés seulement (une barre chargée est refusée en une
ligne) : la compression simple de la méthode ELS, courbe c ; toute flexion « non couverte » ; l'Eurocode 3 « non couvert »
en une ligne (« Eurocode 3 : non couvert pour une cornière (v1). »). Une cornière flambe autour de ses axes principaux
u-u et v-v (`section.py`) : chacun prend la plus grande longueur de flambement saisie, v-v (le plus petit rayon) gouverne
(À VÉRIFIER). Doc p. 8 : L 60 × 60 × 8 de 2 m, λ̄ 1,820, φ_c 0,236, P_adm 33,4 kN (`tests/test_barre_nuances.py`).

### 4.3 Ce que le module corrige ou ajoute par rapport aux classeurs

1. P_cr avec (u·L)² — le classeur oublie u (sans effet quand u = 1, ce qui est le cas des six classeurs).
2. u et L séparés par axe.
3. φ_min dans la flexion composée — les classeurs récents prennent φ de l'axe faible, les anciens φ de l'axe fort.
4. Verdicts explicites : P ≤ P_adm par axe ; σ ≤ f_y/s ; σ_eq ≤ f_y/s ; chacun avec son taux.
5. Garde-fous : une seule courbe par axe (le classeur rend φ = 1 sans choix, et ADDITIONNE α et β si deux cases sont
   cochées) ; alerte si λ̄ > 1,8 (fin du tableau de la doc ; les courbes b et c se croisent vers λ̄ = 2,67) ; alerte si
   λ > 200 (doc p. 7) ; ARRÊT si PE ≤ 0, avec l'inertie minimale I_min = s·P·(u·L)² / (π²·E).
6. « Coef secu » (P_adm / P) s'appelle « réserve P_adm / P ».
7. Poids propre ajouté à P pour un poteau (option).
8. Hors du domaine de la doc — nuance hors AE235 à AE355 (lue sur la valeur NOMINALE), profil creux formé à froid de
   paroi de plus de 6 mm, paroi de plus de 80 mm (13 HD et HL), et, pour le déversement, un PRS monosymétrique : statut
   « hors domaine » (lot A, brief § 9 ; lot B, brief v12 § 19.1 point 5 et § 18.6) — le calcul est montré, jamais
   « vérifié », jamais caché derrière le taux de la barre. **f_y : la NOMINALE de la nuance** (235, 275, 355 N/mm²) quelle
   que soit l'épaisseur, comme les classeurs (lot B, brief v12 § 6.1, § 13.2 point 5, § 18.3 ; la note l'écrit « f_y
   nominale, comme les classeurs du bureau ») ; l'Eurocode 3 garde la table selon l'épaisseur (HEB 300 S235 : 235 ici,
   225 en EC3). **Verdict bloqué** (« non couvert », le calcul reste montré du côté défavorable) quand une donnée manque ou
   se contredit : bords des semelles d'un PRS ou fabrication d'un tube non saisis ; un I dont I_y ≤ I_z (des valeurs de
   l'axe faible dans les cases de l'axe fort, comme FLAMBECBD).
9. La tolérance de 5 % de la doc (p. 13 et 23) n'est PAS appliquée (option éteinte ; question Q4).
10. Le coefficient C et le déversement de la doc (§ 4.4, 4.5), que les classeurs ne calculent pas.
11. La traction et la flexion simple : la doc ne traite que la compression. Le module écrit alors la résistance simple
    σ = |N|/A + M/W ≤ f_y/s (φ = 1, PE = 1), et le dit.

### 4.4 Le coefficient C (doc p. 20 et 21)

C = 1 par défaut (comme les classeurs). Options : moments d'extrémité seuls, C = √(0,3·(1 + ψ²) + 0,4·ψ), ψ = M₂/M₁,
|M₂| ≤ |M₁| (de 1 pour ψ = 1 à 0,447 pour ψ = −1) ; charges appliquées sur la barre : C = 0,85.

### 4.5 Le déversement : la méthode des classeurs « Déversement » (lot B, brief v12 § 6.4)

`acier/barre/historique_deversement.py`. **Par défaut** pour un I ou un H fléchi autour de l'axe fort (DÉCISION BUREAU
n° 5, tranchée le 16/11/2026) :

    λ_e,d = 3,9·√(l·h / (b·t_f·C_d))      λ̄ = λ_e,d / λ_E (f_y nominale : 93,91 / 86,81 / 76,41)
    Φ = 1 + n·(λ̄ − 0,2) + λ̄²     φ_d = [Φ − √(Φ² − 4·λ̄²)] / (2·λ̄²) ; φ_d = 1 si λ̄ ≤ 0,2     (NBN B 51-002, § 4.3.3.5.3)
    σ_adm,d = f_y/s × φ_d        vérification « déversement » : σ_M = M_y/W_y ≤ σ_adm,d (avec son taux)

l = L_d, distance entre maintiens latéraux de la semelle COMPRIMÉE (donnée distincte de L_cr,y et L_cr,z ; barre
chargée : celle du tronçon) ; b et t_f de la semelle comprimée (un moment positif comprime la semelle haute) ; C_d saisi,
1 par défaut (1 à 3 : borne de saisie du module, À VÉRIFIER). Le 3,9 : √(3π·√2,6) = 3,899 (doc p. 18, torsion de
Saint-Venant seule, deux semelles égales), gardé comme les classeurs ; la note le dit une fois, dans les hypothèses (barre
chargée : celles de la ligne qui gouverne, plus celles des lignes des contraintes et du déversement, sans doublon — elles
restent dites quand H-EQ gouverne à l'appui ; deux tronçons de L_d différents, par exemple H-EQ et H-DEV de part et
d'autre d'un appui intermédiaire : chaque ligne « Déversement » dit son tronçon et sa vérification, celle de la gouvernante
d'abord, en une ligne de moins de 110 caractères — « Déversement sur 0–2 m (contrainte équivalente) : C_d = 1, classe b (…) » :
le sujet perd sa source (norme, page, « classeurs ») et garde ce qui décrit le cas (« profil fermé ») ; L_d est tu, en tête
ou en fin, quand il vaut la longueur du tronçon ; deux vérifications sur le même tronçon : le premier mot de chaque titre,
« (flexion, déversement) »). n (tableau 11) : a 0,206 ; b 0,339 ; c 0,489 ; d 0,756. Classe (tableau 12, ligne « axe
faible », un seul choix) : I laminé h/b > 1,2 → b (t_f ≤ 40), c (≤ 80), d au-delà (pas de ligne : note 3, prudent) ;
h/b ≤ 1,2 → c (≤ 80), d au-delà ; PRS oxycoupé → c (t_f ≤ 40), d au-delà ; larges plats → d ; U → c (fonction seule) ;
un autre profil (I recuit, à semelles additionnelles) : classe « à fournir » (le message nomme le profil) — la flexion est
non couverte, le flambement se calcule. Colonne lue : f_y ≤ 430 N/mm² ; au-delà (S460), la même, et l'hypothèse le dit (À VÉRIFIER : l'autre colonne
n'est pas dans le brief). Une classe imposée (`historique.classe_d` ; vide ou None : celle du tableau 12) est comparée à celle du tableau 12
(alerte ; S460 : elle dit la colonne f_y ≤ 430, À VÉRIFIER). φ_ed = φ_d entre dans le terme de moment fort de H9 et H10 : avec P = 0 et
C = 1, H10 redonne la vérification des classeurs — la ligne des contraintes n'est alors pas répétée.

Messages (sans changer les valeurs) : C_d > 1 avec la charge sur la semelle haute (« gardez C_d = 1 ») ; PRS : « formule
prudente : elle néglige le gauchissement » sur la ligne, « classes du flambement et du déversement différentes » dans les
hypothèses (pas avec une classe imposée) ; PRS monosymétrique : hors domaine (H-DEV, H-SIGMA et H-EQ, classeurs ou variante du fascicule), calcul avec
la semelle comprimée, W = W_el,y,min (déversement et H10). Un PRS à I_y ≤ I_z (semelles larges, âme basse) : les
charges, dans le plan de l'âme, le fléchissent autour de son axe faible, que la méthode ne traite pas — verdict bloqué,
avec ce message.

**Les corrections des classeurs** (brief § 6.7) sont NOMMÉES par le moteur (`historique_deversement.CORRECTIONS`) et
imprimées « Correction du classeur : … » dans la vérification concernée, seulement quand elles jouent sur le cas :
P_cr avec (u·L)² et φ_min (ligne des contraintes), |V_y| + |V_z| (contrainte équivalente), et pour le déversement :
(1) φ_d = 1 sous λ̄ = 0,2 (la seule ligne qui le dit) ; (2) une classe pour 40 < t_f ≤ 80 mm et pour t_f = 80 mm ;
(3) les seuils de 40 et 80 mm du tableau 12 (la feuille « Calcul automatique » prenait 100 mm, joue pour
80 < t_f ≤ 100 mm) ; (4) σ_adm,d avec la f_y de la nuance (DEVERSEMENT_CAD, feuille S355 : 157) — dite à f_y = 355
seulement, la seule erreur connue (À VÉRIFIER pour une S275) ; (5) la classe du type de profilé ; (6) la ligne « axe
faible » seule — pas pour un laminé de plus de 80 mm, où les deux lignes donnent sans doute d (À VÉRIFIER : la ligne
« axe fort » du tableau 12 est à fournir). Une classe imposée : aucune correction de classe. « Une seule courbe par axe » est un garde-fou de la saisie : il ne joue sur aucun cas calculé. Les choix du
bureau (f_y nominale, s, C, C_d, courbes et classes) ne sont jamais présentés comme des corrections. La page « Écarts
avec les classeurs du bureau » (lot C) lira la liste `corrections` du résultat (barre chargée : le texte de l'occurrence
retenue de sa vérification, car il change d'un tronçon à l'autre).

**Profils fermés** (RHS, caissons ; doc p. 17-18) : λ_ed = √(π·√(E/G)·L_d·W_y / √(I_z·I_t)), G = E/2,6 = 80 769 N/mm²,
facteur π·√2,6 = 5,0657 (variante « facteur de la doc 5,1 », `historique.facteur_doc` : λ_ed + 0,338 %), courbe du fascicule (caisson soudé b,
recuit a ; autres : leur ligne du tableau p. 4). **Tube rond ou carré** : φ_ed = 1, règle complémentaire (EC3
§ 6.3.2.1(2) : I_y = I_z), dite comme telle ; le flambement sous P reste vérifié.

**La variante du fascicule p. 19** (le défaut jusqu'au lot A), éteinte : ni calculée ni affichée tant qu'on ne la choisit
pas (`historique.variante` = « fascicule ») — la semelle comprimée et un sixième de l'âme vus comme une barre comprimée :
I_d = t_f·b³/12, A_d = b·t_f + h_w·t_w/6, λ_ed = L_d/r_d, courbe de l'axe faible du fascicule. Elle sert aux exemples de
la doc (p. 23, p. 25) et aux anciennes lectures de FLAMBECBD et FLAMBEBAD. La note dit quelle variante décide.

**À savoir** (E8) : la formule des classeurs néglige le gauchissement (torsion de Saint-Venant seule) ; celle du
fascicule, la torsion de Saint-Venant (la semelle comprimée vue comme une barre). Elles sont prudentes pour une poutre haute (PRS-C,
poutre de 28 m : 18 maintiens selon les classeurs, 8 selon l'Eurocode 3). Une semelle comprimée maintenue en continu
(dalle, bac) donne φ_ed = 1.

### 4.6 Hors de la méthode ELS

La doc ne traite ni le voilement local, ni l'âme sous charge concentrée, ni les raidisseurs. Dans une note « méthode
ELS seule », ces contrôles paraissent dans un bloc séparé « Contrôles complémentaires EN 1993-1-5 » (§ 5.9).

## 5. L'Eurocode 3 (EN 1993-1-1:2005 + AC:2009 + A1:2014, ANB:2018 ; EN 1993-1-5:2006)

Le brief (§ 7 et 8) est repris tel quel, SAUF les corrections E1 à E3. Ne sont notés ici que les points vérifiés un à un
et les paramètres nationaux ; chaque module du moteur (`acier/barre/ec3_*.py`) porte ses formules et leurs clauses.

### 5.1 Matériau — § 3.2
E = 210 000 N/mm² ; G = 81 000 N/mm² ; ν = 0,3 ; ε = √(235/f_y). Lot A : f_y et f_u selon l'épaisseur ET le produit,
la table unique du site (`acier/materiaux.py` ; NBN EN 1993-1-1 ANB:2018 3.2.1(1), option a) — tableau 3.3 ANB pour
les laminés (S235 235 / 225 / 215 au-delà de 16 et 40 mm ; S275 275 / 265 / 255 / 245 au-delà de 16, 40 et 63 mm ;
S355 355 / 345 / 335 / 325 ; f_u de 3 à 100 mm : 360, 410, 470), tableau 3.4 ANB (S460N : 460 / 440 / 430 / 410 ;
f_u 540), tableaux 3.8 et 3.10 ANB pour les creux finis à chaud et formés à froid. La f_y de CALCUL est celle de la paroi
la plus épaisse, max(t_f ; t_w) ; l'ε, celui de CHAQUE paroi, avec la f_y de SON épaisseur (classe, largeurs efficaces,
λ̄_p, 72·ε/η, λ̄_w, λ̄_F, 15·ε·t_w). Au-delà de 80 mm : hors du domaine du module (refus, `normes.T_MAX_TABLEAU`).

**Nuances et norme produit (lot C, C1b ; brief v12 § 4.1, § 7.1).** La liste du SEUL module Barres est
`acier/barre/nuances.py` : S235, S275, S355, S420, S460 (les modules Flambement, cornière et le Catalogue gardent
`materiaux.NUANCES`, sans S420). La clé `norme_produit` de la barre choisit la ligne de S420 et S460 : vide = N/NL
(EN 10025-3, tableaux 3.4, 3.9 et 3.11 ANB), « M » = M/ML (EN 10025-4, tableau 3.5 ANB ; tubes formés à froid :
tableau 3.12 ANB) ; elle est sans effet sur S235 à S355 (EN 10025-2). Valeurs : S420N 420 / 400 / 390 / 370 au-delà de
16, 40 et 63 mm (f_u 520) ; S420M 420 / 400 / 390 / 380 (f_u 520 / 500 / 480) ; S460M comme S460N jusqu'à 100 mm
(f_u 540 / 530 / 510). Le défaut N/NL est le côté sûr pour f_y (au plus M/ML à toute épaisseur), pas pour f_u, qu'aucune
vérification n'emploie en v1 (À VÉRIFIER). Un produit sans ligne dans l'ANB (S420 formé à froid N/NL, un creux fini à
chaud M/ML) : refus « valeur à fournir ». Le résultat Eurocode 3 (`ec3_section.verifier`, `ec3.verifier`) rend
`norme_produit` (celle qui sert), `fu`, `fu_source` et `fu_ref` (la note dit l'origine de f_y et de f_u, option a) ; la
norme produit est dite une fois, dans la source de f_y. Les plats d'un raidisseur et un PRS prennent la même liste (S420
compris) ; `raidisseurs.dimensionner` reçoit aussi la norme produit. **À FAIRE (lot C, vague 1, C1)** : `moteur.py` et
`ame.py` ne passent pas encore la clé `norme_produit` de la barre à `ES.lire`, `ES.verifier`, `EC.verifier` et
`RA.dimensionner` — par `moteur.verifier`, « M » est donc ignoré sans rien dire : f_y de N/NL (au plus celle de M/ML,
côté sûr), mais f_u et la source de N/NL (HL 920x725 S420 « M » : 370 et f_u 520 au lieu de 380 et 480), et un creux
fini à chaud « M » calculé en S420NH au lieu d'être refusé. Méthode ELS : f_y nominale (420 pour S420), « hors domaine »
hors AE235 à AE355, comme S460.

### 5.2 Classification — § 5.5, tableau 5.2 (vérifié)
Âme en flexion 72ε / 83ε / 124ε ; en compression 33ε / 38ε / 42ε ; en flexion composée : classe 1 396ε/(13α − 1) si
α > 0,5, sinon 36ε/α ; classe 2 456ε/(13α − 1), sinon 41,5ε/α ; classe 3 42ε/(0,67 + 0,33ψ) si ψ > −1, sinon
62ε·(1 − ψ)·√(−ψ). Semelle en console comprimée 9ε / 10ε / 14ε. Tube rond d/t ≤ 50ε² / 70ε² / 90ε². Parois d'un
tube rectangulaire : c = h − 3t, b − 3t. La classe se refait par combinaison (elle dépend de N).

**Ce que le moteur a tranché** (`acier/barre/ec3_classe.py`, lot 2) :
- UNE classe par combinaison — celle de l'état réel (N et M ensemble) — employée par toutes les vérifications de la
  combinaison (comme le logiciel du bureau) ;
- âme fléchie et comprimée : α = 0,5 + N_Ed/(2·c·t_w·f_y) (deux âmes pour un tube : 2·t) et ψ = 2·N_Ed/(A·f_y) − 1,
  les formules d'usage ; N_Ed = 0 redonne 72ε / 83ε / 124ε ;
- du côté sûr : une semelle — ou la paroi d'un tube qui sert de semelle à l'un des deux moments — est classée en
  compression UNIFORME. Pour un I, rien n'est perdu. Pour un tube rectangulaire en flexion DÉVIÉE, la classe peut en
  être dégradée (RHS 200x100x5 S355 : classe 1 sous M_y, classe 4 dès qu'un M_z s'ajoute) — dit dans les hypothèses,
  et inscrit aux améliorations (`AMELIORATIONS_PAR_MODULE.md`, B1).

### 5.3 Classe 4 — EN 1993-1-5 § 4.4 (vérifié)
λ̄_p = (b̄/t)/(28,4·ε·√k_σ) ; paroi interne ρ = (λ̄_p − 0,055·(3 + ψ))/λ̄_p² ≤ 1 au-delà de 0,5 + √(0,085 − 0,055ψ) ;
console ρ = (λ̄_p − 0,188)/λ̄_p² ≤ 1 au-delà de 0,748. A_eff, W_eff, e_N. Tube rond de classe 4 : non couvert.

**Ce que le moteur a tranché** : A_eff sous la compression SEULE et W_eff sous la flexion SEULE (EN 1993-1-5 § 4.3(3)
et (4)) ; e_N = 0 (deux axes de symétrie) ; pour W_eff, la membrure comprimée d'abord, puis l'âme avec le ψ de la
section à membrure réduite (§ 4.4(3)) ; les parties inefficaces sont RETIRÉES de la section brute du catalogue, congés
compris ; W_eff se rapporte à la fibre comprimée de la section BRUTE (du côté sûr quand le bout d'une console est
retiré). I, H, tubes carrés et rectangulaires — le brief ne demandait que les I et H. Contrôle : recalcul par
ASSEMBLAGE des parties efficaces sur des sections sans congé, à 10⁻⁹.

### 5.4 Résistances de section — § 6.2 (vérifié)
N_pl,Rd, N_c,Rd, M_c,Rd par classe ; V_pl,Rd = A_v·(f_y/√3)/γ_M0 — A_v,z = A − 2·b·t_f + (t_w + 2r)·t_f ≥ η·h_w·t_w
(le catalogue porte la première expression, sans le plancher) ; A_v,y = 2·b·t_f pour un I ou H LAMINÉ (lot A, DÉCISION
BUREAU n° 3 ; le catalogue et la Bibliothèque Acier portent encore A − h_w·t_w), A − h_w·t_w pour un I soudé (cas (e)) ;
U (fonction `aires_cisaillement` seulement : le module ne vérifie pas un U, E-SECTION « non couvert ») : A_v,z du catalogue
(cas (b)), A_v,y = min(2·b·t_f, A − h_w·t_w) pour un U laminé — aucun cas de 6.2.6(3) ne le vise, (e) nomme les U
soudés : la plus petite des deux (troisième passe du lot A ; avant, l'A_v,z de l'âme : UPN 380 52,43 cm² au lieu de
32,64) — A − h_w·t_w pour un U soudé ; une cornière est refusée ; la variante de la DÉCISION BUREAU n° 3 (lot C, C1b) :
l'autre règle d'un I ou H laminé (A − h_w·t_w si le défaut 2·b·t_f décide, et l'inverse), son V_pl,y,Rd et son taux,
rendus pour information (`grandeurs["A_v_variante"]` ; HEB 200 S235 : 814,1 kN décide, la variante 851,8 kN), avec SON
texte (`MSG_AVY_VARIANTE` : « Pour information (décision du bureau n° 3, non retenue) : A_v,y = … », jamais celui de la
règle prise) — I soudé, IPN et PRS n'en ont pas ;
tube rectangulaire A·h/(b + h) ; tube rond 2A/π. Flexion + effort
tranchant (6.2.8) ; flexion + effort normal (6.2.9 : (6.33) à (6.41)) ; classe 3 (6.42) ; classe 4 (6.44) ; 6.2.10.
Tube rond en N + M : M_N,Rd = M_pl,Rd·(1 − n^1,7) (§ 6.2.9.1(6), ligne de l'AC:2009 ; lot B — le critère linéaire (6.2)
d'avant était plus sévère : CHS 168,3x6,3, N = 200 kN, M = 14,1 kN·m : 62,9 % → 40,6 %). IPN : A_v,y = min(2·b·t_f ;
A − h_w·t_w), comme les U (brief v12 § 19.2 ; IPN 200 : 20,13 cm² au lieu de 20,34). PRS : § 5.10.

**Ce que le moteur a tranché** (`acier/barre/ec3_section.py`, lot 2) :
- tube rond : la RÉSULTANTE des deux moments et celle des deux efforts tranchants (il n'a pas d'axe) ; M_Ed ≤ M_N,Rd ;
- contrôle élastique (6.1) — § 6.2.1(5), POUR INFORMATION (lot B, test 28 du § 11.5) : aux fibres extrêmes et aux
  jonctions âme-semelles d'un I, σ_x = N/A + M_y·z/I_y + M_z·y/I_z et τ = V_z·S/(I_y·t_w) au même point, σ_z = 0 ; il
  n'entre dans aucun verdict (grandeur « controle_elastique ») ; au-delà de 1, ses formules et un message court. V_y n'y
  est pas compté. La charge sur la semelle tendue (EN 1993-1-5, 7.2(2)), où il décide, est au bloc de l'âme ;
- semelle en console fléchie (M_z présent) : les limites du tableau 5.2 (9ε/α, 10ε/α au bord libre comprimé ;
  9ε/(α·√α), 10ε/(α·√α) au bord libre tendu ; 21ε·√k_σ) sont dans `ec3_classe.limites_console_flechie` ; `classer` les
  prend à leur borne prudente, sans calculer α ni ψ de la console : 9ε et 10ε (classes 1 et 2), et 21ε·√0,43 = 13,77ε
  en classe 3 — le plus petit k_σ du tableau 4.2 de l'EN 1993-1-5, sous les 14ε de la compression uniforme, que
  21ε·√k_σ passe quand ψ approche 1 (lot B ; À VÉRIFIER ; test 29 sur `classer`) ;
- un I ou un H saisi avec I_y < I_z : refusé, « axes inversés ? » (test 15 ; `section.saisie`) ;
- effort tranchant élevé (V_Ed > 0,5·V_pl,Rd) : (6.30) pour un I de classe 1 ou 2 fléchi autour de l'axe fort ; partout
  ailleurs, et dès qu'un effort normal s'ajoute (§ 6.2.10), (1 − ρ) sur TOUTE la section — du côté sûr, et c'est dit ;
- le plancher η·h_w·t_w de l'aire de cisaillement : avec η = 1,00 des laminés (lot A), il ne gouverne pour aucun I du
  catalogue (avec 1,20, il gouvernait 61 profilés : IPE 750, HL, HE 900 et 1000 — mesuré le 16/11/2026) ;
- h_w/t_w > 72·ε/η : le voilement par cisaillement est VÉRIFIÉ depuis le lot 4 (`ec3_ame.cisaillement`, « A-Vb » et
  « A-VbM », § 5.9) — il était rendu « non couvert » avant (11 profilés en S235, 46 en S355, 93 en S460 — tous des tubes
  minces en S235) ;
- chaque paramètre se lit dans `parametres.py` ; une barre peut en surcharger (`"parametres": {…}`) pour rejouer un
  exemple publié avec d'autres coefficients.

**Les benchmarks ECCS N°119** (critère d'acceptation du brief pour ce lot) : leurs pages (aide de SCIA Engineer) ne
sont pas joignables d'ici (domaine hors de la liste autorisée du réseau ; essayé encore au lot B, le 16/11/2026). Les
classes et les résistances sont donc prouvées autrement : limites du texte aux points donnés en clair, profilés de
classe connue, calculs à la main sur les valeurs du catalogue, recoupement avec le module Flambement en service sur tout
le catalogue. `tests/test_barre_benchmarks.py` tient la liste : les benchmarks 17 à 20 et 22 y sont « bloqués — données
à fournir » ; le 21 et les recoupements SCIA (S1 à S3) et LTBeam (L17a) y sont comparés (§ 5.7).

### 5.5 Flambement — § 6.3.1 (vérifié ; déjà en service dans `acier/flambement`)
χ, Φ, λ̄ = √(A·f_y/N_cr) ; α : a₀ 0,13, a 0,21, b 0,34, c 0,49, d 0,76 ; tableau 6.2 ; négligé si λ̄ ≤ 0,2 ou
N_Ed/N_cr ≤ 0,04. Flambement par torsion des I/H (§ 6.3.1.4) : N_cr,T = (G·I_t + π²·E·I_w/L_T²)/i₀², courbe de l'axe z.

**Ce que le moteur a tranché** (`acier/barre/ec3_flambement.py`, lot 3) : les courbes et χ sont ceux du module
Flambement en service (une seule source) ; s'y ajoutent le I soudé (b et c, puis c et d au-delà de 40 mm de semelle) et
le caisson soudé (b). La torsion n'est rendue que si N_cr,T < N_cr,z — sinon elle ne peut pas gouverner. À défaut
d'indication, la torsion et le déversement prennent la longueur de l'axe faible. Le HEB 200 du classeur (7,20 m,
432,72 kN) : N_b,z,Rd = 569,2 kN, taux 76 % (la méthode ELS : 112 %).

### 5.6 Déversement — § 6.3.2 (vérifié ; E1)
M_b,Rd = χ_LT·W_y·f_y/γ_M1 ; laminés (§ 6.3.2.3) : Φ_LT = 0,5·[1 + α_LT·(λ̄_LT − λ̄_LT,0) + β·λ̄_LT²], χ_LT ≤ 1 et
≤ 1/λ̄_LT² ; tableau 6.5 (h/b ≤ 2 : b ; > 2 : c). Facteur f = 1 − 0,5·(1 − k_c)·[1 − 2·(λ̄_LT − 0,8)²] ≤ 1 ;
χ_LT,mod = min(χ_LT/f ; 1 ; 1/λ̄_LT²) (AC:2009). f et tableau 6.6 : obligatoires en Belgique (ANB:2018,
6.3.2.3(2)). k_c (tableau 6.6) : moment constant 1,0 ; linéaire 1/(1,33 − 0,33ψ) ; deux appuis simples, charge répartie
**0,94**, charge concentrée à mi-portée **0,86** ; deux encastrements 0,90 et 0,77 ; un encastrement et un appui simple
0,91 et 0,82 ; diagramme hors tableau : k_c = 1. Négligé si λ̄_LT ≤ λ̄_LT,0 ou M_Ed/M_cr ≤ λ̄_LT,0².

**M_cr : numérique.** Éléments finis de poutre à deux champs (déplacement latéral v, rotation de torsion φ, fonctions
d'Hermite) :

    Π = ½∫(E·I_z·v″² + E·I_w·φ″² + G·I_t·φ′²)dx + ∫M_y·v″·φ dx − ½∫q·z_g·φ² dx − ½·Σ P_i·z_g,i·φ(x_i)²

z_g > 0 pour une charge vers le bas appliquée au-dessus du centre de cisaillement. Appui à fourche : v = φ = 0.
M_cr = λ_cr·|M_y|max. Par TRONÇON entre deux sections tenues en déplacement latéral ET en torsion ; un maintien d'une
seule semelle est une liaison à l'intérieur du tronçon (v ∓ (h − t_f)/2·φ = 0). Contrôle : formule du NCCI SN003.

**Ce que le moteur fait** (`acier/barre/mcr.py`, `ec3_deversement.py`, lot 3) :
- le solveur est en Python pur (il tourne dans le navigateur) : matrices en bande, décomposition LDLᵀ. Le COMPTE des
  pivots négatifs encadre le premier multiplicateur critique ; dans l'encadrement, la sécante sur le déterminant (le
  produit des pivots) converge en une dizaine de décompositions — la bissection seule en demandait 35. Deux chargements
  proportionnels (1,35·G et 1,00·G ; G et Q de même forme) ont le même M_cr : il n'est calculé qu'une fois. Mesuré le
  14/11/2026 : 5 millièmes de seconde par moment critique (33 avant), 9 pour la barre neuve de l'écran (80 avant),
  140 pour une poutre continue de trois travées à trois cas (590 avant) — résultats inchangés à 10⁻¹⁰.
  Convention : z vers le haut, φ dans le sens
  direct, u(z) = v − z·φ, donc Π porte **− ∫M_y·v″·φ** (le signe ne compte que pour un maintien d'une seule semelle) ;
- il redonne les sept valeurs R1 du solveur indépendant au centième de kN·m, la formule fermée à 10⁻⁶, la console
  classique (4,013), et « un maintien de la semelle comprimée à mi-portée vaut un maintien complet » (250,95 kN·m) ;
- bouts d'un tronçon : fourche, encastrement, libre (la console) ; maintien d'une semelle : liaison exacte ;
- mode « efforts imposés » : quatre diagrammes (constant, linéaire ψ, charge répartie, charge concentrée), la charge
  au centre de cisaillement, sur la semelle haute ou sous la basse, DESCENDANTE quand M_y,Ed > 0 ; quand M_y,Ed < 0,
  le sens réel n'est pas connu (une charge de gravité, ou un soulèvement, le seul qui donne ce diagramme entre deux
  fourches) : le module calcule M_cr dans les DEUX sens et garde le plus petit, pour toute section, et l'hypothèse le dit
  en une ligne — À VÉRIFIER (PRS-B, 10 m, charge répartie sous la semelle basse : 628,60 kN·m et non 1 343,80, Ritz
  indépendant ; un laminé : sous la semelle basse comme sur la haute). Le signe change aussi z_j et la semelle comprimée
  d'un PRS monosymétrique ; un diagramme non décrit est pris CONSTANT ;
- un tube rectangulaire relève du cas général (§ 6.3.2.2, courbe d) ; tubes ronds et carrés : pas de déversement ;
- le couple (λ̄_LT,0 ; β) : (0,2 ; 1,0), UN couple (lot A) — NBN EN 1993-1-1 ANB:2018, 6.3.2.3(1) : M_cr calculé sur la
  section brute avec les combinaisons, la distribution réelle des moments et les maintiens, ce que fait le solveur du
  module, maintiens présents ou non ; (0,4 ; 0,75) suppose les maintiens TOTALEMENT ignorés dans M_cr, une option que le
  module n'a pas. Le seuil de 6.3.2.2(4) des tubes lit le même λ̄_LT,0 (0,2). Les clés « sans maintiens » de la table
  sont retirées, et une surcharge qui les nomme est refusée avec son motif.

**Ce que le lot B a ajouté** (brief v12, § 7.7 et 19.2 ; `ec3_deversement.py`, `mcr.py`, tests 1, 9, 10, 20 et 21
du § 11.5 dans `tests/test_barre_ec3_barre.py`) :
- **k_c reconnu sur tout le tableau 6.6** (`kc_portee`) : une portée seule, tronçon = la portée, sans maintien intérieur,
  sous des charges uniformes sur toute la portée ou des charges à mi-portée seulement — deux appuis 0,94 / 0,86 ;
  bi-encastrée 0,90 / 0,77 ; encastrée-appuyée 0,91 / 0,82 ; et le moment linéaire sur un tronçon de la barre chargée
  entre deux fourches, sans maintien ni charge dedans (`kc_lineaire`, ψ lu sur le diagramme de la combinaison). Tout le
  reste garde k_c = 1 (f = 1) : un encastrement partiel (couples d'extrémité et charge répartie), des charges mêlées
  (cas S1 de SCIA : 1,099), une console, une travée de poutre continue chargée ;
- **la variante (0,4 ; 0,75)** de l'ANB:2018, 6.3.2.3(1), sur demande seulement (`deversement.couple` = « variante ») :
  poutre de bâtiment (`ouvrage`), au moins un maintien présent et TOUS ignorés dans M_cr (`maintiens_negliges`). Le
  module ne l'admet que s'il voit que M_cr ne compte aucun maintien : M_cr du solveur (pas un M_cr donné), aucun
  maintien dans le tronçon, et un tronçon qui va d'un appui à l'autre — la barre chargée le dit pour chaque tronçon
  (`entre_appuis`, faux s'il est borné par un maintien complet), en efforts imposés L_c doit couvrir la barre (la plus
  grande longueur des axes). Ailleurs elle est refusée, avec un motif court (« hors bâtiment », « aucun maintien »,
  « M_cr donné », « maintiens comptés dans M_cr », « portée non donnée »), et (0,2 ; 1,0) s'applique. Un couple par
  défaut changé par surcharge de la table (autre annexe, cas d'essai) s'applique partout, sans ce contrôle : l'alerte
  « couple_hors_anb » le dit ;
- **DÉCISION BUREAU n° 4** : le § 6.3.2.3 décide pour les I et H laminés (`methode_deversement_lamines`, table des
  paramètres) ; le § 6.3.2.2 (tableau 6.4 : courbe a si h/b ≤ 2, b au-delà) est calculé à côté et rendu « pour
  information » (`grandeurs.deversement.information` : χ_LT, M_b,Rd, taux, source). R3 : 0,5838 / 86,22 / 0,797 et
  0,4687 / 69,21 / 0,993. L'affichage à l'écran et dans la note est au lot C ;
- **le contrôle analytique de l'annexe E ANB** : une seule table, celle du moteur (`mcr.py` : `coefficients_annexe_e`,
  `annexe_e`, `kwt` ; § 5.10 pour z_j et C₃) — (E.1), qui redonne (E.8) quand z_j = 0, tableaux E.1 ANB et E.2 ANB pour
  k_z = k_ω = 1, note b (C₁/1,05 si K_ωt ≤ 1, jamais sous 1). Un ψ qui n'est pas une ligne du tableau E.1 ANB : contrôle
  bloqué, le solveur seul (l'équation (E.2) n'est pas employée). Il redonne les sept valeurs R1 de la colonne ANB au
  centième, avec G = 81 000 et avec G = E/2,6. Le solveur reste la référence ;
- **C₁ de l'annexe D ANB** (étape 1) : M_cr/M_cr,0 du solveur, les charges ramenées au centre de cisaillement (annexe E
  ANB, méthode avancée) — et non k_c⁻² (AC:2009), que l'ANB écarte. Charge centrale : 1,36 au lieu de 1,352.

### 5.7 Flexion composée avec instabilité — § 6.3.3, annexe D ANB (vérifié ; E2 ; lot B)
(6.61) et (6.62) ; méthode 1. Tableau A.2 : moment linéaire C_mi,0 = 0,79 + 0,21ψ + 0,36·(ψ − 0,33)·N_Ed/N_cr,i ;
charge répartie **1 + 0,03**·N_Ed/N_cr,i ; charge concentrée **1 − 0,18**·N_Ed/N_cr,i ; cas général
1 + (π²·E·I_i·|δ_x|/(L²·|M_i,Ed(x)|) − 1)·N_Ed/N_cr,i.

**Ce que le moteur fait** (`acier/barre/ec3_interaction.py`, lot 3) : (6.61) et (6.62) avec tous les facteurs de
l'annexe A (μ, w ≤ 1,5, n_pl, a_LT, C_my, C_mz, C_mLT, b_LT à e_LT, C_yy à C_zz et leurs planchers, k_yy à k_zz ;
facteurs élastiques en classes 3 et 4) ; C₁ = 1/k_c² ; une barre sans déversement possible a λ̄_0 = 0 ; une flexion
déviée SANS effort normal passe par les mêmes expressions ; une barre tendue est prise sans effort normal (du côté
sûr). Une CONSOLE sous effort normal et flexion : (6.61)/(6.62) rendues « non couvert » (lot A, brief § 7.8, DÉCISION
BUREAU n° 9 : aucun C_m ne vaut pour une console) ; N seul et M seul restent vérifiés.

**Lot B : l'annexe D ANB codée en entier** (ANB:2018, 6.3.3(5) : méthode 1 normative, écrite par l'annexe D ANB, prise
comme un tout ; pages 21 à 28 relues) :
- **étape 1** : I_t ≥ I_y → « non susceptible de se déformer par torsion » ; sinon le critère
  λ̄_0 ≤ 0,2·√C₁·[(1 − N_Ed/N_cr,z)·(1 − N_Ed/N_cr,TF)]^¼, avec N_cr,TF = N_cr,T pour une section à deux axes de
  symétrie ((F.4) redonne (F.3)) et C₁ de l'annexe E ANB (§ 5.6) ;
- **étape 2.a** : ni χ_LT ni C_mLT dans (6.61) et (6.62), C_my = C_my,0, C_ij sans b_LT, c_LT, d_LT, e_LT ;
  **étape 2.b** : les formules complètes. La note dit l'étape et sa raison ;
- **C_mi,0 sans borne inférieure** (tableau D.1 ANB ; réponse du propriétaire, brief § 19.1 point 2) : ψ = −1 et
  N_Ed/N_cr = 0,1 donnent 0,5321. Le paramètre « borne_Cmi0 » est retiré de la table (une surcharge qui le nomme est
  refusée). SCIA S2 : (6.61) 0,600 et (6.62) 0,648, comme SCIA (0,60 ; 0,65) — 0,631 et 0,667 avec l'ancienne borne ;
- **C_zz : e_LT DANS le facteur n_pl** (annexe D ANB p. 28 ; AC:2009, modification 35 b). Le moteur gardait l'écriture
  de 2005 (e_LT dehors), plus sévère : quand N_Ed tend vers 0, C_zz tend maintenant vers 1. Effet sur une barre
  comprimée et fléchie autour des deux axes, à l'étape 2.b : k_zz baisse (benchmark 21 : (6.62) 0,986 → 0,873) ;
- **n_pl avec γ_M1** (annexe D ANB) ; si le client fixe γ_M0 ≠ γ_M1, la variante γ_M0 de l'annexe A corrigée est
  calculée et affichée (message, paramètre « npl_gamma » à confirmer) : l'équivalence n'est pas établie ;
- **classes 3 et 4** : w = 1, C_yy = C_zz = 1, C_yz = C_zy = 0,6 — les k_ij sont ceux de l'annexe A sans ses facteurs.

**Le benchmark 21 de l'aide de SCIA Engineer** (ECCS N°119, exemple 5 : IPE 500 S235, 3,75 m, N_Ed = 500 kN, flexion
déviée). Sa page n'est pas joignable d'ici ; la recherche en a rendu les données et une partie des résultats. Sont
reproduits : classe 1, N_c,Rd 2 715 kN, M_c,y,Rd 516, M_c,z,Rd 78,9 kN·m, V_pl,z,Rd 814 kN (0,3 %), V_pl,y,Rd 912 kN
(1 %, avec A − h_w·t_w ; le module prend 2·b·t_f depuis le lot A, DÉCISION BUREAU n° 3 : 868,3 kN) ; μ_y 1,00, w_y 1,138, μ_z 0,937, w_z 1,5, C_mz,0 0,771, M_cr,0 899 kN·m, λ̄_0 0,757, a_LT 0,998 (0,2 %).
Lot B : les six facteurs du brief qui dépendent du diagramme de M_y, avec M_y = 200 kN·m sous une charge concentrée à
mi-portée — diagramme DÉDUIT de C_my,0 = 0,999 = 1 − 0,18·N_Ed/N_cr,y, à vérifier sur la page (Q8) : C_my,0, b_LT et
C_yy au chiffre près ; c_LT, d_LT, e_LT 1,8 à 2,0 % au-dessus (tous ∝ 1/χ_LT : ils impliquent χ_LT ≈ 0,826 dans la
référence, le module prend 0,811 : § 6.3.2.3, courbe c, f avec k_c = 0,86). Ce 0,826 reste à expliquer (règle 7) :
aucune lecture essayée ne le donne — § 6.3.2.2 courbe b 0,812 ; § 6.3.2.3 courbe b avec f 0,836 (k_c = 0,94) et
0,870 (0,86) ; couple (0,4 ; 0,75) courbe c 0,857 (k_c = 1) et 0,919 (0,86) ; un extrait de la page dit que la référence prend le
cas général du déversement en méthode 1 (À VÉRIFIER, Q8). Le test les tient à ± 3 %, sans borne basse. χ_z 0,644,
λ̄_y 0,195, λ̄_z 0,927 et les courbes a / b de la référence sont aussi reproduits. Les taux de la référence manquent.
L'annexe D ANB est recalculée d'une traite dans le test, pour neuf barres, à 10⁻⁹ (plus un tronçon court en étape 2.a
avec χ_LT < 1, et un HEB 200 court dont λ̄_0 tombe entre les limites avec et sans le facteur (1 − N_Ed/N_cr,TF)).

### 5.8 Les paramètres nationaux (une seule table dans le moteur : `acier/barre/parametres.py`)

| Paramètre | Valeur | Source | Statut |
|---|---|---|---|
| γ_M0 ; γ_M1 ; γ_M2 | 1,00 ; 1,00 ; 1,25 | NBN EN 1993-1-1 ANB:2018, 6.1(1)B (bâtiments) | pointé (lot C) |
| E ; G | 210 000 ; 81 000 N/mm² | EN 1993-1-1 § 3.2.6(1) (G arrondi par le même paragraphe) | pointé (lot C) |
| λ̄_LT,0 ; β (§ 6.3.2.3) | 0,2 ; 1,0 (le couple par défaut ; la variante, ligne suivante, sur demande) | NBN EN 1993-1-1 ANB:2018, 6.3.2.3(1) : M_cr avec les combinaisons, la distribution réelle des moments et les maintiens (§ 5.6) | pointé |
| Variante (λ̄_LT,0 ; β) | 0,4 ; 0,75 — sur demande, poutre de bâtiment, maintiens tous ignorés dans M_cr (lot B) | ANB:2018, 6.3.2.3(1) | pointé |
| Facteur f et tableau 6.6 (§ 6.3.2.3(2)) | formule EN, tableau complet (lot B) | ANB:2018, 6.3.2.3(2) : normatifs | pointé |
| Méthode de déversement des I et H laminés | § 6.3.2.3 ; § 6.3.2.2 pour information | ANB:2018 (n'impose rien) | DÉCISION BUREAU n° 4 |
| Déversement d'un I soudé, PRS ou I saisi « soudé » (`prs_soudee_equivalente`) | non : § 6.3.2.2 (k_c = 1, f = 1), § 6.3.2.3 pour information ; oui (section bisymétrique « soudée équivalente ») : § 6.3.2.3, § 6.3.2.2 pour information ; un PRS monosymétrique : § 6.3.2.2 seul | EN 1993-1-1 § 6.3.2.3(1) : « soudées équivalentes », équivalence non définie | DÉCISION BUREAU n° 10 (défaut : non) |
| η₁ de l'interaction (7.2) de l'âme (`eta1_ame`) | élastique (W_el) ; la variante plastique (W_pl, classes 1 et 2) dite à côté | EN 1993-1-5 § 7.2 | DÉCISION BUREAU n° 1 (lot C) |
| Contact d'extrémité « inconnu » d'une force du type (c) (`contact_inconnu`) | non : verdict sur s_s = 0 dans (6.10) ; « oui » : s_s gardé ; l'autre lecture dite pour information | EN 1993-1-5 § 6, (6.10) | DÉCISION BUREAU n° 2 (lot C) |
| A_v,y d'un I ou H laminé (`avy_lamine`) | 2·b·t_f ; variante A − h_w·t_w | EN 1993-1-1 | DÉCISION BUREAU n° 3 (lot C ; Q3) |
| Console en flexion composée N + M (`console_NM`) | non couvert ; aucune variante calculable tant que le bureau n'a pas fixé de méthode | § 6.3.3, annexe D ANB | DÉCISION BUREAU n° 9 (lot C) — **À VÉRIFIER** |
| Méthode de § 6.3.3 | méthode 1, écriture de l'annexe D ANB | ANB:2018, 6.3.3(5) | pointé (lot B) |
| Borne inférieure de C_mi,0 | aucune (clé retirée, lot B) | tableau D.1 ANB | pointé |
| γ de n_pl quand γ_M0 ≠ γ_M1 | γ_M1 (annexe D ANB) ; la variante γ_M0 affichée | annexe D ANB / AC:2009 | **À CONFIRMER** (seulement si γ_M0 ≠ γ_M1) |
| η (EN 1993-1-5 § 5.1) | 1,00 pour les I et H laminés et les tubes ; 1,20 pour les sections soudées (PRS, I soudés, caissons), jusqu'au S460 (lot A ; PRS : lot B) | NBN EN 1993-1-5 ANB:2011, 5.1(2), note complémentaire (renvoi au 6.2.6(3) de l'EN 1993-1-1) | pointé (lot B) |
| γ_G,sup ; γ_G,inf ; γ_Q | 1,35 ; 1,00 ; 1,5 (0 si favorable) | NBN EN 1990 ANB:2021, tableau A1.2(B), CC2 | pointé (lot C) |
| Combinaison ELU | (6.10) par défaut ; option (6.10a) et (6.10b), la plus défavorable retenue | NBN EN 1990 ANB:2021, tableau A1.2(B) | pointé (lot C) |
| ξ de (6.10b) | 0,85 (ξ·γ_G,sup = 1,1475) ; 1,00 pour une application géotechnique | NBN EN 1990 ANB:2021, tableau A1.2(B) | pointé (lot C) |
| ψ₀ / ψ₁ / ψ₂ | A, B : 0,7/0,5/0,3 ; C, D, F : 0,7/0,7/0,6 ; E (fluides et matières stockés compris) : 1,0/0,9/0,8 ; G : 0,7/0,5/0,3 ; H : 0/0/0 | NBN EN 1990 ANB:2021, tableau A1.1 | pointé (lot C) |
| ψ de la neige, du vent, de la température | 0,5/**0**/0 (ψ₁ = 0, altitude ≤ 1 000 m) ; 0,6/0,2/0 ; 0,6/0,5/0 | NBN EN 1990 ANB:2021, tableau A1.1 | pointé (lot C ; ψ₁ de la neige valait 0,2, la valeur de l'EN) |
| ψ₀ de courte durée | 0,3 — la neige et le vent seulement (mise en œuvre restreinte, ci-dessous) | NBN EN 1990 ANB:2021, tableau A1.1, note 3 | pointé (lot C) |
| ψ₀ de l'exécution, des machines | saisi, sans défaut (exécution : 1,0 / – / 0,2 au tableau, au cas par cas) | tableau A1.1 ANB, notes 1 et 2 | une donnée de la barre (lot C) |
| Combinaison de la méthode ELS | G + ΣQ, tout à 100 % | décision du bureau du 16/11/2026 (brief § 13.2 point 5) | tranché (Q4) |

Les DÉCISIONS BUREAU (n° 1, 2, 3, 4, 9, 10) ont UN statut dans la table (`DECISION`) et une entrée {valeur par défaut,
variante, valeurs admises, point du § 13.2, source} ; chaque moteur lit la sienne par `parametres.jeu()` — changer le
défaut dans la table change le verdict, l'autre variante est rendue à côté (`tests/test_barre_moteur.py`, mutations du
lot C). `parametres.decisions()` les rend dans l'ordre de leur numéro (ce que la note imprimera, C5). Le seuil de 95 %
(n° 7) n'y est pas : c'est celui du site (`api_web/commun.py`).

« À pointer » (plus aucune entrée depuis le lot C : les trois annexes belges sont lues, brief § 13.1) : la valeur de l'EN,
connue, restait à lire sur le texte belge. « À CONFIRMER » : la valeur change un résultat et sa source n'est pas le
texte — elle est imprimée comme telle dans la note ; il n'en reste qu'un, n_pl (`tests/test_barre_combinaisons.py`).

**Les combinaisons, telles que le moteur les écrit** (`acier/barre/combinaisons.py` ; lot 5, puis lot C, C2) : pour
chaque ARRANGEMENT des cas variables (chacun présent ou absent — la disposition défavorable, travée par travée), chaque
JEU de γ_G et chaque nature en tête parmi les présentes :

- **(6.10)**, le défaut : Σ γ_G·G + 1,5·Q₁ + Σ 1,5·ψ₀·Q_i ;
- **(6.10a) et (6.10b)**, l'option « 6.10ab » : (6.10a) 1,35·G_sup + Σ 1,5·ψ₀·Q_i (toutes les variables à ψ₀ ; Q₁,
  l'accompagnement principal, à son ψ₀,₁) ; (6.10b) ξ·1,35·G_sup + 1,5·Q₁ + Σ 1,5·ψ₀·Q_i ; G_inf = 1,00 dans les deux ;
  ξ = 1,00 en géotechnique. Les deux sont versées : l'enveloppe garde la plus défavorable. Sans action variable,
  (6.10b) vaut ξ fois (6.10a) sur les G défavorables : elle n'est pas versée.
- Les cas d'une MÊME nature sont une seule action disposée librement : en tête, ils reçoivent tous γ_Q.
- **γ_G par source** (`À VÉRIFIER`) : chaque cas permanent est une source, γ_G,sup ou γ_G,inf indépendamment (le
  côté sûr ; le texte de l'EN 1990 n'est pas lu) — un G défavorable sur une travée (ou un débord), un G favorable sur
  l'autre. Deux cas de même répartition (charges proportionnelles, aux mêmes abscisses et niveaux : le poids propre et
  un G réparti sur toute la barre) font une source : mêlés, leurs γ donneraient les mêmes efforts multipliés par un
  facteur compris entre les deux extrêmes. Continue de deux travées de 5 m, G₁ = G₂ = 10 kN/m : « 1,35·G1 + G2 »
  donne 25,826 kN·m en travée 1, contre 23,730 pour 1,35·(G₁ + G₂).
- **Règle belge de courte durée** (note 3 du tableau A1.1 ANB), mise en œuvre RESTREINTE du brief § 5 : seules la
  neige et le vent reçoivent ψ₀ = 0,3 — derrière l'autre, d'office ; derrière une action déclarée « de courte durée »
  (tous les cas présents de sa nature) ; « derrière » : en tête dans (6.10) et (6.10b), accompagnement principal dans
  (6.10a). Une charge d'exploitation ou une température d'accompagnement garde le ψ₀ du tableau. Option « règle
  désactivée » : le tableau partout. Quand deux actions de courte durée suivent une action de longue durée, la
  combinaison porte « limite » et la note le dit.
- **Exécution, machines** : ψ₀ saisi par cas, sans défaut ; demandé seulement quand il sert (le cas accompagne une
  autre nature, ou l'expression est (6.10a)). S'il manque alors, `combinaisons.elu` lève `AFournir` (« ψ₀ à fournir »),
  que le moteur et l'âme rendent en statut « bloqué » (`combinaisons.statut_refus`, lu par `moteur.py` et `ame.py` ;
  intégration du lot C) : l'Eurocode 3 et l'âme « bloqué », la raison en une ligne, la comparaison « bloqué », la
  méthode ELS (qui ne lit pas ψ₀) calculée et jugée — jamais une exception (elle arrêtait l'écran et la construction
  du site ; le contrôle : `tests/test_barre_combinaisons.py`, « de bout en bout »). Non saisi et sans emploi (l'exécution seule, en (6.10)) : les hypothèses
  disent « non saisi, sans emploi ici ». Un ψ₀ saisi sur une autre nature ne sert pas. À la saisie, l'écran contrôle
  par `combinaisons.controler`, où un ψ₀ manquant n'est pas un refus (le calcul est bloqué, la saisie reste libre).
- **Limites** : au plus quatre NATURES variables (le nombre de cas : `entree.CAS_MAX` = 6). Au plus 512 combinaisons
  ELU quand il y a plusieurs sources permanentes (une seule est toujours admise) : au-delà, la barre est refusée
  (« au plus k cas permanents de répartitions différentes »), jamais une combinaison abandonnée en silence. Ce budget
  compte des combinaisons, pas du temps, et aucune barre que l'écran admet ne l'atteint (six cas et le poids propre :
  573 au plus, à une source ; 439 à trois) — un budget en temps, ou γ_G,sup et γ_G,inf mêlés seulement là où cela
  peut gouverner, est une décision du propriétaire (`À VÉRIFIER`, mesures ci-dessous).
- **Noms** : « 1,35·G + 1,5·Q » — plusieurs cas permanents au même γ s'écrivent « G » (chacun sinon : « 1,35·G1 +
  G2 ») ; « (6.10a) » ou « (6.10b) » en suffixe ; le facteur s'écrit à deux décimales (1,1475 → « 1,15 »).
  `combinaisons.hypotheses(options, cas)` rend les lignes de la note, une chacune.

Mesures (17/11/2026, CPython, `moteur.verifier`) — nombre de combinaisons ELU, quatre natures (Q, neige, vent,
température), un cas chacune :

| Sources permanentes | (6.10) | (6.10a)/(6.10b) |
|---|---|---|
| 1 | 65 | 109 |
| 2 | 131 | 219 |
| 3 | 263 | 439 |
| 4 | refusée (527) | refusée (879) |

Six cas variables de quatre natures, une source : 321 et 537 ; la barre la plus lourde que l'écran admet : 573
((6.10a)/(6.10b), une source, six cas de quatre natures). Continue HEA 200 de 5 × 5 m, quatre natures — la plus
défavorable de deux séries de mesures (17/11/2026) : deux sources, 3,9 s en (6.10) et 7,5 s en (6.10a)/(6.10b) ; trois
sources, 8,1 s et 15,5 s ; la barre à 573 combinaisons, 16,9 s (avant le lot C : 65 combinaisons, 2,0 s). Le seuil du
plan (deux fois le temps du lot B) est dépassé ; le navigateur est plus lent et le sélecteur multiplie ce temps par le
nombre de profilés essayés. Sélecteur d'une barre de 6 m, poids propre + G réparti et ponctuel (deux sources) : 1,1 → 2,5 s ;
la barre neuve (une source) : inchangée (0,6 s).

Méthode ELS : G + ΣQ à 100 % pour chaque arrangement (décision du bureau du 16/11/2026, brief § 13.2 point 5).

### 5.9 EN 1993-1-5 : âme et raidisseurs (vérifié ; en service depuis le lot 4)

**Le texte** (EN 1993-1-5:2006 ; η = 1,20 jusqu'au S460 dans l'EN — la NBN EN 1993-1-5 ANB:2011, 5.1(2), le réserve aux
sections soudées : 1,00 pour les laminés et les tubes, lot A). Voilement par cisaillement (§ 5 : à vérifier si
h_w/t_w > 72·ε/η ; V_bw,Rd = χ_w·f_yw·h_w·t/(√3·γ_M1) ; λ̄_w = h_w/(86,4·t·ε) avec des raidisseurs aux seuls appuis ;
χ_w du tableau 5.1 ; V_bf,Rd ; k_τ = 5,34 + 4·(h_w/a)² si a/h_w ≥ 1, sinon 4 + 5,34·(h_w/a)²). Âme sous une force
concentrée (§ 6 : types (a), (b), (c) ; k_F ; F_cr = 0,9·k_F·E·t_w³/h_w ; m₁ = f_yf·b_f/(f_yw·t_w) ; m₂ = 0,02·(h_w/t_f)²
si λ̄_F > 0,5 ; l_y ; l_e = k_F·E·t_w²/(2·f_yw·h_w) ≤ s_s + c — le facteur 2 est celui du texte (6.13) ; λ̄_F ;
χ_F = 0,5/λ̄_F ≤ 1 ; F_Rd = f_yw·χ_F·l_y·t_w/γ_M1 ; s_s ≤ h_w). Interactions (§ 7.1 ; § 7.2 : η₂ + 0,8·η₁ ≤ 1,4).
Voilement induit par la semelle (§ 8). Raidisseurs (§ 9 : section efficace — les plats et 15·ε·t_w d'âme de chaque
côté ; flambement sur 0,75·h_w, courbe c ; I_T/I_p ≥ 5,3·f_y/E ; raidisseurs intermédiaires I_st, N_st).

**Ce que le moteur fait** (`acier/barre/ec3_ame.py`, `raidisseurs.py`, `ame.py` ; lot 4). Un TROISIÈME bloc de
résultats, « âme », à côté des deux méthodes, et TOUJOURS calculé — sous les combinaisons ELU (6.10) de la barre
chargée, quelle que soit la méthode cochée : la méthode ELS ne traite rien de tout cela (§ 4.6).

- **Voilement par cisaillement** (« A-Vb »), dans chaque section à vérifier de chaque combinaison, dès que l'âme est
  élancée : V_Ed ≤ V_bw,Rd. Du côté sûr : pas de raidisseur intermédiaire, montant d'extrémité NON rigide
  (χ_w = 0,83/λ̄_w), la contribution des semelles V_bf,Rd négligée. Un tube carré ou rectangulaire a deux âmes. Puis
  l'interaction (7.1) (« A-VbM ») quand V_Ed > 0,5·V_bw,Rd et M_Ed > M_f,Rd — vérifiée dans toutes les sections, même à
  moins de h_w/2 d'un appui (du côté sûr) ; sous un effort normal, ou pour un tube, elle est rendue « non couvert ».
  Avec η = 1,00 des laminés (lot A), 55 couples profilé – nuance du catalogue ont une âme élancée en S235, S355 et S460
  (60 avec le S275) : l'IPE 750x134 en S355, des IPE, IPE A, HEA, HEAA et HL hauts en S460, des tubes minces — 147 avec
  η = 1,20 (170 avec le S275 ; 150 avant le lot A, avec l'ancienne f_y et un seul ε) — compté le 16/11/2026 sur la table
  actuelle (`ec3_section.lire` puis `ec3_ame.elancement`, ε de l'âme, sur tout le catalogue).
- **Les LIEUX** où une force appuie sur l'âme : chaque APPUI simple où la barre est POSÉE (la réaction pousse la semelle
  basse) ; chaque CHARGE PONCTUELLE qui pousse la semelle où elle est posée (vers le bas sur la semelle haute, vers le
  haut sous la semelle basse). Un appui « par l'âme » (platine, cornières, gousset), une charge appliquée au centre de
  cisaillement, un encastrement n'appuient sur aucune semelle : ils ne sont pas des lieux, et les hypothèses le disent.
  Les charges ponctuelles de plusieurs cas à la même abscisse sont UNE force (la plus courte de leurs longueurs
  d'appui) ; une charge au droit d'un appui partage son lieu.
- **Le TYPE** (figure 6.1) se déduit de la géométrie — il n'est pas saisi : (c) près d'un bout de la barre, tant que
  2 + 6·(s_s + c)/h_w < 6, c étant la distance du bout de la barre au bord de la longueur chargée (le débord saisi,
  pour un appui de rive) ; (a) ailleurs ; (b), en plus, pour la force qui passe d'une semelle à l'autre quand un lieu
  est chargé des deux côtés (la plus petite des deux forces, la plus courte des deux longueurs d'appui).
- **La longueur d'appui rigide s_s** se saisit — pour les appuis (une valeur pour la barre, avec le débord c en rive),
  et pour chaque charge ponctuelle. Elle n'est jamais supposée (brief § 8.2) : « non saisie » (None) n'est pas 0 saisi —
  elle est prise à sa borne, et dite (ci-dessous).
- **s_s ou c NON SAISI** (champ vide : « borne 0 » ; brief § 8.3, point 5, À VÉRIFIER) : calcul à la borne 0, sous F_Rd
  réel (balayage). Tout passe : « sans raidisseur (… : borne prudente) » ; sinon « décision impossible : … à saisir », taux
  à la borne.
  (Seul est demandé ce dont dépend ce qui ne passe pas : le critère 6.2.1(5) ne lit que s_s, et le prend, s_s vide, au pire de
  0 et h_w. Une note du site d'avant le lot B — sans la clé « zero_saisi » — est relue avec ses 0 « non saisis ».)
- **DÉCISION IMPOSSIBLE** — le lieu est « non couvert », jamais « pas de raidisseur », une ligne par manque en pied du
  bloc, son taux donné pour information : s_s ou c quand la borne ne passe pas ; le MAINTIEN latéral de la semelle
  chargée (§ 6.1(1) : la section 6 la suppose tenue), qui n'a pas de borne — tenue à un appui (une fourche), ou par un
  maintien « complet » ou de cette semelle à la même abscisse ; quand elle n'est pas comprimée partout, l'autre aussi ; pour
  un raidisseur sous une charge, les DEUX semelles (longueur 0,75·h_w, § 9.4(2)). Seul le maintien manque : le fait, le taux
  des vérifications (faites comme si elle était tenue) et le remède, en une ligne — « Semelle haute sans maintien
  (EN 1993-1-5, 6.1(1)) — 37 % si elle est tenue ; ajouter un maintien (À VÉRIFIER). » ; au-delà de 100 %, « ajouter un maintien
  complet et un raidisseur » ; la ligne du lieu le dit sans le taux, que sa pastille donne (règle prudente gardée ; le brief
  § 11.3 dit « pas de raidisseur » pour R2 : À TRANCHER). Le raidisseur d'appui d'une âme élancée (§ 5.1(2)) est dû
  quelles que soient les données. Barre neuve : « s_s à saisir » (157 % à la borne, critère 6.2.1(5)) ; à
  s_s = 100 mm, sans raidisseur (22 %, c = 0 : borne). R2 : appuis sans raidisseur (borne, 61 %) ; sous la charge, sans
  maintien à mi-portée, la ligne ci-dessus (37 %) ; avec un maintien à 3 m, sans raidisseur.
- **La résistance de l'âme** (lot A) : F_Rd = f_y·χ_F·l_y·t_w/γ_M1 avec la f_y de CALCUL de la section ; λ̄_F avec la f_y
  de l'âme selon t_w. Les DEUX branches de m₂ (m₂ = 0,02·(h_w/t_f)², cohérente si λ̄_F > 0,5 ; m₂ = 0, si λ̄_F ≤ 0,5) sont
  calculées : quand les deux sont cohérentes, la plus PETITE F_Rd est retenue, et dite (brief § 8.2). Type (c) : le
  CONTACT d'extrémité est « inconnu » (DÉCISION BUREAU n° 2) — s_s = 0 dans (6.10) seulement, k_F et le plafond de l_e
  gardent s_s ; la variante avec contact est donnée pour information (IPE 300, s_s = 100 mm, c = 0 : 199,8 kN, 248,9 kN
  avec contact).
- **L'âme seule**, par lieu et par combinaison : η₂ = F_Ed/F_Rd (« A-Lnn-F » ; « A-Lnn-B » pour la force transmise),
  puis, dès qu'un moment ou un effort normal s'y trouve, (η₂ + 0,8·η₁)/1,4 ≤ 1 (« A-Lnn-FM ») avec η₁ ÉLASTIQUE —
  |N|/(f_y·A) + |M|/(f_y·W_el), A_eff et W_eff en classe 4 (Q10) — la classe prise sous le N et le M de l'occurrence
  (l'IPE 400 est de classe 3 en compression pure, de classe 1 sous un moment).
- **LA DÉCISION**, lieu par lieu : « sans raidisseur », avec les taux ; ou « raidisseur » dès qu'une combinaison dépasse
  η₂ = 1 ou l'interaction — et à chaque appui posé d'une âme élancée au cisaillement (§ 5.1(2)). La raison est écrite
  (« Sans raidisseur : F_Ed = 675,0 kN > F_Rd = 585,2 kN » ; ou le critère 6.2.1(5) d'une semelle tendue).
- **Le raidisseur** : une paire de plats symétriques, sur toute la hauteur de l'âme. Épaisseur : la première de 8, 10,
  12, 15, 20 mm qui porte la plus grande force du lieu. Largeur : la demi-semelle (b − t_w)/2 au plus, au pas de 5 mm,
  réduite tant que le plat est trop élancé pour son épaisseur (b_s/t_s ≤ 14·ε ; I_T/I_p ≥ 5,3·f_y/E — c'est la torsion qui
  décide : b_s/t_s ≤ 12,97 en S235). Chanfrein : le congé du profilé, arrondi à 5 mm vers le haut (un IPN : r₁ = t_w).
  Deux vérifications : l'ÉCRASEMENT au contact de la semelle, plats moins chanfreins (« A-Lnn-R » : F_Ed ≤ A_c·f_y/γ_M0)
  — c'est lui qui dimensionne sur un laminé — et le FLAMBEMENT hors du plan de l'âme (« A-Lnn-Rb » : la section en
  croix, 15·ε·t_w d'âme de chaque côté dans la limite de l'âme disponible — le bout de la barre, la moitié de la
  distance au raidisseur voisin —, longueur 0,75·h_w, courbe c). Le raidisseur d'appui d'une âme élancée doit aussi
  avoir la rigidité d'un appui (I_st ≥ 0,75·h_w·t_w³). Il porte TOUTE la force ; il est de la nuance de la barre, chaque
  plat à la f_y de SON épaisseur (lot A : 20 mm en S235 → 225) ; la section en croix est calculée à min(f_y des plats ;
  f_y de l'âme), 15·ε·t_w avec l'ε de l'âme ; l'écrasement au contact, et 14·ε, gardent la f_y du plat.
- **Voilement induit par la semelle** (§ 8), avec k = 0,4 (du côté sûr) et f_y des semelles : aucun I du catalogue ne le
  dépasse, même en S460 ; une ligne d'hypothèse le dit, la vérification « A-S » ne paraît que s'il est dépassé. Semelles
  inégales (PRS) : la plus grande des deux aires (NBN EN 1993-1-5 ANB:2011, 8(2)).
- **Les PRS en I** (lot B ; section selon le contrat de `acier/barre/prs.py`) : section soudée, donc η = 1,20 partout, même
  quand les bords des semelles ne sont pas saisis ; h_w = l'âme saisie entre semelles ; b_f et t_f de la semelle CHARGÉE
  (m₁, m₂, l_y) ; force transmise (type b) : la semelle qui donne la plus petite F_Rd (choix prudent déclaré) ; M_f,Rd
  (§ 7.1) : la plus petite semelle sur le bras entre plans moyens ; raidisseur : chanfrein qui dégage le cordon (a·√2),
  largeur sur la plus étroite semelle. PRS-A : h_w/t_w = 36,0 ≤ 72·ε_w/η = 48,82.
- **Semelle chargée comprimée ou tendue** (lot B ; § 7.2(2)) : σ_x = N/A + M·z/I au plan moyen de la semelle chargée
  (compression positive). Comprimée : l'interaction (7.2). Tendue : pas d'interaction — le critère 6.2.1(5) de
  l'EN 1993-1-1 à la jonction âme-semelle (« A-Lnn-EL »), avec σ_x, σ_z,Ed = F_Ed/(s_e·t_w), s_e = s_s + 2·t_f
  (§ 3.2.3, z = 0) et τ = V·S_f/(I_y·t_w). σ_x nulle ou de signes mêlés (un appui d'extrémité) : les deux, le plus
  défavorable retenu. Le critère décide aussi le raidisseur : à un appui d'extrémité avec s_s = 0 SAISIE,
  σ_z,Ed = F/(2·t_f·t_w) — l'IPE 300 de la barre neuve (44,4 kN) demande un raidisseur (critère 1,57) ; avec
  s_s = 100 mm, non (6 %). σ_z,Ed garde s_s même sans contact d'extrémité établi (le brief ne met s_s = 0 que dans
  (6.10), « les autres paramètres inchangés » — À VÉRIFIER).
- **Seuil de résistance locale** (lot B) : à M_Ed imposé, min(F_Rd ; (1,4 − 0,8·η₁)·F_Rd) — R4 : 380,08 kN (η₁
  élastique), 387,76 kN (η₁ plastique) ; une phrase dit que ce n'est pas la capacité de la barre. η₁ plastique
  (classes 1 et 2, DÉCISION BUREAU n° 1) est affiché en variante ; le verdict reste élastique.
- **Contact d'extrémité saisi** (lot B ; DÉCISION BUREAU n° 2) : « appuis.contact » et « contact » d'une charge
  ponctuelle, « oui » | « non » | « inconnu » (défaut, gardé absent de la barre tant qu'il n'est pas saisi) ; « inconnu » :
  verdict de « non » et message « hypothèse de contact non établie » avec la valeur « oui » pour information.
- **Raidisseur et cisaillement** (lot B) : sous une charge, dans une âme élancée, le flambement du raidisseur porte
  N = F_Ed + N_st, N_st = V_Ed − h_w·t·f_y/(√3·λ̄_w²·γ_M1) ≥ 0 (§ 9.3.3(3)), V_Ed à 0,5·h_w du bord le plus sollicité de
  chacun des deux panneaux qu'il borde (les bords : les raidisseurs et les bouts de la barre ; le plus grand, sur
  toutes les combinaisons) ; I_st,min (9.5)/(9.6) avec le seuil √2. PRS 300 × 20 / 1 000 × 6 / 250 × 20 en S355, 10 m,
  q_Ed = 50 kN/m, F_Ed = 300 kN à mi-portée : V_Ed = 375 kN à 0,5 m de l'appui, N_st = 162,4 kN (au droit de la charge,
  150 kN, il serait nul).
- **La vue avant/après** (§ 15.4, calcul ; l'affichage est au lot C) : chaque lieu rend « sans » (σ_z,Ed, F_Rd, η₂,
  interaction et sa variante plastique, critère 6.2.1(5), seuil, état de la semelle, contact) et « avec » (A_st,
  σ = N/A_st, N_b,Rd, taux ; A_c, σ de contact, f_y du plat, taux), et le rappel « un raidisseur transversal ne réduit
  pas τ » (une fois par bloc, dans les hypothèses) ; le cisaillement qui ne passe pas le dit aussi (plaque de doublage ou
  profil plus lourd) : le voilement (« A-Vb ») et, pour un laminé, V_pl,Rd (« E-Vz », brief § 8.3, point 8) ; E-Vz qui passe
  mais a le plus fort taux de l'Eurocode 3 et de l'âme, quand une force concentrée appuie sur l'âme (un lieu au moins) :
  « le cisaillement gouverne », sans le remède (À VÉRIFIER) ; sans lieu, pas de question de raidisseur : rien.
- **Efforts imposés** : seuls le voilement par cisaillement et le § 8 sont vérifiés — les appuis et les charges ne sont
  pas décrits, et le bloc le dit.
- **Ce qui n'est PAS fait** (v1), et dit — jamais rendu « ok » : la paroi d'un profil creux sous une force concentrée
  (EN 1993-1-8 : un avertissement) ; les raidisseurs intermédiaires d'une âme élancée et le montant d'extrémité rigide
  (k_τ, χ_w = 1,37/(0,7 + λ̄_w), V_bf,Rd : sans objet tant que a → ∞) ; les exemples du commentaire JRC (EUR 22898) :
  données à fournir ;
  les soudures des raidisseurs (EN 1993-1-8) ; DEUX FORCES RAPPROCHÉES sur la même semelle — leurs longueurs chargées
  l_y se recouvrent —, que le § 6.3(2) demande de vérifier aussi ensemble : la vérification « A-GR » est alors rendue
  « non couvert » ; une réaction de soulèvement (ancrage : un avertissement).

**La preuve** — `tests/test_barre_ame.py`, 230 contrôles (lot B : PRS-A et PRS-C, tests 2, 13, 17, 18, 31 à 34 et 36 du
brief, seuil de R4 ; 31 mutations neuves) ; avant, 181 (lot A, deuxième passe) : le cas R4 du brief (IPE 300 : 283,2 /
387,8 / 425,1 kN en type (a) ; en type (c), 199,8 kN sans contact — 248,9 et 335,9 kN avec contact, pour information —,
avec leurs valeurs intermédiaires) ; un témoin écrit depuis le texte de la norme, égal au moteur sur les 275 I et H du
catalogue, deux nuances, douze charges ; l'IPE 750x134 S355 à la main (V_bw,Rd = 1 722,0 kN ; l'IPE 750x147 n'est plus
élancé avec η = 1,00) et l'interaction (7.1) ; un caisson soudé saisi (η = 1,20 : 72·ε/η = 60·ε, plateau χ_w = 1,20) ; les raidisseurs à la main (IPE 300, HEB 300) et la forme du plat sur
tout le catalogue (4 125 plats : chaque I, trois nuances, cinq épaisseurs) ; la décision lieu par lieu (R2 : la charge de 36 kN, η₂ = 0,093, « pas de
raidisseur ») ; 623 formules recalculées par leur gabarit et relues sur leurs nombres imprimés ; 82 mutations.

### 5.10 Profilés reconstitués soudés (PRS) en I — brief v12 § 18 (lot B, bloc B2)

**Le moteur** : `acier/barre/prs.py` — `section_prs(b1, t1, hw, tw, b2, t2, a, nuance, bords, nom)` rend une section au
format de `section.py` (forme « I » ; 1 = semelle haute), avec ce que les autres blocs lisent : `prs` (les plats),
`monosymetrique`, `z_g`, `z_s`, `int_z_y2z2`, `Wel_y_haut`, `Wel_y_bas`, `Wel_y` (le plus petit), `Wpl_y`, `Iw`, `It`,
`If_haut`, `If_bas`, `Avz` = 1,20·h_w·t_w, `Avy` = A − h_w·t_w, `eta`, `fabrication` (« soude_oxycoupe »,
« soude_larges_plats », None si les bords ne sont pas saisis) ; `zj_psi(section, semelle_comprimee)` par tronçon. Les
clés communes d'un I : `b` = la plus petite largeur (h/b du déversement), `tf` = la plus grande épaisseur (tableau 6.2,
f_y de calcul) — choix prudents déclarés.

**Les propriétés** (18.4), sur les trois rectangles, cordons non comptés : I_t = Σ (b·t³/3)·(1 − 0,63·t/b) sur les TROIS
plats ; I_w = h_s²·I₁·I₂/(I₁ + I₂) ; centre de cisaillement à h_s·I₂/(I₁ + I₂) sous le plan moyen de la semelle haute ;
W_pl,y autour de l'axe neutre plastique ; z_j = z_s − 0,5·∫z·(y² + z²)dA/I_y, positif vers la semelle comprimée
(annexe E ANB, 2(7)) ; (E.3), (E.4), (E.5) pour information.

**La saisie** (18.2) : refus bloquants — dimension non positive, t_w ≥ b, c ≤ 0 ; gorge non saisie : s = 0 et un
message ; bords non saisis : verdict de la méthode ELS bloqué, calcul pour information en larges plats (le cas
défavorable ; une seule règle, `historique.verifier` — intégration du lot B : le refus du moteur, qui ne montrait rien, est
retiré), l'Eurocode 3 se calcule. Le moteur prend un PRS par ses plats
(`"prs"`) ou par son nom (`"profil"`) dans une bibliothèque (`"bibliotheque"`) au FORMAT des profilés personnalisés du
Catalogue (`{"famille": "PRS", "dims": {"h", "b", "tw", "tf"} (+ "b2", "tf2", "a")}`, la nuance) : seules les
dimensions et la nuance sont stockées, tout se recalcule.

**L'écran (lot C, bloc C6 ; `api_web/profil_prs.py`)** — rubrique « Barre », sous l'acier : « Catalogue · PRS ». Un PRS
part des plats du profilé en I de la barre (IPE 300 : 150 × 10,7 / 278,6 × 7,1), gorge et bords NON saisis ; il se saisit
en trois lignes « b × t » (semelle haute ; âme et gorge a ; semelle basse : chemins `prs/b1` … `prs/a`), puis les bords
(`prs/bords` ; vides : la méthode ELS « Bloqué », puce jamais verte) ; une ligne d'aide dit h = t₁ + h_w + t₂ et la norme
produit (la table des aciers : EN 10025-2) ; un refus est celui de `section_prs`, en une ligne (« Saisie refusée : PRS :
… »). La case « section soudée équivalente » (`prs/soudee` → `parametres.prs_soudee_equivalente`) n'est offerte qu'à un
PRS bisymétrique, et retirée quand il devient monosymétrique. Retour au catalogue : le profilé en I le plus proche (h,
puis b). **La bibliothèque** est le CARNET du module (`profil.CARNET`, il voyage avec chaque requête ; l'écran le garde
dans ce navigateur, `js/carnet.js`), au format de `prs.vers_bibliotheque` (aucune propriété), relu par
`prs.de_bibliotheque` ; « Enregistrer dans la bibliothèque » / « Retirer » sous le bouton du profilé, qui ouvre la
bibliothèque (le sélecteur d'un PRS : chaque PRS vérifié avec la barre, sans ★). La barre garde une COPIE de ses plats
(`prs`) : la note se suffit. À VÉRIFIER : la page Catalogue garde ses profilés personnalisés dans SA note et ne lit pas
ce carnet en v1. **La vue « Section »** est le croquis coté (`dessin_profil.section_prs`) : plats, cordons, axes PAR LE
CENTRE DE GRAVITÉ, G (cercle barré) et C (losange), une lettre et un symbole chacun ; « mm » et la méthode dessous ; le
tableau des propriétés avec leur origine (`profil_prs.proprietes` : A, z_g, I_y, I_z, W_el,y, W_pl,y, I_t, I_w ; z_s, z_j
et ψ_f d'un monosymétrique, semelle haute comprimée ; f_y de chaque plat et de calcul) — le même tableau que la fiche de
la note (`profil.fiche_profil(nom, barre)`, clé « tableau » : la note du PRS est au bloc C5). La masse au bouton : ρ·A
(`prs.MASSE_VOLUMIQUE`). Preuve : `tests/test_api_web_profil_prs.py` (74 contrôles, 22 mutations).

**Les règles EC3** (18.5) : c depuis le pied des cordons (s = a·√2) ; âme d'un monosymétrique fléchi en paroi fléchie et
comprimée (α plastique, ψ élastique, bord de c à f_y) même sans N ; semelle tendue sans limite ; ε et f_y plat par plat,
f_y de calcul = la plus petite ; classe 4 (b̄ = b_w), avec e_N et ΔM_y = N_Ed·|e_N| (du côté sûr) ; η = 1,20 partout ;
N + M : (6.36) à semelles égales, sommation linéaire (6.2) sinon ; M + V : (6.30) à semelles égales, (6.29) sinon (âme à
(1 − ρ)·f_y, axe neutre plastique recalculé) ; flambement : I soudé (t_f ≤ 40 : b/c ; > 40 : c/d) ; torsion (F.3) pour un
bisymétrique, flexion-torsion « non couvert » pour un monosymétrique comprimé ; déversement : M_cr du solveur AVEC le
terme de Wagner ∫M_y·z_j·φ′² (`mcr.py`), charges et maintiens repérés depuis le centre de cisaillement ; § 6.3.2.2 par
défaut (courbes c/d, b = la plus petite largeur) ; § 6.3.2.3 seulement pour un bisymétrique déclaré « section soudée
équivalente » (paramètre `prs_soudee_equivalente`, DÉCISION BUREAU n° 10, défaut non coché ; l'autre méthode affichée
pour information, écrite une seule fois pour les laminés et les I soudés : `grandeurs.deversement.information`, avec f
et le couple qu'elle emploie — la variante (0,4 ; 0,75), quand elle est admise, est dite : « § 6.3.2.3 (variante 0,4 ;
0,75) pour information » ; un PRS monosymétrique n'en a pas). La même règle vaut pour un I saisi « soudé » qui n'est pas
un PRS (intégration du lot B) : les tableaux 6.4 et 6.5 lui donnent les mêmes courbes c et d et le § 6.3.2.3 divise par
f ≤ 1 — il restait au § 6.3.2.3, M_b,Rd trop haut de 2 à 7,5 % (défaut non conservatif, latent : aucun écran ne saisit
encore un tel I) ; contrôle (E.1) de l'annexe E ANB (`mcr.annexe_e`, tableau E.1 ANB : C₁ et la colonne C₃ de k_z = 1,
selon le signe de ψ_f — 1 de ψ = 1 à 0 ; pour ψ_f > 0 : 0,850 ; 1,3 − 1,2·ψ_f ; 0,55 − ψ_f (1 sinon) ; −ψ_f à ψ = −1
dans les deux colonnes ; tableau E.2 ANB ;
z_g positif quand la charge va vers le centre de cisaillement, annexe E ANB 2(8) ; −0,9 ≤ ψ_f ≤ 0,9, sinon bloqué).
Sous moments d'extrémité, (E.1) reste sous le solveur (PRS-B, ψ = 0 : 2 093,2 contre 2 274,32 kN·m, Ritz 2 274,31) ;
6.3.3 : bisymétrique comme un laminé, monosymétrique « non couvert ».

**La bibliothèque, deux lecteurs** : le Catalogue (`api_web/catalogue._def_perso`) relit un PRS bisymétrique du module
Barres sous le même nom ; il REFUSE un PRS monosymétrique (il ne calcule que des semelles égales) et une dimension
inconnue — jamais une relecture bisymétrique d'un monosymétrique. Un fichier qui porte un PRS monosymétrique n'est pas
perdu : cet enregistrement seul est écarté (gardé tel quel, `persos_ecartes`), une ligne le dit, les autres sont lus.

**Une source pour les propriétés d'un PRS** (R2) : le Catalogue prend I_t (18.4), I_w, A_v,z = 1,20·h_w·t_w et
A_v,y = A − h_w·t_w de `prs.proprietes_catalogue` (PRS 600x300x15x30 : I_t = 565,67 cm⁴ au lieu de Σ b·t³/3 = 600,8, non
conservatif) ; ses réserves sur I_t et A_v,z ne s'affichent plus. C'est la seule divergence avec l'écran Streamlit,
NOMMÉE (`api_web/catalogue.PRS_BARRES`) : la parité (`tests/test_catalogue_moteur.py`) se prouve divergence retirée, A, I
et W restent les deux calculs liés par un test.

**La preuve** — `tests/test_barre_prs.py` : PRS-A et PRS-B du brief (§ 11.8) à sa tolérance (M_cr à 0,01 kN·m :
1 241,73 / 715,69 kN·m sous moment constant, 942,71 sans Wagner ; M_b,Rd 797,39 et 617,55 kN·m) ; les « autres tests
PRS » 1 à 5 ; la section efficace recalculée autrement (un I saisi ; un assemblage de parties efficaces) ; le moteur
(plats, bibliothèque, barre chargée, laminés inchangés) ; le sens des charges (efforts imposés, M < 0, les deux sens aux
trois niveaux, charge répartie et ponctuelle : 806,49 / 912,30 / 628,60 et 965,87 / 1 131,52 / 705,91 kN·m ; soulèvement
en barre chargée : 1 033,76, ponctuel 1 328,40 ; console : 4 000,51 — recoupés par un solveur de Ritz indépendant) ;
PRS-C (§ 11.9, 28 m, § 6.3.2.2 courbe d, 7 / 8 / 9 maintiens de la semelle haute : taux 1,007 / 0,967 / 0,937 ; M_cr à 8
maintiens 28 294,6 kN·m, Ritz 28 279,2, +0,05 %), par les fonctions du moteur (poids de l'acier au centre de gravité) et
de bout en bout par la barre chargée (un seul niveau de charge : l'acier aussi sur la semelle haute, M_cr −0,1 %) ;
93 mutations, dont le terme de Wagner retiré ou de signe inversé et les charges ramenées au centre de cisaillement (PRS-C :
les mêmes charges au centre de cisaillement donnent un M_cr plus grand d'au moins 0,5 %, contrôle direct). Reste (brief
§ 13.4) : la qualification externe d'un M_cr monosymétrique par LTBeam (cas L12 à L16,
`docs/validation/barres_cas_ltbeam_scia.md`).

**Lot C, deux manques du modèle de la barre** (vus sur PRS-C) : (1) une hauteur de charge PAR CHARGE (au moins le
centre de gravité pour le poids propre) — la barre chargée n'a qu'un `niveau` pour toutes ses charges ; (2) une nature
d'action « exécution » (γ_Q = 1,5, ψ à fixer) — `combinaisons.NATURES` n'en a pas, le béton frais de PRS-C passe en
« exploitation_A » pour avoir 1,5.

## 6. Les valeurs de référence

### 6.1 Les quatre classeurs récents (u = 1 ; unités des classeurs : N, cm, cm⁴, cm³, mm²)

| Cas | Données | Résultats (cellules) |
|---|---|---|
| HEB 200 | P = 432 720 ; f_y 235 ; s 1,5 ; L 720 ; i 8,54 / 5,07 ; I 5696 / 2003 ; W 569,6 / 200,3 ; A 7810 ; âme 2483 ; ailes 6000 ; courbes b (fort) / c (faible) ; M = 0 | λ̄_z 1,512164186 ; φ_z 0,316435900 ; φ_y 0,668818546 ; P_adm,z 387 180,42 ; P_cr,z 800 820,16 ; PE_z 0,189480942 ; σ 175,0935651 → taux 1,118 |
| HEA 120 | P = 112 000 ; L 100 ; i 4,89 / 3,02 ; I 606 / 231 ; W 106 / 38,5 ; A 2534 ; âme 846 ; ailes 960 ; b / c | φ_z 0,918065630 ; φ_y 0,982155600 ; P_adm,z 364 465,93 ; PE_z 0,964910413 ; σ 48,14350259 |
| CHS 168,3 × 6 | P = 35 000 ; L 700 ; i 5,74 ; I 1009 ; W = 1009/8,415 ; A 2520 ; c / c ; M faible = 0,05·P + poids (78 600·A/10⁶ N/m)·L²/8 = 1750 + 1213,191 N·m | φ 0,393034043 ; PE 0,876988652 ; σ_P 35,33762314 ; σ_M 28,17919784 ; σ 63,51682099 |
| Tube Ø610 × 12,5 | P = 5 000 000 ; f_y 450 ; L 1000 ; i 21,13 ; I 104 754 ; W 3506 ; A 23 464 ; a / a | φ 0,855232758 ; P_adm 6 020 154,43 ; PE 0,654560809 ; σ 249,1630435 |

### 6.2 Les deux anciens classeurs

| Cas | Données | Résultats |
|---|---|---|
| FLAMBECBD (HEA 160) | P = 300 000 ; f_y 235 ; L 175 ; i 3,98 sur les deux axes ; A 3877 ; c (faible) / b (fort) ; I 615,6 ; W 76,95 ; M 3000 N·m | φ 0,860491946 / 0,913104363 ; P_adm 522 659,94 ; P_cr 4 166 213,81 ; PE 0,891988261 ; σ du classeur (φ fort) 128,4504854 ; avec φ_min : 133,63 |
| FLAMBEBAD (IPE 270) | P = 185 000 ; f_y 355 ; L 340 ; i 3,02 / 11,23 ; A 4594 ; b (faible) / a (fort) ; I 5790 ; W 428 ; poids 361 N/m | φ 0,351913858 / 0,960096134 ; P_adm,z 382 617,17 ; M 521,645 N·m ; σ du classeur 43,19590210 ; avec φ_min : 115,68 |

### 6.3 Les exemples de la doc (recalculés sans arrondi ; entre parenthèses, ce que la doc imprime)

- p. 6, caisson 100 × 100 × 1 cm, AE235, 20 m : λ̄ 0,522 (0,54), φ_b 0,891 (0,88), P_adm 5582 kN (5500).
- p. 8, cornière L 60 × 60 × 8, 2 m : λ 170,9 (171), λ̄ 1,820 (1,82), φ_c 0,236 (0,23), P_adm 33,4 kN (32,6).
- p. 14, HEA 220, 5 m, P = 500 kN, M = 25 kN·m : P_cr 4485 kN (4496), φ_b 0,863 (0,86), σ = 90,1 + 58,3 = 148,4 (149) ;
  148,3 avec le HEA 220 du catalogue (la valeur du brief v12).
- p. 23, poutre HEA 200, 8 m, P = 38 kN, M = 48 kN·m, avec la dispense : φ_c 0,2614 (0,264), σ = 27,0 + 129,6 = 156,6
  (26 + 130 = 156) ; 27,0 + 129,7 = 156,7 avec le HEA 200 du catalogue (le brief v12) ; sans la dispense (variante du
  fascicule) : φ_ed 0,307 → 422 N/mm².
- L'écart de 0,1 N/mm² vient des DONNÉES, pas de la formule : la doc imprime A et W arrondis (HEA 200 : 53,8 cm² et
  389 cm³ ; le catalogue : 53,83 et 388,67), le brief a recalculé avec le catalogue. Les deux lectures sont figées.
- p. 24-25, colonne HEA 300, 6 m, u = 1,5, P = 168 kN, M = 110 kN·m, C = 0,85, variante du fascicule : φ_c 0,4009 (0,4),
  r_d 8,30 cm, φ_ed 0,676 (0,7), P_cr 4673 kN, PE 0,9461 (0,95), σ = 37,3 + 116,0 = 153,3 (37,3 + 111,5 = 148,8) ;
  37,3 + 116,1 = 153,3 avec le HEA 300 du catalogue (le brief v12).

### 6.3 bis Le déversement des classeurs, les lectures cohérentes et les PRS (lot B, brief v12 11.1, 11.1 bis, 11.8, 11.9)

`tests/test_barre_historique.py` § 4 bis les retrouve tous, à 0,01 N/mm² et 0,0001 près : les sept cas du tableau 11.1 bis
(HE 200 B 115,70 ; feuille S355 147,50 ; U 65,32 ; φ_d = 1 ; t_f 50 et 80 mm ; IPE 600 avec C_d = 1 et 1,12) et la table de
φ_d ; FLAMBECBD 133,63 / 107,52 / 105,96 (et le mélange bloqué) ; FLAMBEBAD 117,42 / 117,20 ; CHS 59,75 (C = 0,85 : 55,15) ;
REALEX 0,831 / 0,984 / 1,018 / 0,865 ; PRS-A (classeurs 90,31 et 78,79 ; fascicule 0,3005 et 0,2847) ; PRS-C (σ = 117,08 ;
8 maintiens 89,68, ne passe pas ; aux tiers 42,91 ; 18 maintiens 118,3).

### 6.4 L'Eurocode 3 (du brief, recalculé)

- χ pour λ̄ = 0,5 / 1,0 / 1,5 / 2,0 — a₀ : 0,9513 / 0,7253 / 0,3953 / 0,2323 ; a : 0,9243 / 0,6656 / 0,3724 / 0,2229 ;
  b : 0,8842 / 0,5970 / 0,3422 / 0,2095 ; c : 0,8430 / 0,5399 / 0,3145 / 0,1962 ; d : 0,7793 / 0,4671 / 0,2766 / 0,1766.
- R1, IPE 300 (I_z 603,8 cm⁴, I_t 20,12 cm⁴, I_w 125,9·10³ cm⁶), 6 m, fourches : moment constant 90,47 kN·m (exact) ;
  charge répartie au centre de cisaillement 102,31, semelle haute 78,83, semelle basse 132,69 ; charge centrale 123,12,
  89,19, 168,95 (z_g = ±150 mm).
- R2 : G = 2,42 kN/m, P_G = 10 kN, P_Q = 15 kN → q_Ed 3,267 kN/m, P_Ed 36,00 kN, M_Ed 68,70 kN·m, V_Ed 27,80 kN.
- R3 (W_pl,y 628,4 cm³, S235, f = 1) : au centre M_cr 118,18 kN·m, λ̄_LT 1,1179 → (0,2 ; 1,0) χ_LT 0,5246, taux 0,887 ;
  semelle haute M_cr 86,98, λ̄_LT 1,3030 → 0,4254, 1,094 (le couple (0,4 ; 0,75), retiré au lot A, donnait 0,6278, 0,741
  et 0,5220, 0,891).
- R4 (h_w 278,6 ; m₁ 21,13) : (a) s_s 50 → F_Rd 283,2 kN ; 100 → 387,8 ; 150 → 425,1 ; (c) s_s 100, c = 0 → 199,8
  (contact non établi, lot A ; 248,9 avec contact — 261,9 avant le lot A, qui gardait la plus grande des deux branches de
  m₂) ; c = 50 → 199,8 (335,9 avec contact) ; A_v 2568 mm², V_pl,Rd 348,4 kN.

## 7. L'inventaire du dépôt (14/11/2026)

| Besoin | Ce qui existe | Ce qui manque |
|---|---|---|
| Profilés | `acier/donnees/profiles.json` : 1 419 profilés (IPE, IPE A/O, IPN, HEA/B/M/AA, HL, HD, HP, UPE, UPN, L, CHS, SHS, RHS, finis à chaud et formés à froid) ; tous les champs : h, b, t_w, t_f, r, A, I_y, I_z, W_el, W_pl, i, I_t, I_w, A_v,z, A_v,y, masse, c/t | rien pour la v1 (I_t et I_w sont calculés sur le contour exact : IPE 300 19,77 cm⁴ et 126 333 cm⁶, contre 20,12 et 125 900 dans les catalogues) |
| Flambement EC3 | `acier/flambement` : compression seule, classe 4, tableau 6.2, sélecteur ; module « Flambement » du site | N + M, déversement, section |
| Choix de profilé | `acier/choix_profil` : la copie de l'ancien écran (efforts saisis, résistances élastiques, flambement à courbe choisie) ; sept défauts non conservatifs écrits (`CHOIX_PROFIL_MOTEUR.md` § 7) dont « déversement non calculé » | tout le reste : c'est le module à refaire |
| Statique | `rdm/poutres.py` : formules fermées, deux appuis, console, encastrements, poutre continue de 2 à 5 travées ; charges réparties, triangulaires, ponctuelles, couples ; M, V, flèche ; contrôles internes | cas de charge et combinaisons ; effort normal ; 8 charges au plus |
| Combinaisons EN 1990 | seulement G / G + Q dans la poutre en bois | un moteur (6.10), catégories et ψ |
| Note PDF | le gabarit `ndc_pdf` (planche : schémas à gauche, deux colonnes de calcul, page de suite) ; le kit des formules « grandeur = formule = valeurs = résultat » des notes acier | la note des barres ; « Choix de profilé » n'a pas de note aujourd'hui |
| Écran | l'établi compact (rubriques à gauche, dessin au centre, résultats à droite ; PC, tablette, téléphone) | l'écran des barres |

## 8. Les textes des normes

Le brief demande que les textes priment. Ils ne sont pas dans le dépôt et ne doivent pas y entrer (dépôt public,
normes sous droits). Deux pages de documentation d'éditeur de logiciel ont servi de recoupement pour l'ANB ; l'une
n'est pas lisible d'ici. Pour lever les « À CONFIRMER » : envoyer en pièce jointe les pages de l'ANB de l'EN 1993-1-1
(clauses 6.3.2.2, 6.3.2.3, 6.3.3, annexe A) et de l'ANB de l'EN 1990 (tableau des ψ) — elles seront lues, pas déposées.

## 9. L'architecture et les lots

Un paquet neuf, `acier/barre/`, bibliothèque standard seule (il tourne dans le navigateur), unités internes N et mm :

| Fichier | Rôle |
|---|---|
| `parametres.py` | LA table des paramètres nationaux (valeur, source, statut) |
| `section.py` | un profilé du catalogue — ou des propriétés saisies — vu par les deux méthodes |
| `historique.py` | la méthode du § 4 |
| `resultat.py` | la forme commune de ce que rend une méthode (vérifications, blocs de formules, alertes, synthèse) ; `fixe` : L'ARRONDI D'AFFICHAGE du module, le seul |
| `ec3_classe.py`, `ec3_section.py`, `ec3_flambement.py`, `mcr.py`, `ec3_deversement.py`, `ec3_interaction.py` | EN 1993-1-1 |
| `ec3.py` | la barre vérifiée selon l'Eurocode 3 : la lecture, puis la section, le flambement, le déversement, l'annexe A |
| `ec3_ame.py`, `raidisseurs.py` | EN 1993-1-5 : les formules de l'âme (cisaillement, force concentrée, interactions, semelle) ; la paire de plats |
| `ame.py` | le bloc de l'âme d'une barre : où une force appuie, sous quelle combinaison, faut-il un raidisseur (§ 5.9) |
| `enveloppe.py` | l'occurrence la plus défavorable de chaque vérification, sur toutes les combinaisons et les sections |
| `combinaisons.py`, `statique.py` | les cas de charge, les combinaisons, les efforts le long de la barre (sur `rdm/poutres.py`) |
| `entree.py`, `moteur.py` | la barre, sa relecture ; les deux méthodes, la synthèse |
| `formules.py` | les gabarits des formules de la note (un facteur d'unités s'y écrit `1eN`) |
| `ecriture.py` | ce qui s'imprime : « symbole = formule = valeurs = résultat », la même ligne à l'écran et dans la note ; `relire` refait le calcul sur les nombres imprimés |

Chaque vérification rend `{id, methode, titre, reference, formule, termes, valeur, limite, taux, statut, unite,
hypotheses, messages}` ; le moteur calcule et nomme, l'écran et la note composent et formatent.

**La barre chargée** (`statique.py`, `entree.py`, `moteur.py`, lot 5). Deux modes : « efforts » (les classeurs : les
efforts de service vont à la méthode ELS, ceux de calcul à l'Eurocode) et « chargee » — appuis (deux appuis,
console, encastrée–appuyée, bi-encastrée, continue de 2 à 5 travées), cas de charge (répartie, triangulaire,
ponctuelle, couple ; un effort normal par cas), poids propre, maintiens (complet, semelle haute, semelle basse). La
statique est celle du formulaire du dépôt (`rdm/poutres.py`), cas par cas, puis superposée.
- L'enveloppe : pour chaque vérification, l'occurrence la plus défavorable — sa combinaison, son abscisse ou son
  tronçon, et les formules de CETTE occurrence.
- Les sections à vérifier : les appuis, chaque rupture de chargement et les extremums du moment — là où l'effort
  tranchant de la combinaison s'annule (entre deux ruptures, V est un polynôme de degré 2 au plus : ses racines sont
  exactes, sans balayage). Les lectures des cas sont gardées : toutes les combinaisons relisent les mêmes sections.
- Eurocode 3 : la SECTION dans chaque section à vérifier, avec les
  efforts qui s'y trouvent ENSEMBLE ; la BARRE par tronçon de déversement, sous le diagramme réel (M_cr numérique,
  charges à leur niveau). k_c n'est lu dans le tableau 6.6 que si le tronçon EST un de ses cas ; C_my,0 par la formule
  générale (flèche et moment de la portée) sur deux appuis, sinon moment constant.
- Méthode ELS : dans chaque section ET aux deux bouts de chaque tronçon (un maintien qui n'est ni un appui ni un point
  de charge), avec la longueur de déversement de son tronçon ; à une frontière, le côté gauche avec le tronçon de gauche,
  le côté droit avec celui de droite (le moment est le même, pas L_d) : la poutre retournée donne le même taux. Avant le
  lot B, la section frontière prenait le L_d du tronçon de gauche pour ses deux côtés, et le bout d'un tronçon sans
  charge n'était pas vérifié (non conservatif : 2 + 6 m → 173 %, 6 + 2 m → 210 % ; maintiens à 4,5 et 5,5 m d'une poutre
  de 10 m → 119 % au lieu de 198 %). Un maintien d'une seule semelle ne compte que si elle est comprimée sur tout le
  tronçon.
- Les deux méthodes : un appui à une somme de portées inexacte en binaire (3,37 + 2,18 = 5,550000000000001) est reconnu
  à 10⁻⁹·max(1, L) près (rdm/poutres.py, lot B) ; avant, l'abscisse lue à 5,55 perdait le côté droit de l'appui et le
  moment maximal de la travée suivante (IPE 300, 3,37 + 2,18 + 3,45 m : E-My 27,5 % au lieu de 28,6 %, E-Vz 22,7 % au
  lieu de 23,3 %, H-DEV 60,4 % au lieu de 63,7 %, selon le sens de dessin).
- Ce que ces deux lectures justes déplacent (balayage du lot B ; à attendre, ce n'est pas un réglage) : Eurocode 3,
  E-MV jusqu'à +22 %, E-Vz +10,5 %, E-Mb +6,7 % ; méthode ELS, H-EQ jusqu'à +13,1 %, et des baisses — jusqu'à −2,1 %,
  puis jusqu'à −15 % quand l'effort tranchant du côté droit d'une frontière n'est plus lu avec le L_d du tronçon de
  gauche. Une abscisse TAPÉE au bout d'une barre à portées sommées (2,8 + 3,4 m : une fin de charge ou un maintien à
  6,2) est ramenée au bout dans la même tolérance (`rdm/poutres.ramener`, intégration du lot B) ; au-delà (6,2 + 10⁻⁶),
  elle reste refusée.
- Hypothèses dites dans la note : les appuis tiennent le déplacement latéral et la torsion (fourches) ; une console est
  encastrée à sa racine.
- Longueurs de flambement d'un champ vide (`moteur._longueurs`) : axe fort, la plus grande travée ; axe faible, le plus
  long tronçon entre sections tenues ; la torsion prend celle de l'axe faible. Une CONSOLE : 2,1·L autour des DEUX axes
  tant que son bout est libre — il se déplace latéralement comme dans le plan (brief § 13.3, « encastrée – libre », u
  recommandé 2,1 ; `moteur.U_CONSOLE`). Lot B : l'axe faible prenait L, défaut non conservatif corrigé (console de 2 m :
  2 → 4 m, puis 4,2 m à l'intégration : 2 → 2,1). La longueur proposée EST la longueur de flambement ; u reste le
  multiplicateur saisi (1 par défaut). Un maintien complet au bout en fait une section tenue : la plus grande distance
  entre sections tenues. Des maintiens complets intermédiaires seuls laissent 2,1·L (la longueur exacte est plus courte —
  3,77 m pour une console de 3 m tenue à mi-longueur, au lieu de 6,30 — mais la partie libre ne se prend pas isolément
  sans justification) : hypothèse prudente, avis « À VÉRIFIER » d'une ligne, dit une fois (sous la première méthode
  rendue) et seulement si la barre est comprimée — sans N > 0, L_cr,z ne change aucun résultat. Une longueur saisie est
  prise telle quelle.
- Méthode ELS d'une console à bout libre (intégration du lot B) : le bout libre n'est plus compté comme une section tenue
  (L_d valait la longueur de la console) ; L_d = 2,1·L, la règle du flambement, maintiens intermédiaires non comptés, et
  le déversement est « hors domaine » (la formule des classeurs suppose deux sections tenues ; brief § 6.2, point 8) :
  calcul pour information, jamais « vérifié », À VÉRIFIER. IPE 300 de 3 m, 20 kN au bout : H-DEV 113 % → 185 % ;
  IPE 240 de 2 m tenu à 1 m, N = 50 kN : 38 % (« ok ») → 67 % (« hors domaine »). Un maintien complet au bout : L_d =
  la longueur de la console, dans le domaine.
- Une méthode qui REFUSE la barre (une paroi de plus de 80 mm : au-delà du domaine du module ; une section sans courbe)
  est rendue « non couverte » avec son motif ; l'autre reste calculée. Depuis le lot A, la méthode ELS refuse elle aussi
  au-delà de 80 mm (13 HD et HL du catalogue).
- Hors de la barre chargée (v1) : la flexion d'axe faible et les charges horizontales (le mode « efforts » les
  prend) ; la flèche de service est donnée à titre d'information, sans verdict.
- R2 du brief, de bout en bout depuis les charges : M_Ed 68,70 kN·m, V_Ed 27,80 kN, déversement 89,5 % (Eurocode 3) —
  et **184 % par la méthode ELS** : le déversement de la doc, sur 6 m sans maintien, est très sévère (Q4). Lot A :
  le déversement passe à 109,9 % (Eurocode 3), avec le couple (0,2 ; 1,0) de l'ANB:2018 (89,5 % avec l'ancien couple).

**Intégration** : dans « Choix de profilé » (demande du propriétaire). Le module garde ce qu'il sait faire — comparer
des profilés et retenir le plus léger qui tient — et gagne la barre : ses appuis, ses charges, ses maintiens, ses
raidisseurs ; ses deux méthodes ; sa note. La copie de l'ancien écran (`acier/choix_profil`) est remplacée quand le
nouvel écran la couvre ; le module « Flambement » reste en l'état (`AMELIORATIONS_PAR_MODULE.md`, A1).

**L'écran et la note** (lot 6, 14/11/2026). Le module « Choix de profilé » est l'écran des barres ; son ancien écran
(un profilé, une flexion, `acier/choix_profil`) n'est plus chargé par le site.

| Fichier | Rôle |
|---|---|
| `api_web/profil.py` | la note (version 2 : `{"zero_saisi": true, "elements": [barre…], "actif"}` — sans changer de version : le site d'avant le lot B relit une note neuve telle quelle, ou la refuse si une donnée y est vide ; mais il suffit qu'il l'OUVRE pour la réécrire sans la clé, ses 0 saisis redevenant « non saisis »), ses saisies, le sélecteur de profilés, les vues, la relecture des notes de la version 1 et du site d'avant le lot B (sans « zero_saisi », un 0 de s_s ou de c est « non saisi ») ; une autre version — ou une version qui n'est pas un entier — est refusée |
| `api_web/profil_saisie.py` | la colonne de gauche : quatre rubriques — Barre, Charges (ou Efforts), Maintiens et flambement, Méthodes |
| `api_web/profil_resultat.py` | la colonne de droite et les fenêtres de calcul : un verdict par méthode, un bloc par méthode, la comparaison, les paramètres à confirmer |
| `api_web/profil_mots.py` | les mots et les nombres de l'écran — et de la note |
| `api_web/dessin_barre.py` | la barre : ses cas de charge, ses appuis, ses maintiens, les enveloppes de M et de V ; les taux de toutes les vérifications |
| `api_web/dessin_profil.py` | la section à l'échelle (I, H, tubes ronds, carrés, rectangulaires) |
| `api_web/note_profil.py` | la note PDF |

- **Une barre garde ses deux modes** : passer de « barre chargée » à « efforts imposés » ne perd aucune saisie. La barre
  neuve est une IPE 300 de 6 m, tenue à mi-portée : elle tient par les deux méthodes (ELS 74 %, Eurocode 58 % — 52 %
  avant le lot A) — sans ce maintien, la méthode ELS la refuse (178,5 % ; l'écran, qui arrondit vers le défavorable,
  écrit 179 % — Q4), et l'Eurocode 3 aussi depuis le lot A (113,7 %).
- **Un verdict par méthode**, jamais un seul chiffre : la méthode ELS et l'Eurocode 3 ne comparent pas la même
  chose (des contraintes de service à σ_adm ; des efforts pondérés à des résistances). Le bloc « Comparaison » met leurs
  taux côte à côte, sujet par sujet.
- **Chaque vérification ouvre SA fenêtre de calcul** : ses formules, écrites par `ecriture.py`, la combinaison et
  l'abscisse où elle a été trouvée. « Tout le calcul » d'une méthode y ajoute le bilan de ses combinaisons.
- **Un seul arrondi d'affichage** : le demi vers l'extérieur, sur la valeur exacte — la règle de l'écran
  (`api_web/commun.fr`) et des moteurs (`acier/js.to_fixed`), appelée par `resultat.fixe`. Une formule, une phrase, un
  tableau et la fiche du catalogue écrivent donc le même chiffre d'un même nombre. Trouvé sur une photo du lot 4 :
  un raidisseur qui porte 1950 × 235 / 10³ = 458,25 kN s'écrivait « 458,3 kN » au tableau et « 458,2 kN » au bout de sa
  formule — l'arrondi de Python va au chiffre pair sur un demi exact. Le même défaut touchait 29 valeurs du catalogue
  (HEB 180 : I_y = 3 831,25 cm⁴, écrit « 3 831,2 » dans la barre et « 3 831,3 » dans le catalogue). Aucun calcul ne
  change : seul le dernier chiffre ÉCRIT d'un demi exact. `tests/test_barre_ecriture.py` et `tests/test_api_web_profil.py`
  refusent qu'un format de Python (« {x:.1f} ») revienne écrire un nombre du module.
- **Le sélecteur** : les profilés de la famille de la barre (ou de celles qu'on coche), triés par masse, chacun vérifié
  par le moteur avec les méthodes de la barre ; ✓ le retenu, ★ le plus léger qui tient.
- **Une note de la version 1** (l'ancien écran) est relue en « efforts imposés », Eurocode seul : ses efforts de calcul,
  sa longueur de flambement, son profilé.
- **La note PDF** a la planche des notes Poutre et Dalle (`ndc_pdf/styles.py` : le panneau de gauche et son dessin, deux
  colonnes de calcul, des vérifications numérotées, une bande de verdict ; une planche qui déborde se poursuit sur une
  page suivante). Par barre : la barre et ses charges ; la méthode ELS D'ABORD — ses hypothèses, puis chaque
  vérification comme les classeurs l'écrivent — ; l'Eurocode 3 ; la comparaison ; les paramètres à confirmer ; ce que la
  note ne vérifie pas. **Elle imprime les méthodes cochées à l'écran** (rubrique « Méthodes » : ELS, Eurocode 3,
  les deux) : ce que l'écran montre est ce qui s'imprime (Q9). Le dessin du panneau est celui de l'écran, peint tel
  quel ; chaque formule est la ligne de la fenêtre de calcul. Deux allégements, pour la lecture : la ligne du taux n'est
  pas répétée (le verdict dit les deux membres et le taux) ; (6.62) ne réécrit pas les facteurs que (6.61) vient
  d'écrire. Sous le nom de la barre, la garde et l'en-tête écrivent le taux AVANT la méthode, comme le verdict de
  l'écran (« S235 · 6,00 m · 74 % ELS · 58 % Eurocode 3 » : « Eurocode 3 9 % » se lisait « 39 % ») ; l'en-tête
  d'une planche garde ce résumé entier s'il tient, sinon ses seuls taux, sinon rien — et un nom qui passerait sous la
  pastille d'état cède par « … » (un nom de 60 caractères, sur la garde).
- **Ce que le kit des notes a reçu** (`ndc_pdf`, sans effet sur les notes qui ne s'en servent pas — les notes de la
  poutre et des fondations de grues sortent identiques, page à page) : `\bar{λ}` (mathx) ; trois genres d'éléments d'une
  planche — une substitution repliée sur ses « = » et, si une ligne dépasse encore la colonne, coupée devant un
  opérateur (jamais réduite tant qu'elle peut se couper) ; un paragraphe en texte riche ; un petit tableau — ; un
  verdict en texte riche, avec son taux calé à droite ; la hauteur à garder sous la figure du panneau (`reserve`) ; le
  peintre des dessins en mode riche (indices des symboles, aires translucides).
- **Trouvé en faisant le lot, et corrigé** : le facteur de déversement écrit « χ_LT » alors que sa valeur était celle
  de χ_LT,mod (le contrôle « une grandeur écrite deux fois garde sa valeur », 199 000 termes) ; les enveloppes de M et
  de V portent LEURS abscisses, doublées là où le diagramme saute (le tracé prenait le zéro du dehors aux deux bouts) ;
  « N_Ed/N_cr,y = N_Ed/N_cr,y = … » écrit deux fois ; la section d'un tube rond, que l'écran ne savait pas dessiner
  (il n'a ni h ni b) — trouvé par la suite de la note ; le nom d'une barre borné à 40 caractères, quand un élément du
  site en admet 60 partout (dalle, cornière, fondation de grue) — trouvé par le harnais du navigateur, qui en saisit 58
  sur la tablette en portrait : la borne est 60 (`profil.NOM_MAX`), 12 pour le nom d'un cas de charge, qui s'écrit sur
  le dessin (`NOM_CAS_MAX`).
- **Trouvé par le harnais des barres** (filtre « barres », ci-dessous), et corrigé : le titre d'un bloc de méthode
  ROGNÉ dès que son état est long (« MÉTHODE HIS… » devant « Conforme, attention · 97 % » ou « Non conforme ·
  179 % ») — le bloc porte le titre court, celui du bouton de la rubrique Méthodes (« Historique », « ELS » depuis le
  16/11/2026), et la feuille du
  module fait passer l'état dessous plutôt que de rogner un titre ; sur une poutre continue, la cote du minimum de V
  (au milieu de la barre) écrite SUR la légende des maintiens — vu à l'œil sur une capture, mesuré depuis par le
  harnais (aucun texte du dessin n'en recouvre un autre) et par `tests/test_api_web_profil.py` ; une portée hors des
  bornes de son champ refusée par la page sans dire la borne (« Valeur refusée : « 80 » ») — le refus dit sa raison,
  pour tous les modules (`js/commun.js`, `valeurSaisie` et `motRefus` : « … — entre 0,05 et 60 »).
- **Trouvé par la batterie complète**, et corrigé : (1) la ligne « Profilé » disait « plus léger : IPE 300 » seulement
  quand le sélecteur avait déjà vérifié la famille — le résultat d'une note dépendait de ce qui avait été ouvert avant,
  ce que le moteur partagé du site interdit (un écran rend ce que rend un moteur neuf : `tests/test_moteur_partage.py`,
  P1, la référence rejouée dans l'ordre inverse dans le même interpréteur). La ligne ne le dit plus ; il se lit dans le
  sélecteur (★), et `tests/test_api_web_profil.py` garde que la même note rend les mêmes octets, sélecteur ouvert ou non
  (A12 : le redire sur la ligne, calculé en tâche de fond ?). (2) Le mot calé à droite d'un verdict de la note
  s'appelait « droite » dans le kit — un verdict porte déjà « gauche », « op », « droite », les deux membres de ce
  qu'il compare (`ndc_pdf/data.py`) : la note de démonstration du kit imprimait son membre de droite une seconde fois
  (`tests/test_export_pdf_ndc.py` : texte et pixels de la référence). La clé s'appelle « cale ». (3)
  `tests/test_note_identification.py` ne connaissait pas la note des barres : elle y est, avec les huit autres — le nom,
  le chantier, l'indice et la date de la note du site, aux endroits prévus.
- **Le harnais du navigateur** (`site/web/verifier_site.js`) : le contrôle commun (« module:profil », « formats:profil »)
  ouvre la barre neuve aux cinq formats ; « note_pdf » télécharge la note riche aux trois formats et y relit les taux de
  l'écran ; le filtre **« barres »** (rejoué par « module:profil ») joue les GESTES de l'ingénieur sur une même note —
  une charge saisie, une ponctuelle ajoutée, déplacée, retirée, le maintien retiré puis remis, la poutre continue de
  trois travées, les méthodes, un profilé choisi dans le sélecteur (la page, sélecteur ouvert, rend ce que rend le témoin, qui ne l'a pas ouvert),
  deux saisies refusées (par la page, par le moteur), les efforts imposés et le retour à la barre chargée. Son TÉMOIN : les mêmes gestes rejoués hors du navigateur par le
  moteur du dépôt (CPython) ; après chaque geste, le texte de la colonne des résultats, les textes du dessin de la barre
  et la liste des vues sont ceux du témoin — un bouton qui enverrait un autre champ, un paquet du moteur en retard sur
  le dépôt, un nombre que Pyodide calculerait autrement se voient là. Puis trois barres plus chargées que la neuve (une
  ponctuelle ; trois travées et trois cas ; des efforts imposés) aux trois formats : rien ne déborde, aucun nom coupé,
  chaque valeur lisible dans son champ, les cibles au doigt, le dessin visible et sans texte sur un autre.
- **Mesuré** (Chromium sans tête, moteur dans le navigateur) : moteur prêt en 3,5 s ; un recalcul en 0,13 s ; le
  sélecteur d'une famille de 23 profilés en 0,8 s. La note, hors navigateur : 0,5 s pour une barre, 1,1 s pour trois.

**L'âme et ses raidisseurs à l'écran et dans la note** (lot 4, `api_web/profil*.py`, `dessin_barre.py`,
`note_profil.py`).
- **La saisie** : une rubrique de plus, « Appuis » (barre chargée qui a un appui simple) — la barre POSÉE sur ses appuis
  ou tenue « par l'âme » ; posée, sa longueur d'appui s_s et son débord c en rive (mm). Chaque charge ponctuelle dit sa
  longueur d'appui (« à 3 m · sur 100 mm ») tant que les charges ne sont pas appliquées au centre de cisaillement. Une
  note d'avant se relit : posée, s_s = 0, c = 0.
- **Les résultats** : sous les verdicts des deux méthodes, UNE ligne « Âme » — et seulement s'il y a quelque chose à
  faire : des raidisseurs à poser (« Âme · Raidisseur : 2 plats 140 × 15 mm (charge à 2,00 m) », en or), une
  vérification qui ne passe pas ou qui n'est pas couverte (en rouge). Puis, après les blocs des méthodes, le bloc « Âme
  et raidisseurs » : son état (« Sans raidisseur · 77 % », « 1 raidisseur · 87 % »), une ligne par LIEU — sa décision,
  ce que sa vérification la plus défavorable compare, son taux — qui ouvre la fenêtre de calcul du lieu (toutes ses
  vérifications : l'âme seule et la flexion, ou la raison du raidisseur, son écrasement, son flambement). Le bloc
  s'efface quand rien n'appuie sur l'âme. La comparaison des deux méthodes porte le taux de l'âme sur une ligne à part,
  à cheval sur les deux colonnes : il vaut pour les deux. Le statut de la barre est le plus sévère des méthodes cochées
  ET de l'âme ; son taux affiché reste celui des méthodes (un raidisseur dimensionné au plus juste est souvent à 95 %
  et plus : c'est son métier).
- **Les dessins** : un raidisseur est un plat debout sur la barre, à son abscisse, avec sa légende ; la vue des taux a un
  groupe « Âme et raidisseurs », une barre par lieu (« Appui A · 2 plats 70 × 15 »).
- **La note** : après les méthodes, le bloc « Âme et raidisseurs — NBN EN 1993-1-5 » — « Contrôles complémentaires
  EN 1993-1-5 » dans une note « méthode ELS seule » (brief § 6.5) — : ses hypothèses, le tableau des décisions
  (lieu, force, décision, taux), le bilan de ses combinaisons, son verdict, puis chaque vérification avec ses formules
  (deux appuis identiques ne répètent pas leurs facteurs). Le résumé sous le nom de la barre dit les raidisseurs
  (« … · 52 % Eurocode 3 · 1 raidisseur »). Ce que la note ne vérifie pas : les assemblages, les appuis et les soudures
  des raidisseurs (EN 1993-1-8), les raidisseurs intermédiaires d'une âme élancée, le feu et la fatigue.
- **Le harnais du navigateur** (« barres ») : huit gestes de plus — la longueur d'appui des appuis, une charge de
  150 kN posée près de l'appui (le raidisseur paraît : la ligne « Âme », le plat dessiné), sa longueur d'appui saisie
  (il s'efface), les appuis « par l'âme » —, la fenêtre d'un lieu, et une quatrième barre aux trois formats (un
  raidisseur : la ligne « Âme » se lit en entier, à 44 px au doigt).
- **Trouvé en faisant le lot** : (1) le congé d'un IPN n'est pas rangé sous « r » dans le catalogue (r₁, r₂) — le
  chanfrein d'un raidisseur d'IPN valait 0 : il prend r₁ = t_w (`tests/test_barre_ame.py`, les 4 125 plats du catalogue) ;
  (2) deux charges ponctuelles rapprochées n'étaient vérifiées qu'une à une — le § 6.3(2) les veut aussi ensemble : le
  bloc est rendu « à vérifier » (A14) ; (3) la vérification « E-Vb », « non couverte » depuis le lot 2, est remplacée par
  le calcul (« A-Vb ») dans `ec3_section.py` aussi : la section vérifiée seule ne se tait pas non plus.

| Lot | Contenu | Preuve | État |
|---|---|---|---|
| 0 | deuxième avis, inventaire, plan (ce document) | § 2 | **fait** (14/11/2026) |
| 1 | méthode ELS, efforts imposés (`section.py`, `historique.py`, `formules.py`) | `tests/test_barre_historique.py` : 83 contrôles — les 108 valeurs du tableau, les six classeurs à 10⁻⁹, les cinq exemples de la doc, chaque correction, chaque formule recalculée par son gabarit, les 1 419 profilés du catalogue, 29 mutations | **fait** (14/11/2026) ; à l'écran (lot 6) |
| 2 | matériau, classes, section efficace, résistances de section (`parametres.py`, `resultat.py`, `ec3_classe.py`, `ec3_section.py`) | `tests/test_barre_ec3_section.py` : 128 contrôles — les limites du tableau 5.2 et les k_σ du texte ; les classes de profilés connus ; classe et A_eff du module Flambement sur tout le catalogue (4 nuances) ; W_eff recalculé par assemblage à 10⁻⁹ ; résistances et interactions à la main ; 376 formules recalculées par leur gabarit ; 27 384 calculs sur le catalogue ; 52 mutations | **fait** (14/11/2026) ; à l'écran (lot 6) |
| 3 | flambement, torsion, déversement (M_cr numérique), annexe A (`mcr.py`, `ec3_flambement.py`, `ec3_deversement.py`, `ec3_interaction.py`, `ec3.py`) | `tests/test_barre_ec3_barre.py` : 127 contrôles — les 20 valeurs de χ ; les charges de flambement du module Flambement sur tout le catalogue ; M_cr : formule fermée, R1 (sept valeurs), C₁ du moment linéaire, console, maintiens d'une semelle ; R3 avec les deux couples ; l'annexe A recalculée d'une traite pour neuf barres à 10⁻⁹ ; le benchmark 21 (douze valeurs) ; 694 formules recalculées par leur gabarit ; 1 044 barres du catalogue ; 73 mutations | **fait** (14/11/2026) ; à l'écran (lot 6) |
| 4 | EN 1993-1-5 : âme, charges concentrées, raidisseurs (`ec3_ame.py`, `raidisseurs.py`, `ame.py`, `enveloppe.py`) ; à l'écran et dans la note | `tests/test_barre_ame.py` : 160 contrôles (§ 5.9) ; `tests/test_api_web_profil.py` : 77 ; `tests/test_note_profil.py` : 107 (quinze notes) ; le harnais « barres » : 81 | **fait** (14/11/2026) |
| 5 | barre chargée : appuis, charges, combinaisons, sections critiques, tronçons (`combinaisons.py`, `statique.py`, `entree.py`, `moteur.py`) | `tests/test_barre_moteur.py` : 82 contrôles — (6.10) à la main ; la somme pondérée des cas = le formulaire résolu d'un coup ; R2 de bout en bout ; la disposition défavorable d'une poutre continue ; les efforts d'une section pris ensemble ; console, maintiens d'une semelle ; les deux méthodes = leurs appels directs ; 318 formules ; 40 mutations | **fait** (14/11/2026) ; à l'écran (lot 6) |
| 6 | écran (trois formats) et note (ELS, Eurocode 3, les deux) — `api_web/profil*.py`, `dessin_barre.py`, `note_profil.py` | `tests/test_api_web_profil.py` : 66 contrôles — le résultat est celui du moteur, 24 saisies, 21 refus, les deux modes, la relecture de la version 1, le sélecteur, le résultat qui ne dépend pas de ce qui a été ouvert avant, les 17 familles dessinées, aucun texte du dessin d'une barre sur un autre ; `tests/test_note_profil.py` : 89 contrôles — douze notes dites par pdftotext (chaque taux, chaque membre de critère, chaque résultat et chaque terme des formules), aucun mot hors de sa colonne, la sonde « comme dans le navigateur », 20 mutations ; le harnais du navigateur : « module:profil », « formats:profil », « note_pdf » et « barres » (69 contrôles : 20 gestes comparés au moteur du dépôt, trois barres aux trois formats) | **fait** (14/11/2026) |
| 7 | revue : comparaison des méthodes, « À CONFIRMER » restants | — | à faire |

### 9.1 Le socle du lot C (C0, 17/11/2026)

- **Le statut « bloqué »** (`resultat.BLOQUE`) : une donnée manque ou se contredit — les bords des semelles d'un PRS, la
  fabrication d'un tube, une classe de déversement à fournir (méthode ELS), une décision de l'âme impossible (s_s à
  saisir). Le calcul reste fait du côté défavorable, son taux montré POUR INFORMATION, jamais jugé. Gravité :
  non conforme > bloqué > non couvert > hors domaine > conforme (`resultat.pire_statut` ; un statut inconnu lève).
  À l'écran et dans la note : « Bloqué », en un mot ; la raison en une ligne (le premier message de la vérification).
- **La couleur d'un taux sous son statut** : UNE fonction, `profil_mots.couleur_taux(statut, taux)` — rouge si la
  vérification ne passe pas ; « à vérifier » (v) sous bloqué, non couvert ou hors domaine ; sinon la couleur du taux.
  La pastille du verdict, la ligne « Profilé », les cases de la comparaison (écran et note) la lisent. Sous le verdict
  d'une méthode, une ligne dit ce que le taux ne juge pas : « + 1 non couvert : flexion composée ». Mesuré : console de
  3 m, N = 50 kN — la puce Eurocode 3 (57 %) était verte, elle est « v ».
- **Les deux méthodes toujours calculées** : `moteur.verifier` rend toujours la méthode ELS et l'Eurocode 3 ;
  `b["methodes"]` reste lu (anciennes notes) mais ne décide plus du calcul. La NOTE imprime `etat["impression"]` :
  « deux » (défaut), « historique » ou « ec3 ». Une note sans cette clé : version 1 → « ec3 » ; une seule et même méthode
  cochée dans toutes ses barres → celle-là ; sinon « deux ». Le segment « Méthodes » de l'écran écrit cette clé.
- **La comparaison par sujets** (`acier/barre/comparaison.py`) : sept sujets (flambement axe faible, axe fort, flexion
  composée, déversement, cisaillement, âme sous charges, raidisseurs) ; une ligne {sujet, historique, ec3, commun,
  ecart} ; une case {taux (le plus fort JUGÉ), statut (le pire), gouvernant}. Tout identifiant que rend le moteur est
  rangé (`tests/test_barre_comparaison.py`, quinze barres). Une méthode qui refuse la barre (E-SECTION) est « non
  couvert » sur chaque ligne. Les lignes « Le plus défavorable » et l'âme comme méthode sont parties.
- **Le contrat des entrées du lot C** (`entree.normaliser`, chaque clé son défaut et son refus d'une ligne) : `prs`,
  `norme_produit`, `ouvrage`, `combinaisons` {expression, geotechnique, regle_courte_duree}, cas `courte_duree`, `psi0`,
  `N_troncons`, charge `niveau`, ponctuelle `introduction`, maintien `neglige`, `position`, `angle`, `appuis.detail`,
  `points`, `flambement.{y,z}.courbe`, `poids_propre_efforts`, `debords` ; au plus 24 maintiens. Rien de neuf n'est
  CALCULÉ (les blocs C1 à C7 le feront). La tolérance de 5 % de la méthode ELS est retirée (brief § 13.2, point 5) : forcée
  à non, l'ancienne valeur gardée (`tolerance_retiree`) pour la note.
- **Les DÉCISIONS BUREAU** dans la table (§ 5.8) : aucun défaut ne change, aucun taux ne bouge.
- **Reprise de C0 (17/11/2026, après la contre-vérification)** :
  - le passage en efforts imposés copie les longueurs de flambement automatiques en données, et s'en SOUVIENT
    (`profil.COPIEES`) : au retour en barre chargée, une longueur copiée et pas retouchée redevient automatique. Mesuré
    avant : barre neuve → efforts imposés → retour → console de 3 m : L_z = 3 m gardé au lieu de 2,1 × 3 = 6,3 m (IPE 330,
    N = 50 kN : E-Nb-z 6,9 % au lieu de 19,8 %). Une note enregistrée avant la correction n'a pas ce souvenir : ses
    longueurs restent des données (À VÉRIFIER) ;
  - la note PDF peint les tableaux des lieux de l'âme et des combinaisons par `couleur_taux`, comme l'écran : un taux
    bloqué, non couvert ou hors domaine n'est plus vert (avant : 156,7 % « décision impossible » en vert) ;
  - une barre PRS (venue d'un fichier) se relit et se calcule ; l'écran, le sélecteur et la note la refusent en une
    ligne (`profil.refus_prs`) tant que sa fiche et son croquis n'existent pas (bloc C6).

### 9.2 La barre chargée de bout en bout (lot C, C1, 17/11/2026)

Le moteur (`moteur.py`, `statique.py`, `ame.py`) emploie les clés posées par C0. Rien de neuf à l'écran (C3, C4) ni dans la
note (C5) : ils liront ce que le moteur rend.

- **Appui « par l'âme », âme élancée** (défaut NON CONSERVATIF corrigé) : le raidisseur d'appui du § 5.1(2) de l'EN 1993-1-5
  est dû à chaque appui, posé ou tenu par l'âme ; dimensionné sous la réaction entière (À VÉRIFIER).
- **Une hauteur par charge** (`niveau`, à défaut celle de la barre) ; le **poids propre au centre de gravité** (z_g = −z_s,
  0 pour un I bisymétrique). M_cr lit chaque charge à sa cote ; l'âme, chaque ponctuelle sur sa semelle ; le message de
  C_d > 1 (méthode ELS) tombe dès qu'une charge est sur la semelle haute. Un PRS sous charges à plusieurs hauteurs : le
  contrôle (E.1) n'a qu'un z_g, le solveur seul.
- **Maintiens négligés dans M_cr** : les tronçons de M_cr les ignorent (`statique.troncons(…, pour_mcr=True)`) ; le
  flambement et L_d de la méthode ELS les gardent. Tous négligés : la variante (0,4 ; 0,75) est demandée, l'Eurocode 3
  l'admet (bâtiment) ou la refuse (« hors bâtiment ») — test 9 du § 11.5 : R2 0,741 et 0,891. Un tronçon n'est « entre
  appuis » que si ses deux bouts sont des appuis : le bout libre d'une console ou d'un débord n'en est pas un — une console
  au maintien négligé garde le couple par défaut (IPE 300, 3 m : 0,872 et non 0,751 ; contre-vérification du lot C).
- **Position de la barre** : horizontale (poids propre en flexion), verticale (q·L dans N, en compression, sur toute la
  barre — prudent, À VÉRIFIER), inclinée de α (q·cos α en flexion, q·L·sin α dans N ; sans angle : refusée). Efforts
  imposés : les interrupteurs « dans P » (P += q·L) et « en flexion » (M += q·L²/8, V += q·L/2) ajoutent le poids propre
  au service et, × γ_G,sup, à l'ELU — compté deux fois si les efforts saisis le contiennent (À VÉRIFIER, dit dans la note).
- **Courbe imposée par axe** : celle de la méthode ELS (forcée, dite) ; l'Eurocode 3 garde son tableau 6.2.
- **Sections critiques** : les points demandés et les racines du moment sont des sections vérifiées, rendues pour le
  dessin (`statique["sections"]`).
- **C_mi,0 de la travée extraite** (continue, encastrée, bi-encastrée) : la formule générale avec la flèche et la portée de
  la travée du tronçon ; le M du tronçon (jamais plus grand que celui de la travée : du côté sûr).
- **Appuis un par un** (`appuis.detail`) : la torsion (fourche ou encastrée) entre dans M_cr ; s_s, c et le contact, dans
  les lieux de l'âme.
- **PRS chargés depuis leurs charges** : la semelle comprimée d'un tronçon est celle que σ_x = N/A + M·z/I_y comprime le
  plus, au plan moyen, à la section du plus grand |M| ; aucune comprimée (une traction l'emporte) : le signe de M, dit
  (À VÉRIFIER). PRS-C : M_ELS 2 853,66 kN·m, σ 117,11 ; EC3 M_Ed 4 200,2 ; 1,007 / 0,967 / 0,937 (7, 8, 9 maintiens).
- **N par tronçon** (`N_troncons`) : chaque section à son N, dans les deux méthodes ; le flambement et (6.61), (6.62) au
  plus grand N de la barre (prudent, À VÉRIFIER) — la méthode ELS, qui les lit dans le même calcul, vérifie la section
  aux deux (un tirant HEB 200 de −300 / −100 kN : 1,517 et non 1,353, contre-vérification du lot C) ; η₁ de l'âme au N du
  lieu.
- **Mode d'introduction déclaré** d'une ponctuelle (a, b, c) : appliqué tel quel ; « auto », la règle du module.
- **Débords** (porte-à-faux, rdm/poutres) : la barre va de 0 à g + Σ portées + d ; le bout d'un débord est libre. La
  longueur de flambement d'un axe où le bout du débord est libre est celle d'Euler de la barre ENTIÈRE sur ses points
  tenus (`statique.longueur_euler` : E·I et N constants, appuis articulés — la travée voisine tourne avec le débord), au
  moins 2,1 × le débord ; elle sert aussi de L_d au tronçon du débord (méthode ELS, hors domaine). 4 m + 2,5 m : 7,89 m
  (2,1 × 2,5 = 5,25 m surestimait N_cr 2,26 fois — contre-vérification du lot C) ; 6 m + 1,5 m : 7,91 m. Une alerte,
  dite une fois (À VÉRIFIER) ; le verdict ELS reste « hors domaine » quand la travée gouverne le déversement.
- **Origine des longueurs rendues** (`longueurs[a]["origine"]`) : « portee », « troncon », « console », « debord »,
  « saisie ».
- **R2 de bout en bout** (critère du lot C) : `tests/test_barre_moteur.py`, § 10 — M_Ed 68,70, V_Ed 27,80 ; propriétés du
  brief : 118,18 / 0,887 et 86,98 / 1,093 ; catalogue (I_t 19,77 cm⁴) : 117,52 / 0,890 et 86,35 / 1,0996.

### 9.3 Les résultats à l'écran (lot C, C4, 18/11/2026)

`api_web/profil_resultat.py`, `profil_mots.py`, `dessin_barre.py` ; contrôles : `tests/test_api_web_profil_resultat.py`.

- **Une information à un endroit.** Le taux d'une méthode est écrit au seul verdict : ni l'en-tête des blocs, ni la ligne
  « Profilé » (plus de masse : le bouton du profilé, à gauche, la dit), ni le pied. R2 : « 144 % » 7 → 5 fois dans la
  colonne, « 110 % » 4 → 3 (le verdict, les lignes de vérification, les cases de la comparaison).
- **Le verdict ELS** montre la flexion, le déversement ou le flambement — jamais le contrôle hérité H-EQ (brief § 6.1) ;
  quand H-EQ s'écrit plus haut, son taux est dit sous le verdict (À VÉRIFIER : le mettre en tête quand il refuse seul).
- **Comparaison** : par sujet, ELS, Eurocode 3, l'écart (points, nommé par `comparaison.py`) et ce qui gouverne.
- **Écarts avec les classeurs** (brief § 6.7) : la ligne « n corrections des classeurs » du bloc ELS ouvre la fenêtre
  « ecarts » — une ligne par correction rendue (`corrections`), où elle joue ; jamais un choix du bureau.
- **Avant/après raidisseur** (§ 15.4) : en tête de la fenêtre du lieu raidi, grandeur · sans · avec (F_Rd, σ_z,Ed, η₂,
  interaction ou critère 6.2.1(5) ; σ de la section efficace, N_b,Rd, contact), puis le rappel sur τ. Test 18 : 387,8 kN ;
  1,161 (450/387,76 ; le brief écrit 1,160) ; 1 002,2 kN · 0,449 ; 204,5 ≤ 225 N/mm² · 0,909.
- **Fenêtres** : une vérification ne montre que ses formules et ses messages (les hypothèses : « Tout le calcul ») ;
  P_cr,y et P_cr,z cachés sans effort normal ; « Tout le calcul » de l'Eurocode 3 dit l'autre variante de chaque décision
  (§ 6.3.2.2 pour information, A_v,y, n_pl avec l'autre γ_M). -FM écrit comme la norme : η₂ + 0,8·η₁ = 0,513 ≤ 1,4
  (`ame.py` : valeur = la somme, limite 1,4 ; le taux 0,366 ne change pas).
- **Dessin** : le signe du niveau de chaque charge (▲ ● ▼, des formes) quand les niveaux sont mêlés, le maintien négligé
  creux, la position (verticale, inclinée), N par tronçon, les débords (barre entière, appuis de `statique.appuis_x`), les
  points vérifiés (p1, p2…) ; chaque repère a sa légende en mots.

## 10. À confirmer par le bureau

| # | Question | En attendant |
|---|---|---|
| Q1 | ANB de l'EN 1993-1-1 : la méthode 1 imposée ; la borne de C_mi,0 | tranchés au lot B : méthode 1 sous l'écriture de l'annexe D ANB (6.3.3(5)) ; pas de borne (tableau D.1 ANB) |
| Q2 | ANB de l'EN 1990 : ψ de la neige, du vent, de la température | valeurs EN |
| Q3 | f_u du S355 ; A_v,y des I laminés | tranchés au lot A : f_u selon l'épaisseur, tableau 3.3 ANB (470) ; A_v,y = 2·b·t_f (DÉCISION BUREAU n° 3) |
| Q4 | Méthode ELS : combinaison de service à 100 % ; tolérance de 5 % (non) ; C = 1 par défaut ; déversement activé par défaut | tranché le 16/11/2026 (brief v12 § 13.2 point 5) : la méthode des classeurs « Déversement », C = C_d = 1, f_y nominale ; la variante du fascicule éteinte |
| Q5 | Tube Ø610 : fini à chaud (courbe a) ou formé à froid et soudé (c) ; horizontal ou vertical (poids propre) | a ; non compté |
| Q6 | Seuil du statut « à la limite » dans la note | celui des notes existantes (95 %) |
| Q7 | Les fichiers d'origine (classeurs, documentation) : les garder hors du dépôt public | hors du dépôt |
| Q9 | La note imprime les méthodes cochées à l'écran (pas de choix au moment d'imprimer) ; la méthode ELS y vient la première | lot C (C0d) : l'écran montre toujours les deux ; la note imprime « les deux » (défaut), l'ELS seule ou l'Eurocode 3 seul — le choix de la NOTE (etat « impression ») ; la méthode ELS vient la première |
| Q8 | Les pages « Benchmarks 17 à 22 » de l'aide de SCIA Engineer (ECCS N°119), en pièce jointe (captures ou PDF), avec les remarques de SCIA sur les erreurs de la référence : elles ne sont pas joignables d'ici | benchmark 21 : quatorze valeurs du brief comparées (diagramme de M_y déduit) ; 17 à 20 et 22 bloqués (`tests/test_barre_benchmarks.py`) |
| Q10 | Âme sous force concentrée et flexion (EN 1993-1-5 § 7.2) : η₁ ÉLASTIQUE (M_Ed/(W_el·f_y)), même pour une section de classe 1 ou 2 — prudent ; le brief le laisse « à confirmer » | élastique |
| Q11 | La longueur d'appui s_s non saisie : le bureau a-t-il une valeur d'usage (la largeur d'un appui courant) à proposer par défaut ? | lot B : vide, calcul à la borne 0 (prudent) ; « décision impossible : s_s à saisir » si la borne ne passe pas (brief § 8.3, point 5) |
| Q12 | Les appuis sont « posés » par défaut (la réaction appuie sur l'âme) ; le bureau attache-t-il plutôt ses poutres par l'âme (platine, cornières) ? | posés |
| Q13 | Les raidisseurs proposés : plats de 8, 10, 12, 15, 20 mm, largeur au pas de 5 mm jusqu'à la demi-semelle, chanfrein égal au congé arrondi à 5 mm — les habitudes de l'atelier ? | ainsi |
| Q14 | Les jeux de données JRC (« Commentary and worked examples to EN 1993-1-5 ») cités par le brief pour le lot 4 ne sont pas joignables d'ici : en pièce jointe, ils s'ajouteraient au cas R4 | R4, le témoin écrit depuis la norme, les calculs à la main |

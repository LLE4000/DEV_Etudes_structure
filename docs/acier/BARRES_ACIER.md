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
| H9 | Flexion composée plane | σ = P/(A·φ_min) + C·M/(φ_ed·W·PE) ≤ f_y / s | doc p. 20, formule (5) |
| H10 | Flexion composée déviée | σ = P/(A·φ_min) + C_y·M_y/(φ_ed·W_y·PE_y) + C_z·M_z/(W_z·PE_z) ≤ f_y / s | doc p. 26 |
| H11 | Cisaillement | τ_z = V_z / A_âme ; τ_y = V_y / A_ailes ; τ = τ_y + τ_z | classeurs |
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
8. Hors du domaine de la doc — nuance hors AE235 à AE355 (lue sur la valeur NOMINALE : une S235 épaisse y reste), profil
   creux formé à froid de paroi de plus de 6 mm : statut « hors domaine » (lot A, brief § 9) — le calcul est montré, jamais
   « vérifié », jamais caché derrière le taux de la barre. f_y : celle de l'Eurocode 3 selon l'épaisseur (brief § 13.2
   point 5, § 5.1) ; au-delà de 80 mm, la barre est refusée (« non couvert »), comme par l'Eurocode 3.
9. La tolérance de 5 % de la doc (p. 13 et 23) n'est PAS appliquée (option éteinte ; question Q4).
10. Le coefficient C et le déversement de la doc (§ 4.4, 4.5), que les classeurs ne calculent pas.
11. La traction et la flexion simple : la doc ne traite que la compression. Le module écrit alors la résistance simple
    σ = |N|/A + M/W ≤ f_y/s (φ = 1, PE = 1), et le dit.

### 4.4 Le coefficient C (doc p. 20 et 21)

C = 1 par défaut (comme les classeurs). Options : moments d'extrémité seuls, C = √(0,3·(1 + ψ²) + 0,4·ψ), ψ = M₂/M₁,
|M₂| ≤ |M₁| (de 1 pour ψ = 1 à 0,447 pour ψ = −1) ; charges appliquées sur la barre : C = 0,85.

### 4.5 Le déversement de la doc (p. 15 à 19, exemple p. 25)

Pour un I ou un H fléchi autour de l'axe fort, le déversement est remplacé par le flambement d'une barre fictive — la
semelle comprimée et un sixième de l'âme :

    I_d = t_f·b³/12        A_d = b·t_f + h_w·t_w/6  (h_w = h − 2·t_f)        r_d = √(I_d / A_d)
    λ_ed = L_c / r_d       φ_ed = φ(λ_ed / λ_E), courbe de l'axe FAIBLE du profil

L_c est la distance entre maintiens latéraux de la semelle comprimée (la longueur réelle, sans u : la doc prend 600 cm
pour une colonne dont u = 1,5). φ_ed divise le terme de moment fort (H9, H10). Tube rond ou carré : φ_ed = 1 ; tube
rectangulaire : λ_ed = √(π·√(E/G)·L_c·W_y / √(I_z·I_t)) (doc p. 18), avec la courbe du tube.

**À savoir** (E8) : cette formule néglige la rigidité de torsion (la doc : « pour les grandes poutres à âme pleine ») ;
elle est très sévère pour un H trapu non maintenu — l'exemple de la doc elle-même (HEA 200 sur 8 m) tombe à
φ_ed = 0,31, quand l'Eurocode 3 donne χ_LT ≈ 0,70. Elle est activée par défaut, comme le veut le brief ; une semelle
comprimée maintenue en continu (dalle, bac) donne φ_ed = 1 et retrouve le classeur. Question Q4.

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
32,64) — A − h_w·t_w pour un U soudé ; une cornière est refusée ;
tube rectangulaire A·h/(b + h) ; tube rond 2A/π. Flexion + effort
tranchant (6.2.8) ; flexion + effort normal (6.2.9 : (6.33) à (6.41)) ; classe 3 (6.42) ; classe 4 (6.44) ; 6.2.10.
Tube rond en N + M : critère linéaire (6.2), prudent.

**Ce que le moteur a tranché** (`acier/barre/ec3_section.py`, lot 2) :
- tube rond : la RÉSULTANTE des deux moments et celle des deux efforts tranchants (il n'a pas d'axe) ; n + M/M_pl,Rd ≤ 1 ;
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
sont pas joignables d'ici (domaine hors de la liste autorisée du réseau). Les classes et les résistances sont donc
prouvées autrement : limites du texte aux points donnés en clair, profilés de classe connue, calculs à la main sur les
valeurs du catalogue, recoupement avec le module Flambement en service sur tout le catalogue. À rejouer si les pages
deviennent lisibles, ou sur des cas recalculés par le bureau dans son logiciel.

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
≤ 1/λ̄_LT² ; tableau 6.5 (h/b ≤ 2 : b ; > 2 : c). Facteur f = 1 − 0,5·(1 − k_c)·[1 − 2·(λ̄_LT − 0,8)²] ≤ 1 ; k_c
(tableau 6.6) : moment constant 1,0 ; linéaire 1/(1,33 − 0,33ψ) ; charge répartie sur deux appuis **0,94** ; charge
concentrée à mi-portée **0,86** ; diagramme hors tableau : k_c = 1. Négligé si λ̄_LT ≤ λ̄_LT,0 ou M_Ed/M_cr ≤ λ̄_LT,0².

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
  au centre de cisaillement, sur la semelle haute ou sous la basse ; un diagramme non décrit est pris CONSTANT ;
- un tube rectangulaire relève du cas général (§ 6.3.2.2, courbe d) ; tubes ronds et carrés : pas de déversement ;
- le couple (λ̄_LT,0 ; β) : (0,2 ; 1,0), UN couple (lot A) — NBN EN 1993-1-1 ANB:2018, 6.3.2.3(1) : M_cr calculé sur la
  section brute avec les combinaisons, la distribution réelle des moments et les maintiens, ce que fait le solveur du
  module, maintiens présents ou non ; (0,4 ; 0,75) suppose les maintiens TOTALEMENT ignorés dans M_cr, une option que le
  module n'a pas. Le seuil de 6.3.2.2(4) des tubes lit le même λ̄_LT,0 (0,2). Les clés « sans maintiens » de la table
  sont retirées, et une surcharge qui les nomme est refusée avec son motif.

### 5.7 Flexion composée avec instabilité — § 6.3.3, annexe A (vérifié ; E2)
(6.61) et (6.62) ; méthode 1. Tableau A.2 : moment linéaire C_mi,0 = 0,79 + 0,21ψ + 0,36·(ψ − 0,33)·N_Ed/N_cr,i ;
charge répartie **1 + 0,03**·N_Ed/N_cr,i ; charge concentrée **1 − 0,18**·N_Ed/N_cr,i ; cas général
1 + (π²·E·I_i·|δ_x|/(L²·|M_i,Ed(x)|) − 1)·N_Ed/N_cr,i.

**Ce que le moteur fait** (`acier/barre/ec3_interaction.py`, lot 3) : (6.61) et (6.62) avec tous les facteurs de
l'annexe A (μ, w ≤ 1,5, n_pl, a_LT, C_my, C_mz, C_mLT, b_LT à e_LT, C_yy à C_zz et leurs planchers, k_yy à k_zz ;
facteurs élastiques en classes 3 et 4) ; C₁ = 1/k_c² ; une barre sans déversement possible a λ̄_0 = 0 ; une flexion
déviée SANS effort normal passe par les mêmes expressions ; une barre tendue est prise sans effort normal (du côté
sûr). Une CONSOLE sous effort normal et flexion : (6.61)/(6.62) rendues « non couvert » (lot A, brief § 7.8, DÉCISION
BUREAU n° 9 : aucun C_m ne vaut pour une console) ; N seul et M seul restent vérifiés. La borne inférieure de C_mi,0
(supposée 1 − N_Ed/N_cr,i) est active par défaut et **change un résultat** : sur
l'exemple 5 de l'ECCS, C_mz,0 passe de 0,771 à 0,842 — à confirmer (Q1).

**Le benchmark 21 de l'aide de SCIA Engineer** (ECCS N°119, exemple 5 : IPE 500 S235, 3,75 m, N_Ed = 500 kN, flexion
déviée). Sa page n'est pas joignable d'ici ; la recherche en a rendu les données et une partie des résultats. Sont
reproduits : classe 1, N_c,Rd 2 715 kN, M_c,y,Rd 516, M_c,z,Rd 78,9 kN·m, V_pl,z,Rd 814 kN (0,3 %), V_pl,y,Rd 912 kN
(1 %, avec A − h_w·t_w ; le module prend 2·b·t_f depuis le lot A, DÉCISION BUREAU n° 3 : 868,3 kN) ; μ_y 1,00, w_y 1,138, μ_z 0,937, w_z 1,5, C_mz,0 0,771, M_cr,0 899 kN·m, λ̄_0 0,757, a_LT 0,998 (0,2 %). Les
facteurs qui dépendent du diagramme de M_y (b_LT à e_LT, C_yy…) attendent la page elle-même (Q8) ; en attendant,
l'annexe A est recalculée d'une traite dans le test, pour neuf barres, à 10⁻⁹.

### 5.8 Les paramètres nationaux (une seule table dans le moteur : `acier/barre/parametres.py`)

| Paramètre | Valeur | Source | Statut |
|---|---|---|---|
| γ_M0 ; γ_M1 ; γ_M2 | 1,00 ; 1,00 ; 1,25 | valeurs EN, reprises par l'ANB | à pointer sur l'ANB |
| λ̄_LT,0 ; β (§ 6.3.2.3) | 0,2 ; 1,0 (un seul couple, lot A) | NBN EN 1993-1-1 ANB:2018, 6.3.2.3(1) : M_cr avec les combinaisons, la distribution réelle des moments et les maintiens (§ 5.6) | pointé |
| Facteur f (§ 6.3.2.3(2)) | formule EN | brief | à pointer |
| Méthode de § 6.3.3 | méthode 1 (annexe A) | ANB d'après l'aide de STAAD.Pro | **À CONFIRMER** |
| Borne inférieure de C_mi,0 (tableau A.2) | 1 − N_Ed/N_cr,i (supposée, comme l'annexe française) | l'aide de STAAD.Pro dit qu'il y en a une, sans la donner | **À CONFIRMER** |
| η (EN 1993-1-5 § 5.1) | 1,00 pour les I et H laminés et les tubes ; 1,20 pour les sections soudées (I soudés, caissons), jusqu'au S460 (lot A) | NBN EN 1993-1-5 ANB:2011, 5.1(2), note complémentaire | à pointer |
| Combinaison ELU | (6.10) : 1,35·G + 1,5·Q₁ + Σ 1,5·ψ₀·Q_i | NBN EN 1990 ANB | à pointer |
| ψ₀ / ψ₁ / ψ₂ | A, B : 0,7/0,5/0,3 ; C, D, F : 0,7/0,7/0,6 ; E : 1,0/0,9/0,8 ; G : 0,7/0,5/0,3 ; H : 0/0/0 | NBN EN 1990 ANB (brief) | à pointer |
| ψ de la neige, du vent, de la température | 0,5/0,2/0 ; 0,6/0,2/0 ; 0,6/0,5/0 (valeurs EN) | — | **À CONFIRMER** |
| Combinaison de la méthode ELS | G + ΣQ, tout à 100 % | pratique du bureau | **À CONFIRMER** (Q4) |

« À pointer » : la valeur est celle de l'EN, connue ; il reste à la lire sur le texte belge. « À CONFIRMER » : la valeur
change un résultat et sa source n'est pas le texte — elle est imprimée comme telle dans la note.

**Les combinaisons, telles que le moteur les écrit** (`acier/barre/combinaisons.py`, lot 5) : pour chaque ARRANGEMENT
des cas variables (chacun présent ou absent — c'est la recherche de la disposition défavorable, travée par travée) et
chaque nature dominante parmi les présents, γ_G·ΣG + γ_Q·Q_dominante + Σ γ_Q·ψ₀·Q_autres, avec γ_G = 1,35 puis 1,00
(G favorable). Les cas d'une MÊME nature sont une seule action disposée librement : ils reçoivent tous γ_Q quand leur
nature domine. Au plus quatre cas variables. Méthode ELS : G + ΣQ à 100 % pour chaque arrangement.

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
  et pour chaque charge ponctuelle. NON SAISIE, elle vaut 0 : le cas le plus défavorable, et c'est écrit (hypothèses,
  message de la vérification). Elle n'est jamais supposée (brief § 8.2).
- **La résistance de l'âme** (lot A) : F_Rd = f_y·χ_F·l_y·t_w/γ_M1 avec la f_y de CALCUL de la section ; λ̄_F avec la f_y
  de l'âme selon t_w. Les DEUX branches de m₂ (m₂ = 0,02·(h_w/t_f)², cohérente si λ̄_F > 0,5 ; m₂ = 0, si λ̄_F ≤ 0,5) sont
  calculées : quand les deux sont cohérentes, la plus PETITE F_Rd est retenue, et dite (brief § 8.2). Type (c) : le
  CONTACT d'extrémité est « inconnu » (DÉCISION BUREAU n° 2) — s_s = 0 dans (6.10) seulement, k_F et le plafond de l_e
  gardent s_s ; la variante avec contact est donnée pour information (IPE 300, s_s = 100 mm, c = 0 : 199,8 kN, 248,9 kN
  avec contact).
- **L'âme seule**, par lieu et par combinaison : η₂ = F_Ed/F_Rd (« A-Lnn-F » ; « A-Lnn-B » pour la force transmise),
  puis, dès qu'un moment ou un effort normal s'y trouve, (η₂ + 0,8·η₁)/1,4 ≤ 1 (« A-Lnn-FM ») avec η₁ ÉLASTIQUE —
  |N|/(f_y·A) + |M|/(f_y·W_el), A_eff et W_eff en classe 4 (Q10).
- **LA DÉCISION**, lieu par lieu : « sans raidisseur », avec les taux ; ou « raidisseur » dès qu'une combinaison dépasse
  η₂ = 1 ou l'interaction — et à chaque appui posé d'une âme élancée au cisaillement (§ 5.1(2)). La raison est écrite
  (« Sans raidisseur : F_Ed = 675,0 kN > F_Rd = 611,2 kN »).
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
- **Voilement induit par la semelle** (§ 8), avec k = 0,4 (du côté sûr) : aucun I du catalogue ne le dépasse, même en
  S460 ; une ligne d'hypothèse le dit, la vérification « A-S » ne paraît que s'il est dépassé.
- **Efforts imposés** : seuls le voilement par cisaillement et le § 8 sont vérifiés — les appuis et les charges ne sont
  pas décrits, et le bloc le dit.
- **Ce qui n'est PAS fait** (v1), et dit — jamais rendu « ok » : la paroi d'un profil creux sous une force concentrée
  (EN 1993-1-8 : un avertissement) ; les raidisseurs intermédiaires d'une âme élancée et le montant d'extrémité rigide ;
  les soudures des raidisseurs (EN 1993-1-8) ; DEUX FORCES RAPPROCHÉES sur la même semelle — leurs longueurs chargées
  l_y se recouvrent —, que le § 6.3(2) demande de vérifier aussi ensemble : la vérification « A-GR » est alors rendue
  « non couvert » ; une réaction de soulèvement (ancrage : un avertissement).

**La preuve** — `tests/test_barre_ame.py`, 181 contrôles (lot A, deuxième passe) : le cas R4 du brief (IPE 300 : 283,2 /
387,8 / 425,1 kN en type (a) ; en type (c), 199,8 kN sans contact — 248,9 et 335,9 kN avec contact, pour information —,
avec leurs valeurs intermédiaires) ; un témoin écrit depuis le texte de la norme, égal au moteur sur les 275 I et H du
catalogue, deux nuances, douze charges ; l'IPE 750x134 S355 à la main (V_bw,Rd = 1 722,0 kN ; l'IPE 750x147 n'est plus
élancé avec η = 1,00) et l'interaction (7.1) ; un caisson soudé saisi (η = 1,20 : 72·ε/η = 60·ε, plateau χ_w = 1,20) ; les raidisseurs à la main (IPE 300, HEB 300) et la forme du plat sur
tout le catalogue (4 125 plats : chaque I, trois nuances, cinq épaisseurs) ; la décision lieu par lieu (R2 : la charge de 36 kN, η₂ = 0,093, « pas de
raidisseur ») ; 623 formules recalculées par leur gabarit et relues sur leurs nombres imprimés ; 82 mutations.

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
- p. 14, HEA 220, 5 m, P = 500 kN, M = 25 kN·m : P_cr 4485 kN (4496), φ_b 0,863 (0,86), σ = 90,1 + 58,3 = 148,4 (149).
- p. 23, poutre HEA 200, 8 m, P = 38 kN, M = 48 kN·m, avec la dispense : φ_c 0,2614 (0,264), σ = 27,0 + 129,6 = 156,6
  (26 + 130 = 156) ; sans la dispense : φ_ed 0,307 → 422 N/mm².
- p. 24-25, colonne HEA 300, 6 m, u = 1,5, P = 168 kN, M = 110 kN·m, C = 0,85 : φ_c 0,4009 (0,4), r_d 8,30 cm, φ_ed 0,676
  (0,7), P_cr 4673 kN, PE 0,9461 (0,95), σ = 37,3 + 116,0 = 153,3 (37,3 + 111,5 = 148,8).

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
- Méthode ELS : dans chaque section, avec la longueur de déversement de son tronçon — un maintien d'une seule
  semelle ne compte que si elle est comprimée sur tout le tronçon.
- Hypothèses dites dans la note : les appuis tiennent le déplacement latéral et la torsion (fourches) ; une console est
  encastrée à sa racine et flambe sur deux fois sa longueur ; l'axe faible flambe sur le plus long tronçon.
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
| `api_web/profil.py` | la note (version 2 : `{"elements": [barre…], "actif"}`), ses saisies, le sélecteur de profilés, les vues, la relecture des notes de la version 1 |
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

## 10. À confirmer par le bureau

| # | Question | En attendant |
|---|---|---|
| Q1 | ANB de l'EN 1993-1-1 : la méthode 1 imposée ; la borne de C_mi,0 (le couple (λ̄_LT,0 ; β) est lu : (0,2 ; 1,0), 6.3.2.3(1), lot A) | § 5.8, imprimé « à confirmer » |
| Q2 | ANB de l'EN 1990 : ψ de la neige, du vent, de la température | valeurs EN |
| Q3 | f_u du S355 ; A_v,y des I laminés | tranchés au lot A : f_u selon l'épaisseur, tableau 3.3 ANB (470) ; A_v,y = 2·b·t_f (DÉCISION BUREAU n° 3) |
| Q4 | Méthode ELS : combinaison de service à 100 % ; tolérance de 5 % (non) ; C = 1 par défaut ; **déversement de la doc activé par défaut** (sévère pour un H non maintenu, § 4.5) | oui, oui, oui |
| Q5 | Tube Ø610 : fini à chaud (courbe a) ou formé à froid et soudé (c) ; horizontal ou vertical (poids propre) | a ; non compté |
| Q6 | Seuil du statut « à la limite » dans la note | celui des notes existantes (95 %) |
| Q7 | Les fichiers d'origine (classeurs, documentation) : les garder hors du dépôt public | hors du dépôt |
| Q9 | La note imprime les méthodes cochées à l'écran (pas de choix au moment d'imprimer) ; la méthode ELS y vient la première | ainsi |
| Q8 | Les pages « Benchmarks 17 à 22 » de l'aide de SCIA Engineer (ECCS N°119), en pièce jointe (captures ou PDF) : elles ne sont pas joignables d'ici | douze valeurs du benchmark 21 reproduites ; l'annexe A recalculée d'une traite dans le test |
| Q10 | Âme sous force concentrée et flexion (EN 1993-1-5 § 7.2) : η₁ ÉLASTIQUE (M_Ed/(W_el·f_y)), même pour une section de classe 1 ou 2 — prudent ; le brief le laisse « à confirmer » | élastique |
| Q11 | La longueur d'appui s_s non saisie vaut 0 (le cas le plus défavorable) ; le bureau a-t-il une valeur d'usage (la largeur d'un appui courant) à proposer par défaut ? | 0, et c'est écrit |
| Q12 | Les appuis sont « posés » par défaut (la réaction appuie sur l'âme) ; le bureau attache-t-il plutôt ses poutres par l'âme (platine, cornières) ? | posés |
| Q13 | Les raidisseurs proposés : plats de 8, 10, 12, 15, 20 mm, largeur au pas de 5 mm jusqu'à la demi-semelle, chanfrein égal au congé arrondi à 5 mm — les habitudes de l'atelier ? | ainsi |
| Q14 | Les jeux de données JRC (« Commentary and worked examples to EN 1993-1-5 ») cités par le brief pour le lot 4 ne sont pas joignables d'ici : en pièce jointe, ils s'ajouteraient au cas R4 | R4, le témoin écrit depuis la norme, les calculs à la main |

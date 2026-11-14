# Barres en acier — méthode historique (NBN B 51-001) et Eurocode 3

*Ouvert le 14/11/2026. Ce document est la SPÉCIFICATION VÉRIFIÉE du chantier « barres acier » : il remplace le brief
reçu ce jour-là (rédigé en conversation à partir des classeurs du bureau), qu'il reprend, corrige et complète. Il est
tenu à jour lot par lot.*

**Demande du propriétaire (14/11/2026).** Vérifier un profilé métallique — poutre, poteau, poutre-poteau — avec ses
instabilités (flambement, déversement, voilement local, raidisseurs), par DEUX méthodes calculées ensemble : la méthode
historique du bureau (ancienne norme belge NBN B 51-001, contraintes admissibles, charges de service) et l'Eurocode 3
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
| E3 | brief § 7.1, f_u du S355 | tableau 3.1 de l'édition 2005 (EN 10025-2) : 355 / **510** (t ≤ 40 mm), 335 / 470 ; 490 est la valeur du S355 N/NL (et de l'édition 2022) | aucun en v1 (f_u n'entre dans aucune vérification retenue) | 510, `À CONFIRMER` sur le texte |
| E4 | brief § 11.2, HEB 200 en compression | N_b,z,Rd = 570,1 kN suppose λ = L/i avec i = 5,07 cm (l'arrondi du classeur). Avec N_cr = π²·E·I_z/L² (la définition de l'EC3, I_z = 2003 cm⁴) : λ̄ = 1,5139, χ = 0,3101, **569,1 kN** | 0,2 % | les deux valeurs sont dans le test ; le moteur suit la définition de l'EC3 |
| E5 | brief § 4.1 | « tableau 13.2 » : c'est le tableau 13.1 (doc p. 16) | — | corrigé |
| E6 | brief § 6.4, « tubes : φ_ed = 1 » | la doc (p. 17-18) donne une formule pour les profils FERMÉS : λ_ed = √(π·√(E/G)·ℓ·W_x/√(I_y·J_T)) (« 5,1 », exact 5,066), « généralement proche de l'unité ». Exact pour un tube rond ou carré (pas de déversement) ; pour un tube RECTANGULAIRE fléchi autour de l'axe fort, la formule de la doc s'applique | ≤ 2 % | tube rond ou carré : 1 ; rectangulaire : la formule de la doc, avec la courbe du tube |
| E7 | doc p. 18 et 19 | deux coquilles dans les formules INTERMÉDIAIRES du moment critique des profils ouverts : (h_a/2ℓ)² pour (h_a/2ℓ²)² (p. 18) ; π⁴ pour π² au premier terme (p. 19) | aucun : la méthode n'emploie que les formules (3) et (4) de la p. 19 | noté |
| E8 | doc p. 20, dispense « C/φ_ed = 1 si P/(A·φ) ≤ 0,15·σ_adm » | annotée « FAUX » dans la marge ; l'exemple de la p. 23 l'applique pourtant (à 0,17·σ_adm). Sans elle, cet exemple (HEA 200, 8 m, sans maintien) donne φ_ed = 0,307 et σ = 422 N/mm² au lieu de 130 | majeur pour une poutre non maintenue | la dispense n'est PAS reprise (brief) ; voir § 4.5 et la question Q4 |
| E9 | classeurs, poids propre | 78,6 kN/m³ × A (g = 10) ; la doc p. 8 prend 7,85 t/m³ × 9,81 = 77,0 | 2 % sur un terme mineur | le site prend la masse du catalogue × 9,81 ; le test des classeurs reprend leur saisie |

Données des classeurs à signaler (non corrigées dans les tests, § 6) : A = 2520 mm² pour le CHS 168,3 × 6 (catalogue :
3059) ; « section ailes » du HEA 120 = une seule aile (960 au lieu de 1920 mm²) ; W = 3506 cm³ pour le tube Ø610 × 12,5
(I/(D/2) = 3435 : le classeur divise par le rayon MOYEN) ; valeurs d'axe faible dans les cases « fort » de FLAMBECBD ;
f_y = 450 N/mm² pour le tube Ø610 (hors des nuances AE235 à AE355 de la doc).

## 4. La méthode historique (NBN B 51-001, contraintes admissibles)

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
| H13 | Poids propre | q = masse × g ; M = q·L²/8 ; V = q·L/2 (poutre sur deux appuis) | classeurs |

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
8. Nuance hors AE235 à AE355 : alerte « extrapolation ».
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

### 4.6 Hors de la méthode historique

La doc ne traite ni le voilement local, ni l'âme sous charge concentrée, ni les raidisseurs. Dans une note « méthode
historique seule », ces contrôles paraissent dans un bloc séparé « Contrôles complémentaires EN 1993-1-5 » (§ 5.9).

## 5. L'Eurocode 3 (EN 1993-1-1:2005 + AC:2009 + A1:2014, ANB:2018 ; EN 1993-1-5:2006)

Le brief (§ 7 et 8) est repris tel quel, SAUF les corrections E1 à E3. Ne sont notés ici que les points vérifiés un à un
et les paramètres nationaux ; chaque module du moteur (`acier/barre/ec3_*.py`) porte ses formules et leurs clauses.

### 5.1 Matériau — § 3.2
E = 210 000 N/mm² ; G = 81 000 N/mm² ; ν = 0,3 ; ε = √(235/f_y). Tableau 3.1, épaisseur max(t_f ; t_w) :
S235 235/360 puis 215/360 ; S275 275/430 puis 255/410 ; S355 355/510 puis 335/470 (E3) ; S460 : 460/540 puis 430/540
(EN 10025-3) — le moteur du flambement existant (`acier/flambement/normes.py`) porte déjà f_y ; f_u n'est pas employé.

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
(le catalogue porte la première expression, sans le plancher) ; A_v,y = A − h_w·t_w (c'est la valeur du catalogue ;
par analogie avec le cas (e), `À CONFIRMER`) ; tube rectangulaire A·h/(b + h) ; tube rond 2A/π. Flexion + effort
tranchant (6.2.8) ; flexion + effort normal (6.2.9 : (6.33) à (6.41)) ; classe 3 (6.42) ; classe 4 (6.44) ; 6.2.10.
Tube rond en N + M : critère linéaire (6.2), prudent.

**Ce que le moteur a tranché** (`acier/barre/ec3_section.py`, lot 2) :
- tube rond : la RÉSULTANTE des deux moments et celle des deux efforts tranchants (il n'a pas d'axe) ; n + M/M_pl,Rd ≤ 1 ;
- effort tranchant élevé (V_Ed > 0,5·V_pl,Rd) : (6.30) pour un I de classe 1 ou 2 fléchi autour de l'axe fort ; partout
  ailleurs, et dès qu'un effort normal s'ajoute (§ 6.2.10), (1 − ρ) sur TOUTE la section — du côté sûr, et c'est dit ;
- le plancher η·h_w·t_w de l'aire de cisaillement gouverne pour 58 profilés du catalogue (IPE 750, HL, HE 900 et 1000) ;
- h_w/t_w > 72·ε/η : le voilement par cisaillement relève du lot 4 — en attendant, la vérification « E-Vb » est rendue
  « non couverte », jamais « ok » (11 profilés en S235, 46 en S355, 93 en S460 — tous des tubes minces en S235) ;
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
432,72 kN) : N_b,z,Rd = 569,2 kN, taux 76 % (la méthode historique : 112 %).

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
- le couple (λ̄_LT,0 ; β) : (0,4 ; 0,75) pour un tronçon nu, (0,2 ; 1,0) dès qu'un maintien d'une seule semelle entre
  dans M_cr — **à confirmer**, et le résultat le dit (`grandeurs["a_confirmer"]`).

### 5.7 Flexion composée avec instabilité — § 6.3.3, annexe A (vérifié ; E2)
(6.61) et (6.62) ; méthode 1. Tableau A.2 : moment linéaire C_mi,0 = 0,79 + 0,21ψ + 0,36·(ψ − 0,33)·N_Ed/N_cr,i ;
charge répartie **1 + 0,03**·N_Ed/N_cr,i ; charge concentrée **1 − 0,18**·N_Ed/N_cr,i ; cas général
1 + (π²·E·I_i·|δ_x|/(L²·|M_i,Ed(x)|) − 1)·N_Ed/N_cr,i.

**Ce que le moteur fait** (`acier/barre/ec3_interaction.py`, lot 3) : (6.61) et (6.62) avec tous les facteurs de
l'annexe A (μ, w ≤ 1,5, n_pl, a_LT, C_my, C_mz, C_mLT, b_LT à e_LT, C_yy à C_zz et leurs planchers, k_yy à k_zz ;
facteurs élastiques en classes 3 et 4) ; C₁ = 1/k_c² ; une barre sans déversement possible a λ̄_0 = 0 ; une flexion
déviée SANS effort normal passe par les mêmes expressions ; une barre tendue est prise sans effort normal (du côté
sûr). La borne inférieure de C_mi,0 (supposée 1 − N_Ed/N_cr,i) est active par défaut et **change un résultat** : sur
l'exemple 5 de l'ECCS, C_mz,0 passe de 0,771 à 0,842 — à confirmer (Q1).

**Le benchmark 21 de l'aide de SCIA Engineer** (ECCS N°119, exemple 5 : IPE 500 S235, 3,75 m, N_Ed = 500 kN, flexion
déviée). Sa page n'est pas joignable d'ici ; la recherche en a rendu les données et une partie des résultats. Sont
reproduits : classe 1, N_c,Rd 2 715 kN, M_c,y,Rd 516, M_c,z,Rd 78,9 kN·m, V_pl,z,Rd 814 kN (0,3 %), V_pl,y,Rd 912 kN
(1 %) ; μ_y 1,00, w_y 1,138, μ_z 0,937, w_z 1,5, C_mz,0 0,771, M_cr,0 899 kN·m, λ̄_0 0,757, a_LT 0,998 (0,2 %). Les
facteurs qui dépendent du diagramme de M_y (b_LT à e_LT, C_yy…) attendent la page elle-même (Q8) ; en attendant,
l'annexe A est recalculée d'une traite dans le test, pour neuf barres, à 10⁻⁹.

### 5.8 Les paramètres nationaux (une seule table dans le moteur : `acier/barre/parametres.py`)

| Paramètre | Valeur | Source | Statut |
|---|---|---|---|
| γ_M0 ; γ_M1 ; γ_M2 | 1,00 ; 1,00 ; 1,25 | valeurs EN, reprises par l'ANB | à pointer sur l'ANB |
| λ̄_LT,0 ; β (§ 6.3.2.3), M_cr calculé en IGNORANT les maintiens latéraux | 0,4 ; 0,75 | ANB:2018 d'après l'aide de STAAD.Pro (seule source trouvée ; la page n'a pas pu être relue d'ici) | **À CONFIRMER** |
| λ̄_LT,0 ; β, M_cr calculé AVEC les maintiens (section brute) | 0,2 ; 1,0 | idem | **À CONFIRMER** |
| Facteur f (§ 6.3.2.3(2)) | formule EN | brief | à pointer |
| Méthode de § 6.3.3 | méthode 1 (annexe A) | ANB d'après l'aide de STAAD.Pro | **À CONFIRMER** |
| Borne inférieure de C_mi,0 (tableau A.2) | 1 − N_Ed/N_cr,i (supposée, comme l'annexe française) | l'aide de STAAD.Pro dit qu'il y en a une, sans la donner | **À CONFIRMER** |
| η (EN 1993-1-5 § 5.1) | 1,20 jusqu'au S460 | valeur EN | à pointer |
| Combinaison ELU | (6.10) : 1,35·G + 1,5·Q₁ + Σ 1,5·ψ₀·Q_i | NBN EN 1990 ANB | à pointer |
| ψ₀ / ψ₁ / ψ₂ | A, B : 0,7/0,5/0,3 ; C, D, F : 0,7/0,7/0,6 ; E : 1,0/0,9/0,8 ; G : 0,7/0,5/0,3 ; H : 0/0/0 | NBN EN 1990 ANB (brief) | à pointer |
| ψ de la neige, du vent, de la température | 0,5/0,2/0 ; 0,6/0,2/0 ; 0,6/0,5/0 (valeurs EN) | — | **À CONFIRMER** |
| Combinaison de la méthode historique | G + ΣQ, tout à 100 % | pratique du bureau | **À CONFIRMER** (Q4) |

« À pointer » : la valeur est celle de l'EN, connue ; il reste à la lire sur le texte belge. « À CONFIRMER » : la valeur
change un résultat et sa source n'est pas le texte — elle est imprimée comme telle dans la note.

**Les combinaisons, telles que le moteur les écrit** (`acier/barre/combinaisons.py`, lot 5) : pour chaque ARRANGEMENT
des cas variables (chacun présent ou absent — c'est la recherche de la disposition défavorable, travée par travée) et
chaque nature dominante parmi les présents, γ_G·ΣG + γ_Q·Q_dominante + Σ γ_Q·ψ₀·Q_autres, avec γ_G = 1,35 puis 1,00
(G favorable). Les cas d'une MÊME nature sont une seule action disposée librement : ils reçoivent tous γ_Q quand leur
nature domine. Au plus quatre cas variables. Méthode historique : G + ΣQ à 100 % pour chaque arrangement.

### 5.9 EN 1993-1-5 : âme et raidisseurs (vérifié)
Voilement par cisaillement (§ 5 : à vérifier si h_w/t_w > 72ε/η ; V_bw,Rd, χ_w du tableau 5.1, V_bf,Rd ; k_τ = 5,34 +
4·(h_w/a)² si a/h_w ≥ 1, sinon 4 + 5,34·(h_w/a)²). Âme sous charge concentrée (§ 6 : types (a), (b), (c) ; k_F ;
F_cr = 0,9·k_F·E·t_w³/h_w ; m₁, m₂ ; l_y ; l_e = k_F·E·t_w²/(2·f_yw·h_w) ≤ s_s + c — le facteur 2 est celui du texte ;
λ̄_F ; χ_F = 0,5/λ̄_F ≤ 1 ; F_Rd). Interactions (§ 7.1, § 7.2 : η₂ + 0,8·η₁ ≤ 1,4, η₁ élastique). Voilement induit par
la semelle (§ 8). Raidisseurs (§ 9 : section efficace plats + 15·ε·t_w de chaque côté ; flambement sur 0,75·h_w,
courbe c ; I_t/I_p ≥ 5,3·f_y/E ; raidisseurs intermédiaires I_st, N_st). La DÉCISION « raidisseur oui / non » se prend à
chaque appui et sous chaque charge ponctuelle.

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
- R3 (W_pl,y 628,4 cm³, S235, f = 1) : au centre M_cr 118,18 kN·m, λ̄_LT 1,1179 → (0,4 ; 0,75) χ_LT 0,6278, taux 0,741 ;
  (0,2 ; 1,0) χ_LT 0,5246, taux 0,887 ; semelle haute M_cr 86,98, λ̄_LT 1,3030 → 0,5220, 0,891 ; 0,4254, 1,094.
- R4 (h_w 278,6 ; m₁ 21,13) : (a) s_s 50 → F_Rd 283,2 kN ; 100 → 387,8 ; 150 → 425,1 ; (c) s_s 100, c = 0 → 261,9 ;
  c = 50 → 335,9 ; A_v 2568 mm², V_pl,Rd 348,4 kN.

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
| `resultat.py` | la forme commune de ce que rend une méthode (vérifications, blocs de formules, alertes, synthèse) |
| `ec3_classe.py`, `ec3_section.py`, `ec3_flambement.py`, `mcr.py`, `ec3_deversement.py`, `ec3_interaction.py` | EN 1993-1-1 |
| `ec3.py` | la barre vérifiée selon l'Eurocode 3 : la lecture, puis la section, le flambement, le déversement, l'annexe A |
| `ec3_ame.py`, `raidisseurs.py` | EN 1993-1-5 |
| `combinaisons.py`, `statique.py` | les cas de charge, les combinaisons, les efforts le long de la barre (sur `rdm/poutres.py`) |
| `entree.py`, `moteur.py` | la barre, sa relecture ; les deux méthodes, la synthèse |
| `formules.py` | les gabarits des formules de la note |

Chaque vérification rend `{id, methode, titre, reference, formule, termes, valeur, limite, taux, statut, unite,
hypotheses, messages}` ; le moteur calcule et nomme, l'écran et la note composent et formatent.

**La barre chargée** (`statique.py`, `entree.py`, `moteur.py`, lot 5). Deux modes : « efforts » (les classeurs : les
efforts de service vont à la méthode historique, ceux de calcul à l'Eurocode) et « chargee » — appuis (deux appuis,
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
- Méthode historique : dans chaque section, avec la longueur de déversement de son tronçon — un maintien d'une seule
  semelle ne compte que si elle est comprimée sur tout le tronçon.
- Hypothèses dites dans la note : les appuis tiennent le déplacement latéral et la torsion (fourches) ; une console est
  encastrée à sa racine et flambe sur deux fois sa longueur ; l'axe faible flambe sur le plus long tronçon.
- Une méthode qui REFUSE la barre (semelle de plus de 80 mm, hors du tableau 3.1) est rendue « non couverte » avec son
  motif ; l'autre reste calculée.
- Hors de la barre chargée (v1) : la flexion d'axe faible et les charges horizontales (le mode « efforts » les
  prend) ; la flèche de service est donnée à titre d'information, sans verdict.
- R2 du brief, de bout en bout depuis les charges : M_Ed 68,70 kN·m, V_Ed 27,80 kN, déversement 89,5 % (Eurocode 3) —
  et **184 % par la méthode historique** : le déversement de la doc, sur 6 m sans maintien, est très sévère (Q4).

**Intégration** : dans « Choix de profilé » (demande du propriétaire). Le module garde ce qu'il sait faire — comparer
des profilés et retenir le plus léger qui tient — et gagne la barre : ses appuis, ses charges, ses maintiens, ses
raidisseurs ; ses deux méthodes ; sa note. La copie de l'ancien écran (`acier/choix_profil`) est remplacée quand le
nouvel écran la couvre ; le module « Flambement » reste en l'état (`AMELIORATIONS_PAR_MODULE.md`, A1).

| Lot | Contenu | Preuve | État |
|---|---|---|---|
| 0 | deuxième avis, inventaire, plan (ce document) | § 2 | **fait** (14/11/2026) |
| 1 | méthode historique, efforts imposés (`section.py`, `historique.py`, `formules.py`) | `tests/test_barre_historique.py` : 83 contrôles — les 108 valeurs du tableau, les six classeurs à 10⁻⁹, les cinq exemples de la doc, chaque correction, chaque formule recalculée par son gabarit, les 1 419 profilés du catalogue, 29 mutations | **fait** (14/11/2026) ; pas encore à l'écran |
| 2 | matériau, classes, section efficace, résistances de section (`parametres.py`, `resultat.py`, `ec3_classe.py`, `ec3_section.py`) | `tests/test_barre_ec3_section.py` : 128 contrôles — les limites du tableau 5.2 et les k_σ du texte ; les classes de profilés connus ; classe et A_eff du module Flambement sur tout le catalogue (4 nuances) ; W_eff recalculé par assemblage à 10⁻⁹ ; résistances et interactions à la main ; 376 formules recalculées par leur gabarit ; 27 384 calculs sur le catalogue ; 52 mutations | **fait** (14/11/2026) ; pas encore à l'écran |
| 3 | flambement, torsion, déversement (M_cr numérique), annexe A (`mcr.py`, `ec3_flambement.py`, `ec3_deversement.py`, `ec3_interaction.py`, `ec3.py`) | `tests/test_barre_ec3_barre.py` : 127 contrôles — les 20 valeurs de χ ; les charges de flambement du module Flambement sur tout le catalogue ; M_cr : formule fermée, R1 (sept valeurs), C₁ du moment linéaire, console, maintiens d'une semelle ; R3 avec les deux couples ; l'annexe A recalculée d'une traite pour neuf barres à 10⁻⁹ ; le benchmark 21 (douze valeurs) ; 694 formules recalculées par leur gabarit ; 1 044 barres du catalogue ; 73 mutations | **fait** (14/11/2026) ; pas encore à l'écran |
| 4 | EN 1993-1-5 : âme, charges concentrées, raidisseurs | R4 | à faire |
| 5 | barre chargée : appuis, charges, combinaisons, sections critiques, tronçons (`combinaisons.py`, `statique.py`, `entree.py`, `moteur.py`) | `tests/test_barre_moteur.py` : 82 contrôles — (6.10) à la main ; la somme pondérée des cas = le formulaire résolu d'un coup ; R2 de bout en bout ; la disposition défavorable d'une poutre continue ; les efforts d'une section pris ensemble ; console, maintiens d'une semelle ; les deux méthodes = leurs appels directs ; 318 formules ; 40 mutations | **fait** (14/11/2026) ; pas encore à l'écran |
| 6 | écran (trois formats) et note (historique, Eurocode 3, les deux) | harnais, photos | à faire |
| 7 | revue : comparaison des méthodes, « À CONFIRMER » restants | — | à faire |

## 10. À confirmer par le bureau

| # | Question | En attendant |
|---|---|---|
| Q1 | ANB de l'EN 1993-1-1 : les deux couples (λ̄_LT,0 ; β) et leur condition exacte ; la méthode 1 imposée ; la borne de C_mi,0 | § 5.8, imprimé « à confirmer » |
| Q2 | ANB de l'EN 1990 : ψ de la neige, du vent, de la température | valeurs EN |
| Q3 | f_u du S355 (510 ou 490) ; A_v,y des I laminés | 510 ; A − h_w·t_w |
| Q4 | Méthode historique : combinaison de service à 100 % ; tolérance de 5 % (non) ; C = 1 par défaut ; **déversement de la doc activé par défaut** (sévère pour un H non maintenu, § 4.5) | oui, oui, oui |
| Q5 | Tube Ø610 : fini à chaud (courbe a) ou formé à froid et soudé (c) ; horizontal ou vertical (poids propre) | a ; non compté |
| Q6 | Seuil du statut « à la limite » dans la note | celui des notes existantes (95 %) |
| Q7 | Les fichiers d'origine (classeurs, documentation) : les garder hors du dépôt public | hors du dépôt |
| Q8 | Les pages « Benchmarks 17 à 22 » de l'aide de SCIA Engineer (ECCS N°119), en pièce jointe (captures ou PDF) : elles ne sont pas joignables d'ici | douze valeurs du benchmark 21 reproduites ; l'annexe A recalculée d'une traite dans le test |

# Soudures — étude de conception avant développement

*Écrite le 04/11/2026 à la demande du bureau : « quelque chose de très ergonomique qui
vérifie toutes les différentes soudures qu'on peut faire, avec un switch entre l'Eurocode
et la méthode ELS du bureau, des schémas où l'on comprend dans quel sens vont les efforts,
puis des feuilles de rapport très bien faites ».*

## 0 · La décision en une page

**Un seul moteur, tous les cas.** Une soudure d'angle ne sait rien du cas d'emploi : elle
reçoit une force par unité de longueur et la porte sur sa gorge. Tout ce qui change d'un cas
à l'autre, c'est la GÉOMÉTRIE du groupe de cordons et la façon dont les efforts de
l'assemblage se répartissent dessus. Le module est donc bâti sur un **groupe de cordons en
trois dimensions** (chaque cordon : deux points, une gorge, la direction de ses deux ailes),
une **répartition élastique** des six composantes d'effort (N, V_y, V_z, M_x, M_y, M_z)
sur ce groupe, et la **décomposition exacte** de la force obtenue en σ⊥, τ⊥, τ∥ sur la
gorge — la géométrie du cordon décide des signes, pas une règle de pouce. Les
configurations courantes (recouvrement, T, console, gousset, platine d'about de profilé,
tube sur platine, bout à bout) ne sont que des **constructeurs** de ce groupe à partir de
quelques dimensions ; un cas « cordons libres » reçoit n'importe quelle géométrie.

**Deux méthodes, un switch.** « Eurocode » = EN 1993-1-8 §4.5.3, efforts de calcul (ELU),
méthode directionnelle par défaut (simplifiée en option) avec γM2 = 1,25 et βw de la nuance.
« Méthode ELS (bureau) » = efforts de service sans coefficient, résultante par unité de
longueur comparée à k·a avec **k = 100 kg par mm de gorge et par cm de cordon** (défaut du
bureau), et, à côté, la valeur que l'Eurocode donnerait au même niveau de charge :
f_vw,d/1,5 = **138,6 N/mm² pour le S235** (141 kg/(mm·cm) avec 1 kg = 9,81 N, 139 avec
1 kg = 10 N), 174,5 pour le S355. Le bureau voit d'un coup d'œil la marge que prend sa règle.

**Des schémas qui montrent le sens des efforts.** Chaque cas est dessiné par le même code
qui calcule (source unique) : les pièces en gris, les cordons en orange, les efforts en
rouge avec leur point d'application et leur sens, un encart « la gorge » qui montre σ⊥,
τ⊥, τ∥ au point le plus chargé, et une carte des taux le long des cordons avec le point
gouvernant.

**Une note de deux pages**, comme les autres modules (kit `ndc_pdf`) : calcul, dessins,
tableaux, conclusion ; méthode et détail par cordon.

## 1 · Ce que l'application a déjà, et ce que le module réutilise

| Existant | Réemploi |
|---|---|
| `acier/resistances.py` : `fvw_d(fu, βw, γM2)`, `fu_bw_cordon` (acier de référence du cordon = la pièce la plus faible) | la méthode simplifiée et le βw du cordon |
| `acier/donnees/bibliotheques.json` : nuances S235 … S460 (fy, fu, βw) | source unique des nuances |
| `acier/donnees/profiles.json` (catalogue) | dimensions des profilés pour la platine d'about, le tube sur platine |
| kit `ndc_pdf` (palette Encre, `Doc`, `Frame`, `chip`, `draw_text`) et `plancher/charges/peintre.py` (SVG → reportlab) | la note PDF |
| `modules/navigation.py`, charte des écrans (barre d'outils, onglets à libellé fixe, carte résultat, clés préfixées) | l'écran |
| `plancher/charges` comme patron (état `cas` multiples, JSON, harnais, captures) | l'architecture |

## 2 · Le modèle mécanique

### 2.1 Le cordon

Un cordon d'angle est un segment P₁P₂ (mm) de gorge a, posé dans le dièdre de deux faces
perpendiculaires. Il porte la direction de son axe **e_a**, et les directions **u_A**, **u_B**
de ses deux ailes (de la racine vers chaque face, perpendiculaires à e_a et entre elles).
Le plan de gorge passe par la racine et coupe l'angle droit en deux : il contient l'axe et la
bissectrice **(u_A + u_B)/√2**. Sa normale est donc **n_t = (u_A − u_B)/√2**, et la direction
dans le plan de gorge perpendiculaire à l'axe est **t_⊥ = (u_A + u_B)/√2**.

Une force **f** par unité de longueur (N/mm) transmise par le cordon donne sur sa gorge,
d'aire a par unité de longueur :

- σ⊥ = (f · n_t)/a ; τ⊥ = (f · t_⊥)/a ; τ∥ = (f · e_a)/a.

C'est la décomposition de la figure 4.5 de l'EN 1993-1-8, sans règle de pouce : une force
normale au plan de joint (arrachement d'un T) et une force dans le plan de joint
perpendiculaire au cordon (cisaillement frontal) donnent toutes deux |σ⊥| = |τ⊥| = f/(a√2),
mais leurs signes se combinent selon la position du cordon — ce qu'un calcul « à la
résultante » ne voit pas.

### 2.2 Le groupe

Les cordons d'un assemblage forment un groupe, traité comme des lignes d'aire a·dl
(méthode élastique classique) :

- A_w = Σ a·l ; centre de gravité G des gorges ; I_x, I_y, I_xy, J = I_x + I_y (par rapport à
  G, axes du plan de joint) ;
- les efforts (N normal au plan de joint, V_x, V_y dans le plan, M_x, M_y de flexion,
  M_z de torsion) sont donnés en un point de référence et transportés en G ;
- force par unité de longueur en un point P d'un cordon de gorge a :
  **f = a·[ (V_x, V_y, N)/A_w + (M_x·(y − y_G) − M_y·(x − x_G))/I … ·e_z + M_z/J·(e_z × GP) ]**
  (flexion : contrainte normale linéaire ; torsion : contrainte tangentielle
  proportionnelle au rayon, perpendiculaire au rayon) ;
- f est évaluée aux deux extrémités et au milieu de chaque cordon (elle y varie
  linéairement : le maximum est à une extrémité).

Pour des gorges différentes dans un même groupe, les inerties sont pondérées par a et la
force par unité de longueur de chaque cordon est proportionnelle à sa gorge : un cordon
deux fois plus épais prend deux fois plus de force, à contrainte égale.

### 2.3 Les critères

**Eurocode, méthode directionnelle (§4.5.3.2)** : √(σ⊥² + 3·(τ⊥² + τ∥²)) ≤ fu/(βw·γM2) et
σ⊥ ≤ 0,9·fu/γM2, avec fu et βw de l'acier de référence du cordon (la pièce la plus faible,
Tab. 4.1 : 0,8 / 0,85 / 0,9 / 1,0 / 1,0 pour S235 / S275 / S355 / S420 / S460), γM2 = 1,25
(ANB belge).

**Eurocode, méthode simplifiée (§4.5.3.3)** : |f| ≤ F_w,Rd = f_vw,d·a, f_vw,d = fu/(√3·βw·γM2).
Toujours du côté sûr par rapport à la directionnelle (elle traite toute force comme du
cisaillement) ; proposée en option, parce que les bureaux la connaissent.

**Méthode ELS (bureau)** : efforts de service (coefficients 1), résultante |f| ≤ k·a avec
k en kg/(mm·cm), converti en N/mm² par k·g/10 (g = 9,81 N/kg par défaut ; 10 en option) :
100 kg/(mm·cm) ↔ 98,1 N/mm². À côté, la **valeur équivalente de l'Eurocode** au niveau de
service : f_vw,d/γ_F avec γ_F = 1,5 (rapport usuel ELU/ELS d'une charge de chantier) —
138,6 N/mm² pour le S235, 155,8 pour le S275, 174,5 pour le S355 — et un bouton « prendre la
valeur Eurocode ». La règle du bureau est donc **plus sûre de 28 %** sur du S235 ; sur du
S355 elle laisse 43 % de marge, ce que la page dit en clair.

**Bout à bout** (§4.7) : pleine pénétration → la résistance est celle de la pièce la plus
faible, aucun calcul de cordon (la page le dit et vérifie la pièce : σ = N/(b·t) ≤ fy/γM0) ;
pénétration partielle → cordon d'angle de gorge = profondeur de pénétration.

**Longueurs** : longueur efficace l_eff = l − 2a (§4.5.1, cordon sans retour d'angle ;
option désactivable si les retours sont exécutés) ; l_eff ≥ max(30 mm ; 6a) sinon le cordon
ne compte pas ; gorge a ≥ 3 mm ; assemblage long par recouvrement (§4.11) :
βLw,1 = 1,2 − 0,2·L_j/(150a) ≤ 1 quand L_j > 150a.

### 2.4 Exemple chiffré (console, § 4 cas C3)

Console en plat de 10 mm soudée sur un poteau par deux cordons verticaux de 200 mm, gorge
5 mm, S235 ; V_Ed = 60 kN à 150 mm de la face du poteau.

| Grandeur | Valeur |
|---|---|
| M = V·e | 9,0 kN·m |
| I_w = 2·a·l³/12 | 6,67·10⁶ mm⁴ |
| f_n en tête de cordon = M·(l/2)·a/I_w | 675 N/mm |
| f_a = V/(2l) | 150 N/mm |
| σ⊥ = τ⊥ = f_n/(a√2) ; τ∥ = f_a/a | 95,5 ; 95,5 ; 30,0 N/mm² |
| directionnelle : √(σ⊥² + 3(τ⊥² + τ∥²)) = 197,9 ≤ 360 | **η = 0,55** |
| simplifiée : |f| = 691,5 N/mm ≤ 207,8·5 = 1 039 N/mm | η = 0,67 |
| ELS bureau (V_ser = 40 kN) : 461 N/mm ≤ 98,1·5 = 490 N/mm | η = 0,94 |
| ELS avec la valeur Eurocode (138,6 N/mm²) : 461 ≤ 693 | η = 0,67 |

La même console est « à 55 % » selon l'Eurocode et « à 94 % » selon la règle du bureau :
c'est exactement ce que le switch doit rendre visible.

## 3 · Catalogue des configurations

| Code | Cas | Ce qu'on saisit | Groupe construit | Efforts |
|---|---|---|---|---|
| C1 | Recouvrement | largeurs, épaisseurs, recouvrement, cordons latéraux et frontaux (oui/non), a | 2 cordons latéraux (τ∥) ± 1 ou 2 frontaux (σ⊥ = τ⊥) | N dans le plan ± V transversal |
| C2 | Jonction en T | épaisseur de la pièce rapportée, longueur, un ou deux cordons, a | 1 ou 2 cordons le long de la jonction, ailes sur chaque pièce | N (arrachement), V_a (le long), V_p (en travers), M (flexion du T) |
| C3 | Console fléchie | hauteur du plat, épaisseur, excentricité, a | 2 cordons verticaux (ou 3 côtés) | V à l'excentricité e → M = V·e + V |
| C4 | Gousset en torsion | plat posé à plat, cordons sur 1 à 4 côtés, a | périmètre choisi | V dans le plan à l'excentricité → torsion + cisaillement |
| C5 | Platine d'about de profilé | profilé du catalogue (IPE, HE…), a_semelles, a_âme, cordons une ou deux faces | semelles (4 cordons) + âme (2) | N, V_z, M_y (+ V_y, M_z) |
| C6 | Tube sur platine | tube du catalogue (CHS, SHS, RHS), a, pourtour continu ou par n tronçons de l mm | périmètre (cercle discrétisé, rectangle à coins) ou ses tronçons (le premier centré en haut) | N, V, M, T |
| C7 | Bout à bout | largeur, épaisseur, pénétration (pleine / partielle p) | pleine : vérification de la pièce ; partielle : cordon de gorge p | N, V, M |
| C8 | Cordons libres | liste d'éléments : segments (x₁, y₁, x₂, y₂) ou arcs (centre, rayon, angles — 0 → 360 : cercle), gorge, côté de la pièce rapportée, continus ou par tronçons | tel quel (arcs en cordes de 15°) | les six composantes au point de référence |

**Soudure discontinue (04/11/2026).** Toutes les configurations acceptent « n tronçons de
l mm » sur leurs cordons droits et leurs pourtours : aux deux bouts et régulièrement entre
sur un cordon droit, régulièrement à partir du haut sur un pourtour ; chaque tronçon est
une chaîne ouverte raccourcie de a à chaque bout, soumise à l − 2a ≥ max(30 mm ; 6a) ;
l'espacement est rappelé avec la limite de l'EN 1993-1-8 §4.3.2.2 (≤ min(200 mm ; 12 t en
compression ; 16 t en traction)) et marqué « DÉPASSÉ » le cas échéant (`trace.py`).

Chaque constructeur rend aussi les éléments du dessin (pièces, cotes) : le dessin est une
vue du même objet.

## 4 · Les schémas

Principes, les mêmes pour tous les cas :

- **vue principale** adaptée au cas (élévation pour la console, coupe A-A et plan pour le
  T, vue de dessus pour le recouvrement, face pour le gousset, face et élévation pour C5 et
  C6, plan et coupe pour le bout à bout), en **style dessin technique** (demande du
  04/11/2026, le premier rendu « n'était pas assez technique ») : pièces au trait fort,
  sections hachurées (deux sens pour deux pièces, cordon coupé en orange hachuré), cotes
  fines à lignes d'attache et pointes, axes en trait mixte, trait de coupe sur le plan,
  arêtes cachées en interrompu, mini-vue de dessus des cordons sur la console, coupe de
  la soudure bout à bout ;
- **efforts en rouge** (`#9C3341`) : flèche droite pour une force, flèche courbe pour un
  moment, point d'application marqué, libellé avec la valeur ; le sens du dessin EST le
  sens positif de la saisie (une valeur négative retourne la flèche) ;
- **encart « la gorge »** : la section du cordon le plus chargé avec sa gorge a, et les trois
  composantes σ⊥ (normale à la gorge), τ⊥ (dans la gorge, en travers), τ∥ (dans la gorge,
  le long) avec leurs valeurs — c'est la figure 4.5 de l'EN 1993-1-8 appliquée au cas ;
- **carte des taux** : chaque cordon coloré par son taux (vert → ocre → rouge), le point
  gouvernant marqué ●, les extrémités annotées ;
- une **légende** d'une ligne sous le dessin.

Le dessin est un SVG en millimètres (même convention que le plancher), rendu à l'écran et
peint dans la note par `peintre.py`.

## 5 · L'écran

Barre d'outils : 🏠 Accueil · [cas ▾] · ＋ Cas · 📋 · 🗑️ · 💾 Enregistrer · 📂 Ouvrir ·
📄 Note. Sous la barre, le **switch** en segmented control, bien visible :
**« Eurocode (ELU) » | « Méthode ELS (bureau) »** — il change le libellé des efforts
(« Efforts de calcul ELU » / « Efforts de service »), la résistance affichée et le texte de
la carte résultat.

Colonne gauche : le dessin (vue principale + encart + carte des taux), les vues au choix
(Élévation / Plan / Gorge). Colonne droite : la carte **RÉSULTAT** (pastille, η max, cordon
et point gouvernants, critère, méthode, phrase d'une ligne), puis cinq onglets à libellé
fixe : **1 · Configuration** (cas, pièces, nuances, profilé du catalogue), **2 · Cordons**
(gorges, longueurs, longueur efficace, retours, βLw), **3 · Efforts** (six composantes,
point de référence, sens rappelé par le dessin), **4 · Méthode** (directionnelle /
simplifiée, γM2, k du bureau et son équivalent Eurocode, g, γ_F), **5 · Détail** (tableau
par cordon et par point : f, σ⊥, τ⊥, τ∥, σ_c, η). En bas, « Méthode et sources ».

Trois formats (R1) ; clés `sdr_` ; cas multiples ; fichier JSON ; AppTest ; captures.

## 6 · La note (feuille de rapport)

Deux pages A4 paysage, kit `ndc_pdf` : page 1 CALCUL — bandeau (projet, cas, méthode
choisie, pastille, η max), dessin principal et encart de gorge peints depuis le SVG,
hypothèses (méthode, acier de référence, coefficients, longueur efficace), tableau des
cordons (position, longueur, gorge, l_eff), tableau des vérifications (par cordon : f, σ⊥,
τ⊥, τ∥, σ_c, résistance, η), conclusion ; page 2 MÉTHODE ET DÉTAIL — les formules de la
méthode employée, la comparaison Eurocode / bureau, la carte des taux, les références.

## 7 · Architecture et lots

```
acier/soudures/
  cordon.py        Cordon, décomposition sur la gorge (§2.1)
  groupe.py        Groupe : centre, inerties, répartition élastique des six composantes (§2.2)
  criteres.py      Eurocode directionnelle / simplifiée, ELS bureau, bout à bout, longueurs (§2.3)
  configurations.py  C1 … C8 : dimensions → groupe + éléments du dessin
  etat.py          cas, défauts, JSON, construction depuis l'état
  dessin.py        SVG : vues, efforts, encart, carte des taux
  methode.py       textes de méthode (écran et note), références
  note.py          la note PDF
modules/soudures.py          l'écran
tests/test_soudures_moteur.py, test_soudures_ecran.py, test_soudures_note.py
tools/captures_soudures.js
docs/acier/SOUDURES.md       document de référence
```

| Lot | Contenu | Preuve |
|---|---|---|
| SW1 · moteur | cordon, groupe, critères, configurations, état | tests analytiques : recouvrement (τ∥ = N/(2·a·l)), frontal (σ⊥ = τ⊥ = N/(a·l·√2)), T en traction, console (3M/l²), gousset (polaire), platine (semelles = M/h), simplifiée ≥ directionnelle, conversions k |
| SW2 · écran | dessins techniques, écran, switch, cas, JSON | harnais AppTest (50), captures trois formats (`tools/captures_soudures.js`), images |
| SW2a · tronçons | soudures discontinues, arcs, éléments libres (`trace.py`) | tests analytiques (78 au total) |
| SW2b · liaison | `modules/liaison.py` : les widgets suivent l'état (Soudures et Charges sur plancher) | contrôle navigateur dans les captures (état « tube »), harnais des deux écrans |
| SW3 · note | note deux pages, peintre SVG partagé (`ndc_pdf/svg.py`), doc `docs/acier/SOUDURES.md` | tests PDF (43), batterie, commit, push |

## 8 · Points à trancher par le bureau (défauts retenus en attendant)

| | Question | Défaut retenu |
|---|---|---|
| P-1 | k de la méthode ELS | 100 kg/(mm·cm), modifiable ; l'équivalent Eurocode affiché à côté |
| P-2 | conversion kg → N | g = 9,81 N/kg ; option 10 (la pratique qui donne « 138 ») |
| P-3 | rapport ELU/ELS pour l'équivalent Eurocode | γ_F = 1,5, modifiable |
| P-4 | γM2 | 1,25 (ANB belge), modifiable dans Méthode |
| P-5 | longueur efficace | l − 2a par défaut (§4.5.1) ; « retours d'angle exécutés » rend l entière |
| P-6 | méthode Eurocode par défaut | directionnelle (la plus juste) ; simplifiée en option |
| P-7 | la méthode ELS compare la résultante (comme la simplifiée) | oui — c'est la règle du bureau ; la directionnelle reste lisible dans Détail |
| P-8 | fatigue, aciers > S460, soudures sur acier inoxydable | hors V1, dits dans « ce que l'outil ne fait pas » — les cordons discontinus, d'abord exclus, sont entrés le 04/11/2026 à la demande du bureau (tronçons sur toutes les configurations) |
| P-9 | position des tronçons | aux deux bouts et régulièrement entre sur un cordon droit (les bouts doivent être soudés) ; régulièrement à partir du haut sur un pourtour ; l'espacement §4.3.2.2 est rappelé, pas imposé |
| P-10 | l'écran garde-t-il la valeur du navigateur ou celle de l'état ? | celle de l'ÉTAT, toujours : `modules/liaison.py` réécrit les widgets depuis l'état à chaque changement hors widget (défaut trouvé par les captures : C3 → C6 affichait l'ancienne gorge) |

## 9 · Sources

EN 1993-1-8:2005 §4 (4.3 cordons, 4.5.1 longueur efficace, 4.5.3.2 méthode directionnelle,
4.5.3.3 méthode simplifiée, 4.7 soudures bout à bout, 4.11 assemblages longs, Tab. 4.1 βw) ;
NBN EN 1993-1-8 ANB (γM2 = 1,25) ; pratique du bureau (k = 100 kg/(mm·cm)) ; méthode
élastique des groupes de cordons (traités classiques d'assemblages : répartition linéaire
en flexion, polaire en torsion).

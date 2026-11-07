# Moteur « Choix de profilé » — `acier/choix_profil`

Le calcul de l'écran Streamlit **`modules/choix_profile.py`** (« Choix de profilé métallique optimisé (EC3) »), extrait
sans Streamlit ni pandas pour tourner dans le navigateur (site « Stab Méthodes », module `profil`, établi
`choix-profile`). L'écran Streamlit reste la **référence** et le secours : il n'a pas été modifié.

> **Défauts NON CONSERVATIFS de l'écran Streamlit** — reproduits pour la parité, DITS en rouge sur le site, détaillés
> au §7 :
> **D1** V<sub>y,Ed</sub> n'est jamais vérifié (A<sub>v,y</sub> absent de la forme de catalogue lue : η<sub>Vy</sub> = 0) ;
> **D2** une traction se RETRANCHE de la flexion (η<sub>N</sub> sans valeur absolue) ;
> **D3** f<sub>y</sub> n'est pas réduit au-delà de 40 mm d'épaisseur (39 HD/HL : +9,3 % en S235, +6 % en S355 ; au-delà
> de 80 mm, hors du tableau 3.1) ;
> **D4** pas d'interaction effort tranchant – moment (§6.2.8) ;
> **D5** pas de classement de section : 28 âmes de classe 4 en compression dès le S235 (67 en S355) et, en S355, trois
> HP à semelles de classe 4 — là, même en flexion simple, W<sub>el</sub> surestime la résistance ;
> **D6** le flambement ignore le moment (pas d'interaction §6.3.3) et une seule courbe sert aux deux axes ;
> **D7** le déversement n'est pas calculé (la case et L<sub>LT</sub> ne font rien).

| Fichier | Rôle |
|---|---|
| `acier/choix_profil/calcul.py` | les fonctions de l'écran **recopiées dans le même ordre d'opérations** : `load_profiles` (→ `nettoyer`, `profils`), `ec3_section_resistance` (→ `resistances_section`), `utilisation_ratio_section` (→ `taux_section`), `ec3_buckling_chi` (→ `flambement_chi`, `chi_courbe`), `fmt_no_trailing_zeros` (→ `fmt_sans_zeros`) |
| `acier/choix_profil/entree.py` | les listes et les valeurs par défaut des widgets de l'écran ; la normalisation STRICTE d'un élément |
| `acier/choix_profil/moteur.py` | le corps de `show()` : le tableau trié, l'optimal de l'écran, le profilé retenu et sa fiche, ses taux, ses formules, ses avertissements ; les candidats du sélecteur |
| `api_web/profil.py`, `api_web/dessin_profil.py`, `site/web/css/modules/profil.css` | le module du site (mise en forme seulement) et ses dessins |
| `tests/test_choix_profil_moteur.py` | la preuve de parité avec l'écran (faux `st`), les contrôles à la main, la campagne de mutations |
| `tests/test_api_web_profil.py` | ce que l'écran du site affiche = ce que le moteur calcule |

Bibliothèque standard seule ; numpy est chargé par le catalogue des profilés (`acier/profiles`, source unique), comme
pour tous les modules acier. Le test le prouve dans un sous-processus où `sys.meta_path` refuse streamlit, pandas,
matplotlib, scipy, reportlab, PIL et fitz, sous Python 3.12 et 3.13 (celui de Pyodide).

## 1. La méthode (ce que l'écran calcule)

Unités de l'écran : efforts en kN, moments en kN·m, f<sub>y</sub> en N/mm² ; aires en cm², modules en cm³, inerties en
cm⁴ ; longueurs de flambement en m.

**Les profilés.** `CAT.legacy` lit `acier/donnees/profiles.json` pour les familles I/H (IPE, IPE A, IPE O, IPN, HEA,
HEB, HEM, HEAA, HL, HD, HP : 275 profilés). L'écran relit chaque enregistrement champ par champ (`pick`, `to_float`) et
écarte un profilé sans W<sub>el,y</sub> > 0. Les familles proposées sont celles des profilés chargés, triées ; HEA est
cochée à l'ouverture ; **aucune famille cochée = toutes**. Le filtre « I<sub>y</sub> min » écarte les profilés
d'inertie inférieure (0 : sans filtre).

**Les résistances de section** (élastiques), pour chaque profilé :

    N_Rd  = A·10²·f_y / γ_M0              M_y,Rd = W_el,y·10³·f_y / γ_M0      M_z,Rd = W_el,z·10³·f_y / γ_M0
    V_z,Rd = A_v,z·10²·f_y / (√3·γ_M0)    V_y,Rd = A_v,y·10²·f_y / (√3·γ_M0)

f<sub>y</sub> = 235, 275 ou 355 N/mm² (`int(acier[1:])`), sans réduction pour l'épaisseur (défaut D3). Une
résistance dont la donnée manque vaut None et son taux vaut 0.

**Les taux.** η<sub>N</sub> = N<sub>Ed</sub>/N<sub>Rd</sub> (SANS valeur absolue, défaut D2) ;
η<sub>My</sub> = |M<sub>y,Ed</sub>|/M<sub>y,Rd</sub> ; η<sub>Mz</sub>, η<sub>Vy</sub>, η<sub>Vz</sub> de même ;
η<sub>NM</sub> = η<sub>N</sub> + η<sub>My</sub> + η<sub>Mz</sub> ; η<sub>section</sub> = max(η<sub>NM</sub> ;
η<sub>Vy</sub> ; η<sub>Vz</sub>). Un effort « optionnel » (N, M<sub>z</sub>, V<sub>y</sub>, T) vaut 0 quand sa case est
décochée.

**Le flambement** (case cochée ET N<sub>Ed</sub> > 0), pour chaque axe dont L<sub>cr</sub> > 0 :

    N_cr = π²·E·I·10⁴ / (L_cr·10³)²        λ̄ = √(A·10²·f_y / N_cr)        φ = 0,5·(1 + α·(λ̄ − 0,2) + λ̄²)
    χ = min(1 ; 1 / (φ + √max(φ² − λ̄² ; 0)))        N_b,Rd = χ·A·10²·f_y / γ_M1        η_b = N_Ed / min(N_b,Rd,y ; N_b,Rd,z)

E = 210 000 N/mm² (la valeur par défaut d'`ec3_buckling_chi`, jamais passée). Une seule courbe (α) pour les deux axes,
choisie par l'utilisateur (a0 0,13 · a 0,21 · b 0,34 par défaut · c 0,49 · d 0,76).

**Le taux d'un profilé** : η = max(η<sub>section</sub> ; η<sub>b</sub>) (η<sub>section</sub> seul sans flambement
calculé). Le tableau de l'écran porte `Utilisation [%]` = round(η·100, 3), et il est **trié par cette colonne
arrondie** (croissante).

**L'optimal de l'écran** (« 📌 Profilé optimal ») : `(100 − util[util ≤ 100]).idxmin()` — le profilé le PLUS CHARGÉ
dont l'utilisation ARRONDIE ne dépasse pas 100 %, le premier du tri à égalité. C'est l'index par défaut du menu
« Sélectionner un profilé » ; sans optimal (rien ne tient), le menu montre la première ligne (le moins chargé).

**La colonne de droite** : le profilé sélectionné, ses deux tableaux (dimensions h, b, t<sub>w</sub>, t<sub>f</sub>,
r ; propriétés poids, A, W<sub>el,y</sub>, W<sub>el,z</sub>, A<sub>v,y</sub>, A<sub>v,z</sub>, I<sub>y</sub>,
I<sub>z</sub>), les résistances et les taux en LaTeX, χ et λ̄ par axe, et les messages (données absentes,
déversement non calculé).

## 2. D'où vient chaque grandeur

| Grandeur | Source |
|---|---|
| h, b, t<sub>w</sub>, t<sub>f</sub>, r (r<sub>1</sub> des IPN), A, A<sub>v,z</sub>, W<sub>el,y</sub>, W<sub>el,z</sub>, I<sub>y</sub>, I<sub>z</sub>, masse | `acier/donnees/profiles.json` par `CAT.legacy` (la forme « ancienne » des enregistrements) |
| A<sub>v,y</sub> | **jamais** : la forme ancienne ne le porte pas (défaut D1) |
| f<sub>y</sub> | la nuance choisie, `int(acier[1:])` |
| E | 210 000 N/mm², valeur par défaut de la fonction de flambement |
| α | le menu des courbes de l'écran |
| γ<sub>M0</sub>, γ<sub>M1</sub> | saisis, 1,0 par défaut, bornés à [0,8 ; 1,5] par l'écran |

## 3. L'élément (entrée)

Un élément est un dictionnaire (unités de l'écran) : `nom`, `familles` (liste ; vide = toutes), `acier`, `My`, `Vz`,
`avec_N` et `N`, `avec_Mz` et `Mz`, `avec_Vy` et `Vy`, `avec_T` et `T`, `gamma_M0`, `gamma_M1`, `Iy_min`,
`flambement`, `Lcr_y`, `Lcr_z`, `courbe` (a0, a, b, c, d ; le libellé de l'écran « c (0.49) » est admis),
`deversement`, `L_LT`, `profil` (None : le profilé que l'écran retient, son optimal).

`entree.normaliser` complète une clé absente avec la valeur de l'écran et **refuse** (ValueError nommant le champ) ce
que l'écran n'accepterait pas : γ hors de [0,8 ; 1,5], I<sub>y</sub> min, L<sub>cr</sub> ou L<sub>LT</sub> négatifs,
famille, acier, courbe ou profilé inconnus, clé inconnue, texte ou booléen à la place d'un nombre, nombre infini ou NaN.
Les efforts peuvent être négatifs (l'écran l'accepte). Le site refuse de même, avec un message en français du bureau
(« L_cr,y ne peut pas être négatif… »), et l'élément reste intact.

**Le domaine du site** (`entree.hors_domaine`), plus étroit que celui de l'écran, qui ne borne ni les efforts ni les
longueurs par le haut : là où son calcul DÉBORDE, la saisie est refusée avec le nom du champ (règle 2 du contrat :
jamais ramenée en silence). Mesuré dans l'écran : M<sub>y</sub> = 10³⁰³ kN·m → OverflowError ; L<sub>cr</sub> = 10¹⁵⁰ m →
OverflowError ; L<sub>cr</sub> = 10⁻²⁰⁰ m → division par zéro (N<sub>cr</sub>) ; M<sub>y</sub> = 10³⁰⁰ → un taux de trois cents
chiffres ; I<sub>y</sub> min = 10³⁰⁸ → un texte de trois cents chiffres. Bornes : |effort| ≤ 10⁶ kN ou kN·m (seize fois
le N<sub>Rd</sub> du plus fort profilé, HL 920x1377 en S355 : 62 257 kN) ; L<sub>cr</sub>, L<sub>LT</sub> nuls ou de 1 mm à
1 000 m ; I<sub>y</sub> min ≤ 10⁸ cm⁴. Les bornes elles-mêmes se calculent (testé).

## 4. La sortie de `verifier(element)`

JSON strict (aucun NaN), nombres non arrondis sauf les colonnes arrondies de l'écran. Toutes les clés :

- `reference` (l'écran de référence), `element` (l'élément normalisé), `acier` {`nuance`, `fy`} ;
- `efforts` : `N`, `My`, `Mz`, `Vy`, `Vz`, `T` — ceux que le calcul reçoit (0 case décochée) ;
- `flambement` : `actif` (la case), `calcule` (case cochée et N > 0), `Lcr_y`, `Lcr_z`, `courbe`, `alpha` ;
  `deversement` : `actif`, `L_LT` (jamais utilisé) ;
- `familles_disponibles`, `familles` (cochées ; toutes si aucune), `Iy_min` ;
- `lignes` : le tableau de l'écran, trié comme lui ; chaque ligne : `nom`, `famille`, `masse`, `taux` (η exact),
  `taux_section`, `taux_flambement`, `utilisation` (round(η·100, 3)), `u_section`, `u_flamb`, `h`, `Wel_y`, `Wel_z`,
  `Avy`, `Avz`, `A`, `Iy`, `Iz`, `tient` (`utilisation` ≤ 100, le critère de l'écran) ;
- `n_profils`, `n_tiennent`, `optimal` (l'optimal de l'écran), `plus_leger` (AJOUT : le plus léger dont η ≤ 1) ;
- `retenu` (le profilé de la colonne de droite), `retenu_auto`, `choix_hors_tableau` ;
- `profil` : la fiche du retenu — `nom`, `famille`, `masse`, `dimensions`, `proprietes` (les deux tableaux de l'écran),
  `resistances` (`N_Rd`, `My_Rd`, `Mz_Rd`, `Vy_Rd`, `Vz_Rd`, en kN et kN·m), `ratios` (`rN`, `rMy`, `rMz`, `rVy`,
  `rVz`, `util_NM`, `util`), `axes` (par axe : `axe`, `Lcr`, `I`, `Ncr`, `lambdabar`, `phi`, `chi`, `Nb_Rd`,
  `Nb_Rd_N`), `Nb_Rd`, `taux_flambement`, `taux`, `formules` ;
- `verifications` : une par taux du retenu — `cle` (N, My, Mz, Vz, Vy, NM, section, flambement), `nom`, `symbole`,
  `valeur`, `limite`, `unite`, `taux`, `actif`, `ecran` (l'écran affiche ce taux), `resistance_absente`, `formules` ;
- `taux_max`, `statut` (ok, nok, vide), `gouvernant` {`cle`, `nom`, `taux`} : le premier maximum dans l'ordre de
  l'écran (NM, Vy, Vz), puis le flambement s'il dépasse strictement ;
- `avertissements` : [{`code`, `message`, `module`}] — `module` est le texte exact que l'écran affiche (None s'il se
  tait).

Chaque formule (`profil.formules`) : `cle`, `symbole`, `termes` [{`symbole`, `valeur`, `unite`}] dans l'ordre de la
formule, `resultat`, `unite`. Le site écrit « symbole = expression = valeurs = résultat » à partir de ces seuls
termes ; le test recalcule chaque résultat à partir d'eux (plus de 1 000 formules).

`candidats(element, familles)` : les lignes du sélecteur — les profilés des familles demandées (filtre I<sub>y</sub>
min compris) TRIÉS PAR MASSE, chacun avec `retenu` et `plus_leger`.

## 5. Ce que le moteur AJOUTE à l'écran (et nomme comme tel)

- `plus_leger` : le plus léger dont η ≤ 1 (exact) — l'étoile ★ du site. L'écran ne propose que son « optimal », le
  plus CHARGÉ qui tient, qui n'est pas forcément le plus léger (HEA + HEB, N = 600 kN, M<sub>y</sub> = 60 kN·m :
  l'écran retient HEB 180 à 99 %, 51,2 kg/m ; HEA 220 tient à 89 % pour 50,5 kg/m).
- `phi` (φ de chaque axe), `Nb_Rd` et le taux de flambement du retenu : l'écran ne montre que χ et λ̄ dans sa colonne
  de droite (le taux n'est que dans la colonne `u_flamb` du tableau).
- `gouvernant`, `verifications`, `formules` : la mise en note du calcul (aucun calcul nouveau).
- `avertissements` : ceux de l'écran (`module` porte son texte) et ceux que l'écran tait : `torsion_ignoree`,
  `Vy_Rd_absent` (non conservatif), `traction_retranchee` (non conservatif), `flambement_sans_compression`,
  `aucun_ne_tient` (« l'écran retient le moins chargé » quand rien n'est choisi ; « le moins chargé est … ; … est celui
  choisi » quand un profilé est choisi), `sans_effort`, `choix_hors_tableau`, `fy_epaisseur` (non conservatif, D3),
  `classe_4` (non conservatif, D5 : semelle c/t<sub>f</sub> > 14ε dès qu'un moment ou une compression est saisi ; âme
  c/t<sub>w</sub> > 42ε en compression pure dès que N<sub>Ed</sub> > 0 — EN 1993-1-1 tableau 5.2, contrôle simple et du côté
  sûr ; DIT, jamais recalculé : le taux reste celui de l'écran).

## 6. La preuve de parité (`tests/test_choix_profil_moteur.py`)

L'écran est **exécuté** : `show()` tourne avec un faux `st` qui enregistre chaque widget, chaque texte, chaque formule
LaTeX et chaque tableau au lieu de les afficher (le vrai pandas calcule le tableau et son style). Pour chaque cas
(75 : les défauts, chaque acier, chaque courbe, chaque famille seule, chaque avertissement, les bornes — I<sub>y</sub> égal
au filtre, efforts infimes arrondis à 0, efforts saisis puis décochés —, 24 tirages au sort à graine fixe, puis η = 1
exactement et l'arrondi à 100,000 %), le moteur et l'écran sont comparés :

1. le tableau : chaque colonne de chaque ligne, **au bit près** (44 135 comparaisons numériques, toutes identiques ;
   écart maximal 0), l'ordre des lignes, les couleurs des lignes (optimal, tient, ne tient pas) ;
2. le menu « Sélectionner un profilé » : ses options dans l'ordre et son index par défaut ;
3. la colonne de droite : le titre, les deux tableaux de la fiche, CHAQUE formule LaTeX (les nombres écrits par
   l'écran), chaque message (`st.info`, `st.warning`, `st.caption`), dans l'ordre ;
4. les taux exacts de chaque profilé, recalculés par les fonctions de l'écran, et les résistances, taux et χ du retenu.

S'y ajoutent : les fonctions recopiées appelées contre celles de l'écran sur des entrées construites (`load_profiles`
sur des enregistrements limites — W<sub>el</sub> nul ou illisible, « Poids » avant « masse », A<sub>v,y</sub> présent — ;
résistances, taux et flambement sur 3 000 tirages, bit pour bit) ; les défauts et les bornes de chaque widget relevés en
exécutant l'écran ; des contrôles à la main (HEA 200
S235 en N et mm ; HEB 200, L<sub>cr,z</sub> = 4 m, courbe c) ; les valeurs PUBLIÉES de χ à λ̄ = 1,0 (EN 1993-1-1 :
0,6656 · 0,5970 · 0,5399 · 0,4671) ; le sélecteur repassé par l'écran ; la couverture (49 situations attendues
réellement atteintes) ; la pureté (sans streamlit ni pandas, Python 3.12 et 3.13, résultats identiques).

**Mutations** (`--mutations`) : 43 altérations du moteur (et le témoin), chacune rejouée dans un arbre miroir (le dépôt n'est jamais
modifié) : chacune fait rougir le test, le témoin reste vert (dont les bornes du domaine du site et les limites de la
classe 4).

    /usr/bin/python3.12 tests/test_choix_profil_moteur.py
    /usr/bin/python3.12 tests/test_choix_profil_moteur.py --mutations

## 7. Les défauts de l'écran Streamlit — reproduits (parité) et ÉCRITS

Les non conservatifs d'abord.

- **D1 — V<sub>y,Ed</sub> n'est jamais vérifié (non conservatif).** L'écran lit A<sub>v,y</sub> dans la forme ancienne
  du catalogue, qui ne le porte pas : V<sub>y,Rd</sub> est absent et η<sub>Vy</sub> = 0 quel que soit V<sub>y,Ed</sub>.
  Le site montre la ligne « Cisaillement selon y · non vérifié » (sans taux) et un message rouge.
- **D2 — Une traction allège la flexion (non conservatif).** η<sub>N</sub> = N<sub>Ed</sub>/N<sub>Rd</sub> n'a pas de
  valeur absolue : N < 0 rend η<sub>N</sub> négatif et le RETRANCHE de η<sub>My</sub> + η<sub>Mz</sub>. Message rouge.
- **D3 — f<sub>y</sub> n'est pas réduit pour les parois épaisses (non conservatif).** L'écran prend 235/275/355 N/mm²
  quelle que soit l'épaisseur ; EN 1993-1-1 tableau 3.1 donne 215/255/335 N/mm² pour 40 < t ≤ 80 mm. 39 profilés
  (HD, HL) ont une paroi de plus de 40 mm : leurs résistances sont surestimées (+9,3 % en S235, +6 % en S355) ; au-delà
  de 80 mm (HL 920 lourds), hors du tableau. Avertissement `fy_epaisseur`, rouge sur le site.
- **D4 — Pas d'interaction effort tranchant – moment (non conservatif au-delà de V<sub>Ed</sub> > 0,5·V<sub>pl,Rd</sub>).**
  Le cisaillement est vérifié à part (max) ; EN 1993-1-1 §6.2.8 réduit alors la résistance en flexion.
- **D5 — Pas de classement de section (non conservatif).** Les résistances sont élastiques (W<sub>el</sub>, juste
  pour les classes 1 à 3), mais une paroi de classe 4 n'est pas réduite : 28 profilés I/H ont une âme de classe 4 en
  compression pure dès le S235 (c<sub>w</sub>/t<sub>w</sub> > 42ε : IPE 550, IPE 600, IPE 750, IPE A 360…), 41 en S275,
  67 en S355 ; et en S355, les semelles des HP 305x79, HP 360x109 et HP 400x122 sont de classe 4
  (c<sub>f</sub>/t<sub>f</sub> > 14ε = 11,4) : là, MÊME EN FLEXION SIMPLE, W<sub>el,y</sub> surestime M<sub>y,Rd</sub>
  (mesuré par la relecture du 07/11/2026 ; la première version de cette fiche disait la flexion du côté sûr).
  Avertissement `classe_4`, rouge sur le site.
- **D6 — Le flambement ignore le moment (non conservatif pour un poteau fléchi).** η = max(η<sub>section</sub> ;
  η<sub>b</sub>) : l'interaction flambement–flexion (§6.3.3, facteurs k) n'est pas faite. Une seule courbe pour les
  deux axes : à l'utilisateur de prendre la plus défavorable (EN 1993-1-1 tableau 6.2 en donne une par axe).
- **D7 — Le déversement n'est pas calculé (non conservatif pour une poutre non tenue).** La case et L<sub>LT</sub>
  existent, mais l'écran affiche seulement « LTB complet à activer… » ; L<sub>LT</sub> n'est pas utilisée.
- **D8 — T<sub>Ed</sub> est saisi mais n'entre dans aucun calcul** (avertissement `torsion_ignoree`).
- **D9 — L'« optimal » est le plus chargé, pas le plus léger**, et il est jugé sur l'utilisation ARRONDIE à 10⁻³ % :
  un profilé à η = 1,000004 s'écrit 100,000 % et « tient ». Le site garde ce choix par défaut (parité) mais marque le
  plus léger qui tient (★) et colore chaque taux par sa valeur exacte (rouge au-delà de 100 %), avec un message rouge
  quand l'arrondi fait tenir le retenu.
- **D10 — Le tableau est trié par utilisation croissante** : les profilés les plus lourds viennent en tête, loin de la
  décision. Le site le trie par masse (mêmes colonnes, mêmes valeurs).
- **D11 — Sans effort, tout « tient » à 0 %** et l'écran retient la première ligne (HEA 100) ; avertissement
  `sans_effort`.
- **D12 — La colonne de droite tait le taux de flambement** (seulement χ et λ̄) et donne V<sub>y,Rd</sub> sans
  substitution ; le site écrit toutes les formules développées.
- **D13 — Décocher un effort optionnel efface sa valeur** (Streamlit oublie le widget non affiché) ; le site la garde
  (elle n'entre pas dans le calcul tant que la case est décochée).

## 8. Le site face à l'écran Streamlit

| Écran Streamlit | Site |
|---|---|
| Types de profilés (multiselect, HEA par défaut, vide = tous) | section 3 « Profilés comparés » : onze cases ; aucune cochée = toutes, dit en clair |
| My,Ed, Vz,Ed, Acier | section 2 « Efforts (ELU) », section 1 (acier en segments, f<sub>y</sub> rappelé) |
| Ajouter N,Ed / Mz,Ed / Vy,Ed / T,Ed (cases et valeurs) | section 2, « Autres efforts » : une case, le champ apparaît |
| γM0, γM1 (0,8 à 1,5) | section 5 « Coefficients partiels » (bornes tenues, saisie hors bornes refusée) |
| Iy min. | section 3 |
| Flambement : case, Lcr,y, Lcr,z, courbe α | section 4 « Instabilités » |
| Déversement : case, L_LT, légende | section 4, avec l'aide « non calculé » et l'avertissement |
| Sélectionner un profilé (défaut : l'optimal) | ligne « Profilé retenu » du résultat → sélecteur trié par masse (✓ retenu, ★ plus léger qui tient, propriétés au survol, familles en puces) ; case « Suivre l'optimal » |
| 📌 Profilé optimal | « Optimal de l'écran », sous le profilé retenu, avec le plus léger qui tient |
| Afficher tous les profilés ✓/✗ (tableau coloré) | vue « Tableau » (mêmes colonnes — sauf A<sub>v,y</sub>, toujours vide, omise et dite —, mêmes couleurs, trié par masse) |
| Fiche : dimensions, propriétés | vue « Section » : la section à l'échelle (contour exact du catalogue, hachures, axes, cotes, repères) et les deux tableaux |
| Formules EC3 (N_Rd, My,Rd, Mz,Rd, Vz,Rd, Vy,Rd), utilisations η | fenêtre de calcul : tableau vérification / formule / valeur / limite / taux, puis le détail « symbole = expression = valeurs = résultat » ; les puissances de 10 sont écrites jusqu'à l'unité du résultat (« 65,25 × 10² × 235 / 1 × 10⁻³ = 1 533,5 kN » : l'écran écrit « … / 1.00 = 1533.5 kN », un résultat en N donné en kN) — chaque substitution évaluée telle qu'écrite retrouve son résultat (testé) |
| χ, λ̄ par axe | fenêtre de calcul (N<sub>cr</sub>, λ̄, φ, χ, N<sub>b,Rd</sub>, η<sub>b</sub>) et vue « Flambement » (les cinq courbes χ(λ̄), les points du retenu) |
| Messages (données absentes, LTB, aucun profilé) | messages du résultat (rouges quand non conservatif) ; aucun profilé : la toile le dit et dit comment rouvrir les filtres |
| — | plusieurs éléments par note ; vue « Taux / masse » (chaque profilé de la sélection, retenu et plus léger marqués) |

**Hors périmètre** : la note PDF (le bouton est grisé sur tout le site). **Non couvert** : rien de l'écran Streamlit.
Les vues « Tableau » et « Taux / masse » ne se cliquent pas pour choisir un profilé (le choix passe par le sélecteur).

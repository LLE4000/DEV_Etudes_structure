# Poutres béton armé — le ferraillage détaillé et les dispositions constructives (site, 11/2026)

Ce document décrit ce que le site ajoute au calcul de la méthode du bureau (`beton/poutre`, extrait de l'écran
Streamlit `modules/poutre.py` v2.40, parité prouvée par `tests/test_poutre_moteur.py`) : le **ferraillage détaillé**
(lits positionnés, répartition horizontale, cadres multiples, armatures latérales) et le **contrôle des dispositions
constructives**. Le calcul de flexion et d'effort tranchant de la méthode n'est pas modifié : une poutre sans aucune des
clés ci-dessous est calculée **à l'octet près** comme avant.

## 1. Les clés facultatives de l'état (`beton/poutre/entree.py`)

| Où | Clé | Sens | Absente |
|---|---|---|---|
| lit | `pos` | lit 1 : distance parement → axe (cm) ; lit i > 1 : **entraxe vertical** depuis le lit précédent (cm) | la règle du module : enrobage + arr½(Ø étrier + Ø/2) + jeu, puis demi-Ø arrondis + jeu entre lits |
| lit | `rep` | répartition horizontale : `uniforme` (entre les montants de l'étrier), `aligne` (sur les barres du lit 1 de la face, les angles d'abord), `entraxe` (groupe centré) | `uniforme` |
| lit | `sh` | entraxe horizontal (cm, axe à axe) quand `rep` = `entraxe` | réparti |
| ligne d'effort tranchant | `de`, `a` | les barres du lit 1 inférieur que le cadre entoure (1 à n) ; une épingle n'a que `de` | cadre : toute la largeur ; épingle : la barre du milieu |
| section | `laterales` | `{"n": barres par face, "d": Ø}` | la règle automatique des armatures de peau de la note PDF : n = ⌈écart / s_max⌉ − 1 si l'écart entre lits 1 dépasse s_max |
| section | `justifs` | `{identifiant d'observation: {"texte", "empreinte"}}` : l'examen d'une observation « à examiner » par l'ingénieur (texte ≤ 600 caractères, facultatif) | aucune observation examinée |
| poutre | `dg` | le plus gros granulat (mm) | 20 mm |
| poutre | `peau_d`, `peau_smax` | Ø et écart maximal des latérales automatiques | 10 mm, 30 cm |

Le **centre de gravité** de chaque face (et donc la hauteur utile d) est calculé sur les positions réelles des lits
(Σ A·e / Σ A), comme le module ; le y_G imposé de l'écran (`yG_inf`, `yG_sup`) le remplace toujours. L'effort tranchant
compte les **branches verticales réelles** : un cadre = 2 brins, une épingle = 1 brin, A_st = Σ brins × π·Ø²/4. Le pas
`pas` est l'espacement LONGITUDINAL des cadres (dans la longueur de la poutre) ; la configuration de la section (combien
de cadres, lesquels entourent quelles barres) est l'autre information, distincte.

## 2. Les dispositions constructives (`beton/poutre/dispositions.py`)

Contrôlées section par section, sur la géométrie réelle des barres (les mêmes positions que le dessin et la note) :

1. l'espacement LIBRE entre barres voisines d'un lit ;
2. les barres d'un lit entre les montants de l'étrier (sinon : **non conforme**) ;
3. l'espacement libre entre deux lits successifs (la plus petite distance nu à nu, lits alignés ou décalés) ;
4. les deux faces ne se touchent pas ;
5. les latérales ne touchent aucune barre ;
6. l'enrobage des barres longitudinales : c + Ø étrier ≥ Ø (adhérence, c_min,b = Ø) ;
7. **chaque cadre fermé tient une barre dans ses angles supérieurs** (un cadre intérieur sous une face de deux barres
   n'en a pas : barres de montage à prévoir) ;
8. une position de cadre hors des barres du lit 1 est signalée ;
9. une **épingle** a une barre haute à son droit (sinon : barre de montage à prévoir) ;
10. les barres d'un lit supérieur d'une face sont **superposées** à celles du lit précédent (EC2 8.2(3) : barres
    « disposées verticalement les unes au-dessus des autres », un espace pour le vibreur) — le mode « Superposée » le fait.

Minimum de l'espacement libre, EC2 8.2(2), valeurs recommandées (k₁ = 1, k₂ = 5 mm) :

    s_min = max(k₁·Ø ; d_g + k₂ ; 20 mm)        Ø : la plus grosse des deux barres voisines

États (deuxième passe, 11/2026) :

- **NON CONFORME** : un espacement libre sous s_min (EC2 8.2(2) est un MINIMUM, pas un avis : un espacement de 10 mm pour
  25 mm requis reste non conforme tant que la géométrie n'est pas corrigée), un chevauchement ou une barre hors du cadre
  (marqués en plus « ne se pose pas » : le sélecteur ne les propose pas), un enrobage c + Ø étrier < Ø ;
- **À EXAMINER** (en ambre, jamais dans le vert des taux) : un angle de cadre sans barre, une épingle sans barre haute, des
  barres non superposées (8.2(3)), un cadre hors des barres du lit 1 — ce que l'ingénieur peut régler autrement que par
  la géométrie (barres de montage au plan, un vibreur qui passe ailleurs) ;
- **EXAMINÉE — JUSTIFIÉE** : une observation « à examiner » que l'ingénieur a marquée examinée, avec une justification
  facultative. Elle est gardée dans la note (`justifs`) avec l'**EMPREINTE** de ce qui a été examiné (les grandeurs de
  l'observation) : si la disposition change, l'empreinte ne correspond plus et l'observation redevient « à examiner »
  (« justification à revoir », le texte d'avant cité). Une **non-conformité ne se justifie jamais** : le moteur ignore
  une justification posée sur elle (`tests/test_poutre_dispositions.py`, `tests/test_api_web_poutre.py`).

## 3. L'incohérence relevée : le jeu entre lits par défaut

La méthode du bureau place le lit 2 à `arr½(Ø₁/2) + jeu + arr½(Ø₂/2)` du lit 1, avec un jeu par défaut de **1 cm**.
Pour deux lits Ø16 : 1 + 1 + 1 = 3 cm d'entraxe, soit **14 mm libres** — sous le minimum de l'EC2 (25 mm pour
d_g = 20 mm). Le calcul de flexion n'en est pas faux (d est calculé sur la position réelle), mais la disposition ne
permet pas un bétonnage correct.

Correction proposée, justifiée : porter le jeu entre lits à **2,1 cm** (1 + 2,1 + 1 = 4,1 cm, 41 − 16 = 25 mm libres),
ou imposer l'entraxe vertical du lit. Le site **ne change pas** le défaut (il garderait la parité avec Streamlit et
changerait les notes existantes) : il le **signale** — « Non conforme » dans les dispositions (depuis la deuxième
passe : un espacement sous le minimum est une non-conformité), et une phrase dans les Options qui chiffre le jeu
suffisant, calculée sur les règles du moteur. Avec deux lits Ø20 : 1 + 1 + 1 = 3 cm d'entraxe, **10 mm libres**.

## 4. Ce que le site montre et rend

- **Écran** (`api_web/poutre.py`, deuxième passe 11/2026 — `docs/site/REFONTE_ETABLI.md`, « La deuxième passe ») : à gauche l'arborescence
  de la note (chaque poutre « nom · b × h cm », sous la poutre ouverte ses quatre rubriques : Poutre ; Sections et
  sollicitations ; Armatures de la section courante — lits, étriers, latérales ; Options) ; au centre le dessin coté
  (`api_web/coupe_ba.py`) sous son en-tête technique ; à droite un bloc par section (la section courante seule, toutes
  dans la vue « Toutes »), coloré par son état, et la liste de contrôle des dispositions. Une section **sans aucun
  effort** est « Données incomplètes », jamais « Conforme » ; sans V, l'effort tranchant et les étriers sont « non
  effectuée ».
- **Les barres qu'un cadre entoure** se choisissent dans les données (les barres du lit 1, un clic par bout) ou au
  dessin (les poignées du cadre choisi s'aimantent aux barres) : la saisie « de-à » (`…/etriers/<k>/plage`) ne désigne
  jamais une barre absente, et une barre retirée au lit 1 ramène les cadres qui l'entouraient (un message le dit).
- **Le dessin** : chaque cadre à sa place réelle (nus extérieurs à x0, x1 du moteur, branches au nu de l'enrobage —
  deux cadres qui se recouvrent partagent leurs branches), angles cintrés autour de la barre d'angle, fermeture à 135°
  en haut à gauche avec retours de max(5 Ø ; 50 mm) (EC2 fig. 8.5), acier en double trait.
- **Note PDF** (`api_web/note_poutre.py`) : le générateur VALIDÉ de Streamlit (`modules/export_pdf.py`, `ndc_pdf/`),
  dans le navigateur, sur les valeurs que l'écran Streamlit aurait en mémoire (`api_web/poutre_streamlit.py`), complété
  des abscisses et étiquettes des lits, des latérales, d'une cinquième vérification « Dispositions constructives » et
  de l'état de la planche. Parité site ↔ planche Streamlit : `tests/test_note_poutre.py` (15 cas, 17 sections).
  **Défaut corrigé (deuxième passe)** : les barres de la coupe étaient posées depuis la ligne d'enrobage au lieu du
  parement (décalées de c) — avec deux étriers sur les barres 1 → 3 et 2 → 4, le second sortait de la poutre. La coupe
  pose désormais barres et cadres aux positions du moteur (x0, x1 de chaque cadre), chaque observation des
  dispositions a sa ligne (et sa justification), et une planche qui ne peut pas tout montrer refuse de se faire au lieu
  de tronquer en silence. Figé par `tests/test_note_poutre.py` (quatre coupes dessinées et mesurées, mutations).
- **Fichiers** : le `.stab` part en binaire (jamais « .stab.json ») ; un enregistrement de l'écran Streamlit
  (`poutre_ba.json`) s'ouvre sur le site (aller-retour vérifié grandeur par grandeur).

## 5. Ce qui n'est pas couvert

- Les armatures latérales n'entrent pas dans le calcul (dessin, note et dispositions seulement), comme dans Streamlit.
- Les cadres partiels sont positionnés sur les barres du **lit 1 inférieur** (la convention de l'écran Streamlit) ; un
  cadre défini sur la face supérieure n'existe pas. Les poignées du dessin s'aimantent donc à ces barres-là.
- La disposition « aligne » prend les barres du lit 1 de la même face ; une répartition libre barre par barre n'est
  pas offerte (l'entraxe imposé couvre le besoin courant d'un groupe centré).

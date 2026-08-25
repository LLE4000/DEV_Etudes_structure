# ndc_pdf — générateur PDF de note de calcul BA

Générateur **Python / ReportLab**, dessin vectoriel. Aucun HTML, aucun
navigateur, aucune police système : tout est embarqué.

## Ce que ça produit

Un rapport complet :

- **page de garde A4 portrait** : bandeau, cartouche PROJET / PARTIE / DATE /
  INDICE, sommaire des sections avec pastille d'état et numéro de page ;
- **une planche A4 paysage par section** : coupe cotée à gauche sur fond gris,
  dimensions / matériaux / sollicitations dessous, calculs sur deux colonnes à
  droite en flux continu.

```
python -m ndc_pdf.build sortie      # les dix palettes + le catalogue
```

```python
from ndc_pdf.styles import STYLES
style = {s.key: s for s in STYLES}["01_encre"]
doc = style.build("note.pdf")       # garde=False pour une planche seule
doc.save()
print(doc.warnings)                 # liste vide = rien ne déborde
```

## Modules

| Fichier | Rôle |
|---|---|
| `data.py` | **le seul fichier à brancher** sur le moteur de calcul |
| `styles.py` | mise en page, page de garde, palettes |
| `mathx.py` | analyseur mini-LaTeX + rendu vectoriel des formules |
| `section.py` | dessin de la coupe béton armé |
| `kit.py` | `Doc`, `Frame`, texte, tableaux, formules, tableaux clé/valeur |
| `fonts.py` | enregistrement des polices, repli glyphes |
| `build.py` | génération en lot et catalogue |
| `fonts/` | 25 TTF (OFL / GUST / Apache) |

## Contrat de `data.py`

```python
DOC = dict(bureau=..., date=..., indice=..., projet="", partie="", titre=...)

SECTIONS = [dict(
    poutre="Poutre 1", section="Section A",
    beton="C30/37", acier="B500", etat="Non vérifié",
    coupe=dict(b=200, h=400, enrobage=30, cadre_dia=10, d=340,
               lit_inf=dict(n=2, dia=16), lit_sup=dict(n=2, dia=16),
               b_label="b = 20 cm", h_label="h = 40 cm",
               c_label="c = 3,0 cm", d_label="d = 34 cm",
               lab_sup="Lit 1 : 2 Ø16", lab_sup2="402 mm²",
               lab_inf="Lit 1 : 2 Ø16", lab_inf2="402 mm²",
               lab_cadre="Étrier : Ø10", lab_cadre2="30 cm"),
    blocs=[("DIMENSIONS", [...]), ("MATÉRIAUX", [...]),
           ("SOLLICITATIONS", [...])],
    verifs=[dict(num=1, titre="...", items=[...], verdicts=[...])],
)]
```

Lignes de bloc : `(libellé | None, symbole | None, valeur, unité)`.

Items d'une vérification :

| Tuple | Rendu |
|---|---|
| `("f", libellé, formule)` | libellé en petit gris, formule dessous |
| `("v", libellé, symbole, valeur, unité)` | ligne libellé … symbole … valeur |
| `("t", texte)` | ligne en gras (« On prend 2 Ø16 ») |
| `("s", sous-titre)` | intertitre espacé (« ÉTRIERS ») |
| `("k", i)` | conclusion `verdicts[i]` |

Verdict : `dict(etat="ok" | "att" | "ko", texte="...")`.
Vert = vérifié, ocre = admissible à la limite, rouge = non vérifié.

## Formules

`\sqrt{...}`, `\frac{a}{b}`, `\max{a ; b ; c}`, `\min{...}`, `A_{s,req}`,
`10^{6}`, `\u{N/mm}^{2}` (unité romaine), `\res{...}` (résultat en gras
couleur), `\cdot \le \ge \tau \phi`, `(...)`.

Fractions composées en ligne avec parenthèses automatiques au dénominateur,
accolades `max{}` remplacées par un filet vertical, opérateurs en gris,
réduction de corps automatique si la formule dépasse la colonne.

## Palettes

Dix palettes sobres dans `styles.py` (`Encre`, `Ardoise`, `Acier`, `Marine`,
`Graphite`, `Sauge`, `Prusse`, `Prune`, `Terre`, `Bordeaux`). Changer de
palette = quatre lignes : `panel`, `acc`, `concrete`, `ko`.

## Garde-fous

- `doc.warnings` liste en points ce qui sortirait de la page. Vide = bon.
- Le symbole recule automatiquement si la valeur calée à droite le toucherait.
- `NDC_FONT_DIR` pointe un autre dossier de polices si besoin.

## Portage vers un autre dépôt

La fonctionnalité tient volontairement dans TROIS chemins — rien d'autre
n'a été touché :

| Chemin | Rôle |
|---|---|
| `ndc_pdf/` | ce dossier, autonome : code + polices + `reference/` (étalon) |
| `modules/export_pdf.py` | le branchement (`generer_rapport_pdf` v3.0) — remplace le fichier |
| `tests/test_export_pdf_ndc.py` | 34 contrôles, dont l'identité au rendu de référence |

Copier ces trois chemins dans l'autre dépôt suffit. Dépendances :
`reportlab` (l'application — fontTools vient avec) ; `pymupdf`
(les tests uniquement). Vérification après portage :

    python tests/test_export_pdf_ndc.py     # 34 OK attendu

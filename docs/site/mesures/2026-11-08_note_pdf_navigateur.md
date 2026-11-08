# Note PDF « Poutres béton armé » faite dans le navigateur — mesures du 08/11/2026

Mesure datée (elle ne se réécrit pas ; une nouvelle mesure fait un nouveau fichier).

## Conditions

- Chromium (Playwright) sans limitation de réseau, PC 1440 × 900, serveur local `site/web/outils/serveur_local.py`
  (en-têtes de sécurité de Cloudflare reproduits) ;
- note d'essai : une poutre 30 × 65 cm, deux sections (A : 4Ø20 + 4Ø16 en bas, deux cadres ; B : 5Ø16 en haut,
  un cadre et une épingle) — `scratchpad/f3/cas1.json` de la séance ;
- moteur Pyodide 0.29.5 déjà démarré (la page ouverte) ; mesure du clic sur « Note PDF » au fichier reçu.

## Résultats

| | Durée | Octets |
|---|---|---|
| Paquet de la note `moteur/note.zip` (chargé à la PREMIÈRE note, vérifié par son empreinte) | — | 2 069 956 |
| Première note (paquet chargé, déballé, note faite) | 854 ms | PDF 109 067 |
| Note suivante (paquet déjà là) | 160 ms | — |
| Même note en CPython 3.12 (`api_web/note_poutre.fichier`) | 370 ms | PDF ≈ 109 000 |

Le paquet du moteur (`moteur/moteur.zip`, chargé au démarrage) n'a pas grossi de la note : 1 523 ko (il en faisait
1 516 avant la finalisation ; la différence est le ferraillage détaillé et le dessin `coupe_ba`).

## Contenu du paquet de la note

86 fichiers : `modules/export_pdf.py`, `ndc_pdf/*.py`, six polices (Carlito ×3, Poppins Medium, DejaVu Sans ×2),
les fichiers de reportlab 5.1.0 que la note charge (relevés en faisant de vraies notes, fontTools et Pillow refusés),
la licence de reportlab, un Pillow factice (`site/note_pdf/PIL`).

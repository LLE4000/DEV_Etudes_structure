# Composants tiers du site

| Composant | Où | Licence |
|---|---|---|
| Pyodide 0.29.5 (Python 3.13 compilé en WebAssembly) | `vendor/pyodide-0.29.5/` (paquet npm officiel `pyodide`) | MPL 2.0 ; CPython : licence PSF |
| numpy 2.2.5, roue Pyodide | `vendor/pyodide-0.29.5/numpy-…whl` (empreinte = `pyodide-lock.json` officiel) | BSD 3 clauses |
| Icônes Lucide | intégrées aux pages | ISC (`LICENCE-icones.txt`) |

Le reste (pages, scripts, `api_web`, le moteur) appartient au bureau.

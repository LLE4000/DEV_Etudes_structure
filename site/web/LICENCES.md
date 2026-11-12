# Composants tiers du site

| Composant | Où | Licence |
|---|---|---|
| Pyodide 0.29.5 (Python 3.13 compilé en WebAssembly) | `vendor/pyodide-0.29.5/` (paquet npm officiel `pyodide`) | MPL 2.0 ; CPython : licence PSF |
| numpy 2.2.5, roue Pyodide | `vendor/pyodide-0.29.5/numpy-…whl` (empreinte = `pyodide-lock.json` officiel) | BSD 3 clauses |
| Icônes Lucide | intégrées aux pages | ISC (`LICENCE-icones.txt`) |
| Moteur de plans RBAR 0.4.38, version navigateur | `vendor/rbar-plans-web-0.4.38/` (paquet de RBAR, déposé par `outils/vendre_rbar_plans.py`) | RBAR : logiciel du propriétaire du site (dépôt RBAR) ; son runtime .NET 8 (`_framework/`) : MIT (.NET Foundation), avec ses composants tiers (dont les données ICU : licence Unicode) |
| Polices Liberation Sans 2.1 (dans le moteur de plans) | `vendor/rbar-plans-web-0.4.38/polices/` | SIL Open Font License 1.1 (`polices/LICENCE-Liberation.txt`) |

Le reste (pages, scripts, `api_web`, le moteur) appartient au bureau.

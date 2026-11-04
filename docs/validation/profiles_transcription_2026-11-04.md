# Transcription du catalogue des profilés — 2026-11-04

*MESURE datée (ne se réécrit pas). Produite par `tools/transcrire_profiles.py` : dimensions nominales relevées dans des paquets publiés, toutes les propriétés CALCULÉES par `acier/profiles` (intégration du contour, Saint-Venant par différences finies, Vlasov, EN 1993-1-1 §6.2.6, EN 10210-2 annexe B), puis comparées aux valeurs publiées par trois sources indépendantes. Rien n'est recopié de mémoire.*

**1419 profilés dans 21 familles, calculés en 303 s.**

| Famille | Nombre | Norme des dimensions | Source des dimensions |
|---|---|---|---|
| IPE | 23 | EN 10365:2017 (Euronorm 19-57) | structuralcodes (IPE 80–600) + steelsnakes (IPE 750) |
| IPE A | 18 | EN 10365:2017 | steelsnakes |
| IPE O | 13 | EN 10365:2017 | steelsnakes |
| IPN | 21 | DIN 1025-1 / EN 10365:2017 | structuralcodes |
| HEA | 24 | EN 10365:2017 (Euronorm 53-62) | structuralcodes |
| HEB | 24 | EN 10365:2017 (Euronorm 53-62) | structuralcodes |
| HEM | 24 | EN 10365:2017 (Euronorm 53-62) | structuralcodes |
| HEAA | 24 | EN 10365:2017 | steelsnakes |
| HL | 35 | EN 10365:2017 | steelsnakes |
| HD | 38 | EN 10365:2017 | structuralcodes |
| HP | 31 | EN 10365:2017 | structuralcodes |
| UPE | 14 | EN 10365:2017 | structuralcodes |
| UPN | 18 | DIN 1026-1 / EN 10365:2017 | structuralcodes |
| L | 197 | EN 10056-1:2017 | structuralcodes + steelsnakes (tailles manquantes) |
| LI | 36 | EN 10056-1:2017 | structuralcodes + steelsnakes (tailles manquantes) |
| CHS | 186 | EN 10210-2:2019 | eurocodepy ∪ Blue Book (finis à chaud) |
| SHS | 155 | EN 10210-2:2019 | eurocodepy ∪ Blue Book |
| RHS | 199 | EN 10210-2:2019 | eurocodepy ∪ Blue Book |
| CHS-F | 106 | EN 10219-2:2019 | Blue Book (formés à froid) |
| SHS-F | 96 | EN 10219-2:2019 | Blue Book |
| RHS-F | 137 | EN 10219-2:2019 | Blue Book |

## Contrôle croisé : propriétés calculées contre valeurs publiées

Écart relatif |calculé / publié − 1| ; tolérance de la famille × propriété entre parenthèses. Un dépassement est listé avec ses trois pires cas ; il ne corrige JAMAIS une valeur à la main (RV2) : il questionne une dimension ou une convention, et la réponse est écrite ici.

| Famille | Propriété | Source | n | écart max | écart moyen | tolérance | verdict |
|---|---|---|---|---|---|---|---|
| IPE | A | eurocodepy | 18 | 0.05 % | 0.01 % | 1 % | OK |
| IPE | A | steel-sections (structuralcodes) | 18 | 0.01 % | 0.00 % | 1 % | OK |
| IPE | A | steelsnakes (ArcelorMittal) | 23 | 0.57 % | 0.14 % | 1 % | OK |
| IPE | masse | eurocodepy | 18 | 0.28 % | 0.07 % | 1 % | OK |
| IPE | masse | steel-sections (structuralcodes) | 18 | 0.05 % | 0.01 % | 1 % | OK |
| IPE | masse | steelsnakes (ArcelorMittal) | 23 | 0.46 % | 0.14 % | 1 % | OK |
| IPE | Iy | eurocodepy | 18 | 0.02 % | 0.01 % | 2 % | OK |
| IPE | Iy | steel-sections (structuralcodes) | 18 | 0.05 % | 0.01 % | 2 % | OK |
| IPE | Iy | steelsnakes (ArcelorMittal) | 23 | 0.27 % | 0.09 % | 2 % | OK |
| IPE | Wel_y | eurocodepy | 18 | 0.04 % | 0.01 % | 2 % | OK |
| IPE | Wel_y | steelsnakes (ArcelorMittal) | 23 | 0.31 % | 0.08 % | 2 % | OK |
| IPE | Wpl_y | eurocodepy | 18 | 0.03 % | 0.01 % | 2 % | OK |
| IPE | Wpl_y | steelsnakes (ArcelorMittal) | 23 | 0.25 % | 0.09 % | 2 % | OK |
| IPE | iy | eurocodepy | 18 | 0.09 % | 0.03 % | 1 % | OK |
| IPE | iy | steelsnakes (ArcelorMittal) | 23 | 0.31 % | 0.11 % | 1 % | OK |
| IPE | Avz | eurocodepy | 18 | 0.11 % | 0.03 % | 2 % | OK |
| IPE | Iz | eurocodepy | 18 | 0.05 % | 0.01 % | 2 % | OK |
| IPE | Iz | steel-sections (structuralcodes) | 18 | 0.13 % | 0.05 % | 2 % | OK |
| IPE | Iz | steelsnakes (ArcelorMittal) | 23 | 0.50 % | 0.11 % | 2 % | OK |
| IPE | Wel_z | eurocodepy | 18 | 0.03 % | 0.01 % | 2 % | OK |
| IPE | Wel_z | steelsnakes (ArcelorMittal) | 23 | 7.73 % | 1.14 % | 2 % | DÉPASSEMENT |
| IPE | Wpl_z | eurocodepy | 18 | 0.02 % | 0.01 % | 2 % | OK |
| IPE | Wpl_z | steelsnakes (ArcelorMittal) | 23 | 0.59 % | 0.12 % | 2 % | OK |
| IPE | iz | steelsnakes (ArcelorMittal) | 23 | 0.37 % | 0.11 % | 2 % | OK |
| IPE | It | eurocodepy | 18 | 2.21 % | 0.19 % | 8 % | OK |
| IPE | It | steelsnakes (ArcelorMittal) | 23 | 1.91 % | 0.64 % | 8 % | OK |
| IPE | Iw | eurocodepy | 18 | 3.45 % | 2.10 % | 10 % | OK |
| IPE | Iw | steelsnakes (ArcelorMittal) | 23 | 1.11 % | 0.43 % | 10 % | OK |
| IPE A | A | steelsnakes (ArcelorMittal) | 18 | 0.38 % | 0.12 % | 1 % | OK |
| IPE A | masse | steelsnakes (ArcelorMittal) | 18 | 0.47 % | 0.13 % | 1 % | OK |
| IPE A | Iy | steelsnakes (ArcelorMittal) | 18 | 0.31 % | 0.10 % | 2 % | OK |
| IPE A | Wel_y | steelsnakes (ArcelorMittal) | 18 | 0.26 % | 0.09 % | 2 % | OK |
| IPE A | Wpl_y | steelsnakes (ArcelorMittal) | 18 | 0.35 % | 0.12 % | 2 % | OK |
| IPE A | iy | steelsnakes (ArcelorMittal) | 18 | 0.26 % | 0.09 % | 1 % | OK |
| IPE A | Iz | steelsnakes (ArcelorMittal) | 18 | 0.24 % | 0.08 % | 2 % | OK |
| IPE A | Wel_z | steelsnakes (ArcelorMittal) | 18 | 4.57 % | 0.63 % | 2 % | DÉPASSEMENT |
| IPE A | Wpl_z | steelsnakes (ArcelorMittal) | 18 | 0.50 % | 0.13 % | 2 % | OK |
| IPE A | iz | steelsnakes (ArcelorMittal) | 18 | 0.33 % | 0.13 % | 2 % | OK |
| IPE A | It | steelsnakes (ArcelorMittal) | 18 | 3.28 % | 1.59 % | 8 % | OK |
| IPE A | Iw | steelsnakes (ArcelorMittal) | 18 | 3.67 % | 0.64 % | 10 % | OK |
| IPE O | A | steelsnakes (ArcelorMittal) | 13 | 0.29 % | 0.08 % | 1 % | OK |
| IPE O | masse | steelsnakes (ArcelorMittal) | 13 | 0.39 % | 0.13 % | 1 % | OK |
| IPE O | Iy | steelsnakes (ArcelorMittal) | 13 | 0.35 % | 0.12 % | 2 % | OK |
| IPE O | Wel_y | steelsnakes (ArcelorMittal) | 13 | 0.32 % | 0.13 % | 2 % | OK |
| IPE O | Wpl_y | steelsnakes (ArcelorMittal) | 13 | 0.32 % | 0.11 % | 2 % | OK |
| IPE O | iy | steelsnakes (ArcelorMittal) | 13 | 0.35 % | 0.16 % | 1 % | OK |
| IPE O | Iz | steelsnakes (ArcelorMittal) | 13 | 0.27 % | 0.12 % | 2 % | OK |
| IPE O | Wel_z | steelsnakes (ArcelorMittal) | 13 | 1.93 % | 0.38 % | 2 % | OK |
| IPE O | Wpl_z | steelsnakes (ArcelorMittal) | 13 | 0.27 % | 0.07 % | 2 % | OK |
| IPE O | iz | steelsnakes (ArcelorMittal) | 13 | 0.14 % | 0.07 % | 2 % | OK |
| IPE O | It | steelsnakes (ArcelorMittal) | 13 | 0.92 % | 0.30 % | 8 % | OK |
| IPE O | Iw | steelsnakes (ArcelorMittal) | 13 | 0.62 % | 0.39 % | 10 % | OK |
| HEA | A | eurocodepy | 24 | 0.01 % | 0.01 % | 1 % | OK |
| HEA | A | steel-sections (structuralcodes) | 24 | 0.01 % | 0.00 % | 1 % | OK |
| HEA | A | steelsnakes (ArcelorMittal) | 24 | 0.48 % | 0.14 % | 1 % | OK |
| HEA | masse | eurocodepy | 24 | 0.17 % | 0.04 % | 1 % | OK |
| HEA | masse | steel-sections (structuralcodes) | 24 | 0.01 % | 0.01 % | 1 % | OK |
| HEA | masse | steelsnakes (ArcelorMittal) | 24 | 0.24 % | 0.11 % | 1 % | OK |
| HEA | Iy | eurocodepy | 24 | 0.05 % | 0.01 % | 2 % | OK |
| HEA | Iy | steel-sections (structuralcodes) | 24 | 0.02 % | 0.01 % | 2 % | OK |
| HEA | Iy | steelsnakes (ArcelorMittal) | 24 | 0.53 % | 0.11 % | 2 % | OK |
| HEA | Wel_y | eurocodepy | 24 | 0.04 % | 0.01 % | 2 % | OK |
| HEA | Wel_y | steelsnakes (ArcelorMittal) | 24 | 0.33 % | 0.09 % | 2 % | OK |
| HEA | Wpl_y | eurocodepy | 24 | 0.04 % | 0.01 % | 2 % | OK |
| HEA | Wpl_y | steelsnakes (ArcelorMittal) | 24 | 0.42 % | 0.10 % | 2 % | OK |
| HEA | iy | eurocodepy | 24 | 0.12 % | 0.02 % | 1 % | OK |
| HEA | iy | steelsnakes (ArcelorMittal) | 24 | 0.48 % | 0.13 % | 1 % | OK |
| HEA | Avz | eurocodepy | 24 | 0.05 % | 0.02 % | 2 % | OK |
| HEA | Iz | eurocodepy | 24 | 0.04 % | 0.01 % | 2 % | OK |
| HEA | Iz | steel-sections (structuralcodes) | 24 | 0.08 % | 0.02 % | 2 % | OK |
| HEA | Iz | steelsnakes (ArcelorMittal) | 24 | 0.39 % | 0.14 % | 2 % | OK |
| HEA | Wel_z | eurocodepy | 24 | 0.04 % | 0.01 % | 2 % | OK |
| HEA | Wel_z | steelsnakes (ArcelorMittal) | 24 | 1.33 % | 0.19 % | 2 % | OK |
| HEA | Wpl_z | eurocodepy | 24 | 0.05 % | 0.01 % | 2 % | OK |
| HEA | Wpl_z | steelsnakes (ArcelorMittal) | 24 | 0.40 % | 0.14 % | 2 % | OK |
| HEA | iz | steelsnakes (ArcelorMittal) | 24 | 0.11 % | 0.04 % | 2 % | OK |
| HEA | It | eurocodepy | 24 | 2.20 % | 0.19 % | 8 % | OK |
| HEA | It | steelsnakes (ArcelorMittal) | 24 | 4.01 % | 1.49 % | 8 % | OK |
| HEA | Iw | eurocodepy | 24 | 4.67 % | 2.03 % | 10 % | OK |
| HEA | Iw | steelsnakes (ArcelorMittal) | 24 | 0.45 % | 0.22 % | 10 % | OK |
| HEB | A | eurocodepy | 24 | 0.01 % | 0.00 % | 1 % | OK |
| HEB | A | steel-sections (structuralcodes) | 24 | 0.01 % | 0.00 % | 1 % | OK |
| HEB | A | steelsnakes (ArcelorMittal) | 24 | 0.38 % | 0.10 % | 1 % | OK |
| HEB | masse | eurocodepy | 24 | 0.19 % | 0.03 % | 1 % | OK |
| HEB | masse | steel-sections (structuralcodes) | 24 | 0.02 % | 0.00 % | 1 % | OK |
| HEB | masse | steelsnakes (ArcelorMittal) | 24 | 0.27 % | 0.10 % | 1 % | OK |
| HEB | Iy | eurocodepy | 24 | 0.03 % | 0.01 % | 2 % | OK |
| HEB | Iy | steel-sections (structuralcodes) | 24 | 0.01 % | 0.01 % | 2 % | OK |
| HEB | Iy | steelsnakes (ArcelorMittal) | 24 | 0.36 % | 0.09 % | 2 % | OK |
| HEB | Wel_y | eurocodepy | 24 | 0.04 % | 0.01 % | 2 % | OK |
| HEB | Wel_y | steelsnakes (ArcelorMittal) | 24 | 0.25 % | 0.09 % | 2 % | OK |
| HEB | Wpl_y | eurocodepy | 24 | 0.04 % | 0.01 % | 2 % | OK |
| HEB | Wpl_y | steelsnakes (ArcelorMittal) | 24 | 0.30 % | 0.12 % | 2 % | OK |
| HEB | iy | eurocodepy | 24 | 0.11 % | 0.02 % | 1 % | OK |
| HEB | iy | steelsnakes (ArcelorMittal) | 24 | 0.37 % | 0.10 % | 1 % | OK |
| HEB | Avz | eurocodepy | 24 | 0.03 % | 0.01 % | 2 % | OK |
| HEB | Iz | eurocodepy | 24 | 0.03 % | 0.01 % | 2 % | OK |
| HEB | Iz | steel-sections (structuralcodes) | 24 | 0.04 % | 0.02 % | 2 % | OK |
| HEB | Iz | steelsnakes (ArcelorMittal) | 24 | 0.41 % | 0.13 % | 2 % | OK |
| HEB | Wel_z | eurocodepy | 24 | 0.04 % | 0.01 % | 2 % | OK |
| HEB | Wel_z | steelsnakes (ArcelorMittal) | 24 | 1.38 % | 0.17 % | 2 % | OK |
| HEB | Wpl_z | eurocodepy | 24 | 0.05 % | 0.01 % | 2 % | OK |
| HEB | Wpl_z | steelsnakes (ArcelorMittal) | 24 | 0.37 % | 0.11 % | 2 % | OK |
| HEB | iz | steelsnakes (ArcelorMittal) | 24 | 0.18 % | 0.05 % | 2 % | OK |
| HEB | It | eurocodepy | 24 | 2.55 % | 0.28 % | 8 % | OK |
| HEB | It | steelsnakes (ArcelorMittal) | 24 | 2.36 % | 0.50 % | 8 % | OK |
| HEB | Iw | eurocodepy | 24 | 4.77 % | 2.15 % | 10 % | OK |
| HEB | Iw | steelsnakes (ArcelorMittal) | 24 | 0.57 % | 0.19 % | 10 % | OK |
| HEM | A | eurocodepy | 24 | 0.00 % | 0.00 % | 1 % | OK |
| HEM | A | steel-sections (structuralcodes) | 24 | 0.00 % | 0.00 % | 1 % | OK |
| HEM | A | steelsnakes (ArcelorMittal) | 24 | 0.30 % | 0.09 % | 1 % | OK |
| HEM | masse | eurocodepy | 24 | 0.06 % | 0.02 % | 1 % | OK |
| HEM | masse | steel-sections (structuralcodes) | 24 | 0.01 % | 0.00 % | 1 % | OK |
| HEM | masse | steelsnakes (ArcelorMittal) | 24 | 0.27 % | 0.10 % | 1 % | OK |
| HEM | Iy | eurocodepy | 24 | 0.03 % | 0.01 % | 2 % | OK |
| HEM | Iy | steel-sections (structuralcodes) | 24 | 0.01 % | 0.00 % | 2 % | OK |
| HEM | Iy | steelsnakes (ArcelorMittal) | 24 | 0.40 % | 0.10 % | 2 % | OK |
| HEM | Wel_y | eurocodepy | 24 | 0.05 % | 0.01 % | 2 % | OK |
| HEM | Wel_y | steelsnakes (ArcelorMittal) | 24 | 0.30 % | 0.09 % | 2 % | OK |
| HEM | Wpl_y | eurocodepy | 24 | 0.03 % | 0.01 % | 2 % | OK |
| HEM | Wpl_y | steelsnakes (ArcelorMittal) | 24 | 0.42 % | 0.11 % | 2 % | OK |
| HEM | iy | eurocodepy | 24 | 0.06 % | 0.02 % | 1 % | OK |
| HEM | iy | steelsnakes (ArcelorMittal) | 24 | 0.33 % | 0.11 % | 1 % | OK |
| HEM | Avz | eurocodepy | 24 | 0.02 % | 0.01 % | 2 % | OK |
| HEM | Iz | eurocodepy | 24 | 0.03 % | 0.01 % | 2 % | OK |
| HEM | Iz | steel-sections (structuralcodes) | 24 | 0.01 % | 0.01 % | 2 % | OK |
| HEM | Iz | steelsnakes (ArcelorMittal) | 24 | 0.47 % | 0.14 % | 2 % | OK |
| HEM | Wel_z | eurocodepy | 24 | 0.04 % | 0.02 % | 2 % | OK |
| HEM | Wel_z | steelsnakes (ArcelorMittal) | 24 | 0.42 % | 0.18 % | 2 % | OK |
| HEM | Wpl_z | eurocodepy | 24 | 0.04 % | 0.01 % | 2 % | OK |
| HEM | Wpl_z | steelsnakes (ArcelorMittal) | 24 | 0.40 % | 0.13 % | 2 % | OK |
| HEM | iz | steelsnakes (ArcelorMittal) | 24 | 0.10 % | 0.04 % | 2 % | OK |
| HEM | It | eurocodepy | 24 | 0.27 % | 0.06 % | 8 % | OK |
| HEM | It | steelsnakes (ArcelorMittal) | 24 | 0.59 % | 0.32 % | 8 % | OK |
| HEM | Iw | eurocodepy | 24 | 5.82 % | 2.50 % | 10 % | OK |
| HEM | Iw | steelsnakes (ArcelorMittal) | 24 | 0.61 % | 0.32 % | 10 % | OK |
| HEAA | A | steelsnakes (ArcelorMittal) | 24 | 0.51 % | 0.12 % | 1 % | OK |
| HEAA | masse | steelsnakes (ArcelorMittal) | 24 | 0.43 % | 0.11 % | 1 % | OK |
| HEAA | Iy | steelsnakes (ArcelorMittal) | 24 | 0.39 % | 0.12 % | 2 % | OK |
| HEAA | Wel_y | steelsnakes (ArcelorMittal) | 24 | 0.38 % | 0.10 % | 2 % | OK |
| HEAA | Wpl_y | steelsnakes (ArcelorMittal) | 24 | 0.51 % | 0.12 % | 2 % | OK |
| HEAA | iy | steelsnakes (ArcelorMittal) | 24 | 0.40 % | 0.12 % | 1 % | OK |
| HEAA | Iz | steelsnakes (ArcelorMittal) | 24 | 0.14 % | 0.06 % | 2 % | OK |
| HEAA | Wel_z | steelsnakes (ArcelorMittal) | 24 | 2.29 % | 0.29 % | 2 % | DÉPASSEMENT |
| HEAA | Wpl_z | steelsnakes (ArcelorMittal) | 24 | 0.41 % | 0.09 % | 2 % | OK |
| HEAA | iz | steelsnakes (ArcelorMittal) | 24 | 0.15 % | 0.04 % | 2 % | OK |
| HEAA | It | steelsnakes (ArcelorMittal) | 24 | 9.03 % | 4.58 % | 8 % | DÉPASSEMENT |
| HEAA | Iw | steelsnakes (ArcelorMittal) | 24 | 0.87 % | 0.24 % | 10 % | OK |
| HL | A | steelsnakes (ArcelorMittal) | 35 | 0.35 % | 0.06 % | 1 % | OK |
| HL | masse | steelsnakes (ArcelorMittal) | 35 | 0.68 % | 0.13 % | 1 % | OK |
| HL | Iy | steelsnakes (ArcelorMittal) | 35 | 1.21 % | 0.21 % | 2 % | OK |
| HL | Wel_y | steelsnakes (ArcelorMittal) | 35 | 1.19 % | 0.27 % | 2 % | OK |
| HL | Wpl_y | steelsnakes (ArcelorMittal) | 35 | 1.30 % | 0.26 % | 2 % | OK |
| HL | iy | steelsnakes (ArcelorMittal) | 35 | 0.16 % | 0.06 % | 1 % | OK |
| HL | Iz | steelsnakes (ArcelorMittal) | 35 | 0.40 % | 0.07 % | 2 % | OK |
| HL | Wel_z | steelsnakes (ArcelorMittal) | 35 | 0.24 % | 0.09 % | 2 % | OK |
| HL | Wpl_z | steelsnakes (ArcelorMittal) | 35 | 0.47 % | 0.14 % | 2 % | OK |
| HL | iz | steelsnakes (ArcelorMittal) | 35 | 0.51 % | 0.15 % | 2 % | OK |
| HL | It | steelsnakes (ArcelorMittal) | 35 | 5.90 % | 1.19 % | 8 % | OK |
| HL | Iw | steelsnakes (ArcelorMittal) | 35 | 1.55 % | 0.29 % | 10 % | OK |
| HD | A | steelsnakes (ArcelorMittal) | 5 | 0.38 % | 0.15 % | 1 % | OK |
| HD | masse | steelsnakes (ArcelorMittal) | 5 | 0.08 % | 0.05 % | 1 % | OK |
| HD | Iy | steelsnakes (ArcelorMittal) | 5 | 0.53 % | 0.22 % | 2 % | OK |
| HD | Wel_y | steelsnakes (ArcelorMittal) | 5 | 0.27 % | 0.12 % | 2 % | OK |
| HD | Wpl_y | steelsnakes (ArcelorMittal) | 5 | 0.31 % | 0.15 % | 2 % | OK |
| HD | iy | steelsnakes (ArcelorMittal) | 5 | 0.40 % | 0.22 % | 1 % | OK |
| HD | Iz | steelsnakes (ArcelorMittal) | 5 | 0.11 % | 0.07 % | 2 % | OK |
| HD | Wel_z | steelsnakes (ArcelorMittal) | 5 | 0.22 % | 0.09 % | 2 % | OK |
| HD | Wpl_z | steelsnakes (ArcelorMittal) | 5 | 0.08 % | 0.05 % | 2 % | OK |
| HD | iz | steelsnakes (ArcelorMittal) | 5 | 0.06 % | 0.03 % | 2 % | OK |
| HD | It | steelsnakes (ArcelorMittal) | 5 | 9.03 % | 4.87 % | 8 % | DÉPASSEMENT |
| HD | Iw | steelsnakes (ArcelorMittal) | 5 | 0.30 % | 0.20 % | 10 % | OK |
| HP | A | steelsnakes (ArcelorMittal) | 31 | 0.44 % | 0.16 % | 1 % | OK |
| HP | masse | steelsnakes (ArcelorMittal) | 31 | 0.64 % | 0.10 % | 1 % | OK |
| HP | Iy | steelsnakes (ArcelorMittal) | 31 | 0.44 % | 0.10 % | 2 % | OK |
| HP | Wel_y | steelsnakes (ArcelorMittal) | 31 | 0.30 % | 0.10 % | 2 % | OK |
| HP | Wpl_y | steelsnakes (ArcelorMittal) | 31 | 0.32 % | 0.11 % | 2 % | OK |
| HP | iy | steelsnakes (ArcelorMittal) | 31 | 0.42 % | 0.17 % | 1 % | OK |
| HP | Iz | steelsnakes (ArcelorMittal) | 31 | 0.40 % | 0.15 % | 2 % | OK |
| HP | Wel_z | steelsnakes (ArcelorMittal) | 31 | 0.31 % | 0.09 % | 2 % | OK |
| HP | Wpl_z | steelsnakes (ArcelorMittal) | 31 | 0.42 % | 0.13 % | 2 % | OK |
| HP | iz | steelsnakes (ArcelorMittal) | 31 | 0.09 % | 0.03 % | 2 % | OK |
| HP | It | steelsnakes (ArcelorMittal) | 31 | 2.36 % | 0.40 % | 8 % | OK |
| HP | Iw | steelsnakes (ArcelorMittal) | 31 | 0.54 % | 0.22 % | 10 % | OK |
| UPE | A | steelsnakes (ArcelorMittal) | 14 | 0.30 % | 0.08 % | 1 % | OK |
| UPE | masse | steelsnakes (ArcelorMittal) | 14 | 0.28 % | 0.09 % | 1 % | OK |
| UPE | Iy | steelsnakes (ArcelorMittal) | 14 | 0.26 % | 0.10 % | 2 % | OK |
| UPE | Wel_y | steelsnakes (ArcelorMittal) | 14 | 0.26 % | 0.07 % | 2 % | OK |
| UPE | Wpl_y | steelsnakes (ArcelorMittal) | 14 | 0.29 % | 0.08 % | 2 % | OK |
| UPE | iy | steelsnakes (ArcelorMittal) | 14 | 0.38 % | 0.11 % | 1 % | OK |
| UPE | Iz | steelsnakes (ArcelorMittal) | 14 | 0.45 % | 0.16 % | 2 % | OK |
| UPE | Wel_z | steelsnakes (ArcelorMittal) | 14 | 3.33 % | 1.01 % | 2 % | DÉPASSEMENT |
| UPE | Wpl_z | steelsnakes (ArcelorMittal) | 14 | 15.62 % | 3.19 % | 2 % | DÉPASSEMENT |
| UPE | iz | steelsnakes (ArcelorMittal) | 14 | 0.24 % | 0.07 % | 2 % | OK |
| UPE | It | steelsnakes (ArcelorMittal) | 14 | 0.48 % | 0.21 % | 8 % | OK |
| UPE | Iw | steelsnakes (ArcelorMittal) | 14 | 7.79 % | 5.19 % | 10 % | OK |
| UPN | A | steel-sections (structuralcodes) | 18 | 0.01 % | 0.00 % | 1 % | OK |
| UPN | A | steelsnakes (ArcelorMittal) | 18 | 0.67 % | 0.14 % | 1 % | OK |
| UPN | masse | steel-sections (structuralcodes) | 18 | 0.04 % | 0.01 % | 1 % | OK |
| UPN | masse | steelsnakes (ArcelorMittal) | 18 | 0.62 % | 0.19 % | 1 % | OK |
| UPN | Iy | steelsnakes (ArcelorMittal) | 18 | 0.35 % | 0.15 % | 2 % | OK |
| UPN | Wel_y | steelsnakes (ArcelorMittal) | 18 | 0.30 % | 0.12 % | 2 % | OK |
| UPN | Wpl_y | steelsnakes (ArcelorMittal) | 18 | 2.04 % | 0.41 % | 2 % | DÉPASSEMENT |
| UPN | iy | steelsnakes (ArcelorMittal) | 18 | 1.01 % | 0.18 % | 1 % | DÉPASSEMENT |
| UPN | Iz | steelsnakes (ArcelorMittal) | 18 | 1.10 % | 0.54 % | 2 % | OK |
| UPN | Wel_z | steelsnakes (ArcelorMittal) | 18 | 7.37 % | 1.62 % | 2 % | DÉPASSEMENT |
| UPN | Wpl_z | steelsnakes (ArcelorMittal) | 18 | 1.52 % | 0.49 % | 2 % | OK |
| UPN | iz | steelsnakes (ArcelorMittal) | 18 | 0.77 % | 0.29 % | 2 % | OK |
| UPN | It | steelsnakes (ArcelorMittal) | 18 | 4.46 % | 1.56 % | 8 % | OK |
| UPN | Iw | steelsnakes (ArcelorMittal) | 18 | 8.75 % | 4.84 % | 10 % | OK |
| L | A | steelsnakes (ArcelorMittal) | 192 | 1.07 % | 0.15 % | 1 % | DÉPASSEMENT |
| L | masse | steelsnakes (ArcelorMittal) | 192 | 2.22 % | 0.18 % | 1 % | DÉPASSEMENT |
| L | Iy | steelsnakes (ArcelorMittal) | 192 | 3.65 % | 0.30 % | 2 % | DÉPASSEMENT |
| L | Wel_y | steelsnakes (ArcelorMittal) | 192 | 2.04 % | 0.30 % | 2 % | DÉPASSEMENT |
| L | Iz | steelsnakes (ArcelorMittal) | 192 | 3.65 % | 0.30 % | 2 % | DÉPASSEMENT |
| L | It | steelsnakes (ArcelorMittal) | 192 | 3.71 % | 2.04 % | 8 % | OK |
| L | Iu | steelsnakes (ArcelorMittal) | 192 | 1.67 % | 0.31 % | 2 % | OK |
| L | Iv | steelsnakes (ArcelorMittal) | 192 | 2.35 % | 0.55 % | 3 % | OK |
| LI | A | steelsnakes (ArcelorMittal) | 32 | 0.34 % | 0.12 % | 1 % | OK |
| LI | masse | steelsnakes (ArcelorMittal) | 32 | 0.39 % | 0.13 % | 1 % | OK |
| LI | Iy | steelsnakes (ArcelorMittal) | 32 | 0.44 % | 0.12 % | 2 % | OK |
| LI | Wel_y | steelsnakes (ArcelorMittal) | 32 | 0.32 % | 0.11 % | 2 % | OK |
| LI | Iz | steelsnakes (ArcelorMittal) | 32 | 0.64 % | 0.14 % | 2 % | OK |
| LI | It | steelsnakes (ArcelorMittal) | 32 | 2.64 % | 1.89 % | 8 % | OK |
| LI | Iu | steelsnakes (ArcelorMittal) | 32 | 0.46 % | 0.12 % | 2 % | OK |
| LI | Iv | steelsnakes (ArcelorMittal) | 32 | 0.67 % | 0.29 % | 3 % | OK |
| CHS | A | eurocodepy | 162 | 0.39 % | 0.10 % | 1 % | OK |
| CHS | A | steelsnakes (Blue Book) | 103 | 0.39 % | 0.12 % | 1 % | OK |
| CHS | masse | eurocodepy | 162 | 0.44 % | 0.09 % | 1 % | OK |
| CHS | masse | steelsnakes (Blue Book) | 103 | 0.44 % | 0.11 % | 1 % | OK |
| CHS | Iy | eurocodepy | 162 | 0.32 % | 0.03 % | 2 % | OK |
| CHS | Iy | steelsnakes (Blue Book) | 103 | 0.45 % | 0.09 % | 2 % | OK |
| CHS | Wel_y | eurocodepy | 162 | 0.31 % | 0.06 % | 2 % | OK |
| CHS | Wel_y | steelsnakes (Blue Book) | 103 | 0.48 % | 0.09 % | 2 % | OK |
| CHS | Wpl_y | eurocodepy | 162 | 0.43 % | 0.06 % | 2 % | OK |
| CHS | Wpl_y | steelsnakes (Blue Book) | 103 | 0.44 % | 0.09 % | 2 % | OK |
| CHS | iy | eurocodepy | 162 | 0.39 % | 0.10 % | 1 % | OK |
| CHS | Avz | eurocodepy | 162 | 0.00 % | 0.00 % | 2 % | OK |
| CHS | Iz | eurocodepy | 162 | 0.32 % | 0.03 % | 2 % | OK |
| CHS | Wel_z | eurocodepy | 162 | 0.31 % | 0.06 % | 2 % | OK |
| CHS | Wpl_z | eurocodepy | 162 | 0.43 % | 0.06 % | 2 % | OK |
| CHS | It | eurocodepy | 162 | 0.32 % | 0.03 % | 8 % | OK |
| CHS | It | steelsnakes (Blue Book) | 103 | 0.46 % | 0.09 % | 8 % | OK |
| CHS | Wt | eurocodepy | 162 | 0.43 % | 0.05 % | 5 % | OK |
| CHS | Wt | steelsnakes (Blue Book) | 103 | 0.43 % | 0.10 % | 5 % | OK |
| SHS | A | eurocodepy | 123 | 0.34 % | 0.09 % | 1 % | OK |
| SHS | A | steelsnakes (Blue Book) | 123 | 0.37 % | 0.10 % | 1 % | OK |
| SHS | masse | eurocodepy | 123 | 0.42 % | 0.10 % | 1 % | OK |
| SHS | masse | steelsnakes (Blue Book) | 123 | 0.43 % | 0.10 % | 1 % | OK |
| SHS | Iy | eurocodepy | 123 | 0.44 % | 0.06 % | 2 % | OK |
| SHS | Iy | steelsnakes (Blue Book) | 123 | 0.44 % | 0.11 % | 2 % | OK |
| SHS | Wel_y | eurocodepy | 123 | 0.47 % | 0.09 % | 2 % | OK |
| SHS | Wel_y | steelsnakes (Blue Book) | 123 | 0.48 % | 0.10 % | 2 % | OK |
| SHS | Wpl_y | eurocodepy | 123 | 0.40 % | 0.07 % | 2 % | OK |
| SHS | Wpl_y | steelsnakes (Blue Book) | 123 | 0.40 % | 0.09 % | 2 % | OK |
| SHS | iy | eurocodepy | 123 | 0.37 % | 0.08 % | 1 % | OK |
| SHS | Avz | eurocodepy | 123 | 4.13 % | 1.72 % | 2 % | DÉPASSEMENT |
| SHS | Iz | eurocodepy | 123 | 0.44 % | 0.06 % | 2 % | OK |
| SHS | Wel_z | eurocodepy | 123 | 0.47 % | 0.09 % | 2 % | OK |
| SHS | Wpl_z | eurocodepy | 123 | 0.40 % | 0.07 % | 2 % | OK |
| SHS | It | eurocodepy | 123 | 0.30 % | 0.04 % | 8 % | OK |
| SHS | It | steelsnakes (Blue Book) | 123 | 0.39 % | 0.09 % | 8 % | OK |
| SHS | Wt | eurocodepy | 123 | 0.28 % | 0.06 % | 5 % | OK |
| SHS | Wt | steelsnakes (Blue Book) | 123 | 0.46 % | 0.09 % | 5 % | OK |
| RHS | A | eurocodepy | 125 | 0.42 % | 0.10 % | 1 % | OK |
| RHS | A | steelsnakes (Blue Book) | 161 | 0.42 % | 0.10 % | 1 % | OK |
| RHS | masse | eurocodepy | 125 | 0.42 % | 0.10 % | 1 % | OK |
| RHS | masse | steelsnakes (Blue Book) | 161 | 0.43 % | 0.11 % | 1 % | OK |
| RHS | Iy | eurocodepy | 125 | 0.36 % | 0.05 % | 2 % | OK |
| RHS | Iy | steelsnakes (Blue Book) | 161 | 0.36 % | 0.09 % | 2 % | OK |
| RHS | Wel_y | eurocodepy | 125 | 0.42 % | 0.09 % | 2 % | OK |
| RHS | Wel_y | steelsnakes (Blue Book) | 161 | 0.43 % | 0.10 % | 2 % | OK |
| RHS | Wpl_y | eurocodepy | 125 | 0.37 % | 0.08 % | 2 % | OK |
| RHS | Wpl_y | steelsnakes (Blue Book) | 161 | 0.47 % | 0.09 % | 2 % | OK |
| RHS | iy | eurocodepy | 125 | 0.43 % | 0.10 % | 1 % | OK |
| RHS | Avz | eurocodepy | 125 | 7.68 % | 3.11 % | 2 % | DÉPASSEMENT |
| RHS | Iz | eurocodepy | 125 | 0.37 % | 0.06 % | 2 % | OK |
| RHS | Iz | steelsnakes (Blue Book) | 161 | 0.39 % | 0.11 % | 2 % | OK |
| RHS | Wel_z | eurocodepy | 125 | 0.40 % | 0.09 % | 2 % | OK |
| RHS | Wpl_z | eurocodepy | 125 | 0.39 % | 0.09 % | 2 % | OK |
| RHS | It | eurocodepy | 125 | 0.41 % | 0.07 % | 8 % | OK |
| RHS | It | steelsnakes (Blue Book) | 161 | 0.37 % | 0.10 % | 8 % | OK |
| RHS | Wt | eurocodepy | 125 | 0.37 % | 0.08 % | 5 % | OK |
| RHS | Wt | steelsnakes (Blue Book) | 161 | 0.46 % | 0.10 % | 5 % | OK |
| CHS-F | A | steelsnakes (Blue Book) | 106 | 0.39 % | 0.11 % | 1 % | OK |
| CHS-F | masse | steelsnakes (Blue Book) | 106 | 0.44 % | 0.09 % | 1 % | OK |
| CHS-F | Iy | steelsnakes (Blue Book) | 106 | 0.45 % | 0.10 % | 2 % | OK |
| CHS-F | Wel_y | steelsnakes (Blue Book) | 106 | 0.40 % | 0.10 % | 2 % | OK |
| CHS-F | Wpl_y | steelsnakes (Blue Book) | 106 | 0.44 % | 0.09 % | 2 % | OK |
| CHS-F | It | steelsnakes (Blue Book) | 106 | 0.46 % | 0.09 % | 8 % | OK |
| CHS-F | Wt | steelsnakes (Blue Book) | 106 | 0.43 % | 0.10 % | 5 % | OK |
| SHS-F | A | steelsnakes (Blue Book) | 96 | 0.47 % | 0.12 % | 1 % | OK |
| SHS-F | masse | steelsnakes (Blue Book) | 96 | 0.41 % | 0.10 % | 1 % | OK |
| SHS-F | Iy | steelsnakes (Blue Book) | 96 | 0.40 % | 0.10 % | 2 % | OK |
| SHS-F | Wel_y | steelsnakes (Blue Book) | 96 | 0.52 % | 0.11 % | 2 % | OK |
| SHS-F | Wpl_y | steelsnakes (Blue Book) | 96 | 0.35 % | 0.09 % | 2 % | OK |
| SHS-F | It | steelsnakes (Blue Book) | 96 | 0.41 % | 0.10 % | 8 % | OK |
| SHS-F | Wt | steelsnakes (Blue Book) | 96 | 0.31 % | 0.10 % | 5 % | OK |
| RHS-F | A | steelsnakes (Blue Book) | 137 | 0.47 % | 0.14 % | 1 % | OK |
| RHS-F | masse | steelsnakes (Blue Book) | 137 | 0.41 % | 0.09 % | 1 % | OK |
| RHS-F | Iy | steelsnakes (Blue Book) | 137 | 0.42 % | 0.10 % | 2 % | OK |
| RHS-F | Wel_y | steelsnakes (Blue Book) | 137 | 0.48 % | 0.10 % | 2 % | OK |
| RHS-F | Wpl_y | steelsnakes (Blue Book) | 137 | 0.40 % | 0.10 % | 2 % | OK |
| RHS-F | Iz | steelsnakes (Blue Book) | 137 | 0.45 % | 0.10 % | 2 % | OK |
| RHS-F | It | steelsnakes (Blue Book) | 137 | 0.46 % | 0.11 % | 8 % | OK |
| RHS-F | Wt | steelsnakes (Blue Book) | 137 | 0.37 % | 0.10 % | 5 % | OK |

## Dépassements et leur lecture

- **IPE · Wel_z** (steelsnakes (ArcelorMittal)) : IPE 80 : 3.691 calculé / 4 publié (7.7 %) ; IPE 120 : 8.646 calculé / 9 publié (3.9 %) ; IPE 100 : 5.789 calculé / 6 publié (3.5 %). **Lecture :** la source publie W_el,z arrondi à l'unité pour les petits profilés (IPE 80 : « 4 » pour 3,69) ; l'écart est l'arrondi de la source, pas le calcul.
- **IPE A · Wel_z** (steelsnakes (ArcelorMittal)) : IPE A 100 : 4.772 calculé / 5 publié (4.6 %) ; IPE A 160 : 13.27 calculé / 13 publié (2.1 %) ; IPE A 200 : 23.43 calculé / 23 publié (1.9 %). **Lecture :** la source publie W_el,z arrondi à l'unité pour les petits profilés (IPE 80 : « 4 » pour 3,69) ; l'écart est l'arrondi de la source, pas le calcul.
- **HEAA · Wel_z** (steelsnakes (ArcelorMittal)) : HEAA 100 : 18.41 calculé / 18 publié (2.3 %) ; HEAA 120 : 26.47 calculé / 27 publié (2.0 %) ; HEAA 140 : 39.26 calculé / 39 publié (0.7 %). **Lecture :** la source publie W_el,z arrondi à l'unité pour les petits profilés (IPE 80 : « 4 » pour 3,69) ; l'écart est l'arrondi de la source, pas le calcul.
- **HEAA · It** (steelsnakes (ArcelorMittal)) : HEAA 260 : 27.38 calculé / 30.1 publié (9.0 %) ; HEAA 300 : 43.52 calculé / 47.8 publié (8.9 %) ; HEAA 320 : 49.5 calculé / 53.6 publié (7.7 %). **Lecture :** les catalogues emploient la formule empirique I_t = 2/3·b·t_f³ + 1/3·(h − 2t_f)·t_w³ + 2·α·D⁴ − 0,42·t_f⁴, dont le terme de congé 2·α·D⁴ est un ajustement ; pour des semelles minces à grand congé (HE AA, HD légers : r/t_f ≈ 2,5) il surestime I_t de 8 à 10 % par rapport à la solution exacte de Saint-Venant, qui est retenue (convergée à 0,1 % entre deux pas de grille ; sécuritaire pour le déversement).
- **HD · It** (steelsnakes (ArcelorMittal)) : HD 260x54.1 : 27.38 calculé / 30.1 publié (9.0 %) ; HD 320x74.2 : 49.5 calculé / 53.6 publié (7.7 %) ; HD 260x68.2 : 52.02 calculé / 54.2 publié (4.0 %). **Lecture :** les catalogues emploient la formule empirique I_t = 2/3·b·t_f³ + 1/3·(h − 2t_f)·t_w³ + 2·α·D⁴ − 0,42·t_f⁴, dont le terme de congé 2·α·D⁴ est un ajustement ; pour des semelles minces à grand congé (HE AA, HD légers : r/t_f ≈ 2,5) il surestime I_t de 8 à 10 % par rapport à la solution exacte de Saint-Venant, qui est retenue (convergée à 0,1 % entre deux pas de grille ; sécuritaire pour le déversement).
- **UPE · Wel_z** (steelsnakes (ArcelorMittal)) : UPE 100 : 10.63 calculé / 11 publié (3.3 %) ; UPE 160 : 22.58 calculé / 23 publié (1.8 %) ; UPE 200 : 34.43 calculé / 35 publié (1.6 %). **Lecture :** la source publie W_el,z arrondi à l'unité pour les petits profilés (IPE 80 : « 4 » pour 3,69) ; l'écart est l'arrondi de la source, pas le calcul.
- **UPE · Wpl_z** (steelsnakes (ArcelorMittal)) : UPE 400 : 220.8 calculé / 191 publié (15.6 %) ; UPE 360 : 189.2 calculé / 178 publié (6.3 %) ; UPE 330 : 161.7 calculé / 156 publié (3.7 %). **Lecture :** W_pl,z par l'axe d'aires égales (dans l'âme pour les grands UPE) : UPE 400 → âme 400 × 13,5 = 54 cm² sur 91,9, axe à 11,5 mm du dos, 2·(18 × 101,5)·52,75 + 400·(11,5²/2 + 2²/2) = 220 cm³, calcul à la main ; la valeur publiée (191) ne correspond pas à cette définition et n'est pas reprise.
- **UPN · Wpl_y** (steelsnakes (ArcelorMittal)) : UPN 350 : 899.3 calculé / 918 publié (2.0 %) ; UPN 80 : 31.9 calculé / 32.3 publié (1.2 %) ; UPN 50 : 12.96 calculé / 13.1 publié (1.1 %). **Lecture :** semelles coniques : la source arrondit et modélise l'inclinaison autrement ; 2 % au pire.
- **UPN · iy** (steelsnakes (ArcelorMittal)) : UPN 320 : 11.98 calculé / 12.1 publié (1.0 %) ; UPN 280 : 10.84 calculé / 10.9 publié (0.6 %) ; UPN 50 : 1.929 calculé / 1.92 publié (0.5 %). **Lecture :** arrondi de la source (12,1 pour 11,98).
- **UPN · Wel_z** (steelsnakes (ArcelorMittal)) : UPN 50 : 3.705 calculé / 4 publié (7.4 %) ; UPN 80 : 6.351 calculé / 6 publié (5.8 %) ; UPN 100 : 8.387 calculé / 8 publié (4.8 %). **Lecture :** la source publie W_el,z arrondi à l'unité pour les petits profilés (IPE 80 : « 4 » pour 3,69) ; l'écart est l'arrondi de la source, pas le calcul.
- **L · A** (steelsnakes (ArcelorMittal)) : L 300x300x26 : 149.6 calculé / 148 publié (1.1 %) ; L 300x300x31 : 176.7 calculé / 175 publié (1.0 %) ; L 300x300x28 : 160.5 calculé / 159 publié (0.9 %). **Lecture :** tailles hors EN 10056-1 (300 × 300 × 26 à 35) : la source arrondit l'aire à 3 chiffres ; écart ≤ 1,1 %.
- **L · masse** (steelsnakes (ArcelorMittal)) : L 110x110x14 : 22.78 calculé / 23.3 publié (2.2 %) ; L 100x100x18 : 25.84 calculé / 26.2 publié (1.4 %) ; L 300x300x26 : 117.4 calculé / 116 publié (1.2 %). **Lecture :** L 110 × 110 × 14 : la source publie A = 29,0 cm² (= calcul) mais une masse de 23,3 kg/m, incohérente avec son aire (22,8) ; l'erreur est dans la source.
- **L · Iy** (steelsnakes (ArcelorMittal)) : L 200x200x16 : 2341 calculé / 2430 publié (3.6 %) ; L 300x300x28 : 1.367e+04 calculé / 1.34e+04 publié (2.0 %) ; L 300x300x27 : 1.325e+04 calculé / 1.3e+04 publié (1.9 %). **Lecture :** L 200 × 200 × 16 : la source publie 2 430 ; le calcul à la main (ailes 16 × 200 et 184 × 16, centre de gravité à 55,9 mm, congés) donne 2 342 ; chiffres inversés dans la source (2 340).
- **L · Wel_y** (steelsnakes (ArcelorMittal)) : L 300x300x25 : 572.5 calculé / 561 publié (2.0 %) ; L 300x300x26 : 593.7 calculé / 582 publié (2.0 %) ; L 300x300x29 : 656.6 calculé / 645 publié (1.8 %). **Lecture :** suit I_y et l'arrondi de la source sur les très grandes cornières.
- **L · Iz** (steelsnakes (ArcelorMittal)) : L 200x200x16 : 2341 calculé / 2430 publié (3.6 %) ; L 300x300x28 : 1.367e+04 calculé / 1.34e+04 publié (2.0 %) ; L 300x300x27 : 1.325e+04 calculé / 1.3e+04 publié (1.9 %). **Lecture :** même cas que I_y (cornière égale).
- **SHS · Avz** (eurocodepy) : SHS 60x60x8 : 7.976 calculé / 8.32 publié (4.1 %) ; SHS 50x50x6.3 : 5.293 calculé / 5.506 publié (3.9 %) ; SHS 40x40x5 : 3.366 calculé / 3.5 publié (3.8 %). **Lecture :** eurocodepy n'emploie pas A·h/(b + h) de l'EN 1993-1-1 §6.2.6(3) f) ; la formule de la norme est retenue.
- **RHS · Avz** (eurocodepy) : RHS 80x40x8 : 10.63 calculé / 11.52 publié (7.7 %) ; RHS 50x30x5 : 4.207 calculé / 4.5 publié (6.5 %) ; RHS 90x50x8 : 12.31 calculé / 13.12 publié (6.2 %). **Lecture :** idem SHS.

## Méthodes et conventions

- I/H à ailes parallèles : contour exact avec quatre congés r ; IPN : semelles à 14 %, t_f à b/4 de l'axe, congé r1, arrondi r2 ; UPN : semelles à 8 % (5 % au-delà de 300), t_f à b/2 du bout d'aile (h ≤ 300) ou au milieu de la partie inclinée (h > 300) — conventions retenues parce qu'elles retrouvent les aires publiées ;
- cornières : congé r1 au talon, arrondis r2 aux bouts ; axes principaux u-v calculés ; I_w non défini ;
- creux : coins r_ext = 1,5·t / r_int = t (EN 10210), 2·t à 3·t (EN 10219) ; I_t et W_t par l'annexe B de l'EN 10210-2 ; tubes ronds exacts ;
- I_t des sections ouvertes : Saint-Venant par différences finies sur le contour exact, pas = épaisseur minimale / 20 (erreur de discrétisation ≈ +0,5 %, mesurée sur le rectangle contre la série exacte) ; les catalogues emploient une formule approchée avec terme de congé — l'écart observé est la différence des deux méthodes, pas une erreur de transcription ;
- I_w : I_z·(h − t_f)²/4 pour les I/H (forme des catalogues), Vlasov sur la ligne moyenne pour les U ;
- A_v : EN 1993-1-1 §6.2.6(3), sans le plancher η·h_w·t_w ;
- masse : A × 7 850 kg/m³ ; surface : périmètre extérieur.


# -*- coding: utf-8 -*-
"""Maquettes du site « Stab Méthodes » (site/maquettes) : ce qu'une capture d'écran ne prouve pas.

- les valeurs du bureau s'affichent telles que les moteurs les donnent, sans arrondi de taux (γ_M2 = 1,25 et non
  « 1,2 », α_b = 12,96 et non « 13,0 ») ;
- la page poutre montre, section par section, les vérifications du module poutre, aux taux et couleurs du module ;
- la hauteur utile d se cote depuis la face COMPRIMÉE : depuis le haut en travée, depuis le bas sur appui ;
- la barre du PC et les onglets du téléphone donnent les cinq mêmes entrées, dans le même ordre ;
- un champ « modifié » garde la valeur du bureau écrite sous lui ; une valeur du bureau ne porte pas d'étiquette ;
- les fichiers du générateur restent lisibles par Python 3.11.

    python3 tests/test_site_maquettes.py
"""
import os
import re
import shutil
import subprocess
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ICI = os.path.join(RACINE, "site", "maquettes")
sys.path.insert(0, ICI)

import construire as C  # noqa: E402
import dessin_poutre as DP  # noqa: E402
import page_parametres as PP  # noqa: E402
import page_poutre as PB  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:200]}"))


def nombre(s: str) -> float:
    return float(s.replace(" ", "").replace("\xa0", "").replace(" ", "").replace(",", "."))


# 1. le format des valeurs : nb() n'arrondit pas ce qu'il écrit, pc() reste réservé aux taux
for x, dec, attendu in ((1.25, 3, "1,25"), (12.96, 3, "12,96"), (0.1709, 4, "0,1709"), (1.0, 3, "1"), (333.3, 1, "333,3")):
    chk(f"nb({x}) = « {attendu} »", C.nb(x, dec) == attendu, C.nb(x, dec))
chk("pc() arrondit à une décimale : il ne convient pas aux coefficients (raison d'être de nb)", C.pc(1.25) != "1,25", C.pc(1.25))

# 2. la page Paramètres : chaque valeur du bureau affichée est celle du moteur
_, html_p, _ = PP.parametres()
gd, pd = C.D["gc_defauts"], C.D["poutre"]["defauts"]
valeurs = re.findall(r'<input value="([^"]*)"', html_p)


def champ_apres(motif: str, html: str) -> str:
    m = re.search(motif + r'.*?<input value="([^"]*)"', html, re.S)
    return m.group(1) if m else ""


for nom, motif, x in (("γ_M2", r"γ<sub>M2</sub>", gd["gamma_M2"]), ("γ_M0", r"γ<sub>M0</sub>", gd["gamma_M0"]),
                      ("σ_adm = f_y / …", r"σ<sub>adm</sub>", gd["gamma_adm"]), ("γ_s", r"γ<sub>s</sub>", pd["gamma_s"]),
                      ("enrobage", r'<span class="sym">c</span><span>Enrobage</span>', pd["enrobage"]),
                      ("jeu entre lits", r"<span>Jeu entre lits</span>", pd["jeu_lits"])):
    v = champ_apres(motif, html_p)
    chk(f"Paramètres : {nom} affiché = {x} (moteur)", v != "" and abs(nombre(v) - x) < 1e-9, v)
chk("Paramètres : aucun champ ne montre « 1,2 » à la place de γ_M2 = 1,25", "1,2" not in valeurs, valeurs)

# 3. la page poutre : les vérifications du module, aux taux et aux couleurs du module
_, html_b, _ = PB.poutre()
rangs = re.findall(r'data-calcul="([AB])" title="Voir le calcul"><div class="l1"><b>([^<]+)</b>.*?<span class="pct (\w+)">([^<]+)</span>',
                   html_b, re.S)
attendus = [(s["nom"], v["nom"], v["niveau"], f"{C.pc(v['taux'])}{C.NBSP}%") for s in C.D["poutre"]["sections"] for v in s["verifs"]]
chk(f"Poutre : {len(attendus)} lignes de résultat, une par vérification du module, dans l'ordre", rangs == attendus,
    (rangs[:3], attendus[:3]))
chk("Poutre : une seule ligne gouvernante", html_b.count('class="el gov"') == 1, html_b.count('class="el gov"'))
for s in C.D["poutre"]["sections"]:
    m = re.search(rf'id="ck-{s["nom"]}".*?<tbody>(.*?)</tbody>', html_b, re.S)
    n = m.group(1).count("<tr") if m else 0
    chk(f"Poutre : la fenêtre de calcul de la section {s['nom']} reprend ses {len(s['verifs'])} vérifications", n == len(s["verifs"]), n)
hyp = C.D["poutre"]["hyp"]
chk("Poutre : α_b écrit comme le module le donne (12,96, pas « 13,0 »)", f"α<sub>b</sub> = {C.nb(hyp['alpha_b'])} " in html_b
    and abs(nombre(C.nb(hyp["alpha_b"])) - hyp["alpha_b"]) < 1e-9, hyp["alpha_b"])
chk("Poutre : V3 sans sollicitations est « à compléter », jamais conforme", "à compléter" in html_b)

# 4. la hauteur utile se cote depuis la face comprimée
def cote_d(tendu: str) -> str:
    g, _, _ = DP.section(30, 50, 3, 8, [(3, 20, 5.0)], [(2, 14, 4.5)], "étrier Ø8 · e = 20 cm", "", tendu=tendu)
    m = re.search(r">d = ([\d.]+)<", g)
    return m.group(1) if m else ""


chk("Dessin : en travée (bas tendu), d = h − 50 = 450 mm, mesuré depuis le haut", cote_d("inf") == "450", cote_d("inf"))
chk("Dessin : sur appui (haut tendu), d = h − 45 = 455 mm, mesuré depuis le bas", cote_d("sup") == "455", cote_d("sup"))
chk("Dessin : la section B (appui) de l'exemple porte sa cote d", ">d = " in C.D["poutre"]["svg"]["B"])

# 5. les mêmes entrées sur le PC et au téléphone
barre = re.findall(r'class="navitem[^"]*" data-tip="([^"]+)"', C.rail("accueil"))
cinq = ["Accueil", "Projets", "Modules", "Équipe", "Paramètres"]
chk("Barre du PC : cinq entrées nommées au survol", barre[:5] == cinq and len([b for b in barre if b in cinq]) == 5, barre)
onglets = re.findall(r"<span>([^<]+)</span></a>", C.mnav("accueil"))
chk("Téléphone : les cinq mêmes entrées, dans le même ordre", onglets == cinq, onglets)

# 5 bis. les modules MASQUÉS (BIBLIOTHEQUES.md § 2.1 ; DALLES_FUSION.md § 2.4 : « predalle », absorbé par « dalle ») ne sont
# listés ni à l'accueil, ni dans le volet « Modules », ni dans « Nouvelle note » — une seule carte « Dalles béton armé »
_, html_a, _ = C.accueil()
_listes = html_a   # (l'accueil porte le rail et son volet « Modules », les tuiles, et la fenêtre « Nouvelle note » : trois listes)
chk("Maquettes : « predalle » est masqué (MASQUES) et reste une clé des familles (le site le sert encore)",
    "predalle" in C.MASQUES and any(k == "predalle" for _, _, ms in C.FAMILLES for k, _ in ms))
chk("Accueil, volet « Modules », « Nouvelle note » : aucun module masqué listé, une seule carte « Dalles béton armé », plus de « Prédalles »",
    "Prédalles" not in _listes and _listes.count("Dalles béton armé") == 3 and not any(C.MOD[k] in _listes for k in C.MASQUES)
    and _listes.count('<a class="tile"') == sum(len(C.listes(ms)) for _, _, ms in C.FAMILLES), _listes.count("Dalles béton armé"))
chk("Accueil : le compte des modules de chaque famille est celui des modules listés",
    all(f"{len(C.listes(ms))} modules" in html_a and f"{n} · {len(C.listes(ms))}<" in html_a for _, n, ms in C.FAMILLES))

# 6. l'origine d'une valeur : « modifié » garde la valeur du bureau sous le champ ; le bureau ne s'étiquette pas
h_mod = C.ligne("Jeu entre lits", C.num("1,5", "cm"), src="modifié:1 cm")
h_bur = C.ligne("Béton", C.sel("C30/37"), full=True, src="bureau")
chk("« modifié » : l'étiquette et la valeur du bureau écrite sous le champ", 'class="src mod"' in h_mod and "bureau : 1 cm" in h_mod, h_mod)
chk("Valeur du bureau : aucune étiquette (le libellé garde sa place)", 'class="src' not in h_bur and 'data-src="bureau"' in h_bur, h_bur)

# 7. Python 3.11 lit le générateur (pas de f-string imbriquée à la 3.12)
fichiers = [os.path.join(ICI, f) for f in ("construire.py", "page_poutre.py", "page_parametres.py", "dessin_poutre.py", "apercu.py")]
py311 = shutil.which("python3.11")
if py311:
    p = subprocess.run([py311, "-c", "import ast, sys\nfor f in sys.argv[1:]: ast.parse(open(f, encoding='utf-8').read(), f)"] + fichiers,
                       capture_output=True, text=True)
    chk("Python 3.11 lit les fichiers du générateur", p.returncode == 0, p.stderr.strip().splitlines()[-1:] if p.stderr else "")
else:
    print("     (python3.11 absent : la lecture par Python 3.11 n'est pas contrôlée sur cette machine)")

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:200])
sys.exit(1 if KO else 0)

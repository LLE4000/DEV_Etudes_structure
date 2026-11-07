# -*- coding: utf-8 -*-
"""Les MUTATIONS qui prouvent que tests/test_reference_refonte.py rougit — et la contre-épreuve qui prouve qu'il ne
rougit pas pour un changement de balisage seul. Chaque changement est posé dans le code, le test lancé sur les modules
concernés, puis le fichier est REMIS à l'identique (octet pour octet, à rebours quand un fichier est muté deux fois) et
contrôlé par ``git diff --quiet``. À lancer sur une copie propre (aucune modification en cours sur les fichiers mutés) :

    /usr/bin/python3.12 tests/reference_refonte/mutations.py            # toutes
    /usr/bin/python3.12 tests/reference_refonte/mutations.py "a " "d1"  # celles dont le nom commence ainsi

Rouge attendu : (a, a') une formule d'un moteur changée de 0,1 %, (b) un avertissement retiré, (b') un message de refus
retiré, (c) une valeur de champ affichée changée, (d2) un balisage que des modules relisent. Vert attendu : (d1) un
balisage changé seul. Ce n'est pas une suite (lancer_tests.py ne lance que tests/test_*.py) : c'est la preuve du test.
"""
import os
import subprocess
import sys
import time

RACINE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = sys.executable

MUTATIONS = [
    ("a · formule du moteur poutre : f_yd = f_yk/γ_s × 1,001 (+0,1 %)", "beton/poutre/moteur.py",
     '"fyd": fyk / gamma_s}', '"fyd": fyk / gamma_s * 1.001}', ["poutre"], "rouge"),
    ("a' · formule du moteur soudures (acier/soudures/criteres.py, §4.5.3.2) : σ_c × 1,001 (+0,1 %)", "acier/soudures/criteres.py",
     "    sc = math.sqrt(sigma_perp ** 2 + 3.0 * (tau_perp ** 2 + tau_par ** 2))\n    lim1 = fu / (bw * gamma_M2)",
     "    sc = 1.001 * math.sqrt(sigma_perp ** 2 + 3.0 * (tau_perp ** 2 + tau_par ** 2))\n    lim1 = fu / (bw * gamma_M2)", ["soudures", "garde_corps"], "rouge"),
    ("b · avertissement retiré : « Nouveau calcul … à valider par le bureau » (flambement)", "api_web/flambement.py",
     """    return (f'<div class="fl-nouveau">{C.message(C.icon("info") + "<span><b>Nouveau calcul</b> EN 1993-1-1 §6.3.1, à valider par le bureau</span>")}'
            '</div>')""",
     """    return '<div class="fl-nouveau"></div>'""", ["flambement"], "rouge"),
    ("b' · message de refus retiré : « Il en faut au moins un » (poutre)", "api_web/poutre.py",
     'return "Il en faut au moins un"', 'return None', ["poutre"], "rouge"),
    ("c · valeur de champ changée à l'affichage : h affiché + 1 cm (poutre)", "api_web/poutre.py",
     'C.num(["h"], p.get("h"), "cm")', 'C.num(["h"], p.get("h") + 1, "cm")', ["poutre"], "rouge"),
]


def poser(fichier, ancien, nouveau):
    chemin = os.path.join(RACINE, fichier)
    with open(chemin, "rb") as f:
        origine = f.read()
    texte = origine.decode("utf-8")
    assert texte.count(ancien) == 1, (fichier, ancien[:60], texte.count(ancien))
    with open(chemin, "wb") as f:
        f.write(texte.replace(ancien, nouveau).encode("utf-8"))
    return chemin, origine


def remettre(chemin, origine):
    with open(chemin, "wb") as f:
        f.write(origine)


def lancer(modules):
    t0 = time.time()
    p = subprocess.run([PY, "tests/test_reference_refonte.py"] + modules, cwd=RACINE, capture_output=True, text=True)
    return p, time.time() - t0


def main(choix):
    sales = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=RACINE, capture_output=True, text=True).stdout
    if sales.strip():
        sys.exit(f"des fichiers suivis sont modifiés — les mutations demandent une copie propre :\n{sales}")
    conformes = 0
    for nom, fichier, ancien, nouveau, modules, attendu in MUTATIONS + CONTRE:
        if fichier is None or (choix and not any(nom.startswith(c) for c in choix)):
            continue
        poses = []
        try:
            for f, a, n in (fichier if isinstance(fichier, list) else [(fichier, ancien, nouveau)]):
                poses.append(poser(f, a, n))
            p, duree = lancer(modules)
        finally:
            for chemin, origine in reversed(poses):        # à rebours : un fichier muté deux fois revient à son origine
                remettre(chemin, origine)
        propre = subprocess.run(["git", "diff", "--quiet", "--"] + [os.path.relpath(c, RACINE) for c, _ in poses], cwd=RACINE).returncode == 0
        ko = [l for l in p.stdout.splitlines() if l.startswith("KO")]
        dernier = [l for l in p.stdout.splitlines() if l.startswith("RÉSULTAT")]
        verdict = "rouge" if p.returncode else "vert"
        conformes += verdict == attendu and propre
        print(f"\n=== {nom}\n    attendu {attendu}, obtenu {verdict} ({dernier[-1] if dernier else '?'}, {duree:.1f} s) — "
              f"{'CONFORME' if verdict == attendu else 'NON CONFORME'} ; fichier remis à l'identique : {propre}")
        lignes = p.stdout.splitlines()
        for i, l in enumerate(lignes):
            if l.startswith("KO"):
                print("    " + l[:200])
                for m in lignes[i + 1:i + 7]:
                    if m.startswith("        |"):
                        print("    " + m[:220])
        if not ko and verdict == "rouge":
            print(p.stdout[-1500:], p.stderr[-1500:])
    print(f"\nMUTATIONS CONFORMES : {conformes}")


# la CONTRE-ÉPREUVE : un changement de balisage SEUL (balises, classes, ordre des blocs, segment marqué par
# aria-pressed au lieu de la classe « on ») — le test doit rester VERT : il fige le calcul et l'affichage, pas la mise en page
_CO = "api_web/commun.py"
_LIGNE_AVANT = """    return (f'<div class="el{" gov" if gov else ""}"{ouvre}><div class="l1"><b>{esc(nom)}</b>{p}</div>'
            f'<div class="l2"><div class="bar"><b class="{niv}" style="width:{_largeur(taux)}%"></b></div>'"""
_LIGNE_APRES = """    return (f'<article class="ligne-resultat{" gouvernante" if gov else ""}"{ouvre}><header><strong>{esc(nom)}</strong>{p}</header>'
            f'<div class="l2"><div class="bar"><b class="{niv}" style="width:{_largeur(taux)}%"></b></div>'"""
_LIGNE_FIN_AVANT = """            f'<span class="pct {niv}"{titre}>{txt}</span></div></div>')"""
_LIGNE_FIN_APRES = """            f'<span class="pct {niv}"{titre}>{txt}</span></div></article>')"""
CONTRE = [
    ("d1 · contre-épreuve VERTE : balisage seul (ligne de résultat enveloppée autrement, grande pastille, section en "
     "<details>, champ en <span>, segment retenu par aria-pressed au lieu de la classe « on », message en <p>) + blocs du "
     "résultat de la poutre réordonnés (pied, liste, messages, verdict)",
     [(_CO, _LIGNE_AVANT, _LIGNE_APRES), (_CO, _LIGNE_FIN_AVANT, _LIGNE_FIN_APRES),
      (_CO, """        return f'<span class="chip xl {niv}"{titre}><small>η</small>{txt}</span>'""",
       """        return f'<mark class="pastille grande n-{niv}"{titre}><sub>η</sub>{txt}</mark>'"""),
      (_CO, """    return (f'<div class="sec{" open" if ouvert else ""}" data-sec="{esc(cle or n)}"><div class="sechd" data-replier="1"><span class="n">{n}</span>'
            f'<span class="tt">{titre}</span><span class="sum">{resume}</span>{icon("chevron-down")}</div>'
            f'<div class="rows">{corps}</div></div>')""",
       """    return (f'<details class="panneau"{" open" if ouvert else ""} data-sec="{esc(cle or n)}"><summary data-replier="1"><em>{n}</em>'
            f'<strong>{titre}</strong><small>{resume}</small></summary>'
            f'<fieldset class="champs">{corps}</fieldset></details>')"""),
      (_CO, """    return (f'<div class="field{" locked" if desactive else ""}">{lk}<input data-chemin="{_attr_chemin(chemin)}" data-sorte="{sorte}"'
            f' inputmode="decimal" value="{esc(v)}"{lim}{vide}{" disabled" if desactive else ""}>{u}</div>')""",
       """    return (f'<span class="saisie{" verrou" if desactive else ""}">{lk}<input class="nombre" data-chemin="{_attr_chemin(chemin)}" data-sorte="{sorte}"'
            f' inputmode="decimal" value="{esc(v)}"{lim}{vide}{" disabled" if desactive else ""}>{u}</span>')"""),
      (_CO, """    bt = "".join(f'<button type="button" class="{"on" if k == valeur else ""}" data-chemin="{_attr_chemin(chemin)}" data-sorte="choix" data-valeur="{esc(k)}"{off}>{esc(t)}</button>'""",
       """    bt = "".join(f'<button type="button" class="segment" aria-pressed="{"true" if k == valeur else "false"}" data-chemin="{_attr_chemin(chemin)}" data-sorte="choix" data-valeur="{esc(k)}"{off}>{esc(t)}</button>'"""),
      (_CO, """    return f'<div class="msg{" grave" if grave else ""}">{texte_}</div>'""",
       """    return f'<p role="alert" class="avis{" fort" if grave else ""}">{texte_}</p>'"""),
      ("api_web/poutre.py", """f'<div class="pbody">{verdict}<div class="els">{rows}</div>{messages}</div>{pied}'""",
       """f'<section class="corps">{pied}<ul class="liste">{rows}</ul>{messages}{verdict}</section>'""")],
     None, None, [], "vert"),
    ("d2 · balisage dont des MODULES dépendent (classe « vtab » du tableau de calcul, petite pastille) : rouge attendu, "
     "le test voit ce que l'œil verrait (catalogue, données béton, sol, plancher relisent ce balisage)",
     [(_CO, """    table = (f'<table class="vtab"><thead>""", """    table = (f'<table class="tableau-calcul"><thead>"""),
      (_CO, """    return f'<span class="chip {niv}"{titre}>{txt}</span>'""", """    return f'<mark class="pastille n-{niv}"{titre}>{txt}</mark>'""")],
     None, None, ["catalogue", "donnees_beton", "sol", "plancher", "poutre"], "rouge"),
]

if __name__ == "__main__":
    main(sys.argv[1:])

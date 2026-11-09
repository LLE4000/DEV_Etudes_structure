# -*- coding: utf-8 -*-
"""Le LANCEUR de suites (lancer_tests.py) et son SUIVI des dépendances (outils/suivi_tests/sitecustomize.py), vérifiés
sur un petit dépôt fabriqué dans un dossier temporaire :

1. premier passage : toutes les suites sont jouées (jamais vues), la rouge est rouge, le témoin n'est écrit que pour
   les vertes ; le suivi a vu le module importé, le fichier lu, le dossier listé — et pas les dossiers que le
   mécanisme d'import liste pour lui-même ;
2. rien n'a bougé : les suites suivies sont INCHANGÉES ; une suite dont un fils Python échappe au suivi (« -I ») est
   « non suivie » et toujours rejouée ; un fils Python ordinaire est suivi (ses lectures comptent) ;
3. un fichier de données change → seules les suites qui l'ont lu sont rejouées ; un fichier ajouté dans un dossier
   listé → la suite qui le liste ; un module importé change → la suite qui l'importe ; un nouveau test → lui seul ;
4. une suite rouge qui devient verte reçoit son témoin, puis est inchangée ; une verte qui devient rouge perd le sien ;
5. « --tout » rejoue tout ; « -j 1 » rend le même total ;
6. mutations : sans l'événement « open » dans le suivi, le changement d'un fichier lu passe inaperçu (le contrôle 3
   rougit) ; sans l'appariement des fils Python, la suite au fils « -I » serait crue suivie ; sans « os.listdir »,
   le fichier ajouté passe inaperçu.

    python3 tests/test_lancer_tests.py
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANCEUR = os.path.join(RACINE, "lancer_tests.py")
SUIVI = os.path.join(RACINE, "outils", "suivi_tests", "sitecustomize.py")

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:300]}"))


SUITE_A = '''import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pkg.a
v = json.load(open(os.path.join(os.path.dirname(__file__), "..", "data", "a.json")))
print(f"RÉSULTAT : {2 + pkg.a.Y + v['a']} OK, 0 échec(s)")
'''
SUITE_B = '''import os, glob
racine = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"RÉSULTAT : {len(glob.glob(os.path.join(racine, 'data', '*')))} OK, 0 échec(s)")
'''
SUITE_C = '''import subprocess, sys
p = subprocess.run([sys.executable, "-I", "-c", "print(1)"], capture_output=True, text=True)
print(f"RÉSULTAT : {int(p.stdout)} OK, 0 échec(s)")
'''
SUITE_D = '''import subprocess, sys, os
racine = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = subprocess.run([sys.executable, "-c", "import json,sys; print(json.load(open(sys.argv[1]))['a'])", os.path.join(racine, "data", "a.json")], capture_output=True, text=True)
print(f"RÉSULTAT : {int(p.stdout)} OK, 0 échec(s)")
'''
ROUGE = 'print("RÉSULTAT : 1 OK, 1 échec(s)"); raise SystemExit(1)\n'
VERTE = 'print("RÉSULTAT : 1 OK, 0 échec(s)")\n'


def fabriquer(dossier, suivi=SUIVI):
    for d in ("tests", "pkg", "data", os.path.join("outils", "suivi_tests")):
        os.makedirs(os.path.join(dossier, d), exist_ok=True)
    shutil.copy(suivi, os.path.join(dossier, "outils", "suivi_tests", "sitecustomize.py"))
    ecrire(dossier, "pkg/__init__.py", "X = 1\n")
    ecrire(dossier, "pkg/a.py", "Y = 2\n")
    ecrire(dossier, "data/a.json", '{"a": 1}\n')
    ecrire(dossier, "data/b.txt", "b\n")
    ecrire(dossier, "tests/test_a.py", SUITE_A)
    ecrire(dossier, "tests/test_b.py", SUITE_B)
    ecrire(dossier, "tests/test_c.py", SUITE_C)
    ecrire(dossier, "tests/test_d.py", SUITE_D)
    ecrire(dossier, "tests/test_e.py", ROUGE)


def ecrire(dossier, rel, texte):
    with open(os.path.join(dossier, rel), "w", encoding="utf-8") as fh:
        fh.write(texte)


def lancer(dossier, *args, lanceur=LANCEUR):
    """Rend (code, {suite: état}, dernière ligne)."""
    p = subprocess.run([sys.executable, lanceur, "--racine", dossier, *args], capture_output=True, text=True, cwd=RACINE)
    etats = {}
    for l in p.stdout.splitlines():
        m = re.match(r"(INCHANGÉ|VERT|ROUGE|IGNORÉ)\s+tests/(test_\w+\.py)", l)
        if m:
            etats[m.group(2)] = m.group(1)
    return p.returncode, etats, (p.stdout.strip().splitlines() or [""])[-1], p.stdout + p.stderr


def temoin(dossier, suite):
    chemin = os.path.join(dossier, ".temoin_tests", suite + ".json")
    if not os.path.exists(chemin):
        return None
    with open(chemin, encoding="utf-8") as fh:
        return json.load(fh)


# ------------------------------------------------------------------ 1. premier passage
chk("le lanceur et le suivi existent", os.path.exists(LANCEUR) and os.path.exists(SUIVI))
tmp = tempfile.mkdtemp(prefix="lanceur_")
fabriquer(tmp)
code, etats, fin, sortie = lancer(tmp)
chk("1. premier passage : cinq suites jouées, code 1 (une rouge)", code == 1 and len(etats) == 5, (code, etats))
chk("1. a, b, c, d vertes, e rouge", all(etats.get(s) == "VERT" for s in ("test_a.py", "test_b.py", "test_c.py", "test_d.py")) and etats.get("test_e.py") == "ROUGE", etats)
chk("1. total 10 OK, 1 échec", "TOTAL : 10 OK, 1 échec(s)" in fin, fin)
ta, tb, tc, td, te = (temoin(tmp, s) for s in ("test_a.py", "test_b.py", "test_c.py", "test_d.py", "test_e.py"))
chk("1. témoins des vertes écrits, pas de la rouge", all(t is not None for t in (ta, tb, tc, td)) and te is None)
chk("1. suite a : le module, le fichier lu et la suite elle-même sont des dépendances",
    ta and {"pkg/a.py", "pkg/__init__.py", "data/a.json", "tests/test_a.py"} <= set(ta["fichiers"]), ta and sorted(ta["fichiers"]))
chk("1. suite a : les dossiers listés par le mécanisme d'import ne sont pas des dépendances", ta and ta["dossiers"] == {}, ta and ta["dossiers"])
chk("1. suite b : le dossier listé (glob) est une dépendance, avec ses noms", tb and tb["dossiers"] == {"data": ["a.json", "b.txt"]}, tb and tb["dossiers"])
chk("1. suite c (fils « -I ») : non suivie, raison dite", tc and tc["suivie"] is False and "fils Python" in tc["raison"], tc and (tc["suivie"], tc["raison"]))
chk("1. suite d (fils Python ordinaire) : suivie, la lecture du fils compte", td and td["suivie"] is True and "data/a.json" in td["fichiers"], td and (td["suivie"], td["fichiers"]))
chk("1. le témoin note la durée et le Python", ta and isinstance(ta["duree"], float) and ta["python"] == sys.version.split()[0])

# ------------------------------------------------------------------ 2. rien n'a bougé
code, etats, fin, sortie = lancer(tmp, "--pourquoi")
chk("2. a, b, d inchangées ; c rejouée (non suivie) ; e rejouée (rouge)",
    etats.get("test_a.py") == "INCHANGÉ" and etats.get("test_b.py") == "INCHANGÉ" and etats.get("test_d.py") == "INCHANGÉ"
    and etats.get("test_c.py") == "VERT" and etats.get("test_e.py") == "ROUGE", etats)
chk("2. --pourquoi dit la raison", "non suivie" in sortie and "jamais vue verte" in sortie)
chk("2. le total compte les inchangées comme vertes", "4/5 suites vertes" in fin and "3 inchangées" in fin, fin)

# ------------------------------------------------------------------ 3. ce qui change rejoue ce qui en dépend
ecrire(tmp, "data/a.json", '{"a": 2}\n')
code, etats, fin, sortie = lancer(tmp, "--pourquoi")
chk("3. data/a.json change → a et d rejouées, b inchangée", etats.get("test_a.py") == "VERT" and etats.get("test_d.py") == "VERT" and etats.get("test_b.py") == "INCHANGÉ", etats)
chk("3. la raison nomme le fichier", "data/a.json a changé" in sortie)
ecrire(tmp, "data/c.txt", "c\n")
code, etats, fin, sortie = lancer(tmp, "--pourquoi")
chk("3. fichier ajouté dans data/ → b rejouée (listage), a et d inchangées", etats.get("test_b.py") == "VERT" and etats.get("test_a.py") == "INCHANGÉ" and etats.get("test_d.py") == "INCHANGÉ", etats)
chk("3. la raison nomme le dossier", "dossier data/ a changé" in sortie)
# (une taille différente : le .pyc de Python se fie à la taille et à l'horodatage à la seconde — même taille dans la
# même seconde, il resservirait l'ancien code ; le lanceur, lui, compare le contenu)
ecrire(tmp, "pkg/a.py", "Y = 3  # modifié\n")
code, etats, fin, sortie = lancer(tmp)
chk("3. pkg/a.py change → a rejouée seule (hors c et e)", etats.get("test_a.py") == "VERT" and etats.get("test_b.py") == "INCHANGÉ" and etats.get("test_d.py") == "INCHANGÉ", etats)
chk("3. le compte de a suit la nouvelle valeur (7 OK)", re.search(r"VERT\s+tests/test_a\.py\s+7 OK", sortie) is not None)
ecrire(tmp, "tests/test_f.py", VERTE)
code, etats, fin, sortie = lancer(tmp)
chk("3. nouveau test → lui seul est joué (et c, e)", etats.get("test_f.py") == "VERT" and all(etats.get(s) == "INCHANGÉ" for s in ("test_a.py", "test_b.py", "test_d.py")), etats)

# ------------------------------------------------------------------ 4. rouge ↔ verte
ecrire(tmp, "tests/test_e.py", VERTE)
code, etats, fin, sortie = lancer(tmp)
chk("4. e devient verte : jouée, code 0, témoin écrit", code == 0 and etats.get("test_e.py") == "VERT" and temoin(tmp, "test_e.py") is not None, (code, etats))
code, etats, fin, sortie = lancer(tmp)
chk("4. puis inchangée", etats.get("test_e.py") == "INCHANGÉ", etats)
ecrire(tmp, "tests/test_e.py", ROUGE)
code, etats, fin, sortie = lancer(tmp)
chk("4. e redevient rouge (sa propre source a changé) : jouée, rouge, témoin retiré", code == 1 and etats.get("test_e.py") == "ROUGE" and temoin(tmp, "test_e.py") is None, (code, etats))

# ------------------------------------------------------------------ 5. --tout, -j 1
code, etats, fin, sortie = lancer(tmp, "--tout")
chk("5. --tout rejoue les six", len(etats) == 6 and "INCHANGÉ" not in etats.values() and "6 jouées" in fin, (etats, fin))
code1, etats1, fin1, _ = lancer(tmp, "--tout", "-j", "1")
chk("5. -j 1 : même total", fin1.split("(")[0] == fin.split("(")[0] and "1 processus" in fin1, (fin, fin1))
chk("5. filtre par nom : seule la suite nommée", set(lancer(tmp, "test_b")[1]) == {"test_b.py"})

# ------------------------------------------------------------------ 6. mutations
with open(SUIVI, encoding="utf-8") as fh:
    suivi_src = fh.read()
with open(LANCEUR, encoding="utf-8") as fh:
    lanceur_src = fh.read()


def mutation(nom, texte_suivi, texte_lanceur, scenario):
    d = tempfile.mkdtemp(prefix="lanceur_mut_")
    chemin_suivi = os.path.join(d, "suivi_mut.py")
    with open(chemin_suivi, "w", encoding="utf-8") as fh:
        fh.write(texte_suivi)
    chemin_lanceur = os.path.join(d, "lancer_mut.py")
    with open(chemin_lanceur, "w", encoding="utf-8") as fh:
        fh.write(texte_lanceur)
    depot = os.path.join(d, "depot")
    os.makedirs(depot)
    fabriquer(depot, chemin_suivi)
    chk(nom, scenario(depot, chemin_lanceur))
    shutil.rmtree(d, ignore_errors=True)


def sans_open(depot, lanceur):
    lancer(depot, lanceur=lanceur)
    ecrire(depot, "data/a.json", '{"a": 2}\n')
    etats = lancer(depot, lanceur=lanceur)[1]
    return etats.get("test_a.py") == "INCHANGÉ"           # à tort : le contrôle 3 rougirait


def sans_appariement(depot, lanceur):
    lancer(depot, lanceur=lanceur)
    return lancer(depot, lanceur=lanceur)[1].get("test_c.py") == "INCHANGÉ"   # à tort : le fils « -I » n'est pas suivi


def sans_listdir(depot, lanceur):
    lancer(depot, lanceur=lanceur)
    ecrire(depot, "data/c.txt", "c\n")
    return lancer(depot, lanceur=lanceur)[1].get("test_b.py") == "INCHANGÉ"  # à tort


mut1 = suivi_src.replace('if evenement == "open":', 'if evenement == "open" and False:')
chk("6. mutation 1 préparée", mut1 != suivi_src)
mutation("6. sans l'événement « open », un fichier lu qui change passe inaperçu (le contrôle 3 le verrait)", mut1, lanceur_src, sans_open)
mut2 = lanceur_src.replace("if ecrits < python_lances:", "if False:")
chk("6. mutation 2 préparée", mut2 != lanceur_src)
mutation("6. sans l'appariement des fils Python, la suite au fils « -I » serait crue suivie", suivi_src, mut2, sans_appariement)
mut3 = suivi_src.replace('elif evenement in ("os.listdir", "os.scandir"):', 'elif evenement in ("os.listdir_", "os.scandir_"):')
chk("6. mutation 3 préparée", mut3 != suivi_src)
mutation("6. sans « os.listdir », un fichier ajouté dans un dossier listé passe inaperçu", mut3, lanceur_src, sans_listdir)

shutil.rmtree(tmp, ignore_errors=True)
print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
sys.exit(1 if KO else 0)

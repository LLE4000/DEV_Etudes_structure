# -*- coding: utf-8 -*-
"""Le CONTRAT commun des modules du site : ce que chaque module du registre (api_web/registre.py) doit tenir, quel
qu'il soit. Un module ajouté entre ici sans rien écrire : il y est soumis d'office.

Pour chaque module :
- une note neuve se calcule et la réponse porte tout ce que l'écran pose (données, résultat, fenêtres, vues, résumé) ;
- l'état de la note survit à un aller-retour JSON (le fichier .stab) et sa relecture est stable ;
- chaque champ du cadre Données pointe un endroit réel de l'état : le réécrire avec sa valeur ne change rien ;
- le PARCOURS (api_web/parcours.py : chaque option de chaque menu, chaque segment, chaque case, chaque ajout, sur deux
  niveaux, chaque sélecteur, les éléments) ne rencontre AUCUNE erreur ;
- chaque ligne qui annonce un calcul a sa fenêtre ; chaque sélecteur annoncé répond ;
- un texte saisi est échappé dans tous les fragments HTML ; aucun fragment ne porte de script ni de « on…= » ;
- une note abîmée rend une erreur lisible, jamais une exception ;
- chaque jauge du résultat est bornée à [0, 100] % ;
- chaque champ « vide = automatique » (C.num(auto=…)) accepte d'être vidé et refuse un texte qui n'est pas un nombre ;
- chaque bouton « télécharger » annoncé (C.bouton_telecharger) rend un fichier valide (nom sûr, type permis, 5 Mo au
  plus, un JSON qui se relit) ; un téléchargement inconnu rend un refus lisible ;
- les éléments de la note s'ajoutent et le dernier ne se supprime pas.

    python3 tests/test_api_web_modules.py            # tous les modules
    python3 tests/test_api_web_modules.py dalle sol  # ceux-là seulement
"""
import copy
import html
import json
import os
import re
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

from api_web import commun as C  # noqa: E402
from api_web import parcours as P  # noqa: E402
from api_web import registre as REG  # noqa: E402
from api_web import routeur as R  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:300]}"))


def req(**k):
    return json.loads(R.traiter(json.dumps(k)))


CLES_REPONSE = ("resultat", "volets", "vues", "phrase", "statut_html", "resume", "donnees", "elements", "etat")
CHAMPS = re.compile(r'<input data-chemin="([^"]+)" data-sorte="(nombre|entier|texte)"[^>]*?value="([^"]*)"')
MENUS = re.compile(r'<select data-chemin="([^"]+)" data-sorte="choix"[^>]*>.*?<option value="([^"]*)" selected', re.S)
SEGMENTS_ON = re.compile(r'<button type="button" class="on" data-chemin="([^"]+)" data-sorte="choix" data-valeur="([^"]*)"')
CASES = re.compile(r'<input type="checkbox" data-chemin="([^"]+)" data-sorte="case"( checked)?')
EN_LIGNE = re.compile(r"<script|\son[a-z]+\s*=", re.I)
PIEGE = '<img src=x onerror=alert(1)>'


def json_lisible(f) -> bool:
    """Un fichier déclaré JSON se relit en JSON (les autres types : rien de plus à lire)."""
    if f.get("type") != "application/json":
        return True
    try:
        json.loads(f["texte"])
        return True
    except ValueError:
        return False


def fragments(r):
    return {"résultat": r["resultat"], "fenêtres": r["volets"], "données": r["donnees"], "éléments": r["elements"],
            "statut": r["statut_html"], "phrase": r["phrase"], "dessins": "".join(v["svg"] for v in r["vues"])}


def un_module(cle):
    print(f"\n=== {cle}")
    mod = REG.module(cle)
    r = req(module=cle, action="nouveau")
    chk(f"{cle} : une note neuve se calcule", r.get("ok"), r.get("erreur"))
    if not r.get("ok"):
        print(r.get("trace", "")[-1500:])
        return
    chk(f"{cle} : la réponse porte tout ce que l'écran pose", all(k in r for k in CLES_REPONSE), [k for k in CLES_REPONSE if k not in r])
    chk(f"{cle} : le résumé a un taux, un niveau de couleur connu, un statut",
        set(r["resume"]) >= {"taux", "niveau", "statut", "gouvernant"} and r["resume"]["niveau"] in C.STATUTS, r["resume"])
    chk(f"{cle} : chaque vue a une clé, un titre et un contenu (dessin SVG, ou bloc HTML : un tableau)",
        all(v.get("cle") and v.get("titre") and v.get("court") and v.get("svg", "").lstrip().startswith(("<svg", "<div")) for v in r["vues"]),
        [v.get("cle") for v in r["vues"]])
    chk(f"{cle} : le cadre Données et le résultat ne sont pas vides", len(r["donnees"]) > 200 and len(r["resultat"]) > 100)
    etat = r["etat"]
    # l'état : JSON, relecture stable
    e2 = mod.normaliser(json.loads(json.dumps(etat)))
    chk(f"{cle} : l'état survit à un aller-retour JSON (.stab) et sa relecture est stable",
        e2 == etat and mod.normaliser(copy.deepcopy(e2)) == e2)
    # chaque champ pointe l'état
    champs = [(c, s, v) for c, s, v in CHAMPS.findall(r["donnees"])]
    champs += [(c, "choix", v) for c, v in MENUS.findall(r["donnees"])]
    champs += [(c, "choix", v) for c, v in SEGMENTS_ON.findall(r["donnees"])]
    champs += [(c, "case", bool(k)) for c, k in CASES.findall(r["donnees"])]
    chk(f"{cle} : le cadre Données porte des champs ({len(champs)})", len(champs) >= 3, len(champs))
    mauvais = []
    for chemin, sorte, valeur in champs:
        v = valeur.replace("&amp;", "&").replace("&quot;", '"').replace("&lt;", "<").replace("&gt;", ">").replace("&#x27;", "'") \
            if isinstance(valeur, str) else valeur
        out = req(module=cle, action="modifier", etat=copy.deepcopy(etat), chemin=chemin.split("/"), valeur=v)
        if not out.get("ok") or out.get("message") or out["etat"] != etat:
            mauvais.append((chemin, valeur, out.get("message") or out.get("erreur")))
    chk(f"{cle} : réécrire chaque champ avec sa valeur affichée ne change rien", not mauvais, mauvais[:3])
    # le parcours
    journal = P.parcourir(cle, R.traiter, maximum=300)
    echecs = [(d, e) for d, ok, e in journal if not ok]
    chk(f"{cle} : le parcours de l'écran ({len(journal)} requêtes) ne rencontre aucune erreur", not echecs, echecs[:3])
    # fenêtres et sélecteurs
    calculs = set(re.findall(r'data-calcul="([^"]+)"', r["resultat"]))
    chk(f"{cle} : chaque ligne qui annonce un calcul a sa fenêtre ({len(calculs)})",
        all(f'id="ck-{c}"' in r["volets"] for c in calculs), [c for c in calculs if f'id="ck-{c}"' not in r["volets"]])
    pks = list(dict.fromkeys(re.findall(r'data-pk="([^"]+)"', r["resultat"])))
    rep = [req(module=cle, action="selecteur", etat=etat, role=pk) for pk in pks]
    chk(f"{cle} : chaque sélecteur annoncé répond ({len(pks)})",
        all(x.get("ok") and ('class="pk-row' in x["html"] or "pk-vide" in x["html"]) for x in rep), [x.get("erreur") for x in rep if not x.get("ok")][:2])
    # l'échappement, le CSP
    textes = [c for c, s, _ in champs if s == "texte"]
    fuites = []
    for chemin in textes:
        out = req(module=cle, action="modifier", etat=copy.deepcopy(etat), chemin=chemin.split("/"), valeur=PIEGE)
        if out.get("ok"):
            fuites += [(chemin, k) for k, h in fragments(out).items() if "<img" in h]
    chk(f"{cle} : un texte saisi est échappé dans tous les fragments ({len(textes)} champ(s) texte)", not fuites, fuites[:3])
    scripts = [k for k, h in fragments(r).items() if EN_LIGNE.search(h)]
    chk(f"{cle} : aucun fragment ne porte de script ni de gestionnaire « on…= »", not scripts, scripts)
    # une note abîmée
    for abime in ({"x": 1}, {"elements": "abîmé"}, [1, 2], {"version": 99, "cas": None, "poutres": None}):
        out = json.loads(R.traiter(json.dumps({"module": cle, "action": "calculer", "etat": abime})))
        if not isinstance(out, dict) or "ok" not in out:
            chk(f"{cle} : une note abîmée rend une réponse", False, abime)
            break
    else:
        chk(f"{cle} : une note abîmée rend une réponse lisible (erreur ou note relue), jamais une exception", True)
    # les jauges : bornées à [0, 100] % (une largeur négative s'afficherait PLEINE)
    largeurs = [float(w) for w in re.findall(r'<div class="bar"><b class="\w*" style="width:(-?[0-9.]+)%"', r["resultat"])]
    chk(f"{cle} : chaque jauge du résultat est bornée à [0, 100] % ({len(largeurs)})", all(0 <= w <= 100 for w in largeurs),
        [w for w in largeurs if not 0 <= w <= 100])
    # les champs « vide = automatique » (C.num(auto=…)) : vidés, ils sont acceptés ; un texte qui n'est pas un nombre
    # est refusé avec un message, la note intacte
    autos = re.findall(r'<input data-chemin="([^"]+)" data-sorte="texte"[^>]*data-vide="auto"', r["donnees"])
    mal = []
    for chemin in autos:
        vide = req(module=cle, action="modifier", etat=copy.deepcopy(etat), chemin=chemin.split("/"), valeur="")
        faux = req(module=cle, action="modifier", etat=copy.deepcopy(etat), chemin=chemin.split("/"), valeur="pas un nombre")
        if not vide.get("ok") or vide.get("message") or not faux.get("ok") or not faux.get("message") or faux["etat"] != etat:
            mal.append((chemin, vide.get("message") or vide.get("erreur"), faux.get("message") or faux.get("erreur")))
    chk(f"{cle} : chaque champ « vide = automatique » accepte le vide et refuse un texte ({len(autos)})", not mal, mal[:3])
    # télécharger : chaque bouton annoncé (C.bouton_telecharger, data-telecharger) rend un fichier valide
    quois = list(dict.fromkeys(q for h in fragments(r).values() for q in re.findall(r'data-telecharger="([^"]*)"', h)))
    fichiers = [(q, req(module=cle, action="telecharger", etat=copy.deepcopy(etat), quoi=html.unescape(q))) for q in quois]
    mauvais = [(q, f.get("erreur") or C.refus_fichier(f.get("fichier")) or "le JSON ne se relit pas") for q, f in fichiers
               if not f.get("ok") or C.refus_fichier(f.get("fichier")) or not json_lisible(f["fichier"])]
    chk(f"{cle} : chaque bouton « télécharger » annoncé rend un fichier valide ({len(quois)})", not mauvais, mauvais[:3])
    if quois:
        chk(f"{cle} : le module qui annonce un téléchargement le déclare (telecharger)", callable(getattr(mod, "telecharger", None)))
    inconnu = req(module=cle, action="telecharger", etat=copy.deepcopy(etat), quoi="pas-un-fichier-connu")
    chk(f"{cle} : un téléchargement inconnu rend un refus lisible, jamais une exception",
        inconnu.get("ok") is False and isinstance(inconnu.get("erreur"), str) and "Traceback" not in inconnu["erreur"], inconnu)
    # les éléments : leur liste (« elements »), ou, dans l'établi compact, le cadre Données — l'arborescence d'avant
    # (prédalle servie pour les anciens liens) ou la colonne de gauche COMPACTE (la dalle, la poutre, 09/11/2026), dont la
    # liste des éléments porte un data-activer par élément, même seul, et leur nombre (data-elements) : les deux s'accordent
    def n_elements(rep):
        return rep["elements"].count('data-activer="') or rep["donnees"].count('data-activer="')

    def n_dit(rep):
        dit = re.search(r'<div class="et-elsel" data-elements="(\d+)">', rep["donnees"])
        return int(dit.group(1)) if dit else None
    a = req(module=cle, action="ajouter", etat=copy.deepcopy(etat))
    chk(f"{cle} : « ajouter » donne un élément de plus dans la liste",
        a.get("ok") and n_elements(a) == n_elements(r) + 1 >= 2, a.get("erreur") or (n_elements(a), n_elements(r)))
    if n_dit(r) is not None:
        chk(f"{cle} : la ligne de l'élément (colonne compacte) dit leur nombre (data-elements), autant que d'entrées dans sa liste — seul puis à deux",
            n_dit(r) == n_elements(r) == 1 and n_dit(a) == n_elements(a) == 2, (n_dit(r), n_elements(r), n_dit(a), n_elements(a)))
    s = req(module=cle, action="supprimer", etat=copy.deepcopy(etat), index=0)
    chk(f"{cle} : le dernier élément ne se supprime pas", s.get("ok") and n_elements(s) == n_elements(r))


choisis = sys.argv[1:] or REG.cles()
inconnus = [c for c in choisis if c not in REG.cles()]
chk("les modules demandés sont au registre", not inconnus, inconnus)
for cle in choisis:
    if cle in REG.cles():
        un_module(cle)

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:200])
sys.exit(1 if KO else 0)

# -*- coding: utf-8 -*-
"""L'API du site (api_web) : ce que l'écran du navigateur demande au moteur, vérifié en Python ordinaire.

- le résultat affiché est celui du moteur (taux, statut, élément gouvernant), arrondi sans quitter sa bande de couleur ;
- chaque champ du cadre Données pointe un endroit réel de l'état, et le réécrire avec sa propre valeur ne change rien ;
- les saisies ont les effets de bord de l'écran Streamlit (type → système et remplissage, catégorie → q_h) ;
- un nombre illisible est refusé sans toucher l'état ;
- le sélecteur est trié par masse, marque UN retenu et l'optimal du moteur (le plus léger qui tient) ;
- les éléments de la note s'ajoutent, se copient, se suppriment (jamais le dernier) ;
- toutes les combinaisons type × fixation × méthode se calculent sans erreur ;
- un nom saisi est échappé dans le HTML ; une requête fautive rend une erreur, jamais une exception.

    python3 tests/test_api_web.py
"""
import copy
import itertools
import json
import os
import re
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

from acier.garde_corps import charges as CH  # noqa: E402
from acier.garde_corps import etat as ET  # noqa: E402
from acier.garde_corps import types as TY  # noqa: E402
from api_web import commun as C  # noqa: E402
from api_web import garde_corps as G  # noqa: E402
from api_web import routeur as R  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:240]}"))


def req(**k):
    return json.loads(R.traiter(json.dumps(k)))


# 1. le résultat est celui du moteur
r = req(module="garde_corps", action="nouveau")
chk("nouveau : la requête aboutit", r["ok"], r.get("erreur"))
cas0 = ET.cas_actif(ET.nouvel_etat())
ref = ET.verifier(cas0)
chk("le taux affiché est celui du moteur, arrondi dans sa bande", r["resume"]["taux"] == C.pct(ref["taux_max"]), (r["resume"], ref["taux_max"]))
chk("le niveau de couleur est le statut du moteur", r["resume"]["niveau"] == C.niveau(ref["taux_max"]))
chk("l'élément gouvernant est celui du moteur", r["resume"]["gouvernant"].endswith(ref["gouvernant"]["libelle"]), r["resume"])
lignes = re.findall(r'<div class="el[^"]*"[^>]*><div class="l1"><b>([^<]+)</b>.*?<span class="pct (\w)">([^<]+)</span>', r["resultat"], re.S)
chk("une ligne de résultat par élément du moteur", len(lignes) == len(ref["elements"]), (len(lignes), len(ref["elements"])))
chk("chaque ligne porte le taux et la couleur de son élément",
    all(t == f"{C.pc(C.pct(e['taux']))}{C.NBSP}%" and n == C.niveau(e["taux"]) for (_, n, t), e in zip(lignes, ref["elements"])), lignes[:2])
chk("une seule ligne gouvernante (une vérification de l'établi compact : « el vc »)", r["resultat"].count('class="el vc gov"') == 1)
chk("les vues du dessin sont celles du moteur", [v["cle"] for v in r["vues"]] == [k for k, _, _ in __import__("acier.garde_corps.schemas", fromlist=["x"]).vues(ref)])
chk("une fenêtre de calcul pour chaque ligne qui en annonce une",
    all(f'id="ck-{c}"' in r["volets"] for c in re.findall(r'data-calcul="([^"]+)"', r["resultat"])))

# 2. chaque champ pointe l'état ; le réécrire avec sa propre valeur ne change rien
etat = r["etat"]
champs = re.findall(r'<input data-chemin="([^"]+)" data-sorte="(\w+)"[^>]*?value="([^"]*)"', r["donnees"])
champs += [(c, "choix", v) for c, v in re.findall(r'<select data-chemin="([^"]+)" data-sorte="choix"[^>]*>.*?<option value="([^"]*)" selected', r["donnees"], re.S)]
chk(f"le cadre Données porte des champs ({len(champs)})", len(champs) > 20, len(champs))
mauvais = []
for chemin, sorte, valeur in champs:
    e2 = copy.deepcopy(etat)
    msg = G.modifier(e2, chemin.split("/"), valeur.replace("&amp;", "&") if sorte != "case" else valeur)
    avant, apres = ET.verifier(ET.cas_actif(etat))["taux_max"], ET.verifier(ET.cas_actif(e2))["taux_max"]
    if msg or abs(avant - apres) > 1e-12:
        mauvais.append((chemin, valeur, msg, avant, apres))
chk("réécrire chaque champ avec sa valeur affichée ne change pas le résultat", not mauvais, mauvais[:3])

# 3. les effets de bord de l'écran Streamlit
e = copy.deepcopy(etat)
G.modifier(e, ["type"], "lisses")
c = ET.cas_actif(e)
chk("le type pose le système et le remplissage (TY.appliquer_type)",
    c["type"] == "lisses" and c["remplissage"]["type"] == TY.TYPES["lisses"]["remplissage"] and c["geometrie"]["systeme"] == TY.TYPES["lisses"]["systeme"])
G.modifier(e, ["charges", "categorie"], "C5")
chk("la catégorie pose q_h (CH.normaliser)", ET.cas_actif(e)["charges"]["qh"] == CH.qh_de_categorie("C5"), ET.cas_actif(e)["charges"]["qh"])
G.modifier(e, ["geometrie", "H"], "1 250,5")
chk("« 1 250,5 » se lit 1250,5", ET.cas_actif(e)["geometrie"]["H"] == 1250.5)
avant = copy.deepcopy(e)
msg = G.modifier(e, ["geometrie", "H"], "douze")
chk("un nombre illisible est refusé, l'état reste intact", msg is not None and e == avant, msg)
G.modifier(e, ["remplissage", "n_lisses"], "3")
chk("un champ entier reste un entier", ET.cas_actif(e)["remplissage"]["n_lisses"] == 3 and isinstance(ET.cas_actif(e)["remplissage"]["n_lisses"], int))
G.modifier(e, ["geometrie", "troncons", "0", "entraxes"], "1200 900; 1500")
chk("les entraxes saisis se lisent en liste", ET.cas_actif(e)["geometrie"]["troncons"][0]["entraxes"] == [1200.0, 900.0, 1500.0])
G.modifier(e, ["charges", "mode_P"], True)
chk("la case « charge P par montant » pose le mode P", ET.cas_actif(e)["charges"]["mode"] == "P")

# 4. le sélecteur
s = req(module="garde_corps", action="selecteur", etat=etat, role="montant")
rows = re.findall(r'<button class="pk-row( cur)?( opt)?" data-choix="([^"]+)".*?<span class="kg">([^<]+)</span>', s["html"], re.S)
masses = [float(m.replace(" ", "").replace(",", ".")) for *_, m in rows]
chk(f"le sélecteur du montant liste des sections ({len(rows)})", len(rows) > 20, len(rows))
chk("trié par masse croissante", masses == sorted(masses), masses[:6])
chk("un seul retenu, et c'est la section du cas", [n for cur, _, n, _ in rows if cur] == [ET.cas_actif(etat)["montant"]["section"]["nom"]])
cands = sorted((c for c in ET.candidats(ET.cas_actif(etat), "montant", ["creux", "froid"], 25.0) if c["section"]["famille"].replace("-F", "") == "RHS"),
               key=lambda c: c["masse"])
best = ET.plus_leger(cands)
chk("l'étoile est l'optimal du moteur (le plus léger qui tient)", [n for _, opt, n, _ in rows if opt] == ([best["section"]["nom"]] if best else []),
    ([n for _, opt, n, _ in rows if opt], best and best["section"]["nom"]))
r2 = req(module="garde_corps", action="choisir", etat=etat, role="montant", nom="RHS 60x40x3")
chk("choisir une section la retient et recalcule", r2["ok"] and ET.cas_actif(r2["etat"])["montant"]["section"]["nom"] == "RHS 60x40x3")

# 5. les éléments de la note
e = copy.deepcopy(etat)
G.gerer_elements(e, "ajouter")
G.gerer_elements(e, "dupliquer")
chk("ajouter puis copier : trois garde-corps, le dernier ouvert", len(e["cas"]) == 3 and e["actif"] == 2)
G.gerer_elements(e, "supprimer", 0)
chk("supprimer le premier garde l'élément ouvert", len(e["cas"]) == 2 and e["actif"] == 1)
# le résumé de la NOTE est son élément le plus chargé, même s'il n'est pas ouvert
G.modifier(e, ["charges", "categorie"], "C5")
e["actif"] = 0
rn = json.loads(R.traiter(json.dumps({"module": "garde_corps", "action": "calculer", "etat": e})))
t_max = max(ET.verifier(c)["taux_max"] for c in e["cas"])
chk("le résumé de la note est son élément le plus chargé (ici l'élément fermé)", rn["resume"]["taux"] == C.pct(t_max) and rn["resume"]["niveau"] == "r",
    (rn["resume"], t_max))
G.gerer_elements(e, "supprimer", 0)
G.gerer_elements(e, "supprimer", 0)
chk("le dernier garde-corps ne se supprime pas", len(e["cas"]) == 1)

# 6. toutes les combinaisons se calculent
fautes = []
for typ, fix, meth in itertools.product(TY.TYPES, ("platine", "laterale", "soudee", "scellee"), ("els", "eurocode")):
    e = G.nouvel_etat()
    G.modifier(e, ["type"], typ)
    G.modifier(e, ["fixation", "type"], fix)
    G.modifier(e, ["methode", "mode"], meth)
    out = json.loads(R.traiter(json.dumps({"module": "garde_corps", "action": "calculer", "etat": e})))
    if not out["ok"]:
        fautes.append((typ, fix, meth, out.get("erreur")))
chk(f"les {len(TY.TYPES) * 8} combinaisons type × fixation × méthode se calculent", not fautes, fautes[:3])

# 7. l'échappement, les erreurs
e = copy.deepcopy(etat)
G.modifier(e, ["nom"], '<script>alert("x")</script>')
rr = req(module="garde_corps", action="calculer", etat=e)
h = rr["donnees"] + rr["resultat"]
chk("un nom saisi est échappé dans le HTML (la ligne de l'élément, la tête des résultats)", "<script>" not in h and h.count("&lt;script&gt;") >= 2)
x = req(module="inconnu", action="calculer")
chk("un module inconnu rend une erreur lisible", x["ok"] is False and "inconnu" in x["erreur"], x)
x = json.loads(R.traiter("pas du json"))
chk("une requête illisible rend une erreur, pas une exception", x["ok"] is False)

# 8. l'établi COMPACT (la mise en forme de la poutre, 11/11/2026) : la colonne de gauche, les blocs, les fenêtres, les vues
d, res_h = r["donnees"], r["resultat"]
chk("à gauche : la colonne compacte — la ligne du garde-corps (son nom s'y modifie, à cet endroit seulement), six rubriques, la dernière "
    "(« Hypothèses ») secondaire ; plus de bloc « Dans cette note », plus de phrase au-dessus du dessin",
    'class="et-arbre et-arbre-c" data-gardecorps="0"' in d and d.count('data-chemin="nom"') == 1 and len(re.findall(r'<div class="sec( open)?" data-sec=', d)) == 6
    and re.findall(r'data-sec="(\w+)"( data-secondaire="1")?', d) == [("gc", ""), ("tr", ""), ("rp", ""), ("ch", ""), ("fx", ""), ("me", ' data-secondaire="1"')]
    and r["elements"] == "" and r["phrase"] == "", re.findall(r'data-sec="(\w+)"( data-secondaire="1")?', d))
chk("à droite : la tête porte le nom, le type et la géométrie du moteur",
    f'<div class="et-rp"><b>G1</b><span>{C.esc(ref["type_libelle"])} · {C.esc(ref["geometrie"]["phrase"])}</span></div>' in res_h)
blocs_h = re.findall(r'<section class="et-bloc (\w+)" data-theme="(\d)">.*?<span class="n">([^<]+)</span><span class="e">([^<]+)</span>(.*?)</section>', res_h, re.S)
par_bloc = {0: [e for e in ref["elements"] if e["cle"] in ET.ROLES], 1: [e for e in ref["elements"] if e["cle"] not in ET.ROLES]}
chk("deux blocs : « Profilés » (les pièces qui se choisissent) et « Fixation » ; chacun porte l'état et le taux de sa ligne la plus chargée, "
    "et autant de lignes que le moteur a d'éléments",
    [(b[1], b[2]) for b in blocs_h] == [("0", "Profilés"), ("1", "Fixation")]
    and all(b[0] == {"p": "ok", "f": "warn", "r": "nok"}[C.niveau(max(e["taux"] for e in par_bloc[int(b[1])]))]
            and b[3] == f"{C.STATUTS[C.niveau(max(e['taux'] for e in par_bloc[int(b[1])]))][1]} · {C.pc(C.pct(max(e['taux'] for e in par_bloc[int(b[1])])))}{C.NBSP}%"
            and b[4].count('class="el vc') == len(par_bloc[int(b[1])]) for b in blocs_h), [(b[0], b[1], b[2], b[3]) for b in blocs_h])
cmps = re.findall(r'<span class="cmp">(.*?)</span><div class="bar"', res_h)
dm = ref["elements"][0]["detail"]
chk("chaque vérification dit ce qu'elle compare, en clair : pour le montant, sa flèche face à sa limite (les nombres du moteur) et son cas de charge",
    len(cmps) == len(ref["elements"]) and cmps[0] == f"Flèche {C.fr(dm['fleche'], 2)} ≤ {C.fr(dm['fleche_lim'], 1)} mm · cas 1" and cmps[1].startswith("σ<sub>éq</sub> ")
    and all(" ≤ " in c for c in cmps),
    cmps[:2])
fen = re.findall(r'id="ck-(s\d)".*?<tbody>(.*?)</tbody>', r["volets"], re.S)
chk("« tout le calcul » d'un bloc a sa fenêtre (s0, s1) : toutes les vérifications du bloc, une seule gouvernante en tête",
    [f[0] for f in fen] == ["s0", "s1"] and all(f[1].count("<tr") >= len(par_bloc[i]) and f[1].count('class="gov"') == 1
                                                 and f[1].index('class="gov"') < f[1].index("</tr>") for i, f in enumerate(fen)),
    [(f[0], f[1].count("<tr")) for f in fen])
chk("chaque vue du dessin porte son en-tête technique : le type, l'acier, la méthode ; la géométrie ; les charges",
    all(v["entete"].startswith('<div class="et-ent">') and C.esc(ref["type_libelle"]) in v["entete"] and C.esc(ref["geometrie"]["phrase"]) in v["entete"]
        and "qh = 0,5 kN/m" in v["entete"] and v["titre"].upper() in v["entete"] for v in r["vues"]), r["vues"][0]["entete"][:300])
e = G.nouvel_etat()
G.modifier(e, ["fixation", "type"], "scellee")
rs = req(module="garde_corps", action="calculer", etat=e)
chk("montant scellé : la fixation n'est pas vérifiée — pas de bloc « Fixation », pas de fenêtre s1, et aucune ligne n'annonce un calcul absent",
    'data-theme="1"' not in rs["resultat"] and 'id="ck-s1"' not in rs["volets"] and 'data-theme="0"' in rs["resultat"]
    and all(f'id="ck-{c}"' in rs["volets"] for c in re.findall(r'data-calcul="([^"]+)"', rs["resultat"])))
e = G.nouvel_etat()
G.gerer_elements(e, "ajouter")
G.gerer_elements(e, "ajouter")
G.gerer_elements(e, "dupliquer", 0)
chk("copier un garde-corps (bouton de sa ligne) : la copie s'insère juste après lui et s'ouvre",
    [c["nom"] for c in e["cas"]] == ["G1", "G1 (copie)", "G2", "G3"] and e["actif"] == 1, [c["nom"] for c in e["cas"]])
G.gerer_elements(e, "supprimer", 3)
G.gerer_elements(e, "ajouter")
chk("un garde-corps neuf porte un nom court, libre dans la note (G1 à l'ouverture, puis G2, G3… : il tient entier dans la ligne de l'élément)",
    G.nouvel_etat()["cas"][0]["nom"] == "G1" and [c["nom"] for c in e["cas"]] == ["G1", "G1 (copie)", "G2", "G4"], [c["nom"] for c in e["cas"]])
rn3 = req(module="garde_corps", action="calculer", etat=e)
chk("une note de plusieurs garde-corps : la liste de la ligne les nomme tous (nom, type, longueur), le statut de la note est dit ; seul, il ne l'est pas",
    rn3["donnees"].count('class="et-elt-i') == 4 and 'data-elements="4"' in rn3["donnees"] and "Barreaudage vertical · 3,60" in rn3["donnees"]
    and rn3["statut_html"] != "" and r["statut_html"] == "")

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:200])
sys.exit(1 if KO else 0)

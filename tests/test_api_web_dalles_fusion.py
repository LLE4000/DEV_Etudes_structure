# -*- coding: utf-8 -*-
"""Le module FUSIONNÉ « Dalles béton armé » (api_web/dalle.py — docs/site/DALLES_FUSION.md § 0 et § 2, lot FUS2),
vérifié en Python ordinaire à travers le routeur (les mêmes requêtes JSON que le navigateur).

- l'ÉTAT fusionné : chaque dalle porte « type » (coulee | predalle), « verrou », « empilement » ; une note neuve est une
  dalle coulée en place, nappes empilées ; les trois formes d'état sont relues ({"dalles"} sans type → coulee,
  {"predalles"} de l'ancien module → predalle, actif conservé, chiffres identiques) et une note abîmée (les deux clés,
  un type inconnu, un drapeau qui n'en est pas un) est REFUSÉE, jamais remplacée par une note neuve ;
- la rubrique « 1 — Caractéristiques » : les segments du type, « h_p prédalle » et le complément h − h_p (lu, jamais
  stocké) ; la bascule coulee ↔ predalle ajoute ou retire h_pre et conserve tout le reste ; les refus géométriques de
  l'ancien module Prédalles s'appliquent, la dalle reste intacte ;
- le MOTEUR est choisi selon le type (beton/dalle ou beton/predalle, inchangés) : les lignes du résultat sont celles de
  l'appel direct du moteur du type sur la même entrée ; l'en-tête, le pied, les hypothèses, le périmètre, les
  avertissements et les dessins sont ceux du type ;
- l'import des DEUX fichiers Streamlit (dalle_ba.json, predalle_ba.json : le type déduit du fichier) ;
- « empilement » : vrai pour une note neuve, faux pour une note relue sans la clé, une note de prédalles, un fichier
  importé ; commun à la note ; option fausse → les chiffres du moteur à l'octet ;
- l'ampoule (sélecteur de nappe de base) garde une nappe de treillis unique cohérente (la transposée sur l'autre face) ;
- les points d'ancrage des autres lots sont branchés (dalle_positions : axe_auto, pour_moteur, annoter, candidats_base ;
  dalle_nappes : ecrire, html ; etabli_compact : compact=True, le nom dans la ligne de la dalle) ;
- cinq MUTATIONS : une altération du code fait rougir le contrôle qui la garde.

    /usr/bin/python3.12 tests/test_api_web_dalles_fusion.py
"""
import copy
import json
import os
import re
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

from api_web import commun as C  # noqa: E402
from api_web import dalle as P  # noqa: E402
from api_web import dalle_nappes as NAP  # noqa: E402
from api_web import dalle_positions as POS  # noqa: E402
from api_web import predalle as AP  # noqa: E402
from api_web import routeur as R  # noqa: E402
from beton import treillis as TR  # noqa: E402
from beton.dalle import entree as EN  # noqa: E402
from beton.dalle import moteur as MO  # noqa: E402
from beton.predalle import entree as EN_P  # noqa: E402
from beton.predalle import moteur as MO_P  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:300]}"))


def req(**k):
    return json.loads(R.traiter(json.dumps(dict(module="dalle", **k))))


def saisir(etat, chemin, valeur):
    """Comme l'écran : le chemin lu dans data-chemin, coupé aux « / », la valeur telle que tapée ; la requête aboutit."""
    out = req(action="modifier", etat=etat, chemin=chemin.split("/"), valeur=valeur)
    assert out["ok"], out
    return out


def actif(etat, i=None):
    return etat["dalles"][etat["actif"] if i is None else i]


def sans_site(d):
    """La dalle sans les clés du site : ce qu'un moteur lit quand les nappes ne sont pas empilées."""
    return {k: v for k, v in d.items() if k not in P.CLES_SITE}


def niveau_de(v):
    return "r" if v["etat"] == "nok" else C.niveau(v["taux"])


# une ligne de vérification (C.ligne_verif) : (gouvernante, section, nom, niveau, taux)
LIGNE = re.compile(r'<div class="el vc( gov)?" data-calcul="(s\d+)-\w+"[^>]*><div class="l1"><b>([^<]+)</b>.*?<span class="pct (\w)"[^>]*>([^<]+)</span></div>', re.S)


def lignes(rep):
    return [(c, n, niv, t) for _, c, n, niv, t in LIGNE.findall(rep["resultat"])]


def lignes_moteur(r, M):
    """Ce que le moteur dit, écrit comme le site l'écrit : (section, nom, couleur, taux) par vérification."""
    return [(f"s{i}", M.nom_ligne(v), niveau_de(v), f"{C.pc(C.pct(v['taux']))}{C.NBSP}%")
            for i, s in enumerate(r["sections"]) for v in s["verifications"]]


def note_predalle():
    """Une note de l'ANCIEN module Prédalles (clé « predalles », sans « type » ni « empilement »), chargée comme la note des
    tests de l'écran : Mx = My = 30, sup 12 / 9, V 60, un renfort 3 Ø12 en inf. Y, et une section d'appui."""
    p = EN_P.nouvelle_predalle()
    s = p["sections"][0]
    s.update({"Mx_inf": 30.0, "My_inf": 30.0, "Mx_sup": 12.0, "My_sup": 9.0, "V": 60.0})
    s["faces"]["inf_y"]["couches"].append(EN_P.nouvelle_couche(base=False))
    b = EN_P.nouvelle_section("B")
    b.update({"Mx_sup": 22.0, "My_sup": 30.0, "V": 85.0})
    p["sections"].append(b)
    return {"version": 1, "predalles": [dict(p, verrou=False)], "actif": 0}


# ------------------------------------------------------------------ 1. la note neuve et l'état fusionné
print("=== 1. La note neuve, l'état fusionné, les clés du site ===")
r0 = req(action="nouveau")
e0 = r0["etat"]
d0 = actif(e0)
chk("une note neuve : version 2, UNE dalle « coulee », déverrouillée, nappes EMPILÉES (§ 4.3), la dalle de l'écran Streamlit",
    r0["ok"] and e0["version"] == 2 and len(e0["dalles"]) == 1 and d0["type"] == "coulee" and d0["verrou"] is False
    and d0["empilement"] is True and sans_site(d0) == EN.nouvelle_dalle(), (e0.get("version"), {k: d0.get(k) for k in P.CLES_SITE}))
chk("les clés du site sont nommées une fois par module et liées ici : dalle.CLES_SITE = dalle_positions.CLES_SITE, et la copie "
    "envoyée au moteur (POS.pour_moteur) n'en porte aucune — le moteur des dalles la lit",
    set(P.CLES_SITE) == set(POS.CLES_SITE) == {"verrou", "type", "empilement"}
    and not (set(POS.pour_moteur(d0)) & set(P.CLES_SITE)) and MO.verifier(POS.pour_moteur(d0))["statut"] == MO.OK)
chk("l'état survit à un aller-retour JSON et sa relecture est stable (normaliser ∘ normaliser)",
    P.normaliser(json.loads(json.dumps(e0))) == e0 and P.normaliser(copy.deepcopy(e0)) == e0)
chk("le type et l'empilement se lisent dans la rubrique 1 et les options : segments « Dalle CSP » (retenu) · « Prédalle + dalle "
    "CSP » (les libellés courts de l'ingénieur, 09/11/2026 ; l'aide du champ dit « CSP : coulé sur place. » d'abord), le résumé "
    "« Dalle CSP · C30/37 · B500 · c = 3 cm », la case « Nappes empilées » cochée, aucune peau",
    P.TYPES_DALLE == (("coulee", "Dalle CSP"), ("predalle", "Prédalle + dalle CSP"))
    and re.search(r'<button type="button" class="on" data-chemin="type" data-sorte="choix" data-valeur="coulee">Dalle CSP</button>'
                  r'<button type="button" class="" data-chemin="type" data-sorte="choix" data-valeur="predalle">Prédalle \+ dalle CSP</button>',
                  r0["donnees"]) is not None
    and P.AIDE_TYPE.startswith("CSP : coulé sur place. ") and f'<span class="aide" title="{C.esc(P.AIDE_TYPE)}">' in r0["donnees"]
    and '<input type="checkbox" data-chemin="empilement" data-sorte="case" checked>' in r0["donnees"] and 'data-chemin="h_pre"' not in r0["donnees"]
    and "Dalle CSP · C30/37 · B500 · c = 3 cm" in r0["donnees"], P.TYPES_DALLE)

# ------------------------------------------------------------------ 2. les trois formes d'état
print("\n=== 2. Les trois formes d'état relues ; la note abîmée refusée ===")
ra = req(action="calculer", etat={"version": 1, "dalles": [{"nom": "Ancienne", "h": 18}], "actif": 0})
da = actif(ra["etat"])
chk("{« dalles »} d'avant, sans type : relue coulée en place, nappes NON empilées (ses chiffres ne bougent pas), version 2",
    ra["ok"] and da["type"] == "coulee" and da["empilement"] is False and da["h"] == 18 and ra["etat"]["version"] == 2, ra.get("erreur"))
NP = note_predalle()
NP["predalles"].append(dict(EN_P.nouvelle_predalle(), nom="Appui", h=24, h_pre=7.0))
NP["actif"] = 1
rp = req(action="calculer", etat=NP)
chk("{« predalles »} de l'ancien module : chaque prédalle devient une dalle de type « predalle », h_pre gardée, actif conservé, "
    "empilement faux — rien n'est perdu",
    rp["ok"] and [d["type"] for d in rp["etat"]["dalles"]] == ["predalle", "predalle"] and rp["etat"]["actif"] == 1
    and actif(rp["etat"])["h_pre"] == 7.0 and actif(rp["etat"])["nom"] == "Appui"
    and all(d["empilement"] is False for d in rp["etat"]["dalles"])
    and sans_site(rp["etat"]["dalles"][0]) == EN_P.normaliser(sans_site(NP["predalles"][0])) if rp["ok"] else False, rp.get("erreur"))
vieux = json.loads(R.traiter(json.dumps(dict(module="predalle", action="calculer", etat=copy.deepcopy(NP)))))
chk("… et ses CHIFFRES sont ceux de l'ancien module Prédalles : mêmes lignes (nom, couleur, taux), même résumé, pour les deux prédalles",
    all(lignes(req(action="calculer", etat=dict(copy.deepcopy(NP), actif=k)))
        == lignes(json.loads(R.traiter(json.dumps(dict(module="predalle", action="calculer", etat=dict(copy.deepcopy(NP), actif=k))))))
        for k in (0, 1)) and rp["resume"] == vieux["resume"], (rp["resume"], vieux["resume"]))
ABIMEES = [({"dalles": [{}], "predalles": [{}]}, "abîmée"), ({"predalles": "abc"}, "liste"), ({"predalles": [{"type": "coulee"}]}, "note abîmée"),
           ({"dalles": [{"type": "autre"}]}, "coulee » ou « predalle"), ({"dalles": [{"empilement": "oui"}]}, "empilement vrai ou faux"),
           ({"dalles": [{"h_pre": 5}]}, "h_pre"), ({"dalles": [{"type": "predalle", "h_pre": 30}]}, "h_pre")]
lues = [(req(action="calculer", etat=x), mot) for x, mot in ABIMEES]
chk("une note abîmée — les DEUX clés, « predalles » qui n'est pas une liste, un type contradictoire ou inconnu, un drapeau qui n'en "
    "est pas un, une peau sur une dalle coulée, une peau impossible — est REFUSÉE avec une erreur claire, jamais remplacée en silence",
    all(not r_["ok"] and r_["erreur"].startswith("ValueError") and mot in r_["erreur"] for r_, mot in lues),
    [(mot, r_.get("erreur", "ok ?!")[:120]) for r_, mot in lues if r_["ok"] or mot not in r_.get("erreur", "")])
chk("une note sans dalle ({« predalles »: []}, {}) est une note neuve ; une dalle « type predalle » dans « dalles » se relit prédalle",
    req(action="calculer", etat={"predalles": []})["etat"] == e0 and req(action="calculer", etat={})["etat"] == e0
    and actif(req(action="calculer", etat={"dalles": [{"type": "predalle", "h_pre": 5}]})["etat"])["h_pre"] == 5.0)

# ------------------------------------------------------------------ 3. la bascule coulee ↔ predalle
print("\n=== 3. La bascule coulée ↔ prédalle ===")
E = saisir(e0, "empilement", False)["etat"]
for ch, x in (("nom", "Ma dalle"), ("sections/0/Mx_inf", "28"), ("sections/0/My_inf", "35"), ("sections/0/V", "60"),
              ("sections/0/faces/inf_y/couches/+", "1"), ("sections/0/faces/inf_y/couches/1/dist", "8"), ("sections/0/faces/sup_x/yG", "4,2"),
              ("sections/+", "1"), ("sections/1/My_sup", "24")):
    E = saisir(E, ch, x)["etat"]
dc = actif(E)
bp = saisir(E, "type", "predalle")
dp = actif(bp["etat"])
chk("coulee → predalle : « h_pre » ajoutée (6 cm, le défaut de l'écran prédalle), le type écrit, tout le reste conservé — nom, "
    "sections, renfort posé à 8 cm, y_G imposé",
    not bp["message"] and dp["type"] == "predalle" and dp["h_pre"] == EN_P.DEFAUTS_PREDALLE["h_pre"] == 6.0
    and {k: v for k, v in dp.items() if k != "h_pre"} == dict(dc, type="predalle") and dp["nom"] == "Ma dalle"
    and dp["sections"][0]["faces"]["inf_y"]["couches"][1]["dist"] == 8.0 and dp["sections"][0]["faces"]["sup_x"]["yG"] == 4.2, bp["message"])
chk("… la rubrique 1 dit le type (« Prédalle h_p = 6 cm »), montre h_p (bornes 3 à 15 cm pour h = 20) et le complément 14 cm ; "
    "l'en-tête des résultats dit « 100 × 20 cm, prédalle 6 cm »",
    "Prédalle h<sub>p</sub> = 6 cm" in bp["donnees"]
    and 'data-chemin="h_pre" data-sorte="nombre" inputmode="decimal" value="6" data-min="3.0" data-max="15.0"' in bp["donnees"]
    and '<span class="dl-ro">14 cm</span>' in bp["donnees"] and "<span>100 × 20 cm, prédalle 6 cm · C30/37 · B500</span>" in bp["resultat"],
    re.findall(r'data-chemin="h_pre"[^>]*>', bp["donnees"]))
bc = saisir(bp["etat"], "type", "coulee")
chk("predalle → coulee : « h_pre » retirée, la dalle redevient EXACTEMENT celle d'avant la bascule",
    not bc["message"] and actif(bc["etat"]) == dc and "h_pre" not in actif(bc["etat"]), bc["message"])
mince = saisir(saisir(e0, "h", "5")["etat"], "type", "predalle")
chk("coulee → predalle sur une dalle de 5 cm : h_pre bornée à h_pre_max(5) = 4 cm (au moins 1 cm coulé en place)",
    not mince["message"] and actif(mince["etat"])["h_pre"] == 4.0 == EN_P.h_pre_max(5), mince["message"])
meme = saisir(bp["etat"], "type", "predalle")
o = req(action="modifier", etat=bp["etat"], chemin=["type"], valeur="autre")
chk("le type réécrit à l'identique ne change rien ; un type inconnu est refusé avec un message, la dalle intacte",
    meme["etat"] == bp["etat"] and o["message"] and "coulee » ou « predalle" in o["message"] and o["etat"] == bp["etat"], o.get("message"))
aj = req(action="ajouter", etat=bp["etat"])
dj = actif(aj["etat"])
chk("« Ajouter une dalle » sur une note ouverte sur une prédalle : une « Prédalle 2 » de type predalle (h = 22, h_pre = 6), avec le γs, "
    "le jeu et l'empilement de la note ; « Copier » garde le type",
    dj["type"] == "predalle" and dj["nom"] == "Prédalle 2" and dj["h"] == 22 and dj["h_pre"] == 6.0 and dj["empilement"] is False
    and actif(req(action="dupliquer", etat=bp["etat"])["etat"])["type"] == "predalle", dj.get("nom"))

# ------------------------------------------------------------------ 4. h_pre, le complément, les refus géométriques
print("\n=== 4. La peau, le complément lu, les refus géométriques ===")
PR = saisir(saisir(e0, "empilement", False)["etat"], "type", "predalle")["etat"]
PR = saisir(PR, "h", "22")["etat"]
r75 = saisir(PR, "h_pre", "7,5")
chk("h_pre 7,5 cm sur h = 22 : stockée ; le complément 14,5 cm est AFFICHÉ, jamais stocké (aucune clé de plus dans l'état) ; le moteur "
    "prédalle pose la base secondaire à 8,0 et le renfort… à 8,5 comme l'ancien module",
    actif(r75["etat"])["h_pre"] == 7.5 and '<span class="dl-ro">14,5 cm</span>' in r75["donnees"]
    and set(actif(r75["etat"])) == set(actif(PR)) and "coule" not in json.dumps(r75["etat"])
    and MO_P.verifier(sans_site(actif(r75["etat"])))["sections"][0]["faces"]["inf_x"]["couches"][0]["e"] == 8.0)
r30 = saisir(r75["etat"], "h", "30")
chk("h 30 cm : le complément suit (22,5 cm) et la borne haute de h_p aussi (15 cm) ; h 10 cm avec h_pre 7,5 : borne 9 cm",
    '<span class="dl-ro">22,5 cm</span>' in r30["donnees"] and 'data-chemin="h_pre" data-sorte="nombre" inputmode="decimal" value="7,5" data-min="3.0" data-max="15.0"' in r30["donnees"]
    and 'data-max="9.0"' in saisir(r75["etat"], "h", "10")["donnees"])
REFUS = [("h_pre", "21", "Épaisseur de la prédalle : entre 3 et 15 cm"), ("h_pre", "2,9", "Épaisseur de la prédalle"), ("h_pre", "15,5", "Épaisseur de la prédalle"),
         ("h_pre", "abc", "illisible"), ("h", "6", "au moins 7 cm"), ("h", "4", "entre 5 et 100")]
mauvais = []
for ch, x, mot in REFUS:
    o = req(action="modifier", etat=copy.deepcopy(PR), chemin=[ch], valeur=x)
    if not (o["ok"] and o["message"] and mot in o["message"] and o["etat"] == PR):
        mauvais.append((ch, x, o.get("message") or o.get("erreur")))
chk(f"{len(REFUS)} refus géométriques de l'ancien module Prédalles (peau hors bornes, h totale qui ne contient plus la peau, peau "
    "illisible) : un message clair, la dalle INTACTE", not mauvais, mauvais)
o = req(action="modifier", etat=copy.deepcopy(e0), chemin=["h_pre"], valeur="5")
chk("h_pre sur une dalle coulée en place : refusée, avec le chemin à suivre (choisir le type prédalle d'abord : le message nomme le "
    "segment par SON libellé, TYPES_TXT — deux copies du même mot, liées ici), la dalle intacte",
    o["message"] and "n'a pas de prédalle" in o["message"] and f"choisir « {P.TYPES_TXT['predalle']} » d'abord" in o["message"]
    and o["etat"] == e0, o.get("message"))
peau = saisir(saisir(PR, "h_pre", "3")["etat"], "sections/0/faces/inf_y/couches/0/treillis", "12/12/150/150")
chk("base Ø12 à 4,0 cm dans une peau de 3 cm : l'avertissement GRAVE du moteur prédalle (« sort de la peau ») est rendu",
    '<div class="msg grave">' in peau["resultat"] and "sort de la peau" in peau["resultat"])

# ------------------------------------------------------------------ 5. le moteur appelé selon le type
print("\n=== 5. Le moteur appelé selon le type : identique à l'appel direct ===")
NP1 = note_predalle()
rep_p = req(action="calculer", etat=NP1)
dp1 = actif(rep_p["etat"])
m_p = MO_P.verifier(sans_site(dp1))
m_d = MO.verifier({k: v for k, v in sans_site(dp1).items() if k != "h_pre"})
chk("type predalle (option d'empilement fausse) : chaque ligne du résultat — nom, couleur, taux — est celle de beton.predalle.moteur.verifier "
    "sur la même entrée ; le résumé aussi ; et ce N'EST PAS ce que dirait le moteur des dalles (base secondaire à 6,5 et non 3,5)",
    lignes(rep_p) == lignes_moteur(m_p, AP) and rep_p["resume"]["taux"] == C.pct(m_p["taux_max"])
    and lignes_moteur(m_p, AP) != lignes_moteur(m_d, P)
    and m_p["sections"][0]["faces"]["inf_x"]["couches"][0]["e"] == 6.5 and m_d["sections"][0]["faces"]["inf_x"]["couches"][0]["e"] == 3.5,
    (lignes(rep_p)[:3], lignes_moteur(m_p, AP)[:3]))
chk("… les chiffres de l'écran Streamlit prédalle (tests/test_api_web_predalle.py) : As,req 573 mm² (principale) et 645 mm² (secondaire)",
    round(m_p["sections"][0]["grandeurs"]["As_req_inf_y"]) == 573 and round(m_p["sections"][0]["grandeurs"]["As_req_inf_x"]) == 645
    and "573" in rep_p["volets"] and "645" in rep_p["volets"])
EC = E
rep_c = req(action="calculer", etat=EC)
m_c = MO.verifier(sans_site(actif(EC)))
chk("type coulee (option fausse) : chaque ligne est celle de beton.dalle.moteur.verifier sur la même entrée",
    lignes(rep_c) == lignes_moteur(m_c, P) and rep_c["resume"]["taux"] == C.pct(m_c["taux_max"]), lignes(rep_c)[:3])
chk("le pied dit le moteur et sa version : « moteur prédalle » v1.1 pour une prédalle, « moteur dalle » v2.2 pour une dalle coulée",
    "moteur prédalle</span><b>v1.1</b>" in rep_p["resultat"] and "moteur dalle</span><b>v2.2</b>" in rep_c["resultat"])
chk("les hypothèses et les formules sont celles du type : « dont prédalle 6 cm et 16 cm coulés en place », l'axe e₁ « c + ⌈Ø/2⌉ … dans la "
    "prédalle » ; rien de tel pour une dalle coulée",
    "dont prédalle <b>6 cm</b> et 16 cm coulés en place" in rep_p["volets"] and "dans la prédalle" in rep_p["volets"]
    and "dont prédalle" not in rep_c["volets"] and "dans la prédalle" not in rep_c["volets"])
chk("le PÉRIMÈTRE de la prédalle (section finale seule ; phase provisoire, interface, flèche non vérifiés) est dit sous le résultat et dans "
    "chaque fenêtre de calcul ; absent d'une dalle coulée",
    "phase provisoire" in rep_p["resultat"] and all("phase provisoire" in x for x in rep_p["volets"].split('<div class="flyout calcul"')[1:])
    and "phase provisoire" not in rep_c["resultat"] + rep_c["volets"])
chk("les AVERTISSEMENTS propres à la prédalle sont présents (renfort Y à 7,0 et base X à 6,5 « posées toutes deux sur la prédalle », les bases "
    "supérieures « au même niveau »), dans le bloc de leur section",
    "posées toutes deux sur la prédalle" in rep_p["resultat"] and "axe 7,0 cm" in rep_p["resultat"]
    and "les bases X et Y supérieures sont calculées au même niveau" in rep_p["resultat"]
    and "posées toutes deux" in re.search(r'<section class="et-bloc \w+" data-section="0">.*?</section>', rep_p["resultat"], re.S).group(0))
chk("les données d'une prédalle disent la POSE de chaque couche (pictogramme et règle : dans la prédalle, sur la prédalle, depuis le dessus) ; "
    "une dalle coulée n'en a pas",
    sorted(set(re.findall(r'<svg class="pd-m pd-(\w+)"', rep_p["donnees"]))) == ["predalle", "superieure", "sur_predalle"]
    and "axe auto · dans la prédalle (c + ½ Ø)" in rep_p["donnees"] and "pd-m" not in rep_c["donnees"])
chk("le sélecteur d'une prédalle dit l'axe de chaque candidat et sa pose (Ø20 dans la peau : 4,0 cm) ; celui d'une dalle coulée, non",
    "axe 4,0 cm (dans la prédalle)" in req(action="selecteur", etat=NP1, role="inf_y-0")["html"]
    and "couche de base (principale), dans la prédalle" in req(action="selecteur", etat=NP1, role="inf_y-0")["html"]
    and "axe " not in req(action="selecteur", etat=EC, role="inf_y-0")["html"].split("pk-row")[1][:400])
# les dessins
vp, vc = rep_p["vues"], rep_c["vues"]
chk("les DESSINS sont ceux du type : prédalle → dessin_predalle (préfixe « pd », la peau hachurée nommée, sa légende) ; coulée → dessin_dalle "
    "(préfixe « dl », sa légende)",
    'id="pd_c0' in vp[0]["svg"] and "prédalle préfabriquée · 60 mm" in vp[0]["svg"] and vp[0]["legende"] == AP.LEGENDES["c"]
    and "fond grisé : la prédalle" in vp[1]["svg"] and 'id="dl_c0' in vc[0]["svg"] and vc[0]["legende"] == P.LEGENDES["c"]
    and "prédalle" not in vc[0]["svg"], (vp[0]["svg"][:80], vc[0]["svg"][:80]))
chk("l'en-tête d'une vue de prédalle dit la peau : « 100 × 22 cm, prédalle 6 cm · C30/37 · B500 · principale Y »",
    "<b>Section A</b><span>· 100 × 22 cm, prédalle 6 cm · C30/37 · B500 · principale Y</span>" in vp[0]["entete"])
# le verrou d'une prédalle
rv = saisir(NP1, "verrou", True)
chk("une prédalle verrouillée : la rubrique « Caractéristiques » en lecture dit le type (« Prédalle + dalle CSP »), « 6 + 16 cm » ; un "
    "cadenas devant le nom dans la LIGNE de la dalle (colonne compacte, 09/11/2026), la tête des résultats un simple titre ; aucune saisie",
    "Prédalle + dalle CSP" in rv["donnees"] and "6 + 16 cm" in rv["donnees"] and 'data-chemin="h_pre"' not in rv["donnees"]
    and re.search(r'<span class="et-elt-nom lu"[^>]*>' + re.escape(C.icon("lock")) + r'<span>Prédalle 1</span></span>', rv["donnees"]) is not None
    and 'data-chemin="nom"' not in rv["donnees"] + rv["resultat"] and '<div class="et-rp"><b>Prédalle 1</b>' in rv["resultat"]
    and "verrouillée" in req(action="modifier", etat=rv["etat"], chemin=["h_pre"], valeur="8")["message"])

# ------------------------------------------------------------------ 6. l'import des deux fichiers Streamlit
print("\n=== 6. La reprise des deux fichiers Streamlit ===")
V1 = {"dalles": [{"id": 1, "nom": "Plancher", "sections": [{"id": 1, "nom": "A"}]}],
      "values": {"dal1_sec1_M_inf": 28.0, "dal1_sec1_M_sup_raw": "-12,0", "dal1_sec1_V": 55.0, "gamma_s": 1.5}}
FP = {"version": "predalle-1.0", "predalles": [{"id": 1, "nom": "Plancher P", "sections": [{"id": 1, "nom": "A"}, {"id": 2, "nom": "Appui"}]}],
      "values": {"pre1_h": 20, "pre1_h_pre": 5.0, "pre1_sec1_My_inf": 35.0, "pre1_sec1_Mx_inf_raw": "-12,0", "pre1_sec2_My_sup": 18.0,
                 "pre1_sec1_ncouches_inf_y": 2, "pre1_sec1_arm_type_inf_y_c2": "n barres", "gamma_s": 1.5}}
fd = saisir(copy.deepcopy(NP1), "fichier_streamlit", {"nom": "dalle_ba.json", "texte": json.dumps(V1)})
dd_ = actif(fd["etat"])
chk("dalle_ba.json (écran Dalle) ouvert dans une note de prédalles : la note devient UNE dalle COULÉE (le type vient du fichier), "
    "« Plancher », Mx,inf 28, empilement faux ; ce que l'écran remplaçait en silence est dit (M_sup « -12,0 » lu 0)",
    len(fd["etat"]["dalles"]) == 1 and dd_["type"] == "coulee" and "h_pre" not in dd_ and dd_["nom"] == "Plancher"
    and dd_["sections"][0]["Mx_inf"] == 28.0 and dd_["empilement"] is False and "lu 0" in (fd["message"] or "") and "1 dalle(s)" in fd["message"],
    fd["message"])
fp = saisir(copy.deepcopy(e0), "fichier_streamlit", {"nom": "predalle_ba.json", "texte": json.dumps(FP)})
pp = actif(fp["etat"])
chk("predalle_ba.json (écran Prédalle) ouvert dans une note neuve : UNE PRÉDALLE, peau 5 cm, ses deux sections et son renfort, "
    "empilement faux ; le moment négatif lu 0 est dit",
    pp["type"] == "predalle" and pp["h_pre"] == 5.0 and pp["nom"] == "Plancher P" and [s["nom"] for s in pp["sections"]] == ["A", "Appui"]
    and len(pp["sections"][0]["faces"]["inf_y"]["couches"]) == 2 and pp["sections"][0]["My_inf"] == 35.0 and pp["empilement"] is False
    and "lu 0" in (fp["message"] or "") and "1 prédalle(s)" in fp["message"], fp["message"])
chk("… et le calcul qui suit est celui du moteur prédalle sur ce fichier (comme beton.predalle.fichier.lire le lit)",
    lignes(fp) == lignes_moteur(MO_P.verifier(sans_site(pp)), AP))
f_peau = copy.deepcopy(FP)
f_peau["values"].update({"pre1_h": 8, "pre1_h_pre": 12.0})
o = saisir(copy.deepcopy(e0), "fichier_streamlit", {"nom": "p.json", "texte": json.dumps(f_peau)})
chk("une peau de 12 cm sur 8 cm dans le fichier : reprise à 7 cm comme l'écran, et c'est dit", actif(o["etat"])["h_pre"] == 7.0 and "ramenée à 7 cm" in o["message"])
f_abs = copy.deepcopy(FP)
f_abs["values"]["pre1_sec1_My_inf"] = 1e300
o1 = saisir(copy.deepcopy(e0), "fichier_streamlit", {"nom": "absurde.json", "texte": json.dumps(f_abs)})
o2 = saisir(copy.deepcopy(e0), "fichier_streamlit", {"nom": "x.json", "texte": "{pas du json"})
o3 = saisir(copy.deepcopy(e0), "fichier_streamlit", {"nom": "x.json", "texte": json.dumps({"values": {"dal1_b": 800}})})
chk("un fichier aux valeurs absurdes, un fichier qui n'est pas du JSON, un fichier hors des bornes de l'écran : refusés, la note intacte",
    all(o_["message"] and o_["etat"] == e0 for o_ in (o1, o2, o3)) and "refusé" in o1["message"] and "JSON" in o2["message"] and "refusé" in o3["message"],
    [o_.get("message") for o_ in (o1, o2, o3)])

# ------------------------------------------------------------------ 7. l'option « nappes empilées »
print("\n=== 7. « empilement » : vrai pour une note neuve, faux quand la clé manque, commun à la note ===")
chk("vrai pour une note neuve (nouvel_etat) ; faux pour une note relue sans la clé, pour une note de prédalles, pour un fichier Streamlit importé",
    P.nouvel_etat()["dalles"][0]["empilement"] is True and da["empilement"] is False and dp1["empilement"] is False
    and dd_["empilement"] is False and pp["empilement"] is False)
deux = req(action="ajouter", etat=e0)["etat"]
off = saisir(deux, "empilement", False)
on = saisir(off["etat"], "empilement", "true")
chk("la case « Nappes empilées » (rubrique Options) écrit la clé sur TOUTES les dalles de la note, comme γs ; décochée puis recochée",
    all(d["empilement"] is False for d in off["etat"]["dalles"]) and all(d["empilement"] is True for d in on["etat"]["dalles"])
    and '<input type="checkbox" data-chemin="empilement" data-sorte="case">' in off["donnees"]
    and '<input type="checkbox" data-chemin="empilement" data-sorte="case" checked>' in on["donnees"])
o = req(action="modifier", etat=e0, chemin=["empilement"], valeur="peut-être")
chk("« peut-être » n'est ni coché ni décoché : refusé, la note intacte", o["message"] and "coché ou décoché" in o["message"] and o["etat"] == e0)
r_on = MO.verifier(POS.pour_moteur(d0))
r_off = MO.verifier(POS.pour_moteur(actif(off["etat"], 0)))
chk("option VRAIE : la base secondaire (inf. X) est posée sur la principale — le moteur reçoit son axe (3,5 + 0,5 + 0,5 = 4,5 cm) et le champ "
    "vide l'affiche « auto · 4,5 » ; option FAUSSE : la copie envoyée au moteur est la dalle d'avant, à l'octet, et le résultat celui du moteur",
    r_on["sections"][0]["faces"]["inf_x"]["couches"][0]["e"] == 4.5 and r_on["sections"][0]["faces"]["inf_y"]["couches"][0]["e"] == 3.5
    and 'data-chemin="sections/0/faces/inf_x/couches/0/dist" data-sorte="texte" inputmode="decimal" value="" data-vide="auto" placeholder="auto · 4,5"' in r0["donnees"]
    and POS.pour_moteur(actif(off["etat"], 0)) == sans_site(actif(off["etat"], 0)) and r_off == MO.verifier(sans_site(actif(off["etat"], 0)))
    and lignes(off) == lignes_moteur(r_off, P))
GROUPE = r'<g class="pc couche" data-sel="s0\.{}">(.*?)</g>'


def niveaux(svg, cle_points, cle_traits):
    """Dans la PREMIÈRE coupe (⊥ principale) : les cy des cercles de la couche ``cle_points`` et les y1 des traits de
    ``cle_traits`` (tests/test_api_web_dalle_positions.py, lot FUS4)."""
    coupe = svg.split("Coupe ⊥")[1]
    pts = re.search(GROUPE.format(re.escape(cle_points)), coupe).group(1)
    trs = re.search(GROUPE.format(re.escape(cle_traits)), coupe).group(1)
    return ({float(x) for x in re.findall(r'<circle cx="[\d.]+" cy="([\d.]+)"', pts)},
            {float(x) for x in re.findall(r'<line x1="[\d.]+" y1="([\d.]+)"', trs)})


g_on, g_off = r_on["sections"][0]["grandeurs"], r_off["sections"][0]["grandeurs"]
n_on, n_off = niveaux(r0["vues"][0]["svg"], "inf_y-0", "inf_x-0"), niveaux(off["vues"][0]["svg"], "inf_y-0", "inf_x-0")
chk("la case cochée / décochée fait bouger la hauteur utile de la direction secondaire : d_inf,X = 20 − 4,5 = 15,5 cm cochée, 20 − 3,5 = 16,5 cm "
    "décochée (la principale, d_inf,Y = 16,5, ne bouge pas) — dans le calcul, dans la fenêtre de calcul et dans l'en-tête des résultats",
    (g_on["d_inf_x"], g_off["d_inf_x"], g_on["d_inf_y"], g_off["d_inf_y"]) == (15.5, 16.5, 16.5, 16.5)
    and "20 − 4,5" in r0["volets"] and "20 − 4,5" not in off["volets"], (g_on["d_inf_x"], g_off["d_inf_x"]))
chk("la coupe dessine les deux directions à des niveaux distincts seulement cochée (les traits de la base X descendent de 10 mm quand elle est "
    "posée sur la base Y), et sa légende dit « sur la base Y » seulement cochée — le dessin ne ment jamais",
    n_on[0] == n_off[0] and n_on[1] != n_off[1] and len(n_on[1]) == len(n_off[1]) == 1
    and abs(min(n_on[1]) - min(n_off[1]) + 10.0) < 0.06
    and "· sur la base Y" in r0["vues"][0]["svg"] and "sur la base" not in off["vues"][0]["svg"] and "axe saisi" not in r0["vues"][0]["svg"],
    (n_on, n_off))
pr_on = saisir(saisir(e0, "type", "predalle")["etat"], "sections/0/faces/sup_x/couches/0/treillis", "12/12/150/150")
chk("prédalle, nappes empilées : la base SUPÉRIEURE secondaire est posée sur la principale — la pose le dit (« posée sur la base principale »), la "
    "formule e aussi, et jamais « axe saisi » ; les inférieures gardent la règle du moteur (dans / sur la prédalle)",
    "posée sur la base principale (e<sub>p</sub> + ½ Ø<sub>p</sub> + ½ Ø<sub>s</sub>)" in pr_on["donnees"]
    and "posée sur la base principale" in pr_on["volets"] and "axe saisi" not in pr_on["volets"] + pr_on["donnees"]
    and "axe auto · dans la prédalle (c + ½ Ø)" in pr_on["donnees"] and "axe auto · sur la prédalle" in pr_on["donnees"],
    re.findall(r"axe auto · [^<]+", pr_on["donnees"])[:6])

# ------------------------------------------------------------------ 8. les points d'ancrage des autres lots, branchés
print("\n=== 8. Les points d'ancrage : nappes (FUS3), positions (FUS4), colonne compacte (FUS5) ===")
nb = saisir(e0, "sections/0/nappes/inf/mode", "barres")
chk("un chemin « sections/0/nappes/inf/mode » passe par dalle_nappes.ecrire : les deux bases inférieures deviennent des barres, les supérieures "
    "restent un treillis",
    not nb["message"] and [actif(nb["etat"])["sections"][0]["faces"][f]["couches"][0]["type"] for f in ("inf_x", "inf_y", "sup_x", "sup_y")]
    == ["barres", "barres", "treillis", "treillis"], nb["message"])
o = req(action="modifier", etat=e0, chemin=["sections", "0", "nappes", "inf", "zzz"], valeur="1")
chk("un chemin de nappe inconnu : le refus de dalle_nappes (ValueError) devient un message, la dalle intacte",
    o["message"] and "Champ de nappe inconnu" in o["message"] and o["etat"] == e0, o.get("message"))
chk("la rubrique « 3 — Armatures » est le HTML de dalle_nappes (les segments « Treillis soudé · Barres · Treillis + renforts » par nappe), "
    "et chaque couche garde sa clé data-cible « s0.<face>-0 »",
    r0["donnees"].count('data-chemin="sections/0/nappes/inf/mode"') == 3 and r0["donnees"].count('data-chemin="sections/0/nappes/sup/mode"') == 3
    and {f"s0.{f}-0" for f in EN.FACES} <= set(re.findall(r'data-cible="([^"]+)"', r0["donnees"])))
# l'ampoule garde la nappe cohérente (demande de FUS3)
des = next(x for x in TR.liste_choix() if x != TR.TREILLIS_DEFAUT)
am = req(action="choisir", etat=e0, role="inf_y-0", nom=f"Treillis {des}")
s_am = actif(am["etat"])["sections"][0]
chk(f"l'AMPOULE sur une nappe de treillis unique : « Treillis {des} » choisi en inf. Y pose sa transposée en inf. X — la nappe reste un treillis "
    "unique (mode « treillis », cohérent) ; les supérieures ne bougent pas",
    am["ok"] and s_am["faces"]["inf_y"]["couches"][0]["treillis"] == des and s_am["faces"]["inf_x"]["couches"][0]["treillis"] == NAP.transposee(des)
    and NAP.mode(s_am, "inf", "y")["mode"] == "treillis" and NAP.mode(s_am, "inf", "y")["coherent"]
    and s_am["faces"]["sup_x"]["couches"][0]["treillis"] == TR.TREILLIS_DEFAUT, am.get("erreur"))
E_r = saisir(e0, "sections/0/faces/inf_y/couches/+", "1")["etat"]
am2 = req(action="choisir", etat=E_r, role="inf_x-0", nom=f"Treillis {des}")
s2 = actif(am2["etat"])["sections"][0]
chk("… avec un renfort (mode « treillis + renforts ») : la transposée aussi, le renfort gardé ; des BARRES choisies laissent la nappe « incohérente », "
    "et l'écran le dit avec « En faire un treillis »",
    s2["faces"]["inf_y"]["couches"][0]["treillis"] == NAP.transposee(des) and len(s2["faces"]["inf_y"]["couches"]) == 2
    and NAP.mode(s2, "inf", "y")["mode"] == "treillis_renforts"
    and not NAP.mode(actif((ab := req(action="choisir", etat=e0, role="inf_y-0", nom="Ø12/150"))["etat"])["sections"][0], "inf", "y")["coherent"]
    and "En faire un treillis" in ab["donnees"] and C.esc(NAP.AVERTISSEMENT) in ab["donnees"])
bar = saisir(nb["etat"], "sections/0/faces/inf_x/couches/0/d", "16")["etat"]
am3 = req(action="choisir", etat=bar, role="inf_y-0", nom=f"Treillis {des}")
chk("… sur une nappe qui était en BARRES, un treillis choisi ne touche que sa face (l'autre garde ses barres Ø16) : la règle ne vaut que pour "
    "un treillis unique",
    actif(am3["etat"])["sections"][0]["faces"]["inf_x"]["couches"][0] == actif(bar)["sections"][0]["faces"]["inf_x"]["couches"][0]
    and actif(am3["etat"])["sections"][0]["faces"]["inf_y"]["couches"][0]["type"] == "treillis")
chk("la colonne de gauche est COMPACTE (etabli_compact.arbre, compact=True) et le nom se modifie dans la LIGNE de la dalle "
    "(arbre_compact, chemin « nom » ; plus dans la tête des résultats) : même chemin qu'avant, donc le résumé et le .stab suivent",
    re.search(r'<div class="et-arbre et-arbre-c" data-dalle="0"><div class="et-elsel" data-elements="1"><input data-chemin="nom" '
              r'data-sorte="texte" value="Dalle 1" class="et-elt-nom"', r0["donnees"]) is not None and "et-arbre-h" not in r0["donnees"]
    and 'data-chemin="nom"' not in r0["resultat"]
    and saisir(e0, "nom", "Hourdis")["resume"]["gouvernant"].startswith("Hourdis ·"))
chk("le y_G calculé affiché et l'axe automatique viennent de dalle_positions (yG_calcule, axe_auto) : réécrits tels quels, ils restent automatiques",
    actif(saisir(e0, "sections/0/faces/inf_x/couches/0/dist", "4,5")["etat"])["sections"][0]["faces"]["inf_x"]["couches"][0]["dist"] is None
    and actif(saisir(e0, "sections/0/faces/inf_x/yG", C.nb(POS.yG_calcule(d0, d0["sections"][0], "inf_x")))["etat"])["sections"][0]["faces"]["inf_x"]["yG"] is None
    and actif(saisir(e0, "sections/0/faces/inf_x/couches/0/dist", "3,5")["etat"])["sections"][0]["faces"]["inf_x"]["couches"][0]["dist"] == 3.5)

# ------------------------------------------------------------------ 9. les mutations : une altération du code fait rougir un contrôle
print("\n=== 9. Les mutations ===")


def muter(nom, module, attr, remplacant, controle):
    """Le contrôle est vert, ROUGE sous la mutation, vert de nouveau une fois le code remis."""
    original = getattr(module, attr)
    avant = controle()
    setattr(module, attr, remplacant)
    try:
        pendant = controle()
    except Exception as e:  # noqa: BLE001 — une mutation peut faire lever : le contrôle n'est pas vert
        pendant = False
        print("       (mutation : exception", type(e).__name__, ")")
    finally:
        setattr(module, attr, original)
    apres = controle()
    chk(f"mutation « {nom} » : le contrôle rougit, puis reverdit", avant and not pendant and apres, (avant, pendant, apres))


def ctl_moteur():
    r_ = req(action="calculer", etat=note_predalle())
    return r_["ok"] and lignes(r_) == lignes_moteur(MO_P.verifier(sans_site(actif(r_["etat"]))), AP)


muter("le moteur des dalles appelé quel que soit le type", P, "_MO", lambda d: MO, ctl_moteur)


def ctl_bascule():
    o_ = req(action="modifier", etat=saisir(e0, "h", "5")["etat"], chemin=["type"], valeur="predalle")
    return o_["ok"] and not o_["message"] and actif(o_["etat"]).get("h_pre") == 4.0


def _type_sans_peau(d, valeur):
    d["type"] = valeur


muter("la bascule vers la prédalle n'ajoute pas h_pre bornée à h_pre_max(h)", P, "_changer_type", _type_sans_peau, ctl_bascule)
muter("une note neuve sans nappes empilées", P, "nouvel_etat",
      lambda: {"version": 2, "dalles": [P._site(EN.nouvelle_dalle())], "actif": 0},
      lambda: req(action="nouveau")["etat"]["dalles"][0]["empilement"] is True)
muter("le type du fichier Streamlit n'est plus déduit (le lecteur des dalles lit tout)", P, "FI_P", P.FI,
      lambda: actif(saisir(copy.deepcopy(e0), "fichier_streamlit", {"nom": "p.json", "texte": json.dumps(FP)})["etat"]).get("h_pre") == 5.0)
muter("une case lit n'importe quel texte comme cochée", P, "_lire_bool", lambda v: bool(v),
      lambda: "coché ou décoché" in (req(action="modifier", etat=e0, chemin=["empilement"], valeur="peut-être")["message"] or ""))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

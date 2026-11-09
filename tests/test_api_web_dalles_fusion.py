# -*- coding: utf-8 -*-
"""Le module FUSIONNÉ « Dalles béton armé » (api_web/dalle.py — docs/site/DALLES_FUSION.md § 0 et § 2, lot FUS2),
vérifié en Python ordinaire à travers le routeur (les mêmes requêtes JSON que le navigateur).

- l'ÉTAT fusionné (version 3) : chaque dalle porte « type » (coulee | predalle) et « verrou » ; une note neuve est une
  dalle coulée en place ; les trois formes d'état sont relues ({"dalles"} sans type → coulee, {"predalles"} de l'ancien
  module → predalle, actif conservé, rien de perdu) et une note abîmée (les deux clés, un type inconnu, un verrou qui
  n'en est pas un) est REFUSÉE, jamais remplacée par une note neuve ;
- la rubrique « 1 — Caractéristiques » : les segments du type, « h_p prédalle » et le complément h − h_p (lu, jamais
  stocké) ; la bascule coulee ↔ predalle ajoute ou retire h_pre et conserve tout le reste ; les refus géométriques de
  l'ancien module Prédalles s'appliquent, la dalle reste intacte ;
- le MOTEUR est choisi selon le type (beton/dalle ou beton/predalle, inchangés) : les lignes du résultat sont celles de
  l'appel direct du moteur du type sur la copie que le site lui envoie (dalle_positions.pour_moteur) ; l'en-tête, le
  pied, les hypothèses, le périmètre, les avertissements et les dessins sont ceux du type ;
- l'import des DEUX fichiers Streamlit (dalle_ba.json, predalle_ba.json : le type déduit du fichier) ;
- les nappes TOUJOURS posées l'une sur l'autre (décision de l'ingénieur du 09/11/2026) : plus d'option (ni case, ni
  chemin « empilement »), une note neuve, une note d'avant (clé absente, fausse, vraie, illisible : relue sans effet), une
  note de l'ancien module Prédalles, un fichier Streamlit — tous empilés ; une note d'avant dont les chiffres changent le
  dit UNE fois, à l'ouverture (un avis, pas une erreur), et seulement elle (S5 compris : une note cochée dont la base
  secondaire repose désormais sur un renfort principal plus haut que sa base) ;
- l'ampoule (sélecteur de nappe de base) garde une nappe de treillis unique cohérente (la transposée sur l'autre face) ;
- les points d'ancrage des autres lots sont branchés (dalle_positions : axe_auto, pour_moteur, annoter, candidats_base ;
  dalle_nappes : ecrire, html ; etabli_compact : compact=True, le nom dans la ligne de la dalle) ;
- des MUTATIONS : une altération du code fait rougir le contrôle qui la garde (dont : l'option qui revient, la clé
  d'avant qui fait refuser la note, l'avis qui ne se dit plus ou qui se redit).

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
from api_web import dessin_dalle as DD  # noqa: E402
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
    """La dalle sans les clés du site, telle que la note la garde (le moteur, lui, reçoit ``POS.pour_moteur(d)`` : les
    couches secondaires posées sur la principale)."""
    return {k: v for k, v in d.items() if k not in P.CLES_SITE}


def avis(rep):
    """Le texte de l'avis d'ouverture rendu en tête du résultat (juste après la tête), ou None."""
    m = re.match(r'<div class="pbody"><div class="et-rp">.*?</div><div class="msg">(.*?)</div><div class="et-blocs">', rep["resultat"], re.S)
    return m.group(1) if m else None


def axe_inf_x(d):
    """L'axe que le MOTEUR reçoit pour la base inf. X (direction secondaire quand la principale est Y)."""
    return MO.verifier(POS.pour_moteur(d))["sections"][0]["faces"]["inf_x"]["couches"][0]["e"] if d["type"] == "coulee" \
        else MO_P.verifier(POS.pour_moteur(d))["sections"][0]["faces"]["sup_x"]["couches"][0]["e"]


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
chk("une note neuve : version 3, UNE dalle « coulee », déverrouillée, la dalle de l'écran Streamlit, et plus de clé « empilement » "
    "(décision du 09/11/2026 : plus d'option) ni d'avis",
    r0["ok"] and e0["version"] == 3 == P.VERSION and len(e0["dalles"]) == 1 and d0["type"] == "coulee" and d0["verrou"] is False
    and "empilement" not in d0 and P.CLE_AVIS not in e0 and sans_site(d0) == EN.nouvelle_dalle() and avis(r0) is None,
    (e0.get("version"), {k: d0.get(k) for k in P.CLES_SITE}))
chk("les clés du site sont nommées une fois par module et liées ici : dalle.CLES_SITE = dalle_positions.CLES_SITE, et la copie "
    "envoyée au moteur (POS.pour_moteur) n'en porte aucune — le moteur des dalles la lit",
    set(P.CLES_SITE) == set(POS.CLES_SITE) == {"verrou", "type"} and P.CLES_ANCIENNES == ("empilement",)
    and not (set(POS.pour_moteur(d0)) & set(P.CLES_SITE)) and MO.verifier(POS.pour_moteur(d0))["statut"] == MO.OK)
chk("l'état survit à un aller-retour JSON et sa relecture est stable (normaliser ∘ normaliser)",
    P.normaliser(json.loads(json.dumps(e0))) == e0 and P.normaliser(copy.deepcopy(e0)) == e0)
chk("le type se lit dans la rubrique 1 : segments « Dalle CSP » (retenu) · « Prédalle + dalle CSP » (les libellés courts de "
    "l'ingénieur, 09/11/2026 ; l'aide du champ dit « CSP : coulé sur place. » d'abord), le résumé « Dalle CSP · C30/37 · B500 · "
    "c = 3 cm », aucune peau ; et la case « Nappes empilées » n'existe plus (ni dans les options, ni ailleurs)",
    P.TYPES_DALLE == (("coulee", "Dalle CSP"), ("predalle", "Prédalle + dalle CSP"))
    and re.search(r'<button type="button" class="on" data-chemin="type" data-sorte="choix" data-valeur="coulee">Dalle CSP</button>'
                  r'<button type="button" class="" data-chemin="type" data-sorte="choix" data-valeur="predalle">Prédalle \+ dalle CSP</button>',
                  r0["donnees"]) is not None
    and P.AIDE_TYPE.startswith("CSP : coulé sur place. ") and f'<span class="aide" title="{C.esc(P.AIDE_TYPE)}">' in r0["donnees"]
    and 'data-chemin="empilement"' not in r0["donnees"] and "Nappes empilées" not in r0["donnees"]
    and 'data-chemin="h_pre"' not in r0["donnees"]
    and "Dalle CSP · C30/37 · B500 · c = 3 cm" in r0["donnees"], P.TYPES_DALLE)

# ------------------------------------------------------------------ 2. les trois formes d'état
print("\n=== 2. Les trois formes d'état relues ; la note abîmée refusée ===")
ra = req(action="calculer", etat={"version": 1, "dalles": [{"nom": "Ancienne", "h": 18}], "actif": 0})
da = actif(ra["etat"])
chk("{« dalles »} d'avant, sans type : relue coulée en place, version 3, ses nappes posées l'une sur l'autre comme toutes (la base "
    "inf. X sur la base Y : 3,5 + 0,5 + 0,5 = 4,5 cm), et l'avis d'ouverture le dit",
    ra["ok"] and da["type"] == "coulee" and "empilement" not in da and da["h"] == 18 and ra["etat"]["version"] == 3
    and axe_inf_x(da) == 4.5 and avis(ra) == P.AVIS["note"], ra.get("erreur"))
NP = note_predalle()
NP["predalles"].append(dict(EN_P.nouvelle_predalle(), nom="Appui", h=24, h_pre=7.0))
NP["actif"] = 1
rp = req(action="calculer", etat=NP)
chk("{« predalles »} de l'ancien module : chaque prédalle devient une dalle de type « predalle », h_pre gardée, actif conservé — rien "
    "n'est perdu ; ses nappes SUPÉRIEURES posées l'une sur l'autre (sup. X sur sup. Y : 4,5 cm), et l'avis le dit",
    rp["ok"] and [d["type"] for d in rp["etat"]["dalles"]] == ["predalle", "predalle"] and rp["etat"]["actif"] == 1
    and actif(rp["etat"])["h_pre"] == 7.0 and actif(rp["etat"])["nom"] == "Appui"
    and all("empilement" not in d for d in rp["etat"]["dalles"]) and axe_inf_x(rp["etat"]["dalles"][0]) == 4.5
    and avis(rp) == P.AVIS["note"]
    and sans_site(rp["etat"]["dalles"][0]) == EN_P.normaliser(sans_site(NP["predalles"][0])) if rp["ok"] else False, rp.get("erreur"))


def par_nom(rep):
    return {(c, n): (niv, t) for c, n, niv, t in lignes(rep)}


ecarts = []
for k in (0, 1):
    neuf_ = par_nom(req(action="calculer", etat=dict(copy.deepcopy(NP), actif=k)))
    vieux_ = par_nom(json.loads(R.traiter(json.dumps(dict(module="predalle", action="calculer", etat=dict(copy.deepcopy(NP), actif=k))))))
    ecarts.append({cle for cle in neuf_ if neuf_[cle] != vieux_.get(cle)})
chk("… ses CHIFFRES sont ceux de l'ancien module Prédalles aux faces INFÉRIEURES (la règle du moteur : dans / sur la prédalle) et à "
    "l'effort tranchant ; seule la nappe supérieure secondaire (X) et la hauteur qui en dépend diffèrent — le prix de la décision du "
    "09/11/2026, dit par l'avis",
    all(e_ <= {(c, n) for c in ("s0", "s1") for n in ("Supérieure X", "Hauteur")} for e_ in ecarts)
    and any(("s0", "Supérieure X") in e_ or ("s1", "Supérieure X") in e_ for e_ in ecarts), ecarts)
# (l'ancienne clé « empilement » est relue SANS EFFET, quelle que soit sa valeur : une note qui la porte n'est jamais refusée pour elle)
tolerees = [req(action="calculer", etat={"version": 2, "dalles": [{"nom": "T", "empilement": v}], "actif": 0}) for v in (True, False, "oui", 3)]
chk("une note d'avant qui porte « empilement » (vrai, faux, « oui », 3) se relit — jamais refusée pour cette clé, qui disparaît de "
    "l'état et ne change rien : les mêmes axes pour les quatre",
    all(t_["ok"] and "empilement" not in actif(t_["etat"]) for t_ in tolerees)
    and len({json.dumps(POS.pour_moteur(actif(t_["etat"])), sort_keys=True) for t_ in tolerees}) == 1
    and axe_inf_x(actif(tolerees[0]["etat"])) == 4.5, [t_.get("erreur") for t_ in tolerees])
ABIMEES = [({"dalles": [{}], "predalles": [{}]}, "abîmée"), ({"predalles": "abc"}, "liste"), ({"predalles": [{"type": "coulee"}]}, "note abîmée"),
           ({"dalles": [{"type": "autre"}]}, "coulee » ou « predalle"), ({"dalles": [{"verrou": "oui"}]}, "verrou vrai ou faux"),
           ({"dalles": [{"h_pre": 5}]}, "h_pre"), ({"dalles": [{"type": "predalle", "h_pre": 30}]}, "h_pre")]
lues = [(req(action="calculer", etat=x), mot) for x, mot in ABIMEES]
chk("une note abîmée — les DEUX clés, « predalles » qui n'est pas une liste, un type contradictoire ou inconnu, un verrou qui n'en "
    "est pas un, une peau sur une dalle coulée, une peau impossible — est REFUSÉE avec une erreur claire, jamais remplacée en silence",
    all(not r_["ok"] and r_["erreur"].startswith("ValueError") and mot in r_["erreur"] for r_, mot in lues),
    [(mot, r_.get("erreur", "ok ?!")[:120]) for r_, mot in lues if r_["ok"] or mot not in r_.get("erreur", "")])
chk("une note sans dalle ({« predalles »: []}, {}) est une note neuve ; une dalle « type predalle » dans « dalles » se relit prédalle",
    req(action="calculer", etat={"predalles": []})["etat"] == e0 and req(action="calculer", etat={})["etat"] == e0
    and actif(req(action="calculer", etat={"dalles": [{"type": "predalle", "h_pre": 5}]})["etat"])["h_pre"] == 5.0)

# ------------------------------------------------------------------ 3. la bascule coulee ↔ predalle
print("\n=== 3. La bascule coulée ↔ prédalle ===")
E = copy.deepcopy(e0)
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
chk("« Ajouter une dalle » sur une note ouverte sur une prédalle : une « Prédalle 2 » de type predalle (h = 22, h_pre = 6), avec le γs "
    "et le jeu de la note ; « Copier » garde le type",
    dj["type"] == "predalle" and dj["nom"] == "Prédalle 2" and dj["h"] == 22 and dj["h_pre"] == 6.0 and "empilement" not in dj
    and actif(req(action="dupliquer", etat=bp["etat"])["etat"])["type"] == "predalle", dj.get("nom"))

# ------------------------------------------------------------------ 4. h_pre, le complément, les refus géométriques
print("\n=== 4. La peau, le complément lu, les refus géométriques ===")
PR = saisir(e0, "type", "predalle")["etat"]
PR = saisir(PR, "h", "22")["etat"]
r75 = saisir(PR, "h_pre", "7,5")
chk("h_pre 7,5 cm sur h = 22 : stockée ; le complément 14,5 cm est AFFICHÉ, jamais stocké (aucune clé de plus dans l'état) ; le moteur "
    "prédalle pose la base secondaire à 8,0 et le renfort… à 8,5 comme l'ancien module",
    actif(r75["etat"])["h_pre"] == 7.5 and '<span class="dl-ro">14,5 cm</span>' in r75["donnees"]
    and set(actif(r75["etat"])) == set(actif(PR)) and "coule" not in json.dumps(r75["etat"])
    and MO_P.verifier(POS.pour_moteur(actif(r75["etat"])))["sections"][0]["faces"]["inf_x"]["couches"][0]["e"] == 8.0)
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
m_p = MO_P.verifier(POS.pour_moteur(dp1))
m_d = MO.verifier({k: v for k, v in sans_site(dp1).items() if k != "h_pre"})
chk("type predalle : chaque ligne du résultat — nom, couleur, taux — est celle de beton.predalle.moteur.verifier sur la copie que le site "
    "lui envoie (pour_moteur) ; le résumé aussi ; et ce N'EST PAS ce que dirait le moteur des dalles (base inf. secondaire à 6,5, sur la "
    "prédalle, et non 3,5)",
    lignes(rep_p) == lignes_moteur(m_p, AP) and rep_p["resume"]["taux"] == C.pct(m_p["taux_max"])
    and lignes_moteur(m_p, AP) != lignes_moteur(m_d, P)
    and m_p["sections"][0]["faces"]["inf_x"]["couches"][0]["e"] == 6.5 and m_d["sections"][0]["faces"]["inf_x"]["couches"][0]["e"] == 3.5,
    (lignes(rep_p)[:3], lignes_moteur(m_p, AP)[:3]))
chk("… les chiffres de l'écran Streamlit prédalle (tests/test_api_web_predalle.py) : As,req 573 mm² (principale) et 645 mm² (secondaire)",
    round(m_p["sections"][0]["grandeurs"]["As_req_inf_y"]) == 573 and round(m_p["sections"][0]["grandeurs"]["As_req_inf_x"]) == 645
    and "573" in rep_p["volets"] and "645" in rep_p["volets"])
EC = E
rep_c = req(action="calculer", etat=EC)
m_c = MO.verifier(POS.pour_moteur(actif(EC)))
chk("type coulee : chaque ligne est celle de beton.dalle.moteur.verifier sur la copie que le site lui envoie",
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
chk("les AVERTISSEMENTS propres à la prédalle sont présents (renfort Y à 7,0 et base X à 6,5 « posées toutes deux sur la prédalle »), dans "
    "le bloc de leur section ; les bases supérieures ne sont plus « au même niveau » (la secondaire posée sur la principale)",
    "posées toutes deux sur la prédalle" in rep_p["resultat"] and "axe 7,0 cm" in rep_p["resultat"]
    and "les bases X et Y supérieures sont calculées" not in rep_p["resultat"]
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
    "« Plancher », Mx,inf 28, ses nappes posées l'une sur l'autre (inf. X à 4,5 cm, l'écran Streamlit disait 3,5) et l'avis « Fichier "
    "Streamlit » en tête du résultat ; ce que l'écran remplaçait en silence est dit (M_sup « -12,0 » lu 0)",
    len(fd["etat"]["dalles"]) == 1 and dd_["type"] == "coulee" and "h_pre" not in dd_ and dd_["nom"] == "Plancher"
    and dd_["sections"][0]["Mx_inf"] == 28.0 and "empilement" not in dd_ and axe_inf_x(dd_) == 4.5 and avis(fd) == P.AVIS["streamlit"]
    and P.CLE_AVIS not in fd["etat"] and "lu 0" in (fd["message"] or "") and "1 dalle(s)" in fd["message"],
    (fd["message"], avis(fd)))
fp = saisir(copy.deepcopy(e0), "fichier_streamlit", {"nom": "predalle_ba.json", "texte": json.dumps(FP)})
pp = actif(fp["etat"])
chk("predalle_ba.json (écran Prédalle) ouvert dans une note neuve : UNE PRÉDALLE, peau 5 cm, ses deux sections et son renfort, "
    "ses nappes supérieures posées l'une sur l'autre (l'avis le dit) ; le moment négatif lu 0 est dit",
    pp["type"] == "predalle" and pp["h_pre"] == 5.0 and pp["nom"] == "Plancher P" and [s["nom"] for s in pp["sections"]] == ["A", "Appui"]
    and len(pp["sections"][0]["faces"]["inf_y"]["couches"]) == 2 and pp["sections"][0]["My_inf"] == 35.0 and "empilement" not in pp
    and axe_inf_x(pp) == 4.5 and avis(fp) == P.AVIS["streamlit"]
    and "lu 0" in (fp["message"] or "") and "1 prédalle(s)" in fp["message"], fp["message"])
chk("… et le calcul qui suit est celui du moteur prédalle sur la copie que le site lui envoie (le fichier comme beton.predalle.fichier.lire "
    "le lit, les nappes supérieures empilées)",
    lignes(fp) == lignes_moteur(MO_P.verifier(POS.pour_moteur(pp)), AP))
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

# ------------------------------------------------------------------ 7. les nappes toujours posées l'une sur l'autre
print("\n=== 7. Les nappes TOUJOURS posées l'une sur l'autre (décision de l'ingénieur du 09/11/2026) ; l'avis d'ouverture ===")
deux = req(action="ajouter", etat=e0)["etat"]
o = req(action="modifier", etat=deux, chemin=["empilement"], valeur=False)
chk("plus d'option : le chemin « empilement » est un champ inconnu (refusé, la note intacte, aucune dalle ne prend la clé) ; aucune case "
    "dans les options d'une dalle coulée ni d'une prédalle",
    o["ok"] and o["message"] and "champ inconnu" in o["message"] and o["etat"] == deux
    and all("empilement" not in d for d in o["etat"]["dalles"])
    and 'data-chemin="empilement"' not in o["donnees"] + saisir(e0, "type", "predalle")["donnees"], o.get("message"))
FORMES = {"note neuve": actif(e0), "note d'avant sans la clé": da, "note d'avant, clé fausse": actif(tolerees[1]["etat"]),
          "note d'avant, clé « oui »": actif(tolerees[2]["etat"]), "dalle_ba.json": dd_,
          "prédalles d'avant (sup. X)": actif(rp["etat"], 0), "fichier Prédalle (sup. X)": pp}
chk("toutes les formes de note sont empilées — note neuve, note d'avant (clé absente, fausse, illisible), fichier Streamlit : la base inf. X "
    "reçue par le moteur à 4,5 cm (posée sur la base Y), jamais à l'automatique du moteur (3,5) ; prédalles d'avant et fichier Prédalle : la "
    "base sup. X à 4,5 (depuis le dessus)",
    all(axe_inf_x(d_) == 4.5 for d_ in FORMES.values()), {k: axe_inf_x(d_) for k, d_ in FORMES.items()})
r_on = MO.verifier(POS.pour_moteur(d0))
r_brut = MO.verifier(sans_site(d0))
chk("la base secondaire posée sur la principale : le moteur reçoit son axe (3,5 + 0,5 + 0,5 = 4,5 cm, la principale à 3,5) et le champ vide "
    "l'affiche « auto · 4,5 » ; l'avertissement « nappes_confondues » du moteur seul (l'écran Streamlit) ne sort plus",
    r_on["sections"][0]["faces"]["inf_x"]["couches"][0]["e"] == 4.5 and r_on["sections"][0]["faces"]["inf_y"]["couches"][0]["e"] == 3.5
    and 'data-chemin="sections/0/faces/inf_x/couches/0/dist" data-sorte="texte" inputmode="decimal" value="" data-vide="auto" placeholder="auto · 4,5"' in r0["donnees"]
    and lignes(r0) == lignes_moteur(r_on, P) and not r_on["avertissements"]
    and [a_["code"] for a_ in r_brut["avertissements"]] == ["nappes_confondues", "nappes_confondues"])
GROUPE = r'<g class="pc couche" data-sel="s0\.{}">(.*?)</g>'


def niveaux(svg, cle_points, cle_traits):
    """Dans la PREMIÈRE coupe (⊥ principale) : les cy des cercles de la couche ``cle_points`` et les y1 des traits de
    ``cle_traits`` (tests/test_api_web_dalle_positions.py, lot FUS4)."""
    coupe = svg.split("Coupe ⊥")[1]
    pts = re.search(GROUPE.format(re.escape(cle_points)), coupe).group(1)
    trs = re.search(GROUPE.format(re.escape(cle_traits)), coupe).group(1)
    return ({float(x) for x in re.findall(r'<circle cx="[\d.]+" cy="([\d.]+)"', pts)},
            {float(x) for x in re.findall(r'<line x1="[\d.]+" y1="([\d.]+)"', trs)})


g_on, g_brut = r_on["sections"][0]["grandeurs"], r_brut["sections"][0]["grandeurs"]
sv_brut = DD.svg_coupe("t", r_brut["sections"][0], r_brut["dalle"], 0)
n_on, n_brut = niveaux(r0["vues"][0]["svg"], "inf_y-0", "inf_x-0"), niveaux(sv_brut, "inf_y-0", "inf_x-0")
chk("la hauteur utile de la direction secondaire est celle de sa nappe posée sur la principale : d_inf,X = 20 − 4,5 = 15,5 cm (le moteur seul, "
    "comme l'écran Streamlit : 20 − 3,5 = 16,5) ; la principale (d_inf,Y = 16,5) ne bouge pas — dans le calcul et dans la fenêtre de calcul",
    (g_on["d_inf_x"], g_brut["d_inf_x"], g_on["d_inf_y"], g_brut["d_inf_y"]) == (15.5, 16.5, 16.5, 16.5)
    and "20 − 4,5" in r0["volets"], (g_on["d_inf_x"], g_brut["d_inf_x"]))
chk("la coupe du site dessine les deux directions à des niveaux distincts (les traits de la base X 10 mm au-dessus de ceux que le moteur seul "
    "poserait), et sa légende dit « sur la base Y » — le dessin dit ce que le calcul utilise",
    n_on[0] == n_brut[0] and len(n_on[1]) == len(n_brut[1]) == 1 and abs(min(n_on[1]) - min(n_brut[1]) + 10.0) < 0.06
    and "· sur la base Y" in r0["vues"][0]["svg"] and "sur la base" not in sv_brut and "axe saisi" not in r0["vues"][0]["svg"],
    (n_on, n_brut))
# S3 (décision du 09/11/2026) : un RENFORT de la direction secondaire posé sur sa nappe, de bout en bout par le routeur
s3 = saisir(e0, "sections/0/faces/inf_x/couches/+", "1")
d_s3 = actif(s3["etat"])
c_s3 = MO.verifier(POS.pour_moteur(d_s3))["sections"][0]["faces"]["inf_x"]["couches"]
chk("S3 — un renfort 3 Ø12 ajouté en inf. X (secondaire) : son champ dit « auto · 5 » (3,5 + 0,5 + 1,0 : posé sur la base Y, dans le lit de la "
    "base X) et le moteur le reçoit là — non à 4,0, au niveau de la base Y (l'automatique du moteur seul) ; la légende de la coupe le dit",
    [c_["e"] for c_ in c_s3] == [4.5, 5.0] and d_s3["sections"][0]["faces"]["inf_x"]["couches"][1]["dist"] is None
    and 'data-chemin="sections/0/faces/inf_x/couches/1/dist" data-sorte="texte" inputmode="decimal" value="" data-vide="auto" placeholder="auto · 5"' in s3["donnees"]
    and "X · 3 Ø12 · axe à 50 · sur la base Y" in s3["vues"][0]["svg"], ([c_["e"] for c_ in c_s3], re.findall(r"axe à \d+[^<]*", s3["vues"][0]["svg"])))
# le contrôle croisé DESSIN ↔ CALCUL : chaque couche de la coupe ⊥ Y (points Y, traits X) est à h − e (inf.) ou e (sup.) de l'axe que le
# MOTEUR a reçu, renfort secondaire compris — rien ne montre un niveau autre que celui du calcul
r_s3 = MO.verifier(POS.pour_moteur(d_s3))
coupe_s3 = s3["vues"][0]["svg"].split("Coupe ⊥")[1]
y0_ = None
faux = []
for cote in ("inf", "sup"):
    for dk in ("y", "x"):
        for j, c_ in enumerate(r_s3["sections"][0]["faces"][f"{cote}_{dk}"]["couches"]):
            m_ = re.search(GROUPE.format(re.escape(f"{cote}_{dk}-{j}")), coupe_s3)
            ys = [float(x) for x in re.findall(r'cy="([\d.]+)"' if dk == "y" else r'y1="([\d.]+)"', m_.group(1))]
            att = (200.0 - c_["e"] * 10.0) if cote == "inf" else c_["e"] * 10.0
            y0_ = ys[0] - att if y0_ is None else y0_
            if any(abs(y - (y0_ + att)) > 0.06 for y in ys):
                faux.append((cote, dk, j, c_["e"], ys[:2]))
chk("dessin ↔ calcul : sur la coupe du site, chaque couche (bases, renfort secondaire) est à l'axe que le moteur a reçu, au dixième de mm",
    not faux and y0_ is not None, faux)
# l'AVIS d'ouverture : une note d'avant dont les chiffres changent le dit une fois, en tête du résultat ; jamais une note d'aujourd'hui
rr = req(action="calculer", etat=ra["etat"])
chk("l'avis d'une note d'avant est un AVIS (« msg », pas « grave »), en tête du résultat, dit UNE fois : l'état rendu ne le porte pas, et la "
    "note rouverte telle qu'enregistrée (version 3) ne le redit pas",
    avis(ra) == P.AVIS["note"] and '<div class="msg grave">' + P.AVIS["note"] not in ra["resultat"] and P.CLE_AVIS not in ra["etat"]
    and avis(rr) is None and P.CLE_AVIS not in rr["etat"] and lignes(rr) == lignes(ra), (avis(ra), avis(rr)))
chk("… dit aussi quand la première requête sur la note d'avant est une saisie (la note relue, puis modifiée)",
    avis(saisir({"version": 1, "dalles": [{"nom": "Ancienne", "h": 18}], "actif": 0}, "sections/0/My_inf", "10")) == P.AVIS["note"])
v2 = {"version": 2, "dalles": [dict(sans_site(d0), type="coulee", verrou=False, empilement=True)], "actif": 0}
v2_r = copy.deepcopy(v2)
v2_r["dalles"][0]["sections"][0]["faces"]["inf_x"]["couches"].append(EN.nouvelle_couche(base=False))
v1_saisie = {"version": 1, "dalles": [copy.deepcopy(sans_site(d0))], "actif": 0}
for f_ in ("inf_x", "sup_x"):
    v1_saisie["dalles"][0]["sections"][0]["faces"][f_]["couches"][0]["dist"] = 4.0
chk("seulement quand les chiffres changent : une note du 09/11 dont l'option était cochée et sans renfort secondaire ne le dit pas (ses bases "
    "secondaires étaient déjà posées ainsi) ; avec un renfort secondaire automatique, oui (S3 le monte) ; une note d'avant dont les axes "
    "secondaires sont SAISIS non plus (rien ne bouge) ; une note neuve jamais",
    avis(req(action="calculer", etat=v2)) is None and avis(req(action="calculer", etat=v2_r)) == P.AVIS["note"]
    and avis(req(action="calculer", etat=v1_saisie)) is None and avis(r0) is None
    and P.chiffres_changes(actif(P.normaliser(v2_r)), True) and not P.chiffres_changes(actif(P.normaliser(v2)), True))
# S5 : la même note cochée, avec un RENFORT PRINCIPAL plus haut que sa base (3 Ø25 en inf. Y) — sa base secondaire, que l'option posait
# sur la base principale, repose désormais sur le renfort : ses chiffres changent, l'avis le dit ; un renfort principal Ø10, au même
# dessus que la base treillis Ø10, ne change rien et ne dit rien
v2_s5, v2_s5_bas = copy.deepcopy(v2), copy.deepcopy(v2)
v2_s5["dalles"][0]["sections"][0]["faces"]["inf_y"]["couches"].append(dict(EN.nouvelle_couche(base=False), d=25))
v2_s5_bas["dalles"][0]["sections"][0]["faces"]["inf_y"]["couches"].append(dict(EN.nouvelle_couche(base=False), d=10))
r_v2s5 = req(action="calculer", etat=v2_s5)
chk("S5 — une note d'avant cochée dont la base secondaire repose désormais sur un renfort principal plus haut (3 Ø25 en inf. Y : la base "
    "inf. X 4,5 → 6,5) le dit à l'ouverture (S3 seul ne l'aurait pas vu : aucun renfort secondaire) ; un renfort principal Ø10 (même dessus "
    "que la base) ne change rien et ne dit rien",
    avis(r_v2s5) == P.AVIS["note"] and P.chiffres_changes(actif(P.normaliser(v2_s5)), True)
    and not P.chiffres_changes(actif(P.normaliser(v2_s5_bas)), True) and avis(req(action="calculer", etat=v2_s5_bas)) is None
    and axe_inf_x(actif(r_v2s5["etat"])) == 6.5 and axe_inf_x(actif(P.normaliser(v2_s5_bas))) == 4.5,
    (avis(r_v2s5), axe_inf_x(actif(r_v2s5["etat"]))))
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

# ------------------------------------------------------------------ 8 bis. les nappes confondues (relecture du 09/11/2026)
print("\n=== 8 bis. Les nappes confondues, dites selon leur cause ===")


def messages(rep):
    """Les messages du résultat, en texte."""
    return [re.sub(r"<[^>]+>", "", m).replace("&#x27;", "'") for m in re.findall(r'<div class="msg[^"]*">(.*?)</div>', rep["resultat"], flags=re.S)]


def note_confondue(type_, *saisies):
    e_ = req(action="nouveau")["etat"]
    for ch, v in (("type", type_),) + saisies:
        o_ = req(action="modifier", etat=e_, chemin=ch.split("/"), valeur=v)
        assert o_["ok"] and not o_["message"], (ch, v, o_.get("message"))
        e_ = o_["etat"]
    return o_


SAISIE_SUP = (("sections/0/faces/sup_x/couches/0/dist", "3,5"), ("sections/0/My_sup", "20"))
PHRASE_SAISI = ("un axe saisi les y pose", "Rendre l'axe saisi à l'automatique (↺) pose la direction secondaire sur la principale")


def ctl_confondues():
    """Faces SUPÉRIEURES (le site y empile : seul un axe saisi confond les bases), dalle coulée ET prédalle : la même
    phrase, qui dit comment rendre l'axe ; faces INFÉRIEURES d'une prédalle (la règle de pose : h_p 4 cm, base Ø16) :
    la phrase de la prédalle, « comme dans l'écran Streamlit »."""
    sup_c = [m for m in messages(note_confondue("coulee", *SAISIE_SUP)) if "confondu" in m or "niveau" in m]
    sup_p = [m for m in messages(note_confondue("predalle", *SAISIE_SUP)) if "confondu" in m or "niveau" in m]
    inf_p = [m for m in messages(note_confondue("predalle", ("h_pre", "4"), ("sections/0/nappes/inf/mode", "barres"),
                                                ("sections/0/faces/inf_y/couches/0/d", "16"), ("sections/0/My_inf", "25")))
             if "niveaux trop proches" in m or "même niveau" in m]
    return (len(sup_c) == len(sup_p) == len(inf_p) == 1
            and all(t in m for t in PHRASE_SAISI for m in sup_c + sup_p) and "comme dans l'écran Streamlit" not in sup_p[0]
            and "les bases X et Y inférieures" in inf_p[0] and "comme dans l'écran Streamlit" in inf_p[0]
            and not any(t in inf_p[0] for t in PHRASE_SAISI))


chk("nappes confondues par un axe saisi (faces supérieures) : la dalle coulée ET la prédalle disent « un axe saisi les y pose … "
    "Rendre l'axe saisi à l'automatique (↺) » (non « comme dans l'écran Streamlit … Saisir son axe », faux depuis que le site "
    "empile) ; les bases inférieures d'une prédalle trop proches par la règle de pose gardent la phrase de la prédalle",
    ctl_confondues(), [messages(note_confondue("predalle", *SAISIE_SUP))])

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
    return r_["ok"] and lignes(r_) == lignes_moteur(MO_P.verifier(POS.pour_moteur(actif(r_["etat"]))), AP)


muter("le moteur des dalles appelé quel que soit le type", P, "_MO", lambda d: MO, ctl_moteur)


def ctl_bascule():
    o_ = req(action="modifier", etat=saisir(e0, "h", "5")["etat"], chemin=["type"], valeur="predalle")
    return o_["ok"] and not o_["message"] and actif(o_["etat"]).get("h_pre") == 4.0


def _type_sans_peau(d, valeur):
    d["type"] = valeur


muter("la bascule vers la prédalle n'ajoute pas h_pre bornée à h_pre_max(h)", P, "_changer_type", _type_sans_peau, ctl_bascule)
muter("le type du fichier Streamlit n'est plus déduit (le lecteur des dalles lit tout)", P, "FI_P", P.FI,
      lambda: actif(saisir(copy.deepcopy(e0), "fichier_streamlit", {"nom": "p.json", "texte": json.dumps(FP)})["etat"]).get("h_pre") == 5.0)
muter("une case lit n'importe quel texte comme cochée", P, "_lire_bool", lambda v: bool(v),
      lambda: "coché ou décoché" in (req(action="modifier", etat=e0, chemin=["verrou"], valeur="peut-être")["message"] or ""))
# l'OPTION qui revient (décision du 09/11/2026 : les nappes toujours empilées)
_axe_auto = POS.axe_auto


def _axe_option(d, s, f, j):
    """L'axe d'avant la décision : la clé « empilement » absente ou fausse → l'automatique du moteur seul."""
    return POS._auto_moteur(d, s["faces"][f]["couches"][j], j, f) if not d.get("empilement") else _axe_auto(d, s, f, j)


muter("l'option revient dans le calcul (une clé « empilement » absente se lit fausse : les nappes au même niveau)", POS, "axe_auto", _axe_option,
      lambda: all(axe_inf_x(actif(req(action="calculer", etat=e_)["etat"])) == 4.5
                  for e_ in ({"version": 1, "dalles": [{"nom": "A"}]}, {"version": 2, "dalles": [{"nom": "A", "empilement": False}]})))
_options = P._options
muter("l'option revient à l'écran (la case « Nappes empilées » dans les options)", P, "_options",
      lambda d: _options(d) + C.case(["empilement"], True, "Nappes empilées : direction secondaire posée sur la principale"),
      lambda: 'data-chemin="empilement"' not in req(action="nouveau")["donnees"])
muter("la clé d'avant n'est plus relue sans effet (une note qui porte « empilement » est refusée)", P, "CLES_ANCIENNES", (),
      lambda: all(req(action="calculer", etat={"version": 2, "dalles": [{"nom": "T", "empilement": v}]})["ok"] for v in (True, False)))
muter("l'avis d'ouverture ne se dit plus", P, "chiffres_changes", lambda *a, **k: False,
      lambda: avis(req(action="calculer", etat={"version": 1, "dalles": [{"nom": "A"}]})) == P.AVIS["note"])
muter("S5 oublié par l'avis (une note cochée : seul un renfort secondaire change, comme avant S5)", P, "chiffres_changes",
      lambda d, deja=False: any(POS.empilee(d, s, f, j) and (j > 0 or not deja) for s in d["sections"] for f in EN.FACES
                                for j in range(len(s["faces"][f]["couches"]))),
      lambda: avis(req(action="calculer", etat=v2_s5)) == P.AVIS["note"])
_resultat = P.resultat


def _resultat_qui_garde(etat):
    """resultat qui laisse l'avis dans l'état : la note enregistrée le redirait à chaque ouverture."""
    garde = etat.get(P.CLE_AVIS)
    out = _resultat(etat)
    if garde:
        etat[P.CLE_AVIS] = garde
    return out


muter("l'avis reste dans l'état rendu (il se redirait à chaque ouverture)", P, "resultat", _resultat_qui_garde,
      lambda: P.CLE_AVIS not in req(action="calculer", etat={"version": 1, "dalles": [{"nom": "A"}]})["etat"])
muter("une prédalle dit ses bases supérieures confondues avec la phrase d'avant l'empilement (« comme dans l'écran Streamlit … "
      "Saisir son axe ») — la relecture du 09/11/2026 retirée", P, "_avertissements_de",
      lambda d, r, i=None: AP.avertissements(r, i) if P.est_predalle(d) else P._avertissements(r, i), ctl_confondues)

# ------------------------------------------------------------------ les NOMS trop longs d'une note relue (petits chantiers, 10/11/2026)
# Avant : un nom de dalle ou de section de plus de AP.NOM_MAX (60) caractères faisait REFUSER toute l'ancienne note
# (``_bornes_site``) — elle ne s'ouvrait plus. Désormais, à la relecture (note enregistrée, .stab, fichier Streamlit), le
# nom est COUPÉ à la limite et un avis le dit une fois, à l'ouverture ; la SAISIE garde son refus clair.
print("\n=== 10. Les noms de plus de 60 caractères d'une note relue : coupés, et l'avis le dit ===")
LONG_D, LONG_S = "Dalle du plancher haut du rez-de-chaussée, travée centrale entre les files B et C", "S" * 64
n_long = {"version": 3, "dalles": [dict(sans_site(d0), type="coulee", verrou=False, nom=LONG_D,
                                        sections=[dict(copy.deepcopy(sans_site(d0)["sections"][0]), nom=LONG_S)])], "actif": 0}
r_long = req(action="calculer", etat=n_long)
d_long = actif(r_long["etat"]) if r_long.get("ok") else {}
chk(f"une note dont la dalle ({len(LONG_D)} caractères) et une section ({len(LONG_S)}) ont un nom trop long S'OUVRE : les noms coupés à "
    f"{AP.NOM_MAX} caractères (rien d'autre ne change), l'état rendu ne porte pas la liste",
    r_long.get("ok") and d_long["nom"] == LONG_D[:AP.NOM_MAX].rstrip() and d_long["sections"][0]["nom"] == LONG_S[:AP.NOM_MAX]
    and len(d_long["nom"]) <= AP.NOM_MAX and P.CLE_NOMS not in r_long["etat"]
    and d_long["sections"][0]["My_inf"] == sans_site(d0)["sections"][0]["My_inf"] and lignes(r_long) == lignes(req(action="calculer", etat=r_long["etat"])),
    r_long.get("erreur") or d_long.get("nom"))
_av = r_long.get("avis") or []
chk("… et l'avis le dit, en tête du résultat ET dans « avis » (le message de la page au téléphone) : la limite, chaque nom coupé et sa "
    "longueur d'origine — un avis (« msg »), pas une erreur",
    len(_av) == 1 and f"coupé à {AP.NOM_MAX}" in _av[0] and f"({len(LONG_D)} caractères)" in _av[0] and f"({len(LONG_S)} caractères)" in _av[0]
    and C.esc(LONG_D[:AP.NOM_MAX].rstrip()) in _av[0] and f'<div class="msg">{_av[0]}</div>' in r_long["resultat"], _av)
r_long2 = req(action="calculer", etat=r_long["etat"])
chk("… UNE fois : la note rouverte telle qu'enregistrée ne le redit pas", "avis" not in r_long2 and "coupé à" not in r_long2["resultat"], r_long2.get("avis"))
chk("la SAISIE garde son refus clair : un nom de 61 caractères est refusé, la dalle inchangée",
    "60 caractères au plus (61 saisis)" in (req(action="modifier", etat=r_long["etat"], chemin=["nom"], valeur="N" * 61)["message"] or ""))
_fd_long = {"dalles": [{"id": 1, "nom": "D" * 70, "sections": [{"id": 1, "nom": "A"}]}], "values": {"dal1_sec1_M_inf": 28.0}}
r_fl = req(action="modifier", etat=copy.deepcopy(e0), chemin=["fichier_streamlit"], valeur={"nom": "long.json", "texte": json.dumps(_fd_long)})
chk("un fichier Streamlit dont la dalle a un nom de 70 caractères est repris (nom coupé, l'avis le dit) au lieu d'être refusé",
    r_fl.get("ok") and actif(r_fl["etat"])["nom"] == "D" * AP.NOM_MAX and any("(70 caractères)" in a for a in r_fl.get("avis") or [])
    and "refusé" not in (r_fl.get("message") or ""), (r_fl.get("message"), r_fl.get("avis")))
from api_web import note_dalle as ND  # noqa: E402
_vu = {}
_fichier = ND.fichier
ND.fichier = lambda etat, infos=None: _vu.setdefault("etat", etat)
try:
    P.telecharger(P.normaliser(copy.deepcopy(n_long)), "note_pdf", {})
finally:
    ND.fichier = _fichier
chk("la note PDF ne reçoit pas la liste des noms coupés (un avis de l'écran), mais les noms coupés",
    P.CLE_NOMS not in _vu.get("etat", {P.CLE_NOMS: 1}) and actif(_vu["etat"])["nom"] == LONG_D[:AP.NOM_MAX].rstrip(), list(_vu.get("etat", {})))
muter("un nom trop long refuse de nouveau toute la note (rien n'est coupé)", P, "_couper_noms", lambda d: [],
      lambda: req(action="calculer", etat=n_long).get("ok") is True)
muter("le nom est coupé sans le dire", P, "avis_noms", lambda c: "",
      lambda: bool(req(action="calculer", etat=n_long).get("avis")))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

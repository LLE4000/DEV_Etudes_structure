# -*- coding: utf-8 -*-
"""L'API du site pour le module « Choix de profilé » — la vérification d'une barre en acier (api_web/profil.py et ses
fichiers : profil_saisie, profil_resultat, profil_mots, dessin_barre), vérifiée en Python ordinaire, à travers le routeur.

  1. LA NOTE NEUVE — une barre chargée qui tient par ses deux méthodes ; un verdict par méthode, aux taux du moteur.
  2. LE RÉSULTAT EST CELUI DU MOTEUR — chaque vérification de ``acier.barre.moteur`` a sa ligne, son taux, sa fenêtre
     de calcul ; chaque formule y est écrite par ``acier.barre.ecriture`` ; la comparaison, les paramètres à confirmer.
  3. LES SAISIES — chaque champ va au bon endroit ; les listes (travées, cas, charges, maintiens) ; une saisie hors
     domaine est REFUSÉE avec un message et la barre reste intacte ; un changement de portée ramène les charges.
  4. LES DEUX MODES — barre chargée, efforts imposés : changer de mode ne perd rien.
  5. LES MÉTHODES — historique, Eurocode 3 ou les deux.
  6. LES ANCIENNES NOTES (version 1) — relues en « efforts imposés », Eurocode seul.
  7. LE SÉLECTEUR — trié par masse, les taux du moteur, le retenu ✓ et le plus léger qui tient ★.
  8. PLUSIEURS BARRES — ajouter, copier, supprimer ; le résumé de la note ; le calcul gardé.
  9. LES VUES ET LA SÛRETÉ — du SVG bien formé ; un nom saisi est échappé partout ; aucune ressource externe.

    /usr/bin/python3.12 tests/test_api_web_profil.py
"""
import copy
import html
import json
import os
import re
import sys
import xml.dom.minidom

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from acier.barre import ecriture as W  # noqa: E402
from acier.barre import entree as EN  # noqa: E402
from acier.barre import moteur as MO  # noqa: E402
from acier.profiles import catalogue as CAT  # noqa: E402
from api_web import commun as C  # noqa: E402
from api_web import profil as P  # noqa: E402
from api_web import profil_resultat as PR  # noqa: E402
from api_web import routeur as R  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:400]}"))


def req(**k):
    return json.loads(R.traiter(json.dumps(dict(module="profil", **k))))


def saisir(etat, chemin, valeur):
    """Comme l'écran : le chemin lu dans data-chemin, coupé aux « / », la valeur telle que tapée."""
    out = req(action="modifier", etat=etat, chemin=chemin.split("/"), valeur=valeur)
    assert out["ok"], out
    return out


def enchainer(etat, *saisies):
    """Plusieurs saisies de suite ; rend la dernière réponse. Une saisie refusée lève."""
    out = None
    for chemin, valeur in saisies:
        out = saisir(etat, chemin, valeur)
        assert out["message"] is None or "ramenés" in out["message"], (chemin, valeur, out["message"])
        etat = out["etat"]
    return out


def note(*barres):
    return {"version": 2, "elements": list(barres) or [P.barre_neuve()], "actif": 0}


def barre(**k):
    b = P.barre_neuve("B1")
    b.update(k)
    return b


def texte(h):
    """Ce qu'on LIT d'un fragment : sans balises, les entités rendues, les espaces insécables ramenées à des espaces."""
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", h)).replace("\u00a0", " ").replace("\u202f", " ").split())


def pastille(t):
    """Le texte de la pastille d'un taux, comme l'écran l'écrit."""
    return f"{C.pc(C.pct(t))} %"


def moteur(el):
    return MO.verifier(el)


POTEAU = barre(profil="HEB 200", mode="efforts", flambement={"y": {"L": 5.0, "u": 1.0}, "z": {"L": 5.0, "u": 1.0}}, Lc=5.0,
               efforts={"service": {"N": 300.0, "My": 20.0, "Mz": 5.0, "Vz": 30.0, "Vy": 0.0}, "elu": {"N": 420.0, "My": 28.0, "Mz": 7.0, "Vz": 42.0, "Vy": 0.0}})

print("=== 1. La note neuve")
neuf = req(action="nouveau")
e0 = neuf["etat"]
el0 = e0["elements"][0]
r0 = moteur(el0)
chk("une note neuve : version 2, une barre B1 — IPE 300, S235, deux appuis de 6 m, G et Q de 5 kN/m, poids propre, tenue à mi-portée",
    neuf["ok"] and e0["version"] == 2 and len(e0["elements"]) == 1 and e0["actif"] == 0 and el0["nom"] == "B1" and el0["profil"] == "IPE 300"
    and el0["nuance"] == "S235" and el0["mode"] == "chargee" and el0["modele"] == "appuis" and el0["travees"] == [6.0]
    and [c["nom"] for c in el0["cas"]] == ["G", "Q"] and el0["poids_propre"] is True and el0["maintiens"] == [{"x": 3.0, "genre": "complet"}]
    and el0["methodes"] == ["historique", "ec3"], el0)
th, te = r0["methodes"]["historique"]["taux_max"], r0["methodes"]["ec3"]["taux_max"]
chk("elle tient par ses DEUX méthodes (historique 74 %, Eurocode 3 52 %) : la note est « Conforme », son résumé porte le taux le plus fort",
    r0["statut"] == "ok" and round(th, 2) == 0.74 and round(te, 2) == 0.52 and neuf["resume"]["statut"] == "Conforme"
    and neuf["resume"]["taux"] == C.pct(max(th, te)) and neuf["resume"]["niveau"] == "p"
    and neuf["resume"]["gouvernant"] == "B1 · IPE 300 · flexion (historique)", (th, te, neuf["resume"]))
res0 = texte(neuf["resultat"])
chk("un VERDICT par méthode : le taux du moteur, la vérification qui gouverne, ce qu'elle compare, sa combinaison et son lieu",
    neuf["resultat"].count('class="verdict"') == 2 and f"η {pastille(th)} Méthode historique · flexion σ 115,6 ≤ σ adm 156,7 N/mm²" in res0
    and "Poids propre + G + Q · x = 3,00 m" in res0 and f"η {pastille(te)} Eurocode 3 · déversement M y,Ed 66,6 ≤ M b,Rd 128,3 kN·m" in res0
    and "1,35·Poids propre + 1,35·G + 1,5·Q · de 0,00 à 3,00 m" in res0, res0[:500])
chk("la tête dit la barre : « B1 », « IPE 300 · S235 · 6,00 m » ; le pied, la masse et la flèche de service (une information)",
    '<div class="et-rp"><b>B1</b><span>IPE 300 · S235 · 6,00 m</span></div>' in neuf["resultat"] and "Masse du profilé 42,2 kg/m" in res0
    and "Flèche de service 10,0 mm · L/599" in res0, res0[-300:])
chk("la colonne de gauche : la ligne de la barre et ses quatre rubriques — Barre, Charges, Maintiens et flambement, Méthodes",
    [m for m in re.findall(r'<span class="tt">([^<]+)</span>', neuf["donnees"])] == ["Barre", "Charges", "Maintiens et flambement", "Méthodes"]
    and 'data-chemin="nom"' in neuf["donnees"] and 'data-pk="profil"' in neuf["donnees"], re.findall(r'<span class="tt">([^<]+)</span>', neuf["donnees"]))

print("=== 2. Le résultat est celui du moteur")


def lignes_du_moteur(rep, r):
    """Pour chaque vérification du moteur : sa ligne est-elle à l'écran avec son taux, et sa fenêtre existe-t-elle ?"""
    manques = []
    for k, x in r["methodes"].items():
        for v in x["verifications"]:
            f = PR.fenetre(k, v["id"])
            m = re.search(rf'<div class="el vc[^"]*" data-calcul="{re.escape(f)}"[^>]*>(.*?)</div></div>', rep["resultat"], re.S)
            if not m:
                manques.append((k, v["id"], "ligne absente"))
                continue
            t = texte(m.group(1))
            if v["titre"] not in t or (v["taux"] is not None and pastille(v["taux"]) not in t):
                manques.append((k, v["id"], t))
            if f'id="ck-{f}"' not in rep["volets"]:
                manques.append((k, v["id"], "fenêtre absente"))
    return manques


CAS = {"la note neuve": note(), "un poteau-poutre en efforts imposés": note(POTEAU),
       "une poutre continue à trois cas, un maintien d'une semelle": note(barre(
           profil="HEA 200", modele="continue", travees=[5.0, 4.0], maintiens=[{"x": 2.5, "genre": "haut"}],
           cas=[{"nom": "G", "nature": "permanent", "N": 0.0, "charges": [{"type": "q", "v": 6.0, "a": 0.0, "b": 9.0}]},
                {"nom": "Q", "nature": "exploitation_B", "N": 0.0, "charges": [{"type": "q", "v": 8.0, "a": 0.0, "b": 5.0}, {"type": "P", "v": 12.0, "a": 7.0}]},
                {"nom": "S", "nature": "neige", "N": 0.0, "charges": [{"type": "tri", "v": 3.0, "a": 5.0, "b": 9.0, "sens": "croissante"}]}])),
       "une barre nue que la méthode historique refuse (IPE 300, 6 m sans maintien)": note(barre(maintiens=[])),
       "un tube carré comprimé et fléchi": note(barre(profil="SHS 150x150x6", nuance="S355", cas=[
           {"nom": "G", "nature": "permanent", "N": 250.0, "charges": [{"type": "q", "v": 4.0, "a": 0.0, "b": 6.0}]},
           {"nom": "V", "nature": "vent", "N": 0.0, "charges": [{"type": "P", "v": 9.0, "a": 2.0}]}]))}
REPONSES = {}
for titre, etat in CAS.items():
    rep = req(action="calculer", etat=etat)
    REPONSES[titre] = rep
    r = moteur(rep["etat"]["elements"][0])
    n = sum(len(x["verifications"]) for x in r["methodes"].values())
    manques = lignes_du_moteur(rep, r)
    chk(f"{titre} : chacune des {n} vérifications du moteur a sa ligne (son titre, son taux) et sa fenêtre de calcul", rep["ok"] and n >= 4 and not manques,
        manques[:3])
    # les verdicts : le taux le plus fort de chaque méthode
    ecarts = [k for k, x in r["methodes"].items() if x["taux_max"] is not None and f"η {pastille(x['taux_max'])} {PR.M.METHODES[k]}" not in texte(rep["resultat"])]
    chk(f"… et chaque verdict porte le taux le plus fort de sa méthode ; le résumé de la note, le plus fort des deux", not ecarts
        and rep["resume"]["taux"] == C.pct(max(x["taux_max"] for x in r["methodes"].values())), (ecarts, rep["resume"]))

rep = REPONSES["une poutre continue à trois cas, un maintien d'une semelle"]
r = moteur(rep["etat"]["elements"][0])
absentes, n = [], 0
for k, x in r["methodes"].items():
    par = {f["cle"]: f for f in x["calcul"]}
    for v in x["verifications"]:
        fen = re.search(rf'<div class="flyout calcul" id="ck-{re.escape(PR.fenetre(k, v["id"]))}">(.*?)</div></div></div>', rep["volets"], re.S).group(1)
        t = texte(fen)
        for cle in v["formules"]:
            f = par[cle]
            if f["gabarit"] == "taux":
                continue
            n += 1
            parts, res = W.morceaux(f)
            if res not in t or parts[2] not in t:
                absentes.append((v["id"], cle, res))
chk(f"les fenêtres de calcul portent CHAQUE formule du moteur ({n}), écrite par acier.barre.ecriture : ses valeurs et son résultat",
    n > 20 and not absentes, absentes[:3])
vol = texte(rep["volets"])
chk("la fenêtre d'une vérification dit la méthode, la barre, la combinaison et le lieu ; ses hypothèses sont celles du moteur",
    "Eurocode 3 · HEA 200 · S235 · 5,00 + 4,00 m ·" in vol and all(texte(PR.M.sub(h)) in vol for x in r["methodes"].values() for h in x["hypotheses"]),
    vol[:300])
chk("« tout le calcul » de chaque méthode (s1, s2) porte toutes ses vérifications et le bilan de ses combinaisons",
    all(f'id="ck-s{i}"' in rep["volets"] for i in (1, 2)) and "Combinaisons" in vol
    and all(c["nom"] in vol for x in r["methodes"].values() for c in x["combinaisons"]))
cmp_ = re.search(r'<section class="et-bloc inc" data-theme="3">(.*?)</section>', rep["resultat"], re.S)
chk("la COMPARAISON : un sujet par ligne, le taux de chaque méthode — ceux que le moteur rend ; sa fenêtre (cmp)",
    cmp_ is not None and all(li["sujet"] in texte(cmp_.group(1)) for li in r["comparaison"])
    and all(pastille(li[k]) in texte(cmp_.group(1)) for li in r["comparaison"] for k in ("historique", "ec3") if li[k] is not None)
    and 'data-calcul="cmp"' in rep["resultat"] and 'id="ck-cmp"' in rep["volets"], texte(cmp_.group(1)) if cmp_ else None)
ac = r["methodes"]["ec3"]["a_confirmer"]
chk(f"les paramètres « à confirmer » de l'annexe nationale ({len(ac)}) : un lien dans la rubrique Méthodes, une fenêtre (ac) avec leur valeur et leur source",
    len(ac) >= 2 and f"{len(ac)} paramètres à confirmer" in texte(rep["donnees"]) and 'data-calcul="ac"' in rep["donnees"]
    and 'id="ck-ac"' in rep["volets"] and all(p["source"] in vol for p in ac))
nu = REPONSES["une barre nue que la méthode historique refuse (IPE 300, 6 m sans maintien)"]
chk("une barre que la méthode historique refuse (179 %) et que l'Eurocode accepte (92 %) : la note est NON CONFORME, les deux verdicts se lisent",
    nu["resume"]["niveau"] == "r" and nu["resume"]["statut"] == "Non conforme" and nu["resume"]["taux"] == 179
    and "η 179 % Méthode historique" in texte(nu["resultat"]) and "η 92 % Eurocode 3" in texte(nu["resultat"])
    and 'class="et-bloc nok"' in nu["resultat"] and 'class="et-bloc ok"' in nu["resultat"], nu["resume"])

print("=== 3. Les saisies")
POSES = [("nom", "Panne 3", lambda e: e["nom"] == "Panne 3"), ("travees/0", "7,25", lambda e: e["travees"] == [7.25]),
         ("poids_propre", False, lambda e: e["poids_propre"] is False), ("nuance", "S355", lambda e: e["nuance"] == "S355"),
         ("niveau", "bas", lambda e: e["niveau"] == "bas"), ("cas/0/nom", "Permanent", lambda e: e["cas"][0]["nom"] == "Permanent"),
         ("cas/1/nature", "neige", lambda e: e["cas"][1]["nature"] == "neige"), ("cas/0/N", "120", lambda e: e["cas"][0]["N"] == 120.0),
         ("cas/0/charges/0/v", "7,5", lambda e: e["cas"][0]["charges"][0]["v"] == 7.5), ("cas/0/charges/0/a", "1", lambda e: e["cas"][0]["charges"][0]["a"] == 1.0),
         ("cas/0/charges/0/b", "4", lambda e: e["cas"][0]["charges"][0]["b"] == 4.0), ("maintiens/0/x", "2", lambda e: e["maintiens"][0]["x"] == 2.0),
         ("maintiens/0/genre", "haut", lambda e: e["maintiens"][0]["genre"] == "haut"), ("flambement/y/L", "4,5", lambda e: e["flambement"]["y"]["L"] == 4.5),
         ("flambement/z/u", "0,7", lambda e: e["flambement"]["z"]["u"] == 0.7), ("methodes", "ec3", lambda e: e["methodes"] == ["ec3"]),
         ("methodes", "historique", lambda e: e["methodes"] == ["historique"]), ("historique/s", "1,33", lambda e: e["historique"]["s"] == 1.33),
         ("historique/C", "norme", lambda e: e["historique"]["C"] == "norme"), ("historique/deversement", False, lambda e: e["historique"]["deversement"] is False),
         ("historique/tolerance", True, lambda e: e["historique"]["tolerance"] is True),
         ("parametres/gamma_M1", "1,1", lambda e: e["parametres"] == {"gamma_M1": 1.1}),
         # (la borne des noms du site : 60 caractères pour un élément — le harnais en saisit 58 — ; 12 pour un cas, qui s'écrit sur le dessin)
         ("nom", "N" * 60, lambda e: e["nom"] == "N" * 60), ("cas/1/nom", "Exploitation", lambda e: e["cas"][1]["nom"] == "Exploitation")]
faux = []
for chemin, valeur, juste in POSES:
    out = req(action="modifier", etat=copy.deepcopy(e0), chemin=chemin.split("/"), valeur=valeur)
    if not out["ok"] or out["message"] or not juste(out["etat"]["elements"][0]):
        faux.append((chemin, valeur, out.get("message") or out.get("erreur")))
chk(f"chaque saisie va au bon endroit ({len(POSES)} champs : nom, portée, poids propre, acier, niveau, cas, charges, maintiens, flambement, méthodes, "
    "options de la méthode historique, γ_M1 — et un nom de 60 caractères, la borne du site)", not faux, faux[:4])
# les listes
out = saisir(e0, "cas/+", "1")
c3 = out["etat"]["elements"][0]["cas"]
chk("« + Cas de charge » : un cas variable de plus (Q2, exploitation A, une charge répartie sur toute la barre) ; « retirer » l'ôte",
    [c["nom"] for c in c3] == ["G", "Q", "Q2"] and c3[2]["nature"] == "exploitation_A" and c3[2]["charges"] == [{"type": "q", "v": 5.0, "a": 0.0, "b": 6.0}]
    and [c["nom"] for c in saisir(out["etat"], "cas/1/-", "1")["etat"]["elements"][0]["cas"]] == ["G", "Q2"], c3)
out = enchainer(e0, ("cas/0/charges/+", "P"), ("cas/0/charges/1/v", "12"), ("cas/0/charges/1/a", "2"))
ch = out["etat"]["elements"][0]["cas"][0]["charges"]
chk("« + Ponctuelle » : une charge de 10 kN à mi-portée, que l'on règle (12 kN à 2 m) ; son type se change (un couple au même endroit, puis une "
    "charge triangulaire sur toute la barre)", ch[1] == {"type": "P", "v": 12.0, "a": 2.0}
    and saisir(out["etat"], "cas/0/charges/1/type", "C")["etat"]["elements"][0]["cas"][0]["charges"][1] == {"type": "C", "v": 12.0, "a": 2.0}
    and saisir(out["etat"], "cas/0/charges/1/type", "tri")["etat"]["elements"][0]["cas"][0]["charges"][1]
    == {"type": "tri", "v": 12.0, "a": 0.0, "b": 6.0, "sens": "croissante"}
    and saisir(out["etat"], "cas/0/charges/0/-", "1")["etat"]["elements"][0]["cas"][0]["charges"] == [{"type": "P", "v": 12.0, "a": 2.0}], ch)
out = saisir(e0, "maintiens/+", "1")
mt = out["etat"]["elements"][0]["maintiens"]
chk("« + Maintien » : au milieu du plus long intervalle libre (1,5 m, puis 4,5 m) ; ils restent triés ; « retirer » l'ôte",
    mt == [{"x": 1.5, "genre": "complet"}, {"x": 3.0, "genre": "complet"}]
    and [m["x"] for m in saisir(out["etat"], "maintiens/+", "1")["etat"]["elements"][0]["maintiens"]] == [1.5, 3.0, 4.5]
    and saisir(out["etat"], "maintiens/0/-", "1")["etat"]["elements"][0]["maintiens"] == [{"x": 3.0, "genre": "complet"}], mt)
out = saisir(e0, "modele", "continue")
ec = out["etat"]["elements"][0]
chk("« Continue » : une seconde travée égale à la première, les charges qui couvraient la barre la suivent (0 à 12 m) ; « + Travée », « retirer »",
    ec["travees"] == [6.0, 6.0] and ec["cas"][0]["charges"][0]["b"] == 12.0 and out["message"] is None
    and saisir(out["etat"], "travees/+", "1")["etat"]["elements"][0]["travees"] == [6.0, 6.0, 6.0]
    and "au moins deux travées" in saisir(out["etat"], "travees/0/-", "1")["message"], (ec["travees"], out["message"]))
out2 = saisir(out["etat"], "modele", "console")
chk("revenir à une seule travée ramène ce qui dépasse dans la barre et le DIT ; une charge entièrement dehors est retirée",
    out2["etat"]["elements"][0]["travees"] == [6.0] and out2["etat"]["elements"][0]["cas"][0]["charges"][0]["b"] == 6.0
    and saisir(enchainer(out["etat"], ("cas/1/charges/0/a", "7"), ("cas/0/charges/+", "P"), ("cas/0/charges/1/a", "9"), ("maintiens/+", "1"))["etat"],
               "modele", "appuis")["message"] == "Charges et maintiens ramenés dans la barre.", out2["message"])
court = saisir(e0, "travees/0", "4")["etat"]["elements"][0]
chk("raccourcir la portée : la charge qui couvrait toute la barre la suit (0 à 4 m), le maintien à 3 m reste",
    court["cas"][0]["charges"][0]["b"] == 4.0 and court["maintiens"] == [{"x": 3.0, "genre": "complet"}], court["cas"][0])
auto = saisir(saisir(e0, "flambement/z/L", "2")["etat"], "flambement/z/L", "")
chk("une longueur de flambement VIDÉE redevient automatique : le champ est vide, son texte dit la longueur que le moteur prend (« auto · 3 »)",
    auto["etat"]["elements"][0]["flambement"]["z"]["L"] is None and 'placeholder="auto · 3"' in auto["donnees"]
    and 'placeholder="auto · 6"' in auto["donnees"], re.findall(r'placeholder="[^"]*"', auto["donnees"]))
# les refus
REFUS = [("travees/0", "-1", "portée"), ("travees/0", "abc", "n'est pas un nombre"), ("travees/0", "80", "entre 0,05 et 60"),
         ("cas/0/charges/0/b", "9", "cas « G »"), ("cas/1/nom", "G", "portent le nom"), ("cas/0/nature", "pluie", "nature inconnue"),
         ("cas/0/-", "1", None), ("nuance", "S999", "nuance inconnue"), ("maintiens/0/x", "7", "hors de la barre"), ("maintiens/0/genre", "milieu", "genre"),
         ("flambement/y/u", "0", "coefficient u"), ("flambement/y/L", "abc", "n'est pas un nombre"), ("historique/s", "5", "entre 1 et 3"),
         ("parametres/gamma_M0", "2", "entre 0,9 et 1,5"), ("methodes", "aci", "méthode inconnue"), ("mode", "libre", "mode inconnu"),
         ("modele", "arc", "modèle inconnu"), ("inconnu", "1", "Champ inconnu"), ("cas/9/N", "1", "introuvable"), ("travees/+", "1", "poutre continue"),
         ("nom", "N" * 61, "au plus 60"), ("cas/1/nom", "Exploitation2", "au plus 12")]
faux = []
for chemin, valeur, mot in REFUS:
    out = req(action="modifier", etat=copy.deepcopy(e0), chemin=chemin.split("/"), valeur=valeur)
    if chemin == "cas/0/-":                       # (retirer un cas quand il en reste un : permis)
        if not out["ok"] or out["message"]:
            faux.append((chemin, out.get("message")))
        continue
    if not out["ok"] or not out["message"] or mot not in out["message"] or out["etat"] != e0:
        faux.append((chemin, valeur, out.get("message") or out.get("erreur")))
chk(f"une saisie hors domaine est REFUSÉE avec le message du moteur et la barre reste INTACTE ({len(REFUS) - 1} refus : portée négative ou "
    "illisible, charge hors de la barre, deux cas du même nom, nature, nuance, maintien hors de la barre, u nul, s, γ, champ inconnu, un nom de 61 "
    "caractères, un nom de cas de 13…)", not faux, faux[:4])
e5 = e0
for _ in range(3):
    e5 = saisir(e5, "cas/+", "1")["etat"]
six = saisir(e5, "cas/+", "1")
chk("au plus quatre cas VARIABLES : le cinquième cas ajouté est permanent (G2) ; le changer en variable est refusé, le moteur le dit",
    [c["nature"] != "permanent" for c in e5["elements"][0]["cas"]] == [False, True, True, True, True]
    and six["etat"]["elements"][0]["cas"][-1] == dict(nom="G2", nature="permanent", N=0.0, charges=[{"type": "q", "v": 5.0, "a": 0.0, "b": 6.0}])
    and "au plus 4 cas de charge variables" in saisir(six["etat"], "cas/5/nature", "vent")["message"]
    and "au plus 6 cas de charge" in saisir(six["etat"], "cas/+", "1")["message"], [c["nom"] for c in six["etat"]["elements"][0]["cas"]])

print("=== 4. Les deux modes")
ef = saisir(e0, "mode", "efforts")
eb = ef["etat"]["elements"][0]
chk("passer en « efforts imposés » : les longueurs de flambement que la barre chargée prenait d'elle-même deviennent des données (6 m, 3 m) ; "
    "les charges restent dans la note", eb["mode"] == "efforts" and eb["flambement"] == {"y": {"L": 6.0, "u": 1.0}, "z": {"L": 3.0, "u": 1.0}}
    and eb["cas"] == el0["cas"] and eb["maintiens"] == el0["maintiens"] and ef["message"] is None, eb["flambement"])
chk("sans effort saisi : « aucun effort à vérifier » pour chaque méthode, aucun bloc ; la note est « Données incomplètes »",
    texte(ef["resultat"]).count("aucun effort à vérifier") == 2 and 'class="et-bloc ' not in ef["resultat"] and ef["resume"]["niveau"] == "n"
    and ef["resume"]["statut"] == "Données incomplètes" and ef["resume"]["taux"] is None, ef["resume"])
chk("la rubrique 2 devient « Efforts » (service et ELU côte à côte), la 3 « Flambement et déversement » ; la vue de la barre s'efface",
    [m for m in re.findall(r'<span class="tt">([^<]+)</span>', ef["donnees"])] == ["Barre", "Efforts", "Flambement et déversement", "Méthodes"]
    and 'data-chemin="efforts/service/My"' in ef["donnees"] and 'data-chemin="efforts/elu/My"' in ef["donnees"]
    and [v["cle"] for v in ef["vues"]] == ["section"] and [v["cle"] for v in neuf["vues"]] == ["barre", "section", "taux"], [v["cle"] for v in ef["vues"]])
ef2 = enchainer(ef["etat"], ("efforts/elu/N", "420"), ("efforts/elu/My", "28"), ("efforts/service/N", "300"), ("efforts/service/My", "20"),
                ("diagramme/y/forme", "lineaire"), ("diagramme/y/psi", "-0,5"), ("Lc", "2,5"))
eb2 = ef2["etat"]["elements"][0]
chk("les efforts imposés, la forme du diagramme (linéaire, ψ = −0,5) et la distance entre sections tenues vont au moteur : le résultat est le sien",
    eb2["efforts"]["elu"]["N"] == 420.0 and eb2["efforts"]["service"]["My"] == 20.0 and eb2["diagramme"]["y"] == {"forme": "lineaire", "psi": -0.5}
    and eb2["Lc"] == 2.5 and not lignes_du_moteur(ef2, moteur(eb2)) and 'data-chemin="diagramme/y/psi"' in ef2["donnees"], lignes_du_moteur(ef2, moteur(eb2))[:2])
retour = saisir(ef2["etat"], "mode", "chargee")
er = retour["etat"]["elements"][0]
chk("revenir en « barre chargée » : les charges et les maintiens sont ceux d'avant ; les efforts imposés restent dans la note pour un prochain retour",
    er["mode"] == "chargee" and er["cas"] == el0["cas"] and er["maintiens"] == el0["maintiens"] and er["efforts"]["elu"]["N"] == 420.0
    and er["diagramme"]["y"]["forme"] == "lineaire" and saisir(retour["etat"], "mode", "efforts")["etat"]["elements"][0]["efforts"]["elu"]["My"] == 28.0)
une = saisir(saisir(ef["etat"], "methodes", "ec3")["etat"], "efforts/elu/My", "50")
chk("une seule méthode en efforts imposés : une seule colonne d'efforts (ELU pour l'Eurocode, service pour l'historique)",
    'data-chemin="efforts/elu/My"' in une["donnees"] and 'data-chemin="efforts/service/My"' not in une["donnees"]
    and 'data-chemin="efforts/service/My"' in saisir(ef["etat"], "methodes", "historique")["donnees"])

print("=== 5. Les méthodes")
for choix, cle, autre in (("historique", "historique", "Eurocode 3"), ("ec3", "ec3", "Méthode historique")):
    out = saisir(e0, "methodes", choix)
    r = moteur(out["etat"]["elements"][0])
    chk(f"« {choix} » seul : un verdict, un bloc, ni comparaison ni {autre} ; le résumé est le sien", list(r["methodes"]) == [cle]
        and out["resultat"].count('class="verdict"') == 1 and out["resultat"].count('class="et-bloc ') == 1 and autre not in texte(out["resultat"])
        and "Comparaison" not in texte(out["resultat"]) and 'id="ck-cmp"' not in out["volets"] and out["resume"]["taux"] == C.pct(r["methodes"][cle]["taux_max"])
        and not lignes_du_moteur(out, r), texte(out["resultat"])[:200])
hi = saisir(e0, "methodes", "historique")
chk("la méthode historique seule : ses options s'affichent (s, C, déversement, tolérance), celles de l'Eurocode (γ) s'effacent — et l'inverse",
    'data-chemin="historique/s"' in hi["donnees"] and 'data-chemin="parametres/gamma_M0"' not in hi["donnees"]
    and 'data-chemin="historique/s"' not in saisir(e0, "methodes", "ec3")["donnees"])
g11 = saisir(e0, "parametres/gamma_M1", "1,1")
chk("γ_M1 = 1,1 va au moteur (le déversement passe de 52 à 57 %) ; le champ vidé rend la valeur de l'annexe nationale",
    round(moteur(g11["etat"]["elements"][0])["methodes"]["ec3"]["taux_max"] / te, 3) == 1.1
    and saisir(g11["etat"], "parametres/gamma_M1", "")["etat"]["elements"][0]["parametres"] == {}, moteur(g11["etat"]["elements"][0])["methodes"]["ec3"]["taux_max"])

print("=== 6. Les anciennes notes (version 1)")
V1 = {"version": 1, "actif": 0, "elements": [{"nom": "P1", "acier": "S355", "familles": ["HEB"], "My": 60.0, "Vz": 40.0, "avec_N": True, "N": 600.0,
                                               "avec_Mz": False, "Mz": 99.0, "flambement": True, "Lcr_y": 6.0, "Lcr_z": 3.0, "courbe": "c",
                                               "deversement": False, "L_LT": 4.0, "gamma_M0": 1.0, "gamma_M1": 1.1, "profil": "HEB 200"}]}
anc = req(action="calculer", etat=V1)
ea = anc["etat"]["elements"][0] if anc["ok"] else {}
chk("une note de la version 1 est relue en « efforts imposés », Eurocode 3 seul : ses efforts de calcul, son acier, son profilé, ses longueurs de "
    "flambement, son γ_M1 ; un effort dont la case n'était pas cochée est nul ; sans déversement coché, L_c = 0",
    anc["ok"] and anc["etat"]["version"] == 2 and ea["mode"] == "efforts" and ea["methodes"] == ["ec3"] and ea["nuance"] == "S355" and ea["profil"] == "HEB 200"
    and ea["efforts"]["elu"] == {"N": 600.0, "My": 60.0, "Mz": 0.0, "Vz": 40.0, "Vy": 0.0} and ea["flambement"] == {"y": {"L": 6.0, "u": 1.0}, "z": {"L": 3.0, "u": 1.0}}
    and ea["Lc"] == 0.0 and ea["parametres"] == {"gamma_M1": 1.1} and ea["nom"] == "P1", ea)
chk("… et son résultat est celui du moteur sur cette barre (flambement, flexion composée ; pas de déversement : L_c = 0)",
    not lignes_du_moteur(anc, moteur(ea)) and "E-Mb" not in {v["id"] for v in moteur(ea)["methodes"]["ec3"]["verifications"]}
    and anc["resume"]["taux"] == C.pct(moteur(ea)["methodes"]["ec3"]["taux_max"]), lignes_du_moteur(anc, moteur(ea))[:2])
V1b = copy.deepcopy(V1)
V1b["elements"][0].update({"profil": None, "familles": ["HEA", "HEB"]})
suivi = req(action="calculer", etat=V1b)
es = suivi["etat"]["elements"][0]
plus_legers = [n for n in P._noms(["HEA", "HEB"]) if moteur(dict(es, profil=n))["statut"] == "ok"]
chk("une ancienne note qui « suivait l'optimal » (aucun profilé choisi) reçoit le plus LÉGER de ses familles que l'Eurocode accepte aujourd'hui",
    suivi["ok"] and es["profil"] == plus_legers[0] and moteur(es)["statut"] == "ok", (es.get("profil"), plus_legers[:2]))
chk("une note relue est stable : la relire encore ne la change pas", req(action="calculer", etat=anc["etat"])["etat"] == anc["etat"])
chk("une note illisible est refusée avec un message qui nomme la barre", "barre 1" in req(action="calculer", etat=note(barre(profil="IPE 9999")))["erreur"]
    and "barre 2" in req(action="calculer", etat={"version": 2, "actif": 0, "elements": [P.barre_neuve(), "n'importe quoi"]})["erreur"])

print("=== 7. Le sélecteur")
sel = req(action="selecteur", etat=e0, role="profil")
rows = re.findall(r'<button class="pk-row([^"]*)"[^>]*data-choix="([^"]+)"[^>]*>(.*?)</button>', sel["html"], re.S)
noms = [n for _, n, _ in rows]
ipe = [p["nom"] for p in sorted((p for p in CAT.profils() if p["famille"] == "IPE"), key=lambda p: (p["masse"], p["nom"]))]
chk("par défaut : la famille du profilé de la barre (les 23 IPE), triée par masse", sel["ok"] and noms == ipe and len(noms) == 23, noms[:5])
ecarts = []
for cls, n, h in rows:
    r = moteur(dict(el0, profil=n))
    t = max(x["taux_max"] for x in r["methodes"].values())
    if f"{C.pc(C.pct(min(t, 9.99)))} %" not in h or ("r" in re.search(r'class="chip (\w)"', h).group(1)) != (r["statut"] == "nok"):
        ecarts.append((n, t, texte(h)))
chk("chaque profilé porte le taux le plus fort que le MOTEUR lui donne (les deux méthodes de la barre), rouge quand une méthode le refuse", not ecarts, ecarts[:3])
tiennent = [n for n in ipe if moteur(dict(el0, profil=n))["statut"] == "ok"]
chk("le retenu est coché (IPE 300) ; l'étoile marque le plus LÉGER qui tient — ici le retenu lui-même, donc pas d'étoile ailleurs",
    [n for c, n, _ in rows if " cur" in c] == ["IPE 300"] and [n for c, n, _ in rows if " opt" in c] == [tiennent[0]] and tiennent[0] == "IPE 300", tiennent[:2])
lourd = req(action="choisir", etat=e0, role="profil", nom="IPE 400")
sel2 = req(action="selecteur", etat=lourd["etat"], role="profil")
chk("choisir un autre profilé le retient (IPE 400) : coché dans le sélecteur, l'étoile reste sur l'IPE 300",
    lourd["etat"]["elements"][0]["profil"] == "IPE 400" and 'pk-row cur"' in re.sub(r'pk-row cur opt', 'x', sel2["html"]).replace(' aria-disabled="true"', "")
    and re.findall(r'<button class="pk-row opt"[^>]*data-choix="([^"]+)"', sel2["html"]) == ["IPE 300"], texte(lourd["resultat"])[:300])
# (le moteur partagé du site : un écran rend ce que rend un moteur NEUF — tests/test_moteur_partage.py, P1, a trouvé que la
#  ligne « Profilé » disait « plus léger : IPE 300 » seulement quand le sélecteur avait déjà vérifié la famille)
P._CANDIDATS.clear()
P._RESULTATS.clear()
neuf_ = {k: req(action="calculer", etat=lourd["etat"])[k] for k in ("resultat", "donnees", "volets")}
req(action="selecteur", etat=lourd["etat"], role="profil")
req(action="selecteur", etat=lourd["etat"], role="profil", familles=["HEA", "IPE"])
apres_ = {k: req(action="calculer", etat=lourd["etat"])[k] for k in ("resultat", "donnees", "volets")}
chk("le résultat d'une note ne dépend PAS de ce qui a été ouvert avant : le sélecteur ouvert (toute la famille vérifiée), la même note rend les mêmes "
    "résultats, la même colonne de gauche, les mêmes fenêtres de calcul, à l'octet ; le plus léger qui tient se lit dans le sélecteur (★), pas sur la ligne",
    neuf_ == apres_ and "plus léger" not in texte(apres_["resultat"]) and len(P._CANDIDATS) >= 23, [k for k in neuf_ if neuf_[k] != apres_[k]])
fam = req(action="selecteur", etat=e0, role="profil", familles=["HEA", "SHS"])
noms_f = re.findall(r'data-choix="([^"]+)"', fam["html"])
chk("d'autres familles demandées (HEA, SHS) : leurs profilés, triés par masse ; les puces des 17 familles vérifiables (I, H, tubes), celles-là allumées",
    fam["ok"] and {CAT.par_nom()[n]["famille"] for n in noms_f} == {"HEA", "SHS"} and [CAT.par_nom()[n]["masse"] for n in noms_f] == sorted(CAT.par_nom()[n]["masse"] for n in noms_f)
    and re.findall(r'class="fchip on" data-famille="([^"]+)"', fam["html"]) == ["HEA", "SHS"] and len(re.findall(r'data-famille="', fam["html"])) == 17
    and "UPE" not in re.findall(r'data-famille="([^"]+)"', fam["html"]), len(noms_f))
chk("un profilé inconnu, ou d'une famille que le module ne vérifie pas (une cornière), est refusé ; un sélecteur inconnu aussi",
    not req(action="choisir", etat=e0, role="profil", nom="IPE 9999")["ok"] and not req(action="choisir", etat=e0, role="profil", nom="L 100x100x10")["ok"]
    and not req(action="selecteur", etat=e0, role="autre")["ok"])
gros = req(action="selecteur", etat=note(barre(profil="HD 400x818.0", nuance="S355")), role="profil")
chk("un profilé qu'une méthode ne couvre pas (HD 400x818 en S355 : semelle de plus de 80 mm) le dit, sans couleur de conformité, et n'a pas d'étoile",
    gros["ok"] and re.search(r'data-choix="HD 400x818\.0"[^>]*>.*?<span class="chip c">', gros["html"], re.S) is not None
    and 'pk-row cur opt' not in gros["html"], re.findall(r'data-choix="HD 400x818\.0"[^>]*>(.*?)</button>', gros["html"], re.S)[:1])

print("=== 8. Plusieurs barres ; le calcul gardé")
a2 = req(action="ajouter", etat=e0)
d2 = req(action="dupliquer", etat=a2["etat"], index=0)
chk("ajouter une barre (B2, ouverte) ; copier la première (« B1 (copie) », insérée juste après, ouverte) ; supprimer ; ouvrir",
    [e["nom"] for e in a2["etat"]["elements"]] == ["B1", "B2"] and a2["etat"]["actif"] == 1
    and [e["nom"] for e in d2["etat"]["elements"]] == ["B1", "B1 (copie)", "B2"] and d2["etat"]["actif"] == 1
    and [e["nom"] for e in req(action="supprimer", etat=d2["etat"], index=1)["etat"]["elements"]] == ["B1", "B2"]
    and req(action="activer", etat=d2["etat"], index=2)["etat"]["actif"] == 2 and len(req(action="supprimer", etat=e0, index=0)["etat"]["elements"]) == 1)
mixte = req(action="calculer", etat={"version": 2, "actif": 0, "elements": [P.barre_neuve("Solive"), barre(nom="Panne", maintiens=[])]})
chk("le résumé de la NOTE est sa barre la plus chargée, son état le moins bon ; chaque barre a la couleur de son état dans la liste",
    mixte["resume"]["niveau"] == "r" and mixte["resume"]["taux"] == 179 and mixte["resume"]["gouvernant"].startswith("Panne · IPE 300 · flexion")
    and 'class="status r"' in mixte["statut_html"] and re.findall(r'<span class="nom et-n-(\w+)">', mixte["donnees"]) == ["ok", "nok"], mixte["resume"])
appels = [0]
verifier = MO.verifier


def espion(b):
    appels[0] += 1
    return verifier(b)


MO.verifier = espion
try:
    P._RESULTATS.clear()
    trois = {"version": 2, "actif": 0, "elements": [P.barre_neuve("A"), P.barre_neuve("B"), barre(nom="C", maintiens=[])]}
    req(action="calculer", etat=trois)
    n1 = appels[0]
    req(action="modifier", etat=trois, chemin=["nom"], valeur="Autre nom")
    n2 = appels[0]
    req(action="modifier", etat=trois, chemin=["travees", "0"], valeur="5,5")
    n3 = appels[0]
finally:
    MO.verifier = verifier
chk("le calcul est GARDÉ : trois barres dont deux identiques au nom près ne demandent que deux calculs ; renommer n'en demande aucun ; "
    "changer la portée de la barre ouverte, un seul", (n1, n2 - n1, n3 - n2) == (2, 0, 1), (n1, n2 - n1, n3 - n2))

print("=== 9. Les vues et la sûreté")
mal = []
for titre, rep in REPONSES.items():
    for v in rep["vues"]:
        try:
            xml.dom.minidom.parseString(v["svg"])
        except Exception as ex:  # noqa: BLE001
            mal.append((titre, v["cle"], str(ex)[:80]))
        if not v["entete"] or not v["titre"] or not v["court"]:
            mal.append((titre, v["cle"], "en-tête"))
chk("chaque vue est du SVG bien formé, avec son en-tête : la barre et ses enveloppes, la section, les taux", not mal
    and [v["cle"] for v in REPONSES["la note neuve"]["vues"]] == ["barre", "section", "taux"], mal[:3])
# chaque famille du module a sa section dessinée (trouvé par la note PDF : un tube rond n'a ni h ni b, le dessin plantait)
sans_dessin = []
for fam in P.FAMILLES:
    nom = next(p["nom"] for p in CAT.profils() if p["famille"] == fam)
    etat_f = copy.deepcopy(e0)
    rep = req(action="choisir", etat=etat_f, role="profil", nom=nom)
    if not rep.get("ok"):
        sans_dessin.append((fam, nom, rep.get("erreur")))
        continue
    v = next(v for v in rep["vues"] if v["cle"] == "section")["svg"]
    rond = "d" in P.fiche_profil(nom)["dims"]
    tube = "t" in P.fiche_profil(nom)["dims"]
    try:
        xml.dom.minidom.parseString(v)
    except Exception as err:  # noqa: BLE001
        sans_dessin.append((fam, nom, f"SVG mal formé : {err}"))
        continue
    if (">d = " in v) != rond or (">h = " in v) == rond or (">b = " in v) == rond or (">t = " in v) != tube or (">t<tspan" in v) == tube:
        sans_dessin.append((fam, nom, "cotes", rond, tube))
chk(f"chacune des {len(P.FAMILLES)} familles du module a sa section dessinée et son résultat : un I ou un H avec h, b, t_w, t_f ; un tube carré ou "
    "rectangulaire avec h, b, t ; un tube rond avec d et t (il n'a ni h ni b)", not sans_dessin and len(P.FAMILLES) == 17, sans_dessin[:3])
svg = REPONSES["la note neuve"]["vues"][0]["svg"]
env = r0["statique"]["enveloppes"]["ec3"]
chk("le dessin de la barre porte les charges de chaque cas (5 kN/m), la portée, et les extrêmes des enveloppes du moteur (M 66,6 kN·m ; V ± 44,4 kN)",
    svg.count("5 kN/m") == 2 and "L = 6,00 m" in svg and f">{C.fr(max(env['M_max']), 1)}<" in svg and f">{C.fr(max(env['V_max']), 1)}<" in svg
    and f">−{C.fr(-min(env['V_min']), 1)}<" in svg, re.findall(r">([^<>]*\d,\d[^<>]*)<", svg)[:12])


def boites(dessin):
    """Les boîtes ESTIMÉES des textes d'un dessin : [(texte, x0, x1, y0, y1)] — 0,55 du corps par caractère (prudent : plus
    étroit que le vrai texte, pour ne pas crier à tort), de 0,8 du corps au-dessus de la ligne de base à 0,1 dessous. Le
    harnais du navigateur, lui, mesure les vraies boîtes (« barres », verifier_site.js)."""
    out = []
    for m in re.finditer(r'<text x="([-\d.]+)" y="([-\d.]+)" text-anchor="(\w+)" font-size="([\d.]+)"[^>]*>(.*?)</text>', dessin, re.S):
        x, y, ancre, fs = float(m.group(1)), float(m.group(2)), m.group(3), float(m.group(4))
        t = html.unescape(re.sub(r"<[^>]+>", "", m.group(5))).replace("\u200b", "")
        large = 0.55 * fs * len(t)
        x0 = x - large / 2 if ancre == "middle" else (x - large if ancre == "end" else x)
        out.append((t, x0, x0 + large, y - 0.8 * fs, y + 0.1 * fs))
    return out


def recouvrements(dessin):
    """Les textes d'un dessin qui se recouvrent (plus de 2 px dans les deux sens) : [(texte, texte)]."""
    b = boites(dessin)
    return [(u[0], v[0]) for i, u in enumerate(b) for v in b[i + 1:] if min(u[2], v[2]) - max(u[1], v[1]) > 2 and min(u[4], v[4]) - max(u[3], v[3]) > 2]


# (trouvé à l'œil sur une capture du harnais, 14/11/2026 : sur une poutre continue, le minimum de V tombe au milieu de la
#  barre — sa cote « −81,4 » passait sur la légende des maintiens, écrite juste sous le diagramme)
DESSINS = {titre: rep["vues"][0]["svg"] for titre, rep in REPONSES.items() if rep["vues"][0]["cle"] == "barre"}
DESSINS["trois travées, trois cas, une ponctuelle"] = enchainer(e0, ("modele", "continue"), ("travees/+", 1), ("cas/1/charges/+", "P"), ("cas/+", 1))["vues"][0]["svg"]
DESSINS["une console"] = enchainer(e0, ("modele", "console"))["vues"][0]["svg"]
DESSINS["bi-encastrée, deux ponctuelles et un couple"] = enchainer(e0, ("modele", "bi_enc"), ("cas/0/charges/+", "P"), ("cas/1/charges/+", "P"), ("cas/1/charges/1/a", "1"),
                                                                   ("cas/1/charges/+", "P"), ("cas/1/charges/2/type", "C"))["vues"][0]["svg"]
dessus = {titre: recouvrements(d) for titre, d in DESSINS.items()}
chk(f"dans le dessin d'une barre, aucun texte n'en recouvre un autre — {len(DESSINS)} barres : la neuve, une poutre continue à trois cas, trois travées avec "
    "une ponctuelle (la cote du minimum de V ne passe pas sur la légende des maintiens), une console, une bi-encastrée à deux ponctuelles et un couple",
    len(DESSINS) >= 6 and all(len(boites(d)) >= 10 for d in DESSINS.values()) and not any(dessus.values()), {k: v[:3] for k, v in dessus.items() if v})
bande = texte(REPONSES["la note neuve"]["vues"][0]["entete"])
chk("l'en-tête d'une vue : le profilé et l'acier, la barre en une ligne, ses maintiens, et les plus grands efforts de calcul",
    "BARRE ET EFFORTS · IPE 300 · S235" in bande and "Deux appuis · 6,00 m · 2 cas de charge, 2 charges" in bande and "Maintiens : complet à 3,00 m" in bande
    and "66,6 kN·m" in bande and "44,4 kN" in bande, bande)
tx = REPONSES["une barre nue que la méthode historique refuse (IPE 300, 6 m sans maintien)"]["vues"][2]["svg"]
chk("la vue « Taux » : une barre par vérification, la méthode en titre, le taux écrit ; un taux qui dépasse est rouge",
    "Méthode historique" in tx and "Eurocode 3" in tx and ">179 %<" in tx and ">92 %<" in tx and "#DC2626" in tx, re.findall(r">(\d+ %)<", tx))
mechant = '<script>alert(1)</script>"\'&'
sale = req(action="calculer", etat=note(barre(nom=mechant[:40], cas=[{"nom": "<b>G</b>", "nature": "permanent", "N": 0.0,
                                                                         "charges": [{"type": "q", "v": 5.0, "a": 0.0, "b": 6.0}]}])))
# (l'état et le résumé de la note sont des DONNÉES, que l'écran pose en texte ; ce qui est du HTML : les fragments et les dessins)
tout = sale["resultat"] + sale["volets"] + sale["donnees"] + sale["statut_html"] + "".join(v["svg"] + v["entete"] for v in sale["vues"])
chk("un nom de barre ou de cas saisi est ÉCHAPPÉ partout (résultat, fenêtres, colonne de gauche, dessins) : aucune balise n'en sort",
    sale["ok"] and "<script>" not in tout and "<b>G</b>" not in tout and "&lt;script&gt;" in sale["resultat"] and "&lt;b&gt;G&lt;/b&gt;" in sale["volets"]
    and "&lt;b&gt;G&lt;/b&gt;" in sale["vues"][0]["svg"])
partout = "".join(json.dumps(rep, ensure_ascii=False) for rep in REPONSES.values()) + sel["html"]
chk("aucune ressource externe, aucun script, aucun gestionnaire d'événement dans ce que le module rend",
    not re.search(r"https?://(?!www\.w3\.org/2000/svg)", partout) and "<script" not in partout and not re.search(r"\son\w+=", partout))
chk("la réponse du routeur est du JSON strict pour chaque cas (aucun nombre non fini)", all(rep["ok"] for rep in REPONSES.values())
    and "NaN" not in partout and "Infinity" not in partout)

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

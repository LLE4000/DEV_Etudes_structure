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
 10. UN SEUL ARRONDI — le tableau, la formule et le catalogue écrivent le même chiffre d'un même nombre.
 11. LOT A — une méthode hors de son domaine (« Hors domaine », l'ocre « v »), le poids propre que le moteur nomme, la
     source de f_y ; chaque contrôle a sa mutation.
 15-16. LOT B — le format de la note (version 2 et sa clé « zero_saisi » : un 0 saisi n'est pas un champ vide ; relue par
     le site d'avant, 766659c, tirée de l'historique) ; s_s et c se vident.

    /usr/bin/python3.12 tests/test_api_web_profil.py
"""
import copy
import html
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import xml.dom.minidom

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from acier.barre import ecriture as W  # noqa: E402
from acier.barre import entree as EN  # noqa: E402
from acier.barre import moteur as MO  # noqa: E402
from acier.profiles import catalogue as CAT  # noqa: E402
from api_web import commun as C  # noqa: E402
from api_web import dessin_barre as DB  # noqa: E402
from api_web import profil as P  # noqa: E402
from api_web import profil_mots as M  # noqa: E402
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
    return P.etat_de(list(barres) or [P.barre_neuve()])


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
chk("une note neuve : version 2 et sa clé « zero_saisi » (lot B, sans changer de version — le harnais suit : version 3 à la 3e reprise), une "
    "barre B1 — IPE 300, S235, deux appuis de 6 m, G et Q de 5 kN/m, poids propre, tenue à mi-portée",
    neuf["ok"] and e0["version"] == P.VERSION == 2 and e0.get(P.ZERO_SAISI) is True and len(e0["elements"]) == 1 and e0["actif"] == 0 and el0["nom"] == "B1" and el0["profil"] == "IPE 300"
    and el0["nuance"] == "S235" and el0["mode"] == "chargee" and el0["modele"] == "appuis" and el0["travees"] == [6.0]
    and [c["nom"] for c in el0["cas"]] == ["G", "Q"] and el0["poids_propre"] is True and el0["maintiens"] == [{"x": 3.0, "genre": "complet"}]
    and el0["methodes"] == ["historique", "ec3"], el0)
th, te = r0["methodes"]["historique"]["taux_max"], r0["methodes"]["ec3"]["taux_max"]
chk("ses DEUX méthodes tiennent (ELS 74 %, Eurocode 3 58 % — lot A : le couple (λ̄_LT,0 ; β) = (0,2 ; 1,0) de l'ANB:2018, 6.3.2.3(1)) ; "
    "mais s_s et c ne sont pas saisis, et à leur borne 0 l'âme ne passe pas (critère 6.2.1(5), 157 %) : décision impossible (brief § 8.3, "
    "point 5) — la note est « À vérifier », son résumé porte le taux le plus fort ; seule s_s est demandée (le critère ne lit pas c — le harnais "
    "suit : « s_s et c » à la 2e reprise)",
    r0["statut"] == "non_couvert" and r0["ame"]["statut"] == "non_couvert" and round(th, 2) == 0.74 and round(te, 2) == 0.58
    and neuf["resume"]["statut"] == "À vérifier" and neuf["resume"]["taux"] == C.pct(max(th, te)) and neuf["resume"]["niveau"] == "v"
    and neuf["resume"]["gouvernant"] == "B1 · IPE 300 · flexion (ELS)"
    and "Âme · Décision impossible : s_s à saisir (appui A, appui B)" in texte(neuf["resultat"]), (th, te, neuf["resume"]))
# la base décidée : la note neuve, s_s = 100 mm et c = 0 saisis
DECIDE = enchainer(e0, ("appuis/ss", "100"), ("appuis/c", "0"))
chk("… s_s = 100 mm et c = 0 saisis : l'âme décide (pas de raidisseur, 22 %) et la note redevient « Conforme »",
    DECIDE["resume"]["statut"] == "Conforme" and DECIDE["resume"]["niveau"] == "p" and moteur(DECIDE["etat"]["elements"][0])["statut"] == "ok"
    and "Âme ·" not in texte(DECIDE["resultat"]), DECIDE["resume"])
res0 = texte(neuf["resultat"])
chk("un VERDICT par méthode : le taux du moteur, la vérification qui gouverne, ce qu'elle compare, sa combinaison et son lieu",
    neuf["resultat"].count('class="verdict"') == 2 and f"η {pastille(th)} Méthode ELS (NBN B 51-001) · flexion σ 115,7 ≤ σ adm 156,7 N/mm²" in res0
    and "Poids propre + G + Q · x = 3,00 m" in res0 and f"η {pastille(te)} Eurocode 3 · déversement M y,Ed 66,7 ≤ M b,Rd 115,6 kN·m" in res0
    and "1,35·Poids propre + 1,35·G + 1,5·Q · de 0,00 à 3,00 m" in res0, res0[:500])
chk("le poids propre de la saisie est celui que le MOTEUR nomme (78,6 kN/m³ × A = 0,42 kN/m) — l'écran ne le recalcule pas",
    round(r0["poids_propre"], 4) == 0.4230 and "(0,42 kN/m)" in texte(neuf["donnees"]), texte(neuf["donnees"])[:300])
chk("la tête dit la barre : « B1 », « IPE 300 · S235 · 6,00 m » ; le pied, la masse et la flèche de service (une information)",
    '<div class="et-rp"><b>B1</b><span>IPE 300 · S235 · 6,00 m</span></div>' in neuf["resultat"] and "Masse du profilé 42,2 kg/m" in res0
    and "Flèche de service 10,0 mm · L/599" in res0, res0[-300:])
chk("la colonne de gauche : la ligne de la barre et ses cinq rubriques — Barre, Charges, Maintiens et flambement, Appuis, Méthodes",
    [m for m in re.findall(r'<span class="tt">([^<]+)</span>', neuf["donnees"])] == ["Barre", "Charges", "Maintiens et flambement", "Appuis", "Méthodes"]
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
           profil="HEA 200", modele="continue", travees=[5.0, 4.0], maintiens=[{"x": 2.5, "genre": "haut"}, {"x": 7.0, "genre": "haut"}],
           cas=[{"nom": "G", "nature": "permanent", "N": 0.0, "charges": [{"type": "q", "v": 6.0, "a": 0.0, "b": 9.0}]},
                {"nom": "Q", "nature": "exploitation_B", "N": 0.0, "charges": [{"type": "q", "v": 8.0, "a": 0.0, "b": 5.0},
                                                                               {"type": "P", "v": 12.0, "a": 7.0, "ss": 100.0}]},
                {"nom": "S", "nature": "neige", "N": 0.0, "charges": [{"type": "tri", "v": 3.0, "a": 5.0, "b": 9.0, "sens": "croissante"}]}],
           appuis={"mode": "poses", "ss": 100.0, "c": 0.0})),
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
cmp_ = re.search(r'<section class="et-bloc inc" data-theme="4">(.*?)</section>', rep["resultat"], re.S)
chk("la COMPARAISON (le quatrième bloc, après les deux méthodes et l'âme) : un sujet par ligne, le taux de chaque méthode — ceux que le moteur "
    "rend ; l'âme, qui vaut pour les deux, y porte UN taux ; sa fenêtre (cmp)",
    cmp_ is not None and all(li["sujet"] in texte(cmp_.group(1)) for li in r["comparaison"])
    and all(pastille(li[k]) in texte(cmp_.group(1)) for li in r["comparaison"] for k in ("historique", "ec3") if li[k] is not None)
    and r["comparaison"][-1]["sujet"] == "Âme et raidisseurs (EN 1993-1-5)" and r["comparaison"][-1]["commun"] == r["ame"]["taux_max"]
    and f'Âme et raidisseurs (EN 1993-1-5) {pastille(r["ame"]["taux_max"])}' in texte(cmp_.group(1)) and cmp_.group(1).count("pf-cmp-c") == 1
    and 'data-calcul="cmp"' in rep["resultat"] and 'id="ck-cmp"' in rep["volets"] and "vaut pour les deux méthodes" in texte(rep["volets"]),
    texte(cmp_.group(1)) if cmp_ else None)
chk("(lot A) la poutre continue n'a plus de paramètre « à confirmer » : le couple (λ̄_LT,0 ; β) est celui de l'ANB:2018, 6.3.2.3(1) — ni lien "
    "ni fenêtre (ac)", not r["methodes"]["ec3"]["a_confirmer"] and "à confirmer" not in texte(rep["donnees"]) and 'id="ck-ac"' not in rep["volets"])
rpp = REPONSES["un poteau-poutre en efforts imposés"]
ac = moteur(rpp["etat"]["elements"][0])["methodes"]["ec3"]["a_confirmer"]
chk(f"les paramètres « à confirmer » de l'annexe nationale (le poteau-poutre : {len(ac)}, ceux de l'interaction) : un lien dans la rubrique "
    "Méthodes, une fenêtre (ac) avec leur valeur et leur source",
    len(ac) >= 2 and f"{len(ac)} paramètres à confirmer" in texte(rpp["donnees"]) and 'data-calcul="ac"' in rpp["donnees"]
    and 'id="ck-ac"' in rpp["volets"] and all(p["source"] in texte(rpp["volets"]) for p in ac))
nu = REPONSES["une barre nue que la méthode historique refuse (IPE 300, 6 m sans maintien)"]
chk("une barre que la méthode ELS refuse (179 %) et que l'Eurocode refuse aussi (114 % — lot A : avec le couple (0,2 ; 1,0) de "
    "l'ANB:2018, elle passait à 92 %) : la note est NON CONFORME, les deux verdicts se lisent, deux blocs rouges",
    nu["resume"]["niveau"] == "r" and nu["resume"]["statut"] == "Non conforme" and nu["resume"]["taux"] == 179
    and "η 179 % Méthode ELS (NBN B 51-001)" in texte(nu["resultat"]) and "η 114 % Eurocode 3" in texte(nu["resultat"])
    and nu["resultat"].count('class="et-bloc nok"') == 2, nu["resume"])

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
         ("nom", "N" * 60, lambda e: e["nom"] == "N" * 60), ("cas/1/nom", "Exploitation", lambda e: e["cas"][1]["nom"] == "Exploitation"),
         # (lot 4 : ce que l'âme reçoit de ses appuis)
         ("appuis/ss", "100", lambda e: e["appuis"] == {"mode": "poses", "ss": 100.0, "c": None}),
         ("appuis/c", "50", lambda e: e["appuis"] == {"mode": "poses", "ss": None, "c": 50.0}),
         ("appuis/mode", "ame", lambda e: e["appuis"]["mode"] == "ame")]
faux = []
for chemin, valeur, juste in POSES:
    out = req(action="modifier", etat=copy.deepcopy(e0), chemin=chemin.split("/"), valeur=valeur)
    if not out["ok"] or out["message"] or not juste(out["etat"]["elements"][0]):
        faux.append((chemin, valeur, out.get("message") or out.get("erreur")))
chk(f"chaque saisie va au bon endroit ({len(POSES)} champs : nom, portée, poids propre, acier, niveau, cas, charges, maintiens, flambement, méthodes, "
    "options de la méthode historique, γ_M1, appuis — et un nom de 60 caractères, la borne du site)", not faux, faux[:4])
# les listes
out = saisir(e0, "cas/+", "1")
c3 = out["etat"]["elements"][0]["cas"]
chk("« + Cas de charge » : un cas variable de plus (Q2, exploitation A, une charge répartie sur toute la barre) ; « retirer » l'ôte",
    [c["nom"] for c in c3] == ["G", "Q", "Q2"] and c3[2]["nature"] == "exploitation_A" and c3[2]["charges"] == [{"type": "q", "v": 5.0, "a": 0.0, "b": 6.0}]
    and [c["nom"] for c in saisir(out["etat"], "cas/1/-", "1")["etat"]["elements"][0]["cas"]] == ["G", "Q2"], c3)
out = enchainer(e0, ("cas/0/charges/+", "P"), ("cas/0/charges/1/v", "12"), ("cas/0/charges/1/a", "2"))
ch = out["etat"]["elements"][0]["cas"][0]["charges"]
chk("« + Ponctuelle » : une charge de 10 kN à mi-portée, que l'on règle (12 kN à 2 m) ; son type se change (un couple au même endroit, puis une "
    "charge triangulaire sur toute la barre) ; sa longueur d'appui n'est pas saisie (None, jamais 0 — le harnais suit)",
    ch[1] == {"type": "P", "v": 12.0, "a": 2.0, "ss": None}
    and saisir(out["etat"], "cas/0/charges/1/type", "C")["etat"]["elements"][0]["cas"][0]["charges"][1] == {"type": "C", "v": 12.0, "a": 2.0}
    and saisir(out["etat"], "cas/0/charges/1/type", "tri")["etat"]["elements"][0]["cas"][0]["charges"][1]
    == {"type": "tri", "v": 12.0, "a": 0.0, "b": 6.0, "sens": "croissante"}
    and saisir(out["etat"], "cas/0/charges/0/-", "1")["etat"]["elements"][0]["cas"][0]["charges"] == [{"type": "P", "v": 12.0, "a": 2.0, "ss": None}], ch)
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
         ("nom", "N" * 61, "au plus 60"), ("cas/1/nom", "Exploitation2", "au plus 12"),
         ("appuis/ss", "-3", "de 0 à 2000 mm"), ("appuis/ss", "2500", "de 0 à 2000 mm"), ("appuis/c", "abc", "n'est pas un nombre"),
         ("appuis/mode", "soude", "inconnu"), ("cas/0/charges/0/ss", "50", "Champ inconnu")]
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
    "caractères, un nom de cas de 13, une longueur d'appui négative ou de 2,5 m, un mode d'appui inconnu, la longueur d'appui d'une charge "
    "répartie…)", not faux, faux[:4])
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
for choix, cle, autre in (("historique", "historique", "Eurocode 3"), ("ec3", "ec3", "Méthode ELS")):
    out = saisir(e0, "methodes", choix)
    r = moteur(out["etat"]["elements"][0])
    chk(f"« {choix} » seul : un verdict, le bloc de la méthode et celui de l'âme (EN 1993-1-5 : il vaut pour toute méthode), ni comparaison ni "
        f"{autre} ; le résumé est le sien", list(r["methodes"]) == [cle]
        and out["resultat"].count('class="verdict"') == 1 and out["resultat"].count('class="et-bloc ') == 2 and autre not in texte(out["resultat"])
        and re.findall(r'<span class="n">([^<]+)</span>', out["resultat"]) == [PR.M.METHODES_COURT[cle], "Âme et raidisseurs"]
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
    anc["ok"] and anc["etat"]["version"] == P.VERSION and ea["mode"] == "efforts" and ea["methodes"] == ["ec3"] and ea["nuance"] == "S355" and ea["profil"] == "HEB 200"
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
    and "barre 2" in req(action="calculer", etat={"version": P.VERSION, "actif": 0, "elements": [P.barre_neuve(), "n'importe quoi"]})["erreur"])

print("=== 6 bis. L'âme et ses raidisseurs (EN 1993-1-5)")
ra = r0["ame"]
bloc = re.search(r'<section class="et-bloc inc" data-theme="3">(.*?)</section>', neuf["resultat"], re.S)
tb = texte(bloc.group(1)) if bloc else ""
chk("la barre neuve (s_s et c non saisis) : le bloc « Âme et raidisseurs » (le troisième) dit « À vérifier », la norme et les combinaisons ELU ; "
    "une ligne par appui — « décision impossible : s_s à saisir », avec le taux à la borne (157 %, en gris) ; l'avertissement en pied (le "
    "harnais suit : sans taux, « à fournir », à la 2e reprise du lot B — brief § 8.3, point 5)",
    bloc is not None and tb.startswith("Âme et raidisseurs À vérifier") and "NBN EN 1993-1-5 · combinaisons ELU (6.10) · 3 combinaisons" in tb
    and all(f"{L['nom']} {pastille(L['taux'])} décision impossible : s s à saisir" in tb and round(L["taux"], 3) == 1.567 for L in ra["lieux"])
    and len(ra["lieux"]) == 2 and "Appui A, appui B : décision impossible : s s à saisir." in tb and ra["taux_max"] is None, tb[:400])
chk("… sous les verdicts, la ligne « Âme » dit ce qui manque, et ouvre tout le calcul du bloc",
    'class="pf-vame inc" data-calcul="s3"' in neuf["resultat"] and "Âme · Décision impossible : s_s à saisir (appui A, appui B)" in texte(neuf["resultat"]))
rd = moteur(DECIDE["etat"]["elements"][0])["ame"]
bd = re.search(r'<section class="et-bloc ok" data-theme="3">(.*?)</section>', DECIDE["resultat"], re.S)
tbd = texte(bd.group(1)) if bd else ""
chk("… s_s = 100 mm et c = 0 saisis : « Sans raidisseur · 22 % » ; une ligne par appui — ce que l'âme compare (F_Ed 44,5 ≤ F_Rd 199,8 kN, R4 "
    "type (c), contact non établi), « sans raidisseur », le taux du moteur ; rien à poser : pas de ligne « Âme » sous les verdicts",
    bd is not None and tbd.startswith("Âme et raidisseurs Sans raidisseur · 22 %") and len(rd["lieux"]) == 2
    and all(f"{L['nom']} {pastille(L['taux'])} F Ed 44,5 ≤ F Rd 199,8 kN · sans raidisseur" in tbd for L in rd["lieux"])
    and round(rd["taux_max"], 2) == 0.22 and "pf-vame" not in DECIDE["resultat"], tbd[:300])
manques = []
for L in ra["lieux"]:
    f = PR.fenetre("ame", L["base"])
    fen = re.search(rf'<div class="flyout calcul" id="ck-{re.escape(f)}">(.*?)</div></div></div>', neuf["volets"], re.S)
    if f'data-calcul="{f}"' not in bloc.group(1) or not fen:
        manques.append((L["nom"], "ligne ou fenêtre absente"))
        continue
    t = texte(fen.group(1))
    par = {x["cle"]: x for x in ra["calcul"]}
    for vid in L["verifications"]:
        v = next(x for x in ra["verifications"] if x["id"] == vid)
        for cle in v["formules"]:
            if par[cle]["gabarit"] == "taux":
                continue
            parts, res = W.morceaux(par[cle])
            if res not in t or parts[-1] not in t:
                manques.append((L["nom"], cle, res))
    if "décision impossible : s s à saisir" not in t or any(texte(PR.M.sub(h)) not in t for h in ra["hypotheses"]):
        manques.append((L["nom"], "en-tête ou hypothèses"))
chk("chaque LIEU ouvre SA fenêtre de calcul (a-A-L00, a-A-L01) : toutes ses formules (à la borne s_s = c = 0) écrites par "
    "acier.barre.ecriture, les hypothèses du bloc, « décision impossible » ; « tout le calcul » du bloc (s3) porte ses vérifications et ses "
    "combinaisons",
    not manques and 'id="ck-s3"' in neuf["volets"] and 'data-calcul="s3"' in bloc.group(1), manques[:3])
v_el = next(v for v in ra["verifications"] if v["id"].endswith("-EL"))
chk("… une vérification donnée pour information au-delà de 1 (non couverte : le critère 6.2.1(5) à s_s = 0, 1,567) se dit « > », jamais "
    "« ≤ » (profil_mots.comparaison)",
    v_el["statut"] == "non_couvert" and round(v_el["taux"], 3) == 1.567 and texte(M.comparaison(v_el)) == "η = 1,567 > 1", texte(M.comparaison(v_el)))
lourde = barre(profil="HEB 300", travees=[4.0], maintiens=[{"x": 2.0, "genre": "complet"}], appuis={"mode": "poses", "ss": 200.0, "c": 0.0},
               cas=[{"nom": "G", "nature": "permanent", "N": 0.0, "charges": [{"type": "P", "v": 500.0, "a": 2.0, "ss": 0.0}]}])
rep = req(action="calculer", etat=note(lourde))
rl = moteur(rep["etat"]["elements"][0])["ame"]
tr_ = texte(rep["resultat"])
Lc = next(L for L in rl["lieux"] if L["genre"] == "charge")
chk("HEB 300, 675 kN à mi-portée : l'âme seule ne suffit pas — sous les verdicts, « Âme · Raidisseur : 2 plats 140 × 15 mm (charge à 2,00 m) » ; "
    "le bloc dit « 1 raidisseur · 91 % » (aux appuis d'extrémité, σ_x = 0 : le critère 6.2.1(5) à la jonction, 0,910, gouverne le bloc — le "
    "harnais suit : 87 % avant le lot B, brief § 8.2) et sa ligne nomme les plats avant ce qu'ils comparent",
    rep["ok"] and Lc["decision"] == "raidisseur" and 'class="pf-vame warn" data-calcul="s3"' in rep["resultat"]
    and "Âme · Raidisseur : 2 plats 140 × 15 mm (charge à 2,00 m)" in tr_ and "Âme et raidisseurs 1 raidisseur · 91 %" in tr_
    and "Appui A 91 % η = 0,910 ≤ 1 · sans raidisseur" in tr_
    and "Charge à 2,00 m 87 % 2 plats 140 × 15 mm · F Ed 675,0 ≤ N c,Rd 775,5 kN" in tr_ and 'class="pf-raid"' in rep["resultat"], tr_[:400])
fen = re.search(rf'<div class="flyout calcul" id="ck-{re.escape(PR.fenetre("ame", Lc["base"]))}">(.*?)</div></div></div>', rep["volets"], re.S)
tf = texte(fen.group(1)) if fen else ""
chk("… la fenêtre du lieu : la raison (« Sans raidisseur : F_Ed = 675,0 kN > F_Rd = 585,2 kN » — lot A : f_y = 225 N/mm², t_f = 19 mm), la "
    "résistance de l'âme seule, puis l'écrasement "
    "et le flambement du raidisseur",
    "Sans raidisseur : F Ed = 675,0 kN > F Rd = 585,2 kN" in tf and "raidisseur, écrasement au contact" in tf and "raidisseur, flambement" in tf
    and "585,2 kN" in tf and "775,5 kN" in tf and "2 plats 140 × 15 mm" in tf, tf[:300])
vb = next(x for x in rep["vues"] if x["cle"] == "barre")["svg"]
vt = next(x for x in rep["vues"] if x["cle"] == "taux")["svg"]
chk("… le dessin de la barre porte le raidisseur (un plat debout à 2,00 m) et sa légende ; la vue des taux, une barre par lieu avec ses plats",
    vb.count(f'fill="{DB.RAIDISSEUR}"') == 2 and "raidisseur transversal : une paire de plats" in html.unescape(vb)
    and "Charge à 2,00 m · 2 plats 140 × 15" in html.unescape(vt) and "Âme et raidisseurs" in vt and "Appui A" in vt
    and next(x for x in neuf["vues"] if x["cle"] == "barre")["svg"].count(f'fill="{DB.RAIDISSEUR}"') == 0, vb.count(DB.RAIDISSEUR))
ss100 = saisir(e0, "appuis/ss", "100")
c0 = DECIDE
def fenetre_appui(rep_):
    """Le texte de la fenêtre du premier appui."""
    L_ = moteur(rep_["etat"]["elements"][0])["ame"]["lieux"][0]
    f_ = re.search(rf'<div class="flyout calcul" id="ck-{re.escape(PR.fenetre("ame", L_["base"]))}">(.*?)</div></div></div>', rep_["volets"], re.S)
    return texte(f_.group(1)) if f_ else ""


c50 = saisir(c0["etat"], "appuis/c", "50")
chk("la longueur d'appui saisie seule (100 mm) : à la borne c = 0, « sans raidisseur (c = 0 : borne prudente) », 22 % (le harnais suit : "
    "« décision impossible : c à fournir » à la 2e reprise) ; avec c = 0 saisi : F_Rd = 199,8 kN (le cas R4 du brief, type (c) : le contact "
    "d'extrémité n'est pas établi, s_s = 0 dans (6.10) — lot A) — 22 %, sans borne ; avec contact, 248,9 kN pour information ; 50 mm de débord : "
    "le verdict reste 199,8 kN, l'information passe à 335,9 kN",
    "F Ed 44,5 ≤ F Rd 199,8 kN · sans raidisseur (c = 0 : borne prudente)" in texte(ss100["resultat"]) and "Sans raidisseur · 22 %" in texte(ss100["resultat"])
    and "borne prudente" not in texte(c0["resultat"])
    and "F Ed 44,5 ≤ F Rd 199,8 kN" in texte(c0["resultat"]) and "Sans raidisseur · 22 %" in texte(c0["resultat"])
    and "s s = 100 mm, débord en rive c = 0 mm" in fenetre_appui(c0)
    and "hypothèse de contact non établie : s s = 0 dans (6.10) (verdict) ; avec contact, F Rd = 248,9 kN, pour information" in fenetre_appui(c0)
    and "F Ed 44,5 ≤ F Rd 199,8 kN" in texte(c50["resultat"]) and "débord en rive c = 50 mm" in fenetre_appui(c50)
    and "avec contact, F Rd = 335,9 kN, pour information" in fenetre_appui(c50), texte(ss100["resultat"])[-700:-400])
par_ame = saisir(e0, "appuis/mode", "ame")
chk("appuis « par l'âme » : plus rien n'appuie sur l'âme — le bloc disparaît, la comparaison redevient le troisième bloc, sans ligne de l'âme ; "
    "la rubrique n'offre plus ni longueur d'appui ni débord",
    "Âme et raidisseurs" not in texte(par_ame["resultat"]) and 'data-theme="3"' in par_ame["resultat"] and 'data-theme="4"' not in par_ame["resultat"]
    and 'data-chemin="appuis/ss"' not in par_ame["donnees"] and 'data-chemin="appuis/ss"' in neuf["donnees"]
    and 'data-chemin="appuis/c"' in neuf["donnees"] and 'id="ck-s3"' not in par_ame["volets"])
rub = lambda rep_: re.findall(r'<span class="tt">([^<]+)</span>', rep_["donnees"])          # noqa: E731
chk("la rubrique « Appuis » n'existe que pour une barre chargée qui a un appui simple : absente d'une console, d'une barre bi-encastrée et des "
    "efforts imposés ; présente sur une barre encastrée – appuyée et une poutre continue",
    all("Appuis" not in rub(saisir(e0, "modele", m_)) for m_ in ("console", "bi_enc")) and "Appuis" not in rub(saisir(e0, "mode", "efforts"))
    and all("Appuis" in rub(saisir(e0, "modele", m_)) for m_ in ("enc_app", "continue")))
pon = saisir(e0, "cas/0/charges/+", "P")
p120 = saisir(pon["etat"], "cas/0/charges/1/ss", "120")
chk("une charge ponctuelle dit sa longueur d'appui (« sur … mm ») quand elle appuie sur une semelle ; appliquée au centre de cisaillement, le "
    "champ s'efface (elle entre par l'âme) et le lieu disparaît du bloc",
    'data-chemin="cas/0/charges/1/ss"' in pon["donnees"] and p120["etat"]["elements"][0]["cas"][0]["charges"][1]["ss"] == 120.0
    and "Charge à 3,00 m" in texte(p120["resultat"])
    and 'data-chemin="cas/0/charges/1/ss"' not in saisir(p120["etat"], "niveau", "centre")["donnees"]
    and "Charge à 3,00 m" not in texte(saisir(p120["etat"], "niveau", "centre")["resultat"])
    and saisir(p120["etat"], "niveau", "centre")["etat"]["elements"][0]["cas"][0]["charges"][1]["ss"] == 120.0)
# (lot A : η = 1,00 pour un laminé — ANB:2011, 5.1(2) — l'IPE 750x147 ne voile plus (h_w/t_w = 54,5 < 72ε/η = 58,6) ; l'IPE 750x134
# (59,9) voile : V_bw,Rd = 1722,0 kN < V_pl,z,Rd = 1965,4 kN)
voile = req(action="calculer", etat=note(barre(mode="efforts", profil="IPE 750x134", nuance="S355", methodes=["ec3"],
                                                 flambement={"y": {"L": 0.0, "u": 1.0}, "z": {"L": 0.0, "u": 1.0}}, Lc=0.0,
                                                 efforts={"service": {"N": 0.0, "My": 0.0, "Mz": 0.0, "Vz": 0.0, "Vy": 0.0},
                                                          "elu": {"N": 0.0, "My": 0.0, "Mz": 0.0, "Vz": 1850.0, "Vy": 0.0}})))
tv = texte(voile["resultat"])
chk("une âme qui VOILE alors que l'Eurocode 3 passe (IPE 750x134 S355, V_z = 1 850 kN) : « Âme · non conforme — voilement de l'âme par "
    "cisaillement », le bloc en rouge (107 %), la note NON CONFORME",
    voile["ok"] and 'class="pf-vame nok"' in voile["resultat"] and "Âme · non conforme — voilement de l'âme par cisaillement" in tv
    and "Âme et raidisseurs Non conforme · 107 %" in tv and voile["resume"]["niveau"] == "r" and "Eurocode 3 Conforme · 94 %" in tv, tv[:400])
chk("… le taux de la barre est alors celui de l'ÂME, qui ne passe pas (107 %, et non les 94 % de l'Eurocode) : le résumé de la note, la ligne "
    "« Profilé » ; une âme qui PASSE ne compte pas — un raidisseur à 98 % ne change pas le taux d'une barre à 74 %",
    voile["resume"]["taux"] == 107 and voile["resume"]["gouvernant"] == "B1 · IPE 750x134 · voilement de l'âme par cisaillement (âme)"
    and "Profilé IPE 750x134 107 %" in tv and PR.pire(moteur(voile["etat"]["elements"][0]))[0] == "ame"
    and PR.taux_barre(moteur(rep["etat"]["elements"][0])) == max(x["taux_max"] for x in moteur(rep["etat"]["elements"][0])["methodes"].values())
    and neuf["resume"]["taux"] == 74 and PR.pire(r0)[0] == "historique", (voile["resume"], tv[:300]))

print("=== 7. Le sélecteur")
# (la note neuve, s_s et c saisis — sans eux, l'âme ne décide pour aucun profilé : brief § 8.3, point 5)
eD = DECIDE["etat"]
elD = eD["elements"][0]
sel = req(action="selecteur", etat=eD, role="profil")
rows = re.findall(r'<button class="pk-row([^"]*)"[^>]*data-choix="([^"]+)"[^>]*>(.*?)</button>', sel["html"], re.S)
noms = [n for _, n, _ in rows]
ipe = [p["nom"] for p in sorted((p for p in CAT.profils() if p["famille"] == "IPE"), key=lambda p: (p["masse"], p["nom"]))]
chk("par défaut : la famille du profilé de la barre (les 23 IPE), triée par masse", sel["ok"] and noms == ipe and len(noms) == 23, noms[:5])
ecarts = []
for cls, n, h in rows:
    r = moteur(dict(elD, profil=n))
    t = max(x["taux_max"] for x in r["methodes"].values())
    if f"{C.pc(C.pct(min(t, 9.99)))} %" not in h or ("r" in re.search(r'class="chip (\w)"', h).group(1)) != (r["statut"] == "nok"):
        ecarts.append((n, t, texte(h)))
chk("chaque profilé porte le taux le plus fort que le MOTEUR lui donne (les deux méthodes de la barre), rouge quand une méthode le refuse", not ecarts, ecarts[:3])
tiennent = [n for n in ipe if moteur(dict(elD, profil=n))["statut"] == "ok"]
chk("le retenu est coché (IPE 300) ; l'étoile marque le plus LÉGER qui tient — ici le retenu lui-même, donc pas d'étoile ailleurs",
    [n for c, n, _ in rows if " cur" in c] == ["IPE 300"] and [n for c, n, _ in rows if " opt" in c] == [tiennent[0]] and tiennent[0] == "IPE 300", tiennent[:2])
lourd = req(action="choisir", etat=eD, role="profil", nom="IPE 400")
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
# (la solive : s_s et c saisis, l'âme décide — sans eux, elle serait « à vérifier » : brief § 8.3, point 5)
mixte = req(action="calculer", etat=P.etat_de([dict(P.barre_neuve("Solive"), appuis={"mode": "poses", "ss": 100.0, "c": 0.0}),
                                                barre(nom="Panne", maintiens=[])]))
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
    trois = P.etat_de([P.barre_neuve("A"), P.barre_neuve("B"), barre(nom="C", maintiens=[])])
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
chk("le dessin de la barre porte les charges de chaque cas (5 kN/m), la portée, et les extrêmes des enveloppes du moteur (M 66,7 kN·m ; V ± 44,5 kN)",
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
# (demande du propriétaire, 14/11/2026 : « l'effort tranchant il retombe à 0 […] là tu l'arrêtes en l'air ») — chaque
# trait d'enveloppe (M et V, ses deux courbes) PART de l'axe et Y RETOMBE : son premier et son dernier point sont sur
# l'axe du diagramme, à l'abscisse des extrémités de la barre, et le trait monte puis descend à la verticale
def _traits_enveloppe(dessin, couleur):
    """[(points du trait, y de l'axe)] : les courbes d'enveloppe d'une couleur, et l'axe (le trait d'encre qui les suit)."""
    out = []
    for m in re.finditer(r'<path d="(M[^"]+)" fill="none" stroke="' + re.escape(couleur) + r'" stroke-width="1\.5"', dessin):
        pts = [tuple(float(v) for v in c.strip().split()) for c in m.group(1)[1:].split("L")]
        axe = re.search(r'<path d="M([-\d.]+) ([-\d.]+)H([-\d.]+)"', dessin[m.end():])
        out.append((pts, float(axe.group(2)) if axe else None))
    return out


_fermes = {}
for titre, d in DESSINS.items():
    for nom, coul in (("V", DB.DIAG_V), ("M", DB.DIAG_M)):
        traits = _traits_enveloppe(d, coul)
        _fermes[f"{titre} · {nom}"] = len(traits) == 2 and all(
            y0 is not None and abs(p[0][1] - y0) < 0.051 and abs(p[-1][1] - y0) < 0.051 and abs(p[0][0] - p[1][0]) < 0.051
            and abs(p[-1][0] - p[-2][0]) < 0.051 for p, y0 in traits)
_v_neuve = _traits_enveloppe(DESSINS["la note neuve"], DB.DIAG_V)
chk(f"le diagramme de l'effort tranchant (et celui du moment) RETOMBE À ZÉRO aux extrémités — {len(DESSINS)} barres : chaque trait d'enveloppe "
    "part de l'axe, monte à la verticale jusqu'à V(0), suit la courbe, puis redescend à la verticale sur l'axe en L (plus de trait arrêté en l'air)",
    all(_fermes.values()) and len(_fermes) == 2 * len(DESSINS) and all(abs(p[1][1] - p[0][1]) > 10 for p, _ in _v_neuve),
    {k: v for k, v in _fermes.items() if not v})
bande = texte(REPONSES["la note neuve"]["vues"][0]["entete"])
chk("l'en-tête d'une vue : le profilé et l'acier, la barre en une ligne, ses maintiens, et les plus grands efforts de calcul",
    "BARRE ET EFFORTS · IPE 300 · S235" in bande and "Deux appuis · 6,00 m · 2 cas de charge, 2 charges" in bande and "Maintiens : complet à 3,00 m" in bande
    and "66,7 kN·m" in bande and "44,5 kN" in bande, bande)
tx = REPONSES["une barre nue que la méthode historique refuse (IPE 300, 6 m sans maintien)"]["vues"][2]["svg"]
chk("la vue « Taux » : une barre par vérification, la méthode en titre, le taux écrit ; un taux qui dépasse est rouge",
    "Méthode ELS (NBN B 51-001)" in tx and "Eurocode 3" in tx and ">179 %<" in tx and ">114 %<" in tx and "#DC2626" in tx, re.findall(r">(\d+ %)<", tx))
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

print("=== 10. Un seul arrondi : le tableau, la formule et le catalogue écrivent le même chiffre")


def nu(s):
    """Un nombre imprimé sans son habillage : ni espace des milliers, ni zéros de fin, le signe du clavier."""
    s = s.replace("\u202f", "").replace("−", "-")
    return s.rstrip("0").rstrip(",") if "," in s else s


reprise = barre(profil="IPE 450", travees=[5.0], maintiens=[{"x": 0.4, "genre": "complet"}, {"x": 2.7, "genre": "complet"}],
                appuis={"mode": "poses", "ss": 200.0, "c": 0.0},
                cas=[{"nom": "G", "nature": "permanent", "N": 0.0, "charges": [{"type": "q", "v": 4.0, "a": 0.0, "b": 5.0},
                                                                               {"type": "P", "v": 150.0, "a": 0.4, "ss": 0.0}]},
                     {"nom": "Q", "nature": "exploitation_A", "N": 0.0, "charges": [{"type": "q", "v": 3.0, "a": 0.0, "b": 5.0},
                                                                                    {"type": "P", "v": 110.0, "a": 0.4, "ss": 0.0}]}])
rr = req(action="calculer", etat=note(reprise))
ar = moteur(rr["etat"]["elements"][0])["ame"]
Lr = next(L for L in ar["lieux"] if L["genre"] == "charge")
vr = next(x for x in ar["verifications"] if x["id"] == Lr["base"] + "-R")
fen = re.search(rf'<div class="flyout calcul" id="ck-{re.escape(PR.fenetre("ame", Lr["base"]))}">(.*?)</div></div></div>', rr["volets"], re.S)
tf = texte(fen.group(1)) if fen else ""
chk("IPE 450, une charge de 367,5 kN à 0,40 m : son raidisseur (2 plats 90 × 15 mm) porte N_c,Rd = 1950 × 235 / 10³ = 458,25 kN, un demi EXACT — "
    "la fenêtre l'écrit « 458,3 kN » dans le tableau ET au bout de la formule (elle écrivait « 458,3 » au tableau et « 458,2 » dans la formule : "
    "deux arrondis)",
    rr["ok"] and vr["limite"] == 458.25 and Lr["raidisseur"]["designation"] == "2 plats 90 × 15 mm" and "N c,Rd = 458,3 kN" in tf
    and "N c,Rd = A c ·f y,p /γ M0 = 1950 × 235 / 1,0 / 10³ = 458,3 kN" in tf and "458,2 kN" not in tf, tf[900:1100])
resultats = [f["resultat"] for el in (rr["etat"]["elements"][0], req(action="calculer", etat=note(lourde))["etat"]["elements"][0], e0["elements"][0], POTEAU)
             for bloc_ in list(moteur(el)["methodes"].values()) + [moteur(el)["ame"]] for f in bloc_["calcul"]
             if isinstance(f["resultat"], (int, float)) and not isinstance(f["resultat"], bool)]
demis = [k / 8.0 for k in range(-41, 42)] + [100.0 + k / 16.0 for k in range(33)] + [458.25, 3831.25, 13052.5, 10.125, 121.25, 258.5]
ecarts = [(x, nd, C.fr(x, nd), W.nombre(x, nd)) for x in resultats + demis for nd in (0, 1, 2, 3) if nu(C.fr(x, nd)) != nu(W.nombre(x, nd))]
chk(f"le nombre du tableau (commun.fr) et celui de la formule (acier.barre.ecriture.nombre) sont le MÊME chiffre : {len(resultats)} résultats de "
    f"formules de quatre barres et {len(demis)} demis exacts en binaire, à 0, 1, 2 et 3 décimales",
    len(resultats) > 250 and len(demis) > 120 and not ecarts, ecarts[:4])
python = [x for x in demis for nd in (0, 1, 2) if nu(f"{x:.{nd}f}".replace(".", ",")) != nu(C.fr(x, nd))]
chk("… et ce contrôle voit l'écart : sur ces demis, l'arrondi de Python (au chiffre pair) écrit un autre chiffre que l'écran", len(python) > 40, len(python))
lus = [M.n(3831.25, 1), M.n(10.125, 2), M.n(13052.5, 0), M.n(121.25, 1), M.n(-2.5, 0), M.n(1234567.891, 2), M.n(0.0), M.n(-0.0001, 3), M.n(None),
       M.n(210000)]
try:
    lus.append(M.n(float("nan")))
except Exception as ex:  # noqa: BLE001 — un nombre non fini levait : le contrôle le dit au lieu de s'arrêter
    lus.append(f"exception {type(ex).__name__}")
chk("une donnée du catalogue ou une saisie (profil_mots.n) suit l'arrondi du site : 3831,25 → « 3 831,3 » ; 10,125 → « 10,13 » ; 13052,5 → « 13 053 » ; "
    "121,25 → « 121,3 » ; −2,5 → « −3 » ; sans zéro inutile, les milliers séparés, jamais « −0 », « — » sans valeur ou non fini",
    lus == ["3\u202f831,3", "10,13", "13\u202f053", "121,3", "−3", "1\u202f234\u202f567,89", "0", "0", "—", "210\u202f000", "—"], lus)
CHAMPS = {"masse": ("masse", 1), "A": ("A", 2), "I_y": ("Iy", 1), "I_z": ("Iz", 1), "W_el,y": ("Wel_y", 1), "W_pl,y": ("Wpl_y", 1),
          "W_pl,z": ("Wpl_z", 1), "A_v,z": ("Avz", 2), "I_t": ("It", 2), "I_w": ("Iw", 0)}
faux, pairs, ecrits = [], 0, 0
for nom_, rec in CAT.par_nom().items():
    for sym, ecrit, _u in P.fiche_profil(nom_)["proprietes"]:
        cle, nd = CHAMPS[sym]
        ecrits += 1
        pairs += nu(f"{rec[cle]:.{nd}f}".replace(".", ",")) != nu(C.fr(rec[cle], nd))
        if nu(ecrit) != nu(C.fr(rec[cle], nd)):
            faux.append((nom_, sym, rec[cle], ecrit, C.fr(rec[cle], nd)))
hb = dict((s_, v_) for s_, v_, _u in P.fiche_profil("HEB 180")["proprietes"])
chk(f"la fiche d'un profilé écrit les chiffres du catalogue ({ecrits} valeurs de {len(CAT.par_nom())} profilés) : HEB 180, I_y = 3 831,25 cm⁴ → "
    f"« 3 831,3 », comme la page du catalogue — {pairs} valeurs sont des demis exacts que l'arrondi de Python écrivait autrement (« 3 831,2 »)",
    ecrits > 12000 and not faux and pairs >= 25 and hb["I_y"] == "3\u202f831,3" and "I<sub>y</sub> = 3\u202f831,3 cm⁴" in P._proprietes("HEB 180", None),
    (faux[:3], pairs, hb.get("I_y")))
FORMAT_PYTHON = re.compile(r"\{[^{}\n]*:,?\.(?:\d+|\{[^{}\n]*\})[feg]\}|%\.\d+[feg]|\.format\(")
ECRAN = ("profil.py", "profil_mots.py", "profil_resultat.py", "profil_saisie.py", "note_profil.py")
restes = [(f, m_.group(0)) for f in ECRAN for m_ in FORMAT_PYTHON.finditer(open(os.path.join(RACINE, "api_web", f), encoding="utf-8").read())]
chk("aucun fichier de l'écran et de la note des barres n'arrondit un nombre par un format de Python (« {x:.1f} », « %.1f », « .format( ») — les "
    "coordonnées d'un dessin (dessin_barre.py) ne sont pas des nombres écrits", not restes and FORMAT_PYTHON.search('f"{float(x):.{dec}f}"'), restes[:4])

print("=== 11. Lot A : une méthode hors de son domaine, le poids propre du moteur, la source de f_y — et leurs mutations")
from api_web import dessin_profil as DP  # noqa: E402


def lot_a():
    """Les contrôles du lot A sur l'écran, rejoués sous chaque mutation : [(nom, vrai ?, info)]."""
    P._RESULTATS.clear()
    hd = req(action="calculer", etat=note(barre(nuance="S460")))
    t = texte(hd["resultat"]) if hd.get("ok") else ""
    tx = next(x for x in hd["vues"] if x["cle"] == "taux")["svg"] if hd.get("ok") else ""
    nv = req(action="nouveau")
    return [
        ("une barre en S460 se calcule (la méthode ELS est hors de son domaine, NBN B 51-001 : AE235 à AE355) ; sa méthode dit "
         "« Hors domaine · 53 % », jamais « Conforme » ; son bloc est à vérifier (inc) ; la note « À vérifier » (v), au taux du moteur",
         hd.get("ok") and "ELS Hors domaine · 53 %" in t and '<section class="et-bloc inc" data-theme="1">' in hd["resultat"]
         and hd["resume"]["niveau"] == "v" and hd["resume"]["statut"] == "À vérifier" and hd["resume"]["taux"] == 53
         and "hors domaine, calcul pour information" in t, (hd.get("resume"), hd.get("erreur"), t[:200])),
        ("… sa grande pastille est ocre (v) ; dans la vue des taux, ses vérifications sont ocres, celles de l'Eurocode 3 restent vertes",
         'class="chip xl v"' in hd.get("resultat", "") and 'fill="#B86E00"' in tx and 'fill="#86EFAC"' in tx, tx[:120]),
        ("le poids propre de la saisie est celui que le MOTEUR nomme (« (0,42 kN/m) », 78,6 kN/m³ × A)",
         "Poids propre du profilé (0,42 kN/m)" in texte(nv["donnees"]), texte(nv["donnees"])[:300]),
        ("la fenêtre d'une vérification dit d'où vient f_y : la table unique, son tableau de l'ANB et sa bande d'épaisseur",
         "Acier S235 : f y = 235 N/mm² (tableau 3.3 ANB (EN 10025-2:2004, S235JR), t ≤ 16 mm)" in texte(nv["volets"]),
         re.findall(r"Acier S235 : f y[^.]{0,120}", texte(nv["volets"]))[:2]),
    ]


for nom_, ok_, info_ in lot_a():
    chk(nom_, ok_, info_)
css = open(os.path.join(RACINE, "site", "web", "css", "site.css"), encoding="utf-8").read()
vrf = re.search(r":root \{ --vrf: (#[0-9A-F]{6}); --vrf-fond: #[0-9A-F]{6}; --vrf-encre: (#[0-9A-F]{6}); \}", css)
chk("l'ocre « v » des dessins est celui de la pastille de l'écran (copie NOMMÉE de --vrf et --vrf-encre, site/web/css/site.css)",
    vrf is not None and DP.NIVEAU["v"] == (vrf.group(1), vrf.group(2)), (DP.NIVEAU.get("v"), vrf.groups() if vrf else None))


def _sans_poids_propre(vrai):
    def f(el):
        r = dict(vrai(el))
        r.pop("poids_propre", None)
        return r
    return f


MUTATIONS_LOT_A = (
    ("une vérification hors domaine reprend la couleur de son taux (la vue des taux la peint en vert)", M, "niveau",
     lambda vrai: (lambda v: ("r" if v["statut"] == "nok" else C.niveau(v["taux"])) if v.get("taux") is not None else vrai(v))),
    ("une méthode hors domaine est dite d'un mot de taux (« À vérifier »)", PR, "mot_methode", lambda vrai: (lambda x: PR.mot_niveau(PR.niveau_methode(x)))),
    ("l'écran ne lit plus le poids propre que le moteur nomme", MO, "verifier", _sans_poids_propre),
    ("la fenêtre ne dit plus la source de f_y (l'hypothèse « Acier » du moteur réduite à f_y)", MO.ES, "source_fy",
     lambda vrai: (lambda *a, **k: "")),
)
for nom_, objet_, attr_, fabrique_ in MUTATIONS_LOT_A:
    if not hasattr(objet_, attr_):
        chk(f"mutation « {nom_} » : l'attribut {attr_} existe", False, objet_)
        continue
    vrai_ = getattr(objet_, attr_)
    setattr(objet_, attr_, fabrique_(vrai_))
    try:
        rouges_ = [n_ for n_, ok_, _i in lot_a() if not ok_]
    except Exception as err_:  # noqa: BLE001 — une mutation qui fait planter l'écran est vue
        rouges_ = [f"exception : {err_}"]
    finally:
        setattr(objet_, attr_, vrai_)
        P._RESULTATS.clear()
    chk(f"mutation « {nom_} » : {len(rouges_)} contrôle(s) rougissent", bool(rouges_))
chk("témoin : sans mutation, les contrôles du lot A sont verts", all(ok_ for _n, ok_, _i in lot_a()))

print("=== 12. Le nom de la méthode du bureau : « Méthode ELS (NBN B 51-001) », « ELS » là où la place manque — et ses mutations")
import importlib.util  # noqa: E402

from acier.barre import historique as HI  # noqa: E402
from api_web import profil_saisie as PS  # noqa: E402


def visible(h):
    """Ce que l'œil lit d'un fragment ET ce que le survol montre : le texte et les attributs, sans les data-* (les clés
    internes, « data-valeur="historique" », ne changent pas : aucune migration de format)."""
    return html.unescape(re.sub(r'\sdata-[\w-]+="[^"]*"', "", h))


def nom_els():
    """Les contrôles du nom, rejoués sous chaque mutation : [(nom, vrai ?, info)]."""
    P._RESULTATS.clear()
    nv = req(action="nouveau")
    hi_ = saisir(nv["etat"], "methodes", "historique")
    ef_ = enchainer(nv["etat"], ("mode", "efforts"), ("efforts/elu/My", "60"), ("efforts/service/My", "42"))
    hd_ = saisir(nv["etat"], "nuance", "S460")
    tx_ = next(x for x in nv["vues"] if x["cle"] == "taux")["svg"]
    tout = [visible(r_[k]) for r_ in (nv, hi_, ef_, hd_) for k in ("resultat", "donnees", "volets")] + [visible(v["svg"]) for r_ in (nv, hi_, ef_, hd_)
                                                                                                         for v in r_["vues"]]
    restes = [m_.group(0) for t_ in tout for m_ in re.finditer(r".{0,40}[Hh]istorique.{0,30}", t_)]
    return [
        ("le verdict de la méthode du bureau la nomme en entier : « Méthode ELS (NBN B 51-001) · flexion »",
         f"η {pastille(th)} Méthode ELS (NBN B 51-001) · flexion σ 115,7 ≤ σ adm 156,7 N/mm²" in texte(nv["resultat"]), texte(nv["resultat"])[:160]),
        ("… son bloc porte le titre court « ELS » (la place manque devant « Conforme · 74 % ») ; la comparaison, l'en-tête de colonne « ELS »",
         re.findall(r'<span class="n">([^<]+)</span>', nv["resultat"]) == ["ELS", "Eurocode 3", "Âme et raidisseurs", "Comparaison"]
         and '<div class="pf-cmp pf-cmp-h"><span></span><span>ELS</span><span>Eurocode 3</span></div>' in nv["resultat"],
         re.findall(r'<span class="n">([^<]+)</span>', nv["resultat"])),
        ("la rubrique « Méthodes » : le segment « ELS » (sa clé reste « historique »), le titre « Méthode ELS (NBN B 51-001) », le résumé "
         "« ELS + eurocode 3 · s 1,5 »",
         re.search(r'data-chemin="methodes" data-sorte="choix" data-valeur="historique"[^>]*>ELS</button>', nv["donnees"]) is not None
         and '<div class="grp">Méthode ELS (NBN B 51-001)</div>' in nv["donnees"] and "ELS + eurocode 3 · s 1,5" in texte(nv["donnees"])
         and 'data-chemin="historique/s"' in nv["donnees"] and nv["etat"]["elements"][0]["methodes"] == ["historique", "ec3"],
         re.findall(r'data-valeur="historique"[^>]*>[^<]*<', nv["donnees"])),
        ("le résumé de la note (l'accueil, la liste des notes) : « flexion (ELS) »", nv["resume"]["gouvernant"] == "B1 · IPE 300 · flexion (ELS)", nv["resume"]),
        ("« tout le calcul » de la méthode : « B1 · méthode ELS (NBN B 51-001) », sans répéter la norme dessous ; la comparaison : colonne « ELS », "
         "« La méthode ELS (NBN B 51-001) compare des contraintes de service »",
         "<b>B1 · méthode ELS (NBN B 51-001)</b><span>IPE 300 · S235 · 6,00\u00a0m · tout le calcul</span>" in nv["volets"]
         and '<th class="n">ELS</th><th class="n">Eurocode 3</th>' in nv["volets"]
         and "La méthode ELS (NBN B 51-001) compare des contraintes de service" in texte(nv["volets"]),
         re.findall(r"<b>B1 · m[^<]*</b><span>[^<]*</span>", nv["volets"])),
        ("la vue des taux : « Méthode ELS (NBN B 51-001) » en titre, la norme pas écrite une seconde fois à droite",
         ">Méthode ELS (NBN B 51-001)<" in tx_ and tx_.count("NBN B 51-001") == 1 and ">NBN EN 1993-1-1<" in tx_, re.findall(r">([^<]*NBN[^<]*)<", tx_)),
        ("le sélecteur : « vérifiés par : ELS et eurocode 3 »", "vérifiés par : ELS et eurocode 3" in P.selecteur(nv["etat"], "profil"), ""),
        ("les efforts imposés : l'aide du mode et celle de la colonne « Service » nomment la méthode EN ENTIER, « la méthode ELS (NBN B 51-001) » "
         "(des survols : la place ne manque pas — troisième passe du lot A)",
         "ceux de service pour la méthode ELS (NBN B 51-001), ceux de calcul" in visible(ef_["donnees"])
         and "Efforts non pondérés : ceux de la méthode ELS (NBN B 51-001)." in visible(ef_["donnees"]),
         re.findall(r"(?:ceux de service pour|Efforts non pondérés : ceux de) la [^,.]*", visible(ef_["donnees"]))),
        ("une nuance hors du domaine : le message du MOTEUR nomme la méthode en entier, « Hors du domaine de la méthode ELS (NBN B 51-001) — "
         "nuance hors AE235 à AE355 : calcul pour information » — le nom de l'écran (profil_mots.METHODES), une seule forme",
         "Hors du domaine de la méthode ELS (NBN B 51-001) — nuance hors AE235 à AE355 : calcul pour information" in texte(hd_["volets"])
         and f"Hors du domaine de la {M.minuscule(M.METHODES['historique'])} — " in texte(hd_["volets"]),
         re.findall(r"Hors du domaine[^.]{0,80}", texte(hd_["volets"]))[:1]),
        ("nulle part — résultats, saisie, fenêtres, vues, survols ; barre neuve, méthode seule, efforts imposés, S460 — l'écran n'écrit plus "
         "« historique »", not restes, restes[:4]),
    ]


def _source_mutee(module, ancien, nouveau):
    """Une copie d'un fichier de l'écran, son TEXTE muté (une seule occurrence de ``ancien``)."""
    src = open(module.__file__, encoding="utf-8").read()
    if src.count(ancien) != 1:
        raise ValueError(f"texte à muter absent ou répété dans {module.__name__}")
    spec = importlib.util.spec_from_file_location(module.__name__ + "_mute", module.__file__)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = module.__package__
    exec(compile(src.replace(ancien, nouveau), module.__file__, "exec"), mod.__dict__)
    return mod


def _message_historique(vrai):
    def f(d):
        r = vrai(d)
        for v in r["verifications"]:
            v["messages"] = [m_.replace("méthode ELS", "méthode historique") for m_ in v["messages"]]
        return r
    return f


def _message_sans_norme(vrai):
    def f(d):
        r = vrai(d)
        for v in r["verifications"]:
            v["messages"] = [m_.replace("méthode ELS (NBN B 51-001)", "méthode ELS") for m_ in v["messages"]]
        return r
    return f


for nom_, ok_, info_ in nom_els():
    chk(nom_, ok_, info_)
MUTATIONS_NOM = (
    ("l'ancien nom revient dans les titres (« Méthode historique »)", M, "METHODES", lambda vrai: dict(vrai, historique="Méthode historique")),
    ("l'ancien nom court revient (« Historique »)", M, "METHODES_COURT", lambda vrai: dict(vrai, historique="Historique")),
    ("le segment du choix des méthodes redit « Historique »", M, "CHOIX_METHODES",
     lambda vrai: tuple((k, "Historique" if k == "historique" else t) for k, t in vrai)),
    ("« ELS » écrit en minuscules au fil d'une phrase (« els + eurocode 3 », « (els) »)", M, "minuscule", lambda vrai: (lambda s: s.lower())),
    ("la note de la comparaison redit « La méthode historique »", PR, "NOTE_COMPARAISON", lambda vrai: vrai.replace("méthode ELS (NBN B 51-001)", "méthode historique")),
    ("l'aide de la colonne « Service » redit « la méthode historique »", PS, "AIDE_JEU", lambda vrai: dict(vrai, service="Efforts non pondérés : ceux de la méthode historique.")),
    ("l'aide du mode « Efforts imposés » redit « la méthode historique »", M, "MODES",
     lambda vrai: tuple((k, t, a.replace("méthode ELS", "méthode historique")) for k, t, a in vrai)),
    ("le message « hors domaine » du moteur redit « la méthode historique »", MO.HI, "verifier", _message_historique),
    ("troisième passe — l'aide du mode « Efforts imposés » dit « méthode ELS » sans la norme", M, "MODES",
     lambda vrai: tuple((k, t, a.replace("méthode ELS (NBN B 51-001)", "méthode ELS")) for k, t, a in vrai)),
    ("troisième passe — l'aide de la colonne « Service » dit « méthode ELS » sans la norme", PS, "AIDE_JEU",
     lambda vrai: dict(vrai, service="Efforts non pondérés : ceux de la méthode ELS.")),
    ("troisième passe — le message « hors domaine » du moteur dit « méthode ELS » sans la norme", MO.HI, "verifier", _message_sans_norme),
    ("la vue des taux écrit la norme une seconde fois à droite", P, "DB",
     lambda vrai: _source_mutee(vrai, '        if x["norme"] not in titre:  ', '        if True:  ')),
    ("« tout le calcul » répète la norme sous le titre", P, "R",
     lambda vrai: _source_mutee(vrai, '        norme = "" if x["norme"] in M.METHODES[k] else', '        norme = f"{C.esc(x[\'norme\'])} · " if True else')),
)
for nom_, objet_, attr_, fabrique_ in MUTATIONS_NOM:
    if not hasattr(objet_, attr_):
        chk(f"mutation « {nom_} » : l'attribut {attr_} existe", False, objet_)
        continue
    vrai_ = getattr(objet_, attr_)
    try:
        setattr(objet_, attr_, fabrique_(vrai_))
        rouges_ = [n_ for n_, ok_, _i in nom_els() if not ok_]
    except Exception as err_:  # noqa: BLE001 — une mutation qui fait planter l'écran est vue
        rouges_ = [f"exception : {err_}"]
    finally:
        setattr(objet_, attr_, vrai_)
        P._RESULTATS.clear()
    chk(f"mutation « {nom_} » : {len(rouges_)} contrôle(s) rougissent", bool(rouges_))
chk("témoin : sans mutation, les contrôles du nom sont verts", all(ok_ for _n, ok_, _i in nom_els()))

# ------------------------------------------------------------------ 13. le refus au-delà de 80 mm, à la française (troisième passe)
from acier.flambement import normes as NO  # noqa: E402


def refus_80():
    """Le refus d'un profilé de paroi > 80 mm tel que l'écran l'écrit : [(nom, vrai ?, info)] (rejoué sous mutation)."""
    P._RESULTATS.clear()
    r_ = req(action="calculer", etat=note(barre(profil="HL 920x970")))
    t_ = " ".join(texte(r_[k]) for k in ("resultat", "volets"))
    return [("HL 920x970 (semelle de 89,9 mm) : l'écran écrit « épaisseur 89,9 mm > 80 mm : hors du domaine du module », virgule décimale, "
             "jamais « 89.9 »", "épaisseur 89,9 mm > 80 mm : hors du domaine du module" in t_ and "89.9" not in t_,
             re.findall(r"épaisseur [^:]*:", t_)[:2])]


for nom_, ok_, info_ in refus_80():
    chk(nom_, ok_, info_)
vrai_ = NO.limite_elastique


def _fy_point(nuance, t, froid=False, produit=None):
    if t > NO.T_MAX_TABLEAU:                             # l'ancien refus : le nombre écrit par Python (« 89.9 »)
        raise NO.HorsDomaine(f"épaisseur {t:g} mm > {NO.T_MAX_TABLEAU:g} mm : hors du domaine du module")
    return vrai_(nuance, t, froid, produit)


try:
    NO.limite_elastique = _fy_point
    rouges_ = [n_ for n_, ok_, _i in refus_80() if not ok_]
finally:
    NO.limite_elastique = vrai_
    P._RESULTATS.clear()
chk(f"mutation « troisième passe — refus de 80 mm écrit avec un point décimal » : {len(rouges_)} contrôle(s) rougissent", bool(rouges_))

print("=== 14. (2e reprise du lot B) s_s et c non saisis à l'écran : la borne dite, le taux sur la ligne, le résumé des Appuis — et leurs mutations")
from api_web import profil_saisie as PS  # noqa: E402


def borne_ecran():
    """Les contrôles de la borne sur l'écran, rejoués sous chaque mutation : [(nom, vrai ?, info)]."""
    P._RESULTATS.clear()
    n_ = req(action="nouveau")
    s_ = saisir(n_["etat"], "appuis/ss", "100")
    z_ = saisir(s_["etat"], "appuis/c", "0")

    def resume_ap(rep_):
        t_ = texte(rep_["donnees"])
        i_ = t_.find("Appuis posée")
        return t_[i_:i_ + 40] if i_ >= 0 else ""
    tn, ts, tz = texte(n_["resultat"]), texte(s_["resultat"]), texte(z_["resultat"])
    return [
        ("le résumé de la rubrique Appuis dit ce qui manque — « s_s à saisir · c à saisir » ; s_s saisie : « s_s 100 mm · c à saisir » ; "
         "c = 0 SAISI se lit (« c 0 mm »), il n'est pas « à saisir »",
         resume_ap(n_).startswith("Appuis posée · s s à saisir · c à saisir") and resume_ap(s_).startswith("Appuis posée · s s 100 mm · c à saisir")
         and resume_ap(z_).startswith("Appuis posée · s s 100 mm · c 0 mm"), (resume_ap(n_), resume_ap(s_), resume_ap(z_))),
        ("la ligne d'un lieu impossible porte le taux à la borne (« Appui A 157 % décision impossible : s s à saisir ») ; à la borne qui "
         "passe, la ligne dit la borne (« sans raidisseur (c = 0 : borne prudente) ») ; c = 0 saisi : « sans raidisseur », sans borne",
         "Appui A 157 % décision impossible : s s à saisir" in tn and "Appui A 22 % F Ed 44,5 ≤ F Rd 199,8 kN · sans raidisseur (c = 0 : borne prudente)" in ts
         and "Appui A 22 % F Ed 44,5 ≤ F Rd 199,8 kN · sans raidisseur Appui B" in tz, (tn[-300:], ts[-300:])),
    ]


for nom_, ok_, info_ in borne_ecran():
    chk(nom_, ok_, info_)


def _resume_ancien(vrai):
    def f(el):
        ap = el["appuis"]
        if ap["mode"] != "poses":
            return vrai(el)
        ss = f"s<sub>s</sub> {M.n(ap['ss'], 0)} mm" if ap["ss"] is not None else "s<sub>s</sub> à saisir"
        return f"{M.NOM_MODE_APPUI['poses']} · {ss}" + (f" · c {M.n(ap['c'], 0)} mm" if ap["c"] else "")
    return f


MUTATIONS_BORNE = (
    ("le résumé des Appuis tait c (manquant, ou 0 saisi)", PS, "_resume_appuis", _resume_ancien),
    ("la ligne d'un lieu ne dit pas la borne", M, "decision_lieu",
     lambda vrai: (lambda L: "sans raidisseur" if L["decision"] == "aucun" else vrai(L))),
    ("la ligne d'un lieu impossible perd son taux", MO, "verifier",
     lambda vrai: (lambda el: {**vrai(el), "ame": {**vrai(el)["ame"], "lieux": [dict(L, taux=None if L["decision"] == "impossible" else L["taux"])
                                                                           for L in vrai(el)["ame"]["lieux"]]}})),
)
for nom_, objet_, attr_, fabrique_ in MUTATIONS_BORNE:
    if not hasattr(objet_, attr_):
        chk(f"mutation « {nom_} » : l'attribut {attr_} existe", False, objet_)
        continue
    vrai_ = getattr(objet_, attr_)
    setattr(objet_, attr_, fabrique_(vrai_))
    try:
        rouges_ = [n_ for n_, ok_, _i in borne_ecran() if not ok_]
    except Exception as err_:  # noqa: BLE001 — une mutation qui fait planter l'écran est vue
        rouges_ = [f"exception : {err_}"]
    finally:
        setattr(objet_, attr_, vrai_)
        P._RESULTATS.clear()
    chk(f"mutation « {nom_} » : {len(rouges_)} contrôle(s) rougissent", bool(rouges_))
chk("témoin : sans mutation, les contrôles de la borne à l'écran sont verts", all(ok_ for _n, ok_, _i in borne_ecran()))

print("=== 15. (lot B) le format de la note : une note du site d'avant (ses 0 de s_s et c : des défauts) ; une note neuve (vide et 0 "
      "distincts) ; l'aide de la longueur d'appui — et leurs mutations")
AIDE_APPUIS = "Longueur d'appui ou débord vide : calcul à la borne 0 (prudent)."


def note_v2():
    """Les contrôles de la relecture d'une note du site d'avant le lot B, d'une note neuve et de l'aide de s_s, rejoués sous chaque
    mutation : [(nom, vrai ?, info)]."""
    P._RESULTATS.clear()
    haute = barre(profil="IPE 750x147", travees=[4.0], poids_propre=False, maintiens=[{"x": 2.0, "genre": "complet"}],
                  appuis={"mode": "poses", "ss": 200.0, "c": 0.0},
                  cas=[{"nom": "G", "nature": "permanent", "N": 0.0, "charges": [{"type": "q", "v": 220.0, "a": 0.0, "b": 4.0},
                                                                                {"type": "P", "v": 1.0, "a": 2.0, "ss": 0.0}]}])
    v2 = req(action="calculer", etat={"version": 2, "actif": 0, "elements": [haute]})          # le site d'avant : sans la clé
    v3 = req(action="calculer", etat=note(haute))
    e2 = v2["etat"]["elements"][0] if v2.get("ok") else {}
    e3 = v3["etat"]["elements"][0] if v3.get("ok") else {}
    a2, a3 = (lieu_a(moteur(e)) if e else None for e in (e2, e3))
    don = html.unescape(req(action="nouveau")["donnees"])
    ch = html.unescape(req(action="calculer", etat=note(barre(cas=[{"nom": "G", "nature": "permanent", "N": 0.0,
                                                                     "charges": [{"type": "P", "v": 5.0, "a": 3.0}]}])))["donnees"])
    # une note neuve : c = 0 SAISI, s_s vidée — rejouée deux fois (l'état rendu est celui que l'écran renvoie)
    n1 = req(action="modifier", etat=req(action="modifier", etat=P.nouvel_etat(), chemin=["appuis", "c"], valeur="0")["etat"], chemin=["appuis", "ss"],
             valeur="")
    n2 = req(action="calculer", etat=n1["etat"]) if n1.get("ok") else {}
    n3 = req(action="calculer", etat=n2["etat"]) if n2.get("ok") else {}
    ap3 = n3["etat"]["elements"][0]["appuis"] if n3.get("ok") else {}
    v9 = req(action="calculer", etat=dict(note(haute), version=3))
    return [
        ("une note du SITE D'AVANT le lot B (version 2, sans « zero_saisi » : il écrivait 0,0 pour s_s et c non saisis) : relue, ses 0 deviennent "
         "« non saisis » — appuis (c) et charge ponctuelle (s_s) ; une valeur non nulle est gardée (s_s = 200 mm) ; rendue au format neuf",
         v2.get("ok") and v2["etat"]["version"] == P.VERSION == 2 and v2["etat"].get(P.ZERO_SAISI) is True and e2["appuis"]["ss"] == 200.0
         and e2["appuis"]["c"] is None and e2["mode"] == "chargee" and e2["cas"][0]["charges"][1]["ss"] is None and e2["profil"] == "IPE 750x147",
         e2.get("appuis")),
        ("… le côté prudent : IPE 750x147 S235, 4 m, 220 kN/m, s_s = 200 mm — relue, l'appui A dit « décision impossible : c à saisir » (borne "
         "542,1 kN < 594,0 kN) ; la même note au format neuf (c = 0 SAISI) décide « sans raidisseur » (674,0 kN, 88 %) — alors qu'à c = 20 mm, il "
         "faut un raidisseur (553,4 kN) : le 0 d'une note du site d'avant n'est pas une saisie",
         a2 is not None and a2["decision"] == "impossible" and a2["raison"] == "Décision impossible : c à saisir." and round(a2["sans"]["F_Rd"], 1) == 542.1
         and a3 is not None and a3["decision"] == "aucun" and round(a3["sans"]["F_Rd"], 1) == 674.0 and e3["appuis"]["c"] == 0.0
         and req(action="calculer", etat=v2["etat"])["etat"] == v2["etat"], (a2 and a2["raison"], a3 and a3["decision"])),
        ("une note NEUVE garde « vide » et « 0 » distincts : c = 0 saisi reste 0, s_s vidée reste non saisie (null), après deux relectures ; la "
         "note garde sa version (2) et sa clé « zero_saisi » — sans changer de version (le harnais suit : version 3 à la 3e reprise)",
         n3.get("ok") and ap3 == {"mode": "poses", "ss": None, "c": 0.0} and n3["etat"]["version"] == 2 and n3["etat"].get(P.ZERO_SAISI) is True
         and n3["etat"] == n2["etat"], ap3),
        ("une note d'un AUTRE format (version 3) est refusée — jamais lue comme une autre (le site d'avant prenait toute version inconnue pour une "
         "version 1 : « efforts imposés »)",
         not v9.get("ok") and "format inconnu" in v9.get("erreur", ""), v9.get("erreur")),
        ("l'aide de la longueur d'appui (rubriques Appuis et Charges, et le survol de « s_s » et « sur ») dit le sens d'un champ VIDE — « calcul "
         "à la borne 0 (prudent) » — et plus jamais « 0 : le cas le plus défavorable » (faux au type (c))",
         AIDE_APPUIS in don and "Vide : calcul à la borne 0 (prudent)." in don and "(vide : calcul à la borne 0, prudent)" in ch
         and "Vide : calcul à la borne 0 (prudent)." in ch and "défavorable" not in don + ch, re.findall(r".{60}défavorable.{20}", don + ch)[:2]),
    ]


def lieu_a(r):
    return next((L for L in r["ame"]["lieux"] if L["nom"] == "Appui A"), None)


for nom_, ok_, info_ in note_v2():
    chk(nom_, ok_, info_)


def _sans_cle(vrai):
    def etat_de(elements, actif=0):
        e = vrai(elements, actif)
        e.pop(P.ZERO_SAISI, None)
        return e
    return etat_de


MUTATIONS_V2 = (
    ("une note du site d'avant relue telle quelle (ses 0 pris pour des saisies)", P, "_de_v2", lambda vrai: (lambda e: e)),
    ("une note du site d'avant prise pour une v1 (convertie en efforts imposés)", P, "_de_v2", lambda vrai: P._migrer),
    ("la clé « zero_saisi » ignorée à la relecture (les 0 SAISIS d'une note neuve ramenés à « non saisi »)", P, "normaliser",
     lambda vrai: _source_mutee(P, "    neuve = version == VERSION and etat.get(ZERO_SAISI) is True", "    neuve = False").normaliser),
    ("la clé « zero_saisi » plus écrite (une note neuve relue comme une note du site d'avant)", P, "etat_de", _sans_cle),
    ("une version inconnue lue comme une version 1", P, "normaliser",
     lambda vrai: _source_mutee(P, "    if version not in (1, VERSION):", "    if False:").normaliser),
    ("l'aide de s_s redit « 0 : le cas le plus défavorable »", M, "AIDE_SS", lambda vrai: vrai.replace("Vide : calcul à la borne 0 (prudent).", "0 : le cas le plus défavorable.")),
    ("l'aide des Appuis redit « longueur d'appui à 0 : le cas le plus défavorable »", P, "S",
     lambda vrai: _source_mutee(vrai, "ou débord vide : calcul à la borne 0 (prudent).", "à 0 : le cas le plus défavorable.")),
)
for nom_, objet_, attr_, fabrique_ in MUTATIONS_V2:
    if not hasattr(objet_, attr_):
        chk(f"mutation « {nom_} » : l'attribut {attr_} existe", False, objet_)
        continue
    vrai_ = getattr(objet_, attr_)
    try:
        setattr(objet_, attr_, fabrique_(vrai_))
        rouges_ = [n_ for n_, ok_, _i in note_v2() if not ok_]
    except Exception as err_:  # noqa: BLE001 — une mutation qui fait planter l'écran est vue
        rouges_ = [f"exception : {err_}"]
    finally:
        setattr(objet_, attr_, vrai_)
        P._RESULTATS.clear()
    chk(f"mutation « {nom_} » : {len(rouges_)} contrôle(s) rougissent", bool(rouges_))
chk("témoin : sans mutation, les contrôles du format et de l'aide sont verts", all(ok_ for _n, ok_, _i in note_v2()))

print("=== 16. (lot B) s_s et c se VIDENT (retour à « non saisi ») ; une note neuve relue par le SITE D'AVANT (766659c) — et leurs mutations")
ANCIEN_SITE = "766659c"          # le site en ligne avant le lot B (rapport du lot A, 16/11/2026) : la page qu'un cache ou un retour arrière rendrait
_ANCIEN: dict = {}


def ancien_site():
    """Le dossier où le site d'avant est tiré de l'historique (git archive : api_web, acier, rdm, beton — ce que son profil.py importe), une
    fois ; None, et la raison, si l'historique ne le donne pas."""
    if "dossier" not in _ANCIEN:
        try:
            tar = subprocess.run(["git", "-C", RACINE, "archive", ANCIEN_SITE, "api_web", "acier", "rdm", "beton"], capture_output=True, check=True).stdout
            d = tempfile.mkdtemp(prefix="profil_766659c_")
            with tarfile.open(fileobj=io.BytesIO(tar)) as t:
                t.extractall(d, filter="data")
            _ANCIEN.update(dossier=d, raison=None)
        except (OSError, subprocess.CalledProcessError, tarfile.TarError) as e:
            _ANCIEN.update(dossier=None, raison=f"le commit {ANCIEN_SITE} n'est pas lisible : {e}")
    return _ANCIEN["dossier"]


def lu_par_ancien(etats):
    """Les notes ``etats`` (un dict nom → état) relues par le ``normaliser`` du site d'avant, dans un processus à part : nom → {"ok", "etat" |
    "erreur"}."""
    d = ancien_site()
    if d is None:
        return {}
    code = ("import json, sys\nsys.path.insert(0, %r)\nfrom api_web import profil as P\nout = {}\n"
            "for k, e in json.loads(sys.stdin.read()).items():\n"
            "    try:\n        out[k] = {'ok': True, 'etat': P.normaliser(e)}\n"
            "    except Exception as ex:\n        out[k] = {'ok': False, 'erreur': type(ex).__name__ + ' : ' + str(ex)}\n"
            "print(json.dumps(out))\n") % d
    p_ = subprocess.run([sys.executable, "-B", "-c", code], input=json.dumps(etats), capture_output=True, text=True, cwd=d)
    return json.loads(p_.stdout) if p_.returncode == 0 else {"_": {"ok": False, "erreur": p_.stderr[-300:]}}


def vider_et_relire():
    """Les contrôles : un champ s_s ou c se vide (écran et état) ; la note neuve relue par le site d'avant. [(nom, vrai ?, info)]."""
    P._RESULTATS.clear()
    e_ = req(action="modifier", etat=P.nouvel_etat(), chemin=["cas", "0", "charges", "+"], valeur="P")["etat"]
    pleins = e_
    for ch_, v_ in ((["appuis", "ss"], "100"), (["appuis", "c"], "0"), (["cas", "0", "charges", "1", "ss"], "80")):
        pleins = req(action="modifier", etat=pleins, chemin=ch_, valeur=v_)["etat"]
    vides = pleins
    rep_ = None
    for ch_ in (["appuis", "ss"], ["appuis", "c"], ["cas", "0", "charges", "1", "ss"]):
        rep_ = req(action="modifier", etat=vides, chemin=ch_, valeur="")
        vides = rep_["etat"] if rep_.get("ok") else vides
    el_p, el_v = pleins["elements"][0], vides["elements"][0]
    champs = re.findall(r'<input data-chemin="(appuis/ss|appuis/c|cas/0/charges/1/ss)" data-sorte="(\w+)"[^>]*?value="([^"]*)"[^>]*?(data-vide="auto" '
                        r'placeholder="([^"]*)")?>', (rep_ or {}).get("donnees", ""))
    abc = req(action="modifier", etat=pleins, chemin=["appuis", "c"], valeur="abc")
    lus = lu_par_ancien({"neuve": P.nouvel_etat(), "pleine": pleins, "videe": vides})
    pl_ = lus.get("pleine", {})
    el_a = pl_["etat"]["elements"][0] if pl_.get("ok") else {}
    return [
        ("un champ s_s ou c SAISI se VIDE — appuis (s_s, c) et charge ponctuelle (s_s) : « » rend « non saisi » (null), comme un champ « vide = "
         "auto » ; à l'écran, ces champs acceptent d'être vidés (data-vide) et disent « à saisir » vides ; un texte qui n'est pas un nombre reste refusé",
         (el_p["appuis"]["ss"], el_p["appuis"]["c"], el_p["cas"][0]["charges"][1]["ss"]) == (100.0, 0.0, 80.0)
         and (el_v["appuis"]["ss"], el_v["appuis"]["c"], el_v["cas"][0]["charges"][1]["ss"]) == (None, None, None) and rep_.get("ok")
         and sorted(champs) == sorted([("appuis/ss", "texte", "", 'data-vide="auto" placeholder="à saisir"', "à saisir"),
                                       ("appuis/c", "texte", "", 'data-vide="auto" placeholder="à saisir"', "à saisir"),
                                       ("cas/0/charges/1/ss", "texte", "", 'data-vide="auto" placeholder="à saisir"', "à saisir")])
         and abc.get("ok") and "n'est pas un nombre" in (abc.get("message") or "") and abc["etat"]["elements"][0]["appuis"]["c"] == 0.0,
         (el_v["appuis"], champs, abc.get("message"))),
        (f"le SITE D'AVANT ({ANCIEN_SITE}, son normaliser exécuté) relit une note NEUVE telle quelle quand tout est saisi — 0 saisi = 0, et aucun "
         "autre champ ne change ; une donnée vide (null) : il la REFUSE, franchement (« nombre attendu »), jamais une autre lecture (une version 3 "
         "y devenait « efforts imposés », s_s et c à 0)",
         _ANCIEN.get("raison") is None and pl_.get("ok") and pl_["etat"]["version"] == 2 and el_a == {k: el_p[k] for k in el_a}
         and set(el_a) == set(el_p) and lus.get("neuve", {}).get("ok") is False and "nombre attendu" in lus["neuve"].get("erreur", "")
         and lus.get("videe", {}).get("ok") is False and "nombre attendu" in lus["videe"].get("erreur", ""),
         _ANCIEN.get("raison") or {k: (v.get("ok"), v.get("erreur")) for k, v in lus.items()}),
    ]


for nom_, ok_, info_ in vider_et_relire():
    chk(nom_, ok_, info_)
MUTATIONS_VIDE = (
    ("s_s et c lus comme un nombre obligatoire (le champ ne se vide plus)", P, "_lire_vide", lambda vrai: P._lire),
    ("le champ s_s ou c rendu sans « vide » (l'écran refuse un nombre vide)", M, "SS_VIDE", lambda vrai: ""),
    ("la note écrite en version 3 (le site d'avant la lirait comme une version 1)", P, "VERSION", lambda vrai: 3),
    ("une donnée vide écrite 0 (le site d'avant la lirait comme un 0 saisi)", P, "_lire_vide", lambda vrai: (lambda v, q: 0.0 if not str(v).strip() else P._lire(v, q))),
)
for nom_, objet_, attr_, fabrique_ in MUTATIONS_VIDE:
    if not hasattr(objet_, attr_):
        chk(f"mutation « {nom_} » : l'attribut {attr_} existe", False, objet_)
        continue
    vrai_ = getattr(objet_, attr_)
    try:
        setattr(objet_, attr_, fabrique_(vrai_))
        rouges_ = [n_ for n_, ok_, _i in vider_et_relire() if not ok_]
    except Exception as err_:  # noqa: BLE001 — une mutation qui fait planter l'écran est vue
        rouges_ = [f"exception : {err_}"]
    finally:
        setattr(objet_, attr_, vrai_)
        P._RESULTATS.clear()
    chk(f"mutation « {nom_} » : {len(rouges_)} contrôle(s) rougissent", bool(rouges_))
chk("témoin : sans mutation, les contrôles du champ vidé et de la relecture par le site d'avant sont verts", all(ok_ for _n, ok_, _i in vider_et_relire()))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

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
     le site d'avant, 766659c, tirée de l'historique : ce contrôle demande l'historique complet) ; s_s et c se vident.
 17. LOT B — le message du maintien à l'écran (la ligne sans le taux de sa pastille, la bande le lieu en tête).

    /usr/bin/python3.12 tests/test_api_web_profil.py
"""
import atexit
import copy
import html
import io
import json
import os
import re
import shutil
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
from acier.barre import resultat as RE  # noqa: E402
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
    and [c["nom"] for c in el0["cas"]] == ["G", "Q"] and el0["poids_propre"] is True and el0["maintiens"] == [{"x": 3.0, "genre": "complet", "neglige": False}]
    and el0["methodes"] == ["historique", "ec3"], el0)
th, te = r0["methodes"]["historique"]["taux_max"], r0["methodes"]["ec3"]["taux_max"]
chk("ses DEUX méthodes tiennent (ELS 88 % — lot B : le déversement des classeurs, brief v12 § 6.4 ; Eurocode 3 58 % — lot A : le couple "
    "(λ̄_LT,0 ; β) = (0,2 ; 1,0) de l'ANB:2018, 6.3.2.3(1)) ; mais s_s et c ne sont pas saisis, et à leur borne 0 l'âme ne passe pas (critère "
    "6.2.1(5), 157 %) : décision impossible (brief § 8.3, point 5) — la note est « À vérifier », son résumé porte le taux le plus fort ; seule "
    "s_s est demandée (le critère ne lit pas c — le harnais suit : « s_s et c » à la 2e reprise)",
    r0["statut"] == "bloque" and r0["ame"]["statut"] == "bloque" and round(th, 2) == 0.88 and round(te, 2) == 0.58
    and neuf["resume"]["statut"] == "À vérifier" and neuf["resume"]["taux"] == C.pct(max(th, te)) and neuf["resume"]["niveau"] == "v"
    and neuf["resume"]["gouvernant"] == "B1 · IPE 300 · déversement (ELS)"
    and "Âme · Décision impossible : s s à saisir (appui A, appui B)" in texte(neuf["resultat"]), (th, te, neuf["resume"]))
# la base décidée : la note neuve, s_s = 100 mm et c = 0 saisis
DECIDE = enchainer(e0, ("appuis/ss", "100"), ("appuis/c", "0"))
chk("… s_s = 100 mm et c = 0 saisis : l'âme décide (pas de raidisseur, 22 %) et la note redevient « Conforme »",
    DECIDE["resume"]["statut"] == "Conforme" and DECIDE["resume"]["niveau"] == "p" and moteur(DECIDE["etat"]["elements"][0])["statut"] == "ok"
    and "Âme ·" not in texte(DECIDE["resultat"]), DECIDE["resume"])
res0 = texte(neuf["resultat"])
# (lot C, C1, le harnais suit : le poids propre au centre de GRAVITÉ, et non sur la semelle haute avec les charges — M_cr monte,
#  M_b,Rd 115,6 → 115,7 kN·m)
# (lot C, C2, le harnais suit : le poids propre et G ont la même répartition, ils font une source et s'écrivent « G »)
chk("un VERDICT par méthode : le taux du moteur, la vérification qui gouverne, ce qu'elle compare, sa combinaison et son lieu",
    neuf["resultat"].count('class="verdict"') == 2 and f"η {pastille(th)} Méthode ELS (NBN B 51-001) · déversement σ M 84,2 ≤ σ adm,d 95,3 N/mm²" in res0
    and "G + Q · x = 3,00 m" in res0 and f"η {pastille(te)} Eurocode 3 · déversement M y,Ed 66,7 ≤ M b,Rd 115,7 kN·m" in res0
    and "1,35·G + 1,5·Q · de 0,00 à 3,00 m" in res0 and "Poids propre +" not in res0, res0[:500])
chk("le poids propre de la saisie est celui que le MOTEUR nomme (78,6 kN/m³ × A = 0,42 kN/m) — l'écran ne le recalcule pas",
    round(r0["poids_propre"], 4) == 0.4230 and "(0,42 kN/m)" in texte(neuf["donnees"]), texte(neuf["donnees"])[:300])
# (lot C, C4 — le harnais suit : la masse n'est plus au pied, le bouton du profilé, à gauche, la dit — une information à un endroit)
chk("la tête dit la barre : « B1 », « IPE 300 · S235 · 6,00 m » ; le pied, la flèche de service (une information) — la masse au seul bouton",
    '<div class="et-rp"><b>B1</b><span>IPE 300 · S235 · 6,00 m</span></div>' in neuf["resultat"] and "kg/m" not in res0 and "42,2 kg/m" in texte(neuf["donnees"])
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
    # les verdicts : le taux le plus fort de chaque méthode. (lot C, C4 — le harnais suit : la méthode ELS n'a jamais le contrôle
    # hérité H-EQ en tête ; quand il est plus fort, son taux — le plus fort de la méthode — est écrit SOUS le verdict, pas caché)
    def _dit(k, x):
        g_ = PR.tete(k, x)
        if g_ is PR.gouvernante(x):
            return [f"η {pastille(x['taux_max'])} {PR.M.METHODES[k]}"]
        h_ = PR.herite_au_dela(x, g_)
        return [f"η {pastille(g_['taux'])} {PR.M.METHODES[k]}"] + ([f"+ {PR.M.minuscule(h_['titre'])} (contrôle hérité) : {pastille(h_['taux'])}"] if h_ else [])
    ecarts = [k for k, x in r["methodes"].items() if x["taux_max"] is not None and any(t_ not in texte(rep["resultat"]) for t_ in _dit(k, x))]
    chk(f"… et chaque verdict porte le taux le plus fort de sa méthode (l'ELS : sous le verdict quand c'est le contrôle hérité) ; le résumé de "
        f"la note, le plus fort des deux", not ecarts
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
# (lot B : en flexion simple, la méthode ELS n'écrit plus que la ligne du déversement des classeurs — ses P_cr, qui ne servaient pas, ne
# sont plus répétés : 19 formules ; le seuil est ce compte, pour qu'une formule perdue se voie)
chk(f"les fenêtres de calcul portent CHAQUE formule du moteur ({n}), écrite par acier.barre.ecriture : ses valeurs et son résultat",
    n >= 19 and not absentes, absentes[:3])
vol = texte(rep["volets"])
chk("la fenêtre d'une vérification dit la méthode, la barre, la combinaison et le lieu ; ses hypothèses sont celles du moteur",
    "Eurocode 3 · HEA 200 · S235 · 5,00 + 4,00 m ·" in vol and all(texte(PR.M.sub(h)) in vol for x in r["methodes"].values() for h in x["hypotheses"]),
    vol[:300])
chk("« tout le calcul » de chaque méthode (s1, s2) porte toutes ses vérifications et le bilan de ses combinaisons",
    all(f'id="ck-s{i}"' in rep["volets"] for i in (1, 2)) and "Combinaisons" in vol
    and all(c["nom"] in vol for x in r["methodes"].values() for c in x["combinaisons"]))
cmp_ = re.search(r'<section class="et-bloc inc" data-theme="4">(.*?)</section>', rep["resultat"], re.S)
# (lot C, C0f — le harnais suit : la comparaison de acier/barre/comparaison.py, ses sujets du § 9 ; plus de « Le plus défavorable » ; l'âme,
# « commun », sur sa propre ligne à cheval sur les deux colonnes)
chk("la COMPARAISON (le quatrième bloc, après les deux méthodes et l'âme) : un sujet par ligne, la case de chaque méthode — le taux que le "
    "moteur rend ; l'âme, qui vaut pour les deux, porte UNE pastille par ligne commune ; plus de ligne « Le plus défavorable » ; sa fenêtre (cmp)",
    cmp_ is not None and all(li["sujet"] in texte(cmp_.group(1)) for li in r["comparaison"])
    and all(pastille(li[k]["taux"]) in texte(cmp_.group(1)) for li in r["comparaison"] for k in ("historique", "ec3", "commun")
            if li[k] is not None and li[k]["taux"] is not None)
    and cmp_.group(1).count("pf-cmp-c") == sum(1 for li in r["comparaison"] if li["commun"] is not None) >= 1
    and "Le plus défavorable" not in texte(cmp_.group(1))
    and 'data-calcul="cmp"' in rep["resultat"] and 'id="ck-cmp"' in rep["volets"] and "vaut pour les deux méthodes" in texte(rep["volets"]),
    texte(cmp_.group(1)) if cmp_ else None)
chk("(lot A) la poutre continue n'a plus de paramètre « à confirmer » : le couple (λ̄_LT,0 ; β) est celui de l'ANB:2018, 6.3.2.3(1) — ni lien "
    "ni fenêtre (ac)", not r["methodes"]["ec3"]["a_confirmer"] and "à confirmer" not in texte(rep["donnees"]) and 'id="ck-ac"' not in rep["volets"])
rpp = REPONSES["un poteau-poutre en efforts imposés"]
chk("(lot B) le poteau-poutre n'a plus de paramètre « à confirmer » : l'interaction suit l'annexe D ANB (méthode 1 lue sur l'ANB:2018, "
    "6.3.3(5) ; C_mi,0 sans borne, tableau D.1 ANB) — ni lien ni fenêtre (ac)",
    not moteur(rpp["etat"]["elements"][0])["methodes"]["ec3"]["a_confirmer"] and 'id="ck-ac"' not in rpp["volets"])
# (lot B : le harnais suit — le seul point « à confirmer » qui reste à l'interaction est γ_M0 ≠ γ_M1, brief § 7.8)
rpp = req(action="calculer", etat=note(dict(POTEAU, parametres={"gamma_M0": 1.05})))
ac = moteur(rpp["etat"]["elements"][0])["methodes"]["ec3"]["a_confirmer"]
# (lot C, C3 — le harnais suit : le lien « 1 paramètre à confirmer » devient UNE ligne près des champs γ_M, qui ouvre la même fenêtre)
chk(f"les paramètres « à confirmer » de l'annexe nationale (le poteau-poutre avec γ_M0 = 1,05 ≠ γ_M1 : {len(ac)}, le γ de n_pl) : une ligne près "
    "des champs γ_M (« γ_M de n_pl quand γ_M0 ≠ γ_M1 : γ_M1, à confirmer. »), qui ouvre une fenêtre (ac) avec leur valeur et leur source",
    len(ac) == 1 and "γ M de n pl quand γ M0 ≠ γ M1 : γ M1 , à confirmer." in texte(rpp["donnees"]) and 'data-calcul="ac"' in rpp["donnees"]
    and 'id="ck-ac"' in rpp["volets"] and all(p["source"] in texte(rpp["volets"]) for p in ac))
nu = REPONSES["une barre nue que la méthode historique refuse (IPE 300, 6 m sans maintien)"]
chk("une barre que la méthode ELS refuse (139 % — lot B : le déversement des classeurs ; la variante du fascicule donnait 179 %) et que "
    "l'Eurocode refuse aussi (113 % — lot C : le poids propre au centre de gravité, 114 % avant ; lot A : avec le couple (0,2 ; 1,0) de "
    "l'ANB:2018, elle passait à 92 %) : la note est NON CONFORME, les "
    "deux verdicts se lisent, deux blocs rouges",
    nu["resume"]["niveau"] == "r" and nu["resume"]["statut"] == "Non conforme" and nu["resume"]["taux"] == 139
    and "η 139 % Méthode ELS (NBN B 51-001)" in texte(nu["resultat"]) and "η 113 % Eurocode 3" in texte(nu["resultat"])
    and nu["resultat"].count('class="et-bloc nok"') == 2, nu["resume"])

print("=== 3. Les saisies")
POSES = [("nom", "Panne 3", lambda e: e["nom"] == "Panne 3"), ("travees/0", "7,25", lambda e: e["travees"] == [7.25]),
         ("poids_propre", False, lambda e: e["poids_propre"] is False), ("nuance", "S355", lambda e: e["nuance"] == "S355"),
         ("niveau", "bas", lambda e: e["niveau"] == "bas"), ("cas/0/nom", "Permanent", lambda e: e["cas"][0]["nom"] == "Permanent"),
         ("cas/1/nature", "neige", lambda e: e["cas"][1]["nature"] == "neige"), ("cas/0/N", "120", lambda e: e["cas"][0]["N"] == 120.0),
         ("cas/0/charges/0/v", "7,5", lambda e: e["cas"][0]["charges"][0]["v"] == 7.5), ("cas/0/charges/0/a", "1", lambda e: e["cas"][0]["charges"][0]["a"] == 1.0),
         ("cas/0/charges/0/b", "4", lambda e: e["cas"][0]["charges"][0]["b"] == 4.0), ("maintiens/0/x", "2", lambda e: e["maintiens"][0]["x"] == 2.0),
         ("maintiens/0/genre", "haut", lambda e: e["maintiens"][0]["genre"] == "haut"), ("flambement/y/L", "4,5", lambda e: e["flambement"]["y"]["L"] == 4.5),
         # (lot C, C0d : « methodes » choisit ce que la NOTE imprime — etat["impression"], contrôlé plus bas ; C0e : la tolérance de 5 % est
         #  retirée, son chemin aussi — le harnais suit)
         ("flambement/z/u", "0,7", lambda e: e["flambement"]["z"]["u"] == 0.7), ("historique/s", "1,33", lambda e: e["historique"]["s"] == 1.33),
         ("historique/C", "norme", lambda e: e["historique"]["C"] == "norme"), ("historique/deversement", False, lambda e: e["historique"]["deversement"] is False),
         ("parametres/gamma_M1", "1,1", lambda e: e["parametres"] == {"gamma_M1": 1.1}),
         # (la borne des noms du site : 60 caractères pour un élément — le harnais en saisit 58 — ; 12 pour un cas, qui s'écrit sur le dessin)
         ("nom", "N" * 60, lambda e: e["nom"] == "N" * 60), ("cas/1/nom", "Exploitation", lambda e: e["cas"][1]["nom"] == "Exploitation"),
         # (lot 4 : ce que l'âme reçoit de ses appuis)
         ("appuis/ss", "100", lambda e: e["appuis"] == {"mode": "poses", "ss": 100.0, "c": None, "detail": None}),
         ("appuis/c", "50", lambda e: e["appuis"] == {"mode": "poses", "ss": None, "c": 50.0, "detail": None}),
         ("appuis/mode", "ame", lambda e: e["appuis"]["mode"] == "ame")]
faux = []
for chemin, valeur, juste in POSES:
    out = req(action="modifier", etat=copy.deepcopy(e0), chemin=chemin.split("/"), valeur=valeur)
    if not out["ok"] or out["message"] or not juste(out["etat"]["elements"][0]):
        faux.append((chemin, valeur, out.get("message") or out.get("erreur")))
chk(f"chaque saisie va au bon endroit ({len(POSES)} champs : nom, portée, poids propre, acier, niveau, cas, charges, maintiens, flambement, "
    "options de la méthode historique, γ_M1, appuis — et un nom de 60 caractères, la borne du site)", not faux, faux[:4])
imp_ = {v_: req(action="modifier", etat=copy.deepcopy(e0), chemin=[ch_], valeur=v_) for ch_, v_ in (("impression", "ec3"), ("methodes", "historique"))}
chk("(lot C, C0d) le segment de la rubrique « Méthodes » choisit ce que la NOTE imprime (etat « impression », pour toutes ses barres ; « methodes », "
    "l'ancien chemin, y mène aussi) — la barre n'en change pas, ses deux méthodes restent calculées",
    all(o_["ok"] and o_["message"] is None and o_["etat"]["impression"] == v_ and o_["etat"]["elements"] == e0["elements"] for v_, o_ in imp_.items())
    and e0["impression"] == "deux", {v_: (o_.get("message"), o_["etat"].get("impression")) for v_, o_ in imp_.items()})
# les listes
out = saisir(e0, "cas/+", "1")
c3 = out["etat"]["elements"][0]["cas"]
chk("« + Cas de charge » : un cas variable de plus (Q2, exploitation A, une charge répartie sur toute la barre) ; « retirer » l'ôte",
    [c["nom"] for c in c3] == ["G", "Q", "Q2"] and c3[2]["nature"] == "exploitation_A"
    and c3[2]["charges"] == [{"type": "q", "v": 5.0, "a": 0.0, "b": 6.0, "niveau": None}]          # (lot C, C0e : le niveau d'une charge, vide)
    and [c["nom"] for c in saisir(out["etat"], "cas/1/-", "1")["etat"]["elements"][0]["cas"]] == ["G", "Q2"], c3)
out = enchainer(e0, ("cas/0/charges/+", "P"), ("cas/0/charges/1/v", "12"), ("cas/0/charges/1/a", "2"))
ch = out["etat"]["elements"][0]["cas"][0]["charges"]
chk("« + Ponctuelle » : une charge de 10 kN à mi-portée, que l'on règle (12 kN à 2 m) ; son type se change (un couple au même endroit, puis une "
    "charge triangulaire sur toute la barre) ; sa longueur d'appui n'est pas saisie (None, jamais 0 — le harnais suit)",
    # (lot C, C0e — le harnais suit : le niveau d'une charge (vide : celui de la barre) et l'introduction d'une ponctuelle (« auto »))
    ch[1] == {"type": "P", "v": 12.0, "a": 2.0, "ss": None, "niveau": None, "introduction": "auto"}
    and saisir(out["etat"], "cas/0/charges/1/type", "C")["etat"]["elements"][0]["cas"][0]["charges"][1] == {"type": "C", "v": 12.0, "a": 2.0, "niveau": None}
    and saisir(out["etat"], "cas/0/charges/1/type", "tri")["etat"]["elements"][0]["cas"][0]["charges"][1]
    == {"type": "tri", "v": 12.0, "a": 0.0, "b": 6.0, "sens": "croissante", "niveau": None}
    and saisir(out["etat"], "cas/0/charges/0/-", "1")["etat"]["elements"][0]["cas"][0]["charges"]
    == [{"type": "P", "v": 12.0, "a": 2.0, "ss": None, "niveau": None, "introduction": "auto"}], ch)
out = saisir(e0, "maintiens/+", "1")
mt = out["etat"]["elements"][0]["maintiens"]
chk("« + Maintien » : au milieu du plus long intervalle libre (1,5 m, puis 4,5 m) ; ils restent triés ; « retirer » l'ôte",
    mt == [{"x": 1.5, "genre": "complet", "neglige": False}, {"x": 3.0, "genre": "complet", "neglige": False}]       # (lot C, C0e : « neglige »)
    and [m["x"] for m in saisir(out["etat"], "maintiens/+", "1")["etat"]["elements"][0]["maintiens"]] == [1.5, 3.0, 4.5]
    and saisir(out["etat"], "maintiens/0/-", "1")["etat"]["elements"][0]["maintiens"] == [{"x": 3.0, "genre": "complet", "neglige": False}], mt)
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
    court["cas"][0]["charges"][0]["b"] == 4.0 and court["maintiens"] == [{"x": 3.0, "genre": "complet", "neglige": False}], court["cas"][0])
auto = saisir(saisir(e0, "flambement/z/L", "2")["etat"], "flambement/z/L", "")
chk("une longueur de flambement VIDÉE redevient automatique : le champ est vide, son texte dit la longueur que le moteur prend (« auto · 3 »)",
    auto["etat"]["elements"][0]["flambement"]["z"]["L"] is None and 'placeholder="auto · 3"' in auto["donnees"]
    and 'placeholder="auto · 6"' in auto["donnees"], re.findall(r'placeholder="[^"]*"', auto["donnees"]))
# les refus
REFUS = [("travees/0", "-1", "portée"), ("travees/0", "abc", "n'est pas un nombre"), ("travees/0", "80", "entre 0,05 et 60"),
         ("cas/0/charges/0/b", "9", "cas « G »"), ("cas/1/nom", "G", "portent le nom"), ("cas/0/nature", "pluie", "nature inconnue"),
         ("cas/0/-", "1", None), ("nuance", "S999", "nuance inconnue"), ("maintiens/0/x", "7", "hors de la barre"), ("maintiens/0/genre", "milieu", "genre"),
         ("flambement/y/u", "0", "coefficient u"), ("flambement/y/L", "abc", "n'est pas un nombre"), ("historique/s", "5", "entre 1 et 3"),
         ("parametres/gamma_M0", "2", "entre 0,9 et 1,5"), ("methodes", "aci", "méthode inconnue"), ("impression", "aci", "méthode inconnue"),
         ("historique/tolerance", True, "Champ inconnu"), ("mode", "libre", "mode inconnu"),
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
# (lot C, C2, le harnais suit : la limite compte les NATURES d'actions variables, plus les cas — quatre natures différentes
#  sur les cas 1 à 4, une cinquième est refusée, une nature déjà présente passe)
quatre = enchainer(six["etat"], ("cas/2/nature", "neige"), ("cas/3/nature", "vent"), ("cas/4/nature", "temperature"))["etat"]
refus5, admis5 = saisir(quatre, "cas/5/nature", "exploitation_B"), saisir(quatre, "cas/5/nature", "exploitation_A")
chk("au plus quatre NATURES variables : le cinquième cas ajouté est permanent (G2) ; une cinquième nature est refusée, le moteur le dit ; "
    "une nature déjà présente passe",
    [c["nature"] != "permanent" for c in e5["elements"][0]["cas"]] == [False, True, True, True, True]
    and six["etat"]["elements"][0]["cas"][-1] == dict(nom="G2", nature="permanent", N=0.0, charges=[{"type": "q", "v": 5.0, "a": 0.0, "b": 6.0, "niveau": None}],
                                                      courte_duree=False, psi0=None, N_troncons=None)
    and "au plus 4 natures d'actions variables (la barre en a 5)" in (refus5["message"] or "") and refus5["etat"] == quatre
    and not admis5["message"] and admis5["etat"]["elements"][0]["cas"][5]["nature"] == "exploitation_A"
    and "au plus 6 cas de charge" in saisir(six["etat"], "cas/+", "1")["message"], [c["nom"] for c in six["etat"]["elements"][0]["cas"]])

print("=== 4. Les deux modes")
ef = saisir(e0, "mode", "efforts")
eb = ef["etat"]["elements"][0]
chk("passer en « efforts imposés » : les longueurs de flambement que la barre chargée prenait d'elle-même deviennent des données (6 m, 3 m) ; "
    "les charges restent dans la note", eb["mode"] == "efforts"
    and eb["flambement"] == {"y": {"L": 6.0, "u": 1.0, "courbe": None}, "z": {"L": 3.0, "u": 1.0, "courbe": None}}
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
# (lot C, C0d — le harnais suit : les deux méthodes sont toujours calculées ; la note seule n'imprime qu'une méthode)
chk("(lot C) efforts imposés, la note n'imprimant que l'Eurocode 3 : les DEUX colonnes d'efforts restent (service et ELU), les deux méthodes "
    "se calculent",
    'data-chemin="efforts/elu/My"' in une["donnees"] and 'data-chemin="efforts/service/My"' in une["donnees"]
    and list(moteur(une["etat"]["elements"][0])["methodes"]) == ["historique", "ec3"])

print("=== 5. Les méthodes")
# (lot C, C0d — le harnais suit : le moteur calcule TOUJOURS les deux méthodes et l'écran les montre toujours, deux verdicts l'un sous l'autre ;
# le choix ne décide que de ce que la NOTE imprime)
for choix in ("historique", "ec3"):
    out = saisir(e0, "methodes", choix)
    r = moteur(out["etat"]["elements"][0])
    chk(f"(lot C) la note n'imprime que « {choix} » : l'écran garde ses DEUX verdicts, les deux blocs, l'âme et la comparaison — rien de plus ; "
        f"le résumé et le calcul ne changent pas", list(r["methodes"]) == ["historique", "ec3"] and out["etat"]["impression"] == choix
        and out["resultat"].count('class="verdict"') == 2 and re.findall(r'<span class="n">([^<]+)</span>', out["resultat"]) == ["ELS", "Eurocode 3", "Âme et raidisseurs", "Comparaison"]
        and out["resume"] == neuf["resume"] and out["resultat"] == neuf["resultat"] and not lignes_du_moteur(out, r), texte(out["resultat"])[:200])
hi = saisir(e0, "methodes", "historique")
chk("(lot C) les options des DEUX méthodes s'affichent toujours (s, C, déversement ; γ) — la case « tolérance de 5 % » est partie (brief § 13.2, "
    "point 5 : non)",
    'data-chemin="historique/s"' in hi["donnees"] and 'data-chemin="parametres/gamma_M0"' in hi["donnees"]
    and 'data-chemin="historique/s"' in saisir(e0, "methodes", "ec3")["donnees"] and 'data-chemin="historique/tolerance"' not in hi["donnees"])
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
    and anc["etat"]["impression"] == "ec3"                       # (lot C, C0d : une note v1 imprime l'Eurocode 3 seul)
    and ea["efforts"]["elu"] == {"N": 600.0, "My": 60.0, "Mz": 0.0, "Vz": 40.0, "Vy": 0.0}
    and ea["flambement"] == {"y": {"L": 6.0, "u": 1.0, "courbe": None}, "z": {"L": 3.0, "u": 1.0, "courbe": None}}
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
# (intégration du lot C, vague 1 — le harnais suit la rencontre de C1 et C2 : le poids propre au centre de GRAVITÉ (C1), G à la
#  hauteur de la barre, font deux répartitions, donc deux sources de γ_G (C2) — 7 combinaisons au lieu de 3 ; section 22)
chk("la barre neuve (s_s et c non saisis) : le bloc « Âme et raidisseurs » (le troisième) dit « Bloqué » (lot C, C0c — le harnais suit : « À "
    "vérifier » avant ; une donnée manque, la décision est bloquée), la norme et les combinaisons ELU ; "
    "une ligne par appui — « décision impossible : s_s à saisir », avec le taux à la borne (157 %, en gris) ; l'avertissement en pied (le "
    "harnais suit : sans taux, « à fournir », à la 2e reprise du lot B — brief § 8.3, point 5)",
    bloc is not None and tb.startswith("Âme et raidisseurs Bloqué") and "NBN EN 1993-1-5 · combinaisons ELU (6.10) · 7 combinaisons" in tb
    and len(ra["combinaisons"]) == 7
    and all(f"{L['nom']} {pastille(L['taux'])} décision impossible : s s à saisir" in tb and round(L["taux"], 3) == 1.567 for L in ra["lieux"])
    and len(ra["lieux"]) == 2 and "Appui A, appui B : décision impossible : s s à saisir." in tb and ra["taux_max"] is None, tb[:400])
chk("… sous les verdicts, la ligne « Âme » dit ce qui manque, et ouvre tout le calcul du bloc",
    'class="pf-vame inc" data-calcul="s3"' in neuf["resultat"] and "Âme · Décision impossible : s s à saisir (appui A, appui B)" in texte(neuf["resultat"]))
rd = moteur(DECIDE["etat"]["elements"][0])["ame"]
bd = re.search(r'<section class="et-bloc ok" data-theme="3">(.*?)</section>', DECIDE["resultat"], re.S)
tbd = texte(bd.group(1)) if bd else ""
chk("… s_s = 100 mm et c = 0 saisis : « Sans raidisseur · 22 % » ; une ligne par appui — ce que l'âme compare (F_Ed 44,5 ≤ F_Rd 199,8 kN, R4 "
    "type (c), contact non établi), « sans raidisseur », le taux du moteur ; rien à poser : pas de ligne « Âme » sous les verdicts",
    bd is not None and tbd.startswith("Âme et raidisseurs Sans raidisseur · 22 %") and len(rd["lieux"]) == 2
    and all(f"{L['nom']} {pastille(L['taux'])} F Ed 44,5 ≤ F Rd 199,8 kN" in tbd for L in rd["lieux"])
    and tbd.count("ans raidisseur") == 1           # (lot C, C4 — le harnais suit : « sans raidisseur » une fois, dans l'en-tête)
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
    # (lot C, C4 — le harnais suit : la fenêtre d'un lieu ne montre que SES formules et messages ; les hypothèses du bloc sont dans
    # « Tout le calcul » du bloc, s3)
    if "décision impossible : s s à saisir" not in t or any(texte(PR.M.sub(h)) in t for h in ra["hypotheses"] if len(h) > 40):
        manques.append((L["nom"], "en-tête ou hypothèses"))
s3_ = re.search(r'<div class="flyout calcul" id="ck-s3">(.*?)</div></div></div>', neuf["volets"], re.S)
if not s3_ or any(texte(PR.M.sub(h)) not in texte(s3_.group(1)) for h in ra["hypotheses"]):
    manques.append(("s3", "les hypothèses du bloc"))
chk("chaque LIEU ouvre SA fenêtre de calcul (a-A-L00, a-A-L01) : toutes ses formules (à la borne s_s = c = 0) écrites par "
    "acier.barre.ecriture, les hypothèses du bloc, « décision impossible » ; « tout le calcul » du bloc (s3) porte ses vérifications et ses "
    "combinaisons",
    not manques and 'id="ck-s3"' in neuf["volets"] and 'data-calcul="s3"' in bloc.group(1), manques[:3])
v_el = next(v for v in ra["verifications"] if v["id"].endswith("-EL"))
chk("… une vérification donnée pour information au-delà de 1 (non couverte : le critère 6.2.1(5) à s_s = 0, 1,567) se dit « > », jamais "
    "« ≤ » (profil_mots.comparaison)",
    v_el["statut"] == "bloque" and round(v_el["taux"], 3) == 1.567 and texte(M.comparaison(v_el)) == "η = 1,567 > 1", texte(M.comparaison(v_el)))
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
    """Le texte de la fenêtre du premier appui, puis celui de « Tout le calcul » du bloc de l'âme (s3). (lot C, C4 — le harnais suit :
    la fenêtre d'un lieu ne redit plus les hypothèses du bloc — s_s, c — : elles sont dans « Tout le calcul », dites une fois)"""
    L_ = moteur(rep_["etat"]["elements"][0])["ame"]["lieux"][0]
    f_ = re.search(rf'<div class="flyout calcul" id="ck-{re.escape(PR.fenetre("ame", L_["base"]))}">(.*?)</div></div></div>', rep_["volets"], re.S)
    s3 = re.search(r'<div class="flyout calcul" id="ck-s3">(.*?)</div></div></div>', rep_["volets"], re.S)
    return (texte(f_.group(1)) if f_ else "") + " | " + (texte(s3.group(1)) if s3 else "")


c50 = saisir(c0["etat"], "appuis/c", "50")
chk("la longueur d'appui saisie seule (100 mm) : à la borne c = 0, « sans raidisseur (c = 0 : borne prudente) », 22 % (le harnais suit : "
    "« décision impossible : c à fournir » à la 2e reprise) ; avec c = 0 saisi : F_Rd = 199,8 kN (le cas R4 du brief, type (c) : le contact "
    "d'extrémité n'est pas établi, s_s = 0 dans (6.10) — lot A) — 22 %, sans borne ; avec contact, 248,9 kN pour information ; 50 mm de débord : "
    "le verdict reste 199,8 kN, l'information passe à 335,9 kN",
    "F Ed 44,5 ≤ F Rd 199,8 kN · c = 0 : borne prudente" in texte(ss100["resultat"]) and "Sans raidisseur · 22 %" in texte(ss100["resultat"])
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
    and "Âme et raidisseurs Non conforme · 107 %" in tv and voile["resume"]["niveau"] == "r" and "Eurocode 3 Conforme NBN" in tv, tv[:400])
chk("… le taux de la barre est alors celui de l'ÂME, qui ne passe pas (107 %, et non les 94 % de l'Eurocode) : le résumé de la note, la ligne "
    "« Profilé » ; une âme qui PASSE ne compte pas — un raidisseur à 98 % ne change pas le taux d'une barre à 74 %",
    voile["resume"]["taux"] == 107 and voile["resume"]["gouvernant"] == "B1 · IPE 750x134 · voilement de l'âme par cisaillement (âme)"
    and "Profilé IPE 750x134 Eurocode 3" in tv and PR.pire(moteur(voile["etat"]["elements"][0]))[0] == "ame"      # (C4 : la ligne sans taux)
    and PR.taux_barre(moteur(rep["etat"]["elements"][0])) == max(x["taux_max"] for x in moteur(rep["etat"]["elements"][0])["methodes"].values())
    and neuf["resume"]["taux"] == 88 and PR.pire(r0)[0] == "historique", (voile["resume"], tv[:300]))

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
    mixte["resume"]["niveau"] == "r" and mixte["resume"]["taux"] == 139 and mixte["resume"]["gouvernant"].startswith("Panne · IPE 300 · déversement")
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
# (lot C, C3 — le harnais suit C1b : les U et les cornières, en efforts imposés seulement, sont choisis dans une barre en efforts imposés)
e0_efforts = saisir(e0, "mode", "efforts")["etat"]
for fam in P.FAMILLES:
    nom = next(p["nom"] for p in CAT.profils() if p["famille"] == fam)
    etat_f = copy.deepcopy(e0_efforts if fam in P.FAMILLES_EFFORTS else e0)
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
    if fam in P.FAMILLES_DEUX and ((">d = " in v) != rond or (">h = " in v) == rond or (">b = " in v) == rond or (">t = " in v) != tube
                                   or (">t<tspan" in v) == tube):
        sans_dessin.append((fam, nom, "cotes", rond, tube))
chk(f"chacune des {len(P.FAMILLES)} familles du module a sa section dessinée et son résultat : un I ou un H avec h, b, t_w, t_f ; un tube carré ou "
    "rectangulaire avec h, b, t ; un tube rond avec d et t (il n'a ni h ni b) ; (lot C, C1b) UPN, UPE et L en efforts imposés",
    not sans_dessin and len(P.FAMILLES) == 20 and P.FAMILLES_EFFORTS == ("UPN", "UPE", "L"), sans_dessin[:3])
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
    "Méthode ELS (NBN B 51-001)" in tx and "Eurocode 3" in tx and ">139 %<" in tx and ">113 %<" in tx and "#DC2626" in tx, re.findall(r">(\d+ %)<", tx))
mechant = '<script>alert(1)</script>"\'&'
# (lot C, C2, le harnais suit : une charge répartie sur toute la barre a la répartition du poids propre — les deux font une
#  source, « G », et le nom saisi disparaîtrait des dessins ; une ponctuelle garde la sienne)
sale = req(action="calculer", etat=note(barre(nom=mechant[:40], cas=[{"nom": "<b>G</b>", "nature": "permanent", "N": 0.0,
                                                                         "charges": [{"type": "P", "v": 5.0, "a": 2.0}]}])))
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
         "« Hors domaine · 70 % » (lot B : f_y nominale 460 et déversement des classeurs), jamais « Conforme » ; son bloc est à vérifier (inc) ; "
         "la note « À vérifier » (v), au taux du moteur",
         hd.get("ok") and "ELS Hors domaine NBN" in t and "η 70 % Méthode ELS" in t and '<section class="et-bloc inc" data-theme="1">' in hd["resultat"]
         and hd["resume"]["niveau"] == "v" and hd["resume"]["statut"] == "À vérifier" and hd["resume"]["taux"] == 70
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
        ("le verdict de la méthode du bureau la nomme en entier : « Méthode ELS (NBN B 51-001) · déversement » (lot B : la ligne des classeurs)",
         f"η {pastille(th)} Méthode ELS (NBN B 51-001) · déversement σ M 84,2 ≤ σ adm,d 95,3 N/mm²" in texte(nv["resultat"]), texte(nv["resultat"])[:160]),
        ("… son bloc porte le titre court « ELS » (la place manque devant « Conforme · 74 % ») ; la comparaison, l'en-tête de colonne « ELS »",
         re.findall(r'<span class="n">([^<]+)</span>', nv["resultat"]) == ["ELS", "Eurocode 3", "Âme et raidisseurs", "Comparaison"]
         and '<div class="pf-cmp pf-cmp-h"><span></span><span>ELS</span><span>Eurocode 3</span><span>Écart</span></div>' in nv["resultat"],
         re.findall(r'<span class="n">([^<]+)</span>', nv["resultat"])),
        ("la rubrique « Méthodes » : le segment « ELS » (sa clé reste « historique »), le titre « Méthode ELS (NBN B 51-001) », le résumé "
         "« note : les deux · s 1,5 » (lot C, C3 — le harnais suit : ce que la note imprime, les deux méthodes étant toujours calculées)",
         re.search(r'data-chemin="impression" data-sorte="choix" data-valeur="historique"[^>]*>ELS</button>', nv["donnees"]) is not None
         and '<div class="grp">Méthode ELS (NBN B 51-001)</div>' in nv["donnees"] and "note : les deux · s 1,5" in texte(nv["donnees"])
         and 'data-chemin="historique/s"' in nv["donnees"] and nv["etat"]["elements"][0]["methodes"] == ["historique", "ec3"],
         re.findall(r'data-valeur="historique"[^>]*>[^<]*<', nv["donnees"])),
        ("le résumé de la note (l'accueil, la liste des notes) : « déversement (ELS) »", nv["resume"]["gouvernant"] == "B1 · IPE 300 · déversement (ELS)", nv["resume"]),
        ("« tout le calcul » de la méthode : « B1 · méthode ELS (NBN B 51-001) », sans répéter la norme dessous ; la comparaison : colonne « ELS », "
         "« La méthode ELS (NBN B 51-001) compare des contraintes de service »",
         "<b>B1 · méthode ELS (NBN B 51-001)</b><span>IPE 300 · S235 · 6,00\u00a0m · tout le calcul</span>" in nv["volets"]
         and '<th class="n">ELS</th><th class="n">Eurocode 3</th>' in nv["volets"]
         and "ELS : contraintes de service ≤ f y /s" in texte(nv["volets"]),          # (lot C, C4 : la note en une ligne)
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
        # (2e finition du lot B) un ancrage perdu ROUGIT la suite : la boucle qui l'appelle prend une exception pour un rouge — la
        # mutation y passerait pour rouge sans avoir rien muté
        chk(f"le texte à muter est dans {module.__name__}, une seule fois : « {ancien.strip()[:70]} »", False, src.count(ancien))
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
    # (lot C, C4 — le harnais suit : la note de la comparaison tient en une ligne, « ELS : … ; EC3 : … »)
    ("la note de la comparaison redit « La méthode historique »", PR, "NOTE_COMPARAISON", lambda vrai: vrai.replace("ELS :", "La méthode historique :")),
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
# (intégration du lot C, vague 1 — le harnais suit C1b : la barre lit f_y dans acier/barre/nuances.py, qui écrit le refus des
#  80 mm lui-même — muter acier/flambement/normes.limite_elastique ne l'atteint plus, la mutation n'y faisait plus rien rougir)
from acier.barre import nuances as NU  # noqa: E402
vrai_ = NU._lire


def _lire_point(lecteur, nuance, t, produit, norme_produit):
    if t > NO.T_MAX_TABLEAU:                             # l'ancien refus : le nombre écrit par Python (« 89.9 »)
        raise NU.Refus(f"épaisseur {t:g} mm > {NO.T_MAX_TABLEAU:g} mm : hors du domaine du module")
    return vrai_(lecteur, nuance, t, produit, norme_produit)


try:
    NU._lire = _lire_point
    rouges_ = [n_ for n_, ok_, _i in refus_80() if not ok_]
finally:
    NU._lire = vrai_
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
        ("le résumé de la rubrique Appuis dit un champ vide comme le champ l'affiche — « s_s borne 0 · c borne 0 » ; s_s saisie : « s_s 100 mm · "
         "c borne 0 » ; c = 0 SAISI se lit (« c 0 mm »), il n'est pas vide (le harnais suit, 2e finition : « à saisir »)",
         resume_ap(n_).startswith("Appuis posée · s s borne 0 · c borne 0") and resume_ap(s_).startswith("Appuis posée · s s 100 mm · c borne 0")
         and resume_ap(z_).startswith("Appuis posée · s s 100 mm · c 0 mm"), (resume_ap(n_), resume_ap(s_), resume_ap(z_))),
        ("la ligne d'un lieu impossible porte le taux à la borne (« Appui A 157 % décision impossible : s s à saisir ») ; à la borne qui "
         "passe, la ligne dit la borne (« sans raidisseur (c = 0 : borne prudente) ») ; c = 0 saisi : « sans raidisseur », sans borne",
         # (lot C, C4 — le harnais suit : « sans raidisseur » dit une fois, dans l'en-tête du bloc ; la ligne garde sa borne)
         "Appui A 157 % décision impossible : s s à saisir" in tn and "Appui A 22 % F Ed 44,5 ≤ F Rd 199,8 kN · c = 0 : borne prudente" in ts
         and "Appui A 22 % F Ed 44,5 ≤ F Rd 199,8 kN Appui B" in tz and "Sans raidisseur · 22 %" in tz, (tn[-300:], ts[-300:])),
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
    ("le résumé des Appuis dit « à saisir » un champ vide (le champ dit « borne 0 »)", PS, "_resume_appuis",
     lambda vrai: _source_mutee(PS, 'else f"s<sub>s</sub> {M.SS_VIDE}"', 'else "s<sub>s</sub> à saisir"')._resume_appuis),
    # (lot C, C4 — le harnais suit : quand l'en-tête dit « Sans raidisseur », la ligne écrit sa borne seule, sans decision_lieu — la
    #  mutation retire donc la borne du lieu lui-même)
    ("la ligne d'un lieu ne dit pas la borne", MO, "verifier",
     lambda vrai: (lambda el: (lambda r_: {**r_, "ame": {**r_["ame"], "lieux": [dict(L, borne=None) for L in r_["ame"]["lieux"]]}})(vrai(el)))),
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
    vb, vf = (req(action="calculer", etat=dict(note(haute), version=v_)) for v_ in (True, 2.0))
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
         n3.get("ok") and ap3 == {"mode": "poses", "ss": None, "c": 0.0, "detail": None} and n3["etat"]["version"] == 2 and n3["etat"].get(P.ZERO_SAISI) is True
         and n3["etat"] == n2["etat"], ap3),
        ("une note d'un AUTRE format (version 3) est refusée — jamais lue comme une autre (le site d'avant prenait toute version inconnue pour une "
         "version 1 : « efforts imposés »)",
         not v9.get("ok") and "format inconnu" in v9.get("erreur", ""), v9.get("erreur")),
        ("… de même une version qui n'est pas un ENTIER — true (que Python compte pour 1 : « efforts imposés ») ou 2.0 — : le site écrit l'entier 2",
         all(not x.get("ok") and "format inconnu" in x.get("erreur", "") for x in (vb, vf)), (vb.get("erreur"), vf.get("erreur"))),
        # (lot C, C3 — le harnais suit : les deux paragraphes qui redisaient AIDE_SS sous les Appuis et les Charges sont RETIRÉS ; le sens
        #  du champ vide reste dit, une fois, au survol de « s_s » et de « sur », et le champ vide écrit « borne 0 »)
        ("l'aide de la longueur d'appui (le survol de « s_s » aux Appuis et de « sur » aux Charges) dit le sens d'un champ VIDE — « calcul "
         "à la borne 0 (prudent) » — et plus jamais « 0 : le cas le plus défavorable » (faux au type (c)) ; plus de paragraphe qui le redit",
         AIDE_APPUIS not in don and "Vide : calcul à la borne 0 (prudent)." in don and "(vide : calcul à la borne 0, prudent)" not in ch
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
    ("une version true ou 2.0 prise pour 1 ou 2 (l'égalité de Python : true == 1)", P, "normaliser",
     lambda vrai: _source_mutee(P, "    if isinstance(version, bool) or not isinstance(version, int) or version not in (1, VERSION):",
                                "    if version not in (1, VERSION):").normaliser),
    ("une version inconnue lue comme une version 1", P, "normaliser",
     lambda vrai: _source_mutee(P, "    if isinstance(version, bool) or not isinstance(version, int) or version not in (1, VERSION):",
                                "    if False:").normaliser),
    ("l'aide de s_s redit « 0 : le cas le plus défavorable »", M, "AIDE_SS", lambda vrai: vrai.replace("Vide : calcul à la borne 0 (prudent).", "0 : le cas le plus défavorable.")),
    ("le survol de « sur » (une ponctuelle) redit « 0 : le cas le plus défavorable »", P, "S",
     lambda vrai: _source_mutee(vrai, '<span title="{C.esc(M.AIDE_SS)}">sur</span>', '<span title="0 : le cas le plus défavorable.">sur</span>')),
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
# les options de la méthode ELS ajoutées au lot B (B1 : C_d, variante, facteur 5,1, classe imposée) : le site d'avant ne les connaît pas et les
# laisse tomber à la relecture (intégration du lot B, défaut croisé B1 × B4 : ce § les croyait gardées) — sans enjeu, aucune note d'avant ne compte
# (décision du propriétaire, 17/11)
OPTIONS_ELS_LOT_B = ("Cd", "variante", "facteur_doc", "classe_d")
# (lot C, C0e) le contrat des entrées du lot C : des clés neuves (chacune son défaut), que le site d'avant ne connaît pas non plus — le harnais suit
CLES_LOT_C = ("prs", "norme_produit", "ouvrage", "combinaisons", "position", "angle", "points", "debords", "poids_propre_efforts")


def sans_options_els(el):
    """L'élément ``el`` sans les options ELS du lot B (OPTIONS_ELS_LOT_B) ni les clés du lot C (CLES_LOT_C, et celles des cas, des charges, des
    maintiens, des appuis, du flambement, de la tolérance retirée)."""
    cas = [{**{k: v for k, v in c.items() if k not in ("courte_duree", "psi0", "N_troncons")},
            "charges": [{k: v for k, v in ch.items() if k not in ("niveau", "introduction")} for ch in c.get("charges") or []]}
           for c in el.get("cas") or []]
    return {**{k: v for k, v in el.items() if k not in CLES_LOT_C}, "cas": cas,
            "historique": {k: v for k, v in (el.get("historique") or {}).items() if k not in OPTIONS_ELS_LOT_B + ("tolerance_retiree",)},
            "maintiens": [{k: v for k, v in m.items() if k != "neglige"} for m in el.get("maintiens") or []],
            "appuis": {k: v for k, v in (el.get("appuis") or {}).items() if k != "detail"},
            "flambement": {a: {k: v for k, v in f.items() if k != "courbe"} for a, f in (el.get("flambement") or {}).items()}}


def ancien_site():
    """Le dossier où le site d'avant est tiré de l'historique (git archive : api_web, acier, rdm, beton — ce que son profil.py importe), une
    fois ; None, et la raison, si l'historique ne le donne pas."""
    if "dossier" not in _ANCIEN:
        try:
            tar = subprocess.run(["git", "-C", RACINE, "archive", ANCIEN_SITE, "api_web", "acier", "rdm", "beton"], capture_output=True, check=True).stdout
            d = tempfile.mkdtemp(prefix="profil_766659c_")
            atexit.register(shutil.rmtree, d, True)           # (effacé à la fin du test)
            with tarfile.open(fileobj=io.BytesIO(tar)) as t:
                t.extractall(d, filter="data")
            _ANCIEN.update(dossier=d, raison=None)
        except (OSError, subprocess.CalledProcessError, tarfile.TarError) as e:
            _ANCIEN.update(dossier=None, raison=f"le commit {ANCIEN_SITE} n'est pas lisible (ce contrôle demande l'historique complet : "
                                                f"git fetch --unshallow) : {e}")
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
         "auto » ; à l'écran, ces champs acceptent d'être vidés (data-vide) et disent, vides, le sens du vide — « borne 0 », comme l'aide (le "
         "harnais suit, 2e finition : « à saisir ») ; un texte qui n'est pas un nombre reste refusé",
         (el_p["appuis"]["ss"], el_p["appuis"]["c"], el_p["cas"][0]["charges"][1]["ss"]) == (100.0, 0.0, 80.0)
         and (el_v["appuis"]["ss"], el_v["appuis"]["c"], el_v["cas"][0]["charges"][1]["ss"]) == (None, None, None) and rep_.get("ok")
         and sorted(champs) == sorted([("appuis/ss", "texte", "", 'data-vide="auto" placeholder="borne 0"', "borne 0"),
                                       ("appuis/c", "texte", "", 'data-vide="auto" placeholder="borne 0"', "borne 0"),
                                       ("cas/0/charges/1/ss", "texte", "", 'data-vide="auto" placeholder="borne 0"', "borne 0")])
         and abc.get("ok") and "n'est pas un nombre" in (abc.get("message") or "") and abc["etat"]["elements"][0]["appuis"]["c"] == 0.0,
         (el_v["appuis"], champs, abc.get("message"))),
        (f"le SITE D'AVANT ({ANCIEN_SITE}, son normaliser exécuté) relit une note NEUVE telle quelle quand tout est saisi — 0 saisi = 0, et aucun "
         "autre champ ne change, sauf les options ELS du lot B (C_d, variante, facteur 5,1, classe imposée), qu'il ne connaît pas et laisse "
         "tomber (intégration du lot B : le harnais suit) ; une donnée vide (null) : il la REFUSE, franchement (« nombre attendu »), jamais une "
         "autre lecture (une version 3 y devenait « efforts imposés », s_s et c à 0)",
         _ANCIEN.get("raison") is None and pl_.get("ok") and pl_["etat"]["version"] == 2 and el_a == sans_options_els({k: el_p[k] for k in el_a})
         and set(el_a) == set(el_p) - set(CLES_LOT_C)
         and set(el_p["historique"]) - set(el_a.get("historique") or {}) == set(OPTIONS_ELS_LOT_B) | {"tolerance_retiree"}
         and lus.get("neuve", {}).get("ok") is False and "nombre attendu" in lus["neuve"].get("erreur", "")
         and lus.get("videe", {}).get("ok") is False and "nombre attendu" in lus["videe"].get("erreur", ""),
         _ANCIEN.get("raison") or {k: (v.get("ok"), v.get("erreur")) for k, v in lus.items()}),
    ]


def credit_mutation(nom_, rouges_):
    """(intégration du lot B, point 29 a) Le verdict d'une mutation de ce § : (libellé, vert ?). Sans le site d'avant dans l'historique (un
    clone superficiel), le contrôle de relecture est rouge DE TOUTE FAÇON : chaque mutation « rougirait » et serait créditée — un vert
    silencieux. Elle est alors dite NON JOUÉE, en rouge, avec le remède."""
    if ancien_site() is None:
        return (f"mutation « {nom_} » : NON JOUÉE — {ANCIEN_SITE} absent de l'historique (git fetch --deepen=200 ou --unshallow)", False)
    return (f"mutation « {nom_} » : {len(rouges_)} contrôle(s) rougissent", bool(rouges_))


chk(f"le site d'avant ({ANCIEN_SITE}) se lit dans l'historique — sans lui, ce § ne prouve rien : rouge clair, jamais un vert silencieux",
    ancien_site() is not None, _ANCIEN.get("raison"))
for nom_, ok_, info_ in vider_et_relire():
    chk(nom_, ok_, info_)
# (point 29 a) l'absence du site d'avant, simulée : la relecture rougit et aucune mutation n'est créditée
_garde = dict(_ANCIEN)
_ANCIEN.clear()
_ANCIEN.update(dossier=None, raison="absence simulée")
try:
    simule_ = (credit_mutation("témoin", ["un rouge"]), [ok_ for n_, ok_, _i in vider_et_relire() if "SITE D'AVANT" in n_])
finally:
    _ANCIEN.clear()
    _ANCIEN.update(_garde)
    P._RESULTATS.clear()
chk(f"(point 29 a) {ANCIEN_SITE} absent (simulé) : le contrôle de relecture est rouge et une mutation qui « rougit » n'est PAS créditée — "
    "elle est dite non jouée", simule_[0][1] is False and "NON JOUÉE" in simule_[0][0] and simule_[1] == [False], simule_)
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
    chk(*credit_mutation(nom_, rouges_))
chk("témoin : sans mutation, les contrôles du champ vidé et de la relecture par le site d'avant sont verts", all(ok_ for _n, ok_, _i in vider_et_relire()))

print("=== 17. (2e finition du lot B) le maintien à l'écran : la ligne et la fenêtre sans le taux de leur pastille, la bande le lieu en tête — "
      "et leurs mutations")


def maintien_ecran():
    """Les contrôles du message du maintien à l'écran (une charge de 20 kN à mi-portée, posée sur la semelle haute, sans maintien ; appuis par
    l'âme : le seul lieu), rejoués sous chaque mutation : [(nom, vrai ?, info)]."""
    P._RESULTATS.clear()
    b_ = barre(maintiens=[], appuis={"mode": "ame", "ss": None, "c": None},
               cas=[{"nom": "G", "nature": "permanent", "N": 0.0, "charges": [{"type": "P", "v": 20.0, "a": 3.0, "ss": 100.0}]}])
    r_ = req(action="calculer", etat=note(b_))
    tr_, tv_ = texte(r_.get("resultat", "")), texte(r_.get("volets", ""))
    fait = "semelle haute sans maintien (EN 1993-1-5, 6.1(1))"
    b2_ = barre(maintiens=[], appuis={"mode": "ame", "ss": None, "c": None},
                cas=[{"nom": "G", "nature": "permanent", "N": 0.0, "charges": [{"type": "P", "v": 250.0, "a": 0.5, "ss": None}]}])
    tr2_ = texte(req(action="calculer", etat=note(b2_)).get("resultat", ""))
    return [
        ("la LIGNE du lieu dit le fait et le remède sans le taux, que sa pastille donne (« Charge à 3,00 m 24 % semelle haute sans maintien (…) ; "
         "ajouter un maintien (À VÉRIFIER) ») ; la BANDE, le lieu en tête (« Âme · Charge à 3,00 m : semelle haute … — 24 % si elle est tenue ; … »), "
         "jamais « …) (charge à 3,00 m) » ; l'avertissement, la phrase entière ; « si elle est tenue » deux fois (bande, avertissement), jamais "
         "dans les fenêtres (la -F, le titre du lieu : sans le taux)",
         f"Charge à 3,00 m 24 % {fait} ; ajouter un maintien (À VÉRIFIER)" in tr_
         and f"Âme · Charge à 3,00 m : {fait} — 24 % si elle est tenue ; ajouter un maintien (À VÉRIFIER)" in tr_
         and f"Charge à 3,00 m : {fait} — 24 % si elle est tenue ; ajouter un maintien (À VÉRIFIER)." in tr_
         and "(À VÉRIFIER) (charge" not in tr_ and tr_.count("si elle est tenue") == 2
         and f"x = 3,00 m · {fait} ; ajouter un maintien (À VÉRIFIER)" in tv_ and "si elle est tenue" not in tv_,
         re.findall(r".{40}sans maintien.{80}", tr_ + " | " + tv_)[:4]),
        # (intégration du lot B, point 29 c) s_s manque AUSSI (250 kN à 0,50 m, s_s non saisi, la borne ne passe pas) : la raison finit par une
        # parenthèse, le lieu passe en tête — sans « décision impossible : » derrière lui (deux « : » de suite)
        ("la BANDE d'un lieu en tête dont la décision est impossible (s_s à saisir, semelle sans maintien) : « Âme · Charge à 0,50 m : s s à "
         "saisir ; semelle haute sans maintien (…) », jamais « Charge à 0,50 m : décision impossible : … » (l'avertissement garde sa phrase "
         "entière)",
         f"Âme · Charge à 0,50 m : s s à saisir ; {fait}" in tr2_ and "Âme · Charge à 0,50 m : décision impossible" not in tr2_,
         re.findall(r"Âme · .{90}", tr2_)[:1]),
    ]


for nom_, ok_, info_ in maintien_ecran():
    chk(nom_, ok_, info_)
MUTATIONS_MAINTIEN = (
    ("la ligne d'un lieu redit le taux de sa pastille (la raison entière)", M, "decision_lieu", lambda vrai: (lambda L: vrai(dict(L, ligne=None)))),
    ("la bande met le lieu en queue, après une parenthèse (« …) (charge à 3,00 m) »)", M, "resume_ame",
     lambda vrai: _source_mutee(M, 'if t.endswith(")") else', 'if False else').resume_ame),
    ("(point 29 c) le lieu en tête suivi de « décision impossible : »", M, "resume_ame",
     lambda vrai: _source_mutee(M, "t = t[len(AMEB.DECISION_IMPOSSIBLE):] if t.startswith(AMEB.DECISION_IMPOSSIBLE) else t", "t = t").resume_ame),
)
for nom_, objet_, attr_, fabrique_ in MUTATIONS_MAINTIEN:
    if not hasattr(objet_, attr_):
        chk(f"mutation « {nom_} » : l'attribut {attr_} existe", False, objet_)
        continue
    vrai_ = getattr(objet_, attr_)
    try:
        setattr(objet_, attr_, fabrique_(vrai_))
        rouges_ = [n_ for n_, ok_, _i in maintien_ecran() if not ok_]
    except Exception as err_:  # noqa: BLE001 — une mutation qui fait planter l'écran est vue
        rouges_ = [f"exception : {err_}"]
    finally:
        setattr(objet_, attr_, vrai_)
        P._RESULTATS.clear()
    chk(f"mutation « {nom_} » : {len(rouges_)} contrôle(s) rougissent", bool(rouges_))
chk("témoin : sans mutation, les contrôles du maintien à l'écran sont verts", all(ok_ for _n, ok_, _i in maintien_ecran()))

print("=== 18. (intégration du lot B) la ligne « Âme » sous les verdicts écrit s_s en INDICE, comme les lignes du bloc — et sa mutation")


def ame_indice():
    """La ligne « Âme » de la barre neuve (s_s non saisie aux appuis), rejouée sous la mutation : [(nom, vrai ?, info)]."""
    P._RESULTATS.clear()
    r_ = req(action="calculer", etat=note())
    b_ = re.search(r'<button type="button" class="pf-vame[^"]*"[^>]*>(.*?)</button>', r_.get("resultat", ""), re.S)
    h_ = b_.group(1) if b_ else ""
    return [("la ligne « Âme » écrit « s<sub>s</sub> à saisir » — ce que l'œil lit, « s s », comme les lignes du bloc ; jamais « s_s » à "
             "l'écran (le harnais du navigateur l'a trouvé)",
             "s<sub>s</sub> à saisir" in h_ and "s_s" not in h_ and texte(h_) == "Âme · Décision impossible : s s à saisir (appui A, appui B)",
             h_[-160:])]


for nom_, ok_, info_ in ame_indice():
    chk(nom_, ok_, info_)
_vrai_ame = PR.verdict_ame
try:
    PR.verdict_ame = _source_mutee(PR, "{M.sub(texte)}</span></button>", "{C.esc(texte)}</span></button>").verdict_ame
    rouges_ = [n_ for n_, ok_, _i in ame_indice() if not ok_]
except Exception as err_:  # noqa: BLE001 — une mutation qui fait planter l'écran est vue
    rouges_ = [f"exception : {err_}"]
finally:
    PR.verdict_ame = _vrai_ame
    P._RESULTATS.clear()
chk(f"mutation « la ligne « Âme » échappée sans indice (C.esc) » : {len(rouges_)} contrôle(s) rougissent", bool(rouges_))
chk("témoin : sans mutation, la ligne « Âme » est en indice", all(ok_ for _n, ok_, _i in ame_indice()))

print("=== 19. (lot C, C0) un taux sous son statut, le statut « bloqué », ce que la note imprime, la clé « prs » — et leurs mutations")


def _console(v):
    """Une console de 3 m, N = 50 kN de compression, G et Q de ``v`` kN/m : l'Eurocode 3 ne couvre pas N + M en console (E-661)."""
    el = P.barre_neuve()
    el.update(modele="console", travees=[3.0], maintiens=[])
    el["cas"][0]["N"] = 50.0
    for c_ in el["cas"]:
        c_["charges"][0].update(b=3.0, v=v)
    return el


def _prs(t1, bords=None):
    """Une barre PRS (300 × t1 / 540 × 15 / 300 × 30, S355), bords des semelles non saisis par défaut."""
    el = P.barre_neuve()
    el.pop("profil")
    el.update(nuance="S355", prs={"b1": 300, "t1": t1, "hw": 540, "tw": 15, "b2": 300, "t2": 30, "a": 6, "bords": bords},
              maintiens=[{"x": x_, "genre": "complet"} for x_ in (1.5, 3.0, 4.5)])
    return P.normaliser(P.etat_de([el]))["elements"][0]


def lot_c0():
    """Les contrôles du socle du lot C sur l'écran, rejoués sous chaque mutation : [(nom, vrai ?, info)]."""
    P._RESULTATS.clear()
    out = []
    h = req(action="calculer", etat=note(_console(5.0)))["resultat"]
    puces = re.findall(r'<div class="verdict"[^>]*><span class="chip xl ([a-z]+)', h)
    out.append(("C0b — console 3 m, N = 50 kN : la puce du verdict Eurocode 3 n'est PAS verte (57 %, montré pour information sous « non "
                "couvert » : « v ») ; dessous, en une ligne, « + 1 non couvert : flexion composée » — une fois",
                puces == ["r", "v"] and texte(h).count("+ 1 non couvert : flexion composée") == 1, (puces, texte(h)[:300])))
    cmp_ = re.search(r'<div class="pf-cmp"><span>Flexion composée(?:<small[^>]*>[^<]*</small>)?</span>(.*?)</div>', h)     # (C4 : ce qui gouverne)
    out.append(("C0b — … la case Eurocode 3 de « Flexion composée » dans la comparaison (45 %, sous un non couvert) n'est pas verte non plus",
                cmp_ is not None and re.findall(r'class="chip ([a-z]+)"', cmp_.group(1)) == ["r", "v"], cmp_ and cmp_.group(1)))
    leg = req(action="calculer", etat=note(_console(1.0)))
    # (lot C, C4 — le harnais suit : la ligne « Profilé » n'écrit plus de taux — les verdicts le disent ; ils restent « à vérifier »)
    prof = re.search(r'<div class="els pf-profil">(.*?)</div></div>', leg["resultat"], re.S)
    puces_ = re.findall(r'<div class="verdict"[^>]*><span class="chip xl ([a-z]+)', leg["resultat"])
    out.append(("C0b — une console légère (rien ne refuse ; l'Eurocode 3 non couvert, l'ELS hors domaine) : aucune puce verte (les deux « à "
                "vérifier ») ; la ligne « Profilé » sans taux ; la note « à vérifier »",
                prof is not None and "%" not in texte(prof.group(1)) and puces_ == ["v", "v"] and leg["resume"]["niveau"] == "v", (puces_, prof and prof.group(1)[:200])))
    rp = MO.verifier(_prs(30))
    vp = PR.verdicts(rp)
    out.append(("C0c — un PRS aux bords non saisis : le verdict ELS dit « Bloqué » en un mot (puce « v », sans taux), sa raison en UNE ligne "
                "(« Bords des semelles à saisir… ») ; « + 1 bloqué : cisaillement » dessous ; l'en-tête du bloc dit « Bloqué »",
                rp["methodes"]["historique"]["statut"] == RE.BLOQUE
                and re.findall(r'class="chip xl ([a-z -]+)"[^>]*>([^<]*)', vp)[0] == ("v pf-mot", "Bloqué")
                and texte(vp).count("Bords des semelles à saisir (oxycoupés ou larges plats) : verdict bloqué.") == 1
                and "+ 1 bloqué : cisaillement" in texte(vp) and PR.etat_methode(rp["methodes"]["historique"]) == "Bloqué", texte(vp)[:300]))
    tb_ = re.search(r'<section class="et-bloc inc" data-theme="3">(.*?)</section>', req(action="nouveau")["resultat"], re.S)
    out.append(("C0c — la barre neuve (s_s non saisie : décision de l'âme impossible) : le bloc de l'âme dit « Bloqué »",
                tb_ is not None and texte(tb_.group(1)).startswith("Âme et raidisseurs Bloqué"), tb_ and texte(tb_.group(1))[:80]))
    # C0d — ce que la note imprime
    b_h = P.barre_neuve("B1")
    b_h["methodes"] = ["historique"]
    b_e = P.barre_neuve("B2")
    b_e["methodes"] = ["ec3"]
    v2_ = lambda *els: {"version": 2, P.ZERO_SAISI: True, "elements": list(els), "actif": 0}     # noqa: E731 — une note sans « impression »
    ded = (P.normaliser(v2_(b_h, copy.deepcopy(b_h)))["impression"], P.normaliser(v2_(b_h, b_e))["impression"],
           P.normaliser(v2_(P.barre_neuve()))["impression"], P.normaliser(P.etat_de([P.barre_neuve()], impression="historique"))["impression"])
    out.append(("C0d — une note sans « impression » (d'avant le lot C) : une seule et même méthode dans toutes ses barres → celle-là ; "
                "sinon les deux ; une note qui l'a la garde",
                ded == ("historique", "deux", "deux", "historique"), ded))
    try:
        P.normaliser(P.etat_de([P.barre_neuve()], impression="aci"))
        out.append(("C0d — une impression inconnue est refusée", False, "acceptée"))
    except ValueError as e_:
        out.append(("C0d — une impression inconnue est refusée", "impression de la note" in str(e_), str(e_)))
    imp = P.normaliser(v2_(b_h))
    out.append(("C0d — la note qui n'imprime que l'ELS : le moteur calcule quand même LES DEUX méthodes, l'écran montre deux verdicts",
                list(MO.verifier(imp["elements"][0])["methodes"]) == ["historique", "ec3"]
                and req(action="calculer", etat=imp)["resultat"].count('class="verdict"') == 2, imp["impression"]))
    # C0e — la clé « prs » est gardée et signe le résultat
    p1, p2 = _prs(30), _prs(25)
    out.append(("C0e — deux PRS de plats différents : deux signatures (la clé « prs » est dans COMMUNS), deux résultats",
                p1.get("prs") is not None and P._signature(p1) != P._signature(p2)
                and P.calculer(p1)["methodes"]["ec3"]["taux_max"] != P.calculer(p2)["methodes"]["ec3"]["taux_max"],
                (p1.get("profil"), p2.get("profil"))))
    return out


for nom_, ok_, info_ in lot_c0():
    chk(nom_, ok_, info_)


def _sans_prs(vrai):
    """La mutation « prs hors de COMMUNS » touche les deux tuples (COMMUNS et CLES), comme avant le lot C."""
    return tuple(k for k in vrai if k != "prs")


MUTATIONS_C0 = (
    ("la puce du verdict peinte au taux seul (C.niveau), sans son statut", P, "R",
     lambda vrai: _source_mutee(vrai, 'M.couleur_taux(x["statut"], t), xl=True)', 'C.niveau(t), xl=True)')),
    # (lot C, C4 — le harnais suit : la ligne « Profilé » n'a plus de taux ni de couleur ; la mutation qui la peignait au taux seul
    #  devient : la ligne « Profilé » redit le taux de la barre, en vert)
    ("la ligne « Profilé » redit le taux de la barre, en vert", P, "R",
     lambda vrai: _source_mutee(vrai, '<span>{C.esc(r["profil"])}</span>{C.icon("chevron-down")}</button></div></div>\')',
                                '<span>{C.esc(r["profil"])}</span>{C.icon("chevron-down")}</button><span class="pct p">{M.taux_txt(taux_barre(r))}</span></div></div>\')')),
    ("la case de la comparaison peinte au taux seul (niveau_taux)", P, "R",
     lambda vrai: _source_mutee(vrai, 'return M.couleur_taux(case["statut"], case["taux"]) if case else "c"',
                                'return C.niveau(case["taux"]) if case and case["taux"] is not None else "c"')),
    ("la ligne « + 1 non couvert » retirée du verdict", P, "R", lambda vrai: _source_mutee(vrai, "reste = non_juges(x, g)", 'reste = ""')),
    ("couleur_taux ne connaît plus les statuts à vérifier", M, "STATUTS_A_VERIFIER", lambda vrai: ()),
    ("le statut « bloqué » dit « À vérifier » (le mot du niveau)", PR, "mot_methode",
     lambda vrai: (lambda x: PR.mot_niveau(PR.niveau_methode(x)) if x["statut"] == RE.BLOQUE else vrai(x))),
    ("la raison d'un verdict bloqué remplacée par sa comparaison", PR, "verdicts",
     lambda vrai: _source_mutee(PR, 'if g["statut"] == RE.BLOQUE and g["messages"] else', 'if False else').verdicts),
    ("une note sans « impression » imprime toujours les deux", P, "_impression_deduite", lambda vrai: (lambda etat: "deux")),
    ("« impression » oubliée à la relecture (toujours déduite)", P, "normaliser",
     lambda vrai: _source_mutee(P, "impression = etat.get(IMPRESSION, _impression_deduite(etat))", "impression = _impression_deduite(etat)").normaliser),
    ("« prs » hors de COMMUNS (et de CLES)", P, "COMMUNS", _sans_prs),
)
for nom_, objet_, attr_, fabrique_ in MUTATIONS_C0:
    if not hasattr(objet_, attr_):
        chk(f"mutation « {nom_} » : l'attribut {attr_} existe", False, objet_)
        continue
    vrai_, cles_ = getattr(objet_, attr_), P.CLES
    try:
        setattr(objet_, attr_, fabrique_(vrai_))
        if attr_ == "COMMUNS":
            P.CLES = _sans_prs(cles_)
        rouges_ = [n_ for n_, ok_, _i in lot_c0() if not ok_]
    except Exception as err_:  # noqa: BLE001 — une mutation qui fait planter l'écran est vue
        rouges_ = [f"exception : {err_}"]
    finally:
        setattr(objet_, attr_, vrai_)
        P.CLES = cles_
        P._RESULTATS.clear()
    chk(f"mutation « {nom_} » : {len(rouges_)} contrôle(s) rougissent", bool(rouges_))
chk("témoin : sans mutation, les contrôles du socle du lot C sont verts", all(ok_ for _n, ok_, _i in lot_c0()))

print("=== 20. (lot C, reprise de C0) les longueurs copiées au passage en efforts imposés redeviennent automatiques au retour — et leurs mutations")


def longueurs_copiees():
    """[(nom, vrai ?, info)], rejoués sous chaque mutation. Mesuré avant (f012e40 et f9e26aa) : la barre neuve passée en efforts
    imposés, rendue à la barre chargée, puis console de 3 m sans maintien, gardait L_y = 6 m et L_z = 3 m comme des SAISIES
    (« auto · 3 » au champ) au lieu de 2,1 × 3 = 6,3 m automatiques — sans un mot. Sous N = 50 kN, E-Nb-z : IPE 300 8,5 % au lieu
    de 25,5 % ; IPE 330 (le geste « console » du harnais) 6,9 % au lieu de 19,8 %."""
    P._RESULTATS.clear()
    out = []
    e_ = enchainer(note(), ("mode", "efforts"), ("mode", "chargee"), ("modele", "console"), ("travees/0", "3"), ("maintiens/0/-", "1"))
    el_ = e_["etat"]["elements"][0]
    Ls_ = P.calculer(el_)["longueurs"]
    champ_ = re.search(r'<input[^>]*data-chemin="flambement/z/L"[^>]*>', e_["donnees"])
    out.append(("reprise C0 — efforts imposés, retour, console de 3 m : L_y et L_z AUTOMATIQUES (2,1 × 3 = 6,3 m) ; le champ L_z est vide et "
                "dit « auto · 6,3 » (ce que lit le geste « console » du harnais)",
                el_["flambement"]["y"]["L"] is None and el_["flambement"]["z"]["L"] is None and Ls_["z"]["auto"] and Ls_["y"]["auto"]
                and abs(Ls_["z"]["L"] - 6.3) < 1e-9 and abs(Ls_["y"]["L"] - 6.3) < 1e-9 and P.COPIEES not in el_
                and champ_ is not None and 'value=""' in champ_.group(0) and 'placeholder="auto · 6,3"' in champ_.group(0),
                (el_["flambement"], Ls_, champ_ and champ_.group(0))))
    e2_ = enchainer(note(), ("mode", "efforts"), ("flambement/z/L", "3"), ("mode", "chargee"), ("modele", "console"), ("travees/0", "3"))
    el2_ = e2_["etat"]["elements"][0]
    out.append(("reprise C0 — une longueur RETAPÉE en efforts imposés (L_z = 3 m, la valeur copiée) est une saisie : elle reste au retour ; "
                "l'autre redevient automatique",
                el2_["flambement"]["z"]["L"] == 3.0 and el2_["flambement"]["y"]["L"] is None, el2_["flambement"]))
    ef_ = saisir(note(), "mode", "efforts")["etat"]
    rel_ = P.normaliser(json.loads(json.dumps(ef_)))
    out.append(("reprise C0 — le souvenir des longueurs copiées suit la note (relue d'un fichier : {y: 6, z: 3}) ; une barre neuve ne le porte pas",
                rel_["elements"][0].get(P.COPIEES) == {"y": 6.0, "z": 3.0} and P.COPIEES not in P.nouvel_etat()["elements"][0],
                rel_["elements"][0].get(P.COPIEES)))
    autre_ = copy.deepcopy(ef_)
    autre_["elements"][0]["flambement"]["z"]["L"] = 5.0          # (un fichier dont la longueur n'est plus celle qui a été copiée)
    er_ = saisir(autre_, "mode", "chargee")["etat"]["elements"][0]
    out.append(("reprise C0 — une longueur qui n'est plus celle copiée (5 m au lieu de 3) n'est jamais rendue automatique",
                er_["flambement"]["z"]["L"] == 5.0 and er_["flambement"]["y"]["L"] is None, er_["flambement"]))
    return out


for nom_, ok_, info_ in longueurs_copiees():
    chk(nom_, ok_, info_)

MUTATIONS_COPIEES = (
    ("le retour en barre chargée garde les longueurs copiées (le défaut mesuré)", "_appliquer",
     '                if el["flambement"][a]["L"] == Lc:\n                    el["flambement"][a]["L"] = None',
     '                if False:\n                    el["flambement"][a]["L"] = None'),
    ("une longueur retapée en efforts imposés n'est plus retenue comme saisie", "_appliquer",
     "            el[COPIEES].pop(cles[1], None)\n", "            pass\n"),
    ("une longueur qui n'est plus la copie rendue automatique quand même", "_appliquer",
     '                if el["flambement"][a]["L"] == Lc:', "                if True:"),
    ("le souvenir des longueurs copiées perdu à la relecture", "_completer", "    if cop:\n        b[COPIEES] = cop", "    if False:\n        b[COPIEES] = cop"),
)
for nom_, attr_, ancien_, nouveau_ in MUTATIONS_COPIEES:
    vrai_ = getattr(P, attr_)
    try:
        setattr(P, attr_, getattr(_source_mutee(P, ancien_, nouveau_), attr_))
        rouges_ = [n_ for n_, ok_, _i in longueurs_copiees() if not ok_]
    except Exception as err_:  # noqa: BLE001 — une mutation qui fait planter l'écran est vue
        rouges_ = [f"exception : {err_}"]
    finally:
        setattr(P, attr_, vrai_)
        P._RESULTATS.clear()
    chk(f"mutation « {nom_} » : {len(rouges_)} contrôle(s) rougissent", bool(rouges_), rouges_)
chk("témoin : sans mutation, les contrôles des longueurs copiées sont verts", all(ok_ for _n, ok_, _i in longueurs_copiees()))

print("=== 21. (lot C, reprise de C0 ; C6) une barre PRS : l'écran et le sélecteur la montrent, la note la refuse clairement (C5), jamais un KeyError")


def note_prs():
    """[(nom, vrai ?, info)], rejoués sous chaque mutation. Mesuré avant (f9e26aa) : « prs » gardée par COMMUNS (C0e), la barre se
    relit et se calcule, puis l'écran plantait (fiche_profil : KeyError « PRS 600x300x15x30 »), le sélecteur aussi, et la note
    refusait par un message obscur (« …, 'PRS 600x300x15x30' »). (lot C, C6 — le harnais suit : l'écran montre le PRS, son croquis,
    le sélecteur sa bibliothèque ; tests/test_api_web_profil_prs.py le détaille) : la NOTE, seule, la refuse en une ligne qui nomme le
    PRS, en attendant C5."""
    P._RESULTATS.clear()
    out = []
    e_ = P.etat_de([_prs(30)])
    nom_ = e_["elements"][0]["profil"]
    mot_ = P.refus_prs(nom_)
    rc_, rs_ = req(action="calculer", etat=e_), req(action="selecteur", etat=e_, role=P.ROLE)
    rn_ = req(action="telecharger", etat=e_, quoi="note_pdf")
    out.append(("reprise C0, C6 — une barre PRS (relue, calculée : ELS « bloqué ») : l'écran la montre (sa vue Section est le croquis du PRS), le "
                "sélecteur ouvre la bibliothèque des PRS ; la note la REFUSE en une ligne qui nomme le PRS — aucun KeyError",
                P.calculer(e_["elements"][0])["statut"] == RE.BLOQUE and len(mot_) <= 120 and nom_.startswith("PRS ")
                and rc_.get("ok") is True and any(v_["cle"] == "section" and 'data-repere="C"' in v_["svg"] for v_ in rc_["vues"])
                and rs_.get("ok") is True and "PRS" in rs_["html"]
                and rn_.get("ok") is False and mot_ in str(rn_.get("erreur")) and "KeyError" not in str(rn_.get("erreur")),
                [str(x_.get("erreur"))[:140] for x_ in (rc_, rs_, rn_)]))
    deux_ = P.etat_de([P.barre_neuve("B1"), _prs(30)], actif=0)
    rd_ = req(action="calculer", etat=deux_)
    out.append(("reprise C0 — la barre ouverte est un profilé du catalogue, une autre de la note un PRS : l'écran se rend",
                rd_.get("ok") is True and texte(rd_["resultat"]).count("Méthode ELS") >= 1, str(rd_.get("erreur"))[:200]))
    return out


for nom_, ok_, info_ in note_prs():
    chk(nom_, ok_, info_)

MUTATIONS_PRS = (
    ("fiche_profil lit le catalogue sans garde (le KeyError d'avant)", "fiche_profil",
     "    rec = CAT.par_nom().get(nom)\n    if rec is None:\n        raise ValueError(refus_prs(nom))\n", "    rec = CAT.par_nom()[nom]\n"),
    ("le sélecteur ouvre la famille d'un PRS sans garde", "selecteur",
     "    if el.get(\"prs\") is not None:          # (lot C, C6)", "    if False:          # (lot C, C6)"),
)
for nom_, attr_, ancien_, nouveau_ in MUTATIONS_PRS:
    vrai_ = getattr(P, attr_)
    try:
        setattr(P, attr_, getattr(_source_mutee(P, ancien_, nouveau_), attr_))
        rouges_ = [n_ for n_, ok_, _i in note_prs() if not ok_]
    except Exception as err_:  # noqa: BLE001 — une mutation qui fait planter l'écran est vue
        rouges_ = [f"exception : {err_}"]
    finally:
        setattr(P, attr_, vrai_)
        P._RESULTATS.clear()
    chk(f"mutation « {nom_} » : {len(rouges_)} contrôle(s) rougissent", bool(rouges_), rouges_)
chk("témoin : sans mutation, les contrôles de la note PRS sont verts", all(ok_ for _n, ok_, _i in note_prs()))

print("=== 22. (intégration du lot C, vague 1) le poids propre (C1) et les sources de γ_G (C2) : les combinaisons de l'âme — et sa mutation")
from acier.barre import combinaisons as CO  # noqa: E402


def combinaisons_neuve():
    """La barre neuve : le poids propre au centre de GRAVITÉ (C1), G (5 kN/m sur toute la barre) à la hauteur de la barre — deux
    répartitions, deux sources de γ_G (C2). 2 × 2 arrangements des permanents, Q présent ou non, moins « G » seul (proportionnel
    à « 1,35·G ») : 7 combinaisons, que le bloc de l'âme compte (rejoué sous mutation)."""
    P._RESULTATS.clear()
    r_ = req(action="nouveau")
    b_ = re.search(r'<section class="et-bloc inc" data-theme="3">(.*?)</section>', r_["resultat"], re.S)
    t_ = texte(b_.group(1)) if b_ else ""
    noms_ = [c_["nom"] for c_ in moteur(r_["etat"]["elements"][0])["ame"]["combinaisons"]]
    return [("barre neuve : le poids propre et G, à deux hauteurs, sont deux sources de γ_G — 7 combinaisons ELU au bloc de l'âme, dont "
             "« Poids propre + 1,35·G + 1,5·Q » et « 1,35·Poids propre + G + 1,5·Q »",
             "combinaisons ELU (6.10) · 7 combinaisons" in t_ and len(noms_) == 7 and "Poids propre + 1,35·G + 1,5·Q" in noms_
             and "1,35·Poids propre + G + 1,5·Q" in noms_, (re.findall(r"· \d+ combinaisons?", t_), noms_))]


for nom_, ok_, info_ in combinaisons_neuve():
    chk(nom_, ok_, info_)
vrai_ = CO._repartition
try:
    CO._repartition = _source_mutee(CO, '                 ch.get("niveau"), ch.get("introduction")',
                                    '                 None, ch.get("introduction")')._repartition
    rouges_ = [n_ for n_, ok_, _i in combinaisons_neuve() if not ok_]
except Exception as err_:  # noqa: BLE001 — une mutation qui fait planter l'écran est vue
    rouges_ = [f"exception : {err_}"]
finally:
    CO._repartition = vrai_
    P._RESULTATS.clear()
chk(f"mutation « le niveau d'une charge hors de sa répartition (le poids propre réuni à G) » : {len(rouges_)} contrôle(s) rougissent",
    bool(rouges_), rouges_)
chk("témoin : sans mutation, le contrôle des combinaisons de la barre neuve est vert", all(ok_ for _n, ok_, _i in combinaisons_neuve()))

print("=== 23. (lot C, C3) la saisie : chaque champ neuf passe par profil._appliquer et se relit ; les anciennes notes ; les retraits ; "
      "les champs conditionnels ; n maintiens réguliers ; le sélecteur — et leurs mutations")
from acier.barre import historique_deversement as HD3  # noqa: E402
from api_web import profil_saisie as S3  # noqa: E402


def _relu(etat):
    """L'état tel qu'un fichier le rend (JSON), relu par profil.normaliser."""
    return P.normaliser(json.loads(json.dumps(etat)))


def _jouer(etat, gestes):
    """Les saisies l'une après l'autre (profil.modifier, comme l'écran) ; [(chemin, mot)] des saisies refusées ou inconnues."""
    refus = []
    for chemin, valeur in gestes:
        mot = P.modifier(etat, chemin.split("/"), valeur)
        if mot is not None and ("refusée" in str(mot) or "inconnu" in str(mot)):
            refus.append((chemin, str(mot)))
    return refus


# (chemin, valeur tapée, ce qu'on relit dans la barre relue, l'attendu) — l'un après l'autre, sur la même note
GESTES_C3 = [
    ("historique/Cd", "1,3", lambda b: b["historique"]["Cd"], 1.3),
    ("historique/variante", "fascicule", lambda b: b["historique"]["variante"], "fascicule"),
    ("historique/variante", "classeurs", lambda b: b["historique"]["variante"], "classeurs"),
    ("historique/facteur_doc", "true", lambda b: b["historique"]["facteur_doc"], True),
    ("historique/classe_d", "c", lambda b: b["historique"]["classe_d"], "c"),
    ("historique/classe_d", "", lambda b: b["historique"]["classe_d"], None),
    ("ouvrage", "autre", lambda b: b["ouvrage"], "autre"),
    ("combinaisons/expression", "6.10ab", lambda b: b["combinaisons"]["expression"], "6.10ab"),
    ("combinaisons/geotechnique", "true", lambda b: b["combinaisons"]["geotechnique"], True),
    ("combinaisons/regle_courte_duree", "false", lambda b: b["combinaisons"]["regle_courte_duree"], False),
    ("cas/1/courte_duree", "true", lambda b: b["cas"][1]["courte_duree"], True),
    ("cas/1/nature", "execution", lambda b: (b["cas"][1]["nature"], b["cas"][1]["psi0"]), ("execution", None)),
    ("cas/1/psi0", "0,8", lambda b: b["cas"][1]["psi0"], 0.8),
    ("cas/1/psi0", "", lambda b: b["cas"][1]["psi0"], None),
    ("cas/1/nature", "exploitation_A", lambda b: b["cas"][1]["nature"], "exploitation_A"),
    ("cas/0/charges/0/niveau", "bas", lambda b: b["cas"][0]["charges"][0]["niveau"], "bas"),
    ("cas/0/charges/0/niveau", "haut", lambda b: b["cas"][0]["charges"][0]["niveau"], None),       # = celui de la barre : None
    ("cas/0/N_troncons", "troncons", lambda b: b["cas"][0]["N_troncons"], [{"x2": 3.0, "N": 0.0}, {"x2": 6.0, "N": 0.0}]),
    ("cas/0/N_troncons/0/N", "30", lambda b: b["cas"][0]["N_troncons"][0]["N"], 30.0),
    ("cas/0/N_troncons/0/x2", "2", lambda b: b["cas"][0]["N_troncons"][0]["x2"], 2.0),
    ("cas/0/N_troncons/+", "1", lambda b: [t["x2"] for t in b["cas"][0]["N_troncons"]], [2.0, 4.0, 6.0]),
    ("cas/0/N_troncons/2/-", "1", lambda b: [t["x2"] for t in b["cas"][0]["N_troncons"]], [2.0, 6.0]),
    ("cas/0/N_troncons", "constant", lambda b: b["cas"][0]["N_troncons"], None),
    ("cas/0/charges/+", "P", lambda b: b["cas"][0]["charges"][1]["introduction"], "auto"),
    ("cas/0/charges/1/introduction", "c", lambda b: b["cas"][0]["charges"][1]["introduction"], "c"),
    ("cas/0/charges/1/contact", "non", lambda b: b["cas"][0]["charges"][1]["contact"], "non"),
    ("cas/0/charges/1/-", "1", lambda b: len(b["cas"][0]["charges"]), 1),
    ("maintiens/0/neglige", "true", lambda b: b["maintiens"][0]["neglige"], True),
    ("position", "verticale", lambda b: b["position"], "verticale"),
    ("position", "inclinee", lambda b: (b["position"], b["angle"]), ("inclinee", M.ANGLE_DEFAUT)),
    ("angle", "30", lambda b: b["angle"], 30.0),
    ("position", "horizontale", lambda b: b["position"], "horizontale"),
    ("points/+", "1", lambda b: b["points"], [3.0]),
    ("points/0", "2", lambda b: b["points"], [2.0]),
    ("points/0/-", "1", lambda b: b["points"], []),
    ("nuance", "S460", lambda b: b["nuance"], "S460"),
    ("norme_produit", "M", lambda b: b["norme_produit"], "M"),
    ("nuance", "S355", lambda b: (b["nuance"], b["norme_produit"]), ("S355", None)),
    ("debords/1", "1,2", lambda b: b["debords"], [0.0, 1.2]),
    ("debords/1", "", lambda b: b["debords"], [0.0, 0.0]),
    ("appuis/detail", "un_par_un", lambda b: [d["torsion"] for d in b["appuis"]["detail"]], ["fourche", "fourche"]),
    ("appuis/detail/0/torsion", "encastre", lambda b: b["appuis"]["detail"][0]["torsion"], "encastre"),
    ("appuis/detail/1/ss", "120", lambda b: b["appuis"]["detail"][1]["ss"], 120.0),
    ("appuis/detail/1/ss", "", lambda b: b["appuis"]["detail"][1]["ss"], None),
    ("appuis/detail/1/contact", "oui", lambda b: b["appuis"]["detail"][1]["contact"], "oui"),
    ("appuis/detail", "identiques", lambda b: b["appuis"]["detail"], None),
    ("appuis/contact", "non", lambda b: b["appuis"]["contact"], "non"),
    ("mode", "efforts", lambda b: b["mode"], "efforts"),
    ("poids_propre_efforts/N", "true", lambda b: b["poids_propre_efforts"], {"N": True, "M": False}),
    ("poids_propre_efforts/M", "true", lambda b: b["poids_propre_efforts"], {"N": True, "M": True}),
    ("flambement/y/courbe", "b", lambda b: b["flambement"]["y"]["courbe"], "b"),
    ("flambement/y/courbe", "", lambda b: b["flambement"]["y"]["courbe"], None),
    ("mode", "chargee", lambda b: b["mode"], "chargee"),
]


def champs_c3():
    """[(nom, vrai ?, info)] : chaque champ neuf écrit par profil._appliquer, relu par profil.normaliser (un fichier), ce qu'on a tapé."""
    P._RESULTATS.clear()
    etat = P.nouvel_etat()
    faux, refus = [], []
    for chemin, valeur, lire, attendu in GESTES_C3:
        refus += _jouer(etat, [(chemin, valeur)])
        try:
            relu = _relu(etat)["elements"][0]
            vu = lire(relu)
        except Exception as err_:  # noqa: BLE001 — une relecture qui lève est un rouge
            vu = f"exception : {err_}"
        if vu != attendu:
            faux.append((chemin, valeur, vu, attendu))
    # un s_s d'appui saisi puis VIDÉ (appuis un par un) : None, l'âme reprend la borne 0 — « Bloqué » sur la barre neuve (157 % à la borne)
    ame_ = P.nouvel_etat()
    refus += _jouer(ame_, [("appuis/detail", "un_par_un"), ("appuis/detail/0/ss", "100"), ("appuis/detail/1/ss", "100")])
    plein_ = P.calculer(_relu(ame_)["elements"][0])["ame"]["statut"]
    refus += _jouer(ame_, [("appuis/detail/1/ss", "")])
    vide_ = _relu(ame_)["elements"][0]
    out_ame = (vide_["appuis"]["detail"][1]["ss"] is None and plein_ == RE.OK and P.calculer(vide_)["ame"]["statut"] == RE.BLOQUE,
               (plein_, vide_["appuis"]["detail"], P.calculer(vide_)["ame"]["statut"]))
    imp = P.nouvel_etat()
    mot = P.modifier(imp, ["impression"], "ec3")
    out = [(f"C3 — les {len(GESTES_C3)} saisies des champs neufs passent par profil._appliquer (aucun « champ inconnu », aucun refus) et se "
            "RELISENT dans la barre relue d'un fichier : C_d, variante, facteur 5,1, classe imposée, ouvrage, combinaisons, ξ, règle de courte "
            "durée, courte durée et ψ₀ d'un cas, niveau d'une charge (celui de la barre : None), N par tronçon, introduction et contact d'une "
            "ponctuelle, maintien négligé, position et angle, points vérifiés, nuance et norme produit, débords, appuis un par un, contact, "
            "poids propre et courbe des efforts imposés",
            not faux and not refus, (faux[:4], refus[:4]))]
    out.append(("C3 — appuis un par un : s_s des deux appuis à 100 mm, l'âme est conforme ; celui de B VIDÉ redevient None (non saisi) et l'âme, "
                "reprise à la borne 0, est « Bloqué »", *out_ame))
    out.append(("C3 — « Note (toutes les barres) » écrit l'impression de la NOTE ; normaliser la garde",
                mot is None and _relu(imp)["impression"] == "ec3" and imp["elements"][0] == P.nouvel_etat()["elements"][0], mot))
    return out


def anciennes_c3():
    """[(nom, vrai ?, info)] : une note d'avant le lot C (v1 ; v2 sans « zero_saisi » ; v2 neuve sans « impression ») se relit sans
    changer une saisie ; l'impression est déduite (C0)."""
    P._RESULTATS.clear()
    b_ = P.barre_neuve("B1")
    vieille = sans_options_els(b_)
    vieille["methodes"] = ["historique"]
    out = []
    for nom_, etat_ in (("v2 neuve sans « impression »", {"version": 2, P.ZERO_SAISI: True, "actif": 0, "elements": [copy.deepcopy(vieille)]}),
                        ("v2 du site d'avant le lot B", {"version": 2, "actif": 0, "elements": [copy.deepcopy(vieille)]})):
        n_ = _relu(etat_)
        el_ = n_["elements"][0]
        garde = {k: el_[k] for k in vieille if k not in ("cas", "historique", "maintiens", "appuis", "flambement")}
        out.append((f"C3 — une note {nom_} : relue, aucune saisie ne change (les clés neuves à leur défaut), l'impression DÉDUITE (« historique ») "
                    "— rien n'est réécrit dans les barres",
                    n_["impression"] == "historique" and garde == {k: vieille[k] for k in garde} and el_["methodes"] == ["historique"]
                    and sans_options_els(el_)["cas"] == vieille["cas"] and el_["position"] == "horizontale" and el_["points"] == []
                    and el_["debords"] == [0.0, 0.0] and all(m_["neglige"] is False for m_ in el_["maintiens"]) and _relu(n_) == n_,
                    {k: (el_[k], vieille[k]) for k in garde if el_[k] != vieille[k]}))
    n1 = _relu(P._ANCIENNE)
    out.append(("C3 — une note de la version 1 : relue en efforts imposés, impression « ec3 », la barre de _migrer (nuance S355 de la liste du "
                "module Barres, HEB 200)", n1["impression"] == "ec3" and n1["elements"][0]["nuance"] == "S355"
                and n1["elements"][0]["profil"] == "HEB 200" and n1["elements"][0]["mode"] == "efforts", n1["elements"][0]["profil"]))
    return out


# les textes RETIRÉS (766659c : profil_saisie.py l. 122-124, 144, 187-188, 191, 193, 224-225, 227, 249, 252, 260-263 ; l'aide de u)
RETIRES_C3 = ("1,33", "Tolérance", "Compte pour le déversement", "Aucun maintien : la barre", "Complet : déplacement latéral et torsion empêchés. Semelle",
              "Vides : la portée", "N positif en compression ; moments", "Platine d'about, cornières, gousset : la réaction entre",
              "La réaction appuie sur l'âme (EN 1993-1-5) : le module", "paramètre à confirmer", "paramètres à confirmer", "2 console",
              "0,7 encastré", "Déversement (semelle comprimée)", "(vide : calcul à la borne 0, prudent)")


def _donnees(el, impression="deux", r=None):
    return S3.formulaire(el, P.calculer(el) if r is None else r, impression)


def conditions_c3():
    """[(nom, vrai ?, info)] : chaque champ conditionnel n'apparaît que dans son cas ; les textes retirés sont absents partout."""
    P._RESULTATS.clear()
    out = []
    neuve = P.barre_neuve()
    d0 = _donnees(neuve)
    tous = [d0, _donnees(dict(POTEAU)), _donnees(dict(POTEAU, parametres={"gamma_M0": 1.05})),
            _donnees(barre(cas=[{"nom": "G", "nature": "permanent", "N": 0.0, "charges": [{"type": "P", "v": 5.0, "a": 3.0}]}]))]
    restes = [t_ for t_ in RETIRES_C3 for d_ in tous if t_ in html.unescape(d_)]
    out.append(("C3 — les textes inutiles ou faux sont RETIRÉS de la saisie : « 1,33 avec le vent » (sans source), la case « Tolérance de 5 % » "
                "(13.2 point 5), « k paramètres à confirmer », les aides qui redisaient une infobulle (Charges, Appuis, Maintiens, Efforts), l'aide "
                "de u fausse (« 2 console »)", not restes, restes))
    out.append(("C3 — l'aide de u : 13.3 (1 ; 0,80 ; 0,65 ; console : 2,1·L déjà proposé, u = 1) ; « Note (toutes les barres) » ; le champ L_z "
                "vide dit « auto · 3 » et son origine au survol",
                html.unescape(d0).count(M.AIDE_U) == 2 and "Note (toutes les barres)" in d0
                and 'placeholder="auto · 3"' in d0 and "Vide : le plus long tronçon entre sections tenues." in html.unescape(d0), ""))
    cond = {}
    # la règle de courte durée et la case « courte durée » : seulement si la barre porte de la neige ou du vent
    neige = barre(cas=neuve["cas"] + [{"nom": "S", "nature": "neige", "N": 0.0, "charges": [{"type": "q", "v": 2.0, "a": 0.0, "b": 6.0}]}])
    dn = _donnees(neige)
    cond["règle et case de courte durée : absentes sans neige ni vent, présentes avec la neige (la case sur Q, pas sur S ni G)"] = (
        "combinaisons/regle_courte_duree" not in d0 and "courte_duree" not in d0 and 'data-chemin="combinaisons/regle_courte_duree"' in dn
        and 'data-chemin="cas/1/courte_duree"' in dn and 'cas/2/courte_duree' not in dn and 'cas/0/courte_duree' not in dn)
    # le contact : seulement sur un lieu du type (c), que le moteur nomme
    larges = barre(appuis={"mode": "poses", "ss": 150.0, "c": 300.0, "detail": None})
    rl, rn = P.calculer(larges), P.calculer(neuve)
    cond["contact d'extrémité : présent quand le moteur dit un appui du type (c) (barre neuve, s_s et c à la borne 0), absent sinon (s_s 150, c 300 mm)"] = (
        S3.lieux_c(rn, "appui") and 'data-chemin="appuis/contact"' in d0 and not S3.lieux_c(rl, "appui")
        and 'data-chemin="appuis/contact"' not in _donnees(larges))
    # la classe imposée : seulement quand le moteur la dit « à fournir » (ou qu'elle est déjà imposée)
    faux_r = copy.deepcopy(rn)
    faux_r["methodes"]["historique"]["alertes"] = [{"code": "a_fournir", "niveau": "attention", "message": "classe à fournir"}]
    cond["classe de déversement imposée : absente hors « à fournir », présente quand le moteur la dit à fournir"] = (
        "historique/classe_d" not in d0 and 'data-chemin="historique/classe_d"' in _donnees(neuve, r=faux_r))
    tube = barre(profil=next(p_["nom"] for p_ in CAT.profils() if p_["famille"] == "RHS"))
    dt = _donnees(tube)
    cond["facteur 5,1 : seulement pour un profil fermé (RHS) ; C_d et la variante : seulement pour un I (C_d caché en variante fascicule)"] = (
        "historique/facteur_doc" not in d0 and 'data-chemin="historique/facteur_doc"' in dt and 'data-chemin="historique/Cd"' in d0
        and "historique/Cd" not in dt and "historique/variante" not in dt
        and "historique/Cd" not in _donnees(barre(historique=dict(neuve["historique"], variante="fascicule"))))
    cond["ψ₀ : seulement pour « exécution » et « machines »"] = (
        "/psi0" not in d0 and 'data-chemin="cas/1/psi0"' in _donnees(barre(cas=[neuve["cas"][0], dict(neuve["cas"][1], nature="machines")])))
    cond["ξ = 1 (géotechnique) : seulement en 6.10a/b"] = (
        "combinaisons/geotechnique" not in d0 and 'data-chemin="combinaisons/geotechnique"' in _donnees(barre(combinaisons={"expression": "6.10ab",
                                                                                                                       "geotechnique": False, "regle_courte_duree": True})))
    cond["norme produit : seulement pour S420 et S460"] = ("norme_produit" not in d0 and 'data-chemin="norme_produit"' in _donnees(barre(nuance="S420")))
    cond["angle : seulement pour une barre inclinée ; position : seulement avec le poids propre"] = (
        'data-chemin="angle"' not in d0 and 'data-chemin="angle"' in _donnees(barre(position="inclinee", angle=20.0))
        and 'data-chemin="position"' in d0 and 'data-chemin="position"' not in _donnees(barre(poids_propre=False)))
    cond["débords : seulement « deux appuis » et « continue »"] = (
        'data-chemin="debords/0"' in d0 and "debords/" not in _donnees(barre(modele="console", travees=[3.0], maintiens=[],
                                                                              cas=[{"nom": "G", "nature": "permanent", "N": 0.0, "charges": [{"type": "P", "v": 5.0, "a": 3.0}]}])))
    froid = next(p_["nom"] for p_ in CAT.profils() if p_["famille"] == "SHS-F" and (p_.get("t") or 0) > 6)
    cond["courbe par axe : en efforts imposés, seulement quand le tableau n'en donne pas (SHS-F de plus de 6 mm), absente pour un IPE"] = (
        "flambement/y/courbe" not in _donnees(dict(POTEAU)) and 'data-chemin="flambement/y/courbe"' in _donnees(dict(POTEAU, profil=froid)))
    cond["entrée dans l'âme : sur une ponctuelle posée sur une semelle, pas au centre de cisaillement"] = (
        'charges/1/introduction' in _donnees(barre(cas=[dict(neuve["cas"][0], charges=neuve["cas"][0]["charges"] + [{"type": "P", "v": 5.0, "a": 3.0}])]))
        and 'charges/1/introduction' not in _donnees(barre(niveau="centre", cas=[dict(neuve["cas"][0], charges=neuve["cas"][0]["charges"]
                                                                                        + [{"type": "P", "v": 5.0, "a": 3.0}])])))
    mauvais = [k_ for k_, v_ in cond.items() if not v_]
    out.append((f"C3 — chaque champ conditionnel n'apparaît que là où il compte ({len(cond)} cas) : " + " ; ".join(cond), not mauvais, mauvais))
    return out


def reguliers_c3():
    """[(nom, vrai ?, info)] : le geste « n maintiens réguliers entre a et b » (PRS-C, 28 m : 8 → 3,111 m ; 18 → 1,474 m)."""
    P._RESULTATS.clear()
    e_ = P.etat_de([barre(travees=[28.0], cas=[{"nom": "G", "nature": "permanent", "N": 0.0, "charges": [{"type": "q", "v": 5.0, "a": 0.0, "b": 28.0}]}])])
    r8 = _jouer(e_, [("maintiens/reguliers", "8")])
    x8 = [m_["x"] for m_ in _relu(e_)["elements"][0]["maintiens"]]
    r18 = _jouer(e_, [("maintiens/reguliers", "18")])
    x18 = [m_["x"] for m_ in _relu(e_)["elements"][0]["maintiens"]]
    r3 = _jouer(e_, [("maintiens/reguliers", "3 2 10")])
    x3 = [m_["x"] for m_ in _relu(e_)["elements"][0]["maintiens"]]
    mauvais = P.modifier(e_, ["maintiens", "reguliers"], "trois")
    pas = lambda xs: sorted({round(b_ - a_, 3) for a_, b_ in zip([0.0] + xs, xs + [28.0])})        # noqa: E731
    return [("C3 — « 8 » sur une barre de 28 m (PRS-C, 11.9) : 8 maintiens complets au pas de 3,111 m ; « 18 » : 18 au pas de 1,474 m (ceux qui y "
             "étaient remplacés) ; « 3 2 10 » : 2, 4, 6, 8 m remplacés de 2 à 10 m (3 maintiens à 4, 6 et 8 m), les autres gardés ; un texte "
             "illisible est refusé en une ligne",
             not (r8 or r18 or r3) and len(x8) == 8 and x8[0] == 3.111 and pas(x8) == [3.111, 3.112]
             and len(x18) == 18 and x18[0] == 1.474 and set(pas(x18)) <= {1.473, 1.474, 1.474}
             and [x for x in x3 if 2.0 <= x <= 10.0] == [4.0, 6.0, 8.0] and len(x3) == 18 - sum(1 for x in x18 if 2.0 < x < 10.0) + 3
             and "maintiens réguliers" in str(mauvais) and len(str(mauvais)) <= 140,
             (x8[:3], pas(x8), x18[:2], pas(x18), x3[:6], mauvais))]


def selecteur_c3():
    """[(nom, vrai ?, info)] : le ★ du sélecteur juge les DEUX méthodes (et l'âme) : le plus léger dont la barre est conforme partout."""
    P._RESULTATS.clear()
    P._CANDIDATS.clear()
    b_ = barre(appuis={"mode": "ame", "ss": None, "c": None, "detail": None})
    h_ = P.selecteur(P.etat_de([b_]), "profil")
    etoile = re.search(r'<button class="pk-row[^"]*\bopt\b[^"]*"[^>]*data-choix="([^"]+)"', h_)
    nom_ = etoile.group(1) if etoile else None
    legers = P._noms(["IPE"])
    avant = legers[:legers.index(nom_)] if nom_ in legers else []
    r_ = MO.verifier(dict(b_, profil=nom_)) if nom_ else None
    ec3_seul = [n_ for n_ in avant if MO.verifier(dict(b_, profil=n_))["methodes"]["ec3"]["statut"] == RE.OK]
    return [("C3 — le ★ du sélecteur juge les deux méthodes : le plus léger IPE dont l'ELS ET l'Eurocode 3 tiennent ; un plus léger que "
             "l'Eurocode 3 seul accepte n'a pas l'étoile",
             r_ is not None and r_["statut"] == RE.OK and all(r_["methodes"][k_]["statut"] == RE.OK for k_ in ("historique", "ec3")) and ec3_seul,
             (nom_, ec3_seul[:3]))]


PARTIES_C3 = {"champs": champs_c3, "anciennes": anciennes_c3, "conditions": conditions_c3, "reguliers": reguliers_c3, "selecteur": selecteur_c3}
for f_ in PARTIES_C3.values():
    for nom_, ok_, info_ in f_():
        chk(nom_, ok_, info_)

MUTATIONS_C3 = (     # (nom, partie rejouée, module, attribut, (ancien, nouveau) du texte du module)
    ("la case « négligé dans M_cr » ignorée", "champs", P, "_appliquer", ('                mts[j]["neglige"] = _oui(valeur)', "                pass")),
    ("le niveau d'une charge égal à celui de la barre gardé (il ne suit plus le défaut)", "champs", P, "_appliquer",
     ('chs[j]["niveau"] = None if valeur == el["niveau"] else str(valeur)', 'chs[j]["niveau"] = str(valeur)')),
    ("un s_s d'appui vidé lu comme un nombre obligatoire", "champs", P, "_appliquer",
     ('(_lire_vide(valeur, "longueur d\'appui" if cles[3] == "ss" else "débord") if cles[3] in ("ss", "c") else str(valeur))',
      '(_lire(valeur, "longueur d\'appui" if cles[3] == "ss" else "débord") if cles[3] in ("ss", "c") else str(valeur))')),
    ("la norme produit gardée sur une nuance qu'elle ne change pas", "champs", P, "_appliquer",
     ("        if valeur not in NU.AVEC_NORME:", "        if False:")),
    ("n maintiens réguliers au pas (b − a)/n", "reguliers", P, "maintiens_reguliers", ("(min(b, L) - a) / (n + 1)", "(min(b, L) - a) / n")),
    ("le ★ jugé par l'Eurocode 3 seul", "selecteur", P, "_candidat",
     ('            r = MO.verifier(dict(el, profil=nom))\n            c = (_taux(r), r["statut"])',
      '            r = MO.verifier(dict(el, profil=nom))\n            c = (_taux(r), r["methodes"]["ec3"]["statut"])')),
    ("le contact montré partout", "conditions", P, "S", ("        if contacts:", "        if True:")),
    ("la règle de courte durée montrée sans neige ni vent", "conditions", P, "S",
     ('        if any(c["nature"] in CO.COURTES for c in el["cas"]):', "        if True:")),
    ("la classe imposée montrée hors « à fournir »", "conditions", P, "S", ('    if _alerte(r, "a_fournir") or h.get("classe_d"):', "    if True:")),
    ("ψ₀ montré pour toute nature", "conditions", P, "S", ('        if c["nature"] in CO.PSI0_SAISI:', "        if True:")),
    ("« 1,33 avec le vent » rétabli", "conditions", P, "S", ("1,5 pour les charges de service.", "1,5 pour les charges de service ; 1,33 avec le vent.")),
    ("l'aide de u fausse rétablie", "conditions", M, "AIDE_U", None),
)
for nom_, partie_, objet_, attr_, texte_ in MUTATIONS_C3:
    vrai_, s3_vrai_ = getattr(objet_, attr_), S3
    try:
        if texte_ is None:
            setattr(objet_, attr_, "Coefficient de longueur de flambement : 1 articulé, 0,7 encastré – articulé, 0,5 bi-encastré, 2 console.")
        elif attr_ == "S":                      # la saisie mutée : celle que l'écran (P.S) et ces contrôles (S3) appellent
            S3 = _source_mutee(s3_vrai_, *texte_)
            P.S = S3
        else:
            setattr(objet_, attr_, getattr(_source_mutee(objet_, *texte_), attr_))
        rouges_ = [n_ for n_, ok_, _i in PARTIES_C3[partie_]() if not ok_]
    except Exception as err_:  # noqa: BLE001 — une mutation qui fait planter l'écran est vue
        rouges_ = [f"exception : {err_}"]
    finally:
        setattr(objet_, attr_, vrai_)
        S3 = s3_vrai_
        P._RESULTATS.clear()
        P._CANDIDATS.clear()
    chk(f"mutation « {nom_} » : {len(rouges_)} contrôle(s) rougissent", bool(rouges_), rouges_)
chk("témoin : sans mutation, les contrôles de la saisie du lot C (C3) sont verts", all(ok_ for f_ in PARTIES_C3.values() for _n, ok_, _i in f_()))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

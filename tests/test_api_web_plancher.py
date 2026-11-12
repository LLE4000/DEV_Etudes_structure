# -*- coding: utf-8 -*-
"""Le module « Charges sur plancher » du site (api_web/plancher.py) : ce que l'écran affiche est ce que le moteur
calcule ; chaque saisie, chaque refus, chaque bouton, les sélecteurs, l'échappement, les cas de la note, la pureté
(le module tourne sans streamlit ni pandas, comme dans le navigateur). Chaque contrôle rougit si on retire ce qu'il
protège. La parité avec l'écran Streamlit est prouvée à part : tests/test_charges_plancher_parite.py.

    python3.12 tests/test_api_web_plancher.py
"""
import copy
import json
import os
import re
import subprocess
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

from api_web import commun as C  # noqa: E402
from api_web import plancher as W  # noqa: E402
from api_web import routeur as R  # noqa: E402
from beton.hourdis import catalogue as KH  # noqa: E402
from plancher.charges import diffusion as DF  # noqa: E402
from plancher.charges import etat as ET  # noqa: E402
from plancher.charges import methode as MET  # noqa: E402
from plancher.charges import operations as OP  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{str(info)[:500]}]" if info and not cond else ""))


def req(**k):
    return json.loads(R.traiter(json.dumps(dict(module="plancher", **k))))


def cas(e):
    return ET.cas_actif(e)


def mod(e, chemin, valeur):
    return W.modifier(e, chemin, valeur)


# ------------------------------------------------------------------ 1. la note neuve : celle de l'écran d'origine
print("\n=== 1. La note neuve ===")
e0 = W.nouvel_etat()
ref = ET.nouvel_etat()
chk("la note neuve est celle de l'écran Streamlit (un cas C1, le hourdis par défaut, aucune charge) + les brouillons du site",
    {k: v for k, v in e0.items() if k != "site"} == {**ref, "cas": [dict(ref["cas"][0], site=W._site_cas())]})
# Décision de l'ingénieur du 11/11/2026 : un cas neuf démarre sur le SP 200 (la variante la moins armée), lu au catalogue
h0 = e0["cas"][0]["plancher"]["hourdis"]
f0 = KH.fiche_plancher("ergon_2018", "SP 200-4X")
chk("note neuve : le type par défaut est « SP 200-4X · Ergon 2018 », ses valeurs LUES au catalogue (M_Rd 51, V_Rd 89, g 2,78, "
    "I, E, M_Rd avec chape 66, 50 mm exigés, C30/37)",
    h0["type_cat"] == ET.TYPE_HOURDIS_DEFAUT == f0["nom"] and all(h0[k] == f0[k] for k in ET.CHAMPS_FICHE_HOURDIS)
    and (h0["M_Rd"], h0["V_Rd"], h0["g"], h0["M_Rd_chape"], h0["e_chape"], h0["chape"]["fck_exigee"]) == (51, 89, 2.78, 66, 0.05, 30.0),
    {k: h0.get(k) for k in ("type_cat", "M_Rd", "V_Rd", "g", "M_Rd_chape", "e_chape")})
r0 = W.resultat(W.nouvel_etat())
# à la main : g = 2,78 × 1,2 = 3,336 kN/m ; q = 1 × 1,2 ; M_Ed = (1,35 × 3,336 + 1,5 × 1,2) × 7,2² / 8 = 40,85 kN·m ; 40,85 / 51 = 80 %
chk("note neuve : conforme à 80 % (M_Ed = 40,85 kN·m ≤ M_Rd = 51 kN·m, recalculé à la main), le plan et la méthode",
    r0["resume"] == {"taux": 80, "niveau": "p", "statut": "Conforme", "gouvernant": "C1 · Flexion · H7"}
    and "40,85" in r0["resultat"] and "51,00" in r0["resultat"] and abs((1.35 * 2.78 * 1.2 + 1.5 * 1.2) * 7.2 ** 2 / 8 - 40.85) < 0.005
    and [v["cle"] for v in r0["vues"]][:1] == ["plan"] and "methode" in [v["cle"] for v in r0["vues"]], r0["resume"])
chk("le type par défaut est la variante la MOINS armée de son profil (un défaut du côté sûr)",
    KH.nom_type(KH.catalogue("ergon_2018"), KH.profil(KH.catalogue("ergon_2018"), "SP 200")["variantes"][0]) == ET.TYPE_HOURDIS_DEFAUT)
eg = W.nouvel_etat()
chk("retour à la fiche générique : accepté", mod(eg, ["plancher", "hourdis", "type_cat"], ET.TYPE_HOURDIS_GENERIQUE) is None)
hg = eg["cas"][0]["plancher"]["hourdis"]
chk("fiche générique : aucune condition du fabricant précédent ne reste (M_Rd et V_Rd à 0, ni épaisseur exigée, classe exigée neutre)",
    (hg["M_Rd"], hg["V_Rd"], hg["M_Rd_chape"], hg["e_chape"], hg["chape"]["fck_exigee"]) == (0.0, 0.0, None, None, 25.0),
    (hg["M_Rd"], hg["V_Rd"], hg["M_Rd_chape"], hg["e_chape"], hg["chape"]["fck_exigee"]))
r0 = W.resultat(eg)
chk("fiche générique : « À compléter », le message du moteur (M_Rd et V_Rd à renseigner), le plan, la méthode, la bibliothèque",
    r0["resume"]["niveau"] == "c" and "M<sub>Rd</sub> et V<sub>Rd</sub>" in r0["resultat"] and [v["cle"] for v in r0["vues"]] == ["plan", "methode"])
chk("fiche générique : le message du moteur est écrit UNE fois (le verdict y renvoie) et le site dit où compléter (section 2)",
    r0["resultat"].count("du catalogue à renseigner") == 1 and "Où compléter" in r0["resultat"] and "Section 2 « Hourdis »" in r0["resultat"])

# ------------------------------------------------------------------ 2. ce que l'écran affiche = ce que le moteur calcule
print("\n=== 2. Ce que l'écran affiche = ce que le moteur calcule ===")
for sorte in ("hourdis", "dalle"):
    e = W.exemple(sorte)
    c = cas(e)
    calc = W._calculer(e, c)
    res = calc["res"]
    r = W.resultat(e)
    t = res["taux_max"]
    chk(f"{sorte} : le verdict porte le taux du moteur ({C.pc(C.pct(t))} %) et sa couleur",
        f'<span class="chip xl {C.niveau(t)}"><small>η</small>{C.pc(C.pct(t))}{C.NBSP}%</span>' in r["resultat"])
    # (établi compact, 12/11/2026 : le taux est sur la première ligne ; sans taux, l'état se dit en mots — « et-mot »)
    fams = re.findall(r'<div class="el vc[^"]*" data-calcul="[^"]+" title="Voir le détail"><div class="l1"><b>([^<]+)</b>.*?'
                      r'<span class="(?:pct|et-mot) ([a-z])"[^>]*>([^<]+)</span>', r["resultat"])
    attendu = []
    for fam in W.ORDRE_FAM:
        vs = sorted((v for v in res["verifications"] if v["famille"] == fam), key=W._cle_famille, reverse=True)
        if vs:
            txt = W._texte_taux(vs[0])
            attendu.append((W.LIB_FAM[fam], W._niv(vs[0]), txt if txt else f"{C.pc(C.pct(vs[0]['taux']))}{C.NBSP}%"))
    chk(f"{sorte} : une ligne par famille, la plus défavorable, taux et couleur du moteur", fams[:len(attendu)] == attendu, (fams, attendu))
    eng = next(x for x in c["charges"] if x["type"] == "engin")
    lignes = OP.appuis_engin(c, res, calc["index"], eng["ref"])
    # (établi compact : « P1 · 45 kN » et son taux sur la première ligne, « porté par H7 » sur la seconde)
    vus = re.findall(r'<b>(P\d|R\d) · ([\d  ]+)\xa0kN</b>.*?<span class="pct ([a-z])">([^<]+)</span></div><div class="l2">'
                     r'<span class="cmp">porté par (\w+)</span>', r["resultat"])
    chk(f"{sorte} : « Appuis de {eng['ref']} » = le tableau de l'écran d'origine (appui, charge, élément, taux)",
        [(a, ch, el, txt) for a, ch, _, txt, el in vus] == [(li["appui"], C.fr(li["charge"], 0), li["element"], f"{C.pc(C.pct(li['taux']))}{C.NBSP}%") for li in lignes],
        (vus, lignes))
    chk(f"{sorte} : chaque ligne qui annonce un calcul a sa fenêtre", all(f'id="ck-{k}"' in r["volets"] for k in re.findall(r'data-calcul="([^"]+)"', r["resultat"])))
    chk(f"{sorte} : le résumé de la note = son cas le plus chargé", r["resume"]["taux"] == C.pct(t) and r["resume"]["gouvernant"].startswith(c["nom"]))

# la fenêtre d'un élément de hourdis : les formules substituées sont celles du moteur
e = W.exemple("hourdis")
calc = W._calculer(e, cas(e))
r = W.resultat(e)
mauvais = []
for el in calc["res"]["elements"]:
    m = re.search(rf'<div class="flyout calcul" id="ck-el-{el["index"]}">(.*?)</div></div></div>', r["volets"], re.S)
    h = m.group(1) if m else ""
    gG, gQ, L = el["gamma_G"], el["gamma_Q"], calc["plancher"].L
    VG, VQ = DF.tranchant_uniforme(el["w_g"], L), DF.tranchant_uniforme(el["w_q"], L)
    RA = (el["V_A"] - gG * VG) / gQ - VQ
    ok = (abs(gG * el["M_G"] + gQ * el["M_Q"] - el["M_max"]) < 1e-9 * max(1, el["M_max"])
          and abs(DF.moment_uniforme(el["w_q"], el["x_M"], L) + sum(DF.moment_ponctuel(P, x, el["x_M"], L) for x, P, _ in el["charges"]) - el["M_Q"]) < 1e-9 * max(1, el["M_Q"])
          and abs(gG * VG + gQ * (VQ + RA) - el["V_A"]) < 1e-9 * max(1, el["V_A"])
          and f"{W._d(gG)} × {W._r(el['M_G'])} + {W._d(gQ)} × {W._r(el['M_Q'])}" in h and W._r(el["M_max"], "kN·m") in h
          and W._r(el["V_max"], "kN") in h and f"{C.pc(C.pct(el['taux_M']))} %" in h)
    if not ok:
        mauvais.append(el["nom"])
chk("chaque élément : M_Ed = γG·M_G + γQ·M_Q, M_Q = w_Q·x(L − x)/2 + Σ M_P et V_A écrits avec les valeurs du moteur (et qui se recalculent)", not mauvais, mauvais)
# la même vérification lue PAR MÈTRE de largeur (demande de l'ingénieur du 11/11/2026) : efforts et résistances divisés par b
mauvais_m = []
for el in calc["res"]["elements"]:
    m = re.search(rf'<div class="flyout calcul" id="ck-el-{el["index"]}">(.*?)</div></div></div>', r["volets"], re.S)
    h = re.sub(r"<[^>]+>", "", m.group(1) if m else "")
    b_el = el["y1"] - el["y0"]
    attendu = [f"{W._r(el['M_max'])} / {W._d(b_el)}", W._r(el["M_max"] / b_el, "kN·m/m"), f"{W._r(el['M_adm'])} / {W._d(b_el)}",
               W._r(el["M_adm"] / b_el, "kN·m/m"), W._r(el["V_max"] / b_el, "kN/m"), W._r(el["V_adm"] / b_el, "kN/m")]
    if "Par mètre de largeur" not in h or not all(re.sub(r"<[^>]+>", "", a) in h for a in attendu):
        mauvais_m.append(el["nom"])
chk("chaque élément : le groupe « Par mètre de largeur » écrit M_Ed / b, M_Rd / b, V_Ed / b, V_Rd / b avec la largeur de l'élément",
    not mauvais_m, mauvais_m)
rn = W.resultat(W.nouvel_etat())
hn = re.sub(r"<[^>]+>", "", rn["volets"])
chk("note neuve (SP 200-4X, 1,2 m) : m_Rd = 51,00 / 1,2 = 42,50 kN·m/m et v_Rd = 89,00 / 1,2 = 74,17 kN/m (à la main)",
    "51,00 / 1,2" in hn and "42,50\xa0kN·m/m" in hn.replace(" ", "\xa0") and "74,17" in hn, hn[hn.find("Par mètre"):hn.find("Par mètre") + 300])
chk("la fenêtre de la filière écrit M_Rd = W_el·f_y/γM0 et γQ·M_max", "W<sub>el</sub>·f<sub>y</sub> / γ<sub>M0</sub>" in r["volets"] and 'id="ck-filiere"' in r["volets"])
chk("la fenêtre des plateaux écrit c = t·√(f / (3·p_Ed)) et l'aire efficace", 'id="ck-plateau"' in r["volets"] and "t·√(f / (3·p<sub>Ed</sub>))" in r["volets"]
    and "a<sub>eff</sub>" in r["volets"])
chk("la fenêtre des effets locaux montre la membrure sous les plus défavorables", 'id="ck-local"' in r["volets"] and "Membrure sous" in r["volets"])
d = W.resultat(W.exemple("dalle"))
chk("la fenêtre de la dalle écrit les résistances de l'application (α_b·μ·b·d², A_s·f_yd·0,9·d, τ_adm,I) et le poinçonnement",
    all(x in d["volets"] for x in ("α<sub>b</sub>·μ·b·d²", "A<sub>s</sub>·f<sub>yd</sub>·0,9·d", "τ<sub>adm,I</sub>", "v<sub>Rd,c</sub>", "F<sub>Rdu</sub>")))

# ------------------------------------------------------------------ 3. la saisie
print("\n=== 3. La saisie : chaque champ, les refus, les boutons ===")
e = W.nouvel_etat()
ok = [mod(e, ["plancher", "L"], "6,5"), mod(e, ["plancher", "hourdis", "M_Rd"], "1 250,5"), mod(e, ["plancher", "hourdis", "I"], ""),
      mod(e, ["plancher", "hourdis", "p_loc_adm"], "0"), mod(e, ["params", "k_flexion_heft240"], "0,5"), mod(e, ["plancher", "sens"], "2"),
      mod(e, ["plancher", "type"], "predalle"), mod(e, ["plancher", "predalle", "portee_etayee"], "2,4")]
c = cas(e)
chk("les saisies à la française se lisent (6,5 ; 1 250,5), « vide » ou 0 = non renseigné pour les champs facultatifs",
    not any(ok) and c["plancher"]["L"] == 6.5 and c["plancher"]["hourdis"]["M_Rd"] == 1250.5 and c["plancher"]["hourdis"]["I"] is None
    and c["plancher"]["hourdis"]["p_loc_adm"] is None and c["params"]["k_flexion_heft240"] == 0.5 and c["plancher"]["sens"] == 2
    and c["plancher"]["type"] == "predalle" and c["plancher"]["predalle"]["portee_etayee"] == 2.4)
for chemin, v, mot in [(["plancher", "L"], "0,3", "au moins 0,5"), (["plancher", "hourdis", "n_elements"], "0", "au moins 1"),
                       (["plancher", "hourdis", "b_element"], "0", "strictement positif"), (["params", "gamma_G"], "-1", "strictement positif"),
                       (["params", "part_tranchant_element"], "1,5", "au plus 1"), (["plancher", "type"], "bois", "Choix inconnu"),
                       (["plancher", "dalle", "fck_t"], "4", "sous la plus petite classe"), (["plancher", "hourdis", "type_cat"], "inconnu", "inconnu"),
                       (["plancher", "hourdis", "parts_annexe_C"], "0:0,5", "sans « : »"), (["plancher", "hourdis", "parts_annexe_C"], "0:-1", "positifs"),
                       (["selection"], "rep:Z9", "Appui inconnu"), (["n'existe", "pas"], "1", "Champ inconnu"), (["plancher", "L"], "x", "illisible")]:
    e = W.exemple("hourdis")
    avant = json.dumps(e, sort_keys=True)
    m = mod(e, chemin, v)
    if not (m and mot in m and json.dumps(e, sort_keys=True) == avant):
        chk(f"refus de {'/'.join(chemin)} = {v!r} ({mot})", False, m)
        break
else:
    chk("13 saisies hors domaine sont REFUSÉES avec leur raison, et la note reste intacte", True)
e = W.exemple("hourdis")
c = cas(e)
mod(e, ["charges", 1, "a"], "200")
mod(e, ["charges", 0, "x"], "3,6004")
chk("semelle saisie en mm, gardée en m ; positions au mm (3 décimales, comme le plan de l'écran d'origine)",
    c["charges"][1]["a"] == 0.2 and c["charges"][0]["x"] == 3.6)
n = len(c["charges"])
mod(e, ["charges", "+"], "lineaire")
mod(e, ["charges", 0, "dupliquer"], "1")
chk("ajouter une file d'étais, dupliquer un engin (décalé de 1 m, plateaux recopiés) : les fonctions de l'écran d'origine",
    len(c["charges"]) == n + 2 and c["charges"][-1]["ref"] == "E2" and c["charges"][-1]["x"] == round(min(c["charges"][0]["x"] + 1, 7.2), 3)
    and "E2.P1" in c["repartiteurs"])
mod(e, ["charges", len(c["charges"]) - 1, "-"], "1")
chk("retirer une charge retire ses répartiteurs", "E2.P1" not in c["repartiteurs"] and all(x["ref"] != "E2" for x in c["charges"]))
mod(e, ["charges", 0, "P", 2], "75")
chk("la charge d'un appui d'engin s'écrit à son nom (P3)", c["charges"][0]["P"]["P3"] == 75.0)
refs = W._refs(e, c)
mod(e, ["appuis", 1, "repartiteur"], "aucun")
chk("répartiteur « aucun »", refs[1] not in c["repartiteurs"])
mod(e, ["appuis", 1, "repartiteur"], "plateau:Tôle acier S235 · 800 × 800 × 20")
mod(e, ["appuis", 1, "dx"], "0,1")
chk("plateau choisi et décalé", c["repartiteurs"][refs[1]] == {"type": "plateau", "nom": "Tôle acier S235 · 800 × 800 × 20", "dx": 0.1, "dy": 0.0})
mod(e, ["appuis", 1, "idem"], "1")
chk("« idem » : le même plateau sous tous les appuis de l'engin", all(c["repartiteurs"][r_]["nom"].endswith("800 × 800 × 20") for r_ in refs[:4]))
nf = len(c["filieres"])
mod(e, ["appuis", 2, "repartiteur"], "filiere:nouvelle")
chk("« nouvelle filière » : une HEB 140 en travers, longue de trois éléments, centrée sur l'appui (etat.ajouter_filiere)",
    len(c["filieres"]) == nf + 1 and c["filieres"][-1]["nom"] == "HEB 140" and c["filieres"][-1]["longueur"] == 3.6)
fid = c["filieres"][-1]["id"]
i = len(c["filieres"]) - 1
mod(e, ["filieres", i, "porte", 3], True)
chk("une filière porte un autre appui (case « appuis portés »)", c["repartiteurs"][refs[3]] == {"type": "filiere", "id": fid})
mod(e, ["filieres", i, "porte", 3], False)
chk("… et le lâche", refs[3] not in c["repartiteurs"])
m = mod(e, ["filieres", i, "nom"], "Plateau inconnu")
c = cas(e)                                   # une saisie refusée remet la note entière : on relit le cas
chk("élément de filière inconnu : refusé", m and "inconnu" in m)
mod(e, ["filieres", i, "-"], "1")
chk("retirer une filière libère ses appuis", all(x["id"] != fid for x in c["filieres"]) and refs[2] not in c["repartiteurs"])
mod(e, ["site", "balaye"], c["charges"][0]["ref"])
mod(e, ["selection"], f"rep:{refs[2]}")
chk("l'engin balayé et l'appui de la coupe se choisissent", W._balaye(c) == "E1" and c["selection"] == f"rep:{refs[2]}"
    and any(v["cle"] == "coupe" and v["titre"] == f"Coupe sous {refs[2]}" for v in W.resultat(e)["vues"]))

# proposition d'une filière, son application
e = W.exemple("hourdis")
W._faire(e, (["appuis", 4, "repartiteur"], "aucun"), (["filieres", 0, "-"], "1"), (["charges", 1, "P"], "120"))
r = W.resultat(e)
chk("non conforme : le bouton « Proposer une filière » apparaît (section 4, ouverte)", 'data-chemin="proposition/+"' in r["donnees"] and r["resume"]["niveau"] == "r")
mod(e, ["proposition", "+"], "1")
p = cas(e)["site"]["proposition"]
r = W.resultat(e)
chk("la proposition est dite (texte de l'écran d'origine) et s'applique", p["trouve"] and C.esc(OP.phrase_proposition(p)) in r["resultat"]
    and 'data-chemin="proposition/appliquer"' in r["donnees"])
avant = copy.deepcopy(e)
mod(e, ["charges", 0, "x"], "3,5")
chk("une proposition faite sur un autre état n'est plus montrée ni appliquée", mod(e, ["proposition", "appliquer"], "1") is not None)
e = avant
mod(e, ["proposition", "appliquer"], "1")
chk("« Appliquer » : la filière rejoint le cas et la note redevient conforme", len(cas(e)["filieres"]) == 1 and W.resume_note(e, [W._calculer(e, cas(e))["res"]])["niveau"] != "r")



def comme_le_navigateur(x):
    """Ce que JSON.stringify fait de l'état : un flottant entier (45.0) s'écrit « 45 »."""
    if isinstance(x, float) and x.is_integer():
        return int(x)
    if isinstance(x, dict):
        return {k: comme_le_navigateur(v) for k, v in x.items()}
    if isinstance(x, list):
        return [comme_le_navigateur(v) for v in x]
    return x


e = W.exemple("hourdis")
W._faire(e, (["appuis", 4, "repartiteur"], "aucun"), (["filieres", 0, "-"], "1"), (["charges", 1, "P"], "120"), (["proposition", "+"], "1"))
n = W.normaliser(comme_le_navigateur(json.loads(json.dumps(e))))
chk("la proposition survit à l'aller-retour par le navigateur (45.0 écrit « 45 ») : toujours montrée, toujours applicable",
    "Proposition :" in W.resultat(n)["resultat"] and mod(n, ["proposition", "appliquer"], "1") is None and len(cas(n)["filieres"]) == 1)

# fiches ajoutées
e = W.nouvel_etat()
chk("fiche d'engin sans nom : refusée (« Un nom est nécessaire. »)", mod(e, ["note", "fiche_engin", "enregistrer"], "1") == "Un nom est nécessaire.")
mod(e, ["note", "fiche_engin", "nom"], "Grue <b>X</b>")
mod(e, ["note", "fiche_engin", "appuis", 1, "nom"], "P1")
chk("deux appuis du même nom : refusé", "différent" in (mod(e, ["note", "fiche_engin", "enregistrer"], "1") or ""))
mod(e, ["note", "fiche_engin", "appuis", 1, "nom"], "P2")
mod(e, ["note", "fiche_engin", "appuis", "+"], "1")
mod(e, ["note", "fiche_engin", "appuis", 4, "P"], "40")
chk("fiche d'engin enregistrée = operations.fiche_engin, posable", mod(e, ["note", "fiche_engin", "enregistrer"], "1") is None
    and e["fiches"]["engins"][0]["nom"] == "Grue <b>X</b>" and len(e["fiches"]["engins"][0]["appuis"]) == 5
    and mod(e, ["charges", "+engin"], "Grue <b>X</b>") is None)
mod(e, ["note", "fiche_rep", "nom"], "Tôle acier S235 · 600 × 600 × 20")
chk("fiche de répartiteur : un nom de la bibliothèque livrée est refusé", "livrée" in (mod(e, ["note", "fiche_rep", "enregistrer"], "1") or ""))
W._faire(e, (["note", "fiche_rep", "nom"], "Madrier C24"), (["note", "fiche_rep", "type"], "filiere"), (["note", "fiche_rep", "materiau"], "Bois C24"),
         (["note", "fiche_rep", "f"], "14,8"), (["note", "fiche_rep", "A"], "75"), (["note", "fiche_rep", "B"], "225"), (["note", "fiche_rep", "enregistrer"], "1"))
fiche = e["fiches"]["repartiteurs"][0]
chk("fiche de filière libre (madrier sur champ : rectangle b × h) enregistrée = operations.fiche_repartiteur",
    fiche == OP.fiche_repartiteur("filiere", "Madrier C24", "Bois C24", 14.8, 210000.0, 0.0, "", 75.0, 225.0, 20.0, OP.RECTANGLE))
W._faire(e, (["plancher", "hourdis", "M_Rd"], "90"), (["plancher", "hourdis", "V_Rd"], "70"), (["note", "nom_fiche_hourdis"], "SP 200 · 2026"),
         (["note", "fiche_hourdis", "enregistrer"], "1"))
chk("« Garder ces valeurs comme fiche » : la fiche de hourdis entre au catalogue du fichier et devient le type",
    e["fiches"]["hourdis"][0]["M_Rd"] == 90.0 and cas(e)["plancher"]["hourdis"]["type_cat"] == "SP 200 · 2026")

# le fichier de l'écran Streamlit
h = W.exemple("hourdis")
texte = ET.texte_json({k: v for k, v in h.items() if k != "site"})
e = W.nouvel_etat()
chk("un fichier JSON de l'écran Streamlit s'ouvre (tous les cas, les fiches)", mod(e, ["note", "ouvrir"], {"nom": "c.json", "texte": texte}) is None
    and cas(e)["nom"] == "C1 levage en rive" and json.dumps(W._calculer(e, cas(e))["res"], sort_keys=True) == json.dumps(W._calculer(h, cas(h))["res"], sort_keys=True))
chk("un fichier qui n'est pas du module est refusé, la note reste", "refusé" in mod(e, ["note", "ouvrir"], {"nom": "x.json", "texte": '{"module": "poutre"}'}))

# chaque champ affiché pointe l'état : le réécrire avec sa valeur ne change rien (sur les notes complètes)
CHAMPS = re.compile(r'<input data-chemin="([^"]+)" data-sorte="(nombre|entier|texte)"[^>]*?value="([^"]*)"')
MENUS = re.compile(r'<select data-chemin="([^"]+)" data-sorte="choix"[^>]*>.*?<option value="([^"]*)" selected', re.S)
for sorte in ("hourdis", "dalle", "predalle"):
    e = W.exemple(sorte)
    W._faire(e, (["note", "fiche_rep", "type"], "filiere"))
    h = W.resultat(e)["donnees"]
    champs = CHAMPS.findall(h) + [(c_, "choix", v) for c_, v in MENUS.findall(h)]
    mauvais = []
    for chemin, _, v in champs:
        v = v.replace("&amp;", "&").replace("&quot;", '"').replace("&lt;", "<").replace("&gt;", ">").replace("&#x27;", "'")
        e2 = copy.deepcopy(e)
        m = mod(e2, chemin.split("/"), v)
        if m or json.dumps(e2, sort_keys=True) != json.dumps(e, sort_keys=True):
            mauvais.append((chemin, v, m))
    chk(f"{sorte} : réécrire chacun des {len(champs)} champs affichés avec sa valeur ne change rien (mm, positions, menus)", not mauvais and len(champs) > 40, mauvais[:3])

# ------------------------------------------------------------------ 4. les sélecteurs
print("\n=== 4. Les sélecteurs ===")
e = W.exemple("hourdis")
h = W.selecteur(e, "rep-0")
noms = re.findall(r'data-choix="([^"]+)"', h)
masses = [float(x.replace(",", ".").replace(C.NBSP, "")) for x in re.findall(r'<span class="kg">([^<]+)</span>', h)]
chk("répartiteur d'un appui : sans, chaque plateau, chaque filière — triés par masse", noms[0] == W.SANS and masses == sorted(masses) and len(noms) >= 7, noms)
chk("le retenu (✓) est le plateau de l'appui", re.search(r'class="pk-row cur[^"]*" data-choix="Tôle 600 × 600 × 20"', h) is not None)
opt = re.findall(r'class="pk-row[^"]*opt[^"]*" data-choix="([^"]+)"', h)
chk("★ : le plus léger qui tient (les plateaux sans résistance f sont « refusé »)", len(opt) == 1 and ">refusé<" in h, opt)
W.choisir_section(e, "rep-0", "Tôle 800 × 800 × 20")
chk("choisir une ligne la retient", cas(e)["repartiteurs"]["E1.P1"]["nom"] == "Tôle acier S235 · 800 × 800 × 20")
h = W.selecteur(e, "fil-0", ["HEB"])
noms = re.findall(r'data-choix="([^"]+)"', h)
chk("profilé d'une filière : la famille cochée seulement, par masse", noms and all(n.startswith("HEB ") for n in noms) and 'data-famille="IPE"' in h)
opt = re.findall(r'class="pk-row[^"]*opt[^"]*" data-choix="([^"]+)"', h)
W.choisir_section(e, "fil-0", opt[0] if opt else "HEB 160")
chk("choisir un profilé le retient", cas(e)["filieres"][0]["nom"] == (opt[0] if opt else "HEB 160"))
try:
    W.choisir_section(e, "fil-0", "IPE 9999")
    chk("un choix inconnu est refusé", False)
except ValueError:
    chk("un choix inconnu est refusé", True)

# ------------------------------------------------------------------ 5. les vues
print("\n=== 5. Les vues ===")
r = W.resultat(W.exemple("hourdis"))
vues = {v["cle"]: v for v in r["vues"]}
chk("vues : plan, coupe, balayage, détail, méthode (avec la bibliothèque et ses sources, comme l'onglet Méthode d'origine)",
    list(vues) == ["plan", "coupe", "balayage", "detail", "methode"] and "Bibliothèque et sources" in vues["methode"]["svg"]
    and all(C.esc(n) in vues["methode"]["svg"] for n in ET.gabarits(W.exemple("hourdis"))))
chk("le plan est statique (aucune poignée ni attribut d'interaction du composant cliquable)",
    not re.search(r'class="ed|data-(group|drag|key|action)=', vues["plan"]["svg"]) and "<title>" not in vues["plan"]["svg"])
ids = re.findall(r'id="([^"]+)"', "".join(v["svg"] for v in r["vues"]))
chk("les identifiants internes des dessins sont uniques dans la page (préfixe par vue)", len(ids) == len(set(ids)), [x for x in ids if ids.count(x) > 1][:3])
chk("code couleur du site dans le plan : l'orange de l'écran d'origine (seuil 85 %) n'y est plus", "#c2410c" not in vues["plan"]["svg"].lower()
    and "#ffedd5" not in vues["plan"]["svg"].lower())
chk("la méthode porte les textes PARTAGÉS (plancher/charges/methode.py), comme l'onglet de l'écran d'origine et la note",
    all(C.esc(s_["titre"]) in vues["methode"]["svg"] for s_ in MET.SECTIONS) and C.esc(MET.A_TRANSCRIRE[0]) in vues["methode"]["svg"])
chk("le balayage dit les positions admissibles (phrase de l'écran d'origine)", "Positions admissibles" in vues["balayage"]["svg"] or "Aucune position" in vues["balayage"]["svg"])

# ------------------------------------------------------------------ 6. l'échappement, les cas, la relecture
print("\n=== 6. Échappement, cas multiples, relecture ===")
PIEGE = '<img src=x onerror=alert(1)>'
e = W.exemple("hourdis")
W._faire(e, (["nom"], PIEGE), (["note", "fiche_engin", "nom"], PIEGE), (["note", "fiche_engin", "appuis", 0, "nom"], PIEGE + "1"),
         (["note", "fiche_engin", "appuis", 0, "P"], "30"), (["note", "fiche_engin", "appuis", 1, "P"], "30"), (["note", "fiche_engin", "appuis", 2, "P"], "30"),
         (["note", "fiche_engin", "appuis", 3, "P"], "30"), (["note", "fiche_engin", "enregistrer"], "1"), (["charges", "+engin"], PIEGE))
r = W.resultat(e)
frag = {k: r[k] for k in ("resultat", "volets", "donnees", "elements", "phrase", "statut_html")}
frag["vues"] = "".join(v["svg"] for v in r["vues"])
fuites = [k for k, v in frag.items() if "<img" in v]
chk("un nom piégé (cas, engin, appui) est échappé dans tous les fragments, dessins compris", not fuites, fuites)
sel = W.selecteur(e, "rep-5")
chk("… et dans les sélecteurs", "<img" not in sel)
e = W.exemple("hourdis")
W.gerer_elements(e, "ajouter")
W.gerer_elements(e, "activer", 0)
W.gerer_elements(e, "dupliquer")
chk("cas : ajouter (C2), copier (« (copie) » : la copie s'insère juste après le cas copié, et s'ouvre), activer",
    [c_["nom"] for c_ in e["cas"]] == ["C1 levage en rive", "C1 levage en rive (copie)", "C2"] and e["actif"] == 1)
W.gerer_elements(e, "supprimer", 0)
W.gerer_elements(e, "supprimer", 0)
W.gerer_elements(e, "supprimer", 0)
chk("le dernier cas ne se supprime pas", len(e["cas"]) == 1)
e = W.exemple("hourdis")
e["cas"].append(W.exemple("dalle")["cas"][0])
mod(e, ["plancher", "hourdis", "M_Rd"], "60")
r = W.resultat(e)
chk("le résumé de la note est son cas le plus défavorable (non conforme d'abord)", r["resume"]["niveau"] == "r" and r["resume"]["gouvernant"].startswith("C1"))
for abime, mot in [({"cas": "x"}, "liste de cas"), ({"cas": [{"plancher": {"type": "bois"}}]}, "type de plancher"),
                   ({"cas": [{"charges": [{"type": "engin", "ref": "E1", "gabarit": "absent", "x": 1, "y": 1}]}]}, "absente de la bibliothèque"),
                   ({"cas": [{"plancher": {"L": "7"}}]}, "nombre attendu")]:
    try:
        W.normaliser(abime)
        chk(f"note abîmée ({mot})", False)
    except ValueError as ex:
        if mot not in str(ex):
            chk(f"note abîmée ({mot})", False, ex)
chk("une note abîmée lève une erreur claire (liste de cas, type, gabarit absent, nombre)", True)
vieux = W.exemple("hourdis")
del vieux["cas"][0]["identification"], vieux["cas"][0]["params"]["alpha_cc"], vieux["site"]
chk("un fichier ancien (clés manquantes) est complété avec les défauts de l'écran d'origine",
    W.normaliser(vieux)["cas"][0]["params"]["alpha_cc"] == 1.0 and W.normaliser(vieux)["cas"][0]["identification"]["indice"] == "0")
# Une note d'AVANT le catalogue des hourdis n'a ni e_chape, ni M_Rd_chape, ni fck_exigee : elle est complétée des valeurs
# NEUTRES — jamais de celles du SP 200 d'un cas neuf, qui lui feraient exiger 50 mm de couche et un C30/37 qu'elle n'a pas.
avant_cat = W.exemple("hourdis")
for k_ in ("e_chape", "M_Rd_chape"):
    avant_cat["cas"][0]["plancher"]["hourdis"].pop(k_, None)
avant_cat["cas"][0]["plancher"]["hourdis"]["chape"].pop("fck_exigee", None)
h_av = W.normaliser(avant_cat)["cas"][0]["plancher"]["hourdis"]
chk("une note d'avant le catalogue (sans e_chape, M_Rd_chape, fck_exigee) est complétée des valeurs NEUTRES, pas de celles du SP 200 "
    "d'un cas neuf (aucune épaisseur de couche exigée, pas de M_Rd avec couche, C25)",
    h_av["e_chape"] is None and not h_av["M_Rd_chape"] and h_av["chape"]["fck_exigee"] == 25.0
    and ET.nouvel_etat()["cas"][0]["plancher"]["hourdis"]["e_chape"] == 0.05,
    (h_av.get("e_chape"), h_av.get("M_Rd_chape"), h_av["chape"].get("fck_exigee")))
chk("le format de fichier de l'écran Streamlit est relu tel quel", W.normaliser(ET.payload(ET.nouvel_etat()))["cas"][0]["nom"] == "C1")
rq = req(action="calculer", etat=W.exemple("dalle"))
chk("par le routeur : réponse complète", rq.get("ok") and rq["resume"]["niveau"] in ("p", "f", "r"))

# ------------------------------------------------------------------ 6 bis. la relecture indépendante (07/11/2026)
print("\n=== 6 bis. Relecture indépendante : défaut non conservatif du moteur DIT, bornes, lisibilité ===")


def tranchant_p3(etat_h, parts, ptr):
    """Un étançon de 100 kN au milieu d'un hourdis sans poids propre ni charge présente, coefficients 1."""
    e = W.nouvel_etat()
    W._faire(e, (["plancher", "hourdis", "M_Rd"], "120"), (["plancher", "hourdis", "V_Rd"], "90"), (["plancher", "q_presentes"], "0"),
             (["plancher", "hourdis", "g"], "0"), (["plancher", "hourdis", "etat"], etat_h), (["plancher", "hourdis", "parts_annexe_C"], parts),
             (["params", "part_tranchant_element"], ptr), (["params", "gamma_Q"], "1"), (["params", "gamma_G"], "1"),
             (["plancher", "hourdis", "p_loc_adm"], "5"), (["charges", "+"], "ponctuel"), (["charges", 0, "P"], "100"),
             (["charges", 0, "x"], "1"), (["charges", 0, "y"], "4,2"))
    res = W._calculer(e, cas(e))["res"]
    return e, sum(x["V_A"] + x["V_B"] for x in res["elements"])


e_p3, somme = tranchant_p3("poses", "0:0.5 ; 1:0.25", "0,7")
_, somme_ok = tranchant_p3("clavetes", "0:0.5 ; 1:0.25", "0,7")
r = W.resultat(e_p3)
chk("DÉFAUT NON CONSERVATIF du moteur (le même dans Streamlit) : part du tranchant 0,7 sans répartition active → 30 % de la "
    "charge ne sont portés par aucun élément (Σ réactions = 70 kN pour 100 kN) ; répartition active : 100 kN",
    abs(somme - 70.0) < 1e-6 and abs(somme_ok - 100.0) < 1e-6, (somme, somme_ok))
chk("… le site le DIT en rouge, SOUS LE VERDICT (avant les lignes du résultat), et dans la fenêtre de l'élément",
    "Tranchant sous-estimé (P-3)" in r["resultat"] and 'msg grave"><b>Tranchant sous-estimé' in r["resultat"] and "NON CONSERVATIF" in r["volets"]
    and r["resultat"].index("Tranchant sous-estimé") < r["resultat"].index('<div class="els">'))
e2 = copy.deepcopy(e_p3)
W._faire(e2, (["params", "part_tranchant_element"], "1"))
e3 = copy.deepcopy(e_p3)
W._faire(e3, (["plancher", "hourdis", "etat"], "clavetes"))
chk("… et se tait quand le moteur porte bien toute la charge (part 1, ou répartition active)",
    "Tranchant sous-estimé" not in W.resultat(e2)["resultat"] and "Tranchant sous-estimé" not in W.resultat(e3)["resultat"])
chk("les parts s'écrivent à la française (« 70 % », jamais « 70% »)", "70%" not in r["donnees"] + r["volets"] and "70\xa0%" in r["donnees"])

e = W.exemple("dalle")
W._faire(e, (["plancher", "type"], "predalle"), (["plancher", "predalle", "portee_etayee"], "2,5"), (["plancher", "predalle", "joints_armes"], False))
r = W.resultat(e)
chk("prédalle sur étais (P-8, hypothèse « attention » du moteur) : « Conforme, attention » comme l'écran d'origine, même à 54 %",
    r["resume"]["taux"] < 95 and "Conforme, attention" in r["statut_html"] and "étais en place" in r["resultat"])

for chemin, v, mot in [(["plancher", "L"], "1000", "au plus 40"), (["plancher", "hourdis", "n_elements"], "10000", "au plus 100"),
                       (["params", "gamma_Q"], "50", "au plus 5"), (["charges", 2, "lx"], "40", "au plus 2500"),
                       (["charges", 0, "P", 0], "-5", "au moins 0")]:
    e = W.exemple("hourdis")
    if chemin == ["charges", 2, "lx"]:
        W._faire(e, (["charges", 2, "ly"], "9"), (["charges", 2, "x0"], "0"), (["charges", 2, "y0"], "0"))
    avant = json.dumps(e, sort_keys=True)
    m = mod(e, chemin, v)
    if not (m and mot in m and json.dumps(e, sort_keys=True) == avant):
        chk(f"borne de {'/'.join(map(str, chemin))} = {v}", False, m)
        break
else:
    chk("bornes hautes du site (portée, éléments, γ, stock découpé en plus de 2 500 charges) : refus clair, note intacte", True)
e = W.exemple("dalle")
chk("dalle : une hauteur utile d ≥ h est refusée (l'écran d'origine l'acceptait et M_adm grandissait avec d)",
    "doit rester sous" in (mod(e, ["plancher", "dalle", "d"], "0,25") or "") and "doit rester sous" in (mod(e, ["plancher", "dalle", "h"], "0,15") or "")
    and mod(e, ["plancher", "dalle", "d"], "0,18") is None)
grand = W.exemple("hourdis")
grand["cas"][0]["plancher"]["L"] = 1000.0
try:
    W.normaliser(grand)
    chk("un fichier .stab à la portée de 1 000 m est refusé (il bloquerait le navigateur)", False)
except ValueError as ex:
    chk("un fichier .stab à la portée de 1 000 m est refusé (il bloquerait le navigateur)", "au plus 40" in str(ex), ex)

r = W.resultat(W.exemple("hourdis"))
el = re.search(r'id="ck-el-1">(.*?)</div></div></div>', r["volets"], re.S).group(1)
chk("fenêtre d'un élément : un stock découpé tient sur UNE ligne (« S1 (16 morceaux) ») et un appui à cheval sur deux éléments aussi",
    "S1 <small>(16 morceaux)</small>" in el and el.count(">E1.P3<") == 1 and "S1.1.1" not in el)
calc = W._calculer(W.exemple("hourdis"), cas(W.exemple("hourdis")))
e1 = calc["res"]["elements"][1]
som = sum(DF.moment_ponctuel(P, x, e1["x_M"], calc["plancher"].L) for x, P, _ in e1["charges"])
chk("… et la ligne Σ porte la somme des M_P que M_Q reprend (une projection qui se recalcule)", f"<b>{W._r(som, 'kN·m')}</b>" in el
    and f"+ {W._r(som)}" in el)
d = W.resultat(W.exemple("dalle"))
chk("fenêtre de la dalle : la file d'étais découpée tient sur une ligne, τ_adm,I garde trois décimales (0,564)",
    "F1 <small>(10 morceaux)</small>" in d["volets"] and "0,564 × 0,75" in d["volets"])
chk("effets locaux de la dalle : F_Rdu et v_Rd,max écrits avec leurs valeurs (min(…), ν = 0,6·(1 − f_ck/250))",
    f"min({C.fr(600000, 0)} × 20,00 × √({C.fr(950000, 0)} / {C.fr(600000, 0)})" in d["volets"] and "ν = 0,6·(1 − f<sub>ck</sub>/250)" in d["volets"] and "u<sub>0</sub>" in d["volets"])
tout = d["resultat"] + d["volets"] + "".join(v["svg"] for v in d["vues"]) + d["resume"]["gouvernant"] + W.selecteur(W.exemple("dalle"), "rep-0")
chk("les textes du moteur s'écrivent à la française partout (« y = 3,74 m », jamais « y = 3.74 m »)", not re.search(r"= ?\d+\.\d", tout),
    re.findall(r".{20}= ?\d+\.\d.{10}", tout)[:3])

ctx_e = W.exemple("hourdis")
c_ = cas(ctx_e)
k_ = W._calculer(ctx_e, c_)
ctx = ET.contexte_dessin(ctx_e, c_, k_["res"], k_["plancher"], k_["index"])
from api_web import dessin_plancher as DPL  # noqa: E402
from plancher.charges import dessin as DS  # noqa: E402
c2 = DPL.echapper(dict(ctx))
avant_r = c2["engins"][0]["rayon_etiquette"]
DPL._rayon_etiquette(c2)
v_ = DS.Vue(ctx["L"], ctx["B"], 1)
k_e = max(1.0, v_.largeur_svg / DS.LARGEUR_REF)
bas = max((ap["y"] - c2["engins"][0]["y"]) * 1e3 + max(ap["b"] * 500, 300) + 120 + 340 * k_e for ap in c2["engins"][0]["empreintes"])
haut = 0.75 * DS.FS_S * k_e                                   # le haut des lettres, au-dessus de la ligne de base
chk("plan : le nom de l'engin passe SOUS les étiquettes de ses appuis (il les chevauchait)", avant_r - haut < bas < c2["engins"][0]["rayon_etiquette"] - haut,
    (avant_r, bas, c2["engins"][0]["rayon_etiquette"]))
plan = next(v for v in r["vues"] if v["cle"] == "plan")["svg"]
chk("plan : la légende des symboles (semelle, plateau, aire efficace, filière, couleurs) est dessinée sous le plan, comme dans l'écran d'origine",
    all(f">{t}<" in plan for t in ("semelle", "plateau", "aire efficace", "filière", "élément 95–100 %", "&gt; 100 %")) and 'fill="url(#chp' in plan)
e = W.exemple("hourdis")
W._faire(e, (["charges", 0, "x"], "9"))
r = W.resultat(e)
chk("une charge hors du plancher : le moteur le dit, le site dit où corriger (section 3)", "hors du plancher" in r["resultat"]
    and "Section 3 « Charges » : la position x, y" in r["resultat"] and "Vérification impossible" in r["resultat"])


# ------------------------------------------------------------------ 7. pureté : sans streamlit ni pandas, comme dans le navigateur
print("\n=== 7. Pureté : le module se charge et calcule sans streamlit, pandas, matplotlib, reportlab, scipy ===")
SOUS = r'''
import json, sys
INTERDITS = ("streamlit", "pandas", "reportlab", "matplotlib", "scipy", "PIL", "fitz")
class Refus:
    def find_spec(self, nom, chemin=None, cible=None):
        if nom.split(".")[0] in INTERDITS:
            raise ImportError("refusé : " + nom)
        return None
sys.meta_path.insert(0, Refus())
sys.path.insert(0, sys.argv[1])
from api_web import plancher as W
r = W.resultat(W.exemple("hourdis"))
s = W.selecteur(W.exemple("hourdis"), "fil-0")
charges = sorted(m for m in sys.modules if m.startswith("plancher.charges"))
print(json.dumps({"resume": r["resume"], "charges": charges, "interdits": [m for m in sys.modules if m.split(".")[0] in INTERDITS]}))
'''
pr = subprocess.run([sys.executable, "-I", "-c", SOUS, RACINE], capture_output=True, text=True, check=False)
out = json.loads(pr.stdout) if pr.returncode == 0 else {}
chk("sous un sys.meta_path qui refuse streamlit, pandas, matplotlib, reportlab, scipy : le module calcule (numpy seul admis)",
    pr.returncode == 0 and not out.get("interdits") and out["resume"] == W.resultat(W.exemple("hourdis"))["resume"], pr.stderr[-600:])
chk("ni note.py ni peintre.py (reportlab) ne sont chargés", out and "plancher.charges.note" not in out["charges"] and "plancher.charges.peintre" not in out["charges"],
    out.get("charges"))

print("\n=== Catalogue Ergon 2018 (beton/donnees/hourdis, 11/11/2026) ===")
from beton.hourdis import catalogue as KH  # noqa: E402
ERG = "SP 200-4-2X · Ergon 2018"
GEN = "Hourdis générique 200 · à renseigner"
e = W.nouvel_etat()
sec = W._sec_hourdis(e, cas(e))
opts = re.findall(r'<option value="([^"]*)"', sec)
chk("menu des types : la fiche générique d'abord, hors groupe, puis un groupe par profil Ergon (7), 54 options",
    opts[0] == GEN and len(opts) == 54 and sec.count("<optgroup") == 7 and 'label="Ergon 2018 · SP 150"' in sec
    and 'label="Ergon 2018 · SPZ 500"' in sec and sec.index(GEN) < sec.index("<optgroup"), (len(opts), opts[:3]))
chk("menu des types : chaque variante une fois, dans l'ordre du fabricant",
    opts[1:] == [f["nom"] for f in KH.fiches_plancher()])
chk("type Ergon choisi : aucun message", mod(e, ["plancher", "hourdis", "type_cat"], ERG) is None)
h = cas(e)["plancher"]["hourdis"]
chk("type Ergon : valeurs de calcul par élément recopiées (b 1,2 m, h 0,2 m, g 2,78, E 38 214, I 6,3·10⁻⁴, M_Rd 113, V_Rd 94, Rd)",
    (h["b_element"], h["h"], h["g"], h["E"], h["M_Rd"], h["V_Rd"], h["nature"]) == (1.2, 0.2, 2.78, 38214.0, 113.0, 94.0, "Rd")
    and abs(h["I"] - 6.3e-4) < 1e-15, h)
chk("type Ergon : M_Rd avec chape 141, V_Rd avec chape vide (non publié), épaisseur exigée 0,05 m, classe exigée 30 MPa",
    h["M_Rd_chape"] == 141.0 and h["V_Rd_chape"] is None and h["e_chape"] == 0.05 and h["chape"]["fck_exigee"] == 30.0, h)
chk("type Ergon : la source cite le guide et ses pages", "Ergon" in h["source"] and "p. 30" in h["source"])
W._faire(e, (["plancher", "hourdis", "chape", "oui"], True), (["plancher", "hourdis", "chape", "fck_t"], "30"),
         (["plancher", "hourdis", "chape", "epaisseur"], "0,05"), (["charges", "+"], "ponctuel"))
calc = W._calculer(e, cas(e))
el = calc["res"]["elements"]
chk("chape C30/37 de 50 mm : M_adm = M_Rd avec chape 141, V_adm = V_Rd nu 94 (prudent)",
    el and all(abs(x["M_adm"] - 141.0) < 1e-9 and abs(x["V_adm"] - 94.0) < 1e-9 for x in el), [(x["M_adm"], x["V_adm"]) for x in el])
hy = " ".join(W._hyps_communes(cas(e), calc))
chk("l'en-tête des calculs le dit : « M_Rd avec chape, V_Rd nu »", "M<sub>Rd</sub> <b>avec chape</b>, V<sub>Rd</sub> nu" in hy, hy)
chk("une ligne d'hypothèse du moteur le dit aussi (V_Rd nu, valeur avec chape non publiée)",
    any("V_Rd nu" in (v.get("methode") or "") for v in calc["res"]["verifications"] if v["famille"] == "hypothese"))
sec = W._sec_hourdis(e, cas(e))
chk("chape cochée : le champ « épaisseur exigée » est à l'écran, à 0,05 m", "e_chape" in sec and "épaisseur exigée" in sec)
chk("épaisseur exigée négative refusée", mod(e, ["plancher", "hourdis", "e_chape"], "-0,01") is not None and cas(e)["plancher"]["hourdis"]["e_chape"] == 0.05)
W._faire(e, (["plancher", "hourdis", "chape", "epaisseur"], "0,04"))
el = W._calculer(e, cas(e))["res"]["elements"]
chk("chape de 40 mm < 50 mm : valeurs nues (M_adm 113)", el and all(abs(x["M_adm"] - 113.0) < 1e-9 for x in el))
W._faire(e, (["plancher", "hourdis", "chape", "epaisseur"], "0,06"), (["plancher", "hourdis", "chape", "fck_t"], "25"))
el = W._calculer(e, cas(e))["res"]["elements"]
chk("chape à 25 MPa < C30/37 exigé : valeurs nues (M_adm 113)", el and all(abs(x["M_adm"] - 113.0) < 1e-9 for x in el))
W._faire(e, (["note", "nom_fiche_hourdis"], "SP 200-4-2X bureau"), (["note", "fiche_hourdis", "enregistrer"], "1"))
fh = e["fiches"]["hourdis"][-1]
chk("« Garder comme fiche » depuis un type Ergon : la fiche garde l'épaisseur et la classe exigées",
    fh.get("nom") == "SP 200-4-2X bureau" and fh.get("e_chape") == 0.05 and fh.get("fck_exigee") == 30.0 and fh.get("M_Rd_chape") == 141.0, fh)
chk("retour à la fiche générique : l'épaisseur exigée d'Ergon ne reste pas", mod(e, ["plancher", "hourdis", "type_cat"], GEN) is None
    and cas(e)["plancher"]["hourdis"]["e_chape"] is None and cas(e)["plancher"]["hourdis"]["M_Rd"] == 0.0)
sec = W._sec_hourdis(e, cas(e))
chk("une fiche du fichier va au groupe « Fiches de ce fichier »", 'label="Fiches de ce fichier"' in sec and "SP 200-4-2X bureau" in sec)
vm = next(v for v in W.resultat(e)["vues"] if v["cle"] == "methode")
contenu = json.dumps(vm, ensure_ascii=False)
chk("méthode : UNE ligne pour le catalogue (« Hourdis Ergon 2018 », 7 profils, 53 variantes), pas une par variante",
    "Hourdis Ergon 2018" in contenu and "53 variantes" in contenu and "SP 150-5X" not in contenu, contenu[:200])
# un fichier d'avant le catalogue (sans e_chape) s'ouvre ; l'épaisseur exigée est vide (aucune condition)
vieux = {k: v for k, v in W.exemple("hourdis").items() if k != "site"}
for c_ in vieux["cas"]:
    c_["plancher"]["hourdis"].pop("e_chape", None)
e = W.nouvel_etat()
chk("un fichier d'avant le catalogue (sans e_chape) s'ouvre ; e_chape complété à vide",
    mod(e, ["note", "ouvrir"], {"nom": "v.json", "texte": ET.texte_json(vieux)}) is None
    and cas(e)["plancher"]["hourdis"]["e_chape"] is None)

# ------------------------------------------------------------------ l'établi compact (la mise en forme des poutres, 12/11/2026)
import html as _html  # noqa: E402


def _texte(h):
    return re.sub(r"\s+", " ", _html.unescape(re.sub(r"<[^>]+>", "", h))).replace("\u202f", " ").replace("\xa0", " ").strip()


def _rubriques(h):
    return re.findall(r'<div class="sec( open)?" data-sec="([^"]+)"( data-secondaire="1")?>', h)


SEC2 = ' data-secondaire="1"'
e = W.exemple("hourdis")
r = W.resultat(e)
d, h = r["donnees"], r["resultat"]
calc = W._calculer(e, cas(e))
res = calc["res"]
chk("compact : la colonne de gauche est celle de l'établi compact — la ligne du cas ouvert (le plancher se lit dans la liste), puis ses rubriques",
    'class="et-arbre et-arbre-c" data-casplancher="0"' in d and d.count('data-chemin="nom"') == 1
    and '<span class="nom">C1 levage en rive</span><span class="dim">· Hourdis · 7,20' in d, d[:300])
chk("compact : sept rubriques, aux mêmes numéros qu'avant (les messages du moteur y renvoient) ; la première seule ouverte ; « Hypothèses » et "
    "« Affichage et note » secondaires", _rubriques(d) == [(" open", "pl", ""), ("", "ho", ""), ("", "ch", ""), ("", "re", ""), ("", "bi", ""),
                                                           ("", "hy", SEC2), ("", "af", SEC2)], _rubriques(d))
n0 = W.resultat(W.nouvel_etat())
chk("compact : une note neuve ouvre aussi « Bibliothèque et fiches » (aucune charge : on y prend l'engin)",
    [k for o, k, _ in _rubriques(n0["donnees"]) if o] == ["pl", "bi"], _rubriques(n0["donnees"]))
e_nok = W.exemple("hourdis")
mod(e_nok, ["plancher", "hourdis", "M_Rd"], "60")
chk("compact : un cas non conforme ouvre « Répartiteurs » (la proposition de filière y est)",
    [k for o, k, _ in _rubriques(W.resultat(e_nok)["donnees"]) if o] == ["pl", "re"], _rubriques(W.resultat(e_nok)["donnees"]))
chk("compact : le nom du cas ne se saisit plus dans la rubrique « Plancher » ; les explications du type, de la nature et des parts C.1 sont au "
    "survol de ⓘ", 'data-chemin="nom"' not in d.split('data-sec="pl"')[1].split('data-sec="ho"')[0]
    and 'title="Le catalogue d&#x27;un fabricant (Ergon 2018' in d and 'title="Rd : valeurs de CALCUL du catalogue' in d
    and 'title="Figure C.1 de l&#x27;EN 1168' in d and '<div class="grp">Identification de la note PDF<span class="aide" title="' in d, "")
chk("compact : la tête du résultat — le nom du cas, puis le plancher ; plus de phrase sous le titre de la page",
    f'<div class="et-rp"><b>C1 levage en rive</b><span>{C.esc(OP.resume_plancher(cas(e)))}</span></div>' in h and r["phrase"] == "", h[:300])
B = re.findall(r'<section class="et-bloc (\w+)" data-theme="(\d)">.*?<span class="n">([^<]+)</span><span class="e">([^<]+)</span>.*?'
               r'class="et-tout" data-calcul="([\w-]+)"', h, re.S)
chk("compact : trois blocs — « Vérifications », « Appuis de E1 », « Étançons » —, chacun son état et son taux ; « tout le calcul » ouvre une "
    "fenêtre qui existe (celle de la vérification gouvernante, celle de l'élément du premier appui)",
    [(b[1], b[2]) for b in B] == [("0", "Vérifications"), ("1", "Appuis de E1"), ("2", "Étançons")]
    and B[0][3] == f"Conforme, attention · {C.pc(C.pct(res['taux_max']))}{C.NBSP}%" and B[0][0] == "warn" and B[2][0] == "ok"
    and all(f'id="ck-{b[4]}"' in r["volets"] for b in B), B)
chk("compact : le bloc des vérifications dit la règle (ELU et ses coefficients) ; celui d'un engin, son nom",
    '<div class="et-bloc-ef">ELU · γ<sub>G</sub> = 1,35 · γ<sub>Q</sub> = 1,5</div>' in h and '<div class="et-bloc-ef">Télescopique rotatif</div>' in h, "")
CMP = [_texte(x) for x in re.findall(r'<span class="cmp">(.*?)</span>', h)]
chk("compact : chaque famille dit ce qu'elle compare (« V_Ed = 86,02 kN ≤ V_Rd = 90,00 kN ») ; sans taux, la valeur et ce qui manque ; une "
    "information dépassée s'écrit « > »",
    CMP[:7] == ["MEd = 90,36 kN·m ≤ MRd = 120,00 kN·m", "VEd = 86,02 kN ≤ VRd = 90,00 kN", "p = 0,167 N/mm² ≤ padm = 1,50 N/mm²",
                "c = 150 mm · cmax = 150 mm", "γQ·M = 16,04 kN·m ≤ MRd = 50,67 kN·m", "γQ·F = 34,51 kN · à transcrire",
                "Mser = 62,49 kN·m > Mcr = 60,00 kN·m"], CMP[:7])
chk("compact : chaque appui dit sa charge, son répartiteur (il se choisit) et l'élément qui le porte ; une seule ligne gouvernante",
    CMP[7:] == ["porté par H7", "porté par H7", "porté par H2", "porté par H2", "porté par H5"]
    and len(re.findall(r'<b>P\d · \d+\xa0kN</b><button class="prof" data-pk="rep-\d"', h)) == 4 and h.count('class="el vc gov"') == 1, CMP[7:])
chk("compact : chaque vue porte son en-tête technique — le titre, le plancher, les charges posées",
    all('class="et-ent"' in v.get("entete", "") and _texte(OP.resume_plancher(cas(e))) in _texte(v["entete"])
        and "3 objet(s) · 250 kN d'engins et d'étançons (charges de service)" in _texte(v["entete"]) for v in r["vues"])
    and r["vues"][0]["entete"].startswith('<div class="et-ent"><div class="et-ent-t"><div class="l1"><b>PLAN DU PLANCHER</b>'), r["vues"][0].get("entete"))
chk("compact : plus de cadre « Cas de cette note » ; le statut du cas reste dit même seul (une hypothèse peut le passer en « attention » : P-8)",
    r["elements"] == "" and "Conforme, attention" in r["statut_html"])
W.gerer_elements(e, "ajouter")
W.gerer_elements(e, "dupliquer", 0)
chk("compact : copier C1 (par son rang) — la copie s'insère juste après lui, et s'ouvre",
    [c_["nom"] for c_ in e["cas"]] == ["C1 levage en rive", "C1 levage en rive (copie)", "C2"] and e["actif"] == 1, [c_["nom"] for c_ in e["cas"]])

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

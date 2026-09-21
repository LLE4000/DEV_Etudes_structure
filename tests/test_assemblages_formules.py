# -*- coding: utf-8 -*-
"""Formules, substitutions numériques, notation et synthèse (écran et note).

Chaque garantie rougit si on la retire :
  1. SUBSTITUTIONS : sur les 31 cas de référence, chaque substitution
     imprimée (formules.py) est réévaluée avec les nombres tels qu'ils sont
     imprimés et redonne le résultat du moteur (à l'arrondi d'affichage
     près, 1,2 %) ; toute vérification active a un gabarit ; le texte
     rendu contient la forme littérale ET la forme numérique.
  2. NOTATION : ``ec`` écrit hc, c, dc,sup, dc,inf, Δz et ne touche ni aux
     autres mots ni aux formules ; ``ref_courte`` ne répète jamais
     « EN 1993-1-8 » et garde le paragraphe ; les lignes d'alerte sont
     courtes et chiffrées.
  3. SYNTHÈSE : chaque vérification active apparaît exactement une fois
     dans les cinq tableaux ; le taux par élément est le plus grand des
     taux du groupe ; la dimensionnante est dans son tableau.

Lancement : python3 tests/test_assemblages_formules.py
"""
import json
import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from acier.assemblages.poutre_poutre.doubles_cornieres import moteur, formules, notation, synthese  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom
          + (f"   [{info}]" if info and not cond else ""))


with open("acier/reference/reference_double_corniere.json", encoding="utf-8") as fh:
    REF = json.load(fh)

print("=== 1. Substitutions numériques ===")
n = 0
ecarts = []
sans = set()
for cas in REF["cas"]:
    R = moteur.compute(cas["inputs"])
    for c, subs in formules.toutes(R):
        if not subs:
            sans.add(c.key)
        for s in subs:
            if not s.template:
                continue
            n += 1
            ok, v = formules.verifier(s)
            if not ok:
                ecarts.append(f"{cas['id']} {c.key} {s.nom} : moteur {s.res!r}, substitution {v!r}")
chk(f"{n} substitutions réévaluées sur 31 cas, {len(ecarts)} écart(s)", n > 2000 and not ecarts, " || ".join(ecarts[:3]))
chk("toute vérification active a un gabarit de substitution", not sans, str(sorted(sans)))
R0 = moteur.compute({})
t = formules.textes(R0, R0.ck["pdS"])
chk("pdS (dimensionnante) : forme littérale, forme numérique et résultat",
    "Fb,ver,Rd = k1,ver·αb,ver·fu·d·tw/γM2 = 2,5 × 0,5303 × 510 × 20 × 8,5/1,25/10³ = 91,95 kN" in t[0]
    and t[-1].startswith("η = √((Fz/Fb,ver,Rd)² + (Fx/Fb,hor,Rd)²)") and t[-1].endswith("= 74,0 %"), str(t))
t = formules.textes(R0, R0.ck["bv_S"])
chk("bv_S : |NEd| et facteurs d'unités 10³, η = FEd/FRd",
    "|NEd|/n" in t[1] and "10³" in t[0] and t[2] == "η = FEd/FRd = 66,7/188,2 = 35,4 %", str(t))
chk("la substitution ρ borne à 0 comme le moteur (cas VEd = 900)",
    all(formules.verifier(s)[0] for c, subs in formules.toutes(moteur.compute(dict(V_Ed=900))) for s in subs if s.nom == "ρ"))
s = formules.Sub("x", "a*b/1000", dict(a=("a", 2.5), b=("b", 1000.0)), 2.5, "kN")
chk("rendu : « x = a·b = 2,5 × 1000/10³ = 2,5 kN » (facteur d'unité hors forme littérale)",
    formules.rendre(s) == "x = a·b = 2,5 × 1000/10³ = 2,5 kN", formules.rendre(s))

print("\n=== 2. Notation et références ===")
ec = notation.ec
chk("ec : Lc → hc, ln → c, dnt/dnb → dc,sup/dc,inf, dn → dc, décalage → Δz",
    ec("zc + Lc = 50 ; gh + ln ; dnt ≥ tf + r − décalage ; dnb ; dn ≤ h/2") == "zc + hc = 50 ; gh + c ; dc,sup ≥ tf + r − Δz ; dc,inf ; dc ≤ h/2")
chk("ec : ne touche ni « Lc² » (→ hc²), ni « kN », ni « Anv », ni « seln »",
    ec("Wel = tc·Lc²/6 ; 12 kN ; Anv ; seln") == "Wel = tc·hc²/6 ; 12 kN ; Anv ; seln")
restes = []
for cas in REF["cas"]:
    R = moteur.compute(cas["inputs"])
    for c in R.checks:
        for txt in (c.label, c.formula, c.vals):
            e = ec(txt)
            if any(x in (" " + e + " ") for x in (" Lc ", " ln ", " dnt ", " dnb ", "décalage")):
                restes.append(c.key + " : " + e[:60])
    for a in R.alerts:
        for txt in (a.msg, a.why):
            e = ec(txt)
            if any(x in (" " + e + " ") for x in (" Lc ", " ln ", " dnt ", " dnb ", "décalage")):
                restes.append(a.id + " : " + e[:60])
chk("aucun symbole de l'outil ne subsiste dans les textes affichés (31 cas)", not restes, str(restes[:3]))
rc = notation.ref_courte
chk("ref_courte : EN 1993-1-8 implicite, Tab., MSB, P358, ECCS",
    rc("EN 1993-1-8 Tableau 3.4 ; groupe excentré : MSB Part 5 §4.2.1.1") == "Tab. 3.4 · MSB §4.2.1.1"
    and rc("MSB Part 5 §4.2.2 (d'après ECCS n°126)") == "MSB §4.2.2 · ECCS"
    and rc("SCI P358 Check 10 (un seul côté chargé)") == "P358 Ch. 10"
    and rc("EN 1993-1-1 §6.2.5 ; bras de levier : hypothèse de l'outil") == "EC3-1-1 §6.2.5 · hyp. outil",
    rc("EN 1993-1-8 Tableau 3.4 ; groupe excentré : MSB Part 5 §4.2.1.1"))
longues = [rc(c.ref) for c in R0.checks if len(rc(c.ref)) > 30 or "EN 1993-1-8" in rc(c.ref)]
chk("références courtes ≤ 30 caractères, sans « EN 1993-1-8 »", not longues, str(longues))
Rb = moteur.compute(dict(LC_u=260))
a = [x for x in Rb.alerts if x.id == "h_dispo"][0]
titre, det = notation.ligne_alerte(a, Rb)
chk("ligne d'alerte : titre court + première phrase chiffrée en notation Eurocode",
    titre == "Impossible — hauteur disponible insuffisante" and det.startswith("zc + hc = 50 + 260 = 310 mm"), titre + " | " + det)
Rd = moteur.compute(dict(e1S_u=15))
a = [x for x in Rd.alerts if x.id.startswith("dist_")][0]
titre, det = notation.ligne_alerte(a, Rd)
chk("ligne d'alerte Tableau 3.3 : valeur, limite (1,2·d0 = 26,4 mm)",
    titre.startswith("Tab. 3.3 — ") and det == "15,0 mm < 1,2·d0 = 26,4 mm", titre + " | " + det)

print("\n=== 3. Synthèse par élément ===")
manq = []
for cas in REF["cas"]:
    R = moteur.compute(cas["inputs"])
    vus = []
    for cle, _, _ in synthese.TABLES:
        vus += [l["key"] for l in synthese.lignes(R, cle)]
    for l in synthese.matrice_cornieres(R):
        vus += [x["key"] for x in (l["A"], l["B"]) if x]
    actives = [c.key for c in R.checks if c.active]
    if sorted(vus) != sorted(actives) or len(vus) != len(set(vus)):
        manq.append(cas["id"] + " : " + str(sorted(set(actives) ^ set(vus))))
    for nom, eta, ok, pire in synthese.taux_par_element(R):
        grp = {v: k for k, v in synthese.ELEMENTS}[nom]
        on = [c for c in R.checks if c.grp == grp and c.active]
        if abs(eta - max(c.eta for c in on)) > 1e-12 or ok != all(c.ok for c in on) or pire.grp != grp:
            manq.append(cas["id"] + " : taux " + nom)
chk("chaque vérification active apparaît exactement une fois dans les tableaux (31 cas)", not manq, str(manq[:3]))
chk("taux par élément du cas par défaut : Boulons, Cornières, Portée, Porteuse",
    [x[0] for x in synthese.taux_par_element(R0)] == ["Boulons", "Cornières", "Portée", "Porteuse"]
    and abs(synthese.taux_par_element(R0)[2][1] - R0.eta_max) < 1e-12)
m = synthese.matrice_cornieres(R0)
chk("matrice des cornières : pression diamétrale A et B, max = la plus sollicitée",
    m[0]["lab"] == "Pression diamétrale" and m[0]["A"]["key"] == "pdA" and m[0]["B"]["key"] == "pdB"
    and m[0]["max"]["key"] == ("pdA" if R0.ck["pdA"].eta >= R0.ck["pdB"].eta else "pdB"))
chk("refs_fusionnees sans doublon", synthese.refs_fusionnees(["Tab. 3.4 · MSB §4.2.1.1", "Tab. 3.4"]) == "Tab. 3.4 · MSB §4.2.1.1")

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:400])
sys.exit(1 if KO else 0)

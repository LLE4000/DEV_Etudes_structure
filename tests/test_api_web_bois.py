# -*- coding: utf-8 -*-
"""L'API du site pour les poutres bois (api_web/bois.py), vérifiée en Python ordinaire.

- le résultat affiché est celui du moteur bois/poutre : taux, couleur, gouvernant, chiffres clés de l'écran Streamlit ;
- les taux au-delà de 999 % sont plafonnés dans les pastilles et les jauges, exacts dans le détail du calcul ;
- un taux exactement égal à 100 % est conforme (vert foncé), comme le « ≤ 1.0 » de l'écran ;
- les fenêtres de calcul (ELU, ELS) détaillent chaque formule du moteur avec ses valeurs, et les écarts EC5 qui jouent ;
- chaque saisie va au bon endroit ; une saisie hors du domaine de l'écran Streamlit est refusée avec un message, la
  poutre reste intacte (jamais ramenée en silence) ; les champs suivent les bascules (sur mesure, appui) ;
- le sélecteur de sections est trié par masse, marque la retenue ✓ et la plus légère qui tient ★ du moteur ;
- les poutres de la note s'ajoutent, se copient, se suppriment (jamais la dernière) ; le résumé est la plus chargée ;
- les dessins sont à l'échelle et portent les valeurs du moteur ; un nom saisi est échappé partout.

    python3 tests/test_api_web_bois.py
"""
import copy
import json
import os
import re
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

from api_web import bois as B  # noqa: E402
from api_web import commun as C  # noqa: E402
from api_web import dessin_bois as DB  # noqa: E402
from api_web import routeur as R  # noqa: E402
from bois.poutre import moteur as MO  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:300]}"))


def plat(x):
    """Les fragments relus « à plat » pour les comparer à des textes écrits ici : l'espace fine des milliers (C.fr)
    devient une espace, l'apostrophe échappée une apostrophe (les autres échappements restent : § 8 les contrôle)."""
    if isinstance(x, str):
        return x.replace("\u202f", " ").replace("&#x27;", "'")
    if isinstance(x, list):
        return [plat(y) for y in x]
    if isinstance(x, dict):
        return {k: (v if k == "etat" else plat(v)) for k, v in x.items()}
    return x


def req(**k):
    return plat(json.loads(R.traiter(json.dumps(k))))


def saisir(etat, chemin, valeur):
    """Comme l'écran : le chemin lu dans data-chemin, coupé aux « / », la valeur telle que tapée."""
    out = req(module="bois", action="modifier", etat=copy.deepcopy(etat), chemin=chemin.split("/"), valeur=valeur)
    assert out["ok"], out
    return out


def etat_de(**k):
    e = B.nouvel_etat()
    e["poutres"][0].update(k)
    return B.normaliser(e)


LIGNE = re.compile(r'<div class="el( gov)?"( data-calcul="(\w+)")?[^>]*><div class="l1"><b>([^<]+)</b>(.*?)</div>'
                   r'<div class="l2"><div class="bar"><b class="(\w)" style="width:([\d.]+)%"></b></div><span class="pct (\w)">([^<]+)</span>', re.S)


def lignes(html):
    return [{"gov": bool(g), "calcul": cal, "nom": nom, "piece": piece, "barre": n1, "largeur": float(w), "niveau": n2, "txt": t}
            for g, _, cal, nom, piece, n1, w, n2, t in LIGNE.findall(html)]


def niveau_de(v):
    return "r" if v["etat"] == MO.NOK else C.niveau(v["taux"])


# ------------------------------------------------------------------ 1. une note neuve
print("=== 1. Une note neuve : le résultat est celui du moteur ===")
r = req(module="bois", action="nouveau")
chk("nouveau : la requête aboutit", r["ok"], r.get("erreur"))
p0 = dict(MO.nouvelle_poutre(), nom="P1")
chk("la poutre neuve est celle de l'écran Streamlit (C24, SC 1, long terme, 38 × 150, L 4 m, G 2, Q 3…), nommée P1",
    r["etat"] == {"version": 1, "poutres": [p0], "actif": 0}, r["etat"])
ref = MO.verifier(p0)
chk("le résumé : taux plafonné à 999 %, rouge, « Non conforme », gouvernant = flèche instantanée (celui du moteur)",
    r["resume"] == {"taux": 999, "niveau": "r", "statut": "Non conforme", "gouvernant": "P1 · flèche instantanée"}
    and ref["gouvernant"]["cle"] == "fleche_inst", r["resume"])
L0 = lignes(r["resultat"])
attendu = ref["verifications"]
chk("une ligne par vérification du moteur, dans l'ordre de l'écran (flexion, cisaillement, deux flèches)",
    [x["nom"] for x in L0] == [B.COURTS[v["cle"]] for v in attendu] == ["Flexion", "Cisaillement", "Flèche instantanée", "Flèche finale"],
    [x["nom"] for x in L0])
chk("chaque ligne : taux du moteur (arrondi dans sa bande, plafonné à 999 %), couleur du moteur, jauge bornée à 100 %",
    all(x["txt"] == f"{C.pc(C.pct(min(v['taux'], B.PLAFOND)))}{C.NBSP}%" and x["niveau"] == x["barre"] == niveau_de(v) and x["largeur"] <= 100
        for x, v in zip(L0, attendu)), [(x["txt"], v["taux"]) for x, v in zip(L0, attendu)])
chk("la flexion à 782 % (σ = 101,05 MPa sur f_m,d = 12,92 MPa) s'affiche 782 %, pas plafonnée",
    L0[0]["txt"] == f"782{C.NBSP}%" and abs(ref["grandeurs"]["u_flex"] - 7.8195) < 1e-4)
chk("une seule ligne gouvernante, celle du moteur", [x["nom"] for x in L0 if x["gov"]] == ["Flèche instantanée"])
chk("les lignes ELU ouvrent la fenêtre « elu », les flèches la fenêtre « els » ; les deux fenêtres existent",
    [x["calcul"] for x in L0] == ["elu", "elu", "els", "els"] and 'id="ck-elu"' in r["volets"] and 'id="ck-els"' in r["volets"])
chk("le sélecteur de section s'ouvre depuis la ligne « Flexion », et seulement elle", re.findall(r'data-pk="([^"]+)"', r["resultat"]) == ["section"]
    and 'data-pk="section"' in L0[0]["piece"])
grp = re.findall(r'<div class="elgrp">([^<]+)<span class="pct (\w)">([^<]+)</span>', r["resultat"])
chk("deux groupes, « ELU · résistance » et « ELS · flèches », chacun avec son taux le plus fort",
    grp == [("ELU · résistance", "r", f"782{C.NBSP}%"), ("ELS · flèches", "r", f"999{C.NBSP}%")], grp)
g = ref["grandeurs"]
facts = re.findall(r"<dt>(.*?)</dt><dd>(.*?)</dd>", r["resultat"])
chk("les chiffres clés de l'écran Streamlit (ses quatre métriques, M_Ed et V_Ed) avec leurs unités",
    [d for _, d in facts] == ["38 × 150 mm", "4,00 m", "11 000 MPa", "420 kg/m³", "14,40 kN·m", "14,40 kN"], facts)
chk("le verdict : la pastille plafonnée et la comparaison exacte « w_inst = 141 768,6 mm > L/300 = 13,3 mm »",
    '<span class="chip xl r"><small>η</small>999' in r["resultat"] and "141 768,6 mm &gt; L/300 = 13,3 mm" in r["resultat"])
chk("l'écart d'unité des flèches est dit en rouge, avec les valeurs corrigées de l'unité (141,8 et 131,6 mm, non retenues)",
    re.search(r'<div class="msg grave"><b>Flèches de l\'écran Streamlit 1 000 fois trop grandes</b>.*?141,8 mm.*?131,6 mm '
              r'\(à titre indicatif, non retenues\)', r["resultat"]) is not None)
chk("le poids propre non compté est dit (≈ 0,023 kN/m pour 38 × 150 en C24)", "Poids propre non compté (≈ 0,023 kN/m)" in r["resultat"])
chk("le pied : la masse indicative du bois (ρ_mean·A·L = 9,6 kg) et la légende des couleurs",
    f"9,6{C.NBSP}kg" in r["resultat"] and C.legende() in r["resultat"])

# ------------------------------------------------------------------ 2. les fenêtres de calcul
print("\n=== 2. Les fenêtres de calcul : chaque formule du moteur, valeurs substituées ===")
elu, els = r["volets"].split('id="ck-els"')
n_elu = sum(len(v["formules"]) for v in attendu if v["cle"] in B.ELU) - 1     # V_Ed partagée : une ligne de moins (q_ELU, I…)
chk("la fenêtre ELU détaille les formules de la flexion et du cisaillement", elu.count('class="fx"') >= 11, elu.count('class="fx"'))
chk("… q_ELU = 1,35 × 2 + 1,5 × 3 = 7,20 kN/m ; M_Ed = 7,2 × 4² / 8 = 14,40 kN·m ; σ = 14,4 × 10⁶ / 142 500 = 101,05 MPa",
    "1,35 × 2 + 1,5 × 3" in elu and "7,20 kN/m" in elu and "7,2 × 4² / 8" in elu and "14,4 × 10⁶ / 142 500" in elu and "101,05 MPa" in elu)
chk("… f_m,d = 0,7 × 24 × 1 / 1,3 = 12,92 MPa, et η = 101,05 / 12,92 = 782,0 % en pastille rouge (exact, non plafonné)",
    "0,7 × 24 × 1 / 1,3" in elu and "12,92 MPa" in elu and "101,05 / 12,92" in elu and f'<span class="chip r">782,0{C.NBSP}%</span>' in elu)
chk("la fenêtre ELS écrit la flèche de l'écran telle qu'il la calcule : 5 × (5 × 10³) × (4 × 10³)⁴ / (384 × 11 000 × 10 687 500)",
    "5 × (5 × 10³) × (4 × 10³)⁴ / (384 × 11 000 × 10 687 500)" in els and "141 768,6 mm" in els)
chk("… et le taux exact de la flèche instantanée : 1 063 264 %", f"1 063 264{C.NBSP}%" in els)
chk("le tableau de chaque fenêtre : une ligne par vérification, valeur et limite écrites, le taux plafonné en pastille",
    elu.count("<tr class=") == 2 and els.count("<tr class=") == 2 and "σ<sub>m,Ed</sub> = 101,05 MPa" in elu and "f<sub>m,d</sub> = 12,92 MPa" in elu
    and "L/300 = 13,3 mm" in els)
chk("les écarts EC5 qui jouent sont au pied des fenêtres : k_cr à l'ELU, l'unité et la flèche finale à l'ELS ; pas l'appui (décoché)",
    "k<sub>cr</sub> = 0,67" in elu and "1 000 fois trop grandes" in els and "(1 − ψ₂)" in els and "NON pondérée" not in elu)
r_ap = req(module="bois", action="modifier", etat=r["etat"], chemin=["appui"], valeur=True)
elu_ap = r_ap["volets"].split('id="ck-els"')[0]
chk("appui coché : une ligne « Compression à l'appui » (a = 60 mm), son détail R = (G + Q)·L / 2, et l'écart « non pondérée »",
    [x["nom"] for x in lignes(r_ap["resultat"])][2] == "Compression à l'appui" and "a = 60 mm" in r_ap["resultat"]
    and "5 × 4 / 2" in elu_ap and "NON pondérée" in elu_ap, [x["nom"] for x in lignes(r_ap["resultat"])])

# ------------------------------------------------------------------ 3. la saisie
print("\n=== 3. La saisie : chaque champ au bon endroit, les refus ===")
e0 = r["etat"]
for chemin, valeur, cle, attendu_v in (("L", "4,5", "L", 4.5), ("G", "1 2", "G", 12.0), ("Q", "0", "Q", 0.0), ("gamma_G", "1,5", "gamma_G", 1.5),
                                       ("psi2", "0,6", "psi2", 0.6), ("classe", "C30", "classe", "C30"), ("service", "3", "service", 3),
                                       ("duree", "court", "duree", "court"), ("section", "63x200", "section", "63x200"),
                                       ("lim_inst", "L/400", "lim_inst", "L/400"), ("lim_fin", "L/300", "lim_fin", "L/300"),
                                       ("nom", "Solive S3", "nom", "Solive S3")):
    out = saisir(e0, chemin, valeur)
    p = out["etat"]["poutres"][0]
    chk(f"« {chemin} » = « {valeur} » → {attendu_v!r}, sans message, le reste intact",
        out["message"] is None and p[cle] == attendu_v and {k: v for k, v in p.items() if k != cle} == {k: v for k, v in e0["poutres"][0].items() if k != cle})
mes = saisir(e0, "standard", "non")
chk("« Sur mesure » : les champs b et h (80 × 240, les défauts de l'écran) remplacent le menu des sections",
    'data-chemin="b"' in mes["donnees"] and 'data-chemin="h"' in mes["donnees"] and 'data-chemin="section"' not in mes["donnees"]
    and "80 × 240" in mes["resultat"])
chk("… et la section sur mesure se dit dans la ligne « Flexion »", "80 × 240 sur mesure · C24" in mes["resultat"])
chk("la largeur d'appui est grisée tant que l'appui n'est pas coché, active ensuite (comme l'écran)",
    re.search(r'data-chemin="a"[^>]*disabled', r["donnees"]) is not None and re.search(r'data-chemin="a"[^>]*disabled', r_ap["donnees"]) is None)
REFUS = (("L", "0,2", "entre 0,5 et 12 m"), ("L", "12,5", "entre 0,5 et 12 m"), ("G", "-1", "entre 0 et 50 kN/m"), ("Q", "51", "entre 0 et 50 kN/m"),
         ("psi2", "0", "entre 0,1 et 1"), ("gamma_G", "0,9", "entre 1 et 2"), ("gamma_Q", "2,5", "entre 1 et 2"), ("a", "20", "entre 30 et 200 mm"),
         ("L", "abc", "Nombre illisible"), ("service", "4", "n'est pas proposé"), ("service", "1,5", "illisible"),
         ("classe", "C45", "n'est pas proposé"), ("duree", "longue", "n'est pas proposé"), ("section", "50x200", "n'est pas proposé"),
         ("lim_inst", "L/500", "n'est pas proposé"), ("standard", "peut-être", "vrai ou faux"), ("inconnu", "1", "Champ inconnu"))
mal = []
for chemin, valeur, texte in REFUS:
    out = saisir(e0, chemin, valeur)
    if not (out["message"] and texte in out["message"] and out["etat"] == e0):
        mal.append((chemin, valeur, out["message"]))
mes_b = saisir(mes["etat"], "b", "20")
mes_h = saisir(mes["etat"], "h", "600")
if not (mes_b["message"] and "entre 30 et 300 mm" in mes_b["message"] and mes_b["etat"] == mes["etat"]):
    mal.append(("b", "20", mes_b["message"]))
if not (mes_h["message"] and "entre 80 et 500 mm" in mes_h["message"]):
    mal.append(("h", "600", mes_h["message"]))
chk(f"{len(REFUS) + 2} saisies hors du domaine de l'écran Streamlit refusées avec un message clair ; la poutre reste intacte (jamais ramenée)",
    not mal, mal)
print("     ex. :", saisir(e0, "L", "0,2")["message"])
chk("deux niveaux de chemin (« L/x ») sont refusés, pas une exception", saisir(e0, "L/x", "1")["message"].startswith("Champ inconnu"))

# ------------------------------------------------------------------ 4. le taux de 100 % exactement
print("\n=== 4. Les bornes de couleur ===")
e100 = etat_de(classe="C14", service=3, duree="moyen", section="100x250", G=1.0, Q=2.0, L=2.0, appui=True, a=30.0)
r100 = req(module="bois", action="calculer", etat=e100)
l100 = [x for x in lignes(r100["resultat"]) if x["nom"] == "Compression à l'appui"][0]
v100 = next(v for v in MO.verifier(e100["poutres"][0])["verifications"] if v["cle"] == "appui")
chk("compression à l'appui EXACTEMENT 100 % : conforme pour l'écran (≤ 1,0), vert foncé « f » sur le site, affiché 100 %",
    v100["taux"] == 1.0 and v100["etat"] == MO.OK and l100["niveau"] == "f" and l100["txt"] == f"100{C.NBSP}%", (v100["taux"], l100))
e_ok = etat_de(section="75x225", L=0.8, G=0.02, Q=0.0, classe="C30")
r_ok = req(module="bois", action="calculer", etat=e_ok)
chk("une poutre qui tient partout : « Conforme », vert pâle, aucune ligne rouge",
    r_ok["resume"]["niveau"] == "p" and r_ok["resume"]["statut"] == "Conforme" and all(x["niveau"] == "p" for x in lignes(r_ok["resultat"])),
    (r_ok["resume"], [x["txt"] for x in lignes(r_ok["resultat"])]))

# ------------------------------------------------------------------ 5. le sélecteur
print("\n=== 5. Le sélecteur de sections ===")
s = req(module="bois", action="selecteur", etat=e_ok, role="section")
rows = re.findall(r'<button class="pk-row( cur)?( opt)?" data-choix="([^"]+)" data-prop="([^"]*)"><span class="mk">.*?</span>'
                  r'<span class="nm">([^<]+)</span><span class="kg">([^<]+)</span><span class="chip (\w)">([^<]+)</span>', s["html"])
cands = MO.candidats_sections(e_ok["poutres"][0])
chk("onze sections, dans l'ordre du moteur (par aire, donc par masse) ; la colonne est la masse en kg/m, croissante",
    [x[2] for x in rows] == [B._tag(c["nom"]) for c in cands] and len(rows) == 11
    and [float(x[5].replace(",", ".")) for x in rows] == sorted(float(x[5].replace(",", ".")) for x in rows), [x[2] for x in rows])
chk("la retenue ✓ (75 × 225) et la plus légère qui tient ★ du moteur sont marquées",
    [x[2] for x in rows if x[0]] == ["75 × 225"] and [x[2] for x in rows if x[1]] == [B._tag(MO.optimal(e_ok["poutres"][0])["nom"])], rows[:2])
chk("chaque ligne : le taux du moteur (plafonné) et sa couleur ; les propriétés A, I, W et la vérification qui gouverne",
    all(x[7] == f"{C.pc(C.pct(min(c['taux'], B.PLAFOND)))}{C.NBSP}%" and x[6] == ("r" if not c["ok"] else C.niveau(c["taux"]))
        and "A = " in x[3] and "cm⁴" in x[3] and "gouverne" in x[3] for x, c in zip(rows, cands)))
opt = MO.optimal(e_ok["poutres"][0])
chk("la ligne « Flexion » porte l'étoile de la plus légère qui tient quand elle n'est pas la retenue",
    f"Plus légère qui tient : {B._tag(opt['nom'])}" in r_ok["resultat"])
ch = req(module="bois", action="choisir", etat=e_ok, role="section", nom=B._tag(opt["nom"]))
chk(f"choisir « {B._tag(opt['nom'])} » la retient (l'étiquette du sélecteur est relue) ; l'étoile disparaît",
    ch["etat"]["poutres"][0]["section"] == opt["nom"] and ch["etat"]["poutres"][0]["standard"] is True and 'class="leger"' not in ch["resultat"])
ch2 = req(module="bois", action="choisir", etat=mes["etat"], role="section", nom="63 × 175")
chk("choisir une section depuis une poutre sur mesure la repasse en « standard »",
    ch2["etat"]["poutres"][0]["standard"] is True and ch2["etat"]["poutres"][0]["section"] == "63x175" and ch2["etat"]["poutres"][0]["b"] == 80.0)
s0 = req(module="bois", action="selecteur", etat=e0, role="section")
chk("aucune section ne tient (la note neuve) : pas d'étoile, et le sous-titre le dit", "aucune ne tient" in s0["html"] and 'pk-row opt' not in s0["html"]
    and 'pk-row cur' in s0["html"])

# ------------------------------------------------------------------ 6. les poutres de la note
print("\n=== 6. Les poutres de la note ===")
a = req(module="bois", action="ajouter", etat=e_ok)
chk("« Ajouter une poutre » : P2 aux défauts de l'écran, ouverte", [p["nom"] for p in a["etat"]["poutres"]] == ["P1", "P2"] and a["etat"]["actif"] == 1)
d = req(module="bois", action="dupliquer", etat=a["etat"])
chk("« Copier » : la poutre ouverte est copiée, « (copie) »", [p["nom"] for p in d["etat"]["poutres"]] == ["P1", "P2", "P2 (copie)"])
chk("le résumé de la note est sa poutre la plus chargée (P2 à 999 %, pas P1 conforme)",
    d["resume"]["gouvernant"].startswith("P2") and d["resume"]["niveau"] == "r")
act = req(module="bois", action="activer", etat=d["etat"], index=0)
chk("« activer » ouvre P1 : ses données et son résultat", act["etat"]["actif"] == 0 and "75 × 225" in act["resultat"] and 'value="P1"' in act["donnees"])
sup = req(module="bois", action="supprimer", etat=act["etat"], index=0)
chk("supprimer P1 : il reste P2 et sa copie", [p["nom"] for p in sup["etat"]["poutres"]] == ["P2", "P2 (copie)"])
dern = req(module="bois", action="supprimer", etat=e0, index=0)
chk("la dernière poutre ne se supprime pas", len(dern["etat"]["poutres"]) == 1)
items = re.findall(r'<div class="ni( on)?" data-activer="(\d)"><span class="t">([^<]+)<span>([^<]+)</span></span><span class="pct (\w)">([^<]+)</span>',
                   d["elements"])
chk("la liste « Dans cette note » : nom, dimensions, classe, taux et couleur de chaque poutre",
    [(x[2].strip(), x[3]) for x in items] == [("P1", "· 75 × 225 · C30"), ("P2", "· 38 × 150 · C24"), ("P2 (copie)", "· 38 × 150 · C24")]
    and [x[4] for x in items] == ["p", "r", "r"], items)

# ------------------------------------------------------------------ 7. les dessins
print("\n=== 7. Les dessins : à l'échelle, les valeurs du moteur ===")
vues = {v["cle"]: v for v in r_ap["vues"]}
chk("trois vues : schéma et efforts, déformée, section", list(vues) == ["poutre", "deformee", "section"])
rp = MO.verifier(r_ap["etat"]["poutres"][0])
k = DB.LARGEUR / rp["geometrie"]["L_mm"]
rect = re.search(r'<rect x="([\d.]+)" y="[\d.]+" width="([\d.]+)" height="([\d.]+)" fill="' + DB.BOIS, vues["poutre"]["svg"])
chk("schéma : la poutre à l'échelle de la portée — hauteur h·k, longueur L·k + un débord a/2 de chaque côté (4 unités au moins)",
    rect and abs(float(rect.group(3)) - 150 * k) < 0.06 and abs(float(rect.group(2)) - (4000 * k + 2 * max(30 * k, 4.0))) < 0.06,
    rect.groups() if rect else None)
pads = re.findall(r'<rect x="[\d.]+" y="[\d.]+" width="([\d.]+)" height="6" fill="' + DB.ARDOISE, vues["poutre"]["svg"])
chk("schéma : les deux appuis de largeur a à la même échelle (appui coché) ; aucun sinon",
    len(pads) == 2 and all(abs(float(w) - 60 * k) < 0.06 for w in pads) and DB.ARDOISE + '"/>' not in r["vues"][0]["svg"].split("height=\"6\"")[0][-60:])
for txt in ("Q = 3 kN/m", "G = 2 kN/m", "= 14,40 kN", "= 14,40 kN·m", "L = 4 000", "1,35 × 2 + 1,5 × 3 = 7,20 kN/m", "+14,40", "−14,40"):
    if txt not in vues["poutre"]["svg"]:
        chk(f"schéma : « {txt} » écrit", False)
chk("schéma : charges, réactions, V et M aux valeurs du moteur, portée cotée", all(t in vues["poutre"]["svg"] for t in (
    "Q = 3 kN/m", "G = 2 kN/m", "= 14,40 kN", "= 14,40 kN·m", "L = 4 000", "1,35 × 2 + 1,5 × 3 = 7,20 kN/m", "+14,40", "−14,40")))
chk("déformée : les flèches et taux de l'écran dans la légende (141 768,6 mm · 1 063 264 %)",
    "141 768,6 mm" in vues["deformee"]["svg"] and "1 063 264 %" in vues["deformee"]["svg"] and "L/300" in vues["deformee"]["svg"])
sec = re.search(r'<rect x="[\d.]+" y="[\d.]+" width="([\d.]+)" height="([\d.]+)" fill="' + DB.BOIS_CLAIR, vues["section"]["svg"])
chk("section : le rectangle a la proportion b / h exacte (38 / 150)", sec and abs(float(sec.group(1)) / float(sec.group(2)) - 38 / 150) < 0.002,
    sec.groups() if sec else None)
chk("section : σ_m,Ed, f_m,d, τ_Ed, f_v,d du moteur sous leurs diagrammes", all(t in vues["section"]["svg"] for t in ("= 101,05", "= 12,92", "= 3,79", "= 2,15")))
chk("chaque dessin est un SVG à viewBox, sans script ni ressource externe",
    all(v["svg"].startswith("<svg id=") and "viewBox" in v["svg"] and "<script" not in v["svg"] and "href" not in v["svg"].replace("url(#", "")
        for v in r_ap["vues"]))
ids = re.findall(r'clipPath id="([^"]+)"', vues["section"]["svg"])
chk("le détourage des cernes porte un identifiant propre à la section (deux dessins d'une page ne se mêlent pas)",
    ids and "38x150" in ids[0])
chk("la phrase de la première vue : section, classe, portée, q_ELU", r["phrase"] == "38 × 150 mm · C24 · L = 4,00 m · q<sub>ELU</sub> = 7,20 kN/m", r["phrase"])

# ------------------------------------------------------------------ 8. échappement
print("\n=== 8. Échappement ===")
piege = '<img src=x onerror=alert(1)>'
pe = saisir(e0, "nom", piege)
fr = [pe["resultat"], pe["volets"], pe["donnees"], pe["elements"], "".join(v["svg"] for v in pe["vues"]), pe["phrase"]]
chk("un nom piégé est échappé dans tous les fragments (données, éléments, fenêtres)", not any("<img" in f for f in fr)
    and "&lt;img" in pe["elements"] and "&lt;img" in pe["volets"])
abime = req(module="bois", action="calculer", etat={"version": 1, "poutres": [{"classe": "C99"}], "actif": 0})
chk("une poutre abîmée dans un fichier .stab : une erreur lisible, jamais une exception",
    abime["ok"] is False and "classe" in abime["erreur"], abime.get("erreur"))

# ------------------------------------------------------------------ 9. ce que la relecture a corrigé
print("\n=== 9. Relecture : défaut non conservatif de plus, messages, petits nombres, dessins réduits, robustesse ===")
eG = etat_de(section="63x200", duree="moyen", L=3.0, G=3.0, Q=0.5)
rG = req(module="bois", action="calculer", etat=eG)
cgG = MO.verifier(eG["poutres"][0])["indicatif"]["combinaison_G"]
eluG = rG["volets"].split('id="ck-els"')[0]
chk("combinaison 1,35·G seule (G domine, moyen terme) : un message ROUGE dit qu'elle gouverne, avec le taux du moteur (104 %), non retenu",
    re.search(r'<div class="msg grave"><b>Combinaison γ<sub>G</sub>·G seule non vérifiée par l\'écran Streamlit</b>.*?k<sub>mod</sub> permanent = 0,6'
              r'.*?flexion ' + re.escape(B._pc(cgG["u_flex"])) + r'.*?non retenu', rG["resultat"]) is not None
    and B._pc(cgG["u_flex"]).startswith("103,7"), B._pc(cgG["u_flex"]))
chk("… et le pied du calcul ELU le chiffre (q = 4,05 kN/m, flexion et cisaillement) ; le verdict reste celui de l'écran Streamlit",
    "à titre indicatif : q = γ<sub>G</sub>·G = 4,05 kN/m" in eluG and B._pc(cgG["u_shear"]) in eluG
    and [x["txt"] for x in lignes(rG["resultat"])][0] == f"92{C.NBSP}%")
chk("… ce message vient EN TÊTE des messages (avant l'écart d'unité des flèches) : le défaut non conservatif d'abord",
    0 <= rG["resultat"].find("Combinaison γ<sub>G</sub>·G seule") < rG["resultat"].find("Flèches de l'écran Streamlit 1 000 fois"))
rGt = req(module="bois", action="calculer", etat=etat_de(section="63x200", duree="moyen", L=0.5, G=3.0, Q=0.5))
cgT = MO.verifier(rGt["etat"]["poutres"][0])["indicatif"]["combinaison_G"]
chk("une combinaison G seule qui gouverne mais reste petite (L = 0,5 m : quelques %) : pas de message, ses chiffres au pied du calcul ELU",
    "Combinaison γ" not in rGt["resultat"] and cgT["u_flex"] < 0.95 and B._pc(cgT["u_flex"]) in rGt["volets"], B._pc(cgT["u_flex"]))
rGp = req(module="bois", action="calculer", etat=etat_de(section="63x200", duree="permanent", L=3.0, G=3.0, Q=0.5))
chk("durée « permanente » : ni message ni écart « combinaison G » (l'écran la couvre déjà)",
    "Combinaison γ" not in rGp["resultat"] and "γ_G·G seule" not in rGp["volets"] and "γ<sub>G</sub>·G seule" not in rGp["volets"])
chk("la note neuve (Q domine) : l'écart est cité au pied du calcul ELU, mais aucun message (la combinaison ne gouverne pas)",
    "Combinaison γ" not in r["resultat"] and "γ<sub>G</sub>·G seule" in elu)
chk("le rappel de l'écran Streamlit « ajuster la base et les facteurs selon l'Annexe nationale » est sous le résultat",
    "selon l'Annexe nationale" in r["resultat"])
chk("aucun champ ne porte de borne locale (data-min / data-max) : le refus vient du moteur, avec son domaine, et le champ revient à la "
    "valeur retenue", "data-min" not in r["donnees"] + mes["donnees"] + r_ap["donnees"] and "data-max" not in r["donnees"] + mes["donnees"])
mL = saisir(e0, "L", "0,2")["message"]
chk("le message de refus dit le domaine en clair : « La portée L doit rester entre 0,5 et 12 m, comme dans l'écran Streamlit (reçu 0,2) »",
    mL == "Saisie refusée : la portée L doit rester entre 0,5 et 12 m, comme dans l'écran Streamlit (reçu 0,2)", mL)
chk("un petit effort garde deux chiffres significatifs : M_Ed = 0,0022 kN·m (pas « 0,00 ») dans les chiffres clés et sur le schéma",
    "<dt>M<sub>Ed</sub></dt><dd>0,0022 kN·m</dd>" in r_ok["resultat"] and "= 0,0022 kN·m" in r_ok["vues"][0]["svg"]
    and "0,00 kN" not in r_ok["vues"][0]["svg"])
r0 = req(module="bois", action="calculer", etat=etat_de(G=0.0, Q=0.0))
chk("sans charge : V s'écrit 0, jamais « −0,00 »", "−0,00" not in r0["vues"][0]["svg"] and "+0,00" not in r0["vues"][0]["svg"])
rH = req(module="bois", action="calculer", etat=etat_de(standard=False, b=60.0, h=500.0, L=0.5, G=50.0, Q=50.0, appui=True, a=200.0))
svgH = rH["vues"][0]["svg"]
hH = re.search(r'<rect x="[\d.]+" y="[\d.]+" width="[\d.]+" height="([\d.]+)" fill="' + DB.BOIS, svgH)
padsH = re.findall(r'<rect x="[\d.]+" y="[\d.]+" width="([\d.]+)" height="6" fill="' + DB.ARDOISE, svgH)
chk("poutre haute sur portée courte (500 mm sur 0,5 m, a = 200 mm) : hauteur et appuis RÉDUITS pour la lisibilité, et c'est écrit",
    hH and float(hH.group(1)) == DB.HB_MAX and all(abs(float(w) - DB.APPUI_MAX * DB.LARGEUR) < 0.06 for w in padsH) and len(padsH) == 2
    and "dessin réduit pour la lisibilité : h = 500 mm, a = 200 mm" in svgH, (hH.group(1) if hH else None, padsH))
chk("… et une poutre courante n'est pas réduite (pas de mention)", "dessin réduit" not in r["vues"][0]["svg"])
chk("les étiquettes posées sur un trait (L/300, L/200, L/2, compression, traction) sont détourées de blanc",
    all(re.search(r'paint-order="stroke"[^>]*>' + re.escape(t) + "<", svg) for t, svg in
        (("L/300", r["vues"][1]["svg"]), ("L/200", r["vues"][1]["svg"]), ("L/2", r["vues"][0]["svg"]),
         ("compression", r["vues"][2]["svg"]), ("traction", r["vues"][2]["svg"]))))
chk("les écarts du pied des fenêtres écrivent leurs indices (k<sub>cr</sub>, w<sub>inst,Q</sub>, G<sub>mean</sub>)",
    "k<sub>cr</sub>" in elu and "w<sub>inst,Q</sub>" in els and "G<sub>mean</sub>" in els)
inf = req(module="bois", action="calculer", etat=dict(e0, actif=float("inf")))
txt = req(module="bois", action="activer", etat=e0, index="abc")
chk("une note dont le rang actif est infini, un rang « abc » : relus (poutre 1), jamais une erreur",
    inf["ok"] and inf["etat"]["actif"] == 0 and txt["ok"] and txt["etat"]["actif"] == 0, (inf.get("erreur"), txt.get("erreur")))
pg = saisir(e0, "classe", piege)
chk("un texte piégé dans un menu (classe) : refusé, la poutre intacte, aucun fragment ne le porte brut",
    pg["message"] and pg["etat"] == e0 and not any("<img" in x for x in (pg["resultat"], pg["volets"], pg["donnees"], pg["elements"])))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:240])
sys.exit(1 if KO else 0)

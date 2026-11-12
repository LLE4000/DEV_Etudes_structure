# -*- coding: utf-8 -*-
"""L'API du site pour le choix de profilé (api_web/profil.py), vérifiée en Python ordinaire.

- une note neuve est celle de l'écran Streamlit à son ouverture (HEA, S235, aucun effort) ;
- le résultat affiché est celui du moteur acier/choix_profil : le verdict (taux, couleur, vérification gouvernante et sa
  comparaison), une ligne par vérification avec son taux et sa couleur, le profilé retenu, l'optimal de l'écran et le
  plus léger qui tient ;
- la fenêtre de calcul porte chaque formule du moteur, son résultat écrit comme l'écran l'écrit (kN au dixième, taux à
  10⁻³), les hypothèses ; la vue « Tableau » porte chaque ligne du tableau de l'écran, colorée comme lui ;
- les avertissements du moteur sont affichés, en rouge quand ils sont non conservatifs ;
- chaque saisie va au bon endroit ; une saisie hors du domaine de l'écran est refusée avec un message et l'élément reste
  intact ; les cases des familles, la case « Suivre l'optimal » ;
- le sélecteur : trié par masse, le retenu ✓ et le plus léger qui tient ★, les familles en puces, le filtre I_y min ;
  un profilé d'une autre famille ajoute sa famille ;
- plusieurs éléments par note ; le résumé est le plus chargé ;
- un nom saisi est échappé partout ; aucune ressource externe, aucun script.

    python3 tests/test_api_web_profil.py
"""
import json
import os
import re
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

from acier.choix_profil import calcul as CA  # noqa: E402
from acier.choix_profil import entree as EN  # noqa: E402
from acier.choix_profil import moteur as MO  # noqa: E402
from api_web import commun as C  # noqa: E402
from api_web import profil as P  # noqa: E402
from api_web import routeur as R  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:300]}"))


def req(**k):
    return json.loads(R.traiter(json.dumps(dict(module="profil", **k))))


def saisir(etat, chemin, valeur):
    """Comme l'écran : le chemin lu dans data-chemin, coupé aux « / », la valeur telle que tapée."""
    out = req(action="modifier", etat=etat, chemin=chemin.split("/"), valeur=valeur)
    assert out["ok"], out
    return out


def note(**k):
    e = EN.nouvel_element("P1")
    e.update(k)
    return {"version": 1, "elements": [e], "actif": 0}


def texte(h):
    """Le texte d'un fragment HTML, sans balises ni espaces insécables."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", h).replace(C.NBSP, " ").replace("\u202f", " ")).strip()


LIGNE = re.compile(r'<div class="el vc( gov)?"[^>]*><div class="l1"><b>([^<]+)</b>.*?<span class="pct (\w)">([^<]+)</span>', re.S)

# ------------------------------------------------------------------ 1. la note neuve
print("=== 1. La note neuve : l'écran Streamlit à son ouverture")
r = req(action="nouveau")
chk("nouveau : la requête aboutit", r["ok"], r.get("erreur"))
el0 = r["etat"]["elements"][0]
chk("la note neuve : un élément P1 aux valeurs de l'écran (HEA, S235, aucun effort, γ = 1, courbe b)",
    el0 == EN.nouvel_element("P1") and el0["familles"] == ["HEA"] and el0["acier"] == "S235" and el0["courbe"] == "b", el0)
res0 = MO.verifier(el0)
chk("sans effort, l'écran retient la première ligne (HEA 100) à 0 % ; le résultat le dit", res0["retenu"] == "HEA 100"
    and r["resume"]["taux"] == 0 and "HEA 100" in r["resultat"] and "Aucun effort saisi" in r["resultat"], r["resume"])
chk("trois vues : la section, le tableau, taux / masse (pas de flambement sans compression)",
    [v["cle"] for v in r["vues"]] == ["section", "tableau", "courbe"], [v["cle"] for v in r["vues"]])

# ------------------------------------------------------------------ 2. un poteau : le résultat est celui du moteur
print("\n=== 2. Un poteau (HEA + HEB, N = 600 kN, M_y = 60 kN·m, flambement courbe c) : le résultat du moteur")
ETAT = note(familles=["HEA", "HEB"], My=60.0, Vz=40.0, avec_N=True, N=600.0, flambement=True, Lcr_y=6.0, Lcr_z=3.0, courbe="c")
r = req(action="calculer", etat=ETAT)
m = MO.verifier(ETAT["elements"][0])
chk("le calcul aboutit", r["ok"], r.get("erreur"))
chk(f"le retenu est l'optimal de l'écran ({m['optimal']}) et le plus léger qui tient ({m['plus_leger']}) en diffère",
    m["retenu"] == m["optimal"] == "HEB 180" and m["plus_leger"] == "HEA 220", (m["retenu"], m["optimal"], m["plus_leger"]))
chk("le résumé de la note : le taux du moteur arrondi sans changer de bande, sa couleur, son gouvernant",
    r["resume"] == {"taux": C.pct(m["taux_max"]), "niveau": C.niveau(m["taux_max"]), "statut": C.STATUTS[C.niveau(m["taux_max"])][1],
                    "gouvernant": "P1 · HEB 180 · interaction N + M"}, r["resume"])
verdict = texte(re.search(r'<div class="verdict"[^>]*>.*?</div></div>', r["resultat"], re.S).group(0))
chk("le verdict : η 99 %, « HEB 180 · interaction N + M », η_N + η_My + η_Mz = 0,391 + 0,600 + 0,000 = 0,991 ≤ 1",
    "99 %" in verdict and "HEB 180 · interaction N + M" in verdict and "0,391 + 0,600 + 0,000 = 0,991 ≤ 1" in verdict, verdict)
lignes = {nom: (bool(gov), niv, pc) for gov, nom, niv, pc in LIGNE.findall(r["resultat"])}
v = {x["cle"]: x for x in m["verifications"]}
attendu = {"Profilé retenu": (False, C.niveau(m["taux_max"]), f"{C.pc(C.pct(m['taux_max']))} %"),
           "Effort normal": (False, C.niveau(v["N"]["taux"]), f"{C.pc(C.pct(v['N']['taux']))} %"),
           "Flexion autour de y": (False, C.niveau(v["My"]["taux"]), f"{C.pc(C.pct(v['My']['taux']))} %"),
           "Interaction N + M": (True, C.niveau(v["NM"]["taux"]), f"{C.pc(C.pct(v['NM']['taux']))} %"),
           "Cisaillement selon z": (False, C.niveau(v["Vz"]["taux"]), f"{C.pc(C.pct(v['Vz']['taux']))} %"),
           "Flambement": (False, C.niveau(v["flambement"]["taux"]), f"{C.pc(C.pct(v['flambement']['taux']))} %")}
chk("une ligne par vérification affichée (N, M_y, N + M, V_z, flambement), son taux et sa couleur du moteur ; l'interaction gouverne",
    {k: (g, n, p.replace(C.NBSP, " ")) for k, (g, n, p) in lignes.items()} == attendu, lignes)
chk("M_z et V_y non saisis : pas de ligne ; l'interaction est montrée car N est saisi", "Flexion autour de z" not in lignes
    and "Cisaillement selon y" not in lignes)
RETENU = texte(re.search(r'<div class="el vc[^"]*"[^>]*><div class="l1"><b>Profilé retenu</b>.*?</div></div>', r["resultat"], re.S).group(0))
chk("la ligne du retenu dit sa masse et le plus léger qui tient (« 51,2 kg/m · plus léger : HEA 220, 50,5 kg/m »)",
    "HEB 180" in RETENU and "51,2 kg/m · plus léger : HEA 220, 50,5 kg/m" in RETENU, RETENU)
chk("l'optimal de l'écran (HEB 180 · 51,2 kg/m · 99 %) est dit là où il se décide : sous la case « Suivre l'optimal » des données",
    "Optimal de l'écran : HEB 180 · 51,2 kg/m · 99 %." in texte(r["donnees"]), texte(r["donnees"])[-500:])
chk("le bloc dit la sélection : 48 profilés comparés (HEA, HEB) · 38 tiennent",
    '<div class="et-bloc-ef">48 profilés comparés (HEA, HEB) · 38 tiennent</div>' in r["resultat"], texte(r["resultat"])[:400])
chk("la ligne du profilé ouvre le sélecteur (data-pk) et son étoile nomme le plus léger qui tient",
    'data-pk="profil"' in r["resultat"] and "Le plus léger qui tient : HEA 220 · 89 %" in r["resultat"])
chk("chaque ligne ouvre la fenêtre de calcul du profilé", len(re.findall(r'class="el vc[^"]*" data-calcul="s0"', r["resultat"])) == 6
    and 'id="ck-s0"' in r["volets"])
chk("le pied : la masse du profilé retenu (51,2 kg/m) et la légende des couleurs", "51,2" in texte(r["resultat"].split('class="pfoot"')[1])
    and C.legende() in r["resultat"])

# ------------------------------------------------------------------ 3. la fenêtre de calcul
print("\n=== 3. La fenêtre de calcul : les formules du moteur, écrites comme l'écran les écrit")
fen = r["volets"]
ft = texte(fen)
Rz = m["profil"]["resistances"]
for k, u in (("N_Rd", "kN"), ("My_Rd", "kN·m"), ("Vz_Rd", "kN")):
    s = f"{Rz[k]:,.1f}".replace(",", " ").replace(".", ",")  # texte() ramène toutes les espaces à « »
    chk(f"{k} = {s} {u} : la valeur de l'écran (« :.1f »)", f"{s} {u}" in ft, s)
rt = m["profil"]["ratios"]
chk("les taux à 10⁻³ comme l'écran (η_N, η_My, η_Vz, η_NM, η_section)",
    all(f"{rt[k]:.3f}".replace(".", ",") in ft for k in ("rN", "rMy", "rVz", "util_NM", "util")), {k: rt[k] for k in rt})
for a in m["profil"]["axes"]:
    chk(f"flambement {a['axe']}-{a['axe']} : N_cr, λ̄, φ, χ, N_b,Rd du moteur", all(x in ft for x in (
        f"{a['Ncr']:,.1f}".replace(",", " ").replace(".", ","), f"{a['lambdabar']:.3f}".replace(".", ","), f"{a['phi']:.3f}".replace(".", ","),
        f"{a['chi']:.3f}".replace(".", ","), f"{a['Nb_Rd']:,.1f}".replace(",", " ").replace(".", ","))), a)
# les lignes « symbole = expression = substitution = résultat » de la fenêtre, morceau par morceau
FX = [[texte(x) for x in fx.split('<span class="eq">=</span>')] for fx in re.findall(r'<div class="fx">(.*?)</div>', fen)]
subst = {f[0]: (f[2], f[-1]) for f in FX if len(f) >= 4}
chk("les substitutions, puissances de 10 écrites jusqu'à l'unité du résultat : « 65,25 × 10² × 235 / 1 × 10⁻³ = 1 533,5 kN » (N_Rd), "
    "« 425,69 × 10³ × 235 / 1 × 10⁻⁶ = 100,0 kN·m », « π² × 210 000 × 3 831,25 × 10⁴ / (6 × 10³)² × 10⁻³ = 2 205,8 kN » (N_cr,y)",
    subst.get("NRd") == ("65,25 × 10² × 235 / 1 × 10⁻³", "1 533,5 kN") and subst.get("My,Rd") == ("425,69 × 10³ × 235 / 1 × 10⁻⁶", "100,0 kN·m")
    and subst.get("Vz,Rd") == ("20,24 × 10² × 235 / (√3 × 1) × 10⁻³", "274,7 kN")
    and subst.get("Ncr,y") == ("π² × 210 000 × 3 831,25 × 10⁴ / (6 × 10³)² × 10⁻³", "2 205,8 kN")
    and subst.get("Nb,Rd,y") == ("0,641 × 65,25 × 10² × 235 / 1 × 10⁻³", "982,9 kN"), {k: subst.get(k) for k in ("NRd", "My,Rd", "Ncr,y")})


# chaque substitution en kN ou kN·m, évaluée TELLE QU'ELLE EST ÉCRITE, retrouve le résultat écrit (à l'arrondi des termes
# près) : une puissance de 10 oubliée le décale de mille
def _eval(expr):
    e = expr.replace(" ", "").replace(",", ".").replace("×", "*").replace("π²", "(3.141592653589793**2)").replace("√3", "(3**0.5)")
    e = e.replace("10⁻³", "1e-3").replace("10⁻⁶", "1e-6").replace("10²", "1e2").replace("10³", "1e3").replace("10⁴", "1e4").replace(")²", ")**2")
    assert re.fullmatch(r"[0-9.e*/()+-]+", e), e
    return eval(e, {"__builtins__": {}})  # noqa: S307 — chiffres et opérateurs seulement (contrôlé ci-dessus)


evals = [(k, x, abs(_eval(x) / float(y.rsplit(" ", 1)[0].replace(" ", "").replace(",", ".")) - 1))
         for k, (x, y) in subst.items() if y.endswith((" kN", " kN·m"))]
chk(f"les {len(evals)} substitutions en kN et kN·m, évaluées telles qu'écrites, donnent leur résultat (à 0,5 % près : termes arrondis)",
    len(evals) >= 7 and all(e < 5e-3 for *_, e in evals), [x for x in evals if x[2] >= 5e-3][:3] or len(evals))
chk("les hypothèses : acier, f_y, γ, section élastique, E et la courbe c (α = 0,49), le profilé et sa masse",
    all(x in ft for x in ("Acier S235", "fy = 235 N/mm²", "E = 210 000 N/mm²", "courbe c (α = 0,49)", "HEB 180 · 51,22 kg/m")), ft[:600])
chk("le tableau de la fenêtre : une ligne par vérification affichée, la gouvernante marquée", fen.count('<tr class="') == 5
    and fen.count('<tr class="gov">') == 1)
chk("χ_y, λ̄_y, φ_y ont leurs indices (aucun « _ » dans la fenêtre) et λ̄ sa barre", "_" not in re.sub(r'(id|class|data-[a-z]+)="[^"]*"', "", fen)
    and "χ<sub>y</sub>" in fen and '<span class="pf-barre">λ</span><sub>y</sub>' in fen)

# ------------------------------------------------------------------ 4. les vues
print("\n=== 4. Les vues : section à l'échelle, tableau de l'écran, taux / masse, flambement")
vues = {x["cle"]: x for x in r["vues"]}
chk("quatre vues avec le flambement calculé", list(vues) == ["section", "tableau", "courbe", "flambement"], list(vues))
svg = vues["section"]["svg"]
chk("la section : le nom, le contour hachuré, les cotes h = 180 et b = 180, les axes, les repères t_w, t_f, r",
    "HEB 180" in svg and "<path" in svg and "h = 180" in svg and "b = 180" in svg and ">y<" in svg and "8,5" in svg and "14" in svg)
chk("la section : les deux tableaux de l'écran (dimensions et propriétés), A_v,y absent écrit « — »",
    "DIMENSIONS" in svg and "PROPRIÉTÉS" in svg and "425,694" in svg and ">—<" in svg)
tab = vues["tableau"]["svg"]
rangs = re.findall(r'<tr class="([^"]+)"><td class="mk">(.*?)</td><th class="nm">([^<]+)</th>', tab)
chk(f"le tableau : les {m['n_profils']} lignes de l'écran", len(rangs) == m["n_profils"] == 48, len(rangs))
par = {li["nom"]: li for li in m["lignes"]}
chk("le tableau : trié par masse croissante", [par[n]["masse"] for _, _, n in rangs] == sorted(li["masse"] for li in m["lignes"]))
couleurs = all(("opt" if n == m["optimal"] else ("ok" if par[n]["tient"] else "nok")) == c.split()[0] for c, _, n in rangs)
chk("le tableau : les couleurs de l'écran (optimal, tient, ne tient pas), le retenu ✓, le plus léger ★",
    couleurs and [n for c, _, n in rangs if "cur" in c] == ["HEB 180"] and [n for _, mk, n in rangs if "★" in mk] == ["HEA 220"])
chk("le tableau : l'utilisation de l'écran à 10⁻³ % (HEB 180 : 99,104)", "99,104" in tab and f"{par['HEB 180']['utilisation']}".replace(".", ",") == "99,104")
chk("la courbe taux / masse : un point par profilé, le retenu cerclé, le plus léger étoilé",
    vues["courbe"]["svg"].count("<title>") == 48 and "✓ HEB 180" in vues["courbe"]["svg"] and "★ HEA 220" in vues["courbe"]["svg"])
chk("le flambement : les cinq courbes, la retenue nommée, les deux points du retenu",
    vues["flambement"]["svg"].count("<polyline") == 5 and "c · α = 0,49 · retenue" in vues["flambement"]["svg"]
    and vues["flambement"]["svg"].count('r="5.5"') == 2)
TETE = re.search(r'<div class="et-rp">.*?</div>', r["resultat"]).group(0)
chk("la tête du résultat : le nom, puis le retenu, l'acier et les efforts ; plus de phrase sous le titre de la page",
    TETE.startswith('<div class="et-rp"><b>P1</b><span>') and texte(TETE).endswith("HEB 180 · S235 · My,Ed = 60 kN·m · Vz,Ed = 40 kN · NEd = 600 kN")
    and r["phrase"] == "", TETE)

# ------------------------------------------------------------------ 5. les avertissements
print("\n=== 5. Les avertissements du moteur, rouges quand ils sont non conservatifs")
rv = req(action="calculer", etat=note(My=40.0, Vz=20.0, avec_Vy=True, Vy=150.0))
chk("V_y saisi : la ligne « Cisaillement selon y » dit « non vérifié », sans taux, et le message est rouge",
    ('<b>Cisaillement selon y</b><span class="et-mot c">non vérifié</span></div><div class="l2"><span class="cmp">A<sub>v,y</sub> absent : '
     'V<sub>y,Ed</sub> n\'est pas vérifié</span></div></div>') in rv["resultat"]
    and '<div class="msg grave">V<sub>y,Ed</sub> n\'est PAS vérifié' in rv["resultat"])
rt_ = req(action="calculer", etat=note(familles=["IPE"], My=80.0, Vz=20.0, avec_N=True, N=-300.0))
chk("traction : le message rouge « se RETRANCHE »", "se RETRANCHE" in rt_["resultat"] and 'msg grave">N<sub>Ed</sub> négatif' in rt_["resultat"])
jauge_n = re.search(r'<b>Effort normal</b><span class="pct \w">([^<]*)<.*?style="(width:[^"]*)"', rt_["resultat"], re.S)
chk("traction : η_N négatif écrit tel quel (−38 %), sa jauge VIDE (une largeur négative serait ignorée : jauge pleine) ; aucune largeur "
    "négative nulle part", jauge_n and jauge_n.group(2) == "width:0%" and jauge_n.group(1).startswith("-38")
    and not re.search(r"width:-", rt_["resultat"]), jauge_n and jauge_n.groups())
rf = req(action="calculer", etat=note(familles=["HD"], My=250.0, profil="HD 320x300.0"))
chk("paroi de 48 mm : f_y non réduit, message rouge (215 N/mm² au tableau 3.1)", 'msg grave">Paroi de 48 mm' in rf["resultat"]
    and "215 N/mm²" in rf["resultat"])
rc4 = req(action="calculer", etat=note(familles=["HP"], My=50.0, acier="S355", profil="HP 305x79"))
chk("semelle de classe 4 (HP 305x79 en S355) : message rouge, indices écrits (c/t<sub>f</sub>, W<sub>eff</sub>), aucun « _ »",
    'msg grave">Semelle élancée : c/t<sub>f</sub> = 12,0' in rc4["resultat"] and "W<sub>eff</sub>" in rc4["resultat"]
    and "_" not in texte(re.sub(r'(id|class|data-[a-z]+)="[^"]*"', "", rc4["resultat"])), texte(rc4["resultat"])[-400:])
ra = req(action="calculer", etat=note(My=60.0, Iy_min=1e7))
chk("aucun profilé : verdict « Aucun profilé », statut « Calcul impossible », aucune fenêtre ; la toile dit pourquoi (une vue, pas "
    "« pas de dessin »)", "Aucun profilé" in ra["resultat"] and ra["resume"]["niveau"] == "x" and ra["volets"] == ""
    and [v["cle"] for v in ra["vues"]] == ["tableau"] and "Aucun profilé à comparer" in ra["vues"][0]["svg"]
    and "I<sub>y</sub> ≥ 10" in ra["vues"][0]["svg"], ra.get("vues"))
rt2 = req(action="calculer", etat=note(My=30.0, avec_T=True, T=5.0, deversement=True, L_LT=6.0))
chk("torsion et déversement : les deux messages (non calculés)", "n'entre dans aucun calcul" in rt2["resultat"]
    and "Déversement non vérifié" in rt2["resultat"])
chk("les symboles des messages ont leurs indices, grecs compris (χ<sub>LT</sub>, M<sub>cr</sub>) : aucun « _ » dans le texte du résultat",
    "χ<sub>LT</sub>" in rt2["resultat"] and "_" not in texte(re.sub(r'(id|class|data-[a-z]+)="[^"]*"', "", rt2["resultat"])),
    re.findall(r"\S*_\S*", texte(rt2["resultat"]))[:5])
# la borne de l'arrondi : η = 1 + 2·10⁻⁶ s'écrit 100,000 % ; l'écran le compte comme tenant, le site le dit en rouge
MyRd = CA.resistances_section(CA.profil("HEA 200"), 235, 1.0)["My_Rd_Nmm"] / 1e6
rb = req(action="calculer", etat=note(My=MyRd * (1 + 2e-6)))
mb = MO.verifier(note(My=MyRd * (1 + 2e-6))["elements"][0])
chk("η = 1 + 2·10⁻⁶ : l'écran retient HEA 200 (100,000 %, « tient ») ; le site le colore en rouge et l'écrit",
    mb["retenu"] == "HEA 200" and mb["taux_max"] > 1 and rb["resume"]["niveau"] == "r" and "arrondit l'utilisation de HEA 200" in rb["resultat"],
    (mb["retenu"], mb["taux_max"]))

# ------------------------------------------------------------------ 6. la saisie
print("\n=== 6. La saisie : chaque champ à sa place, les refus, les cases")
e = note()
out = saisir(e, "My", "1 250,5")
chk("« 1 250,5 » (espace, virgule) → M_y = 1 250,5 kN·m", out["etat"]["elements"][0]["My"] == 1250.5 and out["message"] is None)
for chemin, val, morceau in (("Lcr_y", "-1", "ne peut pas être négatif"), ("Iy_min", "-5", "ne peut pas être négatif"),
                             ("gamma_M0", "2", "entre 0,8 et 1,5"), ("gamma_M1", "0,5", "entre 0,8 et 1,5"), ("My", "abc", "illisible"),
                             ("familles/UPN", True, "Famille inconnue"), ("acier/x", "S235", "Champ inconnu"), ("acier", "S450", "Saisie refusée"),
                             ("courbe", "e", "Saisie refusée"), ("profil", "HEA 205", "Saisie refusée")):
    out = saisir(e, chemin, val)
    chk(f"{chemin} = {val!r} : refusé (« {morceau} »), l'élément intact", morceau in (out["message"] or "") and out["etat"] == e, out["message"])
# les saisies que l'écran Streamlit accepte mais où son calcul DÉBORDE (M_y = 1e303 : OverflowError ; L_cr = 1e-200 :
# division par zéro ; L_cr = 1e150 : OverflowError ; M_y = 1e300 : un taux de trois cents chiffres ; I_y min = 1e308 : un
# texte de trois cents chiffres) : refusées proprement, avec le nom du champ, jamais une exception
ef = note(My=10.0, avec_N=True, N=100.0, flambement=True, Lcr_y=3.0, Lcr_z=3.0)
for chemin, val, morceau in (("My", "1e308", "M_y,Ed : au plus 1 000 000"), ("My", "-1e303", "M_y,Ed : au plus"), ("My", "1e300", "M_y,Ed : au plus"),
                             ("N", "2e6", "N_Ed : au plus"), ("Vz", "1e7", "V_z,Ed : au plus"), ("T", "1e308", "T_Ed : au plus"),
                             ("Lcr_y", "1e-200", "L_cr,y : 0 (sans vérification), ou de 1 mm à 1 000 m"), ("Lcr_z", "1e150", "L_cr,z : 0"),
                             ("Lcr_y", "0,0005", "L_cr,y : 0"), ("L_LT", "5000", "L_LT : 0"), ("Iy_min", "1e308", "I_y min. : au plus 100 000 000 cm⁴")):
    out = saisir(ef, chemin, val)
    chk(f"{chemin} = {val} : refusé (« {morceau} »), l'élément intact, aucune exception",
        out.get("ok") and morceau in (out["message"] or "").replace("\u202f", " ") and out["etat"] == ef, out.get("message") or out.get("erreur"))
ok_bornes = [saisir(ef, c, v) for c, v in (("My", "1e6"), ("My", "-1e6"), ("Lcr_y", "0,001"), ("Lcr_y", "1000"), ("Lcr_y", "0"), ("Iy_min", "1e8"))]
chk("les bornes elles-mêmes sont admises (±10⁶ ; 1 mm et 1 000 m ; 0 ; 10⁸ cm⁴) et se calculent",
    all(o.get("ok") and o["message"] is None for o in ok_bornes), [o.get("message") or o.get("erreur") for o in ok_bornes])
gros = saisir(note(familles=["IPE A"], My=1e6, profil="IPE A 80"), "Mz", "0")
chk("le pire cas admis (M_y = 10⁶ kN·m sur l'IPE A 80) : un taux fini, en rouge", gros.get("ok") and gros["resume"]["niveau"] == "r"
    and isinstance(gros["resume"]["taux"], int) and gros["resume"]["taux"] < 10 ** 9, gros.get("resume"))
chk("le pire cas admis : la puce du verdict se resserre (classe pf-long) pour tenir ses huit chiffres",
    '<div class="verdict pf-long" data-calcul="s0"' in gros["resultat"])
ch = req(action="calculer", etat=note(familles=["IPE"], My=5000.0, profil="IPE 300"))
au = req(action="calculer", etat=note(familles=["IPE"], My=5000.0))
moins = MO.verifier(note(familles=["IPE"], My=5000.0)["elements"][0])["lignes"][0]["nom"]
chk(f"rien ne tient : l'écran retient le moins chargé ({moins}) ; un profilé CHOISI reste le retenu et le message le dit sans mentir",
    f"l'écran retient le moins chargé, {moins}" in au["resultat"] and f"<b>{moins} · " in au["resultat"]
    and f"le moins chargé est {moins} ; IPE 300 est celui choisi" in ch["resultat"] and "<b>IPE 300 · " in ch["resultat"]
    and "l'écran retient" not in ch["resultat"], (texte(au["resultat"])[-300:], texte(ch["resultat"])[-300:]))
for etat_abime in ({"version": 1, "elements": [dict(EN.nouvel_element(), My=1e303)]},
                   {"version": 1, "elements": [dict(EN.nouvel_element(), avec_N=True, N=5.0, flambement=True, Lcr_y=1e-200)]}):
    rr = req(action="calculer", etat=etat_abime)
    chk("une note .stab hors du domaine du site : refusée par le moteur avec le nom du champ (jamais une exception de calcul)",
        rr["ok"] is False and rr["erreur"].startswith("ValueError") and "borne du site" in rr["erreur"], rr.get("erreur"))
s_abime = req(action="selecteur", etat=note(My=45.0), role="profil", familles="HEB")
chk("un sélecteur demandé avec des familles abîmées (un texte) : celles de l'élément", s_abime["ok"] and "HEA 100" in s_abime["html"], s_abime)
out = saisir(e, "familles/IPE", True)
chk("cocher IPE : HEA et IPE, dans l'ordre de l'écran", out["etat"]["elements"][0]["familles"] == ["HEA", "IPE"])
out = saisir(out["etat"], "familles/HEA", False)
out = saisir(out["etat"], "familles/IPE", False)
chk("tout décocher : aucune famille = toutes (275 profilés), dit en clair", out["etat"]["elements"][0]["familles"] == []
    and "<b>toutes</b> sont comparées" in out["donnees"] and "275 profilés comparés (toutes les familles)" in texte(out["resultat"]))
d0 = req(action="calculer", etat=note())["donnees"]
chk("les efforts optionnels : la case seule tant qu'elle est décochée", 'data-chemin="avec_N"' in d0 and 'data-chemin="N"' not in d0)
d1 = saisir(note(), "avec_N", True)["donnees"]
chk("cocher N : son champ apparaît", 'data-chemin="N"' in d1)
d2 = saisir(note(), "flambement", True)
chk("cocher le flambement : L_cr,y, L_cr,z, la courbe (cinq, b par défaut), et l'aide « en compression »",
    all(f'data-chemin="{k}"' in d2["donnees"] for k in ("Lcr_y", "Lcr_z", "courbe")) and d2["donnees"].count("<option") == 5
    and '<option value="b" selected>' in d2["donnees"] and "qu'en compression" in d2["donnees"])
out = saisir(note(), "N", "300")
chk("N saisi mais case décochée : la valeur est gardée, le calcul reçoit 0 (comme l'écran)",
    out["etat"]["elements"][0]["N"] == 300.0 and MO.verifier(out["etat"]["elements"][0])["efforts"]["N"] == 0.0)
# la case « Suivre l'optimal »
e1 = note(My=45.0, Vz=30.0)
out = saisir(e1, "auto", False)
chk("décocher « Suivre l'optimal » : le retenu du moment est figé (HEA 160)", out["etat"]["elements"][0]["profil"] == "HEA 160", out["etat"]["elements"][0])
out2 = saisir(out["etat"], "My", "120")
chk("… et il le reste quand les efforts changent (HEA 160 à plus de 100 %)", MO.verifier(out2["etat"]["elements"][0])["retenu"] == "HEA 160"
    and out2["resume"]["niveau"] == "r")
out3 = saisir(out2["etat"], "auto", True)
chk("recocher : l'optimal de l'écran revient", out3["etat"]["elements"][0]["profil"] is None
    and MO.verifier(out3["etat"]["elements"][0])["retenu"] == MO.verifier(out3["etat"]["elements"][0])["optimal"])

# ------------------------------------------------------------------ 7. le sélecteur
print("\n=== 7. Le sélecteur de profilés")
s = req(action="selecteur", etat=ETAT, role="profil")
h = s["html"]
rows = re.findall(r'<button class="pk-row( cur)?( opt)?" data-choix="([^"]+)" data-prop="[^"]*">.*?<span class="kg">([^<]+)</span>', h, re.S)
cands = MO.candidats(ETAT["elements"][0])
chk("ses lignes : les candidats du moteur, triés par masse", [n for *_, n, _ in rows] == [c["nom"] for c in cands] and len(rows) == 48
    and [c["masse"] for c in cands] == sorted(c["masse"] for c in cands))
chk("le retenu ✓ (HEB 180) et le plus léger qui tient ★ (HEA 220)", [n for cur, _, n, _ in rows if cur] == ["HEB 180"]
    and [n for _, opt, n, _ in rows if opt] == ["HEA 220"])
chk("onze familles en puces, HEA et HEB allumées", h.count('class="fchip') == 11 and h.count('class="fchip on"') == 2)
chk("au survol : les propriétés du profilé (h × b, A, I_y, W_el,y, A_v,z, η)", "HEB 180</b> · h × b = 180 × 180 mm · A = 65,25 cm²" in h)
s2 = req(action="selecteur", etat=ETAT, role="profil", familles=["IPE"])
chk("une autre famille cochée dans les puces : ses profilés seuls", re.findall(r'data-choix="([^"]+)"', s2["html"])[0] == "IPE 80"
    and all(n.startswith("IPE") for n in re.findall(r'data-choix="([^"]+)"', s2["html"])))
s3 = req(action="selecteur", etat=note(My=45.0, Iy_min=5000.0), role="profil")
chk("le filtre I_y min s'applique au sélecteur", all(CA.profil(n)["Iy"] >= 5000.0 for n in re.findall(r'data-choix="([^"]+)"', s3["html"]))
    and "I<sub>y</sub> ≥ 5\u202f000 cm⁴" in s3["html"])
c = req(action="choisir", etat=note(My=45.0, Vz=30.0), role="profil", nom="IPE 300")
chk("choisir IPE 300 (famille non cochée) : IPE est ajoutée, IPE 300 est retenu, la case « Suivre l'optimal » se décoche",
    c["etat"]["elements"][0]["familles"] == ["HEA", "IPE"] and c["etat"]["elements"][0]["profil"] == "IPE 300"
    and "Profilé retenu</b>" in c["resultat"] and ">IPE 300<" in c["resultat"]
    and re.search(r'data-chemin="auto" data-sorte="case"( checked)?', c["donnees"]).group(1) is None)
c2 = req(action="choisir", etat=note(familles=[], My=45.0), role="profil", nom="HL 920x344")
chk("aucune famille cochée (toutes) : le choix ne coche rien", c2["etat"]["elements"][0]["familles"] == [])
for role, nom in (("montant", "IPE 300"), ("profil", "HEA 205")):
    x = req(action="choisir", etat=note(), role=role, nom=nom)
    chk(f"choisir {role} = {nom} : refusé par une erreur lisible", not x["ok"] and x["erreur"], x.get("erreur"))

# ------------------------------------------------------------------ 8. les éléments de la note
print("\n=== 8. Plusieurs éléments par note")
a = req(action="ajouter", etat=ETAT)
chk("ajouter : P2 (une note neuve de l'écran), ouvert", [x["nom"] for x in a["etat"]["elements"]] == ["P1", "P2"] and a["etat"]["actif"] == 1)
a2 = req(action="dupliquer", etat=a["etat"])
chk("copier : « P2 (copie) », ouvert", a2["etat"]["elements"][2]["nom"] == "P2 (copie)" and a2["etat"]["actif"] == 2)
chk("le résumé de la note : son élément le plus chargé (P1 · HEB 180 · 99 %), quel que soit l'élément ouvert",
    a2["resume"]["taux"] == 99 and a2["resume"]["gouvernant"].startswith("P1 · HEB 180"), a2["resume"])
chk("la liste de la ligne de l'élément : chaque élément avec son profilé retenu et son acier",
    '<span class="nom">P1</span><span class="dim">· HEB 180 · S235</span>' in a2["donnees"] and a2["donnees"].count('data-activer="') == 3)
a3 = req(action="supprimer", etat=a2["etat"], index=0)
chk("supprimer P1 : il en reste deux, l'ouvert suit", [x["nom"] for x in a3["etat"]["elements"]] == ["P2", "P2 (copie)"] and a3["etat"]["actif"] == 1)
a4 = req(action="ajouter", etat=a3["etat"])
chk("un nom libre : P3 (P2 existe déjà)", a4["etat"]["elements"][-1]["nom"] == "P3", [x["nom"] for x in a4["etat"]["elements"]])
a5 = req(action="activer", etat=a2["etat"], index=0)
chk("activer P1 : ses données et son résultat", a5["etat"]["actif"] == 0 and "HEB 180" in a5["resultat"])

# ------------------------------------------------------------------ 9. l'état, l'échappement, la sécurité
print("\n=== 9. L'état, l'échappement, la sécurité")
chk("une note relue est stable (normaliser)", P.normaliser(json.loads(json.dumps(a2["etat"]))) == a2["etat"])
for abime, morceau in (({"elements": "abîmé"}, "illisibles"), ({"elements": [{"My": "x"}]}, "élément 1"), ({"elements": [{"familles": ["UPN"]}]}, "UPN")):
    x = req(action="calculer", etat=abime)
    chk(f"note abîmée {str(abime)[:40]} : erreur lisible ({morceau})", not x["ok"] and morceau in x["erreur"], x.get("erreur"))
PIEGE = '<img src=x onerror=alert(1)>'
x = saisir(ETAT, "nom", PIEGE)
frag = x["resultat"] + x["volets"] + x["donnees"] + x["elements"] + x["phrase"] + "".join(v["svg"] for v in x["vues"])
chk("un nom piégé est échappé partout (données, éléments, fenêtre, vues)", "<img" not in frag and "&lt;img" in x["donnees"] and "&lt;img" in x["resultat"] and "&lt;img" in x["volets"])
tout = r["resultat"] + r["volets"] + r["donnees"] + r["elements"] + "".join(v["svg"] for v in r["vues"]) + s["html"]
chk("aucun script, aucun gestionnaire « on…= », aucune ressource externe", not re.search(r"<script|\son[a-z]+\s*=|https?://(?!www\.w3\.org)", tout, re.I))

# ------------------------------------------------------------------ 10. les nombres
print("\n=== 10. Les nombres à la française")
chk("_n : le format des tableaux de l'écran (fmt_no_trailing_zeros, 26,2455 → 26,245 comme lui), milliers séparés, signe moins typographique, 210 000 n'est pas « 21 »",
    P._n(210000.0, 0) == "210\u202f000" and P._n(1294070.0) == "1\u202f294\u202f070" and P._n(26.2455) == CA.fmt_sans_zeros(26.2455).replace(".", ",") == "26,245" and P._n(-300.0) == "−300"
    and P._n(-0.0) == "0" and P._n(None) == "—" and P._n(0.5) == "0,5")
chk("_t et _r : les formats de l'écran (« :.3f », « :.1f ») à la virgule", P._t(0.99104) == "0,991" and P._r(1533.4999) == "1\u202f533,5")

# ------------------------------------------------------------------ 11. l'établi compact (la mise en forme des poutres, 11/11/2026)
print("\n=== 11. L'établi compact")
d = r["donnees"]
res = r["resultat"]
chk("la colonne de gauche est celle de l'établi compact : la ligne de l'élément ouvert, puis ses rubriques",
    'class="et-arbre et-arbre-c" data-choixprofil="0"' in d and d.count('data-chemin="nom"') == 1 and ">Nom<" not in d)
RUBRIQUES = re.findall(r'<div class="sec( open)?" data-sec="([^"]+)"( data-secondaire="1")?>', d)
chk("quatre rubriques — les efforts (seule ouverte), les profilés comparés, les instabilités, les hypothèses (secondaire)",
    RUBRIQUES == [(" open", "ef", ""), ("", "pf", ""), ("", "in", ""), ("", "co", ' data-secondaire="1"')], RUBRIQUES)
chk("chaque rubrique repliée dit l'essentiel : les efforts, l'acier et les familles, l'instabilité vérifiée, γ_M0 et γ_M1",
    all(x in texte(d) for x in ("My 60 · Vz 40 · N 600", "S235 · HEA, HEB", "flambement, courbe c", "γM0 1 · γM1 1")), texte(d)[:300])
chk("l'acier se choisit dans la rubrique des profilés (elle n'a plus de rubrique à elle)",
    'data-chemin="acier"' in d.split('data-sec="pf"')[1].split('data-sec="in"')[0])
BLOC = re.search(r'<section class="et-bloc (\w+)" data-theme="0">.*?</section>', res, re.S)
niv_m = C.niveau(m["taux_max"])
chk("un bloc « Vérifications » : son état et son taux (ceux du verdict), la fenêtre « tout le calcul »",
    BLOC and BLOC.group(1) == {"p": "ok", "f": "warn", "r": "nok"}[niv_m] and "Vérifications" in BLOC.group(0)
    and f"{C.STATUTS[niv_m][1]} · 99 %" in texte(BLOC.group(0)) and 'class="et-tout" data-calcul="s0"' in BLOC.group(0), BLOC and texte(BLOC.group(0))[:200])
CMP = [texte(x) for x in re.findall(r'<span class="cmp">(.*?)</span>', res)]
AXE = min(m["profil"]["axes"], key=lambda a_: a_["Nb_Rd_N"])["axe"]          # l'axe dont N_b,Rd est le plus faible (le moteur)
chk("chaque ligne dit ce qu'elle compare : l'effort face à sa résistance, la somme de l'interaction face à 1, N_Ed face à N_b,Rd et son axe",
    CMP == ["51,2 kg/m · plus léger : HEA 220, 50,5 kg/m", "NEd 600 ≤ NRd 1 533,5 kN", "|My,Ed| 60 ≤ My,Rd 100,0 kN·m", "ηN + ηMy + ηMz = 0,991 ≤ 1",
            "|Vz,Ed| 40 ≤ Vz,Rd 274,7 kN", f"NEd 600 ≤ Nb,Rd {texte(P._r(v['flambement']['limite']))} kN · axe {AXE}-{AXE}"] and AXE == "y", CMP)
chk("le verdict ouvre la même fenêtre ; il n'y a qu'UNE fenêtre de calcul", '<div class="verdict" data-calcul="s0"' in res and r["volets"].count('id="ck-') == 1)
chk("chaque vue porte son en-tête technique : le titre, le retenu et l'acier, la sélection ; à droite les efforts saisis, une fois",
    all('class="et-ent"' in x.get("entete", "") and "<span>· HEB 180 · S235</span>" in x["entete"]
        and "48 profilés comparés (HEA, HEB) · 38 tiennent" in texte(x["entete"])
        and x["entete"].count('<span><i>M</i><sub>y,Ed</sub><b>60,0</b><small>kN·m</small></span><span><i>V</i><sub>z,Ed</sub><b>40,0</b><small>kN</small></span>'
                              '<span><i>N</i><sub>Ed</sub><b>600,0</b><small>kN</small></span>') == 1 for x in r["vues"])
    and [re.search(r'<b>([^<]+)</b>', x["entete"]).group(1) for x in r["vues"]] == ["SECTION", "TABLEAU DES PROFILÉS", "TAUX ET MASSE", "FLAMBEMENT"],
    r["vues"][0].get("entete"))
chk("aucun en-tête n'écrit un indice brut (« M_y,Ed ») : ses lignes sont du texte", not any("_" in texte(x["entete"]) for x in r["vues"]))
chk("plus de cadre « Dans cette note » ni de phrase ; un élément seul : pas de statut de note (le verdict le dit)",
    r["elements"] == "" and r["phrase"] == "" and r["statut_html"] == "")
chk("deux éléments : la note dit son statut, le moins bon des deux", "Conforme" in a["statut_html"] and a["statut_html"] != "", a["statut_html"])
n3 = req(action="dupliquer", etat=a["etat"], index=0)
chk("copier P1 (par son rang) : la copie s'insère juste après lui, et s'ouvre", [x["nom"] for x in n3["etat"]["elements"]] == ["P1", "P1 (copie)", "P2"]
    and n3["etat"]["actif"] == 1, [x["nom"] for x in n3["etat"]["elements"]])
CMPX = [texte(x) for x in re.findall(r'<span class="cmp">(.*?)</span>', ch["resultat"])]
chk("une vérification dépassée s'écrit « > », jamais « ≤ » (IPE 300 sous 5 000 kN·m)", any("|My,Ed| 5 000 &gt; My,Rd" in x for x in CMPX) and not any("≤" in x for x in CMPX[1:2])
    and "aucun profilé comparé ne tient" in CMPX[0], CMPX)
chk("le pire cas admis : l'état du bloc plafonne à « > 999 % » (huit chiffres casseraient son en-tête)",
    "Non conforme · &gt; 999 %" in texte(re.search(r'<section class="et-bloc nok".*?</section>', gros["resultat"], re.S).group(0)))
chk("aucun profilé : la tête dit l'acier et les efforts, l'en-tête de la toile dit qu'il n'y a rien à comparer",
    '<div class="et-rp"><b>P1</b><span>S235 · ' in ra["resultat"] and "aucun profilé à comparer (HEA)" in texte(ra["vues"][0]["entete"])
    and "et-bloc" not in ra["resultat"], texte(ra["resultat"])[:200])
un = req(action="calculer", etat=note(familles=["HEA"], My=45.0, Vz=30.0))
chk("le retenu est le plus léger qui tient : la ligne le dit, sans étoile", "kg/m · le plus léger qui tient" in texte(un["resultat"])
    and 'class="leger"' not in un["resultat"] and MO.verifier(un["etat"]["elements"][0])["plus_leger"] == MO.verifier(un["etat"]["elements"][0])["retenu"],
    texte(un["resultat"])[:300])

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:240])
sys.exit(1 if KO else 0)

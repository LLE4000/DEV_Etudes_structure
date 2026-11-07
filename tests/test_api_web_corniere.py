# -*- coding: utf-8 -*-
"""L'API du site pour les cornières ancrées (api_web/corniere.py), vérifiée en Python ordinaire.

- le résultat affiché est celui du moteur acier/corniere : taux, couleur, vérification gouvernante, pièces, groupes ;
- une vérification non conforme est rouge, la note « Non conforme » ;
- les fenêtres de calcul portent les vérifications du moteur, et chaque formule substituée REDONNE son résultat
  (les valeurs écrites, réévaluées, retombent sur le résultat du moteur à la précision d'affichage) ;
- chaque champ du cadre Données (catalogue / saisie libre × linéaire / ponctuelle) pointe un endroit réel de la
  cornière : le réécrire avec sa valeur ne change rien ;
- une saisie hors du domaine de l'écran Streamlit est REFUSÉE avec un message, et la cornière reste intacte ;
- la désignation tapée se relit au catalogue ; le passage en saisie libre et en charge ponctuelle révèle ses champs ;
- le sélecteur est trié par masse, marque la retenue (✓) et la plus légère qui tient (★) du moteur, filtre par famille ;
- les cornières de la note s'ajoutent, se copient, s'ouvrent, se suppriment (jamais la dernière) ; le résumé est la
  plus chargée ;
- les dessins : quatre vues, texte lisible à l'écran (≈ 13 px dans la toile), aucune valeur indéfinie ;
- un nom saisi est échappé partout ; le module et ses dessins se chargent sans streamlit, pandas, matplotlib…
- relecture : la ★ jamais sur une cornière dont l'aile B est plus courte que e (grise au sélecteur), refus de
  t ≥ aile, des calculs qui débordent et des noms trop longs, note à l'indice abîmé, dessin impossible dit ;
- bout à bout : l'écran Streamlit rejoué (faux st) face aux chiffres AFFICHÉS par le site.

    python3 tests/test_api_web_corniere.py
"""
import copy
import json
import math
import os
import re
import subprocess
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

from acier.corniere import entree as EN  # noqa: E402
from acier.corniere import moteur as MO  # noqa: E402
from api_web import commun as C  # noqa: E402
from api_web import corniere as K  # noqa: E402
from api_web import dessin_corniere as DC  # noqa: E402
from api_web import routeur as R  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:260]}"))


def req(**k):
    return json.loads(R.traiter(json.dumps(dict(module="corniere", **k))))


def saisir(etat, chemin, valeur):
    """Comme l'écran : le chemin lu dans data-chemin, coupé aux « / », la valeur telle que tapée."""
    out = req(action="modifier", etat=etat, chemin=chemin.split("/"), valeur=valeur)
    assert out["ok"], out
    return out


LIGNE = re.compile(r'<div class="el( gov)?"[^>]*><div class="l1"><b>([^<]+)</b>.*?<span>([^<]*)</span>.*?<span class="pct (\w)">([^<]+)</span>', re.S)
GROUPE = re.compile(r'<div class="elgrp">([^<]+)<span class="pct (\w)">([^<]+)</span></div>')
CHAMPS = re.compile(r'<input data-chemin="([^"]+)" data-sorte="(nombre|entier|texte)"[^>]*?value="([^"]*)"')
SEG_ON = re.compile(r'<button type="button" class="on" data-chemin="([^"]+)" data-sorte="choix" data-valeur="([^"]*)"')
PK = re.compile(r'<button class="pk-row( cur)?( opt)?" data-choix="([^"]+)" data-prop="[^"]*"><span class="mk">.*?</span>'
                r'<span class="nm">([^<]+)</span><span class="kg">([^<]+)</span><span class="chip (\w)">([^<]+)</span>', re.S)


def niv(v):
    return "r" if v["etat"] == MO.NOK else C.niveau(v["taux"])


def pct_txt(t):
    return f"{C.pc(C.pct(t))}{C.NBSP}%"


def lire_fr(s):
    return float(s.replace(C.NBSP, "").replace(" ", "").replace(" ", "").replace(",", "."))


# ====================================================================== 1. une note neuve
r = req(action="nouveau")
chk("nouveau : la requête aboutit", r["ok"], r.get("erreur"))
chk("la note neuve : une cornière, celle de l'écran Streamlit à l'ouverture (L 100×100×10, 5 kN/m, e 2,3 cm)",
    r["etat"] == {"version": 1, "cornieres": [MO.nouvelle_corniere()], "actif": 0}, r["etat"])
ref = MO.verifier(MO.nouvelle_corniere())
chk("le résumé est celui du moteur : taux arrondi dans sa bande, niveau, statut, gouvernant nommé",
    r["resume"] == {"taux": C.pct(ref["taux_max"]), "niveau": C.niveau(ref["taux_max"]), "statut": "Conforme",
                    "gouvernant": "Cornière 1 · transversal ELS · cisaillement"}, r["resume"])


def controle_resultat(r, nom_cas):
    """Le cadre Résultat face au moteur : une ligne « Cornière », les deux groupes, une ligne par vérification."""
    c = r["etat"]["cornieres"][r["etat"]["actif"]]
    m = MO.verifier(c)
    lignes = LIGNE.findall(r["resultat"])
    groupes = GROUPE.findall(r["resultat"])
    fautes = []
    if len(lignes) != 5:
        fautes.append(f"{len(lignes)} lignes")
        return fautes
    gov_c, nom_c, piece_c, n_c, t_c = lignes[0]
    attendu_piece = m["section"]["designation"] if c["standard"] else f"Saisie {C.nb(c['A'])}×{C.nb(c['B'])}×{C.nb(c['t'])}"
    if (nom_c, C.esc(attendu_piece) if c["standard"] else attendu_piece, n_c, t_c) != (
            "Cornière", piece_c, "r" if m["statut"] == MO.NOK else C.niveau(m["taux_max"]), pct_txt(m["taux_max"])):
        fautes.append(f"ligne Cornière : {lignes[0]}")
    for (gov, nom, piece, n, t), v in zip(lignes[1:], m["verifications"]):
        attendu = (" gov" if v["cle"] == m["gouvernant"]["cle"] else "", K.NOMS[v["cle"]], f"{v['etat_limite']} · {K._ru(v['valeur'], 'N/mm²')}",
                   niv(v), pct_txt(v["taux"]))
        if (gov, nom, piece, n, t) != attendu:
            fautes.append(f"{v['cle']} : {(gov, nom, piece, n, t)} ≠ {attendu}")
    for (titre, n, t), groupe in zip(groupes, (MO.TRANSVERSAL, MO.LONGITUDINAL)):
        vs = [v for v in m["verifications"] if v["groupe"] == groupe]
        tg = max(v["taux"] for v in vs)
        if (titre, n, t) != (K.GROUPES[groupe][0], "r" if any(v["etat"] == MO.NOK for v in vs) else C.niveau(tg), pct_txt(tg)):
            fautes.append(f"groupe {groupe}")
    if len(groupes) != 2:
        fautes.append(f"{len(groupes)} groupes")
    vg = next(v for v in m["verifications"] if v["cle"] == m["gouvernant"]["cle"])
    verdict = re.search(r'<div class="verdict">(.*?)</div></div>', r["resultat"], re.S).group(1)
    if not (f'chip xl {"r" if m["statut"] == MO.NOK else C.niveau(m["taux_max"])}' in verdict
            and f'{C.symb_math(vg["symbole"])} = {K._r(vg["valeur"], "N/mm²")} {"≤" if vg["ok"] else "&gt;"} {C.symb_math(K.LIMITES_COURTES[vg["cle"]])} = {C.nb(vg["limite"])} N/mm²' in verdict):
        fautes.append(f"verdict : {verdict[:200]}")
    statut = {"p": "Conforme", "f": "Conforme, attention", "r": "Non conforme"}["r" if m["statut"] == MO.NOK else C.niveau(m["taux_max"])]
    if statut not in r["statut_html"]:
        fautes.append(f"statut {r['statut_html']}")
    codes = {a["code"] for a in m["avertissements"]}
    msgs = re.findall(r'<div class="msg( grave)?">', r["resultat"])
    if len(msgs) != len(codes) or (("excentricite_hors_aile" in codes) != (msgs[:1] == [" grave"])):        # le grave EN TÊTE
        fautes.append("avertissements")
    masse = f"{C.fr(m['section']['masse'], 2)}{C.NBSP}kg/m" if m["section"]["masse"] else "saisie libre"
    if f"<b>{masse}</b>" not in r["resultat"]:
        fautes.append("masse au pied")
    return fautes


chk("note neuve : verdict, ligne « Cornière », groupes transversal / longitudinal, une ligne par vérification = le moteur",
    not controle_resultat(r, "neuve"), controle_resultat(r, "neuve"))
chk("la ligne « Cornière » ouvre le sélecteur et porte l'étoile de la plus légère qui tient (L 45×45×7)",
    'data-pk="corniere"' in r["resultat"] and "Plus légère qui tient : L 45×45×7 · 89 %" in r["resultat"], r["resultat"][:400])

# ====================================================================== 2. des cas variés : le résultat suit le moteur
CAS = [("non conforme partout (V = 25 kN/m, e = 20 cm, L 60×60×6)", dict(profil="L 60x60x6", V=25.0, e=20.0)),
       ("ELS par σ seul (τ_lim 5)", dict(profil="L 40x40x4", V=3.0, e=12.0, tau_lim_ELS=5.0)),
       ("charge ponctuelle 8 kN sur 0,6 m", dict(profil="L 80x80x8", charge="ponctuelle", V=8.0, L=0.6)),
       ("saisie libre 150 × 90 × 7,5", dict(standard=False, A=150.0, B=90.0, t=7.5, V=40.0, e=6.5)),
       ("e > B : la charge tombe au-delà de l'aile", dict(profil="L 50x50x5", e=8.0)),
       ("catalogue t = 35 mm", dict(profil="L 250x250x35", V=300.0, e=10.0)),
       ("V = 0", dict(profil="L 50x50x5", V=0.0)),
       ("bande 95–100 % (vert foncé)", dict(profil="L 100x100x10", V=7.8, e=1.0))]
for nom, champs in CAS:
    c = MO.normaliser(dict(MO.nouvelle_corniere(), **champs))
    rr = req(action="calculer", etat={"version": 1, "cornieres": [c], "actif": 0})
    f = controle_resultat(rr, nom) if rr["ok"] else [rr.get("erreur")]
    chk(f"{nom} : le résultat affiché est celui du moteur", not f, f)
rr = req(action="calculer", etat={"version": 1, "cornieres": [MO.normaliser(dict(MO.nouvelle_corniere(), profil="L 100x100x10", V=7.8, e=1.0))], "actif": 0})
chk("le cas « vert foncé » est bien dans la bande 95–100 % (le contrôle couvre la couleur f)", rr["resume"]["niveau"] == "f", rr["resume"])
rr = req(action="calculer", etat={"version": 1, "cornieres": [MO.normaliser(dict(MO.nouvelle_corniere(), profil="L 60x60x6", V=25.0, e=20.0))], "actif": 0})
chk("non conforme : chip rouge, statut « Non conforme », résumé rouge", rr["resume"]["niveau"] == "r" and "Non conforme" in rr["statut_html"]
    and "chip xl r" in rr["resultat"], rr["resume"])

# ====================================================================== 3. les fenêtres de calcul
FX = re.compile(r'<div class="fx">(.*?)</div>', re.S)
r0 = req(action="nouveau")
v = r0["volets"].replace("\u202f", " ").replace("\xa0", " ")   # les espaces fines des milliers, lues comme des espaces
chk("deux fenêtres de calcul : transversal (3 vérifications) et longitudinal (1)",
    'id="ck-transversal"' in v and 'id="ck-longitudinal"' in v and v.count("<tr class=") == 4, v.count("<tr class="))
for txt in ("1,35 × 5", "5,00 × 0,023", "1 000 × 10² / 6", "0,115 × 10⁶ / 16 667", "√(6,90² + 3 × 0,50²)", "6,95 / 155", "0,50 / 0,8",
            "10 × 30", "5,00 × 0,3", "Modèle lame · b = 1 000 mm", "Charge linéaire V = 5 kN/m", "σ<sub>lim,ELS</sub> = 155",
            "Fixations tous les <b>s = 30 cm</b>", "<b>Côté non sûr</b> — hypothèse provisoire de l'écran Streamlit"):
    if C.esc(txt).replace("&lt;", "<").replace("&gt;", ">") not in v and txt not in v:
        chk(f"la fenêtre écrit « {txt} »", False, "absent")
        break
else:
    chk("les fenêtres écrivent les valeurs substituées attendues (γ·V, V·e, b·t²/6, M/W, √(σ² + 3τ²), taux, panneau, hypothèses)", True)


def evaluer(expr):
    """Une substitution écrite (« 0,115 × 10⁶ / 16 667 », « √(6,90² + 3 × 0,50²) ») réévaluée."""
    e = (expr.replace(C.NBSP, "").replace("\u202f", "").replace(" ", "").replace(" ", "").replace(" ", "").replace(",", ".").replace("×", "*")
         .replace("10⁶", "1e6").replace("10³", "1e3").replace("²", "**2").replace("³", "**3").replace("√", "sqrt"))
    return eval(e, {"__builtins__": {}}, {"sqrt": math.sqrt})  # noqa: S307 — une expression écrite par le module, chiffres seulement


ecarts, n_f = [], 0
corn_essai = [MO.nouvelle_corniere()] + [MO.normaliser(dict(MO.nouvelle_corniere(), **ch)) for _, ch in CAS]
corn_essai += [MO.normaliser(dict(MO.nouvelle_corniere(), standard=False, t=t, V=V, e=e, s=s, charge=ch, L=L, gamma_ELU=g))
               for t, V, e, s, ch, L, g in ((2.0, 0.37, 0.4, 10.0, "lineaire", 1.0, 1.0), (13.5, 63.2, 17.3, 85.0, "ponctuelle", 2.35, 1.43),
                                            (20.0, 150.0, 12.0, 150.0, "ponctuelle", 0.1, 1.5), (4.5, 2.0, 3.15, 27.5, "lineaire", 1.0, 1.35))]
for c in corn_essai:
    m = MO.verifier(c)
    for groupe, blocs in m["calcul"].items():
        for b in blocs:
            for f in b["formules"]:
                n_f += 1
                x = K._fx(f)
                parts = x["parts"]
                if len(parts) < 3:                      # V_ELS = V : la donnée elle-même
                    continue
                val = evaluer(parts[2])
                res = f["resultat"]
                if not (abs(val - res) <= 0.03 * abs(res) + 1e-9):
                    ecarts.append(f"{c['nom']}/{f['symbole']} : « {parts[2]} » = {val} ≠ {res}")
chk(f"chaque formule substituée REDONNE le résultat du moteur ({n_f} formules, {len(corn_essai)} cornières ; écart ≤ 3 % : l'arrondi d'affichage)",
    not ecarts and n_f > 300, ecarts[:4])
fx = FX.findall(r0["volets"])
chk(f"le détail des formules : « symbole = expression = valeurs = résultat » ({len(fx)} lignes), les taux en pastille de couleur",
    len(fx) == sum(len(b["formules"]) for bl in ref["calcul"].values() for b in bl) and r0["volets"].count('<span class="chip p">') >= 4, len(fx))

# ====================================================================== 4. les champs : chacun pointe la cornière
def modes():
    e = req(action="nouveau")["etat"]
    yield "catalogue, linéaire", e
    yield "catalogue, ponctuelle", saisir(e, "charge", "ponctuelle")["etat"]
    e2 = saisir(e, "standard", "saisie")["etat"]
    yield "saisie libre, linéaire", e2
    yield "saisie libre, ponctuelle", saisir(e2, "charge", "ponctuelle")["etat"]


for nom_mode, etat in modes():
    rr = req(action="calculer", etat=etat)
    champs = CHAMPS.findall(rr["donnees"]) + [(c, "choix", v) for c, v in SEG_ON.findall(rr["donnees"])]
    mauvais = []
    for chemin, _s, valeur in champs:
        out = req(action="modifier", etat=copy.deepcopy(etat), chemin=chemin.split("/"),
                  valeur=valeur.replace("&amp;", "&").replace("&quot;", '"').replace("&#x27;", "'").replace("&lt;", "<").replace("&gt;", ">"))
        if not out["ok"] or out["message"] or out["etat"] != etat:
            mauvais.append((chemin, valeur, out.get("message") or out.get("erreur")))
    attendus = {"nom", "standard", "charge", "V", "e", "gamma_ELU", "s", "bord", "sig_lim_ELS", "tau_lim_ELS", "sig_lim_ELU"}
    attendus |= {"profil"} if "catalogue" in nom_mode else {"A", "B", "t"}
    attendus |= {"L"} if "ponctuelle" in nom_mode else set()
    presents = {c for c, _s, _v in champs}
    chk(f"{nom_mode} : les {len(champs)} champs sont ceux de l'écran Streamlit et les réécrire ne change rien",
        not mauvais and presents == attendus, (mauvais[:2], sorted(presents ^ attendus)))

# ====================================================================== 5. les saisies, les refus
e0 = req(action="nouveau")["etat"]
r1 = saisir(e0, "V", "12,5")
chk("une saisie « 12,5 » (virgule) est lue 12,5 kN/m et le calcul suit", r1["etat"]["cornieres"][0]["V"] == 12.5 and not controle_resultat(r1, "V"))
REFUS = (("V", "-1", "hors du domaine"), ("e", "-0,5", "hors du domaine"), ("gamma_ELU", "0,9", "hors du domaine"), ("s", "5", "hors du domaine"),
         ("tau_lim_ELS", "0,05", "hors du domaine"), ("sig_lim_ELS", "abc", "Nombre illisible"), ("profil", "L 99x99x9", "absente du catalogue"),
         ("profil", "IPE 200", "absente du catalogue"), ("standard", "peut-être", "Origine"), ("charge", "répartie", "type de charge"),
         ("inconnu", "3", "Champ inconnu"))
fautes = []
for chemin, valeur, motif in REFUS:
    out = req(action="modifier", etat=copy.deepcopy(e0), chemin=chemin.split("/"), valeur=valeur)
    if not (out["ok"] and out["message"] and motif.lower() in out["message"].lower() and out["etat"] == e0):
        fautes.append((chemin, valeur, out.get("message")))
chk(f"{len(REFUS)} saisies hors du domaine de l'écran Streamlit sont REFUSÉES avec un message, la cornière intacte", not fautes, fautes)
e_libre = saisir(e0, "standard", "saisie")["etat"]
fautes = []
for chemin, valeur in (("t", "25"), ("t", "1,5"), ("A", "9"), ("B", "301")):
    out = req(action="modifier", etat=copy.deepcopy(e_libre), chemin=[chemin], valeur=valeur)
    if not (out["message"] and "hors du domaine" in out["message"] and out["etat"] == e_libre):
        fautes.append((chemin, valeur, out.get("message")))
chk("saisie libre : t hors de 2–20 mm, A ou B hors de 10–300 mm refusés (comme l'écran)", not fautes, fautes)
out = req(action="modifier", etat=copy.deepcopy(saisir(e0, "charge", "ponctuelle")["etat"]), chemin=["L"], valeur="6")
chk("charge ponctuelle : L = 6 m refusée (0,1 à 5 m)", out["message"] and "hors du domaine" in out["message"], out.get("message"))
r2 = saisir(e0, "profil", "65 x 100 x 8")
chk("une désignation tapée (« 65 x 100 x 8 ») se relit au catalogue : L 100x65x8, ailes inégales",
    r2["etat"]["cornieres"][0]["profil"] == "L 100x65x8" and "Ailes inégales" in r2["donnees"], r2["etat"]["cornieres"][0]["profil"])
r3 = saisir(e_libre, "profil", "L 80×80×8")
chk("taper une désignation en saisie libre repasse la cornière au catalogue",
    r3["etat"]["cornieres"][0]["standard"] is True and r3["etat"]["cornieres"][0]["profil"] == "L 80x80x8")
r4 = req(action="calculer", etat=e_libre)
chk("saisie libre : les champs A, B, t apparaissent, la fiche du catalogue disparaît, la masse n'est pas inventée",
    'data-chemin="t"' in r4["donnees"] and 'data-chemin="profil"' not in r4["donnees"] and "<dl" not in r4["donnees"]
    and "Saisie 60×60×6" in r4["resultat"] and "<b>saisie libre</b>" in r4["resultat"])
r5 = saisir(e0, "charge", "ponctuelle")
chk("charge ponctuelle : le champ L apparaît (la barre latérale de l'écran), V passe en kN, la phrase le dit",
    'data-chemin="L"' in r5["donnees"] and '<span class="u">kN</span>' in r5["donnees"] and "V = 5 kN ·" in r5["phrase"] and "s = 30 cm" in r5["phrase"])
chk("charge linéaire : pas de champ L", 'data-chemin="L"' not in r0["donnees"])

# ====================================================================== 6. le sélecteur
s = req(action="selecteur", etat=e0, role="corniere")
rows = PK.findall(s["html"])
cands = MO.candidats(e0["cornieres"][0])
masses = [lire_fr(x[4]) for x in rows]
chk(f"sélecteur : les {len(cands)} cornières du catalogue, triées par masse", s["ok"] and len(rows) == len(cands) == 233 and masses == sorted(masses), len(rows))
cur = [x[2] for x in rows if x[0]]
opt = [x[2] for x in rows if x[1]]
o = MO.optimal(e0["cornieres"][0])
chk("✓ sur la retenue (L 100×100×10), ★ sur la plus légère qui tient du moteur (L 45×45×7)",
    cur == ["L 100×100×10"] and opt == [MO.designation(o["nom"])] == ["L 45×45×7"], (cur, opt))
chk("chaque ligne : désignation, masse et taux du moteur, rouge quand elle ne tient pas",
    all(nm == MO.designation(x["nom"]) and kg == C.fr(x["masse"], 2) and t == pct_txt(min(x["taux"], 9.99))
        and n == ("r" if not x["ok"] else C.niveau(x["taux"])) for (_c, _o, _ch, nm, kg, n, t), x in zip(rows, cands)))
s_li = req(action="selecteur", etat=e0, role="corniere", familles=["LI"])
rows_li = PK.findall(s_li["html"])
chk("filtre « ailes inégales » : seules les LI, la puce cochée",
    rows_li and all(EN.section_catalogue(EN.nom_catalogue(x[2]))["famille"] == "LI" for x in rows_li)
    and 'class="fchip on" data-famille="LI"' in s_li["html"] and 'class="fchip" data-famille="L"' in s_li["html"], len(rows_li))
chk("le sélecteur ouvre la fenêtre près de la retenue ou de l'optimale (data-debut)",
    f'data-debut="{max(0, min([x[2] for x in rows].index("L 45×45×7"), [x[2] for x in rows].index("L 100×100×10")) - 4)}"' in s["html"])
rc = req(action="choisir", etat=e_libre, role="corniere", nom="L 45×45×4,5")
chk("choisir « L 45×45×4,5 » : la cornière passe au catalogue avec ce profil (la désignation du bureau se relit)",
    rc["ok"] and rc["etat"]["cornieres"][0]["standard"] is True and rc["etat"]["cornieres"][0]["profil"] == "L 45x45x4.5", rc.get("erreur"))
rk = req(action="choisir", etat=e0, role="corniere", nom="L 99×99×9")
chk("choisir une cornière inconnue : erreur lisible du routeur, jamais une exception", not rk["ok"] and "absente du catalogue" in rk["erreur"], rk)
ro = req(action="choisir", etat=e0, role="corniere", nom="L 45×45×7")
chk("retenir l'optimale : l'étoile disparaît de la ligne « Cornière »", ro["ok"] and "Plus légère qui tient" not in ro["resultat"])

# ====================================================================== 7. les cornières de la note
a = req(action="ajouter", etat=e0)
a = req(action="ajouter", etat=a["etat"])
noms = [c["nom"] for c in a["etat"]["cornieres"]]
chk("ajouter deux fois : Cornière 2 et Cornière 3, la dernière ouverte", noms == ["Cornière 1", "Cornière 2", "Cornière 3"] and a["etat"]["actif"] == 2, noms)
a = saisir(a["etat"], "V", "40")
d = req(action="dupliquer", etat=a["etat"])
chk("copier : la cornière ouverte est copiée (« (copie) »), avec ses charges",
    d["etat"]["cornieres"][-1]["nom"] == "Cornière 3 (copie)" and d["etat"]["cornieres"][-1]["V"] == 40.0 and d["etat"]["actif"] == 3)
chk("le résumé de la note est sa cornière la plus chargée", d["resume"]["gouvernant"].startswith("Cornière 3") and
    d["resume"]["taux"] == C.pct(max(MO.verifier(c)["taux_max"] for c in d["etat"]["cornieres"])), d["resume"])
chk("« Dans cette note » : une ligne par cornière, son taux, l'ouverte en surbrillance",
    d["elements"].count('data-activer="') == 4 and d["elements"].count('class="ni on"') == 1
    and all(pct_txt(MO.verifier(c)["taux_max"]) in d["elements"] for c in d["etat"]["cornieres"]))
act = req(action="activer", etat=d["etat"], index=0)
chk("ouvrir la première : le cadre Données est le sien", act["etat"]["actif"] == 0 and 'value="Cornière 1"' in act["donnees"])
sp = req(action="supprimer", etat=act["etat"], index=2)
chk("supprimer la troisième : il en reste trois, l'ouverte ne change pas",
    [c["nom"] for c in sp["etat"]["cornieres"]] == ["Cornière 1", "Cornière 2", "Cornière 3 (copie)"] and sp["etat"]["actif"] == 0)
seul = req(action="supprimer", etat=e0, index=0)
chk("la dernière cornière ne se supprime pas", len(seul["etat"]["cornieres"]) == 1)
noms4 = req(action="ajouter", etat=saisir(e0, "nom", "Cornière 2")["etat"])["etat"]["cornieres"]
chk("ajouter après une « Cornière 2 » déjà nommée : le nom suivant libre (Cornière 3)", [c["nom"] for c in noms4] == ["Cornière 2", "Cornière 3"])

# ====================================================================== 8. l'état, une note abîmée
chk("l'état survit à un aller-retour JSON et sa relecture est stable",
    K.normaliser(json.loads(json.dumps(d["etat"]))) == d["etat"])
ab = req(action="calculer", etat={"version": 1, "cornieres": [{"V": -3}], "actif": 0})
chk("une cornière abîmée dans un fichier : erreur lisible", not ab["ok"] and "hors du domaine" in ab["erreur"], ab.get("erreur"))
ab2 = req(action="calculer", etat={"version": 1, "cornieres": "abîmé", "actif": 0})
chk("une liste de cornières abîmée : erreur lisible", not ab2["ok"] and "cornieres" in ab2["erreur"], ab2.get("erreur"))
ab3 = req(action="calculer", etat={"version": 1, "cornieres": [MO.nouvelle_corniere()], "actif": 7})
chk("un actif hors de la liste est ramené à la dernière cornière", ab3["ok"] and ab3["etat"]["actif"] == 0)

# ====================================================================== 9. les dessins
fautes = []
for c in corn_essai:
    rr = req(action="calculer", etat={"version": 1, "cornieres": [c], "actif": 0})
    vs = rr["vues"]
    if [v["cle"] for v in vs] != ["coupe", "elevation", "efforts", "section"]:
        fautes.append(f"{c['nom']} : vues {[v['cle'] for v in vs]}")
    for v in vs:
        px = DC.taille_texte(v["svg"])
        if not (v["svg"].startswith("<svg") and v["legende"] and 11.0 <= px <= 15.0
                and not re.search(r'="[^"]*\b(nan|inf|NaN|Infinity)\b', v["svg"])):
            fautes.append(f"{c['nom']}/{v['cle']} : texte {px:.1f} px")
chk(f"quatre vues pour chaque cornière ({len(corn_essai)} cas, petites et grandes) ; le texte se lit à 11–15 px dans la toile ; aucune valeur indéfinie",
    not fautes, fautes[:4])
w, h = DC.aire_dessin(r0["vues"][3]["svg"])
chk("la section d'une cornière à ailes égales est dessinée à l'échelle (boîte presque carrée)", 0.7 < w / h < 1.3, (w, h))
chk("la coupe nomme la cornière, la charge, e et la fixation non vérifiée",
    all(x in r0["vues"][0]["svg"] for x in ("L 100×100×10 ancrée au mur", "V = 5 kN/m", "fixation non vérifiée", "A = 100", "B = 100")))
chk("aucun texte de dessin n'écrit un indice « _ » brut (M_ELS…) : les textes, hors feuille de style",
    not any(re.search(r">[^<]*\w_\w[^<]*<", re.sub(r"<style>.*?</style>", "", v["svg"], flags=re.S)) for v in r0["vues"]))

# ====================================================================== 10. échappement, pureté
PIEGE = '<img src=x onerror=alert(1)>'
rp = saisir(e0, "nom", PIEGE)
frag = rp["resultat"] + rp["volets"] + rp["donnees"] + rp["elements"] + rp["phrase"] + "".join(v["svg"] for v in rp["vues"])
chk("un nom piégé est échappé partout (données, éléments, fenêtres, dessins)", "<img" not in frag and "&lt;img" in rp["elements"])
SOUS = r'''
import sys
INTERDITS = ("streamlit", "pandas", "reportlab", "matplotlib", "scipy", "PIL", "fitz")
class Refus:
    def find_spec(self, nom, chemin=None, cible=None):
        if nom.split(".")[0] in INTERDITS:
            raise ImportError(nom)
sys.meta_path.insert(0, Refus())
sys.path.insert(0, sys.argv[1])
avant = set(sys.modules)
import json
from api_web import routeur
r = json.loads(routeur.traiter(json.dumps({"module": "corniere", "action": "nouveau"})))
tiers = sorted({m.split(".")[0] for m in set(sys.modules) - avant} - set(sys.stdlib_module_names) - {"acier", "api_web"})
print(json.dumps({"ok": r["ok"], "tiers": tiers}))
'''
pr = subprocess.run([sys.executable, "-I", "-c", SOUS, RACINE], capture_output=True, text=True, check=False)
out = json.loads(pr.stdout) if pr.returncode == 0 else {"ok": False, "tiers": pr.stderr[-300:]}
chk("le module du site et ses dessins calculent sans streamlit, pandas, matplotlib… (seuls la bibliothèque standard et numpy)",
    out["ok"] and set(out["tiers"]) <= {"numpy"}, out)

# ====================================================================== 11. relecture : la charge sur l'aile, les refus, les cas limites
e8 = saisir(e0, "e", "8")                                   # e = 80 mm : L 45×45×7 « tient » mais son aile B ne fait que 45 mm
chk("e = 80 mm : l'étoile de la ligne « Cornière » propose une cornière dont l'aile B reçoit la charge (L 80×80×7), pas L 45×45×7",
    "Plus légère qui tient : L 80×80×7" in e8["resultat"], re.search(r'title="(Plus légère[^"]*)"', e8["resultat"]))
s8 = req(action="selecteur", etat=e8["etat"], role="corniere")
rows8 = PK.findall(s8["html"])
c8 = MO.candidats(e8["etat"]["cornieres"][0])
gris = [x[3] for x in rows8 if x[5] == "c"]
chk(f"sélecteur, e = 80 mm : les {len(gris)} cornières qui tiennent mais dont l'aile B est trop courte sont GRISES, jamais étoilées ; "
    "l'en-tête dit pourquoi, la ligne aussi",
    gris and all(not x["aile"] and x["ok"] for x in c8 if x["designation"] in gris)
    and len(gris) == sum(x["ok"] and not x["aile"] for x in c8) and [x[3] for x in rows8 if x[1]] == ["L 80×80×7"]
    and "gris : aile B &lt; e" in s8["html"] and "la charge tombe au-delà de l&#x27;aile B" in s8["html"], gris[:5])
fautes = []
for chemin, valeur, motif in (("t", "12", "pas une cornière"), ("A", "5,5", "hors du domaine"), ("V", "1e200", "déborde"),
                              ("e", "1e300", "déborde"), ("s", "1e308", "déborde"), ("nom", "x" * (K.NOM_MAX + 1), "trop long")):
    etat_l = saisir(saisir(e0, "standard", "saisie")["etat"], "A", "12")["etat"]
    out = req(action="modifier", etat=copy.deepcopy(etat_l), chemin=[chemin], valeur=valeur)
    if not (out["ok"] and out["message"] and motif in out["message"] and out["etat"] == etat_l):
        fautes.append((chemin, valeur, out.get("message") or out.get("erreur")))
chk("refus de plus que l'écran, chacun avec son message et la cornière intacte : t ≥ aile (ce n'est plus une cornière), "
    "calcul qui déborde (V, e, s démesurés), nom de plus de 60 caractères", not fautes, fautes)
fautes = []
for actif in ("x", [1], True, None, -3, 2.7):
    out = req(action="calculer", etat={"version": 1, "cornieres": [MO.nouvelle_corniere()] * 2, "actif": actif})
    if not (out["ok"] and out["etat"]["actif"] in (0, 1)):
        fautes.append((actif, out.get("erreur")))
chk("une note dont l'indice de la cornière ouverte est abîmé (texte, liste, booléen, négatif) s'ouvre quand même", not fautes, fautes)
r0m = saisir(e0, "V", "-0")
chk("« -0 » saisi s'écrit 0 dans le champ (jamais « -0 »)", 'data-chemin="V" data-sorte="nombre" inputmode="decimal" value="0"' in r0m["donnees"])
r_dec = saisir(e0, "e", "2,3456")
chk("une saisie fine (e = 2,3456 cm) est réécrite telle quelle dans son champ, sans arrondi",
    'value="2,3456"' in r_dec["donnees"] and r_dec["etat"]["cornieres"][0]["e"] == 2.3456)
r_s = saisir(e0, "s", "1e30")
vues_s = {v["cle"]: v["svg"] for v in r_s["vues"]}
chk("proportions absurdes (s = 10²⁸ m) : le calcul est rendu, l'élévation dit « dessin impossible à cette échelle », les autres vues sont dessinées",
    r_s["ok"] and "Dessin impossible à cette échelle" in vues_s["elevation"]
    and all("Dessin impossible" not in vues_s[k] for k in ("coupe", "efforts", "section")), list(vues_s))
chk("l'avertissement non conservatif de l'écran (vérification longitudinale) est titré en gras « Côté non sûr »",
    "<b>Côté non sûr</b> : la vérification « longitudinale »" in r0["resultat"])
r_hors = saisir(saisir(e0, "profil", "L 50x50x5")["etat"], "e", "8")
chk("e > B : l'avertissement grave, titré, en tête", re.search(r'<div class="els">.*?</div></div></div><div class="msg grave"><b>La charge tombe au-delà de l&#x27;aile B</b>',
                                                       r_hors["resultat"], re.S) is not None)

# ====================================================================== 12. bout à bout : l'écran Streamlit rejoué face à ce que le SITE affiche
# (la chaîne écran → moteur est prouvée au bit par tests/test_corniere_moteur.py, moteur → site plus haut ; ici on lit
# directement les chiffres AFFICHÉS des deux côtés : bandeaux et markdown de l'écran, lignes et fenêtres du site)
try:
    import logging
    import warnings
    logging.disable(logging.WARNING)
    warnings.simplefilter("ignore")
    import streamlit  # noqa: F401 — l'écran l'importe
    import modules.corniere as MC
    import modules.navigation as NAV
except ImportError:                                            # pragma: no cover
    MC = None
    print("     (streamlit absent : le contrôle bout à bout n'est pas rejoué sur cette machine)")


class _Col:
    def __init__(self, st):
        self.st = st

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def metric(self, label, value, *a, **k):
        self.st.ev.append(("metric", label, value))

    def write(self, *a, **k):
        self.st.ev.append(("write",) + a)


class _St:
    """Un faux « st » : il sert les saisies (par libellé) et enregistre ce que l'écran affiche."""
    def __init__(self, saisies):
        self.sa, self.ev, self.sidebar = saisies, [], self

    def __getattr__(self, nom):
        if nom in ("header", "subheader", "divider", "caption", "info", "success", "markdown", "write", "warning", "error"):
            return lambda *a, **k: self.ev.append((nom,) + a)
        raise AttributeError(nom)

    def columns(self, spec, **_):
        return [_Col(self) for _ in range(spec if isinstance(spec, int) else len(spec))]

    def button(self, *a, **k):
        return False

    def toggle(self, label, value=False, **_):
        return self.sa.get(label, value)

    def selectbox(self, label, options, index=0, **_):
        return self.sa.get(label, list(options)[index])

    def radio(self, label, options, index=0, **_):
        return self.sa.get(label, list(options)[index])

    def number_input(self, label, min_value=None, max_value=None, value=None, step=None, **_):
        return self.sa.get(label, value)


def ecran_affiche(c, cle_ecran=None):
    """Ce que l'écran Streamlit AFFICHE pour la cornière c : (bandeaux en %, ✅/❌, σ_eq,ELS, τ_ELS, σ_eq,ELU, σ_eq,long)."""
    sa = {"Utiliser une cornière **standard**": c["standard"], "Type de charge": {"lineaire": "Linéaire (kN/m)", "ponctuelle": "Ponctuelle (kN)"}[c["charge"]],
          ("Effort tranchant V (kN/m)" if c["charge"] == "lineaire" else "Effort tranchant V (kN)"): c["V"],
          "Excentricité e (cm) – distance charge/cornière": c["e"], "Coefficient global (ELU)": c["gamma_ELU"],
          "Espacement des fixations s (cm)": c["s"], "Distance de l’âme au bord du mur (cm)": c["bord"], "σ_lim, ELS (N/mm²)": c["sig_lim_ELS"],
          "τ_lim, ELS (N/mm²)": c["tau_lim_ELS"], "σ_eq lim, ELU (N/mm²)": c["sig_lim_ELU"], "Longueur d’influence L pour charge ponctuelle (m)": c["L"],
          "Largeur aile A (mm)": c["A"], "Largeur aile B (mm)": c["B"], "Épaisseur t (mm)": c["t"]}
    if c["standard"]:
        sa["Section"] = cle_ecran
    f = _St(sa)
    vrai, vrai_nav = MC.st, NAV.st
    MC.st, NAV.st = f, f
    try:
        MC.show()
    finally:
        MC.st, NAV.st = vrai, vrai_nav
    md = "\n".join(e[1] for e in f.ev if e[0] == "markdown")
    succ = [e[1] for e in f.ev if e[0] == "success"]
    info = [e[1] for e in f.ev if e[0] == "info"]
    m1 = re.search(r"ELS : σ_eq/lim = (\d+)% ; τ/lim = (\d+)%  (\S+)", succ[0])
    m2 = re.search(r"ELU : σ_eq/lim = (\d+)%  (\S+)", succ[1])
    m3 = re.search(r"σ_eq/lim = (\d+)% (\S+)", info[-1])
    eq = re.search(r"σ_eq\(ELS\)=\*\*([\d.]+)\*\* ; σ_eq\(ELU\)=\*\*([\d.]+)", md)
    tau = re.search(r"τ\(ELS\)=\*\*([\d.]+)\*\*", md)
    eql = re.search(r"σ_eq_long\(ELU\)=\*\*([\d.]+)", md)
    return ([int(m1.group(1)), int(m1.group(2)), int(m2.group(1)), int(m3.group(1))],
            [m1.group(3) == "✅", m2.group(2) == "✅", m3.group(2) == "✅"],
            [float(eq.group(1)), float(tau.group(1)), float(eq.group(2)), float(eql.group(1))])


if MC is not None:
    BOUT = [("défauts de l'écran", MO.nouvelle_corniere(), "100x100x10"),
            ("ponctuelle 8 kN sur 0,6 m", dict(MO.nouvelle_corniere(), profil="L 80x80x8", charge="ponctuelle", V=8.0, L=0.6), "80x80x8"),
            ("saisie libre 150 × 90 × 7,5, non conforme", dict(MO.nouvelle_corniere(), standard=False, A=150.0, B=90.0, t=7.5, V=40.0, e=6.5), None),
            ("ELU dans la bande 95–100 %", dict(MO.nouvelle_corniere(), V=30.0, e=6.0, tau_lim_ELS=5.0, sig_lim_ELU=150.0), "100x100x10"),
            ("τ à 100,4 % (l'écran écrit « 100% ❌ »)", dict(MO.nouvelle_corniere(), V=8.032, e=1.0), "100x100x10"),
            ("pile à la limite τ = τ_lim", dict(MO.nouvelle_corniere(), V=8.0, e=1.0), "100x100x10"),
            ("critères S355, non ronds", dict(MO.nouvelle_corniere(), profil="L 90x90x9", V=35.15, e=6.35, gamma_ELU=1.43, s=27.5,
                                               sig_lim_ELS=355.0, tau_lim_ELS=205.0, sig_lim_ELU=355.0), "90x90x9")]
    fautes = []
    for nom, c, cle in BOUT:
        c = MO.normaliser(c)
        pct_e, ok_e, val_e = ecran_affiche(c, cle)
        rr = req(action="calculer", etat={"version": 1, "cornieres": [c], "actif": 0})
        lignes = LIGNE.findall(rr["resultat"])[1:]
        pct_s = [int(re.match(r"(\d+)", x[4]).group(1)) for x in lignes]
        niv_s = [x[3] for x in lignes]
        val_s = [lire_fr(re.search(r"· ([\d\s , ]+) N/mm²", x[2]).group(1)) for x in lignes]
        ok_s = [niv_s[0] != "r" and niv_s[1] != "r", niv_s[2] != "r", niv_s[3] != "r"]
        if any(abs(a - b) > 1 for a, b in zip(pct_s, pct_e)) or ok_s != ok_e:
            fautes.append(f"{nom} : % site {pct_s} / écran {pct_e} ; verdicts site {ok_s} / écran {ok_e}")
        tol = (0.05 + 0.005, 0.005 + 0.005, 0.05 + 0.005, 0.05 + 0.005)       # demi-unité affichée de chaque côté
        if any(abs(a - b) > t + 1e-9 for a, b, t in zip(val_s, val_e, tol)):
            fautes.append(f"{nom} : contraintes site {val_s} / écran {val_e}")
    chk(f"bout à bout, {len(BOUT)} cas : les taux et contraintes AFFICHÉS par le site = ceux de l'écran Streamlit rejoué (à l'arrondi "
        "d'affichage près : ±1 point, ½ unité du dernier chiffre), et rouge ⇔ ❌", not fautes, fautes)

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

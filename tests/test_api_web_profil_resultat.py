# -*- coding: utf-8 -*-
"""(lot C, C4) Les RÉSULTATS du module « Choix de profilé » (api_web/profil_resultat.py, api_web/profil_mots.py,
api_web/dessin_barre.py), vérifiés en Python ordinaire, à travers le routeur et ``profil.resultat``, comme l'écran les lit.

  1. « BLOQUÉ » — une méthode bloquée (PRS aux bords non saisis) se rend sans KeyError ; sa puce dit le mot, jamais verte.
  2. LES ÉCARTS AVEC LES CLASSEURS (brief § 6.7) — la ligne « n corrections des classeurs » du bloc ELS ouvre la fenêtre
     « ecarts » : une ligne par correction QUE LE MOTEUR REND (``corrections``), sur des barres du catalogue qui les
     déclenchent (``pcr_u`` : u ≠ 1 ; ``phi_min`` : u_y·L_y ≠ u_z·L_z) ; jamais un choix du bureau.
  3. AVANT / APRÈS RAIDISSEUR (brief § 15.4) — test 18 du § 11.5 : 387,8 kN ; 1,160 ; 1 002,2 kN ; 0,449 ; 0,909 ; le rappel
     sur τ ; aucun tableau sans raidisseur proposé.
  4. UNE INFORMATION À UN ENDROIT — R2 : le taux de chaque méthode écrit une fois au niveau de la méthode (le verdict) ; ni la
     masse ni le taux de la barre à la ligne « Profilé » ; l'ELS jamais en tête par le contrôle hérité (H-EQ), mais par la
     plus forte des autres ; P_cr caché sans effort normal dans la vérification ; une fenêtre de vérification sans les hypothèses de sa méthode ; « sans raidisseur » une fois.
  5. -FM écrit comme la norme — « η₂ + 0,8·η₁ = 0,513 ≤ 1,4 », taux 0,366 ; à l'écran et dans la note PDF.
  6. LA COMPARAISON — les sujets, l'écart en points, ce qui gouverne ; « Tout le calcul » de l'Eurocode 3 dit l'autre
     variante de chaque décision (§ 6.3.2.2 pour information, A_v,y, n_pl avec l'autre γ_M).
  7. LE DESSIN — niveaux mêlés (▲ ● ▼ dessinés), maintien négligé (creux), position, N par tronçon, débords, points vérifiés ;
     les lettres présentes, aucun texte sur un autre, rien hors de la scène.
  8. MUTATIONS — chaque altération plausible fait rougir au moins un contrôle.

    /usr/bin/python3.12 -B tests/test_api_web_profil_resultat.py
"""
import copy
import html
import json
import os
import re
import sys
import types

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from acier.barre import raidisseurs as RA  # noqa: E402
from acier.barre import resultat as RE  # noqa: E402
from api_web import dessin_barre  # noqa: E402,F401
from api_web import note_profil  # noqa: E402,F401
from api_web import profil  # noqa: E402,F401
from api_web import profil_mots  # noqa: E402,F401
from api_web import profil_resultat  # noqa: E402,F401
from api_web import registre as REG  # noqa: E402
from api_web import routeur as R  # noqa: E402

OK, KO = [], []
# R2 mesuré AVANT le bloc C4 (base de la vague 2, 44af2cd) : « 144 % » 7 fois dans la colonne, « 110 % » 4 fois (9 et 4 de plus
# dans les fenêtres ; après : 7 et 4)
AVANT_R2 = {"144": 7, "110": 4}


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:400]}"))


def mod(nom):
    """Le module tel qu'il est CHARGÉ (une mutation le remplace le temps d'un passage)."""
    return sys.modules[f"api_web.{nom}"]


def req(**k):
    return json.loads(R.traiter(json.dumps(dict(module="profil", **k))))


def suite(*gestes):
    """Une note neuve, puis des saisies comme l'écran les envoie ; rend la dernière réponse."""
    r = req(action="nouveau")
    for ch, v in gestes:
        r = req(action="modifier", etat=r["etat"], chemin=ch.split("/"), valeur=v)
        assert r["ok"], (ch, v, r.get("message"))
    return r


def texte(h: str) -> str:
    """Ce que l'œil lit d'un morceau de HTML : chaque balise est une séparation (« f<sub>y</sub> » se lit « f y »), les
    entités rendues, les espaces (insécables, fines) réduits."""
    return re.sub(r"[\s\u00a0\u202f\u2009\u200b]+", " ", html.unescape(re.sub(r"<[^>]+>", " ", h))).strip()


def compte(t: str, p: str) -> int:
    return len(re.findall(rf"(?<![\d,]){p} ?%", t))


def fenetre(volets: str, ident: str) -> str:
    """Le HTML de la fenêtre ``ck-<ident>`` (jusqu'à la suivante)."""
    i = volets.find(f'id="ck-{ident}"')
    if i < 0:
        return ""
    j = volets.find('<div class="flyout calcul"', i + 10)
    return volets[i:j if j > 0 else len(volets)]


def hyp(fen: str) -> str:
    m = re.search(r'<div class="ck-hyp">(.*?)</div><div class="ck-body">', fen, re.S)
    return m.group(1) if m else None


def etat_de(el):
    return mod("profil").normaliser({"version": mod("profil").VERSION, "actif": 0, "elements": [el]})


def barre(**k):
    el = mod("profil").barre_neuve("B")
    el.update(copy.deepcopy(k))
    return el


def P_(v, a, ss=100.0, **k):
    return dict({"type": "P", "v": v, "a": a, "ss": ss}, **k)


# R2 du brief (11.3), tel que le plan l'a mesuré (sans maintien) ; R2M : la semelle haute tenue au droit de la charge
R2 = dict(poids_propre=False, maintiens=[], niveau="haut",
          cas=[{"nom": "G", "nature": "permanent", "N": 0.0, "charges": [{"type": "q", "v": 2.42, "a": 0.0, "b": 6.0}, P_(10.0, 3.0)]},
               {"nom": "Q", "nature": "exploitation_A", "N": 0.0, "charges": [P_(15.0, 3.0)]}])
R2M = dict(R2, maintiens=[{"x": 3.0, "genre": "haut"}])
# (reprise) N dans le seul cas Q : le déversement gouverne dans « G » (P = 0), la flexion composée dans « G + Q » (P = 20 kN)
NQ = dict(poids_propre=False, maintiens=[],
          cas=[{"nom": "G", "nature": "permanent", "N": 0.0, "charges": [{"type": "q", "v": 12.0, "a": 0.0, "b": 6.0}]},
               {"nom": "Q", "nature": "exploitation_A", "N": 20.0, "charges": [{"type": "q", "v": -1.0, "a": 0.0, "b": 6.0}]}])
# test 18 du § 11.5 : IPE 300 S235, type (a), F = 1,5 × 300 = 450 kN, s_s = 100 mm, la semelle tenue au droit de la charge
B18 = dict(travees=[2.0], poids_propre=False, appuis={"mode": "poses", "ss": 100.0, "c": 0.0}, maintiens=[{"x": 1.0, "genre": "complet"}],
           cas=[{"nom": "Q", "nature": "exploitation_A", "N": 0.0, "charges": [P_(300.0, 1.0)]}])
# le dessin : niveaux mêlés, un maintien négligé, un débord, trois points vérifiés (deux proches), N par tronçon, inclinée
DESSIN = dict(maintiens=[{"x": 3.5, "genre": "complet", "neglige": True}, {"x": 5.0, "genre": "haut"}], debords=[0.5, 0.0],
              points=[1.5, 1.6, 6.0], position="inclinee", angle=30.0,
              cas=[{"nom": "G", "nature": "permanent", "N": 0.0, "charges": [{"type": "q", "v": 5.0, "a": 0.0, "b": 6.0}, P_(20.0, 2.5, niveau="bas")],
                    "N_troncons": [{"x2": 2.5, "N": 50.0}, {"x2": 6.5, "N": 30.0}]},     # (intégration C : jusqu'au bout de la barre, 6,5 m)
                   {"nom": "Q", "nature": "exploitation_A", "N": 0.0, "charges": [{"type": "q", "v": 5.0, "a": 2.5, "b": 6.0, "niveau": "centre"}]}])


def ecran(el):
    """(l'état normalisé, le résultat de l'écran, le résultat du moteur)"""
    etat = etat_de(el)
    return etat, mod("profil").resultat(etat), mod("profil").calculer(etat["elements"][0])


def textes_svg(svg: str):
    """Les textes d'un dessin et leur boîte estimée (0,55 du corps par caractère) : (texte, x1, y1, x2, y2)."""
    out = []
    for m in re.finditer(r'<text x="([-\d.]+)" y="([-\d.]+)" text-anchor="(\w+)" font-size="([\d.]+)"[^>]*>(.*?)</text>', svg, re.S):
        x, y, a, fs = float(m[1]), float(m[2]), m[3], float(m[4])
        t = html.unescape(re.sub(r"<[^>]+>", "", m[5])).replace("​", "")
        w = 0.55 * fs * len(t)
        x1 = x if a == "start" else (x - w if a == "end" else x - w / 2)
        out.append((t, x1, y - 0.75 * fs, x1 + w, y + 0.2 * fs))
    return out


def dessus(ts):
    return [(a[0], b[0]) for i, a in enumerate(ts) for b in ts[i + 1:]
            if min(a[3], b[3]) - max(a[1], b[1]) > 1 and min(a[4], b[4]) - max(a[2], b[2]) > 1]


def controles():
    """[(nom, vrai ?, info)] : tout ce que ce fichier vérifie — rejoué sous chaque mutation."""
    PR, M, DB = mod("profil_resultat"), mod("profil_mots"), mod("dessin_barre")
    out = []

    def c(nom, cond, info=""):
        out.append((nom, bool(cond), info))

    # ---------------------------------------------------------------- 1. « Bloqué »
    try:
        rP = suite(("prs", "prs"), *[(f"prs/{k}", v) for k, v in (("b1", "300"), ("t1", "30"), ("hw", "540"), ("tw", "15"), ("b2", "300"),
                                                                  ("t2", "30"), ("a", "6"))])
        v0 = re.search(r'<div class="verdict"[^>]*>(.*?)</div></div>', rP["resultat"], re.S).group(1)
        puce = re.search(r'<span class="chip xl ([a-z])[^"]*"[^>]*>(.*?)</span>', v0)
        c("1. PRS aux bords non saisis : la méthode ELS « Bloqué » se rend (résultat, fenêtres, vues : pas de KeyError) — la puce dit le MOT, "
          "jamais un taux pour information, jamais verte ; la raison en une ligne ; l'en-tête du bloc « Bloqué »",
          puce is not None and puce.group(1) not in ("p", "f") and texte(puce.group(2)) == "Bloqué" and "Bords des semelles à saisir" in v0
          and re.search(r'<span class="n">ELS</span><span class="e">Bloqué</span>', rP["resultat"]) is not None and rP["volets"] and rP["vues"],
          v0[:300])
        dev_ = re.search(r'<div class="el vc[^"]*" data-calcul="h-H-DEV"[^>]*><div class="l1">.*?<span class="pct ([a-z])"[^>]*>(.*?)</span>', rP["resultat"], re.S)
        c("1. … sa ligne « Déversement », bloquée, montre le taux calculé pour information (en larges plats) — jamais en vert",
          dev_ is not None and dev_.group(1) not in ("p", "f") and "%" in texte(dev_.group(2)), dev_.groups() if dev_ else None)
    except Exception as e:  # noqa: BLE001
        c("1. PRS aux bords non saisis : rendu sans erreur", False, f"{type(e).__name__}: {e}")

    # ---------------------------------------------------------------- 2. les écarts avec les classeurs
    cas_ecarts = {
        "pcr_u (u_y = 2)": suite(("mode", "efforts"), ("efforts/service/N", "100"), ("efforts/service/My", "20"), ("efforts/elu/N", "150"),
                                 ("efforts/elu/My", "30"), ("flambement/y/u", "2")),
        "phi_min (L_y = 6 m, L_z = 1 m)": suite(("mode", "efforts"), ("efforts/service/N", "100"), ("efforts/service/My", "20"),
                                                ("efforts/elu/N", "150"), ("efforts/elu/My", "30"), ("flambement/y/L", "6"), ("flambement/z/L", "1")),
    }
    for nom, r in cas_ecarts.items():
        x = mod("profil").calculer(r["etat"]["elements"][0])["methodes"]["historique"]
        cs = x["corrections"]
        fen = fenetre(r["volets"], M.FENETRE_ECARTS)
        lignes = re.findall(r'<td class="nm">(.*?)<span>(.*?)</span></td>', fen, re.S)
        pastilles = [texte(m) for m in re.findall(r'<td class="n"><span class="chip c">(.*?)</span></td>', fen)]
        titres = {v["id"]: v["titre"] for v in x["verifications"]}
        attendu = [texte(M.sub(k["texte"][:1].upper() + k["texte"][1:])) for k in cs]
        ligne = re.search(rf'<div class="el vc pf-lecarts" data-calcul="{M.FENETRE_ECARTS}"[^>]*><div class="l1"><b>(.*?)</b>', r["resultat"])
        c(f"2. écarts — {nom} : le moteur rend {[k['id'] for k in cs]} ; la ligne « {M.mot_corrections(len(cs))} » du bloc ELS ouvre la "
          "fenêtre ; une ligne par correction, son texte, et où elle joue — exactement celles du moteur",
          len(cs) >= 2 and ligne is not None and ligne.group(1) == M.mot_corrections(len(cs)) and [texte(li[0]) for li in lignes] == attendu
          and [texte(li[1]) for li in lignes] == ["Joue sur : " + ", ".join(M.minuscule(titres[i]) for i in k["verifications"]) for k in cs],
          ([k["id"] for k in cs], [texte(li[0]) for li in lignes], [texte(li[1]) for li in lignes]))
        c(f"2. écarts — {nom} (reprise, lisible à 390 px) : « où » n'est jamais logé dans la pastille de la dernière case (50 px au téléphone) — "
          "elle dit « — » ; les quatre colonnes vides sans titre",
          pastilles == ["—"] * len(cs) and re.search(r'<thead><tr><th>[^<]+</th><th class="fo"></th><th class="n"></th><th class="n"></th><th class="n"></th></tr>', fen)
          is not None, pastilles)
        c(f"2. écarts — {nom} : jamais un choix du bureau présenté comme une correction (f_y nominale, s, C, C_d, courbes et classes, "
          "charges de service)",
          fen and not re.search(r"nominale|s = 1,5|C = 1|C_?d = 1|charges de service|[Cc]ourbes et classes", texte(fen)), texte(fen)[:300])
    ids = {k: [x["id"] for x in mod("profil").calculer(r["etat"]["elements"][0])["methodes"]["historique"]["corrections"]]
           for k, r in cas_ecarts.items()}
    c("2. écarts — les deux barres déclenchent bien leur correction (pcr_u, phi_min) : le contrôle porte sur ce qu'il annonce",
      "pcr_u" in ids["pcr_u (u_y = 2)"] and "phi_min" in ids["phi_min (L_y = 6 m, L_z = 1 m)"], ids)
    css = css_profil()
    ck = f"#ck-{M.FENETRE_ECARTS}"
    c("2. écarts — la feuille du module (copie NOMMÉE de profil_mots.FENETRE_ECARTS : « #ck-ecarts ») retire les quatre cases vides de cette "
      "fenêtre et laisse le texte passer à la ligne : la correction prend toute la largeur au téléphone",
      f"{ck} .vtab th:not(:first-child), {ck} .vtab td:not(:first-child) {{ display: none; }}" in css
      and f"{ck} .vtab td.nm {{ white-space: normal; overflow-wrap: anywhere; }}" in css, ck)
    rN = suite(("mode", "efforts"), ("efforts/service/N", "50"), ("efforts/elu/N", "75"), ("flambement/y/L", "3"), ("flambement/z/L", "3"))
    xN = mod("profil").calculer(rN["etat"]["elements"][0])["methodes"]["historique"]
    c("2. écarts — sans correction qui joue (un poteau, N seul), ni ligne ni fenêtre",
      not xN["corrections"] and "pf-lecarts" not in rN["resultat"] and f'id="ck-{M.FENETRE_ECARTS}"' not in rN["volets"], xN["corrections"])

    # ---------------------------------------------------------------- 3. avant / après raidisseur (test 18)
    _, e18, m18 = ecran(barre(**B18))
    L18 = next(L for L in m18["ame"]["lieux"] if L["genre"] == "charge")
    f18 = fenetre(e18["volets"], PR.fenetre(M.AME, L18["base"]))
    h18 = hyp(f18) or ""
    t18 = texte(h18)
    c("3. test 18 (§ 15.4) : en tête de la fenêtre du lieu, le tableau grandeur · sans · avec — F_Rd 387,8 kN, η₂ 1,160 ; avec 2 plats "
      "70 × 20 : σ 101,0 N/mm², N_b,Rd 1 002,2 kN (0,449), contact 204,5 ≤ 225 N/mm² (0,909) ; le rappel sur τ en une ligne",
      L18["decision"] == "raidisseur" and 'class="pf-avap"' in h18 and "Sans raidisseur" in t18 and "2 plats 70 × 20 mm" in t18
      and abs(L18["sans"]["eta2"] - 1.160) <= 1e-3 and "η₂ 1,161" in t18             # (450/387,76 = 1,1605 ; le brief : 450/387,8)
      and all(s in t18 for s in ("F Rd 387,8 kN", "101,0 N/mm²", "1 002,2 kN · 0,449", "204,5 ≤ 225 N/mm² · 0,909"))
      and t18.endswith(RA.RAPPEL_TAU) and t18.count(RA.RAPPEL_TAU) == 1, t18)
    _, eM, mM = ecran(barre(**R2M))
    LM = next(L for L in mM["ame"]["lieux"] if L["genre"] == "charge")
    c("3. … sans raidisseur proposé (R2, la charge de 36 kN tenue) : pas de tableau avant/après ni de rappel",
      LM["decision"] == "aucun" and "pf-avap" not in eM["volets"] and RA.RAPPEL_TAU not in eM["volets"], LM["decision"])

    # ---------------------------------------------------------------- 4. une information à un endroit
    e2, r2, m2 = ecran(barre(**R2))
    col = texte(r2["resultat"])
    ts = {k: str(round(100 * x["taux_max"])) for k, x in m2["methodes"].items()}
    # au niveau de la MÉTHODE, le taux n'est écrit qu'au verdict : le reste vient des lignes (une vérification) et des cases de
    # la comparaison (un sujet) qui portent le même chiffre
    lignes_ = [str(C_.pct(v["taux"])) for x in m2["methodes"].values() for v in x["verifications"] if v["taux"] is not None]
    cases = [str(C_.pct(li[k]["taux"])) for li in m2["comparaison"] for k in ("historique", "ec3") if li[k] and li[k]["taux"] is not None]
    attendu = {t: 1 + lignes_.count(t) + cases.count(t) for t in ts.values()}
    obtenu = {t: compte(col, t) for t in ts.values()}
    c(f"4. R2 : le taux de chaque méthode n'est écrit qu'UNE fois au niveau de la méthode, au verdict — compté dans la colonne : avant "
      f"{AVANT_R2}, après {obtenu} (verdict + lignes + cases de la comparaison)",
      ts == {"historique": "144", "ec3": "110"} and obtenu == attendu == {"144": 5, "110": 3} and all(obtenu[t] < AVANT_R2[t] for t in obtenu),
      (ts, obtenu, attendu))
    prof = re.search(r'<div class="els pf-profil">(.*?)</div></div>', r2["resultat"], re.S)
    entetes = re.findall(r'<span class="n">(.*?)</span><span class="e">(.*?)</span>', r2["resultat"])
    c("4. … ni la ligne « Profilé » (ni masse, ni taux de la barre), ni l'en-tête d'un bloc de méthode, ni celui de la comparaison n'écrivent "
      "de taux ; la masse n'est plus au pied de la colonne (le bouton du profilé, à gauche, la dit)",
      prof is not None and "%" not in texte(prof.group(1)) and "kg/m" not in texte(prof.group(1))
      and all("%" not in texte(e) for n_, e in entetes if n_ in ("ELS", "Eurocode 3", "Comparaison")) and "kg/m" not in col
      and "kg/m" in texte(r2["donnees"]), (prof.group(1)[:200] if prof else None, entetes))
    v_els = texte(re.search(r'<div class="verdict"[^>]*>(.*?)</div></div>', r2["resultat"], re.S).group(1))
    c("4. R2, méthode ELS : le verdict est le DÉVERSEMENT (143,7 %), jamais le contrôle hérité H-EQ (143,8 %, ne décide jamais seul) ; la "
      "ligne marquée du bloc est la même ; même écrit, le contrôle hérité ne s'ajoute pas sous le verdict (144 % des deux côtés)",
      v_els.startswith("η 144 % Méthode ELS (NBN B 51-001) · déversement") and "contrôle hérité" not in v_els
      and 'data-calcul="h-H-DEV" title="Voir le détail"' in re.search(r'<div class="el vc gov"[^>]*>', r2["resultat"]).group(0), v_els)
    faux = {"gouvernant": "H-EQ", "taux_max": 1.50, "statut": RE.NOK,
            "verifications": [{"id": "H-DEV", "titre": "Déversement", "taux": 0.90, "statut": RE.OK},
                              {"id": "H-EQ", "titre": "Contrainte équivalente", "taux": 1.50, "statut": RE.NOK}]}
    t_ = PR.tete("historique", faux)
    c("4. … le contrôle hérité plus fort que le reste (150 % contre 90 %) : le verdict montre le déversement, et le contrôle hérité ne "
      "disparaît pas — il est dit sous le verdict ; l'Eurocode 3 garde son gouvernant",
      t_ is not None and t_["id"] == "H-DEV" and (PR.herite_au_dela(faux, t_) or {}).get("id") == "H-EQ"
      and PR.tete("ec3", dict(faux, gouvernant="H-EQ"))["id"] == "H-EQ", t_)
    rEQ = suite(("mode", "efforts"), ("efforts/service/N", "50"), ("efforts/service/My", "20"), ("efforts/service/Vz", "300"),
                ("efforts/elu/N", "75"), ("efforts/elu/My", "30"), ("efforts/elu/Vz", "450"))
    xEQ = mod("profil").calculer(rEQ["etat"]["elements"][0])["methodes"]["historique"]
    # l'oracle : les vérifications que rend le MOTEUR (pas PR.tete) — la plus forte des non héritées au taux jugé
    juges = [v for v in xEQ["verifications"] if v["id"] not in M.HERITES and v["statut"] in RE.JUGES and v["taux"] is not None]
    plus = max(juges, key=lambda v: v["taux"])
    hEQ = next(v for v in xEQ["verifications"] if v["id"] == "H-EQ")
    vEQ = re.search(r'<div class="verdict" data-calcul="([^"]+)"[^>]*>(.*?)</div></div>', rEQ["resultat"], re.S)
    tEQ = texte(vEQ.group(2)) if vEQ else ""
    govEQ = re.search(r'<div class="el vc gov"[^>]*>', rEQ["resultat"])
    c("4. … (reprise) H-EQ gouverne (N + M_y + V_z : 138 %) et QUATRE vérifications non héritées ont un taux jugé : le verdict ELS montre la "
      "PLUS FORTE d'entre elles (flexion composée, 48 %), jamais une plus faible ; la ligne marquée du bloc est la même ; le contrôle hérité, "
      "écrit plus haut, est dit dessous",
      xEQ["gouvernant"] == "H-EQ" and len(juges) == 4 and plus["id"] == "H-SIGMA" and vEQ is not None and vEQ.group(1) == "h-H-SIGMA"
      and tEQ.startswith(f"η {C_.pc(C_.pct(plus['taux']))} % Méthode ELS (NBN B 51-001) · flexion composée")
      and f"+ contrainte équivalente (contrôle hérité) : {C_.pc(C_.pct(hEQ['taux']))} %" in tEQ
      and govEQ is not None and 'data-calcul="h-H-SIGMA"' in govEQ.group(0), (tEQ, [(v["id"], round(v["taux"], 3)) for v in juges]))
    rS = suite(("mode", "efforts"), ("efforts/service/My", "20"), ("efforts/service/Mz", "3"), ("efforts/elu/My", "30"), ("efforts/elu/Mz", "4"))
    rSP = suite(("mode", "efforts"), ("efforts/service/N", "40"), ("efforts/service/My", "20"), ("efforts/service/Mz", "3"),
                ("efforts/elu/N", "60"), ("efforts/elu/My", "30"), ("efforts/elu/Mz", "4"))
    c("4. P = 0 (M_y et M_z seuls) : la flexion composée de la méthode ELS ne montre ni P_cr,y ni P_cr,z ; avec P = 40 kN, oui",
      'id="ck-h-H-SIGMA"' in rS["volets"] and "P<sub>cr" not in rS["volets"] and "P<sub>cr" in rSP["volets"], None)
    _, eNQ, mNQ = ecran(barre(**NQ))
    xNQ = mNQ["methodes"]["historique"]
    vNQ = {v["id"]: v for v in xNQ["verifications"]}
    fNQ = fenetre(eNQ["volets"], "h-H-SIGMA")
    pcr = re.findall(r'<div class="fx"><span class="s">(P<sub>cr,[yz]</sub>)</span>', fNQ)
    c("4. … P décidé par la VÉRIFICATION (reprise) : N dans le seul cas Q, le déversement gouverne à « G » (P = 0 dans les grandeurs de la "
      "méthode) ; la flexion composée, trouvée à « G + Q » (P = 20 kN, PE dans sa formule), garde P_cr,y et P_cr,z dans sa fenêtre",
      xNQ["gouvernant"] == "H-DEV" and vNQ["H-DEV"]["combinaison"] == "G" and not xNQ["grandeurs"]["P"]
      and vNQ["H-SIGMA"]["combinaison"] == "G + Q" and len(pcr) == 2 and "PE<sub>y</sub>" in fNQ, (xNQ["gouvernant"], pcr, texte(fNQ)[:300]))
    f_dev = fenetre(r2["volets"], "h-H-DEV")
    f_tout = fenetre(r2["volets"], "s1")
    c("4. une fenêtre de vérification ne montre que SES formules et SES messages : sa bande des hypothèses est vide ; « Tout le calcul » "
      "de la méthode dit les hypothèses, f_y une fois",
      hyp(f_dev) == "" and texte(hyp(f_tout) or "").count("f y / s = 235") == 1 and "σ adm,d" in texte(f_dev), (hyp(f_dev), texte(hyp(f_tout) or "")[:200]))
    am = re.search(r'<section class="et-bloc [a-z]+" data-theme="3">(.*?)</section>', eM["resultat"], re.S)
    am = am.group(1) if am else ""
    c("4. âme de R2 tenue (tous les lieux sans raidisseur) : « Sans raidisseur · 61 % » UNE fois, dans l'en-tête ; les lignes gardent leur "
      "borne (« c = 0 : borne prudente »)",
      texte(am).count("Sans raidisseur") == 1 and "sans raidisseur" not in texte(am)
      and f"Sans raidisseur · {C_.pc(C_.pct(mM['ame']['taux_max']))} %" in texte(am) and texte(am).count("s s = 0, c = 0 : borne prudente") == 2,
      texte(am)[:400])
    c("4. les notes de la colonne : une ligne de 120 caractères au plus (la comparaison, les paramètres à confirmer)",
      len(PR.NOTE_COMPARAISON) <= 120 and len(PR.NOTE_A_CONFIRMER) <= 120 and "\n" not in PR.NOTE_COMPARAISON + PR.NOTE_A_CONFIRMER,
      (len(PR.NOTE_COMPARAISON), len(PR.NOTE_A_CONFIRMER)))

    # ---------------------------------------------------------------- 5. -FM écrit comme la norme
    vFM = next(v for v in mM["ame"]["verifications"] if v["id"] == LM["base"] + "-FM")
    lieu_ = re.search(rf'<div class="el vc[^"]*" data-calcul="{PR.fenetre(M.AME, LM["base"])}"[^>]*>(.*?)</div></div>', eM["resultat"], re.S)
    fFM = texte(fenetre(eM["volets"], PR.fenetre(M.AME, LM["base"])))
    c("5. -FM (R2 tenue) : le moteur rend valeur 0,513 = η₂ + 0,8·η₁, limite 1,4, taux 0,366 ; la ligne du lieu l'écrit comme la norme, "
      "« η₂ + 0,8·η₁ = 0,513 ≤ 1,4 », sa pastille 37 % ; la fenêtre aussi (valeur, limite)",
      round(vFM["valeur"], 3) == 0.513 and vFM["limite"] == 1.4 and round(vFM["taux"], 3) == 0.366 and lieu_ is not None
      and "η₂ + 0,8·η₁ = 0,513 ≤ 1,4" in texte(lieu_.group(1)) and "37 %" in texte(lieu_.group(1))
      and "η₂ + 0,8·η₁ = 0,513 1,4" in fFM, (texte(lieu_.group(1)) if lieu_ else None, fFM[:300]))
    vd = mod("note_profil").verification(M.AME, mM["ame"], vFM)["verdicts"][0]
    c("5. … la note PDF écrit la même ligne (reprise) : « η₂ + 0,8·η₁ = 0,513 ≤ 1,4 », taux 36,6 % — jamais « = 0,5 ≤ 1,4 = 1,4 »",
      vd["texte"] == "$η_{2} + 0,8·η_{1}$ = 0,513 ≤ 1,4" and vd["cale"] == "36,6 %", vd)

    # ---------------------------------------------------------------- 6. la comparaison, « Tout le calcul » de l'Eurocode 3
    bc = re.search(r'<section class="et-bloc inc" data-theme="4">(.*?)</section>', r2["resultat"], re.S)
    bc = bc.group(1) if bc else ""
    fc = texte(fenetre(r2["volets"], "cmp"))
    dev = next(li for li in m2["comparaison"] if li["sujet"] == "Déversement")
    cis = next(li for li in m2["comparaison"] if li["sujet"] == "Cisaillement")
    e_dev = "−" + C_.fr(-dev["ecart"], 1) + " pts"
    c("6. la comparaison de R2 : une ligne par sujet — ELS, Eurocode 3, l'ÉCART en points (« −33,7 pts » au déversement, le moteur le "
      "nomme), ce qui gouverne quand ce n'est pas le sujet (« ELS : contrainte équivalente · Eurocode 3 : effort tranchant… ») ; ni état "
      "ni sous-titre au bloc ; la fenêtre « cmp » dit la même chose, la note en une ligne",
      dev["ecart"] < 0 and e_dev == "−33,7 pts" and PR.ecart_txt(dev) == e_dev.replace(" ", "\u00a0") and e_dev in texte(bc) and e_dev in fc
      and "ELS : contrainte équivalente · Eurocode 3 : effort tranchant, parallèle à l'âme" in texte(bc)
      and "ELS : contrainte équivalente · Eurocode 3 : effort tranchant, parallèle à l'âme" in fc
      and texte(PR.ecart_txt(cis)) in fc and "le taux le plus fort" not in r2["resultat"] and "sujets" not in texte(bc)
      and "ELS : contraintes de service ≤ f y /s ; EC3 : efforts pondérés ≤ résistances." in fc, (texte(bc)[:300], fc[:300]))
    fe = texte(fenetre(r2["volets"], "s2"))
    c("6. « Tout le calcul » de l'Eurocode 3 (R2) : « § 6.3.2.2 pour information : M_b,Rd = …, taux … % (DÉCISION BUREAU n° 4 : … décide) »",
      re.search(r"Pour information § 6\.3\.2\.2 pour information : M b,Rd = \d+,\d kN·m, taux \d+(,\d)? % \(DÉCISION BUREAU n° 4 : § 6\.3\.2\.3 décide\)\.",
                fe) is not None, fe[-400:])
    rV = suite(("mode", "efforts"), ("efforts/service/Vy", "20"), ("efforts/elu/Vy", "30"), ("efforts/service/Mz", "2"), ("efforts/elu/Mz", "3"))
    fV = texte(fenetre(rV["volets"], "s2"))
    c("6. … V_y sur un laminé : la variante de la décision n° 3, une ligne (A_v,y = A − h_w·t_w, V_pl,y,Rd, taux)",
      re.search(r"Pour information \(décision du bureau n° 3, non retenue\) : A v,y = A − h w ·t w = [\d ]+,\d mm², V pl,y,Rd = [\d ]+,\d kN, taux", fV)
      is not None, fV[-400:])
    rG = suite(("mode", "efforts"), ("parametres/gamma_M1", "1,1"), ("efforts/service/N", "200"), ("efforts/service/My", "20"),
               ("efforts/elu/N", "300"), ("efforts/elu/My", "30"))
    fG = texte(fenetre(rG["volets"], "s2"))
    c("6. … γ_M0 ≠ γ_M1 (N + M) : n_pl avec l'autre γ_M, une ligne dans « Tout le calcul », et plus au pied du bloc (dit UNE fois)",
      re.search(r"n pl avec l'autre γ M \(à confirmer\) : n pl = \d,\d{3} ; \(6\.61\) \d,\d{3} ; \(6\.62\) \d,\d{3}\.", fG) is not None
      and "n_pl avec γ" not in texte(rG["resultat"]) and "n pl avec γ" not in texte(rG["resultat"]), (fG[-300:],))

    # ---------------------------------------------------------------- 7. le dessin
    eD, rD, mD = ecran(barre(**DESSIN))
    elD = eD["elements"][0]
    svg = DB.barre("t", elD, mD)
    ts_ = textes_svg(svg)
    mots = [t[0] for t in ts_]
    larg = float(re.search(r'viewBox="0 0 ([\d.]+)', svg).group(1))
    c("7. le dessin : les lettres et les repères — appuis A et B (le débord de 0,50 m à gauche : la barre fait 6,50 m), « débord 0,50 m », "
      "p1 p2 p3 (points vérifiés), « inclinée 30° », N par tronçon (« N = 50 kN », « N = 30 kN »), et les légendes en mots",
      all(t in mots for t in ("A", "B", "p1", "p2", "p3", "débord 0,50 m", "inclinée 30°", "N = 50 kN", "N = 30 kN", "semelle haute",
                              "centre de cisaillement", "semelle basse")) and any(t.startswith("maintien creux") for t in mots)
      and any(t.startswith("p1, p2") for t in mots), mots)
    xA = next(t for t in ts_ if t[0] == "A")
    xB = next(t for t in ts_ if t[0] == "B")
    K = DB.Cadre(larg, 6.5)
    c("7. … l'appui A au bout du débord (x = 0,50 m), l'appui B au bout de la portée (6,50 m), dans la scène",
      abs((xA[1] + xA[3]) / 2 - K.X(0.5)) < 1.0 and abs((xB[1] + xB[3]) / 2 - K.X(6.5)) < 1.0 and xB[3] <= larg, (xA, xB, K.X(0.5), K.X(6.5)))
    c("7. … aucun texte sur un autre, aucun hors de la scène", not dessus(ts_) and all(t[1] >= -0.5 and t[3] <= larg + 0.5 for t in ts_),
      (dessus(ts_), [t for t in ts_ if t[1] < 0 or t[3] > larg]))
    creux = re.findall(rf'<path d="M[^"]*l-4\.5 [-\d.]+h9z" fill="#fff" stroke="{DB.DP.ACCENT}"', svg)
    signes = {"haut": len(re.findall(r'l3\.6 6\.4h-7\.2z" fill="#9C3341"', svg)), "bas": len(re.findall(r'l3\.6 -6\.4h-7\.2z" fill="#9C3341"', svg)),
              "centre": len(re.findall(r'<circle cx="[\d.]+" cy="[\d.]+" r="3" fill="#9C3341"', svg))}
    c("7. … le maintien négligé dessiné CREUX (deux triangles, et celui de sa légende) ; le niveau de chaque charge par sa FORME (▲ ● ▼, "
      "dessinées : lisibles en noir et blanc) quand les niveaux sont mêlés — et pas autrement",
      len(creux) == 3 and signes["haut"] >= 2 and signes["bas"] >= 2 and signes["centre"] >= 2
      and not re.search(r'l3\.6 6\.4h-7\.2z', DB.barre("n", e2["elements"][0], m2)), (len(creux), signes))
    return out


C_ = sys.modules["api_web.commun"]
CSS_PROFIL = os.path.join(RACINE, "site", "web", "css", "modules", "profil.css")
CSS_MUTEE = {}


def css_profil() -> str:
    """La feuille du module, telle qu'elle est — ou mutée (une mutation de ``MUTATIONS`` sur ce fichier)."""
    return CSS_MUTEE.get(CSS_PROFIL) or open(CSS_PROFIL, encoding="utf-8").read()


def muter(nom_mod, ancien, nouveau):
    """Les contrôles qui rougissent quand le texte ``ancien`` du module ``nom_mod`` devient ``nouveau`` ; None si le texte
    n'y est pas une seule fois. Le module muté remplace l'original partout où un module du site le tient (et dans le
    registre du routeur), puis l'original est remis, quoi qu'il arrive. Une FEUILLE (un chemin .css) : son texte muté est
    lu par ``css_profil`` le temps d'un passage."""
    if nom_mod.endswith(".css"):
        src = open(nom_mod, encoding="utf-8").read()
        if src.count(ancien) != 1:
            return None
        CSS_MUTEE[nom_mod] = src.replace(ancien, nouveau)
        try:
            return [n for n, ok, _ in controles() if not ok]
        finally:
            CSS_MUTEE.clear()
    orig = sys.modules[nom_mod]
    src = open(orig.__file__, encoding="utf-8").read()
    if src.count(ancien) != 1:
        return None
    m = types.ModuleType(nom_mod)
    m.__file__, m.__package__ = orig.__file__, orig.__package__
    exec(compile(src.replace(ancien, nouveau), orig.__file__, "exec"), m.__dict__)
    poses = [(x, k) for x in list(sys.modules.values()) if getattr(x, "__name__", "").startswith("api_web") and x is not orig
             for k, v in list(vars(x).items()) if v is orig]
    cle = nom_mod.rpartition(".")[2]
    dans_reg = REG._MODULES.get(cle) is orig
    try:
        for x, k in poses:
            setattr(x, k, m)
        sys.modules[nom_mod] = m
        if dans_reg:
            REG._MODULES[cle] = m
        rouges = [n for n, ok, _ in controles() if not ok]
    except Exception as e:  # noqa: BLE001 — une mutation qui fait planter un contrôle le fait rougir
        rouges = [f"exception {type(e).__name__}: {e}"]
    finally:
        for x, k in poses:
            setattr(x, k, orig)
        sys.modules[nom_mod] = orig
        if dans_reg:
            REG._MODULES[cle] = orig
    return rouges


PRm, MOm, DBm, NPm = "api_web.profil_resultat", "api_web.profil_mots", "api_web.dessin_barre", "api_web.note_profil"
MUTATIONS = [
    (PRm, "une méthode bloquée : la puce montre le taux pour information", "if g is not gouvernante(x) else", "if g.get(\"taux\") is not None else"),
    (MOm, "« bloqué » peint dans la couleur de son taux (vert)", "STATUTS_A_VERIFIER = (RE.BLOQUE, RE.NON_COUVERT, RE.HORS_DOMAINE)",
     "STATUTS_A_VERIFIER = (RE.NON_COUVERT, RE.HORS_DOMAINE)"),
    (PRm, "la ligne « n corrections des classeurs » oubliée", '    if not n_:\n        return ""\n    return (f\'<div class="el vc pf-lecarts"',
     '    if True:\n        return ""\n    return (f\'<div class="el vc pf-lecarts"'),
    (PRm, "la fenêtre des écarts ne garde que la première correction", '"niveau": "c", "etat_txt": "—"} for c in cs]', '"niveau": "c", "etat_txt": "—"} for c in cs[:1]]'),
    (PRm, "la fenêtre des écarts sans le lieu de la correction", '"sous": C.esc(ou_correction(c, titres))', '"sous": ""'),
    (PRm, "« où » remis dans la pastille de la dernière case (50 px au téléphone)", '"niveau": "c", "etat_txt": "—"} for c in cs]',
     '"niveau": "c", "etat_txt": ou_correction(c, titres)} for c in cs]'),
    (CSS_PROFIL, "la feuille garde les quatre cases vides de la fenêtre des écarts", "#ck-ecarts .vtab th:not(:first-child), #ck-ecarts .vtab td:not(:first-child) { display: none; }",
     ""),
    (CSS_PROFIL, "le texte d'une correction sur une seule ligne (débordement)", "#ck-ecarts .vtab td.nm { white-space: normal; overflow-wrap: anywhere; }", ""),
    (PRm, "le tableau avant/après jamais rendu", '    if L["decision"] != "raidisseur" or not s or not a:', "    if True:"),
    (PRm, "avant/après : N_b,Rd lu dans la colonne « sans »", '("N_b,Rd", rien, f"{M.r1(a[\'N_b_Rd\'])}', '("N_b,Rd", rien, f"{M.r1(s[\'F_Rd\'])}'),
    (PRm, "avant/après : le rappel sur τ oublié", 'rappel = f\'<span class="pf-rappel">{M.sub(L["rappel"])}</span>\' if L.get("rappel") else ""', 'rappel = ""'),
    (PRm, "l'en-tête d'un bloc de méthode redit le taux", "    return mot_methode(x)\n",
     '    return mot_methode(x) + (f" · {C.pc(C.pct(x[\'taux_max\']))} %" if x.get("taux_max") is not None else "")\n'),
    (PRm, "la ligne « Profilé » redit le taux de la barre", '<span>{C.esc(r["profil"])}</span>{C.icon("chevron-down")}</button></div></div>\')',
     '<span>{C.esc(r["profil"])}</span>{C.icon("chevron-down")}</button><span class="pct">{M.taux_txt(taux_barre(r))}</span></div></div>\')'),
    (PRm, "la masse rendue au pied de la colonne", '    s = ""\n    fl = (r.get("statique") or {}).get("fleche")',
     '    s = f\'<div class="mass"><b>{M.n(r.get("masse"), 1)} kg/m</b></div>\'\n    fl = (r.get("statique") or {}).get("fleche")'),
    (PRm, "le contrôle hérité H-EQ remis en tête du verdict ELS", '    if k != "historique" or g is None or g["id"] not in M.HERITES:', "    if True:"),
    (PRm, "le verdict ELS montre la plus FAIBLE des vérifications non héritées", 'return max(autres, key=lambda v: v["taux"]) if autres else g',
     'return min(autres, key=lambda v: v["taux"]) if autres else g'),
    (PRm, "le contrôle hérité plus fort tu sous le verdict", "                 and C.pct(v[\"taux\"]) > C.pct(t[\"taux\"])), None)", "                 and False), None)"),
    (PRm, "P_cr montré sans effort normal", '    sans_P = x.get("methode") == "historique" and not effort_normal(par, v)', "    sans_P = False"),
    (PRm, "P_cr caché aussi sous un effort normal", '    sans_P = x.get("methode") == "historique" and not effort_normal(par, v)',
     '    sans_P = x.get("methode") == "historique"'),
    (PRm, "P_cr décidé par les grandeurs de la méthode (la gouvernante), pas par la vérification", "and not effort_normal(par, v)",
     'and not (x.get("grandeurs") or {}).get("P")'),
    (PRm, "les hypothèses de la méthode redites dans chaque fenêtre de vérification",
     'f"{M.METHODES[k]} · {sous}" + (f" · {C.esc(ou)}" if ou else ""), [], [_ligne_tableau(v, True)], grp)',
     'f"{M.METHODES[k]} · {sous}" + (f" · {C.esc(ou)}" if ou else ""), hyps, [_ligne_tableau(v, True)], grp)'),
    (PRm, "« sans raidisseur » redit sur chaque ligne de l'âme", '    return etat_ame(a).startswith("Sans raidisseur")', "    return False"),
    (MOm, "-FM : la somme écrite comme un effort (« 0,5 ≤ 1,4 »)", "    if not v[\"unite\"]:                  # (lot C, C4)", "    if False:                  # (lot C, C4)"),
    (PRm, "-FM : la fenêtre écrit la somme au dixième", '    elif not v["unite"]:                       # (lot C, C4)', "    elif False:                       # (lot C, C4)"),
    (NPm, "-FM : la note écrit la somme comme un effort (« 0,5 ≤ 1,4 = 1,4 »)", "    if not u:                   # (lot C, C4)",
     "    if False:                   # (lot C, C4)"),
    (PRm, "la comparaison sans l'écart", '    e = li.get("ecart")\n    if e is None:', '    e = None\n    if e is None:'),
    (PRm, "la comparaison sans ce qui gouverne", "        if t and t.lower() != li[\"sujet\"].lower():", "        if False:"),
    (PRm, "« Tout le calcul » de l'Eurocode 3 sans le § 6.3.2.2 pour information", '    if info and info.get("M_b_Rd") is not None and', '    if False and'),
    (PRm, "… sans la variante de A_v,y", "    var = g.get(\"A_v_variante\")\n", "    var = None\n"),
    (PRm, "… sans n_pl avec l'autre γ_M", '    vg = (g.get("annexe_A") or {}).get("variante_gamma_M0")', "    vg = None"),
    (PRm, "n_pl avec l'autre γ_M dit deux fois (le pied du bloc aussi)", 'ALERTES_DANS_LE_CALCUL = ("npl_gamma_M0",)', "ALERTES_DANS_LE_CALCUL = ()"),
    (DBm, "le maintien négligé dessiné plein", "    peint = f'fill=\"#fff\" stroke=\"{DP.ACCENT}\" stroke-width=\"1.2\"' if neglige else", "    peint = f'fill=\"{DP.ACCENT}\"' if False else"),
    (DBm, "le niveau des charges jamais dessiné", "    meles = _niveaux_meles(el)\n", "    meles = False\n"),
    (DBm, "le niveau dessiné même quand il est le même partout", "    return len({ch.get(\"niveau\") or el[\"niveau\"] for c in el[\"cas\"] for ch in c[\"charges\"]}) > 1",
     "    return True"),
    (DBm, "les points vérifiés oubliés", "    for k, xp in enumerate(el.get(\"points\") or []):", "    for k, xp in enumerate([]):"),
    (DBm, "deux points proches : leurs lettres l'une sur l'autre", "        hauteur = hauteur + 1 if x - dernier < 16 else 0", "        hauteur = 0"),
    (DBm, "les débords sans cote", "        if b_ - a_ <= 1e-9:\n            continue", "        if True:\n            continue"),
    (DBm, "la barre dessinée sans ses débords (la portée seule)", "    K = Cadre(largeur, ST.longueur(el))", '    K = Cadre(largeur, sum(el["travees"]))'),
    (DBm, "les appuis sans le débord gauche", "    xs = ST.appuis_x(el)\n", '    xs = [sum(el["travees"][:i]) for i in range(len(el["travees"]) + 1)]\n'),
    (DBm, "N par tronçon oublié (un seul N, ou rien)", "    if c.get(\"N_troncons\"):\n        dessus += _N_troncons(c, y, K)", "    if False:\n        dessus += _N_troncons(c, y, K)"),
    (DBm, "N par tronçon sans la place de plus sous son cas", "        if c.get(\"N_troncons\"):\n            y += N_TRONCONS", "        if False:\n            y += N_TRONCONS"),
    (DBm, "la position de la barre tue", '    if el.get("position") and el["position"] != "horizontale":', "    if False:"),
]


def main():
    print("== 1 à 7. contrôles")
    for nom, ok, info in controles():
        chk(nom, ok, info)
    print("== 8. mutations")
    for nom_mod, nom, ancien, nouveau in MUTATIONS:
        rouges = muter(nom_mod, ancien, nouveau)
        if rouges is None:
            chk(f"mutation « {nom} » : le texte à muter est dans {nom_mod}, une seule fois", False, "texte absent ou répété")
        else:
            chk(f"mutation « {nom} » ({nom_mod}) : {len(rouges)} contrôle(s) rougissent", bool(rouges), "AUCUN contrôle ne rougit")
    chk("après les mutations, les modules sont rendus : les contrôles repassent tous", all(ok for _, ok, _ in controles()))
    print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
    for nom, info in KO:
        print("   -", nom, "|", str(info)[:300])
    return 1 if KO else 0


if __name__ == "__main__":
    sys.exit(main())

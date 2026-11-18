# -*- coding: utf-8 -*-
"""L'ÂME d'une barre vérifiée selon l'EN 1993-1-5 — le troisième bloc de résultats, à côté des deux méthodes, et
TOUJOURS calculé : la méthode historique ne traite ni le voilement de l'âme, ni la charge concentrée, ni les
raidisseurs (docs/acier/BARRES_ACIER.md § 5.9 ; brief § 6.5 et § 8.3).

    le voilement par CISAILLEMENT d'une âme élancée (§ 5, et § 7.1 avec la flexion), dans chaque section à vérifier ;
    l'âme sous une FORCE CONCENTRÉE (§ 6), à CHAQUE appui posé et sous CHAQUE charge ponctuelle qui appuie sur une
    semelle — seule (η₂), puis avec la flexion (§ 7.2) ;
    LA DÉCISION, lieu par lieu : « pas de raidisseur », avec les taux ; ou « raidisseur » — une paire de plats
    dimensionnée et vérifiée (raidisseurs.py, § 9) ;
    le voilement de l'âme induit par la semelle comprimée (§ 8).

Les efforts sont ceux de CALCUL : les combinaisons ELU (6.10) de la barre chargée, quelle que soit la méthode cochée.

OÙ UNE FORCE APPUIE SUR L'ÂME (les « lieux ») :
- un APPUI simple, quand la barre y est POSÉE (``appuis.mode`` « poses ») et que la réaction pousse la semelle basse ;
  un appui « par l'âme » (platine, cornières, gousset) n'appuie sur aucune semelle ; un encastrement est un assemblage,
  hors du module ; une réaction de soulèvement demande un ancrage, hors du module — signalée ;
- une CHARGE PONCTUELLE qui pousse la semelle où elle est posée : vers le bas sur la semelle haute, vers le haut sous
  la semelle basse (``niveau``) ; appliquée au centre de cisaillement, elle entre par l'âme.
Les charges ponctuelles de plusieurs cas à la même abscisse sont UNE force (la plus courte de leurs longueurs d'appui) ;
une charge au droit d'un appui partage son lieu : chaque semelle chargée y est vérifiée, et la force qui passe d'une
semelle à l'autre — la plus petite des deux — l'est en type (b).

LE TYPE (figure 6.1) se déduit de la géométrie : (c) près d'un bout de la barre, tant que 2 + 6·(s_s + c)/h_w < 6 —
c étant la distance du bout de la barre au bord de la longueur chargée, avec le débord saisi pour un appui de rive — ;
(a) ailleurs.

UN RAIDISSEUR EST NÉCESSAIRE dès qu'une combinaison dépasse η₂ = 1 ou (η₂ + 0,8·η₁)/1,4 = 1 ; et à chaque appui
posé d'une âme élancée au cisaillement (§ 5.1(2)). Il est dimensionné pour la plus grande force du lieu.

UNE DONNÉE QUI MANQUE (brief § 8.3, point 5 ; § 8.2 : s_s jamais supposée) — la longueur d'appui s_s, le débord c d'un
appui de rive (sauf quand la force y est du type (a) à c = 0 : c n'y change rien). La BORNE — la donnée à 0 et, au type
(c), la branche m₂ = 0, cohérente ou non (ec3_ame.resistance, borne) — donne une F_Rd sous la F_Rd réelle, quels que soient
s_s et c (prouvé par balayage, tests/test_barre_ame.py ; F_Rd elle-même n'est pas monotone au type (c)) : le lieu est
vérifié à la borne. Tout passe → « pas de raidisseur », borne dite, taux donnés ; sinon « décision impossible : s_s à
saisir », le taux à la borne sur la même ligne (À VÉRIFIER : lecture du point 5) — n'y sont demandées que les données
dont dépend ce qui ne passe pas (le critère 6.2.1(5) ne lit que s_s) ; s_s saisie et le critère seul en cause : le
raidisseur est sûr. Le MAINTIEN latéral de la semelle chargée (§ 6.1(1)) n'a pas de borne — tenue à un appui (fourche)
ou par un maintien « complet » ou de cette semelle à la même abscisse ; quand elle n'est pas comprimée partout, l'autre
aussi ; pour un raidisseur sous une charge, les DEUX semelles (longueur 0,75·h_w, § 9.4(2)) : sans lui, « décision
impossible ». Le lieu impossible est « non couvert », jamais « pas de raidisseur ».

Ce que ce bloc ne fait pas (v1), et dit : la charge concentrée sur un profil creux (EN 1993-1-8) ; les raidisseurs
intermédiaires d'une âme élancée (elle est vérifiée sans eux) ; les soudures des raidisseurs (EN 1993-1-8) ; DEUX FORCES
RAPPROCHÉES sur la même semelle — leurs longueurs chargées l_y se recouvrent —, que le § 6.3(2) demande de vérifier aussi
ENSEMBLE (leur somme, sur leur entraxe) : la vérification « A-GR » est alors rendue « non couvert », jamais « ok ».
"""
from typing import Dict, List, Optional, Sequence, Tuple

from acier.flambement import normes as NO

from . import combinaisons as CO
from . import ec3_ame as AM
from . import ec3_section as ES
from . import parametres as PA
from . import raidisseurs as RA
from . import resultat as RE
from . import statique as ST
from .enveloppe import Enveloppe

METHODE, NORME = AM.METHODE, AM.NORME
EPS = 1e-9
INTRODUCTIONS_DECLAREES = ("a", "b", "c")      # le mode d'introduction d'une ponctuelle saisi (« auto » : la règle du module)
MEME_ABSCISSE = 1e-6             # m : deux forces à la même abscisse
MODES_APPUI = ("poses", "ame")
AUTRE = {"haut": "bas", "bas": "haut"}
# une donnée non saisie prise à sa borne : dite une fois dans les hypothèses du bloc (règle 1 du brief : elle change la décision ;
# la lecture du brief § 8.3, point 5, À VÉRIFIER — le brief, document interne, n'est pas cité dans le texte rendu)
BORNE_HYPOTHESE = "s_s ou c non saisi : calculé à sa borne 0 (À VÉRIFIER)."
# le critère 6.2.1(5), s_s non saisie, pris au bout h_w (le pire des deux) : la fenêtre imprime s_s = h_w, elle le dit
MOT_PIRE_HW = "s_s non saisie : le pire de 0 et h_w."
# la semelle chargée sans maintien au droit de la force : la section 6 la suppose tenue latéralement (EN 1993-1-5, 6.1(1)) —
# règle prudente gardée (À VÉRIFIER, à trancher : brief § 11.3, R2 « pas de raidisseur ») ; le fait, le taux, le remède
REF_MAINTIEN = f"{AM.REF}, 6.1(1)"
# le raidisseur sous une charge : sa longueur 0,75·h_w suppose les DEUX semelles tenues (§ 9.4(2)) — le fait, la décision et
# le remède, sur le modèle de sans_maintien (intégration du lot B) : « Semelle basse sans maintien (EN 1993-1-5, 9.4(2)) :
# raidisseur nécessaire ; ajouter un maintien complet (À VÉRIFIER). »
REF_RAIDISSEUR_TENU = f"{AM.REF}, 9.4(2)"
MOT_RAIDISSEUR_TENU = "raidisseur nécessaire ; ajouter un maintien complet (À VÉRIFIER)."
# une âme ÉLANCÉE au cisaillement : des raidisseurs transversaux sont dus à chaque appui (EN 1993-1-5 § 5.1(2)) — que la barre y
# soit posée ou tenue par l'âme (lot C, C1 point 15 : un appui « par l'âme » n'en demandait aucun, défaut NON CONSERVATIF —
# le voilement par cisaillement A-Vb est calculé AVEC ces raidisseurs d'appui) ; une seule phrase pour les deux modes
RAISON_APPUI_ELANCEE = "Âme élancée au cisaillement : raidisseurs aux appuis (§ 5.1(2))."
# l'appui par l'âme : le raidisseur d'appui porte la réaction entière (hypothèse prudente : l'assemblage la donne à l'âme)
HYP_APPUI_AME = "Appui par l'âme, âme élancée : raidisseur d'appui requis (§ 5.1(2)), sous la réaction entière (À VÉRIFIER)."


def raidisseur_tenu(libres: Sequence[str]) -> str:
    """La raison d'un lieu qui demande un raidisseur sous une charge dont une semelle (``libres`` : « haut », « bas ») n'est
    pas tenue : la ou les semelles nommées, la clause, puis MOT_RAIDISSEUR_TENU (la décision et le remède)."""
    quoi = "Semelles haute et basse" if len(libres) == 2 else f"Semelle {'haute' if libres[0] == 'haut' else 'basse'}"
    return f"{quoi} sans maintien ({REF_RAIDISSEUR_TENU}) : {MOT_RAIDISSEUR_TENU}"


def _fy(b: Dict, s: Dict) -> Tuple[float, float, Optional[str]]:
    """(f_y de calcul de la barre, f_y de son âme selon t_w, d'où vient la première) — la table unique, lue par
    ec3_section.lire ; Refus si la nuance ou l'épaisseur en sortent."""
    d = ES.lire({"section": s, "nuance": b["nuance"], "parametres": b["parametres"]})
    return d["fy"], d["fy_parois"]["ame"], d["fy_source"]


def _fy_semelle(b: Dict, s: Dict) -> float:
    """f_y,f — la f_y des semelles selon t_f (table unique) : celle du voilement induit par la semelle (§ 8)."""
    d = ES.lire({"section": s, "nuance": b["nuance"], "parametres": b["parametres"]})
    return d["fy_parois"].get("semelle", d["fy"])


def refusee(message: str, statut: str = RE.NON_COUVERT) -> Dict:
    """Le bloc d'une barre dont l'âme ne peut pas être vérifiée (une épaisseur au-delà de 80 mm (domaine du module)…) : « non couvert » ;
    ``statut`` « bloqué » : une valeur à fournir (le ψ₀ d'une exécution ou de machines, combinaisons.statut_refus)."""
    J = RE.Journal(METHODE, NORME)
    J.verif("A-AME", "Vérification impossible", NORME, "", None, None, "", [], statut=statut,
            messages=[J.alerte("a_fournir" if statut == RE.BLOQUE else "refus", "attention", message)])
    r = J.rendre()
    r.update({"combinaison": None, "combinaisons": [], "a_confirmer": [], "lieux": []})
    return r


def _communes(b: Dict, s: Dict, fy: float, P: Dict, fyw: Optional[float] = None, source: Optional[str] = None,
              fyf: Optional[float] = None) -> Tuple[List[str], Optional[Dict], Optional[Dict]]:
    """Les hypothèses que les deux modes disent, l'élancement de l'âme (None : sans objet) et le voilement induit par
    la semelle (None hors des I). ``fyw`` : la f_y de l'âme selon t_w ; ``source`` : d'où vient la f_y de calcul ;
    ``fyf`` : celle des semelles (§ 8)."""
    fyw = fy if fyw is None else fyw
    hyp = [f"Efforts de calcul (ELU) ; γ_M0 = {RE.fr(P['gamma_M0'], 2)}, γ_M1 = {RE.fr(P['gamma_M1'], 2)}.",
           f"Acier {b['nuance']} : f_y = {RE.fr(fy)} N/mm²" + (f" ({source})" if source else "") + f" ; η = {RE.fr(AM.eta_de(s, P), 2)}."]
    if fyw != fy:                       # (lot C, C5 : une ligne par fait, 120 caractères au plus)
        hyp.append(f"Âme : f_y,w = {RE.fr(fyw)} N/mm² (son ε pour 72·ε/η, λ̄_w et λ̄_F).")
    el = AM.elancement(s, fy, P, fy_ame=fyw)
    if el is not None:
        if el["elancee"]:
            hyp.append(f"Âme élancée (h_w/t_w = {RE.fr(el['rapport'], 1)} > 72·ε/η = {RE.fr(el['limite'], 1)}) : voilement par cisaillement "
                       "vérifié, avec la flexion.")
            hyp.append("Voilement par cisaillement : montant d'extrémité non rigide, sans la contribution des semelles (côté sûr).")
        else:
            hyp.append(f"Âme : h_w/t_w = {RE.fr(el['rapport'], 1)} ≤ 72·ε/η = {RE.fr(el['limite'], 1)} — pas de voilement par cisaillement "
                       "à vérifier (§ 5.1(2)).")
    sm = AM.semelle(s, fy, P, fy_semelle=fyf)
    if sm is not None and sm["ok"]:
        hyp.append(f"Voilement de l'âme induit par la semelle (§ 8) : h_w/t_w = {RE.fr(sm['rapport'], 1)} ≤ {RE.fr(sm['limite'])} — "
                   "sans objet.")
    return hyp, el, sm


# ------------------------------------------------------------------ efforts imposés
def efforts(b: Dict, s: Dict) -> Dict:
    """Le bloc de l'âme d'une barre aux EFFORTS IMPOSÉS : le voilement par cisaillement sous les efforts de calcul. Les
    appuis et les charges ne sont pas connus : aucune charge concentrée n'est vérifiée, et le bloc le dit."""
    try:
        P = PA.jeu(b["parametres"])
        fy, fyw, src = _fy(b, s)
    except RE.Refus as e:
        return refusee(str(e))
    e = b["efforts"]["elu"]
    J = RE.Journal(METHODE, NORME)
    hyp, _, sm = _communes(b, s, fy, P, fyw, src, _fy_semelle(b, s))
    AM.cisaillement(J, s, fy, P, abs(e["Vz"]) * 1e3, abs(e["My"]) * 1e6, e["N"] * 1e3, fy_ame=fyw)
    AM.cisaillement(J, s, fy, P, abs(e["Vy"]) * 1e3, abs(e["Mz"]) * 1e6, e["N"] * 1e3, direction="y", fy_ame=fyw)
    if sm is not None and not sm["ok"]:
        AM.ecrire_semelle(J, s, fy, P, sm)
    hyp.append("Efforts imposés : les appuis et les charges ne sont pas décrits — l'âme sous une force concentrée n'est pas vérifiée.")
    r = J.rendre()
    r.update({"hypotheses": hyp, "combinaison": None, "combinaisons": [], "a_confirmer": [], "lieux": []})
    return r


# ------------------------------------------------------------------ barre chargée : les lieux
def lieux(b: Dict, cas: Sequence[Dict], sols: Sequence, ame_elancee: bool = False) -> List[Dict]:
    """Les endroits où une force appuie sur l'âme, de gauche à droite : {"base" (préfixe de ses vérifications), "nom",
    "genre" (« appui » | « charge »), "x" (m), "appui" (son rang, ou None), "charges" [(rang du cas, valeur kN)],
    "ss" {"haut", "bas"} (mm ; None : semelle non chargée ; 0, sa borne, quand elle n'est pas saisie), "manque" {"haut", "bas"}
    (les données non saisies de chaque semelle chargée : « s_s », « c »), "tenue" {"haut", "bas"} (la semelle est tenue
    latéralement au droit du lieu), "dispo" (mm d'âme jusqu'à chaque bout de la barre)}. ``ame_elancee`` : une âme élancée au
    cisaillement — chaque appui tenu PAR L'ÂME est aussi un lieu (« par_ame » : aucune force sur une semelle, mais le raidisseur
    d'appui du § 5.1(2) y est dû)."""
    Lt = ST.longueur(b)                              # la barre, débords compris (son premier appui en x = g)
    ap = b["appuis"]
    appuis = list(sols[0].appuis) if sols else []
    poses = ap["mode"] == "poses"
    detail = ap.get("detail") or []

    def donnee(i: Optional[int], cle: str, defaut=None):
        """s_s, c ou le contact de l'appui ``i`` : saisi APPUI PAR APPUI (``appuis.detail``, lot C), sinon la valeur commune."""
        d = detail[i] if i is not None and i < len(detail) else None
        if d is not None and cle in d:
            return d[cle]
        return ap.get(cle, defaut)

    def rive(x: float) -> bool:
        return abs(x) < MEME_ABSCISSE or abs(x - Lt) < MEME_ABSCISSE

    def de_rive(k: int) -> Optional[int]:
        """Le rang de l'appui POSÉ au bout ``k`` de la barre (0 : gauche, -1 : droite) ; None : pas d'appui posé au bout (un
        débord, un encastrement)."""
        x_bout = 0.0 if k == 0 else Lt
        if not appuis or not poses:
            return None
        i = 0 if k == 0 else len(appuis) - 1
        a = appuis[i]
        return i if a["type"] == "appui" and abs(a["x"] - x_bout) < MEME_ABSCISSE else None

    def depassement(k: int) -> float:
        """L'âme au-delà de l'axe d'un appui de rive posé : la moitié de sa longueur d'appui, et son débord — non
        saisis, 0 : la borne pour la force voisine (À VÉRIFIER : son lieu est vérifié à la borne, brief § 8.3, point 5)."""
        i = de_rive(k)
        if i is None:
            return 0.0
        return (donnee(i, "ss") or 0.0) / 2.0 + (donnee(i, "c") or 0.0)

    def tenue(x: float, cote: str) -> bool:
        """La semelle ``cote`` est-elle tenue latéralement en x : par un appui (une fourche, quel que soit son mode), ou par un
        maintien « complet » ou de cette semelle à la même abscisse (§ 6.1(1) ; brief § 8.3, point 5) — À VÉRIFIER : sans
        eux, décision impossible."""
        if any(abs(a["x"] - x) < MEME_ABSCISSE for a in appuis):
            return True
        return any(abs(m["x"] - x) < MEME_ABSCISSE and m.get("genre", "complet") in ("complet", cote) for m in b.get("maintiens") or [])

    def manque_rive(k: int) -> List[str]:
        """Les données non saisies de l'appui de rive posé au bout ``k`` (0 : gauche, -1 : droite) : elles font le c d'une
        force voisine du type (c)."""
        i = de_rive(k)
        if i is None:
            return []
        return (["s_s"] if donnee(i, "ss") is None else []) + (["c"] if donnee(i, "c") is None else [])

    out: List[Dict] = []
    if poses:
        for i, a in enumerate(appuis):
            if a["type"] == "appui":                 # un appui : une fourche, les deux semelles tenues (le déversement le suppose)
                ss_i, c_i = donnee(i, "ss"), donnee(i, "c")
                manque = (["s_s"] if ss_i is None else []) + (["c"] if c_i is None and rive(a["x"]) else [])
                out.append({"x": a["x"], "appui": i, "nom": f"Appui {a['nom']}", "genre": "appui", "charges": [],
                            "ss": {"haut": None, "bas": ss_i or 0.0}, "contact": {"haut": None, "bas": donnee(i, "contact", "inconnu")},
                            "manque": {"haut": [], "bas": manque}, "tenue": {"haut": True, "bas": True}})
    elif ame_elancee:
        for i, a in enumerate(appuis):
            if a["type"] == "appui":                 # tenu par l'âme : aucune semelle chargée, le raidisseur d'appui dû (§ 5.1(2))
                out.append({"x": a["x"], "appui": i, "nom": f"Appui {a['nom']}", "genre": "appui", "charges": [], "par_ame": True,
                            "ss": {"haut": None, "bas": None}, "contact": {"haut": None, "bas": None},
                            "manque": {"haut": [], "bas": []}, "tenue": {"haut": True, "bas": True}})
    # les charges ponctuelles qui appuient sur une semelle : chacune à SON niveau (lot C : une hauteur par charge ; à défaut celui
    # de la barre) ; au centre de cisaillement, elle entre par l'âme
    groupes: Dict[float, Dict] = {}
    for ci, c in enumerate(cas):
        for ch in c.get("charges") or []:
            niv = ch.get("niveau") or b["niveau"]
            if ch["type"] == "P" and ch["v"] and niv in ("haut", "bas"):
                g = groupes.setdefault(round(ch["a"], 6), {"x": ch["a"], "parts": [], "ss": {"haut": [], "bas": []},
                                                           "contact": {"haut": [], "bas": []}, "intro": []})
                g["parts"].append((ci, ch["v"], niv))
                g["ss"][niv].append(ch.get("ss"))
                g["contact"][niv].append(ch.get("contact", "inconnu"))
                g["intro"].append(ch.get("introduction", "auto"))
    for g in sorted(groupes.values(), key=lambda g: g["x"]):
        hote = next((L for L in out if L["appui"] is not None and not L.get("par_ame") and abs(L["x"] - g["x"]) < MEME_ABSCISSE), None)
        if hote is None:
            hote = {"x": g["x"], "appui": None, "nom": f"Charge à {RE.fr(g['x'], 2)} m", "genre": "charge", "charges": [],
                    "ss": {"haut": None, "bas": None}, "contact": {"haut": None, "bas": None}, "manque": {"haut": [], "bas": []},
                    "tenue": {k: tenue(g["x"], k) for k in ("haut", "bas")}}
            out.append(hote)
        hote["charges"] = g["parts"]
        # le mode d'introduction DÉCLARÉ (EN 1993-1-5, figure 6.1 ; lot C) : « auto », la règle du module ; plusieurs charges au
        # même endroit qui en déclarent deux différents : la règle du module (À VÉRIFIER)
        declares = {m for m in g["intro"] if m != "auto"}
        hote["introduction"] = declares.pop() if len(declares) == 1 else "auto"
        for niv in ("haut", "bas"):
            if not g["ss"][niv] or hote["ss"][niv] is not None:
                continue
            # plusieurs charges : la plus courte de leurs longueurs d'appui — inconnue dès qu'une n'est pas saisie : la
            # borne 0 pour le groupe, jamais la plus courte des SAISIES (la manquante peut être plus courte)
            connue = None not in g["ss"][niv]
            hote["ss"][niv] = min(g["ss"][niv]) if connue else 0.0
            hote["manque"][niv] = [] if connue else ["s_s"]
            # le contact de plusieurs charges : établi s'il l'est pour toutes ; « non » dès qu'une le refuse
            k_ = g["contact"][niv]
            hote["contact"][niv] = "oui" if all(k == "oui" for k in k_) else "non" if "non" in k_ else "inconnu"
    out.sort(key=lambda L: L["x"])
    e_g, e_d = depassement(0), depassement(-1)
    for k, L in enumerate(out):
        L["base"] = f"A-L{k:02d}"
        L["dispo"] = (L["x"] * 1e3 + e_g, (Lt - L["x"]) * 1e3 + e_d)
        # le bout le plus proche : ses données manquantes font le c d'une force du type (c) — l'appui de rive lui-même les porte déjà
        L["manque_bout"] = [] if L["genre"] == "appui" else manque_rive(0 if L["dispo"][0] <= L["dispo"][1] else -1)
    return out


def forces(L: Dict, niveau: str, sols: Sequence, facteurs: Sequence[float]) -> Dict[str, float]:
    """Les forces qui APPUIENT sur chaque semelle du lieu sous une combinaison (kN, ≥ 0) : {"haut", "bas"} ; et
    "soulevement" (kN > 0) quand la réaction de l'appui tire la barre vers le bas. Chaque charge à SON niveau (``lieux`` :
    (rang du cas, valeur, niveau)) ; ``niveau`` : celui d'une charge qui n'en porte pas."""
    P = {"haut": 0.0, "bas": 0.0}
    for part in L["charges"]:
        niv = part[2] if len(part) > 2 else niveau
        if niv in P:
            P[niv] += facteurs[part[0]] * part[1]
    R = 0.0
    if L["appui"] is not None:
        R = sum(f * sol.appuis[L["appui"]]["R"] for f, sol in zip(facteurs, sols) if f)
    return {"haut": P["haut"] if P["haut"] > EPS else 0.0,
            "bas": (R if R > EPS else 0.0) + (-P["bas"] if P["bas"] < -EPS else 0.0),
            "soulevement": -R if R < -EPS else 0.0}


def resistances(L: Dict, s: Dict, fy: float, P: Dict, fyw: Optional[float] = None) -> Dict[str, Dict]:
    """La résistance de l'âme SEULE au lieu ``L``, par semelle chargée (« haut », « bas ») et, quand les deux le sont,
    pour la force transmise de l'une à l'autre (« b ») — acier/barre/ec3_ame.resistance, contact d'extrémité non établi
    (le verdict du type (c) ; la variante avec contact y est donnée pour information) ; ``fyw`` : la f_y de l'âme.
    Chaque résistance garde l'âme qui l'a donnée (« ame » : b_f et t_f de la semelle chargée, brief § 8.2) ; la force
    transmise d'un PRS à semelles inégales prend la semelle qui donne la plus petite F_Rd (choix prudent déclaré).
    Chacune dit aussi ses données non saisies qui COMPTENT (« manque » : « s_s », « c » ; « rive » : celles de l'appui de
    rive qui font le c d'une force du type (c)) : elle est alors calculée à leur BORNE 0 (``ec3_ame.resistance``, borne)."""
    d = min(L["dispo"])
    out = {}
    for cote, ss in L["ss"].items():
        if ss is None:
            continue
        a = AM.ame_de(s, cote)
        hw = a["hw"]
        c = max(d - ss / 2.0, 0.0)
        type_ = "c" if 2.0 + 6.0 * (min(ss, hw) + c) / hw < AM.KF_C_MAX else "a"
        if L.get("genre") == "charge" and (L.get("introduction") or "auto") in INTRODUCTIONS_DECLAREES:
            type_ = L["introduction"]                    # le mode DÉCLARÉ d'une ponctuelle, appliqué tel quel (lot C)
        # c ne change rien à une force du type (a) à c = 0 : elle le reste pour tout c ≥ 0, et F_Rd n'en dépend pas
        manque = [m for m in ((L.get("manque") or {}).get(cote) or []) if not (m == "c" and type_ == "a")]
        rive = list(L.get("manque_bout") or []) if type_ == "c" else []
        contact = (L.get("contact") or {}).get(cote) or "inconnu"
        out[cote] = dict(AM.resistance(a, fy, P, ss, type_, c, contact=contact, fy_ame=fyw, borne=bool(manque or rive)), ame=a,
                         manque=manque, rive=rive)
    if len(out) == 2:
        manque = ["s_s"] if any("s_s" in out[k]["manque"] for k in ("haut", "bas")) else []     # le type (b) ne lit pas c
        essais = [dict(AM.resistance(AM.ame_de(s, k), fy, P, min(L["ss"]["haut"], L["ss"]["bas"]), "b", fy_ame=fyw, borne=bool(manque)),
                       ame=AM.ame_de(s, k), manque=manque, rive=[]) for k in ("haut", "bas")]
        out["b"] = min(essais, key=lambda r: r["FRd"])
    return out


def borne(r: Dict) -> Optional[str]:
    """Ce que la borne prend pour la force vérifiée ``r`` (``resistances``) : « s_s = 0, c = 0 » ; None quand rien ne manque."""
    parts = [f"{m} = 0" for m in r.get("manque") or []] + ([f"{_et(r['rive'])} de l'appui de rive = 0"] if r.get("rive") else [])
    return ", ".join(parts) or None


def portee(r: Dict) -> float:
    """La longueur chargée l_y de la force ``r`` pour chercher les forces rapprochées (§ 6.3(2)) : la sienne ; quand une
    donnée manque, la plus grande possible (ec3_ame.ly_plafond, s_s = h_w s'il manque) — la borne basse de l_y y serait
    du côté dangereux (À VÉRIFIER)."""
    if not (r.get("manque") or r.get("rive")):
        return r["ly"]
    return AM.ly_plafond(r["ame"], r["ame"]["hw"] if "s_s" in r["manque"] else r["ss"])


def _titre(L: Dict, cote: str) -> str:
    if cote == "b":
        return f"{L['nom']} : force transmise d'une semelle à l'autre"
    if L["genre"] == "charge":
        return f"{L['nom']} : âme sous la charge"
    return f"{L['nom']} : âme sous la réaction" if cote == "bas" else f"{L['nom']} : âme sous la charge appliquée"


def _saisie(L: Dict, cote: str) -> float:
    """La longueur d'appui SAISIE de la force vérifiée (mm) : celle de sa semelle ; la plus courte des deux pour la
    force transmise."""
    return min(L["ss"].values()) if cote == "b" else L["ss"][cote]


def _mots_force(r: Dict, saisie: float) -> List[str]:
    """Ce qu'on dit de la force vérifiée : ses données non saisies (leur borne), ou sa longueur d'appui bornée à h_w ;
    puis le type de la force et les choix du calcul (ec3_ame.mots_resistance)."""
    mots = []
    b = borne(r)
    if b:
        mots.append(f"Non saisi{'s' if ',' in b or ' et ' in b else ''} : {b} (borne prudente).")
    elif saisie > r["ss"]:
        mots.append(f"s_s bornée à h_w = {RE.fr(r['ss'], 1)} mm.")
    return mots + AM.mots_resistance(r)


def _et(mots: Sequence[str]) -> str:
    return " et ".join(mots) if len(mots) <= 2 else ", ".join(mots[:-1]) + " et " + mots[-1]


def _sans_maintien(cotes: Sequence[str]) -> str:
    """« semelle haute sans maintien (EN 1993-1-5, 6.1(1)) » ; les deux : « semelles haute et basse … »."""
    quoi = "semelles haute et basse" if len(cotes) == 2 else f"semelle {'haute' if cotes[0] == 'haut' else 'basse'}"
    return f"{quoi} sans maintien ({REF_MAINTIEN})"


def sans_maintien(cotes: Sequence[str], taux: float, ligne: bool = False) -> str:
    """La raison d'un lieu dont SEUL le maintien manque (``cotes`` : les semelles à tenir, « haut », « bas ») — le fait, le
    taux obtenu comme si elles étaient tenues (les mêmes vérifications : le maintien n'y entre pas) et le remède, en une
    ligne : « Semelle haute sans maintien (EN 1993-1-5, 6.1(1)) — 37 % si elle est tenue ; ajouter un maintien (À VÉRIFIER). »
    Au-delà de 100 %, la force demande, tenue, un raidisseur — dont la longueur suppose les DEUX semelles tenues (§ 9.4(2)) :
    le remède est alors « ajouter un maintien complet et un raidisseur » (un profilé trop étroit pour en recevoir un le dit une
    fois tenu). ``ligne`` : sans le taux, que sa pastille donne déjà (la ligne du lieu, sa fenêtre -F, le tableau de la note)."""
    t = _sans_maintien(cotes)
    tenue = "si elles sont tenues" if len(cotes) == 2 else "si elle est tenue"
    remede = "ajouter un maintien complet et un raidisseur" if taux > 1.0 else "ajouter un maintien"
    taux_ = "" if ligne else f" — {RE.fr(100.0 * taux)} % {tenue}"
    return f"{t[:1].upper() + t[1:]}{taux_} ; {remede} (À VÉRIFIER)."


# η₁ de l'interaction (7.2), dit dans l'hypothèse du bloc : la DÉCISION BUREAU n° 1 lue dans la table (acier/barre/parametres.py)
ETA1_HYP = {"elastique": "η₁ élastique", "plastique": "η₁ plastique en classes 1 et 2 (décision du bureau)"}
DECISION_IMPOSSIBLE = "Décision impossible : "         # le début de la raison d'une décision impossible (api_web/profil_mots le lit)


def impossible(donnees: Sequence[str], maintiens: Sequence[str], rive: Sequence[str] = ()) -> str:
    """La raison d'une décision impossible faute de données (brief § 8.3, point 5), en une phrase : « Décision impossible :
    s_s à saisir. » ``maintiens`` : les semelles qui ne sont pas tenues non plus (« haut », « bas ») ; ``rive`` : les données
    manquantes de l'appui de rive voisin (le c d'une force du type (c))."""
    quoi = ([_et(donnees)] if donnees else []) + ([f"{_et(rive)} de l'appui de rive"] if rive else [])
    parts = ([f"{' ; '.join(quoi)} à saisir"] if quoi else []) + ([_sans_maintien(maintiens)] if maintiens else [])
    return DECISION_IMPOSSIBLE + " ; ".join(parts) + "."


def critere_pire(F: float, essais: Sequence[float], tf: float, tw: float, sx: float, tau: float, fy: float, gM0: float) -> Dict:
    """Le critère 6.2.1(5) à la jonction sous la force F (N), au PIRE des longueurs d'appui ``essais`` (mm) : la saisie ; ou,
    s_s non saisie, ses deux bouts 0 et h_w — le critère est quadratique en σ_z, son plus grand sur l'intervalle est à un
    bout. {"ss", "sz", "se", "crit"}."""
    pire = None
    for u in essais:
        sz = AM.sigma_z(F, u, tf, tw)
        crit = AM.critere_elastique(sx, sz["sigma_z"], tau, fy, gM0)
        if pire is None or crit > pire["crit"] + EPS:
            pire = {"ss": u, "sz": sz["sigma_z"], "se": sz["se"], "crit": crit}
    return pire


def cisaillement_panneaux(x0: float, bords: Sequence[float], combs: Sequence, hw: float) -> float:
    """V_Ed de N_st (kN) au raidisseur en ``x0`` (m) : dans chacun des deux panneaux qu'il borde (``bords`` : les
    raidisseurs et les bouts de la barre), à 0,5·h_w (``hw`` en m) du bord le plus sollicité — NOTE du § 9.3.3(3), brief
    § 8.4 ; le plus grand des deux panneaux, sur toutes les combinaisons (À VÉRIFIER : enveloppe, F_Ed et N_st pris
    chacun à leur maximum ; un panneau plus court que h_w : à mi-longueur)."""
    gauche = max((u for u in bords if u < x0 - MEME_ABSCISSE), default=None)
    droite = min((v for v in bords if v > x0 + MEME_ABSCISSE), default=None)
    pire = 0.0
    for u, v in ((gauche, x0), (x0, droite)):
        if u is None or v is None:
            continue
        d = min(0.5 * hw, (v - u) / 2.0)
        for C in combs:
            Vu, Vv = abs(C.lire(u)["V_d"]), abs(C.lire(v)["V_g"])
            e = C.lire(u + d if Vu >= Vv else v - d)
            pire = max(pire, abs(e["V_g"]), abs(e["V_d"]))
    return pire


# ------------------------------------------------------------------ barre chargée
def chargee(b: Dict, s: Dict, cas: Sequence[Dict], sols: Sequence, N_cas: Sequence[float], pts: Sequence[float]) -> Dict:
    """Le bloc de l'âme d'une barre CHARGÉE : la forme de acier/barre/resultat.py (chaque vérification dit sa combinaison
    et son abscisse), et "lieux" — la décision de chaque appui et de chaque charge ponctuelle :

        {"base", "nom", "genre", "x", "type", "ss" (mm), "F_Ed" (kN, la plus grande force du lieu), "sans": {"F_Rd" (kN),
         "eta2", "interaction" (None sans flexion)} (l'âme seule, au plus défavorable), "decision" (« aucun » |
         « raidisseur » | « impossible »), "raison", "ligne" (seul le maintien manque : la raison sans son taux, que la pastille
         de la ligne donne ; None sinon), "borne" (« s_s = 0, c = 0 : borne prudente » quand une donnée non
         saisie est prise à sa borne, None sinon), "raidisseur" (None | {"bs", "ts", "chanfrein", "designation"}),
         "verifications" (leurs identifiants), "gouvernant", "taux" (aussi d'un lieu impossible, pour information),
         "statut"}."""
    try:
        P = PA.jeu(b["parametres"])
        fy, fyw, src = _fy(b, s)
    except RE.Refus as e:
        return refusee(str(e))
    gM0 = P["gamma_M0"]
    env = Enveloppe(METHODE, NORME)
    hyp, el, sm = _communes(b, s, fy, P, fyw, src, _fy_semelle(b, s))
    alertes: Dict[str, Dict] = {}
    # (lot C, C0e) les combinaisons de la BARRE : l'âme suit son expression ((6.10), ou (6.10a)/(6.10b) avec C2)
    try:
        combs = [ST.Combinaison(co["nom"], sols, co["facteurs"], N_cas) for co in CO.elu(cas, P, b.get("combinaisons"))]
    except RE.Refus as e:                       # (intégration du lot C) un ψ₀ à fournir : l'âme est BLOQUÉE, jamais une exception
        return refusee(str(e), CO.statut_refus(e))

    # ---- le voilement par cisaillement, dans chaque section à vérifier
    if el is not None and el["elancee"]:
        for C in combs:
            for x in ST.sections(b, C, pts):
                for M, V, N in ST.cotes_n(C, x):        # (N de la section : lot C, N par tronçon)
                    if abs(V) < EPS:
                        continue
                    J = RE.Journal(METHODE, NORME)
                    AM.cisaillement(J, s, fy, P, abs(V) * 1e3, abs(M) * 1e6, N * 1e3, fy_ame=fyw)
                    env.verser(J.rendre(), {"combinaison": C.nom, "x": x})
    if sm is not None and not sm["ok"]:
        J = RE.Journal(METHODE, NORME)
        AM.ecrire_semelle(J, s, fy, P, sm)
        env.verser(J.rendre(), {"combinaison": combs[0].nom})

    # ---- les forces concentrées : où, combien, sous quelle combinaison
    a = AM.ame_de(s)
    endroits = lieux(b, cas, sols, ame_elancee=el is not None and el["elancee"] and a is not None)
    par_ame = [L for L in endroits if L.get("par_ame")]          # les appuis tenus par l'âme d'une âme élancée (§ 5.1(2))
    ap = b["appuis"]
    genres = {x["type"] for x in (sols[0].appuis if sols else [])}
    if genres - {"appui"}:
        hyp.append("Encastrement : un assemblage, hors du module — aucune force concentrée n'y est vérifiée.")
    if "appui" in genres and ap["mode"] == "poses":
        if any(ap.get("detail") or []):                 # (lot C) s_s, c et contact saisis appui par appui : le lieu les dit
            hyp.append("Barre posée sur ses appuis : longueur d'appui et débord saisis appui par appui.")
        else:
            ss_t = "s_s non saisie" if ap["ss"] is None else f"s_s = {RE.fr(ap['ss'])} mm"
            c_t = "c non saisi" if ap["c"] is None else f"c = {RE.fr(ap['c'])} mm"
            hyp.append(f"Barre posée sur ses appuis : longueur d'appui {ss_t}, débord en rive {c_t}.")
    elif "appui" in genres:
        hyp.append("Appuis par l'âme (platine, cornières, gousset) : aucune force concentrée sur l'âme aux appuis.")
        if par_ame:
            hyp.append(HYP_APPUI_AME)
    niveaux_P = {ch.get("niveau") or b["niveau"] for c in cas for ch in c.get("charges") or [] if ch["type"] == "P"}
    for niv in ("haut", "bas", "centre"):            # une hypothèse par niveau (lot C : une hauteur par charge)
        if niv in niveaux_P:
            hyp.append({"haut": "Charges ponctuelles posées sur la semelle haute.",
                        "bas": "Charges ponctuelles suspendues à la semelle basse : seule une charge vers le haut appuie sur l'âme.",
                        "centre": "Charges ponctuelles appliquées au centre de cisaillement : elles entrent par l'âme, sans force concentrée "
                                  "sur une semelle."}[niv])
    occurrences: Dict[str, List[Dict]] = {}
    for L in endroits:
        occ = []
        for C in ([] if L.get("par_ame") else combs):          # (un appui par l'âme : aucune force sur une semelle)
            f = forces(L, b["niveau"], sols, C.facteurs)
            if f["soulevement"] > 0:
                alertes.setdefault(f"soulevement_{L['base']}", {"code": f"soulevement_{L['base']}", "niveau": "info",
                                                                "message": f"{L['nom']} : réaction de soulèvement ({C.nom}) — ancrage hors du module."})
            if f["haut"] > 0 or f["bas"] > 0:
                e = C.lire(L["x"])
                Ms = e["M_g"] if abs(e["M_g"]) >= abs(e["M_d"]) else e["M_d"]
                # τ de la jonction : le plus grand V_Ed de part et d'autre du lieu (À VÉRIFIER : côté sûr)
                occ.append({"C": C, "haut": f["haut"], "bas": f["bas"], "N": max(C.N_en(L["x"], False), C.N_en(L["x"], True)),
                            "M": max(abs(e["M_g"]), abs(e["M_d"])), "M_s": Ms,
                            "V": max(abs(e["V_g"]), abs(e["V_d"]))})
        occurrences[L["base"]] = occ
    charges = [L for L in endroits if occurrences[L["base"]]]
    if a is None:
        if charges and s["forme"] in ("tube_carre", "tube_rect", "tube_rond"):
            alertes["tube_charge"] = {"code": "tube_charge", "niveau": "info",
                                      "message": "Profil creux : l'âme sous une force concentrée (appuis, charges ponctuelles) n'est pas "
                                                 "vérifiée ici — EN 1993-1-8."}
        charges = []
    else:
        hyp.append(f"Force concentrée : sans raidisseur transversal voisin (a → ∞) ; {ETA1_HYP[P['eta1_ame']]} (§ 7.2).")
        hyp.append("Semelle chargée tendue : critère 6.2.1(5) de l'EN 1993-1-1 à la jonction (§ 7.2(2)).")
        hyp.append("Semelle chargée tenue à un appui ou par un maintien au droit de la force (§ 6.1(1)).")
        if any(L["genre"] == "appui" for L in endroits):
            hyp.append("Le déversement suppose des appuis en fourche : un raidisseur d'appui peut être nécessaire pour l'assurer.")

    # ---- la classe de la section SOUS LES EFFORTS DE L'OCCURRENCE (EN 1993-1-1 § 5.5) : W_pl de la variante η₁ plastique
    #      (classes 1 et 2), A_eff et W_eff du η₁ élastique en classe 4
    classes: Dict[Tuple[float, float], Tuple] = {}

    def classement(N: float, M: float) -> Tuple[Optional[int], float, float, str, str]:
        """(classe, A, W, leurs noms) sous (N kN, M kN·m signé) : A et W_el ; A_eff et W_eff en classe 4."""
        cle = (round(N, 9), round(M, 9))
        if cle not in classes:
            g = ES.verifier({"section": s, "nuance": b["nuance"], "NEd": N * 1e3, "MyEd": M * 1e6, "parametres": b["parametres"],
                             "voilement": False})["grandeurs"]
            cl, ef = g.get("classe"), g.get("efficace")
            if cl == 4 and ef is not None:
                classes[cle] = (cl, ef["A_eff"] if N > 0 else s["A"], ef["Weff_y"] or s["Wel_y"], "A_eff" if N > 0 else "A", "W_eff,y")
            else:
                classes[cle] = (cl, s["A"], s["Wel_y"], "A", "W_el,y")
        return classes[cle]

    # ---- la semelle chargée est-elle comprimée ou tendue (brief § 8.2) ? σ_x au plan moyen ; le critère 6.2.1(5) à la jonction
    def elastique(o: Dict, k: str, F: float, r: Dict) -> Optional[Dict]:
        """Le contrôle de la semelle chargée ``k`` sous la force F (kN) de l'occurrence ``o`` : {"etat", "sx", "sz", "tau",
        "crit", …} — "crit" None quand la semelle est comprimée partout (l'interaction (7.2) seule)."""
        cl, A, W_, _, _ = classement(o["N"], o["M_s"])
        I = s["Iy"] * W_ / s["Wel_y"]                    # classe 4 : I_eff tiré de W_eff (À VÉRIFIER : approché)
        nv = AM.niveaux_semelle(s, k)
        etat = AM.etat_semelle([AM.sigma_x(o["N"] * 1e3, o["M_s"] * 1e6, A, I, nv["z_moyen"])])
        out = {"etat": etat, "cote": k, "F": F, "A": A, "I": I, "nv": nv, "crit": None}
        if etat == AM.COMPRIMEE:
            return out
        # σ_x nul (un appui d'extrémité) : les deux contrôles, le plus défavorable retenu (choix déclaré du brief § 8.2, À VÉRIFIER)
        sx = AM.sigma_x(o["N"] * 1e3, o["M_s"] * 1e6, A, I, nv["z_jonction"])
        tau = o["V"] * 1e3 * nv["S"] / (I * s["tw"])      # τ à la jonction : V·S_f/(I_y·t_w), S_f de la semelle chargée (À VÉRIFIER)
        # σ_z,Ed garde s_s même quand le contact d'extrémité n'est pas établi : le brief (§ 8.2) ne met s_s = 0 que dans (6.10),
        # « les autres paramètres inchangés » (À VÉRIFIER). s_s non saisie : le critère, quadratique en σ_z, est le plus grand à
        # un BOUT de l'intervalle — s_s = 0 (σ_z le plus fort) ou s_s = h_w (quand σ_x > 2·σ_z) : le plus grand des deux
        essais = [r["ss"]] + ([r["ame"]["hw"]] if "s_s" in (r.get("manque") or []) else [])
        out.update(critere_pire(F * 1e3, essais, nv["tf"], s["tw"], sx, tau, fy, gM0), sx=sx, tau=tau)
        out["pire_hw"] = len(essais) > 1 and out["ss"] == essais[1]          # le bout h_w retenu : la fenêtre le dit
        return out

    # ---- première passe : l'âme seule tient-elle, sous chaque combinaison ?
    etudes: Dict[str, Dict] = {}
    for L in charges:
        res = resistances(L, s, fy, P, fyw)
        pire = {"eta2": 0.0, "inter": None, "F": 0.0, "cote": None, "C": None, "crit": None, "V": 0.0}
        charges_ = set()
        etats_: Dict[str, set] = {"haut": set(), "bas": set()}
        for o in occurrences[L["base"]]:
            essais = [(k, o[k]) for k in ("haut", "bas") if o[k] > 0 and k in res]
            if len(essais) == 2:
                essais.append(("b", min(o["haut"], o["bas"])))
            o["cote"], o["F"] = max(essais, key=lambda t: t[1] * 1e3 / res[t[0]]["FRd"])
            o["eta2"] = o["F"] * 1e3 / res[o["cote"]]["FRd"]
            o["b"] = dict(essais).get("b")
            o["Fmax"] = max(o["haut"], o["bas"])
            # § 7.2(2) : semelle tendue → pas d'interaction, le critère 6.2.1(5) ; σ_x nul → les deux contrôles
            ctrl = [elastique(o, k, o[k], res[k]) for k in ("haut", "bas") if o[k] > 0 and k in res]
            o["el"] = max((x for x in ctrl if x["crit"] is not None), key=lambda x: x["crit"], default=None)
            o["etats"] = {x["cote"]: x["etat"] for x in ctrl}
            for x in ctrl:
                charges_.add(x["cote"])
                etats_[x["cote"]].add(x["etat"])
            comprimee = any(x["etat"] != AM.TENDUE for x in ctrl)
            o["eta1"] = o["inter"] = o["eta1_autre"] = o["inter_autre"] = o["inter_pl"] = None
            if comprimee and (o["M"] > EPS or abs(o["N"]) > EPS):
                cl, A, W, nomA, nomW = classement(o["N"], o["M_s"])
                # DÉCISION BUREAU n° 1 (brief § 13.2), LUE DANS LA TABLE (parametres « eta1_ame ») : η₁ élastique par défaut ; la
                # variante plastique, classes 1 et 2 sous les efforts de l'occurrence (commentaire JRC, p. 98) — celle qui ne décide
                # pas est rendue pour information
                a_pl = cl in (1, 2) and bool(s.get("Wpl_y"))         # (classes 3 et 4 : pas de variante plastique)
                Wp = s["Wpl_y"] if a_pl else W
                el1 = AM.eta1(o["N"] * 1e3, o["M"] * 1e6, A, W, fy, gM0)
                pl1 = AM.eta1(o["N"] * 1e3, o["M"] * 1e6, A, Wp, fy, gM0)
                plast = P["eta1_ame"] == "plastique"
                o.update({"A": A, "W": Wp if plast else W, "nomA": nomA, "nomW": "W_pl,y" if plast and a_pl else nomW, "classe": cl})
                o["eta1"], o["eta1_autre"] = (pl1, el1) if plast else (el1, pl1)
                o["inter"], o["inter_autre"] = AM.interaction(o["eta2"], o["eta1"]), AM.interaction(o["eta2"], o["eta1_autre"])
                o["inter_pl"] = AM.interaction(o["eta2"], pl1)
                FRd_k = res[o["cote"]]["FRd"]
                o["seuil"] = (AM.seuil_local(FRd_k, el1) / 1e3, AM.seuil_local(FRd_k, pl1) / 1e3)
            if o["eta2"] > pire["eta2"]:
                pire.update({"eta2": o["eta2"], "cote": o["cote"], "C": o["C"], "FRd": res[o["cote"]]["FRd"], "Fo": o["F"]})
            if o["inter"] is not None and (pire["inter"] is None or o["inter"] > pire["inter"]):
                pire.update({"inter": o["inter"], "C_inter": o["C"], "somme": o["eta2"] + AM.INTERACTION_72[0] * o["eta1"],
                             "inter_pl": o["inter_pl"], "seuil": o["seuil"], "M_inter": o["M"]})
            if o["el"] is not None and (pire["crit"] is None or o["el"]["crit"] > pire["crit"]):
                pire.update({"crit": o["el"]["crit"], "C_crit": o["C"], "el": o["el"]})
            pire["F"] = max(pire["F"], o["Fmax"])
            pire["V"] = max(pire["V"], o["V"])
        raison = None
        if pire["eta2"] > 1.0:
            raison = (f"Sans raidisseur : F_Ed = {RE.fr(pire['Fo'], 1)} kN > F_Rd = {RE.fr(pire['FRd'] / 1e3, 1)} kN "
                      f"({pire['C'].nom}).")
        elif pire["inter"] is not None and pire["inter"] > 1.0:
            raison = (f"Sans raidisseur : η₂ + 0,8·η₁ = {RE.fr(pire['somme'], 2)} > 1,4 ({pire['C_inter'].nom}).")
        elif pire["crit"] is not None and pire["crit"] > 1.0:
            raison = (f"Sans raidisseur : critère 6.2.1(5) à la jonction = {RE.fr(pire['crit'], 2)} > 1, "
                      f"{AM.ETATS_SEMELLE[pire['el']['etat']]} ({pire['C_crit'].nom}).")
        echec = raison is not None                 # l'âme seule ne passe pas (à la borne quand une donnée manque)
        certain = el is not None and el["elancee"] and L["genre"] == "appui"
        if certain:                                # § 5.1(2) : le raidisseur d'appui est dû, quelles que soient les données
            raison = RAISON_APPUI_ELANCEE
        # brief § 8.3, point 5 : jamais « pas de raidisseur » par défaut. s_s et c non saisis : la BORNE 0 (sous F_Rd réel, prouvé
        # par balayage) — tout passe, pas de raidisseur ; sinon, décision impossible, la donnée à saisir. Le maintien de la semelle
        # chargée et, quand elle n'est pas comprimée partout, de l'autre (§ 6.1(1)) n'a pas de borne : décision impossible
        bout = sorted({m for k in charges_ for m in res[k]["rive"]}, key=["s_s", "c"].index)
        donnees = sorted({m for k in charges_ for m in res[k]["manque"]}, key=["s_s", "c"].index)
        # … sauf quand ce qui ne passe pas n'en dépend pas : le critère 6.2.1(5) ne lit ni c ni l'appui de rive — s_s saisie, le
        # raidisseur est sûr, quel que soit c ; s_s non saisie, elle seule est à saisir
        sur = pire["crit"] is not None and pire["crit"] > 1.0 and "s_s" not in donnees
        if echec and pire["eta2"] <= 1.0 and not (pire["inter"] is not None and pire["inter"] > 1.0):
            donnees, bout = [m for m in donnees if m == "s_s"], []          # le critère 6.2.1(5) seul ne passe pas
        a_saisir = bool(donnees or bout) and echec and not certain and not sur
        requis = set(charges_) | {AUTRE[k] for k in charges_ if etats_[k] - {AM.COMPRIMEE}}
        maintiens = [k for k in ("haut", "bas") if k in requis and not L["tenue"][k]]
        # seul le maintien manque : le taux des vérifications (faites comme si la semelle était tenue) est dit, avec le remède
        tenu = max(pire["eta2"], pire["inter"] or 0.0, pire["crit"] or 0.0)
        manque = (None if certain or not (a_saisir or maintiens) else
                  impossible(donnees, maintiens, bout) if a_saisir else sans_maintien(maintiens, tenu))
        # la ligne du lieu, sa fenêtre -F et le tableau de la note : la même raison, sans le taux que leur pastille donne déjà
        ligne = sans_maintien(maintiens, tenu, ligne=True) if manque and not a_saisir else None
        if manque is None and raison and L["genre"] == "charge" and not all(L["tenue"].values()):
            # le raidisseur sous une charge : 0,75·h_w suppose les DEUX semelles tenues (§ 9.4(2)) ; sinon, longueur non chiffrée
            manque = raidisseur_tenu([k for k in ("haut", "bas") if not L["tenue"][k]])
        b_ = []
        for k in ([] if certain or a_saisir else sorted(charges_)):
            b_ += [x for x in (borne(res[k]) or "").split(", ") if x and x not in b_]
        b_ = ", ".join(b_)
        # (lot C, C0c) une DONNÉE à saisir (s_s, c : la borne 0 ne passe pas) : la décision est BLOQUÉE ; seul le maintien manque
        # (rien à saisir, une disposition à prévoir) : non couverte
        etudes[L["base"]] = {"res": res, "pire": pire, "raison": raison, "raidisseur": None, "manque": manque, "ligne": ligne,
                             "borne": f"{b_} : borne prudente" if b_ else None, "bloque": a_saisir and not certain}
    if any(r_.get("manque") or r_.get("rive") for e_ in etudes.values() for r_ in e_["res"].values()):
        hyp.append(BORNE_HYPOTHESE)                      # une fois par bloc : la lecture du § 8.3, point 5, change la décision

    # ---- deux forces rapprochées sur la même semelle (§ 6.3(2)) : à vérifier aussi ensemble — ce bloc ne le fait pas, et le dit
    proches = []
    for L1, L2 in zip(charges, charges[1:]):
        r1, r2 = etudes[L1["base"]]["res"], etudes[L2["base"]]["res"]
        ecart = (L2["x"] - L1["x"]) * 1e3
        if any(ecart < (portee(r1[k]) + portee(r2[k])) / 2.0 for k in ("haut", "bas") if k in r1 and k in r2):
            proches.append(f"{L1['nom'][:1].lower() + L1['nom'][1:]} et {L2['nom'][:1].lower() + L2['nom'][1:]}")
    if proches:
        J = RE.Journal(METHODE, NORME)
        J.verif("A-GR", "Forces concentrées rapprochées", f"{AM.REF} § 6.3(2)", "", None, None, "", [], statut=RE.NON_COUVERT,
                messages=[J.alerte("forces_rapprochees", "attention", f"Forces rapprochées ({' ; '.join(proches)}) : à vérifier ensemble "
                                                                         "(§ 6.3(2)), non vérifié ici.")])
        env.verser(J.rendre(), {"combinaison": combs[0].nom})

    # ---- les raidisseurs : chacun avec l'âme disponible jusqu'au bout de la barre, ou jusqu'à mi-chemin du voisin
    raidis = [L for L in charges if etudes[L["base"]]["raison"] and not etudes[L["base"]]["manque"] and RA.gabarit(s) is not None]
    bords = sorted({0.0, ST.longueur(b)} | {L["x"] for L in raidis})          # les bords des panneaux d'âme (N_st)
    for i, L in enumerate(raidis):
        g, d = L["dispo"]
        if i > 0:
            g = min(g, (L["x"] - raidis[i - 1]["x"]) * 1e3 / 2.0)
        if i < len(raidis) - 1:
            d = min(d, (raidis[i + 1]["x"] - L["x"]) * 1e3 / 2.0)
        etude = etudes[L["base"]]
        rigide = el is not None and el["elancee"] and L["genre"] == "appui"
        # sous une charge, dans une âme élancée, le raidisseur sert aussi au cisaillement : N = F_Ed + N_st (§ 9.3.3(3) ; brief
        # § 8.4) — V_Ed à 0,5·h_w du bord du panneau le plus sollicité ; f_y,w de N_st : la f_y de calcul (À VÉRIFIER : côté sûr)
        Nst, cis = 0.0, None
        if el is not None and el["elancee"] and L["genre"] == "charge":
            lw = AM.resistance_cisaillement(el, fy, P)["lambda_w"]
            cis = {"V": cisaillement_panneaux(L["x"], bords, combs, el["hw"] / 1e3) * 1e3, "fyw": fy, "lambda_w": lw}
            Nst = RA.effort_cisaillement(cis["V"], el["hw"], el["tw"], fy, lw, P["gamma_M1"])
        etude["raidisseur"] = RA.dimensionner(s, fy, P, etude["pire"]["F"] * 1e3, (g, d), appui_rigide=rigide, nuance=b["nuance"], fy_ame=fyw,
                                              Nst=Nst)
        if etude["raidisseur"] is not None:
            etude["raidisseur"]["cisaillement"] = cis

    # ---- seconde passe : ce qui est vérifié, combinaison par combinaison ; une décision impossible : « bloqué » (une donnée à
    #      saisir) ou « non couvert » (le maintien manque), les taux à la borne (ou aux données) pour information
    for L in charges:
        etude = etudes[L["base"]]
        res, p, manque = etude["res"], etude["raidisseur"], etude["manque"]
        nc = (RE.BLOQUE if etude["bloque"] else RE.NON_COUVERT) if manque else None

        def info(F: float, FRd: float) -> Optional[float]:
            return F * 1e3 / FRd if manque else None

        for o in occurrences[L["base"]]:
            J = RE.Journal(METHODE, NORME)
            r = res[o["cote"]]
            if p is not None:
                cote0 = etude["pire"]["cote"]
                avant = AM.ecrire_resistance(J, res[cote0]["ame"], fy, P, res[cote0])
                RA.ecrire(J, L["base"], L["nom"], s, fy, P, dict(p, F=o["Fmax"] * 1e3), avant,
                          [etude["raison"]] + _mots_force(res[cote0], _saisie(L, cote0)))
            else:
                cotes = [k for k in ("haut", "bas") if o[k] > 0 and k in res]
                k = max(cotes, key=lambda k: o[k] * 1e3 / res[k]["FRd"])
                mots = ([etude["ligne"] or manque] if manque else []) + _mots_force(res[k], _saisie(L, k))
                trop_etroit = bool(etude["raison"]) and not manque       # il faut un raidisseur, et le profilé n'en reçoit pas
                if trop_etroit:
                    mots.append(J.alerte(f"sans_raidisseur_{L['base']}", "attention", f"{L['nom']} : raidisseur nécessaire, mais le profilé "
                                                                                          "est trop étroit pour en recevoir un."))
                J.verif(f"{L['base']}-F", _titre(L, k), f"{AM.REF} § 6.2", "F_Ed ≤ F_Rd", o[k], res[k]["FRd"] / 1e3, "kN",
                        AM.ecrire_resistance(J, res[k]["ame"], fy, P, res[k]), messages=mots, statut=RE.NOK if trop_etroit else nc,
                        taux=o[k] * 1e3 / res[k]["FRd"] if trop_etroit else info(o[k], res[k]["FRd"]))
                if o["b"] is not None:
                    J.verif(f"{L['base']}-B", _titre(L, "b"), f"{AM.REF} § 6.2", "F_Ed ≤ F_Rd", o["b"], res["b"]["FRd"] / 1e3, "kN",
                            AM.ecrire_resistance(J, res["b"]["ame"], fy, P, res["b"], prefixe="b_"),
                            messages=_mots_force(res["b"], _saisie(L, "b")), statut=nc, taux=info(o["b"], res["b"]["FRd"]))
                if o["eta1"] is not None:
                    cles = [J.f("eta2", "η₂", [RE.terme("F_Ed", o["F"], "kN"), RE.terme("F_Rd", r["FRd"] / 1e3, "kN")], o["eta2"], "", "a_eta2"),
                            AM.ecrire_eta1(J, o["N"] * 1e3, o["M"] * 1e6, o["A"], o["W"], fy, gM0, o["nomA"], o["nomW"])]
                    cles.append(J.f("i72", "η", [RE.terme("η₂", o["eta2"]), RE.terme("η₁", o["eta1"])], o["inter"], "", "a_72"))
                    autre = "élastique" if P["eta1_ame"] == "plastique" else "plastique"
                    mots72 = [f"η₁ {autre} (classe {o['classe']}, variante) : {RE.fr(o['eta1_autre'], 3)}, taux {RE.fr(100.0 * o['inter_autre'], 1)} %."
                              if o["eta1_autre"] != o["eta1"] else f"η₁ élastique seul (classe {o['classe']} : pas de variante plastique).",
                              f"Seuil de résistance locale à M_Ed = {RE.fr(o['M'], 2)} kN·m imposé : {RE.fr(o['seuil'][0], 1)} kN "
                              f"(η₁ plastique : {RE.fr(o['seuil'][1], 1)} kN) — ce n'est pas la capacité de la barre."]
                    J.verif(f"{L['base']}-FM", f"{L['nom']} : force concentrée et flexion", f"{AM.REF} § 7.2", "(η₂ + 0,8·η₁)/1,4 ≤ 1",
                            o["inter"], 1.0, "", cles, taux=o["inter"], messages=mots72, statut=nc)
                if o["el"] is not None:
                    x = o["el"]
                    nv = x["nv"]
                    cles = [J.f("se", "s_e", [RE.terme("s_s", x["ss"], "mm"), RE.terme("t_f", nv["tf"], "mm")], x["se"], "mm", "a_se"),
                            J.f("sigma_z", "σ_z,Ed", [RE.terme("F_Ed", x["F"], "kN"), RE.terme("s_e", x["se"], "mm"), RE.terme("t_w", s["tw"], "mm")],
                                x["sz"], "N/mm²", "a_sigma_z"),
                            J.f("sigma_x", "σ_x", [RE.terme("N_Ed", o["N"], "kN"), RE.terme("A", x["A"], "mm²"), RE.terme("M_Ed", o["M_s"], "kN·m"),
                                                   RE.terme("z", nv["z_jonction"], "mm"), RE.terme("I_y", x["I"] / 1e4, "cm⁴")], x["sx"], "N/mm²", "a_sigma_x"),
                            J.f("tau_j", "τ_Ed", [RE.terme("V_Ed", o["V"], "kN"), RE.terme("S_f", nv["S"] / 1e3, "cm³"), RE.terme("I_y", x["I"] / 1e4, "cm⁴"),
                                                  RE.terme("t_w", s["tw"], "mm")], x["tau"], "N/mm²", "a_tau_j"),
                            J.f("c615", "critère", [RE.terme("σ_x", x["sx"], "N/mm²"), RE.terme("σ_z", x["sz"], "N/mm²"), RE.terme("τ", x["tau"], "N/mm²"),
                                                    RE.terme("f_y", fy, "N/mm²"), RE.terme("γ_M0", gM0)], x["crit"], "", "a_615")]
                    J.verif(f"{L['base']}-EL", f"{L['nom']} : critère élastique à la jonction (6.2.1(5))", "EN 1993-1-1 § 6.2.1(5) ; "
                            f"{AM.REF} § 7.2(2)", "(σ_x/f)² + (σ_z/f)² − (σ_x/f)·(σ_z/f) + 3·(τ/f)² ≤ 1", x["crit"], 1.0, "", cles, taux=x["crit"],
                            messages=[AM.MOTS_ETAT[x["etat"]]] + ([MOT_PIRE_HW] if x["pire_hw"] else []), statut=nc)
            env.verser(J.rendre(), {"combinaison": o["C"].nom, "x": L["x"]})

    # ---- un appui tenu par l'âme d'une âme élancée (§ 5.1(2)) : le raidisseur d'appui est dû — jamais « sans raidisseur » ; il
    #      est dimensionné sous la plus grande réaction (hypothèse prudente, À VÉRIFIER : HYP_APPUI_AME)
    appuis_ame: Dict[str, Dict] = {}
    for L in par_ame:
        Rs = [(abs(sum(f * sol.appuis[L["appui"]]["R"] for f, sol in zip(C.facteurs, sols) if f)), C) for C in combs]
        F, C_F = max(Rs, key=lambda t: t[0])
        p = RA.dimensionner(s, fy, P, F * 1e3, L["dispo"], appui_rigide=True, nuance=b["nuance"], fy_ame=fyw) if RA.gabarit(s) else None
        J = RE.Journal(METHODE, NORME)
        if p is not None:
            RA.ecrire(J, L["base"], L["nom"], s, fy, P, p, pourquoi=[RAISON_APPUI_ELANCEE])
        else:                                    # le raidisseur est dû, et le profilé n'en reçoit pas : non conforme, dit
            J.verif(f"{L['base']}-R", f"{L['nom']} : raidisseur d'appui", f"{AM.REF} § 5.1(2)", "raidisseur d'appui", None, None, "", [],
                    statut=RE.NOK, messages=[RAISON_APPUI_ELANCEE, J.alerte(f"sans_raidisseur_{L['base']}", "attention", f"{L['nom']} : "
                                                                           "raidisseur nécessaire, mais le profilé est trop étroit pour en recevoir un.")])
        env.verser(J.rendre(), {"combinaison": C_F.nom, "x": L["x"]})
        appuis_ame[L["base"]] = {"F": F, "p": p}

    r = env.rendre()
    par_id = {v["id"]: v for v in r["verifications"]}
    rendus = []
    impossibles: Dict[str, List[str]] = {}
    for L in charges:
        etude = etudes[L["base"]]
        pire, p, manque = etude["pire"], etude["raidisseur"], etude["manque"]
        vs = [v for i, v in par_id.items() if i.startswith(L["base"] + "-")]
        gouv = max(vs, key=lambda v: (v["statut"] == RE.NOK, v["taux"] or 0.0))
        r0 = etude["res"][pire["cote"]]
        sans = {"F_Rd": r0["FRd"] / 1e3, "eta2": pire["eta2"], "interaction": pire["inter"], "interaction_plastique": pire.get("inter_pl"),
                "critere": pire["crit"], "seuil": None if pire.get("seuil") is None else
                {"M_Ed": pire["M_inter"], "elastique": pire["seuil"][0], "plastique": pire["seuil"][1]},
                # la compression locale de l'âme sous F_Ed, à la jonction (§ 3.2.3 ; brief § 15.4) : la semelle de la plus grande force
                "sigma_z": AM.sigma_z(pire["F"] * 1e3, r0["ss"], r0["ame"]["tf"], s["tw"])["sigma_z"],
                "semelle": sorted({e_ for o in occurrences[L["base"]] for e_ in o["etats"].values()}),
                "contact": r0.get("contact_etat") if r0["type"] == "c" else None}
        avec = None if p is None else RA.vue(p, pire["F"] * 1e3, p.get("Nst") or 0.0)
        decision = "impossible" if manque else "raidisseur" if p is not None else "aucun"
        if manque:
            impossibles.setdefault(manque, []).append(L["nom"])
        statut = RE.NOK if any(v["statut"] == RE.NOK for v in vs) else (RE.BLOQUE if etude["bloque"] else RE.NON_COUVERT) if manque else RE.OK
        rendus.append({"base": L["base"], "nom": L["nom"], "genre": L["genre"], "x": L["x"], "type": r0["type"], "ss": r0["ss"],
                       "F_Ed": pire["F"], "sans": sans, "avec": avec,
                       "rappel": RA.RAPPEL_TAU if p is not None else None,
                       "decision": decision, "raison": manque or etude["raison"], "manque": manque, "ligne": etude["ligne"],
                       "raidisseur": None if p is None else {"bs": p["bs"], "ts": p["ts"], "chanfrein": p["chanfrein"],
                                                             "designation": RA.designation(p)},
                       "verifications": [v["id"] for v in vs], "gouvernant": gouv["id"], "borne": etude["borne"],
                       # une décision impossible garde son taux, sur la même ligne : à la borne, ou pour information (non couvert)
                       "taux": gouv["taux"], "statut": statut})
    for L in par_ame:                                    # les appuis par l'âme d'une âme élancée : le raidisseur d'appui (§ 5.1(2))
        F, p = appuis_ame[L["base"]]["F"], appuis_ame[L["base"]]["p"]
        vs = [v for i, v in par_id.items() if i.startswith(L["base"] + "-")]
        gouv = max(vs, key=lambda v: (v["statut"] == RE.NOK, v["taux"] or 0.0))
        rendus.append({"base": L["base"], "nom": L["nom"], "genre": "appui", "x": L["x"], "type": None, "ss": None, "F_Ed": F,
                       "sans": None, "avec": None if p is None else RA.vue(p, F * 1e3), "rappel": RA.RAPPEL_TAU if p is not None else None,
                       "decision": "raidisseur" if p is not None else "impossible", "raison": RAISON_APPUI_ELANCEE, "manque": None,
                       "ligne": None, "raidisseur": None if p is None else {"bs": p["bs"], "ts": p["ts"], "chanfrein": p["chanfrein"],
                                                                           "designation": RA.designation(p)},
                       "verifications": [v["id"] for v in vs], "gouvernant": gouv["id"], "borne": None, "taux": gouv["taux"],
                       "statut": RE.NOK if any(v["statut"] == RE.NOK for v in vs) else RE.OK, "par_ame": True})
    rendus.sort(key=lambda L: L["x"])
    if any(L["decision"] == "raidisseur" for L in rendus):
        # (leurs soudures : profil_mots.LIMITES — dites une fois)
        hyp.append("Raidisseurs : f_y de chaque plat selon son épaisseur (table unique, option a de l'ANB) ; f_y,d = min(plats ; âme).")
        hyp.append(RA.RAPPEL_TAU)
    connues = {x["code"] for x in r["alertes"]}
    r["alertes"] += [x for k, x in alertes.items() if k not in connues]
    for i, (texte, noms) in enumerate(impossibles.items()):           # une ligne par manque : « Appui A, appui B : décision impossible … »
        r["alertes"].append({"code": f"decision_impossible_{i}", "niveau": "attention",
                             "message": f"{', '.join([noms[0]] + [n[:1].lower() + n[1:] for n in noms[1:]])} : {texte[:1].lower() + texte[1:]}"})
    r.update({"hypotheses": hyp, "lieux": rendus})
    return r

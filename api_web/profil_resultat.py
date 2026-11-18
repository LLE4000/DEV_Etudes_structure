# -*- coding: utf-8 -*-
"""Barres en acier (module « Choix de profilé ») — les RÉSULTATS : la colonne de droite et les fenêtres de calcul.

À droite : la tête (la barre, son profilé, son acier) ; un VERDICT par méthode (le taux le plus fort et ce qu'il
compare) et, dessous, ce que l'ÂME demande quand elle demande quelque chose (des raidisseurs, un défaut) ; le profilé
retenu (il ouvre le sélecteur) ; un bloc par méthode — chaque vérification dit ce qu'elle compare, où elle a été
trouvée, et ouvre SA fenêtre de calcul — ; le bloc de l'âme et de ses raidisseurs (EN 1993-1-5 : une ligne par appui et
par charge ponctuelle, avec sa décision) ; la comparaison des deux méthodes, sujet par sujet ; les avertissements du
moteur ; la masse et la flèche de service.

Les fenêtres de calcul : une par vérification (« h-H-SIGMA », « e-E-Mb », « a-A-Vb »), une par LIEU de l'âme
(« a-A-L00 » : toutes ses vérifications), « tout le calcul » de chaque bloc (« s1 », « s2 », « s3 » : son numéro), la
comparaison (« cmp ») et les paramètres à confirmer (« ac »). Chaque formule y est écrite par ``acier/barre/ecriture``
— la même ligne que dans la note PDF.

Ce fichier MET EN FORME ce que le moteur a calculé et nommé : il ne calcule rien.
"""
import re
from typing import Dict, List, Optional, Tuple

from acier.barre import comparaison as CP
from acier.barre import ecriture as W
from acier.barre import resultat as RE

from . import commun as C
from . import etabli_compact as E
from . import profil_mots as M

ROLE = "profil"
N = C.NBSP
# (lot C, C4 : une ligne chacune — l'ancienne NOTE_COMPARAISON faisait 230 caractères)
NOTE_COMPARAISON = "ELS : contraintes de service ≤ f_y/s ; EC3 : efforts pondérés ≤ résistances."
NOTE_A_CONFIRMER = "Source autre que le texte de l'annexe belge : à lire sur le texte avant de s'y fier."


def fenetre(methode: str, id_: str) -> str:
    """Le nom de la fenêtre de calcul d'une vérification."""
    return f"{M.PREFIXE[methode]}-{id_}"


def methodes(r: Dict) -> List[Tuple[int, str, Dict]]:
    """Les méthodes rendues, dans l'ordre de l'écran : (numéro de leur bloc, clé, résultat)."""
    return [(i + 1, k, x) for i, (k, x) in enumerate(r["methodes"].items())]


def comparable(r: Dict) -> bool:
    """Les deux méthodes sont-elles rendues, avec au moins un taux à mettre en regard (acier/barre/comparaison.py) ?"""
    return any((li[k] or {}).get("taux") is not None for li in r["comparaison"] for k in ("historique", "ec3"))


def gouvernante(x: Dict) -> Optional[Dict]:
    """La vérification qui gouverne une méthode ; sans gouvernant (rien n'est jugé), la première bloquée, sinon la
    première non couverte."""
    if x.get("gouvernant"):
        return next(v for v in x["verifications"] if v["id"] == x["gouvernant"])
    return next((v for st in (RE.BLOQUE, RE.NON_COUVERT) for v in x["verifications"] if v["statut"] == st), None)


def tete(k: str, x: Dict) -> Optional[Dict]:
    """La vérification que le VERDICT d'une méthode montre (lot C, C4) : celle qui gouverne (``gouvernante``) — sauf, pour
    la méthode ELS, le contrôle hérité du classeur (``profil_mots.HERITES``, brief § 6.1 : il ne décide jamais seul) quand
    une autre vérification a un taux jugé : la plus forte de celles-là (la flexion, le déversement, le flambement). Le
    bloc marque la même ligne ; le contrôle hérité garde la sienne, et son taux quand il dépasse (``herite_au_dela``)."""
    g = gouvernante(x)
    if k != "historique" or g is None or g["id"] not in M.HERITES:
        return g
    autres = [v for v in x["verifications"] if v["id"] not in M.HERITES and v["statut"] in RE.JUGES and v["taux"] is not None]
    return max(autres, key=lambda v: v["taux"]) if autres else g


def herite_au_dela(x: Dict, t: Optional[Dict]) -> Optional[Dict]:
    """Le contrôle hérité qu'un verdict ne montre pas, quand son taux ÉCRIT dépasse celui du verdict : il ne disparaît
    jamais derrière lui (une ligne sous le verdict). None sinon — R2 : 143,8 % et 143,7 % s'écrivent tous deux 144 %."""
    if t is None or t.get("taux") is None:
        return None
    return next((v for v in x["verifications"] if v["id"] in M.HERITES and v["id"] != t["id"] and v["taux"] is not None
                 and C.pct(v["taux"]) > C.pct(t["taux"])), None)


def non_juges(x: Dict, g: Optional[Dict]) -> str:
    """Ce que le verdict d'une méthode ne dit pas et qui n'est pas jugé, en UNE ligne (du texte) : « + 1 non couvert :
    flexion composée », « + 2 bloqués : flambement, axe faible ; flexion composée » — le sujet de chaque vérification
    bloquée ou non couverte (acier/barre/comparaison.py), sauf celle que le verdict montre ; vide sinon (brief § 9 : un
    contrôle non couvert ou bloqué ne disparaît jamais derrière le taux global)."""
    morceaux = []
    for st in (RE.BLOQUE, RE.NON_COUVERT):
        vs = [v for v in x["verifications"] if v["statut"] == st and (g is None or v["id"] != g["id"])]
        if not vs:
            continue
        sujets = list(dict.fromkeys(M.minuscule(CP.sujet_de(v["id"]) or v["titre"]) for v in vs))
        mot = M.MOT_STATUT[st].lower() if len(vs) == 1 else M.MOTS_STATUT_PLURIEL[st]
        morceaux.append(f"+ {len(vs)} {mot} : {' ; '.join(sujets)}")
    return " · ".join(morceaux)


def pire(r: Dict) -> Tuple[Optional[str], Optional[Dict]]:
    """(la clé, le résultat) du bloc qui porte le taux le plus fort d'une barre : une de ses méthodes ; ou l'âme (clé
    ``profil_mots.AME``) quand elle NE PASSE PAS et dépasse les méthodes. Une âme qui passe ne compte pas : un raidisseur
    dimensionné au plus juste est souvent à 95 % et plus — il ne dit rien du profilé. (None, None) sans aucun taux."""
    blocs = [(k, x) for k, x in r["methodes"].items() if x.get("taux_max") is not None]
    a = r.get("ame") or {}
    if a.get("statut") == RE.NOK and a.get("taux_max") is not None:
        blocs.append((M.AME, a))
    return max(blocs, key=lambda kx: kx[1]["taux_max"]) if blocs else (None, None)


def taux_barre(r: Dict) -> Optional[float]:
    """Le taux le plus fort d'une barre (``pire``) ; None quand rien n'a de taux."""
    x = pire(r)[1]
    return x["taux_max"] if x else None


def niveau_methode(x: Dict) -> str:
    """La couleur d'une méthode : celle de son taux sous son statut (``profil_mots.couleur_taux``) — rouge si elle refuse ;
    « v » (à vérifier) si une donnée manque (bloqué), si une vérification n'est pas couverte, ou si la méthode est hors de
    son domaine ; « n » si elle n'a rien eu à vérifier."""
    if x["statut"] == RE.OK and x.get("taux_max") is None:
        return "n"
    return M.couleur_taux(x["statut"], x.get("taux_max"))


def mot_niveau(niv: str) -> str:
    """Ce qu'on dit d'un niveau : « Conforme », « Conforme, attention » (95 à 100 %), « Non conforme », « À vérifier »
    (une vérification non couverte), « Sans effort » — l'écran et la note PDF le disent avec les mêmes mots."""
    return {"r": "Non conforme", "v": "À vérifier", "n": "Sans effort"}.get(niv, C.STATUTS[niv][1] if niv in C.STATUTS else "Conforme")


def mot_methode(x: Dict) -> str:
    """Ce qu'on dit d'une méthode : le mot de son niveau ; « Hors domaine » quand elle est hors de son domaine (brief
    § 9 : le calcul est montré, jamais « conforme ») ; « Bloqué » quand une donnée manque."""
    if x["statut"] in (RE.HORS_DOMAINE, RE.BLOQUE):
        return M.MOT_STATUT[x["statut"]]
    return mot_niveau(niveau_methode(x))


def niveau_taux(case: Optional[Dict]) -> str:
    """La couleur du taux d'une CASE de la comparaison (acier/barre/comparaison.py : son taux, son statut) — celle d'un
    taux sous un statut (``profil_mots.couleur_taux``) : jamais verte quand le sujet a une vérification bloquée, non
    couverte ou hors domaine. L'écran et la note la lisent."""
    return M.couleur_taux(case["statut"], case["taux"]) if case else "c"


def etat_methode(x: Dict) -> str:
    """« Conforme », « Non conforme », « À vérifier », « Hors domaine » (du texte, pour l'en-tête d'un bloc). (lot C, C4)
    Sans le taux : le verdict de la méthode le dit (mesuré sur R2 : « 184 % » s'écrivait 8 fois dans la colonne)."""
    return mot_methode(x)


# ------------------------------------------------------------------ la tête et le verdict
def dims(el: Dict) -> str:
    """« IPE 300 · S235 · 6,00 m » : ce qu'on lit d'une barre sous son nom (HTML)."""
    s = f"{C.esc(el['profil'])} · {C.esc(el['nuance'])}"
    if el["mode"] == "chargee":
        s += f" · {' + '.join(M.m(x) for x in el['travees'])}{N}m"
    return s


def bloc_ame(r: Dict) -> Optional[int]:
    """Le numéro du bloc de l'âme, juste après ceux des méthodes ; None quand il n'a rien à montrer (ni vérification, ni
    avertissement : rien n'appuie sur l'âme)."""
    a = r["ame"]
    return len(r["methodes"]) + 1 if (a["verifications"] or a["alertes"]) else None


def etat_ame(a: Dict) -> str:
    """L'état du bloc de l'âme, pour son en-tête (du texte) : « Sans raidisseur · 77 % », « 2 raidisseurs · 97 % »,
    « Non conforme · 110 % », « À vérifier », « Bloqué » (une donnée manque : le mot des méthodes, ``mot_methode``)."""
    niv = niveau_methode(a)
    k = len(M.raidis(a))
    if niv in ("r", "v") or not a["verifications"]:
        mot = mot_methode(a) if a["verifications"] else "Non vérifiée"
    elif k:
        mot = f"{k} raidisseur{'s' if k > 1 else ''}"
    else:
        mot = "Sans raidisseur" if a["lieux"] else mot_niveau(niv)
    if a.get("taux_max") is None:
        return mot
    p = C.pct(a["taux_max"])
    return f"{mot} · " + (f">{N}{C.PLAFOND_TAUX}{N}%" if p > C.PLAFOND_TAUX else f"{C.pc(p)}{N}%")


def verdict_ame(r: Dict) -> str:
    """Sous les verdicts des méthodes, ce que l'âme DEMANDE — une ligne, et seulement s'il y a quelque chose à faire :
    des raidisseurs à poser, une vérification qui ne passe pas ou qui n'est pas couverte. Elle ouvre tout le calcul du bloc."""
    a, i = r["ame"], bloc_ame(r)
    if i is None or (a["statut"] == RE.OK and not M.raidis(a)):
        return ""
    if a["statut"] == RE.OK:
        cls, texte = "warn", M.resume_ame(a)
    elif a["statut"] in (RE.BLOQUE, RE.NON_COUVERT) and any(L["decision"] == "impossible" for L in a["lieux"]):
        cls, texte = "inc", M.resume_ame(a)                 # une donnée manque : la dire (brief § 8.3, point 5)
    else:
        g = gouvernante(a)
        cls = "nok" if a["statut"] == RE.NOK else "inc"
        texte = f"{M.MOT_STATUT[a['statut']].lower()} — {g['titre'][:1].lower() + g['titre'][1:]}" if g else M.MOT_STATUT[a["statut"]].lower()
    return (f'<button type="button" class="pf-vame {cls}" data-calcul="s{i}" title="Voir le calcul">{C.icon("triangle-alert")}'
            f'<span><b>Âme</b> · {M.sub(texte)}</span></button>')     # s_s en indice, comme les lignes du bloc (M.sub échappe)


def verdicts(r: Dict) -> str:
    """Un verdict par méthode : la pastille de son taux le plus fort, la vérification qui gouverne et ce qu'elle compare ;
    il ouvre la fenêtre de calcul de cette vérification. Dessous, ce que l'âme demande (``verdict_ame``)."""
    out = ""
    ms = methodes(r)
    for _, k, x in ms:
        g = tete(k, x)
        if g is None:
            out += (f'<div class="verdict"><div class="gov"><b>{M.METHODES[k]}</b>aucun effort à vérifier</div></div>')
            continue
        # le taux du verdict : celui de la vérification qu'il montre (la plus forte, sauf le contrôle hérité de l'ELS),
        # dans la couleur de la MÉTHODE — une méthode qui refuse ne se peint jamais en vert
        t = g["taux"] if g is not gouvernante(x) else x.get("taux_max")     # (une méthode bloquée : sans taux jugé, son mot)
        if t is not None:
            pastille = C.chip(C.pct(t), M.couleur_taux(x["statut"], t), xl=True)
        else:                                    # rien n'a de taux jugé : un refus, une donnée qui manque, une vérification non couverte
            pastille = f'<span class="chip xl {M.NIVEAU_STATUT.get(x["statut"], "c")} pf-mot">{M.MOT_STATUT[x["statut"]]}</span>'
        ou = M.lieu(g)
        suite = f'<br><span class="pf-ou">{C.esc(g["combinaison"])}{" · " + C.esc(ou) if ou else ""}</span>' if g.get("combinaison") else ""
        # une vérification BLOQUÉE dit sa raison, en une ligne (son premier message) — pas le calcul pour information
        quoi = C.esc(g["messages"][0]) if g["statut"] == RE.BLOQUE and g["messages"] else M.comparaison(g)
        reste = non_juges(x, g)
        h = herite_au_dela(x, g)
        if h is not None:                       # le contrôle hérité dépasse ce que le verdict montre : dit, jamais caché
            reste = " · ".join(t_ for t_ in (reste, f"+ {M.minuscule(h['titre'])} ({M.MOT_HERITE}) : {C.pc(C.pct(h['taux']))}{N}%") if t_)
        reste = f'<br><span class="pf-ou pf-nc">{C.esc(reste)}</span>' if reste else ""
        out += (f'<div class="verdict" data-calcul="{fenetre(k, g["id"])}" title="Voir le calcul">{pastille}<div class="gov">'
                f'<b>{M.METHODES[k]} · {C.esc(g["titre"][:1].lower() + g["titre"][1:])}</b>{quoi}{suite}{reste}</div></div>')
    return f'<div class="pf-verdicts n{len(ms)}">{out}</div>{verdict_ame(r)}'


# ------------------------------------------------------------------ les blocs
def _ligne_profil(r: Dict) -> str:
    """La ligne « Profilé » : le bouton qui ouvre le sélecteur. Le plus léger de la famille qui tient se lit DANS le
    sélecteur (★) : la ligne le disait quand le sélecteur avait déjà vérifié la famille — le résultat d'une note dépendait
    alors de ce qui avait été ouvert avant, ce que le moteur partagé du site interdit (un écran rend ce que rend un moteur
    neuf ; trouvé par tests/test_moteur_partage.py, P1, le 14/11/2026). (lot C, C4) Ni la masse (le bouton du profilé, à
    gauche, la dit), ni le taux de la barre (les verdicts le disent) : une information à un endroit."""
    return (f'<div class="el vc fixe"><div class="l1"><b>Profilé</b><button class="prof" data-pk="{ROLE}" title="Choisir une disposition">'
            f'<span>{C.esc(r["profil"])}</span>{C.icon("chevron-down")}</button></div></div>')


def _ligne(k: str, v: Dict, gov: bool) -> str:
    """Une vérification : son titre, ce qu'elle compare et où, son taux — ou son état en mots quand elle n'en a pas."""
    ou = M.lieu(v)
    cmp_ = M.comparaison(v) + (f' <span class="pf-ou">· {C.esc(ou)}</span>' if ou else "")
    taux = C.pct(v["taux"]) if v["taux"] is not None else None
    return C.ligne_verif(v["titre"], cmp_, taux, M.niveau(v), calcul=fenetre(k, v["id"]), gov=gov,
                         etat_txt=M.MOT_STATUT[v["statut"]].lower() if taux is None else "")


def _efforts_bloc(el: Dict, k: str, x: Dict) -> str:
    """La ligne sous l'en-tête d'un bloc : la norme, puis les charges que la méthode prend."""
    quoi = "charges de service" if k == "historique" else "combinaisons ELU (6.10)"
    if el["mode"] == "efforts":
        quoi = "efforts de service" if k == "historique" else "efforts de calcul"
    n_c = len(x.get("combinaisons") or [])
    return f"{C.esc(x['norme'])} · {quoi}" + (f" · {n_c} combinaison{'s' if n_c > 1 else ''}" if n_c else "")


def sans_raidisseur_en_tete(a: Dict) -> bool:
    """L'en-tête du bloc de l'âme dit-il « Sans raidisseur » (``etat_ame``) ? Ses lignes ne le redisent pas alors (lot C,
    C4 : une information à un endroit)."""
    return etat_ame(a).startswith("Sans raidisseur")


def _ligne_lieu(a: Dict, L: Dict, gov: bool) -> str:
    """Un LIEU de l'âme (un appui, une charge ponctuelle) : sa décision — « sans raidisseur » ou la paire de plats —, ce
    que sa vérification la plus défavorable compare, son taux ; elle ouvre la fenêtre du lieu (toutes ses vérifications).
    « Sans raidisseur » n'est dit qu'une fois, dans l'en-tête, quand il vaut pour tout le bloc : la ligne garde sa borne."""
    v = next(x for x in a["verifications"] if x["id"] == L["gouvernant"])
    niv = M.niveau(v)
    if L["decision"] == "raidisseur":
        cmp_ = f'<b class="pf-raid">{C.esc(L["raidisseur"]["designation"])}</b> <span class="pf-ou">· {M.comparaison(v)}</span>'
    elif L["decision"] == "impossible":                 # rien n'est décidé : la donnée qui manque (brief § 8.3, point 5), et le taux
        cmp_, niv = f'<b>{M.sub(M.decision_lieu(L))}</b>', M.NIVEAU_STATUT[RE.NON_COUVERT]      # à la borne, en gris
    elif sans_raidisseur_en_tete(a):
        cmp_ = M.comparaison(v) + (f' <span class="pf-ou">· {M.sub(L["borne"])}</span>' if L.get("borne") else "")
    else:
        cmp_ = f'{M.comparaison(v)} <span class="pf-ou">· {M.sub(M.decision_lieu(L))}</span>'
    return C.ligne_verif(L["nom"], cmp_, C.pct(L["taux"]) if L["taux"] is not None else None, niv, calcul=fenetre(M.AME, L["base"]),
                         gov=gov, etat_txt=M.MOT_STATUT[L["statut"]].lower())


def _bloc_ame(el: Dict, r: Dict) -> str:
    """Le bloc de l'âme : le voilement par cisaillement (une âme élancée), puis une ligne par lieu ; ses avertissements."""
    a, i = r["ame"], bloc_ame(r)
    if i is None:
        return ""
    des_lieux = {k for L in a["lieux"] for k in L["verifications"]}
    g = a.get("gouvernant")
    rows = "".join(_ligne(M.AME, v, v["id"] == g) for v in a["verifications"] if v["id"] not in des_lieux)
    rows += "".join(_ligne_lieu(a, L, g in L["verifications"]) for L in a["lieux"])
    pied = "".join(C.message(M.sub(x["message"]), x.get("niveau") != "info") for x in a["alertes"])
    return E.bloc(i, M.TITRE_AME, M.CLASSE_BLOC.get(niveau_methode(a), "inc") if a["verifications"] else "inc", etat_ame(a),
                  _efforts_bloc(el, M.AME, a), rows, f'<div class="pf-msgs">{pied}</div>' if pied else "", theme=True)


def sujet_commun(li: Dict) -> str:
    """Le nom de la case COMMUNE d'une ligne de la comparaison (l'âme, EN 1993-1-5, vaut pour les deux méthodes) : le
    sujet seul quand la ligne n'a qu'elle (« Âme sous charges »), sinon « Cisaillement, âme (EN 1993-1-5) »."""
    seule = li["historique"] is None and li["ec3"] is None
    return li["sujet"] if seule else f"{li['sujet']}, âme (EN 1993-1-5)"


def _pastille_case(case: Optional[Dict]) -> str:
    """La pastille d'une case de la comparaison : son taux dans sa couleur (``niveau_taux``) ; sans taux, son état."""
    if case is None:
        return '<span class="chip c">—</span>'
    if case["taux"] is None:
        return f'<span class="chip {niveau_taux(case)}">{C.esc(M.MOT_STATUT[case["statut"]].lower())}</span>'
    return C.chip(C.pct(case["taux"]), niveau_taux(case))


def ecart_txt(li: Dict) -> str:
    """L'écart d'une ligne de la comparaison (le moteur le nomme : taux Eurocode 3 − taux ELS, en points de pourcentage),
    écrit au dixième — « −34,7 pts », « +12,0 pts » ; « — » quand une méthode n'a pas de taux (du texte)."""
    e = li.get("ecart")
    if e is None:
        return "—"
    return ("+" if e > 0 else M.MOINS if e < 0 else "") + f"{C.fr(abs(e), 1)}{N}pts"


def _titre_de(r: Dict, k: str, case: Optional[Dict]) -> Optional[str]:
    """Le titre de la vérification qui gouverne une case de la comparaison (``gouvernant`` : un identifiant du moteur)."""
    if not case or not case.get("gouvernant"):
        return None
    x = r["ame"] if k == "commun" else r["methodes"][k]
    return next((v["titre"] for v in x["verifications"] if v["id"] == case["gouvernant"]), None)


def gouverne_txt(r: Dict, li: Dict) -> str:
    """Ce qui gouverne une ligne de la comparaison, pour chaque méthode (du texte) : « ELS : contrainte équivalente ·
    Eurocode 3 : effort tranchant, parallèle à l'âme » — seulement le titre qui ne redit pas le sujet (« Déversement » :
    rien) ; vide sinon."""
    morceaux = []
    for k in ("historique", "ec3"):
        t = _titre_de(r, k, li[k])
        if t and t.lower() != li["sujet"].lower():
            morceaux.append(f"{M.METHODES_COURT[k]} : {M.minuscule(t)}")
    return " · ".join(morceaux)


def _comparaison(r: Dict) -> str:
    """Le tableau de la comparaison (acier/barre/comparaison.py) : un sujet par ligne — le taux de chaque méthode en
    pastille, leur écart (points), et ce qui gouverne sous le sujet quand ce n'est pas le sujet lui-même ; l'âme, qui vaut
    pour les deux, porte UNE pastille à cheval sur les deux colonnes."""
    rows = (f'<div class="pf-cmp pf-cmp-h"><span></span><span>{M.METHODES_COURT["historique"]}</span><span>{M.METHODES_COURT["ec3"]}</span>'
            f'<span>Écart</span></div>')
    for li in r["comparaison"]:
        if li["historique"] is not None or li["ec3"] is not None:
            gv = gouverne_txt(r, li)
            gv = f'<small class="pf-ou">{C.esc(gv)}</small>' if gv else ""
            rows += (f'<div class="pf-cmp"><span>{C.esc(li["sujet"])}{gv}</span>{_pastille_case(li["historique"])}{_pastille_case(li["ec3"])}'
                     f'<span class="pf-ecart">{C.esc(ecart_txt(li))}</span></div>')
        if li["commun"] is not None:
            rows += f'<div class="pf-cmp pf-cmp-c"><span>{C.esc(sujet_commun(li))}</span>{_pastille_case(li["commun"])}<span></span></div>'
    return rows


def _ligne_ecarts(x: Dict) -> str:
    """(lot C, C4 — brief § 6.7) La ligne du bloc ELS qui ouvre « Écarts avec les classeurs du bureau » : « n corrections
    des classeurs » ; rien quand aucune correction ne joue sur la barre."""
    n_ = len(x.get("corrections") or [])
    if not n_:
        return ""
    return (f'<div class="el vc pf-lecarts" data-calcul="{M.FENETRE_ECARTS}" title="Voir les {M.TITRE_ECARTS}"><div class="l1">'
            f'<b>{C.esc(M.mot_corrections(n_))}</b>{C.icon("chevron-right")}</div></div>')


def blocs(el: Dict, r: Dict) -> str:
    """Le profilé, puis un bloc par méthode, le bloc de l'âme et, quand les deux méthodes sont rendues, leur comparaison."""
    out = f'<div class="els pf-profil">{_ligne_profil(r)}</div>'
    ms = methodes(r)
    for i, k, x in ms:
        if not x["verifications"]:
            continue                                       # aucun effort : le verdict le dit, un bloc vide n'ajoute rien
        g = (tete(k, x) or {}).get("id")                   # (la ligne que le verdict montre : jamais le contrôle hérité)
        rows = "".join(_ligne(k, v, v["id"] == g) for v in x["verifications"])
        if k == "historique":
            rows += _ligne_ecarts(x)
        pied = "".join(C.message(M.sub(a["message"]), a.get("niveau") != "info") for a in x["alertes"]
                       if a.get("code") not in ALERTES_DANS_LE_CALCUL)
        niv = niveau_methode(x)
        # (le titre COURT, celui du bouton de la rubrique « Méthodes » : « Méthode historique » se lisait « MÉTHODE HIS… »
        #  devant « Conforme, attention · 97 % » ou « Non conforme · 179 % » — trouvé par le harnais des barres, 14/11/2026 ;
        #  depuis le 16/11/2026, « ELS » : le nom entier, « Méthode ELS (NBN B 51-001) », est celui du verdict et des fenêtres)
        out += E.bloc(i, M.METHODES_COURT[k], M.CLASSE_BLOC.get(niv, "inc"), etat_methode(x), _efforts_bloc(el, k, x), rows,
                      f'<div class="pf-msgs">{pied}</div>' if pied else "", theme=True)
    out += _bloc_ame(el, r)
    if comparable(r):
        # (lot C, C0f : la ligne « Le plus défavorable » est partie — les verdicts la disent ; C4 : ni l'état ni le sous-titre,
        #  qui redisaient les sujets et les deux taux — le tableau dit tout)
        out += E.bloc(len(ms) + (2 if bloc_ame(r) else 1), "Comparaison", "inc", "", "", _comparaison(r), theme=True, calcul="cmp")
    return out


def pied(r: Dict) -> str:
    """Sous les blocs : la flèche de service (une information : elle n'est pas vérifiée). (lot C, C4) La masse du profilé
    se lit dans son bouton, à gauche — plus ici."""
    s = ""
    fl = (r.get("statique") or {}).get("fleche")
    if fl and fl.get("valeur"):
        rapport = f" · L/{C.fr(fl['rapport'], 0)}" if fl.get("rapport") else ""
        s += (f'<div class="mass pf-fleche" title="Flèche de service la plus grande ({C.esc(fl["combinaison"])}, à {M.m(fl["x"])} m) : une '
              f'information, elle n\'est pas vérifiée"><span>Flèche de service</span><b>{C.fr(abs(fl["valeur"]), 1)}{N}mm{rapport}</b></div>')
    return f'<div class="pfoot">{s}<div class="legend">{C.legende()}</div></div>'


# ------------------------------------------------------------------ les fenêtres de calcul
def _fx(bloc: Dict, symboles=None) -> str:
    """Une formule : « symbole = formule = valeurs = résultat » (acier/barre/ecriture)."""
    parts, res = W.morceaux(bloc, symboles)
    h = f'<span class="s">{M.sub(parts[0])}</span>' + "".join(f'<span class="eq">=</span><span>{M.sub(p)}</span>' for p in parts[1:])
    return f'<div class="fx">{h}<span class="eq">=</span><span class="r">{C.esc(res)}</span></div>'


GABARITS_P = ("P_cr", "PE")            # les formules de la méthode ELS qui ne parlent que de l'effort normal P


def effort_normal(par: Dict[str, Dict], v: Dict) -> Optional[float]:
    """L'effort normal P de CETTE vérification, tel que le moteur le nomme dans SES formules (le terme « P » de σ_P ou de
    PE, celui de la combinaison où elle a été trouvée) ; None quand aucune ne le nomme. (lot C, C4 — reprise : les
    grandeurs de la méthode sont celles de l'occurrence GOUVERNANTE ; une flexion composée trouvée dans une autre
    combinaison y perdait P_cr alors que sa formule employait PE.)"""
    return next((t["valeur"] for cle in v["formules"] for t in par[cle].get("termes") or () if t.get("symbole") == "P"), None)


def interaction(v: Dict) -> bool:
    """La vérification compare-t-elle une somme sans unité à 1 (une interaction) ? Son taux EST alors sa dernière formule."""
    return not v["unite"] and v.get("limite") is not None and abs(v["limite"] - 1.0) < 1e-12


def formules_de(x: Dict, v: Dict, taux: bool = True) -> List[Tuple[Dict, Optional[List[str]]]]:
    """Les formules d'une vérification, dans l'ordre du moteur : [(bloc, symboles ou None)]. Son taux s'écrit avec les
    deux membres de son critère (« η = σ/σ_adm ») ; une interaction n'en a pas (sa dernière formule le donne) ; ``taux``
    faux : sans cette ligne (la note PDF, dont le verdict dit déjà les deux membres et le taux)."""
    par = {f["cle"]: f for f in x["calcul"]}
    out: List[Tuple[Dict, Optional[List[str]]]] = []
    # (lot C, C4) sans effort normal DANS CETTE VÉRIFICATION, P_cr,y et P_cr,z (et PE) de la méthode ELS ne servent à rien : cachés
    sans_P = x.get("methode") == "historique" and not effort_normal(par, v)
    for cle in v["formules"]:
        f = par[cle]
        if sans_P and f["gabarit"] in GABARITS_P:
            continue
        if f["gabarit"] == "taux":
            g, _, d = (v.get("critere") or "").partition(" ≤ ")
            if taux and d and not interaction(v):
                out.append((f, [g, d]))
        else:
            out.append((f, None))
    return out


def _groupe(x: Dict, v: Dict) -> str:
    """Les formules d'une vérification (``formules_de``), ses messages, sous son titre."""
    lignes = "".join(_fx(f, syms) for f, syms in formules_de(x, v))
    msgs = "".join(f'<div class="pf-msg">{M.sub(t)}</div>' for t in v["messages"])
    if not lignes and not msgs:
        return ""
    ou = M.lieu(v)
    sous = " · ".join(t for t in (v.get("combinaison") or "", ou) if t)
    st = f'<span class="ck-comb">{C.esc(sous)}</span>' if sous else ""
    return f'<div class="ck-g"><div class="ck-t">{C.esc(v["titre"])}{st}</div>{lignes}{msgs}</div>'


def _ligne_tableau(v: Dict, gov: bool) -> Dict:
    crit = v.get("critere") or ""
    g, _, d = crit.partition(" ≤ ")
    if v["valeur"] is None:
        valeur, limite = "—", "—"
    elif interaction(v):
        valeur, limite = C.fr(v["valeur"], 3), "1"
    elif not v["unite"]:                       # (lot C, C4) une somme sans unité à sa limite : « η₂ + 0,8·η₁ = 0,513 », « 1,4 »
        valeur, limite = f"{M.sub(g)} = {C.fr(v['valeur'], 3)}", M.n(v["limite"], 3)
    else:
        valeur = f"{M.sub(g)} = {M.r1(v['valeur'])}{N}{C.esc(v['unite'])}"
        limite = f"{M.sub(d)} = {M.r1(v['limite'])}{N}{C.esc(v['unite'])}"
    ligne = {"nom": C.esc(v["titre"]), "sous": C.esc(v["reference"]), "formule": M.sub(crit), "valeur": valeur, "limite": limite,
             "taux": C.pct(v["taux"]) if v["taux"] is not None else None, "niveau": M.niveau(v), "gov": gov}
    if v["taux"] is None:
        ligne["etat_txt"] = M.MOT_STATUT[v["statut"]]
    return ligne


def _hyps(el: Dict, r: Dict, x: Dict) -> List[str]:
    """Les hypothèses d'une méthode, en pastilles : celles que le moteur écrit."""
    return [f"<b>{C.esc(r['profil'])}</b> · {C.esc(el['nuance'])}"] + [M.sub(h) for h in x["hypotheses"]]


def _combinaisons(x: Dict) -> str:
    """Le bilan des combinaisons d'une méthode : chacune, la vérification qui y gouverne et son taux."""
    if len(x.get("combinaisons") or []) < 2:
        return ""
    titres = {v["id"]: v["titre"] for v in x["verifications"]}
    lignes = ""
    for c in x["combinaisons"]:
        niv = "r" if c["statut"] == RE.NOK else (C.niveau(c["taux"]) if c["taux"] is not None else "c")
        quoi = titres.get(c["gouvernant"], c["gouvernant"] or "—")
        lignes += (f'<div class="fx pf-comb"><span class="s">{C.esc(c["nom"])}</span><span>{C.esc(quoi[:1].lower() + quoi[1:])}</span>'
                   f'<span class="chip {niv}">{M.taux_txt(c["taux"])}</span></div>')
    return f'<div class="ck-g"><div class="ck-t">Combinaisons</div>{lignes}</div>'


def _case_txt(case: Optional[Dict]) -> str:
    """Une case de la comparaison dans le tableau de la fenêtre : son taux, ou son état en mots, ou « — » (HTML)."""
    if case is None:
        return "—"
    return M.taux_txt(case["taux"]) if case["taux"] is not None else C.esc(M.MOT_STATUT[case["statut"]].lower())


def _fenetre_comparaison(el: Dict, r: Dict, nom: str) -> str:
    """La fenêtre « cmp » (lot C, C4) : les sujets de acier/barre/comparaison.py, une ligne chacun — le taux ELS, le taux
    Eurocode 3, leur écart en points, et ce qui gouverne chaque méthode sous le sujet ; l'âme, une ligne qui vaut pour les
    deux. Sous le tableau, ce que les deux méthodes comparent, en une ligne."""
    lignes = []
    for li in r["comparaison"]:
        if li["historique"] is not None or li["ec3"] is not None:
            lignes.append({"nom": C.esc(li["sujet"]), "sous": C.esc(gouverne_txt(r, li)), "formule": "",
                           "valeur": _case_txt(li["historique"]), "limite": _case_txt(li["ec3"]), "niveau": "c", "etat_txt": ecart_txt(li)})
        if li["commun"] is not None:                 # l'âme : un seul taux, il vaut pour les deux méthodes
            c = li["commun"]
            lignes.append({"nom": C.esc(sujet_commun(li)), "sous": "vaut pour les deux méthodes", "formule": "",
                           "valeur": _case_txt(c), "limite": _case_txt(c), "niveau": "c", "etat_txt": "—"})
    detail = f'<div class="ck-g"><div class="pf-msg">{M.sub(NOTE_COMPARAISON)}</div></div>'
    return C.fenetre_calcul("cmp", f"{nom} · comparaison des deux méthodes", dims(el), [], lignes, detail,
                            colonnes=("Sujet", "", M.METHODES_COURT["historique"], M.METHODES_COURT["ec3"], "Écart"))


def _fenetre_ecarts(el: Dict, r: Dict, nom: str) -> str:
    """(lot C, C4 — brief § 6.7) « Écarts avec les classeurs du bureau » : une ligne par correction QUI JOUE sur la barre
    (``corrections`` de la méthode ELS : son texte, la vérification où elle joue). Rien d'autre : les choix du bureau
    (f_y nominale, s, C, C_d, courbes et classes) ne sont pas des corrections, et le moteur ne les y range jamais."""
    x = r["methodes"].get("historique") or {}
    cs = x.get("corrections") or []
    if not cs:
        return ""
    titres = {v["id"]: v["titre"] for v in x["verifications"]}
    lignes = [{"nom": M.sub(c["texte"][:1].upper() + c["texte"][1:]), "sous": "", "formule": "", "valeur": "", "limite": "", "niveau": "c",
               "etat_txt": ", ".join(M.minuscule(titres.get(i, i)) for i in c["verifications"]) or "—"} for c in cs]
    return C.fenetre_calcul(M.FENETRE_ECARTS, f"{nom} · {M.TITRE_ECARTS}", f"{M.METHODES['historique']} · {dims(el)}", [], lignes, "",
                            colonnes=(M.mot_corrections(len(cs)).capitalize(), "", "", "", "Où"))


def pour_information(x: Dict) -> List[str]:
    """(lot C, C4) « Tout le calcul » de l'Eurocode 3 : ce que le moteur calcule POUR INFORMATION — l'autre variante de
    chaque décision du bureau qu'il nomme, une ligne chacune (du texte) : la méthode de déversement que la table ne retient
    pas (n° 4, n° 10), A_v,y de l'autre règle (n° 3), n_pl avec l'autre γ_M quand γ_M0 ≠ γ_M1. Une ligne déjà dite par une
    vérification (le § 6.3.2.2 d'un PRS, dans le déversement) ne se redit pas."""
    g = x.get("grandeurs") or {}
    dits = " ".join(m for v in x.get("verifications") or [] for m in v["messages"])
    out = []
    info = (g.get("deversement") or {}).get("information")
    if info and info.get("M_b_Rd") is not None and "pour information : M_b,Rd" not in dits:
        dec = re.search(r"\(([^()]*DÉCISION BUREAU[^()]*)\)", info.get("source") or "")
        out.append(f"{info['methode']} pour information : M_b,Rd = {M.r1(info['M_b_Rd'] / 1e6)}{N}kN·m, taux {C.pc(C.pct(info['taux']))}{N}%"
                   + (f" ({dec.group(1)})." if dec else "."))
    var = g.get("A_v_variante")
    if var:
        out.append(f"{var['message'].rstrip('.')} = {M.n(var['A_v_y'], 1)}{N}mm², V_pl,y,Rd = {M.r1(var['V_pl_y_Rd'] / 1e3)}{N}kN, "
                   f"taux {C.pc(C.pct(var['taux']))}{N}%.")
    vg = (g.get("annexe_A") or {}).get("variante_gamma_M0")
    if vg:
        out.append(f"n_pl avec l'autre γ_M (à confirmer) : n_pl = {C.fr(vg['n_pl'], 3)} ; (6.61) {C.fr(vg['t61'], 3)} ; (6.62) {C.fr(vg['t62'], 3)}.")
    return out


# (lot C, C4) l'avertissement que « Tout le calcul » de l'Eurocode 3 dit désormais (``pour_information``) : plus au pied du bloc
ALERTES_DANS_LE_CALCUL = ("npl_gamma_M0",)


def avant_apres(L: Dict) -> str:
    """(lot C, C4 — brief § 15.4) Le tableau AVANT/APRÈS d'un lieu qui reçoit un raidisseur, en tête de sa fenêtre : la
    grandeur, sans raidisseur (l'âme seule : F_Rd, σ_z,Ed, η₂, l'interaction (7.2) ou le critère 6.2.1(5)), avec le
    raidisseur proposé (la section efficace : σ, N_b,Rd ; le contact) ; puis le rappel sur τ, en une ligne. Vide sans
    raidisseur proposé (HTML : des « span », pour la bande des hypothèses de la fenêtre)."""
    s, a = L.get("sans"), L.get("avec")
    if L["decision"] != "raidisseur" or not s or not a:
        return ""
    rien = "—"
    rangs = [("F_Rd", f"{M.r1(s['F_Rd'])}{N}kN", rien), ("σ_z,Ed", f"{M.r1(s['sigma_z'])}{N}N/mm²", rien), ("η₂", C.fr(s["eta2"], 3), rien)]
    if s.get("interaction") is not None:
        rangs.append(("(η₂ + 0,8·η₁)/1,4", C.fr(s["interaction"], 3), rien))
    if s.get("critere") is not None:
        rangs.append(("critère 6.2.1(5)", C.fr(s["critere"], 3), rien))
    rangs += [("σ, section efficace", rien, f"{M.r1(a['sigma'])}{N}N/mm²"),
              ("N_b,Rd", rien, f"{M.r1(a['N_b_Rd'])}{N}kN · {C.fr(a['taux_flambement'], 3)}")]
    fyc = f" ≤ {M.n(a['f_y_contact'], 1)}" if a.get("f_y_contact") else ""
    rangs.append(("contact", rien, f"{M.r1(a['sigma_contact'])}{fyc}{N}N/mm² · {C.fr(a['taux_contact'], 3)}"))
    cases = "".join(f'<span class="g">{M.sub(g_)}</span><span>{v1}</span><span>{v2}</span>' for g_, v1, v2 in rangs)
    rappel = f'<span class="pf-rappel">{M.sub(L["rappel"])}</span>' if L.get("rappel") else ""
    return (f'<span class="pf-avap"><span class="h">Grandeur</span><span class="h">Sans raidisseur</span>'
            f'<span class="h">{C.esc(L["raidisseur"]["designation"])}</span>{cases}</span>{rappel}')


def _fenetre_a_confirmer(r: Dict, nom: str) -> str:
    """Les paramètres de l'annexe nationale dont la source reste à confirmer : leur valeur, d'où elle vient."""
    ac = (r["methodes"].get("ec3") or {}).get("a_confirmer") or []
    if not ac:
        return ""
    lignes = [{"nom": M.sub(p["nom"]), "sous": C.esc(p["source"]), "formule": "", "valeur": C.esc(valeur_param(p["valeur"])), "limite": "—",
               "niveau": "c", "etat_txt": "à confirmer"} for p in ac]
    detail = f'<div class="ck-g"><div class="ck-t">Pourquoi « à confirmer »</div><div class="pf-msg">{C.esc(NOTE_A_CONFIRMER)}</div></div>'
    return C.fenetre_calcul("ac", f"{nom} · paramètres à confirmer", "annexe nationale belge de l'Eurocode 3", [], lignes, detail,
                            colonnes=("Paramètre", "", "Valeur prise", "", "État"))


def valeur_param(v) -> str:
    """La valeur d'un paramètre de l'annexe nationale, écrite : un nombre, une liste, un nom."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return M.n(v, 3)
    if isinstance(v, (list, tuple)):
        return " ; ".join(M.n(x, 2) for x in v)
    return str(v).replace("_", " ")


def volets(el: Dict, r: Dict, nom: str) -> str:
    """Les fenêtres de calcul : une par vérification, « tout le calcul » de chaque méthode, la comparaison, les
    paramètres à confirmer."""
    out = ""
    sous = dims(el)
    for i, k, x in methodes(r):
        hyps = _hyps(el, r, x)
        g = (tete(k, x) or {}).get("id")
        tout = ""
        for v in x["verifications"]:
            grp = _groupe(x, v)
            tout += grp
            ou = " · ".join(t for t in (v.get("combinaison") or "", M.lieu(v)) if t)
            # (lot C, C4) la fenêtre d'une vérification ne montre que SES formules et SES messages : les hypothèses de la
            # méthode (f_y, s, les courbes…) sont dites une fois, dans « Tout le calcul »
            out += C.fenetre_calcul(fenetre(k, v["id"]), f"{nom} · {C.esc(v['titre'][:1].lower() + v['titre'][1:])}",
                                    f"{M.METHODES[k]} · {sous}" + (f" · {C.esc(ou)}" if ou else ""), [], [_ligne_tableau(v, True)], grp)
        lignes = [_ligne_tableau(v, v["id"] == g) for v in x["verifications"]]
        norme = "" if x["norme"] in M.METHODES[k] else f"{C.esc(x['norme'])} · "     # (« Méthode ELS (NBN B 51-001) » la porte déjà)
        info = "".join(f'<div class="pf-msg">{M.sub(t)}</div>' for t in pour_information(x)) if k == "ec3" else ""
        info = f'<div class="ck-g"><div class="ck-t">Pour information</div>{info}</div>' if info else ""
        out += C.fenetre_calcul(f"s{i}", f"{nom} · {M.minuscule(M.METHODES[k])}", f"{norme}{sous} · tout le calcul", hyps, lignes,
                                tout + _combinaisons(x) + info)
    out += _fenetres_ame(el, r, nom)
    if comparable(r):
        out += _fenetre_comparaison(el, r, nom)
    out += _fenetre_ecarts(el, r, nom)
    out += _fenetre_a_confirmer(r, nom)
    return out


def _fenetres_ame(el: Dict, r: Dict, nom: str) -> str:
    """Les fenêtres du bloc de l'âme : une par vérification du cisaillement, une par LIEU (toutes ses vérifications —
    l'âme seule, la flexion, ou le raidisseur), et « tout le calcul » du bloc."""
    a, i = r["ame"], bloc_ame(r)
    if i is None:
        return ""
    out = ""
    sous = dims(el)
    hyps = _hyps(el, r, a)
    g = a.get("gouvernant")
    par_id = {v["id"]: v for v in a["verifications"]}
    des_lieux = {k for L in a["lieux"] for k in L["verifications"]}
    tout = ""
    for v in a["verifications"]:
        grp = _groupe(a, v)
        tout += grp
        if v["id"] in des_lieux:
            continue
        ou = " · ".join(t for t in (v.get("combinaison") or "", M.lieu(v)) if t)
        out += C.fenetre_calcul(fenetre(M.AME, v["id"]), f"{nom} · {C.esc(v['titre'][:1].lower() + v['titre'][1:])}",
                                f"{M.TITRE_AME} · {sous}" + (f" · {C.esc(ou)}" if ou else ""), [], [_ligne_tableau(v, True)], grp)
    for L in a["lieux"]:
        vs = [par_id[k] for k in L["verifications"]]
        quoi = M.sub(M.decision_lieu(L))
        # (lot C, C4) en tête de la fenêtre du lieu : le tableau avant/après quand un raidisseur est proposé (brief § 15.4) ;
        # les hypothèses du bloc restent dans « Tout le calcul »
        ap = avant_apres(L)
        out += C.fenetre_calcul(fenetre(M.AME, L["base"]), f"{nom} · {C.esc(L['nom'][:1].lower() + L['nom'][1:])}",
                                f"{M.TITRE_AME} · {sous} · x = {M.m(L['x'])} m · {quoi}", [ap] if ap else [],
                                [_ligne_tableau(v, v["id"] == L["gouvernant"]) for v in vs], "".join(_groupe(a, v) for v in vs))
    lignes = [_ligne_tableau(v, v["id"] == g) for v in a["verifications"]]
    out += C.fenetre_calcul(f"s{i}", f"{nom} · {M.TITRE_AME.lower()}", f"{C.esc(a['norme'])} · {sous} · tout le calcul", hyps, lignes,
                            tout + _combinaisons(a))
    return out

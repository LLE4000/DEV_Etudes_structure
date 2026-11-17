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
from typing import Dict, List, Optional, Tuple

from acier.barre import comparaison as CP
from acier.barre import ecriture as W
from acier.barre import resultat as RE

from . import commun as C
from . import etabli_compact as E
from . import profil_mots as M

ROLE = "profil"
N = C.NBSP
NOTE_COMPARAISON = (f"La {M.minuscule(M.METHODES['historique'])} compare des contraintes de service à σ_adm = f_y / s ; l'Eurocode 3 "
                    "compare des efforts pondérés à des résistances. Les deux taux se lisent côte à côte, ils ne se déduisent pas l'un de "
                    "l'autre.")
NOTE_A_CONFIRMER = ("Ces valeurs changent un résultat et leur source n'est pas le texte de l'annexe nationale belge : elles sont à lire "
                    "sur le texte avant de s'y fier.")


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
    """« Conforme · 92 % », « Non conforme · 179 % », « À vérifier », « Hors domaine · 53 % » (du texte, pour l'en-tête
    d'un bloc)."""
    niv = niveau_methode(x)
    mot = mot_methode(x)
    if x.get("taux_max") is None:
        return mot
    p = C.pct(x["taux_max"])
    return f"{mot} · " + (f">{N}{C.PLAFOND_TAUX}{N}%" if p > C.PLAFOND_TAUX else f"{C.pc(p)}{N}%")


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
        g = gouvernante(x)
        if g is None:
            out += (f'<div class="verdict"><div class="gov"><b>{M.METHODES[k]}</b>aucun effort à vérifier</div></div>')
            continue
        if x.get("taux_max") is not None:
            pastille = C.chip(C.pct(x["taux_max"]), M.couleur_taux(x["statut"], x["taux_max"]), xl=True)
        else:                                    # rien n'a de taux jugé : un refus, une donnée qui manque, une vérification non couverte
            pastille = f'<span class="chip xl {M.NIVEAU_STATUT.get(x["statut"], "c")} pf-mot">{M.MOT_STATUT[x["statut"]]}</span>'
        ou = M.lieu(g)
        suite = f'<br><span class="pf-ou">{C.esc(g["combinaison"])}{" · " + C.esc(ou) if ou else ""}</span>' if g.get("combinaison") else ""
        # une vérification BLOQUÉE dit sa raison, en une ligne (son premier message) — pas le calcul pour information
        quoi = C.esc(g["messages"][0]) if g["statut"] == RE.BLOQUE and g["messages"] else M.comparaison(g)
        reste = non_juges(x, g)
        reste = f'<br><span class="pf-ou pf-nc">{C.esc(reste)}</span>' if reste else ""
        out += (f'<div class="verdict" data-calcul="{fenetre(k, g["id"])}" title="Voir le calcul">{pastille}<div class="gov">'
                f'<b>{M.METHODES[k]} · {C.esc(g["titre"][:1].lower() + g["titre"][1:])}</b>{quoi}{suite}{reste}</div></div>')
    return f'<div class="pf-verdicts n{len(ms)}">{out}</div>{verdict_ame(r)}'


# ------------------------------------------------------------------ les blocs
def _ligne_profil(r: Dict) -> str:
    """La ligne « Profilé » : le bouton qui ouvre le sélecteur, et sa masse. Le plus léger de la famille qui tient se
    lit DANS le sélecteur (★) : la ligne le disait quand le sélecteur avait déjà vérifié la famille — le résultat d'une
    note dépendait alors de ce qui avait été ouvert avant, ce que le moteur partagé du site interdit (un écran rend ce
    que rend un moteur neuf ; trouvé par tests/test_moteur_partage.py, P1, le 14/11/2026)."""
    quoi = f"{M.n(r['masse'], 1)}{N}kg/m" if r.get("masse") else ""
    t = taux_barre(r)
    niv = M.couleur_taux(r["statut"], t)          # (sans taux : « c », ou « r » pour une barre qui ne passe pas)
    return C.ligne_verif("Profilé", quoi, C.pct(t) if t is not None else None, niv, pk=ROLE, piece=C.esc(r["profil"]), etat_txt="—")


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


def _ligne_lieu(a: Dict, L: Dict, gov: bool) -> str:
    """Un LIEU de l'âme (un appui, une charge ponctuelle) : sa décision — « sans raidisseur » ou la paire de plats —, ce
    que sa vérification la plus défavorable compare, son taux ; elle ouvre la fenêtre du lieu (toutes ses vérifications)."""
    v = next(x for x in a["verifications"] if x["id"] == L["gouvernant"])
    niv = M.niveau(v)
    if L["decision"] == "raidisseur":
        cmp_ = f'<b class="pf-raid">{C.esc(L["raidisseur"]["designation"])}</b> <span class="pf-ou">· {M.comparaison(v)}</span>'
    elif L["decision"] == "impossible":                 # rien n'est décidé : la donnée qui manque (brief § 8.3, point 5), et le taux
        cmp_, niv = f'<b>{M.sub(M.decision_lieu(L))}</b>', M.NIVEAU_STATUT[RE.NON_COUVERT]      # à la borne, en gris
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


def _comparaison(r: Dict) -> str:
    """Le tableau de la comparaison (acier/barre/comparaison.py) : un sujet par ligne, la case de chaque méthode en
    pastille ; l'âme, qui vaut pour les deux, porte UNE pastille à cheval sur les deux colonnes."""
    rows = f'<div class="pf-cmp pf-cmp-h"><span></span><span>{M.METHODES_COURT["historique"]}</span><span>{M.METHODES_COURT["ec3"]}</span></div>'
    for li in r["comparaison"]:
        if li["historique"] is not None or li["ec3"] is not None:
            rows += f'<div class="pf-cmp"><span>{C.esc(li["sujet"])}</span>{_pastille_case(li["historique"])}{_pastille_case(li["ec3"])}</div>'
        if li["commun"] is not None:
            rows += f'<div class="pf-cmp pf-cmp-c"><span>{C.esc(sujet_commun(li))}</span>{_pastille_case(li["commun"])}</div>'
    return rows


def blocs(el: Dict, r: Dict) -> str:
    """Le profilé, puis un bloc par méthode, le bloc de l'âme et, quand les deux méthodes sont rendues, leur comparaison."""
    out = f'<div class="els pf-profil">{_ligne_profil(r)}</div>'
    ms = methodes(r)
    for i, k, x in ms:
        if not x["verifications"]:
            continue                                       # aucun effort : le verdict le dit, un bloc vide n'ajoute rien
        g = x.get("gouvernant")
        rows = "".join(_ligne(k, v, v["id"] == g) for v in x["verifications"])
        pied = "".join(C.message(M.sub(a["message"]), a.get("niveau") != "info") for a in x["alertes"])
        niv = niveau_methode(x)
        # (le titre COURT, celui du bouton de la rubrique « Méthodes » : « Méthode historique » se lisait « MÉTHODE HIS… »
        #  devant « Conforme, attention · 97 % » ou « Non conforme · 179 % » — trouvé par le harnais des barres, 14/11/2026 ;
        #  depuis le 16/11/2026, « ELS » : le nom entier, « Méthode ELS (NBN B 51-001) », est celui du verdict et des fenêtres)
        out += E.bloc(i, M.METHODES_COURT[k], M.CLASSE_BLOC.get(niv, "inc"), etat_methode(x), _efforts_bloc(el, k, x), rows,
                      f'<div class="pf-msgs">{pied}</div>' if pied else "", theme=True)
    out += _bloc_ame(el, r)
    if comparable(r):
        # (lot C, C0f : la ligne « Le plus défavorable » est partie — les verdicts la disent ; l'en-tête compte les sujets)
        n_s = len(r["comparaison"])
        out += E.bloc(len(ms) + (2 if bloc_ame(r) else 1), "Comparaison", "inc", f"{n_s} sujet{'s' if n_s > 1 else ''}",
                      "le taux le plus fort de chaque méthode, sujet par sujet", _comparaison(r), theme=True, calcul="cmp")
    return out


def pied(r: Dict) -> str:
    """Sous les blocs : la masse du profilé, la flèche de service (une information : elle n'est pas vérifiée)."""
    s = ""
    if r.get("masse"):
        s += f'<div class="mass"><span>Masse du profilé</span><b>{M.n(r["masse"], 1)}{N}kg/m</b></div>'
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


def interaction(v: Dict) -> bool:
    """La vérification compare-t-elle une somme sans unité à 1 (une interaction) ? Son taux EST alors sa dernière formule."""
    return not v["unite"] and v.get("limite") is not None and abs(v["limite"] - 1.0) < 1e-12


def formules_de(x: Dict, v: Dict, taux: bool = True) -> List[Tuple[Dict, Optional[List[str]]]]:
    """Les formules d'une vérification, dans l'ordre du moteur : [(bloc, symboles ou None)]. Son taux s'écrit avec les
    deux membres de son critère (« η = σ/σ_adm ») ; une interaction n'en a pas (sa dernière formule le donne) ; ``taux``
    faux : sans cette ligne (la note PDF, dont le verdict dit déjà les deux membres et le taux)."""
    par = {f["cle"]: f for f in x["calcul"]}
    out: List[Tuple[Dict, Optional[List[str]]]] = []
    for cle in v["formules"]:
        f = par[cle]
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


def _fenetre_comparaison(el: Dict, r: Dict, nom: str) -> str:
    lignes = []
    for li in r["comparaison"]:
        if li["historique"] is not None or li["ec3"] is not None:
            cases = [c for c in (li["historique"], li["ec3"]) if c is not None and c["taux"] is not None]
            pire = max(cases, key=lambda c: c["taux"], default=None)
            ligne = {"nom": C.esc(li["sujet"]), "sous": "", "formule": "",
                     "valeur": M.taux_txt((li["historique"] or {}).get("taux")), "limite": M.taux_txt((li["ec3"] or {}).get("taux")),
                     "taux": C.pct(pire["taux"]) if pire else None, "niveau": niveau_taux(pire), "gov": False}
            if pire is None:
                ligne["etat_txt"] = "—"
            lignes.append(ligne)
        if li["commun"] is not None:                 # l'âme : un seul taux, il vaut pour les deux méthodes
            c = li["commun"]
            ligne = {"nom": C.esc(sujet_commun(li)), "sous": "vaut pour les deux méthodes", "formule": "", "valeur": "—", "limite": "—",
                     "taux": C.pct(c["taux"]) if c["taux"] is not None else None, "niveau": niveau_taux(c), "gov": False}
            if c["taux"] is None:
                ligne["etat_txt"] = M.MOT_STATUT[c["statut"]].lower()
            lignes.append(ligne)
    detail = f'<div class="ck-g"><div class="ck-t">Ce que les deux méthodes comparent</div><div class="pf-msg">{C.esc(NOTE_COMPARAISON)}</div></div>'
    return C.fenetre_calcul("cmp", f"{nom} · comparaison des deux méthodes", dims(el), [], lignes, detail,
                            colonnes=("Sujet", "", M.METHODES_COURT["historique"], M.METHODES_COURT["ec3"], "Le plus fort"))


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
        g = x.get("gouvernant")
        tout = ""
        for v in x["verifications"]:
            grp = _groupe(x, v)
            tout += grp
            ou = " · ".join(t for t in (v.get("combinaison") or "", M.lieu(v)) if t)
            out += C.fenetre_calcul(fenetre(k, v["id"]), f"{nom} · {C.esc(v['titre'][:1].lower() + v['titre'][1:])}",
                                    f"{M.METHODES[k]} · {sous}" + (f" · {C.esc(ou)}" if ou else ""), hyps, [_ligne_tableau(v, True)], grp)
        lignes = [_ligne_tableau(v, v["id"] == g) for v in x["verifications"]]
        norme = "" if x["norme"] in M.METHODES[k] else f"{C.esc(x['norme'])} · "     # (« Méthode ELS (NBN B 51-001) » la porte déjà)
        out += C.fenetre_calcul(f"s{i}", f"{nom} · {M.minuscule(M.METHODES[k])}", f"{norme}{sous} · tout le calcul", hyps, lignes,
                                tout + _combinaisons(x))
    out += _fenetres_ame(el, r, nom)
    if comparable(r):
        out += _fenetre_comparaison(el, r, nom)
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
                                f"{M.TITRE_AME} · {sous}" + (f" · {C.esc(ou)}" if ou else ""), hyps, [_ligne_tableau(v, True)], grp)
    for L in a["lieux"]:
        vs = [par_id[k] for k in L["verifications"]]
        quoi = M.sub(M.decision_lieu(L))
        out += C.fenetre_calcul(fenetre(M.AME, L["base"]), f"{nom} · {C.esc(L['nom'][:1].lower() + L['nom'][1:])}",
                                f"{M.TITRE_AME} · {sous} · x = {M.m(L['x'])} m · {quoi}", hyps,
                                [_ligne_tableau(v, v["id"] == L["gouvernant"]) for v in vs], "".join(_groupe(a, v) for v in vs))
    lignes = [_ligne_tableau(v, v["id"] == g) for v in a["verifications"]]
    out += C.fenetre_calcul(f"s{i}", f"{nom} · {M.TITRE_AME.lower()}", f"{C.esc(a['norme'])} · {sous} · tout le calcul", hyps, lignes,
                            tout + _combinaisons(a))
    return out

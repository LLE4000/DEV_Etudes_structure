"""Assemblages : CE QUI NE VA PAS, et comment le corriger — le bloc « À corriger » du résultat (26/11/2026).

Le propriétaire, sur une capture de la platine d'about (« Impossible — platine trop étroite (pince e2) », « η 131 % ») :
« Ça me dit que c'est pas bon : que ça me propose une solution bonne. Les pinces, par exemple, je suis pas bon : que ça
me propose un truc qui serait correct. […] que ce soit plus simple à modifier quand un truc ne va pas, à trouver une
solution. »

Quand le moteur ne déclare PAS l'assemblage vérifié, le résultat s'ouvre donc sur ce bloc, avant les vérifications :

1. chaque DÉFAUT DE GÉOMÉTRIE que le moteur nomme (une alerte bloquante, une pince ou un entraxe hors du Tableau 3.3),
   avec ce qu'il en dit, et les RETOUCHES qui le lèvent — « b<sub>p</sub> 150 → 155 mm » : un bouton, un geste. Chaque
   retouche a été calculée par le moteur avant d'être proposée (``acier/assemblages/corrections.py``) : c'est une valeur
   avec laquelle il ne signale plus ce défaut, ni aucun nouveau ;
2. la SOLUTION VÉRIFIÉE du prédimensionnement (``acier/assemblages/propositions.py`` : boulons, pièce d'attache et gorge
   redimensionnés, chaque candidat jugé par le calcul complet) — celle qui garde le diamètre choisi quand il en existe
   une, et la conseillée ; « Appliquer » la recopie, « Autres solutions » ouvre la liste ;
3. sans aucune solution : la vérification qui bloque, nommée, et « Profilés qui conviennent » — la liste des profilés
   plus hauts de la famille, chacun avec SA solution vérifiée (cherchée au clic : une recherche par profilé).

Une retouche ou une solution ne change jamais un effort, un acier, un niveau ni le grugeage ; seul « Profilés qui
conviennent » change un profilé, et il le dit. Rien n'est calculé ici (R3) : ce fichier demande au moteur, CHOISIT parmi
ce qu'il a rendu et COMPOSE les phrases. Les six attaches poutre–poutre et poutre–poteau ; le pied de poteau n'a pas de
prédimensionnement (ses alertes restent dans ses messages).
"""
import json
from typing import Dict, List, Optional, Tuple

from acier.assemblages import corrections as CO
from acier.bibliotheques import DB, find

from . import assemblages as A
from . import commun as C

# les saisies qu'une RETOUCHE peut changer : celles qu'une proposition du prédimensionnement change (la pièce d'attache,
# les boulons, la gorge) — jamais un effort, un profilé, un acier, un niveau (z_C, Δz), le jeu ni le grugeage
RETOUCHABLES = frozenset({
    "n1_u", "n2_u", "p1_u", "p2_u", "e1_u", "e2b_u", "p3_u", "hp_u", "bp_u", "tp_u", "a_w",
    "n1S_u", "n2S_u", "p1S_u", "p2_S", "e1S_u", "n1P_u", "n2P_u", "p1P_u", "p2_P", "e1P_u", "gA_u", "LC_u", "a_S", "a_P",
})
# la grille d'une retouche : une épaisseur suit la série des tôles, une gorge avance au millimètre ; une cote, de 5 mm
# en 5 mm, un nombre de rangées, de 1 en 1 (les défauts de ``corrections``)
GRILLES = {"tp_u": tuple(DB["epais"]), "a_w": 1.0, "a_S": 1.0, "a_P": 1.0}
# les profilés essayés par « Profilés qui conviennent », au plus (une recherche complète par profilé) ; la liste
# s'arrête dès que PROFILS_TROUVES profilés conviennent
PROFILS_ESSAYES = 8
PROFILS_TROUVES = 3
MEME_HAUTEUR = 0.9       # un profilé d'une AUTRE famille est essayé dès 90 % de la hauteur du profilé en place
TITRE = "À corriger"
BLOC = 2                 # le rang du bloc pour le repli (0 : Vérifications, 1 : Prédimensionnement — js/compact.js)
TITRE_PROFILS = "Profilés qui conviennent"
SOUS_TITRE_PROFILS = "un clic retient le profilé ET sa solution vérifiée · efforts, aciers et niveaux gardés"
APPLIQUER = "Appliquer"
AUTRES = "Autres solutions"
# la dernière recherche de corrections (UNE seule) : le résultat est refait à chaque saisie, pas la recherche
_FAITES: Dict = {}
# … et les recherches par profilé de la dernière liste « Profilés qui conviennent »
_PAR_PROFIL: Dict = {}


def _cle_etat(cfg, u: Dict) -> Tuple:
    return cfg.cle, json.dumps(u, sort_keys=True, default=str, ensure_ascii=False)


def corrections(cfg, u: Dict, R) -> List:
    """Les corrections VÉRIFIÉES des défauts de géométrie de l'assemblage (``corrections.corriger``) ; gardées tant que
    les saisies ne changent pas."""
    cle = _cle_etat(cfg, u)
    if _FAITES.get("cle") != cle:
        base = A.a_calculer(cfg, u)
        permises = {k for k in RETOUCHABLES if k in cfg.E.CHAMPS and cfg.E.champ_visible(u, k)}
        _FAITES.clear()
        _FAITES.update(cle=cle, X=CO.corriger(lambda v: A.moteur(cfg, v), base, R, permises, A.COMPTES | set(cfg.entiers), GRILLES))
    return _FAITES["X"]


def _symbole(cfg, k: str) -> str:
    """Le nom court d'une saisie sur un bouton (HTML) : son symbole — celui que le dessin cote, « p<sub>1</sub> » —,
    sinon son libellé."""
    sym, lab = cfg.libelles.get(k, ("", cfg.E.CHAMPS[k]["l"]))
    return sym or lab


def _valeur(cfg, k: str, x: float) -> str:
    return C.nb(int(round(x))) if (k in A.COMPTES or k in cfg.entiers) else C.nb(x)


def bouton_retouche(cfg, r) -> str:
    """Le bouton d'UNE retouche : « b<sub>p</sub> 150 → 155 mm ». Un geste : la valeur part au moteur comme si on
    l'avait tapée dans son champ (``modifier``, le chemin du champ). ✓ quand tout l'assemblage est alors vérifié."""
    un = A._unite(cfg.E.CHAMPS[r.cle])
    de, vers = _valeur(cfg, r.cle, r.de), _valeur(cfg, r.cle, r.vers)
    unite = f" {un}" if un else ""
    titre = f"{A._nom_champ(cfg, r.cle)} : {de} → {vers}{unite}" + (
        " — l'assemblage est alors vérifié" if r.verifie
        else (" — l'assemblage est alors à vérifier (" + r.a_verifier + ")" if r.get("a_verifier") else ""))
    bon = f'<span class="ok" aria-hidden="true">{C.icon("check")}</span>' if r.verifie else ""
    return (f'<button type="button" class="asm-ret{" bon" if r.verifie else ""}" data-agir data-chemin="v/{C.esc(r.cle)}" '
            f'data-valeur="{C.esc(vers)}" title="{C.esc(titre)}"><span class="s">{_symbole(cfg, r.cle)}</span> {de} → <b>{vers}</b>{unite}{bon}</button>')


def dire_defaut(cfg, R, d) -> Tuple[str, str]:
    """(titre, détail) d'un défaut, en texte : ceux de l'alerte du moteur (``notation.ligne_alerte``) ; une pince non
    conforme sans alerte : son nom, sa valeur et sa limite, sa règle."""
    if d.alerte is not None:
        return cfg.N.ligne_alerte(d.alerte, R)
    p = d.pince
    cmp_ = f"{C.fr(p.val, 1)} mm < {C.fr(p.min, 1)} mm" if p.min is not None and p.val < p.min else f"{C.fr(p.val, 1)} mm > {C.fr(p.max, 1)} mm"
    return p.lab, f"{cmp_} ({p.rule})" if getattr(p, "rule", "") else cmp_


def _defaut(cfg, R, x) -> str:
    titre, det = dire_defaut(cfg, R, x.defaut)
    boutons = "".join(bouton_retouche(cfg, r) for r in x.retouches)
    return (f'<div class="asm-def" data-defaut="{C.esc(x.defaut.id)}"><p><b>{C.esc(titre)}</b>' + (f' <span>{C.esc(det)}</span>' if det else "")
            + '</p>' + (f'<div class="asm-rets">{boutons}</div>' if boutons else "") + '</div>')


def solutions_a_montrer(cfg, u: Dict, X) -> List:
    """Les solutions que le bloc montre, deux au plus : la CONSEILLÉE (la première de la recherche : au taux cible, sans
    réserve, le moins de boulons) et celle qui GARDE le diamètre de boulon choisi quand il en existe une autre — avant
    la conseillée si elle tient elle aussi le taux cible sans réserve (on a choisi ses boulons : la disposition correcte
    avec eux d'abord), après sinon."""
    if not X.solutions:
        return []
    conseillee = X.solutions[0]
    meme = next((s for s in X.solutions if s.boulon == u.get("boulon_u")), None)
    if meme is None or meme is conseillee:
        return [conseillee]
    return [meme, conseillee] if (meme.cible and not meme.reserves) else [conseillee, meme]


def _solution(cfg, s, conseillee: bool, gardee: bool) -> str:
    nom = cfg.predim["nom"](s.u)
    etoile = f'<span class="leger" title="la solution conseillée">{C.icon("star")}</span>' if conseillee else ""
    mot = f"avec vos {C.esc(s.boulon)}" if gardee else "conseillée"
    return (f'<div class="asm-sol"><div class="t"><span class="m">{mot}</span>{etoile}<b>{C.esc(nom)}</b>'
            f'<span class="pct {A.niveau_R(s.R)}">{C.pc(A.taux(A.eta_R(s.R)))}{C.NBSP}%</span></div>'
            f'<div class="d">{A._detail_proposition(cfg, s)}</div>'
            f'<button type="button" class="btn sm primary" data-agir data-chemin="solution" data-valeur="{C.esc(nom)}">{APPLIQUER}</button></div>')


def piece_en_cause(cfg, X) -> Optional[str]:
    """Sans solution : le PROFILÉ qu'il faut changer (« prof_S » la pièce portée, « prof_P » la porteuse), d'après la
    vérification qui bloque dans le meilleur candidat ; None quand c'est la géométrie de base qui est refusée (aucun
    profilé plus haut n'y change rien : c'est elle qu'il faut corriger)."""
    if X.blocage is None and X.get("geometrie") is not None:
        return None
    b = X.blocage
    if b is None or b.gov is None:
        return "prof_S"                           # rien ne tient dans la hauteur disponible : une poutre plus haute
    porteuse = A.pour(cfg, b.R).S.ELEMENTS[3][0]
    return "prof_P" if b.gov.grp == porteuse else "prof_S"


def _candidats(cle: str, u: Dict, familles=None) -> List:
    """Les profilés à essayer à la place de ``u[cle]``, du plus petit au plus grand : ceux de SA famille plus HAUTS que
    lui ; et, pour une autre famille demandée (les puces du sélecteur), ceux d'au moins sa hauteur à ``MEME_HAUTEUR``
    près (un HEB 300 à la place d'un HEA 320 : la même classe de hauteur, une âme plus épaisse)."""
    cur = u.get(cle)
    try:
        h0 = find(DB["profils"], cur)["h"]
    except (ValueError, KeyError, TypeError):
        return []
    sienne = A._fam(cur)
    fams = [f for f in (familles or []) if f in A.FAMILLES_PROFILS] or [sienne]
    return sorted((p for p in DB["profils"] if A._fam(p["n"]) in fams and p["n"] != cur
                   and p["h"] > (h0 + 1e-9 if A._fam(p["n"]) == sienne else MEME_HAUTEUR * h0)), key=lambda p: (p["h"], p["b"]))


def _recherche_profil(cfg, u: Dict, cle: str, nom: str):
    """La recherche des solutions avec le profilé ``nom`` à la place de ``u[cle]`` ; gardée pour le clic qui la recopie."""
    k = (_cle_etat(cfg, u), cle, nom)
    if k not in _PAR_PROFIL:
        if len(_PAR_PROFIL) > 40:
            _PAR_PROFIL.clear()
        v = dict(u, **{cle: A.valeur_widget(cfg, cle, nom)})
        try:
            _PAR_PROFIL[k] = A.chercher_solutions(cfg, v)
        except Exception:  # noqa: BLE001 — un profilé que le moteur ne calcule pas n'est pas un candidat
            _PAR_PROFIL[k] = None
    return _PAR_PROFIL[k]


def _pourquoi(cfg, X) -> str:
    """Pourquoi un profilé essayé n'a pas de solution (HTML, une phrase) : la vérification qui reste au-delà de 100 %, ou
    l'alerte de géométrie que toutes les dispositions gardent."""
    if X is None:
        return "le moteur ne le calcule pas."
    b = X.blocage
    if b is not None and b.gov is not None:
        c = A.pour(cfg, b.R)
        return (f"{C.esc(b.gov.grp)} · {C.esc(A._minuscule(A._court(c, b.gov)))} reste à "
                f"{C.pc(A.taux(min(b.R.eta_max, 9.99)))}{C.NBSP}%.")
    if X.get("geometrie") is not None:
        Rg = X.geometrie
        a = next((x for x in Rg.alerts if x.block), None)
        if a is not None:
            return C.esc(A.pour(cfg, Rg).N.ligne_alerte(a, Rg)[0]) + "."
    return "aucune disposition de boulons ne tient dans la hauteur disponible."


def sel_profils(cfg, u: Dict, familles=None) -> str:
    """Le sélecteur « Profilés qui conviennent » : les profilés plus hauts de la famille, du plus petit au plus grand,
    chacun avec SA meilleure solution vérifiée et son taux (★ le plus petit qui convient) — ou, en rouge, ce qui y
    bloque encore. La liste s'arrête après ``PROFILS_TROUVES`` profilés qui conviennent."""
    X = A.propositions(cfg, u)
    cle = piece_en_cause(cfg, X) or "prof_S"
    # (le nom de la pièce : celui du sélecteur de son profilé — « Poteau · profilé » pour une attache sur poteau)
    nom_piece = getattr(cfg, "titres_profils", {}).get(cle) or ("Poutre secondaire · profilé" if cle == "prof_S" else "Poutre principale · profilé")
    # SEULS les profilés qui conviennent sont listés (sept lignes rouges « 999 % » pour une verte n'apprenaient rien) ;
    # ceux qu'on a essayés sans succès sont comptés, et le premier dit pourquoi (souvent : une poutre plus haute heurte
    # la semelle de la porteuse — une géométrie que la recherche ne touche pas)
    rows, refuses = [], []
    essayes = _candidats(cle, u, familles)[:PROFILS_ESSAYES]
    for p in essayes:
        Xp = _recherche_profil(cfg, u, cle, p["n"])
        if Xp is not None and Xp.solutions:
            s = Xp.solutions[0]
            prop = f"{p['n']} · h {C.nb(p['h'])} · b {C.nb(p['b'])} · t<sub>w</sub> {C.nb(p['tw'])} · t<sub>f</sub> {C.nb(p['tf'])} mm"
            rows.append({"nom": p["n"], "col": C.fr(p["h"], 0), "taux": C.pct(min(A.eta_R(s.R), 9.99)), "niveau": A.niveau_R(s.R),
                         "cur": False, "opt": not rows, "prop": f"{prop} · solution : <b>{C.esc(cfg.predim['nom'](s.u))}</b>"})
            if len(rows) >= PROFILS_TROUVES:
                break
        else:
            refuses.append((p["n"], Xp))
    cur = u.get(cle)
    fams = [f for f in (familles or []) if f in A.FAMILLES_PROFILS] or [A._fam(cur)]
    filtres = "".join(f'<button type="button" class="fchip{" on" if f in fams else ""}" data-famille="{f}">{f}</button>' for f in A.FAMILLES_PROFILS)
    tete = ""
    if refuses:
        nom, Xr = refuses[0]
        tete = (f'<div class="asm-aucune{" info" if rows else ""}">' + ("Aucun des profilés essayés ne convient" if not rows else
                f"{len(refuses)} profilé{'s' if len(refuses) > 1 else ''} essayé{'s' if len(refuses) > 1 else ''} sans solution")
                + f" ({C.esc(', '.join(n for n, _x in refuses[:4]))}{'…' if len(refuses) > 4 else ''}) — {C.esc(nom)} : {_pourquoi(cfg, Xr)}"
                + ("" if rows else " Essayer une autre famille, réduire le grugeage ou revoir les efforts.") + "</div>")
    elif not essayes:
        tete = f'<div class="asm-aucune">{C.esc(str(cur))} est le plus haut de sa famille : essayer une autre famille.</div>'
    return C.selecteur("profil_solution", f"{TITRE_PROFILS} · {nom_piece.split(' · ')[0].lower()}", SOUS_TITRE_PROFILS, filtres + tete, rows, 0, "h mm",
                       etoile="le plus petit qui convient", vide="")


def choisir_profil(cfg, u: Dict, nom: str) -> None:
    """Le profilé retenu dans « Profilés qui conviennent » : il remplace celui de la pièce en cause, et SA meilleure
    solution vérifiée est recopiée (boulons, pièce d'attache, gorge)."""
    find(DB["profils"], nom)                      # un nom inconnu lève ValueError
    cle = piece_en_cause(cfg, A.propositions(cfg, u)) or "prof_S"
    Xp = _recherche_profil(cfg, u, cle, nom)
    u[cle] = A.valeur_widget(cfg, cle, nom)
    if Xp is not None and Xp.solutions:
        u.update(A.changements(cfg, u, Xp.solutions[0].u))


def bloc(cfg, R, u: Dict) -> Tuple[str, frozenset]:
    """Le bloc « À corriger » d'un assemblage que le moteur ne déclare pas vérifié (HTML), et les identifiants des
    alertes qu'il montre (les messages du pied ne les redisent pas). ("", ∅) : rien à dire ici — l'assemblage est
    vérifié, ou la configuration n'a pas de prédimensionnement."""
    if R is None or R.verified or not cfg.predim:
        return "", frozenset()
    cors = corrections(cfg, u, R)
    X = A.propositions(cfg, u)
    defauts = "".join(_defaut(cfg, R, x) for x in cors)
    montrees = solutions_a_montrer(cfg, u, X)
    if montrees:
        # des solutions existent : les défauts et leurs retouches d'abord (on garde sa disposition), puis les solutions
        corps = defauts + "".join(_solution(cfg, s, s is X.solutions[0], s is not X.solutions[0]) for s in montrees)
        reste = len(X.solutions) - len(montrees)
        if reste > 0:
            corps += (f'<div class="asm-sol-a"><button type="button" class="prof" data-pk="solution" title="Toutes les solutions vérifiées">'
                      f'<span>{AUTRES} ({reste})</span>{C.icon("chevron-down")}</button></div>')
    else:
        # AUCUNE disposition ne passe : c'est dit d'abord (retoucher une pince n'y changera rien), avec les profilés
        corps = f'<div class="asm-aucune">{A.aucune_solution(cfg, X, court=True)}</div>'
        cle = piece_en_cause(cfg, X)
        if cle is not None and _candidats(cle, u):
            corps += (f'<div class="asm-sol-a"><button type="button" class="prof" data-pk="profil_solution" title="Les profilés plus hauts, chacun avec sa solution vérifiée">'
                      f'<span>{TITRE_PROFILS}</span>{C.icon("chevron-down")}</button></div>')
        corps += defauts
    n = len(cors)
    etat_txt = (f"{n} défaut{'s' if n > 1 else ''} de géométrie" if n else "résistance insuffisante")
    tete = (f'<div class="et-bloc-h"><button type="button" class="et-bloc-t" data-bloc="{BLOC}" aria-expanded="true" title="Replier ou déplier">'
            f'{C.icon("chevron-down")}<span class="n">{TITRE}</span><span class="e">{C.esc(etat_txt)}</span></button></div>')
    html = f'<section class="et-bloc nok asm-corr" data-theme="{BLOC}">{tete}<div class="et-bloc-c">{corps}</div></section>'
    return html, frozenset(x.defaut.alerte.id for x in cors if x.defaut.alerte is not None)

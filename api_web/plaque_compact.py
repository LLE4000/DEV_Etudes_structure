# -*- coding: utf-8 -*-
"""L'établi compact des DALLES et des PRÉDALLES béton armé (api_web/dalle.py, api_web/predalle.py) : ce que les deux
écrans partagent — l'écran Streamlit prédalle est la copie adaptée de l'écran dalle, et leurs moteurs partagent les
formules (beton/predalle s'appuie sur beton/dalle). Les fonctions sont aux MÊMES PLACES que celles de la poutre
(api_web/etabli_compact.py ; docs/site/REFONTE_ETABLI.md) :

- à gauche, sous l'élément ouvert, quatre rubriques : 1 l'élément (béton, acier, dimensions), 2 « Sections et
  sollicitations » (une carte par section, ses quatre moments et V), 3 « Armatures » de la section COURANTE (les quatre
  nappes, chaque couche sur une ligne qui se déplie), 4 « Options » ;
- au centre, chaque vue sous son en-tête technique (la section, ses nappes, la bande des efforts) ;
- à droite, un bloc par section (``bloc``), coloré par son état, ses vérifications sur deux lignes.

Les pièces portent la MÊME clé dans les données (``data-cible``) et sur les dessins (``data-sel``, api_web/dessin_dalle.py
et dessin_predalle.py) : ``s<i>.<face>-<j>`` pour la couche j d'une nappe (« s0.inf_y-0 », la base de la nappe
inférieure Y de la section 0), ``s<i>.efforts`` pour les sollicitations, ``beton`` pour l'élément lui-même.

Ce fichier ne calcule rien : il met en forme ce que le moteur a rendu et CHOISIT parmi ce qu'il a nommé (l'état d'une
section : non conforme, données incomplètes, conforme)."""
from typing import Dict, List, Optional, Sequence, Tuple

from beton import treillis as TR

from . import commun as C
from . import etabli_compact as E

TYPES = (("treillis", "Treillis"), ("barres", "Barres"), ("n_barres", "n barres"))   # les libellés de l'écran
FACES_TXT = {"inf": "inférieure", "sup": "supérieure"}


def _cm(x: float) -> str:
    """Une longueur CALCULÉE en cm (axe, y_G) au plus au 1/100 : 3,5 ; 4,86."""
    return C.nb(round(x, 2), 2)


def titre_face(f: str) -> str:
    """« Inférieure Y »."""
    cote, dk = f.split("_")
    return f"{FACES_TXT[cote].capitalize()} {dk.upper()}"


def ordre_faces(principale: str) -> List[str]:
    """Les quatre nappes dans l'ordre de l'écran : inférieures puis supérieures, la direction principale d'abord."""
    autre = "x" if principale == "y" else "y"
    return [f"inf_{principale}", f"inf_{autre}", f"sup_{principale}", f"sup_{autre}"]


# ------------------------------------------------------------------ rubrique 2 : les sections et leurs sollicitations
def cartes(sections: Sequence[Dict]) -> str:
    """Une carte par section (api_web/etabli_compact.carte_section) : son nom, son menu « ⋯ » (Dupliquer, Supprimer), et
    ses sollicitations en un TABLEAU compact (13/11/2026, demande du propriétaire — docs/site/DALLES_FUSION.md § 8.3) :
    une ligne par DIRECTION (X : les moments portés par les barres X ; Y : par les barres Y), une colonne par face
    tendue — M⁺ (moment positif : la face inférieure est tendue, en travée), M⁻ (moment négatif : la face supérieure est
    tendue, sur appui), saisis sans signe — et V, l'effort tranchant de la section, à côté (UN par section : τ ne lit pas
    la direction). Deux lignes au lieu de quatre ; les unités dans les champs ; les explications au survol des titres."""
    seule = len(sections) == 1
    tete = ('<span></span>'
            '<span class="h" title="Moment positif : la face INFÉRIEURE est tendue (en travée). Saisi sans signe.">'
            'M<sup>+</sup> · inf. tendue</span>'
            '<span class="h" title="Moment négatif : la face SUPÉRIEURE est tendue (sur appui). Saisi sans signe.">'
            'M<sup>−</sup> · sup. tendue</span>'
            '<span class="h" title="Effort tranchant de la section : un seul, τ = V / (k·b·h) ne dépend pas de la direction.">V</span>')
    out = ""
    for i, s in enumerate(sections):
        ch = ["sections", i]
        corps = (f'<div class="et-mom et-mom4">{tete}'
                 f'<span class="d" title="X : les moments portés par les barres X">X</span>'
                 f'{C.num(ch + ["Mx_inf"], s["Mx_inf"], "kN·m")}{C.num(ch + ["Mx_sup"], s["Mx_sup"], "kN·m")}'
                 f'<div class="et-v">{C.num(ch + ["V"], s["V"], "kN")}</div>'
                 f'<span class="d" title="Y : les moments portés par les barres Y">Y</span>'
                 f'{C.num(ch + ["My_inf"], s["My_inf"], "kN·m")}{C.num(ch + ["My_sup"], s["My_sup"], "kN·m")}</div>')
        out += E.carte_section(i, s["nom"], ch + ["copie"], ch + ["-"], seule, corps, copier_txt="Dupliquer la section")
    return (f'<div class="et-secs">{out}</div>'
            + E.pied_sections("Valeurs de service (ELS), sur la bande de largeur b, saisies sans signe. X : les moments portés "
                              "par les barres X ; Y : par les barres Y. M⁺ tend la face inférieure (travée), M⁻ la face "
                              "supérieure (appui).", "ELS · bande b"))


# ------------------------------------------------------------------ rubrique 3 : les armatures de la section courante
def auto_ou_saisi(chemin: Sequence, valeur: Optional[float], auto: float, revenir: str) -> str:
    """Un axe ou un centre de gravité « vide = automatique » : le champ (l'automatique en texte indicatif) ; SAISI, le
    bouton « ↺ » à côté le rend à l'automatique (la valeur vidée, comme l'écran d'avant du site), son infobulle dit ce
    que serait l'automatique. ``revenir`` : « Revenir à l'axe automatique »…"""
    champ = C.num(chemin, valeur, "cm", auto=f"auto · {_cm(auto)}")
    if valeur is None:
        return champ
    return (f'<div class="et-auto">{champ}<button type="button" class="et-raz" data-chemin="{C._attr_chemin(chemin)}" '
            f'data-valeur="" title="{C.esc(revenir)} ({_cm(auto)} cm)" aria-label="{C.esc(revenir)}">{C.icon("undo-2")}</button></div>')


def couche(EN, i: int, f: str, j: int, c: Dict, geo_c: Dict, details: str = "", aide_axe: str = "", dit_auto: bool = True) -> str:
    """Une COUCHE d'une nappe, sur une ligne : son repère (B la base, R1… les renforts), son type, puis le treillis ou
    le Ø et l'espacement (ou le nombre de barres) ; dépliée (un clic sur le repère, ou sur le dessin) : son axe
    (vide = automatique ; saisi, « ↺ » le rend à l'automatique), ``details`` (ce que le module ajoute : la pose d'une
    couche de prédalle), sa section au mètre — et, l'axe saisi, ce que serait l'automatique (``dit_auto`` ; la
    prédalle le dit dans sa pose) —, et « Retirer » pour un renfort. ``EN`` : le module d'entrée du moteur (ses bornes
    et ses Ø)."""
    base = ["sections", i, "faces", f, "couches", j]
    tag = "B" if j == 0 else f"R{j}"
    nom = "Couche de base" if j == 0 else f"Renfort {j}"
    if c["type"] == "treillis":
        opts = TR.liste_choix()
        if c["treillis"] not in opts:
            opts = opts + [c["treillis"]]          # un ancien fichier : la désignation reste utilisable
        nappe = f'<div class="et-tr">{C.choix(base + ["treillis"], c["treillis"], [(x, x) for x in opts])}</div>'
        t = geo_c["treillis"]
        info = (f"treillis soudé : porteurs Ø{t[0]} tous les {t[2]} mm ; répartition Ø{t[1]} tous les {t[3]} mm (non comptés, "
                "comme dans l'écran Streamlit)")
    else:
        dia = f'<div class="et-dia">{C.choix(base + ["d"], c["d"], [(x, f"Ø{x}") for x in EN.DIAMETRES])}</div>'
        if c["type"] == "barres":
            nappe = dia + f'<div class="et-e">{C.num(base + ["esp"], c["esp"], "mm", entier=True, mini=EN.ESP_MIN, maxi=EN.ESP_MAX)}</div>'
            info = f"barres Ø{c['d']} tous les {c['esp']} mm"
        else:
            nappe = f'<div class="et-e et-n">{C.num(base + ["n"], c["n"], "", entier=True, mini=EN.N_MIN, maxi=EN.N_MAX)}</div>' + dia
            info = f"{c['n']} barres Ø{c['d']} réparties dans la bande"
    tete = (f'<div class="et-l1"><button type="button" class="et-tag" data-deplier="1" title="{nom} : axe et détail">{tag}</button>'
            f'<div class="et-ty">{C.choix(base + ["type"], c["type"], TYPES)}</div>{nappe}'
            f'<button type="button" class="et-dep" data-deplier="1" title="Axe et détail">{C.icon("chevron-down")}</button></div>')
    if c["dist"] is not None and dit_auto:
        info += f" · axe saisi, l'automatique serait {_cm(geo_c['e_auto'])} cm"
    det = (C.ligne("e axe", auto_ou_saisi(base + ["dist"], c["dist"], geo_c["e_auto"], "Revenir à l'axe automatique"), "",
                   aide_axe or "Du parement à l'axe des barres. Vide : l'axe automatique ; une valeur saisie l'impose.")
           + details
           + f'<div class="hint et-info">{C.esc(info)} · A<sub>s</sub> = {C.fr(geo_c["As_pm"], 0)}{C.NBSP}mm²/m</div>')
    if j > 0:
        det += E.retirer(base + ["-"], "Retirer ce renfort")
    return f'<div class="et-piece" data-cible="s{i}.{f}-{j}">{tete}<div class="et-det">{det}</div></div>'


def nappe(EN, i: int, f: str, face: Dict, geo: Dict, v: Dict, principale: str, couches: str, aide_renfort: str,
          verrou: bool = False) -> str:
    """Une NAPPE (face × direction) : son titre (la principale marquée), la section nécessaire que la vérification
    demande, le bouton du sélecteur de la couche de base ; ses couches (``couches``, HTML), « + Renfort » ; le centre de
    gravité (vide = calculé)."""
    _, dk = f.split("_")
    princ = '<span class="et-princ" title="Direction principale">princ.</span>' if dk == principale else ""
    prop = ("" if verrou else f'<button type="button" class="et-prop" data-pk="{f}-{i}" title="Proposer une nappe de base qui tient '
            f'(sélecteur)">{C.icon("lightbulb")}</button>')
    ajouter = (E.plus(["sections", i, "faces", f, "couches", "+"], "1", "Renfort") if len(face["couches"]) < EN.MAX_COUCHES else "")
    aide = f'<span class="et-pied-a" title="{C.esc(aide_renfort)}">A<sub>s</sub> posé {C.fr(geo["As_pm"], 0)}{C.NBSP}mm²/m</span>'
    yg = C.ligne("y<sub>G</sub> imposé", auto_ou_saisi(["sections", i, "faces", f, "yG"], face["yG"], geo["yG_calcule"],
                                                       "Revenir au centre de gravité calculé"),
                 "", "Centre de gravité des couches de la nappe, mesuré depuis le parement. Vide : calculé (Σ A·e / Σ A). "
                     "L'imposer remplace le calcul dans d = h − y_G.")
    return (f'<div class="et-grp"><div class="et-gh"><span class="t">{titre_face(f)}</span>{princ}'
            f'<span class="et-req">A<sub>s</sub> néc. <b>{C.fr(v["valeur"], 0)}</b>{C.NBSP}mm²</span>{prop}</div>'
            f'{couches}<div class="et-pied">{ajouter}{aide}</div>{yg}</div>')


def resume_armatures(rs: Dict) -> str:
    """Le résumé (rubrique repliée) des armatures d'une section : la base inférieure et la base supérieure de la
    direction principale."""
    p = rs["principale"]
    inf, sup = rs["faces"][f"inf_{p}"]["couches"], rs["faces"][f"sup_{p}"]["couches"]
    plus = lambda cs: f" +{len(cs) - 1}" if len(cs) > 1 else ""  # noqa: E731
    return f"inf. {p.upper()} {inf[0]['valeur']}{plus(inf)} · sup. {p.upper()} {sup[0]['valeur']}{plus(sup)}"


def rubrique_armatures(noms: Sequence[str], contenus: Sequence[str], resumes: Sequence[str]) -> Tuple[str, str]:
    """(corps, résumé) de la rubrique « Armatures » : le choix de la section courante, puis les nappes de chaque
    section ; le résumé de chaque section (``.et-r3``), celui de la section courante seul visible."""
    r3 = "".join(f'<span class="et-r3" data-section="{i}">{C.esc(t)}</span>' for i, t in enumerate(resumes))
    return E.armatures(noms, contenus), r3


def formulaire(cle1: str, titre1: str, resume1: str, s1: str, s2: str, n_sections: int, s3: str, r3: str, resume4: str, s4: str) -> str:
    """Les quatre rubriques de l'élément ouvert, aux places de celles de la poutre : 1 l'élément, 2 les sections et
    leurs sollicitations, 3 les armatures (de la section courante), 4 les options (repliées, « secondaires »)."""
    return (C.section(1, titre1, resume1, s1, True, cle1)
            + C.section(2, "Sections et sollicitations", f"{n_sections} section{'s' if n_sections > 1 else ''}", s2, False, "so")
            + C.section(3, "Armatures", r3, s3, False, "arm")
            + C.section(4, "Options", resume4, s4, False, "op", secondaire=True))


# ------------------------------------------------------------------ au centre : les vues et leur en-tête
def bande(s: Dict) -> List[Tuple[str, str, float, str]]:
    """La bande des efforts d'une section : les quatre moments et V."""
    return [("M", "x,inf", s["Mx_inf"], "kN·m"), ("M", "x,sup", s["Mx_sup"], "kN·m"), ("M", "y,inf", s["My_inf"], "kN·m"),
            ("M", "y,sup", s["My_sup"], "kN·m"), ("V", "", s["V"], "kN")]


def _nappes_txt(rs: Dict, cote: str) -> str:
    """« Inf. Y 10/10/100/100 + 3 Ø12 · X Ø8/200 » : les nappes d'une face, la direction principale d'abord."""
    p = rs["principale"]
    autre = "x" if p == "y" else "y"
    return (("Inf." if cote == "inf" else "Sup.") + " "
            + " · ".join(f"{dk.upper()} {rs['faces'][f'{cote}_{dk}']['libelle']}" for dk in (p, autre)))


def vues(sections: Sequence[Dict], element: Dict, dessin, prefixe: str, sous_titre: str, legendes: Dict[str, str]) -> List[Dict]:
    """Les vues de l'élément : pour chaque section, ses coupes (« c<i> ») et le plan de ses nappes (« p<i> »), sous le
    même en-tête technique, dont la dernière ligne est la légende du dessin (``legendes`` : « c » les coupes, « p » le
    plan — ce qu'on lit sur le dessin : barres coupées en points…) ; dès deux sections, « Toutes » (la coupe principale
    de chacune, à la même échelle, sous la légende des coupes). ``dessin`` : api_web/dessin_dalle ou dessin_predalle ; ``sous_titre`` : les
    dimensions et les matériaux (HTML)."""
    out = []
    for i, s in enumerate(sections):
        nom = s["nom"] if not s["nom"].lower().startswith("section") else s["titre"]
        for cle, titre, court, svg in (("c", "Coupes", "Coupe", dessin.svg_coupe), ("p", "Plan des nappes", "Plan", dessin.svg_plan)):
            ent = E.entete(s["titre"], f"{sous_titre} · principale {s['principale'].upper()}",
                           [_nappes_txt(s, "inf"), _nappes_txt(s, "sup")], bande(s), legendes[cle])
            out.append({"cle": f"{cle}{i}", "titre": f"{titre} · {s['titre']}", "court": f"{court} {nom}"[:24],
                        "svg": svg(f"{prefixe}_{cle}{i}", s, element, i), "entete": ent, "legende": legendes[cle]})
    if len(sections) > 1:
        out.append({"cle": "toutes", "titre": "Toutes les sections", "court": "Toutes",
                    "svg": dessin.svg_toutes(f"{prefixe}_toutes", sections, element),
                    "entete": E.entete("Toutes les sections", sous_titre,
                                       [f"{len(sections)} sections · même échelle · la coupe perpendiculaire à la direction "
                                        "principale de chacune"], None, legendes["c"]),
                    "legende": legendes["c"]})
    return out


# ------------------------------------------------------------------ à droite : un bloc par section
def etat_bloc(MO, rs: Dict) -> Tuple[str, str]:
    """L'état d'une SECTION, la couleur de l'en-tête de son bloc, comme la poutre : NON CONFORME d'abord — une
    non-conformité trouvée sans aucun effort (acier excessif, d utile ≤ 0) n'est jamais cachée ; puis « Données
    incomplètes » sans aucun effort (ses vérifications passent toutes et ne disent rien) ; sinon conforme."""
    if rs["statut"] == MO.NOK:
        return "nok", "Non conforme"
    if not rs["sollicitee"]:
        return "inc", "Données incomplètes"
    return "ok", "Conforme"


def niveau_element(MO, M, r: Dict) -> str:
    """Le niveau d'un élément calculé (la pastille de la note) : non conforme > données incomplètes (une section sans
    effort) > le taux de sa vérification gouvernante. ``M`` : le module (api_web.dalle, api_web.predalle)."""
    if r["statut"] == MO.NOK:
        return "r"
    if not r["sollicitee"]:
        return "n"
    return C.niveau(M.gouvernante(r)[1]["taux"])


def comparaison(M, v: Dict, s: Dict) -> str:
    """La ligne 2 d'une vérification : la valeur et sa limite, NOMMÉES (« A_s,néc = 707 ≤ A_s = 785 mm² ») ; une
    vérification non conforme sous 100 % (acier excessif) dit ce qui la rend non conforme (M._verdict) ; une face dont
    le d utile est ≤ 0 le dit, QUEL QUE SOIT son taux (« d = −0,5 cm ≤ 0 · nappe hors de l'épaisseur ») — c'est la
    ligne qui le dit, plus un message sous le bloc (13/11/2026 : au-delà de 100 %, la ligne disait « A_s,néc = 16 667 >
    A_s = 785 mm² » et la raison n'était que dans ce message)."""
    if M._sous_limite_non_conforme(v) or v["cle"] not in M.EN.FACES or "d_utile_negatif" in v["motifs"]:
        return M._verdict(v, s)
    le = "≤" if v["taux"] <= 1 else "&gt;"
    motif = f" · {M._motifs(v)}" if v["motifs"] and v["taux"] <= 1 else ""
    return (f"A<sub>s,néc</sub>{C.NBSP}={C.NBSP}{M._n(v['valeur'], 'mm²')} {le} "
            f"A<sub>s</sub>{C.NBSP}={C.NBSP}{M._n(v['limite'], 'mm²')}{C.NBSP}mm²{motif}")


def piece_courte(v: Dict, s: Dict) -> str:
    """La nappe qu'une vérification de face nomme sur sa ligne 1, à côté de son taux : la désignation courte de sa base
    (« 10/10/100/100 », « Ø10/150 », « 3 Ø12 »), ses renforts ajoutés (« Ø10/150 + 3 Ø12 », « … + 2 renforts »)."""
    cs = s["faces"][v["cle"]]["couches"]
    if len(cs) == 1:
        return C.esc(cs[0]["valeur"])
    if len(cs) == 2:
        return C.esc(f"{cs[0]['valeur']} + {cs[1]['valeur']}")
    return C.esc(f"{cs[0]['valeur']} + {len(cs) - 1} renforts")


def bloc(M, i: int, s: Dict, nom_element: str, hyps: List[str], gov: Optional[Tuple[int, Dict]],
         pied_calcul: str = "", pied: str = "") -> Tuple[str, str]:
    """Le bloc d'UNE section (api_web/etabli_compact.bloc) : l'en-tête (son titre, son état, coloré), les efforts, les
    vérifications (deux lignes ; un clic ouvre le détail de la vérification, le bouton de l'en-tête tout le calcul de
    la section), puis ``pied`` (les avertissements de la section). ``M`` : le module (api_web.dalle, api_web.predalle)
    dont on lit les formules mises en mots ; ``gov`` : (section, vérification) qui gouverne la note, marquée. Rend
    (html, volets).

    La nappe d'une face y est NOMMÉE (« 10/10/100/100 »), en texte : elle ne s'y choisit plus. Le résultat dit ce qui
    tient ; les armatures se modifient à UN endroit, la rubrique « Armatures » (son sélecteur garde la retenue ✓ et la
    plus petite qui tient ★) — le propriétaire, 13/11/2026 : « éviter de proposer à nouveau l'édition des armatures
    dans les résultats ; une information doit être encodée une seule fois »."""
    cls, etat_txt = etat_bloc(M.MO, s)
    v_s = M.gouvernante_section(s)
    rows, volets, tout = "", "", ""
    titre_calcul = f"{C.esc(nom_element)} · {C.esc(s['titre'])}"
    sous_titre = f"{M._efforts_txt(s)} · méthode du bureau"
    for v in s["verifications"]:
        face = v["cle"] in M.EN.FACES
        piece = piece_courte(v, s) if face else ""
        rows += C.ligne_verif(M.nom_ligne(v), comparaison(M, v, s), C.pct(v["taux"]), M._niv(v), calcul=f"s{i}-{v['cle']}",
                              piece=piece, gov=gov is not None and gov[0] == i and gov[1] is v and cls != "inc")
        titre = v["titre"]
        if v["cle"] == "hauteur" and v.get("face"):
            cote, dk = v["face"].split("_")
            titre += f" · la famille {cote}. {dk.upper()} gouverne (la plus exigeante)"
        detail = (f'<div class="ck-g"><div class="ck-t">{C.esc(titre)}</div>'
                  f'{C.lignes_fx([M._formule(f) for f in v["formules"]] + [M._taux_fx(v, s)])}</div>')
        tout += detail
        volets += C.fenetre_calcul(f"s{i}-{v['cle']}", f"{titre_calcul} · {C.esc(M.nom_minuscule(v))}", sous_titre, hyps,
                                   [M._ligne_tableau(v, s, True)], detail, pied=pied_calcul)
    lignes = [M._ligne_tableau(v, s, v is v_s) for v in s["verifications"]]
    volets += C.fenetre_calcul(f"s{i}", titre_calcul, sous_titre, hyps, lignes, tout, pied=pied_calcul)
    efforts = f"{M._efforts_txt(s)} · principale {s['principale'].upper()}"
    return E.bloc(i, s["titre"], cls, etat_txt, efforts, rows, pied), volets

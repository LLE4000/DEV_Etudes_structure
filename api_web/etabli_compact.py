# -*- coding: utf-8 -*-
"""Les briques PARTAGÉES de l'établi compact (site/web/construire.py ETABLI_COMPACT ; js/compact.js ; css/etabli.css).

Les modules « Poutres », « Dalles » et « Prédalles » béton armé les assemblent, pour que leurs fonctions soient AU MÊME
ENDROIT dans les trois écrans (docs/site/REFONTE_ETABLI.md, « Un seul établi pour les trois modules béton armé ») :
- à gauche, l'ARBORESCENCE de la note (``arbre``) : chaque élément « nom · dimensions », son menu « ⋯ » (renommer,
  copier, supprimer), et sous l'élément ouvert ses rubriques en accordéon ; les menus et les petits boutons (``menu``,
  ``action``, ``plus``, ``retirer``) ;
- au centre, l'EN-TÊTE technique d'une vue (``entete``) : trois lignes, et à droite la bande compacte des efforts ;
- à droite, la tête des résultats (``tete_resultats`` : le nom de l'élément et ses dimensions) puis un BLOC par section
  (``bloc``), dont l'en-tête est coloré par l'état de la section.

Ce fichier ne calcule rien et ne choisit rien : chaque module lui donne ses noms, ses dimensions, ses lignes et ses états
déjà mis en mots ; il en fait le balisage que js/compact.js et css/etabli.css lisent."""
import re
from typing import NamedTuple, Optional, Sequence, Tuple

from . import commun as C

_SANS_BALISES = re.compile(r"<[^>]+>")   # les dimensions sont du HTML (« · 25 × 50 cm ») ; un title veut du texte


class Genre(NamedTuple):
    """Les mots d'un élément de la note, pour les libellés de l'arborescence : (« Poutres », « la poutre »,
    « cette poutre », « une poutre »). Le genre se lit dans l'article (« un garde-corps », « une barre ») : les phrases
    de la ligne de l'élément s'accordent avec lui (``masculin``)."""
    liste: str
    le: str
    ce: str
    un: str

    @property
    def masculin(self) -> bool:
        return self.un.startswith("un ")


POUTRE = Genre("Poutres", "la poutre", "cette poutre", "une poutre")
DALLE = Genre("Dalles", "la dalle", "cette dalle", "une dalle")
PREDALLE = Genre("Prédalles", "la prédalle", "cette prédalle", "une prédalle")


# ------------------------------------------------------------------ menus et petits boutons
def menu(items: str, titre: str, bouton: Optional[str] = None, classe: str = "") -> str:
    """Un menu discret : le bouton (« ⋯ » ; ``bouton`` : un autre contenu, du HTML) et la liste (cachée) de ses actions.
    ``classe`` : une classe de plus sur le menu (``et-elt-liste`` : la liste des éléments de la note)."""
    return (f'<div class="et-mn{" " + classe if classe else ""}"><button type="button" class="et-mn-b" data-menu="1" aria-haspopup="menu" '
            f'aria-expanded="false" title="{C.esc(titre)}">{bouton if bouton is not None else C.icon("ellipsis")}</button>'
            f'<div class="et-mn-l" role="menu" hidden>{items}</div></div>')


def action(attrs: str, icone: str, texte: str, danger: bool = False, actif: bool = True) -> str:
    """Une action d'un menu « ⋯ » ; ``attrs`` : ce que js/etabli.js lit (data-chemin + data-valeur, data-renommer…)."""
    return (f'<button type="button" role="menuitem" class="{"danger" if danger else ""}" {attrs}{"" if actif else " disabled"}>'
            f'{C.icon(icone)}{C.esc(texte)}</button>')


def ligne_export(attrs: str, icone: str, titre: str, texte: str = "", actif: bool = True) -> str:
    """Une ligne du menu « Exporter » : son titre, puis EN CLAIR ce qu'elle produit (``texte``) ; ``attrs`` : ce que
    js/etabli.js lit (``data-telecharger="<quoi>"``, ``data-calcul="<fenêtre>"``). Éteinte (``actif`` faux), elle dit
    pourquoi dans son texte."""
    t = f'<small>{C.esc(texte)}</small>' if texte else ""
    return (f'<button type="button" role="menuitem" class="et-exp-i" {attrs}{"" if actif else " disabled"}>{C.icon(icone)}'
            f'<span><b>{C.esc(titre)}</b>{t}</span></button>')


def exporter(lignes: Sequence[str], titre: str = "Exporter") -> str:
    """Le bouton « Exporter » et son menu (un menu discret du kit, ouvert vers le haut quand la place manque en bas :
    js/compact.js) : UN endroit pour tous les fichiers d'une note — une ligne par export (``ligne_export``)."""
    # (le chevron pointe vers le haut — le menu s'ouvre au-dessus : css/etabli.css retourne celui du kit)
    bouton = f'{C.icon("download")}<span>{C.esc(titre)}</span>{C.icon("chevron-down")}'
    return (f'<div class="et-mn et-exp"><button type="button" class="btn sm et-exp-b" data-menu="1" aria-haspopup="menu" '
            f'aria-expanded="false" title="{C.esc(titre)} : les fichiers de cette note">{bouton}</button>'
            f'<div class="et-mn-l et-exp-l" role="menu" hidden>{"".join(lignes)}</div></div>')


def plus(chemin: Sequence, valeur: str, texte: str) -> str:
    """« + Lit », « + Section » : un bouton qui ajoute (le chemin finit par « + »)."""
    return (f'<button type="button" class="et-add" data-chemin="{C._attr_chemin(chemin)}" data-valeur="{C.esc(valeur)}">'
            f'{C.icon("plus")}{C.esc(texte)}</button>')


def retirer(chemin: Sequence, texte: str) -> str:
    """« Retirer ce lit », « Retirer ce renfort » : dans le détail d'une pièce (le chemin finit par « - »)."""
    return (f'<button type="button" class="et-ret" data-chemin="{C._attr_chemin(chemin)}" data-valeur="1">'
            f'{C.icon("trash-2")}{C.esc(texte)}</button>')


# ------------------------------------------------------------------ à gauche : l'arborescence de la note
def arbre(genre: Genre, elements: Sequence[Tuple[str, str, bool]], actif: int, corps: str, marque: str, compact: bool = False,
          **ligne) -> str:
    """La colonne de gauche : l'ARBORESCENCE de la note. Chaque élément ``(nom, dimensions, verrou)`` — le nom en texte,
    les dimensions en HTML (« · 25 × 50 cm ») — s'écrit sans taux ni pastille (le résultat est à droite) ; l'élément
    ouvert (``actif``) se développe et montre ``corps``, ses rubriques ; un clic sur un autre l'ouvre (le premier se
    replie). Le NOM de l'élément ouvert se modifie sur sa ligne même (champ « nom » : les rubriques n'ont plus de champ
    Nom, demande de l'ingénieur, 11/2026) — sauf verrouillé, où il se lit ; « Renommer » du menu « ⋯ » y place le
    curseur. Copier, supprimer : dans le même menu. ``marque`` : l'attribut qui porte le rang (``data-poutre``,
    ``data-dalle``…).

    ``compact`` (docs/site/DALLES_FUSION.md § 1 ; la dalle et la poutre l'emploient, site/web/construire.ARBRE_COMPACT) :
    la colonne sans en-tête « LISTE (n) » ni arborescence — ``arbre_compact`` (``ligne`` : ses options, ``cadenas``,
    ``niveaux``)."""
    if compact:
        return arbre_compact(genre, elements, actif, corps, marque, **ligne)
    out = ""
    seul = len(elements) == 1
    for i, (nom_txt, dims, verrou) in enumerate(elements):
        on = i == actif
        nom = C.esc(nom_txt)
        # (verrouillé, le nom se lit sans se modifier : « Renommer » n'aurait aucun champ où poser le curseur)
        actions = (action(f'data-renommer="{i}"', "pencil", "Renommer", False, not verrou)
                   + action(f'data-dupliquer-index="{i}"', "copy", f"Copier {genre.le}")
                   + action(f'data-supprimer="{i}" data-confirmer="Supprimer {genre.le} « {nom} » de la note ?"', "trash-2",
                            f"Supprimer {genre.le}", True, not seul))
        cadenas = C.icon("lock") if verrou else ""
        dedans = f'<div class="et-pt-corps">{corps}</div>' if on else ""
        if on and not verrou:
            # le chevron replie ou déplie l'élément ; le nom est un champ (la ligne garde son allure de titre)
            tete = (f'<button type="button" class="et-pt-t on et-pt-chev" data-activer="{i}" aria-expanded="true" title="Replier ou déplier">'
                    f'{C.icon("chevron-right")}</button>'
                    f'<span class="t et-pt-n"><input data-chemin="nom" data-sorte="texte" value="{nom}" class="et-pt-nom" '
                    f'size="{max(4, len(nom_txt) + 1)}" spellcheck="false" aria-label="Nom de {genre.ce}" title="Nom de {genre.ce} : '
                    f'cliquer pour le modifier"><span class="dim">{dims}</span></span>')
        else:
            tete = (f'<button type="button" class="et-pt-t{" on" if on else ""}" data-activer="{i}" aria-expanded="{"true" if on else "false"}" '
                    f'title="{"Replier ou déplier" if on else f"Ouvrir {genre.ce}"}">{C.icon("chevron-right")}'
                    f'<span class="t">{cadenas}<span class="nom">{nom}</span><span class="dim">{dims}</span></span></button>')
        out += (f'<div class="et-pt{" on" if on else ""}" data-{marque}="{i}"><div class="et-pt-l">{tete}'
                f'{menu(actions, f"Actions de {genre.le}")}</div>{dedans}</div>')
    return (f'<div class="et-arbre"><div class="et-arbre-h"><span>{genre.liste}</span><span class="n">{len(elements)}</span></div>'
            f'{out}<button type="button" class="et-pt-add" data-ajouter="1">{C.icon("plus")}Ajouter {genre.un}</button></div>')


# l'ÉTAT d'un élément, dit par la couleur de son nom (``niveaux`` d'arbre_compact) : le niveau de la note (C.niveau, ou
# « r » non conforme, « n » données incomplètes) → la classe du nom : conforme (vert), non conforme (rouge), incomplet
ETATS_NOM = {"p": "ok", "f": "ok", "r": "nok", "n": "inc"}
ETATS_TXT = {"ok": "conforme", "nok": "non conforme", "inc": "données incomplètes"}


def arbre_compact(genre: Genre, elements: Sequence[Tuple[str, str, bool]], actif: int, corps: str, marque: str, *,
                  cadenas: bool = False, niveaux: Optional[Sequence[str]] = None) -> str:
    """La colonne de gauche COMPACTE (docs/site/DALLES_FUSION.md § 1, revue avec l'ingénieur le 09/11/2026) : ni
    en-tête « LISTE (n) », ni arborescence ; les rubriques de l'élément ouvert (``corps``) sous UNE LIGNE ``.et-elsel``,
    que js/etabli.js pose dans l'en-tête du panneau, à côté d'« Annuler » et « Rétablir » (l'emplacement ``.et-elt``,
    site/web/construire.compact) :
    - le NOM de l'élément ouvert, un champ à l'allure d'un titre : un clic, et on le renomme (``data-chemin="nom"``,
      « modifier » en quittant le champ ou sur Entrée ; Échap rend l'ancien nom) — verrouillé, il se lit, un cadenas
      devant ;
    - la LISTE des éléments de la note (un menu : chacun avec ses dimensions, l'ouvert coché, ``data-activer="i"`` ; et
      « Ajouter … », ``data-ajouter``) — le compte sur le bouton dès deux éléments ;
    - COPIER l'élément ouvert (``data-dupliquer-index``, la copie s'insère après lui) et le SUPPRIMER (``data-supprimer``,
      avec confirmation ; éteint quand il est seul : une note garde au moins un élément).
    Les attributs sont ceux que js/etabli.js lisait déjà sur l'arborescence (agirElement).

    Deux options (la dalle, 13/11/2026, demande du propriétaire — sans elles, la sortie est la même, à l'octet près) :
    - ``cadenas`` : le VERROU de l'élément ouvert est un bouton de la ligne, à droite avec copier et supprimer
      (``data-chemin="verrou"``, la valeur qu'un clic pose) — et non plus une case de la première rubrique ;
    - ``niveaux`` (un par élément : « p », « f », « r », « n ») : l'ÉTAT de chaque élément dit par la couleur de son nom,
      dans la liste et sur la ligne — vert conforme (taux ≤ 100 %), rouge non conforme, neutre sans effort — pour
      reconnaître d'un coup d'œil l'élément ouvert et son état."""
    seul = len(elements) == 1
    actif = min(max(0, actif), len(elements) - 1)
    nom_txt, _dims, verrou = elements[actif]
    nom = C.esc(nom_txt)
    etats = [ETATS_NOM.get(n, "inc") for n in niveaux] if niveaux is not None else None
    cl_etat = f" et-n-{etats[actif]}" if etats else ""
    dit_etat = f" — {ETATS_TXT[etats[actif]]}" if etats else ""
    e_ = "" if genre.masculin else "e"
    if verrou:
        ou = "le cadenas, à droite, " if cadenas else "la case « Verrouiller » de la première rubrique "
        champ = (f'<span class="et-elt-nom lu{cl_etat}" title="{C.esc(genre.le.capitalize())} est verrouillé{e_} : son nom se lit '
                 f'({ou}la déverrouille){dit_etat}">{C.icon("lock")}<span>{nom}</span></span>')
    else:
        champ = (f'<input data-chemin="nom" data-sorte="texte" value="{nom}" class="et-elt-nom{cl_etat}" size="{max(4, len(nom_txt) + 1)}" '
                 f'spellcheck="false" autocomplete="off" aria-label="Nom de {genre.ce}" title="Nom de {genre.ce} : cliquer pour le modifier{dit_etat}">')
    items = ""
    for i, (n, d, v) in enumerate(elements):
        on = i == actif
        etat_i = f" ({ETATS_TXT[etats[i]]})" if etats else ""
        items += (f'<button type="button" role="menuitemradio" aria-checked="{"true" if on else "false"}" class="et-elt-i{" on" if on else ""}" '
                  f'data-activer="{i}" title="{"Ouvert" if on else f"Ouvrir {genre.ce}"} : {C.esc(n)} {C.esc(_SANS_BALISES.sub("", d).strip())}{etat_i}">'
                  f'{C.icon("check") if on else "<span class=\"ic\"></span>"}{C.icon("lock") if v else ""}'
                  f'<span class="nom{f" et-n-{etats[i]}" if etats else ""}">{C.esc(n)}</span><span class="dim">{d}</span></button>')
    items += action('data-ajouter="1"', "plus", f"Ajouter {genre.un}")
    compte = f'<span class="n">{len(elements)}</span>' if not seul else ""
    un = "un" if genre.masculin else "une"
    liste = menu(items, f"{genre.liste} de la note : en ouvrir {un} autre, en ajouter {un}", C.icon("chevron-down") + compte, "et-elt-liste")
    copier = (f'<button type="button" class="et-elt-b" data-dupliquer-index="{actif}" title="Copier {genre.le} (la copie s\'ajoute juste après)" '
              f'aria-label="Copier {genre.le}">{C.icon("copy")}</button>')
    supprimer = (f'<button type="button" class="et-elt-b danger" data-supprimer="{actif}" data-confirmer="Supprimer {genre.le} « {nom} » de la note ?" '
                 f'title="{f"Supprimer {genre.le}" if not seul else f"Une note garde au moins {genre.un}"}" aria-label="Supprimer {genre.le}"'
                 f'{" disabled" if seul else ""}>{C.icon("trash-2")}</button>')
    verrouiller = ""
    if cadenas:
        quoi = f"Déverrouiller {genre.le}" if verrou else f"Verrouiller {genre.le} (lecture seule)"
        verrouiller = (f'<button type="button" class="et-elt-b et-verrou{" on" if verrou else ""}" data-chemin="verrou" '
                       f'data-valeur="{"false" if verrou else "true"}" aria-pressed="{"true" if verrou else "false"}" title="{quoi}" '
                       f'aria-label="{quoi}">{C.icon("lock" if verrou else "lock-open")}</button>')
    return (f'<div class="et-arbre et-arbre-c" data-{marque}="{actif}"><div class="et-elsel{" et-elsel-d" if cadenas else ""}" '
            f'data-elements="{len(elements)}">{champ}{liste}{verrouiller}{copier}{supprimer}</div>{corps}</div>')


# ------------------------------------------------------------------ au centre : l'en-tête technique d'une vue
def entete(titre: str, sous_titre: str, lignes: Sequence[str], bande: Optional[Sequence[Tuple[str, str, float, str]]],
           legende: str = "") -> str:
    """L'en-tête technique d'une vue : trois lignes (``titre`` en capitales et ``sous_titre`` — du HTML du module, la
    section et ses matériaux ; puis ``lignes``, du texte : les aciers…) et, à droite, la bande compacte des efforts — une
    fois, sans couleur d'alerte. ``bande`` : ``(symbole, indice, valeur, unité)`` par effort ; None : pas de bande.
    ``legende`` (texte) : ce qu'on lit sur le dessin (« barres coupées en points… »), en dernière ligne, discrète."""
    l1 = f'<div class="l1"><b>{C.esc(titre)}</b><span>· {sous_titre}</span></div>'
    corps = "".join(f'<div class="l{k + 2}">{C.esc(t)}</div>' for k, t in enumerate(lignes))
    if legende:
        corps += f'<div class="lg">{C.esc(legende)}</div>'
    eff = ""
    if bande is not None:
        eff = ('<div class="et-eff">'
               + "".join(f'<span><i>{s}</i>{f"<sub>{x}</sub>" if x else ""}<b>{C.fr(v, 1)}</b><small>{u}</small></span>'
                         for s, x, v, u in bande)
               + '</div>')
    return f'<div class="et-ent"><div class="et-ent-t">{l1}{corps}</div>{eff}</div>'


# ------------------------------------------------------------------ à droite : les résultats
def tete_resultats(nom: str, dims: str) -> str:
    """La tête du cadre des résultats : le nom de l'élément ouvert (HTML échappé par le module) et ses dimensions — un
    titre qui se lit ; le nom se modifie à UN seul endroit, la ligne de l'élément en tête de la colonne de gauche
    (``arbre_compact``)."""
    return f'<div class="et-rp"><b>{nom}</b><span>{dims}</span></div>'


def bloc(i: int, titre: str, cls: str, etat_txt: str, efforts: str, rows: str, pied: str = "", *, theme: bool = False, calcul: str = "") -> str:
    """Le bloc d'UNE section : l'en-tête (son ``titre`` et son état ``etat_txt``, en texte ; la couleur ``cls`` : ok,
    warn, nok, inc), les efforts (HTML), les vérifications ``rows`` (deux lignes chacune, C.ligne_verif), puis ``pied``
    (ce que le module ajoute : ses avertissements). Le bouton de l'en-tête ouvre « tout le calcul » (fenêtre ``s<i>``).

    ``theme`` (les cornières, 11/2026) : un bloc THÉMATIQUE — « Sections », « Ancrages », « Comparaison » — et non celui
    d'une section de l'élément : il porte ``data-theme`` au lieu de ``data-section``, et js/compact.js le laisse toujours
    visible (un bloc de section ne se montre que pour la section courante) tout en le repliant comme les autres.

    ``calcul`` (les soudures, 12/11/2026) : la fenêtre que le bouton de l'en-tête ouvre, quand le module NOMME la sienne
    (celle du cordon gouvernant, « c2 » ; celle des quatre lectures, « m ») — sans lui, ``s<i>``, et la sortie est la
    même, à l'octet près."""
    tete = (f'<div class="et-bloc-h"><button type="button" class="et-bloc-t" data-bloc="{i}" aria-expanded="true" title="Replier ou déplier">'
            f'{C.icon("chevron-down")}<span class="n">{C.esc(titre)}</span><span class="e">{C.esc(etat_txt)}</span></button>'
            f'<button type="button" class="et-tout" data-calcul="{C.esc(calcul) if calcul else f"s{i}"}" '
            f'title="Tout le calcul{"" if theme else " de la section"}">{C.icon("file-text")}</button></div>')
    return (f'<section class="et-bloc {cls}" data-{"theme" if theme else "section"}="{i}">{tete}<div class="et-bloc-c">'
            f'<div class="et-bloc-ef">{efforts}</div><div class="els">{rows}</div>{pied}</div></section>')


# ------------------------------------------------------------------ à droite : les MOTS communs du béton armé (E15)
# La poutre, la dalle pleine et la prédalle disent leurs résultats avec les MÊMES mots (docs/site/CHARTE_ERGONOMIE.md
# E15, lot 4) : écrits une fois, ici. Ce qu'un calcul de flexion de la méthode du bureau ne vérifie PAS (la poutre, la
# dalle coulée en place ; la prédalle a sa phrase, plus longue) — une phrase sous la colonne des résultats.
NON_VERIFIEES_FLEXION = "Non vérifiées : la flèche et la fissuration."


def cm_calcule(x: float) -> str:
    """Une longueur CALCULÉE en cm (un axe, un centre de gravité y_G) au plus au 1/100 : 3,5 ; 4,86 ; 6."""
    return C.nb(round(x, 2), 2)


def valeur_hauteur(v) -> str:
    """La valeur de la ligne « Hauteur » de la colonne des résultats, la somme DÉTAILLÉE (DAL8 — le propriétaire,
    13/11/2026 : « on doit voir la part qui a été calculée plus ce qu'on prend comme enrobage ») : « h_min = 12,6 + 3,5 =
    16,1 cm » — h_u,min (la part calculée) + y_G (l'enrobage mécanique), les termes de la formule « h_min » que le moteur
    nomme (la dalle, la prédalle et la poutre la nomment de même). Sans la formule (un moteur qui ne la nommerait pas),
    la seule somme."""
    N = C.NBSP
    f = next((f for f in v.get("formules", []) if f["cle"] == "h_min"), None)
    if f is None:
        return f"h<sub>min</sub>{N}={N}{C.fr(v['valeur'], 1)}{N}cm"
    t = {x["symbole"]: x["valeur"] for x in f["termes"]}
    return f"h<sub>min</sub>{N}={N}{C.fr(t['hu_min'], 1)}{N}+{N}{cm_calcule(t['yG'])}{N}={N}{C.fr(f['resultat'], 1)}{N}cm"


def comparaison_acier(necessaire: str, pose: str, depasse: bool, suite: str = "") -> str:
    """La ligne 2 d'une vérification d'ARMATURE : « A_s,néc = 707 ≤ A_s = 785 mm² » — la section nécessaire
    (max(A_s,req ; A_s,min)) face à la section posée, nommées ; ``necessaire`` et ``pose`` déjà écrits (mm² entiers) ;
    ``depasse`` : « > » au lieu de « ≤ » ; ``suite`` (HTML) : ce que le module ajoute (« · acier excessif »…)."""
    N = C.NBSP
    return f"A<sub>s,néc</sub>{N}={N}{necessaire} {'&gt;' if depasse else '≤'} A<sub>s</sub>{N}={N}{pose}{N}mm²{suite}"


def acier_excessif(pose: str, maxi: str) -> str:
    """La ligne 2 d'une armature NON CONFORME sous 100 % parce qu'elle dépasse le maximum : « A_s = 4 825 mm² >
    A_s,max = 3 200 mm² · acier excessif » — la comparaison qui la rend non conforme, et non un taux qui tient."""
    N = C.NBSP
    return f"A<sub>s</sub>{N}={N}{pose}{N}mm² &gt; A<sub>s,max</sub>{N}={N}{maxi}{N}mm² · acier excessif"


# ------------------------------------------------------------------ à gauche : les MOTS communs du béton armé (E15)
# Les rubriques de la poutre, de la dalle pleine et de la prédalle disent les mêmes choses avec les MÊMES mots
# (docs/site/CHARTE_ERGONOMIE.md E15, lot 4) : écrits une fois, ici.
def symbole(lettre: str, mot: str) -> str:
    """Une dimension dite par son SYMBOLE (« b », « h », « c ») : le mot se lit au survol (13/11/2026, § 8.2 des
    dalles : « bande », « épaisseur », « enrobage » ne se répètent plus à côté du symbole). ``lettre`` : du HTML."""
    return f'<span class="et-sym" title="{C.esc(mot)}">{lettre}</span>'


def entetes_moments(titre_v: str) -> str:
    """Les titres du tableau des efforts d'une section : M⁺ (la face inférieure tendue), M⁻ (la face supérieure
    tendue), saisis sans signe, et V — leur explication au survol. ``titre_v`` : ce que le survol de V dit."""
    return ('<span class="h" title="Moment positif : la face INFÉRIEURE est tendue (en travée). Saisi sans signe.">'
            'M<sup>+</sup> · inf. tendue</span>'
            '<span class="h" title="Moment négatif : la face SUPÉRIEURE est tendue (sur appui). Saisi sans signe.">'
            'M<sup>−</sup> · sup. tendue</span>'
            f'<span class="h" title="{C.esc(titre_v)}">V</span>')


def as_necessaire(valeur: str) -> str:
    """La section NÉCESSAIRE d'une armature, dans le titre de son groupe : « A_s néc. 588 mm² »."""
    return f'<span class="et-req">A<sub>s</sub> néc. <b>{valeur}</b>{C.NBSP}mm²</span>'


def as_pose(valeur: str, unite: str, aide: str) -> str:
    """La section POSÉE d'une armature, au pied de son groupe, à droite : « A_s posé 785 mm²/m » ; ``aide`` au survol."""
    return f'<span class="et-pied-a" title="{C.esc(aide)}">A<sub>s</sub> posé {valeur}{C.NBSP}{unite}</span>'


def bouton_liste(pk: str, quoi: str) -> str:
    """Le bouton qui ouvre la LISTE des armatures possibles d'une face (le sélecteur, triées par section ; ★ la plus
    petite qui tient) — une icône de liste (c'était une ampoule, 13/11/2026). ``quoi`` : ce que la liste choisit."""
    return (f'<button type="button" class="et-prop" data-pk="{C.esc(pk)}" title="Choisir {C.esc(quoi)} dans la liste des armatures, '
            f'triées par section (★ : la plus petite qui tient)">{C.icon("list-filter")}</button>')


def ligne_methode(reference: str) -> str:
    """La ligne « Méthode » des Options : la méthode du bureau en clair ; la référence du moteur au survol."""
    return (f'<div class="row full et-meth"><label><span>Méthode</span></label>'
            f'<b title="{C.esc(reference)}">Méthode du bureau · ELS · z = 0,9 d</b></div>')


def resume_options(gamma_s: float, jeu: float, pas) -> str:
    """Le résumé de la rubrique « Options », repliée : « γs 1,5 · jeu de pose 1 cm · arrondi 0,5 cm »."""
    return f"γ<sub>s</sub> {C.nb(gamma_s)} · jeu de pose {C.nb(jeu)} cm · arrondi {C.txt_arrondi(pas)}"


# ------------------------------------------------------------------ à gauche : un élément VERROUILLÉ (le cadenas)
def verrou_aide(genre: Genre) -> str:
    """La phrase en tête d'un élément verrouillé : « La dalle est verrouillée : ses données se lisent, elles ne se
    modifient pas. Le cadenas, à droite de son nom, la déverrouille. » (la dalle, la poutre : E15)."""
    e_, pron = ("", "le") if genre.masculin else ("e", "la")
    return (f"{genre.le[0].upper()}{genre.le[1:]} est verrouillé{e_} : ses données se lisent, elles ne se modifient pas. "
            f"Le cadenas, à droite de son nom, {pron} déverrouille.")


def verrou_refus(genre: Genre) -> str:
    """Le refus d'une saisie dans un élément verrouillé : « Dalle verrouillée : le cadenas, à droite de son nom, la
    déverrouille »."""
    e_, pron = ("", "le") if genre.masculin else ("e", "la")
    nom = genre.le.split(" ", 1)[1]
    return f"{nom[0].upper()}{nom[1:]} verrouillé{e_} : le cadenas, à droite de son nom, {pron} déverrouille"


def lecture(label: str, valeur: str, sym: str = "") -> str:
    """Une donnée d'un élément verrouillé, en lecture seule (``valeur`` : du HTML du module)."""
    return C.ligne(label, f'<span class="dl-ro">{valeur}</span>', sym)


# ------------------------------------------------------------------ à gauche : les sections et la section courante
def carte_section(i: int, nom: str, copier: Sequence, supprimer: Sequence, seule: bool, corps: str, *,
                  copier_txt: str = "Copier la section") -> str:
    """La carte d'une section (rubrique « Sections et sollicitations ») : le bouton qui la rend COURANTE (dessin et
    résultats), son nom, son menu « ⋯ » (copier, supprimer — jamais la dernière), puis ses efforts (``corps``, HTML du
    module). ``copier``, ``supprimer`` : les chemins que le module lit (« sections/0/dup », « sections/0/- »…) ;
    ``copier_txt`` : le libellé de la copie (les dalles : « Dupliquer la section », 13/11/2026) — sans lui, la sortie
    est la même, à l'octet près."""
    actions = (action(f'data-chemin="{C._attr_chemin(copier)}" data-valeur="1"', "copy", copier_txt)
               + action(f'data-chemin="{C._attr_chemin(supprimer)}" data-valeur="1"', "trash-2", "Supprimer la section", True, not seule))
    return (f'<div class="et-sec et-piece" data-cible="s{i}.efforts" data-section="{i}">'
            f'<div class="et-sec-h"><button type="button" class="et-sec-pick" data-section-courante="{i}" '
            f'title="Afficher et vérifier cette section"><span class="o"></span></button>'
            f'{C.texte(["sections", i, "nom"], nom, "nom")}{menu(actions, "Actions de la section")}</div>'
            f'{corps}</div>')


def pied_sections(titre: str, aide: str) -> str:
    """Sous les cartes : « + Section », et à droite ce que les efforts signifient (``aide`` en HTML, ``titre`` au survol)."""
    return (f'<div class="et-pied">{plus(["sections", "+"], "1", "Section")}'
            f'<span class="et-pied-a" title="{C.esc(titre)}">{aide}</span></div>')


def armatures(noms: Sequence[str], contenus: Sequence[str]) -> str:
    """La rubrique « Armatures » : le choix de la section COURANTE (dès deux sections), puis les armatures de chaque
    section (``contenus``, HTML du module), celle de la section courante seule visible (js/compact.js)."""
    onglets = "".join(f'<button type="button" data-section-courante="{i}" class="{"on" if i == 0 else ""}">{C.esc(n)}</button>'
                      for i, n in enumerate(noms))
    blocs = "".join(f'<div class="et-arm{" on" if i == 0 else ""}" data-section="{i}">{c}</div>' for i, c in enumerate(contenus))
    return ((f'<div class="et-secsel"><span>Section</span><div class="seg" role="group" aria-label="Section réglée, dessinée et vérifiée">{onglets}</div></div>'
             if len(noms) > 1 else "") + blocs)

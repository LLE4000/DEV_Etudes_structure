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
from typing import NamedTuple, Optional, Sequence, Tuple

from . import commun as C


class Genre(NamedTuple):
    """Les mots d'un élément de la note, pour les libellés de l'arborescence : (« Poutres », « la poutre »,
    « cette poutre », « une poutre »)."""
    liste: str
    le: str
    ce: str
    un: str


POUTRE = Genre("Poutres", "la poutre", "cette poutre", "une poutre")
DALLE = Genre("Dalles", "la dalle", "cette dalle", "une dalle")
PREDALLE = Genre("Prédalles", "la prédalle", "cette prédalle", "une prédalle")


# ------------------------------------------------------------------ menus et petits boutons
def menu(items: str, titre: str) -> str:
    """Un menu discret « ⋯ » : le bouton, et la liste (cachée) de ses actions."""
    return (f'<div class="et-mn"><button type="button" class="et-mn-b" data-menu="1" aria-haspopup="menu" aria-expanded="false" '
            f'title="{C.esc(titre)}">{C.icon("ellipsis")}</button><div class="et-mn-l" role="menu" hidden>{items}</div></div>')


def action(attrs: str, icone: str, texte: str, danger: bool = False, actif: bool = True) -> str:
    """Une action d'un menu « ⋯ » ; ``attrs`` : ce que js/etabli.js lit (data-chemin + data-valeur, data-renommer…)."""
    return (f'<button type="button" role="menuitem" class="{"danger" if danger else ""}" {attrs}{"" if actif else " disabled"}>'
            f'{C.icon(icone)}{C.esc(texte)}</button>')


def plus(chemin: Sequence, valeur: str, texte: str) -> str:
    """« + Lit », « + Section » : un bouton qui ajoute (le chemin finit par « + »)."""
    return (f'<button type="button" class="et-add" data-chemin="{C._attr_chemin(chemin)}" data-valeur="{C.esc(valeur)}">'
            f'{C.icon("plus")}{C.esc(texte)}</button>')


def retirer(chemin: Sequence, texte: str) -> str:
    """« Retirer ce lit », « Retirer ce renfort » : dans le détail d'une pièce (le chemin finit par « - »)."""
    return (f'<button type="button" class="et-ret" data-chemin="{C._attr_chemin(chemin)}" data-valeur="1">'
            f'{C.icon("trash-2")}{C.esc(texte)}</button>')


# ------------------------------------------------------------------ à gauche : l'arborescence de la note
def arbre(genre: Genre, elements: Sequence[Tuple[str, str, bool]], actif: int, corps: str, marque: str) -> str:
    """La colonne de gauche : l'ARBORESCENCE de la note. Chaque élément ``(nom, dimensions, verrou)`` — le nom en texte,
    les dimensions en HTML (« · 25 × 50 cm ») — s'écrit sans taux ni pastille (le résultat est à droite) ; l'élément
    ouvert (``actif``) se développe et montre ``corps``, ses rubriques ; un clic sur un autre l'ouvre (le premier se
    replie). Renommer, copier, supprimer : dans le menu « ⋯ » de l'élément. ``marque`` : l'attribut qui porte le rang
    (``data-poutre``, ``data-dalle``…)."""
    out = ""
    seul = len(elements) == 1
    for i, (nom_txt, dims, verrou) in enumerate(elements):
        on = i == actif
        nom = C.esc(nom_txt)
        actions = (action(f'data-renommer="{i}"', "pencil", "Renommer")
                   + action(f'data-dupliquer-index="{i}"', "copy", f"Copier {genre.le}")
                   + action(f'data-supprimer="{i}" data-confirmer="Supprimer {genre.le} « {nom} » de la note ?"', "trash-2",
                            f"Supprimer {genre.le}", True, not seul))
        cadenas = C.icon("lock") if verrou else ""
        dedans = f'<div class="et-pt-corps">{corps}</div>' if on else ""
        out += (f'<div class="et-pt{" on" if on else ""}" data-{marque}="{i}"><div class="et-pt-l">'
                f'<button type="button" class="et-pt-t{" on" if on else ""}" data-activer="{i}" aria-expanded="{"true" if on else "false"}" '
                f'title="{"Replier ou déplier" if on else f"Ouvrir {genre.ce}"}">{C.icon("chevron-right")}'
                f'<span class="t">{cadenas}<span class="nom">{nom}</span><span class="dim">{dims}</span></span></button>'
                f'{menu(actions, f"Actions de {genre.le}")}</div>{dedans}</div>')
    return (f'<div class="et-arbre"><div class="et-arbre-h"><span>{genre.liste}</span><span class="n">{len(elements)}</span></div>'
            f'{out}<button type="button" class="et-pt-add" data-ajouter="1">{C.icon("plus")}Ajouter {genre.un}</button></div>')


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
    """La tête du cadre des résultats : le nom de l'élément ouvert (HTML échappé par le module) et ses dimensions."""
    return f'<div class="et-rp"><b>{nom}</b><span>{dims}</span></div>'


def bloc(i: int, titre: str, cls: str, etat_txt: str, efforts: str, rows: str, pied: str = "") -> str:
    """Le bloc d'UNE section : l'en-tête (son ``titre`` et son état ``etat_txt``, en texte ; la couleur ``cls`` : ok,
    warn, nok, inc), les efforts (HTML), les vérifications ``rows`` (deux lignes chacune, C.ligne_verif), puis ``pied``
    (ce que le module ajoute : ses avertissements). Le bouton de l'en-tête ouvre « tout le calcul » (fenêtre ``s<i>``)."""
    tete = (f'<div class="et-bloc-h"><button type="button" class="et-bloc-t" data-bloc="{i}" aria-expanded="true" title="Replier ou déplier">'
            f'{C.icon("chevron-down")}<span class="n">{C.esc(titre)}</span><span class="e">{C.esc(etat_txt)}</span></button>'
            f'<button type="button" class="et-tout" data-calcul="s{i}" title="Tout le calcul de la section">{C.icon("file-text")}</button></div>')
    return (f'<section class="et-bloc {cls}" data-section="{i}">{tete}<div class="et-bloc-c">'
            f'<div class="et-bloc-ef">{efforts}</div><div class="els">{rows}</div>{pied}</div></section>')


# ------------------------------------------------------------------ à gauche : les sections et la section courante
def carte_section(i: int, nom: str, copier: Sequence, supprimer: Sequence, seule: bool, corps: str) -> str:
    """La carte d'une section (rubrique « Sections et sollicitations ») : le bouton qui la rend COURANTE (dessin et
    résultats), son nom, son menu « ⋯ » (copier, supprimer — jamais la dernière), puis ses efforts (``corps``, HTML du
    module). ``copier``, ``supprimer`` : les chemins que le module lit (« sections/0/dup », « sections/0/- »…)."""
    actions = (action(f'data-chemin="{C._attr_chemin(copier)}" data-valeur="1"', "copy", "Copier la section")
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

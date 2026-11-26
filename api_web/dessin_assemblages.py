"""Le DESSIN QUI SE MANIPULE des assemblages — la traduction, et elle seule.

Le kit des schémas (``acier/assemblages/poutre_poutre/doubles_cornieres/schemas.py``, partagé par les sept
configurations) sait depuis l'écran Streamlit dire ce que chaque objet du dessin RÈGLE :

- une COTE modifiable porte la clé de sa saisie (``data-key="e1_u"``) ;
- une PIÈCE porte son groupe (``data-group="beamS"``, ``"bolts"``, ``"plate"``…) ;
- une POIGNÉE d'un groupe de boulons porte son action (``data-action="n1_u:+1"``).

Sur le site, ces attributs étaient coupés (« les cotes se saisissent dans les Données »). Le propriétaire, 26/11/2026 :
« avant, je cliquais sur le dessin et je pouvais tout choisir […] réactive ça, c'était assez pratique ». Ce fichier les
traduit dans le vocabulaire COMMUN des dessins qui se manipulent (``site/web/js/plan.js``, le même composant que le
Générateur SAF, aux trois formats) :

- ``data-plan-cote``  = {"chemin", "valeur", "unite"} : un appui ouvre un champ SUR la cote ; la valeur tapée part au
  moteur (« modifier », le chemin du champ des Données : aucune règle nouvelle, R3) ;
- ``data-plan-pk``    = le rôle d'un SÉLECTEUR (profilés, boulons, cornières, tiges) : un appui sur la pièce l'ouvre —
  la liste que le résultat ouvre déjà, chaque ligne calculée par le moteur ;
- ``data-plan-champ`` = le chemin d'un champ des DONNÉES : un appui sur la pièce ouvre sa rubrique (au téléphone,
  l'onglet Données vient devant) et montre ce champ ;
- ``data-plan-clic``  = {"chemin", "valeur"} : une rangée de boulons de plus, ou de moins.

Le dessin est un RACCOURCI, jamais le seul accès (R1) : une cible n'est posée que si son champ EXISTE dans le cadre
Données rendu (``chemins`` : les ``data-chemin`` du formulaire) ou si son sélecteur existe dans la configuration
(``roles``). Ce qui n'a pas de cible redevient inerte (ni curseur, ni survol : un objet qui promet un geste le tient).
Le GLISSER du kit (``data-drag`` : le jeu, le décalage, une tige) n'est pas branché : ses attributs sont retirés.

Rien n'est calculé ici (R3) : on lit la valeur que le moteur a relue (``R.u``) et la clé que le kit a écrite.
"""
import json
import re
from typing import Dict, Iterable, Optional, Sequence, Tuple

from . import commun as C

# ce qu'un appui sur une PIÈCE ouvre : la première cible qui EXISTE dans la configuration, dans l'ordre —
#   ("pk", rôle, clé) : un sélecteur, quand la configuration l'a ET que son champ « v/<clé> » est dans les Données ;
#   ("champ", clé)    : le champ « v/<clé> » des Données ;
#   ("chemin", c)     : un champ par son chemin entier (une ligne d'une table : « v/tiges/{i}/x »).
_PROFIL_P = (("pk", "prof_P", "prof_P"), ("champ", "prof_P"))
PIECES: Dict[str, Tuple[Tuple[str, ...], ...]] = {
    "beamP": _PROFIL_P,
    "col": _PROFIL_P + (("pk", "prof_c", "prof_c"), ("champ", "prof_c"), ("champ", "forme")),
    "beamS": (("pk", "prof_S", "prof_S"), ("champ", "prof_S")),
    "recut": (("champ", "d_nt"), ("champ", "rec_S"), ("champ", "prof_S")),
    "bolts": (("pk", "boulon", "boulon_u"), ("champ", "boulon_u")),
    "cleat": (("pk", "corniere", "corn_u"), ("champ", "corn_u")),
    "plate": (("champ", "tp_u"), ("champ", "bp_u"), ("champ", "hp_u")),
    "weld": (("champ", "a_w"),),
    "weldS": (("champ", "a_S"), ("champ", "fix_S")),
    "weldP": (("champ", "a_P"), ("champ", "fix_P")),
    "efforts": (("champ", "V_Ed"), ("champ", "N_Ed")),
    # l'étiquette « n1 × n2 » d'un groupe de boulons : ses rangées, ses entraxes, ses pinces
    "G": (("champ", "n1_u"),),
    "S": (("champ", "n1S_u"),),
    "P": (("champ", "n1P_u"),),
    # le pied de poteau
    "pedestal": (("champ", "bc_u"), ("champ", "beton")),
    "grout": (("champ", "t_g"),),
    "anchors": (("pk", "ancr", "ancr"), ("champ", "ancr")),
}
# … et ses pièces numérotées (la platine à la main) : la tige i, l'élément soudé i, le raidisseur i
PIECES_NUMEROTEES = (
    (re.compile(r"^bolt(\d+)$"), (("chemin", "v/tiges/{i}/x"), ("pk", "ancr", "ancr"), ("champ", "ancr"))),
    (re.compile(r"^elem(\d+)$"), (("chemin", "v/elems/{i}/type"),)),
    (re.compile(r"^raid(\d+)$"), (("chemin", "v/raid/{i}/x"),)),
)
# ce que l'on dit de chaque geste (aria-label et bulle : le doigt n'a pas de survol, le lecteur d'écran non plus)
DIRE_PK = "Choisir dans la liste"
DIRE_CHAMP = "Ouvrir ses données"
DIRE_COTE = "Modifier {sym}"

_BALISE = re.compile(r'<g((?: [a-zA-Z0-9:_-]+="[^"]*")+)>')
_ATTRIBUT = re.compile(r' ([a-zA-Z0-9:_-]+)="([^"]*)"')
# les attributs du kit qui disent « je me manipule » ; ceux du GLISSER sont toujours retirés (non branché)
_GLISSER = ("data-drag", "data-dsym", "data-drag2", "data-dsym2")
_GESTE = ("tabindex", "role", "aria-label")


def _attr(d: Dict) -> str:
    return C.esc(json.dumps(d, ensure_ascii=False, separators=(",", ":")))


def _nombre(v) -> Optional[float]:
    """La valeur d'une cote, en nombre fini ; None sinon (une saisie vide, un texte : la cote ne s'ouvre pas)."""
    if isinstance(v, bool) or v is None:
        return None
    try:
        x = float(str(v).replace(",", "."))
    except (TypeError, ValueError):
        return None
    return x if x == x and abs(x) != float("inf") else None


def cible_piece(groupe: str, chemins: Iterable[str], roles: Iterable[str]) -> Optional[Tuple[str, str]]:
    """Ce qu'un appui sur la pièce ``groupe`` ouvre : ("pk", rôle) ou ("champ", chemin), la première cible de PIECES qui
    existe — un sélecteur de la configuration dont le champ est dans les Données, un champ du cadre Données rendu.
    None : la pièce reste inerte."""
    essais = PIECES.get(groupe)
    i = None
    if essais is None:
        for motif, liste in PIECES_NUMEROTEES:
            m = motif.match(groupe)
            if m:
                essais, i = liste, m.group(1)
                break
    for essai in essais or ():
        genre = essai[0]
        if genre == "pk":
            if essai[1] in roles and f"v/{essai[2]}" in chemins:
                return "pk", essai[1]
            continue
        chemin = essai[1].format(i=i) if genre == "chemin" else f"v/{essai[1]}"
        if chemin in chemins:
            return "champ", chemin
    return None


def _sans(classe: str, mot: str) -> str:
    return " ".join(c for c in classe.split() if c != mot)


def manipulable(svg: str, valeurs: Dict, chemins: Iterable[str], roles: Iterable[str], unites: Optional[Dict] = None,
                maxi: Optional[Dict] = None, signees: Iterable[str] = ()) -> str:
    """Le dessin du kit (rendu avec ``interactive`` et ``poignees``) dans le vocabulaire de ``js/plan.js``.

    ``valeurs`` : les saisies que le moteur a relues (``R.u``) ; ``chemins`` : les ``data-chemin`` du cadre Données ;
    ``roles`` : les sélecteurs de la configuration ; ``unites`` : {clé: unité} des cotes ; ``maxi`` : {clé: le plus
    grand nombre de rangées admis} ; ``signees`` : les cotes qui peuvent être négatives (un décalage)."""
    chemins, roles, signees = frozenset(chemins), frozenset(roles), frozenset(signees)
    unites, maxi = unites or {}, maxi or {}

    def balise(m):
        attrs = dict(_ATTRIBUT.findall(m.group(1)))
        if not any(k in attrs for k in ("data-key", "data-group", "data-action") + _GLISSER):
            return m.group(0)
        for k in _GLISSER:
            attrs.pop(k, None)
        actif = False
        cle = attrs.get("data-key")
        if cle is not None:
            v = _nombre(valeurs.get(cle))
            if f"v/{cle}" in chemins and v is not None:
                cote = {"chemin": f"v/{cle}", "valeur": v, "unite": unites.get(cle, "")}
                if cle in signees:
                    cote["signe"] = True
                attrs["data-plan-cote"] = _attr(cote)
                attrs["aria-label"] = DIRE_COTE.format(sym=attrs.get("data-sym") or cle)   # (déjà échappé par le kit)
                actif = True
            else:
                attrs.pop("data-key")
                attrs.pop("data-sym", None)
        groupe = attrs.get("data-group")
        if groupe is not None:
            c = cible_piece(groupe, chemins, roles)
            if c is not None:
                genre, x = c
                attrs["data-plan-pk" if genre == "pk" else "data-plan-champ"] = C.esc(x)
                attrs["aria-label"] = DIRE_PK if genre == "pk" else DIRE_CHAMP
                actif = True
            else:
                attrs.pop("data-group")
        action = attrs.get("data-action")
        if action is not None:
            cle_a, _, pas = action.partition(":")
            n = _nombre(valeurs.get(cle_a))
            try:
                suivant = int(n) + int(pas) if n is not None else None
            except ValueError:
                suivant = None
            if suivant is not None and f"v/{cle_a}" in chemins and 1 <= suivant <= maxi.get(cle_a, 50):
                attrs["data-plan-clic"] = _attr({"chemin": f"v/{cle_a}", "valeur": suivant})
                actif = True
            else:
                attrs.pop("data-action")
                attrs["class"] = (_sans(attrs.get("class", ""), "ed") + " off").strip()
        if not actif:
            for k in _GESTE:
                attrs.pop(k, None)
            if "class" in attrs:
                attrs["class"] = _sans(attrs["class"], "ed")
        return "<g" + "".join(f' {k}="{v}"' for k, v in attrs.items()) + ">"

    return _BALISE.sub(balise, svg)


def cibles(svg: str) -> Dict[str, Sequence]:
    """Ce qu'un dessin rendu offre au doigt : {"cotes": [{chemin, valeur, unite}], "pk": [rôles], "champs": [chemins],
    "clics": [{chemin, valeur}]} — le contrat que les tests relisent (chaque cible existe dans le cadre Données)."""
    import html as _html

    def lus(nom):
        return [_html.unescape(x) for x in re.findall(rf' {nom}="([^"]*)"', svg)]

    return {"cotes": [json.loads(x) for x in lus("data-plan-cote")], "pk": lus("data-plan-pk"),
            "champs": lus("data-plan-champ"), "clics": [json.loads(x) for x in lus("data-plan-clic")]}

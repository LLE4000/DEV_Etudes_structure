# -*- coding: utf-8 -*-
"""La RÈGLE DE POSITION des armatures (beton/position.py, décision du propriétaire du 13/11/2026) — commune à la poutre,
à la dalle pleine et à la prédalle — prouvée à la main.

1. Les exemples du propriétaire : « 3 + 0,6 + 1 (jeu) = 4,6 → 5,0 cm » ; une couche posée sur une autre, « façon A »
   (tout depuis le béton, UN arrondi à la fin : 3 + 1 + 1,2 + 0,5 = 5,7 → 6,0, et non 5,0 + 0,6 + 0,5 = 6,1 → 6,5) ;
   une poutre (c + Ø étrier + Ø/2 + jeu).
2. L'arrondi : au pas SUPÉRIEUR (vers l'intérieur : d diminue), un nombre déjà sur le pas ne bouge pas — même quand la
   somme des flottants le manque d'un cheveu (0,4 + 0,8 + 0,3 = 1,5000000000000002) ; pas nul : aucun arrondi.
3. Les valeurs par défaut décidées : jeu de pose 1 cm, arrondi 0,5 cm, 0 entre deux couches ; les pas offerts.
4. Le détail terme à terme que l'écran écrit, et la relecture d'un pas d'arrondi (refusé hors de la liste, jamais
   remplacé en silence).
5. Le jeu de pose d'une note d'avant la règle : 1 cm au moins (poutre, dalle pleine, prédalle, fichier Streamlit).

    python3 tests/test_position.py
"""
import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

from beton import position as P  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:300]}"))


def proche(a, b):
    return abs(a - b) <= 1e-12


print("=== 1. Les exemples du propriétaire")
r = P.sur_parement(3.0, 12)
chk("un Ø12 sur l'enrobage de 3 cm : position réelle 3 + 0,6 = 3,6 cm (le dessin)", proche(r, 3.6), r)
chk("son axe de calcul : 3,6 + 1 (jeu) = 4,6 → 5,0 cm", P.axe_calcul(r, 1.0, 0.5) == 5.0, P.axe_calcul(r, 1.0, 0.5))
dessus = P.dessus(r, 12)
r2 = P.sur_couche(dessus, 10)
chk("un Ø10 posé sur ce Ø12 : dessus 3 + 1,2 = 4,2, position réelle 4,2 + 0,5 = 4,7 cm", proche(dessus, 4.2) and proche(r2, 4.7), (dessus, r2))
chk("« façon A » : 3 + 1 (jeu) + 1,2 + 0,5 = 5,7 → 6,0 cm, UN arrondi à la fin (et non ⌈4,6⌉ + 0,6 + 0,5 = 6,1 → 6,5)",
    P.axe_calcul(r2, 1.0, 0.5) == 6.0 and P.arrondi_haut(P.axe_calcul(r, 1.0, 0.5) + 0.6 + 0.5, 0.5) == 6.5)
rp = P.sur_parement(3.0, 16, 8)
chk("une poutre, étrier Ø8, Ø16 : 3 + 0,8 + 0,8 = 4,6 cm réel ; 4,6 + 1 = 5,6 → 6,0 cm au calcul",
    proche(rp, 4.6) and P.axe_calcul(rp, 1.0, 0.5) == 6.0, (rp, P.axe_calcul(rp, 1.0, 0.5)))
chk("la prédalle de 6 cm, c = 3 : Ø12 de 3,0 à 4,2 cm, Ø10 posé dessus de 4,2 à 5,2 cm — dans la peau (le dessin)",
    proche(r - 0.6, 3.0) and proche(P.dessus(r, 12), 4.2) and proche(r2 - 0.5, 4.2) and proche(P.dessus(r2, 10), 5.2) <= 6.0)

print("=== 2. L'arrondi")
chk("au pas SUPÉRIEUR : 4,6 → 5,0 ; 4,1 → 4,5 ; 5,0 reste 5,0 ; au pas de 1 cm, 4,6 → 5,0 et 5,2 → 6,0",
    [P.arrondi_haut(x, 0.5) for x in (4.6, 4.1, 5.0)] == [5.0, 4.5, 5.0] and [P.arrondi_haut(x, 1.0) for x in (4.6, 5.2)] == [5.0, 6.0])
somme = 0.4 + 0.8 + 0.3
chk(f"une somme de flottants qui manque le pas d'un cheveu (0,4 + 0,8 + 0,3 = {somme!r}) ne passe pas au pas suivant : "
    "1,5 (et non 2,0)", somme != 1.5 and P.arrondi_haut(somme, 0.5) == 1.5, P.arrondi_haut(somme, 0.5))
chk("pas nul : aucun arrondi (4,6 reste 4,6) ; jeu nul : la position réelle arrondie seule",
    P.arrondi_haut(4.6, 0.0) == 4.6 and P.axe_calcul(4.6, 0.0, 0.0) == 4.6 and P.axe_calcul(3.6, 0.0, 0.5) == 4.0)

print("=== 3. Les valeurs par défaut décidées le 13/11/2026")
chk("jeu de pose 1 cm, arrondi 0,5 cm, 0 entre deux couches ; pas offerts : aucun, 0,5 cm, 1 cm",
    (P.JEU_POSE, P.ARRONDI, P.JEU_COUCHES, P.ARRONDIS) == (1.0, 0.5, 0.0, (0.0, 0.5, 1.0)))
chk("la règle par défaut : axe_calcul(3,6) = ⌈3,6 + 1⌉₀,₅ = 5,0", P.axe_calcul(3.6) == 5.0)
chk("un jeu entre couches s'ajoute entre deux couches posées l'une sur l'autre : 4,2 + 0,3 + 0,5 = 5,0",
    proche(P.sur_couche(4.2, 10, 0.3), 5.0))

print("=== 4. Le détail du calcul, la relecture d'un pas")
det = P.detail([("c", 3.0), ("Ø/2", 0.6)], 1.0, 0.5)
chk("le détail : termes c = 3 et Ø/2 = 0,6, réel 3,6, jeu 1, brut 4,6, pas 0,5, axe 5,0",
    [t["symbole"] for t in det["termes"]] == ["c", "Ø/2"] and proche(det["reel"], 3.6) and det["jeu"] == 1.0
    and proche(det["brut"], 4.6) and det["pas"] == 0.5 and det["axe"] == 5.0, det)
chk("un pas relu : 0,5 et 1 et 0 admis (entiers ou flottants)",
    P.lire_arrondi(0.5) == 0.5 and P.lire_arrondi(1) == 1.0 and P.lire_arrondi(0) == 0.0)
refus = []
for x in (0.3, 2, "0,5", None, True):
    try:
        P.lire_arrondi(x)
        refus.append(x)
    except ValueError:
        pass
chk("un pas hors de la liste (0,3 ; 2), un texte, rien, un booléen : refusés avec un message (ValueError)", not refus, refus)

print("=== 5. Le jeu de pose d'une note d'avant la règle")
lus = [P.jeu_note_d_avant(x) for x in (0, 0.0, 0.5, None, 1.0, 2.5)]
chk("relu : 0, 0,5 et absent portés à 1 cm (le jeu du bureau) ; 1 et 2,5 cm gardés (un jeu plus grand est un choix)",
    lus == [1.0, 1.0, 1.0, 1.0, 1.0, 2.5], lus)
chk("une valeur qui n'est pas un nombre (texte, booléen) est rendue telle quelle : la relecture de la note la refuse",
    P.jeu_note_d_avant("abc") == "abc" and P.jeu_note_d_avant(True) is True)

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:240])
sys.exit(1 if KO else 0)

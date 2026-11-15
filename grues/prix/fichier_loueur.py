# -*- coding: utf-8 -*-
"""Le fichier Excel « Mise à jour des prix » d'UN loueur (cahier § 7.1, § 11.5) : ``DEMANDE_PRIX_{ANNÉE}_{LOUEUR}.xlsx``.

Le CONTENU et la MÉCANIQUE sont ceux des modèles joints au cahier (feuilles, textes, identifiants, clés, valeurs,
listes déroulantes, protection, compteurs, sens des couleurs) ; la MISE EN FORME est refaite (§ 11.5) : les couleurs du
site, un en-tête en clair, de l'air entre les blocs, des traits fins, des cases à remplir en teinte douce à bord net, les
prix manquants bien visibles, les nombres à droite.

Ce que le fichier porte, et ce qu'il ne porte JAMAIS :
- un loueur ne voit que SES prix : les champs et postes « loueur du catalogue » de ses catalogues, plus les postes et
  champs qu'il fournit pour les grues d'un AUTRE loueur (feuille OPTIONS, clé « prix:* » : le catalogue n'est pas nommé) ;
- jamais un prix d'un autre loueur, jamais une donnée interne (fondations, kWh, forfait électricité, renforcement,
  réglages, prix réels de chantier), jamais nos remarques ni nos corrections (la valeur envoyée est celle du loueur) ;
- colonne A cachée = l'identifiant de la ligne (G001…, P:CODE, NOUVEAU, AUTRE-xx, L:…, Q-…) ; une ligne cachée « #cle »
  donne la clé de chaque colonne (une clé « _… » n'est pas relue) : le fichier se RELIT par identifiant et clé
  (``lecture_loueur``), jamais par position ;
- les valeurs envoyées sont recopiées 40 colonnes plus loin (cachées) : une case modifiée passe en vert, et « À LIRE »
  compte en direct les cases modifiées et les prix encore à compléter.

    preparer(base, nom_loueur, annee, demande_par, retour_avant, aujourdhui) -> (nom_du_fichier, octets)"""
import datetime as dt
import re
from typing import Dict, List, Optional, Tuple

from . import base as B
from . import xlsx as X

LOUEUR_DU_CATALOGUE = "loueur du catalogue"
FORMAT = "demande_prix_grues"
VERSION_MODELE = 1
DECALAGE = 40       # les copies cachées des valeurs envoyées
PREMIERE = 7        # la première ligne de données d'une feuille de tableau
LIGNE_CLES = 6      # la ligne « #cle » (cachée)
GRUTIERS = ("GRUTIER", "HS", "TELESC_GRUT", "TELESC_HS")
UNITES_NUMERIQUES = ("kVA", "kWh/mois", "m", "t", "h")

FEUILLE_DU_TYPE = {"Grue tour": "GRUES TOUR", "Grue à montage rapide": "MONTAGE RAPIDE", "Grue mobile": "GRUES MOBILES"}
TYPE_COURT = {"Grue tour": "Grue tour", "Grue à montage rapide": "Montage rapide", "Grue mobile": "Grue mobile"}
LARGEUR = {"identité": 34, "info": 13, "prix": 12.5, "unité": 11, "technique": 15, "parc": 11, "texte": 34}
FORMAT_NOMBRE = {"€/J": "#,##0.00", "€/S": "#,##0.00", "€/M": "#,##0.00", "€/h": "#,##0.00", "FF": "#,##0",
                 "selon unité": "#,##0.00", "kVA": "#,##0", "kWh/mois": "#,##0", "coef": "0.00", "m": "0.0", "t": "0.0",
                 "h": "0", "%": "0.00", "€/trajet": "#,##0.00"}
MOTS_PRIX = ("compris", "non disponible", "sur demande")

# ------------------------------------------------------------------ l'identité du site (site/web/css/charte.css)
ENCRE, ENCRE_2, ENCRE_3 = "0D1525", "4A5568", "8B95A7"
ACCENT, ACCENT_DOUX, ACCENT_TRAIT = "2B4BCF", "EEF1FE", "CDD6FD"
TRAIT, TRAIT_CLAIR, FOND_INFO, FOND_NA = "D3D8E0", "E5E8ED", "F4F5F8", "ECEEF2"
JAUNE, JAUNE_BORD = "FFF6D6", "E2C25C"          # à vérifier, à remplir
ORANGE, ORANGE_BORD, ORANGE_ENCRE = "FDDCC6", "E08A50", "9A3412"    # le prix qui nous manque
VERT, VERT_ENCRE = "DCFCE7", "166534"           # modifié par le loueur
ROUGE = "B91C1C"


def _s(**k) -> Dict:
    return k


TITRE = _s(taille=15, gras=True, couleur=ENCRE, v="center")
SOUS_TITRE = _s(taille=9, couleur=ENCRE_2, retour=True, v="center")
NOTE = _s(taille=9, italique=True, couleur=ENCRE_3, v="center")
GROUPE = _s(taille=9, gras=True, couleur=ACCENT, h="center", bas=ACCENT_TRAIT, bas_epais=True)
ENTETE = _s(taille=9, gras=True, couleur=ENCRE, fond=ACCENT_DOUX, bordure=ACCENT_TRAIT, h="center", retour=True)
SEPARATEUR = _s(taille=9, gras=True, couleur=ACCENT, v="bottom", bas=ACCENT_TRAIT)


def _case(sorte: str, h: Optional[str] = None, format_: Optional[str] = None, **k) -> Dict:
    """Une case de tableau : « info » (grise, verrouillée), « jaune » (à vérifier / à remplir), « orange » (le prix qui
    nous manque, en gras), « na » (ne s'applique pas)."""
    if sorte == "info":
        s = _s(taille=9, couleur=ENCRE_2, fond=FOND_INFO, bordure=TRAIT_CLAIR)
    elif sorte == "orange":
        s = _s(taille=9, gras=True, couleur=ORANGE_ENCRE, fond=ORANGE, bordure=ORANGE_BORD, libre=True)
    elif sorte == "na":
        s = _s(taille=9, couleur=ENCRE_3, fond=FOND_NA, bordure=TRAIT_CLAIR)
    else:
        s = _s(taille=9, couleur=ENCRE, fond=JAUNE, bordure=JAUNE_BORD, libre=True)
    s.update(h=h, retour=True, format=format_, **k)
    return s


def affiche(v, categorie: str):
    """La valeur écrite dans le fichier : 0 → « compris » pour un prix ; None → vide."""
    if v is None:
        return None
    if categorie == "prix" and isinstance(v, (int, float)) and not isinstance(v, bool) and v == 0:
        return "compris"
    return v


def nom_du_fichier(annee: int, nom: str) -> str:
    n = re.sub(r'[\\/:*?"<>|]+', "", str(nom)).strip().upper().replace(" ", "_")
    return f"DEMANDE_PRIX_{annee}_{n}.xlsx"


def annee_par_defaut(aujourdhui: dt.date) -> int:
    """§ 7.1 : l'année suivante à partir de novembre, sinon l'année en cours."""
    return aujourdhui.year + 1 if aujourdhui.month >= 11 else aujourdhui.year


# ------------------------------------------------------------------ le fichier
class _Fichier:
    def __init__(self, base: Dict, loueur: Dict, annee: int):
        self.base, self.loueur, self.annee = base, loueur, int(annee)
        self.nom = loueur.get("nom") or ""
        self.cats = [c for c in loueur.get("catalogues") or [] if B.catalogue(base, c)]
        self.c = X.Classeur(f"Mise à jour des prix {self.annee} – {self.nom}")
        self.code_de = {c["nom"]: c.get("code_calcul") for c in B.catalogues(base)}

    # ---------------------------------------------------------------- outils
    def type_de(self, cas: str) -> str:
        return (B.catalogue(self.base, cas) or {}).get("type") or ""

    def grues(self, cas: str) -> List[Dict]:
        return B.grues_du(self.base, cas)

    def a_des_grues(self, cas: str) -> bool:
        return bool(B.grues_du(self.base, cas, retirees=True))

    def sans_nom(self, texte: str) -> str:
        """Un libellé sans le nom du loueur lui-même (« Châssis hors standard – {loueur} » → « Châssis hors standard »)."""
        t = str(texte or "")
        for sep in (" – ", " - ", " "):
            t = t.replace(sep + self.nom, "")
        return t

    def note_annees(self, cas_liste) -> str:
        parts = []
        for cas in cas_liste:
            ys = sorted({g.get("annee_tarif") for g in self.grues(cas)}, key=lambda y: (y is None, y or 0))
            typ = self.type_de(cas)
            if not ys:
                parts.append(f"{typ} : pas encore de liste chez nous")
            elif ys == [None]:
                parts.append(f"{typ} : année non indiquée dans nos fichiers")
            else:
                parts.append(f"{typ} : tarif " + ", ".join(str(y) for y in ys if y))
        return " · ".join(parts)

    def entete(self, f: X.Feuille, titre: str, sous: str, note: str, ncol: int):
        f.hauteur(1, 30)
        f.fusionner(1, 2, 1, ncol)
        f.ecrire(1, 2, titre, TITRE)
        f.fusionner(2, 2, 2, ncol)
        f.ecrire(2, 2, sous, SOUS_TITRE)
        f.hauteur(2, 30)
        f.fusionner(3, 2, 3, ncol)
        f.ecrire(3, 2, note, NOTE)
        f.hauteur(3, 18)
        f.quadrillage = False

    def groupes(self, f: X.Feuille, groupes: List[Tuple[int, str]], ligne: int = 4):
        i = 0
        while i < len(groupes):
            j = i
            while j + 1 < len(groupes) and groupes[j + 1][1] == groupes[i][1]:
                j += 1
            c1, c2, lib = groupes[i][0], groupes[j][0], groupes[i][1]
            if c2 > c1:
                f.fusionner(ligne, c1, ligne, c2)
            for c in range(c1, c2 + 1):
                f.ecrire(ligne, c, (lib or None) if c == c1 else None, GROUPE if lib else None)
            i = j + 1
        f.hauteur(ligne, 20)

    def cles(self, f: X.Feuille, cles: List[Optional[str]], ligne: int = LIGNE_CLES):
        f.ecrire(ligne, 1, "#cle")
        for i, k in enumerate(cles):
            if k:
                f.ecrire(ligne, 2 + i, k)
        f.cacher_ligne(ligne)

    def copie(self, f: X.Feuille, l: int, c: int, v):
        f.ecrire(l, c + DECALAGE, v)

    def vert(self, f: X.Feuille, c1: int, l1: int, c2: int, l2: int):
        """Une case modifiée (différente de sa copie cachée) devient verte."""
        f.condition(f"{X.ref(l1, c1)}:{X.ref(l2, c2)}", f"{X.ref(l1, c1)}<>{X.ref(l1, c1 + DECALAGE)}", VERT, VERT_ENCRE)

    def cacher_copies(self, f: X.Feuille, c1: int, c2: int):
        for c in range(c1 + DECALAGE, c2 + DECALAGE + 1):
            f.cacher_colonne(c)

    @staticmethod
    def v_prix(f: X.Feuille, plage: str):
        a = plage.split(":")[0]
        f.validation(plage, "custom", f'OR(ISNUMBER({a}),{a}="compris",{a}="non disponible",{a}="sur demande")',
                     "Valeur non reconnue", "Écrivez un prix (nombre), ou : compris, non disponible, sur demande.")

    @staticmethod
    def v_liste(f: X.Feuille, plage: str, items: List[str]):
        f.validation(plage, "list", '"' + ",".join(items) + '"', "Valeur non prévue", "Choisissez dans la liste.")

    @staticmethod
    def v_entier(f: X.Feuille, plage: str):
        f.validation(plage, "whole", "0", "Nombre entier", "Écrivez un nombre entier (0, 1, 2…).", "greaterThanOrEqual")

    @staticmethod
    def v_date(f: X.Feuille, plage: str):
        f.validation(plage, "date", "43831", "Date", "Écrivez une date (jj/mm/aaaa).", "greaterThan")

    def finir(self, f: X.Feuille, paysage: bool = True, titres: Optional[str] = None, fige: Optional[str] = None):
        f.paysage = paysage
        f.titres = titres
        f.fige = fige
        f.protegee = True

    # ---------------------------------------------------------------- une feuille de grues (un catalogue)
    def feuille_grues(self, cas: str, n_neuves: int) -> Dict:
        cat = B.catalogue(self.base, cas)
        lg = cat.get("code_calcul")
        titre = FEUILLE_DU_TYPE.get(cat.get("type"), str(cat.get("type") or cas).upper()[:31])
        f = self.c.feuille(titre)
        cols = [{"cle": "modele", "libelle": "Modèle", "unite": "", "categorie": "identité", "essentiel": False, "groupe": "Grue"}]
        if lg == "T":
            cols.append({"cle": "_variante", "libelle": "Variante", "unite": "", "categorie": "info", "essentiel": False, "groupe": "Grue"})
        cols += [c for c in B.champs(self.base, lg) if c.get("fourni_par") == LOUEUR_DU_CATALOGUE and c.get("cle") != "modele"]
        ncol = 1 + len(cols)
        lignes = self.grues(cas)
        consigne = ("Prix HTVA. Corrigez les cases jaunes si besoin, complétez les cases orange. "
                    "« non disponible » si l'option n'existe pas pour ce modèle. "
                    "« Plus proposée » : choisissez Oui si vous ne louez plus ce modèle.")
        if lg == "MA":
            consigne = ("Prix HTVA, grutier compris. Corrigez les cases jaunes si besoin, complétez les cases orange. "
                        "« sur demande » si vous ne donnez pas de prix fixe. « Plus proposée » : Oui si vous ne louez plus ce modèle.")
        self.entete(f, f"{titre} — {self.nom} — prix {self.annee}", consigne, "Prix actuels : " + self.note_annees([cas]) + ".", ncol)
        self.groupes(f, [(2 + i, c.get("groupe") or "") for i, c in enumerate(cols)])
        for i, c in enumerate(cols):
            lib, u = c.get("libelle") or c["cle"], c.get("unite") or ""
            if c["cle"] == "nb_disponibles":
                lib = f"Disponibles en {self.annee}"
            if c["cle"] == "telecommande_loueur_prix":
                u = "prix"
            if c["cle"] == "telecommande_loueur_unite":
                lib, u = "Unité", "€/J, €/S, €/M"
            if u and u != "Oui":
                lib += f"\n({u})"
            f.ecrire(5, 2 + i, lib, ENTETE)
            f.largeur(2 + i, 14 if c["cle"] == "type_capacite" else LARGEUR.get(c.get("categorie"), 12))
        f.hauteur(5, 42)
        self.cles(f, [c["cle"] for c in cols])
        f.cacher_colonne(1)
        orange: List[str] = []

        def ligne(l: int, g: Optional[Dict]):
            neuve = g is None
            f.ecrire(l, 1, "NOUVEAU" if neuve else g.get("id"))
            for i, c in enumerate(cols):
                col, k, cat_ = 2 + i, c["cle"], c.get("categorie")
                if neuve:
                    v = None
                elif k == "_variante":
                    v = (f"HSC ≤ {g['hsc_variante_max_m']} m" if g.get("hsc_variante_max_m") else
                         f"HSC > {g['hsc_variante_min_m']} m" if g.get("hsc_variante_min_m") else None)
                elif k == "remarques_loueur":
                    v = None        # nos remarques internes ne sont jamais envoyées
                else:
                    v = affiche(g.get(k), cat_)
                fige_ = cat_ == "info" or (not neuve and cat_ == "identité")
                num_tech = cat_ == "technique" and c.get("unite") in UNITES_NUMERIQUES
                h = "left" if (cat_ in ("identité", "texte", "info") or (cat_ == "technique" and not num_tech)) else \
                    ("center" if cat_ in ("unité",) or k == "retiree" else "right")
                fmt = FORMAT_NOMBRE.get(c.get("unite")) if cat_ in ("prix", "technique") else ("0" if cat_ == "parc" else None)
                if fige_:
                    sorte = "info"
                elif not neuve and v is None and c.get("essentiel"):
                    sorte = "orange"
                    orange.append(X.ref(l, col))
                else:
                    sorte = "jaune"
                f.ecrire(l, col, v, _case(sorte, h, fmt))
                self.copie(f, l, col, v)
            f.hauteur(l, 18)

        l = PREMIERE
        for g in lignes:
            ligne(l, g)
            l += 1
        derniere_connue = l - 1
        sep = "Autres modèles que vous proposez — une ligne par modèle" if lignes else "Vos modèles — une ligne par modèle"
        f.fusionner(l, 2, l, ncol)
        f.ecrire(l, 2, sep, SEPARATEUR)
        self.copie(f, l, 2, sep)
        f.hauteur(l, 28)
        l += 1
        premiere_neuve = l
        for _ in range(n_neuves):
            ligne(l, None)
            l += 1
        derniere = l - 1
        for i, c in enumerate(cols):
            col = 2 + i
            plages = ([(PREMIERE, derniere_connue)] if lignes else []) + [(premiere_neuve, derniere)]
            for a, b in plages:
                sq = f"{X.ref(a, col)}:{X.ref(b, col)}"
                if c.get("categorie") == "prix":
                    self.v_prix(f, sq)
                elif c["cle"] == "telecommande_loueur_unite":
                    self.v_liste(f, sq, ["€/J", "€/S", "€/M"])
                elif c["cle"] == "retiree":
                    self.v_liste(f, sq, ["Oui", "Non"])
                elif c.get("categorie") == "parc":
                    self.v_entier(f, sq)
        if lignes:
            self.vert(f, 2, PREMIERE, ncol, derniere_connue)
        self.vert(f, 2, premiere_neuve, ncol, derniere)
        self.cacher_copies(f, 2, ncol)
        self.finir(f, True, "$4:$5", f"C{PREMIERE}")
        desc = {"Grue tour": "vos grues tour : location, montage, conduite, châssis, électricité, parc",
                "Grue à montage rapide": "vos grues à montage rapide : location, montage, conduite, électricité, parc",
                "Grue mobile": ("vos grues mobiles, grutier compris : prix horaire, déplacement, ballast"
                                if lg == "MA" else "vos grues mobiles : location, transport et montage, parc")}.get(cat.get("type"), "vos grues")
        return {"feuille": titre, "cas": cas, "lignes": len(lignes), "plage": (2, PREMIERE, ncol, derniere), "orange": orange,
                "desc": desc}

    # ---------------------------------------------------------------- accessoires, grutiers (un poste par ligne)
    def feuille_postes(self, titre: str, sous: str, codes: List[str]) -> Dict:
        f = self.c.feuille(titre)
        ncol = 3 + len(self.cats) + 1
        self.entete(f, f"{titre} — {self.nom} — prix {self.annee}", sous, "Prix actuels : " + self.note_annees(self.cats) + ".", ncol)
        self.groupes(f, [(2, ""), (3, "")] + [(4 + i, "Prix HTVA") for i in range(len(self.cats))] + [(ncol, "")])
        tetes = ["Poste", "Unité"] + [TYPE_COURT.get(self.type_de(c), self.type_de(c)) for c in self.cats] + ["Remarques"]
        larg = [44, 11] + [14] * len(self.cats) + [36]
        for i, t in enumerate(tetes):
            f.ecrire(5, 2 + i, t, ENTETE)
            f.largeur(2 + i, larg[i])
        f.hauteur(5, 30)
        self.cles(f, ["_libelle", "unite"] + [f"prix:{c}" for c in self.cats] + ["remarque"])
        f.cacher_colonne(1)
        orange: List[str] = []
        par_logique = self.base.get("postes_par_logique") or {}
        l = PREMIERE
        for code in codes:
            p = B.poste(self.base, code)
            f.ecrire(l, 1, f"P:{code}")
            f.ecrire(l, 2, p.get("poste"), _case("info", "left"))
            self.copie(f, l, 2, p.get("poste"))
            u_libre = bool(p.get("unite_a_confirmer"))
            f.ecrire(l, 3, p.get("unite"), _case("jaune" if u_libre else "info", "center"))
            self.copie(f, l, 3, p.get("unite"))
            if u_libre:
                self.v_liste(f, f"C{l}", ["€/J", "€/h", "€/S", "€/M", "FF"])
            ppc = p.get("prix_par_cas") or {}
            for i, cas in enumerate(self.cats):
                col = 4 + i
                applicable = cas in ppc or (not self.a_des_grues(cas) and code in (par_logique.get(self.code_de.get(cas)) or []))
                if not applicable:
                    f.ecrire(l, col, "—", _case("na", "center"))
                    self.copie(f, l, col, "—")
                    continue
                v = affiche(ppc.get(cas), "prix")
                manque = v is None and self.a_des_grues(cas)
                if manque:
                    orange.append(X.ref(l, col))
                f.ecrire(l, col, v, _case("orange" if manque else "jaune", "right", "0.00" if p.get("unite") == "coef" else "#,##0.00"))
                self.copie(f, l, col, v)
                self.v_prix(f, X.ref(l, col))
            f.ecrire(l, ncol, None, _case("jaune", "left"))
            f.hauteur(l, 18)
            l += 1
        derniere = l - 1
        self.vert(f, 2, PREMIERE, ncol, derniere)
        self.cacher_copies(f, 2, ncol)
        self.finir(f, True, "$4:$5", f"C{PREMIERE}")
        return {"feuille": titre, "lignes": len(codes), "plage": (2, PREMIERE, ncol, derniere), "orange": orange}

    # ---------------------------------------------------------------- options pour les grues d'un autre loueur
    def feuille_options(self) -> Optional[Dict]:
        champs = []
        for lg, liste in (self.base.get("champs_grues") or {}).items():
            for ch in liste or []:
                if isinstance(ch, dict) and ch.get("fourni_par") == self.nom:
                    for c in B.catalogues(self.base):
                        if c.get("code_calcul") == lg and c.get("loueur") != self.nom:
                            champs.append((c["nom"], ch))
        postes = [p for p in B.postes(self.base) if p.get("fourni_par") == self.nom]
        if not champs and not postes:
            return None
        f = self.c.feuille("OPTIONS")
        ncol = 4
        rien = all(v is None for p in postes for v in (p.get("prix_par_cas") or {}).values()) and not champs
        self.entete(f, f"OPTIONS — {self.nom} — prix {self.annee}",
                    "Options que vous nous louez pour des grues tour venant d'un autre loueur. Prix HTVA. "
                    "« non disponible » si vous ne pouvez pas la fournir pour ce modèle.",
                    "Prix actuels : " + ("nous n'avons pas encore vos prix." if rien else "voir chaque ligne."), ncol)
        for i, w in enumerate([44, 16, 40]):
            f.largeur(2 + i, w)
        f.cacher_colonne(1)
        orange: List[str] = []
        blocs = []
        l = 4
        for cas, ch in champs:
            lignes = [g for g in self.grues(cas) if g.get(ch["cle"]) != 0]
            lib = self.sans_nom(ch.get("libelle"))
            f.fusionner(l, 2, l, ncol)
            f.ecrire(l, 2, f"{lib} à louer pour nos grues tour", SEPARATEUR)
            f.hauteur(l, 26)
            l += 1
            for i, t in enumerate(["Modèle de la grue", f"{lib}\n({ch.get('unite')})", "Remarques"]):
                f.ecrire(l, 2 + i, t, ENTETE)
            f.hauteur(l, 30)
            l += 1
            f.ecrire(l, 1, "#cle")
            f.ecrire(l, 2, "_modele")
            f.ecrire(l, 3, ch["cle"])
            f.ecrire(l, 4, "remarque")
            f.cacher_ligne(l)
            l += 1
            b0 = l
            for g in lignes:
                f.ecrire(l, 1, g.get("id"))
                f.ecrire(l, 2, g.get("modele"), _case("info", "left"))
                self.copie(f, l, 2, g.get("modele"))
                v = affiche(g.get(ch["cle"]), "prix")
                f.ecrire(l, 3, v, _case("jaune", "right", "#,##0.00"))
                self.copie(f, l, 3, v)
                f.ecrire(l, 4, None, _case("jaune", "left"))
                l += 1
            if lignes:
                self.v_prix(f, f"C{b0}:C{l - 1}")
                blocs.append((2, b0, ncol, l - 1))
            l += 1
        if postes:
            f.fusionner(l, 2, l, ncol)
            f.ecrire(l, 2, "Châssis", SEPARATEUR)
            f.hauteur(l, 26)
            l += 1
            for i, t in enumerate(["Poste", "Prix HTVA", "Remarques"]):
                f.ecrire(l, 2 + i, t, ENTETE)
            f.hauteur(l, 22)
            l += 1
            for p in postes:
                ppc = p.get("prix_par_cas") or {}
                cas = next(iter(ppc), None)
                f.ecrire(l, 1, "#cle")
                f.ecrire(l, 2, "_libelle")
                f.ecrire(l, 3, "prix:*")      # tous les catalogues où ce loueur fournit ce poste, sans les nommer
                f.ecrire(l, 4, "remarque")
                f.cacher_ligne(l)
                l += 1
                f.ecrire(l, 1, f"P:{p.get('code')}")
                lib = self.sans_nom(p.get("poste")) + f" ({p.get('unite')})"
                f.ecrire(l, 2, lib, _case("info", "left"))
                self.copie(f, l, 2, lib)
                v = affiche(ppc.get(cas), "prix") if cas is not None else None
                f.ecrire(l, 3, v, _case("orange" if v is None else "jaune", "right", "#,##0.00"))
                if v is None:
                    orange.append(f"C{l}")
                self.copie(f, l, 3, v)
                self.v_prix(f, f"C{l}")
                f.ecrire(l, 4, None, _case("jaune", "left"))
                f.hauteur(l, 18)
                blocs.append((2, l, ncol, l))
                l += 1
        for c1, r1, c2, r2 in blocs:
            self.vert(f, c1, r1, c2, r2)
        self.cacher_copies(f, 2, ncol)
        self.finir(f, False)
        return {"feuille": "OPTIONS", "lignes": sum(b[3] - b[1] + 1 for b in blocs), "blocs": blocs, "orange": orange,
                "desc": "options que vous louez pour nos autres grues tour (télécommande, châssis)"}

    # ---------------------------------------------------------------- autres prestations
    def feuille_autres(self, n: int = 15) -> Dict:
        f = self.c.feuille("AUTRES")
        ncol = 6
        self.entete(f, f"AUTRES PRESTATIONS — {self.nom} — prix {self.annee}",
                    "Tout ce que vous proposez et qui n'est pas dans les autres feuilles : options, accessoires, services. Prix HTVA.",
                    "Une ligne par prestation.", ncol)
        for i, (t, w) in enumerate(zip(["Prestation", "Pour quelle grue", "Unité", "Prix HTVA", "Remarques"], [44, 18, 10, 13, 40])):
            f.ecrire(5, 2 + i, t, ENTETE)
            f.largeur(2 + i, w)
        f.hauteur(5, 24)
        self.cles(f, ["libelle", "type", "unite", "prix", "remarque"])
        f.cacher_colonne(1)
        for k in range(n):
            l = PREMIERE + k
            f.ecrire(l, 1, f"AUTRE-{k + 1:02d}")
            for c in range(2, 7):
                f.ecrire(l, c, None, _case("jaune", "right" if c == 5 else ("center" if c in (3, 4) else "left"),
                                           "#,##0.00" if c == 5 else None))
            f.hauteur(l, 18)
        derniere = PREMIERE + n - 1
        types = sorted({TYPE_COURT.get(self.type_de(c), self.type_de(c)) for c in self.cats})
        self.v_liste(f, f"C{PREMIERE}:C{derniere}", types + ["Toutes"])
        self.v_liste(f, f"D{PREMIERE}:D{derniere}", ["€/J", "€/S", "€/M", "€/h", "FF"])
        self.v_prix(f, f"E{PREMIERE}:E{derniere}")
        self.finir(f, True, "$5:$5", f"C{PREMIERE}")
        return {"feuille": "AUTRES", "premiere": PREMIERE, "derniere": derniere}

    # ---------------------------------------------------------------- questions
    def questions(self) -> List[Dict]:
        """Les questions qui partent : celles du loueur qui ne sont pas « traitées »."""
        return [q for q in self.loueur.get("questions") or [] if isinstance(q, dict) and q.get("id") and not q.get("traitee")]

    def feuille_questions(self) -> Dict:
        f = self.c.feuille("QUESTIONS")
        ncol = 5
        qs = self.questions()
        self.entete(f, f"QUESTIONS — {self.nom}", f"{len(qs)} questions courtes pour lever nos doutes. Une réponse en quelques mots suffit.",
                    "La dernière ligne est libre pour toute autre information utile.", ncol)
        for i, (t, w) in enumerate(zip(["N°", "Question", "Ce que nous avons aujourd'hui", "Votre réponse"], [5, 52, 36, 52])):
            f.ecrire(5, 2 + i, t, ENTETE)
            f.largeur(2 + i, w)
        f.hauteur(5, 24)
        self.cles(f, ["_n", "_question", "_info", "reponse"])
        f.cacher_colonne(1)
        l = PREMIERE
        for i, q in enumerate(qs + [{"id": "REMARQUE", "question": "Autre remarque ou information utile", "info_actuelle": ""}]):
            f.ecrire(l, 1, q["id"])
            f.ecrire(l, 2, i + 1 if q["id"] != "REMARQUE" else "+", _case("info", "center", gras=True))
            f.ecrire(l, 3, q.get("question"), _case("info", "left"))
            f.ecrire(l, 4, q.get("info_actuelle"), _case("info", "left", italique=True))
            f.ecrire(l, 5, None, _case("jaune", "left"))
            f.hauteur(l, 36)
            l += 1
        self.finir(f, True, "$5:$5")
        return {"feuille": "QUESTIONS", "premiere": PREMIERE, "derniere": l - 1, "n": len(qs)}

    # ---------------------------------------------------------------- À LIRE
    def a_lire(self, f: X.Feuille, infos: List[Dict], autres: Dict, questions: Dict, a_des_grues: bool,
               demande_par: Optional[str], retour_avant: Optional[dt.date]):
        f.quadrillage = False
        for c, w in {1: 2, 2: 5, 3: 34, 4: 26, 5: 15, 6: 15, 7: 15, 8: 15}.items():
            f.largeur(c, w)
        f.cacher_colonne(1)
        ncol = 8
        BANDE = _s(taille=11, gras=True, couleur=ACCENT, v="bottom", bas=ACCENT_TRAIT)
        ETIQ = _s(taille=10, gras=True, couleur=ENCRE)

        def bande(l, texte, h=24):
            f.fusionner(l, 2, l, ncol)
            for c in range(2, ncol + 1):
                f.ecrire(l, c, texte if c == 2 else None, BANDE)
            f.hauteur(l, h)

        def para(l, texte, h=30, **k):
            f.fusionner(l, 3, l, ncol)
            f.ecrire(l, 3, texte, _s(taille=k.pop("taille", 10), couleur=ENCRE, v="top", retour=True, **k))
            f.hauteur(l, h)

        f.fusionner(1, 2, 1, ncol)
        f.ecrire(1, 2, f"Mise à jour des prix {self.annee}", _s(taille=20, gras=True, couleur=ENCRE, v="center"))
        f.fusionner(2, 2, 2, ncol)
        for c in range(2, ncol + 1):
            f.ecrire(2, c, self.nom if c == 2 else None, _s(taille=13, gras=True, couleur=ACCENT, v="top", bas=ACCENT, bas_epais=True))
        f.hauteur(1, 36)
        f.hauteur(2, 26)
        # « Demandé par » et « Retour souhaité avant le » : remplis par nous avant l'envoi
        f.ecrire(4, 1, "#cle")
        f.ecrire(4, 3, "_libelle")
        f.ecrire(4, 4, "valeur")
        f.cacher_ligne(4)
        for l, (cle, lib, sorte, v) in zip((5, 6), (("L:demande_par", "Demandé par", "txt", demande_par or None),
                                                     ("L:retour_avant", "Retour souhaité avant le", "date", retour_avant))):
            f.ecrire(l, 1, cle)
            f.ecrire(l, 3, lib, ETIQ)
            f.ecrire(l, 4, v, _s(taille=10, couleur=ENCRE, bordure=TRAIT, libre=True, h="left",
                                 format="DD/MM/YYYY" if sorte == "date" else None))
            if sorte == "date":
                self.v_date(f, f"D{l}")
            f.hauteur(l, 18)
        bande(8, "Pourquoi ce fichier ?")
        if a_des_grues:
            txt = ("Nous chiffrons nos installations de chantier avec ces prix. Ce fichier reprend vos grues et vos prestations, "
                   "avec les prix que nous avons aujourd'hui. Une mise à jour par an suffit : vos grues sont ensuite chiffrées "
                   "chez nous avec vos prix exacts.")
        else:
            txt = ("Nous chiffrons nos installations de chantier avec ces prix. Nous n'avons pas encore votre liste de grues : "
                   "ajoutez-la dans la feuille GRUES TOUR. Une mise à jour par an suffit : vos grues sont ensuite chiffrées "
                   "chez nous avec vos prix exacts.")
        para(9, txt, h=40)
        bande(11, "En 3 étapes")
        etapes = ["Corrigez les cases jaunes si le prix a changé. Une case modifiée devient verte.",
                  "Complétez les cases orange : ce sont les prix qui nous manquent.",
                  "Ajoutez vos autres modèles et prestations dans les lignes vides, et répondez aux questions."]
        if not a_des_grues:
            etapes = ["Ajoutez vos grues dans la feuille GRUES TOUR, une ligne par modèle.",
                      "Complétez vos accessoires, vos grutiers et les cases orange : ce sont les prix qui nous manquent.",
                      "Ajoutez vos autres prestations dans la feuille AUTRES et répondez aux questions."]
        for i, e in enumerate(etapes):
            f.ecrire(12 + i, 2, i + 1, _s(taille=11, gras=True, couleur="FFFFFF", fond=ACCENT, h="center"))
            para(12 + i, e, h=22)
        para(15, "Renvoyez-nous ensuite ce fichier, sans renommer les feuilles. Merci !", h=22, gras=True)
        bande(17, "Dans une case prix, écrivez")
        for i, t in enumerate(["un nombre (€ HTVA)", "compris", "non disponible", "sur demande"]):
            st = _case("jaune", "center")
            st.pop("libre", None)        # un exemple, pas une case à remplir
            f.ecrire(18, 3 + i, t, st)
        f.hauteur(18, 20)
        bande(20, "Couleurs")
        for i, (t, sorte) in enumerate([("à vérifier", "jaune"), ("prix qui nous manque", "orange"), ("modifié par vous", "vert"),
                                        ("information", "info")]):
            st = _s(taille=9, couleur=VERT_ENCRE, fond=VERT, bordure="86EFAC", h="center") if sorte == "vert" else _case(sorte, "center")
            st.pop("libre", None)
            f.ecrire(21, 3 + i, t, st)
        f.hauteur(21, 20)
        bande(23, "Unités")
        para(24, "€/J par jour ouvrable · €/S par semaine · €/M par mois · €/h par heure · FF forfait", h=18, taille=9)
        bande(25, "Prix actuels dans ce fichier")
        para(26, self.note_annees(self.cats) + ".", h=18, taille=9)
        bande(28, "Vos coordonnées")
        f.ecrire(29, 1, "#cle")
        f.ecrire(29, 3, "_libelle")
        f.ecrire(29, 4, "valeur")
        f.cacher_ligne(29)
        l = 30
        f.ecrire(l, 3, "Société", _case("info", "left", gras=True))
        f.ecrire(l, 4, self.nom, _case("info", "left"))
        f.hauteur(l, 18)
        l += 1
        coords = [("L:rempli_par", "Rempli par", "txt"), ("L:mail", "Mail", "txt"), ("L:telephone", "Téléphone", "txt"),
                  ("L:date", "Date", "date"), ("L:valable_du", "Prix valables à partir du", "date"),
                  ("L:valable_au", "Prix valables jusqu'au", "date"),
                  ("L:inchanges_valables", f"Les prix que vous n'avez pas modifiés restent valables en {self.annee}", "oui")]
        for cle, lib, sorte in coords:
            f.ecrire(l, 1, cle)
            f.ecrire(l, 3, lib, _case("info", "left", gras=True))
            f.ecrire(l, 4, None, _case("jaune", "left" if sorte == "txt" else "center", "DD/MM/YYYY" if sorte == "date" else None))
            if sorte == "date":
                self.v_date(f, f"D{l}")
            if sorte == "oui":
                self.v_liste(f, f"D{l}", ["Oui", "Non"])
            f.hauteur(l, 32 if sorte == "oui" else 18)
            l += 1
        l += 1
        bande(l, "Contenu et état du fichier")
        l += 1
        f.fusionner(l, 4, l, 5)
        for c, t in ((3, "Feuille"), (4, "Ce qu'elle contient"), (5, None), (6, "Lignes"), (7, "Cases modifiées"), (8, "Prix à compléter")):
            f.ecrire(l, c, t, ENTETE)
        f.hauteur(l, 30)
        l += 1
        LIGNE = _s(taille=9, couleur=ENCRE, bordure=TRAIT_CLAIR)
        CHIFFRE = _s(taille=9, couleur=ENCRE, bordure=TRAIT_CLAIR, h="center", format="0")
        MANQUE = _s(taille=9, gras=True, couleur=ROUGE, bordure=TRAIT_CLAIR, h="center", format='0;-0;"—"')
        TIRET = _s(taille=9, couleur=ENCRE_3, bordure=TRAIT_CLAIR, h="center")
        for info in infos:
            q = "'" + info["feuille"].replace("'", "''") + "'"

            def diff(c1, r1, c2, r2):
                return (f"SUMPRODUCT(--({q}!{X.ref(r1, c1)}:{X.ref(r2, c2)}<>"
                        f"{q}!{X.ref(r1, c1 + DECALAGE)}:{X.ref(r2, c2 + DECALAGE)}))")
            mods = "+".join(diff(*b) for b in info["blocs"]) if "blocs" in info else diff(*info["plage"])
            manque = "+".join(f"COUNTBLANK({q}!{a})" for a in info["orange"]) or "0"
            f.ecrire(l, 3, info["feuille"], _s(taille=9, gras=True, couleur=ENCRE, bordure=TRAIT_CLAIR))
            f.fusionner(l, 4, l, 5)
            f.ecrire(l, 4, info["desc"], _s(taille=9, couleur=ENCRE_2, bordure=TRAIT_CLAIR, retour=True))
            f.ecrire(l, 5, None, LIGNE)
            f.ecrire(l, 6, info["lignes"], CHIFFRE)
            f.ecrire(l, 7, f"={mods or '0'}", CHIFFRE)
            f.ecrire(l, 8, f"={manque}", MANQUE)
            f.hauteur(l, 30)
            l += 1
        f.ecrire(l, 3, "AUTRES", _s(taille=9, gras=True, couleur=ENCRE, bordure=TRAIT_CLAIR))
        f.fusionner(l, 4, l, 5)
        f.ecrire(l, 4, "tout ce qui n'est pas dans les autres feuilles", _s(taille=9, couleur=ENCRE_2, bordure=TRAIT_CLAIR))
        f.ecrire(l, 5, None, LIGNE)
        f.ecrire(l, 6, f"=COUNTA('AUTRES'!B{autres['premiere']}:B{autres['derniere']})", CHIFFRE)
        f.ecrire(l, 7, "—", TIRET)
        f.ecrire(l, 8, "—", TIRET)
        f.hauteur(l, 22)
        l += 1
        f.ecrire(l, 3, "QUESTIONS", _s(taille=9, gras=True, couleur=ENCRE, bordure=TRAIT_CLAIR))
        f.fusionner(l, 4, l, 5)
        f.ecrire(l, 4, f"{questions['n']} questions courtes", _s(taille=9, couleur=ENCRE_2, bordure=TRAIT_CLAIR))
        f.ecrire(l, 5, None, LIGNE)
        f.ecrire(l, 6, questions["n"], CHIFFRE)
        f.ecrire(l, 7, f"=COUNTA('QUESTIONS'!E{questions['premiere']}:E{questions['derniere']})", CHIFFRE)
        f.ecrire(l, 8, f"=MAX(0,{questions['n']}-COUNTA('QUESTIONS'!E{questions['premiere']}:E{questions['derniere'] - 1}))", MANQUE)
        f.hauteur(l, 22)
        l += 1
        PETIT = _s(taille=8, italique=True, couleur=ENCRE_3, h="center", v="top", retour=True)
        f.ecrire(l, 7, "réponses", PETIT)
        f.ecrire(l, 8, "questions sans réponse", PETIT)
        f.hauteur(l, 24)
        l += 2
        f.fusionner(l, 2, l, ncol)
        f.ecrire(l, 2, "Vos prix restent confidentiels : ils ne servent qu'à nos chiffrages internes.",
                 _s(taille=9, italique=True, couleur=ENCRE_2))
        f.paysage = False
        f.ajuster_hauteur = 1
        f.protegee = True

    # ---------------------------------------------------------------- le classeur
    def fabriquer(self, demande_par: Optional[str], retour_avant: Optional[dt.date], aujourdhui: dt.date) -> bytes:
        lire = self.c.feuille("À LIRE")
        infos = []
        a_des_grues = any(self.a_des_grues(c) for c in self.cats)
        for cas in self.cats:
            infos.append(self.feuille_grues(cas, 5 if self.a_des_grues(cas) else 15))
        tout_ma = bool(self.cats) and all(self.code_de.get(c) == "MA" for c in self.cats)
        par_logique = self.base.get("postes_par_logique") or {}
        acc, gru = [], []
        for p in B.postes(self.base):
            if p.get("fourni_par") != LOUEUR_DU_CATALOGUE:
                continue
            for cas in self.cats:
                if cas in (p.get("prix_par_cas") or {}) or (not self.a_des_grues(cas) and p.get("code") in (par_logique.get(self.code_de.get(cas)) or [])):
                    (gru if p.get("code") in GRUTIERS else acc).append(p.get("code"))
                    break
        if acc:
            sous = ("Accessoires et services, par type de grue. Prix HTVA. « — » : ne s'applique pas à ce type de grue. "
                    "Si l'unité est en jaune, corrigez-la si besoin.")
            if tout_ma:
                sous = "Suppléments et services. Prix HTVA. Si l'unité est en jaune, corrigez-la si besoin."
            info = self.feuille_postes("ACCESSOIRES", sous, acc)
            info["desc"] = ("suppléments horaires, déplacement du grutier, transport du ballast le week-end" if tout_ma
                            else "accessoires et services : anti-collision, caméra, talkies, Greenbox, rails…")
            infos.append(info)
        if gru:
            info = self.feuille_postes("GRUTIERS", "Taux horaires de vos grutiers, par type de grue. Prix HTVA.", gru)
            info["desc"] = "taux horaires des grutiers et heures sup."
            infos.append(info)
        opt = self.feuille_options()
        if opt:
            infos.append(opt)
        autres = self.feuille_autres()
        questions = self.feuille_questions()
        self.a_lire(lire, infos, autres, questions, a_des_grues, demande_par, retour_avant)
        meta = self.c.feuille("_META")
        lignes = [("format", FORMAT), ("version_modele", VERSION_MODELE), ("loueur_id", self.loueur.get("id")), ("loueur", self.nom),
                  ("annee", self.annee), ("genere_le", aujourdhui.isoformat())]
        lignes += [(f"feuille:{i['feuille']}", i["cas"]) for i in infos if i.get("cas")]
        for i, (k, v) in enumerate(lignes):
            meta.ecrire(i + 1, 1, k)
            meta.ecrire(i + 1, 2, v)
        meta.etat = "veryHidden"
        meta.pied = None
        return self.c.octets()


def preparer(base: Dict, nom_loueur: str, annee: int, demande_par: Optional[str] = None, retour_avant=None,
             aujourdhui: Optional[dt.date] = None) -> Tuple[str, bytes]:
    """(nom du fichier, octets du classeur) : la demande de prix ``annee`` au loueur ``nom_loueur``."""
    l = B.loueur(base, nom_loueur)
    if not l:
        raise ValueError("loueur inconnu")
    a = int(annee)
    if not 2000 <= a <= 2100:
        raise ValueError("année illisible")
    from .periode import lire_date
    rv = lire_date(retour_avant) if retour_avant else None
    oct_ = _Fichier(base, l, a).fabriquer((demande_par or "").strip()[:60] or None, rv, aujourdhui or dt.date.today())
    return nom_du_fichier(a, nom_loueur), oct_

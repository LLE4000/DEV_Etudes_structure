# -*- coding: utf-8 -*-
"""L'ENVELOPPE d'une méthode sur toute une barre : pour chaque vérification, l'occurrence la plus DÉFAVORABLE rencontrée
— une combinaison, une section ou un tronçon —, avec les formules de CETTE occurrence ; et le bilan de chaque
combinaison. Le moteur des barres (moteur.py) et celui de l'âme (ame.py) y versent leurs résultats.

Ce qu'elle rend a la forme de acier/barre/resultat.py, où chaque vérification dit sa combinaison et son lieu.
"""
from typing import Dict, List, Optional, Sequence, Tuple

from . import resultat as RE

# l'ordre des vérifications d'une méthode (les autres suivent, par identifiant)
ORDRE = {"ec3": ("E-SECTION", "E-N", "E-My", "E-Mz", "E-Vz", "E-Vy", "E-MV", "E-NM", "E-Nb-y", "E-Nb-z", "E-Nb-T", "E-Mb",
                 "E-661", "E-662"),
         "historique": ("H-N-z", "H-N-y", "H-SIGMA", "H-DEV", "H-EQ"),
         "ame": ("A-AME", "A-Vb", "A-Vb-y", "A-VbM", "A-S", "A-GR")}
IDS_BARRE = ("E-Nb", "E-Mb", "E-66")             # les vérifications de la barre (les autres sont celles de la section)
# les lignes dont l'occurrence retenue AJOUTE ses hypothèses à celles de la gouvernante (sans doublon) : la flexion et le
# déversement de la méthode ELS (L_d, classe, origine du 3,9, variante) restent dits quand la ligne qui gouverne est prise
# là où rien ne fléchit (H-EQ à l'appui)
HYPOTHESES_AUSSI = {"historique": ("H-SIGMA", "H-DEV")}


def sujet(h: str) -> Optional[str]:
    """Le SUJET d'une hypothèse : son texte avant « : » (« Déversement (classeurs, NBN B 51-002) ») ; None sans « : »."""
    s, sep, _ = h.partition(" : ")
    return s if sep else None


def rang(v: Dict) -> Tuple[int, float]:
    """L'ordre de gravité d'une occurrence : un refus sans taux passe devant tout taux ; un taux devant le reste."""
    if v["statut"] == RE.NOK and v["taux"] is None:
        return (2, 0.0)
    if v["taux"] is not None:
        return (1, v["taux"])
    return (0, 0.0)


class Enveloppe:
    """Par vérification, l'occurrence la plus défavorable rencontrée, et le bilan de chaque combinaison."""

    def __init__(self, methode: str, norme: str):
        self.methode, self.norme = methode, norme
        self.pire: Dict[str, Dict] = {}
        self.bilan: Dict[str, Dict] = {}
        self.alertes: Dict[str, Dict] = {}
        self.a_confirmer: Dict[str, Dict] = {}
        self.corrections: Dict[str, Dict] = {}         # les corrections des classeurs qui ont joué (méthode ELS)

    def verser(self, r: Dict, ou: Dict, prefixes: Optional[Sequence[str]] = None, sauf: Optional[Sequence[str]] = None) -> None:
        """Verse le résultat ``r`` d'une méthode, trouvé ``ou`` ({"combinaison", "x" | "troncon"}) ; ``prefixes`` : ne
        garder que ces vérifications ; ``sauf`` : les écarter."""
        blocs = {f["cle"]: f for f in r["calcul"]}
        gardees = []
        for v in r["verifications"]:
            if prefixes and not v["id"].startswith(tuple(prefixes)):
                continue
            if sauf and v["id"].startswith(tuple(sauf)):
                continue
            gardees.append(v)
            gravite = rang(v)
            courant = self.pire.get(v["id"])
            if courant is None or gravite > courant["rang"]:
                self.pire[v["id"]] = {"rang": gravite, "v": v, "blocs": [blocs[k] for k in v["formules"]], "ou": ou, "r": r}
        if gardees:
            b = self.bilan.setdefault(ou["combinaison"], {"nom": ou["combinaison"], "rang": (0, 0.0), "gouvernant": None, "non_couvert": False,
                                                          "hors_domaine": False})
            for v in gardees:
                if rang(v) > b["rang"]:
                    b["rang"], b["gouvernant"] = rang(v), v["id"]
                b["non_couvert"] = b["non_couvert"] or v["statut"] == RE.NON_COUVERT
                b["hors_domaine"] = b["hors_domaine"] or v["statut"] == RE.HORS_DOMAINE
        for a in r["alertes"]:
            self.alertes.setdefault(a["code"], a)
        for p in r["grandeurs"].get("a_confirmer") or []:
            self.a_confirmer.setdefault(p["cle"], p)
        for c in r.get("corrections") or []:
            self.corrections.setdefault(c["id"], c)

    def _grandeurs(self, gouv: Optional[Dict]) -> Dict:
        """Les grandeurs rendues avec la méthode. Historique : celles de l'occurrence gouvernante. Eurocode 3 : celles de
        la SECTION la plus sollicitée (classe, résistances…), puis — chacune prise là où SA vérification est la plus
        défavorable — le flambement, le déversement et l'annexe A."""
        if gouv is None:
            return {}
        if self.methode != "ec3":
            return gouv["r"]["grandeurs"]

        def de(ids: Sequence[str]) -> Optional[Dict]:
            trouves = [self.pire[i] for i in ids if i in self.pire]
            return max(trouves, key=lambda o: o["rang"])["r"]["grandeurs"] if trouves else None

        barre = de([i for i in self.pire if i.startswith(IDS_BARRE)])
        section = de([i for i in self.pire if not i.startswith(IDS_BARRE)])
        G = dict(section or barre or {})
        for cle, ids in (("flambement", ("E-Nb-z", "E-Nb-y", "E-Nb-T")), ("deversement", ("E-Mb",)), ("annexe_A", ("E-661", "E-662"))):
            g = de(ids) or barre
            if g is not None and cle in g:
                G[cle] = g[cle]
        return G

    def _hypotheses(self, gouvernant: Optional[str]) -> List[str]:
        """Les hypothèses de la méthode : celles de la gouvernante, puis celles des lignes de ``HYPOTHESES_AUSSI``, sans
        doublon, chacune à sa place (après celle qui la précède ; une autre version d'un même sujet, juste après lui). Deux
        versions d'un même SUJET (``sujet`` : le déversement de deux tronçons, L_d différents) disent chacune leur tronçon
        et la vérification dont elles viennent."""
        if gouvernant is None:
            return []
        lignes = [[h, [gouvernant]] for h in self.pire[gouvernant]["r"]["hypotheses"]]     # [texte, vérifications]
        for id_ in HYPOTHESES_AUSSI.get(self.methode, ()):
            if id_ not in self.pire or id_ == gouvernant:
                continue
            i = 0
            for h in self.pire[id_]["r"]["hypotheses"]:
                j = next((k for k, li in enumerate(lignes) if li[0] == h), None)
                if j is None:
                    pareils = [k for k, li in enumerate(lignes) if sujet(h) is not None and sujet(li[0]) == sujet(h)]
                    j = pareils[-1] + 1 if pareils else i
                    lignes.insert(j, [h, []])
                lignes[j][1].append(id_)
                i = j + 1
        hyp = []
        for h, ids in lignes:
            s = sujet(h)
            if s is not None and any(sujet(li[0]) == s and li[1] != ids for li in lignes):
                h = f"{s}, {self._origine(ids)} : {h[len(s) + 3:]}"
            hyp.append(h)
        return hyp

    def _origine(self, ids: Sequence[str]) -> str:
        """D'où vient une hypothèse : « tronçon de 1,70 à 3,90 m (déversement) » — le tronçon de chaque occurrence retenue,
        avec le titre de sa vérification ; sans tronçon, les titres seuls."""
        par_troncon: Dict[Optional[Tuple[float, float]], List[str]] = {}
        for i in ids:
            o = self.pire[i]
            t = o["ou"].get("troncon")
            titre = o["v"]["titre"]
            par_troncon.setdefault(tuple(t) if t else None, []).append(titre[:1].lower() + titre[1:])
        return " ; ".join(f"tronçon de {RE.fr(t[0], 2)} à {RE.fr(t[1], 2)} m ({', '.join(ts)})" if t else ", ".join(ts)
                          for t, ts in par_troncon.items())

    def rendre(self) -> Dict:
        """Le résultat de la méthode sur toute la barre : la forme de acier/barre/resultat.py, où chaque vérification
        dit sa combinaison et son lieu ; ``calcul`` porte les formules de chaque occurrence retenue (clés « id/clé ») ;
        ``hypotheses`` : celles de la gouvernante, puis celles des lignes de ``HYPOTHESES_AUSSI`` (``_hypotheses``)."""
        verifs, calcul = [], []
        for id_ in list(ORDRE[self.methode]) + sorted(set(self.pire) - set(ORDRE[self.methode])):
            if id_ not in self.pire:
                continue
            o = self.pire[id_]
            v = dict(o["v"])
            v["formules"] = [f"{id_}/{b['cle']}" for b in o["blocs"]]
            v["messages"] = list(v["messages"])
            v.update({"combinaison": o["ou"]["combinaison"], "x": o["ou"].get("x"), "troncon": o["ou"].get("troncon")})
            calcul += [dict(b, cle=f"{id_}/{b['cle']}") for b in o["blocs"]]
            verifs.append(v)
        s = RE.synthese(verifs)
        gouv = self.pire[s["gouvernant"]] if s["gouvernant"] else None
        hyp = self._hypotheses(s["gouvernant"])
        # une correction des classeurs : le texte de l'occurrence RETENUE de sa vérification (chaque correction n'en vise
        # qu'une) — il change d'un tronçon à l'autre (φ_d prolongé, classe de la semelle comprimée) ; à défaut, la première
        # rencontrée
        corrections: Dict[str, Dict] = {}
        for v in verifs:
            for c in self.pire[v["id"]]["r"].get("corrections") or []:
                if v["id"] in c["verifications"]:
                    corrections.setdefault(c["id"], c)
        for c in self.corrections.values():
            corrections.setdefault(c["id"], c)
        combis = [{"nom": b["nom"], "gouvernant": b["gouvernant"], "taux": b["rang"][1] if b["rang"][0] == 1 else None,
                   "statut": RE.NOK if (b["rang"][0] == 2 or b["rang"][1] > 1.0) else RE.NON_COUVERT if b["non_couvert"]
                   else RE.HORS_DOMAINE if b["hors_domaine"] else RE.OK}
                  for b in self.bilan.values()]
        return {"methode": self.methode, "norme": self.norme, "statut": s["statut"], "taux_max": s["taux_max"], "gouvernant": s["gouvernant"],
                "combinaison": gouv["ou"]["combinaison"] if gouv else None, "verifications": verifs, "calcul": calcul,
                "grandeurs": self._grandeurs(gouv), "alertes": list(self.alertes.values()),
                "hypotheses": hyp, "a_confirmer": list(self.a_confirmer.values()),
                "combinaisons": combis, "corrections": list(corrections.values())}

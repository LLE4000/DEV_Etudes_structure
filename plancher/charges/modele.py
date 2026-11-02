# -*- coding: utf-8 -*-
"""Objets du module « Charges sur plancher » : le plancher et ses éléments
porteurs, les charges, les répartiteurs. Dataclasses pures, sans Streamlit.

Repère du plan : ``x`` le long de la portée, de l'appui A (x = 0) à
l'appui B (x = L) ; ``y`` en travers, de 0 à B (largeur du plancher). Les
éléments porteurs — hourdis, ou bandes d'une dalle — sont des bandes
``[y0, y1]`` parallèles à ``x``, chacune une poutre sur deux appuis.

Unités : longueurs en m, charges en kN, kN/m et kN/m², contraintes en MPa
(N/mm²), inerties en m⁴, raideurs EI en kN·m². Les dimensions de contact
des semelles et plateaux sont en m dans les objets du plan ; les
fonctions locales (bande efficace, pression) travaillent en mm et N.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math

# ---------------------------------------------------------------- constantes nommées
HOURDIS, DALLE, PREDALLE = "hourdis", "dalle", "predalle"
POSES, CLAVETES, CHAPES = "poses", "clavetes", "chapes"
RD, SERVICE = "Rd", "service"          # nature d'une valeur de catalogue (P-1)
EN_TRAVERS, DANS_LE_SENS = "y", "x"    # orientation d'une filière : en travers de la portée, ou dans son sens
RAIDEUR_INFINIE = 1.0e12               # kN/m : ressort posé sur un appui


# ---------------------------------------------------------------- plancher
@dataclass
class Element:
    """Bande porteuse ``[y0, y1]`` : un hourdis, ou une bande de dalle."""
    index: int
    nom: str
    y0: float
    y1: float

    @property
    def largeur(self):
        return self.y1 - self.y0

    @property
    def y_centre(self):
        return 0.5 * (self.y0 + self.y1)


@dataclass
class Chape:
    """Couche de compression d'un plancher de hourdis (P-17).

    ``fck_t`` : résistance du béton de la chape à l'âge du calcul (maturité,
    module ``age_beton``) ; ``fck_exigee`` : classe que le fabricant exige
    pour ses valeurs « avec dalle de compression » ; ``collaborante`` :
    surface rugueuse et joints remplis, selon le fabricant."""
    epaisseur: float                 # m
    fck_t: float                     # MPa
    collaborante: bool = True
    fck_exigee: Optional[float] = None
    gamma: float = 25.0              # kN/m³

    @property
    def poids(self):
        """kN/m²."""
        return self.epaisseur * self.gamma

    def valeurs_avec_chape_admises(self):
        """P-17 : les résistances « avec chape » du catalogue ne sont acquises
        que si la chape est collaborante ET a atteint la classe exigée ; une
        classe exigée inconnue vaut refus (conservatif)."""
        return bool(self.collaborante and self.fck_exigee is not None
                    and self.fck_t >= self.fck_exigee)


@dataclass
class Plancher:
    """Base commune : portée, largeur, charges réparties déjà présentes."""
    nom: str = "Plancher"
    L: float = 6.0                   # m, portée entre les appuis A et B
    q_presentes: float = 0.0         # kN/m², stocks, personnel… (EN 1991-1-6 §4.11, saisi)

    @property
    def type(self):
        raise NotImplementedError

    @property
    def B(self):
        raise NotImplementedError

    def elements(self) -> List[Element]:
        raise NotImplementedError

    def EI_element(self, i: int) -> float:
        """kN·m² de l'élément ``i``."""
        raise NotImplementedError

    def g_total(self) -> float:
        """Charges réparties présentes, kN/m² : poids propre + chape + autres."""
        raise NotImplementedError

    def raideur(self, i: int, a: float) -> float:
        """Raideur du ressort de l'élément ``i`` pour une charge à l'abscisse
        ``a`` (m) : k = 3·E·I·L / (a²·(L − a)²), la flèche d'une poutre sur
        deux appuis sous une charge ponctuelle (§ 5.2 de l'étude). Sur un
        appui, la raideur est « infinie »."""
        if a <= 0.0 or a >= self.L:
            return RAIDEUR_INFINIE
        return 3.0 * self.EI_element(i) * self.L / (a * a * (self.L - a) ** 2)

    def element_de(self, y: float) -> Optional[Element]:
        """L'élément qui contient ``y`` (bord supérieur inclus dans le dernier)."""
        els = self.elements()
        for e in els:
            if e.y0 <= y < e.y1:
                return e
        return els[-1] if els and abs(y - els[-1].y1) < 1e-9 else None


@dataclass
class Hourdis(Plancher):
    """Plancher de hourdis alvéolés précontraints, sur deux appuis.

    Résistances du catalogue PAR ÉLÉMENT (kN·m, kN) avec leur nature (P-1) ;
    ``I`` par élément (m⁴, donnée fabricant) et ``E`` (MPa) pour la raideur ;
    ``p_loc_adm`` : pression locale admissible sur la membrure (MPa, donnée
    fabricant, P-5) ; ``parts_annexe_C`` : parts de la figure C.1 de
    l'EN 1168 fournies par le bureau après transcription, sous la forme
    ``{0: part de l'élément chargé, 1: part de chaque premier voisin, 2: …}``
    — ``None`` = aucune répartition (défaut conservatif)."""
    b_element: float = 1.20
    n_elements: int = 8
    h: float = 0.20
    g: float = 2.9                   # kN/m², poids propre du hourdis nu
    E: float = 36000.0               # MPa
    I: Optional[float] = None        # m⁴ par élément (fabricant)
    etat: str = CLAVETES
    M_Rd: float = 0.0                # kN·m / élément
    V_Rd: float = 0.0                # kN / élément
    nature: str = RD
    M_Rd_chape: Optional[float] = None
    V_Rd_chape: Optional[float] = None
    M_cr: Optional[float] = None     # kN·m / élément, moment de fissuration (P-11)
    p_loc_adm: Optional[float] = None
    chape: Optional[Chape] = None
    parts_annexe_C: Optional[Dict[int, float]] = None
    source: str = ""

    @property
    def type(self):
        return HOURDIS

    @property
    def B(self):
        return self.b_element * self.n_elements

    def elements(self):
        return [Element(i, f"H{i + 1}", i * self.b_element, (i + 1) * self.b_element)
                for i in range(self.n_elements)]

    def inertie(self):
        """m⁴ : la donnée fabricant, ou, à défaut, une estimation NOMMÉE
        (section pleine réduite de 40 % pour les alvéoles) qui ne sert qu'à
        la raideur relative sous une filière, jamais à une résistance."""
        if self.I is not None:
            return self.I
        return 0.6 * self.b_element * self.h ** 3 / 12.0

    def EI_element(self, i):
        return self.E * 1.0e3 * self.inertie()    # MPa → kN/m², × m⁴

    def g_total(self):
        return self.g + (self.chape.poids if self.chape else 0.0) + self.q_presentes

    def repartition_active(self):
        """L'annexe C ne s'applique qu'à des joints clavetés et durcis, ou
        à un plancher chapé ; des hourdis seulement posés ne partagent rien."""
        return self.etat in (CLAVETES, CHAPES) and bool(self.parts_annexe_C)


@dataclass
class DalleBA(Plancher):
    """Dalle en béton armé coulée en place, portant dans le sens ``x``.

    ``beton`` : classe de ``beton_classes.json`` ; ``fck_t`` (MPa) remplace
    la classe par celle, arrondie vers le bas, que la maturité autorise.
    ``As_inf`` : armature inférieure dans le sens porteur (mm²/m) ; ``d`` :
    hauteur utile (m) ; ``armature_transversale_ok`` : armature de
    répartition ≥ 20 % de la principale (EN 1992-1-1 §9.3.1.1(2)), condition
    de la largeur efficace ; ``h1`` : revêtement diffusant (m)."""
    h: float = 0.20
    d: float = 0.17
    beton: str = "C30/37"
    fck_t: Optional[float] = None
    As_inf: float = 524.0            # mm²/m
    fyk: float = 500.0
    gamma_s: float = 1.15
    armature_transversale_ok: bool = True
    h1: float = 0.0
    b_bande: float = 1.0
    largeur: float = 9.6             # m, largeur totale B
    gamma_beton: float = 25.0
    E: Optional[float] = None        # MPa, sinon Ecm de la classe

    @property
    def type(self):
        return DALLE

    @property
    def B(self):
        return self.largeur

    def elements(self):
        els, i, y = [], 0, 0.0
        while y < self.B - 1e-9:
            y1 = min(y + self.b_bande, self.B)
            els.append(Element(i, f"B{i + 1}", y, y1))
            y, i = y1, i + 1
        return els

    def EI_element(self, i):
        from . import resistances
        E = self.E if self.E is not None else resistances.Ecm(self.fck_calcul())
        e = self.elements()[i]
        return E * 1.0e3 * e.largeur * self.h ** 3 / 12.0

    def fck_calcul(self):
        """fck retenu : la classe saisie, ou la classe arrondie vers le bas
        que ``fck_t`` autorise (§ 3.2 de l'étude)."""
        from . import resistances
        return resistances.fck_de_classe(self.classe_calcul())

    def classe_calcul(self):
        from . import resistances
        if self.fck_t is None:
            return self.beton
        c = resistances.classe_pour_fck(self.fck_t)
        if c is None:
            raise ValueError(f"fck(t) = {self.fck_t} MPa : sous la plus petite classe de beton_classes.json")
        return c

    def g_total(self):
        return self.gamma_beton * self.h + self.q_presentes

    @property
    def rho_l(self):
        """Taux d'armature longitudinale pour le poinçonnement."""
        return self.As_inf / (1000.0 * self.d * 1000.0)


@dataclass
class Predalle(DalleBA):
    """Prédalle + chape durcie : une dalle unidirectionnelle (§ 3.3), avec
    deux particularités : la largeur efficace est plafonnée à la prédalle
    si les joints ne sont pas armés (``joints_armes``), et une charge à
    cheval sur un joint est signalée (P-16). ``chape_durcie`` = False :
    refus de toute charge d'engin (P-7). ``portee_etayee`` : entraxe des
    étais encore en place, qui remplace la portée (P-8, avec avertissement)."""
    b_predalle: float = 2.40
    joints_armes: bool = True
    chape_durcie: bool = True
    portee_etayee: Optional[float] = None

    @property
    def type(self):
        return PREDALLE

    def joints(self) -> List[float]:
        """Ordonnées des joints entre prédalles."""
        js, y = [], self.b_predalle
        while y < self.B - 1e-9:
            js.append(y)
            y += self.b_predalle
        return js


# ---------------------------------------------------------------- charges
@dataclass
class Empreinte:
    """Charge ponctuelle de service ``P`` (kN) sur une empreinte ``a × b``
    (m, a le long de x) centrée en (x, y) ; ``phi`` : coefficient dynamique
    (P-6) ; ``ref`` : identifiant unique, clé des répartiteurs."""
    ref: str
    x: float
    y: float
    a: float
    b: float
    P: float
    phi: float = 1.0
    origine: str = ""

    @property
    def P_dyn(self):
        return self.P * self.phi


@dataclass
class ChargeLineaire:
    """Charge linéaire ``p`` (kN/m) sur ``longueur``, de largeur de contact
    ``largeur``, centrée en (x, y), orientée ``"x"`` ou ``"y"``."""
    ref: str
    x: float
    y: float
    longueur: float
    largeur: float
    p: float
    orientation: str = EN_TRAVERS
    phi: float = 1.0

    def discretiser(self, pas: float = 0.25) -> List[Empreinte]:
        """Découpage en charges ponctuelles équivalentes (pas ≤ ``pas``)."""
        n = max(1, int(math.ceil(self.longueur / pas)))
        dl = self.longueur / n
        out = []
        for k in range(n):
            s = -self.longueur / 2 + (k + 0.5) * dl
            if self.orientation == DANS_LE_SENS:
                out.append(Empreinte(f"{self.ref}.{k + 1}", self.x + s, self.y, dl, self.largeur,
                                     self.p * dl, self.phi, self.ref))
            else:
                out.append(Empreinte(f"{self.ref}.{k + 1}", self.x, self.y + s, self.largeur, dl,
                                     self.p * dl, self.phi, self.ref))
        return out


@dataclass
class ChargeSurfacique:
    """Charge surfacique ``q`` (kN/m²) sur le rectangle [x0, x0 + lx] × [y0, y0 + ly]."""
    ref: str
    x0: float
    y0: float
    lx: float
    ly: float
    q: float
    phi: float = 1.0

    def discretiser(self, pas: float = 0.30) -> List[Empreinte]:
        nx = max(1, int(math.ceil(self.lx / pas)))
        ny = max(1, int(math.ceil(self.ly / pas)))
        dx, dy = self.lx / nx, self.ly / ny
        out = []
        for i in range(nx):
            for j in range(ny):
                out.append(Empreinte(f"{self.ref}.{i + 1}.{j + 1}", self.x0 + (i + 0.5) * dx,
                                     self.y0 + (j + 0.5) * dy, dx, dy, self.q * dx * dy, self.phi, self.ref))
        return out


@dataclass
class AppuiGabarit:
    """Un appui d'un gabarit d'engin, dans le repère de l'engin (x vers
    l'avant, y vers la gauche) : roue, pied, chenille (découpée en points)."""
    nom: str
    dx: float
    dy: float
    a: float
    b: float
    P: float                         # kN, charge de service par appui (tableau de stabilité)
    phi: float = 1.0


@dataclass
class Gabarit:
    """Fiche d'engin : géométrie et charges par appui, avec sa source."""
    nom: str
    type_appuis: str                 # "pieds", "roues", "chenilles", "ponctuel", "surfacique"
    appuis: List[AppuiGabarit] = field(default_factory=list)
    source: str = ""
    generique: bool = True
    notes: str = ""

    def charges_renseignees(self):
        return all(ap.P > 0 for ap in self.appuis)


@dataclass
class Engin:
    """Un gabarit posé sur le plan : centre (x, y), angle 0 / 90 / 180 / 270."""
    ref: str
    gabarit: Gabarit
    x: float
    y: float
    angle: int = 0

    def empreintes(self) -> List[Empreinte]:
        if self.angle % 90 != 0:
            raise ValueError("Angle d'un engin : 0, 90, 180 ou 270° seulement (V1).")
        k = (self.angle // 90) % 4
        out = []
        for ap in self.gabarit.appuis:
            dx, dy, a, b = ap.dx, ap.dy, ap.a, ap.b
            for _ in range(k):               # rotation de +90° : (dx, dy) → (−dy, dx)
                dx, dy, a, b = -dy, dx, b, a
            out.append(Empreinte(f"{self.ref}.{ap.nom}", self.x + dx, self.y + dy, a, b, ap.P, ap.phi, self.ref))
        return out


# ---------------------------------------------------------------- répartiteurs
@dataclass
class Plateau:
    """Plateau posé directement sous une empreinte (§ 5.1) : matériau de
    résistance ``f`` (MPa : fy/γM0 pour l'acier, f_m,d pour le bois ou
    l'OSB, saisi avec sa source), épaisseur ``t`` (mm), dimensions ``A × Bp``
    (mm) ; ``rigide_declare`` + ``p_adm_declare`` : plaque de fournisseur
    déclarée rigide, pression admissible publiée."""
    nom: str
    f: Optional[float]
    t: float
    A: float
    Bp: float
    rigide_declare: bool = False
    p_adm_declare: Optional[float] = None
    dx: float = 0.0                  # décalage du plateau par rapport à l'empreinte (m)
    dy: float = 0.0
    source: str = ""
    masse: float = 0.0               # kg, pour le classement des candidats


@dataclass
class Filiere:
    """Poutre de répartition (§ 5.2) : raideur ``EI`` (kN·m²), module ``W_el``
    (cm³), limite ``fy`` (MPa), longueur (m), centre (x, y), orientation
    ``"y"`` (en travers des éléments) ou ``"x"`` (dans le sens porteur),
    largeur de semelle ``b_semelle`` (m). ``refs`` : empreintes portées."""
    nom: str
    EI: float
    W_el: float
    fy: float
    longueur: float
    x: float
    y: float
    orientation: str = EN_TRAVERS
    b_semelle: float = 0.10
    refs: List[str] = field(default_factory=list)
    gamma_M0: float = 1.0
    masse: float = 0.0               # kg/m
    source: str = ""

    def axe(self):
        """Intervalle couvert le long de l'axe de la filière."""
        c = self.y if self.orientation == EN_TRAVERS else self.x
        return c - self.longueur / 2, c + self.longueur / 2

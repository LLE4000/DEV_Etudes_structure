# -*- coding: utf-8 -*-
"""RÉUNIR plusieurs PDF en un seul, en Python pur : le navigateur (Pyodide) n'a ni pymupdf ni pypdf, et reportlab
écrit des PDF mais ne les relit pas.

Les PAGES sont recopiées TELLES QUELLES : leurs flux de contenu (le dessin, le texte, compressés), leurs polices
incorporées, leurs états graphiques passent octet pour octet ; seuls les NUMÉROS des objets sont décalés, et chaque
page est rattachée au nouvel arbre des pages. Le contenu d'une note réunie est donc celui de la note seule — ce que
``tests/test_note_assemblages.py`` prouve (pdftotext de chaque note seule = celui de ses pages dans le document réuni).

Ce que le document réuni porte en plus : les SIGNETS (un par partie : titre, page), le titre et l'auteur. Il ne
porte rien d'autre des documents d'origine (ni leurs informations, ni leurs signets).

Le lecteur couvre ce que reportlab écrit (table des renvois classique, longueurs de flux directes) et ce qu'écrivent
les générateurs ordinaires du même genre : un PDF qu'il ne sait pas lire (renvois compressés en flux d'objets, PDF
chiffré, longueur indirecte introuvable) est REFUSÉ avec un message, jamais recopié à moitié.

    reunir([octets_1, octets_2, …], signets=[(titre, page), …], titre="…", auteur="…") → octets
    nombre_de_pages(octets) → int
"""
import hashlib
import re
from typing import Dict, List, Optional, Sequence, Tuple

_ESPACES = b" \t\r\n\f\x00"
_DELIMITEURS = b"()<>[]{}/%"
# une référence « n g R » : ses nombres ne prolongent pas un nom (« /F2 8 0 R » : 8, jamais 2)
_REF = re.compile(rb"(?<![^ \t\r\n\f\x00\[\]<>{}()/%])(\d+)\s+(\d+)\s+R(?![^ \t\r\n\f\x00\[\]<>{}()/%])")
_ENTETE_OBJ = re.compile(rb"\s*(\d+)\s+(\d+)\s+obj")
# les attributs qu'une page HÉRITE de ses ancêtres dans l'arbre des pages (PDF 1.7, § 7.7.3.4)
_HERITES = (b"/Resources", b"/MediaBox", b"/CropBox", b"/Rotate")


class PdfIllisible(ValueError):
    """Un PDF que ce lecteur ne sait pas relire (le message dit pourquoi)."""


# ------------------------------------------------------------------ la lecture des valeurs
def _blancs(o: bytes, i: int) -> int:
    n = len(o)
    while i < n:
        c = o[i]
        if c in _ESPACES:
            i += 1
        elif c == 0x25:                                   # « % » : un commentaire, jusqu'à la fin de la ligne
            while i < n and o[i] not in b"\r\n":
                i += 1
        else:
            break
    return i


def _fin_chaine(o: bytes, i: int) -> int:
    """La fin (exclue) de la chaîne littérale qui commence en ``i`` (« ( ») : parenthèses imbriquées, échappements."""
    prof, i, n = 0, i, len(o)
    while i < n:
        c = o[i]
        if c == 0x5C:                                     # « \ » : le caractère suivant est pris tel quel
            i += 2
            continue
        if c == 0x28:
            prof += 1
        elif c == 0x29:
            prof -= 1
            if prof == 0:
                return i + 1
        i += 1
    raise PdfIllisible("chaîne non fermée")


def _fin_valeur(o: bytes, i: int) -> int:
    """La fin (exclue) de la valeur PDF qui commence en ``i`` (après les blancs) : dictionnaire, tableau, chaîne,
    nom, nombre, référence « n g R », booléen, null."""
    i = _blancs(o, i)
    if o.startswith(b"<<", i):
        i += 2
        while True:
            i = _blancs(o, i)
            if o.startswith(b">>", i):
                return i + 2
            i = _fin_valeur(o, i)                         # la clé
            i = _fin_valeur(o, i)                         # la valeur
    c = o[i:i + 1]
    if c == b"[":
        i += 1
        while True:
            i = _blancs(o, i)
            if o.startswith(b"]", i):
                return i + 1
            i = _fin_valeur(o, i)
    if c == b"(":
        return _fin_chaine(o, i)
    if c == b"<":
        j = o.find(b">", i)
        if j < 0:
            raise PdfIllisible("chaîne hexadécimale non fermée")
        return j + 1
    if c == b"/":
        i += 1
        while i < len(o) and o[i] not in _ESPACES and o[i] not in _DELIMITEURS:
            i += 1
        return i
    m = _REF.match(o, i)
    if m:
        return m.end()
    j = i
    while j < len(o) and o[j] not in _ESPACES and o[j] not in _DELIMITEURS:
        j += 1
    if j == i:
        raise PdfIllisible(f"valeur illisible à l'octet {i}")
    return j


def _entrees(corps: bytes) -> List[Tuple[bytes, bytes]]:
    """Les entrées d'un dictionnaire de premier niveau, dans l'ordre : [(b"/Clé", valeur brute)]."""
    i = _blancs(corps, 0)
    if not corps.startswith(b"<<", i):
        raise PdfIllisible("dictionnaire attendu")
    i += 2
    out = []
    while True:
        i = _blancs(corps, i)
        if corps.startswith(b">>", i):
            return out
        j = _fin_valeur(corps, i)
        k = _blancs(corps, j)
        fin = _fin_valeur(corps, k)
        out.append((corps[i:j], corps[k:fin]))
        i = fin


def _dict(entrees: Sequence[Tuple[bytes, bytes]]) -> bytes:
    return b"<< " + b" ".join(k + b" " + v for k, v in entrees) + b" >>"


def _ref(valeur: bytes) -> Optional[int]:
    m = _REF.fullmatch(valeur.strip())
    return int(m.group(1)) if m else None


def _refs_tableau(valeur: bytes) -> List[int]:
    v = valeur.strip()
    if not (v.startswith(b"[") and v.endswith(b"]")):
        raise PdfIllisible("tableau attendu")
    return [int(m.group(1)) for m in _REF.finditer(_hors_chaines(v))]


def _hors_chaines(corps: bytes) -> bytes:
    """Le corps où chaque chaîne (littérale ou hexadécimale) est remplacée par des blancs de même longueur : ce qui
    ressemble à « 3 0 R » dans un titre n'est jamais pris pour une référence."""
    out, i, n = bytearray(corps), 0, len(corps)
    while i < n:
        c = corps[i]
        if c == 0x28:
            j = _fin_chaine(corps, i)
        elif c == 0x3C and not corps.startswith(b"<<", i):
            j = corps.find(b">", i) + 1
            if j <= 0:
                raise PdfIllisible("chaîne hexadécimale non fermée")
        elif corps.startswith(b"<<", i) or corps.startswith(b">>", i):
            i += 2
            continue
        else:
            i += 1
            continue
        out[i:j] = b" " * (j - i)
        i = j
    return bytes(out)


def _renumeroter(corps: bytes, table: Dict[int, int]) -> bytes:
    """Le corps où chaque référence « n g R » (hors des chaînes) devient « table[n] 0 R »."""
    masque = _hors_chaines(corps)
    morceaux, dernier = [], 0
    for m in _REF.finditer(masque):
        n = int(m.group(1))
        if n not in table:
            raise PdfIllisible(f"référence vers un objet absent ({n} {int(m.group(2))} R)")
        morceaux += [corps[dernier:m.start()], b"%d 0 R" % table[n]]
        dernier = m.end()
    morceaux.append(corps[dernier:])
    return b"".join(morceaux)


# ------------------------------------------------------------------ un PDF relu
class Lecteur:
    """Un PDF relu : ses objets (numéro → (valeur brute, flux ou None)) et ses pages dans l'ordre (numéros, avec les
    attributs hérités de l'arbre des pages)."""

    def __init__(self, octets: bytes):
        if not isinstance(octets, (bytes, bytearray)) or not bytes(octets[:5]) == b"%PDF-":
            raise PdfIllisible("ce n'est pas un PDF")
        self.o = bytes(octets)
        self.version = self.o[5:8].decode("latin-1")
        self.decalages = self._table()
        self._cache: Dict[int, Tuple[bytes, Optional[bytes]]] = {}
        racine = _ref(dict(self.trailer).get(b"/Root", b""))
        if racine is None:
            raise PdfIllisible("pas de catalogue (/Root)")
        pages = _ref(dict(_entrees(self.objet(racine)[0])).get(b"/Pages", b""))
        if pages is None:
            raise PdfIllisible("pas d'arbre des pages")
        self.pages: List[Tuple[int, Dict[bytes, bytes]]] = []
        self._parcourir(pages, {}, set())

    def _table(self) -> Dict[int, int]:
        o = self.o
        k = o.rfind(b"startxref")
        if k < 0:
            raise PdfIllisible("pas de « startxref »")
        try:
            debut = int(o[k + 9:].split()[0])
        except (ValueError, IndexError):
            raise PdfIllisible("« startxref » sans décalage lisible") from None
        if not o.startswith(b"xref", _blancs(o, debut)):
            raise PdfIllisible("table des renvois compressée (flux d'objets) : non prise en charge")
        fin = o.find(b"trailer", debut)
        if fin < 0:
            raise PdfIllisible("pas de « trailer »")
        decalages, lignes = {}, o[debut:fin].split(b"xref", 1)[1]
        i = 0
        toks = lignes.split()
        try:
            while i < len(toks):
                if i + 1 < len(toks) and len(toks[i]) != 10:   # un en-tête de sous-section : « premier nombre »
                    num, compte = int(toks[i]), int(toks[i + 1])
                    i += 2
                    for _ in range(compte):
                        dec, _gen, etat = toks[i:i + 3]
                        if etat == b"n":
                            decalages[num] = int(dec)
                        num += 1
                        i += 3
                else:
                    raise PdfIllisible("table des renvois illisible")
        except (ValueError, IndexError) as e:                 # une entrée tronquée ou qui n'est pas un nombre
            raise PdfIllisible(f"table des renvois illisible ({e})") from None
        t = _blancs(o, fin + 7)
        self.trailer = _entrees(o[t:_fin_valeur(o, t)])
        if b"/Encrypt" in dict(self.trailer):
            raise PdfIllisible("PDF chiffré")
        if b"/Prev" in dict(self.trailer):
            raise PdfIllisible("PDF mis à jour par ajouts (plusieurs tables des renvois) : non pris en charge")
        return decalages

    def objet(self, num: int) -> Tuple[bytes, Optional[bytes]]:
        """(valeur brute, flux brut ou None) de l'objet ``num``."""
        if num in self._cache:
            return self._cache[num]
        if num not in self.decalages:
            raise PdfIllisible(f"objet {num} absent de la table des renvois")
        o = self.o
        m = _ENTETE_OBJ.match(o, self.decalages[num])
        if not m or int(m.group(1)) != num:
            raise PdfIllisible(f"objet {num} introuvable à son décalage")
        i = _blancs(o, m.end())
        j = _fin_valeur(o, i)
        corps, flux = o[i:j], None
        k = _blancs(o, j)
        if o.startswith(b"stream", k):
            k += 6
            if o.startswith(b"\r\n", k):
                k += 2
            elif o[k:k + 1] in (b"\n", b"\r"):
                k += 1
            lg = dict(_entrees(corps)).get(b"/Length", b"").strip()
            n = _ref(lg)
            if n is not None:                                     # une longueur indirecte
                lg = self.objet(n)[0].strip()
            if not lg.isdigit():
                raise PdfIllisible(f"longueur du flux {num} illisible")
            flux = o[k:k + int(lg)]
            if not o.startswith(b"endstream", _blancs(o, k + int(lg))):
                raise PdfIllisible(f"flux {num} : « endstream » n'est pas à sa longueur")
        self._cache[num] = (corps, flux)
        return corps, flux

    def _parcourir(self, num: int, herites: Dict[bytes, bytes], vus: set) -> None:
        if num in vus:
            raise PdfIllisible("arbre des pages circulaire")
        vus.add(num)
        e = dict(_entrees(self.objet(num)[0]))
        typ = e.get(b"/Type", b"").strip()
        if typ == b"/Pages":
            h = dict(herites, **{k: e[k] for k in _HERITES if k in e})
            for kid in _refs_tableau(e.get(b"/Kids", b"[]")):
                self._parcourir(kid, h, vus)
        elif typ == b"/Page":
            self.pages.append((num, {k: v for k, v in herites.items() if k not in e}))
        else:
            raise PdfIllisible(f"nœud de l'arbre des pages inattendu ({typ.decode('latin-1')})")

    def atteints(self, depart: Sequence[int]) -> List[int]:
        """Les objets atteints depuis ``depart`` (références des valeurs, jamais le contenu des flux), dans l'ordre
        de la découverte."""
        vus, file, ordre = set(), list(depart), []
        while file:
            n = file.pop(0)
            if n in vus:
                continue
            vus.add(n)
            ordre.append(n)
            corps, _ = self.objet(n)
            for m in _REF.finditer(_hors_chaines(corps)):
                file.append(int(m.group(1)))
        return ordre


def nombre_de_pages(octets: bytes) -> int:
    return len(Lecteur(octets).pages)


# ------------------------------------------------------------------ l'écriture
def texte_pdf(s: str) -> bytes:
    """Une chaîne de texte PDF (titre, signet) : UTF-16 grand-boutiste avec son BOM, en hexadécimal — les accents et
    le tiret cadratin passent sans table de codage."""
    return b"<FEFF" + s.encode("utf-16-be").hex().upper().encode("ascii") + b">"


def reunir(documents: Sequence[bytes], signets: Sequence[Tuple[str, int]] = (), titre: str = "", auteur: str = "") -> bytes:
    """Le document qui met bout à bout les pages de ``documents`` (dans l'ordre), avec ``signets`` [(titre, numéro de
    page à partir de 1)], le titre et l'auteur. Reproductible : mêmes pages, mêmes signets → mêmes octets."""
    lecteurs = [Lecteur(d) for d in documents]
    sortie: List[Tuple[bytes, Optional[bytes]]] = []        # objet n° i + 1
    sortie.append((b"", None))                              # 1 : le catalogue (écrit à la fin)
    sortie.append((b"", None))                              # 2 : l'arbre des pages
    pages_sortie: List[int] = []
    for L in lecteurs:
        herites = dict(L.pages)                              # page → attributs hérités de ses ancêtres
        depart = []
        for p, h in L.pages:
            for k, v in _entrees(L.objet(p)[0]) + list(h.items()):
                if k != b"/Parent":
                    depart += [int(m.group(1)) for m in _REF.finditer(_hors_chaines(v))]
        # les pages d'abord (dans l'ordre), puis ce qu'elles atteignent — jamais leurs anciens parents
        table = {}
        for n in [p for p, _ in L.pages] + L.atteints(depart):
            if n not in table:
                table[n] = len(sortie) + 1
                sortie.append((b"", None))
        for n, nouveau in table.items():
            corps, flux = L.objet(n)
            if n in herites:
                h = herites[n]
                entrees = [(k, _renumeroter(v, table)) for k, v in _entrees(corps) if k != b"/Parent"]
                entrees += [(k, _renumeroter(v, table)) for k, v in h.items()]
                entrees.append((b"/Parent", b"2 0 R"))
                corps = _dict(entrees)
                pages_sortie.append(nouveau)
            else:
                corps = _renumeroter(corps, table)
            sortie[nouveau - 1] = (corps, flux)
    sortie[1] = (b"<< /Type /Pages /Count %d /Kids [ %s ] >>" % (len(pages_sortie), b" ".join(b"%d 0 R" % p for p in pages_sortie)), None)
    catalogue = [(b"/Type", b"/Catalog"), (b"/Pages", b"2 0 R")]
    signets = [(t, p) for t, p in signets if 1 <= int(p) <= len(pages_sortie)]
    if signets:
        racine = len(sortie) + 1
        premiers = racine + 1
        sortie.append((b"", None))
        for i, (t, p) in enumerate(signets):
            e = [(b"/Title", texte_pdf(str(t))), (b"/Parent", b"%d 0 R" % racine),
                 (b"/Dest", b"[ %d 0 R /XYZ null null null ]" % pages_sortie[int(p) - 1])]
            if i:
                e.append((b"/Prev", b"%d 0 R" % (premiers + i - 1)))
            if i + 1 < len(signets):
                e.append((b"/Next", b"%d 0 R" % (premiers + i + 1)))
            sortie.append((_dict(e), None))
        sortie[racine - 1] = (b"<< /Type /Outlines /First %d 0 R /Last %d 0 R /Count %d >>"
                              % (premiers, premiers + len(signets) - 1, len(signets)), None)
        catalogue.append((b"/Outlines", b"%d 0 R" % racine))
    sortie[0] = (_dict(catalogue), None)
    infos = [(b"/Producer", texte_pdf("Stab Méthodes"))]
    if titre:
        infos.insert(0, (b"/Title", texte_pdf(titre)))
    if auteur:
        infos.insert(1 if titre else 0, (b"/Author", texte_pdf(auteur)))
    sortie.append((_dict(infos), None))
    n_info = len(sortie)

    version = max([L.version for L in lecteurs] + ["1.4"])
    morceaux = [b"%PDF-" + version.encode("latin-1") + b"\n%\xe2\xe3\xcf\xd3\n"]
    position = len(morceaux[0])
    decalages = []
    for i, (corps, flux) in enumerate(sortie, start=1):
        decalages.append(position)
        if flux is None:
            bloc = b"%d 0 obj\n" % i + corps + b"\nendobj\n"
        else:
            bloc = b"%d 0 obj\n" % i + corps + b"\nstream\n" + flux + b"\nendstream\nendobj\n"
        morceaux.append(bloc)
        position += len(bloc)
    empreinte = hashlib.md5(b"".join(morceaux)).hexdigest().encode("ascii")
    xref = [b"xref\n0 %d\n" % (len(sortie) + 1), b"0000000000 65535 f \n"]
    xref += [b"%010d 00000 n \n" % d for d in decalages]
    morceaux += xref
    morceaux.append(b"trailer\n<< /Size %d /Root 1 0 R /Info %d 0 R /ID [ <%s> <%s> ] >>\nstartxref\n%d\n%%%%EOF\n"
                    % (len(sortie) + 1, n_info, empreinte, empreinte, position))
    return b"".join(morceaux)

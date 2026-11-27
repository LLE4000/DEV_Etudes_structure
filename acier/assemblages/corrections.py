# -*- coding: utf-8 -*-
"""Les CORRECTIONS d'une géométrie que le moteur refuse, VÉRIFIÉES par le calcul complet (26/11/2026).

Demande du propriétaire : « Ça me dit que c'est pas bon : que ça me propose une solution bonne. Les pinces, par exemple,
je suis pas bon : que ça me propose un truc qui serait correct. »

Quand le moteur refuse une disposition — une pince ou un entraxe hors du Tableau 3.3, une platine trop étroite, une
pièce trop haute pour la place disponible —, il le dit par une ALERTE bloquante (``R.alerts``) ou par une ligne non
conforme de son tableau des distances (``R.dist``), et il nomme les SAISIES en cause (``fields``). Ce fichier ne
remplace AUCUNE formule et n'en écrit aucune : il CHERCHE, pour chaque défaut, la plus petite retouche d'UNE saisie qui
le fait tomber, et laisse le calcul complet (``calculer``) JUGER chaque essai. Une correction proposée est donc, par
construction, une valeur avec laquelle le moteur ne signale plus ce défaut — et n'en signale aucun nouveau.

La recherche (``corriger``), pour chaque défaut et chaque saisie permise qu'il nomme, dans les deux sens :

1. la valeur avance par CRANS sur la grille de la saisie (5 mm pour une cote, 1 pour un nombre de rangées, la série du
   catalogue pour une épaisseur) : 1, 2, 3, 4, 6, 8, 12, 16 crans ;
2. au premier cran, si la pince en cause ne se rapproche pas de sa limite, le sens est abandonné (un calcul) ;
3. le premier cran qui LÈVE le défaut est gardé, puis resserré entre lui et le cran d'avant ;
4. « lever » : l'alerte n'est plus émise, la pince est conforme, et le calcul n'a ni nouvelle alerte bloquante ni
   nouvelle pince non conforme (une platine élargie qui déborde de l'aile n'est pas une correction).

Les retouches d'un défaut sont rendues la moins intrusive d'abord : une cote avant un nombre de rangées, le moins de
crans. Une correction ne touche jamais un effort, un profilé, un acier, un niveau ni le grugeage : c'est l'appelant qui
dit les saisies ``permises`` (celles qu'une proposition du prédimensionnement change, ``propositions.py``).

Référence : aucune règle de calcul ici — les limites (EN 1993-1-8 Tableau 3.3) sont celles que le moteur vérifie.
Origine de la copie : AUCUNE — composant NEUF.
"""
import math

from acier.js import AttrDict

# le pas d'une cote (mm) ; un nombre de rangées avance de 1 ; une épaisseur suit sa série (``grilles``)
PAS_MM = 5.0
# les crans essayés, dans l'ordre (16 crans de 5 mm : 80 mm — au-delà, ce n'est plus une retouche)
CRANS = (1, 2, 3, 4, 6, 8, 12, 16)
# le nombre de calculs complets d'une recherche, au plus (un calcul : 1 à 3 ms hors navigateur, quatre fois plus dedans)
BUDGET = 72
# … et par défaut (un défaut qui nomme huit saisies ne prend pas tout le budget aux suivants)
BUDGET_DEFAUT = 30


def bloquantes(R):
    """Les identifiants des alertes bloquantes d'un calcul."""
    return {a.id for a in (R.get("alerts") or []) if a.block}


def pinces_ko(R):
    """Les identifiants des pinces et entraxes non conformes d'un calcul (Tableau 3.3)."""
    return {p.id for p in (R.get("dist") or []) if not p.ok}


def defauts(R):
    """Ce que le moteur REFUSE dans la géométrie de ``R`` : [AttrDict(id, alerte, pince, fields)], dans l'ordre des
    alertes du moteur — une alerte bloquante et la pince non conforme qu'elle signale ne font qu'UN défaut ; une pince
    non conforme sans alerte en est un aussi (``alerte`` None)."""
    par_alerte = {}
    for p in (R.get("dist") or []):
        if not p.ok:
            par_alerte.setdefault(p.alert, p)
    out, vues = [], set()
    for a in (R.get("alerts") or []):
        p = par_alerte.get(a.id)
        if not a.block and p is None:
            continue
        vues.add(a.id)
        champs = list(dict.fromkeys(list(a.fields or []) + list(p.fields if p is not None else [])))
        out.append(AttrDict(id=a.id, alerte=a, pince=p, fields=champs))
    for ident, p in par_alerte.items():
        if ident not in vues:
            out.append(AttrDict(id="pince:" + p.id, alerte=None, pince=p, fields=list(p.fields or [])))
    return out


def ecart(p):
    """De combien la pince ``p`` sort de ses limites (mm, ≥ 0)."""
    if p is None:
        return 0.0
    if p.min is not None and p.val < p.min:
        return p.min - p.val
    if p.max is not None and p.val > p.max:
        return p.val - p.max
    return 0.0


def _pince(R, ident):
    return next((p for p in (R.get("dist") or []) if p.id == ident), None)


def leve(d, R0, R1):
    """Le calcul ``R1`` a-t-il LEVÉ le défaut ``d`` de ``R0`` — sans en créer un autre ?"""
    if R1 is None:
        return False
    b0, b1, k0, k1 = bloquantes(R0), bloquantes(R1), pinces_ko(R0), pinces_ko(R1)
    if d.alerte is not None and d.alerte.id in b1:
        return False
    if d.pince is not None and d.pince.id in k1:
        return False
    return b1 <= b0 and k1 <= k0


def cran(v0, sens, k, pas):
    """La valeur à ``k`` crans de ``v0`` dans le ``sens`` (+1 ou −1), sur la grille ``pas`` — un nombre (le premier cran
    ramène sur la grille : 62 → 65 ou 60) ou une série de valeurs (les épaisseurs du catalogue). None hors de la grille."""
    if isinstance(pas, (list, tuple)):
        serie = sorted(float(x) for x in pas)
        serie = [x for x in serie if x > v0 + 1e-9] if sens > 0 else [x for x in serie if x < v0 - 1e-9][::-1]
        return serie[k - 1] if k <= len(serie) else None
    base = math.floor(v0 / pas + 1e-9) * pas if sens > 0 else math.ceil(v0 / pas - 1e-9) * pas
    v = base + sens * k * pas
    return v if v >= pas - 1e-9 else None


def corriger(calculer, u, R, permises, entieres=(), grilles=None, nb=3, budget=BUDGET):
    """Les corrections des défauts de ``R`` (le calcul des saisies ``u``).

    ``calculer`` : saisies → calcul complet du moteur, ou None s'il ne calcule pas ; ``permises`` : les saisies qu'une
    correction peut changer ; ``entieres`` : celles qui comptent (un cran = 1) ; ``grilles`` : {saisie: pas ou série}.

    Rend [AttrDict(defaut, retouches)] dans l'ordre des défauts ; ``retouches`` — au plus ``nb``, la moins intrusive
    d'abord — : [AttrDict(cle, de, vers, crans, R, verifie)] (``R`` : le calcul avec la retouche ; ``verifie`` : tout
    l'assemblage y est-il vérifié). ``essais`` (calculs complets) est porté par le premier élément."""
    grilles = grilles or {}
    essais = [0]
    faits = {}

    def essayer(cle, v):
        k = (cle, round(float(v), 6))
        if k not in faits:
            essais[0] += 1
            w = dict(u)
            w[cle] = int(round(v)) if cle in entieres else v
            try:
                faits[k] = calculer(w)
            except Exception:  # noqa: BLE001 — une valeur que le moteur ne calcule pas n'est pas une correction
                faits[k] = None
        return faits[k]

    out = []
    for d in defauts(R):
        fin = min(budget, essais[0] + BUDGET_DEFAUT)
        trouvees = []
        for cle in d.fields:
            if cle not in permises or isinstance(u.get(cle), bool):
                continue
            try:
                v0 = float(str(u.get(cle)).replace(",", "."))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(v0):
                continue
            pas = grilles.get(cle, 1.0 if cle in entieres else PAS_MM)
            for sens in (1, -1):
                avant, bon = 0, None
                for k in CRANS:
                    if essais[0] >= fin:
                        break
                    v = cran(v0, sens, k, pas)
                    if v is None:
                        break
                    R1 = essayer(cle, v)
                    if leve(d, R, R1):
                        bon = (k, v, R1)
                        break
                    # au premier cran, la pince en cause doit se rapprocher de sa limite : sinon ce sens ne mène nulle part
                    if k == 1 and d.pince is not None:
                        p1 = _pince(R1, d.pince.id) if R1 is not None else None
                        if p1 is None or ecart(p1) >= ecart(d.pince) - 1e-9:
                            break
                    avant = k
                if bon is None:
                    continue
                # resserré entre le dernier cran refusé et celui qui lève le défaut
                for k in range(avant + 1, bon[0]):
                    if essais[0] >= fin:
                        break
                    v = cran(v0, sens, k, pas)
                    R1 = essayer(cle, v) if v is not None else None
                    if leve(d, R, R1):
                        bon = (k, v, R1)
                        break
                k, v, R1 = bon
                # « vérifié » : sans raison à justifier (lot de clôture du 27/11/2026) ; sinon la raison est dite
                trouvees.append(AttrDict(cle=cle, de=v0, vers=float(int(round(v))) if cle in entieres else float(v), crans=k, R=R1,
                                         verifie=bool(R1.verified and not R1.get("a_verifier")),
                                         a_verifier=str(R1.get("a_verifier") or "") if R1.verified else ""))
        ordre = {c: i for i, c in enumerate(d.fields)}
        trouvees.sort(key=lambda x: (x.cle in entieres, x.crans, ordre[x.cle]))
        out.append(AttrDict(defaut=d, retouches=trouvees[:nb], essais=0))
    if out:
        out[0].essais = essais[0]
    return out

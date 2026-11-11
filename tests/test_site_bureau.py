"""Les PROJETS, l'ÉQUIPE et le JOURNAL D'ACTIVITÉ du site (docs/site/PROJETS_EQUIPE.md).

Ce que ce fichier vérifie, sans navigateur :
  1. les données (js/equipe.js, js/journal.js, js/projets.js, et la signature que js/notes.js pose), jouées par node avec
     un localStorage simulé : les initiales (1 à 5, proposées d'après le nom), les fiches refusées et pourquoi, la
     signature d'une note et sa ligne au journal, le journal qui ne se répète pas, les projets que les notes nomment et
     ceux du registre, le renommage qui réécrit les notes, ce qu'on ne peut pas supprimer, un registre abîmé jamais écrasé
     sans copie, des noms piégés rendus comme du texte ;
  2. les pages (site/web/construire.py) : Projets et Équipe existent, le menu y mène depuis toutes les pages, l'accueil
     porte les projets et l'activité ; les copies liées (les pages du bureau, les teintes, les longueurs) ;
  3. des mutations : chacune abîme une ligne d'une copie des sources et rejoue CE fichier dessus — il doit rougir.

Le vrai navigateur (les quatre formats, les gestes) : site/web/verifier_site.js, filtre « bureau ».
"""
import json
import os
import re
import subprocess
import sys

import shutil
import tempfile

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(RACINE, "site", "web")
sys.path.insert(0, WEB)
import construire as SW  # noqa: E402  (le générateur du site)

# Les MUTATIONS rejouent CE FICHIER en entier sur une copie abîmée des sources (un sous-processus par mutation) : la
# copie est donnée par ces variables, et la copie ne rejoue pas les mutations à son tour.
JS = os.environ.get("BUREAU_JS") or os.path.join(WEB, "js")          # les scripts jugés
CSS = os.environ.get("BUREAU_CSS") or os.path.join(WEB, "css")        # les feuilles jugées
REJOUE = bool(os.environ.get("BUREAU_REJOUE"))                        # vrai dans le sous-processus d'une mutation

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:420]}"))


def source(nom):
    return open(os.path.join(JS, nom), encoding="utf-8").read()


def feuille(nom):
    return open(os.path.join(CSS, nom), encoding="utf-8").read()


# ================================================================================================ 1 · les données (node)
ESSAI = r"""
class Memoire {
  constructor(refuser = () => null, depart = {}) { this.m = new Map(Object.entries(depart)); this.refuser = refuser; }
  get length() { return this.m.size; }
  key(i) { return [...this.m.keys()][i] ?? null; }
  getItem(k) { return this.m.has(k) ? this.m.get(k) : null; }
  setItem(k, v) { const e = this.refuser(k); if (e) throw e; this.m.set(k, String(v)); }
  removeItem(k) { this.m.delete(k); }
}
globalThis.localStorage = new Memoire();
const base = %s;
const E = await import(base + "equipe.js"), J = await import(base + "journal.js"), P = await import(base + "projets.js"), N = await import(base + "notes.js");
const out = {};
const jouer = (nom, memoire, f) => { globalThis.localStorage = memoire; try { out[nom] = f(memoire); } catch (e) { out[nom] = { erreur: String(e && e.stack || e) }; } };
const T = 1762862400000;                         // un instant fixe pour les heures du journal

// ---- les initiales
out.initiales = {
  valides: ["lle", "L.L.E", " a b ", "ÉB", "J2", "ABCDE", "ABCDEF", "", "a-b_c", "<b>", "L L"].map((t) => E.initialesValides(t)),
  proposees: [["Louis Lejeune", []], ["Jean-Marc Dupont", []], ["Louis", []], ["Anne Van den Berg", []], ["Louis Lejeune", ["LLE"]], ["Élodie Brun", []], ["  ", []],
              ["Louis Lejeune", ["LLE", "LL", "LLJ", "LOL", "LLEJ"]]].map(([n, p]) => E.proposerInitiales(n, p)),
  emails: ["a@b.be", "a@b", "a b@c.be", "", "louis.lejeune@bureau.example"].map((t) => E.emailValide(t)),
};

// ---- l'équipe : se présenter, ajouter, refuser, modifier, retirer
jouer("equipe", new Memoire(), (mem) => {
  const avant = { etat: E.equipe().etat, moi: E.moi(), sig: E.signature() };
  const a = E.ajouter({ nom: "  Louis   Lejeune ", email: "louis@bureau.example" }, { estMoi: true });
  const b = E.ajouter({ nom: "Marie Durand", initiales: "md" });
  const refus = {
    sansNom: E.ajouter({ nom: " " }), initialesPrises: E.ajouter({ nom: "Lucie Lebon", initiales: "LLE" }),
    initialesLongues: E.ajouter({ nom: "Paul", initiales: "ABCDEF" }), email: E.ajouter({ nom: "Paul Martin", email: "paul@" }),
    emailPris: E.ajouter({ nom: "Luc Petit", email: "LOUIS@bureau.example" }),
  };
  const propose = E.ajouter({ nom: "Lucie Lebon" });                // LLE est pris : une autre proposition, libre
  const modif = E.modifier(b.membre.id, { initiales: "MDU", couleur: 5 }), modifRefus = E.modifier(b.membre.id, { initiales: "LLE" });
  const eq = E.equipe();
  const sig = E.signature();
  const retire = E.retirer(propose.membre.id), apres = E.equipe().membres.map((m) => m.initiales);
  const autre = E.choisirMoi(b.membre.id), sig2 = E.signature();
  E.retirer(b.membre.id);
  return { avant, a, b, refus, propose: propose.membre && propose.membre.initiales, modif: modif.ok && modif.membre, modifRefus,
           noms: eq.membres.map((m) => [m.nom, m.initiales, m.couleur]), moi: eq.moi && eq.moi.initiales, sig, retire, apres, autre, sig2,
           moiRetire: E.moi(), cles: [...mem.m.keys()] };
});

// ---- la pastille : des initiales dans un rond, un nom piégé rendu comme du texte, « ? » sans signature
out.pastille = { moi: E.pastille({ id: "x", ini: "LLE", nom: "Louis Lejeune", c: 3 }), piege: E.pastille({ ini: "AB", nom: '<img src=x onerror=alert(1)>"', c: 99 }),
                 inconnu: E.pastille(null), membre: E.pastille({ id: "m", nom: "Marie Durand", initiales: "MD", couleur: 2 }, "lg"),
                 refuse: E.pastille({ ini: "<script>", nom: "X", c: 1 }) };

// ---- une note signée, et sa ligne au journal ; le journal ne se répète pas
jouer("notes", new Memoire(), (mem) => {
  const anonyme = N.nouvelle({ module: "poutre", nom: "Sans auteur", chantier: "Réalex" });
  N.ecrire(anonyme);
  E.ajouter({ nom: "Louis Lejeune" }, { estMoi: true });
  const n = N.nouvelle({ module: "poutre", nom: "Poutres R+1", chantier: "Réalex", dossier: "Étude de la dalle" });
  const ok = N.ecrire(n);
  const cree = J.lire().map((e) => [e.k, e.q && e.q.ini, e.n && e.n.nom, e.p, e.s]);
  // dix modifications à la suite : aucune ligne de plus (créée à l'instant, la ligne dit déjà tout)
  for (let i = 0; i < 10; i++) { n.etat = { i }; N.ecrire(n); }
  const apresModifs = J.lire().length;
  // rouvrir sans modifier : ni date, ni ligne, ni « par »
  const relue = N.lire(n.id), parAvant = relue.par && relue.par.ini;
  N.ecrire(relue, { date: false });
  // renommer, ranger ailleurs
  relue.nom = "Poutres R+2"; N.ecrire(relue);
  relue.dossier = "Fondations"; N.ecrire(relue);
  const gestes = J.lire().slice(0, 2).map((e) => [e.k, e.n.nom, e.s, e.de]);
  // (l'export télécharge un fichier : il a besoin d'un document — joué par le harnais)
  const sup = N.supprimer(n.id);
  const fin = J.lire().map((e) => e.k);
  const stockee = JSON.parse(mem.getItem("stab.notes.v1"));
  return { ok, cree, apresModifs, parAvant, gestes, sup, fin, auteurAnonyme: stockee[anonyme.id].auteur || null, cles: [...mem.m.keys()].sort() };
});

// ---- le journal : une modification plus tard (une autre demi-heure) fait une ligne ; puis les suivantes l'avancent
jouer("journal", new Memoire(), () => {
  E.ajouter({ nom: "Louis Lejeune" }, { estMoi: true });
  const note = { id: "n1", nom: "Dalle", module: "dalle", chantier: "Réalex", dossier: "" };
  J.noter("note.creee", { note, quand: T });
  const a = J.noter("note.modifiee", { note, quand: T + 5 * 60000 });              // 5 min après la création : rien de plus
  const b = J.noter("note.modifiee", { note, quand: T + 45 * 60000 });             // 45 min : une ligne
  const c = J.noter("note.modifiee", { note, quand: T + 45 * 60000 + 20000 });     // 20 s après : silence (pas d'écriture)
  const d = J.noter("note.modifiee", { note, quand: T + 60 * 60000 });             // 15 min après : la même ligne, avancée
  const lignes = J.lire().map((e) => [e.k, (e.t - T) / 60000]);
  // un autre auteur sur la même note : sa propre ligne
  const m2 = E.ajouter({ nom: "Marie Durand" });
  E.choisirMoi(m2.membre.id);
  J.noter("note.modifiee", { note: { ...note, id: "n1" }, quand: T + 62 * 60000 });
  const deux = J.lire().slice(0, 2).map((e) => [e.k, e.q.ini]);
  // un geste inconnu, une ligne mal formée dans le stockage : ignorés
  const inconnu = J.noter("note.volee", { note });
  const brut = JSON.parse(localStorage.getItem("stab.journal.v1"));
  brut.unshift({ t: "hier", k: "note.creee" }, null, { t: T, k: "projet.cree", p: "<b>X</b>", q: { ini: "<i>", nom: "x" } });
  localStorage.setItem("stab.journal.v1", JSON.stringify(brut));
  const relu = J.lire();
  const filtres = { projet: J.lire({ projet: "réalex" }).length, autre: J.lire({ projet: "Autre" }).length, membre: J.lire({ membre: m2.membre.id }).length, note: J.lire({ note: "n1" }).length, max: J.lire({ max: 2 }).length };
  const phrases = relu.slice(0, 4).map((e) => J.phrase(e));
  // le plafond : au-delà de MAX lignes, les plus anciennes tombent
  for (let i = 0; i < J.MAX + 30; i++) J.noter("projet.cree", { projet: `P${i}`, quand: T + 100 * 60000 + i });
  return { a, b, c, d, lignes, deux, inconnu, premier: [relu[0].k, relu[0].p, relu[0].q], n: relu.length, filtres, phrases, plafond: J.lire().length, max: J.MAX,
           temps: [J.ilYA(T - 20000, T), J.ilYA(T - 5 * 60000, T), J.ilYA(T - 3 * 3600000, T), J.jourDe(T, T), J.jourDe(T - 86400000, T)] };
});

// ---- le journal ne fait jamais échouer : stockage plein → la note s'écrit, le journal non, rien ne lève
jouer("journal_plein", new Memoire((k) => (k === "stab.journal.v1" ? new DOMException("plein", "QuotaExceededError") : null)), (mem) => {
  const n = N.nouvelle({ module: "poutre", nom: "Malgré tout" });
  return { ok: N.ecrire(n), note: !!N.lire(n.id), journal: mem.getItem("stab.journal.v1"), noter: J.noter("projet.cree", { projet: "X" }) };
});

// ---- les projets : ceux des notes, ceux du registre ; créer, sous-projets, renommer (les notes suivent), supprimer
jouer("projets", new Memoire(), (mem) => {
  E.ajouter({ nom: "Louis Lejeune" }, { estMoi: true });
  const note = (nom, chantier, dossier, modifie) => { const n = N.nouvelle({ module: "poutre", nom, chantier, dossier }); N.ecrire(n); const x = N.lire(n.id); x.modifie = modifie; N.ecrire(x, { date: false }); return n.id; };
  const a = note("Poutres R+1", "Réalex", "Étude de la dalle", T + 3000), b = note("Dalle R+1", "réalex ", "étude  de la dalle", T + 4000), c = note("Garde-corps", "Réalex", "", T + 1000);
  const d = note("Libre", "", "", T + 500), e = note("Semelles", "Tour Sud", "Fondations", T + 2000);
  const m0 = P.modele();
  const vue = (m) => m.projets.map((p) => [p.nom, p.nb, p.connu, p.notes.map((n) => n.nom), p.sous.map((s) => [s.nom, s.notes.map((n) => n.nom)]), p.auteurs.map((g) => g.ini)]);
  const depart = { vue: vue(m0), sans: m0.sans.map((n) => n.nom), nb: m0.nbNotes };
  const cree = P.creer("  École   des Tilleuls "), encore = P.creer("école des tilleuls"), videNom = P.creer("   ");
  const adopte = P.creer("RÉALEX");                            // un projet que les notes nommaient : il entre au registre, sans doublon
  const sous = [P.creerSous("École des Tilleuls", "Charpente"), P.creerSous("École des Tilleuls", "charpente "), P.creerSous("École des Tilleuls", " "), P.creerSous("Inconnu", "X")];
  const apresCreation = vue(P.modele());
  // renommer : les notes suivent ; un nom déjà pris est refusé ; changer la casse est permis
  const pris = P.renommer("Réalex", "Tour Sud"), ren = P.renommer("réalex", "Résidence Réalex"), casse = P.renommer("Tour Sud", "TOUR SUD");
  const chantiers = [a, b, c, e].map((id) => N.lire(id).chantier);
  const dates = [a, b].map((id) => N.lire(id).modifie - T);       // renommer ne date pas les notes
  const rs = P.renommerSous("Résidence Réalex", "Étude de la dalle", "Dalle R+1 à R+3"), rsPris = P.creerSous("Résidence Réalex", "Voiles") && P.renommerSous("Résidence Réalex", "Voiles", "dalle r+1 à r+3");
  const dossiers = [a, b].map((id) => N.lire(id).dossier);
  // déplacer une note ; supprimer : refusé tant qu'il reste des notes
  const dep = P.deplacer(d, "École des Tilleuls", "Charpente"), depNote = [N.lire(d).chantier, N.lire(d).dossier, N.lire(d).modifie - T];
  const supSous = [P.supprimerSous("École des Tilleuls", "Charpente"), P.supprimerSous("Résidence Réalex", "Voiles")];
  const sup = [P.supprimer("Résidence Réalex"), P.supprimer("Tour Sud")];
  P.deplacer(d, "", "");
  const supVide = [P.supprimerSous("École des Tilleuls", "Charpente"), P.supprimer("École des Tilleuls"), P.supprimer("Jamais créé")];
  const arch = P.archiver("TOUR SUD"), archive = P.trouver("tour sud").archive, rouv = P.archiver("Tour Sud", false);
  const fin = vue(P.modele());
  const journal = J.lire().map((x) => [x.k, x.p, x.s, x.de].filter(Boolean).join(" | "));
  return { depart, cree, encore, videNom, adopte, sous, apresCreation, pris, ren, casse, chantiers, dates, rs, rsPris, dossiers, dep, depNote, supSous, sup, supVide,
           arch, archive, rouv, fin, journal, cles: [...mem.m.keys()].sort() };
});

// ---- un registre ABÎMÉ n'est jamais écrasé sans copie ; si la copie échoue, rien n'est écrit
const abime = '{"realex":{"nom":"Réalex","sous":["A"';
jouer("abime", new Memoire(() => null, { "stab.projets.v1": abime, "stab.equipe.v1": "[1,2]" }), (mem) => {
  const lus = { projets: P.modele().projets.length, equipe: E.equipe().etat };
  const c = P.creer("Nouveau"), a = E.ajouter({ nom: "Louis Lejeune" });
  const copies = [...mem.m.keys()].filter((k) => k.includes(".illisible."));
  return { lus, c: c.ok, a: a.ok, copies: copies.map((k) => k.split(".illisible.")[0]).sort(), textes: copies.map((k) => mem.getItem(k)).sort(), apres: P.modele().projets.map((p) => p.nom) };
});
jouer("abime_sans_copie", new Memoire((k) => (k.includes(".illisible.") ? new DOMException("plein", "QuotaExceededError") : null), { "stab.projets.v1": abime }), (mem) => {
  const c = P.creer("Nouveau");
  return { ok: c.ok, erreur: c.erreur, intact: mem.getItem("stab.projets.v1") === abime, cles: [...mem.m.keys()] };
});

// ---- un .stab reçu : la signature de son auteur est relue comme des DONNÉES (textes bornés), celui qui importe signe
globalThis.localStorage = new Memoire();
E.ajouter({ nom: "Marie Durand" }, { estMoi: true });
const fichier = (auteur) => ({ name: "x.stab", text: async () => JSON.stringify({ format: "stab", version: 1, note: { module: "poutre", nom: "Reçue", chantier: "Réalex", etat: { poutres: [] }, auteur } }) });
const r1 = await N.importer(fichier({ id: "ailleurs", ini: "lle", nom: "Louis Lejeune", c: 4 }));
const r2 = await N.importer(fichier({ ini: "<script>alert(1)</script>", nom: "<img onerror=x>", c: "rouge" }));
const r3 = await N.importer(fichier({ ini: "AB", nom: "x".repeat(500), c: 42, html: "<b>" }));
out.importer = { auteur: r1.auteur, par: r1.par && r1.par.ini, piege: r2.auteur, borne: [r3.auteur.ini, r3.auteur.nom.length, r3.auteur.c, Object.keys(r3.auteur).sort()],
                 journal: J.lire().slice(0, 3).map((e) => [e.k, e.q.ini]) };
console.log(JSON.stringify(out));
""" % json.dumps("file://" + JS + "/")

try:
    sortie = subprocess.run(["node", "--input-type=module", "-e", ESSAI], capture_output=True, text=True, timeout=120)
    D = json.loads(sortie.stdout.strip().split("\n")[-1])
except Exception as e:  # noqa: BLE001
    D = {"_": str(e), "stderr": locals().get("sortie") and sortie.stderr[-1200:]}
chk("les scripts des données se chargent dans node et toutes les scènes se jouent (aucune exception)",
    "_" not in D and not [k for k, v in D.items() if isinstance(v, dict) and "erreur" in v and k not in ("abime_sans_copie",)]
    and "erreur" not in str(D.get("equipe", {}).get("erreur", "")), {k: v for k, v in D.items() if k == "_" or k == "stderr" or (isinstance(v, dict) and v.get("erreur") and k != "abime_sans_copie")})

print("\n=== 1 · L'équipe")
I = D.get("initiales", {})
chk("initiales : 1 à 5 lettres ou chiffres, mises en capitales, points et espaces retirés ; six lettres, rien, ou une balise : refusées",
    I.get("valides") == ["LLE", "LLE", "AB", "ÉB", "J2", "ABCDE", "", "", "ABC", "", "LL"], I.get("valides"))
chk("initiales PROPOSÉES d'après le nom : Louis Lejeune → LLE (l'initiale du prénom, deux lettres du nom), Jean-Marc Dupont → JMD, un seul mot → "
    "trois lettres ; des initiales déjà prises : une autre proposition, libre",
    I.get("proposees", [None] * 8)[:4] == ["LLE", "JMD", "LOU", "AVDB"] and I["proposees"][4] not in ("LLE", "") and I["proposees"][5] == "ÉBR"
    and I["proposees"][6] == "" and I["proposees"][7] not in ("LLE", "LL", "LLJ", "LOL", "LLEJ", ""), I.get("proposees"))
chk("e-mail : une adresse plausible (a@b.be) ; « a@b », un espace, rien : non", I.get("emails") == [True, False, False, False, True], I.get("emails"))
Q = D.get("equipe", {})
chk("équipe vide : personne ne s'est présenté — pas de « moi », pas de signature", Q.get("avant") == {"etat": "absente", "moi": None, "sig": None}, Q.get("avant"))
chk("se présenter : le nom (espaces resserrés), l'e-mail, les initiales proposées (LLE), une teinte ; la signature porte les initiales et le nom, pas l'e-mail",
    (Q.get("a") or {}).get("ok") is True and Q["a"]["membre"]["nom"] == "Louis Lejeune" and Q["a"]["membre"]["initiales"] == "LLE"
    and Q["a"]["membre"]["email"] == "louis@bureau.example" and Q.get("moi") == "LLE"
    and set(Q.get("sig") or {}) == {"id", "ini", "nom", "c"} and Q["sig"]["ini"] == "LLE" and Q["sig"]["nom"] == "Louis Lejeune", (Q.get("a"), Q.get("sig")))
R = Q.get("refus", {})
chk("fiches refusées, avec le champ et la raison : sans nom ; initiales déjà prises (dit par qui) ; six lettres ; e-mail incomplet ; e-mail déjà pris (sans la casse)",
    all(R.get(k, {}).get("ok") is False for k in ("sansNom", "initialesPrises", "initialesLongues", "email", "emailPris"))
    and R["sansNom"]["champ"] == "nom" and R["initialesPrises"]["champ"] == "initiales" and "Louis Lejeune" in R["initialesPrises"]["erreur"]
    and R["initialesLongues"]["champ"] == "initiales" and R["email"]["champ"] == "email" and R["emailPris"]["champ"] == "email", R)
chk("deux membres ne portent jamais les mêmes initiales : « Lucie Lebon » reçoit une proposition autre que LLE ; modifier vers LLE est refusé",
    Q.get("propose") not in (None, "", "LLE") and (Q.get("modifRefus") or {}).get("ok") is False and (Q.get("modif") or {}).get("initiales") == "MDU"
    and Q["modif"]["couleur"] == 5, (Q.get("propose"), Q.get("modif"), Q.get("modifRefus")))
chk("les membres sont triés par nom, chacun sa teinte (deux membres, deux teintes) ; retirer un membre l'ôte de la liste",
    [x[0] for x in Q.get("noms", [])] == sorted(x[0] for x in Q.get("noms", [])) and len(Q.get("noms", [])) == 3
    and len({x[2] for x in Q["noms"][:2]}) == 2 and Q.get("retire") is True and len(Q.get("apres", [])) == 2, (Q.get("noms"), Q.get("apres")))
chk("« Je suis… » : un autre membre devient la personne du poste (sa signature) ; le retirer : plus personne",
    Q.get("autre") is True and (Q.get("sig2") or {}).get("ini") == "MDU" and Q.get("moiRetire") is None and Q.get("cles") == ["stab.equipe.v1"], (Q.get("sig2"), Q.get("moiRetire"), Q.get("cles")))
PA = D.get("pastille", {})
chk("la pastille : les initiales dans un rond de la teinte du membre, son nom en infobulle ; « ? » pour un auteur inconnu",
    'class="avatar c3 n3"' in PA.get("moi", "") and ">LLE</span>" in PA["moi"] and 'title="Louis Lejeune"' in PA["moi"]
    and 'class="avatar c2 n2 lg"' in PA.get("membre", "") and ">MD</span>" in PA["membre"] and "inconnu" in PA.get("inconnu", "") and ">?</span>" in PA["inconnu"], PA)
chk("pastille piégée : un nom « <img onerror> » est écrit comme du TEXTE (échappé), une teinte hors palette revient à 0, des initiales « <script> » : auteur inconnu",
    "<img" not in PA.get("piege", "<img") and "&lt;img" in PA["piege"] and 'class="avatar c0 n2"' in PA["piege"] and "inconnu" in PA.get("refuse", "") and "<script" not in PA["refuse"], PA)

print("\n=== 2 · La signature d'une note et le journal")
NT = D.get("notes", {})
chk("une note créée est SIGNÉE (auteur) et laisse UNE ligne : « LLE a créé la note Poutres R+1 », avec son projet et son sous-projet ; avant qu'on se présente : sans auteur",
    NT.get("ok") is True and NT.get("cree", [[None]])[0] == ["note.creee", "LLE", "Poutres R+1", "Réalex", "Étude de la dalle"]
    and NT["cree"][-1][:2] == ["note.creee", None] and NT.get("auteurAnonyme") is None, NT.get("cree"))
chk("dix modifications dans la foulée de la création : aucune ligne de plus ; rouvrir sans modifier : rien non plus (la note garde son dernier auteur)",
    NT.get("apresModifs") == 2 and NT.get("parAvant") == "LLE", (NT.get("apresModifs"), NT.get("parAvant")))
chk("renommer la note, la ranger dans un autre sous-projet : deux lignes (« a renommé… (avant : Poutres R+1) », « a rangé… ») ; la supprimer : une ligne",
    NT.get("gestes") == [["note.classee", "Poutres R+2", "Fondations", ""], ["note.renommee", "Poutres R+2", "Étude de la dalle", "Poutres R+1"]]
    and NT.get("sup") is True and NT.get("fin", [None])[0] == "note.supprimee" and NT.get("cles") == ["stab.equipe.v1", "stab.journal.v1", "stab.notes.v1"], (NT.get("gestes"), NT.get("fin"), NT.get("cles")))
JO = D.get("journal", {})
chk("le journal ne se répète pas : modifier 5 min après la création n'ajoute rien ; 45 min après : une ligne ; 20 s plus tard : silence ; "
    "15 min plus tard : la MÊME ligne, avancée à la dernière heure",
    [JO.get(k) for k in "abcd"] == [True, True, False, True] and JO.get("lignes") == [["note.modifiee", 60], ["note.creee", 0]], ([JO.get(k) for k in "abcd"], JO.get("lignes")))
chk("un autre auteur sur la même note a SA ligne ; un geste inconnu n'est pas noté",
    JO.get("deux") == [["note.modifiee", "MDU"], ["note.modifiee", "LLE"]] and JO.get("inconnu") is False, (JO.get("deux"), JO.get("inconnu")))
chk("un journal réécrit à la main : les lignes mal formées sont ignorées, une signature « <i> » devient un auteur inconnu, un nom « <b>X</b> » reste un texte",
    JO.get("premier") == ["projet.cree", "<b>X</b>", None] and JO.get("n") == 4, (JO.get("premier"), JO.get("n")))
chk("le journal se filtre : par projet (sans la casse), par membre, par note ; « max » le borne",
    JO.get("filtres") == {"projet": 3, "autre": 0, "membre": 1, "note": 3, "max": 2}, JO.get("filtres"))
PH = JO.get("phrases", [{}] * 4)
chk("ce qu'une ligne dit : « Marie Durand · a modifié la note · Dalle · Réalex » ; sans auteur : « Quelqu'un »",
    PH[0].get("qui") == "Quelqu'un" and PH[0].get("verbe") == "a créé le projet" and PH[1].get("qui") == "Marie Durand"
    and PH[1].get("verbe") == "a modifié la note" and PH[1].get("objet") == "Dalle" and PH[1].get("lieu") == "Réalex", PH)
chk("au-delà de 400 lignes, les plus anciennes tombent ; les heures se disent « à l'instant », « il y a 5 min », « il y a 3 h », « Aujourd'hui », « Hier »",
    JO.get("plafond") == JO.get("max") == 400 and JO.get("temps") == ["à l'instant", "il y a 5 min", "il y a 3 h", "Aujourd'hui", "Hier"], (JO.get("plafond"), JO.get("temps")))
JP = D.get("journal_plein", {})
chk("le journal ne peut JAMAIS faire échouer une note : stockage du journal refusé → la note s'écrit quand même, rien ne lève, noter rend faux",
    JP == {"ok": True, "note": True, "journal": None, "noter": False}, JP)

print("\n=== 3 · Les projets")
PR = D.get("projets", {})
dep = PR.get("depart", {})
chk("les projets viennent des NOTES déjà là (leur « chantier ») : Réalex (3 notes : une sans sous-projet, deux dans « Étude de la dalle », malgré la casse "
    "et les espaces) et Tour Sud ; le plus récemment actif d'abord ; la note sans projet est à part",
    [p[0] for p in dep.get("vue", [])] == ["Réalex", "Tour Sud"] and dep["vue"][0][1] == 3 and dep["vue"][0][2] is False
    and dep["vue"][0][3] == ["Garde-corps"] and dep["vue"][0][4] == [["Étude de la dalle", ["Dalle R+1", "Poutres R+1"]]] and dep["vue"][0][5] == ["LLE"]
    and dep.get("sans") == ["Libre"] and dep.get("nb") == 5, dep)
chk("créer un projet : son nom suffit (espaces resserrés) ; le recréer (autre casse) ne fait PAS de doublon ; un nom vide est refusé ; "
    "un projet que seules des notes nommaient entre au registre sans doublon",
    PR.get("cree") == {"ok": True, "existe": False, "nom": "École des Tilleuls"} and PR.get("encore") == {"ok": True, "existe": True, "nom": "École des Tilleuls"}
    and (PR.get("videNom") or {}).get("ok") is False and PR.get("adopte") == {"ok": True, "existe": True, "nom": "Réalex"}
    and [p[0] for p in PR.get("apresCreation", [])].count("Réalex") == 1, (PR.get("cree"), PR.get("encore"), PR.get("adopte")))
sous = PR.get("sous", [{}] * 4)
chk("sous-projets : créer « Charpente » (vide, il existe quand même) ; le recréer : pas de doublon ; nom vide ou projet inconnu : refusés",
    sous[0] == {"ok": True, "nom": "Charpente"} and sous[1].get("existe") is True and sous[2].get("ok") is False and sous[3].get("ok") is False
    and [p for p in PR.get("apresCreation", []) if p[0] == "École des Tilleuls"][0][4] == [["Charpente", []]], (sous, PR.get("apresCreation")))
chk("renommer un projet réécrit le nom dans TOUTES ses notes, sans les dater ; un nom déjà porté par un autre projet est refusé (pas de fusion par mégarde) ; "
    "changer seulement la casse est permis",
    (PR.get("pris") or {}).get("ok") is False and "existe déjà" in PR["pris"]["erreur"] and (PR.get("ren") or {}).get("ok") is True
    and PR.get("chantiers") == ["Résidence Réalex"] * 3 + ["TOUR SUD"] and PR.get("dates") == [3000, 4000] and (PR.get("casse") or {}).get("ok") is True,
    (PR.get("pris"), PR.get("chantiers"), PR.get("dates")))
chk("renommer un sous-projet réécrit le « dossier » de ses notes ; un nom déjà pris dans le projet est refusé",
    (PR.get("rs") or {}).get("ok") is True and PR.get("dossiers") == ["Dalle R+1 à R+3"] * 2 and (PR.get("rsPris") or {}).get("ok") is False, (PR.get("rs"), PR.get("dossiers"), PR.get("rsPris")))
chk("ranger une note dans un projet et un sous-projet ne la date pas (elle ne remonte pas dans « Reprendre »)",
    (PR.get("dep") or {}).get("ok") is True and PR.get("depNote") == ["École des Tilleuls", "Charpente", 500], (PR.get("dep"), PR.get("depNote")))
chk("on ne perd JAMAIS une note en supprimant : un sous-projet ou un projet qui a encore des notes ne se supprime pas (la raison est dite) ; vides, oui",
    PR.get("supSous", [{}])[0].get("ok") is False and "1 note" in PR["supSous"][0]["erreur"] and PR["supSous"][1].get("ok") is True
    and all(x.get("ok") is False for x in PR.get("sup", [{}])) and "3 notes" in PR["sup"][0]["erreur"]
    and all(x.get("ok") is True for x in PR.get("supVide", [{}])), (PR.get("supSous"), PR.get("sup"), PR.get("supVide")))
chk("archiver un projet le marque (ses notes restent) ; le rouvrir le rend",
    (PR.get("arch") or {}).get("ok") is True and PR.get("archive") is True and (PR.get("rouv") or {}).get("ok") is True
    and [p[0] for p in PR.get("fin", [])] == ["Résidence Réalex", "TOUR SUD"], (PR.get("archive"), PR.get("fin")))
jr = PR.get("journal", [])
chk("chaque geste sur un projet laisse sa ligne au journal : créé, sous-projet créé, renommé (avec l'ancien nom), archivé, rouvert, supprimé",
    all(any(x.startswith(k) for x in jr) for k in ("projet.cree | École des Tilleuls", "sous.cree | École des Tilleuls | Charpente",
                                                    "projet.renomme | Résidence Réalex | Réalex", "sous.renomme | Résidence Réalex | Dalle R+1 à R+3 | Étude de la dalle",
                                                    "projet.archive | TOUR SUD", "projet.rouvert | TOUR SUD", "projet.supprime | École des Tilleuls", "sous.supprime"))
    and PR.get("cles") == ["stab.equipe.v1", "stab.journal.v1", "stab.notes.v1", "stab.projets.v1"], (jr[:14], PR.get("cles")))
AB = D.get("abime", {})
chk("un registre ABÎMÉ (projets tronqués, équipe remplacée par un tableau) se lit vide ; écrire dessus COPIE d'abord son texte à part (…illisible.<heure>)",
    AB.get("lus") == {"projets": 0, "equipe": "illisible"} and AB.get("c") is True and AB.get("a") is True
    and AB.get("copies") == ["stab.equipe.v1", "stab.projets.v1"] and AB.get("textes") == sorted(['{"realex":{"nom":"Réalex","sous":["A"', "[1,2]"])
    and AB.get("apres") == ["Nouveau"], AB)
AS = D.get("abime_sans_copie", {})
chk("si la copie échoue, RIEN n'est écrit : le registre abîmé reste intact, le geste est refusé et le dit",
    AS.get("ok") is False and "pas pu être enregistré" in str(AS.get("erreur")) and AS.get("intact") is True and AS.get("cles") == ["stab.projets.v1"], AS)
IM = D.get("importer", {})
chk("un .stab reçu : la signature de son auteur est gardée (LLE, Louis Lejeune) et celui qui l'importe signe la dernière main ; le journal dit « a importé »",
    (IM.get("auteur") or {}).get("ini") == "LLE" and IM["auteur"]["nom"] == "Louis Lejeune" and IM["auteur"]["c"] == 4 and IM.get("par") == "MDU"
    and IM.get("journal") == [["note.importee", "MDU"]] * 3, IM)
chk("un .stab piégé : des initiales « <script> » → pas d'auteur gardé (celui qui importe devient l'auteur) ; un nom de 500 caractères est borné, "
    "une teinte hors palette revient à 0, aucune autre clé n'est gardée",
    (IM.get("piege") or {}).get("ini") == "MDU" and IM.get("borne") == ["AB", 80, 0, ["c", "id", "ini", "nom"]], (IM.get("piege"), IM.get("borne")))

# ================================================================================================ 4 · les sources
print("\n=== 4 · Les sources")
for f in ("memoire.js", "equipe.js", "journal.js", "projets.js"):
    s = source(f)
    chk(f"js/{f} : aucun effet au chargement et rien du document (ni document, ni window, ni history) — il tourne dans node",
        not re.search(r"\b(document|window|history)\.", s), re.findall(r"\b(?:document|window|history)\.\w+", s)[:4])
NJ = source("notes.js")
chk("js/notes.js : toute écriture des notes passe par changer() — ecrire, supprimer, reclasser — et chacune laisse sa trace au journal sans en dépendre",
    len(re.findall(r"localStorage\.setItem\(CLE", NJ)) == 1 and len(re.findall(r"= changer\(", NJ)) == 3 and "J.noter(" in NJ
    and re.search(r"if \(ok\) \{[^}]*J\.noter", NJ, re.S) is not None, re.findall(r"\w+ changer\(", NJ))
chk("js/journal.js : noter() ne lève jamais (tout son corps est gardé) et borne le journal", re.search(r"export function noter\(.*\) \{\n  try \{", source("journal.js")) is not None
    and "j.length = MAX" in source("journal.js"))

# ================================================================================================ 5 · les pages
print("\n=== 5 · Les pages (site/web/construire.py)")
PAGES_R = SW.pages_rendues()
_mut_page = os.environ.get("BUREAU_PAGE")          # une mutation des pages : « fichier|avant|après », appliquée ici
if _mut_page:
    _f, _av, _ap = _mut_page.split("|", 2)
    assert _av in PAGES_R[_f], (_f, _av)
    PAGES_R = {**PAGES_R, _f: PAGES_R[_f].replace(_av, _ap)}
ACC, PRJ, EQU = PAGES_R["index.html"], PAGES_R.get("projets.html", ""), PAGES_R.get("equipe.html", "")


def ids(html):
    return re.findall(r'\bid="([^"]+)"', html)


def corps(html):
    return dict(re.findall(r'\s([\w-]+)="([^"]*)"', re.search(r"<body[^>]*>", html).group(0))) if "<body" in html else {}


chk("les pages Projets et Équipe existent (projets.html, equipe.html), le corps dit leur écran (data-page) — 24 pages au site",
    len(PAGES_R) == 24 and corps(PRJ) == {"data-page": "projets"} and corps(EQU) == {"data-page": "equipe"}
    and SW.PAGES_BUREAU == {"projets": "projets.html", "equipe": "equipe.html"}, (len(PAGES_R), corps(PRJ), corps(EQU)))
# le MENU : depuis CHAQUE page, la barre (PC, tablette) et les onglets du téléphone (là où il y en a) mènent aux projets
# et à l'équipe — des LIENS, plus « bientôt » ; Paramètres, lui, attend encore les comptes
# (une page de REDIRECTION — predalle — n'a ni barre ni écran : elle renvoie à une autre)
_rail = {n: re.search(r'<nav class="nav">.*?</nav>', h, re.S).group(0) for n, h in PAGES_R.items() if '<nav class="nav">' in h}
_mnav = {n: re.search(r'<nav class="mnav">.*?</nav>', h, re.S).group(0) for n, h in PAGES_R.items() if '<nav class="mnav">' in h}
_rail_ko = [n for n, r in _rail.items() if not ('data-tip="Projets" href="projets"' in r and 'data-tip="Équipe" href="equipe"' in r
                                                and "Projets · bientôt" not in r and "Équipe · bientôt" not in r and r.count('data-bientot="1"') == 1
                                                and 'data-tip="Paramètres · bientôt" aria-disabled="true" data-bientot="1"' in r)]
_mnav_ko = [n for n, r in _mnav.items() if not (re.search(r'<a class="(on)?" href="projets">', r) and re.search(r'<a class="(on)?" href="equipe">', r)
                                                and r.count('data-bientot="1"') == 1 and len(re.findall(r"<a ", r)) == 5)]
chk(f"le menu mène aux Projets et à l'Équipe depuis les {len(_rail)} pages (la barre) et depuis les {len(_mnav)} pages qui ont les onglets du "
    "téléphone : des liens, plus « bientôt » — seul Paramètres attend encore",
    len(_rail) == 24 - len(SW.REDIRECTIONS) and not _rail_ko and len(_mnav) == 6 and not _mnav_ko, (len(_rail), _rail_ko, _mnav_ko))
chk("chaque page du bureau allume SON entrée du menu (et elle seule) : Accueil, Projets, Équipe",
    all(len(re.findall(r'class="navitem on"', _rail[n])) == 1 and f'class="navitem on" data-tip="{nom}"' in _rail[n]
        and re.search(rf'<a class="on" href="{href}">', _mnav[n]) and len(re.findall(r'<a class="on"', _mnav[n])) == 1
        for n, nom, href in (("index.html", "Accueil", r"\./"), ("projets.html", "Projets", "projets"), ("equipe.html", "Équipe", "equipe"))),
    {n: re.findall(r'<a class="(?:navitem )?on"[^>]*>', _rail[n] + _mnav[n]) for n in ("index.html", "projets.html", "equipe.html")})


def une_fois(html, noms):
    i = ids(html)
    return [n for n in noms if i.count(n) != 1]


ACC_IDS = ["moi", "bonjour", "date", "ouvrir-nouvelle", "recentes", "ac-projet-form", "ac-projet", "ac-projet-creer", "chantiers", "ac-qui", "activite",
           "importer", "nouvelle", "nn-chantier", "nn-dossier", "nn-nom", "nn-creer", "nn-chantiers", "nn-dossiers", "icones", "toast"]
chk("l'ACCUEIL porte les projets (le champ qui en crée un, la liste, « Tout voir » vers la page) et l'activité (qui je suis, le journal, « Équipe ») — "
    "chaque repère une seule fois ; la pastille de la personne mène à l'équipe",
    not une_fois(ACC, ACC_IDS) and re.search(r'<a class="btn ghost sm" href="projets">Tout voir', ACC) is not None
    and re.search(r'<a class="btn ghost sm" href="equipe">Équipe', ACC) is not None and '<a class="bu-moi" id="moi" href="equipe"' in ACC
    and 'placeholder="Nouveau projet : le nom du chantier"' in ACC, une_fois(ACC, ACC_IDS))
PRJ_IDS = ["pj-fil", "importer", "ouvrir-nouvelle", "pj", "pj-arbre", "pj-champ-form", "pj-champ", "pj-champ-creer", "pj-noeuds", "pj-liste", "pj-volet",
           "nouvelle", "nn-chantier", "nn-dossier", "nn-nom", "nn-creer", "icones", "toast"]
chk("la page PROJETS : l'arbre et son champ « Chercher ou créer un projet », la liste, le volet, « Nouvelle note », l'import — chaque repère une fois ; "
    "UNE barre pour les trois formats (.top.bu-top, pas d'en-tête .mhead à part) : le fil, l'import et « Nouvelle note » restent au téléphone",
    not une_fois(PRJ, PRJ_IDS) and '<header class="top bu-top">' in PRJ and 'class="mhead"' not in PRJ
    and '<div class="proj bu-proj" id="pj" data-vue="arbre">' in PRJ and 'placeholder="Chercher ou créer un projet"' in PRJ
    and PRJ.count("data-theme-toggle") == 2, (une_fois(PRJ, PRJ_IDS), PRJ.count("data-theme-toggle")))
EQU_IDS = ["eq-nb", "eq-moi", "eq-titre", "eq-membres", "eq-form", "eq-nom", "eq-ini", "eq-email", "eq-erreur", "eq-filtre", "eq-journal", "icones", "toast"]
chk("la page ÉQUIPE : moi, les membres, le formulaire d'ajout (nom, initiales, e-mail), le journal — chaque repère une fois ; l'e-mail est un champ "
    "d'adresse, facultatif ; la page dit où l'équipe est enregistrée (ce navigateur)",
    not une_fois(EQU, EQU_IDS) and re.search(r'<input id="eq-email" type="email"[^>]*placeholder="adresse e-mail \(facultatif\)"', EQU) is not None
    and "enregistrés <b>dans ce navigateur</b>" in EQU, une_fois(EQU, EQU_IDS))
chk("« Nouvelle note » (accueil et projets, la même fenêtre) : Projet, Sous-projet, Nom de la note — les projets et sous-projets connus sont proposés",
    all(re.search(r'<label><span>Projet</span><input id="nn-chantier" list="nn-chantiers"', h) and re.search(r'<label><span>Sous-projet</span><input id="nn-dossier" list="nn-dossiers"', h)
        and "<span>Chantier</span>" not in h and "<span>Dossier</span>" not in h for h in (ACC, PRJ))
    and SW.fenetre_nouvelle_note() in ACC and SW.fenetre_nouvelle_note() in PRJ)
# les trois pages chargent la feuille du bureau, après la charte et site.css ; aucun formulaire n'a d'action (la politique de
# sécurité dit form-action 'none' : tout passe par les scripts) ; aucun geste écrit dans la page
_feuilles = {n: re.findall(r'<link rel="stylesheet" href="([^"]+)">', PAGES_R[n]) for n in ("index.html", "projets.html", "equipe.html")}
chk("les trois pages du bureau chargent css/bureau.css après la charte et site.css ; leurs formulaires n'ont pas d'action et ne portent aucun geste en ligne",
    all(f == ["css/charte.css", "css/site.css", "css/bureau.css"] for f in _feuilles.values())
    and all(not re.search(r"<form[^>]*\baction=", h) and not re.search(r"\son[a-z]+=\"", h) for h in (ACC, PRJ, EQU)), _feuilles)
# les ICÔNES : celles que les écrans posent (V.icone("…")) sont déposées dans les trois pages, inertes — et dessinées
_scripts_bureau = ("accueil.js", "ecran_projets.js", "ecran_equipe.js", "vues_bureau.js", "nouvelle.js")
_demandees = sorted({m for f in _scripts_bureau for m in re.findall(r'\bicone\("([\w-]+)"\)', source(f))})
_deposees = {n: dict(re.findall(r'<template data-icone="([\w-]+)">(.*?)</template>', h, re.S)) for n, h in (("index.html", ACC), ("projets.html", PRJ), ("equipe.html", EQU))}
_modules = {n: re.findall(r'<template data-icone-module="(\w+)">', h) for n, h in (("index.html", ACC), ("projets.html", PRJ), ("equipe.html", EQU))}
chk(f"les {len(_demandees)} icônes que les écrans du bureau posent sont déposées dans les trois pages (des <template> inertes), chacune un dessin ; "
    "le pictogramme de chaque module aussi",
    len(_demandees) >= 15 and all(set(_demandees) <= set(d) and all("<svg" in v for v in d.values()) for d in _deposees.values())
    and all(m == list(SW.PAGES) for m in _modules.values()) and set(_demandees) <= set(SW.ICONES_BUREAU),
    {n: sorted(set(_demandees) - set(d)) for n, d in _deposees.items()})

# ================================================================================================ 6 · les copies liées (R2)
print("\n=== 6 · Les copies liées")
APP = source("app.js")
EQ, PJ, JO_, VB, EP, EE = source("equipe.js"), source("projets.js"), source("journal.js"), source("vues_bureau.js"), source("ecran_projets.js"), source("ecran_equipe.js")
BU, SITE_CSS = feuille("bureau.css"), feuille("site.css")
_app_pages = re.search(r'export const PAGES_BUREAU = \[([^\]]*)\];', APP)
chk("les pages du bureau : construire.PAGES_BUREAU et js/app.js PAGES_BUREAU disent les mêmes, dans le même ordre ; l'application les monte sans moteur, "
    "chacune par son écran",
    _app_pages is not None and re.findall(r'"(\w+)"', _app_pages.group(1)) == list(SW.PAGES_BUREAU) == list(SW.ADRESSES_BUREAU)
    and 'const SANS_MOTEUR = new Set(["accueil", ...PAGES_BUREAU]);' in APP
    and re.search(r"accueil: ouvrirAccueil,\s*projets: ouvrirProjets,\s*equipe: ouvrirEquipe,", APP) is not None, _app_pages and _app_pages.group(0))
_n_teintes = int(re.search(r"export const COULEURS = (\d+);", EQ).group(1))
_teintes_css = re.findall(r"^\.avatar\.c(\d) \{ background: (#[0-9A-F]{6}); color: #fff; \}", BU, re.M)
chk(f"les teintes des pastilles : js/equipe.js en compte {_n_teintes}, css/bureau.css en peint {len(_teintes_css)} (.avatar.c0 à .avatar.c{_n_teintes - 1}), "
    "toutes différentes, texte blanc",
    _n_teintes == 8 and [int(c) for c, _ in _teintes_css] == list(range(_n_teintes)) and len({h for _, h in _teintes_css}) == _n_teintes, _teintes_css)


def luminance(hexa):
    v = [int(hexa[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    r, g, b_ = [(c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4) for c in v]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b_


_contrastes = {h: round(1.05 / (luminance(h) + 0.05), 2) for _, h in _teintes_css}
chk("les initiales se LISENT dans chaque teinte : le blanc y a un contraste de 4,5 au moins (WCAG AA)",
    bool(_contrastes) and min(_contrastes.values()) >= 4.5, _contrastes)
_ini_max = int(re.search(r"export const INITIALES_MAX = (\d+);", EQ).group(1))
_max_ini = (re.findall(r'<input id="eq-ini" maxlength="(\d+)"', EQU) + re.findall(r'name="initiales" maxlength="(\d+)"', VB)
            + re.findall(r'data-moi="initiales" value="[^"]*" maxlength="(\d+)"', EE) + re.findall(r'name="initiales" value="[^"]*" maxlength="(\d+)"', EE))
chk(f"les initiales : {_ini_max} caractères au plus, dans les données (INITIALES_MAX) comme dans les QUATRE champs où on les saisit ; la pastille a sa "
    "forme pour 1 à 5 lettres (.n3, .n4, .n5)",
    _ini_max == 5 and _max_ini == ["5"] * 4 and all(f".avatar.n{k}" in BU for k in (3, 4, 5)) and "/^[\\p{L}\\p{N}]{1,5}$/u" in EQ, _max_ini)
_nom_max = int(re.search(r"export const NOM_MAX = (\d+);", PJ).group(1))
chk(f"le nom d'un projet : {_nom_max} caractères au plus, dans les données (NOM_MAX) comme dans les champs qui le créent (accueil, arbre, « Nouvelle note ») "
    "et ceux qui le renomment",
    _nom_max == 80 and re.findall(r'<input id="(?:ac-projet|pj-champ)" maxlength="(\d+)"', ACC + PRJ) == ["80", "80"]
    and re.findall(r'<input id="nn-(?:chantier|dossier)" list="[\w-]+" maxlength="(\d+)"', ACC) == ["80", "80"] and EP.count('maxlength="${P.NOM_MAX}"') == 3)
chk("l'adresse de connexion (Cloudflare Access) est lue sur le site MÊME (« /cdn-cgi/access/get-identity », connect-src 'self'), sans rien d'un tiers ; "
    "elle ne fait que préremplir un champ resté vide, et son absence ne dit rien",
    'export const ADRESSE_IDENTITE = "/cdn-cgi/access/get-identity";' in EQ and "http" not in re.sub(r"//.*", "", EQ)
    and "connect-src 'self'" in open(os.path.join(WEB, "_headers"), encoding="utf-8").read()
    and re.search(r"E\.emailDeConnexion\(\)\.then\(\(a\) => \{ if \(a && !signal\.aborted && form\.isConnected && !c\(\"email\"\)\.value\)", VB) is not None
    and "} catch (e) { return \"\"; }" in EQ)
chk("la signature d'une note et d'une ligne du journal ne porte JAMAIS l'e-mail (id, initiales, nom, teinte)",
    "export const signatureDe = (m) => (m ? { id: m.id, ini: m.initiales, nom: m.nom, c: m.couleur } : null);" in EQ and "email" not in JO_)
# CE QUE L'ŒIL A VU (11/11/2026, sur les photos) — figé ici, mesuré dans le navigateur par le filtre « bureau » :
#  · au téléphone, la charte (écrite pour la maquette) cachait « .list .tablecard » : un projet s'ouvrait SANS ses notes ;
#  · la page Équipe gardait deux colonnes au téléphone (413 px pour 390) ;
#  · « .chantier » de l'ancien accueil (css/site.css) décalait les nœuds « .node.chantier » de l'arbre et y posait un trait.
_tel = BU[BU.index("@media (max-width: 700px) {"):BU.index("/* ================================================================ au doigt")]
chk("css/bureau.css, au téléphone : la table des notes d'un projet RESTE affichée (la charte la cachait), la page Équipe passe sur une colonne, la barre "
    "des projets reste (la charte cache « .top ») ; css/site.css n'a plus de règle « .chantier »",
    ".bu-proj .list .tablecard { display: block; }" in _tel and ".bu-team { grid-template-columns: minmax(0, 1fr);" in _tel
    and ".top.bu-top { display: flex;" in _tel and not re.search(r"^\.chantier\b", SITE_CSS, re.M))
# l'ÉTABLI parle des mêmes projets : « Enregistrer comme note / Classer la note » (js/rangement.js) dit Projet et
# Sous-projet, propose les listes de js/projets.js (un projet créé vide y est), et écrit le nom comme il existe déjà
RG = source("rangement.js")
_etablis = [h for n, h in PAGES_R.items() if 'id="et-rangement"' in h]
chk(f"dans les {len(_etablis)} pages d'établi, « Classer la note » range dans un PROJET et un SOUS-PROJET (80 caractères, comme partout) ; ses listes "
    "viennent de js/projets.js — la même source que la page des projets et « Nouvelle note »",
    len(_etablis) >= 17 and all(re.search(r'<label class="et-rg-l">Projet<input id="et-rg-chantier" [^>]*maxlength="80" placeholder="Sans projet">', h)
                                and re.search(r'<label class="et-rg-l">Sous-projet<input id="et-rg-dossier" [^>]*maxlength="80" placeholder="Sans sous-projet">', h)
                                and ">Chantier<" not in h and ">Dossier<" not in h for h in _etablis)
    and 'import * as P from "./projets.js";' in RG and "P.modele()" in RG and "N.toutes()" not in RG
    and "n.chantier = p ? p.nom : P.nomPropre(champs.chantier.value);" in RG, len(_etablis))
chk("css/bureau.css : une pile de pastilles ne se recouvre pas (on doit y lire les initiales)",
    ".avstack .avatar, .avstack .avatar:first-child { margin-left: 0; box-shadow: none; }" in BU)

# ================================================================================================ 7 · les mutations
# Chaque mutation abîme UNE ligne d'une copie des sources, puis rejoue CE fichier entier dessus : il doit rougir.
if not REJOUE:
    print("\n=== 7 · Les mutations (chacune doit faire rougir ce fichier)")

    def rejouer(mutations=(), page=None):
        d = tempfile.mkdtemp(prefix="bureau_mut_")
        try:
            shutil.copytree(os.path.join(WEB, "js"), os.path.join(d, "js"))
            shutil.copytree(os.path.join(WEB, "css"), os.path.join(d, "css"))
            for f, avant, apres in mutations:
                ch = os.path.join(d, f)
                src = open(ch, encoding="utf-8").read()
                assert src.count(avant) == 1, (f, avant, src.count(avant))
                open(ch, "w", encoding="utf-8").write(src.replace(avant, apres))
            env = {**os.environ, "BUREAU_JS": os.path.join(d, "js"), "BUREAU_CSS": os.path.join(d, "css"), "BUREAU_REJOUE": "1"}
            if page:
                env["BUREAU_PAGE"] = "|".join(page)
            r = subprocess.run([sys.executable, os.path.abspath(__file__)], capture_output=True, text=True, timeout=300, env=env)
            return r.returncode, [l[5:90] for l in r.stdout.split("\n") if l.startswith("KO   ")]
        finally:
            shutil.rmtree(d, ignore_errors=True)

    code, rouges = rejouer()
    chk("le rejeu sur une copie INTACTE est vert (le banc des mutations ne rougit pas tout seul)", code == 0 and not rouges, rouges)
    MUTATIONS = [
        ("six initiales acceptées", [("js/equipe.js", "/^[\\p{L}\\p{N}]{1,5}$/u", "/^[\\p{L}\\p{N}]{1,6}$/u")], None),
        ("deux membres avec les mêmes initiales", [("js/equipe.js", "  const pris = autres.find((m) => m.initiales === initiales);", "  const pris = null;")], None),
        ("la pastille écrit le nom sans l'échapper", [("js/equipe.js", 'title="${esc(sig.nom)}" aria-label="${esc(sig.nom)}">${esc(sig.ini)}</span>`;',
                                                      'title="${sig.nom}" aria-label="${sig.nom}">${esc(sig.ini)}</span>`;')], None),
        ("la signature porte l'e-mail", [("js/equipe.js", "export const signatureDe = (m) => (m ? { id: m.id, ini: m.initiales, nom: m.nom, c: m.couleur } : null);",
                                         "export const signatureDe = (m) => (m ? { id: m.id, ini: m.initiales, nom: m.nom, c: m.couleur, email: m.email } : null);")], None),
        ("le journal se répète à chaque modification", [("js/journal.js", "export const FUSION_MS = 30 * 60 * 1000;", "export const FUSION_MS = 0;"),
                                                       ("js/journal.js", "const SILENCE_MS = 60 * 1000;", "const SILENCE_MS = 0;")], None),
        ("le journal sans borne", [("js/journal.js", "j.length = MAX", "j.length = j.length")], None),
        ("supprimer un projet qui a des notes", [("js/projets.js", "  if (p.nb) return { ok: false, erreur: `Ce projet a encore", "  if (false) return { ok: false, erreur: `Ce projet a encore")], None),
        ("un registre abîmé écrasé sans copie", [("js/memoire.js", '  if (m.etat === "illisible" && !garderCopie(cle, m.brut)) return false;', "")], None),
        ("une note écrite sans signature", [("js/notes.js", "    if (!avant && !note.auteur && sig) note.auteur = sig;", "    if (false) note.auteur = sig;")], None),
        ("une teinte en moins dans la feuille", [("css/bureau.css", ".avatar.c7 { background: #475569; color: #fff; }", "")], None),
        ("une teinte trop claire pour le blanc", [("css/bureau.css", ".avatar.c1 { background: #0B7A6E; color: #fff; }", ".avatar.c1 { background: #2FB5A6; color: #fff; }")], None),
        ("au téléphone, les notes d'un projet cachées (la règle de la charte)", [("css/bureau.css", "  .bu-proj .list .tablecard { display: block; }\n", "")], None),
        ("la page des projets oubliée par l'application", [("js/app.js", 'export const PAGES_BUREAU = ["projets", "equipe"];', 'export const PAGES_BUREAU = ["equipe"];')], None),
        ("une icône demandée par un écran, absente des pages", [("js/ecran_projets.js", 'V.icone("layers")', 'V.icone("rocket")')], None),
        ("le menu d'une page ne mène plus aux projets", [], ("poutre.html", 'data-tip="Projets" href="projets"', 'data-tip="Projets · bientôt" aria-disabled="true" data-bientot="1"')),
        ("l'accueil sans le journal d'activité", [], ("index.html", '<div id="activite"></div>', "")),
        ("l'équipe : trois initiales au plus dans le champ", [], ("equipe.html", '<input id="eq-ini" maxlength="5"', '<input id="eq-ini" maxlength="3"')),
    ]
    for nom, muts, page in MUTATIONS:
        try:
            code, rouges = rejouer(muts, page)
            chk(f"mutation qui doit rougir : {nom}", code != 0 and len(rouges) >= 1, (code, rouges))
        except Exception as e:  # noqa: BLE001
            chk(f"mutation qui doit rougir : {nom}", False, f"essai en erreur : {e!r}"[:300])

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

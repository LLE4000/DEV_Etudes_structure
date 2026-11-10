// Le MOTEUR de calcul vu de la page : un travailleur (js/travailleur.js, Python par Pyodide) pour TOUT le document
// (lot S4, REFONTE_ETABLI.md, S4-D5, C1, C4, C5). Les écrans qui se succèdent dans l'application (js/app.js) le
// partagent : moteurPartage() rend LE moteur du document, créé à la première demande.
//
// Les requêtes partent UNE à la fois, dans l'ordre : chaque requête est construite (fabrique()) AU MOMENT où son tour
// arrive, avec l'état laissé par la précédente — deux saisies rapides ne s'écrasent pas. Une seule exception : la
// requête en vol d'un client FERMÉ (la modification gardée d'un écran quitté, C1) ne retient pas celles des autres
// clients — elle ne s'applique plus qu'à sa note, rien ne dépend de sa réponse ; elles partent derrière elle (le
// travailleur les traite dans l'ordre, par sa chaîne) et l'écran suivant n'attend pas sa réponse.
//
// Chaque écran parle au moteur par un CLIENT (moteur.client()) : il sait dire ce qu'il a en cours (enCours), et sa
// fermeture (fermer(), synchrone) refuse toute nouvelle demande (EcranFerme), retire de la file ses LECTURES pas encore
// parties (elles rejettent EcranFerme) et GARDE ses MODIFICATIONS (« modifie ») : elles partent et se résolvent ; la
// promesse rendue par fermer() se résout quand le client n'a plus rien en file ni en vol.
//
// Le PRÉCHAUFFAGE (prechauffer(cle)) est une tâche de FOND : une file à part, d'une place (la plus récente remplace
// celle qui n'est pas encore partie), jamais postée tant qu'une requête attend ou court — une vraie requête passe
// toujours devant.
//
// Un travailleur qui TOMBE après son démarrage (erreur fatale de Pyodide, message « mort » ; ou erreur du travailleur
// lui-même) : toutes les requêtes en attente et en vol se résolvent {ok:false, erreur, mort:true}, le travailleur est
// arrêté, et la demande suivante (ou le prochain moteurPartage()) en démarre un neuf — le même objet Moteur, ses
// clients restent valables.
//
// Compatibilité (les écrans d'avant l'application) : new Moteur(surEtape, module) démarre un moteur à part et annonce
// ce module (numpy part alors en parallèle du démarrage s'il s'en sert) ; moteur.demander(fabrique, {note}) passe par
// un client interne, jamais fermé.
import { SONDE, marquer } from "./sonde.js";

export class EcranFerme extends Error {
  constructor(message = "l'écran est fermé : la requête n'est pas partie") {
    super(message);
    this.name = "EcranFerme";
  }
}

const MORT = "le moteur s'est arrêté (erreur fatale de Python) : la requête n'a pas abouti ; un moteur neuf démarre à la demande suivante";
const CLE = /^[a-z][a-z0-9_]*$/;

export class Moteur {
  constructor(surEtape, module = "") {
    this._abonnes = new Set();
    if (typeof surEtape === "function") this._abonnes.add(surEtape);
    this.n = 0;
    this.file = [];          // les requêtes pas encore parties : {client, fabrique, note, modifie, resoudre, rejeter}
    this.envols = new Map(); // les requêtes parties, sans réponse encore (id → requête) : une seule, sauf celles d'un client fermé
    this.attente = null;     // le préchauffage pas encore parti (une place)
    this._pompe = false;     // un tour de file est déjà prévu
    this._sondeModules = []; // SONDE : les lectures de sys.modules demandées, dans l'ordre
    this._libre = new Client(this);
    this._demarrer();
    if (module) this.prechauffer(module);
  }

  _demarrer() {
    const url = new URL("./travailleur.js", import.meta.url);
    // la sonde du démarrage (js/sonde.js) passe au travailleur ; sans elle, son adresse ne change pas
    if (SONDE) url.searchParams.set("sonde", "1");
    marquer("moteur_cree");
    this.etat = "demarrage";
    this.etape = "";
    this.prechauffes = new Set();   // les clés déjà annoncées à CE travailleur (un import ne se refait pas)
    this.pret = new Promise((ok, ko) => { this._ok = ok; this._ko = ko; });
    this.pret.catch(() => {});      // un démarrage raté que personne n'attend n'est pas une erreur de la page
    const w = this.w = new Worker(url, { type: "module" });
    // (un SETTER : le harnais du site enveloppe « onmessage » pour simuler ou retoucher une réponse)
    w.onmessage = ({ data }) => { if (this.w === w) this._recevoir(data); };
    w.onerror = (e) => { if (this.w === w) this._tomber(e && e.message || "le travailleur s'est arrêté"); };
  }

  _recevoir(data) {
    if (!data) return;
    if (data.type === "etape") {
      this.etape = data.texte;
      for (const f of this._abonnes) { try { f(data.texte); } catch (e) { console.error(e); } }
    } else if (data.type === "pret") {
      marquer("pret_recu");
      this.etat = "pret";
      this._ok();
    } else if (data.type === "erreur") {
      this.etat = "erreur";
      this._ko(new Error(data.texte));
    } else if (data.type === "reponse") {
      const r = this.envols.get(data.id);
      if (!r) return;
      // une réponse illisible devient un refus lisible : levée ici, elle laissait la requête sans réponse, et la
      // file — donc tout calcul suivant — bloquée pour de bon (« Calcul… » sans fin)
      let rep;
      try { rep = JSON.parse(data.json); } catch (e) { rep = { ok: false, erreur: `réponse illisible du moteur (${e.message})` }; }
      this.envols.delete(data.id);
      this._finir(r, rep);
      this._prevoir(true);
    } else if (data.type === "mort") {
      this._mourir(data.texte);
    } else if (SONDE && data.type === "sonde") {
      // les mesures du travailleur, une par clé (il ne les envoie qu'avec la sonde) : l'historique et la dernière
      (globalThis.__sondes = globalThis.__sondes || []).push(data);
      globalThis.__sonde = data;
    } else if (SONDE && data.type === "prechauffe") {
      (globalThis.__prechauffes = globalThis.__prechauffes || []).push(data);
    } else if (SONDE && data.type === "sonde_modules") {
      const f = this._sondeModules.shift();   // (dans l'ordre : la chaîne du travailleur les rend dans l'ordre)
      if (f) f(data);
    }
  }

  // le travailleur s'arrête : avant « pret », c'est un démarrage raté ; après, il est mort (un neuf à la demande)
  _tomber(texte) {
    if (this.etat === "pret") return this._mourir(texte);
    if (this.etat !== "demarrage") return;
    this.etat = "erreur";
    this._ko(new Error(texte));
    try { this.w.terminate(); } catch (e) { /* déjà arrêté */ }
    this.w = null;
    this._toutRendre({ ok: false, erreur: `le moteur n'a pas pu démarrer : ${texte}` });
  }

  _mourir(texte) {
    if (this.etat === "mort") return;
    this.etat = "mort";
    this.mort = texte;
    this._ko(new Error(texte || MORT));
    try { this.w.terminate(); } catch (e) { /* déjà arrêté */ }
    this.w = null;
    this.attente = null;
    this._toutRendre({ ok: false, erreur: MORT, mort: true });
  }

  _toutRendre(rep) {
    const tout = [...this.envols.values()].concat(this.file);
    this.envols.clear();
    this.file = [];
    for (const r of tout) this._finir(r, { ...rep });
    for (const f of this._sondeModules.splice(0)) f({ type: "sonde_modules", erreur: rep.erreur });
  }

  // un travailleur neuf, si le précédent est mort (la demande suivante, ou moteurPartage())
  _relancer() {
    if (this.etat === "mort") this._demarrer();
  }

  _finir(r, rep) {
    r.client._n--;
    r.resoudre(rep);
    r.client._verifierVide();
  }

  // le tour suivant : après une réponse, en tâche suivante (la suite de l'écran qui attendait cette réponse — appliquer
  // son état — passe AVANT la fabrique de la requête suivante, quel que soit le nombre de « await » qui l'y mènent)
  _prevoir(apresReponse = false) {
    if (this._pompe) return;
    this._pompe = true;
    const f = () => { this._pompe = false; this._pomper(); };
    if (apresReponse) setTimeout(f, 0); else queueMicrotask(f);
  }

  // une requête en vol retient celle-ci : toute requête d'un client ouvert, et celles de son propre client
  _retenue(r) {
    for (const v of this.envols.values()) if (!v.client.ferme || v.client === r.client) return true;
    return false;
  }

  // (la file garde son ordre : une requête retenue d'un client OUVERT retient tout ce qui la suit ; celle d'un client
  //  fermé — sa seconde modification gardée, qui attend la première — est passée, les suivantes peuvent partir)
  _pomper() {
    if (this.etat === "mort" || (this.etat === "erreur" && !this.w)) return;
    for (let i = 0; i < this.file.length;) {
      if (this._retenue(this.file[i])) { if (this.file[i].client.ferme) { i++; continue; } return; }
      const r = this.file.splice(i, 1)[0];
      let requete;
      try {
        const q = typeof r.fabrique === "function" ? r.fabrique() : r.fabrique;
        requete = JSON.stringify(q);
        if (typeof requete !== "string") throw new Error("requête vide");
      } catch (e) {
        r.client._n--;
        r.rejeter(e);
        r.client._verifierVide();
        continue;
      }
      r.id = ++this.n;
      this.envols.set(r.id, r);
      this.w.postMessage({ id: r.id, requete, note: !!r.note });
    }
    // rien n'attend ni ne court : le préchauffage peut partir
    if (this.attente && this.w && !this.envols.size && !this.file.length) {
      const cle = this.attente;
      this.attente = null;
      this.prechauffes.add(cle);
      this.w.postMessage({ type: "prechauffer", cle });
    }
  }

  _enfiler(client, fabrique, { note = false, modifie = false } = {}) {
    if (this.etat === "erreur" && !this.w) return Promise.resolve({ ok: false, erreur: `le moteur n'a pas pu démarrer` });
    this._relancer();
    return new Promise((resoudre, rejeter) => {
      client._n++;
      this.file.push({ client, fabrique, note, modifie, resoudre, rejeter });
      this._prevoir();
    });
  }

  // un client fermé : ses lectures pas encore parties rejettent EcranFerme ; ses modifications restent en file
  _retirer(client) {
    const gardees = [];
    for (const r of this.file) {
      if (r.client === client && !r.modifie) {
        client._n--;
        r.rejeter(new EcranFerme());
      } else gardees.push(r);
    }
    this.file = gardees;
  }

  surEtape(cb) {
    if (typeof cb !== "function") return () => {};
    this._abonnes.add(cb);
    if (this.etape) { try { cb(this.etape); } catch (e) { console.error(e); } }
    return () => this._abonnes.delete(cb);
  }

  client() { return new Client(this); }

  // une tâche de FOND : importer d'avance le module de cette clé (et numpy s'il s'en sert)
  prechauffer(cle) {
    if (typeof cle !== "string" || !CLE.test(cle)) return;
    this._relancer();
    if (this.prechauffes.has(cle) || this.etat === "erreur") return;
    this.attente = cle;
    this._prevoir();
  }

  // compatibilité : une requête hors client (les écrans d'avant l'application, le harnais, la sonde)
  demander(fabrique, options = {}) { return this._libre.demander(fabrique, options); }

  // SONDE seulement : les modules api_web chargés par le travailleur, et numpy (contrôle du préchauffage)
  sondeModules() {
    if (!SONDE) return Promise.resolve(null);
    this._relancer();
    if (!this.w) return Promise.resolve({ type: "sonde_modules", erreur: "aucun travailleur" });
    return new Promise((ok) => { this._sondeModules.push(ok); this.w.postMessage({ type: "sonde_modules" }); });
  }
}

class Client {
  constructor(moteur) {
    this.moteur = moteur;
    this.ferme = false;
    this._n = 0;
    this._vides = [];
  }

  get enCours() { return this._n; }

  demander(fabrique, { note = false, modifie = false } = {}) {
    if (this.ferme) return Promise.reject(new EcranFerme());
    return this.moteur._enfiler(this, fabrique, { note, modifie });
  }

  fermer() {
    if (!this.ferme) {
      this.ferme = true;
      this.moteur._retirer(this);
      // sa requête en vol ne retient plus celles des autres clients
      this.moteur._prevoir();
    }
    if (!this._n) return Promise.resolve();
    return new Promise((ok) => this._vides.push(ok));
  }

  _verifierVide() {
    if (this._n) return;
    const vides = this._vides;
    this._vides = [];
    for (const f of vides) f();
  }
}

let partage = null;

// LE moteur du document : créé à la première demande ; un travailleur neuf si le précédent est mort
export function moteurPartage() {
  if (!partage) partage = new Moteur();
  else partage._relancer();
  return partage;
}

// un moteur vivant existe déjà (démarré ou qui démarre)
export function moteurExiste() {
  return !!partage && partage.etat !== "mort" && partage.etat !== "erreur";
}

// Le client du travailleur : une requête à la fois, dans l'ordre (chaque requête part avec l'état laissé par la
// précédente — deux saisies rapides ne s'écrasent pas).
export class Moteur {
  constructor(surEtape) {
    this.w = new Worker(new URL("./travailleur.js", import.meta.url), { type: "module" });
    this.n = 0;
    this.attentes = new Map();
    this.file = Promise.resolve();
    this.pret = new Promise((ok, ko) => { this._ok = ok; this._ko = ko; });
    this.w.onmessage = ({ data }) => {
      if (data.type === "etape") surEtape && surEtape(data.texte);
      else if (data.type === "pret") this._ok();
      else if (data.type === "erreur") this._ko(new Error(data.texte));
      else if (data.type === "reponse") {
        const r = this.attentes.get(data.id);
        this.attentes.delete(data.id);
        if (r) r(JSON.parse(data.json));
      }
    };
    this.w.onerror = (e) => this._ko(new Error(e.message || "le travailleur s'est arrêté"));
  }

  // fabrique : une fonction qui construit la requête AU MOMENT où son tour arrive (elle lit l'état le plus récent)
  demander(fabrique) {
    const tour = this.file.then(() => new Promise((ok) => {
      const id = ++this.n;
      this.attentes.set(id, ok);
      this.w.postMessage({ id, requete: JSON.stringify(typeof fabrique === "function" ? fabrique() : fabrique) });
    }));
    this.file = tour.catch(() => {});
    return tour;
  }
}

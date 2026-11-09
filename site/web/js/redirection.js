// La page d'un module ABSORBÉ par un autre (site/web/construire.py, REDIRECTIONS : « predalle » → « dalle »,
// docs/site/DALLES_FUSION.md § 2.4) : elle mène à la page du module qui le sert, en GARDANT l'adresse — « ?note=… »
// (une ancienne note du navigateur), « ?nom=…&chantier=… » (une note neuve demandée à l'ancienne adresse). La page du
// module qui sert relit l'état tel quel (js/notes.js, moduleServi ; api_web/dalle.normaliser). Un script du site, jamais
// écrit dans la page : la politique de sécurité (_headers) refuse les scripts en ligne.
(function () {
  var vers = document.body && document.body.dataset.vers;
  if (!vers) return;
  location.replace(vers + location.search + location.hash);
})();

// Service worker: installable app (V1), and a tablet never keeps an old page. Everything comes
// from the telescope box, so nothing is cached: every request revalidates with the server.
// The server stamps BUILD (a hash of the web files) above this line: a deploy changes this file
// and the new worker takes over at once.
self.addEventListener("install", () => self.skipWaiting());
// The page polls the build and reloads itself (keeping its view), so no navigation here.
self.addEventListener("activate", e => e.waitUntil(self.clients.claim()));
self.addEventListener("fetch", e => {
  if (new URL(e.request.url).pathname.startsWith("/api/")) return;  // live streams stall through a worker
  e.respondWith(fetch(e.request, {cache: "no-cache"}));
});

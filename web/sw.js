// Service worker: installable app (V1), and a tablet never keeps an old page. Everything comes
// from the telescope box, so nothing is cached: every request revalidates with the server.
// The server stamps BUILD (a hash of the web files) above this line, so a deploy changes this
// file, the new worker takes over at once and reloads any open page.
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", e => e.waitUntil((async () => {
  await self.clients.claim();
  const pages = await self.clients.matchAll({type: "window"});
  await Promise.all(pages.map(c => c.navigate(c.url).catch(() => {})));
})()));
self.addEventListener("fetch", e => e.respondWith(fetch(e.request, {cache: "no-cache"})));

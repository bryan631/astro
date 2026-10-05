// Minimal service worker so Chrome offers "Install app" (V1). Everything comes from the
// telescope box itself, so there is nothing to cache: always go to the network.
self.addEventListener("fetch", e => e.respondWith(fetch(e.request)));

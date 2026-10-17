// Hash-роутинг: dist раздаётся с "/" одним index.html — без настройки SPA-fallback на сервере.
export type Route = { name: "upload" } | { name: "job"; id: string } | { name: "about" };

export function parseHash(hash: string): Route {
  const h = hash.replace(/^#\/?/, "");
  const [head, ...rest] = h.split("/");
  if (head === "job" && rest[0]) return { name: "job", id: decodeURIComponent(rest[0]) };
  if (head === "about") return { name: "about" };
  return { name: "upload" };
}

export function routeHash(r: Route): string {
  if (r.name === "job") return `#/job/${encodeURIComponent(r.id)}`;
  if (r.name === "about") return "#/about";
  return "#/";
}

export function navigate(r: Route) {
  const h = routeHash(r);
  if (window.location.hash !== h) window.location.hash = h;
}

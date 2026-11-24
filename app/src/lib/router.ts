// Hash routing keeps every URL working on GitHub Pages under any subpath.
import { useEffect, useState } from "react";

export interface Route { path: string[]; query: URLSearchParams }

function parse(): Route {
  const h = window.location.hash.replace(/^#\/?/, "");
  const [p, q] = h.split("?");
  return { path: p.split("/").filter(Boolean).map(decodeURIComponent), query: new URLSearchParams(q ?? "") };
}

export function useRoute(): Route {
  const [route, setRoute] = useState(parse);
  useEffect(() => {
    const on = () => setRoute(parse());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return route;
}

export function href(path: string, query?: Record<string, string | undefined>): string {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(query ?? {})) if (v) q.set(k, v);
  const s = q.toString();
  return `#/${path.replace(/^\//, "")}${s ? `?${s}` : ""}`;
}

export function go(path: string, query?: Record<string, string | undefined>) {
  window.location.hash = href(path, query);
}

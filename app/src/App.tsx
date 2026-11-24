import { useEffect } from "react";
import { useRoute, href } from "./lib/router";
import { Home } from "./pages/Home";
import { Workspace } from "./pages/Workspace";
import { Compare } from "./pages/Compare";
import { ModelPage } from "./pages/ModelPage";
import { About } from "./pages/About";
import { Brief } from "./pages/Brief";
import { JOURNEYS } from "./lib/data";

export default function App() {
  const { path, query } = useRoute();
  const section = path[0] ?? "";
  useEffect(() => { window.scrollTo(0, 0); }, [section, path[1]]);

  let page: React.ReactNode;
  if (section === "journey") page = <Workspace key={path[1]} journeyId={path[1] ?? "mine"} docParam={query.get("doc")} adding={query.get("add") === "1"} />;
  else if (section === "compare") page = <Compare key={path[1]} journeyId={path[1] ?? JOURNEYS[0]?.id} a={query.get("a")} b={query.get("b")} />;
  else if (section === "model") page = <ModelPage />;
  else if (section === "about") page = <About />;
  else if (section === "brief") page = <Brief journeyId={path[1] ?? "mine"} />;
  else page = <Home />;

  const cur = (s: string) => (section === s ? "page" : undefined);
  return (
    <>
      <header className="topbar">
        <div className="topbar-inner">
          <a className="brand" href={href("")}><span className="brand-mark" aria-hidden="true" />Report Companion</a>
          <nav className="nav" aria-label="Main">
            <a href={href(`journey/${JOURNEYS[0]?.id ?? "mine"}`)} aria-current={section === "journey" && path[1] !== "mine" ? "page" : undefined}>Examples</a>
            <a href={href("journey/mine")} aria-current={section === "journey" && path[1] === "mine" ? "page" : undefined}>My documents</a>
            <a href={href("model")} aria-current={cur("model")}>Model comparison</a>
            <a href={href("about")} aria-current={cur("about")}>How it works</a>
          </nav>
        </div>
        <div className="fiction-strip">Prototype for organising documents, not medical advice. Example patients are fictional.</div>
      </header>
      {page}
    </>
  );
}

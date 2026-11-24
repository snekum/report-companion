import { forgetEverything } from "../lib/store";
import { href } from "../lib/router";

export function About() {
  return (
    <div className="page" style={{ maxWidth: 820 }}>
      <h1>How it works</h1>

      <h2>Where your data goes</h2>
      <ul>
        <li><strong>This website is static.</strong> It has no server of its own and no analytics. Pages and model files are downloaded; nothing you type is uploaded.</li>
        <li><strong>Pasted documents, corrections and questions</strong> are saved in this browser's local storage so they are still here next visit. Clearing site data, or the button below, removes them.</li>
        <li><strong>The rule-based extraction</strong> runs instantly in the page.</li>
        <li><strong>The on-device model</strong> runs in your browser with WebGPU. The first time, the browser downloads the model files (about 1 GB) from Hugging Face and keeps them for later. The document text is processed on your computer and is never sent anywhere.</li>
        <li>There is no fallback to an online AI service. If your browser can't run the model, you can still use the prepared examples and the rules.</li>
      </ul>
      <button className="btn danger" onClick={() => { if (window.confirm("Remove every document, correction and question this site saved in this browser?")) forgetEverything(); }}>
        Forget everything saved on this device
      </button>

      <h2 style={{ marginTop: 28 }}>What the model does</h2>
      <p>
        The model reads one short document and returns a fixed set of fields: the document date, lab results (test, value, unit, the range
        printed on that report, and a date if printed), medicines (name, dose, how often, and whether the document says they're current,
        stopped, or doesn't say), and follow-up instructions. Each item comes with a quote copied from the document.
      </p>
      <p>
        Then plain code checks every item: is the quote really in the document, and do the value, unit and range appear in it? Those
        checks decide what gets highlighted and what is marked "check this". Positions in the text are always found by code, never taken
        from the model.
      </p>

      <h2>The three kinds of result you'll see</h2>
      <ul>
        <li><span className="pill prepared">Prepared, not live</span> made earlier by running the original or fine-tuned model on the example documents. Labelled with when and how.</li>
        <li><span className="pill live">Run on this device</span> made just now in your browser.</li>
        <li><span className="pill rules">Rules</span> pattern matching for lab rows. Fast and predictable, but it only knows a fixed list of test names and reads no medicines.</li>
      </ul>

      <h2>Limits</h2>
      <ul>
        <li>Short plain-text documents only. Scanned PDFs and photos are not supported.</li>
        <li>The model was trained and tested on synthetic documents made from Synthea records. That says little about real clinical documents, which are messier.</li>
        <li>The reference ranges in the examples come from invented, clearly fictional labs. They are not clinical reference intervals.</li>
        <li>Definitions are general and come from MedlinePlus. Nothing here says what a result means for a person.</li>
        <li>Facts can be wrong even when marked "found in text": a quote can exist and still be attached to the wrong field. Always check against the original.</li>
      </ul>
      <p className="small muted">Read the <a href={href("model")}>model comparison</a> for the evaluation numbers, including where the fine-tuned model did not help.</p>
    </div>
  );
}

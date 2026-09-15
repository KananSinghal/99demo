/* Samanvay console. Vanilla JS on purpose: no build step, no node_modules, nothing
   to go wrong at 3am, and it runs on an air-gapped node with a browser and nothing else. */
(function () {
  "use strict";

  // ------------------------------------------------------------------ helpers
  var $ = function (sel, root) { return (root || document).querySelector(sel); };
  var $$ = function (sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); };

  function el(tag, attrs, children) {
    var node = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) {
      if (k === "class") node.className = attrs[k];
      else if (k === "html") node.innerHTML = attrs[k];
      else if (k === "text") node.textContent = attrs[k];
      else if (k.slice(0, 2) === "on") node.addEventListener(k.slice(2), attrs[k]);
      else if (attrs[k] !== null && attrs[k] !== undefined) node.setAttribute(k, attrs[k]);
    });
    (children || []).forEach(function (c) {
      if (c === null || c === undefined || c === false) return;
      node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    });
    return node;
  }
  function esc(s) {
    return String(s === null || s === undefined ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }
  function num(v, d) {
    if (v === null || v === undefined || isNaN(v)) return "-";
    return Number(v).toLocaleString("en-IN", { maximumFractionDigits: d === undefined ? 0 : d });
  }
  function inr(v) {
    if (!v) return "-";
    var n = Number(v);
    if (n >= 1e7) return "₹" + (n / 1e7).toFixed(2) + " cr";
    if (n >= 1e5) return "₹" + (n / 1e5).toFixed(2) + " L";
    return "₹" + num(n);
  }
  function toast(title, msg, kind) {
    var t = el("div", { class: "toast " + (kind || "") }, [
      el("b", { text: title }), el("span", { text: msg || "" })
    ]);
    $("#toast").appendChild(t);
    setTimeout(function () { t.remove(); }, kind === "err" ? 9000 : 4500);
  }

  var busy = 0;
  function setBusy(on) {
    busy += on ? 1 : -1;
    $$("#btn-run, #btn-demo").forEach(function (b) { b.disabled = busy > 0; });
    $("#btn-run").innerHTML = busy > 0 ? '<span class="spin"></span> running' : "Run cascade";
  }

  function api(method, path, body) {
    setBusy(true);
    return fetch(path, {
      method: method,
      headers: body ? { "Content-Type": "application/json" } : {},
      body: body ? JSON.stringify(body) : undefined
    }).then(function (res) {
      return res.json().catch(function () { return {}; }).then(function (data) {
        if (!res.ok) {
          var msg = data.error || ("HTTP " + res.status);
          toast("Request failed", msg, "err");
          throw new Error(msg);
        }
        return data;
      });
    }).finally(function () { setBusy(false); });
  }

  var TONE = { ok: "ok", warn: "warn", stop: "stop", acc: "acc", muted: "muted" };
  var RELATION_TONE = {
    IDENTICAL: "ok", EQUIVALENT: "ok", SUBSTITUTABLE: "acc",
    VARIANT_OF: "warn", DISTINCT: "stop", UNDETERMINED: "warn"
  };

  // ------------------------------------------------------------------ state
  var state = { info: null, queue: [], selected: null, cursor: -1 };

  // ------------------------------------------------------------------ tabs
  $$("#tabs button").forEach(function (btn) {
    btn.addEventListener("click", function () {
      $$("#tabs button").forEach(function (b) { b.classList.remove("on"); });
      btn.classList.add("on");
      $$(".view").forEach(function (v) { v.classList.remove("on"); });
      $("#view-" + btn.dataset.view).classList.add("on");
      if (btn.dataset.view === "analytics") loadAnalytics();
      if (btn.dataset.view === "ledger") loadLedger();
      if (btn.dataset.view === "catalogue" && !$("#catalogue-out").children.length) searchCatalogue();
    });
  });

  $("#btn-theme").addEventListener("click", function () {
    var root = document.documentElement;
    var now = root.getAttribute("data-theme");
    var next = now === "dark" ? "light" : (now === "light" ? "" : "dark");
    if (next) root.setAttribute("data-theme", next); else root.removeAttribute("data-theme");
    try { localStorage.setItem("samanvay-theme", next); } catch (e) { /* private window */ }
  });
  try {
    var saved = localStorage.getItem("samanvay-theme");
    if (saved) document.documentElement.setAttribute("data-theme", saved);
  } catch (e) { /* ignore */ }

  // ------------------------------------------------------------------ actions
  $("#btn-demo").addEventListener("click", function () {
    api("POST", "/api/ingest/demo", {}).then(function (r) {
      toast("Corpus loaded", r.ingested + " records across " + Object.keys(r.by_cpse || {}).length + " CPSEs", "good");
      return boot();
    });
  });

  $("#btn-run").addEventListener("click", function () {
    toast("Cascade running", "nine stages over the whole corpus");
    api("POST", "/api/pipeline/run", { train: true }).then(function (r) {
      if (r.error) { toast("Cascade", r.error, "err"); return; }
      var s = r.stats || {};
      toast("Cascade complete",
        num(s.canonical) + " records → " + num((s.clusters || {}).clusters) + " codes, " +
        num(s.review_queue) + " for review", "good");
      loadQueue(); loadAnalytics();
    });
  });

  // ================================================================== REVIEW
  ["#f-tier", "#f-relation", "#f-class", "#f-cpse"].forEach(function (sel) {
    document.addEventListener("change", function (e) {
      if (e.target.matches(sel)) loadQueue();
    });
  });

  function loadQueue() {
    var q = new URLSearchParams({
      tier: $("#f-tier").value, limit: "60", order: "priority"
    });
    if ($("#f-relation").value) q.set("relation", $("#f-relation").value);
    if ($("#f-class").value) q.set("class_code", $("#f-class").value);
    return api("GET", "/api/proposals?" + q.toString()).then(function (r) {
      state.queue = r.items || [];
      renderQueue(r.total);
      if (state.queue.length) selectIndex(0);
      else $("#evidence").innerHTML = '<div class="panel"><div class="empty">Nothing in this tier.</div></div>';
    });
  }

  function renderQueue(total) {
    var box = $("#queue");
    box.innerHTML = "";
    if (!state.queue.length) {
      box.appendChild(el("div", { class: "empty", text: "Queue is empty." }));
      return;
    }
    state.queue.forEach(function (p, i) {
      var node = el("div", { class: "qitem", onclick: function () { selectIndex(i); } }, [
        el("div", { class: "t", text: (p.evidence && p.evidence.a ? p.evidence.a.description : "").slice(0, 78) }),
        el("span", { class: "pill " + (RELATION_TONE[p.relation] || "muted"), text: p.relation }),
        el("div", { class: "meta" }, [
          el("span", { text: p.a_cpse ? "" : "" }),
          el("span", { text: (p.evidence && p.evidence.a ? p.evidence.a.cpse_code : "?") + " ↔ " + (p.evidence && p.evidence.b ? p.evidence.b.cpse_code : "?") }),
          el("span", { text: "p=" + Number(p.score).toFixed(3) }),
          el("span", { text: p.annual_spend ? inr(p.annual_spend) : "" }),
          p.identity_key ? el("span", { class: "pill acc", text: "id key" }) : null
        ])
      ]);
      node.dataset.idx = i;
      box.appendChild(node);
    });
    var tiles = $("#review-tiles");
    tiles.innerHTML = "";
    var c = (state.info && state.info.counts) || {};
    [["In queue", num(total), "this filter"],
     ["Auto-accepted", num(c.proposals_auto), "within the class error bound"],
     ["National codes", num(c.nmc_active), "minted"],
     ["Ledger events", num(c.ledger_events), "append-only"],
     ["Quarantined", num(c.quarantined), "too little information to resolve"]
    ].forEach(function (t) {
      tiles.appendChild(el("div", { class: "tile" }, [
        el("div", { class: "k", text: t[0] }), el("div", { class: "v", text: t[1] }),
        el("div", { class: "n", text: t[2] })
      ]));
    });
  }

  function selectIndex(i) {
    if (i < 0 || i >= state.queue.length) return;
    state.cursor = i;
    $$("#queue .qitem").forEach(function (n) { n.classList.toggle("on", Number(n.dataset.idx) === i); });
    var node = $('#queue .qitem[data-idx="' + i + '"]');
    if (node) node.scrollIntoView({ block: "nearest" });
    renderEvidence(state.queue[i]);
  }

  function renderEvidence(p) {
    state.selected = p;
    var ev = p.evidence || {};
    var a = ev.a || {}, b = ev.b || {};
    var dec = p.decision || {};
    var box = $("#evidence");
    box.innerHTML = "";

    var attrRows = (ev.attributes || []).map(function (r) {
      return el("div", { class: "r" }, [
        el("span", { class: "k", text: r.label }),
        el("span", { class: "d" }, [
          document.createTextNode(
            (r.a === r.b && r.a) ? r.a
              : ((r.a || "not stated") + "  ∣  " + (r.b || "not stated"))
          ),
          (r.citation || r.caveat) ? el("em", { text: r.citation || r.caveat }) : null,
          (r.citation && r.caveat) ? el("em", { text: r.caveat }) : null
        ]),
        el("span", { class: "s" }, [
          el("span", { class: "pill " + (TONE[r.tone] || "muted"), text: r.status_label })
        ])
      ]);
    });

    var card = el("div", { class: "ev" }, [
      el("div", { class: "top" }, [
        el("span", { class: "t", text: "PROPOSAL #" + p.id + " · " + (a.cpse_code || "?") + " ↔ " + (b.cpse_code || "?") }),
        el("span", {}, [
          el("span", { class: "pill " + (RELATION_TONE[ev.relation] || "muted"),
                       text: ev.relation + (ev.direction ? " " + ev.direction : "") }), document.createTextNode(" "),
          el("span", { class: "pill muted", text: "p " + Number(ev.score).toFixed(3) }), document.createTextNode(" "),
          el("span", { class: "pill " + (dec.tier === "auto_accept" ? "ok" : dec.tier === "auto_reject" ? "stop" : "warn"),
                       text: (dec.tier || "").replace("_", " ") })
        ])
      ]),
      el("div", { class: "pair" }, [
        el("div", {}, [
          el("div", { class: "org", text: (a.cpse_code || "") + " · " + (a.plant || "") }),
          el("div", { class: "cd", text: "MATNR " + (a.matnr || "") + " · UoM " + (a.uom || "") +
            (a.unit_price ? " · ₹" + num(a.unit_price, 2) : "") }),
          el("div", { class: "ds", text: a.description || "" }),
          el("div", { class: "gd", text: a.golden_description || "" })
        ]),
        el("div", {}, [
          el("div", { class: "org", text: (b.cpse_code || "") + " · " + (b.plant || "") }),
          el("div", { class: "cd", text: "MATNR " + (b.matnr || "") + " · UoM " + (b.uom || "") +
            (b.unit_price ? " · ₹" + num(b.unit_price, 2) : "") }),
          el("div", { class: "ds", text: b.description || "" }),
          el("div", { class: "gd", text: b.golden_description || "" })
        ])
      ]),
      el("div", { class: "att" }, attrRows),
      el("div", { class: "foot" }, [
        el("div", {}, [el("b", { text: "Why: " }), document.createTextNode((ev.reasons || []).join("; ") || "-")]),
        ev.why_not_identical ? el("div", { style: "margin-top:4px" },
          [el("b", { text: "Why not identical: " }), document.createTextNode(ev.why_not_identical)]) : null,
        ev.counterfactual ? el("div", { style: "margin-top:4px" },
          [el("b", { text: "Counterfactual: " }), document.createTextNode(ev.counterfactual)]) : null,
        (ev.caveats || []).length ? el("div", { style: "margin-top:4px" },
          [el("b", { text: "Caveats: " }), document.createTextNode(ev.caveats.join("; "))]) : null,
        dec.guarantee ? el("div", { class: "cite", style: "margin-top:6px", text: dec.guarantee }) : null,
        (dec.reasons || []).length ? el("div", { class: "cite", text: dec.reasons.join(" · ") }) : null,
        (ev.citations || []).length ? el("div", { class: "cite", style: "margin-top:4px",
          text: "Standards: " + ev.citations.join(" | ") }) : null
      ]),
      el("div", { class: "actions" }, [
        el("button", { class: "btn ok", onclick: act("endorse"), html: "Endorse for my CPSE <kbd>A</kbd>" }),
        el("button", { class: "btn warn", onclick: act("reject"), html: "Reject <kbd>R</kbd>" }),
        el("button", { class: "btn stop", onclick: act("distinct"), html: "Mark DISTINCT <kbd>D</kbd>" }),
        el("span", { class: "cite", style: "align-self:center;margin-left:auto",
          text: "proof strength " + Number(ev.proof_strength || 0).toFixed(2) })
      ])
    ]);
    box.appendChild(card);

    if ((ev.contributions || []).length) {
      var panel = el("div", { class: "panel" }, [
        el("h3", { html: 'Score contributions <small>the reranker\'s working, not a black box</small>' }),
        el("div", { class: "scroll" }, [buildTable(
          ["Feature", "Value", "Weight", "Contribution"],
          ev.contributions.map(function (c) {
            return [c.feature, c.value.toFixed(3), c.weight.toFixed(3), c.contribution.toFixed(3)];
          }), [false, true, true, true])])
      ]);
      box.appendChild(panel);
    }
  }

  function act(kind) {
    return function () {
      var p = state.selected;
      if (!p) return;
      var actor = $("#f-actor").value || "steward";
      var cpse = $("#f-cpse").value || "";
      var reason = kind === "endorse" ? "confirmed against the evidence card"
        : (kind === "distinct" ? "hard conflict confirmed by the steward" : "not the same item");
      var url = kind === "endorse" ? "/api/proposals/" + p.id + "/endorse" : "/api/proposals/" + p.id + "/reject";
      var body = { actor: actor, actor_cpse: cpse, reason: reason };
      if (kind === "distinct") body.distinct = true;
      api("POST", url, body).then(function (r) {
        if (r.status === "awaiting_second_endorsement") {
          toast("Endorsed", "awaiting " + (r.awaiting_from || []).join(", ") +
            " — a cross-CPSE merge needs a steward in each organisation", "good");
        } else if (r.status === "merged") {
          toast("Merged", "National Material Code " + r.nmc + " · " +
            ((r.change_requests || []).length) + " MDG change requests raised", "good");
        } else {
          toast("Recorded", (r.note || r.status || "done"), "good");
        }
        var next = state.cursor;
        state.queue.splice(state.cursor, 1);
        renderQueue(state.queue.length);
        if (state.queue.length) selectIndex(Math.min(next, state.queue.length - 1));
        else $("#evidence").innerHTML = '<div class="panel"><div class="empty">Queue cleared.</div></div>';
        refreshCounts();
      });
    };
  }

  document.addEventListener("keydown", function (e) {
    if (!$("#view-review").classList.contains("on")) return;
    if (/^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName)) return;
    var k = e.key.toLowerCase();
    if (k === "j") { selectIndex(state.cursor + 1); e.preventDefault(); }
    else if (k === "k") { selectIndex(state.cursor - 1); e.preventDefault(); }
    else if (k === "a") { act("endorse")(); e.preventDefault(); }
    else if (k === "r") { act("reject")(); e.preventDefault(); }
    else if (k === "d") { act("distinct")(); e.preventDefault(); }
  });

  // =============================================================== CATALOGUE
  $("#c-go").addEventListener("click", searchCatalogue);
  $("#c-q").addEventListener("keydown", function (e) { if (e.key === "Enter") searchCatalogue(); });

  function searchCatalogue() {
    var q = new URLSearchParams({ limit: "40" });
    if ($("#c-q").value) q.set("q", $("#c-q").value);
    if ($("#c-class").value) q.set("class_code", $("#c-class").value);
    if ($("#c-cpse").value) q.set("cpse", $("#c-cpse").value);
    api("GET", "/api/catalogue?" + q.toString()).then(function (r) {
      var out = $("#catalogue-out");
      out.innerHTML = "";
      if (!(r.items || []).length) {
        out.appendChild(el("div", { class: "panel" }, [el("div", { class: "empty", text: "No codes match." })]));
        return;
      }
      r.items.forEach(function (item) {
        var members = (item.members || []).map(function (m) {
          return [m.cpse_code, m.plant || "", m.matnr, (m.description || "").slice(0, 60),
                  m.uom || "", m.unit_price ? num(m.unit_price, 2) : "-",
                  m.stock_qty ? num(m.stock_qty) : "-"];
        });
        var f = item.facets || {};
        out.appendChild(el("div", { class: "panel" }, [
          el("h3", { html: esc(item.nmc) + ' <small>' + esc(item.golden_description || "") + "</small>" }),
          el("div", { class: "body" }, [
            el("div", { style: "display:flex;flex-wrap:wrap;gap:6px;margin-bottom:9px" }, [
              el("span", { class: "pill acc", text: "NSC " + (f.nsc || "-") }),
              el("span", { class: "pill muted", text: "UNSPSC " + (f.unspsc || "-") }),
              el("span", { class: "pill muted", text: "HSN " + (f.hsn || "-") }),
              el("span", { class: "pill muted", text: "eCl@ss " + (f.eclass || "-") }),
              el("span", { class: "pill muted", text: "NSN " + (item.nsn || "-") }),
              el("span", { class: "pill " + (item.cpse_count > 1 ? "ok" : "muted"),
                           text: item.cpse_count + " CPSE" + (item.cpse_count === 1 ? "" : "s") }),
              el("button", { class: "btn", style: "margin-left:auto",
                onclick: function () { openRedeploy(item.nmc); },
                text: "Buy or Borrow" })
            ]),
            el("div", { class: "scroll" }, [buildTable(
              ["CPSE", "Plant", "MATNR", "Source description", "UoM", "Price", "Stock"],
              members, [false, false, false, false, false, true, true])])
          ])
        ]));
      });
    });
  }

  function buildTable(headers, rows, numericFlags) {
    var thead = el("thead", {}, [el("tr", {}, headers.map(function (h, i) {
      return el("th", { class: (numericFlags && numericFlags[i]) ? "num" : "", text: h });
    }))]);
    var tbody = el("tbody", {}, rows.map(function (r) {
      return el("tr", {}, r.map(function (c, i) {
        if (c && c.nodeType) return el("td", {}, [c]);
        return el("td", { class: (numericFlags && numericFlags[i]) ? "num" : "", text: String(c) });
      }));
    }));
    return el("table", {}, [thead, tbody]);
  }

  // ================================================================ REDEPLOY
  $("#r-go").addEventListener("click", function () { runRedeploy({}); });
  $("#r-q").addEventListener("keydown", function (e) { if (e.key === "Enter") runRedeploy({}); });

  function openRedeploy(nmc) {
    $$("#tabs button").forEach(function (b) { b.classList.toggle("on", b.dataset.view === "redeploy"); });
    $$(".view").forEach(function (v) { v.classList.remove("on"); });
    $("#view-redeploy").classList.add("on");
    runRedeploy({ nmc: nmc });
  }

  function runRedeploy(extra) {
    var body = {
      qty: Number($("#r-qty").value || 0),
      cpse: $("#r-cpse").value || "",
      min_idle_days: Number($("#r-idle").value || 365)
    };
    if (extra.nmc) body.nmc = extra.nmc; else body.query = $("#r-q").value;
    if (!body.nmc && !body.query) { toast("Buy or Borrow", "enter a requirement first", "err"); return; }
    api("POST", "/api/redeploy", body).then(function (r) {
      var out = $("#redeploy-out");
      out.innerHTML = "";
      if (r.error || (!r.nmc && !r.offers)) {
        out.appendChild(el("div", { class: "panel" }, [el("div", { class: "empty",
          text: r.message || r.error || "could not resolve" })]));
        return;
      }
      out.appendChild(el("div", { class: "tiles" }, [
        tile("Verdict", (r.verdict || "").toUpperCase(), r.nmc || ""),
        tile("Available", num(r.qty_available), "units at other CPSEs"),
        tile("Covers", num(r.qty_covered), "of " + num(r.qty_required) + " required"),
        tile("Procurement avoided", inr(r.procurement_avoided), "at the national mean rate")
      ]));
      out.appendChild(el("div", { class: "panel" }, [
        el("h3", { html: "Resolution <small>" + esc(r.golden_description || "") + "</small>" }),
        el("div", { class: "body" }, [
          el("p", { style: "margin:0 0 8px;font-size:15px", text: r.message }),
          el("p", { class: "cite", style: "margin:0", text: r.disclosure_note || "" })
        ])
      ]));
      if ((r.offers || []).length) {
        out.appendChild(el("div", { class: "panel" }, [
          el("h3", { html: "Offers <small>directed edges are followed, not just identity</small>" }),
          el("div", { class: "scroll" }, [buildTable(
            ["Relation", "CPSE", "Plant", "MATNR", "Description", "Stock", "Idle days", "Value", "Caveats"],
            r.offers.map(function (o) {
              return [o.relation, o.cpse_code, o.plant || "", o.matnr,
                      (o.description || "").slice(0, 52), num(o.stock_qty), num(o.idle_days),
                      inr(o.value), (o.caveats || []).join(" ").slice(0, 90) || "-"];
            }), [false, false, false, false, false, true, true, true, false])])
        ]));
      }
    });
  }

  function tile(k, v, n) {
    return el("div", { class: "tile" }, [
      el("div", { class: "k", text: k }), el("div", { class: "v", text: v }),
      el("div", { class: "n", text: n || "" })
    ]);
  }

  // ================================================================== CREATE
  $("#n-go").addEventListener("click", runCreateCheck);
  $("#n-desc").addEventListener("keydown", function (e) { if (e.key === "Enter") runCreateCheck(); });

  function runCreateCheck() {
    var desc = $("#n-desc").value;
    if (!desc) { toast("Duplicate check", "enter a description", "err"); return; }
    var t0 = performance.now();
    api("POST", "/api/duplicate-check", {
      record: { description: desc, uom: $("#n-uom").value || "NO",
                cpse_code: $("#n-cpse").value || "", matnr: "NEW" }, top_k: 5
    }).then(function (r) {
      var rtt = (performance.now() - t0).toFixed(0);
      var out = $("#create-out");
      out.innerHTML = "";
      out.appendChild(el("div", { class: "tiles" }, [
        tile("Verdict", (r.verdict || "").replace(/_/g, " "), ""),
        tile("Engine time", (r.elapsed_ms || 0) + " ms", "round trip " + rtt + " ms"),
        tile("Class", (r.canonical || {}).class_code || "-", "confidence-gated"),
        tile("Completeness", Number((r.canonical || {}).completeness || 0).toFixed(2), "0 = quarantine")
      ]));
      out.appendChild(el("div", { class: "panel" }, [
        el("h3", { text: "At creation time" }),
        el("div", { class: "body" }, [
          el("p", { style: "margin:0 0 8px;font-size:15px", text: r.message || "" }),
          el("div", { class: "m", style: "font-size:12.5px;margin-bottom:8px",
            text: (r.canonical || {}).golden_description || "" }),
          ((r.canonical || {}).missing_mandatory || []).length
            ? el("p", { class: "cite", text: "missing mandatory: " + r.canonical.missing_mandatory.join(", ") })
            : null,
          el("div", { style: "display:flex;gap:6px;flex-wrap:wrap;margin-top:8px" },
            (r.options || []).map(function (o) { return el("button", { class: "btn", text: o }); }))
        ])
      ]));
      if ((r.matches || []).length) {
        out.appendChild(el("div", { class: "panel" }, [
          el("h3", { text: "Closest existing codes" }),
          el("div", { class: "scroll" }, [buildTable(
            ["Score", "Relation", "NMC", "CPSE", "MATNR", "Existing description", "Stock"],
            r.matches.map(function (m) {
              return [Number(m.score).toFixed(3), m.relation, m.nmc || "-", m.cpse_code, m.matnr,
                      (m.description || "").slice(0, 58), num(m.stock_qty)];
            }), [true, false, false, false, false, false, true])])
        ]));
      }
    });
  }

  // =============================================================== ANALYTICS
  function loadAnalytics() {
    return api("GET", "/api/analytics").then(function (a) {
      var out = $("#analytics-out");
      out.innerHTML = "";
      var c = a.counts || {}, s = a.run_stats || {};

      out.appendChild(el("div", { class: "tiles" }, [
        tile("Source materials", num(c.source_materials), (c.cpses || 0) + " CPSEs"),
        tile("National codes", num(c.nmc_active), num(c.nmc_members) + " members mapped"),
        tile("Duplicate rate", ((s.duplicate_rate || 0) * 100).toFixed(1) + "%", num(s.records_absorbed) + " absorbed"),
        tile("Review queue", num(c.proposals_review), "ranked by value at risk"),
        tile("Directed edges", num(c.substitutions), "substitutable"),
        tile("Ledger events", num(c.ledger_events), "append-only")
      ]));

      if ((a.funnel || []).length) {
        var maxv = Math.max.apply(null, a.funnel.map(function (f) { return Math.log10((f.count || 1) + 1); }));
        out.appendChild(el("div", { class: "panel" }, [
          el("h3", { html: "The cascade <small>log-scaled; every stage reported, including the one that discards</small>" }),
          el("div", { class: "funnel" }, a.funnel.map(function (f) {
            var w = (Math.log10((f.count || 1) + 1) / maxv * 100).toFixed(1);
            return el("div", { class: "fr" }, [
              el("span", { class: "i", text: f.stage }),
              el("span", { class: "nm" }, [document.createTextNode(f.label),
                f.detail ? el("em", { text: f.detail }) : null]),
              el("span", { class: "bar" }, [el("i", { style: "width:" + w + "%" })]),
              el("span", { class: "v", text: num(f.count) })
            ]);
          }))
        ]));
      }

      if ((a.value_curve || []).length) {
        out.appendChild(el("div", { class: "panel" }, [
          el("h3", { html: 'Decisions to value <small>you do not review the whole queue - the top few cover most of the spend at risk</small>' }),
          el("div", { class: "body" }, [lineChart(a.value_curve, "decisions", "share_of_spend",
            "steward decisions", "share of spend at risk")])
        ]));
      }

      var xs = a.cross_sector || {};
      if ((xs.by_sector || []).length) {
        out.appendChild(el("div", { class: "panel" }, [
          el("h3", { html: 'Across sectors <small>' + num(xs.cross_sector_codes) + ' national codes join materials from more than one sector, covering ₹' +
            num(xs.cross_sector_spend) + ' of annual spend</small>' }),
          el("div", { class: "scroll" }, [buildTable(
            ["Sector", "CPSEs", "Materials", "Codes shared with another sector"],
            xs.by_sector.map(function (r) {
              return [r.sector, (r.cpses || []).join(", "), num(r.materials), num(r.shared_codes)];
            }), [false, false, true, true])]),
          (xs.pairs || []).length ? el("div", { class: "scroll" }, [buildTable(
            ["Sector pair", "Shared codes"],
            xs.pairs.map(function (p) { return [p.sectors.join(" ↔ "), num(p.codes)]; }),
            [false, true])]) : null
        ]));
      }

      if ((a.uom_anomalies || []).length) {
        out.appendChild(el("div", { class: "panel" }, [
          el("h3", { html: 'Unit-of-measure traps <small>apparent price gaps that vanish once the units are reconciled</small>' }),
          el("div", { class: "scroll" }, [buildTable(
            ["NMC", "Low", "High", "Raw gap", "Normalised", "Explained by UoM"],
            a.uom_anomalies.slice(0, 12).map(function (u) {
              return [u.nmc,
                      u.low.cpse + " " + u.low.uom + " ₹" + num(u.low.price, 2),
                      u.high.cpse + " " + u.high.uom + " ₹" + num(u.high.price, 2),
                      (u.raw_ratio || 0).toFixed(1) + "×",
                      u.normalised_ratio ? u.normalised_ratio.toFixed(2) + "×" : "-",
                      u.explained_by_uom ? "yes" : "no"];
            }), [false, false, false, true, true, false])])
        ]));
      }

      if ((a.by_class || []).length) {
        out.appendChild(el("div", { class: "panel" }, [
          el("h3", { html: "Extraction quality by class <small>coverage is the share of the class schema actually filled</small>" }),
          el("div", { class: "scroll" }, [buildTable(
            ["Class", "Records", "Avg coverage", "Avg completeness"],
            a.by_class.map(function (r) {
              return [r.class_code, num(r.records), Number(r.avg_coverage || 0).toFixed(2),
                      Number(r.avg_completeness || 0).toFixed(2)];
            }), [false, true, true, true])])
        ]));
      }

      var lc = a.learning_curve || {};
      if ((lc.weights || []).length) {
        out.appendChild(el("div", { class: "panel" }, [
          el("h3", { html: 'Model weights <small>trained on ' + num(lc.trained_on) +
            ' labels mined from the ERP; "moved" is how far each has shifted from its domain prior</small>' }),
          el("div", { class: "scroll" }, [buildTable(
            ["Feature", "Prior", "Weight", "Moved"],
            lc.weights.map(function (w) {
              return [w.feature, w.prior.toFixed(3), w.weight.toFixed(3),
                      (w.moved >= 0 ? "+" : "") + w.moved.toFixed(3)];
            }), [false, true, true, true])])
        ]));
      }

      api("GET", "/api/value").then(function (v) {
        out.appendChild(el("div", { class: "panel" }, [
          el("h3", { html: "The value model <small>every assumption exposed; nothing quoted that cannot be evidenced</small>" }),
          el("div", { class: "scroll" }, [buildTable(
            ["Lever", "Value", "Measured", "Basis", "Caveat"],
            (v.levers || []).map(function (l) {
              return [l.name, inr(l.value), l.measured ? "measured" : "modelled",
                      (l.basis || "").slice(0, 95), (l.caveat || "").slice(0, 110)];
            }), [false, true, false, false, false])]),
          el("div", { class: "body" }, [el("p", { class: "cite", style: "margin:0", text: v.statement || "" })])
        ]));
      });
    });
  }

  function lineChart(rows, xKey, yKey, xLabel, yLabel) {
    var W = 640, H = 200, P = 34;
    var xs = rows.map(function (r) { return r[xKey]; });
    var ys = rows.map(function (r) { return r[yKey]; });
    var xmax = Math.max.apply(null, xs) || 1, ymax = Math.max.apply(null, ys) || 1;
    var px = function (x) { return P + (x / xmax) * (W - P - 12); };
    var py = function (y) { return H - P - (y / ymax) * (H - P - 16); };
    var d = rows.map(function (r, i) { return (i ? "L" : "M") + px(r[xKey]).toFixed(1) + " " + py(r[yKey]).toFixed(1); }).join(" ");
    var area = d + " L" + px(xmax).toFixed(1) + " " + py(0).toFixed(1) + " L" + px(xs[0]).toFixed(1) + " " + py(0).toFixed(1) + " Z";
    var svg = [
      '<svg class="chart" viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="' + esc(yLabel) + ' against ' + esc(xLabel) + '">',
      '<path class="area" d="' + area + '"/>',
      '<path class="line" d="' + d + '"/>',
      '<line class="axis" x1="' + P + '" y1="' + (H - P) + '" x2="' + (W - 8) + '" y2="' + (H - P) + '"/>',
      '<line class="axis" x1="' + P + '" y1="12" x2="' + P + '" y2="' + (H - P) + '"/>',
      '<text x="' + P + '" y="' + (H - P + 14) + '">0</text>',
      '<text x="' + (W - 8) + '" y="' + (H - P + 14) + '" text-anchor="end">' + num(xmax) + "</text>",
      '<text x="' + (W / 2) + '" y="' + (H - 6) + '" text-anchor="middle">' + esc(xLabel) + "</text>",
      '<text x="' + (P - 6) + '" y="16" text-anchor="end">' + (ymax <= 1 ? (ymax * 100).toFixed(0) + "%" : num(ymax)) + "</text>",
      '<text x="6" y="' + (H / 2) + '" transform="rotate(-90 6 ' + (H / 2) + ')" text-anchor="middle">' + esc(yLabel) + "</text>"
    ];
    rows.filter(function (_r, i) { return i % Math.max(1, Math.floor(rows.length / 10)) === 0; })
      .forEach(function (r) {
        svg.push('<circle class="dot" cx="' + px(r[xKey]).toFixed(1) + '" cy="' + py(r[yKey]).toFixed(1) + '" r="2.4"/>');
      });
    svg.push("</svg>");
    var wrap = el("div", { html: svg.join("") });
    return wrap;
  }

  // ================================================================== LEDGER
  function loadLedger() {
    return api("GET", "/api/ledger?limit=60").then(function (r) {
      var out = $("#ledger-out");
      out.innerHTML = "";
      out.appendChild(el("div", { class: "tiles" }, [
        tile("Events", num(r.total), "append-only"),
        tile("Head", (r.head || "").slice(0, 14) + "…", "chain tip")
      ]));
      out.appendChild(el("div", { class: "panel" }, [
        el("h3", { html: "Event log <small>every state change names a human and carries a reason</small>" }),
        el("div", { class: "scroll" }, [buildTable(
          ["Seq", "Event", "Subject", "Actor", "CPSE", "Reason", "Hash", "When"],
          (r.events || []).map(function (e) {
            return [e.seq, e.event_type, (e.subject || "").slice(0, 26), e.actor,
                    e.actor_cpse || "-", (e.reason || "").slice(0, 60),
                    (e.hash || "").slice(0, 10), (e.created_at || "").replace("T", " ").slice(0, 19)];
          }), [true, false, false, false, false, false, false, false])])
      ]));
    });
  }

  $("#l-verify").addEventListener("click", function () {
    api("GET", "/api/ledger/verify").then(function (v) {
      toast(v.valid ? "Chain intact" : "CHAIN BROKEN", v.statement, v.valid ? "good" : "err");
      var out = $("#ledger-out");
      out.insertBefore(el("div", { class: "panel" }, [
        el("h3", { text: "Verification" }),
        el("div", { class: "body" }, [
          el("p", { style: "margin:0 0 5px" }, [
            el("span", { class: "pill " + (v.valid ? "ok" : "stop"), text: v.valid ? "VALID" : "BROKEN" }),
            document.createTextNode(" " + v.events + " events recomputed")
          ]),
          el("p", { class: "cite", style: "margin:0", text: v.statement }),
          el("p", { class: "cite", style: "margin:4px 0 0",
            text: "projection rebuilt from the log matches the materialised tables: " +
              ((v.projection_matches_tables || {}).matches ? "yes" : "NO") })
        ])
      ]), out.firstChild);
    });
  });

  $("#l-replay").addEventListener("click", function () {
    api("GET", "/api/ledger/replay").then(function (r) {
      toast("Projection replayed", r.events_applied + " events → " + r.nmc_count + " codes", "good");
      var keys = Object.keys(r.membership || {}).slice(0, 25);
      var out = $("#ledger-out");
      out.insertBefore(el("div", { class: "panel" }, [
        el("h3", { html: "Replayed projection <small>" + esc(r.statement || "") + "</small>" }),
        el("div", { class: "scroll" }, [buildTable(["NMC", "Members (canonical ids)"],
          keys.map(function (k) { return [k, (r.membership[k] || []).join(", ")]; }), [false, false])])
      ]), out.firstChild);
    });
  });

  $("#l-merkle").addEventListener("click", function () {
    api("GET", "/api/ledger/merkle?publish=1").then(function (m) {
      toast("Merkle root published", m.day + " · " + m.events + " events", "good");
      var out = $("#ledger-out");
      out.insertBefore(el("div", { class: "panel" }, [
        el("h3", { text: "Daily Merkle root " + m.day }),
        el("div", { class: "body" }, [
          el("div", { class: "m", style: "word-break:break-all", text: m.root }),
          el("p", { class: "cite", style: "margin:6px 0 0", text: m.note || "" })
        ])
      ]), out.firstChild);
    });
  });

  $("#l-crs").addEventListener("click", function () {
    api("GET", "/api/erp/change-requests?limit=40").then(function (r) {
      var out = $("#ledger-out");
      out.insertBefore(el("div", { class: "panel" }, [
        el("h3", { html: "ERP outbox <small>" + esc(r.note || "") + "</small>" }),
        el("div", { class: "scroll" }, [buildTable(
          ["Id", "NMC", "CPSE", "MATNR", "Mechanism", "Status", "Ref"],
          (r.items || []).map(function (c) {
            return [c.id, c.nmc_code, c.cpse_code, c.matnr, c.mechanism, c.status, c.external_ref || "-"];
          }), [true, false, false, false, false, false, false])]),
        el("div", { class: "body" }, [
          el("button", { class: "btn primary", text: "Dispatch queued change requests", onclick: function () {
            api("POST", "/api/erp/sync", {}).then(function (s) {
              toast("Dispatched", s.dispatched + " change requests (" + s.mode + " mode)", "good");
              $("#l-crs").click();
            });
          } })
        ])
      ]), out.firstChild);
    });
  });

  // ==================================================================== boot
  function refreshCounts() {
    return api("GET", "/api/info").then(function (info) {
      state.info = info;
      return info;
    });
  }

  function boot() {
    return refreshCounts().then(function (info) {
      var classes = (info.classes || []).map(function (c) { return c.class_code; });
      [["#f-class", true], ["#c-class", true]].forEach(function (pair) {
        var sel = $(pair[0]);
        sel.innerHTML = '<option value="">any</option>';
        classes.forEach(function (c) { sel.appendChild(el("option", { value: c, text: c })); });
      });
      var cpses = (info.cpses || []).map(function (c) { return c.cpse_code; });
      ["#f-cpse", "#c-cpse", "#r-cpse", "#n-cpse"].forEach(function (id) {
        var sel = $(id);
        sel.innerHTML = (id === "#c-cpse") ? '<option value="">any</option>' : "";
        cpses.forEach(function (c) { sel.appendChild(el("option", { value: c, text: c })); });
      });
      if ((info.counts || {}).proposals_open) loadQueue();
      else renderQueue(0);
      return info;
    });
  }

  boot().then(function (info) {
    if (!(info.counts || {}).source_materials) {
      toast("Welcome", "Press “Load corpus”, then “Run cascade”.");
    }
  });
})();

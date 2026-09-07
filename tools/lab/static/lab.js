/* Shared helpers. Deliberately tiny and dependency-free: every page is plain
 * HTML the browser opens directly, so anything here has to survive being read
 * by whoever picks this up next. */
export const api = {
  async get(path) {
    const r = await fetch(path);
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.statusText);
    return r.json();
  },
  async post(path, body) {
    const r = await fetch(path, {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.statusText);
    return r.json();
  },
};

let toastTimer = null;
export function toast(msg, ms = 1600) {
  let el = document.querySelector(".toast");
  if (!el) { el = document.createElement("div"); el.className = "toast"; document.body.append(el); }
  el.textContent = msg;
  requestAnimationFrame(() => el.classList.add("on"));
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("on"), ms);
}

export function chrome(active) {
  const tabs = [["/", "Hub"], ["/curate", "Curate"], ["/leaderboard", "Leaderboard"],
                ["/runs", "Runs"], ["/compare", "Compare"], ["/curation", "Gates"],
                ["/ontology", "Ontology"]];
  const nav = tabs.map(([href, label]) =>
    `<a href="${href}"${href === active ? ' aria-current="page"' : ""}>${label}</a>`).join("");
  document.body.insertAdjacentHTML("afterbegin",
    `<header class="top"><span class="brand">block<b>lab</b></span>
     <nav class="tabs">${nav}</nav><span class="spacer"></span>
     <button id="themeToggle" title="Toggle theme">◐</button></header>`);
  const root = document.documentElement;
  const saved = (() => { try { return localStorage.getItem("lab-theme"); } catch { return null; } })();
  if (saved) root.setAttribute("data-theme", saved);
  document.getElementById("themeToggle").addEventListener("click", () => {
    const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    try { localStorage.setItem("lab-theme", next); } catch { /* private mode */ }
  });
}

export const fmt = {
  n: (v, d = 3) => (v === null || v === undefined || Number.isNaN(v)) ? "—" : Number(v).toFixed(d),
  int: (v) => (v === null || v === undefined) ? "—" : Number(v).toLocaleString(),
  dims: (d) => Array.isArray(d) ? d.join("×") : "—",
};

export function emptyState(el, title, hint) {
  el.innerHTML = `<div class="empty"><p><strong>${title}</strong></p><p class="hint">${hint}</p></div>`;
}

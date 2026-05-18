/**
 * script.js — Frontend logic for Earthquake Prediction landing page
 */

document.addEventListener("DOMContentLoaded", async () => {
    initPredictionForm();
    initQuickLocations();
    initPlots();
    initPerformance();
    initTabs();
    initLightbox();
    initScrollAnimations();
    await initHeroStats();
    initCounters();
});

/* ═══════════ HERO STATS ═══════════ */
async function initHeroStats() {
    try {
        const resp = await fetch("/api/hero_stats");
        const stats = await resp.json();
        
        document.getElementById("hero-events").dataset.count = stats.total_events;
        document.getElementById("hero-r2").textContent = (stats.best_r2 * 100).toFixed(1) + "%";
        document.getElementById("hero-models").textContent = stats.ml_models;
        document.getElementById("hero-years").textContent = stats.years_of_data;
    } catch (e) {
        console.warn("Could not load hero stats:", e);
    }
}

/* ═══════════ COUNTER ANIMATION ═══════════ */
function initCounters() {
    const counters = document.querySelectorAll("[data-count]");
    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
            if (!entry.isIntersecting) return;
            const el = entry.target;
            const target = parseInt(el.dataset.count);
            const duration = 2000;
            const start = performance.now();
            function update(now) {
                const elapsed = now - start;
                const progress = Math.min(elapsed / duration, 1);
                const eased = 1 - Math.pow(1 - progress, 3);
                el.textContent = Math.floor(eased * target).toLocaleString() + "+";
                if (progress < 1) requestAnimationFrame(update);
            }
            requestAnimationFrame(update);
            observer.unobserve(el);
        });
    }, { threshold: 0.5 });
    counters.forEach((el) => observer.observe(el));
}

/* ═══════════ PREDICTION FORM ═══════════ */
function initPredictionForm() {
    const form = document.getElementById("predict-form");
    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        const btn = document.getElementById("predict-btn");
        const btnText = btn.querySelector(".btn-text");
        const btnLoad = btn.querySelector(".btn-loading");

        btnText.style.display = "none";
        btnLoad.style.display = "inline-flex";
        btn.disabled = true;

        const payload = {
            latitude: parseFloat(document.getElementById("latitude").value),
            longitude: parseFloat(document.getElementById("longitude").value),
            depth: parseFloat(document.getElementById("depth").value),
        };

        const year = document.getElementById("year").value;
        const month = document.getElementById("month").value;
        const hour = document.getElementById("hour").value;
        const dow = document.getElementById("day_of_week").value;
        if (year) payload.year = parseInt(year);
        if (month) payload.month = parseInt(month);
        if (hour) payload.hour = parseInt(hour);
        if (dow) payload.day_of_week = parseInt(dow);

        try {
            const resp = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            const data = await resp.json();
            if (data.error) throw new Error(data.error);
            displayResults(data);
        } catch (err) {
            alert("Prediction failed: " + err.message);
        } finally {
            btnText.style.display = "inline";
            btnLoad.style.display = "none";
            btn.disabled = false;
        }
    });
}

function displayResults(data) {
    const card = document.getElementById("results-card");
    card.style.display = "block";
    card.className = "glass-card results-card animate-in";

    // Magnitude
    const magEl = document.getElementById("result-magnitude");
    magEl.textContent = data.magnitude;
    magEl.style.color = data.magnitude >= 6 ? "#ef4444" : data.magnitude >= 4 ? "#f59e0b" : "#10b981";
    const magPct = Math.min((data.magnitude / 10) * 100, 100);
    document.getElementById("mag-bar").style.width = magPct + "%";

    // Risk level
    document.getElementById("result-risk").textContent = data.risk_level;
    document.getElementById("result-risk").style.color = data.risk_color;
    ["low", "medium", "high"].forEach((level) => {
        const badge = document.getElementById("badge-" + level);
        if(badge) badge.classList.toggle("active", level === data.risk_level.toLowerCase());
    });

    // Geological Zone
    if (data.geological_zone) {
        const zoneNameEl = document.getElementById("result-zone-name");
        const zoneDescEl = document.getElementById("result-zone-desc");
        if (zoneNameEl) {
            zoneNameEl.textContent = data.geological_zone.name;
            zoneNameEl.style.color = data.geological_zone.color;
        }
        if (zoneDescEl) zoneDescEl.textContent = data.geological_zone.desc;
    }

    // Probability
    const prob = data.high_mag_probability;
    document.getElementById("result-probability").textContent = prob.toFixed(1) + "%";
    document.getElementById("prob-ring-text").textContent = prob.toFixed(1) + "%";
    const circumference = 2 * Math.PI * 52; // r=52
    const offset = circumference - (prob / 100) * circumference;
    const ring = document.getElementById("prob-ring-fill");
    ring.style.stroke = prob > 50 ? "#ef4444" : prob > 20 ? "#f59e0b" : "#10b981";
    setTimeout(() => { ring.style.strokeDashoffset = offset; }, 100);

    // Summary
    const summary = document.getElementById("result-summary");
    let msg = `For coordinates (${data.input_summary.latitude}, ${data.input_summary.longitude}) at ${data.input_summary.depth_km} km depth — `;
    
    if (data.geological_zone) {
        msg += `📍 Located in <strong>${data.geological_zone.name}</strong>. `;
    }

    if (data.risk_level === "High") {
        msg += `⚠️ <strong>High seismic risk</strong> detected. Predicted magnitude ${data.magnitude} with ${prob.toFixed(1)}% probability of being ≥6.0.`;
    } else if (data.risk_level === "Medium") {
        msg += `⚡ <strong>Moderate seismic risk</strong>. Predicted magnitude ${data.magnitude} with ${prob.toFixed(1)}% high-mag probability.`;
    } else {
        msg += `✅ <strong>Low seismic risk</strong>. Predicted magnitude ${data.magnitude}. Region shows typical background seismicity.`;
    }
    if(summary) summary.innerHTML = msg;

    card.scrollIntoView({ behavior: "smooth", block: "center" });
}

/* ═══════════ QUICK LOCATIONS ═══════════ */
function initQuickLocations() {
    document.querySelectorAll(".quick-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            document.getElementById("latitude").value = btn.dataset.lat;
            document.getElementById("longitude").value = btn.dataset.lon;
            document.getElementById("depth").value = btn.dataset.depth;
            document.getElementById("latitude").focus();
        });
    });
}

/* ═══════════ PLOTS GALLERY ═══════════ */
const PLOT_META = {
    "eda_magnitude_distribution.png": { title: "Magnitude Distribution", tab: "eda" },
    "eda_depth_vs_magnitude.png": { title: "Depth vs Magnitude", tab: "eda" },
    "eda_events_per_year.png": { title: "Events per Year", tab: "eda" },
    "eda_geographic_distribution.png": { title: "Geographic Distribution", tab: "eda" },
    "eda_correlation_heatmap.png": { title: "Correlation Heatmap", tab: "eda" },
    "eda_risk_distribution.png": { title: "Risk Level Distribution", tab: "eda" },
    "eval_model1_predictions.png": { title: "Predicted vs Actual Magnitude", tab: "eval" },
    "eval_model2_confusion.png": { title: "Confusion Matrices", tab: "eval" },
    "eval_model3_roc_curve.png": { title: "ROC Curve", tab: "eval" },
    "eval_model_comparison.png": { title: "Model Comparison", tab: "eval" },
};

let allPlots = [];
let activeTab = "eda";

async function initPlots() {
    try {
        const resp = await fetch("/api/plots");
        allPlots = await resp.json();
        renderPlots();
    } catch (e) {
        console.warn("Could not load plots:", e);
    }
}

function renderPlots() {
    const grid = document.getElementById("plots-grid");
    grid.innerHTML = "";
    const filtered = allPlots.filter((f) => {
        const meta = PLOT_META[f];
        return meta && meta.tab === activeTab;
    });
    filtered.forEach((f) => {
        const meta = PLOT_META[f] || { title: f };
        const card = document.createElement("div");
        card.className = "plot-card";
        card.innerHTML = `<img src="/plots/${f}" alt="${meta.title}" loading="lazy"><div class="plot-title">${meta.title}</div>`;
        card.addEventListener("click", () => openLightbox(`/plots/${f}`, meta.title));
        grid.appendChild(card);
    });
}

/* ═══════════ TABS ═══════════ */
function initTabs() {
    document.querySelectorAll(".tab-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");
            activeTab = btn.dataset.tab;
            renderPlots();
        });
    });
}

/* ═══════════ LIGHTBOX ═══════════ */
function initLightbox() {
    const lb = document.getElementById("lightbox");
    document.getElementById("lightbox-close").addEventListener("click", () => lb.classList.remove("active"));
    lb.addEventListener("click", (e) => { if (e.target === lb) lb.classList.remove("active"); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") lb.classList.remove("active"); });
}

function openLightbox(src, caption) {
    const lb = document.getElementById("lightbox");
    document.getElementById("lightbox-img").src = src;
    document.getElementById("lightbox-caption").textContent = caption;
    lb.classList.add("active");
}

/* ═══════════ PERFORMANCE METRICS ═══════════ */
async function initPerformance() {
    try {
        const resp = await fetch("/api/results");
        const results = await resp.json();
        const container = document.getElementById("perf-cards");
        container.innerHTML = "";
        results.forEach((r) => {
            const card = document.createElement("div");
            card.className = "perf-card";
            let metrics = "";
            if (r.R2_Score !== "") metrics += metricRow("R² Score", r.R2_Score, true);
            if (r.MAE !== "") metrics += metricRow("MAE", r.MAE);
            if (r.RMSE !== "") metrics += metricRow("RMSE", r.RMSE);
            if (r.Accuracy !== "") metrics += metricRow("Accuracy", (parseFloat(r.Accuracy) * 100).toFixed(1) + "%", true);
            if (r.ROC_AUC !== "") metrics += metricRow("ROC AUC", r.ROC_AUC);
            card.innerHTML = `
                <div class="perf-algo">${r.Algorithm}</div>
                <div class="perf-group">${r["Model Group"]}</div>
                ${metrics}
            `;
            container.appendChild(card);
        });
    } catch (e) {
        console.warn("Could not load results:", e);
    }
}

function metricRow(label, value, highlight = false) {
    return `<div class="perf-metric"><span class="perf-metric-label">${label}</span><span class="perf-metric-value ${highlight ? "highlight" : ""}">${value}</span></div>`;
}

/* ═══════════ SCROLL ANIMATIONS ═══════════ */
function initScrollAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = "1";
                entry.target.style.transform = "translateY(0)";
            }
        });
    }, { threshold: 0.1 });

    document.querySelectorAll("section:not(#hero)").forEach((el) => {
        el.style.opacity = "0";
        el.style.transform = "translateY(30px)";
        el.style.transition = "opacity 0.8s ease, transform 0.8s ease";
        observer.observe(el);
    });
}

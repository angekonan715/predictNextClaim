"""Streamlit app: a sideways dashboard of claim risk.

Run from the project root:
    streamlit run app/app.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.evaluate import assign_risk_tier  # noqa: E402
from src.predict import load_model, predict_claim_probability  # noqa: E402

st.set_page_config(page_title="Claim Risk Estimator", page_icon="🚗", layout="wide")

NAVY = "#1B3A4B"
SLATE = "#7C909C"
INK = "#1C2834"
MUTED = "#5C6B7A"
LINE = "#D5DDE6"
PAPER = "#F4F7FA"
CARD = "#FFFFFF"


@st.cache_resource
def get_model(model_mtime: float):
    # model_mtime is part of the cache key, so a replaced model file is loaded
    # on the next visit instead of the copy kept in memory.
    return load_model()


def style_axes(fig, ax) -> None:
    """One chart style for every slide: white card, navy ink, light rules."""
    fig.patch.set_facecolor(CARD)
    ax.set_facecolor(CARD)
    ax.tick_params(colors=INK, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(LINE)
    ax.spines["bottom"].set_color(LINE)


def kpi_stack(items: list[tuple[str, str]]) -> None:
    cards = "".join(
        f'<div class="kpi"><span>{label}</span><strong>{value}</strong></div>'
        for label, value in items
    )
    st.markdown(f'<div class="kpi-stack">{cards}</div>', unsafe_allow_html=True)


def slide_heading(number: str, kicker: str, title: str, body: str) -> None:
    st.markdown(
        f"""
        <p class="kicker"><span>{number}</span> {kicker}</p>
        <h2 class="slide-title">{title}</h2>
        <p class="slide-lede">{body}</p>
        """,
        unsafe_allow_html=True,
    )


def deck_controls(jump_to: int | None) -> None:
    """Arrow keys and the mouse wheel move between slides."""
    start = "null" if jump_to is None else str(jump_to)
    components.html(
        f"""
        <script>
        const parent = window.parent;
        const doc = parent.document;
        function row() {{
          const deck = doc.querySelector('.st-key-deck');
          if (!deck) return null;
          return deck.querySelector('[data-testid="stHorizontalBlock"]');
        }}
        function slides() {{
          const scroller = row();
          if (!scroller) return [];
          return [...scroller.querySelectorAll(':scope > [data-testid="stColumn"]')];
        }}
        function indexNow() {{
          const scroller = row();
          const all = slides();
          if (!scroller || !all.length) return 0;
          let best = 0;
          let bestDist = Infinity;
          all.forEach((slide, i) => {{
            const dist = Math.abs(slide.offsetLeft - scroller.scrollLeft);
            if (dist < bestDist) {{
              bestDist = dist;
              best = i;
            }}
          }});
          return best;
        }}
        function go(index) {{
          const scroller = row();
          const all = slides();
          if (!scroller || !all.length) return;
          const slide = all[Math.max(0, Math.min(all.length - 1, index))];
          scroller.scrollTo({{left: slide.offsetLeft, behavior: 'smooth'}});
        }}
        if (!doc.getElementById('deck-nav')) {{
          const nav = doc.createElement('div');
          nav.id = 'deck-nav';
          nav.innerHTML = `
            <button type="button" class="deck-arrow" data-dir="-1" aria-label="Previous slide">&#8592;</button>
            <span class="deck-count">1 / 4</span>
            <button type="button" class="deck-arrow" data-dir="1" aria-label="Next slide">&#8594;</button>
          `;
          doc.body.appendChild(nav);
        }}
        const nav = doc.getElementById('deck-nav');
        nav.querySelectorAll('.deck-arrow').forEach((button) => {{
          button.onclick = () => {{
            const count = slides().length || 4;
            go(Math.max(0, Math.min(count - 1, indexNow() + Number(button.dataset.dir))));
          }};
        }});
        function paintCount() {{
          const label = nav.querySelector('.deck-count');
          const count = slides().length || 4;
          if (!label) return;
          label.textContent = (indexNow() + 1) + ' / ' + count;
        }}
        const scroller = row();
        if (scroller && !scroller.dataset.deckListening) {{
          scroller.dataset.deckListening = '1';
          scroller.addEventListener('scroll', paintCount, {{passive: true}});
        }}
        if (!parent.__deckBound) {{
          parent.__deckBound = true;
          parent.addEventListener('keydown', (event) => {{
            if (event.key !== 'ArrowRight' && event.key !== 'ArrowLeft') return;
            const target = event.target;
            if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA')) return;
            event.preventDefault();
            const count = slides().length || 4;
            const step = event.key === 'ArrowRight' ? 1 : -1;
            go(Math.max(0, Math.min(count - 1, indexNow() + step)));
          }});
          parent.addEventListener('wheel', (event) => {{
            const current = row();
            if (!current || !current.contains(event.target)) return;
            if (Math.abs(event.deltaX) > Math.abs(event.deltaY)) return;
            const slide = event.target.closest('[data-testid="stColumn"]');
            if (slide && slide.scrollHeight > slide.clientHeight + 8) {{
              const atTop = slide.scrollTop <= 0;
              const atBottom = slide.scrollTop + slide.clientHeight >= slide.scrollHeight - 2;
              if ((event.deltaY < 0 && !atTop) || (event.deltaY > 0 && !atBottom)) return;
            }}
            event.preventDefault();
            current.scrollBy({{left: event.deltaY, behavior: 'auto'}});
          }}, {{passive: false, capture: true}});
        }}
        const start = {start};
        if (start !== null) go(start);
        paintCount();
        </script>
        """,
        height=0,
    )


st.markdown(
    f"""
    <style>
    .stApp {{
      background: {PAPER};
      color: {INK};
    }}
    header[data-testid="stHeader"] {{
      background: transparent;
    }}
    .block-container {{
      max-width: 100%;
      padding: 0.6rem 0.8rem 0.4rem 0.8rem;
    }}
    .st-key-deck > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] {{
      display: flex;
      flex-wrap: nowrap !important;
      gap: 0.8rem;
      overflow-x: auto;
      overflow-y: hidden;
      scroll-snap-type: x mandatory;
      scroll-behavior: smooth;
      width: 100%;
      height: calc(100vh - 1.3rem);
      padding-bottom: 0.2rem;
    }}
    .st-key-deck > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
      min-width: calc(100vw - 2.4rem) !important;
      width: calc(100vw - 2.4rem) !important;
      flex: 0 0 calc(100vw - 2.4rem) !important;
      scroll-snap-align: start;
      height: calc(100vh - 8.2rem);
      overflow-x: hidden;
      overflow-y: auto;
      background: {PAPER};
      border: 1px solid {LINE};
      border-radius: 18px;
      box-shadow: 0 10px 30px rgba(27, 58, 75, 0.06);
      padding: 1.15rem 1.25rem 1rem 1.25rem;
    }}
    .st-key-deck [data-testid="stColumn"] [data-testid="stHorizontalBlock"] {{
      overflow: visible;
      scroll-snap-type: none;
      height: auto;
      width: auto;
      padding: 0;
      gap: 0.8rem;
    }}
    .st-key-deck [data-testid="stColumn"] [data-testid="stColumn"] {{
      min-width: 0 !important;
      width: auto !important;
      flex: 1 1 0 !important;
      height: auto;
      overflow: visible;
      background: transparent;
      border: 0;
      box-shadow: none;
      border-radius: 0;
      padding: 0;
    }}
    .st-key-deck [data-testid="stColumn"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:has(.st-key-result_card) {{
      position: sticky;
      top: 0;
      align-self: flex-start;
    }}
    .kicker {{
      color: {MUTED};
      letter-spacing: 0.14em;
      text-transform: uppercase;
      font-size: 0.78rem;
      margin: 0 0 0.35rem 0;
    }}
    .kicker span {{
      color: {NAVY};
      font-weight: 700;
    }}
    .slide-title {{
      color: {NAVY};
      font-size: 2rem;
      line-height: 1.15;
      margin: 0 0 0.45rem 0;
      font-weight: 650;
    }}
    .slide-lede {{
      color: {MUTED};
      font-size: 1.02rem;
      max-width: 38rem;
      margin: 0 0 0.8rem 0;
      line-height: 1.5;
    }}
    .board,
    .st-key-models_page,
    .st-key-drivers_page {{
      max-width: 980px;
      margin: 0 auto;
    }}
    .st-key-deck > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:has(.st-key-estimate_page) > div {{
      max-width: 980px;
      margin-left: auto;
      margin-right: auto;
    }}
    .board {{
      display: flex;
      flex-direction: column;
      gap: 0.85rem;
      height: auto;
    }}
    .board-head h2 {{
      color: {NAVY};
      font-size: 1.85rem;
      line-height: 1.15;
      margin: 0 0 0.3rem 0;
      font-weight: 650;
    }}
    .board-head p {{
      color: {MUTED};
      max-width: 44rem;
      margin: 0;
      line-height: 1.45;
    }}
    .board-grid {{
      display: grid;
      gap: 0.85rem;
    }}
    .cols-3 {{ grid-template-columns: 1.15fr 0.85fr 1fr; }}
    .cols-2 {{ grid-template-columns: 1.35fr 0.8fr; }}
    .story {{ grid-template-columns: 1.15fr 0.85fr; }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 0.75rem;
    }}
    .metric {{
      background: {CARD};
      border: 1px solid {LINE};
      border-radius: 16px;
      padding: 0.85rem 1rem 0.95rem 1rem;
      box-shadow: 0 8px 22px rgba(27, 58, 75, 0.04);
    }}
    .metric span {{
      color: {MUTED};
      font-size: 0.84rem;
    }}
    .metric b {{
      display: block;
      color: {NAVY};
      font-size: 1.9rem;
      line-height: 1.15;
      margin: 0.15rem 0 0.3rem 0;
      font-weight: 650;
    }}
    .metric p {{
      margin: 0;
      color: {INK};
      font-size: 0.9rem;
      line-height: 1.4;
    }}
    .steps {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 0.75rem;
    }}
    .steps div {{
      background: {CARD};
      border: 1px solid {LINE};
      border-radius: 14px;
      padding: 0.7rem 0.85rem 0.8rem 0.85rem;
    }}
    .steps b {{
      display: block;
      color: {NAVY};
      font-size: 0.78rem;
      letter-spacing: 0.08em;
      margin-bottom: 0.2rem;
    }}
    .steps span {{ color: {INK}; font-size: 0.92rem; line-height: 1.35; }}
    .score-list small {{
      display: block;
      color: {MUTED};
      font-size: 0.82rem;
      font-weight: 400;
      line-height: 1.35;
      margin-top: 0.15rem;
    }}
    .panel {{
      background: {CARD};
      border: 1px solid {LINE};
      border-radius: 16px;
      padding: 1.05rem 1.15rem 0.95rem 1.15rem;
      display: flex;
      flex-direction: column;
      box-shadow: 0 8px 22px rgba(27, 58, 75, 0.04);
    }}
    .panel h3 {{
      margin: 0 0 0.55rem 0;
      color: {NAVY};
      font-size: 1rem;
      font-weight: 650;
    }}
    .panel p, .panel li {{
      color: {INK};
      line-height: 1.5;
      margin: 0;
    }}
    .panel ol, .panel ul {{
      margin: 0;
      padding-left: 1.1rem;
    }}
    .panel li + li {{ margin-top: 0.45rem; }}
    .panel .foot {{
      margin-top: 0.85rem;
      padding-top: 0.7rem;
      color: {MUTED};
      font-size: 0.88rem;
      border-top: 1px solid {LINE};
    }}
    .score-list {{
      list-style: none;
      padding: 0;
      margin: 0;
    }}
    .score-list li {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 1rem;
      padding: 0.72rem 0;
      border-bottom: 1px solid {LINE};
    }}
    .score-list li:last-child {{ border-bottom: 0; }}
    .score-list span {{ color: {MUTED}; }}
    .score-list b {{
      color: {NAVY};
      font-size: 1.55rem;
      font-weight: 650;
    }}
    .rank-row {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 0.15rem 0.6rem;
      margin-bottom: 0.72rem;
    }}
    .rank-row span {{ color: {INK}; font-size: 0.92rem; }}
    .rank-row b {{ color: {NAVY}; }}
    .rank-row i {{
      grid-column: 1 / -1;
      display: block;
      height: 6px;
      border-radius: 99px;
      background: {NAVY};
    }}
    [data-testid="stImage"] img,
    [data-testid="stPyplot"] img {{
      max-width: 100%;
      height: auto;
    }}
    .st-key-models_page [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
    .st-key-drivers_page [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
      background: {CARD} !important;
      border: 1px solid {LINE} !important;
      border-radius: 16px !important;
      padding: 0.75rem 0.85rem !important;
      box-shadow: 0 8px 22px rgba(27, 58, 75, 0.04);
    }}
    .st-key-deck > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:has(.st-key-estimate_page) {{
      background: linear-gradient(165deg, #D5E4EC 0%, #F4F8FA 46%, #E4EEF3 100%);
      border-color: #C5D6E0;
    }}
    .st-key-deck > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:has(.st-key-estimate_page) .stForm {{
      background: {CARD};
      box-shadow: 0 10px 28px rgba(27, 58, 75, 0.07);
    }}
    .st-key-result_card {{
      background: {NAVY};
      border-radius: 18px;
      padding: 1.15rem 1.15rem 0.9rem 1.15rem;
      box-shadow: 0 16px 36px rgba(27, 58, 75, 0.18);
    }}
    .st-key-result_card .kicker,
    .st-key-result_card .kicker span {{
      color: rgba(255, 255, 255, 0.72);
    }}
    .st-key-result_card div[data-testid="stMetric"] {{
      background: rgba(255, 255, 255, 0.08);
      border-color: rgba(255, 255, 255, 0.16);
    }}
    .st-key-result_card div[data-testid="stMetricLabel"],
    .st-key-result_card [data-testid="stCaptionContainer"] {{
      color: rgba(255, 255, 255, 0.76);
    }}
    .st-key-result_card div[data-testid="stMetricValue"] {{
      color: #FFFFFF;
      font-size: 2.4rem;
    }}
    div[data-testid="stMetric"] {{
      background: {PAPER};
      border: 1px solid {LINE};
      border-radius: 12px;
      padding: 0.7rem 0.9rem;
    }}
    div[data-testid="stMetricLabel"] {{
      color: {MUTED};
    }}
    div[data-testid="stMetricValue"] {{
      color: {NAVY};
    }}
    [data-testid="stCaptionContainer"] {{
      color: {MUTED};
    }}
    div[data-testid="stDataFrame"] {{
      border: 1px solid {LINE};
      border-radius: 12px;
    }}
    .stForm {{
      border: 1px solid {LINE};
      border-radius: 14px;
      padding: 0.8rem 1rem 0.4rem 1rem;
      background: {PAPER};
    }}
    .stFormSubmitButton button {{
      background: {NAVY};
      color: white;
      border: 0;
      border-radius: 8px;
    }}
    #deck-nav {{
      position: fixed;
      left: 50%;
      bottom: 0.85rem;
      transform: translateX(-50%);
      z-index: 1000;
      display: flex;
      align-items: center;
      gap: 0.7rem;
      background: {NAVY};
      color: white;
      border-radius: 999px;
      padding: 0.35rem 0.45rem;
      box-shadow: 0 8px 24px rgba(27, 58, 75, 0.25);
    }}
    #deck-nav .deck-arrow {{
      background: transparent;
      color: white;
      border: 1px solid rgba(255,255,255,0.35);
      border-radius: 999px;
      width: 2rem;
      height: 2rem;
      cursor: pointer;
    }}
    #deck-nav .deck-count {{
      font-size: 0.85rem;
      letter-spacing: 0.08em;
      min-width: 3.2rem;
      text-align: center;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

levels_file = config.PROCESSED_DIR / "feature_levels.json"
if not config.MODEL_FILE.exists() or not levels_file.exists():
    st.warning("Train the model first: `python -m src.data_prep` then `python -m src.train`.")
    st.stop()

bundle = get_model(config.MODEL_FILE.stat().st_mtime)
cutoffs = bundle.get("tier_cutoffs", config.RISK_TIER_CUTOFFS)

FRIENDLY = {
    "bonus_score": {"G": "Good record", "N": "Neutral record", "B": "Poor record"},
    "policy_type": {
        "TP": "Third party",
        "TPG": "Third party plus glass",
        "CC": "Combined cover",
        "COMP_E": "Comprehensive with excess",
        "COMP_N": "Comprehensive, no excess",
    },
    "business_type": {"NB": "New policy", "P": "Existing policy"},
    "payment_frequency": {"A": "Once a year", "S": "Twice a year", "Q": "Every three months"},
    "fuel_type": {"D": "Diesel", "G": "Gasoline", "Unknown": "Unknown"},
    "municipality_type": {"I": "Inland", "C": "Coast", "IS": "Island"},
    "circulation_area": {"U": "Town", "R": "Country"},
}

levels = json.loads(levels_file.read_text(encoding="utf-8"))


def pick(column: str, title: str, help_text: str | None = None) -> str:
    """Show a plain-language choice and return the code stored in the data."""
    options = levels.get(column, [])
    labels = FRIENDLY.get(column, {})
    shown = [labels.get(code, code) for code in options]
    choice = st.selectbox(title, shown, help=help_text)
    return options[shown.index(choice)]


FACT_NAMES = {
    "total_exposure": "Time the policy was in force",
    "policy_type": "Coverage",
    "vehicle_value": "Vehicle value",
    "vehicle_age": "Vehicle age",
    "power_to_weight_ratio": "Weight per horsepower",
    "driver_age": "Driver age",
    "payment_frequency": "Payment schedule",
    "age_driving_licence": "Years with a licence",
    "vehicle_brand": "Brand",
    "fuel_type": "Fuel",
    "bonus_score": "Past claims",
    "circulation_area": "Town or country",
    "business_type": "New or existing policy",
    "municipality_type": "Home area",
    "seats": "Seats",
}


def feature_shares(model_bundle: dict) -> pd.DataFrame:
    """Share of LightGBM's splits that use each original fact."""
    fitted = model_bundle["model"].named_steps["model"]
    names = model_bundle["model"].named_steps["preprocess"].get_feature_names_out()
    gain = pd.Series(fitted.booster_.feature_importance(importance_type="gain"), index=names)
    columns = sorted(config.FEATURES + [config.EXPOSURE_COL], key=len, reverse=True)

    def original(name: str) -> str:
        rest = name.split("__", 1)[1]
        if name.startswith("num__"):
            return rest
        for column in columns:
            if rest == column or rest.startswith(column + "_"):
                return column
        return rest

    share = gain.groupby(gain.index.map(original)).sum()
    share = (share / share.sum()).sort_values(ascending=False)
    return pd.DataFrame({
        "Fact": [FACT_NAMES.get(name, name) for name in share.index],
        "Share": share.to_numpy(),
    })


comparison_file = config.FIGURES_DIR / "model_comparison.csv"
scores = pd.read_csv(comparison_file, index_col=0) if comparison_file.exists() else None

with st.container(key="deck"):
    overview, compare, drivers, estimate = st.columns(4, gap="small")

    with overview:
        auc = gap = chance = "—"
        if scores is not None and "LightGBM" in scores.index:
            chosen = scores.loc["LightGBM"]
            auc = f"{chosen['AUC']:.3f}"
            gap = f"{chosen['calibration_gap']:.3f}"
            chance = f"{chosen['mean_predicted']:.0%}"
        note = bundle.get("note", "The saved model estimates the probability of at least one claim.")
        st.markdown(
            f"""
            <div class="board">
              <header class="board-head">
                <p class="kicker"><span>01</span> Overview</p>
                <h2>Car insurance claim risk</h2>
                <p>The chance a driver files at least one claim in the next year.
                Trained on real policies from a Spanish insurer, 2022–2023, and checked on 2024.</p>
              </header>
              <div class="metrics">
                <div class="metric"><span>Sorting score on 2024</span><b>{auc}</b><p>0.50 is a coin flip. Closer to 1 is better.</p></div>
                <div class="metric"><span>Calibration gap</span><b>{gap}</b><p>A predicted 20% really happens about 20% of the time.</p></div>
                <div class="metric"><span>Average predicted chance</span><b>{chance}</b><p>Near the share of 2024 policies that claimed.</p></div>
              </div>
              <div class="board-grid story">
                <article class="panel">
                  <h3>What the model does</h3>
                  <p>{note}</p>
                  <p class="foot">It uses facts known on day one. This is not an insurance quote.</p>
                </article>
                <article class="panel">
                  <h3>What a full year means</h3>
                  <p>The estimate is for a policy in force all next year. Low, Medium, and High are the lower, middle, and upper third of those scores.</p>
                </article>
              </div>
              <div class="steps">
                <div><b>01</b><span>The question and the 2024 check</span></div>
                <div><b>02</b><span>Six models, side by side</span></div>
                <div><b>03</b><span>Which facts move the score</span></div>
                <div><b>04</b><span>An estimate for one full year</span></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with compare:
        with st.container(key="models_page"):
            slide_heading(
                "02",
                "Models",
                "How the six models compared",
                "Trained on 2022–2023 and checked on 2024. A higher sorting score is better.",
            )
            if scores is not None:
                table_col, chart_col = st.columns([1, 1.05], gap="medium")
                with table_col:
                    shown = scores.rename(columns={
                        "AUC": "Sorting score",
                        "calibration_gap": "Calibration gap",
                        "mean_predicted": "Predicted chance",
                    })
                    st.dataframe(shown.round(3), width="stretch")
                with chart_col:
                    plot_df = scores.sort_values("AUC", ascending=True)
                    fig, ax = plt.subplots(figsize=(5.6, 3.4))
                    style_axes(fig, ax)
                    bar_colors = [NAVY if name == "LightGBM" else SLATE for name in plot_df.index]
                    ax.barh(plot_df.index.astype(str), plot_df["AUC"], color=bar_colors)
                    span = float(plot_df["AUC"].max() - plot_df["AUC"].min())
                    ax.set_xlim(
                        float(plot_df["AUC"].min()) - max(0.02, span),
                        float(plot_df["AUC"].max()) + 0.015,
                    )
                    ax.set_xlabel("Sorting score on 2024 (higher is better)")
                    for i, value in enumerate(plot_df["AUC"]):
                        ax.text(value + 0.001, i, f"{value:.3f}", va="center", fontsize=9, color=INK)
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    st.caption("The darkest bar is the model used in the app. The axis is zoomed in.")

    with estimate:
        with st.container(key="estimate_page"):
            slide_heading(
                "04",
                "Estimate",
                "A full next year",
                "Enter the driver, the car, the coverage, and where the car is driven.",
            )
        form_col, result_col = st.columns([1.85, 0.62], gap="medium")
        with form_col:
            with st.form("driver_info"):
                a, b, c = st.columns(3)
                with a:
                    driver_age = st.slider("Driver age", 16, 90, 47)
                with b:
                    years_licensed = st.slider("Years with a licence", 0, 60, 20)
                with c:
                    vehicle_age = st.slider("Vehicle age (years)", 0, 40, 24)
                d, e, f = st.columns(3)
                with d:
                    value_cap = int(levels.get("vehicle_value_cap", 150000))
                    vehicle_value = st.number_input(
                        "Vehicle value", 1000, max(value_cap, 1000), min(25000, value_cap), step=500,
                    )
                with e:
                    vehicle_brand = pick("vehicle_brand", "Brand")
                with f:
                    fuel_type = pick("fuel_type", "Fuel")
                g, h, i = st.columns(3)
                with g:
                    bonus_score = pick(
                        "bonus_score",
                        "Past claims",
                        "The insurer's record from before this year. Good means few past claims. "
                        "Neutral is ordinary. Poor means a history of claims.",
                    )
                with h:
                    business_type = pick("business_type", "Policy is")
                with i:
                    payment_frequency = pick(
                        "payment_frequency",
                        "Payment",
                        "How often the premium is paid. In this data, paying more often goes with a higher claim rate. "
                        "That is a clue about the customer, not proof that the payment plan causes a claim.",
                    )
                policy_type = pick("policy_type", "Coverage")
                j, k, m = st.columns(3)
                with j:
                    municipality_type = pick("municipality_type", "Home area")
                with k:
                    circulation_area = pick("circulation_area", "Usual driving")
                with m:
                    seats = st.slider("Seats", 2, 9, 5)
                power_to_weight_ratio = st.number_input(
                    "Weight per horsepower", 4.0, 40.0, 12.0, step=0.1,
                )
                submitted = st.form_submit_button("Estimate my risk", width="stretch")

        if submitted:
            user = pd.DataFrame([{
                "driver_age": driver_age,
                "vehicle_age": vehicle_age,
                "age_driving_licence": years_licensed,
                "vehicle_value": vehicle_value,
                "seats": seats,
                "power_to_weight_ratio": power_to_weight_ratio,
                "bonus_score": bonus_score,
                "policy_type": policy_type,
                "business_type": business_type,
                "payment_frequency": payment_frequency,
                "fuel_type": fuel_type,
                "vehicle_brand": vehicle_brand,
                "municipality_type": municipality_type,
                "circulation_area": circulation_area,
            }])
            prob = float(predict_claim_probability(bundle, user)[0])
            tier = assign_risk_tier([prob], cutoffs)[0]
            st.session_state["last_prob"] = prob
            st.session_state["last_tier"] = tier

        with result_col:
            with st.container(key="result_card"):
                st.markdown('<p class="kicker">Result</p>', unsafe_allow_html=True)
                if "last_prob" in st.session_state:
                    st.metric("Chance of a claim next year", f"{st.session_state['last_prob']:.1%}")
                    st.metric("Compared with other policies", st.session_state["last_tier"])
                    st.caption(
                        f"Low is under {cutoffs['Low']:.1%}, Medium is under {cutoffs['Medium']:.1%}, "
                        "and High is above that."
                    )
                else:
                    st.metric("Chance of a claim next year", "—")
                    st.caption("The chance appears here after you estimate.")

    with drivers:
        with st.container(key="drivers_page"):
            slide_heading(
                "03",
                "Drivers of the score",
                "What the model uses most",
                "Each bar is how much that fact is used. Time on cover is largest because a short policy has less time to claim.",
            )
            shares = feature_shares(bundle)
            chart_col, rank_col = st.columns([1.45, 0.7], gap="medium")
            with chart_col:
                fig, ax = plt.subplots(figsize=(6.2, 4.2))
                style_axes(fig, ax)
                plot = shares.sort_values("Share", ascending=True)
                ax.barh(plot["Fact"], plot["Share"], color=NAVY)
                ax.set_xlabel("Share of the model's splits")
                ax.set_xlim(0, float(plot["Share"].max()) * 1.18)
                for i, value in enumerate(plot["Share"]):
                    ax.text(value + 0.008, i, f"{value:.0%}", va="center", fontsize=9, color=INK)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            with rank_col:
                top = shares.head(5)
                widest = float(top["Share"].max()) or 1
                rows = "".join(
                    f'<div class="rank-row"><span>{fact}</span><b>{share:.0%}</b>'
                    f'<i style="width:{share / widest * 100:.0f}%"></i></div>'
                    for fact, share in zip(top["Fact"], top["Share"])
                )
                st.markdown(
                    f'<h3 style="margin:0.15rem 0 0.8rem 0;color:{NAVY};font-size:1rem;">Largest shares</h3>{rows}'
                    '<p class="foot">The estimate holds time on cover at a full year.</p>',
                    unsafe_allow_html=True,
                )

deck_controls(3 if submitted else None)

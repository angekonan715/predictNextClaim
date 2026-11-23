"""Streamlit app: a sideways dashboard of claim risk.

Run from the project root:
    streamlit run app/app.py
"""
import io
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


def show_chart(fig) -> None:
    """Draw a chart at the width of its column, with no scrollbar."""
    image = io.BytesIO()
    fig.savefig(image, format="png", bbox_inches="tight", facecolor=CARD, dpi=130)
    plt.close(fig)
    st.image(image, width="stretch")


def comparison_table(frame: pd.DataFrame) -> None:
    """A plain table that fits the page. No sideways scrollbar."""
    labels = {
        "AUC": "Sorting score",
        "Gini": "Gini",
        "Brier": "Brier",
        "calibration_gap": "Calibration gap",
        "mean_predicted": "Predicted chance",
    }
    shown = frame.rename(columns=labels).round(3)
    header = "".join(f"<th>{name}</th>" for name in ["Model", *shown.columns])
    body = []
    for name, row in shown.iterrows():
        cells = "".join(f"<td>{row[column]:.3f}</td>" for column in shown.columns)
        mark = "chosen" if name == "LightGBM" else ""
        body.append(f'<tr class="{mark}"><td>{name}</td>{cells}</tr>')
    st.markdown(
        f'<table class="fit-table"><thead><tr>{header}</tr></thead><tbody>{"".join(body)}</tbody></table>',
        unsafe_allow_html=True,
    )


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
    .st-key-chosen_page,
    .st-key-versus_page,
    .st-key-settings_page,
    .st-key-data_page,
    .st-key-prep_page,
    .st-key-dict_page,
    .st-key-reading_page,
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
    .why-grid {{ grid-template-columns: 1fr 1fr; }}
    .steps {{
      display: grid;
      grid-template-columns: repeat(5, 1fr);
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
    .fit-table {{
      width: 100%;
      border-collapse: collapse;
      background: {CARD};
      border: 1px solid {LINE};
      border-radius: 12px;
      overflow: hidden;
    }}
    .fit-table th, .fit-table td {{
      padding: 0.32rem 0.55rem;
      text-align: right;
      font-size: 0.92rem;
      color: {INK};
      border-bottom: 1px solid {LINE};
    }}
    .fit-table td:not(:first-child) {{
      white-space: nowrap;
    }}
    .fit-table th {{
      color: {MUTED};
      font-weight: 650;
    }}
    .fit-table th:first-child, .fit-table td:first-child {{
      text-align: left;
    }}
    .fit-table tr:last-child td {{
      border-bottom: 0;
    }}
    .fit-table tr.chosen td {{
      background: #E7EEF3;
      color: {NAVY};
      font-weight: 650;
    }}
    .fit-table.dict th, .fit-table.dict td {{
      text-align: left;
      white-space: normal;
      font-size: 0.88rem;
      line-height: 1.35;
    }}
    [data-testid="stImage"] img {{
      max-width: 100%;
      height: auto;
      display: block;
      margin: 0.35rem auto 0 auto;
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
      padding-bottom: 0.15rem;
      margin-bottom: 0;
    }}
    .st-key-deck > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:has(.st-key-estimate_page) [data-testid="stVerticalBlock"] {{
      gap: 0.35rem;
    }}
    .st-key-result_card {{
      background: {NAVY};
      border-radius: 18px;
      padding: 1.15rem 1.15rem 0.9rem 1.15rem;
      box-shadow: 0 16px 36px rgba(27, 58, 75, 0.18);
    }}
    .st-key-result_card .kicker,
    .st-key-result_card .kicker span {{
      color: #FFFFFF;
      font-size: 0.95rem;
    }}
    .st-key-result_card div[data-testid="stMetric"] {{
      background: rgba(255, 255, 255, 0.12);
      border-color: rgba(255, 255, 255, 0.28);
    }}
    .st-key-result_card [data-testid="stMetricLabel"],
    .st-key-result_card [data-testid="stMetricLabel"] p,
    .st-key-result_card [data-testid="stCaptionContainer"],
    .st-key-result_card [data-testid="stCaptionContainer"] p {{
      color: #FFFFFF !important;
      font-size: 1.05rem !important;
      line-height: 1.35 !important;
      -webkit-text-fill-color: #FFFFFF;
    }}
    .st-key-result_card div[data-testid="stMetricValue"],
    .st-key-result_card div[data-testid="stMetricValue"] div {{
      color: #FFFFFF !important;
      font-size: 2.5rem !important;
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
    (
        overview, data, dictionary, prep, compare, choice, versus, settings, drivers, reading, estimate,
    ) = st.columns(11, gap="small")

    with overview:
        auc = gap = chance = "—"
        if scores is not None and "LightGBM" in scores.index:
            best = scores.loc["LightGBM"]
            auc = f"{best['AUC']:.3f}"
            gap = f"{best['calibration_gap']:.3f}"
            chance = f"{best['mean_predicted']:.0%}"
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
              <p class="foot" style="border:0;padding-top:0;margin-top:0">Move sideways, in the order the work was done: where the data comes from, the fields, the cleaning, the six models, why LightGBM won, and then the estimate.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with data:
        with st.container(key="data_page"):
            st.markdown(
                """
                <div class="board">
                  <header class="board-head">
                    <p class="kicker"><span>02</span> The data</p>
                    <h2>Where the policies come from</h2>
                    <p>Anonymised car policies from a Spanish insurer, published by Espinosa, Lledó, and Atance del Olmo (2026) on Mendeley Data.</p>
                  </header>
                  <div class="metrics">
                    <div class="metric"><span>Policy-years</span><b>354,140</b><p>One row is one policy in one calendar year.</p></div>
                    <div class="metric"><span>Columns in the file</span><b>47</b><p>There is no gender column.</p></div>
                    <div class="metric"><span>Years in the file</span><b>2022–2024</b><p>The same policy keeps the same anonymous id.</p></div>
                  </div>
                  <div class="board-grid why-grid">
                    <article class="panel">
                      <h3>What was used to learn</h3>
                      <p>2022 and 2023: 185,720 rows. About 13.6% of those policies had at least one claim. A claim is recorded as 1 when the claim count is above zero, and 0 otherwise.</p>
                      <p class="foot">Most policy-years have no claim. The largest count in the file is 18.</p>
                    </article>
                    <article class="panel">
                      <h3>What was kept for the check</h3>
                      <p>2024 was not used while the model learned. It has 167,853 rows, and about 15.1% of them claimed. That is the year the scores on the first slide come from.</p>
                      <p class="foot">Cleaning later keeps 353,573 of the 354,140 rows. The next slide names the fields that matter most.</p>
                    </article>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with compare:
        with st.container(key="models_page"):
            slide_heading(
                "05",
                "Models",
                "How the six models compared",
                "Trained on 2022–2023 and checked on 2024. A higher sorting score is better.",
            )
            if scores is not None:
                comparison_table(scores)
                plot_df = scores.sort_values("AUC", ascending=True)
                fig, ax = plt.subplots(figsize=(8.0, 1.55))
                style_axes(fig, ax)
                bar_colors = [NAVY if name == "LightGBM" else SLATE for name in plot_df.index]
                ax.barh(plot_df.index.astype(str), plot_df["AUC"], color=bar_colors)
                span = float(plot_df["AUC"].max() - plot_df["AUC"].min())
                ax.set_xlim(
                    float(plot_df["AUC"].min()) - max(0.02, span),
                    float(plot_df["AUC"].max()) + 0.012,
                )
                ax.set_xlabel("Sorting score on 2024 (higher is better)")
                for i, value in enumerate(plot_df["AUC"]):
                    ax.text(value + 0.001, i, f"{value:.3f}", va="center", fontsize=9, color=INK)
                fig.tight_layout()
                show_chart(fig)
                st.caption("The darkest bar is the model used in the app. The axis is zoomed in.")

    with reading:
        with st.container(key="reading_page"):
            st.markdown(
                """
                <div class="board">
                  <header class="board-head">
                    <p class="kicker"><span>10</span> Reading the results</p>
                    <h2>What the results mean</h2>
                    <p>The model is useful for sorting drivers. It is not a perfect prediction of who will claim.</p>
                  </header>
                  <div class="board-grid why-grid">
                    <article class="panel">
                      <h3>The 2024 check</h3>
                      <p>LightGBM’s 0.692 sorting score beats a coin flip. The 0.003 calibration gap means the average predicted chance matches the claims that happened.</p>
                      <p class="foot">About 15% of 2024 policies claimed, and the model predicted about 15%.</p>
                    </article>
                    <article class="panel">
                      <h3>What moves a score</h3>
                      <p>Time on cover is about half of the model’s splits, because a partial year has less time to claim. The estimate fixes that at a full year, so coverage, vehicle value, and vehicle age are what change from one driver to the next.</p>
                      <p class="foot">Broader cover shows a higher claim rate partly because it pays for more kinds of damage.</p>
                    </article>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with estimate:
        with st.container(key="estimate_page"):
            slide_heading(
                "11",
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
                j, k = st.columns(2)
                with j:
                    circulation_area = pick("circulation_area", "Usual driving")
                with k:
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
                "power_to_weight_ratio": power_to_weight_ratio,
                "bonus_score": bonus_score,
                "policy_type": policy_type,
                "business_type": business_type,
                "payment_frequency": payment_frequency,
                "fuel_type": fuel_type,
                "vehicle_brand": vehicle_brand,
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
                    st.metric("Chance next year", f"{st.session_state['last_prob']:.1%}")
                    st.metric("Risk group", st.session_state["last_tier"])
                    st.caption(
                        f"Low is under {cutoffs['Low']:.1%}. Medium is under {cutoffs['Medium']:.1%}. "
                        "High is above that."
                    )
                else:
                    st.metric("Chance next year", "—")
                    st.metric("Risk group", "—")
                    st.caption("Press Estimate my risk. The chance and the group appear here.")

    with choice:
        with st.container(key="chosen_page"):
            st.markdown(
                """
                <div class="board">
                  <header class="board-head">
                    <p class="kicker"><span>06</span> The choice</p>
                    <h2>Why LightGBM</h2>
                    <p>Six models learned from 2022 and 2023. They were then scored on 2024, a year they had not seen.</p>
                  </header>
                  <div class="board-grid why-grid">
                    <article class="panel">
                      <h3>The rule</h3>
                      <p>The highest sorting score wins. A simpler model can replace it only when it is within 0.005 on that score and within 0.002 on the calibration gap.</p>
                      <p class="foot">LightGBM led at 0.692. XGBoost was 0.682, which is 0.010 behind, so the simpler-model rule did not apply.</p>
                    </article>
                    <article class="panel">
                      <h3>How to read the scores</h3>
                      <p>A sorting score of 0.50 is a coin flip. 0.692 means LightGBM puts drivers who claimed above drivers who did not, more often than chance. It is not a perfect ranking.</p>
                      <p class="foot">The calibration gap is 0.003. A predicted 15% happened about 15% of the time on the 2024 policies.</p>
                    </article>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with versus:
        with st.container(key="versus_page"):
            st.markdown(
                """
                <div class="board">
                  <header class="board-head">
                    <p class="kicker"><span>07</span> Two boosted models</p>
                    <h2>LightGBM and XGBoost</h2>
                    <p>Both add 300 small trees. Each new tree corrects the policies the earlier trees scored too high or too low. The learning rate, 0.05, keeps each correction small.</p>
                  </header>
                  <div class="board-grid why-grid">
                    <article class="panel">
                      <h3>XGBoost</h3>
                      <p>It grows a tree level by level. Every branch stops after 4 questions. Each tree sees a random 80% of the policies and 80% of the columns.</p>
                      <p class="foot">Its 2024 sorting score was 0.682, with the same 0.003 calibration gap as LightGBM.</p>
                    </article>
                    <article class="panel">
                      <h3>LightGBM</h3>
                      <p>It splits the leaf that reduces the error the most, so one side of a tree can be deeper than the other. A tree may have at most 31 end groups. Each tree uses every training row and every column.</p>
                      <p class="foot">It also groups similar numbers into bins before it looks for a split, which is why it is built for large files.</p>
                    </article>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with settings:
        with st.container(key="settings_page"):
            st.markdown(
                """
                <div class="board">
                  <header class="board-head">
                    <p class="kicker"><span>08</span> The chosen settings</p>
                    <h2>LightGBM parameters</h2>
                    <p>These are the settings saved in the app. No class weights were used, and training did not stop early.</p>
                  </header>
                  <table class="fit-table">
                    <thead><tr><th>Setting</th><th>Value</th><th>What it does</th></tr></thead>
                    <tbody>
                      <tr><td>Trees</td><td>300</td><td>How many correcting trees are added</td></tr>
                      <tr><td>Learning rate</td><td>0.05</td><td>How large each tree's correction is</td></tr>
                      <tr><td>Leaves</td><td>31</td><td>Most end groups one tree may have</td></tr>
                      <tr><td>Minimum policies</td><td>200</td><td>A split must leave at least this many policies on each side</td></tr>
                      <tr><td>Random start</td><td>42</td><td>A rerun builds the same model</td></tr>
                    </tbody>
                  </table>
                  <p class="foot" style="border:0">Numbers stay on their original scale. Categories such as coverage and brand become yes-or-no columns. Time on cover is one of those columns.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with prep:
        with st.container(key="prep_page"):
            st.markdown(
                """
                <div class="board">
                  <header class="board-head">
                    <p class="kicker"><span>04</span> Preparing the data</p>
                    <h2>Feature engineering</h2>
                    <p>No new calculated columns were added. The work was cleaning and grouping, using 2022–2023 only, so 2024 could not influence the choices.</p>
                  </header>
                  <div class="board-grid why-grid">
                    <article class="panel">
                      <h3>What was built</h3>
                      <p>The target is 1 when the policy had at least one claim, and 0 otherwise. Missing fuel is kept as its own group, Unknown.</p>
                      <p class="foot">Vehicle value is capped at the training 99th percentile, 71,122. A brand with fewer than 1,000 training policies is grouped as Other.</p>
                    </article>
                    <article class="panel">
                      <h3>What was left out</h3>
                      <p>Rows were dropped when time on cover was not positive, a required number was missing, driver age was under 16, or an age or the weight ratio was negative. That kept 353,573 of 354,140 rows.</p>
                      <p class="foot">Premiums, claim counts, claim costs, and policy status stay out. Seats and home area stay out too: each was under 1% of the splits, and the claim rate barely moved.</p>
                    </article>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with dictionary:
        with st.container(key="dict_page"):
            st.markdown(
                """
                <div class="board">
                  <header class="board-head">
                    <p class="kicker"><span>03</span> Data dictionary</p>
                    <h2>The facts that matter most</h2>
                    <p>These are the fields the model uses most, in plain words.</p>
                  </header>
                  <table class="fit-table dict">
                    <thead><tr><th>Fact</th><th>Meaning</th><th>What the data showed</th></tr></thead>
                    <tbody>
                      <tr><td>Time on cover</td><td>Fraction of the year the policy was in force. A full year is 1.</td><td>A short policy has less time to claim. The app sets this to 1.</td></tr>
                      <tr><td>Coverage</td><td>Third party, third party plus glass, combined, or comprehensive.</td><td>Third party is about 10 claims per 100 years. Comprehensive with no excess is about 41.</td></tr>
                      <tr><td>Vehicle value</td><td>Declared insured value of the car, capped at 71,122.</td><td>More expensive cars claim more often.</td></tr>
                      <tr><td>Vehicle age</td><td>Age of the car, in years.</td><td>Cars under 5 years claim about 31 per 100 years. Cars over 20 claim about 19.</td></tr>
                      <tr><td>Weight per horsepower</td><td>Kilograms per horsepower. Higher means a heavier, less powerful car.</td><td>The claim rate falls a little as this number rises.</td></tr>
                      <tr><td>Driver age</td><td>Age of the main driver. The file runs from 18 to 90.</td><td>The claim rate falls as age rises.</td></tr>
                      <tr><td>Payment</td><td>Billed once a year, twice a year, or every three months.</td><td>Quarterly payers claim more often. That is a clue, not a cause.</td></tr>
                    </tbody>
                  </table>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with drivers:
        with st.container(key="drivers_page"):
            slide_heading(
                "09",
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
                show_chart(fig)
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

deck_controls(10 if submitted else None)

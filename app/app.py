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


@st.cache_data
def load_policies() -> pd.DataFrame:
    """Cleaned policy-years used for the relationship charts."""
    frames = [pd.read_csv(path) for path in (config.TRAIN_FILE, config.TEST_FILE)]
    data = pd.concat(frames, ignore_index=True)
    return data.loc[data[config.EXPOSURE_COL] > 0].copy()


def per_hundred_years(frame: pd.DataFrame) -> float:
    """Claims per 100 years on cover."""
    exposure = float(frame[config.EXPOSURE_COL].sum())
    if exposure <= 0:
        return float("nan")
    return 100 * float(frame[config.TARGET].sum()) / exposure


def correlation_chart(data: pd.DataFrame):
    """How the numeric fields move together, and with the claim flag."""
    labels = [
        "Driver age",
        "Vehicle age",
        "Years licensed",
        "Vehicle value",
        "Weight per hp",
        "Claim",
    ]
    matrix = data[config.NUMERIC_FEATURES + [config.TARGET]].corr().to_numpy()
    fig, ax = plt.subplots(figsize=(7.2, 3.05))
    style_axes(fig, ax)
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = matrix[i, j]
            ax.text(
                j, i, f"{value:.2f}",
                ha="center", va="center", fontsize=8,
                color="white" if abs(value) > 0.55 else INK,
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=8, colors=MUTED)
    fig.tight_layout()
    return fig


def rate_bars(ax, labels: list[str], rates: list[float], title: str) -> None:
    ax.barh(labels[::-1], rates[::-1], color=NAVY)
    ax.set_title(title, loc="left", color=NAVY, fontsize=9, pad=2)
    ax.tick_params(labelsize=7, colors=INK)
    ax.set_xlim(0, max(rates) * 1.25 if rates else 1)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(LINE)
    ax.spines["bottom"].set_color(LINE)
    ax.set_facecolor(CARD)


def claim_rate_chart(data: pd.DataFrame):
    """Claim rate for every field the model uses."""
    fig, axes = plt.subplots(3, 4, figsize=(8.6, 3.45))
    fig.patch.set_facecolor(CARD)
    panels = []

    numeric_bins = [
        ("Driver age", "driver_age", [16, 26, 36, 51, 66, 91], ["18–25", "26–35", "36–50", "51–65", "66+"]),
        ("Vehicle age", "vehicle_age", [-0.1, 5, 10, 20, 80], ["0–4", "5–9", "10–19", "20+"]),
        ("Years licensed", "age_driving_licence", [-0.1, 10, 20, 30, 40, 81], ["0–9", "10–19", "20–29", "30–39", "40+"]),
        ("Vehicle value", "vehicle_value", [-0.1, 8000, 15000, 25000, 40000, 10**9], ["Under 8k", "8–15k", "15–25k", "25–40k", "40k+"]),
        ("Weight per hp", "power_to_weight_ratio", [-0.1, 8, 11, 14, 18, 10**9], ["Under 8", "8–11", "11–14", "14–18", "18+"]),
    ]
    for title, column, edges, names in numeric_bins:
        groups = pd.cut(data[column], edges, labels=names, right=False)
        rates = [per_hundred_years(data.loc[groups == name]) for name in names]
        panels.append((title, names, rates))

    category_order = {
        "policy_type": ["TP", "TPG", "CC", "COMP_E", "COMP_N"],
        "payment_frequency": ["A", "S", "Q"],
        "fuel_type": ["D", "G", "Unknown"],
        "bonus_score": ["G", "N", "B"],
        "business_type": ["NB", "P"],
        "circulation_area": ["U", "R"],
    }
    category_titles = {
        "policy_type": "Coverage",
        "payment_frequency": "Payment",
        "fuel_type": "Fuel",
        "bonus_score": "Past claims",
        "business_type": "Policy",
        "circulation_area": "Where driven",
    }
    for column, codes in category_order.items():
        labels = [FRIENDLY.get(column, {}).get(code, code) for code in codes]
        short = [label.replace("Comprehensive, no excess", "Comp., no excess")
                 .replace("Comprehensive with excess", "Comp. with excess")
                 .replace("Third party plus glass", "TP plus glass")
                 .replace("Third party", "Third party")
                 .replace("Every three months", "Quarterly")
                 .replace("Twice a year", "Twice a year")
                 .replace("Once a year", "Annual")
                 .replace("Neutral record", "Neutral")
                 .replace("Good record", "Good")
                 .replace("Poor record", "Poor")
                 .replace("Existing policy", "Existing")
                 .replace("New policy", "New")
                 for label in labels]
        present = [code for code in codes if (data[column] == code).any()]
        rates = [per_hundred_years(data.loc[data[column] == code]) for code in present]
        shown = [short[codes.index(code)] for code in present]
        panels.append((category_titles[column], shown, rates))

    for ax, (title, labels, rates) in zip(axes.ravel(), panels):
        rate_bars(ax, labels, rates, title)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.tight_layout(pad=0.4)
    return fig


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
        if (start !== null) {{
          go(start);
          const target = slides()[start];
          if (target) target.scrollTop = 0;
        }}
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
    .st-key-heat_page,
    .st-key-rates_page,
    .st-key-prep_page,
    .st-key-dict_page,
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
      padding-top: 0.55rem;
      padding-bottom: 0.2rem;
    }}
    .st-key-deck > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:has(.st-key-estimate_page) .stForm {{
      background: {CARD};
      box-shadow: 0 10px 28px rgba(27, 58, 75, 0.07);
      padding-bottom: 0.15rem;
      margin-bottom: 0;
    }}
    .st-key-deck > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:has(.st-key-estimate_page) [data-testid="stVerticalBlock"] {{
      gap: 0.05rem;
    }}
    .st-key-result_card {{
      background: {NAVY};
      border-radius: 16px;
      padding: 0.55rem 0.9rem 0.4rem 0.9rem;
      box-shadow: 0 12px 28px rgba(27, 58, 75, 0.16);
      margin: 0 0 0.35rem 0;
    }}
    .st-key-result_card .kicker,
    .st-key-result_card .kicker span {{
      color: #FFFFFF;
      font-size: 0.95rem;
    }}
    .st-key-result_card div[data-testid="stMetric"] {{
      background: rgba(255, 255, 255, 0.12);
      border-color: rgba(255, 255, 255, 0.28);
      padding: 0.35rem 0.75rem;
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
      font-size: 2rem !important;
    }}
    .st-key-result_card [data-testid="stHorizontalBlock"] {{
      gap: 0.6rem;
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


@st.cache_data
def relationship_images(train_mtime: float, test_mtime: float, layout: int = 2) -> tuple[bytes, bytes, str]:
    """Charts for the relationship slides. Rebuilt only when the data files change."""
    data = load_policies()

    def as_png(fig) -> bytes:
        image = io.BytesIO()
        fig.savefig(image, format="png", bbox_inches="tight", facecolor=CARD, dpi=120)
        plt.close(fig)
        return image.getvalue()

    brand_rates = (
        data.groupby("vehicle_brand", observed=True)
        .apply(per_hundred_years, include_groups=False)
        .sort_values()
    )
    caption = (
        f"Brand is the remaining field. The rate runs from {brand_rates.iloc[0]:.0f} "
        f"per 100 years for {brand_rates.index[0]} to {brand_rates.iloc[-1]:.0f} "
        f"for {brand_rates.index[-1]}."
    )
    return as_png(correlation_chart(data)), as_png(claim_rate_chart(data)), caption


if __import__("os").environ.get("SAVE_RELATIONSHIP_CHARTS") == "1":
    heat_png, rate_png, brand_line = relationship_images(0, 0, 2)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    (config.FIGURES_DIR / "correlation_heatmap.png").write_bytes(heat_png)
    (config.FIGURES_DIR / "claim_rate_by_field.png").write_bytes(rate_png)
    print(brand_line)
    raise SystemExit(0)

comparison_file = config.FIGURES_DIR / "model_comparison.csv"
scores = pd.read_csv(comparison_file, index_col=0) if comparison_file.exists() else None

with st.container(key="deck"):
    (
        overview, data, dictionary, prep, heat, rates, compare, choice, versus, settings, drivers, estimate,
    ) = st.columns(12, gap="small")

    with overview:
        auc = gap = chance = "—"
        if scores is not None and "LightGBM" in scores.index:
            best = scores.loc["LightGBM"]
            auc = f"{best['AUC']:.3f}"
            gap = f"{best['calibration_gap']:.3f}"
            chance = f"{best['mean_predicted']:.0%}"
        st.markdown(
            f"""
            <div class="board">
              <header class="board-head">
                <p class="kicker"><span>01</span> Overview</p>
                <h2>Next-year claim risk</h2>
                <p>Chance of at least one claim over a full policy year. Trained on 2022–2023 and checked on 2024. This is a study result, not a quote.</p>
              </header>
              <div class="metrics">
                <div class="metric"><span>Sorting score</span><b>{auc}</b><p>0.50 is a coin flip. Closer to 1 ranks claimants higher.</p></div>
                <div class="metric"><span>Calibration gap</span><b>{gap}</b><p>Distance between the predicted rate and the rate that occurred.</p></div>
                <div class="metric"><span>Average predicted chance</span><b>{chance}</b><p>Close to the 15% of 2024 policies that claimed.</p></div>
              </div>
              <div class="board-grid story">
                <article class="panel">
                  <h3>Scope</h3>
                  <p>The model uses facts known on the first day of the policy year: the driver, the car, the coverage, and where the car is driven.</p>
                </article>
                <article class="panel">
                  <h3>Risk groups</h3>
                  <p>Low, Medium, and High are the lower, middle, and upper third of full-year scores on the 2024 policies.</p>
                </article>
              </div>
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
                    <h2>Source and design</h2>
                    <p>Anonymised motor policies from a Spanish insurer. Espinosa, Lledó, and Atance del Olmo (2026), Mendeley Data.</p>
                  </header>
                  <div class="metrics">
                    <div class="metric"><span>Policy-years</span><b>354,140</b><p>One row is one policy in one calendar year.</p></div>
                    <div class="metric"><span>Columns</span><b>47</b><p>No gender field. The policy id is stable across years.</p></div>
                    <div class="metric"><span>Years</span><b>2022–2024</b><p>Train on the first two years. Check on 2024.</p></div>
                  </div>
                  <div class="board-grid why-grid">
                    <article class="panel">
                      <h3>Training</h3>
                      <p>185,720 rows in 2022–2023. A claim is 1 when the count is above zero. About 13.6% claimed. The largest count in the file is 18.</p>
                    </article>
                    <article class="panel">
                      <h3>Holdout</h3>
                      <p>167,853 rows in 2024, held out of training. About 15.1% claimed.</p>
                    </article>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with heat:
        with st.container(key="heat_page"):
            slide_heading(
                "05",
                "Relationships",
                "Numeric fields and a claim",
                "Each cell is a correlation. 1 means the two fields rise together. The Claim row is the straight-line link with having at least one claim.",
            )
            heat_file = config.FIGURES_DIR / "correlation_heatmap.png"
            if heat_file.exists():
                st.image(str(heat_file), width="stretch")
            st.caption(
                "Driver age and vehicle age move together (0.86). "
                "The strongest straight-line link with a claim is vehicle value (0.07). "
                "Categories are on the next slide."
            )

    with rates:
        with st.container(key="rates_page"):
            slide_heading(
                "06",
                "Patterns",
                "Claim rate by field",
                "Claims per 100 years on cover, for each field the model uses. A straight line misses several of these bends.",
            )
            rate_file = config.FIGURES_DIR / "claim_rate_by_field.png"
            if rate_file.exists():
                st.image(str(rate_file), width="stretch")
            st.caption(
                "Brand is the remaining field. The rate runs from 15 per 100 years for DAEWOO to 27 for BMW."
            )

    with compare:
        with st.container(key="models_page"):
            slide_heading(
                "07",
                "Models",
                "How the six models compared",
                "Same training rows, same 2024 check. The highlighted row is the model in this app.",
            )
            if scores is not None:
                comparison_table(scores)

    with estimate:
        with st.container(key="estimate_page"):
            slide_heading(
                "12",
                "Estimate",
                "One full policy year",
                "Facts known on day one. Time on cover is fixed at 1.",
            )
            result_slot = st.empty()
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
                        "Insurer record from before this year: good, neutral, or poor.",
                    )
                with h:
                    business_type = pick("business_type", "Policy is")
                with i:
                    payment_frequency = pick(
                        "payment_frequency",
                        "Payment",
                        "Billing schedule. A higher claim rate here is an association, not a cause.",
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

        with result_slot.container():
            with st.container(key="result_card"):
                st.markdown('<p class="kicker">Result</p>', unsafe_allow_html=True)
                chance_col, group_col = st.columns(2)
                if "last_prob" in st.session_state:
                    with chance_col:
                        st.metric("Chance next year", f"{st.session_state['last_prob']:.1%}")
                    with group_col:
                        st.metric("Risk group", st.session_state["last_tier"])
                    st.caption(
                        f"Low is under {cutoffs['Low']:.1%}. Medium is under {cutoffs['Medium']:.1%}. "
                        "High is above that."
                    )
                else:
                    with chance_col:
                        st.metric("Chance next year", "—")
                    with group_col:
                        st.metric("Risk group", "—")
                    st.caption("The chance and the group appear here.")

    with choice:
        with st.container(key="chosen_page"):
            st.markdown(
                """
                <div class="board">
                  <header class="board-head">
                    <p class="kicker"><span>08</span> The choice</p>
                    <h2>Why LightGBM</h2>
                    <p>Highest 2024 sorting score, unless a simpler model is within 0.005 and its calibration gap is within 0.002.</p>
                  </header>
                  <div class="board-grid why-grid">
                    <article class="panel">
                      <h3>Result</h3>
                      <p>LightGBM 0.692. XGBoost 0.682. The gap is 0.010, outside the 0.005 allowance, so LightGBM stays.</p>
                    </article>
                    <article class="panel">
                      <h3>Calibration</h3>
                      <p>Both models have a gap of 0.003. The probabilities are equally honest. Sorting is what separates them.</p>
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
                    <p class="kicker"><span>09</span> Two boosted models</p>
                    <h2>LightGBM and XGBoost</h2>
                    <p>Both add trees in sequence. Each new tree corrects policies the earlier trees scored too high or too low.</p>
                  </header>
                  <div class="board-grid why-grid">
                    <article class="panel">
                      <h3>XGBoost</h3>
                      <p>Grows level by level, at most 4 questions deep. Each tree sees a random 80% of the rows and 80% of the columns.</p>
                    </article>
                    <article class="panel">
                      <h3>LightGBM</h3>
                      <p>Splits the leaf that reduces error most, so one side can be deeper. At most 31 leaves. Every row and column. Continuous values are binned before a split.</p>
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
                    <p class="kicker"><span>10</span> The chosen settings</p>
                    <h2>LightGBM parameters</h2>
                    <p>No class weights. Training ran for all 300 trees.</p>
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
                  <p class="foot" style="border:0">Numeric fields stay on their original scale. Categories become indicator columns.</p>
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
                    <h2>Preparation</h2>
                    <p>No calculated columns. Caps and brand groups were learned on 2022–2023 and applied to 2024.</p>
                  </header>
                  <div class="board-grid why-grid">
                    <article class="panel">
                      <h3>Adjustments</h3>
                      <p>Missing fuel is coded Unknown. Vehicle value is capped at 71,122. Brands under 1,000 training policies become Other. 353,573 of 354,140 rows remain.</p>
                    </article>
                    <article class="panel">
                      <h3>Excluded</h3>
                      <p>Premiums, claim counts, claim costs, and policy status are known only after the year starts. Seats and home area were removed: each was 0.6% of the splits.</p>
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
                    <h2>Fields with the largest effect</h2>
                    <p>Definitions for the inputs the model uses most.</p>
                  </header>
                  <table class="fit-table dict">
                    <thead><tr><th>Fact</th><th>Meaning</th><th>What the data showed</th></tr></thead>
                    <tbody>
                      <tr><td>Time on cover</td><td>Fraction of the year the policy was in force. A full year is 1.</td><td>A shorter policy has less time in which to claim.</td></tr>
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
                "11",
                "Drivers of the score",
                "Share of splits",
                "Time on cover leads because a partial year has less time to claim. The estimate fixes it at 1.",
            )
            shares = feature_shares(bundle)
            fig, ax = plt.subplots(figsize=(8.0, 3.5))
            style_axes(fig, ax)
            plot = shares.sort_values("Share", ascending=True)
            ax.barh(plot["Fact"], plot["Share"], color=NAVY)
            ax.set_xlabel("Share of splits")
            ax.set_xlim(0, float(plot["Share"].max()) * 1.18)
            for i, value in enumerate(plot["Share"]):
                ax.text(value + 0.008, i, f"{value:.0%}", va="center", fontsize=9, color=INK)
            fig.tight_layout()
            show_chart(fig)

deck_controls(11 if submitted else None)

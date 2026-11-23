"""Build the senior-project slide deck. Run from the project root."""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
OUT = ROOT / "reports" / "presentation.pptx"

NAVY = RGBColor(0x1B, 0x3A, 0x4B)
INK = RGBColor(0x1F, 0x29, 0x37)
MUTED = RGBColor(0x4B, 0x55, 0x63)
ORANGE = RGBColor(0xC2, 0x56, 0x0C)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def add_text(slide, text, left, top, width, height, size=20, bold=False, color=INK, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    frame = box.text_frame
    frame.word_wrap = True
    p = frame.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = "Calibri"
    return frame


def bullets(slide, lines, left, top, width, height, size=22):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    frame = box.text_frame
    frame.word_wrap = True
    for i, line in enumerate(lines):
        p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.level = 0
        p.space_after = Pt(10)
        run = p.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.color.rgb = INK
        run.font.name = "Calibri"
    return frame


def chrome(slide, title):
    bar = slide.shapes.add_shape(1, Inches(0), Inches(0), prs.slide_width, Inches(0.18))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ORANGE
    bar.line.fill.background()
    add_text(slide, title, 0.55, 0.35, 12, 0.7, size=32, bold=True, color=NAVY)


def new_slide(title):
    slide = prs.slides.add_slide(BLANK)
    chrome(slide, title)
    return slide


# 1. Title
slide = prs.slides.add_slide(BLANK)
band = slide.shapes.add_shape(1, Inches(0), Inches(0), prs.slide_width, prs.slide_height)
band.fill.solid()
band.fill.fore_color.rgb = NAVY
band.line.fill.background()
accent = slide.shapes.add_shape(1, Inches(0), Inches(6.9), prs.slide_width, Inches(0.6))
accent.fill.solid()
accent.fill.fore_color.rgb = ORANGE
accent.line.fill.background()
add_text(slide, "Predicting next-year\ncar insurance claims", 0.8, 1.8, 11, 2.2, size=44, bold=True, color=WHITE)
add_text(slide, "Which drivers are most likely to file at least one claim?", 0.8, 4.3, 11, 0.6, size=22, color=WHITE)
add_text(slide, "Ange Konan  ·  Senior project  ·  Data Science major, Statistics minor", 0.8, 5.3, 11, 0.5, size=16, color=RGBColor(0xD1, 0xD5, 0xDB))

# 2. Question
slide = new_slide("The question")
bullets(slide, [
    "On the first day of a policy year, the insurer already knows the driver, the car, the cover, and where the car is driven.",
    "The model turns those facts into one number: the chance of at least one claim over a full next year.",
    "That number is a study result. It is not a price and it is not an insurance quote.",
], 0.7, 1.5, 11.8, 4.5, size=24)

# 3. Data
slide = new_slide("The data")
bullets(slide, [
    "354,140 anonymised policy-years from a Spanish insurer, 2022 through 2024.",
    "Each row is one policy in one year. About 14% of rows have at least one claim.",
    "The model learns from 2022 and 2023, then is tested on 2024.",
    "2024 is the stand-in for next year. The file has no later outcome.",
], 0.7, 1.5, 11.8, 4.8, size=24)

# 4. Rules
slide = new_slide("What the model is allowed to see")
bullets(slide, [
    "Kept: facts known on day one. Driver, car, coverage, place, and time on cover.",
    "Left out: claim counts and claim costs. Those are the answer.",
    "Left out: whether the policy was later cancelled. A cancellation can follow a claim.",
    "Left out: the premium. That would mostly copy the insurer's current price.",
    "Rare brands are grouped. Very high vehicle values are capped, using 2022–2023 only.",
], 0.7, 1.4, 11.8, 5.2, size=22)

# 5. Rates
slide = new_slide("Where the claim rate changes")
slide.shapes.add_picture(str(FIG / "claim_rates.png"), Inches(0.4), Inches(1.25), Inches(8.6), Inches(5.0))
bullets(slide, [
    "Younger drivers claim more often: about 32 per 100 years on cover at ages 18–25, about 17 after 65.",
    "Newer and more expensive cars claim more often.",
    "A poor past-claims record is about 34. A good record is about 20.",
    "Paying every three months goes with more claims. That is a clue, not a cause.",
], 9.1, 1.45, 3.9, 5.2, size=16)

# 6. Models
slide = new_slide("Six models, one test")
bullets(slide, [
    "GLM with a cloglog link. Time on cover enters as log(exposure), not as a fitted weight.",
    "Logistic regression, a four-question decision tree, and a random forest of 200 trees.",
    "XGBoost and LightGBM: 300 small trees. Each new tree corrects the earlier ones.",
    "Settings were fixed before looking at 2024. No oversampling. The base rate stays near 15%.",
], 0.7, 1.5, 11.8, 4.8, size=24)

# 7. Results
slide = new_slide("LightGBM sorts best, and the percentages stay honest")
slide.shapes.add_picture(str(FIG / "model_comparison.png"), Inches(0.4), Inches(1.35), Inches(7.4), Inches(4.6))
bullets(slide, [
    "AUC is a sorting score. 0.50 is a coin flip. Higher is better.",
    "LightGBM 0.692. XGBoost 0.683.",
    "GLM and the forest both 0.679. The forest's percentages are farther from the real rate.",
    "The app uses LightGBM.",
], 8.0, 1.6, 4.9, 4.8, size=18)

# 8. Calibration
slide = new_slide("A predicted 20% really happens about 20% of the time")
slide.shapes.add_picture(str(FIG / "calibration.png"), Inches(0.35), Inches(1.3), Inches(6.3), Inches(5.2))
slide.shapes.add_picture(str(FIG / "lift.png"), Inches(6.7), Inches(1.3), Inches(6.3), Inches(5.2))

# 9. Importance
slide = new_slide("What LightGBM uses")
slide.shapes.add_picture(str(FIG / "feature_importance.png"), Inches(0.3), Inches(1.2), Inches(7.6), Inches(5.6))
bullets(slide, [
    "About half the splits use time on cover. A short policy has less time to claim.",
    "The app asks about a full year, so that input is fixed at 1.",
    "Then coverage (15%), vehicle value (6%), and vehicle age (5%).",
    "SHAP, on 2,000 policies from 2024, tells the same story one policy at a time.",
], 8.1, 1.5, 4.8, 5.0, size=18)

# 10. Fairness
slide = new_slide("Fairness: gender is not in the file")
bullets(slide, [
    "Since 2012, EU insurers cannot use gender to set a price.",
    "Gender is not a feature in this model.",
    "None of the 47 columns is sex, gender, or a Spanish equivalent.",
    "A with-versus-without comparison cannot be run. A gap between women and men cannot be measured from this file.",
    "Past claims and payment schedule stay in, because they are known on day one. They were not built to stand in for gender. Home area and seats were left out.",
], 0.7, 1.45, 11.8, 5.2, size=22)

# 11. App
slide = new_slide("The app")
slide.shapes.add_picture(str(FIG / "app_screenshot.png"), Inches(0.4), Inches(1.3), Inches(7.2), Inches(4.6))
bullets(slide, [
    "Enter the day-one facts. The app returns a full-year chance and Low, Medium, or High.",
    "Those bands are the lower, middle, and upper third of 2024 scores. The lines sit near 14% and 20%.",
    "A typical driver scores about 15%, Medium.",
    "angekonan715-predictnextclaim-appapp-vz9rsh.streamlit.app",
], 7.8, 1.45, 5.1, 5.0, size=16)

# 12. Limits
slide = new_slide("What this does not claim")
bullets(slide, [
    "AUC 0.692 beats a coin flip. Many people who claim still sit among people who do not.",
    "One insurer, one country, three years.",
    "A higher rate next to quarterly payment is not proof that the payment plan causes claims.",
    "Broader cover pays for more kinds of damage, so part of that gap is the cover itself.",
    "The file has no driver under 18, and it has no gender field.",
], 0.7, 1.45, 11.8, 5.0, size=22)

prs.save(OUT)
print(OUT)

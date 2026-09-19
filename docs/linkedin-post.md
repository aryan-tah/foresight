# LinkedIn post

Every number below was measured on 18 September 2026 with this repo. Forecasts were made that day, before any of the events.

## The post

```
I gave Google's new AI 1,000 numbers. No dates. No labels. No context.

It pointed at 27 November. That's Black Friday.

Not last year's date, either. The two peaks in its data were 29 Nov 2024 and 28 Nov 2025. Copying would have been wrong. It found the right Friday from the shape of the numbers alone.

The numbers were daily Wikipedia pageviews. The model is TimesFM 3.0, released last month. It takes an array of floats and nothing else. Same test on "Halloween": it put the peak on 31 October.

I can't prove the mechanism. The output is consistent with combining a yearly period with day-of-week phase, and two prior cycles is a small sample.

I built Foresight around this. It pairs two models that fail in opposite ways:

→ LLMs can read a question but can't do numerical forecasting.
→ Time-series foundation models forecast well but can't read.

Claude handles the language side through tool use: it decides which public series measures the question, fetches it live, and explains the result. TimesFM 3.0 runs zero-shot on my laptop CPU (~0.1 s per forecast) and returns a median path plus quantile bands. Raw data never enters the LLM's context. Tools pass series IDs and summary statistics, and the arrays stay server-side.

The part I care most about: every forecast is backtested before it's shown. The last h days are held out and forecast blind, then scored two ways: MAE against a seasonal-naive baseline, and empirical coverage of the p10–p90 band against its nominal 80%. The agent has to ground its stated confidence in that result.

That makes it honest about failure:

• Halloween: 6% error vs 40% for the baseline.
• Bitcoin: the median is a flat line and the forecast is the widening band. For a near-random walk, that's the correct answer. Drawing a confident path would be fabrication.
• openai's PyPI downloads: the 80% band covered 20% of days, because downloads fell ~40% in late August. No univariate model sees a level shift coming, and the app says so.
• News-driven series (e.g. Nvidia pageviews) showed no edge over the baseline in my tests.

Also worth saying: the model called Halloween's date but its median peak (135K views) is well under last year's 269K. Timing is the strong claim. Magnitude isn't.

Forecast made 18 September. I'll post the actuals on 1 November, whichever way it goes.

Stack: Claude Opus 5 (tool use, parallel calls), TimesFM 3.0 on PyTorch/CPU, FastAPI + server-sent events, hand-written SVG charts, no frontend framework.

#MachineLearning #TimeSeries #AIAgents #Forecasting
```

**First comment** (links in the body get suppressed): the repo link. The repo is private; to link it, make it public first:

```bash
gh repo edit aryan-tah/foresight --visibility public --accept-visibility-change-consequences
```

`.env` is gitignored and was never committed, so no key is exposed.

## Why this hook

LinkedIn cuts a post at ~140 characters on mobile and ~210 on desktop, and most readers never tap "see more". Hooks that work have a specific detail, tension, and an open loop. The first line here is 70 characters; both lines sit above the desktop fold. It leads with Black Friday, not Halloween, because a date that moves every year is the surprising part.

### Alternatives

**The public bet**
```
I'm putting an AI's prediction on the record today.

On 1 November you can check whether it was wrong.
```

**The contrast**
```
I asked an AI where Bitcoin is headed. It drew a flat line and refused to pretend.

Then I asked it about Halloween.
```

**Anti-hype (engineers more than recruiters)**
```
Most AI demos hide their failures.

Mine prints them under every answer.
```

**The student angle**
```
LLMs can't forecast. Forecasting models can't read.

So I wired them together. Third-year CS student, one day of work.
```

## Attachments

1. `docs/halloween.png` first (LinkedIn crops to the first image)
2. `docs/black-friday.png`
3. A Bitcoin chart screenshot, for the contrast

A 30-second video usually gets more reach than images. Dark mode, browser ~1100px wide, no sound, captions on.

| Time | Shot |
|---|---|
| 0–3s | Home screen. Click the Halloween example. |
| 3–12s | Let the tool trace tick through in real time. Don't speed it up. |
| 12–18s | The chart appears. Hover the peak so the tooltip shows "31 Oct". |
| 18–22s | Scroll to the backtest strip, pause on "6.2% vs 39.5%". |
| 22–30s | Cut to the Bitcoin chart, hover along the flat line. Caption: "Same model. It knows what it can't predict." |

## Keeping it accurate in the comments

- **"1,000 numbers, no dates"** is literal. `forecaster.py` passes TimesFM a float array and nothing else. Dates exist only on our side, for labelling.
- **"Pointed at"** is the right verb. The median forecast peaked at the position that maps to 27 Nov. The model did not output the words "Black Friday".
- **"Zero-shot"** means no fine-tuning on this data. Don't claim it has never seen Wikipedia pageviews; nobody outside Google knows the pretraining corpus, and that data is plausibly in it. If someone raises it, concede it. The date shift is still the interesting part.
- **The 6% vs 40% backtest** covers the 60 days before 18 September, the quiet run-up. It does not show the model nailing a previous Halloween.
- **Diwali** is the honest counter-example: a lunar-calendar festival, the model was unsure and failed its backtest.
- **TimesFM 3.0 weights are non-commercial.** Don't describe this as a product.

## 1 November follow-up

Pull the actuals and post forecast vs. reality, whichever way it went:

| | Forecast (18 Sep) | Actual |
|---|---|---|
| Halloween peak date | 31 Oct | |
| Halloween peak views | 135K median (83K–217K) | |
| Black Friday peak date | 27 Nov | |
| Black Friday peak views | 40K median (20K–66K) | |

# Causal Research Reframe for `8011`

## 1. What We Actually Have

### Local assets that are genuinely usable

- Raw archive: [`archive (1)(1).zip`](/home/bingkzhao2/8011/DOC/archive%20(1)(1).zip)
  - Contains year-level `papers`, `reviews`, and `links` tables for ICLR 2018-2023.
  - Review files include fields such as `decision`, `rating` or `recommendation`, `confidence`, `keywords`, and year-specific review text sections like `review`, `summary_of_the_paper`, and `strength_and_weaknesses`.
- Existing analysis archives:
  - [`8011_rq1.zip`](/home/bingkzhao2/8011/DOC/8011_rq1.zip)
  - [`rq1+3.zip`](/home/bingkzhao2/8011/DOC/rq1+3.zip)
  - These contain derived CSVs, figures, and Python scripts, but they operationalize the project mainly as a lexicon-based association/prediction exercise.
- Existing project plan workbook:
  - [`Research Plan in Table Format_14887934(1).xlsx`](/home/bingkzhao2/8011/DOC/Research%20Plan%20in%20Table%20Format_14887934(1).xlsx)
- Reference paper:
  - [`The Power of “We” in Science Funding and Publication.pdf`](/home/bingkzhao2/8011/DOC/The%20Power%20of%20“We”%20in%20Science%20Funding%20and%20Publication.pdf)

### Environment note

- `PyPDF2` and `openpyxl` are now available locally, so the PDF and XLSX were directly inspected.
- `statsmodels` is still unavailable in the default environment, so current regression claims can be audited from scripts and tables but not honestly rerun end-to-end here without adding more dependencies.

## 2. Why the Current `rq1/rq3` Workflow Is Not Enough

The current workflow in [`rq1_iclr_analysis.py`](/home/bingkzhao2/8011/DOC/8011_rq1.zip) and [`rq3.py`](/home/bingkzhao2/8011/DOC/rq1+3.zip) is useful as a **first descriptive pass**, but it is not a strong computational social science design for causal inference.

### Main problems

1. **The focal variables are post-evaluation traces, not clearly exogenous treatments.**
   - Sentiment, politeness, toxicity, and constructiveness in reviews are produced by reviewers after they have already formed judgments about paper quality.
   - This creates severe confounding by latent paper quality, novelty, reviewer expertise, and reviewer-specific standards.

2. **The current analysis collapses too much of the review process into paper-level means.**
   - Averaging review language at the paper level discards within-paper reviewer disagreement.
   - It makes it harder to distinguish whether language reflects paper quality, reviewer scoring, or committee-level resolution.

3. **The current scripts are mostly lexicon-based and prediction-oriented.**
   - The workbook plan emphasizes `Logistic Regression`, `Random Forest`, `XGBoost`, `SHAP`, and cross-year validation.
   - That is a predictive workflow, not a causal identification strategy.

4. **The current figures are not aligned with the stronger inferential design used in the reference paper.**
   - The "We" paper does not stop at group means or coefficient plots.
   - It uses fixed-effects regressions with richer controls and then visualizes **marginal predicted probabilities** rather than raw boxplots as the main inferential figures.

5. **Current "path analysis" or mediation logic is too weak if based only on observational aggregates.**
   - If review language is not experimentally or quasi-experimentally induced, mediation/path analysis does not identify mechanism.
   - At best, it provides suggestive decomposition under very strong assumptions.

## 3. What the “We” Paper Actually Does That We Should Learn From

After inspecting the PDF, the relevant methodological lessons are very clear.

### 3.1 Observational core is rich-control fixed-effects modeling

The paper's observational backbone is:

- Binary outcome models for final success outcomes.
- Rich controls for content and quality-related confounders.
- Fixed effects for year and topic or research area.
- Additional author-level controls when the review setting allows them.

For the ICLR part specifically, the paper includes:

- year fixed effects,
- topic fixed effects,
- content controls such as length, references, readability, concreteness, novelty, and promotional language,
- then plots **marginal predictions of acceptance** as a function of the focal linguistic treatment.

### 3.2 Figures are marginal-effect figures, not just descriptive plots

The inferential figures in the reference paper are built around:

- predicted probability curves from fitted models,
- confidence intervals,
- comparisons across review stages or evaluator types,
- not simple raw boxplots as the main evidence.

### 3.3 Causal evidence is separated from observational association

This is the most important point.

The paper does **not** claim that fixed-effects regressions alone prove causality. Instead, it adds a separate causal design:

- an LLM-assisted intervention that rewrites the focal language while holding the underlying content as constant as possible,
- followed by evaluation outcomes and mediation analysis on experimentally generated data.

### 3.4 Mechanism analysis is only credible after treatment variation is created

The paper's mediation design is downstream of a text intervention. That is very different from doing path analysis on naturally occurring review sentiment.

## 4. Implication for Our Project

## Bottom line

If the project is truly about **causal inference**, then the current title and current RQ setup need to be reframed.

### What we can still say from the existing RQ1/RQ3 outputs

These outputs are still useful for:

- descriptive background,
- feature diagnostics,
- measurement sanity checks,
- identifying stable associations worth taking into a stronger design.

They should **not** be used as the paper's core causal evidence.

### What we should not claim from the current outputs

We should not claim:

- that review sentiment or politeness causally changes final acceptance,
- that lexicon-based language scores identify mechanism,
- that cross-year prediction gains tell us whether review comments "matter" causally.

## 5. Recommended Reframed Study Design

## Recommended positioning

The study should become a **two-layer design**:

1. **Observational association layer**  
   Purpose: establish whether review language has incremental association with acceptance after strong controls.

2. **Causal or quasi-causal layer**  
   Purpose: test whether textual framing itself changes evaluation outcomes, ideally through an intervention or a much tighter identification design.

This mirrors the logic of the "We" paper and is much more defensible.

## 6. Revised Research Questions

### Recommended version

| RQ | Research question | Claim type |
| --- | --- | --- |
| RQ1 | How do review-language features vary across accepted and rejected papers, review-score bands, and reviewer disagreement levels? | Descriptive |
| RQ2 | Conditional on score-based quality proxies, year, and topic, how strongly are review-language features associated with final acceptance? | Explanatory association |
| RQ3 | Among borderline papers with similar score profiles, do differences in review language correspond to meaningful changes in acceptance probability? | Quasi-causal / design-based association |
| RQ4 | When review wording is experimentally rewritten while substantive content is held constant, does the probability of a positive evaluation change? | Causal |

### Why this is better

- RQ1 keeps the descriptive story.
- RQ2 keeps the observational regression story.
- RQ3 tightens the comparison to papers where language could plausibly matter.
- RQ4 is where a real causal claim can live.

## 7. Recommended Unit of Analysis

We should stop treating the entire project as one paper-level average table.

### Use at least two levels

| Level | Best use | Why it matters |
| --- | --- | --- |
| Review level (`paper-review`) | Review recommendation, reviewer confidence, review sentiment/tone, stage-specific language | Preserves reviewer heterogeneity |
| Paper level | Final acceptance, mean score, disagreement, meta-review or committee outcome | Matches final decision process |

### Practical recommendation

- Use **paper-level models** for final acceptance.
- Use **review-level models** for reviewer recommendation or positivity.
- Link the two explicitly rather than collapsing everything too early.

## 8. Variable Dictionary for the Reframed Design

| Variable | Construct | Level | Operational definition | Source field or derivation | Scale | Expected direction | Risks or caveats |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `accept` | Final conference outcome | Paper | Binary final decision recoded to accept vs reject | `decision` / `recommendation` fields by year | Binary | Outcome | Label harmonization differs by year |
| `review_recommend` | Reviewer recommendation | Review | Numeric recommendation or binary weak+ accept threshold | `rating` or `recommendation` text parsing | Ordinal / binary | Positive with favorable language | Simultaneously produced with language |
| `mean_score` | Quality proxy | Paper | Average reviewer score | Parsed reviewer recommendations or ratings | Continuous | Strong positive | Not a pure pre-treatment control if causal pathway runs through scores |
| `score_std` | Reviewer disagreement | Paper | Standard deviation of reviewer scores | Within-paper dispersion | Continuous | Ambiguous | Small review counts make it noisy |
| `confidence_mean` | Reviewer certainty | Paper | Average review confidence | Parsed `confidence` fields | Continuous | Positive | Scale changes across years |
| `sentiment` | Review valence | Review or paper | Model- or lexicon-based positivity score | Review text | Continuous | Positive association | Strong overlap with recommendation |
| `politeness` | Interpersonal register | Review or paper | Politeness score from markers or classifier | Review text | Continuous | Possibly positive | May proxy professionalism, not persuasion |
| `constructiveness` | Actionable feedback | Review or paper | Reviewer guidance / suggestion intensity | Review text | Continuous | Ambiguous | Current lexicon definition is weak |
| `topic_fe` | Research area/topic | Paper | Topic categories or keyword-based clusters | `keywords` and possibly embeddings | Categorical | Control | Topic labels inconsistent across years |
| `year_fe` | Submission year | Paper | Fixed effect for review regime and cycle | Archive year | Categorical | Control | Captures policy changes and template changes |
| `review_length` | Review effort / elaboration | Review or paper | Word count or log word count | Review text | Continuous | Ambiguous | Longer reviews often accompany weaker papers |
| `paper_text_controls` | Manuscript content confounders | Paper | Readability, novelty, references, length, promotional language, technical density | Derived from abstract/full text | Multiple | Controls | Must be reconstructed from raw archive |

## 9. Recommended Methods by Research Question

| RQ | Outcome | Method | Why this method fits | Minimum robustness check | Interpretation limit |
| --- | --- | --- | --- | --- | --- |
| RQ1 | Acceptance, recommendation, score bands | Descriptive distributions and subgroup summaries | Maps language patterns without over-claiming | Year/topic splits | Purely descriptive |
| RQ2 | Final acceptance | Logistic regression with year and topic fixed effects, strong paper-level controls, and AME plots | Mirrors the reference paper's observational backbone | Compare models with and without score proxies; cluster by year if possible | Still observational |
| RQ3 | Final acceptance among borderline papers | Restricted-sample logit or matched design within narrow score bands; report AMEs | Makes the comparison more credible where language may matter at the margin | Vary score bandwidth; exact/near matching on year and topic | Still not full causality |
| RQ4 | LLM or human positive-evaluation outcome | Counterfactual text intervention experiment | Directly tests whether wording changes judgments | Multiple rewrite prompts, blinded rating, content-preservation checks | Depends on external evaluator validity |

## 10. What “Borderline Paper” Design Could Look Like

If we want a stronger quasi-causal observational design without leaving the archive, the most promising route is a **borderline paper design**.

### Core idea

Focus on papers where mean reviewer score is near the conference's effective decision threshold. For these papers:

- objective quality is more similar,
- numerical signals are more ambiguous,
- textual framing may plausibly have more room to matter in committee resolution.

### Practical implementation

1. Build year-specific score distributions.
2. Define borderline windows within each year.
   - Example: papers whose mean score falls within a narrow band around the accept/reject overlap region.
3. Match or subclass papers on:
   - year,
   - topic or keyword cluster,
   - score mean,
   - score dispersion,
   - confidence,
   - review count.
4. Estimate the AME of language variables within this restricted sample.

### Why this helps

It does not solve endogeneity, but it moves the design closer to the actual institutional margin where comments could matter.

## 11. What the True Causal Design Should Be

The cleanest causal strategy is **not** another observational regression. It is an intervention.

### Recommended causal experiment

Create paired versions of the same review text:

- original review,
- tone-softened version,
- more positive version,
- less polite version,
- constructiveness-enhanced version.

Keep the substantive arguments, claimed strengths, and claimed weaknesses constant.

Then use blinded evaluators or a validated LLM judge to rate:

- perceived paper quality,
- acceptance likelihood,
- reviewer helpfulness,
- credibility or fairness.

### Why this is defensible

- Treatment is directly manipulated.
- Content can be held approximately constant.
- Outcome is observed under controlled textual variation.
- Mechanism analysis becomes meaningful only after this step.

### Important caution

This experiment identifies the causal effect of **review wording on perceived evaluation**, not necessarily the real conference committee's final behavior. That limitation should be stated explicitly.

## 12. Figure Strategy Should Change Accordingly

If we follow the reference paper, the main inferential figures should no longer be dominated by raw boxplots.

### Main-text figure recommendations

1. **AME curve for final acceptance**
   - x-axis: focal language variable
   - y-axis: predicted acceptance probability
   - panels: sentiment, politeness, perhaps constructiveness
   - controls held at observed medians or means

2. **Borderline-sample AME figure**
   - same idea, but only for borderline papers
   - this is the stronger quasi-causal visual

3. **Stage-specific marginal effects**
   - reviewer recommendation,
   - meta-review or final decision,
   - possibly reviewer positivity as a downstream process variable

4. **Experimental treatment figure**
   - paired estimates for original vs rewritten text
   - report average treatment effect and confidence intervals

### What should move to appendix

- raw boxplots,
- year-by-year descriptive tables,
- lexicon inventory,
- robustness tables,
- full coefficient tables.

## 13. Concrete Next-Step Plan

### Step 1. Rebuild the dataset from the raw archive

We should construct a clean analysis dataset directly from [`archive (1)(1).zip`](/home/bingkzhao2/8011/DOC/archive%20(1)(1).zip), not from the already collapsed `rq1` CSV alone.

Minimum outputs:

- review-level table,
- paper-level table,
- year-harmonized score and confidence parser,
- keyword/topic controls,
- year-specific field harmonization notes.

### Step 2. Separate descriptive, observational, and causal layers

- Descriptive results stay in one section.
- Fixed-effects association models become the main observational section.
- Borderline or experimental design becomes the causal section.

### Step 3. Replace main figures with marginal-effect figures

- Compute predicted probabilities from fitted models.
- Use confidence intervals.
- Put raw descriptive plots in appendix.

### Step 4. Downgrade current RQ1/RQ3 outputs to supporting evidence

Current `rq1` and `rq3` assets should be labeled as:

- exploratory diagnostics,
- feature engineering attempts,
- preliminary association analyses.

They should not be used as the final inferential backbone.

## 14. Recommended Revised Working Title

If the paper keeps both observational and causal components, a better title would be:

**Do Review Comments Matter? Review Language, Marginal Acceptance Effects, and Causal Evidence from Counterfactual Rewriting**

If the causal experiment is not completed, then a safer title is:

**How Review Language Tracks Acceptance Decisions in Peer Review: Fixed-Effects Evidence and Marginal Effects**

## 15. Final Recommendation

The strongest version of this project is:

- **not** "lexicon scores predict acceptance,"
- **not** "path analysis shows mechanism,"
- but rather:

1. an observational fixed-effects study with strong controls and marginal-effect plots,
2. a tighter borderline-paper design for quasi-causal leverage,
3. and ideally a separate counterfactual text intervention for true causal evidence.

That structure is much closer to the methodological standard set by the "We" paper and much more defensible as a computational social science project centered on causal inference.

# Revised Research Plan Table

## Study framing

| Section | Project-specific entry |
| --- | --- |
| Study title | Do Review Comments Matter? Review Language, Marginal Acceptance Effects, and Causal Evidence from Counterfactual Rewriting |
| Research object | ICLR peer-review process on OpenReview, 2018-2023 |
| Unit of analysis | Review-level for reviewer recommendation processes; paper-level for final acceptance |
| Main outcome | Final accept vs reject |
| Secondary outcomes | Reviewer recommendation, reviewer positivity, reviewer engagement, meta-review outcome if recoverable |
| Main predictors | Review language features: sentiment, politeness, constructiveness, toxicity, review length, hedging, technical density |
| Required controls | Mean score, score dispersion, confidence, review count, year fixed effects, topic fixed effects, manuscript-level text controls |
| Data source | `/home/bingkzhao2/8011/DOC/archive (1)(1).zip` as the primary source; `rq1` and `rq1+3` as exploratory diagnostics only |
| Core inferential style | Fixed-effects regression plus average marginal effect plots |
| Strongest causal component | Counterfactual rewriting experiment on review text or manuscript abstract text |

## RQ alignment

| RQ | Hypothesis | Outcome | Predictors | Method | Robustness | Interpretation limit |
| --- | --- | --- | --- | --- | --- | --- |
| RQ1 | Accepted and rejected papers differ in review tone and structure | Acceptance and score bands | Language features | Descriptive subgroup analysis | Year/topic splits | Descriptive only |
| RQ2 | Language retains incremental explanatory power after strong controls | Final acceptance | Language features + quality proxies + FE | Logistic regression with year/topic FE and AMEs | Compare controls-only vs full model | Observational association |
| RQ3 | Language matters most for borderline papers with similar score profiles | Final acceptance in score-overlap sample | Language features in narrow score bands | Restricted-sample or matched design with AMEs | Vary bandwidth and matching rules | Quasi-causal, not definitive |
| RQ4 | Wording itself changes evaluation outcomes when content is held constant | LLM or human evaluation score | Rewritten text treatment | Counterfactual text intervention experiment | Multiple prompts and fidelity checks | Depends on evaluator validity |

## Variable dictionary

| Variable | Construct | Level | Operational definition | Source field or derivation | Scale | Caveat |
| --- | --- | --- | --- | --- | --- | --- |
| `accept` | Final outcome | Paper | Binary final decision | Parsed `decision`/`recommendation` | Binary | Labels differ across years |
| `review_recommend` | Reviewer stance | Review | Numeric or binary recommendation | `rating` or `recommendation` text parsing | Ordinal/binary | Produced jointly with language |
| `mean_score` | Quality proxy | Paper | Mean reviewer score | Parsed score fields | Continuous | Not a pure pre-treatment variable |
| `score_std` | Reviewer disagreement | Paper | SD of reviewer scores | Within-paper calculation | Continuous | Sensitive to small review counts |
| `confidence_mean` | Reviewer certainty | Paper | Mean reviewer confidence | Parsed `confidence` | Continuous | Scale wording changes by year |
| `topic_fe` | Topic control | Paper | Topic or keyword cluster | `keywords` plus clustering if needed | Categorical | Topic labels inconsistent |
| `sentiment` | Valence | Review/paper | Sentiment score from validated pipeline | Review text | Continuous | Strongly overlaps with recommendation |
| `politeness` | Interpersonal tone | Review/paper | Politeness measure | Review text | Continuous | May capture professionalism rather than persuasion |
| `constructiveness` | Actionability | Review/paper | Suggestion or guidance intensity | Review text | Continuous | Current marker design is weak |
| `review_length` | Elaboration | Review/paper | Word count or log word count | Review text | Continuous | Often rises for weaker papers |

## Threats to validity

| Threat | Why it matters | Mitigation |
| --- | --- | --- |
| Latent paper quality confounding | Better papers receive better reviews and better decisions | Strong controls, FE, borderline design |
| Post-treatment ambiguity | Review language and numeric scores are co-produced in evaluation | Separate descriptive and causal claims |
| Measurement drift across years | Review templates and field names change | Year harmonization and year FE |
| Topic heterogeneity | Different areas use different review norms | Topic FE or keyword clustering |
| Weak construct validity of lexicons | Marker-based features may not match real constructs | Human validation or stronger text models |
| Overclaiming mediation | Path analysis on observational data is not causal | Restrict mediation to experimental component |

## Output plan

| Deliverable | Main text or appendix | Role |
| --- | --- | --- |
| Marginal-effect figure for acceptance | Main text | Core observational result |
| Borderline-sample AME figure | Main text | Stronger quasi-causal evidence |
| Experimental treatment-effect figure | Main text | Causal evidence |
| Raw distribution plots | Appendix | Descriptive support |
| Full coefficient tables | Appendix | Reporting transparency |
| Lexicon and parser diagnostics | Appendix | Measurement audit |

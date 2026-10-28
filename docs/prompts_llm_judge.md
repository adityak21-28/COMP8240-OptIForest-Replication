# Optional LLM-as-a-judge protocol

This is an exploratory add-on, not a substitute for AUC-ROC or average precision.

## Proposed evaluator prompt

You are evaluating the *quality of an explanation* for a numerically ranked anomaly, not deciding whether the model is correct. Given a feature vector and a short explanation, rate the explanation from 1 to 5 on:

1. **Numerical faithfulness**: are claims supported by provided values and reference ranges?
2. **Specificity**: does it name the unusual features rather than use generic wording?
3. **Caution**: does it avoid claiming a cause or real-world risk that the data cannot establish?

Return JSON only with `faithfulness`, `specificity`, `caution`, and `short_reason`. Do not infer protected attributes, identity, health status, fraud intent, or other sensitive conclusions.

## Protocol requirements

- Use a small, predeclared sample (for example, 20 high-ranked cases and 20 randomly selected cases).
- Hide ground-truth labels and anomaly-type labels from the evaluator.
- Record the exact model name/version, provider, date, system/user prompt, temperature, raw output and any failures.
- Report the LLM scores separately from detection metrics. Do not use LLM-judge scores to claim detection correctness.
- If possible, compare two independent assessors or repeat with a second evaluator; discuss subjectivity and position bias.
- If no evaluator is available or the output is not suitable, report this as a limitation instead of manufacturing judge scores.

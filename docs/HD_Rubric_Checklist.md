# COMP8240 Final Report — High Distinction Checklist

Based on the Final Report assessment brief uploaded for COMP8240. This checklist is designed for the **default OptIForest project**.

## Requirements and applicable marks

| Criterion | Marks for default project | HD evidence to include |
|---|---:|---|
| Description of chosen paper / justification | 0 for default | Provide only enough paper context for the reader to understand the project; do not spend report space justifying an already-approved default project. |
| Description of original dataset | 0 for default-only item | Describe the original data inside the replication section as needed to explain the experiment. |
| Replication of original work | 8 | Clearly explain environment, exact command, data version, preprocessing, parameters, metric calculation, measured result, comparison with paper, and reasoned discussion of differences. Include logs and run metadata. |
| Construction of new data | 4 | Specify one existing dataset source and one newly constructed dataset, exact formats/properties, preprocessing, label mapping, and how construction is performed. Include data snippets/figures. |
| Results on new data | 6 | Report actual OptIForest results on new data, compare them with an appropriate baseline and original claims, and explain differences/failure cases. Explain any LLM-as-a-judge use or why it was omitted. |
| Ethical considerations | 4 | Discuss the Australian Code for the Responsible Conduct of Research, data provenance/licensing, privacy/re-identification, health-label interpretation, cyber dual-use considerations, and honest reporting. |
| Quality of code | 3 | Public, readable, organised GitHub repository; clear README; requirements; one-command data pipeline; tests; logs; exact instructions the marker can follow. |
| Quality of new data | 3 | Data has the intended schema, useful labels, no accidental leakage, validation checks, reproducible generation, metadata and documented limitations. |
| Report presentation | 4 | Clear argument, strong structure, professional writing, official ACL Overleaf template, 8–10 typical pages and no more than 10 main-body pages excluding references. |

Applicable default-project marks sum to 32/40. The brief identifies default projects as having an 80% maximum for the project assessment tasks; this is a ceiling, not an automatic mark.

## Hard gates before submission

- [ ] Official OptIForest result on the original AD data is supported by actual terminal logs.
- [ ] At least one selected existing new dataset is run with official OptIForest and its dataset version/label mapping is documented.
- [ ] Synthetic v1 is run through official OptIForest; per-regime results are reported where possible.
- [ ] `[RUN]` is removed from every report table, and every result is supported by a committed CSV/log.
- [ ] scikit-learn Isolation Forest results are labelled as a reference baseline, never as OptIForest.
- [ ] The GitHub repository is public and the URL is included in the report.
- [ ] All scripts run from the README instructions and tests pass in a fresh environment.
- [ ] The final report is compiled from the official ACL Overleaf template.
- [ ] Every citation has been checked for relevance and correctness; reference list matches citations.
- [ ] The ethics section is informed by the unit resources and the Australian Code.
- [ ] The report's claims distinguish work actually completed from planned work.

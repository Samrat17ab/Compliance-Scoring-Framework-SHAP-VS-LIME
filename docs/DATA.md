# Dataset and saved evidence

Source: [Home Credit Default Risk — official competition data](https://www.kaggle.com/competitions/home-credit-default-risk/data).

The full `application_train.csv` is intentionally excluded. Download it under the source's terms and place it at the repository root to run training or regenerate explanations.

| Property | Expected value |
|---|---|
| Rows | 307,511 |
| Columns | 122 |
| Identifier | `SK_ID_CURR` |
| Target | `TARGET` (payment-difficulty label) |
| File SHA-256 | `5b853f37fb6814cd4f844aee8eb961b17baadde5899ed3b1bcab9e00d8a10347` |

File-level hashes can differ if line endings or CSV serialization change. The evidence scripts expect the exact recorded file.

## Included artifacts

The repository includes transformed numerical inputs for the 1,000-row evidence sample, anonymized loan identifiers, categorical encodings, training medians, model files, explanation coefficients, and selected applicant visualizations. These are research evidence, not a new unrestricted dataset. Original dataset terms continue to apply to source-derived materials.

The application table does not supply an explicit country or nationality field. No country labels have been inferred or added. Monetary units are not asserted to be a particular national currency.

“Accepted” and “rejected” in plots are classifications at the chosen model threshold. Missing-value replacements are model inputs, not observed applicant facts; case visualizations mark them where documented.

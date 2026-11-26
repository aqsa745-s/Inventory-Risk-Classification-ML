# Sources and data acknowledgement

## Dataset

The sample transaction data bundled in `data/superstore_sample.csv` is the Tableau Sample - Superstore dataset, distributed for demonstration and learning purposes. The dataset is not owned by this project and is not evidence of a real retailer's current inventory position.

## Implementation

The Python analysis pipeline, risk-scoring rules, dashboard layout, report generation, and tests in this project are provided as a separate implementation for this academic task. The project documents its methods and limitations so that results can be reviewed and reproduced.

## Method references

- ADI and CV² are commonly used measures for describing intermittent demand patterns. The cut-offs used here (ADI 1.32 and CV² 0.49) are conventional reference thresholds and should be explained as such in a presentation.
- K-Means is an unsupervised clustering algorithm. Cluster numbers are identifiers only; they do not inherently mean low, medium, or high risk.
- ABC analysis ranks items by contribution to sales value. XYZ analysis groups items by demand variability. The precise thresholds are project assumptions and can be adjusted.
- Safety Stock and Reorder Point calculations are simplified educational estimates. Real deployment requires accurate lead-time, inventory-position, service-level, and stockout data.

## Academic integrity note

If this project is submitted for assessment, follow the institution's rules on acknowledging datasets, external ideas, and any assistance received. Be prepared to explain the calculations and code in your own words.

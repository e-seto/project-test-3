# From Detection to Prevention: Data Science Approaches to Credit Card Fraud

**Authors:** Tyrone Bougiridis, Reuben John, Gurkeerat Kakar, Krystal Kim, Mohammed Ishfaq Mostain, Elizabeth Seto

---

## I. Introduction

Credit card fraud prevention is a critical challenge for financial institutions, requiring a careful balance between preventing financial loss and avoiding disruptions to legitimate customer purchases. This project applies the data science workflow, with descriptive analysis, predictive modelling, and prescriptive decision policies to identify fraud patterns, develop cost-effective detection models, and translate predictions into actionable outcomes.

---

## II. Business Question

Credit card fraud is a challenge in the financial services industry, resulting in financial loss and customer dissatisfaction. Flagging legitimate transactions as fraudulent leads to declined payments and customer frustration, while missing fraudulent transactions results in direct financial loss. As transaction volumes increase and fraud tactics evolve, there is a need for effective and cost-efficient fraud detection systems that balance fraud prevention with a positive customer experience.

### A. Stakeholders

Understanding stakeholders ensures the fraud detection system aligns with business goals. Stakeholders are individuals or groups who influence or are affected by the fraud detection system.

**Internal stakeholders** are responsible for system development and operation:
- **Risk Management Team** — defines fraud rules and thresholds
- **Data Science Team** — develops and maintains predictive models while meeting privacy standards
- **Fraud Operations Team** — investigates flagged transactions and enforces policies
- **Customer Service Team** — assists customers affected by false positives

**External stakeholders** are affected by system outcomes:
- **Customers** — face financial risk or inconvenience
- **Financial institutions** — manage transactions and chargebacks
- **Payment networks** — provide transaction infrastructure
- **Regulatory bodies** — oversee fraud prevention and data protection

### B. Motivation

Credit cards are increasingly used for in-person purchases, online transactions, and bill payments, making their reliability essential. In 2024, they accounted for one in three transactions in Canada [1]. Additionally, fraud results in significant losses, with every dollar lost to fraud costing North American financial institutions approximately $4.41 [2].

Fraud prevention requires balancing financial loss, customer inconvenience, and operational effort. Missed fraud causes direct financial loss, while false positives result in declined legitimate transactions and avoidable operational costs [3]. Given the high volume and real-time nature of transactions, manual review alone is infeasible, requiring automated systems that can balance these costs.

### C. Measurable Impact

Impact is measured by the system's ability to reduce the overall cost of credit card fraud while minimizing false positives, demonstrated by net savings:

$$\text{Net Savings} = \$F - \$(FP + IC)$$

Where $F$ is the financial losses prevented, $FP$ is the cost of false positives, and $IC$ is the system implementation and maintenance costs.

### D. Broader Domain Goal

This project aims to improve fraud decision-making by balancing competing costs using descriptive analysis, predictive modelling, and prescriptive policies. While focused on credit cards, the methodology applies to other financial domains such as e-commerce. The broader goal is to reduce fraud, minimize customer disruption, and improve operational efficiency.

---

## III. Translating to Data Science Question

The business problem of credit card fraud detection can be translated into three connected data science questions:

- **Descriptive:** *What patterns and characteristics distinguish fraudulent transactions from legitimate transactions, including customer demographics, transaction context, and temporal trends?*
- **Predictive:** *How effectively can ensemble machine learning models detect fraudulent credit card transactions in highly imbalanced datasets?*
- **Prescriptive:** *How can model predictions inform policies that reduce fraud losses while minimizing disruption to legitimate customers and operational costs?*

### A. Defining Actionable Objectives

**Descriptive analysis objective:** Identify patterns in fraudulent transactions by examining customer demographics, transaction context, and temporal trends to provide context for feature engineering and modelling.

**Predictive analysis objective:** Build supervised machine learning models that classify transactions as fraudulent or legitimate using historical data. Ensemble methods such as Random Forest, XGBoost, and AdaBoost will be compared. Class imbalance will be addressed using Synthetic Minority Oversampling Technique (SMOTE) and threshold tuning. Feature engineering will focus on transaction amount, time, location, and merchant characteristics. The best-performing model will be chosen to inform policies.

**Prescriptive analysis objective:** Design actionable fraud prevention policies based on model outputs. These policies will be applied to the dataset, and the predictive model will be rerun to evaluate their effects. This allows the project to assess not only how well the model predicts fraud, but how its predictions can be used to guide fraud decisions.

### B. Defining Success Criteria

Success is measured by predictive performance and the impact on decision making. Metrics such as recall, F1-score, precision, and Area Under the Curve (AUC) will be used to compare models, with **recall prioritized** to avoid undetected fraud and **F1-score balancing false positives**. Because fraud detection involves asymmetric costs, cost curves will be used to evaluate the trade-offs between fraud loss and false positives.

> The project is successful if a predictive model combined with a prescriptive policy improves cost trade-offs compared to the same model without the policy.

---

## IV. Team Formation

| Role | Member |
|---|---|
| Project Lead | Krystal Kim |
| Data Lead | Gurkeerat Kakar |
| Model Lead | Reuben John |
| Model Lead | Mohammed Ishfaq Mostain |
| Evaluation Lead | Tyrone Bougiridis |
| Communication Lead | Elizabeth Seto |

- **Project Lead** — manages timelines and milestones
- **Data Lead** — handles data preprocessing and data splits
- **Model Leads** — responsible for model development and tuning
- **Evaluation Lead** — defines metrics and performs error analysis
- **Communication Lead** — responsible for visualizations, final report and presentation

---

## V. Literature Review

Credit card fraud detection has been widely studied as a binary classification task with models commonly evaluated using metrics like accuracy, precision, recall, and F1-score under severe class imbalance [4]. Although useful for comparing models, these metrics do not fully reflect real-world cost trade-offs.

### A. Methods Used

Existing studies primarily rely on machine learning and deep learning approaches. Models such as gradient boost, random forest, and XGBoost are frequently combined with resampling techniques like SMOTE [5], [6]. Other work uses graph-based representations to identify coordinated fraud behaviour [7]. More recent research applies deep learning architectures such as Kolmogorov–Arnold Networks with dynamic oversampling and ensemble feature selection [7]. Privacy-aware approaches, such as federated learning, have also been explored [8].

### B. How They Measure Success

Most studies primarily measured success using accuracy, precision, recall, F1-score [5]–[8], and occasionally AUC [4]. However, optimizing these metrics alone does not account for the unequal costs associated with false positives and false negatives. Operational costs such as manual review of transactions and customer inconvenience are not reflected in the results of these detection systems.

### C. Gap Analysis

Two limitations emerge from the literature review:
1. Most studies rely on standard metrics for model optimization and evaluation.
2. Few studies approach fraud detection as a prescriptive problem where model outputs inform policies.

This project addresses those gaps by integrating predictive modelling with cost-based evaluation using cost curves [9] to shift the focus to understanding how different policies can affect financial loss, customer experience, and operational costs.

---

## VI. Dataset

To support the descriptive, predictive, and prescriptive objectives, a synthetic credit card transaction dataset is used.

### A. Source

The dataset is generated using the **Sparkov Data Generation** simulation, which simulates realistic fraud scenarios while circumventing the privacy constraints of real-world banking data. It exhibits a severe class imbalance, mimicking the "needle in a haystack" problem in fraud detection.

The data spans **24 months** from January 1, 2019 to December 31, 2020. This simulation was selected for two primary reasons:
1. Unlike benchmark datasets that obscure features using PCA, Sparkov provides raw, interpretable attributes that support prescriptive analysis.
2. The two-year span allows the model to learn long-term behavioural baselines, seasonal trends, and weekend vs. weekday patterns.

### B. Features

The raw dataset consists of **22 attributes** per transaction, including the target label (`is_fraud`). Additional features are engineered:

| Engineered Feature | Description |
|---|---|
| **Haversine Distance** | Geodesic distance between customer's home and transaction location (from `lat`, `long`, `merch_lat`, `merch_long`) |
| **Hour of Day** | Extracted from transaction timestamp to identify high-risk time periods |
| **Day of Week** | Extracted from transaction timestamp to capture weekly behavioural patterns |
| **Amount Deviation** | Compares transaction amount against customer's historical average to flag spending spikes |

Demographic attributes such as date of birth and job title are used to analyze susceptibility across customer groups. High-cardinality personally identifiable information (SSNs, credit card numbers) is removed to prevent memorization and improve generalization.

---

## VII. Conclusion

This proposal outlines our approach to addressing credit card fraud prevention through descriptive analysis, cost-efficient fraud classification, and prescriptive decision policies. Using a synthetic dataset, we will analyze fraud patterns, train models accounting for large class imbalances, and use cost curves to quantify the trade-off between fraud loss and false positives.

---

## References

[1] S. Yun, C. Ackerman, G. Mockton, and M. Ghogale, "Canadian payment methods and trends report 2025," 2025.

[2] L. R. Solutions, "Every dollar lost to a fraudster costs north america's financial institutions $4.41 according to lexisnexis true cost of fraud study," Apr. 2024.

[3] J. Morgan, "When false positives spiked, company abandoned fraud prevention tools," Mar. 2026.

[4] M. Akouhar, M. Ouhssini, M. El Fatini, A. Abarda, and E. Agherrabi, "Dynamic oversampling-driven kolmogorov–arnold networks for credit card fraud detection: An ensemble approach to robust financial security," *Egyptian Informatics Journal*, vol. 31, p. 100712, 2025.

[5] S. Jose, D. Devassy, and A. M. Antony, "Detection of credit card fraud using resampling and boosting technique," in *2023 Advanced Computing and Communication Technologies for High Performance Applications (ACCTHPA)*, 2023, pp. 1–8.

[6] J. Jemai, A. Zarrad, and A. Daud, "Identifying fraudulent credit card transactions using ensemble learning," *IEEE Access*, vol. 12, pp. 54 893–54 900, 2024.

[7] N. Mauliddiah and Suharjito, "Implementation graph database framework for credit card fraud detection," *Procedia Computer Science*, vol. 227, pp. 326–335, 2023.

[8] M. S. Farooq, S. F. Munir, M. F. Manzoor, and M. Shaheen, "Ai-driven adaptive federated learning with privacy preservation and imbalance adjustment for financial credit card fraud detection," *Applied Computational Intelligence and Soft Computing*, vol. 2025, no. 1, p. 7116768, 2025.

[9] C. Drummond and R. C. Holte, "Cost curves: An improved method for visualizing classifier performance," *Machine Learning*, vol. 65, no. 1, pp. 95–130, oct 2006.

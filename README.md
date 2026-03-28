# Credit Card Transactions Fraud Detection Project

## Project Overview

This project focuses on developing a machine learning-based fraud detection system for credit card transactions. It applies the data science workflow, with descriptive analysis, predictive modelling, and prescriptive decision policies to identify fraud patterns, develop cost-effective detection models, and translate predictions into actionable outcomes. The goal is to balance fraud prevention with minimizing customer inconvenience and operational costs.

### Key Objectives

- **Descriptive Analysis**: Identify patterns and features that distinguish fraudulent transactions from legitimate ones. This includes analyzing customer demographics and transaction context.
- **Predictive Modeling**: Develop and evaluate machine learning models (Random Forest, XGBoost, and AdaBoost) for detecting fraud based on historical transaction data.
- **Prescriptive Decision Policies**: Design actionable policies derived from model predictions to minimize fraud losses while reducing the disruption caused to legitimate customers.

### Methodology

- **Data Preprocessing**: Address class imbalance using techniques like [TODO: WHAT DID WE DO??] and engineer features such as [TODO: LIST SOME].
- **Modelling**: Compare various ensemble machine learning algorithms (Random Forest, XGBoost, AdaBoost) to identify the best-performing model for fraud detection {TODO: MAYBE ADD A BIT MORE DETAIL??]
- **Cost Evaluation**: Use F1-Score and recall to assess models and a cost formula to evaluate trade-offs between fraud detection accuracy and false positives to ensure the system balances risk and customer experience effectively. [TODO: IS THIS HOW WE EVALUATED]

### Team

| Name        | Email               | Role
| ----------- | ------------------- | -----
| **Tyrone Bougiridis** | tbougiri@ualberta.ca | Evaluation Lead |
| **Reuben John** | rjjohn1@ualberta.ca | Model Lead |
| **Gurkeerat Kakar** | gskakar@ualberta.ca | Data Lead |
| **Krystal Kim** | jueun2@ualberta.ca | Project Lead |
| **Mohammed Ishfaq Mostain** | mostain@ualberta.ca  | Model Lead |
| **Elizabeth Seto** | lseto1@ualberta.ca | Communication Lead |

### Project Files

Trello:
https://trello.com/b/DfKrhMQI/intd-491-credit-card-fraudulent

Data Source:
https://github.com/namebrandon/Sparkov_Data_Generation 

Proposal Presentation Slides: https://docs.google.com/presentation/d/1PLg_hWNRepACWU3XMDVJvzBVoMKmpuuc1Kk0PdXcjSE/edit?usp=sharing

Progress Presentation Slides: https://docs.google.com/presentation/d/1OrnWPGJbq7ebnUc-Hab9KRqM4rdZLml--gu7iJQ-KkI/edit?usp=sharing

---

## Data

The data consists of credit card transactions with the following features:

- `cc_num`: Credit card number (identifier).
- `gender`: Gender of the cardholder.
- `city`: City where the transaction occurred.
- `state`: State where the transaction occurred.
- `zip`: ZIP code of the transaction location.
- `lat`: Latitude of the transaction location.
- `long`: Longitude of the transaction location.
- `city_pop`: Population of the city.
- `job`: Job of the cardholder.
- `category`: Merchant business category.
- `amt`: Transaction amount in dollars.
- `merchant`: Merchant business name.
- `merch_lat`: Latitude of the merchant's location.
- `merch_long`: Longitude of the merchant's location.
- `trans_datetime`: Transaction date and time.
- `age`: Age of the cardholder.
- `is_fraud`: Label indicating whether the transaction is fraudulent (1) or not (0).

The dataset is generated using the [Sparkov Data Generation](https://github.com/namebrandon/Sparkov_Data_Generation) simulation tool, which simulates realistic fraud scenarios while respecting privacy constraints.

Additionally the following features were engineered:

- `haversine_km`: Distance from the merchant and the cardholder locations.
- `cust_amt_mean`: Amount the customer spends on average.
- `amt_dev_from_mean`: Deviation of trenasction fom what the cutomer spends on average.

---

## References
-

---

## License

See the LICENSE file for the project’s licensing details.

#Credit Risk Modeling Project

Credit Scoring Business Understanding

# How Basel II Accord Influences Model Interpretability

The Basel II Capital Accord fundamentally changed financial risk management by introducing three pillars that directly 

impact our modeling approach:

Pillar 1 - Minimum Capital Requirements: Banks must hold capital proportional to risk exposure. This requires 

transparent models that can calculate Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default 

(EAD) with clear audit trails. Regulators must validate these calculations.

Pillar 2 - Supervisory Review: Regulators expect robust internal validation processes. Interpretable models enable:

Effective stress testing and scenario analysis

Clear identification of model limitations

Meaningful dialogue with regulatory bodies

Comprehensive model documentation

# Pillar 3 - Market Discipline: Public disclosure requirements demand that risk metrics are understandable to investors 

and stakeholders. Transparent models facilitate clear communication of risk assessment methodologies.

Result: We need interpretable, well-documented models (like Logistic Regression with Weight of Evidence) that provide:

Clear audit trails

Explainable predictions

Regulatory compliance documentation

Stakeholder confidence in risk assessments

# Why Proxy Variables Are Necessary and Their Business Risks

Why Proxy Variables Are Required:

Our Xente transaction dataset lacks direct "default" labels because:

Transaction data doesn't explicitly track loan defaults

Historical default data may be unavailable or incomplete

We must infer credit risk from behavioral patterns

Common Proxy Variables:

Payment Delinquency: Transactions marked as problematic or requiring intervention

High-Risk Behavior Patterns: Frequent high-value transactions from new accounts

Customer Churn with Outstanding Balances: Customers who stop transacting with pending payments

Fraudulent Transactions: Using FraudResult (1 = fraud) as a risk indicator

Business Risks of Proxy-Based Predictions:

# Risk Type	Impact	Mitigation Strategy

Misclassification Risk	False positives: Rejecting good customers → Lost revenue

False negatives: Approving risky customers → Potential defaults	Regular model validation, threshold optimization

Model Drift Risk	Behavioral patterns change over time, making proxies less representative	Continuous monitoring, 

# periodic retraining

Regulatory Risk	Regulators may question proxy validity → Compliance issues	Comprehensive documentation, regulatory 

alignment

Business Strategy Risk	Over-reliance on proxies may miss emerging risk patterns	Multi-proxy approach, expert validation

Trade-offs: Simple vs. Complex Models in Regulated Finance

Logistic Regression with Weight of Evidence (WoE):

Advantages:

✅ High Interpretability: Clear relationship between features and predictions

✅ Regulatory Compliance: Easier to explain and validate

✅ Stability: Less prone to overfitting with proper feature engineering

✅ Feature Importance: WoE transformation provides intuitive risk indicators

✅ Basel II Friendly: Meets regulatory transparency requirements

Disadvantages:

❌ Linear Assumptions: May not capture complex non-linear relationships

❌ Feature Engineering Intensive: Requires significant domain expertise

❌ Lower Predictive Power: May underperform on complex, high-dimensional data

Gradient Boosting (XGBoost, LightGBM):

Advantages:

✅ High Predictive Accuracy: Often achieves superior performance

✅ Handles Non-linearity: Captures complex feature interactions

✅ Robust to Outliers: More resilient to data anomalies

✅ Built-in Regularization: Reduces overfitting with proper tuning

Disadvantages:

❌ Black Box Nature: Difficult to explain individual predictions

❌ Regulatory Challenges: May not meet "right to explanation" requirements

❌ Overfitting Risk: Without proper regularization

❌ Computational Complexity: Longer training times, more resources

❌ Basel II Concerns: Higher scrutiny and validation requirements

Recommended Approach for Our Context:

Given the regulated financial environment, we recommend:

Primary Model: Logistic Regression with WoE for:

Regulatory compliance and approval

Baseline interpretability

Clear documentation for audits

Secondary Model: Gradient Boosting for:

Performance benchmarking

Identifying complex patterns missed by linear models

Champion-challenger framework

Interpretability Enhancements:

SHAP values for complex model explanations

Partial dependence plots to visualize feature effects

LIME for local interpretability

# Validation Framework:

Regular back-testing and validation

Stress testing under adverse scenarios

Comprehensive model documentation

Decision Rule: Prioritize interpretability over marginal accuracy gains in production, as regulatory compliance and 

stakeholder trust are paramount in financial services.

Key Takeaway: In regulated financial contexts, a well-documented, interpretable model with slightly lower accuracy is 

preferable to a high-performing "black box" that regulators cannot validate. The cost of model opacity often outweighs 

the benefits of marginal predictive improvements.
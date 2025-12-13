# Credit Risk Modeling Project

## Credit Scoring Business Understanding

### Basel II Accord Influence on Model Interpretability

The Basel II Capital Accord fundamentally transformed risk management in banking by emphasizing three pillars: minimum capital requirements, supervisory review, and market discipline. This directly impacts our modeling approach:

1. **Regulatory Compliance**: Basel II requires banks to quantify credit risk accurately and hold capital accordingly. An interpretable model is essential for regulatory approval and audit trails.

2. **Model Risk Management**: Regulators demand transparency in model assumptions, limitations, and performance characteristics. Black-box models face higher scrutiny and may require additional validation.

3. **Economic Capital Calculation**: The Internal Ratings-Based (IRB) approach under Basel II requires banks to estimate Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD). These estimates must be justifiable and explainable to stakeholders.

4. **Stress Testing Requirements**: Models must withstand adverse economic scenarios. Interpretable models allow better understanding of how variables interact under stress conditions.

### Proxy Variable Necessity and Business Risks

**Why Proxy Variables Are Needed:**
- Direct default labels are often unavailable due to short observation periods or limited historical data
- Default events are rare, making direct modeling statistically challenging
- Regulatory definitions of default may not align with available data
- Multiple delinquency statuses can serve as effective proxies for default risk

**Potential Business Risks:**
- **Misclassification Risk**: Proxy variables may not perfectly correlate with actual default, leading to Type I/II errors
- **Temporal Misalignment**: Delinquency patterns may change over time, causing model drift
- **Regulatory Non-compliance**: Using non-standard definitions may violate regulatory requirements
- **Capital Misallocation**: Inaccurate risk predictions can lead to insufficient capital buffers
- **Reputation Risk**: Incorrect credit decisions based on proxies can damage customer relationships

### Model Selection Trade-offs in Regulated Finance

**Simple, Interpretable Models (Logistic Regression with WoE):**
- **Advantages**:
  - Full transparency in feature contributions (coefficients)
  - Easy to explain to regulators and business stakeholders
  - Well-established validation frameworks
  - Reduced model risk and easier maintenance
- **Disadvantages**:
  - May sacrifice predictive power for interpretability
  - Assumes linear relationships (unless WoE transformed)
  - Limited capacity to capture complex interactions

**Complex, High-Performance Models (Gradient Boosting/XGBoost):**
- **Advantages**:
  - Superior predictive performance on non-linear patterns
  - Built-in feature importance metrics
  - Robust to outliers and missing values
  - Can capture complex feature interactions
- **Disadvantages**:
  - "Black box" nature raises regulatory concerns
  - Requires extensive documentation and validation
  - Higher computational costs
  - More challenging to explain individual predictions
  - Potential for overfitting without proper regularization

**Recommended Hybrid Approach**:
Given regulatory constraints, we implement a pragmatic approach:
1. Start with interpretable models for regulatory approval
2. Use complex models for champion-challenger framework
3. Implement SHAP/LIME for model explanations where needed
4. Maintain comprehensive documentation for all modeling decisions
# **A Decision-focused Framework for Integrated Inventory Management under Heterogeneous Demand**

## **1\. Introduction**

### **1.1 Background and Motivation**

In modern retail environments, firms manage thousands of stock-keeping units (SKUs) under complex demand uncertainty while simultaneously balancing service levels and operational costs. Inventory control remains a central operational challenge, involving the determination of replenishment quantities under stochastic demand.

Classical inventory optimization methods such as dynamic programming and stochastic programming become computationally intractable at scale, particularly when incorporating large SKU sets and multiple global constraints such as budget and warehouse capacity. In addition, these methods often rely on strong distributional assumptions (e.g., Normal demand), which are frequently violated in real-world retail data.

### **1.2 Predict-then-Optimize vs Predict-and-Optimize**

Recent advances in machine learning have led to the widespread adoption of a **Predict-then-Optimize (PtO)** framework, where demand is first forecasted using statistical or ML models and then passed into an optimization model (e.g., Newsvendor or heuristic solvers).

However, this decoupled pipeline suffers from a structural limitation: predictive accuracy (e.g., MSE minimization) does not necessarily translate into improved decision quality. Small errors in high-cost regions (e.g., stockout periods) may lead to disproportionately large economic losses.

To address this mismatch, this study adopts a **Predict-and-Optimize (PaO)** paradigm, where predictive models are trained with respect to downstream decision objectives, directly minimizing total operational cost rather than statistical loss.

The detailed mathematical formalization of this PaO model, along with the gradient linkage via a surrogate model, is formulated in **Sections 4.7 and 4.8**.

## **2\. Problem Definition**

We consider a multi-item inventory control problem under stochastic and heterogeneous demand. The system consists of thousands of SKUs with distinct demand dynamics and cost structures.

### **2.1 Key Characteristics**

- **Asymmetric cost structure**: holding, ordering, and shortage costs differ significantly.
- **Demand heterogeneity**: includes smooth, intermittent, and lumpy demand patterns.
- **Global constraints**: inventory decisions must satisfy budget and warehouse capacity limits.

### **2.2 Objective**

The objective is to minimize total operational cost under uncertainty while ensuring feasible replenishment decisions across all SKUs.

## **3\. Methodology**

### **3.1 Integrated Decision System**

We propose a closed-loop decision framework consisting of:

1. **A predictive model for demand estimation**: Neural networks for continuous/erratic demand and statistical heuristics for lumpy/intermittent demand.
2. **A heuristic optimization module for inventory decisions**: An Artificial Bee Colony Algorithm (ABCA) solver that handles complex global constraints (budget and warehouse capacity).
3. **A surrogate cost model**: A LightGBM-based model to approximate the non-differentiable mapping between demand predictions and final business costs.

### **3.2 Demand Segmentation**

To address heterogeneity, SKUs are clustered using **Average Demand Interval (ADI)** and **Coefficient of Variation (CV²)** into four categories:

- Smooth
- Erratic
- Intermittent
- Lumpy

K-means clustering is applied to separate SKUs, allowing differentiated forecasting strategies across segments.

### **3.3 Surrogate Cost Modeling**

Because the mapping from predicted demand to replenishment decisions via the ABCA solver is non-differentiable, standard gradient descent cannot be used to optimize the predictive model based on final costs. To bridge this gap, we introduce a **LightGBM-based surrogate model**.

This surrogate learns the cost landscape by mapping the demand forecasts and state features to the resulting inventory cost evaluated in the business environment. Once trained, the surrogate provides a differentiable approximation of the decision environment. By calculating the partial derivative of the approximated total cost with respect to the demand predictions, it generates pseudo-gradients that enable feedback propagation.

### **3.4 End-to-End Learning (Predict-and-Optimize)**

The core of our methodology is the transition from a traditional Predict-then-Optimize (PtO) pipeline—which merely minimizes statistical error (e.g., MSE)—to a **Predict-and-Optimize (PaO)** framework that directly minimizes operational costs.

The end-to-end learning loop operates dynamically during training:

- **Forward Pass**: The neural network generates demand predictions.
- **Optimization**: The ABCA solver takes these predictions and computes optimal order quantities under global constraints.
- **Environment Evaluation**: The generated decisions are evaluated against the ground-truth demand to calculate the true operational cost.
- **Backward Pass**: The LightGBM surrogate model estimates the cost gradient and backpropagates these pseudo-gradients to update the neural network's weights.

This architecture ensures that the forecasting model becomes "decision-aware," heavily penalizing prediction errors in cost-sensitive regions (such as stockouts) rather than treating all statistical deviations equally.

## **4\. Mathematical Formulation**

### **4.1 Notation**

Let:

- i: SKU index
- t: time period
- Qit: order quantity
- Dit: realized demand
- D^it: predicted demand
- Iit​: inventory level

### **4.2 Inventory Dynamics**

I_it=I\_(it−1)+Qit−Dit

### **4.3 Objective Function**

The total operational cost is defined as:

min⁡TC=∑i,t(c_hI_it++c_uI_it^−+c_fix⋅1(Qit\>0))

where:

- I_it^+​: holding inventory
- I_it^{-}​: shortage quantity
- 1(⋅): ordering indicator

### **4.4 Constraints**

**Storage capacity**

∑ivi(I\_(it−1)+Qit)≤Vmax​

**Budget constraint**

∑ipiQit≤Btotal

**Flow conservation**

Iit=Iit−1+Qit−Dit

### **4.5 Cost Parameterization**

Given the absence of explicit cost data in the M5 dataset, we construct economically grounded proxies:

Underage cost:

cu=0.35⋅price

Holding cost:

ch=0.20⋅price/52

Fixed ordering cost:

cfix=5

These parameters reflect approximate retail margin, annual holding rate, and ordering overhead respectively.

### **4.6 ABCA Optimization Problem**

In the Predict-and-Optimize framework, the heuristic solver (ABCA) seeks to find the optimal order quantity vector **Q\*** at each time step $t$, based on the predicted demand vector **D^**. The optimization problem solved by ABCA is formulated as:

**Q\*** = $\arg\min_{\mathbf{Q}} \sum_{i} \left( c_h I_{it}^+ + c_u I_{it}^- + c_{fix} \cdot \mathbb{1}(Q_{it} > 0) \right)$

Subject to the global storage capacity and budget constraints:

1. $\sum_{i} v_i (I_{i,t-1} + Q_{it}) \leq V_{max}$
2. $\sum_{i} p_i Q_{it} \leq B_{total}$
3. $Q_{it} \geq 0, \quad \forall i$

The ABCA iteratively explores the feasible region defined by these constraints, treating order quantities as "food sources" to globally minimize the expected total cost derived from the network's predictions.

### **4.7 Surrogate Model and Gradient Linkage**

Because the ABCA optimization and environment evaluation steps are non-differentiable, standard backpropagation cannot be used. We introduce a surrogate model $\mathcal{S}$ (implemented via LightGBM) to approximate the true operational cost mapping:

$\text{Cost}_{surrogate} = \mathcal{S}(\mathbf{\hat{D}}, \mathbf{X}_{context})$

where $\mathbf{\hat{D}}$ is the neural network's demand prediction and $\mathbf{X}_{context}$ represents environmental parameters (e.g., cost coefficients).

The surrogate model provides a differentiable approximation of the cost landscape, allowing us to compute pseudo-gradients for the neural network update:

$\frac{\partial \text{TotalCost}}{\partial \mathbf{\hat{D}}} \approx \frac{\partial \mathcal{S}(\mathbf{\hat{D}}, \mathbf{X}_{context})}{\partial \mathbf{\hat{D}}}$

### **4.8 End-to-End Custom Loss Function**

The neural network is trained using a composite loss function that balances pure prediction accuracy with decision-focused operational cost, alongside a service level penalty to prevent severe understocking:

$\mathcal{L}_{total} = \alpha \cdot \mathcal{L}_{pred}(\mathbf{\hat{D}}, \mathbf{D}) + (1-\alpha) \cdot \text{Cost}_{surrogate} + \lambda \cdot \max(0, S_{target} - S_{actual})$

where:

- $\mathcal{L}_{pred}$ is the prediction loss (e.g., MSE).
- $\alpha \in [0,1]$ controls the trade-off between prediction accuracy and decision cost.
- $S_{target}$ is the target service level.
- $\lambda$ is the service penalty weight (`service_penalty_weight`).

## **5\. Data and Experimental Design**

We use the **Walmart M5 forecasting dataset**, containing daily sales data for 3,049 SKUs.

### **5.1 Data Transformation**

- Aggregated to weekly level
- Constructed features: price, calendar effects, historical demand
- Constructed cost structure per SKU

### **5.2 Data Splitting**

- Training: early periods
- Validation: middle periods
- Test: final 13 weeks

All models are evaluated using a unified simulation-based cost function.

## **6\. Experimental Results**

### **6.1 Evaluation Metrics**

We evaluate models using decision-oriented metrics:

- Total cost
- Holding cost
- Shortage cost
- Service level

## **6.2 Baseline (Predict-then-Optimize with Newsvendor Policy)**

The baseline follows a standard **Predict-then-Optimize (PtO)** pipeline:

1. Demand forecasting is performed using LightGBM
2. Forecast outputs (mean μ\\muμ, standard deviation σ\\sigmaσ) are passed into a classical Newsvendor decision rule:

Qit=μit+zσit

This approach assumes normally distributed demand uncertainty and applies a uniform critical fractile across all SKUs.

### **Results**

- Total Cost: 2,511,230.01
- Holding Cost: 523,907.84
- Shortage Cost: 5,472.18
- Service Level: 99.89%

### **Observation**

The baseline policy is highly conservative, driven by the asymmetric cost structure where:

cu≫ch

As a result, the Newsvendor solution systematically overestimates safety stock, leading to:

- Extremely low stockout risk
- Excessively high inventory holding cost
- Near-perfect service level (over-provisioning behavior)

## **6.3 Segmented Model (Segmentation \+ Adaptive Decision Rule)**

The segmented model modifies **both the demand modeling strategy and the downstream decision rule**, introducing a regime-dependent inventory policy.

### **(1) Demand Segmentation**

SKUs are partitioned into heterogeneous demand regimes (Smooth, Erratic, Intermittent, Lumpy) based on ADI and CV² metrics.

### **(2) Decision Rule Modification**

Unlike the baseline, the decision policy is no longer a uniform Newsvendor formulation. Instead, it is adapted as follows:

- **Smooth / Erratic demand:**  
   → ML-based forecasting \+ Newsvendor-style decision rule
- **Intermittent / Lumpy demand:**  
   → Heuristic-based policy (mean/quantile-driven ordering rule)

This implies that the segmented model changes both:

- the demand estimation mechanism
- the inventory decision mapping f(D^)→Q

Therefore, performance differences reflect **joint effects of forecasting and decision rule redesign**, rather than segmentation alone.

### **Results**

- Total Cost: 2,413,536.11
- Holding Cost: 382,589.29
- Shortage Cost: 93,101.83
- Service Level: 98.32%

### **Observation**

The segmented system leads to a structural shift in inventory policy:

- Inventory levels are reduced significantly (lower holding cost)
- Stockout frequency increases (higher shortage cost)
- Service level decreases slightly but remains high

Importantly, the improvement in total cost is not solely attributable to segmentation, but to a **combined effect of:**

1. Relaxation of conservative Newsvendor behavior in high-variance demand classes
2. Introduction of simpler heuristics for lumpy/intermittent demand
3. Reduction of over-reliance on Normal-distribution assumptions

### **6.4 Advanced Decision-Focused Model**

This version integrates surrogate learning and heuristic optimization into an end-to-end **Predict-and-Optimize (PaO)** pipeline.

### **(1) End-to-End Learning Loop**

Instead of minimizing statistical forecasting error (MSE), the neural network is trained dynamically to minimize the final operational cost:

- **Forward Pass**: The neural network infers demand predictions.
- **Decision Making**: Predictions are passed to an Artificial Bee Colony Algorithm (ABCA) solver to generate inventory decisions.
- **Cost Evaluation**: Decisions are evaluated in the inventory environment to calculate the actual business cost.
- **Surrogate Gradient**: A LightGBM-based surrogate model approximates the non-differentiable cost landscape, providing pseudo-gradients to backpropagate and update the neural network weights.

### **Results**

**Baseline variant:**

- Total Cost: 46,126.57
- Service Level: 4.97%

**Segmented variant:**

- Total Cost: 44,865.35
- Service Level: 3.93%

### **Observation**

The decision-focused model achieves a radical reduction in total operational cost compared to the Predict-then-Optimize baselines (from ~2.4M to ~45k). However, this comes at the expense of an extreme collapse in service levels.

This phenomenon reveals a critical vulnerability in unconstrained decision-focused learning:

- **Degenerate Policy**: Driven by the objective to minimize total cost without explicit service level constraints, the model learns an overly aggressive "near-zero inventory" policy.
- **Cost Optimization Loophole**: The surrogate model successfully propagates gradients to reduce holding costs, but the system discovers that systematically taking shortage penalties is mathematically "cheaper" than maintaining safety stock under the current proxy cost structure.

##

## **7\. Discussion**

### **7.1 Key Findings**

**Finding 1: Baseline is overly conservative**  
 Due to strong asymmetry between cu​ and ch​, the system heavily over-stocks.

**Finding 2: Segmentation introduces trade-offs**  
 It reduces holding cost but increases shortage risk significantly.

**Finding 3: Decision-focused learning is unstable**  
Cost-sensitive optimization may lead to degenerate policies when constraints are insufficient. The model discovers that for thousands of low-margin or slow-moving SKUs, the high fixed ordering cost ($c_{fix}$) and holding cost outweigh the shortage penalty. As a result, the global optimizer systematically drives the inventory of these SKUs to zero, sacrificing service levels to achieve a mathematical "global minimum" in total operational cost.

### **7.2 Why ML is not used for Lumpy Demand**

Lumpy demand exhibits:

- Sparse non-zero observations
- High variance spikes
- Zero-inflated structure

Standard ML models trained with statistical loss functions (e.g., MSE) tend to smooth extreme values towards the conditional mean. This smoothing behavior fails to capture the rare but critical demand spikes characteristic of lumpy items. Consequently, using ML forecasts with traditional optimization policies often leads to systematic understocking and severe service level degradation for these SKUs. As a result, simpler statistical or heuristic approaches (mean/quantile-driven ordering rules) are preferred, as they provide greater robustness against structural sparsity and extreme variance.

### **7.3 Trade-off Interpretation**

The reduction in total cost under segmentation is primarily driven by a significant decrease in holding cost, which outweighs increased shortage penalties due to:

cu≫ch

However, this comes at the expense of service level deterioration, indicating a shift from conservative to aggressive inventory policies.

## **8\. Limitations and Future Work**

### **8.1 Limitations**

- Simplified cost structure (fixed margins and holding rates)
- Uniform ordering cost assumption
- No substitution or lost-sales recovery modeling
- Gaussian assumption in Newsvendor policy
- Surrogate model approximation error

### **8.2 Future Work**

- Replace Normal assumption with quantile-based policy
- Introduce (s, S) policy optimization
- Incorporate explicit service-level constraints
- Improve surrogate accuracy with hybrid models
- Integrate global constraints directly into learning loop

## **9\. Conclusion**

This study develops a decision-focused inventory optimization framework under heterogeneous demand. Experimental results show that while segmentation and decision-aware learning improve cost efficiency, they also introduce significant trade-offs in service levels.

Overall, the findings suggest that demand heterogeneity must be explicitly modeled, but segmentation alone is insufficient without carefully calibrated cost structures and constraint-aware decision policies.

## **Appendix A: Model Implementation**

### **A.1 Baseline Model**

## \# TODO: insert baseline training \+ Q computation code

### **A.2 Segmented Model**

## \# TODO: insert segmentation \+ prediction code

### **A.3 Inventory Simulation**

## \# TODO: insert simulate_inventory function

### **A.4 Advanced Decision-Focused Model**

Below are the core implementations of the End-to-End Decision Learning pipeline extracted from the project repository.

**1. Main Initialization (`project/main.py`)**

```python
def main():
    # ... (Initialization details omitted for brevity) ...

    # 1. Initialize Predictor (Neural Network)
    predictor = DemandPredictor(
        input_size=1, hidden_size=64, output_size=1, use_category_embedding=args.use_segmentation
    ).to(device)

    # 2. Initialize Heuristic Solver (ABCA) and Environment
    solver = ABCASolver(max_iter=DEFAULT_SOLVER_MAX_ITER, pop_size=DEFAULT_SOLVER_POP_SIZE)
    env = InventoryEnvironment()

    # 3. Initialize Surrogate Model (LightGBM)
    surrogate = SurrogateModel()

    # 4. Execute End-to-End Training Loop
    train_predict_and_optimize(
        dataloader=dataloader,
        predictor=predictor,
        solver=solver,
        env=env,
        surrogate=surrogate,
        epochs=args.epochs,
        # ...
    )
```

**2. End-to-End Training Loop (`project/train/loop.py`)**

```python
def train_predict_and_optimize(dataloader, predictor, solver, env, surrogate, epochs, ...):
    optimizer = Adam(predictor.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        predictor.train()
        for batch_idx, (features, category_idx, true_demand, cost_params_batch) in enumerate(dataloader):
            optimizer.zero_grad()

            # Step 1: Forward Pass (Demand Prediction)
            y_pred_tensor = predictor(features, category_idx)

            # Step 2 & 3: ABCA Optimization & Environment Evaluation
            predictor_out = PredictorOutput(y_pred=y_pred_np)
            solver_out = solver.solve(predictor_out, cost_params_list, build_global_constraints())
            env_out = env.evaluate_cost(solver_out, true_demand_np, cost_params_list)

            # Step 4: Dynamically Train Surrogate Model
            if len(history_y_pred) >= 1000 and batch_idx % surrogate_update_freq == 0:
                surrogate.train_surrogate(history_y_pred, history_context, history_true_cost)

            # Step 5: Backward Pass via Surrogate Autograd
            if surrogate.is_trained:
                cost_tensor = SurrogateAutogradFunction.apply(y_pred_tensor, context_tensor, surrogate)
                cost_loss = cost_tensor.mean()

                # Combine prediction loss with surrogate cost loss and service penalty
                total_loss = build_total_loss(cost_loss, pred_loss) + service_penalty
                total_loss.backward()

                clip_grad_norm_(predictor.parameters(), max_norm=grad_clip_norm)
                optimizer.step()
```

## **References**

**Lahoud, A. (2025).** From predictions to decisions: A survey on decision-focused learning. _Journal of Artificial Intelligence Research_, 78, 112-145.  
**Theodorou, E. G. (2023).** _Decision-Focused Learning: Integrating Optimization into Neural Network Training_. MIT Press.  
**Yılmaz, M., et al. (2025).** Integrated supply chain–warehouse design under demand uncertainty: A robust stochastic programming approach. _European Journal of Operational Research_, 312(2), 450-468.

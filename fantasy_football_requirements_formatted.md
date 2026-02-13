# Comprehensive Requirements for Fantasy Football Player Performance Forecasting System

---

## I. System Architecture & Model Framework

### A. Position-Specific Modeling Requirement

Based on academic research, separate models must be built for each position (QB, RB, WR, TE) because:

- Each position has unique performance drivers and statistical relationships
- QB success depends heavily on passing attempts, completion percentage, and offensive momentum
- RB performance centers on rushing attempts, receptions, and total touchdowns
- WR/TE models must prioritize targets, receptions, and receiving yards
- Cross-position models show **15-20% lower accuracy** than position-specific models

### B. Multi-Horizon Prediction Architecture

The system requires three separate prediction models for different time horizons:

#### 1. 1-Week Ahead Model (Short-term)
- **Primary approach:** Ensemble of Random Forest + XGBoost + Ridge Regression
- Heavy weighting on rolling 3-4 week statistics (optimal window per research)
- Include opponent-specific defensive metrics
- **Target accuracy:** RMSE ≤ 6-8 fantasy points

#### 2. 4-Week Ahead Model (Medium-term)
- Hybrid LSTM-ARIMA approach for capturing temporal dependencies
- Incorporate injury probability scores and workload trends
- Season-to-date statistics with exponentially weighted moving averages
- **Target accuracy:** RMSE ≤ 8-10 fantasy points

#### 3. 18-Week Ahead Model (Season-long)
- Deep neural network (98+ layer feedforward) with historical pattern recognition
- Previous season statistics heavily weighted (70-80% of features)
- Regression to mean adjustments for outlier performances
- **Target accuracy:** RMSE ≤ 12-15 fantasy points

### C. Utilization Score Prediction for RB/WR/TE
Utilization Score information: https://www.fantasylife.com/articles/fantasy/the-utilization-score-understanding-player-roles-and-performance

Since your non-QB positions use Utilization Score as the dependent variable, your system must:

#### 1. Define Utilization Score Components:

- Snap share percentage (opportunity metric)
- Target/touch share (actual usage)
- Red zone involvement percentage
- High-value touch rate (rushes inside 10-yard line, targets 15+ yards)
- Weighted composite score normalized 0-100

#### 2. Model Architecture:
- Position-specific utilization models separate from fantasy point models
https://www.fantasylife.com/articles/fantasy/the-utilization-score-understanding-player-roles-and-performance
- Tree-based models (Random Forest/XGBoost) perform best for bounded 0-100 scores
- Feature importance analysis to identify role changes

#### 3. Conversion Layer:
- Build secondary model to convert Utilization Score → Fantasy Points
- Account for efficiency metrics (yards per touch, TD rate given opportunities)

---

## II. Data Requirements

### A. Historical Player Performance Data

**Minimum 5 seasons, Optimal 10+ seasons**

#### Required Game-Level Statistics (per player, per week):

**For ALL Positions:**
- Fantasy points (PPR, Half-PPR, Standard scoring)
- Snap count and snap percentage
- Offensive snaps when on field
- Time of possession metrics
- Weather conditions (temperature, wind, precipitation)
- Home/away indicator
- Days since last game

#### Position-Specific Metrics:

**QB:**
- Pass attempts, completions, yards, TDs, INTs
- Completion percentage
- Air yards per attempt
- TD/INT ratio
- Rushing attempts, yards, TDs
- Sack count
- Time in pocket
- Deep ball attempts (20+ yards)
- Red zone attempts and efficiency

**RB:**
- Rush attempts, yards, TDs
- Receptions, targets, receiving yards, receiving TDs
- Red zone carries and targets
- Goal line carries (inside 5-yard line)
- Snap share percentage
- Route participation rate
- Yards after contact
- Broken tackles

**WR/TE:**
- Targets, receptions, receiving yards, TDs
- Air yards and average depth of target (aDOT)
- Yards after catch (YAC)
- Target share percentage
- Red zone targets
- Contested catch rate
- Route tree diversity (% of route types run)
- Slot vs. outside alignment percentage

### B. Team-Level Data (Season and Game-Level)

#### Offensive System Metrics:
- Total plays per game
- Pass/run ratio
- Pace of play (plays per minute)
- Red zone efficiency
- Third down conversion rate
- Time of possession
- Offensive line quality metrics (pressure rate allowed, run blocking grades)
- Coaching staff continuity and scheme

#### Defensive Opposition Metrics:
- Points allowed to position (DVP - Defense vs Position)
- Yards allowed to position
- Fantasy points allowed to position (last 4 weeks, season-to-date)
- Defensive scheme (Cover 2, Cover 3, man coverage rates)
- Pressure rate generated
- Defensive line quality metrics

#### Situational Team Data:
- Win/loss record and playoff positioning
- Point differential (average margin of victory/defeat)
- Offensive coordinator changes
- Home field advantage metrics

### C. Injury Data (Critical for Predictive Accuracy)

Research shows injury data improves model accuracy by **12-18%**.

#### Required data points:

**Historical Injury Database:**
- Injury type and severity (game missed classification)
- Body part affected
- Injury designation (Questionable, Doubtful, Out)
- Recovery timeline patterns by injury type
- Previous injury history by player
- Minimum 5 years of comprehensive injury reports

**Real-Time Injury Tracking:**
- Weekly injury report status
- Practice participation (full/limited/DNP)
- Injury probability score (0-100% chance of playing)
- Workload management indicators
- Return-from-injury production patterns (first 3 games back)

**Predictive Injury Risk Modeling:**
- Cumulative workload metrics (carries/targets over 3-4 weeks)
- Age-adjusted injury probability
- Position-specific injury risk factors (RB: high-touch rate, QB: sack rate)
- Historical injury recurrence rates

### D. Rookie and Draft Capital Data

#### For Rookie Projections (18-week models):

**Draft Information:**
- Draft position (1st round, 2nd round, etc.)
- College statistics (last 2 seasons minimum)
  - QB: Completion %, yards per attempt, TD/INT ratio
  - RB/WR: Yards per touch, target share, breakaway speed

**NFL Combine Measurables:**
- 40-yard dash time
- Vertical jump, broad jump
- 3-cone drill, shuttle time
- Height, weight, arm length

**Context Factors:**
- College competition level adjustments
- Landing spot quality (offensive scheme fit, opportunity score)

#### Historical Draft Class Performance:
- Positional success rates by draft round (% who become fantasy-relevant)
- Rookie year production curves by position
- Year 2-3 progression patterns

### E. Minimum Dataset Sizes

Research indicates optimal training data requirements:

#### For 1-Week Models:
- **Minimum:** 3 seasons of weekly data (~51 weeks per position group)
- **Optimal:** 5+ seasons (~85+ weeks per position group)
- Rolling window: Previous 3-4 weeks heavily weighted

#### For 4-Week Models:
- **Minimum:** 5 seasons
- **Optimal:** 8+ seasons
- Temporal patterns require longer history

#### For 18-Week Models:
- **Minimum:** 8 seasons of complete season data
- **Optimal:** 10+ seasons
- Include at least 2 full economic/competitive cycles

#### Per-Position Player Minimums:
- **QB:** 30+ active players with 8+ games/season for training
- **RB:** 60+ active players (position volatility requires larger sample)
- **WR:** 70+ active players
- **TE:** 30+ active players

---

## III. Feature Engineering Requirements

### A. Temporal Features (Critical for Time Series)

#### Rolling Averages (Windows: 3, 4, 5, 8 weeks):
- Fantasy points per game
- Utilization score (for RB/WR/TE): 
- Usage rate metrics (touches, targets, snaps)
- Efficiency metrics (yards per touch, catch rate)

#### Trend Indicators:
- Direction of performance (improving vs. declining)
- Volatility measures (standard deviation of weekly scores)
- Boom/bust rates (% of weeks >20 points, <5 points)

#### Seasonal Patterns:
- Week of season (1-18)
- Early season (weeks 1-6) vs. mid-season vs. late season indicators
- Divisional game indicator
- Prime-time game indicator

### B. Contextual Features

#### Offensive Momentum Score (Per Research - High Predictive Value):

**Formula:** Weighted combination of:
- Team offensive EPA (Expected Points Added) trend
- Offensive line continuity and performance trend
- Pass/rush success rate trends
- Scoring efficiency trend
- **Time-weighted:** Recent 4 weeks = 60%, weeks 5-8 = 30%, weeks 9+ = 10%

#### Game Script Predictors:
- Vegas point spread
- Over/under total
- Implied team total (over/under + spread)
- Win probability

#### Matchup Quality Indicators:
- Opponent defensive rank by position
- Historical performance against opponent
- Strength of schedule remaining

### C. Player Attribute Features

#### Physical/Demographic:
- Age and age-adjusted performance curves
- Height/weight for position
- NFL experience (years in league)
- Contract year indicator

#### Usage Pattern Classification:
- Workhorse vs. committee back (RBs)
- WR1/WR2/WR3 designation
- Three-down back indicator
- Red zone specialist indicator

---

## IV. Model Selection & Optimization

### A. Recommended Model Architecture by Time Horizon

#### 1-Week Ahead (Highest Accuracy Priority):

**Primary Ensemble Model:**

1. **Random Forest (30% weight):**
   - Handles non-linear relationships, feature importance
   - Hyperparameters: 500-1000 trees, max_depth 10-15, min_samples_split 5-10

2. **XGBoost (40% weight):**
   - Best for structured data, handles missing values
   - Hyperparameters: learning_rate 0.01-0.05, max_depth 6-8, n_estimators 500-1000

3. **Ridge Regression (30% weight):**
   - Provides stable baseline, handles multicollinearity
   - Hyperparameters: alpha 1.0-10.0 (cross-validated)

**Validation Strategy:**
- K-fold cross-validation (k=5) on historical seasons
- Walk-forward validation: Train on seasons 1-N, test on season N+1
- Position-specific tuning for each model

**Expected Performance (Based on Literature):**
- QB RMSE: 6.0-7.5 points
- RB RMSE: 7.0-8.5 points
- WR RMSE: 6.5-8.0 points
- TE RMSE: 5.5-7.0 points

#### 4-Week Ahead (Medium-term):

**Hybrid LSTM-ARIMA Architecture:**

**LSTM Component (60% weight):**
- Architecture: 3-4 LSTM layers, 128-256 units per layer
- Dropout rate: 0.2-0.3 between layers
- Sequence length: 8-12 weeks of historical data
- Activation: Tanh (proven effective in research)
- Optimizer: Adam with learning rate 0.001
- Batch size: 32-64
- Epochs: 50-100 (monitor for overfitting)

**ARIMA Component (40% weight):**
- Auto ARIMA for parameter optimization (p, d, q)
- Typical parameters: p=1-3, d=1, q=1-3
- Seasonal components if detected
- AIC/BIC for model selection

**Expected Performance:**
- RMSE increase of 15-25% vs 1-week models
- QB RMSE: 8-10 points
- RB RMSE: 9-11 points

#### 18-Week Ahead (Season-long):

**Deep Neural Network + Ensemble:**

**Deep Feedforward Network (Research shows 98-layer architectures effective):**
- Input layer: 150-200 features
- Hidden layers: 95+ layers with decreasing nodes (512→256→128→64...)
- Dropout: 0.3-0.5 to prevent overfitting
- Batch normalization between layers
- Activation: ReLU for hidden, Linear for output

**Secondary Ensemble with Traditional ML:**
- Gradient Boosting Machines
- Random Forest
- Linear regression with regularization
- Blend deep learning (70%) + ensemble (30%)

**Expected Performance:**
- RMSE: 12-15 points for QB/RB/WR
- Correlation with actual: r = 0.65-0.75

### B. Hyperparameter Optimization Requirements

#### Grid Search Strategy:
- Exhaustive grid search for critical parameters
- Random search for secondary parameters
- Use validation set (not test set) for tuning

#### Key Hyperparameters to Optimize:

**Tree-Based Models:**
- n_estimators (100-1000)
- max_depth (5-20)
- min_samples_split (2-20)
- min_samples_leaf (1-10)
- learning_rate (0.01-0.3 for boosting)
- subsample ratio (0.6-1.0)

**Neural Networks:**
- Learning rate (0.0001-0.01)
- Batch size (16-128)
- Number of layers (3-10 for LSTM, 50-100+ for deep feedforward)
- Hidden units per layer (64-512)
- Dropout rate (0.2-0.5)
- L1/L2 regularization (0.0001-0.01)

#### Cross-Validation Protocol:
- Minimum 5-fold cross-validation
- Time series-aware splits (no data leakage from future)
- Stratified by position if using combined models
- Report mean and standard deviation of metrics

---

## V. Evaluation Metrics & Benchmarks

### A. Primary Accuracy Metrics

#### Root Mean Squared Error (RMSE):
- Most common metric in fantasy football research
- Penalizes large errors more than small ones
- Position-specific benchmarks (see Section IV.A)

#### Mean Absolute Error (MAE):
- More interpretable than RMSE
- Equal penalty for all errors
- **Target:** MAE 20-25% lower than RMSE

#### Mean Absolute Percentage Error (MAPE):
- Percentage-based for cross-position comparison
- **Target:** <25% for 1-week, <35% for 4-week, <45% for 18-week

#### R-Squared (R²):
- Measures proportion of variance explained
- **Target:** R² > 0.50 for 1-week, > 0.40 for 4-week, > 0.30 for 18-week

### B. Fantasy-Specific Evaluation Metrics

#### Ranking Accuracy:
- Spearman rank correlation between predicted and actual rankings
- **Target:** ρ > 0.65 for top-50 players per position
- More important than absolute score prediction for drafting

#### Tier Classification Accuracy:
- Divide players into tiers (Elite/Strong/Flex/Waiver)
- Classification accuracy for tier assignment
- **Target:** >75% correct tier classification

#### Boom/Bust Prediction:
- Ability to predict high-variance outcomes
- Precision/Recall for predicting 20+ point weeks
- Precision/Recall for predicting <5 point weeks

#### Value Over Replacement Player (VOR) Accuracy:
- Predict VOR rankings, compare to actual
- Critical for draft strategy optimization

### C. Benchmarking Requirements

#### Compare Against:

**1. Expert Consensus Projections (e.g., FantasyPros, FantasyData)**
- Research shows ML models should match or exceed by 5-15%
- Weekly head-to-head RMSE comparison

**2. Naive Baselines:**
- Previous week's score (persistence model)
- Season average to date
- Position average
- Model must beat all baselines by >20%

**3. Position-Specific Thresholds:**
- **QB:** Beat expert projections by 8-12%
- **RB:** Beat expert projections by 10-15% (highest variance position)
- **WR:** Beat expert projections by 8-12%
- **TE:** Beat expert projections by 12-18% (TE predictions notoriously poor)

---

## VI. Special Considerations

### A. Class Imbalance Handling

#### Injury Prediction:
- Severe class imbalance (most players healthy each week)
- Solutions: SMOTE, class weighting, anomaly detection approaches
- Research shows 56% recall, 40% precision achievable for injury risk

#### Rookie Success:
- Most rookies underperform, small % breakout
- Separate models for rookies vs. veterans
- Lower confidence intervals for rookie predictions

### B. Explainable AI Requirements

#### Model Interpretability Tools:
- SHAP (SHapley Additive exPlanations) values for feature importance
- Partial dependence plots for key features
- Individual prediction explanations
- Critical for user trust and debugging

#### Feature Importance Ranking:
- Top 10 features per position per model
- Track feature importance stability across seasons
- Identify when feature relationships change (regime shifts)

### C. Data Quality & Preprocessing

#### Missing Data Handling:
- Maximum 5% missing data per feature acceptable
- Imputation strategies: position-specific medians, carry-forward for streaks
- Document all imputation decisions

#### Outlier Detection:
- Statistical outliers (>3 standard deviations) require investigation
- Legitimate outliers (record-breaking performances) should be kept
- Injury-impacted games may need special handling

#### Feature Scaling:
- Standardization (z-scores) for neural networks
- Tree-based models generally don't require scaling
- Min-max scaling for bounded features (percentages, rates)

### D. Production System Requirements

#### Update Frequency:
- Weekly model retraining with new data
- Daily injury/news updates to input features
- Real-time lineup adjustment for active/inactive status

#### Confidence Intervals:
- Provide prediction intervals (e.g., 80%, 95% CI) not just point estimates
- Wider intervals for volatile players, rookies, injury-prone
- Research shows 88.2% of players within 10-point confidence band achievable

#### Versioning & Monitoring:
- Track model version, training date, performance metrics
- Monitor for model drift (performance degradation over time)
- A/B testing framework for model updates
- Rollback capability if new model underperforms

---

## VII. Success Criteria

Your system will be considered **"robust and accurate"** if it achieves:

### Accuracy Thresholds (1-week predictions):
- RMSE within 10% of expert consensus
- Ranking accuracy (Spearman ρ) > 0.65
- Beat naive baselines by >25%

### Reliability Thresholds:
- 80%+ of predictions within 10 points actual (per ESPN research)
- 70%+ of predictions within 7 points actual
- Stable performance across full season (no >20% accuracy degradation)

### Business Thresholds:
- Weekly update cycle <24 hours
- Prediction generation <5 seconds per player
- Explainable predictions for top-200 players

---

**Document Version:** 1.0  
**Last Updated:** [Current Date]

# Data Science and Machine Learning Development Guidelines

## Technical Stack & Dependencies
- **Core Libraries**: JAX, NumPy, Pandas, Scikit-learn
- **Feature Engineering**: FeatureTools, AutoKeras
- **Optimization**: Optuna, KerasTuning
- **Data Quality**: OpenRefine, PYDQC
- **Visualization**: Matplotlib, Seaborn, Yellowbricks

## Code Style and Structure
- Write concise, technical Python code following PEP 8 guidelines
- Implement functional programming patterns; minimize class usage
- Utilize vectorized operations over explicit loops
- Use descriptive variable names (e.g., `learning_rate`, `feature_importance`)
- Organize code into modular functions and packages

## Data Science Workflow (OSEMN Framework)
### 1. Obtain
- Validate input data types and shapes
- Document data sources and collection methods
- Implement proper error handling for data loading

### 2. Scrub (Data Cleaning)
1. Initial Quality Assessment:
   - Use PYDQC for feature statistics and error detection
   - Apply OpenRefine for user-guided transformations
   - Document all cleaning steps in transformation key table

2. Missing Data Handling:
   - Detect and document missing/sparse data patterns
   - Implement appropriate imputation strategies
   - Provide UI for transformation selection

3. Data Scaling and Normalization:
   - For minimal outliers: MaxAbsScaler or RobustScaler
   - For significant outliers: QuantileTransformer
   - Apply log transformations for skewed distributions
   - Use Box-Cox (positive data) or Yeo-Johnson (any data) transforms
   - For large datasets: bin with QuantileTransformer before transforms

4. Categorical Data Processing:
   - Convert to Boolean, Ordinal, or Nominal formats
   - Apply One-Hot encoding for nominal variables
   - Use Label encoding for ordinal data

### 3. Explore
- Generate comprehensive statistical summaries
- Create visualization suites using Matplotlib/Seaborn
- Analyze data distributions and correlations
- Detect and handle imbalanced datasets

### 4. Model
1. Feature Engineering:
   - Automate using autonormalize and featuretools
   - Apply Lasso/Ridge regularization
   - Implement polynomial feature engineering where appropriate
   - Use binning techniques (Equal Width, Equal Frequency)
   - Optimize binning with OptBinning

2. Dimensionality Reduction:
   - Linear methods: PCA, LDA, QDA, NMF, SVD
   - Non-linear methods: Manifold Learning
   - Select based on data characteristics

3. Model Selection:
   - AutoKeras for deep learning (with sufficient data)
   - Traditional ML models including ensembles and XGBoost
   - Optimize hyperparameters using Optuna
   - Use KerasTuner for neural network architecture

4. Training Best Practices:
   - Select appropriate cross-validation (GridSearch, RandomSearch, Bootstrap)
   - Implement early stopping and model checkpoints
   - Use appropriate batch sizes and learning rates

### 5. iNterpret
1. Classification Metrics:
   - Confusion Matrix
   - Precision, Recall, F1-Score
   - ROC and AUC curves

2. Regression Metrics:
   - MSE (Mean Square Error)
   - RMSE (Root Mean Squared Error)
   - MAE (Mean Absolute Error)

## JAX-Specific Guidelines
- Use `jax.numpy` instead of standard NumPy
- Leverage automatic differentiation with `jax.grad`
- Apply `jax.jit` for performance optimization
- Use `jax.vmap` for vectorization
- Maintain immutability of arrays
- Implement pure functions for JAX transformations

## MLOps Best Practices
- Version control all code and models
- Document hyperparameters and training configurations
- Implement reproducible random seeds
- Monitor model performance in production
- Create automated testing pipelines
- Maintain experiment tracking

## Performance Optimization
- Profile code for bottlenecks
- Optimize memory usage
- Minimize data transfers between CPU/GPU
- Reuse compiled functions when possible
- Implement efficient batching strategies

## Documentation Requirements
- Comprehensive docstrings (PEP 257)
- Clear API documentation
- Usage examples and tutorials
- Performance benchmarks
- Model cards for deployed models

For the latest updates and detailed implementation guidelines, refer to the official documentation of each library.

# Customer Segmentation using K-Means Clustering

## Project Overview

This project uses customer data to identify meaningful customer segments
using unsupervised machine learning.

The workflow includes:

-   Exploratory Data Analysis (EDA)
-   Data preprocessing
-   Feature selection
-   Feature scaling
-   K-Means clustering
-   Hierarchical clustering
-   DBSCAN experimentation
-   Cluster profiling and business interpretation
-   Streamlit deployment

The final application allows a user to enter customer information and
receive a predicted customer segment along with a recommended business
strategy.

------------------------------------------------------------------------

## Customer Segments

The final K-Means model uses **3 clusters**.

  -----------------------------------------------------------------------
  Cluster                 Segment                 General Interpretation
  ----------------------- ----------------------- -----------------------
  0                       High-Value Customers    Higher income, spending
                                                  and purchase activity

  1                       Low-Value /             Lower spending and
                          Low-Conversion          purchase activity with
                          Customers               relatively higher web
                                                  visits

  2                       Deal-Oriented &         Higher deal purchases,
                          Digitally Active        web activity and longer
                          Customers               customer tenure
  -----------------------------------------------------------------------

> Cluster numbers are model-generated labels. The business names are
> interpretations based on the cluster profiles.

------------------------------------------------------------------------

## Features Used

The deployed model expects these 8 features:

1.  `Income`
2.  `Recency`
3.  `Tenure_Days`
4.  `Total_Spending`
5.  `Total_Purchases`
6.  `NumDealsPurchases`
7.  `NumWebVisitsMonth`
8.  `Total_Children`

The same feature order must be maintained when making predictions.

------------------------------------------------------------------------

## Machine Learning Approach

### 1. Data Preprocessing

The dataset was prepared for clustering by handling data-quality issues,
removing unnecessary variables and creating relevant customer-level
features.

### 2. Feature Scaling

`StandardScaler` was used so that variables with different numerical
ranges could contribute fairly to the clustering process.

### 3. K-Means Clustering

K-Means was selected as the primary clustering model.

The selected solution uses:

``` text
K = 3
```

The K-Means silhouette score obtained during evaluation was
approximately:

``` text
0.2656
```

### 4. Hierarchical Clustering

Hierarchical clustering was also tested using a dendrogram and different
cluster counts.

### 5. DBSCAN

DBSCAN was experimented with using different combinations of `eps` and
`min_samples`.

The results showed that DBSCAN produced substantial noise and generally
weaker clustering quality for this dataset compared with K-Means.

Therefore, K-Means was retained as the primary model for deployment.

------------------------------------------------------------------------

## Streamlit Application

The project includes a Streamlit web application.

The application provides:

-   Customer input form
-   Customer segment prediction
-   Segment description
-   Recommended business strategy
-   Customer vs. segment comparison
-   Customer segment distribution
-   Segment profile comparison
-   Customer input summary

------------------------------------------------------------------------

## Project Structure

``` text
Customer_Segmentation/
│
├── app.py
├── scaler.pkl
├── kmeans_model.pkl
├── requirements.txt
├── README.md
│
└── Customer_Segmentation_EDA.ipynb
```

------------------------------------------------------------------------

## Installation

### 1. Clone the repository

``` bash
git clone <your-github-repository-url>
cd Customer_Segmentation
```

### 2. Create a virtual environment

``` bash
python -m venv .venv
```

Activate it on Windows:

``` powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

If Streamlit is not recognized, install it using:

``` bash
python -m pip install streamlit
```

------------------------------------------------------------------------

## Running the Application

Start the Streamlit application with:

``` bash
streamlit run app.py
```

If the `streamlit` command is not recognized, use:

``` bash
python -m streamlit run app.py
```

The application will open locally in your browser, usually at:

``` text
http://localhost:8501
```

------------------------------------------------------------------------

## Model Files

The application requires two trained model files:

``` text
scaler.pkl
kmeans_model.pkl
```

### `scaler.pkl`

Contains the fitted `StandardScaler` used during model training.

### `kmeans_model.pkl`

Contains the trained K-Means clustering model.

The scaler and model must be compatible with the same feature set and
feature order used during training.

------------------------------------------------------------------------

## Important Deployment Note

The model files were created with a specific Python/scikit-learn
environment.

For reliable deployment, use compatible package versions between the
environment where the models were trained and the deployment
environment.

If a pickle compatibility error occurs, retrain/export the model using
the environment used for deployment.

------------------------------------------------------------------------

## Business Interpretation

The clustering results can help a business design different strategies
for different customer groups.

### High-Value Customers

Potential strategies:

-   Loyalty rewards
-   Premium offers
-   Retention campaigns
-   Cross-selling
-   Personalized recommendations

### Low-Value / Low-Conversion Customers

Potential strategies:

-   Personalized recommendations
-   Targeted offers
-   Conversion campaigns
-   Re-engagement campaigns

### Deal-Oriented & Digitally Active Customers

Potential strategies:

-   Discounts
-   Promotional campaigns
-   Online offers
-   Deal-based recommendations
-   Digital re-engagement

------------------------------------------------------------------------

## Project Limitations

-   K-Means assumes relatively compact cluster structures.
-   Cluster labels do not have an inherent business meaning and must be
    interpreted using cluster profiles.
-   The clustering quality is moderate, so the segments should be
    treated as analytical groups rather than absolute customer
    categories.
-   The recommendations are business interpretations and are not
    directly learned by the machine-learning model.
-   The model should be retrained if the underlying customer population
    or feature definitions change significantly.

------------------------------------------------------------------------

## Future Improvements

Possible improvements include:

-   Testing additional clustering algorithms
-   Hyperparameter tuning
-   Better feature engineering
-   Automated model selection
-   Adding customer-level historical data
-   Adding interactive cluster visualizations
-   Monitoring model performance after deployment
-   Adding authentication and database integration
-   Deploying the Streamlit application publicly

------------------------------------------------------------------------

## Technologies Used

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   Matplotlib
-   Seaborn
-   Streamlit
-   Joblib
-   Jupyter Notebook

------------------------------------------------------------------------

## Author

Customer Segmentation project developed as part of a machine-learning /
data-analysis project.

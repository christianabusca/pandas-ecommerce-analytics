# 🛒 Pandas E-Commerce Analytics

> Retail transaction analysis using Pandas for data manipulation and business insights

[![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Data Source](https://img.shields.io/badge/Data-Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/vijayuv/onlineretail)

---

## 📖 About

Personal data analysis project exploring e-commerce transaction patterns using Pandas. Focuses on data cleaning, feature engineering, customer segmentation, and time-series analysis of retail sales data.

**Data Source:** Online Retail Dataset (Kaggle/UCI ML Repository)  
**Scope:** Multi-country transaction data with invoices, products, and customer info  
**Tools:** Pandas, NumPy, Matplotlib/Seaborn

---

## 🗂️ Project Structure

```
pandas-ecommerce-analytics/
├── data/
│   ├── raw/
│   │   └── online_retail.csv
│   └── processed/
│       ├── cleaned_data.csv
│       └── rfm_segments.csv
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_grouping_aggregation.ipynb
│   ├── 04_time_series_analysis.ipynb
│   ├── 05_customer_segmentation.ipynb
│   └── 06_advanced_operations.ipynb
├── src/
│   ├── cleaning.py
│   ├── features.py
│   ├── rfm_analysis.py
│   └── visualizations.py
├── outputs/
│   ├── figures/
│   └── reports/
├── requirements.txt
└── README.md
```

---

## 📊 Analysis Components

### **1. Data Cleaning**
Preparing raw transaction data for analysis
- Load and examine dataset structure (`.info()`, `.describe()`)
- Handle missing values in `CustomerID` and `Description`
- Remove cancelled orders (`InvoiceNo` starting with 'C')
- Filter invalid data (negative quantities/prices)
- Data quality validation

### **2. Feature Engineering**
Creating analytical features
- **Transaction value**: `Quantity × Price`
- **Temporal features**: Extract year, month, day, hour from `InvoiceDate`
- **Revenue categories**: Low/Medium/High based on transaction value
- **Customer flags**: First-time vs. repeat customers
- **Product metrics**: Items per transaction

### **3. Grouping & Aggregation**
Business metrics and KPIs
- **Revenue by country**: Total sales per market
- **Customer metrics**: Average order value (AOV)
- **Product analytics**: Unique products sold per month
- **Top performers**: Best-selling products by quantity and revenue
- **Multi-level aggregations**: Country × Month × Product

### **4. Time Series Analysis**
Temporal patterns and trends
- **Resampling**: Daily, weekly, monthly revenue trends
- **Growth metrics**: Month-over-month (MoM) growth rate
- **Seasonality**: Busiest shopping hours/days
- **Trend decomposition**: Identifying sales patterns
- **Rolling statistics**: Moving averages

### **5. Customer Segmentation**
RFM analysis and customer profiling
- **Recency**: Days since last purchase
- **Frequency**: Number of orders per customer
- **Monetary**: Total customer lifetime value
- **VIP identification**: Top 10% customers by spend
- **Purchase behavior**: One-time vs. repeat customers
- **Segment profiles**: Customer characteristics

### **6. Advanced Operations**
Complex Pandas techniques
- **Pivot tables**: Revenue by Country × Month
- **Custom functions**: `.apply()` for categorization
- **Method chaining**: Clean, readable transformations
- **Multi-index operations**: Hierarchical data analysis
- **Data merging**: Combine multiple datasets

---

## 🎯 Key Business Insights

### Sales Performance
- Total revenue and transaction count
- Revenue distribution by country
- Best-selling products and categories
- Average order value trends

### Customer Behavior
- Customer lifetime value distribution
- Purchase frequency patterns
- Retention rates
- Customer segmentation profiles

### Temporal Patterns
- Peak shopping hours and days
- Seasonal trends
- Growth trajectories
- Sales forecasting foundations

---

## 🛠️ Pandas Techniques Used

**Data Manipulation**
- `.head()`, `.info()`, `.describe()` - Data exploration
- `.loc[]`, `.iloc[]` - Row/column selection
- `.dropna()`, `.fillna()` - Missing data handling
- `.drop_duplicates()` - Data deduplication
- `.astype()` - Type conversion

**Feature Engineering**
- `.dt` accessor - DateTime operations
- `.apply()`, `.map()` - Custom transformations
- `.assign()` - New column creation
- `.cut()`, `.qcut()` - Binning

**Aggregation**
- `.groupby()` - Group operations
- `.agg()` - Multiple aggregations
- `.pivot_table()` - Cross-tabulation
- `.resample()` - Time-series resampling
- `.rolling()` - Moving windows

**Advanced Operations**
- Method chaining - Clean pipelines
- `.merge()`, `.join()` - Combining datasets
- `.query()` - SQL-like filtering
- `.transform()` - Group-wise operations
- Multi-index slicing

---

## 🚀 Setup

### **Prerequisites**
```bash
Python 3.8+
pandas 1.3+
numpy 1.21+
matplotlib, seaborn (for visualizations)
```

### **Installation**
```bash
# Clone the repository
git clone https://github.com/yourusername/pandas-ecommerce-analytics.git
cd pandas-ecommerce-analytics

# Install dependencies
pip install -r requirements.txt
```

### **Data Acquisition**
**Option 1: Kaggle**
1. Visit [Online Retail Dataset](https://www.kaggle.com/datasets/vijayuv/onlineretail)
2. Download `online_retail.csv`
3. Place in `data/raw/`

**Option 2: UCI ML Repository**
```bash
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx
# Convert to CSV if needed
```

**Expected columns:**
- `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`

---

## 📈 Usage

### **Quick Start**
```python
import pandas as pd
from src.cleaning import clean_data
from src.features import engineer_features
from src.rfm_analysis import calculate_rfm

# Load and clean data
df = pd.read_csv('data/raw/online_retail.csv', encoding='ISO-8859-1')
df_clean = clean_data(df)

# Feature engineering
df_features = engineer_features(df_clean)

# RFM analysis
rfm_df = calculate_rfm(df_features)
print(rfm_df.head())
```

### **Example: Data Cleaning**
```python
# Remove cancelled orders and invalid data
df_clean = (df
    .query("~InvoiceNo.str.startswith('C')")  # No cancellations
    .query("Quantity > 0 and UnitPrice > 0")  # Valid values
    .dropna(subset=['CustomerID'])            # No missing customers
)
```

### **Example: Revenue by Country**
```python
# Total revenue per country
revenue_by_country = (df_clean
    .assign(TotalValue=lambda x: x['Quantity'] * x['UnitPrice'])
    .groupby('Country')['TotalValue']
    .sum()
    .sort_values(ascending=False)
)
print(revenue_by_country.head(10))
```

### **Example: RFM Segmentation**
```python
# Calculate RFM metrics
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

rfm = (df_clean
    .groupby('CustomerID')
    .agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
        'InvoiceNo': 'nunique',                                    # Frequency
        'TotalValue': 'sum'                                        # Monetary
    })
    .rename(columns={'InvoiceDate': 'Recency', 
                     'InvoiceNo': 'Frequency', 
                     'TotalValue': 'Monetary'})
)

# Segment customers
rfm['RFM_Score'] = (
    rfm['Recency'].rank(ascending=False, pct=True) * 100 +
    rfm['Frequency'].rank(pct=True) * 100 +
    rfm['Monetary'].rank(pct=True) * 100
) / 3

# Identify VIP customers (top 10%)
vip_threshold = rfm['RFM_Score'].quantile(0.9)
rfm['Segment'] = rfm['RFM_Score'].apply(
    lambda x: 'VIP' if x >= vip_threshold else 'Regular'
)
```

### **Example: Time Series Analysis**
```python
# Monthly revenue trend with growth rate
monthly_revenue = (df_clean
    .set_index('InvoiceDate')
    .resample('M')['TotalValue']
    .sum()
)

monthly_revenue_growth = monthly_revenue.pct_change() * 100
print(f"Average MoM growth: {monthly_revenue_growth.mean():.2f}%")
```

---

## 📊 Sample Outputs

### Revenue Summary
```
Total Revenue: $9,747,748.28
Total Transactions: 397,884
Average Order Value: $24.49
Unique Customers: 4,372
Unique Products: 3,684

Top 5 Countries by Revenue:
1. United Kingdom  $8,187,806.36
2. Netherlands       $284,661.54
3. EIRE              $263,276.82
4. Germany           $221,698.21
5. France            $197,403.90
```

### Customer Segments
```
VIP Customers (Top 10%): 437 customers
- Average Spend: $6,850.32
- Average Orders: 42.3
- Represent 71.2% of total revenue

Repeat Customers: 3,281 (75.0%)
One-time Customers: 1,091 (25.0%)
```

---

## 💡 Development Notes

**Data Quality Checks**
- Always inspect `.info()` for data types
- Check for duplicates with `.duplicated()`
- Validate ranges for numerical columns

**Performance Tips**
- Use `.query()` for faster filtering
- Chain methods for memory efficiency
- Use categorical dtypes for low-cardinality columns
- Avoid loops - vectorize with `.apply()` or NumPy

**Common Patterns**
```python
# Method chaining for clean pipelines
result = (df
    .query("Quantity > 0")
    .assign(Revenue=lambda x: x['Quantity'] * x['Price'])
    .groupby('Country')['Revenue']
    .sum()
    .sort_values(ascending=False)
)
```

---

## 🔬 Future Enhancements

- [ ] Market basket analysis (association rules)
- [ ] Customer churn prediction
- [ ] Product recommendation system
- [ ] Time series forecasting (ARIMA/Prophet)
- [ ] Geographic visualization (folium maps)
- [ ] Interactive dashboard (Plotly/Dash)

---

## 📝 License

MIT License

---

## 🙏 Credits

**Data Source:** Online Retail Dataset (Kaggle/UCI ML Repository)  
**Dataset License:** CC BY 4.0

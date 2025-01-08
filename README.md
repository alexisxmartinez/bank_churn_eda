# Bank Churn EDA

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

# Bank Customer Churn Analysis

This notebook performs Exploratory Data Analysis (EDA) on a bank customer churn dataset. It explores the demographics, financial information, and engagement patterns of customers to understand the factors contributing to churn. The analysis also sets the stage for building a predictive model in the future.

## Dataset

The dataset used is `Bank_Churn.csv`, which contains information about bank customers and whether they have churned (i.e., closed their account). Here's a brief description of the key fields, according to your image:

| Field            | Description                                                        |
|------------------|--------------------------------------------------------------------|
| `CustomerId`      | A unique identifier for each customer                              |
| `Surname`          | The customer's last name                                          |
| `CreditScore`      | A numerical score representing the customer's creditworthiness       |
| `Geography`        | The country where the customer is located (e.g., France, Germany, Spain)  |
| `Gender`           | The customer's gender                                              |
| `Age`              | The customer's age                                                  |
| `Tenure`           | The number of years the customer has been with the bank            |
| `Balance`          | The customer's account balance                                    |
| `NumOfProducts`    | The number of bank products the customer uses                      |
| `HasCrCard`       | Whether the customer has a credit card with the bank (1=Yes, 0=No)    |
| `IsActiveMember`  | Whether the customer is an active member (1=Yes, 0=No)       |
| `EstimatedSalary`| The estimated annual salary of the customer                         |
| `Exited`           | Whether the customer has churned (1=Yes, 0=No)                     |

## Code Overview:

The notebook is structured to perform the following analyses:

1.  **Data Loading and Preparation**:

    *   Imports necessary libraries such as pandas, numpy, plotly, matplotlib and seaborn.
    *   Loads the `Bank_Churn.csv` file into a pandas DataFrame.
    *   Prints the first few rows of the data to show how it looks.
    *   Displays the shape of the DataFrame, data types of columns, and memory usage.
    *   Provides basic descriptive statistics and information about missing values.

2.  **Customer Demographics Analysis**:

    *   **Age Distribution:**
        *   Analyzes the distribution of customer ages using a histogram and boxplot, showing the age ranges of churners vs non churners.
    *   **Geographic Distribution:**
        *   Calculates and visualizes the churn rate for each country (France, Germany, Spain) using a bar chart.

3.  **Financial Analysis**:

    *   **Balance Distribution:**
        *   Analyzes the distribution of account balances, segmented by geography and churn status using a box plot.
    *  **Credit Score Analysis:**
       *   Analyzes the correlation between the credit score and balance using a scatterplot

4.  **Product Analysis:**

    *   **Number of Products:**
        *   Analyzes and shows the churn rate based on the number of products the user has, using a bar chart.

5.  **Correlation Analysis**:

    *   **Correlation Matrix:**
        *   Generates and displays the correlation matrix to identify relationships between numeric variables using a heatmap.

6.  **Customer Segmentation**:

    *   **Balance Segmentation**:
        *   Creates a column that segments the balance of users into `Low`, `Medium-Low`, `Medium-High`, and `High` categories based on quantiles.
    *   **Age Segmentation**:
        *   Creates a column that segments the age of the users into different age groups: 18-29, 30-39, 40-49, 50-59, 60-69, 70-79, 80-89.
    *  **Customer Segmentation Analysis**:
        *   Using a treemap visualizes the different categories of customer segments based on age and balance, and what are the churn rates of each one.

7.  **Saving Insights**:

    *   **Export to JSON:** Saves key insights of the analysis into a JSON file `insights.json` in the `data/processed` directory.

## Key Functions and Techniques

*   **`pandas`:** For data manipulation, cleaning, and analysis.
*   **`numpy`:** For numerical operations.
*   **`plotly.express` and `plotly.graph_objects`:** For creating interactive plots.
*   **`matplotlib.pyplot` and `seaborn`:** For creating static plots.
*   **`json`:** For saving the insights into a json file.
*   **`pandas.cut`**: For creating the age groups.
*   **`pandas.qcut`**: For creating the balance segments.
*   **`.groupby()` and `.agg()`**: To get aggregated data.
*   **`.corr()`**: To get correlations between columns.
*   **`px.imshow()`**: to show heatmaps.
*   **`px.pie()`**: to show pie charts.
*   **`px.box()`**: to show box plots.
*   **`px.scatter()`**: to show scatter plots.
*   **`px.bar()`**: to show bar plots.

## Recommended Analysis and How the Code Addresses Them

Here's how the code addresses the questions you asked:

1.  **What attributes are more common among churners than non-churners? Can churn be predicted using the variables in the data?**

    *   **Code:** The notebook explores churn by using box plots based on different types of categories (age, balance, etc), and displays the churn rate for each one. Also, it uses a correlation matrix that shows which features are related to the `Exited` feature. Finally, it creates segments of customers that can be used to see if there are groups of customers that churn more than others.
    *   **Answer:**
        *   **Churn Rate by Geography:** The code directly calculates the churn rate for each region, showing the rates for France, Germany and Spain. From what can be observed in the bar chart that Germany has a significantly higher churn rate (around 32%), while France and Spain have similar and lower values (around 16%).
        *   **Churn rate by Number of Products:** The code also calculates the churn rate for each number of products, showing that the more products the user has, the smaller is the churn rate, except when the user has 3 products, which shows higher churn (around 82%) than a user with 2 products (around 7%).  However, there are very few users with 3 and 4 products, which might affect the results.
        *   **Churn Rate by Age**: the distribution of the age in the churned vs not churned users shows that older users are more prone to churn, as their balance tends to be higher than the rest.

        *   **Churn Rate by Balance and Age Segment**: The treemap visualizes the churn rate of different segments. For example, a customer in their 50s that has a low balance, tends to have the highest rate of churn.
        *  **Correlation Matrix**: It's observable that `IsActiveMember` has a strong negative correlation with `Exited`, as well as age has a moderate correlation. While other features seem to have a very small correlation.
        *   **Prediction:** The notebook doesn't explicitly predict churn, it sets up the ground work for a model by exploring relevant features. Based on the analysis, variables like `Age`, `IsActiveMember`, `NumOfProducts` and `Balance` seem to be relevant in predicting churn.

2.  **What do the overall demographics of the bank's customers look like?**

    *   **Code:** The notebook uses various plots to look into the demographics of the bank's customers.
    *   **Answer:**
        *   **Age Distribution:** The code shows a visualization of the customers age distributions. Most customers seem to be between 30 and 60 years of age.
         *  **Active vs Inactive:** The code also shows the amount of active vs inactive members, using a pie chart. Around 52% of users are active, and 48% are inactive.

3.  **Is there a difference between German, French, and Spanish customers in terms of account behavior?**

    *   **Code:** The notebook includes charts and statistical insights for each geography.
    *   **Answer:**
        *   **Geographic Churn Analysis:** The notebook visualizes the churn rate of each country. Germany is a standout case with a higher churn rate than the other countries.
        *   **Balance Distribution:** The box plots visualize the distributions of the account balances for each country, although it does not go into further analysis.

4.  **What types of segments exist within the bank's customers?**

    *   **Code:** The notebook creates new columns, `BalanceSegment` and `AgeGroup`, then it creates a treemap to show how these segments are related to churn.
    *   **Answer:**
        *   **Age Segments:** The code creates age segments based on common groups.
        *   **Balance Segments:** The code creates segments of customers based on their balance.
        *   **Customer Segments:** The treemap visualizes the churn rate of the segments of users, combining `AgeGroup` and `BalanceSegment`. This way we can see which groups of users have the highest churn rate.

## Further Steps:

The notebook sets up the stage for future analysis and machine learning modeling.

*   **Feature engineering:** You could create new features based on the insights found in the data (e.g. a feature that measures the usage of different products).
*   **Data cleaning:** Implement more advanced cleaning techniques (e.g., imputations, handle outliers in different ways).
*   **Model building**: You can use the insights found to build a model that predicts churn using supervised machine learning techniques.

## How to run this project

1.   Make sure to have python and git installed.
2.  Clone this repository to your local computer.
3.  Navigate to the project's root directory in the command line.
4.  Create a virtual environment (you can use conda or venv) and activate it.
5.  Install requirements using `pip install -r requirements.txt` (you might need to create a `requirements.txt` file first).
6.  Copy the `Bank_Churn.csv` file in the `data/raw` folder.
7.  Run `jupyter notebook` in the command line to open the jupyter notebook. You can find the file in the `notebooks` folder.
8.   Alternatively, run `streamlit run streamlit_app.py` in the command line to open a local web app with some visualizations from the notebook.

## Tools used

<h3 align="left">Languages and Tools:</h3>
<p align="left">
  <a href="https://www.gnu.org/software/bash/" target="_blank" rel="noreferrer">
    <img src="https://www.vectorlogo.zone/logos/gnu_bash/gnu_bash-icon.svg" alt="bash" width="40" height="40"/>
  </a>
  <a href="https://git-scm.com/" target="_blank" rel="noreferrer">
    <img src="https://www.vectorlogo.zone/logos/git-scm/git-scm-icon.svg" alt="git" width="40" height="40"/>
  </a>
  <a href="https://opencv.org/" target="_blank" rel="noreferrer">
    <img src="https://www.vectorlogo.zone/logos/opencv/opencv-icon.svg" alt="opencv" width="40" height="40"/>
  </a>
  <a href="https://pandas.pydata.org/" target="_blank" rel="noreferrer">
    <img src="https://raw.githubusercontent.com/devicons/devicon/2ae2a900d2f041da66e950e4d48052658d850630/icons/pandas/pandas-original.svg" alt="pandas" width="40" height="40"/>
  </a>
  <a href="https://www.postgresql.org" target="_blank" rel="noreferrer">
    <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/postgresql/postgresql-original-wordmark.svg" alt="postgresql" width="40" height="40"/>
  </a>
  <a href="https://www.python.org" target="_blank" rel="noreferrer">
    <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/>
  </a>
    <a href="https://pytorch.org/" target="_blank" rel="noreferrer">
    <img src="https://www.vectorlogo.zone/logos/pytorch/pytorch-icon.svg" alt="pytorch" width="40" height="40"/>
  </a>
  <a href="https://scikit-learn.org/" target="_blank" rel="noreferrer">
     <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="scikit_learn" width="40" height="40"/>
  </a>
  <a href="https://seaborn.pydata.org/" target="_blank" rel="noreferrer">
    <img src="https://seaborn.pydata.org/_images/logo-mark-lightbg.svg" alt="seaborn" width="40" height="40"/>
  </a>
  <a href="https://www.tensorflow.org" target="_blank" rel="noreferrer">
    <img src="https://www.vectorlogo.zone/logos/tensorflow/tensorflow-icon.svg" alt="tensorflow" width="40" height="40"/>
  </a>
  <a href="https://plotly.com/" target="_blank" rel="noreferrer">
    <img src="https://plotly.com/static/images/plotly-icon.svg" alt="plotly" width="40" height="40"/>
  </a>
    <a href="https://streamlit.io/" target="_blank" rel="noreferrer">
    <img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="streamlit" width="40" height="40"/>
  </a>
     <a href="https://numpy.org/" target="_blank" rel="noreferrer">
    <img src="https://upload.wikimedia.org/wikipedia/commons/3/31/NumPy_logo_2020.svg" alt="numpy" width="40" height="40"/>
  </a>
     <a href="https://www.sqlite.org/" target="_blank" rel="noreferrer">
    <img src="https://www.sqlite.org/images/sqlite370x.gif" alt="sqlite" width="40" height="40"/>
  </a>
</p>

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         bank_churn_eda and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── bank_churn_eda   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes bank_churn_eda a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------


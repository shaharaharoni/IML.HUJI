from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


# removed the id column, date list
# price is y column
# condition, grade is categorical but has a relation
# assumption - grade is monotonically increasing ie higher grade means higher price
# zip code - group by intervals of 20
# lat and long - removed
# year of renovation - made it categorical with dummy values, grouped by decades

def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)
    df = df.dropna()
    dummies = pd.get_dummies(df['zipcode'], drop_first=True).add_prefix("zipcode")
    df = df.join(dummies)
    dummies = pd.get_dummies(df['yr_renovated'], drop_first=True).add_prefix("yr_renovated")
    df = df.join(dummies)
    y = df['price']
    df = df.drop(columns=['id', 'lat', 'long', 'date', 'zipcode', 'yr_renovated', 'price'])
    return df, y


def make_pearson_graph(X, feature_name, y, output_path):
    pearson_correlation = X[feature_name].cov(y) / (X[feature_name].std() * y.std())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X[feature_name], y=y, mode="markers"))
    fig.update_layout(
        title=f'Pearson Correlation between feature\n "{feature_name}" and response: {pearson_correlation}',
        xaxis_title="feature range", yaxis_title="response, prices USD")
    fig.write_image(output_path + '/' + feature_name + '.png')


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correl ation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for feature in X.columns:
        if feature.startswith("zipcode") or feature.startswith("yr_renovated") or feature == 'price':
            continue
        make_pearson_graph(X, feature, y, output_path)


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data('../datasets/house_prices.csv')
    # Question 2 - Feature evaluation with respect to response

    # feature_evaluation(X, y, 'pc')

    # Question 3 - Split samples into training- and testing sets.
    train_x, train_y, test_x, test_y = split_train_test(X, y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    linear_regression = LinearRegression()
    mean_loss = []
    variance_loss = []
    for p in range(10, 101):
        loss = []
        for _ in range(10):
            x_vals = train_x.sample(frac=(p / 100))
            y_vals = train_y.drop(train_y.drop(x_vals.index).index)
            linear_regression.fit(np.asarray(x_vals), np.asarray(y_vals))
            loss += [linear_regression.loss(np.asarray(test_x), np.asarray(test_y))]
        mean_loss += [np.mean(loss)]
        variance_loss += [np.var(loss)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[i / 100 for i in range(10, 101)], y=mean_loss))
    fig.update_layout(
        title="Mean loss as a function of p% of training set",
        xaxis_title="p fraction of training set", yaxis_title="mean loss")
    fig.show()

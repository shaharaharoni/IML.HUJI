import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=['Date'])
    df = df.dropna()
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df = df[df['Temp'] > -20]
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    X = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    X_Israel = X[X['Country'] == 'Israel']
    X_Israel = X_Israel[X_Israel['Temp'] > -20]

    X_Israel['Year'] = X_Israel['Year'].astype(str)
    fig = px.scatter(X_Israel, x=X_Israel['DayOfYear'], y=X_Israel['Temp'], color="Year")
    fig.update_layout(
        title='Temp to Day Of Year ratio',
        xaxis_title="Day of year range", yaxis_title="Temp (celsius)")
    fig.show()

    std_daily_temp = X_Israel.groupby(['Month'])['Temp'].agg(np.std)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[i for i in range(1, 13)], y=std_daily_temp))
    fig.update_layout(
        title='Standard Deviation of temperatures for each month',
        xaxis_title="Months", yaxis_title="Standard Deviation")
    fig.show()

    # Question 3 - Exploring differences between countries
    X_grouped = X.groupby(['Country', 'Month'])
    mean = X_grouped['Temp'].mean().reset_index()
    std = X_grouped['Temp'].std().reset_index()

    fig = go.Figure()
    months = [i for i in range(1, 13)]
    fig.add_trace(
        go.Scatter(x=months, y=mean['Temp'][0:12], name='Israel',
                   error_y=dict(type='data', array=std['Temp'][0:12])))
    fig.add_trace(go.Scatter(x=months, y=mean['Temp'][12:24], name='Jordan',
                             error_y=dict(type='data', array=std['Temp'][12:24])))
    fig.add_trace(go.Scatter(x=months, y=mean['Temp'][24:36], name='South Africa',
                             error_y=dict(type='data', array=std['Temp'][24:36])))
    fig.add_trace(go.Scatter(x=months, y=mean['Temp'][36:48], name='The Netherlands',
                             error_y=dict(type='data', array=std['Temp'][36:48])))
    fig.update_layout(title='Average Monthly Temperatures in 4 Countries', xaxis_title="Month",
                      yaxis_title="Temp (celsius)")
    fig.show()

    # Question 4 - Fitting model for different values of `k`
    y_Israel = X_Israel['Temp']
    X_Israel.drop(columns=['Temp'])
    train_x, train_y, test_x, test_y = split_train_test(X_Israel['DayOfYear'], y_Israel)
    losses = []

    for k in range(1, 11):
        poly_regressor = PolynomialFitting(k)
        poly_regressor.fit(np.asarray(train_x), np.asarray(train_y))
        losses.append(round(poly_regressor.loss(np.asarray(test_x), np.asarray(test_y)), 2))

    for k in range(1, 11):
        print("k value:", k, losses[k - 1])

    fig = go.Figure()
    fig.add_trace(go.Bar(x=[i for i in range(1, 11)], y=[val for val in losses]))
    fig.update_layout(title='Loss of the model', xaxis_title='k value', yaxis_title='loss of model')
    fig.show()

    # Question 5 - Evaluating fitted model on different countries
    poly_regressor = PolynomialFitting(5)
    poly_regressor.fit(X_Israel['DayOfYear'], y_Israel)

    countries = ['Israel', 'Jordan', 'The Netherlands', 'South Africa']
    country_loss = []

    for country in countries:
        country_loss.append(poly_regressor.loss(X[X.Country == country]['DayOfYear'], X[X.Country == country]['Temp']))

    fig = go.Figure()
    fig.add_trace(go.Bar(x=countries, y=country_loss))
    fig.update_layout(title="Model's error over countries, Israel fitted", xaxis_title='Country',
                      yaxis_title="Model's error")
    fig.show()

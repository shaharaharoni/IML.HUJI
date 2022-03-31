from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    estimator = UnivariateGaussian()
    mu, sigma = 10, 1

    # Question 1 - Draw samples and print fitted model
    print("q1")
    original_x = np.random.normal(mu, sigma, size=1000)
    estimator.fit(original_x)
    print(estimator.mu_, estimator.var_)

    # Question 2 - Empirically showing sample mean is consistent
    print("q2")
    estimations = []
    for size in range(10, 1000, 10):
        x = np.random.normal(mu, sigma, size=size)
        estimator.fit(x)
        estimations.append(abs(estimator.mu_ - mu))

    # plot graph
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[i for i in range(10, 1000, 10)], y=estimations))
    fig.update_layout(title="(Question 2) Difference between estimation and true value of expectations, by sample size",
                      xaxis_title="sample size", yaxis_title="difference between estimation and true expectation")
    fig.update_yaxes(range=[0, 0.4])
    fig.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    print("q3")
    estimator.fit(original_x)
    y = estimator.pdf(original_x)

    # plot graph
    fig = go.Figure(go.Scatter(x=original_x, y=y, mode='markers'))
    fig.update_layout(title="(Question 3) Empirical PDF under fitted model",
                      xaxis_title="Drawn Sample Value", yaxis_title="Sample PDF value")
    fig.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    print("q4")
    estimator = MultivariateGaussian()
    mu = [0, 0, 4, 0]
    cov = np.asarray([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    original_x = np.random.multivariate_normal(mu, cov, size=1000)
    estimator.fit(original_x)
    print(estimator.mu_)
    print(estimator.cov_)

    # Question 5 - Likelihood evaluation
    print("q5")
    size = 200
    f1_estimations = np.linspace(-10, 10, size)
    f3_estimations = np.linspace(-10, 10, size)
    log_likelihood_vals = []

    for i in range(size):
        row = []
        for j in range(size):
            row.append(MultivariateGaussian.log_likelihood(np.asarray([f1_estimations[i], 0, f3_estimations[j], 0]),
                                                           cov, original_x))
        log_likelihood_vals.append(row)

    # plot graph
    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=f3_estimations, y=f1_estimations, z=log_likelihood_vals,
                             xaxis='x', yaxis='y',
                             colorscale=[[0.0, "rgb(165,0,38)"],
                                         [0.1111111111111111, "rgb(215,48,39)"],
                                         [0.2222222222222222, "rgb(244,109,67)"],
                                         [0.3333333333333333, "rgb(253,174,97)"],
                                         [0.4444444444444444, "rgb(254,224,144)"],
                                         [0.5555555555555556, "rgb(224,243,248)"],
                                         [0.6666666666666666, "rgb(171,217,233)"],
                                         [0.7777777777777778, "rgb(116,173,209)"],
                                         [0.8888888888888888, "rgb(69,117,180)"],
                                         [1.0, "rgb(49,54,149)"]]))
    fig.update_layout(title="(Question 5) Log-Likelihood Heatmap of samples",
                      xaxis_title="f3 values", yaxis_title="f1 values")
    fig.show()

    # Question 6 - Maximum likelihood
    print("q6")
    row, col = np.unravel_index(np.argmax(log_likelihood_vals), np.asarray(log_likelihood_vals).shape)
    print("f1: ", round(f1_estimations[row], 3), "f3: ", round(f3_estimations[col], 3))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()

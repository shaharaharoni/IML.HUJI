from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from math import atan2, pi

pio.templates.default = "simple_white"


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.asarray(np.load(filename))
    y_true = data[:, -1]
    X = data[:, :-1]
    return X, y_true


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """

    for n, f in [("Linearly Separable", "../datasets/linearly_separable.npy"),
                 ("Linearly Inseparable", "../datasets/linearly_inseparable.npy")]:
        # Load dataset
        data = np.asarray(np.load(f))
        y_true = data[:, -1]
        X = data[:, :-1]

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def loss_callback(perceptron, x, y):
            loss = perceptron._loss(X, y_true)
            losses.append(loss)

        perceptron = Perceptron(callback=loss_callback)
        perceptron.fit(X, y_true)

        # Plot figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[i for i in range(1000)], y=losses))
        fig.update_layout(title=f"Perceptron's loss over the {n} dataset", xaxis_title='number of iterations',
                          yaxis_title='loss')
        fig.show()



def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix
    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse
    cov: ndarray of shape (2,2)
        Covariance of Gaussian
    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))
    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black"), \
           go.Scatter(x=np.array(mu[0]), y=np.array(mu[1]), mode='markers', marker=dict(symbol='x', color='black'))


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["../datasets/gaussian1.npy", "../datasets/gaussian2.npy"]:
        # Load dataset
        X, y_true = load_dataset(f)

        # Fit models and predict over training set
        models, preds, accur = [LDA(), GaussianNaiveBayes()], [], []
        from IMLearn.metrics import accuracy

        for i, m in enumerate(models):
            m.fit(X, y_true)
            preds.append(m.predict(X))
            accur.append(accuracy(y_true, preds[i]))

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        model_names = ['LDA', 'Gaussian Naive Bayes']
        symbols = np.array(["circle", 'star', 'triangle-up'])
        colors = np.array(['blue', 'green', 'red'])

        fig = make_subplots(rows=1, cols=2, subplot_titles=[rf"$\textbf{{{m}}} \ accuracy: {accur[i]}$" for i, m in
                                                            enumerate(model_names)],
                            horizontal_spacing=0.01, vertical_spacing=.03)
        for i, m in enumerate(models):
            fig.add_traces(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                                      marker=dict(color=colors[[int(j) for j in preds[i]]],
                                                  symbol=symbols[[int(n) for n in y_true]]), showlegend=False),
                           rows=(i // 3) + 1, cols=(i % 3) + 1)
            if not i:
                for cls in range(m.classes_.shape[0]):
                    fig.add_traces(get_ellipse(m.mu_[cls], m.cov_), rows=(i // 3) + 1, cols=(i % 3) + 1)
            else:
                for cls in range(m.classes_.shape[0]):
                    fig.add_traces(get_ellipse(m.mu_[cls], np.diag(m.vars_[cls])), rows=(i // 3) + 1, cols=(i % 3) + 1)

        fig.update_layout(title=rf"$\textbf{{Accuracy of Models {model_names} over {f} Dataset}}$",
                          margin=dict(t=100), showlegend=False)
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()

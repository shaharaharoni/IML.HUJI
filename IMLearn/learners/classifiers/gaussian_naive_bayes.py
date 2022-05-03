from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv
from IMLearn.metrics.loss_functions import misclassification_error as mis_err


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        m = X.shape[0]
        d = X.shape[1] if X.ndim > 1 else 1
        self.classes_ = np.unique(y)
        k = self.classes_.shape[0]
        self.mu_ = np.ndarray((m, d))
        self.pi_ = np.ndarray((k,))
        self.vars_ = np.ndarray((k, d))

        for i, cls in enumerate(self.classes_):
            self.mu_[i] = np.mean(X[y == cls], axis=0)
            self.pi_[i] = len(X[y == cls]) / m
            self.vars_[i] = np.var(X[y == cls], axis=0, ddof=1)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        likelihood = self.likelihood(X)
        return np.take(self.classes_, np.argmax(likelihood, axis=1))

    def find_probability(self, X, i, j):
        cov = np.diag(self.vars_[j])
        inv_cov = inv(cov)
        det_cov = det(cov)
        diff = X[i] - self.mu_[j]
        d = X.shape[1]
        return np.log(self.pi_[j]) - np.log(np.sqrt(((2 * np.pi) ** d) * det_cov)) - 0.5 * diff.T @ inv_cov @ diff

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        likelihood_mat = np.ndarray((X.shape[0], self.classes_.shape[0]))
        for sample in range(X.shape[0]):
            for cls in range(self.classes_.shape[0]):
                likelihood_mat[sample][cls] = self.find_probability(X, sample, cls)
        return likelihood_mat

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        y_hat = self.predict(X)
        return mis_err(y, y_hat)

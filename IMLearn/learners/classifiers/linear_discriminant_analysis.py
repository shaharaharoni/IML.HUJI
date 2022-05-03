from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv
from IMLearn.metrics.loss_functions import misclassification_error as mis_err


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `LDA.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        self.mu_ = np.ndarray((self.classes_.shape[0], X.shape[1]))
        self.cov_ = np.zeros((X.shape[1], X.shape[1]))
        self.pi_ = np.zeros((self.classes_.shape[0],))
        cov = np.zeros((X.shape[1], X.shape[1]))

        for i, cls in enumerate(self.classes_):
            label_ind = [i for i in range(len(y)) if y[i] == cls]
            mean_label = np.mean(X[label_ind, :], axis=0)
            self.mu_[i] = mean_label

            cov += np.sum(np.outer(X[ind] - mean_label, X[ind] - mean_label) / y.shape[0] for ind in label_ind)
            self.pi_[i] = len(label_ind) / y.shape[0]

        self.cov_ = cov
        self._cov_inv = inv(self.cov_)

    def find_probability(self, X, i, j):
        return np.log(self.pi_[j]) + (X[i] @ self._cov_inv @ np.transpose(self.mu_[j])) - 0.5 * \
               self.mu_[j] @ self._cov_inv @ np.transpose(self.mu_[j])

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

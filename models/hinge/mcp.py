from sklearn.metrics import (
    log_loss,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    balanced_accuracy_score,
)
from venn_abers import VennAbers
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import warnings

# Suprimir o aviso específico
warnings.filterwarnings("ignore", category=RuntimeWarning, module="venn_abers")


class WrapperOOBBinaryConformalClassifier:
    """
    A modrian class conditional conformal classifier based on Out-of-Bag (OOB) methodology, utilizing a random forest classifier as the underlying learner.
    This class is inspired by the WrapperClassifier classes from the Crepes library.
    """

    def __init__(self, learner: RandomForestClassifier):
        """
        Constructs the classifier with a specified learner and a Venn-Abers calibration layer.

        Parameters:
        learner: RandomForestClassifier
            The base learner to be used in the classifier.

        Attributes:
        learner: RandomForestClassifier
            The base learner employed in the classifier.
        calibration_layer: VennAbers
            The calibration layer utilized in the classifier.
        feature_importances_: array-like of shape (n_features,)
            The feature importances derived from the learner.
        alphas: array-like of shape (n_samples,), default=None
            Nonconformity measure based at the difference between the predicted probability
            of most likely incorrent class label and the predicted probability of the true label.
            Close to zero or negative margin indicates confidence in the true class label, while a large positive margin signals
            confidence in an incorrect class, suggesting unreliable predictions. The mergin measures the risk level of the model's prediction.
        alpha: float, default=0.05
            The significance level applied in the classifier.
        """

        # Ensure the learner is fitted
        check_is_fitted(learner)

        if learner.n_classes_ > 2:
            raise ("Learner has more than 2 labels.")

        # Initialize attributes
        self.learner = learner
        self.calibration_layer = VennAbers()
        self.feature_importances_ = self.learner.feature_importances_
        self.hinge = None
        self.alpha = 0.05
        self.n = None
        self.classes = None
        self.y = None

    def fit(self, y):
        """
        Fits the classifier to the training data. Calculates the conformity score for each training instance.

        Parameters:
        y: array-like of shape (n_samples,)
            The true labels.

        Returns:
        self: object
            Returns self.

        The function works as follows:
        - It first gets the out-of-bag probability predictions from the learner.
        - It then fits the calibration layer to these predictions and the true labels.
        - It computes the probability for each instance.
        - It finally turns these probabilities into non-conformity measure.
        """

        # Get the probability predictions
        y_prob = self.learner.oob_decision_function_

        self.calibration_layer.fit(y_prob, y)
        y_prob, _ = self.calibration_layer.predict_proba(y_prob)
        # We only need the probability for the true class
        self.n = len(self.learner.oob_decision_function_)

        y_prob = y_prob[np.arange(self.n), y]

        hinge = self.generate_non_conformity_score(y_prob)
        self.classes = self.learner.classes_

        # We only need the probability for the true class
        self.hinge = [hinge[y == c] for c in self.classes]
        self.y = y

        return self

    def predict_proba(self, X):
        """
        Predicts the class probabilities for the instances in X.

        Parameters:
        X: array-like of shape (n_samples, n_features)
            The input samples.

        Returns:
        p_prime: array-like of shape (n_samples, n_classes)
            The calibrated class probabilities.
        """

        y_score = self.learner.predict_proba(X)
        p_prime, _ = self.calibration_layer.predict_proba(y_score)
        return p_prime

    def predict(self, X, alpha=None):
        """
        Predicts the classes for the instances in X.

        Parameters:
        X: array-like of shape (n_samples, n_features)
            The input samples.

        Returns:
        predictions: array-like of shape (n_samples,)
            A predicted true class if the model has certainty based on the predefined significance level.
        """

        if alpha is None:
            alpha = self.alpha

        y_pred = self.predict_set(X, alpha)

        return np.where(np.all(y_pred == [0, 1], axis=1), 1, 0)

    def generate_conformal_quantile(self, alpha=None):
        """
        Generates the conformal quantile for conformal prediction.

        This function calculates the conformal quantile based on the non-conformity scores
        of the true label probabilities. The quantile is used as a threshold
        to determine the prediction set in conformal prediction.

        Parameters:
        -----------
        alpha : float, optional
            The significance level for conformal prediction. If None, uses the value
            of self.alpha.

        Returns:
        --------
        float
            The calculated conformal quantile.

        Notes:
        ------
        - The quantile is calculated as the (n+1)*(1-alpha)/n percentile of the non-conformity
          scores, where n is the number of calibration samples.
        - This method uses the self.hinge attribute, which should contain the non-conformity
          scores of the calibration samples.

        """

        if alpha is None:
            alpha = self.alpha

        qhat = np.zeros(len(self.classes))

        q_level = np.ceil((self.n + 1) * (1 - alpha)) / self.n

        for c in self.classes:
            qhat[c] = np.quantile(self.hinge[c], q_level, method="higher")

        return qhat

    def generate_non_conformity_score(self, y_prob):
        """
        Generates the non-conformity score based on the hinge loss.

        This function calculates the non-conformity score for conformal prediction
        using the hinge loss approach.

        Parameters:
        -----------
        y_prob : array-like of shape (n_samples,) or (n_samples, n_classes)
            The predicted probabilities for each class.

        Returns:
        --------
        array-like
            The non-conformity scores, where higher values indicate greater
            non-conformity.

        Notes:
        ------
        - This implementation assumes that y_prob contains probabilities and
          not raw model outputs.

        """
        return 1 - y_prob

    def predict_set(self, X, alpha=None):
        """
        Predicts the possible set of classes for the instances in X based on the predefined significance level.

        Parameters:
        X: array-like of shape (n_samples, n_features)
            The input samples.
        alpha: float, default=None
            The significance level. If None, the value of self.alpha is used.

        Returns:
        prediction_set: array-like of shape (n_samples, n_classes)
            The predicted set of classes. A class is included in the set if its non-conformity score is less
            than or equal to the quantile of the hinge loss distribution at the (n+1)*(1-alpha)/n level.
        """

        if alpha is None:
            alpha = self.alpha

        prediction_set = np.zeros((len(X), len(self.classes)))
        y_prob = self.predict_proba(X)
        nc_score = self.generate_non_conformity_score(y_prob)
        qhat = self.generate_conformal_quantile(alpha)

        for c in self.classes:
            prediction_set[:, c] = (nc_score <= qhat[c])[:, c]

        return prediction_set.astype(int)

    def _expected_calibration_error(self, y, y_prob, M=5):
        """
        Calculates the expected calibration error (ECE) of the classifier.

        Parameters:
        y: array-like of shape (n_samples,)
            The true labels.
        y_prob: array-like of shape (n_samples, n_classes)
            The predicted probabilities.
        M: int, default=5
            The number of bins for the uniform binning approach.

        Returns:
        ece: float
            The expected calibration error.

        The function works as follows:
        - It first creates M bins with uniform width over the interval [0, 1].
        - For each sample, it computes the maximum predicted probability and makes a prediction.
        - It then checks whether each prediction is correct or not.
        - For each bin, it calculates the empirical probability of a sample falling into the bin.
        - If the empirical probability is greater than 0, it computes the accuracy and average confidence of the bin.
        - It then calculates the absolute difference between the accuracy and the average confidence, multiplies it by the empirical probability, and adds it to the total ECE.
        """

        # uniform binning approach with M number of bins
        bin_boundaries = np.linspace(0, 1, M + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        # get max probability per sample i
        confidences = np.max(y_prob, axis=1)
        # get predictions from confidences (positional in this case)
        predicted_label = np.argmax(y_prob, axis=1)

        # get a boolean list of correct/false predictions
        predictions = predicted_label == y

        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # determine if sample is in bin m (between bin lower & upper)
            in_bin = np.logical_and(
                confidences > bin_lower.item(), confidences <= bin_upper.item()
            )
            # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
            prob_in_bin = np.mean(in_bin)

            if prob_in_bin > 0:
                # get the accuracy of bin m: acc(Bm)
                avg_pred = np.mean(predictions[in_bin])
                # get the average confidence of bin m: conf(Bm)
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
                ece += np.abs(avg_pred - avg_confidence_in_bin) * prob_in_bin
        return ece

    def _empirical_coverage(self, X, alpha=None, iterations=100):
        """
        Generate the empirical coverage of the classifier.

        Parameters:
        X: array-like of shape (n_samples, n_features)
            The input samples.
        alpha: float, default=None
            The significance level. If None, the value of self.alpha is used.
        iterations: int, default=100
            The number of iterations for the empirical coverage calculation.

        Returns:
        average_coverage: float
            The average coverage over the iterations. It should be close to 1-alpha.
        """

        if alpha is None:
            alpha = self.alpha

        coverages = np.zeros((iterations,))
        y_prob = self.predict_proba(X)
        scores = 1 - y_prob
        n = int(len(scores) * 0.20)
        classes = [0, 1]

        for i in range(iterations):
            np.random.shuffle(scores)  # shuffle
            calib_scores, val_scores = (scores[:n], scores[n:])  # split
            q_level = np.ceil((n + 1) * (1 - alpha)) / n
            prediction_set = np.zeros((len(val_scores), len(self.classes)))

            for c in classes:
                qhat = np.quantile(calib_scores[:, [c]], q_level, method="higher")
                prediction_set[:, c] = (val_scores <= qhat)[:, c]

            coverages[i] = prediction_set.astype(float).mean()  # see caption
            average_coverage = coverages.mean()  # should be close to 1-alpha

        return average_coverage

    def calibrate_alpha(self, X, y):
        """
        Calibrates the alpha value to minimize the error rate
        using Cost Sensitive Learning methodology, using balanced_accuracy_score.

        Parameters:
        X: array-like of shape (n_samples, n_features)
            The test input samples.
        y: array-like of shape (n_samples,)
            The true labels for X.

        For each alpha value (0.10, 0.09, …, 0.01), we do the following:
        - Calculate predictions y_pred using the self.predict(X, alpha) function.

        Returns:
            The updated instance (self.alpha) with the calibrated alpha value.
        """
        alphas = {
            round(
                k,
                2,
            ): None
            for k in np.linspace(0.01, 0.10, 10)
        }

        for alpha in alphas:
            y_pred = self.predict(X, alpha)
            alphas[alpha] = balanced_accuracy_score(y, y_pred)

        self.alpha = max(alphas, key=alphas.get)

        return self

    def _evaluate_generalization(self, X, y, alpha=None):
        """
        Measure the generalization gap of the model.

        The generalization gap indicates how well the model generalizes
        to unseen data. It is calculated as the difference between the
        error on the training set and the error on the test set.

        Parameters:
        X (array-like): Features of the test set
        y (array-like): Labels of the test set
        alpha (float, optional): Significance level for conformal prediction.
                                 If None, uses the default value.

        Returns:
        float: The generalization gap

        """

        if alpha is None:
            alpha = self.alpha

        nc_score = self.generate_non_conformity_score(
            self.learner.oob_decision_function_
        )

        qhat = self.generate_conformal_quantile(alpha)

        prediction_set = np.zeros((len(nc_score), len(self.classes)))

        for c in self.classes:
            prediction_set[:, c] = (nc_score <= qhat[c])[:, c]

        y_pred = np.where(np.all(prediction_set.astype(int) == [0, 1], axis=1), 1, 0)

        training_error = 1 - balanced_accuracy_score(y_pred, self.y)
        test_error = 1 - balanced_accuracy_score(self.predict(X, alpha), y)
        return training_error - test_error

    def evaluate(self, X, y, alpha=None):
        """
        Evaluates the performance of the conformal classifier on the given test data and labels.

        Parameters:
        X: array-like of shape (n_samples, n_features)
            The test input samples.
        y: array-like of shape (n_samples,)
            The true labels for X.
        alpha: float, default=None
            The significance level. If None, the value of self.alpha is used.

        Returns:
        results: dict
            A dictionary with the following keys and their corresponding values:
            - "one_c": The proportion of instances where the prediction set contains exactly one class.
            - "avg_c": The average size of the prediction set.
            - "empty": The proportion of instances where the prediction set is empty.
            - "error": The error rate, i.e., the proportion of instances where the true label is not in the prediction set.
            - "log_loss": The log loss.
            - "brier_loss": The Brier score loss.
            - "ece": The expected calibration error.
            - "empirical_coverage": The empirical coverage.
        """

        if alpha is None:
            alpha = self.alpha

        results = {}
        y_prob = self.predict_proba(X)
        predict_set = self.predict_set(X, alpha)
        n_digits = 3
        results["one_c"] = round(
            np.sum([np.sum(p) == 1 for p in predict_set]) / len(y), n_digits
        )
        results["avg_c"] = round(np.sum(predict_set) / len(predict_set), n_digits)
        results["empty"] = round(
            np.sum([np.sum(p) == 0 for p in predict_set]) / len(y), n_digits
        )
        results["error"] = round(
            1 - np.sum(predict_set[np.arange(len(y)), y]) / len(y), n_digits
        )
        results["log_loss"] = round(log_loss(y, y_prob[:, 1]), n_digits)
        results["brier_loss"] = round(brier_score_loss(y, y_prob[:, 1]), n_digits)
        results["ece"] = round(self._expected_calibration_error(y, y_prob), n_digits)
        results["empirical_coverage"] = round(
            self._empirical_coverage(X, alpha), n_digits
        )
        results["generalization"] = round(
            self._evaluate_generalization(X, y, alpha), n_digits
        )
        results["auc"] = round(roc_auc_score(y, self.predict_proba(X)[:, 1]), n_digits)
        results["precision"] = round(
            precision_score(y, self.predict(X, alpha)), n_digits
        )
        results["recall"] = round(recall_score(y, self.predict(X, alpha)), n_digits)
        results["alpha"] = alpha

        return pd.DataFrame([results])

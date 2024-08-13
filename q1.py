from .utils import get_data, plot, announce
from .consts import category_dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Q1
def linear_classification(portion=1.):
    """
    Perform linear classification
    :param portion: portion of the data to use
    :return: classification accuracy
    """
    tf = TfidfVectorizer(stop_words='english', max_features=1000)
    x_train, y_train, x_test, y_test = get_data(categories=category_dict.keys(), portion=portion)

    x_train_tf = tf.fit_transform(x_train)
    x_test_tf = tf.transform(x_test)
    model = LogisticRegression()
    model.fit(x_train_tf, y_train)
    y_pred = model.predict(x_test_tf)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


@announce
def q1(portions: list[float]) -> None:
    print("Logistic regression results:")
    logistic_regression_accuracies = []
    for p in portions:
        accuracy = linear_classification(p)
        print(f"\tPortion: {p * 100}%, Accuracy : {accuracy * 100:.2f}%")
        logistic_regression_accuracies.append(accuracy)
    plot(accuracies=logistic_regression_accuracies, portions=portions, model_name="Logistic_Regression")


__all__ = [
    "linear_classification",
    "q1"
]

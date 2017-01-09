import warnings
import joblib
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import classification_report
from sklearn import grid_search
from pprint import pprint

"""
LinerSVC GridSearch 済み．

    # classifier = LinearSVC(C=10, class_weight=None, dual=False, fit_intercept=True,
    #                        intercept_scaling=1, loss='squared_hinge', max_iter=1000,
    #                        multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
    #                        verbose=0)

             precision    recall  f1-score   support

       True       0.70      0.89      0.78      1193
      False       0.54      0.26      0.35       621

avg / total       0.64      0.67      0.63      1814

array([[-2.47778237, -0.73150047, -0.63949442,  0.11491836,  0.53256585,
         0.39947692,  0.06811644, -0.41393095,  0.11590859,  0.15695755]])
array([ 1.14677486])
1
"""

if __name__ == '__main__':
    # 警告を消す．
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    unit_vector_dict = joblib.load("train_data/unit_vectors.dict")
    """:type : dict"""

    all_vector = [value for value in unit_vector_dict.values()]

    all_labels = []
    for key in unit_vector_dict.keys():
        if "true" in key:
            all_labels.append(1)
        else:
            all_labels.append(2)

    train_data, test_data, train_label, test_label = train_test_split(
        all_vector, all_labels)

    # param_grid = [
    #     {"C": [1, 10, 50, 100, 500, 1000],
    #      "kernel": ["rbf", "poly", "sigmoid"], "gamma": ["auto", 1e-2, 1e-3, 1e-4]},
    #     # {"C": [1, 10, 50, 100, 500, 1000],
    #     #  "kernel": ["poly"], "gamma": ["auto", 1e-2, 1e-3, 1e-4]},
    #     # {"C": [1, 10, 50, 100, 500, 1000],
    #     #  "kernel": ["p"], "gamma": ["auto", 1e-2, 1e-3, 1e-4]},
    # ]
    # scores = ["f1"]
    #
    # for score in scores:
    #     print(" ".join(["\n", "*" * 10, score, "*" * 10, "\n", ]))
    #
    #     clf = grid_search.GridSearchCV(SVC(), param_grid=param_grid, cv=5, scoring=score, n_jobs=3)
    #     clf.fit(train_data, train_label)
    #
    #     print("\n::bast param::\n")
    #     print(clf.best_estimator_)
    #
    #     print("\n:mean score:\n")
    #     for params, mean_score, all_scores in clf.grid_scores_:
    #         print("{:.3f} (+/- {:.3f}) for {}".format(mean_score, all_scores.std() / 2, params))
    #
    #     print("\n::test result::\n")
    #     y_true, y_pred = test_label, clf.predict(test_data)
    #     print(classification_report(y_true, y_pred))

    # classifier = SVC(C=500, cache_size=200, class_weight=None, coef0=0.0,
    #                  decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    #                  max_iter=-1, probability=False, random_state=None, shrinking=True,
    #                  tol=0.001, verbose=False)

    classifier = LinearSVC(C=10, class_weight=None, dual=False, fit_intercept=True,
                           intercept_scaling=1, loss='squared_hinge', max_iter=1000,
                           multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
                           verbose=0)
    classifier.fit(train_data, train_label)

    predict_label = classifier.predict(test_data)
    target_label = ["1_true", "2_false"]

    print(classification_report(test_label, predict_label, target_names=target_label))

    pprint(classifier.coef_)

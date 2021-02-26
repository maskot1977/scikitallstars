import numpy as np
import matplotlib.pyplot as plt

def training_summary(objective):
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 8))

    names = [n for n in reversed(list(objective.get_model_names()))]

    score_means = []
    score_stds = []
    second_means = []
    second_stds = []
    selected = []
    sum_second = []
    for name in names:
        score_means.append(np.array(objective.scores[name]).mean())
        score_stds.append(np.array(objective.scores[name]).std())
        second_means.append(np.array(objective.times[name]).mean())
        second_stds.append(np.array(objective.times[name]).std())
        selected.append(len(objective.times[name]))
        sum_second.append(sum(objective.times[name]))

    axes[0].barh(names, score_means, xerr=score_stds)
    axes[0].set_xlabel("score")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].grid()
    axes[1].barh(names, selected)
    axes[1].set_xlabel("selected (times)")
    axes[1].grid()
    axes[1].yaxis.set_visible(False)
    axes[2].barh(names, second_means, xerr=second_stds)
    axes[2].set_xlabel("calculation time (seconds)")
    axes[2].grid()
    axes[2].yaxis.set_visible(False)
    axes[3].barh(names, sum_second)
    axes[3].set_xlabel("total calculation time (seconds)")
    axes[3].grid()
    axes[3].yaxis.set_visible(False)
    plt.show()
    

def feature_importances(allstars_model):
  barh_dict = {}
  for key, value in zip(
      list(allstars_model.x_train.iloc[:, allstars_model.support].columns),
      allstars_model.best_models["RandomForest"].model.feature_importances_
  ):
      barh_dict[key] = value

  keys = list(barh_dict.keys())
  values = barh_dict.values()

  plt.figure(figsize=(6, int(len(keys)/3)))
  plt.title("Feature importances in RF")
  plt.barh(keys, values)
  plt.grid()
  plt.show()
  
  
def stacking_model_contribution(stacking_model):
  plt.title("Stacking model contribution")
  plt.barh(list(stacking_model.named_estimators_.keys()), stacking_model.final_estimator_.feature_importances_)
  plt.grid()
  plt.show()

def allsklearn_classification_metrics(objective, X_test, y_test):
    fig, axes = plt.subplots(
        nrows=3,
        ncols=len(objective.best_models.keys()),
        figsize=(4 * len(objective.best_models.keys()), 4 * 3),
    )
    i = 0
    for name in objective.best_models.keys():
        model = objective.best_models[name]
        if hasattr(model.model, "predict_proba"):
            probas = model.predict_proba(X_test)
        else:
            probas = np.array([[x, x] for x in model.model.decision_function(X_test)])

        fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])
        roc_auc = auc(fpr, tpr)
        precision, recall, thresholds = precision_recall_curve(y_test, probas[:, 1])
        area = auc(recall, precision)
        matrix = confusion_matrix(model.predict(X_test), y_test)
        TN = matrix[0][0]
        FP = matrix[1][0]
        FN = matrix[0][1]
        TP = matrix[1][1]
        data = [TP, FN, FP, TN]
        axes[0][i].set_title(name)
        axes[0][i].pie(
            data,
            counterclock=False,
            startangle=90,
            autopct=lambda x: "{}".format(int(x * sum(data) / 100)),
            labels=["TP", "FN", "FP", "TN"],
            wedgeprops=dict(width=1, edgecolor="w"),
            colors=["skyblue", "orange", "tan", "lime"],
        )
        axes[0][i].text(
            1.0 - 0.5,
            0.0 + 0.7,
            ("%.3f" % ((TN + TP) / (TN + TP + FN + FP))).lstrip("0"),
            size=20,
            horizontalalignment="right",
        )
        axes[1][i].plot([0, 1], [0, 1])
        axes[1][i].plot(fpr, tpr, label="ROC curve (area = %0.2f)" % roc_auc)
        axes[1][i].fill_between(fpr, tpr, alpha=0.5)
        axes[1][i].set_xlim([0.0, 1.0])
        axes[1][i].set_ylim([0.0, 1.0])
        axes[1][i].set_xlabel("False Positive Rate")
        if i == 0:
            axes[1][i].set_ylabel("True Positive Rate")
        axes[1][i].text(
            1.0 - 0.3,
            0.0 + 0.3,
            ("%.3f" % roc_auc).lstrip("0"),
            size=20,
            horizontalalignment="right",
        )
        axes[2][i].plot(recall, precision, label="Precision-Recall curve")
        axes[2][i].fill_between(recall, precision, alpha=0.5)
        axes[2][i].set_xlabel("Recall")
        if i == 0:
            axes[2][i].set_ylabel("Precision")
        axes[2][i].set_xlim([0.0, 1.0])
        axes[2][i].set_ylim([0.0, 1.0])
        axes[2][i].text(
            1.0 - 0.3,
            0.0 + 0.3,
            ("%.3f" % area).lstrip("0"),
            size=20,
            horizontalalignment="right",
        )
        i += 1
    plt.show()


def allsklearn_y_y_plot(objective, X_test, y_test):
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(objective.best_models.keys()),
        figsize=(4 * len(objective.best_models.keys()), 4),
    )
    i = 0
    for name in objective.best_models.keys():
        y_pred = objective.best_models[name].predict(X_test)
        score = r2_score(np.array(y_pred).ravel(), np.array(y_test).ravel())
        axes[i].set_title(name)
        axes[i].scatter(y_test, y_pred, alpha=0.5)
        y_min = min(y_test.min(), y_pred.min())
        y_max = min(y_test.max(), y_pred.max())
        axes[i].plot([y_min, y_max], [y_min, y_max])
        axes[i].text(
            y_max - 0.3,
            y_min + 0.3,
            ("%.3f" % score).lstrip("0"),
            size=15,
            horizontalalignment="right",
        )
        axes[i].set_xlabel("Real")
        if i == 0:
            axes[i].set_ylabel("Predicted")
        i += 1
    plt.show()


def show_allsklearn_metrics(objective, X_test, y_test):
    if objective.is_regressor:
        allsklearn_y_y_plot(objective, X_test, y_test)
    else:
        allsklearn_classification_metrics(objective, X_test, y_test)


def show_metrics(model, X_train, y_train, X_test, y_test):
    if hasattr(model, "predict_proba") or hasattr(model, "decision_function"):
        classification_metrics(model, X_train, y_train, X_test, y_test)
    else:
        y_y_plot(model, X_train, y_train, X_test, y_test)


def classification_metrics(model, X_train, X_test, y_train, y_test):
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(4 * 2, 4 * 3))
    i = 0
    for XX, YY, name in [
        [X_train, y_train, "Training data"],
        [X_test, y_test, "Test data"],
    ]:
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(XX)
        else:
            probas = np.array([[x, x] for x in model.decision_function(XX)])

        fpr, tpr, thresholds = roc_curve(YY, probas[:, 1])
        roc_auc = auc(fpr, tpr)
        precision, recall, thresholds = precision_recall_curve(YY, probas[:, 1])
        area = auc(recall, precision)
        matrix = confusion_matrix(model.predict(XX), YY)
        TN = matrix[0][0]
        FP = matrix[1][0]
        FN = matrix[0][1]
        TP = matrix[1][1]
        data = [TP, FN, FP, TN]
        axes[0][i].set_title(name)
        axes[0][i].pie(
            data,
            counterclock=False,
            startangle=90,
            autopct=lambda x: "{}".format(int(x * sum(data) / 100)),
            labels=["TP", "FN", "FP", "TN"],
            wedgeprops=dict(width=1, edgecolor="w"),
            colors=["skyblue", "orange", "tan", "lime"],
        )
        axes[0][i].text(
            1.0 - 0.5,
            0.0 + 0.7,
            ("%.3f" % ((TN + TP) / (TN + TP + FN + FP))).lstrip("0"),
            size=20,
            horizontalalignment="right",
        )
        axes[1][i].plot([0, 1], [0, 1])
        axes[1][i].plot(fpr, tpr, label="ROC curve (area = %0.2f)" % roc_auc)
        axes[1][i].fill_between(fpr, tpr, alpha=0.5)
        axes[1][i].set_xlim([0.0, 1.0])
        axes[1][i].set_ylim([0.0, 1.0])
        axes[1][i].set_xlabel("False Positive Rate")
        if i == 0:
            axes[1][i].set_ylabel("True Positive Rate")
        axes[1][i].text(
            1.0 - 0.3,
            0.0 + 0.3,
            ("%.3f" % roc_auc).lstrip("0"),
            size=20,
            horizontalalignment="right",
        )
        axes[2][i].plot(recall, precision, label="Precision-Recall curve")
        axes[2][i].fill_between(recall, precision, alpha=0.5)
        axes[2][i].set_xlabel("Recall")
        if i == 0:
            axes[2][i].set_ylabel("Precision")
        axes[2][i].set_xlim([0.0, 1.0])
        axes[2][i].set_ylim([0.0, 1.0])
        axes[2][i].text(
            1.0 - 0.3,
            0.0 + 0.3,
            ("%.3f" % area).lstrip("0"),
            size=20,
            horizontalalignment="right",
        )
        i += 1
    plt.show()


def y_y_plot(model, X_train, X_test, y_train, y_test):

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    y_pred = model.predict(X_train)
    score = model.score(X_train, y_train)
    y_min = min(y_train.min(), y_pred.min())
    y_max = min(y_train.max(), y_pred.max())

    axes[0].set_title("Training data")
    axes[0].scatter(y_train, y_pred, alpha=0.5)
    axes[0].plot([y_min, y_max], [y_min, y_max])
    axes[0].text(
        y_max - 0.3,
        y_min + 0.3,
        ("%.3f" % score).lstrip("0"),
        size=15,
        horizontalalignment="right",
    )
    axes[0].set_xlabel("Real")
    axes[0].set_ylabel("Predicted")

    y_pred = model.predict(X_test)
    score = model.score(X_test, y_test)
    y_min = min(y_test.min(), y_pred.min())
    y_max = min(y_test.max(), y_pred.max())

    axes[1].set_title("Test data")
    axes[1].scatter(y_test, y_pred, alpha=0.5)
    axes[1].plot([y_min, y_max], [y_min, y_max])
    axes[1].text(
        y_max - 0.3,
        y_min + 0.3,
        ("%.3f" % score).lstrip("0"),
        size=15,
        horizontalalignment="right",
    )
    axes[1].set_xlabel("Real")
    axes[1].set_ylabel("Predicted")
    plt.show()

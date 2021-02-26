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

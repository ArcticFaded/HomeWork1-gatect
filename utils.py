from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
from sklearn.tree._tree import TREE_LEAF
from sklearn.base import clone
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')
def post_pruning(tree, index, prune_value):
    if tree.value[index].min() < prune_value:
        # set these nodes to default nodes, will still exist in memory but wont be used for predicting values
        tree.children_left[index] = TREE_LEAF
        tree.children_right[index] = TREE_LEAF
    
    # continue traversing until each node is tested
    if tree.children_left[index] != TREE_LEAF:
        post_pruning(tree, tree.children_left[index], prune_value)
        post_pruning(tree, tree.children_right[index], prune_value)
        
def get_prune(clf, X_train, y_train, X_test, y_test):
    max_score, max_prune = 0, 0
    tolerable_prune = 0
    for prune_value in [1, 3, 5, 10, 15, 20, 25, 30, 40, 50, 100]:
        descision_tree = clone(clf)
        descision_tree.fit(X_train, y_train)

        pre_score = classification_report(descision_tree.predict(X_test) , y_test, output_dict=True)['weighted avg']['f1-score']

        post_pruning(descision_tree.tree_, 0, prune_value)

        post_score = classification_report(descision_tree.predict(X_test) , y_test, output_dict=True)['weighted avg']['f1-score']

        if((pre_score - post_score) < 0.01): # tolerance
            if (post_score > max_score): # sometimes pruning helps with overfitting
                max_prune = prune_value
                max_score = post_score
            tolerable_prune = prune_value

    return max_prune, tolerable_prune

def classification_accuracy(SVC, DecisionTree, BoostedTrees, KNN, ANN, X_train, y_train, X_test, y_test):
    
    prediction = {
        'SVC': [], # a list of predictions
        'DecisionTree': [],
        'BoostedTrees': [],
        'K-NearestNeighbors': [],
        'NeuralNetworks': []
    }
    
    prediction['SVC'] = SVC.predict(X_test)
    prediction['DecisionTree'] = DecisionTree.predict(X_test)
    prediction['BoostedTrees'] = BoostedTrees.predict(X_test)
    prediction['K-NearestNeighbors'] = KNN.predict(X_test)
    prediction['NeuralNetworks'] = ANN.predict(X_test)
    
    for predict in prediction:
        print('{classifer} \n {report}'.format(
            classifer=predict, 
            report=classification_report(np.round(prediction[predict]), y_test)
        ))

def plot_tsne(xy, colors=None, alpha=0.25, figsize=(6,6), s=0.5, cmap='hsv', filename='tsne.png'):
    plt.figure(figsize=figsize)
    plt.margins(0)
    
    fig = plt.scatter(xy[:, 0], xy[:, 1], c=colors, cmap=cmap, alpha=alpha, marker='o', s=s, lw=0, edgecolors='')
    plt.savefig(filename)


def plot_corr(df, size=10, filename='corr.png'):
    names = df.columns
    corr = df.corr()
    fig = plt.figure(figsize=(4*size, 3*size))
    ax = fig.add_subplot(111)
    
    cax = ax.matshow(corr, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, len(names), 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    plt.savefig(filename)

    
def print_report(svc_report, dt_report, bt_report, knn_report, ANN_report):
    names = {
        'SVC': svc_report,
        'Decision Tree': dt_report,
        'Boosted Trees': bt_report,
        'KNearestNeighbors': knn_report,
        'NeuralNetwork': ANN_report
    }
    
    for report in names:
        print(report)
        means = names[report]['mean_test_score']
        stds = names[report]['std_test_score']
        params = names[report]['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
            
def learning_curves(svc, dt, bt, knn, ANN, best_fit_ANN, X, y, filename):
    names = {
        'SVC': svc,
        'Decision Tree': dt,
        'Boosted Trees': bt,
        'KNearestNeighbors': knn,
    }
    
    for report in names:
        plt.figure()
        plt.title(report)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes=np.linspace(.6, 1.0, 5)
                    
        train_sizes, train_scores, test_scores = learning_curve(
            names[report], X, y, cv=3, n_jobs=-1, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                            train_scores_mean + train_scores_std, alpha=0.1,
                            color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                            test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                    label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                    label="Cross-validation score")

        plt.legend(loc="best")
        plt.savefig('learning_curve' + report + filename + '.png')
        
    ANN.fit(X, y, validation_split=0.1, epochs=best_fit_ANN['epochs'], batch_size=best_fit_ANN['batch_size'], verbose=0)
    # summarize history for accuracy
    plt.figure()
    plt.plot(ANN.history.history['acc'])
    plt.plot(ANN.history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('learning_curveANN' + filename + '.png')
    # summarize history for loss
    plt.figure()
    plt.plot(ANN.history.history['loss'])
    plt.plot(ANN.history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('learning_lossANN' + filename + '.png')
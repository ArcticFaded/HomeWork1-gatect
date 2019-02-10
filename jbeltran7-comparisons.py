from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import tree
from models import svc_param_selection, descision_tree_param_selection, boosted_tree_param_selection, k_nearest_neighbors_param_selection, ANN_param_selection, create_model
from utils import get_prune, post_pruning, classification_accuracy, plot_tsne, plot_corr, print_report, learning_curves
from preprocessing import preprocess_bank_data, preprocess_heart_data
import timeit
import graphviz 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('ggplot')



def classifer_creation(df_final, target_column):
    X = df_final.loc[:, df_final.columns != target_column]
    Y = df_final.loc[:, df_final.columns == target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    
    y_train_ravel = y_train.values.ravel()
    print(len(X_train.columns))
    svc_best_param, svc_report = svc_param_selection(X_train, y_train_ravel, 5)
    print('SVM finished')
    dt_best_param, dt_report = descision_tree_param_selection(X_train, y_train_ravel, 5)
    print('Descision tree finished')
    bt_best_param, bt_report = boosted_tree_param_selection(X_train, y_train_ravel, dt_best_param, 5)
    print('Boosted tree finished')
    knn_best_param, knn_report = k_nearest_neighbors_param_selection(X_train, y_train_ravel, 5)
    print('KNN finished')
    ANN_best_param, ANN_report = ANN_param_selection(X_train, y_train_ravel, 5)
    print('ANN finished')
    

    params = {
            'SVC': svc_best_param,
            'DecisionTree': dt_best_param,
            'BoostedTrees': bt_best_param,
            'K-NearestNeighbors': knn_best_param,
            'NeuralNetworks': ANN_best_param
    }

    for best_params in params:
            print('{classifer}: {params}'.format(classifer=best_params, params=params[best_params]))

    svc = svm.SVC(C=svc_best_param['C'], 
                  gamma=svc_best_param['gamma'], 
                  kernel=svc_best_param['kernel'])
    
    dt = DecisionTreeClassifier(max_depth=dt_best_param['max_depth'], 
                                     min_samples_split=dt_best_param['min_samples_split'], 
                                     min_samples_leaf=dt_best_param['min_samples_leaf'], 
                                     max_features=dt_best_param['max_features'])
    
    # check for pruning
    max_prune, prune = get_prune(dt, X_train, y_train, X_test, y_test)
    print('Pruning with highest score: {max_prune}, Pruning with acceptable loss in accuracy: {prune}'.format(
    max_prune=max_prune, prune=prune))
    
    bt = GradientBoostingClassifier(max_depth=bt_best_param['max_depth'], 
                                    min_samples_split=bt_best_param['min_samples_split'], 
                                    min_samples_leaf=bt_best_param['min_samples_leaf'], 
                                    max_features=bt_best_param['max_features'], 
                                    learning_rate=bt_best_param['learning_rate'], 
                                    n_estimators= bt_best_param['n_estimators'])
    
    knn = KNeighborsClassifier(n_neighbors=knn_best_param['n_neighbors'], 
                               p=knn_best_param['p'])
    
    
    ANN = create_model(len(X_train.columns))
    
    learning_curves(svc, dt, bt, knn, ANN, ANN_best_param, X, Y.values.ravel(), target_column)

    start_time = timeit.default_timer()
    svc.fit(X_train, y_train_ravel)
    print("{classifer} took {time} to fit on X and y".format(classifer='svc', time=(timeit.default_timer() - start_time)))
    start_time = timeit.default_timer()
    dt.fit(X_train, y_train_ravel)
    print("{classifer} took {time} to fit on X and y".format(classifer='dt', time=(timeit.default_timer() - start_time)))
    start_time = timeit.default_timer()
    bt.fit(X_train, y_train_ravel)
    print("{classifer} took {time} to fit on X and y".format(classifer='bt', time=(timeit.default_timer() - start_time)))
    start_time = timeit.default_timer()
    knn.fit(X_train, y_train_ravel)
    print("{classifer} took {time} to fit on X and y".format(classifer='knn', time=(timeit.default_timer() - start_time)))
    start_time = timeit.default_timer()
    ANN.fit(X_train, y_train_ravel, 
            epochs=ANN_best_param['epochs'], 
            batch_size=ANN_best_param['batch_size'], 
            verbose=0)
    print("{classifer} took {time} to fit on X and y".format(classifer='ann', time=(timeit.default_timer() - start_time)))
    
    # PRUNE THE MODEL
    dot_data = tree.export_graphviz(dt, out_file=None, feature_names=X_test.columns) 
    graph = graphviz.Source(dot_data) 
    graph.render("pre-pruning-" + target_column)


    post_pruning(dt.tree_, 0, max_prune)

    dot_data = tree.export_graphviz(dt, out_file=None, feature_names=X_test.columns) 
    graph = graphviz.Source(dot_data) 
    graph.render("post-pruning-" + target_column) 

    
    return svc, dt, bt, knn, ANN, X_train, y_train, X_test, y_test


# read csv    
heart = pd.read_csv('./data/heart.csv')
bank = pd.read_csv('./data/bank.csv', sep=';')
# main areas

heart_final, heart_scalers = preprocess_heart_data(heart)
bank_final, heart_scalers = preprocess_bank_data(bank)


SVC, DecisionTree, BoostedTrees, KNN, ANN, X_train, y_train, X_test, y_test = classifer_creation(heart_final, 'target')
classification_accuracy(SVC, DecisionTree, BoostedTrees, KNN, ANN, X_train, y_train, X_test, y_test)
print('==========================================new area==========================================')
SVC, DecisionTree, BoostedTrees, KNN, ANN, X_train, y_train, X_test, y_test = classifer_creation(bank_final, 'y')
classification_accuracy(SVC, DecisionTree, BoostedTrees, KNN, ANN, X_train, y_train, X_test, y_test)
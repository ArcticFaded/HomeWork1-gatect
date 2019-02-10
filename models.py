from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

def create_model(input_dim):
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=input_dim, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def ANN_param_selection(X, y, nfolds):
    callbacks = [EarlyStopping(monitor='loss', patience=2)]
    
    batch_size = [8, 16]
    epochs = [25, 40]
    param_grid = dict(batch_size=batch_size, epochs=epochs)
    print(len(X.columns))
    model = KerasClassifier(build_fn=create_model, input_dim=len(X.columns), verbose=0)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=nfolds, n_jobs=-1, verbose=1, fit_params={'callbacks': callbacks})
    grid_result = grid.fit(X, y)
    return grid_result.best_params_, grid_result.cv_results_

def k_nearest_neighbors_param_selection(X, y, nfolds):
    n_neighbors = [1, 3, 5, 7, 9]
    distance = [1, 2] # compare L1 to L2 distance
    param_grid = {'n_neighbors': n_neighbors, 'p': distance}
    
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=nfolds, n_jobs=-1, verbose=1)
    grid_search.fit(X, y)
    return grid_search.best_params_, grid_search.cv_results_

def boosted_tree_param_selection(X, y, decision_tree, nfolds):
    learning_rate = [0.001, 0.01, 0.1, 1]
    n_estimators = [50, 100, 150, 200, 250]
    max_depths = [decision_tree['max_depth']]
    min_samples_split = [decision_tree['min_samples_split']]
    min_samples_leaf = [decision_tree['min_samples_leaf']]
    max_features = [decision_tree['max_features']]
    param_grid = {'max_depth': max_depths, 
                  'min_samples_split': min_samples_split, 
                  'min_samples_leaf': min_samples_leaf,
                  'max_features': max_features,
                  'learning_rate': learning_rate,
                  'n_estimators': n_estimators
                 }
    grid_search = GridSearchCV(GradientBoostingClassifier(), param_grid, cv=nfolds, n_jobs=-1, verbose=1)
    grid_search.fit(X, y)
    return grid_search.best_params_, grid_search.cv_results_

def descision_tree_param_selection(X, y, nfolds):
    max_depths = [1, 3, 5, 7, 10]
    min_samples_split = [0.01, 0.05, 0.1, 0.5]
    min_samples_leaf = [0.5, 2, 5, 10]
    max_features = [5, 10, 15, 20, len(X.columns)] if len(X.columns) == 25 else [10, 20, 30, 40, len(X.columns)]
    param_grid = {'max_depth': max_depths, 
                  'min_samples_split': min_samples_split, 
                  'min_samples_leaf': min_samples_leaf,
                  'max_features': max_features
                 }
    grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=nfolds, n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search.best_params_, grid_search.cv_results_
    

def svc_param_selection(X, y, nfolds):
    Cs = [0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    kernels = ['rbf', 'linear']
    param_grid = {'C': Cs, 'gamma' : gammas, 'kernel': kernels}
    grid_search = GridSearchCV(svm.SVC(), param_grid, cv=nfolds, n_jobs=-1, verbose=1)
    grid_search.fit(X, y)
    return grid_search.best_params_, grid_search.cv_results_
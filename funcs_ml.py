from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def classify(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    report = classification_report(y_test, preds) 
    print(report)

def cv(model, X, y):
    kf = KFold(n_splits = 5, shuffle=True, random_state=42)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    score = cross_val_score(pipeline, X, y, cv=kf, scoring='accuracy')
    print(f"Cross-validation scores: {score}")
    print(f"Mean accuracy: {score.mean()}")

def grid_search(model, x_train, y_train,x_test, y_test, param_grid):
    gd = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
    gd.fit(x_train, y_train)
    print("Best Parameters:", gd.best_params_)
    print("Best Score:", gd.best_score_)

    best_model = gd.best_estimator_
    test_score = best_model.score(x_test, y_test)
    print("Test Score:", test_score)


    
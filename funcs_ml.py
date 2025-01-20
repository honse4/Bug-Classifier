from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, cross_val_score
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
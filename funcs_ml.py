from sklearn.metrics import classification_report

def classify(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    report = classification_report(y_test, preds) 
    print(report)
    
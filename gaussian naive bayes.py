

x = dataset.drop(["sinif","numara"],axis=1).values
y = dataset["sinif"].values

X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.50, random_state = 5)

X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.50, random_state = 5)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test) 

classifier_nb = GaussianNB()
classifier_nb.fit(X_train, y_train)

y_pred_nb = classifier_nb.predict(X_test)
cm_nb = confusion_matrix(y_test, y_pred_nb)
print(cm_nb)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(classification_report(y_test,y_pred_nb))
print(f"acc_score = {accuracy_nb}")
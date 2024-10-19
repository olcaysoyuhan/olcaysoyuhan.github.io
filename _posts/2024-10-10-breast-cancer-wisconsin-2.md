---
layout: post
title:  "Meme Kanseri Wisconsin (Teşhis) Veri Seti -2-"
author : olcay
categories: [ Veri Bilimi ]
image: https://images.unsplash.com/photo-1631049127390-4fc14daf3df6?q=80&w=2068&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D
image-title: "National Cancer Institute"
image-link: https://unsplash.com/photos/white-and-black-abstract-painting-ERdyFahv-GU
tags : [ python, veri bilimi, kanser ]
---

Bu veri seti, çeşitli özelliklere sahip hücre çekirdeklerinin sınıflandırılması amacıyla kullanılmıştır. Her bir örnek, belirli bir hücre çekirdeği görüntüsünden elde edilen çeşitli ölçümleri içerir. Özellikler, hem ortalama hem de standart hata değerleri ile birlikte hesaplanmış ve her görüntü için "en kötü" veya en büyük üç özelliğin ortalaması da verilmiştir.

**ID number (Kimlik numarası):** Her örnek için benzersiz bir tanımlayıcı.<br>
**Diagnosis (Tanı):** Hücre çekirdeğinin malign (kötü huylu, M) veya benign (iyi huylu, B) olarak sınıflandırılması.<br>
**radius (yarıçap):** Merkezden çevre üzerindeki noktalara olan mesafelerin ortalaması.<br>
**texture (doku):** Gri ölçekli değerlerin standart sapması.<br>
**perimeter (çevre):** Çekirdeğin çevresinin uzunluğu.<br>
**area (alan):** Çekirdeğin kapladığı yüzey alanı.<br>
**smoothness (pürüzsüzlük):** Yarıçap uzunlıklarındaki yerel varyasyon.<br>
**compactness (kompaktlık):** Çevre^2 / alan - 1.0 hesaplaması ile belirlenen kompaktlık ölçüsü.<br>
**concavity (içbükeylik):** Konturun içbükey kısımlarının şiddeti.<br>
**concave points (içbükey noktalar):** Konturun içbükey kısımlarının sayısı.<br>
**symmetry (simetri):** Çekirdeğin simetrik özelliklerinin ölçüsü.<br>
**fractal dimension (fraktal boyut):** Kıyı şeridi yaklaşımı ile hesaplanan ve 1 ile çıkarılan fraktal boyut değeri.<br>

[breast cancer wisconsin data][breast-cancer-wisconsin-data]

---

## Base Model


```python
log_reg_model = LogisticRegression()
```


```python
log_reg_model.fit(X_train_scaler, y_train)
```



```python
train_accuracy = accuracy_score(y_train, log_reg_model.predict(X_train_scaler))
test_accuracy = accuracy_score(y_test, log_reg_model.predict(X_test_scaler))

print(f"Eğitim doğruluğu: {train_accuracy:.2f}")
print(f"Test doğruluğu: {test_accuracy:.2f}")
```

    Eğitim doğruluğu: 0.99
    Test doğruluğu: 0.97
    


```python
cm = confusion_matrix(y_test, log_reg_model.predict(X_test_scaler)); cm
```




    array([[106,   1],
           [  4,  60]])




```python
ConfusionMatrixDisplay(cm, display_labels=['M', 'B']).plot(cmap=plt.cm.Blues)
```




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x79f2359dd300>




    
![png](/assets/images/breast_cancer_files/breast_cancer_55_1.png)
    



```python
print(classification_report(y_test, log_reg_model.predict(X_test_scaler)))
```

                  precision    recall  f1-score   support
    
               0       0.96      0.99      0.98       107
               1       0.98      0.94      0.96        64
    
        accuracy                           0.97       171
       macro avg       0.97      0.96      0.97       171
    weighted avg       0.97      0.97      0.97       171
    
    


```python
probabilities = log_reg_model.predict_proba(X_test_scaler)
positive_probabilities = probabilities[:, 1]

results = pd.DataFrame({
    'True labels': y_test,
    'Predicted probability': positive_probabilities
})

plt.figure(figsize=(10, 10))
sns.scatterplot(x=range(len(positive_probabilities)),
                y=positive_probabilities,
                hue=results['True labels'],
                palette={0: 'blue', 1: 'red'},
                marker='o',
                alpha=0.7)

plt.axhline(0.5, color='gray', linestyle='--', label='Threshold (0.5)')
plt.xlabel('Test Set Indexes')
plt.ylabel('Positive Class Probability')
plt.title('Positive Class Probabilities in the Test Set')
plt.xlim(0, len(positive_probabilities))
plt.legend(loc='upper right')
plt.show()
```


    
![png](/assets/images/breast_cancer_files/breast_cancer_57_0.png)
    



```python
fpr, tpr, thresholds = roc_curve(y_test, positive_probabilities)
roc_auc = roc_auc_score(y_test, positive_probabilities)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC Score: {roc_auc:.3f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('The Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

```


    
![png](/assets/images/breast_cancer_files/breast_cancer_58_0.png)
    


Makine öğrenmesi algoritmalarını kullanırken baz almak için Logistic Regression tercih edildi. Logistic Regression ikili sınıflandırma problemlerindeki başarımı nedeniyle ve veri setinin de bir ikili sınıflandırma problemi olması nedeniyle tercih edildi.


Tüm özellikleri standart tutularak yapılan tahminlemede; %97 doğruluk, kötü huylu tümör tahmininde %98 f1 ve iyi huylu tümör tahmininde %96 f1 başarımı elde edildi. Bu başarımın daha rahat anlaşılması için scatter plot ile çizdirildi. Sınıflandırma algoritmaları değerlendirme metriklerinden roc curve grafiğinde true - positive ayrımının, hesaplanan auc score (0.998) ile uyumlu olacak şekilde yüksek başarıma sahip olduğu gözlemlendi.

## Model Selection

#### KFold Cross Validation


```python
k = 10
k_fold = KFold(n_splits=k, shuffle=True, random_state=42)
```

Model sonuçlarını değerlendirirken KFold Cross Validation yönteminden faydalanılmıştır. K değeri 10 olarak belirlenmiştir. Bu sayede diğer metriklere ek olarak modelin overfit edip etmediğine bakılmıştır.

### Logistic Regression


```python
param_grid = {
              'penalty': ['l1', 'l2'],
              'max_iter': [100, 200, 300, 400, 500],
              'C': [0, 0.5, 1],
              'random_state' : [42]}

clf = GridSearchCV(log_reg_model, param_grid, cv=k_fold, verbose=0, scoring='accuracy', n_jobs=-1)
best_model = clf.fit(X_train, y_train)
print(clf.best_params_)
```

    {'C': 1, 'max_iter': 100, 'penalty': 'l2', 'random_state': 42}
    


```python
clf=LogisticRegression(random_state=42, C=0.5, max_iter=100, penalty='l2')
clf.fit(X_train_scaler,y_train)
predictions = clf.predict(X_test_scaler)
```


```python
def model_metrics(y_test, predictions, X_train_scaler, X_test_scaler):
    """
    This function calculates and plots the metrics of model results given
    the prediction and target variables.

    Input :
    ---
    y_test
    predictions
    X_test_scaler

    Output:
    ---
    Confusion Matrix, Classification Report, Positive Probalities Scatter Plot
    and ROC Curve and AUC Score
    """

    train_accuracy = accuracy_score(y_train, clf.predict(X_train_scaler))
    test_accuracy = accuracy_score(y_test, clf.predict(X_test_scaler))

    print(f"Eğitim doğruluğu: {train_accuracy:.2f}")
    print(f"Standard sapma: {train_accuracy.std():.4f}")
    print(f"Test doğruluğu: {test_accuracy:.2f}")
    print(f"Standard sapma: {test_accuracy.std():.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions); cm
    ConfusionMatrixDisplay(cm, display_labels=['M', 'B']).plot(cmap=plt.cm.Blues)
    # Classification Report

    print(classification_report(y_test, predictions))

    # Positive Probalities Scatter Plot
    probabilities = clf.predict_proba(X_test_scaler)
    positive_probabilities = probabilities[:, 1]

    results = pd.DataFrame({
        'True labels': y_test,
        'Predicted probability': positive_probabilities
    })

    plt.figure(figsize=(10, 10))
    sns.scatterplot(x=range(len(positive_probabilities)),
                    y=positive_probabilities,
                    hue=results['True labels'],
                    palette={0: 'blue', 1: 'red'},
                    marker='o',
                    alpha=0.7)

    plt.axhline(0.5, color='gray', linestyle='--', label='Threshold (0.5)')
    plt.xlabel('Test Set Indexes')
    plt.ylabel('Positive Class Probability')
    plt.title('Positive Class Probabilities in the Test Set')
    plt.xlim(0, len(positive_probabilities))
    plt.legend(loc='upper right')
    plt.show()

    # ROC Curve and AUC Score
    fpr, tpr, thresholds = roc_curve(y_test, positive_probabilities)
    roc_auc = roc_auc_score(y_test, positive_probabilities)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC Score: {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('The Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
```


```python
model_metrics(y_test, predictions, X_train_scaler, X_test_scaler)
```

    Eğitim doğruluğu: 0.99
    Standard sapma: 0.0000
    Test doğruluğu: 0.98
    Standard sapma: 0.0000
                  precision    recall  f1-score   support
    
               0       0.97      1.00      0.99       107
               1       1.00      0.95      0.98        64
    
        accuracy                           0.98       171
       macro avg       0.99      0.98      0.98       171
    weighted avg       0.98      0.98      0.98       171
    
    


    
![png](/assets/images/breast_cancer_files/breast_cancer_68_1.png)
    



    
![png](/assets/images/breast_cancer_files/breast_cancer_68_2.png)
    



    
![png](/assets/images/breast_cancer_files/breast_cancer_68_3.png)
    


Modelin eğitim ve test datalarındaki accuracy değerleri arasındaki küçük fark ve yüksek AUC score değeri, modelin overfitting riskinin düşük olduğu ve modelin yüksek genelleme ve sınıfları ayırt etme yeteneğine sahip olduğu şeklinde yorumlandı.

### Random Forest


```python
rf = RandomForestClassifier()
```


```python
param_grid = {
              'n_estimators': [50, 100, 200],
              'max_depth': [None, 10, 20, 30],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4]}

clf = GridSearchCV(rf, param_grid, cv=k_fold, verbose=0, scoring='accuracy', n_jobs=-1)
best_model = clf.fit(X_train, y_train)
print(clf.best_params_)
```

    {'max_depth': 10, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 50}
    


```python
clf=RandomForestClassifier(max_depth=30, min_samples_leaf=1, min_samples_split=5, n_estimators=50)
clf.fit(X_train_scaler,y_train)
predictions = clf.predict(X_test_scaler)
```


```python
model_metrics(y_test, predictions, X_train_scaler, X_test_scaler)
```

    Eğitim doğruluğu: 0.99
    Standard sapma: 0.0000
    Test doğruluğu: 0.98
    Standard sapma: 0.0000
                  precision    recall  f1-score   support
    
               0       0.96      1.00      0.98       107
               1       1.00      0.94      0.97        64
    
        accuracy                           0.98       171
       macro avg       0.98      0.97      0.97       171
    weighted avg       0.98      0.98      0.98       171
    
    


    
![png](/assets/images/breast_cancer_files/breast_cancer_74_1.png)
    



    
![png](/assets/images/breast_cancer_files/breast_cancer_74_2.png)
    



    
![png](/assets/images/breast_cancer_files/breast_cancer_74_3.png)
    


#### Feature Importances


```python
importances = clf.feature_importances_

features_df = pd.DataFrame({
    'Feature': X.columns.to_list(),
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(features_df)

plt.figure(figsize=(10, 6))
plt.barh(features_df['Feature'], features_df['Importance'])
plt.xlabel('Feature Importance')
plt.title('Importance of Attributes')
plt.show()
```

                        Feature  Importance
    22          perimeter_worst    0.197899
    20             radius_worst    0.151276
    27     concave points_worst    0.113842
    7       concave points_mean    0.082422
    23               area_worst    0.073750
    2            perimeter_mean    0.069089
    0               radius_mean    0.039225
    26          concavity_worst    0.036716
    13                  area_se    0.035470
    3                 area_mean    0.033296
    6            concavity_mean    0.027250
    25        compactness_worst    0.025306
    1              texture_mean    0.020290
    24         smoothness_worst    0.016242
    21            texture_worst    0.013172
    29  fractal_dimension_worst    0.007695
    19     fractal_dimension_se    0.006797
    28           symmetry_worst    0.006414
    4           smoothness_mean    0.006062
    15           compactness_se    0.005781
    5          compactness_mean    0.004559
    10                radius_se    0.004437
    14            smoothness_se    0.004094
    16             concavity_se    0.003734
    8             symmetry_mean    0.003684
    9    fractal_dimension_mean    0.002688
    18              symmetry_se    0.002551
    12             perimeter_se    0.002384
    11               texture_se    0.002246
    17        concave points_se    0.001631
    


    
![png](/assets/images/breast_cancer_files/breast_cancer_76_1.png)
    


Modelin eğitim datası ve test datası arasındaki küçük fark, modelin eğitim verilerine biraz daha iyi uyum sağladığını gösteriyor. AUC score değerinin yüksek olması nedeniyle de modelin overfitting etmediği düşünülmektedir. Modelin yüksek genelleme ve sınıfları ayırt etme yeteneğine sahip olduğuna da karar verilmiştir.

Feature importance değer ve grafiği incelendiğinde; **radius_worst**, **area_worst**, **perimeter_worst** ve **concave_points_mean** özelliklerinin Random Forest modelinin sınıfladırma kararını en fazla etkileyen özellikler olduğu görülmüştür.

### XGBoost


```python
xgboost = XGBClassifier()
```


```python
param_grid = {
              'n_estimators': [50, 100, 200],
              'learning_rate': [0.01, 0.1, 0.2],
              'max_depth': [3, 6, 9]}

clf = GridSearchCV(xgboost, param_grid, cv=k_fold, verbose=0, scoring='accuracy', n_jobs=-1)
best_model = clf.fit(X_train, y_train)
print(clf.best_params_)
```

    {'learning_rate': 0.2, 'max_depth': 3, 'n_estimators': 100}
    


```python
clf=XGBClassifier(learning_rate=0.1, max_depth=6, n_estimators=200)
clf.fit(X_train_scaler,y_train)
predictions = clf.predict(X_test_scaler)
```


```python
model_metrics(y_test, predictions, X_train_scaler, X_test_scaler)
```

    Eğitim doğruluğu: 1.00
    Standard sapma: 0.0000
    Test doğruluğu: 0.98
    Standard sapma: 0.0000
                  precision    recall  f1-score   support
    
               0       0.97      1.00      0.99       107
               1       1.00      0.95      0.98        64
    
        accuracy                           0.98       171
       macro avg       0.99      0.98      0.98       171
    weighted avg       0.98      0.98      0.98       171
    
    


    
![png](/assets/images/breast_cancer_files/breast_cancer_82_1.png)
    



    
![png](/assets/images/breast_cancer_files/breast_cancer_82_2.png)
    


    
![png](/assets/images/breast_cancer_files/breast_cancer_82_3.png)
    


Model eğitim ve test sonuçları arasındaki küçük fark ve yüksek AUC score değeri, modelin overfitting ihtimalinin düşük olduğunu ve genelleme yeteneğinin iyi olduğunu gösteriyor. Eğitim doğruluğunun 1.00 olması, modelin eğitim verilerine tam uyum sağlamış olduğu ve bu nedenle modelin overfitting ihtimalini düşük de olsa düşündürmektedir.

## Best Model

Çalışmada Logistic Regression, Random Forest Classifier ve XGBoost makine öğrenmesi modelleri kullanıldı. Modeller arasında Logistic Regression ve XGBoost daha yüksek doğruluk, genelleme ve sınıfları ayırt edebilme özelliği gösterdi. Her iki modelin birbirine çok yakın sonuçlar vermesi nedeniyle; yapılacak çalışma sonuçlarında açıklanabilirliği açısından Logistic Regression modeli tercih edilmesinin doğru olacağına karar verilmiştir.

Öneri olarak; daha fazla model hiperparametre ayarı kullanılması, veri setine daha fazla veri girişi sağlanması ile overfitting ihtimali azaltılabilir. Tüm bunlara ek olarak farklı sınıflandırma modelleri de kullanılarak daha geniş kapsamlı karşılaştırma sağlanabilir.

---

[kaggle çalışması][breast-cancer-wisconsin-data-kaggle-project]

[breast-cancer-wisconsin-data]: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

[breast-cancer-wisconsin-data-kaggle-project]: https://www.kaggle.com/code/olcayyaclo/breast-cancer

---
layout: post
title:  "Meme Kanseri Wisconsin (Teşhis) Veri Seti -1-"
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

## Import Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import scipy
from scipy.cluster import hierarchy as hc
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, roc_curve, auc, roc_auc_score

warnings.filterwarnings('ignore')
```

## Read Data


```python
PATH = '/content/'
df = pd.read_csv('breast_cancer.csv')
```

## Exploratory Data Analysis (EDA)


```python
df.head(10)
```

![png](/assets/images/breast_cancer_files/breast_cancer_head_10.png)



```python
df.columns
```




    Index(['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
           'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
           'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
           'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
           'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
           'fractal_dimension_se', 'radius_worst', 'texture_worst',
           'perimeter_worst', 'area_worst', 'smoothness_worst',
           'compactness_worst', 'concavity_worst', 'concave points_worst',
           'symmetry_worst', 'fractal_dimension_worst', 'Unnamed: 32'],
          dtype='object')




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 569 entries, 0 to 568
    Data columns (total 33 columns):
     #   Column                   Non-Null Count  Dtype  
    ---  ------                   --------------  -----  
     0   id                       569 non-null    int64  
     1   diagnosis                569 non-null    object 
     2   radius_mean              569 non-null    float64
     3   texture_mean             569 non-null    float64
     4   perimeter_mean           569 non-null    float64
     5   area_mean                569 non-null    float64
     6   smoothness_mean          569 non-null    float64
     7   compactness_mean         569 non-null    float64
     8   concavity_mean           569 non-null    float64
     9   concave points_mean      569 non-null    float64
     10  symmetry_mean            569 non-null    float64
     11  fractal_dimension_mean   569 non-null    float64
     12  radius_se                569 non-null    float64
     13  texture_se               569 non-null    float64
     14  perimeter_se             569 non-null    float64
     15  area_se                  569 non-null    float64
     16  smoothness_se            569 non-null    float64
     17  compactness_se           569 non-null    float64
     18  concavity_se             569 non-null    float64
     19  concave points_se        569 non-null    float64
     20  symmetry_se              569 non-null    float64
     21  fractal_dimension_se     569 non-null    float64
     22  radius_worst             569 non-null    float64
     23  texture_worst            569 non-null    float64
     24  perimeter_worst          569 non-null    float64
     25  area_worst               569 non-null    float64
     26  smoothness_worst         569 non-null    float64
     27  compactness_worst        569 non-null    float64
     28  concavity_worst          569 non-null    float64
     29  concave points_worst     569 non-null    float64
     30  symmetry_worst           569 non-null    float64
     31  fractal_dimension_worst  569 non-null    float64
     32  Unnamed: 32              0 non-null      float64
    dtypes: float64(31), int64(1), object(1)
    memory usage: 146.8+ KB
    


```python
df.duplicated().sum()
```




    0



Veri seti 33 kolon ve 569 satır veriden oluşmakta. '**Unnamed: 32**' kolonu veri içermemekte. '**id**' kolonu int64, '**diagnosis**' kolonu object, veri setinde yer alan diğer kolonlar ise float64 vei tipinde tanımlanmıştır. Veri setinde tekrar eden bir veri bulunamadı.

### Missing Values


```python
df.isnull().sum()
```


![png](/assets/images/breast_cancer_files/breast_cancer_missing_values.png)




```python
df.drop('Unnamed: 32', axis=1, inplace=True)
```


```python
df.shape
```




    (569, 32)



'**Unnamed: 32**' kolonu veri içermediği için veri setinden çıkartılmış ve veri seti 32 kolon olarak güncellenmiştir.


```python
df.corr(numeric_only=True)
```




![png](/assets/images/breast_cancer_files/breast_cancer_corr.png)





```python
plt.figure(figsize=(20, 20))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt='.2f', cmap='Blues')
plt.show()
```


    
![png](/assets/images/breast_cancer_files/breast_cancer_18_0.png)
    


Veri setindeki korelasyon hesaplaması yapılmış ve seaborn kütüphanesi ile ısı haritası çıkarılmıştır. Isı haritası incelendiğinde '**radius mean**' ve '**perimeter mean**' kolonları '**area mean**' kolonuyla, '**radius worst**' ve '**perimeter worst**' kolonlarıda karşılıklı olarak birbireriyle **0.99** oranında pozitif korelasyona; '**fractal dimension mean**' ve '**radius mean**' kolonları **-0.31** oranında negatif korelasyona sahip olduğu görünmektedir.

### Hierarchical Clustering Dendogram

[scipy cluster hierarchy dendrogram][scipy-cluster-hierarchy-dendrogram]

```python
corr = np.round(scipy.stats.spearmanr(df).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(16,10))
dendrogram = hc.dendrogram(z, labels=df.columns, orientation='left', leaf_font_size=16)
plt.show()
```


    
![png](/assets/images/breast_cancer_files/breast_cancer_22_0.png)
    


Veri setindeki korelasyon hesaplamasına farklı bir bakış açısı getirmek için Scipy kütüphanesinde yer alan dendogramdan yararlanılmıştır. Veri seti spearman korelasyon hesabı yapılarak, hiyerarşik kümeleme özelliği olan '**average**' metodu ile tüm veri setindeki değerlerin ortalamaları hesaplanmış ve sonuç dendogram ile gösterilmiştir. Buradan da anlaşılacağı üzere '**area worst**' ve '**radius worst**', '**area mean**' ve '**radius mean**' kolonları arasında kuvvetli bir ilişki vardır. En zayıf ilişki beklenildiği üzere 'id' olarak gözlemlenmiştir.


```python
df.drop('id', axis=1, inplace=True)
```


```python
df.shape
```




    (569, 31)



Veri setinde anlamlı bir etkiye sahip olmayan '**id**' kolonu çıkartıldı, böylece veri 31 kolon içeren son haline ulaştı.

### Data Visualizations


```python
plt.figure()
plt.pie(df['diagnosis'].value_counts(), autopct='%.2f%%', labels=['Benign', 'Malign'])
plt.title('Distribution of Benign and Malignant Diagnoses')
plt.show()

print(df['diagnosis'].value_counts())
```


    
![png](/assets/images/breast_cancer_files/breast_cancer_28_0.png)
    


    diagnosis
    B    357
    M    212
    Name: count, dtype: int64
    

Veri setinde 357 iyi huylu, 212 kötü huylu tümör hücresi bilgisi vardır. Oransal olarak bakıldığında da %62.74 iyi, %37.26 kötü huylu hücreye tekabül etmektedir.


```python
def diagnosis_plots(df, cols):
    """
    This function groups diagnosis features in the dataset and
    plots bar charts to compare them with other features.

    Input :
    ---
    df = data set
    cols = columns

    Output :
    ---
    bar charts

    """


    cols = [col for col in cols if col != 'diagnosis']
    n = len(cols)
    ncols = 2
    nrows = (n + 1) // ncols

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 5 * nrows))
    axs = axs.flatten()

    for i, col in enumerate(cols):
        mean_values = df.groupby('diagnosis')[col].agg('mean')
        axs[i].bar(mean_values.index, mean_values.values)
        axs[i].set_title(col)
        axs[i].set_ylabel('Mean Value')
        axs[i].set_xlabel('Diagnosis')

    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.show()
```


```python
diagnosis_plots(df, df.columns)
```


    
![png](/assets/images/breast_cancer_files/breast_cancer_31_0.png)
    


Grafikler incelendiğinde tüm özelliklerde kötü huylu tümör verisi, iyi huylu tümör verisine göre daha fazla gelmektedir. Bu verilerden anlaşıldığı üzere kötü huylu tümör verilerinde hücrelerdeki içbükeylik, içbükey nokta sayıları, çekirdeğin kapladığı alan ve uzunlukları, yarı çap ve kompaktlık özellikleri kötü huylu tümör tespitinde özellikle belirleyici durumdadır.

## Label Encoder


```python
df.diagnosis.unique()
```




    array(['M', 'B'], dtype=object)




```python
encoder = LabelEncoder()
```


```python
df.diagnosis = encoder.fit_transform(df.diagnosis)
```

Veri setinde kategorik veri tipipne sahip olan 'diagnosis' kolonu label encoder yöntemi ile nümerik hale getirildi. Label encoder yöntemi ilgili kolondaki değerler arasında sıralama ilişkisi olmadığı için tercih edildi.

## Data Split


```python
X = df.drop('diagnosis', axis=1)
```


```python
y = df['diagnosis']
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42, stratify=y)
```


```python
X_train.shape
```




    (398, 30)




```python
X_test.shape
```




    (171, 30)



Veri setinde hedef değişken olarak 'diagnosis' kolounu 'y', diğer kolonlarda özellikleri temsilen 'X' olarak tanımlandı. Hedef değişken dağılımını korumak için stratify özelliği kullanıldı ve veri seti %30 test datası olacak şekilde bölündü. Buna göre train dataları 398 satır veri, test dataları ise 171 satır veriye sahip oldu.

## Standart Scaler


```python
scaler = StandardScaler()
```


```python
X_train_scaler = scaler.fit_transform(X_train)
```


```python
X_test_scaler = scaler.transform(X_test)
```

Değerler arasında büyük sayısal farkların birbirlerine karşı baskın gelmesini önlemek için, veri setindeki özellikleri temsil eden X_train ve X_test değişkenleri 0 - 1 arasına sıkışacak şekilde Standart Scaler uygulandı.

---

[kaggle çalışması][breast-cancer-wisconsin-data-kaggle-project]

[scipy-cluster-hierarchy-dendrogram]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html

[breast-cancer-wisconsin-data]: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

[breast-cancer-wisconsin-data-kaggle-project]: https://www.kaggle.com/code/olcayyaclo/breast-cancer
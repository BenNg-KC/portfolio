---
layout: page
title: Running Injury Prediction
permalink: /running-injury
---

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import joblib
```

```python
fukuchi_data = pd.read_csv(r"C:\Users\benja\Desktop\Capstone_Running_Injury\clean_data\fukuchi_data_final.csv")
ric_data = pd.read_csv(r"C:\Users\benja\Desktop\Capstone_Running_Injury\clean_data\ric_data_final.csv")
```

```python
fukuchi_data.shape
```




    (420, 43)



```python
fukuchi_data.head()
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subject</th>
      <th>age</th>
      <th>height</th>
      <th>mass</th>
      <th>gender</th>
      <th>dominance</th>
      <th>level</th>
      <th>experience</th>
      <th>sessions_per_wk</th>
      <th>treadmill</th>
      <th>...</th>
      <th>10_km</th>
      <th>12_km</th>
      <th>15_km</th>
      <th>21_km</th>
      <th>42_km</th>
      <th>5_km</th>
      <th>6_km</th>
      <th>nan</th>
      <th>ultra (50-70 km)</th>
      <th>ultra_42_km</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>22</td>
      <td>181.0</td>
      <td>62.0</td>
      <td>M</td>
      <td>R</td>
      <td>Competitive</td>
      <td>4</td>
      <td>3</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>22</td>
      <td>181.0</td>
      <td>62.0</td>
      <td>M</td>
      <td>R</td>
      <td>Competitive</td>
      <td>4</td>
      <td>3</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>22</td>
      <td>181.0</td>
      <td>62.0</td>
      <td>M</td>
      <td>R</td>
      <td>Competitive</td>
      <td>4</td>
      <td>3</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>22</td>
      <td>181.0</td>
      <td>62.0</td>
      <td>M</td>
      <td>R</td>
      <td>Competitive</td>
      <td>4</td>
      <td>3</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>22</td>
      <td>181.0</td>
      <td>62.0</td>
      <td>M</td>
      <td>R</td>
      <td>Competitive</td>
      <td>4</td>
      <td>3</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 43 columns</p>
</div>


```python
fukuchi_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 420 entries, 0 to 419
    Data columns (total 43 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   subject           420 non-null    int64  
     1   age               420 non-null    int64  
     2   height            420 non-null    float64
     3   mass              420 non-null    float64
     4   gender            420 non-null    object 
     5   dominance         420 non-null    object 
     6   level             420 non-null    object 
     7   experience        420 non-null    int64  
     8   sessions_per_wk   420 non-null    int64  
     9   treadmill         420 non-null    int64  
     10  asphalt           420 non-null    int64  
     11  grass             420 non-null    int64  
     12  trail             420 non-null    int64  
     13  sand              420 non-null    int64  
     14  concrete          420 non-null    int64  
     15  surface_alt       420 non-null    int64  
     16  run_grp           420 non-null    object 
     17  volume            420 non-null    object 
     18  pace              420 non-null    float64
     19  race_dist         408 non-null    object 
     20  injury            420 non-null    object 
     21  injury_loc        420 non-null    object 
     22  diagnostic_med    420 non-null    object 
     23  diagnostic        420 non-null    object 
     24  injury_on_date    420 non-null    object 
     25  shoe_size         420 non-null    float64
     26  shoe_brand        420 non-null    object 
     27  shoe_model        420 non-null    object 
     28  shoe_pairs        420 non-null    int64  
     29  shoe_change       420 non-null    object 
     30  shoe_comfort      420 non-null    int64  
     31  shoe_insert       420 non-null    object 
     32  race_dist_list    420 non-null    object 
     33  10_km             420 non-null    int64  
     34  12_km             420 non-null    int64  
     35  15_km             420 non-null    int64  
     36  21_km             420 non-null    int64  
     37  42_km             420 non-null    int64  
     38  5_km              420 non-null    int64  
     39  6_km              420 non-null    int64  
     40  nan               420 non-null    int64  
     41  ultra (50-70 km)  420 non-null    int64  
     42  ultra_42_km       420 non-null    int64  
    dtypes: float64(4), int64(23), object(16)
    memory usage: 141.2+ KB

```python
fukuchi_data["speed"] = 60 / fukuchi_data["pace"]

fukuchi_data.drop(columns="pace", inplace=True)
```

I want to merge fukuchi_data with the ric_data, so I will try to remove features that may affect the integration between the two dataframes.

```python
fukuchi_data.drop(columns=['treadmill', 'asphalt','grass','trail','sand','concrete','surface_alt'], inplace=True)

fukuchi_data['injury'] = fukuchi_data['injury'].replace({'Yes':'1','No':'0'})
```

```python
fukuchi_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 420 entries, 0 to 419
    Data columns (total 36 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   subject           420 non-null    int64  
     1   age               420 non-null    int64  
     2   height            420 non-null    float64
     3   mass              420 non-null    float64
     4   gender            420 non-null    object 
     5   dominance         420 non-null    object 
     6   level             420 non-null    object 
     7   experience        420 non-null    int64  
     8   sessions_per_wk   420 non-null    int64  
     9   run_grp           420 non-null    object 
     10  volume            420 non-null    object 
     11  race_dist         408 non-null    object 
     12  injury            420 non-null    object 
     13  injury_loc        420 non-null    object 
     14  diagnostic_med    420 non-null    object 
     15  diagnostic        420 non-null    object 
     16  injury_on_date    420 non-null    object 
     17  shoe_size         420 non-null    float64
     18  shoe_brand        420 non-null    object 
     19  shoe_model        420 non-null    object 
     20  shoe_pairs        420 non-null    int64  
     21  shoe_change       420 non-null    object 
     22  shoe_comfort      420 non-null    int64  
     23  shoe_insert       420 non-null    object 
     24  race_dist_list    420 non-null    object 
     25  10_km             420 non-null    int64  
     26  12_km             420 non-null    int64  
     27  15_km             420 non-null    int64  
     28  21_km             420 non-null    int64  
     29  42_km             420 non-null    int64  
     30  5_km              420 non-null    int64  
     31  6_km              420 non-null    int64  
     32  nan               420 non-null    int64  
     33  ultra (50-70 km)  420 non-null    int64  
     34  ultra_42_km       420 non-null    int64  
     35  speed             420 non-null    float64
    dtypes: float64(4), int64(16), object(16)
    memory usage: 118.3+ KB

```python
fukuchi_data.dtypes
```




    subject               int64
    age                   int64
    height              float64
    mass                float64
    gender               object
    dominance            object
    level                object
    experience            int64
    sessions_per_wk       int64
    run_grp              object
    volume               object
    race_dist            object
    injury               object
    injury_loc           object
    diagnostic_med       object
    diagnostic           object
    injury_on_date       object
    shoe_size           float64
    shoe_brand           object
    shoe_model           object
    shoe_pairs            int64
    shoe_change          object
    shoe_comfort          int64
    shoe_insert          object
    race_dist_list       object
    10_km                 int64
    12_km                 int64
    15_km                 int64
    21_km                 int64
    42_km                 int64
    5_km                  int64
    6_km                  int64
    nan                   int64
    ultra (50-70 km)      int64
    ultra_42_km           int64
    speed               float64
    dtype: object



```python
fukuchi_data.describe()
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subject</th>
      <th>age</th>
      <th>height</th>
      <th>mass</th>
      <th>experience</th>
      <th>sessions_per_wk</th>
      <th>shoe_size</th>
      <th>shoe_pairs</th>
      <th>shoe_comfort</th>
      <th>10_km</th>
      <th>12_km</th>
      <th>15_km</th>
      <th>21_km</th>
      <th>42_km</th>
      <th>5_km</th>
      <th>6_km</th>
      <th>nan</th>
      <th>ultra (50-70 km)</th>
      <th>ultra_42_km</th>
      <th>speed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>420.000000</td>
      <td>420.000000</td>
      <td>420.000000</td>
      <td>420.000000</td>
      <td>420.000000</td>
      <td>420.000000</td>
      <td>420.000000</td>
      <td>420.000000</td>
      <td>420.000000</td>
      <td>420.000000</td>
      <td>420.000000</td>
      <td>420.000000</td>
      <td>420.000000</td>
      <td>420.000000</td>
      <td>420.000000</td>
      <td>420.000000</td>
      <td>420.000000</td>
      <td>420.000000</td>
      <td>420.000000</td>
      <td>420.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>18.242857</td>
      <td>34.557143</td>
      <td>175.861429</td>
      <td>70.227857</td>
      <td>93.914286</td>
      <td>3.700000</td>
      <td>9.521429</td>
      <td>2.257143</td>
      <td>8.157143</td>
      <td>0.471429</td>
      <td>0.014286</td>
      <td>0.028571</td>
      <td>0.657143</td>
      <td>0.485714</td>
      <td>0.257143</td>
      <td>0.028571</td>
      <td>0.028571</td>
      <td>0.028571</td>
      <td>0.028571</td>
      <td>14.628440</td>
    </tr>
    <tr>
      <th>std</th>
      <td>10.516180</td>
      <td>6.648995</td>
      <td>6.804581</td>
      <td>8.250023</td>
      <td>84.708526</td>
      <td>0.817762</td>
      <td>1.006313</td>
      <td>1.381974</td>
      <td>1.295880</td>
      <td>0.499778</td>
      <td>0.118808</td>
      <td>0.166797</td>
      <td>0.475230</td>
      <td>0.500392</td>
      <td>0.437580</td>
      <td>0.166797</td>
      <td>0.166797</td>
      <td>0.166797</td>
      <td>0.166797</td>
      <td>1.482807</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>19.000000</td>
      <td>162.700000</td>
      <td>56.850000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>7.500000</td>
      <td>1.000000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>9.740260</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>9.000000</td>
      <td>30.000000</td>
      <td>169.000000</td>
      <td>64.700000</td>
      <td>24.000000</td>
      <td>3.000000</td>
      <td>8.500000</td>
      <td>1.000000</td>
      <td>8.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>13.574661</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>18.000000</td>
      <td>33.500000</td>
      <td>177.100000</td>
      <td>69.950000</td>
      <td>60.000000</td>
      <td>4.000000</td>
      <td>9.500000</td>
      <td>2.000000</td>
      <td>8.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.492838</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>27.000000</td>
      <td>39.000000</td>
      <td>181.200000</td>
      <td>76.750000</td>
      <td>120.000000</td>
      <td>4.000000</td>
      <td>10.500000</td>
      <td>3.000000</td>
      <td>9.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>15.859031</td>
    </tr>
    <tr>
      <th>max</th>
      <td>39.000000</td>
      <td>51.000000</td>
      <td>192.400000</td>
      <td>101.300000</td>
      <td>300.000000</td>
      <td>6.000000</td>
      <td>12.000000</td>
      <td>6.000000</td>
      <td>10.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>17.821782</td>
    </tr>
  </tbody>
</table>
</div>


```python
fukuchi_data.describe(include='all')
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subject</th>
      <th>age</th>
      <th>height</th>
      <th>mass</th>
      <th>gender</th>
      <th>dominance</th>
      <th>level</th>
      <th>experience</th>
      <th>sessions_per_wk</th>
      <th>run_grp</th>
      <th>...</th>
      <th>12_km</th>
      <th>15_km</th>
      <th>21_km</th>
      <th>42_km</th>
      <th>5_km</th>
      <th>6_km</th>
      <th>nan</th>
      <th>ultra (50-70 km)</th>
      <th>ultra_42_km</th>
      <th>speed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>420.000000</td>
      <td>420.000000</td>
      <td>420.000000</td>
      <td>420.000000</td>
      <td>420</td>
      <td>420</td>
      <td>420</td>
      <td>420.000000</td>
      <td>420.000000</td>
      <td>420</td>
      <td>...</td>
      <td>420.000000</td>
      <td>420.000000</td>
      <td>420.000000</td>
      <td>420.000000</td>
      <td>420.000000</td>
      <td>420.000000</td>
      <td>420.000000</td>
      <td>420.000000</td>
      <td>420.000000</td>
      <td>420.000000</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>M</td>
      <td>R</td>
      <td>Competitive</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Yes</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>408</td>
      <td>324</td>
      <td>378</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>252</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>18.242857</td>
      <td>34.557143</td>
      <td>175.861429</td>
      <td>70.227857</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>93.914286</td>
      <td>3.700000</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.014286</td>
      <td>0.028571</td>
      <td>0.657143</td>
      <td>0.485714</td>
      <td>0.257143</td>
      <td>0.028571</td>
      <td>0.028571</td>
      <td>0.028571</td>
      <td>0.028571</td>
      <td>14.628440</td>
    </tr>
    <tr>
      <th>std</th>
      <td>10.516180</td>
      <td>6.648995</td>
      <td>6.804581</td>
      <td>8.250023</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>84.708526</td>
      <td>0.817762</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.118808</td>
      <td>0.166797</td>
      <td>0.475230</td>
      <td>0.500392</td>
      <td>0.437580</td>
      <td>0.166797</td>
      <td>0.166797</td>
      <td>0.166797</td>
      <td>0.166797</td>
      <td>1.482807</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>19.000000</td>
      <td>162.700000</td>
      <td>56.850000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>9.740260</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>9.000000</td>
      <td>30.000000</td>
      <td>169.000000</td>
      <td>64.700000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>24.000000</td>
      <td>3.000000</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>13.574661</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>18.000000</td>
      <td>33.500000</td>
      <td>177.100000</td>
      <td>69.950000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>60.000000</td>
      <td>4.000000</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.492838</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>27.000000</td>
      <td>39.000000</td>
      <td>181.200000</td>
      <td>76.750000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>120.000000</td>
      <td>4.000000</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>15.859031</td>
    </tr>
    <tr>
      <th>max</th>
      <td>39.000000</td>
      <td>51.000000</td>
      <td>192.400000</td>
      <td>101.300000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>300.000000</td>
      <td>6.000000</td>
      <td>NaN</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>17.821782</td>
    </tr>
  </tbody>
</table>
<p>11 rows × 36 columns</p>
</div>


```python
cat_cols = ['gender','level','volume']

for col in cat_cols:
    plt.figure(figsize=(8,6))
    sns.countplot(data=fukuchi_data, x=col, hue="injury")

    plt.title(f"Injury by {col}")
    plt.xlabel("Gender")
    plt.ylabel("Count")
    plt.show()
```

![png]({{ site.baseurl }}/capstone_model_files/capstone_model/capstone_model_12_0.png){: .center-image }

![png]({{ site.baseurl }}/capstone_model_files/capstone_model/capstone_model_12_1.png){: .center-image }

![png]({{ site.baseurl }}/capstone_model_files/capstone_model/capstone_model_12_2.png){: .center-image }

```python
gender_count = pd.crosstab(fukuchi_data['gender'],fukuchi_data['injury'])

gender_count
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>injury</th>
      <th>0</th>
      <th>1</th>
    </tr>
    <tr>
      <th>gender</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>F</th>
      <td>12</td>
      <td>0</td>
    </tr>
    <tr>
      <th>M</th>
      <td>264</td>
      <td>144</td>
    </tr>
  </tbody>
</table>
</div>


```python
numeric_fukuchi = ["height","mass", "sessions_per_wk", "volume"]

for col_num in numeric_fukuchi:
    plt.figure(figsize=(8,6))
    sns.boxplot(data=fukuchi_data, x="injury", y=col_num)

    plt.title(f"{col_num} by Injury Status")
    plt.xlabel("Injury")
    plt.ylabel(col_num)
    plt.show()
```

![png]({{ site.baseurl }}/capstone_model_files/capstone_model/capstone_model_14_0.png){: .center-image }

![png]({{ site.baseurl }}/capstone_model_files/capstone_model/capstone_model_14_1.png){: .center-image }

![png]({{ site.baseurl }}/capstone_model_files/capstone_model/capstone_model_14_2.png){: .center-image }

![png]({{ site.baseurl }}/capstone_model_files/capstone_model/capstone_model_14_3.png){: .center-image }

```python
plt.figure(figsize=(8,6))
sns.boxplot(data=fukuchi_data, x="shoe_size", y="injury")

plt.title("Distribution of Injury by Shoe Size")
plt.xlabel("Shoe size")
plt.ylabel("Injury")
plt.show()
```

![png]({{ site.baseurl }}/capstone_model_files/capstone_model/capstone_model_15_0.png){: .center-image }

```python
fukuchi_data['injury_loc'].value_counts()
```




    injury_loc
    -                        228
    Nil                       48
    Right Aquilles tendon     24
    Right knee                24
    Ankle bileteral           12
    Left anterior tibia       12
    Left hip                  12
    Left Fibula               12
    Right fibula              12
    Left patela               12
    Right plantar fascia       6
    Left shank                 6
    Left knee                  6
    Right tibia                6
    Name: count, dtype: int64



```python
fukuchi_data['injury_loc'].replace({'-':'Nil'},inplace=True)
```

    C:\Users\benja\AppData\Local\Temp\ipykernel_7212\3184779356.py:1: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
    The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.
    
    For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.
    
    
      fukuchi_data['injury_loc'].replace({'-':'Nil'},inplace=True)

```python
fukuchi_data['injury_loc'].value_counts()
```




    injury_loc
    Nil                      276
    Right knee                24
    Right Aquilles tendon     24
    Left anterior tibia       12
    Ankle bileteral           12
    Left Fibula               12
    Left hip                  12
    Left patela               12
    Right fibula              12
    Right plantar fascia       6
    Left shank                 6
    Left knee                  6
    Right tibia                6
    Name: count, dtype: int64



```python
top_injuries = fukuchi_data['injury_loc'].value_counts().nlargest(10).index
filtered_data = fukuchi_data[fukuchi_data['injury_loc'].isin(top_injuries)]

plt.figure(figsize=(12, 6))
sns.boxplot(x="injury_loc", y="age", data=filtered_data)

plt.xticks(rotation=45)
plt.title("Age Distribution by Top 10 Injury Locations")
plt.show()
```

![png]({{ site.baseurl }}/capstone_model_files/capstone_model/capstone_model_19_0.png){: .center-image }

```python
plt.figure(figsize=(12,6))
sns.countplot(data=fukuchi_data, x="shoe_brand", hue="injury", order=fukuchi_data["shoe_brand"].value_counts().index)

plt.title("Injury Counts by Shoe Brand")
plt.xlabel("Shoe Brand")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()
```

![png]({{ site.baseurl }}/capstone_model_files/capstone_model/capstone_model_20_0.png){: .center-image }

From the graph above, I can see that the highest injury occurrence takes place with Adidas shoes, which has an injury rate above 50%. However, the dataset is small and the data might not be conclusive.

```python
shoe_injury_count = pd.crosstab(fukuchi_data['shoe_brand'], fukuchi_data['injury'])

shoe_injury_rate = (pd.crosstab(fukuchi_data['shoe_brand'], fukuchi_data['injury'], normalize='index').iloc[:, 1])

shoe_injury_count['injury_rate'] = shoe_injury_rate

shoe_injury_count.sort_values(by='injury_rate', ascending=False)
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>injury</th>
      <th>0</th>
      <th>1</th>
      <th>injury_rate</th>
    </tr>
    <tr>
      <th>shoe_brand</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Asics, Adidas, Adidas</th>
      <td>0</td>
      <td>12</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Mizuno, asics</th>
      <td>0</td>
      <td>12</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Sketchers</th>
      <td>12</td>
      <td>24</td>
      <td>0.666667</td>
    </tr>
    <tr>
      <th>Adidas</th>
      <td>36</td>
      <td>48</td>
      <td>0.571429</td>
    </tr>
    <tr>
      <th>Asics</th>
      <td>66</td>
      <td>36</td>
      <td>0.352941</td>
    </tr>
    <tr>
      <th>Nike</th>
      <td>54</td>
      <td>12</td>
      <td>0.181818</td>
    </tr>
    <tr>
      <th>Adidas, Saucony</th>
      <td>12</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Brooks</th>
      <td>12</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Asics/Mizuno</th>
      <td>12</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Adidas, Asics</th>
      <td>12</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Joma</th>
      <td>6</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>New Balance</th>
      <td>12</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Mizuno</th>
      <td>18</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Nike, Adidas</th>
      <td>12</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Puma</th>
      <td>12</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>


Before any potential merge, I want to create a logistic regression to see if any of the features that I am not going to combine are impactful in injury prediction, and if they are, how can I integrate them into my final prediction model.

```python
fukuchi_encoded = pd.get_dummies(fukuchi_data,columns=['shoe_brand','gender','level'],drop_first=True)

fukuchi_encoded.head()
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subject</th>
      <th>age</th>
      <th>height</th>
      <th>mass</th>
      <th>dominance</th>
      <th>experience</th>
      <th>sessions_per_wk</th>
      <th>run_grp</th>
      <th>volume</th>
      <th>race_dist</th>
      <th>...</th>
      <th>shoe_brand_Mizuno</th>
      <th>shoe_brand_Mizuno, asics</th>
      <th>shoe_brand_New Balance</th>
      <th>shoe_brand_Nike</th>
      <th>shoe_brand_Nike, Adidas</th>
      <th>shoe_brand_Puma</th>
      <th>shoe_brand_Sketchers</th>
      <th>gender_M</th>
      <th>level_Elite</th>
      <th>level_Recreational</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>22</td>
      <td>181.0</td>
      <td>62.0</td>
      <td>R</td>
      <td>4</td>
      <td>3</td>
      <td>Yes</td>
      <td>26-35 km</td>
      <td>5 km</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>22</td>
      <td>181.0</td>
      <td>62.0</td>
      <td>R</td>
      <td>4</td>
      <td>3</td>
      <td>Yes</td>
      <td>26-35 km</td>
      <td>5 km</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>22</td>
      <td>181.0</td>
      <td>62.0</td>
      <td>R</td>
      <td>4</td>
      <td>3</td>
      <td>Yes</td>
      <td>26-35 km</td>
      <td>5 km</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>22</td>
      <td>181.0</td>
      <td>62.0</td>
      <td>R</td>
      <td>4</td>
      <td>3</td>
      <td>Yes</td>
      <td>26-35 km</td>
      <td>5 km</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>22</td>
      <td>181.0</td>
      <td>62.0</td>
      <td>R</td>
      <td>4</td>
      <td>3</td>
      <td>Yes</td>
      <td>26-35 km</td>
      <td>5 km</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 50 columns</p>
</div>


I will not be using the following columns for the baseline logistic regression model: subject, dominance, run_grp, volume, race_dist, diagnostic_med, diagnostic, injury_loc, injury_on_date, shoe_model, shoe_change, shoe_insert, race_dist_list.

```python
fukuchi_features = fukuchi_encoded.drop(columns = ['subject','dominance','run_grp','volume','race_dist','diagnostic_med','diagnostic', 'injury_loc', 'injury_on_date','shoe_model','shoe_change','shoe_insert','race_dist_list','injury'])

fukuchi_features.head()
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>height</th>
      <th>mass</th>
      <th>experience</th>
      <th>sessions_per_wk</th>
      <th>shoe_size</th>
      <th>shoe_pairs</th>
      <th>shoe_comfort</th>
      <th>10_km</th>
      <th>12_km</th>
      <th>...</th>
      <th>shoe_brand_Mizuno</th>
      <th>shoe_brand_Mizuno, asics</th>
      <th>shoe_brand_New Balance</th>
      <th>shoe_brand_Nike</th>
      <th>shoe_brand_Nike, Adidas</th>
      <th>shoe_brand_Puma</th>
      <th>shoe_brand_Sketchers</th>
      <th>gender_M</th>
      <th>level_Elite</th>
      <th>level_Recreational</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22</td>
      <td>181.0</td>
      <td>62.0</td>
      <td>4</td>
      <td>3</td>
      <td>10.5</td>
      <td>2</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>22</td>
      <td>181.0</td>
      <td>62.0</td>
      <td>4</td>
      <td>3</td>
      <td>10.5</td>
      <td>2</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>22</td>
      <td>181.0</td>
      <td>62.0</td>
      <td>4</td>
      <td>3</td>
      <td>10.5</td>
      <td>2</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>22</td>
      <td>181.0</td>
      <td>62.0</td>
      <td>4</td>
      <td>3</td>
      <td>10.5</td>
      <td>2</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>22</td>
      <td>181.0</td>
      <td>62.0</td>
      <td>4</td>
      <td>3</td>
      <td>10.5</td>
      <td>2</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 36 columns</p>
</div>


```python
X_log_regression = fukuchi_features
y_log_regression = fukuchi_data['injury']

log_regression1 = LogisticRegression(max_iter=5000)

X_train, X_test, y_train, y_test = train_test_split(X_log_regression, y_log_regression, test_size=0.2, random_state=65)

log_regression1.fit(X_train, y_train)

y_pred = log_regression1.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Test Accuracy:", acc)
```

    Test Accuracy: 0.9285714285714286

The baseline model is able to make accurate predictions for around 93% of the dataset.

```python
y_pred_log = log_regression1.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:\n", cm)
```

    Confusion Matrix:
     [[52  6]
     [ 0 26]]

```python
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               0       1.00      0.90      0.95        58
               1       0.81      1.00      0.90        26
    
        accuracy                           0.93        84
       macro avg       0.91      0.95      0.92        84
    weighted avg       0.94      0.93      0.93        84
    

The model has a recall of 1 for positive injury cases. Hence, the model is excellent at identifying injury cases. The precision of the model at identifying injury cases is at 0.81, meaning that it has around a 20% false positive rate.

```python
from sklearn.metrics import roc_curve, auc

y_pred_proba = log_regression1.predict_proba(X_test)[:, 1]
y_test_roc = y_test.astype(int)
fpr, tpr, thresholds = roc_curve(y_test_roc, y_pred_proba)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0,1], [0,1], color='gray', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Baseline Logistic Regression (Fukuchi dataset)")
plt.legend(loc="lower right")
plt.show()


```

![png]({{ site.baseurl }}/capstone_model_files/capstone_model/capstone_model_32_0.png){: .center-image }

```python
feature_importance = pd.Series(log_regression1.coef_[0], index=X_log_regression.columns)
feature_importance.sort_values(ascending=False).head(10)
```




    42_km                               2.722640
    shoe_brand_Asics, Adidas, Adidas    2.156361
    15_km                               1.248164
    shoe_brand_Mizuno, asics            1.248164
    12_km                               0.801238
    shoe_brand_Sketchers                0.613535
    speed                               0.206381
    height                              0.185280
    age                                 0.054238
    gender_M                            0.051899
    dtype: float64



According to the model, the most important features that cause injury are distance, shoe brand, speed, height and gender.

```python
ric_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3806 entries, 0 to 3805
    Data columns (total 16 columns):
     #   Column           Non-Null Count  Dtype  
    ---  ------           --------------  -----  
     0   sub_id           3806 non-null   int64  
     1   datestring       3806 non-null   object 
     2   filename         3806 non-null   object 
     3   age              3806 non-null   int64  
     4   height           3806 non-null   float64
     5   weight           3806 non-null   float64
     6   gender           3806 non-null   object 
     7   inj_def          3806 non-null   object 
     8   inj_joint        3806 non-null   object 
     9   spec_injury      3806 non-null   object 
     10  activities       3806 non-null   object 
     11  level            3806 non-null   object 
     12  yrs_running      3806 non-null   float64
     13  race_dist        3806 non-null   object 
     14  speed            3806 non-null   float64
     15  activity_groups  3806 non-null   object 
    dtypes: float64(4), int64(2), object(10)
    memory usage: 475.9+ KB

```python
ric_data.drop(columns='filename',inplace=True)
```

```python
#convert m/s to km/h
ric_data["speed"] = ric_data["speed"] * 3.6
```

```python
ric_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3806 entries, 0 to 3805
    Data columns (total 15 columns):
     #   Column           Non-Null Count  Dtype  
    ---  ------           --------------  -----  
     0   sub_id           3806 non-null   int64  
     1   datestring       3806 non-null   object 
     2   age              3806 non-null   int64  
     3   height           3806 non-null   float64
     4   weight           3806 non-null   float64
     5   gender           3806 non-null   object 
     6   inj_def          3806 non-null   object 
     7   inj_joint        3806 non-null   object 
     8   spec_injury      3806 non-null   object 
     9   activities       3806 non-null   object 
     10  level            3806 non-null   object 
     11  yrs_running      3806 non-null   float64
     12  race_dist        3806 non-null   object 
     13  speed            3806 non-null   float64
     14  activity_groups  3806 non-null   object 
    dtypes: float64(4), int64(2), object(9)
    memory usage: 446.1+ KB

```python
ric_data['activity_groups'].unique()
```




    array(["['Walking', 'Light exercise']", "['Light exercise']",
           "['Low activity']", "['Cycling']", "['Triathlon']", "['Running']",
           "['Sports']", "['Running', 'Walking']",
           "['Cycling', 'Light exercise']", "['Running', 'Light exercise']",
           "['Running', 'Walking', 'Cycling']", "['Running', 'Cycling']",
           "['Walking']", "['Running', 'Cycling', 'Light exercise']",
           "['Running', 'Cycling', 'Triathlon']",
           "['Running', 'Walking', 'Light exercise']",
           "['Cycling', 'Triathlon']", "['Walking', 'Cycling']",
           "['Low activity', 'Running', 'Walking']",
           "['Low activity', 'Cycling', 'Light exercise']",
           "['Low activity', 'Cycling']",
           "['Low activity', 'Cycling', 'Triathlon']",
           "['Walking', 'Cycling', 'Light exercise']",
           "['Running', 'Triathlon']", "['Triathlon', 'Light exercise']",
           "['Running', 'Cycling', 'Triathlon', 'Light exercise']",
           "['Running', 'Triathlon', 'Light exercise']",
           "['Low activity', 'Running']"], dtype=object)



```python
ric_data['inj_def'].unique()
```




    array(['2 workouts missed in a row', 'No injury',
           'Continuing to train in pain',
           'Training volume/intensity affected', 'Injured and resting'],
          dtype=object)



```python
ric_data['injury'] = ric_data['inj_def'].apply(lambda x: 0 if x == 'No injury' else 1)
```

```python
numeric_cols = ric_data.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = ['level','race_dist','activity_groups','inj_joint']
ric_categorical = ric_data[categorical_cols].columns
ric_encoded = pd.get_dummies(ric_data[ric_categorical], drop_first=True)
ric_combined = pd.concat([ric_data[numeric_cols],ric_encoded],axis=1)

corr_matrix = ric_combined.corr()

plt.figure(figsize=(32, 24))
sns.heatmap(corr_matrix, annot=True, vmin=-1, vmax=1, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Correlation Matrix of Numeric Variables')
plt.show()
```

![png]({{ site.baseurl }}/capstone_model_files/capstone_model/capstone_model_42_0.png){: .center-image }

None of the columns chosen appear to have close correlations to each other.

```python
counts = pd.crosstab (ric_data['gender'],ric_data['injury'])

counts.plot(kind='bar',width=0.8)
plt.ylabel("Number of runners")
plt.title('Injury by Gender')
plt.legend(title='Injury')
plt.show()
```

![png]({{ site.baseurl }}/capstone_model_files/capstone_model/capstone_model_44_0.png){: .center-image }

```python
activity_injury = pd.crosstab (ric_data['activity_groups'],ric_data['injury'])

activity_injury.plot(kind='bar',width=0.6)
plt.ylabel("Number of runners")
plt.title('Injury by Activity Group')
plt.legend(title='Injury')
plt.show()
```

![png]({{ site.baseurl }}/capstone_model_files/capstone_model/capstone_model_45_0.png){: .center-image }

```python
print(activity_injury.columns)
```

    Index([0, 1, 'injury_rate'], dtype='object', name='injury')

Based on the bar graph, the activity that causes the most number of injuries is purely running.

```python
activity_injury['injury_rate'] = (
    activity_injury[1] / (activity_injury[1] + activity_injury[0])
) * 100

activity_injury.sort_values(by=1, ascending=False)
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>injury</th>
      <th>0</th>
      <th>1</th>
      <th>injury_rate</th>
    </tr>
    <tr>
      <th>activity_groups</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>['Running']</th>
      <td>401</td>
      <td>750</td>
      <td>65.160730</td>
    </tr>
    <tr>
      <th>['Low activity']</th>
      <td>475</td>
      <td>418</td>
      <td>46.808511</td>
    </tr>
    <tr>
      <th>['Running', 'Cycling']</th>
      <td>89</td>
      <td>222</td>
      <td>71.382637</td>
    </tr>
    <tr>
      <th>['Running', 'Light exercise']</th>
      <td>114</td>
      <td>211</td>
      <td>64.923077</td>
    </tr>
    <tr>
      <th>['Sports']</th>
      <td>119</td>
      <td>178</td>
      <td>59.932660</td>
    </tr>
    <tr>
      <th>['Triathlon']</th>
      <td>39</td>
      <td>132</td>
      <td>77.192982</td>
    </tr>
    <tr>
      <th>['Walking']</th>
      <td>17</td>
      <td>92</td>
      <td>84.403670</td>
    </tr>
    <tr>
      <th>['Running', 'Walking']</th>
      <td>12</td>
      <td>70</td>
      <td>85.365854</td>
    </tr>
    <tr>
      <th>['Light exercise']</th>
      <td>6</td>
      <td>55</td>
      <td>90.163934</td>
    </tr>
    <tr>
      <th>['Cycling']</th>
      <td>8</td>
      <td>51</td>
      <td>86.440678</td>
    </tr>
    <tr>
      <th>['Walking', 'Light exercise']</th>
      <td>10</td>
      <td>50</td>
      <td>83.333333</td>
    </tr>
    <tr>
      <th>['Running', 'Cycling', 'Light exercise']</th>
      <td>25</td>
      <td>50</td>
      <td>66.666667</td>
    </tr>
    <tr>
      <th>['Running', 'Triathlon']</th>
      <td>12</td>
      <td>42</td>
      <td>77.777778</td>
    </tr>
    <tr>
      <th>['Walking', 'Cycling']</th>
      <td>0</td>
      <td>29</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>['Cycling', 'Light exercise']</th>
      <td>3</td>
      <td>29</td>
      <td>90.625000</td>
    </tr>
    <tr>
      <th>['Running', 'Walking', 'Light exercise']</th>
      <td>4</td>
      <td>16</td>
      <td>80.000000</td>
    </tr>
    <tr>
      <th>['Running', 'Walking', 'Cycling']</th>
      <td>4</td>
      <td>13</td>
      <td>76.470588</td>
    </tr>
    <tr>
      <th>['Running', 'Cycling', 'Triathlon']</th>
      <td>9</td>
      <td>13</td>
      <td>59.090909</td>
    </tr>
    <tr>
      <th>['Walking', 'Cycling', 'Light exercise']</th>
      <td>0</td>
      <td>7</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>['Running', 'Cycling', 'Triathlon', 'Light exercise']</th>
      <td>0</td>
      <td>5</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>['Triathlon', 'Light exercise']</th>
      <td>2</td>
      <td>5</td>
      <td>71.428571</td>
    </tr>
    <tr>
      <th>['Cycling', 'Triathlon']</th>
      <td>2</td>
      <td>4</td>
      <td>66.666667</td>
    </tr>
    <tr>
      <th>['Low activity', 'Running']</th>
      <td>0</td>
      <td>4</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>['Low activity', 'Cycling']</th>
      <td>0</td>
      <td>2</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>['Low activity', 'Running', 'Walking']</th>
      <td>0</td>
      <td>2</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>['Low activity', 'Cycling', 'Triathlon']</th>
      <td>0</td>
      <td>2</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>['Running', 'Triathlon', 'Light exercise']</th>
      <td>0</td>
      <td>2</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>['Low activity', 'Cycling', 'Light exercise']</th>
      <td>0</td>
      <td>1</td>
      <td>100.000000</td>
    </tr>
  </tbody>
</table>
</div>


From the crosstab above, I can observe that although intensive activities like running, sports and triathlons contribute to a higher injury rate than low activity, other activities like walking, light exercise and cycling also correspond to high injury rates.

```python
plt.figure(figsize=(12, 6))
sns.boxplot(x="inj_joint", y="age", data=ric_data)

plt.xticks(rotation=45)
plt.title("Age Distribution by Injury Location")
plt.show()
```

![png]({{ site.baseurl }}/capstone_model_files/capstone_model/capstone_model_50_0.png){: .center-image }

```python
plt.figure(figsize=(10,6))
sns.histplot(data=ric_data, x="yrs_running", hue="injury", multiple="stack", bins=15)

plt.title("Distribution of Injury by Years of Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Count")
plt.show()
```

![png]({{ site.baseurl }}/capstone_model_files/capstone_model/capstone_model_51_0.png){: .center-image }

From the data, it would appear that the runners with more than 10 years of experience are more likely to be injured. There may be multiple reasons for this:
1) Those with more than 10 years of running experience are likely to have sustained some form of running injury in their previous years of experience and are more susceptible to a relapse.
2) Those with more than 10 years of running experience might be more likely to stretch themselves further and attempt further running distances.

Also, from the data, it can be observed that there are the most number of runners who have less than 10 years of running experience.

When I look at the data broken down into percentages, the injury percentages for different years of experience does not appear to differ that much.

```python
ric_data['inj_joint'].value_counts()
```




    inj_joint
    No Injury           1196
    Knee                 995
    Lower Leg            336
    Thigh                311
    Foot                 272
    Hip/Pelvis           268
    Ankle                207
    Lumbar Spine          82
    Hip                   49
    Other                 45
    Sacroiliac Joint      45
    Name: count, dtype: int64



```python
ric_data['inj_joint'].replace({'Hip':'Hip/Pelvis'},inplace=True)
```

    C:\Users\benja\AppData\Local\Temp\ipykernel_7212\2597625603.py:1: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
    The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.
    
    For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.
    
    
      ric_data['inj_joint'].replace({'Hip':'Hip/Pelvis'},inplace=True)

```python
ric_data['inj_joint'].value_counts()
```




    inj_joint
    No Injury           1196
    Knee                 995
    Lower Leg            336
    Hip/Pelvis           317
    Thigh                311
    Foot                 272
    Ankle                207
    Lumbar Spine          82
    Other                 45
    Sacroiliac Joint      45
    Name: count, dtype: int64



```python
plt.figure(figsize=(12, 6))
sns.boxplot(x="inj_joint", y="age", data=ric_data)
plt.xticks(rotation=45)
plt.title("Age Distribution by Top 10 Injury Locations")
plt.show()
```

![png]({{ site.baseurl }}/capstone_model_files/capstone_model/capstone_model_57_0.png){: .center-image }

Based on the distribution, I can observe that hip/pelvis injuries have the highest median age and should be something older runners should look out for. The injury type with the largest spread is knee injuries, suggesting that knee injuries can be sustained by runners of any age.

```python
pd.crosstab(ric_data['inj_joint'],ric_data['injury'])
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>injury</th>
      <th>0</th>
      <th>1</th>
    </tr>
    <tr>
      <th>inj_joint</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ankle</th>
      <td>2</td>
      <td>205</td>
    </tr>
    <tr>
      <th>Foot</th>
      <td>7</td>
      <td>265</td>
    </tr>
    <tr>
      <th>Hip/Pelvis</th>
      <td>5</td>
      <td>312</td>
    </tr>
    <tr>
      <th>Knee</th>
      <td>113</td>
      <td>882</td>
    </tr>
    <tr>
      <th>Lower Leg</th>
      <td>9</td>
      <td>327</td>
    </tr>
    <tr>
      <th>Lumbar Spine</th>
      <td>3</td>
      <td>79</td>
    </tr>
    <tr>
      <th>No Injury</th>
      <td>1188</td>
      <td>8</td>
    </tr>
    <tr>
      <th>Other</th>
      <td>5</td>
      <td>40</td>
    </tr>
    <tr>
      <th>Sacroiliac Joint</th>
      <td>0</td>
      <td>45</td>
    </tr>
    <tr>
      <th>Thigh</th>
      <td>19</td>
      <td>292</td>
    </tr>
  </tbody>
</table>
</div>


Knee injuries are the most common occurence among runners.

```python
ric_data['age'].hist(bins=15, figsize=(10,6), color='lightgreen', edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Number of runners')
plt.title('Distribution of Age')
plt.show()
```

![png]({{ site.baseurl }}/capstone_model_files/capstone_model/capstone_model_61_0.png){: .center-image }

```python
age_range = ric_data['age'].value_counts().sort_index(ascending=False)

age_range
```




    age
    255     2
    82      1
    79      1
    77      3
    76      4
           ..
    22     82
    21     59
    20     72
    19     71
    18     72
    Name: count, Length: 63, dtype: int64



The two 255 year olds runners don't make sense in the dataset, so I will impute them to the median age.

```python
median_age = ric_data.loc[ric_data['age'] != 255, 'age'].median()

ric_data.loc[ric_data['age'] == 255, 'age'] = median_age

ric_data[ric_data['age'] == 255]
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sub_id</th>
      <th>datestring</th>
      <th>age</th>
      <th>height</th>
      <th>weight</th>
      <th>gender</th>
      <th>inj_def</th>
      <th>inj_joint</th>
      <th>spec_injury</th>
      <th>activities</th>
      <th>level</th>
      <th>yrs_running</th>
      <th>race_dist</th>
      <th>speed</th>
      <th>activity_groups</th>
      <th>injury</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>


```python
ric_data['age'].hist(bins=15, figsize=(10,6), color='skyblue', edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Number of runners')
plt.title('Distribution of Age')
plt.show()
```

![png]({{ site.baseurl }}/capstone_model_files/capstone_model/capstone_model_65_0.png){: .center-image }

```python
ric_data['race_dist'].value_counts()
```




    race_dist
    Casual Runner     2041
    Half Marathon      538
    10K                437
    Full Marathon      342
    5K                 257
    Other distance     191
    Name: count, dtype: int64



```python
ric_data['race_dist'] = ric_data['race_dist'].replace({'Casual Runner':'3_km', 'Half Marathon':'21_km', 'Full Marathon':'42_km','10K':'10_km', '5K':'5_km'})

#For prediction purposes, we will impute Casual Runner as runners who run 3km.
```

```python
ric_data['race_dist'].value_counts()
```




    race_dist
    3_km              2041
    21_km              538
    10_km              437
    42_km              342
    5_km               257
    Other distance     191
    Name: count, dtype: int64



Similar to the fukuchi dataset, I will run a logistic regression on the ric dataset to check if there are certain features that are particularly important to determining injury.

```python
race_dist_encoded = pd.get_dummies(ric_data['race_dist'])

race_dist_encoded

ric_data_merge = pd.concat([ric_data, race_dist_encoded], axis=1)
```

```python
ric_data_merge.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3806 entries, 0 to 3805
    Data columns (total 22 columns):
     #   Column           Non-Null Count  Dtype  
    ---  ------           --------------  -----  
     0   sub_id           3806 non-null   int64  
     1   datestring       3806 non-null   object 
     2   age              3806 non-null   int64  
     3   height           3806 non-null   float64
     4   weight           3806 non-null   float64
     5   gender           3806 non-null   object 
     6   inj_def          3806 non-null   object 
     7   inj_joint        3806 non-null   object 
     8   spec_injury      3806 non-null   object 
     9   activities       3806 non-null   object 
     10  level            3806 non-null   object 
     11  yrs_running      3806 non-null   float64
     12  race_dist        3806 non-null   object 
     13  speed            3806 non-null   float64
     14  activity_groups  3806 non-null   object 
     15  injury           3806 non-null   int64  
     16  10_km            3806 non-null   bool   
     17  21_km            3806 non-null   bool   
     18  3_km             3806 non-null   bool   
     19  42_km            3806 non-null   bool   
     20  5_km             3806 non-null   bool   
     21  Other distance   3806 non-null   bool   
    dtypes: bool(6), float64(4), int64(3), object(9)
    memory usage: 498.2+ KB

The features that I will use for the logistic regression will be: age, height, weight, gender, level, yrs_running, speed race_dist(encoded). The other features that are not used are either not relevant i.e. sub_id (unique identifier), datestring (date) or are directly connected to injury like inj_def and inj_joint.

```python
ric_data_encoded = pd.get_dummies(ric_data, columns=['gender','activity_groups','level'], drop_first=True)

ric_data_encoded.head()
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sub_id</th>
      <th>datestring</th>
      <th>age</th>
      <th>height</th>
      <th>weight</th>
      <th>inj_def</th>
      <th>inj_joint</th>
      <th>spec_injury</th>
      <th>activities</th>
      <th>yrs_running</th>
      <th>...</th>
      <th>activity_groups_['Running', 'Walking']</th>
      <th>activity_groups_['Running']</th>
      <th>activity_groups_['Sports']</th>
      <th>activity_groups_['Triathlon', 'Light exercise']</th>
      <th>activity_groups_['Triathlon']</th>
      <th>activity_groups_['Walking', 'Cycling', 'Light exercise']</th>
      <th>activity_groups_['Walking', 'Cycling']</th>
      <th>activity_groups_['Walking', 'Light exercise']</th>
      <th>activity_groups_['Walking']</th>
      <th>level_Recreational</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100537</td>
      <td>2012-07-03 10:25</td>
      <td>40</td>
      <td>173.0</td>
      <td>68.0</td>
      <td>2 workouts missed in a row</td>
      <td>Hip/Pelvis</td>
      <td>other</td>
      <td>hiking, power walking, pilates</td>
      <td>2.0</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100560</td>
      <td>2012-07-17 10:37</td>
      <td>33</td>
      <td>179.0</td>
      <td>83.0</td>
      <td>No injury</td>
      <td>No Injury</td>
      <td>No injury</td>
      <td>Yoga</td>
      <td>15.0</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>101481</td>
      <td>2012-07-17 10:50</td>
      <td>32</td>
      <td>176.0</td>
      <td>59.0</td>
      <td>No injury</td>
      <td>No Injury</td>
      <td>No injury</td>
      <td>No activity</td>
      <td>1.0</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100591</td>
      <td>2012-08-09 10:01</td>
      <td>51</td>
      <td>173.0</td>
      <td>67.0</td>
      <td>Continuing to train in pain</td>
      <td>Hip/Pelvis</td>
      <td>pain</td>
      <td>cycling</td>
      <td>1.0</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100595</td>
      <td>2012-08-29 12:56</td>
      <td>50</td>
      <td>155.0</td>
      <td>64.0</td>
      <td>2 workouts missed in a row</td>
      <td>Lower Leg</td>
      <td>calf muscle strain</td>
      <td>triathlon</td>
      <td>30.0</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 42 columns</p>
</div>


```python
ric_log_features = ric_data_encoded.drop(columns=['sub_id','datestring','inj_def','inj_joint','spec_injury','activities','race_dist','injury'])

ric_log_features.head()
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>height</th>
      <th>weight</th>
      <th>yrs_running</th>
      <th>speed</th>
      <th>gender_Male</th>
      <th>activity_groups_['Cycling', 'Triathlon']</th>
      <th>activity_groups_['Cycling']</th>
      <th>activity_groups_['Light exercise']</th>
      <th>activity_groups_['Low activity', 'Cycling', 'Light exercise']</th>
      <th>...</th>
      <th>activity_groups_['Running', 'Walking']</th>
      <th>activity_groups_['Running']</th>
      <th>activity_groups_['Sports']</th>
      <th>activity_groups_['Triathlon', 'Light exercise']</th>
      <th>activity_groups_['Triathlon']</th>
      <th>activity_groups_['Walking', 'Cycling', 'Light exercise']</th>
      <th>activity_groups_['Walking', 'Cycling']</th>
      <th>activity_groups_['Walking', 'Light exercise']</th>
      <th>activity_groups_['Walking']</th>
      <th>level_Recreational</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>40</td>
      <td>173.0</td>
      <td>68.0</td>
      <td>2.0</td>
      <td>7.658787</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33</td>
      <td>179.0</td>
      <td>83.0</td>
      <td>15.0</td>
      <td>9.566515</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>32</td>
      <td>176.0</td>
      <td>59.0</td>
      <td>1.0</td>
      <td>9.450318</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>51</td>
      <td>173.0</td>
      <td>67.0</td>
      <td>1.0</td>
      <td>8.034622</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>50</td>
      <td>155.0</td>
      <td>64.0</td>
      <td>30.0</td>
      <td>8.006290</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 34 columns</p>
</div>


```python
X_log_regression2 = ric_log_features
y_log_regression2 = ric_data['injury']

log_regression2 = LogisticRegression(max_iter=5000)

X_train_ric, X_test_ric, y_train_ric, y_test_ric = train_test_split(X_log_regression2, y_log_regression2, test_size=0.2, random_state=65)

log_regression2.fit(X_train_ric, y_train_ric)

y_pred_ric = log_regression2.predict(X_test_ric)
acc2 = accuracy_score(y_test_ric, y_pred_ric)
print("Test Accuracy:", acc2)
```

    Test Accuracy: 0.7257217847769029

The logistic regression model for this dataset is significantly lower than the previous dataset, so I might need to use another machine learning model for it.

```python
y_pred_log2 = log_regression2.predict(X_test_ric)

cm2 = confusion_matrix(y_test_ric, y_pred_ric)

print("Confusion Matrix:\n", cm2)
```

    Confusion Matrix:
     [[101 174]
     [ 35 452]]

The logistic regression model appears to spot a lot of false positives, however its specificity is quite high as it has a rather low false negative rate.

```python
print(classification_report(y_test_ric, y_pred_ric))
```

                  precision    recall  f1-score   support
    
               0       0.74      0.37      0.49       275
               1       0.72      0.93      0.81       487
    
        accuracy                           0.73       762
       macro avg       0.73      0.65      0.65       762
    weighted avg       0.73      0.73      0.70       762
    

```python
feature_importance2 = pd.Series(log_regression2.coef_[0], index=X_log_regression2.columns)
feature_importance2.sort_values(ascending=False).head(10)
```




    activity_groups_['Walking', 'Cycling']                                   1.174061
    activity_groups_['Low activity', 'Running']                              0.668419
    activity_groups_['Running', 'Cycling', 'Triathlon', 'Light exercise']    0.642370
    activity_groups_['Light exercise']                                       0.635129
    level_Recreational                                                       0.473637
    activity_groups_['Walking', 'Cycling', 'Light exercise']                 0.471864
    activity_groups_['Cycling']                                              0.418957
    activity_groups_['Running', 'Triathlon']                                 0.400478
    activity_groups_['Low activity', 'Cycling', 'Triathlon']                 0.269539
    activity_groups_['Running', 'Triathlon', 'Light exercise']               0.239146
    dtype: float64



From the feature importance, it is observed that the types of activities that a runner engages in is a key indicator of injury possibility. While this may be true, it may have overshadowed other features and may indicate that the logistic regression may not be the best model.

```python
y_pred_proba_ric = log_regression2.predict_proba(X_test_ric)[:, 1]
y_test_roc_ric = y_test_ric.astype(int)
fpr_ric, tpr_ric, thresholds_ric = roc_curve(y_test_roc_ric, y_pred_proba_ric)

roc_auc_ric = auc(fpr_ric, tpr_ric)

plt.figure(figsize=(6,6))
plt.plot(fpr_ric, tpr_ric, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc_ric:.2f})')
plt.plot([0,1], [0,1], color='gray', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Logistic Regression (RIC dataset)")
plt.legend(loc="lower right")
plt.show()
```

![png]({{ site.baseurl }}/capstone_model_files/capstone_model/capstone_model_82_0.png){: .center-image }

The false positive rate is a bit too high for the model and it may classify potential injury even though there may not be any injury. This would dissuade runners who can potentially do more. Hence, I will look at other classification models to improve the accuracy of the predictions.

```python
X_rf_ric = ric_log_features 
y_rf_ric = ric_data['injury'] 

X_train_rf_ric, X_test_rf_ric, y_train_rf_ric, y_test_rf_ric = train_test_split(X_rf_ric, y_rf_ric, test_size=0.2, random_state=65)

rf_ric = RandomForestClassifier(
    n_estimators=200,    
    max_depth=None,   
    random_state=65,
    class_weight="balanced"
)

rf_ric.fit(X_train_rf_ric, y_train_rf_ric)

y_pred_rf_ric = rf_ric.predict(X_test_rf_ric)
acc_rf_ric = accuracy_score(y_test_rf_ric, y_pred_rf_ric)
print("Random Forest Classifier Accuracy:", acc_rf_ric)
```

    Random Forest Classifier Accuracy: 0.8070866141732284

The Random Forest model has a higher accuracy at around 80%, however I will need to explore the confusion matrix to make sure that the overall performance has improved.

```python
print("Classification Report:\n", classification_report(y_test_rf_ric, y_pred_rf_ric))
print("Confusion Matrix:\n", confusion_matrix(y_test_rf_ric, y_pred_rf_ric))
```

    Classification Report:
                   precision    recall  f1-score   support
    
               0       0.84      0.58      0.68       275
               1       0.80      0.94      0.86       487
    
        accuracy                           0.81       762
       macro avg       0.82      0.76      0.77       762
    weighted avg       0.81      0.81      0.80       762
    
    Confusion Matrix:
     [[159 116]
     [ 31 456]]

The number of false positives has decreased significantly in the Random Forest Classifier model.

```python
y_pred_proba_rf_ric = rf_ric.predict_proba(X_test_rf_ric)[:, 1]

fpr_rf_ric, tpr_rf_ric, thresholds_rf_ric = roc_curve(y_test_rf_ric.astype(int), y_pred_proba_rf_ric)
roc_auc_rf_ric = auc(fpr_rf_ric, tpr_rf_ric)

plt.figure(figsize=(6,6))
plt.plot(fpr_ric, tpr_ric, color='blue', lw=2, label=f'Log Regression ROC curve (AUC = {roc_auc_ric:.2f})')
plt.plot(fpr_rf_ric, tpr_rf_ric, color='green', lw=2, 
         label=f'Random Forest ROC curve (AUC = {roc_auc_rf_ric:.2f})')
plt.plot([0,1], [0,1], color='gray', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison (RIC dataset)")
plt.legend(loc="lower right")
plt.show()
```

![png]({{ site.baseurl }}/capstone_model_files/capstone_model/capstone_model_88_0.png){: .center-image }

```python
feature_importance_rf_ric = pd.DataFrame({
    "feature": X_rf_ric.columns,
    "importance": rf_ric.feature_importances_
})

feature_importance_rf_ric = feature_importance_rf_ric.sort_values(
    by="importance", ascending=False
)

feature_importance_rf_ric.head(10)
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>speed</td>
      <td>0.214625</td>
    </tr>
    <tr>
      <th>0</th>
      <td>age</td>
      <td>0.177905</td>
    </tr>
    <tr>
      <th>2</th>
      <td>weight</td>
      <td>0.144954</td>
    </tr>
    <tr>
      <th>1</th>
      <td>height</td>
      <td>0.134976</td>
    </tr>
    <tr>
      <th>3</th>
      <td>yrs_running</td>
      <td>0.125735</td>
    </tr>
    <tr>
      <th>14</th>
      <td>activity_groups_['Low activity']</td>
      <td>0.050157</td>
    </tr>
    <tr>
      <th>33</th>
      <td>level_Recreational</td>
      <td>0.025104</td>
    </tr>
    <tr>
      <th>5</th>
      <td>gender_Male</td>
      <td>0.019427</td>
    </tr>
    <tr>
      <th>25</th>
      <td>activity_groups_['Running']</td>
      <td>0.017838</td>
    </tr>
    <tr>
      <th>19</th>
      <td>activity_groups_['Running', 'Light exercise']</td>
      <td>0.012850</td>
    </tr>
  </tbody>
</table>
</div>


The Random Forest model has a more balanced approach towards the most important features, with the top 5 being key features in the dataset and also provides support towards the features that I have selected for my machine learning models.

```python
#This cell works in Marimo but not Jupyter Notebook
param_distribution = {
    "n_estimators": [100, 200, 500],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
    "class_weight": [None, "balanced"]
}

rf_ric_tuning = RandomForestClassifier(random_state=65)

random_search = RandomizedSearchCV(
    estimator=rf_ric_tuning,
    param_distributions=param_distribution,
    n_iter=200,  
    cv=5, 
    scoring="roc_auc", 
    n_jobs=-1,
    random_state=65
)

random_search.fit(X_rf_ric, y_rf_ric)

print("Best Parameters:", random_search.best_params_)
print("Best CV Score:", random_search.best_score_)
```


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    Cell In[85], line 22
         10 rf_ric_tuning = RandomForestClassifier(random_state=65)
         12 random_search = RandomizedSearchCV(
         13     estimator=rf_ric_tuning,
         14     param_distributions=param_distribution,
       (...)     19     random_state=65
         20 )
    ---> 22 random_search.fit(X_rf_ric, y_rf_ric)
         24 print("Best Parameters:", random_search.best_params_)
         25 print("Best CV Score:", random_search.best_score_)
    

    File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\sklearn\base.py:1389, in _fit_context.<locals>.decorator.<locals>.wrapper(estimator, *args, **kwargs)
       1382     estimator._validate_params()
       1384 with config_context(
       1385     skip_parameter_validation=(
       1386         prefer_skip_nested_validation or global_skip_validation
       1387     )
       1388 ):
    -> 1389     return fit_method(estimator, *args, **kwargs)
    

    File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\sklearn\model_selection\_search.py:1024, in BaseSearchCV.fit(self, X, y, **params)
       1018     results = self._format_results(
       1019         all_candidate_params, n_splits, all_out, all_more_results
       1020     )
       1022     return results
    -> 1024 self._run_search(evaluate_candidates)
       1026 # multimetric is determined here because in the case of a callable
       1027 # self.scoring the return type is only known after calling
       1028 first_test_score = all_out[0]["test_scores"]
    

    File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\sklearn\model_selection\_search.py:1951, in RandomizedSearchCV._run_search(self, evaluate_candidates)
       1949 def _run_search(self, evaluate_candidates):
       1950     """Search n_iter candidates from param_distributions"""
    -> 1951     evaluate_candidates(
       1952         ParameterSampler(
       1953             self.param_distributions, self.n_iter, random_state=self.random_state
       1954         )
       1955     )
    

    File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\sklearn\model_selection\_search.py:970, in BaseSearchCV.fit.<locals>.evaluate_candidates(candidate_params, cv, more_results)
        962 if self.verbose > 0:
        963     print(
        964         "Fitting {0} folds for each of {1} candidates,"
        965         " totalling {2} fits".format(
        966             n_splits, n_candidates, n_candidates * n_splits
        967         )
        968     )
    --> 970 out = parallel(
        971     delayed(_fit_and_score)(
        972         clone(base_estimator),
        973         X,
        974         y,
        975         train=train,
        976         test=test,
        977         parameters=parameters,
        978         split_progress=(split_idx, n_splits),
        979         candidate_progress=(cand_idx, n_candidates),
        980         **fit_and_score_kwargs,
        981     )
        982     for (cand_idx, parameters), (split_idx, (train, test)) in product(
        983         enumerate(candidate_params),
        984         enumerate(cv.split(X, y, **routed_params.splitter.split)),
        985     )
        986 )
        988 if len(out) < 1:
        989     raise ValueError(
        990         "No fits were performed. "
        991         "Was the CV iterator empty? "
        992         "Were there no candidates?"
        993     )
    

    File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\sklearn\utils\parallel.py:77, in Parallel.__call__(self, iterable)
         72 config = get_config()
         73 iterable_with_config = (
         74     (_with_config(delayed_func, config), args, kwargs)
         75     for delayed_func, args, kwargs in iterable
         76 )
    ---> 77 return super().__call__(iterable_with_config)
    

    File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\joblib\parallel.py:1986, in Parallel.__call__(self, iterable)
       1984     output = self._get_sequential_output(iterable)
       1985     next(output)
    -> 1986     return output if self.return_generator else list(output)
       1988 # Let's create an ID that uniquely identifies the current call. If the
       1989 # call is interrupted early and that the same instance is immediately
       1990 # reused, this id will be used to prevent workers that were
       1991 # concurrently finalizing a task from the previous call to run the
       1992 # callback.
       1993 with self._lock:
    

    File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\joblib\parallel.py:1914, in Parallel._get_sequential_output(self, iterable)
       1912 self.n_dispatched_batches += 1
       1913 self.n_dispatched_tasks += 1
    -> 1914 res = func(*args, **kwargs)
       1915 self.n_completed_tasks += 1
       1916 self.print_progress()
    

    File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\sklearn\utils\parallel.py:139, in _FuncWrapper.__call__(self, *args, **kwargs)
        137     config = {}
        138 with config_context(**config):
    --> 139     return self.function(*args, **kwargs)
    

    File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\sklearn\model_selection\_validation.py:866, in _fit_and_score(estimator, X, y, scorer, train, test, verbose, parameters, fit_params, score_params, return_train_score, return_parameters, return_n_test_samples, return_times, return_estimator, split_progress, candidate_progress, error_score)
        864         estimator.fit(X_train, **fit_params)
        865     else:
    --> 866         estimator.fit(X_train, y_train, **fit_params)
        868 except Exception:
        869     # Note fit time as time until error
        870     fit_time = time.time() - start_time
    

    File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\sklearn\base.py:1389, in _fit_context.<locals>.decorator.<locals>.wrapper(estimator, *args, **kwargs)
       1382     estimator._validate_params()
       1384 with config_context(
       1385     skip_parameter_validation=(
       1386         prefer_skip_nested_validation or global_skip_validation
       1387     )
       1388 ):
    -> 1389     return fit_method(estimator, *args, **kwargs)
    

    File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\sklearn\ensemble\_forest.py:487, in BaseForest.fit(self, X, y, sample_weight)
        476 trees = [
        477     self._make_estimator(append=False, random_state=random_state)
        478     for i in range(n_more_estimators)
        479 ]
        481 # Parallel loop: we prefer the threading backend as the Cython code
        482 # for fitting the trees is internally releasing the Python GIL
        483 # making threading more efficient than multiprocessing in
        484 # that case. However, for joblib 0.12+ we respect any
        485 # parallel_backend contexts set at a higher level,
        486 # since correctness does not rely on using threads.
    --> 487 trees = Parallel(
        488     n_jobs=self.n_jobs,
        489     verbose=self.verbose,
        490     prefer="threads",
        491 )(
        492     delayed(_parallel_build_trees)(
        493         t,
        494         self.bootstrap,
        495         X,
        496         y,
        497         sample_weight,
        498         i,
        499         len(trees),
        500         verbose=self.verbose,
        501         class_weight=self.class_weight,
        502         n_samples_bootstrap=n_samples_bootstrap,
        503         missing_values_in_feature_mask=missing_values_in_feature_mask,
        504     )
        505     for i, t in enumerate(trees)
        506 )
        508 # Collect newly grown trees
        509 self.estimators_.extend(trees)
    

    File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\sklearn\utils\parallel.py:77, in Parallel.__call__(self, iterable)
         72 config = get_config()
         73 iterable_with_config = (
         74     (_with_config(delayed_func, config), args, kwargs)
         75     for delayed_func, args, kwargs in iterable
         76 )
    ---> 77 return super().__call__(iterable_with_config)
    

    File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\joblib\parallel.py:1986, in Parallel.__call__(self, iterable)
       1984     output = self._get_sequential_output(iterable)
       1985     next(output)
    -> 1986     return output if self.return_generator else list(output)
       1988 # Let's create an ID that uniquely identifies the current call. If the
       1989 # call is interrupted early and that the same instance is immediately
       1990 # reused, this id will be used to prevent workers that were
       1991 # concurrently finalizing a task from the previous call to run the
       1992 # callback.
       1993 with self._lock:
    

    File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\joblib\parallel.py:1914, in Parallel._get_sequential_output(self, iterable)
       1912 self.n_dispatched_batches += 1
       1913 self.n_dispatched_tasks += 1
    -> 1914 res = func(*args, **kwargs)
       1915 self.n_completed_tasks += 1
       1916 self.print_progress()
    

    File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\sklearn\utils\parallel.py:139, in _FuncWrapper.__call__(self, *args, **kwargs)
        137     config = {}
        138 with config_context(**config):
    --> 139     return self.function(*args, **kwargs)
    

    File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\sklearn\ensemble\_forest.py:189, in _parallel_build_trees(tree, bootstrap, X, y, sample_weight, tree_idx, n_trees, verbose, class_weight, n_samples_bootstrap, missing_values_in_feature_mask)
        186     elif class_weight == "balanced_subsample":
        187         curr_sample_weight *= compute_sample_weight("balanced", y, indices=indices)
    --> 189     tree._fit(
        190         X,
        191         y,
        192         sample_weight=curr_sample_weight,
        193         check_input=False,
        194         missing_values_in_feature_mask=missing_values_in_feature_mask,
        195     )
        196 else:
        197     tree._fit(
        198         X,
        199         y,
       (...)    202         missing_values_in_feature_mask=missing_values_in_feature_mask,
        203     )
    

    File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\sklearn\tree\_classes.py:472, in BaseDecisionTree._fit(self, X, y, sample_weight, check_input, missing_values_in_feature_mask)
        461 else:
        462     builder = BestFirstTreeBuilder(
        463         splitter,
        464         min_samples_split,
       (...)    469         self.min_impurity_decrease,
        470     )
    --> 472 builder.build(self.tree_, X, y, sample_weight, missing_values_in_feature_mask)
        474 if self.n_outputs_ == 1 and is_classifier(self):
        475     self.n_classes_ = self.n_classes_[0]
    

    KeyboardInterrupt: 


```python
X_train_rf_ric2, X_test_rf_ric2, y_train_rf_ric2, y_test_rf_ric2 = train_test_split(X_rf_ric, y_rf_ric, test_size=0.2, random_state=65)

rf_ric2 = RandomForestClassifier(
    n_estimators=100,    
    max_depth=30,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='log2',
    random_state=65,
    class_weight="balanced"
)

rf_ric2.fit(X_train_rf_ric2, y_train_rf_ric2)

y_pred_rf_ric2 = rf_ric2.predict(X_test_rf_ric2)
acc_rf_ric2 = accuracy_score(y_test_rf_ric2, y_pred_rf_ric2)
print("Random Forest Classifier Accuracy:", acc_rf_ric2)
```

    Random Forest Classifier Accuracy: 0.8031496062992126

```python
print("Classification Report:\n", classification_report(y_test_rf_ric2, y_pred_rf_ric2))
print("Confusion Matrix:\n", confusion_matrix(y_test_rf_ric2, y_pred_rf_ric2))
```

    Classification Report:
                   precision    recall  f1-score   support
    
               0       0.82      0.58      0.68       275
               1       0.80      0.93      0.86       487
    
        accuracy                           0.80       762
       macro avg       0.81      0.75      0.77       762
    weighted avg       0.81      0.80      0.79       762
    
    Confusion Matrix:
     [[160 115]
     [ 35 452]]

There is no significant improvement in the confusion matrix.

```python
y_pred_proba_ric2 = rf_ric2.predict_proba(X_test_rf_ric2)[:, 1]
y_test_roc_ric2 = y_test_rf_ric2.astype(int)
fpr_ric2, tpr_ric2, _ = roc_curve(y_test_roc_ric2, y_pred_proba_ric2)
roc_auc_ric2 = auc(fpr_ric2, tpr_ric2)

plt.figure(figsize=(6,6))
plt.plot(fpr_rf_ric, tpr_rf_ric, color='green', lw=2, 
         label=f'Random Forest ROC curve (Baseline) = {roc_auc_rf_ric:.2f})')
plt.plot(fpr_ric2, tpr_ric2, color='blue', lw=2,
         label=f'Random Forest ROC curve (Optimized) = {roc_auc_ric2:.2f})')
plt.plot([0,1], [0,1], color='gray', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Random Forest (RIC dataset)")
plt.legend(loc="lower right")
plt.show()
```

![png]({{ site.baseurl }}/capstone_model_files/capstone_model/capstone_model_95_0.png){: .center-image }

The optimized Random Forest Classifier model does not show significant improvements, so it might be better to stick with the original model or optimize using another machine learning model after merging the two datasets.

```python
fukuchi_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 420 entries, 0 to 419
    Data columns (total 36 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   subject           420 non-null    int64  
     1   age               420 non-null    int64  
     2   height            420 non-null    float64
     3   mass              420 non-null    float64
     4   gender            420 non-null    object 
     5   dominance         420 non-null    object 
     6   level             420 non-null    object 
     7   experience        420 non-null    int64  
     8   sessions_per_wk   420 non-null    int64  
     9   run_grp           420 non-null    object 
     10  volume            420 non-null    object 
     11  race_dist         408 non-null    object 
     12  injury            420 non-null    object 
     13  injury_loc        420 non-null    object 
     14  diagnostic_med    420 non-null    object 
     15  diagnostic        420 non-null    object 
     16  injury_on_date    420 non-null    object 
     17  shoe_size         420 non-null    float64
     18  shoe_brand        420 non-null    object 
     19  shoe_model        420 non-null    object 
     20  shoe_pairs        420 non-null    int64  
     21  shoe_change       420 non-null    object 
     22  shoe_comfort      420 non-null    int64  
     23  shoe_insert       420 non-null    object 
     24  race_dist_list    420 non-null    object 
     25  10_km             420 non-null    int64  
     26  12_km             420 non-null    int64  
     27  15_km             420 non-null    int64  
     28  21_km             420 non-null    int64  
     29  42_km             420 non-null    int64  
     30  5_km              420 non-null    int64  
     31  6_km              420 non-null    int64  
     32  nan               420 non-null    int64  
     33  ultra (50-70 km)  420 non-null    int64  
     34  ultra_42_km       420 non-null    int64  
     35  speed             420 non-null    float64
    dtypes: float64(4), int64(16), object(16)
    memory usage: 118.3+ KB

```python
fukuchi_data['injury'] = fukuchi_data['injury'].astype(int)
```

```python
fukuchi_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 420 entries, 0 to 419
    Data columns (total 36 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   subject           420 non-null    int64  
     1   age               420 non-null    int64  
     2   height            420 non-null    float64
     3   mass              420 non-null    float64
     4   gender            420 non-null    object 
     5   dominance         420 non-null    object 
     6   level             420 non-null    object 
     7   experience        420 non-null    int64  
     8   sessions_per_wk   420 non-null    int64  
     9   run_grp           420 non-null    object 
     10  volume            420 non-null    object 
     11  race_dist         408 non-null    object 
     12  injury            420 non-null    int64  
     13  injury_loc        420 non-null    object 
     14  diagnostic_med    420 non-null    object 
     15  diagnostic        420 non-null    object 
     16  injury_on_date    420 non-null    object 
     17  shoe_size         420 non-null    float64
     18  shoe_brand        420 non-null    object 
     19  shoe_model        420 non-null    object 
     20  shoe_pairs        420 non-null    int64  
     21  shoe_change       420 non-null    object 
     22  shoe_comfort      420 non-null    int64  
     23  shoe_insert       420 non-null    object 
     24  race_dist_list    420 non-null    object 
     25  10_km             420 non-null    int64  
     26  12_km             420 non-null    int64  
     27  15_km             420 non-null    int64  
     28  21_km             420 non-null    int64  
     29  42_km             420 non-null    int64  
     30  5_km              420 non-null    int64  
     31  6_km              420 non-null    int64  
     32  nan               420 non-null    int64  
     33  ultra (50-70 km)  420 non-null    int64  
     34  ultra_42_km       420 non-null    int64  
     35  speed             420 non-null    float64
    dtypes: float64(4), int64(17), object(15)
    memory usage: 118.3+ KB

```python
ric_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3806 entries, 0 to 3805
    Data columns (total 16 columns):
     #   Column           Non-Null Count  Dtype  
    ---  ------           --------------  -----  
     0   sub_id           3806 non-null   int64  
     1   datestring       3806 non-null   object 
     2   age              3806 non-null   int64  
     3   height           3806 non-null   float64
     4   weight           3806 non-null   float64
     5   gender           3806 non-null   object 
     6   inj_def          3806 non-null   object 
     7   inj_joint        3806 non-null   object 
     8   spec_injury      3806 non-null   object 
     9   activities       3806 non-null   object 
     10  level            3806 non-null   object 
     11  yrs_running      3806 non-null   float64
     12  race_dist        3806 non-null   object 
     13  speed            3806 non-null   float64
     14  activity_groups  3806 non-null   object 
     15  injury           3806 non-null   int64  
    dtypes: float64(4), int64(3), object(9)
    memory usage: 475.9+ KB

```python
ric_data['yrs_running'].unique()
```




    array([ 2.  , 15.  ,  1.  , 30.  ,  7.  , 20.  ,  8.  , 10.  , 11.  ,
           12.  , 45.  ,  6.  ,  3.  ,  5.  , 13.  , 19.  ,  4.  , 28.  ,
           14.  ,  9.  , 16.  , 24.  , 18.  , 35.  , 25.  , 31.  , 26.  ,
           22.  , 37.  , 34.  , 21.  , 40.  , 27.  , 41.  , 36.  , 29.  ,
            0.  , 47.  , 17.  , 38.  , 23.  , 32.  , 50.  ,  1.5 ,  3.5 ,
           46.  ,  2.5 , 39.  ,  0.5 ,  4.5 , 43.  , 54.  , 33.  ,  0.25])



```python
ric_data['yrs_running'] = ric_data['yrs_running'].astype(int)
```

```python
ric_data.head()
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sub_id</th>
      <th>datestring</th>
      <th>age</th>
      <th>height</th>
      <th>weight</th>
      <th>gender</th>
      <th>inj_def</th>
      <th>inj_joint</th>
      <th>spec_injury</th>
      <th>activities</th>
      <th>level</th>
      <th>yrs_running</th>
      <th>race_dist</th>
      <th>speed</th>
      <th>activity_groups</th>
      <th>injury</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100537</td>
      <td>2012-07-03 10:25</td>
      <td>40</td>
      <td>173.0</td>
      <td>68.0</td>
      <td>Female</td>
      <td>2 workouts missed in a row</td>
      <td>Hip/Pelvis</td>
      <td>other</td>
      <td>hiking, power walking, pilates</td>
      <td>Recreational</td>
      <td>2</td>
      <td>3_km</td>
      <td>7.658787</td>
      <td>['Walking', 'Light exercise']</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100560</td>
      <td>2012-07-17 10:37</td>
      <td>33</td>
      <td>179.0</td>
      <td>83.0</td>
      <td>Female</td>
      <td>No injury</td>
      <td>No Injury</td>
      <td>No injury</td>
      <td>Yoga</td>
      <td>Recreational</td>
      <td>15</td>
      <td>3_km</td>
      <td>9.566515</td>
      <td>['Light exercise']</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>101481</td>
      <td>2012-07-17 10:50</td>
      <td>32</td>
      <td>176.0</td>
      <td>59.0</td>
      <td>Female</td>
      <td>No injury</td>
      <td>No Injury</td>
      <td>No injury</td>
      <td>No activity</td>
      <td>Recreational</td>
      <td>1</td>
      <td>3_km</td>
      <td>9.450318</td>
      <td>['Low activity']</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100591</td>
      <td>2012-08-09 10:01</td>
      <td>51</td>
      <td>173.0</td>
      <td>67.0</td>
      <td>Male</td>
      <td>Continuing to train in pain</td>
      <td>Hip/Pelvis</td>
      <td>pain</td>
      <td>cycling</td>
      <td>Recreational</td>
      <td>1</td>
      <td>3_km</td>
      <td>8.034622</td>
      <td>['Cycling']</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100595</td>
      <td>2012-08-29 12:56</td>
      <td>50</td>
      <td>155.0</td>
      <td>64.0</td>
      <td>Female</td>
      <td>2 workouts missed in a row</td>
      <td>Lower Leg</td>
      <td>calf muscle strain</td>
      <td>triathlon</td>
      <td>Recreational</td>
      <td>30</td>
      <td>3_km</td>
      <td>8.006290</td>
      <td>['Triathlon']</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


```python
ric_data["race_dist"] = ric_data["race_dist"].str.replace("_", " ")
```

```python
ric_data.head()
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sub_id</th>
      <th>datestring</th>
      <th>age</th>
      <th>height</th>
      <th>weight</th>
      <th>gender</th>
      <th>inj_def</th>
      <th>inj_joint</th>
      <th>spec_injury</th>
      <th>activities</th>
      <th>level</th>
      <th>yrs_running</th>
      <th>race_dist</th>
      <th>speed</th>
      <th>activity_groups</th>
      <th>injury</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100537</td>
      <td>2012-07-03 10:25</td>
      <td>40</td>
      <td>173.0</td>
      <td>68.0</td>
      <td>Female</td>
      <td>2 workouts missed in a row</td>
      <td>Hip/Pelvis</td>
      <td>other</td>
      <td>hiking, power walking, pilates</td>
      <td>Recreational</td>
      <td>2</td>
      <td>3 km</td>
      <td>7.658787</td>
      <td>['Walking', 'Light exercise']</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100560</td>
      <td>2012-07-17 10:37</td>
      <td>33</td>
      <td>179.0</td>
      <td>83.0</td>
      <td>Female</td>
      <td>No injury</td>
      <td>No Injury</td>
      <td>No injury</td>
      <td>Yoga</td>
      <td>Recreational</td>
      <td>15</td>
      <td>3 km</td>
      <td>9.566515</td>
      <td>['Light exercise']</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>101481</td>
      <td>2012-07-17 10:50</td>
      <td>32</td>
      <td>176.0</td>
      <td>59.0</td>
      <td>Female</td>
      <td>No injury</td>
      <td>No Injury</td>
      <td>No injury</td>
      <td>No activity</td>
      <td>Recreational</td>
      <td>1</td>
      <td>3 km</td>
      <td>9.450318</td>
      <td>['Low activity']</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100591</td>
      <td>2012-08-09 10:01</td>
      <td>51</td>
      <td>173.0</td>
      <td>67.0</td>
      <td>Male</td>
      <td>Continuing to train in pain</td>
      <td>Hip/Pelvis</td>
      <td>pain</td>
      <td>cycling</td>
      <td>Recreational</td>
      <td>1</td>
      <td>3 km</td>
      <td>8.034622</td>
      <td>['Cycling']</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100595</td>
      <td>2012-08-29 12:56</td>
      <td>50</td>
      <td>155.0</td>
      <td>64.0</td>
      <td>Female</td>
      <td>2 workouts missed in a row</td>
      <td>Lower Leg</td>
      <td>calf muscle strain</td>
      <td>triathlon</td>
      <td>Recreational</td>
      <td>30</td>
      <td>3 km</td>
      <td>8.006290</td>
      <td>['Triathlon']</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


By feature importance, the more important features from both datasets are distance ran, pace/speed, height, weight, gender.

Possible columns to merge on for both datasets: subject_id, age, height, gender, level, experience(yrs_running), race_dist(encoded), pace(speed).

```python
fukuchi_data_merge = fukuchi_data.rename(columns={
    'subject':'sub_id',
    'mass':'weight',
    'experience':'mths_running'
})
```

```python
fukuchi_data_merge['yrs_running'] = np.ceil(fukuchi_data_merge['mths_running'] / 12).astype(int)

fukuchi_data_merge.head()
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sub_id</th>
      <th>age</th>
      <th>height</th>
      <th>weight</th>
      <th>gender</th>
      <th>dominance</th>
      <th>level</th>
      <th>mths_running</th>
      <th>sessions_per_wk</th>
      <th>run_grp</th>
      <th>...</th>
      <th>15_km</th>
      <th>21_km</th>
      <th>42_km</th>
      <th>5_km</th>
      <th>6_km</th>
      <th>nan</th>
      <th>ultra (50-70 km)</th>
      <th>ultra_42_km</th>
      <th>speed</th>
      <th>yrs_running</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>22</td>
      <td>181.0</td>
      <td>62.0</td>
      <td>M</td>
      <td>R</td>
      <td>Competitive</td>
      <td>4</td>
      <td>3</td>
      <td>Yes</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>15.859031</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>22</td>
      <td>181.0</td>
      <td>62.0</td>
      <td>M</td>
      <td>R</td>
      <td>Competitive</td>
      <td>4</td>
      <td>3</td>
      <td>Yes</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>15.859031</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>22</td>
      <td>181.0</td>
      <td>62.0</td>
      <td>M</td>
      <td>R</td>
      <td>Competitive</td>
      <td>4</td>
      <td>3</td>
      <td>Yes</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>15.859031</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>22</td>
      <td>181.0</td>
      <td>62.0</td>
      <td>M</td>
      <td>R</td>
      <td>Competitive</td>
      <td>4</td>
      <td>3</td>
      <td>Yes</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>15.859031</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>22</td>
      <td>181.0</td>
      <td>62.0</td>
      <td>M</td>
      <td>R</td>
      <td>Competitive</td>
      <td>4</td>
      <td>3</td>
      <td>Yes</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>15.859031</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 37 columns</p>
</div>


```python
common_cols = fukuchi_data_merge.columns.intersection(ric_data.columns)
common_cols
```




    Index(['sub_id', 'age', 'height', 'weight', 'gender', 'level', 'race_dist',
           'injury', 'speed', 'yrs_running'],
          dtype='object')



```python
combined_data = pd.concat([ric_data[common_cols], fukuchi_data_merge[common_cols]], ignore_index=True)
```

```python
combined_data.head()
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sub_id</th>
      <th>age</th>
      <th>height</th>
      <th>weight</th>
      <th>gender</th>
      <th>level</th>
      <th>race_dist</th>
      <th>injury</th>
      <th>speed</th>
      <th>yrs_running</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100537</td>
      <td>40</td>
      <td>173.0</td>
      <td>68.0</td>
      <td>Female</td>
      <td>Recreational</td>
      <td>3 km</td>
      <td>1</td>
      <td>7.658787</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100560</td>
      <td>33</td>
      <td>179.0</td>
      <td>83.0</td>
      <td>Female</td>
      <td>Recreational</td>
      <td>3 km</td>
      <td>0</td>
      <td>9.566515</td>
      <td>15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>101481</td>
      <td>32</td>
      <td>176.0</td>
      <td>59.0</td>
      <td>Female</td>
      <td>Recreational</td>
      <td>3 km</td>
      <td>0</td>
      <td>9.450318</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100591</td>
      <td>51</td>
      <td>173.0</td>
      <td>67.0</td>
      <td>Male</td>
      <td>Recreational</td>
      <td>3 km</td>
      <td>1</td>
      <td>8.034622</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100595</td>
      <td>50</td>
      <td>155.0</td>
      <td>64.0</td>
      <td>Female</td>
      <td>Recreational</td>
      <td>3 km</td>
      <td>1</td>
      <td>8.006290</td>
      <td>30</td>
    </tr>
  </tbody>
</table>
</div>


```python
combined_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 4226 entries, 0 to 4225
    Data columns (total 10 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   sub_id       4226 non-null   int64  
     1   age          4226 non-null   int64  
     2   height       4226 non-null   float64
     3   weight       4226 non-null   float64
     4   gender       4226 non-null   object 
     5   level        4226 non-null   object 
     6   race_dist    4214 non-null   object 
     7   injury       4226 non-null   int64  
     8   speed        4226 non-null   float64
     9   yrs_running  4226 non-null   int64  
    dtypes: float64(3), int64(4), object(3)
    memory usage: 330.3+ KB

```python
combined_data['gender'].unique()
```




    array(['Female', 'Male', 'M', 'F'], dtype=object)



```python
combined_data ['gender'] = combined_data['gender'].replace({
    'M': 'Male',
    'F': 'Female'
})
```

```python
feature_selection = ['age','height','weight','speed']

for feature in feature_selection:
    plt.figure(figsize=(6,4))
    sns.boxplot(data=combined_data, x="injury", y=feature)

    plt.title(f"Injury Distribution by {feature}")
    plt.xlabel("Injury")
    plt.ylabel(f"{feature}")
    plt.show()
```

![png]({{ site.baseurl }}/capstone_model_files/capstone_model/capstone_model_115_0.png){: .center-image }

![png]({{ site.baseurl }}/capstone_model_files/capstone_model/capstone_model_115_1.png){: .center-image }

![png]({{ site.baseurl }}/capstone_model_files/capstone_model/capstone_model_115_2.png){: .center-image }

![png]({{ site.baseurl }}/capstone_model_files/capstone_model/capstone_model_115_3.png){: .center-image }

From the boxplots, it would appear that the weight and height columns have anomalies that need to be imputed.

```python
combined_data['height'].max()
```




    np.float64(999.0)



```python
height_anomalies = combined_data[combined_data['height'] == 999]

height_anomalies
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sub_id</th>
      <th>age</th>
      <th>height</th>
      <th>weight</th>
      <th>gender</th>
      <th>level</th>
      <th>race_dist</th>
      <th>injury</th>
      <th>speed</th>
      <th>yrs_running</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>609</th>
      <td>100113</td>
      <td>37</td>
      <td>999.0</td>
      <td>79.0</td>
      <td>Male</td>
      <td>Competitive</td>
      <td>5 km</td>
      <td>1</td>
      <td>9.703950</td>
      <td>29</td>
    </tr>
    <tr>
      <th>1169</th>
      <td>100325</td>
      <td>34</td>
      <td>999.0</td>
      <td>79.0</td>
      <td>Male</td>
      <td>Competitive</td>
      <td>3 km</td>
      <td>0</td>
      <td>9.729828</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2343</th>
      <td>100106</td>
      <td>69</td>
      <td>999.0</td>
      <td>77.0</td>
      <td>Female</td>
      <td>Recreational</td>
      <td>3 km</td>
      <td>1</td>
      <td>2.596493</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2349</th>
      <td>100113</td>
      <td>37</td>
      <td>999.0</td>
      <td>79.0</td>
      <td>Male</td>
      <td>Competitive</td>
      <td>5 km</td>
      <td>1</td>
      <td>4.496085</td>
      <td>29</td>
    </tr>
    <tr>
      <th>3333</th>
      <td>100325</td>
      <td>34</td>
      <td>999.0</td>
      <td>79.0</td>
      <td>Male</td>
      <td>Competitive</td>
      <td>3 km</td>
      <td>0</td>
      <td>4.072189</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>


```python
mean_height = combined_data.loc[combined_data["height"] != 999, "height"].mean()

combined_data.loc[combined_data["height"] == 999, "height"] = mean_height
```

```python
plt.figure(figsize=(6,4))
sns.boxplot(data=combined_data, x="injury", y="height")

plt.title("Injury Distribution by Height")
plt.xlabel("Injury")
plt.ylabel("Height")
plt.show()
```

![png]({{ site.baseurl }}/capstone_model_files/capstone_model/capstone_model_120_0.png){: .center-image }

```python
combined_data['weight'].sort_values(ascending=False)
```




    28      1564.0
    1973     176.0
    195      176.0
    1974     176.0
    196      176.0
             ...  
    1201      44.0
    3373      44.0
    1699      44.0
    2535      42.0
    832       42.0
    Name: weight, Length: 4226, dtype: float64



```python
combined_data[combined_data['weight'] == 1564]
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sub_id</th>
      <th>age</th>
      <th>height</th>
      <th>weight</th>
      <th>gender</th>
      <th>level</th>
      <th>race_dist</th>
      <th>injury</th>
      <th>speed</th>
      <th>yrs_running</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>28</th>
      <td>100082</td>
      <td>52</td>
      <td>152.0</td>
      <td>1564.0</td>
      <td>Male</td>
      <td>Recreational</td>
      <td>Other distance</td>
      <td>1</td>
      <td>8.213175</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>


```python
mean_weight = combined_data.loc[combined_data["weight"] != 1564, "weight"].mean()

combined_data.loc[combined_data["weight"] == 1564, "weight"] = mean_weight
```

```python
plt.figure(figsize=(6,4))
sns.boxplot(data=combined_data, x="injury", y="weight")

plt.title("Weight Distribution by Injury")
plt.xlabel("Injury")
plt.ylabel("Weight")
plt.show()
```

![png]({{ site.baseurl }}/capstone_model_files/capstone_model/capstone_model_124_0.png){: .center-image }

```python
plt.figure(figsize=(6,4))
sns.countplot(data=combined_data, x="gender", hue="injury")

plt.title("Injury Counts by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.legend(title="Injury")
plt.show()
```

![png]({{ site.baseurl }}/capstone_model_files/capstone_model/capstone_model_125_0.png){: .center-image }

```python
plt.figure(figsize=(10,6))
sns.histplot(data=combined_data, x="yrs_running", hue="injury", multiple="fill", bins=15, stat="percent")

plt.title("Distribution of Injury by Years of Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Percentage")
plt.show()
```

![png]({{ site.baseurl }}/capstone_model_files/capstone_model/capstone_model_126_0.png){: .center-image }

Since the Random Forest model worked better for the RIC dataset and had feature importance that was closer to my hypothesis, I will start by using the Random Forest model for the combined dataset and pivot to other learning models if needed.

```python
combined_data['race_dist'].unique()
```




    array(['3 km', '21 km', '10 km', 'Other distance', '5 km', '42 km',
           '21 km, 42 km', '5 km, 10 km, 21 km, 42 km', '10 km, 21 km, 42 km',
           '5 km, 10 km, 21 km', '50-70 km', '5 km, 10 km, 15 km, 21 km',
           '>42 km', '10 km, 21 km, 6 km', '10 km, 21 km', nan,
           '10 km, 12 km', '5 km, 10 km'], dtype=object)



```python
combined_data['race_dist_list'] = combined_data["race_dist"].dropna().apply(lambda x: [d.strip() for d in x.split(",")])

race_dist_labelling = MultiLabelBinarizer()
encoded_dist = race_dist_labelling.fit_transform(combined_data["race_dist_list"].dropna())

dist_df = pd.DataFrame(encoded_dist, columns=race_dist_labelling.classes_, index=combined_data["race_dist_list"].dropna().index)

combined_dist_data = combined_data.join(dist_df).fillna(0)
```

```python
dist_inj = pd.DataFrame()

for dist in dist_df.columns:
    dist_inj[dist] = combined_dist_data.groupby("injury")[dist].sum()

dist_inj = dist_inj.T  

dist_inj.plot(kind="bar", figsize=(10,6))
plt.title("Injury Count by Running Distance")
plt.xlabel("Run Distance")
plt.ylabel("Number of Runners")
plt.legend(title="Injury")
plt.show()
```

![png]({{ site.baseurl }}/capstone_model_files/capstone_model/capstone_model_130_0.png){: .center-image }

```python
dist_inj['injury_rate'] = 100*(dist_inj[1] / (dist_inj[0] + dist_inj[1]))

dist_inj
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>injury</th>
      <th>0</th>
      <th>1</th>
      <th>injury_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10 km</th>
      <td>222.0</td>
      <td>413.0</td>
      <td>65.039370</td>
    </tr>
    <tr>
      <th>12 km</th>
      <td>0.0</td>
      <td>6.0</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>15 km</th>
      <td>0.0</td>
      <td>12.0</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>21 km</th>
      <td>372.0</td>
      <td>442.0</td>
      <td>54.299754</td>
    </tr>
    <tr>
      <th>3 km</th>
      <td>894.0</td>
      <td>1147.0</td>
      <td>56.197942</td>
    </tr>
    <tr>
      <th>42 km</th>
      <td>219.0</td>
      <td>327.0</td>
      <td>59.890110</td>
    </tr>
    <tr>
      <th>5 km</th>
      <td>142.0</td>
      <td>223.0</td>
      <td>61.095890</td>
    </tr>
    <tr>
      <th>50-70 km</th>
      <td>12.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6 km</th>
      <td>12.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>&gt;42 km</th>
      <td>12.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Other distance</th>
      <td>48.0</td>
      <td>143.0</td>
      <td>74.869110</td>
    </tr>
  </tbody>
</table>
</div>


```python
combined_dist_data.head()
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sub_id</th>
      <th>age</th>
      <th>height</th>
      <th>weight</th>
      <th>gender</th>
      <th>level</th>
      <th>race_dist</th>
      <th>injury</th>
      <th>speed</th>
      <th>yrs_running</th>
      <th>...</th>
      <th>12 km</th>
      <th>15 km</th>
      <th>21 km</th>
      <th>3 km</th>
      <th>42 km</th>
      <th>5 km</th>
      <th>50-70 km</th>
      <th>6 km</th>
      <th>&gt;42 km</th>
      <th>Other distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100537</td>
      <td>40</td>
      <td>173.0</td>
      <td>68.0</td>
      <td>Female</td>
      <td>Recreational</td>
      <td>3 km</td>
      <td>1</td>
      <td>7.658787</td>
      <td>2</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100560</td>
      <td>33</td>
      <td>179.0</td>
      <td>83.0</td>
      <td>Female</td>
      <td>Recreational</td>
      <td>3 km</td>
      <td>0</td>
      <td>9.566515</td>
      <td>15</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>101481</td>
      <td>32</td>
      <td>176.0</td>
      <td>59.0</td>
      <td>Female</td>
      <td>Recreational</td>
      <td>3 km</td>
      <td>0</td>
      <td>9.450318</td>
      <td>1</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100591</td>
      <td>51</td>
      <td>173.0</td>
      <td>67.0</td>
      <td>Male</td>
      <td>Recreational</td>
      <td>3 km</td>
      <td>1</td>
      <td>8.034622</td>
      <td>1</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100595</td>
      <td>50</td>
      <td>155.0</td>
      <td>64.0</td>
      <td>Female</td>
      <td>Recreational</td>
      <td>3 km</td>
      <td>1</td>
      <td>8.006290</td>
      <td>30</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>


```python
combined_data['level'].unique()
```




    array(['Recreational', 'Competitive', 'Elite'], dtype=object)



```python
combined_data['level'] = combined_data['level'].replace({'Elite':'Competitive'})
```

```python
level_inj = pd.crosstab (combined_data['level'],combined_data['injury'])

level_inj.plot(kind='bar',width=0.8)
plt.ylabel("Number of runners")
plt.title('Injury by Level')
plt.legend(title='Injury')
plt.show()
```

![png]({{ site.baseurl }}/capstone_model_files/capstone_model/capstone_model_135_0.png){: .center-image }

```python
combined_data_encoding = pd.get_dummies(combined_data, columns=['gender','level'], drop_first=True)

combined_data_encoding.head()
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sub_id</th>
      <th>age</th>
      <th>height</th>
      <th>weight</th>
      <th>race_dist</th>
      <th>injury</th>
      <th>speed</th>
      <th>yrs_running</th>
      <th>race_dist_list</th>
      <th>gender_Male</th>
      <th>level_Recreational</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100537</td>
      <td>40</td>
      <td>173.0</td>
      <td>68.0</td>
      <td>3 km</td>
      <td>1</td>
      <td>7.658787</td>
      <td>2</td>
      <td>[3 km]</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100560</td>
      <td>33</td>
      <td>179.0</td>
      <td>83.0</td>
      <td>3 km</td>
      <td>0</td>
      <td>9.566515</td>
      <td>15</td>
      <td>[3 km]</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>101481</td>
      <td>32</td>
      <td>176.0</td>
      <td>59.0</td>
      <td>3 km</td>
      <td>0</td>
      <td>9.450318</td>
      <td>1</td>
      <td>[3 km]</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100591</td>
      <td>51</td>
      <td>173.0</td>
      <td>67.0</td>
      <td>3 km</td>
      <td>1</td>
      <td>8.034622</td>
      <td>1</td>
      <td>[3 km]</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100595</td>
      <td>50</td>
      <td>155.0</td>
      <td>64.0</td>
      <td>3 km</td>
      <td>1</td>
      <td>8.006290</td>
      <td>30</td>
      <td>[3 km]</td>
      <td>False</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>


```python
dist_cols = ['10 km', '12 km','15 km','21 km','3 km','42 km','5 km','50-70 km','6 km','>42 km','Other distance']

model_data = combined_data_encoding.join(combined_dist_data[dist_cols]).fillna(0)

model_data.head()
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sub_id</th>
      <th>age</th>
      <th>height</th>
      <th>weight</th>
      <th>race_dist</th>
      <th>injury</th>
      <th>speed</th>
      <th>yrs_running</th>
      <th>race_dist_list</th>
      <th>gender_Male</th>
      <th>...</th>
      <th>12 km</th>
      <th>15 km</th>
      <th>21 km</th>
      <th>3 km</th>
      <th>42 km</th>
      <th>5 km</th>
      <th>50-70 km</th>
      <th>6 km</th>
      <th>&gt;42 km</th>
      <th>Other distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100537</td>
      <td>40</td>
      <td>173.0</td>
      <td>68.0</td>
      <td>3 km</td>
      <td>1</td>
      <td>7.658787</td>
      <td>2</td>
      <td>[3 km]</td>
      <td>False</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100560</td>
      <td>33</td>
      <td>179.0</td>
      <td>83.0</td>
      <td>3 km</td>
      <td>0</td>
      <td>9.566515</td>
      <td>15</td>
      <td>[3 km]</td>
      <td>False</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>101481</td>
      <td>32</td>
      <td>176.0</td>
      <td>59.0</td>
      <td>3 km</td>
      <td>0</td>
      <td>9.450318</td>
      <td>1</td>
      <td>[3 km]</td>
      <td>False</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100591</td>
      <td>51</td>
      <td>173.0</td>
      <td>67.0</td>
      <td>3 km</td>
      <td>1</td>
      <td>8.034622</td>
      <td>1</td>
      <td>[3 km]</td>
      <td>True</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100595</td>
      <td>50</td>
      <td>155.0</td>
      <td>64.0</td>
      <td>3 km</td>
      <td>1</td>
      <td>8.006290</td>
      <td>30</td>
      <td>[3 km]</td>
      <td>False</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>


```python
run_data = model_data.drop(columns=['race_dist','race_dist_list'])
```

```python
run_data.head()
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sub_id</th>
      <th>age</th>
      <th>height</th>
      <th>weight</th>
      <th>injury</th>
      <th>speed</th>
      <th>yrs_running</th>
      <th>gender_Male</th>
      <th>level_Recreational</th>
      <th>10 km</th>
      <th>12 km</th>
      <th>15 km</th>
      <th>21 km</th>
      <th>3 km</th>
      <th>42 km</th>
      <th>5 km</th>
      <th>50-70 km</th>
      <th>6 km</th>
      <th>&gt;42 km</th>
      <th>Other distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100537</td>
      <td>40</td>
      <td>173.0</td>
      <td>68.0</td>
      <td>1</td>
      <td>7.658787</td>
      <td>2</td>
      <td>False</td>
      <td>True</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100560</td>
      <td>33</td>
      <td>179.0</td>
      <td>83.0</td>
      <td>0</td>
      <td>9.566515</td>
      <td>15</td>
      <td>False</td>
      <td>True</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>101481</td>
      <td>32</td>
      <td>176.0</td>
      <td>59.0</td>
      <td>0</td>
      <td>9.450318</td>
      <td>1</td>
      <td>False</td>
      <td>True</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100591</td>
      <td>51</td>
      <td>173.0</td>
      <td>67.0</td>
      <td>1</td>
      <td>8.034622</td>
      <td>1</td>
      <td>True</td>
      <td>True</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100595</td>
      <td>50</td>
      <td>155.0</td>
      <td>64.0</td>
      <td>1</td>
      <td>8.006290</td>
      <td>30</td>
      <td>False</td>
      <td>True</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>


```python
run_data['injury'] = run_data['injury'].astype(int)
```

```python
selected_features = run_data.drop(columns=['sub_id','injury'])
```

```python
X_combined = selected_features
y_combined = run_data['injury']

log_regression3 = LogisticRegression(max_iter=5000)

X_train_combined_lr, X_test_combined_lr, y_train_combined_lr, y_test_combined_lr= train_test_split(X_combined, y_combined, test_size=0.2, random_state=65)

log_regression3.fit(X_train_combined_lr, y_train_combined_lr)

y_pred_combined_lr = log_regression3.predict(X_test_combined_lr)
acc3 = accuracy_score(y_test_combined_lr, y_pred_combined_lr)
print("Test Accuracy:", acc3)
print("Confusion Matrix:\n", confusion_matrix(y_test_combined_lr, y_pred_combined_lr))
print("Classification Report:\n", classification_report(y_test_combined_lr, y_pred_combined_lr))
```

    Test Accuracy: 0.6832151300236406
    Confusion Matrix:
     [[142 195]
     [ 73 436]]
    Classification Report:
                   precision    recall  f1-score   support
    
               0       0.66      0.42      0.51       337
               1       0.69      0.86      0.76       509
    
        accuracy                           0.68       846
       macro avg       0.68      0.64      0.64       846
    weighted avg       0.68      0.68      0.67       846
    

```python
y_pred_proba_combined_lr = log_regression3.predict_proba(X_test_combined_lr)[:, 1]
y_test_roc_combined_lr = y_test_combined_lr.astype(int)
fpr_combined_lr, tpr_combined_lr, thresholds_combined_lr = roc_curve(y_test_roc_combined_lr, y_pred_proba_combined_lr)

roc_auc_combined_lr = auc(fpr_combined_lr, tpr_combined_lr)

plt.figure(figsize=(6,6))
plt.plot(fpr_combined_lr, tpr_combined_lr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc_combined_lr:.2f})')
plt.plot([0,1], [0,1], color='gray', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Logistic Regression (Combined dataset)")
plt.legend(loc="lower right")
plt.show()
```

![png]({{ site.baseurl }}/capstone_model_files/capstone_model/capstone_model_143_0.png){: .center-image }

```python
X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(X_combined, y_combined, test_size=0.2, random_state=65)

rf_combined = RandomForestClassifier(
    n_estimators=200,    
    max_depth=None,   
    random_state=65,
    class_weight="balanced"
)

rf_combined.fit(X_train_combined, y_train_combined)

y_pred_combined = rf_combined.predict(X_test_combined)
acc_combined = accuracy_score(y_test_combined, y_pred_combined)
print("Random Forest Classifier Accuracy:", acc_combined)
print("Classification Report:\n", classification_report(y_test_combined, y_pred_combined))
print("Confusion Matrix:\n", confusion_matrix(y_test_combined, y_pred_combined))
```

    Random Forest Classifier Accuracy: 0.8014184397163121
    Classification Report:
                   precision    recall  f1-score   support
    
               0       0.84      0.62      0.71       337
               1       0.78      0.92      0.85       509
    
        accuracy                           0.80       846
       macro avg       0.81      0.77      0.78       846
    weighted avg       0.81      0.80      0.79       846
    
    Confusion Matrix:
     [[208 129]
     [ 39 470]]

The number of false positives is still rather high for the Random Forest Classifier model.

```python
top_features = rf_combined.feature_importances_

feature_importances = pd.DataFrame({
    "Feature": X_combined.columns,
    "Importance": top_features
})

feature_importances.head(10)
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature</th>
      <th>Importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>age</td>
      <td>0.178893</td>
    </tr>
    <tr>
      <th>1</th>
      <td>height</td>
      <td>0.140809</td>
    </tr>
    <tr>
      <th>2</th>
      <td>weight</td>
      <td>0.151805</td>
    </tr>
    <tr>
      <th>3</th>
      <td>speed</td>
      <td>0.250867</td>
    </tr>
    <tr>
      <th>4</th>
      <td>yrs_running</td>
      <td>0.126676</td>
    </tr>
    <tr>
      <th>5</th>
      <td>gender_Male</td>
      <td>0.017158</td>
    </tr>
    <tr>
      <th>6</th>
      <td>level_Recreational</td>
      <td>0.026356</td>
    </tr>
    <tr>
      <th>7</th>
      <td>10 km</td>
      <td>0.021274</td>
    </tr>
    <tr>
      <th>8</th>
      <td>12 km</td>
      <td>0.000290</td>
    </tr>
    <tr>
      <th>9</th>
      <td>15 km</td>
      <td>0.001780</td>
    </tr>
  </tbody>
</table>
</div>


The Random Forest model feature importance is distributed throughout multiple features such as age, height, weight, speed, experience, gender, level and distance.

```python
X_train_boost, X_test_boost, y_train_boost, y_test_boost = train_test_split(
    X_combined, y_combined, test_size=0.2, random_state=65, stratify=y_combined
)

xgb_combined = XGBClassifier(
    n_estimators=200,       
    max_depth=5,            
    learning_rate=0.1,      
    subsample=0.8,          
    colsample_bytree=0.8,   
    random_state=65,
    eval_metric="logloss"     
)

xgb_combined.fit(X_train_boost, y_train_boost)

y_pred_xgb = xgb_combined.predict(X_test_boost)

acc_xgb = accuracy_score(y_test_boost, y_pred_xgb)
print("XGBoost Classifier Accuracy:", acc_xgb)
print("Classification Report:\n", classification_report(y_test_boost, y_pred_xgb))
print("Confusion Matrix:\n", confusion_matrix(y_test_boost, y_pred_xgb))
```

    XGBoost Classifier Accuracy: 0.8522458628841607
    Classification Report:
                   precision    recall  f1-score   support
    
               0       0.87      0.73      0.79       326
               1       0.84      0.93      0.89       520
    
        accuracy                           0.85       846
       macro avg       0.86      0.83      0.84       846
    weighted avg       0.85      0.85      0.85       846
    
    Confusion Matrix:
     [[237  89]
     [ 36 484]]

The XGBoost model gives a higher precision than the Random Forest model as it has fewer false positives than the RF model. However, there are still a significant number of false positives. I will check if there is any way to tune the XGBoost model to make it even more accurate.

```python
#This cell works in Marimo but not Jupyter Notebook

xgb = XGBClassifier(
    random_state=65,
    eval_metric="logloss"
)

param_grid_xgb = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

grid_search_xgb = GridSearchCV(
    xgb,
    param_grid_xgb,
    cv=5,
    scoring="precision",
    n_jobs=-1,
    verbose=2
)

try:
    grid_search_xgb.fit(X_train_boost, y_train_boost)
except Exception as e:
    print("Fit failed with error:", e)

print("Best XGBoost Params:", grid_search_xgb.best_params_)
print("Best XGBoost Precision Score:", grid_search_xgb.best_score_)
```

    Fit failed with error: No module named '_posixsubprocess'


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    Cell In[137], line 28
         25 except Exception as e:
         26     print("Fit failed with error:", e)
    ---> 28 print("Best XGBoost Params:", grid_search_xgb.best_params_)
         29 print("Best XGBoost Precision Score:", grid_search_xgb.best_score_)
    

    AttributeError: 'GridSearchCV' object has no attribute 'best_params_'


```python
X_train_boost2, X_test_boost2, y_train_boost2, y_test_boost2 = train_test_split(
    X_combined, y_combined, test_size=0.2, random_state=65, stratify=y_combined
)

xgb_combined2 = XGBClassifier(
    n_estimators=200,       
    max_depth=7,            
    learning_rate=0.2,      
    subsample=1.0,          
    colsample_bytree=0.8,   
    random_state=65,
    eval_metric="logloss"     
)

xgb_combined2.fit(X_train_boost2, y_train_boost2)

y_pred_xgb2 = xgb_combined2.predict(X_test_boost2)

acc_xgb2 = accuracy_score(y_test_boost2, y_pred_xgb2)
print("XGBoost Classifier Accuracy:", acc_xgb2)
print("Classification Report:\n", classification_report(y_test_boost2, y_pred_xgb2))
print("Confusion Matrix:\n", confusion_matrix(y_test_boost2, y_pred_xgb2))
```

    XGBoost Classifier Accuracy: 0.8652482269503546
    Classification Report:
                   precision    recall  f1-score   support
    
               0       0.86      0.77      0.82       326
               1       0.87      0.92      0.89       520
    
        accuracy                           0.87       846
       macro avg       0.86      0.85      0.85       846
    weighted avg       0.87      0.87      0.86       846
    
    Confusion Matrix:
     [[252  74]
     [ 40 480]]

```python
y_pred_proba_combined = rf_combined.predict_proba(X_test_combined)[:, 1]
y_test_roc_combined = y_test_combined.astype(int)
fpr_combined, tpr_combined, _ = roc_curve(y_test_roc_combined, y_pred_proba_combined)
roc_auc_combined = auc(fpr_combined, tpr_combined)

y_pred_proba_xgb = xgb_combined.predict_proba(X_test_boost)[:, 1]
y_test_roc_xgb = y_test_boost.astype(int)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test_boost, y_pred_proba_xgb)
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)

y_pred_proba_xgb2 = xgb_combined2.predict_proba(X_test_boost2)[:, 1]
y_test_roc_xgb2 = y_test_boost2.astype(int)
fpr_xgb2, tpr_xgb2, _ = roc_curve(y_test_boost2, y_pred_proba_xgb2)
roc_auc_xgb2 = auc(fpr_xgb2, tpr_xgb2)

plt.figure(figsize=(6,6))
plt.plot(fpr_combined_lr, tpr_combined_lr, color='blue', lw=2, label=f'Logistic Regression ROC curve (AUC = {roc_auc_combined_lr:.2f})')
plt.plot(fpr_combined, tpr_combined, color='green', lw=2, label=f'Random Forest ROC curve (AUC = {roc_auc_combined:.2f})')
plt.plot(fpr_xgb, tpr_xgb, color='orange', lw=2, label=f'XGBoost ROC curve (AUC = {roc_auc_xgb:.2f})')
plt.plot(fpr_xgb2, tpr_xgb2, color='purple', lw=2, label=f'Tuned XGBoost ROC curve (AUC = {roc_auc_xgb2:.2f})')
plt.plot([0,1], [0,1], color='gray', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparisons (Combined dataset)")
plt.legend(loc="lower right")
plt.show()
```

![png]({{ site.baseurl }}/capstone_model_files/capstone_model/capstone_model_152_0.png){: .center-image }

```python
xgb_importance = xgb_combined2.feature_importances_

feat_imp_xgb = pd.DataFrame({
    "Feature": X_combined.columns,      
    "Importance": xgb_importance
})

feat_importance = feat_imp_xgb.sort_values(by="Importance", ascending=False)

feat_importance.head(10)
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature</th>
      <th>Importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9</th>
      <td>15 km</td>
      <td>0.174123</td>
    </tr>
    <tr>
      <th>7</th>
      <td>10 km</td>
      <td>0.099101</td>
    </tr>
    <tr>
      <th>10</th>
      <td>21 km</td>
      <td>0.097852</td>
    </tr>
    <tr>
      <th>11</th>
      <td>3 km</td>
      <td>0.096281</td>
    </tr>
    <tr>
      <th>13</th>
      <td>5 km</td>
      <td>0.082986</td>
    </tr>
    <tr>
      <th>6</th>
      <td>level_Recreational</td>
      <td>0.065192</td>
    </tr>
    <tr>
      <th>3</th>
      <td>speed</td>
      <td>0.056356</td>
    </tr>
    <tr>
      <th>12</th>
      <td>42 km</td>
      <td>0.053350</td>
    </tr>
    <tr>
      <th>0</th>
      <td>age</td>
      <td>0.052113</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Other distance</td>
      <td>0.047246</td>
    </tr>
  </tbody>
</table>
</div>


Distance is the most important predictor of injury for the tuned XGBoost model, with 8 out of the top 10 features being related to distance. The other 2 features are level and speed. This reveals that the tuned XGBoost model might be overdependent on one feature for its predictions, which may not be as useful for the general runner populus. Hence, the more well-balanced Random Forest model might be better as a predictive model for our use case, even if it is slightly less accurate.

```python
joblib.dump(rf_combined, "rf_model.pkl")
```




    ['rf_model.pkl']



```python
feature_names = X_combined.columns.tolist()

joblib.dump(feature_names, "rf_features.joblib")
```




    ['rf_features.joblib']



```python

```

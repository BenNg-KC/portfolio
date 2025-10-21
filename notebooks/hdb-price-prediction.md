---
layout: page
title: Project 3
permalink: /Project 3
---

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

```python
hdb = pd.read_csv('./train.csv', dtype={'postal': str})
```

```python
hdb.dtypes
```




    id                     int64
    Tranc_YearMonth       object
    town                  object
    flat_type             object
    block                 object
                          ...   
    sec_sch_name          object
    cutoff_point           int64
    affiliation            int64
    sec_sch_latitude     float64
    sec_sch_longitude    float64
    Length: 78, dtype: object



```python
count_nulls = hdb.isnull().sum()
```

```python
null_percentages = (count_nulls / len(hdb)) * 100

null_percentages
```




    id                   0.0
    Tranc_YearMonth      0.0
    town                 0.0
    flat_type            0.0
    block                0.0
                        ... 
    sec_sch_name         0.0
    cutoff_point         0.0
    affiliation          0.0
    sec_sch_latitude     0.0
    sec_sch_longitude    0.0
    Length: 78, dtype: float64



```python
hdb.plot(kind='scatter', x='planning_area', y='resale_price', alpha=0.1)
```




    <Axes: xlabel='planning_area', ylabel='resale_price'>



![png]({{ site.baseurl }}/Project%203_files/Project%203/Project%203_5_1.png){: .center-image }

```python
hdb['planning_area'].value_counts().head().sort_values(ascending=True).plot(kind='barh')
```




    <Axes: ylabel='planning_area'>



![png]({{ site.baseurl }}/Project%203_files/Project%203/Project%203_6_1.png){: .center-image }

```python
top_15 = hdb['planning_area'].value_counts().head(15).index
plt.figure(figsize=(16, 8))
sns.boxplot(data=hdb[hdb['planning_area'].isin(top_15)],
            x='planning_area', y='resale_price')
plt.xticks(rotation=45)
plt.title('Price Range by Top 15 Planning Areas')
plt.show()
```

![png]({{ site.baseurl }}/Project%203_files/Project%203/Project%203_7_0.png){: .center-image }

```python
import seaborn as sns
```

```python
plt.figure(figsize=(10, 5))
sns.histplot(hdb['resale_price'], bins=50, kde=True)
plt.title('Distribution of HDB Resale Prices')
plt.xlabel('Price (SGD)')
plt.ylabel('Number of Flats')
plt.show()
```

![png]({{ site.baseurl }}/Project%203_files/Project%203/Project%203_9_0.png){: .center-image }

```python
print(hdb['resale_price'].describe())
print(hdb['resale_price'].unique()[:20])
print(hdb['resale_price'].min(), hdb['resale_price'].max())
```

    count    1.506340e+05
    mean     4.491615e+05
    std      1.433076e+05
    min      1.500000e+05
    25%      3.470000e+05
    50%      4.200000e+05
    75%      5.200000e+05
    max      1.258000e+06
    Name: resale_price, dtype: float64
    [680000. 665000. 838000. 550000. 298000. 335000. 433000. 340000. 700000.
     490000. 625000. 270000. 500000. 770000. 345000. 288888. 400000. 800000.
     806000. 455000.]
    150000.0 1258000.0

```python
hdb.plot(kind='scatter', x='Mall_Nearest_Distance', y='resale_price', alpha=0.1)
```




    <Axes: xlabel='Mall_Nearest_Distance', ylabel='resale_price'>



![png]({{ site.baseurl }}/Project%203_files/Project%203/Project%203_11_1.png){: .center-image }

```python
hdb.plot(kind='scatter', x='Hawker_Nearest_Distance', y='resale_price', alpha=0.1)
```




    <Axes: xlabel='Hawker_Nearest_Distance', ylabel='resale_price'>



![png]({{ site.baseurl }}/Project%203_files/Project%203/Project%203_12_1.png){: .center-image }

```python
hdb.plot(kind='scatter', x='mrt_nearest_distance', y='resale_price', alpha=0.1)
```




    <Axes: xlabel='mrt_nearest_distance', ylabel='resale_price'>



![png]({{ site.baseurl }}/Project%203_files/Project%203/Project%203_13_1.png){: .center-image }

```python
import plotly.express as px

px.scatter_mapbox(hdb, lat="Latitude", lon="Longitude", color="resale_price",
                  mapbox_style="carto-positron",
                  hover_name="planning_area", zoom=10)
```

    /var/folders/hl/51hk561j7s7_jm62kls87wv80000gn/T/ipykernel_12846/737955990.py:3: DeprecationWarning: *scatter_mapbox* is deprecated! Use *scatter_map* instead. Learn more at: https://plotly.com/python/mapbox-to-maplibre/
      px.scatter_mapbox(hdb, lat="Latitude", lon="Longitude", color="resale_price",

![png]({{ site.baseurl }}/Project%203_files/Project%203/Project%203_14_1.png){: .center-image }

```python
selected_cols = ['1room_rental','2room_rental', '3room_rental', 'other_room_rental', 'Mall_Nearest_Distance', 'Hawker_Nearest_Distance', 'mrt_nearest_distance', 'resale_price']
```

```python
hdb_selected = hdb[selected_cols]
```

```python

hdb_selected = hdb_selected.select_dtypes(include='number')

plt.figure(figsize=(10, 6))
sns.heatmap(hdb_selected.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Selected Features')
plt.show()
```

![png]({{ site.baseurl }}/Project%203_files/Project%203/Project%203_17_0.png){: .center-image }

```python
selected_cols = ['floor_area_sqm', 'Tranc_Year', 'hdb_age', '1room_sold', '2room_sold', '3room_sold', '4room_sold', '5room_sold', 'exec_sold', 'multigen_sold', 'studio_apartment_sold', '1room_rental','2room_rental','3room_rental', 'other_room_rental', 'Mall_Nearest_Distance', 'Hawker_Nearest_Distance', 'mrt_nearest_distance', 'resale_price', 'bus_stop_nearest_distance', 'pri_sch_nearest_distance']

hdb_selected = hdb[selected_cols]

hdb_selected = hdb_selected.select_dtypes(include='number')

plt.figure(figsize=(15, 9))
sns.heatmap(hdb_selected.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Selected Features')
plt.show()
```

![png]({{ site.baseurl }}/Project%203_files/Project%203/Project%203_18_0.png){: .center-image }

```python
hdb = pd.read_csv('./train_data_clean.csv', dtype={'postal': str})
```

```python
hdb.dtypes
```




    id                     int64
    Tranc_YearMonth       object
    town                  object
    flat_type             object
    block                 object
                          ...   
    sec_sch_name          object
    cutoff_point           int64
    affiliation            int64
    sec_sch_latitude     float64
    sec_sch_longitude    float64
    Length: 81, dtype: object



```python
hdb['Tranc_YearMonth'] = pd.to_datetime(hdb['Tranc_YearMonth'], dayfirst=True)

hdb.dtypes
```




    id                            int64
    Tranc_YearMonth      datetime64[ns]
    town                         object
    flat_type                    object
    block                        object
                              ...      
    sec_sch_name                 object
    cutoff_point                  int64
    affiliation                   int64
    sec_sch_latitude            float64
    sec_sch_longitude           float64
    Length: 81, dtype: object



```python
hdb['YearMonth'] = hdb['Tranc_YearMonth'].dt.to_period('M')
```

```python
hdb['Tranc_YearMonth'].head(12)
```




    0    2016-05-01
    1    2012-07-01
    2    2013-07-01
    3    2012-04-01
    4    2017-12-01
    5    2013-01-01
    6    2018-05-01
    7    2012-03-01
    8    2020-01-01
    9    2014-06-01
    10   2013-06-01
    11   2018-03-01
    Name: Tranc_YearMonth, dtype: datetime64[ns]



```python
# MONTHLY TRANSACTION VOLUME

monthly_volume = hdb.groupby(hdb['Tranc_YearMonth'].dt.to_period('M')).size() #convert to just year and month & group the rows by month and count how many transaction happened each month
monthly_volume.index = monthly_volume.index.to_timestamp() 

plt.figure(figsize=(14, 5))
plt.plot(monthly_volume.index, monthly_volume.values, marker='o', color='blue')
plt.title('Monthly HDB Resale Volume (Grouped by Year-Month)')
plt.xlabel('Month')
plt.ylabel('Number of Transactions')
plt.grid(True)
plt.tight_layout()
plt.show()
```

![png]({{ site.baseurl }}/Project%203_files/Project%203/Project%203_24_0.png){: .center-image }

```python
hdb['Tranc_Year'].dtype

```




    dtype('int64')



```python
yearly_volume = hdb.groupby('Tranc_Year').size()
yearly_volume.plot(kind='bar', figsize=(10, 5))
plt.title('Annual HDB Resale Transaction Volume')
plt.xlabel('Year')
plt.ylabel('Number of Transactions')
plt.show()
```

![png]({{ site.baseurl }}/Project%203_files/Project%203/Project%203_26_0.png){: .center-image }

```python
volume_by_type = hdb.groupby(['Tranc_YearMonth', 'flat_type']).size().unstack()

volume_by_type.plot(figsize=(14, 6))
plt.title('Monthly Transaction Volume by Flat Type')
plt.xlabel('Month')
plt.ylabel('Number of Transactions')
plt.legend(title='Flat Type')
plt.grid(True)
plt.tight_layout()
plt.show()
```

![png]({{ site.baseurl }}/Project%203_files/Project%203/Project%203_27_0.png){: .center-image }

```python
town_volume = hdb.groupby('town').size()
town_volume.plot(kind='bar', figsize=(10, 5))
plt.title('Transaction Volume by Town')
plt.xlabel('Town')
plt.ylabel('Number of Transactions')
plt.show()
```

![png]({{ site.baseurl }}/Project%203_files/Project%203/Project%203_28_0.png){: .center-image }

```python
pivot_town_flat_type = pd.pivot_table(hdb, values='id', index='town', columns='flat_type', aggfunc='count')

pivot_town_flat_type
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
      <th>flat_type</th>
      <th>1 ROOM</th>
      <th>2 ROOM</th>
      <th>3 ROOM</th>
      <th>4 ROOM</th>
      <th>5 ROOM</th>
      <th>EXECUTIVE</th>
      <th>MULTI-GENERATION</th>
    </tr>
    <tr>
      <th>town</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ANG MO KIO</th>
      <td>NaN</td>
      <td>147.0</td>
      <td>3961.0</td>
      <td>1821.0</td>
      <td>891.0</td>
      <td>88.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BEDOK</th>
      <td>NaN</td>
      <td>132.0</td>
      <td>4009.0</td>
      <td>2986.0</td>
      <td>1500.0</td>
      <td>419.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BISHAN</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>355.0</td>
      <td>1398.0</td>
      <td>826.0</td>
      <td>285.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>BUKIT BATOK</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>2173.0</td>
      <td>2212.0</td>
      <td>738.0</td>
      <td>500.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BUKIT MERAH</th>
      <td>82.0</td>
      <td>208.0</td>
      <td>2199.0</td>
      <td>2106.0</td>
      <td>1259.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BUKIT PANJANG</th>
      <td>NaN</td>
      <td>45.0</td>
      <td>669.0</td>
      <td>2661.0</td>
      <td>1691.0</td>
      <td>620.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BUKIT TIMAH</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>71.0</td>
      <td>128.0</td>
      <td>87.0</td>
      <td>83.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>CENTRAL AREA</th>
      <td>NaN</td>
      <td>60.0</td>
      <td>537.0</td>
      <td>481.0</td>
      <td>171.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>CHOA CHU KANG</th>
      <td>NaN</td>
      <td>33.0</td>
      <td>364.0</td>
      <td>3128.0</td>
      <td>2107.0</td>
      <td>711.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>CLEMENTI</th>
      <td>NaN</td>
      <td>28.0</td>
      <td>1944.0</td>
      <td>1153.0</td>
      <td>390.0</td>
      <td>118.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>GEYLANG</th>
      <td>NaN</td>
      <td>157.0</td>
      <td>1904.0</td>
      <td>1318.0</td>
      <td>476.0</td>
      <td>131.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>HOUGANG</th>
      <td>NaN</td>
      <td>36.0</td>
      <td>1675.0</td>
      <td>3396.0</td>
      <td>1575.0</td>
      <td>873.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>JURONG EAST</th>
      <td>NaN</td>
      <td>38.0</td>
      <td>1304.0</td>
      <td>1016.0</td>
      <td>806.0</td>
      <td>306.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>JURONG WEST</th>
      <td>NaN</td>
      <td>86.0</td>
      <td>2221.0</td>
      <td>4246.0</td>
      <td>3619.0</td>
      <td>1279.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>KALLANG/WHAMPOA</th>
      <td>NaN</td>
      <td>80.0</td>
      <td>2056.0</td>
      <td>1326.0</td>
      <td>792.0</td>
      <td>86.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>MARINE PARADE</th>
      <td>NaN</td>
      <td>5.0</td>
      <td>482.0</td>
      <td>243.0</td>
      <td>229.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>PASIR RIS</th>
      <td>NaN</td>
      <td>18.0</td>
      <td>76.0</td>
      <td>1964.0</td>
      <td>1491.0</td>
      <td>1214.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>PUNGGOL</th>
      <td>NaN</td>
      <td>133.0</td>
      <td>481.0</td>
      <td>4077.0</td>
      <td>2839.0</td>
      <td>263.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>QUEENSTOWN</th>
      <td>NaN</td>
      <td>161.0</td>
      <td>1986.0</td>
      <td>1391.0</td>
      <td>543.0</td>
      <td>40.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>SEMBAWANG</th>
      <td>NaN</td>
      <td>51.0</td>
      <td>69.0</td>
      <td>1618.0</td>
      <td>1347.0</td>
      <td>638.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>SENGKANG</th>
      <td>NaN</td>
      <td>154.0</td>
      <td>552.0</td>
      <td>5483.0</td>
      <td>3919.0</td>
      <td>961.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>SERANGOON</th>
      <td>NaN</td>
      <td>19.0</td>
      <td>764.0</td>
      <td>1375.0</td>
      <td>524.0</td>
      <td>441.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>TAMPINES</th>
      <td>NaN</td>
      <td>36.0</td>
      <td>2417.0</td>
      <td>4248.0</td>
      <td>2739.0</td>
      <td>1052.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>TOA PAYOH</th>
      <td>NaN</td>
      <td>145.0</td>
      <td>2260.0</td>
      <td>1444.0</td>
      <td>847.0</td>
      <td>121.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>WOODLANDS</th>
      <td>NaN</td>
      <td>56.0</td>
      <td>1373.0</td>
      <td>5071.0</td>
      <td>3584.0</td>
      <td>1250.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>YISHUN</th>
      <td>NaN</td>
      <td>68.0</td>
      <td>3158.0</td>
      <td>4846.0</td>
      <td>1425.0</td>
      <td>510.0</td>
      <td>35.0</td>
    </tr>
  </tbody>
</table>
</div>


```python
pivot_town_flat_type.dtypes
```




    flat_type
    1 ROOM              float64
    2 ROOM              float64
    3 ROOM              float64
    4 ROOM              float64
    5 ROOM              float64
    EXECUTIVE           float64
    MULTI-GENERATION    float64
    dtype: object



```python
pivot_town_flat_type.plot.bar(stacked=True, figsize=(12,10))

plt.xlabel('Town')
plt.ylabel('Number of Transaction')
# plt.legend(pivot_town_flat_type['flat_type'])
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()
```

![png]({{ site.baseurl }}/Project%203_files/Project%203/Project%203_31_0.png){: .center-image }

```python
hdb.head()
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
      <th>id</th>
      <th>Tranc_YearMonth</th>
      <th>town</th>
      <th>flat_type</th>
      <th>block</th>
      <th>street_name</th>
      <th>storey_range</th>
      <th>floor_area_sqm</th>
      <th>flat_model</th>
      <th>lease_commence_date</th>
      <th>...</th>
      <th>pri_sch_affiliation</th>
      <th>pri_sch_latitude</th>
      <th>pri_sch_longitude</th>
      <th>sec_sch_nearest_dist</th>
      <th>sec_sch_name</th>
      <th>cutoff_point</th>
      <th>affiliation</th>
      <th>sec_sch_latitude</th>
      <th>sec_sch_longitude</th>
      <th>YearMonth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>88471</td>
      <td>2016-05-01</td>
      <td>KALLANG/WHAMPOA</td>
      <td>4 ROOM</td>
      <td>3B</td>
      <td>UPP BOON KENG RD</td>
      <td>10 TO 12</td>
      <td>90</td>
      <td>Model A</td>
      <td>2006</td>
      <td>...</td>
      <td>1</td>
      <td>1.317659</td>
      <td>103.882504</td>
      <td>1138.633422</td>
      <td>Geylang Methodist School</td>
      <td>224</td>
      <td>0</td>
      <td>1.317659</td>
      <td>103.882504</td>
      <td>2016-05</td>
    </tr>
    <tr>
      <th>1</th>
      <td>122598</td>
      <td>2012-07-01</td>
      <td>BISHAN</td>
      <td>5 ROOM</td>
      <td>153</td>
      <td>BISHAN ST 13</td>
      <td>07 TO 09</td>
      <td>130</td>
      <td>Improved</td>
      <td>1987</td>
      <td>...</td>
      <td>1</td>
      <td>1.349783</td>
      <td>103.854529</td>
      <td>447.894399</td>
      <td>Kuo Chuan Presbyterian Secondary School</td>
      <td>232</td>
      <td>0</td>
      <td>1.350110</td>
      <td>103.854892</td>
      <td>2012-07</td>
    </tr>
    <tr>
      <th>2</th>
      <td>170897</td>
      <td>2013-07-01</td>
      <td>BUKIT BATOK</td>
      <td>EXECUTIVE</td>
      <td>289B</td>
      <td>BT BATOK ST 25</td>
      <td>13 TO 15</td>
      <td>144</td>
      <td>Apartment</td>
      <td>1997</td>
      <td>...</td>
      <td>0</td>
      <td>1.345245</td>
      <td>103.756265</td>
      <td>180.074558</td>
      <td>Yusof Ishak Secondary School</td>
      <td>188</td>
      <td>0</td>
      <td>1.342334</td>
      <td>103.760013</td>
      <td>2013-07</td>
    </tr>
    <tr>
      <th>3</th>
      <td>86070</td>
      <td>2012-04-01</td>
      <td>BISHAN</td>
      <td>4 ROOM</td>
      <td>232</td>
      <td>BISHAN ST 22</td>
      <td>01 TO 05</td>
      <td>103</td>
      <td>Model A</td>
      <td>1992</td>
      <td>...</td>
      <td>1</td>
      <td>1.354789</td>
      <td>103.844934</td>
      <td>389.515528</td>
      <td>Catholic High School</td>
      <td>253</td>
      <td>1</td>
      <td>1.354789</td>
      <td>103.844934</td>
      <td>2012-04</td>
    </tr>
    <tr>
      <th>4</th>
      <td>153632</td>
      <td>2017-12-01</td>
      <td>YISHUN</td>
      <td>4 ROOM</td>
      <td>876</td>
      <td>YISHUN ST 81</td>
      <td>01 TO 03</td>
      <td>83</td>
      <td>Simplified</td>
      <td>1987</td>
      <td>...</td>
      <td>0</td>
      <td>1.416280</td>
      <td>103.838798</td>
      <td>312.025435</td>
      <td>Orchid Park Secondary School</td>
      <td>208</td>
      <td>0</td>
      <td>1.414888</td>
      <td>103.838335</td>
      <td>2017-12</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 82 columns</p>
</div>


```python
hdb['Year'] = hdb['Tranc_YearMonth'].dt.year
hdb['Month'] = hdb['Tranc_YearMonth'].dt.month
```

```python
pivot_yoy_month = pd.pivot_table(hdb, values='id', index='Year', columns='Month', aggfunc='count')

pivot_yoy_month
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
      <th>Month</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
    </tr>
    <tr>
      <th>Year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2012</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>1906.0</td>
      <td>1712.0</td>
      <td>1873.0</td>
      <td>1621.0</td>
      <td>1776.0</td>
      <td>1678.0</td>
      <td>1451.0</td>
      <td>1554.0</td>
      <td>1411.0</td>
      <td>1198.0</td>
    </tr>
    <tr>
      <th>2013</th>
      <td>1339.0</td>
      <td>714.0</td>
      <td>1082.0</td>
      <td>1369.0</td>
      <td>1262.0</td>
      <td>1066.0</td>
      <td>1192.0</td>
      <td>1136.0</td>
      <td>998.0</td>
      <td>1112.0</td>
      <td>997.0</td>
      <td>818.0</td>
    </tr>
    <tr>
      <th>2014</th>
      <td>891.0</td>
      <td>782.0</td>
      <td>1145.0</td>
      <td>1262.0</td>
      <td>1175.0</td>
      <td>987.0</td>
      <td>1106.0</td>
      <td>1086.0</td>
      <td>1179.0</td>
      <td>1246.0</td>
      <td>1077.0</td>
      <td>1049.0</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>1007.0</td>
      <td>922.0</td>
      <td>1071.0</td>
      <td>1319.0</td>
      <td>1281.0</td>
      <td>1369.0</td>
      <td>1240.0</td>
      <td>1162.0</td>
      <td>1208.0</td>
      <td>1407.0</td>
      <td>1189.0</td>
      <td>1136.0</td>
    </tr>
    <tr>
      <th>2016</th>
      <td>1030.0</td>
      <td>965.0</td>
      <td>1343.0</td>
      <td>1457.0</td>
      <td>1467.0</td>
      <td>1483.0</td>
      <td>1256.0</td>
      <td>1518.0</td>
      <td>1340.0</td>
      <td>1374.0</td>
      <td>1291.0</td>
      <td>1100.0</td>
    </tr>
    <tr>
      <th>2017</th>
      <td>971.0</td>
      <td>892.0</td>
      <td>1550.0</td>
      <td>1501.0</td>
      <td>1605.0</td>
      <td>1411.0</td>
      <td>1473.0</td>
      <td>1595.0</td>
      <td>1346.0</td>
      <td>1457.0</td>
      <td>1571.0</td>
      <td>1302.0</td>
    </tr>
    <tr>
      <th>2018</th>
      <td>888.0</td>
      <td>947.0</td>
      <td>1521.0</td>
      <td>1491.0</td>
      <td>1403.0</td>
      <td>1608.0</td>
      <td>2088.0</td>
      <td>1664.0</td>
      <td>1615.0</td>
      <td>1623.0</td>
      <td>1530.0</td>
      <td>1158.0</td>
    </tr>
    <tr>
      <th>2019</th>
      <td>1267.0</td>
      <td>1061.0</td>
      <td>1348.0</td>
      <td>1553.0</td>
      <td>1686.0</td>
      <td>1519.0</td>
      <td>1720.0</td>
      <td>1551.0</td>
      <td>1497.0</td>
      <td>1788.0</td>
      <td>1541.0</td>
      <td>1488.0</td>
    </tr>
    <tr>
      <th>2020</th>
      <td>1548.0</td>
      <td>1347.0</td>
      <td>1590.0</td>
      <td>339.0</td>
      <td>284.0</td>
      <td>2008.0</td>
      <td>1965.0</td>
      <td>1947.0</td>
      <td>2005.0</td>
      <td>1961.0</td>
      <td>1903.0</td>
      <td>2007.0</td>
    </tr>
    <tr>
      <th>2021</th>
      <td>2019.0</td>
      <td>1773.0</td>
      <td>1968.0</td>
      <td>1556.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


```python
pivot_month_yoy = pd.pivot_table(hdb, values='id', index='Month', columns='Year', aggfunc='count')

pivot_month_yoy
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
      <th>Year</th>
      <th>2012</th>
      <th>2013</th>
      <th>2014</th>
      <th>2015</th>
      <th>2016</th>
      <th>2017</th>
      <th>2018</th>
      <th>2019</th>
      <th>2020</th>
      <th>2021</th>
    </tr>
    <tr>
      <th>Month</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>1339.0</td>
      <td>891.0</td>
      <td>1007.0</td>
      <td>1030.0</td>
      <td>971.0</td>
      <td>888.0</td>
      <td>1267.0</td>
      <td>1548.0</td>
      <td>2019.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>714.0</td>
      <td>782.0</td>
      <td>922.0</td>
      <td>965.0</td>
      <td>892.0</td>
      <td>947.0</td>
      <td>1061.0</td>
      <td>1347.0</td>
      <td>1773.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1906.0</td>
      <td>1082.0</td>
      <td>1145.0</td>
      <td>1071.0</td>
      <td>1343.0</td>
      <td>1550.0</td>
      <td>1521.0</td>
      <td>1348.0</td>
      <td>1590.0</td>
      <td>1968.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1712.0</td>
      <td>1369.0</td>
      <td>1262.0</td>
      <td>1319.0</td>
      <td>1457.0</td>
      <td>1501.0</td>
      <td>1491.0</td>
      <td>1553.0</td>
      <td>339.0</td>
      <td>1556.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1873.0</td>
      <td>1262.0</td>
      <td>1175.0</td>
      <td>1281.0</td>
      <td>1467.0</td>
      <td>1605.0</td>
      <td>1403.0</td>
      <td>1686.0</td>
      <td>284.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1621.0</td>
      <td>1066.0</td>
      <td>987.0</td>
      <td>1369.0</td>
      <td>1483.0</td>
      <td>1411.0</td>
      <td>1608.0</td>
      <td>1519.0</td>
      <td>2008.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1776.0</td>
      <td>1192.0</td>
      <td>1106.0</td>
      <td>1240.0</td>
      <td>1256.0</td>
      <td>1473.0</td>
      <td>2088.0</td>
      <td>1720.0</td>
      <td>1965.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1678.0</td>
      <td>1136.0</td>
      <td>1086.0</td>
      <td>1162.0</td>
      <td>1518.0</td>
      <td>1595.0</td>
      <td>1664.0</td>
      <td>1551.0</td>
      <td>1947.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1451.0</td>
      <td>998.0</td>
      <td>1179.0</td>
      <td>1208.0</td>
      <td>1340.0</td>
      <td>1346.0</td>
      <td>1615.0</td>
      <td>1497.0</td>
      <td>2005.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1554.0</td>
      <td>1112.0</td>
      <td>1246.0</td>
      <td>1407.0</td>
      <td>1374.0</td>
      <td>1457.0</td>
      <td>1623.0</td>
      <td>1788.0</td>
      <td>1961.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1411.0</td>
      <td>997.0</td>
      <td>1077.0</td>
      <td>1189.0</td>
      <td>1291.0</td>
      <td>1571.0</td>
      <td>1530.0</td>
      <td>1541.0</td>
      <td>1903.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1198.0</td>
      <td>818.0</td>
      <td>1049.0</td>
      <td>1136.0</td>
      <td>1100.0</td>
      <td>1302.0</td>
      <td>1158.0</td>
      <td>1488.0</td>
      <td>2007.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


```python
pivot_month_yoy.plot(kind='line', figsize=(18,12))

plt.xlabel('Months')
plt.ylabel('Number of Transaction')
# plt.legend(pivot_town_flat_type['flat_type'])
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()
```

![png]({{ site.baseurl }}/Project%203_files/Project%203/Project%203_36_0.png){: .center-image }

```python
pivot_price_yoy = pd.pivot_table(hdb, values='resale_price', index='Month', columns='Year', aggfunc='mean')

pivot_price_yoy
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
      <th>Year</th>
      <th>2012</th>
      <th>2013</th>
      <th>2014</th>
      <th>2015</th>
      <th>2016</th>
      <th>2017</th>
      <th>2018</th>
      <th>2019</th>
      <th>2020</th>
      <th>2021</th>
    </tr>
    <tr>
      <th>Month</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>476925.673637</td>
      <td>453744.246914</td>
      <td>438961.289970</td>
      <td>428656.279612</td>
      <td>426957.967044</td>
      <td>445188.958333</td>
      <td>428342.973165</td>
      <td>432401.094961</td>
      <td>486111.352155</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>475333.641457</td>
      <td>449218.039642</td>
      <td>438794.725597</td>
      <td>438115.937824</td>
      <td>446806.514574</td>
      <td>441353.005280</td>
      <td>427591.078228</td>
      <td>440063.419451</td>
      <td>494181.464185</td>
    </tr>
    <tr>
      <th>3</th>
      <td>454145.729801</td>
      <td>476722.468577</td>
      <td>449666.100437</td>
      <td>428293.434174</td>
      <td>438794.813850</td>
      <td>444884.615484</td>
      <td>457970.339250</td>
      <td>431634.441395</td>
      <td>433784.606289</td>
      <td>491633.061992</td>
    </tr>
    <tr>
      <th>4</th>
      <td>453807.551402</td>
      <td>486783.079620</td>
      <td>451409.158479</td>
      <td>429188.147839</td>
      <td>439785.885381</td>
      <td>437858.369087</td>
      <td>449765.100604</td>
      <td>430785.482292</td>
      <td>440207.955752</td>
      <td>498339.220437</td>
    </tr>
    <tr>
      <th>5</th>
      <td>458472.759210</td>
      <td>484650.516640</td>
      <td>445621.650213</td>
      <td>438331.161593</td>
      <td>442041.775733</td>
      <td>447868.000000</td>
      <td>448584.528154</td>
      <td>435119.019573</td>
      <td>451906.746479</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>462182.895126</td>
      <td>475149.283302</td>
      <td>446766.175279</td>
      <td>430557.247626</td>
      <td>436315.323668</td>
      <td>451137.067328</td>
      <td>444689.392413</td>
      <td>431901.712969</td>
      <td>432812.303287</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>461550.139077</td>
      <td>479476.327181</td>
      <td>440766.037071</td>
      <td>428923.888710</td>
      <td>444579.864650</td>
      <td>438644.310930</td>
      <td>443301.020115</td>
      <td>435065.532558</td>
      <td>434323.720611</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>469132.099523</td>
      <td>477704.090669</td>
      <td>445924.458564</td>
      <td>438096.310671</td>
      <td>439914.311594</td>
      <td>445828.531661</td>
      <td>443177.671875</td>
      <td>434492.666667</td>
      <td>446025.978428</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>469140.201930</td>
      <td>469336.260521</td>
      <td>438462.303647</td>
      <td>438266.896523</td>
      <td>440511.769403</td>
      <td>450561.726597</td>
      <td>432941.099071</td>
      <td>431394.213093</td>
      <td>459372.252868</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>475683.658301</td>
      <td>468724.853417</td>
      <td>433264.296148</td>
      <td>429308.952381</td>
      <td>431126.913392</td>
      <td>444324.632807</td>
      <td>431171.556993</td>
      <td>432481.644295</td>
      <td>480547.735339</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>11</th>
      <td>474022.452870</td>
      <td>465696.439318</td>
      <td>432440.732591</td>
      <td>437583.241379</td>
      <td>444708.865995</td>
      <td>444079.732018</td>
      <td>429890.762745</td>
      <td>435795.650227</td>
      <td>473986.078823</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12</th>
      <td>475881.370618</td>
      <td>467988.195599</td>
      <td>435547.962822</td>
      <td>438236.742077</td>
      <td>435875.645455</td>
      <td>451410.759601</td>
      <td>424109.922280</td>
      <td>430987.517473</td>
      <td>481052.772795</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


```python
pivot_price_yoy.plot(kind='line', figsize=(18,12))

plt.xlabel('Months')
plt.ylabel('Average Resale Price')
# plt.legend(pivot_town_flat_type['flat_type'])
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()
```

![png]({{ site.baseurl }}/Project%203_files/Project%203/Project%203_38_0.png){: .center-image }

```python
hdb.set_index('Year', inplace=True)
```

```python
from statsmodels.tsa.seasonal import seasonal_decompose

monthly_avg_price = hdb.groupby('YearMonth')['resale_price'].mean()
```

```python
monthly_avg_price.index = monthly_avg_price.index.to_timestamp()
```

```python
decomposition_hdb = seasonal_decompose(monthly_avg_price, model='additive', period=12)
```

```python
plt.figure(figsize=(24, 16))
decomposition_hdb.plot()
plt.tight_layout()
plt.suptitle('Decomposition of Monthly HDB Resale Prices', fontsize=12, y=1.02)
plt.show()
```


    <Figure size 2400x1600 with 0 Axes>


![png]({{ site.baseurl }}/Project%203_files/Project%203/Project%203_43_1.png){: .center-image }

```python
hdbpredict = hdb.sort_values('Tranc_YearMonth')
```

```python
hdbpredict.head(10)
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
      <th>id</th>
      <th>Tranc_YearMonth</th>
      <th>town</th>
      <th>flat_type</th>
      <th>block</th>
      <th>street_name</th>
      <th>storey_range</th>
      <th>floor_area_sqm</th>
      <th>flat_model</th>
      <th>lease_commence_date</th>
      <th>...</th>
      <th>pri_sch_latitude</th>
      <th>pri_sch_longitude</th>
      <th>sec_sch_nearest_dist</th>
      <th>sec_sch_name</th>
      <th>cutoff_point</th>
      <th>affiliation</th>
      <th>sec_sch_latitude</th>
      <th>sec_sch_longitude</th>
      <th>YearMonth</th>
      <th>Month</th>
    </tr>
    <tr>
      <th>Year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2012</th>
      <td>31310</td>
      <td>2012-03-01</td>
      <td>BISHAN</td>
      <td>4 ROOM</td>
      <td>171</td>
      <td>BISHAN ST 13</td>
      <td>06 TO 10</td>
      <td>84</td>
      <td>Simplified</td>
      <td>1988</td>
      <td>...</td>
      <td>1.349783</td>
      <td>103.854529</td>
      <td>392.039131</td>
      <td>Kuo Chuan Presbyterian Secondary School</td>
      <td>232</td>
      <td>0</td>
      <td>1.350110</td>
      <td>103.854892</td>
      <td>2012-03</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2012</th>
      <td>170001</td>
      <td>2012-03-01</td>
      <td>JURONG EAST</td>
      <td>5 ROOM</td>
      <td>403</td>
      <td>PANDAN GDNS</td>
      <td>11 TO 15</td>
      <td>114</td>
      <td>Standard</td>
      <td>1979</td>
      <td>...</td>
      <td>1.312622</td>
      <td>103.757030</td>
      <td>425.559463</td>
      <td>Commonwealth Secondary School</td>
      <td>237</td>
      <td>0</td>
      <td>1.319128</td>
      <td>103.745756</td>
      <td>2012-03</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2012</th>
      <td>116614</td>
      <td>2012-03-01</td>
      <td>JURONG WEST</td>
      <td>4 ROOM</td>
      <td>538</td>
      <td>JURONG WEST AVE 1</td>
      <td>01 TO 05</td>
      <td>104</td>
      <td>Model A</td>
      <td>1984</td>
      <td>...</td>
      <td>1.346844</td>
      <td>103.719010</td>
      <td>743.150663</td>
      <td>Hua Yi Secondary School</td>
      <td>221</td>
      <td>0</td>
      <td>1.352356</td>
      <td>103.721727</td>
      <td>2012-03</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2012</th>
      <td>9397</td>
      <td>2012-03-01</td>
      <td>BUKIT MERAH</td>
      <td>3 ROOM</td>
      <td>107</td>
      <td>JLN BT MERAH</td>
      <td>06 TO 10</td>
      <td>63</td>
      <td>Standard</td>
      <td>1970</td>
      <td>...</td>
      <td>1.276029</td>
      <td>103.822344</td>
      <td>501.268948</td>
      <td>CHIJ Saint Theresa's Convent</td>
      <td>235</td>
      <td>0</td>
      <td>1.276029</td>
      <td>103.822344</td>
      <td>2012-03</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2012</th>
      <td>153574</td>
      <td>2012-03-01</td>
      <td>SENGKANG</td>
      <td>5 ROOM</td>
      <td>197</td>
      <td>RIVERVALE DR</td>
      <td>01 TO 05</td>
      <td>111</td>
      <td>Improved</td>
      <td>2001</td>
      <td>...</td>
      <td>1.393574</td>
      <td>103.904718</td>
      <td>143.024895</td>
      <td>CHIJ Saint Joseph's Convent</td>
      <td>232</td>
      <td>0</td>
      <td>1.391781</td>
      <td>103.902577</td>
      <td>2012-03</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2012</th>
      <td>15052</td>
      <td>2012-03-01</td>
      <td>TOA PAYOH</td>
      <td>3 ROOM</td>
      <td>85A</td>
      <td>LOR 4 TOA PAYOH</td>
      <td>06 TO 10</td>
      <td>68</td>
      <td>Improved</td>
      <td>1972</td>
      <td>...</td>
      <td>1.337408</td>
      <td>103.847761</td>
      <td>679.831348</td>
      <td>Beatty Secondary School</td>
      <td>211</td>
      <td>0</td>
      <td>1.341791</td>
      <td>103.852065</td>
      <td>2012-03</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2012</th>
      <td>164684</td>
      <td>2012-03-01</td>
      <td>PUNGGOL</td>
      <td>5 ROOM</td>
      <td>101C</td>
      <td>PUNGGOL FIELD</td>
      <td>01 TO 05</td>
      <td>109</td>
      <td>Improved</td>
      <td>2002</td>
      <td>...</td>
      <td>1.400298</td>
      <td>103.907431</td>
      <td>530.792429</td>
      <td>Punggol Secondary School</td>
      <td>194</td>
      <td>0</td>
      <td>1.402126</td>
      <td>103.909119</td>
      <td>2012-03</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2012</th>
      <td>77939</td>
      <td>2012-03-01</td>
      <td>BEDOK</td>
      <td>4 ROOM</td>
      <td>140</td>
      <td>BEDOK RESERVOIR RD</td>
      <td>01 TO 05</td>
      <td>84</td>
      <td>Simplified</td>
      <td>1986</td>
      <td>...</td>
      <td>1.330913</td>
      <td>103.911349</td>
      <td>1195.142944</td>
      <td>Ping Yi Secondary School</td>
      <td>189</td>
      <td>0</td>
      <td>1.327140</td>
      <td>103.920836</td>
      <td>2012-03</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2012</th>
      <td>137444</td>
      <td>2012-03-01</td>
      <td>ANG MO KIO</td>
      <td>5 ROOM</td>
      <td>353</td>
      <td>ANG MO KIO ST 32</td>
      <td>21 TO 25</td>
      <td>110</td>
      <td>Improved</td>
      <td>2001</td>
      <td>...</td>
      <td>1.365201</td>
      <td>103.851032</td>
      <td>366.418427</td>
      <td>Deyi Secondary School</td>
      <td>215</td>
      <td>0</td>
      <td>1.367347</td>
      <td>103.851936</td>
      <td>2012-03</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2012</th>
      <td>95633</td>
      <td>2012-03-01</td>
      <td>JURONG WEST</td>
      <td>4 ROOM</td>
      <td>655B</td>
      <td>JURONG WEST ST 61</td>
      <td>11 TO 15</td>
      <td>85</td>
      <td>Model A2</td>
      <td>2002</td>
      <td>...</td>
      <td>1.336643</td>
      <td>103.699683</td>
      <td>510.316098</td>
      <td>Jurong West Secondary School</td>
      <td>199</td>
      <td>0</td>
      <td>1.335256</td>
      <td>103.702098</td>
      <td>2012-03</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 83 columns</p>
</div>


```python
hdbpredict.set_index('Tranc_YearMonth', inplace=True)
```

```python
monthly_average = hdbpredict['resale_price'].resample('M').mean()
```

    /var/folders/hl/51hk561j7s7_jm62kls87wv80000gn/T/ipykernel_12846/3058426924.py:1: FutureWarning:
    
    'M' is deprecated and will be removed in a future version, please use 'ME' instead.
    

```python
train_size = int(len(monthly_average) * 2/3)
train, test = monthly_average[:train_size], monthly_average[train_size:]
```

```python
from statsmodels.tsa.arima.model import ARIMA
```

```python
fig, ax = plt.subplots(figsize=(8, 4))

train.plot(ax=ax, label="train")
test.plot(ax=ax, label="test")
ax.legend()

plt.show()
```

![png]({{ site.baseurl }}/Project%203_files/Project%203/Project%203_50_0.png){: .center-image }

```python
monthly_average[train_size:]
```




    Tranc_YearMonth
    2018-04-30    449765.100604
    2018-05-31    448584.528154
    2018-06-30    444689.392413
    2018-07-31    443301.020115
    2018-08-31    443177.671875
    2018-09-30    432941.099071
    2018-10-31    431171.556993
    2018-11-30    429890.762745
    2018-12-31    424109.922280
    2019-01-31    428342.973165
    2019-02-28    427591.078228
    2019-03-31    431634.441395
    2019-04-30    430785.482292
    2019-05-31    435119.019573
    2019-06-30    431901.712969
    2019-07-31    435065.532558
    2019-08-31    434492.666667
    2019-09-30    431394.213093
    2019-10-31    432481.644295
    2019-11-30    435795.650227
    2019-12-31    430987.517473
    2020-01-31    432401.094961
    2020-02-29    440063.419451
    2020-03-31    433784.606289
    2020-04-30    440207.955752
    2020-05-31    451906.746479
    2020-06-30    432812.303287
    2020-07-31    434323.720611
    2020-08-31    446025.978428
    2020-09-30    459372.252868
    2020-10-31    480547.735339
    2020-11-30    473986.078823
    2020-12-31    481052.772795
    2021-01-31    486111.352155
    2021-02-28    494181.464185
    2021-03-31    491633.061992
    2021-04-30    498339.220437
    Freq: ME, Name: resale_price, dtype: float64



```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(train, order=(2, 1, 0)).fit()

y_pred = model.predict(start="2018-04-30", end="2021-04-30")
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[1], line 3
          1 from statsmodels.tsa.arima.model import ARIMA
    ----> 3 model = ARIMA(train, order=(2, 1, 0)).fit()
          5 y_pred = model.predict(start="2018-04-30", end="2021-04-30")
    

    NameError: name 'train' is not defined


```python
fig, ax = plt.subplots(figsize=(8, 6))

train.plot(ax=ax, label="train")
test.plot(ax=ax, label="test")
y_pred.plot(ax=ax, label="ARIMA forecast")
ax.legend()

plt.show()
```

```python

```
